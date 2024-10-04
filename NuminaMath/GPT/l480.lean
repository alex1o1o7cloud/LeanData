import Mathlib

namespace conclusion_6_l480_480188

def f (x: ℝ) : ℤ := ⌊ Real.log2 ((2^x + 1) / 9) ⌋

theorem conclusion_6 : ∀ x, 12 < x ∧ x < 13 → f(x) = 9 := 
by {
  assume x,
  assume h : 12 < x ∧ x < 13,
  sorry
}

end conclusion_6_l480_480188


namespace number_of_periods_l480_480481

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l480_480481


namespace no_solution_exists_l480_480729

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l480_480729


namespace inverse_g_neg_two_fifths_eq_zero_l480_480639

def g (x : ℝ) : ℝ := (x^3 - 2) / 5

theorem inverse_g_neg_two_fifths_eq_zero : g⁻¹ (-2 / 5) = 0 := by 
  sorry

end inverse_g_neg_two_fifths_eq_zero_l480_480639


namespace probability_of_exactly_five_correct_letters_is_zero_l480_480788

theorem probability_of_exactly_five_correct_letters_is_zero :
  ∀ (envelopes : Fin 6 → ℕ), (∃! (f : Fin 6 → Fin 6), Bijective f ∧ ∀ i, envelopes (f i) = envelopes i) → 
  (Probability (exactly_five_correct ∈ permutations (Fin 6)) = 0) :=
by
  sorry

noncomputable def exactly_five_correct (σ : Fin 6 → Fin 6) : Prop :=
  (∃ (i : Fin 6), ∀ j, j ≠ i → σ j = j) ∧ ∃ (k : Fin 6), k ≠ i ∧ σ k ≠ k

noncomputable def permutations (α : Type) := List.permutations (List.finRange 6)

end probability_of_exactly_five_correct_letters_is_zero_l480_480788


namespace record_expenditure_l480_480643

def income (amount : ℤ) := amount > 0
def expenditure (amount : ℤ) := -amount

theorem record_expenditure : 
  (income 100 = true) ∧ (100 = +100) ∧ (income (expenditure 80) = false) → (expenditure 80 = -80) := 
by sorry

end record_expenditure_l480_480643


namespace y_coord_equidistant_l480_480008

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l480_480008


namespace simplify_expression_l480_480963

theorem simplify_expression : 3 * Real.sqrt 2 + abs (1 - Real.sqrt 2) + Real.cbrt 8 = 4 * Real.sqrt 2 + 1 := 
by
  sorry

end simplify_expression_l480_480963


namespace problem_l480_480199

-- Define the function f
def f (x m : ℝ) : ℝ := 2^(abs (x - m)) - 1

-- Define the values a, b, and c based on the given conditions
def a (m : ℝ) : ℝ := f (Real.log (3) / Real.log (0.5)) m
def b (m : ℝ) : ℝ := f (Real.log 5 / Real.log 2) m
def c (m : ℝ) : ℝ := f (2 * m) m

-- Define the fact that the function is even, implying m = 0
axiom evens_even (m : ℝ) (x : ℝ) : f x m = f (-x) m → m = 0

-- Formalize the mathematically equivalent proof problem
theorem problem : (evens_even m 0) → c m < a m ∧ a m < b m :=
by
  intros evn
  rw evens_even at evn
  subst evn
  sorry

end problem_l480_480199


namespace find_numbers_sum_eq_S_product_eq_P_l480_480429

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480429


namespace completing_the_square_result_l480_480032

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l480_480032


namespace longest_side_of_triangle_l480_480252

noncomputable def longest_side (A B : ℝ) (a : ℝ) (H1 : Real.tan A = 1 / 4) (H2 : Real.tan B = 3 / 5) (H3 : a = Real.sqrt 2) : ℝ :=
  let C := π - (A + B) in
  let c := a * (Real.sin C / Real.sin A) in
  c

theorem longest_side_of_triangle
  (A B : ℝ) (a c : ℝ)
  (H1 : Real.tan A = 1 / 4)
  (H2 : Real.tan B = 3 / 5)
  (H3 : a = Real.sqrt 2)
  (hc : c = longest_side A B a H1 H2 H3) :
  c = Real.sqrt 17 := by
  sorry

end longest_side_of_triangle_l480_480252


namespace circle_diameter_and_circumference_l480_480478

theorem circle_diameter_and_circumference (r : ℝ) (π : ℝ) (A : ℝ) (D : ℝ) (C : ℝ) 
    (hA : A = 16 * π) 
    (hArea_eq : A = π * r^2) 
    (hDiam : D = 2 * r) 
    (hCirc : C = 2 * π * r) : 
    D = 8 ∧ C = 8 * π :=
by
  have hr : r^2 = 16, from (by rw [hArea_eq, hA]; field_simp),
  have r_eq : r = 4, from by
    rw [←sqrt_inv_eq_iff_sq_eq, sqrt_eq_iff_sq_eq],
    exact_mod_cast hr,
  
  have hD_eq : D = 2 * 4, from by rw [hDiam]; congr; exact r_eq,
  have hC_eq : C = 2 * π * 4, from by rw [hCirc]; congr; exact r_eq,
  
  split,
  { rw [hD_eq]; norm_num },
  { rw [hC_eq]; norm_num }

end circle_diameter_and_circumference_l480_480478


namespace seat_10_people_round_table_l480_480664

theorem seat_10_people_round_table : 
  let linear_arrangements := Nat.factorial 10,
      rotations := 10
  in linear_arrangements / rotations = 362880 := 
by
  let linear_arrangements := Nat.factorial 10
  let rotations := 10
  have : linear_arrangements = 3628800 := rfl
  simp [this, rotations]
  sorry

end seat_10_people_round_table_l480_480664


namespace determine_integers_from_sums_l480_480341

theorem determine_integers_from_sums (n : ℕ) :
    ∀ (sums : finset ℤ), 
    (sums.card = 2^n - 1) → 
    (0 ∉ sums) → 
    ∃! (numbers : finset ℤ), 
    (∀ sum ∈ sums, ∃ subset ⊆ numbers, sum = subset.sum) :=
sorry

end determine_integers_from_sums_l480_480341


namespace same_function_option_C_l480_480042

-- Definitions of functions given in Option C
def f (x : ℝ) : ℝ := (x + 3)^2
def g (x : ℝ) : ℝ := abs (x + 3)

-- Theorem stating that for all real numbers x, √((x + 3)^2) is equal to |x + 3|
theorem same_function_option_C (x : ℝ) : sqrt ((x + 3)^2) = abs (x + 3) := by sorry

end same_function_option_C_l480_480042


namespace paul_packed_total_toys_l480_480325

def small_box_small_toys : ℕ := 8
def medium_box_medium_toys : ℕ := 12
def large_box_large_toys : ℕ := 7
def large_box_small_toys : ℕ := 3
def small_box_medium_toys : ℕ := 5

def small_box : ℕ := small_box_small_toys + small_box_medium_toys
def medium_box : ℕ := medium_box_medium_toys
def large_box : ℕ := large_box_large_toys + large_box_small_toys

def total_toys : ℕ := small_box + medium_box + large_box

theorem paul_packed_total_toys : total_toys = 35 :=
by sorry

end paul_packed_total_toys_l480_480325


namespace lcm_18_27_l480_480013

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l480_480013


namespace sine_angle_GAC_l480_480466

noncomputable def angle_GAC_sin : ℝ :=
  let
    A : ℝ × ℝ × ℝ := (0,0,0),
    C : ℝ × ℝ × ℝ := (1,2,0),
    G : ℝ × ℝ × ℝ := (1,2,3),
    AC : ℝ × ℝ × ℝ := (1-0, 2-0, 0-0),
    AG : ℝ × ℝ × ℝ := (1-0, 2-0, 3-0)
  in
  real.sqrt ((AC.1)^2 + (AC.2)^2 + (AC.3)^2) / real.sqrt ((AG.1)^2 + (AG.2)^2 + (AG.3)^2)

theorem sine_angle_GAC : angle_GAC_sin = (3 : ℝ) / (real.sqrt 14) :=
  sorry

end sine_angle_GAC_l480_480466


namespace numbers_pairs_sum_prod_l480_480427

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480427


namespace expression_identity_l480_480408

theorem expression_identity (a : ℤ) (h : a = 102) : 
  a^4 - 4 * a^3 + 6 * a^2 - 4 * a + 1 = 104060401 :=
by {
  rw h,
  calc 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 101^4 : by sorry
  ... = 104060401 : by sorry
}

end expression_identity_l480_480408


namespace ninety_one_square_friendly_unique_square_friendly_l480_480513

-- Given conditions
def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18 * m + c = n^2

-- Part (a)
theorem ninety_one_square_friendly : square_friendly 81 :=
sorry

-- Part (b)
theorem unique_square_friendly (c c' : ℤ) (h_c : square_friendly c) (h_c' : square_friendly c') : c = c' :=
sorry

end ninety_one_square_friendly_unique_square_friendly_l480_480513


namespace inequality_proof_l480_480692

theorem inequality_proof (n : ℕ) (x : ℕ → ℝ) (h : ∀ i, 1 ≤ x i) :
  (∑ i in finset.range n, 1 / (x i + 1)) ≥ n / (1 + (∏ i in finset.range n, x i)^(1 / n)) :=
sorry

end inequality_proof_l480_480692


namespace cos_strictly_increasing_intervals_correct_l480_480975

noncomputable def cos_strictly_increasing_intervals : set ℝ :=
  {I : set ℝ | ∃ (k : ℤ), I = set.Ioo (-3 * Real.pi / 8 + k * Real.pi) (Real.pi / 8 + k * Real.pi)}

theorem cos_strictly_increasing_intervals_correct :
  ∀ x : ℝ, strict_mono (λ x, Real.cos (Real.pi / 4 - 2 * x)) ↔ 
  ∃ k : ℤ, -3 * Real.pi / 8 + k * Real.pi < x ∧ x < Real.pi / 8 + k * Real.pi :=
by sorry

end cos_strictly_increasing_intervals_correct_l480_480975


namespace tangent_lines_at_M_find_a_l480_480604

-- Define the given circle
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the tangent line to the circle at point M
def tangent_line1 (x y : ℝ) : Prop := x = 3
def tangent_line2 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0

-- Define the line ax - y + 4 = 0
def line (a x y : ℝ) : Prop := a*x - y + 4 = 0

-- Define the chord length condition
def chord_length_condition (a : ℝ) : Prop :=
  (|a + 2| / (real.sqrt (a^2 + 1)))^2 + (2 * real.sqrt 3 / 2)^2 = 4

-- Prove the equations of the tangent lines at point M
theorem tangent_lines_at_M :
  (∀ x y : ℝ, circle x y → tangent_line1 x y) ∧
  (∀ x y : ℝ, circle x y → tangent_line2 x y) :=
sorry

-- Prove the value of a under given conditions
theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle x y → line a x y → chord_length_condition a) →
  a = -3/4 :=
sorry

end tangent_lines_at_M_find_a_l480_480604


namespace feuerbach_iff_inscribed_l480_480715

-- Definitions for conditions
structure OrthocentricTetrahedron (V : Type) [MetricSpace V] :=
(faces : Fin 4 → Triangle V)
(face_property : ∀ i, is_orthocentric (faces i))

-- Definition of geometrical objects
def FeuerbachSphere (T : OrthocentricTetrahedron V) : Sphere V := sorry
def InscribedSphere (T : Tetrahedron V) : Sphere V := sorry

-- Statement of the theorem
theorem feuerbach_iff_inscribed (T : OrthocentricTetrahedron V) :
  FeuerbachSphere T = InscribedSphere T.to_tetrahedron ↔ is_regular T.to_tetrahedron := sorry

end feuerbach_iff_inscribed_l480_480715


namespace math_problem_l480_480611

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ :=
  sorry  -- definition for calculating the smallest positive period

noncomputable def transform_and_translate (f : ℝ → ℝ) (m : ℝ) : ℝ → ℝ :=
  λ x, f (x + m)

noncomputable def find_range (f : ℝ → ℝ) (a b : ℝ) : set ℝ :=
  {y | ∃ x ∈ set.Icc a b, y = f x}

theorem math_problem
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = sin (2 * x) + sin (2 * x - π / 3))
  (hgt : ∀ m > 0, transform_and_translate f m (π / 8))
  : smallest_positive_period f = π ∧
    ∃ m, m = 5 * π / 12 ∧
         ∀ x, find_range (transform_and_translate f m) 0 (π / 4) = {y | -√3 / 2 ≤ y ∧ y ≤ 3 / 2} :=
sorry

end math_problem_l480_480611


namespace graph_passes_through_point_l480_480750

theorem graph_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : ∃ y, y = a^(2 - 2) + 2 ∧  y = 3 :=
by
  have h₃ : 2 - 2 = 0 := by norm_num
  have h₄ : a^0 = 1 := by exact real.rpow_zero a
  use (a^0 + 2)
  rw [h₃, h₄]
  norm_num
  sorry

end graph_passes_through_point_l480_480750


namespace find_nonnegative_solutions_l480_480141

theorem find_nonnegative_solutions :
  ∀ (x y z : ℕ), 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by
  sorry

end find_nonnegative_solutions_l480_480141


namespace sum_of_first_fifteen_multiples_of_17_l480_480831

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480831


namespace cannot_determine_students_answered_both_correctly_l480_480314

-- Definitions based on the given conditions
def students_enrolled : ℕ := 25
def students_answered_q1_correctly : ℕ := 22
def students_not_taken_test : ℕ := 3
def some_students_answered_q2_correctly : Prop := -- definition stating that there's an undefined number of students that answered question 2 correctly
  ∃ n : ℕ, (n ≤ students_enrolled) ∧ n > 0

-- Statement for the proof problem
theorem cannot_determine_students_answered_both_correctly :
  ∃ n, (n ≤ students_answered_q1_correctly) ∧ n > 0 → false :=
by sorry

end cannot_determine_students_answered_both_correctly_l480_480314


namespace sum_first_fifteen_multiples_seventeen_l480_480824

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480824


namespace find_original_number_l480_480017

theorem find_original_number (x : ℤ) (h : (x + 19) % 25 = 0) : x = 6 :=
sorry

end find_original_number_l480_480017


namespace find_numbers_l480_480455

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480455


namespace unique_solution_3_pow_x_minus_2_pow_y_eq_7_l480_480558

theorem unique_solution_3_pow_x_minus_2_pow_y_eq_7 :
  ∀ x y : ℕ, (1 ≤ x) → (1 ≤ y) → (3 ^ x - 2 ^ y = 7) → (x = 2 ∧ y = 1) :=
by
  intros x y hx hy hxy
  sorry

end unique_solution_3_pow_x_minus_2_pow_y_eq_7_l480_480558


namespace prove_RoseHasMoney_l480_480721
noncomputable def RoseHasMoney : Prop :=
  let cost_of_paintbrush := 2.40
  let cost_of_paints := 9.20
  let cost_of_easel := 6.50
  let total_cost := cost_of_paintbrush + cost_of_paints + cost_of_easel
  let additional_money_needed := 11
  let money_rose_has := total_cost - additional_money_needed
  money_rose_has = 7.10

theorem prove_RoseHasMoney : RoseHasMoney :=
  sorry

end prove_RoseHasMoney_l480_480721


namespace coordinates_of_B_l480_480197

theorem coordinates_of_B (A : ℝ × ℝ) (A_x A_y : ℝ) (hA : A = (-1, 2)) 
  (length_AB : ℝ) (h_length_AB : length_AB = 4) (parallel_to_x : ∀ p₁ p₂ : ℝ × ℝ, p₁.2 = p₂.2): 
  ∃ B : ℝ × ℝ, (B = (3, 2) ∨ B = (-5, 2)) ∧ A.2 = B.2 ∧ abs(B.1 - A.1) = length_AB :=
by {
  -- This is the statement to be proved.
  sorry
}

end coordinates_of_B_l480_480197


namespace S_100_eq_2600_l480_480668

noncomputable def a_n : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := 1 - (-1 : ℕ) ^ n + a_n n

theorem S_100_eq_2600 : 
  let S_100 := (∑ i in (range 100), a_n i)
  in S_100 = 2600 :=
by
  let a_n : ℕ → ℕ
  | 0 := 1
  | 1 := 2
  | (n+2) := 1 - (-1 : ℕ) ^ n + a_n n
  let S_100 := (∑ i in (range 100), a_n i)
  have h_eq : S_100 = 2600 := sorry
  exact h_eq

end S_100_eq_2600_l480_480668


namespace hyperbola_asymptotes_l480_480214

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (ecc : (c : ℝ) = a * √3) :
  ((b / a) = √2) → (asymptotes : ∀ x y : ℝ, x ± (√2) * y = 0) := 
begin
  sorry
end

end hyperbola_asymptotes_l480_480214


namespace chemist_sons_ages_l480_480469

theorem chemist_sons_ages 
    (a b c w : ℕ)
    (h1 : a * b * c = 36)
    (h2 : a + b + c = w)
    (h3 : ∃! x, x = max a (max b c)) :
    (a = 2 ∧ b = 2 ∧ c = 9) ∨ 
    (a = 2 ∧ b = 9 ∧ c = 2) ∨ 
    (a = 9 ∧ b = 2 ∧ c = 2) :=
  sorry

end chemist_sons_ages_l480_480469


namespace power_equivalence_l480_480633

theorem power_equivalence (y : ℝ) (h : 5^(3 * y) = 625) : 5^(3 * y - 2) = 25 :=
by
  sorry

end power_equivalence_l480_480633


namespace project_presentation_periods_l480_480482

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l480_480482


namespace finding_p_q_r_l480_480343

noncomputable def BF_solution (EF angle_EOF AB : ℝ) (p q r : ℕ) : Prop :=
  let r_square_free := ∀ (k : ℕ), k ^ 2 ∣ r → k = 1
  ∧ EF = 400 
  ∧ angle_EOF = 45
  ∧ AB = 900 
  ∧ p + q * (real.sqrt r) * 1 = p + q * sqrt r 
  ∧ true

theorem finding_p_q_r :
  ∃ (p q r : ℕ), (BF_solution 400 45 900 p q r) ∧ (p + q + r = 307) :=
sorry

end finding_p_q_r_l480_480343


namespace range_of_a_opposite_sides_l480_480062

theorem range_of_a_opposite_sides (a : ℝ) : 
  let expr1 := 3 * 3 - 2 * 1 + a,
      expr2 := 3 * (-4) - 2 * 6 + a in
  (expr1 * expr2 < 0) → (-7 < a ∧ a < 24) :=
by
  intros a h
  let expr1 := 3 * 3 - 2 * 1 + a
  let expr2 := 3 * (-4) - 2 * 6 + a
  sorry

end range_of_a_opposite_sides_l480_480062


namespace ab_plus_cd_l480_480237

variables (a b c d : ℝ)

axiom h1 : a + b + c = 1
axiom h2 : a + b + d = 6
axiom h3 : a + c + d = 15
axiom h4 : b + c + d = 10

theorem ab_plus_cd : a * b + c * d = 45.33333333333333 := 
by 
  sorry

end ab_plus_cd_l480_480237


namespace first_player_wins_l480_480322

-- Define the conditions in Lean
def piles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def game_condition := 
  ∀ (turns : ℕ) (nuts_taken : fin turns → ℕ) (nuts_left : ℕ),
  (∀ t, nuts_taken t = 1) →
  (nuts_left = 3) →
  (if nuts_left = 3 ∧ ∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ List.nth piles i = some 1 ∧ List.nth piles j = some 1 ∧ List.nth piles k = some 1
   then turns % 2 = 1
   else turns % 2 = 0)

-- The theorem stating the first player can guarantee a win
theorem first_player_wins : (∃ turns nuts_taken, game_condition turns nuts_taken 3) :=
sorry

end first_player_wins_l480_480322


namespace minimize_side_length_of_triangle_l480_480669

-- Define a triangle with sides a, b, and c and angle C
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ) -- angle C in radians
  (area : ℝ) -- area of the triangle

-- Define the conditions for the problem
def conditions (T : Triangle) : Prop :=
  T.area > 0 ∧ T.C > 0 ∧ T.C < Real.pi

-- Define the desired result
def min_side_length (T : Triangle) : Prop :=
  T.a = T.b ∧ T.a = Real.sqrt ((2 * T.area) / Real.sin T.C)

-- The theorem to be proven
theorem minimize_side_length_of_triangle (T : Triangle) (h : conditions T) : min_side_length T :=
  sorry

end minimize_side_length_of_triangle_l480_480669


namespace hoseok_more_paper_than_minyoung_l480_480704

theorem hoseok_more_paper_than_minyoung : 
  ∀ (initial : ℕ) (minyoung_bought : ℕ) (hoseok_bought : ℕ), 
  initial = 150 →
  minyoung_bought = 32 →
  hoseok_bought = 49 →
  (initial + hoseok_bought) - (initial + minyoung_bought) = 17 :=
by
  intros initial minyoung_bought hoseok_bought h_initial h_min h_hos
  sorry

end hoseok_more_paper_than_minyoung_l480_480704


namespace find_m_l480_480681

-- Conditions given in the problem
variables {x m : ℝ}

def G (x m : ℝ) : ℝ := (8 * x ^ 2 + 20 * x + 5 * m) / 8

-- Statement to prove
theorem find_m (h : ∃ (f : ℝ → ℝ), ∀ x, G x m = (f x) ^ 2) : m = 5 := 
by
  sorry

end find_m_l480_480681


namespace seat_10_people_round_table_l480_480665

theorem seat_10_people_round_table : 
  let linear_arrangements := Nat.factorial 10,
      rotations := 10
  in linear_arrangements / rotations = 362880 := 
by
  let linear_arrangements := Nat.factorial 10
  let rotations := 10
  have : linear_arrangements = 3628800 := rfl
  simp [this, rotations]
  sorry

end seat_10_people_round_table_l480_480665


namespace rectangle_to_rhombus_l480_480120

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ D.2 = C.2 ∧ C.1 = B.1 ∧ B.2 = A.2

def is_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) ≠ 0

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

theorem rectangle_to_rhombus (A B C D : ℝ × ℝ) (h1 : is_rectangle A B C D) :
  ∃ X Y Z W : ℝ × ℝ, is_triangle A B C ∧ is_triangle A D C ∧ is_rhombus X Y Z W :=
by
  sorry

end rectangle_to_rhombus_l480_480120


namespace simple_interest_rate_l480_480866

theorem simple_interest_rate (P : ℝ) (T : ℝ) (r : ℝ) (h1 : T = 10) (h2 : (3 / 5) * P = (P * r * T) / 100) : r = 6 := by
  sorry

end simple_interest_rate_l480_480866


namespace integer_solutions_l480_480151

def numberOfIntegerSolutions : ℕ :=
  14520

theorem integer_solutions : 
  (1 + ⌊120 * n / 121⌋ = ⌈119 * n / 120⌉) ∃ n, n < 14520 :=
by sorry

end integer_solutions_l480_480151


namespace find_numbers_l480_480444

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480444


namespace choose_six_with_consecutive_l480_480716

theorem choose_six_with_consecutive (n : ℕ) (h₁ : n = 49) (h₂ : 6 ≤ n) :
  (∑ k in finset.Icc 1 n, k) = (nat.choose 49 6) - (nat.choose 44 6) :=
by sorry

end choose_six_with_consecutive_l480_480716


namespace div_eq_eight_fifths_l480_480234

theorem div_eq_eight_fifths (a b : ℚ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 :=
by
  sorry

end div_eq_eight_fifths_l480_480234


namespace matching_pair_probability_correct_l480_480344

-- Define the basic assumptions (conditions)
def black_pairs : Nat := 7
def brown_pairs : Nat := 4
def gray_pairs : Nat := 3
def red_pairs : Nat := 2

def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs + red_pairs
def total_shoes : Nat := 2 * total_pairs

-- The probability calculation will be shown as the final proof requirement
def matching_color_probability : Rat :=  (14 * 7 + 8 * 4 + 6 * 3 + 4 * 2 : Int) / (32 * 31 : Int)

-- The target statement to be proven
theorem matching_pair_probability_correct :
  matching_color_probability = (39 / 248 : Rat) :=
by
  sorry

end matching_pair_probability_correct_l480_480344


namespace distance_A1C1_BD1_l480_480353

-- Definitions of the cube and lines
structure Cube :=
(edge_length : ℝ)
(A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ)
(cube_property : edge_length = 1)

def Line (p1 p2 : ℝ × ℝ × ℝ) := set (ℝ × ℝ × ℝ)

def distance_between_lines (l1 l2 : set (ℝ × ℝ × ℝ)) : ℝ := sorry

def A1C1 (c : Cube) : set (ℝ × ℝ × ℝ) := Line c.A1 c.C1
def BD1 (c : Cube) : set (ℝ × ℝ × ℝ) := Line c.B c.D1

theorem distance_A1C1_BD1 (c : Cube) (h : c.edge_length = 1) :
  distance_between_lines (A1C1 c) (BD1 c) = (Real.sqrt 6) / 6 := sorry

end distance_A1C1_BD1_l480_480353


namespace completing_the_square_equation_l480_480037

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l480_480037


namespace min_area_of_circle_l480_480601

variable {a : ℝ} (h_pos : 0 < a)

def on_curve : Prop := a * (2/a) = 2
def tangent_to_line : Prop := ∀ (x y r : ℝ), (x, y) = (a, 2/a) → r = (abs (a + 4/a + 1) / sqrt 5) → x + 2 * y + 1 = 0

theorem min_area_of_circle (h_on_curve : on_curve) (h_tangent : tangent_to_line) : 
  (∃ M : ℝ, π * M^2 = 5 * π) :=
sorry

end min_area_of_circle_l480_480601


namespace ratio_of_apples_l480_480080

/-- The store sold 32 red apples and the combined amount of red and green apples sold was 44. -/
theorem ratio_of_apples (R G : ℕ) (h1 : R = 32) (h2 : R + G = 44) : R / 4 = 8 ∧ G / 4 = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end ratio_of_apples_l480_480080


namespace find_numbers_l480_480449

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480449


namespace total_texts_received_l480_480538

open Nat 

-- Definition of conditions
def textsBeforeNoon : Nat := 21
def initialTextsAfterNoon : Nat := 2
def doublingTimeHours : Nat := 12

-- Definition to compute the total texts after noon recursively
def textsAfterNoon (n : Nat) : Nat :=
  if n = 0 then initialTextsAfterNoon
  else 2 * textsAfterNoon (n - 1)

-- Definition to sum the geometric series 
def sumGeometricSeries (a r n : Nat) : Nat :=
  if n = 0 then 0
  else a * (1 - r ^ n) / (1 - r)

-- Total text messages Debby received
def totalTextsReceived : Nat :=
  textsBeforeNoon + sumGeometricSeries initialTextsAfterNoon 2 doublingTimeHours

-- Proof statement
theorem total_texts_received: totalTextsReceived = 8211 := 
by 
  sorry

end total_texts_received_l480_480538


namespace sphere_volume_l480_480915

theorem sphere_volume 
  (d : ℝ) (A : ℝ) (v : ℝ) 
  (hd : d = 1) 
  (hA : A = π) : 
  v = (4 / 3) * π * (sqrt 2) ^ 3 := by
sorry

end sphere_volume_l480_480915


namespace geometric_arithmetic_sequence_l480_480172

theorem geometric_arithmetic_sequence (a_n : ℕ → ℕ) (q : ℕ) (a1_eq : a_n 1 = 3)
  (an_geometric : ∀ n, a_n (n + 1) = a_n n * q)
  (arithmetic_condition : 4 * a_n 1 + a_n 3 = 8 * a_n 2) :
  a_n 3 + a_n 4 + a_n 5 = 84 := by
  sorry

end geometric_arithmetic_sequence_l480_480172


namespace find_s_when_t_eq_5_l480_480566

theorem find_s_when_t_eq_5 (s : ℝ) (h : 5 = 8 * s^2 + 2 * s) :
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 :=
by sorry

end find_s_when_t_eq_5_l480_480566


namespace total_books_l480_480802

theorem total_books (Tim_books : ℕ) (Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 :=
by
  rw [h1, h2]
  exact Nat.add_comm 44 52
  rfl
  sorry

end total_books_l480_480802


namespace completing_square_correct_l480_480024

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l480_480024


namespace smallest_number_diminished_by_16_divisible_l480_480052

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible_l480_480052


namespace find_interest_rate_l480_480366

noncomputable def compound_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := sorry

theorem find_interest_rate :
  compound_interest_rate 676 625 2 1 = 0.04 := 
sorry

end find_interest_rate_l480_480366


namespace valerie_initial_money_l480_480401

theorem valerie_initial_money (n m C_s C_l L I : ℕ) 
  (h1 : n = 3) (h2 : m = 1) (h3 : C_s = 8) (h4 : C_l = 12) (h5 : L = 24) :
  I = (n * C_s) + (m * C_l) + L :=
  sorry

end valerie_initial_money_l480_480401


namespace probability_five_people_get_right_letter_is_zero_l480_480790

theorem probability_five_people_get_right_letter_is_zero (n : ℕ) (h : n = 6) :
  ∀ (dist : fin n → fin n), (∃! i, dist i = i → ∀ j, j ≠ i → dist j ≠ j) → (0:ℝ) = 1/0 := 
sorry

end probability_five_people_get_right_letter_is_zero_l480_480790


namespace min_value_of_dot_product_l480_480588

noncomputable def minimum_dot_product (m n : ℝ) : ℝ :=
  m^2 + n^2 - 3

theorem min_value_of_dot_product :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 1)
  let F1 : ℝ × ℝ := (-√3, 0)
  let F2 : ℝ × ℝ := (√3, 0)
  let line_eq (x y : ℝ) : Prop := x - 2 * y + 2 = 0
  line_eq :=
  ∃ m n,
  line_eq m n ∧
  minimum_dot_product m n = -11 / 5 :=
by
  sorry

end min_value_of_dot_product_l480_480588


namespace volume_of_circular_pool_approx_l480_480865

-- Definitions for the conditions
axiom π_approx : Real
def diameter : Real := 60
def depth : Real := 6
def radius : Real := diameter / 2

-- Problem statement as a theorem
theorem volume_of_circular_pool_approx :
  π_approx = 3.14159 →
  (π * radius^2 * depth) ≈ 16956.54 := 
by
  sorry

end volume_of_circular_pool_approx_l480_480865


namespace find_x_l480_480868

theorem find_x (x : ℝ) (h : 0.35 * 400 = 0.20 * x): x = 700 :=
sorry

end find_x_l480_480868


namespace emilio_pints_of_water_l480_480549

theorem emilio_pints_of_water (elijah_coffee_pints : ℝ) (total_liquid_cups : ℝ) (conversion_factor : ℝ)
  (h1 : elijah_coffee_pints = 8.5)
  (h2 : conversion_factor = 2)
  (h3 : total_liquid_cups = 36) :
  let elijah_coffee_cups := elijah_coffee_pints * conversion_factor in
  let emilio_water_cups := total_liquid_cups - elijah_coffee_cups in
  let emilio_water_pints := emilio_water_cups / conversion_factor in
  emilio_water_pints = 9.5 :=
by
  sorry

end emilio_pints_of_water_l480_480549


namespace total_amount_shared_l480_480943

theorem total_amount_shared (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) 
  (portion_a : ℕ) (portion_b : ℕ) (portion_c : ℕ)
  (h_ratio : ratio_a = 3 ∧ ratio_b = 4 ∧ ratio_c = 9)
  (h_portion_a : portion_a = 30)
  (h_portion_b : portion_b = 2 * portion_a + 10)
  (h_portion_c : portion_c = (ratio_c / ratio_a) * portion_a) :
  portion_a + portion_b + portion_c = 190 :=
by sorry

end total_amount_shared_l480_480943


namespace arithmetic_sequence_common_difference_l480_480736

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (a 5 = 8) → (a 1 + a 2 + a 3 = 6) → (∀ n : ℕ, a (n + 1) = a 1 + n * d) → d = 2 :=
by
  intros ha5 hsum harr
  sorry

end arithmetic_sequence_common_difference_l480_480736


namespace radius_of_tangent_circle_l480_480392

variables (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (PQ PR QR : ℝ)
variables (ω : Type) [MetricSpace ω]
variables (γ : Type) [MetricSpace γ]

-- Triangular conditions
axiom PQ_eq_PR : PQ = 5
axiom QR_eq : QR = 6
axiom triangle_PQR_inscribed_in_ω : inscribed_triangle_in_circle P Q R ω

-- Result to prove
theorem radius_of_tangent_circle 
  (T : Type) [MetricSpace T] (r : ℝ)
  (center_on_QR : center_on_segment T QR)
  (tangent_to_ω : tangent T ω)
  (tangent_to_PQ : tangent T PQ) :
  r = 20 / 9 :=
sorry

end radius_of_tangent_circle_l480_480392


namespace integer_solution_count_l480_480757

theorem integer_solution_count : 
  ∃ n : ℕ, (∀ x : ℤ, log 3 (abs (x - 2)) < 2 → -7 < x ∧ x < 11) ∧ n = 17 :=
sorry

end integer_solution_count_l480_480757


namespace log_eq_b_minus_d_l480_480598

theorem log_eq_b_minus_d (a b c d : ℕ) 
    (h1 : log a b = (3 : ℝ) / 2) 
    (h2 : log c d = (5 : ℝ) / 4) 
    (h3 : a - c = 9) 
    (ha_pos : 0 < a) 
    (hb_pos : 0 < b)
    (hc_pos : 0 < c) 
    (hd_pos : 0 < d) : b - d = 93 := 
  sorry

end log_eq_b_minus_d_l480_480598


namespace domain_of_sqrt_log_half_l480_480352

theorem domain_of_sqrt_log_half (x : ℝ) : 1 < x ∧ x ≤ 2 ↔ ∃ y : ℝ, y = sqrt (log (1/2) (x - 1)) := 
begin
  sorry
end

end domain_of_sqrt_log_half_l480_480352


namespace number_of_ways_to_paint_three_faces_l480_480946

theorem number_of_ways_to_paint_three_faces :
  ∃ (S : Finset ℕ), S.card = 3 ∧ (∀ x ∈ S, ∀ y ∈ S, x + y ≠ 9) ∧ (S ⊆ (Finset.range 9).erase 0) ∧ S.pairwise (λ a b, a + b ≠ 9) :=
  sorry

end number_of_ways_to_paint_three_faces_l480_480946


namespace percentage_savings_l480_480076

/-
Setting up the variables:
- original price of the jacket is $160
- store discount is $20
- extra coupon discount is $16
-/

theorem percentage_savings (original_price store_discount extra_coupon : ℝ) :
    original_price = 160 ∧ store_discount = 20 ∧ extra_coupon = 16 → 
    ((store_discount + extra_coupon) / original_price) * 100 = 22.5 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end percentage_savings_l480_480076


namespace problem_part1_problem_part2_l480_480999

theorem problem_part1 (n : ℕ) (hn : n > 0) : 
  let a_n := n * (n + 1) in 
  a_n = n * (n + 1) :=
sorry

theorem problem_part2 (b_n a_n : ℕ → ℕ) (hn : ∀ n, a_n n = n * (n + 1)) :
  (∀ n, ∑ k in Finset.range n.succ, (-1)^(n - k) * Nat.choose n k * b_n k = a_n n) →
  (∑ n in Finset.range 13, if b_n n ≤ 2019 * a_n n then 1 else 0) = 12 :=
sorry

end problem_part1_problem_part2_l480_480999


namespace sum_of_first_fifteen_multiples_of_17_l480_480827

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480827


namespace isabella_stops_l480_480547

noncomputable def Q (n : ℕ) : ℚ := (∏ k in Finset.range (n-1), (2 * k : ℚ) / (2 * k + 1)) * (1 / (2 * n + 1))

theorem isabella_stops : ∃ n, Q n < (1:ℚ)/3000 ∧ n = 12 := by
  sorry

end isabella_stops_l480_480547


namespace decimal_digits_of_sqrt2_add_sqrt3_power_1980_l480_480561

/-- The statement that proves the digits to the left and right of the 
    decimal point in the decimal form of (sqrt(2) + sqrt(3))^1980 
    are 7.9 -/
theorem decimal_digits_of_sqrt2_add_sqrt3_power_1980 :
  let a := 5 + Real.sqrt 24 in
  let b := 5 - Real.sqrt 24 in
  let N := a ^ 990 + b ^ 990 in
  0 < b ∧ b < 0.2 → (Real.sqrt 2 + Real.sqrt 3) ^ 1980 = 7.9 :=
by
  let a := (5 : ℝ) + Real.sqrt 24
  let b := (5 : ℝ) - Real.sqrt 24
  let N := a ^ 990 + b ^ 990
  assume h : 0 < b ∧ b < 0.2
  sorry

end decimal_digits_of_sqrt2_add_sqrt3_power_1980_l480_480561


namespace record_expenditure_l480_480644

def income (amount : ℤ) := amount > 0
def expenditure (amount : ℤ) := -amount

theorem record_expenditure : 
  (income 100 = true) ∧ (100 = +100) ∧ (income (expenditure 80) = false) → (expenditure 80 = -80) := 
by sorry

end record_expenditure_l480_480644


namespace ratio_of_pieces_l480_480882

theorem ratio_of_pieces (total_length : ℝ) (short_piece : ℝ) (total_length_eq : total_length = 70) (short_piece_eq : short_piece = 27.999999999999993) :
  let long_piece := total_length - short_piece
  let ratio := short_piece / long_piece
  ratio = 2 / 3 :=
by
  sorry

end ratio_of_pieces_l480_480882


namespace find_f_prime_neg1_l480_480616

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x >= 0 then exp x + a * x^2 else exp (-x) + a * x^2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  if x >= 0 then exp x + 2 * a * x else -exp (-x) + 2 * a * x

theorem find_f_prime_neg1 (a : ℝ) (h : f' 1 a = exp 1 + 1) :
  f' (-1) a = -exp 1 - 1 := 
  sorry

end find_f_prime_neg1_l480_480616


namespace coefficients_sum_eq_four_l480_480128

noncomputable def simplified_coefficients_sum (y : ℚ → ℚ) : ℚ :=
  let A := 1
  let B := 3
  let C := 2
  let D := -2
  A + B + C + D

theorem coefficients_sum_eq_four : simplified_coefficients_sum (λ x => 
  (x^3 + 5*x^2 + 8*x + 4) / (x + 2)) = 4 := by
  sorry

end coefficients_sum_eq_four_l480_480128


namespace base_b_square_of_integer_l480_480124

theorem base_b_square_of_integer (b : ℕ) (h : b > 4) : ∃ n : ℕ, (n * n) = b^2 + 4 * b + 4 :=
by 
  sorry

end base_b_square_of_integer_l480_480124


namespace multiple_of_C_share_l480_480501

theorem multiple_of_C_share (A B C k : ℝ) : 
  3 * A = k * C ∧ 4 * B = k * C ∧ C = 84 ∧ A + B + C = 427 → k = 7 :=
by
  sorry

end multiple_of_C_share_l480_480501


namespace find_m_n_l480_480614

noncomputable def f (m n x : ℝ) : ℝ := m * x ^ (m - n)
noncomputable def f' (m n x : ℝ) : ℝ := 8 * x ^ 3

theorem find_m_n (m n : ℝ) (h1 : ∀ x, deriv (f m n x) = f' m n x)
    (h2 : ∃ m n : ℝ, (deriv (λ x => f m n x) = λ x => f' m n x)) : m ^ n = 1 / 4 :=
by
  sorry

end find_m_n_l480_480614


namespace math_problem_l480_480210

noncomputable def f (ω x : ℝ) : ℝ :=
  (sqrt 3) * (sin (ω * x))^2 + 2 * (sin (ω * x)) * (cos (ω * x)) - (sqrt 3) * (cos (ω * x))^2 - 1

theorem math_problem 
  (ω : ℝ) (hω : ω > 0)
  (min_val : ∀ x, f ω x ≥ -3)
  (f_omega_one_increasing : ∀ x (h : -π/12 < x ∧ x < 5*π/12), deriv (f 1) x > 0)
  (distinct_w_geq : ∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 0 ≤ x1 ∧ x1 ≤ π
                      → 0 ≤ x2 ∧ x2 ≤ π → 0 ≤ x3 ∧ x3 ≤ π 
                      → f ω x1 + f ω x2 + f ω x3 = 3 
                      → ω ≥ 29/12) :
  true := 
  sorry

end math_problem_l480_480210


namespace base8_arithmetic_l480_480004

def base8_to_base10 (n : Nat) : Nat :=
  sorry -- Placeholder for base 8 to base 10 conversion

def base10_to_base8 (n : Nat) : Nat :=
  sorry -- Placeholder for base 10 to base 8 conversion

theorem base8_arithmetic (n m : Nat) (h1 : base8_to_base10 45 = n) (h2 : base8_to_base10 76 = m) :
  base10_to_base8 ((n * 2) - m) = 14 :=
by
  sorry

end base8_arithmetic_l480_480004


namespace minimal_polynomial_with_roots_l480_480564

def polynomial_roots := [3 + Real.sqrt 5, 3 - Real.sqrt 5, 2 + Real.sqrt 6, 2 - Real.sqrt 6]

theorem minimal_polynomial_with_roots :
  ∃ p : Polynomial ℚ, p.leadingCoeff = 1 ∧ Polynomial.eval₂ (algebra_map ℚ ℝ) p (3 + Real.sqrt 5) = 0
  ∧ Polynomial.eval₂ (algebra_map ℚ ℝ) p (3 - Real.sqrt 5) = 0
  ∧ Polynomial.eval₂ (algebra_map ℚ ℝ) p (2 + Real.sqrt 6) = 0
  ∧ Polynomial.eval₂ (algebra_map ℚ ℝ) p (2 - Real.sqrt 6) = 0
  ∧ p = Polynomial.mk (Polynomial.Coeffs [1, -10, 28, 8, -32]) := 
sorry

end minimal_polynomial_with_roots_l480_480564


namespace calc_3_pow_l480_480544

variable {k : ℝ}

theorem calc_3_pow (k : ℝ) : 
  3^(-(3*k + 1)) - 3^(-(3*k - 1)) + 3^(-3*k) = - (5/3) * 3^(-3*k) :=
by sorry

end calc_3_pow_l480_480544


namespace sum_first_fifteen_multiples_seventeen_l480_480822

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480822


namespace sum_of_first_fifteen_multiples_of_17_l480_480834

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480834


namespace sum_of_first_fifteen_multiples_of_17_l480_480837

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480837


namespace simplest_fraction_is_one_l480_480402

theorem simplest_fraction_is_one :
  ∃ m : ℕ, 
  (∃ k : ℕ, 45 * m = k^2) ∧ 
  (∃ n : ℕ, 56 * m = n^3) → 
  45 * m / 56 * m = 1 := by
  sorry

end simplest_fraction_is_one_l480_480402


namespace minimum_pencils_l480_480050

-- Define the given conditions
def red_pencils : ℕ := 15
def blue_pencils : ℕ := 13
def green_pencils : ℕ := 8

-- Define the requirement for pencils to ensure the conditions are met
def required_red : ℕ := 1
def required_blue : ℕ := 2
def required_green : ℕ := 3

-- The minimum number of pencils Constanza should take out
noncomputable def minimum_pencils_to_ensure : ℕ := 21 + 1

theorem minimum_pencils (red_pencils blue_pencils green_pencils : ℕ)
    (required_red required_blue required_green minimum_pencils_to_ensure : ℕ) :
    red_pencils = 15 →
    blue_pencils = 13 →
    green_pencils = 8 →
    required_red = 1 →
    required_blue = 2 →
    required_green = 3 →
    minimum_pencils_to_ensure = 22 :=
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end minimum_pencils_l480_480050


namespace AB_eq_CE_l480_480285

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

axiom triangle (A B C : Point) : Prop

variables {A B C M D E : Point}

-- Given conditions
axiom triangle_ABC : triangle A B C
axiom M_midpoint_BC : M = midpoint B C
axiom D_on_AB : ∃ t ∈ Ioo(0, 1), D = (1 - t) • A + t • B
axiom E_on_AM_CD : ∃ t s : ℝ, E = (1 - t) • A + t • M ∧ E = (1 - s) • C + s • D
axiom AD_eq_DE : (A - D).norm = (D - E).norm

-- Proof statement
theorem AB_eq_CE : (A - B).norm = (C - E).norm :=
sorry

end AB_eq_CE_l480_480285


namespace cost_combination_exists_l480_480705

/-!
Given:
- Nadine spent a total of $105.
- The table costs $34.
- The mirror costs $15.
- The lamp costs $6.
- The total cost of the 2 chairs and 3 decorative vases is $50.

Prove:
- There are multiple combinations of individual chair cost (C) and individual vase cost (V) such that 2 * C + 3 * V = 50.
-/

theorem cost_combination_exists :
  ∃ (C V : ℝ), 2 * C + 3 * V = 50 :=
by {
  sorry
}

end cost_combination_exists_l480_480705


namespace simplify_expression_l480_480110

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l480_480110


namespace train_speed_correct_l480_480081

def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  let speed_mps := length / time
  speed_mps * 3.6

theorem train_speed_correct :
  speed_of_train 60 2.9997600191984644 = 72.003 := by
  -- Proof goes here
  sorry

end train_speed_correct_l480_480081


namespace sequences_not_equal_l480_480723

noncomputable def x_seq : ℕ → ℚ
| 0     := 1 / 8
| (n+1) := x_seq n + (x_seq n)^2

noncomputable def y_seq : ℕ → ℚ
| 0     := 1 / 10
| (n+1) := y_seq n + (y_seq n)^2

theorem sequences_not_equal (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  x_seq m ≠ y_seq n :=
sorry

end sequences_not_equal_l480_480723


namespace towel_area_decrease_l480_480864

theorem towel_area_decrease :
  ∀ (L B : ℝ), 
    let L' := 0.70 * L in
    let B' := 0.85 * B in
    let A := L * B in
    let A' := 0.595 * L * B in
    (A - A') / A * 100 = 40.5 :=
by
  intros L B
  let L' := 0.70 * L
  let B' := 0.85 * B
  let A := L * B
  let A' := 0.595 * L * B
  sorry

end towel_area_decrease_l480_480864


namespace jeff_mean_score_l480_480276

theorem jeff_mean_score:
  let scores := [90, 93.5, 87, 96, 92, 89.5] in
  (∑ x in scores, x) / (scores.length) = 91.33333333333333 :=
by
  sorry

end jeff_mean_score_l480_480276


namespace sum_of_first_fifteen_multiples_of_17_l480_480840

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480840


namespace surface_area_of_regular_tetrahedron_l480_480384

theorem surface_area_of_regular_tetrahedron (a : ℝ) : 
  let surface_area := 4 * (√3 / 4 * a^2)
  in surface_area = √3 * a^2 :=
sorry

end surface_area_of_regular_tetrahedron_l480_480384


namespace binomial_distribution_parameters_l480_480308

noncomputable def E (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def D (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (n : ℕ) (p : ℝ) 
  (h1 : E n p = 2.4) (h2 : D n p = 1.44) : 
  n = 6 ∧ p = 0.4 :=
by
  sorry

end binomial_distribution_parameters_l480_480308


namespace lesser_fraction_l480_480378

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l480_480378


namespace pentagon_sum_of_sides_and_vertices_eq_10_l480_480153

-- Define the number of sides of a pentagon
def number_of_sides : ℕ := 5

-- Define the number of vertices of a pentagon
def number_of_vertices : ℕ := 5

-- Define the sum of sides and vertices
def sum_of_sides_and_vertices : ℕ :=
  number_of_sides + number_of_vertices

-- The theorem to prove that the sum is 10
theorem pentagon_sum_of_sides_and_vertices_eq_10 : sum_of_sides_and_vertices = 10 :=
by
  sorry

end pentagon_sum_of_sides_and_vertices_eq_10_l480_480153


namespace option_d_is_same_as_original_l480_480092

-- Definitions of the functions
def func_original (x : ℝ) : ℝ := real.sqrt (-2 * x^3)
def func_a (x : ℝ) : ℝ := x * real.sqrt (-2 * x)
def func_b (x : ℝ) : ℝ := -real.sqrt (2 * x^3)
def func_c (x : ℝ) : ℝ := x^2 * real.sqrt ((-2) / x)
def func_d (x : ℝ) : ℝ := -x * real.sqrt (-2 * x)

-- The domains of each function
def domain (f : ℝ → ℝ) (x : ℝ) : Prop := 
  ∃ y : ℝ, y = f x

theorem option_d_is_same_as_original :
  (∀ x : ℝ, x ≤ 0 → func_d x = func_original x) ∧
  (∀ x : ℝ, 0 ≤ x → domain func_d x = domain func_original x) := by sorry

end option_d_is_same_as_original_l480_480092


namespace sin_of_angle_on_line_eq_l480_480248

noncomputable def sin_of_angle_on_line (α : Real) : Prop :=
  let p := λ (α : Real), (cos α, sin α)
  ∃ k : Real, k ≠ 0 ∧ p α = (k, 2 * k) ∧ sin α = (2 * Real.sqrt 5) / 5 ∨ sin α = -(2 * Real.sqrt 5) / 5

theorem sin_of_angle_on_line_eq :
  ∀ α : Real, sin_of_angle_on_line α :=
by
  intro α
  sorry

end sin_of_angle_on_line_eq_l480_480248


namespace no_five_correct_letters_l480_480794

theorem no_five_correct_letters (n : ℕ) (hn : n = 6) :
  ∀ (σ : Fin n → Fin n), (∑ i, if σ i = i then 1 else 0) ≠ 5 :=
by
  simp only
  sorry

end no_five_correct_letters_l480_480794


namespace value_of_a_1000_l480_480219

def a : ℕ → ℚ
| 0 := 0  -- a technically starts from 1
| 1 := 1/2
| (n+1) := a n / (2 * a n + 1)

theorem value_of_a_1000 : a 1000 = 1/2000 := 
by {
  sorry
}

end value_of_a_1000_l480_480219


namespace hyperbola_focus_distance_l480_480213

theorem hyperbola_focus_distance (a : ℝ) 
  (h₀ : 0 < a)
  (h₁ : (∃ x y : ℝ, (x^2 / a^2 - y^2 / 27 = 1)))
  (h₂ : ∃ k : ℝ, k = real.tan (π / 3))
  (h₃ : ∃ P : ℝ × ℝ, (P.1^2 / a^2 - P.2^2 / 27 = 1) ∧ dist P (-(6 : ℝ), 0) = 7) :
  ∃ PF2 : ℝ, PF2 = 13 := 
sorry

end hyperbola_focus_distance_l480_480213


namespace range_of_f_on_interval_range_of_t_for_monotonic_decrease_l480_480581

-- Define the function f(x) = x^3 + 3x^2
def f (x : ℝ) := x^3 + 3 * x^2

-- The first theorem proving the range of f on [-4, 0]
theorem range_of_f_on_interval : set.range (λ x, f x) (set.Icc (-4 : ℝ) (0 : ℝ)) = set.Icc (-16 : ℝ) (4 : ℝ) :=
by
  sorry

-- The second theorem proving the range of t for f to be monotonically decreasing on [t, t+1] is [-2, -1)
theorem range_of_t_for_monotonic_decrease : ∀ t, (∃ x ∈ set.Icc (t : ℝ) (t + 1), 0 < 3 * x^2 + 6 * x) ↔ t ∈ set.Ico (-2 : ℝ) (-1 : ℝ) :=
by
  sorry

end range_of_f_on_interval_range_of_t_for_monotonic_decrease_l480_480581


namespace chloe_boxes_l480_480527

/-- Chloe was unboxing some of her old winter clothes. She found some boxes of clothing and
inside each box, there were 2 scarves and 6 mittens. Chloe had a total of 32 pieces of
winter clothing. How many boxes of clothing did Chloe find? -/
theorem chloe_boxes (boxes : ℕ) (total_clothing : ℕ) (pieces_per_box : ℕ) :
  pieces_per_box = 8 -> total_clothing = 32 -> total_clothing / pieces_per_box = boxes -> boxes = 4 :=
by
  intros
  sorry

end chloe_boxes_l480_480527


namespace points_planes_configuration_l480_480580

-- Define the setup of points and planes in 3D space

-- Definitions of the points
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def F_A : Point := sorry
noncomputable def F_B : Point := sorry
noncomputable def F_C : Point := sorry
noncomputable def K_A : Point := sorry
noncomputable def K_B : Point := sorry
noncomputable def K_C : Point := sorry
noncomputable def S : Point := sorry
noncomputable def E : Point := sorry

-- Define the planes (each plane passes through exactly 6 points)
noncomputable def plane1 : Plane := sorry
noncomputable def plane2 : Plane := sorry
noncomputable def plane3 : Plane := sorry
noncomputable def plane4 : Plane := sorry
noncomputable def plane5 : Plane := sorry
noncomputable def plane6 : Plane := sorry
noncomputable def plane7 : Plane := sorry
noncomputable def plane8 : Plane := sorry
noncomputable def plane9 : Plane := sorry
noncomputable def plane10 : Plane := sorry
noncomputable def plane11 : Plane := sorry
noncomputable def plane12 : Plane := sorry

-- Prove the required properties
theorem points_planes_configuration :
  (∃ (P : Fin 12 → Point) (Π : Fin 12 → Plane),
    (∀ i, ∃ l, l.card = 6 ∧ (∀ j < l.card, P (l.nth j) ∈ Π i)) ∧
    (∀ i, ∃ m, m.card = 6 ∧ (∀ j < m.card, Π (m.nth j) ∋ P i)) ∧
    (no_collinear_points (set.range P))) :=
sorry

end points_planes_configuration_l480_480580


namespace orange_juice_per_glass_l480_480936

theorem orange_juice_per_glass :
  let total_pints := 502.75
  let total_glasses := 21
  let pints_per_glass := total_pints / total_glasses
  (Real.round (pints_per_glass * 100) / 100 = 23.94) :=
by
  let total_pints := 502.75
  let total_glasses := 21
  let pints_per_glass := total_pints / total_glasses
  trivial

end orange_juice_per_glass_l480_480936


namespace train_cross_bridge_time_l480_480867

-- Definitions for conditions
def length_train : ℝ := 110  -- in meters
def speed_train_kmh : ℝ := 54  -- in kilometers per hour
def length_bridge : ℝ := 132  -- in meters

-- Function to convert speed from km/hr to m/s
def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

-- Proposition stating the equivalent proof problem
theorem train_cross_bridge_time :
  let total_distance := length_train + length_bridge in
  let speed_train_ms := speed_kmh_to_ms speed_train_kmh in
  total_distance / speed_train_ms = 16.13 :=
by
  sorry

end train_cross_bridge_time_l480_480867


namespace discriminant_greater_than_four_l480_480364

theorem discriminant_greater_than_four {p q : ℝ} 
  (h₁ : (999:ℝ)^2 + p * 999 + q < 0) 
  (h₂ : (1001:ℝ)^2 + p * 1001 + q < 0) :
  (p^2 - 4 * q) > 4 :=
sorry

end discriminant_greater_than_four_l480_480364


namespace C_finishes_work_in_2_5_days_l480_480887

-- Definitions of the conditions
def work_rate_A := 1 / 15
def work_rate_B := 1 / 20
def work_rate_C := 1 / 30

def work_completed_A := work_rate_A * 10
def work_completed_B := work_rate_B * 5
def total_work_completed := work_completed_A + work_completed_B
def remaining_work := 1 - total_work_completed

-- The theorem we need to prove
theorem C_finishes_work_in_2_5_days : 
  (remaining_work / work_rate_C) = 2.5 := 
by 
  sorry 

end C_finishes_work_in_2_5_days_l480_480887


namespace quadratic_has_two_distinct_real_roots_l480_480764

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l480_480764


namespace angle_FHP_eq_angle_A_l480_480937

noncomputable def triangle_properties
  (A B C : Point)
  (h_acute : acute_angled_triangle A B C)
  (H : Point) (H_is_orthocenter : is_orthocenter H A B C)
  (O : Point) (O_is_circumcenter : is_circumcenter O A B C)
  (F : Point) (F_altitude : is_altitude F A B C)
  (P : Point) (P_on_AC : P ∈ AC)
  (P_line : line_through_F_perpendicular_to_OF P F O AC) : Prop :=
  angle F H P = angle A

theorem angle_FHP_eq_angle_A
  (A B C : Point)
  (h_acute : acute_angled_triangle A B C)
  (H : Point) (H_is_orthocenter : is_orthocenter H A B C)
  (O : Point) (O_is_circumcenter : is_circumcenter O A B C)
  (F : Point) (F_altitude : is_altitude F A B C)
  (P : Point) (P_on_AC : P ∈ AC)
  (P_line : line_through_F_perpendicular_to_OF P F O AC) :
  triangle_properties A B C h_acute H H_is_orthocenter O O_is_circumcenter F F_altitude P P_on_AC P_line := by
  sorry

end angle_FHP_eq_angle_A_l480_480937


namespace opinions_will_stabilize_l480_480879

theorem opinions_will_stabilize :
  ∃ t : ℕ, ∀ i : ℕ, i < 101 → (∀ j : ℕ, j < t → opinion_at (j+1) i = opinion_at j i) :=
by
  -- Define the opinions of the wise men as a function from time and index to a boolean (true = +, false = -)
  let opinion_at : ℕ → ℕ → Prop := sorry
  -- Define the update rule for opinions
  let update_opinion : ℕ → ℕ → Prop := sorry

  -- Assume initial conditions
  have h_initial_conditions : ∀ i : ℕ, i < 101 → (opinion_at 0 i = true ∨ opinion_at 0 i = false), from sorry

  -- Define what it means for an opinion to be stable
  let is_stable (t i : ℕ) : Prop :=
    (opinion_at t i = opinion_at t ((i + 1) % 101) ∨ opinion_at t i = opinion_at t ((i + 100) % 101))

  -- Prove that after some time t, all opinions will stabilize
  have h_stability : ∃ t : ℕ, ∀ i : ℕ, i < 101 → is_stable t i,
  from sorry

  show ∃ t : ℕ, ∀ i : ℕ, i < 101 → (∀ j : ℕ, j < t → opinion_at (j+1) i = opinion_at j i), from sorry

end opinions_will_stabilize_l480_480879


namespace mutually_exclusive_events_l480_480255

-- Definitions for the conditions
def num_genuine_items : Nat := 4
def num_defective_items : Nat := 3
def total_items : Nat := num_genuine_items + num_defective_items
def items_selected : Nat := 2

-- Event definitions
def exactly_one_defective (selection : Finset (Fin total_items)) : Prop := 
  selection.filter (λ i, i.val ≥ num_genuine_items).card = 1

def exactly_two_defective (selection : Finset (Fin total_items)) : Prop :=
  selection.filter (λ i, i.val ≥ num_genuine_items).card = 2

def at_least_one_defective (selection : Finset (Fin total_items)) : Prop :=
  selection.filter (λ i, i.val ≥ num_genuine_items).nonempty

def all_genuine (selection : Finset (Fin total_items)) : Prop :=
  selection.filter (λ i, i.val ≥ num_genuine_items).card = 0

-- Lean 4 statement for the proof problem
theorem mutually_exclusive_events 
    (s : Finset (Fin total_items))
    (h_s : s.card = items_selected) :
    (exactly_one_defective s ↔ ¬exactly_two_defective s) ∧ 
    (at_least_one_defective s ↔ ¬all_genuine s) :=
  by
  split
  sorry

end mutually_exclusive_events_l480_480255


namespace equal_chords_l480_480115

noncomputable theory
variables {P A A' B B' C C' : Point} -- Assuming Point is a predefined structure for geometric points

-- Defining conditions as hypotheses:
variables (h1 : ∃ (s₁ s₂ : Sphere), s₁ ≠ s₂ ∧ tangent s₁ s₂ ∧ 
 (sphere_through s₁ A B C P) ∧ (sphere_through s₂ A' B' C' P))
(h2 : ∀ {l₁ l₂ : Line}, (line_through l₁ A A' P) ∧ (line_through l₂ B B' P) ∧ l₁ ≠ l₂)
(h3 : ¬ coplanar {A, A', B, B', C, C', P})

-- The theorem we aim to prove:
theorem equal_chords : dist A A' = dist B B' ∧ dist B B' = dist C C' :=
begin
  sorry
end

end equal_chords_l480_480115


namespace compare_abc_l480_480620

noncomputable def a : ℝ := Real.cbrt (0.5)
noncomputable def b : ℝ := Real.exp (1 / Real.log 2)
noncomputable def c : ℝ := Real.exp (1 / Real.log 5)

theorem compare_abc : (2 * a^3 + a = 2) → (b * Real.log 2 b = 1) → (c * Real.log 5 c = 1) → c > b ∧ b > a :=
by
  intro h1 h2 h3
  sorry

end compare_abc_l480_480620


namespace probability_positive_difference_two_or_greater_l480_480396

theorem probability_positive_difference_two_or_greater :
  let S := {1, 2, 3, 4, 5, 6, 7} in
  let total_pairs := (Finset.powersetLen 2 S).card in
  let consecutive_pairs := (Finset.image (λ i : Finset ℕ, i.erase i.min' (by simp)).erase ∅).card in
  let prob_consecutive_pair := consecutive_pairs / total_pairs in
  let prob_non_consecutive_pair := 1 - prob_consecutive_pair in
  prob_non_consecutive_pair = 5 / 7 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7}
  let total_pairs := (Finset.powersetLen 2 S).card
  let consecutive_pairs := 6 -- There are 6 consecutive pairs in the set {1, 2, 3, 4, 5, 6, 7}
  let prob_consecutive_pair := consecutive_pairs / total_pairs
  let prob_non_consecutive_pair := 1 - prob_consecutive_pair
  have H1 : total_pairs = 21, by
    -- Calculation of binomial coefficient here.
    -- binom(7, 2) = 21
    sorry
  have H2 : consecutive_pairs = 6, by
    -- Consecutive pairs in the set {1, 2, 3, 4, 5, 6, 7}
    sorry
  have H3 : prob_consecutive_pair = 6 / 21, by
    -- Calculation of probability of consecutive pairs.
    sorry
  have H4 : prob_non_consecutive_pair = 1 - 6 / 21, by
    -- Calculate the complement probability.
    sorry
  have H5 : prob_non_consecutive_pair = 5 / 7, by
    -- Simplify the probability.
    sorry
  exact H5

end probability_positive_difference_two_or_greater_l480_480396


namespace find_b_l480_480953

theorem find_b (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_period : ∀ x, 0 ≤ x ∧ x ≤ 2 * π → (y = a * sin(b * x + c) + d) completes exactly 4 periods) : 
  b = 4 := 
by
  sorry

end find_b_l480_480953


namespace exists_root_in_interval_l480_480761

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- Proof statement
theorem exists_root_in_interval : ∃ x : ℝ, -2 < x ∧ x < -1 ∧ f x = 0 :=
by
  have h₁ : f (-2) < 0 := by norm_num
  have h₂ : f (-1) > 0 := by norm_num
  -- There would be additional steps here to rigorously show the root exists in Lean
  sorry

end exists_root_in_interval_l480_480761


namespace find_abc_value_l480_480238

open Real

/- Defining the conditions -/
variables (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a * (b + c) = 156) (h5 : b * (c + a) = 168) (h6 : c * (a + b) = 176)

/- Prove the value of abc -/
theorem find_abc_value :
  a * b * c = 754 :=
sorry

end find_abc_value_l480_480238


namespace number_of_hexagons_l480_480883

-- Definitions based on conditions
def num_pentagons : ℕ := 12

-- Based on the problem statement, the goal is to prove that the number of hexagons is 20
theorem number_of_hexagons (h : num_pentagons = 12) : ∃ (num_hexagons : ℕ), num_hexagons = 20 :=
by {
  -- proof would be here
  sorry
}

end number_of_hexagons_l480_480883


namespace numbers_pairs_sum_prod_l480_480422

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480422


namespace largest_integer_inequality_l480_480405

theorem largest_integer_inequality (x : ℤ) (h : 10 - 3 * x > 25) : x = -6 :=
sorry

end largest_integer_inequality_l480_480405


namespace odd_natural_numbers_comparison_l480_480039

theorem odd_natural_numbers_comparison:
  let N (n : ℕ) : ℕ := n^9 % 10000 in
  (finset.card (finset.filter (λ n, n % 2 = 1 ∧ N n > n) (finset.range 10000))) = 
  (finset.card (finset.filter (λ n, n % 2 = 1 ∧ N n < n) (finset.range 10000))) :=
by 
  sorry

end odd_natural_numbers_comparison_l480_480039


namespace round_table_arrangements_l480_480662

theorem round_table_arrangements :
  let number_of_people := 10
  let linear_arrangements := Nat.factorial number_of_people
  let fixed_person_arrangement := linear_arrangements / number_of_people
  fixed_person_arrangement = Nat.factorial (number_of_people - 1) :=
by
  let number_of_people := 10
  let linear_arrangements := Nat.factorial number_of_people
  let fixed_person_arrangement := linear_arrangements / number_of_people
  have h : fixed_person_arrangement = Nat.factorial 9, from sorry
  exact h

end round_table_arrangements_l480_480662


namespace find_n_l480_480512

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (n : ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ k : ℕ, a (k + 1) = a k + d

def sum_odd_numbered_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), a (2 * i + 1)

def sum_even_numbered_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a (2 * i + 2)

-- Sum of the terms
def sum_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  sum_odd_numbered_terms a n + sum_even_numbered_terms a n

-- The proof statement
theorem find_n (h1 : is_arithmetic_sequence a d)
               (h2 : sum_odd_numbered_terms a n = 6)
               (h3 : sum_even_numbered_terms a n = 5)
               : 2 * n + 1 = 11 :=
by
  sorry

end find_n_l480_480512


namespace round_table_arrangements_l480_480663

theorem round_table_arrangements :
  let number_of_people := 10
  let linear_arrangements := Nat.factorial number_of_people
  let fixed_person_arrangement := linear_arrangements / number_of_people
  fixed_person_arrangement = Nat.factorial (number_of_people - 1) :=
by
  let number_of_people := 10
  let linear_arrangements := Nat.factorial number_of_people
  let fixed_person_arrangement := linear_arrangements / number_of_people
  have h : fixed_person_arrangement = Nat.factorial 9, from sorry
  exact h

end round_table_arrangements_l480_480663


namespace lesser_fraction_l480_480383

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l480_480383


namespace cards_per_player_l480_480100

theorem cards_per_player (n p : ℕ) (h1 : n = 54) (h2 : p = 3) : n / p = 18 :=
by
  rw [h1, h2]
  norm_num

end cards_per_player_l480_480100


namespace count_integers_satisfying_condition_l480_480372

open Nat

theorem count_integers_satisfying_condition :
  {x : ℕ // 4 < x ∧ x < 16}.card = 11 :=
by
  sorry

end count_integers_satisfying_condition_l480_480372


namespace max_boxes_in_large_box_l480_480503

def max_boxes (l_L w_L h_L : ℕ) (l_S w_S h_S : ℕ) : ℕ :=
  (l_L * w_L * h_L) / (l_S * w_S * h_S)

theorem max_boxes_in_large_box :
  let l_L := 8 * 100 -- converted to cm
  let w_L := 7 * 100 -- converted to cm
  let h_L := 6 * 100 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  max_boxes l_L w_L h_L l_S w_S h_S = 2000000 :=
by {
  let l_L := 800 -- converted to cm
  let w_L := 700 -- converted to cm
  let h_L := 600 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  trivial
}

end max_boxes_in_large_box_l480_480503


namespace find_five_digit_number_l480_480136

theorem find_five_digit_number : 
  ∃ (A B C D E : ℕ), 
    (0 < A ∧ A ≤ 9) ∧ 
    (0 < B ∧ B ≤ 9) ∧ 
    (0 < C ∧ C ≤ 9) ∧ 
    (0 < D ∧ D ≤ 9) ∧ 
    (0 < E ∧ E ≤ 9) ∧ 
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E) ∧ 
    (B ≠ C ∧ B ≠ D ∧ B ≠ E) ∧ 
    (C ≠ D ∧ C ≠ E) ∧ 
    (D ≠ E) ∧ 
    (2016 = (10 * D + E) * A * B) ∧ 
    (¬ (10 * D + E) % 3 = 0) ∧ 
    (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E = 85132) :=
sorry

end find_five_digit_number_l480_480136


namespace lesser_fraction_exists_l480_480377

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l480_480377


namespace find_mode_l480_480368

def scores : List ℕ :=
  [105, 107, 111, 111, 112, 112, 115, 118, 123, 124, 124, 126, 127, 129, 129, 129, 130, 130, 130, 130, 131, 140, 140, 140, 140]

def mode (ls : List ℕ) : ℕ :=
  ls.foldl (λmodeScore score => if ls.count score > ls.count modeScore then score else modeScore) 0

theorem find_mode :
  mode scores = 130 :=
by
  sorry

end find_mode_l480_480368


namespace find_numbers_l480_480447

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480447


namespace bathroom_square_footage_l480_480475

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l480_480475


namespace min_weights_needed_l480_480406

theorem min_weights_needed :
  ∃ (weights : List ℕ), (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 → ∃ (left right : List ℕ), m = (left.sum - right.sum)) ∧ weights.length = 5 :=
sorry

end min_weights_needed_l480_480406


namespace line_relationship_l480_480650

-- Defining the concept of skew lines in a 3D space.
def skew (l m : set (ℝ³)) : Prop := 
  ∃ (p : ℝ³), p ∈ l ∧ p ∉ m ∧
  ∃ (q : ℝ³), q ∈ m ∧ q ∉ l

-- Given the conditions that a and b are skew to a line l
variables (a b l : set (ℝ³))
hypothesis h1 : skew a l
hypothesis h2 : skew b l

theorem line_relationship : 
  (a = b ∨ (∃ (p : ℝ³), p ∈ a ∧ p ∈ b) ∨ ¬ (∃ (p : ℝ³), p ∈ a ∧ p ∈ b)) :=
by {
  sorry
}

end line_relationship_l480_480650


namespace product_of_intervals_is_power_of_3_l480_480305

theorem product_of_intervals_is_power_of_3 (p : ℕ) (prime_p : Prime p) (h : p > 3) :
  ∃ (a : List ℤ) (n : ℕ), (∀ i, 0 ≤ i → i < a.length → -((p : ℤ) / 2) < a.nth_le i sorry ∧ a.nth_le i sorry < (p : ℤ) / 2)
  ∧ (∀ i j, 0 ≤ i → i < j → j < a.length → a.nth_le i sorry < a.nth_le j sorry)
  ∧ (∏ i in List.range a.length, (p - a.nth_le i sorry) / |a.nth_le i sorry|) = 3 ^ n := by
  sorry

end product_of_intervals_is_power_of_3_l480_480305


namespace equality_of_expressions_l480_480858

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l480_480858


namespace find_numbers_l480_480452

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480452


namespace initial_value_amount_l480_480135

theorem initial_value_amount (P : ℝ) 
  (h1 : ∀ t, t ≥ 0 → t = P * (1 + (1/8)) ^ t) 
  (h2 : P * (1 + (1/8)) ^ 2 = 105300) : 
  P = 83200 := 
sorry

end initial_value_amount_l480_480135


namespace find_numbers_l480_480438

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480438


namespace matrix_vec_addition_l480_480968

def matrix := (Fin 2 → Fin 2 → ℤ)
def vector := Fin 2 → ℤ

def m : matrix := ![![4, -2], ![6, 5]]
def v1 : vector := ![-2, 3]
def v2 : vector := ![1, -1]

def matrix_vec_mul (m : matrix) (v : vector) : vector :=
  ![m 0 0 * v 0 + m 0 1 * v 1,
    m 1 0 * v 0 + m 1 1 * v 1]

def vec_add (v1 v2 : vector) : vector :=
  ![v1 0 + v2 0, v1 1 + v2 1]

theorem matrix_vec_addition :
  vec_add (matrix_vec_mul m v1) v2 = ![-13, 2] :=
by
  sorry

end matrix_vec_addition_l480_480968


namespace fraction_meaningful_l480_480748

theorem fraction_meaningful (x : ℝ) : (x ≠ -1) ↔ (∃ k : ℝ, k = 1 / (x + 1)) :=
by
  sorry

end fraction_meaningful_l480_480748


namespace division_of_cubics_l480_480003

theorem division_of_cubics (a b c : ℕ) (h_a : a = 7) (h_b : b = 6) (h_c : c = 1) :
  (a^3 + b^3) / (a^2 - a * b + b^2 + c) = 559 / 44 :=
by
  rw [h_a, h_b, h_c]
  -- After these substitutions, the problem is reduced to proving
  -- (7^3 + 6^3) / (7^2 - 7 * 6 + 6^2 + 1) = 559 / 44
  sorry

end division_of_cubics_l480_480003


namespace crates_ratio_l480_480574

-- Definitions
variable (T : ℕ)
axiom Monday_crates : 5
axiom Tuesday_crates : T
axiom Wednesday_crates : T - 2
axiom Thursday_crates : T / 2
axiom Total_crates : 5 + T + (T - 2) + T / 2 = 28

-- Theorem Statement
theorem crates_ratio : T = 10 → T / 5 = 2 := 
by 
  intro h1
  sorry

end crates_ratio_l480_480574


namespace projection_correct_l480_480154

-- Define the given vectors
def u : ℝ × ℝ × ℝ := (4, -1, 3)
def v : ℝ × ℝ × ℝ := (3, -1, 2)

-- Compute the projection of u onto v
def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let scalar := dot_uv / dot_vv
  (scalar * v.1, scalar * v.2, scalar * v.3)

-- Define the expected result
def expected_projection : ℝ × ℝ × ℝ := (57 / 14, -19 / 14, 19 / 7)

-- Verify the projection is as expected
theorem projection_correct : projection u v = expected_projection :=
by
  -- Skipping the proof
  sorry

end projection_correct_l480_480154


namespace four_digit_numbers_without_repetition_count_l480_480229

-- Define the digits and the predicate for a four-digit number without repetitions
def isFourDigitWithoutRepetition (n : ℕ) : Prop :=
  ∃ (x y z t : ℕ), x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {1, 2, 3, 4, 5, 6} ∧ t ∈ {1, 2, 3, 4, 5, 6} ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧ -- Ensure all digits are unique
  n = x * 1000 + y * 100 + z * 10 + t -- Form the four-digit number

-- Main statement to prove
theorem four_digit_numbers_without_repetition_count :
  {n : ℕ | isFourDigitWithoutRepetition n}.to_finset.card = 360 :=
  sorry

end four_digit_numbers_without_repetition_count_l480_480229


namespace platform_length_l480_480419

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) (platform_length : ℝ) :
  train_length = 300 → time_pole = 18 → time_platform = 38 → speed = train_length / time_pole →
  platform_length = (speed * time_platform) - train_length → platform_length = 333.46 :=
by
  introv h1 h2 h3 h4 h5
  sorry

end platform_length_l480_480419


namespace find_two_numbers_l480_480460

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480460


namespace arithmetic_sequence_common_difference_l480_480666

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : a 7 = 8)
  (h2 : ∑ i in Finset.range 7, a i = 42)
  : d = 2 / 3 :=
sorry

end arithmetic_sequence_common_difference_l480_480666


namespace find_digit_for_multiple_of_3_l480_480572

theorem find_digit_for_multiple_of_3 (d : ℕ) (h : d < 10) : 
  (56780 + d) % 3 = 0 ↔ d = 1 :=
by sorry

end find_digit_for_multiple_of_3_l480_480572


namespace max_value_of_f_value_of_f_given_tan_half_alpha_l480_480208

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * (Real.sin x)

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 3) ∧ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 ∧ f x = 3) :=
sorry

theorem value_of_f_given_tan_half_alpha (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_of_f_value_of_f_given_tan_half_alpha_l480_480208


namespace problem_3_problem_4_l480_480576

-- Definitions for the conditions
def zero_lt_half_pi (x : ℝ) : Prop := 0 < x ∧ x < (π / 2)

variables (α β : ℝ)

-- Problem 3
theorem problem_3 (hαβ_interval : zero_lt_half_pi α ∧ zero_lt_half_pi β)
  (h_sin_alpha_2beta : sin (α + 2 * β) = 1 / 3)
  (h_alpha_beta_sum : α + β = 2 * π / 3) :
  sin β = (2 * real.sqrt 6 - 1) / 6 :=
sorry

-- Problem 4
theorem problem_4 (hβ_interval : zero_lt_half_pi β)
  (h_sin_beta : sin β = 4 / 5)
  (h_sin_alpha_2beta : sin (α + 2 * β) = 1 / 3) :
  cos α = (24 + 14 * real.sqrt 2) / 75 :=
sorry

end problem_3_problem_4_l480_480576


namespace probability_five_people_get_right_letter_is_zero_l480_480791

theorem probability_five_people_get_right_letter_is_zero (n : ℕ) (h : n = 6) :
  ∀ (dist : fin n → fin n), (∃! i, dist i = i → ∀ j, j ≠ i → dist j ≠ j) → (0:ℝ) = 1/0 := 
sorry

end probability_five_people_get_right_letter_is_zero_l480_480791


namespace numbers_pairs_sum_prod_l480_480425

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480425


namespace circumcircle_intersections_contain_C1_l480_480394

theorem circumcircle_intersections_contain_C1 
  (A B C A1 B1 C1 D : Point)
  (h_parallel_sides : parallel (line A B) (line A1 B1))
  (h_same_line : collinear A B A1)
  (h_intersection_D : ∃ (O1 O2 : Circle), 
    circumcircle (A1 B C) = O1 ∧ circumcircle (A B1 C) = O2 ∧ intersection O1 O2 = {D}) :
  (C1 ∈ line_through (intersection_point (circumcircle (A1 B C)) (circumcircle (A B1 C))) D) :=
sorry

end circumcircle_intersections_contain_C1_l480_480394


namespace marks_obtained_l480_480500

def student_marks (total_marks : ℕ) (pass_percentage : ℚ) (failed_by : ℤ) : ℤ :=
  let passing_marks := (pass_percentage * total_marks)
  passing_marks - failed_by

theorem marks_obtained : student_marks 800 (33/100) 89 = 175 := 
by
  sorry

end marks_obtained_l480_480500


namespace expression_identity_l480_480407

theorem expression_identity (a : ℤ) (h : a = 102) : 
  a^4 - 4 * a^3 + 6 * a^2 - 4 * a + 1 = 104060401 :=
by {
  rw h,
  calc 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 101^4 : by sorry
  ... = 104060401 : by sorry
}

end expression_identity_l480_480407


namespace max_value_of_expression_l480_480186

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : 
  x + 2 * y + 3 * z ≤ 3 * real.sqrt 14 := 
sorry

end max_value_of_expression_l480_480186


namespace completing_square_correct_l480_480026

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l480_480026


namespace vampire_pints_per_person_l480_480935

-- Definitions based on conditions
def gallons_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7
def people_per_day : ℕ := 4

-- The statement to be proven
theorem vampire_pints_per_person :
  (gallons_per_week * pints_per_gallon) / (days_per_week * people_per_day) = 2 :=
by
  sorry

end vampire_pints_per_person_l480_480935


namespace sunshine_value_sequence_l480_480584

theorem sunshine_value_sequence (a : ℕ → ℝ) (H : ℕ → ℝ) (h_pos : ∀ n, a n > 0) :
  (H n = (n : ℝ) / (∑ i in finset.range n, (i + 1) * a (i + 1))) →
  (H n = 2 / (n + 2)) →
  a n = (2 * n + 1) / (2 * n) :=
by sorry

end sunshine_value_sequence_l480_480584


namespace parallelogram_if_eq_lengths_l480_480873

variable (A B C D : Point)

-- Define check if two lengths are equal (for Lean syntax)
def dist_eq (P Q : Point) : Prop := dist P Q = dist Q P

-- Define midpoints
def midpoint (P Q : Point) : Point := (P + Q) / 2

-- Given conditions
variable (A' := midpoint B C)
variable (B' := midpoint C D)
variable (C' := midpoint D A)
variable (D' := midpoint A B)

-- Main statement
theorem parallelogram_if_eq_lengths :
  dist_eq A (midpoint B C) = dist_eq C (midpoint D A) →
  dist_eq B (midpoint C D) = dist_eq D (midpoint A B) →
  parallelogram A B C D :=
begin
  sorry, -- proof is not required
end

end parallelogram_if_eq_lengths_l480_480873


namespace traveled_distance_is_9_l480_480910

-- Let x be the usual speed in mph
variable (x : ℝ)
-- Let t be the usual time in hours
variable (t : ℝ)

-- Conditions
axiom condition1 : x * t = (x + 0.5) * (3 / 4 * t)
axiom condition2 : x * t = (x - 0.5) * (t + 3)

-- The journey distance d in miles
def distance_in_miles : ℝ := x * t

-- We can now state the theorem to prove that the distance he traveled is 9 miles
theorem traveled_distance_is_9 : distance_in_miles x t = 9 := by
  sorry

end traveled_distance_is_9_l480_480910


namespace triangular_pyramid_height_l480_480264

noncomputable def pyramid_height (a b c h : ℝ) : Prop :=
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2

theorem triangular_pyramid_height {a b c h : ℝ} (h_gt_0 : h > 0) (a_gt_0 : a > 0) (b_gt_0 : b > 0) (c_gt_0 : c > 0) :
  pyramid_height a b c h := by
  sorry

end triangular_pyramid_height_l480_480264


namespace assorted_candies_count_l480_480519

theorem assorted_candies_count
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (chocolate_bars : ℕ)
  (assorted_candies : ℕ) :
  total_candies = 50 →
  chewing_gums = 15 →
  chocolate_bars = 20 →
  assorted_candies = total_candies - (chewing_gums + chocolate_bars) →
  assorted_candies = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end assorted_candies_count_l480_480519


namespace largest_s_value_l480_480295

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end largest_s_value_l480_480295


namespace sum_of_possible_intersections_l480_480568

theorem sum_of_possible_intersections :
  ∀ (L : set (line ℝ)) (hL : L.card = 5)
    (h1 : ∀ l₁ l₂ l₃ ∈ L, l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ → ¬(concurrent l₁ l₂ l₃)),
    ∑ i in {n | ∃ (J : set (line ℝ)), J ⊆ L ∧ J.card = i}, (i.choose 2) = 49 :=
by sorry

end sum_of_possible_intersections_l480_480568


namespace part1_part2_l480_480227

variables (x : ℝ)
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (x, 1)

def vec_sum : ℝ × ℝ := (2 * x + 1, 4)
def vec_diff : ℝ × ℝ := (2 - x, 3)

-- Part 1: When the vectors are parallel
theorem part1 (h : vec_sum = (3 * vec_a.1, 3 * vec_a.2)) : x = 1 / 2 := sorry

-- Part 2: When the vectors are perpendicular
theorem part2 (h : vec_sum.1 * vec_diff.1 + vec_sum.2 * vec_diff.2 = 0) :
  x = -2 ∨ x = 7 / 2 := sorry

end part1_part2_l480_480227


namespace max_sum_at_8_l480_480180

variable {a : ℕ → ℤ}

def arithmetic_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

def S_n (n : ℕ) : ℤ :=
  arithmetic_sum n a

theorem max_sum_at_8 (h1 : S_n 16 > 0) (h2 : S_n 17 < 0) : ∃ m, S_n m = max { S_n k | k : ℕ } ∧ m = 8 :=
sorry

end max_sum_at_8_l480_480180


namespace max_square_plots_l480_480900

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l480_480900


namespace find_fraction_l480_480262

open Classical
noncomputable theory

-- Definitions and conditions
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}
axiom arithmetic_sequence (a_n : ℕ → ℚ) : Prop
axiom sum_of_first_n_terms (a_n : ℕ → ℚ) : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) * ((a_n 2) - (a_n 1))) / 2
axiom condition : a_n 2 / a_n 3 = 1 / 3

-- The theorem to prove
theorem find_fraction (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) (h_seq : arithmetic_sequence a_n) (h_sum : sum_of_first_n_terms a_n) (h_cond : condition) :
  S_n 4 / S_n 5 = 8 / 15 :=
sorry

end find_fraction_l480_480262


namespace area_inside_rectangle_outside_circles_l480_480921

noncomputable def rectangle_area (EF FG : ℝ) : ℝ :=
  EF * FG

noncomputable def circle_area (radius : ℝ) : ℝ :=
  real.pi * radius * radius

noncomputable def effective_circle_area : ℝ :=
  circle_area 2 + circle_area 1.5 + circle_area 2.5 - real.pi * 6

theorem area_inside_rectangle_outside_circles (EF FG : ℝ)
  (hEF : EF = 4) (hFG : FG = 6) : rectangle_area EF FG - effective_circle_area = 3.6 :=
by {
  rw [rectangle_area, effective_circle_area, circle_area],
  norm_num,
  simp [real.pi],
  norm_num,
  sorry
}

end area_inside_rectangle_outside_circles_l480_480921


namespace all_three_clubs_l480_480114

def total_students : ℕ := 40
def music_club_fraction : ℝ := 1 / 4
def science_club_fraction : ℝ := 1 / 5
def neither_club : ℕ := 7
def sports_club : ℕ := 8

def only_music : ℕ := 6
def only_science : ℕ := 5
def only_sports : ℕ := 2
variables (x y z w : ℕ) -- These represent the unknowns in the Venn diagram.

theorem all_three_clubs :
  (total_students - neither_club = 33) ∧ 
  (only_music + x + z + w = total_students * music_club_fraction) ∧
  (only_science + x + y + w = total_students * science_club_fraction) ∧
  (only_sports + z + y + w = sports_club) -> 
  w = 1 :=
by sorry

end all_three_clubs_l480_480114


namespace prob_E_given_D_l480_480505

variable (E L D : Event ω) -- Define the events

-- Given conditions:
variable (prob_E : ℙ E = 0.2)
variable (prob_L_given_not_E : ℙ (L | Eᶜ) = 0.25)
variable [IsProbabilityMeasure ℙ]

theorem prob_E_given_D : ℙ (E | D) = 0.5 :=
by
  -- Calculation of total probability of death event
  
  sorry -- Proof is omitted as per instruction.

end prob_E_given_D_l480_480505


namespace cosine_angle_l480_480627

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-1, 1, 2)
def b : ℝ × ℝ × ℝ := (2, 1, 3)

-- Function to compute the dot product
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2) 

-- Main statement to prove the cosine of the angle between vectors a and b
theorem cosine_angle :
  (dot_product a b) / (magnitude a * magnitude b) = 5 * real.sqrt 21 / 42 :=
by
  sorry

end cosine_angle_l480_480627


namespace hiker_distance_from_start_l480_480904

theorem hiker_distance_from_start (east west south north: ℝ) :
  east = 15 → west = 15 → south = 20 → north = 5 → 
  real.sqrt (0 ^ 2 + (south - north) ^ 2) = 15 :=
by 
  intros h1 h2 h3 h4
  let vertical_movement := south - north
  have h5: vertical_movement = 15 := by linarith [h3, h4]
  have h6: 0 ^ 2 + vertical_movement ^ 2 = 15 ^ 2 := by 
    rw [h5, zero_pow, add_zero]
  rw [real.sqrt_eq_rpow, real.rpow_nat_cast] at h6
  norm_num at h6
  exact h6

end hiker_distance_from_start_l480_480904


namespace greatest_k_for_hexagon_area_l480_480282

-- Defining the setup and problem
variables {A B C D E F G H I : Point}

noncomputable def area_triangle (A B C : Point) : ℝ := sorry -- Placeholder for area of triangle calculation
noncomputable def area_polygon (points : List Point) : ℝ := sorry -- Placeholder for area of polygon calculation

theorem greatest_k_for_hexagon_area (ABC : Triangle) (k : ℝ)
  (h1: is_drawn_externally_square ABC.side1 BADE)
  (h2: is_drawn_externally_square ABC.side2 CBFG)
  (h3: is_drawn_externally_square ABC.side3 ACHI)
  (hx : DEFGHI = hexagon_formed_by_squares ABC BADE CBFG ACHI):
  k = 4 →
  area_polygon [D, E, F, G, H, I] ≥ k * area_triangle A B C :=
begin
  sorry
end

end greatest_k_for_hexagon_area_l480_480282


namespace total_sampled_students_is_80_l480_480274

-- Given conditions
variables (total_students num_freshmen num_sampled_freshmen : ℕ)
variables (total_students := 2400) (num_freshmen := 600) (num_sampled_freshmen := 20)

-- Define the proportion for stratified sampling.
def stratified_sampling (total_students num_freshmen num_sampled_freshmen total_sampled_students : ℕ) : Prop :=
  num_freshmen / total_students = num_sampled_freshmen / total_sampled_students

-- State the theorem: Prove the total number of students to be sampled from the entire school is 80.
theorem total_sampled_students_is_80 : ∃ n, stratified_sampling total_students num_freshmen num_sampled_freshmen n ∧ n = 80 := 
sorry

end total_sampled_students_is_80_l480_480274


namespace fleas_treatment_difference_l480_480073

theorem fleas_treatment_difference:
  ∀ (F : ℝ), 
    (let f1 := 0.40 * F in
    let f2 := 0.55 * f1 in
    let f3 := 0.70 * f2 in
    let f4 := 0.80 * f3 in
    let f5 := 0.90 * f4 in
    f5 = 25) → (F - 25 = 201) :=
by
  intro F
  sorry

end fleas_treatment_difference_l480_480073


namespace problem1_problem2_l480_480061

-- Definition of the first problem
def line_through_point_with_equal_intercepts (a : ℝ) (b : ℝ) : Prop :=
  (a = 2 ∧ b = 3) → (∃ l : ℝ, l ≠ 0 ∧ (l * a + b * l = l * l))

-- Definition of the second problem
def line_through_intersection_and_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  -- Conditions for the intersection point of the lines
  (x1 - 2 * y1 = 3) ∧ (2 * x1 - 3 * y1 = 2) →
  -- Intersection point is (-5, -4)
  (x1 = -5) ∧ (y1 = -4) →
  -- Line perpendicular to 7x + 5y + 1 = 0 has slope -7/5
  (∃ k m : ℝ, k ≠ 0 ∧ m ≠ 0 ∧ m = -5/7 * k ∧ m = (y1 - (-4)) / (x1 - (-5)))

-- First problem statement in Lean
theorem problem1 : line_through_point_with_equal_intercepts 2 3 :=
by {
  intro h,
  cases h with ha hb,
  use (3 * 2 - 2 * 3), 
  split, exact zero_ne_zero,
  exact eq.symm,
}

-- Second problem statement in Lean
theorem problem2 : line_through_intersection_and_perpendicular (-5) (-4) 5 7 :=
by {
  intros h1 h2,
  cases h1,
  cases h2,
  use 5 / 7, 
  use -5 / 7,
  split, 
  exact zero_ne_zero,
  split, exact zero_ne_zero,
  split, exact eq.refl ,
  {Transitivity (y - (-4)) / (x - (-5)), 
   exact eq.refl},
}

end problem1_problem2_l480_480061


namespace octal_to_base7_l480_480971

    theorem octal_to_base7 (n : ℕ) (h : n = 8^2 * 5 + 8^1 * 3 + 8^0 * 2) : 
      nat.toDigits 7 n = [1, 0, 0, 6] :=
    by
      -- Proof is omitted
      sorry
    
end octal_to_base7_l480_480971


namespace sphinx_tiling_impossible_l480_480113

def equilateral_triangle (side_length : ℕ) : Prop :=
  ∃ (n : ℕ), side_length = n

def sphinx_shape (small_triangles_count : ℕ) : Prop :=
  small_triangles_count = 6

theorem sphinx_tiling_impossible (n : ℕ) (sphinx_triangles : ℕ) 
  (h1 : equilateral_triangle 6) 
  (h2 : sphinx_shape 6)
  : ¬ ∃ (covering : (ℕ × ℕ) → (ℕ × ℕ) → ℕ), 
    (∀ (x y : ℕ), 
      let gray_triangles := 15 in 
      let white_triangles := 21 in 
      covering x y = sphinx_triangles) :=
sorry

end sphinx_tiling_impossible_l480_480113


namespace closest_number_proof_l480_480511

noncomputable def closest_to_200000 : ℕ := 199999

theorem closest_number_proof:
  ∀ (n : ℕ), (n = 201000 ∨ n = 199999 ∨ n = 204000) →
  ∃ x : ℕ, x = closest_to_200000 ∧
    abs (200000 - x) ≤ abs (200000 - n) :=
by
  intro n h
  use closest_to_200000
  split
  · exact rfl
  · cases h
    · rw [h]
      norm_num
    · cases h
      · rw [h]
        norm_num
      · rw [h]
        norm_num
  sorry -- ensuring proof


end closest_number_proof_l480_480511


namespace no_permutation_satisfies_inequality_l480_480152

open real

noncomputable def factorial (n : ℕ) : ℕ :=
nat.rec_on n 1 (λ n ih, (n + 1) * ih)

theorem no_permutation_satisfies_inequality :
  ∀ (a : list ℕ), a.perm (list.range 6).map (λ n, n + 1) →
  (1 <= a.length) ∧
  (2 <= a.length) ∧
  (3 <= a.length) ∧
  (4 <= a.length) ∧
  (5 <= a.length) ∧
  (6 <= a.length) →
  (∀ a_i ∈ a, a_i ∈ (list.range 6).map (λ n, n + 1)) →
    ((a.map (λ k, (k + a.index_of k.succ) / 2)).product > factorial 12) → false :=
begin
  intros a hperm hlen hmem hineq,
  sorry
end

end no_permutation_satisfies_inequality_l480_480152


namespace circle_tangency_proof_l480_480464

-- Define the basic entities like the triangle and its sides
variables {R a b c h_a h_b h_c t_a t_b t_c : ℝ}

-- Assume the conditions given in the problem
axiom tangent_circles (triangle_ABC : ∃ (A B C : ℝ), True)
  (tangent_A : t_a = R h_a / (a + h_a))
  (tangent_B : t_b = R h_b / (b + h_b))
  (tangent_C : t_c = R h_c / (c + h_c)) : True

-- State the theorem to be proved
theorem circle_tangency_proof (triangle_ABC : ∃ (A B C : ℝ), True)
  (H_tangent_A : t_a = R h_a / (a + h_a))
  (H_tangent_B : t_b = R h_b / (b + h_b))
  (H_tangent_C : t_c = R h_c / (c + h_c)) : 
  (t_a = R h_a / (a + h_a)) ∧ (t_b = R h_b / (b + h_b)) ∧ (t_c = R h_c / (c + h_c)) :=
by 
  -- The proof is not required, so we add sorry
  sorry

end circle_tangency_proof_l480_480464


namespace y_coord_equidistant_l480_480009

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l480_480009


namespace fourth_root_of_207360000_l480_480533

theorem fourth_root_of_207360000 :
  120 ^ 4 = 207360000 :=
sorry

end fourth_root_of_207360000_l480_480533


namespace find_f_eq_l480_480201

variable {ℝ : Type*} [IsROrC ℝ]

def f (x : ℝ) : ℝ :=
  x^2 + 2 * x * ((fun y, (2 * y + 2))(2))

theorem find_f_eq : (f : ℝ → ℝ) = fun x, x^2 + 12 * x := by
  sorry

end find_f_eq_l480_480201


namespace lions_kill_deers_l480_480240

theorem lions_kill_deers 
  (n : ℕ) 
  (h : 13 lions can kill 13 deers in 13 minutes) 
  (equal_num : n lions and n deers) 
  : n lions can kill n deers in 13 minutes :=
sorry

end lions_kill_deers_l480_480240


namespace pythagorean_set_A_l480_480043

theorem pythagorean_set_A : 
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  x^2 + y^2 = z^2 := 
by
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  sorry

end pythagorean_set_A_l480_480043


namespace sum_of_first_fifteen_multiples_of_17_l480_480841

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480841


namespace number_of_second_graders_l480_480778

-- Define the number of kindergartners
def kindergartners : ℕ := 34

-- Define the number of first graders
def first_graders : ℕ := 48

-- Define the total number of students
def total_students : ℕ := 120

-- Define the proof statement
theorem number_of_second_graders : total_students - (kindergartners + first_graders) = 38 := by
  -- omit the proof details
  sorry

end number_of_second_graders_l480_480778


namespace find_numbers_sum_eq_S_product_eq_P_l480_480431

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480431


namespace conjugate_of_z_l480_480597

def z : ℂ := 1 - complex.i

theorem conjugate_of_z : complex.conj z = 1 + complex.i :=
by
  sorry

end conjugate_of_z_l480_480597


namespace inscribed_square_areas_are_equal_l480_480795

-- Define what it means to inscribe a square in an isosceles right triangle
def isosceles_right_triangle (a b c : ℕ) : Prop :=
  a = b ∧ c = a * √2

-- Define the specific triangle ABC as an isosceles right triangle
def triangle_ABC (a b c : ℕ) : Prop :=
  isosceles_right_triangle a b c

-- Define the first square being inscribed in the isosceles right triangle
def first_square_in_triangle_ABC_has_area (side len_wid_area : ℕ) : Prop :=
  triangle_ABC side side (side * √2) ∧
  len_wid_area = side * side

-- Define the second square being inscribed in the isosceles right triangle
def second_square_in_triangle_ABC_has_area (side len_wid_area : ℕ) : Prop :=
  triangle_ABC side side (side * √2) ∧
  len_wid_area = side * side

-- Prove the areas of both squares are the same given the conditions
theorem inscribed_square_areas_are_equal :
  ∀ (a b c : ℕ), 
    triangle_ABC a b c → 
    first_square_in_triangle_ABC_has_area 22 484 →
    second_square_in_triangle_ABC_has_area 22 484 :=
by
  sorry

end inscribed_square_areas_are_equal_l480_480795


namespace same_function_C_l480_480861

def f_A (x : ℝ) : ℝ := x
def g_A (x : ℝ) : ℝ := if x ≠ 0 then x else 0

def f_B (x : ℝ) : ℝ := abs x
def g_B (x : ℝ) : ℝ := if x ≥ 0 then x else 0

def f_C (x : ℝ) : ℝ := abs (x + 1)
def g_C (x : ℝ) : ℝ := if x ≥ -1 then x + 1 else -x - 1

def f_D (x : ℝ) : ℝ := (x + 1) ^ 2
def g_D (x : ℝ) : ℝ := x ^ 2

theorem same_function_C : 
  (∀ x, f_C x = g_C x) ∧
  ((∃ x, f_A x ≠ g_A x) ∧
   (∃ x, f_B x ≠ g_B x) ∧
   (∃ x, f_D x ≠ g_D x)) :=
by
  sorry

end same_function_C_l480_480861


namespace value_of_f_3x_minus_7_l480_480301

def f (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_f_3x_minus_7 (x : ℝ) : f (3 * x - 7) = 9 * x - 16 :=
by
  -- Proof goes here
  sorry

end value_of_f_3x_minus_7_l480_480301


namespace find_a_l480_480596

theorem find_a (x a : ℝ) (h₁ : x^2 + x - 6 = 0) :
  (ax + 1 = 0 → (a = -1/2 ∨ a = -1/3) ∧ ax + 1 ≠ 0 ↔ false) := 
by
  sorry

end find_a_l480_480596


namespace sum_non_solutions_l480_480680

theorem sum_non_solutions (A B C : ℝ) (h : ∀ x, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9) → x ≠ -12) :
  -12 = -12 := 
sorry

end sum_non_solutions_l480_480680


namespace no_prime_q_satisfies_l480_480732

-- Define the base representation to decimal conversion
def base_q_to_decimal (n : ℕ) (q : ℕ) : ℕ :=
  n.digits q.reverse.foldl (λ acc d => acc * q + d) 0

-- Define the given bases in decimal when q is provided
def lhs (q : ℕ) : ℕ :=
  base_q_to_decimal 1012 q + base_q_to_decimal 307 q +
  base_q_to_decimal 114 q + base_q_to_decimal 126 q + 7

def rhs (q : ℕ) : ℕ :=
  base_q_to_decimal 143 q + base_q_to_decimal 272 q + base_q_to_decimal 361 q

-- Define the condition that lhs equals rhs
def satisfies_condition (q : ℕ) : Prop :=
  lhs q = rhs q

-- The final theorem: number of values of q satisfying the condition is zero
theorem no_prime_q_satisfies :
  ∀ q : ℕ, Prime q → ¬ satisfies_condition q :=
by sorry

end no_prime_q_satisfies_l480_480732


namespace lcm_18_27_l480_480014

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l480_480014


namespace count_squares_in_H_l480_480499

def H : set (ℤ × ℤ) :=
  { p | abs p.1 ≤ 5 ∧ abs p.2 ≤ 5 }

def has_side_length_4 (p1 p2 p3 p4 : ℤ × ℤ) : Prop :=
  let d := 4 in
  (p1.1 - p2.1).natAbs = d ∧ (p1.2 - p2.2).natAbs = 0 ∧
  (p1.1 - p4.1).natAbs = 0 ∧ (p1.2 - p4.2).natAbs = d ∧
  (p2.1 - p3.1).natAbs = 0 ∧ (p2.2 - p3.2).natAbs = d ∧
  (p3.1 - p4.1).natAbs = d ∧ (p3.2 - p4.2).natAbs = 0

theorem count_squares_in_H : 
  (∃ s : set (ℤ × ℤ × ℤ × ℤ), s.card = 4 ∧ ∀ (p1 p2 p3 p4 : ℤ × ℤ), p1 ∈ H → p2 ∈ H → p3 ∈ H → p4 ∈ H → has_side_length_4 p1 p2 p3 p4) :=
begin
  sorry
end

end count_squares_in_H_l480_480499


namespace sum_of_first_fifteen_multiples_of_17_l480_480836

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480836


namespace max_area_difference_l480_480806

theorem max_area_difference (l w l' w' : ℕ) (h1 : 2 * (l + w) = 180) (h2 : 2 * (l' + w') = 180) : 
  |(l * w) - (l' * w')| ≤ 1936 := sorry

end max_area_difference_l480_480806


namespace sum_cool_triangle_areas_l480_480084

-- Defining the "cool" right triangle property
def cool_triangle (a b : ℕ) : Prop :=
  (a * b) = 4 * (a + b)

-- Main theorem stating the sum of different possible "cool" right triangle areas is 118
theorem sum_cool_triangle_areas : 
  (∑ (a b : ℕ) in {(5, 20), (6, 12), (8, 8)}, (a * b) / 2) = 118 :=
by
  sorry

end sum_cool_triangle_areas_l480_480084


namespace ellipse_equation_slope_l480_480198

theorem ellipse_equation_slope 
    (a b : ℝ) 
    (h1 : a > b) 
    (h2 : b > 0) 
    (h3 : ∃ e:ℝ, e = 1 / 2) 
    (h4 : ∃ c:ℝ, c = 2) 
    (h5 : a = 4 ∧ b = 2 * Real.sqrt 3 ∧ c = 2) 
    (P Q : ℝ × ℝ)
    (constraint : ∀ k : ℝ, k ∈ {k : ℝ | (3 + 4 * k^2) * k^2 - 8 * k^2 + 3 = 0}) 
    :
    (∃ a b : ℝ, (a = 4 ∧ b = 2 * Real.sqrt 3)) ∧ 
    (P ≠ Q) ∧ 
    ∃ k : ℝ, k = sqrt 3 / 2 ∨ k = - (sqrt 3 / 2) := 
sorry

end ellipse_equation_slope_l480_480198


namespace number_of_k_less_than_11_l480_480304

def f : ℕ → ℕ 
| 0 := 9
| n := if (n % 3 = 0) then f (n - 1) + 3 else f (n - 1) - 1

theorem number_of_k_less_than_11 : 
  {k : ℕ // f k < 11}.card = 5 := 
sorry

end number_of_k_less_than_11_l480_480304


namespace spending_spring_months_l480_480355

theorem spending_spring_months (spend_end_March spend_end_June : ℝ)
  (h1 : spend_end_March = 1) (h2 : spend_end_June = 4) :
  (spend_end_June - spend_end_March) = 3 :=
by
  rw [h1, h2]
  norm_num

end spending_spring_months_l480_480355


namespace student_desserts_l480_480769

theorem student_desserts (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) (equal_distribution : students ≠ 0) :
  mini_cupcakes = 14 → donut_holes = 12 → students = 13 → (mini_cupcakes + donut_holes) / students = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact div_eq_of_eq_mul_right (by norm_num : (13 : ℕ) ≠ 0) (by norm_num : 26 = 2 * 13)
  sorry

end student_desserts_l480_480769


namespace solve_triplet_l480_480540

theorem solve_triplet (x y z : ℕ) (h : 2^x * 3^y + 1 = 7^z) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2) :=
 by sorry

end solve_triplet_l480_480540


namespace find_q_l480_480137

theorem find_q (q : ℚ) : (18^3 = (8^2) / 2 * 3^(9 * q)) → (q = 2 / 3) :=
by
  intro h,
  -- Convert given equation to prime factors
  sorry -- Proof goes here

end find_q_l480_480137


namespace area_triangle_AOB_l480_480490

open Real

-- Definitions and assumptions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus := (1, 0) -- Focus of the parabola y^2 = 4x is (1, 0)
def origin := (0, 0)

-- defining the distance AF = 3
def distance_AF (A : ℝ × ℝ) : Prop := dist A focus = 3

-- Define the main statement that needs to be proven
theorem area_triangle_AOB {A B : ℝ × ℝ} :
  parabola (A.1) (A.2) ∧ parabola (B.1) (B.2) ∧ distance_AF A →
  A ≠ B ∧ A ≠ origin ∧ B ≠ origin →
  let O := origin in
  ∃ S : ℝ, S = abs (A.1 * B.2 - A.2 * B.1) / 2 ∧ S = 3 * sqrt 2 / 2 :=
sorry

end area_triangle_AOB_l480_480490


namespace shoe_store_promotion_l480_480926

theorem shoe_store_promotion:
  ∀ (original_price : ℕ),
  original_price = 360 →
  let BrandA_price := original_price * 65 / 100 in
  let BrandB_price := original_price - ((original_price / 100) * 40) in
  BrandA_price = 234 ∧ BrandB_price = 240 ∧ 240 - 234 = 6 :=
by
  intro original_price
  intro h_price
  let BrandA_price := original_price * 65 / 100
  let BrandB_price := original_price - ((original_price / 100) * 40)
  have h_BrandA : BrandA_price = 234 := by
    rw [h_price]
    norm_num
  have h_BrandB : BrandB_price = 240 := by
    rw [h_price]
    norm_num
  have h_difference : 240 - 234 = 6 := by
    norm_num
  exact ⟨h_BrandA, h_BrandB, h_difference⟩

end shoe_store_promotion_l480_480926


namespace power_mean_inequalities_l480_480329

variable {n : ℕ} (x : Fin n → ℝ)
noncomputable def power_mean (r : ℝ) (x : Fin n → ℝ) : ℝ :=
  if r = 0 then real.exp (1 / n * ∑ i, real.log (x i))
  else (1 / n * ∑ i, (x i) ^ r) ^ (1 / r)

theorem power_mean_inequalities (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) :
  power_mean (-1) x ≤ power_mean 0 x ∧
  power_mean 0 x ≤ power_mean 1 x ∧
  power_mean 1 x ≤ power_mean 2 x :=
sorry

end power_mean_inequalities_l480_480329


namespace find_alpha_l480_480270

variables {G I : Type} {α β : ℝ}

-- Given condition 1: In the triangle ABC, G is the centroid, and I is the incenter.
def is_centroid (A B C G : Type) : Prop := sorry
def is_incenter (A B C I : Type) : Prop := sorry

-- Given condition 2: α and β are the angles at vertices A and B, respectively.
def angle_at_vertices (A B C : Type) (α β : ℝ) : Prop := sorry

-- Given condition 3: IG is parallel to AB.
def parallel (A B C G I : Type) : Prop := sorry

-- Given condition 4: β = 2 * atan(1 / 3)
def beta_condition (β : ℝ) : Prop := β = 2 * real.arctan (1 / 3)

theorem find_alpha (A B C G I : Type) (α β : ℝ)
  [is_centroid A B C G]
  [is_incenter A B C I]
  [angle_at_vertices A B C α β]
  [parallel A B C G I]
  (h : beta_condition β) : α = real.pi / 2 :=
sorry

end find_alpha_l480_480270


namespace solve_for_y_l480_480232

theorem solve_for_y (x y : ℝ) (h : (x + y)^5 - x^5 + y = 0) : y = 0 :=
sorry

end solve_for_y_l480_480232


namespace value_of_expression_l480_480577

theorem value_of_expression (a : ℝ) (h : a^2 - 2 * a = 1) : 3 * a^2 - 6 * a - 4 = -1 :=
by
  sorry

end value_of_expression_l480_480577


namespace moles_of_silver_nitrate_required_l480_480230

-- Define the conditions
variables (moles_HCl : ℕ)
def ratio_silver_nitrate_to_hcl := 1

-- Define the target proof statement
theorem moles_of_silver_nitrate_required (h : moles_HCl = 3) : 
  moles_HCl * ratio_silver_nitrate_to_hcl = 3 :=
begin
  rw h,
  simp [ratio_silver_nitrate_to_hcl],
  sorry

end moles_of_silver_nitrate_required_l480_480230


namespace coefficient_sum_zero_l480_480609

theorem coefficient_sum_zero : 
  let p := (Polynomial.X^2 - Polynomial.X - 6)^3 * (Polynomial.X^2 + Polynomial.X - 6)^3 in
  (p.coeff 1) + (p.coeff 5) + (p.coeff 9) = 0 :=
by {
  let p := (Polynomial.X^2 - Polynomial.X - 6)^3 * (Polynomial.X^2 + Polynomial.X - 6)^3,
  sorry
}

end coefficient_sum_zero_l480_480609


namespace power_A_beta_l480_480583

open Matrix
open Complex

noncomputable def A : Matrix (Fin 2) (Fin 2) ℂ :=
  Matrix.of ![![3, 5], ![0, -2]]

def beta : Vector ℂ (Fin 2) :=
  ![1, -1]

theorem power_A_beta :
  (A^2016) ⬝ beta = ![2^2016, -(2^2016)] :=
by sorry

end power_A_beta_l480_480583


namespace find_numbers_l480_480442

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480442


namespace tan_A_eq_sqrt3_sin_B_eq_4sqrt3_sub_3_div_10_l480_480651

theorem tan_A_eq_sqrt3 {A : ℝ} (h : sin (A + π / 6) = 2 * cos A) : tan A = sqrt 3 := 
by sorry

theorem sin_B_eq_4sqrt3_sub_3_div_10 {A B : ℝ}
  (h1 : sin (A + π / 6) = 2 * cos A)
  (h2 : 0 < B ∧ B < π / 3)
  (h3 : sin (A - B) = 3 / 5) : sin B = (4 * sqrt 3 - 3) / 10 :=
by sorry

end tan_A_eq_sqrt3_sin_B_eq_4sqrt3_sub_3_div_10_l480_480651


namespace disjoint_subset_count_mod_l480_480284

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem disjoint_subset_count_mod :
  let n := (3^10 - 2 * 2^10 + 1) / 2 in
  n % 1000 = 501 := by
  sorry

end disjoint_subset_count_mod_l480_480284


namespace find_a_l480_480157

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end find_a_l480_480157


namespace arithmetic_sequence_problem_l480_480369

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
variable (h_S_n : ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2)

theorem arithmetic_sequence_problem
  (h1 : S_n 5 = 2 * a_n 5)
  (h2 : a_n 3 = -4) :
  a_n 9 = -22 := sorry

end arithmetic_sequence_problem_l480_480369


namespace perimeter_inequality_l480_480752

section
variables {A B C C1 A1 B1 : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space C1] [metric_space A1] [metric_space B1]

-- Definitions of sides and points of tangency
variable (touches : ∀ {x1 x2}, (x1 ≠ x2) → ∃⊥ (t : linear_map ℝ x1 x2), t x1 = x2)
variable (incircle_touches : touches AB C1 ∧ touches BC A1 ∧ touches CA B1)

-- Definitions of inradius and perimeters
variable (r : ℝ) -- Inradius
variable (P : ℝ) -- Perimeter of ΔABC
variable (P1 : ℝ) -- Perimeter of ΔA1B1C1

-- The goal statement
theorem perimeter_inequality (h_incircle : incircle_touches) 
  (h_r_pos : 0 < r) : P + P1 ≥ 9 * sqrt 3 * r := 
sorry
end

end perimeter_inequality_l480_480752


namespace find_numbers_sum_eq_S_product_eq_P_l480_480435

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480435


namespace sum_of_products_l480_480590

noncomputable def a_n (n : ℕ) : ℕ :=
  3^(n-1)

noncomputable def S_n (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (b 1) + n * (n - 1) * d) / 2

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n + 1

noncomputable def T_n (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finRange n, (a i) * (b i)

theorem sum_of_products (n : ℕ) (h_a1: a_n 1 = 1) (h_a6 : a_n 6 = 243) (h_b1: b_n 1 = 3) (h_S5: S_n b_n 5 = 35) :
  T_n a_n b_n n = n * 3^n :=
begin
  sorry
end

end sum_of_products_l480_480590


namespace completing_square_result_l480_480028

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l480_480028


namespace pos_real_solution_l480_480559

noncomputable def x : ℝ := 25 + 2 * Real.sqrt 159

def p (x : ℝ) := x^2 - 40 * x - 8
def q (x : ℝ) := x^2 + 20 * x + 4
def r (x : ℝ) := 25 * (x^2 - 1 / 2)

theorem pos_real_solution :
  (x > 0) ∧ (p x * q x = r x) :=
by
  sorry

end pos_real_solution_l480_480559


namespace tetrahedron_labeling_unique_l480_480548

-- Define the problem statement and conditions
theorem tetrahedron_labeling_unique :
  ∃! (f : Finset (Fin 4) → Finset ℕ), 
    (∀ (v : Finset (Fin 4)), (|v| = 4) ∧ (∀ (i, j ∈ v), i ≠ j → f(i) ≠ f(j)))
    ∧ (∀ (face : Finset (Fin 2)), ∑ (v ∈ face), f(v) = 18)
    ∧ (∀ (v : Finset (Fin 4)), ∃ (σ : Finset.perm (Fin 4)), f(σ v) = f(v))
    :=
begin
  sorry
end

end tetrahedron_labeling_unique_l480_480548


namespace zero_function_l480_480140

def f (x : ℕ) : ℕ' := sorry

theorem zero_function (f : ℕ → ℕ') 
  (h1 : ∀ x y : ℕ, f (x * y) = f x + f y)
  (h2 : f 30 = 0)
  (h3 : ∀ x : ℕ, x % 10 = 7 → f x = 0) :
  ∀ x : ℕ, f x = 0 :=
sorry

end zero_function_l480_480140


namespace largest_s_value_l480_480296

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end largest_s_value_l480_480296


namespace max_area_quadrilateral_APDQ_l480_480928

theorem max_area_quadrilateral_APDQ (s x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ s/2) 
(hP : ∃ P, P ∈ (set.Icc 0 (s / 2)) ∧ AP = 2 * x) 
(hQ : ∃ Q, Q ∈ (set.Icc 0 s) ∧ CQ = x) :
  (∃ x_max, x_max = s / 3 ∧ (area (s / 3)) = (1/2) * s^2) :=
begin
  -- problem set up
  sorry,
end

end max_area_quadrilateral_APDQ_l480_480928


namespace complement_intersection_eq_l480_480468

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 3, 4} := by
  sorry

end complement_intersection_eq_l480_480468


namespace max_equilateral_triangle_area_l480_480370

theorem max_equilateral_triangle_area (length width : ℝ) (h_len : length = 15) (h_width : width = 12) 
: ∃ (area : ℝ), area = 200.25 * Real.sqrt 3 - 450 := by
  sorry

end max_equilateral_triangle_area_l480_480370


namespace exists_divisible_sum_digits_l480_480290

theorem exists_divisible_sum_digits (n k : ℕ) (hn_pos : 0 < n) (hk_pos : 0 < k) 
    (hn_not_div3 : ¬(3 ∣ n)) (hkn : k ≥ n) : 
  ∃ m : ℕ, (n ∣ m) ∧ (nat.digits 10 m).sum = k :=
by
  sorry

end exists_divisible_sum_digits_l480_480290


namespace problem_statement_l480_480300

theorem problem_statement 
  (a b : ℝ)
  (h1 : a ≠ b)
  (h2 : det ![
    ![1, 6, 16], 
    ![4, a, b], 
    ![4, b, a]
  ] = 0) : a + b = 88 :=
by
  /- The proof is omitted -/
  sorry

end problem_statement_l480_480300


namespace simplify_expression_l480_480266

theorem simplify_expression (x y : ℝ) :
  3 * (x + y) ^ 2 - 7 * (x + y) + 8 * (x + y) ^ 2 + 6 * (x + y) = 
  11 * (x + y) ^ 2 - (x + y) :=
by
  sorry

end simplify_expression_l480_480266


namespace radical_axis_halves_perimeter_l480_480056

open EuclideanGeometry Classical 

theorem radical_axis_halves_perimeter 
  {A B C : Point}
  (w_B w_C : Circle)
  (midpoint_AC midpoint_AB : Point)
  (w_B' w_C' : Circle)
  (h_midpoint_AC : midpoint A C = midpoint_AC)
  (h_midpoint_AB : midpoint A B = midpoint_AB)
  (h_symmetric_w_B : symmetric w_B midpoint_AC = w_B')
  (h_symmetric_w_C : symmetric w_C midpoint_AB = w_C') :
  radical_axis w_B' w_C' = line_dividing_perimeter_evenly A B C :=
sorry

end radical_axis_halves_perimeter_l480_480056


namespace correct_sqrt_div_identity_l480_480041

theorem correct_sqrt_div_identity :
  (sqrt 3 / sqrt (1 / 3) = 3) :=
by
  -- Add the proof here
  sorry

end correct_sqrt_div_identity_l480_480041


namespace value_of_a3_l480_480179

variable {a : ℕ → ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

axiom S5 : sum_of_first_n_terms a 5 = 30

-- Question
theorem value_of_a3 (h : is_arithmetic_sequence a) : a 3 = 6 :=
sorry

end value_of_a3_l480_480179


namespace trig_product_sign_l480_480977

theorem trig_product_sign :
  (sin 2 > 0) →
  (cos 3 < 0) →
  (tan 4 > 0) →
  sin 2 * cos 3 * tan 4 < 0 :=
by
  intros h1 h2 h3
  sorry

end trig_product_sign_l480_480977


namespace square_diagonal_length_l480_480404

theorem square_diagonal_length (A : ℝ) (hA : A = 392) : ∃ d : ℝ, d = 28 :=
by
  let s := real.sqrt A
  have hs : s * s = A := by simp [s, real.sq_sqrt (le_of_eq hA)]
  let d := real.sqrt (2 * s * s)
  have hd : d = 28 := by sorry
  use d,
  exact hd

end square_diagonal_length_l480_480404


namespace find_x_l480_480051

-- Define the side lengths and areas of the squares
def s1 (x : ℝ) : ℝ := Real.sqrt (x^2 + 4 * x + 4)
def s2 (x : ℝ) : ℝ := Real.sqrt (4 * x^2 - 12 * x + 9)

-- Define the perimeter sum condition
def perimeter_sum_condition (x : ℝ) : Prop :=
  4 * (s1 x) + 4 * (s2 x) = 32

-- Prove that x = 3 under the given conditions
theorem find_x (x : ℝ) (h1 : s1 x = x + 2)
                        (h2 : s2 x = 2 * x - 3)
                        (h3 : perimeter_sum_condition x) : x = 3 :=
by
  sorry

end find_x_l480_480051


namespace increasing_neg_infty_to_zero_solve_log_integral_inequality_range_t_for_log_inequality_l480_480683

/-
Problem 1: Prove f(x) is increasing on (-∞, 0) given that
- f is an odd function.
- f is increasing on (0, ∞).
-/
theorem increasing_neg_infty_to_zero {f : ℝ → ℝ} (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_increasing : ∀ x y : ℝ, 0 < x → x < y → f x < f y) :
  ∀ x y : ℝ, x < y → y < 0 → f x < f y :=
sorry

/-
Problem 2: Prove the solution to log(1 - x^2) + 1 > 0 is
x ∈ (-√(1 - 1/e), √(1 - 1/e)).
-/
theorem solve_log_integral_inequality (x : ℝ) :
  log (1 - x^2) + 1 > 0 ↔ x ∈ Ioo (-sqrt (1 - 1/exp 1)) (sqrt (1 - 1/exp 1)) :=
sorry

/-
Problem 3: Prove the range for t such that  log_4 |f(t) + 1| > 0 given
- f(m * n) = f(m) + f(n).
- f(-2) = -1.
-/
theorem range_t_for_log_inequality {f : ℝ → ℝ}
  (h_mul_add : ∀ m n : ℝ, 0 < m → 0 < n → f (m * n) = f m + f n) (h_f_neg2 : f (-2) = -1) :
  ∀ t : ℝ, logb 4 (abs (f t + 1)) > 0 ↔ t ∈ Ioo (-∞) (-2) ∪ Ioo (-2) (-1) ∪ Ioo (1/4) (1/2) ∪ Ioo (1/2) ∞ :=
sorry

end increasing_neg_infty_to_zero_solve_log_integral_inequality_range_t_for_log_inequality_l480_480683


namespace angle_half_l480_480656

-- Definitions for the problem conditions
variables {O A B C D M K P Q : Point}
variables [Circle O]
variables (h_eq_chords : chord_length A B = chord_length C D ∧ chord_length A B = chord_length P Q)
variables (h_perp_bisector_OM : is_perpendicular_bisector O M P Q)
variables (h_perp_bisector_OK : is_perpendicular_bisector O K C D)

-- Main theorem statement
theorem angle_half (h_center : is_center O)
  (h_chords : ∀ (A B C D P Q : Point), chord_length A B = chord_length C D ∧ chord_length A B = chord_length P Q)
  (h_perp_bisector_OM : is_perpendicular_bisector O M P Q)
  (h_perp_bisector_OK : is_perpendicular_bisector O K C D) :
  angle O M K = (1/2 : ℝ) * angle B L D := sorry

end angle_half_l480_480656


namespace derivative_of_y_correct_l480_480055

noncomputable def derivative_of_y (x : ℝ) : ℝ :=
  let y := (4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))) / (16 + (Real.log 4) ^ 2)
  let u := 4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))
  let v := 16 + (Real.log 4) ^ 2
  let du_dx := (4^x * Real.log 4) * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x)) +
               (4^x) * (4 * Real.log 4 * Real.cos (4 * x) + 16 * Real.sin (4 * x))
  let dv_dx := 0
  (du_dx * v - u * dv_dx) / (v ^ 2)

theorem derivative_of_y_correct (x : ℝ) : derivative_of_y x = 4^x * Real.sin (4 * x) :=
  sorry

end derivative_of_y_correct_l480_480055


namespace arcsin_of_neg_one_l480_480116

theorem arcsin_of_neg_one : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_of_neg_one_l480_480116


namespace length_increase_is_30_percent_l480_480359

theorem length_increase_is_30_percent (L B : ℝ) (h1 : L > 0) (h2 : B > 0)
  (h_new_breadth : ∃ B' : ℝ, B' = 1.45 * B)
  (h_new_area : ∃ A' : ℝ, A' = 1.885 * (L * B)) :
  ∃ x : ℝ, x = 30 :=
by
  -- Given/assume the new breadth and area conditions
  rcases h_new_breadth with ⟨B', h_B'⟩
  rcases h_new_area with ⟨A', h_A'⟩
  
  -- Define the new length in terms of percentage increase
  let x := 30
  let L' := L * (1 + x / 100)
  
  have h_eq1 : L' * B' = A' := by
    rw [h_B', h_A']
    
  have h_eq2 : (L * (1 + x / 100)) * (1.45 * B) = 1.885 * (L * B) := by
    rw h_eq1
    
  have h_eq3 : (1 + x / 100) * 1.45 = 1.885 := by
    rw [←h_eq2, h1, h2]
    
  have h_eq4 : 1 + x / 100 = 1.3 := by
    exact eq_div_of_mul_eq_of_ne_zero _ _ h_eq3 (ne_of_gt h1)
    
  have h_final : x / 100 = 0.3 := by
    linarith
    
  exact ⟨x, by linarith⟩

end length_increase_is_30_percent_l480_480359


namespace factorize_expr_l480_480986

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l480_480986


namespace coefficient_x3_expansion_l480_480560

theorem coefficient_x3_expansion (x : ℝ) :
  (∃ c : ℝ, ((2 * x^2 + x - 1)^5).coeff 3 = c) ∧ c = -30 :=
by
  sorry

end coefficient_x3_expansion_l480_480560


namespace seashells_total_l480_480675

theorem seashells_total :
  (joan_seashells jessica_seashells jeremy_seashells : ℕ)
  (h1 : joan_seashells = 6) (h2 : jessica_seashells = 8) (h3 : jeremy_seashells = 12) :
  joan_seashells + jessica_seashells + jeremy_seashells = 26 :=
by
  sorry

end seashells_total_l480_480675


namespace inequality_problem_l480_480195

theorem inequality_problem (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_sum : a + b + c + d = 4) : 
    a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := 
sorry

end inequality_problem_l480_480195


namespace tangent_line_at_0_g_monotonic_f_super_additive_l480_480615

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)
noncomputable def g (x : ℝ) : ℝ := (deriv f) x

-- Problem (Ⅰ)
theorem tangent_line_at_0 : 
    let p := (0, f 0),
        m := (deriv f) 0 in
    (p.snd = f 0) ∧ (m = 1) ∧ ((∀ x, f x - m * x - (p.snd - m * p.fst) = 0) ∧ ∃ x₀, (0, f 0) = (x₀, f x₀)) :=
by
   sorry

-- Problem (Ⅱ)
theorem g_monotonic : ∀ x ∈ set.Ici (0 : ℝ), monotone g :=
by
   sorry

-- Problem (Ⅲ)
theorem f_super_additive (s t : ℝ) (hs : s > 0) (ht : t > 0) : 
    f (s + t) > f s + f t :=
by
   sorry

end tangent_line_at_0_g_monotonic_f_super_additive_l480_480615


namespace tangent_line_eqn_g_strictly_decreasing_l480_480617

noncomputable def f : ℝ → ℝ := λ x, Real.sin x

noncomputable def P : ℝ × ℝ := (Real.pi / 4, f (Real.pi / 4))

noncomputable def g (m : ℝ) : ℝ → ℝ := λ x, m * x - x^3 / 6

theorem tangent_line_eqn : 
  (y - (Real.sqrt 2 / 2) = (Real.sqrt 2 / 2) * (x - Real.pi / 4)) → (x - Real.sqrt 2 * y + 1 - Real.pi / 4 = 0) :=
sorry

theorem g_strictly_decreasing : ∀ (m : ℝ), 
  (m ≤ 0 → ∀ x : ℝ, g m x < g m (x + 1)) ∧ 
  (m > 0 → ∀ x : ℝ, (x < -Real.sqrt (2 * m) ∨ x > Real.sqrt (2 * m)) → g m x < g m (x + 1)) :=
sorry

end tangent_line_eqn_g_strictly_decreasing_l480_480617


namespace matrix_projection_property_l480_480148

noncomputable def Q : Matrix (Fin 3) (Fin 3) ℚ := ![
  [1/6, 1/6, 1/3],
  [1/6, 1/6, 1/3],
  [1/3, 1/3, 2/3]
]

def projection (v : Fin 3 → ℚ) : Fin 3 → ℚ :=
  let u : Fin 3 → ℚ := ![1, 1, 2]
  let dotP := (v 0) * (u 0) + (v 1) * (u 1) + (v 2) * (u 2)
  let dotU := 1^2 + 1^2 + 2^2
  let alpha := dotP / dotU
  ![alpha * 1, alpha * 1, alpha * 2]

theorem matrix_projection_property (v : Fin 3 → ℚ) :
  (Q.mulVec v) = projection v := by
  sorry

end matrix_projection_property_l480_480148


namespace no_regular_n_plus_one_gon_cross_section_l480_480712

-- Lean theorem statement for the given problem
theorem no_regular_n_plus_one_gon_cross_section (n : ℕ) (hn : n ≥ 3):
  ¬ ∃ (P : set (ℝ × ℝ)), is_cross_section_of_regular_ngon_prism P (n + 1) :=
sorry

-- Definitions (these definitions need to be supported appropriately in Lean; 
-- here they are placeholders to fit the theorem statement structure)
def is_cross_section_of_regular_ngon_prism (P : set (ℝ × ℝ)) (sides : ℕ) : Prop :=
  sorry

end no_regular_n_plus_one_gon_cross_section_l480_480712


namespace EddyJourneyTime_l480_480131

-- Definitions of conditions
def FreddyTime := 4
def DistanceAB := 480
def DistanceAC := 300
def SpeedRatio := 2.1333333333333333

-- Objective
theorem EddyJourneyTime : (DistanceAB / (DistanceAB / T)) * 4 = DistanceAC * SpeedRatio → T = 3 := sorry

end EddyJourneyTime_l480_480131


namespace y_coordinate_eq_l480_480006

theorem y_coordinate_eq (y : ℝ) : 
  (∃ y, (√(9 + y^2) = √(4 + (5 - y)^2))) ↔ (y = 2) :=
by
  sorry

end y_coordinate_eq_l480_480006


namespace frisbee_league_tournament_committees_l480_480658

theorem frisbee_league_tournament_committees :
  let total_committees :=
    (Finset.card (Finset.univ : Finset (Fin 4))) *
    Nat.choose 8 4 *
    Nat.choose 8 2 ^ 3
  in total_committees = 6593280 :=
by
  let host_teams := Finset.univ : Finset (Fin 4)
  have host_count : Finset.card host_teams = 4 := by simp
  let host_selection := Nat.choose 8 4
  let non_host_selection := Nat.choose 8 2 ^ 3
  calc
    total_committees
        = host_count * host_selection * non_host_selection
        : by rfl
    ... = 4 * 70 * (28^3) : by simp [host_selection, non_host_selection]
    ... = 4 * 70 * 21952 : by norm_num
    ... = 6593280 : by norm_num
  sorry

end frisbee_league_tournament_committees_l480_480658


namespace triangle_product_l480_480671

-- Definitions for points A, B, C, P, Q, D in triangle ABC
variables {A B C P Q D : Type*}

-- Main problem statement as a Lean theorem
theorem triangle_product (hP_equidistant : dist A P = dist B P)
  (hangle_APB_2ACB : ∠ A P B = 2 * ∠ A C B)
  (hPQ : dist P Q = 1)
  (hintersect_AC_BQ_D : lies_on_inter (line_through A C) (line_through B Q) D)
  (hPB : dist P B = 4)
  (hPD : dist P D = 3) :
  dist A D * dist C D = 21 :=
sorry

end triangle_product_l480_480671


namespace students_taking_history_l480_480257

-- Defining the conditions
def num_students (total_students history_students statistics_students both_students : ℕ) : Prop :=
  total_students = 89 ∧
  statistics_students = 32 ∧
  (history_students + statistics_students - both_students) = 59 ∧
  (history_students - both_students) = 27

-- The theorem stating that given the conditions, the number of students taking history is 54
theorem students_taking_history :
  ∃ history_students, ∃ statistics_students, ∃ both_students, 
  num_students 89 history_students statistics_students both_students ∧ history_students = 54 :=
by
  sorry

end students_taking_history_l480_480257


namespace problem1_problem2_l480_480587

open Set

noncomputable def A := {x : ℝ | x < -3 ∨ x ≥ 2}
noncomputable def B (a : ℝ) := {x : ℝ | x ≤ a - 3}
noncomputable def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

-- Problem 1: Prove that when a=2, (complement_R A) ∩ (B 2) = {x | -3 ≤ x ∧ x ≤ -1}
theorem problem1 : 
  let a := 2
  in let B_2 := B a
  in (complement_R A) ∩ B_2 = {x : ℝ | -3 ≤ x ∧ x ≤ -1} :=
sorry

-- Problem 2: Prove that if A ∩ B = B, then a < 0
theorem problem2 (a : ℝ) :
  (A ∩ (B a) = B a) → a < 0 :=
sorry

end problem1_problem2_l480_480587


namespace tim_works_6_days_a_week_l480_480801

-- Define the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def weekly_earnings : ℝ := 720

-- Define the calculation based on the conditions
def daily_earnings : ℝ := tasks_per_day * pay_per_task
def days_worked_per_week (weekly_earnings daily_earnings : ℝ) : ℝ := weekly_earnings / daily_earnings

-- Prove that Tim works 6 days a week
theorem tim_works_6_days_a_week : days_worked_per_week weekly_earnings daily_earnings = 6 := by
  sorry

end tim_works_6_days_a_week_l480_480801


namespace loss_percentage_approx_4_17_l480_480909

theorem loss_percentage_approx_4_17 :
  let C := 1 / 34.56,
      sp36 := 1 / 36,
      loss := C - sp36,
      loss_percentage := (loss / C) * 100
  in abs (loss_percentage - 4.17) < 0.01 := 
by
  let C := 1 / 34.56
  let sp36 := 1 / 36
  let loss := C - sp36
  let loss_percentage := (loss / C) * 100
  sorry

end loss_percentage_approx_4_17_l480_480909


namespace semicubical_parabola_arc_length_l480_480961

theorem semicubical_parabola_arc_length :
  let y := λ x : ℝ, x^(3 / 2)
  2 * (∫ (x : ℝ) in 0..5, sqrt (1 + (abs (3 / 2 * x^(1 / 2))) ^ 2)) = 670 / 27 :=
by
  let y := λ x : ℝ, x^(3 / 2)
  have h_deriv : derivative y = λ x, (3 / 2) * x^(1 / 2) := sorry
  have arc_length := ∫ (x : ℝ) in 0..5, sqrt (1 + (abs (3 / 2 * x^(1 / 2))) ^ 2) := sorry
  have double_arc_length := 2 * (∫ (x : ℝ) in 0..5, sqrt (1 + (abs (3 / 2 * x^(1 / 2))) ^ 2)) := sorry
  exact double_arc_length = 670 / 27

end semicubical_parabola_arc_length_l480_480961


namespace range_of_m_l480_480185

variable (m : ℝ) -- variable m in the real numbers

-- Definition of proposition p
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

-- Definition of proposition q
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- The theorem statement with the given conditions
theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l480_480185


namespace calculate_a_mul_a_sub_3_l480_480108

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l480_480108


namespace find_missing_number_l480_480068

theorem find_missing_number (x : ℕ) (h : x * 240 = 173 * 240) : x = 173 :=
sorry

end find_missing_number_l480_480068


namespace pictures_per_day_calc_l480_480279

def years : ℕ := 3
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

def number_of_cards : ℕ := total_spent / cost_per_card
def total_images : ℕ := number_of_cards * images_per_card
def days_in_year : ℕ := 365
def total_days : ℕ := years * days_in_year

theorem pictures_per_day_calc : 
  (total_images / total_days) = 10 := 
by
  sorry

end pictures_per_day_calc_l480_480279


namespace AIME_inequality_l480_480690

theorem AIME_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 1 ≤ x i) :
  (∑ i, (1/(x i + 1)))  ≥ n / (1 + (∏ i, x i) ^ (1 / n : ℝ)) :=
sorry

end AIME_inequality_l480_480690


namespace bathroom_square_footage_l480_480474

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l480_480474


namespace sum_first_fifteen_multiples_of_17_l480_480813

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480813


namespace wholesale_costs_l480_480922

noncomputable def wholesale_cost_s_sleeping_bag : ℚ := 28 / 1.15
noncomputable def wholesale_cost_s_tent : ℚ := 80 / 1.20

theorem wholesale_costs :
  wholesale_cost_s_sleeping_bag ≈ 24.35 ∧
  wholesale_cost_s_tent ≈ 66.67 :=
by
  -- We need to show that the approximate values match the given conditions.
  sorry

end wholesale_costs_l480_480922


namespace ordering_of_powers_l480_480011

theorem ordering_of_powers :
  (3:ℕ)^15 < 10^9 ∧ 10^9 < (5:ℕ)^13 :=
by
  sorry

end ordering_of_powers_l480_480011


namespace find_principal_sum_l480_480871

theorem find_principal_sum 
  (CI SI P : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (hCI : CI = 11730) 
  (hSI : SI = 10200) 
  (hT : T = 2) 
  (hCI_formula : CI = P * ((1 + R / 100)^T - 1)) 
  (hSI_formula : SI = (P * R * T) / 100) 
  (h_diff : CI - SI = 1530) :
  P = 34000 := 
by 
  sorry

end find_principal_sum_l480_480871


namespace no_14_consecutive_divisible_primes_less_than_13_exists_21_consecutive_divisible_primes_less_than_17_l480_480545

open Nat

theorem no_14_consecutive_divisible_primes_less_than_13 :
    ¬ ∃ s : ℕ, ∀ i : ℕ, i < 14 → (∃ p ∈ {2, 3, 5, 7, 11}, (s + i) % p = 0) := sorry

theorem exists_21_consecutive_divisible_primes_less_than_17 :
    ∃ s : ℕ, ∀ i : ℕ, i < 21 → (∃ p ∈ {2, 3, 5, 7, 11, 13}, (s + i) % p = 0) := sorry

end no_14_consecutive_divisible_primes_less_than_13_exists_21_consecutive_divisible_primes_less_than_17_l480_480545


namespace probability_of_roots_l480_480916

theorem probability_of_roots (k : ℝ) (h : 3 ≤ k ∧ k ≤ 8) :
  let a := k^2 - 2k - 3
  let b := 3k - 5
  let c := 2
  let discriminant := b^2 - 4*a*c
  let x1 := (-b + real.sqrt discriminant) / (2*a)
  let x2 := (-b - real.sqrt discriminant) / (2*a)
  a ≠ 0 → discriminant ≥ 0 →
  (x1 ≤ 2 * x2 ∨ x2 ≤ 2 * x1) →
  (probability := \(\frac{interval_length [3, 13/3]}{interval_length [3, 8}}\)) →
  probability = 4 / 15 := by
sorry

end probability_of_roots_l480_480916


namespace batsman_average_increases_l480_480067

theorem batsman_average_increases
  (score_17th: ℕ)
  (avg_increase: ℕ)
  (initial_avg: ℕ)
  (final_avg: ℕ)
  (initial_innings: ℕ):
  score_17th = 74 →
  avg_increase = 3 →
  initial_innings = 16 →
  initial_avg = 23 →
  final_avg = initial_avg + avg_increase →
  (final_avg * (initial_innings + 1) = score_17th + (initial_avg * initial_innings)) →
  final_avg = 26 :=
by
  sorry

end batsman_average_increases_l480_480067


namespace sum_first_fifteen_multiples_of_17_l480_480817

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480817


namespace general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l480_480178

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Noncomputable sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}
variable (c : ℤ)

axiom h1 : is_arithmetic_sequence a d
axiom h2 : d > 0
axiom h3 : a 1 * a 2 = 45
axiom h4 : a 0 + a 4 = 18

-- General formula for the nth term
theorem general_formula_for_nth_term :
  ∃ a1 d, a 0 = a1 ∧ d > 0 ∧ (∀ n, a n = a1 + n * d) :=
sorry

-- Arithmetic sequence from Sn/(n+c)
theorem exists_c_makes_bn_arithmetic :
  ∃ (c : ℤ), c ≠ 0 ∧ (∀ n, n > 0 → (arithmetic_sum a n) / (n + c) - (arithmetic_sum a (n - 1)) / (n - 1 + c) = d) :=
sorry

end general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l480_480178


namespace solution_set_l480_480765

def within_bounds (x : ℝ) : Prop := |2 * x + 1| < 1

theorem solution_set : {x : ℝ | within_bounds x} = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end solution_set_l480_480765


namespace find_numbers_l480_480443

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480443


namespace proof_problem_l480_480855

theorem proof_problem
  (x y z : ℤ)
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 3 * y * z + 3)
  (h3 : 13 * y - x = 1) :
  z = 8 := by
  sorry

end proof_problem_l480_480855


namespace five_tuesdays_in_july_implies_five_thursdays_in_august_l480_480731

theorem five_tuesdays_in_july_implies_five_thursdays_in_august
  (N : ℕ)
  (july_has_five_tuesdays : ∀ (d : ℕ), d ∈ {1, 8, 15, 22, 29} ∨ d ∈ {2, 9, 16, 23, 30} ∨ d ∈ {3, 10, 17, 24, 31})
  (july_has_31_days : true)
  (august_has_31_days : true) : 
  ∃ (d : ℕ), d ∈ {3, 10, 17, 24, 31} :=
sorry

end five_tuesdays_in_july_implies_five_thursdays_in_august_l480_480731


namespace logo_new_height_l480_480674

theorem logo_new_height
    (original_width : ℝ) (original_height : ℝ)
    (new_width : ℝ) (proportions_kept : Prop)
    (h1 : original_width = 3)
    (h2 : original_height = 2)
    (h3 : new_width = 15)
    (h4 : proportions_kept ↔ (original_width / original_height = new_width / ?new_height)) :
    (?new_height = 10) :=
by
  sorry  

end logo_new_height_l480_480674


namespace mean_quiz_score_l480_480803

theorem mean_quiz_score : 
  let scores := [88, 90, 94, 86, 85, 91],
      sum_scores := List.sum scores,
      num_quizzes := List.length scores
  in sum_scores / num_quizzes = 89 := by
  let scores := [88, 90, 94, 86, 85, 91]
  let sum_scores := List.sum scores
  let num_quizzes := List.length scores
  have h1 : sum_scores = 534 := by sorry
  have h2 : num_quizzes = 6 := by sorry
  have h3 : sum_scores / num_quizzes = 534 / 6 := by sorry
  show 534 / 6 = 89 from sorry

end mean_quiz_score_l480_480803


namespace area_triangle_ABC_range_sin_square_A_plus_C_l480_480653

variables (A B C a b c : ℝ)
variables (m n : ℝ × ℝ)

-- Conditions
def triangle_ABC_opposite_sides :=
  ∃ A B C a b c,
  m = (Real.cos B, Real.cos C) ∧
  n = (2 * a + c, b) ∧
  (m.1 * n.1 + m.2 * n.2 = 0)

-- Question 1: Area of the triangle given the conditions
def area_of_triangle (a c b : ℝ) : ℝ :=
  if h : (a + c = 4) ∧ (b = Real.sqrt 13) then
    3 * Real.sqrt 3 / 4
  else
    0

-- Question 2: Range of sin²A + sin²C
def range_of_sin_square_A_plus_C (C : ℝ) : set ℝ :=
  {y | 0 < C ∧ C < Real.pi / 3 ∧ 1/2 ≤ y ∧ y < 3/4}

-- Theorem statements
theorem area_triangle_ABC : 
  triangle_ABC_opposite_sides ∧ (a + c = 4) ∧ (b = Real.sqrt 13) → 
  area_of_triangle a c b = 3 * Real.sqrt 3 / 4 := sorry

theorem range_sin_square_A_plus_C : 
  (B = 2 * Real.pi / 3) → 
  range_of_sin_square_A_plus_C C = {y | 1/2 ≤ y ∧ y < 3/4} := sorry

end area_triangle_ABC_range_sin_square_A_plus_C_l480_480653


namespace magnitude_z_eq_sqrt5_l480_480467

noncomputable def magnitude_of_z (a : ℝ) : ℂ := abs (a + complex.I)

theorem magnitude_z_eq_sqrt5 (a : ℝ) (h1 : (a + complex.I)^2 + (a + complex.I) = 1 - 3 * complex.I) :
  magnitude_of_z a = real.sqrt 5 :=
by
  sorry

end magnitude_z_eq_sqrt5_l480_480467


namespace trig_expression_through_point_l480_480249

theorem trig_expression_through_point (t : ℝ) (ht : t ≠ 0):
  ∃ θ : ℝ, 2 * sin θ + cos θ = if t > 0 then (2 / 5) else -(2 / 5) :=
by
  sorry

end trig_expression_through_point_l480_480249


namespace trench_digging_l480_480930

theorem trench_digging():
  ∃ (t : ℝ)
  (r : ℝ)
  (T := 40)
  (n := 6)
  (q := 1/4),
  (5 * r * t = 10) ∧
  (6 * (r * t) = 12) ∧
  (((n * r * t) * (T / (6 * t))) = 100)
  :=
by
  sorry

end trench_digging_l480_480930


namespace max_pieces_two_cuts_correct_l480_480267

-- Definitions based on conditions
def is_max_squares_intersected_by_single_cut (n : ℕ) : Prop :=
  n = 5

def pieces_after_one_cut (squares_intersected : ℕ) : ℕ :=
  2 * squares_intersected + 4

def max_pieces_two_cuts (grid_rows grid_cols : ℕ) : ℕ :=
  if grid_rows = 3 ∧ grid_cols = 3 then 14 else 0

-- Lean 4 statement of the problem
theorem max_pieces_two_cuts_correct :
  ∀ (grid_rows grid_cols : ℕ),
  grid_rows = 3 → grid_cols = 3 →
  is_max_squares_intersected_by_single_cut 5 →
  max_pieces_two_cuts grid_rows grid_cols = 14 :=
begin
  intro grid_rows,
  intro grid_cols,
  intro h_rows,
  intro h_cols,
  intro h_squares,
  unfold is_max_squares_intersected_by_single_cut at h_squares,
  unfold max_pieces_two_cuts,
  split_ifs with h,
  { exact rfl },
  { contradiction }
end

end max_pieces_two_cuts_correct_l480_480267


namespace find_numbers_l480_480451

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480451


namespace upper_limit_in_range_l480_480784

theorem upper_limit_in_range (h : ∃ n : ℕ, 11 ≤ n ∧ n ≤ 12 ∧ n = ⌊11.9⌋) : 
    ∃ upper_limit : ℕ, upper_limit ≥ 100 ∧ (∃ k : ℕ, k * 42 = upper_limit) ∧ ∀ m ≥ 100, m < upper_limit → (∃ k : ℕ, m = k * 42) := 
    upper_limit = 546 :=
by
  sorry

end upper_limit_in_range_l480_480784


namespace diamond_14_3_l480_480569

def diamond (p q : ℝ) : ℝ := 
  if q = 0 then p 
  else if p = 0 then p 
  else sorry -- placeholder for the full definition according to conditions

axiom diamond_comm (p q : ℝ) : diamond p q = diamond q p
axiom diamond_p0 (p : ℝ) : diamond p 0 = p
axiom diamond_p2q (p q : ℝ) : diamond (p + 2) q = diamond p q + 2 * q + 3

theorem diamond_14_3 : diamond 14 3 = 84 :=
sorry

end diamond_14_3_l480_480569


namespace problem_statement_l480_480223

def A : set ℝ := { x | x + 1 ≥ 0 }
def B : set ℝ := { x | x^2 - 2 ≥ 0 }
def complement_B : set ℝ := { x | -sqrt 2 < x ∧ x < sqrt 2 }

theorem problem_statement : A ∩ complement_B = { x | -1 ≤ x ∧ x < sqrt 2 } := sorry

end problem_statement_l480_480223


namespace volume_divided_by_pi_result_l480_480485

noncomputable def radius := 15
noncomputable def angle := 270 -- degrees

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def arc_length (a : ℝ) (c : ℝ) : ℝ := a / 360 * c
def base_radius (arc : ℝ) : ℝ := arc / (2 * Real.pi)
def height (r slant_height : ℝ) : ℝ := Real.sqrt (slant_height^2 - r^2)
def volume (r h : ℝ) : ℝ := 1 / 3 * Real.pi * r^2 * h

theorem volume_divided_by_pi_result :
  let r_base := base_radius (arc_length angle (circumference radius))
      h_cone := height r_base radius in
  volume r_base h_cone / Real.pi = 126.5625 * Real.sqrt 11 :=
by
  let r_base := base_radius (arc_length angle (circumference radius))
  let h_cone := height r_base radius
  have h1 : circumference radius = 30 * Real.pi := sorry
  have h2 : arc_length angle (circumference radius) = 22.5 * Real.pi := sorry
  have h3 : r_base = 11.25 := sorry
  have h4 : h_cone = Real.sqrt 98.4375 := sorry
  have h5 : volume r_base h_cone = 126.5625 * Real.sqrt 11 * Real.pi := sorry
  show volume r_base h_cone / Real.pi = 126.5625 * Real.sqrt 11 from
    sorry

end volume_divided_by_pi_result_l480_480485


namespace expected_yield_l480_480313

noncomputable def garden_length_steps : ℕ := 10
noncomputable def garden_width_steps : ℕ := 30
noncomputable def step_length_feet : ℝ := 3
noncomputable def suitable_area_percentage : ℝ := 0.9
noncomputable def yield_per_sq_foot : ℝ := 3 / 4

theorem expected_yield :
  let l_feet := garden_length_steps * step_length_feet in
  let w_feet := garden_width_steps * step_length_feet in
  let area := l_feet * w_feet in
  let suitable_area := suitable_area_percentage * area in
  let yield := suitable_area * yield_per_sq_foot in
  yield = 1822.5 :=
by
  sorry

end expected_yield_l480_480313


namespace range_m_l480_480612

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem range_m (m : ℝ) : 
  (∃ x ∈ Icc (2 : ℝ) (4 : ℝ), m - f x > 0) ↔ m > 5 :=
by
  sorry

end range_m_l480_480612


namespace product_of_complex_conjugates_l480_480192

theorem product_of_complex_conjugates (i : ℂ) (h : i^2 = -1) : (1 + i) * (1 - i) = 2 :=
by
  sorry

end product_of_complex_conjugates_l480_480192


namespace main_theorem_l480_480679

def table_size (p : ℕ) : ℕ := p^2 + p + 1
def max_colored_cells (p : ℕ) : ℕ := (p + 1) * (table_size p)
def is_nice_coloring {p : ℕ} (T : Fin (table_size p) × Fin (table_size p) → Bool) : Prop :=
  ∀ (i1 i2 j1 j2 : Fin (table_size p)), i1 ≠ i2 ∧ j1 ≠ j2 →
    ¬(T (i1, j1) ∧ T (i1, j2) ∧ T (i2, j1) ∧ T (i2, j2))

def can_partition_tuples (p : ℕ) (S : Fin (table_size p) → (Fin p × Fin p × Fin p) → Prop) : Prop :=
  ∀ (a b c : Fin p), (a + b + c > 0) →
    ∃ (i : Fin (table_size p)), S i (a, b, c) ∧ 
   ( ∀ j, S i (ka, kb, kc) ↔ a + j ≡ k a, b + j ≡ k b, c + j ≡ k c [MOD p])

def coloring {p : ℕ} (S : Fin (table_size p) → (Fin p × Fin p × Fin p) → Prop)
  (T : Fin (table_size p) × Fin (table_size p) → Bool) : 
  ∀ i j, (∑ (a : Fin p) (b : Fin p) (c : Fin p), S i (a, b, c) * S j (a, b, c) ) ≡ 0  [MOD p]  → T (i, j) := sorry

theorem main_theorem (p : ℕ) [Fact (Nat.Prime p)] :
  ∀ (T : Fin (table_size p) × Fin (table_size p) → Bool), 
  (∀i j, T (i,j) ) → 
  is_nice_coloring T → (∀ (S : Fin (table_size p) → (Fin p × Fin p × Fin p) → Prop), can_partition_tuples p S ) → 
  ∃ k, ∑ (i : Fin (table_size p)) (j : Fin (table_size p)), if T (i, j) then 1 else 0 ≤ max_colored_cells p := 
sorry

end main_theorem_l480_480679


namespace manuscript_year_possibilities_l480_480074

noncomputable def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrange_digits (total : Nat) (repeat : Nat) : Nat :=
  factorial total / (factorial repeat)

def number_of_possibilities : Nat :=
  let first_digit_choices := 2
  let remaining_arrangements := arrange_digits 6 4
  first_digit_choices * remaining_arrangements

theorem manuscript_year_possibilities : number_of_possibilities = 60 :=
by
  sorry

end manuscript_year_possibilities_l480_480074


namespace completing_square_result_l480_480029

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l480_480029


namespace correct_calculation_l480_480040

-- Define the conditions
def option_A : Prop := sqrt 2 + sqrt 3 = sqrt 5
def option_B : Prop := sqrt 2 * sqrt 3 = sqrt 6
def option_C : Prop := 3 * sqrt 2 - sqrt 2 = 3
def option_D : Prop := sqrt 10 / sqrt 5 = 2

-- Theorem statement
theorem correct_calculation :
  ¬ option_A ∧ option_B ∧ ¬ option_C ∧ ¬ option_D :=
by
  -- Omitted proof details
  sorry

end correct_calculation_l480_480040


namespace exists_perpendicular_trace_l480_480504

-- Define what we mean by traces and perpendicular planes.
def Plane (α : Type*) := α → Prop
def trace (α : Type*) := α → α
def isPerpendicular (α : Type*) [InnerProductSpace α] (p1 p2 : Plane α) : Prop :=
  ∀ x, p1 x → p2 x → inner x x = 0

-- Given the first traces s1, s1x, prove that there exists a second trace s2 such that the planes are perpendicular.
theorem exists_perpendicular_trace {α : Type*} [InnerProductSpace α]
  (s1 s1x : trace α) (p1 : Plane α) :
  ∃ s2 : trace α, isPerpendicular α p1 (Plane.mk (s1 s1 ∩ s2 s2)) :=
sorry

end exists_perpendicular_trace_l480_480504


namespace beautiful_iff_quad_removal_l480_480283

universe u

structure Polygon (α : Type u) :=
(vertices : list α)
(is_convex : Prop)

structure Triangle (α : Type u) :=
(vertices : list α)
(is_sublist : vertices ⊆ Polygon.vertices)

structure Triangulation (α : Type u) :=
(polygon : Polygon α)
(triangles : list (Triangle α))
(non_intersecting : Prop)

structure BeautifulTriangulation (α : Type u) extends Triangulation α :=
(beautiful_condition : ∀ T ∈ triangles, 
  let removed_effect := remove_triangle_from_polygon polygon T 
  in count_odd_polygons removed_effect = 1)

def remove_diagonals_to_quadrilaterals (α : Type u) (t : Triangulation α) : Prop := 
  ∃ removed_diagonals, 
    (∀ poly ∈ (polygon_after_removal t.polygon removed_diagonals), 
      polygon_is_quadrilateral poly)

theorem beautiful_iff_quad_removal (α : Type u) 
  (t : Triangulation α) : 
  (BeautifulTriangulation α).to_Triangulation t ↔ 
  remove_diagonals_to_quadrilaterals t := 
sorry

end beautiful_iff_quad_removal_l480_480283


namespace range_of_m_l480_480575

-- Definitions for the sets A and B and their properties
def A : set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2m - 1}

-- Statement to prove that B is non-empty and a subset of A implies m is in [2, 3]
theorem range_of_m (m : ℝ) (h : B m ≠ ∅ ∧ B m ⊆ A) : 2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l480_480575


namespace infinite_series_sum_l480_480155

theorem infinite_series_sum :
  let S := ∑' k : ℕ, (k + 1) * (1 / 2023)^k
  in S = 4095562 / 4094884 :=
by
  sorry

end infinite_series_sum_l480_480155


namespace profit_percent_l480_480647

theorem profit_percent (SP CP : ℝ) (h : CP = 0.90 * SP) :
  ((SP - CP) / CP) * 100 ≈ 11.11 :=
by
  sorry

end profit_percent_l480_480647


namespace theater_revenue_l480_480502

theorem theater_revenue
  (total_seats : ℕ)
  (adult_price : ℕ)
  (child_price : ℕ)
  (child_tickets_sold : ℕ)
  (total_sold_out : total_seats = 80)
  (child_tickets_sold_cond : child_tickets_sold = 63)
  (adult_ticket_price_cond : adult_price = 12)
  (child_ticket_price_cond : child_price = 5)
  : total_seats * adult_price + child_tickets_sold * child_price = 519 :=
by
  -- proof omitted
  sorry

end theater_revenue_l480_480502


namespace union_M_N_eq_l480_480622

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 3x + 2 < 0}
def N : Set ℝ := {x | (1/2)^x ≤ 4}

-- The theorem to prove
theorem union_M_N_eq : (M ∪ N) = {x | x ≥ -2} :=
  sorry

end union_M_N_eq_l480_480622


namespace bela_advantage_l480_480949

noncomputable def game_advantage (N K: ℕ) : ℕ :=
  if K = 0 ∨ N % K = 0 then 1 else 0

theorem bela_advantage :
  ∑ N in finset.range 10000, ∑ K in finset.range 100,
    if K = 0 ∨ N % K = 0 then 1 else 0 > 
  9999 * 0.1 := 
sorry

end bela_advantage_l480_480949


namespace divisible_by_4_l480_480570

theorem divisible_by_4 (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n^3 + (n + 1)^3 + (n + 2)^3 = m^3) : 4 ∣ n + 1 :=
sorry

end divisible_by_4_l480_480570


namespace brendan_remaining_money_l480_480958

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l480_480958


namespace max_ways_to_ascend_and_descend_l480_480783

theorem max_ways_to_ascend_and_descend :
  let east := 2
  let west := 3
  let south := 4
  let north := 1
  let ascend_descend_ways (ascend: ℕ) (n_1 n_2 n_3: ℕ) := ascend * (n_1 + n_2 + n_3)
  (ascend_descend_ways south east west north > ascend_descend_ways east west south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways west east south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways north east west south) := sorry

end max_ways_to_ascend_and_descend_l480_480783


namespace largest_non_divisible_subsequence_l480_480563

-- Define the concepts of natural numbers and subsequences
def is_subsequence_of_digits (n m : ℕ) : Prop :=
  ∃ (d : list ℕ), (list.nodup d) ∧ (list.all d (λ x, x < 10)) ∧ (list.to_nat d = m) ∧ (d <+ list.of_digits n)

def not_divisible_by_11 (n : ℕ) : Prop :=
  ¬ (n % 11 = 0)

theorem largest_non_divisible_subsequence :
  ∀ (m : ℕ), is_subsequence_of_digits 987654321 m → not_divisible_by_11 m :=
sorry

end largest_non_divisible_subsequence_l480_480563


namespace y_coordinate_eq_l480_480007

theorem y_coordinate_eq (y : ℝ) : 
  (∃ y, (√(9 + y^2) = √(4 + (5 - y)^2))) ↔ (y = 2) :=
by
  sorry

end y_coordinate_eq_l480_480007


namespace find_numbers_sum_eq_S_product_eq_P_l480_480433

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480433


namespace roundness_of_1728000_l480_480522

def roundness (n : ℕ) : ℕ :=
  let factors := n.factorization
  factors.findD 2 0 + factors.findD 3 0 + factors.findD 5 0

theorem roundness_of_1728000 : roundness 1728000 = 15 :=
by
  sorry

end roundness_of_1728000_l480_480522


namespace max_volume_open_top_box_with_square_base_l480_480162

theorem max_volume_open_top_box_with_square_base (w : ℝ) (x : ℝ) :
  w = 60 ∧ 0 < x ∧ x < 30 → (∀ x, 0 < x ∧ x < 30 → volume x = (60 - 2 * x)^2 * x → volume x ≤ 16000) :=
sorry

end max_volume_open_top_box_with_square_base_l480_480162


namespace piecewise_function_continuity_l480_480696

theorem piecewise_function_continuity (a b : ℝ) :
  (∀ x, if x > 3 then continuous (λ x, a * x + 6) else 
        if -3 ≤ x ∧ x ≤ 3 then continuous (λ x, x - 7) else 
        if x < -3 then continuous (λ x, 3 * x - b) else false) →
  (a = -10/3 ∧ b = 1) → 
  a + b = -7/3 :=
by 
  intros h1 h2
  obtain ⟨ha, hb⟩ := h2
  rw [ha, hb]
  norm_num
  sorry

end piecewise_function_continuity_l480_480696


namespace log_a_b_iff_a_minus_1_b_minus_1_l480_480165

theorem log_a_b_iff_a_minus_1_b_minus_1 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ 1) : 
  (log a b > 0) ↔ ((a - 1) * (b - 1) > 0) := 
by
  sorry

end log_a_b_iff_a_minus_1_b_minus_1_l480_480165


namespace disks_on_circle_area_sum_l480_480556

theorem disks_on_circle_area_sum :
  ∃ (a b c : ℕ), c.prime_factors.PairwiseDisjoint ∧
    (c.prime_factors.pairwise (λ p q, p ∣ q → p = q)) ∧
    a + b + c = 168 ∧
    ∃ r : ℝ, r = 2 - Real.sqrt 3 ∧
    15 * (Real.pi * r^2) = Real.pi * (a - b * Real.sqrt c) :=
by sorry

end disks_on_circle_area_sum_l480_480556


namespace find_x_average_equality_l480_480737

theorem find_x_average_equality :
  let avg1 := (10 + 30 + 50) / 3
  let avg2 (x : ℕ) := (20 + x + 6) / 3
  avg1 = avg2 40 + 8 :=
by
  -- avg1 calculation
  have avg1_eq : avg1 = 30 := by
    sorry
  -- avg2 calculation
  have avg2_eq : avg2 40 = 22 := by
    sorry
  -- Equating the condition
  have h: avg1 = avg2 40 + 8 := by
    sorry
  exact h

end find_x_average_equality_l480_480737


namespace tan_identity_l480_480725

theorem tan_identity (α β γ : ℝ) (h : α + β + γ = 45 * π / 180) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 :=
by
  sorry

end tan_identity_l480_480725


namespace one_three_digit_cube_divisible_by_16_l480_480629

theorem one_three_digit_cube_divisible_by_16 :
  ∃! (n : ℕ), (100 ≤ n ∧ n < 1000 ∧ ∃ (k : ℕ), n = k^3 ∧ 16 ∣ n) :=
sorry

end one_three_digit_cube_divisible_by_16_l480_480629


namespace prove_f_values_l480_480183

-- Define the function f and conditions as hypotheses
variables {f : ℝ → ℝ}
hypothesis h1 : ∀ x : ℝ, f (-x) = -f x
hypothesis h2 : ∀ x : ℝ, f (x - 4) = -f x
hypothesis h3 : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 < f x2

-- State the theorem
theorem prove_f_values : f (-25) < f 80 ∧ f 80 < f 11 :=
by
  sorry

end prove_f_values_l480_480183


namespace intersection_M_N_l480_480623

open Set

-- Define the sets M and N as given in the problem conditions
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^(-x) }
def N : Set ℝ := univ

-- Prove that the intersection of M and N is (0, +∞)
theorem intersection_M_N : M ∩ N = Ioi 0 :=
by
  sorry

end intersection_M_N_l480_480623


namespace number_of_periods_l480_480480

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l480_480480


namespace interest_rate_per_annum_l480_480913

theorem interest_rate_per_annum 
  (principal : ℕ := 1000)
  (time : ℕ := 8)
  (interest_less_than_principal : ℕ := 520) : 
  let interest := principal - interest_less_than_principal in
  let rate := (interest * 100) / (principal * time) in
  rate = 6 :=
by
  let interest := principal - interest_less_than_principal
  let rate := (interest * 100) / (principal * time)
  have h : rate = 6 
  sorry

end interest_rate_per_annum_l480_480913


namespace probability_of_exactly_five_correct_letters_is_zero_l480_480786

theorem probability_of_exactly_five_correct_letters_is_zero :
  ∀ (envelopes : Fin 6 → ℕ), (∃! (f : Fin 6 → Fin 6), Bijective f ∧ ∀ i, envelopes (f i) = envelopes i) → 
  (Probability (exactly_five_correct ∈ permutations (Fin 6)) = 0) :=
by
  sorry

noncomputable def exactly_five_correct (σ : Fin 6 → Fin 6) : Prop :=
  (∃ (i : Fin 6), ∀ j, j ≠ i → σ j = j) ∧ ∃ (k : Fin 6), k ≠ i ∧ σ k ≠ k

noncomputable def permutations (α : Type) := List.permutations (List.finRange 6)

end probability_of_exactly_five_correct_letters_is_zero_l480_480786


namespace pumpkin_contest_l480_480258

/-- In a pumpkin contest:
    - Brad's pumpkin weighs 54 pounds.
    - Betty's pumpkin weighs 4 times Jessica's pumpkin.
    - The difference between the heaviest and lightest pumpkin is 81 pounds.
    Prove that the ratio of the weight of Jessica's pumpkin to the weight of Brad's pumpkin is 5:8.
-/
theorem pumpkin_contest (J : ℝ) :
  (4 * J - 54 = 81) → J / 54 = 5 / 8 :=
by
  intro h
  have h2 : 4 * J = 135 := by linarith
  have J_val : J = 135 / 4 := by linarith
  rw [J_val]
  have : 135 / 4 / 54 = 135 / (4 * 54) := by field_simp
  rw [this]
  norm_num
  norm_num
  rfl

end pumpkin_contest_l480_480258


namespace cheaper_rock_cost_per_ton_l480_480881

theorem cheaper_rock_cost_per_ton (x : ℝ) 
    (h1 : 24 * 1 = 24) 
    (h2 : 800 = 16 * x + 8 * 40) : 
    x = 30 :=
sorry

end cheaper_rock_cost_per_ton_l480_480881


namespace science_club_election_l480_480952

theorem science_club_election :
  let total_candidates := 20
  let past_officers := 10
  let non_past_officers := total_candidates - past_officers
  let positions := 6
  let total_ways := Nat.choose total_candidates positions
  let no_past_officer_ways := Nat.choose non_past_officers positions
  let exactly_one_past_officer_ways := past_officers * Nat.choose non_past_officers (positions - 1)
  total_ways - no_past_officer_ways - exactly_one_past_officer_ways = 36030 := by
    sorry

end science_club_election_l480_480952


namespace kyler_games_won_l480_480709

theorem kyler_games_won (peter_wins peter_losses emma_wins emma_losses kyler_losses : ℕ)
  (h_peter : peter_wins = 5)
  (h_peter_losses : peter_losses = 4)
  (h_emma : emma_wins = 2)
  (h_emma_losses : emma_losses = 5)
  (h_kyler_losses : kyler_losses = 4) : ∃ kyler_wins : ℕ, kyler_wins = 2 :=
by {
  sorry
}

end kyler_games_won_l480_480709


namespace mnqp_is_rhombus_l480_480951

-- Definitions according to problem conditions
variables {P O A B M N C E D Q : Point}
variables (PA PB : Line) [Tangent PA] [Tangent PB]
variables [Midpoint M P A] [Midpoint N A B]
variables (MN : Line) [IntersectsAt C E MN Circle] [N_between_M_C : Between N M C]
variables (PC : Line) [IntersectsAt D PC Circle]
variables (ND_ext : Line) [IntersectsAt Q ND_ext PB]

-- The theorem according to the mathematically equivalent proof problem
theorem mnqp_is_rhombus
  (H1 : P ∈ {PA, PB})
  (H2 : PA ⊥ PB)
  (H3 : M = midpoint P A)
  (H4 : N = midpoint A B)
  (H5 : intersects MN C E)
  (H6 : between N M C)
  (H7 : intersects PC D)
  (H8 : intersects ND_ext Q PB) :
  rhombus MNQP := sorry

end mnqp_is_rhombus_l480_480951


namespace remainder_67pow67_add_67_div_68_l480_480053

-- Lean statement starting with the question and conditions translated to Lean

theorem remainder_67pow67_add_67_div_68 : 
  (67 ^ 67 + 67) % 68 = 66 := 
by
  -- Condition: 67 ≡ -1 mod 68
  have h : 67 % 68 = -1 % 68 := by norm_num
  sorry

end remainder_67pow67_add_67_div_68_l480_480053


namespace marble_group_size_l480_480047

-- Define the conditions
def num_marbles : ℕ := 220
def future_people (x : ℕ) : ℕ := x + 2
def marbles_per_person (x : ℕ) : ℕ := num_marbles / x
def marbles_if_2_more (x : ℕ) : ℕ := num_marbles / future_people x

-- Statement of the theorem
theorem marble_group_size (x : ℕ) :
  (marbles_per_person x - 1 = marbles_if_2_more x) ↔ x = 20 :=
sorry

end marble_group_size_l480_480047


namespace find_fx_expression_l480_480168

theorem find_fx_expression {f : ℝ → ℝ}
(h : ∀ x : ℝ, f(x + 1) = x^2 + 2 * x - 1) :
  f(x) = x^2 - 2 :=
sorry

end find_fx_expression_l480_480168


namespace four_digit_number_count_is_18_l480_480400

theorem four_digit_number_count_is_18 :
  (number_of_four_digit_numbers : ℕ),
  (∀ n, digit_of_number n = 1 ∨ digit_of_number n = 2 ∨ digit_of_number n = 3) ∧
  (∀ n, adjacent_digits n ≠ digit_of_number n) →
  (total_four_digit_numbers number_of_four_digit_numbers = 18) :=
sorry

end four_digit_number_count_is_18_l480_480400


namespace original_money_l480_480514

theorem original_money (M : ℕ) (h1 : 3 * M / 8 ≤ M)
  (h2 : 1 * (M - 3 * M / 8) / 5 ≤ M - 3 * M / 8)
  (h3 : M - 3 * M / 8 - (1 * (M - 3 * M / 8) / 5) = 36) : M = 72 :=
sorry

end original_money_l480_480514


namespace sum_of_first_fifteen_multiples_of_17_l480_480844

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480844


namespace bathroom_square_footage_l480_480476

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l480_480476


namespace unique_triplet_exists_l480_480988

theorem unique_triplet_exists (a b p : ℕ) (hp : Nat.Prime p) : 
  (a + b)^p = p^a + p^b → (a = 1 ∧ b = 1 ∧ p = 2) :=
by sorry

end unique_triplet_exists_l480_480988


namespace hilton_final_marbles_l480_480628

def initial_marbles : ℕ := 26
def marbles_found : ℕ := 6
def marbles_lost : ℕ := 10
def marbles_from_lori := 2 * marbles_lost

def final_marbles := initial_marbles + marbles_found - marbles_lost + marbles_from_lori

theorem hilton_final_marbles : final_marbles = 42 := sorry

end hilton_final_marbles_l480_480628


namespace concrete_volume_needed_l480_480484

theorem concrete_volume_needed
  (width_ft : ℝ) (length_ft : ℝ) (thickness_in : ℝ) 
  (conversion_ft_to_yd : ℝ) (conversion_in_to_yd : ℝ) 
  (extra_percent : ℝ) 
  (width : width_ft = 3) 
  (length : length_ft = 120) 
  (thickness : thickness_in = 6)
  (conversion_ft : conversion_ft_to_yd = 3)
  (conversion_in : conversion_in_to_yd = 36)
  (extra : extra_percent = 10)
  :  8 = ⌈ ((width_ft / conversion_ft_to_yd) * (length_ft / conversion_ft_to_yd) * (thickness_in / conversion_in_to_yd) * (1 + extra_percent / 100)) ⌉ :=
begin
  sorry
end

end concrete_volume_needed_l480_480484


namespace prime_factor_inequality_l480_480925

theorem prime_factor_inequality 
  (x : ℕ → ℤ) 
  (h_1 : x 1 = 2) 
  (h_2 : x 2 = 12) 
  (h_rec : ∀ n, x (n + 2) = 6 * x (n + 1) - x n) 
  (p q : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hp_odd : p % 2 = 1) 
  (hq_xp : q ∣ x p) 
  (hq_ne_2 : q ≠ 2) 
: q ≥ 2 * p - 1 := 
sorry

end prime_factor_inequality_l480_480925


namespace rectangle_area_l480_480389

theorem rectangle_area (P Q R A B C D : Point)
  (h_congruent : Circle P = Circle Q ∧ Circle Q = Circle R ∧ Circle P = Circle R)
  (h_tangent_sides : tangent (Circle P) [AB, BC, CD, DA] ∧ tangent (Circle Q) [AB, BC, CD, DA] ∧ tangent (Circle R) [AB, BC, CD, DA])
  (h_diameter_Q : Circle Q.diameter = 6 ∧ contains (Circle Q) P ∧ contains (Circle Q) R) :
  area_rectangle A B C D = 72 :=
sorry

end rectangle_area_l480_480389


namespace hundred_c_plus_d_l480_480682

-- Define the constants c and d
variables (c d : ℝ)

-- Conditions extracted from the problem
def equation1 := ∀ x : ℝ, (x+c) * (x+d) * (x+15) = 0 → x ≠ -4
def equation2 := ∀ x : ℝ, (x+3c) * (x+5) * (x+9) = 0 → x ≠ -d ∧ x ≠ -15

-- The Lean theorem statement proving the final value
theorem hundred_c_plus_d : equation1 c d → equation2 c d → 100 * c + d = 157 :=
by
  sorry

end hundred_c_plus_d_l480_480682


namespace largest_number_of_members_l480_480087

def largestSubset (s : Set ℕ) := s ⊆ {n | 1 ≤ n ∧ n ≤ 50} ∧ ∀ x ∈ s, ∀ y ∈ s, x ≠ 4 * y ∧ y ≠ 4 * x

theorem largest_number_of_members : ∃ s : Set ℕ, largestSubset s ∧ s.card = 48 :=
begin
  sorry
end

end largest_number_of_members_l480_480087


namespace percentage_teachers_worked_4_years_or_less_l480_480774

variable (y : ℕ)

def total_teachers : ℕ := 4 * y + 3 * y + 3 * y + 5 * y + 6 * y + 2 * y + 1 * y + 1 * y + 1 * y
def experienced_teachers : ℕ := 4 * y + 3 * y + 3 * y + 5 * y

theorem percentage_teachers_worked_4_years_or_less :
  ((experienced_teachers y).toFloat / (total_teachers y).toFloat) * 100 ≈ 57.69 :=
by
  sorry

end percentage_teachers_worked_4_years_or_less_l480_480774


namespace sum_of_first_fifteen_multiples_of_17_l480_480852

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480852


namespace min_mn_sum_l480_480753

theorem min_mn_sum :
  ∃ (m n : ℕ), n > m ∧ m ≥ 1 ∧ 
  (1978^n % 1000 = 1978^m % 1000) ∧ (m + n = 106) :=
sorry

end min_mn_sum_l480_480753


namespace completing_the_square_result_l480_480031

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l480_480031


namespace angle_between_vectors_l480_480589

open Real EuclideanSpace

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

theorem angle_between_vectors
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h : ∥a + 3 • b∥ = ∥a - 3 • b∥) :
  ⟪a, b⟫ = 0 :=
sorry

end angle_between_vectors_l480_480589


namespace completing_the_square_equation_l480_480036

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l480_480036


namespace product_positive_probability_l480_480397

-- Define the set of numbers
def number_set : Set ℤ := {-3, -2, -1, 1, 2, 6}
-- Define the total number of ways to choose 2 elements from a given set of size n
noncomputable def choose_two (n : ℕ) : ℕ := Nat.choose n 2

-- Define the favorable combinations for a product to be positive
noncomputable def favorable_combinations (s : Set ℤ) : ℕ :=
  choose_two 3 + choose_two 3

-- Define the total number of ways to choose 2 elements from the number_set
noncomputable def total_combinations : ℕ :=
  choose_two (number_set.toFinset.card)

-- The probability that the product is positive
noncomputable def positive_product_probability : ℚ :=
  favorable_combinations number_set / total_combinations

theorem product_positive_probability : positive_product_probability = (2 / 5) := by
  sorry

end product_positive_probability_l480_480397


namespace power_sum_l480_480960

theorem power_sum (h : (9 : ℕ) = 3^2) : (2^567 + (9^5 / 3^2) : ℕ) = 2^567 + 6561 := by
  sorry

end power_sum_l480_480960


namespace find_a_2011_l480_480515

noncomputable def is_odd_part (n q : ℕ) : Prop :=
  ∃ a : ℕ, q = 2^a * q ∧ q % 2 = 1

noncomputable def sequence (a₀ : ℕ) : ℕ → ℕ
| 0     := a₀
| (m+1) := let n := 3 * (sequence m) + 1 in
           classical.some (classical.indefinite_description _ (nat.exists_odd_part n))

theorem find_a_2011 :
  sequence (2^2011 - 1) 2011 = (3^2011 - 1) / 2 :=
sorry

end find_a_2011_l480_480515


namespace sin_angle_BAC_l480_480251

theorem sin_angle_BAC (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (angle_ABC : real.angle) (sqrt_2 : ℝ := real.sqrt 2) (abc_cond : triangle_condition := {
    angle_ABC := π / 4,
    length_AB := sqrt_2,
    length_BC := 3
  }) :
  sin (real.angle_of (angle_ABC)) = 3 * real.sqrt 10 / 10 :=
sorry

end sin_angle_BAC_l480_480251


namespace exists_n_for_conditions_l480_480546

theorem exists_n_for_conditions : ∃ (n : ℕ), ∀ (x y : ℝ), ∃ (a : fin n → ℝ), 
  (x = (finset.univ.sum (λ i, a i))) ∧ (y = (finset.univ.sum (λ i, (1 / (a i))))) :=
begin
  use 6,
  intros x y,
  sorry
end

end exists_n_for_conditions_l480_480546


namespace greatest_whole_number_l480_480992

theorem greatest_whole_number (x : ℤ) (h : 5 * x - 4 < 3 - 2 * x) : x ≤ 0 :=
by {
  have h1 : 7 * x - 4 < 3 := by linarith [h],
  have h2 : 7 * x < 7 := by linarith [h1],
  have h3 : x < 1 := by linarith [h2],
  linarith,
}

noncomputable def greatest_whole_number_is (x : ℤ) : x = 0 :=
by {
  apply greatest_whole_number,
  linarith,
}


end greatest_whole_number_l480_480992


namespace find_two_numbers_l480_480458

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480458


namespace ratio_of_standing_men_is_one_eighth_l480_480780

-- Given conditions
def total_passengers : ℕ := 48
def women_fraction : ℚ := 2 / 3
def seated_men : ℕ := 14

-- Derived definitions from given conditions
def total_men : ℕ := (1 - women_fraction) * total_passengers
def standing_men : ℕ := total_men - seated_men
def ratio_standing_to_total_men : ℚ := standing_men / total_men

-- Theorem to prove
theorem ratio_of_standing_men_is_one_eighth :
  ratio_standing_to_total_men = 1 / 8 := by
  sorry

end ratio_of_standing_men_is_one_eighth_l480_480780


namespace exponential_monotonicity_example_l480_480166

theorem exponential_monotonicity_example (m n : ℕ) (a b : ℝ) (h1 : a = 0.2 ^ m) (h2 : b = 0.2 ^ n) (h3 : m > n) : a < b :=
by
  sorry

end exponential_monotonicity_example_l480_480166


namespace complement_in_U_l480_480624

open Set

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def U : Set ℝ := { x : ℝ | -6 < x ∧ x < 2 }
def A : Set ℝ := { x : ℝ | x^2 + 2 * x - 3 < 0 }

theorem complement_in_U :
  compl A ∩ U = (Ioo (-6 : ℝ) (-3) ∪ Ico (1 : ℝ) 2) :=
by
  sorry

end complement_in_U_l480_480624


namespace distance_to_office_is_18_l480_480491

-- Definitions given in the problem conditions
variables (x t d : ℝ)
-- Conditions based on the problem statements
axiom speed_condition1 : d = x * t
axiom speed_condition2 : d = (x + 1) * (3 / 4 * t)
axiom speed_condition3 : d = (x - 1) * (t + 3)

-- The mathematical proof statement that needs to be shown
theorem distance_to_office_is_18 :
  d = 18 :=
by
  sorry

end distance_to_office_is_18_l480_480491


namespace div_fractions_eq_l480_480807

theorem div_fractions_eq : (3/7) / (5/2) = 6/35 := 
by sorry

end div_fractions_eq_l480_480807


namespace prod_fraction_eq_1771_l480_480966

theorem prod_fraction_eq_1771 : (∏ n in Finset.range 20, ((n + 4) / (n + 1): ℚ)) = 1771 := 
by
  sorry

end prod_fraction_eq_1771_l480_480966


namespace gcd_mod_remainder_l480_480286

theorem gcd_mod_remainder (a b : ℕ) : 
  let d := Nat.gcd (2^30^10 - 2) (2^30^45 - 2)
  in d % 2013 = 2012 :=
by
  let d := Nat.gcd (2^30^10 - 2) (2^30^45 - 2)
  have hd : d = 2^(30^5) - 2 := by sorry
  have mod_2013: 2^(30^5) % 2013 = 1 := by sorry
  calc
    d % 2013
    = (2^(30^5) - 2) % 2013 : by rw [hd]
    ... = (1 - 2) % 2013 : by rw [mod_2013]
    ... = 2012 : by norm_num

end gcd_mod_remainder_l480_480286


namespace two_digit_square_difference_l480_480127

-- Define the problem in Lean
theorem two_digit_square_difference :
  ∃ (X Y : ℕ), (10 ≤ X ∧ X ≤ 99) ∧ (10 ≤ Y ∧ Y ≤ 99) ∧ (X > Y) ∧
  (∃ (t : ℕ), (1 ≤ t ∧ t ≤ 9) ∧ (X^2 - Y^2 = 100 * t)) :=
sorry

end two_digit_square_difference_l480_480127


namespace sum_of_first_fifteen_multiples_of_17_l480_480842

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480842


namespace perp_tangent_line_equation_l480_480143

theorem perp_tangent_line_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, 3 * x + y + 2 = 0 ↔ y = -3 * x - 2) ∧
               (∀ x y : ℝ, (2 * x - 6 * y + 1 = 0) → (y = -(1/3) * x + 1/6)) ∧
               (∀ x : ℝ, y = x^3 + 3 * x^2 - 1 → derivative y at x = -3) ∧
               (∃ x : ℝ, 3 * x^2 + 6 * x = -3) :=
sorry

end perp_tangent_line_equation_l480_480143


namespace possible_values_of_m_l480_480902

theorem possible_values_of_m (m : ℝ) : 
  (∃ m ∈ { m : ℝ | let s := (26 + m) / 5 in s = 5 ∨ (5 < m ∧ m < 8) ∧ s = m ∨ 8 < m ∧ s = 8 }) →
  set.card { m | let s := (26 + m) / 5 in s = 5 ∨ (5 < m ∧ m < 8) ∧ s = m ∨ 8 < m ∧ s = 8 } = 3 :=
sorry

end possible_values_of_m_l480_480902


namespace find_x_value_l480_480309

variable {x : ℝ}

def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (k * b.1, k * b.2)

theorem find_x_value (h : opposite_directions (x, 1) (4, x)) : x = -2 :=
sorry

end find_x_value_l480_480309


namespace sum_first_fifteen_multiples_seventeen_l480_480821

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480821


namespace compound_oxygen_atoms_l480_480892

theorem compound_oxygen_atoms 
  (C_atoms : ℕ)
  (H_atoms : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_O : ℝ) :
  C_atoms = 4 →
  H_atoms = 8 →
  total_molecular_weight = 88 →
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  atomic_weight_O = 16.00 →
  (total_molecular_weight - (C_atoms * atomic_weight_C + H_atoms * atomic_weight_H)) / atomic_weight_O = 2 := 
by 
  intros;
  sorry

end compound_oxygen_atoms_l480_480892


namespace limit_example_l480_480875

def f (x : ℝ) : ℝ := (1 / (x - 2)) - (4 / (x^2 - 4))

theorem limit_example : filter.tendsto f (nhds 2) (nhds (1 / 4)) :=
by
  sorry

end limit_example_l480_480875


namespace quadratic_solution_1_l480_480340

theorem quadratic_solution_1 :
  (∃ x, x^2 - 4 * x + 3 = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

end quadratic_solution_1_l480_480340


namespace selling_price_is_correct_l480_480495

def gain : ℝ := 10
def gain_percent : ℝ := 10

def cost_price (gain : ℝ) (gain_percent : ℝ) : ℝ :=
  gain / (gain_percent / 100)

def selling_price (CP : ℝ) (gain : ℝ) : ℝ :=
  CP + gain

theorem selling_price_is_correct : selling_price (cost_price gain gain_percent) gain = 110 :=
by
  -- Lean 4 proof steps would go here
  sorry

end selling_price_is_correct_l480_480495


namespace compute_g3_squared_l480_480733

theorem compute_g3_squared {f g : ℝ → ℝ} (h₁ : ∀ x, x ≥ 1 → f(g(x)) = x^3)
                           (h₂ : ∀ x, x ≥ 1 → g(f(x)) = x^2)
                           (h₃ : g(9) = 9) :
  (g 3)^2 = 729 :=
sorry

end compute_g3_squared_l480_480733


namespace license_plate_combinations_l480_480101

theorem license_plate_combinations :
  let letters := choose 26 2 * nat.factorial 3 / nat.factorial 2,
      digits := choose 10 1 * nat.sub 9 1 * nat.factorial 3 / nat.factorial 2,
      answer := letters * digits
  letters * digits = 877500 :=
by
  sorry

end license_plate_combinations_l480_480101


namespace hair_length_correct_l480_480334

-- Define the initial hair length, the cut length, and the growth length as constants
def l_initial : ℕ := 16
def l_cut : ℕ := 11
def l_growth : ℕ := 12

-- Define the final hair length as the result of the operations described
def l_final : ℕ := l_initial - l_cut + l_growth

-- State the theorem we want to prove
theorem hair_length_correct : l_final = 17 :=
by
  sorry

end hair_length_correct_l480_480334


namespace expected_area_rectangle_stddev_area_rectangle_cm_l480_480099

namespace Rectangle

open ProbabilityTheory -- Import probability theory framework

def X : MeasureTheory.Measure ℝ := sorry -- Define random variable X
def Y : MeasureTheory.Measure ℝ := sorry -- Define random variable Y

axiom independent_XY : MeasureTheory.Independent X Y
axiom E_X : MeasureTheory.Expectation X = 2
axiom E_Y : MeasureTheory.Expectation Y = 1
axiom stddev_X : MeasureTheory.stddev X = 0.003
axiom stddev_Y : MeasureTheory.stddev Y = 0.002

-- Part (a)
theorem expected_area_rectangle : MeasureTheory.Expectation (X * Y) = 2 := 
by {
  -- Proof omitted
  sorry
}

-- Part (b)
theorem stddev_area_rectangle_cm : MeasureTheory.stddev (X * Y) = 50 :=
by {
  -- Proof omitted
  sorry
}

end Rectangle

end expected_area_rectangle_stddev_area_rectangle_cm_l480_480099


namespace months_elapsed_between_dates_l480_480673

theorem months_elapsed_between_dates :
  let today := (3, 3, 2001)
  let most_recent_odd_date := (11, 19, 1999)
  let next_odd_date := (1, 1, 3111) in
  elapsed_months most_recent_odd_date next_odd_date = 13333 :=
by
  sorry

def elapsed_months (start_date end_date : (ℕ, ℕ, ℕ)) : ℕ :=
  let (start_day, start_month, start_year) := start_date
  let (end_day, end_month, end_year) := end_date
  (end_year - start_year) * 12 + (end_month - start_month) + 
  if end_day >= start_day then 0 else -1

end months_elapsed_between_dates_l480_480673


namespace probability_both_counterfeit_given_one_counterfeit_l480_480573

-- Conditions
def total_bills := 20
def counterfeit_bills := 5
def selected_bills := 2
def at_least_one_counterfeit := true

-- Definition of events
def eventA := "both selected bills are counterfeit"
def eventB := "at least one of the selected bills is counterfeit"

-- The theorem to prove
theorem probability_both_counterfeit_given_one_counterfeit : 
  at_least_one_counterfeit →
  ( (counterfeit_bills * (counterfeit_bills - 1)) / (total_bills * (total_bills - 1)) ) / 
    ( (counterfeit_bills * (counterfeit_bills - 1) + counterfeit_bills * (total_bills - counterfeit_bills)) / (total_bills * (total_bills - 1)) ) = 2/17 :=
by
  sorry

end probability_both_counterfeit_given_one_counterfeit_l480_480573


namespace infinite_primes_not_dividing_diff_of_polynomials_l480_480330

theorem infinite_primes_not_dividing_diff_of_polynomials :
  ∃ (f g : ℤ[X]),  
    ¬ (IsConst f ∧ IsConst g) ∧
    (∀ p : ℕ, prime p → ∃^∞ p, ∀ (x y : ℤ), ¬ (p ∣ (f.eval x - g.eval y))) :=
by
  let f : ℤ[X] := polynomial.C 1 * polynomial.X ^ 2
  let g : ℤ[X] := -(polynomial.C 1 * (polynomial.X ^ 2 + polynomial.C 1) ^ 2)
  use [f, g]
  split
  · exact fun h => by simp_rw [polynomial.is_const_iff_degree_eq_zero, polynomial.degree_X_pow] at h; contradiction
  sorry

end infinite_primes_not_dividing_diff_of_polynomials_l480_480330


namespace equivalent_proof_problem_l480_480418

variables {a b c d e : ℚ}

theorem equivalent_proof_problem
  (h1 : 3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55)
  (h2 : 4 * (d + c + e) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d)
  (h5 : d + 1 = e) : 
  a * b * c * d * e = -1912397372 / 78364164096 := 
sorry

end equivalent_proof_problem_l480_480418


namespace distance_from_center_to_tangent_chord_l480_480217

theorem distance_from_center_to_tangent_chord
  (R a m x : ℝ)
  (h1 : m^2 = 4 * R^2)
  (h2 : 16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0) :
  x = R :=
sorry

end distance_from_center_to_tangent_chord_l480_480217


namespace min_colors_needed_to_distinguish_keys_l480_480941

theorem min_colors_needed_to_distinguish_keys :
  ∀ (n : ℕ), n = 8 → (∀ (caps : Finset (Fin 8 → Fin 2)), 
    (∀ i j : Fin 8, i ≠ j → caps i ≠ caps j) → 2 ≤ caps.card) :=
by sorry

end min_colors_needed_to_distinguish_keys_l480_480941


namespace total_ticket_cost_is_14_l480_480098

-- Definitions of the ticket costs
def ticket_cost_hat : ℕ := 2
def ticket_cost_stuffed_animal : ℕ := 10
def ticket_cost_yoyo : ℕ := 2

-- Definition of the total ticket cost
def total_ticket_cost : ℕ := ticket_cost_hat + ticket_cost_stuffed_animal + ticket_cost_yoyo

-- Theorem stating the total ticket cost is 14
theorem total_ticket_cost_is_14 : total_ticket_cost = 14 := by
  -- Proof would go here, but sorry is used to skip it
  sorry

end total_ticket_cost_is_14_l480_480098


namespace min_workers_for_profit_l480_480891

theorem min_workers_for_profit
    (maintenance_fees : ℝ)
    (worker_hourly_wage : ℝ)
    (widgets_per_hour : ℝ)
    (widget_price : ℝ)
    (work_hours : ℝ)
    (n : ℕ)
    (h_maintenance : maintenance_fees = 470)
    (h_wage : worker_hourly_wage = 10)
    (h_production : widgets_per_hour = 6)
    (h_price : widget_price = 3.5)
    (h_hours : work_hours = 8) :
  470 + 80 * n < 168 * n → n ≥ 6 := 
by
  sorry

end min_workers_for_profit_l480_480891


namespace true_statements_count_l480_480218

theorem true_statements_count {a b c : ℝ} :
  let P := (a > b) → (a * c^2 > b * c^2),
      Q := (a * c^2 > b * c^2) → (a > b),
      R := ¬ ((a > b) → (a * c^2 > b * c^2)),
      S := (a ≤ b) → (a * c^2 ≤ b * c^2)
  in (if Q then 1 else 0) + (if R then 1 else 0) + (if S then 1 else 0) = 2 :=
by
  sorry

end true_statements_count_l480_480218


namespace numbers_pairs_sum_prod_l480_480428

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480428


namespace hexagon_interior_angle_sum_l480_480374

theorem hexagon_interior_angle_sum : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  intros n hn
  rw hn
  -- The steps of the proof would continue here, but are not required as per instructions.
  sorry

end hexagon_interior_angle_sum_l480_480374


namespace largest_number_of_square_plots_l480_480898

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l480_480898


namespace probability_all_same_color_l480_480885

open scoped Classical

noncomputable def num_black : ℕ := 5
noncomputable def num_red : ℕ := 4
noncomputable def num_green : ℕ := 6
noncomputable def num_blue : ℕ := 3
noncomputable def num_yellow : ℕ := 2

noncomputable def total_marbles : ℕ :=
  num_black + num_red + num_green + num_blue + num_yellow

noncomputable def prob_all_same_color : ℚ :=
  let p_black := if num_black >= 4 then 
      (num_black / total_marbles) * ((num_black - 1) / (total_marbles - 1)) *
      ((num_black - 2) / (total_marbles - 2)) * ((num_black - 3) / (total_marbles - 3)) else 0
  let p_green := if num_green >= 4 then 
      (num_green / total_marbles) * ((num_green - 1) / (total_marbles - 1)) *
      ((num_green - 2) / (total_marbles - 2)) * ((num_green - 3) / (total_marbles - 3)) else 0
  p_black + p_green

theorem probability_all_same_color :
  prob_all_same_color = 0.004128 :=
sorry

end probability_all_same_color_l480_480885


namespace count_not_divisible_by_4_l480_480125

def floor_sum (a b c n : ℕ) : ℕ :=
  (Nat.floor (a / n) + Nat.floor (b / n) + Nat.floor (c / n))

theorem count_not_divisible_by_4 : 
  let count := (Finset.range 1001).filter (λ n, floor_sum 1950 1951 1952 n % 4 ≠ 0) in
  count.card = 20 := 
by
  sorry

end count_not_divisible_by_4_l480_480125


namespace find_triplets_l480_480142

theorem find_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧ (2 * y^3 + 1 = 3 * x * y) ∧ (2 * z^3 + 1 = 3 * y * z) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 / 2 ∧ y = -1 / 2 ∧ z = -1 / 2) :=
by 
  intro h
  sorry

end find_triplets_l480_480142


namespace sum_of_coordinates_of_A_l480_480292

noncomputable def point := (ℝ × ℝ)
def B : point := (2, 6)
def C : point := (4, 12)
def AC (A C : point) : ℝ := (A.1 - C.1)^2 + (A.2 - C.2)^2
def AB (A B : point) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2
def BC (B C : point) : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem sum_of_coordinates_of_A :
  ∃ A : point, AC A C / AB A B = (1/3) ∧ BC B C / AB A B = (1/3) ∧ A.1 + A.2 = 24 :=
by
  sorry

end sum_of_coordinates_of_A_l480_480292


namespace equality_of_expressions_l480_480859

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l480_480859


namespace expand_expression_l480_480554

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end expand_expression_l480_480554


namespace largest_possible_s_l480_480293

-- Definitions based on the problem conditions
def polygon_interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

-- Main theorem statement
theorem largest_possible_s (r s : ℕ) (h₁ : r ≥ s) (h₂ : s ≥ 3) 
  (h₃ : polygon_interior_angle r = (5/4) * polygon_interior_angle s) : s ≤ 102 :=
by
  sorry

#eval largest_possible_s -- This line is to test build without actual function call, should be removed in real usage.

end largest_possible_s_l480_480293


namespace minimum_m_l480_480625

/-- Definition of set B -/
def B (m : ℝ) : Set (ℝ × ℝ) := { p | 3 * p.1 + 2 * p.2 = m }

/-- The condition that A covers all points on the plane -/
def A : Set (ℝ × ℝ) := { p | True }

theorem minimum_m (m : ℝ) : (A ∩ B m).Nonempty ↔ m = 0 := by
  split
  · intro h
    cases h with p hp
    simp only [Set.mem_inter_iff, A, B, Set.mem_setOf_eq, and_true] at hp
    cases hp with hx hy
    rw [hx, hy] at hx
    ring at hx
  · intro hm
    use (0, 0)
    simp [A, B, hm]

end minimum_m_l480_480625


namespace distance_between_adjacent_parallel_lines_l480_480797

theorem distance_between_adjacent_parallel_lines 
    (chord1_len chord2_len chord3_len : ℝ)
    (h_chord1 : chord1_len = 42)
    (h_chord2 : chord2_len = 36)
    (h_chord3 : chord3_len = 36) : 
    distance_between_adjacent_parallel_lines = 2 * Real.sqrt 2006 :=
by 
  sorry

end distance_between_adjacent_parallel_lines_l480_480797


namespace area_of_given_triangle_l480_480521

noncomputable def area_of_triangle (a b c : ℝ × ℝ × ℝ) : ℝ := 
  let v1 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let v2 := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
  let cross_product := (
    v1.2 * v2.3 - v1.3 * v2.2,
    v1.3 * v2.1 - v1.1 * v2.3,
    v1.1 * v2.2 - v1.2 * v2.1
  )
  0.5 * real.sqrt (
    cross_product.1 ^ 2 +
    cross_product.2 ^ 2 +
    cross_product.3 ^ 2
  )

theorem area_of_given_triangle : area_of_triangle (2, 2, 0) (5, 6, 1) (1, 0, 3) = 5 * real.sqrt 3 := by
  sorry

end area_of_given_triangle_l480_480521


namespace no_five_correct_letters_l480_480792

theorem no_five_correct_letters (n : ℕ) (hn : n = 6) :
  ∀ (σ : Fin n → Fin n), (∑ i, if σ i = i then 1 else 0) ≠ 5 :=
by
  simp only
  sorry

end no_five_correct_letters_l480_480792


namespace completing_square_correct_l480_480025

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l480_480025


namespace find_S_poly_l480_480688

open Polynomial

theorem find_S_poly (P S : R[X])
  (h : X ^ 2023 + 2 = (X^2 - X + 1) * P + S)
  (degree_S : S.degree < 2) :
  S = X + 2 :=
sorry

end find_S_poly_l480_480688


namespace sum_of_first_fifteen_multiples_of_17_l480_480830

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480830


namespace average_price_per_book_l480_480336

theorem average_price_per_book
  (spent1 spent2 spent3 spent4 : ℝ) (books1 books2 books3 books4 : ℕ)
  (h1 : spent1 = 1080) (h2 : spent2 = 840) (h3 : spent3 = 765) (h4 : spent4 = 630)
  (hb1 : books1 = 65) (hb2 : books2 = 55) (hb3 : books3 = 45) (hb4 : books4 = 35) :
  (spent1 + spent2 + spent3 + spent4) / (books1 + books2 + books3 + books4) = 16.575 :=
by {
  sorry
}

end average_price_per_book_l480_480336


namespace complement_union_l480_480222

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_union : U \ (A ∪ B) = {3, 5} :=
by
  sorry

end complement_union_l480_480222


namespace k_intersection_distance_l480_480906

noncomputable def a_b_sum : ℝ → ℝ → ℝ := λ a b, a + b

theorem k_intersection_distance (k a b : ℝ) (hk : k = a + real.sqrt b)
    (h_intersection : ∀ (x : ℝ), y = real.log x / real.log 3 ∨ y = real.log (x + 5) / real.log 3)
    (h_distance : real.abs (real.log k / real.log 3 - real.log (k + 5) / real.log 3) = 0.6) :
    a + b = 8 :=
by sorry

end k_intersection_distance_l480_480906


namespace modulus_sum_l480_480552

noncomputable def modulus (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem modulus_sum : modulus 3 (-3) + modulus 3 3 = 6 * real.sqrt 2 := by
  -- The proof will be filled here
  sorry

end modulus_sum_l480_480552


namespace find_numbers_l480_480450

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480450


namespace four_digit_numbers_count_l480_480150

theorem four_digit_numbers_count : 
  let num_four_digit := (A B C D : ℕ) 
    (hA : 1 ≤ A ∧ A ≤ 9)
    (hB : 0 ≤ B ∧ B ≤ 9)
    (hC : 0 ≤ C ∧ C ≤ 7)
    (hD : 0 ≤ D ∧ D ≤ 9)
    (hCondition : D = C + 2)
  in (Σ A B C D, hA ∧ hB ∧ hC ∧ hD ∧ hCondition)
= 720 := sorry

end four_digit_numbers_count_l480_480150


namespace Polly_tweets_l480_480631

theorem Polly_tweets :
  let HappyTweets := 18 * 50
  let HungryTweets := 4 * 35
  let WatchingReflectionTweets := 45 * 30
  let SadTweets := 6 * 20
  let PlayingWithToysTweets := 25 * 75
  HappyTweets + HungryTweets + WatchingReflectionTweets + SadTweets + PlayingWithToysTweets = 4385 :=
by
  sorry

end Polly_tweets_l480_480631


namespace factorize_expr_l480_480985

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l480_480985


namespace sum_of_first_fifteen_multiples_of_17_l480_480847

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480847


namespace number_of_rowers_l480_480872

theorem number_of_rowers (total_coaches : ℕ) (votes_per_coach : ℕ) (votes_per_rower : ℕ) 
  (htotal_coaches : total_coaches = 36) (hvotes_per_coach : votes_per_coach = 5) 
  (hvotes_per_rower : votes_per_rower = 3) : 
  (total_coaches * votes_per_coach) / votes_per_rower = 60 :=
by 
  sorry

end number_of_rowers_l480_480872


namespace units_digit_base6_product_l480_480523

theorem units_digit_base6_product (a b : ℕ) (h1 : a = 168) (h2 : b = 59) : ((a * b) % 6) = 0 := by
  sorry

end units_digit_base6_product_l480_480523


namespace quadratic_roots_n_value_l480_480367

theorem quadratic_roots_n_value :
  ∃ m p : ℕ, ∃ (n : ℕ) (h : Nat.gcd m p = 1),
  (∃ x1 x2 : ℝ, (3 * x1^2 - 6 * x1 - 9 = 0 ∧ 3 * x2^2 - 6 * x2 - 9 = 0) ∧
   ∀ x, 3 * x^2 - 6 * x - 9 = 0 → x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :=
by
  use [1, 1, 144, Nat.gcd_one_right 1]
  sorry

end quadratic_roots_n_value_l480_480367


namespace probability_of_two_approvals_l480_480917

-- Define the base probabilities
def P_A : ℝ := 0.6
def P_D : ℝ := 1 - P_A

-- Define the binomial coefficient function
def binom (n k : ℕ) := nat.choose n k

-- Define the probability of exactly k successes in n trials
noncomputable def P_exactly_two_approvals :=
  (binom 4 2) * (P_A ^ 2) * (P_D ^ 2)

theorem probability_of_two_approvals : P_exactly_two_approvals = 0.3456 := by
  sorry

end probability_of_two_approvals_l480_480917


namespace completing_square_correct_l480_480023

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l480_480023


namespace roger_trips_l480_480335

variable (trays_per_trip : ℕ) (trays_table_one : ℕ) (trays_table_two : ℕ)

theorem roger_trips
  (h1 : trays_per_trip = 3)
  (h2 : trays_table_one = 15)
  (h3 : trays_table_two = 5) :
  Nat.ceil (trays_table_one / trays_per_trip).toNat + Nat.ceil (trays_table_two / trays_per_trip).toNat = 7 :=
by
  sorry

end roger_trips_l480_480335


namespace Alfred_gain_percentage_l480_480506

-- Defining the conditions
def purchase_price_A := 4700
def repair_cost_A := 600
def selling_price_A := 5800

def purchase_price_B := 3500
def repair_cost_B := 800
def selling_price_B := 4800

def purchase_price_C := 5400
def repair_cost_C := 1000
def selling_price_C := 7000

-- Defining the overall gain percentage calculation
def gain_percentage (total_gain total_cost : ℝ) : ℝ :=
  (total_gain / total_cost) * 100

-- Theorem stating that Alfred's overall gain percentage is 10%
theorem Alfred_gain_percentage : 
  gain_percentage 
    ((selling_price_A - (purchase_price_A + repair_cost_A)) +
     (selling_price_B - (purchase_price_B + repair_cost_B)) +
     (selling_price_C - (purchase_price_C + repair_cost_C))) 
    ((purchase_price_A + repair_cost_A) +
     (purchase_price_B + repair_cost_B) +
     (purchase_price_C + repair_cost_C)) = 10 :=
by
  -- skipping the proof
  sorry

end Alfred_gain_percentage_l480_480506


namespace solve_expr_l480_480632

-- Hypotheses
variables {a b : ℝ}
axiom h1 : 30^a = 2
axiom h2 : 30^b = 7

-- The theorem statement
theorem solve_expr : 10^((2 - a - 2 * b) / (3 * (1 - b))) = 450 / 49 :=
by
  sorry

end solve_expr_l480_480632


namespace dice_probability_event_l480_480896

/-- A fair 8-sided die is rolled until a number 5 or greater appears.
    Calculate the probability that both numbers 2 and 4 appear at least once
    before any number from 5 to 8 is rolled.--/
theorem dice_probability_event (fair_die : Type) [fintype fair_die] [decidable_eq fair_die] 
  (P_number_2_4 : fair_die → Prop) (P_number_5_8 : fair_die → Prop) 
  [decidable_pred P_number_2_4] [decidable_pred P_number_5_8]
  (H_die : (∀ x : fair_die, P_number_2_4 x ∨ P_number_5_8 x) ∧
            (∀ x : fair_die, ¬P_number_2_4 x ∨ ¬P_number_5_8 x)) :
  (∑' (n : ℕ) (h : n ≥ 3), (1 / 2 ^ n) * ((2 ^ (n - 1)) / 2 ^ (n - 1))) = 1 / 4 :=
by
  sorry

end dice_probability_event_l480_480896


namespace dessert_distribution_l480_480768

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end dessert_distribution_l480_480768


namespace sum_of_first_fifteen_multiples_of_17_l480_480850

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480850


namespace largest_n_for_positive_sum_l480_480235

variable {a : ℕ → ℝ}

-- Conditions
axiom arithmetic_sequence (an : ℕ → ℝ) (d : ℝ) : ∀ n, an n = an 0 + n * d
axiom a1_gt_0 : a 0 > 0
axiom a2012_plus_a2013_gt_0 : a 2011 + a 2012 > 0
axiom a2012_times_a2013_lt_0 : a 2011 * a 2012 < 0

-- Define the sum of first n terms S_n
def S (n : ℕ) := ∑ i in Finset.range n, a i

-- Statement of the proof problem
theorem largest_n_for_positive_sum : ∃ n : ℕ, n = 2012 ∧ ∀ (m : ℕ), m > 2012 → S m ≤ 0 := 
sorry

end largest_n_for_positive_sum_l480_480235


namespace ratio_amy_jeremy_l480_480945

variable (Amy Chris Jeremy : ℕ)

theorem ratio_amy_jeremy (h1 : Amy + Jeremy + Chris = 132) (h2 : Jeremy = 66) (h3 : Chris = 2 * Amy) : 
  Amy / Jeremy = 1 / 3 :=
by
  sorry

end ratio_amy_jeremy_l480_480945


namespace sum_of_first_fifteen_multiples_of_17_l480_480832

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480832


namespace angle_between_planes_l480_480088

-- Lean 4 statement of the problem
theorem angle_between_planes 
  (S A B C : Type) 
  (cone : Type)
  (α β γ : ℝ)
  (inscribed_in_cone : S × A × B × C → cone)
  (S_vertex : ∀ (s : S × A × B × C), s.1 = cone.vertex)
  (A_on_base : ∀ (a : A), a ∈ cone.base)
  (B_on_base : ∀ (b : B), b ∈ cone.base)
  (C_on_base : ∀ (c : C), c ∈ cone.base)
  (dih_edge_sa : ∀ (sa : S × A), dihedral_angle sa = α)
  (dih_edge_sb : ∀ (sb : S × B), dihedral_angle sb = β)
  (dih_edge_sc : ∀ (sc : S × C), dihedral_angle sc = γ) :
  angle_between_planes (plane SBC) (tangent_plane SC) = (π - α + β - γ) / 2 := 
sorry

end angle_between_planes_l480_480088


namespace AC_div_AP_l480_480263

variables {F : Type*} [NormedField F] [NormedSpace ℝ F]

-- Definitions based on conditions
variables (A B C D M N P : F)
variables (AB AD AC : ℝ) 
hypothesis (h_rect : ∃ k : ℝ, AB = 8 * k ∧ AD = 4 * k ∧ AC = 4 * (real.sqrt 5) * k)
hypothesis (h_frac1 : AM = (1 / 8) * AB)
hypothesis (h_frac2 : AN = (1 / 4) * AD)
hypothesis (h_intersect : on_line M A B ∧ on_line N A D ∧ intersect AC MN P)

-- Theorem stating the desired result
theorem AC_div_AP : AC / AP = 2 :=
by {
    sorry
}

end AC_div_AP_l480_480263


namespace tim_initial_speed_l480_480800

theorem tim_initial_speed (T : ℕ) (d_initial : ℕ) (d_tim : ℕ) (s_élan : ℕ) (H1 : d_initial = 150) (H2 : d_tim = 100) (H3 : s_élan = 5) (H4 : ∀ t, T + s_élan + 2 * T + 2 * s_élan = d_initial):
  T = 45 :=
by
  sorry

end tim_initial_speed_l480_480800


namespace triangle_construction_conditions_l480_480537

open Classical

noncomputable def construct_triangle (m_a m_b s_c : ℝ) : Prop :=
  m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c

theorem triangle_construction_conditions (m_a m_b s_c : ℝ) :
  construct_triangle m_a m_b s_c ↔ (m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c) :=
by
  sorry

end triangle_construction_conditions_l480_480537


namespace problem1_problem2_l480_480724

-- Sub-problem 1
theorem problem1 (x y : ℝ) (h1 : 9 * x + 10 * y = 1810) (h2 : 11 * x + 8 * y = 1790) : 
  x - y = -10 := 
sorry

-- Sub-problem 2
theorem problem2 (x y : ℝ) (h1 : 2 * x + 2.5 * y = 1200) (h2 : 1000 * x + 900 * y = 530000) :
  x = 350 ∧ y = 200 := 
sorry

end problem1_problem2_l480_480724


namespace right_triangle_area_valid_right_triangle_perimeter_valid_l480_480754

-- Define the basic setup for the right triangle problem
def hypotenuse : ℕ := 13
def leg1 : ℕ := 5
def leg2 : ℕ := 12  -- Calculated from Pythagorean theorem, but assumed here as condition

-- Define the calculated area and perimeter based on the above definitions
def area (a b : ℕ) : ℕ := (1 / 2) * a * b
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- State the proof goals
theorem right_triangle_area_valid : area leg1 leg2 = 30 :=
  by sorry

theorem right_triangle_perimeter_valid : perimeter leg1 leg2 hypotenuse = 30 :=
  by sorry

end right_triangle_area_valid_right_triangle_perimeter_valid_l480_480754


namespace solution_set_f_gt_0_l480_480202

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else - (x^2 - 2*x - 3)

theorem solution_set_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ (-3 < x ∧ x < 0)} :=
by
  sorry

end solution_set_f_gt_0_l480_480202


namespace max_magnitude_OM_OP_l480_480265

-- Define the points M and O
def M : ℝ × ℝ := (real.sqrt 2, real.sqrt 2)
def O : ℝ × ℝ := (0, 0)

-- Define the unit circle predicate
def on_unit_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Define the vector magnitude function
def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Define the vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Prove the maximum magnitude
theorem max_magnitude_OM_OP (P : ℝ × ℝ) (hP : on_unit_circle P) : 
  vector_magnitude (vector_add M P) ≤ 3 :=
by {
  let OM := vector_magnitude M,
  let OP := vector_magnitude P,
  calc 
    vector_magnitude (vector_add M P) ≤ 
      vector_magnitude M + vector_magnitude P : sorry -- triangle inequality
    ... = 2 + 1 : sorry
    ... ≤ 3 : by norm_num
}

end max_magnitude_OM_OP_l480_480265


namespace sum_first_fifteen_multiples_of_17_l480_480812

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480812


namespace investment_plans_count_l480_480903

noncomputable def investment_plans (projects : Finset ℕ) (cities : Finset ℕ) : ℕ :=
  let scenario1 := (projects.card.choose 2) * (cities.card.choose 2) * (cities.card - 2)
  let scenario2 := (cities.card - 1) * (projects.card.factorial / (projects.card - (cities.card - 1)).factorial)
  scenario1 + scenario2

theorem investment_plans_count : investment_plans {1, 2, 3} {1, 2, 3, 4} = 60 :=
  by sorry

end investment_plans_count_l480_480903


namespace find_angle_theta_l480_480989

noncomputable def complex_exponential_sum : ℂ :=
  complex.exp (11 * real.pi * complex.I / 60) +
  complex.exp (23 * real.pi * complex.I / 60) +
  complex.exp (35 * real.pi * complex.I / 60) +
  complex.exp (47 * real.pi * complex.I / 60) +
  complex.exp (59 * real.pi * complex.I / 60) +
  complex.exp (real.pi * complex.I / 60)

theorem find_angle_theta :
  ∃ r θ, 0 ≤ θ ∧ θ < 2 * real.pi ∧ complex_exponential_sum = r * complex.exp (complex.I * θ) ∧ θ = 7 * real.pi / 12 :=
by
  sorry

end find_angle_theta_l480_480989


namespace boy_height_when_tree_40_l480_480911

-- Define the initial conditions
def B_initial : ℕ := 24
def T_initial : ℕ := 16
def T_final : ℕ := 40

-- The growth rate relationship
def tree_growth_rate_twice_boy : Prop := ∀ G : ℕ, T_final - T_initial = 2 * (G * 2)

-- The final height of the boy
def B_final : ℕ := B_initial + (T_final - T_initial) / 2

theorem boy_height_when_tree_40 : B_initial + (T_final - T_initial) / 2 = 36 :=
by
  -- Step 1: Calculate the growth of the tree
  calc
    T_final - T_initial = 24 : by norm_num
  -- Step 2: Calculate the growth of the boy, which is half the growth of the tree
  calc
    24 / 2 = 12 : by norm_num
  -- Step 3: Add the growth of the boy to the initial height of the boy
  calc
    B_initial + 12 = 36 : by norm_num

end boy_height_when_tree_40_l480_480911


namespace digit_at_2000th_position_l480_480216

-- Definition to represent the infinite sequence of concatenated natural numbers
noncomputable def sequence : ℕ → ℕ 
| n := String.toNat ((List.append (List.join (List.ofFn (fun i => (i + 1).toString)) [])).nthD n "0")

-- Statement to prove the digit at the 2000th position
theorem digit_at_2000th_position : (sequence 1999) = 0 :=
by
  sorry

end digit_at_2000th_position_l480_480216


namespace find_C_l480_480969

-- 1. Condition: Conor can chop 12 eggplants, some carrots (C), and 8 potatoes in a day
def daily_vegetables (C : Nat) : Nat := 12 + C + 8

-- 2. Condition: Conor works 4 times a week
def weekly_vegetables (C : Nat) : Nat := 4 * daily_vegetables C

-- 3. Condition: Conor chops 116 vegetables in a week
axiom weekly_total : ∀ (C : Nat), weekly_vegetables C = 116 

-- Proof to find the number of carrots (C)
theorem find_C : ∃ (C : Nat), weekly_vegetables C = 116 ∧ C = 9 := by
  intro C
  exists 9
  sorry

end find_C_l480_480969


namespace binom_60_2_eq_1770_l480_480530

theorem binom_60_2_eq_1770 : nat.choose 60 2 = 1770 :=
by sorry

end binom_60_2_eq_1770_l480_480530


namespace kevin_watermelons_l480_480280

theorem kevin_watermelons (w1 w2 w_total : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) (h_total : w_total = 14.02) : 
  w1 + w2 = w_total → 2 = 2 :=
by
  sorry

end kevin_watermelons_l480_480280


namespace range_of_f_pos_l480_480699

theorem range_of_f_pos
  (f : ℝ → ℝ)
  (h_even : ∀ x, f(x) = f(-x))
  (h_deriv : ∀ x ≠ 0, deriv f x = f'(x))
  (h_f_neg_one : f(-1) = 0)
  (cond : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1} :=
begin
  sorry
end

end range_of_f_pos_l480_480699


namespace part_I_part_II_part_III_l480_480159

def sequence_a (n : ℕ) : ℤ := (-2) ^ n

def S_star (k : ℕ) (seq : ℕ → ℤ) : ℤ :=
  (Finset.range k).sum (λ i, abs (seq (i + 1)))

def L_k (k : ℕ) (seq : ℕ → ℤ) (c : ℤ) : ℤ :=
  (Finset.range k).sum (λ i, abs (seq (i + 1) - c))

def is_omega_coefficient (k : ℕ) (seq : ℕ → ℤ) (c : ℤ) : Prop :=
  L_k k seq c = S_star k seq

theorem part_I : 
  S_star 4 sequence_a = 30 ∧ is_omega_coefficient 4 sequence_a 2 := 
sorry

def sequence_b (n : ℕ) : ℤ := 3 * n - 39
    
theorem part_II : ∃ m : ℕ, is_omega_coefficient m sequence_b 3 ∧ m = 26 :=
sorry

def arithmetic_sequence (a d n : ℤ) := a + n * d

theorem part_III : ∃ (a d : ℤ) (m : ℕ), 
  is_omega_coefficient m (arithmetic_sequence a d) (-1) ∧
  is_omega_coefficient m (arithmetic_sequence a d) 2 ∧
  S_star m (arithmetic_sequence a d) = 507 ∧
  m = 26 :=
sorry

end part_I_part_II_part_III_l480_480159


namespace tangent_line_equation_l480_480327

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P.2 = P.1^2)
  (h_perpendicular : ∃ k : ℝ, k * -1/2 = -1) : 
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end tangent_line_equation_l480_480327


namespace total_amount_paid_l480_480805

theorem total_amount_paid (B : ℕ) (hB : B = 232) (A : ℕ) (hA : A = 3 / 2 * B) :
  A + B = 580 :=
by
  sorry

end total_amount_paid_l480_480805


namespace ratio_expression_value_l480_480119

theorem ratio_expression_value (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 0) (h_eq : x^2 - y^2 = x + y) : 
  x / y + y / x = 2 + 1 / (y^2 + y) :=
by
  sorry

end ratio_expression_value_l480_480119


namespace find_numbers_l480_480453

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480453


namespace largest_possible_s_l480_480294

-- Definitions based on the problem conditions
def polygon_interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

-- Main theorem statement
theorem largest_possible_s (r s : ℕ) (h₁ : r ≥ s) (h₂ : s ≥ 3) 
  (h₃ : polygon_interior_angle r = (5/4) * polygon_interior_angle s) : s ≤ 102 :=
by
  sorry

#eval largest_possible_s -- This line is to test build without actual function call, should be removed in real usage.

end largest_possible_s_l480_480294


namespace angle_PAO_eq_angle_QAO_l480_480393

theorem angle_PAO_eq_angle_QAO (A B C O P Q : Point) (Γ : Circle) (α β γ : Angle) :
  -- Condition 1: Triangle ABC has circumcircle Γ
  In_circle (Γ, A) ∧ In_circle (Γ, B) ∧ In_circle (Γ, C) →
  -- Condition 2: Circle with center O is tangent to BC at P and internally to Γ at Q
  Tangent_at (Circle_center_radius O (dist O P), BC, P) ∧ Tangent_at (Circle_center_radius O (dist O Q), Γ, Q) →
  -- Condition 3: Q lies on the arc BC of Γ not containing A
  On_arc (Γ, B, C, Q, A) →
  -- Condition 4: ∠BAO = ∠CAO
  ∠ (B, A, O) = ∠ (C, A, O) →
  -- Conclusion: ∠PAO = ∠QAO
  ∠ (P, A, O) = ∠ (Q, A, O) :=
by
  intros h_circle h_tangency h_arc h_angle_eq
  sorry

end angle_PAO_eq_angle_QAO_l480_480393


namespace minimum_value_u_l480_480170

noncomputable theory

open Complex

def min_value_u (z : ℂ) : ℝ :=
  abs (z^2 - z + 1)

theorem minimum_value_u (z : ℂ) (h : abs z = 2) : 
  ∃ (u_min : ℝ), u_min = (3 / 2) * real.sqrt 3 ∧ ∀ w : ℂ, abs w = 2 → min_value_u w ≥ u_min :=
sorry

end minimum_value_u_l480_480170


namespace remainder_is_v_l480_480918

theorem remainder_is_v (x y u v : ℤ) (hx : x > 0) (hy : y > 0)
  (hdiv : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + (2 * u + 1) * y) % y = v :=
by
  sorry

end remainder_is_v_l480_480918


namespace domain_of_f_l480_480743

noncomputable def f (x : ℝ) : ℝ :=  1 / Real.sqrt (1 - x^2) + x^0

theorem domain_of_f (x : ℝ) : (x > -1 ∧ x < 1 ∧ x ≠ 0) ↔ (x ∈ (-1, 0) ∨ x ∈ (0, 1)) :=
by
  sorry

end domain_of_f_l480_480743


namespace incorrect_statement_l480_480759

-- Definition of the conditions
def cond1 : Prop :=
  ∀ {A : Type}, (AIDS_patient : A) → (healthy_individual : A) →
  probability_cancer (AIDS_patient) > probability_cancer (healthy_individual)

def cond2 : Prop :=
  ∀ {C : Type}, (cancer_cell : C) →
  abnormal_differentiation cancer_cell → 
  can_proliferate_infinitely cancer_cell ∧ 
  can_metastasize cancer_cell

def cond3 : Prop :=
  ∀ {N : Type}, (nitrite : N) →
  can_alter_gene_structure nitrite → 
  carcinogen nitrite

-- The problem statement to prove
theorem incorrect_statement : cond1 ∧ cond2 ∧ cond3 → ¬ (long_term_contact_with_cancer_patients_increases_cancer_risk_for_healthy_individuals) :=
by
  intros,
  sorry

end incorrect_statement_l480_480759


namespace binary_arithmetic_l480_480535

/-- Define the function for binary addition. 
Notice that we do not need to define the detailed steps of binary addition and subtraction here; 
we can use built-in Lean functions if available, or we can assume their correctness. -/
def binary_add (a b : ℕ) : ℕ := nat.bitwise bxor a b

/-- Define the function for binary subtraction.
As above, we can assume the correctness of basic arithmetic operations. -/
def binary_sub (a b : ℕ) : ℕ := nat.sub a b

theorem binary_arithmetic : 
  (binary_sub (binary_add 0b10101 0b11011) 0b1010) = 0b110110 := 
by
  sorry

end binary_arithmetic_l480_480535


namespace wrongly_entered_mark_l480_480497

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ marks_instead_of_45 number_of_pupils (total_avg_increase : ℝ),
     marks_instead_of_45 = 45 ∧
     number_of_pupils = 44 ∧
     total_avg_increase = 0.5 →
     x = marks_instead_of_45 + total_avg_increase * number_of_pupils) →
  x = 67 :=
by
  intro h
  sorry

end wrongly_entered_mark_l480_480497


namespace repeating_decimals_subtraction_l480_480021

theorem repeating_decimals_subtraction :
  let x := (⟨2345, 9999⟩ : ℚ) in
  let y := (⟨6789, 9999⟩ : ℚ) in
  let z := (⟨1234, 9999⟩ : ℚ) in
  x - y - z = (⟨-5678, 9999⟩ : ℚ) :=
by
  sorry

end repeating_decimals_subtraction_l480_480021


namespace binom_10_4_l480_480106

theorem binom_10_4 : Nat.choose 10 4 = 210 := 
by sorry

end binom_10_4_l480_480106


namespace find_two_numbers_l480_480457

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480457


namespace maruska_initial_coins_placed_maruska_consistent_withdrawals_l480_480312

-- Define initial conditions and establish the context
variable (coins : ℕ)

-- Theorem for the first part of the problem
theorem maruska_initial_coins_placed (doubled : ∀ n : ℕ, 2 * n) :
  ∀ (withdrawal_amount : ℕ), (withdrawal_amount = 40) →
  ∀ (coins : ℕ), coins = 35 →
  ∀ (Thursday_withdrawal : coins - 40 = 0),
  ∀ (Wednesday_withdrawal : (coins / 2 + 40) = coins),
  ∀ (Tuesday_withdrawal : ((coins / 2 + 40) / 2 + 40) = coins),
  ∀ (Monday_placing : (((coins / 2 + 40) / 2 + 40) / 2 = coins)),
  coins = 35 :=
begin
  sorry
end

-- Theorem for the second part of the problem
theorem maruska_consistent_withdrawals (doubled : ∀ n : ℕ, 2 * n) :
  ∀ (withdrawal_amount : ℕ), (withdrawal_amount = 40) →
  ∀ (initial_coins : ℕ), 
  (initial_coins / 2 = withdrawal_amount) →
  initial_coins = 40 := 
begin
  sorry
end

end maruska_initial_coins_placed_maruska_consistent_withdrawals_l480_480312


namespace find_k_intersect_lines_l480_480362

theorem find_k_intersect_lines :
  ∃ (k : ℚ), ∀ (x y : ℚ), 
  (2 * x + 3 * y + 8 = 0) → (x - y - 1 = 0) → (x + k * y = 0) → k = -1/2 :=
by sorry

end find_k_intersect_lines_l480_480362


namespace num_valid_pairs_l480_480738

theorem num_valid_pairs : 
  let count_pairs := (λ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (b - a = 2)),
  num_of_pairs := (finset.univ.filter (λ p : ℕ × ℕ, (count_pairs p.1 p.2))).card
  in num_of_pairs = 5 :=
by 
  /- Additional context is provided to highlight the proof setup.
  Definitions for pairs of digits a and b such that conditions are satisfied. -/
  let count_pairs := (λ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (b - a = 2)),
  num_of_pairs := (finset.univ.filter (λ p : ℕ × ℕ, (count_pairs p.1 p.2))).card,
  sorry

end num_valid_pairs_l480_480738


namespace find_numbers_l480_480441

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480441


namespace probability_sin_cos_l480_480331

theorem probability_sin_cos (h : ∀ x, x ∈ set.Icc 0 real.pi) :
  (set.Icc 0 real.pi).measure (λ x, sin x + cos x ≥ sqrt 6 / 2) = 1 / 3 :=
sorry

end probability_sin_cos_l480_480331


namespace baseball_card_final_value_l480_480066

theorem baseball_card_final_value 
  (initial_value : ℝ)
  (decrease_first_year : ℝ)
  (decrease_second_year : ℝ)
  (decrease_third_year : ℝ)
  (decrease_fourth_year : ℝ)
  (value_after_first_year : ℝ := initial_value * (1 - decrease_first_year))
  (value_after_second_year : ℝ := value_after_first_year * (1 - decrease_second_year))
  (value_after_third_year : ℝ := value_after_second_year * (1 - decrease_third_year))
  (value_after_fourth_year : ℝ := value_after_third_year * (1 - decrease_fourth_year)) :
  value_after_fourth_year ≈ 69.22 :=
sorry

end baseball_card_final_value_l480_480066


namespace quadratic_function_is_parabola_l480_480719

theorem quadratic_function_is_parabola (a : ℝ) (b : ℝ) (c : ℝ) :
  ∃ k h, ∀ x, (y = a * (x - h)^2 + k) ∧ a ≠ 0 → (y = 3 * (x - 2)^2 + 6) → (a = 3 ∧ h = 2 ∧ k = 6) → ∀ x, (y = 3 * (x - 2)^2 + 6) := 
by
  sorry

end quadratic_function_is_parabola_l480_480719


namespace product_has_k_plus_one_prime_divisors_l480_480288

theorem product_has_k_plus_one_prime_divisors
  (k : ℕ) (h_k : k > 1)
  (a : ℕ → ℕ)
  (h_mono : ∀ i j, i < j → a i < a j) :
  ∃ p, nat.prime p ∧
  (finset.card (finset.filter (λ p, nat.prime p) (finset.image (λ (i : ℕ × ℕ), a i.1 + a i.2) {i | i.1 < i.2}.to_finset)) > k) :=
sorry

end product_has_k_plus_one_prime_divisors_l480_480288


namespace instantaneous_velocity_at_t5_l480_480079

noncomputable def s (t : ℝ) : ℝ := 4 * t ^ 2 - 3

theorem instantaneous_velocity_at_t5 : (deriv s 5) = 40 := by
  sorry

end instantaneous_velocity_at_t5_l480_480079


namespace triangle_angle_C_eq_triangle_max_area_l480_480655

theorem triangle_angle_C_eq (a b c : ℝ) (A B C : ℝ)
  (h₀ : (2 * a + b) * Real.cos C + c * Real.cos B = 0)
  (h₁ : 0 < C)
  (h₂ : C < Real.pi)
  : C = 2 * Real.pi / 3 := sorry

theorem triangle_max_area (a b c : ℝ)
  (C : ℝ)
  (h₀ : (2 * a + b) * Real.cos C + c * Real.cos (Real.acos (c^2 - a^2 - b^2 + 2*a*b * (Real.cos C))) = 0)
  (h₁ : c = 6)
  (h₂ : C = 2 * Real.pi / 3)
  : ∃ (S : ℝ), S = 3 * Real.sqrt 3 := sorry

end triangle_angle_C_eq_triangle_max_area_l480_480655


namespace group_size_l480_480348

theorem group_size (n : ℕ) (T : ℕ) (h1 : T = 14 * n) (h2 : T + 32 = 16 * (n + 1)) : n = 8 :=
by
  -- We skip the proof steps
  sorry

end group_size_l480_480348


namespace matrix_not_invertible_values_l480_480571

theorem matrix_not_invertible_values (x y z : ℝ)
  (h : det ![![x, y, z], ![y, z, x], ![z, x, y]] = 0) :
  (∃ v : ℝ, v = (x / (y + z) + y / (z + x) + z / (x + y)) ∧ (v = -3 ∨ v = 3 / 2)) :=
sorry

end matrix_not_invertible_values_l480_480571


namespace correct_completion_at_crossroads_l480_480947

theorem correct_completion_at_crossroads :
  (∀ (s : String), 
    s = "An accident happened at a crossroads a few meters away from a bank" → 
    (∃ (general_sense : Bool), general_sense = tt)) :=
by
  sorry

end correct_completion_at_crossroads_l480_480947


namespace brendan_remaining_money_l480_480957

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l480_480957


namespace LCM_of_18_and_27_l480_480016

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l480_480016


namespace sum_of_first_fifteen_multiples_of_17_l480_480838

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480838


namespace find_numbers_l480_480445

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480445


namespace bob_age_sum_digits_l480_480508

theorem bob_age_sum_digits
  (A B C : ℕ)  -- Define ages for Alice (A), Bob (B), and Carl (C)
  (h1 : C = 2)  -- Carl's age is 2
  (h2 : B = A + 2)  -- Bob is 2 years older than Alice
  (h3 : ∃ n, A = 2 * n ∧ n > 0 ∧ n ≤ 8 )  -- Alice's age is a multiple of Carl's age today, marking the second of the 8 such multiples 
  : ∃ n, (B + n) % (C + n) = 0 ∧ (B + n) = 50 :=  -- Prove that the next time Bob's age is a multiple of Carl's, Bob's age will be 50
sorry

end bob_age_sum_digits_l480_480508


namespace expression_equality_l480_480857

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l480_480857


namespace sum_of_digits_product_l480_480373

def A : ℕ := 10^2021 - 1
def B : ℕ := 10^2020 - 1
def product : ℕ := A * B

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_product :
  sum_of_digits product = 18189 :=
by sorry

end sum_of_digits_product_l480_480373


namespace volume_ratio_GAEF_to_ABC_A₁B₁C₁_l480_480670

variables (A B C A₁ B₁ C₁ E F G: Type)
variables [geometry A B C A₁ B₁ C₁]
variables 
(AB AC AA₁: ℝ)
(h: homogeneity AB AC AA₁)
(E_on_AB: E ∈ AB)
(F_on_AC: F ∈ AC)
(G_on_AA₁: G ∈ AA₁)

-- Fractions of segments given in the problem
variables 
(AE_ratio: AE = (1 / 2) * AB)
(AF_ratio: AF = (1 / 3) * AC)
(AG_ratio: AG = (2 / 3) * AA₁)

theorem volume_ratio_GAEF_to_ABC_A₁B₁C₁ :
  V_G_AEF / V_ABC_A₁B₁C₁ = 1 / 27 := sorry

end volume_ratio_GAEF_to_ABC_A₁B₁C₁_l480_480670


namespace ratio_AH_HF_EQ_AC_CD_l480_480528

universe u 
variables {X : Type u} 
variables (O1 O2 : circle X) (B C A E F H G D : X) 
variables (tangent_O1_C : tangent O1 C) (line_AF : line (A, F)) (line_AC_ext : extended_line (A, C) (B, G, D))

-- Conditions
variables (intersect_O1_O2 : intersect O1 O2 = {B, C}) 
variables (BC_diameter_O1 : diameter O1 B C)
variables (C_tangent_A_intersect_O2 : tangent_O1_C.intersect O2 = {A})
variables (AB_intersect_O1_E : segment (A, B).intersect O1 = {E})
variables (CE_extended_intersect_O2 : line (C, E).extended.intersect O2 = {F})
variables (H_on_AF : point_on_line_segment H (A, F))
variables (HE_extended_intersect_O1 : line (H, E).extended.intersect O1 = {G})
variables (BG_ext_AC_D : line_AC_ext (B, D))

-- Goal
theorem ratio_AH_HF_EQ_AC_CD :
  ∀ (O1 O2 : circle X) (B C A E F H G D : X)
  (intersect_O1_O2 : intersect O1 O2 = {B, C})
  (BC_diameter_O1 : diameter O1 B C)
  (tangent_O1_C : tangent O1 C)
  (C_tangent_A_intersect_O2 : tangent_O1_C.intersect O2 = {A})
  (AB_intersect_O1_E : segment (A, B).intersect O1 = {E})
  (CE_extended_intersect_O2 : line (C, E).extended.intersect O2 = {F})
  (H_on_AF : point_on_line_segment H (A, F))
  (HE_extended_intersect_O1 : line (H, E).extended.intersect O1 = {G})
  (line_AC_ext : extended_line (A, C) (B, G, D)),
  ((AH / HF) = (AC / CD)) :=
sorry

end ratio_AH_HF_EQ_AC_CD_l480_480528


namespace inequality_one_solution_l480_480123

theorem inequality_one_solution (a : ℝ) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 4 → x = -a) ↔ a = 2 :=
by sorry

end inequality_one_solution_l480_480123


namespace problem_solution_l480_480567

noncomputable def y (n : ℕ) (x : ℝ) : ℝ := x^n * (1 - x)

-- Define the derivative of y
noncomputable def dy_dx (n : ℕ) (x : ℝ) : ℝ := n * x^(n - 1) - (n + 1) * x^n

-- Define the slope of the tangent line at x = 2
noncomputable def slope (n : ℕ) : ℝ := dy_dx n 2

-- Define the tangent point at x = 2
noncomputable def tangent_point (n : ℕ) : ℝ × ℝ := (2, -2^n)

-- Define the equation of the tangent line and find the ordinate a_n
noncomputable def a_n (n : ℕ) : ℝ := (n + 1) * 2^n

-- Define b_n
def b_n (n : ℕ) : ℝ := Real.log2 (a_n n / (n + 1))

-- Define the sum of the first 10 terms of the sequence {b_n}
noncomputable def sum_first_10_terms : ℝ := (Finset.range 10).sum (λ n, b_n n)

theorem problem_solution : sum_first_10_terms = 55 := by
  sorry

end problem_solution_l480_480567


namespace graph_of_f_shifted_is_C_l480_480356

noncomputable def f : ℝ → ℝ :=
  λ x, if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
       else if (0 : ℝ) ≤ x ∧ x ≤ 2 then Real.sqrt(4 - (x - 2)^2) - 2
       else if (2 : ℝ) ≤ x ∧ x ≤ 3 then 2 * (x - 2)
       else 0

noncomputable def f_shifted := λ x, f (x + 3)

theorem graph_of_f_shifted_is_C :
  ∀ x, f_shifted x = (λ x, f (x + 3)) x :=
by
  sorry

end graph_of_f_shifted_is_C_l480_480356


namespace ratio_upstream_downstream_l480_480772

noncomputable def ratio_time_upstream_to_downstream
  (V_b V_s : ℕ) (T_u T_d : ℕ) : ℕ :=
(V_b + V_s) / (V_b - V_s)

theorem ratio_upstream_downstream
  (V_b V_s : ℕ) (hVb : V_b = 48) (hVs : V_s = 16) (T_u T_d : ℕ)
  (hT : ratio_time_upstream_to_downstream V_b V_s T_u T_d = 2) :
  T_u / T_d = 2 := by
  sorry

end ratio_upstream_downstream_l480_480772


namespace determine_a_for_unique_solution_of_quadratic_l480_480621

theorem determine_a_for_unique_solution_of_quadratic :
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0} = {0, 2} :=
sorry

end determine_a_for_unique_solution_of_quadratic_l480_480621


namespace lesser_fraction_l480_480379

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l480_480379


namespace crayons_per_pack_l480_480317

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end crayons_per_pack_l480_480317


namespace problem_1_problem_2_problem_3_l480_480063

variables {α β : ℝ}

def vector_a (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.sin β, 4 * Real.cos β)
def vector_c (β : ℝ) : ℝ × ℝ := (Real.cos β, -4 * Real.sin β)
def tan_alpha_beta (α β : ℝ) : ℝ := Real.tan (α + β)

theorem problem_1 (h : vector_a α ⬝ (vector_b β - 2 • vector_c β) = 0) :
  tan_alpha_beta α β = 2 :=
by sorry

theorem problem_2 :
  ∃ β : ℝ, ∀ b c : ℝ × ℝ, b = vector_b β → c = vector_c β → |b + c| ≤ 4 * Real.sqrt 2 :=
by sorry

theorem problem_3 (h : Real.tan α * Real.tan β = 16) :
  ∃ k : ℝ, vector_a α = k • vector_b β :=
by sorry

end problem_1_problem_2_problem_3_l480_480063


namespace percentage_A_spends_l480_480472

-- Define the constants and assumptions
constant S_A : ℝ := 2250
constant combined_salary : ℝ := 3000
constant P_B : ℝ := 0.85

-- Adding the assumptions
axiom combined_salary_eq : S_A + S_B = combined_salary
axiom equal_savings : S_A * (1 - P_A) = S_B * (1 - P_B)

-- The main theorem to prove
theorem percentage_A_spends : P_A = 0.95 :=
by
  have S_B := combined_salary - S_A
  have savings_B := S_B * (1 - P_B)
  have equal_savings' := equal_savings S_A 2250 S_B
  have equal_savings'' := equal_savings' . subst S_B + subst savings_B
  sorry
  
end percentage_A_spends_l480_480472


namespace students_left_during_year_l480_480661

theorem students_left_during_year (initial_students : ℕ) (new_students : ℕ) (final_students : ℕ) (students_left : ℕ) :
  initial_students = 4 →
  new_students = 42 →
  final_students = 43 →
  students_left = initial_students + new_students - final_students →
  students_left = 3 :=
by
  intro h_initial h_new h_final h_students_left
  rw [h_initial, h_new, h_final] at h_students_left
  exact h_students_left

end students_left_during_year_l480_480661


namespace gcd_lcm_42_30_l480_480994

noncomputable def gcd_lcm_theorem : Prop :=
  let common_prime_factors : List ℕ := [2, 3]
  let unique_prime_factors_42 : List ℕ := [7]
  let unique_prime_factors_30 : List ℕ := [5]
  ∀ (a b : ℕ), 
    a = 42 → b = 30 → 
    gcd a b = (common_prime_factors.product id) ∧ 
    lcm a b = (common_prime_factors.product id) * (unique_prime_factors_42.product id) * (unique_prime_factors_30.product id)

theorem gcd_lcm_42_30 : gcd_lcm_theorem :=
by
  sorry

end gcd_lcm_42_30_l480_480994


namespace sum_of_first_fifteen_multiples_of_17_l480_480843

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480843


namespace domain_of_f_l480_480745

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l480_480745


namespace problem_1_problem_2_l480_480132

-- Definitions of the functions and conditions
def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 (x : ℝ) : f(x) ≤ 5 - f(x - 1) ↔ - (1 / 4) ≤ x ∧ x ≤ (9 / 4) := sorry

noncomputable def M := {x : ℝ | f(x) ≤ f(x + a) - |x - a|}

theorem problem_2 (a : ℝ) (H : (1 / 2 < x ∧ x < 1) → (x ∈ M)) : -1 ≤ a ∧ a ≤ (5 / 2) := sorry

end problem_1_problem_2_l480_480132


namespace brandon_cards_l480_480520

theorem brandon_cards (b m : ℕ) 
  (h1 : m = b + 8) 
  (h2 : 14 = m / 2) : 
  b = 20 := by
  sorry

end brandon_cards_l480_480520


namespace sum_of_first_fifteen_multiples_of_17_l480_480849

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480849


namespace series_result_l480_480303

noncomputable def series_evaluation (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : a > b) : ℝ :=
  ∑' (n : ℕ) in finset.range (n+1), (1 / (((n : ℝ) - 1) * a^3 - ((n : ℝ) - 2) * b^3) / (n * a^3 - ((n : ℝ) - 1) * b^3))

theorem series_result (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : a > b) :
  series_evaluation a b ha_pos hb_pos h = 1 / ((a^3 - b^3) * b^3) :=
sorry

end series_result_l480_480303


namespace fraction_division_l480_480019

theorem fraction_division (a b c d : ℚ) (h1 : a = 3) (h2 : b = 8) (h3 : c = 5) (h4 : d = 12) :
  (a / b) / (c / d) = 9 / 10 :=
by
  sorry

end fraction_division_l480_480019


namespace find_abs_z_l480_480694

noncomputable def complex_proof_problem (z w : ℂ) (h1 : abs (3 * z - w) = 15)
  (h2 : abs (z + 3 * w) = 9) (h3 : abs (z + w) = 6) : Prop :=
abs z = Real.sqrt 15

-- statement of the theorem
theorem find_abs_z (z w : ℂ) (h1 : abs (3 * z - w) = 15)
  (h2 : abs (z + 3 * w) = 9) (h3 : abs (z + w) = 6) : complex_proof_problem z w h1 h2 h3 :=
begin
  sorry
end

end find_abs_z_l480_480694


namespace excircle_opposite_side_b_l480_480672

-- Definition of the terms and assumptions
variables {a b c : ℝ} -- sides of the triangle
variables {r r1 : ℝ}  -- radii of the circles

-- Given conditions
def touches_side_c_and_extensions_of_a_b (r : ℝ) (a b c : ℝ) : Prop :=
  r = (a + b + c) / 2

-- The goal to be proved
theorem excircle_opposite_side_b (a b c : ℝ) (r1 : ℝ) (h1 : touches_side_c_and_extensions_of_a_b r a b c) :
  r1 = (a + c - b) / 2 := 
by
  sorry

end excircle_opposite_side_b_l480_480672


namespace y_coordinate_eq_l480_480005

theorem y_coordinate_eq (y : ℝ) : 
  (∃ y, (√(9 + y^2) = √(4 + (5 - y)^2))) ↔ (y = 2) :=
by
  sorry

end y_coordinate_eq_l480_480005


namespace binomial_expansion_l480_480411

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l480_480411


namespace stool_height_is_correct_l480_480507

noncomputable def height_of_stool : ℕ :=
let
  ceiling_height := 300,         -- Ceiling height in centimeters
  alice_height := 150,           -- Alice's height in centimeters
  additional_reach := 50,        -- Additional reach above her head in centimeters
  light_bulb_distance := 15,     -- Distance of the light bulb below the ceiling in centimeters
  decoration_dist := 5           -- Distance of the decoration below the light bulb in centimeters
in
  let alice_total_reachable_height := alice_height + additional_reach in
  let light_bulb_height := ceiling_height - light_bulb_distance in
  let effective_height_needed := light_bulb_height - decoration_dist in
  effective_height_needed - alice_total_reachable_height

theorem stool_height_is_correct : height_of_stool = 80 :=
  by
    unfold height_of_stool
    unfold effective_height_needed
    unfold light_bulb_height
    unfold alice_total_reachable_height
    rw [←Nat.cast_add, ←Nat.cast_sub]
    norm_num

end stool_height_is_correct_l480_480507


namespace nth_positive_integer_not_of_F_form_l480_480022

noncomputable def G (n : ℕ) : ℕ :=
  n + Int.floor (Real.log (n.to_real + 1 + Int.floor (Real.log (n.to_real + 1))))

noncomputable def F (m : ℝ) : ℤ :=
  Int.floor (Real.exp m)

theorem nth_positive_integer_not_of_F_form (n : ℕ) (m : ℝ) (hm : m ≥ 1) :
  ∀ m, G(n) = n + Int.floor (Real.log (n.to_real + 1 + Int.floor (Real.log (n.to_real + 1)))) :=
sorry

end nth_positive_integer_not_of_F_form_l480_480022


namespace ellipse_equation_constant_slope_ratio_point_on_fixed_line_l480_480181

-- Prove that given an ellipse with the specified properties, its equation is as follows
theorem ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (e_eq : e = √3 / 2)
  (F : ℝ × ℝ) (F_eq : F = (√3, 0)) : 
  (a = 2 ∧ b = 1) → ∀ x y, (x^2 / 4) + y^2 = 1 :=
by
  sorry

-- Prove the ratio of slopes k1 and k2 is constant
theorem constant_slope_ratio (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (e_eq : e = √3 / 2)
  (F : ℝ × ℝ) (F_eq : F = (√3, 0)) (D : ℝ × ℝ) (D_eq : D = (1, 0)) :
  ∀ l P Q (h_intersect : intersects l P Q) A B k1 k2, 
    (k1 = slope A P ∧ k2 = slope B Q) → (k1 / k2 = 1 / 3) :=
by
  sorry

-- Prove that point M lies on a fixed line x = 4
theorem point_on_fixed_line (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (e_eq : e = √3 / 2)
  (F : ℝ × ℝ) (F_eq : F = (√3, 0)) (D : ℝ × ℝ) (D_eq : D = (1, 0)) :
  ∀ l P Q A B k1 k2 M (h_intersect_lines : M ∈ intersects_lines (A, P) (B, Q)),
    (k1 / k2 = 1 / 3) → (M.x = 4) :=
by
  sorry

end ellipse_equation_constant_slope_ratio_point_on_fixed_line_l480_480181


namespace amplitude_sine_wave_l480_480102

theorem amplitude_sine_wave (a b c d : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_oscillates : ∀ x, y = a * sin (b * x + c) + d → -2 ≤ y ∧ y ≤ 4) : a = 3 :=
by
  sorry

end amplitude_sine_wave_l480_480102


namespace polynomial_102_l480_480414

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l480_480414


namespace inequality_proof_l480_480193

variable (α β γ t : ℝ)

-- Conditions
hypothesis (h1 : 4 * α^2 - 4 * t * α - 1 = 0)
hypothesis (h2 : 4 * β^2 - 4 * t * β - 1 = 0)
hypothesis (h3 : α ≠ β)
hypothesis (h4 : α + β = t)
hypothesis (h5 : α * β = -1/4)
hypothesis (h6: ∀ x ∈ [α, β], x ∈ ℝ -> (2 * x - t) / (x^2 + 1) = f(x))
hypothesis (h7 : sin α + sin β + sin γ = 1)
hypothesis (h8 : α ∈ (0, π/2) ∧ β ∈ (0, π/2) ∧ γ ∈ (0, π/2))

-- Definitions
noncomputable def f (x : ℝ) : ℝ := (2 * x - t) / (x^2 + 1)
noncomputable def g (t : ℝ) : ℝ := f (t) (x)_{\text max} - f (t) (x)_{\text min}

-- Theorem
theorem inequality_proof : (1 / g (tan α)) + (1 / g (tan β)) + (1 / g (tan γ)) < (3 / 4) * sqrt 6 := 
by 
sorry

end inequality_proof_l480_480193


namespace binom_60_2_eq_1770_l480_480529

theorem binom_60_2_eq_1770 : nat.choose 60 2 = 1770 :=
by sorry

end binom_60_2_eq_1770_l480_480529


namespace train_speed_l480_480933

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ) :
  length_train = 140 →
  length_bridge = 235.03 →
  time_seconds = 30 →
  (length_train + length_bridge) / time_seconds * 3.6 = 45.0036 :=
by
  intros h1 h2 h3
  sorry

end train_speed_l480_480933


namespace small_cylinders_filled_l480_480895

open Real

noncomputable def largeCylinderVolume : ℝ :=
  let r := 3 -- radius in meters (half of diameter 6)
  let h := 8 -- height in meters
  π * r^2 * h

noncomputable def smallCylinderVolume : ℝ :=
  let r1 := 2 -- base radius in meters
  let r2 := 1 -- top radius in meters
  let h := 5 -- height in meters
  (1 / 3) * π * h * (r1^2 + r1 * r2 + r2^2)

noncomputable def numberOfSmallCylinders : ℕ :=
  ⌊ largeCylinderVolume / smallCylinderVolume ⌋

theorem small_cylinders_filled (largeVolume smallVolume : ℝ)
    (h₁ : largeVolume = largeCylinderVolume)
    (h₂ : smallVolume = smallCylinderVolume) :
    numberOfSmallCylinders = 6 :=
by
  rw [h₁, h₂]
  have h : numberOfSmallCylinders = ⌊ 72 * π / ((35 / 3) * π) ⌋ := rfl
  -- by simplifying, we know that numberOfSmallCylinders = 6
  exact h

#eval small_cylinders_filled largeCylinderVolume smallCylinderVolume sorry sorry -- expected output is the Lean theorem "small_cylinders_filled" with numberOfSmallCylinders == 6.

end small_cylinders_filled_l480_480895


namespace domain_of_log_function_l480_480126

open Real

noncomputable def f (x : ℝ) : ℝ := log 2 (2^x - 3^x)

theorem domain_of_log_function : {x : ℝ | f x ∈ ℝ} = Iio 0 := by
  sorry

end domain_of_log_function_l480_480126


namespace benny_spent_on_baseball_gear_l480_480954

variable initial_amount : ℕ
variable remaining_amount : ℕ

theorem benny_spent_on_baseball_gear (h_initial: initial_amount = 79) (h_remaining: remaining_amount = 32) :
  initial_amount - remaining_amount = 47 :=
by
  sorry

end benny_spent_on_baseball_gear_l480_480954


namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l480_480093

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l480_480093


namespace distance_from_point_to_line_l480_480268

noncomputable def point := (1 : ℝ, 0 : ℝ)
noncomputable def line (x : ℝ) := x = 2

theorem distance_from_point_to_line (P : ℝ × ℝ) (L : ℝ → Prop) : 
  P = point → L = line → distance P L = 1 :=
by
  sorry

end distance_from_point_to_line_l480_480268


namespace num_knights_at_table_is_30_l480_480781

-- Define the types of people (knight or liar)
inductive Person : Type
| knight : Person
| liar : Person

-- Predicate to check if a person always tells the truth
def tells_truth (p : Person) : Prop :=
  Person.recOn p (λ x, true) (λ x, false)

-- Main theorem statement
theorem num_knights_at_table_is_30
  (P : Fin 60 → Person)
  (H : ∀ i : Fin 60, let next_three := [P ((i + 1) % 60), P ((i + 2) % 60), P ((i + 3) % 60)]
    in tells_truth (P i) → (next_three.count tells_truth ≤ 1)) :
  (Finset.univ.filter (λ i, tells_truth (P i))).card = 30 :=
sorry

end num_knights_at_table_is_30_l480_480781


namespace max_lattice_points_on_circle_at_most_one_l480_480057

theorem max_lattice_points_on_circle_at_most_one
  (r : ℝ) (hr : 0 < r) :
  ∀ (x y : ℤ), 
  (x - real.sqrt 2)^2 + (y - real.sqrt 3)^2 = r^2 →
  ∀ (x1 y1 : ℤ),
  (x1 - real.sqrt 2)^2 + (y1 - real.sqrt 3)^2 = r^2 →
  (x, y) = (x1, y1) :=
begin
  sorry
end

end max_lattice_points_on_circle_at_most_one_l480_480057


namespace conjugate_z_l480_480605

def z : ℂ := (2 - complex.i)^2

theorem conjugate_z : conj z = 3 + 4*complex.i :=
by
  unfold z
  sorry

end conjugate_z_l480_480605


namespace completing_the_square_result_l480_480034

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l480_480034


namespace find_numbers_sum_eq_S_product_eq_P_l480_480432

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480432


namespace minimum_workers_needed_l480_480890

theorem minimum_workers_needed 
  (daily_maintenance_fee : ℕ) 
  (worker_wage_per_hour : ℕ) 
  (worker_productivity : ℕ)
  (selling_price_per_pen : ℝ)
  (workday_duration_hours : ℕ) 
  (n : ℕ)
  (total_costs : ℕ := daily_maintenance_fee + (workday_duration_hours * worker_wage_per_hour * n))
  (total_revenue : ℝ := (workday_duration_hours * worker_productivity * selling_price_per_pen * n) ) :
  total_revenue > (total_costs : ℝ) → n ≥ 167 :=  sorry

variables daily_maintenance_fee worker_wage_per_hour worker_productivity 
  selling_price_per_pen workday_duration_hours n

#eval minimum_workers_needed 600 20 7 2.8 9 167

end minimum_workers_needed_l480_480890


namespace find_two_numbers_l480_480461

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480461


namespace tiling_tetromino_divisibility_l480_480978

theorem tiling_tetromino_divisibility (n : ℕ) : 
  (∃ (t : ℕ), n = 4 * t) ↔ (∃ (k : ℕ), n * n = 4 * k) :=
by
  sorry

end tiling_tetromino_divisibility_l480_480978


namespace find_numbers_l480_480448

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480448


namespace sides_of_triangle_l480_480657

theorem sides_of_triangle (a l_alpha : ℝ) (alpha : ℝ) :
  ∃ (b c : ℝ),
    b = (l_alpha * cos (alpha / 2) + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^(1 / 2)
                 + ((l_alpha^2 * cos (alpha / 2) (l_alpha + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^(1 / 2)) /
                    (2 * cos (alpha / 2))) / 2)
            + ( (l_alpha * cos (alpha / 2) + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^(1 / 2))^2 
              - 4 * (l_alpha^2 * cos (alpha / 2) (l_alpha + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^[1 / 2])/ 
              (2 * cos (alpha / 2))) ) ^ (1 / 2) / 2)
    ∧ c = (l_alpha * cos (alpha / 2) + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^(1 / 2)
                 - ((l_alpha^2 * cos (alpha / 2) (l_alpha + (l_alpha^2 * cos (alpha / 2)^2 + a^2) ^ (1 / 2)) /
                    (2 * cos (alpha / 2))) / 2)
            - ( (l_alpha * cos (alpha / 2) + (l_alpha^2 * cos (alpha / 2)^2 + a^2)^(1 / 2))^2 
              - 4 * (l_alpha^2 * cos (alpha / 2) [ l_alpha + (l_alpha^2 * cos (alpha / 2)^2 + a^2) ^ (1 / 2]) / 
              (2 * cos (alpha / 2))) ) ^ (1 / 2) / 2) := 
sorry

end sides_of_triangle_l480_480657


namespace bug_on_square_moves_l480_480886

theorem bug_on_square_moves :
  ∃ (m n : ℕ), (∀ k < n, Nat.gcd k n = 1) ∧
  (Q : ℕ → ℚ) (Q_0 : Q 0 = 1) (Q_n : ∀ n, Q (n + 1) = 1 / 3 * (1 - Q n))
  (m = 44287 ∧ n = 177147) ∧ 
  (Q 12 = m / n) ∧
  (m + n = 221434) :=
by
  sorry

end bug_on_square_moves_l480_480886


namespace joy_reading_hours_l480_480677

theorem joy_reading_hours :
  ∀ (pages_per_20min : ℕ) (time_per_20min : ℕ) (minutes_per_hour : ℕ)
  (total_pages : ℕ) (expected_hours : ℕ),
    pages_per_20min = 8 →
    time_per_20min = 20 →
    minutes_per_hour = 60 →
    total_pages = 120 →
    expected_hours = 5 →
    (total_pages / (pages_per_20min * (minutes_per_hour / time_per_20min))) = expected_hours :=
by
  intros pages_per_20min time_per_20min minutes_per_hour total_pages expected_hours
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end joy_reading_hours_l480_480677


namespace fifth_friend_contribution_l480_480158

variables (a b c d e : ℕ)

theorem fifth_friend_contribution:
  a + b + c + d + e = 120 ∧
  a = 2 * b ∧
  b = (c + d) / 3 ∧
  c = 2 * e →
  e = 12 :=
sorry

end fifth_friend_contribution_l480_480158


namespace poly_a_roots_poly_b_roots_l480_480562

-- Define the polynomials
def poly_a (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 - 11 * x + 6
def poly_b (x : ℤ) : ℤ := x ^ 4 + 4 * x ^ 3 - 9 * x ^ 2 - 16 * x + 20

-- Assert the integer roots for poly_a
theorem poly_a_roots : {x : ℤ | poly_a x = 0} = {-2, 3} := sorry

-- Assert the integer roots for poly_b
theorem poly_b_roots : {x : ℤ | poly_b x = 0} = {1, 2, -2, -5} := sorry

end poly_a_roots_poly_b_roots_l480_480562


namespace expression_value_l480_480156

theorem expression_value : (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := 
by
  sorry

end expression_value_l480_480156


namespace part1_part2_l480_480001

def custom_op (a b : ℤ) : ℤ := a^2 - b + a * b

theorem part1  : custom_op (-3) (-2) = 17 := by
  sorry

theorem part2 : custom_op (-2) (custom_op (-3) (-2)) = -47 := by
  sorry

end part1_part2_l480_480001


namespace proof_problem_l480_480206

-- Define the lines l1 and l2 based on given conditions
def l1_eq : Prop := 3 * x - y + 9 = 0
def l2_eq : Prop := x - y - 1 = 0

-- Given line l3 equation format
def l3_eq (k : ℝ) : Prop := (1 - 3 * k) * x + (k + 1) * y - 3 * k - 1 = 0

-- Define the final conclusions as Prop
def l1_final_eq : Prop := ∀ x y : ℝ, 3 * x - y + 9 = 0
def l2_final_eq : Prop := ∀ x y : ℝ, x - y - 1 = 0
def l3_final_eq : Prop := ∀ x y : ℝ, ((4 * x - y + 11 = 0) ∨ (5 * x - 2 * y + 16 = 0))

theorem proof_problem :
  l1_eq ∧ l2_eq ∧ (∀ k : ℝ, k ∈ ℝ → l3_eq k) →
  l1_final_eq ∧ l2_final_eq ∧ l3_final_eq :=
by
  -- Proof is required
  sorry

end proof_problem_l480_480206


namespace constant_term_of_expansion_l480_480740

noncomputable def binomial_constant_term : ℕ :=
  let x := 1 -- we use dummy variable x since constant term does not depend on x
  let r := 2 -- as computed from the problem, to get the general term where power of x is zero
  nat.choose 5 r * (-2)^r

theorem constant_term_of_expansion :
  binomial_constant_term = 40 :=
by
  -- Dummy implementation for now, expand it to prove the constant term
  sorry

end constant_term_of_expansion_l480_480740


namespace height_parabola_right_triangle_l480_480091

theorem height_parabola_right_triangle
  (x1 x2 x3 : ℝ)
  (A B C : ℝ × ℝ)
  (hA : A = (x1, x1^2))
  (hB : B = (x2, x2^2))
  (hC : C = (x3, x3^2))
  (right_angle : ∃ C : ℝ × ℝ, ∃ D : ℝ × ℝ, C.1 - D.1 ≠ 0)
  (hypotenuse_parallel : A.2 = B.2) :
  let D := (x3, x1^2) in
  x1 ≠ 0 → x2 ≠ 0 → x3 = 0 → abs (x3^2 - x1^2) = 1
:= sorry

end height_parabola_right_triangle_l480_480091


namespace num_funcs_l480_480993

theorem num_funcs (a b c d : ℝ) :
  let f (x : ℝ) := a * x^2 + b * x + c + d * x^3,
      g (x : ℝ) := f x * f (-x) = f (x^3)
  in (b = 0) ∧ (d = 0 ∨ d = 1) ∧ (a = 0 ∨ a = 1) ∧ (c = 0 ∨ c = 1) →
  ∃ (n : ℕ), n = 10 :=
by sorry

end num_funcs_l480_480993


namespace trapezoid_area_sum_l480_480934

theorem trapezoid_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 10) :
  ∃ r1 r2 n1 : ℚ, (r1 * real.sqrt n1 + r2 = 56 ∧ (∃ m, (∃ k, n1 = m * m + 1 ∧ m > 0)) ∧ (∀ p : ℕ, prime p → (p * p) ∣ n1 → false)) :=
sorry

end trapezoid_area_sum_l480_480934


namespace dessert_distribution_l480_480767

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end dessert_distribution_l480_480767


namespace pure_gala_trees_l480_480046

variable (T F G : ℝ)

theorem pure_gala_trees (h1 : F + 0.1 * T = 170) (h2 : F = 0.75 * T): G = T - F -> G = 50 :=
by
  sorry

end pure_gala_trees_l480_480046


namespace product_of_fraction_l480_480810

theorem product_of_fraction (x : ℚ) (h : x = 17 / 999) : 17 * 999 = 16983 := by sorry

end product_of_fraction_l480_480810


namespace total_full_parking_spots_correct_l480_480494

-- Define the number of parking spots on each level
def total_parking_spots (level : ℕ) : ℕ :=
  100 + (level - 1) * 50

-- Define the number of open spots on each level
def open_parking_spots (level : ℕ) : ℕ :=
  if level = 1 then 58
  else if level <= 4 then 58 - 3 * (level - 1)
  else 49 + 10 * (level - 4)

-- Define the number of full parking spots on each level
def full_parking_spots (level : ℕ) : ℕ :=
  total_parking_spots level - open_parking_spots level

-- Sum up the full parking spots on all 7 levels to get the total full spots
def total_full_parking_spots : ℕ :=
  List.sum (List.map full_parking_spots [1, 2, 3, 4, 5, 6, 7])

-- Theorem to prove the total number of full parking spots
theorem total_full_parking_spots_correct : total_full_parking_spots = 1329 :=
by
  sorry

end total_full_parking_spots_correct_l480_480494


namespace angle_MXC_right_l480_480708

variables {A B C D M N X : Type}
variables [metric_space X]

-- Define the square ABCD
def square (A B C D : X) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧ 
  angle A B C = π/2 ∧ angle B C D = π/2 ∧ angle C D A = π/2 ∧ angle D A B = π/2

-- Define points and distances
variables (AM BN : ℝ)
variables (AM_eq_BN : AM = BN)
variables (dist_AM_eq_BN : ∀ M N, dist A M = AM ∧ dist B N = BN)

-- Define the foot of the perpendicular 
variables (foot_perpendicular : ∀ {D AN X}, angle D X AN = π/2)

theorem angle_MXC_right {A B C D M N X : X} 
  (sq : square A B C D)
  (hM : ∀ (AM : ℝ), ∃ M : X, dist A M = AM)
  (hN : ∀ (BN : ℝ), ∃ N : X, dist B N = BN)
  (h_perpendicular : foot_perpendicular D (line_through A N) X) :
  angle M X C = π/2 :=
begin
  sorry
end

end angle_MXC_right_l480_480708


namespace shortest_chord_m_eq_2_l480_480171

def circle := { p : ℝ × ℝ // (p.1 - 2)^2 + (p.2 - 3)^2 = 10 }
def line (m : ℝ) := { p : ℝ × ℝ // m * p.1 + 2 * p.2 = 4 * m + 10 }
def passes_through_point (l : ℝ → set (ℝ × ℝ)) (p : ℝ × ℝ) := ∃ m, l m p

theorem shortest_chord_m_eq_2 :
    ∃ m ∈ ℝ, 
    (∀ p ∈ line m, (1 + (m/2)^2) * ((p.1 - 2)^2 + (p.2 - 3)^2) = 0) ∧ 
    (∃ p ∈ line m, ∀ p' ∈ circle, (p'.1 - p.1) * (2/p'.2 - p.2) = -1) →
    m = 2 :=
begin
  sorry
end

end shortest_chord_m_eq_2_l480_480171


namespace findPrincipalSum_l480_480246

-- Define the given conditions
def rateOfInterest : ℝ := 10 / 100
def timeInYears : ℕ := 2
def differenceInInterest : ℝ := 61
def simpleInterest (P : ℝ) : ℝ := P * rateOfInterest * timeInYears
def compoundInterest (P : ℝ) : ℝ := P * (1 + rateOfInterest)^timeInYears - P

-- State the theorem
theorem findPrincipalSum (P : ℝ) :
  compoundInterest P - simpleInterest P = differenceInInterest → P = 6100 :=
by
  sorry

end findPrincipalSum_l480_480246


namespace completing_the_square_result_l480_480033

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l480_480033


namespace last_two_nonzero_digits_80_fact_l480_480756

theorem last_two_nonzero_digits_80_fact : 
  ∃ n : ℕ, (n = 80.factorial % 100) ∧ n ≠ 0 ∧ last_two_nonzero_digits n = 76 :=
by
  sorry

end last_two_nonzero_digits_80_fact_l480_480756


namespace sin_ratio_area_l480_480652

-- Given conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B)

-- Proof goal for part (I)
theorem sin_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) :
  (Real.sin A / Real.sin C) = 1 / 2 :=
sorry

-- Given additional conditions for part (II)
variables
  (cosB : Real.cos B = 1 / 4)
  (b_eq : b = 2)

-- Proof goal for part (II)
theorem area (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B)
  (cosB : Real.cos B = 1 / 4)
  (b_eq : b = 2) :
  let a := 1
  let c := 2 in
  let sinB := Real.sqrt (1 - (1 / 4)^2) in
  let S := (1 / 2) * a * c * sinB in
  S = Real.sqrt 15 / 4 :=
sorry

end sin_ratio_area_l480_480652


namespace rhombus_area_l480_480741

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 15) : 
  let area := (d1 * d2) / 2 in area = 75 := by
    sorry

end rhombus_area_l480_480741


namespace combined_mpg_19_l480_480718

theorem combined_mpg_19 (m: ℕ) (h: m = 100) :
  let ray_car_mpg := 50
  let tom_car_mpg := 25
  let jerry_car_mpg := 10
  let ray_gas_used := m / ray_car_mpg
  let tom_gas_used := m / tom_car_mpg
  let jerry_gas_used := m / jerry_car_mpg
  let total_gas_used := ray_gas_used + tom_gas_used + jerry_gas_used
  let total_miles := 3 * m
  let combined_mpg := total_miles * 25 / (4 * m)
  combined_mpg = 19 := 
by {
  sorry
}

end combined_mpg_19_l480_480718


namespace range_of_m_l480_480599

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : π < α ∧ α < 2 * π ∨ 3 * π < α ∧ α < 4 * π) 
(h2 : Real.sin α = (2 * m - 3) / (4 - m)) : 
  -1 < m ∧ m < (3 : ℝ) / 2 :=
  sorry

end range_of_m_l480_480599


namespace arrangements_count_correct_l480_480089

def arrangements_total : Nat :=
  let total_with_A_first := (Nat.factorial 5) -- A^5_5 = 120
  let total_with_B_first := (Nat.factorial 4) * 1 -- A^1_4 * A^4_4 = 96
  total_with_A_first + total_with_B_first

theorem arrangements_count_correct : arrangements_total = 216 := 
by
  -- Proof is required here
  sorry

end arrangements_count_correct_l480_480089


namespace fraction_of_income_from_tips_l480_480261

theorem fraction_of_income_from_tips (S : ℝ) :
  let tips_week1 := (11/4) * S,
      salary_week2 := (5/4) * S,
      tips_week2 := (7/3) * salary_week2,
      total_salary := S + salary_week2,
      total_tips := tips_week1 + tips_week2,
      frac_income_tips := total_tips / (total_salary + total_tips)
  in frac_income_tips = 68 / 95 :=
by
  sorry

end fraction_of_income_from_tips_l480_480261


namespace sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l480_480338

theorem sqrt_12_eq_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := sorry

theorem sqrt_1_div_2_eq_sqrt_2_div_2 : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 := sorry

end sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l480_480338


namespace smallest_integral_k_no_real_roots_l480_480976

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ (a b c : ℝ), a = 3 * (k : ℝ) - 2 → b = -15 → c = 8 → b^2 - 4 * a * c < 0) ∧ k = 3 :=
begin
  sorry
end

end smallest_integral_k_no_real_roots_l480_480976


namespace least_value_of_q_minus_p_l480_480271

theorem least_value_of_q_minus_p (y : ℝ) (h1: (y + 5) + (4 * y) > y + 10)
  (h2: (y + 5) + (y + 10) > 4 * y) (h3: (4 * y) + (y + 10) > y + 5) :
  ∃ (p q : ℝ), p < y ∧ y < q ∧ ((q = 5) ∧ (p = 5 / 3) ∧ (q - p = 5 / 3)) :=
begin
  sorry
end

end least_value_of_q_minus_p_l480_480271


namespace _l480_480177

-- 'noncomputable' might be necessary if calculations or constructions involve real numbers
noncomputable def find_circumcenter (A B C : Point) (h : acute_triangle A B C) : Point :=
  let O := circumcenter A B C in
  O

noncomputable theorem satisfies_inequalities_iff_circumcenter
  (A B C : Point) (h : acute_triangle A B C) (P : Point) :
  (1 ≤ (angle A P B / angle A C B) ∧ (angle A P B / angle A C B) ≤ 2 ∧
   1 ≤ (angle B P C / angle B A C) ∧ (angle B P C / angle B A C) ≤ 2 ∧
   1 ≤ (angle C P A / angle C B A) ∧ (angle C P A / angle C B A) ≤ 2) ↔
  P = circumcenter A B C :=
sorry

end _l480_480177


namespace lesser_fraction_exists_l480_480375

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l480_480375


namespace original_price_per_kg_l480_480420

theorem original_price_per_kg (P : ℝ) (S : ℝ) (reduced_price : ℝ := 0.8 * P) (total_cost : ℝ := 400) (extra_salt : ℝ := 10) :
  S * P = total_cost ∧ (S + extra_salt) * reduced_price = total_cost → P = 10 :=
by
  intros
  sorry

end original_price_per_kg_l480_480420


namespace angle_bisectors_right_angle_l480_480524

theorem angle_bisectors_right_angle (A B C K : Type) [IsTriangle A B C] :
  ¬ IsAngleBisector A B K →
  ¬ IsAngleBisector C B K →
  ∀ θ : ℝ, θ = 90 :=
begin
  sorry
end

end angle_bisectors_right_angle_l480_480524


namespace average_speed_l480_480863

theorem average_speed:
  ( ∀ (distance1 distance2 distance3 total_distance : ℝ)
      (time1 time2 time3 total_time : ℝ) 
      (speed1 speed2 : ℝ),
    -- Conditions: Speed and time for each segment
    speed1 = 40 → time1 = 1 →
    speed2 = 60 → time2 = 0.5 → time3 = 2 →
    -- Distance calculations
    distance1 = speed1 * time1 → 
    distance2 = speed2 * time2 → 
    distance3 = speed2 * time3 → 
    total_distance = distance1 + distance2 + distance3 →
    -- Total time calculation
    total_time = time1 + time2 + time3 →
    -- Average speed
    average_speed = total_distance / total_time →
    average_speed ≈ 54.29)
:= sorry

end average_speed_l480_480863


namespace shortest_path_bound_l480_480256

-- Define the context of the problem
def cities_and_roads (C : Type*) [fintype C] (d : C → C → ℝ) : Prop :=
  ∀ c1 c2 : C, c1 ≠ c2 → d c1 c2 > 0

def shortest_path_length {C : Type*} [fintype C] (d : C → C → ℝ) (c : C) : ℝ :=
  -- Placeholder for the actual shortest path length definition
  sorry

-- The main theorem statement
theorem shortest_path_bound {C : Type*} [fintype C] (d : C → C → ℝ)
  (h : cities_and_roads C d) (c1 c2 : C) :
  let ℓ1 := shortest_path_length d c1
  let ℓ2 := shortest_path_length d c2
  in (1 / 1.5) * ℓ1 ≤ ℓ2 ∧ ℓ2 ≤ 1.5 * ℓ1 :=
sorry

end shortest_path_bound_l480_480256


namespace amount_lent_to_b_is_5000_l480_480488

-- Define the conditions
def lent_money_b (P_B : ℝ) (r : ℝ) (t : ℝ) : ℝ := P_B * r * t
def lent_money_c (P_C : ℝ) (r : ℝ) (t : ℝ) : ℝ := P_C * r * t

def total_interest (I_B I_C : ℝ) : ℝ := I_B + I_C

-- Given assumptions and values
def P_C : ℝ := 3000
def r : ℝ := 9 / 100
def t_B : ℝ := 2
def t_C : ℝ := 4
def I_total : ℝ := 1980

-- Define the lending amounts
def I_B (P_B : ℝ) : ℝ := lent_money_b P_B r t_B
def I_C : ℝ := lent_money_c P_C r t_C

-- Prove that the amount lent to B is 5000
theorem amount_lent_to_b_is_5000 (P_B : ℝ) (h : total_interest (I_B P_B) I_C = I_total) : P_B = 5000 :=
by
  sorry

end amount_lent_to_b_is_5000_l480_480488


namespace transform_polynomial_l480_480638

theorem transform_polynomial (x y : ℝ) 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 - x^3 - 2 * x^2 - x + 1 = 0) : x^2 * (y^2 - y - 4) = 0 :=
sorry

end transform_polynomial_l480_480638


namespace inequality_proof_l480_480693

theorem inequality_proof (n : ℕ) (x : ℕ → ℝ) (h : ∀ i, 1 ≤ x i) :
  (∑ i in finset.range n, 1 / (x i + 1)) ≥ n / (1 + (∏ i in finset.range n, x i)^(1 / n)) :=
sorry

end inequality_proof_l480_480693


namespace concurrency_of_tangents_l480_480391

noncomputable theory

-- Circle and geometry definitions provided for Lean
structure Circle (α : Type*) [TopologicalSpace α] :=
(center : α)
(radius : ℝ)

structure Triangle (α : Type*) [TopologicalSpace α] :=
(pointA pointB pointC : α)

def TangentCircle (Γ : Circle ℝ) (A B : α) : Circle ℝ :=
  sorry -- Implicitly define a circle tangent to Γ and sides AB, sides AC

def PointOfTangency (Γ : Circle ℝ) (Γ_a : Circle ℝ) : α :=
  sorry -- Implicitly define the point of tangency between Γ and Γ_a

axiom Monge (Γ Γ' : Circle ℝ) (X Y Z : α) (P_X P_Y P_Z : α) :
  -- Monge's Theorem axiom stating the requirement for concurrencies
  collinear [X, P_X, P_Y, P_Z]

theorem concurrency_of_tangents 
(ABC : Triangle ℝ)
(Γ : Circle ℝ)
(Γ_a : Circle ℝ)
(Γ_b : Circle ℝ)
(Γ_c : Circle ℝ)
(A' B' C' : ℝ) 
(tangent_A : Γ_a = TangentCircle Γ ABC.pointA ABC.pointB)
(tangent_B : Γ_b = TangentCircle Γ ABC.pointB ABC.pointC)
(tangent_C : Γ_c = TangentCircle Γ ABC.pointC ABC.pointA)
(A'_correct : A' = PointOfTangency Γ Γ_a)
(B'_correct : B' = PointOfTangency Γ Γ_b)
(C'_correct : C' = PointOfTangency Γ Γ_c) :
  concurrent ABC.pointA ABC.pointB ABC.pointC A' B' C' := 
by
  apply Monge
  rw [tangent_A, tangent_B, tangent_C]
  rw [A'_correct, B'_correct, C'_correct]
  sorry

end concurrency_of_tangents_l480_480391


namespace jerome_bought_last_month_l480_480277

-- Definitions representing the conditions in the problem
def total_toy_cars_now := 40
def original_toy_cars := 25
def bought_this_month (bought_last_month : ℕ) := 2 * bought_last_month

-- The main statement to prove
theorem jerome_bought_last_month : ∃ x : ℕ, original_toy_cars + x + bought_this_month x = total_toy_cars_now ∧ x = 5 :=
by
  sorry

end jerome_bought_last_month_l480_480277


namespace quadratic_function_positive_difference_l480_480751

/-- Given a quadratic function y = ax^2 + bx + c, where the coefficient a
indicates a downward-opening parabola (a < 0) and the y-intercept is positive (c > 0),
prove that the expression (c - a) is always positive. -/
theorem quadratic_function_positive_difference (a b c : ℝ) (h1 : a < 0) (h2 : c > 0) : c - a > 0 := 
by
  sorry

end quadratic_function_positive_difference_l480_480751


namespace common_chord_length_l480_480804

theorem common_chord_length (r : ℝ) (h_radius : r = 12) (h_overlap : true) :
  common_chord_length r = 12 * Real.sqrt 3 := by
sorry

end common_chord_length_l480_480804


namespace range_of_f_on_interval_l480_480565

noncomputable def f (x : ℝ) := -x^2 + 4*x - 6

theorem range_of_f_on_interval : set.Icc (-11 : ℝ) (-2) = set.range (λ (x : ↥(set.Icc 0 5)), f x) :=
sorry

end range_of_f_on_interval_l480_480565


namespace two_pow_ge_two_mul_l480_480877

theorem two_pow_ge_two_mul (n : ℕ) : 2^n ≥ 2 * n :=
sorry

end two_pow_ge_two_mul_l480_480877


namespace brendan_remaining_money_l480_480959

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l480_480959


namespace completing_the_square_equation_l480_480038

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l480_480038


namespace valid_pairings_count_l480_480526

-- Define the colors for bowl and glass
inductive BowlColor | red | blue | yellow | green
inductive GlassColor | red | blue | green

-- Define the conditions
def bowls : Finset BowlColor := {BowlColor.red, BowlColor.blue, BowlColor.yellow, BowlColor.green}
def glasses : Finset GlassColor := {GlassColor.red, GlassColor.blue, GlassColor.green}
def glassCount (c : GlassColor) : Nat := 2

-- Valid pairings according to the conditions
def validPairings (b : BowlColor) : Finset GlassColor :=
  match b with
  | BowlColor.red     => {GlassColor.red, GlassColor.green}
  | BowlColor.blue    => {GlassColor.blue, GlassColor.red}
  | BowlColor.yellow  => {GlassColor.green, GlassColor.blue}
  | BowlColor.green   => {GlassColor.green, GlassColor.red}

-- Calculate total valid pairings
def countValidPairings : Nat :=
  bowls.toList.sum (λ b => (validPairings b).toList.sum (λ g => glassCount g))

theorem valid_pairings_count : countValidPairings = 13 := by
  sorry

end valid_pairings_count_l480_480526


namespace largest_number_of_square_plots_l480_480899

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l480_480899


namespace project_presentation_periods_l480_480483

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l480_480483


namespace average_is_seven_l480_480244

def average (numbers : List ℕ) : ℕ :=
  numbers.sum / numbers.length

theorem average_is_seven (x : ℕ) (h : x = 12) :
  average [1, 2, 4, 5, 6, 9, 9, 10, 12, x] = 7 :=
by
  rw [h]
  simp [average]
  sorry

end average_is_seven_l480_480244


namespace ellipse_standard_equation_l480_480603

theorem ellipse_standard_equation (a c b : ℝ) (ha : a = 2) (hc : c = √2) (hb : b = √2)
    (isEllipse : a^2 = b^2 + c^2) :
    ∀ x y : ℝ, (x^2 / 4 + y^2 / 2 = 1) :=
begin
  sorry
end

end ellipse_standard_equation_l480_480603


namespace sum_of_first_fifteen_multiples_of_17_l480_480853

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480853


namespace problem_friend_defn_1_problem_friend_real_root_problem_friend_monotonic_sequence_l480_480974

section Problem

-- 1. Define the function g.
def g (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x - 1 / 2

-- 2. Conditions
variables {m : ℝ} (h0 : 0 < m) (h1 : m < 1 / 2)

-- 3. Define the transformed function f.
def f (x : ℝ) : ℝ := m * sin x + 1

-- 4. Prove the function f is correctly derived.
theorem problem_friend_defn_1 : f x = m * sin x + 1 := sorry

-- 5. Prove f(x) = x has exactly one real root.
theorem problem_friend_real_root : ∃! x : ℝ, f x = x := sorry

-- 6. Define the sequence a_n.
def a_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := f (a_seq n)

-- 7. Prove the monotonicity of the sequence.
theorem problem_friend_monotonic_sequence : ∀ n : ℕ, a_seq (n + 1) > a_seq n := sorry

end Problem

end problem_friend_defn_1_problem_friend_real_root_problem_friend_monotonic_sequence_l480_480974


namespace expression_identity_l480_480409

theorem expression_identity (a : ℤ) (h : a = 102) : 
  a^4 - 4 * a^3 + 6 * a^2 - 4 * a + 1 = 104060401 :=
by {
  rw h,
  calc 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 101^4 : by sorry
  ... = 104060401 : by sorry
}

end expression_identity_l480_480409


namespace find_two_numbers_l480_480459

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480459


namespace lesser_fraction_l480_480381

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l480_480381


namespace conversion_proofs_l480_480060

-- Define the necessary constants for unit conversion
def cm_to_dm2 (cm2: ℚ) : ℚ := cm2 / 100
def m3_to_dm3 (m3: ℚ) : ℚ := m3 * 1000
def dm3_to_liters (dm3: ℚ) : ℚ := dm3
def liters_to_ml (liters: ℚ) : ℚ := liters * 1000

theorem conversion_proofs :
  (cm_to_dm2 628 = 6.28) ∧
  (m3_to_dm3 4.5 = 4500) ∧
  (dm3_to_liters 3.6 = 3.6) ∧
  (liters_to_ml 0.6 = 600) :=
by
  sorry

end conversion_proofs_l480_480060


namespace bathroom_square_footage_l480_480473

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l480_480473


namespace find_a_l480_480215

theorem find_a (t a x : ℝ) (h : tx^2 - 6x + t^2 < 0)
  (h_solution_set : ∀ x, (tx^2 - 6x + t^2 < 0) ↔ x ∈ Iio a ∪ Ioi 1) :
  a = -3 :=
sorry

end find_a_l480_480215


namespace cylinder_volume_l480_480720

-- Define the initial conditions of the rectangle
def rectangle_length := 6
def rectangle_width := 4

-- Define the two cases for the cylinder volume
def volume_case_1 : ℝ := 24 / Real.pi
def volume_case_2 : ℝ := 36 / Real.pi

-- The main theorem to prove
theorem cylinder_volume :
  ∃ V, (V = volume_case_1 ∨ V = volume_case_2) := by
  sorry

end cylinder_volume_l480_480720


namespace numbers_pairs_sum_prod_l480_480426

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480426


namespace distribution_of_X_l480_480486

theorem distribution_of_X (X : Type) [Discrete X] (x1 x2 : ℝ) (h1 : x1 < x2)
                          (p1 p2 : ℝ) (h2 : p1 = 0.5) (h3 : p2 = 0.5)
                          (EX : ℝ) (h4 : EX = 3.5)
                          (VX : ℝ) (h5 : VX = 0.25) :
  ∀ x, (x = 3 → P X x = 0.5) ∧ (x = 4 → P X x = 0.5) :=
begin
  sorry
end

end distribution_of_X_l480_480486


namespace need_to_sell_more_rolls_l480_480321

variable (goal sold_grandmother sold_uncle_1 sold_uncle_additional sold_neighbor_1 returned_neighbor sold_mothers_friend sold_cousin_1 sold_cousin_additional : ℕ)

theorem need_to_sell_more_rolls
  (h_goal : goal = 100)
  (h_sold_grandmother : sold_grandmother = 5)
  (h_sold_uncle_1 : sold_uncle_1 = 12)
  (h_sold_uncle_additional : sold_uncle_additional = 10)
  (h_sold_neighbor_1 : sold_neighbor_1 = 8)
  (h_returned_neighbor : returned_neighbor = 4)
  (h_sold_mothers_friend : sold_mothers_friend = 25)
  (h_sold_cousin_1 : sold_cousin_1 = 3)
  (h_sold_cousin_additional : sold_cousin_additional = 5) :
  goal - (sold_grandmother + (sold_uncle_1 + sold_uncle_additional) + (sold_neighbor_1 - returned_neighbor) + sold_mothers_friend + (sold_cousin_1 + sold_cousin_additional)) = 36 := by
  sorry

end need_to_sell_more_rolls_l480_480321


namespace intersect_circle_at_point_eq_distance_l480_480889

theorem intersect_circle_at_point_eq_distance {ABC : Type} [triangle ABC] [incenter I : point]
    (ω : circle) (A B C D E M : point) 
    (tangent_to_AB : tangent_to ω A B)
    (intersects_BC : intersects_at ω B C D)
    (intersects_ext_BC : intersects_at_extension ω B C E)
    (I_on_circumcircle : on_circle I ω)
    (M_on_circumcircle : on_circle M ω)
    : intersects_at I C M → distance M D = distance M E :=
by
  sorry

end intersect_circle_at_point_eq_distance_l480_480889


namespace find_f5_5_l480_480594

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x : ℝ, f(x) * f(x + 2) = -1

axiom condition_eq : ∀ x : ℝ, (1 < x ∧ x < 2) → f(x) = x^3 + Real.sin (Real.pi / 9 * x)

theorem find_f5_5 : f 5.5 = 31 / 8 :=
by
  sorry

end find_f5_5_l480_480594


namespace suitable_survey_method_l480_480044

-- Define the conditions as propositions.
def option_A : Prop := "Sampling survey is suitable for checking typos in manuscripts."
def option_B : Prop := "Comprehensive survey is suitable for understanding the viewership of the Spring Festival Gala."
def option_C : Prop := "Environmental protection department conducts a comprehensive survey on water pollution in a certain section of the Yellow River."
def option_D : Prop := "Comprehensive survey is suitable for investigating the weight situation of students in Grade 8 (1) of a certain school."

-- Define the correct option
def correct_option : Prop := option_D

-- The theorem statement
theorem suitable_survey_method :
  (option_A -> False) ∧ 
  (option_B -> False) ∧ 
  (option_C -> False) ∧ 
  (correct_option) :=
by
  sorry

end suitable_survey_method_l480_480044


namespace evaluate_expression_l480_480983

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end evaluate_expression_l480_480983


namespace eval_expr_l480_480550

theorem eval_expr : 64^(-1/3) + 81^(-1/2) = 13/36 := 
by
  have h1 : 64 = 2^6 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  calc
    64^(-1/3) + 81^(-1/2)
        = 2^(-2) + 3^(-2) : by { rw [h1, h2], norm_num, }
    ... = 1/4 + 1/9 : by norm_num
    ... = 13/36 : by norm_num

end eval_expr_l480_480550


namespace rectangle_area_l480_480360

theorem rectangle_area (b l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 112) : l * b = 588 := by
  sorry

end rectangle_area_l480_480360


namespace log_bounds_sum_l480_480773

theorem log_bounds_sum : (∀ a b : ℕ, a = 18 ∧ b = 19 → 18 < Real.log 537800 / Real.log 2 ∧ Real.log 537800 / Real.log 2 < 19 → a + b = 37) := 
sorry

end log_bounds_sum_l480_480773


namespace bethany_coin_problem_l480_480518

theorem bethany_coin_problem (m n : ℕ) :
  let c := 11 + m + n in
  (1100 + 20 * m + 50 * n) / (11 + m + n) = 52 →
  c ≠ 40 :=
by {
  assume h,
  let k := 528 - 32 * m - 2 * n,
  have h1 : k = 0 := by linarith [h],
  have h2 : n = 264 - 16 * m := by linarith [h1],
  let c := 275 - 15 * m,
  have c_value : c = 275 - 15 * m := by linarith,
  show c ≠ 40,
  sorry
}

end bethany_coin_problem_l480_480518


namespace AIME_inequality_l480_480691

theorem AIME_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 1 ≤ x i) :
  (∑ i, (1/(x i + 1)))  ≥ n / (1 + (∏ i, x i) ^ (1 / n : ℝ)) :=
sorry

end AIME_inequality_l480_480691


namespace distance_JK_distance_DH_distance_BG_l480_480932

-- Define the stations as types:
inductive Station
| A | B | C | D | E | F | G | H | I | J | K
open Station

-- Define the distances between stations as a function:
def distance (s1 s2 : Station) : ℝ

-- Define the problem conditions:
axiom segment_distance_max (s1 s2 : Station) : distance s1 s2 ≤ 12
axiom segment_three_min (s1 s2 s3 s4 : Station) : distance s1 s2 + distance s2 s3 + distance s3 s4 ≥ 17
axiom total_distance_AK : distance A K = 56

-- Questions to be proved:
theorem distance_JK : distance J K = 5 := sorry

theorem distance_DH : distance D H = 22 := sorry

theorem distance_BG : distance B G = 29 := sorry

end distance_JK_distance_DH_distance_BG_l480_480932


namespace find_f_729_l480_480122

variable (f : ℕ+ → ℕ+) -- Define the function f on the positive integers.

-- Conditions of the problem.
axiom h1 : ∀ n : ℕ+, f (f n) = 3 * n
axiom h2 : ∀ n : ℕ+, f (3 * n + 1) = 3 * n + 2 

-- Proof statement.
theorem find_f_729 : f 729 = 729 :=
by
  sorry -- Placeholder for the proof.

end find_f_729_l480_480122


namespace sum_of_x_intercepts_eq_neg_8_6_l480_480328

theorem sum_of_x_intercepts_eq_neg_8_6 (c d : ℕ) (h1 : 2*c*x + 8 = 0) (h2 : 5*x + d = 0) (h3: c > 0) (h4: d > 0) :
  ∑ x in {x : ℚ | ∃ (c d : ℕ), 2*c*x + 8 = 0 ∧ 5*x + d = 0 ∧ c*d = 20}, x = -8.6 :=
by
  sorry

end sum_of_x_intercepts_eq_neg_8_6_l480_480328


namespace min_square_side_length_l480_480702

theorem min_square_side_length (s : ℝ) (h : s^2 ≥ 625) : s ≥ 25 :=
sorry

end min_square_side_length_l480_480702


namespace find_x_for_slope_l480_480184

theorem find_x_for_slope (x : ℝ) (h : (2 - 5) / (x - (-3)) = -1 / 4) : x = 9 :=
by 
  -- Proof skipped
  sorry

end find_x_for_slope_l480_480184


namespace part1_part2_l480_480173

def hyperbola (a b : ℝ) : Prop := ∀ x y : ℝ, a > 0 → b > 0 → (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_slope (a b : ℝ) : Prop := b / a = 1 / Real.sqrt 3

def min_distance_focus_to_point (a b : ℝ) : Prop := 
  let c := Real.sqrt (a^2 + b^2) in
  c - a = 2 - Real.sqrt 3

theorem part1 (a b : ℝ) (h1 : asymptote_slope a b) (h2 : min_distance_focus_to_point a b) :
  hyperbola (Real.sqrt 3) 1 :=
sorry

def line (x : ℝ) : ℝ := x - 2

def intersections (a b : ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | hyperbola a b p.1 p.2 ∧ p.2 = line p.1 }

def perpendicular_intersection (a b : ℝ) (A B : ℝ × ℝ) : set (ℝ × ℝ) :=
  -- Assuming perpendicular_intersection definition here
  sorry

def area_ABCD (a b : ℝ) : ℝ :=
  let A := ((3 + Real.sqrt 6 / 2), line (3 + Real.sqrt 6 / 2)) in
  let B := ((3 - Real.sqrt 6 / 2), line (3 - Real.sqrt 6 / 2)) in
  let D := perpendicular_intersection a b A B in
  let C := perpendicular_intersection a b B A in
  12 * Real.sqrt 6 

theorem part2 (a b : ℝ) (h1 : asymptote_slope a b) (h2 : min_distance_focus_to_point a b):
  area_ABCD (Real.sqrt 3) 1 = 12 * Real.sqrt 6 :=
sorry

end part1_part2_l480_480173


namespace sum_of_integers_l480_480799

theorem sum_of_integers (a b c : ℕ) :
  a > 1 → b > 1 → c > 1 →
  a * b * c = 1728 →
  gcd a b = 1 → gcd b c = 1 → gcd a c = 1 →
  a + b + c = 43 :=
by
  intro ha
  intro hb
  intro hc
  intro hproduct
  intro hgcd_ab
  intro hgcd_bc
  intro hgcd_ac
  sorry

end sum_of_integers_l480_480799


namespace expected_value_segments_l480_480944

theorem expected_value_segments :
  ∃ (X : ℕ), 
  (∀ (n : ℕ) (pts : Finset (ℕ × ℕ)), 
    (n = 100 ∧ 
     pts.card = 4026 ∧
     (∀ (x y z : ℕ) (hx : x < y) (hy : y < z), 
     ((x, y) ∈ pts ∧ (y, z) ∈ pts) → (x, z) ∉ pts) ∧
     (∀ pt, pt ∈ pts → pt = ((p1, p2) : Fin ℕ × Fin ℕ) ∧ p1 ≠ p2 ∧ 1 ≤ p1.1 ∧ p1.1 ≤ 100 ∧ 1 ≤ p2.1 ∧ p2.1 ≤ 100 ∧ (abs (p1.1 - p2.1) >= 50))
     → (X = 1037)))
:= sorry

end expected_value_segments_l480_480944


namespace lower_water_level_by_inches_l480_480386

theorem lower_water_level_by_inches
  (length width : ℝ) (gallons_removed : ℝ) (gallons_to_cubic_feet : ℝ) (feet_to_inches : ℝ) : 
  length = 20 → 
  width = 25 → 
  gallons_removed = 1875 → 
  gallons_to_cubic_feet = 7.48052 → 
  feet_to_inches = 12 → 
  (gallons_removed / gallons_to_cubic_feet) / (length * width) * feet_to_inches = 6.012 := 
by 
  sorry

end lower_water_level_by_inches_l480_480386


namespace find_k_l480_480685

noncomputable section

open Polynomial

-- Define the conditions
variables (h k : Polynomial ℚ)
variables (C : k.eval (-1) = 15) (H : h.comp k = h * k) (nonzero_h : h ≠ 0)

-- The goal is to prove k(x) = x^2 + 21x - 35
theorem find_k : k = X^2 + 21 * X - 35 :=
  by sorry

end find_k_l480_480685


namespace exists_arithmetic_progression_l480_480713

theorem exists_arithmetic_progression (M : ℝ) :
  ∃ (a_0 : ℕ) (m : ℕ), ∀ (n : ℕ), 
    let a_n := a_0 + n * (10^m + 1) in
    (a_n > 0) ∧
    (¬ (10^m + 1) % 10 = 0) ∧
    (digit_sum a_n > M) :=
sorry

end exists_arithmetic_progression_l480_480713


namespace vertex_F_total_path_length_l480_480118

structure Triangle :=
  (side_length : ℝ)

structure Square :=
  (side_length : ℝ)

variables (DEF : Triangle)
variables (JKLM : Square)

def is_equilateral (t : Triangle) : Prop :=
  t.side_length = 3

def in_square (t : Triangle) (s : Square) : Prop :=
  s.side_length = 6 ∧ t.side_length = 3

def total_path_length_traversed (t : Triangle) (s : Square) : ℝ :=
  if in_square t s ∧ is_equilateral t then 24 * Real.pi else 0

theorem vertex_F_total_path_length :
  total_path_length_traversed DEF JKLM = 24 * Real.pi :=
by
  unfold total_path_length_traversed
  rw [if_pos]
  . refl
  split
  . exact ⟨rfl, rfl⟩
  . exact rfl

-- noncomputable def DEF : Triangle := {side_length := 3}
-- noncomputable def JKLM : Square := {side_length := 6}

end vertex_F_total_path_length_l480_480118


namespace shaded_region_area_l480_480775

open Real

noncomputable def area_of_shaded_region (side : ℝ) (radius : ℝ) : ℝ :=
  let area_square := side ^ 2
  let area_sector := π * radius ^ 2 / 4
  let area_triangle := (1 / 2) * (side / 2) * sqrt ((side / 2) ^ 2 - radius ^ 2)
  area_square - 8 * area_triangle - 4 * area_sector

theorem shaded_region_area (h_side : ℝ) (h_radius : ℝ)
  (h1 : h_side = 8) (h2 : h_radius = 3) :
  area_of_shaded_region h_side h_radius = 64 - 16 * sqrt 7 - 3 * π :=
by
  rw [h1, h2]
  sorry

end shaded_region_area_l480_480775


namespace min_value_bSn_l480_480167

noncomputable def a (n : ℕ) := ∫ x in 0..n, (2*x + 1)

noncomputable def S (n : ℕ) := (∑ k in Finset.range n, (1 / (a (k + 1))))

def b (n : ℕ) := n - 8

noncomputable def bSn (n : ℕ) := b n * S n

theorem min_value_bSn : ∃ (n : ℕ), bSn n = -4 :=
by {
  sorry
}

end min_value_bSn_l480_480167


namespace sum_of_first_fifteen_multiples_of_17_l480_480845

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480845


namespace system_no_solution_iff_n_eq_neg_one_l480_480241

def no_solution_system (n : ℝ) : Prop :=
  ¬∃ x y z : ℝ, (n * x + y = 1) ∧ (n * y + z = 1) ∧ (x + n * z = 1)

theorem system_no_solution_iff_n_eq_neg_one (n : ℝ) : no_solution_system n ↔ n = -1 :=
sorry

end system_no_solution_iff_n_eq_neg_one_l480_480241


namespace compare_negatives_l480_480965

theorem compare_negatives : -2 < -3 / 2 :=
by sorry

end compare_negatives_l480_480965


namespace contribution_of_eight_families_l480_480085

/-- Definition of the given conditions --/
def classroom := 200
def two_families := 2 * 20
def ten_families := 10 * 5
def missing_amount := 30

def total_raised (x : ℝ) : ℝ := two_families + ten_families + 8 * x

/-- The main theorem to prove the contribution of each of the eight families --/
theorem contribution_of_eight_families (x : ℝ) (h : total_raised x = classroom - missing_amount) : x = 10 := by
  sorry

end contribution_of_eight_families_l480_480085


namespace purely_imaginary_z1_iff_z1_gt_z2_iff_l480_480169

-- Problem: Given z1 and z2 defined as below,
-- Prove z1 is purely imaginary iff x = -1/2 with the given conditions,
-- and Prove z1 > z2 iff x = 2 with the given conditions

noncomputable def z1 (x : ℝ) : ℂ := (2 * x + 1) + (x^2 - 3 * x + 2) * complex.I
noncomputable def z2 (x : ℝ) : ℂ := (x^2 - 2) + (x^2 + x - 6) * complex.I

theorem purely_imaginary_z1_iff (x : ℝ) :
  (z1 x).re = 0 ↔ x = -1/2 := sorry

theorem z1_gt_z2_iff (x : ℝ) :
  z1 x > z2 x ↔ x = 2 := sorry

end purely_imaginary_z1_iff_z1_gt_z2_iff_l480_480169


namespace reconstruct_triangle_ABC_l480_480350

variable {O A B C A1 B1 C1 O1 O2 O3 : Point}

-- Reflect circumcenter O across sides of triangle ABC to get O1, O2, O3
def isReflectionWithRespectToSide (P₁ P₂ side : Point) : Prop :=
  -- Placeholder definition for reflection across a side
  sorry

def isOrthocenter (P O1 O2 O3 : Point) : Prop :=
  -- Placeholder definition for checking if P is the orthocenter of triangle O1O2O3
  sorry

-- Given conditions
variable (h₁ : isReflectionWithRespectToSide O O1 (lineThrough B C))
variable (h₂ : isReflectionWithRespectToSide O O2 (lineThrough C A))
variable (h₃ : isReflectionWithRespectToSide O O3 (lineThrough A B))
variable (h₄ : isOrthocenter O O1 O2 O3)

-- To prove: We can reconstruct triangle ABC by the perpendicular bisectors of segments OO1, OO2, and OO3
theorem reconstruct_triangle_ABC :
  ∃ A B C : Point, isReflectionWithRespectToSide O O1 (lineThrough B C) ∧
  isReflectionWithRespectToSide O O2 (lineThrough C A) ∧
  isReflectionWithRespectToSide O O3 (lineThrough A B) ∧
  isOrthocenter O O1 O2 O3 :=
begin
  -- The proof is omitted
  sorry
end

end reconstruct_triangle_ABC_l480_480350


namespace divisible_by_power_of_two_l480_480747

variable {α : Type} {A : finset α}

def d (n : ℕ) (A_1 A_2 ... A_n : finset α) : ℕ :=
  (⋃ A_1 A_2 ... A_n).filter (λ x, odd (card (filter (λ i, x ∈ A_i) (finset.range n)))).card

theorem divisible_by_power_of_two (n : ℕ) (A : fin n → finset α) (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  (d n (finset.range n A) -
  finset.sum (finset.range n) (λ i, (A i).card) +
  2 * finset.sum (finset.pair_combinations (finset.range n)) (λ ij, (A ij.1 ∩ A ij.2).card) -
  ∑ j in finset.range k, (if even j then 2 ^ (j - 1) else 0) *
    finset.sum (finset.pair_combinations (finset.range j)) (λ is, finset.card (finset.bInter (finset.finset_of_option is (λ i, A i))))) %
  2^k = 0 :=
sorry

end divisible_by_power_of_two_l480_480747


namespace tetrahedron_cut_off_vertices_l480_480981

theorem tetrahedron_cut_off_vertices :
  ∀ (V E : ℕ) (cut_effect : ℕ → ℕ),
    -- Initial conditions
    V = 4 → E = 6 →
    -- Effect of each cut (cutting one vertex introduces 3 new edges)
    (∀ v, v ≤ V → cut_effect v = 3 * v) →
    -- Prove the number of edges in the new figure
    (E + cut_effect V) = 18 :=
by
  intros V E cut_effect hV hE hcut
  sorry

end tetrahedron_cut_off_vertices_l480_480981


namespace find_numbers_l480_480437

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480437


namespace f_increasing_intervals_g_range_l480_480613

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_increasing_intervals : 
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → 0 ≤ Real.cos x) ∧ (∀ x, 3 * Real.pi / 2 ≤ x → x ≤ 2 * Real.pi → 0 ≤ Real.cos x) :=
sorry

theorem g_range : 
  ∀ x, 0 ≤ x → x ≤ 2 * Real.pi → -1 / 2 ≤ g x ∧ g x ≤ 4 :=
sorry

end f_increasing_intervals_g_range_l480_480613


namespace maximum_f_l480_480888

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def f (p : ℝ) : ℝ :=
  binomial_coefficient 20 2 * p^2 * (1 - p)^18

theorem maximum_f :
  ∃ p_0 : ℝ, 0 < p_0 ∧ p_0 < 1 ∧ f p = f (0.1) := sorry

end maximum_f_l480_480888


namespace sequence_length_l480_480541

theorem sequence_length :
  ∃ n : ℕ, ∀ (a_1 : ℤ) (d : ℤ) (a_n : ℤ), a_1 = -6 → d = 4 → a_n = 50 → a_n = a_1 + (n - 1) * d ∧ n = 15 :=
by
  sorry

end sequence_length_l480_480541


namespace line_circle_separate_l480_480326

open Real

noncomputable def distance_from_point_to_line (a x₀ y₀ : ℝ) := a^2 / sqrt (x₀^2 + y₀^2)

theorem line_circle_separate (a x₀ y₀ : ℝ) (h₀ : a > 0) (h₁ : sqrt (x₀^2 + y₀^2) < a) :
  distance_from_point_to_line a x₀ y₀ > a :=
by
  unfold distance_from_point_to_line
  have h_dist : distance_from_point_to_line a x₀ y₀ = a^2 / sqrt (x₀^2 + y₀^2),
  {
    trivial
  },
  rw h_dist,
  have h_sqrt_pos : sqrt (x₀^2 + y₀^2) > 0,
  {
    exact sqrt_pos'.2 (by linarith)
  },
  have h_ineq : a^2 / sqrt (x₀^2 + y₀^2) > a,
  {
    have h_fraction : 1 / sqrt (x₀^2 + y₀^2) > 1 / a,
    {
      have h_inv_sqrt : sqrt (x₀^2 + y₀^2) < a,
      {
        exact h₁
      },
      exact inv_lt_inv_of_lt h₀ h_inv_sqrt,
    },
    exact (div_lt_div_right h₀).mpr h_fraction,
  },
  exact h_ineq,
  sorry

end line_circle_separate_l480_480326


namespace binomial_identity_example_l480_480189

theorem binomial_identity_example (
  h1 : Nat.choose 24 5 = 42504,
  h2 : Nat.choose 24 6 = 134596,
  h3 : Nat.choose 24 7 = 346104
) : Nat.choose 26 6 = 657800 :=
by
  sorry

end binomial_identity_example_l480_480189


namespace LCM_of_18_and_27_l480_480015

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l480_480015


namespace gibbs_inequality_l480_480298

noncomputable section

open BigOperators

variable {r : ℕ} (p q : Fin r → ℝ)

/-- (p_i) is a probability distribution -/
def isProbabilityDistribution (p : Fin r → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i, p i = 1)

/-- -\sum_{i=1}^{r} p_i \ln p_i \leqslant -\sum_{i=1}^{r} p_i \ln q_i for probability distributions p and q -/
theorem gibbs_inequality
  (hp : isProbabilityDistribution p)
  (hq : isProbabilityDistribution q) :
  -∑ i, p i * Real.log (p i) ≤ -∑ i, p i * Real.log (q i) := 
by
  sorry

end gibbs_inequality_l480_480298


namespace thirtieth_digit_sum_l480_480403

theorem thirtieth_digit_sum 
  (h1 : (1 / 7.0 : ℝ) = 0.142857142857.repeat 6) 
  (h2 : (1 / 3.0 : ℝ) = 0.333333333333.repeat 1) 
  (h3 : (1 / 11.0 : ℝ) = 0.090909090909.repeat 2) : 
  (sum_of_repeating_fractions 1 7 1 3 1 11).nth_digit 30 = 9 := by
  sorry

end thirtieth_digit_sum_l480_480403


namespace monotonic_increasing_interval_of_f_l480_480755

def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increasing_interval_of_f :
  { x : ℝ | ∃ c, (f x > f c → x > c) } = { x : ℝ | x > 2 } :=
by
  sorry

end monotonic_increasing_interval_of_f_l480_480755


namespace product_primes_less_than_4_pow_n_l480_480998

theorem product_primes_less_than_4_pow_n (n : ℕ) (hn : n ≥ 2) :
  let primes_le_n := { p : ℕ | p ≤ n ∧ Nat.Prime p }
  ∑ p in primes_le_n, p < 4 ^ n :=
sorry

end product_primes_less_than_4_pow_n_l480_480998


namespace Ryan_learning_hours_l480_480553

theorem Ryan_learning_hours :
  ∀ (hours_english hours_chinese : ℕ), hours_english = 6 ∧ hours_chinese = 7 → hours_chinese - hours_english = 1 :=
by
  intros hours_english hours_chinese h
  cases h with h_eng h_chi
  rw [h_eng, h_chi]
  norm_num
  exact rfl

end Ryan_learning_hours_l480_480553


namespace solve_for_x_l480_480608

theorem solve_for_x (x : ℝ) (m : ℝ) (h : m = 34): 
  ( ( x ^ ( m + 1 ) ) / ( 5 ^ ( m + 1 ) ) ) * ( ( x ^ 18 ) / ( 4 ^ 18 ) ) = 1 / ( 2 * ( 10 ) ^ 35 ) → 
  x = 1 :=
by
  intro h_eq
  have h_m : m = 34 := h
  rw h_m at h_eq
  sorry

end solve_for_x_l480_480608


namespace sum_of_first_fifteen_multiples_of_17_l480_480828

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480828


namespace sum_of_possible_b_values_l480_480287

theorem sum_of_possible_b_values {b : ℝ} :
  let f := λ x : ℝ, x^3 - 3 * x + b,
      g := λ x : ℝ, x^2 + b * x - 3 in
  (∃ x₀ : ℝ, f x₀ = 0 ∧ g x₀ = 0) → (∑ b in {2, -2, 0}, b) = 0 :=
by sorry

end sum_of_possible_b_values_l480_480287


namespace perp_tangent_line_equation_l480_480144

theorem perp_tangent_line_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, 3 * x + y + 2 = 0 ↔ y = -3 * x - 2) ∧
               (∀ x y : ℝ, (2 * x - 6 * y + 1 = 0) → (y = -(1/3) * x + 1/6)) ∧
               (∀ x : ℝ, y = x^3 + 3 * x^2 - 1 → derivative y at x = -3) ∧
               (∃ x : ℝ, 3 * x^2 + 6 * x = -3) :=
sorry

end perp_tangent_line_equation_l480_480144


namespace cosine_angle_half_l480_480626
-- Import the entire mathlib library for necessary components

-- Define vectors and conditions needed for the problem
variables {V : Type*} [inner_product_space ℝ V] (a b c : V) (h_unit_a : ∥a∥ = 1) (h_unit_b : ∥b∥ = 1)
          (h_dot_ab : ⟪a, b⟫ = 0) (h_c : c = a + (real.sqrt 3) • b)

-- The theorem statement
theorem cosine_angle_half : ⟪a, c⟫ / (∥a∥ * ∥c∥) = 1 / 2 :=
by
  sorry

end cosine_angle_half_l480_480626


namespace compare_fx_hx_general_form_sequence_a_bound_T_2n_l480_480610

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def h (x : ℝ) := x / (x + 1)
noncomputable def sequence_a (n : ℕ) := (n + 1) * 2^n
noncomputable def sum_S (n : ℕ) := 2 * (sequence_a n) - 2^(n + 1)
noncomputable def c_n (n : ℕ) := (-1) ^ (n + 1) * Real.log 2 / Real.log (sequence_a n / (n + 1))
noncomputable def T (n : ℕ) := ∑ i in Finset.range n, c_n (i + 1)

theorem compare_fx_hx (x : ℝ) (hx : 0 < x) : f x > h x := 
by sorry

theorem general_form_sequence_a (n : ℕ) : 
  sequence_a n = (n + 1) * 2^n := 
by sorry

theorem bound_T_2n (n : ℕ) (hn : 2 ≤ n) : 
  T (2 * n) < sqrt 2 / 2 := 
by sorry

end compare_fx_hx_general_form_sequence_a_bound_T_2n_l480_480610


namespace smallest_a_value_l480_480734

theorem smallest_a_value {a b c : ℝ} :
  (∃ (a b c : ℝ), (∀ x, (a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c)) ∧ a > 0 ∧ ∃ n : ℤ, a + b + c = n)
  → (∃ (a : ℝ), a = 1) :=
by
  sorry

end smallest_a_value_l480_480734


namespace find_numbers_sum_eq_S_product_eq_P_l480_480430

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480430


namespace find_missing_percentage_l480_480048

theorem find_missing_percentage (P : ℝ) : (P * 50 = 2.125) → (P * 100 = 4.25) :=
by
  sorry

end find_missing_percentage_l480_480048


namespace problem_ab_cd_eq_l480_480236

theorem problem_ab_cd_eq (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 14) :
  ab + cd = 45 := 
by
  sorry

end problem_ab_cd_eq_l480_480236


namespace arithmetic_sequence_common_difference_l480_480939

theorem arithmetic_sequence_common_difference :
  let a1 := 5
  let n := 30
  let S := 390
  let d := (2 * S - n * a1) / n / (n - 1) in
  d = 16 / 29 :=
by
  sorry

end arithmetic_sequence_common_difference_l480_480939


namespace david_recreation_l480_480281

theorem david_recreation (W : ℝ) (P : ℝ) 
  (h1 : 0.95 * W = this_week_wages) 
  (h2 : 0.5 * this_week_wages = recreation_this_week)
  (h3 : 1.1875 * (P / 100) * W = recreation_this_week) : P = 40 :=
sorry

end david_recreation_l480_480281


namespace probability_five_people_get_right_letter_is_zero_l480_480789

theorem probability_five_people_get_right_letter_is_zero (n : ℕ) (h : n = 6) :
  ∀ (dist : fin n → fin n), (∃! i, dist i = i → ∀ j, j ≠ i → dist j ≠ j) → (0:ℝ) = 1/0 := 
sorry

end probability_five_people_get_right_letter_is_zero_l480_480789


namespace mod_inverse_of_3_mod_31_l480_480149

-- Definition of modular inverse condition
def is_modular_inverse (a b n : ℕ) : Prop :=
  (a * b) % n = 1 % n

theorem mod_inverse_of_3_mod_31 : 
  ∃ a : ℕ, 0 ≤ a ∧ a < 31 ∧ is_modular_inverse 3 a 31 := 
begin
  use 21,
  split, 
  -- 0 ≤ a
  linarith,
  split,
  -- a < 31
  linarith,
  -- is_modular_inverse 3 a 31
  unfold is_modular_inverse,
  norm_num,
end

end mod_inverse_of_3_mod_31_l480_480149


namespace percent_black_design_l480_480259

noncomputable def radii : ℕ → ℝ
| 0 => 3
| (n+1) => radii n + 3

def area (r : ℝ) : ℝ := π * r^2

def is_black (n : ℕ) : Prop :=
  n = 2 ∨ n = 3

def sum_black_areas (n : ℕ) : ℝ :=
  ∑ i in {i | i < n ∧ is_black i}, area (radii (i+1)) - area (radii i)

theorem percent_black_design :
  (sum_black_areas 4 / area (radii 4)) * 100 = 50 := sorry

end percent_black_design_l480_480259


namespace area_triangle_AMH_volume_pyramid_AMHE_height_relative_to_base_AMH_l480_480465

-- Given conditions
def edge_length : ℝ := 6
def M_midpoint_of_EF (M E F : ℝ × ℝ × ℝ) : Prop := M = ((E.1 + F.1) / 2, (E.2 + F.2) / 2, (E.3 + F.3) / 2)

-- Point definitions based on given cube dimensions
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def E : ℝ × ℝ × ℝ := (0, 6, 0)
def F : ℝ × ℝ × ℝ := (6, 6, 0)
def H : ℝ × ℝ × ℝ := (0, 6, 6)
def M : ℝ × ℝ × ℝ := (3, 6, 0)

-- a) Area of triangle AMH
theorem area_triangle_AMH : 
  (3 * (6 * Real.sqrt 6)) / 2 = 9 * Real.sqrt 6 := 
sorry

-- b) Volume of pyramid AMHE (using base area)
theorem volume_pyramid_AMHE :
  (1 / 3) * 9 * 6 = 18 :=
sorry

-- c) Height relative to base AMH
theorem height_relative_to_base_AMH :
  6 / Real.sqrt 6 = Real.sqrt 6 :=
sorry

end area_triangle_AMH_volume_pyramid_AMHE_height_relative_to_base_AMH_l480_480465


namespace coordinate_system_and_parametric_equations_l480_480982

/-- Given the parametric equation of curve C1 is 
  x = 2 * cos φ and y = 3 * sin φ (where φ is the parameter)
  and a coordinate system with the origin as the pole and the positive half-axis of x as the polar axis.
  The polar equation of curve C2 is ρ = 2.
  The vertices of square ABCD are all on C2, arranged counterclockwise,
  with the polar coordinates of point A being (2, π/3).
  Find the Cartesian coordinates of A, B, C, and D, and prove that
  for any point P on C1, |PA|^2 + |PB|^2 + |PC|^2 + |PD|^2 is within the range [32, 52]. -/
theorem coordinate_system_and_parametric_equations
  (φ : ℝ)
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (P : ℝ → ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (t : ℝ)
  (H1 : ∀ φ, P φ = (2 * Real.cos φ, 3 * Real.sin φ))
  (H2 : A = (1, Real.sqrt 3) ∧ B = (-Real.sqrt 3, 1) ∧ C = (-1, -Real.sqrt 3) ∧ D = (Real.sqrt 3, -1))
  (H3 : ∀ p : ℝ × ℝ, ∃ φ, p = P φ)
  : ∀ x y, ∃ (φ : ℝ), P φ = (x, y) →
    ∃ t, t = |P φ - A|^2 + |P φ - B|^2 + |P φ - C|^2 + |P φ - D|^2 ∧ 32 ≤ t ∧ t ≤ 52 := 
sorry

end coordinate_system_and_parametric_equations_l480_480982


namespace projective_transformation_l480_480175

-- Noncomputable as projection involves geometric construction
noncomputable def projection_map (l : Type*) (circle : Type*) (M N : circle) (P : l → l) : Prop :=
∀ (A B C D : l), cross_ratio (P A) (P B) (P C) (P D) = cross_ratio A B C D

-- Main theorem stating that the transformation P is projective
theorem projective_transformation (l : Type*) (circle : Type*) (M N : circle) (P : l → l)
    (hM: M ∉ l) (hN: N ∉ l) (H: projection_map l circle M N P) : 
    ∀ (A B C D : l), cross_ratio (P A) (P B) (P C) (P D) = cross_ratio A B C D :=
sorry

end projective_transformation_l480_480175


namespace find_multiplier_l480_480065

theorem find_multiplier (x : ℝ) (y : ℝ) (h1 : x = 62.5) (h2 : (y * (x + 5)) / 5 - 5 = 22) : y = 2 :=
sorry

end find_multiplier_l480_480065


namespace find_y_l480_480139

theorem find_y (y : ℝ) 
  (h : 2 * real.arctan (1/5) + 2 * real.arctan (1/25) + real.arctan (1/y) = real.pi / 4) : 
  y = 1 :=
sorry

end find_y_l480_480139


namespace extra_toes_l480_480525

noncomputable def dog_nails (dogs : ℕ) (feet_per_dog : ℕ) (nails_per_foot : ℕ) : ℕ :=
  dogs * (feet_per_dog * nails_per_foot)

noncomputable def parrot_claws (parrots : ℕ) (legs_per_parrot : ℕ) (claws_per_leg : ℕ) : ℕ :=
  parrots * (legs_per_parrot * claws_per_leg)

noncomputable def total_nails (dog_nails : ℕ) (parrot_claws : ℕ) : ℕ :=
  dog_nails + parrot_claws

theorem extra_toes {dogs feet_per_dog nails_per_foot parrots legs_per_parrot claws_per_leg standard_parrots total_cut_nails : ℕ} :
  dogs = 4 →
  feet_per_dog = 4 →
  nails_per_foot = 4 →
  parrots = 8 →
  legs_per_parrot = 2 →
  claws_per_leg = 3 →
  standard_parrots = parrots - 1 →
  total_cut_nails = 113 →
  let standard_nails := total_nails (dog_nails dogs feet_per_dog nails_per_foot) (parrot_claws standard_parrots legs_per_parrot claws_per_leg) in
  total_cut_nails - standard_nails = 7 :=
by
  intros
  sorry

end extra_toes_l480_480525


namespace bug_visits_tiles_l480_480082

theorem bug_visits_tiles (width length : ℕ) (gcd_width_length : ℕ) (broken_tile : ℕ × ℕ)
  (h_width : width = 12) (h_length : length = 25) (h_gcd : gcd_width_length = Nat.gcd width length)
  (h_broken_tile : broken_tile = (12, 18)) :
  width + length - gcd_width_length = 36 := by
  sorry

end bug_visits_tiles_l480_480082


namespace sum_of_x_i_l480_480307

theorem sum_of_x_i (S : Set (ℂ × ℂ × ℂ)) 
  (h1 : ∀ ⟨x, y, z⟩ ∈ S, x + y * z = (7 : ℂ))
  (h2 : ∀ ⟨x, y, z⟩ ∈ S, y + x * z = (10 : ℂ))
  (h3 : ∀ ⟨x, y, z⟩ ∈ S, z + x * y = (10 : ℂ)) :
  ∑ ⟨x, _, _⟩ in S, x = 7 := 
sorry

end sum_of_x_i_l480_480307


namespace committee_count_l480_480069

theorem committee_count (A B C : Type) (a_count b_count c_count : ℕ) 
  (ha : a_count = 6) (hb : b_count = 7) (hc : c_count = 5) 
  (a_members : fintype A) (b_members : fintype B) (c_members : fintype C) 
  (card_A : fintype.card A = a_count) (card_B : fintype.card B = b_count) (card_C : fintype.card C = c_count) :
  (∃ f : A × B × C, true) ∧ fintype.card (A × B × C) = 210 :=
by
  sorry

end committee_count_l480_480069


namespace problem_1_problem_2_l480_480190

-- First problem: Proving the expression for f(x)
theorem problem_1 (f : ℝ → ℝ) (t : ℤ) 
  (h1 : ∀ x : ℝ, f x = x^(-t^2 + 2*t + 3))
  (h2 : ∀ x: ℝ, 0 < x → f x < f (x + 1))
  (h3 : ∀ x : ℝ, f x = f (-x)) :
  f = (λ x : ℝ, x^4) :=
sorry

-- Second problem: Proving the range of a
theorem problem_2 (g f: ℝ → ℝ) (a : ℝ)
  (h1 : f = (λ x : ℝ, x^4))
  (h2 : ∀ x : ℝ, g x = log a (a * sqrt (f x) - x))
  (h3 : monotone_on g (set.Icc 2 4))
  (h4 : a > 0) (h5: a ≠ 1) :
  1 / 2 < a ∧ a < 1 :=
sorry

end problem_1_problem_2_l480_480190


namespace complex_frac_eq_l480_480595

theorem complex_frac_eq (a b : ℝ) (i : ℂ) (h : i^2 = -1)
  (h1 : (1 - i) / (1 + i) = a + b * i) : a - b = 1 :=
by
  sorry

end complex_frac_eq_l480_480595


namespace expression_value_l480_480962

theorem expression_value (a b c d : ℤ) (h_a : a = 15) (h_b : b = 19) (h_c : c = 3) (h_d : d = 2) :
  (a - (b - c)) - ((a - b) - c + d) = 4 := 
by
  rw [h_a, h_b, h_c, h_d]
  sorry

end expression_value_l480_480962


namespace david_marks_in_mathematics_l480_480121

theorem david_marks_in_mathematics :
  ∃ M : ℕ,
    let E := 74 in
    let P := 82 in
    let C := 67 in
    let B := 90 in
    let average := 75.6 in
    let num_subjects := 5 in
    let total_marks := average * num_subjects in
    let total_obtained := E + P + C + B + M in
    total_marks = 378 ∧ total_obtained = 378 ∧ M = 65 :=
begin
  use 65,
  sorry,
end

end david_marks_in_mathematics_l480_480121


namespace evaluate_expression_l480_480984

theorem evaluate_expression : (831 * 831) - (830 * 832) = 1 :=
by
  sorry

end evaluate_expression_l480_480984


namespace cos_75_value_l480_480117

noncomputable def cos_75_degree : ℝ :=
  let cos_60 := 1 / 2
  let sin_60 := real.sqrt 3 / 2
  let cos_15 := (real.sqrt 6 + real.sqrt 2) / 4
  let sin_15 := (real.sqrt 6 - real.sqrt 2) / 4
  cos_60 * cos_15 - sin_60 * sin_15

theorem cos_75_value : cos_75_degree = (real.sqrt 6 - real.sqrt 2) / 4 :=
by sorry

end cos_75_value_l480_480117


namespace area_sum_AMP_BNP_eq_area_CQR_l480_480706

variables {A B C P Q R M N : Type*}
variables [Field A] [Field B] [Field C] [Field P] [Field Q] [Field R] [Field M] [Field N]
variables (triangle_ABC : Triangle A B C)
variables (P_on_AB : P ∈ Line A B)
variables (Q_on_BC : Q ∈ Line B C)
variables (R_on_AC : R ∈ Line A C)
variables (PQCR_is_parallelogram : IsParallelogram P Q C R)
variables (M_is_intersection_AQ_PR : M = Intersection (Line A Q) (Line P R))
variables (N_is_intersection_BR_PQ : N = Intersection (Line B R) (Line P Q))

theorem area_sum_AMP_BNP_eq_area_CQR 
  (A B C P Q R M N : Type*) [Field A] [Field B] [Field C] [Field P] [Field Q] [Field R] [Field M] [Field N]
  (triangle_ABC : Triangle A B C)
  (P_on_AB : P ∈ Line A B)
  (Q_on_BC : Q ∈ Line B C)
  (R_on_AC : R ∈ Line A C)
  (PQCR_is_parallelogram : IsParallelogram P Q C R)
  (M_is_intersection_AQ_PR : M = Intersection (Line A Q) (Line P R))
  (N_is_intersection_BR_PQ : N = Intersection (Line B R) (Line P Q)) :
  Area (Triangle A M P) + Area (Triangle B N P) = Area (Triangle C Q R) :=
by sorry

end area_sum_AMP_BNP_eq_area_CQR_l480_480706


namespace probability_of_exactly_five_correct_letters_is_zero_l480_480787

theorem probability_of_exactly_five_correct_letters_is_zero :
  ∀ (envelopes : Fin 6 → ℕ), (∃! (f : Fin 6 → Fin 6), Bijective f ∧ ∀ i, envelopes (f i) = envelopes i) → 
  (Probability (exactly_five_correct ∈ permutations (Fin 6)) = 0) :=
by
  sorry

noncomputable def exactly_five_correct (σ : Fin 6 → Fin 6) : Prop :=
  (∃ (i : Fin 6), ∀ j, j ≠ i → σ j = j) ∧ ∃ (k : Fin 6), k ≠ i ∧ σ k ≠ k

noncomputable def permutations (α : Type) := List.permutations (List.finRange 6)

end probability_of_exactly_five_correct_letters_is_zero_l480_480787


namespace largest_number_l480_480860

theorem largest_number
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ)
  (ha : a = 0.883) (hb : b = 0.8839) (hc : c = 0.88) (hd : d = 0.839) (he : e = 0.889) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by {
  sorry
}

end largest_number_l480_480860


namespace pyramid_surface_area_l480_480299

theorem pyramid_surface_area (A B C D : Type) 
  [has_triangle_structure A B C]
  (outside_plane : point_outside_plane D A B C)
  (all_faces_triangular : pyramid_DABC_faces_triangular D ABC)
  (edge_lengths : ∀ e ∈ edges_of DABC, e.length = 15 ∨ e.length = 36)
  (no_face_equilateral : ∀ face ∈ faces_of DABC, ¬equilateral face) :
  surface_area DABC = 30 * real.sqrt 1239.75 :=
by
  sorry

end pyramid_surface_area_l480_480299


namespace midpoint_of_constant_length_segment_describes_circle_l480_480924

theorem midpoint_of_constant_length_segment_describes_circle
  (a b : Line)           -- two mutually perpendicular skew lines
  (length_ab : ℝ)        -- length of segment AB
  (midpoint_curve : Curve) (γ : Plane)  -- midpoint's path and relevant plane
  (A B : Point)          -- endpoints of the segment
  (M : Point)            -- midpoint of the segment
  (h1 : A ∈ a) (h2 : B ∈ b)
  (h3 : |A - B| = length_ab)  -- length condition of the segment
  (h4 : M = midpoint A B)     -- M is midpoint of A and B
  (a' b' : Line)              -- projections of a and b onto plane γ
  (O := intersection_point a' b')  -- O, the intersection of a' and b'
  (h5 : a' ⊥ b')                     -- projection lines are mutually perpendicular
  (r : ℝ := length_ab / 2)    -- radius of the circle is half the segment length
  : midpoint_curve = Circle(O, r) := 
sorry

end midpoint_of_constant_length_segment_describes_circle_l480_480924


namespace count_distinct_integer_sums_of_special_fractions_l480_480964

def is_special_fraction (a b : ℕ) : Prop :=
  a + b = 17 ∧ a > 0 ∧ b > 0

def special_fractions : List (ℚ) :=
  [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9),
   (9, 8), (10, 7), (11, 6), (12, 5), (13, 4), (14, 3), (15, 2), (16, 1)].map
   (λ p, p.1 / p.2 : ℚ)

def special_fraction_sums : List ℚ :=
  special_fractions.product special_fractions |>.map (λ p, p.1 + p.2)

def distinct_integer_sums : List ℤ :=
  special_fraction_sums.filter_map (λ q, if q.den = 1 then some q.num else none)

theorem count_distinct_integer_sums_of_special_fractions : 
  distinct_integer_sums.length = 8 :=
sorry

end count_distinct_integer_sums_of_special_fractions_l480_480964


namespace completing_the_square_equation_l480_480035

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l480_480035


namespace polynomial_102_l480_480415

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l480_480415


namespace sum_first_fifteen_multiples_of_17_l480_480818

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480818


namespace maximize_ratio_at_pi_over_4_l480_480346

variable (α : ℝ)

-- Define the conditions
def isRightAngledTriangle (α : ℝ) : Prop :=
  0 < α ∧ α < π / 2

def inradius (α R : ℝ) : ℝ :=
  (R * Math.sin α * Math.cos α) / (Math.sin α + Math.cos α + 1)

def circumradius (R : ℝ) : ℝ :=
  R

def radiusRatio (α R : ℝ) : ℝ :=
  inradius α R / circumradius R

-- Statement to prove that α = π / 4 maximizes the ratio
theorem maximize_ratio_at_pi_over_4 : isRightAngledTriangle α → α = π / 4 :=
by
  sorry

end maximize_ratio_at_pi_over_4_l480_480346


namespace cone_and_cylinder_volume_l480_480893

noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem cone_and_cylinder_volume : 
  volume_cone 3 6 = 56.52 ∧ volume_cylinder 3 6 = 169.56 :=
by
  sorry

end cone_and_cylinder_volume_l480_480893


namespace traced_area_computation_l480_480086

noncomputable def area_traced_on_larger_sphere 
  (r_small_sphere : ℝ) 
  (r_inner_sphere : ℝ) 
  (r_outer_sphere : ℝ) 
  (blue_trace_area_on_inner : ℝ) : ℝ :=
  blue_trace_area_on_inner * ((4 * π * r_outer_sphere^2) / (4 * π * r_inner_sphere^2))

theorem traced_area_computation :
  area_traced_on_larger_sphere 1 4 6 17 = 38.25 :=
by
  have : area_traced_on_larger_sphere 1 4 6 17 = 17 * (144 * π / 64 * π),
  calc
    area_traced_on_larger_sphere 1 4 6 17 = 17 * (144 * π / 64 * π) : rfl
    ... = 17 * (9 / 4) : by norm_num
    ... = 38.25 : by norm_num
  exact this

end traced_area_computation_l480_480086


namespace percentage_of_second_investment_l480_480002

theorem percentage_of_second_investment :
  ∀ (total_investment amount_at_4_percent total_with_interest : ℝ),
  total_investment = 1000 →
  amount_at_4_percent ≈ 699.99 →
  total_with_interest = 1046 →
  ∃ rate : ℝ, rate ≈ 0.06 :=
by
  sorry

end percentage_of_second_investment_l480_480002


namespace find_two_numbers_l480_480463

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480463


namespace combination_60_2_l480_480532

theorem combination_60_2 : nat.choose 60 2 = 1770 :=
by sorry

end combination_60_2_l480_480532


namespace tank_capacity_l480_480869

theorem tank_capacity (T : ℚ) (h1 : 0 ≤ T)
  (h2 : 9 + (3 / 4) * T = (9 / 10) * T) : T = 60 :=
sorry

end tank_capacity_l480_480869


namespace simplify_expression_l480_480109

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l480_480109


namespace gym_monthly_cost_l480_480278

theorem gym_monthly_cost (down_payment total_cost total_months : ℕ) (h_down_payment : down_payment = 50) (h_total_cost : total_cost = 482) (h_total_months : total_months = 36) : 
  (total_cost - down_payment) / total_months = 12 := by 
  sorry

end gym_monthly_cost_l480_480278


namespace smoking_related_to_lung_disease_l480_480072

theorem smoking_related_to_lung_disease
  (sample_size : ℕ)
  (k2_calc : ℝ)
  (p_thresh_95 : ℝ)
  (p_thresh_99 : ℝ)
  (condition1 : sample_size = 1000)
  (condition2 : k2_calc = 5.231)
  (condition3 : p_thresh_95 = 3.841)
  (condition4 : p_thresh_99 = 6.635) :
  k2_calc ≥ p_thresh_95 ∧ k2_calc < p_thresh_99 → 
  "smoking is related to lung disease with more than 95% confidence" :=
by 
  sorry

end smoking_related_to_lung_disease_l480_480072


namespace divisors_64n3_l480_480161

theorem divisors_64n3 (n : ℕ) (hn_pos : 0 < n)
  (divisors_84n2 : (∀ d : ℕ , d ∣ 84 * n^2 → 84)) :
  ∃ num_divisors : ℕ, num_divisors = 160 := 
  sorry

end divisors_64n3_l480_480161


namespace max_marked_points_7x7_max_marked_points_13x13_l480_480254

-- Problem statement for 7x7 grid
theorem max_marked_points_7x7 : 
  ∃ k : ℕ, k = 21 ∧ ∀ (x : ℕ → ℕ) (hn : ∑ i in Finset.range 7, x i = k),
    (∑ i in Finset.range 7, x i * (x i - 1) / 2) ≤ 21 :=
by sorry

-- Problem statement for 13x13 grid
theorem max_marked_points_13x13 : 
  ∃ k : ℕ, k = 52 ∧ ∀ (x : ℕ → ℕ) (hn : ∑ i in Finset.range 13, x i = k),
    (∑ i in Finset.range 13, x i * (x i - 1) / 2) ≤ 78 :=
by sorry

end max_marked_points_7x7_max_marked_points_13x13_l480_480254


namespace brown_rabbit_hop_distance_l480_480777

theorem brown_rabbit_hop_distance
  (w : ℕ) (b : ℕ) (t : ℕ)
  (h1 : w = 15)
  (h2 : t = 135)
  (hop_distance_in_5_minutes : w * 5 + b * 5 = t) : 
  b = 12 :=
by
  sorry

end brown_rabbit_hop_distance_l480_480777


namespace angle_ABC_is_126_degrees_l480_480269

variables {A B C D O : Type}
variables [EuclideanGeometry A B C D O]

-- Given conditions
variables (h1 : BC ∥ AD)
variables (h2 : bisects CA (∠ BCD))
variables (h3 : intersection O (diagonals AC BD))
variables (h4 : CD = AO)
variables (h5 : BC = OD)

-- Proof statement
theorem angle_ABC_is_126_degrees (A B C D O : Point)    
(h1 : Parallel BC AD)
(h2 : ∃ P, Bisects CA BCD P)
(h3 : Inter O (diagonals A C B D))
(h4 : Eq CD AO)
(h5 : Eq BC OD)
:
  Angle ABC = 126 :=
begin
  sorry
end

end angle_ABC_is_126_degrees_l480_480269


namespace expected_value_of_difference_is_4_point_5_l480_480782

noncomputable def expected_value_difference : ℚ :=
  (2 * 6 / 56 + 3 * 10 / 56 + 4 * 12 / 56 + 5 * 12 / 56 + 6 * 10 / 56 + 7 * 6 / 56)

theorem expected_value_of_difference_is_4_point_5 :
  expected_value_difference = 4.5 := sorry

end expected_value_of_difference_is_4_point_5_l480_480782


namespace exist_a_not_equal_r_l480_480291

def is_odd_prime (p : ℕ) : Prop := 
  Nat.Prime p ∧ p % 2 = 1

def N (p : ℕ) := (1 / 4 * (p^3 - p) - 1).toNat

def r (n : ℕ) (reds : ℕ) : ℚ := reds / n

theorem exist_a_not_equal_r (p : ℕ) (a : ℕ) :
  is_odd_prime p ∧ 1 ≤ a ∧ a < p →
  ∃ a ∈ (Finset.range (p - 1)).image (+1), 
  ∀ n ∈ Finset.range (N p), ∀ reds, r n reds ≠ a / p := 
by
  sorry

end exist_a_not_equal_r_l480_480291


namespace radius_of_pool_l480_480479
open Real

-- Assumptions and conditions
variables {r : ℝ} (h1 : r > 0)
def pool_area := π * r^2
def larger_circle_area := π * (r + 4)^2
def concrete_wall_area := larger_circle_area - pool_area

-- The problem statement
theorem radius_of_pool (h1 : r > 0) (h2 : concrete_wall_area = (11 / 25) * pool_area) : r = 20 :=
by
  sorry

end radius_of_pool_l480_480479


namespace max_intersections_of_lines_l480_480310

-- Define the conditions: distinct lines and properties of parallelism and intersection at point A
def lines : Set (ℕ → Prop) := λ n, n > 0 ∧ n ≤ 120

def parallel_lines (L : ℕ → Prop) : Prop :=
  ∀ n m, L (5 * n) = L (5 * m)

def lines_through_A (L : ℕ → Prop) (A : Point) : Prop :=
  ∀ n m, L (5 * n - 4) ∩ {A} ≠ ∅

-- The final statement to be proved
theorem max_intersections_of_lines : 
  let L := lines in
  let A := some_point in
  let X := λ n, n % 5 = 0 in 
  let Y := λ n, n % 5 = 1 in 
  let Z := lines \ (X ∪ Y) in
  (num_intersections X + num_intersections Y + num_intersections Z + 
  intersections_between X Y + intersections_between Y Z + 
  intersections_between Z X = 6589) :=
sorry

end max_intersections_of_lines_l480_480310


namespace sonika_initial_deposit_l480_480342

-- Given conditions
def total_amount_after_4_years (P R : ℝ) : ℝ :=
  P + (P * R * 4 / 100)

-- The problem statement
theorem sonika_initial_deposit (P : ℝ) (R : ℝ) :
  total_amount_after_4_years P R = 19500 ∧
  total_amount_after_4_years P (R + 3) = 20940 →
  P = 12000 :=
by
  sorry

end sonika_initial_deposit_l480_480342


namespace calculate_a_mul_a_sub_3_l480_480107

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l480_480107


namespace cone_volume_correct_l480_480498

-- Define the conditions
def cross_section_is_equilateral_triangle (side_length : ℝ) : Prop :=
  side_length = 2

def cone_volume (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

-- Define the theorem to prove the question equals the correct answer given the conditions
theorem cone_volume_correct :
  ∀ (r h : ℝ), cross_section_is_equilateral_triangle 2 →
    (r = 1 ∧ h = Real.sqrt 3) →
    cone_volume r h = (Real.sqrt 3 / 3) * Real.pi :=
by
  intros r h equilateral_triangle (r_eq h_eq)
  rw[r_eq, h_eq]
  sorry

end cone_volume_correct_l480_480498


namespace rate_of_stream_equation_l480_480884

theorem rate_of_stream_equation 
  (v : ℝ) 
  (boat_speed : ℝ) 
  (travel_time : ℝ) 
  (distance : ℝ)
  (h_boat_speed : boat_speed = 16)
  (h_travel_time : travel_time = 5)
  (h_distance : distance = 105)
  (h_equation : distance = (boat_speed + v) * travel_time) : v = 5 :=
by 
  sorry

end rate_of_stream_equation_l480_480884


namespace factorize_m_factorize_x_factorize_xy_l480_480332

theorem factorize_m (m : ℝ) : m^2 + 7 * m - 18 = (m - 2) * (m + 9) := 
sorry

theorem factorize_x (x : ℝ) : x^2 - 2 * x - 8 = (x + 2) * (x - 4) :=
sorry

theorem factorize_xy (x y : ℝ) : (x * y)^2 - 7 * (x * y) + 10 = (x * y - 2) * (x * y - 5) := 
sorry

end factorize_m_factorize_x_factorize_xy_l480_480332


namespace probability_of_neither_in_range_l480_480365

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def is_neither_square_nor_cube_nor_fifth_power (n : ℤ) : Prop :=
  ¬ is_perfect_square n ∧ ¬ is_perfect_cube n ∧ ¬ is_perfect_fifth_power n

theorem probability_of_neither_in_range :
  (Finset.filter is_neither_square_nor_cube_nor_fifth_power (Finset.range 201)).card / 200 = 179 / 200 :=
sorry

end probability_of_neither_in_range_l480_480365


namespace expression_equality_l480_480856

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l480_480856


namespace sum_of_first_fifteen_multiples_of_17_l480_480851

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480851


namespace cyclic_KLMN_l480_480678

noncomputable theory

open scoped Classical

variable {P : Type} [euclidean_geometry P]

-- Definitions based on given conditions
def is_right_triangle (A B C : P) : Prop :=
  ∃ γ : angle, γ + γ = pi ∧ γ = ∠ A B C ∧ ∠ B C A = pi / 2

def is_foot_of_perpendicular (P Q R F : P) : Prop :=
  ∠ P Q F = ∠ Q F R ∧ ∠ P R F = ∠ Q R F ∧ ∠ P F R = ∠ Q F P

def semicircle (PC : set P) [is_semicircle PC] (p q : P) : Prop :=
  ∃ M, is_midpoint p q M ∧ p ≠ q

def cyclic_quad (P Q R S : P) : Prop :=
  ∃ γ δ ε ζ : angle, γ + ε = pi ∧ δ + ζ = pi ∧ γ = ∠ P Q R ∧ δ = ∠ Q R S ∧ ε = ∠ R S P ∧ ζ = ∠ S P Q

-- Main statement
theorem cyclic_KLMN {A B C D E K L M N : P} :
  is_right_triangle A B C →
  D ∈ line_segment A C →
  E ∈ line_segment B C →
  semicircle (set_of (λ p, ∃ q, q ∈ line_segment A C)) ∧
  semicircle (set_of (λ p, ∃ q, q ∈ line_segment B C)) ∧
  semicircle (set_of (λ p, ∃ q, q ∈ line_segment C D)) ∧
  semicircle (set_of (λ p, ∃ q, q ∈ line_segment C E)) →
  K ∈ (C_1 ∩ C_2) ∧
  M ∈ (C_3 ∩ C_4) ∧
  L ∈ (C_2 ∩ C_3) ∧
  N ∈ (C_1 ∩ C_4) →
  cyclic_quad K L M N :=
by sorry

end cyclic_KLMN_l480_480678


namespace geometric_sequence_seventh_term_l480_480667

noncomputable def a_7 (a₁ q : ℝ) : ℝ :=
  a₁ * q^6

theorem geometric_sequence_seventh_term :
  a_7 3 (Real.sqrt 2) = 24 :=
by
  sorry

end geometric_sequence_seventh_term_l480_480667


namespace projection_addition_l480_480695

open Real

variables (u z : ℝ × ℝ)

-- Define the projection operator
def proj (u z : ℝ × ℝ) : ℝ × ℝ :=
  let dot : ℝ := (u.1 * z.1 + u.2 * z.2)
  let norm_sq : ℝ := (z.1 ^ 2 + z.2 ^ 2)
  (dot / norm_sq * z.1, dot / norm_sq * z.2)

theorem projection_addition (h : proj u z = (4, 3)) :
  proj (7 * u) z + (1, -1) = (29, 20) :=
by
  sorry

end projection_addition_l480_480695


namespace quadratic_inequality_solution_set_l480_480205

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, (2 < x ∧ x < 3) → (ax^2 + 5*x + b > 0)) →
  ∃ x : ℝ, (-1/2 < x ∧ x < -1/3) :=
sorry

end quadratic_inequality_solution_set_l480_480205


namespace gain_percent_of_cost_selling_relation_l480_480049

theorem gain_percent_of_cost_selling_relation (C S : ℕ) (h : 50 * C = 45 * S) : 
  (S > C) ∧ ((S - C) / C * 100 = 100 / 9) :=
by
  sorry

end gain_percent_of_cost_selling_relation_l480_480049


namespace max_distance_sum_l480_480636

variables {V : Type} [inner_product_space ℝ V] (a b c d : V)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1) (hd : ∥d∥ = 1)

theorem max_distance_sum :
  ∥a - b∥^2 + ∥a - c∥^2 + ∥a - d∥^2 + ∥b - c∥^2 + ∥b - d∥^2 + ∥c - d∥^2 ≤ 16 :=
sorry

end max_distance_sum_l480_480636


namespace max_square_plots_l480_480901

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l480_480901


namespace problem_solution_l480_480194

variable (x y z : ℝ)

theorem problem_solution
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + x * y = 8)
  (h2 : y + z + y * z = 15)
  (h3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 :=
sorry

end problem_solution_l480_480194


namespace zhang_san_correct_prob_l480_480779

theorem zhang_san_correct_prob
  (P_A1 : ℝ) (h1 : P_A1 = 3/4)
  (P_A2 : ℝ) (h2 : P_A2 = 1/4)
  (P_B_given_A1 : ℝ) (h3 : P_B_given_A1 = 3/4)
  (P_B_given_A2 : ℝ) (h4 : P_B_given_A2 = 1/4) :
  let P_B := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 in
  P_B = 5/8 :=
by {
  sorry
}

end zhang_san_correct_prob_l480_480779


namespace original_price_l480_480676

-- Definitions of conditions
def SalePrice : Float := 70
def DecreasePercentage : Float := 30

-- Statement to prove
theorem original_price (P : Float) (h : 0.70 * P = SalePrice) : P = 100 := by
  sorry

end original_price_l480_480676


namespace find_numbers_l480_480454

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480454


namespace find_values_l480_480948

-- Conditions on the function f
variables {f : ℝ → ℝ}
variables {x : ℝ}

-- Defining the conditions: even function, defined on [-2,2], decreasing on [0,2], value at 1/2.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_defined_on_Icc (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x, a ≤ x ∧ x ≤ b → f x = f x
def is_decreasing_on_Icc (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y
def at_one_half (f : ℝ → ℝ) : Prop := f (1 / 2) = 0

-- Defining the logarithmic inequality
def log_base (b y : ℝ) : ℝ := log y / log b

theorem find_values (h1 : is_even f)
                    (h2 : is_defined_on_Icc f (-2) 2)
                    (h3 : is_decreasing_on_Icc f 0 2)
                    (h4 : at_one_half f) :
  { x : ℝ | f (log_base (1 / 4) x) < 0 } = { x : ℝ | (2 < x ∧ x ≤ 16) ∨ (1 / 16 ≤ x ∧ x < 1 / 2) } :=
by
  sorry

end find_values_l480_480948


namespace tree_growth_years_l480_480854

def initial_height : ℝ := 4
def growth_rate : ℝ := 0.4
def end_of_4th_year_height : ℝ := initial_height + 4 * growth_rate
def final_height : ℝ := end_of_4th_year_height * (8 / 7)

theorem tree_growth_years (initial_height end_of_4th_year_height final_height : ℝ) :
  initial_height = 4 →
  growth_rate = 0.4 →
  end_of_4th_year_height = initial_height + 4 * growth_rate →
  final_height = end_of_4th_year_height * (8 / 7) →
  (final_height - end_of_4th_year_height) / growth_rate + 4 = 6 :=
sorry

end tree_growth_years_l480_480854


namespace max_value_x_sub_2z_l480_480637

theorem max_value_x_sub_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 16) :
  ∃ m, m = 4 * Real.sqrt 5 ∧ ∀ x y z, x^2 + y^2 + z^2 = 16 → x - 2 * z ≤ m :=
sorry

end max_value_x_sub_2z_l480_480637


namespace find_number_l480_480880

theorem find_number (x : ℕ) (h : 112 * x = 70000) : x = 625 :=
by
  sorry

end find_number_l480_480880


namespace numbers_pairs_sum_prod_l480_480424

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480424


namespace find_area_S_find_a_given_c_eq_1_l480_480654

-- Definitions based on the conditions
variables {a b c A B C : Real}
variables (triangle_ABC : Triangle) -- triangle ABC
variables (h1 : 3 * a * sin C = 4 * c * cos A) -- given condition 1
variables (h2 : (vector AB).dot (vector AC) = 3) -- given condition 2

-- Statements to prove
theorem find_area_S (h_triangle : triangle_ABC):
  S(triangle_ABC) = 2 := by sorry

theorem find_a_given_c_eq_1 (h_triangle : triangle_ABC) (h_c : c = 1):
  a = 2 * sqrt 5 := by sorry

end find_area_S_find_a_given_c_eq_1_l480_480654


namespace laptop_price_difference_l480_480536

theorem laptop_price_difference :
  let list_price := 59.99
  let tech_bargains_discount := 15
  let budget_bytes_discount_percentage := 0.30
  let tech_bargains_price := list_price - tech_bargains_discount
  let budget_bytes_price := list_price * (1 - budget_bytes_discount_percentage)
  let cheaper_price := min tech_bargains_price budget_bytes_price
  let expensive_price := max tech_bargains_price budget_bytes_price
  (expensive_price - cheaper_price) * 100 = 300 :=
by
  sorry

end laptop_price_difference_l480_480536


namespace find_solution_l480_480557

open Nat
open Primes

theorem find_solution (n x y z : ℤ) (p : ℕ) (h1 : Prime p) (h2 : x > 0) (h3 : y > 0) (h4 : z > 0) :
  (x^2 + 4 * y^2) * (y^2 + 4 * z^2) * (z^2 + 4 * x^2) = p^n →
  p = 5 →
  (x, y, z) = (1, 1, 1) :=
by
  sorry

end find_solution_l480_480557


namespace tree_height_function_tree_height_at_0_3_l480_480555

-- Definitions of the conditions
def tree_height_linear (k b x : ℝ) : ℝ := k * x + b

theorem tree_height_function :
  ∃ k b : ℝ, (tree_height_linear k b 0.2 = 20) ∧ (tree_height_linear k b 0.28 = 22) ∧
  (∀ x : ℝ, tree_height_linear k b x = 25 * x + 15) :=
begin
  -- Statement that there exist k and b satisfying the given conditions
  use [25, 15], -- Use the solution values directly
  split,
  -- Prove the first condition
  {
    unfold tree_height_linear,
    norm_num,
  },
  split,
  -- Prove the second condition
  {
    unfold tree_height_linear,
    norm_num,
  },
  -- Prove the general form of the linear function
  {
    intros,
    unfold tree_height_linear,
    norm_num,
  }
end

theorem tree_height_at_0_3 :
  ∃ y : ℝ, tree_height_linear 25 15 0.3 = y ∧ y = 22.5 :=
begin
  use 22.5,
  split,
  -- Prove that the height at x = 0.3 is 22.5 using the function
  {
    unfold tree_height_linear,
    norm_num,
  },
  -- Simply state the value derived
  {
    norm_num,
  }
end

end tree_height_function_tree_height_at_0_3_l480_480555


namespace xyz_ineq_l480_480345

theorem xyz_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := 
sorry

end xyz_ineq_l480_480345


namespace inequality_proof_l480_480586

open Real

noncomputable def xyz_pos_real (x y z : ℝ) (alpha : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 1 ∧ alpha ≥ 0 

theorem inequality_proof (x y z α : ℝ) (h : xyz_pos_real x y z α) :
  (x + y + z) (λ (x' y' : ℝ), (x' ^ (α + 3) + y' ^ (α + 3)) / (x' ^ 2 + x' * y' + y' ^ 2)) ≥ 2 := 
sorry

end inequality_proof_l480_480586


namespace inequality_check_l480_480416

theorem inequality_check : (-1 : ℝ) / 3 < -1 / 5 := 
by 
  sorry

end inequality_check_l480_480416


namespace snow_at_least_once_prob_l480_480980

-- Define the conditions for the problem
def prob_snow_day1_to_day4 : ℚ := 1 / 2
def prob_no_snow_day1_to_day4 : ℚ := 1 - prob_snow_day1_to_day4

def prob_snow_day5_to_day7 : ℚ := 1 / 3
def prob_no_snow_day5_to_day7 : ℚ := 1 - prob_snow_day5_to_day7

-- Define the probability of no snow during the first week of February
def prob_no_snow_week : ℚ := (prob_no_snow_day1_to_day4 ^ 4) * (prob_no_snow_day5_to_day7 ^ 3)

-- Define the probability that it snows at least once during the first week of February
def prob_snow_at_least_once : ℚ := 1 - prob_no_snow_week

-- The theorem we want to prove
theorem snow_at_least_once_prob : prob_snow_at_least_once = 53 / 54 :=
by
  sorry

end snow_at_least_once_prob_l480_480980


namespace solution_l480_480385

variable (n : ℕ)

def y (x : ℝ) := Real.exp x

-- Coordinates of the quadrilateral vertices
def vertex1 := (n : ℝ , y n)
def vertex2 := (n+2 : ℝ , y (n+2))
def vertex3 := (n+4 : ℝ , y (n+4))
def vertex4 := (n+6 : ℝ , y (n+6))

def shoelace_area : ℝ :=
    0.5 * Real.abs (
      (y n) * (n+2) +
      (y (n+2)) * (n+4) +
      (y (n+4)) * (n+6) +
      (y (n+6)) * n -
      (y (n+2)) * n -
      (y (n+4)) * (n+2) -
      (y (n+6)) * (n+4) -
      (y n) * (n+6)
    )

theorem solution :
  (shoelace_area n = Real.exp 6 - Real.exp 2) → n = 2 :=
by
  sorry

end solution_l480_480385


namespace sum_of_first_fifteen_multiples_of_17_l480_480846

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in finset.range 15, 17 * (i + 1)) = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480846


namespace permutations_of_BANANA_l480_480543

theorem permutations_of_BANANA : 
  let word := ["B", "A", "N", "A", "N", "A"]
  let total_letters := 6
  let repeated_A := 3
  (total_letters.factorial / repeated_A.factorial) = 120 :=
by
  sorry

end permutations_of_BANANA_l480_480543


namespace probability_three_heads_with_two_tails_before_third_head_l480_480539

-- Define the probability of a fair coin flip resulting in heads or tails
def fair_coin_flip_probability : ℝ := 1 / 2

-- Define the sequence requirement
def sequence_THTH_probability : ℝ := (fair_coin_flip_probability) ^ 4

-- Define P as the probability that Debra will get three heads in a row after THTH
def probability_P : ℝ := 1 / 3

-- Prove that the overall probability is 1/48
theorem probability_three_heads_with_two_tails_before_third_head :
  sequence_THTH_probability * probability_P = 1 / 48 := by
  -- define the known probabilities
  have prob_THTH := sequence_THTH_probability
  have P := probability_P
  
  calc
    prob_THTH * P = (1 / 16) * (1 / 3) : by
      rw [sequence_THTH_probability, probability_P]
    ... = 1 / 48 : by norm_num

end probability_three_heads_with_two_tails_before_third_head_l480_480539


namespace sum_of_all_digits_divisible_by_nine_l480_480710

theorem sum_of_all_digits_divisible_by_nine :
  ∀ (A B C D : ℕ),
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
  sorry

end sum_of_all_digits_divisible_by_nine_l480_480710


namespace max_min_h_range_m_l480_480212

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2 : ℝ) * x^2 - (3 / 2 : ℝ) * x + m
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := g(x, m) - f(x)

theorem max_min_h (m : ℝ) :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, h x m ≤ h 1 m) ∧ (∀ x ∈ Set.Icc (1 : ℝ) 3, h x m ≥ h 2 m) :=
sorry

theorem range_m (m : ℝ) :
  (∃ x : ℝ, x > (3/2 : ℝ) ∧ m = (1 / 2 : ℝ) * x^2 - 1 + Real.log 2 - Real.log (2 * x - 3)) ↔
  m ≥ 1 + Real.log 2 :=
sorry

end max_min_h_range_m_l480_480212


namespace purely_imaginary_complex_number_has_a_equal_neg2_l480_480646

theorem purely_imaginary_complex_number_has_a_equal_neg2 (a : ℝ) : 
  (∀ (a : ℝ), (a^2 - 4) + (a - 2) * complex.I = (0 : ℂ) → a = -2) :=
begin
  sorry
end

end purely_imaginary_complex_number_has_a_equal_neg2_l480_480646


namespace plane_speed_in_still_air_l480_480914

-- Definitions, based on conditions
def distance_with_wind : ℕ := 400
def distance_against_wind : ℕ := 320
def wind_speed : ℕ := 20

-- The main statement to prove
theorem plane_speed_in_still_air : 
  ∃ p : ℕ, (400 / (p + wind_speed)) = (320 / (p - wind_speed)) → p = 180 := 
begin
  sorry
end

end plane_speed_in_still_air_l480_480914


namespace bridge_length_is_195_l480_480361

-- Define the problem context
variable (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ)

-- Specific conditions given in the problem
axiom train_length_def : train_length = 180  -- in meters
axiom train_speed_def : train_speed = 12.5  -- in meters per second after conversion
axiom crossing_time_def : crossing_time = 30  -- in seconds

-- Define the total distance covered by the train
def total_distance := train_speed * crossing_time

-- Define length of the bridge
def bridge_length := total_distance - train_length

-- The theorem statement to prove
theorem bridge_length_is_195 : train_length = 180 → train_speed = 12.5 → crossing_time = 30 → bridge_length train_length train_speed crossing_time = 195 :=
by
  intros h1 h2 h3
  unfold bridge_length total_distance
  rw [h1, h2, h3]
  exact rfl

end bridge_length_is_195_l480_480361


namespace mean_value_log2_on_2_8_l480_480749

-- Define the domain
def D := set.Icc 2 8

-- Define the function f(x) = log2(x)
def f (x : ℝ) : ℝ := real.log x / real.log 2

-- Definition of "mean value"
def has_mean_value (f : ℝ → ℝ) (C : ℝ) (D : set ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃! x₂ ∈ D, (f x₁ + f x₂) / 2 = C

-- Theorem to prove
theorem mean_value_log2_on_2_8 : has_mean_value f 2 D :=
sorry

end mean_value_log2_on_2_8_l480_480749


namespace area_of_triangle_correct_l480_480272

noncomputable def area_of_triangle (a c cosB : ℝ) : ℝ :=
  let sinB := Math.sqrt (1 - cosB^2)
  (1 / 2) * a * c * sinB

theorem area_of_triangle_correct :
  area_of_triangle 2 5 (3 / 5) = 4 := by
  sorry

end area_of_triangle_correct_l480_480272


namespace distance_between_points_l480_480659

theorem distance_between_points (s : ℝ) (p : Fin 5 → (ℝ × ℝ))
    (h : ∀ (i : Fin 5), 0 ≤ p i.1 ∧ p i.1 ≤ s ∧ 0 ≤ p i.2 ∧ p i.2 ≤ s) :
    ∃ (i j : Fin 5), i ≠ j ∧ dist (p i) (p j) ≤ s / 2 :=
by
  sorry

end distance_between_points_l480_480659


namespace find_m_l480_480297

noncomputable def m : ℕ :=
  let S := {d : ℕ | d ∣ 15^8 ∧ d > 0}
  let total_ways := 9^6
  let strictly_increasing_ways := (Nat.choose 9 3) * (Nat.choose 10 3)
  let probability := strictly_increasing_ways / total_ways
  let gcd := Nat.gcd strictly_increasing_ways total_ways
  strictly_increasing_ways / gcd

theorem find_m : m = 112 :=
by
  sorry

end find_m_l480_480297


namespace problem_solution_l480_480972

def diamond (x y k : ℝ) : ℝ := x^2 - k * y

theorem problem_solution (h : ℝ) (k : ℝ) (hk : k = 3) : 
  diamond h (diamond h h k) k = -2 * h^2 + 9 * h :=
by
  rw [hk, diamond, diamond]
  sorry

end problem_solution_l480_480972


namespace part_a_part_b_part_c_part_d_l480_480064

namespace HProblem

-- Part (a)
theorem part_a :
  ∃ (a b c d e f g h i : ℕ),
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i ∧
    a ≠ d ∧ a ≠ g ∧ b ≠ e ∧ b ≠ h ∧ c ≠ f ∧ c ≠ i ∧
    g ≠ e ∧ h ≠ d ∧ i ≠ f ∧
    a + b + c = 29 ∧ d + e + f = 29 ∧ g + h + i = 29 ∧
    {a, b, c, d, e, f, g, h, i} = {3, 5, 7, 15, x₁, x₂, x₃, x₄, x₅}
:= sorry

-- Part (b)
theorem part_b (t : ℕ) :
  ∀ (a b c d e f g h i : ℕ),
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i ∧
    a ≠ d ∧ a ≠ g ∧ b ≠ e ∧ b ≠ h ∧ c ≠ f ∧ c ≠ i ∧
    g ≠ e ∧ h ≠ d ∧ i ≠ f ∧
    a + b + c = S ∧ d + e + f = S ∧ g + h + i = S ∧
    {a, b, c, d, e, f, g, h, i} = {x₁, x₂, x₃, x₄, x₅, x₆, x₇}) →
    t = 6
:= sorry

-- Part (c)
theorem part_c :
  ∀ (a b c d e f g h i : ℕ),
    (a < c) →
    a ∈ {1, 2, 3} 
:= sorry

-- Part (d)
theorem part_d :
  ∀ (k n : ℕ),
    4 ≤ k ∧ k ≤ 18 ∧ 4 ≤ n ∧ n ≤ 18 ∧ k ≠ n ∧
    ∃ (a b c d e f g h i : ℕ),
      a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i ∧
      a ≠ d ∧ a ≠ g ∧ b ≠ e ∧ b ≠ h ∧ c ≠ f ∧ c ≠ i ∧
      g ≠ e ∧ h ≠ d ∧ i ≠ f ∧
      a + b + c = S ∧ d + e + f = S ∧ g + h + i = S ∧
      {a, b, c, d, e, f, g, h, i} = {k, n, x₁, x₂, x₃, x₄, x₅, x₆, x₇} →
    k ∈ {16, 17}
:= sorry

end HProblem

end part_a_part_b_part_c_part_d_l480_480064


namespace sum_of_first_fifteen_multiples_of_17_l480_480835

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480835


namespace lesser_fraction_l480_480380

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l480_480380


namespace lean_proof_problem_l480_480224

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) := a * x - y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := x + a * y + 1 = 0

-- Define the points A and B
def point_A := (0 : ℝ, 1 : ℝ)
def point_B := (-1 : ℝ, 0 : ℝ)

-- Define the propositions to prove
def perp (a : ℝ) := ∀ x y : ℝ, line1 a x y ∧ line2 a x y → (a * 1 - 1 * a = 0)
def passes_through_A (a : ℝ) := line1 a 0 1
def passes_through_B (a : ℝ) := line2 a (-1) 0
def trajectory (x y : ℝ) := x^2 + x + y^2 - y = 0 ∧ x ≠ 0 ∧ y ≠ 0

-- The final theorem statement
theorem lean_proof_problem :
  (∀ a : ℝ, perp a) ∧
  (∀ a : ℝ, passes_through_A a) ∧
  (∀ a : ℝ, passes_through_B a) ∧
  (∃ x y : ℝ, trajectory x y) :=
by
  sorry

end lean_proof_problem_l480_480224


namespace parallel_vectors_l480_480226

variable (λ : ℝ)

def a : ℝ × ℝ := (2, 6)
def b : ℝ × ℝ := (-3, λ)

theorem parallel_vectors (λ : ℝ) : λ = -9 ↔ (2 * λ + 18 = 0) := by
  sorry

end parallel_vectors_l480_480226


namespace triangle_area_l480_480371

-- Define the variables and parameters
variables {A B C a b c : ℝ}

-- Conditions from the problem
axiom sides_opposite : a = side_opposite_angle A
axiom sides_opposite' : b = side_opposite_angle B
axiom sides_opposite'' : c = side_opposite_angle C
axiom trig_relation : b * sin C + c * sin B = 4 * a * sin B * sin C
axiom pythagorean : b^2 + c^2 - a^2 = 8

-- Theorem statement
theorem triangle_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : b * sin C + c * sin B = 4 * a * sin B * sin C) 
  (h2 : b^2 + c^2 - a^2 = 8) : 
  (1/2) * a * b * sin C = sqrt 3 :=
sorry

end triangle_area_l480_480371


namespace num_rooms_l480_480942

theorem num_rooms (r1 r2 w1 w2 p w_paint : ℕ) (h_r1 : r1 = 5) (h_r2 : r2 = 4) (h_w1 : w1 = 4) (h_w2 : w2 = 5)
    (h_p : p = 5) (h_w_paint : w_paint = 8) (h_total_walls_family : p * w_paint = (r1 * w1 + r2 * w2)) :
    (r1 + r2 = 9) :=
by
  sorry

end num_rooms_l480_480942


namespace completing_square_result_l480_480030

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l480_480030


namespace problem_1_problem_2_problem_3_l480_480075

def distances : List ℤ := [+10, -3, +4, -2, -8, +13, -2, -11, +7, +5]

def final_position : ℤ := distances.sum

def maximum_distance : ℤ := distances.scanl (+) 0 |> List.map abs |> List.maximum?.getOrElse 0

def total_fuel (distances : List ℤ) (fuel_per_km : ℚ := 0.2) : ℚ :=
  distances.map Int.natAbs |> List.sum * fuel_per_km

theorem problem_1 : final_position = 13 :=
by
  sorry

theorem problem_2 : maximum_distance = 14 :=
by
  sorry

theorem problem_3 : total_fuel distances = 12.8 :=
by
  sorry

end problem_1_problem_2_problem_3_l480_480075


namespace find_numbers_l480_480439

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480439


namespace quadrilateral_OA_OC_plus_OB_OD_eq_sqrt_abcd_l480_480894

noncomputable def OA (A B C D O : Point) := distance A O
noncomputable def OB (A B C D O : Point) := distance B O
noncomputable def OC (A B C D O : Point) := distance C O
noncomputable def OD (A B C D O : Point) := distance D O

variables {A B C D O : Point}
variables {a b c d : ℝ}

def quadrilateral_circumscribes_circle :=
  ∃ (O : Point), OA A B C D O + OC A B C D O = b + d ∧ OB A B C D O + OD A B C D O = a + c 

theorem quadrilateral_OA_OC_plus_OB_OD_eq_sqrt_abcd 
  (h: quadrilateral_circumscribes_circle) :
  OA A B C D O * OC A B C D O + OB A B C D O * OD A B C D O = real.sqrt (a * b * c * d) :=
sorry

end quadrilateral_OA_OC_plus_OB_OD_eq_sqrt_abcd_l480_480894


namespace theta_values_satisfy_eq_l480_480630

noncomputable def numSolutions : ℝ := 4 

theorem theta_values_satisfy_eq:
    (∃ n : ℕ, n = 4 ∧
      ∀ θ : ℝ, 
        0 < θ ∧ θ ≤ 2 * Real.pi →
        2 - 4 * Real.cos θ + 3 * Real.sin (2 * θ) = 0) :=
by
  let number_of_solutions := numSolutions
  have h : number_of_solutions = 4, from rfl
  sorry

end theta_values_satisfy_eq_l480_480630


namespace three_digit_numbers_without_579_l480_480231

def count_valid_digits (exclusions : List Nat) (range : List Nat) : Nat :=
  (range.filter (λ n => n ∉ exclusions)).length

def count_valid_three_digit_numbers : Nat :=
  let hundreds := count_valid_digits [5, 7, 9] [1, 2, 3, 4, 6, 8]
  let tens_units := count_valid_digits [5, 7, 9] [0, 1, 2, 3, 4, 6, 8]
  hundreds * tens_units * tens_units

theorem three_digit_numbers_without_579 : 
  count_valid_three_digit_numbers = 294 :=
by
  unfold count_valid_three_digit_numbers
  /- 
  Here you can add intermediate steps if necessary, 
  but for now we assert the final goal since this is 
  just the problem statement with the proof omitted.
  -/
  sorry

end three_digit_numbers_without_579_l480_480231


namespace curve_E_equation_and_slope_range_l480_480607

-- Given conditions
def curve_C (x y : ℝ) : Prop := y^2 = 4*x ∧ x > 0
def foci_1 : ℝ × ℝ := (-1, 0)
def foci_2 : ℝ × ℝ := (1, 0)
def intersection (P : ℝ × ℝ) (E : ℝ × ℝ → Prop) : Prop :=
  curve_C P.1 P.2 ∧ E P

-- Ellipse definition for curve E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Point P and distance constraints
def point_P (P : ℝ × ℝ) : Prop :=
  P.1 = 2 / 3 ∧ P.2 = 2 * Real.sqrt (6) / 3 ∧
  P.dist foci_2 = 5 / 3

-- The proof statement
theorem curve_E_equation_and_slope_range :
  ∃ (a b : ℝ)
  (P : ℝ × ℝ) (E : ℝ × ℝ → Prop),
  (ellipse a b P.1 P.2 ∧ point_P P) ∧ E P ∧
  ∀ (k : ℝ),
  (∃ (A B : ℝ × ℝ),
    intersection A E ∧ intersection B E ∧
    ∃ (M : ℝ × ℝ),
      M.1 = (A.1 + B.1) / 2 ∧
      M.2 = (A.2 + B.2) / 2 ∧
      curve_C M.1 M.2) →
  ( - Real.sqrt (6) / 8 < k ∧ k < Real.sqrt (6) / 8 ) :=
sorry

end curve_E_equation_and_slope_range_l480_480607


namespace route_Y_is_quicker_l480_480711

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

def route_X_distance : ℝ := 8
def route_X_speed : ℝ := 40

def route_Y_total_distance : ℝ := 7
def route_Y_construction_distance : ℝ := 1
def route_Y_construction_speed : ℝ := 20
def route_Y_regular_speed_distance : ℝ := 6
def route_Y_regular_speed : ℝ := 50

noncomputable def route_X_time : ℝ :=
  route_time route_X_distance route_X_speed * 60  -- converting to minutes

noncomputable def route_Y_time : ℝ :=
  (route_time route_Y_regular_speed_distance route_Y_regular_speed +
  route_time route_Y_construction_distance route_Y_construction_speed) * 60 -- converting to minutes

theorem route_Y_is_quicker : route_X_time - route_Y_time = 1.8 :=
  by
    sorry

end route_Y_is_quicker_l480_480711


namespace imaginary_part_of_z_l480_480191

def imaginary_unit := Complex.i
def z (x : Complex) := (1 + imaginary_unit) * x = imaginary_unit

theorem imaginary_part_of_z : ∀ x : Complex, z x → Complex.im x = 1 / 2 :=
by
  intro x
  intro hx
  sorry

end imaginary_part_of_z_l480_480191


namespace geometric_sequence_general_term_formula_no_arithmetic_sequence_l480_480176

-- Assume we have a sequence {a_n} and its sum of the first n terms S_n where S_n = 2a_n - n (for n ∈ ℕ*)
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

-- Condition 1: S_n = 2a_n - n
axiom Sn_condition (n : ℕ) (h : n > 0) : S_n n = 2 * a_n n - n

-- 1. Prove that the sequence {a_n + 1} is a geometric sequence with first term and common ratio equal to 2
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r : ℕ, r = 2 ∧ ∀ m : ℕ, a_n (m + 1) + 1 = r * (a_n m + 1) :=
by
  sorry

-- 2. Prove the general term formula an = 2^n - 1
theorem general_term_formula (n : ℕ) (h : n > 0) : a_n n = 2^n - 1 :=
by
  sorry

-- 3. Prove that there do not exist three consecutive terms in {a_n} that form an arithmetic sequence
theorem no_arithmetic_sequence (n k : ℕ) (h : n > 0 ∧ k > 0 ∧ k + 2 < n) : ¬(a_n k + a_n (k + 2) = 2 * a_n (k + 1)) :=
by
  sorry

end geometric_sequence_general_term_formula_no_arithmetic_sequence_l480_480176


namespace correct_value_two_decimal_places_l480_480417

theorem correct_value_two_decimal_places (x : ℝ) 
  (h1 : 8 * x + 8 = 56) : 
  (x / 8) + 7 = 7.75 :=
sorry

end correct_value_two_decimal_places_l480_480417


namespace sum_largest_and_smallest_angles_l480_480619

theorem sum_largest_and_smallest_angles (a b c : ℝ) (h : a / 5 = b / 7 ∧ b / 7 = c / 8) : 
  ∃ θ : ℝ, (cos θ = 1 / 2) ∧ (180° - θ = 120°) :=
begin
  sorry
end

end sum_largest_and_smallest_angles_l480_480619


namespace matrix_B_zero_l480_480634

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)
variable (E : Matrix (Fin n) (Fin n) ℝ := 1)

theorem matrix_B_zero
  (h1 : A * B = B)
  (h2 : det (A - E) ≠ 0) :
  B = 0 := 
by
  sorry

end matrix_B_zero_l480_480634


namespace problem_solution_l480_480591

noncomputable def p (a : ℝ) : Prop := (0 < a) ∧ (a < 1)
noncomputable def q (a : ℝ) : Prop := (a > (5 / 2)) ∨ (a < (1 / 2))

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) :
    (a ≥ 1 / 2 ∧ a < 1) ∨ (a > 5 / 2) :=
begin
    sorry
end

end problem_solution_l480_480591


namespace total_cost_for_group_l480_480097

-- Definitions based on conditions
def number_of_people : Nat := 13
def number_of_kids : Nat := 9
def cost_per_adult_meal : Nat := 7
def kids_eat_free : Bool := true -- This represents that kids' meals are free

-- Required proof statement
theorem total_cost_for_group :
  let number_of_adults := number_of_people - number_of_kids in
  let total_cost := number_of_adults * cost_per_adult_meal in
  total_cost = 28 :=
by
  sorry

end total_cost_for_group_l480_480097


namespace exists_rational_polynomial_with_rational_critical_points_l480_480717

theorem exists_rational_polynomial_with_rational_critical_points :
  ∃ (b c d : ℚ), 
    ∀ (x : ℚ), 
      (polynomial.eval x (polynomial.C 1 * polynomial.X ^ 3 + polynomial.C b * polynomial.X ^ 2 + polynomial.C c * polynomial.X + polynomial.C d) = 0 → 
       (∃ (y : ℚ), polynomial.derivative (polynomial.C 1 * polynomial.X ^ 3 + polynomial.C b * polynomial.X ^ 2 + polynomial.C c * polynomial.X + polynomial.C d) = y)) := 
sorry

end exists_rational_polynomial_with_rational_critical_points_l480_480717


namespace permutation_six_two_l480_480059

-- Definition for permutation
def permutation (n k : ℕ) : ℕ := n * (n - 1)

-- Theorem stating that the permutation of 6 taken 2 at a time is 30
theorem permutation_six_two : permutation 6 2 = 30 :=
by
  -- proof will be filled here
  sorry

end permutation_six_two_l480_480059


namespace problem_statement_l480_480182

noncomputable def ellipse_equation (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

def vertex_area (a b : ℝ) : ℝ :=
  (1 / 2) * a * 2 * b

theorem problem_statement: 
  ∀ (a b c : ℝ), a > b ∧ b > 0 ∧ eccentricity a c = (Real.sqrt 5) / 3 ∧ vertex_area a b = 6 →
  (ellipse_equation 3 2 = λ x y, (x^2 / 9) + (y^2 / 4) = 1) ∧
  (∀ P : Point, P ∈ ellipse 3 2 ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C →
    (line_MN P passes through Point.mk 3 2)) :=
sorry

end problem_statement_l480_480182


namespace no_solution_exists_l480_480728

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l480_480728


namespace sum_of_fractions_limit_one_l480_480714

theorem sum_of_fractions_limit_one :
  (∑' (a : ℕ), ∑' (b : ℕ), (1 : ℝ) / ((a + 1) : ℝ) ^ (b + 1)) = 1 := 
sorry

end sum_of_fractions_limit_one_l480_480714


namespace eval_expr_l480_480242

theorem eval_expr (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a / b)^(b - a) :=
sorry

end eval_expr_l480_480242


namespace rowing_time_l480_480908

-- Define given conditions
def Vm : ℝ := 6 -- Man's speed in still water in km/h
def Vr : ℝ := 3 -- River speed in km/h
def total_distance : ℝ := 4.5 -- Total distance traveled in km

-- The problem is to prove the total time taken to row the distance and back

theorem rowing_time (Vm Vr total_distance : ℝ) (hVm : Vm = 6) (hVr : Vr = 3) (h_total_distance : total_distance = 4.5) :
  let D := total_distance / 2 in 
  let T_up := D / (Vm - Vr) in
  let T_down := D / (Vm + Vr) in
  T_up + T_down = 1 := 
by
  sorry

end rowing_time_l480_480908


namespace arithmetic_operations_between_12345_l480_480273

theorem arithmetic_operations_between_12345 :
  ∃ (a b c d : ℤ), 
    a = 1 + 2 - 3 - 4 + 5 ∨ 
    a = 1 - 2 + 3 + 4 - 5 ∧ 
    a = 1 :=
begin
  use 1,
  left,
  refl,
sorry -- Proof skipped
end

end arithmetic_operations_between_12345_l480_480273


namespace find_numbers_l480_480456

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l480_480456


namespace perpendicular_tangent_line_l480_480145

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end perpendicular_tangent_line_l480_480145


namespace student_desserts_l480_480770

theorem student_desserts (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) (equal_distribution : students ≠ 0) :
  mini_cupcakes = 14 → donut_holes = 12 → students = 13 → (mini_cupcakes + donut_holes) / students = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact div_eq_of_eq_mul_right (by norm_num : (13 : ℕ) ≠ 0) (by norm_num : 26 = 2 * 13)
  sorry

end student_desserts_l480_480770


namespace jordan_more_novels_than_maxime_l480_480130

theorem jordan_more_novels_than_maxime :
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  jordan_novels - maxime_novels = 51 :=
by
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  sorry

end jordan_more_novels_than_maxime_l480_480130


namespace incorrect_parallel_lines_l480_480686

variables (m n : Type) [line m] [line n] (α β : Type) [plane α] [plane β]
variables [different_lines m n] [different_planes α β]
variables [parallel m α] [perpendicular n β] [perpendicular α β]

theorem incorrect_parallel_lines : ¬ (parallel m n) :=
sorry

end incorrect_parallel_lines_l480_480686


namespace math_club_members_count_l480_480351

theorem math_club_members_count 
    (n_books : ℕ) 
    (n_borrow_each_member : ℕ) 
    (n_borrow_each_book : ℕ) 
    (total_borrow_count_books : n_books * n_borrow_each_book = 36) 
    (total_borrow_count_members : 2 * x = 36) 
    : x = 18 := 
by
  sorry

end math_club_members_count_l480_480351


namespace find_numbers_sum_eq_S_product_eq_P_l480_480434

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l480_480434


namespace original_employee_salary_l480_480134

-- Given conditions
def emily_original_salary : ℝ := 1000000
def emily_new_salary : ℝ := 850000
def number_of_employees : ℕ := 10
def employee_new_salary : ℝ := 35000

-- Prove the original salary of each employee
theorem original_employee_salary :
  (emily_original_salary - emily_new_salary) / number_of_employees = employee_new_salary - 20000 := 
by
  sorry

end original_employee_salary_l480_480134


namespace no_solution_for_inequalities_l480_480727

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l480_480727


namespace gcd_840_1764_l480_480358

-- Define the numbers according to the conditions
def a : ℕ := 1764
def b : ℕ := 840

-- The goal is to prove that the GCD of a and b is 84
theorem gcd_840_1764 : Nat.gcd a b = 84 := 
by
  -- The proof steps would normally go here
  sorry

end gcd_840_1764_l480_480358


namespace george_earnings_after_deductions_l480_480164

noncomputable def george_total_earnings : ℕ := 35 + 12 + 20 + 21

noncomputable def tax_deduction (total_earnings : ℕ) : ℚ := total_earnings * 0.10

noncomputable def uniform_fee : ℚ := 15

noncomputable def final_earnings (total_earnings : ℕ) (tax_deduction : ℚ) (uniform_fee : ℚ) : ℚ :=
  total_earnings - tax_deduction - uniform_fee

theorem george_earnings_after_deductions : 
  final_earnings george_total_earnings (tax_deduction george_total_earnings) uniform_fee = 64.2 := 
  by
  sorry

end george_earnings_after_deductions_l480_480164


namespace intersection_A_B_l480_480698

open Set

-- Definitions of the sets A and B as described in the problem
def setA : Set ℝ := {x | abs x ≤ 1}
def setB : Set ℝ := {x | x ≤ 0}

-- Lean theorem statement to prove the intersection of A and B
theorem intersection_A_B :
  setA ∩ setB = {x | -1 ≤ x ∧ x ≤ 0} := by
  sorry

end intersection_A_B_l480_480698


namespace arrangements_count_l480_480785

-- Definition of the conditions
def men : ℕ := 5
def women : ℕ := 2
def total_people : ℕ := men + women
-- The total number of arrangements under given conditions
def arrangements (A_at_edge : Bool) (women_together: Bool): ℕ :=
  if A_at_edge ∧ women_together then 2 * 5! * 2! else 0

-- The theorem to be proved
theorem arrangements_count : arrangements true true = 480 :=
by
  -- This is where the proof would go
  sorry

end arrangements_count_l480_480785


namespace ellie_needs_25ml_of_oil_l480_480133

theorem ellie_needs_25ml_of_oil 
  (oil_per_wheel : ℕ) 
  (number_of_wheels : ℕ) 
  (other_parts_oil : ℕ) 
  (total_oil : ℕ)
  (h1 : oil_per_wheel = 10)
  (h2 : number_of_wheels = 2)
  (h3 : other_parts_oil = 5)
  (h4 : total_oil = oil_per_wheel * number_of_wheels + other_parts_oil) : 
  total_oil = 25 :=
  sorry

end ellie_needs_25ml_of_oil_l480_480133


namespace problem1_problem2_l480_480878

-- Using the conditions from a) and the correct answers from b):
-- 1. Given an angle α with a point P(-4,3) on its terminal side

theorem problem1 (α : ℝ) (x y r : ℝ) (h₁ : x = -4) (h₂ : y = 3) (h₃ : r = 5) 
  (hx : r = Real.sqrt (x^2 + y^2)) 
  (hsin : Real.sin α = y / r) 
  (hcos : Real.cos α = x / r) 
  : (Real.cos (π / 2 + α) * Real.sin (-π - α)) / (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3 / 4 :=
by sorry

-- 2. Let k be an integer
theorem problem2 (α : ℝ) (k : ℤ)
  : (Real.sin (k * π - α) * Real.cos ((k + 1) * π - α)) / (Real.sin ((k - 1) * π + α) * Real.cos (k * π + α)) = -1 :=
by sorry

end problem1_problem2_l480_480878


namespace unique_solution_of_quadratic_eqn_l480_480645
-- Import the required libraries

-- Define the proof problem
theorem unique_solution_of_quadratic_eqn (a : ℝ) (A : set ℝ) :
  (A = { x : ℝ | a * x^2 + a * x + 1 = 0 }) →
  (∃! x : ℝ, x ∈ A) → a = 4 :=
by
  sorry

end unique_solution_of_quadratic_eqn_l480_480645


namespace best_discount_option_l480_480722

-- Define the original price
def original_price : ℝ := 100

-- Define the discount functions for each option
def option_A : ℝ := original_price * (1 - 0.20)
def option_B : ℝ := (original_price * (1 - 0.10)) * (1 - 0.10)
def option_C : ℝ := (original_price * (1 - 0.15)) * (1 - 0.05)
def option_D : ℝ := (original_price * (1 - 0.05)) * (1 - 0.15)

-- Define the theorem stating that option A gives the best price
theorem best_discount_option : option_A ≤ option_B ∧ option_A ≤ option_C ∧ option_A ≤ option_D :=
by {
  sorry
}

end best_discount_option_l480_480722


namespace find_numbers_l480_480446

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l480_480446


namespace exceeds_alpha_beta_l480_480635

noncomputable def condition (α β p q : ℝ) : Prop :=
  q < 50 ∧ α > 0 ∧ β > 0 ∧ p > 0 ∧ q > 0

theorem exceeds_alpha_beta (α β p q : ℝ) (h : condition α β p q) :
  (1 + p / 100) * (1 - q / 100) > 1 → p > 100 * q / (100 - q) := by
  sorry

end exceeds_alpha_beta_l480_480635


namespace dice_sum_10_with_5_prob_l480_480798

open ProbabilityTheory

def six_sided_die := {1, 2, 3, 4, 5, 6}

noncomputable def prob_sum_10_at_least_one_5 : ℚ :=
  Pr (λ (rolls : Fin 3 → ℕ), 
    (∃ i, rolls i = 5) ∧ (rolls 0 + rolls 1 + rolls 2 = 10)) six_sided_die

theorem dice_sum_10_with_5_prob :
  prob_sum_10_at_least_one_5 = 1 / 18 :=
sorry

end dice_sum_10_with_5_prob_l480_480798


namespace lesser_fraction_l480_480382

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l480_480382


namespace pyramid_volume_proof_l480_480920

def pyramid_volume_theorem : Prop :=
  let a := 5 in
  let b := 7 in
  let edge_length := 15 in
  let base_area := a * b in
  let diagonal := Real.sqrt (a^2 + b^2) in
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2) in
  let volume := (1 / 3) * base_area * height in
  volume = (35 * Real.sqrt 188) / 3

theorem pyramid_volume_proof : pyramid_volume_theorem := by
  sorry

end pyramid_volume_proof_l480_480920


namespace households_using_neither_brands_l480_480077

def total_households : Nat := 240
def only_brand_A_households : Nat := 60
def both_brands_households : Nat := 25
def ratio_B_to_both : Nat := 3
def only_brand_B_households : Nat := ratio_B_to_both * both_brands_households
def either_brand_households : Nat := only_brand_A_households + only_brand_B_households + both_brands_households
def neither_brand_households : Nat := total_households - either_brand_households

theorem households_using_neither_brands :
  neither_brand_households = 80 :=
by
  -- Proof can be filled out here
  sorry

end households_using_neither_brands_l480_480077


namespace smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l480_480811

def is_not_prime (n : ℕ) : Prop := ¬ Prime n

def is_not_square (n : ℕ) : Prop := ∀ m : ℕ, m * m ≠ n

def no_prime_factor_less_than_50 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ≥ 50

theorem smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50 :
  (∃ n : ℕ, 0 < n ∧ is_not_prime n ∧ is_not_square n ∧ no_prime_factor_less_than_50 n ∧
  (∀ m : ℕ, 0 < m ∧ is_not_prime m ∧ is_not_square m ∧ no_prime_factor_less_than_50 m → n ≤ m)) →
  ∃ n : ℕ, n = 3127 :=
by {
  sorry
}

end smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l480_480811


namespace quadratic_minimum_value_interval_l480_480203

theorem quadratic_minimum_value_interval (k : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x < 2 → (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (2*k^2 + 2*k - 1)) → (0 ≤ k ∧ k < 1) :=
by {
  sorry
}

end quadratic_minimum_value_interval_l480_480203


namespace subset_solution_l480_480225

theorem subset_solution (a : ℤ):
  let A := {1, 4, a}
  let B := {1, a^2}
  (B ⊂ A) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
by
  -- Definitions of the sets A and B
  let A := {1, 4, a}
  let B := {1, a^2}
  sorry

end subset_solution_l480_480225


namespace decrease_in_average_l480_480070

-- Definitions for the given conditions
def A1 : ℝ := 12.4
def W1 : ℕ := 85
def runs_last_match : ℕ := 26
def wickets_last_match : ℕ := 5
def total_runs_before_last_match : ℝ := A1 * W1
def W2 : ℕ := W1 + wickets_last_match
def total_runs_after_last_match : ℝ := total_runs_before_last_match + runs_last_match
def A2 : ℝ := total_runs_after_last_match / W2

-- The theorem to prove
theorem decrease_in_average : A1 - A2 = 0.4 := by
    sorry

end decrease_in_average_l480_480070


namespace wind_velocity_correct_l480_480758

-- Definitions for the given conditions
def pressure (k A V : ℝ) : ℝ := k * A * V^2

noncomputable def find_k (A V P : ℝ) : ℝ := P / (A * V^2)

-- Initial conditions
def A₁ : ℝ := 2
def V₁ : ℝ := 20
def P₁ : ℝ := 5

-- New conditions
def A₂ : ℝ := 4
def P₂ : ℝ := 20

-- Correct answer
def correct_V₂ : ℝ := 20 * real.sqrt 2

-- Proof statement
theorem wind_velocity_correct :
  let k := find_k A₁ V₁ P₁ in
  let V₂ := real.sqrt (P₂ / (k * A₂)) in
  V₂ = correct_V₂ := 
by {
  sorry -- Proof is omitted
}

end wind_velocity_correct_l480_480758


namespace find_percentage_l480_480078

theorem find_percentage (P N : ℕ) (h₁ : N = 125) (h₂ : N = (P * N / 100) + 105) : P = 16 :=
by
  sorry

end find_percentage_l480_480078


namespace compatible_polygons_l480_480000

open_locale classical

def is_compatible (P Q : Type) [polygon P] [polygon Q] : Prop :=
  ∃ (k : ℕ) (k_pos : 0 < k), 
  ∃ (fP : P → list (polygon)) (fQ : Q → list (polygon)),
  (∀ p ∈ fP P, ∃ q ∈ fQ Q, congruent p q) ∧ 
  (∀ q ∈ fQ Q, ∃ p ∈ fP P, congruent q p)

theorem compatible_polygons (m n : ℕ) (hm : m ≥ 4) (hn : n ≥ 4) 
  (hm_even : even m) (hn_even : even n) :
  ∃ (P Q : Type) [polygon P] [polygon Q], 
  (sides P = m) ∧ (sides Q = n) ∧ is_compatible P Q :=
sorry

end compatible_polygons_l480_480000


namespace largest_x_value_l480_480147

theorem largest_x_value :
  ∃ x, (∀ y, (frac (14 * y ^ 3 - 40 * y ^ 2 + 20 * y - 4) (4 * y - 3) + 6 * y = 8 * y - 3) → y ≤ x) ∧ x = 1.5 := 
sorry

end largest_x_value_l480_480147


namespace symmetric_shift_arctan_l480_480700

-- Define the original function
def f (x: ℝ) := arctan x

-- Define the shifted function
def C (x: ℝ) := f (x - 2)

-- Define the symmetric function
def C' (x: ℝ) := - C (-x)

-- State the theorem
theorem symmetric_shift_arctan :
  C' x = arctan (x + 2) :=
sorry

end symmetric_shift_arctan_l480_480700


namespace completing_square_result_l480_480027

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l480_480027


namespace original_cube_volume_l480_480398

theorem original_cube_volume (a : ℕ) (h1 : V_cube = a^3) 
    (h2 : V_new = (a+1) * (a+1) * (a-2)) 
    (h3 : V_cube = V_new + 27) : a = 5 := 
begin
  sorry
end

end original_cube_volume_l480_480398


namespace monotonic_decreasing_interval_l480_480247

noncomputable def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y ∈ Icc a b, x < y → f x > f y

theorem monotonic_decreasing_interval (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2)
  (hpt : (λ x, 2 * sin (2 * x + ϕ)) 0 = sqrt 3) :
  is_decreasing_on (λ x, 2 * sin (2 * x + ϕ)) (π / 12) (7 * π / 12) :=
begin
  sorry  -- Proof to be completed
end

end monotonic_decreasing_interval_l480_480247


namespace total_nice_people_in_crowd_l480_480509

def number_nice_people_bar (num_bar : ℕ) : ℕ := 
  num_bar -- All Barrys are nice

def number_nice_people_kevin (num_kev : ℕ) : ℕ := 
  num_kev / 2 -- Half of Kevins are nice

def number_nice_people_julie (num_jul : ℕ) : ℕ := 
  num_jul * 3 / 4 -- Three fourths of Julies are nice

def number_nice_people_joe (num_joe : ℕ) : ℕ := 
  num_joe / 10 -- Ten percent of Joes are nice

theorem total_nice_people_in_crowd (num_bar num_kev num_jul num_joe : ℕ)
    (h_bar : num_bar = 24)
    (h_kev : num_kev = 20)
    (h_jul : num_jul = 80)
    (h_joe : num_joe = 50) : 
  number_nice_people_bar num_bar + number_nice_people_kevin num_kev + number_nice_people_julie num_jul + number_nice_people_joe num_joe = 99 := 
by
  rw [h_bar, h_kev, h_jul, h_joe]
  have := (number_nice_people_bar 24 + number_nice_people_kevin 20 + number_nice_people_julie 80 + number_nice_people_joe 50)
  show 24 + 10 + 60 + 5 = 99
  sorry

end total_nice_people_in_crowd_l480_480509


namespace miles_left_l480_480315

theorem miles_left (d_total d_covered d_left : ℕ) 
  (h₁ : d_total = 78) 
  (h₂ : d_covered = 32) 
  (h₃ : d_left = d_total - d_covered):
  d_left = 46 := 
by {
  sorry 
}

end miles_left_l480_480315


namespace edge_length_is_correct_l480_480776

-- Define the given conditions
def volume_material : ℕ := 12 * 18 * 6
def edge_length : ℕ := 3
def number_cubes : ℕ := 48
def volume_cube (e : ℕ) : ℕ := e * e * e

-- Problem statement in Lean:
theorem edge_length_is_correct : volume_material = number_cubes * volume_cube edge_length → edge_length = 3 :=
by
  sorry

end edge_length_is_correct_l480_480776


namespace find_b_l480_480771

def collinear_points_line (b : ℚ) : Prop :=
  let p1 := (5 : ℚ, -7 : ℚ)
  let p2 := (b + 4, 5 : ℚ)
  let p3 := (-3 * b + 6, 3 : ℚ)
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem find_b (b : ℚ) (h : collinear_points_line b) : b = 11 / 23 :=
  sorry

end find_b_l480_480771


namespace final_payment_calculation_l480_480897

noncomputable def cost := 750
noncomputable def down_payment := 300
noncomputable def num_monthly_payments := 9
noncomputable def monthly_payment := 57
noncomputable def interest_rate := 18.666666666666668 / 100

def borrowed := cost - down_payment
def total_paid_over_9_months := monthly_payment * num_monthly_payments
def interest_paid := interest_rate * borrowed
def total_amount_to_be_paid := borrowed + interest_paid
def final_payment := total_amount_to_be_paid - total_paid_over_9_months

theorem final_payment_calculation :
  final_payment = 21 :=
  by
    sorry

end final_payment_calculation_l480_480897


namespace brendan_remaining_money_l480_480956

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l480_480956


namespace complex_fraction_l480_480104

theorem complex_fraction (h : (1 : ℂ) - I = 1 - (I : ℂ)) :
  ((1 - I) * (1 - (2 * I))) / (1 + I) = -2 - I := 
by
  sorry

end complex_fraction_l480_480104


namespace muffin_banana_costs_l480_480940

variable (m b : ℕ) -- Using natural numbers for non-negativity

theorem muffin_banana_costs (h : 3 * (3 * m + 5 * b) = 4 * m + 10 * b) : m = b :=
by
  sorry

end muffin_banana_costs_l480_480940


namespace incircle_radii_equal_l480_480045

-- Definitions for sides of the triangles
def sides_triangle1 : (ℕ × ℕ × ℕ) := (17, 25, 26)
def sides_triangle2 : (ℕ × ℕ × ℕ) := (17, 25, 28)

-- Function to compute the semi-perimeter
def semi_perimeter (a b c : ℕ) : ℝ :=
  (a + b + c) / 2

-- Function to compute the area using Heron's formula
noncomputable def area (a b c : ℕ) (p : ℝ) : ℝ :=
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

-- Function to compute the radius of the incircle
noncomputable def incircle_radius (a b c : ℕ) : ℝ :=
  let p := semi_perimeter a b c
  area a b c p / p

-- Statement of the problem
theorem incircle_radii_equal :
  incircle_radius 17 25 26 = 6 ∧ incircle_radius 17 25 28 = 6 := by
  sorry

end incircle_radii_equal_l480_480045


namespace guessing_game_l480_480950

-- Define the conditions
def number : ℕ := 33
def result : ℕ := 2 * 51 - 3

-- Define the factor (to be proven)
def factor (n r : ℕ) : ℕ := r / n

-- The theorem to be proven
theorem guessing_game (n r : ℕ) (h1 : n = 33) (h2 : r = 2 * 51 - 3) : 
  factor n r = 3 := by
  -- Placeholder for the actual proof
  sorry

end guessing_game_l480_480950


namespace cyclic_quadrilateral_perpendicular_l480_480997

noncomputable def cyclic_quadrilateral_proof_statement (A B C D M N : Point) (Γ : Circle) : Prop :=
  is_diameter Γ A C → 
  cyclic_quadrilateral A B C D → 
  AD_intersects_BC_at_M A D B C M → 
  tangents_intersect_at_N Γ B D N → 
  perpendicular AC MN

-- We will need to assume point definitions and relevant helper definitions/properties
variables {A B C D M N : Point} {Γ : Circle}

theorem cyclic_quadrilateral_perpendicular
  (h1 : is_diameter Γ A C)
  (h2 : cyclic_quadrilateral A B C D)
  (h3 : AD_intersects_BC_at_M A D B C M)
  (h4 : tangents_intersect_at_N Γ B D N) :
  perpendicular AC MN :=
sorry

end cyclic_quadrilateral_perpendicular_l480_480997


namespace probability_x_equals_y_eq_1_over_21_l480_480510

noncomputable def probability_x_equals_y : ℝ :=
  let source_set := {p : ℝ × ℝ | ∃ k : ℤ, 
    (p.1 = - 2 * (k:ℝ) * π + p.2 - π/2 ∨ p.1 = 2 * (k:ℝ) * π + π - p.2  - π/2) ∧
    (-10 * π ≤ p.1 ∧ p.1 ≤ 10 * π ∧ -10 * π ≤ p.2 ∧ p.2 ≤ 10 * π)} in
  let x_equals_y_set := {x : ℝ | ∃ k : ℤ, x = -2 * (k:ℝ) * π + π/4 ∧ -10 * π ≤ x ∧ x ≤ 10 * π} in
  (x_equals_y_set.powerset.card : ℝ) / (source_set.powerset.card : ℝ)

-- Statement to be proved
theorem probability_x_equals_y_eq_1_over_21 : probability_x_equals_y = 1 / 21 := 
by sorry

end probability_x_equals_y_eq_1_over_21_l480_480510


namespace count_valid_sequences_correct_l480_480094

-- Define the conditions of the sequence
def is_valid_sequence (seq : Fin 40 → Fin 41) : Prop :=
  ∃ a20 a31 : Fin 41, 
    (∀ i < 20, seq i ≤ a20) ∧ 
    (seq 20 = a20) ∧ 
    (seq 30 = a31) ∧ 
    (a20 < a31) ∧ 
    (∀ j < 40, seq j ≤ a31) ∧
    (∀ i < 31, seq i ≤ seq 30)

-- The number of such sequences
def count_valid_sequences : ℕ :=
  Nat.choose 40 9 * Nat.factorial 29 * Nat.factorial 9

-- The theorem statement
theorem count_valid_sequences_correct :
  ∃ seqs : Fin 40 → Fin 41,
  is_valid_sequence seqs = count_valid_sequences := 
  sorry

end count_valid_sequences_correct_l480_480094


namespace number_of_tangent_circles_fixed_radius_l480_480648

-- Define the given circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the two externally tangent circles
def circle1 : Circle := ⟨(0, 0), 2⟩
def circle2 : Circle := ⟨(6, 0), 4⟩  -- Tangent and distance between centers is (2+4)

-- Define the problem statement as a theorem
theorem number_of_tangent_circles_fixed_radius :
  ∃ C1 C2 : Circle,
    C1.radius = 6 ∧ C2.radius = 6 ∧ 
    ∀ C : Circle, C.radius = 6 → (tangent_to_both C circle1) ∧ (tangent_to_both C circle2) :=
sorry

-- Define tangent_to_both function
def tangent_to_both (C C1 C2 : Circle) : Prop :=
  dist C.center C1.center = C1.radius + C.radius ∧
  dist C.center C2.center = C2.radius + C.radius

end number_of_tangent_circles_fixed_radius_l480_480648


namespace zero_in_M_l480_480221

theorem zero_in_M : 0 ∈ ({0, 1, 2} : set ℕ) :=
sorry

end zero_in_M_l480_480221


namespace minimum_value_is_9_l480_480592

noncomputable def min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : ℝ :=
  Inf {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ x = 1 / a + 4 / b}

theorem minimum_value_is_9 :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ min_value a b sorry sorry sorry = 9 :=
sorry

end minimum_value_is_9_l480_480592


namespace min_reciprocal_sum_l480_480209

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + sin x

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (4 * a) + f (b - 9) = 0) : 
  (1 / a + 1 / b) = 1 :=
by
  sorry

end min_reciprocal_sum_l480_480209


namespace sum_of_other_endpoint_l480_480323

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (5 + x) / 2 = 3.5) (h2 : (4 + y) / 2 = 10.5) : x + y = 19 := by
sory

end sum_of_other_endpoint_l480_480323


namespace time_to_cross_first_platform_l480_480931

-- Define the given conditions
def length_first_platform : ℕ := 140
def length_second_platform : ℕ := 250
def length_train : ℕ := 190
def time_cross_second_platform : Nat := 20
def speed := (length_train + length_second_platform) / time_cross_second_platform

-- The theorem to be proved
theorem time_to_cross_first_platform : 
  (length_train + length_first_platform) / speed = 15 :=
sorry

end time_to_cross_first_platform_l480_480931


namespace product_min_max_eq_one_ninth_l480_480687

noncomputable def findProduct (x y : ℝ) (h : 3 * x ^ 2 + 6 * x * y + 4 * y ^ 2 = 1) : ℝ :=
  let f xy := 3 * x ^ 2 + 4 * x * y + 3 * y ^ 2 in
  let m := (2 - sqrt 5) / 3 in
  let M := (2 + sqrt 5) / 3 in
  m * M

theorem product_min_max_eq_one_ninth (x y : ℝ) (h : 3 * x ^ 2 + 6 * x * y + 4 * y ^ 2 = 1) :
  findProduct x y h = 1 / 9 :=
  sorry

end product_min_max_eq_one_ninth_l480_480687


namespace y_coord_equidistant_l480_480010

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l480_480010


namespace record_expenditure_l480_480641

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end record_expenditure_l480_480641


namespace sum_first_fifteen_multiples_seventeen_l480_480823

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480823


namespace percentage_loss_15_l480_480487

theorem percentage_loss_15
  (sold_at_loss : ℝ)
  (sold_at_profit : ℝ)
  (percentage_profit : ℝ)
  (cost_price : ℝ)
  (percentage_loss : ℝ)
  (H1 : sold_at_loss = 12)
  (H2 : sold_at_profit = 14.823529411764707)
  (H3 : percentage_profit = 5)
  (H4 : cost_price = sold_at_profit / (1 + percentage_profit / 100))
  (H5 : percentage_loss = (cost_price - sold_at_loss) / cost_price * 100) :
  percentage_loss = 15 :=
by
  sorry

end percentage_loss_15_l480_480487


namespace trulyalya_exists_l480_480090

-- Define the conditions in the problem
def met_character (Alice: Prop) : Prop := 
  ∃ x, (x = Tweedledum ∨ x = Tweedledee) ∧ today_is_a_lying_day x

-- Define the claim about identity and lying
def character_claim (x: Prop) (today_is_a_lying_day: Prop) : Prop := 
  (x = Tweedledum ∨ x = Tweedledee) ∧ today_is_a_lying_day

-- Define the existence of Trulyalya given those claims
def Trulyalya_exists (Alice: Prop) (today_is_a_lying_day: Prop) : Prop :=
  ∃ x, character_claim x today_is_a_lying_day

-- The main theorem statement
theorem trulyalya_exists (Alice: Prop) (today_is_a_lying_day: Prop) :
  Trulyalya_exists Alice today_is_a_lying_day :=
sorry

end trulyalya_exists_l480_480090


namespace simplify_fraction_l480_480337

theorem simplify_fraction (n : ℤ) : 
  (2^(n+4) - 2*(2^n)) / (2*(2^(n+3))) = 7/8 :=
by
  sorry

end simplify_fraction_l480_480337


namespace solve_for_x_l480_480919

theorem solve_for_x (x : ℝ) (h : 0 < x) (h_property : (x / 100) * x^2 = 9) : x = 10 := by
  sorry

end solve_for_x_l480_480919


namespace min_edges_of_different_lengths_l480_480243

noncomputable def differentLengthEdges (tetra : Type) [tetrahedron tetra] : ℕ :=
  sorry

theorem min_edges_of_different_lengths (tetra : Type) [tetrahedron tetra]
  (h : ∀ face : list (list ℝ), face ∉ tetra -> ¬ is_isosceles face) :
  differentLengthEdges tetra = 3 :=
  sorry

end min_edges_of_different_lengths_l480_480243


namespace benny_number_of_days_worked_l480_480955

-- Define the conditions
def total_hours_worked : ℕ := 18
def hours_per_day : ℕ := 3

-- Define the problem statement in Lean
theorem benny_number_of_days_worked : (total_hours_worked / hours_per_day) = 6 := 
by
  sorry

end benny_number_of_days_worked_l480_480955


namespace four_PB_CQ_eq_BC_squared_l480_480739

variables {A B C O P Q : Point}
variables (S : Circle) (AB_AC_eq : Triangle A B C → IsIsosceles A B C)
variables (circle_center_on_base : S.Center = O ∧ Midpoint B C = O)
variables (touches_AB_AC : LetTangents A B C S P Q)
variables (PQ_touches_S : Segment P Q → TouchesCircle S PQ)

theorem four_PB_CQ_eq_BC_squared :
    4 * (dist P B) * (dist Q C) = (dist B C) ^ 2 :=
sorry

end four_PB_CQ_eq_BC_squared_l480_480739


namespace tennis_trio_exists_l480_480471

noncomputable def round_robin_tournament (players : Set ℕ) (played : ℕ → ℕ → Bool) : Prop :=
  ∀ p1 p2 ∈ players, p1 ≠ p2 → (played p1 p2 ∨ played p2 p1)

def exists_three_players (players : Set ℕ) (played : ℕ → ℕ → Bool) : Prop :=
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  ∀ p ∈ (players \ {A, B, C}), ∃ q ∈ {A, B, C}, played q p

theorem tennis_trio_exists :
  (players : Set ℕ) (played : ℕ → ℕ → Bool) (h_players : players.size = 14) 
  (h_round_robin : round_robin_tournament players played) :
  exists_three_players players played :=
sorry

end tennis_trio_exists_l480_480471


namespace pqr_value_l480_480730

theorem pqr_value
  (p q r : ℤ) -- p, q, and r are integers
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) -- non-zero condition
  (h1 : p + q + r = 27) -- sum condition
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 300 / (p * q * r) = 1) -- equation condition
  : p * q * r = 984 := 
sorry 

end pqr_value_l480_480730


namespace find_a_l480_480602

noncomputable def f (a : ℝ) (x : ℝ) := (a * x^2) / (x + 1)

theorem find_a (a : ℝ) (h : ∀ x, f a x = (a * x^2) / (x + 1)) :
  (∀ x, Deriv (f a) x = ((2 * a * x * (x+1) - a * x^2) / (x+1)^2)) →
  (Deriv (f a) 1 = 1) →
  a = 4 / 3 :=
by
  sorry

end find_a_l480_480602


namespace sum_of_first_fifteen_multiples_of_17_l480_480833

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480833


namespace warehouse_painted_area_l480_480083

theorem warehouse_painted_area :
  let length := 8
  let width := 6
  let height := 3.5
  let door_width := 1
  let door_height := 2
  let front_back_area := 2 * (length * height)
  let left_right_area := 2 * (width * height)
  let total_wall_area := front_back_area + left_right_area
  let door_area := door_width * door_height
  let painted_area := total_wall_area - door_area
  painted_area = 96 :=
by
  -- Sorry to skip the actual proof steps
  sorry

end warehouse_painted_area_l480_480083


namespace angle_sum_eq_90_l480_480763

-- Conditions/definitions
variables {A B C D M N : Point}
variables (x : ℝ) -- length of AB
variables (A B C D M N: Point)
variables [rect : Rectangle A B C D]
variables (div_M_N : divides AD 3 [M, N]) -- M and N divide AD in three equal parts

-- Theorem statement
theorem angle_sum_eq_90 :
  ∠AMB + ∠ANB + ∠ADB = 90 :=
sorry

end angle_sum_eq_90_l480_480763


namespace problem_statement_l480_480112

theorem problem_statement : (1 / 8 : ℝ) ^ (-2 / 3) + Real.log 9 / Real.log 3 = 6 := by
  sorry

end problem_statement_l480_480112


namespace numbers_pairs_sum_prod_l480_480423

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l480_480423


namespace trajectory_C_of_M_fixed_point_Q_l480_480600

def point_P_conditions (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 8

def point_F2_symmetric (x y : ℝ) : Prop :=
  (-x - 1)^2 + (-y)^2 = 8

theorem trajectory_C_of_M (x y : ℝ)
    (hP : point_P_conditions x y)
    (hF2 : point_F2_symmetric x y) :
    ∀ M, (2*x + 2*F2, y) = C ∧ x^2/2 + y^2 = 1 := sorry

theorem fixed_point_Q (x₁ x₂ y₁ y₂ k : ℝ)
    (h_line : ∀ x y, y = k * x + 1/3)
    (h_trajectory : ∀ x y, x^2/2 + y^2 = 1)
    (hA : A = (x₁, y₁))
    (hB : B = (x₂, y₂)) :
    ∃ Q : ℝ, (Q = (0, -1)) ∧ 
              ∀ A B, (AB_circle: set (x,y), circle_diameter_AB = (A, B)) := sorry

end trajectory_C_of_M_fixed_point_Q_l480_480600


namespace calculate_expression_l480_480111

theorem calculate_expression :
  4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = -Real.sqrt 3 :=
by
  sorry

end calculate_expression_l480_480111


namespace find_number_l480_480275

theorem find_number (x : ℚ) (h : (49 / 9) / 7 = 5 * x) : x ≈ 0.16 :=
by
    sorry

end find_number_l480_480275


namespace integral_arctg_sqrt_eq_l480_480967

noncomputable def integral_arctg_sqrt : ℝ → ℝ := 
  λ x, x * Real.arctan (Real.sqrt (2 * x - 1)) - (1 / 2) * Real.sqrt (2 * x - 1)

-- Define the problem: Proving that the integral of arctg(sqrt(2x - 1)) equals the expected expression
theorem integral_arctg_sqrt_eq (x : ℝ) : 
  ∫ (t : ℝ) in 0..x, Real.arctan (Real.sqrt (2 * t - 1)) = integral_arctg_sqrt x + C :=
sorry

end integral_arctg_sqrt_eq_l480_480967


namespace find_two_numbers_l480_480462

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l480_480462


namespace same_school_probability_l480_480387

theorem same_school_probability :
  let total_teachers : ℕ := 6
  let teachers_from_school_A : ℕ := 3
  let teachers_from_school_B : ℕ := 3
  let ways_to_choose_2_from_6 : ℕ := Nat.choose total_teachers 2
  let ways_to_choose_2_from_A := Nat.choose teachers_from_school_A 2
  let ways_to_choose_2_from_B := Nat.choose teachers_from_school_B 2
  let same_school_ways : ℕ := ways_to_choose_2_from_A + ways_to_choose_2_from_B
  let probability := (same_school_ways : ℚ) / ways_to_choose_2_from_6 
  probability = (2 : ℚ) / (5 : ℚ) := by sorry

end same_school_probability_l480_480387


namespace sum_of_shaded_areas_l480_480585

-- Given conditions for the problem
def isosceles_triangle (A B C : Type) [has_dist A B C] (l : length A B = length A C) := (B C : diameter (circle))
(A B : AC = 20 cm; ABC : BC = 12 cm; C: circle(r) = 6 cm.)

-- Statement in Lean
theorem sum_of_shaded_areas : 
  ∀ (A B C O : Type) [has_tri A B C A O B O] (h1 : length A B = 20) (h2 : length A C = 20) (h3 : length B C = 12) (h4 : radius (O C) = 6) (h5 : diameter (B C) = 12),
  shaded_area (A B C) == 12 * π - 108 * sqrt 3 :=
by
  sorry

end sum_of_shaded_areas_l480_480585


namespace total_votes_l480_480870

theorem total_votes (V : ℝ) (h1 : 0.60 * V = V - 240) : V = 600 :=
sorry

end total_votes_l480_480870


namespace power_function_expression_l480_480357

theorem power_function_expression (f : ℝ → ℝ) (h : f(2) = 1/4) : f = (λ x, x ^ (-2)) :=
sorry

end power_function_expression_l480_480357


namespace aunt_li_more_cost_effective_l480_480517

theorem aunt_li_more_cost_effective (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (100 * a + 100 * b) / 200 ≥ 200 / ((100 / a) + (100 / b)) :=
by
  sorry

end aunt_li_more_cost_effective_l480_480517


namespace work_completion_time_l480_480862

theorem work_completion_time (hA : 1/10) (hB : 1/15) (hC : 1/20) : 
  (13 * (60 / 13)) / 60 = 1 :=
by
  -- To be proved
  sorry

end work_completion_time_l480_480862


namespace max_of_differentiable_function_l480_480071

variable {a b : ℝ}
variable {f : ℝ → ℝ} (hf : ∀ x ∈ set.Icc a b, differentiable ℝ f)

theorem max_of_differentiable_function (hf : ∀ x ∈ set.Icc a b, differentiable ℝ f) :
  ∃ c ∈ set.Icc a b, ∀ x ∈ set.Icc a b, f c ≥ f x :=
begin
  let S := {x ∈ set.Icc a b | ∀ y ∈ set.Icc a b, f x ≥ f y},
  have hS : S.nonempty,
  -- sorry to skip the proof
  sorry,
  have hmax : ∃ c ∈ S, ∀ x ∈ S, f c ≥ f x,
  from exists_maximum_of_compact -- sorry to skip the proof,
  sorry
  use [c, hc, hc'],
  sorry
end

end max_of_differentiable_function_l480_480071


namespace no_solution_for_inequalities_l480_480726

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l480_480726


namespace sum_three_digit_expansions_base_neg4_plus_i_l480_480582

theorem sum_three_digit_expansions_base_neg4_plus_i :
  let b := (-4 : ℂ) + (1 : ℂ) * complex.i;
  let expand (a_2 a_1 a_0 : ℤ) := a_2 * b^2 + a_1 * b + a_0;
  let valid (a_2 a_1 a_0 : ℕ) := a_2 ≠ 0 ∧ a_1 = 8 * a_2 ∧ 0 ≤ a_0 ∧ a_0 ≤ 16 ∧ a_2 ≤ 2 ∧ a_1 ≤ 16;
  let k_set := {expand a_2 a_1 a_0 | a_2 a_1 a_0 : ℕ, valid a_2 a_1 a_0};
  k_set.sum = -595 :=
by
  sorry

end sum_three_digit_expansions_base_neg4_plus_i_l480_480582


namespace number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l480_480493

theorem number_reduced_by_10_eq_0_09 : ∃ (x : ℝ), x / 10 = 0.09 ∧ x = 0.9 :=
sorry

theorem three_point_two_four_increased_to_three_two_four_zero : ∃ (y : ℝ), 3.24 * y = 3240 ∧ y = 1000 :=
sorry

end number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l480_480493


namespace fraction_equals_decimal_l480_480542

theorem fraction_equals_decimal : (1 / 4 : ℝ) = 0.25 := 
sorry

end fraction_equals_decimal_l480_480542


namespace percent_games_lost_l480_480760

-- Define the conditions
def ratio_games_won_lost (won lost : ℕ) : Prop := won = 7 * lost / 3
def draw_games := 5

-- Define the main statement to prove
theorem percent_games_lost (won lost : ℕ) (h_ratio : ratio_games_won_lost won lost) :
  let total_games := won + lost + draw_games in
  (lost * 100 / total_games).natValue = 30 := 
sorry

end percent_games_lost_l480_480760


namespace binomial_expansion_l480_480412

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l480_480412


namespace sum_first_fifteen_multiples_of_17_l480_480815

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480815


namespace distance_between_lines_is_sqrt2_l480_480742

noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_is_sqrt2 :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_is_sqrt2_l480_480742


namespace area_of_circle_l480_480347

/-- Given a circle with circumference 36π, prove that the area is 324π. -/
theorem area_of_circle (C : ℝ) (hC : C = 36 * π) 
  (h1 : ∀ r : ℝ, C = 2 * π * r → 0 ≤ r)
  (h2 : ∀ r : ℝ, 0 ≤ r → ∃ (A : ℝ), A = π * r^2) :
  ∃ k : ℝ, (A = 324 * π → k = 324) := 
sorry


end area_of_circle_l480_480347


namespace scientific_notation_10870_l480_480938

theorem scientific_notation_10870 : (∃ x : ℝ, x = 10870) → (1.087 * 10^4 = 10870) :=
by
  intro h
  cases h with x hx
  rw hx
  sorry

end scientific_notation_10870_l480_480938


namespace secret_code_count_l480_480660

noncomputable def number_of_secret_codes (colors slots : ℕ) : ℕ :=
  colors ^ slots

theorem secret_code_count : number_of_secret_codes 9 5 = 59049 := by
  sorry

end secret_code_count_l480_480660


namespace left_side_series_n_equals_1_l480_480399

theorem left_side_series_n_equals_1 :
  (1 + 2 + 2^2 = ∑ k in Finset.range 2.succ, 2^k) := 
begin
  -- sorry only
  sorry
end

end left_side_series_n_equals_1_l480_480399


namespace smallest_root_floor_l480_480684

def g (x : Real) : Real := Real.cos x + 3 * Real.sin x + 2 * Real.tan x + x^2

theorem smallest_root_floor :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
by
  sorry

end smallest_root_floor_l480_480684


namespace ganesh_average_speed_l480_480054

variable (D : ℝ) -- the distance between towns X and Y

theorem ganesh_average_speed :
  let time_x_to_y := D / 43
  let time_y_to_x := D / 34
  let total_distance := 2 * D
  let total_time := time_x_to_y + time_y_to_x
  let avg_speed := total_distance / total_time
  avg_speed = 37.97 := by
    sorry

end ganesh_average_speed_l480_480054


namespace min_value_frac_l480_480593

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ (min_val : ℝ), min_val = 9 ∧ ((∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1 / x + 4 / y) ≥ min_val) ∧ (1 / a + 4 / b = min_val)) :=
begin
  sorry
end

end min_value_frac_l480_480593


namespace exists_same_number_of_acquaintances_l480_480516

-- Define a symmetric relationship
def knows (n : ℕ) : Type := 
  {rel : fin n → fin n → Prop // (∀ x y, rel x y → rel y x)}

-- Define the main proof problem
theorem exists_same_number_of_acquaintances (n : ℕ) : 
  ∃ (a b : fin n), a ≠ b ∧ (∑ i : fin n, knows n.1 a i) = (∑ i : fin n, knows n.1 b i) := 
by
  sorry

end exists_same_number_of_acquaintances_l480_480516


namespace binomial_expansion_l480_480410

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l480_480410


namespace conic_curve_cartesian_segment_length_AB_l480_480470

-- Part 1: Cartesian coordinate equation from parametric equations
theorem conic_curve_cartesian (t : ℝ) : 
  (∃ t : ℝ, x = t^2 + 1 / t^2 - 2 ∧ y = t - 1 / t) → y^2 = x := 
sorry

-- Part 2: Length of segment AB given polar equations
theorem segment_length_AB :
  (∃ (A B : ℝ × ℝ), 
    (A = (1, 0) ∧ B = (-(1/2), -(sqrt 3) / 2)) ∧ 
    (A.fst^2 + A.snd^2 = 1 ∧ B.fst^2 + B.snd^2 = 1) ∧ 
    ((B.fst + sqrt 3 * B.snd) = 0) 
  ) → 
  dist (1, 0) (-(1/2), -(sqrt 3) / 2) = sqrt 3 := 
sorry

end conic_curve_cartesian_segment_length_AB_l480_480470


namespace sum_of_alternating_sums_eight_l480_480996

-- Condition: Function to compute the alternating sum for a given subset of natural numbers.
def alternating_sum (s : Finset ℕ) : ℕ :=
  s.to_list.sort (· > ·)  -- Sort the elements in decreasing order.
    .enum.map (λ ⟨i, n⟩, if i % 2 = 0 then n else -n)  -- Alternately add and subtract.
    .sum

-- Problem: Prove that the sum of all alternating sums of non-empty subsets of {1, 2, 3, ..., 8} is 1024.
theorem sum_of_alternating_sums_eight :
  let S := (Finset.range 8).map (λ x => x + 1) in
  let subsets := S.powerset.filter (λ s => s.nonempty) in
  (subsets.sum (λ s => alternating_sum s)) = 1024 :=
by
  sorry

end sum_of_alternating_sums_eight_l480_480996


namespace discount_on_third_shirt_l480_480796

-- Define the conditions
def original_price (n : ℕ) : ℝ := 10
def second_shirt_discount (x : ℝ) : ℝ := x * 0.5
def savings : ℝ := 11

-- Theorem statement: the discount on the third shirt is 60%
theorem discount_on_third_shirt : 
  let price_without_discount := 3 * original_price 1,
      price_with_savings := price_without_discount - savings,
      first_two_shirts_cost := original_price 1 + second_shirt_discount (original_price 1),
      third_shirt_cost_after_discount := price_with_savings - first_two_shirts_cost,
      discount_amount := original_price 1 - third_shirt_cost_after_discount,
      discount_percentage := (discount_amount / original_price 1) * 100
  in
  discount_percentage = 60 := by sorry

end discount_on_third_shirt_l480_480796


namespace machine_production_percentage_difference_l480_480311

theorem machine_production_percentage_difference 
  (X_production_rate : ℕ := 3)
  (widgets_to_produce : ℕ := 1080)
  (difference_in_hours : ℕ := 60) :
  ((widgets_to_produce / (widgets_to_produce / X_production_rate - difference_in_hours) - 
   X_production_rate) / X_production_rate * 100) = 20 := by
  sorry

end machine_production_percentage_difference_l480_480311


namespace find_numbers_l480_480440

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480440


namespace line_equation_l480_480762

def vec_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem line_equation (u : ℝ × ℝ) (h : vec_proj u (3, 1) = (3, 1)) :
  u.2 = -3 * u.1 + 10 :=
sorry

end line_equation_l480_480762


namespace lesser_fraction_exists_l480_480376

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l480_480376


namespace smallest_n_with_common_factor_gt_one_l480_480020

theorem smallest_n_with_common_factor_gt_one :
  ∃ n : ℕ, n > 0 ∧ gcd (12 * n - 3) (8 * n + 9) > 1 ∧ (∀ m : ℕ, m > 0 → gcd (12 * m - 3) (8 * m + 9) > 1 → n ≤ m) :=
begin
  use 3,
  split,
  { linarith, },
  split,
  { have h := gcd_refl (12 * 3 - 3),
    rw [show 12 * 3 - 3 = 33, by norm_num, show 8 * 3 + 9 = 33, by norm_num] at h,
    exact h, },
  { intros m hm hcondition,
    have : 3 ≤ m,
    { 
      sorry, -- Proof that 3 is indeed the smallest
    },
    exact this, }
end

end smallest_n_with_common_factor_gt_one_l480_480020


namespace other_root_of_quadratic_l480_480324

theorem other_root_of_quadratic (h : (5 + 10 * Complex.i) * (5 + 10 * Complex.i) = -100 + 75 * Complex.i) :
    -5 - 10 * Complex.i = -5 - 10 * Complex.i := sorry

end other_root_of_quadratic_l480_480324


namespace teacher_allocation_schemes_l480_480388

theorem teacher_allocation_schemes :
  let teachers := 4
  let schools := 3
  (teachers > 0) → (schools > 0) → 
  (∀ (alloc : (ℕ → ℕ)), ∃ (conditions : Prop), 
    (conditions → (∀ i, 1 ≤ alloc i) ∧ ∑ i, alloc i = teachers ∧ ∏ i in finset.range schools, i.fact ≤ alloc i))
    → 36 :=
by
  intros teachers schools ht hs hab_conditions
  sorry

end teacher_allocation_schemes_l480_480388


namespace largest_possible_median_l480_480809

theorem largest_possible_median (x y : ℤ) : 
  let s := [x, 2 * x, y, 4, 6, 8].sort (≤) in
  (s.nth 2 + s.nth 3) / 2 = 8.5 := 
sorry

end largest_possible_median_l480_480809


namespace tangent_line_at_e_intervals_of_monotonicity_l480_480579
open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_e :
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, y x = 2 * x - exp 1) ∧ (y (exp 1) = f (exp 1)) ∧ (deriv f (exp 1) = deriv y (exp 1)) :=
sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, 0 < x ∧ x < exp (-1) → deriv f x < 0) ∧ (∀ x : ℝ, exp (-1) < x → deriv f x > 0) :=
sorry

end tangent_line_at_e_intervals_of_monotonicity_l480_480579


namespace find_power_l480_480239

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power_l480_480239


namespace problem_1_monotonicity_problem_2_maximum_area_l480_480578

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) - (Real.cos (x + Real.pi / 4))^2

/-- f(x) is monotonically increasing in the intervals [-π/4 + kπ, π/4 + kπ] for k ∈ ℤ -/
theorem problem_1_monotonicity : ∀ k : ℤ, monotone_on (λ x : ℝ, f x) 
        (Set.Icc (-Real.pi / 4 + k * Real.pi) (Real.pi / 4 + k * Real.pi)) := sorry

/-- Given f(A / 2) = 0 and a = 1 in an acute triangle ΔABC,
   the maximum area of ΔABC is (2 + √3) / 4 -/
theorem problem_2_maximum_area (A B C : ℝ) (a b c : ℝ) 
        (h1 : a = 1) (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
        (h3 : 0 < A ∧ A < Real.pi / 2) (h4 : f (A / 2) = 0) :
    (1 / 2) * b * c * Real.sin A ≤ (2 + Real.sqrt 3) / 4 := sorry

end problem_1_monotonicity_problem_2_maximum_area_l480_480578


namespace extremal_points_product_gt_e_square_l480_480211

-- Function definition
def f (x a b : ℝ) : ℝ := b * Real.log x - (1 / 2) * a * x^2 - x + a

-- Conditions
variable (a : ℝ) (H_a : a ≤ 0)

-- Specific case derivatives when b = x
def f_deriv (x a : ℝ) := (2:ℝ) / x - a * x - 1

def extremal_points_candidates : Set ℝ :=
  {x | f_deriv x a = 0}

variable (x₁ x₂ : ℝ) (Hx₁ : x₁ ∈ extremal_points_candidates a)
  (Hx₂ : x₂ ∈ extremal_points_candidates a) (H_distinct : x₁ ≠ x₂)

theorem extremal_points_product_gt_e_square : x₁ * x₂ > Real.exp 2 := sorry

end extremal_points_product_gt_e_square_l480_480211


namespace find_x_for_orthogonal_vectors_l480_480138

noncomputable def x_that_makes_vectors_orthogonal (x : ℝ) : Prop :=
  let v1 := ⟨3, 7⟩ : ℝ × ℝ
  let v2 := ⟨x, -4⟩ : ℝ × ℝ
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_x_for_orthogonal_vectors :
  x_that_makes_vectors_orthogonal (28 / 3) :=
sorry

end find_x_for_orthogonal_vectors_l480_480138


namespace gingerbreads_per_tray_l480_480979

-- Given conditions
def total_baked_gb (x : ℕ) : Prop := 4 * 25 + 3 * x = 160

-- The problem statement
theorem gingerbreads_per_tray (x : ℕ) (h : total_baked_gb x) : x = 20 := 
by sorry

end gingerbreads_per_tray_l480_480979


namespace triangle_angle_properties_l480_480095

theorem triangle_angle_properties (
  (A B C D : Type)
  (h1 : ∠ABC = ∠ACB)
  (h2 : ∠ADC = ∠DAC)
  (h3 : ∠DAB = 21)) :
  ∠ABC = 46 ∧ acute_triangle ABC ∧ acute_triangle ADC :=
by {
    -- We'll use the given conditions to find the solution
    sorry
}

end triangle_angle_properties_l480_480095


namespace combination_60_2_l480_480531

theorem combination_60_2 : nat.choose 60 2 = 1770 :=
by sorry

end combination_60_2_l480_480531


namespace odd_function_value_at_neg_two_l480_480058

noncomputable def f : ℝ → ℝ 
| x => if x ≥ 0 then log 3 (1 + x) else - log 3 (1 - x)

theorem odd_function_value_at_neg_two :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = log 3 (1 + x)) →
  f (-2) = -1 :=
by
  intros odd_def pos_def
  have eq_pos := pos_def 2 (by linarith)
  have odd_eq := odd_def 2
  rw [← eq_pos, odd_eq]
  simp [odd_def, pos_def, f]
  sorry

end odd_function_value_at_neg_two_l480_480058


namespace triangle_side_ratio_l480_480253

theorem triangle_side_ratio (a b c: ℝ) (A B C: ℝ) (h1: b * Real.cos C + c * Real.cos B = 2 * b) :
  a / b = 2 :=
sorry

end triangle_side_ratio_l480_480253


namespace find_numbers_l480_480436

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l480_480436


namespace max_g_of_9_l480_480302

noncomputable def g (x : ℝ) := ∑ i in finset.range (n+1), b i * x^i

theorem max_g_of_9 {b : ℕ → ℝ} (h_nonneg : ∀ n, 0 ≤ b n) 
  (h_g3 : g 3 = 9) (h_g18 : g 18 = 972) :
  g 9 ≤ 81 :=
sorry

end max_g_of_9_l480_480302


namespace rhombus_volume_of_revolution_l480_480923

theorem rhombus_volume_of_revolution (a : ℝ) (bd : ℝ) :
  a = 1 →
  bd = 2 * (sqrt (((a^2 - (a/2)^2)))) →
  let V1 := π * ((bd / 2)^2) * 1,
      V2 := (1/3) * π * (bd^2/4) in
  V1 + 2 * V2 = 3 * π / 2 :=
by
  intros ha hbd,
  let V1 := π * ((bd / 2)^2) * 1,
  let V2 := (1/3) * π * (bd^2/4),
  -- Calculation follows
  sorry

end rhombus_volume_of_revolution_l480_480923


namespace quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l480_480697

structure Point where
  x : ℚ
  y : ℚ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 6, y := 1 }

def line_eq_y_eq_kx_plus_b (k b x : ℚ) : ℚ := k * x + b

def intersects (A : Point) (P : Point × Point) (x y : ℚ) : Prop :=
  ∃ k b, P.1.y = line_eq_y_eq_kx_plus_b k b P.1.x ∧ P.2.y = line_eq_y_eq_kx_plus_b k b P.2.x ∧
         y = line_eq_y_eq_kx_plus_b k b x

theorem quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176 :
  ∃ (p q r s : ℚ), 
    gcd p q = 1 ∧ gcd r s = 1 ∧ intersects A (C, D) (p / q) (r / s) ∧
    (p + q + r + s = 176) :=
sorry

end quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l480_480697


namespace complement_intersection_l480_480701

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 3 ≤ x}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A ∩ B) ↔ (0 ≤ x ∧ x < 3) :=
by {
  sorry
}

end complement_intersection_l480_480701


namespace part1_zero_count_part1_zero_one_part2_range_of_a_l480_480207

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1_zero_count (a : ℝ) (h : 0 < a ∧ a < Real.exp 1) :
  ∀ x : ℝ, (f x a = 0 → False) :=
sorry

theorem part1_zero_one (a : ℝ) (h : a ≤ 0):
  ∃ x : ℝ, f x a = 0 :=
sorry

noncomputable def ineq_func (x x a : ℝ) : ℝ := ax^a * (Real.log x) - x * Real.exp x

theorem part2_range_of_a (a : ℝ) (x : ℝ) (h : x ∈ Set.Ioi 1) :
  f x a ≥ ax^a * Real.log x - x * Real.exp x → a ∈ Set.Iio (Real.exp 1) :=
sorry

end part1_zero_count_part1_zero_one_part2_range_of_a_l480_480207


namespace crayons_per_pack_l480_480318

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end crayons_per_pack_l480_480318


namespace sum_first_fifteen_multiples_seventeen_l480_480819

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480819


namespace sum_first_fifteen_multiples_seventeen_l480_480825

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480825


namespace quadrilateral_parallelogram_l480_480390

noncomputable def is_parallelogram (A B C D : Type) [affine_space A] [affine_space B] [affine_space C] [affine_space D] : Prop :=
-- Definition: A quadrilateral is a parallelogram if its diagonals bisect each other.
∃ P : A, is_bisector (A, C, P) ∧ is_bisector (B, D, P)

theorem quadrilateral_parallelogram {S A B C D : Type} [affine_space S] [affine_space A] [affine_space B] [affine_space C] [affine_space D]
    (SC : parallel S C) (SD : parallel S D) (SA : parallel S A) (SB : parallel S B)
    (P : Type)
    (parallel_A_SC : ∀ (P : Type), parallel A SC) 
    (parallel_B_SD : ∀ (P : Type), parallel B SD) 
    (parallel_C_SA : ∀ (P : Type), parallel C SA) 
    (parallel_D_SB : ∀ (P : Type), parallel D SB)
    (intersect_one_point : ∀ (P Q : Type), intersect_at_one_point P Q) :
is_parallelogram A B C D :=
by
  sorry

end quadrilateral_parallelogram_l480_480390


namespace rhombus_diagonals_sum_squares_l480_480649

-- Definition of the rhombus side length condition
def is_rhombus_side_length (side_length : ℝ) : Prop :=
  side_length = 2

-- Lean 4 statement for the proof problem
theorem rhombus_diagonals_sum_squares (side_length : ℝ) (d1 d2 : ℝ) 
  (h : is_rhombus_side_length side_length) :
  side_length = 2 → (d1^2 + d2^2 = 16) :=
by
  sorry

end rhombus_diagonals_sum_squares_l480_480649


namespace sum_of_first_fifteen_multiples_of_17_l480_480826

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480826


namespace cannot_achieve_l480_480995

def is_valid_coin (value : ℕ) : Prop :=
  value = 1 ∨ value = 5 ∨ value = 10 ∨ value = 25

theorem cannot_achieve (values : Finset ℕ) (coins : List ℕ)
    (h_valid : ∀ x ∈ coins, is_valid_coin x)
    (h_len : coins.length = 5) :
  (22 ∉ (sum <$> (coins.permutations : Finset (List ℕ)))) ∧
  (48 ∉ (sum <$> (coins.permutations : Finset (List ℕ)))) :=
by
  sorry

end cannot_achieve_l480_480995


namespace quadratic_function_y_at_4_neg_l480_480606

theorem quadratic_function_y_at_4_neg : ∀ (a b c : ℝ), 
  let f := λ x : ℝ, a * x^2 + b * x + c in
  (f (-1) = -3) ∧ (f 0 = 1) ∧ (f 1 = 3) ∧ (f 3 = 1) →
  f 4 < 0 :=
by sorry

end quadratic_function_y_at_4_neg_l480_480606


namespace sum_first_fifteen_multiples_seventeen_l480_480820

theorem sum_first_fifteen_multiples_seventeen : 
  let sequence_sum := 17 * (∑ k in set.Icc 1 15, k) in
  sequence_sum = 2040 := 
by
  -- let sequence_sum : ℕ := 17 * (∑ k in finset.range 15, (k + 1))
  sorry

end sum_first_fifteen_multiples_seventeen_l480_480820


namespace ants_distance_after_1990_segments_l480_480395

def unit_cube_vertex := (ℝ × ℝ × ℝ)

def final_position_after_n_segments (segments : ℕ) (path : ℕ → unit_cube_vertex) : unit_cube_vertex :=
  path (segments % 6)

noncomputable def white_ant_path : ℕ → unit_cube_vertex
| 0 => (0, 0, 0)
| 1 => (1, 0, 0)
| 2 => (1, 1, 0)
| 3 => (1, 1, 1)
| 4 => (0, 1, 1)
| 5 => (0, 0, 1)
| _ => white_ant_path ((nat % 6 5) + 1)

noncomputable def black_ant_path : ℕ → unit_cube_vertex
| 0 => (0, 0, 0)
| 1 => (0, 0, 1)
| 2 => (0, 1, 1)
| 3 => (1, 1, 1)
| 4 => (1, 1, 0)
| 5 => (1, 0, 0)
| _ => black_ant_path ((nat % 6 5) + 1)

theorem ants_distance_after_1990_segments :
  let white_final := final_position_after_n_segments 1990 white_ant_path,
      black_final := final_position_after_n_segments 1990 black_ant_path in
      dist white_final black_final = real.sqrt 2 :=
sorry

end ants_distance_after_1990_segments_l480_480395


namespace paula_bracelets_count_l480_480905

-- Defining the given conditions
def cost_bracelet := 4
def cost_keychain := 5
def cost_coloring_book := 3
def total_spent := 20

-- Defining the cost for Paula's items
def cost_paula (B : ℕ) := B * cost_bracelet + cost_keychain

-- Defining the cost for Olive's items
def cost_olive := cost_coloring_book + cost_bracelet

-- Defining the main problem
theorem paula_bracelets_count (B : ℕ) (h : cost_paula B + cost_olive = total_spent) : B = 2 := by
  sorry

end paula_bracelets_count_l480_480905


namespace unique_ordered_triplet_l480_480987

theorem unique_ordered_triplet (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : even r) :
  p^3 + q^2 = 4 * r^2 + 45 * r + 103 → (p, q, r) = (7, 2, 4) :=
by sorry

end unique_ordered_triplet_l480_480987


namespace triangle_area_l480_480703

/-- Given the triangle ABC with medians BD and CE being perpendicular, and 
    lengths BD = 8 and CE = 12, prove that the area of triangle ABC is 64. -/
theorem triangle_area (A B C D E : Point)
  (B_median : median A B D)
  (C_median : median A C E)
  (BD_len : length B D = 8)
  (CE_len : length C E = 12)
  (BD_perp_CE : perpendicular B D C E) :
  area A B C = 64 :=
sorry

end triangle_area_l480_480703


namespace problem_1_problem_2_problem_3_l480_480103

-- First proof statement
theorem problem_1 : 2017^2 - 2016 * 2018 = 1 :=
by
  sorry

-- Definitions for the second problem
variables {a b : ℤ}

-- Second proof statement
theorem problem_2 (h1 : a + b = 7) (h2 : a * b = -1) : (a + b)^2 = 49 :=
by
  sorry

-- Third proof statement (part of the second problem)
theorem problem_3 (h1 : a + b = 7) (h2 : a * b = -1) : a^2 - 3 * a * b + b^2 = 54 :=
by
  sorry

end problem_1_problem_2_problem_3_l480_480103


namespace garrett_cats_count_l480_480316

def number_of_cats_sheridan : ℕ := 11
def difference_in_cats : ℕ := 13

theorem garrett_cats_count (G : ℕ) (h : G - number_of_cats_sheridan = difference_in_cats) : G = 24 :=
by
  sorry

end garrett_cats_count_l480_480316


namespace intersection_proves_pqrs_l480_480306

structure Point where
  x : ℚ
  y : ℚ

noncomputable def A : Point := ⟨0, 0⟩
noncomputable def B : Point := ⟨1, 3⟩
noncomputable def C : Point := ⟨4, 4⟩
noncomputable def D : Point := ⟨5, 0⟩

noncomputable def intersection_point : Point :=
let p : ℚ := 141
let q : ℚ := 32
let r : ℚ := 19
let s : ℚ := 8 
in ⟨p/q, r/s⟩

theorem intersection_proves_pqrs (line_through_B : Line) (h : line_through_B.pass_through B ∧ splits_quadrilateral_equal_area line_through_B (quadrilateral A B C D)) :
  let point := intersection_point in
  point.x.num + point.x.denom + point.y.num + point.y.denom = 200 := sorry

end intersection_proves_pqrs_l480_480306


namespace remainder_of_3056_div_78_l480_480018

-- Define the necessary conditions and the statement
theorem remainder_of_3056_div_78 : (3056 % 78) = 14 :=
by
  sorry

end remainder_of_3056_div_78_l480_480018


namespace num_distinct_three_digit_numbers_l480_480228

def valid_digits : Set ℕ := {1, 2, 3}

def is_distinct_three_digit (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a ∈ valid_digits ∧ b ∈ valid_digits ∧ c ∈ valid_digits ∧
  n = 100 * a + 10 * b + c

theorem num_distinct_three_digit_numbers : 
  (Finset.filter is_distinct_three_digit (Finset.range (1000))).card = 6 :=
by
  sorry

end num_distinct_three_digit_numbers_l480_480228


namespace outlet_pipe_rate_l480_480929

theorem outlet_pipe_rate 
(volume_ft : ℝ) 
(inlet_rate_cub_in_min : ℝ) 
(outlet1_rate_cub_in_min : ℝ) 
(time_min : ℝ) 
(volume_conversion : ℝ) :
    (volume_ft = 30) → 
    (inlet_rate_cub_in_min = 5) →
    (outlet1_rate_cub_in_min = 8) →
    (time_min = 4320) →
    (volume_conversion = 12^3) →
    ∃ x : ℝ, (x + 3) * time_min = volume_ft * volume_conversion / x := 
begin
    intros h1 h2 h3 h4 h5,
    use 9,
    sorry
end

end outlet_pipe_rate_l480_480929


namespace intersection_points_correct_l480_480333

noncomputable def total_intersection_points : ℕ :=
  let pairs := [(4, 6), (4, 9), (4, 10), (6, 9), (6, 10), (9, 10)]
  let intersection_rule (n m : ℕ) (n_smaller : n < m) := 2 * n
  pairs.map (fun pair => (intersection_rule pair.1 pair.2 (by {
    cases pair,
    repeat { 
      { simp, norm_num, done }
    }
  }))).sum

theorem intersection_points_correct :
  let polygons := [4, 6, 9, 10]
  ∀ n ∈ polygons, ∀ m ∈ polygons, n < m →
    no_two_polygons_share_vertex polygons ∧
    no_three_sides_intersect_at_common_point polygons →
    total_intersection_points = 66 := by sorry

end intersection_points_correct_l480_480333


namespace magnitude_relationship_l480_480245

variable {f : ℝ → ℝ}

-- Conditions
def derivative_condition (x : ℝ) : Prop := deriv f x < f x

-- Theorem statement
theorem magnitude_relationship (h : ∀ x, derivative_condition x) : f 3 < real.exp 3 * f 0 :=
sorry

end magnitude_relationship_l480_480245


namespace park_length_l480_480912

theorem park_length (width : ℕ) (trees_per_sqft : ℕ) (num_trees : ℕ) (total_area : ℕ) (length : ℕ)
  (hw : width = 2000)
  (ht : trees_per_sqft = 20)
  (hn : num_trees = 100000)
  (ha : total_area = num_trees * trees_per_sqft)
  (hl : length = total_area / width) :
  length = 1000 :=
by
  sorry

end park_length_l480_480912


namespace max_parallelogram_area_l480_480204

-- Define the fixed point F and the line equation
def F : ℝ × ℝ := (-real.sqrt 3, 0)
def line_eq (x : ℝ) : ℝ := - (4 * real.sqrt 3) / 3

-- Define the moving point M and the ratio condition
def M (x y : ℝ) : Prop := (real.sqrt ((x + real.sqrt 3) ^ 2 + y ^ 2)) / (abs (x + (4 * real.sqrt 3) / 3)) = real.sqrt 3 / 2

-- Define the ellipse C
def C : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 4 + y^2 = 1) }

-- Prove that given conditions |AB| = |DE|, the max area of parallelogram is 4
theorem max_parallelogram_area
  (A B D E : ℝ × ℝ)
  (k t1 t2 : ℝ) (ht : t1 ≠ t2)
  (l1 : ∀ x, l1 = (k * x + t1))
  (l2 : ∀ x, l2 = (k * x + t2))
  (AB_DE_equal : dist A B = dist D E)
  (A_on_C : A ∈ C) (B_on_C : B ∈ C)
  (D_on_C : D ∈ C) (E_on_C : E ∈ C) :
  ∃ S : ℝ, is_parallelogram A B D E ∧ S ≤ 4 :=
by
  sorry

end max_parallelogram_area_l480_480204


namespace max_slope_on_circle_l480_480196

open Real

-- Define the condition that point P(x, y) lies on the circle (x + 2)^2 + y^2 = 1
def on_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the function to compute the value
noncomputable def slope (x y : ℝ) : ℝ := (y - 2) / (x - 1)

-- The main theorem stating the maximum value of the slope function
theorem max_slope_on_circle (x y : ℝ) (h : on_circle x y) : 
    ∃ k : ℝ, k = slope x y ∧ k ≤ (3 + sqrt 3) / 4 :=
sorry

end max_slope_on_circle_l480_480196


namespace factorial_division_l480_480534

theorem factorial_division :
  (50! / 48! : ℕ) = 2450 :=
by {
  sorry
}

end factorial_division_l480_480534


namespace floor_of_pi_l480_480551

noncomputable def floor_of_pi_eq_three : Prop :=
  ⌊Real.pi⌋ = 3

theorem floor_of_pi : floor_of_pi_eq_three :=
  sorry

end floor_of_pi_l480_480551


namespace find_tangent_lines_to_circle_l480_480163

noncomputable def tangent_lines (M : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) :=
  { l : ℝ → ℝ | ∀ (x y : ℝ), (x = 2) ∨ (24 * x - 7 * y - 20 = 0) }

theorem find_tangent_lines_to_circle : 
  let M := (2, 4)
  let C := (1, -3)
  let r := 1
  tangent_lines M C r = { λ (x : ℝ), x = 2, λ (x y : ℝ), 24 * x - 7 * y - 20 = 0 }
:= by
sorry

end find_tangent_lines_to_circle_l480_480163


namespace number_of_attendants_using_pencil_l480_480096

theorem number_of_attendants_using_pencil :
  ∀ (PenUsers : ℕ) (OnlyOneType : ℕ) (BothTypes : ℕ), 
    PenUsers = 15 → OnlyOneType = 20 → BothTypes = 10 → (OnlyOneType + BothTypes - PenUsers + BothTypes = 25) := 
by 
  intros PenUsers OnlyOneType BothTypes hPenUsers hOnlyOneType hBothTypes
  rw [hPenUsers, hOnlyOneType, hBothTypes]
  sorry

end number_of_attendants_using_pencil_l480_480096


namespace intersection_of_M_N_l480_480187

-- Definitions of the sets M and N
def M : Set ℝ := { x | (x + 2) * (x - 1) < 0 }
def N : Set ℝ := { x | x + 1 < 0 }

-- Proposition stating that the intersection of M and N is { x | -2 < x < -1 }
theorem intersection_of_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < -1 } :=
  by
    sorry

end intersection_of_M_N_l480_480187


namespace hyperbola_equation_l480_480174

theorem hyperbola_equation (a b : ℝ) (h0 : a > 0) (h1 : b > 0)
  (h2 : -b / a = -1 / 3)
  (h3 : c^2 = a^2 + b^2 ∧ c = 2 * sqrt 5) :
  (∃ (h : ℝ → ℝ → Prop), ∀ x y, h x y ↔ (x^2 / 18 - y^2 / 2 = 1)) :=
by
  use (λ x y, x^2 / 18 - y^2 / 2 = 1)
  sorry

end hyperbola_equation_l480_480174


namespace percentage_of_muslims_l480_480260

theorem percentage_of_muslims (total_boys : ℕ) (percent_hindus percent_sikhs : ℕ) (other_communities : ℕ) :
  total_boys = 850 →
  percent_hindus = 28 →
  percent_sikhs = 10 →
  other_communities = 136 →
  let hindus := (percent_hindus * total_boys) / 100 in
  let sikhs := (percent_sikhs * total_boys) / 100 in
  let non_muslims := hindus + sikhs + other_communities in
  let muslim_boys := total_boys - non_muslims in
  (muslim_boys * 100) / total_boys = 46 :=
sorry

end percentage_of_muslims_l480_480260


namespace discount_comparison_l480_480492

noncomputable def final_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  P * (1 - d1) * (1 - d2) * (1 - d3)

theorem discount_comparison (P : ℝ) (d11 d12 d13 d21 d22 d23 : ℝ) :
  P = 20000 →
  d11 = 0.25 → d12 = 0.15 → d13 = 0.10 →
  d21 = 0.30 → d22 = 0.10 → d23 = 0.10 →
  final_price P d11 d12 d13 - final_price P d21 d22 d23 = 135 :=
by
  intros
  sorry

end discount_comparison_l480_480492


namespace find_value_of_m_l480_480876

variable (i m : ℂ)

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_value_of_m 
  (i_imag : i.im = 1 ∧ i.re = 0)
  (pure_imag : is_pure_imaginary ((3 - i) * (m + i))) :
  m = - (1 / 3) :=
sorry

end find_value_of_m_l480_480876


namespace medication_expiration_l480_480927
open nat

theorem medication_expiration :
  let release_time := (15, 0) -- 3 PM represented as (hour, minute)
  let expiry_seconds := fact 8
  let seconds_per_minute := 50
  let minutes_per_hour := 60
  let hours_per_day := 24
  let total_seconds_per_day := seconds_per_minute * minutes_per_hour * hours_per_day
  let days_to_expire := (expiry_seconds : ℝ) / (total_seconds_per_day : ℝ)
  let hours_to_expire := days_to_expire * (hours_per_day : ℝ)
  let expiration_time := release_time.1 + (hours_to_expire : ℕ)
  let expiration_day := if expiration_time >= hours_per_day then "February 15" else "February 14"
  let expiration_hour := expiration_time % hours_per_day 
  (expiration_day = "February 15") ∧ (4 <= expiration_hour ∧ expiration_hour < 5) :=
sorry

end medication_expiration_l480_480927


namespace domain_of_f_l480_480744

noncomputable def f (x : ℝ) : ℝ :=  1 / Real.sqrt (1 - x^2) + x^0

theorem domain_of_f (x : ℝ) : (x > -1 ∧ x < 1 ∧ x ≠ 0) ↔ (x ∈ (-1, 0) ∨ x ∈ (0, 1)) :=
by
  sorry

end domain_of_f_l480_480744


namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l480_480233

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l480_480233


namespace average_age_new_students_l480_480349

-- Define the given constants
def O : ℕ := 12
def A_O : ℕ := 40
def N : ℕ := 12
def new_average : ℕ := 36

-- Prove that the average age of the new students is 32
theorem average_age_new_students : ∃ A_N : ℕ, 12 * 40 + 12 * A_N = 24 * 36 ∧ A_N = 32 :=
by {
  use 32,
  have h1 : 12 * 40 + 12 * 32 = 24 * 36,
  { norm_num, },
  simp [h1],
}

end average_age_new_students_l480_480349


namespace coefficient_of_x_is_neg7_l480_480990

def expr : ℝ → ℝ := λ x, 5 * (x - 6) + 6 * (9 - 3 * x^2 + 3 * x) - 10 * (3 * x - 2)

theorem coefficient_of_x_is_neg7 : 
  (∃ c, ∀ x, expr x = c * x) ∧ (expr 1 - expr 0 = -7) :=
by 
  sorry

end coefficient_of_x_is_neg7_l480_480990


namespace sum_of_first_fifteen_multiples_of_17_l480_480839

theorem sum_of_first_fifteen_multiples_of_17 : 
  let k := 17 in
  let n := 15 in
  let sum_first_n_natural_numbers := n * (n + 1) / 2 in
  k * sum_first_n_natural_numbers = 2040 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480839


namespace tara_dice_probability_divisible_by_8_l480_480735

-- Definitions based on the conditions
def standard_die := {1, 2, 3, 4, 5, 6}
def dice_rolls (n : ℕ) := fin n → standard_die

-- Theorem to prove the probability
theorem tara_dice_probability_divisible_by_8 :
  ∀ (product : ℕ) (rolls : dice_rolls 8), 
  (product = ∏ i, rolls i) →
  (∀ i, rolls i ∈ standard_die) →
  probability (product % 8 = 0) = 1143 / 1152 :=
sorry

end tara_dice_probability_divisible_by_8_l480_480735


namespace distance_between_points_l480_480808

def distance (x1 y1 x2 y2 : ℝ) :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : distance 5 (-3) 9 6 = Real.sqrt 97 :=
by
  sorry

end distance_between_points_l480_480808


namespace largest_even_integer_l480_480766

theorem largest_even_integer (x : ℤ) (h₁ : (1:ℤ) ≤ 30) (h₂ : 2 * x + 58 = 13,500) (h₃ : 2 ∣ 58) : x + 58 = 479 :=
begin
  sorry
end

end largest_even_integer_l480_480766


namespace find_a_l480_480220

def setA (a : ℤ) : Set ℤ := {a, 0}

def setB : Set ℤ := {x : ℤ | 3 * x^2 - 10 * x < 0}

theorem find_a (a : ℤ) (h : (setA a ∩ setB).Nonempty) : a = 1 ∨ a = 2 ∨ a = 3 :=
sorry

end find_a_l480_480220


namespace sum_first_fifteen_multiples_of_17_l480_480814

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480814


namespace sum_of_first_fifteen_multiples_of_17_l480_480829

theorem sum_of_first_fifteen_multiples_of_17 : 
  (∑ i in Finset.range 15, 17 * (i + 1)) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480829


namespace geometric_series_sum_l480_480012

theorem geometric_series_sum :
  let a := (1:ℚ) / 2
  let r := (1:ℚ) / 2
  let n := 7
  let sum := ∑ k in Finset.range n, a * r^k
  sum = 127 / 128 :=
by
  let a := (1:ℚ) / 2
  let r := (1:ℚ) / 2
  let n := 7
  let sum := ∑ k in Finset.range n, a * r^k
  have h : sum = a * (1 - r^n) / (1 - r) := sorry
  have h2 : (1:ℚ) / 2 ^ n = 1 / (2 ^ n) := sorry -- Proof for simplification of power
  have h3 : 2 ^ n = 128 := by norm_num
  have h4 : 1 - (1 / 128:ℚ) = 127 / 128 := by norm_num -- Direct calculation
  have h5 : 2 * 1/2 = 1 := by norm_num
  exact h4

end geometric_series_sum_l480_480012


namespace negation_example_l480_480363

open Real

theorem negation_example : 
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n ≥ x^2) ↔ ∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2 := 
  sorry

end negation_example_l480_480363


namespace solve_trig_equation_l480_480339

open Real

theorem solve_trig_equation (x : ℝ) :
  (sqrt 2 * (sin x + cos x) = tan x + cot x) ↔ ∃ l : ℤ, x = (π / 4) + 2 * l * π :=
  sorry

end solve_trig_equation_l480_480339


namespace v_2015_eq_2_l480_480970

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | 5 => 2
  | _ => 0  -- assuming g(x) = 0 for other values, though not used here

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n)

theorem v_2015_eq_2 : v 2015 = 2 :=
by
  sorry

end v_2015_eq_2_l480_480970


namespace polynomial_102_l480_480413

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l480_480413


namespace domain_of_f_l480_480746

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l480_480746


namespace find_a_l480_480618

noncomputable def equation_of_circle := x : ℝ -> y : ℝ -> a : ℝ -> x^2 + y^2 + 2 * x - 2 * y + a = 0

noncomputable def equation_of_line := x : ℝ -> y : ℝ -> x + y + 4 = 0

theorem find_a (a : ℝ) :
  (∃ (x y : ℝ), equation_of_circle x y a ∧ equation_of_line x y) →
  (chord_length : ℝ) = 2 →
  a = -7 :=
by
  sorry

end find_a_l480_480618


namespace divides_2pow18_minus_1_l480_480129

theorem divides_2pow18_minus_1 (n : ℕ) : 20 ≤ n ∧ n < 30 ∧ (n ∣ 2^18 - 1) ↔ (n = 19 ∨ n = 27) := by
  sorry

end divides_2pow18_minus_1_l480_480129


namespace ellipse_properties_l480_480354

namespace MathProof

-- Definitions and conditions
variables {a b c : ℝ} {x y : ℝ}

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def sum_of_distances_eq (F1 F2 G : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  (F1.1 - G.1) + (F2.1 - G.1) + (D.1 - G.1)
def DF1_eq_3GF1 (D F1 G : ℝ × ℝ) : Prop :=
  abs (D.1 - F1.1) = 3 * abs (G.1 - F1.1)
def line_through_fixed_point (x y : ℝ) : Prop := 3 * x = -2 * y - 2

-- Theorem statement
theorem ellipse_properties
  (h₁ : 0 < b) (h₂ : b < a)
  (hE : is_ellipse a b x y)
  (hDGF2 : sum_of_distances_eq (0,0) (0,0) (x,y) (0,0) = 8)
  (h_relation : DF1_eq_3GF1 (0,0) (c,0) (x,y))
  (hyp : x = -4/3 * c ∧ y = 1/3 * b) :
  (is_ellipse 2 (sqrt 2) x y) ∧ (line_through_fixed_point (-2/3) 0) :=
sorry

end MathProof

end ellipse_properties_l480_480354


namespace locus_of_circumcenter_XAY_l480_480907

variables {A B C X Y : Point}

def locus_of_circumcenter (X A Y : Point) : Line := sorry

theorem locus_of_circumcenter_XAY (h₁ : Collinear A B X)
                                  (h₂ : Collinear A C Y)
                                  (h₃ : Distance B X = Distance C Y) :
    Parallel (locus_of_circumcenter X A Y) (angle_bisector A B C) :=
sorry

end locus_of_circumcenter_XAY_l480_480907


namespace calculate_expression_l480_480105

theorem calculate_expression (y : ℝ) (hy : y ≠ 0) : 
  (18 * y^3) * (4 * y^2) * (1/(2 * y)^3) = 9 * y^2 :=
by
  sorry

end calculate_expression_l480_480105


namespace boy_needs_to_sell_75_oranges_to_make_150c_profit_l480_480477

-- Definitions based on the conditions
def cost_per_orange : ℕ := 12 / 4
def sell_price_per_orange : ℕ := 30 / 6
def profit_per_orange : ℕ := sell_price_per_orange - cost_per_orange

-- Problem declaration
theorem boy_needs_to_sell_75_oranges_to_make_150c_profit : 
  (150 / profit_per_orange) = 75 :=
by
  -- Proof will be added here
  sorry

end boy_needs_to_sell_75_oranges_to_make_150c_profit_l480_480477


namespace sum_first_fifteen_multiples_of_17_l480_480816

theorem sum_first_fifteen_multiples_of_17 : 
  ∑ k in Finset.range 15, (k + 1) * 17 = 2040 :=
by
  sorry

end sum_first_fifteen_multiples_of_17_l480_480816


namespace sum_of_first_fifteen_multiples_of_17_l480_480848

theorem sum_of_first_fifteen_multiples_of_17 : 
  ∑ i in Finset.range 15, 17 * (i + 1) = 2040 := 
by
  sorry

end sum_of_first_fifteen_multiples_of_17_l480_480848


namespace line_eqn_in_slope_intercept_form_l480_480489

theorem line_eqn_in_slope_intercept_form :
  ∃ (m b : ℚ), 
    (∀ x y : ℚ, (⟨3, 7⟩ : ℚ × ℚ) • ((⟨x, y⟩ : ℚ × ℚ) - ⟨-2, 8⟩) = 0 → y = m * x + b) 
    ∧ m = -3 / 7 ∧ b = 50 / 7 :=
begin
  sorry
end

end line_eqn_in_slope_intercept_form_l480_480489


namespace number_of_solutions_l480_480973

open Real

noncomputable def T (x : ℝ) : ℝ := tan x - sin x

lemma strictly_increasing_on_0_pi_div_2 : strict_mono (λ x, T x) :=
begin
  intros x y hx,
  sorry
end

lemma T_tends_to_infinity_as_x_tends_to_pi_div_2 : tendsto T (𝓝[<] (π / 2)) at_top :=
begin
  sorry
end

lemma unique_x_for_each_npi (n : ℕ) : ∃! x, T x = n * π :=
begin
  sorry
end

lemma T_value_at_arctan_1000 : T (arctan 1000) ≈ 999 :=
begin
  -- Precise formulation and proof here.
  sorry
end

theorem number_of_solutions : ∃ n : ℕ, has_solution_set_size (tan x = tan (sin x)) (set.Icc 0 (arctan 1000)) (319) :=
begin
  -- Combining the facts from the lemmas
  sorry
end

end number_of_solutions_l480_480973


namespace area_of_triangle_l480_480496

variable {a1 a2 a3 : ℝ}

theorem area_of_triangle (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) :
  let area := (1/2) * Real.sqrt (a1^2 * a2^2 + a2^2 * a3^2 + a3^2 * a1^2) in
  ∃ A1 A2 A3 : ℝ^3, 
    dist (0, 0, 0) A1 = a1 ∧ 
    dist (0, 0, 0) A2 = a2 ∧
    dist (0, 0, 0) A3 = a3 ∧
    ∃ plane : ℝ^3 → Prop, 
      plane A1 ∧ plane A2 ∧ plane A3 ∧ 
      -- Plane intersects at vertex
      plane (0, 0, 0) ∧
      -- The area of the triangle formed by A1, A2, and A3 is given by the expression "area"
      dist A1 A2 = Real.sqrt (a1^2 + a2^2) ∧ 
      dist A2 A3 = Real.sqrt (a2^2 + a3^2) ∧ 
      dist A3 A1 = Real.sqrt (a3^2 + a1^2) ∧
      Triangle.area A1 A2 A3 = area := 
sorry

end area_of_triangle_l480_480496


namespace odd_function_ln_l480_480200

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_ln (x : ℝ) (hx : x < 0) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, 0 ≤ x -> f x = Real.log (x + 1)) →
  f x = -Real.log(-x + 1) :=
by
  sorry

end odd_function_ln_l480_480200


namespace record_expenditure_l480_480642

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end record_expenditure_l480_480642


namespace no_five_correct_letters_l480_480793

theorem no_five_correct_letters (n : ℕ) (hn : n = 6) :
  ∀ (σ : Fin n → Fin n), (∑ i, if σ i = i then 1 else 0) ≠ 5 :=
by
  simp only
  sorry

end no_five_correct_letters_l480_480793


namespace find_cost_price_l480_480421

/-- Define the given conditions -/
def selling_price : ℝ := 100
def profit_percentage : ℝ := 0.15
def cost_price : ℝ := 86.96

/-- Define the relationship between selling price and cost price -/
def relation (CP SP : ℝ) : Prop := SP = CP * (1 + profit_percentage)

/-- State the theorem based on the conditions and required proof -/
theorem find_cost_price 
  (SP : ℝ) (CP : ℝ) 
  (h1 : SP = selling_price) 
  (h2 : relation CP SP) : 
  CP = cost_price := 
by
  sorry

end find_cost_price_l480_480421


namespace crayons_per_pack_l480_480320

theorem crayons_per_pack (total_crayons : ℕ) (packs : ℕ) (crayons_per_pack : ℕ) : 
  total_crayons = 615 ∧ packs = 41 → crayons_per_pack = 15 :=
by
  intro h
  cases h with hc hp
  sorry

end crayons_per_pack_l480_480320


namespace crayons_per_pack_l480_480319

theorem crayons_per_pack (total_crayons : ℕ) (packs : ℕ) (crayons_per_pack : ℕ) : 
  total_crayons = 615 ∧ packs = 41 → crayons_per_pack = 15 :=
by
  intro h
  cases h with hc hp
  sorry

end crayons_per_pack_l480_480319


namespace perpendicular_tangent_line_l480_480146

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end perpendicular_tangent_line_l480_480146


namespace rabbit_total_apples_90_l480_480160

-- Define the number of apples each animal places in a basket
def rabbit_apple_per_basket : ℕ := 5
def deer_apple_per_basket : ℕ := 6

-- Define the number of baskets each animal uses
variable (h_r h_d : ℕ)

-- Define the total number of apples collected by both animals
def total_apples : ℕ := rabbit_apple_per_basket * h_r

-- Conditions
axiom deer_basket_count_eq_rabbit : h_d = h_r - 3
axiom same_total_apples : total_apples = deer_apple_per_basket * h_d

-- Goal: Prove that the total number of apples the rabbit collected is 90
theorem rabbit_total_apples_90 : total_apples = 90 := sorry

end rabbit_total_apples_90_l480_480160


namespace trapezoid_area_difference_l480_480689

theorem trapezoid_area_difference (T : Trapezoid) (h : T.hasRightAngles) 
  (sides : T.hasSides [4, 4, 5, real.sqrt 17]) :
  240 * (areaDifference T) = 240 := 
sorry

end trapezoid_area_difference_l480_480689


namespace prime_divides_sub_one_l480_480289

theorem prime_divides_sub_one (n p q : ℕ) (hn : n > 0) (hp : p.prime) (hq : q ∣ ((n + 1)^p - n^p)) :
  p ∣ (q - 1) :=
sorry

end prime_divides_sub_one_l480_480289


namespace platinum_matrix_for_all_n_geq_3_l480_480874

noncomputable def platinum_matrix_exists (n : ℕ) : Prop :=
  ∃ (M : matrix (fin n) (fin n) ℕ),
  (∀ i j, 1 ≤ M i j ∧ M i j ≤ n) ∧
  (∀ i, (finset.univ).image (M i) = finset.range n.succ) ∧
  (∀ j, (finset.univ).image (M (λ i, j)) = finset.range n.succ) ∧
  (finset.univ.image (λ k, M k k) = finset.range n.succ) ∧
  ∃ (S : finset (fin n × fin n)),
  (S.card = n) ∧
  (∀ (i1 i2 : fin n) (j1 j2 : fin n), 
    (i1 ≠ i2 → (i1, j1) ∈ S → (i2, j2) ∈ S → j1 ≠ j2) ∧
    (i1 ≠ i2 → (i1, j1) ∈ S → (i2, j2) ∈ S → S.disjoint (finset.univ.image (λ x, (x, x))) (S))) 
  
theorem platinum_matrix_for_all_n_geq_3 : ∀ n : ℕ, n ≥ 3 → platinum_matrix_exists n :=
sorry

end platinum_matrix_for_all_n_geq_3_l480_480874


namespace distinct_logarithmic_values_l480_480250

open Finset

theorem distinct_logarithmic_values :
  let S := {1, 2, 3, 4, 5, 6};
  let distinct_logs := 
    (S \ {1}).card * (S \ {1}).card + (S.card - 1);
  distinct_logs = 21 :=
by
  let S := {1, 2, 3, 4, 5, 6};
  let bases := S \ {1};
  have card_bases : bases.card = 5 := by sorry;
  let num_distinct_logs := bases.card * (S.card - 1) + 1;
  show num_distinct_logs = 21, from sorry

end distinct_logarithmic_values_l480_480250


namespace find_q_l480_480640

theorem find_q (p q : ℝ) (h : (-2)^3 - 2*(-2)^2 + p*(-2) + q = 0) : 
  q = 16 + 2 * p :=
sorry

end find_q_l480_480640


namespace greatest_n_l480_480991

open Nat

def sum_squares (a b : ℕ) : ℕ :=
  (b * (b + 1) * (2 * b + 1)) / 6 - (a * (a + 1) * (2 * a + 1)) / 6

theorem greatest_n (n : ℕ) : 
  n ≤ 2022 → 
  isPerfectCube ((sum_squares 0 n) * (sum_squares n (2 * n))) → 
  n = 0 :=
by
  sorry

-- Auxiliary definitions (if needed)
def isPerfectCube (x : ℕ) : Prop := 
  ∃ (k : ℕ), k^3 = x

end greatest_n_l480_480991


namespace max_prime_count_after_appending_digits_l480_480707

theorem max_prime_count_after_appending_digits (N : ℕ) :
  ∃ L : list ℕ, (∀ x ∈ L, prime x) ∧ (∀ m ∈ L, m > N) ∧ (L.length = 6) ∧ (∀ d ∈ (list.range 1 10), ∃ k, k ∈ L ∧ (d::(N.to_string.to_list) = k.to_string.to_list ∨ (N.to_string.to_list ++ [char.of_nat (d + 48)] = k.to_string.to_list))) :=
sorry

end max_prime_count_after_appending_digits_l480_480707
