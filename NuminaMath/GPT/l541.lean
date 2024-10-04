import Mathlib

namespace inequality_holds_for_all_x_iff_l541_541610

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end inequality_holds_for_all_x_iff_l541_541610


namespace main_octahedron_problem_conjecture_l541_541359

-- Define the context and the problem statement
noncomputable def solve_octahedron_problem : Prop :=
  let V := {v : ℝ³ // is_vertex_of_octahedron v} in
  let E := {e : ℝ³ × ℝ³ // is_edge_of_octahedron e} in
  let k := (1 + sqrt 5) / 2 in
  let p (e : ℝ³ × ℝ³) : ℝ³ := divide_edge e 1 k in
  (∃ select : (ℝ³ × ℝ³) → ℝ³,
    (∀ e ∈ E, select e = p e) ∧
    (∀ v ∈ V, ∀ e1 e2 e3 e4 ∈ edges_from v,
      alternating_selection_rule select [e1, e2, e3, e4])) ∧
  (selected_points_form_regular_icosahedron (image select E))

-- Main conjecture stating the problem
theorem main_octahedron_problem_conjecture : solve_octahedron_problem :=
sorry

end main_octahedron_problem_conjecture_l541_541359


namespace roots_of_quadratic_eq_l541_541755

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l541_541755


namespace division_remainder_l541_541259

theorem division_remainder (dividend divisor quotient : ℕ) (h_dividend : dividend = 131) (h_divisor : divisor = 14) (h_quotient : quotient = 9) :
  ∃ remainder : ℕ, dividend = divisor * quotient + remainder ∧ remainder = 5 :=
by
  sorry

end division_remainder_l541_541259


namespace book_distribution_valid_l541_541449

-- Define the given conditions
def distinct_stacks (s : Finset ℕ) : Prop :=
  ∀ a b ∈ s, (a ≠ b) → a ≠ b

noncomputable def book_distribution : Finset ℕ := 
{1, 3, 5, 7, 9, 11, 13, 15, 17, 19}

theorem book_distribution_valid :
  (∑ x in book_distribution, x = 100) ∧
  (distinct_stacks book_distribution) ∧ 
  (∀ a ∈ book_distribution, ∀ b := a // 2, (b ∉ book_distribution) → (∃ c ∈ book_distribution, c = b)) :=
by
  sorry

end book_distribution_valid_l541_541449


namespace red_segments_leq_half_l541_541778

theorem red_segments_leq_half :
  ∀ (red_points : Set ℝ), 
    (∀ x y ∈ red_points, x ≠ y → |x - y| ≥ 0.1) → 
    (∀ I ∈ (set.range (λ k : Fin 10, Icc (k.1 / 10) ((k.1 + 1) / 10))), ∃ x, x ∈ red_points) →
    (∑ i in Finset.range 10, ∑ (x ∈ Icc (i / 10) ((i + 1) / 10)), indicator_fun red_points x) ≤ 0.5 :=
by sorry

end red_segments_leq_half_l541_541778


namespace larger_number_is_33_l541_541363

theorem larger_number_is_33 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : max x y = 33 :=
sorry

end larger_number_is_33_l541_541363


namespace circle_area_ratio_l541_541831

theorem circle_area_ratio 
  (WXYZ : set (ℝ × ℝ))    -- Assume WXYZ is a square 
  (side_length : ℝ)       -- Assume side length of square is 2
  (smaller_circle_center : ℝ × ℝ) -- Center of the smaller circle
  (larger_circle_center : ℝ × ℝ)  -- Center of the larger circle
  (smaller_circle_radius larger_circle_radius : ℝ) -- radii of two circles
  (tangent_midpoint_WZ : smaller_circle_center.2 = 0 ∧ abs(smaller_circle_center.1) = 1) -- smaller circle tangent at midpoint of WZ
  (tangent_midpoint_XY : larger_circle_center.2 = 2 ∧ abs(larger_circle_center.1) = 1) -- larger circle tangent at midpoint of XY
  (tangent_vertex_W : dist (0, 0) smaller_circle_center = smaller_circle_radius)   -- smaller circle tangent at vertex W
  (tangent_vertex_Y : dist (2, 2) larger_circle_center = larger_circle_radius )      -- larger circle tangent at vertex Y
  (areas : π * smaller_circle_radius^2 = π * larger_circle_radius^2) :     -- areas of circles
  π * larger_circle_radius^2 / π * smaller_circle_radius^2 = 1 :=    -- the ratio of the areas
sorry  -- the proof is skipped for this exercise.

end circle_area_ratio_l541_541831


namespace square_fits_in_unit_cube_l541_541867

theorem square_fits_in_unit_cube (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let PQ := Real.sqrt (2 * (1 - x) ^ 2)
  let PS := Real.sqrt (1 + 2 * x ^ 2)
  (PQ > 1.05 ∧ PS > 1.05) :=
by
  sorry

end square_fits_in_unit_cube_l541_541867


namespace sum_of_vectors_l541_541878

noncomputable def vector_u0 : ℝ × ℝ := (2, 1)
noncomputable def vector_z0 : ℝ × ℝ := (3, 2)

noncomputable def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1 ^ 2 + v.2 ^ 2

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let factor := (dot_product v w) / (norm_sq w)
  (factor * w.1, factor * w.2)

noncomputable def vector_u (n : ℕ) : ℝ × ℝ :=
  if n = 0 then
    vector_u0
  else
    let prev_z := (vector_z (n - 1))
    projection prev_z vector_u0

noncomputable def vector_z (n : ℕ) : ℝ × ℝ :=
  if n = 0 then
    vector_z0
  else
    let prev_u := (vector_u n)
    projection prev_u vector_z0

noncomputable def infinite_sum (n : ℕ) : ℝ × ℝ :=
  let geometric_sum (k : ℕ) : ℝ :=
    if k = 0 then 0 else ((8:ℝ) / 5) ^ k
  let sum_x := ∑ k in Finset.range (n + 1), (geometric_sum k) * (vector_u0.1)
  let sum_y := ∑ k in Finset.range (n + 1), (geometric_sum k) * (vector_u0.2)
  (sum_x, sum_y)

theorem sum_of_vectors :
  infinite_sum ∞ = (-16 / 3, -8 / 3) :=
sorry

end sum_of_vectors_l541_541878


namespace proof_problem_l541_541023

-- Define the mathematical objects and relationships in Lean.
def regularTruncatedPyramidCircumscribedAroundSphere
    (n : ℕ)
    (S1 S2 S σ : ℝ)
    (cos_n : ℝ) : Prop :=
  σ * S = 4 * S1 * S2 * cos_n^2

-- Define the conditions of the problem.
variables (n : ℕ)
variables (S1 S2 S σ : ℝ)
variables (cos_n : ℝ)

-- Key condition: cos_n should be the cosine of π/n
def condition_cos (n : ℕ) : Prop :=
  cos_n = Real.cos (π / n)

-- Statement of the proof problem in Lean.
theorem proof_problem 
  (h_cos : condition_cos n)
  : regularTruncatedPyramidCircumscribedAroundSphere n S1 S2 S σ cos_n := 
  by
  unfold regularTruncatedPyramidCircumscribedAroundSphere
  rw h_cos
  sorry  -- Proof steps go here.

end proof_problem_l541_541023


namespace find_n_such_that_sequence_is_product_of_two_primes_l541_541128

def is_product_of_two_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q

def sequence (n : ℕ) : ℕ :=
  2 * n + 49

theorem find_n_such_that_sequence_is_product_of_two_primes :
  ∃ n : ℕ, is_product_of_two_distinct_primes (sequence n) ∧ 
           is_product_of_two_distinct_primes (sequence (n + 1)) :=
sorry

end find_n_such_that_sequence_is_product_of_two_primes_l541_541128


namespace second_derivative_at_0_l541_541090

def f (x : ℝ) : ℝ := sin (x + exp x)

theorem second_derivative_at_0 :
  (deriv (deriv f) 0) = 2 :=
by
  sorry

end second_derivative_at_0_l541_541090


namespace gcd_three_numbers_l541_541080

theorem gcd_three_numbers :
  gcd (gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_three_numbers_l541_541080


namespace last_score_avg_integer_l541_541155

theorem last_score_avg_integer :
  ∃ last : ℕ, (last = 65 ∨ last = 85) ∧
    (∀ (scores : List ℕ), scores = [65, 70, 75, 85, 90, last] →
      (∀ n (prefix : List ℕ), prefix.length = n → (∑ i in prefix, i) % n = 0)) :=
sorry

end last_score_avg_integer_l541_541155


namespace perpendicular_bisector_l541_541471

theorem perpendicular_bisector (x y : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (h_line : x - 2 * y + 1 = 0) : 
  2 * x - y - 1 = 0 :=
sorry

end perpendicular_bisector_l541_541471


namespace binary_to_decimal_and_octal_l541_541875

theorem binary_to_decimal_and_octal (b : ℕ) (h : b = 1011001) : 
  let decimal_value := 89 ∧ let octal_value := 131 := 
  b = 1011001 → 
  (let decimal := 1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 in 
  decimal = 89) ∧ 
  (let decimal := 89 in 
  let octal := (decimal / 64) * 100 + (decimal % 64 / 8) * 10 + (decimal % 64 % 8) in
  octal = 131) :=
by
  sorry

end binary_to_decimal_and_octal_l541_541875


namespace max_tetrahedra_l541_541589

-- Conditions definition
def non_coincident (α β : Prop) : Prop := α ≠ β
def chosen_points (α β : Type) (points_α : Finset α) (points_β : Finset β) : Prop :=
  points_α.card = 5 ∧ points_β.card = 4

-- Theorem statement
theorem max_tetrahedra {α β : Type} (h_non_coincident : non_coincident α β)
  (points_α : Finset α) (points_β : Finset β) (h_chosen_points : chosen_points α β points_α points_β) :
  let total_points := points_α ∪ points_β in
  ∑ {
    (comb_α : Finset α) (comb_β : Finset β) | 
    (comb_α.card + comb_β.card = 4) ∧ 
    (comb_α ⊆ points_α) ∧ 
    (comb_β ⊆ points_β) }, 
    1 = 120 := 
  sorry

end max_tetrahedra_l541_541589


namespace exponential_vs_power_vs_logarithmic_l541_541538

open Real -- Use the Real number space

theorem exponential_vs_power_vs_logarithmic (a : ℝ) (h : a > 1) :
  ∃ x₀ : ℝ, ∀ x : ℝ, x > x₀ → a^x > x^a ∧ x^a > log a x :=
sorry -- Proof goes here

end exponential_vs_power_vs_logarithmic_l541_541538


namespace find_inverse_f_range_a_l541_541123

noncomputable def f (x : ℝ) : ℝ := ( (x + 1) / x ) ^ 2

def inverse_f (y : ℝ) : ℝ := 1 / (Real.sqrt y - 1)

theorem find_inverse_f (x : ℝ) (hx : x ∈ Set.Ioi (0 : ℝ)) :
  Function.LeftInverse inverse_f f :=
by
  sorry
  
theorem range_a (a : ℝ) :
  -1 < a ∧ a < 1 + Real.sqrt 2 ↔ ∀ x ≥ 2, 
  (x - 1) * inverse_f x > a * (a - Real.sqrt x) :=
by
  sorry

end find_inverse_f_range_a_l541_541123


namespace marys_garbage_bill_l541_541691

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l541_541691


namespace coefficient_of_x_l541_541426

theorem coefficient_of_x : 
  let expr := 5 * (x - 6) + 3 * (9 - 3 * x^2 + 2 * x) - 10 * (3 * x - 2)
  in coefficient_of_x_in_expr expr = -19 :=
by
  sorry

end coefficient_of_x_l541_541426


namespace ratio_Rahul_Deepak_l541_541269

variable (R D : ℕ) -- Rahul's present age and Deepak's present age

-- Condition 1: Deepak's present age is 8 years.
def Deepak_age_8 : Prop := D = 8

-- Condition 2: After 10 years, Rahul's age will be 26 years.
def Rahul_future_age_26 : Prop := R + 10 = 26

-- Theorem: The ratio between Rahul and Deepak's ages is 2:1.
theorem ratio_Rahul_Deepak (h1 : Deepak_age_8) (h2 : Rahul_future_age_26) : R / D = 2 := by
  sorry

end ratio_Rahul_Deepak_l541_541269


namespace pass_percentage_set2_l541_541364

-- Conditions
def total_students := 40 + 50 + 60
def pass_percentage_set1 := 100 / 100
def pass_percentage_set3 := 80 / 100
def overall_pass_percentage := 88.66666666666667 / 100

-- Definition of the proof problem
theorem pass_percentage_set2 :
  let pass_students1 := 40 * pass_percentage_set1 in
  let pass_students3 := 60 * pass_percentage_set3 in
  let total_pass_students := overall_pass_percentage * total_students in
  let pass_students2 := total_pass_students - pass_students1 - pass_students3 in
  (pass_students2 / 50) * 100 = 90 :=
by
  sorry

end pass_percentage_set2_l541_541364


namespace correct_statements_selection_l541_541768

theorem correct_statements_selection :
  let population := 10000
  let sample_size := 200
  let selected_statements :=
    [ (1, "The math scores of these 10,000 candidates are the population.")
    , (2, "Each candidate is an individual.")
    , (3, "The math scores of the 200 candidates selected are a sample of the population.")
    , (4, "The sample size is 200.") ]
  let correct_statements :=
    [ (1, true)
    , (2, false)
    , (3, true)
    , (4, true) ]
  (selected_statements.filter (λ s => correct_statements.any (λ c => s.fst = c.fst ∧ c.snd))).map Prod.fst = [1, 3] :=
by
  let population := 10000
  let sample_size := 200
  let selected_statements :=
    [ (1, "The math scores of these 10,000 candidates are the population.")
    , (2, "Each candidate is an individual.")
    , (3, "The math scores of the 200 candidates selected are a sample of the population.")
    , (4, "The sample size is 200.") ]
  let correct_statements :=
    [ (1, true)
    , (2, false)
    , (3, true)
    , (4, true) ]
  
  have h1 : selected_statements[0].2 := "The math scores of these 10,000 candidates are the population."
  have h2 : selected_statements[1].2 := "Each candidate is an individual."
  have h3 : selected_statements[2].2 := "The math scores of the 200 candidates selected are a sample of the population."
  have h4 : selected_statements[3].2 := "The sample size is 200."
  
  have cs1 : correct_statements[0].2 = true := rfl
  have cs2 : correct_statements[1].2 = false := rfl
  have cs3 : correct_statements[2].2 = true := rfl
  have cs4 : correct_statements[3].2 = true := rfl
  
  simp only [List.filter, Prod.fst, List.mem, IEnumerable.map]
  
  have result := [selected_statements[0].1, selected_statements[2].1]
  exact result


end correct_statements_selection_l541_541768


namespace original_length_l541_541369

-- Definitions based on conditions
def length_sawed_off : ℝ := 0.33
def remaining_length : ℝ := 0.08

-- The problem statement translated to a Lean 4 theorem
theorem original_length (L : ℝ) (h1 : L = length_sawed_off + remaining_length) : 
  L = 0.41 :=
by
  sorry

end original_length_l541_541369


namespace prop_A_prop_B_prop_C_prop_D_l541_541346

-- Proposition A
theorem prop_A (a b : ℝ) : a > b → ¬ (1 / b > 1 / a) :=
begin
  -- Intentionally left blank for the proof
  sorry
end

-- Proposition B
theorem prop_B (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
begin
  -- Intentionally left blank for the proof
  sorry
end

-- Proposition C
theorem prop_C (a b c : ℝ) (h1 : b ≠ 0) (h2 : a * c^2 > b * c^2) : a > b :=
begin
  -- Intentionally left blank for the proof
  sorry
end

-- Proposition D
theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 1 / (a - c) < 1 / (b - d) :=
begin
  -- Intentionally left blank for the proof
  sorry
end

end prop_A_prop_B_prop_C_prop_D_l541_541346


namespace number_of_correct_propositions_l541_541990

def proposition1 : Prop := ∀ (boxes : Fin 2 → List ℕ), (∃ (balls: Fin 3 → Fin 2), ∀ i, boxes i).sum > 1
def proposition2 : Prop := ¬ (∃ (x : ℝ), x * x < 0)
def proposition3 : Prop := False -- The proposition about weather is false as given.
def proposition4 : Prop := ∃ (selection : Fin 100 → Bool), selection.count (λ b, b = true) = 5 ∧ 
    ∀ i, if i < 90 then selection i = false else true -- Simplified for illustrative purposes

def is_correct (p : Prop) : bool := p

theorem number_of_correct_propositions : 
  is_correct proposition1 ∧
  is_correct proposition2 ∧
  ¬ is_correct proposition3 ∧
  is_correct proposition4 → 
  Nat := 3 :=
    by sorry

end number_of_correct_propositions_l541_541990


namespace marys_garbage_bill_l541_541693

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l541_541693


namespace trajectory_is_line_segment_l541_541215

structure Point where
  x : ℝ
  y : ℝ

def dist (P Q : Point) : ℝ := real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

theorem trajectory_is_line_segment :
  ∀ (M F1 F2 : Point), F1.x = -4 ∧ F1.y = 0 ∧ F2.x = 4 ∧ F2.y = 0 →
  dist M F1 + dist M F2 = 8 →
  ∃ a b c : ℝ, ∀ M : Point, a * M.x + b * M.y + c = 0 :=
by
  intros M F1 F2 hF hf
  sorry

end trajectory_is_line_segment_l541_541215


namespace arithmetic_sequence_s10_l541_541982

noncomputable def arithmetic_sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_s10 (a : ℤ) (d : ℤ)
  (h1 : a + (a + 8 * d) = 18)
  (h4 : a + 3 * d = 7) :
  arithmetic_sequence_sum 10 a d = 100 :=
by sorry

end arithmetic_sequence_s10_l541_541982


namespace equation1_solutions_equation2_solutions_l541_541729

theorem equation1_solutions (x : ℝ) : 3 * x^2 - 6 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

theorem equation2_solutions (x : ℝ) : x^2 + 4 * x - 1 = 0 ↔ (x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end equation1_solutions_equation2_solutions_l541_541729


namespace geometric_sequence_common_ratio_l541_541558

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 5/4) 
  (h_sequence : ∀ n, a n = a 1 * q ^ (n - 1)) : 
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l541_541558


namespace largest_n_dividing_30_factorial_l541_541485

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541485


namespace pos_integers_difference_count_l541_541136

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def B : Set ℕ := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

theorem pos_integers_difference_count : 
  {d : ℕ | ∃ a ∈ A, ∃ b ∈ B, d = b - a ∧ b ≠ a}.card = 14 :=
by
  sorry

end pos_integers_difference_count_l541_541136


namespace no_positive_integer_n_for_perfect_squares_l541_541711

theorem no_positive_integer_n_for_perfect_squares :
  ∀ (n : ℕ), 0 < n → ¬ (∃ a b : ℤ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end no_positive_integer_n_for_perfect_squares_l541_541711


namespace extreme_point_at_one_l541_541603

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l541_541603


namespace hallway_area_l541_541347

theorem hallway_area :
  let starting_point := (3, 0)
  let end_point := (3, 6)
  let allowable_movement := λ (p : ℝ × ℝ), (dist p end_point <= dist starting_point end_point)
  ∃ S : set (ℝ × ℝ), 
    (∀ p ∈ S, allowable_movement p) ∧
    (9 * Real.sqrt 3 + 21 * Real.pi / 2 = measure_theory.measure_of S volume) :=
begin
  sorry
end

end hallway_area_l541_541347


namespace ellipse_equation_line_equation_l541_541872

-- Definitions to make sure the conditions are captured
def ellipse_eqn (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def passes_through (x y : ℝ) (p q : (ℝ × ℝ)) := (x = p) ∧ (y = q)
def eccentricity (a b c : ℝ) : Prop := c / a = 1 / 2 ∧ b^2 = a^2 - c^2
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ := 1 / 2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

noncomputable def foci_distance (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

-- Theorem to prove the ellipse equation
theorem ellipse_equation 
  (a b : ℝ) (h : a > b > 0) 
  (p q : ℝ × ℝ) (h_pq : p = (1, 3 / 2))
  (h_ecc : eccentricity a b (foci_distance a b)) :
  ellipse_eqn 1 (3 / 2) a b := 
sorry

-- Theorem to prove the line equation when area of triangle F2AB is given
theorem line_equation 
  (a b : ℝ) (h : a > b > 0) 
  (p : ℝ × ℝ) (h_p : p = (1, 3 / 2)) 
  (h_ecc : eccentricity a b (foci_distance a b))
  (h_area : triangle_area (-a) 0 (-1) (k * (-1) + 1) (k * (-1) + 1) (1) = 12 * sqrt 2 / 7) :
  (line_eqn : (x - y + 1 = 0) ∨ (x + y + 1 = 0)) := 
sorry

end ellipse_equation_line_equation_l541_541872


namespace find_a_b_l541_541227

theorem find_a_b (a b : ℚ) (h : (1 + real.sqrt 3)^2 + a * (1 + real.sqrt 3) + b = 0) : 
a = -2 ∧ b = -2 :=
sorry

end find_a_b_l541_541227


namespace investment_total_amount_l541_541037

theorem investment_total_amount (P : ℕ) (r : ℝ) (d : ℕ) (n : ℕ) (A : ℝ) :
    P = 12000 → r = 0.04 → d = 500 → n = 4 →
    A = P * (1 + r)^n + d * ((1 + r)^n - 1) / r →
    A ≈ 16162 :=
by
  intros hP hr hd hn hA
  sorry

end investment_total_amount_l541_541037


namespace quadratic_roots_unique_l541_541913

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l541_541913


namespace red_packet_grabbing_situations_l541_541151

-- Definitions based on the conditions
def numberOfPeople := 5
def numberOfPackets := 4
def packets := [2, 2, 3, 5]  -- 2-yuan, 2-yuan, 3-yuan, 5-yuan

-- Main theorem statement
theorem red_packet_grabbing_situations : 
  ∃ situations : ℕ, situations = 60 :=
by
  sorry

end red_packet_grabbing_situations_l541_541151


namespace fraction_of_number_l541_541303

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541303


namespace coin_experiment_frequency_probability_l541_541161

theorem coin_experiment_frequency_probability (fair_coin : Prop)
  (num_trials : ℕ) (heads_count : ℕ)
  (H_trials : num_trials = 100)
  (H_heads : heads_count = 48)
  (H_fair_coin : fair_coin) :
  (heads_count / num_trials : ℝ) = 0.48 ∧ (0.5 : ℝ) = 0.5 :=
by {
  have H1 : (heads_count : ℝ) / (num_trials : ℝ) = 0.48,
  { sorry },
  have H2 : (0.5 : ℝ) = 0.5,
  { sorry },
  exact ⟨H1, H2⟩
}

end coin_experiment_frequency_probability_l541_541161


namespace ball_distance_traveled_l541_541805

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  n * (a1 + a1 + (n-1) * d) / 2

theorem ball_distance_traveled : 
  total_distance 8 5 20 = 1110 :=
by
  sorry

end ball_distance_traveled_l541_541805


namespace evaluate_expression_l541_541071

theorem evaluate_expression :
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  (sum1 / sum2 - sum2 / sum1) = 11 / 30 :=
by
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  sorry

end evaluate_expression_l541_541071


namespace sequence_formula_l541_541585

open Nat

noncomputable def S : ℕ → ℤ
| n => n^2 - 2 * n + 2

noncomputable def a : ℕ → ℤ
| 0 => 1  -- note that in Lean, sequence indexing starts from 0
| (n+1) => 2*(n+1) - 3

theorem sequence_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 2*n - 3 := by
  sorry

end sequence_formula_l541_541585


namespace alcohol_added_l541_541367

theorem alcohol_added (x : ℝ) :
  let initial_solution_volume := 40
  let initial_alcohol_percentage := 0.05
  let initial_alcohol_volume := initial_solution_volume * initial_alcohol_percentage
  let additional_water := 6.5
  let final_solution_volume := initial_solution_volume + x + additional_water
  let final_alcohol_percentage := 0.11
  let final_alcohol_volume := final_solution_volume * final_alcohol_percentage
  initial_alcohol_volume + x = final_alcohol_volume → x = 3.5 :=
by
  intros
  sorry

end alcohol_added_l541_541367


namespace polynomial_inequality_solution_l541_541911

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 + x^3 - 10 * x^2 + 25 * x > 0 ↔ x > 0 :=
sorry

end polynomial_inequality_solution_l541_541911


namespace first_candidate_votes_percentage_l541_541160

theorem first_candidate_votes_percentage 
( total_votes : ℕ ) 
( second_candidate_votes : ℕ ) 
( P : ℕ ) 
( h1 : total_votes = 2400 ) 
( h2 : second_candidate_votes = 480 ) 
( h3 : (P/100 : ℝ) * total_votes + second_candidate_votes = total_votes ) : 
  P = 80 := 
sorry

end first_candidate_votes_percentage_l541_541160


namespace largest_possible_rational_root_l541_541781

noncomputable def rational_root_problem : Prop :=
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧
  ∀ p q : ℤ, (q ≠ 0) → (a * p^2 + b * p + c * q = 0) → 
  (p / q) ≤ -1 / 99

theorem largest_possible_rational_root : rational_root_problem :=
sorry

end largest_possible_rational_root_l541_541781


namespace solution_amount_of_solution_A_l541_541385

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x + y = 140)
variables (h2 : 0.40 * x + 0.90 * y = 0.80 * 140)

-- State the theorem
theorem solution_amount_of_solution_A : x = 28 :=
by
  -- Here, the proof would be provided, but we replace it with sorry
  sorry

end solution_amount_of_solution_A_l541_541385


namespace Rahim_books_l541_541715

variable (x : ℕ) (total_cost : ℕ) (total_books : ℕ) (average_price : ℚ)

def book_problem_conditions : Prop :=
  total_cost = 520 + 248 ∧
  total_books = x + 22 ∧
  average_price = 12

theorem Rahim_books (h : book_problem_conditions x total_cost total_books average_price):
  x = 42 :=
by
  sorry

end Rahim_books_l541_541715


namespace number_of_possible_values_of_b_l541_541733

theorem number_of_possible_values_of_b :
  (∃ b : ℕ, 5 ∣ b ∧ b ∣ 30 ∧ b > 0 ∧ b ∣ 20) →
  {b // 5 ∣ b ∧ b ∣ 30 ∧ b > 0 ∧ b ∣ 20}.card = 2 :=
by sorry

end number_of_possible_values_of_b_l541_541733


namespace find_a_l541_541261

-- Definition of the curve y = x^3 + ax + 1
def curve (x a : ℝ) : ℝ := x^3 + a * x + 1

-- Definition of the tangent line y = 2x + 1
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- The slope of the tangent line is 2
def slope_of_tangent_line (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem find_a (a : ℝ) : 
  (∃ x₀, curve x₀ a = tangent_line x₀) ∧ (∃ x₀, slope_of_tangent_line x₀ a = 2) → a = 2 :=
by
  sorry

end find_a_l541_541261


namespace gcd_binary_conversion_l541_541446

theorem gcd_binary_conversion (a b : ℕ) (ha : a = 3869) (hb : b = 6497) :
  (Nat.gcd a b).toString (2) = "1001001" := by
  sorry

end gcd_binary_conversion_l541_541446


namespace infinite_a_exists_l541_541528

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ), ∀ (k : ℕ), ∃ (a : ℕ), n^6 + 3 * a = (n^2 + 3 * k)^3 := 
sorry

end infinite_a_exists_l541_541528


namespace fraction_multiplication_l541_541305

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541305


namespace area_ABCD_eq_750_l541_541713

-- Define the quadrilateral and its properties
variable (A B C D E : Point)
variable (AC : Line)
variable (BD : Line)
variable (AB BC CD AE : Real)

-- Define the angles and dimensions given in the problem
axiom H1 : ∠ABC = 90
axiom H2 : ∠ACD = 90
axiom H3 : dist A C = 25
axiom H4 : dist C D = 40
axiom H5 : ae_on AC BD E
axiom H6 : dist A E = 10

-- Define the area function
noncomputable def area_quadrilateral (A B C D : Point) : Real := sorry

-- State the theorem
theorem area_ABCD_eq_750 : area_quadrilateral A B C D = 750 := by
  sorry

end area_ABCD_eq_750_l541_541713


namespace project_completion_l541_541087

theorem project_completion (a b : ℕ) (h1 : 3 * (1 / b : ℚ) + (1 / a : ℚ) + (1 / b : ℚ) = 1) : 
  a + b = 9 ∨ a + b = 10 :=
sorry

end project_completion_l541_541087


namespace calculate_expression_l541_541054

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l541_541054


namespace s_at_7_l541_541821

variables (q s : ℤ → ℤ)

-- Given conditions:
axiom h1 : q 3 = 2
axiom h2 : q 4 = -4
axiom h3 : q (-2) = 5

-- Definition of s(x) and statement to be proved
noncomputable def s (x : ℤ) : ℤ := -x^2 + x + 8

theorem s_at_7 : s 7 = -34 := by
  -- Proof is skipped with sorry
  sorry

end s_at_7_l541_541821


namespace problem_arithmetic_sequences_l541_541532

variable {a b : ℕ → ℝ} -- Assuming sequences a and b are defined as functions from natural numbers to real numbers.

-- Definitions for the sum of the first n terms of the sequences
def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i
def T (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), b i

-- Given condition
axiom condition_Sn_Tn (n : ℕ) : S n / T n = (n + 3) / (2 * n + 1)

-- Correct answer to prove
theorem problem_arithmetic_sequences : a 6 / b 6 = 14 / 23 :=
by
  sorry

end problem_arithmetic_sequences_l541_541532


namespace probability_approx_l541_541832

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

noncomputable def single_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R / 3)^3

noncomputable def total_spheres_volume (R : ℝ) : ℝ :=
  6 * single_sphere_volume R

noncomputable def probability_inside_spheres (R : ℝ) : ℝ :=
  total_spheres_volume R / circumscribed_sphere_volume R

theorem probability_approx (R : ℝ) (hR : R > 0) : 
  abs (probability_inside_spheres R - 0.053) < 0.001 := sorry

end probability_approx_l541_541832


namespace extreme_point_a_zero_l541_541605

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l541_541605


namespace sqrt_neg2023_squared_l541_541865

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end sqrt_neg2023_squared_l541_541865


namespace sum_of_primes_less_than_20_l541_541336

theorem sum_of_primes_less_than_20 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19] in
  (List.sum primes = 77) ∧ (2 ∈ primes) :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  have sum_eq_77 : List.sum primes = 77 := sorry
  have two_in_primes : 2 ∈ primes := sorry
  exact ⟨sum_eq_77, two_in_primes⟩

end sum_of_primes_less_than_20_l541_541336


namespace angle_between_a_and_b_is_150_degrees_l541_541212

variables {V : Type*} [inner_product_space ℝ V]

-- Define unit vectors a, b, c
variables (a b c : V)

-- Define the condition of unit vectors
def is_unit_vector (v : V) : Prop := ∥v∥ = 1

-- Ensure a, b, c are unit vectors
variables (ha : is_unit_vector a) (hb : is_unit_vector b) (hc : is_unit_vector c)

-- Define the condition for the cross product equality
def cross_product_condition : Prop :=
  a × (b × c) = (2 : ℝ) * (b + c) / real.sqrt 3

variable (h_cross_product : cross_product_condition a b c)

-- Define the condition for linear independence
def is_linear_independent : Prop :=
  linear_independent ℝ ![a, b, c]

variable (h_linear_independent : is_linear_independent a b c)

-- Prove the angle between a and b is 150 degrees
theorem angle_between_a_and_b_is_150_degrees :
  angle a b = real.pi * 5 / 6 :=
by sorry

end angle_between_a_and_b_is_150_degrees_l541_541212


namespace fraction_multiplication_l541_541307

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541307


namespace arithmetic_avg_salary_technicians_l541_541254

noncomputable def avg_salary_technicians_problem : Prop :=
  let average_salary_all := 8000
  let total_workers := 21
  let average_salary_rest := 6000
  let technician_count := 7
  let total_salary_all := average_salary_all * total_workers
  let total_salary_rest := average_salary_rest * (total_workers - technician_count)
  let total_salary_technicians := total_salary_all - total_salary_rest
  let average_salary_technicians := total_salary_technicians / technician_count
  average_salary_technicians = 12000

theorem arithmetic_avg_salary_technicians :
  avg_salary_technicians_problem :=
by {
  sorry -- Proof is omitted as per instructions.
}

end arithmetic_avg_salary_technicians_l541_541254


namespace XY_parallel_X_l541_541934

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541934


namespace coefficients_proof_l541_541267

noncomputable def find_coefficients (a b : ℚ) : Prop :=
  let P := λ x : ℚ, a * x^4 + b * x^3 + 18 * x^2 - 8 * x + 3
  let Q := λ x : ℚ, 3 * x^2 - 2 * x + 1
  ∃ e f g : ℚ, P = (3 * x^2 - 2 * x + 1) * (e * x^2 + f * x + g) ∧
    e = 20 ∧ f = 11 / 2 ∧ g = 3 ∧ a = 3 * e ∧ b = 3 * f - 2 * e

theorem coefficients_proof : find_coefficients 60 (-47 / 2) :=
  sorry

end coefficients_proof_l541_541267


namespace only_solution_l541_541470

def pythagorean_euler_theorem (p r : ℕ) : Prop :=
  ∃ (p r : ℕ), Nat.Prime p ∧ r > 0 ∧ (∑ i in Finset.range (r + 1), (p + i)^p) = (p + r + 1)^p

theorem only_solution (p r : ℕ) : pythagorean_euler_theorem p r ↔ p = 3 ∧ r = 2 :=
by
  sorry

end only_solution_l541_541470


namespace largest_n_dividing_30_factorial_l541_541500

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541500


namespace divisibility_by_65_product_of_four_natural_numbers_l541_541710

def N : ℕ := 2^2022 + 1

theorem divisibility_by_65 : ∃ k : ℕ, N = 65 * k := by
  sorry

theorem product_of_four_natural_numbers :
  ∃ a b c d : ℕ, 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ N = a * b * c * d :=
  by sorry

end divisibility_by_65_product_of_four_natural_numbers_l541_541710


namespace initial_water_amount_l541_541135

/-- Define the data related to Harry's hike --/
variables (leak_rate : ℕ) (hike_duration : ℕ) (water_last_mile : ℕ) (water_first_3miles : ℕ) 
          (remaining_water : ℕ) (initial_water : ℕ)

/-- Given conditions: leaking rate, hike duration, water consumed, and remaining after hike --/
def hike_conditions : Prop :=
  leak_rate = 1 ∧ 
  hike_duration = 2 ∧ 
  water_last_mile = 3 ∧ 
  water_first_3miles = 3 ∧ 
  remaining_water = 2 ∧ 
  (initial_water = (leak_rate * hike_duration) + water_last_mile + water_first_3miles + remaining_water)

/-- Prove that the initial amount of water in the canteen was 10 cups. --/
theorem initial_water_amount (h : hike_conditions leak_rate hike_duration water_last_mile water_first_3miles remaining_water initial_water) :
  initial_water = 10 :=
by {
  -- placeholder for proof, to be filled
  sorry 
}

end initial_water_amount_l541_541135


namespace tan_sin_30_computation_l541_541435

theorem tan_sin_30_computation :
  let θ := 30 * Real.pi / 180 in
  Real.tan θ + 4 * Real.sin θ = (Real.sqrt 3 + 6) / 3 :=
by
  let θ := 30 * Real.pi / 180
  have sin_30 : Real.sin θ = 1 / 2 := by sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by sorry
  have tan_30 : Real.tan θ = Real.sin θ / Real.cos θ := by sorry
  have sin_60 : Real.sin (2 * θ) = Real.sqrt 3 / 2 := by sorry
  sorry

end tan_sin_30_computation_l541_541435


namespace calculate_expression_l541_541055

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l541_541055


namespace find_special_two_digit_integer_l541_541392

theorem find_special_two_digit_integer (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : (n + 3) % 3 = 0)
  (h3 : (n + 4) % 4 = 0)
  (h4 : (n + 5) % 5 = 0) :
  n = 60 := by
  sorry

end find_special_two_digit_integer_l541_541392


namespace max_tan_beta_l541_541983

theorem max_tan_beta (α β : ℝ) (h0 : 0 < α) (h1 : α < π / 2) (h2 : 0 < β) (h3 : β < π / 2) 
  (h4 : α + β ≠ π / 2) (h5 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  ∃ β_max, β_max = Real.tan β ∧ β_max ≤ √3 / 3 :=
sorry

end max_tan_beta_l541_541983


namespace side_length_a_range_l541_541551

noncomputable def calculate_a_range 
  (b : ℝ) 
  (B : ℝ) 
  (h_b : b = 2) 
  (h_B : B = 60) : set ℝ :=
  {a | 2 < a ∧ a < 4 * real.sqrt 3 / 3 }

theorem side_length_a_range : 
  ∀ (a b B : ℝ), 
    b = 2 → 
    B = 60 →
    (2 < a ∧ a < 4 * real.sqrt 3 / 3) :=
  by {
  intros a b B h_b h_B,
  sorry
}

end side_length_a_range_l541_541551


namespace fraction_of_number_l541_541312

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541312


namespace sum_of_roots_eq_two_l541_541275

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l541_541275


namespace geometric_progression_sums_l541_541871

def a_gp (n : ℕ) : ℕ := 3^n * (2*n - 1)

def reciprocal_seq : List ℚ :=
  [a_gp 1, a_gp 2, a_gp 3, a_gp 4, a_gp 5].map (λ x, 1 / (x : ℚ))

def sum_reciprocal_seq : ℚ :=
  reciprocal_seq.sum

theorem geometric_progression_sums (n := 5) :
  sum_reciprocal_seq = 29524 / 78732 :=
by
  sorry

end geometric_progression_sums_l541_541871


namespace sean_less_points_than_combined_l541_541619

def tobee_points : ℕ := 4
def jay_points : ℕ := tobee_points + 6
def combined_points_tobee_jay : ℕ := tobee_points + jay_points
def total_team_points : ℕ := 26
def sean_points : ℕ := total_team_points - combined_points_tobee_jay

theorem sean_less_points_than_combined : (combined_points_tobee_jay - sean_points) = 2 := by
  sorry

end sean_less_points_than_combined_l541_541619


namespace money_distribution_l541_541637

-- Conditions
variable (A B x y : ℝ)
variable (h1 : x + 1/2 * y = 50)
variable (h2 : 2/3 * x + y = 50)

-- Problem statement
theorem money_distribution : x = A → y = B → (x + 1/2 * y = 50 ∧ 2/3 * x + y = 50) :=
by
  intro hx hy
  rw [hx, hy]
  exfalso -- using exfalso to skip proof body
  sorry

end money_distribution_l541_541637


namespace angle_sum_not_180_l541_541272

theorem angle_sum_not_180 (R L T : ℝ) (hR : R = 60) (hL : L = 2 * R) (hT : T = 70) : L + R + T ≠ 180 :=
by
  -- Define the conditions
  have hL_eq : L = 2 * R, from hL
  have hR_eq : R = 60, from hR
  have hT_eq : T = 70, from hT

  -- Combine to show the sum
  rw [hR_eq, hL_eq, hR_eq, hT_eq],
  calc 
    2 * 60 + 60 + 70 = 120 + 60 + 70 : by ring
    ... = 250 : by norm_num,
  exact ne_of_gt (by norm_num : 250 > 180)

end angle_sum_not_180_l541_541272


namespace largest_divisor_18n_max_n_l541_541480

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541480


namespace correlation_invariance_l541_541231

-- Define the functions and conditions
variable (X : ℝ → ℝ)
variable (φ : ℝ → ℝ) (non_random_φ : ∀ t, φ t = φ t) -- representing φ(t) is non-random
variable (Y : ℝ → ℝ) (h_Y_def : ∀ t, Y t = X t + φ t)
variable (K_x K_y : ℝ → ℝ → ℝ)

-- Define the correlation function
noncomputable def correlation (f : ℝ → ℝ) : ℝ → ℝ → ℝ :=
  λ t1 t2, ⟨⨑⎵⳽⸴E[(f t1) * (f t2⧪⟁⢴ⴠ

-- The problem to prove
theorem correlation_invariance :
  K_y = correlation Y → K_x = correlation X → K_y = K_x :=
by
  intro hK_y hK_x
  rw [hK_y, hK_x]
  sorry

end correlation_invariance_l541_541231


namespace part_a_part_b_l541_541105

open Polynomial

noncomputable def Q (x : ℝ) : ℝ := (x - 1)^2 + 1

noncomputable def Q_n (x : ℝ) (n : ℕ) : ℝ := iterate Q n x

def a_n (n : ℕ) : ℝ := Inf (Set.image (Q_n 0) (Set.univ))

theorem part_a (h_pos : ∀ n, a_n n > 0) (h_diff : ∃ k, a_n k ≠ a_n (k + 1)) : ∀ n, a_n n < a_n (n + 1) :=
sorry

theorem part_b : ∃ Q : ℝ → ℝ, ∀ n, a_n n < 2021 :=
sorry

end part_a_part_b_l541_541105


namespace value_of_m_l541_541561

theorem value_of_m (a b m : ℝ)
    (h1: 2 ^ a = m)
    (h2: 5 ^ b = m)
    (h3: 1 / a + 1 / b = 1 / 2) :
    m = 100 :=
sorry

end value_of_m_l541_541561


namespace phone_not_answered_prob_l541_541625

noncomputable def P_not_answered_within_4_rings : ℝ :=
  let P1 := 1 - 0.1
  let P2 := 1 - 0.3
  let P3 := 1 - 0.4
  let P4 := 1 - 0.1
  P1 * P2 * P3 * P4

theorem phone_not_answered_prob : 
  P_not_answered_within_4_rings = 0.3402 := 
by 
  -- The detailed steps and proof will be implemented here 
  sorry

end phone_not_answered_prob_l541_541625


namespace cos_value_l541_541110

theorem cos_value {α : ℝ} (h : Real.sin (π / 6 + α) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 := 
by sorry

end cos_value_l541_541110


namespace find_derivative_value_l541_541125

def f (x : ℝ) : ℝ := x^3 - 3*x + 1
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem find_derivative_value :
  f_prime (real.sqrt 2 / 2) = -3 / 2 := 
by {
    sorry
}

end find_derivative_value_l541_541125


namespace total_area_ABCGDE_l541_541174

noncomputable def area_ABCGDE (ABC DEFG : EuclideanGeometry.Shape) : ℝ := 
  let area_ABC := (sqrt 3 / 4) * (6^2)
  let area_DEFG := 6^2
  let area_DCE := (1 / 2) * 6 * 3
  area_ABC + area_DEFG - area_DCE

theorem total_area_ABCGDE 
  (ABC DEFG : EuclideanGeometry.Shape)
  (is_equilateral_triangle : EuclideanGeometry.IsEquilateralTriangle ABC 6)
  (is_square : EuclideanGeometry.IsSquare DEFG 6)
  (is_midpoint : EuclideanGeometry.IsMidpoint D (segment BC)) :
  area_ABCGDE ABC DEFG = 27 + 9 * sqrt 3 :=
by
  sorry

end total_area_ABCGDE_l541_541174


namespace total_marks_math_physics_l541_541833

variable (M P C : ℕ)

theorem total_marks_math_physics (h1 : C = P + 10) (h2 : (M + C) / 2 = 35) : M + P = 60 :=
by
  sorry

end total_marks_math_physics_l541_541833


namespace number_of_passed_boys_l541_541740

-- Define the variables and conditions
variables (T : ℕ) (A : ℕ) (A_p : ℕ) (A_f : ℕ)
variables (P F : ℕ)

-- Define the conditions
def total_boys := T = 120
def average_all_boys := A = 35
def average_passed_boys := A_p = 39
def average_failed_boys := A_f = 15
def total_boys_relation := P + F = T
def total_marks_relation := A * T = P * A_p + F * A_f

-- Define the proof problem
theorem number_of_passed_boys 
  (hT : total_boys)
  (hA : average_all_boys)
  (hA_p : average_passed_boys)
  (hA_f : average_failed_boys)
  (h_relation1 : total_boys_relation)
  (h_relation2 : total_marks_relation) : 
  P = 100 := by
  sorry

end number_of_passed_boys_l541_541740


namespace sqrt_r_squared_plus_c_squared_irrational_l541_541199

theorem sqrt_r_squared_plus_c_squared_irrational
    (a b c : ℤ) (r : ℝ)
    (h1 : a ≠ 0) (h2 : c ≠ 0)
    (h3 : a * r^2 + b * r + c = 0) : ¬ ∃ q : ℚ, q = real.sqrt (r^2 + (c : ℝ)^2) :=
by
  sorry

end sqrt_r_squared_plus_c_squared_irrational_l541_541199


namespace ratio_lateral_surface_area_base_l541_541820

theorem ratio_lateral_surface_area_base
    (K A B C M N : Point)
    (pyramid_regular : RegularTriangularPyramid K A B C)
    (M_midpoint : M ∈ Midpoint(A, K))
    (N_midpoint : N ∈ Midpoint(C, K))
    (plane_perpendicular : Plane(M, B, N) ⊥ Plane(A, K, C)) :
    ratio_lateral_to_base(pyramid_regular) = Real.sqrt 6 := 
sorry

end ratio_lateral_surface_area_base_l541_541820


namespace probability_red_blue_l541_541284

-- Declare the conditions (probabilities for white, green and yellow marbles).
variables (total_marbles : ℕ) (P_white P_green P_yellow P_red_blue : ℚ)
-- implicitly P_white, P_green, P_yellow, P_red_blue are probabilities, therefore between 0 and 1

-- Assume the conditions given in the problem
axiom total_marbles_condition : total_marbles = 250
axiom P_white_condition : P_white = 2 / 5
axiom P_green_condition : P_green = 1 / 4
axiom P_yellow_condition : P_yellow = 1 / 10

-- Proving the required probability of red or blue marbles
theorem probability_red_blue :
  P_red_blue = 1 - (P_white + P_green + P_yellow) :=
sorry

end probability_red_blue_l541_541284


namespace dot_product_identity_l541_541652

-- Define vectors m and n
variables (m n : ℝ^3)

-- The problem statement rephrased in Lean
theorem dot_product_identity (m n : ℝ^3) : 
  (n - m) • (m + n) = (n • n) - (m • m) :=
  sorry

end dot_product_identity_l541_541652


namespace hyperbola_equation_not_midpoint_P_AB_l541_541923

open Function

-- Define the conditions for the hyperbola C
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- The point (-2, √6) lies on the hyperbola
def passes_through_point (a b : ℝ) : Prop :=
  is_hyperbola a b (-2) (Real.sqrt 6)

-- The relationship between a and b due to the asymptotic lines y = ±√2x
def asymptotic_relationship (a b : ℝ) : Prop := 
  b = sqrt 2 * a

-- The formal statement to be proven
theorem hyperbola_equation : ∃ (a b : ℝ), passes_through_point a b ∧ asymptotic_relationship a b ∧ 
  (∀ x y : ℝ, is_hyperbola a b x y ↔ (x^2 - y^2 / 2 = 1)) :=
by 
  sorry

-- Define the condition that P cannot be the midpoint of AB given previous conditions on hyperbola
def not_midpoint (a b : ℝ) : Prop := 
  ∀ (k : ℝ), (k < 3/2) → 
  ¬( ( (k^2 - k) / (k^2 - 2) = 1) ∧ ((2 * k - 2) / (k^2 - 2) = 1))

-- The formal statement to prove that P cannot be the midpoint of AB
theorem not_midpoint_P_AB : 
  ∀ a b : ℝ, passes_through_point a b → asymptotic_relationship a b → not_midpoint a b :=
by 
  sorry

end hyperbola_equation_not_midpoint_P_AB_l541_541923


namespace largest_of_four_consecutive_odd_numbers_l541_541354

theorem largest_of_four_consecutive_odd_numbers (x : ℤ) : 
  (x % 2 = 1) → 
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →
  (x + 6 = 27) :=
by
  sorry

end largest_of_four_consecutive_odd_numbers_l541_541354


namespace athlete_last_finish_l541_541287

theorem athlete_last_finish (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  let T1 := 1 / v1 + 2 / v2 
  let T2 := 1 / v2 + 2 / v3
  let T3 := 1 / v3 + 2 / v1
  T2 > T1 ∧ T2 > T3 :=
by
  sorry

end athlete_last_finish_l541_541287


namespace rectangle_width_l541_541257

theorem rectangle_width (length : ℕ) (perimeter : ℕ) (h1 : length = 20) (h2 : perimeter = 70) :
  2 * (length + width) = perimeter → width = 15 :=
by
  intro h
  rw [h1, h2] at h
  -- Continue the steps to solve for width (can be simplified if not requesting the whole proof)
  sorry

end rectangle_width_l541_541257


namespace quadruple_exists_unique_l541_541525

def digits (x : Nat) : Prop := x ≤ 9

theorem quadruple_exists_unique :
  ∃ (A B C D: Nat),
    digits A ∧ digits B ∧ digits C ∧ digits D ∧
    A > B ∧ B > C ∧ C > D ∧
    (A * 1000 + B * 100 + C * 10 + D) -
    (D * 1000 + C * 100 + B * 10 + A) =
    (B * 1000 + D * 100 + A * 10 + C) ∧
    (A, B, C, D) = (7, 6, 4, 1) :=
by
  sorry

end quadruple_exists_unique_l541_541525


namespace no_real_solutions_l541_541743

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
    (a = 0) ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a ^ 2 > 0) :=
by
  sorry

end no_real_solutions_l541_541743


namespace factorization_correctness1_factorization_correctness2_l541_541901

-- Definition 1: Factorizing the first expression
def factorize_expr1 : Bool :=
  (2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2)

-- Definition 2: Factorizing the second expression
def factorize_expr2 : Bool :=
  (m^2 * (m - n) + n^2 * (n - m) = (m - n)^2 * (m + n))

-- Prove that the expressions can be factorized as stated
theorem factorization_correctness1 : factorize_expr1 :=
  by sorry

theorem factorization_correctness2 : factorize_expr2 :=
  by sorry

end factorization_correctness1_factorization_correctness2_l541_541901


namespace isosceles_right_triangle_hypotenuse_l541_541038

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : ℝ) (hyp : a = 30 ∧ h^2 = a^2 + a^2) : h = 30 * Real.sqrt 2 :=
sorry

end isosceles_right_triangle_hypotenuse_l541_541038


namespace rollo_guinea_pigs_food_l541_541717

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l541_541717


namespace sqrt_neg_num_squared_l541_541862

theorem sqrt_neg_num_squared (n : ℤ) (hn : n = -2023) : Real.sqrt (n ^ 2) = 2023 :=
by
  -- substitute -2023 for n
  rw hn
  -- compute (-2023)^2
  have h1 : (-2023 : ℝ) ^ 2 = 2023 ^ 2 :=
    by rw [neg_sq, abs_of_nonneg (by norm_num)]
  -- show sqrt(2023^2) = 2023
  rw [h1, Real.sqrt_sq]
  exact eq_of_abs_sub_nonneg (by norm_num)

end sqrt_neg_num_squared_l541_541862


namespace part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l541_541580

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem part1_smallest_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem part1_monotonic_interval :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 6) →
  ∃ (b a c : ℝ) (A : ℝ), b + c = 2 * a ∧ 2 * A = A + Real.pi / 3 ∧ 
  f A = 1 / 2 ∧ a = 3 * Real.sqrt 2 := 
sorry

theorem part2_value_of_a :
  ∀ (A b c : ℝ), 
  (∃ (a : ℝ), 2 * a = b + c ∧ 
  f A = 1 / 2 ∧ 
  b * c = 18 ∧ 
  Real.cos A = 1 / 2) → 
  ∃ a, a = 3 * Real.sqrt 2 := 
sorry

end part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l541_541580


namespace chocolate_bar_first_player_wins_l541_541298

theorem chocolate_bar_first_player_wins (n m : ℕ) (h : 1 ≤ n ∧ 1 ≤ m) :
  ∃ strategy : ℕ × ℕ → (ℕ × ℕ) → bool, 
  (∀ a b, strategy (a, b) (1, 1) = false) → 
  (∀ a b, strategy (a, b) (n, m) = true) :=
sorry

end chocolate_bar_first_player_wins_l541_541298


namespace total_students_taught_l541_541839

theorem total_students_taught (students_per_first_year : ℕ) 
  (students_per_year : ℕ) 
  (years_remaining : ℕ) 
  (total_years : ℕ) :
  students_per_first_year = 40 → 
  students_per_year = 50 →
  years_remaining = 9 →
  total_years = 10 →
  students_per_first_year + students_per_year * years_remaining = 490 := 
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃]
  norm_num
  rw h₄
  norm_num

end total_students_taught_l541_541839


namespace general_form_of_equation_l541_541445

theorem general_form_of_equation : 
  ∀ x : ℝ, (x - 1) * (x - 2) = 4 → x^2 - 3 * x - 2 = 0 := by
  sorry

end general_form_of_equation_l541_541445


namespace triangle_parallel_bisectors_l541_541950

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541950


namespace albert_age_l541_541396

theorem albert_age
  (A : ℕ)
  (dad_age : ℕ)
  (h1 : dad_age = 48)
  (h2 : dad_age - 4 = 4 * (A - 4)) :
  A = 15 :=
by
  sorry

end albert_age_l541_541396


namespace sum_of_roots_of_quadratic_l541_541278

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l541_541278


namespace floor_length_l541_541234

/-- Given the rectangular tiles of size 50 cm by 40 cm, which are laid on a rectangular floor
without overlap and with a maximum of 9 tiles. Prove the floor length is 450 cm. -/
theorem floor_length (tiles_max : ℕ) (tile_length tile_width floor_length floor_width : ℕ)
  (Htile_length : tile_length = 50) (Htile_width : tile_width = 40)
  (Htiles_max : tiles_max = 9)
  (Hconditions : (∀ m n : ℕ, (m * n = tiles_max) → 
                  (floor_length = m * tile_length ∨ floor_length = m * tile_width)))
  : floor_length = 450 :=
by 
  sorry

end floor_length_l541_541234


namespace correct_points_per_answer_l541_541225

noncomputable def points_per_correct_answer (total_questions : ℕ) 
  (answered_correctly : ℕ) (final_score : ℝ) (penalty_per_incorrect : ℝ)
  (total_incorrect : ℕ := total_questions - answered_correctly) 
  (points_subtracted : ℝ := total_incorrect * penalty_per_incorrect) 
  (earned_points : ℝ := final_score + points_subtracted) : ℝ := 
    earned_points / answered_correctly

theorem correct_points_per_answer :
  points_per_correct_answer 120 104 100 0.25 = 1 := 
by 
  sorry

end correct_points_per_answer_l541_541225


namespace kylie_daisies_l541_541661

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l541_541661


namespace points_per_enemy_l541_541629

theorem points_per_enemy (total_enemies : ℕ) (destroyed_enemies : ℕ) (total_points : ℕ) 
  (h1 : total_enemies = 7)
  (h2 : destroyed_enemies = total_enemies - 2)
  (h3 : destroyed_enemies = 5)
  (h4 : total_points = 40) :
  total_points / destroyed_enemies = 8 :=
by
  sorry

end points_per_enemy_l541_541629


namespace rightmost_three_digits_of_7_pow_2011_l541_541779

theorem rightmost_three_digits_of_7_pow_2011 :
  (7 ^ 2011) % 1000 = 7 % 1000 :=
by
  sorry

end rightmost_three_digits_of_7_pow_2011_l541_541779


namespace arrange_numbers_l541_541409

def a : ℝ := 3 ^ 0.7
def b : ℝ := Real.log 0.7 / Real.log 3
def c : ℝ := 0.7 ^ 3

theorem arrange_numbers : a > c ∧ c > b :=
by
  -- The following proof steps are skipped.
  sorry

end arrange_numbers_l541_541409


namespace simplify_and_evaluate_l541_541725

theorem simplify_and_evaluate (m : ℤ) (h : m = -2) :
  let expr := (m / (m^2 - 9)) / (1 + (3 / (m - 3)))
  expr = 1 :=
by
  sorry

end simplify_and_evaluate_l541_541725


namespace sum_of_roots_eq_two_l541_541274

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l541_541274


namespace number_of_zeros_derivative_negative_at_midpoint_l541_541119

theorem number_of_zeros (a : ℝ) (h₀ : 0 < a) :
  if a < 1 then ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f x1 = 0 ∧ f x2 = 0
  else if a = 1 then ∃ x : ℝ, f x = 0
  else ∀ x : ℝ, f x ≠ 0 :=
sorry

theorem derivative_negative_at_midpoint (a : ℝ) (x1 x2 : ℝ) (h₀ : 0 < a) (h₁ : 0 < x1) (h₂ : x1 < x2) (h₃ : f x1 = 0) (h₄ : f x2 = 0) :
  f' ((x1 + x2) / 2) < 0 :=
sorry

noncomputable def f (x : ℝ) : ℝ := Real.log x - a * x + 1

noncomputable def f' (x : ℝ) : ℝ := 1 / x - a

end number_of_zeros_derivative_negative_at_midpoint_l541_541119


namespace fraction_as_decimal_l541_541072

theorem fraction_as_decimal : 
  let frac := 7 / 13
  in Real.round (frac * 10^3) / 10^3 = 0.538 :=
by
  let frac := 7 / 13
  have h1 : Real.round (frac * 10^3) / 10^3 = 0.538 := sorry
  exact h1

end fraction_as_decimal_l541_541072


namespace recurrence_relation_solution_l541_541243

theorem recurrence_relation_solution (a : ℕ → ℕ) 
  (h_rec : ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2))
  (h0 : a 0 = 3)
  (h1 : a 1 = 5) :
  ∀ n, a n = 3^n + 2 :=
by
  sorry

end recurrence_relation_solution_l541_541243


namespace drum_depth_l541_541190

theorem drum_depth (rate : ℝ) (time : ℝ) (area : ℝ) (volume : ℝ) :
  rate = 5 → time = 3 → area = 300 → volume = 4500 → volume = area * 15 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end drum_depth_l541_541190


namespace parallel_lines_triangle_l541_541948

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541948


namespace T_n_less_than_2_l541_541760

-- Definitions of the sequences a_n and b_n
def a (n : ℕ) : ℕ := 2 * n + 1
def b (n : ℕ) : ℕ := 2^(n + 1)

-- Definition for the sequence c_n and its sum as T_n
def c (n : ℕ) : ℕ → ℝ := λ n, (a n - 1) / b n
def T (n : ℕ) : ℝ := ∑ i in finset.range(n), c (i+1)

-- The problem statement to be proved
theorem T_n_less_than_2 (n : ℕ) : T n < 2 := 
by
  sorry -- Proof not provided

end T_n_less_than_2_l541_541760


namespace tangent_line_characterization_l541_541679

theorem tangent_line_characterization 
  (α β m n : ℝ) 
  (h_pos_α : 0 < α) 
  (h_pos_β : 0 < β) 
  (h_alpha_beta : 1/α + 1/β = 1)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n) :
  (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ x^α + y^α = 1 → mx + ny = 1) ↔ (m^β + n^β = 1) := 
sorry

end tangent_line_characterization_l541_541679


namespace fly_distance_from_ceiling_l541_541013

noncomputable def distance_from_ceiling (z ceiling_height : ℝ) : ℝ :=
  ceiling_height - z

def distance_to_point_Q (x y z distance : ℝ) : Prop :=
  distance = Real.sqrt (x^2 + y^2 + z^2)

theorem fly_distance_from_ceiling : 
  ∀ (x y z : ℝ) (ceiling_height : ℝ), 
    x = 3 → y = 4 → ceiling_height = 15 →
    distance_to_point_Q x y z 13 →
    distance_from_ceiling z ceiling_height = 3 :=
by 
  intros x y z ceiling_height x_eq y_eq ceiling_height_eq dist_eq
  rw [x_eq, y_eq, ceiling_height_eq, distance_to_point_Q, distance_from_ceiling]
  have h : 13 * 13 = (Real.sqrt (3 * 3 + 4 * 4 + z * z)) * (Real.sqrt (3 * 3 + 4 * 4 + z * z)),
    from by sorry,
  rw [Real.sqrt_mul_self h] at dist_eq,
  linarith,
  sorry

end fly_distance_from_ceiling_l541_541013


namespace polynomial_remainder_eq_l541_541882

noncomputable def polynomial_division_remainder : ℕ :=
  Polynomial.X^4 - 3 * Polynomial.X + 1

noncomputable def polynomial_divisor : ℕ :=
  Polynomial.X^2 - Polynomial.X - 1

theorem polynomial_remainder_eq : 
  Polynomial.euclidean_domain_mod polynomial_division_remainder polynomial_divisor = 3 :=
by
  sorry

end polynomial_remainder_eq_l541_541882


namespace polynomial_roots_correct_l541_541469

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l541_541469


namespace factor_polynomial_l541_541893

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l541_541893


namespace distinct_distributions_of_books_l541_541594

theorem distinct_distributions_of_books : 
  let books := 6 in
  let individuals := 3 in
  ∃ (ways : Nat), ways = 1200 := 
by
  sorry

end distinct_distributions_of_books_l541_541594


namespace smallest_positive_period_range_of_f_l541_541683

noncomputable def f (x : ℝ) : ℝ := 
  (sqrt 3) * sin x * cos x + (sin x) ^ 2 + sin (2 * x - real.pi / 6)

theorem smallest_positive_period :
  ∀ x : ℝ, f(x + real.pi) = f(x) :=
by
  intro x
  sorry

theorem range_of_f (x : ℝ) (h : x ∈ Ioo 0 (real.pi / 2)) :
  f(x) ∈ Ioc (-1 / 2) (5 / 2) :=
by
  intro x h
  sorry

end smallest_positive_period_range_of_f_l541_541683


namespace christine_speed_l541_541429

theorem christine_speed :
  ∀ (distance time : ℕ), distance = 20 ∧ time = 5 → distance / time = 4 :=
by
  intros distance time h
  cases h
  rw [h_left, h_right]
  norm_num

end christine_speed_l541_541429


namespace daisies_left_l541_541663

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end daisies_left_l541_541663


namespace probability_A_and_B_adjacent_l541_541735

theorem probability_A_and_B_adjacent (A B C : Type) [Fintype A] [Fintype B] [Fintype C] :
  (∃ (fintype.card (A ∪ B ∪ C) = 3)
    (fintype.card (A ∪ B) = 2), (probability (A ∪ B adjacent_in (A ∪ B ∪ C)) = 2/3)) :=
sorry

end probability_A_and_B_adjacent_l541_541735


namespace parabola_trajectory_thm_symmetric_point_Q_thm_l541_541584

def parabola_trajectory : Prop :=
  ∀ (x y x_A y_A : ℝ), 
  (y^2 = 4 * x) ∧ 
  (x = 2 - x_A) ∧ 
  (y = -y_A) →
  (y^2 = 8 - 4 * x)

def symmetric_point_Q : Prop :=
  ∃ (t : ℝ), 
  (t = 0 ∨ t = -15 / 4) ∧ 
  let y, x := (4/5 * t, -3/5 * t) in 
  y^2 = 4 * x

theorem parabola_trajectory_thm : parabola_trajectory := by sorry

theorem symmetric_point_Q_thm : symmetric_point_Q := by sorry

end parabola_trajectory_thm_symmetric_point_Q_thm_l541_541584


namespace sales_ratio_l541_541699

def large_price : ℕ := 60
def small_price : ℕ := 30
def last_month_large_paintings : ℕ := 8
def last_month_small_paintings : ℕ := 4
def this_month_sales : ℕ := 1200

theorem sales_ratio :
  (this_month_sales : ℕ) = 2 * (last_month_large_paintings * large_price + last_month_small_paintings * small_price) :=
by
  -- We will just state the proof steps as sorry.
  sorry

end sales_ratio_l541_541699


namespace largest_divisor_18n_max_n_l541_541477

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541477


namespace min_distance_l541_541170

noncomputable def point_on_curve (x₁ y₁ : ℝ) : Prop :=
  y₁ = x₁^2 - Real.log x₁

noncomputable def point_on_line (x₂ y₂ : ℝ) : Prop :=
  x₂ - y₂ - 2 = 0

theorem min_distance 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : point_on_curve x₁ y₁)
  (h₂ : point_on_line x₂ y₂) 
  : (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2 :=
sorry

end min_distance_l541_541170


namespace total_cost_six_years_l541_541033

variable {fees : ℕ → ℝ}

-- Conditions
def fee_first_year : fees 1 = 80 := sorry

def fee_increase (n : ℕ) : fees (n + 1) = fees n + (10 + 2 * (n - 1)) := 
sorry

-- Proof problem: Prove that the total cost is 670
theorem total_cost_six_years : (fees 1 + fees 2 + fees 3 + fees 4 + fees 5 + fees 6) = 670 :=
by sorry

end total_cost_six_years_l541_541033


namespace prob_of_diff_color_l541_541659

/-- Define the three options for colors. --/
inductive Color
| black : Color
| gold : Color
| blue : Color
| white : Color

open Color

/-- Define the probability that the shorts and jersey differ in color. --/
noncomputable def prob_diff_color : ℚ :=
  let shorts_options := [black, gold, blue]
  let jersey_options := [black, white, gold]
  let total_combinations := (shorts_options.length * jersey_options.length : ℚ)
  let same_color_combinations := (1 : ℚ)          -- black.black 
                                       + (1 : ℚ)   -- gold.gold
  let diff_color_combinations := total_combinations - same_color_combinations
  diff_color_combinations / total_combinations

theorem prob_of_diff_color (h1 : shorts_options = [black, gold, blue])
                           (h2 : jersey_options = [black, white, gold]) 
                           (h3 : shorts_options.length = 3)
                           (h4 : jersey_options.length = 3)
                           (h5 : total_combinations = 9)
                           (h6 : same_color_combinations = 2)
                           (h7 : diff_color_combinations = 7) :
  prob_diff_color = 7 / 9 :=
by
  unfold prob_diff_color
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end prob_of_diff_color_l541_541659


namespace count_even_sum_subsets_l541_541139

open Finset

noncomputable def givenSet : Finset ℕ := {31, 47, 58, 62, 89, 132, 164}

def isEven (n : ℕ) : Prop := n % 2 = 0

theorem count_even_sum_subsets :
  (givenSet.powerset.filter (λ s, s.card = 3 ∧ isEven (s.sum id))).card = 16 := sorry

end count_even_sum_subsets_l541_541139


namespace frame_can_be_cut_into_different_parts_l541_541825

theorem frame_can_be_cut_into_different_parts (a b : ℕ) (h1 : a + b = 16) :
  ∃ (parts : Finset (Finset (Fin (6 * 6)))) (flip : parts → parts), 
    (∀ p ∈ parts, ∃ x y, outer_dim a b x y ∧ inner_dim a b x y)
    ∧ reassemble_into_square parts flip ∧ all_parts_different parts :=
sorry

def outer_dim (a b x y : ℕ) : Prop := (x = a + 2) ∧ (y = b + 2)

def inner_dim (a b x y : ℕ) : Prop := (x = a) ∧ (y = b)

def reassemble_into_square (parts : Finset (Finset (Fin (6 * 6)))) (flip : parts → parts) : Prop :=
  finset.sum parts (λ p, part_area p) = 6 * 6

def all_parts_different (parts : Finset (Finset (Fin (6 * 6)))) : Prop :=
  ∀ p1 p2 ∈ parts, p1 ≠ p2 → p1 ≠ flip p2

def part_area (p : Finset (Fin (6 * 6))) : ℕ := p.card

end frame_can_be_cut_into_different_parts_l541_541825


namespace at_least_one_lands_l541_541156

def p : Prop := sorry -- Proposition that Person A lands in the designated area
def q : Prop := sorry -- Proposition that Person B lands in the designated area

theorem at_least_one_lands : p ∨ q := sorry

end at_least_one_lands_l541_541156


namespace sergeant_travel_distance_l541_541814

theorem sergeant_travel_distance (x k : ℝ) (hx : 0 < x) (hk : 1 < k) :
  -- Conditions
  let t1 := 1 / (kx - x),
      t2 := 1 / (kx + x),
      total_time := t1 + t2,
      distance_by_column := 2.4,
      total_distance := 3.6 in
  -- Equation solving for total_time
  total_time = distance_by_column / x →
  -- Calculate total distance
  3 * 1 + distance_by_column = total_distance :=
sorry

end sergeant_travel_distance_l541_541814


namespace greatest_y_value_l541_541247

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) : y ≤ -1 :=
by
  sorry

end greatest_y_value_l541_541247


namespace find_roots_of_polynomial_l541_541465

theorem find_roots_of_polynomial :
  (∃ (a b : ℝ), 
    Multiplicity (polynomial.C a) (polynomial.C (Real.ofRat 2)) = 2 ∧ 
    Multiplicity (polynomial.C b) (polynomial.C (Real.ofRat 1)) = 1) ∧ 
  (x^3 - 7 * x^2 + 14 * x - 8 = 
    (x - 1) * (x - 2)^2) := sorry

end find_roots_of_polynomial_l541_541465


namespace geometric_sequence_a2_a4_l541_541922

-- Definitions based on given conditions in the problem
variables {a n : ℕ} (a_ n_ : ℕ → ℕ)
variables (S : ℕ → ℕ)
variables (r : ℕ → ℝ) {common_ratio : ℝ}

-- Condition: Geometric sequence with all terms positive
hypothesis pos_terms : ∀ (n : ℕ), a_ n > 0

-- Condition: Sum of the first n terms
hypothesis Sn_def : ∀ (n : ℕ), S n = (a_ 1 ) * (1 - common_ratio ^ n) / (1 - common_ratio)

-- Given conditions
hypothesis cond1 : a_ 2 * a_ 4 = 9
hypothesis cond2 : 9 * S 4 = 10 * S 2

-- Prove that a_2 + a_4 = 10
theorem geometric_sequence_a2_a4 : a_ 2 + a_ 4 = 10 :=
sorry

end geometric_sequence_a2_a4_l541_541922


namespace parabola_focus_distance_l541_541549

noncomputable def focus_distance (x y : ℝ) : ℝ := real.sqrt ((x - 0)^2 + (y - 1)^2)

theorem parabola_focus_distance 
  (x y : ℝ) 
  (h_parabola : x^2 = 4 * y) 
  (h_distance : focus_distance x y = 3) :
  y = 2 := 
by
  -- provide the proof here
  sorry

end parabola_focus_distance_l541_541549


namespace f_of_one_half_eq_15_l541_541979

def g (x : ℝ) : ℝ := 1 - 2 * x

def f_comp_g (x : ℝ) (hx: x ≠ 0) : ℝ := (1 - x^2) / x^2

theorem f_of_one_half_eq_15 :
  f_comp_g 1/4 (by norm_num) = 15 := sorry

end f_of_one_half_eq_15_l541_541979


namespace factor_polynomial_l541_541894

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l541_541894


namespace find_number_250_l541_541337

theorem find_number_250 (N : ℤ)
  (h1 : 5 * N = 8 * 156 + 2): N = 250 :=
sorry

end find_number_250_l541_541337


namespace number_of_correct_propositions_l541_541563

-- Definitions for lines, planes, perpendicularity, and parallelism
variable {Line Plane : Type} 
variable (a b c : Line) (α β γ : Plane)

-- Define assumptions about perpendicularity and parallelism
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line (l1 l2 : Line) : Prop := sorry
def perp_plane_plane (p1 p2 : Plane) : Prop := sorry
def parallel_plane (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Propositions
def prop1 := perp_line_plane a α ∧ perp_line_plane b α → parallel_line a b
def prop2 := perp_plane_plane α γ ∧ perp_plane_plane β γ → parallel_plane α β
def prop3 := line_in_plane b α ∧ perp_line_plane b β → perp_plane_plane α β
def prop4 := perp_line_plane a α ∧ parallel_line b β ∧ perp_plane_plane α β → perp_line_plane a b

-- Proof statement
theorem number_of_correct_propositions : (prop1 ∧ prop3) ∧ ¬prop2 ∧ ¬prop4 := by
  sorry

end number_of_correct_propositions_l541_541563


namespace problem_1_l541_541798

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x < 0 then 2^(-x) + x^2 else -8

theorem problem_1 (h_odd : is_odd_function f) (h_def : ∀ x < 0, f x = 2^(-x) + x^2) :
  f 2 = -8 := by
  sorry

end problem_1_l541_541798


namespace person_B_age_l541_541229

variables (a b c d e f g : ℕ)

-- Conditions
axiom cond1 : a = b + 2
axiom cond2 : b = 2 * c
axiom cond3 : c = d / 2
axiom cond4 : d = e - 3
axiom cond5 : f = a * d
axiom cond6 : g = b + e
axiom cond7 : a + b + c + d + e + f + g = 292

-- Theorem statement
theorem person_B_age : b = 14 :=
sorry

end person_B_age_l541_541229


namespace initial_money_l541_541696

-- Let M represent the initial amount of money Mrs. Hilt had.
variable (M : ℕ)

-- Condition 1: Mrs. Hilt bought a pencil for 11 cents.
def pencil_cost : ℕ := 11

-- Condition 2: She had 4 cents left after buying the pencil.
def amount_left : ℕ := 4

-- Proof problem statement: Prove that M = 15 given the above conditions.
theorem initial_money (h : M = pencil_cost + amount_left) : M = 15 :=
by
  sorry

end initial_money_l541_541696


namespace largest_n_dividing_30_factorial_l541_541481

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541481


namespace bears_in_shipment_l541_541390

theorem bears_in_shipment (initial_bears shipment_bears bears_per_shelf total_shelves : ℕ)
  (h1 : initial_bears = 17)
  (h2 : bears_per_shelf = 9)
  (h3 : total_shelves = 3)
  (h4 : total_shelves * bears_per_shelf = 27) :
  shipment_bears = 10 :=
by
  sorry

end bears_in_shipment_l541_541390


namespace range_of_k_l541_541997

noncomputable theory

open Real

def y (k x : ℝ) : ℝ := k * x^2 - 4 * x + k - 3

def condition (k : ℝ) : Prop :=
  ∀ x : ℝ, y k x < 0

theorem range_of_k (k : ℝ) : condition k → k ∈ Iio (-1) :=
sorry

end range_of_k_l541_541997


namespace triangle_area_l541_541624

-- Definitions of the various elements mentioned in the conditions
structure Hexagon :=
  (A B C D E F : Type)

structure Square :=
  (side : ℕ)

structure EquilateralTriangle :=
  (side : ℕ) 

structure EquiangularHexagon (h : Hexagon) :=
  (equiangular : Prop)

structure RightTriangle extends Square, EquilateralTriangle :=
  (right_angle : Prop)

def TriangleKBC :=
  (K B C : Hexagon)
  (right_triangle : RightTriangle)

-- Given conditions
def hexagon_ABCDEF : Hexagon := {A := Unit, B := Unit, C := Unit, D := Unit, E := Unit, F := Unit}
def square_ABJI : Square := {side := 5}
def square_FEHG : Square := {side := 7}
def equil_triangle_JBK : EquilateralTriangle := {side := 5}
def equiangular_hexagon_ABCDEF : EquiangularHexagon hexagon_ABCDEF := {equiangular := true}
def triangle_KBC : TriangleKBC := {right_triangle := {side := 5, right_angle := true}}

-- Problem statement
theorem triangle_area :
  ∀ (hexagon_ABCDEF : Hexagon) 
    (square_ABJI square_FEHG : Square) 
    (equil_triangle_JBK : EquilateralTriangle) 
    (equiangular_hexagon_ABCDEF : EquiangularHexagon hexagon_ABCDEF)
    (triangle_KBC : TriangleKBC),
  (square_FEHG.side = 7) →
  (square_ABJI.side = 5) → 
  (equil_triangle_JBK.side = 5) → 
  (triangle_KBC.right_triangle.right_angle) → 
  (triangle_KBC.right_triangle.side = 7) → 
  17.5 :=
by
  intros hexagon_ABCDEF square_ABJI square_FEHG equil_triangle_JBK equiangular_hexagon_ABCDEF triangle_KBC property1 property2 property3 property4 property5
  exact 17.5
  sorry

end triangle_area_l541_541624


namespace marys_garbage_bill_is_correct_l541_541687

noncomputable def weekly_cost_trash_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_cost_recycling_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_total_cost (trash_cost : ℝ) (recycling_cost : ℝ) : ℝ :=
  trash_cost + recycling_cost

noncomputable def monthly_cost (weekly_cost : ℝ) (num_weeks : ℕ) : ℝ :=
  weekly_cost * num_weeks

noncomputable def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount_and_fine (total_cost : ℝ) (discount : ℝ) (fine : ℝ) : ℝ :=
  total_cost - discount + fine

theorem marys_garbage_bill_is_correct :
  let weekly_trash_cost := weekly_cost_trash_bin 10 2 in
  let weekly_recycling_cost := weekly_cost_recycling_bin 5 1 in
  let weekly_total := weekly_total_cost weekly_trash_cost weekly_recycling_cost in
  let monthly_total := monthly_cost weekly_total 4 in
  let senior_discount := discount monthly_total 0.18 in
  let fine := 20 in
  total_cost_after_discount_and_fine monthly_total senior_discount fine = 102 :=
by
  sorry

end marys_garbage_bill_is_correct_l541_541687


namespace problem1_problem2_l541_541562

variable (α : ℝ)

-- Equivalent problem 1
theorem problem1 (h : Real.tan α = 7) : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 := 
  sorry

-- Equivalent problem 2
theorem problem2 (h : Real.tan α = 7) : Real.sin α * Real.cos α = 7 / 50 := 
  sorry

end problem1_problem2_l541_541562


namespace number_of_valid_functions_l541_541082

theorem number_of_valid_functions : 
  (f : ℝ → ℝ) →
  (∀ x y : ℝ, f(x * y) + f(x) + f(y) - f(x) * f(y) ≥ 2) →
  (constant_functions : Set ℝ → ℕ) :=
by
  sorry

end number_of_valid_functions_l541_541082


namespace polynomial_root_reduction_l541_541684

def P (a : ℕ → ℝ) (n : ℕ) : ℝ[X] := ∑ i in finset.range (n + 1), (a i) * X^i

theorem polynomial_root_reduction (a : ℕ → ℝ) (n : ℕ) (h : 0 < n) 
  (P_has_root : ∃ x : ℝ, (P a n).eval x = 0) (a0_nonzero : a 0 ≠ 0) :
  ∃ seq : list (ℝ[X]), seq.head = P a n ∧ seq.tail.head.eval 0 = a 0 ∧ 
  (∀ Q ∈ seq, ∃ x : ℝ, Q.eval x = 0) :=
sorry

end polynomial_root_reduction_l541_541684


namespace unique_lambda_b_l541_541557

noncomputable def verify_lambda_b (λ : ℝ) (b : ℝ) : Prop :=
  ∀ (θ : ℝ), (cos θ - b)^2 + sin θ^2 = λ^2 * ((cos θ + 2)^2 + sin θ^2)

theorem unique_lambda_b :
  (∃ (λ b : ℝ), λ = 1/2 ∧ b = -1/2 ∧ λ > 0 ∧ λ ≠ -2 ∧ verify_lambda_b λ b) :=
begin
  use (1 / 2),
  use (-1 / 2),
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros θ,
    -- Proof goes here, to show the condition holds with λ = 1/2 and b = -1/2
    sorry
  }
end

end unique_lambda_b_l541_541557


namespace sum_of_roots_of_quadratic_l541_541277

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l541_541277


namespace min_value_of_pf_pq_l541_541127

theorem min_value_of_pf_pq (a b: ℝ) (h1: a > 0) (h2: b > 0) 
  (h_hyperbola: ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_shared_focus: ∃ F : ℝ × ℝ, F = (2, 0) ∧ (∃ x y : ℝ, y^2 = 8 * x ∧ (2,0) = F))
  (h_distance: ∀ F : ℝ × ℝ, F = (2,0) → ∃ asymptote : ℝ × ℝ, asymptote = (1, 3) ∧  d F asymptote = 1) :
  ∀ P : ℝ × ℝ, P ∈ left_branch_of_hyperbola → ∃ Q : ℝ × ℝ, Q = (1, 3) ∧ min_value |PF| + |PQ| = 2 * (sqrt 3) + 3 * (sqrt 2) :=
sorry

end min_value_of_pf_pq_l541_541127


namespace Q_share_of_profit_l541_541706

theorem Q_share_of_profit (P Q T : ℕ) (hP : P = 54000) (hQ : Q = 36000) (hT : T = 18000) : Q's_share = 7200 :=
by
  -- Definitions and conditions
  let P := 54000
  let Q := 36000
  let T := 18000
  have P_ratio := 3
  have Q_ratio := 2
  have ratio_sum := P_ratio + Q_ratio
  have Q's_share := (T * Q_ratio) / ratio_sum
  
  -- Q's share of the profit
  sorry

end Q_share_of_profit_l541_541706


namespace points_ABC_inequality_l541_541414

theorem points_ABC_inequality (n : ℕ) (h : n ≥ 3) (points : Fin (n) → ℝ × ℝ) :
  ∃ A B C : Fin (n), 1 ≤ (euclidean_dist (points A) (points B)) / (euclidean_dist (points A) (points C)) ∧
              (euclidean_dist (points A) (points B)) / (euclidean_dist (points A) (points C)) ≤ (n + 1) / (n - 1) :=
sorry

end points_ABC_inequality_l541_541414


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541513

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541513


namespace speed_of_faster_train_l541_541297

theorem speed_of_faster_train (
    length_train : ℝ,
    speed_slower_train_kmph : ℝ,
    time_to_pass_sec : ℕ
) (h_length_train : length_train = 25)
  (h_speed_slower_train_kmph : speed_slower_train_kmph = 36)
  (h_time_to_pass_sec : time_to_pass_sec = 18) :
  let speed_faster_train_kmph := 46 in
  let rel_speed_mps := (speed_faster_train_kmph - speed_slower_train_kmph) * (5 / 18) in
  let distance_to_cover_m := 2 * length_train in
  distance_to_cover_m = rel_speed_mps * time_to_pass_sec := 
begin
  sorry
end

end speed_of_faster_train_l541_541297


namespace circle_points_sum_l541_541365

theorem circle_points_sum (n : ℕ) (h : n > 0) (points : finset ℕ) 
  (hpoints : points.card = 2 * n)
  (f : (fin 2n) → (fin 2n)) -- function that pairs points such that each point is connected exactly once without intersection
  (hf : ∀ i, i ≠ f i ∧ f (f i) = i ) : 
  ∃ pairs : finset (fin 2n × fin 2n), 
  pairs.card = n ∧ 
  (∀ a b, a ∈ pairs ∧ b ∈ pairs → disjoint (line a) (line b)) ∧ 
  (∑ xy in pairs, |xy.1 - xy.2|) = n^2 := 
sorry

end circle_points_sum_l541_541365


namespace inequality_holds_for_all_x_iff_range_m_l541_541609

theorem inequality_holds_for_all_x_iff_range_m (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Ioc (-10) 2 := by
  sorry

end inequality_holds_for_all_x_iff_range_m_l541_541609


namespace gain_percentages_l541_541025

theorem gain_percentages (A B C : ℝ) :
  let gain_percentage (gain cost_price : ℝ) := (gain / cost_price) * 100 in
  gain_percentage 10 40 = 25 ∧ 
  gain_percentage 15 60 = 25 ∧ 
  gain_percentage 25 75 = 33.33 :=
by
  sorry

end gain_percentages_l541_541025


namespace equation_of_line_l541_541104

-- Definitions related to conditions
def is_vertical_line (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, l x y

def passes_through_P (l : ℝ → ℝ → Prop) : Prop :=
  l 5 10

def distance_from_origin (l : ℝ → ℝ → Prop) (d : ℝ) : Prop :=
  ∃ k : ℝ, ( ∀ x y : ℝ, l x y → k * x - y - 5 * k + 10 = 0) ∧ (|-5 * k + 10| / sqrt (1 + k^2) = d)

-- Theorem statement
theorem equation_of_line (l : ℝ → ℝ → Prop) :
  (passes_through_P l ∧ distance_from_origin l 5) →
  (∀ x y : ℝ, l x y ↔ (x = 5 ∨ 3 * x - 4 * y + 25 = 0)) :=
begin
  sorry
end

end equation_of_line_l541_541104


namespace find_beta_minus_alpha_l541_541592

theorem find_beta_minus_alpha (α β : ℝ) (a b : ℝ × ℝ)
    (h1 : 0 < α ∧ α < β ∧ β < real.pi)
    (h2 : a = (real.cos α, real.sin α))
    (h3 : b = (real.cos β, real.sin β))
    (h4 : real.sqrt (9 * (a.1^2 + a.2^2) + (b.1^2 + b.2^2) + 6 * (a.1 * b.1 + a.2 * b.2)) = 
          real.sqrt ((a.1^2 + a.2^2) - 4 * (a.1 * b.1 + a.2 * b.2) + 4 * (b.1^2 + b.2^2))):
  β - α = 2 * real.pi / 3 := 
sorry

end find_beta_minus_alpha_l541_541592


namespace trapezoid_segment_length_l541_541748

theorem trapezoid_segment_length (a b : ℝ) (h : a > b) :
  let MN := (2 * a * b) / |a - b| in 
  MN = (2 * a * b) / |a - b| := 
by
  sorry

end trapezoid_segment_length_l541_541748


namespace sequence_properties_l541_541650

variable {x : ℕ → ℝ}

-- Conditions
def initial_condition (x1 : ℝ) := x1 > 3

def recurrence_relation (n : ℕ) (x_n_minus_1 : ℝ) : ℝ :=
  (3 * x_n_minus_1 ^ 2 - x_n_minus_1) / (4 * (x_n_minus_1 - 1))

-- Proposition to prove
theorem sequence_properties (x : ℕ → ℝ) (h1 : initial_condition (x 1)) :
  ∀ n : ℕ, 1 ≤ n → 3 < x (n + 1) ∧ x (n + 1) < x n := sorry

end sequence_properties_l541_541650


namespace bowling_tournament_prize_orders_l541_541416

theorem bowling_tournament_prize_orders : 
  let num_games := 5 in
  let choices_per_game := 2 in
  choices_per_game ^ num_games = 32 := 
by 
  -- Define variables
  let num_games := 5
  let choices_per_game := 2
  -- Calculate the number of possible orders
  calc
  choices_per_game ^ num_games 
      = 2 ^ 5 : by rfl
  ... = 32 : by norm_num

end bowling_tournament_prize_orders_l541_541416


namespace num_integer_pairs_l541_541086

theorem num_integer_pairs (m n : ℤ) :
  0 < m ∧ m < n ∧ n < 53 ∧ 53^2 + m^2 = 52^2 + n^2 →
  ∃ k, k = 3 := 
sorry

end num_integer_pairs_l541_541086


namespace exists_two_triangles_with_two_good_sides_l541_541643

-- Definitions based on the problem conditions
def is_good_side (side : ℝ × ℝ) (j k : ℤ) : Prop :=
side.1 = j ∨ side.2 = k

def triangle_with_good_sides (triangle : ℕ × ℕ × ℕ) (j k : ℤ) : Prop :=
(is_good_side (triangle.1, triangle.2) j k ∨ is_good_side (triangle.1, triangle.3) j k) ∧
is_good_side (triangle.2, triangle.3) j k

-- Formulating the proof problem
theorem exists_two_triangles_with_two_good_sides 
  (m n : ℕ) (h_m_odd : m % 2 = 1) (h_n_odd : n % 2 = 1)
  (partition : list (ℕ × ℕ × ℕ)) 
  (h_partition : ∀ t ∈ partition, triangle_with_good_sides t m n) :
  ∃ t₁ t₂ ∈ partition, triangle_with_good_sides t₁ m n ∧ triangle_with_good_sides t₂ m n :=
sorry

end exists_two_triangles_with_two_good_sides_l541_541643


namespace rocky_training_miles_l541_541633

variable (x : ℕ)

theorem rocky_training_miles (h1 : x + 2 * x + 6 * x = 36) : x = 4 :=
by
  -- proof
  sorry

end rocky_training_miles_l541_541633


namespace min_cards_to_ensure_all_suits_l541_541788

noncomputable def min_cards_for_all_suits (total_cards suits cards_per_suit : ℕ) : ℕ :=
  if h1: suits * cards_per_suit = total_cards then
    if h2: suits = 4 then
      if h3: cards_per_suit = 9 then
        28
      else
        sorry
    else
      sorry
  else
    sorry

theorem min_cards_to_ensure_all_suits : min_cards_for_all_suits 36 4 9 = 28 :=
by simp [min_cards_for_all_suits]; sorry

end min_cards_to_ensure_all_suits_l541_541788


namespace cupcakes_needed_l541_541193

theorem cupcakes_needed (x : ℕ) (students_per_class : ℕ) (students_PE_class : ℕ) (total_cupcakes : ℕ) 
  (h1 : students_per_class = 30) (h2 : students_PE_class = 50) (h3 : total_cupcakes = 140) : 
  30 * x + 50 = 140  → x = 3 :=
by
  assume h4 : 30 * x + 50 = 140
  sorry

end cupcakes_needed_l541_541193


namespace bogan_second_feeding_bogan_feeding_20_10_10_l541_541049

theorem bogan_second_feeding :
  (total_maggots first_feeding second_feeding : ℕ) →
  total_maggots = 20 →
  first_feeding = 10 →
  second_feeding = total_maggots - first_feeding →
  second_feeding = 10 :=
by
  intros total_maggots first_feeding second_feeding h_total h_first h_second
  rw [h_total, h_first] at h_second
  exact h_second

# To assert the main theorem correctly and avoid any local assumptions
theorem bogan_feeding_20_10_10 :
  (total_maggots second_feeding first_feeding : ℕ) →
  total_maggots = 20 →
  first_feeding = 10 →
  second_feeding = total_maggots - first_feeding →
  second_feeding = 10 :=
by
  intros total_maggots first_feeding second_feeding h_total h_first h_second
  exact bogan_second_feeding total_maggots first_feeding second_feeding h_total h_first h_second

end bogan_second_feeding_bogan_feeding_20_10_10_l541_541049


namespace parabola_equation_length_of_AB_l541_541018

-- Definitions
def vertex_at_origin (f : ℝ → ℝ) : Prop := f 0 = 0
def symmetric_about_x (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)
def passes_through_M (f : ℝ → ℝ) (M : ℝ × ℝ) : Prop := f M.1 = M.2
def focus_coor (f : ℝ → ℝ) : Prop := ∃ p : ℝ, 0 < p ∧ f (-p/2) = 0

-- Given data
def M : ℝ × ℝ := (-2, 2*real.sqrt 2)
def parabola (x : ℝ) : ℝ := real.sqrt (-4 * x)

-- Problems
theorem parabola_equation : (vertex_at_origin parabola ∧ symmetric_about_x parabola ∧ passes_through_M parabola M) →
  ∃ p : ℝ, parabola = (λ x, real.sqrt (-4 * x)) ∧ focus_coor parabola :=
    by sorry

theorem length_of_AB : (focus_coor parabola ∧ line (1, 0) 45 passes_through focus) →
  chord_length parabola 8 :=
    by sorry

end parabola_equation_length_of_AB_l541_541018


namespace XY_parallel_X_l541_541962

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541962


namespace infinite_sum_value_l541_541757

noncomputable def a : ℕ → ℝ
| 0     => 2015
| 1     => 2015
| (n+2) => (a n - 1) * (b (n+1) + 1)

noncomputable def b : ℕ → ℝ
| 0     => 2013
| 1     => 2013
| (n+2) => a (n+1) * b n - 1

theorem infinite_sum_value :
  (∑' n, b n * (1 / a (n+1) - 1 / a (n+3))) = 1 + 1 / (2014 * 2015) :=
by
  sorry

end infinite_sum_value_l541_541757


namespace find_sum_of_money_invested_l541_541349

theorem find_sum_of_money_invested (P : ℝ) (h1 : SI_15 = P * (15 / 100) * 2)
                                    (h2 : SI_12 = P * (12 / 100) * 2)
                                    (h3 : SI_15 - SI_12 = 720) : 
                                    P = 12000 :=
by
  -- Skipping the proof
  sorry

end find_sum_of_money_invested_l541_541349


namespace treaty_of_versailles_original_day_l541_541736

-- Define the problem in Lean terms
def treatySignedDay : Nat -> Nat -> String
| 1919, 6 => "Saturday"
| _, _ => "Unknown"

-- Theorem statement
theorem treaty_of_versailles_original_day :
  treatySignedDay 1919 6 = "Saturday" :=
sorry

end treaty_of_versailles_original_day_l541_541736


namespace fraction_of_number_l541_541325

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541325


namespace largest_power_of_18_dividing_factorial_30_l541_541494

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541494


namespace tan_add_sin_l541_541434

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l541_541434


namespace no_primes_of_form_2pow5m_plus_2powm_plus_1_l541_541678

theorem no_primes_of_form_2pow5m_plus_2powm_plus_1 {m : ℕ} (hm : m > 0) : ¬ (Prime (2^(5*m) + 2^m + 1)) :=
by
  sorry

end no_primes_of_form_2pow5m_plus_2powm_plus_1_l541_541678


namespace determine_hyperbola_l541_541999

noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def parabola_equation (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem determine_hyperbola (a b p : ℝ) (x y : ℝ) :
  (0 < a ∧ 0 < b ∧
   hyperbola_equation a b x y ∧
   distance (-a, 0) (1, 0) = 3 ∧
   (-1, 1) = (interpreting asymptote of hyperbola with parabola's axis intersection)) →
  a = 4 ∧ b = 4 ∧ ∀ x y, hyperbola_equation 4 4 x y :=
by sorry

end determine_hyperbola_l541_541999


namespace positive_roots_l541_541249

noncomputable def sigma (n : ℕ) (x : Fin n → ℝ) : ℕ → ℝ
| 0       := 1
| 1       := ∑ i, x i
| (k + 1) := ∑ i in Finset.subsets (Finset.univ : Finset (Fin n)) (k + 2), ∏ j in i, x j

theorem positive_roots {n : ℕ} {x : Fin n → ℝ} (h1 : sigma n x 1 > 0)
  (h2 : sigma n x 2 > 0) (h3 : n > 2 → sigma n x 3 > 0) (h4 : n > 3 → sigma n x 4 > 0)
  -- Further conditions up to h5, h6, ..., hn
: ∀ i, 0 ≤ i → i < n → x i > 0 :=
by
  sorry

end positive_roots_l541_541249


namespace select_non_adjacent_row_select_non_adjacent_circle_l541_541763

-- Definitions for combinatorial binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Row of 2n people
theorem select_non_adjacent_row (n r : ℕ) (h : r < n) :
  ∑ (k : ℕ) in (finset.Ico 1 2*n), binom (2*n - r + 1) r = 
  binom (2*n - r + 1) r :=
sorry

-- Problem 2: Circle of 2n people
theorem select_non_adjacent_circle (n r : ℕ) (h : r < n) :
  ∑ (k : ℕ) in (finset.Ico 1 2*n), 2*n/(2*n - r) * binom (2*n - r) r = 
  2*n/(2*n - r) * binom (2*n - r) r :=
sorry

end select_non_adjacent_row_select_non_adjacent_circle_l541_541763


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541512

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541512


namespace fraction_of_number_l541_541321

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541321


namespace no_solutions_l541_541903

theorem no_solutions {x y : ℤ} :
  (x ≠ 1) → (y ≠ 1) →
  ((x^7 - 1) / (x - 1) = y^5 - 1) →
  false :=
by sorry

end no_solutions_l541_541903


namespace find_f_neg4_l541_541102

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 - a * x + b

theorem find_f_neg4 (a b : ℝ) (h1 : f 1 a b = -1) (h2 : f 2 a b = 2) : 
  f (-4) a b = 14 :=
by
  sorry

end find_f_neg4_l541_541102


namespace cos_ratio_of_cot_l541_541140

theorem cos_ratio_of_cot (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x ∧ x < Real.pi / 2) 
  (h4 : Real.cot x = (a^2 - b^2) / (2 * a * b)) : 
  Real.cos x = (a^2 - b^2) / (a^2 + b^2) :=
sorry

end cos_ratio_of_cot_l541_541140


namespace correlation_function_proof_l541_541078

noncomputable def spectral_density (s0 : ℝ) (ω ω0 : ℝ) : ℝ :=
  if -ω0 ≤ ω ∧ ω ≤ ω0 then s0 else 0

noncomputable def correlation_function (s0 ω0 τ : ℝ) : ℝ :=
  if τ = 0 then 2 * s0 * ω0 else (2 * s0 * sin (ω0 * τ)) / τ

theorem correlation_function_proof (s0 ω0 τ : ℝ) :
  let s_x : ℝ → ℝ := spectral_density s0 ω0 in
  let k_x : ℝ → ℝ := correlation_function s0 ω0 in
  k_x τ = (2 * s0 * sin (ω0 * τ)) / τ :=
by
  sorry

end correlation_function_proof_l541_541078


namespace XY_parallel_X_l541_541966

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541966


namespace tan_15pi_over_4_is_neg1_l541_541860

noncomputable def tan_15pi_over_4 : ℝ :=
  Real.tan (15 * Real.pi / 4)

theorem tan_15pi_over_4_is_neg1 :
  tan_15pi_over_4 = -1 :=
sorry

end tan_15pi_over_4_is_neg1_l541_541860


namespace XY_parallel_X_l541_541935

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541935


namespace negation_of_universal_proposition_l541_541747

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
  sorry

end negation_of_universal_proposition_l541_541747


namespace triangle_obtuse_if_sin_a_cos_b_lt_zero_l541_541183

theorem triangle_obtuse_if_sin_a_cos_b_lt_zero 
  (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < π) 
  (h3 : 0 < B) 
  (h4 : B < π) 
  (h5 : sin A * cos B < 0) 
  : B > π / 2 :=
sorry

end triangle_obtuse_if_sin_a_cos_b_lt_zero_l541_541183


namespace marys_garbage_bill_is_correct_l541_541685

noncomputable def weekly_cost_trash_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_cost_recycling_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_total_cost (trash_cost : ℝ) (recycling_cost : ℝ) : ℝ :=
  trash_cost + recycling_cost

noncomputable def monthly_cost (weekly_cost : ℝ) (num_weeks : ℕ) : ℝ :=
  weekly_cost * num_weeks

noncomputable def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount_and_fine (total_cost : ℝ) (discount : ℝ) (fine : ℝ) : ℝ :=
  total_cost - discount + fine

theorem marys_garbage_bill_is_correct :
  let weekly_trash_cost := weekly_cost_trash_bin 10 2 in
  let weekly_recycling_cost := weekly_cost_recycling_bin 5 1 in
  let weekly_total := weekly_total_cost weekly_trash_cost weekly_recycling_cost in
  let monthly_total := monthly_cost weekly_total 4 in
  let senior_discount := discount monthly_total 0.18 in
  let fine := 20 in
  total_cost_after_discount_and_fine monthly_total senior_discount fine = 102 :=
by
  sorry

end marys_garbage_bill_is_correct_l541_541685


namespace parallel_lines_triangle_l541_541947

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541947


namespace sale_price_with_50_percent_profit_l541_541250

theorem sale_price_with_50_percent_profit 
  (P1 P2 P3 : ℝ)
  (h1 : 872 - P1 = P2 - 448)
  (h2 : 650 - P3 = P1 - 550) :
  (P1 * 1.5, P2 * 1.5, P3 * 1.5) = (sale_price_A, sale_price_B, sale_price_C) :=
begin
  sorry
end

end sale_price_with_50_percent_profit_l541_541250


namespace relationship_of_ys_l541_541107

theorem relationship_of_ys :
  ∀ (y₁ y₂ y₃ : ℝ),
    (A : ℝ × ℝ) → (B : ℝ × ℝ) → (C : ℝ × ℝ) →
    A = (-3, y₁) → B = (0, y₂) → C = (3, y₃) →
    (∀ x : ℝ, x ∈ {-3, 0, 3} → ∃ y : ℝ, y = -((x + 2) ^ 2) + 4 ∧
      (A = (-3, y₁) ∧ y₁ = 3) ∧
      (B = (0, y₂) ∧ y₂ = 0) ∧
      (C = (3, y₃) ∧ y₃ = -21)) →
    y₃ < y₂ ∧ y₂ < y₁ := 
by
  intros y₁ y₂ y₃ A B C hA hB hC h_func
  sorry

end relationship_of_ys_l541_541107


namespace largest_c_for_sum_squares_ineq_l541_541081

noncomputable theory
open_locale big_operators

-- Define median of a list of real numbers
def median {n : ℕ} (x : fin n → ℝ) : ℝ :=
if n % 2 = 1 then x (fin.of_nat' ((n / 2 : ℕ) + 1)) else (x (fin.of_nat' (n / 2)) + x (fin.of_nat' (n / 2 + 1))) / 2

theorem largest_c_for_sum_squares_ineq :
  ∃ c : ℝ,
    (∀ (x : fin 101 → ℝ),
      (∑ i, x i) = 0 →
      let M := median x in ∑ i, (x i)^2 ≥ c * M^2) ∧
    c = 5151 / 50 :=
by {
  use 5151 / 50,
  intros x sum_zero M,
  sorry
}

end largest_c_for_sum_squares_ineq_l541_541081


namespace midpoint_incenter_l541_541985

variables {A B C E F P Q O : Type} -- Defining variables

-- Assuming necessary conditions
variable [Incircle : ∀ (Δ : Triangle), Circle (Inscribed Δ) = O]
variable [TangentToSide : ∀ (Δ : Triangle) (s1 s2 : Side Δ), ∃ Q (tangent Q s1) (tangent Q s2)]
variable [Midpoint : ∀ E F, Point (Midsegment EF) = P] -- Midpoint P of EF

-- To prove
theorem midpoint_incenter (ABC : Triangle) (EF : Segment) (P : Point) (O Q : Circle) (E F : Point)
    (incircle_O : Circle O = Incircle ABC)
    (tangent_E : Q.Tangent E AB)
    (tangent_F : Q.Tangent F AC)
    (midpoint_EF : Midpoint E F P) :
  IsIncenter ABC P :=
  sorry

end midpoint_incenter_l541_541985


namespace find_four_numbers_l541_541975

theorem find_four_numbers
  (a d : ℕ)
  (h_pos : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h_sum : (a - d) + a + (a + d) = 48)
  (b c : ℕ)
  (h_geo : b = a ∧ c = a + d)
  (last : ℕ)
  (h_last_val : last = 25)
  (h_geometric_seq : (a + d) * (a + d) = b * last)
  : (a - d, a, a + d, last) = (12, 16, 20, 25) := 
  sorry

end find_four_numbers_l541_541975


namespace new_volume_is_correct_l541_541378

variable (l w h : ℝ)

-- Conditions given in the problem
axiom volume : l * w * h = 4320
axiom surface_area : 2 * (l * w + w * h + h * l) = 1704
axiom edge_sum : 4 * (l + w + h) = 208

-- The proposition we need to prove:
theorem new_volume_is_correct : (l + 2) * (w + 2) * (h + 2) = 6240 :=
by
  -- Placeholder for the actual proof
  sorry

end new_volume_is_correct_l541_541378


namespace sum_of_remainders_when_digit_is_increasing_l541_541073

theorem sum_of_remainders_when_digit_is_increasing (d : ℕ) (hd : d ≤ 4) : 
  (Σ x in {n | n ∈ range 5}, (5 * x + 12) % 37) = 110 :=
by
  sorry

end sum_of_remainders_when_digit_is_increasing_l541_541073


namespace monotonic_intervals_f_extreme_value_g_l541_541579

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1

noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem monotonic_intervals_f (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2, x1 < x2 → f x1 a < f x2 a) ∧
  (a > 0 → (∀ x1 x2, x1 < x2 ∧ x1 > Real.log a → f x1 a < f x2 a) ∧
           (∀ x1 x2, x1 < x2 ∧ x2 < Real.log a → f x1 a > f x2 a)) := sorry

theorem extreme_value_g :
  ∃ x ∈ Set.Ioi (0 : ℝ), (∀ y ∈ Set.Ioo (0 : x), g y < g x) ∧
                         (∀ y ∈ Set.Ioi x, g y < g x) ∧ g 1 = -1 := sorry

end monotonic_intervals_f_extreme_value_g_l541_541579


namespace inequality_holds_for_all_x_iff_l541_541611

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end inequality_holds_for_all_x_iff_l541_541611


namespace diameter_twice_radius_l541_541178

theorem diameter_twice_radius (r d : ℝ) (h : d = 2 * r) : d = 2 * r :=
by
  exact h

end diameter_twice_radius_l541_541178


namespace irreducible_fractions_less_than_50_l541_541373

noncomputable def gcd_is_1 (n : ℕ) : Prop := 
  Int.gcd (5 * n + 6) (6 * n + 5) = 1

def count_irreducible_fractions (N : ℕ) : ℕ :=
  (Finset.range (N + 1)).filter gcd_is_1 |>.card

theorem irreducible_fractions_less_than_50 :
  count_irreducible_fractions 49 = 45 := sorry

end irreducible_fractions_less_than_50_l541_541373


namespace tan_angle_sum_l541_541566

noncomputable def α : ℝ := by sorry  -- We assume α is given in the second quadrant

theorem tan_angle_sum :
  ∀ α : ℝ, (sin α = 3 / 5) → (π / 2 < α ∧ α < π) → tan (α + π / 4) = 1 / 7 :=
by
  sorry

end tan_angle_sum_l541_541566


namespace solve_for_difference_l541_541108

variable (a b : ℝ)

theorem solve_for_difference (h1 : a^3 - b^3 = 4) (h2 : a^2 + ab + b^2 + a - b = 4) : a - b = 2 :=
sorry

end solve_for_difference_l541_541108


namespace prism_volume_l541_541822

theorem prism_volume (s : ℝ) : (s = 2) → (∀ s ≤ 2) → s ^ 3 = 8 :=
by
  intro h1 h2
  sorry

end prism_volume_l541_541822


namespace product_rs_l541_541568

variable (r s : ℝ)
variable (h1 : r > 0) (h2 : s > 0)
variable (h3 : r^3 + s^3 = 1)
variable (h4 : r^6 + s^6 = 15 / 16)

theorem product_rs : r * s = real.cbrt (1 / 48) :=
by
  sorry

end product_rs_l541_541568


namespace sum_squares_alternating_l541_541712

theorem sum_squares_alternating (n : ℕ) (h : n > 0) : 
  ∑ i in finset.range n, if even i then -(i + 1)^2 else i^2 = -(n * (2 * n + 1)) :=
by
  sorry

end sum_squares_alternating_l541_541712


namespace sales_first_month_l541_541017

theorem sales_first_month (S1 S2 S3 S4 S5 S6 : ℝ) 
  (h2 : S2 = 7000) (h3 : S3 = 6800) (h4 : S4 = 7200) (h5 : S5 = 6500) (h6 : S6 = 5100)
  (avg : (S1 + S2 + S3 + S4 + S5 + S6) / 6 = 6500) : S1 = 6400 := by
  sorry

end sales_first_month_l541_541017


namespace max_area_rectangle_edge_length_l541_541988

theorem max_area_rectangle_edge_length (x : ℝ) (h1 : 0 < x) (h2 : x < 2) :
  let y := 4 - x^2 in
  let area := 2 * x * y in
  y = 4 - x^2 → (∀ z : ℝ, (0 < z) → (z < 2) → 2 * z * (4 - z^2) ≤ area) → 
  2 * x = (4 / 3) * Real.sqrt 3 := 
by
  sorry

end max_area_rectangle_edge_length_l541_541988


namespace part_a_part_b_l541_541676

-- Define the integral U_n
noncomputable def U (n : ℕ) : ℝ := ∫ (x : ℝ) in 0..π/2, (real.cos x)^n

-- Factorial with double exclamation (double factorial)
noncomputable def double_fact (n : ℕ) : ℕ :=
if h : n = 0 ∨ n = 1 then 1 else n * double_fact (n - 2)

-- The main theorem to be proved for part (a)
theorem part_a (n : ℕ) : 
  if n % 2 = 1 then U n = (double_fact (n - 1) : ℝ) / (double_fact n)
  else U n = (double_fact (n - 1) : ℝ) / (double_fact n) * (π / 2) := 
sorry

-- The main theorem to be proved for part (b)
theorem part_b : 
  (real.pi : ℝ) = 
  real.lim (λ n : ℕ, 1 / (n + 1) * ((double_fact (2 * (n+1))) : ℝ / 
  (double_fact (2 * (n+1) - 1))) ^ 2) := 
sorry

end part_a_part_b_l541_541676


namespace stereos_production_fraction_l541_541062

/-
Company S produces three kinds of stereos: basic, deluxe, and premium.
Of the stereos produced by Company S last month, 2/5 were basic, 3/10 were deluxe, and the rest were premium.
It takes 1.6 as many hours to produce a deluxe stereo as it does to produce a basic stereo, and 2.5 as many hours to produce a premium stereo as it does to produce a basic stereo.
Prove that the number of hours it took to produce the deluxe and premium stereos last month was 123/163 of the total number of hours it took to produce all the stereos.
-/

def stereos_production (total_stereos : ℕ) (basic_ratio deluxe_ratio : ℚ)
  (deluxe_time_multiplier premium_time_multiplier : ℚ) : ℚ :=
  let basic_stereos := total_stereos * basic_ratio
  let deluxe_stereos := total_stereos * deluxe_ratio
  let premium_stereos := total_stereos - basic_stereos - deluxe_stereos
  let basic_time := basic_stereos
  let deluxe_time := deluxe_stereos * deluxe_time_multiplier
  let premium_time := premium_stereos * premium_time_multiplier
  let total_time := basic_time + deluxe_time + premium_time
  (deluxe_time + premium_time) / total_time

-- Given values
def total_stereos : ℕ := 100
def basic_ratio : ℚ := 2 / 5
def deluxe_ratio : ℚ := 3 / 10
def deluxe_time_multiplier : ℚ := 1.6
def premium_time_multiplier : ℚ := 2.5

theorem stereos_production_fraction : stereos_production total_stereos basic_ratio deluxe_ratio deluxe_time_multiplier premium_time_multiplier = 123 / 163 := by
  sorry

end stereos_production_fraction_l541_541062


namespace largest_n_dividing_factorial_l541_541518

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541518


namespace chord_length_l541_541523

-- Define the polar equation of the curve
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = sqrt 2 * cos (θ + π/4)

-- Define the parametric equations of the line
def line_parametric (x y t : ℝ) : Prop := 
  (x = 1 + 4/5 * t) ∧ 
  (y = -1 - 3/5 * t)

-- Prove that the length of the chord intercepted by the curve from the line is 7/5
theorem chord_length (ρ θ x y t l : ℝ) 
  (h1 : polar_equation ρ θ)
  (h2 : line_parametric x y t) : 
  l = 7/5 :=
sorry

end chord_length_l541_541523


namespace coin_flip_probability_l541_541372

theorem coin_flip_probability :
  let total_flips := 10
  let fixed_heads := 2
  let outcomes := 2^8
  let successful_outcomes := Nat.choose 8 5 + Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8
  outcomes = 256 → successful_outcomes = 93 →
  (successful_outcomes / outcomes : ℝ) = 93 / 256 := by
  have fixed_heads_correct : fixed_heads = 2 := rfl
  have total_flips_correct : total_flips = 10 := rfl
  have outcomes_correct : outcomes = 2^8 := rfl
  have successful_outcomes_correct : successful_outcomes = Nat.choose 8 5 + Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8 := rfl
  sorry

end coin_flip_probability_l541_541372


namespace domain_of_rational_function_l541_541879

theorem domain_of_rational_function 
  (c : ℝ) 
  (h : -7 * (6 ^ 2) + 28 * c < 0) : 
  c < -9 / 7 :=
by sorry

end domain_of_rational_function_l541_541879


namespace simplify_and_evaluate_expr_l541_541240

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((2 * x / (x^2 - 4) - 1 / (x + 2)) / (x - 1) / (x - 2)) = Real.sqrt 3 / 3 :=
by
  unfold Prove
  apply sorry

end simplify_and_evaluate_expr_l541_541240


namespace hyperbola_focal_chord_length_l541_541874

theorem hyperbola_focal_chord_length (a : ℝ) (h : 0 < a) :
  (∃ l₁ l₂ l₃ l₄ : set (ℝ × ℝ), -- Define lines l₁, l₂, l₃, l₄
    (∀ i ∈ {l₁, l₂, l₃, l₄}, -- Each line must intersect the hyperbola at two points A and B
      ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ i ∧ B ∈ i ∧ 
      (( A.1^2 / a^2 - A.2^2 / 4 = 1) ∧ (B.1^2 / a^2 - B.2^2 / 4 = 1) ∧
        (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 64))) -- Ensuring |AB| = 8
  ) ↔ (1 < a ∧ a < 4) := sorry

end hyperbola_focal_chord_length_l541_541874


namespace solve_equation_16_l541_541241

theorem solve_equation_16 (
    x y : ℝ
) (h : 16^(x^2 + y) + 16^(y^2 + x) = 1) : x = -0.5 :=
begin
  sorry
end

end solve_equation_16_l541_541241


namespace largest_power_of_18_dividing_factorial_30_l541_541492

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541492


namespace fraction_of_number_l541_541317

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541317


namespace term_167_is_81281_l541_541756

-- Define the sequence of natural numbers as described
def sequence : List ℕ := 
  [1, 5, 6, 25, 26, 30, 31, 78125, 3125, 25, 5, 1, 81281 -- etc.]

-- Define the function to get the nth term of the sequence
def nth_term (n : ℕ) : ℕ :=
  sequence.get? (n - 1) |>.getD 0

-- Prove that the 167th term in the sequence is 81281
theorem term_167_is_81281 : nth_term 167 = 81281 := 
by 
  -- statement only, no proof required
  sorry

end term_167_is_81281_l541_541756


namespace fraction_multiplication_l541_541309

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541309


namespace number_of_negative_numbers_l541_541842

def count_negatives : List ℝ → ℕ :=
  List.foldl (fun acc x => if x < 0 then acc + 1 else acc) 0

theorem number_of_negative_numbers 
  (S : List ℝ) 
  (h : S = [2, -0.4, 0, -3, 13 / 9, -1.2, 2023, 0.6]) : 
  count_negatives S = 3 := 
by
  rw [h]
  sorry

end number_of_negative_numbers_l541_541842


namespace correct_expression_unique_correct_expression_l541_541399

-- Definitions for each of the given conditions
def A := (x + 2) * (x + 3)
def B := (x + 2) * (x - 3)
def C := (x + 6) * (x - 1)
def D := (x - 2) * (x - 3)

-- Correct answer
theorem correct_expression : C = x^2 + 5*x - 6 :=
by sorry

-- Proving that C is the only correct answer
theorem unique_correct_expression :
  (A ≠ x^2 + 5*x - 6) ∧
  (B ≠ x^2 + 5*x - 6) ∧
  (C = x^2 + 5*x - 6) ∧
  (D ≠ x^2 + 5*x - 6) :=
by sorry

end correct_expression_unique_correct_expression_l541_541399


namespace geometric_seq_common_ratio_l541_541921

theorem geometric_seq_common_ratio (a : ℕ → ℝ) (m : ℝ) (h_sum : ∀ n : ℕ, S n = 3 * 2^n + m) : 
  (common_ratio a = 2) :=
sorry

end geometric_seq_common_ratio_l541_541921


namespace largest_n_dividing_30_factorial_l541_541482

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541482


namespace sqrt_approx_is_correct_l541_541058

noncomputable def decimal_with_ones (n : ℕ) : ℚ :=
  (10^n - 1) / (9 * 10^n)

noncomputable def sqrt_approx_decimal_with_ones : ℚ :=
  1 / 3 * Real.sqrt (1 - 10^(-100))

theorem sqrt_approx_is_correct :
  ∃ (x : ℚ), abs (sqrt_approx_decimal_with_ones - x) < 10^(-200) :=
by
  let n := 100
  have x := decimal_with_ones n
  have approx := sqrt_approx_decimal_with_ones
  use 0.10049987498
  sorry

end sqrt_approx_is_correct_l541_541058


namespace tan_add_sin_l541_541433

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l541_541433


namespace rectangle_to_triangle_l541_541823

theorem rectangle_to_triangle (L W s : ℝ) (h₁ : L * W = (s^2 * real.sqrt 3) / 4) : 
  s = real.sqrt ((4 * L * W) / real.sqrt 3) :=
sorry

end rectangle_to_triangle_l541_541823


namespace cricket_team_right_handed_players_l541_541701

theorem cricket_team_right_handed_players:
  let total_players := 300
  let throwers := 165
  let hitters := 68
  let runners := 68
  let right_handed_players := throwers + ((3 / 5) * hitters).toNat + ((4 / 7) * runners).toNat
  (300 - 165 = 136) → 
  (136 / 2 = 68) →
  ( right_handed_players = 243)
:= by
  intros h1 h2
  sorry

end cricket_team_right_handed_players_l541_541701


namespace locus_of_P_l541_541092

noncomputable def point_locus
    (L1 L2 L3 L4 : Set ℝ×ℝ) 
    (parallel : ∀ {ℓ₁ ℓ₂ ℓ₃ ℓ₄ : Set ℝ×ℝ}, (ℓ₁.parallel ℓ₃) → (ℓ₂.parallel ℓ₄)) 
    (x y : ℝ) 
    (perp_dist : ∀ (P : ℝ×ℝ) (ℓ : Set ℝ×ℝ), ℝ)
    (D : ℝ) : Set (ℝ×ℝ) :=
    let d1 := perp_dist P L1
    let d2 := perp_dist P L2
    let d3 := perp_dist P L3
    let d4 := perp_dist P L4
    if D < x + y 
    then ∅
    else if D = x + y 
         then { P | ∀ (P : ℝ×ℝ), d1 + d2 + d3 + d4 = x + y }
         else { P | ∀ (P : ℝ×ℝ), d1 + d2 + d3 + d4 > x + y }

theorem locus_of_P (L1 L2 L3 L4 : Set ℝ×ℝ) 
    (parallel : ∀ {ℓ₁ ℓ₂ ℓ₃ ℓ₄ : Set ℝ×ℝ}, (ℓ₁.parallel ℓ₃) → (ℓ₂.parallel ℓ₄))
    (x y : ℝ) 
    (perp_dist : ∀ (P : ℝ×ℝ) (ℓ : Set ℝ×ℝ), ℝ) 
    (D : ℝ) :
    ∀ (P : ℝ×ℝ),
    point_locus L1 L2 L3 L4 parallel x y perp_dist D P :=
    by
        intros P
        sorry

end locus_of_P_l541_541092


namespace largest_n_18n_divides_30_factorial_l541_541505

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541505


namespace sine_of_angle_between_line_and_plane_l541_541671

theorem sine_of_angle_between_line_and_plane (d n : ℝ × ℝ × ℝ) :
  let θ := real.arcsin (34 / (15 * real.sqrt 7)) in
  d = (4, 5, 8) → n = (1, -2, 5) →
  sin θ = 34 / (15 * real.sqrt 7) :=
by
  assume d_eq : d = (4, 5, 8),
  assume n_eq : n = (1, -2, 5),
  sorry

end sine_of_angle_between_line_and_plane_l541_541671


namespace correct_subtraction_result_l541_541639

-- Definition of numbers:
def tens_digit := 2
def ones_digit := 4
def correct_number := 10 * tens_digit + ones_digit
def incorrect_number := 59
def incorrect_result := 14
def Z := incorrect_result + incorrect_number

-- Statement of the theorem
theorem correct_subtraction_result : Z - correct_number = 49 :=
by
  sorry

end correct_subtraction_result_l541_541639


namespace median_inequalities_l541_541187

variables {A B C : Type} [linear_ordered_ring A] [linear_ordered_ring B] [linear_ordered_ring C]
variables (a b c : A) (m_a m_b m_c : B) (triangle : C)

noncomputable def median (x y z : A) : B := sqrt ((2 * y^2 + 2 * z^2 - x^2) / 4)

theorem median_inequalities :
  (∀ (a b c : A) (m_a m_b m_c : B) (in_triangle : C),
  m_a = median a b c ∧ m_b = median b c a ∧ m_c = median c a b) →
  (∑ m_a^2 / (b * c) ≥ 9 / 4) ∧ (∑ (m_b^2 + m_c^2 - m_a^2) / (b * c) ≥ 9 / 4) :=
by { sorry } 

end median_inequalities_l541_541187


namespace buratino_candy_spent_l541_541853

theorem buratino_candy_spent :
  ∃ (x y : ℕ), x + y = 50 ∧ 2 * x = 3 * y ∧ y * 5 - x * 3 = 10 :=
by {
  -- Declaration of variables and goal
  let x := 30,
  let y := 20,
  use [x, y],
  split,
  { exact rfl },
  split,
  { exact rfl },
  { exact rfl }
}

end buratino_candy_spent_l541_541853


namespace largest_n_dividing_factorial_l541_541519

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541519


namespace students_taking_German_l541_541152

theorem students_taking_German 
  (total_students : ℕ)
  (students_taking_French : ℕ)
  (students_taking_both : ℕ)
  (students_not_taking_either : ℕ) 
  (students_taking_German : ℕ) 
  (h1 : total_students = 69)
  (h2 : students_taking_French = 41)
  (h3 : students_taking_both = 9)
  (h4 : students_not_taking_either = 15)
  (h5 : students_taking_German = 22) :
  total_students - students_not_taking_either = students_taking_French + students_taking_German - students_taking_both :=
sorry

end students_taking_German_l541_541152


namespace average_expression_l541_541739

-- Define a theorem to verify the given problem
theorem average_expression (E a : ℤ) (h1 : a = 34) (h2 : (E + (3 * a - 8)) / 2 = 89) : E = 84 :=
by
  -- Proof goes here
  sorry

end average_expression_l541_541739


namespace percent_savings_12_roll_package_l541_541811

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l541_541811


namespace quadratic_roots_unique_l541_541912

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l541_541912


namespace find_angle_between_lateral_edge_and_base_plane_l541_541020

-- Definitions for necessary geometric entities
noncomputable def angle_between_lateral_edge_and_base (a : ℝ) (sqrt33 : ℝ) : ℝ :=
  arcsin (1 + sqrt33 / 8)

-- Theorem to prove
theorem find_angle_between_lateral_edge_and_base_plane
  (a : ℝ) (sqrt33 : ℝ)
  (plane_intersects_at_right_angle : ∀ (plane : ℝ → ℝ → ℝ), (∃ v w : ℝ, plane v w = 0))
  (cross_section_area_half_base : ∀ (area_base section_area : ℝ), section_area = area_base / 2) :
  angle_between_lateral_edge_and_base a sqrt33 = arcsin (1 + sqrt33 / 8) :=
by
  sorry

end find_angle_between_lateral_edge_and_base_plane_l541_541020


namespace incorrect_option_C_l541_541539

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)
def g (x : ℝ) : ℝ := Real.cos (x - Real.pi / 2)
def y (x : ℝ) : ℝ := f x * g x 

theorem incorrect_option_C :
  ¬ (∀ x : ℝ, y (Real.pi / 4) = y x ↔ x = Real.pi / 4) := sorry

end incorrect_option_C_l541_541539


namespace area_of_triangle_ABC_circumcenter_of_triangle_ABC_l541_541572

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨8, 3⟩

def triangle_area (A B C : Point) : ℚ := by
  -- area calculation will be filled here
  sorry

def circumcenter (A B C : Point) : Point := by
  -- circumcenter calculation will be filled here
  sorry

theorem area_of_triangle_ABC : triangle_area A B C = 16 :=
  sorry

theorem circumcenter_of_triangle_ABC : circumcenter A B C = ⟨9/2, 7/2⟩ :=
  sorry

end area_of_triangle_ABC_circumcenter_of_triangle_ABC_l541_541572


namespace log_base_change_l541_541776

theorem log_base_change (a b : ℝ) (h₁ : Real.log 2 / Real.log 10 = a) (h₂ : Real.log 3 / Real.log 10 = b) :
    Real.log 18 / Real.log 5 = (a + 2 * b) / (1 - a) := by
  sorry

end log_base_change_l541_541776


namespace isosceles_triangle_sine_of_larger_angle_l541_541165

theorem isosceles_triangle_sine_of_larger_angle (r : ℝ) (triangle_triangle : Triangle ℝ)
  (isosceles : triangle.is_isosceles)
  (base_eq_four_r : triangle.base = 4*r) :
  Real.sin triangle.larger_angle = 24 / 25 :=
by
  sorry

end isosceles_triangle_sine_of_larger_angle_l541_541165


namespace smallest_angle_multiplied_l541_541845

noncomputable def smallestAngleInDivisibleIsoscelesTriangle (alpha : ℝ) : Prop :=
α > 0 ∧ 180 > 2α ∧ (360 = 5α ∨ 7α = 540)

theorem smallest_angle_multiplied :
  ∀ α : ℝ, smallestAngleInDivisibleIsoscelesTriangle α →
  ∃ smallest_angle : ℝ, smallest_angle = (180 / 7) ∧ smallest_angle * 6006 = 154440 
by
  intro α h
  use 180 / 7
  split
  {
    -- Prove that smallest_angle = 180 / 7
    sorry
  }
  {
    -- Prove that smallest_angle * 6006 = 154440
    sorry
  }

end smallest_angle_multiplied_l541_541845


namespace red_ball_higher_than_green_l541_541380

noncomputable def P_red (k : ℕ) : ℝ :=
  (1 / 2 : ℝ) ^ k

noncomputable def P_green (k : ℕ) : ℝ :=
  if k % 2 = 0 then (1 / 2 : ℝ) ^ (k - 1) else (1 / 2 : ℝ) ^ (k + 1)

theorem red_ball_higher_than_green :
  (∑' (j : ℕ), ∑ (k in finset.range j), P_red j * P_green k) = 5 / 32 :=
sorry

end red_ball_higher_than_green_l541_541380


namespace fraction_of_number_l541_541301

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541301


namespace expected_reflections_correct_l541_541641

-- Define the given conditions
def table_length : ℝ := 3
def table_width : ℝ := 1
def initial_position := (table_length / 2, table_width / 2)
def travel_distance : ℝ := 2

-- Define the expected calculation
def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

-- The Lean statement to prove
theorem expected_reflections_correct :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end expected_reflections_correct_l541_541641


namespace XY_parallel_X_l541_541957

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541957


namespace hcf_of_36_and_x_is_12_l541_541472

theorem hcf_of_36_and_x_is_12 (x : ℕ) (h : Nat.gcd 36 x = 12) : x = 48 :=
sorry

end hcf_of_36_and_x_is_12_l541_541472


namespace lemon_pie_degrees_l541_541623

-- Defining the constants
def total_students : ℕ := 45
def chocolate_pie : ℕ := 15
def apple_pie : ℕ := 10
def blueberry_pie : ℕ := 9

-- Defining the remaining students
def remaining_students := total_students - (chocolate_pie + apple_pie + blueberry_pie)

-- Half of the remaining students prefer cherry pie and half prefer lemon pie
def students_prefer_cherry : ℕ := remaining_students / 2
def students_prefer_lemon : ℕ := remaining_students / 2

-- Defining the degree measure function
def degrees (students : ℕ) := (students * 360) / total_students

-- Proof statement
theorem lemon_pie_degrees : degrees students_prefer_lemon = 48 := by
  sorry  -- proof omitted

end lemon_pie_degrees_l541_541623


namespace problem_solution_l541_541168

variables (Emma George Isla Harry Ann : ℕ)

-- Conditions from the problem
def condition1 : Prop := Emma + Harry = George + Isla
def condition2 : Prop := Isla + Emma > George + Harry + 10
def condition3 : Prop := Harry > Emma + George + 12
def condition4 : Prop := 2 * Ann = George - 2
def Ann_value : Prop := Ann = 16

-- Final order from highest to lowest
def final_order (Harry George Emma Isla : ℕ) :=
  Harry > George ∧ George > Emma ∧ Emma > Isla

theorem problem_solution
  (Emma George Isla Harry Ann : ℕ)
  (h1 : condition1 Emma George Isla Harry)
  (h2 : condition2 Emma George Isla Harry)
  (h3 : condition3 Emma George Isla Harry)
  (h4 : condition4 Ann George)
  (ha : Ann_value Ann) :
  final_order Harry George Emma Isla :=
by {
  sorry
}

end problem_solution_l541_541168


namespace count_of_sequence_l541_541595

theorem count_of_sequence : 
  let a := 156
  let d := -6
  let final_term := 36
  (∃ n, a + (n - 1) * d = final_term) -> n = 21 := 
by
  sorry

end count_of_sequence_l541_541595


namespace pyramid_base_sides_and_k_values_l541_541036

-- Defining the conditions
variables (α β : ℝ)
variable (k : ℝ)
variable (n : ℕ)

-- Given conditions
axiom hyp1 : tan α = k * tan β
axiom hyp2 : k = 2 

-- Main theorem statement
theorem pyramid_base_sides_and_k_values (h : hyp1) (hk : hyp2) :
  (n = 3 ∧ k = 2) ∨ (∃ n, n ≥ 3 ∧ k = 1 / cos (Real.pi / n)) :=
by
  sorry

end pyramid_base_sides_and_k_values_l541_541036


namespace lord_moneybag_l541_541219

theorem lord_moneybag (n : ℕ) (hlow : 300 ≤ n) (hhigh : n ≤ 500)
           (h6 : 6 ∣ n) (h5 : 5 ∣ (n - 1)) (h4 : 4 ∣ (n - 2)) 
           (h3 : 3 ∣ (n - 3)) (h2 : 2 ∣ (n - 4)) (hprime : Nat.Prime (n - 5)) :
  n = 426 := by
  sorry

end lord_moneybag_l541_541219


namespace probability_jam_l541_541413

theorem probability_jam (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 ≤ x + y → x + y < 1.5 → 
  let P := ∫⁻ (p : ℝ × ℝ) in indicator (λ z, (1 ≤ (z.1 + z.2) ∧ z.1 + z.2 < 1.5)) (prod_measure (measure_theory.measure_space.volume) (measure_theory.measure_space.volume)) p in
  P = 3/8 := 
by
  sorry

end probability_jam_l541_541413


namespace combined_tax_rate_correct_l541_541419

def combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ) : ℝ :=
  ((Mork_tax_rate * Mork_income) + (Mindy_tax_rate * Mindy_income)) / (Mork_income + Mindy_income)

theorem combined_tax_rate_correct 
  (Mork_income : ℝ) 
  (Mork_tax_rate : ℝ := 0.40) 
  (Mindy_tax_rate : ℝ := 0.25) 
  (Mindy_income : ℝ := 4 * Mork_income) :
  combined_tax_rate Mork_income Mindy_income Mork_tax_rate Mindy_tax_rate = 0.28 :=
by 
  sorry

end combined_tax_rate_correct_l541_541419


namespace positive_integer_triples_l541_541459

theorem positive_integer_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b ∣ (a + 1) ∧ c ∣ (b + 1) ∧ a ∣ (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1 ∨
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 5 ∧ b = 3 ∧ c = 4) :=
by
  sorry

end positive_integer_triples_l541_541459


namespace min_red_cells_4x4_min_red_cells_nxn_l541_541799

theorem min_red_cells_4x4 : 
  ∀ (grid : matrix (fin 4) (fin 4) bool), 
    (∀ rows cols : finset (fin 4), rows.card = 2 ∧ cols.card = 2 → 
      ∃ i j, (i ∉ rows ∧ j ∉ cols) ∧ grid i j = tt) →
    ∃ S : finset (fin 4 × fin 4), S.card = 7 ∧ ∀ i j, (i, j) ∈ S → grid i j = tt :=
sorry

theorem min_red_cells_nxn (n : ℕ) (hn : 5 ≤ n) : 
  ∀ (grid : matrix (fin n) (fin n) bool), 
    (∀ rows cols : finset (fin n), rows.card = 2 ∧ cols.card = 2 →
      ∃ i j, (i ∉ rows ∧ j ∉ cols) ∧ grid i j = tt) → 
    ∃ S : finset (fin n × fin n), S.card = n + 1 ∧ ∀ i j, (i, j) ∈ S → grid i j = tt :=
sorry

end min_red_cells_4x4_min_red_cells_nxn_l541_541799


namespace find_positive_integer_n_l541_541460

noncomputable def is_largest_prime_divisor (p n : ℕ) : Prop :=
  (∃ k, n = p * k) ∧ ∀ q, Prime q ∧ q ∣ n → q ≤ p

noncomputable def is_least_prime_divisor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n ∧ ∀ q, Prime q ∧ q ∣ n → p ≤ q

theorem find_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ p, is_largest_prime_divisor p (n^2 + 3) ∧ is_least_prime_divisor p (n^4 + 6)) ∧
    ∀ m : ℕ, m > 0 ∧ 
      (∃ q, is_largest_prime_divisor q (m^2 + 3) ∧ is_least_prime_divisor q (m^4 + 6)) → m = 3 :=
by sorry

end find_positive_integer_n_l541_541460


namespace circle_tangent_m_value_l541_541556

theorem circle_tangent_m_value :
  ∃ m : ℝ, (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 5 - m) ∧
           (∀ x y : ℝ, (x + 2)^2 + (y + 2)^2 = 1) ∧
           (5 = real.sqrt (5 - m) + 1) ∧
           (m = -11) :=
by
  sorry

end circle_tangent_m_value_l541_541556


namespace factorization_of_polynomial_l541_541889

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l541_541889


namespace sequence_eventually_periodic_l541_541228

def f (k : ℕ) : ℕ :=
let ps := if k = 0 then [] else Nat.factors k
in ps.sum + 1

theorem sequence_eventually_periodic (k : ℕ) : ∃ m n, m < n ∧ ∀ i, f^[i + m] k = f^[i + n] k :=
sorry

end sequence_eventually_periodic_l541_541228


namespace smallest_sevens_in_sequence_l541_541818

noncomputable def smallest_term_sevens (n : ℕ) : ℕ := 13 * n - 10

theorem smallest_sevens_in_sequence : 
  ∃ n : ℕ, smallest_term_sevens n = 7777 := by
  existsi 598
  simp [smallest_term_sevens]
  norm_num
  sorry

end smallest_sevens_in_sequence_l541_541818


namespace students_study_both_l541_541646

theorem students_study_both (total_students fac_students numeric_methods auto_control : ℕ) 
  (h1 : total_students = 663)
  (h2 : fac_students = 0.80 * total_students)
  (h3 : numeric_methods = 240) 
  (h4 : auto_control = 423) 
  (h5 : fac_students = numeric_methods + auto_control - both_subjects) :
  both_subjects = 133 :=
by
  sorry

end students_study_both_l541_541646


namespace solve_wood_problem_l541_541638

variables (x y : ℝ)

def condition1 : Prop := y - x = 4.5
def condition2 : Prop := y / 2 = x - 1

theorem solve_wood_problem (h1 : condition1 x y) (h2 : condition2 x y) :
  ∃ x y : ℝ, (y - x = 4.5) ∧ (y / 2 = x - 1) :=
  sorry

end solve_wood_problem_l541_541638


namespace min_pieces_to_get_both_l541_541220

noncomputable def jerry_layout := 
-- The layout is a simple representation for the proof
{ fish_positions := [(1, 2), (5, 5), (7, 2)],    -- Positions of fish pieces (P)
  sausage_positions := [(2, 2), (5, 6)],         -- Positions of sausage pieces (K)
  both_positions := [(2, 6)]                     -- Position of piece with both fish and sausage
}

-- Define the conditions in Lean
def condition_1 (layout : Type) : Prop :=
  ∀ (x₀ y₀ : ℕ) (h₀ : 1 ≤ x₀ ∧ x₀ + 5 ≤ 8) (h₁ : 1 ≤ y₀ ∧ y₀ + 5 ≤ 8),
  (layout.fish_positions ∩ {(x, y) | x₀ ≤ x ∧ x ≤ x₀ + 5 ∧ y₀ ≤ y ∧ y ≤ y₀ + 5}).size ≥ 2

def condition_2 (layout : Type) : Prop :=
  ∀ (x₀ y₀ : ℕ) (h₀ : 1 ≤ x₀ ∧ x₀ + 2 ≤ 8) (h₁ : 1 ≤ y₀ ∧ y₀ + 2 ≤ 8),
  (layout.sausage_positions ∩ {(x, y) | x₀ ≤ x ∧ x ≤ x₀ + 2 ∧ y₀ ≤ y ∧ y ≤ y₀ + 2}).size ≤ 1

theorem min_pieces_to_get_both (layout : Type) 
  (h₁ : condition_1 layout) (h₂ : condition_2 layout) : 
  ∃ S : finset (ℕ × ℕ), S.card = 5 ∧ 
    ∃ (a b : ℕ), (a, b) ∈ layout.both_positions ∧ (a, b) ∈ S :=
sorry

end min_pieces_to_get_both_l541_541220


namespace XY_parallel_X_l541_541959

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541959


namespace fraction_of_number_l541_541326

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541326


namespace intersection_condition_rectangle_l541_541197

noncomputable def SimsonLine (S P Q R : Point) : Line := sorry
noncomputable def circumscribedCircle (P Q R : Point) : Circle := sorry
noncomputable def inscribedCircle (A B C D E F : Point) : Circle := sorry
noncomputable def l (X : Point) (A B C : Point) : Line := sorry

theorem intersection_condition_rectangle 
  (hexagon_is_inscribed : ∀ (A B C D E F : Point),
    ∃ O : Circle, -- There exists an inscribed circle O 
    O = inscribedCircle A B C D E F) :
  (∀ (A B C D E F : Point) 
      (h₁ : A ∈ circumscribedCircle B C F)
      (h₂ : B ∈ circumscribedCircle A C E)
      (h₃ : D ∈ circumscribedCircle A B F)
      (h₄ : E ∈ circumscribedCircle A B C),
    ∃ P : Point, 
    P ∈ l A B D F ∧ P ∈ l B A C E ∧ P ∈ l D A B F ∧ P ∈ l E A B C
    ↔ Quadrilateral_is_Rectangle C D E F) :=
sorry

end intersection_condition_rectangle_l541_541197


namespace relationship_between_xyz_l541_541582

theorem relationship_between_xyz (x y z : ℝ) (h1 : x - z < y) (h2 : x + z > y) : -z < x - y ∧ x - y < z :=
by
  sorry

end relationship_between_xyz_l541_541582


namespace eldorado_license_plates_count_l541_541176

theorem eldorado_license_plates_count:
  let letters := 26
  let digits := 10
  let total := (letters ^ 3) * (digits ^ 4)
  total = 175760000 :=
by
  sorry

end eldorado_license_plates_count_l541_541176


namespace vector_condition_l541_541132

open Real

def acute_angle (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2) > 0

def not_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 ≠ 0

theorem vector_condition (x : ℝ) :
  acute_angle (2, x + 1) (x + 2, 6) ∧ not_collinear (2, x + 1) (x + 2, 6) ↔ x > -5/4 ∧ x ≠ 2 :=
by
  sorry

end vector_condition_l541_541132


namespace emir_needs_more_money_l541_541888

noncomputable def dictionary_cost : ℝ := 5.50
noncomputable def dinosaur_book_cost : ℝ := 11.25
noncomputable def childrens_cookbook_cost : ℝ := 5.75
noncomputable def science_experiment_kit_cost : ℝ := 8.50
noncomputable def colored_pencils_cost : ℝ := 3.60
noncomputable def world_map_poster_cost : ℝ := 2.40
noncomputable def puzzle_book_cost : ℝ := 4.65
noncomputable def sketchpad_cost : ℝ := 6.20

noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def dinosaur_discount_rate : ℝ := 0.10
noncomputable def saved_amount : ℝ := 28.30

noncomputable def total_cost_before_tax : ℝ :=
  dictionary_cost +
  (dinosaur_book_cost - dinosaur_discount_rate * dinosaur_book_cost) +
  childrens_cookbook_cost +
  science_experiment_kit_cost +
  colored_pencils_cost +
  world_map_poster_cost +
  puzzle_book_cost +
  sketchpad_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + total_sales_tax

noncomputable def additional_amount_needed : ℝ := total_cost_after_tax - saved_amount

theorem emir_needs_more_money : additional_amount_needed = 21.81 := by
  sorry

end emir_needs_more_money_l541_541888


namespace hyperbola_eccentricity_l541_541984

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 5) (h₃ : c = Real.sqrt 21) (c' a' : ℝ) (h₄ : 2 * c' = 4) (h₅ : 2 * a' = 5 - Real.sqrt 21) :
  let e := c' / a' in e = 5 + Real.sqrt 21 :=
by {
  -- the proof is omitted, as per the requirements
  sorry
}

end hyperbola_eccentricity_l541_541984


namespace find_m_for_decreasing_function_l541_541118

theorem find_m_for_decreasing_function (m : ℝ) :
  (m^2 - m - 1 = 1) → f : ℝ → ℝ, f = λ x, (m^2 - m - 1) * x^(m^2 - 2m - 2) → 
  (∀ x : ℝ, x > 0 → f'(x) < 0) → m = 2 :=
begin
  intro h,
  intro f,
  intro h_decreasing,
  sorry
end

end find_m_for_decreasing_function_l541_541118


namespace Bogan_attempt_second_time_l541_541048

variable (m1 e1 e2 T : Nat)

theorem Bogan_attempt_second_time
  (h_initial_maggots : m1 = 10)
  (h_initial_eaten : e1 = 1)
  (h_total_maggots : T = 20) :
  T - m1 = 10 :=
by
  rw [h_initial_maggots, h_total_maggots]
  norm_num

end Bogan_attempt_second_time_l541_541048


namespace volume_removed_percentage_l541_541379

noncomputable def volume_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def volume_cube (s : ℝ) : ℝ :=
  s * s * s

noncomputable def percent_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem volume_removed_percentage :
  let l := 18
  let w := 12
  let h := 10
  let cube_side := 4
  let num_cubes := 8
  let original_volume := volume_rect_prism l w h
  let removed_volume := num_cubes * volume_cube cube_side
  percent_removed original_volume removed_volume = 23.7 := 
sorry

end volume_removed_percentage_l541_541379


namespace are_friendly_l541_541565

-- Define the friendly functions problem
def friendly_functions (y1 y2 : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → x < 1 → -1 < y1(x) - y2(x) ∧ y1(x) - y2(x) < 1

-- Define the two specific functions in question
def y1 (x : ℝ) : ℝ := x^2 - 1
def y2 (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem that these functions are friendly
theorem are_friendly : friendly_functions y1 y2 :=
by {
  sorry
}

end are_friendly_l541_541565


namespace johns_initial_money_l541_541415

theorem johns_initial_money (X : ℝ) 
  (h₁ : (1 / 2) * X + (1 / 3) * X + (1 / 10) * X + 10 = X) : X = 150 :=
sorry

end johns_initial_money_l541_541415


namespace equal_period_lengths_l541_541232

theorem equal_period_lengths :
  let A := 1000 / 2001 in
  let B := 1001 / 2001 in
  period_length A = period_length B :=
sorry

end equal_period_lengths_l541_541232


namespace function_expression_increasing_interval_range_y_le_0_l541_541126

def A : ℝ := 5
def ω : ℝ := 2
def φ : ℝ := -π / 6

def y (x : ℝ) : ℝ := A * sin (ω * x + φ)

def P : ℝ × ℝ := (π / 12, 0)
def Q : ℝ × ℝ := (π / 3, 5)

theorem function_expression : y = λ x, 5 * sin (2 * x - π / 6) := sorry

theorem increasing_interval (k : ℤ) : 
  ∃ I : set ℝ, 
  I = {x | k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3} := sorry

theorem range_y_le_0 (k : ℤ) : 
  ∃ I : set ℝ, 
  I = {x | k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12} := sorry

end function_expression_increasing_interval_range_y_le_0_l541_541126


namespace find_range_of_m_l541_541542

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end find_range_of_m_l541_541542


namespace largest_n_dividing_30_factorial_l541_541484

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541484


namespace area_of_circle_B_l541_541868

noncomputable theory
open Real

def Circle (r : ℝ) : Type := { center : ℝ × ℝ // ∥center∥ = r }

variables (A B : Circle 4)  -- Radius of circle A is 4
variables (r_B : ℝ)  -- Radius of circle B

-- Conditions
axiom cond1 : (sqrt (4^2 + (r_B / 2)^2)) = r_B  -- Offset by half the radius of B
axiom cond2 : ∀ (p : ℝ × ℝ), p ∈ A → p.dist (0, 0) = 0 ∨ (p.dist (0, 0) = r_B)

-- Theorem: To prove that the area of circle B is 36π square units.
theorem area_of_circle_B : π * r_B^2 = 36 * π :=
sorry

end area_of_circle_B_l541_541868


namespace XY_parallel_X_l541_541963

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541963


namespace clea_total_time_l541_541061

-- Definitions based on conditions given
def walking_time_on_stationary (x y : ℝ) (h1 : 80 * x = y) : ℝ :=
  80

def walking_time_on_moving (x y : ℝ) (k : ℝ) (h2 : 32 * (x + k) = y) : ℝ :=
  32

def escalator_speed (x k : ℝ) (h3 : k = 1.5 * x) : ℝ :=
  1.5 * x

-- The actual theorem based on the question
theorem clea_total_time 
  (x y k : ℝ)
  (h1 : 80 * x = y)
  (h2 : 32 * (x + k) = y)
  (h3 : k = 1.5 * x) :
  let t1 := y / (2 * x)
  let t2 := y / (3 * x)
  t1 + t2 = 200 / 3 :=
by
  sorry

end clea_total_time_l541_541061


namespace frequency_of_8th_group_l541_541452

theorem frequency_of_8th_group (size : ℕ) (freq1 freq2 freq3 freq4 freq5 freq6 freq7 : ℕ) (freq_5_to_7 : ℚ) (h_size : size = 64) (h_freq1 : freq1 = 5) (h_freq2 : freq2 = 7) (h_freq3 : freq3 = 11) (h_freq4 : freq4 = 13) (h_freq_5_to_7 : freq_5_to_7 = 0.125) :
  freq5 = (freq_5_to_7 * size).to_nat → freq6 = (freq_5_to_7 * size).to_nat → freq7 = (freq_5_to_7 * size).to_nat → (freq1 + freq2 + freq3 + freq4 + freq5 + freq6 + freq7) + x = size → x = 4 :=
by
  sorry

end frequency_of_8th_group_l541_541452


namespace polynomial_roots_correct_l541_541467

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l541_541467


namespace polynomial_rational_condition_l541_541458

open Polynomial

theorem polynomial_rational_condition (W : ℝ[X]) :
  (∀ x y : ℝ, (x + y).is_rational → (W.eval x + W.eval y).is_rational) ↔
  (∃ a b : ℚ, ∀ x : ℝ, W.eval x = (a : ℝ) * x + b) :=
by
  sorry

end polynomial_rational_condition_l541_541458


namespace soap_box_width_l541_541819

theorem soap_box_width
  (carton_length : ℝ) (carton_width : ℝ) (carton_height : ℝ)
  (box_length : ℝ) (box_height : ℝ) (max_boxes : ℝ) (carton_volume : ℝ)
  (box_volume : ℝ) (W : ℝ) : 
  carton_length = 25 →
  carton_width = 42 →
  carton_height = 60 →
  box_length = 6 →
  box_height = 6 →
  max_boxes = 250 →
  carton_volume = carton_length * carton_width * carton_height →
  box_volume = box_length * W * box_height →
  max_boxes * box_volume = carton_volume →
  W = 7 :=
sorry

end soap_box_width_l541_541819


namespace cost_split_difference_l541_541236

-- Definitions of amounts paid
def SarahPaid : ℕ := 150
def DerekPaid : ℕ := 210
def RitaPaid : ℕ := 240

-- Total paid by all three
def TotalPaid : ℕ := SarahPaid + DerekPaid + RitaPaid

-- Each should have paid:
def EachShouldHavePaid : ℕ := TotalPaid / 3

-- Amount Sarah owes Rita
def SarahOwesRita : ℕ := EachShouldHavePaid - SarahPaid

-- Amount Derek should receive back from Rita
def DerekShouldReceiveFromRita : ℕ := DerekPaid - EachShouldHavePaid

-- Difference between the amounts Sarah and Derek owe/should receive from Rita
theorem cost_split_difference : SarahOwesRita - DerekShouldReceiveFromRita = 60 := by
    sorry

end cost_split_difference_l541_541236


namespace largest_n_dividing_factorial_l541_541517

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541517


namespace boxes_sold_l541_541535

def case_size : ℕ := 12
def remaining_boxes : ℕ := 7

theorem boxes_sold (sold_boxes : ℕ) : ∃ n : ℕ, sold_boxes = n * case_size + remaining_boxes :=
sorry

end boxes_sold_l541_541535


namespace no_unit_square_in_parallelogram_l541_541263

theorem no_unit_square_in_parallelogram {P : Type} [parallelogram P] 
  (h1 : ∀ (height : ℝ), height > 1) :
  ¬ ∃ (unit_square : square) (parallelogram_inscription : P), 
    inscribe unit_square parallelogram_inscription :=
sorry

end no_unit_square_in_parallelogram_l541_541263


namespace fraction_of_number_l541_541314

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541314


namespace infinite_sequence_of_moves_l541_541703

structure Car :=
  (x : ℕ)   -- x-coordinate of the car
  (y : ℕ)   -- y-coordinate of the car
  (dir : ℕ) -- direction (0 for up, 1 for right, 2 for down, 3 for left)

structure Grid :=
  (cars : list Car)

-- Valid direction; can be either 0, 1, 2, or 3
def valid_direction (d : ℕ) : Prop := d < 4

-- Check if no two cars occupy the same cell
def no_two_cars_occupy_same_cell (cars : list Car) : Prop :=
  ∀ (c1 c2 : Car), c1 ∈ cars → c2 ∈ cars → c1 ≠ c2 → (c1.x ≠ c2.x ∨ c1.y ≠ c2.y)

-- Check if the cell in front of each car is empty
def cell_in_front_empty (cars : list Car) : Prop :=
  ∀ (car : Car), car ∈ cars → 
  match car.dir with
  | 0 => (car.x, car.y + 1) ∉ (cars.map (λ c, (c.x, c.y))) -- facing up
  | 1 => (car.x + 1, car.y) ∉ (cars.map (λ c, (c.x, c.y))) -- facing right
  | 2 => (car.x, car.y - 1) ∉ (cars.map (λ c, (c.x, c.y))) -- facing down
  | 3 => (car.x - 1, car.y) ∉ (cars.map (λ c, (c.x, c.y))) -- facing left
  | _ => false

-- Cars facing conditions (no two cars face each other directly across any row or column)
def no_direct_facing (cars : list Car) : Prop :=
  ∀ (c1 c2 : Car), c1 ∈ cars → c2 ∈ cars → (c1.x = c2.x ∧ (c1.dir = 0 ∧ c2.dir = 2 ∨ c1.dir = 2 ∧ c2.dir = 0)) ∨ (c1.y = c2.y ∧ (c1.dir = 1 ∧ c2.dir = 3 ∨ c1.dir = 3 ∧ c2.dir = 1)) → false

-- Prove existence of an infinite sequence of valid moves using each car infinitely many times
theorem infinite_sequence_of_moves (cars : list Car) (h1 : no_two_cars_occupy_same_cell cars) 
  (h2 : cell_in_front_empty cars) (h3 : no_direct_facing cars) :
  ∃ (seq : ℕ → Car), (∀ n, seq n ∈ cars) ∧ (∀ car : Car, car ∈ cars → ∃ inf : ℕ → ℕ, (∀ n, car = seq (inf n))) :=
sorry

end infinite_sequence_of_moves_l541_541703


namespace find_m_value_l541_541628

noncomputable def M (m : ℝ) : ℝ × ℝ × ℝ := (m, -2, 1)
noncomputable def N (m : ℝ) : ℝ × ℝ × ℝ := (0, m, 3)
noncomputable def n : ℝ × ℝ × ℝ := (3, 1, 2)

theorem find_m_value (m : ℝ) (M_plane : (m, -2, 1))
                    (N_plane : (0, m, 3))
                    (normal_vector : (3, 1, 2)) :
    m = 3 :=
by
  -- Skipping the proof part here.
  sorry

end find_m_value_l541_541628


namespace calculate_expression_l541_541059

variable (x y : ℚ)

theorem calculate_expression (h₁ : x = 4 / 6) (h₂ : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  -- proof steps here
  sorry

end calculate_expression_l541_541059


namespace Adam_teaches_students_l541_541837

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end Adam_teaches_students_l541_541837


namespace parabola_intersection_probability_correct_l541_541295

noncomputable def parabola_intersection_probability : ℚ :=
  let die := {1, 2, 3, 4, 5, 6, 7, 8}
  let parabolas_intersect_at_x_axis (a b c d : ℤ) : Prop :=
    let discriminant := (a - c)^2 + 4 * (b - d)
    discriminant ≥ 0
  let total_outcomes := (8 * 8 * 8 * 8 : ℚ)  -- Total possible combinations
  let favorable_outcomes : ℚ := 63 / 64 * total_outcomes  -- The probability is given directly
  favorable_outcomes / total_outcomes

theorem parabola_intersection_probability_correct (a b c d : ℤ) (h_a : a ∈ {1, 2, 3, 4, 5, 6, 7, 8})
    (h_b : b ∈ {1, 2, 3, 4, 5, 6, 7, 8}) (h_c : c ∈ {1, 2, 3, 4, 5, 6, 7, 8}) (h_d : d ∈ {1, 2, 3, 4, 5, 6, 7, 8}) :
  parabola_intersection_probability = 63 / 64 := by 
  sorry


end parabola_intersection_probability_correct_l541_541295


namespace necessary_condition_for_congruence_l541_541870

-- Given two triangles ABC and DEF
variables {ABC DEF : Type} [fintype ABC] [fintype DEF]

-- Condition A: The sums of the areas of two triangles and two corresponding sides are equal.
def condition_A (A B C D E F : ℕ): Prop :=
  let S_ABC := (1 / 2) * A * B * (sin C) in
  let S_DEF := (1 / 2) * D * E * (sin F) in
  S_ABC = S_DEF ∧ A * B = D * E ∧ sin C = sin F 

-- Condition B: The two triangles are congruent
def condition_B (A B C D E F : ℕ): Prop :=
  A = D ∧ B = E ∧ C = F

-- The problem statement: A is a necessary condition for B.
theorem necessary_condition_for_congruence (A B C D E F : ℕ) (hB : condition_B A B C D E F): condition_A A B C D E F :=
  sorry -- proof to be filled in later

end necessary_condition_for_congruence_l541_541870


namespace incorrect_conclusion_a_l541_541636

noncomputable def y1 (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1
noncomputable def y2 (x b : ℝ) : ℝ := x ^ 2 + b * x + 2
noncomputable def y3 (x c : ℝ) : ℝ := x ^ 2 + c * x + 3

theorem incorrect_conclusion_a (b c : ℝ) (hb_pos : 0 < b) (hc_pos : 0 < c) (hbc : b^2 = 2*c) :
  (∃ x : ℝ, y1 x = 0) ∧ (∃ x : ℝ, y2 x b = 0) →
  ¬(∃ x : ℝ, y3 x c = 0 ∧ ∃ x' : ℝ, y3 x' c = 0) →
  false := 
begin
  sorry
end

end incorrect_conclusion_a_l541_541636


namespace mary_garbage_bill_l541_541688

theorem mary_garbage_bill :
  let weekly_cost := 2 * 10 + 1 * 5,
      monthly_cost := weekly_cost * 4,
      discount := 0.18 * monthly_cost,
      discounted_monthly_cost := monthly_cost - discount,
      fine := 20,
      total_bill := discounted_monthly_cost + fine
  in total_bill = 102 :=
by
  let weekly_cost := 2 * 10 + 1 * 5
  let monthly_cost := weekly_cost * 4
  let discount := 0.18 * monthly_cost
  let discounted_monthly_cost := monthly_cost - discount
  let fine := 20
  let total_bill := discounted_monthly_cost + fine
  show total_bill = 102 from sorry

end mary_garbage_bill_l541_541688


namespace XY_parallel_X_l541_541929

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541929


namespace relationship_abc_l541_541680

noncomputable def a : ℝ := 3^0.7
noncomputable def b : ℝ := (1/3)^(-0.8)
noncomputable def c : ℝ := Real.logBase 0.7 0.8

theorem relationship_abc : c < a ∧ a < b := sorry

end relationship_abc_l541_541680


namespace find_a5_l541_541986

open Nat

def increasing_seq (a : Nat → Nat) : Prop :=
  ∀ m n : Nat, m < n → a m < a n

theorem find_a5
  (a : Nat → Nat)
  (h1 : ∀ n : Nat, a (a n) = 3 * n)
  (h2 : increasing_seq a)
  (h3 : ∀ n : Nat, a n > 0) :
  a 5 = 8 :=
by
  sorry

end find_a5_l541_541986


namespace triangle_parallel_bisectors_l541_541954

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541954


namespace smallest_divisors_l541_541443

variable (a b c : ℕ)

def is_natural_number (x : ℚ) : Prop := ∃ (n : ℕ), x = n

def satisfies_conditions (a b c : ℕ) (k : ℕ) : Prop :=
  k < a ∧ k < b ∧ is_natural_number (k : ℚ) ∧ k = (a*b + c^2)/(a + b)

theorem smallest_divisors (a b c : ℕ) (h: ∃ k : ℕ, satisfies_conditions a b c k) : 
  ∃ (n : ℕ), (divisors (a + b)).length = 3 := 
sorry

end smallest_divisors_l541_541443


namespace find_angle_EBC_l541_541548

-- Define a kite and its properties
structure Kite (A B C D : Type) extends Quad (A B C D) :=
(ab_eq_ad : A.dist B = A.dist D)
(cb_eq_cd : C.dist B = C.dist D)
(ac_bisects_BAD : ∠ A B C = 47° ∧ ∠ A D C = 47°)

-- Define the angle BAD and its bisector AC
def angle_BAD_bisected (A B C D E : Type) [Kite A B C D] :=
∠ A B A D = 94° ∧ ∠ B A C = 47°

-- State the theorem we want to prove
theorem find_angle_EBC (A B C D E : Type) [K : Kite A B C D]
  (h1 : ∠ A B A D = 94°)
  (h2 : ∠ B A C = 47°)
  (h3 : side_ext_AB_to_E : E on line (A,B)) :
  ∠ E B C = 94° := sorry

end find_angle_EBC_l541_541548


namespace pig_catch_distance_l541_541707

-- Define the field and the initial positions
def field_width : ℝ := 100
def field_height : ℝ := 100
def initial_distance : ℝ := (field_width ^ 2 + field_height ^ 2) ^ (1/2) -- 100*sqrt(2)

-- Define the speeds
def pig_speed : ℝ := 1 -- Assume pig's speed is 1 unit for simplicity
def pet_speed : ℝ := 2 * pig_speed -- Pet's speed is twice the pig's speed

-- Define the ratio of speeds
def speed_ratio : ℝ := pet_speed / pig_speed -- which is 2

-- Calculate the distance Pet runs to catch the pig
def pet_pursuit_distance (a : ℝ) (n : ℝ) : ℝ := (a * n^2) / (n^2 - 1)

-- Calculate the distance the pig runs
def pig_run_distance (pet_distance : ℝ) : ℝ := pet_distance / 2

theorem pig_catch_distance : pig_run_distance (pet_pursuit_distance initial_distance speed_ratio) = 66 + 2/3 := by
  sorry

end pig_catch_distance_l541_541707


namespace H_functions_l541_541546

-- Define what it means to be an H function
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1

-- Define each function
def f1 (x : ℝ) := Real.exp x + x
def f2 (x : ℝ) := x ^ 2
def f3 (x : ℝ) := 3 * x - Real.sin x
def f4 (x : ℝ) :=
  if x = 0 then 0 else Real.log (Abs.abs x)

-- Theorem stating that only f1 and f3 are H functions
theorem H_functions : 
  (is_H_function f1 ∧ is_H_function f3) ∧ 
  ¬ is_H_function f2 ∧ 
  ¬ is_H_function f4 := 
  by sorry

end H_functions_l541_541546


namespace pizzas_served_during_lunch_l541_541024

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem pizzas_served_during_lunch :
  ∃ lunch_pizzas : ℕ, lunch_pizzas = total_pizzas - dinner_pizzas :=
by
  use 9
  exact rfl

end pizzas_served_during_lunch_l541_541024


namespace fraction_of_number_l541_541313

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541313


namespace greatest_number_of_sundays_in_first_58_days_l541_541780

theorem greatest_number_of_sundays_in_first_58_days (start_on_monday : ℕ → Day) :
    (∃ n : ℕ, n = 8 ∧ ∀ m : ℕ, (if start_on_monday m = Day.sunday then 1 else 0 + 
   if start_on_monday (m+1) = Day.sunday then 1 else 0 + 
   if start_on_monday (m+2) = Day.sunday then 1 else 0 + 
   if start_on_monday (m+3) = Day.sunday then 1 else 0 + 
   if start_on_monday (m+4) = Day.sunday then 1 else 0 + 
   if start_on_monday (m+5) = Day.sunday then 1 else 0 + 
   if start_on_monday (m+6) = Day.sunday then 1 else 0) ≤ n) := sorry

end greatest_number_of_sundays_in_first_58_days_l541_541780


namespace chef_cherries_l541_541006

theorem chef_cherries :
  ∀ (total_cherries used_cherries remaining_cherries : ℕ),
    total_cherries = 77 →
    used_cherries = 60 →
    remaining_cherries = total_cherries - used_cherries →
    remaining_cherries = 17 :=
by
  sorry

end chef_cherries_l541_541006


namespace increasing_sequence_and_limit_l541_541667

variables {a b : ℝ} {f g : ℝ → ℝ} (n : ℕ) [decidable_lt ℝ] 

noncomputable def I_n (a b : ℝ) (f g : ℝ → ℝ) (n : ℕ) : ℝ :=
  ∫ x in set.Icc a b, (f x) ^ (n + 1) / (g x) ^ n

theorem increasing_sequence_and_limit 
  (a b : ℝ) (f g : ℝ → ℝ)
  (h_ab : a < b)
  (h_f_cont : continuous_on f (set.Icc a b))
  (h_g_cont : continuous_on g (set.Icc a b))
  (h_f_pos : ∀ x ∈ set.Icc a b, 0 < f x)
  (h_g_pos : ∀ x ∈ set.Icc a b, 0 < g x)
  (h_int_eq : ∫ x in set.Icc a b, f x = ∫ x in set.Icc a b, g x)
  (h_f_ne_g : ∃ x ∈ set.Icc a b, f x ≠ g x) :
  (∀ n : ℕ, I_n a b f g (n + 1) > I_n a b f g n) ∧
  tendsto (λ n, I_n a b f g n) at_top at_top :=
begin
  sorry
end

end increasing_sequence_and_limit_l541_541667


namespace piecewise_function_value_l541_541207

def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp(x - 1)
else Real.log x^2 - 1 / Real.log 3

theorem piecewise_function_value : f (f 2) = 2 := by
  sorry

end piecewise_function_value_l541_541207


namespace time_downstream_l541_541807

variable (V_b : ℝ) (V_s : ℝ) (T_up : ℝ)

-- Conditions provided in the problem
def speed_of_boat : ℝ := V_b
def speed_of_stream : ℝ := V_s
def time_upstream : ℝ := T_up

-- Additional assumptions based on the problem
axiom speed_boat_value : V_b = 15
axiom speed_stream_value : V_s = 3
axiom time_upstream_value : T_up = 1.5

-- Derived quantities
def upstream_speed : ℝ := speed_of_boat - speed_of_stream
def downstream_speed : ℝ := speed_of_boat + speed_of_stream
def distance_upstream : ℝ := upstream_speed * time_upstream

-- Question to be proved
theorem time_downstream : (distance_upstream / downstream_speed) = 1 :=
by
  rw [distance_upstream, upstream_speed, downstream_speed]
  repeat {rw [speed_boat_value, speed_stream_value, time_upstream_value]}
  norm_num
  sorry

end time_downstream_l541_541807


namespace number_of_real_solutions_l541_541088

noncomputable def complex_magnitude : ℝ → ℝ :=
  λ x, Real.sqrt (1 + x^2)

theorem number_of_real_solutions :
  {x : ℝ // complex_magnitude x = 2}.card = 2 :=
by
  sorry

end number_of_real_solutions_l541_541088


namespace min_value_one_div_a_plus_one_div_b_l541_541098

theorem min_value_one_div_a_plus_one_div_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_one_div_a_plus_one_div_b_l541_541098


namespace volume_of_solid_rotated_around_length_volume_of_solid_rotated_around_width_l541_541824

theorem volume_of_solid_rotated_around_length :
  let length := 6
  let width := 4
  volume (solid_of_rotation length width) = 96 * π :=
sorry

theorem volume_of_solid_rotated_around_width :
  let length := 6
  let width := 4
  volume (solid_of_rotation width length) = 144 * π :=
sorry

end volume_of_solid_rotated_around_length_volume_of_solid_rotated_around_width_l541_541824


namespace monotonic_intervals_and_extreme_values_l541_541905

open Real

def f (x : ℝ) : ℝ := 2 * log x * x^2

theorem monotonic_intervals_and_extreme_values :
  (∀ x ∈ Ioo (0 : ℝ) (exp (- 1 / 2 : ℝ)), deriv f x < 0) ∧
  (∀ x ∈ Ioo (exp (- 1 / 2 : ℝ)) (⊤ : ℝ), deriv f x > 0) ∧
  f (exp (- 1 / 2)) = - (1 / exp 1) :=
by
  sorry

end monotonic_intervals_and_extreme_values_l541_541905


namespace XY_parallel_X_l541_541938

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541938


namespace largest_possible_number_of_neither_l541_541849

theorem largest_possible_number_of_neither
  (writers : ℕ)
  (editors : ℕ)
  (attendees : ℕ)
  (x : ℕ)
  (N : ℕ)
  (h_writers : writers = 45)
  (h_editors_gt : editors > 38)
  (h_attendees : attendees = 90)
  (h_both : N = 2 * x)
  (h_equation : writers + editors - x + N = attendees) :
  N = 12 :=
by
  sorry

end largest_possible_number_of_neither_l541_541849


namespace correct_calculation_l541_541340

variable (a : ℝ)

-- Conditions
def option_A : Prop := a^2 * a^3 = a^5
def option_B : Prop := (a^2)^3 = a^5
def option_C : Prop := (a * b)^3 = a * b^3
def option_D : Prop := a^2 / a^3 = a

-- Problem statement
theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l541_541340


namespace minimize_sum_radii_l541_541358

theorem minimize_sum_radii (A B C : Point) (M H : Point) (hH : foot_of_altitude B A C H)
  (hM : M ∈ line_segment A C) :
  min_radii_sum A B C M H :=
sorry

end minimize_sum_radii_l541_541358


namespace min_odd_integers_l541_541767

theorem min_odd_integers (a b c d e f : ℤ) 
    (h1 : a + b + c = 30) 
    (h2 : a + b + c + d + e = 48) 
    (h3 : a + b + c + d + e + f = 59) : 
    ∃ n : ℕ, n = 1 ∧ (n = (n.bodd.countp (λ x, x % 2 ≠ 0))) :=
  sorry

end min_odd_integers_l541_541767


namespace expression_evaluation_l541_541441

theorem expression_evaluation : (75 * 2024 - 25 * 2024) / 2 = 50600 := by
  calc
    (75 * 2024 - 25 * 2024) / 2
      = (2024 * 50) / 2   : by sorry -- fill this part if necessary
      ... = 50600         : by sorry

end expression_evaluation_l541_541441


namespace smallest_S_is_S8_l541_541203

variables {a_n : ℕ → ℝ} -- Define the arithmetic sequence
def S (n : ℕ) : ℝ := (n * (a_n 1 + a_n n)) / 2 -- Sum of the first n terms of the arithmetic sequence

variables {S_16 S_17 : ℝ} -- Specific values for S_16 and S_17

-- Given conditions
axiom hS16 : S 16 < 0
axiom hS17 : S 17 > 0

-- We need to prove
theorem smallest_S_is_S8 : (∃ n, S n = S 8) :=
by 
  sorry

end smallest_S_is_S8_l541_541203


namespace fraction_of_number_l541_541316

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541316


namespace relationship_even_increasing_l541_541564

-- Even function definition
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Monotonically increasing function definition on interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

variable {f : ℝ → ℝ}

-- The proof problem statement
theorem relationship_even_increasing (h_even : even_function f) (h_increasing : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
by
  sorry

end relationship_even_increasing_l541_541564


namespace card_dealing_probability_l541_541288

noncomputable def probability_ace_then_ten_then_jack : ℚ :=
  let prob_ace := 4 / 52
  let prob_ten := 4 / 51
  let prob_jack := 4 / 50
  prob_ace * prob_ten * prob_jack

theorem card_dealing_probability :
  probability_ace_then_ten_then_jack = 16 / 33150 := by
  sorry

end card_dealing_probability_l541_541288


namespace rollo_guinea_pigs_food_l541_541718

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l541_541718


namespace num_intersections_positive_x_l541_541744

/-
Problem:
  Prove that the number of points in the plane with positive $x$-coordinates that lie on two or more of the graphs
  $y = \log_2 x$, $y = e^x$, $y = x^2$, and $y = 2^x$ is 2.
-/
theorem num_intersections_positive_x : 
  let log2 := λ x : ℝ, Real.log x / Real.log 2,
      exp := λ x : ℝ, Real.exp x,
      sq := λ x : ℝ, x^2,
      exp2 := λ x : ℝ, (2 : ℝ) ^ x in
  2 = (set.univ : set ℝ).countp (λ x, 0 < x ∧ 
  ((log2 x = exp x) ∨ (log2 x = sq x) ∨ (log2 x = exp2 x) ∨
   (exp x = sq x) ∨ (exp x = exp2 x) ∨ (sq x = exp2 x))) sorry

end num_intersections_positive_x_l541_541744


namespace solve_system_l541_541244

theorem solve_system:
  ∃ (x y : ℝ), (26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧ 10 * x^2 + 18 * x * y + 8 * y^2 = 6) ↔
  (x = -1 ∧ y = 2) ∨ (x = -11 ∧ y = 14) ∨ (x = 11 ∧ y = -14) ∨ (x = 1 ∧ y = -2) :=
by
  sorry

end solve_system_l541_541244


namespace line_intersects_x_axis_at_l541_541417

theorem line_intersects_x_axis_at
  (h : ∀ x y : ℝ, 2*y - 3*x = 15 ∧ y = 0 → x = -5 ∧ y = 0) :
  2*0 - 3*(-5) = 15 ∧ 0 = 0 :=
by
  have : 2*(0 : ℝ) - 3*(-5 : ℝ) = 15, by linarith,
  exact ⟨this, rfl⟩

end line_intersects_x_axis_at_l541_541417


namespace largest_n_dividing_30_factorial_l541_541499

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541499


namespace A_time_240m_race_l541_541794

theorem A_time_240m_race (t : ℕ) :
  (∀ t, (240 / t) = (184 / t) * (t + 7) ∧ 240 = 184 + ((184 * 7) / t)) → t = 23 :=
by
  sorry

end A_time_240m_race_l541_541794


namespace tan_sin_30_computation_l541_541437

theorem tan_sin_30_computation :
  let θ := 30 * Real.pi / 180 in
  Real.tan θ + 4 * Real.sin θ = (Real.sqrt 3 + 6) / 3 :=
by
  let θ := 30 * Real.pi / 180
  have sin_30 : Real.sin θ = 1 / 2 := by sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by sorry
  have tan_30 : Real.tan θ = Real.sin θ / Real.cos θ := by sorry
  have sin_60 : Real.sin (2 * θ) = Real.sqrt 3 / 2 := by sorry
  sorry

end tan_sin_30_computation_l541_541437


namespace orthocenter_bisector_l541_541644

noncomputable theory

open EuclideanGeometry

variables {A B C D E F G H I : Point}
variables {O P : Circle}

-- Definitions based on given conditions
def circumcircle_of_ABC := Circle.circumcircle O A B C
def angle_bisector_AD := Segment.angleBisector A D B C
def orthocenter_of_ABC := Triangle.orthocenter H A B C
def perpendicular_CE := Line.perpendicular_at CE A B E
def perpendicular_BE := Line.perpendicular_at BE A C F
def circumcircle_of_AFE := Circle.circumcircle P A F E
def intersection_of_circles := Circle.intersection G O P
def line_GD_intersects_BC := Line.intersects GD B C I

-- Hypotheses based on given conditions
variables (h1 : circumcircle_of_ABC)
variables (h2 : angle_bisector_AD)
variables (h3 : orthocenter_of_ABC)
variables (h4 : perpendicular_CE)
variables (h5 : perpendicular_BE)
variables (h6 : circumcircle_of_AFE)
variables (h7 : intersection_of_circles)
variables (h8 : line_GD_intersects_BC)

theorem orthocenter_bisector:
  ∃ IH, Segment.angleBisector IH H B H C :=
sorry

end orthocenter_bisector_l541_541644


namespace arithmetic_geometric_sequence_solution_l541_541973

theorem arithmetic_geometric_sequence_solution (u v : ℕ → ℝ) (a b u₀ : ℝ) :
  (∀ n, u (n + 1) = a * u n + b) ∧ (∀ n, v (n + 1) = a * v n + b) →
  u 0 = u₀ →
  v 0 = b / (1 - a) →
  ∀ n, u n = a ^ n * (u₀ - b / (1 - a)) + b / (1 - a) :=
by
  intros
  sorry

end arithmetic_geometric_sequence_solution_l541_541973


namespace quadratic_decreasing_interval_l541_541578

theorem quadratic_decreasing_interval {k : ℝ} : 
  (∀ x y : ℝ, -5 ≤ x ∧ x ≤ -1 ∧ -5 ≤ y ∧ y ≤ -1 ∧ x < y → f x ≥ f y) → k ≤ 2 :=
by {
  let f : ℝ → ℝ := λ x, 2 * x ^ 2 + 2 * k * x - 8,
  sorry
}

end quadratic_decreasing_interval_l541_541578


namespace sqrt_neg2023_squared_l541_541864

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end sqrt_neg2023_squared_l541_541864


namespace find_x_l541_541730

theorem find_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) : 
  x = (36 * Real.sqrt 5)^(4/11) := 
sorry

end find_x_l541_541730


namespace mrs_petersons_change_l541_541697

noncomputable def change_in_euros (n : ℕ) (c d r e : ℝ) :=
  let total_cost_before_discount := n * c
  let discount_amount := total_cost_before_discount * d
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let amount_paid_in_usd := e * r
  let change_in_usd := amount_paid_in_usd - total_cost_after_discount
  change_in_usd / r

theorem mrs_petersons_change :
  change_in_euros 10 45 0.10 1.10 500 ≈ 131.82 :=
by
  sorry

end mrs_petersons_change_l541_541697


namespace difference_of_squares_expression_l541_541343

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l541_541343


namespace mean_and_variance_of_y_l541_541570

variables {α : Type*} {n : ℕ} 
variables (x : fin n → ℝ) (y : fin n → ℝ)
variable (h_n : n = 20)
variable (h_mean_x : (∑ i, x i) / n = 1)
variable (h_var_x : (∑ i, (x i - 1)^2) / n = 8)
variable (h_y_def : ∀ i, y i = 2 * x i + 3)

theorem mean_and_variance_of_y :
  (∑ i, y i) / n = 5 ∧ (∑ i, (y i - 5) ^ 2) / n = 32 :=
by
  sorry

end mean_and_variance_of_y_l541_541570


namespace integral_result_l541_541861

noncomputable def integral_eval : Real :=
  ∫ x in Real.pi / 4..Real.arctan 3, 1 / ((3 * Real.tan x + 5) * Real.sin (2 * x))

theorem integral_result : integral_eval = (1 / 10 : ℝ) * Real.log (12 / 7) := by
  sorry

end integral_result_l541_541861


namespace arrange_numbers_l541_541408

noncomputable def a := 6^0.7
noncomputable def b := 0.7^6
noncomputable def c := Real.log 6 / Real.log 0.7

theorem arrange_numbers :
  (a > 1) ∧ (0 < b ∧ b < 1) ∧ (c < 1) → (c < b ∧ b < a) :=
by
  -- In order to use these definitions in proofs, we add the corresponding statements solely based on the conditions
  intro h,
  cases h with ha hb,
  cases hb with hb1 hb,
  cases hb with hb2 hc,
  -- This is the statement that we will ultimately need to prove
  sorry

end arrange_numbers_l541_541408


namespace total_cost_for_39_roses_l541_541040

/-
Problem:
Prove that the cost of a bouquet with 39 roses is $58.75, given the following conditions:
1. There is a base cost of $10 for any bouquet.
2. An additional cost that is proportional to the number of roses.
3. A bouquet with 12 roses costs $25.
-/

def additional_cost_proportional (n : ℕ) (k : ℕ) (base_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (p : ℕ), total_cost = base_cost + p * n ∧ p * 12 = 15 * 12

theorem total_cost_for_39_roses : 
  additional_cost_proportional 39 12 10 25 →
  ∃ (c : ℕ), c = 58.75 := 
by
  intros
  sorry

end total_cost_for_39_roses_l541_541040


namespace exponential_derivative_inequality_l541_541206

variable {f : ℝ → ℝ} (x : ℝ) (x₁ x₂ : ℝ)

theorem exponential_derivative_inequality 
  (hf' : ∀ x, f'(x) + f(x) > 0) 
  (hx : x₁ < x₂) :
  e^(x₁) * f(x₁) < e^(x₂) * f(x₂) := 
sorry

end exponential_derivative_inequality_l541_541206


namespace largest_square_area_l541_541403

-- Given conditions and necessary definitions
def is_right_angle (X Y Z : Type) [InnerProductSpace ℝ X] : Prop := ∃ (c1 c2 : ℝ), X = Y + c1 * (Z - Y) ∧ (c2 = 0)

variables (XY YZ XZ : ℝ)

-- Given conditions for the problem
def angle_XYZ_right : Prop := is_right_angle XYZ
def quadrilateral_areas_sum_eq_450 : Prop := (XY * XY) + (YZ * YZ) + (XZ * XZ) = 450

-- The statement we want to prove
def area_largest_square : Prop := (XZ * XZ) = 225

-- The theorem statement
theorem largest_square_area (h1 : angle_XYZ_right) (h2 : quadrilateral_areas_sum_eq_450) : area_largest_square := 
by
  sorry

end largest_square_area_l541_541403


namespace find_angle_between_vectors_l541_541133

-- Define the problem with the given conditions
variables {a b : EuclideanSpace ℝ (Fin 3)}
condition1 : ‖a‖ = 1
condition2 : ‖b‖ = 1
condition3 : ‖a + b‖ + 2 * inner a b = 0

-- Define the angle function
def angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ := real.arccos (inner u v)

-- State the theorem
theorem find_angle_between_vectors
    (ha : ‖a‖ = 1)
    (hb : ‖b‖ = 1)
    (hab : ‖a + b‖ + 2 * inner a b = 0) :
  angle_between a b = 2 * real.pi / 3 := 
sorry

end find_angle_between_vectors_l541_541133


namespace fish_count_when_james_discovers_l541_541616

def fish_in_aquarium (initial_fish : ℕ) (bobbit_worm_eats : ℕ) (predatory_fish_eats : ℕ)
  (reproduction_rate : ℕ × ℕ) (days_1 : ℕ) (added_fish: ℕ) (days_2 : ℕ) : ℕ :=
  let predation_rate := bobbit_worm_eats + predatory_fish_eats
  let total_eaten_in_14_days := predation_rate * days_1
  let reproduction_events_in_14_days := days_1 / reproduction_rate.snd
  let fish_born_in_14_days := reproduction_events_in_14_days * reproduction_rate.fst
  let fish_after_14_days := initial_fish - total_eaten_in_14_days + fish_born_in_14_days
  let fish_after_14_days_non_negative := max fish_after_14_days 0
  let fish_after_addition := fish_after_14_days_non_negative + added_fish
  let total_eaten_in_7_days := predation_rate * days_2
  let reproduction_events_in_7_days := days_2 / reproduction_rate.snd
  let fish_born_in_7_days := reproduction_events_in_7_days * reproduction_rate.fst
  let fish_after_7_days := fish_after_addition - total_eaten_in_7_days + fish_born_in_7_days
  max fish_after_7_days 0

theorem fish_count_when_james_discovers :
  fish_in_aquarium 60 2 3 (2, 3) 14 8 7 = 4 :=
sorry

end fish_count_when_james_discovers_l541_541616


namespace largest_x_for_perfect_square_l541_541881

-- Define the condition for the expression 4^27 + 4^1010 + 4^x being a perfect square
def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

-- Main theorem to prove
theorem largest_x_for_perfect_square :
  ∃ x : ℤ, (4^27 + 4^1010 + 4^x) = (2^27 + 2^x)^2 ∧ (∀ y : ℤ, (4^27 + 4^1010 + 4^y) = (2^27 + 2^y)^2 → y ≤ 1992)
:= sorry

end largest_x_for_perfect_square_l541_541881


namespace quadratic_no_real_roots_l541_541377

-- Define the quadratic polynomial f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions: f(x) = x has no real roots
theorem quadratic_no_real_roots (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
sorry

end quadratic_no_real_roots_l541_541377


namespace area_of_triangle_AMN_range_of_k_l541_541974

/-- Problem (i) -/
theorem area_of_triangle_AMN 
  (t : ℝ) (x y k : ℝ) 
  (E : x^2 / 4 + y^2 / 3 = 1) 
  (slope : k > 0) 
  (A : (x = -2, y = 0)) 
  (MA_perp_NA : MA ⊥ NA) 
  (AM_eq_AN : |AM| = |AN|) :
  (area $\triangle$ AMN = 144 / 49) :=
sorry

/-- Problem (ii) -/
theorem range_of_k 
  (t x y k : ℝ) 
  (E : x^2 / t + y^2 / 3 = 1) 
  (slope : k > 0) 
  (A : (x = -√4, y = 0)) 
  (MA_perp_NA : MA ⊥ NA) 
  (AM_2eq_AN : 2 * |AM| = |AN|) :
  (k > (real.cbrt 2) ∧ k < 2) :=
sorry

end area_of_triangle_AMN_range_of_k_l541_541974


namespace range_of_m_l541_541124

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m {m : ℝ} :
  (∀ x : ℝ, (f x m = 0) → (∃ y z : ℝ, y ≠ z ∧ f y m = 0 ∧ f z m = 0)) ∧
  (∀ x : ℝ, f (1 - x) m ≥ -1)
  → (0 ≤ m ∧ m < 1) := 
sorry

end range_of_m_l541_541124


namespace XY_parallel_X_l541_541939

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541939


namespace puzzle_pieces_left_l541_541698

theorem puzzle_pieces_left (total_pieces : ℕ) (children : ℕ) (reyn_places : ℕ) : total_pieces = 500 → children = 4 → reyn_places = 25 →
  let rhys_places := 2 * reyn_places,
      rory_places := 3 * reyn_places,
      rina_places := 4 * reyn_places,
      total_placed := reyn_places + rhys_places + rory_places + rina_places
  in total_pieces - total_placed = 250 :=
by
  intros ht hc hr
  let rhys_places := 2 * reyn_places
  let rory_places := 3 * reyn_places
  let rina_places := 4 * reyn_places
  let total_placed := reyn_places + rhys_places + rory_places + rina_places
  have h_total_pieces : total_pieces = 500 := ht
  have h_children : children = 4 := hc
  have h_reyn_places : reyn_places = 25 := hr
  sorry

end puzzle_pieces_left_l541_541698


namespace int_values_satisfy_abs_sum_l541_541745

theorem int_values_satisfy_abs_sum : 
  let satisfies (a : ℤ) := abs (a + 5) + abs (a - 3) = 8
  in (List.filter satisfies (List.range' (-5) 9)).length = 9 :=
by 
  -- Define the satisfies predicate
  let satisfies (a : ℤ) := abs (a + 5) + abs (a - 3) = 8

  -- Generate the range of integers from -5 to 3 (inclusive)
  let int_list := List.range' (-5) 9
  
  -- Filter the list satisfying the condition
  let filtered_list := List.filter satisfies int_list

  -- Check the length of the filtered list
  exact filtered_list.length = 9

end int_values_satisfy_abs_sum_l541_541745


namespace unique_point_on_polyhedron_l541_541009

-- Define the predicate for a point being outside a polyhedron.
def is_outside (K : Point) (polyhedron : ConvexPolyhedron) : Prop :=
  ∀ (M : Point), M ∈ polyhedron → \not (K ∈ polyhedron)

-- Define the predicate for the closest point to K on the polyhedron.
def closest_point_to (K : Point) (polyhedron : ConvexPolyhedron) (P : Point) : Prop :=
  P ∈ polyhedron ∧ ∀ Q ∈ polyhedron, dist K P ≤ dist K Q

-- Define the predicate for P being within every ball with diameter MK.
def lies_within_every_ball (P : Point) (K : Point) (polyhedron : ConvexPolyhedron) : Prop :=
  ∀ (M : Point), M ∈ polyhedron → 
  let ball_mk := Ball (midpoint M K) (dist M K / 2) in P ∈ ball_mk

-- The unique point P on the polyhedron satisfying the condition.
theorem unique_point_on_polyhedron
  (polyhedron : ConvexPolyhedron) (K : Point)
  (h_outside : is_outside K polyhedron) :
  ∃! (P : Point), P ∈ polyhedron ∧ lies_within_every_ball P K polyhedron :=
sorry

end unique_point_on_polyhedron_l541_541009


namespace unspent_portion_l541_541722

theorem unspent_portion 
  (G : ℝ) 
  (gold_limit : ℝ := G) 
  (platinum_limit : ℝ := 2 * G) 
  (gold_balance : ℝ := (1/3) * G) 
  (platinum_balance : ℝ := (1/3) * platinum_limit) : 
  let new_platinum_balance := platinum_balance + gold_balance in
  let unspent_limit := platinum_limit - new_platinum_balance in
  (unspent_limit / platinum_limit) = (2 / 3) := 
by
  sorry

end unspent_portion_l541_541722


namespace circumcircle_CDE_is_tangent_AB_at_F_l541_541201

section
variable {A B C F D E : Type*}
variable [right_triangle ABC]
variable [hypotenuse AB]
variable [midpoint F AB]
variable [circumcenter D AFC]
variable [circumcenter E BFC]

theorem circumcircle_CDE_is_tangent_AB_at_F
  (h1 : is_right_triangle ABC)
  (h2 : is_hypotenuse AB)
  (h3 : is_midpoint F AB)
  (h4 : is_circumcenter D AFC)
  (h5 : is_circumcenter E BFC) :
  tangent_at (circumcircle C D E) AB F :=
sorry
end

end circumcircle_CDE_is_tangent_AB_at_F_l541_541201


namespace sum_of_roots_eq_two_l541_541276

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l541_541276


namespace reporters_not_covering_politics_l541_541371

def total_reporters : ℝ := 8000
def politics_local : ℝ := 0.12 + 0.08 + 0.08 + 0.07 + 0.06 + 0.05 + 0.04 + 0.03 + 0.02 + 0.01
def politics_non_local : ℝ := 0.15
def politics_total : ℝ := politics_local + politics_non_local

theorem reporters_not_covering_politics :
  1 - politics_total = 0.29 :=
by
  -- Required definition and intermediate proof steps.
  sorry

end reporters_not_covering_politics_l541_541371


namespace smallest_number_among_5_8_1_2_l541_541334

theorem smallest_number_among_5_8_1_2 : ∀ x ∈ ({5, 8, 1, 2} : set ℕ), x ≥ 1 :=
by
  sorry

end smallest_number_among_5_8_1_2_l541_541334


namespace alice_net_amount_spent_l541_541702

noncomputable def net_amount_spent : ℝ :=
  let price_per_pint := 4
  let sunday_pints := 4
  let sunday_cost := sunday_pints * price_per_pint

  let monday_discount := 0.1
  let monday_pints := 3 * sunday_pints
  let monday_price_per_pint := price_per_pint * (1 - monday_discount)
  let monday_cost := monday_pints * monday_price_per_pint

  let tuesday_discount := 0.2
  let tuesday_pints := monday_pints / 3
  let tuesday_price_per_pint := price_per_pint * (1 - tuesday_discount)
  let tuesday_cost := tuesday_pints * tuesday_price_per_pint

  let wednesday_returned_pints := tuesday_pints / 2
  let wednesday_refund := wednesday_returned_pints * tuesday_price_per_pint

  sunday_cost + monday_cost + tuesday_cost - wednesday_refund

theorem alice_net_amount_spent : net_amount_spent = 65.60 := by
  sorry

end alice_net_amount_spent_l541_541702


namespace largest_power_of_18_dividing_factorial_30_l541_541491

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541491


namespace sum_of_lengths_of_intervals_l541_541233

theorem sum_of_lengths_of_intervals :
  (∃ I : set ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → I ⊆ {x : ℝ | (k:ℝ)/(x - k) ≥ 5/4}) ∧
                 (∀ a b : ℝ, a ∈ I ∧ b ∈ I → a ≠ b → (a < b → ∃ c d : ℝ, a < c ∧ c < b ∧ d ∈ I ∧ a < d ∧ d < b)) ∧
                 (∀ (c d : ℝ), c ∈ I ∧ d ∈ I → c ≠ d → abs (c - d) = (1988 : ℕ))) :=
sorry

end sum_of_lengths_of_intervals_l541_541233


namespace solution1_solution2_l541_541242

-- Define the first problem
def equation1 (x : ℝ) : Prop :=
  (x + 1) / 3 - 1 = (x - 1) / 2

-- Prove that x = -1 is the solution to the first problem
theorem solution1 : equation1 (-1) := 
by 
  sorry

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  x - y = 1 ∧ 3 * x + y = 7

-- Prove that x = 2 and y = 1 are the solutions to the system of equations
theorem solution2 : system_of_equations 2 1 :=
by 
  sorry

end solution1_solution2_l541_541242


namespace find_special_n_l541_541910

noncomputable def is_good_divisor (n d_i d_im1 d_ip1 : ℕ) : Prop :=
  2 ≤ d_i ∧ d_i ≤ n ∧ ¬ (d_im1 * d_ip1 % d_i = 0)

def number_of_good_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d_i, ∃ d_im1 d_ip1, is_good_divisor n d_i d_im1 d_ip1)
    (finset.Ico 2 n)).card

def number_of_distinct_prime_divisors (n : ℕ) : ℕ :=
  (nat.factors n).to_finset.card

theorem find_special_n (n : ℕ) :
  number_of_good_divisors n < number_of_distinct_prime_divisors n ↔
  n = 1 ∨ ∃ p a, prime p ∧ n = p^a ∧ a ≥ 1 ∨
  ∃ p q a, prime p ∧ prime q ∧ q > p^a ∧ n = p^a * q ∧ a ≥ 1 :=
by
  sorry

end find_special_n_l541_541910


namespace trig_identity_l541_541677

def c : ℝ := 3 * Real.pi / 14

theorem trig_identity :
  (sin (4 * c) * sin (7 * c) * sin (10 * c) * sin (13 * c)) / 
  (sin (c) * sin (2 * c) * sin (3 * c) * sin (5 * c)) = 
  -sin^2 (Real.pi / 7) := 
by
  sorry

end trig_identity_l541_541677


namespace roots_of_polynomial_l541_541463

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l541_541463


namespace factorization_of_polynomial_l541_541890

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l541_541890


namespace remaining_student_in_sample_l541_541060

def sample_size := 4
def total_students := 52
def students_in_sample := {5, 31, 44}

theorem remaining_student_in_sample : 
  ∃ student_num : ℕ, student_num ∉ students_in_sample ∧ 
  student_num ∈ set.range (λ n, 5 + n * 13) :=
sorry

end remaining_student_in_sample_l541_541060


namespace power_sum_l541_541427

theorem power_sum
: (-2)^(2005) + (-2)^(2006) = 2^(2005) := by
  sorry

end power_sum_l541_541427


namespace extreme_point_at_one_l541_541602

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l541_541602


namespace calculate_correct_subtraction_l541_541600

theorem calculate_correct_subtraction (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 :=
by
  sorry

end calculate_correct_subtraction_l541_541600


namespace largest_n_dividing_factorial_l541_541521

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541521


namespace value_of_expression_l541_541866

theorem value_of_expression (x : ℕ) (h : x = 8) : 
  (x^3 + 3 * (x^2) * 2 + 3 * x * (2^2) + 2^3 = 1000) := by
{
  sorry
}

end value_of_expression_l541_541866


namespace Trapezoid_Circumcircle_Property_l541_541131

variable {A B C D E F I J K : Type}
variables (AB CD EF AD BC AE : Set (A × A))
variables [parallel AB CD] [line E BC] [segment F AD]
variables [angle_eq EAD CBF] (midpointK : midpoint E F K) (neqKIJ: K ≠ I ∧ K ≠ J)
variables [segment E BC] [segment AE CD]

noncomputable def belongs_to_circumcircle {X Y Z : Type} (K : Type) : Prop :=
circle (X, Y, Z, K)

theorem Trapezoid_Circumcircle_Property :
  belongs_to_circumcircle K (A, B, I) ↔ belongs_to_circumcircle K (C, D, J) :=
by sorry

end Trapezoid_Circumcircle_Property_l541_541131


namespace prob_inequality_product_of_sines_l541_541116

variables {n : ℕ}
variables {a b : ℕ → ℝ}

-- Declaring that the sum of sequences a and b from 1 to n are equal
def sum_equal (n : ℕ) : Prop :=
  ∑ i in finset.range n, a i = ∑ i in finset.range n, b i

-- Condition on sequence a
def cond_a (n : ℕ) : Prop :=
  (∀ i, 0 < a 1 = a 2) ∧ (∀ i < n - 2, a i + a (i + 1) = a (i + 2))

-- Condition on sequence b
def cond_b (n : ℕ) : Prop :=
  (∀ i, 0 < b 1 ∧ b 1 ≤ b 2) ∧ (∀ i < n - 2, b i + b (i + 1) ≤ b (i + 2))

theorem prob_inequality (n : ℕ) (h1 : sum_equal n) (h2 : cond_a n) (h3 : cond_b n) :
  a (n - 1) + a n ≤ b (n - 1) + b n := sorry

theorem product_of_sines (n : ℕ) : 
  ∏ i in finset.range n, real.sin ((2 * i + 1) * real.pi / (2 * n)) = 1 / 2 ^ (n - 1) := sorry

end prob_inequality_product_of_sines_l541_541116


namespace inequality_range_l541_541751

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, ¬ (a * x^2 - 2 * a * x - 3 ≥ 0)) ↔ (-3 < a ∧ a ≤ 0) :=
by {
  sorry,
}

end inequality_range_l541_541751


namespace floor_ceil_difference_l541_541423

theorem floor_ceil_difference : 
  let a := (18 / 5) * (-33 / 4)
  let b := ⌈(-33 / 4 : ℝ)⌉
  let c := (18 / 5) * (b : ℝ)
  let d := ⌈c⌉
  ⌊a⌋ - d = -2 :=
by
  sorry

end floor_ceil_difference_l541_541423


namespace larger_sample_size_more_accurate_l541_541648

theorem larger_sample_size_more_accurate
  (population_size : ℕ)
  (sample_size1 sample_size2 : ℕ)
  (accuracy1 accuracy2 : ℝ)
  (h1 : sample_size1 < sample_size2) 
  (h2 : estimate_accuracy population_size sample_size1 = accuracy1)
  (h3 : estimate_accuracy population_size sample_size2 = accuracy2)
  (h4 : ∀ (pop_size : ℕ) (samp_size : ℕ), estimate_accuracy pop_size samp_size = f(samp_size / pop_size)) :
  accuracy1 < accuracy2 :=
by
  sorry

end larger_sample_size_more_accurate_l541_541648


namespace smallest_positive_period_of_f_maximum_value_of_f_monotonically_decreasing_interval_of_f_l541_541577

noncomputable def f (x : ℝ) : ℝ := sqrt(3) * sin (2 * x) - cos (2 * x)

theorem smallest_positive_period_of_f : ∀ T > 0, ∀ x, f(x + T) = f(x) ↔ T = π := 
begin
  sorry
end

theorem maximum_value_of_f : ∃ M, ∀ x, f(x) ≤ M ∧ (∃ x, f(x) = M) ∧ M = 2 :=
begin
  sorry
end
  
theorem monotonically_decreasing_interval_of_f (k : ℤ) : 
  ∀ x, (π/3 + k * π ≤ x ∧ x ≤ 5 * π/6 + k * π) → 
  ∃ I, I = set.Icc (π/3 + (k : ℝ) * π) (5 * π / 6 + k * π) :=
begin
  sorry
end

end smallest_positive_period_of_f_maximum_value_of_f_monotonically_decreasing_interval_of_f_l541_541577


namespace probability_log_diff_l541_541673

theorem probability_log_diff (x : ℝ) (h : 0 < x ∧ x < 1) :
  (∃ p : ℝ, floor (log 10 (5 * x)) - floor (log 10 x) = 0 → p = 8/9) :=
sorry

end probability_log_diff_l541_541673


namespace child_ticket_cost_eq_one_l541_541034

theorem child_ticket_cost_eq_one 
  (adult_ticket_cost : ℕ)
  (total_people : ℕ)
  (total_revenue : ℕ)
  (children_count : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_people = 22)
  (h3 : total_revenue = 50)
  (h4 : children_count = 18) :
  ∃ c : ℕ, 32 + 18 * c = 50 ∧ c = 1 :=
by
  -- adult_count = total_people - children_count
  have h_adult_count : total_people - children_count = 4, from sorry,
  -- adult_revenue = adult_count * adult_ticket_cost
  have h_adult_revenue : adult_ticket_cost * 4 = 32, from sorry,
  use 1,
  -- total_revenue = adult_revenue + children_count * c
  rw [h_adult_revenue, h_adult_count],
  exact ⟨rfl, rfl⟩,
  sorry

end child_ticket_cost_eq_one_l541_541034


namespace sum_of_sequence_l541_541106

noncomputable def a (n : ℕ) : ℕ := n * 2^n

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)

theorem sum_of_sequence (n : ℕ) : S n = (n - 1) * 2 ^ (n + 1) + 2 := sorry

end sum_of_sequence_l541_541106


namespace subset_coloring_exists_l541_541675

variable (S : Set (Fin 2002))
variable (N : ℕ)
variable (hN : 0 ≤ N ∧ N ≤ 2 ^ 2002)

theorem subset_coloring_exists (S : Set (Fin 2002)) (N : ℕ) (hN : 0 ≤ N ∧ N ≤ 2 ^ 2002) :
  ∃ (color : Set (Set (Fin 2002)) → bool),
    (∀ A B, color A = tt ∧ color B = tt → color (A ∪ B) = tt) ∧
    (∀ A B, color A = ff ∧ color B = ff → color (A ∪ B) = ff) ∧
    (∃! W, color W = tt ∧ W.card = N) :=
by
  sorry

end subset_coloring_exists_l541_541675


namespace books_in_special_collection_at_end_of_month_l541_541375

noncomputable def num_books_initial : ℝ := 75
noncomputable def loaned_books : ℝ := 59.99999999999999
noncomputable def loaned_books_rounded : ℝ := 60 -- rounding for practical purposes
noncomputable def return_rate : ℝ := 0.70

theorem books_in_special_collection_at_end_of_month :
  let books_returned := return_rate * loaned_books_rounded in
  let books_not_returned := loaned_books_rounded - books_returned in
  let books_remaining := num_books_initial - books_not_returned in
  books_remaining = 57 := by
  sorry

end books_in_special_collection_at_end_of_month_l541_541375


namespace construct_vertices_of_convex_quadrilateral_l541_541238

theorem construct_vertices_of_convex_quadrilateral
  (A B : ℝ × ℝ) -- known positions of A and B
  (convex_ABCD : convex_quadrilateral A B C D)
  (non_concyclic : ¬ concyclic {A, B, C, D})
  (angle_BCD angle_ADC angle_BCA angle_ACD : ℝ) -- given angles
  (Q_C : ℝ × ℝ) (Q_D : ℝ × ℝ) -- intermediate points
  (k_D k_C : set (ℝ × ℝ)) -- circumcircles

  -- Rotational constructions and their intersection points
  (l1_intersects : ∃ (l1 : ℝ × ℝ), rotate_AB_around_A_by_angle_BCD A B angle_BCD = l1)
  (l2_intersects : ∃ (l2 : ℝ × ℝ), rotate_AB_around_B_by_angle_ADC A B angle_ADC = l2)
  (QC_definition : Q_C ∈ intersection l1 l2)

  (l3_intersects : ∃ (l3 : ℝ × ℝ), rotate_AB_around_A_by_angle_BCA_plus_ACD A B (angle_BCA + angle_ACD) = l3)
  (l4_intersects : ∃ (l4 : ℝ × ℝ), rotate_AB_around_B_by_angle_ACD A B angle_ACD = l4)
  (QD_definition : Q_D ∈ intersection l3 l4)

  -- Circumcircles constructions
  (kD_is_circumcircle : k_D = circumcircle A B Q_C)
  (kC_is_circumcircle : k_C = circumcircle A B Q_D)

  -- Intersections defining points C and D
  (D_definition : D ∈ intersection (line Q_C Q_D) k_D)
  (C_definition : C ∈ intersection (line Q_C Q_D) k_C) :

  -- Final proof statement
  (constructed_C_and_D_correct : vertices_constructed_correctly A B C D Q_C Q_D k_D k_C angle_BCD angle_ADC angle_BCA angle_ACD) :=
sorry

end construct_vertices_of_convex_quadrilateral_l541_541238


namespace hyperbola_standard_equation_given_conditions_l541_541114

noncomputable def standard_equation_of_hyperbola (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_standard_equation_given_conditions :
  ∃ (h : ℝ → ℝ → Prop), h 2 3 ∧ 
  (∀ x y : ℝ, h x y ↔ x^2 - (y^2 / 3) = 1) := 
begin
  use standard_equation_of_hyperbola,
  split,
  { -- Verifying the hyperbola passes through (2,3)
    have : standard_equation_of_hyperbola 2 3,
    { unfold standard_equation_of_hyperbola,
      norm_num, },
    exact this,
  },
  { -- Verifying the standard equation of the hyperbola
    intros x y,
    unfold standard_equation_of_hyperbola,
    exact iff.rfl,
  }
end

end hyperbola_standard_equation_given_conditions_l541_541114


namespace theta_value_l541_541560

noncomputable def sin_2pi_over_3 : ℝ := Real.sin (2 * Real.pi / 3)
noncomputable def cos_2pi_over_3 : ℝ := Real.cos (2 * Real.pi / 3)

theorem theta_value (θ : ℝ) 
  (h1 : P = (sin_2pi_over_3, cos_2pi_over_3))
  (h2 : P lies on the terminal side of θ)
  (h3 : θ ∈ Set.Ico 0 (2 * Real.pi)) :
  θ = 11 / 6 * Real.pi :=
sorry

end theta_value_l541_541560


namespace triangle_similarity_and_center_homothety_l541_541674

noncomputable theory

-- Given conditions 
def similar_segments (F1 F2 F3 : Figure) (A1 B1 A2 B2 A3 B3 A1 C1 A2 C2 A3 C3 : Point) : Prop :=
  corresponding_segments F1 F2 F3 A1 B1 A2 B2 A3 B3 ∧
  corresponding_segments F1 F2 F3 A1 C1 A2 C2 A3 C3

-- Statement of the theorem
theorem triangle_similarity_and_center_homothety {F1 F2 F3 : Figure} {A1 B1 A2 B2 A3 B3 A1 C1 A2 C2 A3 C3 : Point}
  (h_segments : similar_segments F1 F2 F3 A1 B1 A2 B2 A3 B3 A1 C1 A2 C2 A3 C3) :
  (Δ A1 B1 A2 B2 A3 B3 ≃ Δ A1 C1 A2 C2 A3 C3) ∧
  (exists O : Point, center_homothety O A1 B1 A2 B2 A3 B3 A1 C1 A2 C2 A3 C3) ∧
  (in_similarity_circle O F1 F2 F3) :=
sorry

end triangle_similarity_and_center_homothety_l541_541674


namespace area_of_largest_square_l541_541404

theorem area_of_largest_square (XZ YZ XY : ℝ) (h₁ : angle XYZ = 90) 
  (h₂ : XZ^2 + YZ^2 + XY^2 = 450) : XY^2 = 225 := 
sorry

end area_of_largest_square_l541_541404


namespace joan_half_dollars_spent_on_wednesday_l541_541223

variable (x : ℝ)
variable (h1 : x * 0.5 + 14 * 0.5 = 9)

theorem joan_half_dollars_spent_on_wednesday :
  x = 4 :=
by
  -- The proof is not required, hence using sorry
  sorry

end joan_half_dollars_spent_on_wednesday_l541_541223


namespace length_PQ_l541_541180

/-- In trapezoid ABCD with BC parallel to AD, let BC = 1500 and AD = 3000.
Let ∠A = 30°, ∠D = 60°, and let P and Q be the midpoints of BC and AD, respectively.
Determine the length PQ. -/
theorem length_PQ (A B C D P Q : Point) (h_trap : parallel BC AD)
  (h_BC : BC = 1500) (h_AD : AD = 3000) 
  (h_angleA : ∠A = 30°) (h_angleD : ∠D = 60°)
  (h_midP : midpoint P B C) (h_midQ : midpoint Q A D) : 
  length (P - Q) = 750 :=
sorry

end length_PQ_l541_541180


namespace matrix_product_is_zero_l541_541431

variables {a b c d : ℝ}

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [0,  d, -c],
    [-d, 0,  b],
    [c, -b,  0]
  ]

def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [2*a^2, ab, 2*ac],
    [ab, 2*b^2, bc],
    [2*ac, bc, 2*c^2]
  ]

theorem matrix_product_is_zero : matrix_A ⬝ matrix_B = !![
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]
] :=
by
  sorry

end matrix_product_is_zero_l541_541431


namespace triangle_AB_BM_ratio_l541_541181

theorem triangle_AB_BM_ratio (A B C M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M]
  (h_median : is_median A B C M)
  (angle_ABM : angle A B M = 40)
  (angle_MBC : angle M B C = 70) :
  dist A B / dist B M = 2 :=
sorry

end triangle_AB_BM_ratio_l541_541181


namespace factorization_of_polynomial_l541_541891

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l541_541891


namespace tangent_sphere_surface_area_l541_541987

noncomputable def cube_side_length (V : ℝ) : ℝ := V^(1/3)
noncomputable def sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem tangent_sphere_surface_area (V : ℝ) (hV : V = 64) : 
  sphere_surface_area (sphere_radius (cube_side_length V)) = 16 * Real.pi :=
by
  sorry

end tangent_sphere_surface_area_l541_541987


namespace midpoint_ellipse_trajectory_l541_541094

theorem midpoint_ellipse_trajectory (x y x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 / 12) + (y0 / 8) = 1 →
  (x1^2 / 24) + (y1^2 / 16) = 1 →
  (x2^2 / 24) + (y2^2 / 16) = 1 →
  x = (x1 + x2) / 2 →
  y = (y1 + y2) / 2 →
  ∃ x y, ((x - 1)^2 / (5 / 2)) + ((y - 1)^2 / (5 / 3)) = 1 :=
by
  sorry

end midpoint_ellipse_trajectory_l541_541094


namespace apples_ratio_l541_541854

theorem apples_ratio (bonnie_apples samuel_extra_apples samuel_left_over samuel_total_pies : ℕ) 
  (h_bonnie : bonnie_apples = 8)
  (h_samuel_extra : samuel_extra_apples = 20)
  (h_samuel_left_over : samuel_left_over = 10)
  (h_pie_ratio : samuel_total_pies = (8 + 20) / 7) :
  (28 - samuel_total_pies - 10) / 28 = 1 / 2 := 
by
  sorry

end apples_ratio_l541_541854


namespace arithmetic_seq_sum_l541_541159

theorem arithmetic_seq_sum {a : ℕ → ℝ} (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end arithmetic_seq_sum_l541_541159


namespace find_c_for_circle_radius_5_l541_541534

theorem find_c_for_circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 + 8 * y + c = 0 
    → x^2 + 4 * x + y^2 + 8 * y = 5^2 - 25) 
  → c = -5 :=
by
  sorry

end find_c_for_circle_radius_5_l541_541534


namespace range_of_a_l541_541995

noncomputable def f (x : ℝ) := (Real.log (2 * x)) / x

theorem range_of_a 
  (h : ∀ x, f(x)^2 + a * f(x) > 0 → (∃! x : ℤ, 0 < x) → ∃ x1 x2 : ℤ, x1 ≠ x2) :
  -Real.log 2 < a ∧ a ≤ - (1 / 3) * Real.log 6 :=
begin
  sorry
end

end range_of_a_l541_541995


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541511

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541511


namespace cost_of_gravelling_path_l541_541028

open Real

theorem cost_of_gravelling_path :
  let l := 120
  let s := 95
  let h := 65
  let w1 := 2.5
  let w2 := 4
  let cost_per_sq_m := 0.80
  let a := l - w2
  let b := s - w1
  let area := 0.5 * (a + b) * h
  5421 = area * cost_per_sq_m :=
by
  let l := 120
  let s := 95
  let h := 65
  let w1 := 2.5
  let w2 := 4
  let cost_per_sq_m := 0.80
  let a := l - w2
  let b := s - w1
  let area := 0.5 * (a + b) * h
  exact sorry

end cost_of_gravelling_path_l541_541028


namespace magnitude_subtraction_of_unit_vectors_l541_541533

variables {V : Type*} [inner_product_space ℝ V]

/-- Given unit vectors a and b with an angle of π/3 between them, the magnitude of a - b is 1. -/
theorem magnitude_subtraction_of_unit_vectors
  {a b : V} (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (angle_ab : real.angle a b = real.pi / 3) :
  ∥a - b∥ = 1 :=
by
  sorry

end magnitude_subtraction_of_unit_vectors_l541_541533


namespace minimize_expression_l541_541097

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 2 < b) (hab : a + b = 3) : 
  a = 2 / 3 ↔ (∀ x y : ℝ, (0 < x) → (2 < y) → (x + y = 3) → (∃ x_min : ℝ, x_min = 2 / 3 ∧ x_min minimizes (4 / x + 1 / (y-2)))) :=
by
  sorry

end minimize_expression_l541_541097


namespace tan_30_deg_plus_4_sin_30_deg_eq_l541_541438

theorem tan_30_deg_plus_4_sin_30_deg_eq :
  let sin30 := 1 / 2 in
  let cos30 := Real.sqrt 3 / 2 in
  let tan30 := sin30 / cos30 in
  tan30 + 4 * sin30 = (Real.sqrt 3 + 6) / 3 :=
by
  sorry

end tan_30_deg_plus_4_sin_30_deg_eq_l541_541438


namespace length_of_longer_leg_of_smallest_triangle_l541_541453

theorem length_of_longer_leg_of_smallest_triangle :
  ∀ (hypotenuse : ℕ), 
  (∀ (n : ℕ), n = 3 → hypotenuse = 8 → 
  let shorter_leg₁ := hypotenuse / 2 in
  let longer_leg₁ := shorter_leg₁ * sqrt 3 in
  let shorter_leg₂ := longer_leg₁ / 2 in
  let longer_leg₂ := shorter_leg₂ * sqrt 3 in
  let shorter_leg₃ := longer_leg₂ / 2 in
  let longer_leg₃ := shorter_leg₃ * sqrt 3 in
  let shorter_leg₄ := longer_leg₃ / 2 in
  let longer_leg₄ := shorter_leg₄ * sqrt 3 in
  longer_leg₄ = 9 / 2) 
:= sorry

end length_of_longer_leg_of_smallest_triangle_l541_541453


namespace case1_BL_case2_BL_l541_541770

variable (AD BD BL AL : ℝ)

theorem case1_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 3)
  (h₃ : AB = 6 * Real.sqrt 13)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 2 * AL)
  : BL = 16 * Real.sqrt 3 - 12 := by
  sorry

theorem case2_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 6)
  (h₃ : AB = 30)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 4 * AL)
  : BL = (16 * Real.sqrt 6 - 6) / 5 := by
  sorry

end case1_BL_case2_BL_l541_541770


namespace unpainted_area_l541_541771

-- Definitions based on the provided conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def intersection_angle : ℝ := real.pi / 4  -- 45 degrees in radians

-- Statement of the theorem to be proved
theorem unpainted_area :
  let d := board1_width / real.sin intersection_angle in
  let area := board2_width * d in
  area = 35 * real.sqrt 2 :=
by
  sorry

end unpainted_area_l541_541771


namespace all_roots_coincide_l541_541886

-- Define what it means for a quadratic polynomial to have a repeated root.
def has_repeated_root (P : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, P = λ x, a * x^2 + b * x + c ∧ b^2 - 4 * a * c = 0

-- Define the main statement to prove.
theorem all_roots_coincide (P Q : ℝ → ℝ)
  (hP : has_repeated_root P)
  (hQ : has_repeated_root Q)
  (hPQ : has_repeated_root (λ x => P x + Q x)) :
  ∃ r : ℝ, (∃ a b c d : ℝ, P = λ x, a * (x - r)^2 ∧ Q = λ x, b * (x - r)^2 ∧ P x + Q x = (a + b) * (x - r)^2) :=
sorry

end all_roots_coincide_l541_541886


namespace NOQZ_has_same_product_as_MNOQ_l541_541068

/-- Each letter of the alphabet is assigned a value (A=1, B=2, C=3, ..., Z=26). -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13
  | 'N' => 14 | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19
  | 'T' => 20 | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _   => 0  -- We'll assume only uppercase letters are inputs

/-- The product of a four-letter list is the product of the values of its four letters. -/
def list_product (lst : List Char) : ℕ :=
  lst.map letter_value |>.foldl (· * ·) 1

/-- The product of the list MNOQ is calculated. -/
def product_MNOQ : ℕ := list_product ['M', 'N', 'O', 'Q']
/-- The product of the list BEHK is calculated. -/
def product_BEHK : ℕ := list_product ['B', 'E', 'H', 'K']
/-- The product of the list NOQZ is calculated. -/
def product_NOQZ : ℕ := list_product ['N', 'O', 'Q', 'Z']

theorem NOQZ_has_same_product_as_MNOQ :
  product_NOQZ = product_MNOQ := by
  sorry

end NOQZ_has_same_product_as_MNOQ_l541_541068


namespace num_pairs_satisfying_equation_l541_541596

theorem num_pairs_satisfying_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 - y^2 = 204 :=
by
  sorry

end num_pairs_satisfying_equation_l541_541596


namespace remainder_of_31_pow_31_plus_31_div_32_l541_541357

theorem remainder_of_31_pow_31_plus_31_div_32 :
  (31^31 + 31) % 32 = 30 := 
by 
  trivial -- Replace with actual proof

end remainder_of_31_pow_31_plus_31_div_32_l541_541357


namespace moles_of_MgSO4_formed_l541_541524

theorem moles_of_MgSO4_formed
  (moles_Mg : ℕ)
  (moles_H2SO4 : ℕ)
  (balanced_eq : moles_Mg = moles_H2SO4) :
  let moles_MgSO4 := moles_Mg in
  moles_Mg = 2 ∧ moles_H2SO4 = 2 -> moles_MgSO4 = 2 :=
by
  intros
  rw [balanced_eq]
  exact and.right a

end moles_of_MgSO4_formed_l541_541524


namespace max_weight_of_grass_seed_l541_541016

-- Definitions for prices and weights of different bag sizes
def price_per_bag_5 : ℝ := 13.80
def price_per_bag_10 : ℝ := 20.43
def price_per_bag_25 : ℝ := 32.25

def weight_per_bag_5 : ℝ := 5
def weight_per_bag_10 : ℝ := 10
def weight_per_bag_25 : ℝ := 25

-- Condition of minimum weight to be bought
def minimum_weight : ℝ := 65

-- Total cost condition
def total_cost : ℝ := 98.73

-- Lean statement to prove the maximum weight the customer can buy
theorem max_weight_of_grass_seed :
  ∃ (n5 n10 n25 : ℕ), 
    ∑ x in [price_per_bag_5, price_per_bag_10, price_per_bag_25], 
      (if x == price_per_bag_5 then n5 else if x == price_per_bag_10 then n10 else n25) * x ≤ total_cost
    ∧ (n5 * weight_per_bag_5 + n10 * weight_per_bag_10 + n25 * weight_per_bag_25) ≥ minimum_weight
    ∧ (∑ x in [price_per_bag_5, price_per_bag_10, price_per_bag_25], 
      (if x == price_per_bag_5 then n5 else if x == price_per_bag_10 then n10 else n25) * x = total_cost)
    ∧ (n5 * weight_per_bag_5 + n10 * weight_per_bag_10 + n25 * weight_per_bag_25) = 75 := 
  sorry

end max_weight_of_grass_seed_l541_541016


namespace cheryl_more_points_l541_541884

-- Define the number of each type of eggs each child found
def kevin_small_eggs : Nat := 5
def kevin_large_eggs : Nat := 3

def bonnie_small_eggs : Nat := 13
def bonnie_medium_eggs : Nat := 7
def bonnie_large_eggs : Nat := 2

def george_small_eggs : Nat := 9
def george_medium_eggs : Nat := 6
def george_large_eggs : Nat := 1

def cheryl_small_eggs : Nat := 56
def cheryl_medium_eggs : Nat := 30
def cheryl_large_eggs : Nat := 15

-- Define the points for each type of egg
def small_egg_points : Nat := 1
def medium_egg_points : Nat := 3
def large_egg_points : Nat := 5

-- Calculate the total points for each child
def kevin_points : Nat := kevin_small_eggs * small_egg_points + kevin_large_eggs * large_egg_points
def bonnie_points : Nat := bonnie_small_eggs * small_egg_points + bonnie_medium_eggs * medium_egg_points + bonnie_large_eggs * large_egg_points
def george_points : Nat := george_small_eggs * small_egg_points + george_medium_eggs * medium_egg_points + george_large_eggs * large_egg_points
def cheryl_points : Nat := cheryl_small_eggs * small_egg_points + cheryl_medium_eggs * medium_egg_points + cheryl_large_eggs * large_egg_points

-- Statement of the proof problem
theorem cheryl_more_points : cheryl_points - (kevin_points + bonnie_points + george_points) = 125 :=
by
  -- Here would go the proof steps
  sorry

end cheryl_more_points_l541_541884


namespace stamps_total_l541_541664

theorem stamps_total (x y : ℕ) (hx : 1.10 * x + 0.70 * y = 10) : x + y = 12 :=
by
  sorry

end stamps_total_l541_541664


namespace central_angles_sum_l541_541382

noncomputable def sum_central_angles_regular_pentagon (in_circle : Prop) (equal_angles : Prop) : ℕ :=
  if in_circle ∧ equal_angles then 360 else 0

theorem central_angles_sum (in_circle : Prop) (equal_angles : Prop) : sum_central_angles_regular_pentagon in_circle equal_angles = 360 := 
  by
    sorry

end central_angles_sum_l541_541382


namespace find_x_l541_541761

theorem find_x :
  ∃ x, (0.625 * 0.0729 * 28.9) / (0.0017 * 0.025 * x) = 382.5 :=
by
  use 0.0805
  have h_num : 0.625 * 0.0729 * 28.9 = 1.3085625 := by norm_num
  have h_denom : 0.0017 * 0.025 = 0.0000425 := by norm_num
  have h_eq : 1.3085625 / (0.0000425 * 0.0805) = 382.5 := by norm_num
  exact h_eq

end find_x_l541_541761


namespace roots_of_polynomial_l541_541462

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l541_541462


namespace find_angle_BDC_l541_541153

theorem find_angle_BDC
  (CAB CAD DBA DBC : ℝ)
  (h1 : CAB = 40)
  (h2 : CAD = 30)
  (h3 : DBA = 75)
  (h4 : DBC = 25) :
  ∃ BDC : ℝ, BDC = 45 :=
by
  sorry

end find_angle_BDC_l541_541153


namespace roots_of_polynomial_l541_541461

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l541_541461


namespace lines_intersect_at_single_point_l541_541226

open EuclideanGeometry

noncomputable def parallelogram (A B C D : Point) : Prop :=
  ∃ (u v : Vector), 
    A + u = B ∧ B + v = C ∧ C - u = D ∧ D - v = A

noncomputable def divides_same_ratio (A B K L C M D : Point) : Prop :=
  ∃ (r : ℝ), 
    r > 0 ∧ r < 1 ∧ K = (1-r) • A + r • B ∧ L = (1-r) • B + r • C ∧ M = (1-r) • C + r • D

noncomputable def parallel (u v : Vector) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = k • u

noncomputable def line_through (P Q : Point) : Line :=
  { p := P, v := Q - P }

noncomputable def single_intersection (A B C D K L M : Point) : Prop := 
  let b_line := line_through B ((1-r) • K + r • L) in
  let c_line := line_through C ((1-r) • K + r • M) in
  let d_line := line_through D ((1-r) • L + r • M) in
  ∃ (P : Point), 
    b_line.contains P ∧ c_line.contains P ∧ d_line.contains P

theorem lines_intersect_at_single_point 
  (A B C D K L M : Point) (h1 : parallelogram A B C D) 
  (h2 : divides_same_ratio A B K L C M D)
: single_intersection A B C D K L M :=
sorry

end lines_intersect_at_single_point_l541_541226


namespace count_valid_numbers_in_range_l541_541398

theorem count_valid_numbers_in_range :
  ∃! (L : List ℕ), (∀ n ∈ L, 1 ≤ n ∧ n ≤ 100 ∧ 
    n % 2 = 1 ∧ 
    n % 3 = 2 ∧ 
    (n % 5 = 3 ∨ n % 5 = 4)) ∧ L.length = 6 
  :=
begin
  sorry
end

end count_valid_numbers_in_range_l541_541398


namespace largest_n_dividing_30_factorial_l541_541496

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541496


namespace ellipse_properties_l541_541552

noncomputable def ellipse_eccentricity : ℝ := √3 / 2

def ellipse_focus : (ℝ × ℝ) := (-√3, 0)

def is_point_on_ellipse (P : ℝ × ℝ) (x₀ y₀ : ℝ) : Prop :=
  x₀ ≠ 0 ∧ y₀ ≠ 0 ∧ (P.fst = x₀ ∧ P.snd = y₀) ∧ (x₀^2 / 4 + y₀^2 = 1)

def tangent_slope (x₀ y₀ : ℝ) : ℝ := -x₀ / (4 * y₀)

def slope_PF1 (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ + √3)

def slope_PF2 (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ - √3)

def expression_is_constant (x₀ y₀ : ℝ) : Prop :=
  let k₀ := tangent_slope x₀ y₀ in
  let k₁ := slope_PF1 x₀ y₀ in
  let k₂ := slope_PF2 x₀ y₀ in
  (k₁ + k₂) / (k₀ * k₁ * k₂) = -8

theorem ellipse_properties (P : ℝ × ℝ) (x₀ y₀ : ℝ) :
  is_point_on_ellipse P x₀ y₀ →
  expression_is_constant x₀ y₀ :=
sorry

end ellipse_properties_l541_541552


namespace artifact_volume_l541_541812

noncomputable def volume_artifact : ℝ :=
  let a := 2 in
  let h := sqrt 3 in
  let r := sqrt 3 / 3 in
  let volume_prism := (sqrt 3) * 2 in
  let volume_cone := (π * (r * r) * 2 / 3) in
  volume_prism - volume_cone

theorem artifact_volume : 
  volume_artifact = 2 * sqrt 3 - 2 * π / 9 :=
by sorry

end artifact_volume_l541_541812


namespace meet_at_C_l541_541192

-- Define the conditions and statement
variables (s : ℝ) (t : ℝ) (distance : ℝ)

-- Jane's speed is twice Hector's speed
def jane_speed := 2 * s

-- They meet when the combined distance is 24 blocks
def combined_distance := s * t + 2 * s * t

-- Problem statement: prove they meet at point C
theorem meet_at_C (h_jane_h: 2 * s = jane_speed)
  (h_combined: combined_distance = 24)
  (h_t_calc: 3 * s * t = 24)
  (h_t: t = 8 / s)
  (h_hector_distance: s * t = 8)
  (h_jane_distance: jane_speed * t = 16)
  : (8 mod 24) = 8 ∧ (16 mod 24) = 16 ∧ (16 - 8=8) → closest_point = C := 
sorry

end meet_at_C_l541_541192


namespace union_of_P_and_Q_l541_541670

def P : Set ℝ := { x | |x| ≥ 3 }
def Q : Set ℝ := { y | ∃ x, y = 2^x - 1 }

theorem union_of_P_and_Q : P ∪ Q = { y | y ≤ -3 ∨ y > -1 } := by
  sorry

end union_of_P_and_Q_l541_541670


namespace arithmetic_seq_terms_greater_than_50_l541_541400

theorem arithmetic_seq_terms_greater_than_50 :
  let a_n (n : ℕ) := 17 + (n-1) * 4
  let num_terms := (19 - 10) + 1
  ∀ (a_n : ℕ → ℕ), ((a_n 1 = 17) ∧ (∃ k, a_n k = 89) ∧ (∀ n, a_n (n + 1) = a_n n + 4)) →
  ∃ m, m = num_terms ∧ ∀ n, (10 ≤ n ∧ n ≤ 19) → a_n n > 50 :=
by
  sorry

end arithmetic_seq_terms_greater_than_50_l541_541400


namespace proof_by_contradiction_conditions_l541_541177

theorem proof_by_contradiction_conditions :
  ∀ (neg_conclusion : Prop) (orig_cond : Prop) (axioms_theorems : Prop),
  let assumption := neg_conclusion in
  let conditions := orig_cond in
  let axioms := axioms_theorems in
  (assumption ∧ conditions ∧ axioms) → True :=
by
  intros
  sorry

end proof_by_contradiction_conditions_l541_541177


namespace find_roots_of_polynomial_l541_541466

theorem find_roots_of_polynomial :
  (∃ (a b : ℝ), 
    Multiplicity (polynomial.C a) (polynomial.C (Real.ofRat 2)) = 2 ∧ 
    Multiplicity (polynomial.C b) (polynomial.C (Real.ofRat 1)) = 1) ∧ 
  (x^3 - 7 * x^2 + 14 * x - 8 = 
    (x - 1) * (x - 2)^2) := sorry

end find_roots_of_polynomial_l541_541466


namespace sum_even_sub_sum_odd_l541_541079

def sum_arith_seq (a1 an d : ℕ) (n : ℕ) : ℕ :=
  n * (a1 + an) / 2

theorem sum_even_sub_sum_odd :
  let n_even := 50
  let n_odd := 15
  let s_even := sum_arith_seq 2 100 2 n_even
  let s_odd := sum_arith_seq 1 29 2 n_odd
  s_even - s_odd = 2325 :=
by
  sorry

end sum_even_sub_sum_odd_l541_541079


namespace find_roots_of_polynomial_l541_541464

theorem find_roots_of_polynomial :
  (∃ (a b : ℝ), 
    Multiplicity (polynomial.C a) (polynomial.C (Real.ofRat 2)) = 2 ∧ 
    Multiplicity (polynomial.C b) (polynomial.C (Real.ofRat 1)) = 1) ∧ 
  (x^3 - 7 * x^2 + 14 * x - 8 = 
    (x - 1) * (x - 2)^2) := sorry

end find_roots_of_polynomial_l541_541464


namespace parallel_lines_triangle_l541_541943

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541943


namespace range_of_m_l541_541541

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (¬(∃ u v : ℝ, u ≠ v ∧ 4*u^2 + 4*(m - 2)*u + 1 = 0 ∧ 4*v^2 + 4*(m - 2)*v + 1 = 0)) →
  m ∈ set.Ioo (-∞ : ℝ) (-2) ∪ set.Ioc 1 2 ∪ set.Ici 3 :=
sorry

end range_of_m_l541_541541


namespace tangent_of_negative_300_degrees_l541_541265

theorem tangent_of_negative_300_degrees :
  let A := {x // ∃ y, (x, y) ∈ set_of (λ p : ℝ × ℝ, 
    p.1^2 + p.2^2 = 1 ∧ p.2 = (tan (real.pi * -300 / 180)) * p.1)} in 
  ∃ (x y : ℝ), A = (x, y) ∧ y / x = sqrt 3 := 
by 
  sorry

end tangent_of_negative_300_degrees_l541_541265


namespace estimated_survival_probability_l541_541273

theorem estimated_survival_probability :
  let survival_rates := [0.750, 0.825, 0.780, 0.790, 0.801, 0.801]
  ∑ rate in survival_rates, rate / survival_rates.length ≈ 0.80 := 
by
  sorry

end estimated_survival_probability_l541_541273


namespace length_of_third_edge_l541_541260

-- Define the edges and volume of the cuboid
def length := 4
def width := 5
def volume := 120
def height := 6

-- State the theorem to prove
theorem length_of_third_edge : 
  length * width * height = volume :=
by
  -- In a complete proof, the steps to prove length * width * height = volume would be given here
  sorry

end length_of_third_edge_l541_541260


namespace sqrt_neg_num_squared_l541_541863

theorem sqrt_neg_num_squared (n : ℤ) (hn : n = -2023) : Real.sqrt (n ^ 2) = 2023 :=
by
  -- substitute -2023 for n
  rw hn
  -- compute (-2023)^2
  have h1 : (-2023 : ℝ) ^ 2 = 2023 ^ 2 :=
    by rw [neg_sq, abs_of_nonneg (by norm_num)]
  -- show sqrt(2023^2) = 2023
  rw [h1, Real.sqrt_sq]
  exact eq_of_abs_sub_nonneg (by norm_num)

end sqrt_neg_num_squared_l541_541863


namespace optionB_is_difference_of_squares_l541_541342

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l541_541342


namespace percent_savings_12_roll_package_l541_541810

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l541_541810


namespace simplify_complex_fraction_l541_541724

open Complex

theorem simplify_complex_fraction :
  (⟨2, 2⟩ : ℂ) / (⟨-3, 4⟩ : ℂ) = (⟨-14 / 25, -14 / 25⟩ : ℂ) :=
by
  sorry

end simplify_complex_fraction_l541_541724


namespace find_k2_minus_k1_l541_541847

-- Given conditions in Lean definitions
variables (k1 k2 a b : ℝ)
variable h_cond1 : (|k1 - k2| / a = 2)
variable h_cond2 : (|k2 - k1| / b = 3)
variable h_cond3 : (b - a = 10 / 3)

-- The statement to be proved
theorem find_k2_minus_k1
  (k1 k2 a b : ℝ)
  (h1 : |(k1 - k2)/a| = 2)
  (h2 : |(k2 - k1)/b| = 3)
  (h3 : b - a = 10 / 3) :
  k2 - k1 = 4 :=
sorry

end find_k2_minus_k1_l541_541847


namespace XY_parallel_X_l541_541960

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541960


namespace sum_cubes_quadratic_roots_l541_541444

noncomputable def sum_of_cubes_of_roots (x1 x2 : ℚ) (h₁ : x1 + x2 = 6 / 5) (h₂ : x1 * x2 = 1 / 5) : ℚ :=
  x1^3 + x2^3

theorem sum_cubes_quadratic_roots : 
  ∃ x1 x2 : ℚ, (5 * x1^2 - 6 * x1 + 1 = 0) ∧ (5 * x2^2 - 6 * x2 + 1 = 0) ∧
               (x1 + x2 = 6 / 5) ∧ (x1 * x2 = 1 / 5) ∧
               (sum_of_cubes_of_roots x1 x2 (by assumption) (by assumption) = 126 / 125) :=
sorry

end sum_cubes_quadratic_roots_l541_541444


namespace XY_parallel_X_l541_541940

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541940


namespace arrange_composites_on_circle_l541_541654

theorem arrange_composites_on_circle (n : ℕ) (h : n = 10^6) : 
  ∃ f : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ n → i ≠ 1 ∧ (∃ p q : ℕ, p ≠ 1 ∧ q ≠ 1 ∧ p * q = i)) ∧ 
  (∀ i, (f i) ≠ 0 ∧ (∀ j, ((f (i+1)) ≠ 0 ∧ ¬coprime (f i) (f (i+1)))) ∧ (f (n+1) = f 1)) :=
sorry

end arrange_composites_on_circle_l541_541654


namespace largest_n_dividing_30_factorial_l541_541501

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541501


namespace carl_centroid_markable_l541_541668

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

def rad (n : ℕ) : ℕ := n.coprime_part

-- The main problem statement as a Lean theorem
theorem carl_centroid_markable (n : ℕ) :
  ∃ m, (∃ (k : ℕ), m = k * rad (2 * n)) ∧ (∀ (points : list (ℝ × ℝ)), points.length = n →
    let centroid :=
      (list.sum (points.map prod.fst) / n, list.sum (points.map prod.snd) / n) in
    (∃ (a b : ℝ × ℝ), ∃ (div_points : list (ℝ × ℝ)),
      length div_points = m - 1 ∧
      (list.map_with_index (λ (i : ℕ) (_ : ℝ × ℝ), (1 - i / m, i / m)) div_points) = centroid)) :=
sorry

end carl_centroid_markable_l541_541668


namespace percent_savings_per_roll_l541_541808

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l541_541808


namespace AM_GM_inequality_l541_541213

theorem AM_GM_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2) ^ n :=
by
  sorry

end AM_GM_inequality_l541_541213


namespace no_int_solutions_p_mod_4_neg_1_l541_541210

theorem no_int_solutions_p_mod_4_neg_1 :
  ∀ (p n : ℕ), (p % 4 = 3) → (∀ x y : ℕ, x^2 + y^2 ≠ p^n) :=
by
  intros
  sorry

end no_int_solutions_p_mod_4_neg_1_l541_541210


namespace coin_prob_not_unique_l541_541039

theorem coin_prob_not_unique (p : ℝ) (w : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : w = 144 / 625) :
  ¬ ∃! p, (∃ w, w = 10 * p^3 * (1 - p)^2 ∧ w = 144 / 625) :=
by
  sorry

end coin_prob_not_unique_l541_541039


namespace fraction_remain_unchanged_l541_541339

theorem fraction_remain_unchanged (m n a b : ℚ) (h : n ≠ 0 ∧ b ≠ 0) : 
  (a / b = (a + m) / (b + n)) ↔ (a / b = m / n) :=
sorry

end fraction_remain_unchanged_l541_541339


namespace _l541_541205

noncomputable theorem geometric_sum_equation
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (r : ℝ)
  (h1 : r = 5)
  (h2 : ∀ n : ℕ, S n = a 1 * (1 - r^n) / (1 - r))
  (h3 : S 8 - 5 * S 7 = 4) :
  a 1 + a 4 = 504 :=
by
  sorry

end _l541_541205


namespace parallel_vectors_l541_541096

def vec_a : ℝ × ℝ × ℝ := (1, 1, 0)
def vec_b : ℝ × ℝ × ℝ := (-1, 0, 2)

def k_vector_a_add_vector_b (k : ℝ) : ℝ × ℝ × ℝ :=
  (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2, k * vec_a.3 + vec_b.3)

def vector_a_add_k_vector_b (k : ℝ) : ℝ × ℝ × ℝ :=
  (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2, vec_a.3 + k * vec_b.3)

theorem parallel_vectors (k : ℝ) :
  (∃ λ : ℝ, k_vector_a_add_vector_b k = (λ * (1 - k), λ * 1, λ * (2 * k))) ↔ (k = 1 ∨ k = -1) := 
sorry

end parallel_vectors_l541_541096


namespace sum_f_1023_l541_541529

def is_rational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def f (n : ℕ) : ℝ :=
  if is_rational (Real.log (n) / Real.log 4) then Real.log (n) / Real.log 4 else 0

theorem sum_f_1023 : ∑ n in Finset.range 1023.succ, f n = 22.5 := 
  sorry

end sum_f_1023_l541_541529


namespace sum_of_two_pos_implies_one_pos_l541_541613

theorem sum_of_two_pos_implies_one_pos (x y : ℝ) (h : x + y > 0) : x > 0 ∨ y > 0 :=
  sorry

end sum_of_two_pos_implies_one_pos_l541_541613


namespace total_surface_area_of_prism_l541_541626

noncomputable def total_surface_area_prism (a Q : ℝ) : ℝ :=
  let S_base := (Math.sqrt 3 / 4) * a^2
  let C1C := (2 * Q) / (a * Math.sqrt 3)
  let S_lateral_face := a * C1C
  2 * S_base + 3 * S_lateral_face

theorem total_surface_area_of_prism (a Q : ℝ) :
  total_surface_area_prism a Q = Math.sqrt 3 * (0.5 * a^2 + 2 * Q) :=
by
  unfold total_surface_area_prism
  sorry

end total_surface_area_of_prism_l541_541626


namespace buratino_spent_dollars_l541_541851

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end buratino_spent_dollars_l541_541851


namespace book_organizing_activity_l541_541774

theorem book_organizing_activity (x : ℕ) (h₁ : x > 0):
  (80 : ℝ) / (x + 5 : ℝ) = (70 : ℝ) / (x : ℝ) :=
sorry

end book_organizing_activity_l541_541774


namespace no_solution_greater_than_5_l541_541723

theorem no_solution_greater_than_5 (n k : ℕ) (h_n_gt_5 : n > 5) : ¬ ((n - 1)! + 1 = n^k) :=
sorry

end no_solution_greater_than_5_l541_541723


namespace cos_double_angle_l541_541649

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -Real.sqrt 3 / 2) : Real.cos (2 * α) = 1 / 2 :=
by
  sorry

end cos_double_angle_l541_541649


namespace letters_with_straight_line_but_no_dot_l541_541621

theorem letters_with_straight_line_but_no_dot
    (A B C : ℕ)
    (hA : A = 10)
    (hB : B = 6)
    (hC : C = 40) :
    ∃ X : ℕ, X = C - (A + B) ∧ X = 24 :=
by
  use C - (A + B)
  split
  · rw [hC, hA, hB]
    norm_num
  · exact rfl

end letters_with_straight_line_but_no_dot_l541_541621


namespace find_n_l541_541665

variable {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
variable (h : 1/a + 1/b + 1/c = 1/(a + b + c))

theorem find_n (n : ℤ) : (∃ k : ℕ, n = 2 * k - 1) → 
  (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) :=
by
  sorry

end find_n_l541_541665


namespace parallel_lines_triangle_l541_541944

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541944


namespace coefficient_of_x2_term_proof_l541_541099

noncomputable def coefficient_of_x2_term (a : ℕ) : ℕ :=
  let expanded_term := (x^2 + 1) * (a * x + 1)^6
  in sorry  -- Placeholder for the actual polynomial expression expansion

theorem coefficient_of_x2_term_proof (a : ℕ) (h : a > 0) (hsum : (1^2 + 1) * (a + 1)^6 = 1458) :
  coefficient_of_x2_term a = 61 :=
  sorry

end coefficient_of_x2_term_proof_l541_541099


namespace difference_two_smallest_integers_divisible_by_k_l541_541285

theorem difference_two_smallest_integers_divisible_by_k (n1 n2 : ℤ) :
  (∀ k, 4 ≤ k ∧ k ≤ 13 → n1 ≡ 1 [MOD k] ∧ n2 ≡ 1 [MOD k]) →
  n1 > 1 → n2 > 1 → n2 > n1 →
  (n2 - n1 = 60060) := 
by 
  sorry

end difference_two_smallest_integers_divisible_by_k_l541_541285


namespace bretschneider_l541_541855

noncomputable def bretschneider_theorem 
  (a b c d m n : ℝ) 
  (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem bretschneider (a b c d m n A C : ℝ) :
  bretschneider_theorem a b c d m n A C :=
sorry

end bretschneider_l541_541855


namespace XY_parallel_X_l541_541958

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541958


namespace sum_of_roots_quadratic_l541_541908

theorem sum_of_roots_quadratic (y1 y2 : ℝ) (h : ∀ y : ℝ, y ∈ {y1, y2} ↔ y^2 - 6*y + 8 = 0) : y1 + y2 = 6 :=
by sorry

end sum_of_roots_quadratic_l541_541908


namespace smallest_k_for_inequality_l541_541083

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality_l541_541083


namespace profit_percentage_67_15_l541_541383

noncomputable def profit_percentage_first_house (selling_price_first selling_price_second : ℝ) (cost_price_second : ℝ) 
  (total_selling_price : ℝ) (net_profit_percentage : ℝ) : ℝ :=
  let total_cost_price := total_selling_price / (1 + net_profit_percentage) in
  let cost_price_first := total_cost_price - cost_price_second in
  ((selling_price_first - cost_price_first) / cost_price_first) * 100

theorem profit_percentage_67_15 (selling_price : ℝ) (loss_percentage_second net_profit_percentage : ℝ)
  (h_selling_price : selling_price = 10000)
  (h_loss_percentage_second : loss_percentage_second = 0.10)
  (h_net_profit_percentage : net_profit_percentage = 0.17) :
  profit_percentage_first_house selling_price selling_price (selling_price / (1 - loss_percentage_second)) 
    (selling_price + selling_price) net_profit_percentage ≈ 67.15 :=
by 
  sorry

end profit_percentage_67_15_l541_541383


namespace black_square_area_l541_541046

-- Define the edge length of the cube
def edge_length := 12

-- Define the total amount of yellow paint available
def yellow_paint_area := 432

-- Define the total surface area of the cube
def total_surface_area := 6 * (edge_length * edge_length)

-- Define the area covered by yellow paint per face
def yellow_per_face := yellow_paint_area / 6

-- Define the area of one face of the cube
def face_area := edge_length * edge_length

-- State the theorem: the area of the black square on each face
theorem black_square_area : (face_area - yellow_per_face) = 72 := by
  sorry

end black_square_area_l541_541046


namespace polynomial_factor_condition_l541_541266

theorem polynomial_factor_condition (c p : ℝ) : 
  (∃ a : ℝ, 3x^3 + cx + 12 = (x^2 + px + 4) * (3x + a)) → c = 9 :=
by
  sorry

end polynomial_factor_condition_l541_541266


namespace simplify_trigonometric_expression_1_correct_simplify_trigonometric_expression_2_correct_simplify_trigonometric_expression_3_correct_l541_541726

noncomputable def simplify_trigonometric_expression_1 : ℝ :=
  sin (76 * Real.pi / 180) * cos (74 * Real.pi / 180) + sin (14 * Real.pi / 180) * cos (16 * Real.pi / 180)

theorem simplify_trigonometric_expression_1_correct : simplify_trigonometric_expression_1 = 1 / 2 :=
  sorry

noncomputable def simplify_trigonometric_expression_2 : ℝ :=
  (1 - tan (59 * Real.pi / 180)) * (1 - tan (76 * Real.pi / 180))

theorem simplify_trigonometric_expression_2_correct : simplify_trigonometric_expression_2 = 2 :=
  sorry

noncomputable def simplify_trigonometric_expression_3 : ℝ :=
  (sin (7 * Real.pi / 180) + cos (15 * Real.pi / 180) * sin (8 * Real.pi / 180)) /
  (cos (7 * Real.pi / 180) - sin (15 * Real.pi / 180) * sin (8 * Real.pi / 180))

theorem simplify_trigonometric_expression_3_correct : simplify_trigonometric_expression_3 = 2 - Real.sqrt 3 :=
  sorry

end simplify_trigonometric_expression_1_correct_simplify_trigonometric_expression_2_correct_simplify_trigonometric_expression_3_correct_l541_541726


namespace fraction_of_number_l541_541315

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541315


namespace union_of_sets_M_N_l541_541129

open Set

theorem union_of_sets_M_N :
  let M := {0, 1}
  let N := {1, 2}
  M ∪ N = {0, 1, 2} :=
by
  let M := {0, 1}
  let N := {1, 2}
  sorry

end union_of_sets_M_N_l541_541129


namespace fraction_of_number_l541_541304

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541304


namespace inequality_proof_l541_541209

theorem inequality_proof
  (n : ℕ)
  (h2 : n ≥ 2)
  (a : Fin n → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (S : ℝ := ∑ i, a i)
  :
  ∑ i, (1 + a i) / Real.sqrt (S - a i) ≥ 2 * n / Real.sqrt (n - 1) :=
sorry

end inequality_proof_l541_541209


namespace find_complex_number_l541_541681

theorem find_complex_number (z : ℂ) (h : (z + 2 * conj z) = 3 - I) : z = 1 + I := 
by sorry

end find_complex_number_l541_541681


namespace tan_sin_30_computation_l541_541436

theorem tan_sin_30_computation :
  let θ := 30 * Real.pi / 180 in
  Real.tan θ + 4 * Real.sin θ = (Real.sqrt 3 + 6) / 3 :=
by
  let θ := 30 * Real.pi / 180
  have sin_30 : Real.sin θ = 1 / 2 := by sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by sorry
  have tan_30 : Real.tan θ = Real.sin θ / Real.cos θ := by sorry
  have sin_60 : Real.sin (2 * θ) = Real.sqrt 3 / 2 := by sorry
  sorry

end tan_sin_30_computation_l541_541436


namespace solve_logarithmic_equation_l541_541270

theorem solve_logarithmic_equation :
  ∃ x : ℝ, (log 3 (x^2 - 10) = 1 + log 3 x) ∧ x = 5 :=
by
  -- formal definition of the logarithmic equation to be solved
  have log_eq : ∀ x : ℝ, (x > 0) ∧ (x^2 - 10 > 0) → log 3 (x^2 - 10) = 1 + log 3 x → x = 5,
  {
    -- proof steps can be added here but we place 'sorry' as per instructions
    sorry
  }
  -- use the defined condition and result
  use 5
  split
  {
    -- proof of the validity of the solution
    split
    {
      -- x > 0
      linarith,
      -- x^2 - 10 > 0
      linarith
    },
    -- accurate logarithmic equation
    sorry
  }
  {
    -- show that x = 5
    refl
  }

end solve_logarithmic_equation_l541_541270


namespace wrapping_paper_area_l541_541008

theorem wrapping_paper_area (w h d : ℝ) :
    let total_height := h + d,
        diagonal_of_base := w * Real.sqrt 2,
        side_length := diagonal_of_base + total_height
    in side_length^2 = (w * Real.sqrt 2 + h + d)^2 := by
  sorry

end wrapping_paper_area_l541_541008


namespace sum_of_six_digits_l541_541727

-- Define the set of digits
def digit_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}

-- Define the existence of six different digits that meet the required conditions
theorem sum_of_six_digits 
  (a b c d e f g: ℕ) 
  (ha : a ∈ digit_set) 
  (hb : b ∈ digit_set) 
  (hc : c ∈ digit_set) 
  (hd : d ∈ digit_set) 
  (he : e ∈ digit_set) 
  (hf : f ∈ digit_set) 
  (hg : g ∈ digit_set) 
  (h_neq_ac: a ≠ c) 
  (h_neq_ad: a ≠ d)
  (h_neq_ae: a ≠ e)
  (h_neq_af: a ≠ f)
  (h_neq_ag: a ≠ g)
  (h_neq_bc: b ≠ c)
  (h_neq_bd: b ≠ d)
  (h_neq_be: b ≠ e)
  (h_neq_bf: b ≠ f)
  (h_neq_bg: b ≠ g)
  (h_neq_cd: c ≠ d)
  (h_neq_ce: c ≠ e)
  (h_neq_cf: c ≠ f)
  (h_neq_cg: c ≠ g)
  (h_neq_de: d ≠ e)
  (h_neq_df: d ≠ f)
  (h_neq_dg: d ≠ g)
  (h_neq_ef: e ≠ f)
  (h_neq_eg: e ≠ g)
  (h_neq_fg: f ≠ g)
  (hvc : a + b + c = 24) 
  (hrc : d + e + f + g = 15) 
  (hb_eq_e : b = e) 
  : a + b + c + d + f + g = 30 :=
by 
  sorry

end sum_of_six_digits_l541_541727


namespace complex_eq_num_solutions_l541_541593

/-- We want to prove the number of complex numbers z such
that e^z = (z - 1) / (z + 1) under a modulus constraint -/
theorem complex_eq_num_solutions :
  {z : ℂ | abs z < 50 ∧ exp z = (z - 1) / (z + 1)}.to_finset.card = 16 :=
sorry

end complex_eq_num_solutions_l541_541593


namespace problem_solution_l541_541122

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 1) - 1 / 2

theorem problem_solution (a : ℝ) :
  (f (a + 1) + f (a ^ 2 - 1) > 0) ↔ (a ∈ set.Ioo (-1 : ℝ) 0) :=
by {
  sorry -- proof to be filled in
}

end problem_solution_l541_541122


namespace expected_points_gain_or_loss_l541_541806

-- Define the points associated with each type of outcome
def points (outcome : ℕ) : ℤ :=
  if outcome = 6 then -5
  else outcome

-- Define the probability of each type of outcome
def prob (outcome : ℕ) : ℚ :=
  if outcome = 6 then 1 / 6
  else if outcome ∈ {2, 4} then 1 / 3
  else if outcome ∈ {1, 3, 5} then 1 / 2
  else 0

-- Define the set of possible outcomes
def outcomes := {1, 2, 3, 4, 5, 6}

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  ∑ outcome in outcomes, (prob outcome) * (points outcome)

-- Theorem statement asserting expected value calculation correctness
theorem expected_points_gain_or_loss :
  expected_value = 5.67 := 
sorry

end expected_points_gain_or_loss_l541_541806


namespace num_of_scoring_schemes_l541_541804

/-- Definitions for scores and participants --/
inductive Score
| F | G | E

def judges := Fin 8
def participants := Fin 3

variable (score : judges → Score)
variable (A B C : participants)

/-- Conditions based on the problem statement --/

/-- Condition: A and B get the same score from exactly 4 people --/
def same_score (x y : judges → Score) : ℕ := 
  (Finset.univ.filter (λ i => x i = y i)).card

axiom a_and_b_same_score : same_score score A B = 4

/-- Condition: C differs from A in at least 4 positions --/
def differing_positions (x y : judges → Score) : ℕ := 
  (Finset.univ.filter (λ i => x i ≠ y i)).card 

axiom c_differs_from_a : differing_positions score A C ≥ 4

/-- Condition: C differs from B in at least 4 positions --/
axiom c_differs_from_b : differing_positions score B C ≥ 4

/-- Main theorem to prove --/
theorem num_of_scoring_schemes : (number_of_valid_scoring_schemes : ℕ)
  := 2401

end num_of_scoring_schemes_l541_541804


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541510

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541510


namespace factorization_of_M_l541_541456

theorem factorization_of_M :
  ∀ (x y z : ℝ), x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = 
  (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end factorization_of_M_l541_541456


namespace container_volume_ratio_l541_541418

theorem container_volume_ratio (A B : ℕ) 
  (h1 : (3 / 4 : ℚ) * A = (5 / 8 : ℚ) * B) :
  (A : ℚ) / B = 5 / 6 :=
by
  admit
-- sorry

end container_volume_ratio_l541_541418


namespace sheila_hourly_rate_is_6_l541_541353

variable (weekly_earnings : ℕ) (hours_mwf : ℕ) (days_mwf : ℕ) (hours_tt: ℕ) (days_tt : ℕ)
variable [NeZero hours_mwf] [NeZero days_mwf] [NeZero hours_tt] [NeZero days_tt]

-- Define Sheila's working hours and weekly earnings as given conditions
def weekly_hours := (hours_mwf * days_mwf) + (hours_tt * days_tt)
def hourly_rate := weekly_earnings / weekly_hours

-- Specific values from the given problem
def sheila_weekly_earnings : ℕ := 216
def sheila_hours_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_tt : ℕ := 6
def sheila_days_tt : ℕ := 2

-- The theorem to prove
theorem sheila_hourly_rate_is_6 :
  (sheila_weekly_earnings / ((sheila_hours_mwf * sheila_days_mwf) + (sheila_hours_tt * sheila_days_tt))) = 6 := by
  sorry

end sheila_hourly_rate_is_6_l541_541353


namespace total_books_read_l541_541348

theorem total_books_read (c s : ℕ) : (∑ (i : ℕ) in finset.range c, 60 * s) = 60 * c * s :=
by sorry

end total_books_read_l541_541348


namespace function_relation_l541_541993

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem function_relation:
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by 
  sorry

end function_relation_l541_541993


namespace minimize_ratio_of_segments_l541_541029

noncomputable def triangle_inscribed_in_circle (A B C : Point) (G : Circle) : Prop :=
  ∃ (O : Point) (R : ℝ), G = circle O R ∧ (on_circle G A) ∧ (on_circle G B) ∧ (on_circle G C)

noncomputable def intersection_of_ray_with_circle (A M : Point) (G : Circle) : Point :=
  sorry -- Implementation would define A1 based on intersection with ray AM and circle G

theorem minimize_ratio_of_segments
  (A B C : Point) (G : Circle)
  (h_triangle : triangle_inscribed_in_circle A B C G)
  (M : Point)
  (h_M_in_triangle : inside_triangle A B C M) :
  ∃ I : Point, incenter A B C I ∧
  ∀ M, inside_triangle A B C M →
      (let A1 := intersection_of_ray_with_circle A M G in
      (BM * CM) / (distance M A1)) ≥ 2 * (radius_of_incircle A B C) := 
sorry

end minimize_ratio_of_segments_l541_541029


namespace bananas_eaten_l541_541406

variable (initial_bananas : ℕ) (remaining_bananas : ℕ)

theorem bananas_eaten (initial_bananas remaining_bananas : ℕ) (h_initial : initial_bananas = 12) (h_remaining : remaining_bananas = 10) : initial_bananas - remaining_bananas = 2 := by
  -- Proof goes here
  sorry

end bananas_eaten_l541_541406


namespace fixed_points_when_a1_b_neg2_range_of_a_when_two_distinct_fixed_points_l541_541089

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

theorem fixed_points_when_a1_b_neg2 : 
  ∀ x : ℝ, (x = (f 1 (-2) x)) → (x = -1 ∨ x = 3) := 
by
  sorry

theorem range_of_a_when_two_distinct_fixed_points : 
  ∀ a b : ℝ, (a ≠ 0) → (∃ x : ℝ, x = (f a b x)) → (b ∈ ℝ) →
  ((∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a b x₁ = x₁ ∧ f a b x₂ = x₂)) → 
  (0 < a ∧ a < 1) :=
by
  sorry

end fixed_points_when_a1_b_neg2_range_of_a_when_two_distinct_fixed_points_l541_541089


namespace whites_wash_time_l541_541841

-- Define the conditions
variables (
  (W T : ℕ) -- W is whites washing time, T is total time
  (D_wash D_dry C_wash C_dry : ℕ) -- Washing and drying times for darks and colors
)

-- Specify the concrete values for times (as given in the problem)
axiom whites_drying_time : D_dry = 50
axiom darks_washing_time : D_wash = 58
axiom darks_drying_time : D_dry = 65
axiom colors_washing_time : C_wash = 45
axiom colors_drying_time : C_dry = 54
axiom total_time : T = 344

-- Define the equation for total time
def total_laundry_time (W D_wash D_dry C_wash C_dry T : ℕ) : Prop := 
  W + 50 + 58 + 65 + 45 + 54 = 344

-- Theorem statement: The time it takes for whites in washing machine
theorem whites_wash_time (h1 : total_laundry_time W D_wash D_dry C_wash C_dry T) : 
  W = 72 :=
by {
  -- This is the part where you provide the proof
  sorry
}

end whites_wash_time_l541_541841


namespace range_of_a_l541_541976

variables (a b c : ℝ)

theorem range_of_a (h₁ : a^2 - b * c - 8 * a + 7 = 0)
                   (h₂ : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l541_541976


namespace arithmetic_sequence_sufficient_but_not_necessary_condition_l541_541586

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a_1_a_3_equals_2a_2 (a : ℕ → ℤ) :=
  a 1 + a 3 = 2 * a 2

-- Statement of the mathematical problem
theorem arithmetic_sequence_sufficient_but_not_necessary_condition (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a_1_a_3_equals_2a_2 a ∧ (a_1_a_3_equals_2a_2 a → ¬ is_arithmetic_sequence a) :=
by
  sorry

end arithmetic_sequence_sufficient_but_not_necessary_condition_l541_541586


namespace length_of_DE_l541_541255

variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variable (AB : LineSegment A B) (DE : LineSegment D E)
variable (baseAB : SegmentLength AB = 15)
variable (is_parallel : ∀ (D E : Type), Parallel(AB, DE))
variable (area_condition : ∀ (Δ ABC ΔXZY : Triangle), Area ΔXZY = 0.25 * Area Δ ABC)

theorem length_of_DE (d : distance Metric) : d D E = 7.5 :=
  sorry

end length_of_DE_l541_541255


namespace largest_divisor_18n_max_n_l541_541474

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541474


namespace problem1_problem2_l541_541926

variable (a b c : ℕ → ℝ)

-- Conditions
def S (n : ℕ) : ℝ := n^2 + p * n
def T (n : ℕ) : ℝ := 3 * n^2 - 2 * n

-- Definitions derived from the conditions
def an (n : ℕ) : ℝ := S n - S (n-1)
def bn (n : ℕ) : ℝ := T n - T (n-1)

-- Proof problem statements
theorem problem1 (h : ∀ p, an 10 = bn 10) : p = 36 :=
sorry

theorem problem2 : (c n = bn (2 * n - 1)) ↔ (c n = 12n - 11) :=
sorry

end problem1_problem2_l541_541926


namespace equilateral_triangle_hyperbola_area_square_l541_541762

noncomputable def equilateral_triangle_area_square :
  {A B C : ℝ × ℝ // 
    A.1 * A.2 = 4 ∧ 
    B.1 * B.2 = 4 ∧ 
    C.1 * C.2 = 4 ∧ 
    (A.1 + B.1 + C.1) / 3 = 1 ∧ 
    (A.2 + B.2 + C.2) / 3 = 1 ∧ 
    dist A B = dist B C ∧ 
    dist B C = dist C A} → 
    ℝ :=
λ ⟨A, B, C, hA, hB, hC, hcentroid_x, hcentroid_y, hdist1, hdist2⟩, (6 * real.sqrt 3) ^ 2

theorem equilateral_triangle_hyperbola_area_square :
  ∀ t : {A B C : ℝ × ℝ // 
    A.1 * A.2 = 4 ∧ 
    B.1 * B.2 = 4 ∧ 
    C.1 * C.2 = 4 ∧ 
    (A.1 + B.1 + C.1) / 3 = 1 ∧ 
    (A.2 + B.2 + C.2) / 3 = 1 ∧ 
    dist A B = dist B C ∧ 
    dist B C = dist C A}, 
  equilateral_triangle_area_square t = 108 :=
sorry

end equilateral_triangle_hyperbola_area_square_l541_541762


namespace triangle_parallel_bisectors_l541_541955

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541955


namespace problem_statement_l541_541530

theorem problem_statement : 
  (finset.filter 
    (λ n : ℕ, (1 ≤ n ∧ n ≤ 60) ∧ (nat.factorial ((n + 1)^2 - 1) % (nat.factorial n)^(n + 1) = 0)) 
    (finset.range 61)).card = 59 :=
begin
  sorry
end

end problem_statement_l541_541530


namespace largest_n_dividing_factorial_l541_541520

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541520


namespace volume_of_rotated_segment_l541_541708

theorem volume_of_rotated_segment (a R : ℝ) (h : R > a / 2) :
  let V := 2 * π * ∫ x in -a/2..a/2, (R^2 - x^2)
  V = (π * a^3) / 6 :=
by
  -- Insert proof here
  sorry

end volume_of_rotated_segment_l541_541708


namespace pair_points_intersect_l541_541544

theorem pair_points_intersect :
  ∀ (points : Fin 22 → EuclideanGeometry.Point2D),
    (∀ (i j k : Fin 22),
      i ≠ j → j ≠ k → i ≠ k → ¬ EuclideanGeometry.are_collinear ({points i, points j, points k} : set EuclideanGeometry.Point2D)) →
    ∃ (pairs : Finset (Fin 22 × Fin 22)),
      pairs.card = 11 ∧
      (∀ (p q r s : Fin 22),
        (p, q) ∈ pairs → (r, s) ∈ pairs → disjoint {(points p, points q).segment, (points r, points s).segment} → 
        ∃ (inter : EuclideanGeometry.Point2D),
          (points p, points q).segment.inter (points r, points s).segment = {inter}) →
      5 ≤ set.card 
        {p q r s : Fin 22 |
          (p, q) ∈ pairs ∧ (r, s) ∈ pairs ∧ ∃ (inter : EuclideanGeometry.Point2D),
            (points p, points q).segment.inter (points r, points s).segment = {inter}} :=
by
  sorry

end pair_points_intersect_l541_541544


namespace problem1_problem2_l541_541996

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Problem 1: (0 < m < 1/e) implies g(x) = f(x) - m has two zeros
theorem problem1 (m : ℝ) (h1 : 0 < m) (h2 : m < 1 / Real.exp 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = m ∧ f x2 = m :=
sorry

-- Problem 2: (2/e^2 ≤ a < 1/e) implies f^2(x) - af(x) > 0 has only one integer solution
theorem problem2 (a : ℝ) (h1 : 2 / (Real.exp 2) ≤ a) (h2 : a < 1 / Real.exp 1) :
  ∃! x : ℤ, ∀ y : ℤ, (f y)^2 - a * (f y) > 0 → y = x :=
sorry

end problem1_problem2_l541_541996


namespace part_I_part_II_l541_541590

noncomputable def vector_a : ℝ × ℝ := (1, 0)
noncomputable def vector_b : ℝ × ℝ := (1, 2)
noncomputable def vector_c : ℝ × ℝ := (0, 1)

noncomputable def vector_AB := - vector_a + (3 : ℝ) • vector_c
noncomputable def vector_AC := (4 : ℝ) • vector_a - (2 : ℝ) • vector_c

theorem part_I :
  ∃ (λ μ : ℝ), vector_c = λ • vector_a + μ • vector_b :=
begin
  use [-1 / 2, 1 / 2],
  sorry,
end

theorem part_II :
  angle_between vector_AB vector_AC = 3 * real.pi / 4 :=
begin
  sorry,
end

end part_I_part_II_l541_541590


namespace arithmetic_sequence_property_geometric_sequence_property_l541_541063

namespace ProofProblem 

-- Arithmetic sequence conditions and definitions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_property 
  (a : ℕ → ℤ) (d : ℤ) (h_arith : arithmetic_sequence a d) (h_d_nonzero : d ≠ 0) :
  (∀ n < 2011, sum_first_n a n = sum_first_n a (2011 - n)) → a 1006 = 0 :=
  sorry

-- Geometric sequence conditions and definitions
def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = q * b n

def product_first_n (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  (b 1) ^ n * q ^ (n * (n - 1) / 2)

theorem geometric_sequence_property 
  (b : ℕ → ℤ) (q : ℤ) (h_geom : geometric_sequence b q) (h_q_diff_one : q ≠ 1) :
  (∀ n < 23, product_first_n b n = product_first_n b (23 - n)) → b 12 = 1 :=
  sorry

end ProofProblem

end arithmetic_sequence_property_geometric_sequence_property_l541_541063


namespace friction_force_l541_541022

theorem friction_force (m : ℝ) (g : ℝ) (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  friction_force = m * g * real.sqrt (1 - (real.cos α * real.cos β) ^ 2) :=
sorry

end friction_force_l541_541022


namespace coverage_ring_area_correct_l541_541291

noncomputable def coverage_ring_area : ℝ :=
  let r := 15 in  -- radius of each radar's coverage
  let n := 8 in   -- number of radars, forming a regular octagon
  let w := 18 in  -- width of the coverage ring
  let θ := 180 / n in -- central angle in degrees
  
  let tan_θ := Real.tan (Real.pi * θ / 180) in
  let inner_radius := Real.sqrt (r^2 - (w/2)^2) in
  let outer_radius := inner_radius + w in
  let inner_area := Real.pi * inner_radius^2 in
  let outer_area := Real.pi * outer_radius^2 in
  
  outer_area - inner_area
  
theorem coverage_ring_area_correct :
  coverage_ring_area = 432 * Real.pi / Real.tan (22.5 * Real.pi / 180) := by
  sorry 

end coverage_ring_area_correct_l541_541291


namespace john_bought_metres_l541_541657

-- Define the conditions
def total_cost := 425.50
def cost_per_metre := 46.00

-- State the theorem
theorem john_bought_metres : total_cost / cost_per_metre = 9.25 :=
by
  sorry

end john_bought_metres_l541_541657


namespace triangle_area_l541_541738

theorem triangle_area : 
  ∀ (x y: ℝ), (x / 5 + y / 2 = 1) → (x = 5) ∨ (y = 2) → ∃ A : ℝ, A = 5 :=
by
  intros x y h1 h2
  -- Definitions based on the problem conditions
  have hx : x = 5 := sorry
  have hy : y = 2 := sorry
  have base := 5
  have height := 2
  have area := 1 / 2 * base * height
  use area
  sorry

end triangle_area_l541_541738


namespace probability_quadrant_l541_541100

theorem probability_quadrant (a : Fin 2024 → ℝ)
  (h₁ : ∀ i, a i ≠ 0)
  (h₂ : ∑ i, |a i| / a i = 2000) :
  let num_neg := (Finset.univ.filter (λ i, a i < 0)).card in
  (num_neg : ℝ) / 2024 = 3 / 506 :=
sorry

end probability_quadrant_l541_541100


namespace triangle_parallel_bisectors_l541_541953

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541953


namespace percentage_problem_l541_541606

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 400) : 1.20 * x = 2400 :=
by
  sorry

end percentage_problem_l541_541606


namespace num_permutations_with_first_two_vowels_l541_541716

-- Define the word "CONTEST"
def word : List Char := ['C', 'O', 'N', 'T', 'E', 'S', 'T']

-- Define the vowels in "CONTEST"
def vowels : List Char := ['O', 'E']

-- Define the consonants in "CONTEST"
def consonants : List Char := ['C', 'N', 'T', 'S', 'T']

-- Define function to calculate factorial
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define function to calculate permutations of a list with repetitions
def permutations_with_repetition (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (*) 1 (List.map factorial repeats)

-- The main theorem statement
theorem num_permutations_with_first_two_vowels : 
  let num_first_two_vowels := factorial 2
  let num_remaining_permutations := permutations_with_repetition 5 [2]
  num_first_two_vowels * num_remaining_permutations = 120 :=
by
  sorry

end num_permutations_with_first_two_vowels_l541_541716


namespace gain_amount_is_ten_l541_541052

theorem gain_amount_is_ten (S : ℝ) (C : ℝ) (g : ℝ) (G : ℝ) 
  (h1 : S = 110) (h2 : g = 0.10) (h3 : S = C + g * C) : G = 10 :=
by 
  sorry

end gain_amount_is_ten_l541_541052


namespace find_angle_FYD_l541_541172

noncomputable def angle_FYD (AB CD AXF FYG : ℝ) : ℝ := 180 - AXF

theorem find_angle_FYD (AB CD : ℝ) (AXF : ℝ) (FYG : ℝ) (h1 : AB = CD) (h2 : AXF = 125) (h3 : FYG = 40) :
  angle_FYD AB CD AXF FYG = 55 :=
by
  sorry

end find_angle_FYD_l541_541172


namespace at_least_one_cell_covered_square_fits_in_triangle_l541_541004

theorem at_least_one_cell_covered (T1 T2: set (ℝ × ℝ)) (S : set (ℝ × ℝ)) (h1: S = { (0, 0), (0, 1), (1, 0), (1, 1) }) :
  (S ⊆ T1 ∪ T2) → (∃ (cell : ℝ × ℝ), (cell ∈ S) ∧ (cell ⊆ T1 ∨ cell ⊆ T2)) → False :=
by
  sorry

theorem square_fits_in_triangle (T1 T2: set (ℝ × ℝ)) (h1: (2, 2) = (T1 ∪ T2).dimension) :   
  (∃ T ∈ {T1, T2}, (∃ (P : set (ℝ × ℝ)), P = {(0, 0), (1, 0), (0, 1), (1, 1)} ∧ P ⊆ T)) :=
by
  sorry

end at_least_one_cell_covered_square_fits_in_triangle_l541_541004


namespace fifth_part_value_l541_541144

-- Define the total amount to be divided
def totalAmount : ℝ := 8154

-- Define the seven ratios as a list of fractions
def ratios : List ℝ :=
  [5/11, 7/15, 11/19, 2/9, 17/23, 13/29, 19/31]

-- Compute the sum of the ratios
def sumRatios : ℝ := ratios.sum

-- Compute the fifth part using the given proportionality
def fifthPart : ℝ := (17/23) / sumRatios * totalAmount

-- State the theorem to prove the value of the fifth part
theorem fifth_part_value : fifthPart = 1710.05 := by
  sorry

end fifth_part_value_l541_541144


namespace triangle_AC_length_l541_541651

open Real

theorem triangle_AC_length (A : ℝ) (AB AC S : ℝ) (h1 : A = π / 3) (h2 : AB = 2) (h3 : S = sqrt 3 / 2) : AC = 1 :=
by
  sorry

end triangle_AC_length_l541_541651


namespace largest_divisor_18n_max_n_l541_541479

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541479


namespace division_by_3_l541_541338

theorem division_by_3 (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := 
sorry

end division_by_3_l541_541338


namespace volume_larger_side_l541_541815

-- Lean code does not compute volume directly and only sets up the geometric problem
-- Definitions of geometric entities based on given conditions

def point (x y z : ℚ) : ℚ × ℚ × ℚ := (x, y, z)

def A : ℚ × ℚ × ℚ := point 0 0 0
def C : ℚ × ℚ × ℚ := point 2 2 0
def D : ℚ × ℚ × ℚ := point 0 2 0
def E : ℚ × ℚ × ℚ := point 0 0 2
def F : ℚ × ℚ × ℚ := point 2 0 2

def midpoint (p1 p2 : ℚ × ℚ × ℚ) : ℚ × ℚ × ℚ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def P : ℚ × ℚ × ℚ := midpoint C D
def Q : ℚ × ℚ × ℚ := midpoint E F

def vector (p1 p2 : ℚ × ℚ × ℚ) : ℚ × ℚ × ℚ :=
(p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def cross_product (u v : ℚ × ℚ × ℚ) : ℚ × ℚ × ℚ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def normal_vector : ℚ × ℚ × ℚ := cross_product (vector A P) (vector A Q)

def plane_eq (normal : ℚ × ℚ × ℚ) (P : ℚ × ℚ × ℚ) : ℚ → ℚ → ℚ → Prop :=
λ x y z, normal.1 * x + normal.2 * y + normal.3 * z = normal.1 * P.1 + normal.2 * P.2 + normal.3 * P.3

def volume_of_larger_solid : ℚ :=
8 - -- Compute the actual volume of the smaller segment formed (Placeholder)
sorry

theorem volume_larger_side (A P Q : ℚ × ℚ × ℚ) :
  volume_of_larger_solid = -- Correct value of larger volume
sorry :=
sorry

end volume_larger_side_l541_541815


namespace proportion_x_l541_541352

theorem proportion_x (x : ℝ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_x_l541_541352


namespace mean_score_l541_541526

noncomputable theory

variables {M SD : ℝ}

-- Conditions
def condition1 : Prop := 60 = M - 2 * SD
def condition2 : Prop := 100 = M + 3 * SD

-- Statement
theorem mean_score (h1 : condition1) (h2 : condition2) : M = 76 :=
sorry

end mean_score_l541_541526


namespace probability_of_real_roots_l541_541021

noncomputable def polynomial := (x : ℝ) → x^4 + 3 * a * x^3 + (3 * a - 3) * x^2 + (-5 * a + 4) * x - 3

theorem probability_of_real_roots :
  let interval := set.Icc (-10 : ℝ) 15 in
  let a_unif : interval → ℝ := λ x, ((fst x).val + (snd x).val) / 2 in
  let prob := (23 / 25 : ℝ) in
  ∀ a ∈ interval, 
  ∃ q : ℝ → ℝ, (q(a)) = polynomial a :=
sorry

end probability_of_real_roots_l541_541021


namespace simplify_expression_l541_541907

theorem simplify_expression :
  (3 * Real.sqrt 8) / 
  (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  - (2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 :=
by
  sorry

end simplify_expression_l541_541907


namespace sum_first_12_terms_l541_541112

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean {α : Type} [Field α] (a b c : α) : Prop :=
b^2 = a * c

def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
n * (a 1 + a n) / 2

theorem sum_first_12_terms 
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : arithmetic_sequence a 1)
  (h2 : geometric_mean (a 3) (a 6) (a 11)) :
  sum_arithmetic_sequence a 12 = 96 :=
sorry

end sum_first_12_terms_l541_541112


namespace values_of_a_system_sol_within_square_l541_541091

theorem values_of_a_system_sol_within_square :
  ∃ x y (a : ℝ), (x * Real.sin a - y * Real.cos a = 2 * Real.sin a - Real.cos a) 
                 ∧ (x - 3 * y + 13 = 0) 
                 ∧ (5 ≤ x ∧ x ≤ 9) 
                 ∧ (3 ≤ y ∧ y ≤ 7) 
                 ↔ ∃ k : ℤ, (a = Real.pi / 4 + Real.pi * k) ∨ (a = Real.arctan (5 / 3) + Real.pi * k) :=
sorry

end values_of_a_system_sol_within_square_l541_541091


namespace triangle_parallel_bisectors_l541_541956

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541956


namespace total_bees_approx_l541_541826

-- Define a rectangular garden with given width and length
def garden_width : ℝ := 450
def garden_length : ℝ := 550

-- Define the average density of bees per square foot
def bee_density : ℝ := 2.5

-- Define the area of the garden in square feet
def garden_area : ℝ := garden_width * garden_length

-- Define the total number of bees in the garden
def total_bees : ℝ := bee_density * garden_area

-- Prove that the total number of bees approximately equals 620,000
theorem total_bees_approx : abs (total_bees - 620000) < 1000 :=
by
  sorry

end total_bees_approx_l541_541826


namespace find_x_l541_541134

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_a_minus_b (x : ℝ) : ℝ × ℝ := ((1 - x), (4))

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition of perpendicular vectors
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The theorem to prove
theorem find_x : ∃ x : ℝ, is_perpendicular vector_a (vector_a_minus_b x) ∧ x = 9 :=
by {
  -- Sorry statement used to skip proof
  sorry
}

end find_x_l541_541134


namespace XY_parallel_X_l541_541965

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541965


namespace smallest_n_intersection_nonempty_l541_541208

open Finset

theorem smallest_n_intersection_nonempty :
  ∀ (X : Finset ℕ) (subsets : Finset (Finset X)),
    (∀ A ∈ subsets, A.card ≤ 56) →
    subsets.card = 15 →
    (∀ S : Finset (Finset X), S ⊆ subsets → S.card = 7 → (S.sup id).card ≥ 41) →
    ∃ (S1 S2 S3 : Finset X), S1 ∈ subsets ∧ S2 ∈ subsets ∧ S3 ∈ subsets ∧ (S1 ∩ S2 ∩ S3).nonempty := 
sorry

end smallest_n_intersection_nonempty_l541_541208


namespace find_range_of_m_l541_541543

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end find_range_of_m_l541_541543


namespace Bogan_attempt_second_time_l541_541047

variable (m1 e1 e2 T : Nat)

theorem Bogan_attempt_second_time
  (h_initial_maggots : m1 = 10)
  (h_initial_eaten : e1 = 1)
  (h_total_maggots : T = 20) :
  T - m1 = 10 :=
by
  rw [h_initial_maggots, h_total_maggots]
  norm_num

end Bogan_attempt_second_time_l541_541047


namespace ben_initial_cards_l541_541045

theorem ben_initial_cards (T : ℕ) (B : ℕ) (h1 : T = 20) (h2 : B + 3 = 2 * T) : B = 37 :=
by {
  rw h1 at h2, 
  norm_num at h2,
  exact h2,
  sorry
}

end ben_initial_cards_l541_541045


namespace polynomial_roots_correct_l541_541468

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l541_541468


namespace max_pairs_distinct_sums_l541_541916

theorem max_pairs_distinct_sums : ∀ (k : ℕ), k ≤ 1199 ∧ 
  (∃ (a b : finset ℕ), 
    (∀ i, i ∈ a → i < 3001) ∧ 
    (∀ j, j ∈ b → j < 3001) ∧ 
    (∀ i j, i ∈ a → j ∈ b → i < j) ∧ 
    (∀ i j k l, i < k → j < l → i + j ≠ k + l) ∧ 
    (∀ i j, i ∈ a → j ∈ b → i + j ≤ 3000) ∧
    (a.card = k) ∧ (b.card = k)) :=
by
  sorry

end max_pairs_distinct_sums_l541_541916


namespace expected_value_Z_variance_Z_l541_541846

-- Define the probability mass functions for X and Y
def pmf_X (x : ℕ) : ℝ :=
  if x = 1 then 0.1 else if x = 2 then 0.6 else if x = 3 then 0.3 else 0.0

def pmf_Y (y : ℕ) : ℝ :=
  if y = 0 then 0.2 else if y = 1 then 0.8 else 0.0

-- Define the random variables X and Y
noncomputable def X : ℕ → ℝ := λ x, pmf_X x
noncomputable def Y : ℕ → ℝ := λ y, pmf_Y y

-- Define Z as X + Y
noncomputable def Z : ℕ → ℕ → ℝ := λ x y, X x + Y y

-- Expected value and variance of Z as per the problem statement
theorem expected_value_Z : 
  (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y) * (pmf_X x * pmf_Y y)) = 3 := sorry

theorem variance_Z : 
  (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y)^2 * (pmf_X x * pmf_Y y)) - (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y) * (pmf_X x * pmf_Y y))^2 = 0.52 := sorry

end expected_value_Z_variance_Z_l541_541846


namespace intersection_P_Q_l541_541095

section set_intersection

variable (x : ℝ)

def P := { x : ℝ | x ≤ 1 }
def Q := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : { x | x ∈ P ∧ x ∈ Q } = { x | -1 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end set_intersection

end intersection_P_Q_l541_541095


namespace bake_sale_cookies_total_l541_541836

noncomputable def total_cookies : ℕ :=
  let abigail := 2 * 48
  let grayson := (3 / 4) * 48
  let olivia := 3 * 48
  let isabella := (1 / 2) * grayson
  let ethan_initial := (2 * 2) * 48
  let ethan_converted := ethan_initial / 2
  let ethan := ethan_converted - (0.25 * ethan_converted)
  abigail + grayson + olivia + isabella + ethan

theorem bake_sale_cookies_total : total_cookies = 366 := by
  sorry

end bake_sale_cookies_total_l541_541836


namespace part_I_part_II_l541_541573

noncomputable def f (x : ℝ) : ℝ := (1 + real.log (x + 1)) / x

theorem part_I (x : ℝ) (hx : x ∈ (-1, 0) ∪ (0, +∞)) :
  ∀ y z ∈ (set.Ioo (-1 : ℝ) 0 ∪ set.Ioi (0 : ℝ)), y < z → f y > f z :=
sorry

theorem part_II (x : ℝ) (hx : 0 < x) : 
  f x > 3 / (x + 1) :=
sorry

end part_I_part_II_l541_541573


namespace circumcenter_parallel_l541_541198

-- Define the main theorem statement
theorem circumcenter_parallel {A B C D E F K M N O1 O2 : Type*} [triangle A B C] 
  (hK : is_midpoint K A D) (hE : perp DE AB E) (hF : perp DF AC F) 
  (hM : line_intersection KE BC M) (hN : line_intersection KF BC N)
  (hO1 : is_circumcenter O1 (triangle D E M)) (hO2 : is_circumcenter O2 (triangle D F N)) :
  parallel O1 O2 BC :=
begin
  sorry
end

end circumcenter_parallel_l541_541198


namespace sine_of_largest_angle_in_isosceles_triangle_l541_541164

theorem sine_of_largest_angle_in_isosceles_triangle (r : ℝ) (h1 : r > 0) 
  (h2 : ∃ (α β γ : ℝ), α + β + γ = π ∧ α = α ∧ γ = 2 * α ∧ isosceles ∧ base = 4 * r) :
  ∃ β : ℝ, sin β = 24 / 25 :=
sorry

end sine_of_largest_angle_in_isosceles_triangle_l541_541164


namespace proof_problem_l541_541848

theorem proof_problem (x y z w : ℕ) (h1 : x^3 = y^2) (h2 : z^5 = w^4) (h3 : z - x = 31)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hw_pos : 0 < w) : w - y = -2351 := by
  sorry

end proof_problem_l541_541848


namespace parallel_vectors_l541_541591

variable {x : ℝ}

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem parallel_vectors (h : add_vectors a b = sub_vectors a b) : x = 2 :=
by
  sorry

end parallel_vectors_l541_541591


namespace sin_2pi_minus_alpha_l541_541109

theorem sin_2pi_minus_alpha (α : ℝ) (h₁ : Real.cos (α + Real.pi) = Real.sqrt 3 / 2) (h₂ : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
    Real.sin (2 * Real.pi - α) = -1 / 2 := 
sorry

end sin_2pi_minus_alpha_l541_541109


namespace hexagon_intersect_l541_541553

theorem hexagon_intersect {A B C A1 A2 B1 B2 C1 C2 : Point} 
  (h_equilateral : equilateral_triangle A B C)
  (h_hexagon : equal_sides_hexagon A1 A2 B1 B2 C1 C2) :
  concurrent_lines (line_through A1 B2) (line_through B1 C2) (line_through C1 A2) :=
sorry

end hexagon_intersect_l541_541553


namespace find_percentage_l541_541148

variable (P : ℝ)

def percentage_condition (P : ℝ) : Prop :=
  P * 30 = (0.25 * 16) + 2

theorem find_percentage : percentage_condition P → P = 0.2 :=
by
  intro h
  -- Proof steps go here
  sorry

end find_percentage_l541_541148


namespace work_hours_per_week_l541_541789

variable (h : ℝ)

def hourly_rate : ℝ := 12.50
def widget_rate : ℝ := 0.16
def widgets_produced : ℝ := 1250
def earnings_target : ℝ := 700

theorem work_hours_per_week 
  (h : ℝ) (hourly_rate : ℝ) (widget_rate : ℝ) (widgets_produced : ℝ) (earnings_target : ℝ)
  (wage_equation : hourly_rate * h + widget_rate * widgets_produced = earnings_target) : 
  h = 40 :=
by
  have widget_earnings : ℝ := widget_rate * widgets_produced
  have hourly_earnings_eq : hourly_rate * h = earnings_target - widget_earnings
  have h_value : h = (earnings_target - widget_earnings) / hourly_rate
  rw [widget_earnings, hourly_earnings_eq, h_value]
  sorry

end work_hours_per_week_l541_541789


namespace area_shape_DEFHGT_l541_541454

/-- 
Equilateral triangle ABC has side length 2, squares ABDE and CAFG are formed outside the triangle,
and BCHT is an equilateral triangle formed outside the triangle. We are asked to prove that 
the area of the geometric shape formed by DEFGHT is 3 * sqrt(3) - 2.
-/
theorem area_shape_DEFHGT :
  ∀ (A B C D E F G H T : Type) 
  (ABC_eq_triangle : is_equilateral_triangle ABC 2)
  (ABDE_eq_square : is_square ABDE 2)
  (CAFG_eq_square : is_square CAFG 2)
  (BCHT_eq_triangle : is_equilateral_triangle BCHT 2),
  area (shape DEFGHT) = 3 * sqrt(3) - 2 :=
by sorry

end area_shape_DEFHGT_l541_541454


namespace charging_time_l541_541694

theorem charging_time (fast_rate : ℝ) (reg_rate : ℝ) (t : ℝ)
    (H1 : fast_rate = 1 / 180)
    (H2 : reg_rate = 1 / 540)
    (H3 : t / 3 * fast_rate + 2 * t / 3 * reg_rate = 1) :
    t = 324 := by
  skip

end charging_time_l541_541694


namespace conic_sections_of_given_equation_l541_541067

-- Defining the conditions
def given_equation (y z : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 8

-- Specifying the mathematical proof problem
theorem conic_sections_of_given_equation (y z : ℝ) :
  (∀ y z, given_equation y z → (conic_section y z) = "hyperbola" ∨ (conic_section y z) = "ellipse") := sorry

-- Auxiliary definition
def conic_section (y z : ℝ) : String :=
  if y^2 < 0 then "hyperbola"
  else if y^2 >= 0 then "ellipse"
  else "unknown"

end conic_sections_of_given_equation_l541_541067


namespace intersection_point_l541_541773

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x^2 + 11

theorem intersection_point :
  ∃ x y : ℝ, f x = y ∧ g x = y ∧ x = 2 ∧ y = 19 := by
  sorry

end intersection_point_l541_541773


namespace common_difference_divisible_l541_541732

theorem common_difference_divisible (p : ℕ → ℕ) (d : ℕ) : 
  (∀ n, prime (p n)) → 
  (∀ n, p (n + 1) = p n + d) → 
  (p 0 < p 1 ∧ p 1 < p 2 ∧ p 2 < p 3 ∧ p 3 < p 4 ∧ p 4 < p 5 ∧
   p 5 < p 6 ∧ p 6 < p 7 ∧ p 7 < p 8 ∧ p 8 < p 9 ∧
   p 9 < p 10 ∧ p 10 < p 11 ∧ p 11 < p 12 ∧
   p 12 < p 13 ∧ p 13 < p 14 ∧ p 14 < p 15) →
  2 ∣ d ∧ 3 ∣ d ∧ 5 ∣ d ∧ 7 ∣ d ∧ 11 ∣ d ∧ 13 ∣ d :=
by
  intros prime_p arith_prog primes_ordered
  sorry

end common_difference_divisible_l541_541732


namespace network_connections_l541_541286

theorem network_connections (n m : ℕ) (hn : n = 30) (hm : m = 5) 
(h_total_conn : (n * 4) / 2 = 60) : 
60 + m = 65 :=
by
  sorry

end network_connections_l541_541286


namespace hiking_and_break_time_l541_541777

def violet_water_consumption : ℝ := 800 -- ml per hour
def buddy_water_consumption : ℝ := 400 -- ml per hour
def violet_carry_capacity : ℝ := 4.8 * 1000 -- converted to ml
def buddy_carry_capacity : ℝ := 1.5 * 1000 -- converted to ml
def break_frequency : ℝ := 2 -- hours
def break_duration : ℝ := 0.5 -- hours

def total_water := violet_carry_capacity + buddy_carry_capacity
def total_consumption_per_hour := violet_water_consumption + buddy_water_consumption
def hiking_hours := total_water / total_consumption_per_hour
def number_of_breaks := (hiking_hours / break_frequency).floor
def total_break_time := number_of_breaks * break_duration

theorem hiking_and_break_time : 
  (total_water / total_consumption_per_hour) + ((total_water / total_consumption_per_hour) / break_frequency).floor * break_duration = 6.25 := by 
  sorry

end hiking_and_break_time_l541_541777


namespace fraction_multiplication_l541_541308

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541308


namespace center_of_circle_l541_541007

theorem center_of_circle (a b : ℝ) :
  (∃ (a b : ℝ), (b - 1) / (a - 1) = -1/2 ∧ b = 3/2 ∧ (a - 1)^2 + (b - 2)^2 = (a - 1)^2 + (b - 1)^2) 
  → (a = 0 ∧ b = 3/2) :=
begin
  sorry
end

end center_of_circle_l541_541007


namespace difference_in_ticket_cost_l541_541412

theorem difference_in_ticket_cost 
  (children_ticket_cost : ℝ) (adult_ticket_cost : ℝ) (total_cost : ℝ) (discount : ℝ) 
  (num_adult_tickets : ℕ) (num_children_tickets : ℕ) (diff_expected : ℝ) :
  children_ticket_cost = 4.25 →
  discount = 2 →
  total_cost = 30 →
  num_adult_tickets = 2 →
  num_children_tickets = 4 →
  num_adult_tickets * adult_ticket_cost + num_children_tickets * children_ticket_cost - discount = total_cost →
  adult_ticket_cost - children_ticket_cost = diff_expected →
  diff_expected = 3.25 := 
by 
  intros hc hd ht na nc ha hd
  sorry

end difference_in_ticket_cost_l541_541412


namespace largest_power_of_18_dividing_factorial_30_l541_541489

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541489


namespace type_R_completion_time_l541_541031

theorem type_R_completion_time :
  (∃ R : ℝ, (2 / R + 3 / 7 = 1 / 1.2068965517241381) ∧ abs (R - 5) < 0.01) :=
  sorry

end type_R_completion_time_l541_541031


namespace shape_given_theta_const_l541_541527

noncomputable def cylindrical_coordinates (r : ℝ) (theta : ℝ) (z : ℝ) : Prop := True

theorem shape_given_theta_const (c : ℝ) : 
  (∃ r z : ℝ, cylindrical_coordinates r (c + π / 4) z) → is_plane (λ (r θ z : ℝ), θ = c + π / 4) :=
by
  sorry

end shape_given_theta_const_l541_541527


namespace circumcircle_radius_is_sqrt6_area_of_triangle_is_3sqrt3_over_2_l541_541185

-- Definitions for the problem conditions
variables {A B C : Type*} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (a b c : ℝ)
variables (∠A ∠B ∠C : ℝ)
variables (sinA sinB sinC : ℝ)
variables (circumcircle_radius area_triangle : ℝ)

-- Conditions and problem statements
axiom side_opposite_A : ∠A ≠ 0 ∧ ∠A ≠ π
axiom side_opposite_B : b = 3 * real.sqrt 2
axiom sin_eqn : (c * sinC) / sinA - c = (b * sinB) / sinA - a
axiom angle_bisector_intersect : ∃ D : Point, D ∈ AC ∧ BD = real.sqrt 3

-- Problem 1: Prove the radius of the circumcircle
theorem circumcircle_radius_is_sqrt6 :
  circumcircle_radius = real.sqrt 6 :=
sorry

-- Problem 2: Prove the area of the triangle
theorem area_of_triangle_is_3sqrt3_over_2 :
  area_triangle = (3 * real.sqrt 3) / 2 :=
sorry

end circumcircle_radius_is_sqrt6_area_of_triangle_is_3sqrt3_over_2_l541_541185


namespace distance_between_A_and_B_l541_541069

variables (A B C : Type) (Eddy Freddy : A → Type)
variables (t : ℝ)

def travel_time_E := 3
def travel_time_F := 3
def distance_AC := 300
def speed_ratio : ℝ := 2 / 1

noncomputable def V_F := distance_AC / travel_time_F
noncomputable def V_E := speed_ratio * V_F
noncomputable def Distance_AB := V_E * travel_time_E

theorem distance_between_A_and_B (V_F : ℝ) (V_E : ℝ) :
  Distance_AB = 600 :=
by 
  -- includes conditions
  have t_E : travel_time_E = 3 := rfl,
  have t_F : travel_time_F = 3 := rfl,
  have d_AC : distance_AC = 300 := rfl,
  have ratio : speed_ratio = 2 := rfl,
  sorry -- Proof steps are omitted as instructed

end distance_between_A_and_B_l541_541069


namespace product_of_modified_numbers_less_l541_541289

theorem product_of_modified_numbers_less
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by {
   sorry
}

end product_of_modified_numbers_less_l541_541289


namespace broadcast_sequences_count_l541_541835

noncomputable def count_broadcast_sequences (n_commercials n_games : ℕ) : ℕ :=
  if n_commercials = 3 ∧ n_games = 2 then
    2 * 3 * 3.factorial
  else 0

theorem broadcast_sequences_count :
  count_broadcast_sequences 3 2 = 36 :=
by
  sorry

end broadcast_sequences_count_l541_541835


namespace angle_measure_of_isosceles_triangle_with_pentagon_and_equilateral_triangle_l541_541844

theorem angle_measure_of_isosceles_triangle_with_pentagon_and_equilateral_triangle :
  ∀ (A B C D E F G H : Type) (circle : A → ℝ) (is_regular_pentagon : A → A → A → A → A → Prop)
  (is_equilateral_triangle : F → G → H → Prop)
  (common_vertex : A → F → Prop)
  (isosceles_triangle : A → F → A → A → Prop),
  is_regular_pentagon A B C D E →
  is_regular_pentagon A B C D E →
  is_equilateral_triangle F G H →
  common_vertex A F →
  common_vertex A F →
  isosceles_triangle A F A H →
  ∠F A I = 60 :=
by
  intros
  sorry

end angle_measure_of_isosceles_triangle_with_pentagon_and_equilateral_triangle_l541_541844


namespace largest_n_18n_divides_30_factorial_l541_541506

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541506


namespace buratino_spent_dollars_l541_541850

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end buratino_spent_dollars_l541_541850


namespace find_n_l541_541473

def satisfies_congruences (n : ℤ) : Prop :=
  (50 ≤ n ∧ n ≤ 200) ∧ (n % 8 = 0) ∧ (n % 6 = 4) ∧ (n % 7 = 3)

theorem find_n : ∃ n : ℤ, satisfies_congruences n ∧ n = 136 :=
by {
  use 136,
  unfold satisfies_congruences,
  split,
  { split,
    { norm_num, },
    { norm_num, } },
  split,
  { rw [Int.mod_eq_of_lt, Int.mod_def, Int.sub_self, Int.mod_self],
    exact ⟨rfl⟩,
    norm_num }, 
  split,
  { norm_num },
  { norm_num }
}
sorry

end find_n_l541_541473


namespace violet_balls_least_count_l541_541620

noncomputable def smallest_violet_balls (x : ℕ) : ℕ := 
  x - ((47 * x) / 60) - 27

theorem violet_balls_least_count :
  (∃ x y : ℕ,
    y = smallest_violet_balls x ∧ 
    x / 10 + x / 8 + x / 3 + (x / 10 + 9) + (x / 8 + 10) + 8 + y = x ∧ 
    8 = x / 8 + 10 ∧ 
    y = 25) :=
begin
  sorry
end

end violet_balls_least_count_l541_541620


namespace cos_double_angle_l541_541601

theorem cos_double_angle :
  ∀ α: ℝ, sin (π / 6 + α) = 3 / 5 → cos (2 * α - 2 * π / 3) = -7 / 25 :=
by
  intros α h_sin
  sorry

end cos_double_angle_l541_541601


namespace better_pi_approximation_l541_541394

theorem better_pi_approximation (
  h1 : Real.pi ≈ Real.ofRat (22 / 7),
  h2 : Real.pi ≈ Real.ofRat (223 / 71)
) : (Real.abs (Real.pi - Real.ofRat (223 / 71)) < Real.abs (Real.pi - Real.ofRat (22 / 7))) :=
sorry

end better_pi_approximation_l541_541394


namespace XY_parallel_X_l541_541942

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541942


namespace highway_length_l541_541772

open Nat Real

-- Conditions
def carA_speed : ℝ := 13
def carB_speed : ℝ := 17
def meeting_time : ℝ := 2

-- Question to proof
theorem highway_length :
  let combined_speed := carA_speed + carB_speed in
  let total_distance := combined_speed * meeting_time in
  total_distance = 60 := by
  sorry

end highway_length_l541_541772


namespace fraction_difference_l541_541422

theorem fraction_difference : (let x := 0.%overline 36 in 
                                let y := 0.36 in 
                                x - y = 4 / 1100) := 
by
  sorry

end fraction_difference_l541_541422


namespace parallel_lines_triangle_l541_541949

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541949


namespace ira_addition_olya_subtraction_addition_l541_541421

theorem ira_addition (x : ℤ) (h : (11 + x) / (41 + x : ℚ) = 3 / 8) : x = 7 :=
  sorry

theorem olya_subtraction_addition (y : ℤ) (h : (37 - y) / (63 + y : ℚ) = 3 / 17) : y = 22 :=
  sorry

end ira_addition_olya_subtraction_addition_l541_541421


namespace rachel_editing_time_l541_541714

theorem rachel_editing_time :
  (let write_time := 6 * 30 in
   let research_time := 45 in
   let total_time := 5 * 60 in
   write_time + research_time + editing_time = total_time) →
  editing_time = 75 :=
by
  intros h
  have write_time_def : write_time = 6 * 30 := rfl
  have research_time_def : research_time = 45 := rfl
  have total_time_def : total_time = 5 * 60 := rfl
  rw [write_time_def, research_time_def, total_time_def] at h
  rw [mul_comm 6 30, mul_comm 5 60] at h
  linarith
  sorry


end rachel_editing_time_l541_541714


namespace factorization_of_polynomial_l541_541899

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l541_541899


namespace area_vehicle_reach_l541_541640
-- Import the necessary Mathlib

-- Define the conditions
def speed_on_highways : ℝ := 60
def speed_in_desert : ℝ := 15
def time_minutes : ℝ := 4
def time_hours : ℝ := time_minutes / 60

-- The theorem to be proven
theorem area_vehicle_reach (m n : ℕ) (h_mn : m / n = 200 / 7 ∧ Nat.coprime m n) : m + n = 207 :=
by
  sorry

end area_vehicle_reach_l541_541640


namespace overall_average_is_correct_l541_541793

def section1 : ℕ := 40
def section2 : ℕ := 35
def section3 : ℕ := 45
def section4 : ℕ := 42

def mean1 : ℝ := 50
def mean2 : ℝ := 60
def mean3 : ℝ := 55
def mean4 : ℝ := 45

-- Calculate total marks for each section
def totalMarks := section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- Calculate total number of students
def totalStudents := section1 + section2 + section3 + section4

-- Calculate the overall average marks per student
def overallAverage := totalMarks / totalStudents

theorem overall_average_is_correct :
  overallAverage = 52.25 := by
  sorry

end overall_average_is_correct_l541_541793


namespace decreasing_power_function_on_neg_infty_to_0_l541_541451

theorem decreasing_power_function_on_neg_infty_to_0 :
  ∀ x ∈ set.Iio 0, (y = x^2) → (y' < 0) :=
sorry

end decreasing_power_function_on_neg_infty_to_0_l541_541451


namespace volume_of_ABDH_is_4_3_l541_541064

-- Define the vertices of the cube
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def D : (ℝ × ℝ × ℝ) := (0, 2, 0)
def H : (ℝ × ℝ × ℝ) := (0, 0, 2)

-- Function to calculate the volume of the pyramid
noncomputable def volume_of_pyramid (A B D H : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * 2 * 2

-- Theorem stating the volume of the pyramid ABDH is 4/3 cubic units
theorem volume_of_ABDH_is_4_3 : volume_of_pyramid A B D H = 4 / 3 := by
  sorry

end volume_of_ABDH_is_4_3_l541_541064


namespace ice_cream_amount_l541_541397

/-- Given: 
    Amount of ice cream eaten on Friday night: 3.25 pints
    Total amount of ice cream eaten over both nights: 3.5 pints
    Prove: 
    Amount of ice cream eaten on Saturday night = 0.25 pints -/
theorem ice_cream_amount (friday_night saturday_night total : ℝ) (h_friday : friday_night = 3.25) (h_total : total = 3.5) : 
  saturday_night = total - friday_night → saturday_night = 0.25 :=
by
  intro h
  rw [h_total, h_friday] at h
  simp [h]
  sorry

end ice_cream_amount_l541_541397


namespace ellipse_theorem_l541_541117

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (ell_eq : (a > b)) : bool :=
  (a > b ) &&
  (∃ (c : ℝ), c = sqrt 3 ∧ 
              (a / 2 = sqrt ((a ^ 2) - (b ^ 2))) ∧
              (ell_eq : (a * a = b * b + c * c)))

noncomputable def intersection_area (a b : ℝ) (ha : a > 0) (hb : b > 0) (MN_line : (line_through F with slope 45°)) : ℝ :=
  let M := find_intersection_ellipse_and_line slope slope MN_line
  let N := find_intersection_ellipse_and_line slope slope MN_line
  intersection_area := triangle_area O M N

theorem ellipse_theorem (a b : ℝ) (ha : a > b > 0) (F := (sqrt 3, 0)) (e := sqrt 3 / 2) :
  (∃ (a : ℝ) (b : ℝ), 
  ∃ a = 2 b = 1  :
    (ellipse_equation a b ha hb ∧
    intersection_area a b ha hb = 2 * sqrt 6 / 5) :=
begin
  sorry
end

end ellipse_theorem_l541_541117


namespace fraction_of_number_l541_541322

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541322


namespace factorization_of_polynomial_l541_541898

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l541_541898


namespace rise_in_water_level_l541_541816

theorem rise_in_water_level (edge base_length base_width : ℝ) (cube_volume base_area rise : ℝ) 
  (h₁ : edge = 5) (h₂ : base_length = 10) (h₃ : base_width = 5)
  (h₄ : cube_volume = edge^3) (h₅ : base_area = base_length * base_width) 
  (h₆ : rise = cube_volume / base_area) : 
  rise = 2.5 := 
by 
  -- add proof here 
  sorry

end rise_in_water_level_l541_541816


namespace translate_line_upwards_l541_541769

theorem translate_line_upwards {x y : ℝ} (h : y = -2 * x + 1) :
  y = -2 * x + 3 := by
  sorry

end translate_line_upwards_l541_541769


namespace factorization_of_polynomial_l541_541900

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l541_541900


namespace total_students_taught_l541_541840

theorem total_students_taught (students_per_first_year : ℕ) 
  (students_per_year : ℕ) 
  (years_remaining : ℕ) 
  (total_years : ℕ) :
  students_per_first_year = 40 → 
  students_per_year = 50 →
  years_remaining = 9 →
  total_years = 10 →
  students_per_first_year + students_per_year * years_remaining = 490 := 
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃]
  norm_num
  rw h₄
  norm_num

end total_students_taught_l541_541840


namespace impossibility_of_root_condition_l541_541796

open Int

noncomputable def table_numbers := (51 : ℕ, 150 : ℕ)
def adjacent_pairs (a b : ℕ) : Prop := ((a + 1 = b) ∨ (a = b + 1))

theorem impossibility_of_root_condition :
  ∀ (a b : ℕ), (a ≥ 51 ∧ a ≤ 150) → (b ≥ 51 ∧ b ≤ 150) → 
  adjacent_pairs a b → 
  ¬(∃ x1 x2 : ℤ, (x1 + x2 = b ∧ x1 * x2 = a) ∨ (x1 + x2 = a ∧ x1 * x2 = b)) := 
by 
  sorry

end impossibility_of_root_condition_l541_541796


namespace daisies_left_l541_541662

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end daisies_left_l541_541662


namespace rigid_motion_pattern_l541_541873

-- Define the types of transformations
inductive Transformation
| rotation : ℝ → Transformation -- rotation by an angle
| translation : ℝ → Transformation -- translation by a distance
| reflection_across_m : Transformation -- reflection across line m
| reflection_perpendicular_to_m : ℝ → Transformation -- reflective across line perpendicular to m at a point

-- Define the problem statement conditions
def pattern_alternates (line_m : ℝ → ℝ) : Prop := sorry -- This should define the alternating pattern of equilateral triangles and squares along line m

-- Problem statement in Lean
theorem rigid_motion_pattern (line_m : ℝ → ℝ) (p : Transformation → Prop)
    (h1 : p (Transformation.rotation 180)) -- 180-degree rotation is a valid transformation for the pattern
    (h2 : ∀ d, p (Transformation.translation d)) -- any translation by pattern unit length is a valid transformation
    (h3 : p Transformation.reflection_across_m) -- reflection across line m is a valid transformation
    (h4 : ∀ x, p (Transformation.reflection_perpendicular_to_m x)) -- reflection across any perpendicular line is a valid transformation
    : ∃ t : Finset Transformation, t.card = 4 ∧ ∀ t_val, t_val ∈ t → p t_val ∧ t_val ≠ Transformation.rotation 0 := 
sorry

end rigid_motion_pattern_l541_541873


namespace ordering_of_values_l541_541682

def f (m : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (1 - 2 * m) * x - 3 * m else log m x

theorem ordering_of_values {m : ℝ} (h : m ∈ set.Ico (1/5 : ℝ) (1/2 : ℝ)) :
  let a := f m (-3/2)
  let b := f m 1
  let c := f m 2
  in a < c ∧ c < b := 
sorry

end ordering_of_values_l541_541682


namespace mushrooms_left_l541_541093

-- Define the initial amount of mushrooms.
def init_mushrooms : ℕ := 15

-- Define the amount of mushrooms eaten.
def eaten_mushrooms : ℕ := 8

-- Define the resulting amount of mushrooms.
def remaining_mushrooms (init : ℕ) (eaten : ℕ) : ℕ := init - eaten

-- The proof statement
theorem mushrooms_left : remaining_mushrooms init_mushrooms eaten_mushrooms = 7 :=
by
    sorry

end mushrooms_left_l541_541093


namespace simple_interest_sum_l541_541026

theorem simple_interest_sum
  (R : ℝ)  -- Original interest rate in percent
  (P : ℝ)  -- Principal amount
  (h : (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120) :
  P = 1000 :=
sorry

end simple_interest_sum_l541_541026


namespace vector_EF_expression_l541_541634

-- Define vectors and points in the quadrilateral
variables {A B C D E F : Type*} 
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]
variables (a b : A)

-- Assumptions based on problem conditions
variable (hAB : B = A + a)
variable (hCD : D = C + b)
variable (hE : E = (A + C) / 2)
variable (hF : F = (B + D) / 2)

-- The theorem to be proven
theorem vector_EF_expression :
  F - E = (a + b) / 2 :=
sorry

end vector_EF_expression_l541_541634


namespace construct_sqrt7_from_7sqrt3_l541_541924

-- Definition of the given segment length
def segment_length : ℝ := 7 * Real.sqrt 3

-- The theorem to show it’s possible to construct a segment of length sqrt(7)
theorem construct_sqrt7_from_7sqrt3 (s : ℝ) (h : s = 7 * Real.sqrt 3) : ∃ t : ℝ, t = Real.sqrt 7 :=
by 
  -- We will have to elaborate the proof, using the construction steps outlined.
  -- But currently just as the aim include the hurdle statement.
  sorry

end construct_sqrt7_from_7sqrt3_l541_541924


namespace polynomial_root_exists_l541_541902

theorem polynomial_root_exists (x : ℝ) (hx : x = √3 + √5) :
  (x^4 - 16 * x^2 + 4) = 0 :=
sorry

end polynomial_root_exists_l541_541902


namespace product_of_nonreal_roots_l541_541906

theorem product_of_nonreal_roots :
  let f := (λ x : ℂ, x^6 - 6 * x^5 + 15 * x^4 - 20 * x^3 + 15 * x^2 - 6 * x)
  (∀ x : ℂ, f x = 4060) → 
  let roots := {x // f(x) = 4060} 
  let nonreal_roots := {x // f(x) = 4060 ∧ ∃ z : ℂ, z = x ∧ x.im ≠ 0} 
  (∏ x in nonreal_roots, x) = 1 + (complex.sqrt 8122) := 
sorry

end product_of_nonreal_roots_l541_541906


namespace B_inter_A_complement_eq_one_l541_541588

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3}
def A_complement : Set ℕ := I \ A

theorem B_inter_A_complement_eq_one : B ∩ A_complement = {1} := by
  sorry

end B_inter_A_complement_eq_one_l541_541588


namespace find_table_height_l541_541222

theorem find_table_height (b r g h : ℝ) (h1 : h + b - g = 111) (h2 : h + r - b = 80) (h3 : h + g - r = 82) : h = 91 := 
by
  sorry

end find_table_height_l541_541222


namespace roberto_salary_increase_l541_541235

noncomputable def starting_salary : ℕ := 80000
noncomputable def raise_percent : ℚ := 0.20
noncomputable def current_salary : ℕ := 134400

theorem roberto_salary_increase :
  let raise := (raise_percent * starting_salary) in
  let previous_salary := starting_salary + raise in
  (current_salary - previous_salary) / previous_salary * 100 = 40 :=
by sorry

end roberto_salary_increase_l541_541235


namespace find_point_P_on_parabola_l541_541019

noncomputable def parabola_vertex : ℝ × ℝ := (0, 0)
noncomputable def parabola_focus : ℝ × ℝ := (0, 2)
noncomputable def point_P_in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
noncomputable def distance_PF (x y : ℝ) : ℝ := real.sqrt (x^2 + (y - 2)^2)

theorem find_point_P_on_parabola :
  ∃ (x y : ℝ), point_P_in_first_quadrant x y ∧ distance_PF x y = 90 ∧ (y + 2 = 90 ∧ x = real.sqrt 704) :=
by 
  sorry

end find_point_P_on_parabola_l541_541019


namespace fraction_of_number_l541_541320

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541320


namespace remainder_is_cx_plus_d_l541_541202

-- Given a polynomial Q, assume the following conditions
variables {Q : ℕ → ℚ}

-- Conditions
axiom condition1 : Q 15 = 12
axiom condition2 : Q 10 = 4

theorem remainder_is_cx_plus_d : 
  ∃ c d, (c = 8 / 5) ∧ (d = -12) ∧ 
          ∀ x, Q x % ((x - 10) * (x - 15)) = c * x + d :=
by
  sorry

end remainder_is_cx_plus_d_l541_541202


namespace find_AE_length_l541_541000

-- Definitions and assumptions based on the provided conditions
variables (A B C D E : Point)
variable (h : ℝ) -- common height, perpendicular distance from A and B to CD
variable (a b : ℝ)
variable (AE CD BC AB : ℝ)

axiom isosceles_trapezium (A B C D : Point) : is_isosceles_trapezium A B C D -- AB || CD, and AD = BC
axiom length_AD_eq_BC : distance AD = distance BC
axiom length_AB : distance AB = 5
axiom length_CD : distance CD = 10
axiom perpendicular_AE_EC : is_perpendicular AE EC
axiom length_BC_eq_EC : distance BC = distance EC

-- The main statement to be proved
theorem find_AE_length : ∃ (a b : ℤ), (b ≠ 0 ∧ ¬∃ (k : ℤ), k^2 ∣ b ∧ k ≠ 1) ∧ (AE = a * (sqrt b)) ∧ a + b = 6 :=
by {
  sorry
}

end find_AE_length_l541_541000


namespace sum_of_roots_l541_541280

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l541_541280


namespace tickets_spent_on_beanie_l541_541042

-- Define the initial conditions
def initial_tickets : ℕ := 25
def additional_tickets : ℕ := 15
def tickets_left : ℕ := 18

-- Define the total tickets
def total_tickets := initial_tickets + additional_tickets

-- Define what we're proving: the number of tickets spent on the beanie
theorem tickets_spent_on_beanie : initial_tickets + additional_tickets - tickets_left = 22 :=
by 
  -- Provide proof steps here
  sorry

end tickets_spent_on_beanie_l541_541042


namespace surface_area_comparison_l541_541927

theorem surface_area_comparison (a R : ℝ) (h_eq_volumes : (4 / 3) * Real.pi * R^3 = a^3) :
  6 * a^2 > 4 * Real.pi * R^2 :=
by
  sorry

end surface_area_comparison_l541_541927


namespace kylie_daisies_l541_541660

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l541_541660


namespace extreme_point_a_zero_l541_541604

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l541_541604


namespace critical_points_neither_necessary_nor_sufficient_l541_541978

theorem critical_points_neither_necessary_nor_sufficient (f g : ℝ → ℝ) :
  ¬(∀ F : ℝ → ℝ, F = λ x, f x + g x → (∃ c, (∃ cf, differentiable ℝ f) ∧ (∃ cg, differentiable ℝ g) ∧ (cf c = 0 ∨ cg c = 0) ↔ ∃ c', differentiable ℝ (λ x, f x + g x) ∧ (has_deriv_at (λ x, f x + g x) 0 c'))) :=
sorry

end critical_points_neither_necessary_nor_sufficient_l541_541978


namespace area_of_square_WXYZ_l541_541635

-- Definitions based on the conditions
variable {W X Y Z M N O : Type}
variable [MetricSpace W XYZ]
variable [MetricSpace X XYZ]
variable [MetricSpace Y XYZ]
variable [MetricSpace Z XYZ]
variable [MetricSpace M XYZ]
variable [MetricSpace N XYZ]
variable [MetricSpace O XYZ]
variable (square_WXYZ : Square W X Y Z)
variable (point_M_on_WZ : OnSegment W Z M)
variable (point_N_on_WX : OnSegment W X N)
variable (segments_intersect_at_O : ∃ (O : XYZ), RightAngle (YM O) (ZN O) ∧ Dist Y O = 8 ∧ Dist M O = 9)

-- Problem statement
theorem area_of_square_WXYZ :
  (∃ (s : ℝ), s * s = 137) → Area square_WXYZ = 137 :=
by
  sorry

end area_of_square_WXYZ_l541_541635


namespace largest_power_of_18_dividing_factorial_30_l541_541488

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541488


namespace inequality_holds_for_all_x_iff_range_m_l541_541608

theorem inequality_holds_for_all_x_iff_range_m (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Ioc (-10) 2 := by
  sorry

end inequality_holds_for_all_x_iff_range_m_l541_541608


namespace find_chemistry_marks_l541_541877

theorem find_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℤ)
    (average_marks total_subjects : ℤ)
    (h1 : marks_english = 36)
    (h2 : marks_math = 35)
    (h3 : marks_physics = 42)
    (h4 : marks_biology = 55)
    (h5 : average_marks = 45)
    (h6 : total_subjects = 5) :
    (225 - (marks_english + marks_math + marks_physics + marks_biology)) = 57 :=
by
  sorry

end find_chemistry_marks_l541_541877


namespace bear_cycle_min_k_l541_541005

/-- In a 100x100 grid with a bear cycle, 
    find the minimum k such that removing any row or column results in the maximum
    length of the remaining paths being at most k -/
theorem bear_cycle_min_k :
  ∀ (grid : Fin 100 × Fin 100) (bc : Set (Fin 100 × Fin 100)),
  bc.Nonempty →
  (∀ p : Fin 100 × Fin 100, p ∈ bc) →
  (∃ k : ℕ, k = 5000 ∧
   ∀ (row : Fin 100), ∀ (col : Fin 100), ∀ (path : List (Fin 100 × Fin 100)),
   ((∀ (p : Fin 100 × Fin 100), p ∈ path → p ∈ bc) ∧
   (∀ (i j : Fin 100), (i ≠ j) → path[i] ≠ path[j]) ∧
   (path.head = path.last)) →
   (∑ p in path, (if p.1 = row ∨ p.2 = col then 0 else 1)) ≤ k) :=
sorry

end bear_cycle_min_k_l541_541005


namespace arc_length_of_octagon_side_l541_541381

-- Define the side length of the octagon
def side_length : ℝ := 4

-- Define the number of sides for a regular octagon
def num_sides : ℕ := 8

-- Define the central angle subtended by one side of the octagon in radians
def central_angle : ℝ := 2 * Real.pi / num_sides

-- Define the radius of the circumscribed circle
def radius : ℝ := side_length / (2 * Real.sin (central_angle / 2))

-- Define the circumference of the circle
def circumference : ℝ := 2 * Real.pi * radius

-- Define the fraction of the total circumference represented by the arc
def arc_fraction : ℝ := central_angle / (2 * Real.pi)

-- Calculate the length of the arc intercepted by one side of the octagon
def arc_length : ℝ := arc_fraction * circumference

-- State the proof problem
theorem arc_length_of_octagon_side :
  arc_length = (Real.sqrt 2 * Real.pi) / 2 :=
by
  sorry

end arc_length_of_octagon_side_l541_541381


namespace ratio_bronze_to_silver_l541_541195

noncomputable def num_watches.total : ℕ := 88
noncomputable def num_watches.silver : ℕ := 20
noncomputable def num_watches.gold : ℕ := 9
noncomputable def num_watches.bronze : ℕ := num_watches.total - num_watches.silver - num_watches.gold

theorem ratio_bronze_to_silver :
  (num_watches.bronze : ℚ) / num_watches.silver = 59 / 20 :=
by
  -- We can convert the bronze and silver watch counts to rational numbers
  have bronze_count := num_watches.bronze
  have silver_count := num_watches.silver
  -- Prove that bronze_count = 59
  have bronze_eq : bronze_count = 59 :=
    by linarith [num_watches.total, num_watches.silver, num_watches.gold]
  -- Prove that silver_count = 20
  have silver_eq : silver_count = 20 := rfl
  -- Substitute and finish the proof
  rw [bronze_eq, silver_eq]
  norm_num
  sorry

end ratio_bronze_to_silver_l541_541195


namespace area_after_shortening_other_side_l541_541721

-- Define initial dimensions of the index card
def initial_length := 5
def initial_width := 7
def initial_area := initial_length * initial_width

-- Define the area condition when one side is shortened by 2 inches
def shortened_side_length := initial_length - 2
def new_area_after_shortening_one_side := 21

-- Definition of the problem condition that results in 21 square inches area
def condition := 
  (shortened_side_length * initial_width = new_area_after_shortening_one_side)

-- Final statement
theorem area_after_shortening_other_side :
  condition →
  (initial_length * (initial_width - 2) = 25) :=
by
  intro h
  sorry

end area_after_shortening_other_side_l541_541721


namespace miriam_pushups_on_monday_l541_541695

theorem miriam_pushups_on_monday :
  ∃ M : ℕ, 
    (let T := 7 in
     let W := 2 * T in
     let Th := (M + T + W) / 2 in
     M + T + W + Th = 39) →
  M = 5 :=
by
  sorry

end miriam_pushups_on_monday_l541_541695


namespace snake_length_l541_541828

variable (L : ℝ)

-- Condition: The head of the snake is one-tenth of its length.
def head_length (L : ℝ) : ℝ := L / 10

-- Condition: The length of the rest of its body minus the head is 9 feet.
def body_length (L : ℝ) : ℝ := L - head_length L

theorem snake_length (h : body_length L = 9) : L = 10 :=
by
  sorry

end snake_length_l541_541828


namespace factorization_of_polynomial_l541_541897

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l541_541897


namespace function_is_periodic_l541_541015

noncomputable def f : ℝ → ℝ := sorry

axiom functional_condition : ∀ x : ℝ, f(x + 3) = f(x)

theorem function_is_periodic : (∀ x : ℝ, f(x + 3) = f(x)) → ∃ p > 0, ∀ x : ℝ, f(x + p) = f(x) :=
by
  intro h
  use 3
  split
  · linarith
  · exact h

end function_is_periodic_l541_541015


namespace cartesian_equation_and_min_distance_l541_541887

/-- 
Conditions:
1. The parametric equation of curve C1 is given by:
     x = t,
     y = m - t,
     where t is the parameter, m ∈ ℝ.
2. The polar equation of curve C2 is given by:
     ρ^2 = 3 / (1 + 2 * sin(θ)^2),
     ρ > 0,
     θ ∈ [0, π].
-/
/--
Theorem: Given the conditions, 
1. The Cartesian coordinates equations of C1 and C2 are:
   C1: x + y - m = 0,
   C2: x^2 / 3 + y^2 = 1 with y ≥ 0.
2. If the minimum distance between a point P on C1 and a point Q on C2 is 2√2,
   then the value of m is either -4 - √3 or  6.
-/
theorem cartesian_equation_and_min_distance
  (m : ℝ)
  (C1 : ℝ × ℝ → Prop := λ t, (t.1 = t.2) ∧ (t.2 = m - t.2))
  (C2 : ℝ × ℝ → Prop := λ p, p.1 * p.1 / 3 + p.2 * p.2 = 1 ∧ p.2 ≥ 0)
  (d : ℝ := 2 * real.sqrt 2) :
  (∀ t, C1 (t, m - t) → (t + (m - t) = m)) ∧
  (∀ (ρ θ : ℝ), θ ∈ set.Icc 0 π ∧ ρ = (3 / (1 + 2 * real.sin θ ^ 2)).sqrt → 
     ((ρ * real.cos θ) ^ 2 / 3 + (ρ * real.sin θ) ^ 2 = 1 ∧ ρ * real.sin θ ≥ 0)) ∧
  ((∀ α : ℝ, α ∈ set.Icc (π / 3) (4 * π / 3) →
     (abs (2 * real.sin (α + π / 3) - m) / real.sqrt 2 = 2 * real.sqrt 2 → 
      (m = 6 ∨ m = -4 - real.sqrt 3)) ))
:= 
begin
  sorry
end

end cartesian_equation_and_min_distance_l541_541887


namespace total_food_for_guinea_pigs_l541_541720

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l541_541720


namespace Bridget_skittles_after_giving_l541_541857

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l541_541857


namespace angle_between_unit_vectors_l541_541211

variables {R : Type*} [field R]
variables {s t : R} (hs : s ≠ 0) (ht : t ≠ 0)
variables {a b : R^3} (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)
variables (h : ‖s • a + t • b‖ = ‖t • a - s • b‖)

theorem angle_between_unit_vectors (s t : R) (hs : s ≠ 0) (ht : t ≠ 0) 
                                (a b : R^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)
                                (h : ‖s • a + t • b‖ = ‖t • a - s • b‖) : 
                                (a ⬝ b) = 0 :=
by {
  sorry,
}

end angle_between_unit_vectors_l541_541211


namespace bus_fare_optimal_change_in_passengers_l541_541795

noncomputable def demand (p : ℝ) : ℝ := 4200 - 100 * p
def train_fare : ℝ := 4
def train_capacity : ℝ := 800
noncomputable def bus_cost (y : ℝ) : ℝ := 10 * y + 225

-- Part (a)
theorem bus_fare_optimal : 
  let q := demand,
      π_bus := λ (p : ℝ), p * (3400 - 100 * p) - 10 * (3400 - 100 * p) - 225 in 
  q 4 ≤ 800 ∧ (∀ p, π_bus p ≤  π_bus 22) → p = 22 := 
by sorry

-- Part (b)
theorem change_in_passengers : 
  let q := demand,
      π_bus := λ (p : ℝ), p * (4200 - 100 * p) - 10 * (4200 - 100 * p) - 225,
      initial_passengers := (3400 - 100 * 22) + 800,
      final_passengers := 4200 - 100 * 26 in
  π_bus 26 ≥ π_bus (train_fare) ∧
  final_passengers = 1600 ∧
  initial_passengers - final_passengers = 400 := 
by sorry

end bus_fare_optimal_change_in_passengers_l541_541795


namespace expression_value_l541_541101

theorem expression_value (a b : ℝ) (h₁ : b - a = -6) (h₂ : ab = 7) : a^2b - ab^2 = -42 :=
by
  sorry

end expression_value_l541_541101


namespace janet_time_per_post_l541_541656

/-- Janet gets paid $0.25 per post she checks. She earns $90 per hour. 
    Prove that it takes her 10 seconds to check a post. -/
theorem janet_time_per_post
  (payment_per_post : ℕ → ℝ)
  (hourly_pay : ℝ)
  (posts_checked_hourly : ℕ)
  (secs_per_post : ℝ) :
  payment_per_post 1 = 0.25 →
  hourly_pay = 90 →
  hourly_pay = payment_per_post (posts_checked_hourly) →
  secs_per_post = 10 :=
sorry

end janet_time_per_post_l541_541656


namespace triangle_parallel_bisectors_l541_541952

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541952


namespace difference_of_squares_expression_l541_541344

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l541_541344


namespace fifth_eq_l541_541221

theorem fifth_eq :
  (1 = 1) ∧
  (2 + 3 + 4 = 9) ∧
  (3 + 4 + 5 + 6 + 7 = 25) ∧
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81 :=
by
  intros
  sorry

end fifth_eq_l541_541221


namespace GM_perp_AF_l541_541971

-- Definitions of the terms and conditions based on the problem description
variables 
  (A B C O H D E F G M : Type*)
  [acute_triangle : Triangle A B C]
  [circumcenter O : Circumcenter A B C]
  [orthocenter H : Orthocenter A B C]
  [bisector D : AngleBisector A B C]
  [reflection1 E : Reflection D BC]
  [reflection2 F : Reflection D O]
  [intersection G : Intersection (LineThrough A E) (LineThrough F H)]
  [midpoint M : Midpoint B C]

-- Statement to be proved
theorem GM_perp_AF (acute_triangle : Triangle A B C)
                   (circumcenter_O : Circumcenter A B C)
                   (orthocenter_H : Orthocenter A B C)
                   (angle_bisector_D : AngleBisector A B C)
                   (reflection_E : Reflection angle_bisector_D BC)
                   (reflection_F : Reflection angle_bisector_D circumcenter_O)
                   (intersection_G : Intersection (LineThrough A reflection_E) (LineThrough reflection_F orthocenter_H))
                   (midpoint_M : Midpoint B C) :
  Perpendicular GM AF :=
begin
  sorry
end

end GM_perp_AF_l541_541971


namespace non_congruent_squares_on_5x5_grid_l541_541138

def is_lattice_point (x y : ℕ) : Prop := x ≤ 4 ∧ y ≤ 4

def is_square {a b c d : (ℕ × ℕ)} : Prop :=
((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2) ∧ 
((c.1 - b.1)^2 + (c.2 - b.2)^2 = (a.1 - d.1)^2 + (a.2 - d.2)^2)

def number_of_non_congruent_squares : ℕ :=
  4 + -- Standard squares: 1x1, 2x2, 3x3, 4x4
  2 + -- Diagonal squares: with sides √2 and 2√2
  2   -- Diagonal sides of 1x2 and 1x3 rectangles

theorem non_congruent_squares_on_5x5_grid :
  number_of_non_congruent_squares = 8 :=
by
  -- proof goes here
  sorry

end non_congruent_squares_on_5x5_grid_l541_541138


namespace price_difference_l541_541791

variable (P : ℝ) (hP : P > 0)

theorem price_difference (P : ℝ) (hP : P > 0) : 
  let newPrice := P * 1.2
  let discountedPrice := newPrice * 0.8
  newPrice - discountedPrice = P * 0.24 :=
by
  have newPrice := P * 1.2
  have discountedPrice := newPrice * 0.8
  show newPrice - discountedPrice = P * 0.24 from sorry

end price_difference_l541_541791


namespace parallel_lines_triangle_l541_541946

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541946


namespace find_2a_plus_b_l541_541672

open Real

theorem find_2a_plus_b (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
    (h1 : 4 * (cos a)^3 - 3 * (cos b)^3 = 2) 
    (h2 : 4 * cos (2 * a) + 3 * cos (2 * b) = 1) : 
    2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l541_541672


namespace fib_divisibility_l541_541251

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem fib_divisibility (m n : ℕ) (hm : 1 ≤ m) (hn : 1 < n) : 
  (fib (m * n - 1) - fib (n - 1) ^ m) % fib n ^ 2 = 0 :=
sorry

end fib_divisibility_l541_541251


namespace total_dollars_l541_541194

theorem total_dollars (john emma lucas : ℝ) 
  (h_john : john = 4 / 5) 
  (h_emma : emma = 2 / 5) 
  (h_lucas : lucas = 1 / 2) : 
  john + emma + lucas = 1.7 := by
  sorry

end total_dollars_l541_541194


namespace original_rectangle_area_is_56_l541_541410

-- Conditions
def original_rectangle_perimeter := 30 -- cm
def smaller_rectangle_perimeter := 16 -- cm
def side_length_square := (original_rectangle_perimeter - smaller_rectangle_perimeter) / 2 -- Using the reduction logic

-- Computing the length and width of the original rectangle.
def width_original_rectangle := side_length_square
def length_original_rectangle := smaller_rectangle_perimeter / 2

-- The goal is to prove that the area of the original rectangle is 56 cm^2.

theorem original_rectangle_area_is_56:
  (length_original_rectangle - width_original_rectangle + width_original_rectangle) = 8 -- finding the length
  ∧ (length_original_rectangle * width_original_rectangle) = 56 := by
  sorry

end original_rectangle_area_is_56_l541_541410


namespace pascals_triangle_48th_number_l541_541329

theorem pascals_triangle_48th_number :
  nat.choose 50 47 = 19600 :=
by
  sorry

end pascals_triangle_48th_number_l541_541329


namespace fraction_inequality_l541_541977

-- Given the conditions
variables {c x y : ℝ} (h1 : c > x) (h2 : x > y) (h3 : y > 0)

-- Prove that \frac{x}{c-x} > \frac{y}{c-y}
theorem fraction_inequality (h4 : c > 0) : (x / (c - x)) > (y / (c - y)) :=
by {
  sorry  -- Proof to be completed
}

end fraction_inequality_l541_541977


namespace XY_parallel_X_l541_541969

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541969


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541509

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541509


namespace sequence_values_infinite_ones_monotonic_decrease_l541_541554

-- Definitions and conditions
def sequence (a0 a1 : ℕ) : ℕ → ℕ
| 0 := a0
| 1 := a1
| (n + 2) := if sequence (n + 1) / sequence n > 1 then sequence (n + 1) / sequence n else sequence n / sequence (n + 1)

-- (I) Prove a_4 = 2 and a_5 = 1
theorem sequence_values (a0 a1 : ℕ) (h0 : a0 = 2) (h1 : a1 = 1) : 
  sequence a0 a1 3 = 2 ∧ sequence a0 a1 4 = 1 := by
  sorry

-- (II) If there is a term a_k = 1, then there are infinitely many terms equal to 1
theorem infinite_ones (a0 a1 : ℂ) (k : ℕ) (hk : sequence a0 a1 k = 1) : ∃ (n:ℕ) (m : ℕ → Prop), ∀ m ≥ n, ∃ l, m ≤ l ∧ sequence a0 a1 l = 1 := by
  sorry

-- (III) Prove the sequence {b_n} is monotonically decreasing
def b_sequence (a0 a1 : ℕ) (n : ℕ) : ℕ :=
  max (sequence a0 a1 (2 * n - 1)) (sequence a0 a1 (2 * n))

theorem monotonic_decrease (a0 a1 : ℕ) (h0 : a0 > 1) (hn : ∀ k, sequence a0 a1 k ≠ 1) : ∀ n, b_sequence a0 a1 n > b_sequence a0 a1 (n + 1) := by
  sorry

end sequence_values_infinite_ones_monotonic_decrease_l541_541554


namespace quadratic_roots_sum_cubes_l541_541880

theorem quadratic_roots_sum_cubes (k : ℚ) (a b : ℚ) 
  (h1 : 4 * a^2 + 5 * a + k = 0) 
  (h2 : 4 * b^2 + 5 * b + k = 0) 
  (h3 : a^3 + b^3 = a + b) :
  k = 9 / 4 :=
by {
  -- Lean code requires the proof, here we use sorry to skip it
  sorry
}

end quadratic_roots_sum_cubes_l541_541880


namespace cross_section_area_BCD_l541_541550

-- Define necessary structures and conditions
structure Prism :=
  (height : ℝ)
  (base_edge_length : ℝ)
  (center_top : ℝ)
  (plane_perpendicular_to_AP : Prop)
  (intersection_point_D : ℝ)

-- Given conditions
def ABC_A1B1C1 := Prism.mk 2 1 (sqrt 3 / 3) True 1

-- Prove the area of the cross-section is √13/8
theorem cross_section_area_BCD (prism : Prism) : 
  prism.height = 2 → 
  prism.base_edge_length = 1 →
  prism.center_top = sqrt 3 / 3 →
  prism.plane_perpendicular_to_AP →
  prism.intersection_point_D = 1 →
  (1 / 2) * 1 * (sqrt 13 / 4) = sqrt 13 / 8 :=
by
  intros,
  sorry

end cross_section_area_BCD_l541_541550


namespace teacher_spends_31_dollars_l541_541389

-- Define the number of pens bought
def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5

-- Define the cost per pen
def cost_black_pen : Float := 1.25
def cost_blue_pen : Float := 1.50
def cost_red_pen : Float := 1.75

-- Sum the total cost
def total_cost : Float :=
  num_black_pens * cost_black_pen +
  num_blue_pens * cost_blue_pen +
  num_red_pens * cost_red_pen

-- Statement of the proof problem
theorem teacher_spends_31_dollars :
  total_cost = 31 := by
  sorry

end teacher_spends_31_dollars_l541_541389


namespace triangle_parallel_bisectors_l541_541951

variables {Point : Type} [EuclideanGeometry Point]

/-- Given a triangle DEF, let a circle passing through vertices E and F intersect sides DE and DF at points X and Y, respectively.
The angle bisector of ∠DEY intersects DF at point Y', and the angle bisector of ∠DFX intersects DE at point X'.
Prove that XY is parallel to X'Y' --/
theorem triangle_parallel_bisectors 
  {D E F X Y X' Y' : Point} 
  (hCircleThroughE_F : CircleThrough E F) 
  (hX_on_DE : X ∈ (Segment D E))
  (hY_on_DF : Y ∈ (Segment D F))
  (hX_Y_on_circle : X ∈ hCircleThroughE_F ∧ Y ∈ hCircleThroughE_F)
  (hY'_angle_bisector : Y' ∈ (segment D F) ∧ angle_bisector D (line_through E Y') = (line_through D F))
  (hX'_angle_bisector : X' ∈ (segment D E) ∧ angle_bisector D (line_through F X') = (line_through D E)) :
  parallel (line_through X Y) (line_through X' Y') :=
sorry

end triangle_parallel_bisectors_l541_541951


namespace find_f_prime_2_l541_541994

noncomputable def f (x : ℝ) (f'1 : ℝ) : ℝ := x^2 * f'1 - 3 * x
noncomputable def f' (x : ℝ) (f'1 : ℝ) : ℝ := 2 * x * f'1 - 3

theorem find_f_prime_2 (f'1 : ℝ) (h : f'1 = 3) : f' 2 f'1 = 9 :=
by
  rw [f']
  rw h
  norm_num

end find_f_prime_2_l541_541994


namespace num_subsets_containing_5_and_7_l541_541362

theorem num_subsets_containing_5_and_7 :
  let s := {1, 2, 3, 4, 5, 6, 7}
  in (∃ A : set ℕ, A ⊆ s ∧ 5 ∈ A ∧ 7 ∈ A ∧ finset.card A = 32) :=
sorry

end num_subsets_containing_5_and_7_l541_541362


namespace pollution_index_minimum_l541_541395

noncomputable def pollution_index (k a b : ℝ) (x : ℝ) : ℝ :=
  k * (a / (x ^ 2) + b / ((18 - x) ^ 2))

theorem pollution_index_minimum (k : ℝ) (h₀ : 0 < k) (h₁ : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 18) :
  ∀ a b x : ℝ, a = 1 → x = 6 → pollution_index k a b x = pollution_index k 1 8 6 :=
by
  intros a b x ha hx
  rw [ha, hx, pollution_index]
  sorry

end pollution_index_minimum_l541_541395


namespace B_pow_2023_eq_B_l541_541196

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1/2, 0, (Real.sqrt 3) / 2],
    ![0, 1, 0],
    ![-(Real.sqrt 3) / 2, 0, 1 / 2]
  ]

theorem B_pow_2023_eq_B : B ^ 2023 = B := by
  sorry

end B_pow_2023_eq_B_l541_541196


namespace avg_price_of_towels_l541_541393

def towlesScenario (t1 t2 t3 : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  (t1 * price1 + t2 * price2 + t3 * price3) / (t1 + t2 + t3)

theorem avg_price_of_towels :
  towlesScenario 3 5 2 100 150 500 = 205 := by
  sorry

end avg_price_of_towels_l541_541393


namespace XY_parallel_X_l541_541932

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541932


namespace inverse_double_application_l541_541734

-- Given function definition
def f (x : ℝ) : ℝ := 3 * x - 2

-- Inverse function definition based on the given condition
def f_inv (x : ℝ) : ℝ := (x + 2) / 3

-- Problem statement: prove that f_inv (f_inv (14)) = 22 / 9
theorem inverse_double_application:
  f_inv (f_inv 14) = 22 / 9 :=
sorry

end inverse_double_application_l541_541734


namespace find_x_find_a_l541_541146

-- Definitions based on conditions
def inversely_proportional (p q : ℕ) (k : ℕ) := p * q = k

-- Given conditions for (x, y)
def x1 : ℕ := 36
def y1 : ℕ := 4
def k1 : ℕ := x1 * y1 -- or 144
def y2 : ℕ := 9

-- Given conditions for (a, b)
def a1 : ℕ := 50
def b1 : ℕ := 5
def k2 : ℕ := a1 * b1 -- or 250
def b2 : ℕ := 10

-- Proof statements
theorem find_x (x : ℕ) : inversely_proportional x y2 k1 → x = 16 := by
  sorry

theorem find_a (a : ℕ) : inversely_proportional a b2 k2 → a = 25 := by
  sorry

end find_x_find_a_l541_541146


namespace largest_n_18n_divides_30_factorial_l541_541502

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541502


namespace angle_600_in_third_quadrant_l541_541803

def angle_quadrant (θ : ℝ) : ℕ :=
  if θ % 360 < 90 then 1
  else if θ % 360 < 180 then 2
  else if θ % 360 < 270 then 3
  else 4

theorem angle_600_in_third_quadrant : angle_quadrant 600 = 3 :=
  sorry

end angle_600_in_third_quadrant_l541_541803


namespace Adam_teaches_students_l541_541838

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end Adam_teaches_students_l541_541838


namespace largest_n_dividing_30_factorial_l541_541483

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541483


namespace bogan_second_feeding_bogan_feeding_20_10_10_l541_541050

theorem bogan_second_feeding :
  (total_maggots first_feeding second_feeding : ℕ) →
  total_maggots = 20 →
  first_feeding = 10 →
  second_feeding = total_maggots - first_feeding →
  second_feeding = 10 :=
by
  intros total_maggots first_feeding second_feeding h_total h_first h_second
  rw [h_total, h_first] at h_second
  exact h_second

# To assert the main theorem correctly and avoid any local assumptions
theorem bogan_feeding_20_10_10 :
  (total_maggots second_feeding first_feeding : ℕ) →
  total_maggots = 20 →
  first_feeding = 10 →
  second_feeding = total_maggots - first_feeding →
  second_feeding = 10 :=
by
  intros total_maggots first_feeding second_feeding h_total h_first h_second
  exact bogan_second_feeding total_maggots first_feeding second_feeding h_total h_first h_second

end bogan_second_feeding_bogan_feeding_20_10_10_l541_541050


namespace hyperbola_eccentricity_l541_541103

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : 2 = b / a) :
  let e := Real.sqrt (1 + (b / a) ^ 2) in
  e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l541_541103


namespace XY_parallel_X_l541_541931

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541931


namespace weight_on_earth_l541_541283

theorem weight_on_earth 
  (W_moon_1 : ℝ) (W_moon_2 : ℝ) (W_earth_2 : ℝ) :
  (W_moon_1 * W_earth_2) / W_moon_2 = 132.43 → 
  W_moon_2 = 27.2 → 
  W_earth_2 = 136 → 
  W_moon_1 = 26.6 → 
  ∃ W_earth_1, W_earth_1 = 132.43 :=
by
  intro h1 h2 h3 h4
  use (W_moon_1 * W_earth_2) / W_moon_2
  rw [h1];
  exact rfl
  sorry

end weight_on_earth_l541_541283


namespace largest_power_of_18_dividing_factorial_30_l541_541490

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541490


namespace fraction_of_number_l541_541319

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541319


namespace quiz_scores_l541_541157

noncomputable def achievable_score (S : ℕ) : ℕ → ℕ → Bool :=
λ c u, (4 * c + (3 / 2 * u) = S) ∧ (c + u ≤ 30 + (S - 4 * c) / (3 / 2)) ∧ (30 - c - u ≥ 0)

noncomputable def count_distinct_ways : ℕ → ℕ
| 0 := 0
| (S + 1) := if (achievable_score (S + 1) 3) then 1 + count_distinct_ways S else count_distinct_ways S

noncomputable def sum_of_achievable_scores : ℕ :=
Nat.sum (· ≤ 120) (λ S, if (achievable_score S 3) then S else 0)

theorem quiz_scores (S : ℕ) (h : (0 ≤ S ∧ S ≤ 120)) : ∃ sum, sum = sum_of_achievable_scores :=
begin
  sorry
end

end quiz_scores_l541_541157


namespace find_ratio_DE_EF_l541_541182

variable (A B C D E F: Type) 
-- Declare that D is on the line segment AB with ratio 2:3
axiom ratio_AD_DB (AD DB : ℝ) (h1 : AD / DB = 2 / 3)
-- Declare that E is on the line segment BC with ratio 2:3
axiom ratio_BE_EC (BE EC: ℝ) (h2 : BE / EC = 2 / 3)
-- State that lines DE and AC intersect at point F
axiom intersect_F (DE EF : ℝ) (h3 : ... )

theorem find_ratio_DE_EF (AD DB BE EC: ℝ) (h1 : AD / DB = 2 / 3) (h2 : BE / EC = 2 / 3) : 
  DE / EF = 1 / 2 :=
sorry

end find_ratio_DE_EF_l541_541182


namespace sin_double_theta_l541_541147

theorem sin_double_theta (θ : ℝ) (h : ∑' n : ℕ, (sin θ)^(2*n) = 3) : sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
  sorry

end sin_double_theta_l541_541147


namespace inconsistent_team_game_conditions_l541_541883

theorem inconsistent_team_game_conditions :
  ∀ (P : ℝ), (0 ≤ P ∧ P ≤ 1) →
  (75 = 100 + (75 - 100) ∧ 70 / 100 = (P * 100 + 0.5 * (75 - 100)) / 75) → false :=
by
  intro P
  intro hP_bounds
  intro h_conditions
  cases h_conditions with h_total_games h_win_rate
  sorry

end inconsistent_team_game_conditions_l541_541883


namespace min_value_expression_l541_541545

noncomputable def sinSquare (θ : ℝ) : ℝ :=
  Real.sin (θ) ^ 2

theorem min_value_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ > 0) (h₂ : θ₂ > 0) (h₃ : θ₃ > 0) (h₄ : θ₄ > 0)
  (sum_eq_pi : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (2 * sinSquare θ₁ + 1 / sinSquare θ₁) *
  (2 * sinSquare θ₂ + 1 / sinSquare θ₂) *
  (2 * sinSquare θ₃ + 1 / sinSquare θ₃) *
  (2 * sinSquare θ₄ + 1 / sinSquare θ₁) ≥ 81 := 
by
  sorry

end min_value_expression_l541_541545


namespace ptolemy_theorem_l541_541752

theorem ptolemy_theorem 
  {A B C D M : Type*}
  [has_mul A] [has_mul B] [has_mul C] [has_mul D] [has_mul M] [add_comm_monoid A] [add_comm_monoid B] [add_comm_monoid C] [add_comm_monoid D] [add_comm_monoid M] [has_le A] [has_le B] [has_le C] [has_le D] [has_le M] [has_lt A] [has_lt B] [has_lt C] [has_lt D] [has_lt M] :
  (AB: A) (BC: B) (CD: C) (DA: D)
  (AC: M) (BD: M)
  h1 : inscribed_quadrilateral ABCD
  (h2 : A, B, C, D, M: Type*) :
  AB * CD + AD * BC = AC * BD :=
sorry

end ptolemy_theorem_l541_541752


namespace perpendicular_tangents_l541_541569

theorem perpendicular_tangents (a b : ℝ):
  (∀ x y : ℝ, (ax - by - 2 = 0) ∧ (y = x^3) ∧ (x = 1) ∧ (y = 1)) → 3 * (a/b) = -1 → b = -3 * a :=
by
  sorry

end perpendicular_tangents_l541_541569


namespace experiment_success_probability_l541_541766

/-- 
There are three boxes, each containing 10 balls. 
- The first box contains 7 balls marked 'A' and 3 balls marked 'B'.
- The second box contains 5 red balls and 5 white balls.
- The third box contains 8 red balls and 2 white balls.

The experiment consists of:
1. Drawing a ball from the first box.
2. If a ball marked 'A' is drawn, drawing from the second box.
3. If a ball marked 'B' is drawn, drawing from the third box.
The experiment is successful if the second ball drawn is red.

Prove that the probability of the experiment being successful is 0.59.
-/
theorem experiment_success_probability (P : ℝ) : 
  P = 0.59 :=
sorry

end experiment_success_probability_l541_541766


namespace total_amount_after_further_period_is_850_l541_541388

-- Given conditions
def initial_sum : ℝ := 500
def amount_after_2_years : ℝ := 600
def time_2_years : ℕ := 2

-- The rate of interest can be derived (but we assume it's straightforward from the given conditions)
def rate_of_interest := (amount_after_2_years - initial_sum) / (initial_sum * time_2_years)

def additional_time : ℕ := 5
def total_time := time_2_years + additional_time

-- Total amount after further period of 5 years
def amount_after_total_time := initial_sum + (initial_sum * rate_of_interest * total_time)

theorem total_amount_after_further_period_is_850 : 
  amount_after_total_time = 850 := by
    sorry

end total_amount_after_further_period_is_850_l541_541388


namespace range_of_a_l541_541113

noncomputable def f (a : ℝ) (x : ℝ) := log x - a * (x - 1)
def g (x : ℝ) := exp x

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x1 y1, y1 = f a x1 →
    (∃ l1, l1 = λ x, (1/e) * x → l1 0 = 0) → ∀ x2 y2, y2 = g x2 →
    (∃ l2, l2 = λ x, e * x → l2 0 = 0) → ∃ x, x1 = x2 ∧ y1 = y2) →
  (frac (e-1) e < a ∧ a < frac (e^2-1) e) :=
begin
  sorry
end

end range_of_a_l541_541113


namespace angle_bisector_passes_through_fixed_point_l541_541293

-- Definitions
variables {P : Type*} [Point P]

-- Two non-intersecting wooden circles glued to a plane
variables (O1 O2 : P) (r : ℝ) (hO1O2 : O1 ≠ O2)

-- A wooden triangle with one gray side and one black side
variables (A B C : P) (gray_side black_side : triangle_side P)
variables (hgray : gray_side A B) (hblack : black_side B C)

-- Conditions for touching the circles
variables (t_gray : touches_circle gray_side O1 r)
variables (t_black : touches_circle black_side O2 r)

-- The fixed point through which the angle bisector always passes
def fixed_point (O1 O2 : P) : P := midpoint O1 O2

-- The theorem to prove
theorem angle_bisector_passes_through_fixed_point 
    (h_conditions : ∀ (A B C : P), gray_side A B ∧ black_side B C ∧ touches_circle gray_side O1 r ∧ touches_circle black_side O2 r):
    let bisector_line := angle_bisector gray_side black_side in
    bisector_line ∈ line_through (fixed_point O1 O2) :=
sorry

end angle_bisector_passes_through_fixed_point_l541_541293


namespace percent_savings_per_roll_l541_541809

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l541_541809


namespace arithmetic_sequence_problem_l541_541171

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem 
  (h : is_arithmetic_sequence a)
  (h_cond : a 2 + 2 * a 6 + a 10 = 120) :
  a 3 + a 9 = 60 :=
sorry

end arithmetic_sequence_problem_l541_541171


namespace smallest_n_for_multiple_of_7_l541_541248

theorem smallest_n_for_multiple_of_7 (x y : ℤ) (h1 : x % 7 = -1 % 7) (h2 : y % 7 = 2 % 7) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 ∧ n = 4 :=
sorry

end smallest_n_for_multiple_of_7_l541_541248


namespace find_theta_l541_541218

noncomputable theory
open real

variables {a b c d : euclidean_space ℝ (fin 3)}

-- Assumptions of given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom norm_d : ∥d∥ = 1
axiom norm_c : ∥c∥ = 3
axiom vec_triple_prod_zero : a ⨯ (a ⨯ c) + b + d = 0

-- Theorem stating the possible values of θ
theorem find_theta (θ : ℝ) :
  θ = (arccos (sqrt 7 / 3)) ∨ θ = (arccos (-sqrt 7 / 3)) :=
sorry

end find_theta_l541_541218


namespace distance_from_point_to_circle_center_l541_541332

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2 * x - 4 * y + 8

-- Center calculation as per the given equation completion and point.
def circle_center_x : ℝ := 1
def circle_center_y : ℝ := -2
def point_x : ℝ := -3
def point_y : ℝ := 4

-- Distance formula
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Proof problem
theorem distance_from_point_to_circle_center :
  distance circle_center_x circle_center_y point_x point_y = 2 * Real.sqrt 13 := 
  sorry

end distance_from_point_to_circle_center_l541_541332


namespace radius_of_spheres_in_unit_cube_l541_541653

theorem radius_of_spheres_in_unit_cube : 
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (let d := 2 * r in 
      (d + d = 1) ∧ 
      (r = 1 / 4)) :=
begin
  sorry
end

end radius_of_spheres_in_unit_cube_l541_541653


namespace tan_x_eq_sqrt3_intervals_of_monotonic_increase_l541_541918

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin (x - Real.pi / 6), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, 1)

noncomputable def f (x : ℝ) : ℝ :=
  (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proof for part 1
theorem tan_x_eq_sqrt3 (x : ℝ) (h₀ : m x = n x) : Real.tan x = Real.sqrt 3 :=
sorry

-- Proof for part 2
theorem intervals_of_monotonic_increase (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) ↔ 
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) :=
sorry

end tan_x_eq_sqrt3_intervals_of_monotonic_increase_l541_541918


namespace years_of_interest_l541_541027

noncomputable def principal : ℝ := 2600
noncomputable def interest_difference : ℝ := 78

theorem years_of_interest (R : ℝ) (N : ℝ) (h : (principal * (R + 1) * N / 100) - (principal * R * N / 100) = interest_difference) : N = 3 :=
sorry

end years_of_interest_l541_541027


namespace fraction_of_number_l541_541323

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541323


namespace wheel_distance_covered_l541_541614

/-- Given a wheel of diameter 42 cm and the number of revolutions 
is 4.003639672429482, the distance covered is approximately 528.55 cm. -/
theorem wheel_distance_covered :
  let d := 42
  let r := d / 2
  let pi := Real.pi
  let C := 2 * pi * r
  let revol := 4.003639672429482
  let dist := C * revol
  Real.approx dist 528.55 :=
by
  sorry

end wheel_distance_covered_l541_541614


namespace problem1_problem2_l541_541992

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2 + 1 / 2

theorem problem1 :
  ∀ x, (π / 2 ≤ x ∧ x < π) → monotone_on f (Set.Icc (π / 2) π) := by
  sorry

noncomputable def A : ℝ := π / 3
noncomputable def a : ℝ := sqrt 19
noncomputable def b : ℝ := 5
noncomputable def c : ℝ := 3
noncomputable def area (b c : ℝ) (A : ℝ) : ℝ := (1 / 2) * b * c * sin A

theorem problem2 :
  f A = 0 ∧ a = sqrt 19 ∧ b = 5 ∧ c = 3 → area b c A = 15 * sqrt 3 / 4 := by
  sorry

end problem1_problem2_l541_541992


namespace real_solutions_system_of_equations_l541_541904

theorem real_solutions_system_of_equations 
  (a b c : ℝ) 
  (x y z : ℝ) 
  (h1 : x^2 * y^2 + x^2 * z^2 = a * x * y * z) 
  (h2 : y^2 * z^2 + y^2 * x^2 = b * x * y * z) 
  (h3 : z^2 * x^2 + z^2 * y^2 = c * x * y * z) :
  let s := (a + b + c) / 2 in
  (x = Real.sqrt ((s-b)*(s-c)) ∨ x = -Real.sqrt ((s-b)*(s-c))) ∧
  (y = Real.sqrt ((s-a)*(s-c)) ∨ y = -Real.sqrt ((s-a)*(s-c))) ∧
  (z = Real.sqrt ((s-a)*(s-b)) ∨ z = -Real.sqrt ((s-a)*(s-b))) ↔
  (a + b > c ∧ a + c > b ∧ b + c > a ∨ -a + -b > -c ∧ -a + -c > -b ∧ -b + -c > -a) := 
sorry

end real_solutions_system_of_equations_l541_541904


namespace Bridget_Skittles_Final_l541_541858

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l541_541858


namespace range_of_m_l541_541540

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (¬(∃ u v : ℝ, u ≠ v ∧ 4*u^2 + 4*(m - 2)*u + 1 = 0 ∧ 4*v^2 + 4*(m - 2)*v + 1 = 0)) →
  m ∈ set.Ioo (-∞ : ℝ) (-2) ∪ set.Ioc 1 2 ∪ set.Ici 3 :=
sorry

end range_of_m_l541_541540


namespace max_ab_bc_cd_l541_541214

theorem max_ab_bc_cd {a b c d : ℝ} (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_sum : a + b + c + d = 200) (h_a : a = 2 * d) : 
  ab + bc + cd ≤ 14166.67 :=
sorry

end max_ab_bc_cd_l541_541214


namespace largest_n_18n_divides_30_factorial_l541_541508

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541508


namespace locus_of_P_is_parabola_l541_541224

noncomputable def locus_of_P (P : ℝ × ℝ) : Prop :=
  let A := (1 : ℝ, 0)
  let B (x y : ℝ) := (1 / Real.sqrt ((x - 1)^2 + y^2) * (x - 1), 1 / Real.sqrt ((x - 1)^2 + y^2) * y)
  let k (x y : ℝ) := 1 / Real.sqrt ((x - 1)^2 + y^2)
  P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 1 = 0

theorem locus_of_P_is_parabola :
  ∀ (P : ℝ × ℝ), locus_of_P P ↔ P.2 ^ 2 = 2 * P.1 - 1 :=
by
  intro P
  sorry

end locus_of_P_is_parabola_l541_541224


namespace valid_pairs_count_l541_541597

theorem valid_pairs_count :
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 11 ∧
    ∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 + p.2 ≤ 200 ∧ (p.1 + p.2⁻¹) / (p.1⁻¹ + p.2) = 17) :=
sorry

end valid_pairs_count_l541_541597


namespace largest_square_area_l541_541402

-- Given conditions and necessary definitions
def is_right_angle (X Y Z : Type) [InnerProductSpace ℝ X] : Prop := ∃ (c1 c2 : ℝ), X = Y + c1 * (Z - Y) ∧ (c2 = 0)

variables (XY YZ XZ : ℝ)

-- Given conditions for the problem
def angle_XYZ_right : Prop := is_right_angle XYZ
def quadrilateral_areas_sum_eq_450 : Prop := (XY * XY) + (YZ * YZ) + (XZ * XZ) = 450

-- The statement we want to prove
def area_largest_square : Prop := (XZ * XZ) = 225

-- The theorem statement
theorem largest_square_area (h1 : angle_XYZ_right) (h2 : quadrilateral_areas_sum_eq_450) : area_largest_square := 
by
  sorry

end largest_square_area_l541_541402


namespace probability_point_below_parabola_l541_541246

theorem probability_point_below_parabola : 
  (∑ a in Finset.Ico 1 10, ∑ b in Finset.Ico 1 10, 
    if b < a * (-a)^2 + b * (-a) then 1 else 0) / 81 = 2 / 81 :=
by
  sorry

end probability_point_below_parabola_l541_541246


namespace area_of_triangle_l541_541331

theorem area_of_triangle {a b c : ℝ} (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) :
  let s := (a + b + c) / 2,
      area := Math.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 := sorry

end area_of_triangle_l541_541331


namespace f_f_2_eq_0_l541_541607

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem f_f_2_eq_0 : f(f(2)) = 0 := by
  sorry

end f_f_2_eq_0_l541_541607


namespace quadratic_roots_distinct_real_l541_541758

theorem quadratic_roots_distinct_real (a b c : ℝ) (h_eq : 2 * a = 2 ∧ 2 * b + -3 = b ∧ 2 * c + 1 = c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x : ℝ, (2 * x^2 + (-3) * x + 1 = 0) ↔ (x = x1 ∨ x = x2)) :=
by
  sorry

end quadratic_roots_distinct_real_l541_541758


namespace XY_parallel_X_l541_541941

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541941


namespace find_general_formula_find_sum_T2n_l541_541972

-- Definitions based on conditions
def a (n : ℕ) : ℤ := 3 * n - 1
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * 3)) / 2
def b (n : ℕ) : ℤ := (-1) ^ n * a n + 2 ^ (n + 1)
def T (n : ℕ) : ℤ := ∑ i in Finset.range (2 * n), b (i + 1)

-- Proof problem 1: General formula for the arithmetic sequence a_n
theorem find_general_formula (n : ℕ) : 
  (a 3 = 8) ∧ (S 5 = 2 * a 7) → (a n = 3 * n - 1) :=
sorry

-- Proof problem 2: Sum of the first 2n terms of the sequence b_n
theorem find_sum_T2n (n : ℕ) : 
  (b n = (-1) ^ n * (3 * n - 1) + 2 ^ (n + 1)) → 
  (T n = 3 * n + 4 ^ (n + 1) - 4) :=
sorry

end find_general_formula_find_sum_T2n_l541_541972


namespace problem_proof_l541_541920

variable (f : ℝ → ℝ)
variable (hp : ∀ x ∈ Ioo (0 : ℝ) (Real.pi / 2), f x < f' x * Real.tan x)

theorem problem_proof : sqrt 3 * f (Real.pi / 6) < f (Real.pi / 3) := sorry

end problem_proof_l541_541920


namespace intersection_points_of_graphs_l541_541731

open Real

theorem intersection_points_of_graphs (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃! x : ℝ, (f (x^3) = f (x^6)) ∧ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  -- Provide the structure of the proof
  sorry

end intersection_points_of_graphs_l541_541731


namespace range_of_a_l541_541917

noncomputable def A (x : ℝ) : Prop := x^2 - x ≤ 0
noncomputable def B (x : ℝ) (a : ℝ) : Prop := 2^(1 - x) + a ≤ 0

theorem range_of_a (a : ℝ) : (∀ x, A x → B x a) → a ≤ -2 := by
  intro h
  -- Proof steps would go here
  sorry

end range_of_a_l541_541917


namespace PlatformC_location_l541_541647

noncomputable def PlatformA : ℝ := 9
noncomputable def PlatformB : ℝ := 1 / 3
noncomputable def PlatformC : ℝ := 7
noncomputable def AB := PlatformA - PlatformB
noncomputable def AC := PlatformA - PlatformC

theorem PlatformC_location :
  AB = (13 / 3) * AC → PlatformC = 7 :=
by
  intro h
  simp [AB, AC, PlatformA, PlatformB, PlatformC] at h
  sorry

end PlatformC_location_l541_541647


namespace problem_part1_problem_part2_l541_541184

-- Let \( \triangle ABC \) be a triangle with sides \( a \), \( b \), and \( c \) opposite to angles \( A \), \( B \), and \( C \)
-- given the condition \(\frac{a}{{2-\cos A}} = \frac{b}{{\cos B}}\).
-- Prove the following:

theorem problem_part1 (a b c : ℝ) (A B C : ℝ) (h1 : a / (2 - cos A) = b / cos B) :
  c = 2 * b := 
sorry

-- With \( a = 3 \), if the angle bisector of angle \( A \) intersects \( BC \) at point \( D \), 
-- and \( AD = 2\sqrt{2} \), prove that the area of \( \triangle ABC \) is 3.

theorem problem_part2 (a b c : ℝ) (A B C D : ℝ) (h2 : a = 3) (h3 : cos A * D = 2 * sqrt 2) :
  (1 / 2) * b * c * sin A = 3 :=
sorry

end problem_part1_problem_part2_l541_541184


namespace second_derivative_at_0_l541_541120

def f : ℝ → ℝ := λ x, sin x - x

theorem second_derivative_at_0 : deriv (deriv f) 0 = 0 := by
  sorry

end second_derivative_at_0_l541_541120


namespace eval_floor_abs_value_l541_541070

theorem eval_floor_abs_value : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry -- Proof is to be filled in

end eval_floor_abs_value_l541_541070


namespace perfect_square_trinomial_l541_541143

theorem perfect_square_trinomial {m : ℝ} :
  (∃ (a : ℝ), x^2 + 2 * m * x + 9 = (x + a)^2) → (m = 3 ∨ m = -3) :=
sorry

end perfect_square_trinomial_l541_541143


namespace fraction_of_number_l541_541324

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541324


namespace equation_of_BC_area_of_triangle_l541_541989

section triangle_geometry

variables (x y : ℝ)

/-- Given equations of the altitudes and vertex A, the equation of side BC is 2x + 3y + 7 = 0 -/
theorem equation_of_BC (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ (a, b, c) = (2, 3, 7) := 
sorry

/-- Given equations of the altitudes and vertex A, the area of triangle ABC is 45/2 -/
theorem area_of_triangle (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (area : ℝ), (area = (45 / 2)) := 
sorry

end triangle_geometry

end equation_of_BC_area_of_triangle_l541_541989


namespace largest_n_18n_divides_30_factorial_l541_541504

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541504


namespace largest_n_dividing_30_factorial_l541_541497

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541497


namespace shaded_region_area_l541_541237

-- Conditions given in the problem
def diameter (d : ℝ) := d = 4
def length_feet (l : ℝ) := l = 2

-- Proof statement
theorem shaded_region_area (d l : ℝ) (h1 : diameter d) (h2 : length_feet l) : 
  (l * 12 / d * (d / 2)^2 * π = 24 * π) := by
  sorry

end shaded_region_area_l541_541237


namespace number_of_cars_on_street_l541_541428

-- Definitions based on conditions
def cars_equally_spaced (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

def distance_between_first_and_last_car (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 242

def distance_between_cars (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

-- Given all conditions, prove n = 45
theorem number_of_cars_on_street (n : ℕ) :
  cars_equally_spaced n →
  distance_between_first_and_last_car n →
  distance_between_cars n →
  n = 45 :=
sorry

end number_of_cars_on_street_l541_541428


namespace number_of_lattice_points_in_intersection_l541_541066

/-- Definition of the first condition for points 
    lying inside the first sphere centered at (0, 0, 10) with radius 8 -/
def in_first_sphere (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 10)^2 ≤ 64

/-- Definition of the second condition for points 
    lying inside the second sphere centered at (0,0,0) with radius 5 -/
def in_second_sphere (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 ≤ 25

/-- Definition of points lying in the intersection of both spheres -/
def in_intersection (x y z : ℤ) : Prop :=
  in_first_sphere x y z ∧ in_second_sphere x y z

/-- The proof obligation stating the number of points satisfying 
    the intersection condition is 21 -/
theorem number_of_lattice_points_in_intersection :
  { p : ℤ × ℤ × ℤ // in_intersection p.1 p.2 p.3 }.to_finset.card = 21 := 
sorry

end number_of_lattice_points_in_intersection_l541_541066


namespace tan_alpha_plus_beta_eq_5_over_16_tan_beta_eq_31_over_43_l541_541111

variable (α β : ℝ)

-- Condition 1: tan(π + α) = -1/3
def tan_alpha_condition : Prop := Real.tan (Real.pi + α) = -1 / 3

-- Condition 2: tan(α + β) = (sin α + 2 cos α) / (5 cos α - sin α)
def tan_alpha_plus_beta_condition : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Part 1: Prove tan(α + β) = 5 / 16
theorem tan_alpha_plus_beta_eq_5_over_16 (h1 : tan_alpha_condition α) (h2 : tan_alpha_plus_beta_condition α β) : 
  Real.tan (α + β) = 5 / 16 :=
by
  sorry

-- Part 2: Prove tan β = 31 / 43 given additional condition
def tan_beta_identity : Prop := Real.tan (α + β) = (Real.tan α + Real.tan β) / (1 - Real.tan α * Real.tan β)

theorem tan_beta_eq_31_over_43 (h1 : tan_alpha_condition α) (h2 : tan_alpha_plus_beta_eq_5_over_16 α β h1 h2) (h3 : tan_beta_identity α β) :
  Real.tan β = 31 / 43 :=
by
  sorry

end tan_alpha_plus_beta_eq_5_over_16_tan_beta_eq_31_over_43_l541_541111


namespace depth_B_is_correct_l541_541817

-- Given: Diver A is at a depth of -55 meters.
def depth_A : ℤ := -55

-- Given: Diver B is 5 meters above diver A.
def offset : ℤ := 5

-- Prove: The depth of diver B
theorem depth_B_is_correct : (depth_A + offset) = -50 :=
by
  sorry

end depth_B_is_correct_l541_541817


namespace ping_pong_rackets_sold_l541_541830

theorem ping_pong_rackets_sold (total_revenue : ℝ) (avg_price_per_pair : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 539) (h2 : avg_price_per_pair = 9.8) :
  pairs_sold = total_revenue / avg_price_per_pair :=
by
  have h_calculation : pairs_sold = 539 / 9.8 := sorry
  exact h_calculation

end ping_pong_rackets_sold_l541_541830


namespace work_increase_percentage_l541_541167

theorem work_increase_percentage (p : ℕ) (hp : p > 0) : 
  let absent_fraction := 1 / 6
  let work_per_person_original := 1 / p
  let present_people := p - p * absent_fraction
  let work_per_person_new := 1 / present_people
  let work_increase := work_per_person_new - work_per_person_original
  let percentage_increase := (work_increase / work_per_person_original) * 100
  percentage_increase = 20 :=
by
  sorry

end work_increase_percentage_l541_541167


namespace problem_l541_541536

-- Define the real numbers m and n
variables (m n : ℝ)
-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := m - n * i

-- The theorem to prove
theorem problem (h : z = m - n * i) : m + n * i = 2 - i :=
sorry

end problem_l541_541536


namespace symmetric_scanning_codes_count_l541_541384

structure Grid (n : ℕ) :=
  (cells : Fin n × Fin n → Bool)

def is_symmetric_90 (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - j, i)

def is_symmetric_reflection_mid_side (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - i, j) ∧ g.cells (i, j) = g.cells (i, 7 - j)

def is_symmetric_reflection_diagonal (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (j, i)

def has_at_least_one_black_and_one_white (g : Grid 8) : Prop :=
  ∃ i j, g.cells (i, j) ∧ ∃ i j, ¬g.cells (i, j)

noncomputable def count_symmetric_scanning_codes : ℕ :=
  (sorry : ℕ)

theorem symmetric_scanning_codes_count : count_symmetric_scanning_codes = 62 :=
  sorry

end symmetric_scanning_codes_count_l541_541384


namespace rearranged_cards_returnable_l541_541627

theorem rearranged_cards_returnable (n : ℕ) (a : Fin n → ℕ) : 
  ∃ (swaps : List (Fin n × Fin n)), 
  (∀ {i j}, (i, j) ∈ swaps → i ≠ j) ∧
  (∀ (k), k ∈ swaps.foldl (λ (a : Array (Fin n)), fun (i : Fin n) (j : Fin n), a.swap i j) (Array.ofFn (λ i, a i)) → k.1 + 1 = k.2) ∧
  List.sum (swaps.map (λ ⟨i, j⟩, 2 * |(a i) - (a j)|)) ≤ (∑ i, abs ((a i) - i)) :=
by
  sorry

end rearranged_cards_returnable_l541_541627


namespace area_of_largest_square_l541_541405

theorem area_of_largest_square (XZ YZ XY : ℝ) (h₁ : angle XYZ = 90) 
  (h₂ : XZ^2 + YZ^2 + XY^2 = 450) : XY^2 = 225 := 
sorry

end area_of_largest_square_l541_541405


namespace find_digit_A_l541_541150

-- Definitions from the conditions in the problem
def deck_size : ℕ := 52
def hand_size : ℕ := 13
def card_to_include := "ace_of_spades"
def formatted_number := "635A13559600"

-- Proposition stating the proof problem
theorem find_digit_A :
  let n := deck_size - 1 in
  let k := hand_size - 1 in
  let number_of_ways := Nat.choose n k in
  let number_str := number_of_ways.repr in
  ∃ (A : ℕ), number_str = "635" ++ A.repr ++ "13559600" ∧ A = 0 := 
by {
  sorry
}

end find_digit_A_l541_541150


namespace round_trip_time_l541_541271

def boat_speed := 9 -- speed of the boat in standing water (kmph)
def stream_speed := 6 -- speed of the stream (kmph)
def distance := 210 -- distance to the place (km)

def upstream_speed := boat_speed - stream_speed
def downstream_speed := boat_speed + stream_speed

def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_time := time_upstream + time_downstream

theorem round_trip_time : total_time = 84 := by
  sorry

end round_trip_time_l541_541271


namespace quadratic_distinct_roots_l541_541915

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l541_541915


namespace largest_n_dividing_30_factorial_l541_541495

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541495


namespace fraction_multiplication_l541_541306

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541306


namespace find_M_find_min_a_find_range_b_l541_541130

-- Prove that M = (-1, 2) given M = { x | x^2 - x - 2 < 0 }
theorem find_M : 
  (∀ x : ℝ, x ∈ {x | x^2 - x - 2 < 0} ↔ x ∈ set.Ioo (-1 : ℝ) 2) := 
by 
  sorry

-- Prove that the minimum value of a is -1 given M = (-1, 2) and M ⊇ N
theorem find_min_a (a b : ℝ) :
  (set.Ioo (-1 : ℝ) 2).superset (set.Ioo a b) → (a ≥ -1) :=
by 
  sorry

-- Prove that the range of b is [2, +∞) given M = (-1, 2) and M ∩ N = M
theorem find_range_b (a b : ℝ) :
  (set.Ioo (-1 : ℝ) 2).inter (set.Ioo a b) = (set.Ioo (-1 : ℝ) 2) → (2 ≤ b) :=
by 
  sorry

end find_M_find_min_a_find_range_b_l541_541130


namespace student_A_more_stable_performance_l541_541296

theorem student_A_more_stable_performance
    (mean : ℝ)
    (n : ℕ)
    (variance_A variance_B : ℝ)
    (h1 : mean = 1.6)
    (h2 : n = 10)
    (h3 : variance_A = 1.4)
    (h4 : variance_B = 2.5) :
    variance_A < variance_B :=
by {
    -- The proof is omitted as we are only writing the statement here.
    sorry
}

end student_A_more_stable_performance_l541_541296


namespace decreasing_interval_log_computation_true_statements_count_range_of_m_l541_541797

-- 1. Prove the monotonically decreasing interval of f(x) = log(-x^2 + 2x) is (1, 2).
theorem decreasing_interval (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  ∀ {y : ℝ}, -y^2 + 2*y > 0 :=
sorry

-- 2. Prove log₃(5/4) + 4 ^ log₂3 - log₃(5/36) = 11.
theorem log_computation : logb 3 (5/4) + 4 ^ logb 2 3 - logb 3 (5/36) = 11 :=
sorry

-- 3. Prove the number of true statements among the given propositions is 1.
theorem true_statements_count :
  let p1 := ¬(∃ x : ℝ, x^2 + 1 > 3 * x) = ∀ x : ℝ, x^2 + 1 ≤ 3 * x,
      p2 := ¬(p ∨ q) = ¬p ∧ ¬q,
      p3 := (a > 2) → (a > 5),
      p4 := ¬(xy = 0 → x = 0 ∧ y = 0)
  in (p1, p2, p3, p4).true_count = 1 :=
sorry

-- 4. Prove the range of values for m is (1, +∞) for f(x) = log(mx^2 + 2mx + 1) to have range ℝ.
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2*mx + 1 > 0) ↔ m > 1 :=
sorry

end decreasing_interval_log_computation_true_statements_count_range_of_m_l541_541797


namespace Bridget_skittles_after_giving_l541_541856

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l541_541856


namespace solve_equation_l541_541145

theorem solve_equation : 
  ∀ x : ℝ, (9 / x ^ 3 = x / 81) → (x = 3 * real.sqrt 3) :=
by
  intro x h
  sorry

end solve_equation_l541_541145


namespace union_of_sets_l541_541587

theorem union_of_sets : 
  let A := {1, 2, 3} 
  let B := {2, 3, 4} 
  A ∪ B = {1, 2, 3, 4} :=
by
  let A := {1, 2, 3}
  let B := {2, 3, 4}
  show A ∪ B = {1, 2, 3, 4}
  sorry

end union_of_sets_l541_541587


namespace triangle_inequality_AD_DE_AF_l541_541230

variables {A B C D E F M : Type} [AffineSpace ℝ A] [IsAffineSubspace ℝ B] [IsAffineSubspace ℝ C]
          [IsAffineSubspace ℝ D] [IsAffineSubspace ℝ E] [IsAffineSubspace ℝ F]

-- Conditions
def midpoint (A B M : Type) [x : IsAffineSubspace ℝ A] [y : IsAffineSubspace ℝ B] [z : IsAffineSubspace ℝ M] : Prop :=
  sorry

def isosceles_triangle (A B C : Type) [x : IsAffineSubspace ℝ A] [y : IsAffineSubspace ℝ B] [z : IsAffineSubspace ℝ C] : Prop := 
  sorry

def on_extension (A C D : Type) [x : IsAffineSubspace ℝ A] [y : IsAffineSubspace ℝ C] [z : IsAffineSubspace ℝ D] : Prop := 
  sorry

def on_segment (B M E : Type) [x : IsAffineSubspace ℝ B] [y : IsAffineSubspace ℝ M] [z : IsAffineSubspace ℝ E] : Prop := 
  sorry

def circumcircle_intersect (C D E M E' : Type) [x : IsAffineSubspace ℝ C] [y : IsAffineSubspace ℝ D] [z : IsAffineSubspace ℝ E] [w : IsAffineSubspace ℝ M] [v : IsAffineSubspace ℝ E'] : Prop := 
  sorry

-- Lean statement
theorem triangle_inequality_AD_DE_AF :
  midpoint A B M →
  isosceles_triangle A B C →
  on_extension A C D →
  on_segment B M E →
  circumcircle_intersect C D E M F →
  AD + DE > AF ∧ AD + AF > DE ∧ DE + AF > AD := 
sorry

end triangle_inequality_AD_DE_AF_l541_541230


namespace b_is_perfect_square_l541_541666

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem b_is_perfect_square (a b : ℕ)
  (h_positive : 0 < a) (h_positive_b : 0 < b)
  (h_gcd_lcm_multiple : (Nat.gcd a b + Nat.lcm a b) % (a + 1) = 0)
  (h_le : b ≤ a) : is_perfect_square b :=
sorry

end b_is_perfect_square_l541_541666


namespace mary_garbage_bill_l541_541689

theorem mary_garbage_bill :
  let weekly_cost := 2 * 10 + 1 * 5,
      monthly_cost := weekly_cost * 4,
      discount := 0.18 * monthly_cost,
      discounted_monthly_cost := monthly_cost - discount,
      fine := 20,
      total_bill := discounted_monthly_cost + fine
  in total_bill = 102 :=
by
  let weekly_cost := 2 * 10 + 1 * 5
  let monthly_cost := weekly_cost * 4
  let discount := 0.18 * monthly_cost
  let discounted_monthly_cost := monthly_cost - discount
  let fine := 20
  let total_bill := discounted_monthly_cost + fine
  show total_bill = 102 from sorry

end mary_garbage_bill_l541_541689


namespace coefficient_of_x2_l541_541256

noncomputable def binomial_expansion_coefficient : ℤ :=
  let poly := (2 + x) * (1 - 2 * x) ^ 5 in
  coeff poly (x^2) sorry

theorem coefficient_of_x2 (x : ℂ) :
  coeff (binomial_expansion_coefficient) (x^2) = 70 :=
sorry

end coefficient_of_x2_l541_541256


namespace calculate_expression_l541_541057

theorem calculate_expression : 
  (3 * 7.5 * (6 + 4) / 2.5) = 90 := 
by
  sorry

end calculate_expression_l541_541057


namespace largest_n_18n_divides_30_factorial_l541_541503

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541503


namespace calculate_expression_l541_541056

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l541_541056


namespace quadratic_roots_opposite_signs_l541_541076

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ x * y < 0) ↔ (a < 0) :=
sorry

end quadratic_roots_opposite_signs_l541_541076


namespace greatest_prime_saturated_two_digit_l541_541792

-- Definition of prime saturation condition
def prime_saturated (d : ℕ) : Prop :=
  let prime_factors := unique_factors d in
  (prime_factors.prod id) < Nat.sqrt d

-- Statement of the problem
theorem greatest_prime_saturated_two_digit : ∃ d : ℕ, d = 96 ∧ prime_saturated d ∧ ∀ k : ℕ, k > d ∧ k < 100 → ¬ prime_saturated k := 
  by sorry

end greatest_prime_saturated_two_digit_l541_541792


namespace linear_relationship_max_profit_donation_range_l541_541704

-- Definitions based on conditions:
def cost_price := 50 : ℕ
def profit_margin := 0.52 : ℝ
def price_point1 := 60 : ℕ
def price_point2 := 70 : ℕ
def sales_volume1 := 240 : ℕ
def sales_volume2 := 180 : ℕ

-- (1) Prove the functional relationship between y and x is y = -6x + 600
theorem linear_relationship : ∃ k b : ℤ, y = k * x + b ∧ (k = -6) ∧ (b = 600) :=
sorry

-- (2) Prove selling price that maximizes daily profit and calculate the maximum profit
theorem max_profit : ∃ x : ℕ, 50 ≤ x ∧ x ≤ 76 ∧ ∃ w : ℕ, w = (x - 50) * (-6 * x + 600) ∧ (w = 3750) ∧ (x = 75) :=
sorry

-- (3) Prove the range of values for n such that daily profit after donation increases
theorem donation_range (n : ℝ) : (50 ≤ x ∧ x ≤ 76) ∧ (1 < n) ∧ (n < 2) →
  ∃ w' : ℝ, w' = (-6 * x + 600) * (x - 50 - n) → (w' > w) :=
sorry

end linear_relationship_max_profit_donation_range_l541_541704


namespace pratt_certificate_space_bound_l541_541239

-- Define the Pratt certificate space function λ(p)
noncomputable def pratt_space (p : ℕ) : ℝ := sorry

-- Define the log_2 function (if not already available in Mathlib)
noncomputable def log2 (x : ℝ) : ℝ := sorry

-- Assuming that p is a prime number
variable {p : ℕ} (hp : Nat.Prime p)

-- The proof problem
theorem pratt_certificate_space_bound (hp : Nat.Prime p) :
  pratt_space p ≤ 6 * (log2 p) ^ 2 := 
sorry

end pratt_certificate_space_bound_l541_541239


namespace no_positive_integer_makes_polynomial_prime_l541_541531

open Nat

theorem no_positive_integer_makes_polynomial_prime :
  ∀ n : ℕ, 0 < n → ¬ Prime (n^3 - 9 * n^2 + 23 * n - 17) :=
by
  sorry

end no_positive_integer_makes_polynomial_prime_l541_541531


namespace factorization_simplification_system_of_inequalities_l541_541001

-- Problem 1: Factorization
theorem factorization (x y : ℝ) : 
  x^2 * (x - 3) + y^2 * (3 - x) = (x - 3) * (x + y) * (x - y) := 
sorry

-- Problem 2: Simplification
theorem simplification (x : ℝ) (hx : x ≠ 0) (hx1 : 5 * x ≠ 3) (hx2 : 5 * x ≠ -3) : 
  (2 * x / (5 * x - 3)) / (3 / (25 * x^2 - 9)) * (x / (5 * x + 3)) = (2 / 3) * x^2 := 
sorry

-- Problem 3: System of inequalities
theorem system_of_inequalities (x : ℝ) : 
  ((x - 3) / 2 + 3 ≥ x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) ↔ -2 < x ∧ x ≤ 1 := 
sorry

end factorization_simplification_system_of_inequalities_l541_541001


namespace ratio_AD_BC_l541_541407

variables (A B C D M N P K Q : Point)
variables (a b : Real)
variables (circle : Circle)
variables [isosceles_trapezoid ABCD]
variables [tangency AB circle M]
variables [tangency AD circle N]
variables (h_intersect : MN ∩ AC = P)
variables (h_ratio : NP / PM = 2)

theorem ratio_AD_BC : (AD / BC) = 3 :=
by
  have h1 : AB = 8 * b,
    sorry,
  have h2 : AD = 4 * a,
    sorry,
  have h3 : BC = 4 * b,
    sorry,
  have final : (AD / BC) = (4 * a) / (4 * b) = a / b,
    sorry,
  have result : a / b = 3,
    sorry,
  rw result,
  exact sorry

end ratio_AD_BC_l541_541407


namespace Bridget_Skittles_Final_l541_541859

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l541_541859


namespace sum_base4_of_195_and_61_l541_541746

theorem sum_base4_of_195_and_61 :
  let (n1, n2) := (195, 61),
      sum_base10 := n1 + n2,
      sum_base4 := 10000
  in sum_base4 = sum_base10.toBase 4 :=
by sorry

end sum_base4_of_195_and_61_l541_541746


namespace largest_n_dividing_factorial_l541_541522

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541522


namespace find_rate_per_kg_grapes_l541_541051

-- Define the main conditions
def rate_per_kg_mango := 55
def total_payment := 985
def kg_grapes := 7
def kg_mangoes := 9

-- Define the problem statement
theorem find_rate_per_kg_grapes (G : ℝ) : 
  (kg_grapes * G + kg_mangoes * rate_per_kg_mango = total_payment) → 
  G = 70 :=
by
  sorry

end find_rate_per_kg_grapes_l541_541051


namespace sector_area_l541_541813

open Real

theorem sector_area (C : ℝ) (angle θ : ℝ) (h1: C = 16 * π) (h2: θ = 90) : 
  let r := C / (2 * π),
      A := π * r^2,
      sector_area := (θ / 360) * A
  in
  sector_area = 16 * π :=
by 
  sorry

end sector_area_l541_541813


namespace first_divisor_l541_541765

-- Definitions
def is_divisible_by (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Theorem to prove
theorem first_divisor (x : ℕ) (h₁ : ∃ l, l = Nat.lcm x 35 ∧ is_divisible_by 1400 l ∧ 1400 / l = 8) : 
  x = 25 := 
sorry

end first_divisor_l541_541765


namespace equation_of_circle_l541_541169

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 6 * x + 1

-- Define the circle
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- The proof problem statement in Lean
theorem equation_of_circle :
  (∀ x y : ℝ, (curve x = y ∧ x = 0 → circle 0 (curve 0))
  ∧ (curve x = 0 ∧ y = 0 → circle x 0)) → 
   (∀ x y : ℝ, (curve x = y ∧ (x = 0 ∨ y = 0)) → circle x y) :=
by
  sorry

end equation_of_circle_l541_541169


namespace solutions_exist_iff_l541_541448

variable (a b : ℝ)

theorem solutions_exist_iff :
  (∃ x y : ℝ, (x^2 + y^2 + xy = a) ∧ (x^2 - y^2 = b)) ↔ (-2 * a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2 * a) :=
sorry

end solutions_exist_iff_l541_541448


namespace decimal_to_fraction_simplify_l541_541258

noncomputable def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem decimal_to_fraction_simplify
  (m n : ℕ)
  (h1 : 0.1824 = 1824 / 10000)
  (h2 : gcd 1824 10000 = 16)
  (h3 : m = 1824 / 16)
  (h4 : n = 10000 / 16)
  : m + n = 739 :=
by
  sorry

end decimal_to_fraction_simplify_l541_541258


namespace largest_n_dividing_factorial_l541_541516

theorem largest_n_dividing_factorial :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) ↔ m ≤ n) ∧ n = 7 :=
sorry

end largest_n_dividing_factorial_l541_541516


namespace increasing_interval_of_g_l541_541800

noncomputable def g (x : ℝ) : ℝ := x * (2 - x)

theorem increasing_interval_of_g : set.Iic (1 : ℝ) = {x : ℝ | x ∈ set.Iic 1 ∧ strict_mono_incr_on g {x | x ≤ 1} } :=
by
  intros 
  sorry 

end increasing_interval_of_g_l541_541800


namespace total_kayaks_by_end_of_july_l541_541420

theorem total_kayaks_by_end_of_july : 
    let kayaks : ℕ → ℕ := λ n, 8 * 3^n
    let sum := ∑ n in Finset.range 6, kayaks n
    sum = 968 :=
by
    sorry

end total_kayaks_by_end_of_july_l541_541420


namespace matrix_inverse_correctness_l541_541583

-- Define the matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![1, 1]]

-- Define the supposed inverse matrix M_inv
def M_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![-1, 1]]

-- Statement of the problem: M * M_inv should be the identity matrix
theorem matrix_inverse_correctness : M ⬝ M_inv = 1 := by
  sorry

end matrix_inverse_correctness_l541_541583


namespace det_A_l541_541869

-- Define the matrix A
noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sin 1, Real.cos 2, Real.sin 3],
   ![Real.sin 4, Real.cos 5, Real.sin 6],
   ![Real.sin 7, Real.cos 8, Real.sin 9]]

-- Define the explicit determinant calculation
theorem det_A :
  Matrix.det A = Real.sin 1 * (Real.cos 5 * Real.sin 9 - Real.sin 6 * Real.cos 8) -
                 Real.cos 2 * (Real.sin 4 * Real.sin 9 - Real.sin 6 * Real.sin 7) +
                 Real.sin 3 * (Real.sin 4 * Real.cos 8 - Real.cos 5 * Real.sin 7) :=
by
  sorry

end det_A_l541_541869


namespace extremum_value_and_min_on_interval_l541_541121

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem extremum_value_and_min_on_interval
  (a b c : ℝ)
  (h1_eq : 12 * a + b = 0)
  (h2_eq : 4 * a + b = -8)
  (h_max : 16 + c = 28) :
  min (min (f a b c (-3)) (f a b c 3)) (f a b c 2) = -4 :=
by sorry

end extremum_value_and_min_on_interval_l541_541121


namespace convex_polygon_width_one_contains_circle_convex_polygon_width_one_no_larger_circle_l541_541350

-- Problem statements

theorem convex_polygon_width_one_contains_circle :
  ∀ (M : convex_polygon), (width M = 1) → (∃ (r : ℝ), r = 1 / 3 ∧ inscribed_circle_radius M r) := by
  sorry

theorem convex_polygon_width_one_no_larger_circle :
  ∃ (M : convex_polygon), (width M = 1) ∧ (∀ (r : ℝ), (r > 1 / 3) → ¬ inscribed_circle_radius M r) := by
  sorry

end convex_polygon_width_one_contains_circle_convex_polygon_width_one_no_larger_circle_l541_541350


namespace travel_distance_l541_541030

-- Define the conditions
def distance_10_gallons := 300 -- 300 miles on 10 gallons of fuel
def gallons_10 := 10 -- 10 gallons

-- Given the distance per gallon, calculate the distance for 15 gallons
def distance_per_gallon := distance_10_gallons / gallons_10

def gallons_15 := 15 -- 15 gallons

def distance_15_gallons := distance_per_gallon * gallons_15

-- Proof statement
theorem travel_distance (d_10 : distance_10_gallons = 300)
                        (g_10 : gallons_10 = 10)
                        (g_15 : gallons_15 = 15) :
  distance_15_gallons = 450 :=
  by
  -- The actual proof goes here
  sorry

end travel_distance_l541_541030


namespace XY_parallel_X_l541_541933

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541933


namespace train_passing_time_l541_541351

theorem train_passing_time (distance : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (time : ℝ) :
  distance = 125 → speed_kmh = 60 → speed_ms = (speed_kmh * 1000) / 3600 → time = distance / speed_ms →
  time ≈ 7.5 :=
by
  intros distance_eq speed_kmh_eq speed_conversion time_eq
  rw [distance_eq, speed_kmh_eq, speed_conversion] at time_eq
  sorry

end train_passing_time_l541_541351


namespace at_least_100_arcs_of_21_points_l541_541764

noncomputable def count_arcs (n : ℕ) (θ : ℝ) : ℕ :=
-- Please note this function needs to be defined appropriately, here we assume it computes the number of arcs of θ degrees or fewer between n points on a circle
sorry

theorem at_least_100_arcs_of_21_points :
  ∃ (n : ℕ), n = 21 ∧ count_arcs n (120 : ℝ) ≥ 100 :=
sorry

end at_least_100_arcs_of_21_points_l541_541764


namespace second_tree_branches_l541_541876

theorem second_tree_branches
  (height_first_tree : ℕ) (branches_first_tree : ℕ)
  (height_second_tree : ℕ)
  (height_third_tree : ℕ) (branches_third_tree : ℕ)
  (height_final_tree : ℕ) (branches_final_tree : ℕ)
  (average_branches_per_foot : ℕ)
  (ht1 : height_first_tree = 50) 
  (br1 : branches_first_tree = 200)
  (ht2 : height_second_tree = 40)
  (ht3 : height_third_tree = 60) 
  (br3 : branches_third_tree = 180)
  (ht4 : height_final_tree = 34) 
  (br4 : branches_final_tree = 153)
  (avg_br : average_branches_per_foot = 4)
  : height_second_tree * average_branches_per_foot = 160 :=
by
  simp [ht2, avg_br]
  exact rfl

end second_tree_branches_l541_541876


namespace change_factor_l541_541253

theorem change_factor (n : ℕ) (avg_original avg_new : ℕ) (F : ℝ)
  (h1 : n = 10) (h2 : avg_original = 80) (h3 : avg_new = 160) 
  (h4 : F * (n * avg_original) = n * avg_new) :
  F = 2 :=
by
  sorry

end change_factor_l541_541253


namespace exist_non_zero_function_iff_sum_zero_l541_541075

theorem exist_non_zero_function_iff_sum_zero (a b c : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, a * f (x * y + f z) + b * f (y * z + f x) + c * f (z * x + f y) = 0) ∧ ¬ (∀ x : ℝ, f x = 0)) ↔ (a + b + c = 0) :=
by {
  sorry
}

end exist_non_zero_function_iff_sum_zero_l541_541075


namespace circle_equation_standard_form_l541_541618

theorem circle_equation_standard_form (h k r : ℝ) (hc : h = 1) (kc : k = -2) (rc : r = 6) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 1)^2 + (y + 2)^2 = 36 :=
by
  intros x y
  rw [hc, kc, rc]
  norm_num
  exact iff.rfl
  sorry

end circle_equation_standard_form_l541_541618


namespace no_three_digit_numbers_meet_conditions_l541_541599

theorem no_three_digit_numbers_meet_conditions :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (n % 10 = 5) ∧ (n % 10 = 0) → false := 
by {
  sorry
}

end no_three_digit_numbers_meet_conditions_l541_541599


namespace lines_intersect_l541_541447

theorem lines_intersect (a b : ℝ) 
  (h₁ : ∃ y : ℝ, 4 = (3/4) * y + a ∧ y = 3)
  (h₂ : ∃ x : ℝ, 3 = (3/4) * x + b ∧ x = 4) :
  a + b = 7/4 :=
sorry

end lines_intersect_l541_541447


namespace correct_cube_representation_l541_541290

-- Define the template structure
structure Template :=
  (white_faces : List ℕ)  -- Faces that are white
  (gray_faces : List ℕ)   -- Faces that are gray
  (opposing_faces : List (ℕ × ℕ))  -- List of pairs of opposing faces

-- Define the Problem conditions
def guilherme_template : Template :=
{ white_faces := [1, 2, 3],
  gray_faces := [4, 5, 6],
  opposing_faces := [(1, 4), (2, 5), (3, 6)] }

-- Define the type of Figures (representing the painted cube)
structure Cube :=
  (faces : List (ℕ × String))  -- List of faces and their colors

-- Given possible figures
def figure_A : Cube := { faces := [(1, "white"), (2, "gray"), (3, "gray"), (4, "white"), (5, "white"), (6, "gray")] }
def figure_B : Cube := { faces := [(1, "white"), (2, "gray"), (3, "white"), (4, "gray"), (5, "gray"), (6, "white")] }
def figure_C : Cube := { faces := [(1, "white"), (2, "white"), (3, "white"), (4, "gray"), (5, "gray"), (6, "gray")] }
def figure_D : Cube := { faces := [(1, "white"), (2, "gray"), (3, "white"), (4, "gray"), (5, "white"), (6, "gray")] }
def figure_E : Cube := { faces := [(1, "white"), (2, "white"), (3, "gray"), (4, "gray"), (5, "gray"), (6, "white")] }

-- Define the main theorem to be proved: figure_C represents the correct cube from guilherme_template
theorem correct_cube_representation : ∀ (cube : Cube), 
  (cube = figure_A ∨ cube = figure_B ∨ cube = figure_C ∨ cube = figure_D ∨ cube = figure_E) →
  (∀ face in cube.faces, (face.2 = "white" → face.1 ∈ guilherme_template.white_faces) ∧ 
                        (face.2 = "gray" → face.1 ∈ guilherme_template.gray_faces)) ∧
  (∀ pair in guilherme_template.opposing_faces, 
    ∃ f1 f2, (f1.1 = pair.1 ∧ f2.1 = pair.2 ∧ f1.2 ≠ f2.2) → f1 ∈ cube.faces ∧ f2 ∈ cube.faces) → 
  cube = figure_C :=
sorry  -- Proof skipped

end correct_cube_representation_l541_541290


namespace marys_garbage_bill_l541_541692

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l541_541692


namespace optionB_is_difference_of_squares_l541_541341

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l541_541341


namespace greatest_value_of_a2_b2_c2_d2_l541_541200

-- Conditions given in the problem.
variables (a b c d : ℝ)

axiom h1 : a + b = 20
axiom h2 : ab + c + d = 100
axiom h3 : ad + bc = 250
axiom h4 : cd = 144

-- Proof problem statement.
theorem greatest_value_of_a2_b2_c2_d2 : a^2 + b^2 + c^2 + d^2 ≤ 1760 :=
by
  sorry

end greatest_value_of_a2_b2_c2_d2_l541_541200


namespace bus_routes_theorem_l541_541189

open Function

def bus_routes_exist : Prop :=
  ∃ (routes : Fin 10 → Set (Fin 10)), 
  (∀ (s : Finset (Fin 10)), (s.card = 8) → ∃ (stop : Fin 10), ∀ i ∈ s, stop ∉ routes i) ∧
  (∀ (s : Finset (Fin 10)), (s.card = 9) → ∀ (stop : Fin 10), ∃ i ∈ s, stop ∈ routes i)

theorem bus_routes_theorem : bus_routes_exist :=
sorry

end bus_routes_theorem_l541_541189


namespace keychain_arrangement_l541_541632

theorem keychain_arrangement (house car locker office key5 key6 : ℕ) :
  (∃ (A B : ℕ), house = A ∧ car = A ∧ locker = B ∧ office = B) →
  (∃ (arrangements : ℕ), arrangements = 24) :=
by
  sorry

end keychain_arrangement_l541_541632


namespace XY_parallel_X_l541_541967

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541967


namespace sine_of_largest_angle_in_isosceles_triangle_l541_541163

theorem sine_of_largest_angle_in_isosceles_triangle (r : ℝ) (h1 : r > 0) 
  (h2 : ∃ (α β γ : ℝ), α + β + γ = π ∧ α = α ∧ γ = 2 * α ∧ isosceles ∧ base = 4 * r) :
  ∃ β : ℝ, sin β = 24 / 25 :=
sorry

end sine_of_largest_angle_in_isosceles_triangle_l541_541163


namespace largest_divisor_18n_max_n_l541_541476

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541476


namespace each_mouse_not_visit_with_every_other_once_l541_541012

theorem each_mouse_not_visit_with_every_other_once : 
    (∃ mice: Finset ℕ, mice.card = 24 ∧ (∀ f : ℕ → Finset ℕ, 
    (∀ n, (f n).card = 4) ∧ 
    (∀ i j, i ≠ j → (f i ∩ f j ≠ ∅) → (f i ∩ f j).card ≠ (mice.card - 1)))
    ) → false := 
by
  sorry

end each_mouse_not_visit_with_every_other_once_l541_541012


namespace total_number_of_items_l541_541790

-- Define the conditions as equations in Lean
def model_cars_price := 5
def model_trains_price := 8
def total_amount := 31

-- Initialize the variable definitions for number of cars and trains
variables (c t : ℕ)

-- The proof problem: Show that given the equation, the sum of cars and trains is 5
theorem total_number_of_items : (model_cars_price * c + model_trains_price * t = total_amount) → (c + t = 5) := by
  -- Proof steps would go here
  sorry

end total_number_of_items_l541_541790


namespace fraction_of_number_l541_541311

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l541_541311


namespace distinct_flavors_count_l541_541728

theorem distinct_flavors_count : 
    (number_of_distinct_flavors (6 : ℕ) (4 : ℕ)) = 21 :=
sorry

end distinct_flavors_count_l541_541728


namespace kangaroo_chase_l541_541615

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase_l541_541615


namespace lattice_point_distance_l541_541376

theorem lattice_point_distance (d : ℝ) : 
  (∃ (r : ℝ), r = 2020 ∧ (∀ (A B C D : ℝ), 
  A = 0 ∧ B = 4040 ∧ C = 2020 ∧ D = 4040) 
  ∧ (∃ (P Q : ℝ), P = 0.25 ∧ Q = 1)) → 
  d = 0.3 := 
by
  sorry

end lattice_point_distance_l541_541376


namespace sine_central_angle_is_rational_product_l541_541737

noncomputable def central_angle_sine := sorry

theorem sine_central_angle_is_rational_product (r : ℝ) (BC AD : ℝ) (A B O : Point)
  (circle : Circle O r)
  (h1 : r = 5)
  (h2 : BC = 6)
  (h3 : is_bisected_by A D BC)
  (h4 : ∀ C', C' ≠ D → ¬is_bisected_by A C' BC) :
  let θ := ∠AOB in
  is_rational (sin θ) ∧ central_angle_sine = 7 / 25 :=
begin
  sorry
end

end sine_central_angle_is_rational_product_l541_541737


namespace baker_bakes_25_hours_per_week_mon_to_fri_l541_541368

-- Define the conditions
def loaves_per_hour_per_oven := 5
def number_of_ovens := 4
def weekend_baking_hours_per_day := 2
def total_weeks := 3
def total_loaves := 1740

-- Calculate the loaves per hour
def loaves_per_hour := loaves_per_hour_per_oven * number_of_ovens

-- Calculate the weekend baking hours in one week
def weekend_baking_hours_per_week := weekend_baking_hours_per_day * 2

-- Calculate the loaves baked on weekends in one week
def loaves_on_weekends_per_week := loaves_per_hour * weekend_baking_hours_per_week

-- Calculate the total loaves baked on weekends in 3 weeks
def loaves_on_weekends_total := loaves_on_weekends_per_week * total_weeks

-- Calculate the loaves baked from Monday to Friday in 3 weeks
def loaves_on_weekdays_total := total_loaves - loaves_on_weekends_total

-- Calculate the total hours baked from Monday to Friday in 3 weeks
def weekday_baking_hours_total := loaves_on_weekdays_total / loaves_per_hour

-- Calculate the number of hours baked from Monday to Friday in one week
def weekday_baking_hours_per_week := weekday_baking_hours_total / total_weeks

-- Proof statement
theorem baker_bakes_25_hours_per_week_mon_to_fri :
  weekday_baking_hours_per_week = 25 :=
by
  sorry

end baker_bakes_25_hours_per_week_mon_to_fri_l541_541368


namespace f_zero_f_odd_range_of_x_l541_541065

variable {f : ℝ → ℝ}

axiom func_property (x y : ℝ) : f (x + y) = f x + f y
axiom f_third : f (1 / 3) = 1
axiom f_positive (x : ℝ) : x > 0 → f x > 0

-- Part (1)
theorem f_zero : f 0 = 0 :=
sorry

-- Part (2)
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- Part (3)
theorem range_of_x (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 :=
sorry

end f_zero_f_odd_range_of_x_l541_541065


namespace largest_divisor_18n_max_n_l541_541475

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541475


namespace polynomial_satisfies_conditions_l541_541074

-- Define the problem conditions and answer
noncomputable def P : ℝ → ℝ → ℝ := λ x y, (x + y)^(n - 1) * (x - 2 * y)

-- Conditions for the problem
def homogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def condition2 (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0

def condition3 (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

-- Main theorem statement
theorem polynomial_satisfies_conditions (n : ℕ) :
  homogeneous P n ∧ condition2 P ∧ condition3 P :=
by
  -- Proof goes here
  sorry

end polynomial_satisfies_conditions_l541_541074


namespace tan_add_sin_l541_541432

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l541_541432


namespace BR_perp_CR_l541_541149

variables {A B C I P Q R : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space I]
variable [metric_space R]

-- Definitions and conditions based on the problem statement
def triangle_isosceles (A B C : Type) (AB AC : ℝ) : Prop :=
  AB = AC

def is_incenter (I A B C : Type) : Prop := sorry -- Placeholder for incenter property definition

def circle_centered (O A : Type) (radius : ℕ) : Type :=
  sorry -- Placeholder for circle-centered definition

def circle_passing_through (O B : Type) : Type :=
  sorry -- Placeholder for circle passing through points definition

variables (AB AC IB : ℝ)
variables AB_eq_AC : triangle_isosceles A B C AB AC
variables (I_is_incenter : is_incenter I A B C)
variable (Gamma1 : circle_centered A A AB)
variable (Gamma2 : circle_centered I I IB)
variable (Gamma3 : circle_passing_through B I)

variable (P : Type) (Q : Type)
variables (P_on_Gamma1 : P ∈ Gamma1) (Q_on_Gamma2 : Q ∈ Gamma2)
variables (Gamma3_inter_P : P ∈ Gamma3) (Gamma3_inter_Q : Q ∈ Gamma3)
variable (P_ne_B : P ≠ B) (Q_ne_B : Q ≠ B)

variables (I_segments_eq : IB = IB)  -- I as incenter, thus distances to B and C

-- Definition of the intersection point R
variables (IP : Type) (BQ : Type)
variable (IP_inter_BQ : R)

-- The final proof goal
theorem BR_perp_CR (A B C I P Q R : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space I] [metric_space R] 
  (AB AC IB : ℝ)
  (AB_eq_AC : triangle_isosceles A B C AB AC)
  (I_is_incenter : is_incenter I A B C)
  (Gamma1 : circle_centered A A AB)
  (Gamma2 : circle_centered I I IB)
  (Gamma3 : circle_passing_through B I)
  (P : Type) (Q : Type)
  (P_on_Gamma1 : P ∈ Gamma1) (Q_on_Gamma2 : Q ∈ Gamma2)
  (Gamma3_inter_P : P ∈ Gamma3) (Gamma3_inter_Q : Q ∈ Gamma3)
  (P_ne_B : P ≠ B) (Q_ne_B : Q ≠ B)
  (I_segments_eq : IB = IB)
  (IP : Type) (BQ : Type)
  (IP_inter_BQ : R) :
  sorry  -- Placeholder for proof that BR ⊥ CR

end BR_perp_CR_l541_541149


namespace triangle_perimeter_l541_541645

-- Define the conditions:
def is_isosceles_triangle (A B C : Type) [metric_space A] (AB AC BC : ℝ) :=
  AB = AC

variable (A B C : Type) [metric_space A] (BC AB : ℝ)
variable [cond1 : is_isosceles_triangle A B C AB AB BC] 

-- Define the proof problem:
theorem triangle_perimeter (BC AB : ℝ) (h1 : BC = 8) (h2 : AB = 13)
  (h3 : is_isosceles_triangle A B C AB AB BC) : 
  2 * AB + BC = 34 := 
sorry

end triangle_perimeter_l541_541645


namespace quadratic_distinct_roots_l541_541914

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l541_541914


namespace largest_n_18n_divides_30_factorial_l541_541507

theorem largest_n_18n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ fact 30 ↔ m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_n_18n_divides_30_factorial_l541_541507


namespace tan_30_deg_plus_4_sin_30_deg_eq_l541_541439

theorem tan_30_deg_plus_4_sin_30_deg_eq :
  let sin30 := 1 / 2 in
  let cos30 := Real.sqrt 3 / 2 in
  let tan30 := sin30 / cos30 in
  tan30 + 4 * sin30 = (Real.sqrt 3 + 6) / 3 :=
by
  sorry

end tan_30_deg_plus_4_sin_30_deg_eq_l541_541439


namespace history_percentage_l541_541387

theorem history_percentage (H : ℕ) (math_percentage : ℕ := 72) (third_subject_percentage : ℕ := 69) (overall_average : ℕ := 75) :
  (math_percentage + H + third_subject_percentage) / 3 = overall_average → H = 84 :=
by
  intro h
  sorry

end history_percentage_l541_541387


namespace equation_of_circle_l541_541919

-- Defining the conditions
noncomputable def circle_center_in_second_quadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b > 0

def passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x - a)^2 + (y - b)^2 = b^2

def tangent_to_line (a b : ℝ) (L : ℝ → ℝ → ℝ) : Prop :=
  (L a b = 0) ∧ (L a (b - b) = 0)

def tangent_to_x_axis (b : ℝ) : Prop :=
  b = 10

-- The theorem we need to prove
theorem equation_of_circle :
  ∃ (a b : ℝ), circle_center_in_second_quadrant a b ∧
    passes_through_point a b (1, 2) ∧
    tangent_to_line a b (λ x y, 3*x - 4*y + 5) ∧
    tangent_to_x_axis b ∧
    (∀ x y, (x + 5)^2 + (y - 10)^2 = 100) :=
sorry

end equation_of_circle_l541_541919


namespace garden_perimeter_ratio_l541_541386

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end garden_perimeter_ratio_l541_541386


namespace standard_deviation_transformation_l541_541571

variable {α : Type*} [NormedSpace ℝ α]

theorem standard_deviation_transformation (a : ℕ → α) (S : ℝ) (n : ℕ) 
(h : ∀ (x : ℕ), x < n → ‖a x - (1 / n) * ∑ i in Finset.range n, a i‖^2 = S^2) :
  (√((∑ x in Finset.range n, ‖2 * a x - 3 • 1 - (1 / n) * ∑ i in Finset.range n, (2 * a i - 3 • 1)‖^2) / n)) = 2 * S := by
  sorry

end standard_deviation_transformation_l541_541571


namespace limit_sequence_l541_541709

theorem limit_sequence 
  (a_n : ℕ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ n, a_n n = (3 - n^2) / (4 + 2 * n^2)) 
  (h2 : a = -1 / 2) : 
  filter.tendsto a_n filter.at_top (nhds a) :=
sorry

end limit_sequence_l541_541709


namespace factorize_cubed_sub_four_l541_541455

theorem factorize_cubed_sub_four (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
by
  sorry

end factorize_cubed_sub_four_l541_541455


namespace eval_expression_l541_541753

theorem eval_expression : 
  \frac{25 * 8 + 1 / (\frac{5}{7})}{2014 - 201.4 * 2} = \frac{1}{8} := by
  sorry

end eval_expression_l541_541753


namespace radius_circle_l541_541173

-- Define the initial conditions
variables (r : ℝ) (inclination : ℝ) (radius : ℝ) 
variables (tangent : ℝ)
variables (π : ℝ)
variables (sqrt : ℝ → ℝ)

-- Assume the known values from the problem
def initial_conditions :=
  r = 10 ∧ inclination = π / 4 ∧ tangent = radius ∧ sqrt(30) = π/3

-- Prove that the radius of the circle formed by the tangency points is 7.3
theorem radius_circle (R : ℝ) : initial_conditions → R = 7.3 :=
by
  sorry

end radius_circle_l541_541173


namespace non_neg_sum_of_squares_l541_541559

theorem non_neg_sum_of_squares (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by
  sorry

end non_neg_sum_of_squares_l541_541559


namespace fraction_of_number_l541_541327

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541327


namespace isosceles_triangle_sine_of_larger_angle_l541_541166

theorem isosceles_triangle_sine_of_larger_angle (r : ℝ) (triangle_triangle : Triangle ℝ)
  (isosceles : triangle.is_isosceles)
  (base_eq_four_r : triangle.base = 4*r) :
  Real.sin triangle.larger_angle = 24 / 25 :=
by
  sorry

end isosceles_triangle_sine_of_larger_angle_l541_541166


namespace XY_parallel_X_l541_541964

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541964


namespace parallel_lines_triangle_l541_541945

theorem parallel_lines_triangle (DEF : Triangle) 
  (circle_through_EF : Circle) 
  (X Y : Point)
  (X_on_DE : X ∈ segment DEF.D DEF.E)
  (Y_on_DF : Y ∈ segment DEF.D DEF.F)
  (Y'_on_DF : ∃ r : Point, bisector_angle DEF.D E Y r ∧ Y' = r) 
  (X'_on_DE : ∃ s : Point, bisector_angle DEF.D F X s ∧ X' = s) 
  (circle_contains_XY : X ∈ circle_through_EF ∧ Y ∈ circle_through_EF)
  (circle_contains_EF : E ∈ circle_through_EF ∧ F ∈ circle_through_EF) :
  XY ∥ X'Y' :=
by
  sorry

end parallel_lines_triangle_l541_541945


namespace fraction_of_number_l541_541318

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l541_541318


namespace pictures_at_museum_l541_541786

-- Define the given conditions
def z : ℕ := 24
def k : ℕ := 14
def p : ℕ := 22

-- Define the number of pictures taken at the museum
def M : ℕ := 12

-- The theorem to be proven
theorem pictures_at_museum :
  z + M - k = p ↔ M = 12 :=
by
  sorry

end pictures_at_museum_l541_541786


namespace CindyCorrectCalculation_l541_541430

def CindyCalculation (x : ℝ) : ℝ :=
  (x - 4) / 8
  
theorem CindyCorrectCalculation :
  ∀ x : ℝ, (x - 8) / 4 = 24 → CindyCalculation x = 12.5 :=
by 
  intros x h
  sorry

end CindyCorrectCalculation_l541_541430


namespace fraction_multiplication_l541_541310

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l541_541310


namespace trace_is_ellipse_l541_541742

noncomputable def trace_shape (a b : ℝ) (h : a^2 + b^2 = 4) : set (ℝ × ℝ) :=
  {p | ∃ x y : ℝ, p = (x, y) ∧ x = (5/4 * a) ∧ y = (3/4 * b)}

theorem trace_is_ellipse (a b : ℝ) (h : a^2 + b^2 = 4) :
  trace_shape a b h = {p | ∃ x y : ℝ, p = (x, y) ∧ (x / (5/2))^2 + (y / (3/2))^2 = 1} :=
sorry

end trace_is_ellipse_l541_541742


namespace sum_of_vertices_l541_541085

theorem sum_of_vertices (pentagon_vertices : Nat := 5) (hexagon_vertices : Nat := 6) :
  (2 * pentagon_vertices) + (2 * hexagon_vertices) = 22 :=
by
  sorry

end sum_of_vertices_l541_541085


namespace marys_garbage_bill_is_correct_l541_541686

noncomputable def weekly_cost_trash_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_cost_recycling_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_total_cost (trash_cost : ℝ) (recycling_cost : ℝ) : ℝ :=
  trash_cost + recycling_cost

noncomputable def monthly_cost (weekly_cost : ℝ) (num_weeks : ℕ) : ℝ :=
  weekly_cost * num_weeks

noncomputable def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount_and_fine (total_cost : ℝ) (discount : ℝ) (fine : ℝ) : ℝ :=
  total_cost - discount + fine

theorem marys_garbage_bill_is_correct :
  let weekly_trash_cost := weekly_cost_trash_bin 10 2 in
  let weekly_recycling_cost := weekly_cost_recycling_bin 5 1 in
  let weekly_total := weekly_total_cost weekly_trash_cost weekly_recycling_cost in
  let monthly_total := monthly_cost weekly_total 4 in
  let senior_discount := discount monthly_total 0.18 in
  let fine := 20 in
  total_cost_after_discount_and_fine monthly_total senior_discount fine = 102 :=
by
  sorry

end marys_garbage_bill_is_correct_l541_541686


namespace XY_parallel_X_l541_541961

-- Given a triangle DEF
variables {D E F X Y X' Y' : Type}
variables [triangle DEF D E F]
variables (circle : Circle E F)
variables (DE DF DX' DY' DF DX DY : Line)
variables (X'Y' : Line)

-- Circle passing through vertices E and F intersects sides DE and DF at points X and Y respectively
variables (X_on_DE : X ∈ DE)
variables (X_on_circle : X ∈ circle)
variables (Y_on_DF : Y ∈ DF)
variables (Y_on_circle : Y ∈ circle)

-- Angle bisectors
variables (bisector_DEY : bisects DE Y' DF)
variables (bisector_DFX : bisects DF X' DE)

-- Proof statement
theorem XY_parallel_X'Y' (X Y X' Y' DE DF DY DX' bisector_DFX bisector_DEY):
  parallel XY X'Y' := 
sorry

end XY_parallel_X_l541_541961


namespace matrix_N_satisfies_property_l541_541450

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ :=
  !![-2, 0, 0; 0, -2, 0; 0, 0, -2]

theorem matrix_N_satisfies_property (u : ℝ × ℝ × ℝ) :
  (∀ (v : Fin 3 → ℝ), (N.mul_vec v) = (λ i, -2 * v i)) :=
by
  intros
  simp [N, Matrix.mul_vec, Matrix.dot_product, Matrix.row_vec]
  sorry

end matrix_N_satisfies_property_l541_541450


namespace find_breadth_of_plot_l541_541355

-- Define the conditions
def length_of_plot (breadth : ℝ) := 3 * breadth
def area_of_plot := 2028

-- Define what we want to prove
theorem find_breadth_of_plot (breadth : ℝ) (h1 : length_of_plot breadth * breadth = area_of_plot) : breadth = 26 :=
sorry

end find_breadth_of_plot_l541_541355


namespace count_3digit_integers_with_product_36_l541_541885

theorem count_3digit_integers_with_product_36 : 
  ∃ (n : ℕ), (n = 21) ∧ 
  ∀ (digits : list ℕ), (∀ x ∈ digits, 1 ≤ x ∧ x ≤ 9) ∧ (digits.prod = 36) → digits.length = 3 → n = 21 :=
by
  sorry

end count_3digit_integers_with_product_36_l541_541885


namespace quadratic_inequality_l541_541574

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_inequality (h : f b c (-1) = f b c 3) : f b c 1 < c ∧ c < f b c 3 :=
by
  sorry

end quadratic_inequality_l541_541574


namespace book_read_ratio_l541_541787

variables (B : ℕ) -- Number of books Brad read last month

-- Conditions
def WilliamBooksLastMonth := 6
def BradBooksThisMonth := 8 
def WilliamBooksThisMonth := 2 * BradBooksThisMonth
def WilliamBooksTotal := WilliamBooksLastMonth + WilliamBooksThisMonth
def BradBooksTotal := WilliamBooksTotal - 4
def BradBooksLastMonth := BradBooksTotal - BradBooksThisMonth

-- The Ratio we need to prove
def Ratio := BradBooksLastMonth / WilliamBooksLastMonth

theorem book_read_ratio (h : BradBooksLastMonth = B) (hB : B = 10) :
  Ratio = 5 / 3 :=
by
  rw [←hB] 
  sorry

end book_read_ratio_l541_541787


namespace fraction_of_number_l541_541300

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541300


namespace quadratic_coefficient_conversion_l541_541035

theorem quadratic_coefficient_conversion :
  ∀ x : ℝ, (3 * x^2 - 1 = 5 * x) → (3 * x^2 - 5 * x - 1 = 0) :=
by
  intros x h
  rw [←sub_eq_zero, ←h]
  ring

end quadratic_coefficient_conversion_l541_541035


namespace sum_of_possible_values_N_l541_541749

variable (a b c N : ℕ)

theorem sum_of_possible_values_N :
  (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (c = 2 * a + b) → N = 136 := 
by
  sorry

end sum_of_possible_values_N_l541_541749


namespace simplify_expression_l541_541980

theorem simplify_expression (x : ℝ) (h : x^2 + x - 6 = 0) : 
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 :=
sorry

end simplify_expression_l541_541980


namespace sequence_a4_l541_541925

theorem sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) 
  (hS : ∀ n, S n = (n + 1) / (n + 2))
  (hS0 : S 0 = a 0)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 4 = 1 / 30 := 
sorry

end sequence_a4_l541_541925


namespace total_cost_eq_420_52_l541_541041

/-- 
Service fee per vehicle: $2.10
Mini-van:
- Base capacity: 65 liters
- Price per liter: $0.75
Truck:
- Base capacity: 120% larger than mini-van base (65*2.2) = 143 liters
- Price per liter: $0.85
- Truck 1: Standard capacity (143 liters)
- Truck 2: 15% larger than standard
-/
theorem total_cost_eq_420_52:
  let service_fee := 2.10
  let mini_van_base_capacity := 65.0
  let mini_van_price_per_liter := 0.75
  let truck_base_capacity := mini_van_base_capacity * 2.2
  let truck_price_per_liter := 0.85

  let mini_van_1_capacity := mini_van_base_capacity * 1.10
  let mini_van_2_capacity := mini_van_base_capacity * 0.95
  let mini_van_3_capacity := mini_van_base_capacity

  let truck_1_capacity := truck_base_capacity
  let truck_2_capacity := truck_base_capacity * 1.15

  let mini_van_1_cost := (mini_van_1_capacity * mini_van_price_per_liter) + service_fee
  let mini_van_2_cost := (mini_van_2_capacity * mini_van_price_per_liter) + service_fee
  let mini_van_3_cost := (mini_van_3_capacity * mini_van_price_per_liter) + service_fee

  let truck_1_cost := (truck_1_capacity * truck_price_per_liter) + service_fee
  let truck_2_cost := (truck_2_capacity * truck_price_per_liter) + service_fee

  let total_cost := mini_van_1_cost + mini_van_2_cost + mini_van_3_cost + truck_1_cost + truck_2_cost
  in total_cost = 420.52 := 
sorry

end total_cost_eq_420_52_l541_541041


namespace quadratic_roots_equation_l541_541294

theorem quadratic_roots_equation (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 12) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C (-11) 
  : ℝ[X]).is_root x ∧ (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X
  + Polynomial.C (-11) : ℝ[X]).is_root y := 
sorry

end quadratic_roots_equation_l541_541294


namespace tan_30_deg_plus_4_sin_30_deg_eq_l541_541440

theorem tan_30_deg_plus_4_sin_30_deg_eq :
  let sin30 := 1 / 2 in
  let cos30 := Real.sqrt 3 / 2 in
  let tan30 := sin30 / cos30 in
  tan30 + 4 * sin30 = (Real.sqrt 3 + 6) / 3 :=
by
  sorry

end tan_30_deg_plus_4_sin_30_deg_eq_l541_541440


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541514

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541514


namespace count_incorrect_propositions_l541_541843

def proposition1 := ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0
def proposition2 (f : ℝ → ℝ) := (∀ x, f(x + 2) = -f(x)) → (∀ x, f(x - 4) = f(x))
def proposition3 := (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → real.exp x ≥ 1) ∨ ∃ x : ℝ, x^2 + x + 1 < 0
def proposition4 (a : ℝ) := a = -1 → ∃ x : ℝ, a*x^2 + 2*x - 1 = 0

theorem count_incorrect_propositions :
  (¬(proposition1) ∧ ¬(proposition2 (λ x, 0)) ∧ ¬(proposition3) ∧ ¬(proposition4 (-1))) = 1 :=
sorry

end count_incorrect_propositions_l541_541843


namespace power_equation_l541_541425

theorem power_equation : 8^((1 : ℝ) / 3) + 2^(4 + Real.logb 2 3) = 50 := by
  sorry

end power_equation_l541_541425


namespace XY_parallel_X_l541_541968

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541968


namespace XY_parallel_X_l541_541937

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541937


namespace largest_n_dividing_30_factorial_l541_541487

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541487


namespace XY_parallel_X_l541_541930

-- Declaration of the points and their relationships based on given conditions
variable (D E F X Y X' Y' : Type)
variable (DE DF : Set (D → X))
variable (circle_passes_through_EF : Circle E F)
variable (circle_intersects_X : X ∈ circle_passes_through_EF ∩ DE)
variable (circle_intersects_Y : Y ∈ circle_passes_through_EF ∩ DF)
variable (angle_bisector_EXY : line (angle.bisector_of ∠ DEY) ∩ DF = set.singleton Y')
variable (angle_bisector_FXY : line (angle.bisector_of ∠ DFX) ∩ DE = set.singleton X')

-- The statement to prove that XY is parallel to X'Y'
theorem XY_parallel_X'Y' :
  ∥ line_through X Y ∥ line_through X' Y' :=
sorry

end XY_parallel_X_l541_541930


namespace claire_shirts_proof_l541_541245

theorem claire_shirts_proof : 
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := 
by
  intro brian_shirts andrew_shirts steven_shirts claire_shirts
  intros h_brian h_andrew h_steven h_claire
  sorry

end claire_shirts_proof_l541_541245


namespace buratino_candy_spent_l541_541852

theorem buratino_candy_spent :
  ∃ (x y : ℕ), x + y = 50 ∧ 2 * x = 3 * y ∧ y * 5 - x * 3 = 10 :=
by {
  -- Declaration of variables and goal
  let x := 30,
  let y := 20,
  use [x, y],
  split,
  { exact rfl },
  split,
  { exact rfl },
  { exact rfl }
}

end buratino_candy_spent_l541_541852


namespace solution_range_l541_541581

variable {f : ℝ → ℝ}

-- Definitions corresponding to the conditions
def f_neg (x : ℝ) : Prop := f x < 0 → x > 0
def f_nonneg (x : ℝ) : Prop := f x ≥ 0 → x ≤ 1

theorem solution_range :
  (∀ x, f (log (x^2 - 6 * x + 20)) < 0 ∧ f ((2 * x + 1) / (x - 1)) ≥ 0) →
  { x : ℝ | f ((2 * x^2 - x - 1) / (x^2 - 2 * x + 1)) * f (log (x^2 - 6 * x + 20)) ≤ 0 } = set.Ico (-2 : ℝ) 1 :=
sorry

end solution_range_l541_541581


namespace function_sum_l541_541567

noncomputable def f : ℝ → ℝ := sorry

axiom function_property (a b : ℝ) : f(a + b) = f(a) * f(b)
axiom function_value : f(1) = 2

theorem function_sum : (finset.range 1009).sum (λ n, f(2 * n + 2) / f(2 * n + 1)) = 2018 := 
by { sorry }

end function_sum_l541_541567


namespace factorization_of_polynomial_l541_541892

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l541_541892


namespace broken_line_points_in_square_l541_541188

noncomputable def distance (p1 p2 : Point) : ℝ := sorry
noncomputable def distance_along_L (L : list Point) (p1 p2 : Point) : ℝ := sorry
noncomputable def in_square (p : Point) (side_len : ℝ) : Prop := sorry
noncomputable def close_to_L (square : list Point) (L : list Point) (eps : ℝ): Prop := sorry

theorem broken_line_points_in_square :
  ∀ (square : list Point) (L : list Point),
  length (square) = 4 →
  all_points_close_to_L (square) (L) 0.5 →
  ∃ (F1 F2 : Point), F1 ∈ L ∧ F2 ∈ L ∧ distance F1 F2 ≤ 1 ∧ distance_along_L L F1 F2 ≥ 198 :=
by
  sorry

end broken_line_points_in_square_l541_541188


namespace number_of_special_permutations_l541_541669

-- Define the conditions as variables and hypotheses
def is_special_permutation (a : Fin 18 → Fin 19) : Prop :=
  ∃ (idx_lt : Fin 9 → Fin 18) (idx_gt : Fin 9 → Fin 18), 
    (∀ i j, i < j → (a (idx_lt i)).val > (a (idx_lt j)).val) ∧
    (a (idx_lt 8)).val = 1 ∧
    (∀ i j, i < j → (a (idx_gt i)).val < (a (idx_gt j)).val) ∧
    (∀ i, a i ∈ (Finset.range 1 19).erase 1) 

theorem number_of_special_permutations : 
  (Finset.univ.filter is_special_permutation).card = 24310 :=
by
  sorry

end number_of_special_permutations_l541_541669


namespace area_of_ABC_l541_541162

-- Lean 4 statement representing the problem described
theorem area_of_ABC :
  ∀ (A B C D H : Type) (AB BC AC CH CD : ℝ),
  (triangle_is_isosceles A B C) ∧
  (AB = 2) ∧
  (BC = 2) ∧
  (AD_is_angle_bisector A B C D) ∧
  (point_on_line D B C) ∧
  (tangent_to_circumcircle DH A D B) ∧
  (point_on_line H A C) ∧
  (CD = sqrt(2) * CH)
  → area_triangle A B C = sqrt(4 * sqrt(2) - 5) :=
begin
  sorry
end

end area_of_ABC_l541_541162


namespace varphi_value_l541_541292

theorem varphi_value :
  ∃ (φ : ℝ), (∀ x : ℝ, f x = 2 * Real.sin (x + 2 * φ)) ∧ abs φ < Real.pi / 2 ∧
    (∀ x : ℝ, ∃ k : ℤ, Real.sin (x + (Real.pi / 2) + 2 * φ) - Real.sin (2 * k * Real.pi + (Real.pi / 2)) = 0) ∧
    (2 * Real.sin (2 * φ) > 0) →
    φ = 3 * Real.pi / 8 :=
sorry

end varphi_value_l541_541292


namespace odd_divisors_count_l541_541598

theorem odd_divisors_count :
    {n : ℕ | 0 < n ∧ n < 150 ∧ (∃ k : ℕ, n = k^2)}.to_finset.card = 12 :=
by
  sorry

end odd_divisors_count_l541_541598


namespace locus_of_center_of_circle_c_is_parabola_l541_541216

noncomputable def circle_c_tangent_to_line_and_circle := 
  ∃ p : Point, 
    TangentToCircle p (0, 3) 1 ∧ 
    TangentToLine p 0 

theorem locus_of_center_of_circle_c_is_parabola :
  ∀ p, circle_c_tangent_to_line_and_circle p →
    IsParabola (LocusOfCenter p) :=
by
  sorry

end locus_of_center_of_circle_c_is_parabola_l541_541216


namespace XY_parallel_X_l541_541970

-- Define the setup for the problem
variables {D E F X Y X' Y' : Type} [metric_space D] [metric_space E] [metric_space F]
          [metric_space X] [metric_space Y] [metric_space X'] [metric_space Y']
          [is_triangle D E F]
          (γ : circle (metric_space.point E) (metric_space.point F))
          (hX : γ.intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X))
          (hY : γ.intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y))
          (bisector_DEY_intersects_DF_at_Y' : angle.bisector (∠ (metric_space.point D) (metric_space.point E) (metric_space.point Y)).intersects (segment (metric_space.point D) (metric_space.point F)) (metric_space.point Y'))
          (bisector_DFX_intersects_DE_at_X' : angle.bisector (∠ (metric_space.point D) (metric_space.point F) (metric_space.point X)).intersects (segment (metric_space.point D) (metric_space.point E)) (metric_space.point X'))

-- State the theorem to prove
theorem XY_parallel_X'Y' : parallel (line_through (metric_space.point X) (metric_space.point Y)) 
                                      (line_through (metric_space.point X') (metric_space.point Y')) :=
sorry

end XY_parallel_X_l541_541970


namespace largest_power_of_18_dividing_factorial_30_l541_541493

theorem largest_power_of_18_dividing_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 7 :=
by
  sorry

end largest_power_of_18_dividing_factorial_30_l541_541493


namespace sum_of_roots_l541_541281

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l541_541281


namespace mean_median_difference_is_minus_4_l541_541158

-- Defining the percentages of students scoring specific points
def perc_60 : ℝ := 0.20
def perc_75 : ℝ := 0.55
def perc_95 : ℝ := 0.10
def perc_110 : ℝ := 1 - (perc_60 + perc_75 + perc_95) -- 0.15

-- Defining the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_95 : ℝ := 95
def score_110 : ℝ := 110

-- Calculating the mean score
def mean_score : ℝ := (perc_60 * score_60) + (perc_75 * score_75) + (perc_95 * score_95) + (perc_110 * score_110)

-- Given the median score
def median_score : ℝ := score_75

-- Defining the expected difference
def expected_difference : ℝ := mean_score - median_score

theorem mean_median_difference_is_minus_4 :
  expected_difference = -4 := by sorry

end mean_median_difference_is_minus_4_l541_541158


namespace XY_parallel_X_l541_541936

theorem XY_parallel_X'Y' :
  ∀ {D E F X Y Y' X' : Type} [linear_order E] [linear_order F]
  (h_circle : circle_through E F)
  (h_X : X ∈ (line_through D E) ∧ X ∈ h_circle)
  (h_Y : Y ∈ (line_through D F) ∧ Y ∈ h_circle)
  (h_Y' : Y' ∈ (internal_bisector ∠(D, E, Y)) ∩ (line_through D F))
  (h_X' : X' ∈ (internal_bisector ∠(D, F, X)) ∩ (line_through D E)),
  parallel (line_through X Y) (line_through X' Y') := 
sorry

end XY_parallel_X_l541_541936


namespace polygon_sides_l541_541612

theorem polygon_sides (h : 1440 = (n - 2) * 180) : n = 10 := 
by {
  -- Here, the proof would show the steps to solve the equation h and confirm n = 10
  sorry
}

end polygon_sides_l541_541612


namespace sin_double_angle_identity_l541_541537

theorem sin_double_angle_identity (α : ℝ) (h : sin α + cos α = 1 / 3) : 
    sin (2 * α) = -8 / 9 :=
sorry

end sin_double_angle_identity_l541_541537


namespace quadratic_representation_integer_coefficients_iff_l541_541802

theorem quadratic_representation (A B C x : ℝ) :
  ∃ (k l m : ℝ), 
  k = 2 * A ∧ l = A + B ∧ m = C ∧ 
  (A * x^2 + B * x + C = k * (x * (x - 1) / 2) + l * x + m) :=
sorry

theorem integer_coefficients_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ (n : ℤ), A * x^2 + B * x + C = n) ↔ 
  (∃ (k l m : ℤ), k = 2 * A ∧ l = A + B ∧ m = C) :=
sorry

end quadratic_representation_integer_coefficients_iff_l541_541802


namespace sum_of_roots_is_18_l541_541217

theorem sum_of_roots_is_18 {f : ℝ → ℝ}
  (H_symm : ∀ x : ℝ, f (3 + x) = f (3 - x))
  (H_roots : (∃ s : Finset ℝ, s.card = 6 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x y ∈ s, x ≠ y → f x = 0 ∧ f y = 0 )) :
  (Finset.sum (s : Finset ℝ) id) = 18 :=
sorry

end sum_of_roots_is_18_l541_541217


namespace correlation_coefficient_l541_541179

theorem correlation_coefficient (variation_explained_by_height : ℝ)
    (variation_explained_by_errors : ℝ)
    (total_variation : variation_explained_by_height + variation_explained_by_errors = 1)
    (percentage_explained_by_height : variation_explained_by_height = 0.71) :
  variation_explained_by_height = 0.71 := 
by
  sorry

end correlation_coefficient_l541_541179


namespace angle_bisector_of_right_angle_triangle_l541_541705

theorem angle_bisector_of_right_angle_triangle 
  (A B C O : Point) 
  (h_triangle : ∃ (A B C : Point), right_triangle A B C ∧ hypotenuse A B C = AB)
  (h_square : square_on_hypotenuse_ext_center AB A B C O) 
  : angle_bisector CO ACB :=
sorry

end angle_bisector_of_right_angle_triangle_l541_541705


namespace shaded_area_eq_l541_541411

open EuclideanGeometry
open Real

/-- Definitions for the problem setup -/
def pointA : Point := { x := 2, y := 0 }
def pointO : Point := { x := 0, y := 0 }
def radiusO : ℝ := 1
def distanceOA : ℝ := dist pointO pointA

def is_tangent (p1 p2 : Point) (c : Circle) (line : Line) : Prop :=
  dist p1 p2 = radius c ∧ dist p2 (closest_pnt p2 line) = radius c

def circleO : Circle := { center := pointO, radius := radiusO }

noncomputable def pointB : Point := 
  closest_pnt pointA (tangent_to circleO pointA)

noncomputable def chordBC : Chord := 
  { p1 := pointB, p2 := reflect_over pointB pointA }

noncomputable def pointC : Point := chordBC.p2

/-- The Lean statement to prove the area of the shaded region -/
theorem shaded_area_eq :
  area (sector circleO pointB pointC) = π / 6 :=
by
  sorry

end shaded_area_eq_l541_541411


namespace total_amount_proof_l541_541370

variable (A : ℝ) -- Amount lent at 5% interest
variable (T : ℝ) -- Total amount of money
variable (income : ℝ) -- Total yearly annual income

noncomputable def total_amount (A T income : ℝ) : Prop := 
  0.05 * A + 0.06 * (T - A) = income

theorem total_amount_proof (hA : A = 1000.0000000000005)
                          (hincome : income = 140) : 
  total_amount A 2500 income :=
by
  rw [hA, hincome]
  unfold total_amount
  linarith

end total_amount_proof_l541_541370


namespace problem_l541_541442

theorem problem
: 15 * (1 / 17) * 34 = 30 := by
  sorry

end problem_l541_541442


namespace eggs_in_nests_l541_541658

theorem eggs_in_nests (x : ℕ) (h1 : 2 * x + 3 + 4 = 17) : x = 5 :=
by
  /- This is where the proof would go, but the problem only requires the statement -/
  sorry

end eggs_in_nests_l541_541658


namespace fraction_of_number_l541_541328

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l541_541328


namespace inequality_b_does_not_hold_l541_541142

theorem inequality_b_does_not_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬(a + d > b + c) ↔ a + d ≤ b + c :=
by
  -- We only need the statement, so we add sorry at the end
  sorry

end inequality_b_does_not_hold_l541_541142


namespace correct_expression_l541_541783

theorem correct_expression (a b : ℝ) : (a - b) * (b + a) = a^2 - b^2 :=
by
  sorry

end correct_expression_l541_541783


namespace num_distinct_monograms_l541_541137

theorem num_distinct_monograms : (finset.card {s : finset char | s.card = 3 ∧ ∀ a b : char, (a ∈ s ∧ b ∈ s ∧ a < b → true)}) = 2600 :=
sorry

end num_distinct_monograms_l541_541137


namespace exponential_function_option_D_l541_541345

def is_exponential (f : ℝ → ℝ) : Prop := ∃ (a b : ℝ), b ≠ 0 ∧ b > 0 ∧ ∀ x, f x = a * (b ^ x)

theorem exponential_function_option_D :
  let A := (λ (x : ℝ), 2^(x+1)),
      B := (λ (x : ℝ), x^3),
      C := (λ (x : ℝ), 3 * 2^x),
      D := (λ (x : ℝ), 3^(-x))
  in is_exponential D :=
by
  sorry

end exponential_function_option_D_l541_541345


namespace total_jackets_sold_l541_541011

-- Define the conditions
def price_before_noon : ℝ := 31.95
def price_after_noon : ℝ := 18.95
def total_receipts : ℝ := 5108.30
def jackets_sold_after_noon : ℕ := 133

-- Define the result to be proven
theorem total_jackets_sold (x : ℕ) (y : ℝ) (z : ℝ) :
  x = jackets_sold_after_noon →
  y = price_before_noon →
  z = total_receipts - (price_after_noon * jackets_sold_after_noon) / price_before_noon →
  x + z.to_nat = 214 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_jackets_sold_l541_541011


namespace pizzas_successfully_served_l541_541827

theorem pizzas_successfully_served (served returned : ℕ) (h_served : served = 9) (h_returned : returned = 6) : 
    served - returned = 3 :=
by
  rw [h_served, h_returned]
  exact Nat.sub_self 6

end pizzas_successfully_served_l541_541827


namespace find_a_b_plus_2023_l541_541981

theorem find_a_b_plus_2023 (a b : ℝ) (h : 1^2 + a * 1 - b = 0) : a - b + 2023 = 2022 :=
by
  have h1 : 1 + a - b = 0 := by simp [h]
  have h2 : a - b = -1 := by linarith [h1]
  linarith [h2]

end find_a_b_plus_2023_l541_541981


namespace sum_f_1_to_2018_l541_541014

def f : ℝ → ℝ :=
  λ x, if -3 ≤ x ∧ x < -1 then -(x + 2)^2 else
       if -1 ≤ x ∧ x < 3 then x else
       f (x - 6)

theorem sum_f_1_to_2018 :
  (∑ k in Finset.range 2018, f (k + 1)) = 338 :=
by
  sorry

end sum_f_1_to_2018_l541_541014


namespace sqrt_area_inequality_l541_541700

noncomputable theory

variables {A B C D O L K M N : Type} [ordered_field A] 
variables (ABCD AKON LOMC : ℝ) (k k₁ k₂ : ℝ)

-- Given conditions
def point_inside_quadrilateral (O : A) (ABCD : set A) : Prop :=
∃ A B C D, O ∈ convex_hull A B C D

def lines_parallel (O : A) (P Q : set A) (R S : set A) : Prop :=
parallel (line_through O P) (line_through Q R) ∧ parallel (line_through O R) (line_through Q S)

-- Define the areas of convex quadrilateral and its partitions
def area (S : set A) : ℝ := sorry

-- Main theorem to prove: √k ≥ √k₁ + √k₂
theorem sqrt_area_inequality (hO: point_inside_quadrilateral O ABCD)
  (hP1: lines_parallel O A B C L)
  (hP2: lines_parallel O A K O N)
  (hP3: lines_parallel O A D C M)
  (hP4: lines_parallel O M N D A)
  (h_area: k = area (ABCD)) (h_area1: k₁ = area (AKON)) (h_area2: k₂ = area (LOMC))
  : real.sqrt k ≥ real.sqrt k₁ + real.sqrt k₂ := sorry

end sqrt_area_inequality_l541_541700


namespace find_m_l541_541262

theorem find_m (n : ℝ) (m : ℝ) (h : n > 1) (h_pass : ∀ x, f x = log n (x + m)) : 
  f (-2) = 0 → m = 3 :=
by
  sorry

end find_m_l541_541262


namespace max_sum_of_factors_48_l541_541204

theorem max_sum_of_factors_48 (clubsuit heartsuit : ℕ) (h : clubsuit * heartsuit = 48) : 
               clubsuit + heartsuit ≤ 49 :=
begin
  -- This part is left as an exercise or proof
  sorry
end

end max_sum_of_factors_48_l541_541204


namespace count_letters_with_both_l541_541622

theorem count_letters_with_both (a b c x : ℕ) 
  (h₁ : a = 24) 
  (h₂ : b = 7) 
  (h₃ : c = 40) 
  (H : a + b + x = c) : 
  x = 9 :=
by {
  -- Proof here
  sorry
}

end count_letters_with_both_l541_541622


namespace final_number_after_increase_l541_541366

-- Define the original number and the percentage increase
def original_number : ℕ := 70
def increase_percentage : ℝ := 0.50

-- Define the function to calculate the final number after the increase
def final_number : ℝ := original_number * (1 + increase_percentage)

-- The proof statement that the final number is 105
theorem final_number_after_increase : final_number = 105 :=
by
  sorry

end final_number_after_increase_l541_541366


namespace mary_garbage_bill_l541_541690

theorem mary_garbage_bill :
  let weekly_cost := 2 * 10 + 1 * 5,
      monthly_cost := weekly_cost * 4,
      discount := 0.18 * monthly_cost,
      discounted_monthly_cost := monthly_cost - discount,
      fine := 20,
      total_bill := discounted_monthly_cost + fine
  in total_bill = 102 :=
by
  let weekly_cost := 2 * 10 + 1 * 5
  let monthly_cost := weekly_cost * 4
  let discount := 0.18 * monthly_cost
  let discounted_monthly_cost := monthly_cost - discount
  let fine := 20
  let total_bill := discounted_monthly_cost + fine
  show total_bill = 102 from sorry

end mary_garbage_bill_l541_541690


namespace express_train_meets_count_l541_541829

-- Defining the parameters as given in the problem
def travel_time_minutes : ℕ := 3 * 60 + 30
def departure_interval_minutes : ℕ := 60
def first_departure_A_time_minutes : ℕ := 6 * 60
def first_departure_B_time_minutes : ℕ := 6 * 60
def departure_A_time_minutes : ℕ := 9 * 60

-- Define the total travel time after the 9:00 AM departure from A
def arrival_time_B_minutes : ℕ := departure_A_time_minutes + travel_time_minutes

-- Define the Lean statement to prove
theorem express_train_meets_count :
  ∀ (travel_time_minutes departure_interval_minutes first_departure_A_time_minutes first_departure_B_time_minutes departure_A_time_minutes arrival_time_B_minutes : ℕ),
    travel_time_minutes = 210 →
    departure_interval_minutes = 60 →
    first_departure_A_time_minutes = 360 →
    first_departure_B_time_minutes = 360 →
    departure_A_time_minutes = 540 →
    arrival_time_B_minutes = 750 →
    (arrival_time_B_minutes - first_departure_B_time_minutes) / departure_interval_minutes - 1 = 6 :=
by
  intros travel_time_minutes departure_interval_minutes first_departure_A_time_minutes first_departure_B_time_minutes departure_A_time_minutes arrival_time_B_minutes
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  exact dec_trivial

-- Sorry to skip proof
lemma express_train_meets_six_times :
  (arrival_time_B_minutes - first_departure_B_time_minutes) / departure_interval_minutes - 1 = 6 :=
sorry

end express_train_meets_count_l541_541829


namespace trapezoidal_section_length_l541_541834

theorem trapezoidal_section_length 
  (total_area : ℝ) 
  (rectangular_area : ℝ) 
  (parallel_side1 : ℝ) 
  (parallel_side2 : ℝ) 
  (trapezoidal_area : ℝ)
  (H1 : total_area = 55)
  (H2 : rectangular_area = 30)
  (H3 : parallel_side1 = 3)
  (H4 : parallel_side2 = 6)
  (H5 : trapezoidal_area = total_area - rectangular_area) :
  (trapezoidal_area = 25) → 
  (1/2 * (parallel_side1 + parallel_side2) * L = trapezoidal_area) →
  L = 25 / 4.5 :=
by
  sorry

end trapezoidal_section_length_l541_541834


namespace largest_divisor_18n_max_n_l541_541478

theorem largest_divisor_18n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n ≤ 7 :=
by
  have h1 : 18 = 2 * 3^2 := by norm_num
  have factorial_30 := nat.factorial 30
  have h2 : (∃ n, 18^n = (2^n * (3^2)^n)) := 
    by existsi n; rw [pow_eq_pow, h1, pow_mul]
  have two_factor := nat.factors_in_factorial 30 2
  have three_factor := nat.factors_in_factorial 30 3
  sorry

theorem max_n (n : ℕ) : ∀ n, 18^n ∣ nat.factorial 30 → n = 7 :=
by sorry

end largest_divisor_18n_max_n_l541_541478


namespace sage_can_verify_with_one_weighing_l541_541264

theorem sage_can_verify_with_one_weighing (weights : Fin 7 → ℕ) (bags : Fin 7 → Fin 100 → ℕ) (indicated : Fin 7):
  (∀ i, weights i ∈ {7, 8, 9, 10, 11, 12, 13}) → 
  (∀ i j, i ≠ j → bags i j = weights i) → 
  ∃ (A : Fin 7 → Fin 1) , (A indicated = 7) :=
sorry

end sage_can_verify_with_one_weighing_l541_541264


namespace area_difference_l541_541175

theorem area_difference (AH HB FC : ℝ) (hₐ : AH = 6) (h_b : HB = 3) (h_𝐹 : FC = 9) 
    (right_angle1 : ∠ACD = 90°) (right_angle2 : ∠HBC = 90°) : 
    let area_AFH := (1 / 2) * AH * FC
        area_BHC := (1 / 2) * HB * FC
        x := area_AFH - area_BHC
    in x = 13.5 :=
by
  sorry

end area_difference_l541_541175


namespace faye_rows_l541_541457

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (rows_created : ℕ) :
  total_pencils = 12 → pencils_per_row = 4 → rows_created = 3 := by
  sorry

end faye_rows_l541_541457


namespace tiger_distance_proof_l541_541032

-- Declare the problem conditions
def tiger_initial_speed : ℝ := 25
def tiger_initial_time : ℝ := 3
def tiger_slow_speed : ℝ := 10
def tiger_slow_time : ℝ := 4
def tiger_chase_speed : ℝ := 50
def tiger_chase_time : ℝ := 0.5

-- Compute individual distances
def distance1 := tiger_initial_speed * tiger_initial_time
def distance2 := tiger_slow_speed * tiger_slow_time
def distance3 := tiger_chase_speed * tiger_chase_time

-- Compute the total distance
def total_distance := distance1 + distance2 + distance3

-- The final theorem to prove
theorem tiger_distance_proof : total_distance = 140 := by
  sorry

end tiger_distance_proof_l541_541032


namespace fraction_of_number_l541_541299

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541299


namespace solution_set_of_inequality_l541_541759

theorem solution_set_of_inequality (x : ℝ) : (|2 * x - 1| < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_of_inequality_l541_541759


namespace parking_allocation_methods_l541_541401

theorem parking_allocation_methods : 
  let total_spaces := 6
  let total_companies := 4
  let pairs := 2
  (total_companies - pairs) + pairs = 4 → 24 :=
by
  let total_methods := 24
  sorry

end parking_allocation_methods_l541_541401


namespace roots_of_quadratic_eq_l541_541754

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l541_541754


namespace eventual_zero_stable_positions_4n_minimum_non_stable_non_zero_l541_541360

-- Define the game state and operations
structure GameState where
  a : Nat
  b : Nat
  c : Nat
  next_to_move : Fin 3
  deriving DecidableEq, Repr

def move (state : GameState) : GameState :=
  match state.next_to_move, state with
  | 0,  {a, b, c, _} =>
    if a % 2 == 0 then
      {a := a / 2, b := b + a / 2, c, next_to_move := 1}
    else
      {a := a - 1, b, c, next_to_move := 1}
  | 1,  {a, b, c, _} =>
    if b % 2 == 0 then
      {a, b := b / 2, c := c + b / 2, next_to_move := 2}
    else
      {a, b := b - 1, c, next_to_move := 2}
  | 2,  {a, b, c, _} =>
    if c % 2 == 0 then
      {a := a + c / 2, b, c := c / 2, next_to_move := 0}
    else
      {a, b, c := c - 1, next_to_move := 0}
  | _, _ => {a, b, c, next_to_move := 0}

noncomputable def iterate_moves (state : GameState) (steps : Nat) : GameState :=
  Nat.iterate steps move state

-- Problem (a)
theorem eventual_zero (n : Nat) : iterate_moves {a := 1, b := 2, c := 2, next_to_move := 0} n = {a := 0, b := 0, c := 0, next_to_move := 0} :=
  sorry

-- Define stable position
def is_stable (state : GameState) : Prop :=
  iterate_moves state 3 = state

-- Problem (b)
theorem stable_positions_4n (state : GameState) (h : is_stable state) : ∃ n : Nat, state.a + state.b + state.c = 4 * n :=
  sorry

-- Problem (c)
theorem minimum_non_stable_non_zero (coins : Nat) (state : GameState) (h1 : state.a + state.b + state.c = coins)
  (h2 : ¬is_stable state)
  (h3 : ∀ n, iterate_moves state n ≠ {a := 0, b := 0, c := 0, next_to_move := 0}) : coins = 9 :=
  sorry

end eventual_zero_stable_positions_4n_minimum_non_stable_non_zero_l541_541360


namespace geometric_sequence_arithmetic_condition_l541_541547

theorem geometric_sequence_arithmetic_condition (q : ℝ) (hq : q^2 = 1)
  (a : ℕ → ℝ) (ha6 : a 6 = 2)
  (hgeo : ∀ n, a (n + 1) = a n * q) :
  a 4 = 2 :=
by
  have h_a5 : a 5 = a 6 / q := by rw [ha6, (hgeo 5)]; ring
  have h_a7 : a 7 = a 6 * q := by rw [ha6, (hgeo 6)]; ring
  have h_a9 : a 9 = a 6 * (q ^ 3) := by rw [ha6, (hgeo 6), (hgeo 7)]; ring
  have h_arith_seq := (h_a7, h_a5, h_a9, hq) -- arithmetic sequence condition
  apply h_a4 -- Hence a_4 = 2
  sorry

end geometric_sequence_arithmetic_condition_l541_541547


namespace triangle_side_ratios_l541_541186

-- Define the necessary assumptions and structures for the problem
structure Triangle :=
  (A B C : Point)
  (is_right_angled_at : is_right_angled B)
  (altitude_median_A : forms_triangle_with_altitude_median A)
  (altitude_median_B : forms_triangle_with_altitude_median B)

-- Define the Point type and related properties
axiom Point : Type
axiom is_right_angled : Point → Prop
axiom forms_triangle_with_altitude_median : Point → Prop

-- Define the proof statement
theorem triangle_side_ratios (ΔABC : Triangle) : 
  side_ratio ΔABC = (1, 2 * sqrt 2, 3) := 
sorry

end triangle_side_ratios_l541_541186


namespace trapezoid_area_correct_l541_541330

def point (x y : ℝ) := (x, y)

def line_eq_1 (p : ℝ × ℝ) := p.2 = p.1  -- y = x
def line_eq_2 (p : ℝ × ℝ) := p.2 = 8    -- y = 8
def line_eq_3 (p : ℝ × ℝ) := p.2 = 3    -- y = 3
def y_axis (p : ℝ × ℝ) := p.1 = 0       -- x = 0

noncomputable def vertex1 := point 8 8
noncomputable def vertex2 := point 3 3
noncomputable def vertex3 := point 0 8
noncomputable def vertex4 := point 0 3

def base1 : ℝ := 3
def base2 : ℝ := 8
def height : ℝ := 8 - 3

noncomputable def trapezoid_area := 0.5 * (base1 + base2) * height

theorem trapezoid_area_correct : trapezoid_area = 27.5 := by
  simp [trapezoid_area, base1, base2, height]
  sorry

end trapezoid_area_correct_l541_541330


namespace calculate_x_l541_541424

theorem calculate_x : 121 + 2 * 11 * 8 + 64 = 361 :=
by
  sorry

end calculate_x_l541_541424


namespace percent_decrease_apr_to_may_l541_541750

theorem percent_decrease_apr_to_may (P : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → (1.35 * P = P + 0.35 * P))
  (h2 : ∀ x : ℝ, P * (1.35 * (1 - x / 100) * 1.5) = 1.62000000000000014 * P)
  (h3 : 0 < x ∧ x < 100)
  : x = 20 :=
  sorry

end percent_decrease_apr_to_may_l541_541750


namespace lambda_range_l541_541576

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem lambda_range (λ : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                   x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
                   (∀ x ≠ 0, f x =/ is real) ∧
                   Real.sqrt (f x) + 2 / Real.sqrt (f x) - λ = 0)
   ↔
   λ > Real.exp 1 + 2 / Real.exp 1 :=
sorry

end lambda_range_l541_541576


namespace units_digit_product_l541_541909

theorem units_digit_product (a b : ℕ) (ha : a % 10 = 7) (hb : b % 10 = 4) :
  (a * b) % 10 = 8 := 
by
  sorry

end units_digit_product_l541_541909


namespace min_nails_fix_convex_polygon_l541_541010

-- Definition of a convex polygon
structure ConvexPolygon (α : Type) [LinearOrderedField α] :=
(vertices : List (α × α))
(is_convex : ∀ (x y z : α × α), x ∈ vertices → y ∈ vertices → z ∈ vertices → z ∈ LineSegment x y → z ∈ vertices)

-- Definition of a fixing set of nails
def is_fixing_set {α : Type} [LinearOrderedField α] (P : ConvexPolygon α) (nails : List (α × α)) : Prop :=
∀ (x y : α × α), x ≠ y → ∃ (n1 n2 n3 n4 : (α × α)), 
  n1 ∈ nails ∧ n2 ∈ nails ∧ n3 ∈ nails ∧ n4 ∈ nails ∧ 
  (n1.1 ≤ x.1 ∧ n1.2 ≤ x.2 ∨ n1.1 ≥ x.1 ∧ n1.2 ≥ x.2) ∧ 
  (n2.1 ≤ y.1 ∧ n2.2 ≤ y.2 ∨ n2.1 ≥ y.1 ∧ n2.2 ≥ y.2) ∧ 
  (n3.1 ≤ y.1 ∧ n3.2 ≤ y.2 ∨ n3.1 ≥ y.1 ∧ n3.2 ≥ y.2) ∧ 
  (n4.1 ≤ x.1 ∧ n4.2 ≤ x.2 ∨ n4.1 ≥ x.1 ∧ n4.2 ≥ x.2)

-- Minimum number of nails required to fix any convex polygon
theorem min_nails_fix_convex_polygon {α : Type} [LinearOrderedField α] (P : ConvexPolygon α) :
  ∃ (nails : List (α × α)), is_fixing_set P nails ∧ nails.length = 4 := 
sorry

end min_nails_fix_convex_polygon_l541_541010


namespace quadrilateral_parallelogram_l541_541801

variables {A B C D E : Type}
variables [linear_ordered_field A]
variables [add_comm_group B]
variables [module A B]

def is_midpoint (E : B) (A B: B) : Prop := (E -ᵥ A) = (B -ᵥ E)

def same_area (u v: B) (α: A) : Prop := 
  (1 / 2) * u +ᵥ α * (v -ᵥ u) = (1 / 2) * α * ∥v -ᵥ u∥

theorem quadrilateral_parallelogram 
  (A B C D E : B) 
  (h1 : is_midpoint E A C) 
  (h2 : is_midpoint E B D) 
  : (same_area A E B) ∧ (same_area C E D) ∧ 
    (same_area A E D) ∧ (same_area B E C)
  → parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l541_541801


namespace option_A_option_B_option_C_option_D_verify_options_l541_541785

open Real

-- Option A: Prove the maximum value of x(6-x) given 0 < x < 6 is 9.
theorem option_A (x : ℝ) (h1 : 0 < x) (h2 : x < 6) : 
  ∃ (max_value : ℝ), max_value = 9 ∧ ∀(y : ℝ), 0 < y ∧ y < 6 → y * (6 - y) ≤ max_value :=
sorry

-- Option B: Prove the minimum value of x^2 + 1/(x^2 + 3) for x in ℝ is not -1.
theorem option_B (x : ℝ) : ¬(∃ (min_value : ℝ), min_value = -1 ∧ ∀(y : ℝ), (y ^ 2) + 1 / (y ^ 2 + 3) ≥ min_value) :=
sorry

-- Option C: Prove the maximum value of xy given x + 2y + xy = 6 and x, y > 0 is 2.
theorem option_C (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y + x * y = 6) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 2 * v + u * v = 6 → u * v ≤ max_value :=
sorry

-- Option D: Prove the minimum value of 2x + y given x + 4y + 4 = xy and x, y > 0 is 17.
theorem option_D (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 4 * y + 4 = x * y) : 
  ∃ (min_value : ℝ), min_value = 17 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 4 * v + 4 = u * v → 2 * u + v ≥ min_value :=
sorry

-- Combine to verify which options are correct
theorem verify_options (A_correct B_correct C_correct D_correct : Prop) :
  A_correct = true ∧ B_correct = false ∧ C_correct = true ∧ D_correct = true :=
sorry

end option_A_option_B_option_C_option_D_verify_options_l541_541785


namespace solve_equation_l541_541002

theorem solve_equation (x : ℝ) :
  (x + 1)^2 = (2 * x - 1)^2 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_equation_l541_541002


namespace factor_polynomial_l541_541896

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l541_541896


namespace largest_n_dividing_30_factorial_l541_541486

theorem largest_n_dividing_30_factorial (n : ℕ) :
  (18^n) ∣ (nat.factorial 30) → n ≤ 7 :=
by 
  sorry

end largest_n_dividing_30_factorial_l541_541486


namespace largest_even_digit_multiple_of_5_under_8000_is_6880_l541_541333

noncomputable def is_even_digit (n : Nat) : Prop :=
  ∀ d ∈ Int.to_list n, d % 2 = 0

noncomputable def is_multiple_of_5 (n : Nat) : Prop :=
  n % 5 = 0

noncomputable def largest_even_digit_multiple_of_5_less_than (limit : Nat) : Nat :=
  6880

theorem largest_even_digit_multiple_of_5_under_8000_is_6880 : 
    ∀ n : Nat, n < 8000 → is_even_digit n → is_multiple_of_5 n → n ≤ 6880 :=
  by
  intro n h1 h2 h3
  sorry

end largest_even_digit_multiple_of_5_under_8000_is_6880_l541_541333


namespace equilateral_triangle_side_length_l541_541391

noncomputable def side_lengths := (6, 10, 11)

def perimeter (a b c : ℕ) : ℕ := a + b + c

def equilateral_side_length (p : ℕ) : ℕ := p / 3

theorem equilateral_triangle_side_length :
  let (a, b, c) := side_lengths
  let p := perimeter a b c
  equilateral_side_length p = 9 :=
by 
  let (a, b, c) := side_lengths
  let p := perimeter a b c
  have h1 : a = 6 := by rfl
  have h2 : b = 10 := by rfl
  have h3 : c = 11 := by rfl
  have h4 : p = 27 := by simp [perimeter, *]
  have h5 : equilateral_side_length p = 9 := by simp [equilateral_side_length, h4]; norm_num 
  exact h5

end equilateral_triangle_side_length_l541_541391


namespace raisins_in_first_box_l541_541044

def total_raisins := 437
def raisins_second_box := 74
def raisins_other_boxes := 97

theorem raisins_in_first_box :
  let total_other_boxes := 3 * raisins_other_boxes,
      total_known_boxes := total_other_boxes + raisins_second_box in
  437 = total_known_boxes + 72 :=
by
  let total_other_boxes := 3 * raisins_other_boxes,
      total_known_boxes := total_other_boxes + raisins_second_box
  have h : total_raisins = total_known_boxes + 72 := by linarith
  exact h

end raisins_in_first_box_l541_541044


namespace largest_n_for_18_pow_n_div_30_factorial_l541_541515

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define conditions in Lean
def highest_power (p n : ℕ) : ℕ :=
(nat.div n p + nat.div n (p ^ 2) + nat.div n (p ^ 3) + nat.div n (p ^ 4) + nat.div n (p ^ 5))

lemma power_of_2_in_30! : highest_power 2 30 = 26 := by sorry
lemma power_of_3_in_30! : highest_power 3 30 = 14 := by sorry

-- Lean statement translating (question, conditions, correct answer) tuple
theorem largest_n_for_18_pow_n_div_30_factorial :
  ∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, 18^m ∣ factorial 30 → m ≤ 7 :=
by
  use 7
  split
  - sorry
  - intros m hm
    sorry

end largest_n_for_18_pow_n_div_30_factorial_l541_541515


namespace two_digit_sum_divisibility_l541_541335

noncomputable def is_prime (n : ℕ) : Prop := n.prime

theorem two_digit_sum_divisibility :
  ∑ n in ({n : ℕ | 10 * (n / 10) + n % 10 = n ∧
                let a := n / 10 in let b := n % 10 in
                (a + b > 1) ∧ is_prime (a + b) ∧
                (a + b) ∣ n ∧ (a * b) ∣ n ∧ (abs (a - b)) ∣ n}.to_finset),
  n = 180 :=
by
  sorry

end two_digit_sum_divisibility_l541_541335


namespace probability_of_consonant_initials_l541_541617

def number_of_students : Nat := 30
def alphabet_size : Nat := 26
def redefined_vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
def number_of_vowels : Nat := redefined_vowels.card
def number_of_consonants : Nat := alphabet_size - number_of_vowels

theorem probability_of_consonant_initials :
  (number_of_consonants : ℝ) / (number_of_students : ℝ) = 2/3 := 
by
  -- Proof goes here
  sorry

end probability_of_consonant_initials_l541_541617


namespace sum_of_roots_of_quadratic_l541_541279

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l541_541279


namespace cows_and_chickens_l541_541154

theorem cows_and_chickens (H : ℕ) : 
  let C := 6 in
  let L := 4 * C + 2 * H in
  let Heads := C + H in
  L > 2 * Heads →
  L - 2 * Heads = 12 :=
by
  intros
  let C := 6
  let L := 4 * C + 2 * H
  let Heads := C + H
  have h1 : L - 2 * Heads = 12 := sorry
  exact h1

end cows_and_chickens_l541_541154


namespace max_a_l541_541141

variable {a x : ℝ}

theorem max_a (h : x^2 - 2 * x - 3 > 0 → x < a ∧ ¬ (x < a → x^2 - 2 * x - 3 > 0)) : a = 3 :=
sorry

end max_a_l541_541141


namespace regression_prediction_l541_541115

theorem regression_prediction
  (slope : ℝ) (centroid_x centroid_y : ℝ) (b : ℝ)
  (h_slope : slope = 1.23)
  (h_centroid : centroid_x = 4 ∧ centroid_y = 5)
  (h_intercept : centroid_y = slope * centroid_x + b)
  (x : ℝ) (h_x : x = 10) :
  centroid_y = 5 →
  slope = 1.23 →
  x = 10 →
  b = 5 - 1.23 * 4 →
  (slope * x + b) = 12.38 :=
by
  intros
  sorry

end regression_prediction_l541_541115


namespace rate_is_correct_l541_541043

noncomputable def rate_of_interest (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem rate_is_correct :
  rate_of_interest 10000 18500 8 = 10.625 := 
by
  sorry

end rate_is_correct_l541_541043


namespace range_of_function_l541_541268

theorem range_of_function : 
  ∃ (S : Set ℝ), (∀ x : ℝ, y = |sin x| - 2 * sin x → y ∈ S) ∧ S = Set.Icc (-1) 3 := 
sorry

end range_of_function_l541_541268


namespace minimum_k_conditions_l541_541782

theorem minimum_k_conditions (k : ℝ) :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k)) ↔ k = 3/2 :=
sorry

end minimum_k_conditions_l541_541782


namespace max_area_PQR_max_area_incenters_l541_541630

def triangle (A B C : Type) := true   -- placeholder for the actual triangle object

def area (T : triangle) : ℝ := sorry   -- placeholder for the area function

variable (A B C : Type)

-- \(\triangle ABC\) is an equilateral triangle with unit area
axiom equilateral_triangle_unit_area (T : triangle A B C) : true

-- external equilateral triangles \(\triangle APB\), \(\triangle BQC\), and \(\triangle CRA\)
axiom equilateral_triangles_external (T1 T2 T3 : triangle) : true

-- \(\angle APB = \angle BQC = \angle CRA = 60^\circ\)
axiom angles_60_deg (angle1 angle2 angle3 : ℝ) : angle1 = 60 ∧ angle2 = 60 ∧ angle3 = 60

-- Problem 1: Proving the maximum area of triangle \(\triangle PQR\) is 4
theorem max_area_PQR 
  (T1 T2 T3 : triangle) 
  (h1 : equilateral_triangle_unit_area (triangle A B C)) 
  (h2 : equilateral_triangles_external T1 T2 T3) 
  (h3 : angles_60_deg 60 60 60) : 
  area (triangle P Q R) = 4 := sorry

-- Problem 2: Proving the maximum area of the triangle whose vertices are the incenters of \(\triangle APB\), \(\triangle BQC\), and \(\triangle CRA\) is 1
theorem max_area_incenters
  (T1 T2 T3 : triangle)
  (h1 : equilateral_triangle_unit_area (triangle A B C))
  (h2 : equilateral_triangles_external T1 T2 T3)
  (h3 : angles_60_deg 60 60 60) :
  area (triangle (incenter T1) (incenter T2) (incenter T3)) = 1 := sorry

end max_area_PQR_max_area_incenters_l541_541630


namespace find_b_l541_541374

theorem find_b (b c : ℝ) : 
  (-11 : ℝ) = (-1)^2 + (-1) * b + c ∧ 
  17 = 3^2 + 3 * b + c ∧ 
  6 = 2^2 + 2 * b + c → 
  b = 14 / 3 :=
by
  sorry

end find_b_l541_541374


namespace length_of_train_A_l541_541775

noncomputable def train_length (speed_A_kmh speed_B_kmh speed_A_ms speed_B_ms train_A_kmh train_B_kmh cross_time_s := 
  let relative_speed_kmh := speed_A_kmh + speed_B_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)
  let distance_covered_m := relative_speed_ms * cross_time_s
  distance_covered_m - train_B_kmh) (speed_A_kmh := 54) (speed_B_kmh := 36) (cross_time_s := 14) (train_B_kmh := 150) : 
  ℝ := 200

theorem length_of_train_A : train_length 54 36 (54 * (5/18)) (36 * (5/18)) 54 36 14 150 = 200 := 
  by sorry

end length_of_train_A_l541_541775


namespace calculate_expression_l541_541053

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l541_541053


namespace largest_n_dividing_30_factorial_l541_541498

theorem largest_n_dividing_30_factorial (n : ℕ) :
  18 ^ 7 ∣ nat.factorial 30 ∧ (∀ m : ℕ, 18 ^ m ∣ nat.factorial 30 → m ≤ 7) :=
by
  sorry

end largest_n_dividing_30_factorial_l541_541498


namespace area_of_pathways_l541_541655

theorem area_of_pathways 
  (num_rows : ℕ) (num_cols : ℕ) (bed_width : ℕ) (bed_height : ℕ) (pathway_width : ℕ)
  (total_width : ℕ) (total_height : ℕ) (total_area : ℕ) (bed_area : ℕ) (pathway_area : ℕ)
  (h1 : num_rows = 4) (h2 : num_cols = 3) (h3 : bed_width = 4) (h4 : bed_height = 3) (h5 : pathway_width = 2)
  (h6 : total_width = num_cols * bed_width + (num_cols + 1) * pathway_width)
  (h7 : total_height = num_rows * bed_height + (num_rows + 1) * pathway_width)
  (h8 : total_area = total_width * total_height)
  (h9 : bed_area = num_rows * num_cols * (bed_width * bed_height))
  (h10 : pathway_area = total_area - bed_area)
  : pathway_area = 196 :=
by {
  subst_vars, -- Substitute h1 through h10 into the statement
  sorry       -- Skip the proof
}

end area_of_pathways_l541_541655


namespace new_students_joined_l541_541252

theorem new_students_joined 
  (A_o A_n A_new O : ℕ) 
  (h1 : A_o = 40) 
  (h2 : A_n = 32) 
  (h3 : A_new = 36) 
  (h4 : O = 2) :
  let N := (A_o * O - A_new * O) / (A_new - A_n) in N = 2 :=
by
  sorry

end new_students_joined_l541_541252


namespace length_of_faster_train_is_75m_l541_541356

-- Definition of initial conditions
def slower_train_speed : ℝ := 32  -- in kmph
def faster_train_speed : ℝ := 50  -- in kmph
def time_to_pass : ℝ := 15        -- in seconds

-- Definition of derived conditions
def relative_speed_kmph : ℝ := faster_train_speed - slower_train_speed  -- in kmph
def relative_speed_kmps : ℝ := relative_speed_kmph / 3600  -- converting kmph to kmps

-- Definition of the expected answer
def length_of_faster_train_km : ℝ := relative_speed_kmps * time_to_pass / 1 -- in km

-- Conversion from km to meters
def length_of_faster_train_m : ℝ := length_of_faster_train_km * 1000

theorem length_of_faster_train_is_75m : length_of_faster_train_m = 75 := 
by sorry  -- Proof omitted

end length_of_faster_train_is_75m_l541_541356


namespace monotonicity_and_extremes_range_of_m_l541_541575

def has_extreme_values_at (f: ℝ → ℝ) (a b: ℝ) : Prop :=
  ∀ x, f.deriv x = 0 → (x = a ∨ x = b)

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem monotonicity_and_extremes :
  (∀ x < -1, f'(x) > 0) ∧ (∀ x, x > 3 → f'(x) > 0) ∧
  (∀ x, -1 < x ∧ x < 3 → f'(x) < 0) ∧
  f(-1) = 5 ∧ f(3) = -27 :=
begin
  -- sorry is added to skip proof
  sorry
end

theorem range_of_m :
  ∃! m : ℝ, (m > 5 ∨ m < -27) →
    ∀ y, (y = m) → ∃! x, (f(x) = y) :=
begin
  -- sorry is added to skip proof
  sorry
end

end monotonicity_and_extremes_range_of_m_l541_541575


namespace relationship_to_circle_and_line_equation_l541_541555

-- Definition of the circle C
def circleC (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 25

-- Definition of the point P
def pointP : ℝ × ℝ := (2, 1)

-- Theorem to prove relationship between P and C, and equation of the line
theorem relationship_to_circle_and_line_equation :
  let P := (2, 1)
  in (∃ x y : ℝ, circleC x y ∧ (P.1 + 1)^2 + (P.2 - 2)^2 < 25) ∧
     (∃ k : ℝ, 
        ((k = 0 ∧ ∀ x y : ℝ, x = 2 → circleC x y) ∨ 
         (k ≠ 0 ∧ ∀ x y : ℝ, 4 * x + 3 * y = 11 → circleC x y))) :=
by
  sorry

end relationship_to_circle_and_line_equation_l541_541555


namespace factor_polynomial_l541_541895

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l541_541895


namespace exists_perpendicular_line_l541_541003

variable (a b : Line) (M : Point)

-- define what it means for a and b to be skew lines
def skew (a b : Line) : Prop :=
  ¬(∃ P : Point, P ∈ a ∧ P ∈ b) ∧ ¬(∃ α : Plane, a ⊂ α ∧ b ⊂ α)

-- define what it means for a line to be perpendicular to both a and b
def perpendicular_to_both (c a b : Line) : Prop :=
  ∀ (M : Point), M ∈ c → M ⊥ a ∧ M ⊥ b

theorem exists_perpendicular_line (a b : Line) (h_skew : skew a b) :
  ∃ c : Line, perpendicular_to_both c a b :=
sorry

end exists_perpendicular_line_l541_541003


namespace BC_length_proof_l541_541642

noncomputable def length_of_side_BC (A B C D : Type) [Quadrilateral A B C D] 
  (AD CD : ℝ) (cos_ADC sin_BCA : ℝ) (Circumcircle : Prop) : ℝ :=
  if AD = 4 ∧ CD = 7 ∧ cos_ADC = (1 / 2) ∧ sin_BCA = (1 / 3) ∧ Circumcircle then
    (real.sqrt 37) / (3 * real.sqrt 3) * (real.sqrt 24 - 1)
  else
    0

theorem BC_length_proof (A B C D : Type) [Quadrilateral A B C D] 
  (AD CD : ℝ) (cos_ADC sin_BCA : ℝ) (Circumcircle : Prop) 
  (hAD : AD = 4) (hCD : CD = 7) (hcos : cos_ADC = (1 / 2)) (hsin : sin_BCA = (1 / 3)) 
  (hCircumcircle : Circumcircle) : 
  length_of_side_BC A B C D AD CD cos_ADC sin_BCA Circumcircle = 
  (real.sqrt 37) / (3 * real.sqrt 3) * (real.sqrt 24 - 1) :=
by sorry

end BC_length_proof_l541_541642


namespace infinitely_many_arithmetic_progression_triples_l541_541077

theorem infinitely_many_arithmetic_progression_triples :
  ∃ (u v: ℤ) (a b c: ℤ), 
  (∀ n: ℤ, (a = 2 * u) ∧ 
    (b = 2 * u + v) ∧
    (c = 2 * u + 2 * v) ∧ 
    (u > 0) ∧
    (v > 0) ∧
    ∃ k m n: ℤ, 
    (a * b + 1 = k * k) ∧ 
    (b * c + 1 = m * m) ∧ 
    (c * a + 1 = n * n)) :=
sorry

end infinitely_many_arithmetic_progression_triples_l541_541077


namespace fraction_of_number_l541_541302

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l541_541302


namespace minimum_value_of_f_l541_541991

def f (a x : ℝ) : ℝ := x + a / (x - 2)

theorem minimum_value_of_f (a x : ℝ) (h1 : f a 3 = 7) (h2 : x > 2) : ∃ m : ℝ, m = 6 :=
by
  sorry

end minimum_value_of_f_l541_541991


namespace total_food_for_guinea_pigs_l541_541719

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l541_541719


namespace special_prime_sum_l541_541084

open Nat

def is_special_prime (p : ℕ) : Prop :=
  p > 1 ∧ p < 200 ∧ Nat.Prime p ∧
  p % 6 = 1 ∧
  p % 7 = 6

theorem special_prime_sum : (∑ p in Finset.filter is_special_prime (Finset.range 201)) = 460 :=
sorry

end special_prime_sum_l541_541084


namespace triangle_area_l541_541928

noncomputable def A_eq_pi_over_3 (A : ℝ) : Prop :=
  (∃ a b c pi_over_3 : ℝ, sin (A - pi_over_3 / 6) = cos A ∧ A = pi_over_3)

noncomputable def area_of_triangle (a b c S A : ℝ) : Prop :=
  a = 1 ∧ b + c = 2 ∧ A = Real.pi / 3 → S = (1 / 2) * b * c * Real.sin A

theorem triangle_area (a b c S A : ℝ)
  (h1 : a = 1) (h2 : b + c = 2) (h3 : A = Real.pi / 3)
  (h4 : sin (A - Real.pi / 6) = cos A) :
  A_eq_pi_over_3 A ∧ area_of_triangle a b c S A := 
by
  sorry

end triangle_area_l541_541928


namespace sum_of_roots_l541_541282

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l541_541282


namespace count_ways_to_get_plus_sign_at_top_l541_541631

-- Define the condition for the sign of the top cell of the pyramid
def sign_of_top (a b c d e : Int) : Int :=
  (a * b * c * d) * (b * c * d * e)

-- Define the requirement for the top cell to be "+"
def top_is_plus (a b c d e : Int) : Prop :=
  sign_of_top a b c d e = 1

-- Define the range of the signs as +1 or -1
def is_sign (x : Int) : Prop :=
  x = 1 ∨ x = -1

-- Define a condition that validates whether all inputs are valid signs
def valid_signs (a b c d e : Int) : Prop :=
  is_sign a ∧ is_sign b ∧ is_sign c ∧ is_sign d ∧ is_sign e

-- Define the main theorem
theorem count_ways_to_get_plus_sign_at_top :
  (∃ a b c d e : Int, valid_signs a b c d e ∧ top_is_plus a b c d e) → 8 := sorry

end count_ways_to_get_plus_sign_at_top_l541_541631


namespace new_person_weight_l541_541741

variable (W : ℝ)

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ)
  (h1 : avg_increase = 3.5) (h2 : num_persons = 10) (h3 : replaced_weight = 65) :
  W = 100 :=
by
  have total_weight_increase : ℝ := num_persons * avg_increase
  have weight_difference : ℝ := total_weight_increase
  have new_weight : ℝ := replaced_weight + weight_difference
  have result : W = new_weight := sorry
  rw [←result]
  sorry

end new_person_weight_l541_541741


namespace process_flowchart_incorrect_statement_l541_541784

theorem process_flowchart_incorrect_statement :
  ¬(∃ loop : Prop, loop = "A loop can appear in the process flowchart") :=
sorry

end process_flowchart_incorrect_statement_l541_541784


namespace hyperbola_asymptotes_correct_l541_541998

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_e : e = sqrt 6 / 2) : Prop :=
  let c := sqrt (a^2 + b^2) in
  e = c / a → 1 + (b^2 / a^2) = e^2 → (b / a = sqrt 2 / 2) → ∀ x : ℝ, ∃ y : ℝ,
  y = (sqrt 2 / 2) * x ∨ y = (-(sqrt 2 / 2)) * x

theorem hyperbola_asymptotes_correct : ∀ (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_e : sqrt 6 / 2 = sqrt 6 / 2),
  hyperbola_asymptotes a b (sqrt 6 / 2) h_a_pos h_b_pos h_e :=
by
  intros
  sorry

end hyperbola_asymptotes_correct_l541_541998


namespace jackson_pay_rate_l541_541191

def hours_vacuuming := 2 * 2
def hours_washing_dishes := 0.5
def hours_cleaning_bathroom := 3 * hours_washing_dishes

def total_hours := hours_vacuuming + hours_washing_dishes + hours_cleaning_bathroom
def total_earnings := 30

theorem jackson_pay_rate : total_earnings / total_hours = 5 := by
  sorry

end jackson_pay_rate_l541_541191


namespace set_B_expression_l541_541361

theorem set_B_expression :
  let A := {-1, 0, 1}
  let B := {x | ∃ t ∈ A, x = t^2}
  B = {0, 1} :=
by
  sorry

end set_B_expression_l541_541361
