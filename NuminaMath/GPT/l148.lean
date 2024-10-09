import Mathlib

namespace beaver_group_l148_14827

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l148_14827


namespace rope_length_after_knots_l148_14812

def num_ropes : ℕ := 64
def length_per_rope : ℕ := 25
def length_reduction_per_knot : ℕ := 3
def num_knots : ℕ := num_ropes - 1
def initial_total_length : ℕ := num_ropes * length_per_rope
def total_reduction : ℕ := num_knots * length_reduction_per_knot
def final_rope_length : ℕ := initial_total_length - total_reduction

theorem rope_length_after_knots :
  final_rope_length = 1411 := by
  sorry

end rope_length_after_knots_l148_14812


namespace solve_N_l148_14822

noncomputable def N (a b c d : ℝ) := (a + b) / c - d

theorem solve_N : 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  N a b c d = -1 :=
by 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  let n := N a b c d
  sorry

end solve_N_l148_14822


namespace are_names_possible_l148_14856

-- Define the structure to hold names
structure Person where
  first_name  : String
  middle_name : String
  last_name   : String

-- List of 4 people
def people : List Person :=
  [{ first_name := "Ivan", middle_name := "Ivanovich", last_name := "Ivanov" },
   { first_name := "Ivan", middle_name := "Petrovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Ivanovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Petrovich", last_name := "Ivanov" }]

-- Define the problem theorem
theorem are_names_possible :
  ∃ (people : List Person), 
    (∀ (p1 p2 p3 : Person), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → (p1.first_name ≠ p2.first_name ∨ p1.first_name ≠ p3.first_name ∨ p2.first_name ≠ p3.first_name) ∧
    (p1.middle_name ≠ p2.middle_name ∨ p1.middle_name ≠ p3.middle_name ∨ p2.middle_name ≠ p3.middle_name) ∧
    (p1.last_name ≠ p2.last_name ∨ p1.last_name ≠ p3.last_name ∨ p2.last_name ≠ p3.last_name)) ∧
    (∀ (p1 p2 : Person), p1 ≠ p2 → (p1.first_name = p2.first_name ∨ p1.middle_name = p2.middle_name ∨ p1.last_name = p2.last_name)) :=
by
  -- Place proof here
  sorry

end are_names_possible_l148_14856


namespace extremum_and_monotonicity_inequality_for_c_l148_14844

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem extremum_and_monotonicity (α : ℝ) (h_extremum : ∀ (x : ℝ), x = Real.exp 2 → f x α = 0) :
  (∃ α : ℝ, (∀ x : ℝ, x > Real.exp 2 → f x α > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < Real.exp 2 → f x α < 0)) := sorry

theorem inequality_for_c (c : ℝ) (α : ℝ) (h_extremum : α = 3)
  (h_ineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 3 → f x α < 2 * c^2 - c) :
  (1 < c) ∨ (c < -1 / 2) := sorry

end extremum_and_monotonicity_inequality_for_c_l148_14844


namespace gnollish_valid_sentences_count_l148_14883

/--
The Gnollish language consists of 4 words: "splargh," "glumph," "amr," and "bork."
A sentence is valid if "splargh" does not come directly before "glumph" or "bork."
Prove that there are 240 valid 4-word sentences in Gnollish.
-/
theorem gnollish_valid_sentences_count : 
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  valid_sentences = 240 :=
by
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  have : valid_sentences = 240 := by sorry
  exact this

end gnollish_valid_sentences_count_l148_14883


namespace WorldCup_group_stage_matches_l148_14816

theorem WorldCup_group_stage_matches
  (teams : ℕ)
  (groups : ℕ)
  (teams_per_group : ℕ)
  (matches_per_group : ℕ)
  (total_matches : ℕ) :
  teams = 32 ∧ 
  groups = 8 ∧ 
  teams_per_group = 4 ∧ 
  matches_per_group = teams_per_group * (teams_per_group - 1) / 2 ∧ 
  total_matches = matches_per_group * groups →
  total_matches = 48 :=
by 
  -- sorry lets Lean skip the proof.
  sorry

end WorldCup_group_stage_matches_l148_14816


namespace x_intercept_of_line_l148_14834

-- Definition of line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Proposition that the x-intercept of the line 4x + 7y = 28 is (7, 0)
theorem x_intercept_of_line : line_eq 7 0 :=
by
  show 4 * 7 + 7 * 0 = 28
  sorry

end x_intercept_of_line_l148_14834


namespace find_phi_symmetric_l148_14836

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.sqrt 3 * (Real.cos (2 * x)))

theorem find_phi_symmetric : ∃ φ : ℝ, (φ = Real.pi / 12) ∧ ∀ x : ℝ, f (-x + φ) = f (x + φ) := 
sorry

end find_phi_symmetric_l148_14836


namespace max_difference_in_volume_l148_14808

noncomputable def computed_volume (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def max_possible_volume (length width height : ℕ) (error : ℕ) : ℕ :=
  (length + error) * (width + error) * (height + error)

theorem max_difference_in_volume :
  ∀ (length width height error : ℕ), length = 150 → width = 150 → height = 225 → error = 1 → 
  max_possible_volume length width height error - computed_volume length width height = 90726 :=
by
  intros length width height error h_length h_width h_height h_error
  rw [h_length, h_width, h_height, h_error]
  simp only [computed_volume, max_possible_volume]
  -- Intermediate calculations
  sorry

end max_difference_in_volume_l148_14808


namespace calc_exponent_result_l148_14805

theorem calc_exponent_result (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := 
by
  sorry

end calc_exponent_result_l148_14805


namespace squares_area_ratios_l148_14879

noncomputable def squareC_area (x : ℝ) : ℝ := x ^ 2
noncomputable def squareD_area (x : ℝ) : ℝ := 3 * x ^ 2
noncomputable def squareE_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem squares_area_ratios (x : ℝ) (h : x ≠ 0) :
  (squareC_area x / squareE_area x = 1 / 36) ∧ (squareD_area x / squareE_area x = 1 / 4) := by
  sorry

end squares_area_ratios_l148_14879


namespace vector_relationship_l148_14892

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
          (A A1 B D E : V) (x y z : ℝ)

-- Given Conditions
def inside_top_face_A1B1C1D1 (E : V) : Prop :=
  ∃ (y z : ℝ), (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧
  E = A1 + y • (B - A) + z • (D - A)

-- Prove the desired relationship
theorem vector_relationship (h : E = x • (A1 - A) + y • (B - A) + z • (D - A))
  (hE : inside_top_face_A1B1C1D1 A A1 B D E) : 
  x = 1 ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) :=
sorry

end vector_relationship_l148_14892


namespace infinite_area_sum_ratio_l148_14863

theorem infinite_area_sum_ratio (T t : ℝ) (p q : ℝ) (h_ratio : T / t = 3 / 2) :
    let series_ratio_triangles := (p + q)^2 / (3 * p * q)
    let series_ratio_quadrilaterals := (p + q)^2 / (2 * p * q)
    (T * series_ratio_triangles) / (t * series_ratio_quadrilaterals) = 1 :=
by
  -- Proof steps go here
  sorry

end infinite_area_sum_ratio_l148_14863


namespace ratio_of_doctors_to_engineers_l148_14884

variables (d l e : ℕ) -- number of doctors, lawyers, and engineers

-- Conditions
def avg_age := (40 * d + 55 * l + 50 * e) / (d + l + e) = 45
def doctors_avg := 40 
def lawyers_avg := 55 
def engineers_avg := 50 -- 55 - 5

theorem ratio_of_doctors_to_engineers (h_avg : avg_age d l e) : d = 3 * e :=
sorry

end ratio_of_doctors_to_engineers_l148_14884


namespace line_equation_passing_through_and_perpendicular_l148_14810

theorem line_equation_passing_through_and_perpendicular :
  ∃ A B C : ℝ, (∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → -2 * x + y + 1 = 0 ∧ 
(x = 2 ∧ y = -1) → 2 * x + y - 3 = 0) :=
by
  sorry

end line_equation_passing_through_and_perpendicular_l148_14810


namespace total_animal_eyes_l148_14878

-- Define the conditions given in the problem
def numberFrogs : Nat := 20
def numberCrocodiles : Nat := 10
def eyesEach : Nat := 2

-- Define the statement that we need to prove
theorem total_animal_eyes : (numberFrogs * eyesEach) + (numberCrocodiles * eyesEach) = 60 := by
  sorry

end total_animal_eyes_l148_14878


namespace cookie_boxes_condition_l148_14847

theorem cookie_boxes_condition (n : ℕ) (M A : ℕ) :
  M = n - 8 ∧ A = n - 2 ∧ M + A < n ∧ M ≥ 1 ∧ A ≥ 1 → n = 9 :=
by
  intro h
  sorry

end cookie_boxes_condition_l148_14847


namespace A_number_is_35_l148_14809

theorem A_number_is_35 (A B : ℕ) 
  (h_sum_digits : A + B = 8) 
  (h_diff_numbers : 10 * B + A = 10 * A + B + 18) :
  10 * A + B = 35 :=
by {
  sorry
}

end A_number_is_35_l148_14809


namespace Euclid_Middle_School_AMC8_contest_l148_14814

theorem Euclid_Middle_School_AMC8_contest (students_Germain students_Newton students_Young : ℕ)
       (hG : students_Germain = 11) 
       (hN : students_Newton = 8) 
       (hY : students_Young = 9) : 
       students_Germain + students_Newton + students_Young = 28 :=
by
  sorry

end Euclid_Middle_School_AMC8_contest_l148_14814


namespace value_of_3m_2n_l148_14885

section ProofProblem

variable (m n : ℤ)
-- Condition that x-3 is a factor of 3x^3 - mx + n
def factor1 : Prop := (3 * 3^3 - m * 3 + n = 0)
-- Condition that x+4 is a factor of 3x^3 - mx + n
def factor2 : Prop := (3 * (-4)^3 - m * (-4) + n = 0)

theorem value_of_3m_2n (h₁ : factor1 m n) (h₂ : factor2 m n) : abs (3 * m - 2 * n) = 45 := by
  sorry

end ProofProblem

end value_of_3m_2n_l148_14885


namespace point_transform_l148_14861

theorem point_transform : 
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  P' = (-3, 0) :=
by
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  show P' = (-3, 0)
  sorry

end point_transform_l148_14861


namespace tangent_line_at_A_increasing_intervals_decreasing_interval_l148_14855

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 + 1

-- Define the derivatives at x
noncomputable def f' (x : ℝ) := 6 * x^2 + 6 * x

-- Define the tangent line equation at a point
noncomputable def tangent_line (x : ℝ) := 12 * x - 6

theorem tangent_line_at_A :
  tangent_line 1 = 6 :=
  by
    -- proof omitted
    sorry

theorem increasing_intervals :
  (∀ x ∈ Set.Ioi 0, f' x > 0) ∧
  (∀ x ∈ Set.Iio (-1), f' x > 0) :=
  by
    -- proof omitted
    sorry

theorem decreasing_interval :
  ∀ x ∈ Set.Ioo (-1) 0, f' x < 0 :=
  by
    -- proof omitted
    sorry

end tangent_line_at_A_increasing_intervals_decreasing_interval_l148_14855


namespace x_pow_twelve_l148_14898

theorem x_pow_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 = 322 :=
sorry

end x_pow_twelve_l148_14898


namespace range_of_m_l148_14833

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (¬ (proposition_p m) ∧ proposition_q m) ∨ (proposition_p m ∧ ¬ (proposition_q m)) →
  (1/3 <= m ∧ m < 15) :=
sorry

end range_of_m_l148_14833


namespace raft_minimum_capacity_l148_14889

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l148_14889


namespace find_a_plus_b_l148_14882

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1))
  (h2 : ∀ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1) → x + y = 0) : 
  a + b = 2 :=
sorry

end find_a_plus_b_l148_14882


namespace rightmost_three_digits_of_5_pow_1994_l148_14876

theorem rightmost_three_digits_of_5_pow_1994 : (5 ^ 1994) % 1000 = 625 :=
by
  sorry

end rightmost_three_digits_of_5_pow_1994_l148_14876


namespace octahedron_coloring_l148_14890

theorem octahedron_coloring : 
  ∃ (n : ℕ), n = 6 ∧
  ∀ (F : Fin 8 → Fin 4), 
    (∀ (i j : Fin 8), i ≠ j → F i ≠ F j) ∧
    (∃ (pairs : Fin 8 → (Fin 4 × Fin 4)), 
      (∀ (i : Fin 8), ∃ j : Fin 4, pairs i = (j, j)) ∧ 
      (∀ j, ∃ (i : Fin 8), F i = j)) :=
by
  sorry

end octahedron_coloring_l148_14890


namespace labourer_total_payment_l148_14891

/--
A labourer was engaged for 25 days on the condition that for every day he works, he will be paid Rs. 2 and for every day he is absent, he will be fined 50 p. He was absent for 5 days. Prove that the total amount he received in the end is Rs. 37.50.
-/
theorem labourer_total_payment :
  let total_days := 25
  let daily_wage := 2.0
  let absent_days := 5
  let fine_per_absent_day := 0.5
  let worked_days := total_days - absent_days
  let total_earnings := worked_days * daily_wage
  let total_fine := absent_days * fine_per_absent_day
  let total_received := total_earnings - total_fine
  total_received = 37.5 :=
by
  sorry

end labourer_total_payment_l148_14891


namespace M_greater_than_N_l148_14875

-- Definitions based on the problem's conditions
def M (x : ℝ) : ℝ := (x - 3) * (x - 7)
def N (x : ℝ) : ℝ := (x - 2) * (x - 8)

-- Statement to prove
theorem M_greater_than_N (x : ℝ) : M x > N x := by
  -- Proof is omitted
  sorry

end M_greater_than_N_l148_14875


namespace distinct_positive_integer_quadruples_l148_14840

theorem distinct_positive_integer_quadruples 
  (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a + b = c * d) (h8 : a * b = c + d) :
  (a, b, c, d) = (1, 5, 2, 3)
  ∨ (a, b, c, d) = (1, 5, 3, 2)
  ∨ (a, b, c, d) = (5, 1, 2, 3)
  ∨ (a, b, c, d) = (5, 1, 3, 2)
  ∨ (a, b, c, d) = (2, 3, 1, 5)
  ∨ (a, b, c, d) = (2, 3, 5, 1)
  ∨ (a, b, c, d) = (3, 2, 1, 5)
  ∨ (a, b, c, d) = (3, 2, 5, 1) :=
  sorry

end distinct_positive_integer_quadruples_l148_14840


namespace perpendicular_bisector_AC_circumcircle_eqn_l148_14850

/-- Given vertices of triangle ABC, prove the equation of the perpendicular bisector of side AC --/
theorem perpendicular_bisector_AC (A B C D : ℝ×ℝ) (hA: A = (0, 2)) (hC: C = (4, 0)) (hD: D = (2, 1)) :
  ∃ k b, (k = 2) ∧ (b = -3) ∧ (∀ x y, y = k * x + b ↔ 2 * x - y - 3 = 0) :=
sorry

/-- Given vertices of triangle ABC, prove the equation of the circumcircle --/
theorem circumcircle_eqn (A B C D E F : ℝ×ℝ) (hA: A = (0, 2)) (hB: B = (6, 4)) (hC: C = (4, 0)) :
  ∃ k, k = 10 ∧ 
  (∀ x y, (x - 3) ^ 2 + (y - 3) ^ 2 = k ↔ x ^ 2 + y ^ 2 - 6 * x - 2 * y + 8 = 0) :=
sorry

end perpendicular_bisector_AC_circumcircle_eqn_l148_14850


namespace evaluate_g_5_times_l148_14848

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x + 2 else 3 * x + 1

theorem evaluate_g_5_times : g (g (g (g (g 1)))) = 12 := by
  sorry


end evaluate_g_5_times_l148_14848


namespace remainder_mod_500_l148_14800

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l148_14800


namespace count_total_balls_l148_14877

def blue_balls : ℕ := 3
def red_balls : ℕ := 2

theorem count_total_balls : blue_balls + red_balls = 5 :=
by {
  sorry
}

end count_total_balls_l148_14877


namespace trajectory_moving_point_l148_14806

theorem trajectory_moving_point (x y : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 ↔ x^2 + y^2 = 1 := by
  sorry

end trajectory_moving_point_l148_14806


namespace right_triangle_conditions_l148_14832

theorem right_triangle_conditions (A B C : ℝ) (a b c : ℝ):
  (C = 90) ∨ (A + B = C) ∨ (a/b = 3/4 ∧ a/c = 3/5 ∧ b/c = 4/5) →
  (a^2 + b^2 = c^2) ∨ (A + B + C = 180) → 
  (C = 90 ∧ a^2 + b^2 = c^2) :=
sorry

end right_triangle_conditions_l148_14832


namespace sum_n_max_value_l148_14835

noncomputable def arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem sum_n_max_value :
  (∃ n : Nat, n = 9 ∧ sum_arithmetic_sequence 25 (-3) n = 117) :=
by
  let a1 := 25
  let d := -3
  use 9
  -- To complete the proof, we would calculate the sum of the first 9 terms
  -- of the arithmetic sequence with a1 = 25 and difference d = -3.
  sorry

end sum_n_max_value_l148_14835


namespace big_boxes_count_l148_14845

theorem big_boxes_count
  (soaps_per_package : ℕ)
  (packages_per_box : ℕ)
  (total_soaps : ℕ)
  (soaps_per_box : ℕ)
  (H1 : soaps_per_package = 192)
  (H2 : packages_per_box = 6)
  (H3 : total_soaps = 2304)
  (H4 : soaps_per_box = soaps_per_package * packages_per_box) :
  total_soaps / soaps_per_box = 2 :=
by
  sorry

end big_boxes_count_l148_14845


namespace number_of_valid_strings_l148_14801

def count_valid_strings (n : ℕ) : ℕ :=
  4^n - 3 * 3^n + 3 * 2^n - 1

theorem number_of_valid_strings (n : ℕ) :
  count_valid_strings n = 4^n - 3 * 3^n + 3 * 2^n - 1 :=
by sorry

end number_of_valid_strings_l148_14801


namespace dodecagon_area_l148_14853

theorem dodecagon_area (a : ℝ) : 
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  dodecagon_area = (3 * a^2) / 2 :=
by
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  sorry

end dodecagon_area_l148_14853


namespace min_value_inequality_l148_14858

theorem min_value_inequality (a b c : ℝ) (h : a + b + c = 3) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a + b) + 1 / c) ≥ 4 / 3 :=
sorry

end min_value_inequality_l148_14858


namespace bus_trip_distance_l148_14880

theorem bus_trip_distance 
  (T : ℝ)  -- Time in hours
  (D : ℝ)  -- Distance in miles
  (h : D = 30 * T)  -- condition 1: the trip with 30 mph
  (h' : D = 35 * (T - 1))  -- condition 2: the trip with 35 mph
  : D = 210 := 
by
  sorry

end bus_trip_distance_l148_14880


namespace abs_inequality_solution_l148_14866

theorem abs_inequality_solution (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ (-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10) := 
by {
  sorry
}

end abs_inequality_solution_l148_14866


namespace question1_question2_l148_14839

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - abs (2 * x - 1)

theorem question1 (x : ℝ) :
  ∀ a, a = 2 → (f x 2 + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2) := by
sorry

theorem question2 (a : ℝ) :
  (∀ x, 1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end question1_question2_l148_14839


namespace solve_system_l148_14807

theorem solve_system :
  ∃ (x y : ℚ), (4 * x - 35 * y = -1) ∧ (3 * y - x = 5) ∧ (x = -172 / 23) ∧ (y = -19 / 23) :=
by
  sorry

end solve_system_l148_14807


namespace largest_natural_number_has_sum_of_digits_property_l148_14874

noncomputable def largest_nat_num_digital_sum : ℕ :=
  let a : ℕ := 1
  let b : ℕ := 0
  let d3 := a + b
  let d4 := 2 * a + 2 * b
  let d5 := 4 * a + 4 * b
  let d6 := 8 * a + 8 * b
  100000 * a + 10000 * b + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem largest_natural_number_has_sum_of_digits_property :
  largest_nat_num_digital_sum = 101248 :=
by
  sorry

end largest_natural_number_has_sum_of_digits_property_l148_14874


namespace percentage_taxed_on_excess_income_l148_14860

noncomputable def pct_taxed_on_first_40k : ℝ := 0.11
noncomputable def first_40k_income : ℝ := 40000
noncomputable def total_income : ℝ := 58000
noncomputable def total_tax : ℝ := 8000

theorem percentage_taxed_on_excess_income :
  ∃ P : ℝ, (total_tax - pct_taxed_on_first_40k * first_40k_income = P * (total_income - first_40k_income)) ∧ P * 100 = 20 := 
by
  sorry

end percentage_taxed_on_excess_income_l148_14860


namespace investment_calculation_l148_14825

noncomputable def calculate_investment_amount (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_calculation :
  let A := 80000
  let r := 0.07
  let n := 12
  let t := 7
  let P := calculate_investment_amount A r n t
  abs (P - 46962) < 1 :=
by
  sorry

end investment_calculation_l148_14825


namespace find_m_l148_14870

def h (x m : ℝ) := x^2 - 3 * x + m
def k (x m : ℝ) := x^2 - 3 * x + 5 * m

theorem find_m (m : ℝ) (h_def : ∀ x, h x m = x^2 - 3 * x + m) (k_def : ∀ x, k x m = x^2 - 3 * x + 5 * m) (key_eq : 3 * h 5 m = 2 * k 5 m) :
  m = 10 / 7 :=
by
  sorry

end find_m_l148_14870


namespace mean_of_set_with_median_l148_14899

theorem mean_of_set_with_median (m : ℝ) (h : m + 7 = 10) :
  (m + (m + 2) + (m + 7) + (m + 10) + (m + 12)) / 5 = 9.2 :=
by
  -- Placeholder for the proof.
  sorry

end mean_of_set_with_median_l148_14899


namespace map_length_conversion_l148_14895

-- Define the given condition: 12 cm on the map represents 72 km in reality.
def length_on_map := 12 -- in cm
def distance_in_reality := 72 -- in km

-- Define the length in cm we want to find the real-world distance for.
def query_length := 17 -- in cm

-- State the proof problem.
theorem map_length_conversion :
  (distance_in_reality / length_on_map) * query_length = 102 :=
by
  -- placeholder for the proof
  sorry

end map_length_conversion_l148_14895


namespace interest_calculation_l148_14803

theorem interest_calculation (P : ℝ) (r : ℝ) (CI SI : ℝ → ℝ) (n : ℝ) :
  P = 1300 →
  r = 0.10 →
  (CI n - SI n = 13) →
  (CI n = P * (1 + r)^n - P) →
  (SI n = P * r * n) →
  (1.10 ^ n - 1 - 0.10 * n = 0.01) →
  n = 2 :=
by
  intro P_eq r_eq diff_eq CI_def SI_def equation
  -- Sorry, this is just a placeholder. The proof is omitted.
  sorry

end interest_calculation_l148_14803


namespace solve_for_a_l148_14887

-- Defining the equation and given solution
theorem solve_for_a (x a : ℝ) (h : 2 * x - 5 * a = 3 * a + 22) (hx : x = 3) : a = -2 := by
  sorry

end solve_for_a_l148_14887


namespace math_problem_l148_14818

theorem math_problem (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := 
by
  sorry

end math_problem_l148_14818


namespace smallest_class_number_l148_14871

-- Define the conditions
def num_classes : Nat := 24
def num_selected_classes : Nat := 4
def total_sum : Nat := 52
def sampling_interval : Nat := num_classes / num_selected_classes

-- The core theorem to be proved
theorem smallest_class_number :
  ∃ x : Nat, x + (x + sampling_interval) + (x + 2 * sampling_interval) + (x + 3 * sampling_interval) = total_sum ∧ x = 4 := by
  sorry

end smallest_class_number_l148_14871


namespace find_segment_AD_length_l148_14851

noncomputable def segment_length_AD (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] :=
  ∃ (angle_BAD angle_ABC angle_BCD : Real)
    (length_AB length_CD : Real)
    (perpendicular : X) (angle_BAX angle_ABX : Real)
    (length_AX length_DX length_AD : Real),
    angle_BAD = 60 ∧
    angle_ABC = 30 ∧
    angle_BCD = 30 ∧
    length_AB = 15 ∧
    length_CD = 8 ∧
    angle_BAX = 30 ∧
    angle_ABX = 60 ∧
    length_AX = length_AB / 2 ∧
    length_DX = length_CD / 2 ∧
    length_AD = length_AX - length_DX ∧
    length_AD = 3.5

theorem find_segment_AD_length (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] : segment_length_AD A B C D X :=
by
  sorry

end find_segment_AD_length_l148_14851


namespace pump_A_time_to_empty_pool_l148_14841

theorem pump_A_time_to_empty_pool :
  ∃ (A : ℝ), (1/A + 1/9 = 1/3.6) ∧ A = 6 :=
sorry

end pump_A_time_to_empty_pool_l148_14841


namespace minimize_expression_l148_14864

theorem minimize_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
(h4 : x^2 + y^2 + z^2 = 1) : 
  z = Real.sqrt 2 - 1 :=
sorry

end minimize_expression_l148_14864


namespace tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l148_14868

-- First proof problem
theorem tan_theta_eq2_simplifies_to_minus1 (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (θ - 6 * Real.pi) + Real.sin (Real.pi / 2 - θ)) / 
  (2 * Real.sin (Real.pi + θ) + Real.cos (-θ)) = -1 := sorry

-- Second proof problem
theorem sin_cos_and_tan_relation (x : ℝ) (hx1 : - Real.pi / 2 < x) (hx2 : x < Real.pi / 2) 
  (h : Real.sin x + Real.cos x = 1 / 5) : Real.tan x = -3 / 4 := sorry

end tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l148_14868


namespace smallest_positive_integer_b_no_inverse_l148_14857

theorem smallest_positive_integer_b_no_inverse :
  ∃ b : ℕ, b > 0 ∧ gcd b 30 > 1 ∧ gcd b 42 > 1 ∧ b = 6 :=
by
  sorry

end smallest_positive_integer_b_no_inverse_l148_14857


namespace final_bill_correct_l148_14872

def initial_bill := 500.00
def late_charge_rate := 0.02
def final_bill := initial_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem final_bill_correct : final_bill = 520.20 := by
  sorry

end final_bill_correct_l148_14872


namespace total_amount_spent_l148_14869

/-
  Define the original prices of the games, discount rate, and tax rate.
-/
def batman_game_price : ℝ := 13.60
def superman_game_price : ℝ := 5.06
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

/-
  Prove that the total amount spent including discounts and taxes equals $16.12.
-/
theorem total_amount_spent :
  let batman_discount := batman_game_price * discount_rate
  let superman_discount := superman_game_price * discount_rate
  let batman_discounted_price := batman_game_price - batman_discount
  let superman_discounted_price := superman_game_price - superman_discount
  let total_before_tax := batman_discounted_price + superman_discounted_price
  let sales_tax := total_before_tax * tax_rate
  let total_amount := total_before_tax + sales_tax
  total_amount = 16.12 :=
by
  sorry

end total_amount_spent_l148_14869


namespace find_x_squared_plus_inv_squared_l148_14865

theorem find_x_squared_plus_inv_squared (x : ℝ) (hx : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := 
by
sorry

end find_x_squared_plus_inv_squared_l148_14865


namespace fried_hop_edges_in_three_hops_l148_14846

noncomputable def fried_hop_probability : ℚ :=
  let moves : List (Int × Int) := [(-1, 0), (1, 0), (0, -1), (0, 1)]
  let center := (2, 2)
  let edges := [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
  -- Since the exact steps of solution calculation are complex,
  -- we assume the correct probability as per our given solution.
  5 / 8

theorem fried_hop_edges_in_three_hops :
  let p := fried_hop_probability
  p = 5 / 8 := by
  sorry

end fried_hop_edges_in_three_hops_l148_14846


namespace number_of_birds_is_20_l148_14802

-- Define the given conditions
def distance_jim_disney : ℕ := 50
def distance_disney_london : ℕ := 60
def total_travel_distance : ℕ := 2200

-- Define the number of birds
def num_birds (B : ℕ) : Prop :=
  (distance_jim_disney + distance_disney_london) * B = total_travel_distance

-- The theorem stating the number of birds
theorem number_of_birds_is_20 : num_birds 20 :=
by
  unfold num_birds
  sorry

end number_of_birds_is_20_l148_14802


namespace yuko_in_front_of_yuri_l148_14826

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end yuko_in_front_of_yuri_l148_14826


namespace gcd_78_143_l148_14859

theorem gcd_78_143 : Nat.gcd 78 143 = 13 :=
by
  sorry

end gcd_78_143_l148_14859


namespace maximum_ab_l148_14815

theorem maximum_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3*a + 8*b = 48) : ab ≤ 24 :=
by
  sorry

end maximum_ab_l148_14815


namespace time_to_read_18_pages_l148_14831

-- Definitions based on the conditions
def reading_rate : ℚ := 2 / 4 -- Amalia reads 4 pages in 2 minutes
def pages_to_read : ℕ := 18 -- Number of pages Amalia needs to read

-- Goal: Total time required to read 18 pages
theorem time_to_read_18_pages (r : ℚ := reading_rate) (p : ℕ := pages_to_read) :
  p * r = 9 := by
  sorry

end time_to_read_18_pages_l148_14831


namespace students_on_bus_l148_14828

theorem students_on_bus (initial_students : ℝ) (students_got_on : ℝ) (total_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : students_got_on = 3.0) : 
  total_students = 13.0 :=
by 
  sorry

end students_on_bus_l148_14828


namespace ratio_of_red_to_blue_marbles_l148_14804

theorem ratio_of_red_to_blue_marbles (total_marbles yellow_marbles : ℕ) (green_marbles blue_marbles red_marbles : ℕ) 
  (odds_blue : ℚ) 
  (h1 : total_marbles = 60) 
  (h2 : yellow_marbles = 20) 
  (h3 : green_marbles = yellow_marbles / 2) 
  (h4 : red_marbles + blue_marbles = total_marbles - (yellow_marbles + green_marbles)) 
  (h5 : odds_blue = 0.25) 
  (h6 : blue_marbles = odds_blue * (red_marbles + blue_marbles)) : 
  red_marbles / blue_marbles = 11 / 4 := 
by 
  sorry

end ratio_of_red_to_blue_marbles_l148_14804


namespace restaurant_sodas_l148_14894

theorem restaurant_sodas (M : ℕ) (h1 : M + 19 = 96) : M = 77 :=
by
  sorry

end restaurant_sodas_l148_14894


namespace fraction_filled_l148_14843

variables (E P p : ℝ)

-- Condition 1: The empty vessel weighs 12% of its total weight when filled.
axiom cond1 : E = 0.12 * (E + P)

-- Condition 2: The weight of the partially filled vessel is one half that of a completely filled vessel.
axiom cond2 : E + p = 1 / 2 * (E + P)

theorem fraction_filled : p / P = 19 / 44 :=
by
  sorry

end fraction_filled_l148_14843


namespace employee_hourly_pay_l148_14830

-- Definitions based on conditions
def initial_employees := 500
def daily_hours := 10
def weekly_days := 5
def monthly_weeks := 4
def additional_employees := 200
def total_payment := 1680000
def total_employees := initial_employees + additional_employees
def monthly_hours_per_employee := daily_hours * weekly_days * monthly_weeks
def total_monthly_hours := total_employees * monthly_hours_per_employee

-- Lean 4 statement proving the hourly pay per employee
theorem employee_hourly_pay : total_payment / total_monthly_hours = 12 := by sorry

end employee_hourly_pay_l148_14830


namespace no_valid_transformation_l148_14837

theorem no_valid_transformation :
  ¬ ∃ (n1 n2 n3 n4 : ℤ),
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 :=
by
  sorry

end no_valid_transformation_l148_14837


namespace domain_of_f_l148_14886

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  Real.log ((m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1)

theorem domain_of_f (m : ℝ) :
  (∀ x : ℝ, 0 < (m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1) ↔ (m > 7/3 ∨ m ≤ 1) :=
by { sorry }

end domain_of_f_l148_14886


namespace greatest_sum_on_circle_l148_14862

theorem greatest_sum_on_circle : 
  ∃ x y : ℤ, x^2 + y^2 = 169 ∧ x ≥ y ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 169 → x' ≥ y' → x + y ≥ x' + y') := 
sorry

end greatest_sum_on_circle_l148_14862


namespace point_M_on_y_axis_l148_14842

theorem point_M_on_y_axis (t : ℝ) (h : t - 3 = 0) : (t-3, 5-t) = (0, 2) :=
by
  sorry

end point_M_on_y_axis_l148_14842


namespace sale_price_is_207_l148_14896

-- Define a namespace for our problem
namespace BicyclePrice

-- Define the conditions as constants
def priceAtStoreP : ℝ := 200
def regularPriceIncreasePercentage : ℝ := 0.15
def salePriceDecreasePercentage : ℝ := 0.10

-- Define the regular price at Store Q
def regularPriceAtStoreQ : ℝ := priceAtStoreP * (1 + regularPriceIncreasePercentage)

-- Define the sale price at Store Q
def salePriceAtStoreQ : ℝ := regularPriceAtStoreQ * (1 - salePriceDecreasePercentage)

-- The final theorem we need to prove
theorem sale_price_is_207 : salePriceAtStoreQ = 207 := by
  sorry

end BicyclePrice

end sale_price_is_207_l148_14896


namespace initial_group_size_l148_14824

theorem initial_group_size (n : ℕ) (W : ℝ) 
  (h1 : (W + 20) / n = W / n + 4) : 
  n = 5 := 
by 
  sorry

end initial_group_size_l148_14824


namespace count_4_digit_numbers_with_conditions_l148_14820

def num_valid_numbers : Nat :=
  432

-- Statement declaring the proposition to be proved
theorem count_4_digit_numbers_with_conditions :
  (count_valid_numbers == 432) :=
sorry

end count_4_digit_numbers_with_conditions_l148_14820


namespace pizza_topping_combinations_l148_14893

theorem pizza_topping_combinations (T : Finset ℕ) (hT : T.card = 8) : 
  (T.card.choose 1 + T.card.choose 2 + T.card.choose 3 = 92) :=
by
  sorry

end pizza_topping_combinations_l148_14893


namespace lattice_points_count_l148_14873

theorem lattice_points_count : ∃ n : ℕ, n = 8 ∧ (∃ x y : ℤ, x^2 - y^2 = 51) :=
by
  sorry

end lattice_points_count_l148_14873


namespace second_degree_polynomial_inequality_l148_14849

def P (u v w x : ℝ) : ℝ := u * x^2 + v * x + w

theorem second_degree_polynomial_inequality 
  (u v w : ℝ) (h : ∀ a : ℝ, 1 ≤ a → P u v w (a^2 + a) ≥ a * P u v w (a + 1)) :
  u > 0 ∧ w ≤ 4 * u :=
by
  sorry

end second_degree_polynomial_inequality_l148_14849


namespace tina_money_left_l148_14811

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l148_14811


namespace magnitude_difference_l148_14823

noncomputable def vector_a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def vector_b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

theorem magnitude_difference (a b : ℝ × ℝ) 
  (ha : a = (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180)))
  (hb : b = (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))) :
  (Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)) = Real.sqrt 3 :=
by
  sorry

end magnitude_difference_l148_14823


namespace angle_215_third_quadrant_l148_14881

-- Define the context of the problem
def angle_vertex_origin : Prop := true 

def initial_side_non_negative_x_axis : Prop := true

noncomputable def in_third_quadrant (angle: ℝ) : Prop := 
  180 < angle ∧ angle < 270 

-- The theorem to prove the condition given
theorem angle_215_third_quadrant : 
  angle_vertex_origin → 
  initial_side_non_negative_x_axis → 
  in_third_quadrant 215 :=
by
  intro _ _
  unfold in_third_quadrant
  sorry -- This is where the proof would go

end angle_215_third_quadrant_l148_14881


namespace smallest_among_l148_14854

theorem smallest_among {a b c d : ℤ} (h1 : a = -4) (h2 : b = -3) (h3 : c = 0) (h4 : d = 1) :
  a < b ∧ a < c ∧ a < d :=
by
  rw [h1, h2, h3, h4]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_among_l148_14854


namespace Tn_lt_Sn_div_2_l148_14838

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n

noncomputable def S (n : ℕ) : ℝ := 
  (3 / 2) * (1 - (1 / 3)^n)

noncomputable def T (n : ℕ) : ℝ := 
  (3 / 4) * (1 - (1 / 3)^n) - (n / 2) * (1 / 3)^(n + 1)

theorem Tn_lt_Sn_div_2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l148_14838


namespace john_paid_more_l148_14813

theorem john_paid_more 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (tip_percentage : ℝ) 
  (discounted_price : ℝ)
  (john_tip : ℝ) 
  (john_total : ℝ)
  (jane_tip : ℝ)
  (jane_total : ℝ) 
  (difference : ℝ) :
  original_price = 42.00000000000004 →
  discount_percentage = 0.10 →
  tip_percentage = 0.15 →
  discounted_price = original_price - (discount_percentage * original_price) →
  john_tip = tip_percentage * original_price →
  john_total = original_price + john_tip →
  jane_tip = tip_percentage * discounted_price →
  jane_total = discounted_price + jane_tip →
  difference = john_total - jane_total →
  difference = 4.830000000000005 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end john_paid_more_l148_14813


namespace find_denomination_of_bills_l148_14867

variables 
  (bills_13 : ℕ)  -- Denomination of the bills Tim has 13 of
  (bills_5 : ℕ := 5)  -- Denomination of the bills Tim has 11 of, which are $5 bills
  (bills_1 : ℕ := 1)  -- Denomination of the bills Tim has 17 of, which are $1 bills
  (total_amt : ℕ := 128)  -- Total amount Tim needs to pay
  (num_bills_13 : ℕ := 13)  -- Number of bills of unknown denomination
  (num_bills_5 : ℕ := 11)  -- Number of $5 bills
  (num_bills_1 : ℕ := 17)  -- Number of $1 bills
  (min_bills : ℕ := 16)  -- Minimum number of bills to be used

theorem find_denomination_of_bills : 
  num_bills_13 * bills_13 + num_bills_5 * bills_5 + num_bills_1 * bills_1 = total_amt →
  num_bills_13 + num_bills_5 + num_bills_1 ≥ min_bills → 
  bills_13 = 4 :=
by
  intros h1 h2
  sorry

end find_denomination_of_bills_l148_14867


namespace correct_equation_for_growth_rate_l148_14829

def initial_price : ℝ := 6.2
def final_price : ℝ := 8.9
def growth_rate (x : ℝ) : ℝ := initial_price * (1 + x) ^ 2

theorem correct_equation_for_growth_rate (x : ℝ) : growth_rate x = final_price ↔ initial_price * (1 + x) ^ 2 = 8.9 :=
by sorry

end correct_equation_for_growth_rate_l148_14829


namespace range_of_f_l148_14852

noncomputable def f (x y : ℝ) := (x^3 + y^3) / (x + y)^3

theorem range_of_f :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 1 → (1 / 4) ≤ f x y ∧ f x y < 1) :=
by
  sorry

end range_of_f_l148_14852


namespace smallest_even_integer_l148_14819

theorem smallest_even_integer (n : ℕ) (h_even : n % 2 = 0)
  (h_2digit : 10 ≤ n ∧ n ≤ 98)
  (h_property : (n - 2) * n * (n + 2) = 5 * ((n - 2) + n + (n + 2))) :
  n = 86 :=
by
  sorry

end smallest_even_integer_l148_14819


namespace required_bike_speed_l148_14888

theorem required_bike_speed (swim_distance run_distance bike_distance swim_speed run_speed total_time : ℝ)
  (h_swim_dist : swim_distance = 0.5)
  (h_run_dist : run_distance = 4)
  (h_bike_dist : bike_distance = 12)
  (h_swim_speed : swim_speed = 1)
  (h_run_speed : run_speed = 8)
  (h_total_time : total_time = 1.5) :
  (bike_distance / ((total_time - (swim_distance / swim_speed + run_distance / run_speed)))) = 24 :=
by
  sorry

end required_bike_speed_l148_14888


namespace cocktail_cost_per_litre_is_accurate_l148_14821

noncomputable def mixed_fruit_juice_cost_per_litre : ℝ := 262.85
noncomputable def acai_berry_juice_cost_per_litre : ℝ := 3104.35
noncomputable def mixed_fruit_juice_litres : ℝ := 35
noncomputable def acai_berry_juice_litres : ℝ := 23.333333333333336

noncomputable def cocktail_total_cost : ℝ := 
  (mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_litres) +
  (acai_berry_juice_cost_per_litre * acai_berry_juice_litres)

noncomputable def cocktail_total_volume : ℝ := 
  mixed_fruit_juice_litres + acai_berry_juice_litres

noncomputable def cocktail_cost_per_litre : ℝ := 
  cocktail_total_cost / cocktail_total_volume

theorem cocktail_cost_per_litre_is_accurate : 
  abs (cocktail_cost_per_litre - 1399.99) < 0.01 := by
  sorry

end cocktail_cost_per_litre_is_accurate_l148_14821


namespace benny_initial_comics_l148_14897

variable (x : ℕ)

def initial_comics (x : ℕ) : ℕ := x

def comics_after_selling (x : ℕ) : ℕ := (2 * x) / 5

def comics_after_buying (x : ℕ) : ℕ := (comics_after_selling x) + 12

def traded_comics (x : ℕ) : ℕ := (comics_after_buying x) / 4

def comics_after_trading (x : ℕ) : ℕ := (3 * (comics_after_buying x)) / 4 + 18

theorem benny_initial_comics : comics_after_trading x = 72 → x = 150 := by
  intro h
  sorry

end benny_initial_comics_l148_14897


namespace contrapositive_inequality_l148_14817

theorem contrapositive_inequality (x : ℝ) :
  ((x + 2) * (x - 3) > 0) → (x < -2 ∨ x > 0) :=
by
  sorry

end contrapositive_inequality_l148_14817
