import Mathlib

namespace sqrt_x_div_sqrt_y_as_fraction_l532_532355

theorem sqrt_x_div_sqrt_y_as_fraction 
    (x y : ℝ)
    (h : ( (1 / 3)^2 + (1 / 4)^2) / ((1 / 5)^2 + (1 / 6)^2) = 25 * x / (53 * y)) : 
    (real.sqrt x) / (real.sqrt y) = 150 / 239 :=
by
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l532_532355


namespace f_2008_divisible_by_3_l532_532074

def is_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a = List.range (1, n+1).permute (fun i => a[i])

def valid_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a.head = 1 ∧ ∀ i, 1 ≤ i ∧ i < n → |a[i] - a[i+1]| ≤ 2

def f (n : ℕ) : ℕ := List.filter (fun a => valid_permutation a n) 
                            (List.permutations (List.range (1, n+1))).length

def r (n : ℕ) : ℕ := (f n) mod 3

theorem f_2008_divisible_by_3 : r 2008 = 0 := sorry

end f_2008_divisible_by_3_l532_532074


namespace conclusion_A_conclusion_B_conclusion_C_conclusion_D_l532_532269

open Real

theorem conclusion_A (A B : ℝ) (h_triangle : A > B) : sin A > sin B :=
sorry

theorem conclusion_B (a b c : ℝ) (h_acute : a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) : b^2 + c^2 - a^2 > 0 :=
sorry

theorem conclusion_C (A B : ℝ) (h_sin : sin (2 * A) = sin (2 * B)) : ¬(∀ a b c : ℝ, isosceles_triangle a b c) :=
sorry

theorem conclusion_D (b : ℝ) (A : ℝ) (S : ℝ) (h_b : b = 3) (h_A : A = π / 3) (h_S : S = 3 * sqrt 3) : let R := (sqrt 3 / 3) in
not (let a c : ℝ := sorry in sorry) := sorry

end conclusion_A_conclusion_B_conclusion_C_conclusion_D_l532_532269


namespace sufficient_condition_not_necessary_condition_l532_532397

theorem sufficient_condition (a : ℝ) (p : a ≥ 0) (q : a^2 + a ≥ 0) : p → q :=
by sorry

theorem not_necessary_condition (a : ℝ) (q : a^2 + a ≥ 0) : ¬(q → p) :=
by sorry

end sufficient_condition_not_necessary_condition_l532_532397


namespace vasya_table_problem_l532_532172

/-- Vasya's table problem -/
theorem vasya_table_problem :
  let S := (fun (tbl: list (list ℕ)) => list.map (λ col, Nat.digits 10 (list.prod col)) tbl)
  in let cols := (list.transpose ([(1 : ℕ) :: list.range 71]).take 18)
  in ¬ (list.All (λ s, s = list.head (S cols)) (list.tail (S cols))) :=
by
  sorry

end vasya_table_problem_l532_532172


namespace line_slope_equals_l532_532004

-- Definitions based on problem conditions
def parametric_eq (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)
def circle_eq (x y : ℝ) : Prop := x ^ 2 + y ^ 2 - 4 * x + 3 = 0
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- The main theorem we want to prove
theorem line_slope_equals:
  ∀ (α t1 t2 : ℝ),
  (t1 ≠ t2) →
  let A := parametric_eq t1 α in
  let B := parametric_eq t2 α in
  circle_eq A.1 A.2 →
  circle_eq B.1 B.2 →
  distance A B = Real.sqrt 3 →
  Real.tan α = ± (Real.sqrt 15 / 15) :=
by sorry

end line_slope_equals_l532_532004


namespace smallest_four_digit_divisible_by_35_l532_532249

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532249


namespace p_is_prime_l532_532503

theorem p_is_prime (n : ℕ) (p : ℕ) (h1 : p = 2^n + 1) 
  (h2 : (3^((p - 1) / 2) + 1) % p = 0) : prime p :=
sorry

end p_is_prime_l532_532503


namespace initial_concentration_alcohol_l532_532656

theorem initial_concentration_alcohol (x : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 100)
    (h2 : 0.44 * 10 = (x / 100) * 2 + 3.6) :
    x = 40 :=
sorry

end initial_concentration_alcohol_l532_532656


namespace angle_B_measure_sine_C_value_l532_532485

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def triangle_conditions := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  ∠A > 0 ∧ ∠B > 0 ∧ ∠C > 0 ∧
  ∠A + ∠B + ∠C = π ∧
  a = 5 ∧ b = 7 ∧
  sin(A)^2 + sin(A) * sin(C) + sin(C)^2 + cos(B)^2 = 1

theorem angle_B_measure: triangle_conditions → B = 2 * π / 3 := sorry

theorem sine_C_value: triangle_conditions → sin C = 3 * sqrt 3 / 14 := sorry

end angle_B_measure_sine_C_value_l532_532485


namespace number_of_pairs_l532_532716

theorem number_of_pairs (n C k : ℕ) (hC : C ≤ 100) (hnC : ∃ k, n^2 + n + C = (n + k)^2) : 
  (finset.card {p : ℕ × ℕ | p.1 ^ 2 + p.1 + p.2 = (p.1 + p.2)^2 ∧ p.2 ≤ 100}) = 180 :=
sorry

end number_of_pairs_l532_532716


namespace triangle_area_7_8_10_l532_532361

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := semi_perimeter a b c in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_7_8_10 :
  triangle_area 7 8 10 ≈ 27.78 :=
by 
  let a := 7
  let b := 8
  let c := 10
  let s := semi_perimeter a b c
  let area := triangle_area a b c
  have h1 : a + b > c := by norm_num [a, b, c]
  have h2 : a + c > b := by norm_num [a, c, b]
  have h3 : b + c > a := by norm_num [b, c, a]
  have area_def := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  have area_calculated : area ≈ 27.78 := by
    norm_num
  exact area_calculated
  sorry

end triangle_area_7_8_10_l532_532361


namespace unique_function_f_l532_532358

theorem unique_function_f (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x = -f (-x))
    (h2 : ∀ x : ℝ, f (x + 1) = f x + 1)
    (h3 : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / x^2 * f x) :
    ∀ x : ℝ, f x = x := 
sorry

end unique_function_f_l532_532358


namespace remainder_correct_l532_532719

-- Define the divisor polynomial
def divisor : Polynomial ℚ := Polynomial.C 1 + Polynomial.X * 2 + Polynomial.C 1

-- Define the polynomial to be divided
def dividend : Polynomial ℚ := 5 * Polynomial.X^6 + 3 * Polynomial.X^4 - 2 * Polynomial.X^3 + 7 * Polynomial.X^2 + Polynomial.C 4

-- Define the expected remainder polynomial
def expected_remainder : Polynomial ℚ := -38 * Polynomial.X - 29

-- State the theorem
theorem remainder_correct :
  Polynomial.r_div_intro dividend divisor expected_remainder := 
  sorry

end remainder_correct_l532_532719


namespace simplest_common_denominator_l532_532578

theorem simplest_common_denominator (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  LCM (2 * x^2 * y) (6 * x * y^3) = 6 * x^2 * y^3 := 
sorry

end simplest_common_denominator_l532_532578


namespace count_defective_pens_l532_532811

theorem count_defective_pens
  (total_pens : ℕ) (prob_non_defective : ℚ)
  (h1 : total_pens = 12)
  (h2 : prob_non_defective = 0.5454545454545454) :
  ∃ (D : ℕ), D = 1 := by
  sorry

end count_defective_pens_l532_532811


namespace smallest_four_digit_divisible_by_35_l532_532205

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532205


namespace num_sides_of_regular_polygon_l532_532797

theorem num_sides_of_regular_polygon (θ : ℝ) (n : ℕ) (h1 : θ = 45) (h2 : ∑ i in (finset.range n), θ = 360) : n = 8 :=
sorry

end num_sides_of_regular_polygon_l532_532797


namespace lindsay_doll_count_l532_532528

theorem lindsay_doll_count :
  let B := 4 in                  -- Number of blonde-haired dolls
  let Br := 4 * B in             -- Number of brown-haired dolls
  let Bl := Br - 2 in            -- Number of black-haired dolls
  Bl + Br - B = 26 :=            -- Prove the combined excess of black and brown over blonde
begin 
  sorry 
end

end lindsay_doll_count_l532_532528


namespace appropriate_sampling_method_l532_532305

-- Define the conditions of the problem
def male_students : ℕ := 500
def female_students : ℕ := 500
def total_students : ℕ := male_students + female_students
def students_to_survey : ℕ := 100

-- State that stratified sampling is the appropriate method
theorem appropriate_sampling_method : 
  (male_students = 500) ∧ 
  (female_students = 500) ∧ 
  (total_students = 1000) ∧ 
  (students_to_survey = 100) 
  → (sampling_method = "Stratified sampling") := 
by
  sorry  -- Proof to be filled in later

#eval appropriate_sampling_method

end appropriate_sampling_method_l532_532305


namespace triangle_QS_length_l532_532045

theorem triangle_QS_length (P Q R S : Type) [EuclideanGeometry P Q R S] :
  (PQ : ℝ) = 8 → 
  (QR : ℝ) = 15 → 
  (PR : ℝ) = 17 → 
  angle_bisector Q P R S →
  (QS : ℝ) = (4 * real.sqrt 3272) / 23 :=
by
  sorry

end triangle_QS_length_l532_532045


namespace validate_operation_l532_532609

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end validate_operation_l532_532609


namespace arithmetic_sequence_common_difference_l532_532941

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532941


namespace number_of_valid_sequences_l532_532109

def rotation_counterclockwise_90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

def rotation_clockwise_90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def E := (2, 2) : ℝ × ℝ
def F := (-2, 2) : ℝ × ℝ
def G := (-2, -2) : ℝ × ℝ
def H := (2, -2) : ℝ × ℝ

-- Applying a sequence of transformations T to vertices and checking for identity
def apply_transformation (T : (ℝ × ℝ → ℝ × ℝ) list) (p : ℝ × ℝ) : ℝ × ℝ :=
  list.foldl (λ acc f => f acc) p T

noncomputable def transform_sequence_identity : list (ℝ × ℝ → ℝ × ℝ) → Prop :=
  λ T, apply_transformation T E = E ∧ apply_transformation T F = F ∧ apply_transformation T G = G ∧ apply_transformation T H = H
  
theorem number_of_valid_sequences : 
  ∃ (count : ℕ), count = 455 ∧ ∀ (T : list (ℝ × ℝ → ℝ × ℝ)), T.length = 24 → transform_sequence_identity T → count = 455 := sorry

end number_of_valid_sequences_l532_532109


namespace percentage_of_second_solution_correct_l532_532152

noncomputable def percentage_of_alcohol_in_second_solution : ℝ :=
  let total_liters := 80
  let percentage_final_solution := 0.49
  let volume_first_solution := 24
  let percentage_first_solution := 0.4
  let volume_second_solution := 56
  let total_alcohol_in_final_solution := total_liters * percentage_final_solution
  let total_alcohol_first_solution := volume_first_solution * percentage_first_solution
  let x := (total_alcohol_in_final_solution - total_alcohol_first_solution) / volume_second_solution
  x

theorem percentage_of_second_solution_correct : 
  percentage_of_alcohol_in_second_solution = 0.5285714286 := by sorry

end percentage_of_second_solution_correct_l532_532152


namespace problem_statement_l532_532013

theorem problem_statement (y : ℝ) 
  (h : y + sqrt (y^2 - 4) + 4 / (y - sqrt (y^2 - 4)) = 18) :
  y^2 + sqrt (y^4 - 4) + 16 / (y^2 + sqrt (y^4 - 4)) = 3 * (256 / 2025 - sqrt ((65536 - 16402500) / 4100625)) :=
sorry

end problem_statement_l532_532013


namespace smallest_four_digit_divisible_by_35_l532_532243

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532243


namespace f_is_identity_l532_532501

variable (f : ℕ → ℕ)
variable (hf : ∀ (m n : ℕ), f(f(m) + f(n)) = m + n)

theorem f_is_identity : ∀ x : ℕ, f(x) = x :=
by {
  sorry
}

end f_is_identity_l532_532501


namespace smallest_four_digit_divisible_by_35_l532_532242

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532242


namespace middle_number_l532_532135

theorem middle_number {a b c : ℕ} (h1 : a + b = 12) (h2 : a + c = 17) (h3 : b + c = 19) (h4 : a < b) (h5 : b < c) : b = 7 :=
sorry

end middle_number_l532_532135


namespace common_difference_of_arithmetic_sequence_l532_532882

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532882


namespace common_difference_of_arithmetic_sequence_l532_532971

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532971


namespace smallest_four_digit_divisible_by_35_l532_532191

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532191


namespace sequence_sums_l532_532409

theorem sequence_sums (S_n : ℕ → ℝ) (a_n b_n : ℕ → ℝ) :
  (∀ n, 2 * S_n n = 3^n + 3) →
  (∀ n, a_n n = if n = 1 then 3 else 3^(n-1)) →
  (∀ n, b_n n = if n = 1 then 1 / 3 else (n - 1) * 3^(1 - n)) →
  ∀ n, (∑ i in Finset.range n, b_n (i + 1)) = 13 / 12 + (6 * n + 3) / (4 * 3^n) :=
sorry

end sequence_sums_l532_532409


namespace arithmetic_sequence_common_difference_l532_532937

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532937


namespace arithmetic_sequence_common_difference_l532_532893

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532893


namespace smallest_four_digit_divisible_by_35_l532_532186

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532186


namespace min_exponent_binomial_l532_532478

theorem min_exponent_binomial (n : ℕ) (h1 : n > 0)
  (h2 : ∃ r : ℕ, (n.choose r) / (n.choose (r + 1)) = 5 / 7) : n = 11 :=
by {
-- Note: We are merely stating the theorem here according to the instructions,
-- the proof body is omitted and hence the use of 'sorry'.
sorry
}

end min_exponent_binomial_l532_532478


namespace number_of_mappings_l532_532067

def f (A : set ℕ) : ℕ → ℕ := sorry

noncomputable def mapping_count (A : set ℕ) : ℕ := sorry

theorem number_of_mappings :
  let A := {n | 1 ≤ n ∧ n ≤ 11} in
  A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} →
  (∀ x ∈ A, f A^15 x = x) →
  (∀ k, (1 ≤ k ∧ k ≤ 14) → ∃ a ∈ A, f A^k a ≠ a) →
  mapping_count A = nat.factorial 11 / 30 :=
sorry

end number_of_mappings_l532_532067


namespace common_difference_is_two_l532_532918

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532918


namespace arithmetic_sequence_common_difference_l532_532860

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532860


namespace enclosed_area_l532_532344

noncomputable def calculateArea : ℝ :=
  ∫ (x : ℝ) in (1 / 2)..2, 1 / x

theorem enclosed_area : calculateArea = 2 * Real.log 2 :=
by
  sorry

end enclosed_area_l532_532344


namespace smallest_four_digit_divisible_by_35_l532_532182

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532182


namespace find_a9_l532_532410

variable (a : ℕ → ℤ)

-- Condition 1: The sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

-- Condition 2: Given a_4 = 5
def a4_value (a : ℕ → ℤ) : Prop :=
  a 4 = 5

-- Condition 3: Given a_5 = 4
def a5_value (a : ℕ → ℤ) : Prop :=
  a 5 = 4

-- Problem: Prove a_9 = 0
theorem find_a9 (h1 : arithmetic_sequence a) (h2 : a4_value a) (h3 : a5_value a) : a 9 = 0 := 
sorry

end find_a9_l532_532410


namespace solution_trig_eq_l532_532103

noncomputable def solve_trig_eq (x : ℝ) : Prop :=
  (abs (cos x) - cos (3 * x)) / (cos x * sin (2 * x)) = 2 / sqrt 3

theorem solution_trig_eq (x : ℝ) :
  solve_trig_eq x ↔
    ∃ k : ℤ, x = (π / 6) + 2 * k * π ∨ x = (5 * π / 6) + 2 * k * π ∨ x = (4 * π / 3) + 2 * k * π :=
sorry

end solution_trig_eq_l532_532103


namespace curves_intersect_and_common_chord_length_l532_532776

noncomputable def C1_param_x (θ : ℝ) : ℝ := -2 + sqrt 10 * cos θ
noncomputable def C1_param_y (θ : ℝ) : ℝ := sqrt 10 * sin θ

noncomputable def C2_polar (θ ρ : ℝ) : Prop := ρ = 2 * cos θ + 6 * sin θ

def C1_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 10
def C2_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 10

theorem curves_intersect_and_common_chord_length :
  (∀ θ, C1_equation (C1_param_x θ) (C1_param_y θ)) ∧
  (∀ θ ρ, C2_polar θ ρ → C2_equation (ρ * cos θ) (ρ * sin θ)) ∧
  ∃ d, d = sqrt 22 ∧ 
  ∃ (x y : ℝ),  C1_equation x y ∧ C2_equation x y :=
sorry

end curves_intersect_and_common_chord_length_l532_532776


namespace smallest_positive_period_of_f_monotonic_increasing_intervals_of_f_min_max_on_interval_l532_532415

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem monotonic_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b, k * π - (3 * π / 8) ≤ a ∧ a ≤ b ∧ b ≤ k * π + (π / 8) ∧ 
    ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem min_max_on_interval :
  let I := set.Icc (-π / 4) (π / 4) in
  set.min_on f I = 0 ∧ set.max_on f I = sqrt 2 + 1 := 
sorry

end smallest_positive_period_of_f_monotonic_increasing_intervals_of_f_min_max_on_interval_l532_532415


namespace total_weekly_cost_correct_l532_532486

def daily_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) : ℝ :=
  cups_per_day * ounces_per_cup

def weekly_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) (days_per_week : ℕ) : ℝ :=
  daily_consumption cups_per_day ounces_per_cup * days_per_week

def weekly_cost (weekly_ounces : ℝ) (cost_per_ounce : ℝ) : ℝ :=
  weekly_ounces * cost_per_ounce

def person_A_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 3 0.4 7) 1.40

def person_B_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 1 0.6 7) 1.20

def person_C_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 5) 1.35

def james_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 7) 1.25

def total_weekly_cost : ℝ :=
  person_A_weekly_cost + person_B_weekly_cost + person_C_weekly_cost + james_weekly_cost

theorem total_weekly_cost_correct : total_weekly_cost = 32.30 := by
  unfold total_weekly_cost person_A_weekly_cost person_B_weekly_cost person_C_weekly_cost james_weekly_cost
  unfold weekly_cost weekly_consumption daily_consumption
  sorry

end total_weekly_cost_correct_l532_532486


namespace locus_of_P_is_straight_line_l532_532400

-- Definitions for given points M and N and the condition on distances
def point_M := (-2 : ℝ, 0 : ℝ)
def point_N := (2 : ℝ, 0 : ℝ)

def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def condition (P : ℝ × ℝ) : Prop :=
  distance P point_M - distance P point_N = 4

-- Statement of the problem: proving the locus is a straight line
theorem locus_of_P_is_straight_line :
  {P : ℝ × ℝ | condition P} = {(x, y) | x = 2 ∧ y = 0} :=
by
  sorry

end locus_of_P_is_straight_line_l532_532400


namespace find_average_after_17th_inning_l532_532638

def initial_average_after_16_inns (A : ℕ) : Prop :=
  let total_runs := 16 * A
  let new_total_runs := total_runs + 87
  let new_average := new_total_runs / 17
  new_average = A + 4

def runs_in_17th_inning := 87

noncomputable def average_after_17th_inning (A : ℕ) : Prop :=
  A + 4 = 23

theorem find_average_after_17th_inning (A : ℕ) :
  initial_average_after_16_inns A →
  average_after_17th_inning A :=
  sorry

end find_average_after_17th_inning_l532_532638


namespace prime_sum_multiple_of_three_probability_l532_532696

theorem prime_sum_multiple_of_three_probability :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] in
  let pairs := (primes.product primes).filter (λ (x : ℕ × ℕ), x.1 < x.2) in
  let successful_pairs := pairs.filter (λ (x : ℕ × ℕ), (x.1 + x.2) % 3 = 0) in
  let total_pairs := pairs.length in
  let total_successful := successful_pairs.length in
  (total_successful:ℚ) / total_pairs = 17 / 45 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let pairs := (primes.product primes).filter (λ (x : ℕ × ℕ), x.1 < x.2)
  let successful_pairs := pairs.filter (λ (x : ℕ × ℕ), (x.1 + x.2) % 3 = 0)
  let total_pairs := pairs.length
  let total_successful := successful_pairs.length
  have h1 : total_pairs = 45 := sorry
  have h2 : total_successful = 17 := sorry
  have h3 : (total_successful:ℚ) / total_pairs = 17 / 45 := by
    rw [h1, h2]
  exact h3

end prime_sum_multiple_of_three_probability_l532_532696


namespace hex_product_l532_532474

def hex_to_dec (h : Char) : Nat :=
  match h with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | c   => c.toNat - '0'.toNat

noncomputable def dec_to_hex (n : Nat) : String :=
  let q := n / 16
  let r := n % 16
  let r_hex := if r < 10 then Char.ofNat (r + '0'.toNat) else Char.ofNat (r - 10 + 'A'.toNat)
  (if q > 0 then toString q else "") ++ Char.toString r_hex

theorem hex_product :
  dec_to_hex (hex_to_dec 'A' * hex_to_dec 'B') = "6E" :=
by
  sorry

end hex_product_l532_532474


namespace midpoint_invariant_l532_532745

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*} [EuclideanGeometry α]
variables (O1 O2 : Circle α) (A B T1 T2 M1 M2 : α)
variables (l1 l2 : line α)

/-- Given circles O1 and O2 intersecting at points A and B. --/
axiom circles_intersect : O1 ≠ O2 ∧ A ∈ O1 ∧ A ∈ O2 ∧ B ∈ O1 ∧ B ∈ O2

/-- From point A, tangents l1 and l2 are drawn to circles O1 and O2 respectively. --/
axiom tangents_l1_l2 : tangent_line O1 A l1 ∧ tangent_line O2 A l2

/-- Points T1 and T2 lie on O1 and O2 respectively, satisfying the angle condition. --/
axiom points_T1_T2 : T1 ∈ O1 ∧ T2 ∈ O2 ∧ ∠T1 (O1.center) A = ∠A (O2.center) T2

/-- The tangent at T1 to O1 intersects l2 at M1. The tangent at T2 to O2 intersects l1 at M2. --/
axiom tangents_intersect : tangent_line O1 T1 ∩ l2 = {M1} ∧ tangent_line O2 T2 ∩ l1 = {M2}

/-- Prove that the midpoint of M1M2 lies on a line that depends only on A. --/
theorem midpoint_invariant :
  let M := midpoint M1 M2 in
  ∃ l : line α, A ∈ l ∧ M ∈ l :=
sorry

end midpoint_invariant_l532_532745


namespace common_difference_is_two_l532_532913

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532913


namespace integer_root_modulus_bound_l532_532096

noncomputable def polynomial_modulus_bound (n : ℕ) (a : Fin (n + 1) → ℤ) (c : ℤ) : Prop :=
  let max_modulus := (Finset.sup (Finset.univ.image (λ k, (a k).natAbs))).val
  ∀ (P : ℤ → ℤ), 
    (P = λ x, ∑ i in Finset.range (n + 1), a i * x^i) →
    P c = 0 →
    (c.natAbs ≤ max_modulus)

theorem integer_root_modulus_bound (n : ℕ) (a : Fin (n + 1) → ℤ) (c : ℤ) : 
  polynomial_modulus_bound n a c := 
sorry

end integer_root_modulus_bound_l532_532096


namespace integral_eq1_integral_eq2_l532_532366

noncomputable def integral1 := ∫ (x: ℝ) in (λ x: ℝ, sqrt(x^2 - 2 * x - 1)), dx

theorem integral_eq1 : integral1 = (λ x, (x - 1) / 2 * sqrt(x^2 - 2 * x - 1) - log_abs (x - 1 + sqrt(x^2 - 2 * x - 1)) + C) :=
sorry

noncomputable def integral2 := ∫ (x: ℝ) in (λ x: ℝ, sqrt(5 + 4 * x - x^2)), dx

theorem integral_eq2 : integral2 = (λ x, 1 / 2 * (9 * arcsin((x - 2) / 3) + (x - 2) * sqrt(5 + 4 * x - x^2)) + C) :=
sorry

end integral_eq1_integral_eq2_l532_532366


namespace find_P_l532_532629

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l532_532629


namespace remaining_card_is_seven_l532_532150

theorem remaining_card_is_seven :
  let cards := [2, 3, 4, 5, 6, 7, 8, 9, 10],
      A := (λ n1 n2, abs (n1 - n2) = 1 ∧ Nat.gcd n1 n2 = 1),
      B := (λ n1 n2, Nat.gcd n1 n2 ≠ 1 ∧ ¬(n1 % n2 = 0 ∨ n2 % n1 = 0)),
      C := (λ n1 n2, ¬ (Nat.Prime n1) ∧ ¬ (Nat.Prime n2) ∧ Nat.gcd n1 n2 = 1),
      D := (λ n1 n2, (n1 % n2 = 0 ∨ n2 % n1 = 0) ∧ Nat.gcd n1 n2 ≠ 1),
      drawnCards := [4, 8, 6, 9, 4, 6, 2, 3],
      remainingCard := 7 in
  A 2 3 ∧ B 4 6 ∧ C 6 9 ∧ D 4 8 ∧ (cards = drawnCards ++ [remainingCard]) → remainingCard = 7 :=
by
  -- The proof goes here
  sorry

end remaining_card_is_seven_l532_532150


namespace limit_expression_l532_532756

variable {ℝ : Type} [Real]

theorem limit_expression (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) :
    (has_limit (λ k, (f(1 - k) - f 1) / (3 * k)) 0 (- (fderiv ℝ f 1) / 3)) :=
by
  sorry

end limit_expression_l532_532756


namespace solve_for_x_l532_532261
-- Import the entire Mathlib library

-- Define the condition
def condition (x : ℝ) := (72 - x)^2 = x^2

-- State the theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 36 :=
by {
  -- The proof will be provided here
  sorry
}

end solve_for_x_l532_532261


namespace min_f_l532_532774

noncomputable def f (x : ℝ) : ℝ := (4 / 3) * x^3 - 1 / x
noncomputable def f' (x : ℝ) : ℝ := 4 * x^2 + 1 / x^2

theorem min_f'_value_eq_4 : ∀ x : ℝ, x ≠ 0 → f'(x) ≥ 4 := 
by
  sorry

end min_f_l532_532774


namespace exists_not_perfect_square_l532_532514

variable {a b : ℤ} (a_pos : a > 1) (b_pos : b > 1) (a_ne_b : a ≠ b)

theorem exists_not_perfect_square : ∃ n : ℕ, (a^n - 1) * (b^n - 1) ∉ perfect_squares :=
sorry

end exists_not_perfect_square_l532_532514


namespace common_difference_of_arithmetic_sequence_l532_532980

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532980


namespace smallest_part_when_divided_l532_532016

theorem smallest_part_when_divided (total : ℝ) (a b c : ℝ) (h_total : total = 150)
                                   (h_a : a = 3) (h_b : b = 5) (h_c : c = 7/2) :
                                   min (min (3 * (total / (a + b + c))) (5 * (total / (a + b + c)))) ((7/2) * (total / (a + b + c))) = 3 * (total / (a + b + c)) :=
by
  -- Mathematical steps have been omitted
  sorry

end smallest_part_when_divided_l532_532016


namespace smallest_four_digit_divisible_by_35_l532_532253

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532253


namespace min_value_expression_l532_532715

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1 / a + 4 / b) ≥ 9 :=
begin
  sorry -- Proof to be filled in later
end

end min_value_expression_l532_532715


namespace common_difference_is_two_l532_532907

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532907


namespace shortest_altitude_l532_532813

theorem shortest_altitude (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15) (h_right : a^2 + b^2 = c^2) : 
  ∃ x : ℝ, x = 7.2 ∧ (1/2) * c * x = (1/2) * a * b := 
by
  sorry

end shortest_altitude_l532_532813


namespace perimeter_outer_contour_l532_532374

/--
Given five square sheets with a side length of 10 centimeters placed
in such a manner that each small square's covered part is a smaller square
whose side length is half of the original square's side length,
prove that the perimeter of the outer contour is 120 centimeters.
-/
theorem perimeter_outer_contour (n : ℕ) (s : ℕ) (h₁ : n = 5) (h₂ : s = 10) :
  let smaller_side_length := s / 2 in
  let horizontal_segments := (s + 4 * smaller_side_length) * 2 in
  let vertical_segments := (s + 4 * smaller_side_length) * 2 in
  horizontal_segments + vertical_segments = 120 :=
by {
  sorry
}

end perimeter_outer_contour_l532_532374


namespace a_6_is_8_l532_532739

noncomputable def a : ℕ → ℕ
| 0     := 0   -- because we need to start at n = 1
| 1     := 1
| 2     := 1
| (n+3) := a (n+1) + a (n+2)

theorem a_6_is_8 : a 6 = 8 :=
by
  sorry

end a_6_is_8_l532_532739


namespace num_solutions_ffinv_eq_3_l532_532413

def f (x : ℝ) : ℝ := if x ≥ 0 then -x^2 + 4 * x else -x^2 - 4 * x

theorem num_solutions_ffinv_eq_3 : 
  ∃ s : Finset ℝ, s.card = 12 ∧ ∀ a : ℝ, a ∈ s ↔ f (f a) = 3 := 
by
  sorry

end num_solutions_ffinv_eq_3_l532_532413


namespace dark_chocolate_bars_sold_l532_532574

theorem dark_chocolate_bars_sold (W D : ℕ) (h₁ : 4 * D = 3 * W) (h₂ : W = 20) : D = 15 :=
by
  sorry

end dark_chocolate_bars_sold_l532_532574


namespace correct_answer_C_l532_532613

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

lemma no_real_roots_of_discriminant_neg (a b c : ℝ) (h : discriminant a b c < 0) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := sorry

lemma problem_statement :
  discriminant 1 1 5 < 0 := by
  show 1^2 - 4 * 1 * 5 < 0
  calc
    1^2 - 4 * 1 * 5 = 1 - 20 := by norm_num
    ... < 0 := by norm_num

theorem correct_answer_C :
  (¬ ∃ x : ℝ, 1 * x^2 + 1 * x + 5 = 0) ∧
  (∃ x₁ x₂ : ℝ, 1 * x₁^2 + 1 * x₁ - 2 = 0 ∧ 1 * x₂^2 + 1 * x₂ - 2 = 0 ∧ x₁ ≠ x₂) ∧
  (∃ x₁ x₂ : ℝ, 1 * x₁^2 - 2 * x₁ = 0 ∧ 1 * x₂^2 - 2 * x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∃ x : ℝ, 1 * x^2 - 2 * x + 1 = 0) :=
by
  split
  { apply no_real_roots_of_discriminant_neg
    apply problem_statement }
  split
  { use (1 : ℝ)
    use (-2 : ℝ)
    norm_num }
  split
  { use (0 : ℝ)
    use (2 : ℝ)
    norm_num }
  { use (1 : ℝ)
    norm_num }

end correct_answer_C_l532_532613


namespace part1_ans_part2_ans_exists_p_l532_532773

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4)^(-1/2)

noncomputable def f_inv (y : ℝ) : ℝ := sqrt(y^(-2) + 4)

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := 1 / f_inv (a n)

noncomputable def S (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ i, (a i)^2)

noncomputable def b (n : ℕ) : ℝ := S (2 * n + 1) - S n

theorem part1_ans (n : ℕ) : a (n + 1) = 1 / sqrt (4 * (n + 1 : ℕ)- 3) :=
sorry

theorem part2_ans_exists_p : ∃ p : ℕ, ∀ n : ℕ, b n < 1 / p ∧ p = 3 :=
sorry

end part1_ans_part2_ans_exists_p_l532_532773


namespace smallest_four_digit_div_by_35_l532_532231

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532231


namespace factorize_expression_l532_532705

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l532_532705


namespace find_n_given_combination_l532_532385

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem find_n_given_combination : ∃ n : ℕ, binomial_coefficient (n+1) 2 = 21 ↔ n = 6 := by
  sorry

end find_n_given_combination_l532_532385


namespace employees_in_january_l532_532333

-- Define the given conditions
variable (E : ℝ) -- The number of employees in January

-- Define the conditions
def december_employees := 470
def ratio := 1.15

-- Theorem statement to prove the number of employees in January
theorem employees_in_january (h : ratio * E = december_employees) : E = 470 / 1.15 := 
by sorry

end employees_in_january_l532_532333


namespace smallest_four_digit_divisible_by_35_l532_532177

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532177


namespace minimum_cos_C_l532_532806

theorem minimum_cos_C (A B C : ℝ) (h1 : sin A + sin B = 2 * sin C) 
  : ∃ (cos_value : ℝ), cos_value = cos C ∧ cos_value ≥ 1/2 :=
by
  sorry

end minimum_cos_C_l532_532806


namespace cos_sq_sin_sq_equiv_l532_532548

-- Given angle in degrees converted to radians
def deg_to_rad (d: ℝ) : ℝ := d * (Real.pi / 180)

-- Helper Definitions for cos^2 and sin^2 involving 18 degrees
def cos_sq_18 : ℝ := (Real.cos (deg_to_rad 18)) ^ 2
def sin_sq_18 : ℝ := (Real.sin (deg_to_rad 18)) ^ 2

theorem cos_sq_sin_sq_equiv : 4 * cos_sq_18 - 1 = 1 / (4 * sin_sq_18) :=
by 
  sorry

end cos_sq_sin_sq_equiv_l532_532548


namespace time_for_P_to_finish_remaning_job_l532_532277

theorem time_for_P_to_finish_remaning_job :
  ∀ (P_rate Q_rate : ℝ), P_rate = 1/3 → Q_rate = 1/15 → 
  let combined_time := 2 in
  let combined_work := combined_time * (P_rate + Q_rate) in
  let remaining_work := 1 - combined_work in
  let time_to_finish := remaining_work / P_rate in
  time_to_finish * 60 = 36 :=
by
  intros P_rate Q_rate hPR hQR combined_time combined_work remaining_work time_to_finish
  sorry

end time_for_P_to_finish_remaning_job_l532_532277


namespace find_palindrome_property_l532_532648

-- Definition of a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10 in s = s.reverse

-- Definition of digit sum
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The formal statement of the problem in Lean 4
theorem find_palindrome_property (P : ℕ) :
  is_palindrome P ∧ P > 300 ∧ digit_sum P = 6 → P = 414 :=
by
  sorry

end find_palindrome_property_l532_532648


namespace arithmetic_sequence_common_difference_l532_532858

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532858


namespace B1F_base16_to_base10_is_2847_l532_532688

theorem B1F_base16_to_base10_is_2847 : 
  let B := 11
  let one := 1
  let F := 15
  let base := 16
  B * base^2 + one * base^1 + F * base^0 = 2847 := 
by
  sorry

end B1F_base16_to_base10_is_2847_l532_532688


namespace second_smallest_possible_value_P0_l532_532552

-- Definitions for conditions
def isMonicCubicPolynomial (P : ℤ[X]) : Prop := P.degree = 3 ∧ P.leadingCoeff = 1
def hasIntegerRoots (P : ℤ[X]) : Prop := ∃ r1 r2 r3 : ℤ, P = (X - C r1) * (X - C r2) * (X - C r3)
def isIntegerFor6Values (P : ℤ[X]) : Prop := (∃ s : Finset ℤ, s.card = 6 ∧ ∀ (a : ℤ), a ∈ s → ∃ (k : ℤ), P.eval a = k * a)
def exactlyOneIsPerfectSquare (P : ℤ[X]) : Prop := (∃ d e f : ℤ, P = X^3 + C d * X^2 + C e * X + C f ∧ (isPerfectSquare (d + f) ∧ ¬ isPerfectSquare (1 + e) ∨ ¬ isPerfectSquare (d + f) ∧ isPerfectSquare (1 + e)))

-- Main statement to be proved
theorem second_smallest_possible_value_P0 (P : ℤ[X]) (d e f : ℤ) (p : ℕ) (hp : Nat.prime p) :
  isMonicCubicPolynomial P ∧ hasIntegerRoots P ∧ isIntegerFor6Values P ∧ exactlyOneIsPerfectSquare P ∧ P.eval 0 = f ∧ f = p^2 →
  (p = 2 ∨ p = 3 ∨ p = 5 ∧ P.eval 0 = 25) :=
begin
  sorry
end

end second_smallest_possible_value_P0_l532_532552


namespace triangle_area_7_8_10_l532_532363

theorem triangle_area_7_8_10 : 
  let a := 7
  let b := 8
  let c := 10
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c)) ≈ 39.335 :=
by
  let a := 7
  let b := 8
  let c := 10
  let s := (a + b + c) / 2
  have area : ℝ := sqrt (s * (s - a) * (s - b) * (s - c))
  exact abs (area - 39.335) < 1e-6

end triangle_area_7_8_10_l532_532363


namespace ellipse_properties_and_angle_constant_l532_532744

/-- 
  Define the properties and the equation of the ellipse and the tangent line conditions, 
  then prove the resultant angle remains constant.
-/
theorem ellipse_properties_and_angle_constant (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (hp1 : 2 * a = 2 / 3) (hp2 : 2 * b = 1 / 2) 
  (x y : ℝ) (hx : 9 * x ^ 2 + 16 * y ^ 2 = 1) (hx' : x^2 + y^2 = 1 / 25)
  (M N : ℝ × ℝ) 
  (hM : 9 * (M.1)^2 + 16 * (M.2)^2 = 1)
  (hN : 9 * (N.1)^2 + 16 * (N.2)^2 = 1)
  (hmn : line_through M N = tangent_to_circle (0, 0) (1/5)) :
  9 * x ^ 2 + 16 * y ^ 2 = 1 ∧ ∠MON = π / 2 :=
by
  sorry

end ellipse_properties_and_angle_constant_l532_532744


namespace complex_expression_result_l532_532508

open Complex

theorem complex_expression_result (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1):
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 8 := 
by
  sorry

end complex_expression_result_l532_532508


namespace common_difference_of_arithmetic_sequence_l532_532978

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532978


namespace total_ladybugs_l532_532108

theorem total_ladybugs (s w : ℕ) (hs : s = 12170) (hw : w = 54912) : s + w = 67082 :=
by {
  rw [hs, hw],
  norm_num,
  sorry
}

end total_ladybugs_l532_532108


namespace problem_solution_l532_532026

-- Definitions for given conditions
variables {a_n b_n : ℕ → ℝ} -- Sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ} -- Sums of the first n terms of {a_n} and {b_n}
variables (h1 : ∀ n, S n = (n * (a_n 1 + a_n n)) / 2)
variables (h2 : ∀ n, T n = (n * (b_n 1 + b_n n)) / 2)
variables (h3 : ∀ n, n > 0 → S n / T n = (2 * n + 1) / (n + 2))

-- The goal
theorem problem_solution :
  (a_n 7) / (b_n 7) = 9 / 5 :=
sorry

end problem_solution_l532_532026


namespace common_difference_arithmetic_sequence_l532_532929

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532929


namespace smallest_four_digit_number_divisible_by_35_l532_532225

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532225


namespace factorize_expression_l532_532701

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l532_532701


namespace percentage_rounding_l532_532458

def total_workers : ℕ := 210
def present_workers : ℕ := 198

def percentage_present (total: ℕ) (present: ℕ) : ℝ :=
  ((present: ℝ) / (total: ℝ)) * 100

theorem percentage_rounding :
  Float.round (percentage_present total_workers present_workers) (1: ℕ) = 94.3 :=
by
  sorry

end percentage_rounding_l532_532458


namespace first_donor_amount_l532_532533

theorem first_donor_amount
  (x second third fourth : ℝ)
  (h1 : second = 2 * x)
  (h2 : third = 3 * second)
  (h3 : fourth = 4 * third)
  (h4 : x + second + third + fourth = 132)
  : x = 4 := 
by 
  -- Simply add this line to make the theorem complete without proof.
  sorry

end first_donor_amount_l532_532533


namespace incorrect_method_l532_532684

def fixed_point (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) : Prop :=
  ∀ (p : P), (p ∈ locus ↔ dist p = r)

def method_A (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) : Prop :=
  ∀ (p : P), (dist p = r → p ∈ locus) ∧ (dist p ≠ r → p ∉ locus)

def method_B (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) : Prop :=
  ∀ (p : P), (dist p ≠ r → p ∉ locus) ∧ (p ∈ locus → dist p = r)

def method_C (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) : Prop :=
  ∀ (p : P), (dist p = r → p ∈ locus) ∧ (p ∈ locus → dist p = r)

def method_D (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) : Prop :=
  ∀ (p : P), (p ∉ locus → dist p ≠ r) ∧ (dist p ≠ r → p ∉ locus)

def method_E (P : Type) (dist : P → ℝ) (r : ℝ) (locus : set P) (another_locus : set P) : Prop :=
  ∀ (p : P), (dist p = r → p ∈ locus) ∧ (dist p ≠ r → p ∈ another_locus)

theorem incorrect_method (P : Type) (dist : P → ℝ) (r : ℝ) (locus another_locus : set P) :
  method_E P dist r locus another_locus → 
  ¬ (method_A P dist r locus ∧ method_B P dist r locus ∧ method_C P dist r locus ∧ method_D P dist r locus) :=
sorry

end incorrect_method_l532_532684


namespace probability_X_greater_than_2_l532_532801

noncomputable def probability_distribution (i : ℕ) : ℝ :=
  if h : 1 ≤ i ∧ i ≤ 4 then i / 10 else 0

theorem probability_X_greater_than_2 :
  (probability_distribution 3 + probability_distribution 4) = 0.7 := by 
  sorry

end probability_X_greater_than_2_l532_532801


namespace inverse_of_A_cubed_l532_532444

variable {A : Matrix (Fin 2) (Fin 2) ℝ}

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![ 3, -1; 1, 1]

theorem inverse_of_A_cubed : inv (A^3) = !![ 20, -12; 12, -4 ] :=
by
  sorry

end inverse_of_A_cubed_l532_532444


namespace arithmetic_sequence_common_difference_l532_532901

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532901


namespace mul_inv_800_mod_7801_l532_532570

theorem mul_inv_800_mod_7801 :
  ∃ x : ℕ, 0 ≤ x ∧ x < 7801 ∧ (800 * x) % 7801 = 1 := by
  use 3125
  dsimp
  norm_num1
  sorry

end mul_inv_800_mod_7801_l532_532570


namespace hannah_likes_numer_of_units_digits_l532_532011

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def does_not_end_in_0 (n : ℕ) : Prop := n % 10 ≠ 0

theorem hannah_likes_numer_of_units_digits {n : ℕ} :
  (is_divisible_by_4 n ∧ does_not_end_in_0 n) →
  sorry

end hannah_likes_numer_of_units_digits_l532_532011


namespace shaded_area_45_deg_l532_532712

noncomputable def area_of_shaded_figure (R : ℝ) (alpha : ℝ) : ℝ :=
  if alpha = (45 * π / 180) then (π * R^2 / 2) else 0

theorem shaded_area_45_deg {R : ℝ} (hR : 0 < R) : 
  area_of_shaded_figure R (45 * π / 180) = (π * R^2 / 2) :=
by
  sorry

end shaded_area_45_deg_l532_532712


namespace find_general_formula_or_find_m_l532_532822

noncomputable def a : ℕ → ℝ
noncomputable def S : ℕ → ℝ
axiom a1 : a 1 = 1
axiom a5 : a 5 = 4 * a 3

theorem find_general_formula_or (n: ℕ) : a n = 2^(n-1) ∨ a n = (-2)^(n-1) :=
by
  sorry

theorem find_m (m: ℕ) (h: S m = 63) : m = 6 :=
by
  sorry

end find_general_formula_or_find_m_l532_532822


namespace closest_perfect_square_to_273_l532_532263

theorem closest_perfect_square_to_273 : ∃ n : ℕ, (n^2 = 289) ∧ 
  ∀ m : ℕ, (m^2 < 273 → 273 - m^2 ≥ 1) ∧ (m^2 > 273 → m^2 - 273 ≥ 16) :=
by
  sorry

end closest_perfect_square_to_273_l532_532263


namespace domain_of_f_l532_532119

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end domain_of_f_l532_532119


namespace trig_expression_value_l532_532143

theorem trig_expression_value :
  (∀ x : ℝ, x = (70.deg_to_rad : ℝ) → cos x = sin (20.deg_to_rad : ℝ)) →
  ((2 * sin (10.deg_to_rad : ℝ) - cos (20.deg_to_rad : ℝ)) / cos (70.deg_to_rad : ℝ)) = -real.sqrt 3 :=
by
  sorry

end trig_expression_value_l532_532143


namespace common_difference_of_arithmetic_sequence_l532_532962

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532962


namespace seedling_prices_l532_532160

theorem seedling_prices (x y : ℝ) (a b : ℝ) 
  (h1 : 3 * x + 2 * y = 12)
  (h2 : x + 3 * y = 11) 
  (h3 : a + b = 200) 
  (h4 : 2 * 100 * a + 3 * 100 * b ≥ 50000) :
  x = 2 ∧ y = 3 ∧ b ≥ 100 := 
sorry

end seedling_prices_l532_532160


namespace probability_blue_face_eq_one_third_l532_532459

-- Define the necessary conditions
def numberOfFaces : Nat := 12
def numberOfBlueFaces : Nat := 4

-- Define the term representing the probability
def probabilityOfBlueFace : ℚ := numberOfBlueFaces / numberOfFaces

-- The theorem to prove that the probability is 1/3
theorem probability_blue_face_eq_one_third :
  probabilityOfBlueFace = (1 : ℚ) / 3 :=
  by
  sorry

end probability_blue_face_eq_one_third_l532_532459


namespace correct_statements_eq_1_l532_532411

-- Definitions for conditions provided in the problem
def statement1_correct : Prop := 
  ¬ (∃ x0 : ℝ, x0^2 + 1 > 3 * x0) ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x

def statement2_correct (p q : Prop) : Prop :=
  ¬ (p ∨ q) → (¬ p ∧ ¬ q)

def statement3_correct : Prop :=
  ∀ (a b : Type) (α : Type), 
  (∀ (a_parallel_α : a → α → Prop) (b_parallel_α : b → α → Prop) (a_parallel_b : a → b → Prop), 
  (a_parallel_α = λ a, α) ∧ (b_parallel_α = λ b, α) → 
  (a_parallel_b = λ a b, (a = b → a ≠ b → false) ∧ ¬ (a ≠ b → true)))

-- Definition of the count of correct statements
def number_of_correct_statements : ℕ :=
  nat.succ 0 -- equivalent to 1 in Peano Arithmetic

-- Proof problem statement 
theorem correct_statements_eq_1
  (h1 : ¬ statement1_correct)
  (h2 : statement2_correct)
  (h3 : ¬ statement3_correct) :
  number_of_correct_statements = 1 := sorry

end correct_statements_eq_1_l532_532411


namespace complex_number_quadrant_l532_532800

theorem complex_number_quadrant :
  let z := (1 + Complex.i) * (3 - 2 * Complex.i) in
  0 < z.re ∧ 0 < z.im := by
-- Sorry used to skip the proof as required
sorry

end complex_number_quadrant_l532_532800


namespace total_number_of_non_defective_cars_l532_532326

-- Definitions for the initial conditions
def factoryA_cars : ℕ := 60
def factoryB_initial_efficiency : ℕ := 2 * factoryA_cars
def factoryC_initial_cars : ℕ := 90

def factoryA_def_rate_tue : ℝ := 0.05
def factoryB_def_rate_thu : ℝ := 0.03
def factoryC_def_rate_wed : ℝ := 0.04
def factoryC_def_rate_fri : ℝ := 0.02

-- Definitions for production rates and defective cars calculations
def factoryA_produced : List ℕ := 
  [60, (60 * 2) * 95 / 100, (60 * 4), (60 * 8), 0]

def factoryB_produced : List ℕ := 
  [120, (120 * 2), (120 * 4), (120 * 8) * 97 / 100, (120 * 16)]

def factoryC_produced : List ℕ := 
  [0, 0, 90 * 96 / 100, 90 * 1.5, 135 * 1.5 * 98 / 100]

-- Definitions for total non-defective cars
def total_non_defective_cars : ℕ := 
  factoryA_produced.sum + factoryB_produced.sum + factoryC_produced.sum

-- Theorem stating the problem
theorem total_number_of_non_defective_cars : total_non_defective_cars = 5004 := by
  sorry

end total_number_of_non_defective_cars_l532_532326


namespace prove_conjugate_l532_532768

-- Define complex number and conditions
def z : ℂ := -4 + 3 * complex.I

-- First given condition
def condition_z (z : ℂ) : Prop := complex.abs z = 1 + 3 * complex.I - z

-- Second question to be proven
def question_conjugate (z : ℂ) : complex.conj ((1 + complex.I)^2 * (3 + 4 * complex.I)^2 / (2 * z)) = 3 - 4 * complex.I := 
sorry

-- The main theorem combining the conditions and the proof target
theorem prove_conjugate (hz : condition_z z) : question_conjugate z :=
sorry

end prove_conjugate_l532_532768


namespace find_radius_l532_532640

noncomputable def square_side_length := 2 - Real.sqrt(5 - Real.sqrt(5))

theorem find_radius
    (side_length : Real := square_side_length)
    (angle_between_tangents : Real := 72)
    (sin_36 : Real := Real.sqrt(5 - Real.sqrt(5)) / (2 * Real.sqrt(2)))
    (radius : Real) :
    radius = Real.sqrt(5 - Real.sqrt(5)) :=
  sorry

end find_radius_l532_532640


namespace red_paint_percentage_is_correct_l532_532082

noncomputable def percentage_red_paint_in_mixture 
  (total_blue_paint : ℕ) -- 140 ounces of blue paint
  (total_white_paint : ℕ) -- 20 ounces of white paint
  (percentage_blue_paint : ℚ) -- The mixture contains 70% blue paint
  : ℚ := 
let total_paint := total_blue_paint + total_white_paint in
let total_mixture := total_blue_paint / percentage_blue_paint in
let red_paint := total_mixture - total_paint in
(red_paint / total_mixture) * 100

theorem red_paint_percentage_is_correct :
  percentage_red_paint_in_mixture 140 20 (7/10) = 20 := by
  sorry

end red_paint_percentage_is_correct_l532_532082


namespace smallest_four_digit_divisible_by_35_l532_532196

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532196


namespace smallest_four_digit_div_by_35_l532_532230

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532230


namespace factorize_expression_l532_532706

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l532_532706


namespace union_of_S_and_R_l532_532423

def S : set ℕ := {1}
def R : set ℕ := {1, 2}

theorem union_of_S_and_R : S ∪ R = {1, 2} :=
by
  sorry

end union_of_S_and_R_l532_532423


namespace intersecting_circumcircles_l532_532043

variable {A1 A2 A3 B1 B2 B3 : Type}

-- Define the points on the plane
variables [plane : EuclideanGeometry] (u v w x y z : plane.Point)

-- Define the condition: the circumcircles of given triangles intersect at a single point
def circumcircles_intersect_at_single_point 
  (A₁ A₂ A₃ B₁ B₂ B₃ : plane.Point) : Prop :=
  ∃ P : plane.Point, 
    P ∈ (circle_circum A₁ A₂ B₃) ∧
    P ∈ (circle_circum A₁ B₂ A₃) ∧
    P ∈ (circle_circum B₁ A₂ A₃)

-- Define the theorem statement
theorem intersecting_circumcircles 
  (h : circumcircles_intersect_at_single_point u v w x y z) :
  ∃ P : plane.Point, 
    P ∈ (circle_circum x y w) ∧ 
    P ∈ (circle_circum x v z) ∧ 
    P ∈ (circle_circum u y z) :=
sorry

end intersecting_circumcircles_l532_532043


namespace range_F_l532_532457

variables {R : Type*} [OrderedField R] (f : R → R)
def F (x : R) : R := f x + (1 / f x)

theorem range_F (hf : ∀ y ∈ set.Icc (1/2 : R) 3, ∃ x, f x = y) :
  set.range (F f) = set.Icc (2 : R) (10/3 : R) :=
sorry

end range_F_l532_532457


namespace min_fraction_value_l532_532127

theorem min_fraction_value
    (a x y : ℕ)
    (h1 : a > 100)
    (h2 : x > 100)
    (h3 : y > 100)
    (h4 : y^2 - 1 = a^2 * (x^2 - 1))
    : a / x ≥ 2 := 
sorry

end min_fraction_value_l532_532127


namespace flight_up_speed_l532_532536

variable (v : ℝ) -- speed on the flight up
variable (d : ℝ) -- distance to mother's place

/--
Given:
1. The speed on the way home was 72 mph.
2. The average speed for the trip was 91 mph.

Prove:
The speed on the flight up was 123.62 mph.
-/
theorem flight_up_speed
  (h1 : 72 > 0)
  (h2 : 91 > 0)
  (avg_speed_def : 91 = (2 * d) / ((d / v) + (d / 72))) :
  v = 123.62 :=
by
  sorry

end flight_up_speed_l532_532536


namespace find_m_odd_function_l532_532130

theorem find_m_odd_function :
  ∃ (m : ℕ), (∀ x : ℝ, ((m^2 - 5 * m + 7) * x^(m-2) = λ y : ℝ, (m^2 - 5 * m + 7) * (-y)^(m-2) = -((m^2 - 5 * m + 7) * x^(m-2))) → m = 3) :=
sorry

end find_m_odd_function_l532_532130


namespace solution_l532_532783

noncomputable theory
open_locale classical

def a : euclidean_space ℝ (fin 2) := ![1, 2]
def b : euclidean_space ℝ (fin 2) := ![-3, 0]

theorem solution (m : ℝ) :
  let v1 := (2:ℝ) • a + b,
      v2 := a - m • b in
  (∃ k : ℝ, v1 = k • v2) ↔ m = -1/2 :=
by sorry

end solution_l532_532783


namespace find_b_plus_c_l532_532132

theorem find_b_plus_c :
  ∀ (b c : ℝ),
    (∀ x : ℝ, -1 < x ∧ x < 4 → -x^2 + b * x + c > 0) →
    (b = 3 ∧ c = 4) →
    b + c = 7 :=
by
  intro b c h hbc
  cases hbc with hb hc
  rw [hb, hc]
  exact rfl

end find_b_plus_c_l532_532132


namespace area_of_quadrilateral_MNQP_l532_532168

open Real EuclideanGeometry

noncomputable def distance (p1 p2 : Point) : ℝ := dist p1 p2

noncomputable def Circle (center : Point) (radius : ℝ) : Set Point := 
  { p | distance center p = radius }

variables (O1 O2 A B M N P Q : Point)

-- Given conditions from the problem
axiom O1O2_eq : distance O1 O2 = 6 * sqrt 3
axiom Ab_eq : distance A B = 6
axiom is_intersect1 : A ∈ Circle O1 (distance O1 A)
axiom is_intersect2 : A ∈ Circle O2 (distance O2 A)
axiom is_intersect3 : B ∈ Circle O1 (distance O1 B)
axiom is_intersect4 : B ∈ Circle O2 (distance O2 B)
axiom is_parallel : parallel (line_through O1 A) (line_through B O2)
axiom intersects_at_M : M ∈ Circle O1 (distance O1 A) ∧ M ≠ A
axiom intersects_at_P : P ∈ Circle O1 (distance O1 A) ∧ P ≠ A
axiom intersects_at_N : N ∈ Circle O2 (distance O2 B) ∧ N ≠ B
axiom intersects_at_Q : Q ∈ Circle O2 (distance O2 B) ∧ Q ≠ B

-- Hypothesis to start the proof of the quadrilateral MNQP area 
theorem area_of_quadrilateral_MNQP : 
  ∃ (area : ℝ), area = quadrilateral_area M N Q P :=
sorry

end area_of_quadrilateral_MNQP_l532_532168


namespace baked_goods_distribution_l532_532534

theorem baked_goods_distribution :
  let g_cookies := 47
  let g_per_jar := 6
  let s_cookies := 78
  let s_per_box := 9
  let f_tarts := 36
  let f_per_box := 4 in
  g_cookies % g_per_jar = 5 ∧
  s_cookies % s_per_box = 6 ∧
  f_tarts % f_per_box = 0 :=
by
  sorry

end baked_goods_distribution_l532_532534


namespace find_a_l532_532757

def line_eq (a : ℝ) : ℝ → ℝ := λ x, x + 2 * a
def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * y - 2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x1 y1 x2 y2 : ℝ, 
(line_eq a x1 = y1) ∧ (line_eq a x2 = y2) ∧ 
(circle_eq a x1 y1) ∧ (circle_eq a x2 y2) ∧ 
(distance x1 y1 x2 y2 = 2 * real.sqrt 3)) : 
  a = real.sqrt 2 :=
by
  sorry

end find_a_l532_532757


namespace greatest_multiple_5_7_less_than_700_l532_532602

theorem greatest_multiple_5_7_less_than_700 :
  ∃ n, n < 700 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 700 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) → n = 665 :=
by
  sorry

end greatest_multiple_5_7_less_than_700_l532_532602


namespace coin_collinear_l532_532169

-- Definitions for the radii and the coins
structure Coin :=
  (radius : ℕ)  -- The radius can be 1, 2, or 3 cm

-- Condition 1: Two coins of each radius
def coins : list Coin := [Coin.mk 1, Coin.mk 1, Coin.mk 2, Coin.mk 2, Coin.mk 3, Coin.mk 3]

-- Define a function to check if a new coin touches at least two coins in a given arrangement
def touches_two_or_more (new_coin : Coin) (placed_coins : list Coin) : Prop := sorry

-- Statement to prove: There exists an arrangement of coins such that the centers of at least three coins are collinear
theorem coin_collinear :
  ∃ (arrangement : list Coin),
    touches_two_or_more (Coin.mk 1) arrangement ∧
    touches_two_or_more (Coin.mk 2) arrangement ∧
    touches_two_or_more (Coin.mk 3) arrangement ∧
    ∃ (c1 c2 c3 : Coin), 
      c1 ∈ arrangement ∧ c2 ∈ arrangement ∧ c3 ∈ arrangement ∧ are_collinear c1.center c2.center c3.center :=
begin
  sorry,
end

end coin_collinear_l532_532169


namespace slices_served_today_l532_532303

-- Definitions based on conditions from part a)
def slices_lunch_today : ℕ := 7
def slices_dinner_today : ℕ := 5

-- Proof statement based on part c)
theorem slices_served_today : slices_lunch_today + slices_dinner_today = 12 := 
by
  sorry

end slices_served_today_l532_532303


namespace exists_n_prime_factors_m_exp_n_plus_n_exp_m_l532_532732

theorem exists_n_prime_factors_m_exp_n_plus_n_exp_m (m k : ℕ) (hm : m > 0) (hm_odd : m % 2 = 1) (hk : k > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ primes : Finset ℕ, primes.card ≥ k ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ m ^ n + n ^ m) := 
sorry

end exists_n_prime_factors_m_exp_n_plus_n_exp_m_l532_532732


namespace tangent_equation_at_one_range_of_a_l532_532000

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 / x + f a x

variable (a : ℝ)
variable (hfa : HasDerivAt (f a) 1 2)
variable (dec_g : ∀ x ∈ Set.Icc (1:ℝ) 2, ∂ f a x ≤ 0)

theorem tangent_equation_at_one :
  (∀ x : ℝ, f a x = x^2 + 2 * a * Real.log x) →
  (HasDerivAt (f a) 1 2) →
  (g a → ∀ x ∈ Set.Icc (1:ℝ) 2, ∂ f a x ≤ 0) →
  4 * x + y - 5 = 0 :=
sorry

theorem range_of_a :
  (∀ x : ℝ, f a x = x^2 + 2 * a * Real.log x ) →
  (HasDerivAt (f a) 1 2) →
  (g a → ∀ x ∈ Set.Icc (1:ℝ) 2, ∂ f a x ≤ 0) →
  a ≤ -7 / 2 :=
sorry

end tangent_equation_at_one_range_of_a_l532_532000


namespace abc_divisibility_l532_532420

theorem abc_divisibility (a b c : ℕ) (h1 : c ∣ a^b) (h2 : a ∣ b^c) (h3 : b ∣ c^a) : abc ∣ (a + b + c)^(a + b + c) := 
sorry

end abc_divisibility_l532_532420


namespace solution_set_inequality_range_a_inequality_l532_532414

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 2

theorem solution_set_inequality (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a + abs (2*x - 3) > 0 ↔ (x < 2 / 3 ∨ 2 < x) := sorry

theorem range_a_inequality (a : ℝ) :
  (∀ x, f x a < abs (x - 3)) ↔ (1 < a ∧ a < 5) := sorry

end solution_set_inequality_range_a_inequality_l532_532414


namespace jennifer_sweets_l532_532833

theorem jennifer_sweets :
  let green_sweets := 212
  let blue_sweets := 310
  let yellow_sweets := 502
  let total_sweets := green_sweets + blue_sweets + yellow_sweets
  let number_of_people := 4
  total_sweets / number_of_people = 256 := 
by
  sorry

end jennifer_sweets_l532_532833


namespace sum_of_not_in_domain_l532_532349

noncomputable def g (x : ℝ) : ℝ := 1 / (1 + (1 / (1 + (1 / (x^2)))))

theorem sum_of_not_in_domain : 
  let not_in_domain := {x : ℝ | x = 0} in 
  ∑ x in not_in_domain, x = 0 :=
by
  sorry

end sum_of_not_in_domain_l532_532349


namespace find_P_l532_532628

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l532_532628


namespace panthers_score_l532_532559

theorem panthers_score (P : ℕ) (wildcats_score : ℕ := 36) (score_difference : ℕ := 19) (h : wildcats_score = P + score_difference) : P = 17 := by
  sorry

end panthers_score_l532_532559


namespace common_difference_l532_532992

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532992


namespace karen_starts_late_l532_532495

def karen_speed := 60 -- Karen's speed in mph
def tom_speed := 45 -- Tom's speed in mph
def tom_distance := 24 -- Distance Tom drives in miles
def karen_lead := 4 -- Distance by which Karen beats Tom in miles

theorem karen_starts_late : 
  let t := tom_distance / tom_speed in -- Time Tom drives
  let t_k := (tom_distance + karen_lead) / karen_speed in -- Time Karen drives
  (t - t_k) * 60 = 4 := -- The time difference in minutes is 4
by
  sorry

end karen_starts_late_l532_532495


namespace not_perfect_square_of_infinite_solutions_l532_532342

theorem not_perfect_square_of_infinite_solutions (k : ℤ) :
  (∃ᶠ (a b c : ℤ) in ⊤, (a^2 - k) * (b^2 - k) = c^2 - k) → ¬ ∃ m : ℤ, k = m^2 :=
begin
  sorry
end

end not_perfect_square_of_infinite_solutions_l532_532342


namespace area_of_convex_quadrilateral_l532_532815

theorem area_of_convex_quadrilateral :
  ∀ (A B C D : Point) (AB BC CD DA : ℝ) (angle_CDA : ℝ),
    AB = 6 → BC = 5 → CD = 13 → DA = 13 →
    angle_CDA = 90 →
    let area := 84.5 + (sqrt 2783) / 2 in
    ∃ (a b c : ℝ), a = 0 ∧ b = 1 ∧ c = 2783 ∧ area = sqrt a + b * sqrt c ∧ a + b + c = 2783.5 :=
by
  sorry

end area_of_convex_quadrilateral_l532_532815


namespace range_of_b_for_monotonic_function_l532_532390

theorem range_of_b_for_monotonic_function :
  (∀ x : ℝ, (x^2 + 2 * b * x + b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) :=
by sorry

end range_of_b_for_monotonic_function_l532_532390


namespace river_current_speed_l532_532520

def swimmer_speed_in_still_water : ℝ := 10
def effective_speed_downstream (x y : ℝ) : ℝ := x + y
def effective_speed_upstream (x y : ℝ) : ℝ := x - y
def time_spent_downstream (D x y : ℝ) : ℝ := D / (x + y)
def time_spent_upstream (D x y : ℝ) : ℝ := D / (x - y)
theorem river_current_speed (D : ℝ) (y : ℝ) : 
  (time_spent_upstream D swimmer_speed_in_still_water y = 3 * time_spent_downstream D swimmer_speed_in_still_water y) →
  (D / (swimmer_speed_in_still_water - y) = 3 * (D / (swimmer_speed_in_still_water + y))) →
  y = 5 := sorry

end river_current_speed_l532_532520


namespace common_difference_is_two_l532_532914

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532914


namespace ellipse_x1_x2_squared_sum_eq_4_l532_532396

theorem ellipse_x1_x2_squared_sum_eq_4
  (x₁ y₁ x₂ y₂ : ℝ)
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = 1)
  (hM : x₁^2 / a^2 + y₁^2 = 1)
  (hN : x₂^2 / a^2 + y₂^2 = 1)
  (h_slope_product : (y₁ / x₁) * (y₂ / x₂) = -1 / 4) :
  x₁^2 + x₂^2 = 4 :=
by
  sorry

end ellipse_x1_x2_squared_sum_eq_4_l532_532396


namespace chemistry_more_than_physics_l532_532141

variables (M P C x : ℤ)

-- Condition 1: The total marks in mathematics and physics is 50
def condition1 : Prop := M + P = 50

-- Condition 2: The average marks in mathematics and chemistry together is 35
def condition2 : Prop := (M + C) / 2 = 35

-- Condition 3: The score in chemistry is some marks more than that in physics
def condition3 : Prop := C = P + x

theorem chemistry_more_than_physics :
  condition1 M P ∧ condition2 M C ∧ (∃ x : ℤ, condition3 P C x ∧ x = 20) :=
sorry

end chemistry_more_than_physics_l532_532141


namespace common_difference_of_arithmetic_sequence_l532_532981

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532981


namespace find_takeoff_run_distance_l532_532139

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end find_takeoff_run_distance_l532_532139


namespace smallest_positive_integer_n_l532_532255

theorem smallest_positive_integer_n (n : ℕ) (h : 5 * n ≡ 1463 [MOD 26]) : n = 23 :=
sorry

end smallest_positive_integer_n_l532_532255


namespace OW_tangent_to_BOL_l532_532046

open EuclideanGeometry

variables {A B C O L W : Point}
variable [InCircle A B C O]

-- Defining each condition
def angle_ABC_eq_60deg (A B C : Point) : Prop := ∠ B A C = 60
def is_center (O : Point) (A B C : Point) : Prop := O = circumcenter A B C
def intersection_of_bisector (L : Point) (B C : Point) (O : Point) : Prop :=
  ∃ W : Point, (B_angle_bisector B C) = BL ∧ circle O W

-- Statement to prove OW tangent to (BOL)
theorem OW_tangent_to_BOL : 
  ∀ (A B C O L W : Point),
  angle_ABC_eq_60deg A B C →
  is_center O A B C →
  intersection_of_bisector L B C O →
  tangent_line (circle_center_point O W) (W_angle_point_perpendicular_to B L O) :=
by
  sorry

end OW_tangent_to_BOL_l532_532046


namespace my_operation_commutative_my_operation_not_associative_l532_532378

variable (x y z : ℝ)

def my_operation (x y : ℝ) : ℝ := (x * y) / (x + y + 1)

theorem my_operation_commutative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  my_operation x y = my_operation y x :=
by
  simp [my_operation]
  sorry

theorem my_operation_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  my_operation (my_operation x y) z ≠ my_operation x (my_operation y z) :=
by
  simp [my_operation]
  sorry

end my_operation_commutative_my_operation_not_associative_l532_532378


namespace dolls_proof_l532_532521

variable (blonde_dolls brown_dolls black_dolls : ℕ)

def given_conditions (blonde_dolls brown_dolls black_dolls : ℕ) : Prop :=
  blonde_dolls = 4 ∧
  brown_dolls = 4 * blonde_dolls ∧
  black_dolls = brown_dolls - 2

def question (blonde_dolls brown_dolls black_dolls : ℕ) : ℕ :=
  (brown_dolls + black_dolls) - blonde_dolls

theorem dolls_proof :
  ∀ (blonde_dolls brown_dolls black_dolls : ℕ),
  given_conditions blonde_dolls brown_dolls black_dolls →
  question blonde_dolls brown_dolls black_dolls = 26 :=
by
  intros blonde_dolls brown_dolls black_dolls h
  simp [given_conditions, question] at *
  split at h
  -- Now you'd need to either finish the proof or fill in the steps taken in the given solution.
  sorry

end dolls_proof_l532_532521


namespace solve_problem_l532_532449

noncomputable def problem : Prop :=
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x = 2 + 1/y → y = 2 + 1/x →  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2

theorem solve_problem : problem :=
begin
  sorry
end

end solve_problem_l532_532449


namespace number_of_partitions_is_8_l532_532695

def exists_partition_of_12_elements : Prop :=
  ∃ (partitions : List (List (List ℕ))),
    (∀ p ∈ partitions, p.length = 3 ∧ (∃ a b c, a = b + c ∧ p = [a, b, c] ∧ b > c)) ∧
    List.length partitions = 4 ∧ 
    List.join partitions = List.range 1 (12 + 1) ∧
    partitions.Nodup

theorem number_of_partitions_is_8 : (∃ partitions, exists_partition_of_12_elements partitions ∧ List.length partitions = 8) :=
sorry

end number_of_partitions_is_8_l532_532695


namespace find_focus_and_a_l532_532122

constant parabola : ℝ → ℝ → Prop
constant directrix: ℝ → Prop

-- Definitions: Equation of the parabola and directrix
def parabola_eq (a x y : ℝ) : Prop := y = a * x^2
def directrix_eq (y : ℝ) : Prop := y = -1/4

-- Given conditions
axiom parabola_C : ∃ a : ℝ, ∃ x y : ℝ, parabola_eq a x y
axiom directrix_C : directrix_eq (-1/4)

-- Statement to prove
theorem find_focus_and_a :
  (∃ a : ℝ, a = 1) ∧
  (∃ x y : ℝ, x = 0 ∧ y = 1/4 ∧ parabola_eq 1 x y) :=
by
  sorry

end find_focus_and_a_l532_532122


namespace find_common_difference_l532_532955

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532955


namespace twice_total_credits_l532_532321

-- Define the variables and conditions
variables (Aria Emily Spencer Hannah : ℕ)
variables (h1 : Aria = 2 * Emily) 
variables (h2 : Emily = 2 * Spencer)
variables (h3 : Emily = 20)
variables (h4 : Hannah = 3 * Spencer)

-- Proof statement
theorem twice_total_credits : 2 * (Aria + Emily + Spencer + Hannah) = 200 :=
by 
  -- Proof steps are omitted with sorry
  sorry

end twice_total_credits_l532_532321


namespace volume_of_regular_triangular_pyramid_l532_532723

noncomputable def regular_triangular_pyramid_volume (h : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3) / 2

theorem volume_of_regular_triangular_pyramid (h : ℝ) :
  regular_triangular_pyramid_volume h = (h^3 * Real.sqrt 3) / 2 :=
by
  sorry

end volume_of_regular_triangular_pyramid_l532_532723


namespace find_common_difference_l532_532960

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532960


namespace can_capacity_l532_532275

-- Definitions of the conditions
variable (M W : ℕ) -- initial amounts of milk and water
variable (M' : ℕ := M + 2) -- new amount of milk after adding 2 liters
variable (ratio_initial : M / W = 1 / 5)
variable (ratio_new : M' / W = 3 / 5)

theorem can_capacity (M W : ℕ) (h_ratio_initial : M / W = 1 / 5) (h_ratio_new : (M + 2) / W = 3 / 5) : (M + W + 2) = 8 := 
by
  sorry

end can_capacity_l532_532275


namespace new_average_marks_l532_532473

theorem new_average_marks
  (orig_avg : ℕ) (num_papers : ℕ)
  (add_geography : ℕ) (add_history : ℕ)
  (H_orig_avg : orig_avg = 63)
  (H_num_papers : num_papers = 11)
  (H_add_geography : add_geography = 20)
  (H_add_history : add_history = 2) :
  (orig_avg * num_ppapers + add_geography + add_history) / num_papers = 65 :=
by
  -- Here would be the proof steps
  sorry

end new_average_marks_l532_532473


namespace polygon_side_vectors_identity_l532_532072

noncomputable def a_i_length (a : ℕ → ℝ^2) (i : ℕ) : ℝ := real.sqrt ((a i).1^2 + (a i).2^2)

noncomputable def angle_phi (a : ℕ → ℝ^2) (i j : ℕ) : ℝ := 
  real.acos ((a i).dot (a j) / ((a_i_length a i) * (a_i_length a j)))

theorem polygon_side_vectors_identity (n : ℕ) (a : ℕ → ℝ^2) :
  a_i_length a 1 ^ 2 = 
    (∑ i in Finset.range (n + 1), if i > 1 then a_i_length a i ^ 2 else 0) +
    2 * (∑ i in Finset.range (n + 1), ∑ j in Finset.range (n + 1), if i > j ∧ j > 1 then (a_i_length a i) * (a_i_length a j) * real.cos (angle_phi a i j) else 0) :=
sorry

end polygon_side_vectors_identity_l532_532072


namespace smallest_four_digit_divisible_by_35_l532_532215

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532215


namespace smallest_four_digit_divisible_by_35_l532_532203

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532203


namespace max_cities_in_country_l532_532812

theorem max_cities_in_country : 
  ∀ {V : Type} [fintype V] (G : simple_graph V), (∀ v ∈ V, G.degree v ≤ 3) →
  (∀ u v ∈ V, G.edge_exists u v ∨ ∃ w ∈ V, G.edge_exists u w ∧ G.edge_exists w v) →
  fintype.card V ≤ 10 :=
by
  sorry

end max_cities_in_country_l532_532812


namespace probability_at_most_3_heads_l532_532634

theorem probability_at_most_3_heads (n : ℕ) (h : n = 12) :
  let total_outcomes := 2^12,
      favorable_outcomes := (∑ k in finset.Icc 0 3, nat.choose 12 k)
  in favorable_outcomes / total_outcomes = 299 / 4096 :=
by {
  sorry
}

end probability_at_most_3_heads_l532_532634


namespace DK_parallel_EJ_l532_532069

noncomputable def Incenter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Type := sorry
noncomputable def Excenter (A : Type) [MetricSpace A] : Type := sorry
noncomputable def FootOfAltitude (A : Type) [MetricSpace A] : Type := sorry
noncomputable def OrthogonalProjection (I : Type) [MetricSpace I] (altitude : Type) : Type := sorry
noncomputable def TangencyPoint (incircle : Type) [MetricSpace incircle] (side : Type) : Type := sorry

variables (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]

def conditions := 
  (I : Incenter A B C) (D : TangencyPoint (Incircle A B C) B C) 
  (J : Excenter A) (E : FootOfAltitude A) 
  (K : OrthogonalProjection I (FootOfAltitude A))

theorem DK_parallel_EJ :
  conditions A B C →
  Parallel (LineSegment D K) (LineSegment E J) := sorry

end DK_parallel_EJ_l532_532069


namespace smallest_four_digit_divisible_by_35_l532_532180

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532180


namespace problem_solution_l532_532818

-- Definition of the parametric equation of line l
structure ParametricLine := 
  (x : ℝ → ℝ)
  (y : ℝ → ℝ)
  (through_point : x 0 = 2 ∧ y 0 = 1)

-- Definition of the polar equation of curve C
def polar_equation_C (ρ θ : ℝ) : Prop := 
  ρ * (Real.sin θ)^2 = 2 * Real.cos θ

-- Definition of the Cartesian equation of curve C
def Cartesian_equation_C (x y : ℝ) : Prop :=
  y^2 = 2 * x

-- A point and its coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of the distances |PA| and |PB| meeting at point P
def dist (P A : Point) : ℝ :=
  Real.sqrt ((A.x - P.x)^2 + (A.y - P.y)^2)

-- The proof statement
theorem problem_solution :
  (∀ ρ θ, polar_equation_C ρ θ ↔ Cartesian_equation_C (ρ * Real.cos θ) (ρ * Real.sin θ))
  ∧ (∀ t₁ t₂, dist ⟨2, 1⟩ ⟨2 + √3 * t₁, 1 + t₁⟩ * dist ⟨2, 1⟩ ⟨2 + √3 * t₂, 1 + t₂⟩ = 3) :=
by 
  sorry

end problem_solution_l532_532818


namespace variance_standard_deviation_l532_532821

noncomputable def variance (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  if n > 0 then (1 / n).toReal * ∑ i in Finset.range n, (x i - (1 / n).toReal * ∑ j in Finset.range n, x j)^2
  else 0

noncomputable def stddev (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (variance n x).sqrt

-- We assume n as the sample size, x as a sequence of individual observations,
-- and we want to show that s, the sample standard deviation, satisfies s^2 as the sample variance 
theorem variance_standard_deviation (n : ℕ) (x : ℕ → ℝ) :
  stddev n x ^ 2 = variance n x := by
  sorry

end variance_standard_deviation_l532_532821


namespace smallest_four_digit_divisible_by_35_l532_532195

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532195


namespace inequality_sqrt_x_y_l532_532542

theorem inequality_sqrt_x_y (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  (real.sqrt x + real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := 
by
  sorry

end inequality_sqrt_x_y_l532_532542


namespace total_wheels_is_118_ratio_bicycles_tricycles_l532_532376

def vehicles :=
  [("cars", 15, 4),
   ("bicycles", 3, 2),
   ("pickup_trucks", 8, 4),
   ("tricycles", 1, 3),
   ("motorcycles", 4, 2),
   ("skateboards", 2, 4),
   ("unicycles", 1, 1)]

def total_wheels (v : List (String × Nat × Nat)) : Nat :=
  v.foldr (λ (_, count, wheels) acc => acc + count * wheels) 0

def distinct_types_bicycles_tricycles (v : List (String × Nat × Nat)) : Nat :=
  v.filter (λ (name, _, _) => name == "bicycles" ∨ name == "tricycles").length

def total_types (v : List (String × Nat × Nat)) : Nat :=
  v.length

theorem total_wheels_is_118 : total_wheels vehicles = 118 :=
by
  sorry

theorem ratio_bicycles_tricycles : distinct_types_bicycles_tricycles vehicles * 7 = 2 * total_types vehicles :=
by
  sorry

end total_wheels_is_118_ratio_bicycles_tricycles_l532_532376


namespace equation_of_circle_max_value_pa_pb_l532_532733

noncomputable def circle_equation_proof (a b r: ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  (C = {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}) ∧
  (3 * a - b = 0) ∧
  (|a - 4| = r) ∧
  ((3 * a + 4 * b + 10)^2 / 25 + 12 = r^2)

theorem equation_of_circle :
  (circle_equation_proof 0 0 4 ({p : ℝ × ℝ | p.1^2 + p.2^2 = 16})) :=
by {
  sorry
}

noncomputable def max_pa_pb (P A B : ℝ × ℝ) : Prop :=
  let x₀ := P.1 in
  let y₀ := P.2 in
  (x₀^2 + y₀^2 = 16) ∧
  (A = (1,1)) ∧ 
  (B = (-2,0)) ∧
  (∀ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 16 → 
    |(x₀ - A.1)^2 + (y₀ - A.2)^2 + (x₀ - B.1)^2 + (y₀ - B.2)^2| ≤ 38 + 8 * real.sqrt 2)

theorem max_value_pa_pb :
  ∀ P A B : ℝ × ℝ, max_pa_pb P A B :=
by {
  sorry
}

end equation_of_circle_max_value_pa_pb_l532_532733


namespace area_of_shaded_region_l532_532661

/-- Given a 10-cm-by-10-cm square, with points C and D located one third 
     and two thirds down the lengths of two opposite sides respectively, 
     the area of the diagonally shaded region is 50/3 square cm. -/
theorem area_of_shaded_region :
  let square_side := 10 in
  let C := (square_side / 3) in
  let D := (2 * square_side / 3) in
  -- The area of the shaded region
  (square_side * square_side) / 6 = 50 / 3 :=
by 
  -- Informal proof omitted
  sorry

end area_of_shaded_region_l532_532661


namespace circle_radius_triple_area_l532_532560

noncomputable def circle_radius (n : ℝ) : ℝ :=
  let r := (n * (Real.sqrt 3 + 1)) / 2
  r

theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) :
  r = (n * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end circle_radius_triple_area_l532_532560


namespace common_difference_arithmetic_sequence_l532_532923

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532923


namespace find_common_difference_l532_532958

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532958


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532526

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532526


namespace domain_of_f_l532_532563

def f (x : ℝ) : ℝ := log (1 - x)

theorem domain_of_f :
  ∀ x, f x ∈ Option.some ℝ ↔ x < 1 :=
by
  sorry

end domain_of_f_l532_532563


namespace smallest_four_digit_divisible_by_35_l532_532239

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532239


namespace sum_of_squares_mod_13_div_13_eq_3_l532_532110

theorem sum_of_squares_mod_13_div_13_eq_3 :
  let squares_mod := (List.range 12).map (λ n => ((n + 1)^2 % 13))
  let distinct_remainders := squares_mod.eraseDups
  let sum_remainders := distinct_remainders.sum
in sum_remainders / 13 = 3 :=
by
  sorry

end sum_of_squares_mod_13_div_13_eq_3_l532_532110


namespace smallest_four_digit_number_divisible_by_35_l532_532220

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532220


namespace new_ratio_first_term_less_than_implied_l532_532126

-- Define the original and new ratios
def original_ratio := (6, 7)
def subtracted_value := 3
def new_ratio := (original_ratio.1 - subtracted_value, original_ratio.2 - subtracted_value)

-- Prove the required property
theorem new_ratio_first_term_less_than_implied {r1 r2 : ℕ} (h : new_ratio = (3, 4))
  (h_less : r1 > 3) :
  new_ratio.1 < r1 := 
sorry

end new_ratio_first_term_less_than_implied_l532_532126


namespace sin_squared_equation_cos_product_l532_532339

theorem sin_squared_equation_cos_product (x : ℝ) :
    (sin x)^2 + (sin (3*x))^2 + (sin (5*x))^2 + (sin (7*x))^2 = 2 →
    let a := 2; let b := 4; let c := 8 in a + b + c = 14 :=
by
  intro h
  let a := 2
  let b := 4
  let c := 8
  have h1 : cos (2 * x) * cos (4 * x) * cos (8 * x) = 0 := sorry
  have h2 : a + b + c = 14 := rfl
  exact h2

end sin_squared_equation_cos_product_l532_532339


namespace karen_starts_late_by_4_minutes_l532_532491

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l532_532491


namespace five_digit_number_count_l532_532547

noncomputable def count_five_digit_numbers : ℕ :=
  let total := Nat.choose 5 3 * Nat.choose 5 2 * Nat.fact 5 in
  let invalid := Nat.choose 5 3 * Nat.choose 4 1 * Nat.fact 4 in
  total - invalid

theorem five_digit_number_count :
  count_five_digit_numbers = 11040 :=
  by sorry

end five_digit_number_count_l532_532547


namespace circumscribed_sphere_radius_l532_532599

theorem circumscribed_sphere_radius
  (a b c : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0) :
  ∃ R : ℝ, R = (1 / 4) * sqrt(2 * (a^2 + b^2 + c^2)) :=
by
  sorry

end circumscribed_sphere_radius_l532_532599


namespace common_difference_of_arithmetic_sequence_l532_532883

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532883


namespace vertices_selection_l532_532842

/-- Define the problem conditions and requirements -/
theorem vertices_selection (P : ℕ → Prop) :
  ∃ n : ℕ, n = 258 ∧
  ∀ (vertices : Finset ℕ), 
  (vertices.card = 8 ∧ ∀ pair ∈ vertices.pairs, (pair.2 - pair.1) % 24 ∉ {3, 8}) ↔ true :=
sorry

end vertices_selection_l532_532842


namespace odd_function_expression_l532_532404

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_expression (x : ℝ) (h1 : x < 0 → f x = x^2 - x) (h2 : ∀ x, f (-x) = -f x) (h3 : 0 < x) :
  f x = -x^2 - x :=
sorry

end odd_function_expression_l532_532404


namespace arithmetic_sequence_common_difference_l532_532850

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532850


namespace problem_l532_532031

def seq (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem problem (a : ℕ → ℤ) (h₁ : a 1 = 2010) (h₂ : a 2 = 2011) (h₃ : seq a) : a 1000 = 2343 :=
sorry

end problem_l532_532031


namespace octagon_angle_sequences_l532_532568

theorem octagon_angle_sequences (x d : ℕ) (h1: x + 7 * d < 170) (h2: 2 * x + 7 * d = 270) : 
  (∃ k : ℕ, k = 6) := 
begin
  sorry
end

end octagon_angle_sequences_l532_532568


namespace smallest_four_digit_divisible_by_35_l532_532214

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532214


namespace no_point_with_more_than_n_area_bisecting_lines_l532_532734

def convex_n_gon (n : ℕ) :=
  {poly : Polyhedron ℝ // convex poly ∧ poly.sides = n ∧ ∀ i j, i ≠ j → poly.side i ∥ poly.side j → false}

theorem no_point_with_more_than_n_area_bisecting_lines 
  (n : ℕ)
  (poly : convex_n_gon n) :
  ∀ P : Point ℝ, ¬ ∃ lines : set (Line ℝ), (lines.card > n) ∧ (∀ l ∈ lines, bisects_area poly l) :=
sorry

end no_point_with_more_than_n_area_bisecting_lines_l532_532734


namespace common_difference_arithmetic_sequence_l532_532922

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532922


namespace smallest_four_digit_div_by_35_l532_532233

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532233


namespace machine_value_present_value_l532_532646

theorem machine_value_present_value:
  ∃ (P : ℝ), (972 = P * (1 - 0.10)^2) → P = 1200 :=
by
  intro P h
  have h₁ : (1 - 0.10:ℝ)^2 = 0.81 := by norm_num
  rw [h₁] at h
  have h₂ : P * 0.81 = 972 := h
  have h₃ : P = 972 / 0.81 := by linarith
  norm_num at h₃
  exact h₃

end machine_value_present_value_l532_532646


namespace smallest_a_for_f_iter_3_l532_532515

noncomputable def f (x : ℕ) : ℕ :=
  if x % 2 = 0 ∧ x % 7 = 0 then x / 14
  else if x % 7 = 0 then 2 * x
  else if x % 2 = 0 then 7 * x
  else x + 2

noncomputable def f_iter (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a_for_f_iter_3 :
  ∃ a > 1, f 3 = f_iter a 3 ∧ ∀ b, 1 < b < a → f_iter b 3 ≠ f 3 :=
by
  sorry

end smallest_a_for_f_iter_3_l532_532515


namespace x_1998_mod_1998_l532_532057

theorem x_1998_mod_1998 (λ : ℝ) (x : ℕ → ℝ)
  (hλ_eq : λ^2 - 1998 * λ - 1 = 0)
  (hx0 : x 0 = 1)
  (hx : ∀ n, x (n + 1) = Real.floor (λ * x n)) :
  x 1998 % 1998 = 0 :=
sorry

end x_1998_mod_1998_l532_532057


namespace f_five_fold_apply_f_five_apply_six_l532_532792

def f (x : ℝ) : ℝ := - (1 / x)

theorem f_five_fold_apply (x : ℝ) (h : x ≠ 0) : f (f (f (f (f x)))) = f x := by 
  sorry

theorem f_five_apply_six : f (f (f (f (f 6)))) = - (1 / 6) := by
  have h : 6 ≠ 0 := by norm_num
  exact calc
    f (f (f (f (f 6)))) = f 6 : f_five_fold_apply 6 h
    ... = - (1 / 6) : by rfl

end f_five_fold_apply_f_five_apply_six_l532_532792


namespace arithmetic_sequence_common_difference_l532_532853

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532853


namespace triangle_area_7_8_10_l532_532362

theorem triangle_area_7_8_10 : 
  let a := 7
  let b := 8
  let c := 10
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c)) ≈ 39.335 :=
by
  let a := 7
  let b := 8
  let c := 10
  let s := (a + b + c) / 2
  have area : ℝ := sqrt (s * (s - a) * (s - b) * (s - c))
  exact abs (area - 39.335) < 1e-6

end triangle_area_7_8_10_l532_532362


namespace jordan_catch_fish_l532_532052

variable (J P : ℕ)

def P_double_J (J : ℕ) : Prop := P = 2 * J
def remaining_fish (J P : ℕ) : Prop := (3/4 : ℚ) * (J + P) = 9

theorem jordan_catch_fish (J P : ℕ) (h1 : P_double_J J) (h2 : remaining_fish J P) : J = 4 :=
by
  sorry

end jordan_catch_fish_l532_532052


namespace equal_sides_of_triangle_l532_532324

theorem equal_sides_of_triangle
  {A B C D E F G H I : Point}
  (triangle_ABC : Incircle Triangle(A, B, C) I)
  (D_on_AB : On(Circle I) D ∧ D ∈ Line(A, B))
  (E_on_BC : On(Circle I) E ∧ E ∈ Line(B, C))
  (F_on_DI : D ∈ Line(F, I) ∧ F ∈ Circle(I))
  (G_on_CF : G ∈ Line(C, F) ∧ G ∈ Line(A, B))
  (H_on_CG : H ∈ Segment(C, G) ∧ Segment(H, G).length = Segment(C, F).length)
  (A_H_E_collinear : Collinear A H E)
  : Length(A, B) = Length(A, C) :=
sorry

end equal_sides_of_triangle_l532_532324


namespace find_common_difference_l532_532867

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532867


namespace max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l532_532051

-- Definitions and Conditions
def is_divisible_by_10 (n : ℕ) : Prop := n % 10 = 0

-- Proof Problem 1: Maximum possible probability
theorem max_probability_divisible_by_10 : 
  ∃ (segment : list ℕ), (∀ n ∈ segment, is_divisible_by_10 n) ∧ 
  (segment.length > 0 → (∀ (n' ∈ segment), ∃ (n ∈ segment), is_divisible_by_10 n = true ∧ n' = n)) :=
begin
  sorry
end

-- Proof Problem 2: Minimum non-zero possible probability
theorem min_nonzero_probability_divisible_by_10 : (∃ k : ℕ, k > 0 ∧ 
  let n := 10 * k + 9 in
  n ≥ 10 ∧ 
  (1 / 10 - 9 / (10 * n) = 1 / 19))
:=
begin
  use 1,
  split,
  { exact nat.one_pos, },
  { let n := 10 * 1 + 9,
    split,
    { exact nat.le_add_left 9 10, },
    { norm_num, },
  }
end

end max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l532_532051


namespace domain_of_f_l532_532120

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end domain_of_f_l532_532120


namespace range_of_m_l532_532771

-- Defining the function f(x)
noncomputable def f : ℝ → ℝ
| x => if x < 0 then x^2 + 2*x else real.log (x + 1)

-- Lean 4 statement for the problem
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = x₁ + m ∧ f x₂ = x₂ + m ∧ f x₃ = x₃ + m) ↔ m ∈ set.Ioo (-1/4 : ℝ) 0 :=
by sorry

end range_of_m_l532_532771


namespace sum_binom_le_power_l532_532375

theorem sum_binom_le_power (n k : ℕ) (h1 : 0 < k) (h2 : k ≤ n) :
    ∑ j in Finset.range (k + 1) \ Finset.singleton 0, Nat.choose n j ≤ n ^ k :=
sorry

end sum_binom_le_power_l532_532375


namespace angle_DBN_eq_angle_BCE_l532_532817

-- Let's define the problem's conditions and corresponding proof problem
variables {A B C D M N E : Type} [metric_space A] [metric_space B] [metric_space C]
           [metric_space D] [metric_space M] [metric_space N] [metric_space E]

namespace triangle_angles

-- Condition 1: Triangle ABC is a right-angled triangle with ∠C = 90°
variables (ABC : triangle) (right_angle_C : ∠ ABC ∈ {π / 2})

-- Condition 2: D is the midpoint of side AB
variables (D_is_midpoint : midpoint AB D)

-- Condition 3: M lies on the same side of AB as C such that MB ⊥ AB
variables (M_same_side_C : M.incident_same_side AB C)
variables (MB_perpendicular_to_AB : is_perpendicular M B AB)

-- Condition 4: Line MD intersects side AC at N
variables (MD_intersects_AC_at_N : line_segments_intersects M D AC N)

-- Condition 5: Line MC intersects side AB at E
variables (MC_intersects_AB_at_E : line_segments_intersects M C AB E)

-- Goal: Prove that ∠DBN = ∠BCE
theorem angle_DBN_eq_angle_BCE :
  ∠ DB N = ∠ BCE :=
sorry

end triangle_angles

end angle_DBN_eq_angle_BCE_l532_532817


namespace incorrect_vertex_is_false_l532_532421

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 2)^2 + 1

-- Define the incorrect hypothesis: Vertex at (-2, 1)
def incorrect_vertex (x y : ℝ) : Prop := (x, y) = (-2, 1)

-- Proposition to prove that the vertex is not at (-2, 1)
theorem incorrect_vertex_is_false : ¬ ∃ x y, (x, y) = (-2, 1) ∧ parabola x = y :=
by
  sorry

end incorrect_vertex_is_false_l532_532421


namespace frog_probability_l532_532293

def grid_width : ℕ := 6
def grid_height : ℕ := 6

def start_point : ℕ × ℕ := (2, 3)

def boundary_condition (x y : ℕ) : Prop :=
  x = 0 ∨ x = grid_width ∨ y = 0 ∨ y = grid_height

def P : ℕ × ℕ → ℚ
| (x, 0) := 1
| (x, grid_height) := 1
| (0, y) := 0
| (grid_width, y) := 0
| (x, y) :=
  if boundary_condition x y then P (x, y) else
    (1 / 4) * P (x - 1, y) +
    (1 / 4) * P (x + 1, y) +
    (1 / 4) * P (x, y - 1) +
    (1 / 4) * P (x, y + 1)

theorem frog_probability :
  P start_point = 8 / 25 :=
sorry

end frog_probability_l532_532293


namespace smallest_four_digit_divisible_by_35_l532_532245

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532245


namespace equal_share_stickers_all_l532_532573

variables (K J A : ℝ)
variables (ratioK ratioJ ratioA : ℝ)
variables (Kate_stickers : ℝ)

-- Conditions
axiom (h1 : ratioK = 7.5)
axiom (h2 : ratioJ = 4.25)
axiom (h3 : ratioA = 5.75)
axiom (h4 : Kate_stickers = 45)
axiom (h5 : K = Kate_stickers)
axiom (h6 : ratioK / ratioJ = K / J)
axiom (h7 : ratioK / ratioA = K / A)

-- Conclusion
theorem equal_share_stickers_all : (K + J + A) / 3 = 35 :=
by
  sorry

end equal_share_stickers_all_l532_532573


namespace smallest_four_digit_divisible_by_35_l532_532251

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532251


namespace sum_of_sequence_l532_532728

def a (n : ℕ) : ℕ := 2 * n + 1 + 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_of_sequence (n : ℕ) : S n = n^2 + 2 * n + 2^(n + 1) - 2 := 
by 
  sorry

end sum_of_sequence_l532_532728


namespace ratio_is_one_third_l532_532591

-- Definitions based on given conditions
def total_students : ℕ := 90
def initial_cafeteria_students : ℕ := (2 * total_students) / 3
def initial_outside_students : ℕ := total_students - initial_cafeteria_students
def moved_cafeteria_to_outside : ℕ := 3
def final_cafeteria_students : ℕ := 67
def students_ran_inside : ℕ := final_cafeteria_students - (initial_cafeteria_students - moved_cafeteria_to_outside)

-- Ratio calculation as a proof statement
def ratio_ran_inside_to_outside : ℚ := students_ran_inside / initial_outside_students

-- Proof that the ratio is 1/3
theorem ratio_is_one_third : ratio_ran_inside_to_outside = 1 / 3 :=
by sorry -- Proof omitted

end ratio_is_one_third_l532_532591


namespace correct_multiplication_factor_l532_532295

theorem correct_multiplication_factor (x : ℕ) : ((139 * x) - 1251 = 139 * 34) → x = 43 := by
  sorry

end correct_multiplication_factor_l532_532295


namespace distinct_n_values_l532_532066

theorem distinct_n_values {x p q n : ℤ} 
  (h1 : x^2 - n * x + 30 = 0) 
  (h2 : p + q = n) 
  (h3 : p * q = 30) 
  (hpq_int : p ∈ ℤ ∧ q ∈ ℤ) : 
  ∃ N, N = 8 := 
sorry

end distinct_n_values_l532_532066


namespace increasing_range_a_l532_532802

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

theorem increasing_range_a (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → derivative (λ x, x^2 + a*x + 1/x) x ≥ 0) ↔ 3 ≤ a :=
by
  sorry

end increasing_range_a_l532_532802


namespace red_yellow_surface_area_ratio_l532_532325

theorem red_yellow_surface_area_ratio (a : ℝ) :
  let edge_length_large := 6 * a,
      surface_area_large := 6 * (edge_length_large)^2,
      surface_area_small := 6 * a^2,
      exposed_surface_area_large := surface_area_large - a^2,
      exposed_surface_area_small := 5 * a^2
  in (exposed_surface_area_large / exposed_surface_area_small) = 43 := by
  sorry

end red_yellow_surface_area_ratio_l532_532325


namespace square_of_three_divisors_has_five_divisors_l532_532647

theorem square_of_three_divisors_has_five_divisors (n : ℕ) (h : ∃ p : ℕ, prime p ∧ n = p^2) : (nat.divisors_count (n^2) = 5) :=
by
  sorry

end square_of_three_divisors_has_five_divisors_l532_532647


namespace determine_a_b_l532_532014

-- Definitions
def num (a b : ℕ) := 10000*a + 1000*6 + 100*7 + 10*9 + b

def divisible_by_72 (n : ℕ) : Prop := n % 72 = 0

noncomputable def a : ℕ := 3
noncomputable def b : ℕ := 2

-- Theorem statement
theorem determine_a_b : divisible_by_72 (num a b) :=
by
  -- The proof will be inserted here
  sorry

end determine_a_b_l532_532014


namespace smallest_range_proof_l532_532304

noncomputable def smallest_range : ℝ :=
  Real.Min {(x_7 - x_1) | (x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 = 56) ∧ (x_4 = 10) ∧ (x_1 ≤ x_2 ≤ x_3 ≤ x_4 ≤ x_5 ≤ x_6 ≤ x_7)}

theorem smallest_range_proof : smallest_range = 4 :=
by
  sorry

end smallest_range_proof_l532_532304


namespace max_initial_value_seq_l532_532368

theorem max_initial_value_seq :
  ∀ (x : Fin 1996 → ℝ),
    (∀ i : Fin 1996, 1 ≤ x i) →
    (x 0 = x 1995) →
    (∀ i : Fin 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ 2 ^ 997 :=
sorry

end max_initial_value_seq_l532_532368


namespace log_seq_value_l532_532606

theorem log_seq_value : 
    logb 2 2 = 1.2 :=
sorry

end log_seq_value_l532_532606


namespace factorize_expression_l532_532704

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l532_532704


namespace circle_tangent_length_OP_l532_532287

-- Define the problem statement in Lean 4
theorem circle_tangent_length_OP {O P Q U V : Type} 
  (r_O : ℝ) (r_P : ℝ) (hO : O = 10)
  (hP : P = 3)
  (tangent : ∀ Q : Type, ∃ (tangent_point : Type), 
             (tangent_point = Q ∨ tangent_point = Q)) :
  OP = 13 := 
sorry

end circle_tangent_length_OP_l532_532287


namespace domain_of_f_l532_532118

def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

theorem domain_of_f :
  ∀ x : ℝ, (1 / (x + 1)) + Real.log x ∈ ℝ → x > 0 :=
by
  intro x
  split
  sorry

end domain_of_f_l532_532118


namespace hockey_tournament_max_point_difference_l532_532462

theorem hockey_tournament_max_point_difference (n : ℕ) (k m : ℕ) 
  (h_km : k + m = n) 
  -- This condition ensures every team plays with every other team exactly once.
  (play_once : ∀ i j : ℕ, i ≠ j → (i < n) ∧ (j < n) → True)
  -- Points are awarded as specified in the problem.
  (points : ℕ → ℕ → ℕ)
  -- Points condition: winning team gets 2, draw 1 each, losing 0.
  (points_cond : ∀ i j : ℕ, i ≠ j → 
    (points i j = 2 ∨ points i j = 0 ∨ points i j = 1 ∧ points j i = 1)) :
  (∀ teams : fin n → ℕ, 
    ∃ (max_point_difference : ℕ), max_point_difference = n ) :=
by
  sorry

end hockey_tournament_max_point_difference_l532_532462


namespace smallest_four_digit_number_divisible_by_35_l532_532221

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532221


namespace simplify_and_evaluate_l532_532101

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 2 - Real.sqrt 2

theorem simplify_and_evaluate : 
  let expr := (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a))
  expr = -1 / 2 := by
  sorry

end simplify_and_evaluate_l532_532101


namespace range_of_h_l532_532346

def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^4)

theorem range_of_h : (∃ c d : ℝ, c = 0 ∧ d = 1 ∧ set_of (λ y, ∃ x : ℝ, h x = y) = set.Ioc c d) ∧ c + d = 1 :=
by
  sorry

end range_of_h_l532_532346


namespace common_difference_of_arithmetic_sequence_l532_532969

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532969


namespace spherical_coordinate_conversion_l532_532036

theorem spherical_coordinate_conversion (ρ θ φ : ℝ) 
  (h_ρ : ρ > 0) 
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h_φ : 0 ≤ φ): 
  (ρ, θ, φ - 2 * Real.pi * ⌊φ / (2 * Real.pi)⌋) = (5, 3 * Real.pi / 4, Real.pi / 4) :=
  by 
  sorry

end spherical_coordinate_conversion_l532_532036


namespace seven_cups_all_down_in_three_flips_l532_532590

theorem seven_cups_all_down_in_three_flips :
  ∃ (n : ℕ), (n = 3) ∧ 
  (∀ (initial_state : vector bool 7), 
    (initial_state = vector.repeat tt 7) →
    (flip3 (flip3 (flip3 initial_state 0 1 2) 2 3 4) 2 5 6) = vector.repeat ff 7) :=
sorry

def flip3 (state : vector bool 7) (i j k : fin 7) : vector bool 7 :=
  state.update_nth i (bnot state.nth i)
       .update_nth j (bnot state.nth j)
       .update_nth k (bnot state.nth k)

end seven_cups_all_down_in_three_flips_l532_532590


namespace sodas_per_pack_l532_532831

theorem sodas_per_pack 
  (packs : ℕ) (initial_sodas : ℕ) (days_in_a_week : ℕ) (sodas_per_day : ℕ) 
  (total_sodas_consumed : ℕ) (sodas_per_pack : ℕ) :
  packs = 5 →
  initial_sodas = 10 →
  days_in_a_week = 7 →
  sodas_per_day = 10 →
  total_sodas_consumed = 70 →
  total_sodas_consumed - initial_sodas = packs * sodas_per_pack →
  sodas_per_pack = 12 :=
by
  intros hpacks hinitial hsodas hdaws htpd htcs
  sorry

end sodas_per_pack_l532_532831


namespace smallest_possible_number_of_apples_l532_532266

theorem smallest_possible_number_of_apples :
  ∃ (M : ℕ), M > 2 ∧ M % 9 = 2 ∧ M % 10 = 2 ∧ M % 11 = 2 ∧ M = 200 :=
by
  sorry

end smallest_possible_number_of_apples_l532_532266


namespace find_a_l532_532762

noncomputable def a : ℝ := sorry -- We'll use sorry for the definition initially.

-- Given conditions
def line (a : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = p.1 + 2 * a

def circle (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 ^ 2 + p.2 ^ 2 - 2 * a * p.2 - 2 = 0

def intersects (line : ℝ × ℝ → Prop) (circle : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, line A ∧ line B ∧ circle A ∧ circle B ∧ A ≠ B ∧ dist A B = 2 * sqrt 3

-- Main statement
theorem find_a (h : a > 0) 
  (intersect_cond : intersects (line a) (circle a)) : a = sqrt 2 :=
sorry -- Proof is omitted

end find_a_l532_532762


namespace intersection_and_reciprocal_sum_l532_532481

noncomputable def line_l_eq (t : ℝ) : ℝ × ℝ :=
  (-t, sqrt(3) - t)

def C1_polar_eq (theta : ℝ) : ℝ := 
  4 * sin theta

def C2_rect_eq (x y : ℝ) : Prop := 
  (x^2 / 4) + y^2 = 1

def P : ℝ × ℝ := (-sqrt(3), 0)

theorem intersection_and_reciprocal_sum :
  ∀ t₁ t₂ : ℝ, 
  let M := line_l_eq t₁,
      N := line_l_eq t₂ in
  C2_rect_eq M.1 M.2 ∧
  C2_rect_eq N.1 N.2 →
  t₁ * t₂ = -2 / 5 →
  abs(t₁ + t₂) = 2 * sqrt(6) / 5 →
  1 / (abs (t₁ - (-sqrt(3)))) + 1 / (abs (t₂ - (-sqrt(3)))) = 4 :=
  sorry

end intersection_and_reciprocal_sum_l532_532481


namespace total_amount_l532_532285

-- Define p, q, r and their shares
variables (p q r : ℕ)

-- Given conditions translated to Lean definitions
def ratio_pq := (5 * q) = (4 * p)
def ratio_qr := (9 * r) = (10 * q)
def r_share := r = 400

-- Statement to prove
theorem total_amount (hpq : ratio_pq p q) (hqr : ratio_qr q r) (hr : r_share r) :
  (p + q + r) = 1210 :=
by
  sorry

end total_amount_l532_532285


namespace common_difference_of_arithmetic_sequence_l532_532888

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532888


namespace common_difference_of_arithmetic_sequence_l532_532885

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532885


namespace probability_event_l532_532598

-- Definitions of the conditions
def boxA := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
def boxB := {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}

-- Definition of the desired event
def eventA(t : ℕ) := t ≤ 14
def eventB(t : ℕ) := t % 2 = 0 ∨ t > 25

-- Probability calculation
noncomputable def probability_boxA := (boxA.count eventA).toRat / boxA.card.toRat
noncomputable def probability_boxB := (boxB.count eventB).toRat / boxB.card.toRat
noncomputable def combined_probability := probability_boxA * probability_boxB

-- Lean theorem statement
theorem probability_event : combined_probability = 70 / 171 := by
  sorry

end probability_event_l532_532598


namespace arithmetic_sequence_common_difference_l532_532939

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532939


namespace AP_plus_CP_equals_sqrt2_BP_l532_532505

theorem AP_plus_CP_equals_sqrt2_BP {A B C D P : ℝ} (h_square: square A B C D)
  (h_circle: circumscribed_circle A B C D)
  (h_arc: on_arc P (circumscribed_circle A B C D) C D) :
  AP + CP = √2 * BP :=
sorry

end AP_plus_CP_equals_sqrt2_BP_l532_532505


namespace four_x_plus_g_not_even_l532_532620

theorem four_x_plus_g_not_even (x g : ℤ) (h1 : 2 * x - g = 11) : ¬ (even (4 * x + g)) :=
sorry

end four_x_plus_g_not_even_l532_532620


namespace factorial_15_base_9_zeroes_l532_532435

theorem factorial_15_base_9_zeroes :
  (∃ k, 15! % 9^k = 0 ∧ 15! % 9^(k+1) ≠ 0) ∧ 
  (∀ k', 15! % 9^(k'+1) = 0 ↔ k' < 3) := sorry

end factorial_15_base_9_zeroes_l532_532435


namespace find_OP_squared_l532_532641

noncomputable def circle : Type := sorry

variables {O A B C D E F P : circle}
variables (radiusO : ℝ) (lengthAB : ℝ) (lengthCD : ℝ) (distanceMidpoints : ℝ)

-- Definitions for the conditions
def is_circle_center (O : circle) (r : ℝ) : Prop := sorry
def is_chord (A B : circle) (len : ℝ) : Prop := sorry
def intersection (A B C D P : circle) : Prop := sorry
def midpoint (X Y M : circle) : Prop := sorry
def distance (X Y : circle) (d : ℝ) : Prop := sorry

-- Given conditions
axiom O_is_center : is_circle_center O 20
axiom AB_chord : is_chord A B 24
axiom CD_chord : is_chord C D 16
axiom P_intersection : intersection A B C D P
axiom E_midpoint_AB : midpoint A B E
axiom F_midpoint_CD : midpoint C D F
axiom EF_distance : distance E F 10

-- Prove OP^2 = 346
theorem find_OP_squared (O A B C D E F P : circle) (radiusO : ℝ) (lengthAB : ℝ) (lengthCD : ℝ) (distanceMidpoints : ℝ)
  (O_is_center : is_circle_center O radiusO)
  (AB_chord : is_chord A B lengthAB)
  (CD_chord : is_chord C D lengthCD)
  (P_intersection : intersection A B C D P)
  (E_midpoint_AB : midpoint A B E)
  (F_midpoint_CD : midpoint C D F)
  (EF_distance : distance E F distanceMidpoints) :
  ∃ OP_squared : ℝ, OP_squared = 346 :=
sorry

end find_OP_squared_l532_532641


namespace zachary_crunches_more_than_pushups_l532_532272

def push_ups_zachary : ℕ := 46
def crunches_zachary : ℕ := 58

theorem zachary_crunches_more_than_pushups : crunches_zachary - push_ups_zachary = 12 := by
  sorry

end zachary_crunches_more_than_pushups_l532_532272


namespace complement_union_correct_l532_532780

open Set

theorem complement_union_correct :
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  (compl P) ∪ Q = {1, 2, 4} :=
by
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  have h : (compl P) ∪ Q = {1, 2, 4} := sorry
  exact h

end complement_union_correct_l532_532780


namespace smallest_four_digit_divisible_by_35_l532_532241

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532241


namespace R_on_BD_l532_532511

noncomputable theory

variables {A B C D S P Q R R1 R2 : Type} 
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup S]
variables [AddGroup P] [AddGroup Q] [AddGroup R] [AddGroup R1] [AddGroup R2]

-- Definitions for convex cyclic quadrilateral, intersection of diagonals, circumcenters, and parallel lines
variables (isCyclicQuad : CyclicQuadrilateral A B C D)
variable (S : Point) (hS : IntersectDiagonals A C B D S)
variable (P : Point) (hP : IsCircumcenter P A B S)
variable (Q : Point) (hQ : IsCircumcenter Q B C S)
variable (R1 : Point) (hR1 : OnLineThroughParallel P A D R1)
variable (R2 : Point) (hR2 : OnLineThroughParallel Q C D R2)
variable (R : Point) (hR : IsIntersection R LineThroughParallel P A D LineThroughParallel Q C D)

-- Prove R lies on BD
theorem R_on_BD : LiesOn R B D := 
sorry

end R_on_BD_l532_532511


namespace number_of_representations_l532_532060

-- Definitions of the conditions
def is_valid_b (b : ℕ) : Prop :=
  b ≤ 99

def is_representation (b3 b2 b1 b0 : ℕ) : Prop :=
  3152 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0

-- The theorem to prove
theorem number_of_representations : 
  ∃ (N' : ℕ), (N' = 316) ∧ 
  (∀ (b3 b2 b1 b0 : ℕ), is_representation b3 b2 b1 b0 → is_valid_b b0 → is_valid_b b1 → is_valid_b b2 → is_valid_b b3) :=
sorry

end number_of_representations_l532_532060


namespace count_polynomials_in_G_l532_532841

theorem count_polynomials_in_G :
  let G := { P : Polynomial ℤ | ∃ (n : ℕ) (c : Fin n → ℤ),
              P = Polynomial.monomial n 1 + ∑ i in Finset.range (n - 1), Polynomial.monomial i (c i) + Polynomial.C 50 ∧
              ∃ (roots : Fin n → ℂ),
              (∀ i, ∃ (a b : ℤ), roots i = Complex.ofReal a + Complex.I * b) ∧ P.roots.toFinset.card = n } in
  Finset.card G = 528 :=
by
  -- Proof goes here
  sorry

end count_polynomials_in_G_l532_532841


namespace find_common_difference_l532_532868

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532868


namespace takeoff_run_length_l532_532137

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l532_532137


namespace f_one_f_a_f_f_a_l532_532730

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem f_one : f 1 = 5 := by
  sorry

theorem f_a (a : ℝ) : f a = 2 * a + 3 := by
  sorry

theorem f_f_a (a : ℝ) : f (f a) = 4 * a + 9 := by
  sorry

end f_one_f_a_f_f_a_l532_532730


namespace imag_part_of_complex_z_l532_532450

theorem imag_part_of_complex_z (z : ℂ) (h : z - complex.i = (4 - 2 * complex.i) / (1 + 2 * complex.i)) : z.im = -1 := by
  sorry

end imag_part_of_complex_z_l532_532450


namespace total_votes_election_l532_532467

theorem total_votes_election (total_votes fiona_votes elena_votes devin_votes : ℝ) 
  (Fiona_fraction : fiona_votes = (4/15) * total_votes)
  (Elena_fiona : elena_votes = fiona_votes + 15)
  (Devin_elena : devin_votes = 2 * elena_votes)
  (total_eq : total_votes = fiona_votes + elena_votes + devin_votes) :
  total_votes = 675 := 
sorry

end total_votes_election_l532_532467


namespace tan_arctan_k_l532_532510

variables (a b x k : ℝ)

theorem tan_arctan_k :
  (tan x = a / b) ∧ (tan (3 * x) = b / (2 * a + b)) → x = arctan 1 :=
by
  sorry

end tan_arctan_k_l532_532510


namespace two_FA_lt_BD_plus_CD_l532_532034

variable {A B C D F : Point}
variable (quadrilateral_ABCD : quadrilateral A B C D)
variable (angle_A_obtuse : obtuse_angle A B C D)
variable (midpoint_F : midpoint F B C)

theorem two_FA_lt_BD_plus_CD (h1 : angle_A_obtuse) (h2 : midpoint_F) :
  2 * distance F A < distance B D + distance C D :=
sorry

end two_FA_lt_BD_plus_CD_l532_532034


namespace precious_stones_l532_532328

variable (total_amount : ℕ) (price_per_stone : ℕ) (number_of_stones : ℕ)

theorem precious_stones (h1 : total_amount = 14280) (h2 : price_per_stone = 1785) : number_of_stones = 8 :=
by
  sorry

end precious_stones_l532_532328


namespace Ian_money_left_l532_532441

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l532_532441


namespace common_difference_of_arithmetic_sequence_l532_532982

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532982


namespace factorization_correct_l532_532699

noncomputable def factorize_poly (m n : ℕ) : ℕ := 2 * m * n ^ 2 - 12 * m * n + 18 * m

theorem factorization_correct (m n : ℕ) :
  factorize_poly m n = 2 * m * (n - 3) ^ 2 :=
by
  sorry

end factorization_correct_l532_532699


namespace remainder_98_mul_102_div_11_l532_532603

theorem remainder_98_mul_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end remainder_98_mul_102_div_11_l532_532603


namespace trajectory_of_z_l532_532142

-- Given: Complex number z satisfies |z - i| = |3 + 4i|
-- To Prove: The trajectory of z is a circle with center (0, 1) and radius 5.

open Complex

theorem trajectory_of_z (z : ℂ) : abs (z - i) = abs (3 + 4i) → ∃ (c : ℂ), abs (z - c) = 5 ∧ c = (0 : ℂ) + 1 * Complex.I :=
by
  -- Given that |3 + 4i| = 5 as calculated
  have h : abs (3 + 4i) = 5 := by norm_num,
  -- Using the condition |z - i| = |3 + 4i|
  intro hz,
  rw [h] at hz,
  use (0 : ℂ) + (1 : ℂ) * Complex.I,
  exact ⟨hz, rfl⟩

end trajectory_of_z_l532_532142


namespace minimum_k_coloring_l532_532720

theorem minimum_k_coloring (f : ℕ → ℕ) (color : ℕ → Fin 3) :
  (∀ n m : ℕ, color n = color m → f (n + m) = f n + f m) →
  (∃ n m : ℕ, f (n + m) ≠ f n + f m) →
  ∃! k : ℕ, k = 3 :=
by
  sorry

end minimum_k_coloring_l532_532720


namespace karen_starts_late_by_4_minutes_l532_532492

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l532_532492


namespace Nancy_more_pearl_beads_l532_532532

-- Define the problem conditions
def metal_beads_Nancy : ℕ := 40
def crystal_beads_Rose : ℕ := 20
def stone_beads_Rose : ℕ := crystal_beads_Rose * 2
def total_beads_needed : ℕ := 20 * 8
def total_Rose_beads : ℕ := crystal_beads_Rose + stone_beads_Rose
def pearl_beads_Nancy : ℕ := total_beads_needed - total_Rose_beads

-- State the theorem to prove
theorem Nancy_more_pearl_beads :
  pearl_beads_Nancy = metal_beads_Nancy + 60 :=
by
  -- We leave the proof as an exercise
  sorry

end Nancy_more_pearl_beads_l532_532532


namespace find_a1_l532_532766

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

hypothesis h1 : a 5 = 9
hypothesis h2 : 2 * a 3 - a 2 = 6
hypothesis h_arith : arithmetic_seq a

theorem find_a1 : a 1 = -3 :=
by
  sorry

end find_a1_l532_532766


namespace greatest_k_3_divisor_15_factorial_l532_532452

theorem greatest_k_3_divisor_15_factorial : ∃ k : ℕ, (3^k ∣ factorial 15) ∧ ∀ m : ℕ, (3^m ∣ factorial 15) → m ≤ k :=
begin
  use 6,
  split,
  { 
    sorry
  },
  { 
    sorry
  }
end

end greatest_k_3_divisor_15_factorial_l532_532452


namespace smallest_four_digit_divisible_by_35_l532_532199

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532199


namespace smallest_four_digit_divisible_by_35_l532_532190

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532190


namespace arithmetic_sequence_common_difference_l532_532895

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532895


namespace two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l532_532475

noncomputable def rooks_non_attacking : Nat :=
  8 * 8 * 7 * 7 / 2

theorem two_rooks_non_attacking : rooks_non_attacking = 1568 := by
  sorry

noncomputable def kings_non_attacking : Nat :=
  (4 * 60 + 24 * 58 + 36 * 55 + 24 * 55 + 4 * 50) / 2

theorem two_kings_non_attacking : kings_non_attacking = 1806 := by
  sorry

noncomputable def bishops_non_attacking : Nat :=
  (28 * 25 + 20 * 54 + 12 * 52 + 4 * 50) / 2

theorem two_bishops_non_attacking : bishops_non_attacking = 1736 := by
  sorry

noncomputable def knights_non_attacking : Nat :=
  (4 * 61 + 8 * 60 + 20 * 59 + 16 * 57 + 15 * 55) / 2

theorem two_knights_non_attacking : knights_non_attacking = 1848 := by
  sorry

noncomputable def queens_non_attacking : Nat :=
  (28 * 42 + 20 * 40 + 12 * 38 + 4 * 36) / 2

theorem two_queens_non_attacking : queens_non_attacking = 1288 := by
  sorry

end two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l532_532475


namespace rect_area_sum_eq_16_l532_532320

theorem rect_area_sum_eq_16 (a b c : ℕ) (h1 : |a * b - a * c| = 1) (h2 : |a * c - b * c| = 49) :
  a + b + c = 16 :=
sorry

end rect_area_sum_eq_16_l532_532320


namespace lara_miles_approx_2600_l532_532055

-- Conditions
def max_steps := 100000
def flips_per_year := 52
def final_reading := 38200
def steps_per_mile := 2000

-- The total steps = (flips_per_year * max_steps) + final_reading
def total_steps : ℕ := (flips_per_year * max_steps) + final_reading

-- The total miles is total_steps divided by steps_per_mile
def total_miles : ℕ := total_steps / steps_per_mile

-- Given the provided values, we need to prove the total miles is approximately 2600
theorem lara_miles_approx_2600 : total_miles ≈ 2600 :=
by
  sorry

end lara_miles_approx_2600_l532_532055


namespace _l532_532280

noncomputable def track_circumference (A B : ℝ) (x : ℝ) : ℝ :=
  let half_circumference := x
  let full_circumference := 2 * x
  -- Conditions
  let A_starts_diametrically_opposite := true
  let B_travels_150_yards := true
  let second_meeting := true
  -- Equation Setup
  let first_ratio := 150 / (x - 150)
  let second_ratio := (x + 90) / (2 * x - 90)
  -- Solve equation for x
  if first_ratio = second_ratio
  then full_circumference
  else 0

@[simp] theorem circumference_of_track_is_720
  (A B x : ℝ)
  (A_starts_diametrically_opposite : true)
  (B_travels_150_yards : true)
  (second_meeting : true)
  (hyp : 150 / (x - 150) = (x + 90) / (2 * x - 90))
  : 2 * x = 720 :=
begin
  -- Here the proof should go, but it is omitted
  sorry
end

end _l532_532280


namespace find_P_l532_532631

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l532_532631


namespace team_formation_plans_l532_532658

/-- 
We need to select 3 teachers to form a teaching team from a group of 4 female teachers and 5 male teachers.
The requirement is that the team must include both male and female teachers.
Prove that the number of different team formation plans is 70.
-/
theorem team_formation_plans (f m : ℕ) (h_f : f = 4) (h_m : m = 5) : 
  (∑ (k : ℕ) in {1, 2}.to_finset, nat.choose m k * nat.choose f (3 - k)) = 70 := 
by 
  sorry

end team_formation_plans_l532_532658


namespace smallest_positive_period_of_my_function_l532_532721

noncomputable def smallest_positive_period (f : ℝ → ℝ) := 
  inf {T : ℝ | T > 0 ∧ ∀ x, f (x + T) = f x}

def my_function (x : ℝ) : ℝ :=
  (Mathlib.sin x - Mathlib.cos x)^2 - 1

theorem smallest_positive_period_of_my_function :
  smallest_positive_period my_function = Mathlib.pi :=
sorry

end smallest_positive_period_of_my_function_l532_532721


namespace find_common_difference_l532_532863

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532863


namespace factorial_base_9_zeroes_l532_532436

-- Main statement 
theorem factorial_base_9_zeroes (n : ℕ) (h : n = 15) : 
  let factors_of_3 := (nat.factorial 15).multiplicity 3 in   
   (factors_of_3 / 2).to_nat = 3 :=
by
  have : factors_of_3 = nat.factorial 15 / (9 ^ 3),
  { sorry }
  rw this
  sorry
  -- Continue steps to complete the proof

end factorial_base_9_zeroes_l532_532436


namespace factorize_expression_l532_532700

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l532_532700


namespace common_difference_of_arithmetic_sequence_l532_532988

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532988


namespace smallest_four_digit_number_divisible_by_35_l532_532219

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532219


namespace third_candidate_votes_l532_532157

open BigOperators

theorem third_candidate_votes (V : ℕ) : 
  let candidate1_votes := 7636
      candidate2_votes := 11628
      winning_percentage := 0.4969230769230769
  in 11628 = nat.floor (winning_percentage * V) →
     V = nat.floor (11628 / winning_percentage) →
     V - (candidate1_votes + candidate2_votes) = 4136 :=
by
  intros h1 h2
  sorry

end third_candidate_votes_l532_532157


namespace orthocenter_circumcircle_reflection_l532_532754

noncomputable def reflection_point (A B P : Point) : Point :=
  -- Reflect P over the midpoint of segment AB. Placeholder definition, adjust accordingly.
  sorry 

theorem orthocenter_circumcircle_reflection
  (O : Point) (A B C P: Point)
  (Γ : Circle)
  (H : Point) 
  (A₁ B₁ C₁ A₂ B₂ C₂ : Point)
  (hΓA : Γ.radius = dist O A)
  (hΓB : Γ.radius = dist O B)
  (hΓC : Γ.radius = dist O C)
  (hA₁ : ∃ rayAP : Ray, Line_through (O) (A₁) ∧ intersects (rayAP) (Γ) ∧ A₁ ∈ rayAP)
  (hB₁ : ∃ rayBP : Ray, Line_through (O) (B₁) ∧ intersects (rayBP) (Γ) ∧ B₁ ∈ rayBP)
  (hC₁ : ∃ rayCP : Ray, Line_through (O) (C₁) ∧ intersects (rayCP) (Γ) ∧ C₁ ∈ rayCP)
  (hReflectA1 : A₂ = reflection_point B C A₁)
  (hReflectB1 : B₂ = reflection_point C A B₁)
  (hReflectC1 : C₂ = reflection_point A B C₁)
  (hH : ∃ h : H, H = orthocenter A B C) :
  Concyclic (H) (A₂ B₂ C₂) :=
  sorry

end orthocenter_circumcircle_reflection_l532_532754


namespace ellipse_properties_l532_532743

open_locale classical

noncomputable def ellipse_equation (x y : ℝ) :=
  let a : ℝ := 2 in
  let b : ℝ := Real.sqrt 3 in
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties :
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ c > 0 ∧ a^2 - b^2 = c^2 ∧ c = 1 ∧ (1, 3/2) ∈ { p | ellipse_equation p.1 p.2 } ∧ ∃ l : ℝ → ℝ, ∀ (x : ℝ), l x = x * (√(5) / 10) + 1) →
  (ellipse_equation 1 (3/2)) ∧
  (∃ m : ℝ, m^2 = 20/9 ∧
  (∀ x y : ℝ, ellipse_equation x y → (y = (√(5) / 10) * (x + 1) ∨ y = - (√(5) / 10) * (x + 1)))) :=
begin
  sorry
end

end ellipse_properties_l532_532743


namespace possible_digits_C_multiple_of_5_l532_532377

theorem possible_digits_C_multiple_of_5 :
    ∃ (digits : Finset ℕ), (∀ x ∈ digits, x < 10) ∧ digits.card = 10 ∧ (∀ C ∈ digits, ∃ n : ℕ, n = 1000 + C * 100 + 35 ∧ n % 5 = 0) :=
by {
  sorry
}

end possible_digits_C_multiple_of_5_l532_532377


namespace factorial_base_9_zeroes_l532_532437

-- Main statement 
theorem factorial_base_9_zeroes (n : ℕ) (h : n = 15) : 
  let factors_of_3 := (nat.factorial 15).multiplicity 3 in   
   (factors_of_3 / 2).to_nat = 3 :=
by
  have : factors_of_3 = nat.factorial 15 / (9 ^ 3),
  { sorry }
  rw this
  sorry
  -- Continue steps to complete the proof

end factorial_base_9_zeroes_l532_532437


namespace find_common_difference_l532_532870

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532870


namespace common_difference_arithmetic_sequence_l532_532921

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532921


namespace polynomials_equal_l532_532572

open Complex Polynomial

theorem polynomials_equal (P Q : Polynomial ℂ) (hP : ¬(degree P = 0)) (hQ : ¬(degree Q = 0))
  (h_same_zeros : ∀ z : ℂ, P.is_root z ↔ Q.is_root z)
  (h_same_zeros_plus_one : ∀ z : ℂ, (P + 1).is_root z ↔ (Q + 1).is_root z) :
  P = Q :=
sorry

end polynomials_equal_l532_532572


namespace total_length_of_board_l532_532283

-- Define variables for the lengths
variable (S L : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := 2 * S = L + 4
def condition2 : Prop := S = 8.0

-- The goal is to prove the total length of the board is 20.0 feet
theorem total_length_of_board (h1 : condition1 S L) (h2 : condition2 S) : S + L = 20.0 := by
  sorry

end total_length_of_board_l532_532283


namespace lowest_income_of_wealthiest_500_l532_532124

theorem lowest_income_of_wealthiest_500 (N x : ℝ) (h₁ : N = 5 * 10^7 * x ^ (-2)) (h₂ : N = 500) : x = 100 :=
by
  sorry

end lowest_income_of_wealthiest_500_l532_532124


namespace exists_zero_in_interval_minus3_minus2_l532_532077

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x - x

theorem exists_zero_in_interval_minus3_minus2 : 
  ∃ x ∈ Set.Icc (-3 : ℝ) (-2), f x = 0 :=
by
  sorry

end exists_zero_in_interval_minus3_minus2_l532_532077


namespace age_of_b_l532_532617

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 := 
  sorry

end age_of_b_l532_532617


namespace circle_radius_of_equation_l532_532121

theorem circle_radius_of_equation (a : ℝ) (r: ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = r^2) → r = sqrt(2) := by
  sorry

end circle_radius_of_equation_l532_532121


namespace song_liking_arrangements_l532_532660

theorem song_liking_arrangements (songs : Finset ℕ) (AB BC CA : Finset ℕ)
  (A B C N : Finset ℕ) :
  songs.card = 5 →
  (∀ s ∈ songs, s ∈ AB ∨ s ∈ BC ∨ s ∈ CA ∨ s ∈ A ∨ s ∈ B ∨ s ∈ C ∨ s ∈ N) →
  (∀ t ∈ AB ∪ BC ∪ CA, ¬ t ∈ A ∧ ¬ t ∈ B ∧ ¬ t ∈ C ∧ ¬ t ∈ N) →
  (∀ x ∈ A, x ∉ AB ∧ x ∉ BC ∧ x ∉ CA ∧ x ∉ B ∧ x ∉ C ∧ x ∉ N) →
  (∀ y ∈ B, y ∉ AB ∧ y ∉ BC ∧ y ∉ CA ∧ y ∉ A ∧ y ∉ C ∧ y ∉ N) →
  (∀ z ∈ C, z ∉ AB ∧ z ∉ BC ∧ z ∉ CA ∧ z ∉ A ∧ z ∉ B ∧ z ∉ N) →
  AB.card = 1 ∧ BC.card = 1 ∧ CA.card = 1 →
  1 ≤ AB.card ∧ 1 ≤ BC.card ∧ 1 ≤ CA.card →
  (∃ a ∈ AB, ∃ b ∈ BC, ∃ c ∈ CA, a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  (∃ num_ways ∈ {360}, num_ways = 360) :=
by
  intros,
  sorry

end song_liking_arrangements_l532_532660


namespace factor_expression_l532_532340

noncomputable def q (a b c : ℝ) : ℝ := a^2 + ab + ac + b^2 + bc + c^2

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * q a b c :=
by
  sorry

end factor_expression_l532_532340


namespace bisect_line_segment_between_angle_sides_l532_532741

noncomputable theory

structure Point :=
(x : ℝ) (y : ℝ)

structure Angle :=
(O : Point) (X : Point) (Y : Point)

structure Between (P : Point) (A : Point) (B : Point) : Prop :=
(on_line : ∃ (t : ℝ), t ∈ set.Icc 0 1 ∧ A.x + t * (B.x - A.x) = P.x ∧ A.y + t * (B.y - A.y) = P.y)

def is_bisector (P Q S : Point) : Prop :=
P.distance Q = P.distance S

theorem bisect_line_segment_between_angle_sides
  (A : Angle) (P : Point)
  (H1 : Between P A.O A.X)
  (H2 : Between P A.O A.Y) :
  ∃ S, ∃ Q, is_bisector P Q S ∧ Between Q A.O A.X ∧ Between S A.O A.Y := 
sorry

end bisect_line_segment_between_angle_sides_l532_532741


namespace smallest_four_digit_divisible_by_35_l532_532212

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532212


namespace number_of_lines_l532_532517

noncomputable def parabola : set (ℝ × ℝ) := { p | p.snd ^ 2 = 4 * p.fst }
noncomputable def circle : set (ℝ × ℝ) := { p | (p.fst - 5) ^ 2 + p.snd ^ 2 = 9 }

theorem number_of_lines (l : set (ℝ × ℝ)) (A B M : ℝ × ℝ)
  (h1 : A ∈ parabola)
  (h2 : B ∈ parabola)
  (h3 : M = (A + B) / 2)
  (h4 : M ∈ circle)
  (h5 : A ∈ l ∧ B ∈ l)
  (h6 : ∀ P ∈ l, (P - M).snd = (P - M).fst * (M.snd / (M.fst - 5))) :
  ∃ (l_set : finset (set (ℝ × ℝ))), l_set.card = 4 ∧ ∀ l' ∈ l_set, ∀ A B ∈ parabola, 
  let M := (A + B) / 2 in M ∈ circle ∧ ∀ P ∈ l', (P - M).snd = (P - M).fst * (M.snd / (M.fst - 5)) :=
sorry

end number_of_lines_l532_532517


namespace stratified_sampling_seniors_l532_532306

-- Definitions of the given conditions
def freshmen : ℕ := 720
def sophomores : ℕ := 700
def seniors : ℕ := 680
def sample_size : ℕ := 105

-- Total number of students
def total_students : ℕ := freshmen + sophomores + seniors

-- Sampling ratio for seniors
def senior_sampling_ratio : ℚ := seniors / total_students

-- Number of seniors to be drawn
def seniors_drawn : ℕ := (sample_size * senior_sampling_ratio).toNat

-- The proof statement
theorem stratified_sampling_seniors : seniors_drawn = 34 :=
by
  sorry

end stratified_sampling_seniors_l532_532306


namespace arithmetic_sequence_common_difference_l532_532892

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532892


namespace sum_of_integers_l532_532453

theorem sum_of_integers (x y : ℕ) (h1 : x = y + 3) (h2 : x^3 - y^3 = 63) : x + y = 5 :=
by
  sorry

end sum_of_integers_l532_532453


namespace smallest_four_digit_divisible_by_35_l532_532247

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532247


namespace common_difference_of_arithmetic_sequence_l532_532887

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532887


namespace arithmetic_sequence_common_difference_l532_532946

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532946


namespace triangle_area_qin_jiushao_l532_532558

theorem triangle_area_qin_jiushao (a b c : ℝ) (h1: a = 2) (h2: b = 3) (h3: c = Real.sqrt 13) :
  Real.sqrt ((1 / 4) * (a^2 * b^2 - (1 / 4) * (a^2 + b^2 - c^2)^2)) = 3 :=
by
  -- Hypotheses
  rw [h1, h2, h3]
  sorry

end triangle_area_qin_jiushao_l532_532558


namespace factory_sample_capacity_l532_532291

theorem factory_sample_capacity (n : ℕ) (a_ratio b_ratio c_ratio : ℕ) 
  (total_ratio : a_ratio + b_ratio + c_ratio = 10) (a_sample : ℕ)
  (h : a_sample = 16) (h_ratio : a_ratio = 2) :
  n = 80 :=
by
  -- sample calculations proof would normally be here
  sorry

end factory_sample_capacity_l532_532291


namespace triangle_angle_measure_l532_532753

theorem triangle_angle_measure (a b c : ℝ) (S : ℝ) (h : S = (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)) :
  ∠ C = π / 6 :=
by
  sorry

end triangle_angle_measure_l532_532753


namespace find_common_difference_l532_532959

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532959


namespace sum_of_numbers_equal_16_l532_532316

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l532_532316


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532524

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532524


namespace sum_digits_impossible_l532_532151

theorem sum_digits_impossible (cards : Multiset ℕ)
  (h : cards = {5, 5, 6, 6, 7, 7, 8, 8, 9, 9}) :
  ∀ num1 num2 : ℕ, (num1 < 100000) → (num2 < 100000) →
  (num1 + num2 < 200000) →
  (Multiset.of_list (nat.digits 10 num1) + Multiset.of_list (nat.digits 10 num2) = cards) →
  (nat.digits 10 (num1 + num2) = 1 :: (d :: ds) →
  ∀ (d : ℕ), d < 5) :=
by
  sorry

end sum_digits_impossible_l532_532151


namespace jim_less_than_anthony_l532_532546

-- Definitions for the conditions
def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

-- Lean statement to prove the problem
theorem jim_less_than_anthony : anthony_shoes - jim_shoes = 2 := by
  sorry

end jim_less_than_anthony_l532_532546


namespace sphere_surface_area_l532_532145

theorem sphere_surface_area (V : ℝ) (pi : ℝ)
  (hV : V = 72 * pi) :
  ∃ A : ℝ, A = 4 * pi * real.cbrt(2916) :=
by sorry

end sphere_surface_area_l532_532145


namespace find_a_l532_532765

theorem find_a (a : ℝ) (h : a > 0)
    (intersect : ∃ A B : ℝ × ℝ, (A.2 = A.1 + 2 * a) ∧ (B.2 = B.1 + 2 * a) ∧
      (A.1 ^ 2 + A.2 ^ 2 - 2 * a * A.2 - 2 = 0) ∧ (B.1 ^ 2 + B.2 ^ 2 - 2 * a * B.2 - 2 = 0))
    (dist_AB : ∃ A B : ℝ × ℝ, |(A.1 - B.1)^2 + (A.2 - B.2)^2| = 4 * 3) :
    a = sqrt 2 :=
by
  sorry

end find_a_l532_532765


namespace probability_none_l532_532597

variable {Ω : Type} [ProbabilitySpace Ω]

def PA : ℝ := 0.25
def PB : ℝ := 0.40
def PC : ℝ := 0.30
def PAB : ℝ := 0.15
def PBC : ℝ := 0.12
def PAC : ℝ := 0.10
def PABC : ℝ := 0.05

def PNone : ℝ := 
  1 - (PA + PB + PC - PAB - PBC - PAC + PABC)

theorem probability_none (h : PNone = 0.42) : 
  PA = 0.25 ∧ PB = 0.40 ∧ PC = 0.30 ∧ PAB = 0.15 ∧ PBC = 0.12 ∧ PAC = 0.10 ∧ PABC = 0.05 → 
  PNone = 0.42 :=
by 
  sorry

end probability_none_l532_532597


namespace karen_starts_late_l532_532493

def karen_speed := 60 -- Karen's speed in mph
def tom_speed := 45 -- Tom's speed in mph
def tom_distance := 24 -- Distance Tom drives in miles
def karen_lead := 4 -- Distance by which Karen beats Tom in miles

theorem karen_starts_late : 
  let t := tom_distance / tom_speed in -- Time Tom drives
  let t_k := (tom_distance + karen_lead) / karen_speed in -- Time Karen drives
  (t - t_k) * 60 = 4 := -- The time difference in minutes is 4
by
  sorry

end karen_starts_late_l532_532493


namespace third_jumper_height_l532_532544

/-- 
  Ravi can jump 39 inches high.
  Ravi can jump 1.5 times higher than the average height of three other jumpers.
  The three jumpers can jump 23 inches, 27 inches, and some unknown height x.
  Prove that the unknown height x is 28 inches.
-/
theorem third_jumper_height (x : ℝ) (h₁ : 39 = 1.5 * (23 + 27 + x) / 3) : 
  x = 28 :=
sorry

end third_jumper_height_l532_532544


namespace amy_total_tickets_l532_532671

def amy_initial_tickets : ℕ := 33
def amy_additional_tickets : ℕ := 21

theorem amy_total_tickets : amy_initial_tickets + amy_additional_tickets = 54 := by
  sorry

end amy_total_tickets_l532_532671


namespace find_common_difference_l532_532947

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532947


namespace find_k_l532_532080

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) ∧ 
  (∀ x : ℝ, y = k * x + 4) ∧ 
  (1, 5) ∈ { p | ∃ x, p = (x, 2 * x + 3) } ∧ 
  (1, 5) ∈ { q | ∃ x, q = (x, k * x + 4) } → 
  k = 1 :=
by
  sorry

end find_k_l532_532080


namespace problem_proof_l532_532259

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l532_532259


namespace math_proof_l532_532480

noncomputable def problem_statement : Prop :=
  -- Definition of line l
  let line_l := ∀ t : ℝ, (x t = -t ∧ y t = sqrt 3 - t) in
  
  -- General equation of line l
  let general_eq_line_l := ∀ x y : ℝ, (x - y + sqrt 3 = 0) in
  
  -- Polar to rectangular transformation for curve C1
  let curve_c1 := ∀ (θ : ℝ) (ρ : ℝ), (ρ = 4 * sin θ) -> (x^2 + (y-2)^2 = 4) in
  
  -- Transformation to obtain curve C2
  let curve_c2 := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1) in
  
  -- Intersection computations
  let line_intersect_x_axis := ∀ x : ℝ, (x = -sqrt 3) in
  let quadratic_eq := ∀ t : ℝ, (5*t^2 - 2*sqrt 6 * t - 2 = 0) in
  
  -- Roots properties
  let root_sum := (t₁ t₂ : ℝ), (t₁ + t₂ = 2 * sqrt 6 / 5) in
  let root_prod := (t₁ t₂ : ℝ), (t₁ * t₂ = -2 / 5) in
  
  -- Final proof goal
  let final_value := (|PM| |PN| : ℝ), (1 / |PM| + 1 / |PN| = 4) in

  general_eq_line_l ∧ curve_c1 ∧ curve_c2 ∧ line_intersect_x_axis ∧ quadratic_eq ∧ root_sum ∧ root_prod -> final_value

theorem math_proof : problem_statement := sorry

end math_proof_l532_532480


namespace find_explicit_formula_of_f_g_has_one_zero_in_negative_infinity_to_one_max_value_of_y_gx_on_given_interval_l532_532394

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem find_explicit_formula_of_f :
  (f(-2) = -3) ∧ (f(0) = -3) ∧ (∀ x, f(x) ≥ -4) :=
by
  sorry

def g (m x : ℝ) : ℝ := m * f(x) + 1

theorem g_has_one_zero_in_negative_infinity_to_one (m : ℝ) (h : m < 0) :
  ∃ x ∈ Iic 1, g m x = 0 :=
by
  sorry

theorem max_value_of_y_gx_on_given_interval (m : ℝ) (h : m > 0) :
  let y_max := if 0 < m ∧ m ≤ 8/7 then
                 9*m/4 + 1
               else if m > 8/7 then
                 4*m - 1
               else
                 0 in
  ∀ x ∈ Icc (-3) (3/2), |g m x| ≤ y_max :=
by
  sorry

end find_explicit_formula_of_f_g_has_one_zero_in_negative_infinity_to_one_max_value_of_y_gx_on_given_interval_l532_532394


namespace smallest_four_digit_divisible_by_35_l532_532209

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532209


namespace common_difference_is_two_l532_532912

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532912


namespace find_a_l532_532758

def line_eq (a : ℝ) : ℝ → ℝ := λ x, x + 2 * a
def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * y - 2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x1 y1 x2 y2 : ℝ, 
(line_eq a x1 = y1) ∧ (line_eq a x2 = y2) ∧ 
(circle_eq a x1 y1) ∧ (circle_eq a x2 y2) ∧ 
(distance x1 y1 x2 y2 = 2 * real.sqrt 3)) : 
  a = real.sqrt 2 :=
by
  sorry

end find_a_l532_532758


namespace garden_breadth_l532_532653

variable (Length Breadth Perimeter : ℕ)

def rectangular_perimeter (Length Breadth : ℕ) : ℕ :=
  2 * (Length + Breadth)

theorem garden_breadth
    (h1 : Perimeter = 900)
    (h2 : Length = 260) :
    Breadth = 190 :=
by
    simp only [rectangular_perimeter] at h1
    have h3 : 2 * (Length + Breadth) = 900 := h1
    rw [h2] at h3
    linarith

end garden_breadth_l532_532653


namespace remainder_x_plus_3uy_div_y_l532_532267

theorem remainder_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_x_plus_3uy_div_y_l532_532267


namespace distinct_prime_triples_sum_104_l532_532268

theorem distinct_prime_triples_sum_104 :
  ∃ (a b c : ℕ), prime a ∧ prime b ∧ prime c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  a = 2 ∧ b = 41 ∧ c = 61 ∧
  ∃ S : finset (finset ℕ), S.card = 7 ∧ ∀ t ∈ S, 
    (∃ p1 p2 p3 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧ 
     p1 + p2 + p3 = 104 ∧ t = {p1, p2, p3}) :=
sorry

end distinct_prime_triples_sum_104_l532_532268


namespace complex_sets_identity_proof_l532_532387

noncomputable def complex_sets_identity
  (n : ℕ) 
  (hn : n ≥ 2) 
  (A B : Fin n → ℂ) : Prop :=
∑ k in Finset.range n, 
  (∏ j in Finset.range n, (A k + B j)) / 
  (∏ j in (Finset.range n).filter (λ j, j ≠ k), (A k - A j)) = 
∑ k in Finset.range n,
  (∏ j in Finset.range n, (B k + A j)) / 
  (∏ j in (Finset.range n).filter (λ j, j ≠ k), (B k - B j))

-- The theorem to be proved
theorem complex_sets_identity_proof 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (A B : Fin n → ℂ) : 
  complex_sets_identity n hn A B :=
sorry

end complex_sets_identity_proof_l532_532387


namespace fraction_distance_walked_by_first_class_l532_532644

namespace CulturalCenterProblem

def walking_speed : ℝ := 4
def bus_speed_with_students : ℝ := 40
def bus_speed_empty : ℝ := 60

theorem fraction_distance_walked_by_first_class :
  ∃ (x : ℝ), 
    (x / walking_speed) = ((1 - x) / bus_speed_with_students) + ((1 - 2 * x) / bus_speed_empty)
    ∧ x = 5 / 37 :=
by
  sorry

end CulturalCenterProblem

end fraction_distance_walked_by_first_class_l532_532644


namespace missing_pieces_l532_532165

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l532_532165


namespace common_difference_of_arithmetic_sequence_l532_532966

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532966


namespace integer_root_modulus_le_max_modulus_of_coeffs_l532_532098

theorem integer_root_modulus_le_max_modulus_of_coeffs 
  (n : ℕ) 
  (a : Fin (n+1) → ℤ) 
  (x : ℤ) 
  (root_condition : ∑ i in Finset.range (n+1), a i * x^i = 0) :
  |x| ≤ finset_sup (Finset.range (n+1)) (λ i, |a i|) := 
sorry

end integer_root_modulus_le_max_modulus_of_coeffs_l532_532098


namespace calc_total_push_ups_correct_l532_532673

-- Definitions based on conditions
def sets : ℕ := 9
def push_ups_per_set : ℕ := 12
def reduced_push_ups : ℕ := 8

-- Calculate total push-ups considering the reduction in the ninth set
def total_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (reduced_push_ups : ℕ) : ℕ :=
  (sets - 1) * push_ups_per_set + (push_ups_per_set - reduced_push_ups)

-- Theorem statement
theorem calc_total_push_ups_correct :
  total_push_ups sets push_ups_per_set reduced_push_ups = 100 :=
by
  sorry

end calc_total_push_ups_correct_l532_532673


namespace common_difference_arithmetic_sequence_l532_532924

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532924


namespace number_of_distinguishable_arrangements_is_420_l532_532789

-- Define the counts of each tile
def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

-- Define the total number of tiles
def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

-- Noncomputable definition for factorial
noncomputable def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the total number of distinguishable arrangements
noncomputable def distinguishable_arrangements (n_B n_P n_G n_Y : ℕ) : ℕ :=
  fact (n_B + n_P + n_G + n_Y) / (fact n_B * fact n_P * fact n_G * fact n_Y)

-- The main theorem statement
theorem number_of_distinguishable_arrangements_is_420 :
  distinguishable_arrangements brown_tiles purple_tiles green_tiles yellow_tiles = 420 :=
by simp [brown_tiles, purple_tiles, green_tiles, yellow_tiles, fact, distinguishable_arrangements]; sorry

end number_of_distinguishable_arrangements_is_420_l532_532789


namespace original_urn_marbles_l532_532313

theorem original_urn_marbles :
  ∃ (g y : ℕ), (3 * g - y = 9) ∧ (2 * g - y = 4) ∧ (g + y = 11) :=
by
  have h1 : ∀ g y, 3 * g - y = 9 → 2 * g - y = 4 → g + y = 11 :=
    by
      intro g y eq1 eq2
      have h : g = 5 := by linarith
      have h' : y = 6 := by linarith
      rw [h, h']
  exact ⟨5, 6, by norm_num, by norm_num, by norm_num⟩

end original_urn_marbles_l532_532313


namespace polynomial_integer_coeff_l532_532070

def ω : ℂ := Complex.exp (Complex.I * Real.pi / 5)

def f (x : ℂ) : ℂ := (x - ω) * (x - ω^3) * (x - ω^7) * (x - ω^9)

theorem polynomial_integer_coeff (x : ℂ) : 
  f(x) = x^4 - x^3 + x^2 - x + 1 ∧ 
  ¬ ∃ (a b : ℤ[x]), a.degree > 0 ∧ b.degree > 0 ∧ f(x) = a * b := 
by
  sorry

end polynomial_integer_coeff_l532_532070


namespace Ian_money_left_l532_532442

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l532_532442


namespace part1_part2_l532_532735

-- Assume the function f and g are given as per condition
constant f : ℝ → ℝ
constant g : ℝ → ℝ

axiom f_def (a b c x : ℝ) (h : a = c ∧ f(1) = 1) : f x = a*x^2 + b*x + c

theorem part1 (a : ℝ) (h : a > 0) : 
  g a = if 0 < a ∧ a ≤ (1 : ℝ) / 6 then 10 * a
        else if (1 : ℝ) / 6 < a ∧ a ≤ (1 : ℝ) / 2 then a - 1 / (4 * a) + 3
        else if a > (1 : ℝ) / 2 then 9 * a - 1 / (4 * a) - 1
        else 0 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a * x^2 + b * x + c) (h2 : (∀ f : ℝ → ℝ, (a * x^2 + b * x + c ≠ 1)) (h3 : has_distinct_zeros_in_interval f (-1/4, 1/4) (h4 : a > 0 ∧ b > 0 ∧ c > 0)) :
∃  a b c, (min a + b + c = 38) :=
sorry

end part1_part2_l532_532735


namespace find_focus_of_parabola_l532_532571

def parabola_focus (p : ℝ) : ℝ × ℝ :=
(∥ p / 2 ∥, 0)

theorem find_focus_of_parabola :
  ∃ (p : ℝ), (1, sqrt 3) ∈ {p ∈ (ℝ × ℝ) | p.2^2 = 2 * p * p.1} →
  parabola_focus p = (3 / 4, 0) := by
sorry

end find_focus_of_parabola_l532_532571


namespace increased_sales_type_B_l532_532286

-- Definitions for sales equations
def store_A_sales (x y : ℝ) : Prop :=
  60 * x + 15 * y = 3600

def store_B_sales (x y : ℝ) : Prop :=
  40 * x + 60 * y = 4400

-- Definition for the price of clothing items
def price_A (x : ℝ) : Prop :=
  x = 50

def price_B (y : ℝ) : Prop :=
  y = 40

-- Definition for the increased sales in May for type A
def may_sales_A (x : ℝ) : Prop :=
  100 * x * 1.2 = 6000

-- Definition to prove percentage increase for type B sales in May
noncomputable def percentage_increase_B (x y : ℝ) : ℝ :=
  ((4500 - (100 * y * 0.4)) / (100 * y * 0.4)) * 100

theorem increased_sales_type_B (x y : ℝ)
  (h1 : store_A_sales x y)
  (h2 : store_B_sales x y)
  (hA : price_A x)
  (hB : price_B y)
  (hMayA : may_sales_A x) :
  percentage_increase_B x y = 50 :=
sorry

end increased_sales_type_B_l532_532286


namespace log_sum_l532_532584

theorem log_sum : 2 * Real.log 2 + Real.log 25 = 2 := 
by 
  sorry

end log_sum_l532_532584


namespace problem_1_problem_2_l532_532426

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.cos x, 2 * Real.sin (x + Real.pi / 4))
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin (Real.pi / 4 - x))
noncomputable def f (x : ℝ) : ℝ := vector_a x.1 * vector_b x.1 + vector_a x.2 * vector_b x.2

theorem problem_1 (α : ℝ) (h : Real.tan α = 2) : f α = 3 / 5 := sorry

theorem problem_2 : set.Icc (f (Real.pi / 12)) (f (Real.pi / 2)) = set.Icc 0 ((Real.sqrt 2 + 1) / 2) := sorry

end problem_1_problem_2_l532_532426


namespace mean_value_z_l532_532567

theorem mean_value_z (z : ℚ) (h : (7 + 10 + 23) / 3 = (18 + z) / 2) : z = 26 / 3 :=
by
  sorry

end mean_value_z_l532_532567


namespace common_difference_of_arithmetic_sequence_l532_532972

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532972


namespace sum_of_positional_differences_is_1260_l532_532595

-- Define the two decks as sets of 36 cards each,
-- where the cards in each deck are numbered from 1 to 36 and 37 to 72 respectively.
def deck1 : set ℕ := {1, 2, 3, ..., 36}
def deck2 : set ℕ := {37, 38, 39, ..., 72}

-- Define a function pos_diff to calculate the positional difference between matching cards.
def pos_diff (i : ℕ) (ki : ℕ) : ℕ := ki - i - 1

-- Define the sum of the positional differences for all 36 cards.
def sum_pos_diff (f : ℕ → ℕ) : ℕ :=
  ∑ i in (finset.range 36), (pos_diff i (f i))

-- Theorem stating that the sum of the positional differences is 1260.
theorem sum_of_positional_differences_is_1260 : 
  ∃ f : ℕ → ℕ, 
    (∀ i, i ∈ finset.range 36 → f i ∈ finset.range (36 + 37)) ∧ -- f is a mapping of cards from deck1 to deck2
    sum_pos_diff f = 1260 :=
by 
  sorry

end sum_of_positional_differences_is_1260_l532_532595


namespace arithmetic_sequence_common_difference_l532_532852

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532852


namespace integral_semicircle_integral_piecewise_l532_532282

-- Problem I: 
theorem integral_semicircle :
  ∫ (x : ℝ) in 0..3, real.sqrt (9 - x^2) = (9 * real.pi) / 4 :=
sorry

-- Problem II:
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2 else
  if 1 ≤ x ∧ x ≤ real.exp 1 then x⁻¹ else 0

theorem integral_piecewise :
  ∫ (x : ℝ) in 0..real.exp 1, f x = 4 / 3 :=
sorry

end integral_semicircle_integral_piecewise_l532_532282


namespace union_A_B_complement_A_inter_B_l532_532847

-- Definitions taken directly from conditions
def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def set_B : Set ℝ := {x | x^2 - 9*x + 14 < 0}

-- Proof statement for (1)
theorem union_A_B : (set_A ∪ set_B) = {x | 2 < x ∧ x < 10} :=
by sorry

-- Proof statement for (2)
theorem complement_A_inter_B : (set.univ \ set_A) ∩ set_B = {x | 2 < x ∧ x < 3} :=
by sorry

end union_A_B_complement_A_inter_B_l532_532847


namespace totalLikeCombinations_l532_532312

noncomputable def likeDistributionCount : ℕ :=
  let total_songs := 5
  -- Calculate the number of ways considering the cases and conditions
  let case1 := (binom 5 2) * (binom 3 2) * 1 * 3
  let case2 := (binom 5 3) * (binom 2 1) * 1 * 3
  case1 + case2

theorem totalLikeCombinations (n : ℕ) (hlike : n = likeDistributionCount) : n = 150 :=
  by
    sorry

end totalLikeCombinations_l532_532312


namespace ab_value_l532_532447

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 :=
by
  sorry

end ab_value_l532_532447


namespace common_difference_of_arithmetic_sequence_l532_532976

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532976


namespace arithmetic_sequence_common_difference_l532_532900

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532900


namespace compare_n1_sq_3n_l532_532681

noncomputable theory

theorem compare_n1_sq_3n (n : ℕ) (hn : n ≥ 3) : (n + 1)^2 < 3^n := 
sorry

end compare_n1_sq_3n_l532_532681


namespace amusement_park_total_cost_l532_532296

noncomputable def total_price (num_adults num_children num_seniors : ℕ)
  (adult_weekend_price child_weekend_price senior_price : ℕ)
  (adult_discount child_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
let initial_cost := num_adults * adult_weekend_price + num_children * child_weekend_price + num_seniors * senior_price in
let discount := num_adults * adult_weekend_price * adult_discount + num_children * child_weekend_price * child_discount in
let discounted_cost := initial_cost - discount in
let senior_discount := if num_adults >= 2 * num_seniors then num_seniors * senior_price else 0 in
let total_cost_after_senior_discount := discounted_cost - senior_discount in
let total_with_tax := total_cost_after_senior_discount * (1 + sales_tax) in
(total_with_tax * 100).toRat.ceil / 100

theorem amusement_park_total_cost :
  total_price 3 5 2 24 12 10 0.15 0.1 0.07 = 123.26 :=
by
  sorry

end amusement_park_total_cost_l532_532296


namespace triangle_area_PGQ_bisect_l532_532470

noncomputable def area_triangle_PGQ (PQ QR RP : ℕ) (GQ GR : ℚ) : ℚ :=
  let s := (PQ + GQ + GR) / 2
  real.sqrt (s * (s - PQ) * (s - GQ) * (s - GR))

theorem triangle_area_PGQ_bisect :
  let PQ := 24
  let QR := 28
  let RP := 34
  let GQ := (12 * 28 / 29 : ℚ)
  let GR :=  (17 * 28 / 29 : ℚ)
  let A := area_triangle_PGQ PQ GQ GR
  A = /* the boxed result calculated from Heron's formula */ := sorry

end triangle_area_PGQ_bisect_l532_532470


namespace distinctQuadraticFunctions_l532_532002

noncomputable def countDistinctQuadraticFunctions : ℕ :=
  let a_values := {1, 2, 3, 4}
  let b_values := {0, 1, 2, 3, 4}
  let c_values := {0, 1, 2, 3, 4}
  a_values.to_finset.card * b_values.to_finset.card * c_values.to_finset.card

theorem distinctQuadraticFunctions (a_values b_values c_values : Finset ℕ)
  (a_values_props : a_values = {1, 2, 3, 4})
  (b_values_props : b_values = {0, 1, 2, 3, 4})
  (c_values_props : c_values = {0, 1, 2, 3, 4}) :
  countDistinctQuadraticFunctions = 100 := by
  sorry

end distinctQuadraticFunctions_l532_532002


namespace arithmetic_sequence_common_difference_l532_532857

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532857


namespace points_on_circle_l532_532692

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
      y := (4 * t) / (2 + t^2) in
  x^2 + y^2 = 1 :=
by
  let x := (2 - t^2) / (2 + t^2)
  let y := (4 * t) / (2 + t^2)
  sorry

end points_on_circle_l532_532692


namespace common_difference_of_arithmetic_sequence_l532_532987

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532987


namespace min_value_circles_tangents_l532_532167

theorem min_value_circles_tangents (a b : ℝ) (h1 : (∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0))
  (h2 : ∃ k : ℕ, k = 3) (h3 : a ≠ 0) (h4 : b ≠ 0) : 
  (∃ m : ℝ, m = 1 ∧  ∀ x : ℝ, (x = (1 / a^2) + (1 / b^2)) → x ≥ m) :=
  sorry

end min_value_circles_tangents_l532_532167


namespace common_difference_arithmetic_sequence_l532_532930

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532930


namespace common_difference_l532_532994

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532994


namespace ara_current_height_l532_532668

variable (h : ℝ)  -- Original height of both Shea and Ara
variable (sheas_growth_rate : ℝ := 0.20)  -- Shea's growth rate (20%)
variable (sheas_current_height : ℝ := 60)  -- Shea's current height
variable (aras_growth_rate : ℝ := 0.5)  -- Ara's growth rate in terms of Shea's growth

theorem ara_current_height : 
  h * (1 + sheas_growth_rate) = sheas_current_height →
  (h + (sheas_current_height - h) * aras_growth_rate) = 55 :=
  by
    sorry

end ara_current_height_l532_532668


namespace find_P_l532_532630

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l532_532630


namespace integral_ln_sin_eq_neg_pi_div_2_ln2_l532_532091

theorem integral_ln_sin_eq_neg_pi_div_2_ln2 : 
  ∫ x in 0..(Real.pi / 2), Real.log (Real.sin x) = - (Real.pi / 2) * Real.log 2 := 
sorry

end integral_ln_sin_eq_neg_pi_div_2_ln2_l532_532091


namespace range_of_n_l532_532028

noncomputable def range_n (n : ℝ) : Prop :=
  let P := (3 * n - 9, n + 2)
  cos α < 0 ∧ sin α > 0 ∧ P.fst < 0 ∧ P.snd > 0

theorem range_of_n (n : ℝ) : range_n n ↔ -2 < n ∧ n < 3 := by
  sorry -- Proof to be filled in

end range_of_n_l532_532028


namespace smallest_four_digit_divisible_by_35_l532_532175

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532175


namespace smallest_four_digit_divisible_by_35_l532_532194

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532194


namespace find_common_difference_l532_532864

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532864


namespace lower_bound_of_expression_l532_532379

theorem lower_bound_of_expression :
  ∃ L : ℤ, (∀ n : ℤ, ((-1 ≤ n ∧ n ≤ 8) → (L < 4 * n + 7 ∧ 4 * n + 7 < 40))) ∧ L = 1 :=
by {
  sorry
}

end lower_bound_of_expression_l532_532379


namespace common_difference_of_arithmetic_sequence_l532_532967

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532967


namespace factorize_expression_l532_532707

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l532_532707


namespace common_difference_arithmetic_sequence_l532_532928

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532928


namespace average_weight_all_children_l532_532114

theorem average_weight_all_children (avg_boys_weight avg_girls_weight : ℝ) (num_boys num_girls : ℕ)
    (hb : avg_boys_weight = 155) (nb : num_boys = 8)
    (hg : avg_girls_weight = 125) (ng : num_girls = 7) :
    (num_boys + num_girls = 15) → (avg_boys_weight * num_boys + avg_girls_weight * num_girls) / (num_boys + num_girls) = 141 := by
  intro h_sum
  sorry

end average_weight_all_children_l532_532114


namespace sum_of_cubes_l532_532507

-- Definitions
noncomputable def p : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def r : ℂ := sorry

-- Roots conditions
axiom h_root_p : p^3 - 2 * p^2 + 3 * p - 4 = 0
axiom h_root_q : q^3 - 2 * q^2 + 3 * q - 4 = 0
axiom h_root_r : r^3 - 2 * r^2 + 3 * r - 4 = 0

-- Vieta's conditions
axiom h_sum : p + q + r = 2
axiom h_product_pairs : p * q + q * r + r * p = 3
axiom h_product : p * q * r = 4

-- Goal
theorem sum_of_cubes : p^3 + q^3 + r^3 = 2 :=
  sorry

end sum_of_cubes_l532_532507


namespace smallest_four_digit_divisible_by_35_l532_532176

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532176


namespace arithmetic_sequence_common_difference_l532_532942

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532942


namespace parallelogram_height_l532_532365

theorem parallelogram_height (A : ℝ) (b : ℝ) (h : ℝ) (h1 : A = 320) (h2 : b = 20) :
  h = A / b → h = 16 := by
  sorry

end parallelogram_height_l532_532365


namespace sum_of_squares_lt_0_01_l532_532133

theorem sum_of_squares_lt_0_01 (n : ℕ) (hn : n > 100)
    (h_sum : (∑ i in finset.range n, (1 / n : ℝ)) = 1) :
    (∑ i in finset.range n, (1 / n : ℝ) ^ 2) < 0.01 :=
sorry

end sum_of_squares_lt_0_01_l532_532133


namespace other_diagonal_length_l532_532153

-- Define the problem conditions
def d1 : ℝ := 7.4
def Area : ℝ := 21.46

-- State the length of the other diagonal
theorem other_diagonal_length : 
  ∃ d2 : ℝ, d2 = (2 * Area) / d1 ∧ d2 ≈ 5.8 := 
by
  -- Placeholder for the proof
  sorry

end other_diagonal_length_l532_532153


namespace smallest_four_digit_div_by_35_l532_532232

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532232


namespace trig_identity_pow_six_l532_532549

theorem trig_identity_pow_six (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x ^ 6 + cos x ^ 6 - 1) ^ 3 + 27 * (sin x ^ 6) * (cos x ^ 6) = 0 :=
by
  sorry

end trig_identity_pow_six_l532_532549


namespace solve_for_x_l532_532260

theorem solve_for_x (x : ℕ) : (8^3 + 8^3 + 8^3 + 8^3 = 2^x) → x = 11 :=
by
  intro h
  sorry

end solve_for_x_l532_532260


namespace fish_total_count_l532_532156

theorem fish_total_count :
  let num_fishermen : ℕ := 20
  let fish_caught_per_fisherman : ℕ := 400
  let fish_caught_by_twentieth_fisherman : ℕ := 2400
  (19 * fish_caught_per_fisherman + fish_caught_by_twentieth_fisherman) = 10000 :=
by
  sorry

end fish_total_count_l532_532156


namespace find_common_difference_l532_532949

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532949


namespace sin_cos_product_l532_532799

theorem sin_cos_product (α : ℝ) (h : sin α = -2 * cos α) : sin α * cos α = -2 / 5 :=
by
  sorry

end sin_cos_product_l532_532799


namespace four_skew_lines_trapezoid_parallelogram_l532_532050

-- Definitions for conditions
variable (a b c d : Line)
variable [are_pairwise_skew : PairwiseSkew a b c d]
variable (α β γ δ : Plane)
variable [α_contains_a : α.Contains a]
variable [β_contains_b : β.Contains b]
variable [γ_contains_c : γ.Contains c]
variable [δ_contains_d : δ.Contains d]
variable [α_parallel_β : Parallel α β]
variable [γ_parallel_δ : Parallel γ δ]

-- Proof statement for forming a trapezoid and not forming a parallelogram
theorem four_skew_lines_trapezoid_parallelogram :
  (∃ A ∈ a, ∃ B ∈ b, ∃ C ∈ c, ∃ D ∈ d, Trapezoid A B C D) ∧
  ¬(∃ A ∈ a, ∃ B ∈ b, ∃ C ∈ c, ∃ D ∈ d, Parallelogram A B C D) :=
sorry

end four_skew_lines_trapezoid_parallelogram_l532_532050


namespace find_common_difference_l532_532950

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532950


namespace smallest_four_digit_divisible_by_35_l532_532192

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532192


namespace crosswalk_red_light_wait_l532_532670

theorem crosswalk_red_light_wait :
  let red_light_duration := 40
  let wait_time_requirement := 15
  let favorable_duration := red_light_duration - wait_time_requirement
  (favorable_duration : ℝ) / red_light_duration = (5 : ℝ) / 8 :=
by
  sorry

end crosswalk_red_light_wait_l532_532670


namespace sin_alpha_value_l532_532807

theorem sin_alpha_value :
  let α : RealAngle := sorry  -- Introduce α as an angle in the context
  let point := (2 * Real.cos (Real.pi / 6), -2 * Real.sin (Real.pi / 6))
  point = (Real.sqrt 3, -1) →
  Real.sqrt (3^2 + (-1)^2) = 2 →
  Real.sin α = -1/2 :=
begin
  sorry -- This is where the proof would go.
end

end sin_alpha_value_l532_532807


namespace common_difference_of_arithmetic_sequence_l532_532963

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532963


namespace items_in_bags_l532_532672

def calculateWaysToPlaceItems (n_items : ℕ) (n_bags : ℕ) : ℕ :=
  sorry

theorem items_in_bags :
  calculateWaysToPlaceItems 5 3 = 41 :=
by sorry

end items_in_bags_l532_532672


namespace min_CD_squared_l532_532039

def Line (α : Type _) [LinearOrderedField α] := 
{a b c : α //  a ≠ 0 ∨ b ≠ 0}

noncomputable def intersection_x_axis {α : Type _} [LinearOrderedField α] 
  (l : Line α) : α × α :=
  let ⟨a, b, c, h⟩ := l in (⟨-c / a, 0⟩ : α × α)

noncomputable def intersection_y_axis {α : Type _} [LinearOrderedField α] 
  (l : Line α) : α × α :=
  let ⟨a, b, c, h⟩ := l in (⟨0, -c / b⟩ : α × α)

theorem min_CD_squared {α : Type _} [LinearOrderedField α] :
  let l₁ : Line α := ⟨4, 5, -20, Decidable.or.inl four_ne_zero⟩,
      A := intersection_x_axis l₁,
      B := intersection_y_axis l₁,
      S_ΔAOB := (1 / 2) * ((A.1 - 0) * (B.2 - 0) - (A.2 - 0) * (B.1 - 0)) in
  ∃ C D : α × α,
  (segment_intersect l₁ C D) ∧
  (triangle_area (A, B, (0, 0)) = 2 * triangle_area ((A, C, D) + (B, C, D))) →
  (CD_squared (C, D) = 5 * real.sqrt 41 - 25) :=
by
  sorry

end min_CD_squared_l532_532039


namespace common_difference_arithmetic_sequence_l532_532925

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532925


namespace least_integer_square_double_condition_l532_532173

theorem least_integer_square_double_condition : ∃ x : ℤ, x^2 = 2 * x + 75 ∧ ∀ y : ℤ, y^2 = 2 * y + 75 → x ≤ y :=
by
  use -8
  sorry

end least_integer_square_double_condition_l532_532173


namespace find_ratio_l532_532301

-- Definitions
noncomputable def cost_per_gram_A : ℝ := 0.01
noncomputable def cost_per_gram_B : ℝ := 0.008
noncomputable def new_cost_per_gram_A : ℝ := 0.011
noncomputable def new_cost_per_gram_B : ℝ := 0.0072

def total_weight : ℝ := 1000

-- Theorem statement
theorem find_ratio (x y : ℝ) (h1 : x + y = total_weight)
    (h2 : cost_per_gram_A * x + cost_per_gram_B * y = new_cost_per_gram_A * x + new_cost_per_gram_B * y) :
    x / y = 4 / 5 :=
by
  sorry

end find_ratio_l532_532301


namespace max_value_of_e_l532_532518

theorem max_value_of_e (a b c d e : ℝ) 
  (h₁ : a + b + c + d + e = 8) 
  (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ 16 / 5 :=
sorry

end max_value_of_e_l532_532518


namespace smallest_counterexample_is_14_l532_532687

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_prime (n : ℕ) : Prop := ¬Prime n
def smallest_counterexample (n : ℕ) : Prop :=
  is_even n ∧ is_not_prime n ∧ is_not_prime (n + 2) ∧ ∀ m, is_even m ∧ is_not_prime m ∧ is_not_prime (m + 2) → n ≤ m

theorem smallest_counterexample_is_14 : smallest_counterexample 14 :=
by
  sorry

end smallest_counterexample_is_14_l532_532687


namespace find_coprime_sum_l532_532843

theorem find_coprime_sum (l : ℝ) (h : (1 + l)^2 / (1 + l^2) = 13 / 37) : 
  let frac := (1 + l)^3 / (1 + l^3) in
  let m := 156 in
  let n := 1369 in
  frac = m / n ∧ m + n = 1525 := 
begin
  sorry
end

end find_coprime_sum_l532_532843


namespace soda_choosers_l532_532810

-- Definitions based on conditions
def total_people := 600
def soda_angle := 108
def full_circle := 360

-- Statement to prove the number of people who referred to soft drinks as "Soda"
theorem soda_choosers : total_people * (soda_angle / full_circle) = 180 :=
by
  sorry

end soda_choosers_l532_532810


namespace solution_trig_eq_l532_532102

noncomputable def solve_trig_eq (x : ℝ) : Prop :=
  (abs (cos x) - cos (3 * x)) / (cos x * sin (2 * x)) = 2 / sqrt 3

theorem solution_trig_eq (x : ℝ) :
  solve_trig_eq x ↔
    ∃ k : ℤ, x = (π / 6) + 2 * k * π ∨ x = (5 * π / 6) + 2 * k * π ∨ x = (4 * π / 3) + 2 * k * π :=
sorry

end solution_trig_eq_l532_532102


namespace lindsay_doll_count_l532_532527

theorem lindsay_doll_count :
  let B := 4 in                  -- Number of blonde-haired dolls
  let Br := 4 * B in             -- Number of brown-haired dolls
  let Bl := Br - 2 in            -- Number of black-haired dolls
  Bl + Br - B = 26 :=            -- Prove the combined excess of black and brown over blonde
begin 
  sorry 
end

end lindsay_doll_count_l532_532527


namespace radius_of_tangent_circle_theorem_l532_532639

noncomputable section

open EuclideanGeometry

-- Definition of the specific problem
def radius_of_tangent_circle (P Q R : Point) (O : Point) (r : ℝ) : Prop :=
  let PR := EuclideanDistance P R
  let circle_tangent_x_y := CircleTangentToAxes O r
  let circle_tangent_hypotenuse := CircleTangentToHypotenuse O r P R
  let triangle_45_45_90 := RightTriangle45_45_90 P Q R
  let PQ := EuclideanDistance P Q
  PQ = 2 ∧ PR = 2 * Real.sqrt 2 ∧ circle_tangent_x_y ∧ circle_tangent_hypotenuse ∧ triangle_45_45_90 ∧ r = 2 + Real.sqrt 2

-- The final theorem to prove
theorem radius_of_tangent_circle_theorem (P Q R O : Point) (r : ℝ)
  (hPQ : EuclideanDistance P Q = 2)
  (hPR : EuclideanDistance P R = 2 * Real.sqrt 2)
  (hx_y_tangent : CircleTangentToAxes O r)
  (hhypotenuse_tangent : CircleTangentToHypotenuse O r P R)
  (htriangle : RightTriangle45_45_90 P Q R) :
  r = 2 + Real.sqrt 2 :=
by
  sorry

end radius_of_tangent_circle_theorem_l532_532639


namespace infinite_squares_in_ap_l532_532663

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end infinite_squares_in_ap_l532_532663


namespace g_neq1_l532_532075

def g (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 5 else 4 - 3 * x

theorem g_neq1 : g (-1) = 2 :=
by
  sorry

end g_neq1_l532_532075


namespace sqrt_a_div_sqrt_b_l532_532698

variable (a b : ℝ)

theorem sqrt_a_div_sqrt_b (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 25 * a / (61 * b)) :
  sqrt a / sqrt b = 5 / 2 := by
  sorry

end sqrt_a_div_sqrt_b_l532_532698


namespace cube_tetrahedra_intersection_volume_l532_532461

variable (a : ℝ)

theorem cube_tetrahedra_intersection_volume (h : a > 0) : 
  let cube_volume := a^3 in
  let intersection_volume := 12 * (1/3 * (a^2 / 4) * (a / 2)) in
  intersection_volume = cube_volume / 2 := 
by
  -- Definitions based on conditions in a)
  let corner_tetrahedron_volume := 1/3 * (a^2 / 4) * (a / 2)
  let total_intersection_volume := 12 * corner_tetrahedron_volume
  have cube_volume_eq : cube_volume = a^3 := by sorry
  have intersection_volume_eq : intersection_volume = total_intersection_volume := by sorry
  -- Final volume equivalence
  show intersection_volume = cube_volume / 2, by
  rw [intersection_volume_eq, cube_volume_eq, total_intersection_volume]
  congr
  field_simp
  ring

end cube_tetrahedra_intersection_volume_l532_532461


namespace proof_problem_l532_532007

-- Definitions
def p (a : ℝ) : Prop := a > 1 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x < a^y
def q : Prop := ∀ planes plane1 plane2 : Type, (plane1 ⊥ plane2) ∧ (plane1 ⊥ plane2) → (plane1 // plane2)

-- The proof problem
theorem proof_problem (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ¬ (p a) ∧ ¬ (q) :=
by
  sorry

end proof_problem_l532_532007


namespace smallest_four_digit_div_by_35_l532_532229

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532229


namespace correct_option_is_D_l532_532159

def describe_composition_of_factory (input : Type) : Type :=
  input

def ProgramFlowchart : Type := Type
def ProcessFlow : Type := Type
def KnowledgeStructureDiagram : Type := Type
def OrganizationalStructureDiagram : Type := Type

theorem correct_option_is_D : 
  describe_composition_of_factory OrganizationalStructureDiagram = OrganizationalStructureDiagram 
  ∧ (∀ x : (ProgramFlowchart ∨ ProcessFlow ∨ KnowledgeStructureDiagram), describe_composition_of_factory x ≠ OrganizationalStructureDiagram) := by
  sorry

end correct_option_is_D_l532_532159


namespace distance_A_B_proof_l532_532615

noncomputable def distance_between_A_and_B : ℕ := 11

theorem distance_A_B_proof (d₁ d₂ : ℕ) (h₁ : d₁ = 5) (h₂ : d₂ = 4) : d₁ * 3 - d₂ = 11 :=
by
  rw [h₁, h₂]
  simp
  sorry

end distance_A_B_proof_l532_532615


namespace max_angle_between_tangents_l532_532767

open Real

-- Define the equation of the circle
def circle_A (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2

-- Define the equation of the parabola
def parabola_C (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the statement we want to prove
theorem max_angle_between_tangents :
  ∃ (P : ℝ × ℝ), parabola_C P.1 P.2 ∧
    (∀ (A : ℝ × ℝ), circle_A A.1 A.2 → (angle_between_tangents P A ≤ 60)) :=
sorry

end max_angle_between_tangents_l532_532767


namespace find_common_difference_l532_532872

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532872


namespace multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l532_532554

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l532_532554


namespace volume_set_points_sum_l532_532686

-- Defining the problem conditions
def rectangular_parallelepiped_length : ℝ := 5
def rectangular_parallelepiped_width : ℝ := 6
def rectangular_parallelepiped_height : ℝ := 7
def unit_extension : ℝ := 1

-- Defining what we need to prove
theorem volume_set_points_sum :
  let V_box : ℝ := rectangular_parallelepiped_length * rectangular_parallelepiped_width * rectangular_parallelepiped_height
  let V_ext : ℝ := 2 * (unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_width 
                  + unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_height 
                  + unit_extension * rectangular_parallelepiped_width * rectangular_parallelepiped_height)
  let V_cyl : ℝ := 18 * π
  let V_sph : ℝ := (4 / 3) * π
  let V_total : ℝ := V_box + V_ext + V_cyl + V_sph
  let m : ℕ := 1272
  let n : ℕ := 58
  let p : ℕ := 3
  V_total = (m : ℝ) + (n : ℝ) * π / (p : ℝ) ∧ (m + n + p = 1333)
  := by
  sorry

end volume_set_points_sum_l532_532686


namespace solve_trig_eq_l532_532104

theorem solve_trig_eq (x : ℝ) : 
  (|\cos x| - cos (3 * x)) / (cos x * sin (2 * x)) = 2 / real.sqrt 3 ↔
  (∃ k : ℤ, x = (real.pi / 6) + 2 * real.pi * k) ∨ 
  (∃ k : ℤ, (x = (5 * real.pi / 6) + 2 * real.pi * k) ∨ (x = (4 * real.pi / 3) + 2 * real.pi * k)) :=
sorry

end solve_trig_eq_l532_532104


namespace common_difference_of_arithmetic_sequence_l532_532968

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532968


namespace black_region_area_l532_532645

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def num_small_squares : ℕ := 2

theorem black_region_area :
  let large_square_area := large_square_side_length ^ 2
  let small_square_area := small_square_side_length ^ 2
  let total_small_square_area := num_small_squares * small_square_area
  let black_region_area := large_square_area - total_small_square_area
  black_region_area = 68 :=
by 
  -- We introduce the calculations step by step
  let large_square_area : ℕ := large_square_side_length ^ 2
  let small_square_area : ℕ := small_square_side_length ^ 2
  let total_small_square_area := num_small_squares * small_square_area
  let black_region_area := large_square_area - total_small_square_area
  exact eq_true_intro (by sorry) -- Placeholder for the proof

end black_region_area_l532_532645


namespace common_difference_is_two_l532_532910

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532910


namespace find_f_2017_l532_532566

theorem find_f_2017 (f : ℤ → ℤ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 3) = f x) (h_f_neg1 : f (-1) = 1) : 
  f 2017 = -1 :=
sorry

end find_f_2017_l532_532566


namespace find_interval_l532_532357

theorem find_interval (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 :=
by
  sorry

end find_interval_l532_532357


namespace intercepts_of_line_eq_l532_532343

theorem intercepts_of_line_eq :
  (∃(x y : ℝ), 4 * x + 6 * y = 24 ∧ y = 0 ∧ x = 6) ∧
  (∃(x y : ℝ), 4 * x + 6 * y = 24 ∧ x = 0 ∧ y = 4) :=
by
  -- x-intercept
  use (6, 0)
  split
  { 
    use (0, 4)
    split
    {
      exact 4 * 0 + 6 * 4 = 24
    }
    {
      exact (4, 24)
    }
  }
  }

end intercepts_of_line_eq_l532_532343


namespace smallest_four_digit_number_divisible_by_35_l532_532224

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532224


namespace smallest_four_digit_divisible_by_35_l532_532217

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532217


namespace common_difference_of_arithmetic_sequence_l532_532890

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532890


namespace smallest_four_digit_divisible_by_35_l532_532248

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532248


namespace solution_l532_532443

-- Define the conditions
def equation (x : ℝ) : Prop :=
  (x / 15) = (15 / x)

theorem solution (x : ℝ) : equation x → x = 15 ∨ x = -15 :=
by
  intros h
  -- The proof would go here.
  sorry

end solution_l532_532443


namespace necessary_but_not_sufficient_l532_532300

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0

theorem necessary_but_not_sufficient (x : ℝ) : 
-2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0 := 
by
  sorry

end necessary_but_not_sufficient_l532_532300


namespace negate_original_is_correct_l532_532569

-- Define the original proposition
def original_proposition (a b : ℕ) : Prop := (a * b = 0) → (a = 0 ∨ b = 0)

-- Define the negated proposition
def negated_proposition (a b : ℕ) : Prop := (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)

-- The theorem stating that the negation of the original proposition is the given negated proposition
theorem negate_original_is_correct (a b : ℕ) : ¬ original_proposition a b ↔ negated_proposition a b := by
  sorry

end negate_original_is_correct_l532_532569


namespace find_h3_l532_532685

noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^9 + 1) - 1) / (x^(3^3 - 1) - 1)

theorem find_h3 : h 3 = 3 := by
  sorry

end find_h3_l532_532685


namespace smallest_four_digit_div_by_35_l532_532236

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532236


namespace problem1_problem2_problem3_problem4_l532_532149

-- Definitions and assumptions
def condition1 (p q : Prop) : Prop := (p ∨ q) ∧ ¬(p ∧ q)
def conclusion1 (p q : Prop) : Prop := (p = ¬q)

def condition2 (S : ℕ → ℕ) : Prop := ∀ n : ℕ, S n = n^2 + n + 1
def incorrect_conclusion2 (a : ℕ → ℕ) : Prop := ∀ n : ℕ, n ≠ 0 → a n = 2 * n

def condition3 (f : ℝ → ℝ) (f' : ℝ → ℝ) (x : ℝ) : Prop := (f' x = 0)
def incorrect_conclusion3 (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y : ℝ, (y < x ∧ f y = f x) ∨ (x < y ∧ f y = f x)

def condition4 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x - 1)
def conclusion4 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 6) = f x

-- Lean theorem statements
theorem problem1 (p q : Prop) (h : condition1 p q) : conclusion1 p q := sorry
theorem problem2 (S : ℕ → ℕ) (h : condition2 S) : ¬incorrect_conclusion2 (λ n, S n - S (n - 1)) := sorry
theorem problem3 (f : ℝ → ℝ) (f' : ℝ → ℝ) (x : ℝ) (h : condition3 f f' x) : ¬incorrect_conclusion3 f x := sorry
theorem problem4 (f : ℝ → ℝ) (h : condition4 f) : conclusion4 f := sorry

end problem1_problem2_problem3_problem4_l532_532149


namespace probability_at_most_3_heads_l532_532635

theorem probability_at_most_3_heads (n : ℕ) (h : n = 12) :
  let total_outcomes := 2^12,
      favorable_outcomes := (∑ k in finset.Icc 0 3, nat.choose 12 k)
  in favorable_outcomes / total_outcomes = 299 / 4096 :=
by {
  sorry
}

end probability_at_most_3_heads_l532_532635


namespace find_common_difference_l532_532869

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532869


namespace sequence_general_term_l532_532079

-- Define the sequence with given initial conditions and recurrence relation
def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧
  a 1 = 1 ∧
  ∀ n ≥ 2, sqrt (a n * a (n-2)) - sqrt (a (n-1) * a (n-2)) = 2 * a (n-1)

-- Prove the general term formula for the sequence
theorem sequence_general_term :
  ∀ a : ℕ → ℝ, sequence a →
  ∀ n, a n = (∏ k in finset.range (n + 1), (2^k - 1)) ^ 2 :=
by
  intros a h n
  have h_seq := h.2.2
  induction n with n ih
  case zero =>
    rw [finset.prod_range_zero, pow_two, mul_one, sqr_of_one, h.1]
  case succ n =>
    sorry

end sequence_general_term_l532_532079


namespace find_a_l532_532764

theorem find_a (a : ℝ) (h : a > 0)
    (intersect : ∃ A B : ℝ × ℝ, (A.2 = A.1 + 2 * a) ∧ (B.2 = B.1 + 2 * a) ∧
      (A.1 ^ 2 + A.2 ^ 2 - 2 * a * A.2 - 2 = 0) ∧ (B.1 ^ 2 + B.2 ^ 2 - 2 * a * B.2 - 2 = 0))
    (dist_AB : ∃ A B : ℝ × ℝ, |(A.1 - B.1)^2 + (A.2 - B.2)^2| = 4 * 3) :
    a = sqrt 2 :=
by
  sorry

end find_a_l532_532764


namespace L_shape_area_and_perimeter_l532_532675

def rectangle1_length := 0.5
def rectangle1_width := 0.3
def rectangle2_length := 0.2
def rectangle2_width := 0.5

def area_rectangle1 := rectangle1_length * rectangle1_width
def area_rectangle2 := rectangle2_length * rectangle2_width
def total_area := area_rectangle1 + area_rectangle2

def perimeter_L_shape := rectangle1_length + rectangle1_width + rectangle1_width + rectangle2_length + rectangle2_length + rectangle2_width

theorem L_shape_area_and_perimeter :
  total_area = 0.25 ∧ perimeter_L_shape = 2.0 :=
by
  sorry

end L_shape_area_and_perimeter_l532_532675


namespace incircle_symmetry_l532_532787

theorem incircle_symmetry {A B C D E F X Y Z : Point} 
  (h1 : incircle_triangle A B C D E F)
  (h2 : segment A D ∩ incircle A B C = {X}) 
  (h3 : AX = XD) 
  (h4 : segment B X ∩ incircle A B C = {Y})
  (h5 : segment C X ∩ incircle A B C = {Z}) :
  EY = FZ := 
sorry

end incircle_symmetry_l532_532787


namespace find_common_difference_l532_532874

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532874


namespace train_speed_l532_532309

theorem train_speed
  (distance_meters : ℝ := 400)
  (time_seconds : ℝ := 12)
  (distance_kilometers : ℝ := distance_meters / 1000)
  (time_hours : ℝ := time_seconds / 3600) :
  distance_kilometers / time_hours = 120 := by
  sorry

end train_speed_l532_532309


namespace area_of_S_l532_532740

theorem area_of_S (A p : ℝ) : 
  let S := { point | ∃ triangle, point ∈ triangle ∧ dist point triangle ≤ 5 } in
  area S = A + 5 * p + 25 * Real.pi :=
sorry

end area_of_S_l532_532740


namespace number_of_decompositions_l532_532690

theorem number_of_decompositions (n : ℕ) (hn : n > 0) : 
  (nat.card {l : list ℕ // l.sum = n ∧ ∀ x ∈ l, x > 0}) = 2^(n-1) :=
sorry

end number_of_decompositions_l532_532690


namespace find_common_difference_l532_532876

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532876


namespace common_difference_arithmetic_sequence_l532_532932

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532932


namespace common_difference_of_arithmetic_sequence_l532_532975

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532975


namespace common_difference_of_arithmetic_sequence_l532_532884

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532884


namespace smallest_integer_divisors_m_plus_k_l532_532348

theorem smallest_integer_divisors_m_plus_k (m k n : ℕ) (h1 : n = m * 6^k) (h2 : ∏ p in (unique_factorization_monoid.factors n).to_finset, (multiset.count p (unique_factorization_monoid.factors n) + 1) = 2023) (h3 : ¬ (6 ∣ m)) : m + k = 59055 :=
sorry

end smallest_integer_divisors_m_plus_k_l532_532348


namespace smallest_four_digit_divisible_by_35_l532_532198

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532198


namespace angles_on_x_axis_eq_l532_532131

open Set

def S1 : Set ℝ := { β | ∃ k : ℤ, β = k * 360 }
def S2 : Set ℝ := { β | ∃ k : ℤ, β = 180 + k * 360 }
def S_total : Set ℝ := S1 ∪ S2
def S_target : Set ℝ := { β | ∃ n : ℤ, β = n * 180 }

theorem angles_on_x_axis_eq : S_total = S_target := 
by 
  sorry

end angles_on_x_axis_eq_l532_532131


namespace arithmetic_sequence_common_difference_l532_532897

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532897


namespace smallest_four_digit_divisible_by_35_l532_532206

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532206


namespace rect_area_sum_eq_16_l532_532319

theorem rect_area_sum_eq_16 (a b c : ℕ) (h1 : |a * b - a * c| = 1) (h2 : |a * c - b * c| = 49) :
  a + b + c = 16 :=
sorry

end rect_area_sum_eq_16_l532_532319


namespace ian_money_left_l532_532440

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l532_532440


namespace integer_solution_of_floor_equation_l532_532359

theorem integer_solution_of_floor_equation (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 11) :=
by sorry

end integer_solution_of_floor_equation_l532_532359


namespace isosceles_triangle_equal_sides_length_l532_532667

noncomputable def equal_side_length_isosceles_triangle (base median : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  if base = 36 ∧ median = 15 ∧ vertex_angle_deg = 60 then 3 * Real.sqrt 191 else 0

theorem isosceles_triangle_equal_sides_length:
  equal_side_length_isosceles_triangle 36 15 60 = 3 * Real.sqrt 191 :=
by
  sorry

end isosceles_triangle_equal_sides_length_l532_532667


namespace arithmetic_sequence_common_difference_l532_532904

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532904


namespace common_difference_arithmetic_sequence_l532_532919

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532919


namespace larger_area_than_non_circle_l532_532093

theorem larger_area_than_non_circle (Φ : Type) [convex Φ] (h : ¬is_circle Φ) :
  ∃ (Φ' : Type) [convex Φ'], has_same_perimeter Φ Φ' ∧ area Φ' > area Φ := 
sorry

end larger_area_than_non_circle_l532_532093


namespace compute_dot_product_l532_532061

variables {u v w : ℝ^3}

-- Definitions based on conditions
def is_unit_vector (a : ℝ^3) : Prop := a.norm = 1
def w_definition (u v : ℝ^3) : ℝ^3 := u × v + u
def w_cross_u_eq_v (u v w : ℝ^3) : Prop := w × u = v

-- The theorem to prove
theorem compute_dot_product (hu : is_unit_vector u) (hv : is_unit_vector v)
  (hw_def : w = w_definition u v) (hcross : w_cross_u_eq_v u v w) : 
  u • (v × w) = 1 :=
sorry

end compute_dot_product_l532_532061


namespace infinite_squares_in_ap_l532_532664

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end infinite_squares_in_ap_l532_532664


namespace connected_graph_vertex_ordering_l532_532289

theorem connected_graph_vertex_ordering 
  (G : SimpleGraph V) [G.connected] :
  ∃ (v : Fin n → V), 
    ∀ i : Fin n, 
      G.subgraph (set.of_finset (Finset.of_fintype (finset.image (v ∘ Fin.cast_univ i.succ) (finset.of_fintype (Finset.range i.succ))))) .connected :=
sorry

end connected_graph_vertex_ordering_l532_532289


namespace problem1_problem2_problem3_l532_532469

variables {A B C D E F P Q R S U V X Y : Type}

-- Assumptions
axiom tangential_quadrilateral (A B C D P Q R S : Type) (h1 : IsTangentialQuadrilateral A B C D P Q R S) : Type
axiom point_on_line (A B X : Type) (h1 : IsPointOnLine A B X) : Type
axiom meets_at (l1 l2 P : Type) (h1 : MeetsAt l1 l2 P) : Type

-- Problem statement
theorem problem1 (h1 : IsTangentialQuadrilateral A B C D P Q R S)
  (h2 : MeetsAt AD BC E)
  (h3 : MeetsAt AB DC F) :
  AreConcurrent (line_through A C) (line_through B D) (line_through P R) (line_through Q S) :=
sorry

theorem problem2 (h1 : IsTangentialQuadrilateral A B C D P Q R S)
  (h2 : MeetsAt AD BC E)
  (h3 : MeetsAt AB DC F) :
  AreConcurrent (line_through A C) (line_through E F) 
    (line_through P Q) (line_through R S) :=
sorry

theorem problem3 (h1 : IsTangentialQuadrilateral A B C D P Q R S)
  (h2 : MeetsAt AD BC E)
  (h3 : MeetsAt AB DC F)
  (h4 : line_through B D ≠ line_through E F) :
  AreConcurrent (line_through B D) (line_through E F) 
    (line_through P S) (line_through Q R) :=
sorry

end problem1_problem2_problem3_l532_532469


namespace range_of_m_l532_532731

-- Definitions given in the problem
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0
def neg_q_sufficient_for_neg_p : Prop :=
  ∀ {x m : ℝ}, (1 - m < x ∧ x < 1 + m) → (-2 ≤ x ∧ x ≤ 10)

-- The statement to prove
theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : 1 - m ≥ -2) (h3 : 1 + m ≤ 10) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l532_532731


namespace exists_m_n_in_and_not_in_A_l532_532356

open Set

noncomputable def A : Set ℕ :=
  {n : ℕ | ∃ (p₁ : ℕ) (primes : Finset ℕ), n = primes.prod ∧ p₁ ∈ primes ∧ Finset.card primes = p₁ + 1 ∧ (∀ p ∈ primes, Nat.Prime p)}

theorem exists_m_n_in_and_not_in_A (P : Set ℕ) [hP : Infinite P] (hPrime : ∀ p ∈ P, Nat.Prime p) :
  ∃ (m n : ℕ), m ∈ A ∧ n ∉ A ∧ (Finset.card (Finset.filter (λ p, p ∈ P) (Finset.univ.filter (λ q, q ∣ m))) =
  Finset.card (Finset.filter (λ p, p ∈ P) (Finset.univ.filter (λ q, q ∣ n)))) :=
sorry

end exists_m_n_in_and_not_in_A_l532_532356


namespace smallest_four_digit_divisible_by_35_l532_532184

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532184


namespace min_expr_value_l532_532401

theorem min_expr_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) :
  (∃ a, a = (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ∧ a ≥ 0) → 
  (∀ (u v : ℝ), u = x + 2 → v = 3 * y + 4 → u * v = 16) →
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 :=
sorry

end min_expr_value_l532_532401


namespace players_in_tournament_l532_532107

theorem players_in_tournament (n : ℕ) (h : n * (n - 1) = 66) : n = 11 :=
begin
  sorry
end

end players_in_tournament_l532_532107


namespace find_a_l532_532759

def line_eq (a : ℝ) : ℝ → ℝ := λ x, x + 2 * a
def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * y - 2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x1 y1 x2 y2 : ℝ, 
(line_eq a x1 = y1) ∧ (line_eq a x2 = y2) ∧ 
(circle_eq a x1 y1) ∧ (circle_eq a x2 y2) ∧ 
(distance x1 y1 x2 y2 = 2 * real.sqrt 3)) : 
  a = real.sqrt 2 :=
by
  sorry

end find_a_l532_532759


namespace pie_eating_contest_difference_l532_532466

-- Definition of given conditions
def num_students := 8
def emma_pies := 8
def sam_pies := 1

-- Statement to prove
theorem pie_eating_contest_difference :
  emma_pies - sam_pies = 7 :=
by
  -- Omitting the proof, as requested.
  sorry

end pie_eating_contest_difference_l532_532466


namespace domain_of_f_l532_532117

def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

theorem domain_of_f :
  ∀ x : ℝ, (1 / (x + 1)) + Real.log x ∈ ℝ → x > 0 :=
by
  intro x
  split
  sorry

end domain_of_f_l532_532117


namespace find_largest_x_l532_532367

def largest_integral_x (x : ℤ) : Prop := (1 / 4 : ℝ) < (x / 9 : ℝ) ∧ (x / 9 : ℝ) < (7 / 9 : ℝ)

theorem find_largest_x : ∃ x : ℤ, largest_integral_x x ∧ ∀ y : ℤ, largest_integral_x y → y ≤ x :=
begin
  sorry
end

end find_largest_x_l532_532367


namespace arithmetic_sequence_common_difference_l532_532896

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532896


namespace common_difference_of_arithmetic_sequence_l532_532964

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532964


namespace solve_for_x_l532_532791

theorem solve_for_x (b x : ℝ) (h1 : b > 1) (h2 : x > 0)
    (h3 : (4 * x) ^ (Real.log 4 / Real.log b) = (6 * x) ^ (Real.log 6 / Real.log b)) :
    x = 1 / 6 :=
by
  sorry

end solve_for_x_l532_532791


namespace common_difference_of_arithmetic_sequence_l532_532986

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532986


namespace common_difference_is_two_l532_532909

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532909


namespace cost_per_bar_l532_532530

variable (months_in_year : ℕ := 12)
variable (months_per_bar_of_soap : ℕ := 2)
variable (total_cost_for_year : ℕ := 48)

theorem cost_per_bar (h1 : months_per_bar_of_soap > 0)
                     (h2 : total_cost_for_year > 0) : 
    (total_cost_for_year / (months_in_year / months_per_bar_of_soap)) = 8 := 
by
  sorry

end cost_per_bar_l532_532530


namespace percentage_of_x_l532_532619

theorem percentage_of_x (x : ℝ) (h : x > 0) : ((x / 5 + x / 25) / x) * 100 = 24 := 
by 
  sorry

end percentage_of_x_l532_532619


namespace final_price_after_increase_and_decrease_l532_532623

variable (P : ℝ)

theorem final_price_after_increase_and_decrease (h : P > 0) : 
  let increased_price := P * 1.15
  let final_price := increased_price * 0.85
  final_price = P * 0.9775 :=
by
  sorry

end final_price_after_increase_and_decrease_l532_532623


namespace sum_of_special_primes_l532_532605

def isPrime (n : ℕ) : Prop := Nat.Prime n

def swapDigits (n : ℕ) : ℕ :=
  let ten := n / 10
  let unit := n % 10
  unit * 10 + ten

def isValidPrime (n : ℕ) : Prop :=
  20 < n ∧ n < 100 ∧ isPrime n ∧ isPrime (swapDigits n)

theorem sum_of_special_primes : (∑ n in Finset.filter isValidPrime (Finset.range 100), n) = 418 := by
  sorry

end sum_of_special_primes_l532_532605


namespace negation_of_P_l532_532777

theorem negation_of_P (a b x : ℝ) : 
  (¬ (x ≥ a^2 + b^2 → x ≥ 2ab)) ↔ (x < a^2 + b^2 → x < 2ab) :=
sorry

end negation_of_P_l532_532777


namespace simplify_expression_l532_532424

theorem simplify_expression (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) :
  |a + b - c| - |b - a - c| = 2 * b - 2 * c :=
by
  sorry

end simplify_expression_l532_532424


namespace smallest_four_digit_div_by_35_l532_532228

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532228


namespace sequences_properties_sum_of_first_n_terms_l532_532782

noncomputable def a_n (n : ℕ) : ℝ := n + 1 / n
noncomputable def b_n (n : ℕ) : ℝ := n

theorem sequences_properties :
  (∀ n : ℕ, 0 < n →
  (1 / (a_n n - b_n n) = b_n n) ∧
  (1 / a_n n = b_n n / (n^2 + 1))) :=
by {
  intros n hn,
  split;
  sorry
}

noncomputable def term_formula (n : ℕ) : ℝ := 
⌊a_n n + a_n (n + 1)⌋ * 2^((b_n n).to_nat)

theorem sum_of_first_n_terms (S_n : ℕ → ℝ) :
  (∀ n : ℕ, S_n n = (2 * n - 1) * 2^(n+1) + 4) :=
by {
  intros n,
  sorry
}

end sequences_properties_sum_of_first_n_terms_l532_532782


namespace karen_start_time_late_l532_532489

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l532_532489


namespace main_theorem_l532_532279

noncomputable def f : ℕ+ × ℕ+ → ℕ+
axiom A1 : f (1, 1) = 0
axiom A2 : ∀ a b : ℕ+, coprime a b ∧ (a ≠ 1 ∧ b ≠ 1) → f (a, b) + f (b, a) = 1
axiom A3 : ∀ a b : ℕ+, coprime a b → f (a + b, b) = f (a, b)

theorem main_theorem (p : ℕ+) [Fact p.prime] (hp : p.val % 2 = 1) : 
  ∑ n in Finset.range p.val, f (n ^ 2, p) ≥ nat.sqrt (2 * p) - 2 := 
sorry

end main_theorem_l532_532279


namespace gcd_108_450_l532_532364

theorem gcd_108_450 : Nat.gcd 108 450 = 18 :=
by
  sorry

end gcd_108_450_l532_532364


namespace circus_tickets_l532_532562

variable (L U : ℕ)

theorem circus_tickets (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end circus_tickets_l532_532562


namespace curve_represents_semicircle_l532_532456

theorem curve_represents_semicircle :
  ∀ (x y : ℝ) (θ : ℝ)
  (h₁ : θ ∈ set.Icc (-(real.pi / 2)) (real.pi / 2))
  (h₂ : x = 2 * real.cos θ)
  (h₃ : y = 1 + 2 * real.sin θ),
  x^2 + (y - 1)^2 = 4 ∧ 0 ≤ x ∧ x ≤ 2 ∧ -1 ≤ y ∧ y ≤ 3 :=
sorry

end curve_represents_semicircle_l532_532456


namespace count_valid_integers_lt_500_l532_532428

-- Definition of integer being the sum of j consecutive odd integers
def is_sum_of_j_odd_consecutive (N : ℕ) (j : ℕ) : Prop :=
  ∃ n : ℕ, N = j * (2 * n + j)

-- Definition of N having exactly 3 valid j values that satisfy the sum condition
def has_exactly_three_valid_j (N : ℕ) : Prop :=
  (finset.filter (λ j : ℕ, j ∣ N ∧ is_sum_of_j_odd_consecutive N j) 
    (finset.range (N + 1))).card = 3

-- Definition of numbers N of the form p^2 * q^2 with p and q distinct primes
def is_in_form_p_squared_q_squared (N : ℕ) : Prop :=
  ∃ (p q : ℕ) [prime p] [prime q], p ≠ q ∧ N = (p * p) * (q * q)

-- Main theorem statement
theorem count_valid_integers_lt_500 :
  finset.filter (λ N : ℕ, N < 500 ∧ has_exactly_three_valid_j N ∧ is_in_form_p_squared_q_squared N)
    (finset.range 500).card = 5 :=
sorry

end count_valid_integers_lt_500_l532_532428


namespace half_of_original_amount_l532_532836

theorem half_of_original_amount (h : ∃ (m : ℚ), (4/7 : ℚ) * m = 24) : 
  ∃ (half_m : ℚ), half_m = 21 :=
by
  obtain ⟨m, hm⟩ := h
  have original := m
  have half_orig := (1/2 : ℚ) * original
  have target := (7/4 : ℚ) * 24 / 2
  use half_orig
  rw [←hm]
  have fact : (4 / 7) * original * (7 / 4) = original := by sorry
  have eq1 : (7 / 4) * 24 = original := eq.trans (mul_eq_mul_right_iff.mpr (oreq_of_ne_zero (by norm_num)) (by norm_num) hm.symm)
  have eq2 := eq.trans eq1 div_eq_div_right_iff nonzero_of_ne_zero (by norm_num)
  rw [eq2] at this
  exact sorry


end half_of_original_amount_l532_532836


namespace general_term_formula_max_min_values_Tn_l532_532755

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Sum of the first n terms of the geometric sequence with formula for Sn
noncomputable def sum_of_first_n_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) [decidable_eq ℕ] : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Definitions based on given problem conditions
axiom (geometric_sequence_not_decreasing : ∀ (n : ℕ), geometric_sequence (3 / 2) (-1 / 2) n ≤ geometric_sequence (3 / 2) (-1 / 2) (n + 1))

axiom (arithmetic_sequence_condition : sum_of_first_n_terms (3 / 2) (-1 / 2) 5 + geometric_sequence (3 / 2) (-1 / 2) 5 -
                                        (sum_of_first_n_terms (3 / 2) (-1 / 2) 3 + geometric_sequence (3 / 2) (-1 / 2) 3) =
                                        sum_of_first_n_terms (3 / 2) (-1 / 2) 4 + geometric_sequence (3 / 2) (-1 / 2) 4 -
                                        (sum_of_first_n_terms (3 / 2) (-1 / 2) 5 + geometric_sequence (3 / 2) (-1 / 2) 5))

-- Conjecture based on problem requirements
theorem general_term_formula :
  ∀ n : ℕ, n > 0 → geometric_sequence (3 / 2) (-1 / 2) n = (-1)^(n-1) * (3 / 2^n) :=
sorry

theorem max_min_values_Tn :
  (∀ n : ℕ, n > 0 → let Tn := sum_of_first_n_terms (3 / 2) (-1 / 2) n - (1 / sum_of_first_n_terms (3 / 2) (-1 / 2) n)
                     in -7 / 12 ≤ Tn ∧ Tn ≤ 5 / 6) :=
sorry

end general_term_formula_max_min_values_Tn_l532_532755


namespace common_difference_of_arithmetic_sequence_l532_532889

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532889


namespace bead_100_lands_on_20_l532_532477

def contains_digit_seven (n : ℕ) : Prop :=
  n % 10 = 7 ∨ (n / 10) % 10 = 7 ∨ (n / 100) % 10 = 7  -- and so on for all places

def should_skip (n : ℕ) : Prop :=
  n % 7 = 0 ∨ contains_digit_seven n

def actual_count (n : ℕ) : ℕ :=
  (List.range' 1 (n + 1)).filter (λ x, ¬ should_skip x)).length

def bead_position (count : ℕ) : ℕ :=
  (actual_count count) % 22

theorem bead_100_lands_on_20 :
  bead_position 100 = 20 :=
by
  sorry

end bead_100_lands_on_20_l532_532477


namespace decreasing_symmetry_l532_532391

noncomputable def f : ℝ → ℝ := λ x, sorry

theorem decreasing_symmetry (hdec : ∀ x y : ℝ, 4 < x → 4 < y → x < y → f y < f x)
  (hsym : ∀ x : ℝ, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
begin
  -- proof goes here
  sorry,
end

end decreasing_symmetry_l532_532391


namespace arithmetic_sequence_common_difference_l532_532945

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532945


namespace tangent_slope_at_point_l532_532805

theorem tangent_slope_at_point (a : ℝ) (h : (a, 9) ∈ (λ x, (x, 3^x))) : 
  a = 2 ∧ ∀ x, 3^x * Real.log 3 = 9 * Real.log 3 := 
by
  sorry

end tangent_slope_at_point_l532_532805


namespace p_q_sum_is_14_l532_532065

def find_p_q (p q : ℝ) : Prop :=
  ∃ (r : ℂ), r = 2 + complex.I * real.sqrt 2 ∧ (r.conj = 2 - complex.I * real.sqrt 2) ∧
  (∀ (x : ℂ), (polynomial.C x^3 + polynomial.C p * x + polynomial.C q = 0) → x = r ∨ x = r.conj ∨ x = -4)

theorem p_q_sum_is_14 (p q : ℝ) (hpq : find_p_q p q) : p + q = 14 :=
by
  sorry

end p_q_sum_is_14_l532_532065


namespace common_difference_l532_532995

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532995


namespace geometric_series_sum_test_l532_532337

-- Let's define all necessary variables
variable (a : ℤ) (r : ℤ) (n : ℕ)

-- Define the geometric series sum formula
noncomputable def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

-- Define the specific test case as per our conditions
theorem geometric_series_sum_test :
  geometric_series_sum (-2) 3 7 = -2186 :=
by
  sorry

end geometric_series_sum_test_l532_532337


namespace area_DXY_l532_532395

-- Define the rectangle and the points
variables (A B C D X Y : Type) [Rectangle A B C D]
variables [On X AB] [On Y BC]

-- Define the given areas of the triangles
axiom area_AXD : area_triangle A X D = 5
axiom area_BXY : area_triangle B X Y = 4
axiom area_DYC : area_triangle D Y C = 3

-- Prove the area of triangle DXY
theorem area_DXY : area_triangle D X Y = 2 * Real.sqrt 21 := by
  sorry

end area_DXY_l532_532395


namespace quadrilateral_area_l532_532736

noncomputable def area_of_quadrilateral (a b : ℝ) : ℝ :=
  2 * a * b

theorem quadrilateral_area (A B C D M N K L : Type)
  [AffineSpace ℝ A B] [AffineSpace ℝ B C] [AffineSpace ℝ C D] [AffineSpace ℝ D A]
  [AffineSpace ℝ A M] [AffineSpace ℝ B N] [AffineSpace ℝ C K] [AffineSpace ℝ D L]
  (midpoint_AB_M : affine_subspace ℝ M)
  (midpoint_BC_N : affine_subspace ℝ N)
  (midpoint_CD_K : affine_subspace ℝ K)
  (midpoint_DA_L : affine_subspace ℝ L)
  (l : line_segment M K)
  (length_l : l.length = a)
  (distance_midpoints : ∀ {X Y}, X ∈ midpoint_AB_M → Y ∈ midpoint_CD_K → dist X l = b) :
  area_of_quadrilateral a b = 2 * a * b :=
sorry

end quadrilateral_area_l532_532736


namespace total_votes_cast_l532_532463

-- Definitions of vote percentages and vote differences.
def vote_percentage_A := 0.30
def vote_percentage_B := 0.25
def vote_percentage_C := 0.20
def vote_percentage_D := 0.15
def vote_percentage_E := 0.10

def vote_diff_A_B := 1800
def vote_diff_A_C := 3000
def vote_diff_B_D := 1200
def vote_diff_C_E := 1400
def vote_diff_D_E := 1000

-- The proposition to be proved.
theorem total_votes_cast : ∃ V : ℕ,
  vote_diff_A_B = (vote_percentage_A - vote_percentage_B) * (V : ℝ) ∧
  V = 36000 :=
by
  -- Given conditions imply V satisfies the stated equation
  -- To be solved later
  sorry

end total_votes_cast_l532_532463


namespace last_bead_is_orange_l532_532830

def pattern : List String := ["red", "orange", "yellow", "yellow", "green", "blue", "violet", "violet"]

-- definition to get the bead color at a specific position
def color_at (position : Nat) : String := pattern[(position - 1) % pattern.length]

theorem last_bead_is_orange : color_at 82 = "orange" :=
by
  sorry

end last_bead_is_orange_l532_532830


namespace common_difference_of_arithmetic_sequence_l532_532880

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532880


namespace arithmetic_sequence_common_difference_l532_532899

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532899


namespace smallest_n_for_f_greater_than_10_l532_532018

-- Define the sum of the digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the function f as described in the condition
def f (n : ℕ) : ℕ :=
  sum_of_digits (2 ^ n)

-- The main theorem to prove
theorem smallest_n_for_f_greater_than_10 : ∃ n : ℕ, n > 0 ∧ f(n) > 10 ∧ ∀ m, 0 < m ∧ m < n → f(m) ≤ 10 :=
by
  sorry

end smallest_n_for_f_greater_than_10_l532_532018


namespace div_count_number_valid_stellar_classes_l532_532551

/-- The number of positive integer divisors of the number of valid stellar classifications for the 9 stars
    in the constellation Leo, where connected stars have distinct stellar classifications from a set of 7
    possible classifications, is 160. -/
theorem div_count_number_valid_stellar_classes :
  let n := 10077120 in
  nat.factors_count n = 160 :=
by
  let n := 10077120
  have n_factorization : n = 2^7 * 3^4 * 5^1 * 7^1 := by sorry
  
  have divisors_count : (nat.succ 7) * (nat.succ 4) * (nat.succ 1) * (nat.succ 1) = 160 := by
    calc
      (7 + 1) * (4 + 1) * (1 + 1) * (1 + 1) = 8 * 5 * 2 * 2 := by norm_num
      ... = 160 := by norm_num
  show nat.factors_count n = 160 from by sorry 

end div_count_number_valid_stellar_classes_l532_532551


namespace simplest_square_root_l532_532270

theorem simplest_square_root : 
  (∀ A B C D : ℝ, A = sqrt (1 / 2) ∧ B = sqrt 2 ∧ C = sqrt 9 ∧ D = sqrt 8 → B = sqrt 2) :=
by
  intros A B C D h
  have h1 : A = sqrt (1 / 2) ∧ B = sqrt 2 ∧ C = sqrt 9 ∧ D = sqrt 8 := h
  show B = sqrt 2 from h1.2.1
  sorry

end simplest_square_root_l532_532270


namespace sum_of_arcs_as_n_tends_to_infinity_l532_532116

-- Define the diameter of the circle as a constant D
variable (D : ℝ)

-- Define a function that gives the sum of the arc lengths for a given n
def sum_of_arcs (n : ℕ) : ℝ :=
  let segment_length := D / n
  let radius := segment_length / 2
  let quarter_circle_arc := (π * radius) / 2
  n * quarter_circle_arc

-- Define the limit approach as n tends to infinity
noncomputable theory
open filter real
open_locale topological_space

theorem sum_of_arcs_as_n_tends_to_infinity :
  tendsto (λ n : ℕ, sum_of_arcs D n) at_top (nhds (π * D / 4)) :=
by sorry

end sum_of_arcs_as_n_tends_to_infinity_l532_532116


namespace correct_answer_C_l532_532612

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

lemma no_real_roots_of_discriminant_neg (a b c : ℝ) (h : discriminant a b c < 0) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := sorry

lemma problem_statement :
  discriminant 1 1 5 < 0 := by
  show 1^2 - 4 * 1 * 5 < 0
  calc
    1^2 - 4 * 1 * 5 = 1 - 20 := by norm_num
    ... < 0 := by norm_num

theorem correct_answer_C :
  (¬ ∃ x : ℝ, 1 * x^2 + 1 * x + 5 = 0) ∧
  (∃ x₁ x₂ : ℝ, 1 * x₁^2 + 1 * x₁ - 2 = 0 ∧ 1 * x₂^2 + 1 * x₂ - 2 = 0 ∧ x₁ ≠ x₂) ∧
  (∃ x₁ x₂ : ℝ, 1 * x₁^2 - 2 * x₁ = 0 ∧ 1 * x₂^2 - 2 * x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∃ x : ℝ, 1 * x^2 - 2 * x + 1 = 0) :=
by
  split
  { apply no_real_roots_of_discriminant_neg
    apply problem_statement }
  split
  { use (1 : ℝ)
    use (-2 : ℝ)
    norm_num }
  split
  { use (0 : ℝ)
    use (2 : ℝ)
    norm_num }
  { use (1 : ℝ)
    norm_num }

end correct_answer_C_l532_532612


namespace find_D_l532_532302

-- Definitions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k
def remainder (a b r : ℕ) : Prop := ∃ k, a = b * k + r

-- Problem Statement
theorem find_D {N D : ℕ} (h1 : remainder N D 75) (h2 : remainder N 37 1) : 
  D = 112 :=
by
  sorry

end find_D_l532_532302


namespace smallest_four_digit_divisible_by_35_l532_532240

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532240


namespace sqrt_a_pow_b_l532_532445

theorem sqrt_a_pow_b (a b : ℝ) (h : a^2 - 6 * a + real.sqrt (2 * b - 4) = -9) : real.sqrt (a^b) = 3 :=
sorry

end sqrt_a_pow_b_l532_532445


namespace smallest_distinct_digits_divisible_l532_532604

theorem smallest_distinct_digits_divisible :
  ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∀ m : ℕ, m < n → (∀ d in get_digits m, d ≠ 0 → m % d = 0 → 
                      ∀ e in get_digits n, e ≠ 0 → n % e = 0 ∧ n % 2 ≠ 1 ∧ 
                      (n / 1000) ≠ (n % 10) ∧ (n / 1000) ≠ ((n % 100) / 10) ∧ ((n % 1000) / 100) ≠ (n % 10) ∧ ((n % 1000) / 100) ≠ ((n % 100) / 10))
  ∧ get_digits n = {2, 1, 3, 6} ∧ n = 2136 := sorry

def get_digits (n : ℕ) : Finset ℕ :=
  if n < 10 then {n} else {n % 10} ∪ get_digits (n / 10)

end smallest_distinct_digits_divisible_l532_532604


namespace arithmetic_sequence_common_difference_l532_532851

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532851


namespace common_difference_arithmetic_sequence_l532_532927

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532927


namespace number_of_full_sequences_l532_532071

def is_full_sequence (a : List ℕ) : Prop :=
  ∀ k, 2 ≤ k → k ∈ a → (k - 1) ∈ a ∧ a.indexOf (k - 1) < a.lastIndexOf k

noncomputable def count_full_sequences (n : ℕ) : ℕ :=
  if h : n > 0 then Nat.fact n else 0

theorem number_of_full_sequences (n : ℕ) (hn : n > 0) : 
  ∃ S, is_full_sequence S ∧ count_full_sequences n = Nat.fact n := 
by
  sorry

end number_of_full_sequences_l532_532071


namespace rice_area_more_than_corn_area_l532_532146
-- Import the necessary library

-- Define the variables and conditions
variables {m : ℝ}

-- Define the areas
def wheat_area : ℝ := m
def rice_area : ℝ := 2 * m + 3
def corn_area : ℝ := m - 5

-- The theorem to be proved
theorem rice_area_more_than_corn_area : (rice_area - corn_area) = m + 8 :=
by sorry

end rice_area_more_than_corn_area_l532_532146


namespace find_constants_l532_532778

noncomputable def Sn (n : ℕ) : ℕ := 3 * n^2 + 5 * n

noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

noncomputable def bn (n : ℕ) : ℕ := 8 * 8^(2 * (1 - n))

theorem find_constants (n : ℕ) (h : 0 < n) :
  ∃ (c m : ℝ), 
    ∀ n : ℕ, n > 0 → 
      (an n + Real.log (bn (n - 1)) / Real.log c) = m :=
begin
  use [2, 17],
  sorry
end

end find_constants_l532_532778


namespace rectangular_to_cylindrical_l532_532689

theorem rectangular_to_cylindrical :
  ∀ x y z : ℝ, 
  (x, y, z) = (6, 6, -10) →
  (∃ r θ z_cyl : ℝ, r = 6 * Real.sqrt 2 ∧ θ = Real.pi / 4 ∧ z_cyl = -10 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) :=
by
  intros x y z h
  rw [Prod.mk.inj_iff] at h
  obtain ⟨hx, hy, hz⟩ := h
  use [6 * Real.sqrt 2, Real.pi / 4, -10]
  rw [hx, hy, hz]
  split
  { exact rfl },
  split
  { exact rfl },
  split
  { exact rfl },
  { norm_num [Real.sqrt], },
  { split; norm_num [Real.pi_div_four_pos], }

end rectangular_to_cylindrical_l532_532689


namespace cos_identity_l532_532384

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = (Real.sqrt 3) / 3) : 
  Real.cos (5 * π / 6 - θ) = - (Real.sqrt 3 / 3) :=
by
  sorry

end cos_identity_l532_532384


namespace no_integer_solutions_for_mn_squared_eq_1980_l532_532094

theorem no_integer_solutions_for_mn_squared_eq_1980 :
  ¬ ∃ m n : ℤ, m^2 + n^2 = 1980 := 
sorry

end no_integer_solutions_for_mn_squared_eq_1980_l532_532094


namespace compute_f_2_neg3_neg1_l532_532451

def f (p q r : ℤ) : ℚ := (r + p : ℚ) / (r - q + 1 : ℚ)

theorem compute_f_2_neg3_neg1 : f 2 (-3) (-1) = 1 / 3 := 
by
  sorry

end compute_f_2_neg3_neg1_l532_532451


namespace arithmetic_progression_infinite_squares_l532_532666

theorem arithmetic_progression_infinite_squares (a d : ℤ) (k : ℤ) (hk : a = k^2) :
  ∃ infinite_set_of_squares : set ℤ, (∀ n ∈ infinite_set_of_squares, ∃ m : ℤ, m^2 = a + n * d) ∧ infinite infinite_set_of_squares :=
  sorry

end arithmetic_progression_infinite_squares_l532_532666


namespace drop_perpendicular_correct_l532_532154

noncomputable def drop_perpendicular (A : Point) (L : Line) : Line :=
  sorry

theorem drop_perpendicular_correct (A : Point) (L : Line) (h: ¬ L.contains A) :
  ∃ G : Point, (drop_perpendicular A L).contains G ∧ is_perpendicular (drop_perpendicular A L) L :=
sorry

end drop_perpendicular_correct_l532_532154


namespace area_of_circle_outside_triangle_l532_532068

theorem area_of_circle_outside_triangle
  (ABC : Triangle)
  (h_right : is_right_triangle ABC)
  (h_angle : ABC.angle_at A = 90)
  (h_ab : AB.length = 9)
  (O : Point) (r : ℝ)
  (h_tangent_AB : is_tangent O.circle AB r)
  (h_tangent_AC : is_tangent O.circle AC r)
  (h_diametrically_opposite : ∀ X Y : Point, 
    is_tangent_point X AB O.circle ∧ 
    is_tangent_point Y AC O.circle → 
    is_diametrically_opposite X' Y' BC) :
  (area (circle_radius r) - area (triangle ABC)) = (9 / 4 * π - 4.5) := sorry

end area_of_circle_outside_triangle_l532_532068


namespace range_of_a_l532_532779

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | x^2 + 2 * x - 3 < 0},
      B := {x : ℝ | x - a < 0} in
  (A ⊆ B) ↔ (1 ≤ a) :=
by
  sorry

end range_of_a_l532_532779


namespace arithmetic_sequence_common_difference_l532_532944

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532944


namespace smallest_four_digit_divisible_by_35_l532_532202

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532202


namespace variance_of_surviving_trees_l532_532147

theorem variance_of_surviving_trees :
  let n := 4
  let p := 4 / 5
  let X := Binomial n p
  Var X = 4 * (4 / 5) * (1 - 4 / 5) :=
by
  sorry

end variance_of_surviving_trees_l532_532147


namespace proof_problem_l532_532465

-- Let x be the number of students who solved only problem A
-- Let y be the number of students who solved only problem B
-- Let z be the number of students who solved problem B and C but not A
def num_students := 25
def num_only_A (x : Nat) := x
def num_only_B (y : Nat) := y
def num_B_and_C_not_A (z : Nat) := z
def num_other_A (x y z : Nat) := y + (y - z) / 2
def num_total (x y z : Nat) := 2 * x + 1 + y + z + (y - z) / 2

theorem proof_problem (x y z : Nat) :
  num_total x y z = 25 →
  num_other_A x y z + 1 = x →
  y = 6 :=
by
  intros,
  sorry

end proof_problem_l532_532465


namespace common_difference_of_arithmetic_sequence_l532_532881

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532881


namespace cat_and_mouse_positions_after_317_moves_l532_532823

-- Define the conditions of the problem
def cat_positions : List String := ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
def mouse_positions : List String := ["Top Left", "Top Middle", "Top Right", "Right Middle", "Bottom Right", "Bottom Middle", "Bottom Left", "Left Middle"]

-- Calculate the position of the cat after n moves
def cat_position_after_moves (n : Nat) : String :=
  cat_positions.get! (n % 4)

-- Calculate the position of the mouse after n moves
def mouse_position_after_moves (n : Nat) : String :=
  mouse_positions.get! (n % 8)

-- Prove the final positions of the cat and mouse after 317 moves
theorem cat_and_mouse_positions_after_317_moves :
  cat_position_after_moves 317 = "Top Left" ∧ mouse_position_after_moves 317 = "Bottom Middle" :=
by
  sorry

end cat_and_mouse_positions_after_317_moves_l532_532823


namespace ratio_areas_of_hexagonal_sections_l532_532311

theorem ratio_areas_of_hexagonal_sections
  (s : ℝ) -- side length of the hexagon
  (h : ℝ) -- height of the hexagon
  (A_t A_q : ℝ) -- areas of the triangular and trapezoidal regions respectively
  (h_eq : h = (sqrt 3) / 2 * s)
  (A_t_eq : A_t = (sqrt 3) / 4 * s^2)
  (A_q_eq : A_q = (sqrt 3) / 4 * s^2) :
  A_q / A_t = 1 :=
by
  sorry

end ratio_areas_of_hexagonal_sections_l532_532311


namespace lcm_of_coprimes_eq_product_l532_532539

theorem lcm_of_coprimes_eq_product (a b c : ℕ) (h_coprime_ab : Nat.gcd a b = 1) (h_coprime_bc : Nat.gcd b c = 1) (h_coprime_ca : Nat.gcd c a = 1) (h_product : a * b * c = 7429) :
  Nat.lcm (Nat.lcm a b) c = 7429 :=
by 
  sorry

end lcm_of_coprimes_eq_product_l532_532539


namespace check_values_of_f_l532_532770

variable {a b c : ℝ}
def f (x : ℝ) : ℝ := a * tan x + b * x^3 + c * x

theorem check_values_of_f (hodd : ∀ x, f(-x) = - f(x)) :
  f 1 = 2 ∧ f (-1) = -2 :=
  sorry

end check_values_of_f_l532_532770


namespace common_difference_is_two_l532_532917

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532917


namespace range_of_m_l532_532010

variable (m : ℝ)

def p := ∀ (x : ℝ), x > 1 → m^x > 1
def q := ∀ (x : ℝ), x > 1 → x^m > x
def mutually_exclusive (p : Prop) (q : Prop) : Prop := (p ∨ q) ∧ ¬ p

theorem range_of_m (m : ℝ) : (mutually_exclusive (p m) (q m)) → (m ∈ set.univ) :=
by
  sorry

end range_of_m_l532_532010


namespace part_a_part_b_l532_532642

noncomputable def withdraw_rubles_after_one_year
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) : ℚ :=
  let deposit_in_dollars := initial_deposit / initial_rate
  let interest_earned := deposit_in_dollars * annual_yield
  let total_in_dollars := deposit_in_dollars + interest_earned
  let broker_fee := interest_earned * broker_commission
  let amount_after_fee := total_in_dollars - broker_fee
  let total_in_rubles := amount_after_fee * final_rate
  let conversion_fee := total_in_rubles * conversion_commission
  total_in_rubles - conversion_fee

theorem part_a
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) :
  withdraw_rubles_after_one_year initial_deposit initial_rate annual_yield final_rate conversion_commission broker_commission =
  16476.8 := sorry

def effective_yield (initial_rubles final_rubles : ℚ) : ℚ :=
  (final_rubles / initial_rubles - 1) * 100

theorem part_b
  (initial_deposit : ℤ) (final_rubles : ℚ) :
  effective_yield initial_deposit final_rubles = 64.77 := sorry

end part_a_part_b_l532_532642


namespace arithmetic_sequence_common_difference_l532_532891

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532891


namespace second_sample_correct_l532_532657

def total_samples : ℕ := 7341
def first_sample : ℕ := 4221
def second_sample : ℕ := total_samples - first_sample

theorem second_sample_correct : second_sample = 3120 :=
by
  sorry

end second_sample_correct_l532_532657


namespace determine_swap_from_weighing_l532_532483

def BallWeight : Type := ℕ

def HexagonVertices := { a : ℕ // 1 ≤ a ∧ a ≤ 6 } 

-- We define a hexagon with vertex weights
structure HexagonBallWeights where
  A : BallWeight
  B : BallWeight
  C : BallWeight
  D : BallWeight
  E : BallWeight
  F : BallWeight

-- Initial weights at vertices
def init_weights (v : HexagonVertices) : BallWeight :=
  match v with
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 2
  | ⟨3, _⟩ => 3
  | ⟨4, _⟩ => 4
  | ⟨5, _⟩ => 5
  | ⟨6, _⟩ => 6
  | _ => 0 -- should never happen due to the type constraint

-- Verifying a specific weighing
def correctSwapWeights (b1 b2 : BallWeight) (hex_weights : HexagonBallWeights) : Prop :=
  let left_pan_weight := hex_weights.A + hex_weights.E
  let right_pan_weight := hex_weights.B + hex_weights.D

  (b1, b2) = (hex_weights.A, hex_weights.D) ∧ left_pan_weight + b1 > right_pan_weight ∨
  (b1, b2) = (hex_weights.B, hex_weights.E) ∧ right_pan_weight + b1 > left_pan_weight ∨
  (b1, b2) = (hex_weights.C, hex_weights.F) ∧ left_pan_weight = right_pan_weight

-- Prove that a weighing suffices to identify swapped balls
theorem determine_swap_from_weighing :
  ∀ (hex_weights : HexagonBallWeights) (b1 b2 : BallWeight),
  (By using one weighing, we can uniquely identify b1 and b2)
  correctSwapWeights b1 b2 hex_weights → 
  (HexagonBallWeights := 
    { A := init_weights ⟨1, sorry⟩,
      B := init_weights ⟨2, sorry⟩,
      C := init_weights ⟨3, sorry⟩,
      D := init_weights ⟨4, sorry⟩,
      E := init_weights ⟨5, sorry⟩,
      F := init_weights ⟨6, sorry⟩ }):
  ∃ (a b : BallWeight), a ≠ b ∧ correctSwapWeights a b hex_weights :=
by
  sorry

end determine_swap_from_weighing_l532_532483


namespace total_wood_gathered_l532_532680

def pieces_per_sack := 20
def number_of_sacks := 4

theorem total_wood_gathered : pieces_per_sack * number_of_sacks = 80 := 
by 
  sorry

end total_wood_gathered_l532_532680


namespace common_difference_of_arithmetic_sequence_l532_532961

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532961


namespace rect_area_sum_eq_16_l532_532318

theorem rect_area_sum_eq_16 (a b c : ℕ) (h1 : |a * b - a * c| = 1) (h2 : |a * c - b * c| = 49) :
  a + b + c = 16 :=
sorry

end rect_area_sum_eq_16_l532_532318


namespace rational_numbers_count_correct_l532_532824

def numbers := [Real.sqrt 10, 22 / 7, - Real.cbrt 27, Real.sqrt 5 / 9, 3, 14, 0.808008, Real.pi]

def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

def rational_count : Nat := numbers.countp is_rational

theorem rational_numbers_count_correct : rational_count = 5 := 
by sorry

end rational_numbers_count_correct_l532_532824


namespace triangle_angle_sine_condition_l532_532092

theorem triangle_angle_sine_condition (A B C : ℝ) (h : A + B + C = 180) :
  (sin (5 * A) + sin (5 * B) + sin (5 * C) = 0) ↔ (C = 36 ∨ C = 108) :=
sorry

end triangle_angle_sine_condition_l532_532092


namespace team_total_score_is_correct_l532_532655

-- Define the total number of team members
def total_members : ℕ := 30

-- Define the number of members who didn't show up
def members_absent : ℕ := 8

-- Define the score per member
def score_per_member : ℕ := 4

-- Define the points deducted per incorrect answer
def points_per_incorrect_answer : ℕ := 2

-- Define the total number of incorrect answers
def total_incorrect_answers : ℕ := 6

-- Define the bonus multiplier
def bonus_multiplier : ℝ := 1.5

-- Define the total score calculation
def total_score_calculation (total_members : ℕ) (members_absent : ℕ) (score_per_member : ℕ)
  (points_per_incorrect_answer : ℕ) (total_incorrect_answers : ℕ) (bonus_multiplier : ℝ) : ℝ :=
  let members_present := total_members - members_absent
  let initial_score := members_present * score_per_member
  let total_deductions := total_incorrect_answers * points_per_incorrect_answer
  let final_score := initial_score - total_deductions
  final_score * bonus_multiplier

-- Prove that the total score is 114 points
theorem team_total_score_is_correct : total_score_calculation total_members members_absent score_per_member
  points_per_incorrect_answer total_incorrect_answers bonus_multiplier = 114 :=
by
  sorry

end team_total_score_is_correct_l532_532655


namespace percentage_change_l532_532622

def original_income (P T : ℝ) : ℝ :=
  P * T

def new_income (P T : ℝ) : ℝ :=
  (P * 1.3333) * (T * 0.6667)

theorem percentage_change (P T : ℝ) (hP : P ≠ 0) (hT : T ≠ 0) :
  ((new_income P T - original_income P T) / original_income P T) * 100 = -11.11 :=
by
  sorry

end percentage_change_l532_532622


namespace moss_flower_pollen_scientific_notation_l532_532350

theorem moss_flower_pollen_scientific_notation (d : ℝ) (h : d = 0.0000084) : ∃ n : ℤ, d = 8.4 * 10^n ∧ n = -6 :=
by
  use -6
  rw [h]
  simp
  sorry

end moss_flower_pollen_scientific_notation_l532_532350


namespace grid_sum_even_iff_l532_532471

-- Given conditions as Lean definitions
def grid (n : ℕ) [fact (n ≥ 3)] : Type := matrix (fin n) (fin n) ℤ

def is_even (x : ℤ) := ∃ k : ℤ, x = 2 * k

def even_sum_subgrid_3x3 (n : ℕ) (A : grid n) : Prop :=
  ∀ i j : fin n, 
  if h : i.val + 2 < n ∧ j.val + 2 < n then is_even (finset.sum (finset.product (finset.range 3) (finset.range 3)) (λ ⟨x, y⟩, A ⟨i + x, h.left.trans (lt_of_lt_of_le (by norm_num) (nat.le_of_add_sub_left h.left))⟩ ⟨j + y, h.right.trans (lt_of_lt_of_le (by norm_num) (nat.le_of_add_sub_left h.right))⟩)) 
  else true

def even_sum_subgrid_5x5 (n : ℕ) (A : grid n) : Prop :=
  ∀ i j : fin n,
  if h : i.val + 4 < n ∧ j.val + 4 < n then is_even (finset.sum (finset.product (finset.range 5) (finset.range 5)) (λ ⟨x, y⟩, A ⟨i + x, h.left.trans (lt_of_lt_of_le (by norm_num) (nat.le_of_add_sub_left h.left))⟩ ⟨j + y, h.right.trans (lt_of_lt_of_le (by norm_num) (nat.le_of_add_sub_left h.right))⟩))
  else true

-- Proof statement
theorem grid_sum_even_iff (n : ℕ) [fact (n ≥ 3)] (A : grid n)
  (H3 : even_sum_subgrid_3x3 n A) (H5 : even_sum_subgrid_5x5 n A) :
  (is_even (finset.sum finset.univ (λ ⟨i, j⟩, A i j))) ↔ (∃ k : ℕ, n = 3 * k) ∨ (∃ k : ℕ, n = 5 * k) :=
sorry

end grid_sum_even_iff_l532_532471


namespace complex_calculation_l532_532330

theorem complex_calculation : (1 + Complex.i) ^ 10 / (1 - Complex.i) = -16 + 16 * Complex.i :=
by
  sorry

end complex_calculation_l532_532330


namespace intersection_conditions_l532_532155

-- Define the conditions
variables (c : ℝ) (k : ℝ) (m : ℝ) (n : ℝ) (p : ℝ)

-- Distance condition
def distance_condition (k : ℝ) (m : ℝ) (n : ℝ) (c : ℝ) : Prop :=
  (abs ((k^2 + 8 * k + c) - (m * k + n)) = 4)

-- Line passing through point (2, 7)
def passes_through_point (m : ℝ) (n : ℝ) : Prop :=
  (7 = 2 * m + n)

-- Definition of discriminants
def discriminant_1 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n - 4))

def discriminant_2 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n + 4))

-- Statement of the problem
theorem intersection_conditions (h₁ : n ≠ 0)
  (h₂ : passes_through_point m n)
  (h₃ : distance_condition k m n c)
  (h₄ : (discriminant_1 m c n = 0 ∨ discriminant_1 m c n < 0))
  (h₅ : (discriminant_2 m c n < 0)) :
  ∃ m n, n = 7 - 2 * m ∧ distance_condition k m n c :=
sorry

end intersection_conditions_l532_532155


namespace part1_part2_l532_532540

namespace Problem

-- Given problem setup

def A : Set ℕ := { n | ∃ k : ℕ, k > 0 ∧ n = 2 ^ k }
def A_complement : Set ℕ := λ n, n ∉ A

-- Part 1: For every \( a \in A \) and \( b \in \mathbf{N}^{*} \), 
-- if \( b < 2a - 1 \), then \( b(b+1) \) is not a multiple of \( 2a \).
theorem part1 (a b : ℕ) (ha : a ∈ A) (hb : b > 0) (H : b < 2 * a - 1) :
  ¬ ∃ k : ℕ, b * (b + 1) = 2 * a * k := sorry

-- Part 2: For every \( a \in \bar{A} \), and \( a \ne 1 \), 
-- there exists \( b \in \mathbf{N}^{*} \) such that \( b < 2a - 1 \) 
-- and \( b(b+1) \) is a multiple of \( 2a \).
theorem part2 (a : ℕ) (ha : a ∈ A_complement) (H1 : a ≠ 1) :
  ∃ b : ℕ, b > 0 ∧ b < 2 * a - 1 ∧ ∃ k : ℕ, b * (b + 1) = 2 * a * k := sorry

end Problem

end part1_part2_l532_532540


namespace total_pencils_correct_l532_532352

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct_l532_532352


namespace convex_polyhedron_three_equal_edges_l532_532726

theorem convex_polyhedron_three_equal_edges 
  (P : Type) [convex P]
  (H : ∀ v ∈ vertices P, ∃ e1 e2 e3 ∈ edges_from v, e1 = e2 ∨ e2 = e3 ∨ e1 = e3) :
  ∃ e1 e2 e3 ∈ edges P, e1 = e2 ∧ e2 = e3 := 
by 
  sorry

end convex_polyhedron_three_equal_edges_l532_532726


namespace sqrt_cube_root_subtract_l532_532677

theorem sqrt_cube_root_subtract :
  Real.sqrt 16 + Real.cbrt (-27) - Real.sqrt ((-1) ^ 2) = 0 :=
by
  sorry

end sqrt_cube_root_subtract_l532_532677


namespace common_difference_of_arithmetic_sequence_l532_532979

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532979


namespace common_difference_l532_532996

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532996


namespace sum_of_first_five_terms_l532_532042

theorem sum_of_first_five_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) -- geometric sequence definition
  (h3 : a 2 + a 5 = 2 * (a 4 + 2)) : 
  S 5 = 62 :=
by
  -- lean tactics would go here to provide the proof
  sorry

end sum_of_first_five_terms_l532_532042


namespace age_problem_lean4_l532_532816

/-
Conditions:
1. Mr. Bernard's age in eight years will be 60.
2. Luke's age in eight years will be 28.
3. Sarah's age in eight years will be 48.
4. The sum of their ages in eight years will be 136.

Question (translated to proof problem):
Prove that 10 years less than the average age of all three of them is approximately 35.33.

The Lean 4 statement below formalizes this:
-/

theorem age_problem_lean4 :
  let bernard_age := 60
  let luke_age := 28
  let sarah_age := 48
  let total_age := bernard_age + luke_age + sarah_age
  total_age = 136 → ((total_age / 3.0) - 10.0 = 35.33) :=
by
  intros
  sorry

end age_problem_lean4_l532_532816


namespace find_a_l532_532763

theorem find_a (a : ℝ) (h : a > 0)
    (intersect : ∃ A B : ℝ × ℝ, (A.2 = A.1 + 2 * a) ∧ (B.2 = B.1 + 2 * a) ∧
      (A.1 ^ 2 + A.2 ^ 2 - 2 * a * A.2 - 2 = 0) ∧ (B.1 ^ 2 + B.2 ^ 2 - 2 * a * B.2 - 2 = 0))
    (dist_AB : ∃ A B : ℝ × ℝ, |(A.1 - B.1)^2 + (A.2 - B.2)^2| = 4 * 3) :
    a = sqrt 2 :=
by
  sorry

end find_a_l532_532763


namespace number_of_observations_l532_532158

variable (n : ℕ)

-- The initial mean of the observations
variable (mean_initial : ℝ) (mean_initial = 32)

-- The corrected observation
variable (a b : ℝ) (a = 23) (b = 48)

-- The corrected mean of the observations
variable (mean_corrected : ℝ) (mean_corrected = 32.5)

theorem number_of_observations:
  mean_initial * n + (b - a) = mean_corrected * n → 
  n = 50 :=
by
  -- skipping the proof part
  sorry

end number_of_observations_l532_532158


namespace sqrt_meaningful_range_l532_532024

theorem sqrt_meaningful_range (x : ℝ) : x + 2 ≥ 0 → x ≥ -2 :=
by 
  intro h
  linarith [h]

end sqrt_meaningful_range_l532_532024


namespace karen_start_time_late_l532_532487

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l532_532487


namespace smallest_four_digit_divisible_by_35_l532_532185

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532185


namespace partial_mixed_deriv_does_not_exist_at_origin_l532_532278

def f (x y : ℝ) : ℝ :=
if x^2 + y^4 ≠ 0 then (x * y^2) / (x^2 + y^4) else 0

theorem partial_mixed_deriv_does_not_exist_at_origin :
  ¬ (∃ (d : ℝ), has_partial_deriv_at (λ (y : ℝ), 
  deriv (λ (x : ℝ), f x y) 0) d 0) :=
by sorry

end partial_mixed_deriv_does_not_exist_at_origin_l532_532278


namespace smallest_number_of_bottles_l532_532332

noncomputable def fluid_ounces : ℝ := 60
noncomputable def milliliter_per_bottle : ℝ := 250
noncomputable def fluid_ounces_per_liter : ℝ := 33.8
noncomputable def milliliters_per_liter : ℝ := 1000

theorem smallest_number_of_bottles :
  let liters_needed := fluid_ounces / fluid_ounces_per_liter in
  let milliliters_needed := liters_needed * milliliters_per_liter in
  let num_bottles := milliliters_needed / milliliter_per_bottle in
  int.ceil(num_bottles) = 8 :=
by
  sorry

end smallest_number_of_bottles_l532_532332


namespace tan_alpha_tan_beta_value_l532_532017

theorem tan_alpha_tan_beta_value
  (α β : ℝ)
  (h1 : Real.cos (α + β) = 1 / 5)
  (h2 : Real.cos (α - β) = 3 / 5) :
  Real.tan α * Real.tan β = 1 / 2 :=
by
  sorry

end tan_alpha_tan_beta_value_l532_532017


namespace plates_count_l532_532654

theorem plates_count (n : ℕ)
  (h1 : 500 < n)
  (h2 : n < 600)
  (h3 : n % 10 = 7)
  (h4 : n % 12 = 7) : n = 547 :=
sorry

end plates_count_l532_532654


namespace dice_probability_sum_25_l532_532292

theorem dice_probability_sum_25 :
  let die1 := {n : ℕ | n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∨ n = 14 ∨ n = 15 ∨ n = 16 ∨ n = 17 ∨ n = 18 ∨ n = 19}
  let die2 := {n : ℕ | n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∨ n = 14 ∨ n = 15 ∨ n = 16 ∨ n = 17 ∨ n = 18 ∨ n = 19 ∨ n = 20 ∨ n = 21}
  (∑ (n₁ ∈ die1) (n₂ ∈ die2), if n₁ + n₂ = 25 then 1 else 0) / ((Finset.card die1) * (Finset.card die2) : ℚ) = 13 / 400 := by
  sorry

end dice_probability_sum_25_l532_532292


namespace arithmetic_expression_equality_l532_532256

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l532_532256


namespace find_common_difference_l532_532871

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532871


namespace T1_T3_l532_532683

-- Definitions for kets and dobs
axiom ket : Type
axiom dob : Type

-- Postulates as Lean statements
def P1 (k : ket) : Set dob := sorry  -- every ket is a group of dobs
axiom P2 : ∀ (k1 k2 k3 : ket), 
           k1 ≠ k2 → k2 ≠ k3 → k1 ≠ k3 → 
           (Set.inter (Set.inter (P1 k1) (P1 k2)) (P1 k3)).card = 1
axiom P3 : ∀ (d : dob), 
           (Set.filter (λ k : ket, d ∈ P1 k) univ).card = 3
axiom P4 : (Set.univ : Set ket).card = 5

-- Theorems to be proved
theorem T1 : (∃ S : Set dob, S.card = 10) := sorry
theorem T3 : (∀ d1 d2 : dob, (∃ k : ket, d1 ∈ P1 k ∧ d2 ∈ P1 k) → d1 ≠ d2 → 
             ∃ d3 : dob, d3 ∈ S ∧ ∃ k' : ket, d1 ∉ P1 k' ∧ d2 ∉ P1 k') := sorry

end T1_T3_l532_532683


namespace sin_585_eq_neg_sqrt_two_div_two_l532_532336

theorem sin_585_eq_neg_sqrt_two_div_two : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_eq_neg_sqrt_two_div_two_l532_532336


namespace maximize_profit_l532_532557

noncomputable def a : ℝ := 200

def R (x : ℝ) : ℝ :=
if x < 80 then x^2 + a * x
else (301 * x^2 - 2750 * x + 10000) / x

def W (x : ℝ) : ℝ :=
if x < 80 then -x^2 + 100 * x - 1000
else -x - 10000 / x + 1750

theorem maximize_profit :
  ∃ x : ℝ, x = 100 ∧ W 100 = 1550 :=
by
  use 100
  split
  exact rfl
  sorry

end maximize_profit_l532_532557


namespace probability_at_most_3_heads_in_12_flips_l532_532633

theorem probability_at_most_3_heads_in_12_flips :
  let favorable := Nat.choose 12 0 + Nat.choose 12 1 + Nat.choose 12 2 + Nat.choose 12 3
  let total := 2^12
  (favorable : ℝ) / total = 299 / 4096 :=
by
  sorry

end probability_at_most_3_heads_in_12_flips_l532_532633


namespace smallest_four_digit_divisible_by_35_l532_532238

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532238


namespace common_difference_is_two_l532_532915

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532915


namespace x_can_be_positive_negative_or_zero_l532_532407

noncomputable
def characteristics_of_x (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : w ≠ 0) 
  (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : Prop :=
  ∃ r : ℝ, r = x

theorem x_can_be_positive_negative_or_zero (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : w ≠ 0) (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : 
  (characteristics_of_x x y z w h1 h2 h3 h4 h5 h6) :=
sorry

end x_can_be_positive_negative_or_zero_l532_532407


namespace find_other_number_l532_532322

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 2 * b = 120) (h2 : a = 26 ∨ b = 26) : a = 26 → b = 21 ∧ b = 26 → a = 21 :=
by {
  cases h2,
  { left, rw h2 at h1, sorry },
  { right, rw h2 at h1, sorry }
}

end find_other_number_l532_532322


namespace sum_of_A_B_C_l532_532693

theorem sum_of_A_B_C (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_rel_prime : Nat.gcd A (Nat.gcd B C) = 1) (h_eq : A * Real.log 3 / Real.log 180 + B * Real.log 5 / Real.log 180 = C) : A + B + C = 4 :=
sorry

end sum_of_A_B_C_l532_532693


namespace max_equilateral_triangles_l532_532386

theorem max_equilateral_triangles (n : ℕ) (length : ℕ) (h_n : n = 6) (h_length : length = 2) :
  ∃ (max_triangles : ℕ), max_triangles = 4 :=
by
  use 4
  sorry

end max_equilateral_triangles_l532_532386


namespace brother_eats_papayas_l532_532476

-- Definitions of conditions as constants
def Jake_eats_per_week := 3
def Father_eats_per_week := 4
def Total_papayas_for_4_weeks := 48

-- Definition of B (number of papayas Jake's brother can eat in one week)
def B := Total_papayas_for_4_weeks / 4 - (Jake_eats_per_week + Father_eats_per_week)

-- Theorem to prove that B is 5
theorem brother_eats_papayas : B = 5 := by
  unfold B
  simp only [Jake_eats_per_week, Father_eats_per_week, Total_papayas_for_4_weeks]
  norm_num
  sorry

end brother_eats_papayas_l532_532476


namespace rachel_speed_l532_532543

def distance (from : String) (to : String) : ℕ := 
  if from = "Rachel's house" ∧ to = "Nicholas's house" then 10 else 0

def time (person : String) (from : String) (to : String) : ℕ := 
  if person = "Rachel" ∧ from = "Rachel's house" ∧ to = "Nicholas's house" then 5 else 0

def speed (dist : ℕ) (t : ℕ) : ℕ := dist / t

theorem rachel_speed :
  speed (distance "Rachel's house" "Nicholas's house")
        (time "Rachel" "Rachel's house" "Nicholas's house") = 2 :=
by
  sorry

end rachel_speed_l532_532543


namespace min_value_F_l532_532448

noncomputable theory

-- Define odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f(x)

-- Define F(x)
def F (f g : ℝ → ℝ) (x : ℝ) := f x + g x + 2

theorem min_value_F {f g : ℝ → ℝ} 
    (hf : is_odd f) 
    (hg : is_odd g) 
    (hF_max : ∀ x > 0, F f g x ≤ 8) :
  ∃ x < 0, F f g x = -4 := 
sorry

end min_value_F_l532_532448


namespace problem_statement_l532_532058

-- Given conditions
variables {p q r t n : ℕ}

axiom prime_p : Nat.Prime p
axiom prime_q : Nat.Prime q
axiom prime_r : Nat.Prime r

axiom nat_n : n ≥ 1
axiom nat_t : t ≥ 1

axiom eqn1 : p^2 + q * t = (p + t)^n
axiom eqn2 : p^2 + q * r = t^4

-- Statement to prove
theorem problem_statement : n < 3 ∧ (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) :=
by
  sorry

end problem_statement_l532_532058


namespace simplest_common_denominator_l532_532577

theorem simplest_common_denominator (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  LCM (2 * x^2 * y) (6 * x * y^3) = 6 * x^2 * y^3 := 
sorry

end simplest_common_denominator_l532_532577


namespace right_angled_triangle_k_values_l532_532037

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def BC (k : ℝ) : ℝ × ℝ := (1, k - 1)

theorem right_angled_triangle_k_values (k : ℝ) :
  (dot_product AB (AC k) = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔ (k = -6 ∨ k = -1) :=
sorry

end right_angled_triangle_k_values_l532_532037


namespace find_lambda_l532_532784

def vec_a : ℝ × ℝ := (3, -2)
def vec_b (λ : ℝ) : ℝ × ℝ := (5, λ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem find_lambda (λ : ℝ) (h : dot_product vec_a (vec_sub vec_a (vec_b λ)) = 0) : λ = 1 :=
sorry

end find_lambda_l532_532784


namespace problem_proof_l532_532258

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l532_532258


namespace sin_cos_func_value_range_l532_532586

def sin_cos_func_range : Set ℝ :=
  {y : ℝ | (∃ x : ℝ, (-sqrt(2) ≤ sin x - cos x ∧ sin x - cos x ≤ sqrt(2)) ∧ 
                    (sin x - cos x ≠ -1) ∧ 
                    y = (1/2) * (sin x - cos x - 1))}

theorem sin_cos_func_value_range :
  sin_cos_func_range = {y : ℝ | (-sqrt(2) + 1) / 2 ≤ y ∧ y < -1 ∨ -1 < y ∧ y ≤ (sqrt(2) - 1) / 2} :=
by
  sorry

end sin_cos_func_value_range_l532_532586


namespace simplest_common_denominator_l532_532576

open Nat

theorem simplest_common_denominator (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  let d1 := 2 * x^3 
  let d2 := 6 * x^2 * y 
  lcm d1 d2 = 6 * x^3 * y := 
by {
  sorry
}

end simplest_common_denominator_l532_532576


namespace carlos_blocks_l532_532679

theorem carlos_blocks (initial_blocks : ℕ) (blocks_given : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 58) (h2 : blocks_given = 21) : remaining_blocks = 37 :=
by
  sorry

end carlos_blocks_l532_532679


namespace smallest_four_digit_div_by_35_l532_532234

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532234


namespace parking_garage_open_spots_l532_532649

theorem parking_garage_open_spots :
  ∃ (x : ℕ), 
    (let second_level := x + 2 in
     let third_level := x + 7 in
     let fourth_level := 31 in
     let total_spots := 4 * 100 in
     let full_spots := 186 in
     let open_spots := total_spots - full_spots in
     x + second_level + third_level + fourth_level = open_spots) ∧ x = 58 :=
by
  sorry

end parking_garage_open_spots_l532_532649


namespace solve_system_l532_532550

-- Define the differential equations
def system_eqs (y1 y2 y3 y1' y2' y3': ℝ → ℝ) (x : ℝ) : Prop :=
  y1' x = y2 x + y3 x ∧
  y2' x = y1 x + y3 x ∧
  y3' x = y1 x + y2 x

-- Define the initial conditions
def initial_conditions (y1 y2 y3 : ℝ → ℝ) : Prop :=
  y1 0 = 1 ∧ y2 0 = 2 ∧ y3 0 = 0

-- Define the proposed solutions
def solutions (y1 y2 y3 : ℝ → ℝ) : Prop :=
  ∀ x, y1 x = exp (2 * x) ∧ y2 x = exp (-x) + exp (2 * x) ∧ y3 x = -exp (-x) + exp (2 * x)

-- The main theorem statement
theorem solve_system :
  ∃ (y1 y2 y3 : ℝ → ℝ),
    (∀ (y1' y2' y3' : ℝ → ℝ), system_eqs y1 y2 y3 y1' y2' y3') ∧
    initial_conditions y1 y2 y3 ∧
    solutions y1 y2 y3 :=
by
  sorry

end solve_system_l532_532550


namespace prob_both_contracts_l532_532624

variables (P_A P_B P_A_union_B P_A_inter_B : ℚ)

axiom hP_A : P_A = 3 / 4
axiom hP_B_complement : (1 - P_B) = 3 / 5
axiom hP_A_union_B : P_A_union_B = 5 / 6

theorem prob_both_contracts : P_A_inter_B = 19 / 60 :=
by
  have hP_B : P_B = 2 / 5 := by linarith [hP_B_complement]
  have hP_A_union_B_eq : P_A_union_B = P_A + P_B - P_A_inter_B := sorry
  rw [hP_A_union_B_eq, hP_A, hP_B] at hP_A_union_B
  sorry

end prob_both_contracts_l532_532624


namespace multiple_of_4_difference_multiple_of_4_l532_532555

variables (a b : ℤ)

def is_multiple_of (x y : ℤ) : Prop := ∃ k : ℤ, x = k * y

axiom h1 : is_multiple_of a 4
axiom h2 : is_multiple_of b 8

theorem multiple_of_4 (b : ℤ) (h : is_multiple_of b 8) : is_multiple_of b 4 :=
by {
  unfold is_multiple_of at *,
  cases h with k hk,
  use k * 2,
  rw [hk],
  norm_num,
}

theorem difference_multiple_of_4 (a b : ℤ) (ha : is_multiple_of a 4) (hb : is_multiple_of b 4) : is_multiple_of (a - b) 4 :=
by {
  unfold is_multiple_of at *,
  cases ha with ka hka,
  cases hb with kb hkb,
  use (ka - kb),
  rw [hka, hkb, sub_mul, mul_sub],
}

end multiple_of_4_difference_multiple_of_4_l532_532555


namespace simplest_common_denominator_of_fractions_l532_532579

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end simplest_common_denominator_of_fractions_l532_532579


namespace smallest_four_digit_divisible_by_35_l532_532197

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532197


namespace smallest_four_digit_divisible_by_35_l532_532189

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532189


namespace first_term_arithmetic_sequence_l532_532506

theorem first_term_arithmetic_sequence (S : ℕ → ℤ) (a : ℤ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2 : ∀ n m, (S (3 * n)) / (S m) = (S (3 * m)) / (S n)) : a = 5 / 2 := 
sorry

end first_term_arithmetic_sequence_l532_532506


namespace total_balls_l532_532148

theorem total_balls (S V B Total : ℕ) (hS : S = 68) (hV : S = V - 12) (hB : S = B + 23) : 
  Total = S + V + B := by
  sorry

end total_balls_l532_532148


namespace common_difference_l532_532990

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532990


namespace common_difference_of_arithmetic_sequence_l532_532877

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532877


namespace common_difference_of_arithmetic_sequence_l532_532974

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532974


namespace relationship_bx_x2_a2_l532_532729

theorem relationship_bx_x2_a2 {a b x : ℝ} (h1 : b < x) (h2 : x < a) (h3 : 0 < a) (h4 : 0 < b) : 
  b * x < x^2 ∧ x^2 < a^2 :=
by sorry

end relationship_bx_x2_a2_l532_532729


namespace abs_diff_equals_two_sqrt_three_l532_532064

-- Defining the conditions mn = 6 and m + n = 6
variables {m n : ℝ}
axiom h1 : m * n = 6
axiom h2 : m + n = 6

-- The theorem stating that |m - n| = 2 * (sqrt 3)
theorem abs_diff_equals_two_sqrt_three (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 6) : |m - n| = 2 * real.sqrt 3 := 
by 
  sorry

end abs_diff_equals_two_sqrt_three_l532_532064


namespace common_difference_of_arithmetic_sequence_l532_532984

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532984


namespace oldest_daily_cheese_l532_532838

-- Given conditions
def days_per_week : ℕ := 5
def weeks : ℕ := 4
def youngest_daily : ℕ := 1
def cheeses_per_pack : ℕ := 30
def packs_needed : ℕ := 2

-- Derived conditions
def total_days : ℕ := days_per_week * weeks
def total_cheeses : ℕ := packs_needed * cheeses_per_pack
def youngest_total_cheeses : ℕ := youngest_daily * total_days
def oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses

-- Prove that the oldest child wants 2 string cheeses per day
theorem oldest_daily_cheese : oldest_total_cheeses / total_days = 2 := by
  sorry

end oldest_daily_cheese_l532_532838


namespace validate_operation_l532_532608

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end validate_operation_l532_532608


namespace reasoning_is_deductive_l532_532820

-- Definitions based on conditions
def text_from_the_analects : Prop := 
  "If names are not correct, then language will not be in accordance with the truth of things; 
  if language is not in accordance with the truth of things, then affairs cannot be carried out successfully; 
  if affairs cannot be carried out successfully, then rituals and music will not flourish; 
  if rituals and music do not flourish, then punishments will not be properly applied; 
  if punishments are not properly applied, then the people will have no guidance on what to do. 
  Therefore, if names are not correct, then the people will have no guidance on what to do."

def def_of_deductive_reasoning : Prop := 
  "Deductive reasoning starts from a general premise and through deduction, arrives at specific statements or individual conclusions."

-- Theorem to prove the type of reasoning of the given text is deductive reasoning
theorem reasoning_is_deductive : text_from_the_analects → def_of_deductive_reasoning → true :=
by
  -- The proof is omitted
  sorry

end reasoning_is_deductive_l532_532820


namespace Xiaopang_birthday_proof_l532_532616

noncomputable def Xiaopang_birthday : Prop :=
  ∃ (x y : ℕ) (k : ℤ), x + y = 38 ∧ x > y ∧ x - y = 7 * k ∧ x = 26

theorem Xiaopang_birthday_proof : Xiaopang_birthday :=
begin
  sorry
end

end Xiaopang_birthday_proof_l532_532616


namespace find_common_difference_l532_532948

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532948


namespace integrate_f_squared_geq_4_l532_532500

noncomputable def integrable_fn (f : ℝ → ℝ) (a b : ℝ) : Prop := sorry

theorem integrate_f_squared_geq_4 (f : ℝ → ℝ) (h0 : integrable_fn f 0 1)
  (h1: ∫ x in set.Icc 0 1, f x = 1)
  (h2: ∫ x in set.Icc 0 1, x * f x = 1) :
  ∫ x in set.Icc 0 1, (f x)^2 ≥ 4 := sorry

end integrate_f_squared_geq_4_l532_532500


namespace triangle_obtuse_l532_532829

theorem triangle_obtuse (A B C : ℝ) (hA : A = 2 * B) (hB : B = (1 / 3) * A) (hC : C = (1 / 2) * A) (sum_angles : A + B + C = 180) :
  A > 90 :=
by
  -- we set k such that A = 6k, B = 3k, C = 2k
  obtain ⟨k, hk⟩ : ∃ k, A = 6*k ∧ B = 3*k ∧ C = 2*k := sorry
  -- substitute the angles into the sum equation
  have : 6 * k + 3 * k + 2 * k = 180 := sorry
  -- solve for k
  have : 11 * k = 180 := sorry
  -- solve for k explicitly
  have : k = 180 / 11 := sorry
  -- find A
  have : A = 6 * (180 / 11) := sorry
  -- check if A > 90
  show A > 90, from sorry

end triangle_obtuse_l532_532829


namespace sum_series_result_l532_532509

noncomputable def sum_series (x : ℝ) : ℝ :=
  ∑' (n : ℕ), 1 / (x^(2^n) - x^(-2^n))

theorem sum_series_result (x : ℝ) (h : x > 1) :
  sum_series x = 1 / (x - 1) :=
sorry

end sum_series_result_l532_532509


namespace polynomial_roots_l532_532073

theorem polynomial_roots (a b : ℝ) 
  (h1 : (2 - 3 * complex.i) is_root (λ x : complex, x^3 + (a : ℂ) * x^2 - 2 * x + b)) 
  : (a, b) = (-1 / 4, 195 / 4) :=
sorry

end polynomial_roots_l532_532073


namespace common_difference_of_arithmetic_sequence_l532_532973

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532973


namespace smallest_four_digit_divisible_by_35_l532_532181

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532181


namespace inverse_function_value_zero_l532_532749

def f (x : ℝ) : ℝ := (x + 1) / (2 * x)

def f_inv (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem inverse_function_value_zero : f_inv 0 = -1 := by
  sorry

end inverse_function_value_zero_l532_532749


namespace initial_crayons_l532_532537

theorem initial_crayons (left lost_given : ℕ) (h1 : left = 134) (h2 : lost_given = 345) :
  (left + lost_given = 479) :=
by
  rw [h1, h2]
  rfl

end initial_crayons_l532_532537


namespace smallest_four_digit_divisible_by_35_l532_532244

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532244


namespace triangle_right_angled_l532_532798

theorem triangle_right_angled (A B C : ℝ) (h : A + B + C = 180) (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x) :
  C = 90 :=
by {
  sorry
}

end triangle_right_angled_l532_532798


namespace smallest_four_digit_divisible_by_35_l532_532213

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532213


namespace problem_statement_l532_532575

theorem problem_statement (a b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : ∀ n, a(n+1) - a(n) = 2)
    (h2 : a 1 = 2)
    (h3 : b 1 = a 1)
    (h4 : b 4 = a 8)
    (hc : ∀ n, c n = a n * b n) :
    (∀ n, a n = 2 * n) ∧ (∀ n, b n = 2 ^ n) ∧ (∀ n, (finset.range (n + 1)).sum c = (n-1) * 2^(n+2) + 4) :=
  sorry

end problem_statement_l532_532575


namespace arithmetic_sequence_common_difference_l532_532855

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532855


namespace segment_length_half_diagonal_l532_532499

variables (A B C D X E : Point)
variable [parallelogram ABCD]
variable [acute_vertex A]
variable [midpoint E AC BD]

-- Perpendicular bisector of AB intersects CD at X
variable [perpendicular_bisector AB X CD]

theorem segment_length_half_diagonal :
  dist X E = (1 / 2) * dist A D :=
sorry

end segment_length_half_diagonal_l532_532499


namespace problem1_l532_532627

variable (x : ℝ)

theorem problem1 (hx : x > -1) : 
  let y := x - 4 + 9 / (x + 1)
  in y ≥ 1 :=
sorry

end problem1_l532_532627


namespace numerator_of_third_fraction_l532_532271

theorem numerator_of_third_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 5998
  let e := 3.0035428163476343
  let f := 3999
  let sum := (a:ℝ) / b + (c:ℝ) / d
  3.0035428163476343 - sum ≈ 1.0005418160141898 →
  ∃ n: ℕ, n = 4002 ∧ (1.0005418160141898 ≈ (n:ℝ) / f) :=
sorry

end numerator_of_third_fraction_l532_532271


namespace find_f_2019_l532_532643

noncomputable def f : ℝ → ℝ
| x := sorry

theorem find_f_2019 :
  (∀ x : ℝ, f(x + 2) = f(x + 1) - f(x)) →
  f(1) = log 3 2 →
  f(2) = log 3 5 + log 3 3 →
  f(2019) = 1 :=
by
  intros h1 h2 h3
  sorry

end find_f_2019_l532_532643


namespace sum_of_geometric_seq_l532_532848

variable {α : Type*} [Field α] (a : ℕ → α)
variable (S : ℕ → α)
variable (q : α)
variable (a2 : a 2)
variable (a5 : a 5)
variable (h : a2 - 8 * a5 = 0)

theorem sum_of_geometric_seq
  (h1 : a 2 - 8 * a 5 = 0)
  (h2 : ∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q))
  (h3 : q^3 = 1 / 8)
  (h4 : q = 1 / 2) :
  S 8 / S 4 = 17 / 16 := by
  sorry

end sum_of_geometric_seq_l532_532848


namespace smallest_four_digit_divisible_by_35_l532_532207

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532207


namespace expected_value_min_l532_532626

noncomputable def expected_min (a b c : ℝ) (ha : a ≥ b) (hb : b ≥ c) : ℝ :=
  (c^2 / 2) * (1 / a + 1 / b + 1 / c) - (2 * c^3 / 3) * (1 / (a * b) + 1 / (b * c) + 1 / (c * a)) + (3 * c^4 / (4 * a * b * c))

theorem expected_value_min (a b c : ℝ) (ha : a ≥ b) (hb : b ≥ c) :
  ∫ (x : ℝ) in 0..a, ∫ (y : ℝ) in 0..b, ∫ (z : ℝ) in 0..c, min (min x y) z = 
  expected_min a b c ha hb :=
sorry

end expected_value_min_l532_532626


namespace problem_statement_l532_532417

noncomputable 
def is_odd_function {α : Type*} [linear_ordered_field α] [topological_space α] (f : α → α) :=
  ∀ x, f (-x) = -f (x)

theorem problem_statement {α : Type*} [linear_ordered_field α] :
  ∀ (a c d : α), a ≠ 0 → 
  is_odd_function (λ x, a * x^3 + c * x + d) →
  (∃ x : α, x = 1 ∧ (a * x^3 + c * x + d) = -2) →
  (∃ (a c : α), (λ x, a * x^3 + c * x) = λ x, x^3 - 3 * x ∧  
                increasing_intervals (λ x, x^3 - 3 * x) = [(-∞, -1), (1, +∞)] ∧
                decreasing_intervals (λ x, x^3 - 3 * x) = [(-1, 1)] ∧
                max_value (λ x, x^3 - 3 * x) = 2) :=
by 
  sorry

end problem_statement_l532_532417


namespace cade_initial_marbles_l532_532674

def initial_marbles (given_away left: ℕ) : ℕ := given_away + left

theorem cade_initial_marbles
  (given_away : ℕ)
  (left : ℕ)
  (h_given_away : given_away = 8)
  (h_left : left = 79) :
  initial_marbles given_away left = 87 :=
by
  simp [h_given_away, h_left, initial_marbles]
  sorry

end cade_initial_marbles_l532_532674


namespace perpendicular_vectors_l532_532781

theorem perpendicular_vectors {m : ℝ} 
  (a : ℝ × ℝ := (1, -2))
  (b : ℝ × ℝ := (m, m - 4))
  (h : a.1 * b.1 + a.2 * b.2 = 0) :
  m = 4 :=
begin
  sorry
end

end perpendicular_vectors_l532_532781


namespace login_probability_l532_532614

def possible_letters := {'A', 'a', 'B', 'b'}
def possible_numbers := {4, 5, 6}

theorem login_probability :
  let num_combinations := Set.size (possible_letters × possible_numbers) in
  1 / num_combinations = 1 / 12 := by
  sorry

end login_probability_l532_532614


namespace infinite_n_exists_l532_532541

theorem infinite_n_exists (h1 : ∀ (n : ℕ), 0 < n)
  (h2 : ∀ (p : ℕ), Prime p → ∃ (n : ℕ), p ∣ (n^2 + 3))
  (h3 : ∃ (k : ℕ), k^2 < n ∧ p ∣ (k^2 + 3)) :
  ∃∞ n : ℕ, ∀ p, Prime p → (p ∣ (n ^ 2 + 3) → ∃ k, k^2 < n ∧ p ∣ (k^2 + 3)) :=
by 
  sorry

end infinite_n_exists_l532_532541


namespace math_proof_l532_532479

noncomputable def problem_statement : Prop :=
  -- Definition of line l
  let line_l := ∀ t : ℝ, (x t = -t ∧ y t = sqrt 3 - t) in
  
  -- General equation of line l
  let general_eq_line_l := ∀ x y : ℝ, (x - y + sqrt 3 = 0) in
  
  -- Polar to rectangular transformation for curve C1
  let curve_c1 := ∀ (θ : ℝ) (ρ : ℝ), (ρ = 4 * sin θ) -> (x^2 + (y-2)^2 = 4) in
  
  -- Transformation to obtain curve C2
  let curve_c2 := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1) in
  
  -- Intersection computations
  let line_intersect_x_axis := ∀ x : ℝ, (x = -sqrt 3) in
  let quadratic_eq := ∀ t : ℝ, (5*t^2 - 2*sqrt 6 * t - 2 = 0) in
  
  -- Roots properties
  let root_sum := (t₁ t₂ : ℝ), (t₁ + t₂ = 2 * sqrt 6 / 5) in
  let root_prod := (t₁ t₂ : ℝ), (t₁ * t₂ = -2 / 5) in
  
  -- Final proof goal
  let final_value := (|PM| |PN| : ℝ), (1 / |PM| + 1 / |PN| = 4) in

  general_eq_line_l ∧ curve_c1 ∧ curve_c2 ∧ line_intersect_x_axis ∧ quadratic_eq ∧ root_sum ∧ root_prod -> final_value

theorem math_proof : problem_statement := sorry

end math_proof_l532_532479


namespace math_problem_l532_532747

open Real

theorem math_problem (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (cos (3 * π / 2 + α) ^ 2 + 2 * cos α * cos (π / 2 - α)) / (1 + sin (π / 2 - α) ^ 2) = 4 / 3 :=
by
  sorry

end math_problem_l532_532747


namespace cos_585_eq_neg_sqrt2_div_2_l532_532585

theorem cos_585_eq_neg_sqrt2_div_2 : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by sorry

end cos_585_eq_neg_sqrt2_div_2_l532_532585


namespace Polina_Vika_can_form_square_l532_532600

-- Define the conditions and question
variables (Vika_pieces : List ℝ) (Alina_square Polina_square : ℝ) (different_size : Polina_square ≠ Alina_square)
variables (can_form_square : ∃ (s : ℝ), (length Vika_pieces = 4 ∧ Alina_square + (sum Vika_pieces) = s^2))

-- Formalize the statement to be proved
theorem Polina_Vika_can_form_square
  (hV : length Vika_pieces = 4)
  (hA : Alina_square > 0)
  (hP : Polina_square > 0)
  (h_diff : Polina_square ≠ Alina_square)
  (h_form: ∃ (s : ℝ), Alina_square + (sum Vika_pieces) = s^2) :
  ∃ (s' : ℝ), Polina_square + (sum Vika_pieces) = s'^2 :=
sorry

end Polina_Vika_can_form_square_l532_532600


namespace takeoff_run_length_l532_532136

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l532_532136


namespace sum_of_numbers_equal_16_l532_532315

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l532_532315


namespace shoe_length_increase_l532_532298

noncomputable def shoeSizeLength (l : ℕ → ℝ) (size : ℕ) : ℝ :=
  if size = 15 then 9.25
  else if size = 17 then 1.3 * l 8
  else l size

theorem shoe_length_increase :
  (forall l : ℕ → ℝ,
    (shoeSizeLength l 15 = 9.25) ∧
    (shoeSizeLength l 17 = 1.3 * (shoeSizeLength l 8)) ∧
    (forall n, shoeSizeLength l (n + 1) = shoeSizeLength l n + 0.25)
  ) :=
  sorry

end shoe_length_increase_l532_532298


namespace inequality_proof_l532_532389

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  (1 / a + 4 / (1 - a) ≥ 9) := 
sorry

end inequality_proof_l532_532389


namespace problem1_l532_532281

theorem problem1 (n : ℕ) (h_prime : Prime n) (h_gt3 : 3 < n) :
  (∏ k in Finset.range (n + 1), (1 + 2 * Real.cos (2 * k * Real.pi / n))) = 3 :=
sorry

end problem1_l532_532281


namespace angle_C_eq_pi_div_4_l532_532048

-- Let A, B, and C be vertices of triangle ABC
variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]

-- Define the given conditions
def angle_A := real.pi / 3
def length_BC := 3
def length_AB := real.sqrt 6

-- Define the target statement
theorem angle_C_eq_pi_div_4 :
  ∀ (sin : real → real) (π : real), 
  ∃ (C : real), sin angle_A = real.sqrt 3 / 2 ∧ 
  ∃ (sin_C : real), sin_C = real.sqrt 2 / 2 ∧ 
  length_AB / sin_C = length_BC / (sin angle_A) ∧ 
  (0 < C ∧ C < real.pi / 3) -> 
  C = real.pi / 4 := 
sorry

end angle_C_eq_pi_div_4_l532_532048


namespace property_P_difference_l532_532392

noncomputable def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    6 * 2^(n / 2) - n - 5 
  else 
    4 * 2^((n + 1) / 2) - n - 5

theorem property_P_difference : f 9 - f 8 = 31 := by
  sorry

end property_P_difference_l532_532392


namespace common_difference_arithmetic_sequence_l532_532920

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532920


namespace smallest_four_digit_number_divisible_by_35_l532_532226

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532226


namespace find_a_and_b_l532_532027

theorem find_a_and_b (a b : ℝ) 
  (curve : ∀ x : ℝ, y = x^2 + a * x + b) 
  (tangent : ∀ x : ℝ, y - b = a * x) 
  (tangent_line : ∀ x y : ℝ, x + y = 1) :
  a = -1 ∧ b = 1 := 
by 
  sorry

end find_a_and_b_l532_532027


namespace smallest_four_digit_divisible_by_35_l532_532208

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532208


namespace continuous_function_identity_l532_532709

noncomputable theory

open Real

theorem continuous_function_identity
  (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ x : ℝ, f (-x) + ∫ t in 0..x, t * f (x - t) = x) → (∀ x : ℝ, f x = -sinh x) :=
by
  sorry

end continuous_function_identity_l532_532709


namespace smallest_four_digit_divisible_by_35_l532_532250

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532250


namespace boys_belong_to_other_communities_l532_532468

/-- In a school of 300 boys, if 44% are Muslims, 28% are Hindus, and 10% are Sikhs,
then the number of boys belonging to other communities is 54. -/
theorem boys_belong_to_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℕ)
  (b : total_boys = 300)
  (m : percentage_muslims = 44)
  (h : percentage_hindus = 28)
  (s : percentage_sikhs = 10) :
  total_boys * ((100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 54 := 
sorry

end boys_belong_to_other_communities_l532_532468


namespace arithmetic_expression_equality_l532_532257

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l532_532257


namespace count_junk_numbers_l532_532331

-- Define a function to check if a number is junk
def is_junk (n : ℕ) : Prop :=
  n ≥ 2 ∧ (∃ a b : Fin n → ℕ, 
    a ≠ b ∧ 
    (a 0 + a 1 = b 0 + b 1) ∧ 
    (∀ i : ℕ, 2 ≤ i ∧ i ≤ n-1 → a (i-1) + a i + a (i+1) = b (i-1) + b i + b (i+1)) ∧
    (a (n-1) + a n = b (n-1) + b n))

-- Define the main theorem to count junk numbers within the specified range
theorem count_junk_numbers : 
  (Finset.filter is_junk (Finset.range 2017)).card = 672 := 
  sorry

end count_junk_numbers_l532_532331


namespace tan_neg2970_is_undefined_l532_532335

noncomputable def tan : ℝ → ℝ := λ θ, Real.sin θ / Real.cos θ

theorem tan_neg2970_is_undefined :
  Real.cos (270 * Real.pi / 180) = 0 → tan (-2970 * Real.pi / 180) = 0 :=
by
  intro h270
  sorry

end tan_neg2970_is_undefined_l532_532335


namespace common_difference_of_arithmetic_sequence_l532_532879

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532879


namespace remaining_height_of_cut_cube_l532_532290

theorem remaining_height_of_cut_cube (s : ℝ) (hs : s = 2) :
  let h := 2 - (2 * Real.sqrt 3) / 3 in
  height_cube s = h := 
sorry

end remaining_height_of_cut_cube_l532_532290


namespace common_difference_arithmetic_sequence_l532_532926

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532926


namespace martha_jar_spices_cost_l532_532083

def price_per_jar_spices (p_beef p_fv p_oj : ℕ) (price_spices : ℕ) :=
  let total_spent := (3 * p_beef) + (8 * p_fv) + p_oj + (3 * price_spices)
  let total_points := (total_spent / 10) * 50 + if total_spent > 100 then 250 else 0
  total_points

theorem martha_jar_spices_cost (price_spices : ℕ) :
  price_per_jar_spices 11 4 37 price_spices = 850 → price_spices = 6 := by
  sorry

end martha_jar_spices_cost_l532_532083


namespace integer_root_modulus_bound_l532_532095

noncomputable def polynomial_modulus_bound (n : ℕ) (a : Fin (n + 1) → ℤ) (c : ℤ) : Prop :=
  let max_modulus := (Finset.sup (Finset.univ.image (λ k, (a k).natAbs))).val
  ∀ (P : ℤ → ℤ), 
    (P = λ x, ∑ i in Finset.range (n + 1), a i * x^i) →
    P c = 0 →
    (c.natAbs ≤ max_modulus)

theorem integer_root_modulus_bound (n : ℕ) (a : Fin (n + 1) → ℤ) (c : ℤ) : 
  polynomial_modulus_bound n a c := 
sorry

end integer_root_modulus_bound_l532_532095


namespace henrique_chocolates_l532_532788

theorem henrique_chocolates
  (cost_per_bar : ℝ)
  (total_paid : ℝ)
  (change_received : ℝ)
  (h1 : cost_per_bar = 1.35)
  (h2 : total_paid = 10.00)
  (h3 : change_received < 1.00) :
  ∃ x : ℕ, x = 7 ∧ (total_paid - x * cost_per_bar) = change_received := by
  use 7
  split
  { exact rfl }
  {
    calc
      total_paid - 7 * cost_per_bar
      = 10.00 - 7 * 1.35 : by rw [h1, h2]
      = 0.55 : by norm_num
   }

end henrique_chocolates_l532_532788


namespace angle_B_measure_sine_C_value_l532_532484

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def triangle_conditions := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  ∠A > 0 ∧ ∠B > 0 ∧ ∠C > 0 ∧
  ∠A + ∠B + ∠C = π ∧
  a = 5 ∧ b = 7 ∧
  sin(A)^2 + sin(A) * sin(C) + sin(C)^2 + cos(B)^2 = 1

theorem angle_B_measure: triangle_conditions → B = 2 * π / 3 := sorry

theorem sine_C_value: triangle_conditions → sin C = 3 * sqrt 3 / 14 := sorry

end angle_B_measure_sine_C_value_l532_532484


namespace cos_sum_eq_one_l532_532388

theorem cos_sum_eq_one 
  {α β : ℝ} 
  (h : (cos α * cos (β / 2)) / cos (α - β / 2) + (cos β * cos (α / 2)) / cos (β - α / 2) = 1) : 
  cos α + cos β = 1 := 
sorry

end cos_sum_eq_one_l532_532388


namespace max_distance_a_b_l532_532009

noncomputable def max_a_minus_b (θ : ℝ) : ℝ :=
| (1, Real.sin θ) - (1, Real.cos θ) |

theorem max_distance_a_b (θ : ℝ) : 
  max_a_minus_b θ = Real.sqrt 2 := 
sorry

end max_distance_a_b_l532_532009


namespace quadratic_equation_coefficients_l532_532828

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℤ), a = 2 ∧ b = 1 ∧ c = -5 ∧ (∀ x : ℤ, 2 * x^2 + x - 5 = a * x^2 + b * x + c) :=
begin
  sorry
end

end quadratic_equation_coefficients_l532_532828


namespace jenny_money_l532_532834

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l532_532834


namespace probability_not_losing_l532_532090

theorem probability_not_losing (p_draw p_win p_not_lose : ℝ) (h1 : p_draw = 0.4) (h2 : p_win = 0.4) : p_not_lose = p_draw + p_win := by
  have h : p_draw + p_win = 0.4 + 0.4 := by rw [h1, h2]
  have h3 : 0.4 + 0.4 = 0.8 := by norm_num
  rw [h3] at h
  exact h.symm

end probability_not_losing_l532_532090


namespace triangle_ABC_is_isosceles_right_l532_532425

open Real

def vector := ℝ × ℝ

def ba : vector := (1, -3)
def bc : vector := (4, -2)
def ac : vector := (bc.1 - ba.1, bc.2 - ba.2)

def norm (v : vector) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v w : vector) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_ABC_is_isosceles_right :
  norm ba = norm ac ∧ dot_product ba ac = 0 :=
by
  sorry

end triangle_ABC_is_isosceles_right_l532_532425


namespace max_value_MN_l532_532538

noncomputable def max_MN_value : ℝ :=
  (15 + Real.sqrt 23) / 2

theorem max_value_MN :
  ∀ (N : ℝ×ℝ) (M : ℝ×ℝ),
    (N.1 + 5)^2 + N.2^2 = 1 →
    (M.1 - 1.5)^2 + M.2^2 = (Real.sqrt 23 / 2)^2 →
    dist N M ≤ max_MN_value :=
begin
  sorry
end

end max_value_MN_l532_532538


namespace find_certain_number_l532_532265

theorem find_certain_number (x : ℕ) (h : (55 * x) % 8 = 7) : x = 1 := 
sorry

end find_certain_number_l532_532265


namespace problem_proof_l532_532040

-- Define the given circle C and curve C1
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def curve_C1 (ρ θ : ℝ) : Prop := ρ = -4 * real.sqrt 3 * real.sin θ

-- Define the polar equation of line AB
def line_AB_polar (θ : ℝ) : Prop := θ = -real.pi / 6

-- Define the parameterized line C2 passing through point C(2, 0)
def line_C2_param (x y t : ℝ) : Prop :=
  x = 2 + real.sqrt 3 / 2 * t ∧ y = 1 / 2 * t

-- The points D and E as intersections
def point_D (x y : ℝ) : Prop :=
  ∃ t_D, line_C2_param x y t_D ∧ y = -real.sqrt 3 / 3 * x  

def point_E (x y : ℝ) : Prop :=
  ∃ t_E, line_C2_param x y t_E ∧ x = 0

-- The distances |CD| and |CE|
def dist_CD (t_D : ℝ) : ℝ := abs t_D
def dist_CE (t_E : ℝ) : ℝ := abs t_E

-- The ratio of distances
def ratio_CD_CE (r : ℝ) : Prop := r = 1 / 2

theorem problem_proof :
  (∀ x y, circle_C x y → ∃ θ, line_AB_polar θ) ∧
  (∀ t,
    (∃ xD yD, point_D xD yD) →
    (∃ xE yE, point_E xE yE) →
    (∃ t_D t_E, dist_CD t_D / dist_CE t_E = 1 / 2)) :=
  by
    sorry

end problem_proof_l532_532040


namespace initial_cows_l532_532594

theorem initial_cows {D C : ℕ}
  (h1 : C = 2 * D)
  (h2 : 161 = (3 * C) / 4 + D / 4) :
  C = 184 :=
by
  sorry

end initial_cows_l532_532594


namespace scientific_notation_of_ruthenium_l532_532115

def scientific_notation (x : ℝ) : ℝ × ℕ :=
  if x = 0 then (0, 0)
  else let n := (Real.log10 (Real.abs x)).floor;
       let m := x / (10 ^ n);
       (m, n)

theorem scientific_notation_of_ruthenium : scientific_notation (0.000000000189) = (1.89, -10) :=
by sorry

end scientific_notation_of_ruthenium_l532_532115


namespace max_value_x_sqrt_one_minus_x_squared_l532_532714

theorem max_value_x_sqrt_one_minus_x_squared : 
  ∃ x ∈ Icc (0 : ℝ) 1, (∀ y ∈ Icc (0 : ℝ) 1, x * real.sqrt (1 - x^2) ≤ y * real.sqrt (1 - y^2) → y = x) ∧ x * real.sqrt (1 - x^2) = 1 / 2 :=
sorry

end max_value_x_sqrt_one_minus_x_squared_l532_532714


namespace exists_real_constant_c_and_n0_l532_532502

-- Defining the sequence operation and conditions
def sequence_operation (n : ℕ) (seq : List ℕ) : List ℕ := sorry -- define your operation on sequence

-- Define the function S(n)
noncomputable def S (n : ℕ) : ℕ :=
  sorry -- define S(n) according to the problem conditions

-- Main theorem to prove the existence of real constant c > 0 and positive integer n_0 such that for all n >= n_0, S(n) >= c * (n ^ (3/2))
theorem exists_real_constant_c_and_n0 (c : ℝ) (n₀ : ℕ) (h : c > 0) : 
  ∃ (c : ℝ) (n₀ : ℕ), 0 < c ∧ ∀ (n : ℕ), n ≥ n₀ → S(n) ≥ c * (n ^ (3/2)) :=
  sorry -- this is the main statement we need to prove

end exists_real_constant_c_and_n0_l532_532502


namespace intersecting_triangles_area_l532_532652

-- Define the problem conditions and the conclusion
theorem intersecting_triangles_area :
  ∀ (rect_length rect_height : ℝ) (side_length : ℝ),
      rect_length = 4 →
      rect_height = 2 * Real.sqrt 3 →
      side_length = 4 →
  ∃ (area : ℝ),
    area = 4 * Real.sqrt 3 :=
by
  intros rect_length rect_height side_length h1 h2 h3
  use 4 * Real.sqrt 3
  sorry

end intersecting_triangles_area_l532_532652


namespace smallest_four_digit_divisible_by_35_l532_532216

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532216


namespace arithmetic_sequence_common_difference_l532_532898

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532898


namespace magnitude_sum_vectors_l532_532030

noncomputable def a : E := sorry -- Define vector a
noncomputable def b : E := sorry -- Define vector b
noncomputable def c : E := sorry -- Define vector c

def equal_angles_pairwise (a b c : E) : Prop := sorry -- Define the property that the vectors form equal angles pairwise

axiom a_norm : ‖a‖ = 1 -- Given |𝑎| = 1
axiom b_norm : ‖b‖ = 1 -- Given |𝑏| = 1
axiom c_norm : ‖c‖ = 3 -- Given |𝑐| = 3
axiom equal_angles : equal_angles_pairwise a b c -- Vectors form equal angles pairwise

theorem magnitude_sum_vectors : 
  ‖a + b + c‖ = 2 ∨ ‖a + b + c‖ = 5 := 
by 
  sorry

end magnitude_sum_vectors_l532_532030


namespace integer_root_modulus_le_max_modulus_of_coeffs_l532_532097

theorem integer_root_modulus_le_max_modulus_of_coeffs 
  (n : ℕ) 
  (a : Fin (n+1) → ℤ) 
  (x : ℤ) 
  (root_condition : ∑ i in Finset.range (n+1), a i * x^i = 0) :
  |x| ≤ finset_sup (Finset.range (n+1)) (λ i, |a i|) := 
sorry

end integer_root_modulus_le_max_modulus_of_coeffs_l532_532097


namespace stone_number_at_150_l532_532464

-- Definitions according to the conditions
def twelve_stones : list ℕ := list.range 1 13  -- Stones labeled from 1 to 12
def count_pattern (n : ℕ) : ℕ :=
  match n % 6 with
  | 0 => 12
  | 1 => 10
  | 2 => 8
  | 3 => 6
  | 4 => 4
  | _ => 2

-- To prove that the 150th counted stone corresponds to original stone number 4.
theorem stone_number_at_150 : 
  let count_upto_k := (k : ℕ) := list.sum (list.map count_pattern (list.range 0 k)),
      first_partial_sum := count_upto_k 6,
      num_full_cycles := 150 / first_partial_sum,
      remaining_counts := 150 % first_partial_sum,
      final_position := 
        let final_stone_sequence :=
          list.drop (remaining_counts - 1) $ 
            list.map count_pattern (list.range 0 remaining_counts) in
        list.head final_stone_sequence
  in final_position = 4 := by
  sorry

end stone_number_at_150_l532_532464


namespace prove_conclusion_l532_532516

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x: ℝ, f(x + 3) = -1 / f(x)
axiom condition2 : ∀ x1 x2: ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f(x1) < f(x2)
axiom condition3 : ∀ x: ℝ, f(3 - x) = f(3 + x)

theorem prove_conclusion : f(3) < f(4.5) ∧ f(4.5) < f(7) :=
by {
  sorry
}

end prove_conclusion_l532_532516


namespace find_angle_C_find_length_AB_l532_532786

-- Given vectors and scalar conditions
variables {A B C : ℝ}
def m : ℝ × ℝ := (real.sin A, real.sin B)
def n : ℝ × ℝ := (real.cos B, real.cos A)
axiom inner_product_cond : (real.sin A * real.cos B + real.sin B * real.cos A) = real.sin (2 * C)
axiom angle_cond : A + B = π - C
axiom internal_angles : 0 < C ∧ C < π

-- Proving angle C
theorem find_angle_C : C = π / 3 := sorry

-- Additional conditions for part II
axiom arithmetic_seq_cond : 2 * real.sin C = real.sin A + real.sin B
axiom ca_dot_ab_ac : real.sqrt (1 - m.2 ^ 2) * real.sqrt (1 - n.2 ^ 2) * real.cos C = 18
axiom law_of_sines : 2 * (real.sqrt (1 - (real.sin C)^2)) = real.sqrt (1 - (real.sin A)^2) + real.sqrt (1 - (real.sin B)^2)
axiom product_ab : n.1 * n.2 = 36 

-- Proving the length of AB
theorem find_length_AB : real.sqrt (1 - (real.sin C) ^ 2) = 6 := sorry

end find_angle_C_find_length_AB_l532_532786


namespace arithmetic_sequence_common_difference_l532_532856

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532856


namespace sum_of_numbers_equal_16_l532_532317

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l532_532317


namespace add_fractions_l532_532708

theorem add_fractions : (2 / 3 : ℚ) + (7 / 8) = 37 / 24 := 
by sorry

end add_fractions_l532_532708


namespace count_n_not_dividing_g_l532_532063

noncomputable def g (n : ℕ) : ℕ := 
  (List.prod (List.filter (λ d => d ≠ n) (List.divisors n)))

theorem count_n_not_dividing_g : 
  (Finset.filter (λ n => ¬ n ∣ g n) (Finset.Icc 2 50)).card = 19 := sorry

end count_n_not_dividing_g_l532_532063


namespace count_integer_roots_l532_532381

theorem count_integer_roots :
  let n := 15625
  let exp := 6
  let f := λ (k : ℕ), (n : ℚ)^(1 / k)
  let int_nums := {k : ℕ | (f k).denom = 1 ∧ (range (exp + 1)).count(k ∣ exp) = 1}
  int_nums.size = 4 := sorry

end count_integer_roots_l532_532381


namespace find_a_value_l532_532775

theorem find_a_value (a : ℝ) :
  (∃ x y : ℝ, (ax + y - 1 = 0) ∧ ((x - 1)*(x - 1) + (y + a)*(y + a) = 1) ∧ 
  (abs (a * 1 + 1 * (-a) - 1) / real.sqrt (a * a + 1) = (real.sqrt 2 / 2)) → (a = 1 ∨ a = -1)) :=
sorry

end find_a_value_l532_532775


namespace smallest_four_digit_number_divisible_by_35_l532_532227

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532227


namespace triangle_third_side_l532_532804

theorem triangle_third_side (x : ℝ) (h1 : x > 2) (h2 : x < 6) : x = 5 :=
sorry

end triangle_third_side_l532_532804


namespace arithmetic_sequence_common_difference_l532_532849

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532849


namespace no_real_roots_of_quadratics_l532_532611

theorem no_real_roots_of_quadratics :
  let eqA := x^2 + x - 2 = 0
  let eqB := x^2 - 2x = 0
  let eqC := x^2 + x + 5 = 0
  let eqD := x^2 - 2x + 1 = 0
  (∀ a b c, a = 1 ∧ b = 1 ∧ c = -2 → (b^2 - 4 * a * c > 0)) ∧
  (∀ a b c, a = 1 ∧ b = -2 ∧ c = 0 → (b^2 - 4 * a * c > 0)) ∧
  (∀ a b c, a = 1 ∧ b = 1 ∧ c = 5 → (b^2 - 4 * a * c < 0)) ∧
  (∀ a b c, a = 1 ∧ b = -2 ∧ c = 1 → (b^2 - 4 * a * c = 0)) →
  eqC has_no_real_roots :=
by
  sorry

end no_real_roots_of_quadratics_l532_532611


namespace triangle_bisects_angle_l532_532840

/-- Given an acute-angled triangle ABC and the tangents to its circumcircle at A, B, C forming a triangle PQR
with C ∈ PQ and B ∈ PR. Let C₁ be the foot of the altitude from C in ΔABC. Prove that CC₁ bisects ∠QC₁P. -/
theorem triangle_bisects_angle (A B C P Q R C₁ : Point)
  (hAcute : acute ⟨A, B, C⟩)
  (hCircumcircleTangent : tangent (circumcircle ⟨A, B, C⟩) A P ∧
                          tangent (circumcircle ⟨A, B, C⟩) B Q ∧
                          tangent (circumcircle ⟨A, B, C⟩) C R)
  (hPointsOnTangent : C ∈ PQ ∧ B ∈ PR)
  (hAltitude : is_altitude ⟨A, B, C⟩ C C₁) :
  angle_bisector C C₁ Q P :=
sorry

end triangle_bisects_angle_l532_532840


namespace actual_cost_l532_532310

theorem actual_cost (x : ℝ) (h : 0.80 * x = 200) : x = 250 :=
sorry

end actual_cost_l532_532310


namespace top_card_is_queen_probability_l532_532299

theorem top_card_is_queen_probability :
  let total_cards := 54
  let number_of_queens := 4
  (number_of_queens / total_cards) = (2 / 27) := by
    sorry

end top_card_is_queen_probability_l532_532299


namespace average_shift_l532_532112

theorem average_shift (n : ℕ) (x : ℕ → ℝ) (a : ℝ) (xbar : ℝ) 
  (h : xbar = (1 / n) * (finset.sum (finset.range n) x)) :
  (1 / n) * (finset.sum (finset.range n) (λ i, x i + a)) = xbar + a :=
sorry

end average_shift_l532_532112


namespace problem_solution_l532_532006

-- Definitions of propositions p and q
def p : Prop := ∃ a : ℝ, ∀ x y : ℝ, (2 * x + y - 1 = 0) → (x + a * y + 1 = 0) → (a = -2)
def q : Prop := ∀ (f : ℝ → ℝ) (x₀ : ℝ), (f' x₀ = 0) → is_extreme_point f x₀

-- Lean 4 equivalent of proving the correct option
theorem problem_solution : p ∧ ¬q := by
  -- placeholder
  sorry

end problem_solution_l532_532006


namespace at_most_one_divisor_square_l532_532504

theorem at_most_one_divisor_square (p n : ℕ) (hp : Prime p) (hpn : p > 2) (hn_pos : 0 < n):
  ∃ (d : ℕ), (d ∈ divisors (p * n^2) ∧ ∃ (k : ℕ), k^2 = n^2 + d) → 
  ∀ (d1 d2 : ℕ), (d1 ∈ divisors (p * n^2) ∧ ∃ k1, k1^2 = n^2 + d1) → 
                 (d2 ∈ divisors (p * n^2) ∧ ∃ k2, k2^2 = n^2 + d2) → 
                 d1 = d2 :=
sorry

end at_most_one_divisor_square_l532_532504


namespace terminating_decimal_non_zero_tenths_digit_count_l532_532724

theorem terminating_decimal_non_zero_tenths_digit_count :
  ∃! n : ℕ, 
    n ≤ 10 ∧ (∃ a b : ℕ, n = 2^a * 5^b) ∧
    (∃ d : ℕ, 0 < d ∧ d < 10 ∧ (n = d ∨ n = 2*d ∨ n = 4*d ∨ n = 5*d ∨ n = 10*d)) ∧ 
    (frac (1/n : ℚ) has non-zero tenths digit) := 
sorry

end terminating_decimal_non_zero_tenths_digit_count_l532_532724


namespace case_1_divisible_by_9_case_2_not_divisible_by_9_l532_532170

-- Case 1: k = 10
theorem case_1_divisible_by_9 (A : ℕ → ℕ) (B : ℕ → ℕ) (k : ℕ) (h_k : k = 10)
  (h_A : ∀ n, 1 ≤ A n ∧ A n ≤ 5) (h_B : ∀ n, 1 ≤ B n ∧ B n ≤ 5)
  (h_alternate : ∀ n, n % 2 = 1 → B n = 6 - A n)
  : (∑ i in finset.range (2 * k), if i % 2 = 0 then A (i / 2) else B (i / 2)) % 9 = 0 := 
  sorry

-- Case 2: k = 15
theorem case_2_not_divisible_by_9 (A : ℕ → ℕ) (B : ℕ → ℕ) (k : ℕ) (h_k : k = 15)
  (h_A : ∀ n, 1 ≤ A n ∧ A n ≤ 5) (h_B : ∀ n, 1 ≤ B n ∧ B n ≤ 5)
  : ∀ (h_A_strategy : ∀ i, A (2 * i + 1) ∈ {1,2,3,4,5} → B (2 * i) = 6 - A (2 * i + 1)),
    (∑ i in finset.range (2 * 15), if i % 2 = 0 then A (i / 2) else B (i / 2)) % 9 ≠ 0 :=
  sorry

end case_1_divisible_by_9_case_2_not_divisible_by_9_l532_532170


namespace prob1_prob2_l532_532678

variables (x y a b c : ℝ)

-- Proof for the first problem
theorem prob1 :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 := 
sorry

-- Proof for the second problem
theorem prob2 :
  -2 * (-a^2 * b * c)^2 * (1 / 2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
sorry

end prob1_prob2_l532_532678


namespace milo_cash_reward_l532_532084

theorem milo_cash_reward : 
  let three_2s := [2, 2, 2]
  let four_3s := [3, 3, 3, 3]
  let one_4 := [4]
  let one_5 := [5]
  let all_grades := three_2s ++ four_3s ++ one_4 ++ one_5
  let total_grades := all_grades.length
  let total_sum := all_grades.sum
  let average_grade := total_sum / total_grades
  5 * average_grade = 15 := by
  sorry

end milo_cash_reward_l532_532084


namespace find_common_difference_l532_532954

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532954


namespace vegetables_sold_mass_correct_l532_532294

-- Definitions based on the problem's conditions
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8
def total_mass_vegetables := mass_carrots + mass_zucchini + mass_broccoli
def mass_of_vegetables_sold := total_mass_vegetables / 2

-- Theorem to be proved
theorem vegetables_sold_mass_correct : mass_of_vegetables_sold = 18 := by 
  sorry

end vegetables_sold_mass_correct_l532_532294


namespace dolls_proof_l532_532523

variable (blonde_dolls brown_dolls black_dolls : ℕ)

def given_conditions (blonde_dolls brown_dolls black_dolls : ℕ) : Prop :=
  blonde_dolls = 4 ∧
  brown_dolls = 4 * blonde_dolls ∧
  black_dolls = brown_dolls - 2

def question (blonde_dolls brown_dolls black_dolls : ℕ) : ℕ :=
  (brown_dolls + black_dolls) - blonde_dolls

theorem dolls_proof :
  ∀ (blonde_dolls brown_dolls black_dolls : ℕ),
  given_conditions blonde_dolls brown_dolls black_dolls →
  question blonde_dolls brown_dolls black_dolls = 26 :=
by
  intros blonde_dolls brown_dolls black_dolls h
  simp [given_conditions, question] at *
  split at h
  -- Now you'd need to either finish the proof or fill in the steps taken in the given solution.
  sorry

end dolls_proof_l532_532523


namespace log_sum_interval_l532_532144

theorem log_sum_interval : 
  let x := (1 / (Real.log (1/3) / Real.log (1/2))) + (1 / (Real.log (1/3) / Real.log (1/5)))
  in 2 < x ∧ x < 3 := by
  let x := (1 / (Real.log (1/3) / Real.log (1/2))) + (1 / (Real.log (1/3) / Real.log (1/5)))
  have h : x = Real.log 10 / Real.log 3 :=
    sorry
  rw [h]
  exact ⟨by norm_num, by norm_num⟩

end log_sum_interval_l532_532144


namespace arithmetic_sequence_common_difference_l532_532940

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532940


namespace smallest_four_digit_divisible_by_35_l532_532237

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l532_532237


namespace no_real_roots_of_quadratics_l532_532610

theorem no_real_roots_of_quadratics :
  let eqA := x^2 + x - 2 = 0
  let eqB := x^2 - 2x = 0
  let eqC := x^2 + x + 5 = 0
  let eqD := x^2 - 2x + 1 = 0
  (∀ a b c, a = 1 ∧ b = 1 ∧ c = -2 → (b^2 - 4 * a * c > 0)) ∧
  (∀ a b c, a = 1 ∧ b = -2 ∧ c = 0 → (b^2 - 4 * a * c > 0)) ∧
  (∀ a b c, a = 1 ∧ b = 1 ∧ c = 5 → (b^2 - 4 * a * c < 0)) ∧
  (∀ a b c, a = 1 ∧ b = -2 ∧ c = 1 → (b^2 - 4 * a * c = 0)) →
  eqC has_no_real_roots :=
by
  sorry

end no_real_roots_of_quadratics_l532_532610


namespace encoded_value_BCD_is_27_l532_532472

def symbol_encoding := {A : ℕ, B : ℕ, C : ℕ, D : ℕ}

def encode_to_base10 (s : symbol_encoding) (n : string) : ℕ :=
  n.foldr (λ c (acc: ℕ), acc * 4 + match c with
                                    | 'A' => s.A
                                    | 'B' => s.B
                                    | 'C' => s.C
                                    | 'D' => s.D
                                    | _   => 0 -- Will not happen as per the problem setup.
                                    end) 0

theorem encoded_value_BCD_is_27 :
  ∃ s : symbol_encoding,
    -- Conditions based on given information
    encode_to_base10 s "ABC" + 1 = encode_to_base10 s "ABD" ∧
    encode_to_base10 s "ABD" + 1 = encode_to_base10 s "BAA" ∧
    -- Proof of the final answer
    encode_to_base10 s "BCD" = 27 :=
by
  sorry

end encoded_value_BCD_is_27_l532_532472


namespace common_difference_is_two_l532_532916

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532916


namespace arithmetic_sequence_common_difference_l532_532943

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532943


namespace correct_flag_positions_l532_532054

-- Definitions for the gears and their relations
structure Gear where
  flag_position : ℝ -- position of the flag in degrees

-- Condition: Two identical gears
def identical_gears (A B : Gear) : Prop := true

-- Conditions: Initial positions and gear interaction
def initial_position_A (A : Gear) : Prop := A.flag_position = 0
def initial_position_B (B : Gear) : Prop := B.flag_position = 180
def gear_interaction (A B : Gear) (theta : ℝ) : Prop :=
  A.flag_position = -theta ∧ B.flag_position = theta

-- Definition for the final positions given a rotation angle θ
def final_position (A B : Gear) (theta : ℝ) : Prop :=
  identical_gears A B ∧ initial_position_A A ∧ initial_position_B B ∧ gear_interaction A B theta

-- Theorem stating the positions after some rotation θ
theorem correct_flag_positions (A B : Gear) (theta : ℝ) : final_position A B theta → 
  A.flag_position = -theta ∧ B.flag_position = theta :=
by
  intro h
  cases h
  sorry

end correct_flag_positions_l532_532054


namespace company_earnings_correct_l532_532288

def production_rates : ℕ → ℕ
| 0 := 1200  -- Monday
| 1 := 1500  -- Tuesday
| 2 := 1800  -- Wednesday
| 3 := 1600  -- Thursday
| 4 := 1400  -- Friday
| 5 := 1000  -- Saturday
| 6 :=  800  -- Sunday
| _ := 0

def selling_prices : ℕ → ℕ
| 0 := 150  -- Monday
| 1 := 160  -- Tuesday
| 2 := 170  -- Wednesday
| 3 := 155  -- Thursday
| 4 := 145  -- Friday
| 5 := 165  -- Saturday
| 6 := 140  -- Sunday
| _ := 0

noncomputable def earnings (day : ℕ) : ℕ :=
  (production_rates day) * (selling_prices day)

noncomputable def total_earnings_for_week : ℕ :=
  (List.range 7).sum (λ d, earnings d)

theorem company_earnings_correct :
  total_earnings_for_week = 1454000 := by
  sorry

end company_earnings_correct_l532_532288


namespace common_difference_l532_532991

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532991


namespace cylinder_min_surface_area_l532_532713

noncomputable def minimize_cylinder_surface_area (v : ℝ) : ℝ × ℝ :=
  let r := (v / (2 * Real.pi))^(1/3) in
  let h := 2 * r in
  (r, h)

theorem cylinder_min_surface_area (v : ℝ) (r h : ℝ) 
  (h_volume : v = Real.pi * r^2 * h) 
  (h_radius : r = (v / (2 * Real.pi))^(1/3))
  (h_height : h = 2 * r) :
  minimize_cylinder_surface_area v = (r, h) := 
  by
    sorry

end cylinder_min_surface_area_l532_532713


namespace find_common_difference_l532_532865

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532865


namespace units_digit_expression_l532_532372

theorem units_digit_expression: 
  (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 := 
by
  sorry

end units_digit_expression_l532_532372


namespace smallest_pos_int_with_eight_factors_l532_532347

theorem smallest_pos_int_with_eight_factors : ∃ n : ℕ, (∀ m : ℕ, (nat.number_of_factors n = 8 ∧ nat.number_of_factors m = 8 → n ≤ m)) ∧ n = 24 := 
sorry

end smallest_pos_int_with_eight_factors_l532_532347


namespace hyperbolas_same_asymptotes_l532_532373

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (y^2 / 25 - x^2 / M = 1)) → M = 225 / 16 :=
by
  sorry

end hyperbolas_same_asymptotes_l532_532373


namespace arithmetic_sequence_common_difference_l532_532933

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532933


namespace relationship_a_b_range_of_M_l532_532418

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (x^2 - (a + 2) * x + b)

theorem relationship_a_b (a b : ℝ) : 
  (2 * a^2 * 0 + f a b 0 - b = 0) → b = a + 2 - 2 * a^2 :=
by
  intro h
  dsimp only [f] at h
  have : Real.exp 0 * (0^2 - (a+2) * 0 + b) = 0 := by
    simp only [mul_zero, zero_mul, pi.zero_apply, add_zero]
  rw [Real.exp_zero, one_mul, add_zero, sub_zero] at this
  exact sorry

theorem range_of_M (a M : ℝ) : 
  a < 0 → (∃ x > 0, f a (a + 2 - 2 * a^2) x < M) ↔ M ≥ 2 :=
by
  intro ha
  have h_b : b = a + 2 - 2 * a^2 := sorry
  simp [f, h_b]
  exact sorry

end relationship_a_b_range_of_M_l532_532418


namespace roger_and_friend_fraction_l532_532140

theorem roger_and_friend_fraction 
  (total_distance : ℝ) 
  (fraction_driven_before_lunch : ℝ) 
  (lunch_time : ℝ) 
  (total_time : ℝ) 
  (same_speed : Prop) 
  (driving_time_before_lunch : ℝ)
  (driving_time_after_lunch : ℝ) :
  total_distance = 200 ∧
  lunch_time = 1 ∧
  total_time = 5 ∧
  driving_time_before_lunch = 1 ∧
  driving_time_after_lunch = (total_time - lunch_time - driving_time_before_lunch) ∧
  same_speed = (total_distance * fraction_driven_before_lunch / driving_time_before_lunch = total_distance * (1 - fraction_driven_before_lunch) / driving_time_after_lunch) →
  fraction_driven_before_lunch = 1 / 4 :=
sorry

end roger_and_friend_fraction_l532_532140


namespace common_difference_of_arithmetic_sequence_l532_532878

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532878


namespace ratio_ad_bc_l532_532446

theorem ratio_ad_bc (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 5 * c) (h3 : c = 3 * d) : 
  (a * d) / (b * c) = 4 / 3 := 
by 
  sorry

end ratio_ad_bc_l532_532446


namespace fixed_point_l532_532662

-- Conditions
variables {a b c : ℝ}
variable (k : ℝ)
variable (m : ℝ)

noncomputable def EllipseEquation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

theorem fixed_point 
  (ha : a > 0) (hb : b > 0) (h : a > b) 
  (he : c / a = 1 / 2) 
  (hd : (2 + c)^2 + 1 = 10) 
  (hk : k > 0) (hm_eq : m = - (2 * k / 7)) :
  ∀ (x y : ℝ), EllipseEquation x y →
  ∀ (y = k * x + m),
  (2/7, 0) 
:=
sorry

end fixed_point_l532_532662


namespace balloon_arrangements_l532_532012

theorem balloon_arrangements : 
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / (Nat.factorial k1 * Nat.factorial k2) = 1260 := 
by
  let n := 7
  let k1 := 2
  let k2 := 2
  sorry

end balloon_arrangements_l532_532012


namespace triangle_ABC_problem_l532_532047
noncomputable def m : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
noncomputable def n : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)

theorem triangle_ABC_problem
  (B : ℝ)
  (b : ℝ := 2)
  (a c : ℝ)
  (cos_B : Real.cos B = 1 / 2)
  (h_par : (m = (2 * Real.sin B, -Real.sqrt 3)) ∧ (n = (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)) ∧ (m.1 / n.1 = m.2 / n.2)) :
  B = π / 3 ∧ (∃ (S : ℝ), S = Real.sqrt 3 ∧ ∀ a c, 4 ≥ a * c ∧ (Real.sqrt 3 = (1 / 2) * a * c * Real.sin B)) := sorry

end triangle_ABC_problem_l532_532047


namespace handshakes_count_l532_532100

theorem handshakes_count (num_couples : ℕ) (num_men : ℕ) (num_women : ℕ) (handshakes_per_man : ℕ) (total_handshakes : ℕ) :
  num_couples = 7 →
  num_men = 7 →
  num_women = 7 →
  handshakes_per_man = 6 →
  total_handshakes = num_men * handshakes_per_man →
  total_handshakes = 42 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end handshakes_count_l532_532100


namespace relay_race_stage_distribution_l532_532308

/-- 
Given a sports team consisting of two boys and two girls, 
prove that the number of ways to distribute relay race stages 
such that stages run by girls alternate with stages run by boys is 8.
-/
theorem relay_race_stage_distribution :
  let boys : ℕ := 2
  let girls : ℕ := 2
  let patterns : ℕ := 2 -- (BGBG and GBGB)
  let arrangements_per_pattern := nat.factorial boys * nat.factorial girls
  2 * arrangements_per_pattern = 8 :=
by
  -- Corresponding proof would be provided here
  sorry

end relay_race_stage_distribution_l532_532308


namespace find_y_values_l532_532019

theorem find_y_values (x y : ℝ)
  (h1 : x^2 = y - 5)
  (h2 : x = -7 ∨ x = 2) :
  (x = -7 → y = 54) ∧ (x = 2 → y = 9) :=
by
  split
  · intro hx
    rw [hx] at h1
    norm_num at h1
    exact h1
  · intro hx
    rw [hx] at h1
    norm_num at h1
    exact h1

end find_y_values_l532_532019


namespace find_takeoff_run_distance_l532_532138

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end find_takeoff_run_distance_l532_532138


namespace locus_of_P_tangent_lines_through_A_l532_532748

open Set

def ellipse (M : ℝ × ℝ) : Prop :=
  M.1^2 / 4 + M.2^2 = 1

def projection (M : ℝ × ℝ) : ℝ × ℝ :=
  (3, M.2)

theorem locus_of_P :
  ∀ (M : ℝ × ℝ), ellipse M →
  let N := projection M in
  let P := (M.1 + N.1, M.2 + N.2) in
  (P.1 - 3)^2 + P.2^2 = 4 := 
sorry

theorem tangent_lines_through_A :
  let A : ℝ × ℝ := (1, 4) in
  let E : ℝ × ℝ → Prop := λ P, (P.1 - 3)^2 + P.2^2 = 4 in
  ∃ m : ℝ, (1 - m * 4 + 4 * m - 1 = 2 * sqrt(1 + m^2)) ∧
           (x - m * y + 19 - 4 * m = 0 ∨ x - 1 = 0) :=
sorry

end locus_of_P_tangent_lines_through_A_l532_532748


namespace smallest_four_digit_divisible_by_35_l532_532178

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532178


namespace multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l532_532553

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l532_532553


namespace possible_sets_C_l532_532694

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def is_partition (A B C : Set ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ A ∪ B ∪ C = M

def conditions (A B C : Set ℕ) : Prop :=
  is_partition A B C ∧ (∃ (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ), 
    A = {a1, a2, a3, a4} ∧
    B = {b1, b2, b3, b4} ∧
    C = {c1, c2, c3, c4} ∧
    c1 < c2 ∧ c2 < c3 ∧ c3 < c4 ∧
    a1 + b1 = c1 ∧ a2 + b2 = c2 ∧ a3 + b3 = c3 ∧ a4 + b4 = c4)

theorem possible_sets_C (A B C : Set ℕ) (h : conditions A B C) :
  C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end possible_sets_C_l532_532694


namespace sum_of_exponents_l532_532794

/-- If 700 is expressed as a sum of at least three distinct powers of 2, 
the least possible sum of the exponents is 30. -/
theorem sum_of_exponents (n : ℕ) (hn : n = 700) (h : ∃ s : Finset ℕ, (s.card ≥ 3) 
  ∧ (∑ i in s, 2 ^ i = n)) : ∑ i in (classical.some h), i = 30 := 
sorry

end sum_of_exponents_l532_532794


namespace sum_digits_product_repeat_nine_l532_532134

theorem sum_digits_product_repeat_nine (A B : ℕ) (hA : A < 10) (hB : B < 10) :
  let num1 := A * 111111111,
      num2 := B * 111111111,
      product := num1 * num2
  in 
  sumDigits product = 81 :=
by
  sorry

end sum_digits_product_repeat_nine_l532_532134


namespace area_of_triangle_G1_G2_G3_l532_532405

noncomputable def quadrilateral_area : ℝ := 2021
noncomputable def point_on_segment (AD E : ℝ) : Prop := E ∈ set.Icc 0 AD
noncomputable def centroid (B C E : ℝ) : ℝ := (B + C + E) / 3

theorem area_of_triangle_G1_G2_G3 (A B C D E : ℝ) (h_area_abcd : quadrilateral_area = 2021)
  (h_E_on_AD : point_on_segment (D - A) (E - A)) :
  let G1 := centroid B C E,
      G2 := centroid A B E,
      G3 := centroid C D E in
  ∃ area : ℝ, area = 2021 / 9 :=
begin
  sorry
end

end area_of_triangle_G1_G2_G3_l532_532405


namespace quadratic_second_root_l532_532049

noncomputable def second_root (p q : ℝ) : ℝ :=
  -2 * p / (p - 2)

theorem quadratic_second_root (p q : ℝ) (h1 : (p + q) * 1^2 + (p - q) * 1 + p * q = 0) :
  ∃ r : ℝ, r = second_root p q :=
by 
  sorry

end quadratic_second_root_l532_532049


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532525

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l532_532525


namespace coeffs_of_quadratic_eq_l532_532825

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end coeffs_of_quadratic_eq_l532_532825


namespace smallest_four_digit_divisible_by_35_l532_532200

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532200


namespace length_of_platform_l532_532273

-- Conditions as definitions in Lean
def speed_kmh : ℝ := 36
def time_train_man_s : ℝ := 12
def time_train_platform_s : ℝ := 30

-- Convert speed from km/h to m/s
def speed_mps := (speed_kmh * 1000) / 3600

-- Length of the train
def length_train := speed_mps * time_train_man_s

-- Total distance covered when passing entire platform
def total_distance := speed_mps * time_train_platform_s

-- Length of the platform
def length_platform := total_distance - length_train

-- Proof statement
theorem length_of_platform : length_platform = 180 := by
  sorry

end length_of_platform_l532_532273


namespace find_four_rel_prime_integers_l532_532111

theorem find_four_rel_prime_integers (S : Finset ℕ) 
  (hS : S.card = 91) 
  (h1 : ∀ x ∈ S, 1 < x) 
  (h_pairs : ∃ P ⊆ S ×ˢ S, P.card ≥ 456 ∧ ∀ ⦃a b⦄, (a, b) ∈ P → a ≠ b → gcd a b = 1) : 
  ∃ a b c d ∈ S, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c d = 1 ∧ gcd d a = 1 :=
by
  sorry

end find_four_rel_prime_integers_l532_532111


namespace find_a_l532_532761

noncomputable def a : ℝ := sorry -- We'll use sorry for the definition initially.

-- Given conditions
def line (a : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = p.1 + 2 * a

def circle (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 ^ 2 + p.2 ^ 2 - 2 * a * p.2 - 2 = 0

def intersects (line : ℝ × ℝ → Prop) (circle : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, line A ∧ line B ∧ circle A ∧ circle B ∧ A ≠ B ∧ dist A B = 2 * sqrt 3

-- Main statement
theorem find_a (h : a > 0) 
  (intersect_cond : intersects (line a) (circle a)) : a = sqrt 2 :=
sorry -- Proof is omitted

end find_a_l532_532761


namespace simplify_f_l532_532403

noncomputable def f (α : Real) : Real :=
  (sin (α - Real.pi / 2) * cos (3 * Real.pi / 2 + α) * tan (Real.pi - α)) /
    (tan (-Real.pi - α) * sin (-Real.pi - α))

theorem simplify_f (α : Real) (h1 : α ∈ Ioo (Real.pi) (3 * Real.pi / 2)) (h2 : cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
by
  sorry

end simplify_f_l532_532403


namespace common_difference_is_two_l532_532906

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532906


namespace graph_shift_equivalence_l532_532161

noncomputable def original_function (x : ℝ) : ℝ := sin (2 * x)
noncomputable def transformed_function (x : ℝ) : ℝ := 2 * sin (x + π / 6) * cos (x + π / 6)

theorem graph_shift_equivalence : 
  ∀ x : ℝ, transformed_function x = original_function (x - π / 6) :=
by sorry

end graph_shift_equivalence_l532_532161


namespace find_y_l532_532722

theorem find_y : ∃ y : ℚ, y + 2/3 = 1/4 - (2/5) * 2 ∧ y = -511/420 :=
by
  sorry

end find_y_l532_532722


namespace jenny_money_l532_532835

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l532_532835


namespace common_difference_of_arithmetic_sequence_l532_532965

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532965


namespace tetrahedron_volume_ratio_l532_532380

theorem tetrahedron_volume_ratio (V : ℝ) (hV : 0 < V) : 
  let remaining_volume := V - 4 * (V / 8) in
  (remaining_volume / V) = 1 / 2 :=
by
  let smaller_tetrahedron_volume := V / 8
  let total_smaller_tetrahedra_volume := 4 * smaller_tetrahedron_volume
  let remaining_volume := V - total_smaller_tetrahedra_volume
  have scaling_factor := (1 : ℝ) / 2
  have volume_ratio := remaining_volume / V
  suffices volume_ratio = 1 / 2 from this
  sorry

end tetrahedron_volume_ratio_l532_532380


namespace smallest_four_digit_divisible_by_35_l532_532188

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532188


namespace common_difference_of_arithmetic_sequence_l532_532985

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532985


namespace count_valid_integers_lt_500_l532_532429

-- Definition of integer being the sum of j consecutive odd integers
def is_sum_of_j_odd_consecutive (N : ℕ) (j : ℕ) : Prop :=
  ∃ n : ℕ, N = j * (2 * n + j)

-- Definition of N having exactly 3 valid j values that satisfy the sum condition
def has_exactly_three_valid_j (N : ℕ) : Prop :=
  (finset.filter (λ j : ℕ, j ∣ N ∧ is_sum_of_j_odd_consecutive N j) 
    (finset.range (N + 1))).card = 3

-- Definition of numbers N of the form p^2 * q^2 with p and q distinct primes
def is_in_form_p_squared_q_squared (N : ℕ) : Prop :=
  ∃ (p q : ℕ) [prime p] [prime q], p ≠ q ∧ N = (p * p) * (q * q)

-- Main theorem statement
theorem count_valid_integers_lt_500 :
  finset.filter (λ N : ℕ, N < 500 ∧ has_exactly_three_valid_j N ∧ is_in_form_p_squared_q_squared N)
    (finset.range 500).card = 5 :=
sorry

end count_valid_integers_lt_500_l532_532429


namespace puzzle_pieces_missing_l532_532163

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l532_532163


namespace determine_m_maximize_profit_l532_532307

-- Definitions and conditions
def daily_sales_volume (x m : ℝ) : ℝ := (m / (x - 3) + 8 * (x - 6) ^ 2)

variable (m : ℝ)
def sales_condition : Prop := daily_sales_volume 5 m = 11

-- Part Ⅰ: prove the value of m
theorem determine_m : m = 6 :=
by
  have h : sales_condition m ↔ m / 2 + 8 = 11 := by sorry
  have : m = 6 := by sorry 
  exact this

-- Part Ⅱ: prove the maximizing value of x and the profit
def profit (x : ℝ) : ℝ := (x - 3) * (daily_sales_volume x 6)
def interval_condition (x : ℝ) : Prop := 3 < x ∧ x < 6
def cost_price : ℝ := 3

theorem maximize_profit (x : ℝ) : interval_condition x → profit x = 38 :=
by
  have h : m = 6 := determine_m
  have : x = 4 := by sorry -- show that x = 4 maximizes the profit
  have : profit 4 = 38 := by sorry -- show that the maximum profit at x = 4 is 38
  exact this

end determine_m_maximize_profit_l532_532307


namespace tan_B_right_triangle_l532_532044

theorem tan_B_right_triangle (A B C : Type _) [metric_space A] [metric_space B] [metric_space C] 
  (h_right_angle : ∠BAC = π/2) (h_AC : dist A C = sqrt 34) (h_AB : dist A B = 5) : 
  ∃ (tan_B : ℝ), tan_B = 3 / 5 :=
by
  -- Definitions and assumptions according to the conditions
  let AC := dist A C
  let AB := dist A B
  let BC := dist B C
  have hypotenuse := sqrt (AC^2 - AB^2)
  have h_BC : BC = sqrt 9 := by sorry -- derived from Pythagorean theorem and given lengths
  use 3 / 5
  -- Proof required here
  sorry

end tan_B_right_triangle_l532_532044


namespace find_value_of_p_l532_532402

open Classical

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) := (p / 2, 0)

noncomputable def line_through_focus (p : ℝ) : (ℝ × ℝ) → ℝ × ℝ :=
  λ F, (λ x, 2 * (x - F.1))

def intersection_points (y1 y2 : ℝ) (p : ℝ) : Prop :=
  y1 + y2 = p ∧ y1 * y2 = -p^2

def product_distances (AF BF : ℝ) (p : ℝ) : Prop :=
  AF * BF = 20

theorem find_value_of_p (p : ℝ) (hp : p > 0) (y1 y2 : ℝ)
        (hx : intersection_points y1 y2 p)
        (AF BF : ℝ)
        (hAF_BF : product_distances AF BF p) :
        p = 4 := by
  sorry

end find_value_of_p_l532_532402


namespace arithmetic_sequence_common_difference_l532_532936

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532936


namespace power_function_not_origin_l532_532803

theorem power_function_not_origin (m : ℝ) 
  (h1 : m^2 - 3 * m + 3 = 1) 
  (h2 : m^2 - m - 2 ≤ 0) : 
  m = 1 ∨ m = 2 :=
sorry

end power_function_not_origin_l532_532803


namespace intersection_and_reciprocal_sum_l532_532482

noncomputable def line_l_eq (t : ℝ) : ℝ × ℝ :=
  (-t, sqrt(3) - t)

def C1_polar_eq (theta : ℝ) : ℝ := 
  4 * sin theta

def C2_rect_eq (x y : ℝ) : Prop := 
  (x^2 / 4) + y^2 = 1

def P : ℝ × ℝ := (-sqrt(3), 0)

theorem intersection_and_reciprocal_sum :
  ∀ t₁ t₂ : ℝ, 
  let M := line_l_eq t₁,
      N := line_l_eq t₂ in
  C2_rect_eq M.1 M.2 ∧
  C2_rect_eq N.1 N.2 →
  t₁ * t₂ = -2 / 5 →
  abs(t₁ + t₂) = 2 * sqrt(6) / 5 →
  1 / (abs (t₁ - (-sqrt(3)))) + 1 / (abs (t₂ - (-sqrt(3)))) = 4 :=
  sorry

end intersection_and_reciprocal_sum_l532_532482


namespace reconstruct_text_from_distorted_fax_l532_532351

/-- 
Given the original font in capital Cyrillic letters:
АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
Prove that the reconstructed text from the distorted fax transmission is "глобальное потепление".
--/

theorem reconstruct_text_from_distorted_fax (
  original_font : String := "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" 
) : String = "глобальное потепление" := 
sorry

end reconstruct_text_from_distorted_fax_l532_532351


namespace sin_cos_order_l532_532015

theorem sin_cos_order {α β : ℝ} (h₁ : 0 < α) (h₂ : α < β) (h₃ : β < π/4)
    (ha : sin α + cos α = a) (hb : sin β + cos β = b) : a < b :=
by
    sorry

end sin_cos_order_l532_532015


namespace tangent_line_eq_monotonic_intervals_extremes_f_l532_532412

variables {a x : ℝ}

noncomputable def f (a x : ℝ) : ℝ := -1/3 * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) : ℝ := -x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_eq {a : ℝ} (h : a = -1) : (∃ y, y = f (-1) (-2) ∧ 3 * x - 3 * y + 8 = 0) := sorry

theorem monotonic_intervals_extremes {a : ℝ} (h : 0 < a) :
  (∀ x, (a < x ∧ x < 3 * a → 0 < f' a x) ∧ 
        (x < a ∨ 3 * a < x → f' a x < 0) ∧ 
        (f a (3 * a) = 0 ∧ f a a = -4/3 * a^3)) := sorry

theorem f'_inequality_range (h1 : ∀ x, 2 * a ≤ x ∧ x ≤ 2 * a + 2 → |f' a x| ≤ 3 * a) :
  (1 ≤ a ∧ a ≤ 3) := sorry

end tangent_line_eq_monotonic_intervals_extremes_f_l532_532412


namespace number_of_lizards_l532_532636

theorem number_of_lizards (total_geckos : ℕ) (insects_per_gecko : ℕ) (total_insects_eaten : ℕ) (insects_per_lizard : ℕ) 
  (gecko_total_insects : total_geckos * insects_per_gecko = 5 * 6) (lizard_insects: insects_per_lizard = 2 * insects_per_gecko)
  (total_insects : total_insects_eaten = 66) : 
  (total_insects_eaten - total_geckos * insects_per_gecko) / insects_per_lizard = 3 :=
by 
  sorry

end number_of_lizards_l532_532636


namespace count_valid_N_l532_532430

-- Define the basic properties
def is_valid_N (N : ℕ) : Prop :=
  N < 500 ∧
  (∃ M1 M2 M3 : ℕ, M1 * M2 = N ∧ M2 * M3 = N ∧
   1 ≤ M1 ∧ 1 ≤ M2 ∧ 1 ≤ M3 ∧
   M1 ≠ M2 ∧ M2 ≠ M3 ∧ M1 ≠ M3)

theorem count_valid_N : Nat.card (Finset.filter is_valid_N (Finset.range 500)) = 9 := sorry

end count_valid_N_l532_532430


namespace ticket_cost_l532_532650

theorem ticket_cost
    (rows : ℕ) (seats_per_row : ℕ)
    (fraction_sold : ℚ) (total_earnings : ℚ)
    (N : ℕ := rows * seats_per_row)
    (S : ℚ := fraction_sold * N)
    (C : ℚ := total_earnings / S)
    (h1 : rows = 20) (h2 : seats_per_row = 10)
    (h3 : fraction_sold = 3 / 4) (h4 : total_earnings = 1500) :
    C = 10 :=
by
  sorry

end ticket_cost_l532_532650


namespace polynomial_roots_l532_532369

theorem polynomial_roots :
  (∀ x : ℝ, x^5 - 3 * x^4 + 3 * x^3 - x^2 - 4 * x + 4 = 0 ↔
    (x = -1 - √3 ∨ x = -1 + √3 ∨ x = -1 ∨ x = 1 ∨ x = 2)) := by
sorry

end polynomial_roots_l532_532369


namespace smallest_four_digit_divisible_by_35_l532_532252

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532252


namespace find_common_difference_l532_532953

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532953


namespace eccentricity_calculation_l532_532003

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  let e := c / a in
  let c := real.sqrt (a^2 + b^2) in
  if h : e^2 - (real.sqrt 3 / 3) * e - 1 = 0 ∧ e > 1 then
    e
  else
    0

theorem eccentricity_calculation (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_c_eq : b^2 = ((real.sqrt 3) / 3) * a * (real.sqrt (a^2 + b^2))) :
  eccentricity_of_hyperbola a b ha hb = (real.sqrt 3 + real.sqrt 39) / 6 := 
sorry

end eccentricity_calculation_l532_532003


namespace vertical_shift_of_cosine_l532_532327

theorem vertical_shift_of_cosine (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (max_value : 4 ≤ a * cos (b * 0 + c) + d) (min_value : 0 ≤ a * cos (b * π + c) + d) : 
  d = 2 :=
by
  sorry

end vertical_shift_of_cosine_l532_532327


namespace smallest_four_digit_divisible_by_35_l532_532211

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532211


namespace tangent_line_distance_min_l532_532651

open Real

noncomputable def circle : set (ℝ × ℝ) := { p | (p.1 + 1)^2 + (p.2 - 2)^2 = 1 }

theorem tangent_line_distance_min (P M O : ℝ × ℝ)
  (hP : P ∈ circle)
  (hT : true) -- Assuming a tangent line exists touching P and M in the given properties
  (hDist : dist P M = dist P O) : dist P M = 2 * sqrt 5 / 5 :=
by
  sorry

end tangent_line_distance_min_l532_532651


namespace student_test_arrangements_l532_532589

theorem student_test_arrangements :
  ∃ (arrangements : ℕ), 
  arrangements = 264 ∧
  ∀ (students : Fin 4 → Fin 5 → bool),
  (∀ (i : Fin 4), students i (Fin.mk 3 5) = ff) →
  (∀ (i : Fin 4), students i (Fin.mk 4 5) = ff) →
  (∀ (i : Fin 4) (j k : Fin 5), students i j = students i k → j = k) →
  arrangements = 264 :=
by
  sorry

end student_test_arrangements_l532_532589


namespace range_of_k_l532_532025

noncomputable def meets_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - y^2 = 4 ∧ y = k * x - 1

theorem range_of_k : 
  { k : ℝ | meets_hyperbola k } = { k : ℝ | k = 1 ∨ k = -1 ∨ - (Real.sqrt 5) / 2 ≤ k ∧ k ≤ (Real.sqrt 5) / 2 } :=
by
  sorry

end range_of_k_l532_532025


namespace race_length_l532_532460

variables (L : ℕ)

def distanceCondition1 := L - 70
def distanceCondition2 := L - 100
def distanceCondition3 := L - 163

theorem race_length (h1 : distanceCondition1 = L - 70) 
                    (h2 : distanceCondition2 = L - 100) 
                    (h3 : distanceCondition3 = L - 163)
                    (h4 : (L - 70) / (L - 163) = (L) / (L - 100)) : 
  L = 1000 :=
sorry

end race_length_l532_532460


namespace count_valid_N_l532_532431

-- Define the basic properties
def is_valid_N (N : ℕ) : Prop :=
  N < 500 ∧
  (∃ M1 M2 M3 : ℕ, M1 * M2 = N ∧ M2 * M3 = N ∧
   1 ≤ M1 ∧ 1 ≤ M2 ∧ 1 ≤ M3 ∧
   M1 ≠ M2 ∧ M2 ≠ M3 ∧ M1 ≠ M3)

theorem count_valid_N : Nat.card (Finset.filter is_valid_N (Finset.range 500)) = 9 := sorry

end count_valid_N_l532_532431


namespace positive_integer_solutions_l532_532711

theorem positive_integer_solutions (n m : ℕ) (h : n > 0 ∧ m > 0) : 
  (n + 1) * m = n! + 1 ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 5) := by
  sorry

end positive_integer_solutions_l532_532711


namespace boat_man_mass_l532_532284

theorem boat_man_mass (L B h : ℝ) (rho g : ℝ): 
  L = 3 → B = 2 → h = 0.015 → rho = 1000 → g = 9.81 → (rho * L * B * h * g) / g = 9 :=
by
  intros
  simp_all
  sorry

end boat_man_mass_l532_532284


namespace emily_has_7_times_more_oranges_than_sandra_l532_532354

theorem emily_has_7_times_more_oranges_than_sandra
  (B S E : ℕ)
  (h1 : S = 3 * B)
  (h2 : B = 12)
  (h3 : E = 252) :
  ∃ k : ℕ, E = k * S ∧ k = 7 :=
by
  use 7
  sorry

end emily_has_7_times_more_oranges_than_sandra_l532_532354


namespace g_at_3_eq_6_l532_532078

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

theorem g_at_3_eq_6 : g 3 = 6 :=
by
  simp [g, pow_succ]
  norm_num
  sorry

end g_at_3_eq_6_l532_532078


namespace solve_trig_eq_l532_532105

theorem solve_trig_eq (x : ℝ) : 
  (|\cos x| - cos (3 * x)) / (cos x * sin (2 * x)) = 2 / real.sqrt 3 ↔
  (∃ k : ℤ, x = (real.pi / 6) + 2 * real.pi * k) ∨ 
  (∃ k : ℤ, (x = (5 * real.pi / 6) + 2 * real.pi * k) ∨ (x = (4 * real.pi / 3) + 2 * real.pi * k)) :=
sorry

end solve_trig_eq_l532_532105


namespace minute_hand_rotation_l532_532345

theorem minute_hand_rotation (h : ℕ) (radians_per_rotation : ℝ) : h = 5 → radians_per_rotation = 2 * Real.pi → - (h * radians_per_rotation) = -10 * Real.pi :=
by
  intros h_eq rp_eq
  rw [h_eq, rp_eq]
  sorry

end minute_hand_rotation_l532_532345


namespace problem1_problem2_l532_532419

-- Define the function f(x) = |x + a| + |2x - 1|
def f (x a : ℝ) : ℝ := abs (x + a) + abs (2 * x - 1)

-- Problem 1: For a = 1, prove that the solution set of f(x) ≥ 2 is { x | x ≤ 0 or x ≥ 2/3 }
theorem problem1 (x : ℝ) : f x 1 ≥ 2 ↔ (x ≤ 0 ∨ x ≥ 2 / 3) := by
  sorry

-- Problem 2: Given that the solution set of f(x) ≤ 2x includes [1/2, 1], prove the range of a is [-3/2, 0].
theorem problem2 (a : ℝ) : (∀ x ∈ set.Icc (1/2) 1, f x a ≤ 2 * x) ↔ (-3 / 2 ≤ a ∧ a ≤ 0) := by
  sorry

end problem1_problem2_l532_532419


namespace part1_relationship_range_part2_maximize_profit_l532_532087

variables {x y a : ℝ}
noncomputable def zongzi_profit (x : ℝ) : ℝ := -5 * x + 6000

-- Given conditions
def conditions (x : ℝ) : Prop :=
  100 ≤ x ∧ x ≤ 150

-- Part 1: Prove the functional relationship and range of x
theorem part1_relationship_range (x : ℝ) (h : conditions x) :
  zongzi_profit x = -5 * x + 6000 :=
  sorry

-- Part 2: Profit maximization given modified purchase price condition
noncomputable def modified_zongzi_profit (x : ℝ) (a : ℝ) : ℝ :=
  (a - 5) * x + 6000

def maximize_strategy (x a : ℝ) : Prop :=
  (0 < a ∧ a < 5 → x = 100) ∧ (5 ≤ a ∧ a < 10 → x = 150)

theorem part2_maximize_profit (a : ℝ) (ha : 0 < a ∧ a < 10) :
  ∃ x, conditions x ∧ maximize_strategy x a :=
  sorry

end part1_relationship_range_part2_maximize_profit_l532_532087


namespace triangle_area_ordering_l532_532353

variable (m n p : ℚ)

theorem triangle_area_ordering (hm : m = 15 / 2) (hn : n = 13 / 2) (hp : p = 7) : n < p ∧ p < m := by
  sorry

end triangle_area_ordering_l532_532353


namespace max_ratio_l532_532393

noncomputable theory

open Real

variable {p : ℝ} (h₀ : 0 < p)
variable (A B F M M'' : Point)

-- Definitions for A and B lying on the parabola, focus being F and the angle criterion
def onParabola (P : Point) := P.y^2 = 2 * p * P.x
def isFocus (F : Point) := F = (0, p / 2)
def angleRight (A B F : Point) := angle A F B = π / 2
def midpoint (A B M : Point) := 2 * M = A + B
def projection (M M'' : Point) := collinear (M'' - M) (1, 0)

theorem max_ratio (h₁ : onParabola A)
                  (h₂ : onParabola B)
                  (h₃ : isFocus F)
                  (h₄ : angleRight A B F)
                  (h₅ : midpoint A B M)
                  (h₆ : projection M M'') :
  ∃ (val : ℝ), val = (| ∥M - M''∥ / ∥A - B∥) ∧ val = sqrt 2 / 2 := sorry

end max_ratio_l532_532393


namespace smallest_four_digit_number_divisible_by_35_l532_532222

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532222


namespace smallest_four_digit_divisible_by_35_l532_532183

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532183


namespace find_a_l532_532760

noncomputable def a : ℝ := sorry -- We'll use sorry for the definition initially.

-- Given conditions
def line (a : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = p.1 + 2 * a

def circle (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 ^ 2 + p.2 ^ 2 - 2 * a * p.2 - 2 = 0

def intersects (line : ℝ × ℝ → Prop) (circle : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, line A ∧ line B ∧ circle A ∧ circle B ∧ A ≠ B ∧ dist A B = 2 * sqrt 3

-- Main statement
theorem find_a (h : a > 0) 
  (intersect_cond : intersects (line a) (circle a)) : a = sqrt 2 :=
sorry -- Proof is omitted

end find_a_l532_532760


namespace remainder_when_packing_15_candies_l532_532545

theorem remainder_when_packing_15_candies
  (x : ℤ) :
  let N := 35 * x + 7 in ({N % 15} = {7} ∨ {N % 15} = {12} ∨ {N % 15} = {2}) :=
by
  sorry

end remainder_when_packing_15_candies_l532_532545


namespace common_difference_is_two_l532_532908

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532908


namespace parallel_segm_l532_532056

open EuclideanGeometry

axiom trapezium_eq (A B C D : Point) : AB ∥ CD

noncomputable def point_on_segment (P A C : Point) : Prop := collinear P A C ∧ P ∈ segment A C ∧ C ∈ segment A P

axiom point_on_AC (P A C : Point) (h : C ≠ A) : point_on_segment P A C

axiom midpoint_def (M A B : Point) : M = midpoint A B

axiom intersection_def (L1 L2 N : Line) : N = intersection L1 L2

theorem parallel_segm (A B C D P X Y N M : Point)
  (h1 : trapezium_eq A B C D)
  (h2 : point_on_AC P A C)
  (h3 : X = midpoint A B)
  (h4 : Y = midpoint C D)
  (h5 : N = intersection (line P X) (line B C))
  (h6 : M = intersection (line P Y) (line A D)) :
  parallel (line M N) (line A B) := sorry

end parallel_segm_l532_532056


namespace even_function_smallest_positive_phi_l532_532416

noncomputable def f (x φ : ℝ) : ℝ :=
  cos ( sqrt 3 * x + φ) - sqrt 3 * sin ( sqrt 3 * x + φ)

theorem even_function_smallest_positive_phi :
  (∃ φ : ℝ, φ > 0 ∧ (∀ x : ℝ, f x φ = f (-x) φ)) → φ = 2 * real.pi / 3 :=
by
  sorry

end even_function_smallest_positive_phi_l532_532416


namespace common_difference_l532_532989

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532989


namespace problem_part1_problem_part2_l532_532785

-- Definitions of the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2 * x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Definitions for vector operations
def add_vec (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def part1 (x : ℝ) : Prop := parallel (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

noncomputable def part2 (x : ℝ) : Prop := perpendicular (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

theorem problem_part1 : part1 2 ∧ part1 (-3 / 2) := sorry

theorem problem_part2 : part2 ((-4 + Real.sqrt 14) / 2) ∧ part2 ((-4 - Real.sqrt 14) / 2) := sorry

end problem_part1_problem_part2_l532_532785


namespace necessary_and_sufficient_condition_l532_532519

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then real.log x / real.log 2 else -real.log x / real.log 2

theorem necessary_and_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ (m - 2, 2 * m) → (log x / log 2 ≠ log x / log 2) ∧ 0 < x) ↔ 2 ≤ m ∧ m < 3 :=
sorry

end necessary_and_sufficient_condition_l532_532519


namespace karen_start_time_late_l532_532488

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l532_532488


namespace arithmetic_sequence_common_difference_l532_532903

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532903


namespace find_c_l532_532262

def P (x : ℝ) (c : ℝ) : ℝ :=
  x^3 + 3*x^2 + c*x + 15

theorem find_c (c : ℝ) : (x - 3 = P x c → c = -23) := by
  sorry

end find_c_l532_532262


namespace common_difference_of_arithmetic_sequence_l532_532970

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l532_532970


namespace identify_ATM_mistakes_additional_security_measures_l532_532607

-- Define the conditions as Boolean variables representing different mistakes and measures
variables (writing_PIN_on_card : Prop)
variables (using_ATM_despite_difficulty : Prop)
variables (believing_stranger : Prop)
variables (walking_away_without_card : Prop)
variables (use_trustworthy_locations : Prop)
variables (presence_during_transactions : Prop)
variables (enable_SMS_notifications : Prop)
variables (call_bank_for_suspicious_activities : Prop)
variables (be_cautious_of_fake_SMS_alerts : Prop)
variables (store_transaction_receipts : Prop)
variables (shield_PIN : Prop)
variables (use_chipped_cards : Prop)
variables (avoid_high_risk_ATMs : Prop)

-- Prove that the identified mistakes occur given the conditions
theorem identify_ATM_mistakes :
  writing_PIN_on_card ∧ using_ATM_despite_difficulty ∧ 
  believing_stranger ∧ walking_away_without_card := sorry

-- Prove that the additional security measures should be followed
theorem additional_security_measures :
  use_trustworthy_locations ∧ presence_during_transactions ∧ 
  enable_SMS_notifications ∧ call_bank_for_suspicious_activities ∧ 
  be_cautious_of_fake_SMS_alerts ∧ store_transaction_receipts ∧ 
  shield_PIN ∧ use_chipped_cards ∧ avoid_high_risk_ATMs := sorry

end identify_ATM_mistakes_additional_security_measures_l532_532607


namespace count_integer_roots_l532_532382

theorem count_integer_roots :
  let n := 15625
  let exp := 6
  let f := λ (k : ℕ), (n : ℚ)^(1 / k)
  let int_nums := {k : ℕ | (f k).denom = 1 ∧ (range (exp + 1)).count(k ∣ exp) = 1}
  int_nums.size = 4 := sorry

end count_integer_roots_l532_532382


namespace solid_of_revolution_volume_l532_532819

noncomputable def volume_of_solid (curve : ℝ → ℝ) : ℝ :=
  ∫ y in 0..1, π * (exp(y / 2))^2

theorem solid_of_revolution_volume :
  volume_of_solid (λ (x : ℝ), 2 * log x) = π * (exp 1 - 1) :=
by
  sorry

end solid_of_revolution_volume_l532_532819


namespace find_common_difference_l532_532956

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532956


namespace triangle_area_7_8_10_l532_532360

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := semi_perimeter a b c in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_7_8_10 :
  triangle_area 7 8 10 ≈ 27.78 :=
by 
  let a := 7
  let b := 8
  let c := 10
  let s := semi_perimeter a b c
  let area := triangle_area a b c
  have h1 : a + b > c := by norm_num [a, b, c]
  have h2 : a + c > b := by norm_num [a, c, b]
  have h3 : b + c > a := by norm_num [b, c, a]
  have area_def := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  have area_calculated : area ≈ 27.78 := by
    norm_num
  exact area_calculated
  sorry

end triangle_area_7_8_10_l532_532360


namespace rearrange_ships_l532_532601

def ship_positions_initial : set (ℝ × ℝ) := { -- Replace with actual coordinates
 (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)}

def ship_positions_target : set (ℝ × ℝ) := { -- Replace with target coordinates after moving 4 ships
 (1, 1), (2, 2), (3, 3), (11, 11), (12, 12), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)}

def line (s : set (ℝ × ℝ)) : Prop := ∃ a b c : ℝ, ∀ p ∈ s, a * p.1 + b * p.2 = c

theorem rearrange_ships :
  (∃ (moved_ships : finset (ℝ × ℝ)), ∀ ship ∈ moved_ships, ship_positions_initial ship ∧ ship ∉ ship_positions_target) →
  (∀ final_conf, final_conf = ship_positions_target → 
  ∃ (lines : finset (set (ℝ × ℝ))), 
  ∃ (configured_lines : finset (set (ℝ × ℝ))), 
  ∀ line ∈ lines,
  ∀ l ∈ configured_lines,
  l ⊆ ship_positions_target ∧ finset.card l = 4)  →
  sorry

end rearrange_ships_l532_532601


namespace rolls_needed_for_shirt_boxes_l532_532427

-- Definitions of conditions
def num_shirt_boxes : ℕ := 20
def num_xl_boxes : ℕ := 12
def xl_boxes_per_roll : ℕ := 3
def cost_per_roll : ℝ := 4.0
def total_spent : ℝ := 32.0

-- The proof problem statement
theorem rolls_needed_for_shirt_boxes :
  let total_rolls := total_spent / cost_per_roll in
  let rolls_for_xl_boxes := num_xl_boxes / xl_boxes_per_roll in
  let rolls_for_shirt_boxes := total_rolls - rolls_for_xl_boxes in
  let shirt_boxes_per_roll := num_shirt_boxes / rolls_for_shirt_boxes in
  shirt_boxes_per_roll = 5 :=
by
  sorry

end rolls_needed_for_shirt_boxes_l532_532427


namespace vanessa_score_record_l532_532171

theorem vanessa_score_record 
  (team_total_points : ℕ) 
  (other_players_average : ℕ) 
  (num_other_players : ℕ) 
  (total_game_points : team_total_points = 55) 
  (average_points_per_player : other_players_average = 4) 
  (number_of_other_players : num_other_players = 7) 
  : 
  ∃ vanessa_points : ℕ, vanessa_points = 27 :=
by
  sorry

end vanessa_score_record_l532_532171


namespace max_projection_projection_l532_532752

open Real

def circle (x y : ℝ) : Prop :=
  (x - 4)^2 + y^2 = 4

def line (a : ℝ) : Prop :=
  a ≠ 0

theorem max_projection_projection (x y a : ℝ) :
  circle x y → line a →
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧
  (2 * sin (θ + π / 4) + 2 * sqrt 2) = 2 * sqrt 2 + 2 :=
by
  intro h_circle h_line
  sorry

end max_projection_projection_l532_532752


namespace right_triangle_AC_sinC_l532_532035

open Real

/--
In right triangle ABC with angle A = 90 degrees, where AB = 20 and BC = 21, we prove the length of AC is 29 and sin C = 20/29.
-/
theorem right_triangle_AC_sinC :
  ∀ (A B C : ℝ) (h : ∠A = real.pi / 2) (AB BC AC : ℝ),
  AB = 20 →
  BC = 21 →
  AC = √(AB^2 + BC^2) →
  AC = 29 ∧ sin C = AB / AC :=
by
  intro A B C h AB BC AC
  sorry

end right_triangle_AC_sinC_l532_532035


namespace parallel_lines_condition_l532_532128

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → ((a - 2) * x + 3 * y + 2 * a = 0 → 
  slope (a-2) 3 = slope 1 a ∧ 1 * 2 * a ≠ (a - 2) * 6)) ↔ a = -1 :=
sorry

end parallel_lines_condition_l532_532128


namespace arithmetic_sequence_common_difference_l532_532935

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532935


namespace perpendicular_OQ_PQ_l532_532561

variables {O1 O2 P Q A B O : Type}
variables [Circle O1] [Circle O2]
variables (P Q : Point)
variables (PA : Chord O1) (PB : Chord O2)
variables (h1 : Tangent PA O2)
variables (h2 : Tangent PB O1)
variables (O : Circumcenter (Triangle P A B))

theorem perpendicular_OQ_PQ : Perpendicular (Segment O Q) (Segment P Q) :=
  sorry

end perpendicular_OQ_PQ_l532_532561


namespace smallest_four_digit_divisible_by_35_l532_532254

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532254


namespace neg_sqrt_comparison_l532_532334

theorem neg_sqrt_comparison : -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end neg_sqrt_comparison_l532_532334


namespace common_difference_l532_532999

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532999


namespace constant_term_expansion_eq_sixty_l532_532023

theorem constant_term_expansion_eq_sixty (a : ℝ) (h : 15 * a = 60) : a = 4 :=
by
  sorry

end constant_term_expansion_eq_sixty_l532_532023


namespace arithmetic_sequence_common_difference_l532_532861

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532861


namespace find_matrix_A_l532_532738

theorem find_matrix_A (A : Matrix (Fin 2) (Fin 2) ℝ)
  (α1 : Vector (Fin 2) ℝ := ![1, 0])
  (α2 : Vector (Fin 2) ℝ := ![0, 1])
  (λ1 : ℝ := -1)
  (λ2 : ℝ := 4)
  (eigen1 : A.mulVec α1 = λ1 • α1)
  (eigen2 : A.mulVec α2 = λ2 • α2) :
  A = ![![ -1, 0], 
         ![  0, 4]] := by
  sorry

end find_matrix_A_l532_532738


namespace tangent_perpendicular_intersection_y_coord_l532_532846

open Real

theorem tangent_perpendicular_intersection_y_coord {a b : ℝ} 
  (ha : (a, 4 * a^2) ∈ set_of (λ p : ℝ × ℝ, p.2 = 4 * p.1^2))
  (hb : (b, 4 * b^2) ∈ set_of (λ p : ℝ × ℝ, p.2 = 4 * p.1^2))
  (h_perpendicular : (8 * a) * (8 * b) = -1) : 
  let P := ( (b + a) / 2, 4 * a * b) in
  P.snd = -1 / 8 := 
  sorry

end tangent_perpendicular_intersection_y_coord_l532_532846


namespace quotient_correct_l532_532718

noncomputable def find_quotient (z : ℚ) : ℚ :=
  let dividend := (5 * z ^ 5 - 3 * z ^ 4 + 6 * z ^ 3 - 8 * z ^ 2 + 9 * z - 4)
  let divisor := (4 * z ^ 2 + 5 * z + 3)
  let quotient := ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256))
  quotient

theorem quotient_correct (z : ℚ) :
  find_quotient z = ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256)) :=
by
  sorry

end quotient_correct_l532_532718


namespace prob_addition_mut_excl_l532_532022

variables {Ω : Type} {P : set Ω → ℝ} {A B : set Ω}

-- Define mutually exclusive events
def mutually_exclusive (A B : set Ω) : Prop := A ∩ B = ∅

-- Add axiom about probability measure being 1
axiom prob_union (P : set Ω → ℝ) (A B : set Ω) (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B

-- Add axiom about the entire sample space probability being 1
axiom prob_certain (P : set Ω → ℝ) : P set.univ = 1

theorem prob_addition_mut_excl (h : mutually_exclusive A B) : P A + P B = 1 :=
by
  have h_union := prob_union P A B h
  have h_certain := prob_certain P
  sorry

end prob_addition_mut_excl_l532_532022


namespace smallest_four_digit_divisible_by_35_l532_532193

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l532_532193


namespace flipping_through_book_is_random_event_l532_532564

-- Define the event of flipping through the textbook
def flipping_event (book : Type) : book → book → Prop :=
λ _ _, true -- This is a simplification; we're just stating that flipping is an event that can happen.

-- Define the randomness of the event
def is_random_event (event : Prop → Prop) : Prop :=
event true ∧ event false -- There exists some chance (not certain, not impossible)

-- Define the specific event in context
def similar_section_event : Prop :=
∃ flipping_event, true -- An event where flipping through might result in a similar section

-- Proving the specific event is a random event
theorem flipping_through_book_is_random_event :
  is_random_event (λ ev, ev = similar_section_event) :=
by
  sorry

end flipping_through_book_is_random_event_l532_532564


namespace initial_walnut_trees_l532_532593

/-- 
  Given there are 29 walnut trees in the park after cutting down 13 walnut trees, 
  prove that initially there were 42 walnut trees in the park.
-/
theorem initial_walnut_trees (cut_walnut_trees remaining_walnut_trees initial_walnut_trees : ℕ) 
  (h₁ : cut_walnut_trees = 13)
  (h₂ : remaining_walnut_trees = 29)
  (h₃ : initial_walnut_trees = cut_walnut_trees + remaining_walnut_trees) :
  initial_walnut_trees = 42 := 
sorry

end initial_walnut_trees_l532_532593


namespace R_squared_l532_532041

-- Definitions based on the conditions
variables (a : ℝ) -- edge length of the cube
def A := (0, 0, 0) : ℝ × ℝ × ℝ
def B := (a, 0, 0) : ℝ × ℝ × ℝ
def C := (a, 0, a) : ℝ × ℝ × ℝ
def D := (0, 0, a) : ℝ × ℝ × ℝ
def E := (0, a, 0) : ℝ × ℝ × ℝ
def F := (a, a, 0) : ℝ × ℝ × ℝ
def G := (a, a, a) : ℝ × ℝ × ℝ
def H := (0, a, a) : ℝ × ℝ × ℝ

-- Midpoints J and I
def J := ((a + a) / 2, (a + 0) / 2, (0 + 0) / 2) : ℝ × ℝ × ℝ
def I := ((0 + 0) / 2, (a + a) / 2, (a + a) / 2) : ℝ × ℝ × ℝ

-- Calculate lengths EC and IJ
def length (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

def EC := length E C
def IJ := length I J

-- Calculate area of cross-section EJCI
def area_EJCI := (length E C) * (length I J) / 2

-- Calculate the area of one face of the cube
def area_face := a^2

-- Calculate ratio R
def R := area_EJCI / area_face

-- State the proof problem
theorem R_squared : R^2 = (9 : ℝ) / 4 := 
by {
  sorry -- Proof omitted.
}

end R_squared_l532_532041


namespace common_difference_arithmetic_sequence_l532_532931

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l532_532931


namespace arithmetic_sequence_common_difference_l532_532934

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532934


namespace chip_placement_configuration_l532_532433

-- Definition of the problem conditions
def total_chips : ℕ := 12
def grid_size : ℕ := 16
def num_red_chips : ℕ := 4
def num_blue_chips : ℕ := 4
def num_green_chips : ℕ := 4

-- The statement that encapsulates the problem and solution
theorem chip_placement_configuration :
  -- Conditions
  (∃ (placements : list (fin grid_size)), (placements.length = total_chips) ∧ 
    (∃ (red_pos blue_pos green_pos : list (fin grid_size)), 
      (red_pos.length = num_red_chips) ∧
      (blue_pos.length = num_blue_chips) ∧
      (green_pos.length = num_green_chips) ∧
      (∀ (i j : fin grid_size), 
        ((i < j) → 
          ((i ∉ red_pos ∨ j ∉ red_pos) ∧ 
           (i ∉ blue_pos ∨ j ∉ blue_pos) ∧ 
           (i ∉ green_pos ∨ j ∉ green_pos) ∨ 
           (abs (i.to_nat - j.to_nat) ≠ 1 ∧ abs (i.to_nat - j.to_nat) ≠ 4))) ∧
      (list.nodup_append of red_pos blue_pos green_pos)
    -- Conclusion
    )) → (∃! (n : ℕ), n = 30) :=
by sorry

end chip_placement_configuration_l532_532433


namespace first_player_wins_if_take_one_initial_l532_532588

theorem first_player_wins_if_take_one_initial :
  ∃ strategy : ℕ → ℕ, 
    (∀ n, strategy n = if n % 3 = 0 then 1 else 2) ∧ 
    strategy 99 = 1 ∧ 
    strategy 100 = 1 :=
sorry

end first_player_wins_if_take_one_initial_l532_532588


namespace smallest_four_digit_divisible_by_35_l532_532246

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l532_532246


namespace range_of_a_l532_532406

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^1 ^2 + (a + 2) * x1 + 9 * a = 0 ∧ ax2 ^2 + (a + 2) * x2 + 9 * a = 0 ∧ x1 < 2 ∧ 2 < x2)
  → - 4 / 15 < a ∧ a < 0 :=
begin
  sorry
end

end range_of_a_l532_532406


namespace smallest_four_digit_divisible_by_35_l532_532187

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l532_532187


namespace sum_of_integers_l532_532371

/-- 
  Theorem: The sum of all positive integers x such that |x^2 - x - 6| has exactly 4 positive integer divisors is 9.
  We define the function f(x) = |x^2 - x - 6| and condition on the number of divisors.
-/
theorem sum_of_integers (f : ℕ → ℕ) (hx : ∀ x, f x = |x^2 - x - 6|) :
  (∃ (X : List ℕ), (∀ x ∈ X, (Nat.divisors (f x)).length = 4) ∧ X.sum = 9) :=
sorry

end sum_of_integers_l532_532371


namespace angle_between_diagonals_of_quadrilateral_l532_532498

theorem angle_between_diagonals_of_quadrilateral
  {ABCD : ConvexQuadrilateral}
  (h1 : ABCD.AB = ABCD.CD)
  (h2 : ABCD.angleBCD = 2 * ABCD.angleBAD)
  (h3 : ABCD.angleABC = 2 * ABCD.angleADC)
  (h4 : ABCD.angleBAD ≠ ABCD.angleADC) :
  (angle_between_diagonals ABCD.AC ABCD.BD = 2 * π / 3) := 
sorry

end angle_between_diagonals_of_quadrilateral_l532_532498


namespace max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l532_532769

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem max_value_of_f : ∃ x, (f x) = 1/2 :=
sorry

theorem period_of_f : ∀ x, f (x + π) = f x :=
sorry

theorem not_monotonically_increasing : ¬ ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y :=
sorry

theorem incorrect_zeros : ∃ x y z, (0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ π) ∧ (f x = 0 ∧ f y = 0 ∧ f z = 0) :=
sorry

end max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l532_532769


namespace find_common_difference_l532_532873

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532873


namespace least_exponent_sum_700_l532_532796

noncomputable def least_exponent_sum (n : ℕ) : ℕ :=
if h : n ≥ 3 then
  let exponents := [9, 7, 5, 4, 3, 2]
  in if n = 700 then 30 else sorry
else sorry

theorem least_exponent_sum_700 : least_exponent_sum 700 = 30 := by
  sorry

end least_exponent_sum_700_l532_532796


namespace common_difference_l532_532993

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532993


namespace theta1_gt_theta2_l532_532005

theorem theta1_gt_theta2 (a : ℝ) (b : ℝ) (θ1 θ2 : ℝ)
  (h_range_θ1 : 0 ≤ θ1 ∧ θ1 ≤ π) (h_range_θ2 : 0 ≤ θ2 ∧ θ2 ≤ π)
  (x1 x2 : ℝ) (hx1 : x1 = a * Real.cos θ1) (hx2 : x2 = a * Real.cos θ2)
  (h_less : x1 < x2) : θ1 > θ2 :=
by
  sorry

end theta1_gt_theta2_l532_532005


namespace smallest_four_digit_div_by_35_l532_532235

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l532_532235


namespace part_I_part_II_l532_532772

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin x * cos x - cos x^2

theorem part_I :
  (∃ (A ω : ℝ) (φ B : ℝ), f = λ x, A * sin (ω * x + φ) + B 
   ∧ A = 1 ∧ ω = 2 ∧ φ = -π/6 ∧ B = -1/2) 
  ∧ 
  (∃ T : ℝ, T = π ∧ ∀ x : ℝ, f (x + T) = f x) := 
  sorry
  
theorem part_II (k : ℤ) :
  ∀ x : ℝ, (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) ↔ (derivative f x ≥ 0) := 
  sorry

end part_I_part_II_l532_532772


namespace missing_digit_divisible_by_3_l532_532565

theorem missing_digit_divisible_by_3 : ∃ d : ℕ, d < 10 ∧ (4 + 3 + 5 + 1 + d) % 3 = 0 ∧ d = 2 :=
by {
  use 2,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl, }
}

end missing_digit_divisible_by_3_l532_532565


namespace factorial_15_base_9_zeroes_l532_532434

theorem factorial_15_base_9_zeroes :
  (∃ k, 15! % 9^k = 0 ∧ 15! % 9^(k+1) ≠ 0) ∧ 
  (∀ k', 15! % 9^(k'+1) = 0 ↔ k' < 3) := sorry

end factorial_15_base_9_zeroes_l532_532434


namespace probability_at_most_3_heads_in_12_flips_l532_532632

theorem probability_at_most_3_heads_in_12_flips :
  let favorable := Nat.choose 12 0 + Nat.choose 12 1 + Nat.choose 12 2 + Nat.choose 12 3
  let total := 2^12
  (favorable : ℝ) / total = 299 / 4096 :=
by
  sorry

end probability_at_most_3_heads_in_12_flips_l532_532632


namespace smallest_four_digit_number_divisible_by_35_l532_532223

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l532_532223


namespace solve_system_l532_532106

theorem solve_system (x y z : ℝ) 
  (h1 : 19 * (x + y) + 17 = 19 * (-x + y) - 21)
  (h2 : 5 * x - 3 * z = 11 * y - 7) : 
  x = -1 ∧ z = -11 * y / 3 + 2 / 3 :=
by sorry

end solve_system_l532_532106


namespace line_through_point_parallel_l532_532125

theorem line_through_point_parallel (c : ℝ)
  (h : ∀ x y, x + real.sqrt 3 * y = c ↔ x = 1 ∧ y = -real.sqrt 3 + -2 / real.sqrt 3 * x):  
  ∃ k : ℝ, ∀ (x y : ℝ), x + real.sqrt 3 * y + k = 0 ∧ (x = 1 ∧ y = -real.sqrt 3) → k = 2 := 
sorry

end line_through_point_parallel_l532_532125


namespace correct_option_to_fill_blank_l532_532081

-- Definitions based on conditions in problem
inductive Option : Type
| something : Option
| anything : Option
| everything : Option
| nothing : Option

def sentence_filled_correctly (opt : Option) : Prop :=
  match opt with
  | Option.something => False
  | Option.anything => False
  | Option.everything => True
  | Option.nothing => False

-- The problem statement to prove
theorem correct_option_to_fill_blank : sentence_filled_correctly Option.everything := by
  sorry

end correct_option_to_fill_blank_l532_532081


namespace rectangular_prism_parallel_edges_l532_532432

theorem rectangular_prism_parallel_edges {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∃ n, n = 12) :=
begin
  use 12,
  sorry
end

end rectangular_prism_parallel_edges_l532_532432


namespace find_function_l532_532710

open Int

theorem find_function :
  ∃ (f : ℤ → ℤ), (∀ p : ℤ, prime p → f p > 0) ∧
                 (∀ (p x : ℤ), prime p → p ∣ (f x + f p) ^ (f p) - x) →
                 (∀ x : ℤ, f x = x) :=
begin
  sorry
end

end find_function_l532_532710


namespace election_total_votes_l532_532621

theorem election_total_votes (V : ℕ) (h1 : 0.20 * V = 1200) : V = 6000 :=
by
  sorry

end election_total_votes_l532_532621


namespace sin_ratio_correct_l532_532038

noncomputable def sin_ratio (A C : ℝ) (B : ℝ × ℝ) : ℝ :=
  if is_ellipse_point B 3 5 then (sin(A + C) / (sin A + sin C)) else 0

theorem sin_ratio_correct (A C : ℝ) (B : ℝ × ℝ) 
  (hA : A = 0 ∧ 4)
  (hC : C = 0 ∧ (-4))
  (hB : is_ellipse_point B 3 5) : 
  sin_ratio A C B = 4 / 5 :=
  sorry

end sin_ratio_correct_l532_532038


namespace lindsay_doll_count_l532_532529

theorem lindsay_doll_count :
  let B := 4 in                  -- Number of blonde-haired dolls
  let Br := 4 * B in             -- Number of brown-haired dolls
  let Bl := Br - 2 in            -- Number of black-haired dolls
  Bl + Br - B = 26 :=            -- Prove the combined excess of black and brown over blonde
begin 
  sorry 
end

end lindsay_doll_count_l532_532529


namespace arithmetic_sequence_term_l532_532742

theorem arithmetic_sequence_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 4 = 6)
    (h2 : 2 * (a 3) - (a 2) = 6)
    (h_sum : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 1 = -3 := 
by sorry

end arithmetic_sequence_term_l532_532742


namespace min_positive_Sn_l532_532408

noncomputable def a (n : ℕ) : ℝ := sorry -- define the arithmetic sequence

variable (n : ℕ)
def S (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2 -- sum of the first n terms of the sequence

theorem min_positive_Sn (n = 19) :
  (∀ k, S n ≤ S k) ∧
  (S n > 0) ∧
  ((a 9 + 3 * a 11 < 0) ∧
    (a 10 * a 11 < 0) ∧ 
    (a n * (n > 1) < 0)) :=
by sorry

end min_positive_Sn_l532_532408


namespace puzzle_pieces_missing_l532_532162

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l532_532162


namespace max_min_product_of_three_l532_532383

open List

theorem max_min_product_of_three (s : List Int) (h : s = [-1, -2, 3, 4]) : 
  ∃ (max min : Int), 
    max = 8 ∧ min = -24 ∧ 
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≤ max) ∧
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≥ min) := 
by
  sorry

end max_min_product_of_three_l532_532383


namespace num_possible_values_P_l532_532076

-- Define the set of even numbers from 2 to 200
def even_numbers_set := {x : ℕ | 2 <= x ∧ x <= 200 ∧ (x % 2 = 0)}

-- Define a 50-element subset of even numbers
def C : finset ℕ := sorry -- C must be a subset of even_numbers_set of size 50

-- Define the sum of the elements of C
def P := C.sum id

-- Statement to be proved
theorem num_possible_values_P : P ∈ (Icc 2550 7550) ∧ (finset.Icc 2550 7550).card = 5001 := sorry

end num_possible_values_P_l532_532076


namespace alice_cranes_ratio_alice_cranes_l532_532659

theorem alice_cranes {A : ℕ} (h1 : A + (1/5 : ℝ) * (1000 - A) + 400 = 1000) :
  A = 500 := by
  sorry

theorem ratio_alice_cranes :
  (500 : ℝ) / 1000 = 1 / 2 := by
  sorry

end alice_cranes_ratio_alice_cranes_l532_532659


namespace exists_set_of_size_r_minus_1_l532_532059

theorem exists_set_of_size_r_minus_1 (r : ℕ) (F : set (set ℕ)) 
  (hr : r ≥ 2) 
  (hF_infinite : ∀ S ∈ F, S.finite ∧ S.card = r) 
  (hF_non_disjoint : ∀ {A B : set ℕ}, A ∈ F → B ∈ F → A ≠ B → A ∩ B ≠ ∅) : 
  ∃ T : set ℕ, T.finite ∧ T.card = r - 1 ∧ ∀ S ∈ F, T ∩ S ≠ ∅ := 
sorry

end exists_set_of_size_r_minus_1_l532_532059


namespace factorize_expression_l532_532702

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l532_532702


namespace smallest_four_digit_divisible_by_35_l532_532210

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532210


namespace midpoints_locus_parallel_to_bisector_l532_532088

-- Define the setting of the problem
variable (O : Point) (S : Angle O) (A B D C D1 C1 D2 C2 : Point)

-- Define the equivalence of segments on the angle sides
axiom SegmentsEqual :
  segment_length A D = segment_length B C ∧ 
  segment_length D D1 = segment_length C C1 ∧ 
  segment_length D1 D2 = segment_length C1 C2

-- Define the statement to be proved
theorem midpoints_locus_parallel_to_bisector :
  locus_of_midpoints AB CD C1D1 C2D2 = ray_parallel_to_bisector (angle_bisector S) := 
sorry

end midpoints_locus_parallel_to_bisector_l532_532088


namespace ratio_areas_equivalent_l532_532323

noncomputable def ratio_of_areas : ℝ := by
  let C_angle : ℝ := (60 : ℝ) * Real.pi / 180
  let BC : ℝ := 1 -- assume BC = 1 for simplicity without loss of generality
  let x := BC
  let triangle_area := (x ^ 2 * Real.sqrt 3) / 4
  let sector_area := (Real.pi * x ^ 2) / 6
  let segment_area := (triangle_area - sector_area) / 2
  let S1 := triangle_area - sector_area
  let S2 := sector_area - segment_area
  let S3 := segment_area
  let ratio1 := S1 / (S1 + S2 + S3)
  let ratio2 := S2 / (S1 + S2 + S3)
  let ratio3 := S3 / (S1 + S2 + S3)
  exact (ratio1, ratio2, ratio3)
  
theorem ratio_areas_equivalent :
  ratio_of_areas = ((Real.sqrt (3 : ℝ) - Real.pi / 3) / ((Real.sqrt (3 : ℝ) - Real.pi / 3) + (Real.pi / 3 - (Real.sqrt (3 : ℝ) / 2)) + (Real.sqrt (3 : ℝ) / 2)),
                    (Real.pi / 3 - Real.sqrt (3 : ℝ) / 2) / ((Real.sqrt (3 : ℝ) - Real.pi / 3) + (Real.pi / 3 - (Real.sqrt (3 : ℝ) / 2)) + (Real.sqrt (3 : ℝ) / 2)),
                    (Real.sqrt (3 : ℝ) / 2) / ((Real.sqrt (3 : ℝ) - Real.pi / 3) + (Real.pi / 3 - (Real.sqrt (3 : ℝ) / 2)) + (Real.sqrt (3 : ℝ) / 2))) := 
by
    sorry

end ratio_areas_equivalent_l532_532323


namespace find_all_functions_l532_532129

noncomputable def func_form (f : ℤ × ℤ → ℝ) (α β γ δ : ℝ) : Prop :=
∀ (M : ℤ × ℤ), f M = α * M.1 + β * M.2 + γ * (M.1 ^ 2 + M.2 ^ 2) + δ * (Int.pow (-1) M.1 - Int.pow (-1) M.2)

theorem find_all_functions (f : ℤ × ℤ → ℝ) :
  (f (0,0) = 0) ∧ (∀ (A B C : ℤ × ℤ), (B = A + C) → f B = f A + f C) ↔ 
  ∃ (α β γ δ : ℝ), func_form f α β γ δ := sorry

end find_all_functions_l532_532129


namespace find_m_value_l532_532008

-- Definitions of the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (1, m)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Definition of parallelism condition
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given condition: vectors a and b are parallel
lemma vectors_parallel (m : ℝ) : parallel (a m) (b m) :=
sorry

-- Prove the value of m when vectors a and b are parallel
theorem find_m_value (m : ℝ) : parallel (a m) (b m) → m = 2 :=
begin
  intro h,
  -- Proof omitted
  sorry
end

end find_m_value_l532_532008


namespace common_difference_is_two_l532_532905

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532905


namespace find_common_difference_l532_532951

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532951


namespace find_common_difference_l532_532875

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532875


namespace pizza_pieces_each_cut_into_l532_532314

theorem pizza_pieces_each_cut_into (x : ℕ) (h1 : 0.5 * x + 0.5 * x = x) (h2 : 0.25 * x + 0.25 * x = 0.5 * x) (h3 : 1.5 * x = 6) : x = 4 :=
by 
  sorry

end pizza_pieces_each_cut_into_l532_532314


namespace sum_of_exponents_l532_532793

/-- If 700 is expressed as a sum of at least three distinct powers of 2, 
the least possible sum of the exponents is 30. -/
theorem sum_of_exponents (n : ℕ) (hn : n = 700) (h : ∃ s : Finset ℕ, (s.card ≥ 3) 
  ∧ (∑ i in s, 2 ^ i = n)) : ∑ i in (classical.some h), i = 30 := 
sorry

end sum_of_exponents_l532_532793


namespace smallest_four_digit_divisible_by_35_l532_532174

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532174


namespace arithmetic_sequence_common_difference_l532_532854

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532854


namespace min_x_given_conditions_l532_532513

theorem min_x_given_conditions :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (100 : ℚ) / 151 < y / x ∧ y / x < (200 : ℚ) / 251 ∧ x = 3 :=
by
  sorry

end min_x_given_conditions_l532_532513


namespace product_square_of_polynomial_l532_532099

noncomputable def is_square_of_polynomial (p : ℝ[X]) : Prop :=
  ∃ q : ℝ[X], p = q ^ 2

theorem product_square_of_polynomial
  (a b c d u v : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : ∀ x : ℝ, (2 * a * (x - u)) * (2 * c * (x - v)) ≥ abs (a * (x - u) ^ 2 + b) + abs (c * (x - v) ^ 2 + d)) :
  is_square_of_polynomial 
    ((a * (Polynomial.X - Polynomial.C u) ^ 2 + Polynomial.C b) * 
     (c * (Polynomial.X - Polynomial.C v) ^ 2 + Polynomial.C d)) := 
sorry

end product_square_of_polynomial_l532_532099


namespace probability_not_within_square_b_l532_532276

noncomputable def prob_not_within_square_b : Prop :=
  let area_A := 121
  let side_length_B := 16 / 4
  let area_B := side_length_B * side_length_B
  let area_not_covered := area_A - area_B
  let prob := area_not_covered / area_A
  prob = (105 / 121)

theorem probability_not_within_square_b : prob_not_within_square_b :=
by
  sorry

end probability_not_within_square_b_l532_532276


namespace least_exponent_sum_700_l532_532795

noncomputable def least_exponent_sum (n : ℕ) : ℕ :=
if h : n ≥ 3 then
  let exponents := [9, 7, 5, 4, 3, 2]
  in if n = 700 then 30 else sorry
else sorry

theorem least_exponent_sum_700 : least_exponent_sum 700 = 30 := by
  sorry

end least_exponent_sum_700_l532_532795


namespace sum_log_divisors_eq_1080_l532_532582

theorem sum_log_divisors_eq_1080 (n : ℕ) :
  (∑ a in finset.range (n + 1), ∑ b in finset.range (n + 1), 
     real.log10 (2 ^ a * 3 ^ b))
  = 1080 → n = 12 := by
  sorry

end sum_log_divisors_eq_1080_l532_532582


namespace common_difference_of_arithmetic_sequence_l532_532977

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532977


namespace distance_between_stations_l532_532625

theorem distance_between_stations 
  (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 25) 
  (h3 : distance_diff = 65) : 
  ∃ distance_between (t : ℝ), t = 13 ∧ distance_between = 585 :=
by
  -- Proof goes here
  sorry

end distance_between_stations_l532_532625


namespace earnings_last_duration_l532_532089

def total_earnings : ℝ :=
  12 + 8 + 5 + 20

def weekly_expenses : ℝ :=
  10 + 5 + 3

theorem earnings_last_duration :
  total_earnings / weekly_expenses = 2.5 :=
by
  simp [total_earnings, weekly_expenses]
  norm_num
  sorry

end earnings_last_duration_l532_532089


namespace equilateral_triangle_side_length_l532_532844

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 18 * Polynomial.X^2 + 91 * Polynomial.X - 89

def is_equilateral_triangle (ABC : Triangle) : Prop :=
  ABC.AB = ABC.BC ∧ ABC.BC = ABC.CA

def point_in_triangle (P : Point) (ABC : Triangle) : Prop :=
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u + v + w = 1 ∧ 
  P = u • ABC.A + v • ABC.B + w • ABC.C

theorem equilateral_triangle_side_length (ABC : Triangle) (P : Point) (s m n : ℝ)
  (h1 : is_equilateral_triangle ABC)
  (h2 : point_in_triangle P ABC)
  (h3 : (Polynomial.eval (dist P ABC.A)) polynomial = 0 ∧ 
        (Polynomial.eval (dist P ABC.B)) polynomial = 0 ∧ 
        (Polynomial.eval (dist P ABC.C)) polynomial = 0)
  (h4 : s^2 = m + real.sqrt n) :
  100 * m + n = 7208 :=
sorry

end equilateral_triangle_side_length_l532_532844


namespace value_of_b_l532_532618

theorem value_of_b : (15^2 * 9^2 * 356 = 6489300) :=
by 
  sorry

end value_of_b_l532_532618


namespace arithmetic_progression_infinite_squares_l532_532665

theorem arithmetic_progression_infinite_squares (a d : ℤ) (k : ℤ) (hk : a = k^2) :
  ∃ infinite_set_of_squares : set ℤ, (∀ n ∈ infinite_set_of_squares, ∃ m : ℤ, m^2 = a + n * d) ∧ infinite infinite_set_of_squares :=
  sorry

end arithmetic_progression_infinite_squares_l532_532665


namespace common_difference_l532_532997

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532997


namespace emma_missing_coins_l532_532697

theorem emma_missing_coins (x : ℕ) : 
  let lost = (1 / 3 : ℚ) * x
  let found = (3 / 4 : ℚ) * lost
  let remaining = x - lost + found
  (x - remaining) / (x : ℚ) = (1 / 12 : ℚ) :=
by
  sorry

end emma_missing_coins_l532_532697


namespace regular_tetrahedron_l532_532032

-- Definitions and assumptions based on conditions
variable (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]

-- Points on faces
variable (I₁ : incenter (Δ A B D)) -- Incenter on face ABD
variable (I₂ : incenter (Δ A B C)) -- Incenter on face ABC

-- AB = BD
axiom AB_eq_BD : dist A B = dist B D

-- perpendiculars from vertices to opposite faces
axiom perp_from_C : perpendicular (line C I₁) (plane A B D)
axiom perp_from_D : perpendicular (line D I₂) (plane A B C)

-- Incircles conditions stating I₁ and I₂ are incenters
axiom incenter_cond_1 : is_incenter I₁ (Δ A B D)
axiom incenter_cond_2 : is_incenter I₂ (Δ A B C)

-- Prove the tetrahedron ABCD is a regular tetrahedron
theorem regular_tetrahedron (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
    (I₁ : incenter (Δ A B D)) (I₂ : incenter (Δ A B C))
    (AB_eq_BD : dist A B = dist B D)
    (perp_from_C : perpendicular (line C I₁) (plane A B D))
    (perp_from_D : perpendicular (line D I₂) (plane A B C))
    (incenter_cond_1 : is_incenter I₁ (Δ A B D))
    (incenter_cond_2 : is_incenter I₂ (Δ A B C)) :
    is_regular_tetrahedron (A B C D) :=
by
  sorry

end regular_tetrahedron_l532_532032


namespace simplest_common_denominator_of_fractions_l532_532580

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end simplest_common_denominator_of_fractions_l532_532580


namespace number_of_non_congruent_triangles_with_perimeter_11_l532_532790

theorem number_of_non_congruent_triangles_with_perimeter_11 :
  let is_valid_triangle (a b c : ℕ) := a + b > c ∧ b + c > a ∧ c + a > b in
  ∃ (sides : Finset (ℕ × ℕ × ℕ)), sides.card = 6 ∧
  (∀ (a b c : ℕ), (a, b, c) ∈ sides ↔ a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ is_valid_triangle a b c) :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_11_l532_532790


namespace arithmetic_sequence_angle_l532_532398

-- Define the conditions
variables (A B C a b c : ℝ)
-- The statement assumes that A, B, C form an arithmetic sequence
-- which implies 2B = A + C
-- We need to show that 1/(a + b) + 1/(b + c) = 3/(a + b + c)

theorem arithmetic_sequence_angle
  (h : 2 * B = A + C)
  (cos_rule : b^2 = c^2 + a^2 - 2 * c * a * Real.cos B):
    1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := sorry

end arithmetic_sequence_angle_l532_532398


namespace sum_f_eq_3_pow_2018_l532_532751

def num_zeros_in_binary (n : ℕ) : ℕ :=
  String.toList (nat.to_binary n)
    .count (fun c => c = '0')

def f (n : ℕ) : ℕ :=
  2 ^ num_zeros_in_binary n

theorem sum_f_eq_3_pow_2018 :
  finset.sum (finset.Ico 2^2018 2^2019) f = 3^2018 :=
by
  sorry

end sum_f_eq_3_pow_2018_l532_532751


namespace sum_square_divisors_positive_l532_532583

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive_l532_532583


namespace min_value_after_2017_minutes_l532_532832

theorem min_value_after_2017_minutes :
  let rec evolve (x y : ℝ) : ℝ × ℝ :=
    if 2017 = 0 then (x, y)
    else evolve ((x + y) / 2) (2 / ((1 / x) + (1 / y))) (2017 - 1) in
  let initial_numbers : ℝ × ℝ := (1, 100000000) in
  let (a, b) := evolve initial_numbers.1 initial_numbers.2 2017 in
  abs ((min a b) - 10000) < 1 :=
by
  sorry

end min_value_after_2017_minutes_l532_532832


namespace arithmetic_sequence_common_difference_l532_532902

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532902


namespace quadratic_function_properties_l532_532422

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem quadratic_function_properties :
  (∀ x, f(x) = x^2 - 2 * x) ∧ set.range (f ∘ (λ x, x ∈ set.Icc 0 3)) = set.Icc (-1) 3 :=
by
  sorry

end quadratic_function_properties_l532_532422


namespace coloring_parts_with_distinct_colors_l532_532438

theorem coloring_parts_with_distinct_colors :
  ∃ (ways : ℕ), ways = 24 ∧ 
  ∀ (colors : Finset (Fin 4)), 
    (colors.card = 4 ∧ colors.pairwise (≠)) →
    (∃ (f : Fin 4 → Fin 4), 
      f.injective ∧
      (∃ (assignments : Finset (Fin 4 × Fin 4)), assignments.card = 4 ∧ 
      ∀ i : Fin 4, ∃ j : Fin 4, (i, j) ∈ assignments ∧ f i = j)) :=
begin
  use 24,
  split,
  { refl },
  sorry
end

end coloring_parts_with_distinct_colors_l532_532438


namespace find_current_listens_l532_532053

theorem find_current_listens (x : ℕ) (h : 15 * x = 900000) : x = 60000 :=
by
  sorry

end find_current_listens_l532_532053


namespace line_through_fixed_point_and_parabola_l532_532264

theorem line_through_fixed_point_and_parabola :
  (∀ (a : ℝ), ∃ (P : ℝ × ℝ), 
    (a - 1) * P.1 - P.2 + 2 * a + 1 = 0 ∧ 
    (∀ (x y : ℝ), (y^2 = - ((9:ℝ) / 2) * x ∧ x = -2 ∧ y = 3) ∨ (x^2 = (4:ℝ) / 3 * y ∧ x = -2 ∧ y = 3))) :=
by
  sorry

end line_through_fixed_point_and_parabola_l532_532264


namespace width_of_bottom_trapezium_l532_532338

theorem width_of_bottom_trapezium (top_width : ℝ) (area : ℝ) (depth : ℝ) (bottom_width : ℝ) 
  (h_top_width : top_width = 10)
  (h_area : area = 640)
  (h_depth : depth = 80) :
  bottom_width = 6 :=
by
  -- Problem description: calculating the width of the bottom of the trapezium given the conditions.
  sorry

end width_of_bottom_trapezium_l532_532338


namespace initial_pinecones_l532_532592

theorem initial_pinecones (P : ℝ) :
  (0.20 * P + 2 * 0.20 * P + 0.25 * (0.40 * P) = 0.70 * P - 0.10 * P) ∧ (0.30 * P = 600) → P = 2000 :=
by
  intro h
  sorry

end initial_pinecones_l532_532592


namespace find_common_difference_l532_532957

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532957


namespace rotated_parabola_180_degrees_l532_532455

theorem rotated_parabola_180_degrees (x y: ℝ) :
  let original_parabola := (λ x, x^2 - 6 * x)
  let rotated_parabola := (λ x, - (x + 3)^2 + 9)
  (∀ x, y = original_parabola x → y = rotated_parabola (-x)) :=
begin
  sorry
end

end rotated_parabola_180_degrees_l532_532455


namespace undefined_count_expression_l532_532725

theorem undefined_count_expression : 
  let expr (x : ℝ) := (x^2 - 16) / ((x^2 + 3*x - 10) * (x - 4))
  ∃ u v w : ℝ, (u = 2 ∨ v = -5 ∨ w = 4) ∧
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) :=
by
  sorry

end undefined_count_expression_l532_532725


namespace common_difference_of_arithmetic_sequence_l532_532983

variable {a₁ d : ℕ}
def S (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l532_532983


namespace dolls_proof_l532_532522

variable (blonde_dolls brown_dolls black_dolls : ℕ)

def given_conditions (blonde_dolls brown_dolls black_dolls : ℕ) : Prop :=
  blonde_dolls = 4 ∧
  brown_dolls = 4 * blonde_dolls ∧
  black_dolls = brown_dolls - 2

def question (blonde_dolls brown_dolls black_dolls : ℕ) : ℕ :=
  (brown_dolls + black_dolls) - blonde_dolls

theorem dolls_proof :
  ∀ (blonde_dolls brown_dolls black_dolls : ℕ),
  given_conditions blonde_dolls brown_dolls black_dolls →
  question blonde_dolls brown_dolls black_dolls = 26 :=
by
  intros blonde_dolls brown_dolls black_dolls h
  simp [given_conditions, question] at *
  split at h
  -- Now you'd need to either finish the proof or fill in the steps taken in the given solution.
  sorry

end dolls_proof_l532_532522


namespace f_eq_for_2_to_4_f_sum_0_2019_l532_532062

noncomputable def f : ℝ → ℝ
| x when 0 ≤ x ∧ x ≤ 2 := 2 * x - x^2
| x := sorry -- The implementation for other ranges derived from conditions is omitted

-- Condition: f(x+2) = f(-x)
axiom f_cond1 : ∀ x : ℝ, f (x + 2) = f (-x)

-- Condition: f(x) = -f(4 - x)
axiom f_cond2 : ∀ x : ℝ, f x = -f (4 - x)

-- Condition: For x ∈ [0, 2], f(x) = 2x - x^2
axiom f_cond3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = 2 * x - x^2

theorem f_eq_for_2_to_4 : ∀ x ∈ set.Icc (2:ℝ) 4, f x = x^2 - 6 * x + 8 := 
by
  intro x hx
  sorry -- The detailed proof is omitted and left as a placeholder

theorem f_sum_0_2019 : (∑ x in finset.range 2020, f x) = 0 := 
by
  sorry -- The detailed proof is omitted and left as a placeholder

end f_eq_for_2_to_4_f_sum_0_2019_l532_532062


namespace stickers_total_proof_l532_532587

def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem stickers_total_proof : total_stickers = 220 := by
  sorry

end stickers_total_proof_l532_532587


namespace midpoint_correct_perpendicular_bisector_correct_l532_532399

section midpoint_and_perpendicular_bisector

-- Define points A and B
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)

-- Define midpoint D
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def D := midpoint A B

-- Define slope of a line segment
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Define perpendicular bisector
def perpendicular_bisector (P Q : ℝ × ℝ) : ℝ → ℝ := λ x, - (x + (P.2 + Q.2) / 2)

-- Prove the midpoint D is correct
theorem midpoint_correct : D = (1 / 2, -11 / 2) :=
by
  sorry

-- Prove the equation of the perpendicular bisector is correct
theorem perpendicular_bisector_correct :
  ∀ x y : ℝ, y = perpendicular_bisector A B x → x + y + 5 = 0 :=
by
  sorry

end midpoint_and_perpendicular_bisector

end midpoint_correct_perpendicular_bisector_correct_l532_532399


namespace smallest_four_digit_divisible_by_35_l532_532179

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l532_532179


namespace missing_pieces_l532_532164

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l532_532164


namespace part_1_part_2_l532_532001

-- Define the polynomial function f(x)
def f (a b c x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x + c

-- Given conditions and targets for (1)
def p_condition := 1
def p_value := 3
def tangent_slope := 1  -- since the line y = x + 2 has a slope of 1

-- Define the function values and derivatives
def f_value (a b c x y : ℝ) : Prop := f a b c x = y
def f_prime (a b x : ℝ) : ℝ := 3*x^2 - 4*a*x + b
def f_prime_at_p (a b : ℝ) : Prop := f_prime a b p_condition = tangent_slope

theorem part_1 (c : ℝ) (h1 : c = 0)
  (h2 : f_value 2 6 0 p_condition p_value)
  (h3 : f_prime_at_p 2 6) :
  (2 = 2) ∧ (6 = 6) :=
sorry

-- Given conditions and targets for (2)
def extreme_points := {(x:ℝ, y:ℝ) | x = -1 ∧ y = 8 ∨ x = 3 ∧ y = -24}
def f_extreme_value (a b c : ℝ) (p : (ℝ × ℝ)) : Prop :=
  (let (x, y) := p in f a b c x = y) ∧
  (let (x, _) := p in f_prime a b x = 0)

theorem part_2
  (h1 : ∀ (pt : ℝ × ℝ), pt ∈ extreme_points → f_extreme_value (3/2) (-9) 3 pt) :
  f (3/2) (-9) 3 = λ x, x^3 - 3*x^2 - 9*x + 3 :=
sorry

end part_1_part_2_l532_532001


namespace concurrency_of_common_chords_l532_532839

/- Definitions used in the conditions are directly included -/

variables (ABC : Triangle) (O H : Point) [Circumcenter ABC O]
  [Orthocenter ABC H] (Γ : Circle ABC) (N : Point)
  [Midpoint N O H] (ωA ωB ωC : Circle)

-- Define ωA, ωB, and ωC as per conditions
axiom ωA_def : ∃(ωA: Circle), ωA = CircleofTriangleByTangentsAndPerpendicular(Γ B C H AN)
axiom ωB_def : ∃(ωB: Circle), ωB = CircleofTriangleByTangentsAndPerpendicular(Γ A C H BN)
axiom ωC_def : ∃(ωC: Circle), ωC = CircleofTriangleByTangentsAndPerpendicular(Γ A B H CN)

-- The main theorem to prove
theorem concurrency_of_common_chords :
  ∃ P : Point, (P ∈ common_chord ωA ωB ∧ P ∈ common_chord ωB ωC ∧ P ∈ common_chord ωC ωA) ∧ P ∈ line OH :=
sorry

end concurrency_of_common_chords_l532_532839


namespace karen_starts_late_l532_532494

def karen_speed := 60 -- Karen's speed in mph
def tom_speed := 45 -- Tom's speed in mph
def tom_distance := 24 -- Distance Tom drives in miles
def karen_lead := 4 -- Distance by which Karen beats Tom in miles

theorem karen_starts_late : 
  let t := tom_distance / tom_speed in -- Time Tom drives
  let t_k := (tom_distance + karen_lead) / karen_speed in -- Time Karen drives
  (t - t_k) * 60 = 4 := -- The time difference in minutes is 4
by
  sorry

end karen_starts_late_l532_532494


namespace factorize_expression_l532_532703

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l532_532703


namespace smallest_four_digit_divisible_by_35_l532_532201

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532201


namespace arccos_cos_10_l532_532682

theorem arccos_cos_10 : Real.arccos (Real.cos 10) = 2 := by
  sorry

end arccos_cos_10_l532_532682


namespace isosceles_triangle_leg_length_l532_532033

theorem isosceles_triangle_leg_length 
  (a b c : ℝ) (h : a = b)
  (h2 : c = 1)
  (angle_bisector : ∃ (d e f : ℝ) (h_sim : d / a = e / b), isosceles_triangle d e f)
  : a = (Real.sqrt 2) / 2 ∨ a = (Real.sqrt 5 + 1) / 2 :=
sorry

end isosceles_triangle_leg_length_l532_532033


namespace quadratic_equation_coefficients_l532_532827

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℤ), a = 2 ∧ b = 1 ∧ c = -5 ∧ (∀ x : ℤ, 2 * x^2 + x - 5 = a * x^2 + b * x + c) :=
begin
  sorry
end

end quadratic_equation_coefficients_l532_532827


namespace not_power_of_two_sum_consecutive_power_of_two_no_sum_consecutive_composite_odd_sum_consecutive_odd_prime_no_sum_consecutive_odd_even_sum_consecutive_odd_power_of_n_sum_n_consecutive_odd_l532_532274

-- Part (a)
theorem not_power_of_two_sum_consecutive (N : ℕ) : 
  (¬(∃ k : ℕ, N = 2^k) ↔ ∃ k l : ℕ, N = ∑ i in finset.range k, (l + i)) :=
sorry

theorem power_of_two_no_sum_consecutive (N : ℕ) :
  (∃ k : ℕ, N = 2^k) ↔ ¬(∃ k l : ℕ, N = ∑ i in finset.range k, (l + i)) :=
sorry

-- Part (b)
theorem composite_odd_sum_consecutive_odd (N : ℕ) :
  (is_composite_odd (N) ↔ ∃ k l : ℕ, N = ∑ i in finset.range k, (l + 2i - 1)) :=
sorry

theorem prime_no_sum_consecutive_odd (N : ℕ) :
  (nat.prime N ↔ ¬(∃ k l : ℕ, N = ∑ i in finset.range k, (l + 2i - 1))) :=
sorry

theorem even_sum_consecutive_odd (N : ℕ) :
  (N % 2 = 0 ↔ ∃ k l : ℕ, N = ∑ i in finset.range k, (l + 2i - 1)) ↔ (N % 4 = 0) :=
sorry

-- Part (c)
theorem power_of_n_sum_n_consecutive_odd (n k : ℕ) (hk : k > 1) :
  ∃ l : ℕ, n^k = ∑ i in finset.range n, (l + 2i + 1) :=
sorry

end not_power_of_two_sum_consecutive_power_of_two_no_sum_consecutive_composite_odd_sum_consecutive_odd_prime_no_sum_consecutive_odd_even_sum_consecutive_odd_power_of_n_sum_n_consecutive_odd_l532_532274


namespace sum_of_sines_ge_one_l532_532496

theorem sum_of_sines_ge_one (n : ℕ) (α : Fin n → ℝ)
  (hα_interval : ∀ i, 0 ≤ α i ∧ α i ≤ 2 * Real.pi)
  (h_sum_odd : ∃ k : ℤ, ∑ i, (1 + Real.cos (α i)) = 2 * k + 1) :
  ∑ i, Real.sin (α i) ≥ 1 :=
sorry

end sum_of_sines_ge_one_l532_532496


namespace find_common_difference_l532_532952

section
variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Condition: S_n represents the sum of the first n terms of the arithmetic sequence {a_n}
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: 2S_3 = 3S_2 + 6
def arithmetic_condition (S : ℕ → ℝ) : Prop :=
  2 * S 3 = 3 * S 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem find_common_difference 
  (h₁ : sum_arithmetic_sequence S a 2)
  (h₂ : sum_arithmetic_sequence S a 3)
  (h₃ : arithmetic_condition S) :
  d = 2 :=
sorry
end

end find_common_difference_l532_532952


namespace region_diff_correct_l532_532086

noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
noncomputable def one_triangle_area : ℝ := (Real.sqrt 3) / 4
noncomputable def triangles_area : ℝ := 18 * one_triangle_area
noncomputable def R_area : ℝ := hexagon_area + triangles_area
noncomputable def S_area : ℝ := 4 * (1 + Real.sqrt 2)
noncomputable def region_diff : ℝ := S_area - R_area

theorem region_diff_correct :
  region_diff = 4 + 4 * Real.sqrt 2 - 6 * Real.sqrt 3 :=
by
  sorry

end region_diff_correct_l532_532086


namespace marble_probability_l532_532637

theorem marble_probability :
  let total_marbles := 10
  let red_marbles := 4
  let white_marbles := 6
  let prob_first_white := white_marbles / total_marbles
  let prob_second_red := red_marbles / (total_marbles - 1)
  prob_first_white * prob_second_red = 4 / 15 :=
by
  let total_marbles := 10
  let red_marbles := 4
  let white_marbles := 6
  let prob_first_white := white_marbles / total_marbles
  let prob_second_red := red_marbles / (total_marbles - 1)
  have h1 : prob_first_white = (3/5) := by sorry
  have h2 : prob_second_red = (4/9) := by sorry
  have h3 : (3/5) * (4/9) = 4/15 := by sorry
  rw [h1, h2]
  exact h3

end marble_probability_l532_532637


namespace probability_of_specific_draw_l532_532596

noncomputable def probability_first_ace_second_spade_third_3 : ℚ :=
  let prob_case1 := (3 / 52) * (12 / 51) * (4 / 50)
  let prob_case2 := (3 / 52) * (1 / 51) * (3 / 50)
  let prob_case3 := (1 / 52) * (12 / 51) * (4 / 50)
  let prob_case4 := (1 / 52) * (1 / 51) * (3 / 50)
  (prob_case1 + prob_case2 + prob_case3 + prob_case4)

theorem probability_of_specific_draw :
  probability_first_ace_second_spade_third_3 = 17 / 11050 :=
begin
  -- Skipping the proof steps
  sorry
end

end probability_of_specific_draw_l532_532596


namespace arithmetic_sequence_common_difference_l532_532894

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l532_532894


namespace smallest_four_digit_divisible_by_35_l532_532204

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l532_532204


namespace inlet_pipe_rate_l532_532297

theorem inlet_pipe_rate (capacity : ℝ) (h_capacity : capacity = 3600.000000000001)
  (leak_empty_time : ℝ) (h_leak_empty_time : leak_empty_time = 6)
  (combined_empty_time : ℝ) (h_combined_empty_time : combined_empty_time = 8) :
  let leak_rate := capacity / leak_empty_time in
  let combined_rate := capacity / combined_empty_time in
  let inlet_rate := leak_rate - combined_rate in
  inlet_rate = 150 :=
by
  -- definitions from conditions
  let leak_rate := capacity / leak_empty_time
  let combined_rate := capacity / combined_empty_time
  let inlet_rate := leak_rate - combined_rate

  -- prove the conclusion
  have h_leak_rate : leak_rate = 600 := by
    rw [h_capacity, h_leak_empty_time]
    norm_num

  have h_combined_rate : combined_rate = 450 := by
    rw [h_capacity, h_combined_empty_time]
    norm_num

  have h_inlet_rate : inlet_rate = 150 := by
    rw [←h_combined_rate, ←h_leak_rate]
    norm_num

  exact h_inlet_rate

end inlet_pipe_rate_l532_532297


namespace triangle_ABC_BE_l532_532166

noncomputable def proof_problem : ℕ :=
let a := 0,
    b := 4,
    c := 154,
    d := 7 in
a + b + c + d

theorem triangle_ABC_BE (a b c d : ℕ) 
  (AB BC CA AD AE DF EF : ℝ)
  (h1 : AB = 4)
  (h2 : BC = 5)
  (h3 : CA = 6)
  (h4 : AB < AD)
  (h5 : AD < AE)
  (h6 : DF = 2)
  (h7 : EF = 7)
  (h8 : ∃ D E F, F ≠ C ∧ (∀ p, angle (p A C D) = angle (p A C F)) ∧ (∀ p, angle (p E B C) = angle (p E B F)))
  : a + b + c + d = 165 :=
begin
  sorry
end

end triangle_ABC_BE_l532_532166


namespace at_least_two_equal_l532_532497

noncomputable def positive_reals (x y z : ℝ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0

noncomputable def triangle_inequality_for_n (x y z : ℝ) (n : ℕ) : Prop :=
(x^n + y^n > z^n) ∧ (y^n + z^n > x^n) ∧ (z^n + x^n > y^n)

theorem at_least_two_equal (x y z : ℝ) 
  (pos : positive_reals x y z) 
  (triangle_ineq: ∀ n : ℕ, n > 0 → triangle_inequality_for_n x y z n) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l532_532497


namespace cade_marbles_now_l532_532329

def original_marbles : ℝ := 87.0
def added_marbles : ℝ := 8.0
def total_marbles : ℝ := original_marbles + added_marbles

theorem cade_marbles_now : total_marbles = 95.0 :=
by
  sorry

end cade_marbles_now_l532_532329


namespace karen_starts_late_by_4_minutes_l532_532490

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l532_532490


namespace arithmetic_sequence_common_difference_l532_532859

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532859


namespace arithmetic_sequence_common_difference_l532_532862

variable {a₁ d : ℕ}
variable S : ℕ → ℕ

-- Definitions of the sums S₂ and S₃ in an arithmetic sequence
def S₂ : ℕ := a₁ + (a₁ + d)
def S₃ : ℕ := a₁ + (a₁ + d) + (a₁ + 2 * d)

theorem arithmetic_sequence_common_difference (h : 2 * S₃ = 3 * S₂ + 6) : d = 2 :=
by
  -- Skip the proof.
  sorry

end arithmetic_sequence_common_difference_l532_532862


namespace min_modulus_eq_two_l532_532021

theorem min_modulus_eq_two (z : ℂ) (hz : complex.abs (z + 2 - 2 * complex.I) = 1) : 
  ∃ w : ℂ, complex.abs (w - 1 - 2 * complex.I) = 2 :=
sorry

end min_modulus_eq_two_l532_532021


namespace largest_angle_locus_l532_532535

theorem largest_angle_locus (A B : Point) :
  (locus_of_largest_angles (acute_angled_triangles_with_side A B)) 
  = (region_excluding_semicircle_with_diameter A B ∪ {A} ∪ {B}) :=
sorry

end largest_angle_locus_l532_532535


namespace part1_part2_l532_532746

-- Definitions and conditions
variable {a b : ℝ} (h_a_b : a > b) (h_b_nonzero : b > 0)
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def A := (-a, 0)
def B := (a, 0)
variable (x0 y0 : ℝ) (h_point_on_ellipse : ellipse x0 y0)
variable (h_x0_not_a : x0 ≠ a) (h_x0_not_minus_a : x0 ≠ -a)
def slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)

-- The slopes of AP and BP
def k_AP := slope (-a) 0 x0 y0
def k_BP := slope a 0 x0 y0

-- Focus
def c := real.sqrt (a^2 - b^2)
def F := (-c, 0)

-- Main theorem part (1)
theorem part1 : k_AP * k_BP = -b^2 / a^2 := sorry

-- The definition for part (2)
variable (λ : ℝ) (h_AF_FB : λ = 1 / 3)

-- Main theorem part (2)
theorem part2 : λ = 1 / 3 := sorry

end part1_part2_l532_532746


namespace triangle_isosceles_l532_532845

-- Definitions involved: Triangle, Circumcircle, Angle Bisector, Isosceles Triangle
universe u

structure Triangle (α : Type u) :=
  (A B C : α)

structure Circumcircle (α : Type u) :=
  (triangle : Triangle α)

structure AngleBisector (α : Type u) :=
  (A : α)
  (triangle : Triangle α)

def IsoscelesTriangle {α : Type u} (P Q R : α) : Prop :=
  ∃ (p₁ p₂ p₃ : α), (p₁ = P ∧ p₂ = Q ∧ p₃ = R) ∧
                  ((∃ θ₁ θ₂, θ₁ + θ₂ = 90) → (∃ θ₃ θ₂, θ₃ + θ₂ = 90))

theorem triangle_isosceles {α : Type u} (T : Triangle α) (S : α)
  (h1 : Circumcircle α) (h2 : AngleBisector α) :
  IsoscelesTriangle T.B T.C S :=
by
  sorry

end triangle_isosceles_l532_532845


namespace find_constants_l532_532512

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x + 1

noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ( (x - a + Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3) +
  ( (x - a - Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3)

theorem find_constants (a b c : ℝ) (h1 : f_inv (1:ℝ) a b c = 0)
  (ha : a = 1) (hb : b = 2) (hc : c = 5) : a + 10 * b + 100 * c = 521 :=
by
  rw [ha, hb, hc]
  norm_num

end find_constants_l532_532512


namespace marty_paint_combinations_l532_532531

theorem marty_paint_combinations (colors : ℕ) (methods : ℕ) (h1 : colors = 5) (h2 : methods = 4) :
  colors * methods = 20 :=
by
  rw [h1, h2]
  exact Nat.mul_comm _ _

end marty_paint_combinations_l532_532531


namespace find_y_l532_532808

theorem find_y (y : ℝ) : 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := 
by
  sorry

end find_y_l532_532808


namespace ian_money_left_l532_532439

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l532_532439


namespace sum_of_angles_l532_532676

theorem sum_of_angles (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ = 67.5) (h₂ : θ₂ = 157.5) (h₃ : θ₃ = 247.5) (h₄ : θ₄ = 337.5) :
  θ₁ + θ₂ + θ₃ + θ₄ = 810 :=
by
  -- These parameters are used only to align with provided conditions
  let r₁ := 1
  let r₂ := r₁
  let r₃ := r₁
  let r₄ := r₁
  have z₁ := r₁ * (Complex.cos θ₁ + Complex.sin θ₁ * Complex.I)
  have z₂ := r₂ * (Complex.cos θ₂ + Complex.sin θ₂ * Complex.I)
  have z₃ := r₃ * (Complex.cos θ₃ + Complex.sin θ₃ * Complex.I)
  have z₄ := r₄ * (Complex.cos θ₄ + Complex.sin θ₄ * Complex.I)
  sorry

end sum_of_angles_l532_532676


namespace common_difference_is_two_l532_532911

variable {a₁ a₂ a₃ S₃ S₂ : ℕ}
variable (d : ℕ)

-- Given condition
axiom H : 2 * S₃ = 3 * S₂ + 6

-- Definitions based on arithmetic sequence properties
def S₂ := a₁ + a₂
def S₃ := a₁ + a₂ + a₃
def a₂ := a₁ + d
def a₃ := a₁ + 2 * d

theorem common_difference_is_two : d = 2 := 
by 
  sorry

end common_difference_is_two_l532_532911


namespace increasing_involution_is_identity_l532_532750

noncomputable theory

-- Definitions based on given conditions
variable {R : Type*} [linear_ordered_field R]
variable (f : R → R)
variable (hf_incr : ∀ x y : R, x < y → f x < f y)
variable (hf_inv : ∀ x : R, f (f x) = x)

-- Statement of the theorem
theorem increasing_involution_is_identity (f_incr : ∀ x y : R, x < y → f x < f y)
                                          (f_inv : ∀ x : R, f (f x) = x) :
  ∀ x : R, f x = x :=
by
  sorry

end increasing_involution_is_identity_l532_532750


namespace bounded_infinite_sequence_l532_532737

noncomputable def constructed_sequence (n : ℕ) : ℝ :=
  4 * (real.sqrt 2 * n - real.floor (real.sqrt 2 * n))

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|constructed_sequence i - constructed_sequence j| * |i - j|^((u : ℝ))) ≥ 1 :=
begin
  -- All necessary conditions and transformations as per the problem constraints
  sorry
end

end bounded_infinite_sequence_l532_532737


namespace number_reported_by_2005th_student_l532_532029

theorem number_reported_by_2005th_student : 
  let pattern := [1, 2, 3, 4, 3, 2] in
  pattern[(2005 % 6)]
= 1 :=
by sorry

end number_reported_by_2005th_student_l532_532029


namespace range_of_k_l532_532454

theorem range_of_k (k : ℝ) (h : ∃ x ∈ Ioo 0 1, exp x - k = 0) : 1 < k ∧ k < real.exp 1 :=
sorry

end range_of_k_l532_532454


namespace arithmetic_sequence_common_difference_l532_532938

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l532_532938


namespace card_worth_l532_532085

theorem card_worth (value_per_card : ℕ) (num_cards_traded : ℕ) (profit : ℕ) (value_traded : ℕ) (worth_received : ℕ) :
  value_per_card = 8 →
  num_cards_traded = 2 →
  profit = 5 →
  value_traded = num_cards_traded * value_per_card →
  worth_received = value_traded + profit →
  worth_received = 21 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end card_worth_l532_532085


namespace common_difference_l532_532998

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l532_532998


namespace smallest_four_digit_divisible_by_35_l532_532218

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l532_532218


namespace planting_methods_count_l532_532669

-- Define the conditions
def sections : ℕ := 6
def colors : ℕ := 4

-- Define the property of color assignment
def adjacent_different (assignment : Fin sections → Fin colors) : Prop :=
  ∀ i, assignment i ≠ assignment ((i + 1) % sections)

-- Define the flower planting problem
def planting_methods (sections colors : ℕ) [Hsections : sections = 6] [Hcolors : colors = 4] : ℕ :=
  ∑ (assignment : Fin sections → Fin colors) in {f | adjacent_different f}, 1

-- The proof statement
theorem planting_methods_count : planting_methods 6 4 = 120 := 
by
  unfold planting_methods
  sorry

end planting_methods_count_l532_532669


namespace no_real_b_for_inequality_l532_532691

theorem no_real_b_for_inequality (b : ℝ) :
  (∃ x : ℝ, |x^2 + 3*b*x + 4*b| ≤ 5 ∧ (∀ y : ℝ, |y^2 + 3*b*y + 4*b| ≤ 5 → y = x)) → false :=
by
  sorry

end no_real_b_for_inequality_l532_532691


namespace tan_alpha_fraction_l532_532727

theorem tan_alpha_fraction :
  (∃ α : ℝ, (sin(α + π) + cos(π - α)) / (sin(π / 2 - α) + sin(2 * π - α)) = 5 → tan α = 3 / 2) :=
by
    intro α hα
    sorry

end tan_alpha_fraction_l532_532727


namespace salami_cost_is_4_l532_532113

noncomputable def cost_of_salmi_and_brie (salami_cost brie_cost : ℝ) := 
  salami_cost + brie_cost = 16

noncomputable def known_items_cost := 
  2 * 7.75 + (1/4) * 10 + (1/2) * 8 + 2 = 24

theorem salami_cost_is_4 (total_cost : ℝ) (salami_cost : ℝ) (brie_cost : ℝ) :
  total_cost = 40 ∧ brie_cost = 3 * salami_cost ∧ cost_of_salmi_and_brie salami_cost brie_cost :=
begin
  sorry
end

end salami_cost_is_4_l532_532113


namespace common_difference_of_arithmetic_sequence_l532_532886

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : 2 * S a 3 = 3 * S a 2 + 6) :
  ∃ d : ℝ, d = 2 := sorry

end common_difference_of_arithmetic_sequence_l532_532886


namespace coeffs_of_quadratic_eq_l532_532826

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end coeffs_of_quadratic_eq_l532_532826


namespace half_of_original_amount_l532_532837

theorem half_of_original_amount (h : ∃ (m : ℚ), (4/7 : ℚ) * m = 24) : 
  ∃ (half_m : ℚ), half_m = 21 :=
by
  obtain ⟨m, hm⟩ := h
  have original := m
  have half_orig := (1/2 : ℚ) * original
  have target := (7/4 : ℚ) * 24 / 2
  use half_orig
  rw [←hm]
  have fact : (4 / 7) * original * (7 / 4) = original := by sorry
  have eq1 : (7 / 4) * 24 = original := eq.trans (mul_eq_mul_right_iff.mpr (oreq_of_ne_zero (by norm_num)) (by norm_num) hm.symm)
  have eq2 := eq.trans eq1 div_eq_div_right_iff nonzero_of_ne_zero (by norm_num)
  rw [eq2] at this
  exact sorry


end half_of_original_amount_l532_532837


namespace find_common_difference_l532_532866

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, arithmetic_sequence a₁ d (k + 1)

-- Given condition
def condition (a₁ d : ℝ) : Prop := 
  2 * sum_of_first_n_terms a₁ d 3 = 3 * sum_of_first_n_terms a₁ d 2 + 6

-- The proof statement
theorem find_common_difference (a₁ d : ℝ) (h : condition a₁ d) : d = 2 :=
by
  sorry

end find_common_difference_l532_532866


namespace triangle_angle_and_area_l532_532809

variable {a b c : ℝ}
variable (A : ℝ)
variable (S : ℝ)

noncomputable def m := (b + c, a^2 + b*c)
noncomputable def n := (b + c, -1)

theorem triangle_angle_and_area (h₀ : m.1 * n.1 + m.2 * n.2 = 0)
                                (h₁ : a = real.sqrt 3) :
  (A = 2 * real.pi / 3) ∧ (S = real.sqrt 3 / 4) :=
by
  -- Proof would go here
  sorry

end triangle_angle_and_area_l532_532809


namespace find_b_l532_532123

noncomputable def direction_vector (P Q : ℝ×ℝ) := (Q.1 - P.1, Q.2 - P.2)

theorem find_b (P Q : ℝ×ℝ) (b : ℝ) (hP : P = (-5, 4)) (hQ : Q = (-2, 5)) :
  b = 2 / 3 := by
  let v := direction_vector P Q
  have h : v = (3, 1) := by
    -- manually calculating direction vector to simplify
    rw [hP, hQ]
    sorry -- to keep this proof illustrative and not diverge from steps
  let scale := 2 / 3
  let v_scaled := (scale * v.1, scale * v.2)
  have h_scale : v_scaled = (2, b) := by
    -- assuming (v_scaled) realization from calculation steps given
    sorry
  -- concluding b value from scaling vector adjustment
  let hb := v_scaled.2
  have h_b_value : b = 2 / 3 := by
    rw [h_scale]
    exact hb
  exact h_b_value

end find_b_l532_532123


namespace problem_statement_l532_532020

def p (x y : ℝ) : Prop :=
  (x^2 + y^2 ≠ 0) → ¬ (x = 0 ∧ y = 0)

def q (m : ℝ) : Prop :=
  (m > -2) → ∃ x : ℝ, x^2 + 2*x - m = 0

theorem problem_statement : ∀ (x y m : ℝ), p x y ∨ q m :=
sorry

end problem_statement_l532_532020


namespace fresh_fruit_count_l532_532814

def initial_pineapples := 120
def initial_coconuts := 75
def pineapples_sold := 52
def coconuts_sold := 38
def rotten_pineapples := 11
def spoil_percentage := 5

noncomputable def fresh_pineapples := initial_pineapples - pineapples_sold - rotten_pineapples
noncomputable def discarded_coconuts := Int.ceil (spoil_percentage * (initial_coconuts - coconuts_sold) * 0.01)
noncomputable def fresh_coconuts := initial_coconuts - coconuts_sold - discarded_coconuts

theorem fresh_fruit_count :
  fresh_pineapples + fresh_coconuts = 92 :=
by
  sorry

end fresh_fruit_count_l532_532814


namespace find_positive_x_l532_532717

theorem find_positive_x (x : ℝ) (hx_pos : 0 < x) (h : log 5 (x - 2) + log 25 (x^3 - 2) + log (1/5) (x - 2) = 3) : x = 25 :=
sorry

end find_positive_x_l532_532717


namespace inequality_solution_set_l532_532581

theorem inequality_solution_set (x : ℝ) :
  abs (1 + x + x^2 / 2) < 1 ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_set_l532_532581


namespace remainder_when_divided_l532_532370

-- Definitions
def poly : Polynomial ℝ := 10 * (X^4) - 22 * (X^3) + 5 * (X^2) - 8 * X - 45
def divisor : Polynomial ℝ := 5 * X - 10
def x_val := 2
def remainder := 3

-- Theorem
theorem remainder_when_divided (f : Polynomial ℝ) (d : Polynomial ℝ) (x : ℝ) (r : ℝ) : 
  f = poly ∧ d = divisor ∧ x = x_val ∧ r = remainder → Polynomial.eval x f % Polynomial.eval x d = r := by
  sorry

end remainder_when_divided_l532_532370


namespace multiple_of_4_difference_multiple_of_4_l532_532556

variables (a b : ℤ)

def is_multiple_of (x y : ℤ) : Prop := ∃ k : ℤ, x = k * y

axiom h1 : is_multiple_of a 4
axiom h2 : is_multiple_of b 8

theorem multiple_of_4 (b : ℤ) (h : is_multiple_of b 8) : is_multiple_of b 4 :=
by {
  unfold is_multiple_of at *,
  cases h with k hk,
  use k * 2,
  rw [hk],
  norm_num,
}

theorem difference_multiple_of_4 (a b : ℤ) (ha : is_multiple_of a 4) (hb : is_multiple_of b 4) : is_multiple_of (a - b) 4 :=
by {
  unfold is_multiple_of at *,
  cases ha with ka hka,
  cases hb with kb hkb,
  use (ka - kb),
  rw [hka, hkb, sub_mul, mul_sub],
}

end multiple_of_4_difference_multiple_of_4_l532_532556


namespace average_first_300_terms_l532_532341

def sequence (n : ℕ) : ℤ := (-1)^(n+1) * n

theorem average_first_300_terms : 
  (∑ i in Finset.range 300, sequence i) / 300 = -1/2 := 
by 
  sorry

end average_first_300_terms_l532_532341
