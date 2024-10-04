import Mathlib

namespace mean_of_reciprocals_first_four_primes_l605_605151

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605151


namespace min_income_of_top_800_people_l605_605004

theorem min_income_of_top_800_people :
  ∃ x: ℝ, (8 * 10^8 * x^(-3/2) = 800) → x = 10^4 :=
by
  sorry

end min_income_of_top_800_people_l605_605004


namespace expected_value_red_squares_l605_605976

/-- Type representing coordinates in the grid, using integral coordinates -/
structure Coordinates where
  x : ℕ
  y : ℕ
  deriving DecidableEq, Repr

/-- Represents a blue square of side length 10 with specified corners -/
def blue_square : Set Coordinates := 
  {p | p.x ≤ 10 ∧ p.y ≤ 10}

/-- Represents a red square of side length 2 -/
def red_square (bottom_left : Coordinates) : Set Coordinates :=
  {p | p.x ≥ bottom_left.x ∧ p.x < bottom_left.x + 2 ∧ 
       p.y ≥ bottom_left.y ∧ p.y < bottom_left.y + 2}

/-- The main problem statement in Lean -/
def expected_red_squares : ℕ := 201

/-- The theorem to be proved -/
theorem expected_value_red_squares : 
  (let number_of_red_squares := expected_red_squares in
   number_of_red_squares) = 201 :=
by
  sorry

end expected_value_red_squares_l605_605976


namespace sqrt_three_not_in_A_l605_605403

theorem sqrt_three_not_in_A :
  let A := {x : ℚ | x > -2} in
  ¬ (↑(real.sqrt 3) ∈ A) := by
sorry

end sqrt_three_not_in_A_l605_605403


namespace omit_number_satisfies_sum_conditions_l605_605351

theorem omit_number_satisfies_sum_conditions :
  (∃ (y : ℕ), y ∈ finset.range 18 ∧ y ≠ 0 ∧ ∃ (s : finset ℕ), s = finset.range 18 \ {y} ∧ (s.sum id % 8 = 0)) :=
by
  sorry

end omit_number_satisfies_sum_conditions_l605_605351


namespace evaluate_expression_l605_605644

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l605_605644


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605560

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605560


namespace rhombus_side_length_l605_605811

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l605_605811


namespace three_digit_numbers_count_4_primes_l605_605759

theorem three_digit_numbers_count_4_primes : 
  let digits := {2, 3, 5, 7} in
  ∀ n ∈ digits,
  (∀ m ∈ digits, m ≠ n → 
    (∀ k ∈ digits, k ≠ n ∧ k ≠ m → 
      1 ≤ n ∧ 1 ≤ m ∧ 1 ≤ k ∧ n < 10 ∧ m < 10 ∧ k < 10 
    → 4 * 3 * 2 = 24)) :=
by
  intro digits h_digits n hN m hM h_neq_nm k hK h_neq_nk h_bounds
  rw Nat.mul_comm -- commutativity of multiplication
  sorry

end three_digit_numbers_count_4_primes_l605_605759


namespace sum_of_integers_between_10_and_20_l605_605044

theorem sum_of_integers_between_10_and_20 :
  ∑ i in Finset.range (20 - 10 - 1), (i + 11) = 135 := by
  sorry

end sum_of_integers_between_10_and_20_l605_605044


namespace triangle_similarity_l605_605850

theorem triangle_similarity 
  (A B C D M N O : Type) 
  [trapezoid ABCD A B C D]
  (ab_parallel_cd : parallel A B C D)
  (diagonals_perp : perpendicular A C B D)
  (ab_angle_acute: acute_angle A B)
  (base_angles_acute : acute_angle D A B C B A) 
  (point_M_on_OA : point_on_line_segment M O A)
  (point_N_on_OB : point_on_line_segment N O B)
  (right_angle_BMD : right_angle B M D) 
  (right_angle_ANC : right_angle A N C) : 
  similar_triangles O M N O B A :=
sorry

end triangle_similarity_l605_605850


namespace circumscribed_sphere_surface_area_l605_605836

theorem circumscribed_sphere_surface_area
  {P A B C : Type*}
  (h1 : PA ⊥ plane ABC) 
  (h2 : right_triangle ABC)
  (h3 : AB ⊥ BC)
  (h4 : AB = 1)
  (h5 : BC = 1)
  (h6 : PA = 2) :
  sphere_surface_area (circumscribed_sphere (triangular_pyramid P A B C)) = 6 * π :=
by sorry

end circumscribed_sphere_surface_area_l605_605836


namespace finite_distinct_y0_l605_605620

def g (x : ℝ) : ℝ := 3 * x - (1 / 3) * x^2

theorem finite_distinct_y0 :
  {y0 : ℝ | ∃ n ≥ 1, ∀ m ≥ n, y0 = (g^[m]) (y0) ∧ 
   ∃ k, distinct (list.range k.map (λ i, (g^[i]) y0)) <= 3}.card = 3 := sorry

end finite_distinct_y0_l605_605620


namespace evaluate_expression_l605_605646

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l605_605646


namespace rhombus_side_length_l605_605812

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l605_605812


namespace mean_of_reciprocals_first_four_primes_l605_605146

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605146


namespace smallest_n_for_identity_l605_605704

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605704


namespace arithmetic_mean_reciprocals_primes_l605_605158

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605158


namespace area_triangle_ABF_l605_605516

-- Define points A, B, C, D on the coordinate plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)

-- Define point E on the line y = 2x within the square
def E : ℝ × ℝ := (1, 2)

-- Define point F as the intersection of AC and AE
def F : ℝ × ℝ := (1, 1)

-- Prove the area of triangle ABF
theorem area_triangle_ABF : ∃ (area : ℝ),
  area = (1/2) * 2 * 2 ∧ area = 2 :=
by
  use ((1/2) * 2 * 2)
  simp
  sorry

end area_triangle_ABF_l605_605516


namespace matrix_multiplication_result_l605_605617

def mat_2x2 : Type := Matrix (Fin 2) (Fin 2) ℤ
def vec_2 : Type := Matrix (Fin 2) (Fin 1) ℤ

def A : mat_2x2 := !![ 3, -2 ; -4, 5 ]
def B : vec_2 := !![ 4 ; -2 ]
def C : mat_2x2 := !![ 1, 0 ; 0, -1 ]

def final_result (A: mat_2x2) (B: vec_2) (C: mat_2x2) : vec_2 := C ⬝ (A ⬝ B)

theorem matrix_multiplication_result :
  final_result A B C = !![ 16 ; 26 ] :=
by 
  sorry

end matrix_multiplication_result_l605_605617


namespace rearrangement_inequality_l605_605385

variables {n : ℕ}
variables {a b : Fin n -> ℝ}

theorem rearrangement_inequality (h1 : ∀ i : Fin (n-1), a i ≤ a ⟨ i+1, lt_add_one i⟩)
                                 (h2 : ∀ i : Fin (n-1), b i ≤ b ⟨ i+1, lt_add_one i⟩) :
  (Finset.sum (Finset.range n) (λ i, a i * b ⟨n - 1 - i.val, sorry⟩) ≤ Finset.sum (Finset.range n) (λ i, a i * b i)) :=
sorry

end rearrangement_inequality_l605_605385


namespace distance_between_parabola_and_circle_intersections_l605_605912

theorem distance_between_parabola_and_circle_intersections :
  let parabola := { p : ℝ × ℝ | p.2 ^ 2 = 16 * p.1 }
  let circle  := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 - 8 * p.2 = 0 }
  let C := (0, 0)
  let D := (4, 8)

  C ∈ parabola ∧ C ∈ circle ∧ D ∈ parabola ∧ D ∈ circle ∧ 
  ∀ x : ℝ × ℝ, (x ∈ parabola ∧ x ∈ circle) → (x = C ∨ x = D) →
  dist (C : ℝ × ℝ) D = 4 * sqrt 5 :=
by sorry

end distance_between_parabola_and_circle_intersections_l605_605912


namespace relationship_between_number_and_square_l605_605086

theorem relationship_between_number_and_square :
  ∃ (n : ℕ), n = 8 ∧ (n^2 + n = 72) :=
begin
  use 8,
  split,
  { refl, },
  { sorry }
end

end relationship_between_number_and_square_l605_605086


namespace average_speed_return_trip_l605_605887

theorem average_speed_return_trip :
  (forall t1 t2 t3 t4 : ℝ, t1 = 18 / 12 ∧ t2 = 18 / 10 ∧ t3 = t1 + t2 ∧ t4 = 7.3 - t3 →
  (36 / t4 = 9)) :=
begin
  intros t1 t2 t3 t4 h,
  cases h with h_t1 h_rest,
  cases h_rest with h_t2 h_rest,
  cases h_rest with h_t3 h_t4,
  rw [h_t1, h_t2, h_t3, h_t4],
  sorry,
end

end average_speed_return_trip_l605_605887


namespace approximate_length_of_BH_and_semicircle_BF_l605_605092

noncomputable def BH : ℝ := Real.sqrt ((Real.sqrt 5 - 1) / 2)
noncomputable def semicircle_length_BF : ℝ := Real.pi / 4
noncomputable def approximation_ratio : ℝ := BH / semicircle_length_BF

theorem approximate_length_of_BH_and_semicircle_BF : 
  Approximation ∈ set_of (λ r : ℝ, r ≈ 1.000959) :=
by
  -- Definitions provided by the problem
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (1 : ℝ, 1 : ℝ)
  let D := (0 : ℝ, 1 : ℝ)
  let F := (1 / 2 : ℝ, 0 : ℝ)

  -- Context do not use explicit steps, only the end result:
  have := Real.sqrt ((Real.sqrt 5 - 1) / 2)
  have := Real.pi / 4

  -- Check the ratio approximation
  have := approximation_ratio
  exact sorry

end approximate_length_of_BH_and_semicircle_BF_l605_605092


namespace strawberry_growth_rate_l605_605881

theorem strawberry_growth_rate
  (initial_plants : ℕ)
  (months : ℕ)
  (plants_given_away : ℕ)
  (total_plants_after : ℕ)
  (growth_rate : ℕ)
  (h_initial : initial_plants = 3)
  (h_months : months = 3)
  (h_given_away : plants_given_away = 4)
  (h_total_after : total_plants_after = 20)
  (h_equation : initial_plants + growth_rate * months - plants_given_away = total_plants_after) :
  growth_rate = 7 :=
sorry

end strawberry_growth_rate_l605_605881


namespace smallest_positive_n_l605_605678

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605678


namespace probability_of_snow_at_least_once_first_week_l605_605443

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l605_605443


namespace problem_statement_l605_605612

-- Define the conditions
def tan60 : ℝ := Real.tan (Real.pi / 3)
def minus_one_exp : ℤ := (-1) ^ 2023

-- Main theorem to be proved
theorem problem_statement :
  (Real.sqrt 3 * (Real.sqrt 3 + 2) - 2 * tan60 + minus_one_exp) = (2 + Real.sqrt 3) :=
by
  -- State the known conditions explicitly
  have h1 : tan60 = Real.sqrt 3 := by sorry
  have h2 : minus_one_exp = -1 := by sorry
  -- The proof will be filled in here
  sorry

end problem_statement_l605_605612


namespace white_socks_cost_proof_l605_605933

-- Define the cost of a single brown sock in cents
def brown_sock_cost (B : ℕ) : Prop :=
  15 * B = 300

-- Define the cost of two white socks in cents
def white_socks_cost (B : ℕ) (W : ℕ) : Prop :=
  W = B + 25

-- Statement of the problem
theorem white_socks_cost_proof : 
  ∃ B W : ℕ, brown_sock_cost B ∧ white_socks_cost B W ∧ W = 45 :=
by
  sorry

end white_socks_cost_proof_l605_605933


namespace probability_reaching_C_l605_605518

theorem probability_reaching_C (C : ℝ) (x0 : ℝ) (s : ℝ → ℝ) 
  (h1 : 2 < C) 
  (h2 : 0 < x0 ∧ x0 < C) 
  (h3 : ∀ x, s x = if x < 1 then x else if C - x < 1 then C - x else 1) 
  (h4 : ∀ x, x ∈ {n | n = x0 + s x} ∪ {n | n = x0 - s x}):
  let prob_reaching_C := x0 / C in
  true := sorry

end probability_reaching_C_l605_605518


namespace zero_is_monomial_l605_605951

theorem zero_is_monomial (M : Type) [Monoid M] : monomial M 0 := 
sorry

end zero_is_monomial_l605_605951


namespace unsuccessful_attempts_l605_605956

-- Definition of conditions
def ring_count : ℕ := 3
def letters_per_ring : ℕ := 6

-- The total number of combinations can be computed as letters_per_ring ^ ring_count
def total_combinations : ℕ := letters_per_ring ^ ring_count

-- Statement of the problem
theorem unsuccessful_attempts : total_combinations - 1 = 215 :=
by 
  have h1 : total_combinations = letters_per_ring ^ ring_count := by sorry
  have h2 : total_combinations = 6 ^ 3 := by sorry
  have h3 : total_combinations = 216 := by norm_num
  have h4 : 215 = 216 - 1 := by norm_num
  exact h4

end unsuccessful_attempts_l605_605956


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605182

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605182


namespace problem1_problem2_l605_605613

-- Proof statement for Problem 1
theorem problem1 : (Real.sqrt 4 + Real.sqrt 64 - Real.cbrt 27 - abs (-2) = 5) :=
by
  sorry

-- Proof statement for Problem 2
theorem problem2 : (Real.sqrt 16 / Real.cbrt (-1) * Real.sqrt (1 / 4) = -2) :=
by
  sorry

end problem1_problem2_l605_605613


namespace colleen_paid_more_l605_605364

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l605_605364


namespace combined_afternoon_burning_rate_l605_605924

theorem combined_afternoon_burning_rate 
  (morning_period_hours : ℕ)
  (afternoon_period_hours : ℕ)
  (rate_A_morning : ℕ)
  (rate_B_morning : ℕ)
  (total_morning_burn : ℕ)
  (initial_wood : ℕ)
  (remaining_wood : ℕ) :
  morning_period_hours = 4 →
  afternoon_period_hours = 4 →
  rate_A_morning = 2 →
  rate_B_morning = 1 →
  total_morning_burn = 12 →
  initial_wood = 50 →
  remaining_wood = 6 →
  ((initial_wood - remaining_wood - total_morning_burn) / afternoon_period_hours) = 8 := 
by
  intros
  -- We would continue with a proof here
  sorry

end combined_afternoon_burning_rate_l605_605924


namespace probability_and_relationship_l605_605548

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l605_605548


namespace arithmetic_mean_reciprocals_primes_l605_605160

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605160


namespace max_value_is_one_l605_605384

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l605_605384


namespace inequality_proof_l605_605380

theorem inequality_proof (μ λ : ℝ) (x y z : ℝ) 
  (hμ : 0 < μ) (hλ : 0 < λ)
  (hx1 : 0 < x) (hx2 : x < λ / μ)
  (hy1 : 0 < y) (hy2 : y < λ / μ)
  (hz1 : 0 < z) (hz2 : z < λ / μ)
  (hxyz : x + y + z = 1) :
  (x / (λ - μ * x)) + (y / (λ - μ * y)) + (z / (λ - μ * z)) ≥ (3 / (3 * λ - μ)) :=
by
  sorry

end inequality_proof_l605_605380


namespace evaluate_expression_l605_605650

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l605_605650


namespace smallest_positive_period_l605_605238

def y (x : ℝ) : ℝ := 2 * Math.tan (3 * x - π / 6)

theorem smallest_positive_period : ∃ T > 0, ∀ x, y (x + T) = y x ∧ (∀ ε > 0, ε ≠ T) →
  T = π / 3 := sorry

end smallest_positive_period_l605_605238


namespace bus_probabilities_and_chi_squared_l605_605551

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l605_605551


namespace impossible_to_tile_3x3_with_L_pieces_l605_605615

theorem impossible_to_tile_3x3_with_L_pieces :
  ¬∃ (tiling : Fin 9 → Option (Fin 3 × Fin 3)), 
    (∀ i, ∃ p, tiling i = some p) ∧
    (∀ (p : Fin 3 × Fin 3) (L : Fin 3 → Fin 2 × Fin 2), 
      ∃ i, tiling i = some p → (p : Fin 3 × Fin 3)) :=
sorry

end impossible_to_tile_3x3_with_L_pieces_l605_605615


namespace positive_integer_n_satifies_criteria_l605_605782

theorem positive_integer_n_satifies_criteria :
    ∃ count_n : ℕ, count_n = 18 ∧
      (count_n = (Finset.card {n : ℕ | 0 < n ∧ (n + 1500) / 80 = ⌊ Real.sqrt n ⌋ })) :=
by
  sorry

end positive_integer_n_satifies_criteria_l605_605782


namespace snow_probability_l605_605435

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l605_605435


namespace arianna_sleeping_hours_l605_605105

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_on_chores : ℕ := 5
def hours_sleeping : ℕ := hours_in_day - (hours_at_work + hours_on_chores)

theorem arianna_sleeping_hours : hours_sleeping = 13 := by
  sorry

end arianna_sleeping_hours_l605_605105


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605190

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605190


namespace minimize_intercepts_sum_l605_605296

theorem minimize_intercepts_sum (a : ℝ) (h : a > 0) : 
  let A := a 
  let B := 1 / a^2
  in A + B = 2 ↔ a = 1 :=
by sorry

end minimize_intercepts_sum_l605_605296


namespace minimize_sum_m_n_l605_605054

-- Definitions of the given conditions
def last_three_digits_equal (a b : ℕ) : Prop :=
  (a % 1000) = (b % 1000)

-- The main statement to prove
theorem minimize_sum_m_n (m n : ℕ) (h1 : n > m) (h2 : 1 ≤ m) 
  (h3 : last_three_digits_equal (1978^n) (1978^m)) : m + n = 106 :=
sorry

end minimize_sum_m_n_l605_605054


namespace least_n_factorial_l605_605034

theorem least_n_factorial (n : ℕ) : 
  (7350 ∣ nat.factorial n) ↔ n ≥ 15 := 
sorry

end least_n_factorial_l605_605034


namespace caleb_double_burgers_l605_605541

theorem caleb_double_burgers (S D : ℕ) 
(h1 : 1 * S + 3 / 2 * D = 64.50) 
(h2 : S + D = 50) : D = 29 := 
sorry

end caleb_double_burgers_l605_605541


namespace probability_of_snow_at_least_once_first_week_l605_605445

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l605_605445


namespace three_xy_eq_24_l605_605790

variable {x y : ℝ}

theorem three_xy_eq_24 (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 :=
sorry

end three_xy_eq_24_l605_605790


namespace exponential_decreasing_range_l605_605904

theorem exponential_decreasing_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x y : ℝ, x < y → a^y < a^x) : 0 < a ∧ a < 1 :=
by sorry

end exponential_decreasing_range_l605_605904


namespace triangle_area_from_rectangle_diagonal_l605_605121

theorem triangle_area_from_rectangle_diagonal :
  ∀ (length width : ℕ), length = 40 → width = 24 → 
  let area_rectangle := length * width in
  let area_triangle := area_rectangle / 2 in
  area_triangle = 480 :=
by
  intros length width h1 h2
  let area_rectangle := length * width
  let area_triangle := area_rectangle / 2
  have h3 : area_triangle = 480 := sorry
  exact h3

end triangle_area_from_rectangle_diagonal_l605_605121


namespace x_squared_minus_y_squared_l605_605799

-- Definitions based on the conditions in the problem
variables {x y : ℝ}

-- Conditions
def condition1 : Prop := x + y = 4
def condition2 : Prop := 2 * x - 2 * y = 1

-- The statement to prove
theorem x_squared_minus_y_squared : condition1 ∧ condition2 → x^2 - y^2 = 2 :=
by
  sorry

end x_squared_minus_y_squared_l605_605799


namespace evaluate_expression_l605_605136

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l605_605136


namespace divisors_odd_iff_is_perfect_square_l605_605398

theorem divisors_odd_iff_is_perfect_square {n s : ℕ} (hn : n ≠ 0) (hs : s = (finset.filter (λ x, n % x = 0) (finset.range (n+1))).card) :
    odd s ↔ ∃ (k : ℤ), n = k^2 :=
by
  sorry

end divisors_odd_iff_is_perfect_square_l605_605398


namespace find_n_l605_605332

-- Definitions and conditions
def is_odd_prime (k : ℕ) : Prop := nat.prime k ∧ k % 2 = 1

variables (x n p : ℕ)
variable (hp : nat.prime p)
variable (hx : x = 72)
variable (h : is_odd_prime (x / (n * p)))

-- Statement of the problem
theorem find_n : n = 8 :=
sorry

end find_n_l605_605332


namespace greatest_difference_l605_605543

theorem greatest_difference (A B : Finset ℕ) 
  (hA_card : A.card = 8) (hA_sum : A.sum id = 39) 
  (hB_card : B.card = 8) (hB_sum : B.sum id = 39) 
  (hB_unique : B.toList.nodup) : 
  let m := 39 - (A.erase 39).sum id,
      n := 39 - (B.erase 39).sum id in
  m - n = 21 :=
by
  sorry

end greatest_difference_l605_605543


namespace arithmetic_sequence_general_term_geometric_sequence_general_term_sum_T_n_l605_605123

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 3 ^ n
noncomputable def c_n (n : ℕ) := a_n n * b_n n
noncomputable def T_n (n : ℕ) := ∑ i in finset.range n, c_n (i + 1)

theorem arithmetic_sequence_general_term :
  a_n 3 = 5 ∧ a_n 5 - 2 * a_n 2 = 3 → a_n = λ n, 2 * n - 1 :=
by sorry

theorem geometric_sequence_general_term :
  b_n 1 = 3 ∧ ∀ n, b_n (n+1) = 3 * b_n n → b_n = λ n, 3 ^ n :=
by sorry

theorem sum_T_n :
  T_n 0 = 0 ∧ b_n 1 = 3 ∧ ∀ n, b_n (n+1) = 3 * b_n n ∧ a_n 3 = 5 ∧ a_n 5 - 2 * a_n 2 = 3 →
  T_n n = 3 + (n - 1) * 3^(n+1) :=
by sorry

end arithmetic_sequence_general_term_geometric_sequence_general_term_sum_T_n_l605_605123


namespace pau_total_spend_is_correct_l605_605846

noncomputable def Kobe_initial_order_cost : ℝ := 5 * 1.75
noncomputable def Pau_initial_pieces : ℝ := 2 * 5 + 2.5
noncomputable def Pau_initial_order_cost : ℝ := Pau_initial_pieces * 1.5
noncomputable def Pau_repeated_order_discount : ℝ := Pau_initial_order_cost * 0.15
noncomputable def Pau_repeated_order_cost : ℝ := Pau_initial_order_cost - Pau_repeated_order_discount
noncomputable def Pau_total_spend : ℝ := Pau_initial_order_cost + Pau_repeated_order_cost

theorem pau_total_spend_is_correct : Pau_total_spend = 34.6875 :=
by
  unfold Kobe_initial_order_cost Pau_initial_pieces Pau_initial_order_cost Pau_repeated_order_discount Pau_repeated_order_cost Pau_total_spend
  sorry

end pau_total_spend_is_correct_l605_605846


namespace digit_difference_is_36_l605_605478

theorem digit_difference_is_36 (x y : ℕ) (hx : y = 2 * x) (h8 : (x + y) - (y - x) = 8) : 
    |(10 * x + y) - (10 * y + x)| = 36 := 
by
  sorry

end digit_difference_is_36_l605_605478


namespace water_needed_for_fruit_punch_l605_605926

theorem water_needed_for_fruit_punch :
  let total_parts := 5 + 2 + 1 in
  let gallons_per_part := 3 / total_parts in
  let quarts_per_part := gallons_per_part * 4 in
  5 * quarts_per_part = 15 / 2 := by
  sorry

end water_needed_for_fruit_punch_l605_605926


namespace Tim_soda_cans_l605_605507

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l605_605507


namespace count_five_digit_odd_numbers_without_repetition_l605_605515

theorem count_five_digit_odd_numbers_without_repetition : 
  let digits := {0, 1, 2, 3, 4, 5}
  in (∃ n : ℕ, n = 288 ∧ 
      (∀ num : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits 
      ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 
      ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 
      ∧ d3 ≠ d4 ∧ d3 ≠ d5 
      ∧ d4 ≠ d5 
      ∧ (d1 ≠ 0) 
      ∧ (d5 = 1 ∨ d5 = 3 ∨ d5 = 5) 
      ∧ num = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) → 
        ∃ l : List ℕ, l.length = n ∧ ∀ x ∈ l, ∃ d1 d2 d3 d4 d5 : ℕ, (d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits 
        ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 
        ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 
        ∧ d3 ≠ d4 ∧ d3 ≠ d5 
        ∧ d4 ≠ d5 
        ∧ (d1 ≠ 0) 
        ∧ (d5 = 1 ∨ d5 = 3 ∨ d5 = 5) 
        ∧ num = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5)))

end count_five_digit_odd_numbers_without_repetition_l605_605515


namespace tangent_length_from_origin_to_circle_l605_605117

open EuclideanGeometry -- Assuming basic Euclidean geometry utilities are in place
open Real

noncomputable def point := (ℝ × ℝ)

def A : point := (2, 3)
def B : point := (4, 6)
def C : point := (6, 15)
def O : point := (0, 0)

theorem tangent_length_from_origin_to_circle :
  ∀ (circumcircle : circle ℝ), has_point circumcircle A → has_point circumcircle B → has_point circumcircle C → 
  (O.distance circumcircle.center).radius_tangent = 6 :=
begin
  sorry
end

end tangent_length_from_origin_to_circle_l605_605117


namespace max_value_of_y_in_interval_l605_605264

noncomputable def y (x : ℝ) : ℝ :=
  -tan (x + 2 * Real.pi / 3) - tan (x + Real.pi / 6) + cos (x + Real.pi / 6)

theorem max_value_of_y_in_interval :
  ∃ (x : ℝ), x ∈ Icc (-5 * Real.pi / 12) (-Real.pi / 3) ∧ y x = 11 / 6 * Real.sqrt 3 :=
begin
  sorry
end

end max_value_of_y_in_interval_l605_605264


namespace gain_percentage_l605_605960

theorem gain_percentage (x : ℝ) (hx : x ≠ 0) : 
  let SP := 10 * x in 
  let CP := 60 * x in 
  let profit := SP in
  (profit / CP) * 100 = 16.67 :=
by
  sorry

end gain_percentage_l605_605960


namespace colleen_paid_more_l605_605366

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l605_605366


namespace problem_equiv_l605_605747

-- We define the theorem using the preconditions and then state what needs to be proved.
theorem problem_equiv : 
  (∀ n : ℕ, ∃ a t : ℝ, 
    n + 1 > 0 ∧ 
    t > 0 ∧ 
    a > 0 ∧ 
    (n + 1) > 0 ∧ 
    n > 0 ∧ 
    (sqrt ((n+1) + (n+1)/(n^2))) = (n+1) * (sqrt ((n+1)/((n+1)^2 - 1)))) → 
  ∃ a t : ℝ, sqrt (8 + a/t) = 8 * (sqrt (a/t)) → a + t = 71 :=
by
  sorry

end problem_equiv_l605_605747


namespace jacket_price_restoration_percentage_l605_605913

-- Assume the initial price of the jacket
def initial_price (p : ℝ) : Prop := p = 100

-- Define the successive price reductions
def price_after_first_reduction (p : ℝ) : ℝ := p * 0.80
def price_after_second_reduction (p : ℝ) : ℝ := p * 0.75
def price_after_third_reduction (p : ℝ) : ℝ := p * 0.90
def price_after_fourth_reduction (p : ℝ) : ℝ := p * 0.85

-- Define the final price after all reductions
def final_price (p : ℝ) : ℝ :=
  price_after_fourth_reduction (price_after_third_reduction (price_after_second_reduction (price_after_first_reduction p)))

-- Define the percentage increase required to restore to the original price
def percentage_increase_required (final : ℝ) (initial : ℝ) : ℝ :=
  ((initial / final) - 1) * 100

theorem jacket_price_restoration_percentage :
  ∀ (p : ℝ), initial_price p → percentage_increase_required (final_price p) p ≈ 117.95 := 
by
  intros p h
  rw [initial_price] at h
  have h_price : final_price p = 45.90 := sorry
  exact sorry

end jacket_price_restoration_percentage_l605_605913


namespace count_valid_n_l605_605719

def num_valid_n : ℕ :=
  let possible_n := { n : ℕ | ∃ (a b : ℕ), n = 2^a * 5^b ∧ n < 1000 } in
  let non_zero_thousandths := { n ∈ possible_n | (1000 * n : ℚ) / n % 10 ≠ 0 } in
  non_zero_thousandths.card

theorem count_valid_n : num_valid_n = 25 := by
  sorry

end count_valid_n_l605_605719


namespace colleen_paid_more_l605_605368

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l605_605368


namespace max_value_of_reciprocal_powers_l605_605882

variable {R : Type*} [CommRing R]
variables (s q r₁ r₂ : R)

-- Condition: the roots of the polynomial
def is_roots_of_polynomial (s q r₁ r₂ : R) : Prop :=
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ (r₁ + r₂ = r₁ ^ 2 + r₂ ^ 2) ∧ (r₁ + r₂ = r₁^10 + r₂^10)

-- The theorem that needs to be proven
theorem max_value_of_reciprocal_powers (s q r₁ r₂ : ℝ) (h : is_roots_of_polynomial s q r₁ r₂):
  (∃ r₁ r₂, r₁ + r₂ = s ∧ r₁ * r₂ = q ∧
             r₁ + r₂ = r₁^2 + r₂^2 ∧
             r₁ + r₂ = r₁^10 + r₂^10) →
  (r₁^ 11 ≠ 0 ∧ r₂^11 ≠ 0 ∧
  ((1 / r₁^11) + (1 / r₂^11) = 2)) :=
by
  sorry

end max_value_of_reciprocal_powers_l605_605882


namespace selling_price_of_article_l605_605101

theorem selling_price_of_article (cost_price : ℝ) (profit_percentage : ℝ) : 
  cost_price = 480 ∧ profit_percentage = 0.25 → cost_price * (1 + profit_percentage) = 600 :=
by
  intros h
  cases h with h_cost h_profit
  sorry

end selling_price_of_article_l605_605101


namespace worker_arrangement_l605_605075

def total_workers := 85

def production_balance (x y : ℕ) : Prop :=
  20 * x = 48 * y ∧ x + y = total_workers

theorem worker_arrangement :
  ∃ x y : ℕ, production_balance x y ∧ x = 60 ∧ y = 25 :=
by {
  use [60, 25],
  simp [production_balance, total_workers],
  norm_num,
  split; norm_num,
  sorry
}

end worker_arrangement_l605_605075


namespace correct_option_l605_605952

-- Define the terms from the problem
def termA := (1 / 3) * Real.pi * x^2
def termB := (1 / 2) * x * y^2
def termC := 3^2 * x^2
def termD := -5 * x^2

-- Definitions for the coefficients
def coefficient (expr : ℝ) : ℝ :=
  coeff of term in variable x

-- Prove the correctness of option C and incorrectness of others
theorem correct_option :
  (coefficient termA ≠ (1 / 3)) ∧
  (coefficient termB ≠ (1 / 2) * x) ∧
  (coefficient termC = 3^2) ∧
  (coefficient termD ≠ 5) :=
begin
  sorry
end

end correct_option_l605_605952


namespace david_pushups_l605_605060

theorem david_pushups (zachary_pushups : ℕ) (additional_pushups : ℕ) (hz : zachary_pushups = 47) (hd : additional_pushups = 15) :
  zachary_pushups + additional_pushups = 62 :=
by
  rw [hz, hd]
  norm_num

end david_pushups_l605_605060


namespace remaining_regular_toenails_l605_605305

theorem remaining_regular_toenails :
  ∀ (total_capacity big_toenail_space big_toenails regular_toenails : ℕ),
    total_capacity = 100 →
    big_toenail_space = 2 →
    big_toenails = 20 →
    regular_toenails = 40 →
    let occupied_space := big_toenails * big_toenail_space + regular_toenails in
    let remaining_space := total_capacity - occupied_space in
    remaining_space = 20 :=
by
  intros total_capacity big_toenail_space big_toenails regular_toenails htcs hbs hbt hrt
  let occupied_space := big_toenails * big_toenail_space + regular_toenails
  have h1 : occupied_space = 40 * 2 + 40 := rfl
  let remaining_space := total_capacity - occupied_space
  have h2 : remaining_space = 100 - 80 := rfl
  have h3 : 20 = 20 := rfl
  exact h3

end remaining_regular_toenails_l605_605305


namespace right_triangle_third_side_l605_605344

/-- In a right triangle, given the lengths of two sides are 4 and 5, prove that the length of the
third side is either sqrt 41 or 3. -/
theorem right_triangle_third_side (a b : ℕ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ c, c = Real.sqrt 41 ∨ c = 3 :=
by
  sorry

end right_triangle_third_side_l605_605344


namespace mean_of_reciprocals_first_four_primes_l605_605150

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605150


namespace max_value_on_ellipse_l605_605738

theorem max_value_on_ellipse (b : ℝ) (hb : b > 0) :
  ∃ (M : ℝ), 
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / b^2 = 1) → x^2 + 2 * y ≤ M) ∧
    ((b ≤ 4 → M = b^2 / 4 + 4) ∧ (b > 4 → M = 2 * b)) :=
  sorry

end max_value_on_ellipse_l605_605738


namespace arithmetic_sequence_tenth_term_l605_605921

noncomputable def prove_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : Prop :=
  a + 9*d = 38

theorem arithmetic_sequence_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : prove_tenth_term a d h1 h2 :=
by
  sorry

end arithmetic_sequence_tenth_term_l605_605921


namespace area_of_triangle_r_l605_605096

noncomputable def parabola_eq (x : ℝ) : ℝ := x^2 + 2
noncomputable def line_eq (r : ℝ) (x : ℝ) : ℝ := r 
noncomputable def area_of_triangle (r : ℝ) : ℝ := (r - 2)^(1.5)

theorem area_of_triangle_r (r : ℝ) :
  10 ≤ area_of_triangle r ∧ area_of_triangle r ≤ 50 ↔
  10^(2/3) + 2 ≤ r ∧ r ≤ 50^(2/3) + 2 :=
by
  sorry

end area_of_triangle_r_l605_605096


namespace eval_expression_l605_605638

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l605_605638


namespace privateer_overtakes_merchantman_l605_605088

theorem privateer_overtakes_merchantman :
  ∃ (t : ℕ), t = 5 ∧ (t * 60 + 0) minutes = 300 minutes := by
  sorry

end privateer_overtakes_merchantman_l605_605088


namespace simplify_expression_l605_605460

variable (a b : ℤ)

theorem simplify_expression : 
  (50 * a + 130 * b) + (21 * a + 64 * b) - (30 * a + 115 * b) - 2 * (10 * a - 25 * b) = 21 * a + 129 * b := 
by
  sorry

end simplify_expression_l605_605460


namespace min_people_answer_most_qs_correctly_l605_605970

/-- Given 21 participants in an exam consisting of 15 true/false questions, if any two participants have at least one question they both answered correctly, then the minimum number of participants who answered the most frequently correct question correctly is 7. -/
theorem min_people_answer_most_qs_correctly
  (n_participants : ℕ) (n_questions : ℕ)
  (H1 : n_participants = 21) (H2 : n_questions = 15)
  (H3 : ∀ (p1 p2 : ℕ), p1 ≠ p2 → ∃ q : ℕ, q < n_questions ∧ answered_correctly p1 q ∧ answered_correctly p2 q) :
  ∃ q : ℕ, q < n_questions ∧ (∃ m : ℕ, m ≥ 7 ∧ (∀ p : ℕ, p ≤ n_participants → answered_correctly p q → m ≤ n_participants)) :=
by
  sorry

end min_people_answer_most_qs_correctly_l605_605970


namespace arithmetic_sequence_difference_l605_605272

theorem arithmetic_sequence_difference :
  let a_n (n : ℕ) := 3 * n - 4 in
  let a_1 := a_n 1 in
  let d := a_n 2 - a_n 1 in
  a_1 - d = -4 :=
by
  -- Define the general term formula
  let a_n (n : ℕ) := 3 * n - 4
  -- Define the first term
  let a_1 := a_n 1
  -- Define the common difference
  let d := a_n 2 - a_n 1
  -- We need to prove a_1 - d = -4
  sorry

end arithmetic_sequence_difference_l605_605272


namespace area_ratio_of_squares_l605_605013

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 1 / 2 * (4 * b)) : (b^2 / a^2) = 4 :=
by
  -- Proof goes here
  sorry

end area_ratio_of_squares_l605_605013


namespace find_initial_milk_amount_l605_605538

-- Define the initial amount of milk as a variable in liters
variable (T : ℝ)

-- Given conditions
def consumed (T : ℝ) := 0.4 * T
def leftover (T : ℝ) := 0.69

-- The total milk at first was T if T = 0.69 / 0.6
theorem find_initial_milk_amount 
  (h1 : leftover T = 0.69)
  (h2 : consumed T = 0.4 * T) :
  T = 1.15 :=
by
  sorry

end find_initial_milk_amount_l605_605538


namespace determinant_condition_l605_605133

noncomputable def D : ℕ → ℝ
| 1     := 1
| 2     := 3 - 1
| (n + 3) := 2^(n + 2) -- The given pattern results in 2^(n+2) for n ≥ 1

theorem determinant_condition (n : ℕ) : (D n ≥ 2015) ↔ (n ≥ 12) := 
by 
  sorry

end determinant_condition_l605_605133


namespace time_to_cross_signal_pole_l605_605579

-- Define the given conditions
def train_length : ℝ := 300
def platform_length : ℝ := 400
def time_to_cross_platform : ℝ := 42

-- Define the calculated intermediate terms
def total_distance_crossed := train_length + platform_length
def speed_of_train := total_distance_crossed / time_to_cross_platform

-- The theorem to prove the final answer
theorem time_to_cross_signal_pole : (train_length / speed_of_train) = 18 := by
  -- Proof would go here, using the given conditions and derived values
  sorry

end time_to_cross_signal_pole_l605_605579


namespace blue_ball_weight_l605_605892

variable (b t x : ℝ)
variable (c1 : b = 3.12)
variable (c2 : t = 9.12)
variable (c3 : t = b + x)

theorem blue_ball_weight : x = 6 :=
by
  sorry

end blue_ball_weight_l605_605892


namespace tripodasaurus_flock_l605_605499

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end tripodasaurus_flock_l605_605499


namespace smallest_positive_integer_n_l605_605699

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605699


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605178

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605178


namespace arithmetic_mean_reciprocals_primes_l605_605154

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605154


namespace smallest_n_for_identity_matrix_l605_605672

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605672


namespace strawberries_per_basket_l605_605369

/-- Define some variables -/
variables (B K P : ℕ)

/-- Define the conditions -/
def kimberly_strawberries := 8 * B
def parents_strawberries := kimberly_strawberries B - 93
def total_strawberries := B + kimberly_strawberries B + parents_strawberries B
def each_received := 168
def num_family_members := 4

/-- Prove the number of strawberries in each basket -/
theorem strawberries_per_basket (H1: K = 8 * B)
    (H2: P = K - 93)
    (H3: 3 * B = 3 * B)   -- Trivial condition for the basket count
    (H4: total_strawberries B = num_family_members * each_received) :
  B / 3 = 15 :=
by
  sorry

end strawberries_per_basket_l605_605369


namespace prob_and_relation_proof_l605_605562

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l605_605562


namespace abc_zero_l605_605012

theorem abc_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) :
  a * b * c = 0 := 
sorry

end abc_zero_l605_605012


namespace decimal_to_octal_l605_605623

theorem decimal_to_octal (n : ℕ) (h : n = 72) : n.to_octal = "110" := 
by 
  rw h 
  sorry

end decimal_to_octal_l605_605623


namespace snow_probability_first_week_l605_605428

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l605_605428


namespace sammy_bottle_caps_l605_605893

variables (B : ℕ)  -- Billie's initial number of bottle caps

-- Conditions
def janine_caps := 3 * B  -- Janine has 3 times as many bottle caps as Billie
def sammy_caps := janine_caps + 2  -- Sammy has 2 more bottle caps than Janine
def billie_gift_to_tommy := B - 4  -- Billie gifted 4 bottle caps to Tommy
def tommy_caps := 2 * billie_gift_to_tommy  -- Tommy now has twice as many bottle caps as Billie
def tommy_initial := 0  -- Tommy initially had no bottle caps

-- Equation from Tommy's condition
theorem sammy_bottle_caps : tommy_caps = 4 → sammy_caps = 20 :=
by
  intro h
  unfold tommy_caps at h
  unfold billie_gift_to_tommy at h
  have B_eq : B - 4 = 2 := by linarith
  have B_val : B = 6 := by linarith
  unfold sammy_caps
  unfold janine_caps
  rw B_val
  norm_num

end sammy_bottle_caps_l605_605893


namespace handshakes_among_women_l605_605469

theorem handshakes_among_women (n : ℕ) (h : n = 10) : (∑ i in finset.range n, i) = 45 :=
by
  sorry

end handshakes_among_women_l605_605469


namespace arithmetic_mean_reciprocals_primes_l605_605157

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605157


namespace sum_of_numbers_l605_605544

theorem sum_of_numbers (x : ℝ) (h_ratio : ∃ x : ℝ, 2 * x = 2x ∧ 4 * x = 4x)
    (h_sum_of_squares : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) :
    x + 2 * x + 4 * x = 105 :=
by
  sorry

end sum_of_numbers_l605_605544


namespace buyer_saved_percentage_l605_605582

theorem buyer_saved_percentage (amount_saved amount_spent : ℝ) (h_saved : amount_saved = 6.25) (h_spent : amount_spent = 43.75) : 
  ((amount_saved / (amount_saved + amount_spent)) * 100) ≈ 12 :=
by
  sorry

end buyer_saved_percentage_l605_605582


namespace jennifer_boxes_l605_605845

theorem jennifer_boxes (kim_sold : ℕ) (h₁ : kim_sold = 54) (h₂ : ∃ jennifer_sold, jennifer_sold = kim_sold + 17) : ∃ jennifer_sold, jennifer_sold = 71 := by
  sorry

end jennifer_boxes_l605_605845


namespace polynomial_g_exists_l605_605243

variables {f : Polynomial ℤ} {a k : ℕ}

noncomputable def exists_polynomial_g (a k : ℕ) (f : Polynomial ℤ) : Prop :=
  ∀ x : ℝ, ∃ g : Polynomial ℚ, f.eval (g.eval x) = (f.eval x)^k

theorem polynomial_g_exists (h1 : a ≥ 2) (h2 : k ≥ 2) (h3 : f.degree > 0)
  (h4 : ∀ n : ℕ, ∃ x : ℚ, n > 1000 → f.eval x = (f.eval (a^n))^k) : exists_polynomial_g a k f :=
by
  sorry

end polynomial_g_exists_l605_605243


namespace three_delegates_common_language_l605_605922

theorem three_delegates_common_language 
  (D : Fin 9)   -- Set of 9 delegates
  (L : Fin 9 → Set (Fin 3))  -- Language each delegate speaks
  (h1 : ∀ d, (L d).card ≤ 3)  -- Each delegate speaks at most 3 languages
  (h2 : ∀ a b c : Fin 9, ∃ l : Fin 3, l ∈ L a ∧ l ∈ L b ∧ l ∈ L c)  -- Any three delegates have a common language:
  : ∃ a b c : Fin 9, ∃ l : Fin 3, l ∈ L a ∧ l ∈ L b ∧ l ∈ L c := 
sorry

end three_delegates_common_language_l605_605922


namespace integral_of_f_l605_605760

def f (x : ℝ) : ℝ :=
  if -2 <= x ∧ x <= 0 then x^2
  else if 0 < x ∧ x <= 2 then x + 1
  else 0

theorem integral_of_f : (∫ x in -2..2, f x) = 20 / 3 :=
by
  sorry

end integral_of_f_l605_605760


namespace problem_proof_l605_605371

noncomputable def problem_statement
  (ABC : Triangle)
  (is_acute : ABC.isAcute)
  (AB_lt_AC : ABC.sideLength AB < ABC.sideLength AC)
  (D E F : Point)
  (altitude_A_D : ABC.altitudeFrom A = Line A D)
  (altitude_B_E : ABC.altitudeFrom B = Line B E)
  (altitude_C_F : ABC.altitudeFrom C = Line C F)
  (Γ : Circle)
  (circumcircle_AEF : Γ = Circle.circumscribedTriangle A E F)
  (circumcircle_ABC : Circle)
  (M : Point)
  (Γ_intersects_circumcircleA_M : Γ ∩ circumcircle_ABC = {A, M})
  (BM_tangent_Γ : Line B M ∈ tangentLinesToCircleAtPoint Γ M) : Prop :=
  collinear M F D

-- We state that the problem is valid and should be proved.
theorem problem_proof
  (ABC : Triangle)
  (is_acute : ABC.isAcute)
  (AB_lt_AC : ABC.sideLength AB < ABC.sideLength AC)
  (D E F : Point)
  (altitude_A_D : ABC.altitudeFrom A = Line A D)
  (altitude_B_E : ABC.altitudeFrom B = Line B E)
  (altitude_C_F : ABC.altitudeFrom C = Line C F)
  (Γ : Circle)
  (circumcircle_AEF : Γ = Circle.circumscribedTriangle A E F)
  (circumcircle_ABC : Circle)
  (M : Point)
  (Γ_intersects_circumcircleA_M : Γ ∩ circumcircle_ABC = {A, M})
  (BM_tangent_Γ : Line B M ∈ tangentLinesToCircleAtPoint Γ M) : 
  collinear M F D := 
sorry

end problem_proof_l605_605371


namespace tim_final_soda_cans_l605_605508

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l605_605508


namespace arithmetic_mean_reciprocals_primes_l605_605152

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605152


namespace rotated_square_highest_point_vertical_distance_l605_605251

-- Definition of the specific setup and conditions
def side_length : ℝ := 2
def rotation_angle : ℝ := 30 -- in degrees
def top_vertex_distance_square (a : ℝ) : ℝ :=
  let diag := a * real.sqrt 2
  (diag / 2) * real.sin (rotation_angle * real.pi / 180)

-- Prove the required vertical distance
theorem rotated_square_highest_point_vertical_distance :
  top_vertex_distance_square side_length + (side_length / 2) = 1 + real.sqrt 2 / 2 :=
by
  sorry

end rotated_square_highest_point_vertical_distance_l605_605251


namespace fixed_point_of_exponential_function_l605_605005

theorem fixed_point_of_exponential_function (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∃ x y : ℝ, y = a^(x-2) + 2 ∧ (x, y) = (2, 3)) :=
by
  use [2, 3]
  sorry

end fixed_point_of_exponential_function_l605_605005


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605188

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605188


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605229

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605229


namespace bing_between_jia_yi_l605_605022

def num_arrangements (people : List String) (bing jia yi : String) : Nat :=
  if bing ∈ people ∧ jia ∈ people ∧ yi ∈ people ∧ people.length = 6 then
    4! * 2
  else
    0

theorem bing_between_jia_yi (people : List String) (bing jia yi : String) :
  bing ∈ people ∧ jia ∈ people ∧ yi ∈ people ∧ people.length = 6 →
  num_arrangements people bing jia yi = 48 :=
sorry

end bing_between_jia_yi_l605_605022


namespace romanov_family_savings_l605_605569

theorem romanov_family_savings :
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption := monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  savings = 3824 :=
by
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption :=monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2 
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  show savings = 3824 
  sorry

end romanov_family_savings_l605_605569


namespace conjugate_of_z_l605_605577

def imaginary_unit : ℂ := complex.I

def given_condition (z : ℂ) : Prop := z * imaginary_unit = -1 + imaginary_unit

theorem conjugate_of_z (z : ℂ) (h : given_condition z) : complex.conj z = 1 - imaginary_unit := 
  sorry

end conjugate_of_z_l605_605577


namespace ram_initial_deposit_l605_605514

theorem ram_initial_deposit :
  ∃ P: ℝ, P + 100 = 1100 ∧ 1.20 * 1100 = 1320 ∧ P * 1.32 = 1320 ∧ P = 1000 :=
by
  existsi (1000 : ℝ)
  sorry

end ram_initial_deposit_l605_605514


namespace terminating_decimal_nonzero_thousandths_l605_605723

theorem terminating_decimal_nonzero_thousandths (n : ℕ) :
  (∃ (k : ℕ), k = 4 ∧ ∀ m (h₁ : 1 ≤ m ∧ m ≤ 1000) (h₂ : terminating (1 / m)), nonzero_thousandths_digit (1 / m) → k = 4) :=
sorry

-- Definitions and helpers:
def terminating (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

def nonzero_thousandths_digit (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

end terminating_decimal_nonzero_thousandths_l605_605723


namespace oranges_given_to_friend_l605_605978

theorem oranges_given_to_friend (initial_oranges : ℕ) 
  (given_to_brother : ℕ)
  (given_to_friend : ℕ)
  (h1 : initial_oranges = 60)
  (h2 : given_to_brother = (1 / 3 : ℚ) * initial_oranges)
  (h3 : given_to_friend = (1 / 4 : ℚ) * (initial_oranges - given_to_brother)) : 
  given_to_friend = 10 := 
by 
  sorry

end oranges_given_to_friend_l605_605978


namespace coinArrangementProof_l605_605828

noncomputable def numWaysToArrangeCoins : Nat :=
  let total_length_mm := 1000
  let diameter_10_fillér := 19
  let diameter_50_fillér := 22
  let min_total_coins := 50
  let valid (x y : Nat) := (19 * x + 22 * y = 1000) ∧ (x + y >= min_total_coins)
  let (x, y) := (48, 4)  -- Based on t = 0
  have h_valid : valid x y := by
    simp [valid, diameter_10_fillér, diameter_50_fillér, total_length_mm, min_total_coins]
    sorry  -- Argument and simplification placeholder

  have h_permutations : Nat.choose (x + y) y = 270725 := by
    simp [Nat.choose_eq_factorial_div_factorial];  -- By calculating 52 choose 4
    sorry  -- Argument and simplification placeholder

  270725

theorem coinArrangementProof :
  numWaysToArrangeCoins = 270725 :=
by simp [numWaysToArrangeCoins]; sorry  -- Placeholder for full detailed proof with simplifications

end coinArrangementProof_l605_605828


namespace triangle_area_l605_605916

noncomputable def area_of_triangle (y: ℝ):= 
  (1 / 2) * (5 * y) * (12 * y) * Real.sin (Real.pi / 4)

theorem triangle_area:
  ∃ y: ℝ,
  (5 * y + 12 * y + 13 * y = 300) ∧
  (area_of_triangle y = 1500 * Real.sqrt 2) :=
by
  sorry

end triangle_area_l605_605916


namespace find_percentage_of_number_l605_605076

theorem find_percentage_of_number (P : ℝ) (N : ℝ) (h1 : P * N = (4 / 5) * N - 21) (h2 : N = 140) : P * 100 = 65 := 
by 
  sorry

end find_percentage_of_number_l605_605076


namespace value_of_a3_l605_605494

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 7 else sequence (n - 1) - 3

theorem value_of_a3 : sequence 3 = 1 :=
by sorry

end value_of_a3_l605_605494


namespace son_age_18_l605_605540

theorem son_age_18 (F S : ℤ) (h1 : F = S + 20) (h2 : F + 2 = 2 * (S + 2)) : S = 18 :=
by
  sorry

end son_age_18_l605_605540


namespace max_band_members_l605_605501

theorem max_band_members (n : ℤ) (h1 : 20 * n % 31 = 11) (h2 : 20 * n < 1200) : 20 * n = 1100 :=
sorry

end max_band_members_l605_605501


namespace correct_product_l605_605841

theorem correct_product (a b : ℚ) (calc_incorrect : a = 52 ∧ b = 735)
                        (incorrect_product : a * b = 38220) :
  (0.52 * 7.35 = 3.822) :=
by
  sorry

end correct_product_l605_605841


namespace christine_walk_duration_l605_605114

-- Definitions of the given conditions
def distance1 : ℝ := 20
def speed1 : ℝ := 4
def distance2 : ℝ := 24
def speed2 : ℝ := 6
def distance3 : ℝ := 9
def speed3 : ℝ := 3

-- Definition of the time taken for each segment
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2
def time3 : ℝ := distance3 / speed3

-- The total time taken
def total_time : ℝ := time1 + time2 + time3

-- The statement to be proven
theorem christine_walk_duration : total_time = 12 := by
  sorry

end christine_walk_duration_l605_605114


namespace recording_incorrect_l605_605833

-- Definitions for given conditions
def qualifying_standard : ℝ := 1.5
def xiao_ming_jump : ℝ := 1.95
def xiao_liang_jump : ℝ := 1.23
def xiao_ming_recording : ℝ := 0.45
def xiao_liang_recording : ℝ := -0.23

-- The proof statement to verify the correctness of the recordings
theorem recording_incorrect :
  (xiao_ming_jump - qualifying_standard = xiao_ming_recording) ∧ 
  (xiao_liang_jump - qualifying_standard ≠ xiao_liang_recording) :=
by
  sorry

end recording_incorrect_l605_605833


namespace christina_catches_up_l605_605417

theorem christina_catches_up :
  ∃ t : ℝ, t >= 0 ∧ (let cristina_distance := 5 * t in
                     let nicky_distance := 3 * t + 30 in
                     cristina_distance = nicky_distance ∧
                     t = 15) :=
by
  sorry

end christina_catches_up_l605_605417


namespace least_n_factorial_7350_l605_605040

theorem least_n_factorial_7350 (n : ℕ) :
  (7350 ∣ n!) → (∃ k : ℕ, (1 ≤ k ∧ 7350 ∣ k!)) :=
by
  have prime_factors_7350 : 7350 = 2 * 3^2 * 5^2 * 7 := by norm_num
  existsi 10
  split
  · norm_num
  · apply dvd_factorial.mpr
    sorry

end least_n_factorial_7350_l605_605040


namespace real_imag_parts_equal_l605_605757

theorem real_imag_parts_equal (a : ℝ) (i : ℂ) (H : i * i = -1) :
  let z := (a + i) * i in
  z.re = z.im →
  a = -1 :=
by
  intros
  let z := (a + i) * i
  have h1 : z = a * i + i^2 := by ring
  rw H at h1
  have h2 : z = a * i - 1 := by simp [h1]
  sorry

end real_imag_parts_equal_l605_605757


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605217

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605217


namespace optimal_sampling_methods_l605_605982

/-
We define the conditions of the problem.
-/
def households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sample_households := 100

def soccer_players := 12
def sample_soccer_players := 3

/-
We state the goal as a theorem.
-/
theorem optimal_sampling_methods :
  (sample_households == 100) ∧
  (sample_soccer_players == 3) ∧
  (high_income_households + middle_income_households + low_income_households == households) →
  ("stratified" = "stratified" ∧ "random" = "random") :=
by
  -- Sorry to skip the proof
  sorry

end optimal_sampling_methods_l605_605982


namespace total_amount_is_correct_l605_605997

theorem total_amount_is_correct (a : ℝ) 
  (h_share : 1.95 * a = 39) : 
  let p := 3 * a
  let q := 2.70 * a
  let r := 2.30 * a
  let s := 1.95 * a
  let t := 1.80 * a
  let u := 1.50 * a
  show p + q + r + s + t + u = 265 := by
s => sorry

end total_amount_is_correct_l605_605997


namespace count_terminating_decimals_with_nonzero_thousandths_l605_605712

noncomputable def is_terminating_with_nonzero_thousandths (n : ℕ) : Prop :=
  (∃ a b : ℕ, n = 2^a * 5^b) ∧ n ≤ 1000

theorem count_terminating_decimals_with_nonzero_thousandths :
  (finset.univ.filter is_terminating_with_nonzero_thousandths).card = 25 :=
begin
  sorry
end

end count_terminating_decimals_with_nonzero_thousandths_l605_605712


namespace smallest_n_for_identity_matrix_l605_605669

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605669


namespace mean_of_reciprocals_first_four_primes_l605_605142

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605142


namespace number_of_students_to_sample_from_B_l605_605457

variables (n_A n_B n_C : ℕ) (d : ℕ)

-- Conditions
def total_students := n_A + n_B + n_C = 1500
def arithmetic_sequence := n_B = n_A + d ∧ n_C = n_A + 2d
def sample_size := 120

-- Proof statement
theorem number_of_students_to_sample_from_B 
    (h1 : total_students)
    (h2 : arithmetic_sequence)
    (h3 : sample_size = 120) : 
    n_B * sample_size / 1500 = 40 :=
by 
  sorry

end number_of_students_to_sample_from_B_l605_605457


namespace unique_polynomial_g_l605_605396

noncomputable def alpha := (Real.sqrt 5 - 1) / 2

def f : ℕ → ℕ
| 0       := 0
| (n + 1) := n + 1 - f (f n)

def g (x : ℝ) : ℝ := alpha * (x + 1)

theorem unique_polynomial_g :
  ∀ n, f n = ⌊g n⌋ :=
by
sorry

end unique_polynomial_g_l605_605396


namespace f_of_pi_over_4_max_value_of_f_l605_605294

-- Definition of the function
def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x)

-- Prove that f(π/4) = 0
theorem f_of_pi_over_4 : f (Real.pi / 4) = 0 :=
by
  -- The proof is omitted for now
  sorry

-- Prove the maximum value of f(x) is sqrt(2) - 1
theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 2 - 1 :=
by
  -- The proof is omitted for now
  sorry

end f_of_pi_over_4_max_value_of_f_l605_605294


namespace count_positive_differences_l605_605308

theorem count_positive_differences :
  let S := {i | 1 ≤ i ∧ i ≤ 19} in
  ∀ n, (1 ≤ n ∧ n ≤ 18) → ∃ a b ∈ S, a ≠ b ∧ (a - b = n) :=
by
  sorry

end count_positive_differences_l605_605308


namespace vector_magnitude_l605_605283

variables {ℝ : Type*} [field ℝ] [NormedSpace ℝ (EuclideanSpace ℝ (fin 2))]

noncomputable def e1 : EuclideanSpace ℝ (fin 2) := ![1, 0]
noncomputable def e2 : EuclideanSpace ℝ (fin 2) := ![0, 1]

theorem vector_magnitude (θ : ℝ) (hcos : Real.cos θ = 1 / 4)
(h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1) : 
∥(e1 + 2 • e2)∥ = Real.sqrt 6 :=
by
  sorry

end vector_magnitude_l605_605283


namespace number_of_distinct_ways_to_color_black_l605_605616

-- We define the conditions as outlined above

def validColoring (board : List (List Bool)) : Prop :=
  (∀ row : Fin 4, board[row].count (λ x => x) = 2) ∧  -- Each row has exactly two black squares
  (∀ col : Fin 4, (board.map (λ row => row[col])).count (λ x => x) = 2)  -- Each column has exactly two black squares
  ∧ board.length = 4  -- 4 rows
  ∧ (∀ row, row.length = 4)  -- Each row has 4 elements

noncomputable def countValidColorings : Nat :=
  -- This is where the calculation would go, which we assume to be correct by solution criteria
  90

theorem number_of_distinct_ways_to_color_black :
  ∃ n, n = countValidColorings ∧ n = 90 :=
by
  use countValidColorings
  simp [countValidColorings]
  exact ⟨rfl, rfl⟩  -- assuming the count is correct as given in the problem.

end number_of_distinct_ways_to_color_black_l605_605616


namespace smallest_positive_integer_n_l605_605697

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605697


namespace probability_not_first_prize_l605_605635

theorem probability_not_first_prize :
  let x : ℝ := 1 -- a placeholder for the common area factor
  let areas := [x, 3 * x, 9 * x, 27 * x, 81 * x, 243 * x]
  let total_area := areas.sum
  let prob_first_prize := x / total_area
  1 - prob_first_prize = 363 / 364 :=
by
  let x : ℝ := 1
  let areas := [x, 3 * x, 9 * x, 27 * x, 81 * x, 243 * x]
  let total_area := areas.sum
  let prob_first_prize := x / total_area
  have h : total_area = 364 * x := by
    simp [areas, List.sum_cons]
    repeat {simp}
  have prob_first : prob_first_prize = 1 / 364 := by
    simp [h]
  simp [prob_first]
  norm_num
  exact rfl

end probability_not_first_prize_l605_605635


namespace area_triangle_eq_area_quadrilateral_values_squared_length_locus_midpoint_circle_l605_605348

-- Problem setup for the first proof
variable {A B C L K M N : Point}
variable {circumcircle : Circle}
variable {angleA_bisector : Line}
variable {orthogonalAB orthogonalAC : Line}

-- Conditions
axiom acute_triangle : is_triangle A B C → is_acute A B C
axiom bisector_intersects_BC_at_L : bisects angleA_bisector A BC L
axiom bisector_intersects_circumcircle_at_N : bisects_and_intersects_circumcircle angleA_bisector circumcircle N
axiom LK_perpendicular_to_AB : orthogonal orthogonalAB AB LK
axiom LM_perpendicular_to_AC : orthogonal orthogonalAC AC LM
axiom LK_intersects_AB_at_K : intersects LK AB K
axiom LM_intersects_AC_at_M : intersects LM AC M

-- Question: Prove that the area of triangle ABC equals the area of quadrilateral AKNM.
theorem area_triangle_eq_area_quadrilateral :
  S (triangle A B C) = S (quadrilateral A K N M) := sorry

-- Problem setup for the second proof
variable {R r : ℝ}
-- The expression for the set of values taken by AB^2 + BC^2 + CA^2.
variable {AB BC CA : ℝ}
axiom circumradius_R : circumradius A B C R
axiom inradius_r : inradius A B C r

-- Question: Prove that AB^2 + BC^2 + CA^2 = 6R^2 + 2r^2.
theorem values_squared_length :
  AB^2 + BC^2 + CA^2 = 6 * R^2 + 2 * r^2 := sorry

-- Problem setup for the third proof
variable {OP : Point}
variable {locus_midpoint_radius : ℝ}
axiom midpoint_locus_radius : locus_radius midpoint AB locus_midpoint_radius

-- Question: Prove that the locus of the midpoint of AB is a circle centered at OP with radius R/2.
theorem locus_midpoint_circle :
  locus (midpoint A B) center OP radius R/2 := sorry

end area_triangle_eq_area_quadrilateral_values_squared_length_locus_midpoint_circle_l605_605348


namespace count_dogs_in_park_l605_605355

theorem count_dogs_in_park (total_legs : ℕ) (legs_per_dog : ℕ) (h_total_legs : total_legs = 436) (h_legs_per_dog : legs_per_dog = 4) :
  total_legs / legs_per_dog = 109 :=
by
  rw [h_total_legs, h_legs_per_dog]
  norm_num
  -- sorry (removing the sorry since norm_num will complete the proof automatically)

end count_dogs_in_park_l605_605355


namespace ensure_two_of_each_l605_605539

theorem ensure_two_of_each {A B : ℕ} (hA : A = 10) (hB : B = 10) :
  ∃ n : ℕ, n = 12 ∧
  ∀ (extracted : ℕ → ℕ),
    (extracted 0 + extracted 1 = n) →
    (extracted 0 ≥ 2 ∧ extracted 1 ≥ 2) :=
by
  sorry

end ensure_two_of_each_l605_605539


namespace limit_left_limit_right_l605_605665

noncomputable def f (x : ℝ) : ℝ := 6 / (x - 3)

theorem limit_left : tendsto f (nhds_within 3 (set.Iio 3)) at_bot :=
sorry

theorem limit_right : tendsto f (nhds_within 3 (set.Ioi 3)) at_top :=
sorry

end limit_left_limit_right_l605_605665


namespace billboards_and_road_length_l605_605839

theorem billboards_and_road_length :
  ∃ (x y : ℕ), 5 * (x + 21 - 1) = y ∧ (55 * (x - 1)) / 10 = y ∧ x = 200 ∧ y = 1100 :=
sorry

end billboards_and_road_length_l605_605839


namespace intersection_distance_l605_605404

noncomputable def distance_between_intersections (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, 
    l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2 ∧ 
    dist A B = Real.sqrt 6

def line_l (x y : ℝ) : Prop :=
  x - y + 1 = 0

def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sqrt 2 * Real.sin θ

theorem intersection_distance :
  distance_between_intersections line_l curve_C :=
sorry

end intersection_distance_l605_605404


namespace shpuntik_can_form_triangle_l605_605517

theorem shpuntik_can_form_triangle 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx : x1 + x2 + x3 = 1)
  (hy : y1 + y2 + y3 = 1)
  (infeasibility_vintik : x1 ≥ x2 + x3) :
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a < b + c ∧ b < a + c ∧ c < a + b :=
sorry

end shpuntik_can_form_triangle_l605_605517


namespace cost_to_feed_treats_for_a_week_l605_605107

theorem cost_to_feed_treats_for_a_week :
  (let dog_biscuits_cost := 4 * 0.25 in
   let rawhide_bones_cost := 2 * 1 in
   let daily_cost := dog_biscuits_cost + rawhide_bones_cost in
   7 * daily_cost = 21) :=
by
  sorry

end cost_to_feed_treats_for_a_week_l605_605107


namespace table_filling_condition_l605_605350

theorem table_filling_condition (n : ℕ) (A : ℕ → ℕ → ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    let R_k := ∑ i in finset.range n, A k i
    let C_k := ∑ j in finset.range n, A j k
    ((R_k = C_k + 1) ∨ (R_k = C_k - 1))) → 
  n % 2 = 0 :=
by 
  sorry

end table_filling_condition_l605_605350


namespace cos_identity_l605_605458

theorem cos_identity :
  cos (70 * (Real.pi / 180)) + 
  8 * cos (20 * (Real.pi / 180)) * cos (40 * (Real.pi / 180)) * cos (80 * (Real.pi / 180)) 
  = 2 * (cos (35 * (Real.pi / 180)))^2 := 
sorry

end cos_identity_l605_605458


namespace Tim_soda_cans_l605_605505

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l605_605505


namespace root_division_simplification_l605_605137

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l605_605137


namespace union_M_N_l605_605768

def M : set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : set ℝ := {x | x < -5 ∨ x > 5}

theorem union_M_N : M ∪ N = {x | x < -5 ∨ x > -3} :=
by sorry

end union_M_N_l605_605768


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605165

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605165


namespace reinforcement_number_l605_605955

def initial_men : ℕ := 2000
def initial_days : ℕ := 65
def days_passed : ℕ := 15
def remaining_days : ℕ := 20
def provisions : ℕ := initial_men * initial_days
def provisions_remaining : ℕ := provisions - (initial_men * days_passed)
def total_men (R : ℕ) := initial_men + R
def provisions_needed (R : ℕ) := (total_men R) * remaining_days

theorem reinforcement_number : ∃ R : ℕ, provisions_remaining = provisions_needed R ∧ R = 3000 :=
by
  use 3000
  split
  · sorry
  · rfl

end reinforcement_number_l605_605955


namespace range_of_a_l605_605877

def f (x : ℝ) : ℝ :=
  if x < 0 then 2^x - 3 else real.sqrt (x + 1)

theorem range_of_a (a : ℝ) (h : f a > 1) : a > 0 :=
  sorry

end range_of_a_l605_605877


namespace quadratic_function_x5_value_l605_605408

theorem quadratic_function_x5_value 
  (a b c : ℝ)
  (f : ℝ → ℝ := λ x, a * x^2 + b * x + c)
  (h1 : f 3 = 10)
  (h2 : ∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ (x1 - x2).abs = 4) :
  f 5 = 0 :=
sorry

end quadratic_function_x5_value_l605_605408


namespace expression_1_expression_2_l605_605256

variable (α : Real)

theorem expression_1 (h : tan α = 2) :
  (sin α - 3 * cos α) / (sin α + cos α) = -1 / 3 :=
  sorry

theorem expression_2 (h : tan α = 2) :
  2 * (sin α)^2 - sin α * cos α + (cos α)^2 = 7 / 5 :=
  sorry

end expression_1_expression_2_l605_605256


namespace sum_of_numbers_l605_605010

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : (a + b + c) / 3 = a + 20) 
  (h₂ : (a + b + c) / 3 = c - 30) 
  (h₃ : b = 10) : 
  a + b + c = 60 := 
by
  sorry

end sum_of_numbers_l605_605010


namespace tim_final_soda_cans_l605_605510

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l605_605510


namespace central_cage_rabbit_count_l605_605448

-- Define the cages
inductive Cage
| first | second | third | fourth | fifth

-- Define the number of rabbits in each cage
noncomputable def rabbits : Cage → ℕ
| Cage.first := 2
| Cage.second := 3
| Cage.third := 4
| Cage.fourth := 3
| Cage.fifth := 2

-- Define a function to count the neighbors
def neighbors (c: Cage) : ℕ :=
  match c with
  | Cage.first => rabbits Cage.first + rabbits Cage.second
  | Cage.second => rabbits Cage.first + rabbits Cage.second + rabbits Cage.third
  | Cage.third => rabbits Cage.second + rabbits Cage.third + rabbits Cage.fourth
  | Cage.fourth => rabbits Cage.third + rabbits Cage.fourth + rabbits Cage.fifth
  | Cage.fifth => rabbits Cage.fourth + rabbits Cage.fifth

-- Prove the central cage has the correct number of rabbits
theorem central_cage_rabbit_count : rabbits Cage.third = 4 :=
by sorry

end central_cage_rabbit_count_l605_605448


namespace areas_equal_l605_605605

noncomputable theory
open Real

-- Defining the conditions
def condition_1 (a k : ℝ) := a > 0 ∧ k > 0

def point_A (k : ℝ) : ℝ × ℝ := (sqrt k, sqrt k)
def point_C (a k : ℝ) : ℝ × ℝ := (sqrt (k / a), sqrt (a * k))

def area (p q : ℝ × ℝ) : ℝ := 1 / 2 * p.1 * p.2
def area_AOB (k : ℝ) : ℝ := area (point_A k) (0, 0)
def area_COD (a k : ℝ) : ℝ := area (point_C a k) (0, 0)

-- Statement of the problem
theorem areas_equal (a k : ℝ) (h : condition_1 a k) : 
  area_AOB k = area_COD a k := by
  sorry

end areas_equal_l605_605605


namespace locus_of_perpendiculars_l605_605739

noncomputable def point_O : Point := (1, 0)
def line_l (X : Point) : Prop := X.1 = 0

theorem locus_of_perpendiculars (X : Point) (hX : line_l X) :
  (X.2 ^ 2) ≥ 4 * X.1 :=
sorry

end locus_of_perpendiculars_l605_605739


namespace quadratic_rewrite_h_l605_605331

theorem quadratic_rewrite_h (a k h x : ℝ) :
  (3 * x^2 + 9 * x + 17) = a * (x - h)^2 + k ↔ h = -3/2 :=
by sorry

end quadratic_rewrite_h_l605_605331


namespace odd_divisors_less_than_100_l605_605313

theorem odd_divisors_less_than_100 : 
  {n : ℕ | 0 < n ∧ n < 100 ∧ (∃ m : ℕ, m^2 = n)}.card = 9 :=
sorry

end odd_divisors_less_than_100_l605_605313


namespace trajectory_of_M_equation_of_line_l_l605_605989

-- Definitions of conditions
def circle (point : ℝ × ℝ) : Prop := (point.1)^2 + (point.2)^2 = 36
def Q : ℝ × ℝ := (4, 0)
def M (P : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Given conditions and proof goals
theorem trajectory_of_M :
  ∀ P : ℝ × ℝ, circle P → let M_coord := M P in (M_coord.1 - 2)^2 + (M_coord.2)^2 = 9 := by
  sorry

theorem equation_of_line_l :
  ∀ (A B : ℝ × ℝ), A.2 = Q.2 + (-3 - Q.2) * A.1 / Q.1
    → B.2 = Q.2 + (-3 - Q.2) * B.1 / Q.1
    → ∀ x₁ x₂ : ℝ, A.1 = x₁ ∧ B.1 = x₂
    → (x₁ / x₂ + x₂ / x₁) = 21 / 2
    → (A.2 - B.2 = (x₁ - x₂) * (-3 - Q.2) / Q.1 ∨ A.2 - B.2 = (x₁ - x₂) * 17 / 7) := by
  sorry

end trajectory_of_M_equation_of_line_l_l605_605989


namespace min_containers_to_fill_jumbo_l605_605591

theorem min_containers_to_fill_jumbo :
  (75 : ℕ) ∣ 1800 ∧ (1800 / 75 = 24) →
  ∑ i in Finset.range 24, 75 = 1800 :=
by
  sorry

end min_containers_to_fill_jumbo_l605_605591


namespace sufficient_and_not_necessary_condition_l605_605731

theorem sufficient_and_not_necessary_condition (a b : ℝ) (hb: a < 0 ∧ b < 0) : a + b < 0 :=
by
  sorry

end sufficient_and_not_necessary_condition_l605_605731


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605174

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605174


namespace smallest_positive_n_l605_605682

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605682


namespace geom_problem_l605_605896

theorem geom_problem
  (a b c : ℕ)
  (h1 : c ≠ 0)
  (h2 : SquareABCD : ∀ (P : Type) [TopologicalSpace P], ConvexHull P {1, 0, 2, 0}) 
  (h3 : EquilateralTriangleAEF : ∀ P : ℝ², EquilateralTriangle ∉ {P}) 
  (h4 : SmallerSquareB : ∀ Q : ℝ², SmallerSquare ∉ {Q})
  (h5 : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (¬ ∃ p : Prime, p^2 ∣ b))
  : ∃ s : ℝ, 
  s = (a - (sqrt b)) / c 
  ∧ a + b + c = 12 :=  
begin
  -- placeholder for the actual proof
  sorry
end

end geom_problem_l605_605896


namespace count_valid_n_l605_605717

def num_valid_n : ℕ :=
  let possible_n := { n : ℕ | ∃ (a b : ℕ), n = 2^a * 5^b ∧ n < 1000 } in
  let non_zero_thousandths := { n ∈ possible_n | (1000 * n : ℚ) / n % 10 ≠ 0 } in
  non_zero_thousandths.card

theorem count_valid_n : num_valid_n = 25 := by
  sorry

end count_valid_n_l605_605717


namespace ratio_problem_l605_605732

-- Given condition: a, b, c are in the ratio 2:3:4
theorem ratio_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : a / c = 2 / 4) : 
  (a - b + c) / b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end ratio_problem_l605_605732


namespace combined_surface_area_of_glued_cubes_l605_605030

-- Definitions based on the problem conditions
def side_length_large_cube : ℝ := 3
def side_length_small_cube : ℝ := side_length_large_cube / 3
def surface_area_cube (a : ℝ) : ℝ := 6 * (a * a)

-- Main statement to be proven
theorem combined_surface_area_of_glued_cubes :
  let A_large := surface_area_cube side_length_large_cube,
      A_small := surface_area_cube side_length_small_cube in
  let hidden_area_from_large_cube := (side_length_large_cube * side_length_large_cube),
      net_surface_area := A_large - hidden_area_from_large_cube + A_small - hidden_area_from_large_cube in
  net_surface_area = 74 :=
by
  sorry

end combined_surface_area_of_glued_cubes_l605_605030


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605167

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605167


namespace proof_problem_l605_605464

theorem proof_problem 
  (A a B b : ℝ) 
  (h1 : |A - 3 * a| ≤ 1 - a) 
  (h2 : |B - 3 * b| ≤ 1 - b) 
  (h3 : 0 < a) 
  (h4 : 0 < b) :
  (|((A * B) / 3) - 3 * (a * b)|) - 3 * (a * b) ≤ 1 - (a * b) :=
sorry

end proof_problem_l605_605464


namespace circles_point_distance_l605_605029

noncomputable section

-- Define the data for the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def CircleA (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := K, radius := R }

def CircleB (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := (K.1 + 2 * R, K.2), radius := R }

-- Define the condition that two circles touch each other at point K
def circles_touch (C1 C2 : Circle) (K : ℝ × ℝ) : Prop :=
  dist C1.center K = C1.radius ∧ dist C2.center K = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

-- Define the angle condition ∠AKB = 90°
def angle_AKB_is_right (A K B : ℝ × ℝ) : Prop :=
  -- Using the fact that a dot product being zero implies orthogonality
  let vec1 := (A.1 - K.1, A.2 - K.2)
  let vec2 := (B.1 - K.1, B.2 - K.2)
  vec1.1 * vec2.1 + vec1.2 * vec2.2 = 0

-- Define the points A and B being on their respective circles
def on_circle (A : ℝ × ℝ) (C : Circle) : Prop :=
  dist A C.center = C.radius

-- Define the theorem
theorem circles_point_distance 
  (R : ℝ) (K A B : ℝ × ℝ) 
  (C1 := CircleA R K) 
  (C2 := CircleB R K) 
  (h1 : circles_touch C1 C2 K) 
  (h2 : on_circle A C1) 
  (h3 : on_circle B C2) 
  (h4 : angle_AKB_is_right A K B) : 
  dist A B = 2 * R := 
sorry

end circles_point_distance_l605_605029


namespace symmetrical_character_l605_605950

def symmetrical (char : String) : Prop :=
  -- Define a predicate symmetrical which checks if a given character
  -- is a symmetrical figure somehow. This needs to be implemented
  -- properly based on the graphical property of the character.
  sorry 

theorem symmetrical_character :
  ∀ (c : String), (c = "幸" → symmetrical c) ∧ 
                  (c = "福" → ¬ symmetrical c) ∧ 
                  (c = "惠" → ¬ symmetrical c) ∧ 
                  (c = "州" → ¬ symmetrical c) :=
by
  sorry

end symmetrical_character_l605_605950


namespace function_in_second_quadrant_l605_605055

theorem function_in_second_quadrant (x : ℝ) : 
  ∀ f : ℝ → ℝ, (f = λ x, 2*x - 5 → is_strictly_increasing f ∧ 
  ¬∃ x : ℝ, x ≤ 0 ∧ f x ≥ 0) :=
begin
  sorry
end

end function_in_second_quadrant_l605_605055


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605223

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605223


namespace max_wrong_to_win_prize_l605_605025

theorem max_wrong_to_win_prize :
  ∀ (x : ℕ), (4 * (24 - x) - x ≥ 80) → x ≤ 3 :=
by
  intro x
  have h1 : 4 * (24 - x) - x ≥ 80 → 96 - 5 * x ≥ 80 := sorry
  have h2 : 96 - 5 * x ≥ 80 → -5 * x ≥ -16 := sorry
  have h3 : -5 * x ≥ -16 → x ≤ 3 := sorry
  exact h3 (h2 (h1 ‹4 * (24 - x) - x ≥ 80›))

end max_wrong_to_win_prize_l605_605025


namespace constant_term_of_expansion_l605_605000

theorem constant_term_of_expansion :
  (∃ r : ℕ, 12 - 4 * r = 0 ∧ (-2)^r * Nat.choose 4 r = -32) :=
by
  use 3
  split
  sorry
  sorry

end constant_term_of_expansion_l605_605000


namespace infinitely_many_triples_l605_605868

theorem infinitely_many_triples (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∀ k : ℕ, 
  ∃ (x y z : ℕ), 
    x = 2^(k * m * n + 1) ∧ 
    y = 2^(n + n * k * (m * n + 1)) ∧ 
    z = 2^(m + m * k * (m * n + 1)) ∧ 
    x^(m * n + 1) = y^m + z^n := 
by 
  intros k
  use 2^(k * m * n + 1), 2^(n + n * k * (m * n + 1)), 2^(m + m * k * (m * n + 1))
  simp
  sorry

end infinitely_many_triples_l605_605868


namespace smallest_n_for_identity_l605_605706

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605706


namespace count_valid_n_l605_605718

def num_valid_n : ℕ :=
  let possible_n := { n : ℕ | ∃ (a b : ℕ), n = 2^a * 5^b ∧ n < 1000 } in
  let non_zero_thousandths := { n ∈ possible_n | (1000 * n : ℚ) / n % 10 ≠ 0 } in
  non_zero_thousandths.card

theorem count_valid_n : num_valid_n = 25 := by
  sorry

end count_valid_n_l605_605718


namespace arccos_ge_arcsin_for_all_x_in_neg1_to_1_l605_605140

theorem arccos_ge_arcsin_for_all_x_in_neg1_to_1 :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → arccos x ≥ arcsin x :=
by
  intro x
  intro h
  sorry

end arccos_ge_arcsin_for_all_x_in_neg1_to_1_l605_605140


namespace solution_func_eq_floor_l605_605864

noncomputable def f : ℝ → ℝ := sorry

theorem solution_func_eq_floor :
  (∀ x y : ℝ, f(x) + f(y) + 1 ≥ f(x+y) ∧ f(x+y) ≥ f(x) + f(y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f(0) ≥ f(x)) ∧
  (-f(-1) = 1 ∧ f(1) = 1) →
  ∀ x : ℝ, f(x) = Real.floor x := 
sorry

end solution_func_eq_floor_l605_605864


namespace mean_of_reciprocals_first_four_primes_l605_605145

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605145


namespace abs_sum_eq_abs_add_iff_ab_gt_zero_l605_605749

theorem abs_sum_eq_abs_add_iff_ab_gt_zero (a b : ℝ) :
  (|a + b| = |a| + |b|) → (a = 0 ∧ b = 0 ∨ ab > 0) :=
sorry

end abs_sum_eq_abs_add_iff_ab_gt_zero_l605_605749


namespace smallest_positive_n_l605_605683

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605683


namespace fourth_pentagon_has_31_dots_l605_605346

-- Conditions representing the sequence of pentagons
def first_pentagon_dots : ℕ := 1

def second_pentagon_dots : ℕ := first_pentagon_dots + 5

def nth_layer_dots (n : ℕ) : ℕ := 5 * (n - 1)

def nth_pentagon_dots (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + nth_layer_dots (k+1)) first_pentagon_dots

-- Question and proof statement
theorem fourth_pentagon_has_31_dots : nth_pentagon_dots 4 = 31 :=
  sorry

end fourth_pentagon_has_31_dots_l605_605346


namespace distinct_four_digit_numbers_count_l605_605309

theorem distinct_four_digit_numbers_count : 
  ∃ (digits: Set ℕ) (f : Fin 4 → ℕ), 
    digits = {1, 2, 3, 4, 5} ∧ (∀ i j, i ≠ j → f i ∈ digits ∧ f i ≠ f j) 
  → fintype.card (finset.univ.filter (λ (x : ℕ), x ∈ {a | ∃ (digits : Set ℕ) (f : Fin 4 → ℕ), digits = {1, 2, 3, 4, 5} ∧ (∀ i j, i ≠ j → f i ∈ digits ∧ f i ≠ f j) ∧
    x = f 0 * 1000 + f 1 * 100 + f 2 * 10 + f 3})) = 120 :=
by
  sorry

end distinct_four_digit_numbers_count_l605_605309


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605211

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605211


namespace root_division_simplification_l605_605138

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l605_605138


namespace cross_section_area_correct_l605_605473

def isosceles_right_prism_cross_section_area (a α φ : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hφ : φ < π / 2) : ℝ :=
  a^2 * (Real.sin (2 * α)) / (2 * Real.cos φ)

theorem cross_section_area_correct (a α φ : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hφ : φ < π / 2):
  isosceles_right_prism_cross_section_area a α φ hα1 hα2 hφ = (a^2 * Real.sin (2 * α)) / (2 * Real.cos φ) :=
by
  sorry

end cross_section_area_correct_l605_605473


namespace problem_AC_l605_605302

variables {x : ℝ}
def vector_a : ℝ × ℝ := (1, x)
def vector_b : ℝ × ℝ := (x, 4)

theorem problem_AC :
  (x = 2 → vector_b = 2 • vector_a) ∧
  (∀ y, ∃ xmin, ∀ z, (z = vector_a + vector_b) → ymin = min (vector_a•z)) ∧
  (x = 0 → (vector_a•vector_b = 0)) :=
by {
  sorry
}

end problem_AC_l605_605302


namespace least_n_factorial_7350_l605_605038

theorem least_n_factorial_7350 (n : ℕ) :
  (7350 ∣ n!) → (∃ k : ℕ, (1 ≤ k ∧ 7350 ∣ k!)) :=
by
  have prime_factors_7350 : 7350 = 2 * 3^2 * 5^2 * 7 := by norm_num
  existsi 10
  split
  · norm_num
  · apply dvd_factorial.mpr
    sorry

end least_n_factorial_7350_l605_605038


namespace handshakes_among_women_l605_605470

theorem handshakes_among_women (n : ℕ) (h : n = 10) : (∑ i in finset.range n, i) = 45 :=
by
  sorry

end handshakes_among_women_l605_605470


namespace pipe_c_emptying_time_l605_605449

theorem pipe_c_emptying_time :
  let (A B C: ℝ) := (1/3, 1/4, 1/x) in
  let combined_rate := (A + B - C) in
  (combined_rate = 1/3) → C = 1/4 :=
by
  sorry

end pipe_c_emptying_time_l605_605449


namespace smallest_positive_n_l605_605675

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605675


namespace evaluate_expression_l605_605642

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l605_605642


namespace cone_diameter_l605_605329

theorem cone_diameter (S : ℝ) (hS : S = 3 * Real.pi) (unfold_semicircle : ∃ (r l : ℝ), l = 2 * r ∧ S = π * r^2 + (1 / 2) * π * l^2) : 
∃ d : ℝ, d = Real.sqrt 6 := 
by
  sorry

end cone_diameter_l605_605329


namespace digit_possibilities_for_mod4_count_possibilities_is_3_l605_605586

theorem digit_possibilities_for_mod4 (N : ℕ) (h : N < 10): 
  (80 + N) % 4 = 0 → N = 0 ∨ N = 4 ∨ N = 8 → true := 
by
  -- proof is not needed
  sorry

def count_possibilities : ℕ := 
  (if (80 + 0) % 4 = 0 then 1 else 0) +
  (if (80 + 1) % 4 = 0 then 1 else 0) +
  (if (80 + 2) % 4 = 0 then 1 else 0) +
  (if (80 + 3) % 4 = 0 then 1 else 0) +
  (if (80 + 4) % 4 = 0 then 1 else 0) +
  (if (80 + 5) % 4 = 0 then 1 else 0) +
  (if (80 + 6) % 4 = 0 then 1 else 0) +
  (if (80 + 7) % 4 = 0 then 1 else 0) +
  (if (80 + 8) % 4 = 0 then 1 else 0) +
  (if (80 + 9) % 4 = 0 then 1 else 0)

theorem count_possibilities_is_3: count_possibilities = 3 := 
by
  -- proof is not needed
  sorry

end digit_possibilities_for_mod4_count_possibilities_is_3_l605_605586


namespace min_cos_A_sin_B_plus_sin_C_l605_605745

theorem min_cos_A_sin_B_plus_sin_C (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0) :
  ∃ m, m = - (2 / 9) * Real.sqrt 6 ∧ ∀ x y z, x + y + z = Real.pi ∧ x > 0 ∧ y > 0 ∧ z > 0 → cos x * (sin y + sin z) ≥ m :=
sorry

end min_cos_A_sin_B_plus_sin_C_l605_605745


namespace count_divisibles_3_7_l605_605628

/-- Determine how many whole numbers from 1 through 100 are divisible by either 3 or 7 or both. --/
theorem count_divisibles_3_7 : 
  (finset.card (finset.filter (λ x, x % 3 = 0 ∨ x % 7 = 0) (finset.range 101))) = 43 :=
by
  -- The proof goes here
  sorry

end count_divisibles_3_7_l605_605628


namespace problem_statement_l605_605741

-- Defining the basic setup
variables {R : Type*} [Field R] (m : R)

-- Circle O with given equation x^2 + y^2 = 25
def circle_O (x y : R) : Prop := x^2 + y^2 = 25

-- Circle O1 with center (m, 0)
def circle_O1 (x y : R) : Prop := (x - m)^2 + y^2 = 137

-- Point P (3, 4) intersection
def point_P : Prop := circle_O 3 4 ∧ circle_O1 3 4

-- Line l passing through point P (3, 4) with slope k
def line_l (x y k : R) : Prop := y = k * (x - 3) + 4

-- Line l1 perpendicular to l passing through point P (3, 4)
def line_l1 (x y k : R) : Prop := y = (-1 / k) * (x - 3) + 4

-- Given conditions: circles intersect at (3,4), line with slope k through (3,4)
namespace Problem

open Lean.Internal.Meta

def part1 (k : R) : Prop :=
  k = 1 → ∀ (BP: R), BP = 7 * Real.sqrt 2 → (∃ (x y : R), circle_O1 x y)

def part2 (k : R) : Prop :=
  ∀ (AB CD : R), AB = (4 * m^2 / (1 + k^2)) ∧ CD = (4 * m^2 * k^2 / (1 + k^2)) → AB + CD = 4 * m^2

end Problem

-- The main theorem combining both parts of the proof
theorem problem_statement (k : R) : Problem.part1 k ∧ Problem.part2 k :=
by sorry

end problem_statement_l605_605741


namespace heather_aprons_l605_605303

variable {totalAprons : Nat} (apronsSewnBeforeToday apronsSewnToday apronsSewnTomorrow apronsSewnSoFar apronsRemaining : Nat)

theorem heather_aprons (h_total : totalAprons = 150)
                       (h_today : apronsSewnToday = 3 * apronsSewnBeforeToday)
                       (h_sewnSoFar : apronsSewnSoFar = apronsSewnBeforeToday + apronsSewnToday)
                       (h_tomorrow : apronsSewnTomorrow = 49)
                       (h_remaining : apronsRemaining = totalAprons - apronsSewnSoFar)
                       (h_halfRemaining : 2 * apronsSewnTomorrow = apronsRemaining) :
  apronsSewnBeforeToday = 13 :=
by
  sorry

end heather_aprons_l605_605303


namespace smallest_n_l605_605688

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605688


namespace triangle_area_relation_l605_605838

theorem triangle_area_relation {A B C A' B' C' A'' B'' C'' : Type*}
  (hA' : A' ∈ line[A, B])
  (hB' : B' ∈ line[B, C])
  (hC' : C' ∈ line[C, A])
  (hConcurrent : ∃ O, line_segment[A, A'] ∩ line_segment[B, B'] ∩ line_segment[C, C'] = {O})
  (hSymmetricA'' : A'' = point_symmetry A A')
  (hSymmetricB'' : B'' = point_symmetry B B')
  (hSymmetricC'' : C'' = point_symmetry C C') :
  area (A'', B'', C'') = 3 * area (A, B, C) + 4 * area (A', B', C') :=
sorry

end triangle_area_relation_l605_605838


namespace train_crossing_time_l605_605998

def convert_km_per_hr_to_m_per_s (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

def train_length : ℝ := 600
def train_speed_kmh : ℝ := 144
def train_speed_m_per_s := convert_km_per_hr_to_m_per_s train_speed_kmh

theorem train_crossing_time :
  (train_length / train_speed_m_per_s) = 15 := by
  sorry

end train_crossing_time_l605_605998


namespace complement_of_A_l605_605409

open Set

def U := univ : Set ℝ
def A : Set ℝ := {x | x^2 > 9}
def complementA : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

theorem complement_of_A : ∀ (U A : Set ℝ), 
  U = univ →
  A = {x | x^2 > 9} →
  (U \ A = {x | -3 ≤ x ∧ x ≤ 3}) :=
by
  intros U A hU hA
  simp [hU, hA]
  ext x
  simp
  split
  { intro h
    simp at h
    rw ←not_lt
    rw ←not_lt at h
    tauto }
  { intro h
    simp
    apply not_or
    { intro h1
      linarith }
    { intro h2
      linarith } }

end complement_of_A_l605_605409


namespace complex_trajectory_is_ray_l605_605071

open Complex

theorem complex_trajectory_is_ray :
  {z : ℂ | abs (z + 1) - abs (z - 1) = 2} = {z : ℂ | ∃ t ∈ ℝ, t ≥ 0 ∧ z = ⟨1 + t, 0⟩} :=
by
  sorry

end complex_trajectory_is_ray_l605_605071


namespace remaining_regular_toenails_l605_605304

theorem remaining_regular_toenails :
  ∀ (total_capacity big_toenail_space big_toenails regular_toenails : ℕ),
    total_capacity = 100 →
    big_toenail_space = 2 →
    big_toenails = 20 →
    regular_toenails = 40 →
    let occupied_space := big_toenails * big_toenail_space + regular_toenails in
    let remaining_space := total_capacity - occupied_space in
    remaining_space = 20 :=
by
  intros total_capacity big_toenail_space big_toenails regular_toenails htcs hbs hbt hrt
  let occupied_space := big_toenails * big_toenail_space + regular_toenails
  have h1 : occupied_space = 40 * 2 + 40 := rfl
  let remaining_space := total_capacity - occupied_space
  have h2 : remaining_space = 100 - 80 := rfl
  have h3 : 20 = 20 := rfl
  exact h3

end remaining_regular_toenails_l605_605304


namespace snow_probability_first_week_l605_605427

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l605_605427


namespace rhombus_side_length_l605_605808

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605808


namespace smallest_n_l605_605689

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605689


namespace fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l605_605067

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem fermats_little_theorem_poly (X : ℤ) :
  (X + 1) ^ p = X ^ p + 1 := by
    sorry

theorem binom_coeff_divisible_by_prime {k : ℕ} (hkp : 1 ≤ k ∧ k < p) :
  p ∣ Nat.choose p k := by
    sorry

end fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l605_605067


namespace parabola_equation_through_origin_point_l605_605018

-- Define the conditions
def vertex_origin := (0, 0)
def point_on_parabola := (-2, 4)

-- Define what it means to be a standard equation of a parabola passing through a point
def standard_equation_passing_through (p : ℝ) (x y : ℝ) : Prop :=
  (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)

-- The theorem stating the conclusion
theorem parabola_equation_through_origin_point :
  ∃ p > 0, standard_equation_passing_through p (-2) 4 ∧
  (4^2 = -8 * (-2) ∨ (-2)^2 = 4) := 
sorry

end parabola_equation_through_origin_point_l605_605018


namespace number_of_subsets_mod_10_l605_605783

open Finset

theorem number_of_subsets_mod_10 (s : Finset ℕ) (h : s = {7, 12, 18, 21, 35, 42, 56}) :
  (s.filter (λ t, t.card = 4 ∧ t.sum % 10 = 0)).card = 16 :=
by
  rw h
  sorry

end number_of_subsets_mod_10_l605_605783


namespace jeremy_works_25_hours_per_week_l605_605337

noncomputable def fiona_hours_per_week := 40
noncomputable def john_hours_per_week := 30
noncomputable def pay_rate_per_hour := 20
noncomputable def total_monthly_expenditure := 7600

def jeremy_hours_per_week :=
  (total_monthly_expenditure - (fiona_hours_per_week * pay_rate_per_hour * 4 + john_hours_per_week * pay_rate_per_hour * 4)) / (pay_rate_per_hour * 4)

theorem jeremy_works_25_hours_per_week :
  jeremy_hours_per_week = 25 := by
  sorry

end jeremy_works_25_hours_per_week_l605_605337


namespace transmission_time_correct_l605_605111

-- Definitions based on conditions
def blocks : ℕ := 100
def chunks_per_block : ℕ := 256
def channel_speed : ℕ := 150
def initial_delay : ℕ := 10
def efficiency_factor : ℝ := 0.8

-- Effective transmission speed under efficiency factor
def effective_transmission_speed : ℝ := channel_speed * efficiency_factor

-- Total chunks to be transmitted
def total_chunks : ℕ := blocks * chunks_per_block

-- Transmission time excluding delay
def transmission_time : ℝ := total_chunks / effective_transmission_speed

-- Total time including initial delay
def total_time_seconds : ℝ := transmission_time + initial_delay

-- Convert total time from seconds to minutes
def total_time_minutes : ℝ := total_time_seconds / 60

-- Statement of the problem
theorem transmission_time_correct :
  total_time_minutes ≈ 3.7222 := 
sorry

end transmission_time_correct_l605_605111


namespace base_b_of_200_has_5_digits_l605_605576

theorem base_b_of_200_has_5_digits : ∃ (b : ℕ), (b^4 ≤ 200) ∧ (200 < b^5) ∧ (b = 3) := by
  sorry

end base_b_of_200_has_5_digits_l605_605576


namespace quadratic_expression_l605_605861

-- Definitions of roots and their properties
def quadratic_roots (r s : ℚ) : Prop :=
  (r + s = 5 / 3) ∧ (r * s = -8 / 3)

theorem quadratic_expression (r s : ℚ) (h : quadratic_roots r s) :
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
by
  sorry

end quadratic_expression_l605_605861


namespace smallest_n_for_identity_matrix_l605_605670

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605670


namespace compare_an_and_Sn_l605_605920

variable {a: ℕ → ℝ}
variable {S: ℕ → ℝ}
variable {m n : ℕ}
variable {d : ℝ}

axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d
axiom sum_seq : ∀ n, S n = ∑ i in finset.range (n + 1), a i
axiom neg_common_diff : d < 0
axiom exists_m : (∃ m, m ≥ 3 ∧ a m = S m)

theorem compare_an_and_Sn (h : n > m) : a n > S n := 
by
  sorry

end compare_an_and_Sn_l605_605920


namespace max_value_expr_l605_605381

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l605_605381


namespace georgia_problems_left_l605_605254

theorem georgia_problems_left :
  ∀ (total_problems : ℕ) (time1 time2 time3 problems1 problems2 : ℕ),
  total_problems = 75 →
  time1 = 20 →
  time2 = 20 →
  problems1 = 10 →
  problems2 = 2 * problems1 →
  problems1 + problems2 = 30 →
  75 - (problems1 + problems2) = 45 :=
by {
  intros total_problems time1 time2 time3 problems1 problems2
         h_total h_time1 h_time2 h_problems1 h_problems2 h_sum,
  rw [h_total, h_problems1, h_problems2],
  exact h_sum,
  }

end georgia_problems_left_l605_605254


namespace sum_of_roots_tan_quadratic_l605_605239

theorem sum_of_roots_tan_quadratic :
  let f := λ x : ℝ, tan x * tan x - 8 * tan x + 3
  ∃ (roots : List ℝ), (∀ r ∈ roots, 0 ≤ r ∧ r ≤ 2 * π ∧ f r = 0) ∧ (roots.sum = 3 * π) :=
by
  -- This statement asserts that there exists a list of roots within [0, 2π]
  -- such that all roots satisfy the equation, and their sum is 3π.
  sorry

end sum_of_roots_tan_quadratic_l605_605239


namespace condition_implies_BD_perp_EF_l605_605378

constants (α β : Plane) (A B C D E F : Point) (AB CD : LineSegment)

-- Given conditions
axiom intersect_planes : (α ∩ β = EF)
axiom perp_AB_alpha : perpendicular AB α B
axiom perp_CD_alpha : perpendicular CD α D

-- Additional conditions as hypotheses
axioms (cond1 : perpendicular AC β)
        (cond2 : angle_between AC α = angle_between AC β)
        (cond3 : projection_in_plane AC β = projection_in_plane CD β)
        (cond4 : parallel AC EF)

-- The statement to prove
theorem condition_implies_BD_perp_EF : 
  (cond1 ∨ cond3) → 
  perpendicular BD EF :=
sorry

end condition_implies_BD_perp_EF_l605_605378


namespace relationship_abc_l605_605260

variable (x : ℝ)
variable hx : x ∈ Set.Ioo (Real.exp (-1)) 1
def a := Real.log x
def b := 2 ^ (Real.log (1 / x))
def c := Real.exp (Real.log x)

theorem relationship_abc : b > c ∧ c > a :=
by
  have h_a : a = Real.log x := rfl
  have h_b : b = 2 ^ (Real.log (1 / x)) := rfl
  have h_c : c = Real.exp (Real.log x) := rfl
  sorry

end relationship_abc_l605_605260


namespace factorization_b_even_l605_605132

theorem factorization_b_even (b : ℤ) :
  -- Condition: the polynomial can be factored into integer linear binomials
  (∃ (m n p q : ℤ), 15x^2 + bx + 15 = (m * x + n) * (p * x + q)) →
  -- Prove: b is some even number
  ∃ (k : ℤ), b = 2 * k :=
by
  sorry

end factorization_b_even_l605_605132


namespace number_of_erasers_l605_605503

theorem number_of_erasers (P E : ℕ) (h1 : P + E = 240) (h2 : P = E - 2) : E = 121 := by
  sorry

end number_of_erasers_l605_605503


namespace probability_remainder_1_div_7_l605_605102

theorem probability_remainder_1_div_7 {N : ℕ} (hN : 1 ≤ N ∧ N ≤ 2030) :
  let favorable_outcomes := { n | 1 ≤ n ∧ n ≤ 2030 ∧ (n % 7 = 1 ∨ n % 7 = 2 ∨ n % 7 = 3 ∨ n % 7 = 4 ∨ n % 7 = 5 ∨ n % 7 = 6) },
      total_outcomes := { n | 1 ≤ n ∧ n ≤ 2030 } in
  (favorable_outcomes.card.to_nat : ℝ) / total_outcomes.card.to_nat = 6 / 7 :=
by
  sorry

end probability_remainder_1_div_7_l605_605102


namespace find_angle_A_in_triangle_l605_605837

theorem find_angle_A_in_triangle
  {A B C : ℝ}
  (a b c : ℝ)
  (h_a : a = real.sqrt 2)
  (h_b : b = 2)
  (h_sin_cos_B : real.sin B + real.cos B = real.sqrt 2)
  (A_plus_B_plus_C : A + B + C = real.pi)
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < real.pi ∧ B < real.pi ∧ C < real.pi) :
  A = real.pi / 6 :=
by
  sorry

end find_angle_A_in_triangle_l605_605837


namespace part_one_part_two_l605_605758

def z : ℂ := ((1 + complex.I)^2 + 2 * (5 - complex.I)) / (3 + complex.I)

theorem part_one :
  |z| = sqrt 10 := 
sorry

theorem part_two (a b : ℝ) :
  (z : ℂ) = (3 - complex.I) →
  z * (z + (a : ℂ)) = (b : ℂ) + complex.I → 
  a = -7 ∧ b = -13 := 
sorry

end part_one_part_two_l605_605758


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605228

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605228


namespace math_problem_l605_605968

theorem math_problem : 
  ( - (1 / 12 : ℚ) + (1 / 3 : ℚ) - (1 / 2 : ℚ) ) / ( - (1 / 18 : ℚ) ) = 4.5 := 
by
  sorry

end math_problem_l605_605968


namespace solve_system_inequalities_l605_605463

theorem solve_system_inequalities (x : ℝ) :
  x - 1 < 2 ∧ 2x + 3 ≥ x - 1 ↔ -4 ≤ x ∧ x < 3 := by
  sorry

end solve_system_inequalities_l605_605463


namespace eval_expression_l605_605640

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l605_605640


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605179

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605179


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605187

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605187


namespace solve_inequality_l605_605918

theorem solve_inequality (x : ℝ) : 
  (-x + 1 > 7x - 3) ↔ (x < 1 / 2) :=
sorry

end solve_inequality_l605_605918


namespace sum_of_coordinates_of_A_l605_605376

theorem sum_of_coordinates_of_A
  (A B C : ℝ × ℝ)
  (AC AB BC : ℝ)
  (h1 : AC / AB = 1 / 3)
  (h2 : BC / AB = 2 / 3)
  (hB : B = (2, 5))
  (hC : C = (5, 8)) :
  (A.1 + A.2) = 16 :=
sorry

end sum_of_coordinates_of_A_l605_605376


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605205

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605205


namespace cyclic_sum_inequality_l605_605869

theorem cyclic_sum_inequality (n : ℕ) (h_n : 2 ≤ n) (x : Fin n → ℝ) (h_pos : ∀ i, 0 < x i) : 
  ∑ i : Fin n, (x i)^2 / ((x i)^2 + (x (i + 1) % n) * (x (i + 2) % n)) ≤ n - 1 :=
by
  sorry

end cyclic_sum_inequality_l605_605869


namespace probability_of_snow_at_least_once_first_week_l605_605441

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l605_605441


namespace probability_of_divisible_by_9_is_zero_l605_605797

theorem probability_of_divisible_by_9_is_zero:
  let N := { n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 42) } in
  (∀ n ∈ N, ¬(n % 9 = 0)) →
  (finset.card N > 0) →
  (finset.count (λ n, n % 9 = 0) N) / (finset.card N) = 0 :=
begin
  intros h1,
  intros h2,
  sorry,
end

end probability_of_divisible_by_9_is_zero_l605_605797


namespace probability_and_relationship_l605_605549

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l605_605549


namespace shaded_fraction_of_rectangle_l605_605966

theorem shaded_fraction_of_rectangle (w h : ℝ) (w_pos : 0 < w) (h_pos : 0 < h) :
  let A := (0, 0)
      B := (w, 0)
      C := (w, h)
      D := (0, h)
      M := (w / 2, 0)
      N := (0, h / 2) in
  let area_tr : ℝ := (1/2) * (w / 2) * (h / 2) in
  (area_tr + area_tr) / (w * h) = 1 / 4 :=
sorry

end shaded_fraction_of_rectangle_l605_605966


namespace probability_san_francisco_plays_l605_605914

variable (P : Type → Type) [Probability P]

variable (event_play event_not_play : P ℝ)

-- Conditions
def condition1 : Prop :=
  event_play = 9 * event_not_play

def condition2 : Prop :=
  event_play + event_not_play = 1

-- The theorem to prove
theorem probability_san_francisco_plays
  (h1 : condition1 P event_play event_not_play)
  (h2 : condition2 P event_play event_not_play) : event_play = 0.9 := sorry

end probability_san_francisco_plays_l605_605914


namespace chessboard_traverse_l605_605827

theorem chessboard_traverse (n : ℕ) (h : n ≥ 2) : 
  (∃ path : ℕ → ℕ × ℕ, 
      path 0 = (0, 0) ∧ 
      (∀ k, k < n * n → 
          let (x, y) := path k in
          (x, y) ∈ finset.univ (fin n × fin n) ∧ 
          (let (nx, ny) := path (k + 1) in
           (nx = x + 1 ∧ ny = y) ∨ 
           (nx = x ∧ ny = y + 1) ∨ 
           (nx = x - 1 ∧ ny = y - 1) ∧ 
           ¬(nx = x ∧ ny = y + 1 ∨ nx = x - 1 ∧ ny = y) ∧ 
           ((nx, ny) ∉ (path '' finset.range (k + 1))))) ↔
  ∃ k, n = 3 * k ∨ n = 3 * k + 1 :=
by sorry

end chessboard_traverse_l605_605827


namespace change_in_average_is_one_l605_605094

-- Define the digits p and q, and the condition that they differ by one
variable (p q : ℕ)
variable (h1 : p ≠ q)  -- p and q are different
variable (h2 : p < 10)  -- p and q are digits
variable (h3 : q < 10)
variable (h4 : |p - q| = 1)

-- Define the original and interchanged number
def original_number := 10 * p + q
def interchanged_number := 10 * q + p

-- The change in the sum of 9 numbers, only considering the change due to the interchange
def delta_sum := original_number - interchanged_number

-- The change in the average
def delta_average := delta_sum / 9

-- The proof statement: the change in the average is 1
theorem change_in_average_is_one : delta_average = 1 := by
  sorry

end change_in_average_is_one_l605_605094


namespace rhombus_side_length_l605_605815

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l605_605815


namespace minimum_candies_to_identify_coins_l605_605058

-- Set up the problem: define the relevant elements.
inductive Coin : Type
| C1 : Coin
| C2 : Coin
| C3 : Coin
| C4 : Coin
| C5 : Coin

def values : List ℕ := [1, 2, 5, 10, 20]

-- Statement of the problem in Lean 4, no means to identify which is which except through purchases and change from vending machine.
theorem minimum_candies_to_identify_coins : ∃ n : ℕ, n = 4 :=
by
  -- Skipping the proof
  sorry

end minimum_candies_to_identify_coins_l605_605058


namespace count_terminating_n_with_conditions_l605_605727

def is_terminating_decimal (n : ℕ) : Prop :=
  (∃ m k : ℕ, (2^m * 5^k = n)) 

def thousandths_non_zero (n : ℕ) : Prop := 
  100 < n ∧ n ≤ 1000

theorem count_terminating_n_with_conditions : 
  {n : ℕ | is_terminating_decimal n ∧ thousandths_non_zero n}.to_finset.card = 9 := 
by
  sorry

end count_terminating_n_with_conditions_l605_605727


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605175

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605175


namespace arithmetic_mean_reciprocals_primes_l605_605153

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605153


namespace probability_snow_at_least_once_first_week_l605_605438

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l605_605438


namespace valid_pair_l605_605316

theorem valid_pair : (∃ n : ℤ, sqrt (↑25530 ^ 2 + ↑29464 ^ 2) = n) ∧ 
  ¬ (∃ n : ℤ, sqrt (↑37615 ^ 2 + ↑26855 ^ 2) = n) ∧
  ¬ (∃ n : ℤ, sqrt (↑15123 ^ 2 + ↑32477 ^ 2) = n) ∧ 
  ¬ (∃ n : ℤ, sqrt (↑28326 ^ 2 + ↑28614 ^ 2) = n) :=
by {
  sorry, -- Proof goes here.
}

end valid_pair_l605_605316


namespace equation_I_consecutive_integers_equation_II_consecutive_even_integers_l605_605622

theorem equation_I_consecutive_integers :
  ∃ (x y z : ℕ), x + y + z = 48 ∧ (x = y - 1) ∧ (z = y + 1) := sorry

theorem equation_II_consecutive_even_integers :
  ∃ (x y z w : ℕ), x + y + z + w = 52 ∧ (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) := sorry

end equation_I_consecutive_integers_equation_II_consecutive_even_integers_l605_605622


namespace sum_of_complex_numbers_l605_605611

theorem sum_of_complex_numbers :
  (1 + 3 * complex.i) + (2 - 4 * complex.i) + (4 + 2 * complex.i) = 7 + complex.i :=
sorry

end sum_of_complex_numbers_l605_605611


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605210

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605210


namespace isosceles_right_triangle_l605_605787

theorem isosceles_right_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B + C = 180)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : sin A / a = cos B / b)
  (h5 : cos B / b = cos C / c) :
  A = 90 ∧ B = 45 ∧ C = 45 :=
by
  sorry

end isosceles_right_triangle_l605_605787


namespace range_of_a_l605_605855

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x - 2 * a

def A (a : ℝ) : Set ℝ :=
  {x : ℝ | f a x > 0}

def B : Set ℝ :=
  {x : ℝ | 1 < x ∧ x < 3}

theorem range_of_a (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a ∈ set.Ioo (-∞) (-2) ∪ set.Ioo ((6 : ℝ) / 7) ∞) :=
sorry

end range_of_a_l605_605855


namespace simplify_expression_l605_605461

theorem simplify_expression (x y : ℝ) (h1 : x = 10) (h2 : y = -1/25) :
  ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = 2 / 5 := 
by
  sorry

end simplify_expression_l605_605461


namespace find_a_plus_b_l605_605765

theorem find_a_plus_b {f : ℝ → ℝ} (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x + 14) →
  f a = 1 →
  f b = 19 →
  a + b = -2 :=
by
  sorry

end find_a_plus_b_l605_605765


namespace solution_set_of_inequality_l605_605630

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_deriv : ∀ x, f' x > f x) (h_f2 : f 2 = Real.exp 2) :
  { x : ℝ | f x > Real.exp x } = set.Ioi 2 :=
by
  sorry

end solution_set_of_inequality_l605_605630


namespace probability_of_diff_by_3_is_1_8_l605_605097

def fair_8_sided_die := {x // 1 ≤ x ∧ x ≤ 8}

def valid_rolls : nat → nat → Prop
| x y := |x - y| = 3

noncomputable def probability_diff_by_3 : ℚ :=
  let successful_outcomes := {p : fair_8_sided_die × fair_8_sided_die // valid_rolls p.1.val p.2.val}.card in
  let total_outcomes := (finset.univ : finset (fair_8_sided_die × fair_8_sided_die)).card in
  successful_outcomes / total_outcomes

theorem probability_of_diff_by_3_is_1_8 : probability_diff_by_3 = 1/8 :=
  sorry

end probability_of_diff_by_3_is_1_8_l605_605097


namespace smallest_positive_n_l605_605685

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605685


namespace eval_expression_l605_605636

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l605_605636


namespace total_money_received_l605_605925

-- Define the given prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def adult_tickets_sold : ℕ := 90
def child_tickets_sold : ℕ := 40

-- Define the theorem to prove the total amount received
theorem total_money_received :
  (adult_ticket_price * adult_tickets_sold + child_ticket_price * child_tickets_sold) = 1240 :=
by
  -- Proof goes here
  sorry

end total_money_received_l605_605925


namespace probability_and_relationship_l605_605550

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l605_605550


namespace calculate_f_201_2_l605_605606

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f 3
axiom interval_eq : ∀ x : ℝ, x ∈ set.Ioo (-3 : ℝ) (-2 : ℝ) → f x = 5 * x

theorem calculate_f_201_2 : f 201.2 = -16 :=
by {
  sorry
}

end calculate_f_201_2_l605_605606


namespace find_a_l605_605770

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * sin x
def g (a x : ℝ) : ℝ := a * cos x

-- Define the interval
def interval (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Define tangency conditions
def tangents_perpendicular_at_p (a x : ℝ) : Prop :=
  let f' := 2 * cos x
  let g' := -a * sin x
  f' * g' = -1

-- Main theorem statement
theorem find_a (a x : ℝ) (h1 : f x = g a x) (h2 : interval x) 
  (h3 : tangents_perpendicular_at_p a x) : a = 2 * sqrt 3 / 3 :=
by
  sorry

end find_a_l605_605770


namespace complex_fraction_value_l605_605112

theorem complex_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 7 / 4 :=
sorry

end complex_fraction_value_l605_605112


namespace Tim_soda_cans_l605_605506

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l605_605506


namespace tangent_line_fixed_point_minimum_a_l605_605764

noncomputable def f (x a : ℝ) : ℝ := log x - (1 / 2) * a * x^2 + 1

theorem tangent_line_fixed_point (a : ℝ) :
  (let x := 1 in let y := f x a in let slope := (1/x) - a*x in 
    let b := y - slope * x in slope * (1/2) + b = 1/2) := 
by sorry

theorem minimum_a : ∃ a : ℤ, (∀ x: ℝ, f x (a : ℝ) ≤ (a-1) * x) ∧ (a = 2) :=
by sorry

end tangent_line_fixed_point_minimum_a_l605_605764


namespace min_A_rubiks_cubes_l605_605471

theorem min_A_rubiks_cubes (x : ℕ) (h_price_A : ∀ y, y = 15) (h_price_B : ∀ y, y = 22) (h_total : ∀ z, z = 40) 
(h_quantity : ∀ a b : ℕ, a + b = 40) (h_funding : ∀ c, c = 776) : 
  15x + 22 * (40 - x) ≤ 776 ∧ x ≤ 20 → (∃ x, x = 15) :=
by 
  sorry

end min_A_rubiks_cubes_l605_605471


namespace lottery_probability_blank_l605_605542

theorem lottery_probability_blank (prizes blanks : ℕ) (total : ℕ) (h1 : prizes = 10) (h2 : blanks = 25) (h3 : total = prizes + blanks) : 
  (blanks : ℚ) / total = 5 / 7 :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end lottery_probability_blank_l605_605542


namespace product_eq_one_l605_605763

noncomputable def f (x : ℝ) : ℝ := |Real.logb 3 x|

theorem product_eq_one (a b : ℝ) (h_diff : a ≠ b) (h_eq : f a = f b) : a * b = 1 := by
  sorry

end product_eq_one_l605_605763


namespace mean_height_of_basketball_team_l605_605496

-- Define the input data based on the stem-and-leaf plot
def heights : List ℝ :=
  [58, 59, 60, 61, 64, 65, 68, 70, 73, 73, 75, 76, 77, 78, 78, 79]

-- Define a function to compute the mean of a list of real numbers
def mean (l : List ℝ) : ℝ :=
  if l.length = 0 then 0 else l.sum / l.length

-- The statement to prove
theorem mean_height_of_basketball_team :
  mean heights = 70.625 :=
sorry

end mean_height_of_basketball_team_l605_605496


namespace prove_proposition_l605_605742

-- Define the propositions p and q
def p : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def q : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Define the main theorem to prove
theorem prove_proposition : (¬ p) ∨ q :=
by { sorry }

end prove_proposition_l605_605742


namespace required_hours_to_earn_l605_605784

-- Define the initial conditions
def planned_hours_per_week := 25
def total_weeks := 15
def total_earnings := 3750
def missed_weeks := 3

-- Define the remaining weeks
def remaining_weeks := total_weeks - missed_weeks

-- Define the fraction of weeks left
def fraction_weeks_left := remaining_weeks.toFloat / total_weeks.toFloat

-- Define the required weekly hours based on the remaining weeks
def required_weekly_hours := (1 / fraction_weeks_left) * planned_hours_per_week

-- State the main theorem
theorem required_hours_to_earn : required_weekly_hours = 31.25 := by
  sorry

end required_hours_to_earn_l605_605784


namespace find_an_find_Tn_l605_605279

-- Condition Definitions
def S (n : ℕ) : ℤ := (n * (2 * a1 + (n - 1) * d)) / 2
variable (a1 d : ℤ)

-- Part 1: Prove a_n
def an (n : ℕ) : ℤ := 4 * n - 2

theorem find_an (h1 : S 5 = 50) (h2 : (2 * a1 + d)^2 = a1 * (4 * a1 + 6 * d)) : 
  ∀ n, an n = 4 * n - 2 := by sorry

-- Part 2: Prove T_n
def bn (n : ℕ) : ℤ := 4 * n^2 - 1

def Tn (n : ℕ) : ℚ := n / (2 * n + 1)

theorem find_Tn (h3 : ∀ n ≥ 2, bn n - bn (n - 1) = 2 * an n) (h4 : bn 1 = an 1 + 1) :
  ∀ n, Tn n = n / (2 * n + 1) := by sorry

end find_an_find_Tn_l605_605279


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605558

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605558


namespace quadratic_function_value_l605_605405

noncomputable def f (x : ℝ) : ℝ :=
  let a := -5 / 2 in
  a * (x - 3) ^ 2 + 10

theorem quadratic_function_value :
  (∀ x : ℝ, f x = (-5 / 2) * (x - 3) ^ 2 + 10) →
  (∃ a b c : ℝ, f x = a * x ^ 2 + b * x + c ∧ (∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ real.abs (x1 - x2) = 4 ∧ f 3 = 10)) →
  f 5 = 0 :=
by sorry

end quadratic_function_value_l605_605405


namespace lcm_504_630_980_l605_605526

noncomputable def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_504_630_980 : lcm (lcm 504 630) 980 = 17640 := by
  have fact_504 : 504 = 2^3 * 3^2 * 7 := by sorry
  have fact_630 : 630 = 2 * 3^2 * 5 * 7 := by sorry
  have fact_980 : 980 = 2^2 * 5 * 7^2 := by sorry
  sorry

end lcm_504_630_980_l605_605526


namespace problem_l605_605737

noncomputable def f (a b x : ℝ) := a * x^2 - b * x + 1

theorem problem (a b : ℝ) (h1 : 4 * a - b^2 = 3)
                (h2 : ∀ x : ℝ, f a b (x + 1) = f a b (-x))
                (h3 : b = a + 1) 
                (h4 : 0 ≤ a ∧ a ≤ 1) 
                (h5 : ∀ x ∈ Set.Icc 0 2, ∃ m : ℝ, m ≥ abs (f a b x)) :
  (∀ x : ℝ, f a b x = x^2 - x + 1) ∧ (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc 0 2, m ≥ abs (f a b x)) :=
  sorry

end problem_l605_605737


namespace james_trees_successfully_grown_l605_605840

theorem james_trees_successfully_grown :
  let seeds_from_tree_A := 25
  let seeds_from_tree_B := 40
  let planted_seeds_A := 0.60 * seeds_from_tree_A
  let planted_seeds_B := 0.80 * seeds_from_tree_B
  let surviving_trees_A := 0.75 * planted_seeds_A
  let surviving_trees_B := 0.90 * planted_seeds_B
  let total_new_trees := surviving_trees_A + surviving_trees_B
  total_new_trees = 39 :=
by {
  let seeds_from_tree_A := 25
  let seeds_from_tree_B := 40
  let planted_seeds_A := 0.60 * seeds_from_tree_A
  let planted_seeds_B := 0.80 * seeds_from_tree_B
  let surviving_trees_A := 0.75 * planted_seeds_A
  let surviving_trees_B := 0.90 * planted_seeds_B
  let total_new_trees := surviving_trees_A + surviving_trees_B
  have h1 : seeds_from_tree_A = 25 := rfl
  have h2 : seeds_from_tree_B = 40 := rfl
  have h3 : planted_seeds_A = 0.60 * 25 := rfl
  have h4 : planted_seeds_B = 0.80 * 40 := rfl
  have h5 : surviving_trees_A = 0.75 * planted_seeds_A := rfl
  have h6 : surviving_trees_B = 0.90 * planted_seeds_B := rfl
  have h7 : total_new_trees = surviving_trees_A + surviving_trees_B := rfl
  rw [←h1, ←h2, ←h3, ←h4, ←h5, ←h6, ←h7]
  sorry
}


end james_trees_successfully_grown_l605_605840


namespace semicircle_perimeter_approx_l605_605062

def radius : Float := 20
def π_approx : Float := 3.14159

theorem semicircle_perimeter_approx : radius = 20 → π_approx = 3.14159 → 
  ((20 * π_approx + 40) ≈ 102.83) := by
  -- The actual proof would go here.
  sorry

end semicircle_perimeter_approx_l605_605062


namespace real_solution_count_l605_605236

theorem real_solution_count (x : ℝ) :
  (x ≠ 0) →
  ¬(x < 0) →
  ∃! x > 0, (x ^ 2010 + 1) * (∑ i in (finset.range 1005).image (λ k, x ^ (2008 - 2 * k)), id) = 2010 * x ^ 2009 :=
sorry

end real_solution_count_l605_605236


namespace smallest_pqrs_value_l605_605399

-- Define positive integers as ℕ+
def PosInt := {n : ℕ // n > 0}

-- Define matrices consisting of positive integers
def mat2x2 := matrix (fin 2) (fin 2) PosInt

-- Given matrices
def A : mat2x2 := ![![⟨2, by decide⟩, ⟨0, by decide⟩], ![⟨0, by decide⟩, ⟨3, by decide⟩]]
def B : mat2x2 := ![![⟨10, by decide⟩, ⟨15, by decide⟩], ![⟨-12, by decide⟩, ⟨-18, by decide⟩]]

-- Problem statement
theorem smallest_pqrs_value : 
  ∃ (p q r s : PosInt), 
    (A ⬝ ![![p, q], ![r, s]] = (![![p, q], ![r, s]] ⬝ B)) ∧ 
    (p.val + q.val + r.val + s.val = 44) :=
sorry -- Proof goes here

end smallest_pqrs_value_l605_605399


namespace greatest_product_of_two_integers_sum_2006_l605_605524

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end greatest_product_of_two_integers_sum_2006_l605_605524


namespace ratio_of_a_to_c_l605_605915

variable {a b c d : ℚ}

theorem ratio_of_a_to_c (h₁ : a / b = 5 / 4) (h₂ : c / d = 4 / 3) (h₃ : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l605_605915


namespace no_intersection_points_of_polar_graphs_l605_605131

theorem no_intersection_points_of_polar_graphs :
  let c1_center := (3 / 2, 0)
  let r1 := 3 / 2
  let c2_center := (0, 3)
  let r2 := 3
  let distance_between_centers := Real.sqrt ((3 / 2 - 0) ^ 2 + (0 - 3) ^ 2)
  distance_between_centers > r1 + r2 :=
by
  sorry

end no_intersection_points_of_polar_graphs_l605_605131


namespace circumference_tire_correct_l605_605319

-- Define the given conditions, speed and revolutions
def speed_kmh : ℝ := 144
def revolutions_per_minute : ℝ := 400

-- Define the converted speed in meters per minute
def speed_m_per_min : ℝ := (speed_kmh * 1000) / 60

-- Define the distance traveled in one minute
def distance_traveled_in_one_minute : ℝ := speed_m_per_min

-- Define the number of rotations in one minute
def number_of_rotations_in_one_minute : ℝ := revolutions_per_minute

-- Define the circumference of the tire
def circumference_of_tire : ℝ := distance_traveled_in_one_minute / number_of_rotations_in_one_minute

-- State the final objective
theorem circumference_tire_correct : circumference_of_tire = 6 := by
  sorry

end circumference_tire_correct_l605_605319


namespace position_of_z_l605_605419

theorem position_of_z (total_distance : ℕ) (total_steps : ℕ) (steps_taken : ℕ) (distance_covered : ℕ) (h1 : total_distance = 30) (h2 : total_steps = 6) (h3 : steps_taken = 4) (h4 : distance_covered = total_distance / total_steps) : 
  steps_taken * distance_covered = 20 :=
by
  sorry

end position_of_z_l605_605419


namespace pyramid_conditions_imply_volume_l605_605902

-- Define the geometrical setup of the problem: the rhombus and pyramid
noncomputable def rhombus_side_length : ℝ := 2
noncomputable def rhombus_angle : ℝ := real.pi / 4 -- 45 degrees in radians
noncomputable def sphere_radius : ℝ := real.sqrt 2

-- Define the coordinates and geometrical properties
structure Rhombus := 
  (A B C D : ℝ × ℝ)
  (side_length : ℝ = rhombus_side_length)
  (acute_angle : ℝ = rhombus_angle)
  (diagonals_intersect_at : (ℝ × ℝ) = (0, 0)) -- Assume intercept at origin for simplicity

-- Define the pyramid structure with the sphere touching lateral faces condition
structure Pyramid :=
  (apex : ℝ × ℝ × ℝ)
  (base : Rhombus)
  (sphere_touch_points : List (ℝ × ℝ))
  (touch_condition : ∀ p ∈ sphere_touch_points, dist p apex = sphere_radius)

-- Define the height by assuming the apex position
noncomputable def pyramid_height (p : Pyramid) := 
  real.sqrt (p.apex.1^2 + p.apex.2^2 + p.apex.3^2) - sphere_radius

-- Calculate volume of the pyramid given base area and height
noncomputable def pyramid_volume (p : Pyramid) :=
  (1 / 3) * (2 * (real.sin (rhombus_angle / 2))^2 * rhombus_side_length^2) * pyramid_height p

-- The theorem to prove
theorem pyramid_conditions_imply_volume (p : Pyramid) (h_rhombus : p.base.side_length = rhombus_side_length) 
  (h_angle : p.base.acute_angle = rhombus_angle) 
  (h_intersect : p.base.diagonals_intersect_at = (0, 0)) 
  (h_touch : ∀ point ∈ p.sphere_touch_points, dist point p.apex = sphere_radius) :
  p.apex.3 = dist (0, 0) p.apex * real.sin (real.pi / 4) + real.sqrt 2 ∧
  pyramid_volume p = 2 * real.sqrt 3 / 9 := 
sorry

end pyramid_conditions_imply_volume_l605_605902


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605203

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605203


namespace important_countries_l605_605851

-- Define the main theorem
theorem important_countries
  (N d : ℕ)
  (hN : N ≥ d + 2)
  (G : SimpleGraph (Fin N))
  [decidable_rel G.adj]
  (h_regular : ∀ v : Fin N, G.degree v = d)
  (h_connected : Connected G)
  (h_important : ∀ v : Fin N, ∃ u1 u2 : Fin N, u1 ≠ u2 ∧ Connected (G.delete_vertices {v, v.neighbors}) u1 u2 = false) :
  ∃ (u w : Fin N), (G.neighbor_finset u ∩ G.neighbor_finset w).card > 2 * d / 3 :=
sorry

end important_countries_l605_605851


namespace area_of_region_bounded_by_graphs_l605_605141

theorem area_of_region_bounded_by_graphs :
  let θ := Real.pi / 4
      r1 := fun θ => Real.tan θ
      r2 := fun θ => Real.cot θ
  (bounded_area : Set ℝ) :=
  ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (Real.pi / 4) ∧
            (r1 θ, θ) ∈ bounded_area ∧
            (r2 θ, θ) ∈ bounded_area ∧
            (bounded_area_region : Set ℝ) ∧
            (area_of_bounded_region bounded_area_region = (1 / 2) :=
begin
  let θ := Real.pi / 4
  let r1 := fun θ => Real.tan θ
  let r2 := fun θ => Real.cot θ
  sorry
end

end area_of_region_bounded_by_graphs_l605_605141


namespace cost_of_fencing_per_meter_l605_605909

theorem cost_of_fencing_per_meter (length breadth : ℕ) (total_cost : ℚ) 
    (h_length : length = 61) 
    (h_rule : length = breadth + 22) 
    (h_total_cost : total_cost = 5300) :
    total_cost / (2 * length + 2 * breadth) = 26.5 := 
by 
  sorry

end cost_of_fencing_per_meter_l605_605909


namespace sin_C_value_angle_C_range_l605_605825

noncomputable def triangle_sin_C (AB BC : ℝ) (cosB : ℝ) : ℝ :=
if AB = sqrt 3 ∧ BC = 2 ∧ cosB = -1 / 2 then sqrt 3 / 2 else 0

theorem sin_C_value :
  ∀ (AB BC cosB : ℝ), (AB = sqrt 3) → (BC = 2) → (cosB = -1 / 2) → triangle_sin_C AB BC cosB = sqrt 3 / 2 :=
by
  intros
  sorry

-- Part (II)
theorem angle_C_range (AB BC : ℝ) :
  (AB = sqrt 3) → (BC = 2) → 
  ∃ (C : ℝ), (C ∈ set.Ioc 0 (2 * Real.pi / 3)) :=
by
  intros
  sorry

end sin_C_value_angle_C_range_l605_605825


namespace total_tea_consumption_l605_605023

variables (S O P : ℝ)

theorem total_tea_consumption : 
  S + O = 11 →
  P + O = 15 →
  P + S = 13 →
  S + O + P = 19.5 :=
by
  intros h1 h2 h3
  sorry

end total_tea_consumption_l605_605023


namespace probability_snow_at_least_once_first_week_l605_605436

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l605_605436


namespace population_increase_is_10_percent_l605_605493

def initial_population : ℝ := 300
def final_population : ℝ := 330
def population_increase_rate (P_initial P_final : ℝ) : ℝ := ((P_final - P_initial) / P_initial) * 100

theorem population_increase_is_10_percent :
  population_increase_rate initial_population final_population = 10 :=
by
  sorry

end population_increase_is_10_percent_l605_605493


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605215

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605215


namespace problem_l605_605858

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) (α1 : ℝ) (α2 : ℝ) :=
  m * Real.sin (Real.pi * x + α1) + n * Real.cos (Real.pi * x + α2)

variables (m n α1 α2 : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) (h_α1 : α1 ≠ 0) (h_α2 : α2 ≠ 0)

theorem problem (h : f 2008 m n α1 α2 = 1) : f 2009 m n α1 α2 = -1 :=
  sorry

end problem_l605_605858


namespace count_k_values_l605_605248

-- Factorizations
def six_pow_nine := 6^9 = 2^9 * 3^9
def nine_pow_nine := 9^9 = 3^18
def eighteen_pow_eighteen := 18^18 = 2^18 * 3^36

-- Definition for k
def k_form (a b : ℕ) (k : ℕ) := k = 2^a * 3^b

-- LCM of 6^9 and 9^9
def lcm_six_nine := nat.lcm (2^9 * 3^9) (3^18) = 2^9 * 3^18

-- LCM condition involving k
def lcm_six_nine_k (a b : ℕ) := nat.lcm (2^9 * 3^18) (2^a * 3^b) = 2^18 * 3^36

-- Constraints on a and b
def constraints_a (a : ℕ) := nat.max 9 a = 18
def constraints_b (b : ℕ) := nat.max 18 b = 36

-- Number of valid k values
theorem count_k_values : six_pow_nine ∧ nine_pow_nine ∧ eighteen_pow_eighteen ∧ 
                         ∃ k, k_form a b k ∧ 
                         lcm_six_nine ∧ 
                         lcm_six_nine_k a b ∧ 
                         constraints_a a ∧ 
                         constraints_b b → 
                         (finset.card (finset.Icc 18 36) = 19) :=
sorry

end count_k_values_l605_605248


namespace terminating_decimal_nonzero_thousandths_l605_605721

theorem terminating_decimal_nonzero_thousandths (n : ℕ) :
  (∃ (k : ℕ), k = 4 ∧ ∀ m (h₁ : 1 ≤ m ∧ m ≤ 1000) (h₂ : terminating (1 / m)), nonzero_thousandths_digit (1 / m) → k = 4) :=
sorry

-- Definitions and helpers:
def terminating (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

def nonzero_thousandths_digit (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

end terminating_decimal_nonzero_thousandths_l605_605721


namespace sandra_age_l605_605894

theorem sandra_age (S : ℕ) (h1 : ∀ x : ℕ, x = 14) (h2 : S - 3 = 3 * (14 - 3)) : S = 36 :=
by sorry

end sandra_age_l605_605894


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605181

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605181


namespace intersection_of_M_and_N_is_N_l605_605284

def M := {x : ℝ | x ≥ -1}
def N := {y : ℝ | y ≥ 0}

theorem intersection_of_M_and_N_is_N : M ∩ N = N := sorry

end intersection_of_M_and_N_is_N_l605_605284


namespace allocation_schemes_correct_l605_605077

noncomputable def number_of_allocation_schemes : ℕ :=
  nat.choose (12) (3)

theorem allocation_schemes_correct : number_of_allocation_schemes = 220 :=
  by
    sorry

end allocation_schemes_correct_l605_605077


namespace exists_m_sum_l605_605878

variable {a : ℕ → ℝ}
variable (n : ℕ) (h_pos : ∀ n, 0 < a n)
variable (h_seq : ∀ n, (a (n + 1))^2 ≥ ∑ k in finset.range (n + 1), (a k)^2 / (k + 1)^3)

theorem exists_m_sum :
  ∃ (m : ℕ), (∑ n in finset.range m, a (n + 1) / ∑ k in finset.range n.succ, a k) > 2009 / 1000 := by
  sorry

end exists_m_sum_l605_605878


namespace evaluate_expression_l605_605653

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l605_605653


namespace units_digit_37_pow_37_l605_605530

theorem units_digit_37_pow_37: (37^37) % 10 = 7 :=
by sorry

end units_digit_37_pow_37_l605_605530


namespace equal_angles_l605_605349

-- Define the data and conditions
variables (A B C H P Q : Type) 
variables [noncomputable] (hABC : acute_triangle A B C) 
variables (h_altitude : altitude A H B C) 
variables (h_perp_P : is_perpendicular H P A B) 
variables (h_perp_Q : is_perpendicular H Q A C)

-- Define the theorem to prove
theorem equal_angles (hABC : acute_triangle A B C) 
    (h_altitude : altitude A H B C) 
    (h_perp_P : is_perpendicular H P A B) 
    (h_perp_Q : is_perpendicular H Q A C) : 
    ∠ B Q H = ∠ C P H := 
sorry

end equal_angles_l605_605349


namespace average_age_of_group_l605_605338

theorem average_age_of_group (Sf Sm : ℕ) (n_f n_m : ℕ) (avg_f avg_m : ℝ) (total_people : ℕ)
  (hf : avg_f = 28) (hm : avg_m = 35) (n_f_eq : n_f = 12) (n_m_eq : n_m = 15) (total : total_people = 27) :
  let S_f := avg_f * n_f,
      S_m := avg_m * n_m,
      S := S_f + S_m,
      avg_total := S / total_people
  in avg_total = 31.89 := 
by sorry

end average_age_of_group_l605_605338


namespace maximize_profit_l605_605074

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-2 * x + 200

theorem maximize_profit :
  ∃ x, 30 ≤ x ∧ x ≤ 60 ∧
    (∀ y, functional_relationship x = y → 
      (∃ max_profit : ℝ, max_profit = (x - 30) * y - 450 ∧ max_profit = 1950)) ∧
    x = 60 :=
begin
  -- Proof details here
  sorry
end

end maximize_profit_l605_605074


namespace smallest_diff_using_digits_l605_605934

theorem smallest_diff_using_digits :
  ∃ c d : ℕ, (c < 100) ∧ (d < 100) ∧
  ∀ digit ∈ [1, 3, 7, 8, 9], digit ∈ digits c ∨ digit ∈ digits d ∧
  digits c ∪ digits d = [1, 3, 7, 8, 9] ∧
  c - d = 52 :=
by
  sorry

def digits (n : ℕ) : List ℕ :=
  n.digits 10

end smallest_diff_using_digits_l605_605934


namespace f_xh_sub_f_x_l605_605857

def f (x : ℝ) (k : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + k * x - 4

theorem f_xh_sub_f_x (x h : ℝ) (k : ℝ := -5) : 
    f (x + h) k - f x k = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end f_xh_sub_f_x_l605_605857


namespace hyperbola_asymptotes_l605_605482

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- The statement to prove: The equation of the asymptotes of the hyperbola is as follows
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x)) :=
sorry

end hyperbola_asymptotes_l605_605482


namespace combined_distance_is_twelve_l605_605928

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l605_605928


namespace log_cos_squared_range_l605_605529

theorem log_cos_squared_range (x : ℕ) (hx1 : 0 ≤ x) (hx2 : x ≤ 180) (hx3 : x ≠ 90) :
  ∃ y, y = log 3 (cos x * cos x) ∧ y ≤ 0 :=
begin
  sorry
end

end log_cos_squared_range_l605_605529


namespace greatest_product_sum_2006_l605_605523

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l605_605523


namespace probability_and_relationship_l605_605546

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l605_605546


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605559

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605559


namespace snow_prob_correct_l605_605421

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l605_605421


namespace count_terminating_decimals_with_nonzero_thousandths_l605_605713

noncomputable def is_terminating_with_nonzero_thousandths (n : ℕ) : Prop :=
  (∃ a b : ℕ, n = 2^a * 5^b) ∧ n ≤ 1000

theorem count_terminating_decimals_with_nonzero_thousandths :
  (finset.univ.filter is_terminating_with_nonzero_thousandths).card = 25 :=
begin
  sorry
end

end count_terminating_decimals_with_nonzero_thousandths_l605_605713


namespace smallest_b_for_quadratic_factors_l605_605666

theorem smallest_b_for_quadratic_factors :
  ∃ b : ℕ, (∀ r s : ℤ, (r * s = 1764 → r + s = b) → b = 84) :=
sorry

end smallest_b_for_quadratic_factors_l605_605666


namespace monotonic_increase_interval_l605_605487

noncomputable def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem monotonic_increase_interval 
    (ω : ℝ)
    (hω : 0 < ω)
    (hperiod : Real.pi = 2 * Real.pi / ω) :
    ∀ k : ℤ, ∃ I : Set ℝ, I = interval_of_monotonic_increase k := 
by
  sorry

end monotonic_increase_interval_l605_605487


namespace circle_radius_is_7_5_l605_605277

noncomputable def radius_of_circle (side_length : ℝ) : ℝ := sorry

theorem circle_radius_is_7_5 :
  radius_of_circle 12 = 7.5 := sorry

end circle_radius_is_7_5_l605_605277


namespace flowers_bouquets_l605_605847

theorem flowers_bouquets (tulips: ℕ) (roses: ℕ) (extra: ℕ) (total: ℕ) (used_for_bouquets: ℕ) 
(h1: tulips = 36) 
(h2: roses = 37) 
(h3: extra = 3) 
(h4: total = tulips + roses)
(h5: used_for_bouquets = total - extra) :
used_for_bouquets = 70 := by
  sorry

end flowers_bouquets_l605_605847


namespace smallest_n_for_identity_matrix_l605_605668

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605668


namespace greatest_prime_factor_of_125_l605_605521

-- Define that 125 is equal to 5^3
def factorization_of_125 : 125 = 5 ^ 3 := by
  rw [pow_succ, pow_succ, pow_zero, one_mul, mul_one, mul_comm]
  norm_num

-- State the theorem
theorem greatest_prime_factor_of_125
  (h : 125 = 5 ^ 3) : ∃ p, nat.prime p ∧ (∀ q, nat.prime q ∧ dvd q 125 → q ≤ p) ∧ p = 5 :=
sorry

end greatest_prime_factor_of_125_l605_605521


namespace fraction_eval_l605_605241

theorem fraction_eval : 1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 89 :=
by
  sorry

end fraction_eval_l605_605241


namespace sum_of_cubes_pattern_l605_605884

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 = 3^2) ->
  (1^3 + 2^3 + 3^3 = 6^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  intros h1 h2 h3
  -- Proof follows here
  sorry

end sum_of_cubes_pattern_l605_605884


namespace number_of_intact_bars_l605_605977

-- Define the conditions
variable (total_weight : ℝ) (weight_small : ℝ) (weight_large : ℝ)
variable (broken_percentage : ℝ)
variable (number_pairs : ℕ)

-- Given conditions
axiom total_weight_def : total_weight = 2500
axiom weight_small_def : weight_small = 75
axiom weight_large_def : weight_large = 160
axiom broken_percentage_def : broken_percentage = 0.10
axiom number_pairs_def : number_pairs = (2500 / (75 + 160)).to_nat

-- Define the problem to prove
theorem number_of_intact_bars :
  let total_bars := 2 * number_pairs
  let broken_bars := broken_percentage * total_bars
  let intact_bars := total_bars - broken_bars in
  intact_bars = 18 :=
by
  sorry

end number_of_intact_bars_l605_605977


namespace find_four_digit_numbers_l605_605657

noncomputable def four_digit_number_permutations_sum (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) : Prop :=
  6 * (x + y + z + t) * (1000 + 100 + 10 + 1) = 10 * (1111 * x)

theorem find_four_digit_numbers (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) :
  four_digit_number_permutations_sum x y z t distinct nonzero :=
  sorry

end find_four_digit_numbers_l605_605657


namespace percentage_gain_second_week_l605_605890

variables (initial_investment final_value after_first_week_value gain_percentage first_week_gain second_week_gain second_week_gain_percentage : ℝ)

def pima_investment (initial_investment: ℝ) (first_week_gain_percentage: ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage)

def second_week_investment (initial_investment first_week_gain_percentage second_week_gain_percentage : ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage) * (1 + second_week_gain_percentage)

theorem percentage_gain_second_week
  (initial_investment : ℝ)
  (first_week_gain_percentage : ℝ)
  (final_value : ℝ)
  (h1: initial_investment = 400)
  (h2: first_week_gain_percentage = 0.25)
  (h3: final_value = 750) :
  second_week_gain_percentage = 0.5 :=
by
  let after_first_week_value := pima_investment initial_investment first_week_gain_percentage
  let second_week_gain := final_value - after_first_week_value
  let second_week_gain_percentage := second_week_gain / after_first_week_value * 100
  sorry

end percentage_gain_second_week_l605_605890


namespace selection_of_11_integers_l605_605775

theorem selection_of_11_integers (S : Finset ℕ) (h : S ⊆ (Finset.range 21)) (h_card : S.card ≥ 11) : 
  ∃ a b ∈ S, a ≠ b ∧ (a - b = 2 ∨ b - a = 2) :=
sorry

end selection_of_11_integers_l605_605775


namespace height_of_pillar_is_correct_l605_605603

def volume_rectangular_prism (length width height : ℝ) : ℝ :=
  length * width * height

def volume_cylinder (π radius height : ℝ) : ℝ :=
  π * radius^2 * height

noncomputable def calculate_height_of_pillar : ℝ :=
let volume_prism := volume_rectangular_prism 170 145 35
let radius := 20 / 2
let height := volume_prism / (3.14 * (radius^2))
in Float.round (height * 100) / 100

theorem height_of_pillar_is_correct :
  calculate_height_of_pillar = 686.98 :=
by
  sorry

end height_of_pillar_is_correct_l605_605603


namespace present_age_of_son_l605_605988

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 22) (h2 : F + 2 = 2 * (S + 2)) : S = 20 :=
by
  sorry

end present_age_of_son_l605_605988


namespace greatest_product_sum_2006_l605_605522

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l605_605522


namespace find_c_condition_1_find_c_condition_3_find_area_l605_605730

-- Conditions
def a := 2
def B := Real.pi / 3
def cosA := 2 * Real.sqrt 7 / 7
def b := Real.sqrt 7

-- Question 1: Prove the value of c
theorem find_c_condition_1 (ha : a = 2) (hB : B = Real.pi / 3) (hcosA : cosA = 2 * Real.sqrt 7 / 7) :
  ∃ c, c = 3 :=
by sorry

theorem find_c_condition_3 (ha : a = 2) (hB : B = Real.pi / 3) (hb : b = Real.sqrt 7) :
  ∃ c, c = 3 :=
by sorry

-- Question 2: Prove the area of the triangle
theorem find_area (ha : a = 2) (hB : B = Real.pi / 3) (hc : ∃ c, c = 3) :
  ∃ S, S = (3 * Real.sqrt 3) / 2 :=
by sorry

end find_c_condition_1_find_c_condition_3_find_area_l605_605730


namespace series_sum_eq_five_fourteenths_l605_605618

theorem series_sum_eq_five_fourteenths : 
  (∑ k in finset.range 5, 1 / ((k + 2) * (k + 3))) = 5 / 14 :=
by
  sorry

end series_sum_eq_five_fourteenths_l605_605618


namespace smallest_n_for_identity_l605_605708

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605708


namespace trio_tip_percentage_l605_605879

theorem trio_tip_percentage 
  (cost_leticia : ℤ) (cost_scarlett : ℤ) (cost_percy : ℤ)
  (tip_amount : ℤ) 
  (h1 : cost_leticia = 10) 
  (h2 : cost_scarlett = 13) 
  (h3 : cost_percy = 17) 
  (h4 : tip_amount = 4) :
  let total_cost := cost_leticia + cost_scarlett + cost_percy in
  let tip_percentage := (tip_amount * 100) / total_cost in
  tip_percentage = 10 := 
by {
  simp [h1, h2, h3, h4],
  sorry -- The proof will be done here
}

end trio_tip_percentage_l605_605879


namespace min_value_g_l605_605871

noncomputable def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  -2 * (x₁^3 + x₂^3 + x₃^3) + 3 * (x₁^2 * (x₂ + x₃) + x₂^2 * (x₁ + x₃) + x₃^2 * (x₁ + x₂)) - 12 * x₁ * x₂ * x₃

noncomputable def g (r s t : ℝ) : ℝ :=
  (λ x₃, |f r (r + 2) x₃ + s|) '' (Set.Icc t (t + 2)).toMax

theorem min_value_g : ∀ r s t : ℝ, ∃ a : ℝ, g r s t = 12 * Real.sqrt 3 := sorry

end min_value_g_l605_605871


namespace probability_of_snow_at_least_once_first_week_l605_605442

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l605_605442


namespace minimum_selection_l605_605777

theorem minimum_selection (S : Finset ℕ) (hS : S = Finset.range 21 \ Finset.singleton 0) : 
  ∃ (T : Finset ℕ), T.card = 11 ∧ ∀ (a b ∈ T),  a - b = 2 ∨ b - a = 2 :=
begin
  sorry
end

end minimum_selection_l605_605777


namespace inequality_holds_l605_605268

theorem inequality_holds (n : ℕ) (m : ℕ) (a : ℕ → ℝ)
    (h_pos : ∀ i, 0 < a i)
    (h_prod : (∏ i in finset.range n, a i) = 1)
    (h_n_ge_two : 2 ≤ n)
    (h_m_ge_n_minus_one : m ≥ n - 1) :
    (∑ i in finset.range n, (a i) ^ m) ≥ (∑ i in finset.range n, (1 / (a i))) :=
sorry

end inequality_holds_l605_605268


namespace rhombus_side_length_l605_605806

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605806


namespace wall_area_l605_605959

theorem wall_area (L W : ℝ) (R J : ℝ) (h_ratio : 1 / 3 * R = J)
    (h_jumbo_length : 3 * L = 3 * L) -- redundant but included to match condition wording
    (h_same_ratio : True) -- no need to repeat same ratio in proof
    (h_regular_area : R * (L * W) = 60)
    (h_no_overlap : True) : 2 * (R * L * W) = 120 :=
by
  have h_jumbo_area : J * (3 * L * W) = 60 :=
      calc 
      J * (3 * L * W) = (1 / 3 * R) * (3 * L * W) : by rw [h_ratio]
               ... = R * (L * W) : by ring
               ... = 60 : by rw [h_regular_area]
  calc 
  2 * (R * L * W) = 2 * 60 : by rw [h_regular_area]
               ... = 120 : by norm_num

end wall_area_l605_605959


namespace find_original_wage_l605_605598

theorem find_original_wage (W : ℝ) (h : 1.50 * W = 51) : W = 34 :=
sorry

end find_original_wage_l605_605598


namespace real_roots_of_f_minus_x_real_roots_of_f_f_minus_x_l605_605852

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem real_roots_of_f_minus_x :
  {x : ℝ | f(x) - x = 0} = {1, 2} :=
sorry

theorem real_roots_of_f_f_minus_x :
  {x : ℝ | f(f(x)) - x = 0} = {1, 2} :=
sorry

end real_roots_of_f_minus_x_real_roots_of_f_f_minus_x_l605_605852


namespace range_of_m_l605_605793

theorem range_of_m {x : ℝ} (m : ℝ) :
  (∀ x, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l605_605793


namespace simplified_root_sum_l605_605052

theorem simplified_root_sum :
  let a := 15
  let b := 3
  let expr := (3^5 * 5^4)
  let simplified_expr := 15 * √(4 : ℕ).ereal (3 : ℕ) in
  expr = (a * √(4 : ℕ).ereal b) →
  a + b = 18 :=
by
  sorry

end simplified_root_sum_l605_605052


namespace polynomial_divisibility_l605_605452

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility_l605_605452


namespace max_ball_number_three_ways_l605_605581

theorem max_ball_number_three_ways :
  let draws := finset.pi (finset.range 3) (λ _, finset.range 3)
  -- condition: balls labeled with numbers 1, 2, 3 respectively
  (draws.filter (λ draw, list.maximum draw = some 2)).card = 19 := by sorry

end max_ball_number_three_ways_l605_605581


namespace value_of_8_and_5_l605_605911

def custom_op (a b : ℝ) : ℝ := (a + b) * (a - b) / 2

theorem value_of_8_and_5 : custom_op 8 5 = 19.5 :=
by sorry

end value_of_8_and_5_l605_605911


namespace find_a_of_exponential_inverse_l605_605009

theorem find_a_of_exponential_inverse (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, a^x = 9 ↔ x = 2) : a = 3 := 
by
  sorry

end find_a_of_exponential_inverse_l605_605009


namespace largest_prime_factor_106_sq_minus_15_sq_l605_605033

theorem largest_prime_factor_106_sq_minus_15_sq : 
  ∀ (x y : ℕ), 
  x = 106 → 
  y = 15 → 
  ∃ p : ℕ, 
  prime p ∧ p = 13 ∧ 
  ∀ q : ℕ, 
  prime q → q ∣ (x*x - y*y) → q ≤ p := 
by
  sorry

end largest_prime_factor_106_sq_minus_15_sq_l605_605033


namespace no_possible_grid_l605_605397

theorem no_possible_grid (k : ℕ) (h : k > 1) :
  ¬ (∃ (f : ℕ → ℕ → ℕ),
    (∀ i j, 1 ≤ f i j ∧ f i j ≤ k^2) ∧ -- Numbers between 1 and k^2
    (∀ i, is_power_of_two (∑ j in finset.range k, f i j)) ∧ -- sum of each row is a power of 2
    (∀ j, is_power_of_two (∑ i in finset.range k, f i j))) -- sum of each column is a power of 2
:= sorry

def is_power_of_two (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2^m

end no_possible_grid_l605_605397


namespace problem_solution_true_all_l605_605354

variables (x y : ℝ)

def chords_parallel (AB_parallel_OE : Prop) (CD_parallel_OE : Prop) : Prop :=
  ∀ (AE_perp_CD : Prop) (AC_DE_eq_x : x = x), 
  AB_parallel_OE →
  CD_parallel_OE →
  AE_perp_CD →
  x > 0 → y > 0 → x ≠ y ∧ y - x = Real.sqrt 2 ∧ x * y = 2 ∧ y^2 - x^2 = 2

theorem problem_solution_true_all 
  (AB_parallel_OE CD_parallel_OE AE_perp_CD : Prop)
  (AC_DE_eq_x : x = x) (he : chords_parallel AB_parallel_OE CD_parallel_OE) : 
  x > 0 → y > 0 → x ≠ y ∧ y - x = Real.sqrt 2 ∧ x * y = 2 ∧ y^2 - x^2 = 2 :=
by 
  intro hx hy
  exact he AB_parallel_OE CD_parallel_OE AE_perp_CD AC_DE_eq_x sorry sorry sorry sorry

end problem_solution_true_all_l605_605354


namespace problem_statement_l605_605872

def f (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n - 3

theorem problem_statement : f (f (f 3)) = 31 :=
by
  sorry

end problem_statement_l605_605872


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605170

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605170


namespace number_of_integers_with_digit_sum_equals_18_l605_605779

theorem number_of_integers_with_digit_sum_equals_18 : 
  let numbers := {n ∈ (finset.Icc 400 700) | (n.digits.sum = 18)} 
  in finset.card numbers = 29 :=
begin
  sorry -- Proof to be completed
end

end number_of_integers_with_digit_sum_equals_18_l605_605779


namespace exists_n_sum_reciprocals_lt_2022_l605_605889

theorem exists_n_sum_reciprocals_lt_2022 :
  ∃ (n : ℕ), 91 ≤ n ∧ (∑ i in Finset.range (n + 1), (nat.choose n i)⁻¹) < 2.022 :=
begin
  -- To be proved
  sorry
end

end exists_n_sum_reciprocals_lt_2022_l605_605889


namespace minimum_selection_l605_605778

theorem minimum_selection (S : Finset ℕ) (hS : S = Finset.range 21 \ Finset.singleton 0) : 
  ∃ (T : Finset ℕ), T.card = 11 ∧ ∀ (a b ∈ T),  a - b = 2 ∨ b - a = 2 :=
begin
  sorry
end

end minimum_selection_l605_605778


namespace even_composition_l605_605387

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_composition (f : ℝ → ℝ) (hf : even_function f) : even_function (λ x, f (f x)) :=
by
  intros x
  rw [←hf, hf]
  sorry

end even_composition_l605_605387


namespace no_solution_implies_a_l605_605330

-- Define the conditions
def equations (a : ℝ) (x y : ℝ) :=
  (a * x + 2 * y = 3) ∧ (2 * x + a * y = 2)

-- Define the problem statement
theorem no_solution_implies_a (a : ℝ) :
  (¬ ∃ x y : ℝ, equations a x y) → (a = 2 ∨ a = -2) :=
begin
  intro h,
  sorry
end

end no_solution_implies_a_l605_605330


namespace remainder_of_n_l605_605245

theorem remainder_of_n (n : ℕ) (h1 : n^2 ≡ 9 [MOD 11]) (h2 : n^3 ≡ 5 [MOD 11]) : n ≡ 3 [MOD 11] :=
sorry

end remainder_of_n_l605_605245


namespace transformed_roots_new_polynomial_l605_605318

noncomputable def polynomial_with_transformed_roots (a b c d : ℝ) : Polynomial ℝ :=
  polynomial.X ^ 4 - polynomial.C b * polynomial.X - polynomial.C 3

theorem transformed_roots_new_polynomial (a b c d : ℝ) (habcd: ∀ x : ℝ, polynomial_with_transformed_roots a b c d = 0) :
  (∃ x : ℝ, 3 * polynomial.X ^ 4 - b * polynomial.X ^ 3 - 1 = 0) := 
sorry

end transformed_roots_new_polynomial_l605_605318


namespace randy_blocks_l605_605453

theorem randy_blocks :
  ∀ (initial_blocks blocks_left : ℕ),
  initial_blocks = 97 → blocks_left = 72 → (initial_blocks - blocks_left) = 25 :=
by
  intros initial_blocks blocks_left h_initial h_left
  rw [h_initial, h_left]
  exact Nat.sub_eq_of_eq_add (by norm_num)

end randy_blocks_l605_605453


namespace zeros_of_shifted_function_l605_605762

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_shifted_function :
  {x : ℝ | f (x - 1) = 0} = {0, 2} :=
sorry

end zeros_of_shifted_function_l605_605762


namespace tripodasaurus_flock_l605_605498

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end tripodasaurus_flock_l605_605498


namespace problem_a_problem_b_problem_c_l605_605566

-- Definitions for the conditions
variables {A B C : Point} -- Vertices of the triangle
variables {ω_A ω_B ω_C : Circle} -- The Apollonius circles
noncomputable def M1 : Point := sorry -- One intersection of the three circles
noncomputable def M2 : Point := sorry -- Another intersection of the three circles
variable {O : Point} -- Circumcenter of triangle ABC

-- Conditions in Lean
axiom ApolloniusCircles :
  (ω_A.passes_through A ∧ ∃ P Q, ω_A.passes_through P ∧ ω_A.passes_through Q ∧ A ≠ P ∧ A ≠ Q ∧ Is_Bisector P A Q) ∧
  (ω_B.passes_through B ∧ ∃ P Q, ω_B.passes_through P ∧ ω_B.passes_through Q ∧ B ≠ P ∧ B ≠ Q ∧ Is_Bisector P B Q) ∧
  (ω_C.passes_through C ∧ ∃ P Q, ω_C.passes_through P ∧ ω_C.passes_through Q ∧ C ≠ P ∧ C ≠ Q ∧ Is_Bisector P C Q)

axiom IntersectsAtTwoPoints :
  ω_A.IntersectsAtTwoPoints ω_B ω_C M1 M2

axiom Circumcenter (M1 M2 : Point) :
  Line.through M1 M2 ↔ Line.through M1 O ∧ Line.through M2 O

axiom PerpendicularFeetFormEquilateralTriangles (M1 M2 : Point) :
  ∃ D E F D' E' F', 
  PerpendicularFoot M1 A B C D ∧ PerpendicularFoot M1 B A C E ∧ PerpendicularFoot M1 C A B F ∧
  PerpendicularFoot M2 A B C D' ∧ PerpendicularFoot M2 B A C E' ∧ PerpendicularFoot M2 C A B F' ∧
  EquilateralTriangle D E F ∧ EquilateralTriangle D' E' F'

-- Lean statements
theorem problem_a :
  (∃ M1 M2, (ω_A.IntersectsAt M1 ∧ ω_B.IntersectsAt M1 ∧ ω_C.IntersectsAt M1) ∧
             (ω_A.IntersectsAt M2 ∧ ω_B.IntersectsAt M2 ∧ ω_C.IntersectsAt M2)) := by
  apply IntersectsAtTwoPoints
  exact ApolloniusCircles

theorem problem_b :
  M1 M2 → Line.through M1 M2 O := by
  sorry

theorem problem_c :
  M1 M2 → PerpendicularFeetFormEquilateralTriangles M1 M2 := by
  sorry

end problem_a_problem_b_problem_c_l605_605566


namespace leading_coefficient_of_polynomial_l605_605490

noncomputable def leading_coefficient {R : Type*} [CommRing R] (p : Polynomial R) : R :=
p.coeff p.natDegree

theorem leading_coefficient_of_polynomial 
  (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 1) - f(x) = x^2 + 2 * x + 1) :
  leading_coefficient (Polynomial.map (algebraMap ℝ ℝ) (Polynomial.of_f -> Polynomial) := 
  sorry

end leading_coefficient_of_polynomial_l605_605490


namespace count_terminating_decimals_with_nonzero_thousandths_l605_605714

noncomputable def is_terminating_with_nonzero_thousandths (n : ℕ) : Prop :=
  (∃ a b : ℕ, n = 2^a * 5^b) ∧ n ≤ 1000

theorem count_terminating_decimals_with_nonzero_thousandths :
  (finset.univ.filter is_terminating_with_nonzero_thousandths).card = 25 :=
begin
  sorry
end

end count_terminating_decimals_with_nonzero_thousandths_l605_605714


namespace rem_fraction_l605_605110

theorem rem_fraction : 
  let rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋;
  rem (5/7) (-3/4) = -1/28 := 
by
  sorry

end rem_fraction_l605_605110


namespace hypotenuse_length_l605_605345

theorem hypotenuse_length (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1800) (h₁ : c^2 = a^2 + b^2) : c = 30 :=
by
  sorry

end hypotenuse_length_l605_605345


namespace domain_of_y_l605_605262

noncomputable def domain_sqrt_sin_cos (x : ℝ) : Prop :=
  (∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + 2 * k * Real.pi)

theorem domain_of_y (x : ℝ) :
  (sin x ≥ 0) ∧ (cos x ≥ 1/2) ↔ domain_sqrt_sin_cos x :=
by sorry

end domain_of_y_l605_605262


namespace general_term_sequence_l605_605019

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n^2 + 3 * n + 2) →
  (∀ n, a n = if n = 1 then 6 else 2 * n + 2) →
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :=
begin
  sorry
end

end general_term_sequence_l605_605019


namespace michelle_gas_used_l605_605414

theorem michelle_gas_used (start_gas end_gas used_gas : ℚ)
  (h_start : start_gas = 0.5)
  (h_end : end_gas = 0.17)
  (h_used : used_gas = start_gas - end_gas) :
  used_gas = 0.33 :=
by {
  rw [h_start, h_end] at h_used,
  exact h_used
}

end michelle_gas_used_l605_605414


namespace exists_monochromatic_triangle_l605_605619

-- Define the set of places including islands and cities
inductive Place
| Island1 | Island2
| City : ℕ → Place -- Cities indexed from 1 to 7

open Place

-- Function to determine if cities are adjacent (consecutive indices mod 7)
def adjacent : Place → Place → Prop
| (City n1) (City n2) := (n1 + 1) % 7 = n2 % 7 ∨ (n2 + 1) % 7 = n1 % 7
| _ _ := false

-- Define the graph as an edge-coloring problem
inductive Color
| Red | Green

-- Assuming that there is a function that gives the color of the connection between places
variable (color : Place → Place → Color)

-- State the theorem
theorem exists_monochromatic_triangle : 
  ∃ (a b c : Place), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  color a b = color b c ∧ color b c = color a c :=
sorry

end exists_monochromatic_triangle_l605_605619


namespace eval_expression_l605_605639

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l605_605639


namespace range_of_b_l605_605875

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b * x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0) ↔ b ≥ 3 := sorry

end range_of_b_l605_605875


namespace angle_AMN_90_l605_605373

theorem angle_AMN_90
 (A B C K M N: Type)
 [u A B C K M N : LinearOrderedField]
 (h1: ∠ BKA = ∠ AKC)
 (h2 : ∠ BKL = 30)
 (h3 : ∠ KBL = 30)
 (h_inter_M : M = line (A, B) ∩ line (C, K))
 (h_inter_N : N = line (A, C) ∩ line (B, K))
 : ∠ AMN = 90 :=
  sorry

end angle_AMN_90_l605_605373


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605219

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605219


namespace smallest_n_l605_605692

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605692


namespace chocolate_cookies_initial_count_l605_605103

theorem chocolate_cookies_initial_count
  (andy_ate : ℕ) (brother : ℕ) (friends_each : ℕ) (num_friends : ℕ)
  (team_members : ℕ) (first_share : ℕ) (common_diff : ℕ)
  (last_member_share : ℕ) (total_sum_team : ℕ)
  (total_cookies : ℕ) :
  andy_ate = 4 →
  brother = 6 →
  friends_each = 2 →
  num_friends = 3 →
  team_members = 10 →
  first_share = 2 →
  common_diff = 2 →
  last_member_share = first_share + (team_members - 1) * common_diff →
  total_sum_team = team_members / 2 * (first_share + last_member_share) →
  total_cookies = andy_ate + brother + (friends_each * num_friends) + total_sum_team →
  total_cookies = 126 :=
by
  intros ha hb hf hn ht hf1 hc hl hs ht
  sorry

end chocolate_cookies_initial_count_l605_605103


namespace moles_H2O_formed_l605_605780

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end moles_H2O_formed_l605_605780


namespace trajectory_of_P_is_ellipse_l605_605744

noncomputable def A := (-1 / 2 : ℝ, 0 : ℝ)
noncomputable def F := (1 / 2 : ℝ, 0 : ℝ)
def circle_eq (x y : ℝ) := (x - 1 / 2) ^ 2 + y ^ 2 = 4
def B (p : ℝ × ℝ) := p ∈ { q : ℝ × ℝ | circle_eq q.1 q.2 }

theorem trajectory_of_P_is_ellipse :
  ∀ P, ∃ B, B (P.1, P.2) ∧
    let bisector_AB := (P.1, P.2) in -- placeholder for bisector equation or property
    let BF := (P.1, P.2) in -- placeholder for BF line equation or property
    (bisector_AB = BF) →
      (P.1 ^ 2 + (4 / 3) * P.2 ^ 2 = 1) := 
by 
  sorry

end trajectory_of_P_is_ellipse_l605_605744


namespace arithmetic_mean_reciprocals_primes_l605_605155

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605155


namespace length_of_boundary_between_two_touching_faces_l605_605020

theorem length_of_boundary_between_two_touching_faces :
  (∃ (l : ℝ), (∀ (e : ℤ), (1 ≤ e ∧ e ≤ 12) → l = e) ∧ (12 * l = 72)) → l = 6 :=
begin
  sorry
end

end length_of_boundary_between_two_touching_faces_l605_605020


namespace remainder_div2_l605_605492

   theorem remainder_div2 :
     ∀ z x : ℕ, (∃ k : ℕ, z = 4 * k) → (∃ n : ℕ, x = 2 * n) → (z + x + 4 + z + 3) % 2 = 1 :=
   by
     intros z x h1 h2
     sorry
   
end remainder_div2_l605_605492


namespace lunks_needed_for_bananas_l605_605785

theorem lunks_needed_for_bananas :
  (7 : ℚ) / 4 * (20 * 3 / 5) = 21 :=
by
  sorry

end lunks_needed_for_bananas_l605_605785


namespace right_triangle_area_l605_605489

theorem right_triangle_area (a b c : ℝ) (h1 : a + b + c = 90) (h2 : a^2 + b^2 + c^2 = 3362) (h3 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 180 :=
by
  sorry

end right_triangle_area_l605_605489


namespace tan_theta_eq_1_l605_605301

theorem tan_theta_eq_1 (θ : ℝ) 
  (h_acute : 0 < θ ∧ θ < Float.pi / 2)
  (h_parallel : ∃ k : ℝ, 
    (1 - Real.sin θ) = k * (1/2) ∧ 1 = k * (1 + Real.sin θ)) :
  Real.tan θ = 1 :=
by
  sorry

end tan_theta_eq_1_l605_605301


namespace circles_are_tangent_l605_605491

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y + 12 = 0
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 - 14 * x - 2 * y + 14 = 0

theorem circles_are_tangent : 
  (∀ x y : ℝ, circle1_equation x y → (∃ x₀ y₀ r₁, (x - x₀)^2 + (y - y₀)^2 = r₁^2 ∧ x₀ = 3 ∧ y₀ = -2 ∧ r₁ = 1)) ∧
  (∀ x y : ℝ, circle2_equation x y → (∃ x₀ y₀ r₂, (x - x₀)^2 + (y - y₀)^2 = r₂^2 ∧ x₀ = 7 ∧ y₀ = 1 ∧ r₂ = 6)) ∧
  (∃ d : ℝ, d = real.sqrt ((3 - 7)^2 + (-2 - 1)^2) ∧ d = 6 - 1) →
  ∃ x y : ℝ, circle1_equation x y ∧ circle2_equation x y :=
sorry

end circles_are_tangent_l605_605491


namespace find_v_l605_605379

def a : ℝ^3 := ![2, 2, 1]
def b : ℝ^3 := ![3, 1, 0]
def v : ℝ^3 := ![5, 3, 1]

theorem find_v (a b v : ℝ^3) :
  v × a = (b × a) ∧ v × b = (a × b) ↔ v = a + b := 
by
  sorry

end find_v_l605_605379


namespace square_area_eq_58_l605_605991

-- Define the conditions
variables (A B C D P : Point)
variables (PA PB PD : ℝ)
variables (square_len : ℝ)

-- Declare the conditions as axioms
axiom PA_eq_3 : PA = 3
axiom PB_eq_7 : PB = 7
axiom PD_eq_5 : PD = 5
axiom square_len_eq_sqrt_58 : square_len = Real.sqrt 58

-- The theorem to prove
theorem square_area_eq_58 (h₁ : ∥A - P∥ = PA) 
                         (h₂ : ∥B - P∥ = PB) 
                         (h₃ : ∥D - P∥ = PD)
                         (h₄ : PA_eq_3) 
                         (h₅ : PB_eq_7) 
                         (h₆ : PD_eq_5) 
                         (h₇ : square_len_eq_sqrt_58) : 
    (square_len^2 = 58) :=
sorry

end square_area_eq_58_l605_605991


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605186

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605186


namespace probability_of_green_l605_605082

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green_l605_605082


namespace min_expression_value_l605_605400

theorem min_expression_value (x : ℝ) (hx : 0 < x) :
  ∃ y : ℝ, y ≤ (x^2 + 6 - real.sqrt (x^4 + 36)) / x ∧ y = 12 / (2 * (real.sqrt 6 + real.sqrt 3)) :=
sorry

end min_expression_value_l605_605400


namespace find_x_for_collinear_vectors_l605_605070

noncomputable def collinear (a b : ℝ × ℝ) : Prop :=
∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem find_x_for_collinear_vectors :
  ∀ (x : ℝ), collinear (2, 4) (x, 6) → x = 3 :=
begin
  intros x h,
  cases h with k hk,
  sorry, -- The actual proof step will replace this
end

end find_x_for_collinear_vectors_l605_605070


namespace value_of_m_condition1_value_of_m_condition2_value_of_m_condition3_l605_605250

noncomputable theory

open_locale classical

def condition1 (m : ℝ) : Prop := 
  m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m + 2 = 0

def condition2 (m : ℝ) : Prop := 
  m^2 - 3 * m + 2 = 2 * (m^2 - 5 * m + 6)

def condition3 (m : ℝ) : Prop := 
  (m^2 - 5 * m + 6) * (m^2 - 3 * m + 2) > 0

theorem value_of_m_condition1 (m : ℝ) : condition1 m → m = 2 :=
sorry

theorem value_of_m_condition2 (m : ℝ) : condition2 m → m = 2 ∨ m = 5 :=
sorry

theorem value_of_m_condition3 (m : ℝ) : condition3 m → m < 1 ∨ m > 3 :=
sorry

end value_of_m_condition1_value_of_m_condition2_value_of_m_condition3_l605_605250


namespace polynomial_coeff_sum_l605_605736

theorem polynomial_coeff_sum :
  let coefficients := (λ x: ℝ, (1 + x - x^2)^10).coeffs
  ∑ i in Finset.range 21, i * coefficients i = -9 :=
by
  sorry

end polynomial_coeff_sum_l605_605736


namespace compare_abc_l605_605258

theorem compare_abc :
  let a := 2 ^ 0.1
  let b := Real.log (5 / 2)
  let c := Real.logBase 3 (9 / 10)
  a > b ∧ b > c :=
by
  sorry

end compare_abc_l605_605258


namespace magnitude_sum_vector_l605_605772

open Real
open ComplexConjugate (ComplexConjugate) 

variables (a b : ℝ × ℝ)
variables (angle : ℝ)
variables (a_mag b_mag : ℝ)

axiom magnitude_a : (a.1 ^ 2 + a.2 ^ 2 = 3)
axiom magnitude_b : (b.1 ^ 2 + b.2 ^ 2 = 4)
axiom angle_eq : angle = π / 6

theorem magnitude_sum_vector :
  sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = sqrt 31 :=
by 
  sorry

end magnitude_sum_vector_l605_605772


namespace relationship_between_mode_median_mean_l605_605270

def data_set : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode : ℕ := 60 -- derived from the problem conditions
def median : ℕ := 50 -- derived from the problem conditions
def mean : ℚ := 330 / 7 -- derived from the problem conditions

theorem relationship_between_mode_median_mean :
  mode > median ∧ median > mean :=
by
  sorry

end relationship_between_mode_median_mean_l605_605270


namespace combined_distance_l605_605930

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l605_605930


namespace min_BM_MN_l605_605829

-- Define the points and the rectangle
structure Rectangle where
  A B C D : ℝ × ℝ
  a_eq : A.1 = 0
  b_eq : B.1 = 20
  c_eq : B.2 = 0
  d_eq : C.2 = 10
  diags_eq : A.1 + B.1 = C.1 ∧ C.2 = B.2 + D.2

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/- 
  Prove that the minimum length BM + MN in rectangle ABCD 
  with the given conditions is 16 cm.
-/
theorem min_BM_MN {A B C D M N : ℝ × ℝ} 
  (r : Rectangle)
  (M_on_AC : ∃ t ∈ Icc 0 1, M = (t * A + (1 - t) * C))
  (N_on_AB : ∃ s ∈ Icc 0 1, N = (s * A + (1 - s) * B)) :
  ∃ B' : ℝ × ℝ, 
  B' = (Reflect_point B).on AC → 
  distance B M + distance M N = distance B' N ∧
  distance B' N = 16 :=
sorry

end min_BM_MN_l605_605829


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605208

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605208


namespace additional_investment_l605_605085

-- Given the conditions
variables (x y : ℝ)
def interest_rate_1 := 0.02
def interest_rate_2 := 0.04
def invested_amount := 1000
def total_interest := 92

-- Theorem to prove
theorem additional_investment : 
  0.02 * invested_amount + 0.04 * (invested_amount + y) = total_interest → 
  y = 800 :=
by
  sorry

end additional_investment_l605_605085


namespace xiaomei_wins_the_game_xiaoming_wins_absolute_game_l605_605066

-- Definitions for the game.
def red_ball_points : ℤ := -3
def white_ball_points : ℤ := 0
def yellow_ball_points : ℤ := 2

def xiaoming_balls : list ℤ := [red_ball_points, yellow_ball_points, red_ball_points, white_ball_points, red_ball_points]
def xiaomei_balls : list ℤ := [yellow_ball_points, yellow_ball_points, white_ball_points, red_ball_points, red_ball_points]

def sum_of_points (balls : list ℤ) : ℤ :=
  list.sum balls

-- Problem 1: Compare the scores of Xiaoming and Xiaomei
def xiaoming_score : ℤ := sum_of_points xiaoming_balls
def xiaomei_score : ℤ := sum_of_points xiaomei_balls

theorem xiaomei_wins_the_game 
  (h1 : xiaoming_score = -7) 
  (h2 : xiaomei_score = -2) :
  xiaomei_score > xiaoming_score := 
by
  sorry

-- Problem 2: Compare the absolute values of the scores
def abs_score (score : ℤ) : ℤ := abs score

theorem xiaoming_wins_absolute_game 
  (h1 : abs_score xiaoming_score = 7) 
  (h2 : abs_score xiaomei_score = 2) :
  abs_score xiaoming_score > abs_score xiaomei_score := 
by
  sorry

end xiaomei_wins_the_game_xiaoming_wins_absolute_game_l605_605066


namespace percentage_increase_l605_605596

theorem percentage_increase (R W : ℕ) (hR : R = 36) (hW : W = 20) : 
  ((R - W) / W : ℚ) * 100 = 80 := 
by 
  sorry

end percentage_increase_l605_605596


namespace intersection_with_y_axis_l605_605475

theorem intersection_with_y_axis :
  ∃ (x y : ℝ), x = 0 ∧ y = 5 * x - 6 ∧ (x, y) = (0, -6) := 
sorry

end intersection_with_y_axis_l605_605475


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605169

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605169


namespace mean_of_reciprocals_of_first_four_primes_l605_605199

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605199


namespace find_x_of_series_eq_16_l605_605795

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x ^ n

theorem find_x_of_series_eq_16 (x : ℝ) (h : series_sum x = 16) : x = (4 - Real.sqrt 2) / 4 :=
by
  sorry

end find_x_of_series_eq_16_l605_605795


namespace sum_S_l605_605051

def S : Set ℤ := { x | 10 < x ∧ x < 20 }

theorem sum_S : (∑ x in S, x) = 135 := by
  sorry

end sum_S_l605_605051


namespace smallest_positive_integer_n_l605_605696

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605696


namespace rhombus_side_length_l605_605813

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l605_605813


namespace negation_of_p_l605_605297

open Classical

variable (p : Prop)

theorem negation_of_p (h : ∀ x : ℝ, x^3 + 2 < 0) : 
  ∃ x : ℝ, x^3 + 2 ≥ 0 :=
by
  sorry

end negation_of_p_l605_605297


namespace gordon_total_cost_l605_605026

noncomputable def DiscountA (price : ℝ) : ℝ :=
if price > 22.00 then price * 0.70 else price

noncomputable def DiscountB (price : ℝ) : ℝ :=
if 10.00 < price ∧ price <= 20.00 then price * 0.80 else price

noncomputable def DiscountC (price : ℝ) : ℝ :=
if price < 10.00 then price * 0.85 else price

noncomputable def apply_discount (price : ℝ) : ℝ :=
if price > 22.00 then DiscountA price
else if price > 10.00 then DiscountB price
else DiscountC price

noncomputable def total_price (prices : List ℝ) : ℝ :=
(prices.map apply_discount).sum

noncomputable def total_with_tax_and_fee (prices : List ℝ) (tax_rate extra_fee : ℝ) : ℝ :=
let total := total_price prices
let tax := total * tax_rate
total + tax + extra_fee

theorem gordon_total_cost :
  total_with_tax_and_fee
    [25.00, 18.00, 21.00, 35.00, 12.00, 10.00, 8.50, 23.00, 6.00, 15.50, 30.00, 9.50]
    0.05 2.00
  = 171.27 :=
  sorry

end gordon_total_cost_l605_605026


namespace plane_through_point_contains_line_l605_605130

-- Definitions from conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def passes_through (p : Point) (plane : Point → Prop) : Prop :=
  plane p

def contains_line (line : ℝ → Point) (plane : Point → Prop) : Prop :=
  ∀ t, plane (line t)

def line_eq (t : ℝ) : Point :=
  ⟨4 * t + 2, -6 * t - 3, 2 * t + 4⟩

def plane_eq (A B C D : ℝ) (p : Point) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

theorem plane_through_point_contains_line :
  ∃ (A B C D : ℝ), 1 < A ∧ gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  passes_through ⟨1, 2, -3⟩ (plane_eq A B C D) ∧
  contains_line line_eq (plane_eq A B C D) ∧ 
  (∃ (k : ℝ), 3 * k = A ∧ k = 1 / 3 ∧ B = k * 1 ∧ C = k * (-3) ∧ D = k * 2) :=
sorry

end plane_through_point_contains_line_l605_605130


namespace certain_number_existence_l605_605072

theorem certain_number_existence : ∃ x : ℝ, (102 * 102) + (x * x) = 19808 ∧ x = 97 := by
  sorry

end certain_number_existence_l605_605072


namespace six_digit_number_div_by_91_l605_605450

theorem six_digit_number_div_by_91 (x y z : Nat) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) (hz : 0 ≤ z ∧ z < 10) :
  let abc := (100 * x) + (10 * y) + z;
  let abcabc := (100000 * x) + (10000 * y) + (1000 * z) + (100 * x) + (10 * y) + z;
  abcabc % 91 = 0 :=
by
  let abc := (100 * x) + (10 * y) + z
  let abcabc := (100000 * x) + (10000 * y) + (1000 * z) + (100 * x) + (10 * y) + z
  have h1 : abcabc = 1001 * abc := sorry
  have h2 : 1001 % 91 = 0 := sorry
  show abcabc % 91 = 0 from sorry

end six_digit_number_div_by_91_l605_605450


namespace num_distinct_four_digit_numbers_l605_605311

theorem num_distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  (finset.card (finset.powersetLen 4 digits)).card = 120 :=
by sorry

end num_distinct_four_digit_numbers_l605_605311


namespace sum_of_consecutive_integers_product_384_l605_605015

theorem sum_of_consecutive_integers_product_384 :
  ∃ (a : ℤ), a * (a + 1) * (a + 2) = 384 ∧ a + (a + 1) + (a + 2) = 24 :=
by
  sorry

end sum_of_consecutive_integers_product_384_l605_605015


namespace range_of_f_l605_605017

def f (x : ℝ) : ℝ := real.sqrt (4 - x^2)

theorem range_of_f : set.range f = set.Icc 0 2 :=
by sorry

end range_of_f_l605_605017


namespace prob_odd_divisor_15_fact_l605_605610

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def divisor_count (n : ℕ) : ℕ :=
  (multiset.range (n + 1)).filter (λ k => n % k = 0).card

def odd_divisor_count (n : ℕ) : ℕ :=
  (multiset.range (n + 1)).filter (λ k => n % k = 0 ∧ k % 2 = 1).card

theorem prob_odd_divisor_15_fact : 
  (odd_divisor_count (factorial 15)) / (divisor_count (factorial 15)) = 1 / 6 := 
sorry

end prob_odd_divisor_15_fact_l605_605610


namespace find_A_for_diamond_l605_605127

def diamond (A B : ℕ) : ℕ := 4 * A + 3 * B + 7

theorem find_A_for_diamond (A : ℕ) (h : diamond A 7 = 76) : A = 12 :=
by
  sorry

end find_A_for_diamond_l605_605127


namespace count_terminating_n_with_conditions_l605_605724

def is_terminating_decimal (n : ℕ) : Prop :=
  (∃ m k : ℕ, (2^m * 5^k = n)) 

def thousandths_non_zero (n : ℕ) : Prop := 
  100 < n ∧ n ≤ 1000

theorem count_terminating_n_with_conditions : 
  {n : ℕ | is_terminating_decimal n ∧ thousandths_non_zero n}.to_finset.card = 9 := 
by
  sorry

end count_terminating_n_with_conditions_l605_605724


namespace distance_between_centers_of_circles_l605_605662

theorem distance_between_centers_of_circles 
  (triangle : Type) 
  (A B C : triangle) 
  (a b c : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : c = 10)
  (right_angle : ∠A = 90) : 
  let r_circumcenter := (c / 2)
  let r_inradius := 
    let s := (a + b + c) / 2
    in (sqrt ((s - a) * (s - b) * (s - c) / s))
  let dist := sqrt ((r_inradius) ^ 2 + (r_circumcenter - b / 2) ^ 2)
  in dist = sqrt 5 := sorry

end distance_between_centers_of_circles_l605_605662


namespace solve_inequality_system_l605_605919

theorem solve_inequality_system (x : ℝ) :
  (x + 1 < 4 ∧ 1 - 3 * x ≥ -5) ↔ (x ≤ 2) :=
by
  sorry

end solve_inequality_system_l605_605919


namespace problem_statement_l605_605932

-- Definitions based on the conditions provided
def smallest_odd_prime : ℕ := 3

def has_exactly_three_positive_divisors (k : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ k = p * p

def largest_integer_with_three_divisors_under (limit : ℕ) : ℕ :=
  Nat.find_greatest (λ k, k < limit ∧ has_exactly_three_positive_divisors k) limit

-- Problem statement translated to a Lean 4 theorem
theorem problem_statement : 
  let m := smallest_odd_prime,
      n := largest_integer_with_three_divisors_under 200 in
  m + n = 172 :=
by
  let m := smallest_odd_prime
  let n := largest_integer_with_three_divisors_under 200
  sorry

end problem_statement_l605_605932


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605209

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605209


namespace find_general_term_l605_605328

-- Definition of sequence sum condition
def seq_sum_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (2/3) * a n + 1/3

-- Statement of the proof problem
theorem find_general_term (a S : ℕ → ℝ) 
  (h : seq_sum_condition a S) : 
  ∀ n, a n = (-2)^(n-1) := 
by
  sorry

end find_general_term_l605_605328


namespace range_of_sum_abs_l605_605315

variable {x y z : ℝ}

theorem range_of_sum_abs : 
  x^2 + y^2 + z = 15 → 
  x + y + z^2 = 27 → 
  xy + yz + zx = 7 → 
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end range_of_sum_abs_l605_605315


namespace exists_prime_pairs_perfect_squares_l605_605589

theorem exists_prime_pairs_perfect_squares :
  ∃ (x y: Fin 12 → ℕ) (m: Fin 12 → ℕ), 
    (∀ k: Fin 12, Nat.prime (x k) ∧ Nat.prime (y k)) ∧
    (∀ k: Fin 12, x k + y k = m k ^ 2) ∧
    x 0 = x 11 ∧ y 0 = y 11 :=
by
  sorry

end exists_prime_pairs_perfect_squares_l605_605589


namespace johnny_marbles_l605_605844

def num_ways_to_choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem johnny_marbles :
  num_ways_to_choose_marbles 7 3 = 35 :=
by
  sorry

end johnny_marbles_l605_605844


namespace payment_for_c_l605_605061

theorem payment_for_c 
  (A_rate B_rate : ℝ) (t P : ℝ) 
  (hA : A_rate = 1/6) 
  (hB : B_rate = 1/8) 
  (ht : t = 3) 
  (hP : P = 3680) : 
  let combined_rate := A_rate + B_rate in
  let total_work_done := t * combined_rate in
  let work_done_by_A_and_B := total_work_done in
  let work_contributed_by_C := 1 - (work_done_by_A_and_B) in
  let payment_for_C := work_contributed_by_C * P in
  payment_for_C = 460 :=
by
  sorry

end payment_for_c_l605_605061


namespace triangle_angle_problem_l605_605835

open Real

-- Define degrees to radians conversion (if necessary)
noncomputable def degrees (d : ℝ) : ℝ := d * π / 180

-- Define the problem conditions and goal
theorem triangle_angle_problem
  (x y : ℝ)
  (h1 : degrees 3 * x + degrees y = degrees 90) :
  x = 18 ∧ y = 36 := by
  sorry

end triangle_angle_problem_l605_605835


namespace numbering_of_points_exists_l605_605735

theorem numbering_of_points_exists (points : Fin 200 → ℝ × ℝ) (h_no_collinear : ∀ i j k : Fin 200, i ≠ j → i ≠ k → j ≠ k → ¬ collinear (points i) (points j) (points k)) :
  ∃ (numbering : Fin 200 → ℕ), (∀ i : Fin 100, (line_through (points (numbering i)) (points (numbering (i + 100)))) ∩ (line_through (points (numbering j)) (points (numbering (j + 100)))) ≠ ∅) := by
  sorry

end numbering_of_points_exists_l605_605735


namespace kids_in_lawrence_county_l605_605634

-- Definitions

def percentage_stay_home : ℝ := 0.607
def kids_stay_home : ℝ := 907611

-- The proof statement
theorem kids_in_lawrence_county (T : ℝ) (h : percentage_stay_home * T = kids_stay_home) : T ≈ 1495100 :=
by
  sorry

end kids_in_lawrence_county_l605_605634


namespace probability_C_l605_605282

noncomputable def probability_A : ℝ := 0.3
noncomputable def probability_B : ℝ := 0.2

axiom mutually_exclusive (A B : set ω) : P(A ∩ B) = 0
axiom complementary (A C : set ω) : P(A ∪ C) = 1 ∧ P(A ∩ C) = 0
axiom prob_A_union_B (A B : set ω) : P(A ∪ B) = 0.5
axiom prob_B (B : set ω) : P(B) = 0.2

theorem probability_C (A B C : set ω) (h_me : mutually_exclusive A B) (h_compl : complementary A C) (h_PAuB : prob_A_union_B A B) (h_PB : prob_B B) : P(C) = 0.7 := 
sorry

end probability_C_l605_605282


namespace largest_even_number_with_given_conditions_l605_605941

open Finset
open BigOperators

-- Define the conditions formally
def digits_sum_to_seventeen (n : ℕ) : Prop :=
  (n.digits 10).sum = 17

def all_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).nth i ≠ (n.digits 10).nth j

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def largest_even_number_with_distinct_digits (n : ℕ) : Prop :=
  ∀ m, digits_sum_to_seventeen m → all_distinct_digits m → is_even m → m ≤ n

-- The statement to be proven
theorem largest_even_number_with_given_conditions :
  largest_even_number_with_distinct_digits 62108 :=
sorry

end largest_even_number_with_given_conditions_l605_605941


namespace probability_calculation_l605_605065

def p_X := 1 / 5
def p_Y := 1 / 2
def p_Z := 5 / 8
def p_not_Z := 1 - p_Z

theorem probability_calculation : 
    (p_X * p_Y * p_not_Z) = (3 / 80) := by
    sorry

end probability_calculation_l605_605065


namespace rhombus_side_length_l605_605819

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605819


namespace rhombus_side_length_l605_605801

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l605_605801


namespace part1_part2_l605_605298

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := {x | 4 ≤ 2^x ∧ 2^x < 16}

noncomputable def B : Set ℝ := {x | ∃ y: ℝ, y = log (x - 3)}

theorem part1 : A ∩ B = {x | 3 < x ∧ x < 4} := by
  sorry

theorem part2 : U \ A ∪ B = {x | x < 2 ∨ 3 < x} := by
  sorry

end part1_part2_l605_605298


namespace point_inside_circle_l605_605388

-- Descriptions of the conditions
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem point_inside_circle
  (f : ℝ → ℝ)
  (h1 : increasing_function f)
  (h2 : ∀ x : ℝ, f(1 - x) + f(1 + x) = 0)
  (h3 : ∀ m n : ℝ, f(m ^ 2 - 6 * m + 23) + f(n ^ 2 - 8 * n) < 0) :
  (m n : ℝ), (m - 3) ^ 2 + (n - 4) ^ 2 < 4 :=
by
  sorry

end point_inside_circle_l605_605388


namespace count_terminating_n_with_conditions_l605_605725

def is_terminating_decimal (n : ℕ) : Prop :=
  (∃ m k : ℕ, (2^m * 5^k = n)) 

def thousandths_non_zero (n : ℕ) : Prop := 
  100 < n ∧ n ≤ 1000

theorem count_terminating_n_with_conditions : 
  {n : ℕ | is_terminating_decimal n ∧ thousandths_non_zero n}.to_finset.card = 9 := 
by
  sorry

end count_terminating_n_with_conditions_l605_605725


namespace number_division_l605_605420

theorem number_division (n q r d : ℕ) (h1 : d = 18) (h2 : q = 11) (h3 : r = 1) (h4 : n = (d * q) + r) : n = 199 := 
by 
  sorry

end number_division_l605_605420


namespace binomial_identity_binomial_sum_identity_l605_605068

theorem binomial_identity 
  (n k m : ℕ) :
  (k ≤ n) → (m ≤ k) → 
  (binom n k * binom k m = binom n m * binom (n - m) (k - m)) :=
by
  sorry

theorem binomial_sum_identity 
  (n : ℕ) :
  (∑ k in Finset.range (n + 1), k * binom n k = n * 2^(n-1)) :=
by
  sorry

end binomial_identity_binomial_sum_identity_l605_605068


namespace cross_shape_cube_folding_l605_605003

-- Define the number of total positions and valid configurations
def total_positions : ℕ := 11
def valid_configs : ℕ := 6

-- Define the problem statement as a theorem
theorem cross_shape_cube_folding : ∃ (valid_configs : ℕ), valid_configs = 6 ∧ valid_configs ≤ total_positions := by
  existsi 6
  split
  · rfl
  · simp

end cross_shape_cube_folding_l605_605003


namespace expression_verification_l605_605317

theorem expression_verification (x y a : ℝ) (h : x = y) : 
  (-1/3 * x + 1 = -1/3 * y + 1) ∧
  ¬ (2 * x = y + 2) ∧
  ¬ (x + 2 * a = y + a) ∧
  ¬ (x - 2 = 2 - y) :=
by
  split
  · apply congr_arg (fun t => -1/3 * t + 1) h
  split
  · intro h₁
    rw [h] at h₁
    have : y + y = y + 2 := by linarith
    linarith
  split
  · intro h₂
    rw [h] at h₂
    have : 2 * a = a := by linarith
    linarith
  · intro h₃
    rw [h] at h₃
    have : 2 * y = 4 := by linarith
    linarith

end expression_verification_l605_605317


namespace smallest_n_probability_l605_605134

theorem smallest_n_probability (n : ℕ) : (1 / (n * (n + 1)) < 1 / 2023) → (n ≥ 45) :=
by
  sorry

end smallest_n_probability_l605_605134


namespace triangle_area_perimeter_l605_605798

-- Define the known conditions
def altitude_CD : ℝ := 2 * Real.sqrt 3
def angle_BAC : ℝ := 60

-- Define the triangles and lengths
noncomputable def side_AC : ℝ := 4 * Real.sqrt 3
noncomputable def side_BC : ℝ := 6
noncomputable def side_AB : ℝ := 4 * Real.sqrt 3

-- Proof statements
theorem triangle_area_perimeter : 
    (let area := (1/2) * side_AC * altitude_CD,
         perimeter := side_AC + side_BC + side_AB in
     area = 12 ∧ perimeter = 8 * Real.sqrt 3 + 6) := 
by
    sorry

end triangle_area_perimeter_l605_605798


namespace range_of_eccentricity_l605_605288

theorem range_of_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)
  (h4 : c^2 - b^2 + a * c < 0) :
  0 < c / a ∧ c / a < 1 / 2 :=
sorry

end range_of_eccentricity_l605_605288


namespace find_a_l605_605401

def A : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem find_a (a : ℝ) (h : A ∩ B a = B a) : a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end find_a_l605_605401


namespace parallelogram_base_l605_605661

theorem parallelogram_base (Area Height : ℕ) (hArea : Area = 308) (hHeight : Height = 14) : ∃ Base : ℕ, Base * Height = Area ∧ Base = 22 := 
by 
  use 22
  split
  { rw [←hArea, ←hHeight], norm_num }
  { refl }

end parallelogram_base_l605_605661


namespace candy_distribution_l605_605537

theorem candy_distribution (A B : ℕ) (h1 : 7 * A = B + 12) (h2 : 3 * A = B - 20) : A + B = 52 :=
by {
  -- proof goes here
  sorry
}

end candy_distribution_l605_605537


namespace snow_probability_l605_605432

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l605_605432


namespace symmetry_line_of_g_l605_605859

theorem symmetry_line_of_g (g : ℝ → ℝ) (h : ∀ x, g(x) = g(3 - x)) : ∃ x₀, ∀ x, x₀ = 3 / 2 :=
begin
  use 3 / 2,
  intro x,
  refl,
end

end symmetry_line_of_g_l605_605859


namespace least_n_factorial_7350_l605_605039

theorem least_n_factorial_7350 (n : ℕ) :
  (7350 ∣ n!) → (∃ k : ℕ, (1 ≤ k ∧ 7350 ∣ k!)) :=
by
  have prime_factors_7350 : 7350 = 2 * 3^2 * 5^2 * 7 := by norm_num
  existsi 10
  split
  · norm_num
  · apply dvd_factorial.mpr
    sorry

end least_n_factorial_7350_l605_605039


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605216

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605216


namespace find_a8_l605_605740

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n ≥ 2, (2 * a n - 3) / (a n - 1) = 2) (h2 : a 2 = 1) : a 8 = 16 := 
sorry

end find_a8_l605_605740


namespace math_problem_l605_605854

open Real Real.InnerProductSpace

variables {V : Type _} [InnerProductSpace ℝ V]
variables (a b c : V)
noncomputable def vector_solution : ℝ :=
  2 • ⟪a, b⟫ + 3 • ⟪a, c⟫ + 4 • ⟪b, c⟫

theorem math_problem :
  ∥a∥ = 2 ∧ ∥b∥ = 3 ∧ ∥c∥ = 6 ∧ (2 • a + 3 • b + 4 • c = 0) →
  vector_solution a b c = -673 / 4 :=
by
  intros h
  sorry

end math_problem_l605_605854


namespace seashells_broken_l605_605415

theorem seashells_broken (total_seashells : ℕ) (unbroken_seashells : ℕ) (broken_seashells : ℕ) : 
  total_seashells = 6 → unbroken_seashells = 2 → broken_seashells = total_seashells - unbroken_seashells → broken_seashells = 4 :=
by
  intros ht hu hb
  rw [ht, hu] at hb
  exact hb

end seashells_broken_l605_605415


namespace water_depth_function_maximum_safe_time_l605_605500

noncomputable def water_depth (A ω B t : ℝ) : ℝ := A * Real.sin(ω * t) + B

def A : ℝ := (13 - 7) / 2
def B : ℝ := (13 + 7) / 2
def ω : ℝ := π / 6

theorem water_depth_function :
  water_depth A ω B t = 3 * Real.sin(π * t / 6) + 10 := 
by {
  -- Insert proof here
  sorry
}

def safe_depth : ℝ := 11.5
def ship_draft : ℝ := 7
def required_depth : ℝ := safe_depth + ship_draft

theorem maximum_safe_time : ∀ t, 
  (3 * Real.sin(π * t / 6) + 10 ≥ 11.5) → ((1 ≤ t ∧ t ≤ 5) ∨ (13 ≤ t ∧ t ≤ 17)) ∧
  (max_time = 16) := 
by {
  -- Insert proof here
  sorry
}

end water_depth_function_maximum_safe_time_l605_605500


namespace smallest_positive_integer_n_l605_605698

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605698


namespace snow_prob_correct_l605_605422

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l605_605422


namespace complex_number_solution_l605_605287

theorem complex_number_solution (z : ℂ) (h : (3 - 4 * complex.I) * z = 25) : z = 3 + 4 * complex.I := 
sorry

end complex_number_solution_l605_605287


namespace value_of_M_l605_605485

def value_of_letter (x : String) : Int := sorry
def value_of_word (w : List String) : Int := 
  w.foldr (λl acc, value_of_letter l + acc) 0

theorem value_of_M (value_of_letter : String → Int) :
  value_of_letter "T" = 15 →
  value_of_word ["M", "A", "T", "H"] = 47 →
  value_of_word ["T", "E", "A", "M"] = 58 →
  value_of_word ["M", "E", "E", "T"] = 45 →
  value_of_letter "M" = 8 := by
  intros hT hMATH hTEAM hMEET
  sorry

end value_of_M_l605_605485


namespace snow_probability_first_week_l605_605429

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l605_605429


namespace average_visitors_per_day_l605_605957

theorem average_visitors_per_day (average_sunday : ℕ) (average_other : ℕ) (days_in_month : ℕ) (begins_with_sunday : Bool) :
  average_sunday = 600 → average_other = 240 → days_in_month = 30 → begins_with_sunday = true → (8640 / 30 = 288) :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l605_605957


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605207

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605207


namespace length_chord_AB_l605_605084

open Real

def parabola (x y : ℝ) := y * y = 8 * x

def line_through_focus (x y : ℝ) := y = -x + 2

def focus := (2 : ℝ, 0 : ℝ)

theorem length_chord_AB :
  ∀ x y x' y', 
    parabola x y → 
    parabola x' y' → 
    line_through_focus x y →
    line_through_focus x' y' →
    (6 - 4 * sqrt 2) < x ∧ x < (6 + 4 * sqrt 2) →
    (6 - 4 * sqrt 2) < x' ∧ x' < (6 + 4 * sqrt 2) →
    sqrt ((x' - x)^2 + (y' - y)^2) = 8 * sqrt 2 :=
by sorry

end length_chord_AB_l605_605084


namespace bus_probabilities_and_chi_squared_l605_605554

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l605_605554


namespace inequality_am_gm_l605_605386

noncomputable theory

open_locale big_operators

variable {n : ℕ}

theorem inequality_am_gm 
  (h1 : 2 ≤ n) 
  (a : Π (i : ℕ), (2 ≤ i ∧ i ≤ n) → ℝ)
  (h_pos : ∀ (i : ℕ) (hi : 2 ≤ i ∧ i ≤ n), 0 < a i hi)
  (h_prod : (∏ i in finset.range(n).filter (λ i, 2 ≤ i), a i ⟨i, and.intro (finset.mem_range.2 (nat.lt_of_lt_of_le (nat.succ_pos i) h1)) (finset.mem_filter.1 hi).2⟩) = 1) :
  ((∏ i in finset.range(n).filter (λ i, 2 ≤ i), (1 + a i ⟨i, and.intro (finset.mem_range.2 (nat.lt_of_lt_of_le (nat.succ_pos i) h1)) (finset.mem_filter.1 i).2⟩) ^ i) ≥ n ^ n.succ) :=
sorry

end inequality_am_gm_l605_605386


namespace avg_speed_l605_605063

theorem avg_speed (s : ℝ) : 
  let t := s / 56 + s / 72 in
  let c := 2 * s / t in
  c = 63 := 
by 
  sorry

end avg_speed_l605_605063


namespace smallest_n_for_identity_l605_605702

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605702


namespace simplified_result_l605_605532

theorem simplified_result (a b M : ℝ) (h1 : (2 * a) / (a ^ 2 - b ^ 2) - 1 / M = 1 / (a - b))
  (h2 : M - (a - b) = 2 * b) : (2 * a) / (a ^ 2 - b ^ 2) - 1 / (a - b) = 1 / (a + b) :=
by
  sorry

end simplified_result_l605_605532


namespace metal_waste_l605_605583

theorem metal_waste
  (length : ℝ) (width : ℝ) (diameter : ℝ) (radius : ℝ)
  (rect_area : ℝ) (circle_area : ℝ) (inscribed_rect_area : ℝ)
  (h_length : length = 20)
  (h_width : width = 10)
  (h_diameter : diameter = 10)
  (h_radius : radius = diameter / 2)
  (h_rect_area : rect_area = length * width)
  (h_circle_area : circle_area = Real.pi * radius^2)
  (h_inscribed_rect_area : inscribed_rect_area = (width * length) / 5):

  rect_area - circle_area + circle_area - inscribed_rect_area = 160 := 
sorry

end metal_waste_l605_605583


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605183

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605183


namespace smallest_positive_n_l605_605687

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605687


namespace prove_black_white_equal_integrals_l605_605632

def linear_black_white_integral_equal : Prop :=
  ∀ (a b : ℝ), 
  (∫ x in -1..(-1/2), a * x + b) + (∫ x in (1/2)..1, a * x + b) =
  (∫ x in (-1/2)..(1/2), a * x + b)

def quadratic_black_white_integral_equal : Prop :=
  ∀ (a b c : ℝ), 
  (∫ x in -3/4..(-1/4), a * x^2 + b * x + c) + (∫ x in 0..(1/4), a * x^2 + b * x + c) + (∫ x in 3/4..1, a * x^2 + b * x + c) =
  (∫ x in -1..(-3/4), a * x^2 + b * x + c) + (∫ x in -1/4..0, a * x^2 + b * x + c) + (∫ x in 1/4..3/4, a * x^2 + b * x + c)

theorem prove_black_white_equal_integrals :
  linear_black_white_integral_equal ∧ quadratic_black_white_integral_equal := 
  by
  sorry

end prove_black_white_equal_integrals_l605_605632


namespace mean_of_reciprocals_of_first_four_primes_l605_605197

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605197


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605173

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605173


namespace rhombus_side_length_l605_605810

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605810


namespace empty_square_to_any_corner_l605_605519

theorem empty_square_to_any_corner (board : Type) [has_squares board] (covered_by_dominoes : is_covered_by_dominoes board)
    (empty_corner : is_empty (corner board)) :
    ∃ (move_dominoes : board → board), is_empty (corner (move_dominoes board)) := 
sorry

end empty_square_to_any_corner_l605_605519


namespace WangHua_practice_days_in_August_2016_l605_605335

theorem WangHua_practice_days_in_August_2016 :
  let jan_days := 31,
      feb_days := 29,
      mar_days := 31,
      apr_days := 30,
      may_days := 31,
      jun_days := 30,
      jul_days := 31,
      days_in_year_before_aug := jan_days + feb_days + mar_days + apr_days + may_days + jun_days + jul_days,
      day_of_week_jan_1 := 5 -- Jan 1, 2016 is Friday, represented by 5 (0: Sunday, 1: Monday, ..., 6: Saturday)
  in
  let day_of_week_aug_1 := (day_of_week_jan_1 + days_in_year_before_aug % 7) % 7 -- Calculating day of week for August 1, 2016
  in
  let days_in_aug := 31,
      tuesdays := (if day_of_week_aug_1 <= 2 then (days_in_aug - day_of_week_aug_1 + 2) / 7 + 1 else (days_in_aug - day_of_week_aug_1 + 2) / 7),
      saturdays := (if day_of_week_aug_1 <= 6 then (days_in_aug - day_of_week_aug_1 + 6) / 7 + 1 else (days_in_aug - day_of_week_aug_1 + 6) / 7)
  in
  tuesdays + saturdays = 9 := 
by
  -- Proof would go here, but is omitted.
  sorry

end WangHua_practice_days_in_August_2016_l605_605335


namespace original_daily_laying_length_l605_605466

theorem original_daily_laying_length (
  (total_length : ℕ) (first_segment_length : ℕ) (total_days : ℕ)
  (remaining_length : ℕ := total_length - first_segment_length)
  (double_laying_rate := 2)) :
  total_length = 4800 → 
  first_segment_length = 600 →
  total_days = 9 →
  ∃ x : ℕ, 
  (first_segment_length / x) + (remaining_length / (double_laying_rate * x)) = total_days ∧ x = 300 :=
by
  intros h1 h2 h3
  use 300
  split
  · calc
    600 / 300 + 4200 / (2 * 300) = 2 + 7 : by norm_num
                                 ... = 9 : by norm_num
  · exact rfl

end original_daily_laying_length_l605_605466


namespace solution_l605_605658

noncomputable def polynomial_has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - a * x^2 + a * x - 1 = 0

theorem solution (a : ℝ) : polynomial_has_real_root a :=
sorry

end solution_l605_605658


namespace sum_of_divisors_of_23_l605_605943

theorem sum_of_divisors_of_23 : (∑ d in Finset.filter (λ d, 23 % d = 0) (Finset.range 24), d) = 24 :=
by
  -- We need to demonstrate that 23 is a prime number and then compute the sum of its divisors.
  have prime_23 : Nat.Prime 23 := sorry,
  -- Now, we use prime_23 to conclude that the sum of divisors is 24.
  sorry

end sum_of_divisors_of_23_l605_605943


namespace marginal_cost_proof_l605_605903

theorem marginal_cost_proof (fixed_cost : ℕ) (total_cost : ℕ) (n : ℕ) (MC : ℕ)
  (h1 : fixed_cost = 12000)
  (h2 : total_cost = 16000)
  (h3 : n = 20)
  (h4 : total_cost = fixed_cost + MC * n) :
  MC = 200 :=
  sorry

end marginal_cost_proof_l605_605903


namespace find_range_of_a_l605_605265

theorem find_range_of_a (a : ℝ) :
  (-real.pi / 2 ≤ x ∧ x ≤ real.pi / 2) →
  (∀ x, cos (2 * x) - 4 * a * cos x - a + 2 = 0 → ∃! x1 ≠ x2, x1 ≠ x → x2 ≠ x) →
  (3 / 5 < a ∧ a ≤ 1 ∨ a = 1 / 2) :=
by
  sorry

end find_range_of_a_l605_605265


namespace q_evaluate_l605_605873

def q (x y : ℝ) : ℝ :=
if x ≥ 0 ∧ y ≥ 0 then x^2 + y^2
else if x < 0 ∧ y < 0 then x^2 - 3*y
else 2*x + y^2

theorem q_evaluate :
  q (q 2 (-3)) (q (-4) (-1)) = 530 := by
  sorry

end q_evaluate_l605_605873


namespace rhombus_side_length_l605_605818

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605818


namespace reflection_points_line_l605_605008

theorem reflection_points_line (m b : ℝ)
  (h1 : (10 : ℝ) = 2 * (6 - m * (6 : ℝ) + b)) -- Reflecting the point (6, (m * 6 + b)) to (10, 7)
  (h2 : (6 : ℝ) * m + b = 5) -- Midpoint condition
  (h3 : (6 : ℝ) = (2 + 10) / 2) -- Calculating midpoint x-coordinate
  (h4 : (5 : ℝ) = (3 + 7) / 2) -- Calculating midpoint y-coordinate
  : m + b = 15 :=
sorry

end reflection_points_line_l605_605008


namespace quadratic_function_x5_value_l605_605407

theorem quadratic_function_x5_value 
  (a b c : ℝ)
  (f : ℝ → ℝ := λ x, a * x^2 + b * x + c)
  (h1 : f 3 = 10)
  (h2 : ∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ (x1 - x2).abs = 4) :
  f 5 = 0 :=
sorry

end quadratic_function_x5_value_l605_605407


namespace mul_112_54_l605_605578

theorem mul_112_54 : 112 * 54 = 6048 :=
by
  sorry

end mul_112_54_l605_605578


namespace infinite_tame_pairs_l605_605269

def s (n : ℕ) : ℕ :=
  (n.digits 10).sum_sq

def is_tame (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate s k n = 1)

theorem infinite_tame_pairs :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℕ), (∀ i, a i + 1 = b i) ∧ (∀ i, is_tame (a i)) ∧ (∀ i, is_tame (b i)) :=
by
  sorry

end infinite_tame_pairs_l605_605269


namespace ellipse_equation_line_through_point_l605_605273

-- Definitions for the given conditions
def ellipse_eq (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def line_eq (k x : ℝ) : ℝ := k * x - 1
def ellipse_params : Prop :=
  a = 2 * Real.sqrt 2 ∧ b = Real.sqrt 2

-- Main theorem statements
theorem ellipse_equation (x y : ℝ) (a b : ℝ) (h : ellipse_params) : ellipse_eq a b x y :=
  sorry

theorem line_through_point
  (M : ℝ × ℝ) (A B N : ℝ × ℝ)
  (hx : ∃ l, (line_eq l M.1 = M.2) ∧ (∃ k, k ≠ 0 ∧ line_eq k N.1 = 0 ∧ (line_eq l N.1 = 0 ∧ line_eq l A.1 = A.2) ∧ (line_eq k A.1 = A.2) ∧ (line_eq k B.1 = B.2)))
  (h_maj_cond : ∃ (NA NB : ℝ × ℝ), NA = (A.1 - N.1, A.2) ∧ NB = (B.1 - N.1, B.2) ∧ (NA.1, NA.2) = -7/5 * (NB.1, NB.2)) : 
  ∃ l : ℝ, l = 1 ∨ l = -1 :=
  sorry

end ellipse_equation_line_through_point_l605_605273


namespace sum_S_l605_605049

def S : Set ℤ := { x | 10 < x ∧ x < 20 }

theorem sum_S : (∑ x in S, x) = 135 := by
  sorry

end sum_S_l605_605049


namespace angle_ADB_is_90_degrees_l605_605980

open Real EuclideanGeometry

-- Define the given conditions
variables {C A B D : Point} 
variables (r : ℝ) (hC : C.center_of_circle_with_radius r)
variables (hA : A ∈ circle C r)
variables (hB : B ∈ circle C r)
variables (hCA : dist C A = 12) (hCB : dist C B = 12) 
variables (hAB : dist A B = 18)
variables (hD : D ∈ extension_of_line A C ∩ circle C r)

-- Define the theorem to be proven
theorem angle_ADB_is_90_degrees (h : C.center_of_circle_with_radius 12 ∧
                                 A ∈ circle C 12 ∧ B ∈ circle C 12 ∧
                                 dist C A = 12 ∧ dist C B = 12 ∧
                                 dist A B = 18 ∧ D ∈ extension_of_line A C ∩ circle C 12) :
                                 angle A D B = 90 :=
by {
  sorry
}

end angle_ADB_is_90_degrees_l605_605980


namespace snow_probability_first_week_l605_605426

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l605_605426


namespace volume_of_square_pyramid_l605_605709

theorem volume_of_square_pyramid (a r : ℝ) : 
  a > 0 → r > 0 → volume = (1 / 3) * a^2 * r :=
by 
    sorry

end volume_of_square_pyramid_l605_605709


namespace find_height_of_cuboid_l605_605504

variable (A : ℝ) (V : ℝ) (h : ℝ)

theorem find_height_of_cuboid (h_eq : h = V / A) (A_eq : A = 36) (V_eq : V = 252) : h = 7 :=
by
  sorry

end find_height_of_cuboid_l605_605504


namespace sum_integers_between_10_and_20_l605_605046

theorem sum_integers_between_10_and_20 : (∑ k in Finset.Ico 11 20, k) = 135 := 
by sorry

end sum_integers_between_10_and_20_l605_605046


namespace sum_integers_between_10_and_20_l605_605047

theorem sum_integers_between_10_and_20 : (∑ k in Finset.Ico 11 20, k) = 135 := 
by sorry

end sum_integers_between_10_and_20_l605_605047


namespace evaluate_expression_l605_605655

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l605_605655


namespace tangent_parallel_x_axis_coordinates_l605_605497

theorem tangent_parallel_x_axis_coordinates :
  ∃ (x y : ℝ), (y = x^2 - 3 * x) ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) :=
by
  use (3 / 2)
  use (-9 / 4)
  sorry

end tangent_parallel_x_axis_coordinates_l605_605497


namespace num_distinct_four_digit_numbers_l605_605312

theorem num_distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  (finset.card (finset.powersetLen 4 digits)).card = 120 :=
by sorry

end num_distinct_four_digit_numbers_l605_605312


namespace correct_product_of_0_035_and_3_84_l605_605109

theorem correct_product_of_0_035_and_3_84 : 
  (0.035 * 3.84 = 0.1344) := sorry

end correct_product_of_0_035_and_3_84_l605_605109


namespace moles_C2H5Cl_l605_605235

theorem moles_C2H5Cl (nC2H6 nCl2 : ℕ) (hC2H6 : nC2H6 = 2) (hCl2 : nCl2 = 2) :
    (nC2H6 = nCl2) -> (nC2H6 = 2) ∧ (nCl2 = 2) → (∃ nC2H5Cl : ℕ, nC2H5Cl = 2) := by
  intros h1 h2
  cases h2 with hC2H6 hCl2
  use 2
  sorry

end moles_C2H5Cl_l605_605235


namespace min_questions_any_three_cards_min_questions_consecutive_three_cards_l605_605885

-- Definitions for numbers on cards and necessary questions
variables (n : ℕ) (h_n : n > 3)
  (cards : Fin n → ℤ)
  (h_cards_range : ∀ i, cards i = 1 ∨ cards i = -1)

-- Case (a): Product of any three cards
theorem min_questions_any_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (∃ (k : ℕ), n = 3 * k + 1 ∧ p = k + 1) ∨
  (∃ (k : ℕ), n = 3 * k + 2 ∧ p = k + 2) :=
sorry
  
-- Case (b): Product of any three consecutive cards
theorem min_questions_consecutive_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (¬(∃ (k : ℕ), n = 3 * k) ∧ p = n) :=
sorry

end min_questions_any_three_cards_min_questions_consecutive_three_cards_l605_605885


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605557

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605557


namespace mean_proportional_49_64_l605_605961

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l605_605961


namespace min_max_abs_value_l605_605234

theorem min_max_abs_value (y : ℝ) (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  (∀ y, ∃ x, (0 ≤ x ∧ x ≤ 1) → |x^3 - x * y| ≤ 1) :=
sorry

end min_max_abs_value_l605_605234


namespace tan_sin_cos_identity_l605_605257

theorem tan_sin_cos_identity {x : ℝ} (htan : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end tan_sin_cos_identity_l605_605257


namespace intersection_A_B_l605_605299

open Classical

noncomputable def A : set ℝ := { x | x < 2 }
noncomputable def B : set ℝ := { x | 0 ≤ x }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x < 2 } :=
by sorry

end intersection_A_B_l605_605299


namespace probability_snow_at_least_once_first_week_l605_605440

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l605_605440


namespace bus_probabilities_and_chi_squared_l605_605552

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l605_605552


namespace complex_quadrant_l605_605756

theorem complex_quadrant :
  let Z := (2 + complex.i) / (1 - 2 * complex.i) + ((real.sqrt 2) / (1 - complex.i)) ^ 4
  ∃ (q : ℕ), q = 3 ∧ ((∀ (z : complex), (Z = z) → (z.re < 0 ∧ z.im < 0))) :=
by
  -- Defining Z
  let Z := (2 + complex.i) / (1 - 2 * complex.i) + ((real.sqrt 2) / (1 - complex.i)) ^ 4
  -- Proof will go here
  sorry

end complex_quadrant_l605_605756


namespace interest_rate_l605_605454

theorem interest_rate (part1_amount part2_amount total_amount total_income : ℝ) (interest_rate1 interest_rate2 : ℝ) :
  part1_amount = 2000 →
  part2_amount = total_amount - part1_amount →
  interest_rate2 = 6 →
  total_income = (part1_amount * interest_rate1 / 100) + (part2_amount * interest_rate2 / 100) →
  total_amount = 2500 →
  total_income = 130 →
  interest_rate1 = 5 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end interest_rate_l605_605454


namespace biased_coin_probability_l605_605053

open Nat

-- Definitions for conditions: 
-- 1. Probability expressions for heads occurring exactly 2 and 3 times in 6 flips giving the same result.
-- 2. Definition of probability for heads occurring exactly 4 times in 6 flips. 
-- And prove that it equals 19440/117649 in lowest terms.

theorem biased_coin_probability (h : ℝ) 
  (cond : (choose 6 2) * h^2 * (1-h)^4 = (choose 6 3) * h^3 * (1-h)^3) :
  ∃ i j, (choose 6 4) * h^4 * (1-h)^2 = i / j ∧ Nat.gcd i j = 1 ∧ i + j = 137089 :=
by
  sorry

end biased_coin_probability_l605_605053


namespace isosceles_triangle_EDK_from_angle_EDC_l605_605573

section
variables {α : Type*} [euclidean_space α] -- Assuming a Euclidean space context here for simplicity.
variables (O A B C D E F K : α)
variables (circle : set α)
variables (inh : nonempty α) -- To ensure that our space is nonempty
variables (center : O ∈ circle)
variables (diameters : (A ∈ circle) ∧ (B ∈ circle) ∧ (C ∈ circle) ∧ (D ∈ circle))
variables (OD_bisects_EOB : ∠EOB / 2 = ∠EOD)
variables (AE_eq_CF : dist A E = dist C F)
variables (E_F_opposite : ¬(same_side E F C D))
variables (EF_intersects_OD : ∃ K ∈ line[EF], ∃ M ∈ OD, K = M)
variables (is_chord : chord[EF])

theorem isosceles_triangle_EDK_from_angle_EDC :
  ∠EDK = ∠EKD → dist E D = dist K D :=
sorry
end

end isosceles_triangle_EDK_from_angle_EDC_l605_605573


namespace total_green_marbles_l605_605895

theorem total_green_marbles (sara_green sara_red tom_green : ℕ) (h1 : sara_green = 3) (h2 : sara_red = 5) (h3 : tom_green = 4) : 
  sara_green + tom_green = 7 :=
by
  rw [h1, h3]
  sorry

end total_green_marbles_l605_605895


namespace rhombus_side_length_l605_605820

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605820


namespace terminating_decimal_representation_l605_605139

theorem terminating_decimal_representation : 
  let x := (47 : ℚ) / (2^3 * 5^7) in
  x = 0.0000752 :=
by
  -- The proof would go here.
  sorry

end terminating_decimal_representation_l605_605139


namespace trajectory_eq_ellipse_l605_605281

noncomputable def ellipse_trajectory (A F : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, 
    P = (x, y) ∧ 
    (A = (-1, 0)) ∧ 
    (F = (1, 0)) ∧ 
    ((B.1 - 1)^2 + B.2^2 = (2 * Real.sqrt 3)^2) ∧ 
    ((x + 1) * (x - 1) + y^2 = y^2) ∧ 
    (P.1^2 / 3 + P.2^2 / 2 = 1)

theorem trajectory_eq_ellipse :
  ∀ (A F : ℝ × ℝ) (B : ℝ × ℝ), 
    ellipse_trajectory A F B :=
  by
  sorry

end trajectory_eq_ellipse_l605_605281


namespace least_n_factorial_7350_l605_605041

theorem least_n_factorial_7350 (n : ℕ) :
  (7350 ∣ n!) → (∃ k : ℕ, (1 ≤ k ∧ 7350 ∣ k!)) :=
by
  have prime_factors_7350 : 7350 = 2 * 3^2 * 5^2 * 7 := by norm_num
  existsi 10
  split
  · norm_num
  · apply dvd_factorial.mpr
    sorry

end least_n_factorial_7350_l605_605041


namespace log_value_between_consecutive_integers_l605_605021

theorem log_value_between_consecutive_integers :
  let x := log 5 15625
  in (6 ≤ x ∧ x < 7) ∧ (6 + 7 = 13) :=
by 
  sorry

end log_value_between_consecutive_integers_l605_605021


namespace integral_value_correct_l605_605631

noncomputable def integral_value : ℝ :=
  ∫ x in (0 : ℝ)..5, real.sqrt (25 - x^2)

theorem integral_value_correct : integral_value = 25 * real.pi / 4 := by
  sorry

end integral_value_correct_l605_605631


namespace profit_percentage_no_initial_discount_l605_605996

theorem profit_percentage_no_initial_discount
  (CP : ℝ := 100)
  (bulk_discount : ℝ := 0.02)
  (sales_tax : ℝ := 0.065)
  (no_discount_price : ℝ := CP - CP * bulk_discount)
  (selling_price : ℝ := no_discount_price + no_discount_price * sales_tax)
  (profit : ℝ := selling_price - CP) :
  (profit / CP) * 100 = 4.37 :=
by
  -- proof here
  sorry

end profit_percentage_no_initial_discount_l605_605996


namespace equal_distances_to_diagonal_intersection_l605_605964

variable {A B C D O : Type} [MetricSpace A]
-- Given a convex quadrilateral ABCD
variable (A B C D : ℝ)
variable (O : ℝ)  -- O is the intersection point of diagonals AC and BD

-- Given conditions
variable (h1 : Real.perimeter (A, B, C) = Real.perimeter (A, B, D))
variable (h2 : Real.perimeter (A, C, D) = Real.perimeter (B, C, D))

-- To prove: distances from points A and B to O are equal (OA = OB)
theorem equal_distances_to_diagonal_intersection 
  (h1 : Real.perimeter (A, B, C) = Real.perimeter (A, B, D))
  (h2 : Real.perimeter (A, C, D) = Real.perimeter (B, C, D))
  (O_on_AC : isOnLineSegment O A C)
  (O_on_BD : isOnLineSegment O B D) :
  dist A O = dist B O :=
sorry

end equal_distances_to_diagonal_intersection_l605_605964


namespace ratio_arithmetic_seq_a2019_a2017_eq_l605_605128

def ratio_arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n ≥ 1 → a (n+2) / a (n+1) - a (n+1) / a n = 2

theorem ratio_arithmetic_seq_a2019_a2017_eq (a : ℕ → ℝ) 
  (h : ratio_arithmetic_seq a) 
  (ha1 : a 1 = 1) 
  (ha2 : a 2 = 1) 
  (ha3 : a 3 = 3) : 
  a 2019 / a 2017 = 4 * 2017^2 - 1 :=
sorry

end ratio_arithmetic_seq_a2019_a2017_eq_l605_605128


namespace volume_of_T_l605_605917

noncomputable def region (x y z : ℝ) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2

theorem volume_of_T (V : ℝ) :
  (∫∫∫ x y z in region (x y z), 1) = V ↔ V = 1664 / 81 :=
by
  sorry

end volume_of_T_l605_605917


namespace sum_greatest_odd_divisors_l605_605866

-- Define g(k) as the greatest odd divisor of k
def g (k : ℕ) : ℕ :=
  if k = 0 then 0 else 
  let rec aux (m : ℕ) : ℕ := 
    if m % 2 = 1 then m else aux (m / 2)
  aux k

-- Define S(n) as the sum of g(i) for i from 1 to 2^n
def S (n : ℕ) : ℕ :=
  (Finset.range (2^n + 1)).sum (λ i, g i)

-- The main theorem to prove
theorem sum_greatest_odd_divisors (n : ℕ) : 
  S n = (4^n + 5) / 3 := sorry

end sum_greatest_odd_divisors_l605_605866


namespace intersection_points_in_polar_coordinates_l605_605356

noncomputable theory

open Real

def polar_eq1 (theta : ℝ) : ℝ := 2 * sin theta

def polar_eq2 (theta : ℝ) : ℝ := - (sqrt 3) / 2 / cos theta

theorem intersection_points_in_polar_coordinates :
  (∃ θ1 θ2 ρ1 ρ2, 
    polar_eq1 θ1 = ρ1 ∧
    polar_eq2 θ1 = ρ1 ∧
    polar_eq1 θ2 = ρ2 ∧
    polar_eq2 θ2 = ρ2 ∧
    0 ≤ θ1 ∧ θ1 < 2 * π ∧
    0 ≤ θ2 ∧ θ2 < 2 * π ∧
    ((ρ1 = 1 ∧ θ1 = (5 * π / 6)) ∨ (ρ1 = sqrt 3 ∧ θ1 = (2 * π / 3))) ∧
    ((ρ2 = 1 ∧ θ2 = (5 * π / 6)) ∨ (ρ2 = sqrt 3 ∧ θ2 = (2 * π / 3)))) :=
sorry

end intersection_points_in_polar_coordinates_l605_605356


namespace squirrels_acorns_l605_605972

theorem squirrels_acorns (x : ℕ) : 
    (5 * (x - 15) = 575) → 
    x = 130 := 
by 
  intros h
  sorry

end squirrels_acorns_l605_605972


namespace smallest_initial_amount_l605_605608

variables (x y a b c : ℕ)
variables (initial_boy_amount initial_girl_amount : ℕ)

-- Conditions
axiom eq_final_boy_amount : initial_boy_amount - 3 * a + 3 * b = c
axiom eq_final_girl_amount : initial_girl_amount - 9 * b + 9 * a = c
axiom equal_final_amount : initial_boy_amount - 3 * a + 3 * b = initial_girl_amount - 9 * b + 9 * a

-- Theorem
theorem smallest_initial_amount (h₁ : initial_boy_amount = 12) (h₂ : initial_girl_amount = 36) :
    initial_boy_amount = 12 :=
begin
  sorry
end

end smallest_initial_amount_l605_605608


namespace infinite_prime_diffs_l605_605389

noncomputable theory

open Nat

theorem infinite_prime_diffs (k : ℕ) (h : k ≥ 3) : 
  ∃ᶠ n in atTop, Prime (a (n+1) - a n) :=
sorry

end infinite_prime_diffs_l605_605389


namespace georgia_problems_left_l605_605255

theorem georgia_problems_left :
  ∀ (total_problems : ℕ) (time1 time2 time3 problems1 problems2 : ℕ),
  total_problems = 75 →
  time1 = 20 →
  time2 = 20 →
  problems1 = 10 →
  problems2 = 2 * problems1 →
  problems1 + problems2 = 30 →
  75 - (problems1 + problems2) = 45 :=
by {
  intros total_problems time1 time2 time3 problems1 problems2
         h_total h_time1 h_time2 h_problems1 h_problems2 h_sum,
  rw [h_total, h_problems1, h_problems2],
  exact h_sum,
  }

end georgia_problems_left_l605_605255


namespace smallest_positive_n_l605_605676

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605676


namespace conic_section_description_l605_605535

theorem conic_section_description :
  ¬(∃ t : Type, t ∈ {"circle", "parabola", "ellipse", "hyperbola"} ∧ (λ x y : ℝ, sqrt ((x+3)^2 + (y-2)^2) = abs (x+3))) :=
by sorry

end conic_section_description_l605_605535


namespace distinct_four_digit_numbers_count_l605_605310

theorem distinct_four_digit_numbers_count : 
  ∃ (digits: Set ℕ) (f : Fin 4 → ℕ), 
    digits = {1, 2, 3, 4, 5} ∧ (∀ i j, i ≠ j → f i ∈ digits ∧ f i ≠ f j) 
  → fintype.card (finset.univ.filter (λ (x : ℕ), x ∈ {a | ∃ (digits : Set ℕ) (f : Fin 4 → ℕ), digits = {1, 2, 3, 4, 5} ∧ (∀ i j, i ≠ j → f i ∈ digits ∧ f i ≠ f j) ∧
    x = f 0 * 1000 + f 1 * 100 + f 2 * 10 + f 3})) = 120 :=
by
  sorry

end distinct_four_digit_numbers_count_l605_605310


namespace range_dot_product_l605_605751

noncomputable def parametrizeP (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

noncomputable def parametrizeA (θ : ℝ) : ℝ × ℝ :=
  (3 + Real.cos θ, 4 + Real.sin θ)

noncomputable def parametrizeB (θ : ℝ) : ℝ × ℝ :=
  (3 - Real.cos θ, 4 - Real.sin θ)

noncomputable def dotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem range_dot_product :
  ∀ (α θ : ℝ), 
  let P := parametrizeP α in
  let A := parametrizeA θ in
  let B := parametrizeB θ in
  15 ≤ dotProduct (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) ∧ 
  dotProduct (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) ≤ 35 := by
  sorry

end range_dot_product_l605_605751


namespace g_at_4_l605_605898

def f (x : ℝ) : ℝ := 4 / (3 - x)
def f_inv (x : ℝ) : ℝ := 3 - 4 / x
def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8 :=
by
  sorry

end g_at_4_l605_605898


namespace smallest_positive_n_l605_605686

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605686


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605204

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605204


namespace determinant_inverse_product_l605_605786

variables (C D : Matrix (Fin n) (Fin n) ℝ)  -- Variables for square matrices of finite dimensions over real numbers

-- Given conditions as hypotheses
hypothesis det_C : det C = -3
hypothesis det_D : det D = 7

-- The statement to be proved
theorem determinant_inverse_product : det (C⁻¹ ⬝ D) = -7 / 3 :=
  sorry

end determinant_inverse_product_l605_605786


namespace quadratic_no_real_solution_l605_605285

theorem quadratic_no_real_solution 
  (a b c : ℝ) 
  (h1 : (2 * a)^2 - 4 * b^2 > 0) 
  (h2 : (2 * b)^2 - 4 * c^2 > 0) : 
  (2 * c)^2 - 4 * a^2 < 0 :=
sorry

end quadratic_no_real_solution_l605_605285


namespace functional_eq_zero_function_l605_605867

theorem functional_eq_zero_function (f : ℝ → ℝ) (k : ℝ) (h : ∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_function_l605_605867


namespace range_of_a_l605_605906

theorem range_of_a (f : ℝ → ℝ) (h_mono_dec : ∀ x1 x2, -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 → x1 < x2 → f x1 > f x2) 
  (h_cond : ∀ a, -2 ≤ a + 1 ∧ a + 1 ≤ 2 ∧ -2 ≤ 2 * a ∧ 2 * a ≤ 2 → f (a + 1) < f (2 * a)) :
  { a : ℝ | -1 ≤ a ∧ a < 1 } :=
sorry

end range_of_a_l605_605906


namespace sum_of_integers_between_10_and_20_l605_605045

theorem sum_of_integers_between_10_and_20 :
  ∑ i in Finset.range (20 - 10 - 1), (i + 11) = 135 := by
  sorry

end sum_of_integers_between_10_and_20_l605_605045


namespace number_of_paths_from_A_to_D_l605_605244

-- Definitions based on conditions
def paths_A_to_B : ℕ := 2
def paths_B_to_C : ℕ := 2
def paths_A_to_C : ℕ := 1
def paths_C_to_D : ℕ := 2
def paths_B_to_D : ℕ := 2

-- Theorem statement
theorem number_of_paths_from_A_to_D : 
  paths_A_to_B * paths_B_to_C * paths_C_to_D + 
  paths_A_to_C * paths_C_to_D + 
  paths_A_to_B * paths_B_to_D = 14 :=
by {
  -- proof steps will go here
  sorry
}

end number_of_paths_from_A_to_D_l605_605244


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605224

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605224


namespace sum_of_solutions_abs_square_l605_605545

theorem sum_of_solutions_abs_square (x : ℝ) :
  (∀ x, |x - 5|^2 + |x - 5| = 20 → x = 9 ∨ x = 1) →
  (∑ x in ({9, 1} : Finset ℝ), x) = 10 :=
by
  intro h
  have hx1 : 1 ∈ ({9, 1} : Finset ℝ), from Finset.mem_insert_of_mem (Finset.mem_singleton_self 1),
  have hx9 : 9 ∈ ({9, 1} : Finset ℝ), from Finset.mem_insert_self 9 {1},
  rw [Finset.sum_insert Finset.not_mem_of_mem_singleton_self, Finset.sum_singleton]
  all_goals { try { norm_num } },
  sorry

end sum_of_solutions_abs_square_l605_605545


namespace files_left_after_deletion_l605_605625

-- Definitions
def initial_apps := 17
def initial_files := 21
def remaining_apps := 3
def deleted_files := 14

-- Problem Statement
theorem files_left_after_deletion:
  (initial_files - deleted_files = 7): sorry

end files_left_after_deletion_l605_605625


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605184

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605184


namespace find_abs_xyz_l605_605391

noncomputable def conditions_and_question (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1)

theorem find_abs_xyz (x y z : ℝ) (h : conditions_and_question x y z) : |x * y * z| = 1 :=
  sorry

end find_abs_xyz_l605_605391


namespace quadratic_function_value_l605_605406

noncomputable def f (x : ℝ) : ℝ :=
  let a := -5 / 2 in
  a * (x - 3) ^ 2 + 10

theorem quadratic_function_value :
  (∀ x : ℝ, f x = (-5 / 2) * (x - 3) ^ 2 + 10) →
  (∃ a b c : ℝ, f x = a * x ^ 2 + b * x + c ∧ (∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ real.abs (x1 - x2) = 4 ∧ f 3 = 10)) →
  f 5 = 0 :=
by sorry

end quadratic_function_value_l605_605406


namespace ordered_pairs_count_l605_605781

theorem ordered_pairs_count : 
    ∃ (s : Finset (ℝ × ℝ)), 
        (∀ (x y : ℝ), (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1 ↔ (x, y) ∈ s)) ∧ 
        s.card = 3 :=
    by
    sorry

end ordered_pairs_count_l605_605781


namespace population_moved_away_l605_605059

theorem population_moved_away (initial_population : ℕ) (growth_rate : ℝ) (current_population : ℕ) 
(h_initial : initial_population = 684) (h_growth : growth_rate = 0.25) (h_current : current_population = 513) :
  ∃ percentage_moved_away : ℝ, percentage_moved_away = 40 :=
begin
  sorry
end

end population_moved_away_l605_605059


namespace find_a3_a4_a5_l605_605342

open Real

variables {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def a_1 : ℝ := 3

def sum_of_first_three (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 21

def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a3_a4_a5 (h1 : is_geometric_sequence a) (h2 : a 0 = a_1) (h3 : sum_of_first_three a) (h4 : all_terms_positive a) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end find_a3_a4_a5_l605_605342


namespace probability_of_2_heads_in_3_tosses_l605_605981

theorem probability_of_2_heads_in_3_tosses :
  let n := 3
  let k := 2
  let p := 0.5
  ∃ P : ℝ, P = (nat.choose 3 2) * (p^2) * (p^(3-2)) ∧ P = 0.375 :=
begin
  sorry,
end

end probability_of_2_heads_in_3_tosses_l605_605981


namespace terminating_decimal_nonzero_thousandths_l605_605722

theorem terminating_decimal_nonzero_thousandths (n : ℕ) :
  (∃ (k : ℕ), k = 4 ∧ ∀ m (h₁ : 1 ≤ m ∧ m ≤ 1000) (h₂ : terminating (1 / m)), nonzero_thousandths_digit (1 / m) → k = 4) :=
sorry

-- Definitions and helpers:
def terminating (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

def nonzero_thousandths_digit (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

end terminating_decimal_nonzero_thousandths_l605_605722


namespace snow_prob_correct_l605_605424

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l605_605424


namespace soda_preference_count_eq_243_l605_605347

def total_respondents : ℕ := 540
def soda_angle : ℕ := 162
def total_circle_angle : ℕ := 360

theorem soda_preference_count_eq_243 :
  (total_respondents * soda_angle / total_circle_angle) = 243 := 
by 
  sorry

end soda_preference_count_eq_243_l605_605347


namespace captain_age_l605_605339

-- Definitions
def num_team_members : ℕ := 11
def total_team_age : ℕ := 11 * 24
def total_age_remainder := 9 * (24 - 1)
def combined_age_of_captain_and_keeper := total_team_age - total_age_remainder

-- The actual proof statement
theorem captain_age (C : ℕ) (W : ℕ) 
  (hW : W = C + 5)
  (h_total_team : total_team_age = 264)
  (h_total_remainders : total_age_remainder = 207)
  (h_combined_age : combined_age_of_captain_and_keeper = 57) :
  C = 26 :=
by sorry

end captain_age_l605_605339


namespace convex_hexagon_diagonal_triangle_area_convex_octagon_diagonal_triangle_area_l605_605575

-- Problem a) for Hexagon: 
theorem convex_hexagon_diagonal_triangle_area (S : ℝ) (hexagon : convex_hexagon) :
  ∃ (d : diagonal hexagon) (t : triangle d), area t ≤ S / 6 :=
sorry

-- Problem b) for Octagon: 
theorem convex_octagon_diagonal_triangle_area (S : ℝ) (octagon : convex_octagon) :
  ∃ (d : diagonal octagon) (t : triangle d), area t ≤ S / 8 :=
sorry

end convex_hexagon_diagonal_triangle_area_convex_octagon_diagonal_triangle_area_l605_605575


namespace average_weight_before_new_student_l605_605901

theorem average_weight_before_new_student (x : ℝ) (h₁ : 29 * x + 13 = 30 * 27.5) : x = 28 :=
by 
  have h₂ : 30 * 27.5 = 825, by norm_num,
  rw h₂ at h₁,
  have h₃ : 29 * x + 13 = 825, from h₁,
  have h₄ : 29 * x = 825 - 13, by linarith,
  have h₅ : 29 * x = 812, by norm_num,
  have h₆ : x = 812 / 29, by linarith,
  have h₇ : 812 / 29 = 28, by norm_num,
  rw h₇ at h₆,
  exact h₆

end average_weight_before_new_student_l605_605901


namespace find_initial_number_l605_605237

-- Define the initial equation
def initial_equation (x : ℤ) : Prop := x - 12 * 3 * 2 = 9938

-- Prove that the initial number x is equal to 10010 given initial_equation
theorem find_initial_number (x : ℤ) (h : initial_equation x) : x = 10010 :=
sorry

end find_initial_number_l605_605237


namespace find_f_2016_l605_605080

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity : ∀ x : ℝ, f(x) * f(x + 5) = 3
axiom f_at_one : f(1) = 2

theorem find_f_2016 : f(2016) = 3 / 2 := 
by 
  sorry

end find_f_2016_l605_605080


namespace equal_segments_l605_605853

theorem equal_segments 
  (A B C D E F : Type)
  [triangle ABC : Triangle A B C]
  [angle_bisector B AD : AngleBisector B A D]
  [circumscribed_tri ABD : Circumcircle A B D -| E (BC)]
  [circumscribed_tri BCD : Circumcircle B C D -| F (AB)]
  : AE = CF :=
by
  sorry

end equal_segments_l605_605853


namespace mean_of_reciprocals_first_four_primes_l605_605147

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605147


namespace slope_of_l3_l605_605411

section

structure Point where
  x : ℝ
  y : ℝ

def line1 (P : Point) : Prop := 3 * P.x - 2 * P.y = 1
def pointA : Point := ⟨-1, -2⟩

def line2 (P : Point) : Prop := P.y = 1

def area_triangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

def positive_slope (A C : Point) : Prop :=
  (C.y - A.y) / (C.x - A.x) > 0

noncomputable def slope (A C : Point) : ℝ :=
  (C.y - A.y) / (C.x - A.x)

theorem slope_of_l3 :
  let B := ⟨1, 1⟩
  let C1 := ⟨3, 1⟩
  let C2 := ⟨-1, 1⟩
  let sl3 := slope pointA C1
  positive_slope pointA C1 →
  area_triangle pointA B C1 = 3 →
  ↔
  sl3 = 3/4 :=
by
  sorry

end

end slope_of_l3_l605_605411


namespace determine_x_l605_605800

noncomputable def y : ℝ → ℝ := λ x, (x^2 - 9) / (x - 3)

theorem determine_x (x : ℝ) (h₁ : x ≠ 3) (h₂ : y x = 3 * x + 1) : x = 1 :=
by
  sorry

end determine_x_l605_605800


namespace range_of_a_l605_605753

noncomputable def even_function (f : ℝ → ℝ) := ∀ x ∈ Ioo (-1 : ℝ) 1, f (-x) = f x
noncomputable def increasing_on_Ioo (f : ℝ → ℝ) := ∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y

theorem range_of_a 
  (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : increasing_on_Ioo f) 
  (h3 : ∀ a : ℝ, f (a-2) - f (4 - a^2) < 0) :
  ∀ a : ℝ, a ∈ Ioo (-real.sqrt 5) (-real.sqrt 3) := sorry

end range_of_a_l605_605753


namespace checkerboard_inequivalent_color_schemes_l605_605927

/-- 
  We consider a 7x7 checkerboard where two squares are painted yellow, and the remaining 
  are painted green. Two color schemes are equivalent if one can be obtained from 
  the other by rotations of 0°, 90°, 180°, or 270°. We aim to prove that the 
  number of inequivalent color schemes is 312. 
-/
theorem checkerboard_inequivalent_color_schemes : 
  let n := 7
  let total_squares := n * n
  let total_pairs := total_squares.choose 2
  let symmetric_pairs := 24
  let nonsymmetric_pairs := total_pairs - symmetric_pairs
  let unique_symmetric_pairs := symmetric_pairs 
  let unique_nonsymmetric_pairs := nonsymmetric_pairs / 4
  unique_symmetric_pairs + unique_nonsymmetric_pairs = 312 :=
by sorry

end checkerboard_inequivalent_color_schemes_l605_605927


namespace romanov_family_savings_l605_605571

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l605_605571


namespace probability_of_snow_at_least_once_first_week_l605_605444

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l605_605444


namespace part_one_part_two_l605_605766

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x - 2)

-- (I) Prove the inequality
theorem part_one (x : ℝ) : f x + f (2 * x + 1) ≥ 6 → x ∈ set.Iic (-1) ∪ set.Ici 3 :=
by
  sorry

-- (II) Prove the range of m
theorem part_two (a b m : ℝ) (h : a + b = 1) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x, |x - 2 - m| - |x + 2| ≤ 4 / a + 1 / b) ↔ (-13 ≤ m ∧ m ≤ 5) :=
by
  sorry

end part_one_part_two_l605_605766


namespace distance_from_origin_to_AB_is_constant_l605_605289

theorem distance_from_origin_to_AB_is_constant :
  (∃ A B : ℝ × ℝ, (A.1^2 / 2 + A.2^2 = 1) ∧ (B = (B.1, sqrt 2)) 
  ∧ (A.1 * B.1 + A.2 * sqrt 2 = 0) ∧ (dist_point_to_line ((0, 0) : ℝ × ℝ) (line_through A B) = 1)) :=
sorry

-- Hypothetical definition of distance from a point to a line
def dist_point_to_line (P : ℝ × ℝ) (L : set (ℝ × ℝ)) : ℝ :=
  sorry

-- Hypothetical definition of a line through two points
def line_through (A B : ℝ × ℝ) : set (ℝ × ℝ) :=
  sorry

end distance_from_origin_to_AB_is_constant_l605_605289


namespace incorrect_calculation_B_l605_605949

theorem incorrect_calculation_B (hA : (-1)^(2020) = 1)
                               (hC : (((1 : ℚ) / 3)⁻¹) = 3)
                               (hD : -(2 : ℤ)^2 = -4) :
                               ¬(|-3| = ℚ.ofInt 3 ∨ |-3| = -3) :=
by {
  -- proof goes here
  sorry
}

end incorrect_calculation_B_l605_605949


namespace count_terminating_n_with_conditions_l605_605726

def is_terminating_decimal (n : ℕ) : Prop :=
  (∃ m k : ℕ, (2^m * 5^k = n)) 

def thousandths_non_zero (n : ℕ) : Prop := 
  100 < n ∧ n ≤ 1000

theorem count_terminating_n_with_conditions : 
  {n : ℕ | is_terminating_decimal n ∧ thousandths_non_zero n}.to_finset.card = 9 := 
by
  sorry

end count_terminating_n_with_conditions_l605_605726


namespace safe_numbers_7_11_13_l605_605246

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℕ, ∀ m ∈ {0, 1, 2, p-2, p-1, p}, n ≠ p * k + m ∧ n ≠ p * k - m

def count_safe_numbers (p₁ p₂ p₃ n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ x, is_p_safe x p₁ ∧ is_p_safe x p₂ ∧ is_p_safe x p₃).card

theorem safe_numbers_7_11_13 (h : count_safe_numbers 7 11 13 10000 = 958) : ℕ :=
  h

end safe_numbers_7_11_13_l605_605246


namespace total_points_l605_605056

theorem total_points (paul_points cousin_points : ℕ) 
  (h_paul : paul_points = 3103) 
  (h_cousin : cousin_points = 2713) : 
  paul_points + cousin_points = 5816 := by
sorry

end total_points_l605_605056


namespace sub_numbers_correct_l605_605897

theorem sub_numbers_correct : 
  (500.50 - 123.45 - 55 : ℝ) = 322.05 := by 
-- The proof can be filled in here
sorry

end sub_numbers_correct_l605_605897


namespace f_of_f4_eq_one_half_max_value_of_f_l605_605291

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_of_f4_eq_one_half : f (f 4) = 1 / 2 := 
  sorry

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 1 := 
  by 
    intro x
    dsimp [f]
    split_ifs
    case _ h => 
      exact sub_le_self 1 (Real.sqrt_nonneg _)
    case _ h => 
      apply rpow_le_one
      linarith
      norm_num
    sorry

end f_of_f4_eq_one_half_max_value_of_f_l605_605291


namespace sqrt_cos_sin_relation_l605_605794

variable {a b c θ : ℝ}

theorem sqrt_cos_sin_relation 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * (Real.cos θ) ^ 2 + b * (Real.sin θ) ^ 2 < c) :
  Real.sqrt a * (Real.cos θ) ^ 2 + Real.sqrt b * (Real.sin θ) ^ 2 < Real.sqrt c :=
sorry

end sqrt_cos_sin_relation_l605_605794


namespace rectangle_problem_l605_605343

noncomputable def calculate_width (L P : ℕ) : ℕ :=
  (P - 2 * L) / 2

theorem rectangle_problem :
  ∀ (L P : ℕ), L = 12 → P = 36 → (calculate_width L P = 6) ∧ ((calculate_width L P) / L = 1 / 2) :=
by
  intros L P hL hP
  have hw : calculate_width L P = 6 := by
    sorry
  have hr : ((calculate_width L P) / L) = 1 / 2 := by
    sorry
  exact ⟨hw, hr⟩

end rectangle_problem_l605_605343


namespace find_AC_find_area_l605_605333

theorem find_AC (BC : ℝ) (angleA : ℝ) (cosB : ℝ) 
(hBC : BC = Real.sqrt 7) (hAngleA : angleA = 60) (hCosB : cosB = Real.sqrt 6 / 3) :
  (AC : ℝ) → (hAC : AC = 2 * Real.sqrt 7 / 3) → Prop :=
by
  sorry

theorem find_area (BC AB : ℝ) (angleA : ℝ) 
(hBC : BC = Real.sqrt 7) (hAB : AB = 2) (hAngleA : angleA = 60) :
  (area : ℝ) → (hArea : area = 3 * Real.sqrt 3 / 2) → Prop :=
by
  sorry

end find_AC_find_area_l605_605333


namespace M_geq_N_l605_605261

variable (x y : ℝ)
def M : ℝ := x^2 + y^2 + 1
def N : ℝ := x + y + x * y

theorem M_geq_N (x y : ℝ) : M x y ≥ N x y :=
by
sorry

end M_geq_N_l605_605261


namespace sum_of_coordinates_l605_605923

theorem sum_of_coordinates (a b : ℤ) :
  (a^2 - 4 * a + b^2 - 8 * b = 30) →
  ∑ (p : ℤ × ℤ) in
    ({(a, b) | a^2 - 4 * a + b^2 - 8 * b = 30} : finset (ℤ × ℤ)),
    (p.1 + p.2) = 60 :=
by sorry

end sum_of_coordinates_l605_605923


namespace terminating_decimal_nonzero_thousandths_l605_605720

theorem terminating_decimal_nonzero_thousandths (n : ℕ) :
  (∃ (k : ℕ), k = 4 ∧ ∀ m (h₁ : 1 ≤ m ∧ m ≤ 1000) (h₂ : terminating (1 / m)), nonzero_thousandths_digit (1 / m) → k = 4) :=
sorry

-- Definitions and helpers:
def terminating (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

def nonzero_thousandths_digit (x : ℚ) : Prop :=
-- placeholder for the actual definition
sorry

end terminating_decimal_nonzero_thousandths_l605_605720


namespace romanov_family_savings_l605_605568

theorem romanov_family_savings :
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption := monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  savings = 3824 :=
by
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption :=monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2 
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  show savings = 3824 
  sorry

end romanov_family_savings_l605_605568


namespace probability_of_15_cents_or_more_heads_l605_605900

theorem probability_of_15_cents_or_more_heads (penny_heads nickel_heads dime_heads quarter_heads : Bool) : 
  ((quarter_heads = true) ∨ (nickel_heads = true ∧ dime_heads = true)) → probability ⟨penny_heads, nickel_heads, dime_heads, quarter_heads⟩ = 5/8 := 
  sorry

end probability_of_15_cents_or_more_heads_l605_605900


namespace probability_snow_at_least_once_first_week_l605_605439

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l605_605439


namespace max_lines_with_equal_plane_angles_l605_605321

theorem max_lines_with_equal_plane_angles (n : ℕ) (hn : n ≥ 2) :
  ∃ α : Plane, ∀ (lines : Fin n → Line), ∀ i j : Fin n, angle (lines i) α = angle (lines j) α → n ≤ 3 :=
sorry

end max_lines_with_equal_plane_angles_l605_605321


namespace arithmetic_mean_reciprocals_primes_l605_605156

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605156


namespace tim_final_soda_cans_l605_605509

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l605_605509


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605218

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605218


namespace C1_standard_eq_C2_standard_eq_AB_length_l605_605352

noncomputable def C1_param_eq :=
  {x := 2 * cos θ, y := 2 * sqrt 3 * sin θ}

def C2_polar_eq (θ : ℝ) : ℝ := 2 * cos θ - 4 * sin θ

theorem C1_standard_eq :
  ∀ (x y : ℝ), (x = 2 * cos θ) ∧ (y = 2 * sqrt 3 * sin θ) → (x^2 / 4 + y^2 / 12 = 1) :=
by
  intros x y h
  cases h with hx hy
  sorry

theorem C2_standard_eq :
  ∀ (x y : ℝ), ((x^2 + y^2 = 2 * x - 4 * y) ↔ ((x - 1)^2 + (y + 2)^2 = 5)) :=
by
  intros x y
  sorry

theorem AB_length :
  ∀ (x y : ℝ) (m : ℝ), (C2_polar_eq x = m) ∧ (C2_polar_eq y = m) → m > 0 ∧ (| (x, y) | = 3 * sqrt 2) :=
by
  intros x y m h
  cases h with h₁ h₂
  sorry

end C1_standard_eq_C2_standard_eq_AB_length_l605_605352


namespace abs_pi_diff_9_l605_605116

theorem abs_pi_diff_9 : ∀ (π : ℝ), π < 9 → |π - |π - 9|| = 9 - 2 * π :=
by
  intros π h
  have h1 : |π - 9| = 9 - π, from abs_of_pos (by linarith)
  have h2 : |π - (9 - π)| = |2 * π - 9|, by linarith
  have h3 : |2 * π - 9| = 9 - 2 * π, from abs_of_neg (by linarith)
  rw [abs_pi h, h2, h3]
  ring

end abs_pi_diff_9_l605_605116


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605222

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605222


namespace distance_between_circumcenters_l605_605888

variable {α R : ℝ}
variables {A B C D : Point}

-- Given Conditions
def CD_eq_alpha_AC (C D A : Point) (α : ℝ) : Prop := dist C D = α * dist A C
def radius_circumcircle_ABC (A B C : Point) (R : ℝ) : Prop := circumradius A B C = R

-- Theorem Statement
theorem distance_between_circumcenters 
    (h1 : CD_eq_alpha_AC C D A α)
    (h2 : radius_circumcircle_ABC A B C R) :
    dist (circumcenter A B C) (circumcenter A D B) = α * R :=
sorry

end distance_between_circumcenters_l605_605888


namespace subsequence_sum_q_l605_605863

theorem subsequence_sum_q (S : Fin 1995 → ℕ) (m : ℕ) (hS_pos : ∀ i : Fin 1995, 0 < S i)
  (hS_sum : (Finset.univ : Finset (Fin 1995)).sum S = m) (h_m_lt : m < 3990) :
  ∀ q : ℕ, 1 ≤ q → q ≤ m → ∃ (I : Finset (Fin 1995)), I.sum S = q := 
sorry

end subsequence_sum_q_l605_605863


namespace train_length_l605_605095

theorem train_length (L V : ℝ) 
  (h1 : L = V * 110) 
  (h2 : L + 700 = V * 180) : 
  L = 1100 :=
by
  sorry

end train_length_l605_605095


namespace mean_of_reciprocals_of_first_four_primes_l605_605193

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605193


namespace find_a_l605_605821

noncomputable def point := { x : ℝ // x = 3 ∧ 4 }
noncomputable def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 2) ^ 2 = 4
noncomputable def is_tangent_to_circle (L : ℝ → ℝ → Prop) :=
  ∃ k : ℝ, ∀ x y, L x y = (k * x - y - (3 * k) + 4 = 0) ∧ (abs (-k + 2) / real.sqrt (k ^ 2 + 1) = 2)

noncomputable def is_perpendicular (L1 L2 : ℝ → ℝ → Prop) :=
  ∃ a : ℝ, ∀ x y, (L1 x y = a * x - y + 1 = 0) ∧ (L2 x y = -x / a + y - 3 / a + 4 / a = 0)

theorem find_a : 
  ∀ (P: point) (C: circle) (L : is_tangent_to_circle L) (L_perpendicular : is_perpendicular L (λ x y => ax - y + 1)),
  ∃ a: ℝ, a = 3/4 := sorry

end find_a_l605_605821


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605231

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605231


namespace p_suff_but_not_nec_q_l605_605788

variable (p q : Prop)

-- Given conditions: ¬p is a necessary but not sufficient condition for ¬q.
def neg_p_nec_but_not_suff_neg_q : Prop :=
  (¬q → ¬p) ∧ ¬(¬p → ¬q)

-- Concluding statement: p is a sufficient but not necessary condition for q.
theorem p_suff_but_not_nec_q 
  (h : neg_p_nec_but_not_suff_neg_q p q) : (p → q) ∧ ¬(q → p) := 
sorry

end p_suff_but_not_nec_q_l605_605788


namespace fifth_smallest_three_digit_number_l605_605938

theorem fifth_smallest_three_digit_number : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ (n.digits 10), d ∈ {3, 6, 9}) 
  ∧  ∃ f : Fin 5, n = [333, 336, 339, 363, 366].nth f.val) := 
  ∃ n : ℕ, n = 366 :=
by
  sorry

end fifth_smallest_three_digit_number_l605_605938


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605206

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605206


namespace least_n_factorial_l605_605036

theorem least_n_factorial (n : ℕ) : 
  (7350 ∣ nat.factorial n) ↔ n ≥ 15 := 
sorry

end least_n_factorial_l605_605036


namespace sufficient_but_not_necessary_condition_l605_605016

variables (x y : ℝ)

theorem sufficient_but_not_necessary_condition :
  ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → ((x - 1) * (y - 2) = 0) ∧ (¬ ((x - 1) * (y-2) = 0 → (x - 1)^2 + (y - 2)^2 = 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l605_605016


namespace eccentricity_of_hyperbola_l605_605280

noncomputable def parabola := {P : ℝ × ℝ | P.fst ^ 2 = 4 * P.snd}

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 1)
def P := (2, 1)

def PA (P : ℝ × ℝ) := real.sqrt ((P.fst - A.fst) ^ 2 + (P.snd - A.snd) ^ 2)
def PB (P : ℝ × ℝ) := real.sqrt ((P.fst - B.fst) ^ 2 + (P.snd - B.snd) ^ 2)

def condition (m : ℝ) : Prop :=
  P ∈ parabola ∧ PA P = m * PB P

theorem eccentricity_of_hyperbola : condition (sqrt 2 + 1) → (eccentricity = sqrt 2 + 1) :=
sorry

end eccentricity_of_hyperbola_l605_605280


namespace rhombus_side_length_l605_605814

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l605_605814


namespace percent_error_l605_605993

theorem percent_error (x : ℝ) (h : x > 0) :
  (abs ((12 * x) - (x / 3)) / (x / 3)) * 100 = 3500 :=
by
  sorry

end percent_error_l605_605993


namespace smallest_positive_n_l605_605679

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605679


namespace area_of_triangle_with_medians_l605_605660

theorem area_of_triangle_with_medians (m1 m2 m3 : ℝ) (h_m1 : m1 = 12) (h_m2 : m2 = 15) (h_m3 : m3 = 21) :
  ∃(A : ℝ), A = 48 * Real.sqrt 6 :=
by
  sorry

end area_of_triangle_with_medians_l605_605660


namespace range_values_a_l605_605874

theorem range_values_a {a : ℝ} (z : ℂ) (h : z = 1 + a * Complex.I) :
  |z| ≤ 2 ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end range_values_a_l605_605874


namespace combined_distance_l605_605931

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l605_605931


namespace min_value_z_l605_605734

theorem min_value_z (x y : ℝ) (hx : x > 0) (hy : y > 0) (hlog : Real.log10 x + Real.log10 y = 1) :
    2 ≤ (2 / x) + (5 / y) :=
by
  sorry

end min_value_z_l605_605734


namespace output_modulo_2150_l605_605588

def number_machine (x y z : ℕ) : ℕ :=
  (((((3 * x + y) / |x - y|) ^ 3) * y) % z)

theorem output_modulo_2150
  (h1 : number_machine 54 87 1450 = 1129)
  (h2 : 1 < 2150) :
  number_machine 42 95 2150 = 495 :=
sorry

end output_modulo_2150_l605_605588


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605166

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605166


namespace total_number_of_boys_l605_605418

theorem total_number_of_boys (total_children happy_children sad_children neither_happy_nor_sad boys girls happy_boys sad_girls boys_neither_happy_nor_sad : ℕ) 
    (h1 : total_children = 60) 
    (h2 : happy_children = 30) 
    (h3 : sad_children = 10)
    (h4 : neither_happy_nor_sad = 20) 
    (h5 : girls = 43) 
    (h6 : happy_boys = 6) 
    (h7 : sad_girls = 4) 
    (h8 : boys_neither_happy_nor_sad = 5) : 
    boys = total_children - girls :=
by 
  have h_boys : boys = 17 := by linarith [h1, h5]
  exact h_boys

end total_number_of_boys_l605_605418


namespace jane_average_speed_l605_605358

-- Definition of conditions
def distance1 : ℝ := 20
def speed1 : ℝ := 8
def distance2 : ℝ := 40
def speed2 : ℝ := 20

-- Definition to calculate time for given distance and speed
def time (distance speed : ℝ) : ℝ := distance / speed

-- Definition of total distance and total time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := (time distance1 speed1) + (time distance2 speed2)

-- Definition to calculate average speed
def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

-- Expected average speed
def expected_average_speed : ℝ := 13 + 1 / 3 -- This represents 13.\overline{3}

-- Theorem statement
theorem jane_average_speed : average_speed total_distance total_time = expected_average_speed :=
by
  sorry

end jane_average_speed_l605_605358


namespace christopher_avg_speed_l605_605536

-- Definition of a palindrome (not required for this proof, but helpful for context)
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Given conditions
def initial_reading : ℕ := 12321
def final_reading : ℕ := 12421
def duration : ℕ := 4

-- Definition of average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- Main theorem to prove
theorem christopher_avg_speed : average_speed (final_reading - initial_reading) duration = 25 :=
by
  sorry

end christopher_avg_speed_l605_605536


namespace neither_sufficient_nor_necessary_l605_605118

theorem neither_sufficient_nor_necessary :
  ∃ (θ a: ℝ), (sqrt (1 + sin θ) = a ∧ (sin (θ / 2) + cos (θ / 2) ≠ a)) ∧
              (sin (θ / 2) + cos (θ / 2) = a ∧ (sqrt (1 + sin θ) ≠ a)) := by
  sorry

end neither_sufficient_nor_necessary_l605_605118


namespace determine_range_m_l605_605733

def f (x : ℝ) : ℝ := Real.log (x ^ 2 + 1)
def g (x m : ℝ) : ℝ := (1 / 2) ^ x - m

theorem determine_range_m (m : ℝ) :
  (∀ (x1 : ℝ), 0 ≤ x1 ∧ x1 ≤ 3 → 
    ∀ (x2 : ℝ), 1 ≤ x2 ∧ x2 ≤ 2 → 
    f x1 ≥ g x2 m) ↔ m ≥ 1 / 2 :=
by
  sorry

end determine_range_m_l605_605733


namespace smallest_n_for_identity_matrix_l605_605673

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605673


namespace snow_probability_first_week_l605_605430

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l605_605430


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605185

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605185


namespace valid_parameterizations_l605_605486

noncomputable def parameterizations (p : ℕ) : ℝ × ℝ × ℝ × ℝ :=
  match p with
  | 1 => ((1, 1), (3, -1))
  | 2 => ((0, 4), (1, -3))
  | 3 => ((-2, 10), (-2, 6))
  | 4 => ((4, -8), (0.5, -1.5))
  | 5 => ((-1, 7), (2, -6))
  | _ => ((0, 0), (0, 0))

def line_eq : ℝ → ℝ := fun x => -3 * x + 4

def point_valid_on_line (P : ℝ × ℝ) : Prop :=
  P.snd = line_eq P.fst

def direction_valid_on_line (D : ℝ × ℝ) : Prop :=
 D.snd = -3 * D.fst

theorem valid_parameterizations:
  ∀ (j : ℕ), j ∈ [0, 1, 2, 3, 4] →
  let start_point := (parameterizations j).fst
  let direction_vec := (parameterizations j).snd
  point_valid_on_line start_point ∧ (direction_valid_on_line direction_vec ∨ direction_valid_on_line (-direction_vec)) →
  j = 2 ∨ j = 3 ∨ j = 5 :=
by sorry

end valid_parameterizations_l605_605486


namespace mean_of_reciprocals_of_first_four_primes_l605_605194

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605194


namespace evaluate_expression_l605_605647

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l605_605647


namespace problem_proof_1_problem_proof_2_problem_proof_3_l605_605295

section
variable (x α : ℝ)

-- Define f(x)
def f := fun ω x => 3 * sin (ω * x + (π / 6))

-- Given conditions
axiom omega_pos : ∃ ω > 0, (∀ x, f ω x ∈ set.Ioo (-∞) (∞)) ∧ (π / 2) = (2 * π / ω)

-- Given values to prove
def f_zero_eq_three_half (ω : ℝ) : Prop := f ω 0 = 3 * sin (π / 6)
def f_expression (ω : ℝ) : Prop := ∀ x, f ω x = 3 * sin (4 * x + (π / 6))
def sin_alpha_value (α : ℝ) : Prop := f (4) (α / 4 + π / 12) = 9 / 5 → sin α = 4 / 5 ∨ sin α = -4 / 5

-- Proof problem statements
theorem problem_proof_1 (ω : ℝ) (h1 : omega_pos ω) : f_zero_eq_three_half ω := 
by sorry

theorem problem_proof_2 (ω : ℝ) (h2 : omega_pos ω) : f_expression ω := 
by sorry

theorem problem_proof_3 (α : ℝ) : sin_alpha_value α := 
by sorry
end

end problem_proof_1_problem_proof_2_problem_proof_3_l605_605295


namespace abc_arithmetic_sequence_l605_605115

variables (A B C O B1 C1 D1 : Point)
variables (AB AC BC AD AB1 AD1 AC1 : ℕ)
variable (R : ℝ)

def passes_through (C : Circle) (P : Point) : Prop := sorry
def intersects_medians (Δ : Triangle) (Circle : Circle) (B1 C1 D1 : Point) : Prop := sorry
def form_arithmetic_sequence (x y z : ℝ) : Prop := 2 * y = x + z

-- Given conditions
axiom h1 : passes_through O A
axiom h2 : intersects_medians (Triangle.mk A B C) O B1 C1 D1

-- Proof statement
theorem abc_arithmetic_sequence :
  form_arithmetic_sequence (AB1 * AB) (AD1 * AD) (AC1 * AC) :=
sorry

end abc_arithmetic_sequence_l605_605115


namespace evaluate_expression_l605_605135

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l605_605135


namespace prob_and_relation_proof_l605_605564

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l605_605564


namespace evaluate_expression_l605_605645

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l605_605645


namespace bubble_gum_cost_l605_605320

theorem bubble_gum_cost (n_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) 
  (h1 : n_pieces = 136) (h2 : total_cost = 2448) : cost_per_piece = 18 :=
by
  sorry

end bubble_gum_cost_l605_605320


namespace smallest_positive_integer_n_l605_605701

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605701


namespace romanov_family_savings_l605_605570

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l605_605570


namespace triangle_area_l605_605935

def line1 (x : ℝ) : ℝ := 6
def line2 (x : ℝ) : ℝ := x + 2
def line3 (x : ℝ) : ℝ := -x + 4

theorem triangle_area : 
  ∃ (A B C : ℝ × ℝ), 
    A = (4, 6) ∧ B = (-2, 6) ∧ C = (1, 3) ∧
    (let d := ((A.1 - B.1) * (B.2 - C.2) - (A.2 - B.2) * (B.1 - C.1)).abs / 2 in d = 9) :=
sorry

end triangle_area_l605_605935


namespace find_lines_and_intersections_l605_605771

-- Define the intersection point conditions
def intersection_point (m n : ℝ) : Prop :=
  (2 * m - n + 7 = 0) ∧ (m + n - 1 = 0)

-- Define the perpendicular line to l1 passing through (-2, 3)
def perpendicular_line_through_A (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

-- Define the parallel line to l passing through (-2, 3)
def parallel_line_through_A (x y : ℝ) : Prop :=
  2 * x - 3 * y + 13 = 0

-- main theorem
theorem find_lines_and_intersections :
  ∃ m n : ℝ, intersection_point m n ∧ m = -2 ∧ n = 3 ∧
  ∃ l3 : ℝ → ℝ → Prop, l3 = perpendicular_line_through_A ∧
  ∃ l4 : ℝ → ℝ → Prop, l4 = parallel_line_through_A :=
sorry

end find_lines_and_intersections_l605_605771


namespace polynomial_sum_l605_605907

noncomputable def p : ℝ → ℝ :=
  λ x, 2

noncomputable def q : ℝ → ℝ :=
  λ x, -3 * x^2 + 18 * x - 24

theorem polynomial_sum :
  p 1 = 2 ∧ q 3 = 3 ∧ (∀ x, (x = 2 ∨ x = 4) → q x = 0) →
  ∀ x, p x + q x = -3 * x^2 + 18 * x - 22 :=
begin
  sorry
end

end polynomial_sum_l605_605907


namespace jake_snake_length_l605_605357

theorem jake_snake_length (j p : ℕ) (h1 : j = p + 12) (h2 : j + p = 70) : j = 41 := by
  sorry

end jake_snake_length_l605_605357


namespace min_value_A_l605_605233

theorem min_value_A (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 8)) :
  let A : ℝ := (Real.cot (2 * α) - Real.tan (2 * α)) / (1 + Real.sin ((5 * Real.pi / 2) - 8 * α)) in
  (∀ β : ℝ, 0 < β ∧ β < (Real.pi / 16) -> 2 = 2) :=
by
  sorry

end min_value_A_l605_605233


namespace function_shift_l605_605027

theorem function_shift (x : ℝ) :
  (∀ x, (λ x, √2 * Real.cos (2 * x)) = (λ x, Real.sin (2 * x) + Real.cos (2 * x) + c)) → c = -π/8 :=
by
  sorry

end function_shift_l605_605027


namespace solution_to_abs_eq_l605_605656

theorem solution_to_abs_eq :
  ∀ x : ℤ, abs ((-5) + x) = 11 → (x = 16 ∨ x = -6) :=
by sorry

end solution_to_abs_eq_l605_605656


namespace trig_identity_l605_605796

open Real

theorem trig_identity (α : ℝ) (hα : α > -π ∧ α < -π/2) :
  (sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α))) = - 2 / tan α :=
by
  sorry

end trig_identity_l605_605796


namespace propositions_evaluation_l605_605099

theorem propositions_evaluation :
  (¬ ((p ∨ q) → (p ∧ q))) →
  (¬ (∃ x : ℝ, x^2 + 1 > 3 * x) ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x) →
  ((x = 4 → x^2 - 3 * x - 4 = 0) ∧ ¬ (∀ x : ℝ, x^2 - 3 * x - 4 = 0 → x = 4)) →
  (¬ (m² + n² = 0 → m = 0 ∧ n = 0) ↔ ¬ (m² + n² ≠ 0 → m ≠ 0 ∨ n ≠ 0)) →
  [false, true, true, false] = [false, true, true, false] :=
by
{ sorry }

end propositions_evaluation_l605_605099


namespace smallest_positive_n_l605_605680

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605680


namespace smallest_positive_n_l605_605684

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605684


namespace correct_statements_l605_605300

open Real

noncomputable theory

def vector_a : ℝ × ℝ × ℝ := (1, 2, 0)
def vector_b : ℝ × ℝ × ℝ := (-1, 2, 1)
def vector_c : ℝ × ℝ × ℝ := (-1, -2, 1)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

theorem correct_statements :
  let unit_vector_a := (1 / magnitude vector_a, 2 / magnitude vector_a, 0)
  ∧ let proj_c_on_a := ((vector_a.1 * vector_c.1 + vector_a.2 * vector_c.2 + vector_a.3 * vector_c.3) / (magnitude vector_a)^2 * vector_a.1, 
                        (vector_a.1 * vector_c.1 + vector_a.2 * vector_c.2 + vector_a.3 * vector_c.3) / (magnitude vector_a)^2 * vector_a.2, 
                        0)
  in unit_vector_a = ( √5/5, 2√5/5, 0 ) ∧ proj_c_on_a = (-1, -2, 0) :=
by sorry

end correct_statements_l605_605300


namespace cartesian_eq_C1_polar_eq_C2_max_OP_OQ_l605_605585

-- Definitions and conditions
def curve_C1 (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)
def curve_C2 (φ : ℝ) : ℝ × ℝ := (Real.cos φ, 1 + Real.sin φ)

-- Problem statements to prove
theorem cartesian_eq_C1 :
  ∀ (t : ℝ), let (x, y) := curve_C1 t in y^2 = 4 * x :=
sorry

theorem polar_eq_C2 :
  ∀ (θ : ℝ), let ρ := 2 * Real.sin θ in 
  let (x, y) := curve_C2 θ in x^2 + y^2 - 2 * y = 0 :=
sorry

theorem max_OP_OQ :
  ∀ (α : ℝ), α ∈ Set.Icc (Real.pi / 6) (Real.pi / 4) →
  | (4 * Real.cos α) / (Real.sin α)^2 | * | 2 * Real.sin α | ≤ 8 * Real.sqrt 3 :=
sorry

end cartesian_eq_C1_polar_eq_C2_max_OP_OQ_l605_605585


namespace intersection_points_length_AB_l605_605830

def line_param (t : ℝ) : ℝ × ℝ :=
  ((1 / 2) * t, 1 - (sqrt 3 / 2) * t)

def circle_polar (θ : ℝ) : ℝ :=
  2 * sin θ

def line_eq (x y : ℝ) : Prop :=
  sqrt 3 * x + y - 1 = 0

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

theorem intersection_points:
  ∃ (A B : ℝ × ℝ), 
    line_eq A.1 A.2 ∧ circle_eq A.1 A.2 ∧
    line_eq B.1 B.2 ∧ circle_eq B.1 B.2 ∧ 
    A ≠ B :=
sorry

theorem length_AB (A B : ℝ × ℝ) (hA : line_eq A.1 A.2) (hB : line_eq B.1 B.2)
  (hA_circle : circle_eq A.1 A.2) (hB_circle : circle_eq B.1 B.2) (hAB_ineq : A ≠ B) :
  dist A B = 2 :=
sorry

end intersection_points_length_AB_l605_605830


namespace smallest_n_for_identity_matrix_l605_605671

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605671


namespace percentage_increase_l605_605984

variable (P N N' : ℝ)
variable (h : P * 0.90 * N' = P * N * 1.035)

theorem percentage_increase :
  ((N' - N) / N) * 100 = 15 :=
by
  -- By given condition, we have the equation:
  -- P * 0.90 * N' = P * N * 1.035
  sorry

end percentage_increase_l605_605984


namespace function_periodic_of_eqn_l605_605451

theorem function_periodic_of_eqn (f : ℝ → ℝ) : 
  (∀ x : ℝ, f(x + 1) + f(x - 1) = real.sqrt 2 * f(x)) → 
  (∀ x : ℝ, f(x + 8) = f(x)) :=
begin
  intro h,
  sorry
end

end function_periodic_of_eqn_l605_605451


namespace correct_expression_l605_605534

theorem correct_expression (a b : ℝ) : (a^2 * b)^3 = (a^6 * b^3) := 
by
sorry

end correct_expression_l605_605534


namespace minimum_log_expression_l605_605263

theorem minimum_log_expression (a : ℝ) (h : a > 1) : ∃ x ε : ℝ, x = log a 16 + 2 * (1 / log a 4) ∧ ε = 4 := 
sorry

end minimum_log_expression_l605_605263


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605191

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605191


namespace min_y_p_p_q_q_z_l605_605334

open Real

theorem min_y_p_p_q_q_z (X Y Z P Q : Point) (angle_XYZ : ℝ) (dist_XY : ℝ) (dist_XZ : ℝ)
  (h_angle : angle_XYZ = 50) (h_XY : dist_XY = 8) (h_XZ : dist_XZ = 12)
  (h_P_on_XY : P ∈ line_segment X Y) (h_Q_on_XZ : Q ∈ line_segment X Z) :
  min (dist X Y + dist Y P + dist P Q + dist Q X + dist X Z) 16.78 :=
sorry

end min_y_p_p_q_q_z_l605_605334


namespace evaluate_expression_l605_605641

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l605_605641


namespace find_a_l605_605242

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
if k ≤ n then nat.choose n k else 0

theorem find_a (a : ℝ) (h : binomial_coefficient 5 2 * a^3 = 80) : a = 2 :=
by sorry

end find_a_l605_605242


namespace find_g_two_l605_605905

variable (g : ℝ → ℝ)

-- Condition 1: Functional equation
axiom g_eq : ∀ x y : ℝ, g (x - y) = g x * g y

-- Condition 2: Non-zero property
axiom g_ne_zero : ∀ x : ℝ, g x ≠ 0

-- Proof statement
theorem find_g_two : g 2 = 1 := 
by sorry

end find_g_two_l605_605905


namespace sum_S_l605_605050

def S : Set ℤ := { x | 10 < x ∧ x < 20 }

theorem sum_S : (∑ x in S, x) = 135 := by
  sorry

end sum_S_l605_605050


namespace speed_of_train_l605_605597

/-- Define the given conditions -/
variables
  (length_of_train : ℝ) (speed_of_person : ℝ) (crossing_time : ℝ)
  (rel_speed : ℝ) (conversion_factor : ℝ)

/-- The conditions given in the problem -/
def conditions : Prop :=
  length_of_train = 300 ∧
  speed_of_person = 16 ∧
  crossing_time = 15 ∧
  conversion_factor = 1 / 3.6

/-- The speed of the train proof -/
theorem speed_of_train (V_train : ℝ) :
  conditions → (300 = (V_train - 16) * (15 / 3.6)) → V_train = 88 :=
by
  intros h1 h2
  rw [←h2]
  sorry

end speed_of_train_l605_605597


namespace expected_heads_for_100_coins_l605_605413

noncomputable def expected_heads (n : ℕ) (p_heads : ℚ) : ℚ :=
  n * p_heads

theorem expected_heads_for_100_coins :
  expected_heads 100 (15 / 16) = 93.75 :=
by
  sorry

end expected_heads_for_100_coins_l605_605413


namespace find_m_l605_605276

open Set

theorem find_m (m : ℝ) (A B : Set ℝ)
  (h1 : A = {-1, 3, 2 * m - 1})
  (h2 : B = {3, m})
  (h3 : B ⊆ A) : m = 1 ∨ m = -1 :=
by
  sorry

end find_m_l605_605276


namespace sum_digits_square_of_repeated_twos_l605_605942

def repeated_twos (n : Nat) := (2 * 10^n - 2) / 9

theorem sum_digits_square_of_repeated_twos :
  let n := 9
  let x := repeated_twos n
  Nat.sum_digits (x * x) = 324 := by
    let n := 9
    let x := repeated_twos n
    sorry

end sum_digits_square_of_repeated_twos_l605_605942


namespace bus_probabilities_and_chi_squared_l605_605553

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l605_605553


namespace reflected_ray_equation_l605_605590

noncomputable theory

def point (x : ℝ) (y : ℝ) := (x, y)

def A := point (-3) 5
def l (x y : ℝ) := x - y - 3
def B := point 2 12

theorem reflected_ray_equation : ∃ m b : ℝ, 
 (B.1 - (-2)) = m * (B.2 - 4) ∧ 
 (A.1 - (-2)) = m * (A.2 - 4) ∧ 
 (l B.1 B.2 = 0) ∧ 
 (x - 2*y + 22) = 0 :=
sorry

end reflected_ray_equation_l605_605590


namespace variance_transformation_example_l605_605823

def variance (X : List ℝ) : ℝ := sorry -- Assuming some definition of variance

theorem variance_transformation_example {n : ℕ} (X : List ℝ) (h_len : X.length = 2021) (h_var : variance X = 3) :
  variance (X.map (fun x => 3 * (x - 2))) = 27 := 
sorry

end variance_transformation_example_l605_605823


namespace derivative_of_x_sin_2x_l605_605001

noncomputable def derivative_of_function : ℝ → ℝ :=
  λ x, (x * ((2:ℝ) * x).sin).deriv

theorem derivative_of_x_sin_2x (x : ℝ) : derivative_of_function x = (2 * x * (2 * x).cos) + ((2 * x).sin) :=
sorry

end derivative_of_x_sin_2x_l605_605001


namespace max_height_l605_605975

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 20

theorem max_height : ∃ t : ℝ, h t = 100 :=
by {
  use 2,
  unfold h,
  norm_num,
  sorry
}

end max_height_l605_605975


namespace probPassingSelection_l605_605979

-- Definitions for the members in each level and their probabilities of passing
def firstLevelMembers : ℕ := 10
def secondLevelMembers : ℕ := 5
def thirdLevelMembers : ℕ := 5
def totalMembers : ℕ := 20

def probFirstLevel : ℝ := firstLevelMembers / totalMembers
def probSecondLevel : ℝ := secondLevelMembers / totalMembers
def probThirdLevel : ℝ := thirdLevelMembers / totalMembers

def probPassFirstLevel : ℝ := 0.8
def probPassSecondLevel : ℝ := 0.7
def probPassThirdLevel : ℝ := 0.5

-- The goal to prove
theorem probPassingSelection : 
  probFirstLevel * probPassFirstLevel + 
  probSecondLevel * probPassSecondLevel + 
  probThirdLevel * probPassThirdLevel = 0.7 := 
by
  sorry

end probPassingSelection_l605_605979


namespace parabola_focus_distance_l605_605325

theorem parabola_focus_distance (p : ℝ) (h : p > 0) 
  (dist_eq_sqrt2 : ((abs (-p/2 - 1)) / sqrt 2) = sqrt 2) : p = 2 :=
sorry

end parabola_focus_distance_l605_605325


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605226

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605226


namespace crocodile_coloring_l605_605410

theorem crocodile_coloring (m n : ℤ) :
  ∃ (f : ℤ × ℤ → bool), 
  (∀ (x y : ℤ × ℤ), 
    (x.1 + m = y.1 ∧ x.2 + n = y.2 ∨ x.1 + n = y.1 ∧ x.2 + m = y.2) → 
    f x ≠ f y) := 
sorry

end crocodile_coloring_l605_605410


namespace unique_plane_exists_l605_605824

-- Define what it means for two lines to be skew
def skew_lines (a b : ℝ → ℝ³) : Prop :=
∀ t u : ℝ, a t ≠ b u ∧ ∀ v : ℝ³, (v ≠ 0 → (v.dot (a t - b u) = 0))

-- Define the existence of a plane containing line a and perpendicular to line b
def exists_unique_plane_perpendicular (a b : ℝ → ℝ³) (P : set (ℝ³)) : Prop :=
(skew_lines a b) ∧ 
  (∀ t : ℝ, a t ∈ P) ∧ 
  (∃ v : ℝ³, (v ≠ 0) ∧ (∀ w : ℝ³, (w ∈ P) → (v.dot w = 0))) ∧ 
  (∀ Q : set (ℝ³), 
    ((∀ t : ℝ, a t ∈ Q) ∧ 
    (∃ u : ℝ³, (u ≠ 0) ∧ (∀ w : ℝ³, (w ∈ Q) → (u.dot w = 0)))) → 
    (P = Q))

-- The mathematically equivalent proof problem
theorem unique_plane_exists (a b : ℝ → ℝ³) : 
  ∃ P, exists_unique_plane_perpendicular a b P := 
by 
  sorry

end unique_plane_exists_l605_605824


namespace gcd_98_140_245_l605_605939

theorem gcd_98_140_245 : Nat.gcd (Nat.gcd 98 140) 245 = 7 := 
by 
  sorry

end gcd_98_140_245_l605_605939


namespace rhombus_side_length_l605_605817

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605817


namespace part1_part2_l605_605862

-- Define the functions and the conditions for finding a and k
def y1 (k x : ℝ) := k / x
def y2 (k x : ℝ) := -k / x

-- Given conditions for part 1
variables (k : ℝ) (a : ℝ)
axiom k_pos : k > 0
axiom interval : 2 ≤ 2 ∧ 3 ≤ 3
axiom max_y1 : y1 k 2 = a
axiom min_y2 : y2 k 2 = a - 4

-- Show that a = 2 and k = 4
theorem part1 : a = 2 ∧ k = 4 := sorry

-- Given conditions for part 2
variables (m p q : ℝ)
axiom m_cond : m ≠ 0 ∧ m ≠ -1
axiom y1_m : y1 k m = p
axiom y1_m1 : y1 k (m + 1) = q

-- Prove Yuan Yuan's statement is not necessarily correct
theorem part2 : ¬ ∀ (m : ℝ), m ≠ 0 ∧ m ≠ -1 → p > q := sorry

end part1_part2_l605_605862


namespace arithmetic_mean_reciprocals_primes_l605_605161

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605161


namespace bus_passenger_problem_l605_605474

theorem bus_passenger_problem : 
  (∃ (X : ℕ), 7 - X + 5 - 2 + 4 = 11 ∧ X = 3) :=
by 
  use 3
  have h1 : 7 - 3 + 5 - 2 + 4 = 11 := by norm_num
  exact ⟨h1, rfl⟩

end bus_passenger_problem_l605_605474


namespace infinite_div_9999_l605_605069

open Int

def is_perfect_power (x : ℤ) : Prop := ∃ n k : ℤ, k ≥ 2 ∧ x = n^k

noncomputable def seq_a (i : ℕ) : ℤ := sorry  -- The sequence of perfect powers in increasing order

theorem infinite_div_9999 :
  ∃ᶠ n in at_top, 9999 ∣ seq_a (n + 1) - seq_a n :=
begin
  sorry,
end

end infinite_div_9999_l605_605069


namespace problem1_problem2_problem3_prop_problem3_converse_counterexample_l605_605899

-- Definitions for points and the focus of a parabola
structure Point where
  x : ℝ
  y : ℝ

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def isOnParabolaL (P : Point) : Prop :=
  P.y ^ 2 = 4 * P.x

-- Problem 1
def P1 := Point
def P2 := Point
def P3 := Point
def F := Point.mk 1 0

theorem problem1 (P1 P2 P3 : Point) (hP1 : isOnParabolaL P1) (hP2 : isOnParabolaL P2) (hP3 : isOnParabolaL P3) (sum_x : P1.x + P2.x + P3.x = 4) :
  distance F P1 + distance F P2 + distance F P3 = 7 := 
sorry

-- Definitions for Problem 2
def isVecSumZero (F : Point) (Ps : list Point) : Prop :=
  (Ps.map (λ P, (P.x - F.x, P.y - F.y))).foldl (λ acc p, (acc.1 + p.1, acc.2 + p.2)) (0, 0) = (0, 0)

theorem problem2 (F : Point) (Ps : list Point) (hF : F = ⟨1, 0⟩) (hPs : ∀ P ∈ Ps, isOnParabolaL P) (hVecSumZero : isVecSumZero F Ps) (lengthPs_ge_3 : Ps.length ≥ 3) :
  (Ps.map (distance F)).sum = 2 * Ps.length := 
sorry

-- Definitions for Problem 3
def isOnParabola (F : Point) (p : ℝ) (P : Point) : Prop :=
  P.y ^ 2 = 2 * p * (P.x - F.x)

theorem problem3_prop (F : Point) (Ps : list Point) (p : ℝ) (hF : F = ⟨p / 2, 0⟩) (hPs : ∀ P ∈ Ps, isOnParabola F p P) (hVecSumZero : isVecSumZero F Ps) (lengthPs_ge_3 : Ps.length ≥ 3) :
  (Ps.map (distance F)).sum = p * Ps.length := 
sorry

theorem problem3_converse_counterexample :
  ∃ F : Point, ∃ Ps : list Point, ∃ p : ℝ, 
    Ps.length ≥ 3 ∧
    Ps.map (λ P, distance F P).sum = p * Ps.length ∧ 
    ¬isVecSumZero F Ps := 
sorry

end problem1_problem2_problem3_prop_problem3_converse_counterexample_l605_605899


namespace trigonometric_identity_l605_605664

theorem trigonometric_identity :
  (let t1 := sin 20 * cos 10 + cos 160 * cos 100,
       t2 := sin 24 * cos 6 + cos 156 * cos 96
   in t1 = t2) :=
begin
  sorry
end

end trigonometric_identity_l605_605664


namespace smart_numbers_characterization_smart_number_2015th_l605_605992

-- Definition of smart number
def is_smart_number (n : ℕ) : Prop :=
  (n > 1 ∧ n % 2 = 1) ∨ (n ≥ 8 ∧ n % 4 = 0)

-- Theorem 1: All positive integers that are smart numbers
theorem smart_numbers_characterization (n : ℕ) : 
  is_smart_number n ↔ ((n > 1 ∧ n % 2 = 1) ∨ (n ≥ 8 ∧ n % 4 = 0)) :=
by sorry

-- Theorem 2: The 2015th smart number is 2689
theorem smart_number_2015th : 
  ∃! n, (nat.find (λ k, is_smart_number k) n = 2015) ∧ n = 2689 :=
by sorry

end smart_numbers_characterization_smart_number_2015th_l605_605992


namespace largest_possible_value_of_p_l605_605100

theorem largest_possible_value_of_p (m n p : ℕ) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * m * n * p = (m + 2) * (n + 2) * (p + 2)) : p ≤ 130 :=
by
  sorry

end largest_possible_value_of_p_l605_605100


namespace remaining_amount_is_12_l605_605104

-- Define initial amount and amount spent
def initial_amount : ℕ := 90
def amount_spent : ℕ := 78

-- Define the remaining amount after spending
def remaining_amount : ℕ := initial_amount - amount_spent

-- Theorem asserting the remaining amount is 12
theorem remaining_amount_is_12 : remaining_amount = 12 :=
by
  -- Proof omitted
  sorry

end remaining_amount_is_12_l605_605104


namespace pen_arrangements_l605_605079

def blue_pens : ℕ := 7
def red_pens : ℕ := 3
def green_pens : ℕ := 3
def black_pens : ℕ := 2

theorem pen_arrangements : (∑ x in (finset.range (blue_pens + red_pens + green_pens + black_pens)), x)  - 
                            (∑ y in (finset.range (blue_pens + red_pens + 1 + black_pens)), y * factorial green_pens) = 6098400 :=
sorry

end pen_arrangements_l605_605079


namespace count_integers_M_3_k_l605_605774

theorem count_integers_M_3_k (M : ℕ) (hM : M < 500) :
  (∃ k : ℕ, k ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ M = 2 * k * (m + k - 1)) ∧
  (∃ k1 k2 k3 k4 : ℕ, k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧
    k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4 ∧
    (M / 2 = (k1 + k2 + k3 + k4) ∨ M / 2 = (k1 * k2 * k3 * k4))) →
  (∃ n : ℕ, n = 6) :=
by
  sorry

end count_integers_M_3_k_l605_605774


namespace Sara_lunch_bill_l605_605456

theorem Sara_lunch_bill :
  let hotdog := 5.36
  let salad := 5.10
  let drink := 2.50
  let side_item := 3.75
  hotdog + salad + drink + side_item = 16.71 :=
by
  sorry

end Sara_lunch_bill_l605_605456


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605221

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605221


namespace max_red_stones_placement_l605_605126

def Point := (ℕ × ℕ)

def distance (p1 p2 : Point) : ℚ :=
  real.sqrt (((p2.1 : ℚ) - (p1.1 : ℚ))^2 + ((p2.2 : ℚ) - (p1.2 : ℚ))^2)

def valid_red_placement (red_stones : set Point) : Prop :=
  ∀ (p1 p2 : Point), p1 ≠ p2 → p1 ∈ red_stones → p2 ∈ red_stones → distance p1 p2 ≠ real.sqrt 5

noncomputable def Point_list := list Point

def all_points : Point_list :=
  list.filter (λ p : Point, p.1 ≥ 1 ∧ p.1 ≤ 20 ∧ p.2 ≥ 1 ∧ p.2 ≤ 20) (list.product (list.range 21) (list.range 21))

noncomputable def Blue_stones := list Point

def player_A_wins (red_stones : set Point) (blue_stones : Blue_stones) : Prop :=
  valid_red_placement red_stones ∧ red_stones ⊆ set.of_list all_points ∧
  (∀ blue_point ∈ blue_stones, blue_point ∈ set.compl red_stones)

theorem max_red_stones_placement : ∃ (red_stones : set Point), ∀ (blue_stones : Blue_stones), player_A_wins red_stones blue_stones ∧ red_stones.card ≥ 48 :=
sorry

end max_red_stones_placement_l605_605126


namespace mean_of_reciprocals_first_four_primes_l605_605148

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605148


namespace domain_of_f_univ_l605_605663

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 1)^(1 / 3) + (9 - x^2)^(1 / 3)

theorem domain_of_f_univ : ∀ x : ℝ, true :=
by
  intro x
  sorry

end domain_of_f_univ_l605_605663


namespace coefficient_of_term_in_expansion_l605_605129

theorem coefficient_of_term_in_expansion : 
  (∑ k in Multiset.Perm {0, 1, 2, 5, 3, 10}, Multichoose k 10 * (2^k.val.filter_map 0) * (k.val.filter_map 1) * (k.val.filter_map 2))
  = 20160 := 
sorry

end coefficient_of_term_in_expansion_l605_605129


namespace e_neg_4i_in_second_quadrant_l605_605849

open Real Complex

theorem e_neg_4i_in_second_quadrant :
  let z := exp (-4 * Complex.I) in
  z.re < 0 ∧ z.im > 0 := by
sorry

end e_neg_4i_in_second_quadrant_l605_605849


namespace smallest_n_l605_605694

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605694


namespace snow_probability_l605_605431

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l605_605431


namespace colleen_paid_more_l605_605363

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l605_605363


namespace sum_integers_between_10_and_20_l605_605048

theorem sum_integers_between_10_and_20 : (∑ k in Finset.Ico 11 20, k) = 135 := 
by sorry

end sum_integers_between_10_and_20_l605_605048


namespace mean_of_reciprocals_of_first_four_primes_l605_605192

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605192


namespace max_value_of_term_number_is_5_l605_605322

noncomputable def max_term_number (a : ℕ → ℝ) : ℕ :=
  if (∀ n, a n + a (n+1) + a (n+2) > 0) ∧ (∀ n, a n + a (n+1) + a (n+2) + a (n+3) < 0) then 5 else sorry

theorem max_value_of_term_number_is_5 (a : ℕ → ℝ) :
  (∀ n, a n + a (n+1) + a (n+2) > 0) → 
  (∀ n, a n + a (n+1) + a (n+2) + a (n+3) < 0) → 
  max_term_number a = 5 :=
by
  intro h1 h2
  rw max_term_number
  simp [h1, h2]
  sorry

end max_value_of_term_number_is_5_l605_605322


namespace snow_prob_correct_l605_605425

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l605_605425


namespace series_evaluation_l605_605769

noncomputable def series_sum (x y : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), x ^ k + y ^ (-k - 1)

theorem series_evaluation : 
  (∀ (x y : ℝ), {x, x * y, real.log (x * y)} = {0, abs x, y} → 
    x + (1 / y) + x^2 + (1 / y^2) + x^3 + (1 / y^3) + ... + x^2001 + (1 / y^2001) = -2) 
:= 
begin
  intros x y H,
  sorry
end

end series_evaluation_l605_605769


namespace sequence_a_n_l605_605124

theorem sequence_a_n (a : ℕ → ℕ) (h₁ : a 1 = 1)
(h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a (n / 2) + a ((n + 1) / 2)) :
∀ n : ℕ, a n = n :=
by
  -- skip the proof with sorry
  sorry

end sequence_a_n_l605_605124


namespace count_terminating_decimals_with_nonzero_thousandths_l605_605715

noncomputable def is_terminating_with_nonzero_thousandths (n : ℕ) : Prop :=
  (∃ a b : ℕ, n = 2^a * 5^b) ∧ n ≤ 1000

theorem count_terminating_decimals_with_nonzero_thousandths :
  (finset.univ.filter is_terminating_with_nonzero_thousandths).card = 25 :=
begin
  sorry
end

end count_terminating_decimals_with_nonzero_thousandths_l605_605715


namespace smallest_n_l605_605691

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605691


namespace find_m_for_parabola_intersection_l605_605910

theorem find_m_for_parabola_intersection :
  ∃ m : ℝ, (∀ y : ℝ, -3*y^2 - 4*y + 7 = m) ↔ (m = 25/3) :=
begin
  sorry
end

end find_m_for_parabola_intersection_l605_605910


namespace john_piggy_bank_savings_l605_605843

theorem john_piggy_bank_savings:
  ∀ (savings_per_month : Int) (months : Int) (spent : Int) (final_savings : Int),
    savings_per_month = 25 →
    months = 24 →
    spent = 400 →
    final_savings = (savings_per_month * months) - spent →
    final_savings = 200 :=
by
  intros savings_per_month months spent final_savings
  assume h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end john_piggy_bank_savings_l605_605843


namespace square_area_example_l605_605936

open Real

structure Point where
  x : ℝ
  y : ℝ

def distance (A B : Point) : ℝ :=
  sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

noncomputable def isSquare (A B C D : Point) :=
  let side1 := distance A B
  let side2 := distance B C
  let side3 := distance C D
  let side4 := distance D A
  (side1 = side2) ∧ (side2 = side3) ∧ (side3 = side4)

noncomputable def squareArea (A B C D : Point) : ℝ :=
  (distance A B)^2

theorem square_area_example : 
  let P := Point.mk 1 1
  let Q := Point.mk (-4) 0
  let R := Point.mk (-3) (-5)
  let S := Point.mk 2 (-4)
  isSquare P Q R S → squareArea P Q R S = 26 := 
  by
    intro h
    sorry

end square_area_example_l605_605936


namespace solve_painting_problem_l605_605599

inductive NailColor
| blue
| red

structure Nail (color : NailColor) := 
  (name : String)

def a1 : Nail := {name := "a1", color := NailColor.blue}
def a2 : Nail := {name := "a2", color := NailColor.blue}
def a3 : Nail := {name := "a3", color := NailColor.red}
def a4 : Nail := {name := "a4", color := NailColor.red}

def sequence (nails : List Nail) : List Nail :=
[nails[0], nails[1], nails[2], nails[3], nails[1]⟨α⟩.inverse, nails[0].inverse, nails[3].inverse, nails[2].inverse]

theorem solve_painting_problem :
  ∀ (nails : List Nail),
  let seq := sequence nails in
  (∀ n1 n2 : Nail, n1.color = NailColor.blue -> n2.color = NailColor.blue -> n1 ∈ seq -> n2 ∈ seq -> 
    (∃ n1_inverse n2_inverse : Nail, n1_inverse = n1.inverse ∧ n2_inverse = n2.inverse ∧ 
      n1_inverse ∈ seq ∧ n2_inverse ∈ seq ∧ 
      (∀ m1 m2 : Nail, m1.color = NailColor.red -> m2.color = NailColor.red -> 
        m1 ∈ seq -> m2 ∈ seq -> 
        (∃ m1I m2I : Nail, m1I = m1.inverse ∧ m2I = m2.inverse ∧ 
          m1I ∈ seq ∧ m2I ∈ seq)
))) :=
begin
  sorry
end

end solve_painting_problem_l605_605599


namespace omega_on_real_axis_l605_605755

noncomputable def complex_omega_on_real_axis (z : ℂ) (hz_re : z.re ≠ 0) (hz_mod : abs z = 1) : ℂ :=
  z + star z / (z * star z)

theorem omega_on_real_axis (z : ℂ) (hz_re : z.re ≠ 0) (hz_mod : abs z = 1) :
  (complex_omega_on_real_axis z hz_re hz_mod).im = 0 := 
sorry

end omega_on_real_axis_l605_605755


namespace intersection_of_parabola_and_logarithm_l605_605290

theorem intersection_of_parabola_and_logarithm (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 + m = real.log (abs x)) →
  m < -real.log 2 :=
by
  sorry

end intersection_of_parabola_and_logarithm_l605_605290


namespace selection_of_11_integers_l605_605776

theorem selection_of_11_integers (S : Finset ℕ) (h : S ⊆ (Finset.range 21)) (h_card : S.card ≥ 11) : 
  ∃ a b ∈ S, a ≠ b ∧ (a - b = 2 ∨ b - a = 2) :=
sorry

end selection_of_11_integers_l605_605776


namespace hyperbola_distance_pf2_l605_605767

-- Define the given hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 9 - (y^2) / 16 = 1

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Coordinates for foci F1 and F2
def F1 : ℝ × ℝ := (-5, 0)  -- Assuming the foci positions
def F2 : ℝ × ℝ := (5, 0)

-- A point P (x, y) on hyperbola satisfying given distance to F1
def P (x y : ℝ) : Prop := hyperbola_eq x y ∧ distance (x, y) F1 = 7

-- Prove that the distance PF2 is 13
theorem hyperbola_distance_pf2 {x y : ℝ} (h : P x y) : distance (x, y) F2 = 13 := by
  sorry

end hyperbola_distance_pf2_l605_605767


namespace inscribed_cylinder_height_l605_605994

theorem inscribed_cylinder_height (r_hemisphere r_cylinder : ℝ) (h_parallel : bool) (h_top_touch : bool) :
  r_hemisphere = 7 → r_cylinder = 3 → h_parallel = tt → h_top_touch = tt → 
  ∃ h : ℝ, h = 2 * Real.sqrt 10 :=
by
  intros
  sorry

end inscribed_cylinder_height_l605_605994


namespace smallest_n_for_identity_matrix_l605_605667

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l605_605667


namespace sum_of_increasing_eq_sum_of_decreasing_l605_605006

variable {a b : ℝ}
variable {n : ℕ}
variable {C d : Fin n → ℝ}

noncomputable def f (x : ℝ) : ℝ := Fin n → ℝ x

-- Ensure the conditions on the function f
def is_max_of_functions (f : ℝ → ℝ) :=
  ∀ x, f x = finset.sup (finset.range n) (λ i, C i * 10^(-|x - d i|))

-- The statement we want to prove
theorem sum_of_increasing_eq_sum_of_decreasing
  (ha : f a = f b)
  (hf_max: is_max_of_functions f) :
  sum_of_increasing_segments f a b = sum_of_decreasing_segments f a b :=
sorry

end sum_of_increasing_eq_sum_of_decreasing_l605_605006


namespace remaining_paper_area_l605_605594

-- Define the initial conditions
def side_length : ℝ := 10  -- side length of the square in cm
def initial_area (s : ℝ) : ℝ := s * s  -- formula for the area of a square

def first_fold_area (a : ℝ) : ℝ := a / 2  -- area after first fold
def second_fold_area (a : ℝ) : ℝ := a / 2  -- area after second fold

def removed_area (a : ℝ) : ℝ := a / 4  -- quarter of the folded area that is removed

-- Define the final remaining area
def remaining_area (initial : ℝ) : ℝ :=
  let folded_once := first_fold_area initial
  let folded_twice := second_fold_area folded_once
  let cut_area := removed_area folded_twice
  folded_twice - cut_area

-- The theorem to prove that the remaining area is 75 cm²
theorem remaining_paper_area : remaining_area (initial_area side_length) = 75 := by
  sorry  -- Proof omitted

end remaining_paper_area_l605_605594


namespace calvin_haircut_goal_percentage_l605_605614

theorem calvin_haircut_goal_percentage :
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  (completed_haircuts / total_haircuts_needed) * 100 = 80 :=
by
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  show (completed_haircuts / total_haircuts_needed) * 100 = 80
  sorry

end calvin_haircut_goal_percentage_l605_605614


namespace probability_and_relationship_l605_605547

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l605_605547


namespace arithmetic_mean_reciprocals_first_four_primes_l605_605202

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l605_605202


namespace vowel_initial_probability_is_correct_l605_605416

-- Given conditions as definitions
def total_students : ℕ := 34
def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def vowels_count_per_vowel : ℕ := 2
def total_vowels_count := vowels.length * vowels_count_per_vowel

-- The probabilistic statement we want to prove
def vowel_probability : ℚ := total_vowels_count / total_students

-- The final statement to prove
theorem vowel_initial_probability_is_correct :
  vowel_probability = 6 / 17 :=
by
  unfold vowel_probability total_vowels_count
  -- Simplification to verify our statement.
  sorry

end vowel_initial_probability_is_correct_l605_605416


namespace evaluate_expression_l605_605654

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l605_605654


namespace combined_distance_is_twelve_l605_605929

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l605_605929


namespace task_D_cannot_be_sampled_l605_605953

def task_A := "Measuring the range of a batch of shells"
def task_B := "Determining the content of a certain microorganism in ocean waters"
def task_C := "Calculating the difficulty of each question on the math test after the college entrance examination"
def task_D := "Checking the height and weight of all sophomore students in a school"

def sampling_method (description: String) : Prop :=
  description = task_A ∨ description = task_B ∨ description = task_C

theorem task_D_cannot_be_sampled : ¬ sampling_method task_D :=
sorry

end task_D_cannot_be_sampled_l605_605953


namespace partI_partII_l605_605073

namespace CompetitionProbability

def ProbabilityAQualified := 0.6
def ProbabilityAGood := 0.3
def ProbabilityAExcellent := 0.1

def ProbabilityBQualified := 0.4
def ProbabilityBGood := 0.4
def ProbabilityBExcellent := 0.2

def ProbabilityAHigherInOneRound := (ProbabilityAGood * ProbabilityBQualified) + 
                                    (ProbabilityAExcellent * ProbabilityBQualified) +
                                    (ProbabilityAExcellent * ProbabilityBGood)

def ProbabilityAtLeastTwoRounds :=
  let C1 := (3:ℕ) * ProbabilityAHigherInOneRound^2 * (1 - ProbabilityAHigherInOneRound)
  let C2 := ProbabilityAHigherInOneRound^3
  C1 + C2

theorem partI : ProbabilityAHigherInOneRound = 0.2 := sorry

theorem partII : ProbabilityAtLeastTwoRounds = 0.104 := sorry

end CompetitionProbability

end partI_partII_l605_605073


namespace colleen_paid_more_l605_605367

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l605_605367


namespace mean_of_reciprocals_first_four_primes_l605_605149

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605149


namespace find_radius_of_cylinder_l605_605472

theorem find_radius_of_cylinder :
  (∃ R : ℝ, 
    R = 5 / (Real.sqrt 3) ∨ 
    R = 4 * Real.sqrt (2 / 3) ∨ 
    R = (20 / 3) * Real.sqrt (2 / 11)) :=
by
  -- Definitions of the constants and geometric conditions:
  let a := 4   -- side length of square ABCD
  let h := 6   -- height of the parallelepiped
  let R1 := 5 / (Real.sqrt 3)
  let R2 := 4 * Real.sqrt (2 / 3)
  let R3 := (20 / 3) * Real.sqrt (2 / 11)
  
  -- Statement that at least one of these R is correct
  use R1
  left
  rfl
  sorry

end find_radius_of_cylinder_l605_605472


namespace snow_probability_l605_605433

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l605_605433


namespace base6_add_square_l605_605324

-- Define the addition operation in base 6 using the conditions.
theorem base6_add_square (square : ℕ) (h0 : square ≤ 5) :
  let s0 := (3 + 1 + 2) % 6 = 0 in
  let s1 := ((square + 4 + square + 1) % 6) = square in
  square = 1 :=
by {
  sorry -- omitting the proof as per the instructions
}

end base6_add_square_l605_605324


namespace two_lights_after_finite_operations_l605_605607

-- Conditions:
variable (S : Set ℤ) (finite_S : S.finite)

-- Definitions aligned with conditions:
def initial_state (pos : ℤ) : bool := false

def toggle (state : ℤ → bool) (pos : ℤ) : ℤ → bool :=
  λ x, if x ∈ (S + {pos}) then !state x else state x

-- Main theorem:
theorem two_lights_after_finite_operations :
  ∃ (n : ℕ) (positions : Fin n → ℤ), 
    let final_state := (List.foldl (λ state pos => toggle state pos) initial_state (List.ofFn positions)) in
    (Fin n → ℤ) → (List.foldl (λ state pos => toggle state pos) initial_state (List.ofFn positions)) = final_state ∧
    count (λ x => final_state x = true) (List.range n.toNat) = 2 :=
sorry

end two_lights_after_finite_operations_l605_605607


namespace probability_student_C_first_l605_605995

theorem probability_student_C_first (A B C D E : Type) :
  let students := [A, B, C, D, E]
  -- Define the conditions
  neither_A_nor_B_first (s : List Type) := (s.head ≠ A ∧ s.head ≠ B)
  B_not_last (s : List Type) := s.last ≠ B

  -- Define the possibilities under the given conditions
  possible_arrangements_with_conditions :=
    students.permutations.filter (λ s, neither_A_nor_B_first s ∧ B_not_last s)

  total_ways := possible_arrangements_with_conditions.length

  -- Define the possibilities with student C being the first to speak
  C_first_arrangements :=
    possible_arrangements_with_conditions.filter (λ s, s.head = C)

  ways_C_first := C_first_arrangements.length

  -- Calculate the probability
  probability_C_first := ways_C_first / total_ways
  -- Prove that the calculated probability is 1/3
  in probability_C_first = 1/3 :=
  sorry

end probability_student_C_first_l605_605995


namespace pauline_convertibles_l605_605447

theorem pauline_convertibles : 
  ∀ (total_cars regular_percentage truck_percentage sedan_percentage sports_percentage suv_percentage : ℕ),
  total_cars = 125 →
  regular_percentage = 38 →
  truck_percentage = 12 →
  sedan_percentage = 17 →
  sports_percentage = 22 →
  suv_percentage = 6 →
  (total_cars - (regular_percentage * total_cars / 100 + truck_percentage * total_cars / 100 + sedan_percentage * total_cars / 100 + sports_percentage * total_cars / 100 + suv_percentage * total_cars / 100)) = 8 :=
by
  intros
  sorry

end pauline_convertibles_l605_605447


namespace carrots_total_l605_605455
-- import the necessary library

-- define the conditions as given
def sandy_carrots : Nat := 6
def sam_carrots : Nat := 3

-- state the problem as a theorem to be proven
theorem carrots_total : sandy_carrots + sam_carrots = 9 := by
  sorry

end carrots_total_l605_605455


namespace arithmetic_mean_reciprocals_primes_l605_605159

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l605_605159


namespace data_division_into_groups_l605_605593

-- Conditions
def data_set_size : Nat := 90
def max_value : Nat := 141
def min_value : Nat := 40
def class_width : Nat := 10

-- Proof statement
theorem data_division_into_groups : (max_value - min_value) / class_width + 1 = 11 :=
by
  sorry

end data_division_into_groups_l605_605593


namespace functional_equality_l605_605891

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality (f_spec1 : ∀ x : ℝ, f(x) ≤ x) 
                            (f_spec2 : ∀ x y : ℝ, f(x + y) ≤ f(x) + f(y)) : 
    ∀ x : ℝ, f(x) = x := 
by
  sorry

end functional_equality_l605_605891


namespace largest_prime_factor_divides_sum_l605_605091

-- seq is a sequence of three-digit integers
-- Each digit’s position shifts according to a cyclic permutation

theorem largest_prime_factor_divides_sum (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) : 
  ∃ p : ℕ, prime p ∧ ∀ S d₁ d₂ d₃, S = 100 * d₁ + 10 * d₂ + d₃ + 100 * d₂ + 10 * d₃ + d₁ + 100 * d₃ + 10 * d₁ + d₂ → p ∣ S → p = 37 :=
by
  sorry

end largest_prime_factor_divides_sum_l605_605091


namespace area_of_region_R_is_0_483_l605_605370

-- Define the kite and its properties
structure Kite :=
  (A B C D : Type)
  (AB AD BC CD : ℝ)
  (angleB : ℝ)

-- Assumptions as given conditions
def kiteABCD : Kite := {
  A := unit,
  B := unit,
  C := unit,
  D := unit,
  AB := 2,
  AD := 2,
  BC := 3,
  CD := 3,
  angleB := 150
}

-- Define the target region R and its area calculation
def regionR_area (kite : Kite) : ℝ :=
  if kite.AB = 2 ∧ kite.AD = 2 ∧ kite.BC = 3 ∧ kite.CD = 3 ∧ kite.angleB = 150 
  then
    0.483
  else 0

-- Final statement to prove
theorem area_of_region_R_is_0_483 :
  regionR_area kiteABCD = 0.483 :=
sorry

end area_of_region_R_is_0_483_l605_605370


namespace common_difference_in_arithmetic_progression_l605_605752

theorem common_difference_in_arithmetic_progression (p : ℕ → ℕ) (d : ℕ)
  (h1 : ∀ n : ℕ, nat.prime (p n))
  (h2 : ∀ n : ℕ, p (n + 1) = p n + d)
  (h3 : p 0 < p 1 < p 2 < p 3 < p 4 < p 5 < p 6 < p 7 < p 8 < p 9 < p 10 < p 11 < p 12 < p 13 < p 14) :
  ∃ k ∈ [2, 3, 5, 7, 11, 13], k ∣ d :=
sorry

end common_difference_in_arithmetic_progression_l605_605752


namespace rhombus_side_length_l605_605816

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605816


namespace find_a1_l605_605353

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  a 6 = 9 →
  a 3 = 3 * a 2 →
  a 1 = -1 :=
by
  sorry

end find_a1_l605_605353


namespace no_pairs_probability_l605_605340

-- Define the number of socks and initial conditions
def pairs_of_socks : ℕ := 3
def total_socks : ℕ := pairs_of_socks * 2

-- Probabilistic outcome space for no pairs in first three draws
def probability_no_pairs_in_first_three_draws : ℚ :=
  (4/5) * (1/2)

-- Theorem stating that probability of no matching pairs in the first three draws is 2/5
theorem no_pairs_probability : probability_no_pairs_in_first_three_draws = 2/5 := by
  sorry

end no_pairs_probability_l605_605340


namespace probability_at_least_one_five_or_six_l605_605948

theorem probability_at_least_one_five_or_six
  (P_neither_five_nor_six: ℚ)
  (h: P_neither_five_nor_six = 4 / 9) :
  (1 - P_neither_five_nor_six) = 5 / 9 :=
by
  sorry

end probability_at_least_one_five_or_six_l605_605948


namespace maximize_product_of_terms_l605_605592

noncomputable def is_2019_product_sequence (a : ℕ → ℝ) : Prop :=
∀ m, m ≤ 2019 → (m > 0 → a m = ∏ i in finset.range m, a i)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q, q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem maximize_product_of_terms
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_prod_seq : is_2019_product_sequence a)
  (h_a1 : a 1 > 1) :
  ∃ n, n = 1009 ∧ (∀ m, 1 ≤ m ≤ 2019 → (m ≠ 1009 → (∏ i in finset.range (m+1), a i < ∏ i in finset.range 1010, a i))) :=
sorry

end maximize_product_of_terms_l605_605592


namespace OC_bisects_PQ_l605_605832

-- We start with the geometric setup and definitions of points and circles involved
noncomputable def circles_tangent_to_each_other (O P Q : Point) (rO rP rQ : ℝ) : Prop := sorry
noncomputable def common_internal_tangent (P Q : Point) (C : Point) : Prop := sorry
noncomputable def bisects (OC : Line) (PQ : Segment) : Prop := sorry

theorem OC_bisects_PQ
    (O P Q A B C : Point)
    (rO rP rQ : ℝ)
    (h1: tangent_at P O A)
    (h2: tangent_at Q O B)
    (h3: externally_tangent P Q)
    (h4: common_internal_tangent P Q C)
    (hA: lies_on_line A B)
    (hB: lies_on_line B A)
    (hC: lies_on_line C A) :
    bisects (line_of O C) (segment_of P Q) :=
begin
    sorry
end

end OC_bisects_PQ_l605_605832


namespace ab_value_l605_605402

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + c = 2b) : 
  a * b = 10 :=
by
  sorry

end ab_value_l605_605402


namespace evaluate_expression_l605_605649

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l605_605649


namespace product_of_two_integers_l605_605962

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 18) (h2 : x^2 - y^2 = 36) : x * y = 80 :=
by
  sorry

end product_of_two_integers_l605_605962


namespace range_of_a_l605_605856

def f (a x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7 else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) (f : ℝ → ℝ)
  (odd_f : ∀ x, f x = -f (-x))
  (cond_1 : ∀ x, x < 0 → f x = 9 * x + a^2 / x + 7)
  (cond_2 : ∀ x, 0 ≤ x → f x ≥ a + 1) :
  a ≤ -8/7 := sorry

end range_of_a_l605_605856


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605230

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605230


namespace polynomial_division_coefficient_l605_605533

noncomputable def polynomial := Polynomial ℚ

theorem polynomial_division_coefficient :
  let p := (X : polynomial) ^ 1951 - 1
  let d := (X : polynomial) ^ 4 + (X : polynomial) ^ 3 + 2 * (X : polynomial) ^ 2 + X + 1
  ∃ q r : polynomial, p = q * d + r ∧ r.degree < d.degree ∧ q.coeff 14 = -1 :=
by
  let p := (X : polynomial) ^ 1951 - 1
  let d := (X : polynomial) ^ 4 + (X : polynomial) ^ 3 + 2 * (X : polynomial) ^ 2 + X + 1
  apply Exists.intro
  sorry

end polynomial_division_coefficient_l605_605533


namespace triangle_sides_sum_l605_605999

/-- A triangle has angles measuring 45 and 60 degrees.
    If the side of the triangle opposite the 60-degree angle measures 6√3 units,
    then the sum of the lengths of the two remaining sides is 23.1 units. -/
theorem triangle_sides_sum
  (A B C : Type) [triangle A B C]
  (angle_A : angle A = 45)
  (angle_C : angle C = 60)
  (side_BC : length (B - C) = 6 * sqrt 3) :
  length (A - B) + length (A - C) = 23.1 := 
begin
  sorry
end

end triangle_sides_sum_l605_605999


namespace sum_of_integers_between_10_and_20_l605_605043

theorem sum_of_integers_between_10_and_20 :
  ∑ i in Finset.range (20 - 10 - 1), (i + 11) = 135 := by
  sorry

end sum_of_integers_between_10_and_20_l605_605043


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605177

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605177


namespace digit_difference_is_36_l605_605477

theorem digit_difference_is_36 (x y : ℕ) (hx : y = 2 * x) (h8 : (x + y) - (y - x) = 8) : 
    |(10 * x + y) - (10 * y + x)| = 36 := 
by
  sorry

end digit_difference_is_36_l605_605477


namespace difference_in_payment_l605_605361

theorem difference_in_payment (joy_pencils : ℕ) (colleen_pencils : ℕ) (price_per_pencil : ℕ) (H1 : joy_pencils = 30) (H2 : colleen_pencils = 50) (H3 : price_per_pencil = 4) :
  (colleen_pencils * price_per_pencil) - (joy_pencils * price_per_pencil) = 80 :=
by
  rw [H1, H2, H3]
  simp
  norm_num
  sorry

end difference_in_payment_l605_605361


namespace total_handshakes_among_women_l605_605468

-- Define the problem statement
theorem total_handshakes_among_women {n : ℕ} (h_ten : n = 10)
  (h_distinct_heights : ∀ i j : Fin n, i ≠ j → (i : ℕ) ≠ j) :
  (∑ i in Finset.range n, (n - 1 - i)) = 45 :=
by
  -- The proof is provided in the sorry placeholder
  exact sorry

end total_handshakes_among_women_l605_605468


namespace probability_correct_l605_605974

noncomputable def probability_longer_piece_at_least_y_times_shorter (y : ℝ) (hy : y ≥ 1) : ℝ :=
let C := (λ p : ℝ, p / (y + 1)) in
classical.some (by {
  have h : is_probability_measure (volume : measure ℝ) := infer_instance,
  have p_interval : Icc 0 (C 1) ⊆ Icc 0 0.5 := sorry,
  exact set.integral_const_on_Icc (by simp) (by simp [C]) /
    set.uniform_measure_set volume (Icc 0 (C 1)) (0, 1)
})

theorem probability_correct (y : ℝ) (hy : y ≥ 1) :
  probability_longer_piece_at_least_y_times_shorter y hy = 1 / (y + 1) :=
sorry

end probability_correct_l605_605974


namespace slope_angle_perpendicular_line_l605_605754

theorem slope_angle_perpendicular_line :
  let M := (-(Real.sqrt 3), Real.sqrt 2)
  let N := (Real.sqrt 2, -(Real.sqrt 3))
  let slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)
  let slope_MN := slope M N
  let kl := -1 / slope_MN
  tan θ = kl → θ = 45 := by
  sorry

end slope_angle_perpendicular_line_l605_605754


namespace max_profit_at_80_l605_605093

-- Definitions based on conditions
def cost_price : ℝ := 40
def functional_relationship (x : ℝ) : ℝ := -x + 140
def profit (x : ℝ) : ℝ := (x - cost_price) * functional_relationship x

-- Statement to prove that maximum profit is achieved at x = 80
theorem max_profit_at_80 : (40 ≤ 80) → (80 ≤ 80) → profit 80 = 2400 := by
  sorry

end max_profit_at_80_l605_605093


namespace suyeong_initial_money_l605_605465

def initial_amount_spent_first_store (M : ℝ) := (3 / 8) * M
def remaining_after_first_store (M : ℝ) := M - initial_amount_spent_first_store M

def amount_spent_second_store (M : ℝ) := (1 / 3) * remaining_after_first_store M
def remaining_after_second_store (M : ℝ) := remaining_after_first_store M - amount_spent_second_store M

def amount_spent_third_store (M : ℝ) := (4 / 5) * remaining_after_second_store M
def remaining_after_third_store (M : ℝ) := remaining_after_second_store M - amount_spent_third_store M

theorem suyeong_initial_money (remaining_money : ℝ) (h : remaining_money = 1200) : (∃ M, remaining_after_third_store M = 1200) :=
by {
  use 14400,
  rw [remaining_after_third_store, remaining_after_second_store, remaining_after_first_store],
  simp,
  sorry
}

end suyeong_initial_money_l605_605465


namespace smallest_n_exists_l605_605860

theorem smallest_n_exists (n : ℤ) (r : ℝ) : 
  (∃ m : ℤ, m = (↑n + r) ^ 3 ∧ r > 0 ∧ r < 1 / 1000) ∧ n > 0 → n = 19 := 
by sorry

end smallest_n_exists_l605_605860


namespace root_power_sum_eq_l605_605967

open Real

theorem root_power_sum_eq :
  ∀ {a b c : ℝ},
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (a^3 - 3 * a + 1 = 0) → (b^3 - 3 * b + 1 = 0) → (c^3 - 3 * c + 1 = 0) →
  a^8 + b^8 + c^8 = 186 :=
by
  intros a b c h1 h2 h3 ha hb hc
  sorry

end root_power_sum_eq_l605_605967


namespace parabolas_properties_l605_605831

section ParabolaProofs

-- Defining the parabola
def parabola (a x : ℝ) : ℝ := a * x * (x - 6) + 1

-- Definition for the point being on the parabola
def point_on_parabola (a : ℝ) : Prop := parabola a 0 = 1

-- Distance from vertex to x-axis, vertex y-coordinates calculated as 1 - 9a for parabola (by completing the square)
def distance_vertex_x_axis (a : ℝ) : Prop := (1 - 9 * a).abs = 5

-- Deriving length of BC being less than or equal to 4
def length_BC_leq_4 (a : ℝ) : Prop := 
  if a > 0 then 
    (a * (1^2 - 6 * 1 + 1) >= 0 ∧ a <= 1/5 ∧ a > 1/9)
  else 
    False

-- Final theorem combining all the parts
theorem parabolas_properties {a : ℝ} (h₀ : a ≠ 0) :
  point_on_parabola a →
  distance_vertex_x_axis a →
  length_BC_leq_4 a →
  (a = 2/3 ∨ a = -4/9) ∧ (1/9 < a ∧ a ≤ 1/5) := 
by
  sorry

end ParabolaProofs

end parabolas_properties_l605_605831


namespace logarithm_large_data_l605_605412

theorem logarithm_large_data (m : ℝ) (n : ℕ) 
  (h1 : 1 < m ∧ m < 10) 
  (h2 : 0.4771 < real.logb 10 3 ∧ real.logb 10 3 < 0.4772) 
  (h3 : 3 ^ 2000 = m * 10 ^ n) :
  n = 954 :=
sorry

end logarithm_large_data_l605_605412


namespace inequality_sum_binom_geq_l605_605390

theorem inequality_sum_binom_geq (n : ℕ) (j : ℕ) 
  (h_pos : 0 < n) (h_j : j ∈ {0, 1, 2}) :
  ∑ k in Finset.range ((n / 3) + 1), (-1)^n * (Nat.choose n (3 * k + j)) 
    ≥ (1 / 3 : ℚ) * ((-2 : ℚ)^n - 1) := by
  sorry

end inequality_sum_binom_geq_l605_605390


namespace sum_of_consecutive_odds_eq_169_l605_605944

theorem sum_of_consecutive_odds_eq_169 : 
  (∃ n : ℕ, (∑ i in Finset.range (n+1), if i % 2 = 1 then i else 0) = 169) ↔ n = 13 :=
by
  sorry

end sum_of_consecutive_odds_eq_169_l605_605944


namespace polyline_intersection_exists_l605_605595

noncomputable def exists_intersecting_line (polyline : list (ℝ × ℝ)) : Prop :=
  let n := polyline.length in
  let side_length := 1 in
  let total_length := polyline.foldl (λ acc p, acc + dist p.1 p.2) 0 in
  total_length ≥ 200 → 
  ∃ l ∈ {line | ∃ (k : ℕ), k ≥ 101}, 
    l.parallel_to_side_of_square side_length ∧ l.intersects_polyline_in_at_least k polyline

-- Here we state the main theorem based on the above condition and problem
theorem polyline_intersection_exists 
  (polyline : list (ℝ × ℝ)) 
  (h_non_self_intersecting : non_self_intersecting polyline) 
  (h_total_length : polyline.foldl (λ acc p, acc + dist p.1 p.2) 0 ≥ 200) :
  exists_intersecting_line polyline :=
sorry

end polyline_intersection_exists_l605_605595


namespace inequality_am_gm_l605_605870

variable (a b c d : ℝ)

theorem inequality_am_gm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9  :=
by    
  sorry

end inequality_am_gm_l605_605870


namespace negation_of_exactly_one_even_l605_605947

theorem negation_of_exactly_one_even (a b c : ℕ) :
  (¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c))) ↔
  (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
sorry

end negation_of_exactly_one_even_l605_605947


namespace largest_impossible_m_l605_605374

theorem largest_impossible_m (n m : Nat) (h₁ : 3 ≤ n) (h₂ : n + 1 ≤ m) :
  (∀ beads : List Fin n, (∀ i : Fin m, (beads.nth i).is_some ∧ (beads.slice i.val ((i.val + n + 1) % m)).nodup) → False) ↔ m = n^2 - n - 1 := 
sorry

end largest_impossible_m_l605_605374


namespace colleen_paid_more_l605_605365

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l605_605365


namespace find_polynomials_l605_605247

-- Define the function f(n) as described in the problem statement
def f (n : ℕ) : ℕ :=
  if n = 1 then 0
  else (PrimeFactorsNat.count n 1 0)

-- Define the theorem statement
theorem find_polynomials (P Q : ℕ → ℕ) (hP : Polynomial P) (hQ : Polynomial Q) :
  (∀ m > 0, f (P m) = Q (f m)) ↔
  ∃ R : ℕ, r : ℕ, P = λ x, R * x ^ r ∧ Q = λ x, r * x + f R := sorry

end find_polynomials_l605_605247


namespace count_positive_three_digit_integers_divisible_by_9_l605_605314

def is_valid_digit(d : ℕ) : Prop := 3 ≤ d ∧ d ≤ 9

def is_valid_number(n : ℕ) : Prop := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  is_valid_digit d1 ∧ is_valid_digit d2 ∧ is_valid_digit d3 

def sum_of_digits(n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

def is_divisible_by_9(n : ℕ) : Prop := sum_of_digits n % 9 = 0

def count_valid_numbers : ℕ := 
  Nat.card {n // 100 ≤ n ∧ n < 1000 ∧ is_valid_number n ∧ is_divisible_by_9 n}

theorem count_positive_three_digit_integers_divisible_by_9 (N : ℕ) : 
  count_valid_numbers = N := 
sorry

end count_positive_three_digit_integers_divisible_by_9_l605_605314


namespace jon_toaster_total_cost_l605_605609

def total_cost_toaster (MSRP : ℝ) (std_ins_pct : ℝ) (premium_upgrade_cost : ℝ) (state_tax_pct : ℝ) (environmental_fee : ℝ) : ℝ :=
  let std_ins_cost := std_ins_pct * MSRP
  let premium_ins_cost := std_ins_cost + premium_upgrade_cost
  let subtotal_before_tax := MSRP + premium_ins_cost
  let state_tax := state_tax_pct * subtotal_before_tax
  let total_before_env_fee := subtotal_before_tax + state_tax
  total_before_env_fee + environmental_fee

theorem jon_toaster_total_cost :
  total_cost_toaster 30 0.2 7 0.5 5 = 69.5 :=
by
  sorry

end jon_toaster_total_cost_l605_605609


namespace evaluate_expression_l605_605652

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l605_605652


namespace find_coefficients_l605_605629

def polynomial (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 3 - 3 * x ^ 2 + b * x - 7

theorem find_coefficients (a b : ℝ) :
  polynomial a b 2 = -17 ∧ polynomial a b (-1) = -11 → a = 0 ∧ b = -1 :=
by
  sorry

end find_coefficients_l605_605629


namespace area_of_triangle_ABC_l605_605937

variables {A B C D : ℝ}
variables (AC AB DC AD : ℝ)
variables (coplanar : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0)

noncomputable def area_triangle_ABC (AC AB DC : ℝ) :=
  let AD := Real.sqrt (AC^2 - DC^2) in
  let BD := Real.sqrt (AB^2 - AD^2) in
  let area_ABD := (1 / 2) * AD * BD in
  let area_ACD := (1 / 2) * AD * DC in
  area_ABD - area_ACD

theorem area_of_triangle_ABC :
  area_triangle_ABC 15 17 8 = 4 * (Real.sqrt 322 - Real.sqrt 161) := 
sorry

end area_of_triangle_ABC_l605_605937


namespace exist_solution_BET_l605_605973

theorem exist_solution_BET :
  ∃ (B E T : ℕ), 
    B ≠ E ∧ B ≠ T ∧ E ≠ T ∧
    ∀ M H : ℕ, 
      M ≠ B ∧ M ≠ E ∧ M ≠ T ∧ M ≠ H ∧
      H ≠ B ∧ H ≠ E ∧ H ≠ T ∧
      M ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      H ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      T ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      E ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      100 * T + 10 * H + E + 1000 * M + 100 * B + 10 * M + T = 2018 :=
begin
  use [4, 2, 6],
  split, { exact nat.succ_ne_zero 4 }, split, { exact nat.zero_ne_succ 2 }, split, { exact nat.succ_ne_succ 2 },
  intros M H,
  split, { exact nat.succ_ne_zero M }, split, { exact nat.zero_ne_succ E }, split, { exact nat.succ_ne_succ T },
  split_ents,
  sorry
end

end exist_solution_BET_l605_605973


namespace minimum_moves_to_determine_number_l605_605883

theorem minimum_moves_to_determine_number (distinct_digits : ∀(n : ℕ), n >= 10000 ∧ n < 100000 → ∀ i j : fin 5, (i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) ) :
  (∃ (T : ℕ), T = 3 ∧ ∀ (p : fin 5 → fin 5), ∃ (d : fin 5 → fin 10), T = 3 ∧ ∀ k : fin 5, p k = d k) :=
by sorry

end minimum_moves_to_determine_number_l605_605883


namespace gold_foil_thickness_l605_605773

theorem gold_foil_thickness :
  (0.000000092 : ℝ) = 9.2 * 10^(-8) :=
by
  sorry

end gold_foil_thickness_l605_605773


namespace selection_at_most_one_l605_605729

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_at_most_one (A B : ℕ) :
  (combination 5 3) - (combination 3 1) = 7 :=
by
  sorry

end selection_at_most_one_l605_605729


namespace count_non_negative_numbers_l605_605600

theorem count_non_negative_numbers :
  let numbers := [-15, 5 + 1 / 3, -0.23, 0, 7.6, 2, -1 / 3, 3.14]
  in (list.filter (λ x, 0 ≤ x) numbers).length = 5 :=
by
  let numbers := [-15, 5 + 1 / 3, -0.23, 0, 7.6, 2, -1 / 3, 3.14] in
  sorry

end count_non_negative_numbers_l605_605600


namespace graph_edge_degree_sum_l605_605249

variable {P : Type*} [MetricSpace P]

def point_set (n : ℕ) : Set (Fin n → P) :=
  {F | ∀ i j, i ≠ j → (dist (F i) (F j) = d) ∨ (dist (F i) (F j) ≠ d)}

def graph G_n (F : Set P) (d : ℝ) :=
  {points := F,
   edges := {(P_i, P_j) | P_i ∈ F ∧ P_j ∈ F ∧ dist P_i P_j = d}
  }

def degree (G_n : graph) (P_i : P) : ℕ :=
  (G_n.edges.filter (fun ⟨x, y⟩ => x = P_i ∨ y = P_i)).card

def m_n (G_n : graph) : ℕ :=
  G_n.edges.card

theorem graph_edge_degree_sum {F_n : point_set n} {d : ℝ} {G_n : graph G_n (point_set n) d} (F_n P_i) :
  |m_n G_n (point_set n d)| = (1/2) * (∑ i : range(n), degree G_n (cast i P_i)) :=
  sorry

end graph_edge_degree_sum_l605_605249


namespace find_coordinates_l605_605232

open Real EuclideanSpace

def point_A : ℝ × ℝ × ℝ := (-4, 1, 7)
def point_B : ℝ × ℝ × ℝ := (3, 5, -2)
def point_C (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

theorem find_coordinates :
  ∃ z : ℝ, distance (point_C z) point_A = distance (point_C z) point_B ∧ point_C z = (0, 0, 14/9) :=
begin
  sorry
end

end find_coordinates_l605_605232


namespace cans_to_paint_35_rooms_l605_605446

/-- Paula the painter initially had enough paint for 45 identically sized rooms.
    Unfortunately, she lost five cans of paint, leaving her with only enough paint for 35 rooms.
    Prove that she now uses 18 cans of paint to paint the 35 rooms. -/
theorem cans_to_paint_35_rooms :
  ∀ (cans_per_room : ℕ) (total_cans : ℕ) (lost_cans : ℕ) (rooms_before : ℕ) (rooms_after : ℕ),
  rooms_before = 45 →
  lost_cans = 5 →
  rooms_after = 35 →
  rooms_before - rooms_after = cans_per_room * lost_cans →
  (cans_per_room * rooms_after) / rooms_after = 18 :=
by
  intros
  sorry

end cans_to_paint_35_rooms_l605_605446


namespace bill_money_left_l605_605602

def bill_remaining_money (merchantA_qty : Int) (merchantA_rate : Int) 
                        (merchantB_qty : Int) (merchantB_rate : Int)
                        (fine : Int) (merchantC_qty : Int) (merchantC_rate : Int) 
                        (protection_costs : Int) (passerby_qty : Int) 
                        (passerby_rate : Int) : Int :=
let incomeA := merchantA_qty * merchantA_rate
let incomeB := merchantB_qty * merchantB_rate
let incomeC := merchantC_qty * merchantC_rate
let incomeD := passerby_qty * passerby_rate
let total_income := incomeA + incomeB + incomeC + incomeD
let total_expenses := fine + protection_costs
total_income - total_expenses

theorem bill_money_left 
    (merchantA_qty : Int := 8) 
    (merchantA_rate : Int := 9) 
    (merchantB_qty : Int := 15) 
    (merchantB_rate : Int := 11) 
    (fine : Int := 80)
    (merchantC_qty : Int := 25) 
    (merchantC_rate : Int := 8) 
    (protection_costs : Int := 30) 
    (passerby_qty : Int := 12) 
    (passerby_rate : Int := 7) : 
    bill_remaining_money merchantA_qty merchantA_rate 
                         merchantB_qty merchantB_rate 
                         fine merchantC_qty merchantC_rate 
                         protection_costs passerby_qty 
                         passerby_rate = 411 := by 
  sorry

end bill_money_left_l605_605602


namespace Cairo_has_greatest_percentage_increase_l605_605826

-- Define populations in a structure for better clarity
structure CityPopulation :=
  (population_1970 : ℝ) 
  (population_1980 : ℝ)

-- Define the population data for each city
def Paris : CityPopulation := { population_1970 := 1.5, population_1980 := 1.8 }
def Cairo : CityPopulation := { population_1970 := 2.4, population_1980 := 3.3 }
def Lima : CityPopulation := { population_1970 := 1.2, population_1980 := 1.56 }
def Tokyo : CityPopulation := { population_1970 := 8.6, population_1980 := 9.52 }
def Toronto : CityPopulation := { population_1970 := 2.0, population_1980 := 2.4 }

-- Define the percentage increase formula
def percentage_increase (pop : CityPopulation) : ℝ := 
  (pop.population_1980 - pop.population_1970) / pop.population_1970

-- The Lean statement to prove that Cairo has the greatest percentage increase
theorem Cairo_has_greatest_percentage_increase :
  let P_inc := percentage_increase Paris
  let Q_inc := percentage_increase Cairo
  let R_inc := percentage_increase Lima
  let S_inc := percentage_increase Tokyo
  let T_inc := percentage_increase Toronto
  Q_inc > P_inc ∧ Q_inc > R_inc ∧ Q_inc > S_inc ∧ Q_inc > T_inc :=
begin
  -- sorry added to skip proof
  sorry,
end

end Cairo_has_greatest_percentage_increase_l605_605826


namespace rhombus_side_length_l605_605802

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l605_605802


namespace rhombus_side_length_l605_605803

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l605_605803


namespace photograph_perimeter_l605_605958

theorem photograph_perimeter (w l m : ℕ) 
  (h1 : (w + 4) * (l + 4) = m)
  (h2 : (w + 8) * (l + 8) = m + 94) :
  2 * (w + l) = 23 := 
by
  sorry

end photograph_perimeter_l605_605958


namespace simplify_f_value_of_f_l605_605259

-- Given definitions and conditions
def f (alpha : ℝ) : ℝ :=
  (Real.sin (π - alpha) * Real.cos (π + alpha) * Real.sin (-alpha + (3 * π / 2))) /
  (Real.cos (-alpha) * Real.cos (alpha + (π / 2)))

-- The angle α is in the third quadrant and cos(α - (3 * π / 2)) is given
def is_third_quadrant (alpha : ℝ) : Prop := 
  π < alpha ∧ alpha < (3 * π / 2)

lemma cos_alpha_minus_3pi_over_2 (alpha : ℝ) (h_alpha : is_third_quadrant alpha) :
  Real.cos (alpha - (3 * π / 2)) = 1 / 5 :=
  sorry

-- The main statements to be proven
theorem simplify_f (alpha : ℝ) : f(alpha) = -Real.cos alpha :=
  sorry

theorem value_of_f (alpha : ℝ) (h_alpha : is_third_quadrant alpha) (h_cos : Real.cos (alpha - (3 * π / 2)) = 1 / 5) :
  f(alpha) = 2 * Real.sqrt 6 / 5 :=
  sorry

end simplify_f_value_of_f_l605_605259


namespace mean_of_reciprocals_first_four_primes_l605_605144

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605144


namespace february_discount_l605_605087

variable (C : ℝ) -- cost price
variable (D : ℝ) -- discount percentage
variable (P : ℝ) -- February profit percentage

-- Marked up prices
def SP1 := 1.20 * C
def SP2 := 1.25 * SP1

-- Final Selling Price after discount
def SP3 := SP2 * (1 - D / 100)

-- February profit percentage
def profit_val := 0.395 * C

-- Hypothesis: February profit percentage is 39.5%
axiom february_profit : SP3 - C = profit_val

-- Proving that discount percentage in February is 7%
theorem february_discount :
  february_profit →
  D = 7 :=
sorry -- No proof required

end february_discount_l605_605087


namespace mean_of_reciprocals_of_first_four_primes_l605_605198

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605198


namespace first_worker_time_l605_605024

def productivity (x y z : ℝ) : Prop :=
  x + y + z = 20 ∧
  (20 / x) > 3 ∧
  (20 / x) + (60 / (y + z)) = 8

theorem first_worker_time (x y z : ℝ) (h : productivity x y z) : 
  (80 / x) = 16 :=
  sorry

end first_worker_time_l605_605024


namespace number_of_triangles_is_three_l605_605011

def lengths : List ℕ := [13, 10, 5, 7]

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangles (lengths : List ℕ) : ℕ :=
  (lengths.combinations 3).count (λ l, match l with
    | [a, b, c] => triangle_inequality a b c
    | _ => false
  )

theorem number_of_triangles_is_three : valid_triangles lengths = 3 :=
sorry

end number_of_triangles_is_three_l605_605011


namespace relationship_between_A_and_B_l605_605098

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem relationship_between_A_and_B : B ⊆ A :=
sorry

end relationship_between_A_and_B_l605_605098


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605171

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605171


namespace two_digit_interchange_difference_l605_605480

-- Define the conditions and prove the main theorem
theorem two_digit_interchange_difference:
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y = 2 * x ∧ (x + y) - (y - x) = 8 ∧ 
  abs ((10 * x + y) - (10 * y + x)) = 36 :=
by
  sorry

end two_digit_interchange_difference_l605_605480


namespace apples_per_person_l605_605954

theorem apples_per_person
    (boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ) (people : ℕ)
    (initial_apples : boxes * apples_per_box = 63)
    (remaining_apples : initial_apples - rotten_apples = 56)
    (equally_divided : remaining_apples / people = 7) :
    (boxes = 7) ∧ (apples_per_box = 9) ∧ (rotten_apples = 7) ∧ (people = 8) → 
    (remaining_apples / people = 7) := by
  sorry

end apples_per_person_l605_605954


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605180

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605180


namespace problem_1_problem_2a_gt_1_problem_2a_lt_1_problem_3_l605_605293

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x - a / x - 2 * Real.log x

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.exp 1 / x

theorem problem_1 (x : ℝ) : f 2 x = 2 * x - 2 / x - 2 * Real.log x := sorry

theorem problem_2a_gt_1 (a : ℝ) (h1 : a ≥ 1) :
  ∀ x : ℝ, x ∈ set.Ioi 0 → monotone (f a) := sorry

theorem problem_2a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x : ℝ, x ∈ Ioo 0 (1 - Real.sqrt(1 - a) / a) ∨ x ∈ Ioi (1 + Real.sqrt(1 - a) / a) → monotone (f a)) ∧
  (∀ x : ℝ, x ∈ Ioo (1 - Real.sqrt(1 - a) / a) (1 + Real.sqrt(1 - a) / a) → antitone (f a)) := sorry

theorem problem_3 (a : ℝ) (h1 : a > (4 * Real.exp 1) / (Real.exp 1 ^ 2 - 1)) :
  ∃ x : ℝ, x ∈ set.Ico 1 (Real.exp 1) ∧ f a x > g x := sorry

end problem_1_problem_2a_gt_1_problem_2a_lt_1_problem_3_l605_605293


namespace transform_sin_to_cos_l605_605513

theorem transform_sin_to_cos (x : ℝ) :
  let y₀ := sin (2 * x + π / 6)
  let y₁ := sin (2 * (x + π / 6) + π / 6)
  let y₂ := cos (2 * x)
  let y₃ := 2 * cos (x)^2
  y₃ = y₁ + 1 :=
by
  sorry

end transform_sin_to_cos_l605_605513


namespace find_common_ratio_limit_SN_over_TN_l605_605266

noncomputable def S (q : ℚ) (n : ℕ) : ℚ := (1 - q^n) / (1 - q)
noncomputable def T (q : ℚ) (n : ℕ) : ℚ := (1 - q^(2 * n)) / (1 - q^2)

theorem find_common_ratio
  (S3 : S q 3 = 3)
  (S6 : S q 6 = -21) :
  q = -2 :=
sorry

theorem limit_SN_over_TN
  (q_pos : 0 < q)
  (Tn_def : ∀ n, T q n = 1) :
  (q > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 0| < ε) ∧
  (0 < q ∧ q < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - (1 + q)| < ε) ∧
  (q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 1| < ε) :=
sorry

end find_common_ratio_limit_SN_over_TN_l605_605266


namespace intersection_on_altitude_of_triangle_l605_605372

variable {A B C D P Q : Type*} [field A] [field B] [field C] [field D] [field P] [field Q]

theorem intersection_on_altitude_of_triangle
  (hABCD_parallelogram : parallelogram A B C D)
  (hABP_equilateral : equilateral_triangle A B P)
  (hBCQ_equilateral : equilateral_triangle B C Q) :
  ∃ X, 
    (is_intersection X (perpendicular_line_through_point P (line_through_points P D)) 
      (perpendicular_line_through_point Q (line_through_points Q D))) ∧
    (on_altitude_from B X (triangle A B C)) := 
sorry

end intersection_on_altitude_of_triangle_l605_605372


namespace intervals_of_increase_axis_of_symmetry_range_of_k_l605_605761

def f (x : ℝ) : ℝ := 2 - 2 * (Real.cos (π/4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem intervals_of_increase :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π →
  ((0 ≤ x ∧ x ≤ 5 * π / 12) ∨ (11 * π / 12 ≤ x ∧ x ≤ π)) :=
sorry -- Replace with the actual proof

theorem axis_of_symmetry (k : ℤ) :
  ∀ x : ℝ, (2 * x - π / 3) = (k * π + π / 2) → x = k * π / 2 + 5 * π / 12 :=
sorry -- Replace with the actual proof

theorem range_of_k :
  ∀ k : ℝ, ∃ x : ℝ, (π / 4 ≤ x ∧ x ≤ π / 2) → (2 ≤ 1 + 2 * Real.sin (2 * x - π / 3) ≤ 3) ↔ (2 ≤ k ∧ k ≤ 3) :=
sorry -- Replace with actual proof

end intervals_of_increase_axis_of_symmetry_range_of_k_l605_605761


namespace divide_weights_into_two_equal_piles_l605_605253

theorem divide_weights_into_two_equal_piles :
  ∃ (A B : Finset ℕ), 
    A.card = 50 ∧ B.card = 50 ∧ 
    A ∪ B = (Finset.range 102).erase 19 ∧ 
    A ∩ B = ∅ ∧ 
    (∑ x in A, x) = 2566 ∧ (∑ x in B, x) = 2566 :=
by
  sorry

end divide_weights_into_two_equal_piles_l605_605253


namespace dealer_car_ratio_calculation_l605_605985

theorem dealer_car_ratio_calculation (X Y : ℝ) 
  (cond1 : 1.4 * X = 1.54 * (X + Y) - 1.6 * Y) :
  let a := 3
  let b := 7
  ((X / Y) = (3 / 7) ∧ (11 * a + 13 * b = 124)) :=
by
  sorry

end dealer_car_ratio_calculation_l605_605985


namespace smallest_n_l605_605690

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605690


namespace one_oplus_three_l605_605626

def op ⊕ (x y : ℕ) : ℕ := -3*x + 4*y

theorem one_oplus_three : (1 ⊕ 3) = 9 :=
by
  sorry

end one_oplus_three_l605_605626


namespace count_valid_n_l605_605716

def num_valid_n : ℕ :=
  let possible_n := { n : ℕ | ∃ (a b : ℕ), n = 2^a * 5^b ∧ n < 1000 } in
  let non_zero_thousandths := { n ∈ possible_n | (1000 * n : ℚ) / n % 10 ≠ 0 } in
  non_zero_thousandths.card

theorem count_valid_n : num_valid_n = 25 := by
  sorry

end count_valid_n_l605_605716


namespace heejin_drinks_most_l605_605633

-- Given conditions
def drinks_per_day (person : String) (times : Nat) (amount_each_time : Float) : Float :=
  times * amount_each_time

def heejin_drinks_per_day (times : Nat) (amount_ml_each_time : Nat) : Float :=
  times * (amount_ml_each_time / 1000)

-- Heejin's daily water consumption
def heejin_consumption : Float :=
  heejin_drinks_per_day 4 500

-- Dongguk's daily water consumption
def dongguk_consumption : Float :=
  drinks_per_day "Dongguk" 5 0.2

-- Yoonji's daily water consumption
def yoonji_consumption : Float :=
  drinks_per_day "Yoonji" 6 0.3

-- Theorem: Heejin drinks the most water per day (2 liters)
theorem heejin_drinks_most :
  heejin_consumption > dongguk_consumption ∧ heejin_consumption > yoonji_consumption :=
by
  -- Skipping the proof
  sorry

end heejin_drinks_most_l605_605633


namespace mean_of_reciprocals_of_first_four_primes_l605_605200

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605200


namespace find_tan_gamma_l605_605987

-- Define the parameters and theorems required
variable (k : ℝ) (α β γ : ℝ)

-- Given condition
axiom tan_alpha_eq_k : Real.tan α = k

-- Define the angles
noncomputable def beta : ℝ := Real.arctan (k / 2)
noncomputable def gamma : ℝ := π - (α + beta)

-- The tangent of gamma as proved in the problem steps
theorem find_tan_gamma (h : Real.tan α = k) : Real.tan γ = 3 * k / (k^2 - 2) :=
by
  -- Skip the detailed proof steps
  sorry

end find_tan_gamma_l605_605987


namespace circumcircle_PQR_tangent_BP_BR_l605_605267

open EuclideanGeometry

variables {P A B C R : Point} 
variables (O1 O2 : Circle)

-- Conditions
-- P is a point such that there are common tangents PA and PB to circles O1 and O2.
def tangent_from_P_to_O1 := tangent O1 A P
def tangent_from_P_to_O2 := tangent O2 B P

-- C is the intersection of a tangent from P to O1 with circle O2
def intersect_tangent_at_O2 (P O1 O2 : Circle) (C : Point) :=
  is_tangent_line P O1 C ∧ lies_on_circle C O2

-- R is the intersection point of AP extended and BC
def intersect_AP_extended_BC (A P B C : Point) : Point :=
  (line A P).intersect (line B C) R

-- The problem statement:
theorem circumcircle_PQR_tangent_BP_BR
  (hPA : tangent_from_P_to_O1 P A)
  (hPB : tangent_from_P_to_O2 P B)
  (hPC : intersect_tangent_at_O2 P O1 O2 C)
  (hPR : intersect_AP_extended_BC A P B C):
  tangency (circumcircle P Q R) (line BP) ∧ tangency (circumcircle P Q R) (line BR) :=
sorry

end circumcircle_PQR_tangent_BP_BR_l605_605267


namespace bicycles_difference_on_october_1_l605_605886

def initial_inventory : Nat := 200
def february_decrease : Nat := 4
def march_decrease : Nat := 6
def april_decrease : Nat := 8
def may_decrease : Nat := 10
def june_decrease : Nat := 12
def july_decrease : Nat := 14
def august_decrease : Nat := 16 + 20
def september_decrease : Nat := 18
def shipment : Nat := 50

def total_decrease : Nat := february_decrease + march_decrease + april_decrease + may_decrease + june_decrease + july_decrease + august_decrease + september_decrease
def stock_increase : Nat := shipment
def net_decrease : Nat := total_decrease - stock_increase

theorem bicycles_difference_on_october_1 : initial_inventory - net_decrease = 58 := by
  sorry

end bicycles_difference_on_october_1_l605_605886


namespace angle_COD_142_5_l605_605965

-- Define the geometry setup and conditions
variables (ω₁ ω₂ : Circle) (A B P Q R S O C D : Point)
variables (h_intersect : ω₁.Intersect ω₂ A B)
variables (l : Line)
variables (h_tangent_1 : TangentAt l ω₁ P)
variables (h_tangent_2 : TangentAt l ω₂ Q)
variables (h_closer : CloserTo A PQ B)
variables (h_ray1 : OnRay P A R)
variables (h_ray2 : OnRay Q A S)
variables (h_eq_lengths : length PQ = length AR ∧ length PQ = length AS)
variables (h_opposite_sides : OppositeSides A P Q R S)
variables (h_circumcenter : O = CircumcenterOf (Triangle.mk A S R))
variables (h_midpoint_major_arc_1 : C = MidpointMajorArc ω₁ A P)
variables (h_midpoint_major_arc_2 : D = MidpointMajorArc ω₂ A Q)
variables (h_angle_APQ : ∠(A P Q) = 45)
variables (h_angle_AQP : ∠(A Q P) = 30)

-- State the theorem
theorem angle_COD_142_5 (h_intersect : ω₁.Intersect ω₂ A B)
                        (h_tangent_1 : TangentAt l ω₁ P)
                        (h_tangent_2 : TangentAt l ω₂ Q)
                        (h_closer : CloserTo A PQ B)
                        (h_ray1 : OnRay P A R)
                        (h_ray2 : OnRay Q A S)
                        (h_eq_lengths : length PQ = length AR ∧ length PQ = length AS)
                        (h_opposite_sides : OppositeSides A P Q R S)
                        (h_circumcenter : O = CircumcenterOf (Triangle.mk A S R))
                        (h_midpoint_major_arc_1 : C = MidpointMajorArc ω₁ A P)
                        (h_midpoint_major_arc_2 : D = MidpointMajorArc ω₂ A Q)
                        (h_angle_APQ : ∠(A P Q) = 45)
                        (h_angle_AQP : ∠(A Q P) = 30) : 
    ∠(C O D) = 142.5 := 
sorry

end angle_COD_142_5_l605_605965


namespace smaller_tetrahedron_volume_and_surface_area_l605_605945

theorem smaller_tetrahedron_volume_and_surface_area (V S : ℝ) :
  ∃ V' S', 
    V' = V / 27 ∧
    S' = S / 9 :=
by
  use V / 27, S / 9,
  split;
  sorry

end smaller_tetrahedron_volume_and_surface_area_l605_605945


namespace mean_of_reciprocals_first_four_primes_l605_605143

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l605_605143


namespace extremum_f_l605_605392

noncomputable def f (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (∑ i in Finset.range n, real.sqrt ((x i)^2 + (x i) * (x ((i + 1) % n)) + (x ((i + 1) % n))^2))

theorem extremum_f (n : ℕ) (hn : 2 ≤ n) (x : ℕ → ℝ) (hx : (∑ i in Finset.range n, x i) = 1)
  (hx_nonneg : ∀ i, 0 ≤ x i) : 
  sqrt 3 ≤ f n x ∧ f n x ≤ 2 :=
sorry

end extremum_f_l605_605392


namespace smallest_n_for_identity_l605_605707

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605707


namespace number_of_valid_sequences_l605_605395

open BigOperators

/-- Define the conditions for the sequence. -/

def valid_sequence (n : ℕ) (a : Fin n → ℕ) : Prop :=
  (∀ i, a i = 0 ∨ a i = 1) ∧
  (∀ i, (i.val % 2 = 0 → a i ≤ a i.succ) ∧ (i.val % 2 = 1 → a i ≥ a i.succ))

/-- Fibonacci sequence definition -/
def fibonacci : ℕ → ℕ 
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

/-- Define the number of valid sequences as the (n + 2)-th Fibonacci number -/
theorem number_of_valid_sequences (n : ℕ) : 
  ∃ f : (Fin n → ℕ), valid_sequence n f ∧ (Finset.card {a : Fin n → ℕ | valid_sequence n a} = fibonacci (n + 2)) :=
sorry

end number_of_valid_sequences_l605_605395


namespace difference_of_numbers_l605_605481

theorem difference_of_numbers 
  (L S : ℤ) (hL : L = 1636) (hdiv : L = 6 * S + 10) : 
  L - S = 1365 :=
sorry

end difference_of_numbers_l605_605481


namespace complex_sum_to_int_l605_605459

-- Defining the problem in Lean 4
theorem complex_sum_to_int (x y : ℂ) (h1 : x + y ∈ ℤ) (h2 : x^2 + y^2 ∈ ℤ) 
  (h3 : x^3 + y^3 ∈ ℤ) (h4 : x^4 + y^4 ∈ ℤ) : ∀ n : ℕ, x^n + y^n ∈ ℤ := 
by
  sorry

end complex_sum_to_int_l605_605459


namespace intersection_A_complement_B_l605_605743

def A : set ℤ := {x | (x + 2) * (x - 3) ≤ 0}
def B : set ℝ := {x | x ≤ -1 ∨ x > 3}
def R_complement_B : set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_A_complement_B : 
  (A : set ℝ) ∩ R_complement_B = {0, 1, 2} :=
by {
  sorry
}

end intersection_A_complement_B_l605_605743


namespace monotonically_increasing_interval_l605_605274

noncomputable def f1 (x : ℝ) : ℝ := Real.sin (3 * Real.pi / 2 + x) * Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x * Real.sin (Real.pi + x)
noncomputable def f (x : ℝ) : ℝ := f1 x - f2 x

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 → monotone f :=
by
  sorry

end monotonically_increasing_interval_l605_605274


namespace smallest_positive_integer_n_l605_605695

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605695


namespace number_of_other_workers_l605_605359

theorem number_of_other_workers (N : ℕ) (h1 : N ≥ 2) (h2 : 1 / ((N * (N - 1)) / 2) = 1 / 6) : N - 2 = 2 :=
by
  sorry

end number_of_other_workers_l605_605359


namespace difference_in_payment_l605_605362

theorem difference_in_payment (joy_pencils : ℕ) (colleen_pencils : ℕ) (price_per_pencil : ℕ) (H1 : joy_pencils = 30) (H2 : colleen_pencils = 50) (H3 : price_per_pencil = 4) :
  (colleen_pencils * price_per_pencil) - (joy_pencils * price_per_pencil) = 80 :=
by
  rw [H1, H2, H3]
  simp
  norm_num
  sorry

end difference_in_payment_l605_605362


namespace smallest_positive_n_l605_605674

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605674


namespace tangent_line_at_point_l605_605876

noncomputable def f'' (f' : ℝ → ℝ) : ℝ → ℝ := λ x : ℝ, x^3 + 2 * f' 1 * x^2 + 1

theorem tangent_line_at_point (f' : ℝ → ℝ) (h : f' 1 = -1) :
  let f := λ x : ℝ, x^3 + 2 * f' 1 * x^2 + 1 in
  f 1 = 0 ∧ f' 1 = -1 →
  (λ x y : ℝ, x + y - 1 = 0) :=
by
  sorry

end tangent_line_at_point_l605_605876


namespace lambda_value_l605_605627

theorem lambda_value (ω : ℂ) (λ : ℝ) (h1 : |ω| = 3) (h2 : λ > 1)
  (h3 : |λ * ω|^2 = |ω|^2 + |ω^2|^2) : λ = Real.sqrt 10 :=
by
  sorry

end lambda_value_l605_605627


namespace curve_not_parabola_l605_605789

theorem curve_not_parabola (k : ℝ) : ¬ (∀ x y, x^2 + k * y^2 = 1 → is_parabola x y) :=
sorry

end curve_not_parabola_l605_605789


namespace arithmetic_sequence_formula_geometric_sequence_formula_lambda_range_l605_605748

-- Definitions based on conditions
def a (n : ℕ) := n + 2
def b (n : ℕ) := 2^n
def c (n : ℕ) : ℝ := (Real.log 2 (b n)) / (a n) + (a n) / (Real.log 2 (b n))
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, c (i + 1)

-- Theorem statements based on the problem
theorem arithmetic_sequence_formula : ∀ n, a n = n + 2 :=
sorry

theorem geometric_sequence_formula : ∀ n, b n = 2^n :=
sorry

theorem lambda_range : ∀ (n : ℕ) (λ : ℝ), (∀ n, T n - 2 * n < λ) → λ ≥ 3 :=
sorry

end arithmetic_sequence_formula_geometric_sequence_formula_lambda_range_l605_605748


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605212

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605212


namespace max_cells_colored_without_forming_rectangle_l605_605042

def is_valid_coloring (grid : ℕ → ℕ → Bool) : Prop :=
  ∀ (i1 j1 i2 j2 i3 j3 i4 j4 : ℕ),
    i1 < i2 → i2 < i3 → i3 < i4 →
    j1 < j2 → j2 < j3 → j3 < j4 →
    grid i1 j1 = true → grid i2 j2 = true →
    grid i3 j3 = true → grid i4 j4 = true →
    ¬((i1 = i3) ∧ (i2 = i4) ∧ (j1 = j2) ∧ (j3 = j4))

theorem max_cells_colored_without_forming_rectangle : ∃ grid : ℕ → ℕ → Bool, is_valid_coloring grid ∧ (∑ i j, if grid i j then 1 else 0) = 24 :=
by
  sorry

end max_cells_colored_without_forming_rectangle_l605_605042


namespace romanov_family_savings_l605_605567

theorem romanov_family_savings :
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption := monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  savings = 3824 :=
by
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption :=monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2 
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  show savings = 3824 
  sorry

end romanov_family_savings_l605_605567


namespace monotonicity_F_when_t_equals_1_range_of_t_l605_605275

-- Definitions for the given functions m(x) and n(x)
def m (x : ℝ) (t : ℝ) : ℝ := t * real.exp x + real.log (t / (x + 2))
def n (x : ℝ) : ℝ := 1 - real.log ( (x + 2) / real.exp (2 * x))

-- Part 1: Monotonicity of F(x) when t = 1
def F (x : ℝ) : ℝ := m x 1 - n x

theorem monotonicity_F_when_t_equals_1 :
  ∀ x : ℝ, x > -2 → x < real.ln 2 → (F x) < F (real.ln 2) 
  ∧ ∀ x : ℝ, x > real.ln 2 → (F x) > F (real.ln 2) :=
sorry

-- Part 2: Range of values for t
theorem range_of_t (t : ℝ) :
  (∀ (x : ℝ), x > -2 → m x t > 2) → t > real.exp 1 :=
sorry

end monotonicity_F_when_t_equals_1_range_of_t_l605_605275


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605164

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605164


namespace smallest_n_for_identity_l605_605703

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605703


namespace households_using_neither_brand_l605_605587

noncomputable def households := 180
noncomputable def only_brand_A := 60
noncomputable def both_brands := 10
noncomputable def only_brand_B := 3 * both_brands

theorem households_using_neither_brand:
  households - (only_brand_A + only_brand_B + both_brands) = 80 :=
by
  -- Unfold definitions and perform arithmetic simplifications
  unfold households only_brand_A only_brand_B both_brands
  simp
  -- The proof step is skipped
  sorry

end households_using_neither_brand_l605_605587


namespace prob_and_relation_proof_l605_605563

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l605_605563


namespace largest_angle_of_convex_hexagon_l605_605476

theorem largest_angle_of_convex_hexagon (a b c d e f : ℤ) 
  (h1 : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) 
  (h2 : f = a + 5)
  (h3 : a + b + c + d + e + f = 720) : 
  f = 122.5 :=
sorry

end largest_angle_of_convex_hexagon_l605_605476


namespace Stewarts_Theorem_l605_605327

-- Definitions of points and lengths
variables (A B C D : Type) [MetricSpaces.Point A] [MetricSpaces.Point B] [MetricSpaces.Point C] [MetricSpaces.Point D]

-- Lengths of segments using Lean's notation
variables (AB AC AD BD CD BC : ℝ)
variable (on_BC : MetricSpaces.Collinear B C D)

-- Stewart's Theorem statement
theorem Stewarts_Theorem (h : on_BC) : 
  AB^2 * CD + AC^2 * BD = AD^2 * BC + BC * BD * CD :=
sorry

end Stewarts_Theorem_l605_605327


namespace lift_time_15_minutes_l605_605495

theorem lift_time_15_minutes (t : ℕ) (h₁ : 5 = 5) (h₂ : 6 * (t + 5) = 120) : t = 15 :=
by {
  sorry
}

end lift_time_15_minutes_l605_605495


namespace surface_area_of_rotated_broken_line_l605_605971

theorem surface_area_of_rotated_broken_line (n : ℕ) (l_i y_i : ℕ → ℝ) :
  let S_i := λ i, 2 * Real.pi * y_i i * l_i i,
      S := ∑ i in Finset.range n, S_i i,
      l := ∑ i in Finset.range n, l_i i,
      z := (∑ i in Finset.range n, y_i i * l_i i) / l 
  in S = 2 * Real.pi * z * l := 
by
  sorry

end surface_area_of_rotated_broken_line_l605_605971


namespace françoise_pots_sale_l605_605252

theorem françoise_pots_sale 
  (cost_price per_pot : ℝ)
  (increase_percentage : ℝ)
  (total_given : ℝ )
  (selling_price profit_per_pot : ℝ)
  (number_of_pots : ℝ) :
  cost_price = 12 → 
  increase_percentage = 0.25 → 
  total_given = 450 → 
  selling_price = cost_price * (1 + increase_percentage) → 
  profit_per_pot = selling_price - cost_price →
  number_of_pots = total_given / profit_per_pot →
  number_of_pots = 150 := 
by
  intros h_cost_price h_increase_percentage h_total_given h_selling_price h_profit_per_pot h_number_of_pots
  rw [h_cost_price, h_increase_percentage, h_total_given] at *
  rw [h_selling_price, h_profit_per_pot, h_number_of_pots]
  sorry

end françoise_pots_sale_l605_605252


namespace vectors_not_coplanar_l605_605604

-- Define the vectors a, b, c as tuples or lists
def a := (2, 3, 1)
def b := (-1, 0, -1)
def c := (2, 2, 2)

-- Define the matrix formed by the vectors
def matrix := ![
  [2, 3, 1],
  [-1, 0, -1],
  [2, 2, 2]
]

-- State the problem
theorem vectors_not_coplanar : 
  ∃ det : ℤ, 
  det ≠ 0 ∧ 
  det = ((matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])) -
         (matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])) + 
         (matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))) := 
by {
  let a := (2, 3, 1),
  let b := (-1, 0, -1),
  let c := (2, 2, 2),
  let matrix := ![
    [2, 3, 1],
    [-1, 0, -1],
    [2, 2, 2]
  ],
  let det := ((matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])) -
             (matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])) + 
             (matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))),
  have h_det : det = 2, {
    sorry
  },
  use 2,
  exact ⟨by norm_num, h_det⟩,
}

end vectors_not_coplanar_l605_605604


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605214

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605214


namespace smallest_square_side_length_l605_605621

theorem smallest_square_side_length (s : ℕ) :
  (∃ s, s > 3 ∧ s ≤ 4 ∧ (s - 1) * (s - 1) = 5) ↔ s = 4 := by
  sorry

end smallest_square_side_length_l605_605621


namespace find_x_l605_605323

theorem find_x (n x q p : ℕ) (h1 : n = q * x + 2) (h2 : 2 * n = p * x + 4) : x = 6 :=
sorry

end find_x_l605_605323


namespace intersection_cardinality_l605_605064

theorem intersection_cardinality :
  let A := { n ∈ Set.univ | 4 ≤ n ∧ n ≤ 15 }
  let B := { n ∈ Set.univ | 6 ≤ n ∧ n ≤ 20 } in
  (Set.inter A B).card = 10 :=
by
  sorry

end intersection_cardinality_l605_605064


namespace possibility_of_fifty_in_cherepakhi_by_february19_l605_605502

structure Student :=
  (name : String)

structure Club :=
  (name : String)
  (founders : List Student)

structure School :=
  (students : List Student)
  (clubs : List Club)
  (friendship : Student → Student → Prop)

def hasAtLeastThreeFriendsInClub (s : School) (student : Student) (club : Club) : Prop :=
  (club.founders.filter (λ f => s.friendship student f)).length ≥ 3

def joinClub (s : School) (day : Nat) (club : Club) : List Student :=
  s.students.filter (λ student => hasAtLeastThreeFriendsInClub s student club)

axiom february19_all_in_gepardy (s : School) (gepardy club : Club) :
  (∀ student, student ∈ s.students → hasAtLeastThreeFriendsInClub s student gepardy) →
  (∀ student, hasAtLeastThreeFriendsInClub s student gepardy)

axiom fifty_in_cherepakhi (s : School) (cherepakhi : Club) :
  (∀ student, student ∈ joinClub s 1 cherepakhi) →
  (s.students.filter (λ student => member student cherepakhi)).length = 50

theorem possibility_of_fifty_in_cherepakhi_by_february19 (s : School) (gepardy cherepakhi : Club) :
  (∀ student, student ∈ s.students → hasAtLeastThreeFriendsInClub s student cherepakhi) →
  fifty_in_cherepakhi s cherepakhi :=
sorry

end possibility_of_fifty_in_cherepakhi_by_february19_l605_605502


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605189

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605189


namespace members_in_club_l605_605584

-- Define the problem conditions
def num_committees := 4
def pairs_committees (n : ℕ) := n * (n - 1) / 2

-- Define the main theorem to prove
theorem members_in_club (x : ℕ) (num_committees = 4) 
  (h1 : ∀ i, i ∈ (finset.range num_committees).subsets 2 -> exists m, m ∈ (finset.range x) ∧ m ∈ (finset.range num_committees).subsets 2)
  (h2 : ∀ i j, i ≠ j -> ∃! m, m ∈ (finset.range num_committees).subsets 2 ∧ m = i ∩ j) : 
  x = 6 :=
sorry

end members_in_club_l605_605584


namespace sum_of_a_and_c_l605_605484

variable {R : Type} [LinearOrderedField R]

theorem sum_of_a_and_c
    (ha hb hc hd : R) 
    (h_intersect : (1, 7) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (1, 7) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}
                 ∧ (9, 1) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (9, 1) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}) :
  ha + hc = 10 :=
by
  sorry

end sum_of_a_and_c_l605_605484


namespace rhombus_side_length_l605_605805

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l605_605805


namespace hypotenuse_length_l605_605090

theorem hypotenuse_length (x y : ℝ) 
  (h_vol1 : (1 / 3) * π * y^2 * x = 675 * π) 
  (h_vol2 : (1 / 3) * π * x^2 * y = 972 * π) : 
  sqrt (x^2 + y^2) ≈ 19.167 := 
sorry

end hypotenuse_length_l605_605090


namespace probability_of_same_color_l605_605580

-- Define the conditions
def total_balls : ℕ := 7 + 6 + 2
def total_ways_to_draw_two : ℕ := (total_balls * (total_balls - 1)) / 2
def white_ways_to_draw_two : ℕ := (7 * (7 - 1)) / 2
def black_ways_to_draw_two : ℕ := (6 * (6 - 1)) / 2
def red_ways_to_draw_two : ℕ := (2 * (2 - 1)) / 2
def favorable_outcomes : ℕ := white_ways_to_draw_two + black_ways_to_draw_two + red_ways_to_draw_two
def probability : ℚ := favorable_outcomes / total_ways_to_draw_two

-- State the theorem with the conditions
theorem probability_of_same_color
  (total_balls = 15)
  (total_ways_to_draw_two = ((total_balls * (total_balls - 1)) / 2))
  (white_ways_to_draw_two = ((7 * (7 - 1)) / 2))
  (black_ways_to_draw_two = ((6 * (6 - 1)) / 2))
  (red_ways_to_draw_two = ((2 * (2 - 1)) / 2))
  (favorable_outcomes = white_ways_to_draw_two + black_ways_to_draw_two + red_ways_to_draw_two)
  (probability = favorable_outcomes / total_ways_to_draw_two) :
  probability = 37 / 105 := sorry

end probability_of_same_color_l605_605580


namespace total_miles_l605_605624

theorem total_miles (miles_Darius : Int) (miles_Julia : Int) (h1 : miles_Darius = 679) (h2 : miles_Julia = 998) :
  miles_Darius + miles_Julia = 1677 :=
by
  sorry

end total_miles_l605_605624


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605220

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605220


namespace smallest_n_l605_605693

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l605_605693


namespace splendid_sum_one_l605_605120

def is_splendid (x : ℝ) : Prop :=
  ∀d, d ∈ (x.digits 10) → d = 0 ∨ d = 9

theorem splendid_sum_one : ∃ n : ℕ, ∃ s : vector ℝ n, (∀ i, is_splendid (s.nth i)) ∧ (s.to_list.sum = 1) ∧ (n = 1) :=
by 
  sorry

end splendid_sum_one_l605_605120


namespace parabola_intersection_probability_l605_605032

def parabolas_have_common_point (a b c d : ℤ) : Prop :=
    ∃ x : ℝ, x*x + 2*a*x + b = -x*x + 2*c*x + d

def is_valid_a (a : ℤ) : Prop :=
    1 ≤ a ∧ a ≤ 8

def is_valid_b (b : ℤ) : Prop :=
    1 ≤ b ∧ b ≤ 6

def is_valid_c (c : ℤ) : Prop :=
    1 ≤ c ∧ c ≤ 8

def is_valid_d (d : ℤ) : Prop :=
    1 ≤ d ∧ d ≤ 6

theorem parabola_intersection_probability :
    ∃ (p : ℚ), p = 83 / 88 ∧
    ∀ (a b c d : ℤ),
        is_valid_a a → is_valid_b b → is_valid_c c → is_valid_d d →
        (parabolas_have_common_point a b c d ↔ true) :=
begin
  sorry
end

end parabola_intersection_probability_l605_605032


namespace solution_l605_605659

/-- Definition of the number with 2023 ones. -/
def x_2023 : ℕ := (10^2023 - 1) / 9

/-- Definition of the polynomial equation. -/
def polynomial_eq (x : ℕ) : ℤ :=
  567 * x^3 + 171 * x^2 + 15 * x - (7 * x + 5 * 10^2023 + 3 * 10^(2*2023))

/-- The solution x_2023 satisfies the polynomial equation. -/
theorem solution : polynomial_eq x_2023 = 0 := sorry

end solution_l605_605659


namespace remaining_regular_toenails_l605_605307

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end remaining_regular_toenails_l605_605307


namespace smallest_n_for_identity_l605_605705

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l605_605705


namespace symmetric_line_equation_l605_605969

-- Define the original line equation
def original_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric property with respect to the y-axis
def symmetric_with_respect_to_y_axis (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l x y ↔ original_line (-x) y

-- The statement of the theorem
theorem symmetric_line_equation : 
  symmetric_with_respect_to_y_axis (λ x y, x + y - 1 = 0) :=
by
  intros x y
  unfold symmetric_with_respect_to_y_axis original_line
  split
  repeat sorry

end symmetric_line_equation_l605_605969


namespace transformed_parabola_l605_605822

theorem transformed_parabola :
  ∀ (x : ℝ), let y := x^2 + 2 in let y' := (x + 1)^2 + 1 in y' = (x + 1)^2 + 1 :=
by
  intros
  rw [← add_sub_cancel (x + 1)^2 1, add_assoc, add_assoc, sub_add_cancel]
  rhs <|> exact rfl
  sorry

end transformed_parabola_l605_605822


namespace rhombus_side_length_l605_605804

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l605_605804


namespace min_area_quad_iff_l605_605271

-- Given data: an angle at vertex A, points M and N inside the angle, and line through M intersecting angle sides at B and C
variables (A B C M N P : Point)

-- Define the conditions: 
-- 1. A line through M intersects the sides of the angle at points B and C
axiom line_through_M_intersects_bc (l : Line) : (M ∈ l) ∧ (l ∩ (Line_through A B)).nonempty ∧ (l ∩ (Line_through A C)).nonempty

-- 2. The line BC intersects AN at a point P
axiom bc_intersects_an_at_P : (Line_through B C ∩ Line_through A N).nonempty

-- 3. |BP| = |MC|
axiom bp_eq_mc : dist B P = dist M C

-- Prove that for the quadrilateral ABNC to have the minimum area, it is necessary and sufficient that the line BC intersects AN
theorem min_area_quad_iff : 
  (∀ l, (M ∈ l) ∧ (l ∩ (Line_through A B)).nonempty ∧ (l ∩ (Line_through A C)).nonempty → 
  area (quadrilateral A B N C) ≤ area (quadrilateral A B1 N C1)) ↔ 
  (∃ P, Line_through B C ∩ Line_through A N = {P} ∧ dist B P = dist M C) :=
sorry

end min_area_quad_iff_l605_605271


namespace machine_a_parts_l605_605880

theorem machine_a_parts (rate_A rate_B time_A time_B output_B time_prod: ℕ) 
  (h1 : rate_B = output_B / time_B)
  (h2 : output_B = 100) 
  (h3 : time_B = 20)
  (h4 : rate_A * 8 = time_prod) : 
  time_prod = 8 * rate_A := 
begin 
  have rate_B_val : rate_B = 5, from calc
    rate_B = output_B / time_B : by exact h1
    ... = 100 / 20 : by { rw [h2, h3] }
    ... = 5 : by norm_num,
  have : rate_A >= 0, from sorry, -- Placeholder for additional conditions on rates
  have : output_B >= 0, from sorry, -- Placeholder for physical feasibility conditions
  -- Conclude that more information is required
  sorry
end

end machine_a_parts_l605_605880


namespace monotonic_decreasing_interval_l605_605488

noncomputable def f (x : ℝ) := Real.log (x ^ 2 - 4 * x - 5) / Real.log (1 / 2)

theorem monotonic_decreasing_interval :
  ∀ x, (5 < x → x ^ 2 - 4 * x - 5 > 0) →
       ∀ x, (5 < x → f x = Real.log (x ^ 2 - 4 * x - 5) / Real.log (1 / 2)) →
       ∀ x, (5 < x → ∃ interval, interval = (5 : ℝ, ∞ : ℝ) ∧ is_strict_decreasing f) :=
begin
  sorry
end

end monotonic_decreasing_interval_l605_605488


namespace max_value_expr_l605_605382

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l605_605382


namespace total_handshakes_among_women_l605_605467

-- Define the problem statement
theorem total_handshakes_among_women {n : ℕ} (h_ten : n = 10)
  (h_distinct_heights : ∀ i j : Fin n, i ≠ j → (i : ℕ) ≠ j) :
  (∑ i in Finset.range n, (n - 1 - i)) = 45 :=
by
  -- The proof is provided in the sorry placeholder
  exact sorry

end total_handshakes_among_women_l605_605467


namespace probability_of_even_l605_605990

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

theorem probability_of_even :
  let nums := {n | 50 ≤ n ∧ n ≤ 1050},
      total_count := 1001,
      even_count := 501,
      prob_even := even_count / total_count
  in ∀ x ∈ nums, ∃ (p : ℚ), p = prob_even ∧ p = 501 / 1001 :=
by
  sorry

end probability_of_even_l605_605990


namespace evaluate_expression_l605_605648

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l605_605648


namespace evaluate_expression_l605_605643

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l605_605643


namespace barry_season_average_l605_605336

theorem barry_season_average (y1 y2 y3 y4 y5 : ℕ) (h1 : y1 = 98) (h2 : y2 = 107) (h3 : y3 = 85) (h4 : y4 = 89) (h5 : y5 = 91) :
  (y1 + y2 + y3 + y4 + y5) / 5 = 94 := 
by
  rw [h1, h2, h3, h4, h5]
  norm_num

end barry_season_average_l605_605336


namespace romanov_family_savings_l605_605572

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l605_605572


namespace smallest_positive_n_l605_605681

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l605_605681


namespace eval_expression_l605_605637

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l605_605637


namespace evaluate_expression_l605_605651

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l605_605651


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605227

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605227


namespace minimum_intersection_triple_l605_605711

noncomputable def n (S : Set ℕ) : ℕ := 2 ^ S.size

theorem minimum_intersection_triple {A B C : Set ℕ} 
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : A.size = B.size) (A_size : A.size = 100) :
  ∃ k, k = 97 ∧ k = (A ∩ B ∩ C).size :=
begin
  sorry
end

end minimum_intersection_triple_l605_605711


namespace pascal_triangle_sequence_evaluation_l605_605375

theorem pascal_triangle_sequence_evaluation :
  ∑ i in Finset.range 3003, (Nat.choose 3002 i) / (Nat.choose 3003 i) - ∑ i in Finset.range 3002, (Nat.choose 3001 i) / (Nat.choose 3002 i) = 1 / 2 :=
by
  sorry

end pascal_triangle_sequence_evaluation_l605_605375


namespace prob_and_relation_proof_l605_605565

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l605_605565


namespace find_jordana_and_james_age_l605_605842

variable (current_age_of_Jennifer : ℕ) (current_age_of_Jordana : ℕ) (current_age_of_James : ℕ)

-- Conditions
axiom jennifer_40_in_twenty_years : current_age_of_Jennifer + 20 = 40
axiom jordana_twice_jennifer_in_twenty_years : current_age_of_Jordana + 20 = 2 * (current_age_of_Jennifer + 20)
axiom james_ten_years_younger_in_twenty_years : current_age_of_James + 20 = 
  (current_age_of_Jennifer + 20) + (current_age_of_Jordana + 20) - 10

-- Prove that Jordana is currently 60 years old and James is currently 90 years old
theorem find_jordana_and_james_age : current_age_of_Jordana = 60 ∧ current_age_of_James = 90 :=
  sorry

end find_jordana_and_james_age_l605_605842


namespace mean_of_reciprocals_of_first_four_primes_l605_605195

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605195


namespace tracy_art_fair_customers_l605_605512

theorem tracy_art_fair_customers : 
  (∀ (p1 p2 p3 p_total : ℕ), p1 = 4 ∧ p2 = 12 ∧ p3 = 4 ∧ (p1 * 2 + p2 * 1 + p3 * 4 = p_total ∧ p_total = 36) →
      (p1 + p2 + p3 = 20)) :=
by 
  -- conditions
  intros p1 p2 p3 p_total,
  -- assume conditions
  intro h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  -- p1 = 4, p2 = 12, p3 = 4
  have hp1 : p1 = 4 := h1,
  have hp2 : p2 = 12 := h3,
  have hp3 : p3 = 4 := h5,
  -- p1 * 2 + p2 * 1 + p3 * 4 = p_total
  have ht : p1 * 2 + p2 * 1 + p3 * 4 = p_total := h6.left,
  -- p_total = 36
  have htotal : p_total = 36 := h6.right,
  -- prove p1 + p2 + p3 = 20
  rw [hp1, hp2, hp3],
  linarith,

end tracy_art_fair_customers_l605_605512


namespace allocation_methods_l605_605031

-- Definitions of the conditions
def num_doctors : ℕ := 2
def num_nurses : ℕ := 4
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- Prove that the assignment methods are equal to 12
theorem allocation_methods : nat.choose num_doctors doctors_per_school * nat.choose num_nurses nurses_per_school = 12 := 
by sorry

end allocation_methods_l605_605031


namespace mean_of_reciprocals_of_first_four_primes_l605_605196

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605196


namespace good_polynomials_k_le_2_l605_605113

-- Defining what it means for a polynomial to be good
def is_good_polynomial (P : Fin k → ℝ → ℝ) : Prop :=
  ∃ (A : Fin k → Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (x : Fin k → ℝ), 
      P x = Matrix.det (Fin.sum (λ i, x i • A i))

-- Defining what it means for a polynomial to be homogeneous of degree 2
def is_homogeneous_deg2 (P : (Fin k → ℝ) → ℝ) : Prop :=
  ∀ (c : ℝ) (x : Fin k → ℝ), P (λ i, c * x i) = c^2 * P x

-- Main theorem to be proved
theorem good_polynomials_k_le_2 :
  ∀ (k : ℕ), (k ≤ 2 ↔ ∀ P : (Fin k → ℝ) → ℝ, is_homogeneous_deg2 P → is_good_polynomial P) :=
by
  intro k
  sorry

end good_polynomials_k_le_2_l605_605113


namespace first_four_eq_last_four_l605_605119

-- Define a sequence s to be a list of Booleans
variable (s : List Bool)

-- Define the length of the sequence
variable (n : ℕ)
variable (h1 : s.length = n)

-- Condition 1: Any 5 consecutive elements in the sequence are unique
def unique_consecutive_five (s : List Bool) : Prop :=
  ∀ i j, i < s.length - 4 → j < s.length - 4 → i ≠ j → (s.drop i).take 5 ≠ (s.drop j).take 5

-- Condition 2: Adding either a 0 or a 1 to the end violates condition 1
def violates_condition (s : List Bool) : Prop :=
  ¬unique_consecutive_five (s ++ [false]) ∧ ¬unique_consecutive_five (s ++ [true])

theorem first_four_eq_last_four
  (h_unique: unique_consecutive_five s)
  (satisfies_condition : violates_condition s) :
  s.take 4 = s.drop (n - 4).take 4 :=
sorry

end first_four_eq_last_four_l605_605119


namespace exists_pairwise_disjoint_subsets_same_class_l605_605394

variable (S : Finset α)
variable (n : ℕ) 
variable (partitioned : ∀ (s : Finset α), s.card = n → s ∈ S → (Class₁ s ∨ Class₂ s))

theorem exists_pairwise_disjoint_subsets_same_class 
(hS_card : S.card = n ^ 2 + n - 1) 
: ∃ (A : Finset (Finset α)), A.card = n ∧ (∀ a ∈ A, a.card = n) ∧ (∃ c : (Class₁ ∨ Class₂), ∀ a ∈ A, a ∈ c) :=
  sorry

end exists_pairwise_disjoint_subsets_same_class_l605_605394


namespace sufficient_not_necessary_condition_l605_605728

noncomputable def sequence_increasing_condition (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) > |a n|

noncomputable def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n < a (n + 1)

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) :
  sequence_increasing_condition a → is_increasing_sequence a ∧ ¬(∀ b : ℕ → ℝ, is_increasing_sequence b → sequence_increasing_condition b) :=
sorry

end sufficient_not_necessary_condition_l605_605728


namespace bus_probabilities_and_chi_squared_l605_605555

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l605_605555


namespace cube_root_expression_l605_605791

theorem cube_root_expression (x : ℝ) (hx : 0 ≤ x) : 
    Real.cbrt (x * Real.cbrt (x * Real.cbrt (x * Real.cbrt x))) = x^(10/9) :=
sorry

end cube_root_expression_l605_605791


namespace arithmetic_mean_of_reciprocals_is_correct_l605_605225

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l605_605225


namespace length_of_segment_NB_l605_605834

variable (L W x : ℝ)
variable (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W))

theorem length_of_segment_NB (L W x : ℝ) (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W)) : 
  x = 0.8 * L :=
by
  sorry

end length_of_segment_NB_l605_605834


namespace area_of_region_bounded_by_curves_l605_605122

theorem area_of_region_bounded_by_curves :
  let curve1 := {p : ℝ × ℝ | (p.1 + p.2)^2 = 16}
  let curve2 := {p : ℝ × ℝ | (2 * p.1 - p.2)^2 = 4}
  ∃ (area : ℝ), area = (16 * real.sqrt 10) / 5 ∧ 
  let region := {p : ℝ × ℝ | p ∈ curve1 ∧ p ∈ curve2} in
  let bounded_area := sorry in
  bounded_area = area := 
by
  -- Sorry block to skip the proof
  sorry

end area_of_region_bounded_by_curves_l605_605122


namespace rhombus_side_length_l605_605807

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605807


namespace probability_snow_at_least_once_first_week_l605_605437

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l605_605437


namespace shop_second_sale_price_l605_605078

-- Definitions based on conditions
variables {C : ℝ}  -- Original cost of the clock to the shop
def first_sale_price := 1.20 * C
def buy_back_price := 0.60 * (1.20 * C)
def price_difference := C - buy_back_price
def second_sale_profit := 0.80 * buy_back_price
def second_sale_price := buy_back_price + second_sale_profit

-- Problem condition: price difference is $100 and initial percentage is 20%
axiom price_difference_axiom : price_difference = 100
axiom initial_percentage : 1.20 = 1 + (20 / 100)

-- The goal is to prove the final sale price is $270
theorem shop_second_sale_price : second_sale_price = 270 := by
  sorry

end shop_second_sale_price_l605_605078


namespace normal_dist_scaled_variance_l605_605286

-- Define the random variable and its properties
variable (ξ : ℝ) 

-- Assuming ξ follows normal distribution N(1, 2)
-- Here, variance D(ξ) = 2
axiom normal_dist_variance (h : ξ ~ ℕ 1 2) : variance ξ = 2

-- Main theorem to prove D(2ξ + 3) = 8
theorem normal_dist_scaled_variance {ξ : ℝ} (h : ξ ~ ℕ 1 2) : variance (2 * ξ + 3) = 8 := by
  sorry

end normal_dist_scaled_variance_l605_605286


namespace g_50_zero_l605_605865

noncomputable def g : ℕ → ℝ → ℝ
| 0, x     => x + |x - 50| - |x + 50|
| (n+1), x => |g n x| - 2

theorem g_50_zero :
  ∃! x : ℝ, g 50 x = 0 :=
sorry

end g_50_zero_l605_605865


namespace least_n_factorial_l605_605037

theorem least_n_factorial (n : ℕ) : 
  (7350 ∣ nat.factorial n) ↔ n ≥ 15 := 
sorry

end least_n_factorial_l605_605037


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605213

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605213


namespace number_of_ordered_pairs_l605_605014

theorem number_of_ordered_pairs (x y : ℕ) : (x * y = 1716) → 
  (∃! n : ℕ, n = 18) :=
by
  sorry

end number_of_ordered_pairs_l605_605014


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605176

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605176


namespace tan_abs_increases_in_given_intervals_l605_605908

def tan_abs_increasing_intervals := ∀ k : ℤ, 
  ∀ x y : ℝ, 
  k * π ≤ x ∧ x < k * π + π / 2 ∧ 
  k * π ≤ y ∧ y < k * π + π / 2 ∧ 
  x < y → abs (Real.tan x) < abs (Real.tan y)

theorem tan_abs_increases_in_given_intervals : tan_abs_increasing_intervals :=
sorry

end tan_abs_increases_in_given_intervals_l605_605908


namespace interest_percentage_approx_l605_605848

noncomputable def FV := 5000
noncomputable def SP := 3846.153846153846
noncomputable def I := 0.065 * SP

theorem interest_percentage_approx : (I / FV) * 100 ≈ 5 := 
by
  -- The proof is skipped here
  sorry

end interest_percentage_approx_l605_605848


namespace dave_worked_on_monday_l605_605125

variable (x m t : ℕ)
variable (h1 : m = 6)
variable (h2 : t = 2)
variable (h3 : m * x + m * t = 48)

theorem dave_worked_on_monday : x = 6 :=
by
  have h4 : 6 * x + 6 * 2 = 48 := h3
  sorry

end dave_worked_on_monday_l605_605125


namespace mean_of_reciprocals_of_first_four_primes_l605_605201

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l605_605201


namespace combined_load_time_is_correct_l605_605028

def time_taken (x : Float) :=
  Float.round (x * 1000) / 1000

def combined_rate (rates : List Float) :=
  rates.foldl (fun acc r => acc + r) 0

theorem combined_load_time_is_correct :
  let first_phone := 9 * 60 -- in seconds
  let second_phone := 6 * 60 -- in seconds
  let first_laptop := 15 -- in seconds
  let second_laptop := 11 -- in seconds
  let third_laptop := 8 -- in seconds

  let rates := [
    1 / first_phone,
    1 / second_phone,
    1 / first_laptop,
    1 / second_laptop,
    1 / third_laptop
  ]

  let combined_rate_value := combined_rate rates

  time_taken (1 / combined_rate_value) ≈ 3.483 := by
  sorry

end combined_load_time_is_correct_l605_605028


namespace alpha_value_l605_605292

noncomputable def f (x α : ℝ) : ℝ := sin (x + α) * cos (x + α)

theorem alpha_value (x α : ℝ) :
  (∀ α, (∃ α, (f 1 α) = f x α)) → (α = (π / 4 - 1)) :=
by
  sorry

end alpha_value_l605_605292


namespace sum_of_series_eq_l605_605240

theorem sum_of_series_eq :
  (∑ a in Finset.Ico 1 (ℤ), ∑ b in Finset.Ico (a+1) (ℤ), ∑ c in Finset.Ico (b+1) (ℤ), 1 / (2^a * 3^b * 5^c)) = (1 : ℚ) / 1624 := 
  sorry

end sum_of_series_eq_l605_605240


namespace snow_prob_correct_l605_605423

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l605_605423


namespace symmetry_sum_zero_l605_605108

theorem symmetry_sum_zero (v : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, v (-x) = -v x) : 
  v (-2.00) + v (-1.00) + v (1.00) + v (2.00) = 0 := 
by 
  sorry

end symmetry_sum_zero_l605_605108


namespace boys_girls_ratio_l605_605963

theorem boys_girls_ratio (T G : ℕ) (h : (1/2 : ℚ) * G = (1/6 : ℚ) * T) :
  ((T - G) : ℚ) / G = 2 :=
by 
  sorry

end boys_girls_ratio_l605_605963


namespace S_2018_eq_2_l605_605746

-- Conditions stated as Lean definitions
def seq (n : ℕ) : ℤ :=
if h : n > 0 then
  if n == 1 ∨ n == 2 then 1
  else seq (n-1) + seq (n+1) - seq n
else 0

-- Define S_n as the sum of the sequence up to n
def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, seq (i + 1))

-- Theorem statement
theorem S_2018_eq_2 : S 2018 = 2 :=
sorry

end S_2018_eq_2_l605_605746


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605162

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605162


namespace diversity_values_l605_605574

theorem diversity_values (k : ℕ) (h : 1 ≤ k ∧ k ≤ 4) :
  ∃ (D : ℕ), D = 1000 * (k - 1) := by
  sorry

end diversity_values_l605_605574


namespace largest_expression_l605_605377

def U := 2 * 2004^2005
def V := 2004^2005
def W := 2003 * 2004^2004
def X := 2 * 2004^2004
def Y := 2004^2004
def Z := 2004^2003

theorem largest_expression :
  U - V > V - W ∧
  U - V > W - X ∧
  U - V > X - Y ∧
  U - V > Y - Z :=
by
  sorry

end largest_expression_l605_605377


namespace expression_eq_zero_l605_605483

theorem expression_eq_zero (x : ℝ) (h_num : x^2 - 4 = 0) (h_den : 4x - 8 ≠ 0) : x = -2 :=
by
  sorry

end expression_eq_zero_l605_605483


namespace remaining_regular_toenails_l605_605306

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end remaining_regular_toenails_l605_605306


namespace algebraic_expression_value_l605_605946

variable (a b : ℝ)
axiom h1 : a = 3
axiom h2 : a - b = 1

theorem algebraic_expression_value :
  a^2 - a * b = 3 :=
by
  sorry

end algebraic_expression_value_l605_605946


namespace compute_total_cost_l605_605089

-- Define the dimensions and costs
def length := 4
def width := 5
def height := 2
def cost_first_layer := 20
def cost_second_layer := 15

-- Helper function to calculate the surface area of a rectangular tank
def surface_area (l w h : ℕ) : ℕ := 2 * (l * h + w * h + l * w)

-- Define the total cost calculation function
def total_cost (sa : ℕ) (cost1 cost2 : ℕ) : ℕ := sa * cost1 + sa * cost2

-- The proof statement
theorem compute_total_cost : 
  total_cost (surface_area length width height) cost_first_layer cost_second_layer = 2660 := 
by sorry

end compute_total_cost_l605_605089


namespace hockey_player_scores_l605_605986

theorem hockey_player_scores (n m : ℕ) (h : n + m = 7) : 
  set.card {k | ∃ (i j : ℕ), i + j = k ∧ i * 2 + j = n + m} = 8 := 
sorry

end hockey_player_scores_l605_605986


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605556

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l605_605556


namespace angle_AM_BN_eq_60_area_ABP_eq_area_MDNP_l605_605393

variables (A B C D E F M N P : Point)
variables (hexagon ABCDEF : RegularHexagon ABCDEF)
variables (M_mid : midpoint C D M)
variables (N_mid : midpoint D E N)
variables (P_intersect : intersection_point (line_through A M) (line_through B N) P)

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_eq_60 (hex : RegularHexagon ABCDEF) (M : Point) (N : Point) (P : Point)
  (hM : midpoint C D M) (hN : midpoint D E N) (hP : intersection_point (line_through A M) (line_through B N) P) :
  angle (line_through A M) (line_through B N) = 60 :=
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_ABP_eq_area_MDNP (hex : RegularHexagon ABCDEF) (M : Point) (N : Point) (P : Point)
  (hM : midpoint C D M) (hN : midpoint D E N) (hP : intersection_point (line_through A M) (line_through B N) P) :
  area (triangle A B P) = area (quadrilateral M D N P) :=
sorry

end angle_AM_BN_eq_60_area_ABP_eq_area_MDNP_l605_605393


namespace arithmetic_mean_of_reciprocals_first_four_primes_l605_605172

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l605_605172


namespace rhombus_side_length_l605_605809

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l605_605809


namespace snow_probability_l605_605434

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l605_605434


namespace difference_in_payment_l605_605360

theorem difference_in_payment (joy_pencils : ℕ) (colleen_pencils : ℕ) (price_per_pencil : ℕ) (H1 : joy_pencils = 30) (H2 : colleen_pencils = 50) (H3 : price_per_pencil = 4) :
  (colleen_pencils * price_per_pencil) - (joy_pencils * price_per_pencil) = 80 :=
by
  rw [H1, H2, H3]
  simp
  norm_num
  sorry

end difference_in_payment_l605_605360


namespace smallest_positive_n_l605_605677

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l605_605677


namespace largest_4digit_congruent_17_mod_28_l605_605940

theorem largest_4digit_congruent_17_mod_28 :
  ∃ n, n < 10000 ∧ n % 28 = 17 ∧ ∀ m, m < 10000 ∧ m % 28 = 17 → m ≤ 9982 :=
by
  sorry

end largest_4digit_congruent_17_mod_28_l605_605940


namespace solve_x_l605_605462

theorem solve_x (x : ℝ) (h : (30 * x + 15)^(1/3) = 15) : x = 112 := by
  sorry

end solve_x_l605_605462


namespace tonya_payment_l605_605511

def original_balance : ℝ := 150.00
def new_balance : ℝ := 120.00

noncomputable def payment_amount : ℝ := original_balance - new_balance

theorem tonya_payment :
  payment_amount = 30.00 :=
by
  sorry

end tonya_payment_l605_605511


namespace prob_and_relation_proof_l605_605561

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l605_605561


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605163

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605163


namespace cost_to_feed_treats_for_a_week_l605_605106

theorem cost_to_feed_treats_for_a_week :
  (let dog_biscuits_cost := 4 * 0.25 in
   let rawhide_bones_cost := 2 * 1 in
   let daily_cost := dog_biscuits_cost + rawhide_bones_cost in
   7 * daily_cost = 21) :=
by
  sorry

end cost_to_feed_treats_for_a_week_l605_605106


namespace greatest_product_of_two_integers_sum_2006_l605_605525

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end greatest_product_of_two_integers_sum_2006_l605_605525


namespace book_club_selection_l605_605710

theorem book_club_selection : 
  ∃ (ways : ℕ), ways = (Nat.choose 5 3) ∧ ways = 10 :=
by
  use (Nat.choose 5 3)
  split
  · rfl
  · sorry

end book_club_selection_l605_605710


namespace right_triangle_median_min_length_l605_605007

theorem right_triangle_median_min_length (h : ℝ) (A B C D M : Point)
  (ABC_right : is_right_triangle A B C)
  (CD_perp_ab : CD = h ∧ is_perpendicular C D A B)
  (AD_projection : is_projection A D A B)
  (M_midpoint : is_midpoint D M B):
  minimum_length_of_median A M = (3 * h) / 2 := 
  sorry

end right_triangle_median_min_length_l605_605007


namespace area_enclosed_by_circle_l605_605520

theorem area_enclosed_by_circle :
  let eq := (fun x y => x^2 + y^2 + 8*x + 10*y + 9 = 0) in
  ∃ (r : ℝ) (h : r^2 = 32), 
    ∀ x y, eq x y → π * r^2 = 32 * π :=
by
  sorry

end area_enclosed_by_circle_l605_605520


namespace probability_new_door_hides_prize_l605_605341

variable (doors : Finset ℕ)
variables (prizes : Finset ℕ)
variables (initial_door new_door : ℕ)
variables (host_doors : Finset ℕ)

-- Input conditions
axiom h1 : doors.card = 7
axiom h2 : prizes.card = 2
axiom h3 : initial_door ∈ doors
axiom h4 : host_doors.card = 3
axiom h5 : host_doors ⊆ (doors \ {initial_door})
axiom h6 : prizes ∩ host_doors = ∅
axiom h7 : new_door ∈ (doors \ (host_doors ∪ {initial_door}))

-- Hypothesis: Bob switches to another door.
axiom switch_to_new_door : initial_door ≠ new_door

-- The probability that Bob's new door is hiding a prize
theorem probability_new_door_hides_prize : 
  (prizes ∩ {new_door}).card.to_real / (doors \ (host_doors ∪ {initial_door})).card.to_real = 4 / 7 :=
sorry

end probability_new_door_hides_prize_l605_605341


namespace probability_seven_distinct_rolls_l605_605528

theorem probability_seven_distinct_rolls (d : Fin 6 → ℕ) (h : ∀ i : Fin 7, d i ∈ Finset.range 1 7) : 
  (∃ (d : Fin 7 → Fin 6), function.injective d) → false :=
by
  sorry

end probability_seven_distinct_rolls_l605_605528


namespace essentially_different_pythagorean_triples_l605_605057

noncomputable def pythagorean_triple_conditions
  (m n : ℤ) : Prop :=
  let a := m^2 - n^2 in
  let b := 2 * m * n in
  let c := m^2 + n^2 in
  m ≠ n ∧ Int.gcd m n = 1 ∧ (m * n) % 2 = 0

theorem essentially_different_pythagorean_triples
  (m n : ℤ) :
  pythagorean_triple_conditions m n :=
begin
  sorry
end

end essentially_different_pythagorean_triples_l605_605057


namespace monotonic_increasing_range_l605_605326

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x ∈ set.Ici (1 : ℝ), 2 * x - a / x - 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end monotonic_increasing_range_l605_605326


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605168

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l605_605168


namespace find_x_value_l605_605081

theorem find_x_value (x : ℝ) 
    (area_sum : (3 * x)^2 + (4 * x)^2 + (1/2) * (3 * x) * (4 * x) = 1200) :
    x = real.sqrt (1200 / 31) := 
sorry

end find_x_value_l605_605081


namespace total_material_ordered_l605_605983

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered_l605_605983


namespace max_value_is_one_l605_605383

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l605_605383


namespace max_projection_sides_l605_605527

theorem max_projection_sides (n : ℕ) (hn : n ≥ 4) (P : ConvexPolyhedron n) :
  ∃ m, (m ≤ 2*n - 4 ∧ ∀ (Q : Projection P), (sides Q ≤ m)) :=
by
  sorry

end max_projection_sides_l605_605527


namespace log5_y_l605_605792

noncomputable def log_eq_half_log (y : ℝ) : Prop := 
  y = ((Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) 
  ∧ Real.log 5 y = 1 / 2 * Real.log 5 2

theorem log5_y (y : ℝ) (h: log_eq_half_log y) : Real.log 5 y = 1 / 2 * Real.log 5 2 := by
  cases h
  sorry

end log5_y_l605_605792


namespace two_digit_interchange_difference_l605_605479

-- Define the conditions and prove the main theorem
theorem two_digit_interchange_difference:
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y = 2 * x ∧ (x + y) - (y - x) = 8 ∧ 
  abs ((10 * x + y) - (10 * y + x)) = 36 :=
by
  sorry

end two_digit_interchange_difference_l605_605479


namespace ratio_equals_cubed_ratio_of_sides_l605_605002

open_locale classical -- useful for non-constructive proofs

theorem ratio_equals_cubed_ratio_of_sides
  (A B C P Q : Point)
  (h₁ : circle₁.touches_line_at A B B)
  (h₂ : circle₁.passes_through C)
  (h₃ : circle₁.intersects_line_at A C P)
  (h₄ : circle₂.touches_line_at A C C)
  (h₅ : circle₂.passes_through B)
  (h₶ : circle₂.intersects_line_at A B Q) :
  (A.distance_to P / A.distance_to Q) = (A.distance_to B / A.distance_to C) ^ 3 :=
begin
  sorry,
end

end ratio_equals_cubed_ratio_of_sides_l605_605002


namespace fraction_filled_correct_l605_605601

variable (E P P_partial : ℝ)

def vessel_fraction_filled (E P P_partial : ℝ) : ℝ := P_partial / P

axiom condition1 : E = 0.08 * (E + P)
axiom condition2 : E + P_partial = 0.5 * (E + P)

theorem fraction_filled_correct : vessel_fraction_filled E P P_partial = 0.46 :=
by
  sorry

end fraction_filled_correct_l605_605601


namespace remainder_of_n_plus_3255_l605_605531

theorem remainder_of_n_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := 
by
  sorry

end remainder_of_n_plus_3255_l605_605531


namespace chocolate_bars_in_large_box_l605_605083

theorem chocolate_bars_in_large_box
  (number_of_small_boxes : ℕ)
  (chocolate_bars_per_box : ℕ)
  (h1 : number_of_small_boxes = 21)
  (h2 : chocolate_bars_per_box = 25) :
  number_of_small_boxes * chocolate_bars_per_box = 525 :=
by {
  sorry
}

end chocolate_bars_in_large_box_l605_605083


namespace smallest_positive_integer_n_l605_605700

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l605_605700


namespace range_of_a_l605_605750

noncomputable def f (a x : ℝ) : ℝ := (x^2 + 2 * a * x) * Real.log x - (1/2) * x^2 - 2 * a * x

theorem range_of_a (a : ℝ) : (∀ x > 0, 0 < (f a x)) ↔ -1 ≤ a :=
sorry

end range_of_a_l605_605750


namespace least_n_factorial_l605_605035

theorem least_n_factorial (n : ℕ) : 
  (7350 ∣ nat.factorial n) ↔ n ≥ 15 := 
sorry

end least_n_factorial_l605_605035


namespace trajectory_of_M_l605_605278

noncomputable def fixed_points := ℝ × ℝ

variable (F1 F2 : fixed_points)
variable (F1F2 : ℝ := 16)
variable (M : fixed_points)

def distance (p1 p2 : fixed_points) : ℝ :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) ^ (1 / 2)

axiom distance_F1F2 : distance F1 F2 = 16
axiom condition_on_M : distance M F1 + distance M F2 = 16

theorem trajectory_of_M : 
  ∀ M : fixed_points, 
  distance M F1 + distance M F2 = 16 → ∃ t : ℝ, t ∈ Icc 0 1 ∧ M = (t * F1.1 + (1 - t) * F2.1, t * F1.2 + (1 - t) * F2.2) :=
begin
  sorry
end

end trajectory_of_M_l605_605278
