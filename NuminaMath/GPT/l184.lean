import Mathlib

namespace arg_cubed_sum_eq_pi_l184_184353

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry

axiom norm_z1 : complex.abs z1 = 3
axiom norm_z2 : complex.abs z2 = 5
axiom norm_z1_z2 : complex.abs (z1 + z2) = 7

theorem arg_cubed_sum_eq_pi : complex.arg (z1^3 + z2^3) = real.pi := sorry

end arg_cubed_sum_eq_pi_l184_184353


namespace sqrt_pow_expr_eq_27_l184_184874

theorem sqrt_pow_expr_eq_27 : (sqrt ((sqrt 3)^3))^4 = 27 := by
  sorry

end sqrt_pow_expr_eq_27_l184_184874


namespace problem_statement_l184_184700

theorem problem_statement (a x : ℝ) (h1 : ∀ x : ℝ, x^2 + a * x + a^2 ≥ 0) 
                            (h2 : ∀ x : ℝ, sin x + cos x ≤ real.sqrt 2) :
                            (∀ x : ℝ, x^2 + a * x + a^2 ≥ 0 ∧ ¬ ∃ x : ℝ, sin x + cos x = 2) :=
sorry

end problem_statement_l184_184700


namespace functional_relationship_find_selling_price_maximum_profit_l184_184796

noncomputable def linear_relation (x : ℤ) : ℤ := -5 * x + 150
def profit_function (x : ℤ) : ℤ := -5 * x * x + 200 * x - 1500

theorem functional_relationship (x : ℤ) (hx : 10 ≤ x ∧ x ≤ 15) : linear_relation x = -5 * x + 150 :=
by sorry

theorem find_selling_price (h : ∃ x : ℤ, (10 ≤ x ∧ x ≤ 15) ∧ ((-5 * x + 150) * (x - 10) = 320)) :
  ∃ x : ℤ, x = 14 :=
by sorry

theorem maximum_profit (hx : 10 ≤ 15 ∧ 15 ≤ 15) : profit_function 15 = 375 :=
by sorry

end functional_relationship_find_selling_price_maximum_profit_l184_184796


namespace number_of_truthful_dwarfs_l184_184139

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184139


namespace cos_arcsin_eq_l184_184845

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l184_184845


namespace min_value_expr_l184_184858

theorem min_value_expr (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, m > 0 ∧ (forall (n : ℕ), 0 < n → (n/2 + 50/n : ℝ) ≥ 10) ∧ 
           (n = 10) → (n/2 + 50/n : ℝ) = 10 :=
by
  sorry

end min_value_expr_l184_184858


namespace average_after_11th_inning_is_30_l184_184039

-- Define the conditions as Lean 4 definitions
def score_in_11th_inning : ℕ := 80
def increase_in_avg : ℕ := 5
def innings_before_11th : ℕ := 10

-- Define the average before 11th inning
def average_before (x : ℕ) : ℕ := x

-- Define the total runs before 11th inning
def total_runs_before (x : ℕ) : ℕ := innings_before_11th * (average_before x)

-- Define the total runs after 11th inning
def total_runs_after (x : ℕ) : ℕ := total_runs_before x + score_in_11th_inning

-- Define the new average after 11th inning
def new_average_after (x : ℕ) : ℕ := total_runs_after x / (innings_before_11th + 1)

-- Theorem statement
theorem average_after_11th_inning_is_30 : 
  ∃ (x : ℕ), new_average_after x = average_before x + increase_in_avg → new_average_after 25 = 30 :=
by
  sorry

end average_after_11th_inning_is_30_l184_184039


namespace part1_part2_part3_l184_184615

-- Part 1: Prove that if the tangent line condition holds, then a = -2
theorem part1 (a : ℝ) (h : ∀ (x : ℝ), 6 * x - 2 * (1 / 2 * x ^ 2 - a * Real.log x) - 5 = 0) : 
  a = -2 := sorry

-- Part 2: Prove the range for a under the given conditions
theorem part2 (a : ℝ) (h : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (1 / 2 * x₁ ^ 2 + a * Real.log x₁ - (1 / 2 * x₂ ^ 2 + a * Real.log x₂)) / (x₁ - x₂) > 2) : 
  1 ≤ a := sorry

-- Part 3: Prove the range for a given an interval condition
theorem part3 (a : ℝ) (h : ∃ x_0 ∈ Icc 1 Real.exp 1, (1 * x_0 - 1 / 2 * x_0 ^ 2 + a * Real.log x_0 + a / x_0 - 2) < 0) : 
  a ∈ Set.Iio (-2) ∪ Set.Ioi ((Real.exp 1 ^ 2 + 1) / (Real.exp 1 - 1)) := sorry

end part1_part2_part3_l184_184615


namespace volume_filled_water_surface_area_exposed_air_l184_184516

def cone_volume (π : ℝ) (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def cone_surface_area (π : ℝ) (r h : ℝ) : ℝ := π * r * sqrt (r^2 + h^2)

theorem volume_filled_water {π : ℝ} {r h : ℝ} :
  (cone_volume π (2 / 3 * r) (2 / 3 * h)) / (cone_volume π r h) = 8 / 27 := 
by sorry

theorem surface_area_exposed_air {π : ℝ} {r h : ℝ} :
  (cone_surface_area π r h - cone_surface_area π (2 / 3 * r) (2 / 3 * h)) / (cone_surface_area π r h) = 5 / 9 := 
by sorry

end volume_filled_water_surface_area_exposed_air_l184_184516


namespace problem_solution_l184_184923

variables (x y : ℝ)

def cond1 : Prop := 4 * x + y = 12
def cond2 : Prop := x + 4 * y = 18

theorem problem_solution (h1 : cond1 x y) (h2 : cond2 x y) : 20 * x^2 + 24 * x * y + 20 * y^2 = 468 :=
by
  -- Proof would go here
  sorry

end problem_solution_l184_184923


namespace composite_numbers_condition_l184_184671

theorem composite_numbers_condition (n : ℕ) (h : n > 1 ∧ ¬Prime n) : 
  (∃ m, ∀ d, d ∣ n → d ≠ n → d ≠ 1 → (d + 1 ∣ m ∧ d + 1 ≠ m ∧ d + 1 ≠ 1)) → 
  (n = 4 ∨ n = 8) :=
begin
  sorry
end

end composite_numbers_condition_l184_184671


namespace sequence_hypothesis_l184_184756

-- Defining the sequence
def a : ℕ → ℝ
| 0       := 1 -- a₁ = 1
| 1       := 1 -- a₂ = 1
| (n + 2) := 1 + a (n) / a (n + 1) -- aₙ₊₁ = 1 + a₍ₙ₋₁₎ / aₙ for n ≥ 2

-- Hypothesis conditions
lemma a_5_gt_2 : a 4 > 2 := sorry
lemma a_6_lt_2 : a 5 < 2 := sorry
lemma a_7_gt_2 : a 6 > 2 := sorry
lemma a_8_lt_2 : a 7 < 2 := sorry

-- Main theorem to prove the hypothesis
theorem sequence_hypothesis:
  (∀ n ≥ 4, (n % 2 = 1 → a n > 2) ∧ (n % 2 = 0 → a n < 2)) := 
begin
  -- Using induction and predefined conditions
  sorry
end

end sequence_hypothesis_l184_184756


namespace sequence_inequality_l184_184587

-- Defining the lean theorem
theorem sequence_inequality (k : ℕ) (a : Fin (2 * k + 1) → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_nonincreasing : ∀ i j, i ≤ j → a i ≥ a j) :
  ∑ i in Finset.filter (λ i : Fin (2 * k + 1), i.val % 2 = 0) Finset.univ (a i)^2 -
  ∑ i in Finset.filter (λ i : Fin (2 * k + 1), i.val % 2 = 1) Finset.univ (a i)^2 ≥
  (∑ i in Finset.filter (λ i : Fin (2 * k + 1), i.val % 2 = 0) Finset.univ (a i) -
  ∑ i in Finset.filter (λ i : Fin (2 * k + 1), i.val % 2 = 1) Finset.univ (a i))^2 :=
sorry

end sequence_inequality_l184_184587


namespace age_sum_proof_l184_184776

theorem age_sum_proof (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 20) : a + b + c = 52 :=
by
  sorry

end age_sum_proof_l184_184776


namespace total_pull_ups_per_week_l184_184374

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l184_184374


namespace fishmonger_total_sales_l184_184440

theorem fishmonger_total_sales (first_week_sales : ℕ) (multiplier : ℕ) : 
  first_week_sales = 50 → multiplier = 3 → first_week_sales + first_week_sales * multiplier = 200 :=
by
  intros h_first h_mult
  rw [h_first, h_mult]
  simp
  sorry

end fishmonger_total_sales_l184_184440


namespace city_of_pythagoras_schools_l184_184191

noncomputable def number_of_schools
  (n : ℕ)
  (team_size : ℕ)
  (total_students : ℕ)
  (positions : list ℕ)
  (median_position : ℕ) : ℕ :=
  if 41 <= median_position ∧ median_position < 82
  then 40
  else 0

theorem city_of_pythagoras_schools :
  let team_size := 4
  let positions := [41, 82]
  let total_students := team_size * 40
  let median_position := 2 * team_size
  number_of_schools 40 team_size total_students positions median_position = 40 :=
by
  sorry

end city_of_pythagoras_schools_l184_184191


namespace odd_function_g_l184_184987

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184987


namespace odd_function_check_l184_184946

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184946


namespace modulus_of_complex_number_l184_184921

-- Definitions of the real numbers x and y such that the given condition holds
variables {x y : ℝ}

-- The given condition
def given_condition : Prop := i * (x + y * i) = 3 + 4 * i

-- The proof problem: prove the modulus of (x + yi) is 5 given the condition
theorem modulus_of_complex_number (h : given_condition) : complex.abs (x + y * complex.i) = 5 :=
sorry

end modulus_of_complex_number_l184_184921


namespace domain_of_f_of_f_l184_184612

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_f_of_f :
  {x : ℝ | x ≠ -3 ∧ x ≠ -8 / 5} =
  {x : ℝ | ∃ y : ℝ, f x = y ∧ y ≠ -3 ∧ x ≠ -3} :=
by
  sorry

end domain_of_f_of_f_l184_184612


namespace triangle_movement_l184_184317

theorem triangle_movement (x1 y1 x2 y2 x3 y3 : ℝ) : 
  let new_y1 := y1 + 3
      new_y2 := y2 + 3
      new_y3 := y3 + 3 in
  (new_y1 = y1 + 3) ∧ (new_y2 = y2 + 3) ∧ (new_y3 = y3 + 3) :=
by
  sorry

end triangle_movement_l184_184317


namespace selection_case_1_selection_case_2_l184_184579

variable (F : Finset ℕ) (M : Finset ℕ)
-- Assuming the group of 5 female students and 4 male students
variables (hF : F.card = 5) (hM : M.card = 4)

theorem selection_case_1 :
  (choose M 2) * (choose F 2) * (factorial 4) = 1440 := sorry

theorem selection_case_2 :
  (((choose M 1) * (choose F 3)) + ((choose M 2) * (choose F 2)) + ((choose M 3) * (choose F 1))) * (factorial 4) = 2880 := sorry

end selection_case_1_selection_case_2_l184_184579


namespace remaining_liquid_weight_l184_184710

theorem remaining_liquid_weight 
  (liqX_content : ℝ := 0.20)
  (water_content : ℝ := 0.80)
  (initial_solution : ℝ := 8)
  (evaporated_water : ℝ := 2)
  (added_solution : ℝ := 2)
  (new_solution_fraction : ℝ := 0.25) :
  ∃ (remaining_liquid : ℝ), remaining_liquid = 6 := 
by
  -- Skip the proof to ensure the statement is built successfully
  sorry

end remaining_liquid_weight_l184_184710


namespace running_time_l184_184627

def side_length : ℝ := 40
def speed_km_per_hr : ℝ := 9

def perimeter := 4 * side_length
def conversion_factor := 1000 / 3600
def speed_m_per_s := speed_km_per_hr * conversion_factor
def distance := perimeter
def time := distance / speed_m_per_s

theorem running_time :
  side_length = 40 →
  speed_km_per_hr = 9 →
  time = 64 :=
by
  intros h1 h2
  unfold side_length at h1
  unfold speed_km_per_hr at h2
  calc time = distance / speed_m_per_s : rfl
      ... = perimeter / speed_m_per_s : by rw h1
      ... = (4 * side_length) / (speed_km_per_hr * conversion_factor) : by rw perimeter
      ... = (4 * 40) / (9 * (1000 / 3600)) : by rw [h1, h2]
      ... = 160 / (9 * (1000 / 3600)) : rfl
      ... = 160 / (9 / 3.6) : rfl
      ... = 160 / 2.5 : rfl
      ... = 64 : rfl

end running_time_l184_184627


namespace complement_union_l184_184623

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}
def complement_U_A : Set ℕ := U \ A

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement_U_A ∪ B) = {0, 2, 4} := by
  sorry

end complement_union_l184_184623


namespace count_valid_sequences_l184_184313

def valid_sequence (seq : List ℕ) : Prop :=
  seq.head = 1 ∧ seq.last = 6 ∧ 
  ∀ (i : ℕ), (1 ≤ i ∧ i ≤ 4) →
  ¬ ((seq.get? i = some (seq.get! (i - 1) + 1) ∧
      seq.get? (i + 1) = some (seq.get! i + 1)) ∨
     (seq.get? i = some (seq.get! (i - 1) - 1) ∧
      seq.get? (i + 1) = some (seq.get! i - 1)))

theorem count_valid_sequences: 
  (List.permutations [1, 2, 3, 4, 5, 6]).count valid_sequence = 4 :=
sorry

end count_valid_sequences_l184_184313


namespace part1_part2_l184_184831

open Real

variable {x y a: ℝ}

-- Condition for the second proof to avoid division by zero
variable (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4)

theorem part1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * (x * y) := 
by sorry

theorem part2 (h1: a ≠ 1) (h2: a ≠ 4) (h3: a ≠ -4) : 
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := 
by sorry

end part1_part2_l184_184831


namespace sum_perpendiculars_equilateral_triangle_6cm_l184_184101

noncomputable def sum_perpendiculars_equilateral_triangle (s : ℝ) : ℝ :=
  let h := (sqrt 3 / 2) * s in
  let d := h / 3 in
  3 * d

theorem sum_perpendiculars_equilateral_triangle_6cm :
  sum_perpendiculars_equilateral_triangle 6 = 3 * sqrt 3 :=
by sorry

end sum_perpendiculars_equilateral_triangle_6cm_l184_184101


namespace train_length_l184_184043

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmph = 60 → time_sec = 12 → 
  length = speed_kmph * (1000 / 3600) * time_sec → 
  length = 200.04 :=
by
  intros h_speed h_time h_length
  sorry

end train_length_l184_184043


namespace odd_function_check_l184_184948

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184948


namespace tunnel_width_is_span_minimal_excavation_is_optimized_l184_184500

noncomputable def tunnel_width_span := 
  let P := (11, 4.5)
  let a := 22 / 2
  let b := 6
  let l := 2 * sqrt ((11 ^ 2 + (4.5 / 6) ^ 2) / ((4.5 ^ 2) / (b ^ 2)))
  l ≈ 33.3

noncomputable def minimal_excavation_span_height := 
  let P := (11, 4.5)
  let a := 11 * sqrt 2
  let b := (9 / 2) * sqrt 2
  let S := (π / 2) * a * b
  let l := 2 * a
  let h := b
  (l ≈ 31.1) ∧ (h ≈ 6.4)

theorem tunnel_width_is_span : tunnel_width_span := sorry

theorem minimal_excavation_is_optimized : minimal_excavation_span_height := sorry

end tunnel_width_is_span_minimal_excavation_is_optimized_l184_184500


namespace exists_n_consecutive_not_prime_power_l184_184706

theorem exists_n_consecutive_not_prime_power (n : ℕ) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ ∃ p : ℕ, p.prime ∧ (m + k = p^x) :=
by
  sorry

end exists_n_consecutive_not_prime_power_l184_184706


namespace truncated_polyhedron_vertex_count_truncated_polyhedron_edge_count_l184_184802

theorem truncated_polyhedron_vertex_count (num_edges : ℕ) (h : num_edges = 100) : 
    let num_vertices := 2 * num_edges in
    num_vertices = 200 :=
by
  sorry

theorem truncated_polyhedron_edge_count (num_edges : ℕ) (h : num_edges = 100) : 
    let num_vertices := 2 * num_edges in
    let num_new_edges := (num_vertices * 3) / 2 in
    num_new_edges = 300 :=
by
  sorry

end truncated_polyhedron_vertex_count_truncated_polyhedron_edge_count_l184_184802


namespace tenth_difference_optimal_number_l184_184854

-- Definitions
def isDifferenceOptimalNumber (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n + 1 ∧ x = m^2 - n^2

def differenceOptimalNumbers : List ℕ :=
  List.filter isDifferenceOptimalNumber (List.range 1000)

-- Proving the 10th difference optimal number
theorem tenth_difference_optimal_number :
  (differenceOptimalNumbers.get? 9) = some 32 :=
by
  sorry

end tenth_difference_optimal_number_l184_184854


namespace trigonometric_identity_l184_184058

-- Given conditions and the main theorem to be proven
theorem trigonometric_identity (α : ℝ) :
  (ctg α := 1 / (Real.tan α)) →
  let ctg_2α := 1 / (Real.tan (2 * α))
  let ctg_4α := 1 / (Real.tan (4 * α))
  let lhs := (ctg_2α ^ 2 - 1) / (2 * ctg_2α) - Real.cos (8 * α) * ctg_4α
  let rhs := Real.sin (8 * α)
  lhs = rhs := 
sorry

end trigonometric_identity_l184_184058


namespace hulk_jump_exceeds_2000_l184_184403

theorem hulk_jump_exceeds_2000 {n : ℕ} (h : n ≥ 1) :
  2^(n - 1) > 2000 → n = 12 :=
by
  sorry

end hulk_jump_exceeds_2000_l184_184403


namespace cos_arcsin_l184_184835

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l184_184835


namespace transformed_function_is_odd_l184_184981

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184981


namespace expand_and_simplify_l184_184192

theorem expand_and_simplify (x y : ℝ) : 
  (x + 6) * (x + 8 + y) = x^2 + 14 * x + x * y + 48 + 6 * y :=
by sorry

end expand_and_simplify_l184_184192


namespace odd_function_check_l184_184939

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184939


namespace third_question_is_worth_39_l184_184046

def x : ℕ := 31  -- Points for the first question

def third_question_points := x + 8  -- Points for the third question

theorem third_question_is_worth_39 :
  let x := 31 in
  let third_question_points := x + 8 in
  third_question_points = 39 :=
by
  rw [third_question_points, x]
  rfl

end third_question_is_worth_39_l184_184046


namespace largest_number_in_sample_is_481_l184_184074

theorem largest_number_in_sample_is_481 :
  ∀ (employees : Fin 500) (sample : List (Fin 500)),
    (sample.nth (5 : Fin 500) = 6) → 
    (sample.nth (30 : Fin 500) = 31) → 
    (∀ n : ℕ, n < 20 → sample.nth (n.succ * 25 - 19) = (6 + 25 * (n : ℕ))) →
    sample.nth (19 * 25 + 6 - 25) = 481 :=
by sorry

end largest_number_in_sample_is_481_l184_184074


namespace parametric_to_ordinary_eq_l184_184887

variable (t : ℝ)

theorem parametric_to_ordinary_eq (h1 : x = Real.sqrt t + 1) (h2 : y = 2 * Real.sqrt t - 1) (h3 : t ≥ 0) :
    y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end parametric_to_ordinary_eq_l184_184887


namespace relationship_of_coefficients_l184_184896

theorem relationship_of_coefficients (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0) 
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end relationship_of_coefficients_l184_184896


namespace num_values_n_l184_184358

theorem num_values_n (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d) :
  ∃ (values : Finset ℝ), Finset.card values = 3 ∧
    ∀ (x y z t : ℝ), (x, y, z, t) ∈ List.permutations [a, b, c, d] → 
      (x - y)^2 + (y - z)^2 + (z - t)^2 + (t - x)^2 ∈ values :=
sorry

end num_values_n_l184_184358


namespace recurring_decimal_as_fraction_l184_184563

theorem recurring_decimal_as_fraction (h : (0.02).recurring = (2 / 99)) : (2.07).recurring = 68 / 33 :=
sorry

end recurring_decimal_as_fraction_l184_184563


namespace james_after_paying_debt_l184_184689

variables (L J A : Real)

-- Define the initial conditions
def total_money : Real := 300
def debt : Real := 25
def total_with_debt : Real := total_money + debt

axiom h1 : J = A + 40
axiom h2 : J + A = total_with_debt

-- Prove that James owns $170 after paying off half of Lucas' debt
theorem james_after_paying_debt (h1 : J = A + 40) (h2 : J + A = total_with_debt) :
  (J - (debt / 2)) = 170 :=
  sorry

end james_after_paying_debt_l184_184689


namespace berta_can_force_win_l184_184103

def losing_position (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2^(m+2) - 2

theorem berta_can_force_win (N : ℕ) (h : N ≥ 100000) : ∃ n : ℕ, (losing_position n) ∧ n ≥ N :=
by {
  use 131070,  -- Providing the correct answer as a candidate
  split,
  -- Showing that 131070 is a losing position
  { existsi 15,
    exact eq.refl _ },
  -- Showing that 131070 ≥ N
  exact le_of_eq (eq.refl N)
}

end berta_can_force_win_l184_184103


namespace prove_distance_to_asymptote_is_sqrt_10_l184_184726

noncomputable def hyperbola_focus_to_asymptote_distance : ℝ :=
  let a2 := 20
  let b2 := 5
  let point_on_H1 := (2 * Real.sqrt 15, Real.sqrt 5)
  let λ := (point_on_H1.1^2 / a2) - (point_on_H1.2^2 / b2)
  let H1_eq := λ = 2
  let a := Real.sqrt (a2 * λ)
  let b := Real.sqrt (b2 * λ)
  let c := Real.sqrt (a^2 + b^2)
  let focus := (c, 0)
  let asymptote_slope := 1 / 2
  let asymptote_eq := fun x y => x - 2 * y = 0
  let distance := focus.1 / Real.sqrt (1 + (2^2) : ℝ) 
  distance

theorem prove_distance_to_asymptote_is_sqrt_10 :
  hyperbola_focus_to_asymptote_distance = Real.sqrt 10 := 
sorry

end prove_distance_to_asymptote_is_sqrt_10_l184_184726


namespace odd_entries_count_eq_n_l184_184341

open Nat

noncomputable def f : ℕ × ℕ → ℕ
| (0, _) => 0
| (_, 0) => 0
| (1, 1) => n
| (i+1, j+1) => floor (f (i, j+1) / 2) + floor (f (i+1, j) / 2)

def g (k : ℕ) : ℕ :=
(k+1).sum fun i => f (k-i, i)

theorem odd_entries_count_eq_n (n : ℕ)
  (h1 : ∀ i, f (0, i) = 0)
  (h2 : ∀ i, f (i, 0) = 0)
  (h3 : f (1, 1) = n)
  (h4 : ∀ i j, 1 < i * j → f (i, j) = floor (f (i-1, j) / 2) + floor (f (i, j-1) / 2))
  : ∑ k in range (k + 1), (g k) = n :=
sorry

end odd_entries_count_eq_n_l184_184341


namespace triangle_right_angled_acute_22_5_degrees_l184_184899

theorem triangle_right_angled_acute_22_5_degrees
  {ABC : Type}
  (C H_3 L_3 M_3 : ABC)
  (angle_ACH2 angle_H3CL3 angle_L3CM3 angle_M3CB : ℝ)
  (h1 : is_altitude C H_3 ABC)
  (h2 : is_angle_bisector C L_3 ABC)
  (h3 : is_median C M_3 ABC)
  (h4 : angle_ACH2 = angle_H3CL3)
  (h5 : angle_H3CL3 = angle_L3CM3)
  (h6 : angle_L3CM3 = angle_M3CB) :
  is_right_triangle ABC ∧ angle_ABC ABC = 22.5 := 
  sorry

end triangle_right_angled_acute_22_5_degrees_l184_184899


namespace transformed_function_is_odd_l184_184980

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184980


namespace max_scientists_l184_184783

open Real

theorem max_scientists (x : ℝ) (hx : x > 4) :
  ∃ n : ℕ, n ≤ 2 * ⌊ x / (2 * x - 8) ⌋ :=
by sorry

end max_scientists_l184_184783


namespace truthful_dwarfs_count_l184_184151

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184151


namespace number_of_mappings_l184_184277

-- Definitions
def A : Finset ℝ := {a | ∃ i : Fin 100, a = (i : ℝ)}
def B : Finset ℝ := {b | ∃ j : Fin 50, b = (j : ℝ)}

-- Hypothesis: f is a non-decreasing function with every element of B having a preimage in A
def f (A B : Finset ℝ) := {f | ∀ x ∈ A, ∃ y ∈ B, f x = y ∧ ∀ (x1 x2 : ℝ), x1 ≤ x2 → (f x1) ≤ (f x2)}

-- Statement to prove
theorem number_of_mappings (A B : Finset ℝ) (hA : A.card = 100) (hB : B.card = 50) :
  ∃ n, n = Finset.card {f | ∀ (x ∈ A), ∃ (y ∈ B), f x = y ∧ ∀ (x1 x2 : ℝ), x1 ≤ x2 → (f x1) ≤ (f x2)} ∧
  n = Nat.choose 99 49 :=
sorry

end number_of_mappings_l184_184277


namespace dwarfs_truthful_count_l184_184156

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184156


namespace trigonometric_polynomial_has_n_roots_l184_184686

-- Introducing the variables and conditions
variables {n : ℕ}
variables (a : ℕ → ℝ)
variables (h : ∀ k, 0 < a k)
variables (h_order : ∀ k l, k < l → a k < a l)

-- Definition for the trigonometric polynomial
noncomputable def trigonometric_polynomial (ϕ : ℝ) : ℝ :=
  ∑ k in range (n + 1), a k * Real.cos (k * ϕ)

-- Statement of the theorem
theorem trigonometric_polynomial_has_n_roots :
  ∃ (roots : Finset ℝ), roots.card = n ∧ ∀ ϕ ∈ roots, 0 ≤ ϕ ∧ ϕ ≤ π ∧ trigonometric_polynomial a ϕ = 0 :=
sorry

end trigonometric_polynomial_has_n_roots_l184_184686


namespace floor_abs_neg_45_7_l184_184185

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184185


namespace cuboid_diagonal_length_l184_184075

theorem cuboid_diagonal_length (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2) 
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) : 
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 :=
sorry

end cuboid_diagonal_length_l184_184075


namespace problem_perimeter_remaining_quadrilateral_l184_184085

theorem problem_perimeter_remaining_quadrilateral
  {P Q R T : Point}
  (h1 : right_triangle P Q R)
  (h2 : hypotenuse P R = 5)
  (h3 : side_length P Q = 3)
  (h4 : side_length Q R = 4)
  (h5 : TQ_along_PQ : TQ_along PQ T Q)
  (h6 : TR_along_QR : TR_along QR T R)
  (h7 : side_length T Q = 2) :
  perimeter_quad P T R Q = 8 + sqrt 21 :=
by sorry

end problem_perimeter_remaining_quadrilateral_l184_184085


namespace payment_ways_l184_184517

theorem payment_ways (x y : ℕ) (h1 : 20 * x + 50 * y = 270) (hx_pos : 0 < x) (hy_pos : 0 < y):
  -- Define the set of all (x, y) pairs that satisfy the equation
  (20 * x + 50 * y = 270) → 
  -- Count the number of such pairs
  ({(x, y) : ℕ × ℕ | 20 * x + 50 * y = 270 ∧ 0 < x ∧ 0 < y}).card = 3 :=
sorry

end payment_ways_l184_184517


namespace cos_arcsin_l184_184833

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l184_184833


namespace B_months_grazing_eq_five_l184_184092

-- Define the conditions in the problem
def A_oxen : ℕ := 10
def A_months : ℕ := 7
def B_oxen : ℕ := 12
def C_oxen : ℕ := 15
def C_months : ℕ := 3
def total_rent : ℝ := 175
def C_rent_share : ℝ := 45

-- Total ox-units function
def total_ox_units (x : ℕ) : ℕ :=
  A_oxen * A_months + B_oxen * x + C_oxen * C_months

-- Prove that the number of months B's oxen grazed is 5
theorem B_months_grazing_eq_five (x : ℕ) :
  total_ox_units x = 70 + 12 * x + 45 →
  (C_rent_share / total_rent = 45 / total_ox_units x) →
  x = 5 :=
by
  intros h1 h2
  sorry

end B_months_grazing_eq_five_l184_184092


namespace num_four_digit_numbers_divisible_by_5_l184_184819

theorem num_four_digit_numbers_divisible_by_5 : 
  ∃ n : ℕ, n = 108 ∧ 
  ∀ (x : ℕ), 
    (x < 10000 ∧ x ≥ 1000) →               -- x is a four-digit number
    (∀ i j, i ≠ j → 
      (x.digit i 10 ≠ x.digit j 10 ∧       -- Digits are not repeated
      ∃ d, d ∈ {0, 1, 2, 3, 4, 5}) ∧       -- Digits are from the set {0, 1, 2, 3, 4, 5}
    (x % 5 = 0) →                          -- x is divisible by 5
    ∃! y, y < 10000 ∧ y ≥ 1000 ∧           -- y is also a four-digit number
    y % 5 = 0 ∧
    ∀ i j, i ≠ j → 
    (y.digit i 10 ≠ y.digit j 10 ∧         -- Digits are not repeated
    ∃ d, d ∈ {0, 1, 2, 3, 4, 5}) ∧
    x = y)                                 -- There is exactly one such y for each x
   := 
begin
  sorry
end

end num_four_digit_numbers_divisible_by_5_l184_184819


namespace present_population_l184_184732

theorem present_population (P : ℕ) (h1 : P * 11 / 10 = 264) : P = 240 :=
by sorry

end present_population_l184_184732


namespace point_P1_satisfies_conditions_point_P2_satisfies_conditions_l184_184276

-- Define points A, B, and line l
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)
def line_l (P : ℝ × ℝ) : Prop := 4 * P.1 + 3 * P.2 - 2 = 0

-- Define distance function from a point P to a line
def dist_to_line (P : ℝ × ℝ) : ℝ :=
  abs (4 * P.1 + 3 * P.2 - 2) / (real.sqrt (4^2 + 3^2))

-- Define points P1 and P2 to be checked
def P1 : ℝ × ℝ := (27 / 7, -8 / 7)
def P2 : ℝ × ℝ := (1, -4)

-- Define the condition |PA| = |PB|
def eq_distance (P : ℝ × ℝ) : Prop :=
  (real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)) = (real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2))

-- Prove the points P1 and P2 satisfy the given conditions
theorem point_P1_satisfies_conditions : eq_distance P1 ∧ dist_to_line P1 = 2 := by
  sorry

theorem point_P2_satisfies_conditions : eq_distance P2 ∧ dist_to_line P2 = 2 := by
  sorry

end point_P1_satisfies_conditions_point_P2_satisfies_conditions_l184_184276


namespace round_trip_ratio_l184_184297

noncomputable theory

-- Definitions
def ship_downstream_speed (x : ℝ) : ℝ := 5 * x
def ship_upstream_speed (x : ℝ) : ℝ := 2 * x

-- Providing assumptions
variables (x s : ℝ) (h_ne_zero : x ≠ 0) (hs_ne_zero : s ≠ 0)

-- Average speed for a round trip
def average_round_trip_speed : ℝ :=
  let downstream := ship_downstream_speed x
  let upstream := ship_upstream_speed x
  (2 * s) / ((s / downstream) + (s / upstream))

-- Theorem statement
theorem round_trip_ratio :
  (average_round_trip_speed x s h_ne_zero hs_ne_zero) / (ship_downstream_speed x) = 4 / 7 :=
sorry

end round_trip_ratio_l184_184297


namespace max_surface_area_of_cylinder_surface_area_volume_of_circumscribed_sphere_l184_184307

noncomputable def max_surface_area_cylinder_in_cone (radius_cone slant_height_cone h_cylinder : ℝ) : ℝ :=
  let r_cylinder := 1 in
  let h' := h_cylinder in
  2 * π * (r_cylinder + h')

theorem max_surface_area_of_cylinder
  (radius_cone : ℝ) (slant_height_cone : ℝ) (h_cylinder : ℝ)
  (h_cone := Real.sqrt (slant_height_cone^2 - radius_cone^2))
  (cond1 : radius_cone = 2) (cond2 : slant_height_cone = 4) (cond3 : h_cylinder = Real.sqrt 3)
  : max_surface_area_cylinder_in_cone radius_cone slant_height_cone h_cylinder = 2 * (1 + Real.sqrt 3) * π :=
sorry

noncomputable def circumscribed_sphere_surface_volume (r_sphere : ℝ) : (ℝ × ℝ) :=
  (4 * π * r_sphere^2, (4 / 3) * π * r_sphere^3)

theorem surface_area_volume_of_circumscribed_sphere
  (r_cylinder : ℝ) (h_cylinder : ℝ)
  (radius_sphere := Real.sqrt (r_cylinder^2 + (h_cylinder / 2)^2))
  (cond1 : r_cylinder = 1) (cond2: h_cylinder = Real.sqrt 3)
  : circumscribed_sphere_surface_volume (radius_sphere) = (7 * π, (7 * Real.sqrt 7 * π) / 6) :=
sorry

end max_surface_area_of_cylinder_surface_area_volume_of_circumscribed_sphere_l184_184307


namespace transformed_function_is_odd_l184_184973

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184973


namespace triangle_PZQ_area_is_50_l184_184314

noncomputable def area_triangle_PZQ (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) : ℝ :=
  let RS := PQ -- since PQRS is a rectangle, RS = PQ
  let XY := RS - RX - SY
  let height := 2 * QR -- height is doubled due to triangle similarity ratio
  let area := 0.5 * PQ * height
  area

theorem triangle_PZQ_area_is_50 (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) :
  area_triangle_PZQ PQ QR RX SY hPQ hQR hRX hSY = 50 :=
  sorry

end triangle_PZQ_area_is_50_l184_184314


namespace largest_sum_is_3973_l184_184781

def is_prime : ℕ → Prop := sorry

def congruent_mod (a b n : ℕ) : Prop := (a % n) = (b % n)

def largest_prime_sum (nums : List ℕ) (k : ℕ) : ℕ :=
  -- This function will form k 3-digit numbers from the given nums list,
  -- calculate the sum and check if it's prime and congruent to 1 mod 4. 
  -- The function will return the largest such sum.
  sorry

theorem largest_sum_is_3973 :
  let nums := [1, 2, 3, 6, 7, 7, 8, 9, 9, 9]
  in largest_prime_sum nums 4 = 3973 :=
by
  sorry

end largest_sum_is_3973_l184_184781


namespace largest_inscribed_sphere_volume_l184_184861

theorem largest_inscribed_sphere_volume (b : ℝ) (hb : b > 0) : 
  let V := (π * b^3) / (6 * real.sqrt 3) in V =
  let r := b / (2 * real.sqrt 3) in
  (4 / 3) * π * r^3 :=
by
  sorry

end largest_inscribed_sphere_volume_l184_184861


namespace area_enclosed_by_parabola_l184_184716

noncomputable def area_under_parabola : ℝ :=
  ∫ x in 0..(1/2), (2 * x^2 - x)

theorem area_enclosed_by_parabola :
  abs (area_under_parabola) = (1 / 24) :=
by
  sorry

end area_enclosed_by_parabola_l184_184716


namespace odd_function_g_l184_184988

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184988


namespace cos_arcsin_eq_l184_184843

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l184_184843


namespace no_possible_k_for_prime_roots_l184_184826

theorem no_possible_k_for_prime_roots :
  ∀ (p q : ℕ), prime p → prime q → (p + q = 59) → ∃ (k : ℕ), k = p * q → false :=
  by sorry

end no_possible_k_for_prime_roots_l184_184826


namespace fraction_of_money_on_cd_l184_184110

variables (m c: ℝ)

-- Conditions
def one_third_money := m / 3
def half_cds_cost := c / 2

-- Establish the relationship
def relationship : Prop := one_third_money = half_cds_cost

-- Prove the fraction of total money spent on each individual CD
theorem fraction_of_money_on_cd (n : ℝ) (h : relationship) : (c / n) = (m / 3) :=
by sorry

end fraction_of_money_on_cd_l184_184110


namespace count_possible_x_values_l184_184096

theorem count_possible_x_values (x y : ℕ) (H : (x + 2) * (y + 2) - x * y = x * y) :
  (∃! x, ∃ y, (x - 2) * (y - 2) = 8) :=
by {
  sorry
}

end count_possible_x_values_l184_184096


namespace new_average_after_exclusion_l184_184405

theorem new_average_after_exclusion (S : ℕ) (h1 : S = 27 * 5) (excluded : ℕ) (h2 : excluded = 35) : (S - excluded) / 4 = 25 :=
by
  sorry

end new_average_after_exclusion_l184_184405


namespace original_volume_proof_l184_184084

def original_height_increase (h : ℚ) : ℚ := h + 2
def original_surface_area (h : ℚ) (a : ℚ) : ℚ := 2 * a^2 + 4 * a * h
def cube_surface_area (h : ℚ) : ℚ := 6 * (h + 2)^2
def surface_area_increase_condition (h : ℚ) : Prop :=
  cube_surface_area(h) - original_surface_area(h, h + 2) = 56
def original_volume (h : ℚ) (a : ℚ) : ℚ := a^2 * h

theorem original_volume_proof (h : ℚ) (H : surface_area_increase_condition h) : 
  original_volume h (original_height_increase h) = 400 / 27 :=
sorry

end original_volume_proof_l184_184084


namespace find_omega_l184_184924

noncomputable def sin_function_properties {ω : ℝ} {φ : ℝ} (h1 : ω > 0) (h2 : 0 ≤ φ ∧ φ ≤ π) : Prop :=
f(x) = sin(ω * x + φ) ∧ 
(∀ x : ℝ, sin(ω * x + φ) = sin(-ω * x + φ)) ∧ 
(sin(ω * (3 * π / 4) + φ) = 0) ∧ 
(monotonic_on (λ x, sin(ω * x + φ)) (set.Icc 0 (π / 2)))

theorem find_omega {ω : ℝ} (h1 : ω > 0) (h2 : 0 ≤ φ ∧ φ ≤ π)
  (h3 : sin_function_properties h1 h2) : ω = 2 / 3 ∨ ω = 2 :=
sorry

end find_omega_l184_184924


namespace probability_tiles_l184_184748

theorem probability_tiles :
  let A := finset.range 25
  let B := finset.range' 10 30
  let P_A := (finset.range 18).card / A.card
  let evens := (finset.filter (λ n, n % 2 = 0) B).card
  let greater_than_30 := (finset.filter (λ n, n > 30) B).card
  let evens_g_or_thirty := evens + (finset.filter (λ n, n > 30 ∧ n % 2 ≠ 0) B).card
  let P_B := evens_g_or_thirty / B.card
  let result := P_A * P_B
  result = 323 / 750 := 
by sorry

end probability_tiles_l184_184748


namespace matrix_identity_l184_184346

noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  N * N = 4 • N + -11 • I :=
by
  sorry

end matrix_identity_l184_184346


namespace tilly_counts_total_stars_l184_184013

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l184_184013


namespace dwarfs_truthful_count_l184_184164

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184164


namespace rate_per_kg_mangoes_l184_184820

theorem rate_per_kg_mangoes 
  (weight_grapes : ℕ) 
  (rate_grapes : ℕ) 
  (weight_mangoes : ℕ) 
  (total_paid : ℕ)
  (total_grapes_cost : ℕ)
  (total_mangoes_cost : ℕ)
  (rate_mangoes : ℕ) 
  (h1 : weight_grapes = 14) 
  (h2 : rate_grapes = 54)
  (h3 : weight_mangoes = 10) 
  (h4 : total_paid = 1376) 
  (h5 : total_grapes_cost = weight_grapes * rate_grapes)
  (h6 : total_mangoes_cost = total_paid - total_grapes_cost) 
  (h7 : rate_mangoes = total_mangoes_cost / weight_mangoes):
  rate_mangoes = 62 :=
by
  sorry

end rate_per_kg_mangoes_l184_184820


namespace x_intercept_perpendicular_line_l184_184122

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := 2 * x + 4 * y = 8

-- Define the slope-intercept form of a line
def slope_intercept_form (m b x y : ℝ) : Prop := y = m * x + b

-- Define the perpendicular condition for slopes
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Assume the y-intercept of the perpendicular line
def y_intercept := 5

-- Define the x-intercept calculation
def x_intercept (b m : ℝ) : ℝ := -b / m

-- The theorem statement
theorem x_intercept_perpendicular_line : x_intercept y_intercept 2 = -5 / 2 :=
by
  sorry

end x_intercept_perpendicular_line_l184_184122


namespace area_closed_figure_sin_l184_184292

theorem area_closed_figure_sin (a : ℝ) (h : (x ^ 9).coeff_of_expansion ((x^2 - 1 / (a*x)) ^ 9) = -21 / 2) :
  2 * ∫ (x : ℝ) in (0 : ℝ)..a, sin x = 2 - 2 * cos 2 :=
by
  sorry

end area_closed_figure_sin_l184_184292


namespace triangle_probability_l184_184237

open Finset

def lengths : Finset ℕ := {3, 4, 6, 8, 10, 12, 15}

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c

def valid_triangles (s : Finset ℕ) : Finset (ℕ × ℕ × ℕ) :=
  s.subsets 3 |>.filter (λ t, match t.toList with
                       | [a, b, c] => can_form_triangle a b c
                       | _ => false
                     )

/-- Calculate the probability of randomly selecting three sticks that can form a triangle. -/
noncomputable def probability_of_triangle : ℚ :=
  (valid_triangles lengths).card / (lengths.subsets 3).card

-- 35 is the total number of ways to choose 3 sticks out of 7
theorem triangle_probability : probability_of_triangle = 3 / 7 :=
  sorry

end triangle_probability_l184_184237


namespace draw_all_red_balls_by_4th_l184_184005

theorem draw_all_red_balls_by_4th :
  let total_balls := 10,
      white_balls := 8,
      red_balls := 2 in 
  let P_event_A := (2 / total_balls) * ((9 / 10) ^ 2) * (1 / 10),
      P_event_B := (8 / total_balls) * (2 / total_balls) * (9 / 10) * (1 / 10),
      P_event_C := ((8 / total_balls) ^ 2) * (2 / total_balls) * (1 / 10) in
  (P_event_A + P_event_B + P_event_C = 353 / 5000) :=
sorry

end draw_all_red_balls_by_4th_l184_184005


namespace b_is_dk_squared_l184_184668

theorem b_is_dk_squared (a b : ℤ) (h : ∃ r1 r2 r3 : ℤ, (r1 * r2 * r3 = b) ∧ (r1 + r2 + r3 = a) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 0))
  : ∃ d k : ℤ, (b = d * k^2) ∧ (d ∣ a) := 
sorry

end b_is_dk_squared_l184_184668


namespace train_length_1080_l184_184809

def length_of_train (speed time : ℕ) : ℕ := speed * time

theorem train_length_1080 (speed time : ℕ) (h1 : speed = 108) (h2 : time = 10) : length_of_train speed time = 1080 := by
  sorry

end train_length_1080_l184_184809


namespace problem_statement_l184_184687

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then real.log x / real.log 9 else 4^(-x) + 3/2

theorem problem_statement : f 27 + f (-real.log 3 / real.log 4) = 6 :=
by
  sorry

end problem_statement_l184_184687


namespace transformed_function_is_odd_l184_184960

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184960


namespace work_combined_days_l184_184478

theorem work_combined_days (A B : Type) (work : ℕ) (days_B : ℕ) (days_A : ℕ) 
  (h1 : 2 * days_B = days_A) (h2 : days_B = 18) : (1 / days_A + 1 / days_B) ≠ 0 → 1 / (1 / days_A + 1 / days_B) = 6 :=
by
  -- Assign the number of days B takes to complete the work
  let days_B := 18
  -- Since A works twice as fast as B
  let days_A := days_B / 2
  -- Compute the combined rate of work
  let combined_rate := 1 / days_A + 1 / days_B
  -- Ensure the combined rate is non-zero
  have : combined_rate ≠ 0 := by
    sorry -- (Proof that combined_rate is non-zero)
  -- Calculate the total days required for A and B together to complete the work
  calc
    1 / combined_rate = 1 / (1 / days_A + 1 / days_B) := by sorry
                   ... = 6 := by sorry

end work_combined_days_l184_184478


namespace dwarfs_truthful_count_l184_184161

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184161


namespace find_pairs_l184_184878

theorem find_pairs (c d : ℕ) (c_gt_1 : c > 1) (d_gt_1 : d > 1) :
  (∀ (Q : Polynomial ℤ) (monicQ : Monic Q) (degQ : degree Q = d) (p : ℕ) (prime_p : Prime p) (p_condition : p > c * (2 * c + 1)),
    ∃ S : Finset ℤ, S.card ≤ (2 * c - 1) * p / (2 * c + 1) ∧
    (∀ t : ℤ, ∃ s ∈ S, ∃ k : ℕ, t ≡ (Q^[k]) s [ZMOD p])) ↔ 
  (2 ≤ d ∧ d ≤ c) := sorry

end find_pairs_l184_184878


namespace insurance_plan_percentage_l184_184336

theorem insurance_plan_percentage
(MSRP : ℝ) (I : ℝ) (total_cost : ℝ) (state_tax_rate : ℝ)
(hMSRP : MSRP = 30)
(htotal_cost : total_cost = 54)
(hstate_tax_rate : state_tax_rate = 0.5)
(h_total_cost_eq : MSRP + I + state_tax_rate * (MSRP + I) = total_cost) :
(I / MSRP) * 100 = 20 :=
by
  -- You can leave the proof as sorry, as it's not needed for the problem
  sorry

end insurance_plan_percentage_l184_184336


namespace common_tangents_of_circles_l184_184720

theorem common_tangents_of_circles :
  let C1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 2)^2 = 9 }
  let C2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 4 }
  num_common_tangents C1 C2 = 3 :=
by
  sorry

def num_common_tangents : Set (ℝ × ℝ) → Set (ℝ × ℝ) → ℕ :=
  sorry

end common_tangents_of_circles_l184_184720


namespace journey_speed_second_half_l184_184090

theorem journey_speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (v : ℝ) : 
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 224 →
  v = 24 :=
by
  intro h
  sorry

end journey_speed_second_half_l184_184090


namespace find_percentage_l184_184069

theorem find_percentage (P : ℝ) : 100 * (P / 100) + 20 = 100 → P = 80 :=
by
  sorry

end find_percentage_l184_184069


namespace sum_sin_identities_l184_184384

theorem sum_sin_identities (α : ℝ) (n : ℕ) :
  (∑ i in Finset.range n, Real.sin ((i + 1) * α)) = (Real.sin ((n + 1) * α / 2) / Real.sin (α / 2)) * Real.sin (n * α / 2) :=
sorry

end sum_sin_identities_l184_184384


namespace max_mammoths_l184_184849

theorem max_mammoths : ∀ (m : ℕ),
  (∀ (total_arrows_per_mammoth : ℕ) (total_diagonals : ℕ) (max_arrows_per_diagonal : ℕ), 
    total_arrows_per_mammoth = 3 → total_diagonals = 30 → max_arrows_per_diagonal = 2 →
    (total_arrows_per_mammoth * m) <= (total_diagonals * max_arrows_per_diagonal)) →
  m ≤ 20 :=
by 
  intros m h
  specialize h 3 30 2 rfl rfl rfl
  have h1 : m * 3 ≤ 60 := h
  have h2 : m ≤ 20, from (nat.div_le_iff_le_mul 3).mp h1
  exact h2

end max_mammoths_l184_184849


namespace combination_sum_three_digit_odd_count_l184_184061

-- Proof Problem 1 in Lean 4 statement
theorem combination_sum (n : ℕ) (h1 : 19 / 2 ≤ n) (h2 : n ≤ 21 / 2) : 
  (Combin.ofNat (3 * n) (38 - n) + Combin.ofNat (21 + n) (3 * n) = 466) := 
sorry

-- Proof Problem 2 in Lean 4 statement
theorem three_digit_odd_count : 
  (∃ (a b c : ℕ), {a, b, c} ⊆ {0, 2, 1, 3, 5} ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ((a = 0 ∨ a = 2) ∧ b ∈ {1, 3, 5} ∧ c ∈ {1, 3, 5}) ∧ 
  (odd (c + 10 * b + 100 * a) ∨ odd (c + 10 * a + 100 * b)) := 18 := 
sorry

end combination_sum_three_digit_odd_count_l184_184061


namespace transformed_function_is_odd_l184_184975

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184975


namespace quadratic_eq_other_solution_l184_184290

theorem quadratic_eq_other_solution (c : ℝ) (h : c = -540) :
  ∃ (y : ℝ), y ≠ 18 ∧ (∀ x : ℝ, x^2 + 12 * x + c = 0 → (x = 18 ∨ x = y)) :=
by
  use -30
  have f : (18 : ℝ) ≠ -30 := by norm_num
  exact ⟨f, λ x h' => by simp [*, add_eq_zero_iff_eq_neg]⟩

end quadratic_eq_other_solution_l184_184290


namespace tan_subtraction_simplify_l184_184394

theorem tan_subtraction_simplify :
  tan (Real.pi / 12) - tan (5 * Real.pi / 12) = -4 * Real.sqrt 3 := by
sorry

end tan_subtraction_simplify_l184_184394


namespace final_composition_of_mixture_l184_184897

noncomputable def milk_after_n_operations (initial_milk : ℝ) (operations: ℕ → ℝ) (n : ℕ) : ℝ :=
match n with
| 0     => initial_milk
| (n+1) => (1 - operations n) * milk_after_n_operations initial_milk operations n

theorem final_composition_of_mixture :
  let initial_volume := 100
  let operations := [0.20, 0.30, 0.40].cycle.take 9
  let final_milk := milk_after_n_operations initial_volume (λ n, (operations : list ℝ) !! n) 9
  final_milk = 3.7933056 ∧ initial_volume - final_milk = 96.2066944 :=
by
  sorry

end final_composition_of_mixture_l184_184897


namespace margie_drive_distance_l184_184369

-- Conditions
def car_mpg : ℝ := 45  -- miles per gallon
def gas_price : ℝ := 5 -- dollars per gallon
def money_spent : ℝ := 25 -- dollars

-- Question: Prove that Margie can drive 225 miles with $25 worth of gas.
theorem margie_drive_distance (h1 : car_mpg = 45) (h2 : gas_price = 5) (h3 : money_spent = 25) :
  money_spent / gas_price * car_mpg = 225 := by
  sorry

end margie_drive_distance_l184_184369


namespace isosceles_right_triangle_properties_l184_184295

-- Define the area of an isosceles right triangle with given hypotenuse
def area_of_isosceles_right_triangle (c : ℚ) : ℚ := (c^2) / 4

-- Define the perimeter of an isosceles right triangle with given hypotenuse
def perimeter_of_isosceles_right_triangle (c : ℚ) : ℝ := (2 + Real.sqrt 2) * (c : ℝ)

-- Main statement
theorem isosceles_right_triangle_properties (c : ℚ) :
  ∃ S l, S = area_of_isosceles_right_triangle c ∧ l = perimeter_of_isosceles_right_triangle c ∧
  (∃ r : ℚ, S = r) ∧ ¬∃ r : ℚ, l = r :=
by
  sorry

end isosceles_right_triangle_properties_l184_184295


namespace lcm_inequality_l184_184893

open Nat

-- Assume positive integers n and m, with n > m
theorem lcm_inequality (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n > m) :
  Nat.lcm m n + Nat.lcm (m+1) (n+1) ≥ 2 * m * Real.sqrt n := 
  sorry

end lcm_inequality_l184_184893


namespace max_squared_distance_sum_l184_184784

theorem max_squared_distance_sum (a : ℝ) (n : ℕ) (h_pos : a > 0) (h_n : n > 1)
  (x : Fin n → ℝ) (h_sum_sq : ∑ i, (x i)^2 = a) :
  (∑ 1 ≤ i < j ≤ n, (x i - x j)^2) ≤ n * a :=
sorry

end max_squared_distance_sum_l184_184784


namespace eval_floor_abs_neg_45_7_l184_184178

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184178


namespace systematic_sampling_interval_l184_184015

theorem systematic_sampling_interval :
  ∀ (total_students sample_size : ℕ), total_students = 1000 → sample_size = 50 → (total_students / sample_size) = 20 :=
by
  intros total_students sample_size h1 h2
  rw [h1, h2]
  norm_num
  sorry

end systematic_sampling_interval_l184_184015


namespace eccentricity_sqrt2_l184_184905

def hyperbola (a b : ℝ) := ∀ ⦃x y : ℝ⦄, x^2 / a^2 - y^2 / b^2 = 1

def foci_coordinates (a b c : ℝ) := 
  let c := real.sqrt (a^2 + b^2) in
  ((-c, 0), (c, 0))

def point_A_B (a b c : ℝ) := 
  let y_A := b * real.sqrt (1 + (c / a)^2) in
  ((-c, y_A), (-c, -y_A))

def points_PQF2_perimeter_is_12 (a b c : ℝ) :=
  let F_2 := (c, 0) in
  let y_A := b * real.sqrt (1 + (c / a)^2) in
  2 * real.sqrt (c^2 + y_A^2) + 2 * y_A = 12

def maximum_value_ab (a b : ℝ) := ∀ ⦃a' b' : ℝ⦄, 
  points_PQF2_perimeter_is_12 a' b' (real.sqrt (a'^2 + b'^2)) → 
  a * b ≥ a' * b'

theorem eccentricity_sqrt2 (a b : ℝ) (h₁: a > 0) (h₂: b > 0) 
  (h₃: hyperbola a b)
  (h₄: points_PQF2_perimeter_is_12 a b (real.sqrt (a^2 + b^2)))
  (h₅: maximum_value_ab a b)
  :
  let c := real.sqrt (a^2 + b^2) in
  let e := c / a in
  e = real.sqrt 2 :=
sorry

end eccentricity_sqrt2_l184_184905


namespace problem1_problem2_l184_184582

-- Define the universe U
def U : Set ℝ := Set.univ

-- Define the sets A and B
def A : Set ℝ := { x | -4 < x ∧ x < 4 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Statement of the first proof problem: Prove A ∩ B is equal to the given set
theorem problem1 : A ∩ B = { x | -4 < x ∧ x ≤ 1 ∨ 4 > x ∧ x ≥ 3 } :=
by
  sorry

-- Statement of the second proof problem: Prove the complement of (A ∪ B) in the universe U is ∅
theorem problem2 : Set.compl (A ∪ B) = ∅ :=
by
  sorry

end problem1_problem2_l184_184582


namespace max_obtuse_angles_with_15_rays_l184_184021

theorem max_obtuse_angles_with_15_rays : 
  let n := 15 in 
  let total_pairs := (n * (n - 1)) / 2 in 
  ∀ (obtuse_angle_pairs : ℕ), obtuse_angle_pairs ≤ total_pairs ∧ (obtuse_angle_pairs * 2 <= total_pairs) → 
  max(obtuse_angle_pairs) = 75 :=
by sorry

end max_obtuse_angles_with_15_rays_l184_184021


namespace height_of_spherical_cap_case1_height_of_spherical_cap_case2_l184_184521

variable (R : ℝ) (c : ℝ)
variable (h_c_gt_1 : c > 1)

-- Case 1: Not including the circular cap in the surface area
theorem height_of_spherical_cap_case1 : ∃ m : ℝ, m = (2 * R * (c - 1)) / c :=
by
  sorry

-- Case 2: Including the circular cap in the surface area
theorem height_of_spherical_cap_case2 : ∃ m : ℝ, m = (2 * R * (c - 2)) / (c - 1) :=
by
  sorry

end height_of_spherical_cap_case1_height_of_spherical_cap_case2_l184_184521


namespace find_range_of_k_l184_184256

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l184_184256


namespace diana_total_bike_time_l184_184862

theorem diana_total_bike_time :
  ∀ (total_distance: ℕ) (speed_fast: ℕ) (time_fast: ℕ) (speed_slow: ℕ),
  total_distance = 10 →
  speed_fast = 3 →
  time_fast = 2 →
  speed_slow = 1 →
  (time_fast + (total_distance - speed_fast * time_fast) / speed_slow) = 6 :=
by
  intros total_distance speed_fast time_fast speed_slow h1 h2 h3 h4
  calc
    time_fast + (total_distance - speed_fast * time_fast) / speed_slow = time_fast + (10 - 3 * 2) / 1 : by rw [h1, h2, h3, h4]
    ... = 2 + (10 - 6) / 1 : by rfl
    ... = 2 + 4 / 1 : by rfl
    ... = 2 + 4 : by rfl
    ... = 6 : by rfl

end diana_total_bike_time_l184_184862


namespace alcohol_by_volume_l184_184711

/-- Solution x is 10% alcohol by volume and is 50 ml.
    Solution y is 30% alcohol by volume and is 150 ml.
    We must prove the final solution is 25% alcohol by volume. -/
theorem alcohol_by_volume (vol_x vol_y : ℕ) (conc_x conc_y : ℕ) (vol_mix : ℕ) (conc_mix : ℕ) :
  vol_x = 50 →
  conc_x = 10 →
  vol_y = 150 →
  conc_y = 30 →
  vol_mix = vol_x + vol_y →
  conc_mix = 100 * (vol_x * conc_x + vol_y * conc_y) / vol_mix →
  conc_mix = 25 :=
by
  intros h1 h2 h3 h4 h5 h_cons
  sorry

end alcohol_by_volume_l184_184711


namespace arithmetic_series_sum_l184_184115

theorem arithmetic_series_sum :
  let a1 := 15
  let an := 35
  let d := 0.2
  arithmetic_series_sum (a1 an d) = 2525 := by
sorry

end arithmetic_series_sum_l184_184115


namespace number_of_truthful_dwarfs_is_correct_l184_184135

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184135


namespace floor_abs_neg_45_7_l184_184183

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184183


namespace vector_decomposition_l184_184054

theorem vector_decomposition :
  let x := (13, 2, 7 : ℝ × ℝ × ℝ)
  let p := (5, 1, 0 : ℝ × ℝ × ℝ)
  let q := (2, -1, 3 : ℝ × ℝ × ℝ)
  let r := (1, 0, -1 : ℝ × ℝ × ℝ)
  x = (3 : ℝ) • p + (1 : ℝ) • q + (-4 : ℝ) • r :=
by
  sorry

end vector_decomposition_l184_184054


namespace find_t_for_no_remainder_l184_184195

def P (x : ℝ) : ℝ := 6 * x^2 - 7 * x + 8
def Q (x : ℝ) (t : ℝ) : ℝ := 5 * x^2 + t * x + 12
def Quotient (x : ℝ) : ℝ := 4 * x^2 - 9 * x + 12

theorem find_t_for_no_remainder (t : ℝ) : 
  (P x) = (Q x t) * (Quotient x) → 
  t = -7 / 12 :=
sorry

end find_t_for_no_remainder_l184_184195


namespace evaluate_expression_l184_184870

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l184_184870


namespace eval_floor_abs_neg_45_7_l184_184176

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184176


namespace mean_of_xyz_l184_184717

theorem mean_of_xyz (x y z : ℚ) (eleven_mean : ℚ)
  (eleven_sum : eleven_mean = 32)
  (fourteen_sum : 14 * 45 = 630)
  (new_mean : 14 * 45 = 630) :
  (x + y + z) / 3 = 278 / 3 :=
by
  sorry

end mean_of_xyz_l184_184717


namespace lesser_solution_of_quadratic_eq_l184_184461

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l184_184461


namespace james_louise_age_sum_l184_184660

variables (J L : ℝ)

theorem james_louise_age_sum
  (h₁ : J = L + 9)
  (h₂ : J + 5 = 3 * (L - 3)) :
  J + L = 32 :=
by
  /- Proof goes here -/
  sorry

end james_louise_age_sum_l184_184660


namespace number_of_truthful_dwarfs_l184_184140

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184140


namespace length_of_other_train_l184_184790

theorem length_of_other_train
  (length_train1 : ℝ)
  (speed_train1_kmph : ℝ)
  (speed_train2_kmph : ℝ)
  (time_crossing_s : ℝ)
  (incline_deg : ℝ) :
  length_train1 = 300 →
  speed_train1_kmph = 120 →
  speed_train2_kmph = 100 →
  time_crossing_s = 9 →
  incline_deg = 3 →
  let speed_train1_ms := speed_train1_kmph * 1000 / 3600 in
  let speed_train2_ms := speed_train2_kmph * 1000 / 3600 in
  let relative_speed_ms := speed_train1_ms + speed_train2_ms in
  let total_distance := relative_speed_ms * time_crossing_s in
  let length_train2 := total_distance - length_train1 in
  length_train2 = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  let speed_train1_ms := 120 * 1000 / 3600
  let speed_train2_ms := 100 * 1000 / 3600
  let relative_speed_ms := speed_train1_ms + speed_train2_ms
  let total_distance := relative_speed_ms * 9
  let length_train2 := total_distance - 300
  have : length_train2 = 250 := by sorry
  exact this

end length_of_other_train_l184_184790


namespace truthful_dwarfs_count_l184_184153

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184153


namespace coefficient_of_x3_in_expansion_l184_184322

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x3_in_expansion :
  let T_r := λ r : ℕ, binom 4 r * (-2)^r in
  T_r 4 + T_r 1 * (-2)^0 = 8 :=
by
  sorry

end coefficient_of_x3_in_expansion_l184_184322


namespace proposition_correct_l184_184381

def min_positive_period_sine :=
  ∀ x, y = abs (sin (2 * x - (π / 4))) → min_pos_period y = π / 2

def symmetry_about_point :=
  ∀ x, y = cos (x - (π / 3)) → symmetric_about y (2 * π / 3)

theorem proposition_correct :
  (∃ p, min_positive_period_sine p) ∨ (∃ q, symmetry_about_point q) :=
sorry

end proposition_correct_l184_184381


namespace pascals_triangle_sum_l184_184676

theorem pascals_triangle_sum (n : ℕ) : 
  let g := λ n, Real.log (2 * (2^n))
  in g(n) / Real.log2.exp = n + 1 :=
by {
  let g : ℕ → ℝ := λ n, Real.log (2 * (2^n)),
  sorry
}

end pascals_triangle_sum_l184_184676


namespace proof_correctness_l184_184617

-- Definitions for Lean
def logarithm_decreasing (a : ℝ) := (a > 0) ∧ (a ≠ 1) ∧ (∀ x y : ℝ, (1 < x) ∧ (x < y) → (log a x > log a y))
def log_a_2_negative (a : ℝ) := (log a 2 < 0)
def log_a_2_nonnegative (a : ℝ) := (log a 2 >= 0)

-- Proposition statements
def original_proposition (a : ℝ) := logarithm_decreasing a → log_a_2_negative a
def converse_proposition (a : ℝ) := log_a_2_negative a → logarithm_decreasing a
def contrapositive_proposition (a : ℝ) := log_a_2_nonnegative a → ¬logarithm_decreasing a

-- Theorem to prove correctness of converse and contrapositive propositions
theorem proof_correctness (a : ℝ) : original_proposition a → (converse_proposition a ∧ contrapositive_proposition a) := 
by 
  sorry

end proof_correctness_l184_184617


namespace infinitely_many_b_eq_finitely_many_p_eq_l184_184895

-- Definition of b(n) and p(n)
def b (n : ℕ) : ℕ := (factors n).length
def p (n : ℕ) : ℕ := (divisors n).sum

-- Proof statements
theorem infinitely_many_b_eq (k : ℕ) (h : k > 1) : 
  ∃∞ (n : ℕ), b n = k^2 - k + 1 := sorry

theorem finitely_many_p_eq (k : ℕ) (h : k > 1) : 
  {n : ℕ | p n = k^2 - k + 1}.finite := sorry

end infinitely_many_b_eq_finitely_many_p_eq_l184_184895


namespace determine_natural_number_l184_184125

theorem determine_natural_number (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p) :
  (p + q > r) ∧ (q + r > p) ∧ (r + p > q) → (p = 2 ∧ q = 3 ∧ r = 5) → (∑ i in [((p + q) / r) + ((q + r) / p) + ((r + p) / q)], i) = 7 :=
by
  sorry

end determine_natural_number_l184_184125


namespace tourist_groupings_l184_184754

theorem tourist_groupings : 
  let guides := 2 
  let tourists := 8 
  ∑ k in Finset.range (tourists + 1), if k > 0 ∧ k < tourists then (Nat.choose tourists k) else 0 = 254 :=
by
  sorry

end tourist_groupings_l184_184754


namespace initial_distance_between_stations_l184_184448

theorem initial_distance_between_stations
  (speedA speedB distanceA : ℝ)
  (rateA rateB : speedA = 40 ∧ speedB = 30)
  (dist_travelled : distanceA = 200) :
  (distanceA / speedA) * speedB + distanceA = 350 := by
  sorry

end initial_distance_between_stations_l184_184448


namespace range_of_function_l184_184424

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1 / 4))^2 - Real.log x / Real.log (1 / 4) + 5

theorem range_of_function : ∀ x : ℝ, 2 ≤ x → x ≤ 4 → 
  (∃ y : ℝ, y = f x ∧ (23 / 4 ≤ y ∧ y ≤ 7)) :=
by
  intro x hx1 hx2
  have hx : x ∈ Set.Icc 2 4 := ⟨hx1, hx2⟩
  use f x
  split
  · rfl
  · have : ∀ t : ℝ, -1 ≤ t → t ≤ -1/2 → (t - 1/2)^2 + 19/4 ∈ Set.Icc (23 / 4) 7 := sorry
    sorry

end range_of_function_l184_184424


namespace cottonwood_fiber_scientific_notation_l184_184187

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l184_184187


namespace bamboo_capacity_l184_184650

theorem bamboo_capacity :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 d : ℚ),
    a_1 + a_2 + a_3 = 4 ∧
    a_6 + a_7 + a_8 + a_9 = 3 ∧
    a_2 = a_1 + d ∧
    a_3 = a_1 + 2*d ∧
    a_4 = a_1 + 3*d ∧
    a_5 = a_1 + 4*d ∧
    a_7 = a_1 + 5*d ∧
    a_8 = a_1 + 6*d ∧
    a_9 = a_1 + 7*d ∧
    a_4 = 1 + 8/66 ∧
    a_5 = 1 + 1/66 :=
sorry

end bamboo_capacity_l184_184650


namespace front_exit_probability_correct_l184_184788

-- Define the necessary constants for our problem
structure Problem :=
  (initial_students : ℕ := 10)
  (insertion_probability : ℚ := 1 / 11)
  (removal_probability : ℚ := 1 / 2)
  (initial_position : ℕ := 8)
  (front_exit_probability : ℚ)

-- Define boundary conditions for the sequence
def boundary_conditions (p : ℕ → ℚ) : Prop :=
  p 0 = 1 ∧ p 11 = 0

-- Define the recurrence relation for probabilities
def recurrence_relation (p : ℕ → ℚ) : ℕ → Prop :=
  λ k, k > 0 ∧ k < 11 → p k = (11 - k) * p (k - 1) / 22 + k * p (k + 1) / 22

-- State the main problem in Lean
theorem front_exit_probability_correct :
  ∃ p : ℕ → ℚ, boundary_conditions p ∧ recurrence_relation p ∧ p 8 = 7 / 128 :=
sorry

end front_exit_probability_correct_l184_184788


namespace inequality_holds_iff_p_le_8_l184_184578

noncomputable def inequality_holds_for_p_x (p : ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x < π / 2) →
  (1 + 1 / sin x) ^ 3 ≥ p / (tan x) ^ 2

theorem inequality_holds_iff_p_le_8 (p : ℝ) :
  inequality_holds_for_p_x p ↔ p ≤ 8 :=
begin
  sorry
end

end inequality_holds_iff_p_le_8_l184_184578


namespace product_of_consecutive_multiples_of_4_divisible_by_192_l184_184780

theorem product_of_consecutive_multiples_of_4_divisible_by_192 :
  ∀ (n : ℤ), 192 ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) :=
by
  intro n
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_192_l184_184780


namespace numPythagoreanTriples_l184_184458

def isPythagoreanTriple (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x^2 + y^2 = z^2

theorem numPythagoreanTriples (n : ℕ) : ∃! T : (ℕ × ℕ × ℕ) → Prop, 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (T (2^(n+1))) :=
sorry

end numPythagoreanTriples_l184_184458


namespace find_a_l184_184433

theorem find_a (a : ℝ) : (∀ x : ℝ, f x = 2 / (3^x + 1) + a) → (f 1 = 0) → a = -1 / 2 := by
  sorry

end find_a_l184_184433


namespace tan_sub_eq_minus_2sqrt3_l184_184397

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end tan_sub_eq_minus_2sqrt3_l184_184397


namespace acute_angle_l184_184279

variables (x : ℝ)

def a : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem acute_angle (x : ℝ) : 
  (-2 / 3 < x) → x ≠ -2 / 3 → dot_product (2, x) (1, 3) > 0 :=
by
  intros h1 h2
  sorry

end acute_angle_l184_184279


namespace periodic_sequence_abs_l184_184912

theorem periodic_sequence_abs {a : ℕ → ℝ} 
  (h1 : ∀ i, a i = a (i + 100)) 
  (h2 : 0 ≤ a 1)
  (h3 : a 1 + a 2 ≤ 0)
  (h4 : a 1 + a 2 + a 3 ≥ 0)
  (h5 : ∀ n, if odd n then a_sum a n ≥ 0 else a_sum a n ≤ 0) :
  abs (a 99) ≥ abs (a 100) := 
sorry

noncomputable def a_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, a (k + 1)

end periodic_sequence_abs_l184_184912


namespace average_speed_correct_l184_184794

def speed1 := 30 -- kph
def distance1 := 10 -- km

def speed2 := 45 -- kph
def distance2 := 35 -- km

def speed3 := 55 -- kph
def time3 := 0.5 -- hours

def speed4 := 50 -- kph
def time4 := 2 / 3 -- hours

def total_distance := distance1 + distance2 + speed3 * time3 + speed4 * time4

def total_time := (distance1 / speed1) + (distance2 / speed2) + time3 + time4

theorem average_speed_correct :
  total_distance / total_time = 50.57 := by
  sorry

end average_speed_correct_l184_184794


namespace system_solution_l184_184193

variable (x y z t : ℝ)

theorem system_solution :
  (∃ (x y z t : ℝ), 
    (x * y + z + t = 1) ∧ 
    (y * z + t + x = 3) ∧ 
    (z * t + x + y = -1) ∧ 
    (t * x + y + z = 1) ∧
    x = 1 ∧ 
    y = 0 ∧ 
    z = -1 ∧ 
    t = 2) :=
  by
  let a : ℝ := 1
  let b : ℝ := 0
  let c : ℝ := -1
  let d : ℝ := 2
  have h₁ : a * b + c + d = 1 := by sorry
  have h₂ : b * c + d + a = 3 := by sorry
  have h₃ : c * d + a + b = -1 := by sorry 
  have h₄ : d * a + b + c = 1 := by sorry
  exists a b c d
  exact ⟨h₁, h₂, h₃, h₄, rfl, rfl, rfl, rfl⟩

end system_solution_l184_184193


namespace parallel_XB1_YC_l184_184378

open_locale classical

variables (A B C X Y B1 : Type*) 
variables [has_coe_to_fun (A B C X Y B1)] (P Q : Α → B)
variables {a b c x y b1 : P Q}

variables (triangle_ABC : Prop)
variables (AX_BY : dist A X = dist B Y)
variables (angle_XYB_BAC : ∃ (v u w : P Q), angle (P u w) = angle (P v w))
variables (angle_bisector_B : is_angle_bisector B A C B1)


/- To prove: -/
theorem parallel_XB1_YC
(triangle_ABC : is_triangle P Q)
(AX_BY : dist A X = dist B Y)
(angle_XYB_BAC : ∃ (v u w: P Q), angle (P u w) = angle (P v w))
(angle_bisector_B : is_angle_bisector B A C B1) :
parallel (line X B1) (line Y C) :=
sorry

end parallel_XB1_YC_l184_184378


namespace road_time_l184_184471

theorem road_time (departure : ℕ) (arrival : ℕ) (stops : List ℕ) : 
  departure = 7 ∧ arrival = 20 ∧ stops = [25, 10, 25] → 
  ((arrival - departure) * 60 - stops.sum) / 60 = 12 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 hstops
  have h_duration : (20 - 7) * 60 = 780 := rfl
  have h_stops : stops.sum = 60 := by
    simp [hstops]
  have h_total_time_on_road : (780 - 60) / 60 = 12 := rfl
  exact h_total_time_on_road

end road_time_l184_184471


namespace min_avg_distance_l184_184434

theorem min_avg_distance (n : ℕ) (h : n ≥ 3) (d : (ℕ × ℕ) → ℕ) 
  (H1 : ∀ (A B : ℕ), A ≠ B → (∃ p : List ℕ, A :: p ++ [B] ∈ all_paths n ∧ length p = d (A, B))) :
  ∃ G : Graph (ℕ → ℕ),
    (∀ (A B : ℕ), A ≠ B → ((d (A, B) = 1) ∨ (d (A, B) = 2))) ∧
    ( ∑ A B in finset.univ.filter (λ AB, (AB.1 ≠ AB.2)), d (A, B) ) / ( choose n 2 ) = 3 / 2 :=
sorry

end min_avg_distance_l184_184434


namespace mixed_groups_count_l184_184004

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l184_184004


namespace tourist_groupings_l184_184753

theorem tourist_groupings : 
  let guides := 2 
  let tourists := 8 
  ∑ k in Finset.range (tourists + 1), if k > 0 ∧ k < tourists then (Nat.choose tourists k) else 0 = 254 :=
by
  sorry

end tourist_groupings_l184_184753


namespace min_positive_period_f_max_value_f_decreasing_intervals_g_l184_184272

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem min_positive_period_f : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x : ℝ, f (x + 2*Real.pi) = f x) :=
sorry

theorem max_value_f : 
  ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 2 :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (-x)

theorem decreasing_intervals_g :
  ∀ (k : ℤ), ∀ x : ℝ, (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
  ∀ (h : x ≤ Real.pi * 2 * (↑k+1)), g x ≥ g (x + Real.pi) :=
sorry

end min_positive_period_f_max_value_f_decreasing_intervals_g_l184_184272


namespace odd_function_check_l184_184936

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184936


namespace simplification_evaluation_l184_184398

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 :=
by
  sorry

end simplification_evaluation_l184_184398


namespace ab_intersect_x_axis_fixed_point_l184_184229

theorem ab_intersect_x_axis_fixed_point (x1 y1 x2 y2 p λ : ℝ) (hp : p > 0) 
    (hA : y1^2 = 2 * p * x1) (hB : y2^2 = 2 * p * x2) :
    (∃ m : ℝ, ∀ (x : ℝ), (∃ (k : ℝ), (y1 = k * (x1 - m)) ∧ (y2 = k * (x2 - m))) -> (x = m)) ↔ (y1 * y2 = λ) :=
by
  sorry

end ab_intersect_x_axis_fixed_point_l184_184229


namespace quadratic_vertex_l184_184721

theorem quadratic_vertex (x y : ℝ) (h : y = -3 * x^2 + 2) : (x, y) = (0, 2) :=
sorry

end quadratic_vertex_l184_184721


namespace area_of_triangle_l184_184758

-- Define the equations of the lines
def line1 (x : ℝ) : ℝ := 7
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define points of intersection
def point1 : ℝ × ℝ := (5, 7)
def point2 : ℝ × ℝ := (-5, 7)
def point3 : ℝ × ℝ := (0, 2)

-- Define a function to calculate the area of the triangle using the vertices
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2) - 
                   (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p1.1)))

-- Theorem statement
theorem area_of_triangle : triangle_area (point1) (point2) (point3) = 25 :=
by
  -- skip the proof
  sorry

end area_of_triangle_l184_184758


namespace security_deposit_amount_correct_l184_184664

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end security_deposit_amount_correct_l184_184664


namespace ratio_fraction_l184_184640

variable (X Y Z : ℝ)
variable (k : ℝ) (hk : k > 0)

-- Given conditions
def ratio_condition := (3 * Y = 2 * X) ∧ (6 * Y = 2 * Z)

-- Statement
theorem ratio_fraction (h : ratio_condition X Y Z) : 
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end ratio_fraction_l184_184640


namespace cos_arcsin_eq_l184_184842

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l184_184842


namespace two_trains_meet_at_distance_l184_184451

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end two_trains_meet_at_distance_l184_184451


namespace probability_roots_of_unity_l184_184351

theorem probability_roots_of_unity (v w : ℂ) (h1 : v ^ 401 = 1) (h2 : w ^ 401 = 1) (h3 : v ≠ w) :
  (∀ v w, v ≠ w ∧ v ^ 401 = 1 ∧ w ^ 401 = 1 → sqrt (3 + sqrt 5) ≤ complex.abs (v + w) → false) :=
by {
  intros,
  sorry
}

end probability_roots_of_unity_l184_184351


namespace find_initial_jellybeans_l184_184131

-- Definitions of the initial conditions
def jellybeans_initial (x : ℝ) (days : ℕ) (remaining : ℝ) := 
  days = 4 ∧ remaining = 48 ∧ (0.7 ^ days) * x = remaining

-- The theorem to prove
theorem find_initial_jellybeans (x : ℝ) : 
  jellybeans_initial x 4 48 → x = 200 :=
sorry

end find_initial_jellybeans_l184_184131


namespace robot_speedup_l184_184528

noncomputable def time_ratio (a v : ℝ) : ℝ :=
  let t1 := 9 * a / v
  let t2 := a / v
  t1 / t2

theorem robot_speedup {a v : ℝ} (h_a_pos : a > 0) (h_v_pos : v > 0) :
  time_ratio a v = 9 :=
by
  dsimp [time_ratio]
  field_simp [h_a_pos.ne', h_v_pos.ne']
  norm_num
  sorry

end robot_speedup_l184_184528


namespace max_abs_diff_inequality_l184_184340

theorem max_abs_diff_inequality 
  (n : ℕ)
  (a b : Fin n → ℝ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j) 
  (a' b' : Fin n → ℝ)
  (a'_sorted : ∀ i j, i < j → a' i < a' j)
  (b'_sorted : ∀ i j, i < j → b' i < b' j) 
  : 
  max (Fin n) (λ i, abs (a i - b i)) ≥ max (Fin n) (λ i, abs (a' i - b' i)) :=
by 
  sorry

end max_abs_diff_inequality_l184_184340


namespace second_divisor_13_l184_184795

theorem second_divisor_13 (N D : ℤ) (k m : ℤ) 
  (h1 : N = 39 * k + 17) 
  (h2 : N = D * m + 4) : 
  D = 13 := 
sorry

end second_divisor_13_l184_184795


namespace odd_function_check_l184_184930

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184930


namespace normal_distribution_properties_l184_184532

open ProbabilityTheory

noncomputable def standard_normal_CDF (x : ℝ) : ℝ := P(λ (ξ : ℝ), ξ < x)

theorem normal_distribution_properties :
  (∀ x : ℝ, standard_normal_CDF x = P(λ (ξ : ℝ), ξ < x)) → 
  (standard_normal_CDF 0 = 0.5) ∧
  (∀ x : ℝ, standard_normal_CDF x = 1 - standard_normal_CDF (-x)) ∧
  (P(λ (ξ : ℝ), |ξ| < 2) = 2 * standard_normal_CDF 2 - 1) :=
by sorry

end normal_distribution_properties_l184_184532


namespace ellipse_range_k_l184_184259

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l184_184259


namespace odd_function_check_l184_184957

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184957


namespace lesser_solution_of_quadratic_l184_184463

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l184_184463


namespace problem_l184_184223

theorem problem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, S n = ∑ i in finset.range n, a (i + 1))
  (h3 : ∀ n ≥ 2, (a n - S (n - 1)) ^ 2 = S n * S (n - 1))
  (h4 : a 1 = 1) :
  ∀ n, log 2 (a (n + 1) / 6) = 2 * n - 3 :=
by sorry

end problem_l184_184223


namespace dwarfs_truthful_count_l184_184148

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184148


namespace genuine_items_count_l184_184443

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l184_184443


namespace green_light_hours_l184_184534

theorem green_light_hours (d : ℝ) (r_g_y_r : ℝ × ℝ × ℝ)
  (h_d : d = 24) (h_r : r_g_y_r = (6, 1, 3)) : 
  d * (r_g_y_r.1 / (r_g_y_r.1 + r_g_y_r.2 + r_g_y_r.3)) = 14.4 :=
by {
  sorry,
}

end green_light_hours_l184_184534


namespace max_value_of_a_l184_184639

theorem max_value_of_a {a : ℝ} (h : ∀ x ≥ 1, -3 * x^2 + a ≤ 0) : a ≤ 3 :=
sorry

end max_value_of_a_l184_184639


namespace IsConcyclic_l184_184601

open EuclideanGeometry

variables {A B C P X Y M : Point}
variables {BC : Line}
variables (h1 : IsIsoscelesTriangle ABC)
variables (h2 : IsMidpointOf M B C)
variables (h3 : PB < PC ∧ Parallel PA BC)
variables (h4 : OnExtension PB X ∧ OnExtension PC Y)
variables (h5 : ∠ PXM = ∠ PYM)

theorem IsConcyclic (h1 : IsIsoscelesTriangle ABC) 
                     (h2 : IsMidpointOf M B C)
                     (h3 : PB < PC ∧ Parallel PA BC)
                     (h4 : OnExtension PB X ∧ OnExtension PC Y)
                     (h5 : ∠ PXM = ∠ PYM) : 
    Concyclic A P X Y :=
by
  sorry

end IsConcyclic_l184_184601


namespace double_add_treble_l184_184804

theorem double_add_treble (original : ℕ) (h : original = 7) : 
  let doubled := original * 2,
      added := doubled + 9,
      trebled := added * 3 in trebled = 69 :=
by
  subst h
  -- proof steps would go here, ending with "exact rfl"
  sorry

end double_add_treble_l184_184804


namespace mixed_groups_count_l184_184002

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l184_184002


namespace inequality_S_n_l184_184699

def r_n (n a b : ℕ) : ℕ := (a * b) % n

def S_n (n : ℕ) : ℕ :=
  ∑ a in Finset.range n, ∑ b in Finset.range n, r_n n a b

theorem inequality_S_n (n : ℕ) (hn : 0 < n) :
  (1 : ℝ) / 2 - 1 / Real.sqrt n ≤ (S_n n : ℝ) / n^3 ∧ (S_n n : ℝ) / n^3 ≤ 1 / 2 :=
sorry

end inequality_S_n_l184_184699


namespace transformed_function_is_odd_l184_184969

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184969


namespace road_time_l184_184472

theorem road_time (departure : ℕ) (arrival : ℕ) (stops : List ℕ) : 
  departure = 7 ∧ arrival = 20 ∧ stops = [25, 10, 25] → 
  ((arrival - departure) * 60 - stops.sum) / 60 = 12 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 hstops
  have h_duration : (20 - 7) * 60 = 780 := rfl
  have h_stops : stops.sum = 60 := by
    simp [hstops]
  have h_total_time_on_road : (780 - 60) / 60 = 12 := rfl
  exact h_total_time_on_road

end road_time_l184_184472


namespace TotalGenuineItems_l184_184446

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l184_184446


namespace p_sufficient_not_necessary_for_q_l184_184595

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- State the theorem that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l184_184595


namespace odd_function_check_l184_184942

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184942


namespace evaluate_floor_abs_neg_l184_184170

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184170


namespace agreed_period_of_service_l184_184507

theorem agreed_period_of_service (x : ℕ) (rs800 : ℕ) (rs400 : ℕ) (servant_period : ℕ) (received_amount : ℕ) (uniform : ℕ) (half_period : ℕ) :
  rs800 = 800 ∧ rs400 = 400 ∧ servant_period = 9 ∧ received_amount = 400 ∧ half_period = x / 2 ∧ servant_period = half_period → x = 18 :=
by sorry

end agreed_period_of_service_l184_184507


namespace number_of_matching_integers_l184_184629

def sum_of_digit_factorials (n : ℕ) : ℕ :=
  n.digits 10 |> List.map (λ d, (d.factorial : ℕ)) |> List.sum

theorem number_of_matching_integers : 
  { n : ℕ // n < 2010 ∧ sum_of_digit_factorials n = n } → ℕ := 
by
  sorry

#eval number_of_matching_integers

end number_of_matching_integers_l184_184629


namespace least_number_of_stamps_is_6_l184_184111

noncomputable def exist_stamps : Prop :=
∃ (c f : ℕ), 5 * c + 7 * f = 40 ∧ c + f = 6

theorem least_number_of_stamps_is_6 : exist_stamps :=
sorry

end least_number_of_stamps_is_6_l184_184111


namespace min_value_of_abc_l184_184604

variables {a b c : ℝ}

noncomputable def satisfies_condition (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ (b + c) / a + (a + c) / b = (a + b) / c + 1

theorem min_value_of_abc (a b c : ℝ) (h : satisfies_condition a b c) : (a + b) / c ≥ 5 / 2 :=
sorry

end min_value_of_abc_l184_184604


namespace infinite_set_exists_l184_184220

open Nat

theorem infinite_set_exists (f : ℕ → ℕ) (h : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ M : Set ℕ, Set.Infinite M ∧ ∀ i j k ∈ M, (i - j) * f k + (j - k) * f i + (k - i) * f j = 0 :=
sorry

end infinite_set_exists_l184_184220


namespace ratio_of_lengths_l184_184379

-- Define the problem statement
theorem ratio_of_lengths {A B C K U L M N : Point} 
  (h_K_midpoint : midpoint K A C)
  (h_U_midpoint : midpoint U B C)
  (h_L_on_CK : L ∈ segment C K)
  (h_M_on_CU : M ∈ segment C U)
  (h_LM_parallel_KU : parallel L M K U)
  (h_N_on_AB : N ∈ segment A B)
  (h_AN_AB_ratio : |AN| / |AB| = 3 / 7)
  (h_area_ratio : area_polygon U M L K / area_polygon M L K N U = 3 / 7) :
  |LM| / |KU| = 1 / 2 :=
sorry -- proof is not required

end ratio_of_lengths_l184_184379


namespace average_score_first_2_matches_l184_184718

theorem average_score_first_2_matches (A : ℝ) 
  (h1 : 3 * 40 = 120) 
  (h2 : 5 * 36 = 180) 
  (h3 : 2 * A + 120 = 180) : 
  A = 30 := 
by 
  have hA : 2 * A = 60 := by linarith [h3]
  have hA2 : A = 30 := by linarith [hA]
  exact hA2

end average_score_first_2_matches_l184_184718


namespace units_digit_sum_1_to_15_l184_184761

-- Definition of factorial units digit properties
def factorial_units_digit (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 4
  | _ => 0

-- Sum of unique units digits of factorials from 1! to 4!
def sum_units_digits (n : ℕ) : ℕ :=
  if n <= 4 then (list.range (n + 1)).map factorial_units_digit |>.sum
  else (list.range 5).map factorial_units_digit |>.sum

-- Theorems proving conditions and desired result
theorem units_digit_sum_1_to_15 : sum_units_digits 15 % 10 = 3 := by
  sorry

end units_digit_sum_1_to_15_l184_184761


namespace distance_from_P_to_FA_l184_184082

theorem distance_from_P_to_FA (d_AB d_BC d_CD d_DE d_EF x : ℝ) 
  (h1 : d_AB = 1) 
  (h2 : d_BC = 2) 
  (h3 : d_CD = 5) 
  (h4 : d_DE = 7) 
  (h5 : d_EF = 6) 
  (height : ∀ a b c d e f : ℝ, a + d = 8 ∧ b + e = 8 ∧ c + f = 8) 
  : x = 3 :=
by 
  have h6 : d_CD + x = 8 := height 1 2 5 7 6 x
  rw [←h3] at h6
  linarith
  
-- sorry here to skip the proof

end distance_from_P_to_FA_l184_184082


namespace find_SM_l184_184811

variables (A B C D E F H S M O : Point)
variables (R : ℝ)
variables (BC AD EO BD CE BE : Line)
variables (circle : Circle)
variables (trapezoid : trapezoid inscribed in circle)
variables (CH : ℝ)

-- Conditions
axiom BC_parallel_AD : parallel BC AD
axiom trapezoid_inscribed : inscribed trapezoid circle
axiom CE_diameter : diameter CE circle
axiom BE_intersect_AD_at_F : intersect BE AD F
axiom H_perp_F_to_CE : perp_from_line H F CE
axiom S_midpoint_EO : midpoint S EO
axiom M_midpoint_BD : midpoint M BD
axiom CH_eq : CH = 9 * R / 8

-- Proof problem
theorem find_SM : 
  SM = 3 * R / (2 * sqrt 2) :=
sorry

end find_SM_l184_184811


namespace odd_function_g_l184_184992

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184992


namespace part1_part2_l184_184499

theorem part1 (x y : ℝ) (h1 : y = x + 30) (h2 : 2 * x + 3 * y = 340) : x = 50 ∧ y = 80 :=
by {
  -- Later, we can place the steps to prove x = 50 and y = 80 here.
  sorry
}

theorem part2 (m : ℕ) (h3 : 0 ≤ m ∧ m ≤ 50)
               (h4 : 54 * (50 - m) + 72 * m = 3060) : m = 20 :=
by {
  -- Later, we can place the steps to prove m = 20 here.
  sorry
}

end part1_part2_l184_184499


namespace total_commute_time_l184_184692

theorem total_commute_time 
  (first_bus : ℕ) (delay1 : ℕ) (wait1 : ℕ) 
  (second_bus : ℕ) (delay2 : ℕ) (wait2 : ℕ) 
  (third_bus : ℕ) (delay3 : ℕ) 
  (arrival_time : ℕ) :
  first_bus = 40 →
  delay1 = 10 →
  wait1 = 10 →
  second_bus = 50 →
  delay2 = 5 →
  wait2 = 15 →
  third_bus = 95 →
  delay3 = 15 →
  arrival_time = 540 →
  first_bus + delay1 + wait1 + second_bus + delay2 + wait2 + third_bus + delay3 = 240 :=
by
  intros
  sorry

end total_commute_time_l184_184692


namespace sandwiches_total_slices_l184_184387

def pb : Nat := 3
def tuna : Nat := 4
def turkey : Nat := 2
def pb_slices : Float := 2
def tuna_slices : Float := 3
def turkey_slices : Float := 1.5

theorem sandwiches_total_slices : 
  pb * pb_slices + tuna * tuna_slices + turkey * turkey_slices = 21 :=
by
  sorry

end sandwiches_total_slices_l184_184387


namespace evaluate_floor_abs_neg_l184_184173

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184173


namespace prism_lateral_surface_area_l184_184590

-- Conditions definition for the triangular prism
variables (a h r : ℝ) -- side length, height of prism, radius of sphere
variables (volume_of_sphere : ℝ) (is_equilateral_triangle : Prop) (is_perpendicular_edges : Prop)

-- Given conditions
def condition_volume_sphere : volume_of_sphere = (4 / 3) * real.pi * r^3 := sorry
def condition_radius : r = 1 := sorry
def condition_height : h = 2 := sorry
def condition_side_length : a = 2 * real.sqrt(3) := sorry

-- Main theorem to prove
theorem prism_lateral_surface_area 
  (h_volume_sphere : condition_volume_sphere) 
  (h_radius : condition_radius)
  (h_height : condition_height)
  (h_side_length : condition_side_length) 
  (h_equilateral : is_equilateral_triangle) 
  (h_perpendicular : is_perpendicular_edges) : 
  3 * a * h = 12 * real.sqrt(3) := 
sorry

end prism_lateral_surface_area_l184_184590


namespace find_min_palindrome_addition_l184_184024

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string;
  s = s.reverse

theorem find_min_palindrome_addition :
  ∃ n : ℕ, n > 0 ∧ is_palindrome (54321 + n) ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬is_palindrome (54321 + m) :=
begin
  sorry
end

end find_min_palindrome_addition_l184_184024


namespace arithmetic_sequence_problem_l184_184224

-- Define sequence and sum properties
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

/- Theorem Statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) 
  (h_seq : arithmetic_sequence a d) 
  (h_initial : a 1 = 31) 
  (h_S_eq : S 10 = S 22) :
  -- Part 1: Find S_n
  (∀ n, S n = 32 * n - n ^ 2) ∧
  -- Part 2: Maximum sum occurs at n = 16 and is 256
  (∀ n, S n ≤ 256 ∧ (S 16 = 256 → ∀ m ≠ 16, S m < 256)) :=
by
  -- proof to be provided here
  sorry

end arithmetic_sequence_problem_l184_184224


namespace find_phi_sum_f_l184_184787

noncomputable def f (x : ℝ) (A ω φ : ℝ) : ℝ := A * (sin (ω * x + φ))^2

/-- Assumptions and given conditions -/
variables (A ω φ : ℝ)
variables (hA_pos : A > 0) (hω_pos : ω > 0) (hφ_range : 0 < φ ∧ φ < π / 2)
variables (h_max_val : ∀ x, f x A ω φ ≤ 2 ∧ (∃ x₀, f x₀ A ω φ = 2))
variables (h_symmetry : ∀ x, f x A ω φ = f (x + 2) A ω φ)
variables (h_point : f 1 A ω φ = 2)

/-- The first part of the proof: Prove that φ = 3π/2 -/
theorem find_phi : φ = 3 * π / 2 :=
sorry

/-- The second part of the proof: Prove that the sum f(1) + f(2) + ⋯ + f(2017) = 2018 -/
theorem sum_f : (finset.range 2017).sum (λ i, f (i + 1) A ω φ) = 2018 :=
sorry

end find_phi_sum_f_l184_184787


namespace ticket_price_values_l184_184088

theorem ticket_price_values :
  ∃ (x : ℕ), (∃ (n9 n10 n11 : ℕ), 60 = n9 * x ∧ 75 = n10 * x ∧ 90 = n11 * x) ∧ 
  (finset.card (finset.filter (λ d, 60 % d = 0 ∧ 75 % d = 0 ∧ 90 % d = 0) (finset.range 61)) = 4) :=
begin
  sorry
end

end ticket_price_values_l184_184088


namespace odd_function_check_l184_184953

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184953


namespace trigonometric_identity_l184_184901

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : sin (3 / 2 * Real.pi - θ) + 2 * cos (Real.pi + θ) = sin (2 * Real.pi - θ)) :
  sin θ * cos θ + 2 * (cos θ)^2 = 1 / 2 := 
sorry

end trigonometric_identity_l184_184901


namespace trigonometric_identity_solution_l184_184475

open Real

noncomputable def x_sol1 (k : ℤ) : ℝ := (π / 2) * (4 * k - 1)
noncomputable def x_sol2 (l : ℤ) : ℝ := (π / 3) * (6 * l + 1)
noncomputable def x_sol2_neg (l : ℤ) : ℝ := (π / 3) * (6 * l - 1)

theorem trigonometric_identity_solution (x : ℝ) :
    (3 * sin (x / 2) ^ 2 * cos (3 * π / 2 + x / 2) +
    3 * sin (x / 2) ^ 2 * cos (x / 2) -
    sin (x / 2) * cos (x / 2) ^ 2 =
    sin (π / 2 + x / 2) ^ 2 * cos (x / 2)) →
    (∃ k : ℤ, x = x_sol1 k) ∨
    (∃ l : ℤ, x = x_sol2 l ∨ x = x_sol2_neg l) :=
by
  sorry

end trigonometric_identity_solution_l184_184475


namespace tan_half_angle_sum_l184_184329

theorem tan_half_angle_sum (A B C : ℝ) (h : A + B + C = π) :
  tan (A / 2) * tan (B / 2) + tan (B / 2) * tan (C / 2) + tan (A / 2) * tan (C / 2) = 1 := 
  sorry

end tan_half_angle_sum_l184_184329


namespace crocodile_can_reach_any_cell_l184_184065

def can_reach_by_crocodile (N : ℕ) (start end : ℤ × ℤ) : Prop :=
  ∃ (moves : List (ℤ × ℤ)), 
    moves.head = start ∧ 
    moves.last = end ∧
    (∀ (move₁ move₂ : (ℤ × ℤ)), move₁ ∈ moves ∧ move₂ ∈ moves ∧ 
      ((move₁.1 - move₂.1).abs = 1 ∧ (move₁.2 - move₂.2).abs = N) ∨ 
      ((move₁.1 - move₂.1).abs = N ∧ (move₁.2 - move₂.2).abs = 1))

theorem crocodile_can_reach_any_cell (N : ℕ) : 
  (∀ (start end : ℤ × ℤ), can_reach_by_crocodile N start end) ↔ (N % 2 = 0) :=
begin
  sorry,
end

end crocodile_can_reach_any_cell_l184_184065


namespace hundredth_smallest_element_in_S_l184_184670

-- Define the set S as described in the problem conditions
def S : Set ℕ := {n | ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ n = 2^x + 2^y + 2^z}

-- State the main theorem that needs to be proved
theorem hundredth_smallest_element_in_S : 
  nth_smallest S 100 = 577 :=
sorry

end hundredth_smallest_element_in_S_l184_184670


namespace fraction_value_l184_184723

theorem fraction_value (a : ℕ) (h : a > 0) (h_eq : (a:ℝ) / (a + 35) = 0.7) : a = 82 :=
by
  -- Steps to prove the theorem here
  sorry

end fraction_value_l184_184723


namespace range_log2_cube_root_sin_l184_184129

theorem range_log2_cube_root_sin (x : ℝ) (h1 : 0 < x) (h2 : x < 180) : 
  ∃ y ∈ Icc (-∞ : ℝ∞) 0, y = Real.log 2 (Real.cbrt (Real.sin (Real.toRad x))) := 
sorry

end range_log2_cube_root_sin_l184_184129


namespace bridge_length_l184_184483

noncomputable def length_of_bridge (length_of_train : ℕ) (speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_seconds
  total_distance - length_of_train

theorem bridge_length {length_of_train : ℕ} {speed_kmph : ℕ} {time_seconds : ℕ} :
  length_of_train = 130 →
  speed_kmph = 45 →
  time_seconds = 30 →
  length_of_bridge length_of_train speed_kmph time_seconds = 245 :=
by
  intros h_train h_speed h_time
  rw [h_train, h_speed, h_time]
  unfold length_of_bridge
  norm_num


end bridge_length_l184_184483


namespace quadratic_inequality_solution_l184_184296

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end quadratic_inequality_solution_l184_184296


namespace roots_square_sum_eq_l184_184634

theorem roots_square_sum_eq (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) 
  (h3 : r * s * t = r) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by
  sorry

end roots_square_sum_eq_l184_184634


namespace total_fish_sold_l184_184442

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end total_fish_sold_l184_184442


namespace regular_polyhedron_not_necessary_l184_184658

-- Defining the conditions for the problem
variables (P : Type) [Polyhedron P]
variables [ConvexPolyhedron P]
variables (faces_equal : ∀ (f1 f2 : Face P), f1 = f2)
variables (angles_equal : ∀ (a1 a2 : PolyhedralAngle P), a1 = a2)

-- Stating the problem
theorem regular_polyhedron_not_necessary :
  ¬(∀ (P : Type) [ConvexPolyhedron P] 
    (faces_equal : ∀ (f1 f2 : Face P), f1 = f2) 
    (angles_equal : ∀ (a1 a2 : PolyhedralAngle P), a1 = a2), 
    RegularPolyhedron P) :=
sorry

end regular_polyhedron_not_necessary_l184_184658


namespace number_of_truthful_dwarfs_is_correct_l184_184134

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184134


namespace find_a_l184_184267

def f (a x : ℝ) : ℝ := (x + a) * Real.exp x

def tangent_slope (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (fun x => (Real.exp x) * (x + a + 1))

theorem find_a (a : ℝ) :
  let f_minus_one_slope := (a / Real.exp 1)
  let f_one_slope := ((Real.exp 1) * (2 + a))
  f_minus_one_slope * f_one_slope = -1 ->
  a = -1 :=
by {
  sorry
}

end find_a_l184_184267


namespace part1_general_formula_part2_sum_formula_l184_184250

noncomputable def geometric_sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a 2 + a 3 + a 4 = 39 ∧ a 5 = 2 * a 4 + 3 * a 3

theorem part1_general_formula (a : ℕ → ℝ) (h : geometric_sequence_condition a) :
  ∀ n, a n = 3 ^ (n - 1) :=
sorry

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n + a n

theorem part2_sum_formula (a : ℕ → ℝ) (h : geometric_sequence_condition a) (h1 : ∀ n, a n = 3 ^ (n - 1)) :
  ∀ n, ∑ i in Finset.range n, b a (i + 1) = (3^n + n^2 + n - 1) / 2 :=
sorry

end part1_general_formula_part2_sum_formula_l184_184250


namespace find_ellipse_l184_184253

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the focus points
def focus (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

-- Define the range condition
def range_condition (a b c : ℝ) : Prop :=
  let min_val := b^2 - c^2;
  let max_val := a^2 - c^2;
  min_val = -3 ∧ max_val = 3

-- Prove the equation of the ellipse
theorem find_ellipse (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos ∧ focus a b c ∧ range_condition a b c) →
  (a^2 = 9 ∧ b^2 = 3) :=
by
  sorry

end find_ellipse_l184_184253


namespace sum_floor_series_eq_n_l184_184673

theorem sum_floor_series_eq_n (n : ℕ) (h : 0 < n) :
  ∑ k in Finset.range (nat.succ $ nat.log 2 n), ⌊(n + 2^k) / 2^(k+1)⌋ = n :=
by
  sorry

end sum_floor_series_eq_n_l184_184673


namespace cos_arcsin_eq_l184_184844

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l184_184844


namespace max_value_expression_l184_184553

noncomputable def expression (x : ℝ) : ℝ :=
  (x^2 + 5*x + 12) * (x^2 + 5*x - 12) * (x^2 - 5*x + 12) * (-x^2 + 5*x + 12) / x^4

theorem max_value_expression : ∃ x : ℝ, x ≠ 0 → expression x ≤ 576 := 
begin
  sorry
end

end max_value_expression_l184_184553


namespace proof_AM_PM_eq_BM_sq_l184_184591

open EuclideanGeometry

variable {A B C H M P : Point}

-- Given conditions
axiom is_acute_triangle : ∀ {A B C : Triangle}, AcuteTriangle(A, B, C)
axiom midpoint_M : Midpoint(M, B, C)
axiom orthocenter_H : Orthocenter(H, A, B, C)
axiom perpendicular_HP_AM : PerpendicularSegment(H, P, AM)

-- Theorem statement
theorem proof_AM_PM_eq_BM_sq :
  is_acute_triangle A B C →
  midpoint_M M B C →
  orthocenter_H H A B C →
  perpendicular_HP_AM H P (Segment A M) →
  (Dist A M) * (Dist P M) = (Dist B M)^2 :=
  sorry

end proof_AM_PM_eq_BM_sq_l184_184591


namespace relationship_y1_y2_y3_l184_184636

-- Conditions
def A := (-1, y1)
def B := (2, y2)
def C := (3, y3)
def inv_proportion_function (x : ℝ) : ℝ := -6 / x

-- Points lie on the graph
axiom A_on_graph : inv_proportion_function (-1) = y1
axiom B_on_graph : inv_proportion_function 2 = y2
axiom C_on_graph : inv_proportion_function 3 = y3

-- Prove the relationship
theorem relationship_y1_y2_y3 (y1 y2 y3 : ℝ) (A_on_graph B_on_graph C_on_graph : Prop) : y1 > y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l184_184636


namespace angle_between_vectors_l184_184366

open Real
open EuclideanSpace

variables (a b c d : ℝ^3)
variables (theta : ℝ)

theorem angle_between_vectors
  (ha_norm : ‖a‖ = 1)
  (hb_norm : ‖b‖ = 1)
  (hc_norm : ‖c‖ = 3)
  (triple_product : a × (a × c) + b = 0)
  (dot_da : d ⬝ a = 0)
  (dot_db : d ⬝ b = 0) :
  theta = real.arccos (2 * real.sqrt 2 / 3) ∨ theta = real.arccos (-2 * real.sqrt 2 / 3) :=
sorry

end angle_between_vectors_l184_184366


namespace product_of_coordinates_eq_neg2_l184_184228

-- Definitions and conditions
variables {x y : ℝ}
def A : ℝ × ℝ × ℝ := (1, -2, 11)
def B : ℝ × ℝ × ℝ := (4, 2, 3)
def C : ℝ × ℝ × ℝ := (x, y, 15)

-- Collinearity condition: Points A, B, and C are collinear if the vectors AB and AC are linearly dependent.
def collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, z1) := B in
  let (x2, y2, z2) := A in
  let (x3, y3, z3) := C in
  (x1 - x2) * (y3 - y2) * (z3 - z2) + (y1 - y2) * (z3 - z2) * (x3 - x2) + (z1 - z2) * (x3 - x2) * (y3 - y2) =
  (x3 - x2) * (y1 - y2) * (z3 - z2) + (y3 - y2) * (z1 - z2) * (x3 - x2) + (z3 - z2) * (x1 - x2) * (y3 - y2)

-- The main theorem statement
theorem product_of_coordinates_eq_neg2 (h : collinear A B C) : x * y = -2 :=
  sorry

end product_of_coordinates_eq_neg2_l184_184228


namespace compound_proposition_C_l184_184231

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x 
def q : Prop := ∀ x : ℝ, sin x < x

theorem compound_proposition_C : p ∧ ¬q :=
by sorry

end compound_proposition_C_l184_184231


namespace number_of_truthful_dwarfs_is_correct_l184_184136

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184136


namespace prime_number_condition_l184_184765

theorem prime_number_condition :
  ∃ p : ℕ, p.prime ∧ (∃ a : ℕ, p - 5 = a^2 ∧ p + 8 = (a + 1)^2) ∧ p = 41 :=
by {
  sorry
}

end prime_number_condition_l184_184765


namespace coworker_phone_ratio_l184_184663

theorem coworker_phone_ratio :
  ∀ (initial_phones : ℕ) (fixed_phones : ℕ) (new_phones : ℕ) (phones_each : ℕ),
    initial_phones = 15 →
    fixed_phones = 3 →
    new_phones = 6 →
    phones_each = 9 →
    ((phones_each : ℚ) / (initial_phones - fixed_phones + new_phones) : ℚ) = 1 / 2 :=
by
  assume (initial_phones fixed_phones new_phones phones_each : ℕ)
  assume h₁ : initial_phones = 15
  assume h₂ : fixed_phones = 3
  assume h₃ : new_phones = 6
  assume h₄ : phones_each = 9
  sorry

end coworker_phone_ratio_l184_184663


namespace problem_statement_l184_184611

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 3)

def h (x m : ℝ) := 2 * f x + 1 - m

theorem problem_statement (m : ℝ) :
  (∀ x, f x ≤ 3) →
  (∀ x, f x ≥ -3) →
  (∀ x, x = Real.pi / 12 → f x = 3) →
  (∀ x, x = 7 * Real.pi / 12 → f x = -3) →
  (∀ x, x ∈ Icc (-Real.pi / 3) (Real.pi / 6) → h x m = 0) →
  m ∈ Icc (3 * Real.sqrt 3 + 1) 7 :=
sorry

end problem_statement_l184_184611


namespace apple_picking_ratio_l184_184339

theorem apple_picking_ratio (a b c : ℕ) 
  (h1 : a = 66) 
  (h2 : b = 2 * 66) 
  (h3 : a + b + c = 220) :
  c = 22 → a = 66 → c / a = 1 / 3 := by
    intros
    sorry

end apple_picking_ratio_l184_184339


namespace fish_added_today_l184_184406

theorem fish_added_today (a b total : ℕ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : total = 10) :
  total - (a + b) = 3 :=
by
  rw [h₁, h₂, h₃]
  sorry

end fish_added_today_l184_184406


namespace basketball_total_opponent_points_l184_184068

variable team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]

-- Define conditions
noncomputable def lost_games (scores : List ℕ) : List ℕ :=
  [1, 3, 5, 7, 12, 14]  -- Identified based on loss condition described

noncomputable def opponent_score (team_score : ℕ) (lost_game : Bool) : ℕ :=
  if lost_game then team_score + 2
  else if team_score % 3 == 0 then team_score / 3 else team_score / 2

def total_opponent_score (scores : List ℕ) (loss_ids : List ℕ) : ℕ :=
  scores.mapWithIndex (fun i score => 
    opponent_score score (i + 1 ∈ loss_ids)
  ).sum

-- list of indices indicating the games that were lost
#eval total_opponent_score team_scores [1, 3, 5, 7, 12, 14]

def problem_proof : Prop :=
  total_opponent_score team_scores [1, 3, 5, 7, 12, 14] = 72

theorem basketball_total_opponent_points : problem_proof :=
by
  sorry

end basketball_total_opponent_points_l184_184068


namespace shape_of_r_eq_c_in_cylindrical_coords_l184_184203

variable {c : ℝ}

theorem shape_of_r_eq_c_in_cylindrical_coords (h : c > 0) :
  ∀ (r θ z : ℝ), (r = c) ↔ ∃ (cylinder : ℝ), cylinder = r ∧ cylinder = c :=
by
  sorry

end shape_of_r_eq_c_in_cylindrical_coords_l184_184203


namespace kennedy_miles_home_l184_184338

variable (miles_per_gallon gallons : ℕ)
variable (miles_school miles_softball miles_burger miles_friend : ℕ)

theorem kennedy_miles_home :
  miles_per_gallon = 19 ∧
  gallons = 2 ∧
  miles_school = 15 ∧
  miles_softball = 6 ∧
  miles_burger = 2 ∧
  miles_friend = 4 →
  let total_miles_possible := miles_per_gallon * gallons
  let miles_before_home := miles_school + miles_softball + miles_burger + miles_friend
  let miles_home := total_miles_possible - miles_before_home
  miles_home = 11 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  let total_miles_possible := h1 * h2
  have h11: total_miles_possible = 38 := by
    rw [h1, h2]
    exact rfl
  let miles_before_home := h3 + h4 + h5 + h6
  have h12: miles_before_home = 27 := by
    rw [h3, h4, h5, h6]
    exact rfl
  let miles_home := total_miles_possible - miles_before_home
  have h13: miles_home = 11 := by
    rw [h11, h12]
    exact rfl
  sorry

end kennedy_miles_home_l184_184338


namespace odd_function_check_l184_184952

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184952


namespace smallest_sum_B_c_l184_184283

theorem smallest_sum_B_c : 
  ∃ (B c : ℕ), (B ≤ 4) ∧ (B ≥ 1) ∧ (c > 6) ∧ (\(BBBB_5\) = \(44_c\) ↔ 780 * B = 4 * (c + 1)) ∧ B + c = 195 :=
begin
  sorry
end

end smallest_sum_B_c_l184_184283


namespace cos_double_angle_l184_184217

theorem cos_double_angle (α : ℝ) (hα : 0 < α ∧ α < π/2)
  (h : sin (α - π/4) = 1/3) : cos (2 * α) = - (4 * real.sqrt 2) / 9 :=
sorry

end cos_double_angle_l184_184217


namespace total_cost_to_fix_car_l184_184370

theorem total_cost_to_fix_car : 
  let part_cost := 20 
  let number_of_parts := 2 
  let labor_cost_per_minute := 0.5 
  let working_hours := 6 
  let minutes_per_hour := 60 
  let parts_total_cost := number_of_parts * part_cost 
  let total_working_minutes := working_hours * minutes_per_hour 
  let total_labor_cost := total_working_minutes * labor_cost_per_minute 
  let total_cost := parts_total_cost + total_labor_cost 
  total_cost = 220 :=
by
  let part_cost := 20 
  let number_of_parts := 2 
  let labor_cost_per_minute := 0.5 
  let working_hours := 6 
  let minutes_per_hour := 60 
  let parts_total_cost := number_of_parts * part_cost 
  let total_working_minutes := working_hours * minutes_per_hour 
  let total_labor_cost := total_working_minutes * labor_cost_per_minute 
  let total_cost := parts_total_cost + total_labor_cost 
  have parts_cost_calculation : parts_total_cost = 2 * 20 := by rfl
  have labor_time_calculation : total_working_minutes = 6 * 60 := by rfl
  have labor_cost_calculation : total_labor_cost = 360 * 0.5 := by rfl
  have total_cost_calculation : total_cost = 40 + 180 := by rfl
  have final_result : total_cost = 220 := by norm_num
  exact final_result

end total_cost_to_fix_car_l184_184370


namespace trigonometric_translation_l184_184416

theorem trigonometric_translation :
  ∀ (x : ℝ), 3 * sin (2 * x - π / 6) = 3 * sin (2 * (x - π / 12)) :=
by sorry

end trigonometric_translation_l184_184416


namespace travel_ways_l184_184740

theorem travel_ways (buses trains ferries : ℕ) (hb : buses = 5) (ht : trains = 6) (hf : ferries = 2) :
  buses + trains + ferries = 13 :=
by
  -- conditions
  rw [hb, ht, hf]
  -- simplified calculation leads to the answer
  -- 5 + 6 + 2 = 13, the final answer
  sorry

end travel_ways_l184_184740


namespace odd_function_g_l184_184998

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184998


namespace ten_factorial_base_12_zeros_l184_184419

theorem ten_factorial_base_12_zeros :
  ∃ k : ℕ, (10! : ℕ) = k * 12^k ∧ k = 4 :=
sorry

end ten_factorial_base_12_zeros_l184_184419


namespace deposit_amount_l184_184495

theorem deposit_amount (P : ℝ) (h₀ : 0.1 * P + 720 = P) : 0.1 * P = 80 :=
by
  sorry

end deposit_amount_l184_184495


namespace tip_percentage_calculation_l184_184432

theorem tip_percentage_calculation :
  let total_bill := 211
  let per_person_share := 40.44
  let total_paid := 6 * per_person_share
  let tip_amount := total_paid - total_bill
  let tip_percentage := (tip_amount / total_bill) * 100
  tip_percentage ≈ 15 := by
sorry

end tip_percentage_calculation_l184_184432


namespace smallest_m_inequality_l184_184574

theorem smallest_m_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) : 27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l184_184574


namespace min_value_f_l184_184414

noncomputable def f (x : ℝ) : ℝ := (5 - 4 * x + x^2) / (2 - x)

theorem min_value_f : 
  ∃ x : ℝ, x < 2 ∧ (∀ y : ℝ, y < 2 → f(y) ≥ 2) ∧ f(x) = 2 :=
by
  sorry

end min_value_f_l184_184414


namespace greatest_and_next_greatest_values_l184_184580

theorem greatest_and_next_greatest_values :
  let a := real.exp (0.5 * real.log 2)
  let b := 3
  let c := real.exp (0.5 * real.log 2)
  let d := 9
  d > b ∧ b > a ∧ a = c :=
by
  sorry

end greatest_and_next_greatest_values_l184_184580


namespace limit_n_a_n_l184_184121

noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

def a_n (n : ℕ) (h : 0 < n) : ℝ := Nat.iterate L n (17 / n : ℝ)

theorem limit_n_a_n (n : ℕ) (h : 0 < n) : 
  filter.tendsto (λ n : ℕ, n * a_n n (Nat.pos_of_ne_zero (ne_of_gt h))) 
  filter.at_top (filter.lift' (λ _, filter.principal {x : ℝ | x = 34 / 19})) :=
sorry

end limit_n_a_n_l184_184121


namespace odd_function_check_l184_184955

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184955


namespace cottonwood_fiber_diameter_in_scientific_notation_l184_184190

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l184_184190


namespace faulty_sum_conditions_l184_184019

theorem faulty_sum_conditions (n : List ℤ) : 
  (∑ i in n, i + i * i = 2021) → false :=
by
  sorry

end faulty_sum_conditions_l184_184019


namespace proposition_3_correct_l184_184589

open Real

def is_obtuse (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

theorem proposition_3_correct (A B C : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : 0 < C) (h₃ : A + B + C = π)
  (h : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) : is_obtuse A B C :=
by
  sorry

end proposition_3_correct_l184_184589


namespace cottonwood_fiber_diameter_in_scientific_notation_l184_184189

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l184_184189


namespace tax_increase_correct_l184_184641

def increase_in_tax (initial_tax_rate new_tax_rate item_cost : ℝ) : ℝ :=
  (new_tax_rate - initial_tax_rate) * item_cost

theorem tax_increase_correct (initial_tax_rate new_tax_rate item_cost : ℝ) :
  initial_tax_rate = 0.07 → new_tax_rate = 0.075 → item_cost = 1000 → increase_in_tax initial_tax_rate new_tax_rate item_cost = 5 :=
by
  intros h1 h2 h3;
  simp [increase_in_tax, h1, h2, h3];
  sorry

end tax_increase_correct_l184_184641


namespace growth_comparison_l184_184767

theorem growth_comparison (x : ℝ) (h : ℝ) (hx : x > 0) : 
  (0 < x ∧ x < 1 / 2 → (x + h) - x > ((x + h)^2 - x^2)) ∧
  (x > 1 / 2 → ((x + h)^2 - x^2) > (x + h) - x) :=
by
  sorry

end growth_comparison_l184_184767


namespace tan_subtraction_modified_l184_184284

theorem tan_subtraction_modified (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = (3 : ℝ) / (157465 : ℝ) := by
  have h3 : Real.tan (α - β) = (Real.tan α - Real.tan β) / (1 + (Real.tan α * Real.tan β)^3) :=
    sorry -- this is assumed as given in the conditions
  sorry -- rest of the proof

end tan_subtraction_modified_l184_184284


namespace cistern_wet_surface_area_l184_184040

noncomputable def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  let bottom_surface_area := length * width
  let longer_side_area := 2 * (depth * length)
  let shorter_side_area := 2 * (depth * width)
  bottom_surface_area + longer_side_area + shorter_side_area

theorem cistern_wet_surface_area :
  total_wet_surface_area 9 4 1.25 = 68.5 :=
by
  sorry

end cistern_wet_surface_area_l184_184040


namespace dino_finances_l184_184863

def earnings_per_gig (hours: ℕ) (rate: ℕ) : ℕ := hours * rate

def dino_total_income : ℕ :=
  earnings_per_gig 20 10 + -- Earnings from the first gig
  earnings_per_gig 30 20 + -- Earnings from the second gig
  earnings_per_gig 5 40    -- Earnings from the third gig

def dino_expenses : ℕ := 500

def dino_net_income : ℕ :=
  dino_total_income - dino_expenses

theorem dino_finances : 
  dino_net_income = 500 :=
by
  -- Here, the actual proof would be constructed.
  sorry

end dino_finances_l184_184863


namespace find_range_of_k_l184_184255

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l184_184255


namespace value_of_n_l184_184031

theorem value_of_n (n : ℤ) :
  (10:ℝ)^n = 10^(-6) * sqrt(10^(50) / 0.0001) ↔ n = 21 :=
by
  conv_rhs { rw [← real.rpow_nat_cast, ← real.rpow_nat_cast, real.sqrt_eq_rpow] }
  have h : sqrt (10^54) = 10^27 := by sorry
  have h2 : 10^(-6) * 10^27 = 10^21 := by sorry
  sorry

end value_of_n_l184_184031


namespace sheep_count_l184_184052

theorem sheep_count (S H : ℕ) (h1 : S / H = 2 / 7) (h2 : H * 230 = 12880) : S = 16 :=
by 
  -- Lean proof goes here
  sorry

end sheep_count_l184_184052


namespace range_of_x_statement_l184_184282

theorem range_of_x_statement (x : ℝ) : 
  ¬ ((x ∈ set.Icc 2 5) ∨ (x < 1 ∨ x > 4)) → (x ∈ set.Ico 1 2) := 
by 
  intro h
  suffices : x ∈ set.Ico 1 2, from this
  sorry

end range_of_x_statement_l184_184282


namespace first_inequality_system_of_inequalities_l184_184402

-- First inequality problem
theorem first_inequality (x : ℝ) : 
  1 - (x - 3) / 6 > x / 3 → x < 3 := 
sorry

-- System of inequalities problem
theorem system_of_inequalities (x : ℝ) : 
  (x + 1 ≥ 3 * (x - 3)) ∧ ((x + 2) / 3 - (x - 1) / 4 > 1) → (1 < x ∧ x ≤ 5) := 
sorry

end first_inequality_system_of_inequalities_l184_184402


namespace max_m_divides_f_l184_184215

noncomputable def f (n : ℕ) : ℤ :=
  (2 * n + 7) * 3^n + 9

theorem max_m_divides_f (m n : ℕ) (h1 : n > 0) (h2 : ∀ n : ℕ, n > 0 → m ∣ ((2 * n + 7) * 3^n + 9)) : m = 36 :=
sorry

end max_m_divides_f_l184_184215


namespace total_population_proof_l184_184813

-- Definitions
def total_population (P : ℕ) : Prop :=
  let adults := 0.90 * P
  let adult_women := 0.60 * adults
  adult_women = 21600

-- Hypothesis
theorem total_population_proof : total_population 40000 :=
by
  sorry

end total_population_proof_l184_184813


namespace circumcenter_on_euler_line_l184_184681

theorem circumcenter_on_euler_line {A B C K_a L_a M_a X_a X_b X_c : Type*} 
  (h1 : scalene_triangle ABC)
  (h2 : K_a = angle_bisector_intersection A B C)
  (h3 : L_a = external_angle_bisector_intersection A B C)
  (h4 : M_a = midpoint B C)
  (h5 : circumcircle_intersects_median A K_a L_a M_a X_a)
  (h6 : analogously_construct X_b B K_b L_b M_b)
  (h7 : analogously_construct X_c C K_c L_c M_c) :
  euler_line_circumcenter_intersection ABC X_a X_b X_c :=
sorry

end circumcenter_on_euler_line_l184_184681


namespace floor_abs_neg_45_7_l184_184181

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184181


namespace deliver_all_cargo_l184_184538

theorem deliver_all_cargo (containers : ℕ) (cargo_mass : ℝ) (ships : ℕ) (ship_capacity : ℝ)
  (h1 : containers ≥ 35)
  (h2 : cargo_mass = 18)
  (h3 : ships = 7)
  (h4 : ship_capacity = 3)
  (h5 : ∀ t, (0 < t) → (t ≤ containers) → (t = 35)) :
  (ships * ship_capacity) ≥ cargo_mass :=
by
  sorry

end deliver_all_cargo_l184_184538


namespace travel_time_correct_l184_184465

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l184_184465


namespace value_of_f_neg2_l184_184029

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l184_184029


namespace real_roots_probability_l184_184214

theorem real_roots_probability :
  (∑ a in ({1, 2}: set ℤ), ∑ b in ({-2, -1, 0, 1, 2}: set ℤ), if (a ^ 2 - 4 * b) ≥ 0 then 1 else 0) / 
  (card {1, 2} * card {-2, -1, 0, 1, 2}) = 7 / 10 :=
sorry

end real_roots_probability_l184_184214


namespace probability_three_winning_streaks_l184_184130

theorem probability_three_winning_streaks (wins losses : ℕ) (h_wins : wins = 10) (h_losses : losses = 6) :
  (∃ (p : ℚ), p = 45 / 286 ∧ ∀ (w1 w2 w3 l0 l1 l2 l3 : ℕ),
  w1 + w2 + w3 = wins ∧
  l0 + l1 + l2 + l3 = losses ∧
  l1 > 0 ∧ l2 > 0 →
  ∃ (total : ℕ),
    total = (Combinatorics.choose (wins + losses) losses) ∧
    ∃ (num_wins : ℕ) (num_losses : ℕ),
      num_wins = Combinatorics.choose (wins - 1) 2 ∧
      num_losses = Combinatorics.choose (losses - 1) 3 ∧
      p = (num_wins * num_losses) / total) :=
sorry

end probability_three_winning_streaks_l184_184130


namespace age_ratio_in_years_l184_184009

variable (s d x : ℕ)

theorem age_ratio_in_years (h1 : s - 3 = 2 * (d - 3)) (h2 : s - 7 = 3 * (d - 7)) (hx : (s + x) = 3 * (d + x) / 2) : x = 5 := sorry

end age_ratio_in_years_l184_184009


namespace mixed_groups_count_l184_184003

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l184_184003


namespace find_b_perpendicular_l184_184417

theorem find_b_perpendicular (b : ℝ) : (∀ x y : ℝ, 4 * y - 2 * x = 6 → 5 * y + b * x - 2 = 0 → (1 / 2 : ℝ) * (-(b / 5) : ℝ) = -1) → b = 10 :=
by
  intro h
  sorry

end find_b_perpendicular_l184_184417


namespace smallest_positive_period_of_f_range_of_f_in_interval_l184_184610

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin(2 * x - π / 3)

theorem smallest_positive_period_of_f :
  Function.Periodic f π := sorry

theorem range_of_f_in_interval :
  ∀ x ∈ Icc (π / 4) (π / 2), 1 ≤ f x ∧ f x ≤ 2 := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l184_184610


namespace monthly_interest_payment_l184_184102

def annual_interest (P : ℝ) (R : ℝ) : ℝ := P * R
def monthly_interest (annual : ℝ) : ℝ := annual / 12

theorem monthly_interest_payment
  (P : ℝ) (R : ℝ) (hPR : P = 30400 ∧ R = 0.09) :
  monthly_interest (annual_interest P R) = 228 :=
by
  sorry

end monthly_interest_payment_l184_184102


namespace security_deposit_amount_correct_l184_184665

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end security_deposit_amount_correct_l184_184665


namespace Lewis_found_20_items_l184_184367

-- Define the number of items Tanya found
def Tanya_items : ℕ := 4

-- Define the number of items Samantha found
def Samantha_items : ℕ := 4 * Tanya_items

-- Define the number of items Lewis found
def Lewis_items : ℕ := Samantha_items + 4

-- Theorem to prove the number of items Lewis found
theorem Lewis_found_20_items : Lewis_items = 20 := by
  sorry

end Lewis_found_20_items_l184_184367


namespace remaining_customers_after_some_left_l184_184815

-- Define the initial conditions and question (before proving it)
def initial_customers := 8
def new_customers := 99
def total_customers_after_new := 104

-- Define the hypothesis based on the total customers after new customers added
theorem remaining_customers_after_some_left (x : ℕ) (h : x + new_customers = total_customers_after_new) : x = 5 :=
by {
  -- Proof omitted
  sorry
}

end remaining_customers_after_some_left_l184_184815


namespace ellipse_range_k_l184_184257

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l184_184257


namespace problem_statement_l184_184306

-- Mathematical Definitions
def num_students : ℕ := 6
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 3

def event_A : Prop := ∃ (boyA : ℕ), boyA < num_boys
def event_B : Prop := ∃ (girlB : ℕ), girlB < num_girls

def C (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select 3 out of 6 students
def total_ways : ℕ := C num_students num_selected

-- Probability of event A
def P_A : ℚ := C (num_students - 1) (num_selected - 1) / total_ways

-- Probability of events A and B
def P_AB : ℚ := C (num_students - 2) (num_selected - 2) / total_ways

-- Conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem problem_statement : P_B_given_A = 2 / 5 := sorry

end problem_statement_l184_184306


namespace Will_games_count_l184_184055

theorem Will_games_count (initial_amount spent_per_mower_blade game_cost : ℕ) (h1 : initial_amount = 104)
  (h2 : spent_per_mower_blade = 41) (h3 : game_cost = 9) :
  ((initial_amount - spent_per_mower_blade) / game_cost) = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end Will_games_count_l184_184055


namespace watch_cost_price_l184_184778

noncomputable def cost_price : ℝ := 1166.67

theorem watch_cost_price (CP : ℝ) (loss_percent gain_percent : ℝ) (delta : ℝ) 
  (h1 : loss_percent = 0.10) 
  (h2 : gain_percent = 0.02) 
  (h3 : delta = 140) 
  (h4 : (1 - loss_percent) * CP + delta = (1 + gain_percent) * CP) : 
  CP = cost_price := 
by 
  sorry

end watch_cost_price_l184_184778


namespace value_of_abs_3h_minus_4k_l184_184201

theorem value_of_abs_3h_minus_4k 
  (h k : ℤ)
  (poly : Polynomial ℤ := 3 * Polynomial.X ^ 4 - h * Polynomial.X ^ 2 + Polynomial.C k) 
  (h1 : ∃ p q, poly = (Polynomial.X + 1) * (Polynomial.X - 2) * (Polynomial.X + 3) * p * q) :
  |3 * h - 4 * k| = 3 := 
sorry

end value_of_abs_3h_minus_4k_l184_184201


namespace road_time_l184_184473

theorem road_time (departure : ℕ) (arrival : ℕ) (stops : List ℕ) : 
  departure = 7 ∧ arrival = 20 ∧ stops = [25, 10, 25] → 
  ((arrival - departure) * 60 - stops.sum) / 60 = 12 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 hstops
  have h_duration : (20 - 7) * 60 = 780 := rfl
  have h_stops : stops.sum = 60 := by
    simp [hstops]
  have h_total_time_on_road : (780 - 60) / 60 = 12 := rfl
  exact h_total_time_on_road

end road_time_l184_184473


namespace inequality_proof_l184_184359

theorem inequality_proof (a b : ℝ) (hne : a ≠ 0) : 
  a^2 + b^2 + (1 / a^2) + (b / a) ≥ real.sqrt 3 := 
by 
sory

end inequality_proof_l184_184359


namespace odd_function_g_l184_184989

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184989


namespace point_P_coordinates_l184_184316

theorem point_P_coordinates :
  ∃ (x y : ℝ), (y = (x^3 - 10 * x + 3)) ∧ (x < 0) ∧ (3 * x^2 - 10 = 2) ∧ (x = -2 ∧ y = 15) := by
sorry

end point_P_coordinates_l184_184316


namespace arrange_people_at_least_one_girl_l184_184209

theorem arrange_people_at_least_one_girl :
  let boys := 4 in
  let girls := 3 in
  let total_people := boys + girls in
  let num_selections := (total_people.choose 3) * (Nat.factorial 3) in
  let all_boys_selections := (boys.choose 3) * (Nat.factorial 3) in
  num_selections - all_boys_selections = 186 := by
  trivial

end arrange_people_at_least_one_girl_l184_184209


namespace MaximMethod_CorrectNumber_l184_184371

theorem MaximMethod_CorrectNumber (x y : ℕ) (N : ℕ) (h_digit_x : 0 ≤ x ∧ x ≤ 9) (h_digit_y : 1 ≤ y ∧ y ≤ 9)
  (h_N : N = 10 * x + y)
  (h_condition : 1 / (10 * x + y : ℚ) = 1 / (x + y : ℚ) - 1 / (x * y : ℚ)) :
  N = 24 :=
sorry

end MaximMethod_CorrectNumber_l184_184371


namespace tangent_line_through_origin_eq_ex_l184_184247

-- Define the conditions of the problem
def curve (x : ℝ) : ℝ := Real.exp x
def point_P : ℝ × ℝ := (1, Real.exp 1)
def tangent_line_at (P : ℝ × ℝ) (k : ℝ) : (ℝ → ℝ) := fun x => k * (x - P.1) + P.2

-- State the problem
theorem tangent_line_through_origin_eq_ex :
  (∀ P : ℝ × ℝ, P = point_P),
  ∃ k : ℝ, (∀ (x : ℝ), tangent_line_at point_P k x = k * x) := 
sorry

end tangent_line_through_origin_eq_ex_l184_184247


namespace new_person_weight_l184_184481

theorem new_person_weight (W : ℝ) (initial_weight : ℝ) (weight_increase : ℝ) : 
  (avg_increase : ℝ) (persons : ℤ) (replace_weight : ℝ):
  persons = 8 → 
  avg_increase = 3.5 →
  replace_weight = 62 →
  W = replace_weight + (persons * avg_increase) →
  W = 90 :=
by intros h1 h2 h3 h4;
   rw [h1, h2, h3] at h4;
   exact h4;
   sorry

end new_person_weight_l184_184481


namespace transformed_function_is_odd_l184_184959

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184959


namespace orange_cost_and_average_l184_184524

/-- Prove the total cost and average cost per orange given specific pricing tiers and quantities -/
theorem orange_cost_and_average (cost_4 : ℕ) (cost_7 : ℕ) (cost_10 : ℕ) (groups : ℕ)
  (h4 : cost_4 = 15)
  (h7 : cost_7 = 25)
  (h10 : cost_10 = 32)
  (hgroups : groups = 3) :
  let total_cost := groups * cost_4 + groups * cost_7 + groups * cost_10,
      total_oranges := groups * 4 + groups * 7 + groups * 10,
      average_cost := total_cost / total_oranges in
  total_cost = 216 ∧ average_cost = 10.29 := by
  sorry

end orange_cost_and_average_l184_184524


namespace total_defective_rate_l184_184556

def defective_rate_x : ℝ := 0.005
def defective_rate_y : ℝ := 0.008
def proportion_x : ℝ := 1 / 3
def proportion_y : ℝ := 2 / 3

theorem total_defective_rate :
  (defective_rate_x * proportion_x + defective_rate_y * proportion_y) = 0.007 := 
by
  sorry

end total_defective_rate_l184_184556


namespace regular_polygon_sides_l184_184515

theorem regular_polygon_sides (P s : ℕ) (hP : P = 108) (hs : s = 12) : 
  ∃ n : ℕ, P = n * s ∧ n = 9 :=
by {
  use 9,
  split,
  { rw [hP, hs], norm_num },
  refl
}

end regular_polygon_sides_l184_184515


namespace odd_function_check_l184_184944

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184944


namespace length_of_each_train_l184_184485

-- Define the speeds of the trains in km/hr
def speed_faster_train := 46
def speed_slower_train := 36

-- Define the relative speed in m/s (conversion factor from km/hr to m/s = 5/18)
def relative_speed_m_per_s : ℝ := (speed_faster_train - speed_slower_train) * (5 / 18)

-- Define the time taken for the faster train to pass the slower train in seconds
def passing_time_seconds : ℝ := 27

-- The total distance covered by the faster train which is twice the length of each train
def total_distance_covered (length_train : ℝ) : ℝ := 2 * length_train

-- The distance formula: distance = speed * time
theorem length_of_each_train :
  ∃ L : ℝ, total_distance_covered L = relative_speed_m_per_s * passing_time_seconds ∧ L = 37.5 :=
by
  sorry  -- Proof goes here

end length_of_each_train_l184_184485


namespace work_fraction_left_l184_184354

theorem work_fraction_left (A_days B_days C_days D_days : ℕ) (hA : A_days = 10) (hB : B_days = 15) (hC : C_days = 20) (hD : D_days = 30) : 
  let workA := 1 / (A_days : ℚ)
  let workB := 1 / (B_days : ℚ)
  let workC := 1 / (C_days : ℚ)
  let workD := 1 / (D_days : ℚ)
  let total_work := workA + workB + workC + workD
  let work_in_5_days := total_work * 5 in
  1 - work_in_5_days = 0 := by sorry

end work_fraction_left_l184_184354


namespace cos_arcsin_eq_l184_184841

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l184_184841


namespace alice_winning_strategy_l184_184454

theorem alice_winning_strategy : 
  ∃ (strategy : list Char → list Char), 
  ∀ (initial_string : list Char) (kolya_move : list Char),
  initial_string = ['Г','О','Р','О','Д','С','К','А','Я','_','У','С','Т','Н','А','Я','_','О','Л','И','М','П','И','А','Д','А'] →
  kolya_move = ['А'] →
  strategy (initial_string.erase 'А') = sorry := 
sorry

end alice_winning_strategy_l184_184454


namespace ratio_BD_DC_BG_GC_ratio_PB_PC_BD_DC_l184_184824

-- Given:
variables (A B C H D E F G O P : ℝ)
variables (h_in_triangle: H ∈ triangle A B C)
variables (D_intersection: D ∈ (line (A, H)) ∧ D ∈ (line (B, C)))
variables (E_intersection: E ∈ (line (B, H)) ∧ E ∈ (line (C, A)))
variables (F_intersection: F ∈ (line (C, H)) ∧ F ∈ (line (A, B)))
variables (G_intersection: G ∈ (line (F, E)) ∧ G ∈ (line (B, C)))
variables (O_midpoint: O = midpoint D G)
variables (P_on_circle: circle (O, distance O D) ∩ line (F, E) = {P})

-- To Prove:
theorem ratio_BD_DC_BG_GC
  (h1: H ∈ triangle A B C)
  (hD : D ∈ (line (A, H)) ∧ D ∈ (line (B, C)))
  (hE : E ∈ (line (B, H)) ∧ E ∈ (line (C, A)))
  (hF : F ∈ (line (C, H)) ∧ F ∈ (line (A, B)))
  (hG : G ∈ (line (F, E)) ∧ G ∈ (line (B, C)))
  (hO : O = midpoint D G) :
  (BD: ℝ) = distance B D ∧
  (DC: ℝ) = distance D C ∧
  (BG: ℝ) = distance B G ∧
  (GC: ℝ) = distance G C →
  BD / DC = BG / GC :=
sorry

theorem ratio_PB_PC_BD_DC
  (h1: H ∈ triangle A B C)
  (hD : D ∈ (line (A, H)) ∧ D ∈ (line (B, C)))
  (hE : E ∈ (line (B, H)) ∧ E ∈ (line (C, A)))
  (hF : F ∈ (line (C, H)) ∧ F ∈ (line (A, B)))
  (hG : G ∈ (line (F, E)) ∧ G ∈ (line (B, C)))
  (hO : O = midpoint D G)
  (hP : P ∈ (circle (O, distance O D)) ∧ P ∈ (line (F, E))) :
  (BD: ℝ) = distance B D ∧
  (DC: ℝ) = distance D C ∧
  (PB: ℝ) = distance P B ∧
  (PC: ℝ) = distance P C →
  PB / PC = BD / DC :=
sorry

end ratio_BD_DC_BG_GC_ratio_PB_PC_BD_DC_l184_184824


namespace value_of_expression_l184_184026

theorem value_of_expression : 4 * (8 - 6) - 7 = 1 := by
  -- Calculation steps would go here
  sorry

end value_of_expression_l184_184026


namespace friendship_configuration_ways_l184_184168

theorem friendship_configuration_ways (individuals : Finset ℕ) 
  (all_have_three_friends : ∀ i ∈ individuals, (Finset.filter (λ j, j ≠ i) individuals).card = 3) :
  individuals.card = 8 → 
  (∃ n : ℕ, n = 4160) :=
by
  intro h_card
  use 4160
  sorry

end friendship_configuration_ways_l184_184168


namespace total_number_of_trees_l184_184741

theorem total_number_of_trees (D P : ℕ) (cost_D cost_P total_cost : ℕ)
  (hD : D = 350)
  (h_cost_D : cost_D = 300)
  (h_cost_P : cost_P = 225)
  (h_total_cost : total_cost = 217500)
  (h_cost_equation : cost_D * D + cost_P * P = total_cost) :
  D + P = 850 :=
by
  rw [hD, h_cost_D, h_cost_P, h_total_cost] at h_cost_equation
  sorry

end total_number_of_trees_l184_184741


namespace convert_degrees_to_radians_l184_184548

theorem convert_degrees_to_radians (θ : ℝ) (h : θ = -630) : θ * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end convert_degrees_to_radians_l184_184548


namespace remainder_ab3_mod_n_eq_b_l184_184677

theorem remainder_ab3_mod_n_eq_b (n : ℕ) (a b : ℕ) (ha : Nat.gcd a n = 1) (hb : Nat.gcd b n = 1) (h : a ≡ Nat.invMod b n ^ 2 [MOD n]) :
  (a * b ^ 3) % n = b % n :=
by
  sorry

end remainder_ab3_mod_n_eq_b_l184_184677


namespace nonzero_number_pow_zero_neg_half_pow_zero_equals_one_l184_184057

theorem nonzero_number_pow_zero {a : ℝ} (h : a ≠ 0) : a^0 = 1 := by
  -- This is the theorem that any non-zero real number raised to the power of 0 equals 1.
  sorry

lemma neg_half_pow_zero : (-1 / 2 : ℝ) ≠ 0 := by
  -- Prove that -1 / 2 is a non-zero number.
  norm_num

theorem neg_half_pow_zero_equals_one : (- (1 / 2) : ℝ)^0 = 1 := by
  -- Using the nonzero_number_pow_zero theorem and the lemma that -1 / 2 is non-zero.
  apply nonzero_number_pow_zero
  exact neg_half_pow_zero

end nonzero_number_pow_zero_neg_half_pow_zero_equals_one_l184_184057


namespace minimum_students_l184_184654

theorem minimum_students (P S : ℕ) (H1 : P = 6) (H2: ∀ i : Fin P, ∃ unique (s : Finset (Fin S)), s.card = 1000 ∧ i ∈ s) (H3 : ∀ (a b : Fin S), a ≠ b → ¬ (∀ i : Fin P, a ∈ (H2 i).1 ∧ b ∈ (H2 i).1)) : 
  S ≥ 2000 := 
sorry

end minimum_students_l184_184654


namespace taxi_fare_cost_l184_184302

theorem taxi_fare_cost (P : ℝ) (h : P > 7):
  let base_fare := 5 
  let additional_fare_rate := 1.5
  let total_cost := base_fare + additional_fare_rate * (P - 7)
  total_cost = 1.5 * P - 5.5 :=
by
  let base_fare := 5
  let additional_fare_rate := 1.5
  let total_cost := base_fare + additional_fare_rate * (P - 7)
  have h1 : total_cost = 5 + 1.5 * (P - 7) := by rfl
  rw [h1]
  linarith

end taxi_fare_cost_l184_184302


namespace distance_from_shangri_la_to_atlantis_is_2100_l184_184077

def distance_from_shangri_la_to_atlantis : ℝ :=
  let shangrila : ℂ := 1260 + 1680 * complex.I
  let atlantis : ℂ := 0
  complex.abs (shangrila - atlantis)

theorem distance_from_shangri_la_to_atlantis_is_2100 :
  distance_from_shangri_la_to_atlantis = 2100 :=
by
  sorry

end distance_from_shangri_la_to_atlantis_is_2100_l184_184077


namespace find_range_and_real_k_l184_184264

theorem find_range_and_real_k (k : ℝ) 
  (h : ∃ (x₁ x₂ : ℝ), x² - ↑4 * x + k + 1 = 0)
  : (k ≤ 3) ∧ (∀ (x₁ x₂ : ℝ), x₁ + x₂ = 4 ∧ x₁ * x₂ = k + 1 → (3 / x₁ + 3 / x₂ = x₁ * x₂ - 4 → k = -3)) :=
by
  sorry

end find_range_and_real_k_l184_184264


namespace pentagon_area_l184_184866

theorem pentagon_area (ABCDE: Type) [ConvexPentagon ABCDE] 
  (h_diag_areas : ∀ (diag : Diagonal ABCDE), DiagonalTriangleArea diag = 1) :
  PentagonArea ABCDE = (5 + Real.sqrt 5) / 2 :=
sorry

end pentagon_area_l184_184866


namespace sum_of_consecutive_naturals_l184_184768

theorem sum_of_consecutive_naturals (n : ℕ) : 
  (∃ k a : ℕ, k ≥ 3 ∧ (k % 2 = 1) ∧ n = k * a + (k * (k - 1)) / 2) ↔ n ≥ 6 :=
by 
  split
  {
    intro h,
    obtain ⟨k, a, hk, odd_k, hn⟩ := h,
    sorry
  }
  {
    intro h,
    obtain ⟨k, a, ha⟩ : ∃ k a, k = 3 ∧ n = k * a + (k * (k - 1)) / 2 := sorry,
    exact ⟨k, a, ha⟩
  }

end sum_of_consecutive_naturals_l184_184768


namespace cos_arcsin_l184_184832

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l184_184832


namespace b_investment_l184_184091

theorem b_investment (x : ℝ) (total_profit A_investment B_investment C_investment A_profit: ℝ)
  (h1 : A_investment = 6300)
  (h2 : B_investment = x)
  (h3 : C_investment = 10500)
  (h4 : total_profit = 12600)
  (h5 : A_profit = 3780)
  (ratio_eq : (A_investment / (A_investment + B_investment + C_investment)) = (A_profit / total_profit)) :
  B_investment = 13700 :=
  sorry

end b_investment_l184_184091


namespace odd_function_g_l184_184996

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184996


namespace evaluate_floor_abs_neg_l184_184169

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184169


namespace oranges_in_bin_after_changes_l184_184523

-- Define the initial number of oranges
def initial_oranges : ℕ := 34

-- Define the number of oranges thrown away
def oranges_thrown_away : ℕ := 20

-- Define the number of new oranges added
def new_oranges_added : ℕ := 13

-- Theorem statement to prove the final number of oranges in the bin
theorem oranges_in_bin_after_changes :
  initial_oranges - oranges_thrown_away + new_oranges_added = 27 := by
  sorry

end oranges_in_bin_after_changes_l184_184523


namespace min_value_of_expression_l184_184236

-- Define the conditions in the problem
def conditions (m n : ℝ) : Prop :=
  (2 * m + n = 2) ∧ (m > 0) ∧ (n > 0)

-- Define the problem statement
theorem min_value_of_expression (m n : ℝ) (h : conditions m n) : 
  (∀ m n, conditions m n → (1 / m + 2 / n) ≥ 4) :=
by 
  sorry

end min_value_of_expression_l184_184236


namespace area_annulus_l184_184501

theorem area_annulus (R a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b) (h_ab_pos : 0 < a + b) :
  let r := (R^2 - a * b) ^ (1/2) in -- radius of the circle traced by point C
  π * R^2 - π * r^2 = π * a * b :=
by
  let r := (R^2 - a * b) ^ (1/2)
  have : π * R^2 - π * r^2 = π * a * b
  sorry

end area_annulus_l184_184501


namespace LitterPatrol_pickup_l184_184404

theorem LitterPatrol_pickup :
  ∃ n : ℕ, n = 10 + 8 :=
sorry

end LitterPatrol_pickup_l184_184404


namespace evaluate_floor_abs_neg_l184_184172

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184172


namespace transformed_function_is_odd_l184_184972

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184972


namespace sequence_property_l184_184365

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range (n + 1)).sum a

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, Sn (n + 1) a = 2 * a n + 1) : a 3 = 2 :=
sorry

end sequence_property_l184_184365


namespace security_deposit_amount_l184_184666

-- Definitions of the given conditions
def daily_rate : ℝ := 125.00
def pet_fee : ℝ := 100.00
def service_cleaning_fee_percentage : ℝ := 0.20
def duration_in_days : ℝ := 14 -- 2 weeks
def security_deposit_percentage : ℝ := 0.50

-- Summarize the problem into a theorem
theorem security_deposit_amount :
  let total_rent := duration_in_days * daily_rate in
  let total_with_pet_fee := total_rent + pet_fee in
  let service_cleaning_fee := service_cleaning_fee_percentage * total_with_pet_fee in
  let total_with_service_fee := total_with_pet_fee + service_cleaning_fee in
  let security_deposit := security_deposit_percentage * total_with_service_fee in
  security_deposit = 1110.00 :=
by
  sorry

end security_deposit_amount_l184_184666


namespace locations_entered_summer_l184_184730

theorem locations_entered_summer:
  (∀ (tempsA : Fin 5 → ℝ), 
      (tempsA.sorted.get 2 == 24) ∧ 
      (tempsA.mode == 22) → 
      (∀ i, tempsA i ≥ 22))
  ∧ 
  (∀ (tempsD : Fin 5 → ℝ), 
      (32 ∈ tempsD) ∧ 
      (tempsD.mean == 26) ∧ 
      (tempsD.variance == 10.8) → 
      (∀ i, tempsD i ≥ 22)) :=
by {
  sorry -- Placeholder for the proof
}

end locations_entered_summer_l184_184730


namespace part1_general_formula_part2_sum_formula_l184_184251

noncomputable def geometric_sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a 2 + a 3 + a 4 = 39 ∧ a 5 = 2 * a 4 + 3 * a 3

theorem part1_general_formula (a : ℕ → ℝ) (h : geometric_sequence_condition a) :
  ∀ n, a n = 3 ^ (n - 1) :=
sorry

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n + a n

theorem part2_sum_formula (a : ℕ → ℝ) (h : geometric_sequence_condition a) (h1 : ∀ n, a n = 3 ^ (n - 1)) :
  ∀ n, ∑ i in Finset.range n, b a (i + 1) = (3^n + n^2 + n - 1) / 2 :=
sorry

end part1_general_formula_part2_sum_formula_l184_184251


namespace max_f_sum_l184_184243

noncomputable def f : ℝ → ℝ := sorry

theorem max_f_sum {
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, f(x+1) = 1/2 + real.sqrt(f(x) - f(x)^2)) →
  (∃ y : ℝ, f(0) + f(2017) = y ∧ y = 1 + real.sqrt(2)/2) }:
  sorry

end max_f_sum_l184_184243


namespace covering_count_formula_l184_184344

def covering_count (n : ℕ) : ℕ := {
  -- Covering count is calculated by this sum
  C_n := ∑ j in finset.range (n + 1), (-1) ^ (n - j) * (n.choose j) * 2 ^ (2 ^ j - 1)
}

theorem covering_count_formula (n : ℕ) : covering_count n = ∑ j in finset.range (n + 1), (-1) ^ (n - j) * (n.choose j) * 2 ^ (2 ^ j - 1) :=
sorry

end covering_count_formula_l184_184344


namespace ln_gt_ln_implies_e_gt_e_e_gt_e_does_not_implies_ln_gt_ln_l184_184124

theorem ln_gt_ln_implies_e_gt_e (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ln a > ln b) → (exp a > exp b) :=
by
  sorry

theorem e_gt_e_does_not_implies_ln_gt_ln (a b : ℝ) :
  ¬ (exp a > exp b → ln a > ln b) :=
by
  sorry

end ln_gt_ln_implies_e_gt_e_e_gt_e_does_not_implies_ln_gt_ln_l184_184124


namespace computer_literate_female_employees_l184_184047

theorem computer_literate_female_employees 
    (E : ℕ) (hE : E = 1100)
    (perc_female : ℚ) (h_perc_female : perc_female = 0.60)
    (perc_male : ℚ) (h_perc_male : perc_male = 0.40)
    (perc_c_literate_employees : ℚ) (h_perc_c_literate_employees : perc_c_literate_employees = 0.62)
    (perc_c_literate_males : ℚ) (h_perc_c_literate_males : perc_c_literate_males = 0.50) :
    let F := perc_female * E,
        M := perc_male * E,
        C := perc_c_literate_employees * E,
        CM := perc_c_literate_males * M,
        CF := C - CM
    in CF = 462 := by 
  -- Definitions of all the required variables to reflect the problem conditions
  let F := perc_female * E
  let M := perc_male * E
  let C := perc_c_literate_employees * E
  let CM := perc_c_literate_males * M
  let CF := C - CM
  -- Proof omitted
  sorry

end computer_literate_female_employees_l184_184047


namespace total_dots_not_visible_l184_184744

theorem total_dots_not_visible
  (die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6})
  (num_dice : ℕ := 3)
  (visible_faces : Finset ℕ := {6, 5, 3, 1, 4, 2, 1}) :
  let total_dots := num_dice * die_faces.sum id,
      visible_dots := visible_faces.sum id
  in total_dots - visible_dots = 41 :=
by
  sorry

end total_dots_not_visible_l184_184744


namespace fraction_of_track_in_forest_l184_184016

theorem fraction_of_track_in_forest (n : ℕ) (l : ℝ) (A B C : ℝ) :
  (∃ x, x = 2*l/3 ∨ x = l/3) → (∃ f, 0 < f ∧ f ≤ 1 ∧ (f = 2/3 ∨ f = 1/3)) :=
by
  -- sorry, the proof will go here
  sorry

end fraction_of_track_in_forest_l184_184016


namespace sum_of_extremes_F_l184_184244

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

theorem sum_of_extremes_F (f : ℝ → ℝ) (a : ℝ) (h_odd : is_odd_function f) (h_pos : 0 < a) :
  let F := λ x, f x + 1 in 
  (F a + F (-a)) = 2 := 
by 
  sorry

end sum_of_extremes_F_l184_184244


namespace correct_choice_is_D_l184_184036

variables (A B : Type) [Set A] [Set B]

/-- Define the type of statements that can be either Input or Print --/
inductive Statement
| INPUT : A → Statement
| PRINT : B → Statement

/-- Define the given options as individual statements --/
def optionA : Statement := Statement.INPUT A
def optionB : Statement := Statement.INPUT (B = 3)
def optionC : Statement := Statement.PRINT (2 * x + 1)
def optionD : Statement := Statement.PRINT (4 * x)

-- Prove that option D is the correct choice based on the conditions described.
theorem correct_choice_is_D : (optionA ≠ Statement.PRINT A) ∧ (optionB ≠ Statement.PRINT B) ∧ (optionC ≠ Statement.INPUT C) ∧ (optionD = Statement.PRINT (4 * x)) :=
by {
  sorry -- Detailed proof will be written here.
}

end correct_choice_is_D_l184_184036


namespace odd_function_check_l184_184943

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184943


namespace factorization_problem_l184_184770

-- Expressions A, B, C, D
def A (x : ℝ) : Prop := x^2 - 9 + 8 * x = (x + 3) * (x - 3) + 8 * x
def B (x : ℝ) : Prop := (x + 3) * (x - 3) + 8 * x = x^2 - 9 + 8 * x
def C (a b : ℝ) : Prop := (a + b) * (a - b) = a^2 - b^2
def D (a b c : ℝ) : Prop := a^2 - 2*a*(b-c) - 3*(b-c)^2 = (a - 3*b + 3*c) * (a + b)

-- Proof problem
theorem factorization_problem : ∀ (a b c : ℝ), 
  transforms_to_product (a^2 - 2*a*(b-c) - 3*(b-c)^2) ((a - 3*b + 3*c)*(a + b)) := 
sorry

end factorization_problem_l184_184770


namespace part1_part2_part3_l184_184900

noncomputable def point (x y: ℝ) := (x, y)

def A := point 2 3
def B := point (-2) 0
def C := point 2 0

def circumcircle (x y: ℝ) : Prop :=
  x^2 + (y - 3/2)^2 = 25/4

def internal_angle_bisector_intersection_y_axis := point 0 (2/3)

def slopes_ratio (k1 k2: ℝ) : Prop :=
  k1 / k2 = -7/5

def circle_passing_through (P D C : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃a b c, x^2 + y^2 + a * x + b * y + c = 0 ∧
  (P.1^2 + P.2^2 + a * P.1 + b * P.2 + c = 0) ∧
  (D.1^2 + D.2^2 + a * D.1 + b * D.2 + c = 0) ∧
  (C.1^2 + C.2^2 + a * C.1 + b * C.2 + c = 0)

theorem part1 :
  ∀ (x y : ℝ), circumcircle x y ↔ x^2 + (y - 3/2)^2 = 25/4 :=
sorry

theorem part2 :
  internal_angle_bisector_intersection_y_axis = point 0 (2/3) :=
sorry

theorem part3 :
  let k1 := 1 in let k2 := -5/7 in
  slopes_ratio k1 k2 →
  let P := point (-3/2) (-1/2) in
  let D := point (-1/2) (1/2) in
  circle_passing_through P D C 0 (2/3) :=
sorry

end part1_part2_part3_l184_184900


namespace shadow_length_to_time_l184_184562

theorem shadow_length_to_time (shadow_length_inches : ℕ) (stretch_rate_feet_per_hour : ℕ) (inches_per_foot : ℕ) 
                              (shadow_start_time : ℕ) :
  shadow_length_inches = 360 → stretch_rate_feet_per_hour = 5 → inches_per_foot = 12 → shadow_start_time = 0 →
  (shadow_length_inches / inches_per_foot) / stretch_rate_feet_per_hour = 6 := by
  intros h1 h2 h3 h4
  sorry

end shadow_length_to_time_l184_184562


namespace odd_function_check_l184_184931

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184931


namespace dwarfs_truthful_count_l184_184166

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184166


namespace largest_nonnegative_real_number_f_l184_184683

def f (n : ℕ) : ℝ :=
  if n % 2 = 0 then 0 else 1 / (2 * n)

theorem largest_nonnegative_real_number_f (n : ℕ) (a : fin n → ℝ)
  (h_sum_int : ∑ i, a i ∈ ℤ) : 
  ∃ i, |a i - 1 / 2| ≥ f n :=
sorry

end largest_nonnegative_real_number_f_l184_184683


namespace range_of_t_l184_184642

-- Define the set of integers that satisfy the inequality
def sol_set (t : ℝ) : Set ℝ := 
  {x : ℝ | |3 * x + t| < 4 ∧ x ∈ Int}

-- Define the specific set of integers that should be the solution set
def expected_sol_set : Set ℝ := {1, 2, 3}

-- Theorem stating the range of t for the given condition
theorem range_of_t : 
  (∀ t : ℝ, sol_set t = expected_sol_set → t ∈ Set.Ioo (-7) (-5)) :=
begin
  sorry
end

end range_of_t_l184_184642


namespace exact_value_l184_184123

noncomputable def sin_squared (θ : ℝ) : ℝ := (Real.sin θ) ^ 2

theorem exact_value :
  sqrt ((2 - sin_squared (Real.pi / 9)) *
        (2 - sin_squared (2 * Real.pi / 9)) *
        (2 - sin_squared (4 * Real.pi / 9))) = 1 :=
sorry

end exact_value_l184_184123


namespace general_formula_arithmetic_sum_first_n_terms_geometric_l184_184593

-- Define the arithmetic sequence with given conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∃ a1 a2, a 1 = a1 ∧ a 2 = a2 ∧ a1 + a2 = 10 ∧ a 4 - a 3 = 2

theorem general_formula_arithmetic :
  ∀ (a : ℕ → ℕ),
    (∃ d, arithmetic_sequence a d) →
    ∀ n, a n = 2 * n + 2 :=
by
  sorry

-- Define the geometric sequence with given conditions
def geometric_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) :=
  b 2 = a 3 ∧ b 3 = a 7

theorem sum_first_n_terms_geometric :
  ∀ (a b : ℕ → ℕ),
    (∃ d, arithmetic_sequence a d) →
    geometric_sequence b a →
    ∀ n, (finset.range n).sum b = 2^(n + 1) :=
by
  sorry

end general_formula_arithmetic_sum_first_n_terms_geometric_l184_184593


namespace tycho_jogging_schedule_count_l184_184755

-- Definition of the conditions
def non_consecutive_shot_schedule (days : Finset ℕ) : Prop :=
  ∀ day ∈ days, ∀ next_day ∈ days, day < next_day → next_day - day > 1

-- Definition stating there are exactly seven valid schedules
theorem tycho_jogging_schedule_count :
  ∃ (S : Finset (Finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ non_consecutive_shot_schedule s) ∧ S.card = 7 := 
sorry

end tycho_jogging_schedule_count_l184_184755


namespace trajectory_equation_line_slope_is_constant_l184_184581

/-- Definitions for points A, B, and the moving point P -/ 
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (2, 0)

/-- The condition that the product of the slopes is -3/4 -/
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := P.2 / (P.1 + 2)
  let k_PB := P.2 / (P.1 - 2)
  k_PA * k_PB = -3 / 4

/-- The trajectory equation as a theorem to be proved -/
theorem trajectory_equation (P : ℝ × ℝ) (h : slope_condition P) : 
  P.2 ≠ 0 ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) := 
sorry

/-- Additional conditions for the line l and points M, N -/ 
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def intersect_conditions (P M N : ℝ × ℝ) (k m : ℝ) : Prop :=
  (M.2 = line_l k m M.1) ∧ (N.2 = line_l k m N.1) ∧ 
  (P ≠ M ∧ P ≠ N) ∧ ((P.1 = 1) ∧ (P.2 = 3 / 2)) ∧ 
  (let k_PM := (M.2 - P.2) / (M.1 - P.1)
  let k_PN := (N.2 - P.2) / (N.1 - P.1)
  k_PM + k_PN = 0)

/-- The theorem to prove that the slope of line l is 1/2 -/
theorem line_slope_is_constant (P M N : ℝ × ℝ) (k m : ℝ) 
  (h1 : slope_condition P) 
  (h2 : intersect_conditions P M N k m) : 
  k = 1 / 2 := 
sorry

end trajectory_equation_line_slope_is_constant_l184_184581


namespace truthful_dwarfs_count_l184_184154

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184154


namespace odd_function_g_l184_184994

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184994


namespace totalBalls_l184_184337

def jungkookBalls : Nat := 3
def yoongiBalls : Nat := 2

theorem totalBalls : jungkookBalls + yoongiBalls = 5 := by
  sorry

end totalBalls_l184_184337


namespace golden_section_third_point_l184_184455

noncomputable def x1 : ℕ := 1000 + Nat.floor (0.618 * 1000)
noncomputable def x2 : ℕ := 1000 + 2000 - x1
noncomputable def x3 : ℕ := 1000 + Nat.floor (0.618 * (x2 - 1000))

theorem golden_section_third_point :
  interval = (1000, 2000) →
  x1 = 1618 →
  x2 = 1382 →
  x3 = 1236 := by
  intro interval_eq x1_eq x2_eq
  exact sorry

end golden_section_third_point_l184_184455


namespace total_peaches_l184_184738

theorem total_peaches (num_baskets num_red num_green : ℕ)
    (h1 : num_baskets = 11)
    (h2 : num_red = 10)
    (h3 : num_green = 18) : (num_red + num_green) * num_baskets = 308 := by
  sorry

end total_peaches_l184_184738


namespace cost_of_tax_free_items_l184_184041

theorem cost_of_tax_free_items (total_paid : ℝ) (sales_tax_paid : ℝ) (tax_rate : ℝ) :
  total_paid = 40 → sales_tax_paid = 1.28 → tax_rate = 0.08 → 
  let pre_tax_cost := sales_tax_paid / tax_rate in
  let cost_of_tax_free_items := total_paid - (pre_tax_cost + sales_tax_paid) in
  cost_of_tax_free_items = 22.72 :=
by {
  intros h_total_paid h_sales_tax_paid h_tax_rate,
  let pre_tax_cost := sales_tax_paid / tax_rate,
  let cost_of_tax_free_items := total_paid - (pre_tax_cost + sales_tax_paid),
  have := h_total_paid,
  have := h_sales_tax_paid,
  have := h_tax_rate,
  sorry
}

end cost_of_tax_free_items_l184_184041


namespace prove_convexity_inequality_prove_concavity_inequality_l184_184245

variables {D : Type*} [LinearOrder D] [TopologicalSpace D] [OpenInterval D] [HasSub D] [HasSmul ℝ D]
variable {f : D → ℝ}
variables {x x₀ : D}
variable [Differentiable ℝ f]
variable [Convex ℝ (D → ℝ)]

noncomputable def convexity_inequality_1 (hconvex : ∀ {a b : D}, a < b → convexOn ℝ f (a .. b))
  (hx : x ∈ D) (hx₀ : x₀ ∈ D) : Prop :=
f x ≤ f' x₀ (x - x₀) + f x₀

noncomputable def concavity_inequality_2 (hconcave : ∀ {a b : D}, a < b → concaveOn ℝ f (a .. b))
  (hx : x ∈ D) (hx₀ : x₀ ∈ D) : Prop :=
f x ≥ f' x₀ (x - x₀) + f x₀

theorem prove_convexity_inequality (hconvex : ∀ {a b : D}, a < b → convexOn ℝ f (a .. b))
  (hx : x ∈ D) (hx₀ : x₀ ∈ D) : convexity_inequality_1 hconvex hx hx₀ := 
sorry

theorem prove_concavity_inequality (hconcave : ∀ {a b : D}, a < b → concaveOn ℝ f (a .. b))
  (hx : x ∈ D) (hx₀ : x₀ ∈ D) : concavity_inequality_2 hconcave hx hx₀ := 
sorry

end prove_convexity_inequality_prove_concavity_inequality_l184_184245


namespace find_m_l184_184913

open_locale big_operators

-- Definition of four non-overlapping points and their vector properties

variables (P A B C : Point)
variables (PA PB PC : Vector) -- These represent vectors from P to A, P to B, and P to C.
variable (m : ℝ)

-- Conditions given in the problem
axiom h1 : PA + PB + PC = 0
axiom h2 : PB - PA + PC - PA = m * -PA

-- The theorem to prove
theorem find_m (P A B C : Point) (PA PB PC : Vector) (m : ℝ)
  (h1 : PA + PB + PC = 0)
  (h2 : PB - PA + PC - PA = m * -PA) :
  m = 3 :=
sorry

end find_m_l184_184913


namespace find_t_l184_184364

noncomputable section

open Classical

-- Define the sequence a
def a : ℕ → ℚ
| 1 => 2 / 5
| (n + 1) => a n + a n ^ 2

-- Define S based on the first 2020 terms of the sequence
def S : ℚ := ∑ i in (Finset.range 2020).map (Finset.range 2020).succ, (1 / (a i + 1))

-- Prove the desired property about t
theorem find_t : ∃ t : ℕ, S ∈ (t:ℚ, t+1:ℚ) ∧ t = 2 :=
by
  have a_pos : ∀ n > 0, 0 < a n
  sorry

  have a_mono : ∀ n > 0, a n < a (n + 1)
  sorry

  have t_bound : 9 / 4 < S ∧ S < 5 / 2
  sorry

  use 2
  split
  -- Prove that S lies in the interval (2, 3)
  exact ⟨ (9 / 4), (5 / 2), t_bound⟩, sorry

  -- Prove t = 2
  rfl

end find_t_l184_184364


namespace mike_pull_ups_per_week_l184_184376

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l184_184376


namespace perpendicular_AP_BD_l184_184653

-- Declare the points A, B, C, D, E, and P as points in a geometric setting.
variables {A B C D E P : Type} [point A] [point B] [point C] [point D] [point E] [point P]

-- Define the angles and intersections as per the given conditions.
variables (h1 : ∠ (B, A, C) = ∠ (C, A, D))
          (h2 : ∠ (C, A, D) = ∠ (D, A, E))
          (h3 : ∠ (A, C, B) = 90°)
          (h4 : ∠ (A, D, C) = 90°)
          (h5 : ∠ (A, E, D) = 90°)
          (h_intersect : intersects (line B D) (line C E) P)

-- State the theorem to prove AP ⊥ BD
theorem perpendicular_AP_BD : perpendicular (line A P) (line B D) :=
by
  sorry

end perpendicular_AP_BD_l184_184653


namespace mixed_groups_count_l184_184000

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l184_184000


namespace dwarfs_truthful_count_l184_184162

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184162


namespace sin_double_angle_l184_184602

theorem sin_double_angle (α : ℝ) (h : sin α + 2 * cos α = 0) : sin (2 * α) = -4 / 5 :=
by
  sorry

end sin_double_angle_l184_184602


namespace mikes_sum_divided_by_lauras_sum_l184_184691

theorem mikes_sum_divided_by_lauras_sum :
  let mikes_sum := (List.range 200).map (λ x, (x+1) * 3).sum,
      lauras_sum := (List.range 200).map (λ x, x+1).sum
  in mikes_sum / lauras_sum = 3 :=
by
  let mikes_sum := (List.range 200).map (λ x, (x+1) * 3).sum
  let lauras_sum := (List.range 200).map (λ x, x+1).sum
  show mikes_sum / lauras_sum = 3
  sorry

end mikes_sum_divided_by_lauras_sum_l184_184691


namespace number_of_sides_l184_184512

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end number_of_sides_l184_184512


namespace odd_function_check_l184_184937

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184937


namespace odd_function_g_l184_184997

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184997


namespace percentage_of_palindromes_with_seven_l184_184763

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def contains_seven (n : ℕ) : Prop :=
  let digits := n.toString in
  '7' ∈ digits.toList

theorem percentage_of_palindromes_with_seven :
  let palindromes := {n | 1000 ≤ n ∧ n < 5000 ∧ is_palindrome n}
  let palindromes_with_seven := {n | contains_seven n ∧ n ∈ palindromes}
  let total_palindromes := (palindromes.toFinset.card : ℕ)
  let total_palindromes_with_seven := (palindromes_with_seven.toFinset.card : ℕ)
  (total_palindromes != 0) →
  (total_palindromes_with_seven * 100) / total_palindromes = 19 :=
by
  sorry

end percentage_of_palindromes_with_seven_l184_184763


namespace evaluate_floor_abs_neg_l184_184171

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184171


namespace sum_first_n_natural_numbers_l184_184430

theorem sum_first_n_natural_numbers (n : ℕ) (h : (n * (n + 1)) / 2 = 1035) : n = 46 :=
sorry

end sum_first_n_natural_numbers_l184_184430


namespace slope_parabola_origin_l184_184273

theorem slope_parabola_origin (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 9) :
  (y / x = sqrt 2) ∨ (y / x = -sqrt 2) :=
by 
  sorry

end slope_parabola_origin_l184_184273


namespace complex_expression_1_complex_expression_2_complex_expression_3_l184_184536

theorem complex_expression_1 : (-2 - 4 * complex.I) - (-2 + complex.I) + (1 + 7 * complex.I) = 1 + 2 * complex.I := 
sorry

theorem complex_expression_2 : (1 + complex.I) * (2 + complex.I) * (3 + complex.I) = 12 * complex.I :=
sorry

theorem complex_expression_3 : (3 + complex.I) / (2 + complex.I) = 1 - (1 / 5) * complex.I :=
sorry

end complex_expression_1_complex_expression_2_complex_expression_3_l184_184536


namespace projection_onto_plane_distance_from_point_to_plane_l184_184573

def vector := ℝ × ℝ × ℝ 

noncomputable def dotProduct (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def scaleVector (c : ℝ) (v : vector) : vector :=
  (c * v.1, c * v.2, c * v.3)

noncomputable def subtractVector (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

noncomputable def norm (v : vector) : ℝ :=
  real.sqrt (dotProduct v v)

noncomputable def projection (v n : vector) : vector :=
  scaleVector (dotProduct v n / dotProduct n n) n

theorem projection_onto_plane :
  let v := (2, 3, 4) in
  let n := (2, 3, -1) in
  let proj_v_n := projection v n in
  let proj_v_plane := subtractVector v proj_v_n in
  proj_v_plane = (10/14, 15/14, 65/14) :=
by
  intros
  rw [← proj_v_n, ← proj_v_plane, subtractVector, projection, scaleVector, dotProduct]
  -- Detailed computation can be filled as proof
  sorry

theorem distance_from_point_to_plane :
  let v := (2, 3, 4) in
  let n := (2, 3, -1) in
  let distance := real.abs (dotProduct v n) / norm n in
  distance = 9 / real.sqrt 14 :=
by
  intros
  rw [← distance, dotProduct, norm]
  -- Detailed computation can be filled as proof
  sorry

end projection_onto_plane_distance_from_point_to_plane_l184_184573


namespace differentiable_and_continuous_g_l184_184680

open Real

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
if x = 0 then fderiv ℝ f 0 0 else f x / x

theorem differentiable_and_continuous_g (f : ℝ → ℝ) 
    (h_diff : differentiable ℝ f)
    (h_fdd : differentiable ℝ (fderiv ℝ f))
    (h_cont : continuous (λ x, fderiv ℝ f 0))
    (h_f0 : f 0 = 0) : 
  differentiable ℝ (λ x, if x = 0 then fderiv ℝ f 0 0 else f x / x) ∧
  continuous (fderiv ℝ (λ x, if x = 0 then fderiv ℝ f 0 0 else f x / x)) :=
sorry

end differentiable_and_continuous_g_l184_184680


namespace probability_of_different_suits_l184_184308

-- Define the problem conditions
def deck_size : ℕ := 52
def number_of_suits : ℕ := 4
def cards_per_suit : ℕ := 13
def cards_drawn : ℕ := 3

-- Define the expected probability as the correct answer
def expected_probability : ℚ := 169 / 425

-- Lean statement to prove the probability
theorem probability_of_different_suits :
  (nat.choose deck_size cards_drawn : ℚ) *
  ((nat.choose number_of_suits cards_drawn) * (cards_per_suit ^ cards_drawn : ℚ)) /
  (nat.choose deck_size cards_drawn) = expected_probability := 
sorry

end probability_of_different_suits_l184_184308


namespace number_of_valid_configurations_eq_5_l184_184848

-- Define the conditions
def is_valid_configuration (pos : ℕ) : Prop :=
  1 ≤ pos ∧ pos ≤ 7 ∧ 
  (pos ≠ 1 ∧ pos ≠ 2) -- Positions 1 and 2 obstruct the folding

-- Main statement to prove
theorem number_of_valid_configurations_eq_5 :
  (finset.filter is_valid_configuration (finset.range 7)).card = 5 :=
begin
  sorry
end

end number_of_valid_configurations_eq_5_l184_184848


namespace number_of_possible_values_f2_times_sum_l184_184357

theorem number_of_possible_values_f2_times_sum 
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * f x * y + y^2) :
  let n := 1 in  -- there is only one possible value for f(2)
  let t := 4 in  -- the sum of possible values for f(2)
  n * t = 4 :=
by {
  sorry
}

end number_of_possible_values_f2_times_sum_l184_184357


namespace triangle_area_bounds_l184_184812

theorem triangle_area_bounds (r : ℝ) (h₁ : parabola_vertex = (0, -4))
  (h₂ : ∀ x, y = x^2 - 4) (h₃ : y = r) :
  16 ≤ (r + 4) * sqrt(r + 4) ∧ (r + 4) * sqrt(r + 4) ≤ 144 ↔ 
  0 ≤ r ∧ r ≤ 20 := 
sorry

end triangle_area_bounds_l184_184812


namespace count_pairs_l184_184280

theorem count_pairs : (∃ p : ℕ × ℕ, 0 < p.1 ∧ 0 < p.2 ∧ p.1 ^ 2 + p.2 < 30) → (finset.univ.filter (λ p : ℕ × ℕ, 0 < p.1 ∧ 0 < p.2 ∧ p.1 ^ 2 + p.2 < 30)).card = 90 := sorry

end count_pairs_l184_184280


namespace tan_subtraction_simplify_l184_184395

theorem tan_subtraction_simplify :
  tan (Real.pi / 12) - tan (5 * Real.pi / 12) = -4 * Real.sqrt 3 := by
sorry

end tan_subtraction_simplify_l184_184395


namespace number_of_truthful_dwarfs_l184_184138

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184138


namespace min_value_expression_l184_184286

theorem min_value_expression 
  (a b c : ℝ) 
  (h_a : 0 < a) 
  (h_b : 0 < b) 
  (h_c : 0 < c) 
  (h_eq : b + c = 1) : 
  ∃ x : ℝ, (∀ a b c, (0 < a) → (0 < b) → (0 < c) → (b + c = 1) → 
    \frac{8 * a * c^2 + a}{b * c} + \frac{32}{a + 1} ≥ 24 ∧
    (\frac{8 * a * c^2 + a}{b * c} + \frac{32}{a + 1} = 24 → x = 24)) :=
sorry

end min_value_expression_l184_184286


namespace factorization_example_l184_184769

theorem factorization_example (x: ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
sorry

end factorization_example_l184_184769


namespace measure_of_theta_l184_184456

theorem measure_of_theta 
  (ACB FEG DCE DEC : ℝ)
  (h1 : ACB = 10)
  (h2 : FEG = 26)
  (h3 : DCE = 14)
  (h4 : DEC = 33) : θ = 11 :=
by
  sorry

end measure_of_theta_l184_184456


namespace find_value_of_c_l184_184479

noncomputable def c : ℝ := (8:ℝ)^3 * (9:ℝ)^3 / 679

theorem find_value_of_c : c ≈ 549.703387 :=
by
  -- The proof will be added here
  sorry

end find_value_of_c_l184_184479


namespace cyclic_sum_inequality_l184_184049

variable {n : ℕ}
variable (a b c : Fin n → ℝ)

theorem cyclic_sum_inequality :
  (∑ cycle in [0, 1, 2], Real.sqrt (∑ i : Fin n, (3 * a i - b i - c i)^2)) ≥ 
  (∑ cycle in [0, 1, 2], Real.sqrt (∑ i : Fin n, (a i)^2)) :=
sorry

end cyclic_sum_inequality_l184_184049


namespace dwarfs_truthful_count_l184_184146

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184146


namespace max_area_of_triangle_ABC_l184_184300

-- Definitions and conditions
variables {A B C : ℝ} {a b c : ℝ} [h1 : a = 2] [h2 : (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C]

-- Statement of the problem
theorem max_area_of_triangle_ABC : (∃ b c, (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C ∧ a = 2 ∧ 
  (let area := 0.5 * b * c * Real.sin A in ∀ b' c', (2 + b') * (Real.sin A - Real.sin B) = (c' - b') * Real.sin C → a = 2 → 0.5 * b' * c' * Real.sin A ≤ area)) := 
sorry

end max_area_of_triangle_ABC_l184_184300


namespace chongqing_exam_problem_l184_184490

variables {word : Type} {article : word → Prop} {phrase : word → Prop}

-- Definitions
def countable_noun : word → Prop := sorry
def vowel_sound : word → Prop := sorry
def fixed_phrase : word → word → Prop := sorry

-- Conditions
axiom h_hour_is_countable : countable_noun "hour"
axiom h_hour_starts_with_vowel_sound : vowel_sound "hour"
axiom h_out_of_the_question_is_impossible : fixed_phrase "out" "the"

-- Theorem stating that choice A is the correct answer
theorem chongqing_exam_problem :
  article "an" ∧ fixed_phrase "out" "the" → "an; the" = "an; the" :=
begin
  sorry
end

end chongqing_exam_problem_l184_184490


namespace opposite_endpoint_of_diameter_l184_184540

theorem opposite_endpoint_of_diameter (P : ℝ × ℝ) (center : ℝ × ℝ) (endpoint : ℝ × ℝ) 
  (h1 : center = (5, -2)) 
  (h2 : endpoint = (1, 5)) : 
  ∃ other_endpoint : ℝ × ℝ, other_endpoint = (9, -9) :=
by 
  use (9, -9)
  sorry

end opposite_endpoint_of_diameter_l184_184540


namespace find_nine_l184_184577

-- A necessary definition to work with natural numbers and cubes.
def splits_into_consecutive_odds (m : ℕ) : Prop :=
  ∃ (a : ℕ), a % 2 = 1 ∧ (m^3 = ∑ i in finset.range m, (a + 2 * i))

theorem find_nine (m : ℕ) (h1 : 1 < m) (h2 : splits_into_consecutive_odds m) (h3 : ∃ a, a % 2 = 1 ∧ a = 73) :
  m = 9 :=
sorry

end find_nine_l184_184577


namespace find_min_palindrome_addition_l184_184025

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string;
  s = s.reverse

theorem find_min_palindrome_addition :
  ∃ n : ℕ, n > 0 ∧ is_palindrome (54321 + n) ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬is_palindrome (54321 + m) :=
begin
  sorry
end

end find_min_palindrome_addition_l184_184025


namespace largest_number_is_A_l184_184099

theorem largest_number_is_A (x y z w: ℕ):
  x = (8 * 9 + 5) → -- 85 in base 9 to decimal
  y = (2 * 6 * 6) → -- 200 in base 6 to decimal
  z = ((6 * 11) + 8) → -- 68 in base 11 to decimal
  w = 70 → -- 70 in base 10 remains 70
  max (max x y) (max z w) = x := -- 77 is the maximum
by
  sorry

end largest_number_is_A_l184_184099


namespace rectangle_area_change_l184_184044

theorem rectangle_area_change (L B : ℝ) :
  let A := L * B
  in let new_length := 1.20 * L
  in let new_breadth := 0.80 * B
  in let new_area := new_length * new_breadth
  in new_area = 0.96 * A :=
by
  sorry

end rectangle_area_change_l184_184044


namespace archer_hits_less_than_8_l184_184527

variables (P10 P9 P8 : ℝ)

-- Conditions
def hitting10_ring := P10 = 0.3
def hitting9_ring := P9 = 0.3
def hitting8_ring := P8 = 0.2

-- Statement to prove
theorem archer_hits_less_than_8 (P10 P9 P8 : ℝ)
  (h10 : hitting10_ring P10)
  (h9 : hitting9_ring P9)
  (h8 : hitting8_ring P8)
  (mutually_exclusive: P10 + P9 + P8 <= 1):
  1 - (P10 + P9 + P8) = 0.2 :=
by
  -- Here goes the proof 
  sorry

end archer_hits_less_than_8_l184_184527


namespace problem_min_value_triangle_l184_184916

theorem problem_min_value_triangle
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_angle_bisectors : ∀ A B C A1 B1 C1 : Point, 
    A1 ∈ segment B C ∧ 
    B1 ∈ segment C A ∧
    C1 ∈ segment A B ∧ 
    angle_bisector A A1 B1 B ∧
    angle_bisector B B1 C1 C ∧
    angle_bisector C C1 A A ∧
    concyclic {B, A1, B1, C1}) :
  (min (frac a (b + c) + frac b (c + a) + frac c (a + b))) = 
  ((1 / 2) * (sqrt 17 - 1))
:= sorry

end problem_min_value_triangle_l184_184916


namespace vertical_line_through_M_P_perpendicular_line_through_M_l184_184884

-- Define the given lines
def line1 (x y : ℝ) := 3 * x + 4 * y - 5 = 0
def line2 (x y : ℝ) := 2 * x - 3 * y + 8 = 0

-- Define intersection point M
def M (x y : ℝ) := line1 x y ∧ line2 x y

-- Define point P
def P := (-1 : ℝ, 0 : ℝ)

-- Define perpendicular line
def perpendicular_line (x y : ℝ) := 2 * x + y + 5 = 0

-- Theorem to prove vertical line x = -1 that passes through M and P
theorem vertical_line_through_M_P :
  ∃ (x y : ℝ), M x y ∧ P.1 = x :=
sorry

-- Theorem to prove x - 2y + 5 = 0 passes through M and is perpendicular
theorem perpendicular_line_through_M :
  ∃ (c : ℝ), (∀ (x y : ℝ), M x y → x - 2 * y + c = 0) ∧ (∀ (x y : ℝ), perpendicular_line x y → 2 * x + y + 5 = 0) := 
sorry

end vertical_line_through_M_P_perpendicular_line_through_M_l184_184884


namespace range_of_quotient_l184_184915

theorem range_of_quotient (x y : ℝ) (h : (x - 1)^2 + y^2 = 1) : 
  ∃ k, k = y / (x + 1) ∧ -sqrt (3) / 3 ≤ k ∧ k ≤ sqrt (3) / 3 :=
by
  sorry

end range_of_quotient_l184_184915


namespace truthful_dwarfs_count_l184_184155

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184155


namespace eval_floor_abs_neg_45_7_l184_184177

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184177


namespace team_e_speed_l184_184749

-- Definitions and conditions
variables (v t : ℝ)
def distance_team_e := 300 = v * t
def distance_team_a := 300 = (v + 5) * (t - 3)

-- The theorem statement: Prove that given the conditions, Team E's speed is 20 mph
theorem team_e_speed (h1 : distance_team_e v t) (h2 : distance_team_a v t) : v = 20 :=
by
  sorry -- proof steps are omitted as requested

end team_e_speed_l184_184749


namespace same_function_l184_184771

noncomputable def domain_eq (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = g x

theorem same_function : domain_eq (λ x, (x^(1/3))^3) (λ x, x) :=
by
  sorry

end same_function_l184_184771


namespace percentage_of_girls_relative_to_boys_l184_184644

theorem percentage_of_girls_relative_to_boys (a : ℕ) (h : ℕ) (hab : h = 0.8 * a) :
  (a : ℚ) / h * 100 = 125 := 
sorry

end percentage_of_girls_relative_to_boys_l184_184644


namespace five_inv_mod_31_l184_184566

theorem five_inv_mod_31 : ∃ x : ℤ, 0 ≤ x ∧ x < 31 ∧ (5 * x ≡ 1 [MOD 31]) :=
by 
  use 25
  split
  norm_num
  split
  norm_num
  norm_num
  exact nat.mod_eq_of_lt (by norm_num)
  norm_num
  norm_num
  sorry

end five_inv_mod_31_l184_184566


namespace transformed_function_is_odd_l184_184979

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184979


namespace packs_of_blue_tshirts_l184_184850

theorem packs_of_blue_tshirts (total_tshirts white_packs white_per_pack blue_per_pack : ℕ) 
  (h_white_packs : white_packs = 3) 
  (h_white_per_pack : white_per_pack = 6) 
  (h_blue_per_pack : blue_per_pack = 4) 
  (h_total_tshirts : total_tshirts = 26) : 
  (total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2 := 
by
  -- Proof omitted
  sorry

end packs_of_blue_tshirts_l184_184850


namespace minimum_value_of_polynomial_l184_184196

def polynomial (x : ℝ) : ℝ := (12 - x) * (10 - x) * (12 + x) * (10 + x)

theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial x = -484 :=
by
  sorry

end minimum_value_of_polynomial_l184_184196


namespace P_eq_Q_l184_184059

open Set

variable {R : Type} [LinearOrderedField R]

-- Definitions and conditions
def f (x : R) : R := sorry -- Define the strictly increasing function
axiom f_increasing : StrictMono f
axiom f_bijective : Bijective f

-- Definition of sets P and Q
def P : Set R := {x | x > f x}
def Q : Set R := {x | x > f (f x)}

-- Proof that P = Q
theorem P_eq_Q : P = Q := by
  sorry

end P_eq_Q_l184_184059


namespace travel_time_correct_l184_184466

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l184_184466


namespace problem_sin_ineq_l184_184241

theorem problem_sin_ineq (k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ 2017) :
  ∃! (n : ℕ), n = 11 ∧ 
    (∑ i in Finset.range k, Real.sin (i + 1) * Real.pi / 180 ≠
     ∏ i in Finset.range k, Real.sin (i + 1) * Real.pi / 180) := sorry

end problem_sin_ineq_l184_184241


namespace polygon_diagonals_l184_184572

theorem polygon_diagonals (n : ℕ) (h : n = 150) : 
  (n * (n - 3)) / 2 = 11025 :=
by 
  rw [h]
  simp
  norm_num
  exact eq.refl 11025

end polygon_diagonals_l184_184572


namespace solution_set_inequality_l184_184927

theorem solution_set_inequality {a b c : ℝ} (h₁ : a < 0)
  (h₂ : ∀ x : ℝ, (a * x^2 + b * x + c <= 0) ↔ (x <= -(1/3) ∨ 2 <= x)) :
  (∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -3 ∨ 1/2 < x)) :=
by
  sorry

end solution_set_inequality_l184_184927


namespace regular_polygon_sides_l184_184514

theorem regular_polygon_sides (P s : ℕ) (hP : P = 108) (hs : s = 12) : 
  ∃ n : ℕ, P = n * s ∧ n = 9 :=
by {
  use 9,
  split,
  { rw [hP, hs], norm_num },
  refl
}

end regular_polygon_sides_l184_184514


namespace dwarfs_truthful_count_l184_184157

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184157


namespace number_of_truthful_dwarfs_is_correct_l184_184133

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184133


namespace length_of_plot_l184_184418

open Real

variable (breadth : ℝ) (length : ℝ)
variable (b : ℝ)

axiom H1 : length = b + 40
axiom H2 : 26.5 * (4 * b + 80) = 5300

theorem length_of_plot : length = 70 :=
by
  -- To prove: The length of the plot is 70 meters.
  exact sorry

end length_of_plot_l184_184418


namespace domain_of_f_l184_184882

open Real

def f (x : ℝ) := (x^3 - 4*x + 3) / (|x - 2| + |x + 2|)

theorem domain_of_f : ∀ x : ℝ, x ∈ set.univ :=
by
  intro x
  sorry

end domain_of_f_l184_184882


namespace evaluate_floor_abs_neg_l184_184174

theorem evaluate_floor_abs_neg (x : ℝ) (h₁ : x = -45.7) : 
  floor (|x|) = 45 :=
by
  sorry

end evaluate_floor_abs_neg_l184_184174


namespace total_items_correct_l184_184825

-- Defining the number of each type of items ordered by Betty
def slippers := 6
def lipstick := 4
def hair_color := 8

-- The total number of items ordered by Betty
def total_items := slippers + lipstick + hair_color

-- The statement asserting that the total number of items is 18
theorem total_items_correct : total_items = 18 := 
by 
  -- sorry allows us to skip the proof
  sorry

end total_items_correct_l184_184825


namespace number_of_truthful_dwarfs_is_correct_l184_184132

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184132


namespace odd_function_check_l184_184934

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184934


namespace xy_solution_l184_184265

noncomputable def x := 450 / 19
noncomputable def y := 500 / 19

theorem xy_solution (h₁ : 0.009 / x = 0.01 / y) (h₂ : x + y = 50) : 
  x = 450 / 19 ∧ y = 500 / 19 := 
by 
  sorry

end xy_solution_l184_184265


namespace probability_same_gender_teachers_l184_184704

/-- 
  Given that School A has 2 male and 1 female teachers,
  and School B has 1 male and 2 female teachers,
  prove the probability that two randomly selected teachers, one from each School,
  are of the same gender is 4/9.
-/
theorem probability_same_gender_teachers :
  let schoolA_M := 2
  let schoolA_F := 1
  let schoolB_M := 1
  let schoolB_F := 2
  let total_possibilities := (schoolA_M + schoolA_F) * (schoolB_M + schoolB_F)
  let favorable_cases := schoolA_M * schoolB_M + schoolA_F * schoolB_F
  favorable_cases.toRat / total_possibilities.toRat = 4 / 9 := 
by {
  have schoolA_possibilities := schoolA_M + schoolA_F
  have schoolB_possibilities := schoolB_M + schoolB_F
  have total := schoolA_possibilities * schoolB_possibilities
  have favorable := schoolA_M * schoolB_M + schoolA_F * schoolB_F
  have prob := favorable.toRat / total.toRat
  exact (show prob = 4 / 9, by norm_num)
}

end probability_same_gender_teachers_l184_184704


namespace security_deposit_amount_l184_184667

-- Definitions of the given conditions
def daily_rate : ℝ := 125.00
def pet_fee : ℝ := 100.00
def service_cleaning_fee_percentage : ℝ := 0.20
def duration_in_days : ℝ := 14 -- 2 weeks
def security_deposit_percentage : ℝ := 0.50

-- Summarize the problem into a theorem
theorem security_deposit_amount :
  let total_rent := duration_in_days * daily_rate in
  let total_with_pet_fee := total_rent + pet_fee in
  let service_cleaning_fee := service_cleaning_fee_percentage * total_with_pet_fee in
  let total_with_service_fee := total_with_pet_fee + service_cleaning_fee in
  let security_deposit := security_deposit_percentage * total_with_service_fee in
  security_deposit = 1110.00 :=
by
  sorry

end security_deposit_amount_l184_184667


namespace complement_intersection_l184_184624

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {2, 4}
def N : Set ℕ := {3, 5}

theorem complement_intersection (hU: U = {1, 2, 3, 4, 5}) (hM: M = {2, 4}) (hN: N = {3, 5}) : 
  (U \ M) ∩ N = {3, 5} := 
by 
  sorry

end complement_intersection_l184_184624


namespace rectangle_cylinders_volume_ratio_l184_184066

theorem rectangle_cylinders_volume_ratio 
  (height1 height2 circumference1 circumference2 : ℝ) 
  (volume1 volume2 : ℝ) 
  (h1 : height1 = 9) 
  (h2 : height2 = 6) 
  (c1 : circumference1 = 6) 
  (c2 : circumference2 = 9)
  (r1 : circumference1 = 2 * real.pi * (circumference1 / (2 * real.pi)))
  (r2 : circumference2 = 2 * real.pi * (circumference2 / (2 * real.pi))) 
  (v1 : volume1 = real.pi * (circumference1 / (2 * real.pi))^2 * height1)
  (v2 : volume2 = real.pi * (circumference2 / (2 * real.pi))^2 * height2) :
  (volume2 / volume1) = 3 / 2 :=
sorry

end rectangle_cylinders_volume_ratio_l184_184066


namespace diamonds_in_F10_l184_184722

-- Definition of triangular number
def triangular (n : Nat) : Nat :=
  n * (n + 1) / 2

-- Recursive definition of diamonds in F_n
def diamonds : Nat → Nat
| 1 => 1
| 2 => 5
| n + 1 => diamonds n + 4 * triangular n

theorem diamonds_in_F10 : diamonds 10 = 365 :=
by
  sorry

end diamonds_in_F10_l184_184722


namespace find_n_satisfies_equation_l184_184200

-- Definition of the problem:
def satisfies_equation (n : ℝ) : Prop := 
  (2 / (n + 1)) + (3 / (n + 1)) + (n / (n + 1)) = 4

-- The statement of the proof problem:
theorem find_n_satisfies_equation : 
  ∃ n : ℝ, satisfies_equation n ∧ n = 1/3 :=
by
  sorry

end find_n_satisfies_equation_l184_184200


namespace truthful_dwarfs_count_l184_184152

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184152


namespace complex_power_l184_184599

-- Definitions based on the conditions given
def imaginary_unit := Complex.i

def a : ℝ := 1 / 2
def b : ℝ := -1 / 2

-- The problem statement in Lean
theorem complex_power (h : (1 : ℂ) / (1 + imaginary_unit) = (a : ℂ) + b * imaginary_unit) : (a^b = Real.sqrt 2) :=
  -- Assumptions and intermediate results
  have h1 : a = 1 / 2 := rfl,
  have h2 : b = -1 / 2 := rfl,
  sorry

end complex_power_l184_184599


namespace minimum_value_of_expr_l184_184903

noncomputable def expr (x : ℝ) : ℝ := x + (1 / (x - 5))

theorem minimum_value_of_expr : ∀ (x : ℝ), x > 5 → expr x ≥ 7 ∧ (expr x = 7 ↔ x = 6) := 
by 
  sorry

end minimum_value_of_expr_l184_184903


namespace diagonal_length_AC_l184_184227

open Real

-- Definitions
def isosceles_trapezoid (A B C D : Point) :=
(A.y = B.y) ∧ (C.y = D.y) ∧ (A.x < B.x) ∧ (D.x < C.x) ∧
(B.x - A.x = 24) ∧ (C.x - D.x = 12) ∧ 
((A.x - D.x) ^ 2 + (A.y - D.y) ^ 2 = 169) ∧ 
((B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = 169) 

-- Theorem statement
theorem diagonal_length_AC (A B C D : Point) :
isosceles_trapezoid A B C D → sqrt ((C.x - A.x) ^ 2 + (C.y - A.y) ^ 2) = sqrt 457 :=
by sorry

end diagonal_length_AC_l184_184227


namespace repeating_decimals_count_l184_184575

theorem repeating_decimals_count :
  let num_repeating_decimals := finset.count (λ n, ¬ (∃ k, n = 9 * k)) (finset.range 20) -- count integers n ∈ {1, 2, ..., 19}, that do not satisfy divisible by 9
  in num_repeating_decimals = 17 :=
begin
  -- Let num_repeating_decimals be the count of integers n from 1 to 19 that are not multiples of 9.
  let num_repeating_decimals := 
    (finset.range 20).filter (λ n, ¬ (∃ k, n = 9 * k)).card,
  -- We need to prove that num_repeating_decimals equals to 17.
  have : num_repeating_decimals = 17 := 
    by {
      sorry, -- skip the actual proof
    },
  exact this
end

end repeating_decimals_count_l184_184575


namespace ellipse_range_k_l184_184262

theorem ellipse_range_k (k : ℝ) : 
  (∃ (x y : ℝ) (hk : \(\frac{x^2}{3+k} + \frac{y^2}{2-k} = 1\)), (3 + k > 0) ∧ (2 - k > 0) ∧ (3+k ≠ 2-k)) ↔ 
  k ∈ set.Ioo (-3) (-1/2) ∪ set.Ioo (-1/2) 2 := 
sorry

end ellipse_range_k_l184_184262


namespace cistern_width_l184_184502

theorem cistern_width (l d A : ℝ) (h_l: l = 5) (h_d: d = 1.25) (h_A: A = 42.5) :
  ∃ w : ℝ, 5 * w + 2 * (1.25 * 5) + 2 * (1.25 * w) = 42.5 ∧ w = 4 :=
by
  use 4
  sorry

end cistern_width_l184_184502


namespace dwarfs_truthful_count_l184_184147

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184147


namespace tangent_line_eq_increasing_interval_maximize_f_l184_184609

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem tangent_line_eq (a : ℝ) :
  (∀ x : ℝ, f 3 x = log x - 3 * x) →
  (∃ m b : ℝ, m = -2 ∧ b = -1 ∧ ∀ x y : ℝ, y = f 3 1 → y + 3 = m * (x - 1)) :=
sorry

theorem increasing_interval (a : ℝ) :
  (∀ x : ℝ, f a x = log x - a * x) →
  (∀ x, x = 2 → 1 / x - a = 0) →
  (∃ b : ℝ, b = 1 / 2 → ∀ x, f' b x > 0 → (0 < x) → (x < 2)) :=
sorry

theorem maximize_f (a : ℝ) :
  (∀ x : ℝ, f a x = log x - a * x) →
  (∃ a' : ℝ, (∀ x : ℝ, a' = exp 1 ∧ f a' x = -2) :=
sorry

end tangent_line_eq_increasing_interval_maximize_f_l184_184609


namespace eval_floor_abs_neg_45_7_l184_184180

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184180


namespace min_distance_l184_184252

open Real

def circle_center (x y : ℝ) : Prop := (x + 3)^2 + (y + 4)^2 = 4
def parabola_f (x y : ℝ) : Prop := y^2 = 8 * x
def parabola_directrix (x : ℝ) : Prop := x = -2
def distance (P1 P2 : ℝ × ℝ) : ℝ := real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem min_distance (P : ℝ × ℝ) (hP : parabola_f P.1 P.2) : 
  ∃ m, let C := (-3, -4) in 
       let F := (2, 0) in 
       let m := abs (P.1 + 2) in
       let d_CF := distance C F in
       m + distance P C = √41 := 
  sorry

end min_distance_l184_184252


namespace find_root_l184_184728

theorem find_root (f : ℝ → ℝ) (hf : Function.LeftInverse (Function.rightInverse f)) 
  (h : f⁻¹' {0} = ({2} : Set ℝ)) : (∃ x : ℝ, f x = 0) :=
by
  use 2
  exact h

end find_root_l184_184728


namespace dwarfs_truthful_count_l184_184144

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184144


namespace sum_three_digit_divisibility_l184_184114

theorem sum_three_digit_divisibility : 
    (∑ (N : ℕ) in finset.filter (λ N, 
      let a := N / 100 in 
      let b := (N % 100) / 10 in 
      let c := N % 10 in 
        (1 ≤ a ∧ a ≤ 9) ∧
        (0 ≤ b ∧ b ≤ 9) ∧
        (0 ≤ c ∧ c ≤ 9) ∧
        ((a^2 + b^2 + c^2) ∣ N) ∧
        ((a * b * c) ∣ N))
    (finset.range 1000)) = 781 :=
by
  sorry

end sum_three_digit_divisibility_l184_184114


namespace consecutive_even_number_difference_l184_184294

theorem consecutive_even_number_difference (x : ℤ) (h : x^2 - (x - 2)^2 = 2012) : x = 504 :=
sorry

end consecutive_even_number_difference_l184_184294


namespace area_of_triangle_XYZ_l184_184736

/-- Defining the sides of the triangle XYZ -/
def triangle_XYZ_sides := (4, 4, 6)

/-- Calculate the semi-perimeter of the triangle -/
def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

/-- Calculate the area using Heron's formula -/
def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The proof problem: The area of the triangle with the given sides is 3√7 square miles -/
theorem area_of_triangle_XYZ :
  heron_area 4 4 6 = 3 * Real.sqrt 7 := sorry

end area_of_triangle_XYZ_l184_184736


namespace find_power_of_7_l184_184198

theorem find_power_of_7 :
  (7^(1/4)) / (7^(1/6)) = 7^(1/12) :=
by
  sorry

end find_power_of_7_l184_184198


namespace sequence_term_l184_184274

def S : Nat → Int
| n => 2^n + 3

def a : Nat → Int
| 1 => 5
| n + 1 => 2^n

theorem sequence_term (n : Nat) : a n = if n = 1 then 5 else 2^(n - 1) := by
  sorry

end sequence_term_l184_184274


namespace Randy_drew_pictures_l184_184385

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end Randy_drew_pictures_l184_184385


namespace necessary_and_sufficient_condition_for_negative_root_l184_184782

theorem necessary_and_sufficient_condition_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 1 = 0) ↔ (a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_for_negative_root_l184_184782


namespace sum_of_first_100_terms_l184_184910

noncomputable def an (n : ℕ) := n

theorem sum_of_first_100_terms :
  (∃ a1 a2 a4 : ℕ, a2 = a1 + 1 ∧ a4 = a1 + 3 ∧ (a1 + 1)^2 = a1 * (a1 + 3))
  → (finset.range 100).sum an = 5050 :=
by
  intros h
  let ⟨a1, a2, a4, ha2, ha4, geo⟩ := h
  sorry

end sum_of_first_100_terms_l184_184910


namespace odd_function_g_l184_184991

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184991


namespace odd_function_check_l184_184941

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184941


namespace max_value_of_function_l184_184889

theorem max_value_of_function : ∃ x, (0 ≤ x ∧ x < 2 * π) ∧ 
                                  (∀ y, (0 ≤ y ∧ y < 2 * π) → 
                                  (sin y - sqrt 3 * cos y ≤ sin x - sqrt 3 * cos x)) ∧ 
                                  x = 5 * π / 6 :=
by
  sorry

end max_value_of_function_l184_184889


namespace count_remaining_integers_l184_184427

def T : Set ℕ := {n | n ∈ Finset.range (60 + 1) ∧ n > 0}

def multiple_of (m n : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (m : ℕ) : ℕ :=
  Finset.card (Finset.filter (multiple_of m) (Finset.range (60 + 1)))

theorem count_remaining_integers :
  count_multiples 4 = 15 ∧ count_multiples 5 = 12 ∧ count_multiples 20 = 3 →
  60 - count_multiples 4 - (count_multiples 5 - count_multiples 20) = 36 :=
by
  intros h
  cases h with mul4 rest
  cases rest with mul5 mul20
  have h1 : count_multiples 4 = 15 := mul4
  have h2 : count_multiples 5 = 12 := mul5
  have h3 : count_multiples 20 = 3 := mul20
  sorry

end count_remaining_integers_l184_184427


namespace mary_starting_weight_l184_184690

def initial_weight (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ) : ℕ :=
  final_weight + (lost_3 - gained_4) + (gained_2 - lost_1) + lost_1

theorem mary_starting_weight :
  ∀ (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ),
  final_weight = 81 →
  lost_1 = 12 →
  gained_2 = 2 * lost_1 →
  lost_3 = 3 * lost_1 →
  gained_4 = lost_1 / 2 →
  initial_weight final_weight lost_1 gained_2 lost_3 gained_4 = 99 :=
by
  intros final_weight lost_1 gained_2 lost_3 gained_4 h_final_weight h_lost_1 h_gained_2 h_lost_3 h_gained_4
  rw [h_final_weight, h_lost_1] at *
  rw [h_gained_2, h_lost_3, h_gained_4]
  unfold initial_weight
  sorry

end mary_starting_weight_l184_184690


namespace number_of_truthful_dwarfs_l184_184142

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184142


namespace find_a_plus_2b_l184_184232

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l184_184232


namespace intersection_complement_equivalence_l184_184275

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equivalence :
  ((U \ M) ∩ N) = {3} := by
  sorry

end intersection_complement_equivalence_l184_184275


namespace sum_of_digits_of_largest_n_l184_184350

open Nat

def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Define the conditions as predicates
def is_single_digit_prime (d : ℕ) : Prop := d ∈ [2, 3, 5, 7]
def is_two_digit_prime (e : ℕ) : Prop := Nat.between 10 e 100 ∧ isPrime e
def is_composite_prime (d e : ℕ) : Prop := isPrime (100 * d + e)

-- Main problem statement
theorem sum_of_digits_of_largest_n :
  ∃ d e, is_single_digit_prime d ∧ is_two_digit_prime e ∧ is_composite_prime d e ∧
  ∑ i in (toDigits 10 (d * e * (100 * d + e)), 0) = 24 :=
sorry

end sum_of_digits_of_largest_n_l184_184350


namespace age_of_son_l184_184042

theorem age_of_son (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 20 := 
sorry

end age_of_son_l184_184042


namespace find_n_l184_184600

-- Given conditions as Lean definitions
variable (n : ℕ)
variable (h_n_pos : n > 0)

-- Definition of the lcm function in Lean
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Conditions
variable (h_lcm_40_n : lcm 40 n = 200)
variable (h_lcm_n_45 : lcm n 45 = 180)

-- The theorem to be proved
theorem find_n : n = 180 :=
by
  sorry

end find_n_l184_184600


namespace evaluate_expression_l184_184871

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l184_184871


namespace sum_f_1_to_100_l184_184415

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- Conditions provided in the problem
axiom cond1 (x : ℝ) : f (1 + x) = f (3 - x)
axiom cond2 (x : ℝ) : f (2 + x) = -f (4 - x)

-- Lean theorem statement for the given problem
theorem sum_f_1_to_100 : (finset.range 100).sum (λ n, f (n + 1)) = 0 :=
by
  sorry

end sum_f_1_to_100_l184_184415


namespace eccentricity_hyperbola_l184_184218

-- Define the hyperbola and relevant points
variables {a b : ℝ} (h_a : a > 0) (h_b : b > 0)
def hyperbola := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Define F and A as the right focus and right vertex
def F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0)
def A : ℝ × ℝ := (a, 0)

-- Define point P and Q
def P : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a)
def Q : ℝ × ℝ := sorry -- needs computations for exact expression

-- Define the vector relationship given in the problem
def AP : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)
def AQ : ℝ × ℝ := (Q.1 - A.1, Q.2 - A.2)
def vector_relation := AP = (2 - real.sqrt 2) • AQ

-- Main theorem stating the eccentricity is sqrt(2)
theorem eccentricity_hyperbola : 
  (hyperbola h_a h_b → F → A → P → Q → vector_relation) → 
  real.sqrt (a^2 + b^2) / a = real.sqrt 2 :=
sorry

end eccentricity_hyperbola_l184_184218


namespace transformed_function_is_odd_l184_184977

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184977


namespace odd_function_check_l184_184945

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184945


namespace points_in_square_l184_184119

theorem points_in_square (points : Fin 5 → ℝ × ℝ) (h₀ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 1)
  (h₁ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ 1) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) < 3 / 4 :=
begin
  sorry -- Proof is omitted as per instructions
end

end points_in_square_l184_184119


namespace ellipse_range_k_l184_184258

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l184_184258


namespace cost_function_graph_is_finite_set_of_distinct_points_l184_184696

theorem cost_function_graph_is_finite_set_of_distinct_points :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 20) → 
    (∃ (C : ℕ → ℕ), C(n) = 25 * n) → 
    ∃ S : set (ℕ × ℕ), 
      (∀ z ∈ S, ∃ n ∈ (finset.range 20).map (λ i, i + 1), z = (n, 25 * n)) ∧ 
      (S = {(n, 25 * n) | 1 ≤ n ∧ n ≤ 20}) :=
by
  sorry

end cost_function_graph_is_finite_set_of_distinct_points_l184_184696


namespace divisor_of_factorial_count_valid_n_l184_184576

theorem divisor_of_factorial (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 30) : 
  has_dvd.dvd (∑ i in finset.range n.succ, i^2) (nat.factorial n) := 
sorry

theorem count_valid_n : 
  (finset.filter (λ n, has_dvd.dvd (∑ i in finset.range n.succ, i^2) (nat.factorial n))
  (finset.range 31)).card = 28 := 
sorry

end divisor_of_factorial_count_valid_n_l184_184576


namespace smallest_whole_number_larger_than_sum_l184_184888

theorem smallest_whole_number_larger_than_sum :
    let sum := 2 + 1 / 2 + 3 + 1 / 3 + 4 + 1 / 4 + 5 + 1 / 5 
    let smallest_whole := 16
    (sum < smallest_whole ∧ smallest_whole - 1 < sum) := 
by
    sorry

end smallest_whole_number_larger_than_sum_l184_184888


namespace solve_for_a_l184_184851

def E (a b c : ℝ) : ℝ := a * b^2 + b * c + c

theorem solve_for_a : (E (-5/8) 3 2 = E (-5/8) 5 3) :=
by
  sorry

end solve_for_a_l184_184851


namespace length_of_bridge_l184_184051

open Real

/-- Given a train of length 180 meters traveling at a speed of 45 km/hr crosses a bridge in 30 seconds,
    prove that the length of the bridge is 195 meters. -/
theorem length_of_bridge (length_of_train : ℝ) (speed_km_per_hr : ℝ) (time_seconds : ℝ) 
  (length_of_train = 180) 
  (speed_km_per_hr = 45) 
  (time_seconds = 30) : 
  let speed_m_per_s := speed_km_per_hr * (1000 / 3600) in
  let distance := speed_m_per_s * time_seconds in
  let length_of_bridge := distance - length_of_train in
  length_of_bridge = 195 :=
by
  sorry

end length_of_bridge_l184_184051


namespace triangle_is_isosceles_l184_184701

-- Definitions based on the conditions
variables {A B C D : Type} 
variables [HasVSub E ℝ A B C : E] [InnerProductSpace ℝ E] -- Importing necessary geometric properties and spaces
variables (AC : line_segment A C)
variable (D : E) -- midpoint
variable (B : E)

structure IsMedian (A B C D : E) where
  midpoint_def : D = midpoint ℝ A C

structure IsAngleBisector (A B C D : E) where
  bisector_def : angle A B D = angle C B D

-- Proof problem: To show the triangle is isosceles
theorem triangle_is_isosceles (hMedian : IsMedian A B C D) (hAngleBisector : IsAngleBisector A B C D) : 
  dist A B = dist B C :=
begin
  sorry
end

end triangle_is_isosceles_l184_184701


namespace final_evaluation_l184_184679

noncomputable def alpha : ℝ := Real.arcsin (real.sqrt 2 / 3)
def f (x : ℝ) : ℝ :=
  if 0 <= x ∧ x <= 4 then x
  else sorry -- Placeholder for the function definition outside the given interval [0, 4]

lemma even_function (x : ℝ) : f(-x) = f(x) :=
sorry -- Given that y = f(x) is an even function

lemma periodic_property (x : ℝ) : f(x + 4) = f(4 - x) := 
sorry -- f(x+4) = f(4-x) for all x in ℝ

def main_expr : ℝ := 2016 + (Real.sin (alpha - 2 * Real.pi)) * (Real.sin (Real.pi + alpha)) - 2 * (Real.cos (-alpha))^2

theorem final_evaluation : f(main_expr) = 5 / 9 :=
sorry

end final_evaluation_l184_184679


namespace unique_solution_of_quadratic_l184_184859

theorem unique_solution_of_quadratic (c : ℝ) (h : c ≠ 0) : 
  (∃! b > 0, ∃ x : ℝ, x^2 + (b + (1 / b) + 1) * x + c = 0) ↔ c = 0.5 :=
by
  sorry -- This is where the actual proof would go

end unique_solution_of_quadratic_l184_184859


namespace cards_from_country_correct_l184_184033

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0
def cards_from_country : ℝ := total_cards - cards_from_home

theorem cards_from_country_correct : cards_from_country = 116.0 := by
  -- proof to be added
  sorry

end cards_from_country_correct_l184_184033


namespace transformed_function_is_odd_l184_184958

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184958


namespace arithmetic_sequence_product_l184_184597

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_product (a_1 d : ℤ) :
  (a_n 4 a_1 d) + (a_n 7 a_1 d) = 2 →
  (a_n 5 a_1 d) * (a_n 6 a_1 d) = -3 →
  a_1 * (a_n 10 a_1 d) = -323 :=
by
  sorry

end arithmetic_sequence_product_l184_184597


namespace tilly_star_count_l184_184011

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l184_184011


namespace jeans_average_speed_l184_184539

-- Define the main variables and constants
variables (d : ℝ)

-- Time calculations for Chantal
def t1 : ℝ := d / 5
def t2 : ℝ := 2 * d / 3
def t3 : ℝ := d / 2
def break_time : ℝ := 1/6

-- Total time it takes for Chantal to meet Jean
def T : ℝ := t1 + t2 + break_time + t3

-- Distance Jean meets Chantal (halfway the total distance of 3*d)
def distance_met : ℝ := 1.5 * d

-- Jean's average speed calculation
def jean_speed : ℝ := distance_met / T

-- Theorem that proves Jean's average speed is 45/46 mph
theorem jeans_average_speed : jean_speed d = 45 / 46 :=
by
  dsimp [jean_speed, T, t1, t2, t3, break_time, distance_met]
  field_simp
  ring
  sorry

end jeans_average_speed_l184_184539


namespace floor_ceiling_sum_l184_184868

theorem floor_ceiling_sum : ⌊1.999⌋ + ⌈3.001⌉ = 5 := by
  sorry

end floor_ceiling_sum_l184_184868


namespace general_formula_T30_sum_l184_184911

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Given conditions for the arithmetic sequence
axiom h1 : a 1 + a 4 + a 7 = -24
axiom h2 : a 2 + a 5 + a 8 = -15
axiom h3 : ∀ n : ℕ, a (n + 1) = a n + d

-- Part 1: Prove the general formula for aₙ
theorem general_formula :
  ∀ n, a n = 3 * n - 20 := by
  sorry

-- Part 2: Prove T₃₀ = 909, where T₃₀ is the sum of the first 30 terms of |aₙ|
theorem T30_sum :
  (∑ i in Finset.range 30, abs (a i)) = 909 := by
  sorry

end general_formula_T30_sum_l184_184911


namespace Randy_drew_pictures_l184_184386

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end Randy_drew_pictures_l184_184386


namespace joe_expense_at_market_l184_184333

theorem joe_expense_at_market :
  let oranges_price := 3 * 4.50
  let juices_price := 7 * 0.50
  let honey_price := 3 * 5
  let plants_price := 2 * 18 * 2
  let discount_oranges_juices := 0.10 * (oranges_price + juices_price)
  let discount_honey := 0.05 * honey_price
  let total_after_discount := (oranges_price + juices_price - discount_oranges_juices) + (honey_price - discount_honey) + plants_price
  let tax := 0.08 * total_after_discount
  let total_cost := total_after_discount + tax
  total_cost = 70.79 := by
  let oranges_price := 3 * 4.50
  let juices_price := 7 * 0.50
  let honey_price := 3 * 5
  let plants_price := 2 * 18 * 2
  let discount_oranges_juices := 0.10 * (oranges_price + juices_price)
  let discount_honey := 0.05 * honey_price
  let total_after_discount := (oranges_price + juices_price - discount_oranges_juices) + (honey_price - discount_honey) + plants_price
  let tax := 0.08 * total_after_discount
  let total_cost := total_after_discount + tax
  have oranges_price_def : oranges_price = 13.50 := rfl
  have juices_price_def : juices_price = 3.50 := rfl
  have honey_price_def : honey_price = 15 := rfl
  have plants_price_def : plants_price = 36 := rfl
  have discount_oranges_juices_def : discount_oranges_juices = 1.70 := rfl
  have discount_honey_def : discount_honey = 0.75 := rfl
  have total_after_discount_def : total_after_discount = 65.55 := by
    rw [oranges_price_def, juices_price_def, discount_oranges_juices_def, honey_price_def, discount_honey_def, plants_price_def]
    norm_num
  have tax_def : tax = 5.244 := by
    rw [total_after_discount_def]
    norm_num
  have total_cost_def : total_cost = 70.794 := by
    rw [total_after_discount_def, tax_def]
    norm_num
  show total_cost = 70.79, by
    rw total_cost_def
    norm_num
  sorry

end joe_expense_at_market_l184_184333


namespace defective_units_shipped_percentage_l184_184048

def defective_units_percentage (d : ℝ) (s : ℝ) : ℝ :=
  (d / 100) * (s / 100) * 100

theorem defective_units_shipped_percentage (d s : ℝ) (hd : d = 7) (hs : s = 5) :
  defective_units_percentage d s = 0.35 := by
  sorry

end defective_units_shipped_percentage_l184_184048


namespace gcd_1995_228_eval_f_at_2_l184_184489

-- Euclidean Algorithm Problem
theorem gcd_1995_228 : Nat.gcd 1995 228 = 57 :=
by
  sorry

-- Horner's Method Problem
def f (x : ℝ) : ℝ := 3 * x ^ 5 + 2 * x ^ 3 - 8 * x + 5

theorem eval_f_at_2 : f 2 = 101 :=
by
  sorry

end gcd_1995_228_eval_f_at_2_l184_184489


namespace five_equal_angles_72_degrees_l184_184202

theorem five_equal_angles_72_degrees
  (five_rays : ℝ)
  (equal_angles : ℝ) 
  (sum_angles : five_rays * equal_angles = 360) :
  equal_angles = 72 :=
by
  sorry

end five_equal_angles_72_degrees_l184_184202


namespace length_first_train_correct_l184_184089

noncomputable def length_first_train 
    (speed_train1_kmph : ℕ := 120)
    (speed_train2_kmph : ℕ := 80)
    (length_train2_m : ℝ := 290.04)
    (time_sec : ℕ := 9) 
    (conversion_factor : ℝ := (5 / 18)) : ℝ :=
  let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let total_distance_m := relative_speed_mps * time_sec
  let length_train1_m := total_distance_m - length_train2_m
  length_train1_m

theorem length_first_train_correct 
    (L1_approx : ℝ := 210) :
    length_first_train = L1_approx :=
  by
  sorry

end length_first_train_correct_l184_184089


namespace angle_BCE_45_l184_184655

variable {ℝ : Type*} [linear_ordered_field ℝ]
variables {A B C D E : ℝ → ℝ} -- variables for points

-- conditions
axiom parallel_AB_DC : ∀ (A B C D : ℝ → ℝ), parallel (line (A,B)) (line (C,D))
axiom angle_ABC_90 : ∀ (A B C : ℝ → ℝ), ∠ (A,B,C) = 90
axiom E_midpoint_AD : ∀ (A D : ℝ → ℝ), midpoint E A D
axiom EC_sqrt13 : ∀ (C E : ℝ → ℝ), (dist C E) = sqrt 13
axiom sum_AB_BC_CD_2sqrt26 : ∀ (A B C D : ℝ → ℝ), (length (A,B) + length (B,C) + length (C,D) = 2 * sqrt 26)

-- proving ∠ BCE = 45°
theorem angle_BCE_45 : 
  (∀ (A D : ℝ → ℝ), midpoint E A D) →
  (∀ (A B C : ℝ → ℝ), ∠ (A,B,C) = 90) →
  (∀ (A B C D : ℝ → ℝ), parallel (line (A,B)) (line (C,D))) →
  (∀ (C E : ℝ → ℝ), (dist C E) = sqrt 13) →
  (∀ (A B C D : ℝ → ℝ), (length (A,B) + length (B,C) + length (C,D) = 2 * sqrt 26)) →
  (∀ (B C E : ℝ → ℝ), ∠ (B,C,E) = 45) :=
by
  intro midpoint_def angle_def parallel_def dist_def sum_def
  sorry

end angle_BCE_45_l184_184655


namespace correct_statements_l184_184037

-- Definitions for conditions.
def A_condition := ∀ {n : ℕ}, n ≥ 2 → n ∈ ℕ → 2 * a_{n-1} = a_n
def B_condition (λ : ℝ) := ∀ (n : ℕ), 2 * n^2 + λ * n < 2 * (n+1)^2 + λ * (n+1)
def C_condition (a : ℕ → ℝ) := (a 2) * (a 10) = 4 ∧ (a 2) + (a 10) = 8
def D_condition (S T : ℕ → ℝ) := (S 5) / (T 7) = 15 / 13

-- The proof problem reframed.
theorem correct_statements :
  (A_condition → True) ∧ (∀ λ, B_condition λ → False) ∧ (C_condition a → False) ∧ (D_condition S T → True) :=
by
  -- Each of these will be proved in detail
  sorry

end correct_statements_l184_184037


namespace range_of_a_plus_b_l184_184362

theorem range_of_a_plus_b {f : ℝ → ℝ} (a b : ℝ) (h_lt : a < b) (h_eq : f a = f b)
  (h_f : ∀ x, f x = if x ≤ 0 then 1 - 2^x else 2^x - 1) :
  a + b < 0 := 
begin
  sorry
end

end range_of_a_plus_b_l184_184362


namespace total_cost_jello_l184_184331

def total_cost_james_spent : Real := 259.20

theorem total_cost_jello 
  (pounds_per_cubic_foot : ℝ := 8)
  (gallons_per_cubic_foot : ℝ := 7.5)
  (tablespoons_per_pound : ℝ := 1.5)
  (cost_red_jello : ℝ := 0.50)
  (cost_blue_jello : ℝ := 0.40)
  (cost_green_jello : ℝ := 0.60)
  (percentage_red_jello : ℝ := 0.60)
  (percentage_blue_jello : ℝ := 0.30)
  (percentage_green_jello : ℝ := 0.10)
  (volume_cubic_feet : ℝ := 6) :
  (volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_red_jello * cost_red_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_blue_jello * cost_blue_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_green_jello * cost_green_jello) = total_cost_james_spent :=
by
  sorry

end total_cost_jello_l184_184331


namespace odd_function_check_l184_184933

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184933


namespace fabric_ratio_wednesday_tuesday_l184_184549

theorem fabric_ratio_wednesday_tuesday :
  let fabric_monday := 20
  let fabric_tuesday := 2 * fabric_monday
  let cost_per_yard := 2
  let total_earnings := 140
  let earnings_monday := fabric_monday * cost_per_yard
  let earnings_tuesday := fabric_tuesday * cost_per_yard
  let earnings_wednesday := total_earnings - (earnings_monday + earnings_tuesday)
  let fabric_wednesday := earnings_wednesday / cost_per_yard
  (fabric_wednesday / fabric_tuesday = 1 / 4) :=
by
  sorry

end fabric_ratio_wednesday_tuesday_l184_184549


namespace solve_log_inequality_l184_184712

theorem solve_log_inequality (x : ℝ) (h_ne_zero : x ≠ 0) (h_ne_pm_one : x ≠ 1 ∧ x ≠ -1) :
  log (x^2) ((4*x - 5) / |x - 2|) ≥ 1/2 ↔ -1 + Real.sqrt 6 ≤ x ∧ x ≤ 5 := by
  sorry

end solve_log_inequality_l184_184712


namespace max_marks_l184_184646

theorem max_marks (marks_secured : ℝ) (percentage : ℝ) (max_marks : ℝ) 
  (h1 : marks_secured = 332) 
  (h2 : percentage = 83) 
  (h3 : percentage = (marks_secured / max_marks) * 100) 
  : max_marks = 400 :=
by
  sorry

end max_marks_l184_184646


namespace total_distance_hopped_l184_184737

def distance_hopped (rate: ℕ) (time: ℕ) : ℕ := rate * time

def spotted_rabbit_distance (time: ℕ) : ℕ :=
  let pattern := [8, 11, 16, 20, 9]
  let full_cycles := time / pattern.length
  let remaining_minutes := time % pattern.length
  let full_cycle_distance := full_cycles * pattern.sum
  let remaining_distance := (List.take remaining_minutes pattern).sum
  full_cycle_distance + remaining_distance

theorem total_distance_hopped :
  distance_hopped 15 12 + distance_hopped 12 12 + distance_hopped 18 12 + distance_hopped 10 12 + spotted_rabbit_distance 12 = 807 :=
by
  sorry

end total_distance_hopped_l184_184737


namespace odd_function_check_l184_184940

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184940


namespace tan_identity_l184_184928

theorem tan_identity (x : ℝ) : (tan x + (1 / tan x)) * cos x^2 = 1 / tan x := by
  sorry

end tan_identity_l184_184928


namespace required_speed_for_remainder_trip_l184_184411

theorem required_speed_for_remainder_trip :
  ∀ (d1 d2 d_total : ℝ) (v1 v_required : ℝ) (t_start t_end t_total : ℝ) (t1 t2 : ℝ),
    d1 = 60 →
    d_total = 150 →
    v1 = 80 →
    t_start = 1 →
    t_end = 3 →
    t_total = t_end - t_start →
    t1 = d1 / v1 →
    t2 = t_total - t1 →
    d2 = d_total - d1 →
    v_required = d2 / t2 →
    v_required = 72 :=
begin
  intros d1 d2 d_total v1 v_required t_start t_end t_total t1 t2,
  assume h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  -- Given conditions
  have h11 : t_total = 2, from h6,
  have h12 : t1 = 0.75, from h7,
  have h13 : t2 = t_total - t1, from h8,
  have h14 : d2 = 90, from h9,
  have h15 : v_required = 72, from h10,
  sorry
end

end required_speed_for_remainder_trip_l184_184411


namespace episodes_remaining_l184_184038

theorem episodes_remaining
  (seasons : ℕ) (episodes_per_season : ℕ) (portion_watched : ℚ)
  (h_seasons : seasons = 18)
  (h_episodes_per_season : episodes_per_season = 25)
  (h_portion_watched : portion_watched = 2 / 5) :
  let total_episodes := seasons * episodes_per_season in
  let watched_episodes := (portion_watched * total_episodes).to_nat in
  let remaining_episodes := total_episodes - watched_episodes in
  remaining_episodes = 270 := 
by
  sorry

end episodes_remaining_l184_184038


namespace range_of_a_l184_184613

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - a * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x2 > x1 → (f x1 a / x2 - f x2 a / x1 < 0)) ↔ a ≤ Real.exp 1 / 2 := sorry

end range_of_a_l184_184613


namespace determine_h_l184_184128

variable {R : Type*} [CommRing R]

def h_poly (x : R) : R := -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2

theorem determine_h (x : R) :
  (8*x^4 - 4*x^2 + 2 + h_poly x = 2*x^3 - 6*x + 4) ->
  h_poly x = -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2 :=
by
  intro h
  sorry

end determine_h_l184_184128


namespace hexagonal_pyramid_dihedral_angle_l184_184410

theorem hexagonal_pyramid_dihedral_angle (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) :
  cos (α / 2) = 2 * cos (β / 2) :=
sorry

end hexagonal_pyramid_dihedral_angle_l184_184410


namespace mary_pizza_order_l184_184298

theorem mary_pizza_order (p e r n : ℕ) (h1 : p = 8) (h2 : e = 7) (h3 : r = 9) :
  n = (r + e) / p → n = 2 :=
by
  sorry

end mary_pizza_order_l184_184298


namespace sum_first_n_terms_l184_184363

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 3 ^ n

def c (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

-- Formal Statement
theorem sum_first_n_terms (n : ℕ) : 
  \sum_{i=1}^{n} c i = \frac{n}{2 * n + 1} := 
by sorry

end sum_first_n_terms_l184_184363


namespace sum_of_first_6033_terms_l184_184429

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms
  (a r : ℝ)  
  (h1 : geometric_series_sum a r 2011 = 200)
  (h2 : geometric_series_sum a r 4022 = 380) :
  geometric_series_sum a r 6033 = 542 := 
sorry

end sum_of_first_6033_terms_l184_184429


namespace odd_function_check_l184_184956

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184956


namespace odd_function_no_real_roots_for_all_a_exists_real_root_l184_184268

noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem odd_function (a : ℝ) : ∀ x : ℝ, f (-x) a = -f x a := 
by sorry

theorem no_real_roots_for_all_a : ∀ a : ℝ, ¬ ∀ x : ℝ, f x a = -x :=
by {
  intros a h,
  have h_eq := h 1,
  sorry
}

theorem exists_real_root : ∃ a : ℝ, ∃ x : ℝ, f x a = -x :=
by {
  use -1,
  use 1,
  sorry
}

end odd_function_no_real_roots_for_all_a_exists_real_root_l184_184268


namespace hyperbola_a_value_l184_184616

theorem hyperbola_a_value (a : ℝ) :
  (∀ x y : ℝ, (x^2 / (a + 3) - y^2 / 3 = 1)) ∧ 
  (∀ e : ℝ, e = 2) → 
  a = -2 :=
by sorry

end hyperbola_a_value_l184_184616


namespace sum_in_range_l184_184584

theorem sum_in_range
  (a : ℕ → ℝ)
  (h_sum : ∑ i in finset.range 10, a i ^ 2 / (a i ^ 2 + 1) = 1) :
  -3 ≤ ∑ i in finset.range 10, a i / (a i ^ 2 + 1) ∧ ∑ i in finset.range 10, a i / (a i ^ 2 + 1) ≤ 3 :=
sorry

end sum_in_range_l184_184584


namespace avg_price_pen_l184_184063

-- Define the given conditions
def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def total_cost_of_items : ℕ := 570
def avg_price_pencil : ℕ := 2

-- Prove that the average price of a pen is $14.00
theorem avg_price_pen :
  let total_cost_pencils := num_pencils * avg_price_pencil in
  let total_cost_pens := total_cost_of_items - total_cost_pencils in
  let avg_price_pens := total_cost_pens / num_pens in
  avg_price_pens = 14 :=
by
  sorry

end avg_price_pen_l184_184063


namespace loan_interest_rate_l184_184482

theorem loan_interest_rate (P SI T R : ℕ) (h1 : P = 900) (h2 : SI = 729) (h3 : T = R) :
  (SI = (P * R * T) / 100) -> R = 9 :=
by
  sorry

end loan_interest_rate_l184_184482


namespace total_cost_correct_l184_184109

-- Definitions of the constants based on given problem conditions
def cost_burger : ℕ := 5
def cost_pack_of_fries : ℕ := 2
def num_packs_of_fries : ℕ := 2
def cost_salad : ℕ := 3 * cost_pack_of_fries

-- The total cost calculation based on the conditions
def total_cost : ℕ := cost_burger + num_packs_of_fries * cost_pack_of_fries + cost_salad

-- The statement to prove that the total cost Benjamin paid is $15
theorem total_cost_correct : total_cost = 15 := by
  -- This is where the proof would go, but we're omitting it for now.
  sorry

end total_cost_correct_l184_184109


namespace correct_option_is_c_l184_184100

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c_l184_184100


namespace five_inv_mod_31_l184_184565

theorem five_inv_mod_31 : ∃ x : ℤ, 0 ≤ x ∧ x < 31 ∧ (5 * x ≡ 1 [MOD 31]) :=
by 
  use 25
  split
  norm_num
  split
  norm_num
  norm_num
  exact nat.mod_eq_of_lt (by norm_num)
  norm_num
  norm_num
  sorry

end five_inv_mod_31_l184_184565


namespace angle_between_u_v_l184_184674

noncomputable def angle_between_vectors {R : Type} [inner_product_space 𝕜 R] (u v : R) :=
  real.arccos (inner_product_space.inner u v / (∥u∥ * ∥v∥))

theorem angle_between_u_v :
  let u v : EuclideanSpace ℝ (Fin 3) :=
  (∥u∥ = 1) ∧ (∥v∥ = 1) ∧
  (inner_product_space.inner (u + 3 * v) (3 * u - 2 * v) = 0) →
  angle_between_vectors u v = real.arccos (3 / 7) :=
begin
  assume h,
  sorry
end

end angle_between_u_v_l184_184674


namespace three_blocks_no_same_row_or_col_l184_184518

theorem three_blocks_no_same_row_or_col :
  let n := 4
  let k := 3
  let combinations := (Nat.choose n k) * (Nat.choose n k) * Nat.factorial k
  n = 4 ∧ k = 3 → combinations = 96 :=
begin
  assume h,
  sorry
end

end three_blocks_no_same_row_or_col_l184_184518


namespace jennifer_initial_money_l184_184661

def initial_money_spent (X : ℝ) : ℝ :=
  (1/5) * X + (1/6) * X + (1/2) * X

theorem jennifer_initial_money (X : ℝ) (h : initial_money_spent X + 20 = X) : X = 150 :=
  sorry

end jennifer_initial_money_l184_184661


namespace team_A_days_additional_people_l184_184561

theorem team_A_days (x : ℕ) (y : ℕ)
  (h1 : 2700 / x = 2 * (1800 / y))
  (h2 : y = x + 1)
  : x = 3 ∧ y = 4 :=
by
  sorry

theorem additional_people (m : ℕ)
  (h1 : (200 : ℝ) * 10 * 3 + 150 * 8 * 4 = 10800)
  (h2 : (170 : ℝ) * (10 + m) * 3 + 150 * 8 * 4 = 1.20 * 10800)
  : m = 6 :=
by
  sorry

end team_A_days_additional_people_l184_184561


namespace angle_BDC_invariant_l184_184719

theorem angle_BDC_invariant
  (S1 S2 : Circle)
  (A B C D : Point)
  (h_intersect : S1.intersection S2 = {A, P})
  (h_line_through_A : ∃ l : Line, l.contains A ∧ l.intersects_circle S1 = {A, B} ∧ l.intersects_circle S2 = {A, C})
  (h_tangents : tangent_at_point S1 B D ∧ tangent_at_point S2 C D) :
  ∀ l : Line, l.contains A → ∃ B' C' D' : Point, l.intersects_circle S1 = {A, B'} ∧ l.intersects_circle S2 = {A, C'} ∧ tangent_at_point S1 B' D' ∧ tangent_at_point S2 C' D' ∧ ∠ B' D' C' = ∠ B D C :=
sorry

end angle_BDC_invariant_l184_184719


namespace M_returns_to_initial_position_min_steps_to_return_l184_184053

variables {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable M : A

-- Assume a generic triangle ABC in a metric space context
variables (AB AC BC : Set A)

-- Assume point M starts on segment AB
variable (a : ℝ)

-- Define the movement of M
def move_M_parallel_BC : A → A := sorry
def move_M_parallel_AB : A → A := sorry

-- Define the cyclic movement steps
def movement_cycle (M_initial : A) : ℕ → A
| 0       := M_initial
| (n + 1) := let M_next := move_M_parallel_AB (move_M_parallel_BC (movement_cycle n)) in
             M_next

-- Statement for M returning to its initial position
theorem M_returns_to_initial_position :
  ∃ n : ℕ, n > 0 ∧ movement_cycle M n = M :=
sorry

-- Statement for the minimum number of steps
theorem min_steps_to_return (a : ℝ) (ha : a > 0 ∧ a < 1) :
  ∃ n : ℕ, n > 0 ∧ n ≤ if a = 1/2 then 3 else 6 ∧ movement_cycle M n = M :=
sorry

end M_returns_to_initial_position_min_steps_to_return_l184_184053


namespace fraction_value_l184_184118

theorem fraction_value :
  (∏ i in (finset.range (25)), (24 + (i + 1) : ℚ) / (i + 1)) / 
  (∏ i in (finset.range (23)), (25 + (i + 2) : ℚ) / (i + 1)) = 600 := by
  sorry

end fraction_value_l184_184118


namespace number_of_books_l184_184531

-- Define the conditions
def ratio_books : ℕ := 7
def ratio_pens : ℕ := 3
def ratio_notebooks : ℕ := 2
def total_items : ℕ := 600

-- Define the theorem and the goal to prove
theorem number_of_books (sets : ℕ) (ratio_books : ℕ := 7) (total_items : ℕ := 600) : 
  sets = total_items / (7 + 3 + 2) → 
  sets * ratio_books = 350 :=
by
  sorry

end number_of_books_l184_184531


namespace distance_between_points_l184_184827

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_points :
  let p1 := point 1 0 2
  let p2 := point 1 (-3) 1
  distance p1 p2 = Real.sqrt 10 :=
by
  sorry

end distance_between_points_l184_184827


namespace max_intersections_l184_184368

-- Definitions for the conditions of the problem
def lines : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

def setA (n : ℕ) : Prop := n ∈ lines ∧ n % 5 = 0

def setB (n : ℕ) : Prop := n ∈ lines ∧ (n + 4) % 5 = 0

def setC (n : ℕ) : Prop := n ∈ lines ∧ ¬setA n ∧ ¬setB n

-- The given point B
variable (B : Point) -- assuming Point is properly defined elsewhere

-- Ensure lines in setA are parallel
def parallel_lines (n m : ℕ) (L : ℕ → Line) : Prop := 
  (setA n ∧ setA m) → parallel (L n) (L m)

-- Ensure lines in setB pass through point B
def lines_pass_through_B (n : ℕ) (L : ℕ → Line) : Prop := 
  setB n → passes_through (L n) B

-- Ensure every fifth line starting from L3 is perpendicular to L1
def perpendicular_lines (n : ℕ) (L : ℕ → Line) : Prop :=
  (n - 3) % 5 = 0 → perpendicular (L n) (L 1)

-- Main statement to be proved
theorem max_intersections 
  (L : ℕ → Line)
  (hA_parallel : ∀ n m, parallel_lines n m L)
  (hB_through_B : ∀ n, lines_pass_through_B n L)
  (hC_perpendicular : ∀ n, perpendicular_lines n L) :
  max_intersections (L 1) (L 2) ... (L 150) = 10152 :=
sorry -- Proof to be filled in

end max_intersections_l184_184368


namespace line_passes_through_fixed_point_l184_184892

theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ x y : ℝ, x = 9 ∧ y = -4 ∧ (m-1)*x + (2m-1)*y = m-5 :=
by
  use (9, -4)
  simp
  sorry

end line_passes_through_fixed_point_l184_184892


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l184_184401

theorem solve_eq1 : ∀ (x : ℝ), x^2 - 5 * x = 0 ↔ x = 0 ∨ x = 5 :=
by sorry

theorem solve_eq2 : ∀ (x : ℝ), (2 * x + 1)^2 = 4 ↔ x = -3 / 2 ∨ x = 1 / 2 :=
by sorry

theorem solve_eq3 : ∀ (x : ℝ), x * (x - 1) + 3 * (x - 1) = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem solve_eq4 : ∀ (x : ℝ), x^2 - 2 * x - 8 = 0 ↔ x = -2 ∨ x = 4 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l184_184401


namespace length_BC_l184_184656

variable {A B C : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (triangleABC : Triangle A B C)
variable (right_triangle : triangleABC.is_right)
variable (tanB : triangleABC.tan B = 4 / 3)
variable (length_AB : triangleABC.length AB = 3)

theorem length_BC (triangleABC : Triangle A B C) 
  (right_triangle : triangleABC.is_right) 
  (tanB : triangleABC.tan B = 4 / 3) 
  (length_AB : triangleABC.length AB = 3) : 
  triangleABC.length BC = 5 :=
sorry

end length_BC_l184_184656


namespace exterior_angle_sum_constant_l184_184104

theorem exterior_angle_sum_constant (n : ℕ) (h : n ≥ 3) : exterior_angle_sum n = 360 :=
sorry

end exterior_angle_sum_constant_l184_184104


namespace evaluate_expression_l184_184869

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l184_184869


namespace vertex_connected_to_all_l184_184506

noncomputable section

open Classical

variables {V : Type} [Fintype V] (G : SimpleGraph V)

def A (n : ℕ) := {v : V | ∃ (i : fin n), v = (G.verts.toFinset.filter (λ x, x ∈ V)).nth i}
def B (k : ℕ) := {v : V | ∃ (i : fin k), v = (G.verts.toFinset.filter (λ x, x ∉ (A n))).nth i}

theorem vertex_connected_to_all (n k p : ℕ) (h1 : n + k = Fintype.card V) 
  (h2 : ∀ a : V, a ∈ A n → (fintype.card {b ∈ B k | G.Adj a b} ≥ k - p)) 
  (h3 : n * p < k) : 
  ∃ b : V, b ∈ B k ∧ ∀ a : V, a ∈ A n → G.Adj a b :=
sorry

end vertex_connected_to_all_l184_184506


namespace simplify_expression_l184_184709

variable (a b : ℤ)

theorem simplify_expression :
  (15 * a + 45 * b) + (20 * a + 35 * b) - (25 * a + 55 * b) + (30 * a - 5 * b) = 
  40 * a + 20 * b :=
by
  sorry

end simplify_expression_l184_184709


namespace geometric_series_solution_l184_184311

theorem geometric_series_solution (b1 q : ℝ) (hq : |q| < 1) (h3 : 10.5 = b1 * (1 + q + q^2))
  (hinf : 12 = b1 / (1 - q)) :
  (b1 = 6) ∧ (q = 1/2) ∧ 
  (series : ℕ → ℝ) (n : ℕ) (series 0 = b1) (series 1 = b1 * q) (series 2 = b1 * q^2) := 
begin
  sorry
end

end geometric_series_solution_l184_184311


namespace triangle_tan_ratio_l184_184909

theorem triangle_tan_ratio (A B C a b c : ℝ) 
(h_triangle : ∀ A B C, A + B + C = π)
(h_eq : a * Real.cos B - b * Real.cos A = (3 / 5) * c) :
  Real.tan A / Real.tan B = 4 :=
sorry

end triangle_tan_ratio_l184_184909


namespace dylan_needs_8_trays_l184_184865

-- Define the initial conditions:
def ice_cubes_in_glass : ℕ := 8
def multiplier_for_lemonade : ℕ := 2
def number_of_guests : ℕ := 5
def tray_capacity : ℕ := 14
def percentage_used : ℚ := 0.75
def ice_cubes_used : ℕ := 80

-- Define the floors for final calculation in natural numbers
def initial_ice_cubes : ℕ := (ice_cubes_used : ℚ / percentage_used).ceil.to_nat

-- The total number of trays needed
def trays_needed : ℕ := (initial_ice_cubes.to_nat / tray_capacity.to_nat).ceil.to_nat

theorem dylan_needs_8_trays : trays_needed = 8 := by
  sorry

end dylan_needs_8_trays_l184_184865


namespace incorrect_operation_is_sqrt2_add_sqrt3_eq_sqrt5_l184_184035

theorem incorrect_operation_is_sqrt2_add_sqrt3_eq_sqrt5 :
  (¬ (sqrt 2 + sqrt 3 = sqrt 5)) ∧
  (sqrt 2 * sqrt 3 = sqrt 6) ∧
  (sqrt 2 / sqrt (1 / 2) = 2) ∧
  (sqrt 2 + sqrt 8 = 3 * sqrt 2) →
  ¬ (sqrt 2 + sqrt 3 = sqrt 5) :=
by
  intros h
  exact h.1

end incorrect_operation_is_sqrt2_add_sqrt3_eq_sqrt5_l184_184035


namespace simplify_expression_l184_184400

variable {a b c : ℝ}

-- Assuming the conditions specified in the problem
def valid_conditions (a b c : ℝ) : Prop := (1 - a * b ≠ 0) ∧ (1 + c * a ≠ 0)

theorem simplify_expression (h : valid_conditions a b c) :
  (a + b) / (1 - a * b) + (c - a) / (1 + c * a) / 
  (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) = 
  (b + c) / (1 - b * c) := 
sorry

end simplify_expression_l184_184400


namespace curve_equation_and_max_triangle_area_l184_184221

noncomputable def curve_C : set (ℝ × ℝ) :=
{ p | let (x, y) := p in (x^2 / 4) + y^2 = 1 }

theorem curve_equation_and_max_triangle_area :
  (∀ P : ℝ × ℝ, ((P.1 - sqrt 3)^2 + P.2^2) / abs (P.1 - 4 * sqrt 3 / 3) = sqrt 3 / 2 → 
    (P ∈ curve_C)) ∧
  (∀ l : ℝ → ℝ, ∀ M N : ℝ × ℝ, 
    (l = λ x, k * x + m) → 
    (M ∈ curve_C ∧ N ∈ curve_C ∧ l M.1 = M.2 ∧ l N.1 = N.2) → 
    (let k_OM := M.2 / M.1, k_ON := N.2 / N.1 in (k_OM * k_ON = 5 / 4) → 
    ∃ max_area : ℝ, max_area = 1)) :=
by 
  sorry

end curve_equation_and_max_triangle_area_l184_184221


namespace locus_of_intersection_l184_184301

theorem locus_of_intersection
  (a b : ℝ) (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) (h_neq : a ≠ b) :
  ∃ (x y : ℝ), 
    (∃ c : ℝ, y = (a/c)*x ∧ (x/b + y/c = 1)) 
    ∧ 
    ( (x - b/2)^2 / (b^2/4) + y^2 / (ab/4) = 1 ) :=
sorry

end locus_of_intersection_l184_184301


namespace sum_first_5_terms_arithmetic_l184_184319

variable {a : ℕ → ℝ} -- Defining a sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_eq_1 : a 2 = 1
axiom a4_eq_5 : a 4 = 5

-- Theorem statement
theorem sum_first_5_terms_arithmetic (h_arith : is_arithmetic_sequence a) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end sum_first_5_terms_arithmetic_l184_184319


namespace total_pull_ups_per_week_l184_184373

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l184_184373


namespace seq_eval_l184_184086

noncomputable def a : ℕ → ℕ
| 0     := 0 -- defaulting to 0 for convenience, though not used
| 1     := 1
| 2     := 2
| 3     := 3
| (n+1) := (a n) * (a (n-1)) * (a (n-2)) - 1

theorem seq_eval :
  ((list.range 100).prod.map a) - ((list.range 100).sum.map (λ i, (a i)^2)) = -104 :=
by
  sorry

end seq_eval_l184_184086


namespace original_avg_expenditure_correct_l184_184438

variables (A B C a b c X Y Z : ℝ)
variables (hA : A > 0) (hB : B > 0) (hC : C > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem original_avg_expenditure_correct
    (h_orig_exp : (A * X + B * Y + C * Z) / (A + B + C) - 1 
    = ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42):
    True := 
sorry

end original_avg_expenditure_correct_l184_184438


namespace sales_tax_difference_l184_184045

/-- The difference in sales tax calculation given the changes in rate. -/
theorem sales_tax_difference 
  (market_price : ℝ := 9000) 
  (original_rate : ℝ := 0.035) 
  (new_rate : ℝ := 0.0333) 
  (difference : ℝ := 15.3) :
  market_price * original_rate - market_price * new_rate = difference :=
by
  /- The proof is omitted as per the instructions. -/
  sorry

end sales_tax_difference_l184_184045


namespace cos_arcsin_eq_l184_184840

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l184_184840


namespace suff_not_necessary_l184_184213

theorem suff_not_necessary (a : ℝ) : (a > 2 → a^2 > 2 * a) ∧ (¬ (a^2 > 2 * a → a > 2)) :=
by
  split
  sorry
  sorry

end suff_not_necessary_l184_184213


namespace find_g_and_a_l184_184422

noncomputable def g (x : ℝ) (m n : ℝ) : ℝ :=
  m * x ^ 2 - 2 * m * x + n + 1

noncomputable def f (x : ℝ) (m n a : ℝ) : ℝ :=
  g x m n + (2 - a) * x

theorem find_g_and_a (m n a : ℝ) (h1 : 0 < m)
    (h_max : ∀ x ∈ set.Icc (0 : ℝ) (3 : ℝ), g x m n ≤ 4)
    (h_min : ∀ x ∈ set.Icc (0 : ℝ) (3 : ℝ), 0 ≤ g x m n)
    (h_f_min : ∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), f x m n a ≥ -3) :
    (g = (λ x : ℝ, x^2 - 2 * x + 1)) ∧ (a = -5 ∨ a = 4) :=
sorry

end find_g_and_a_l184_184422


namespace arithmetic_mean_of_primes_in_list_l184_184194

def list_of_numbers := [33, 35, 37, 39, 41, 43, 45]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ Nat.divisors n, m = 1 ∨ m = n

def primes_in_list (lst : List ℕ) : List ℕ :=
  lst.filter is_prime

def arithmetic_mean (lst : List ℕ) : ℚ :=
  let sum := lst.sum
  let count := lst.length
  if count = 0 then 0 else sum / count

theorem arithmetic_mean_of_primes_in_list :
  arithmetic_mean (primes_in_list list_of_numbers) = 40.3 := by
    sorry

end arithmetic_mean_of_primes_in_list_l184_184194


namespace hair_ratio_l184_184530

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end hair_ratio_l184_184530


namespace max_area_inscribed_quadrilateral_in_ellipse_l184_184603

-- Definitions of the conditions
variables {r a b : ℝ}
variables (h_r_pos : r > 0) 
variables (h_a_pos : a > 0) 
variables (h_b_pos : b > 0)
variables (h_a_gt_b : a > b)
variables (max_area_circle : 2 * r^2)

-- Statement to prove
theorem max_area_inscribed_quadrilateral_in_ellipse : 
  (∃ (max_area : ℝ), max_area = 2 * a * b) :=
by {
  -- Assuming we know the maximum area of an inscribed quadrilateral in the circle
  have h_circle_max_area : ∀ (r : ℝ) (h_r : r > 0), ∃ (area : ℝ), area = 2 * r^2 := 
    λ r h_r, ⟨2 * r^2, max_area_circle⟩,
  sorry
}

end max_area_inscribed_quadrilateral_in_ellipse_l184_184603


namespace cos_arcsin_l184_184834

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l184_184834


namespace dwarfs_truthful_count_l184_184149

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184149


namespace first_set_cost_l184_184062

theorem first_set_cost (F S : ℕ) (hS : S = 50) (h_equation : 2 * F + 3 * S = 220) 
: 3 * F + S = 155 := 
sorry

end first_set_cost_l184_184062


namespace divisible_by_11_l184_184585

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 3
| (n+2) := (n+3) * a (n+1) - (n+2) * a n

theorem divisible_by_11 (n : ℕ) : (a n) % 11 = 0 ↔ n = 4 ∨ n = 8 ∨ 10 ≤ n :=
by sorry

end divisible_by_11_l184_184585


namespace solution_l184_184230

-- Define the propositions
def p : Prop := ∀ x : ℝ, 2^x < 3^x
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- State the theorem
theorem solution : ¬ p ∧ q :=
by sorry

end solution_l184_184230


namespace evaluate_expression_l184_184558

theorem evaluate_expression 
  (a b : ℚ)
  (h : ∀ x : ℚ, 0 < x → (a / (10^x + 1) + b / (10^x - 2) = (2 * 10^x - 3) / ((10^x + 1) * (10^x - 2)))) :
  a - 1/2 * b = 6.5 := 
by 
  sorry

end evaluate_expression_l184_184558


namespace dwarfs_truthful_count_l184_184163

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184163


namespace range_of_a_l184_184678

noncomputable def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, q x → p x) : a ≥ 2 ∨ a ≤ -4 :=
sorry

end range_of_a_l184_184678


namespace pages_in_second_chapter_l184_184496

theorem pages_in_second_chapter
  (total_pages : ℕ)
  (first_chapter_pages : ℕ)
  (second_chapter_pages : ℕ)
  (h1 : total_pages = 93)
  (h2 : first_chapter_pages = 60)
  (h3: second_chapter_pages = total_pages - first_chapter_pages) :
  second_chapter_pages = 33 :=
by
  sorry

end pages_in_second_chapter_l184_184496


namespace complex_num_in_first_quadrant_l184_184651

-- Define the complex number in question
def complex_num : ℂ := (-1 + complex.I) / complex.I

-- Assert that the point corresponding to this complex number is in the first quadrant
theorem complex_num_in_first_quadrant : complex_num.re > 0 ∧ complex_num.im > 0 := 
sorry

end complex_num_in_first_quadrant_l184_184651


namespace dwarfs_truthful_count_l184_184145

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l184_184145


namespace tan_sub_eq_minus_2sqrt3_l184_184396

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end tan_sub_eq_minus_2sqrt3_l184_184396


namespace fans_with_all_free_items_l184_184208

theorem fans_with_all_free_items : 
  let n := 4000 in 
  let lcm_75_30_50 := Nat.lcm (Nat.lcm 75 30) 50 in 
  n / lcm_75_30_50 = 26 :=
by
  let n := 4000
  let lcm_75 := 75
  let lcm_30 := 30
  let lcm_50 := 50
  let lcm_75_30 := Nat.lcm lcm_75 lcm_30
  let lcm_75_30_50 := Nat.lcm lcm_75_30 lcm_50
  have h1 : lcm_75_30_50 = 150 := by sorry
  have h2 : n / lcm_75_30_50 = 4000 / 150 := by rw [h1]
  have h3 : 4000 / 150 = 26 := by sorry
  exact nat.div_eq_of_lt sorry -- This is needed to handle the exact division part due to truncation in division.
  sorry

end fans_with_all_free_items_l184_184208


namespace polynomial_eq_2x2_sub_y2_l184_184509

def P (x y : ℝ) := P x y  -- This is a placeholder to be defined by the conditions.

theorem polynomial_eq_2x2_sub_y2 (x y : ℝ) :
  P x y - (x^2 - 3 * y^2) = x^2 + 2 * y^2 → P x y = 2 * x^2 - y^2 :=
by
  intros h
  sorry

end polynomial_eq_2x2_sub_y2_l184_184509


namespace omega_value_C_range_l184_184270

-- Problem (I)
-- Define the function f(x)
def f (ω x : ℝ) : ℝ := cos (ω * x) * sin (ω * x - π / 3) + √3 * cos (ω * x) ^ 2 - √3 / 4

-- Given conditions
def dist_center_symmetry (ω : ℝ) : ℝ := π / 4

theorem omega_value (ω : ℝ) (h : 0 < ω) : dist_center_symmetry ω = π / 4 → ω = 1 :=
sorry

-- Problem (II)
-- Define the function g(x)
def g (ω x C : ℝ) : ℝ := f ω (C * x)

-- Given conditions
def axis_symmetry_cond (C : ℝ) (x : ℝ) : Prop :=
  ∀ k : ℤ, ∀ x : ℝ, g 1 x (C) = (1 / 2) * sin (2 * C * x + π / 3) →
  (x ≠ k * (2 * π) + π / 2) ∧ (x ≠ k * (3 * π) + π / 2)

theorem C_range (C : ℝ) (h1 : 0 < C ∧ C < ∞) :
  (∀ x : ℝ, axis_symmetry_cond C x → C ∉ (2π, 3π)) →
  C ∈ (set.Ioc 0 (1 / 36) ∪ set.Icc (1 / 24) (7 / 36) ∪ set.Icc (7 / 24) (13 / 36)) :=
sorry

end omega_value_C_range_l184_184270


namespace range_of_a_l184_184626

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q (a : ℝ) : Prop := (1 - 4 * a) ≥ 0
def only_one_true (p q : Prop) : Prop := (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_a (a : ℝ) :
  only_one_true (p a) (q a) ↔ a ∈ (-∞, 0) ∪ (1/4, 4) := by
  sorry

end range_of_a_l184_184626


namespace zero_in_interval_l184_184349

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem zero_in_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) (h4 : f 2 > 0) :
  ∃ x ∈ set.Ioo 1.25 1.5, f x = 0 :=
sorry

end zero_in_interval_l184_184349


namespace sum_even_coefficients_l184_184342

theorem sum_even_coefficients (n : ℕ) : 
  let s := (∑ i in Finset.range (n+1), if even i then (a i) else 0)
  in s = (3^n + 1) / 2 := 
by
  sorry

end sum_even_coefficients_l184_184342


namespace least_addition_to_palindrome_l184_184022

theorem least_addition_to_palindrome : 
  ∃ n : ℕ, is_palindrome (54321 + n) ∧ n = 54445 - 54321 := 
begin
  use 54445 - 54321,
  split,
  { 
    -- proof that (54321 + (54445 - 54321)) is a palindrome
    sorry 
  },
  { 
    -- proof that n = 54445 - 54321
    refl
  }
end

end least_addition_to_palindrome_l184_184022


namespace cottonwood_fiber_scientific_notation_l184_184188

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l184_184188


namespace eccentricity_of_ellipse_l184_184225

variable (a b c : ℝ)
variable (F1 F2 : ℝ)

def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def center_to_ab_distance (a b c : ℝ) : Prop := (a * b / real.sqrt (a^2 + b^2) = real.sqrt 6 / 3 * c)
def focal_distance (a c : ℝ) := (c < a)
def ecc (a c : ℝ) := real.sqrt (1 - (c ^ 2 / a ^ 2))

theorem eccentricity_of_ellipse (h₀ : 0 < b) (h₁ : b < a) (h₂ : focal_distance a c)
  (h₃ : b^2 = a^2 - c^2)
  (h₄ : center_to_ab_distance a b c) :
  ecc a c = real.sqrt 2 / 2 := by
sorry

end eccentricity_of_ellipse_l184_184225


namespace no_family_of_lines_exists_l184_184321

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, (1 : ℝ) = k n * (1 : ℝ) + (1 - k n)) ∧
  (∀ n, k (n + 1) = a n - b n ∧ a n = 1 - 1 / k n ∧ b n = 1 - k n) ∧
  (∀ n, k n * k (n + 1) ≥ 0) →
  False :=
by
  sorry

end no_family_of_lines_exists_l184_184321


namespace guaranteed_winning_strategy_l184_184437

variable (a b : ℝ)

theorem guaranteed_winning_strategy (h : a ≠ b) : (a^3 + b^3) > (a^2 * b + a * b^2) :=
by 
  sorry

end guaranteed_winning_strategy_l184_184437


namespace xiaoli_estimate_smaller_l184_184474

variable (x y z : ℝ)
variable (hx : x > y) (hz : z > 0)

theorem xiaoli_estimate_smaller :
  (x - z) - (y + z) < x - y := 
by
  sorry

end xiaoli_estimate_smaller_l184_184474


namespace sum_num_denom_cos_gamma_l184_184304

theorem sum_num_denom_cos_gamma
  (chords : ℕ → ℝ)
  (h_chords_2 : chords 1 = 2)
  (h_chords_4 : chords 2 = 4)
  (h_chords_5 : chords 3 = 5)
  (angles : ℕ → ℝ)
  (h_angles_gamma : angles 1 = γ)
  (h_angles_delta : angles 2 = δ)
  (h_angles_sum : angles 3 = γ + δ)
  (h_angle_sum_lt_pi : γ + δ < π)
  (h_cos_gamma_pos_rational : ∃ p q : ℚ, \(\cos \gamma = \frac{p}{q} \wedge p > 0 \wedge q > 0 \)): 
  ∀ (numerator denominator : ℤ), 
  ∃ (p q : ℚ), \(\cos \gamma = \frac{p}{q} \wedge p.num = numerator \wedge q.num = denominator\) → numerator + denominator = 49 := by
  sorry

end sum_num_denom_cos_gamma_l184_184304


namespace sin_graph_transformation_l184_184447

theorem sin_graph_transformation :
  ∀ (x : ℝ), (1 / 2) * sin (2 * x + π / 6) = (1 / 2) * sin (x + π / 6) :=
sorry

end sin_graph_transformation_l184_184447


namespace twenty_step_paths_l184_184067

theorem twenty_step_paths (paths : ℕ) :
  (∀ p : ℕ × ℕ, 
    ∀ i ∈ finset.range 21, 
    (p.1 = -5 + i ∨ p.2 = -5 + i) → (p.1 ≤ -3 ∨ p.1 ≥ 3 ∨ p.2 ≤ -3 ∨ p.2 ≥ 3)) →
  paths = 4252 :=
sorry

end twenty_step_paths_l184_184067


namespace H_is_orthocenter_of_ABC_l184_184007

-- Given: Three equal circles intersect at a common point H, 
-- forming an acute-angled triangle ABC with the points of their intersection.

-- Definitions:
noncomputable def equalCirclesIntersectAt (H : Point) (A B C : Point) : Prop :=
  ∃ (circle₁ circle₂ circle₃ : Circle), 
    (circle₁.radius = circle₂.radius) ∧ 
    (circle₂.radius = circle₃.radius) ∧ 
    (H ∈ circle₁ ∧ H ∈ circle₂ ∧ H ∈ circle₃) ∧ 
    (A ∈ circle₁ ∧ A ∈ circle₂) ∧ 
    (B ∈ circle₂ ∧ B ∈ circle₃) ∧ 
    (C ∈ circle₃ ∧ C ∈ circle₁) 

-- Assertion: H is the orthocenter of triangle ABC
theorem H_is_orthocenter_of_ABC {H A B C : Point} 
  (h_eqcircles : equalCirclesIntersectAt H A B C)
  (h_acute : acuteAngledTriangle A B C) : orthocenter H A B C :=
sorry

end H_is_orthocenter_of_ABC_l184_184007


namespace commodities_price_difference_l184_184408

theorem commodities_price_difference : 
  ∀ (C1 C2 : ℕ), 
    C1 = 477 → 
    C1 + C2 = 827 → 
    C1 - C2 = 127 :=
by
  intros C1 C2 h1 h2
  sorry

end commodities_price_difference_l184_184408


namespace odd_function_check_l184_184951

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184951


namespace can_cabinet_be_moved_out_through_door_l184_184310

/-
Definitions for the problem:
- Length, width, and height of the room
- Width, height, and depth of the cabinet
- Width and height of the door
-/

structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

def room : Dimensions := { length := 4, width := 2.5, height := 2.3 }
def cabinet : Dimensions := { length := 0.6, width := 1.8, height := 2.1 }
def door : Dimensions := { length := 0.8, height := 1.9, width := 0 }

theorem can_cabinet_be_moved_out_through_door : 
  (cabinet.length ≤ door.length ∧ cabinet.width ≤ door.height) ∨ 
  (cabinet.width ≤ door.length ∧ cabinet.length ≤ door.height) 
∧ 
cabinet.height ≤ room.height ∧ cabinet.width ≤ room.width ∧ 
cabinet.length ≤ room.length → True :=
by
  sorry

end can_cabinet_be_moved_out_through_door_l184_184310


namespace distance_between_stations_proof_l184_184453

-- Definitions from conditions
def train_rate_1 : ℝ := 16
def train_rate_2 : ℝ := 21
def extra_distance : ℝ := 60

-- Let D be the distance traveled by the slower train when they meet
-- The distance traveled by faster train will be D + 60
def distance_by_slower_train (D : ℝ) : Prop := (D / train_rate_1) = ((D + extra_distance) / train_rate_2)

-- Now, based on the condition, we need the actual distance between the two stations
def distance_between_stations : ℝ := λ D : ℝ, 2 * D + extra_distance

theorem distance_between_stations_proof : ∃ D : ℝ, distance_by_slower_train D ∧ distance_between_stations D = 444 :=
by
  use 192
  split
  sorry -- The actual proof goes here

end distance_between_stations_proof_l184_184453


namespace inequality_solution_absolute_inequality_l184_184492

-- Statement for Inequality Solution Problem
theorem inequality_solution (x : ℝ) : |x - 1| + |2 * x + 1| > 3 ↔ (x < -1 ∨ x > 1) := sorry

-- Statement for Absolute Inequality Problem with Bounds
theorem absolute_inequality (a b : ℝ) (ha : -1 ≤ a) (hb : a ≤ 1) (hc : -1 ≤ b) (hd : b ≤ 1) : 
  |1 + (a * b) / 4| > |(a + b) / 2| := sorry

end inequality_solution_absolute_inequality_l184_184492


namespace largest_prime_to_check_31_l184_184746

-- Define a statement to capture the conditions
def is_prime_31_needed : Prop :=
  ∀ n : ℕ, (1000 ≤ n ∧ n ≤ 1100) → ∃ p : ℕ, p ∈ ({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31} : Set ℕ) ∧ p * p > n

-- The theorem statement proving that for any n in the given bounds,
-- 31 is one of the necessary primes to check for primality
theorem largest_prime_to_check_31 :
  is_prime_31_needed :=
by
  intros n hn
  have bounds : 1000 ≤ n ∧ n ≤ 1100 := hn
  have primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
  use 31
  split
  { exact Set.mem_of_mem_val (Set.mem_univ _) }
  { exact nat.sqrt_le_add_one_iff.2 bounds.2 sorry }

end largest_prime_to_check_31_l184_184746


namespace eval_nabla_neg_one_two_l184_184635

-- Define the conditions
def a_ge_neg_one (a : ℝ) := a ≥ -1
def b_ge_neg_one (b : ℝ) := b ≥ -1
def nabla (a b : ℝ) := (a + b) / (1 + a * b)

-- The theorem to be proven
theorem eval_nabla_neg_one_two : nabla (-1) 2 = -1 := 
by 
  -- to be proven
  sorry

end eval_nabla_neg_one_two_l184_184635


namespace evaluate_expression_l184_184560

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end evaluate_expression_l184_184560


namespace eccentricity_of_hyperbola_l184_184291

open Real

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ x y : ℝ, (x^2 + y^2 - 3 * x - 4 * y - 5 = 0) ∧ (ax - by = 0))
  (h4 : b = (3 / 4) * a) :
  let e := (sqrt (a^2 + b^2)) / a in e = 5 / 4 :=
by
  sorry

end eccentricity_of_hyperbola_l184_184291


namespace correct_pair_of_events_l184_184210

-- Definitions of events based on the problem's conditions
def event_at_least_one_black_ball : Type := { b | b ≥ 1 }
def event_both_black_balls : Type := { b | b = 2 }
def event_exactly_one_black_ball : Type := { b | b = 1 }
def event_exactly_two_black_balls : Type := { b | b = 2 }
def event_at_least_one_red_ball : Type := { r | r ≥ 1 }
def event_both_red_balls : Type := { r | r = 2 }

noncomputable def mutually_exclusive_but_not_complementary (A B : Type) : Prop :=
  (∃ x : A, x ∉ B) ∧ (∃ x : B, x ∉ A) ∧ (∀ x : A, ∀ y : B, x ≠ y)

theorem correct_pair_of_events :
  mutually_exclusive_but_not_complementary event_exactly_one_black_ball event_exactly_two_black_balls :=
  sorry

end correct_pair_of_events_l184_184210


namespace z_rate_is_correct_l184_184087

-- Number of units x gets
def units_x_gets (y_share : ℝ) (y_rate : ℝ) : ℝ := y_share / y_rate

-- Share of z
def z_share (total_amount : ℝ) (x_share : ℝ) (y_share : ℝ) : ℝ :=
  total_amount - (x_share + y_share)

-- Rate of z per unit of x
def z_rate_per_unit (z_share : ℝ) (units_x_gets : ℝ) : ℝ := z_share / units_x_gets

-- Prove that z's rate per unit is 0.30
theorem z_rate_is_correct (y_share : ℝ) (total_amount : ℝ) (y_rate : ℝ) (x_share : ℝ) : 
  units_x_gets y_share y_rate = 140 → 
  y_share = 63 → 
  total_amount = 245 → 
  y_rate = 0.45 → 
  x_share = 140 → 
  z_rate_per_unit (z_share total_amount x_share y_share) (units_x_gets y_share y_rate) = 0.30 :=
by
  intros
  sorry

end z_rate_is_correct_l184_184087


namespace choose_30_5_l184_184800

theorem choose_30_5 : nat.choose 30 5 = 142506 :=
by
  sorry

end choose_30_5_l184_184800


namespace william_time_on_road_l184_184470

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l184_184470


namespace odd_function_g_l184_184995

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184995


namespace cost_of_1500_pieces_of_gum_in_dollars_l184_184407

theorem cost_of_1500_pieces_of_gum_in_dollars :
  (2 * 1500 * (1 - 0.10) / 100) = 27 := sorry

end cost_of_1500_pieces_of_gum_in_dollars_l184_184407


namespace find_x_for_y_seventeen_l184_184605

theorem find_x_for_y_seventeen
  (k : ℝ)
  (h1 : ∀ x y : ℝ, (3 * x^2 - 4) / (y + 10) = k)
  (h2 : k = (3 * 1^2 - 4) / (2 + 10))
  (y : ℝ)
  (hx : k = -1 / 12)
  (hy : y = 17) :
  ∃ x : ℝ, x = sqrt (7 / 12) ∨ x = -sqrt (7 / 12) := by
  sorry

end find_x_for_y_seventeen_l184_184605


namespace odd_function_check_l184_184947

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184947


namespace min_k_for_2_to_8_min_k_for_2_to_31_l184_184060

-- Definitions for Problem 1
def can_color_2_to_8 (k : ℕ) :=
  ∀ (colors : fin 7 → ℕ), (∀ i, colors i < k) ∧
  ∀ (m n : fin 7), (n + 2) ∣ (m + 2) → m ≠ n → colors m ≠ colors n

-- Problem 1: Prove that the minimal k for coloring {2, 3, 4, 5, 6, 7, 8} is 3
theorem min_k_for_2_to_8 : ∃ k, can_color_2_to_8 k ∧ k = 3 :=
by {
  -- Proof goes here
  sorry
}

-- Definitions for Problem 2
def can_color_2_to_31 (k : ℕ) :=
  ∀ (colors : fin 30 → ℕ), (∀ i, colors i < k) ∧
  ∀ (m n : fin 30), (n + 2) ∣ (m + 2) → m ≠ n → colors m ≠ colors n

-- Problem 2: Prove that the minimal k for coloring {2, 3, ..., 31} is 4
theorem min_k_for_2_to_31 : ∃ k, can_color_2_to_31 k ∧ k = 4 :=
by {
  -- Proof goes here
  sorry
}

end min_k_for_2_to_8_min_k_for_2_to_31_l184_184060


namespace number_of_truthful_dwarfs_l184_184143

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184143


namespace conic_sections_eccentricities_l184_184555

theorem conic_sections_eccentricities (a b c : ℝ) (h₁ : a = 2) (h₂ : b = -5) (h₃ : c = 2) :
  let Δ := b^2 - 4 * a * c,
  Δ = 9 ∧
  let x₁ := (-b + Real.sqrt Δ) / (2 * a),
  let x₂ := (-b - Real.sqrt Δ) / (2 * a),
  (x₁ = 2 ∧ x₂ = 1/2) ∧
  (1 < x₁ ∧ 0 < x₂ ∧ x₂ < 1) :=
by
  sorry

end conic_sections_eccentricities_l184_184555


namespace dog_distance_22_5_km_l184_184380

noncomputable def distance_dog_travel (initial_distance personA_speed personB_speed dog_speed : ℝ) : ℝ :=
  let meeting_time := initial_distance / (personA_speed + personB_speed)
  dog_speed * meeting_time

theorem dog_distance_22_5_km :
  ∀ (d_a_dog personA_speed personB_speed dog_speed : ℝ),
    d_a_dog = 22.5 → 
    personA_speed = 2.5 → 
    personB_speed = 5 → 
    dog_speed = 7.5 → 
    distance_dog_travel d_a_dog personA_speed personB_speed dog_speed = 22.5 :=
by
  intros
  rw [distance_dog_travel, this, this_1, this_2, this_3]
  have mt : 22.5 / (2.5 + 5) = 3 := by norm_num
  rw [mt]
  norm_num
  sorry

end dog_distance_22_5_km_l184_184380


namespace range_of_a_l184_184621

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀^2 + a * x₀ + a < 0) ↔ (a ∈ set.Iic 0 ∨ a ∈ set.Ici 4) :=
begin
  -- skipped proof
  sorry
end

end range_of_a_l184_184621


namespace find_d_l184_184108

theorem find_d 
    (a b c d : ℝ) 
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_c_pos : 0 < c)
    (h_d_pos : 0 < d)
    (max_val : d + a = 7)
    (min_val : d - a = 1) :
    d = 4 :=
by
  sorry

end find_d_l184_184108


namespace transformed_function_is_odd_l184_184963

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184963


namespace geometric_sequence_b_l184_184240

theorem geometric_sequence_b (a b c : Real) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : ∃ r, b = r * a ∧ c = r * b) :
  b = 1 ∨ b = -1 :=
by
  sorry

end geometric_sequence_b_l184_184240


namespace geometric_sequence_term_sum_of_b_l184_184926

variable (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
variable (hngeometric : ∀ n, a (n + 1) = a n * q)
variable (h01 : q > 1)
variable (h02 : a 2 = 4)
variable (h03 : S 3 = 21)
variable (h04 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q))
variable (hlog : ∀ n, b n = Real.log (a (n + 1)) / Real.log 4)
variable (hsum2 : ∀ n, T n = ∑ i in Finset.range n, 2 / (b i * b (i + 1)))

theorem geometric_sequence_term :
  (∀ n, a n = 4^(n-1)) :=
sorry

theorem sum_of_b :
  (∀ n, T n = (2 * n) / (n + 1)) :=
sorry

end geometric_sequence_term_sum_of_b_l184_184926


namespace transformed_conic_symmetric_eq_l184_184883

def conic_E (x y : ℝ) := x^2 + 2 * x * y + y^2 + 3 * x + y
def line_l (x y : ℝ) := 2 * x - y - 1

def transformed_conic_equation (x y : ℝ) := x^2 + 14 * x * y + 49 * y^2 - 21 * x + 103 * y + 54

theorem transformed_conic_symmetric_eq (x y : ℝ) :
  (∀ x y, conic_E x y = 0 → 
    ∃ x' y', line_l x' y' = 0 ∧ conic_E x' y' = 0 ∧ transformed_conic_equation x y = 0) :=
sorry

end transformed_conic_symmetric_eq_l184_184883


namespace problem_I_problem_II_l184_184271

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

noncomputable def g (x t : ℝ) : ℝ := x * f(x) + t * (-(x / Real.exp x)) + Real.exp(-x)

theorem problem_I : ∃ x, f(x) = 1 := 
begin
  use 0,
  calc 
    f 0 = (0 + 1) / Real.exp 0 : by rw [Real.exp_zero]
    ... = 1 : by norm_num
end

theorem problem_II (t : ℝ) (M N : ℝ) (h : ∀ x ∈ Icc 0 1, M ≥ g x t ∧ N ≤ g x t) (h_MN : M > 2 * N) : 
  t ∈ (Set.Iio (3 - 2 * Real.exp 1)) ∪ (Set.Ioi (3 - Real.exp 1 / 2)) := 
sorry

end problem_I_problem_II_l184_184271


namespace johns_number_is_thirteen_l184_184335

theorem johns_number_is_thirteen (x : ℕ) (h1 : 10 ≤ x) (h2 : x < 100) (h3 : ∃ a b : ℕ, 10 * a + b = 4 * x + 17 ∧ 92 ≤ 10 * b + a ∧ 10 * b + a ≤ 96) : x = 13 :=
sorry

end johns_number_is_thirteen_l184_184335


namespace Josh_riddles_is_8_l184_184330

-- Define the number of riddles for each individual
variables {Josh_riddles Ivory_riddles Taso_riddles : ℕ}

-- Define the conditions from the given problem
def Ivory_riddles_eq_Josh_riddles_plus_4 : Prop := Ivory_riddles = Josh_riddles + 4
def Taso_riddles_eq_2_Ivory_riddles : Prop := Taso_riddles = 2 * Ivory_riddles
def Taso_riddles_eq_24 : Prop := Taso_riddles = 24

-- Prove how many riddles Josh has
theorem Josh_riddles_is_8
  (h1 : Ivory_riddles_eq_Josh_riddles_plus_4)
  (h2 : Taso_riddles_eq_2_Ivory_riddles)
  (h3 : Taso_riddles_eq_24) : Josh_riddles = 8 :=
by
  sorry

end Josh_riddles_is_8_l184_184330


namespace transformed_function_is_odd_l184_184982

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184982


namespace tilly_star_count_l184_184010

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l184_184010


namespace odd_function_g_l184_184999

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184999


namespace cricket_target_l184_184326

theorem cricket_target (run_rate_first_10overs run_rate_next_40overs : ℝ) (overs_first_10 next_40_overs : ℕ)
    (h_first : run_rate_first_10overs = 3.2) 
    (h_next : run_rate_next_40overs = 6.25) 
    (h_overs_first : overs_first_10 = 10) 
    (h_overs_next : next_40_overs = 40) 
    : (overs_first_10 * run_rate_first_10overs + next_40_overs * run_rate_next_40overs) = 282 :=
by
  sorry

end cricket_target_l184_184326


namespace pairwise_sums_modulo_l184_184206

theorem pairwise_sums_modulo (n : ℕ) (h : n = 2011) :
  ∃ (sums_div_3 sums_rem_1 : ℕ),
  (sums_div_3 = (n * (n - 1)) / 6) ∧
  (sums_rem_1 = (n * (n - 1)) / 6) := by
  sorry

end pairwise_sums_modulo_l184_184206


namespace find_sum_l184_184907

-- Define the sum and the double factorial
def sum_product (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range n, (∏ j in Finset.range (i + 1), a j)

def double_factorial : ℕ → ℕ 
| 0     := 1
| 1     := 1 
| (n+2) := (n+2) * double_factorial n

-- The conditions and main theorem
theorem find_sum (n : ℕ) (a : ℕ → ℕ) (h1 : a 0 = 1) (h2 : ∀ i < n, (a (i + 1) ≤ a i + 1)) : 
  sum_product a n = double_factorial (2 * n - 1) := by
  sorry

end find_sum_l184_184907


namespace mike_pull_ups_per_week_l184_184375

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l184_184375


namespace not_both_267_and_269_non_standard_l184_184724

def G : ℤ → ℤ := sorry

def exists_x_ne_c (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def non_standard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_267_and_269_non_standard (G : ℤ → ℤ)
  (h1 : exists_x_ne_c G) :
  ¬ (non_standard G 267 ∧ non_standard G 269) :=
sorry

end not_both_267_and_269_non_standard_l184_184724


namespace exists_colored_triangle_l184_184542

theorem exists_colored_triangle (n : ℕ) (c : ℕ) (h1 : n = 2000) (h2 : c = 999) 
  (polygon : Type) [convex polygon] [f : finset polygon] 
  (coloring : finset (polygon → polygon → ℕ)) 
  (h3 : ∀ x : polygon, x ∈ f) 
  (h4 : ∀ p1 p2 p3 : polygon, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → 
    ¬(coloring x1 x2 = coloring x2 x3 ∧ coloring x2 x3 = coloring x3 x1) 
    → ¬(p1, p2, p3 line on single line)) :
  ∃ t : finset (polygon × polygon), 
    (∀ (x1 x2 x3 : polygon), (x1, x2) ∈ t → (x2, x3) ∈ t → (x3, x1) ∈ t → 
    coloring x1 x2 = coloring x2 x3 ∧ coloring x2 x3 = coloring x3 x1) :=
begin
  sorry,
end

end exists_colored_triangle_l184_184542


namespace find_x_l184_184493

theorem find_x :
  ∃ x : ℝ, 12.1212 + x - 9.1103 = 20.011399999999995 ∧ x = 18.000499999999995 :=
sorry

end find_x_l184_184493


namespace average_headcount_is_11033_l184_184552

def average_headcount (count1 count2 count3 : ℕ) : ℕ :=
  (count1 + count2 + count3) / 3

theorem average_headcount_is_11033 :
  average_headcount 10900 11500 10700 = 11033 :=
by
  sorry

end average_headcount_is_11033_l184_184552


namespace MegatechBudgetAllocation_l184_184070

theorem MegatechBudgetAllocation :
  let
    microphotonics := 12
    home_electronics := 24
    modified_microorganisms := 29
    industrial_lubricants := 8
    astrophysics_degrees := 43.2
    total_degrees := 360
    basic_astrophysics := astrophysics_degrees / total_degrees * 100
    other_categories := microphotonics + home_electronics + modified_microorganisms + industrial_lubricants
    total_percentage := 100
  in
  basic_astrophysics = 12 ∧
  other_categories + basic_astrophysics = 85 ∧
  total_percentage - (other_categories + basic_astrophysics) = 15
:= by
  sorry

end MegatechBudgetAllocation_l184_184070


namespace trigonometric_expression_constant_l184_184383

theorem trigonometric_expression_constant (α : ℝ) (h : α ≠ (Int.pi * n / 2 + Int.pi / 12) for some n in ℤ) :
  (1 - 2 * sin (α - (3 * Real.pi / 2))^2 + (√3) * cos (2 * α + (3 * Real.pi / 2))) / sin ((Real.pi / 6) - 2 * α) = 2 := 
by
  sorry

end trigonometric_expression_constant_l184_184383


namespace problem_sum_neg1_and_powers_of_3_l184_184020

theorem problem_sum_neg1_and_powers_of_3 :
  (∑ k in Finset.range 2014, (-1)^(k+1)) + 3^0 + 3^1 = 4 := 
by
  -- skipping the proof
  sorry

end problem_sum_neg1_and_powers_of_3_l184_184020


namespace sum_of_squares_of_rates_equals_536_l184_184867

-- Define the biking, jogging, and swimming rates as integers.
variables (b j s : ℤ)

-- Condition: Ed's total distance equation.
def ed_distance_eq : Prop := 3 * b + 2 * j + 4 * s = 80

-- Condition: Sue's total distance equation.
def sue_distance_eq : Prop := 4 * b + 3 * j + 2 * s = 98

-- The main statement to prove.
theorem sum_of_squares_of_rates_equals_536 (hb : b ≥ 0) (hj : j ≥ 0) (hs : s ≥ 0) 
  (h1 : ed_distance_eq b j s) (h2 : sue_distance_eq b j s) :
  b^2 + j^2 + s^2 = 536 :=
by sorry

end sum_of_squares_of_rates_equals_536_l184_184867


namespace number_of_vertical_asymptotes_l184_184127

theorem number_of_vertical_asymptotes : 
  let f := λ x : ℝ, (2 * x - 3) / (x^2 + 4 * x - 21) in
  (∀ x : ℝ, (x = -7 ∨ x = 3) → is_vertical_asymptote f x) ∧
  (∀ x : ℝ, ¬(x = -7 ∨ x = 3) → ¬is_vertical_asymptote f x) →
  2 = 2 :=
by
  let f := λ x : ℝ, (2 * x - 3) / (x^2 + 4 * x - 21)
  have hv1 : f (-7) = (2 * (-7) - 3) / ( (-7)^2 + 4 * (-7) - 21) := sorry
  have hv2 : f 3 = (2 * 3 - 3) / (3^2 + 4 * 3 - 21) := sorry
  have v1_zero : (-7)^2 + 4 * (-7) - 21 = 0 := sorry
  have v2_zero : 3^2 + 4 * 3 - 21 = 0 := sorry
  sorry

end number_of_vertical_asymptotes_l184_184127


namespace page_numbering_digits_l184_184421

theorem page_numbering_digits (last_page : ℕ) (h_last_page : last_page = 128) : 
  (Σ n in finset.range last_page, nat.succ (nat.log 10 (n+1))) = 276 :=
by
  rw h_last_page
  sorry

end page_numbering_digits_l184_184421


namespace geometric_sequence_general_formula_sum_first_n_terms_b_l184_184249

noncomputable def geometric_sequence (a : ℕ → ℕ) :=
  (∀ n, a n = 3^(n-1))

theorem geometric_sequence_general_formula
  {a : ℕ → ℕ} (h1: a 2 + a 3 + a 4 = 39)
  (h2 : a 5 = 2 * a 4 + 3 * a 3) :
  geometric_sequence a :=
sorry

noncomputable def sequence_b (b : ℕ → ℕ) (a : ℕ → ℕ) :=
  ∀ n, b n = n + a n

noncomputable def sum_b (T : ℕ → ℕ) (b : ℕ → ℕ) :=
  ∀ n, T n = ∑ k in finset.range n, b (k+1)

theorem sum_first_n_terms_b
  {b : ℕ → ℕ} (a : ℕ → ℕ) (T : ℕ → ℕ)
  (h1: sequence_b b a)
  (h2: geometric_sequence a) :
  sum_b T b → ∀ n, T n = (3^n + n^2 + n - 1) / 2 :=
sorry

end geometric_sequence_general_formula_sum_first_n_terms_b_l184_184249


namespace length_HY_proof_l184_184708

noncomputable def length_HY (side_length : ℝ) : ℝ :=
let CD := side_length in
let CY := 4 * CD in
let CQ := side_length / real.sqrt 2 in
let QH := side_length / real.sqrt 2 in
let QY := CQ + CY in
real.sqrt (QH^2 + QY^2)

theorem length_HY_proof :
  length_HY 3 = real.sqrt (450 + 36 * real.sqrt 2) :=
by
  sorry

end length_HY_proof_l184_184708


namespace trailing_zeros_310_factorial_l184_184420

def count_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem trailing_zeros_310_factorial :
  count_trailing_zeros 310 = 76 := by
sorry

end trailing_zeros_310_factorial_l184_184420


namespace characteristic_value_le_fraction_n_plus_one_l184_184487

def characteristic_value (placement : List (List ℕ)) (n : ℕ) : ℚ :=
  let ratios := 
    List.concatMap (fun row => List.concatMap (fun x => List.map (fun y => (max x y) / (min x y)) (List.filter (· ≠ x) row)) row) placement
  let col_ratios :=
    List.concatMap (fun col => List.concatMap (fun x => List.map (fun y => (max x y) / (min x y)) (List.filter (· ≠ x) col)) placement.transpose) placement.transpose
  List.foldl min (ratios ++ col_ratios) (ratios.head!)

theorem characteristic_value_le_fraction_n_plus_one
  (n : ℕ)
  (placement : List (List ℕ))
  (h1 : placement.length = n)
  (h2 : ∀ row, row ∈ placement → row.length = n)
  (h3 : ∀ i j, 1 ≤ i ∧ i ≤ n^2 ∧ 1 ≤ j ∧ j ≤ n^2 → (placement.flatten).count i ≤ 1 ∧ (placement.flatten).count j ≤ 1) :
  characteristic_value placement n ≤ (↑(n+1) / ↑n : ℚ) := by sorry

end characteristic_value_le_fraction_n_plus_one_l184_184487


namespace floor_abs_neg_45_7_l184_184184

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184184


namespace number_of_customers_left_l184_184814

theorem number_of_customers_left (x : ℕ) (h : 14 - x + 39 = 50) : x = 3 := by
  sorry

end number_of_customers_left_l184_184814


namespace inequality_holds_iff_l184_184856

theorem inequality_holds_iff (n : ℕ) :
  (∀ (x : Fin n → ℝ), 
    ( (∑ i in Finset.univ, (x i) ^ n / n) - (∏ i, x i))
    * (∑ i in Finset.univ, x i) ≥ 0 )
  ↔ (n = 1 ∨ n = 3) :=
sorry

end inequality_holds_iff_l184_184856


namespace negation_all_dogs_playful_l184_184731

variable {α : Type} (dog playful : α → Prop)

theorem negation_all_dogs_playful :
  (¬ ∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬ playful x) :=
by sorry

end negation_all_dogs_playful_l184_184731


namespace polynomial_solution_l184_184879

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x, (x + 2019) * (P.eval x) = x * (P.eval (x + 1))) :
  ∃ C : ℝ, P = Polynomial.C C * Polynomial.X * (Polynomial.X + 2018) :=
sorry

end polynomial_solution_l184_184879


namespace value_of_M_l184_184361

theorem value_of_M :
  let M := (sqrt (5 + sqrt 6) + sqrt (5 - sqrt 6)) / sqrt (sqrt 6 - 1) - sqrt (4 - 2 * sqrt 3)
  in M = 1 :=
by
  let M := (sqrt (5 + sqrt 6) + sqrt (5 - sqrt 6)) / sqrt (sqrt 6 - 1) - sqrt (4 - 2 * sqrt 3)
  show M = 1
  sorry

end value_of_M_l184_184361


namespace finite_swaps_proof_l184_184822

noncomputable def finite_swaps (arr : List ℝ) :=
  ∀ (a b c d : ℝ), 
  (a, b, c, d) ∈ List.cycles_tuples arr 4 → 
  (a - d) * (b - c) < 0 →
  ∃ N : ℕ, ∀ n > N, ¬ (a, c, b, d) ∈ List.cycles_tuples (List.swap arr c b n) 

theorem finite_swaps_proof (arr : List ℝ) : finite_swaps (arr) :=
  sorry

end finite_swaps_proof_l184_184822


namespace baron_is_boasting_l184_184107

noncomputable def a (n : ℕ) : ℕ := sorry -- Assume a sequence of all natural numbers excluding 1

def b : ℕ → ℕ
| 0       := 0 -- Adjusting for Lean's 0-based indexing
| (n + 1) := if n = 0 then 1 else a (b n)

theorem baron_is_boasting :
  ∃ (infinitely_many_n : ℕ → Prop),
  (∀ n, ∃ k > n, k = a k) ∧ (infinitely_many_n (λ x, a x > x)) :=
sorry

end baron_is_boasting_l184_184107


namespace smallest_number_of_people_seated_l184_184798

theorem smallest_number_of_people_seated (num_chairs : ℕ) (h : num_chairs = 72) : 
  (∃ N : ℕ, N = 18 ∧ ∀ any_seating_pattern : ℕ → ℕ, (∑ i in range N, any_seating_pattern i = num_chairs ∧
  (∀ j < num_chairs, (any_seating_pattern (j mod N) = 1 → any_seating_pattern (((j+1) mod num_chairs) mod N) = 0 ∧
  any_seating_pattern (((j+2) mod num_chairs) mod N) = 0 ∧ any_seating_pattern (((j+3) mod num_chairs) mod N) = 0))
  → any_seating_pattern (((j+4) mod num_chairs) mod N) = 1))
  :=
sorry

end smallest_number_of_people_seated_l184_184798


namespace transformed_function_is_odd_l184_184983

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184983


namespace rate_of_interest_per_annum_l184_184808

theorem rate_of_interest_per_annum (SI P : ℝ) (T : ℕ) (hSI : SI = 4016.25) (hP : P = 10040.625) (hT : T = 5) :
  (SI * 100) / (P * T) = 8 :=
by 
  -- Given simple interest formula
  -- SI = P * R * T / 100, solving for R we get R = (SI * 100) / (P * T)
  -- Substitute SI = 4016.25, P = 10040.625, and T = 5
  -- (4016.25 * 100) / (10040.625 * 5) = 8
  sorry

end rate_of_interest_per_annum_l184_184808


namespace mixed_groups_count_l184_184001

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l184_184001


namespace rectangle_area_l184_184649

noncomputable def area_rectangle (AD DC : ℝ) : ℝ:= AD * DC

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def equation_line (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := 
  let slope := (y2 - y1) / (x2 - x1) in
  (slope, y1 - slope * x1)

noncomputable def intersection (m1 c1 m2 c2 : ℝ) : ℝ × ℝ := 
  let x := (c2 - c1) / (m1 - m2) in
  (x, m1 * x + c1)

noncomputable def area_quadrilateral (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  0.5 * ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

theorem rectangle_area (AD DC : ℝ) (midE : E = midpoint 0 0 (2 * DC) 0)
                       (line_BE : ∃ m c, m * 2 * DC + c = DC ∧ equation_line 2 * DC DC (DC, 0) = (m, c))
                       (intersect_AF : ∃ x y, (x, y) = intersection (½) 0 (-1) (2 * DC))
                       (area_AFED : area_quadrilateral  0 AD (fst intersect_AF) (snd intersect_AF) DC 0 0 0 = 54)
                       (longer_side: AD = 2 * DC) : 
  area_rectangle (2 * √162) (√162) = 324 := 
sorry

end rectangle_area_l184_184649


namespace quadratic_inequality_k_range_l184_184633

variable (k : ℝ)

theorem quadratic_inequality_k_range (h : ∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) :
  -1 < k ∧ k < 0 := by
sorry

end quadratic_inequality_k_range_l184_184633


namespace MrMartinBought2Cups_l184_184693

theorem MrMartinBought2Cups (c b : ℝ) (x : ℝ) (h1 : 3 * c + 2 * b = 12.75)
                             (h2 : x * c + 5 * b = 14.00)
                             (hb : b = 1.5) :
  x = 2 :=
sorry

end MrMartinBought2Cups_l184_184693


namespace new_determinant_l184_184672

variable {α β γ D : ℝ}
variables {a b c : ℝ^3}

-- Given determinant of the original matrix
def orig_determinant (a b c : ℝ^3) : ℝ := a.dot_product (b.cross_product c)

-- Given the determinant value D
axiom det_d : orig_determinant a b c = D

-- Prove the determinant of the new matrix
theorem new_determinant :
  let new_matrix :=
    matrix.of_cols (λ _ : fin 3, [α • a + b, β • b + c, γ • c + a].nth _) in
  matrix.det new_matrix = α * β * γ * D :=
sorry

end new_determinant_l184_184672


namespace same_function_D_l184_184098

theorem same_function_D (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  ∀ x : ℝ, x = log a (a ^ x) :=
by
  sorry

end same_function_D_l184_184098


namespace abc_plus_2_gt_a_plus_b_plus_c_l184_184586

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (ha : -1 < a) (ha' : a < 1) (hb : -1 < b) (hb' : b < 1) (hc : -1 < c) (hc' : c < 1) :
  a * b * c + 2 > a + b + c :=
sorry

end abc_plus_2_gt_a_plus_b_plus_c_l184_184586


namespace product_of_primes_l184_184759

def largest_one_digit_primes : list ℕ := [3, 5, 7]
def second_largest_two_digit_prime : ℕ := 89

theorem product_of_primes : (largest_one_digit_primes.product * second_largest_two_digit_prime = 9345) :=
by
    -- Placeholder for proof
    sorry

end product_of_primes_l184_184759


namespace smallest_positive_k_l184_184760

theorem smallest_positive_k (k a n : ℕ) (h_pos : k > 0) (h_cond : 3^3 + 4^3 + 5^3 = 216) (h_eq : k * 216 = a^n) (h_n : n > 1) : k = 1 :=
by {
    sorry
}

end smallest_positive_k_l184_184760


namespace letter_2023rd_in_sequence_l184_184459

noncomputable def sequence : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A', 'A', 'B']

def letter_at_index (n : Nat) : Char :=
  sequence.get! (n % sequence.length)

theorem letter_2023rd_in_sequence : letter_at_index 2022 = 'C' :=
by
  -- Proof skipped
  sorry

end letter_2023rd_in_sequence_l184_184459


namespace isosceles_triangle_EF_ge_one_l184_184647

theorem isosceles_triangle_EF_ge_one
  (A B C E F : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space F]
  [dist : has_dist ℝ]
  (isosceles_triangle : isosceles B A C)
  (point_E_on_AB : E ∈ segment A B)
  (point_F_on_AC : F ∈ segment A C)
  (AE_eq_CF : dist A E = dist C F)
  (BC_eq_2 : dist B C = 2) 
  : dist E F ≥ 1 :=
sorry

end isosceles_triangle_EF_ge_one_l184_184647


namespace find_x_coordinate_l184_184914

theorem find_x_coordinate {m n : ℝ} (h1 : n^2 = 4 * m)
  (h2 : abs (m + 1) / sqrt 2 = dist (m, n) (5, 0)) : m = 3 :=
by
  sorry

end find_x_coordinate_l184_184914


namespace external_contour_length_l184_184413

-- Define the settings and conditions of the problem
variables (a : ℝ) -- radius of the circles
variables (A B C D : Type)
variables [has_dist A] [has_dist B] [has_dist C] [has_dist D]
variables (dist_AD : dist A D = 2 * a)
variables (dist_DC : dist D C = 2 * a)
variables (dist_BC : dist B C = 2 * a)
variables (dist_AB : dist A B = 2 * a)
variables (dist_AC : dist A C = 2 * a)

-- Define the points of tangency
variables (M N P Q : Type)
variables [has_inner_product (M → ℝ)] [has_inner_product (N → ℝ)]
variables [has_inner_product (P → ℝ)] [has_inner_product (Q → ℝ)]

-- State the theorem about the length of the external contour
theorem external_contour_length : external_contour_length A B C D M N P Q a = 6 * π * a :=
by sorry

end external_contour_length_l184_184413


namespace Gwen_still_has_money_in_usd_l184_184890

open Real

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def usd_gift : ℝ := 5.00
noncomputable def eur_gift : ℝ := 20.00
noncomputable def usd_spent_on_candy : ℝ := 3.25
noncomputable def eur_spent_on_toy : ℝ := 5.50

theorem Gwen_still_has_money_in_usd :
  let eur_conversion_to_usd := eur_gift / exchange_rate
  let total_usd_received := usd_gift + eur_conversion_to_usd
  let usd_spent_on_toy := eur_spent_on_toy / exchange_rate
  let total_usd_spent := usd_spent_on_candy + usd_spent_on_toy
  total_usd_received - total_usd_spent = 18.81 :=
by
  sorry

end Gwen_still_has_money_in_usd_l184_184890


namespace num_socks_in_machine_l184_184095

-- Definition of the number of people who played the match
def num_players : ℕ := 11

-- Definition of the number of socks per player
def socks_per_player : ℕ := 2

-- The goal is to prove that the total number of socks in the washing machine is 22
theorem num_socks_in_machine : num_players * socks_per_player = 22 :=
by
  sorry

end num_socks_in_machine_l184_184095


namespace find_base_b_l184_184875

theorem find_base_b (b : ℝ) (h : log b 1024 = -5 / 3) : b = 1 / 64 := 
sorry

end find_base_b_l184_184875


namespace smallest_possible_value_l184_184287

theorem smallest_possible_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^2 + b^2) / (a * b) + (a * b) / (a^2 + b^2) ≥ 2 :=
sorry

end smallest_possible_value_l184_184287


namespace volume_of_prism_l184_184785

-- Define the conditions
variables {a α β : ℝ}
variables (isosceles_prism : Prop)
variable [h1 : a > 0]
variable [h2 : α < π]
variable [h3 : β < π / 2]

-- Define the theorem to be proven
theorem volume_of_prism 
  (ab_eq_ac : isosceles_prism → a = a)
  (angle_CAB_eq_alpha : isosceles_prism → ∠(A, C, B) = α)
  (B1_equidistant : isosceles_prism → B1 is equidistant from all sides of the base)
  (B1B_angle_beta : isosceles_prism → ∠(B1, B, plane(ABC)) = β) :
  volume_of_prism = 
    (a^3 * sin(α) * sin(α / 2) * tan(β)) / (2 * cos((π - α) / 4)) := sorry

end volume_of_prism_l184_184785


namespace geometric_sequence_sum_of_bn_l184_184216

variable (m : ℝ) (h_m_pos : m > 0) (h_m_neq_one : m ≠ 1) 

def f (x : ℝ) : ℝ := Real.log x / Real.log m
noncomputable def a_n (n : ℕ) : ℝ := m^(2*n + 2)
noncomputable def b_n (n : ℕ) : ℝ := a_n m n * f m (a_n m n)
noncomputable def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, b_n m (k + 1)

theorem geometric_sequence (n : ℕ) : a_n m (n + 1) / a_n m n = m^2 := 
by
  sorry

theorem sum_of_bn (n : ℕ) (h_m_sqrt_two : m = Real.sqrt 2) : S_n m n = 2^(n + 3) * n :=
by
  sorry

end geometric_sequence_sum_of_bn_l184_184216


namespace sam_found_pennies_l184_184388

-- Define the function that computes the number of pennies Sam found given the initial and current amounts of pennies
def find_pennies (initial_pennies current_pennies : Nat) : Nat :=
  current_pennies - initial_pennies

-- Define the main proof problem
theorem sam_found_pennies : find_pennies 98 191 = 93 := by
  -- Proof steps would go here
  sorry

end sam_found_pennies_l184_184388


namespace student_arrangement_adjustment_l184_184714

/-- 
Given 10 students arranged in two rows, with 3 students in the front row and 7 students in the back row,
this theorem states that the number of different ways to adjust the arrangement by selecting 2 students from 
the back row to move to the front row, while maintaining the relative order of other students, 
is equal to \( C_{7}^{2}A_{5}^{2} \).
-/
theorem student_arrangement_adjustment :
  let total_students := 10 in
  let front_row := 3 in
  let back_row := 7 in
  let selected_students := 2 in
  let arrangements_in_front_row := 5 in
  (choose back_row selected_students) * (arrangements_in_front_row.choose selected_students) = 
  (C back_row selected_students) * (A arrangements_in_front_row selected_students) := sorry

end student_arrangement_adjustment_l184_184714


namespace eval_f_neg2_l184_184027

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l184_184027


namespace least_number_of_cans_l184_184504

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def leastCansRequired (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_liters := gcd maaza (gcd pepsi sprite)
  maaza / gcd_liters + pepsi / gcd_liters + sprite / gcd_liters

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 40) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  leastCansRequired maaza pepsi sprite = 69 :=
by
  rw [h_maaza, h_pepsi, h_sprite]
  unfold leastCansRequired
  sorry

end least_number_of_cans_l184_184504


namespace angle_between_unit_vectors_l184_184347

variable {V : Type*} [inner_product_space ℝ V]

/-- Given two unit vectors a and b, where a + 3b and 4a - 3b are orthogonal,
the angle θ between a and b is cos⁻¹(5/9). -/
theorem angle_between_unit_vectors (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
  (h_orthog : ⟪a + 3 • b, 4 • a - 3 • b⟫ = 0) :
  real.angle.cos_norm a b = real.arccos (5 / 9) :=
by
-- skip the proof
sorry

end angle_between_unit_vectors_l184_184347


namespace polynomial_sum_coeff_l184_184873

-- Definitions for the polynomials given
def poly1 (d : ℤ) : ℤ := 15 * d^3 + 19 * d^2 + 17 * d + 18
def poly2 (d : ℤ) : ℤ := 3 * d^3 + 4 * d + 2

-- The main statement to prove
theorem polynomial_sum_coeff :
  let p := 18
  let q := 19
  let r := 21
  let s := 20
  p + q + r + s = 78 :=
by
  sorry

end polynomial_sum_coeff_l184_184873


namespace part1_solution_part2_solution_l184_184675

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem part1_solution :
  (∃ x : ℝ, (f x)^2 = f x + 2) ↔ x = Real.log 2 := 
by sorry

theorem part2_solution :
  (∀ x : ℝ, x + b ≤ f x) ↔ b ≤ 1 := 
by sorry

end part1_solution_part2_solution_l184_184675


namespace eval_floor_abs_neg_45_7_l184_184179

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184179


namespace compound_interest_l184_184293

-- Defining the constants and values
def t := 6
def r := 10
def SI := 1800
def P := 3000  -- Derived as P = (SI * 100) / (r * t)
def CI := P * (Real.pow (1 + r / 100) t - 1)

theorem compound_interest (h : CI = 2314.68) : 
  CI = 2314.68 :=
by
  -- Proof not required as per instructions
  sorry

end compound_interest_l184_184293


namespace remaining_cubes_l184_184773

-- The configuration of the initial cube and the properties of a layer
def initial_cube : ℕ := 10
def total_cubes : ℕ := 1000
def layer_cubes : ℕ := (initial_cube * initial_cube)

-- The proof problem: Prove that the remaining number of cubes is 900 after removing one layer
theorem remaining_cubes : total_cubes - layer_cubes = 900 := 
by 
  sorry

end remaining_cubes_l184_184773


namespace number_of_dissimilar_terms_in_expansion_l184_184847

theorem number_of_dissimilar_terms_in_expansion (a b c d : ℕ) : 
  (∑ (i j k l : ℕ) in finset.range(9), if i + j + k + l = 8 then 1 else 0) = 165 := 
sorry

end number_of_dissimilar_terms_in_expansion_l184_184847


namespace smallest_even_n_l184_184801

theorem smallest_even_n (n : ℕ) :
  (∃ n, 0 < n ∧ n % 2 = 0 ∧ (∀ k, 1 ≤ k → k ≤ n / 2 → k = 2213 ∨ k = 3323 ∨ k = 6121) ∧ (2^k * (k!)) % (2213 * 3323 * 6121) = 0) → n = 12242 :=
sorry

end smallest_even_n_l184_184801


namespace no_perfect_cover_l184_184117

-- Define the types of cells in the grid
inductive Cell : Type
| White : Cell
| Black : Cell

-- Define a 1x1 monomino
structure Monomino :=
(pos : ℤ × ℤ) -- Position of the monomino

-- Define the condition for placing monominoes on the grid
def no_shared_edges_or_vertices (m1 m2 : Monomino) : Prop :=
  m1.pos ≠ m2.pos ∧ 
  (m1.pos.1 + 1 <> m2.pos.1 ∧ m1.pos.1 - 1 <> m2.pos.1) ∧ 
  (m1.pos.2 + 1 <> m2.pos.2 ∧ m1.pos.2 - 1 <> m2.pos.2)

-- Define a 1x2 domino
structure Domino :=
(start : ℤ × ℤ) -- Starting position of the domino
(horizontal : bool) -- True if horizontal, False if vertical

-- Define the condition for covering the grid with dominos
def coverable_with_dominos (remaining_cells : List (ℤ × ℤ)) : Prop :=
  ∃ (dominos : List Domino), 2 * dominos.length = remaining_cells.length

-- Main theorem
theorem no_perfect_cover 
(placed_monominoes : List Monomino) 
(h_no_shared_edges_or_vertices : ∀ (m1 m2 : Monomino), m1 ∈ placed_monominoes → m2 ∈ placed_monominoes → m1 ≠ m2 → no_shared_edges_or_vertices m1 m2) :
¬ (∃ remaining_cells, (Set.univ - (placed_monominoes.map Monomino.pos : Set (ℤ × ℤ)) = remaining_cells.to_set) ∧ coverable_with_dominos remaining_cells) :=
sorry

end no_perfect_cover_l184_184117


namespace min_value_seq_l184_184622

theorem min_value_seq (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 26) (h₂ : ∀ n, a (n + 1) - a n = 2 * n + 1) :
  ∃ m, (m > 0) ∧ (∀ k, k > 0 → (a k / k : ℚ) ≥ 10) ∧ (a m / m : ℚ) = 10 :=
by
  sorry

end min_value_seq_l184_184622


namespace closest_fraction_l184_184106

def fraction_of_medals_won := 17 / 100
def option_A := 1 / 4
def option_B := 1 / 5
def option_C := 1 / 6
def option_D := 1 / 7
def option_E := 1 / 8

theorem closest_fraction :
  min (abs (fraction_of_medals_won - option_A))
  ( 
    min (abs (fraction_of_medals_won - option_B))
    ( 
      min (abs (fraction_of_medals_won - option_C))
      (
        min (abs (fraction_of_medals_won - option_D))
        (abs (fraction_of_medals_won - option_E))
      )
    )
  )
= (abs (fraction_of_medals_won - option_C))
:= 
sorry

end closest_fraction_l184_184106


namespace sequence_100th_term_l184_184377

theorem sequence_100th_term
  (a : ℕ → ℝ)
  (h_initial : a 1 = 7)
  (h2 : a 2 = 7^1)
  (h3 : a 3 = 7 + 7^1)
  (h4 : a 4 = 7^2)
  (h5 : a 5 = 7^2 + 7)
  (h6 : a 6 = 7^2 + 7^1)
  (h7 : a 7 = 7^2 + 7^1 + 7)
  : a 100 = 7^6 + 7^5 + 7^2 :=
begin
  sorry,
end

end sequence_100th_term_l184_184377


namespace cos_of_angle_on_abs_neg_line_l184_184606

theorem cos_of_angle_on_abs_neg_line (α : ℝ) (x y : ℝ)
  (h₁ : y = -|x|) 
  (h₂ : α ∈ { α | ∃ x y : ℝ, y = -|x| ∧ x = cos α ∧ y = sin α }) : 
  cos α = ±(Real.sqrt 2 / 2) := 
sorry

end cos_of_angle_on_abs_neg_line_l184_184606


namespace number_of_sides_l184_184513

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end number_of_sides_l184_184513


namespace find_c_l184_184207

theorem find_c (c d : ℝ) (h : ∃ u v w : ℝ, 
  (u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u > 0 ∧ v > 0 ∧ w > 0) ∧ 
  (8 * u^3 + 6 * c * u^2 + 3 * d * u + c = 0 ∧
   8 * v^3 + 6 * c * v^2 + 3 * d * v + c = 0 ∧
   8 * w^3 + 6 * c * w^2 + 3 * d * w + c = 0) ∧ 
  log 3 (u * v * w) = 3) : 
  c = -216 :=
sorry

end find_c_l184_184207


namespace parents_present_l184_184742

theorem parents_present (pupils teachers total_people parents : ℕ)
  (h_pupils : pupils = 724)
  (h_teachers : teachers = 744)
  (h_total_people : total_people = 1541) :
  parents = total_people - (pupils + teachers) :=
sorry

end parents_present_l184_184742


namespace dwarfs_truthful_count_l184_184160

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184160


namespace McKenna_stuffed_animals_count_l184_184372

def stuffed_animals (M K T : ℕ) : Prop :=
  M + K + T = 175 ∧ K = 2 * M ∧ T = K + 5

theorem McKenna_stuffed_animals_count (M K T : ℕ) (h : stuffed_animals M K T) : M = 34 :=
by
  sorry

end McKenna_stuffed_animals_count_l184_184372


namespace possible_φ_values_count_l184_184725

noncomputable def shifted_sine_is_odd (φ : ℝ) : Prop :=
  g x = sin (2 * x - 2 * φ + π / 3) ∧
  ∀ x, g (-x) = -g (x)

theorem possible_φ_values_count (φ : ℝ) :
  (shifted_sine_is_odd φ ∧ 0 < φ ∧ φ < π) → (φ = π / 6 ∨ φ = 2 * π / 3) :=
begin
  sorry,
end

#check possible_φ_values_count

end possible_φ_values_count_l184_184725


namespace probability_comparison_l184_184072

-- Definitions of populations and city counts
variables (P_large P_urban : ℝ) (N_large N_total : ℕ)

-- Assumptions ensuring non-negativity and total counts
variables (hP_urban : 0 < P_urban)
variables (hN_total : 0 < N_total)

-- Theorem stating the comparison of probabilities
theorem probability_comparison (hP_large : 0 ≤ P_large) (hN_large : 0 ≤ N_large) 
  (hPA_gt_PB : (P_large / P_urban) > (N_large.to_real / N_total.to_real)) : 
  (P_large / P_urban) > (N_large.to_real / N_total.to_real) := 
by 
  -- Proof details
  sorry

end probability_comparison_l184_184072


namespace chord_length_range_l184_184919

-- Define the variables and conditions
variables (a b c k m : Real)
variables (P : ℝ × ℝ) (F1 F2 A B : ℝ × ℝ)
variable  (O : ℝ ℝ)
-- Introduction of the foci and properties of the ellipse.
variable (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
variable (ellipse_eq : ∀ x y, (x, y) ∈ ellipse → (x^2 / a^2 + y^2 / b^2 = 1))
variable (P_ell : P ∈ ellipse)

-- Orthogonality and diameter relationship
variable (F1F2_orth : dist P.1 F1.1 = 1)
variable (F1F2_diameter : diameter (circle O (dist F1 F2 / 2)) = dist F1 F2)
variable (PF1_dot : inner (P - F1) (F1 - F2) = 0)
variable (line_tangent : tangent (circle O (dist F1 F2 / 2)) (mk_affine_line k m))
variable (ellipse_intersect : line_intersect ellipse (mk_affine_line k m) A B)
variable (k_pos : k > 0)

-- Scalar product condition and length of the chord AB.
variable (OA_OB : inner (O - A) (O - B) = Real.toRat(λ))

noncomputable def chord_length (k : ℝ) (m : ℝ) : ℝ := 
  2 * Real.sqrt (2 * (k^4 + k^2) / (4 * (k^4 + k^2) + 1))

theorem chord_length_range (k m: ℝ) (hyp : 3/4 ≤ λ ∧ λ ≤ 4/5) :
  (1 / 2 ≤ k^2 ∧ k^2 ≤ 1) → (4 * 2.sqrt / 5 ≤ chord_length k m ∧ chord_length k m ≤ 2.sqrt / 2) :=
sorry

end chord_length_range_l184_184919


namespace fourth_month_sale_is_7200_l184_184803

-- Define the sales amounts for each month
def sale_first_month : ℕ := 6400
def sale_second_month : ℕ := 7000
def sale_third_month : ℕ := 6800
def sale_fifth_month : ℕ := 6500
def sale_sixth_month : ℕ := 5100
def average_sale : ℕ := 6500

-- Total requirements for the six months
def total_required_sales : ℕ := 6 * average_sale

-- Known sales for five months
def total_known_sales : ℕ := sale_first_month + sale_second_month + sale_third_month + sale_fifth_month + sale_sixth_month

-- Sale in the fourth month
def sale_fourth_month : ℕ := total_required_sales - total_known_sales

-- The theorem to prove
theorem fourth_month_sale_is_7200 : sale_fourth_month = 7200 :=
by
  sorry

end fourth_month_sale_is_7200_l184_184803


namespace correct_statements_count_l184_184929

-- Definitions for conditions
def statement1 : Prop := ∀ (X Y : Type), ¬(∀ (x : X) (y : Y), true → (∃ f : X → Y, y = f x))
def statement2 : Prop := ∀ (X Y : Type), ∀ (x : X) (y : Y), correlation(x, y) → functional_relationship(x, y) = false
def statement3 : Prop := ∀ (X Y : Type), ∀ (x : X) (y : Y), regression_analysis(x, y) → functional_relationship(x, y) = true
def statement4 : Prop := ∀ (X Y : Type), ∀ (x : X) (y : Y), regression_analysis(x, y) → correlation(x, y) = true

-- Proof problem: Prove the number of correct statements is 1
theorem correct_statements_count : (statement4) ∧ (¬statement1) ∧ (¬statement2) ∧ (¬statement3) → 1 := 
by
  sorry

end correct_statements_count_l184_184929


namespace largest_y_l184_184885

theorem largest_y (y : ℝ) (h : (⌊y⌋ / y) = 8 / 9) : y ≤ 63 / 8 :=
sorry

end largest_y_l184_184885


namespace prove_years_ago_l184_184078

-- Defining the variables A and X
variables (A X : ℕ)

-- Defining the conditions
def condition1 := A = 50
def condition2 := 5 * (A + 5) - 5 * (A - X) = A

-- Proof problem statement
theorem prove_years_ago (h1 : condition1) (h2 : condition2) : X = 5 :=
sorry

end prove_years_ago_l184_184078


namespace trajectory_is_ellipse_lambda_sum_constant_l184_184596

theorem trajectory_is_ellipse (x y : ℝ) (M N : ℝ × ℝ) 
  (slope_PM slope_PN : ℝ) (P: ℝ × ℝ)
  (hM : M = (√3, 0))
  (hN : N = (-√3, 0))
  (h : slope_PM * slope_PN = -2/3)
  (hx_P : P.1 ≠ √3 ∨ P.1 ≠ -√3) :
  (slope_PM = y / (x - √3)) → (slope_PN = y / (x + √3)) →
  (y ≠ 0) →
  (x^2 / 3 + y^2 / 2 = 1) :=
sorry

theorem lambda_sum_constant (x y : ℝ) (F1 F2 A B : ℝ × ℝ)
  (P : ℝ × ℝ) (λ₁ λ₂ : ℝ)
  (hF1 : F1 = (-1, 0))
  (hF2 : F2 = (1, 0))
  (hEQ1 : y = (λ₁ * (F1.2 - A.2)))
  (hEQ2 : y = (λ₂ * (F2.2 - B.2)))
  (ellipse_eq : 2 * x^2 + 3 * y^2 = 6)
  (λ₁_EQ : λ₁ = x + 2)
  (λ₂_EQ : λ₂ = -x + 2) :
  λ₁ + λ₂ = 4 :=
sorry

end trajectory_is_ellipse_lambda_sum_constant_l184_184596


namespace tetrahedron_volume_and_height_l184_184830

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A1 : Point := ⟨2, 3, 1⟩
def A2 : Point := ⟨4, 1, -2⟩
def A3 : Point := ⟨6, 3, 7⟩
def A4 : Point := ⟨7, 5, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def volume (A1 A2 A3 A4 : Point) : ℝ :=
(1 / 6) * real.abs (
  (A2.x - A1.x) * ((A3.y - A1.y) * (A4.z - A1.z) - (A3.z - A1.z) * (A4.y - A1.y)) -
  (A2.y - A1.y) * ((A3.x - A1.x) * (A4.z - A1.z) - (A3.z - A1.z) * (A4.x - A1.x)) +
  (A2.z - A1.z) * ((A3.x - A1.x) * (A4.y - A1.y) - (A3.y - A1.y) * (A4.x - A1.x))
)

def area (A1 A2 A3 : Point) : ℝ :=
(1 / 2) * real.sqrt (
  real.sq ((A2.y - A1.y) * (A3.z - A1.z) - (A2.z - A1.z) * (A3.y - A1.y)) +
  real.sq ((A2.z - A1.z) * (A3.x - A1.x) - (A2.x - A1.x) * (A3.z - A1.z)) +
  real.sq ((A2.x - A1.x) * (A3.y - A1.y) - (A2.y - A1.y) * (A3.x - A1.x))
)

def height (A1 A2 A3 A4 : Point) : ℝ :=
let V := volume A1 A2 A3 A4 in
let S := area A1 A2 A3 in
(3 * V) / S

theorem tetrahedron_volume_and_height :
  volume A1 A2 A3 A4 = (70 / 3) ∧ height A1 A2 A3 A4 = 5 :=
by
  sorry

end tetrahedron_volume_and_height_l184_184830


namespace part1_part2_part3_l184_184695

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

-- Part 1: Prove the 5th equation
theorem part1 : a 5 = (1 / (5 * 6)) ∧ a 5 = (1 / 5 - 1 / 6) :=
by sorry

-- Part 2: Prove the nth equation
theorem part2 (n : ℕ) : a n = 1 / (n * (n + 1)) ∧ a n = 1 / n - 1 / (n + 1) :=
by sorry

-- Part 3: Calculate the series sum
theorem part3 : (∑ n in Finset.range 2023, 1 / ((n+1) * (n+2))) = 2023 / 2024 :=
by sorry

end part1_part2_part3_l184_184695


namespace odd_function_check_l184_184932

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184932


namespace number_of_valid_a_l184_184360

noncomputable def d1 (a : ℤ) : ℤ :=
  a^2 + 3^a + a * 3^((a + 1) / 2)

noncomputable def d2 (a : ℤ) : ℤ :=
  a^2 + 3^a - a * 3^((a + 1) / 2)

theorem number_of_valid_a : (finset.filter (λ a, (d1 a * d2 a) % 7 = 0) (finset.range 101)).card = 86 :=
  sorry

end number_of_valid_a_l184_184360


namespace find_x_cubed_l184_184510

noncomputable def x_cubed_solution (x : ℝ) (a : ℝ) : ℝ :=
  if (a = 1.5) ∧ (real.cbrt (a - x^3) + real.cbrt (a + x^3) = 1) then 
    real.sqrt (136 / 27)
  else 
    0 -- Default value if conditions are not met (a placeholder, in reality, this won't be hit for valid inputs)

theorem find_x_cubed (x a : ℝ) (h_a : a = 1.5) (h_eq : real.cbrt (a - x^3) + real.cbrt (a + x^3) = 1) : 
  x^3 = real.sqrt (136 / 27) := by
  sorry

end find_x_cubed_l184_184510


namespace female_participation_fraction_l184_184211

noncomputable def fraction_of_females (males_last_year : ℕ) (females_last_year : ℕ) : ℚ :=
  let males_this_year := (1.10 * males_last_year : ℚ)
  let females_this_year := (1.25 * females_last_year : ℚ)
  females_this_year / (males_this_year + females_this_year)

theorem female_participation_fraction
  (males_last_year : ℕ) (participation_increase : ℚ)
  (males_increase : ℚ) (females_increase : ℚ)
  (h_males_last_year : males_last_year = 30)
  (h_participation_increase : participation_increase = 1.15)
  (h_males_increase : males_increase = 1.10)
  (h_females_increase : females_increase = 1.25)
  (h_females_last_year : females_last_year = 15) :
  fraction_of_females males_last_year females_last_year = 19 / 52 := by
  sorry

end female_participation_fraction_l184_184211


namespace quadratic_bound_l184_184682

theorem quadratic_bound (a b : ℝ) : ∃ x0 ∈ set.Icc (-1:ℝ) 1, |((λ x : ℝ, x^2 + a * x + b) x0)| + a ≥ 0 :=
by
  sorry

end quadratic_bound_l184_184682


namespace C1_general_form_C2_rectangular_form_intersection_distance_l184_184925

noncomputable def C1_parametric_1 (t : ℝ) : ℝ := (√5 / 5) * t
noncomputable def C1_parametric_2 (t : ℝ) : ℝ := (2 * √5 / 5) * t - 1

def C2_polar (θ : ℝ) : ℝ := 2 * Real.cos θ - 4 * Real.sin θ

theorem C1_general_form : 
  ∃ (x y : ℝ) (t : ℝ), (x = C1_parametric_1 t) ∧ (y = C1_parametric_2 t) ∧ (y - 2 * x + 1 = 0) := 
sorry

theorem C2_rectangular_form : 
  ∃ (x y : ℝ) (θ : ℝ), (Real.sqrt (x^2 + y^2) = C2_polar θ) ∧ ((x - 1)^2 + (y + 2)^2 = 5) := 
sorry

theorem intersection_distance :
  let center_distance := |-(2 : ℝ) - 2 + 1 / Real.sqrt 5|
  let AB_distance := 2 * Real.sqrt((Real.sqrt 5)^2 - (center_distance / Real.sqrt 5)^2)
  (AB_distance = (8 * Real.sqrt 5 / 5)) :=
sorry

end C1_general_form_C2_rectangular_form_intersection_distance_l184_184925


namespace unique_function_l184_184569

-- Define the function in the Lean environment
def f (n : ℕ) : ℕ := n

-- State the theorem with the given conditions and expected answer
theorem unique_function (f : ℕ → ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) < x * (1 + f y) + 2021) → (∀ x : ℕ, f x = x) :=
by
  intros h x
  -- Placeholder for the proof
  sorry

end unique_function_l184_184569


namespace cos_arcsin_eq_l184_184838

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l184_184838


namespace appointment_plans_count_l184_184898

theorem appointment_plans_count :
  let male_teachers := 5
  let female_teachers := 4
  let total_teachers := 3
  let total_classes := 3
  choose2_from5 := Nat.choose 5 2
  choose1_from4 := Nat.choose 4 1
  choose1_from5 := Nat.choose 5 1
  choose2_from4 := Nat.choose 4 2
  arrangements := 3.factorial
  (choose2_from5 * choose1_from4 * arrangements + choose1_from5 * choose2_from4 * arrangements) = 420 := 
by
  sorry

end appointment_plans_count_l184_184898


namespace reflection_equation_l184_184318

theorem reflection_equation (x y : ℝ) :
  (\forall x, y = (2 * x - 1) / (x - 1)) →
  (y = -x) →
  y = (-x - 1) / (x + 2) :=
sorry

end reflection_equation_l184_184318


namespace solution_set_l184_184550

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then 2^x - 4 else 2^{-x} - 4

theorem solution_set {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, x >= 0 → f x = 2^x - 4)
  : { x : ℝ | f x ≤ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_l184_184550


namespace geometric_sequence_general_formula_sum_first_n_terms_b_l184_184248

noncomputable def geometric_sequence (a : ℕ → ℕ) :=
  (∀ n, a n = 3^(n-1))

theorem geometric_sequence_general_formula
  {a : ℕ → ℕ} (h1: a 2 + a 3 + a 4 = 39)
  (h2 : a 5 = 2 * a 4 + 3 * a 3) :
  geometric_sequence a :=
sorry

noncomputable def sequence_b (b : ℕ → ℕ) (a : ℕ → ℕ) :=
  ∀ n, b n = n + a n

noncomputable def sum_b (T : ℕ → ℕ) (b : ℕ → ℕ) :=
  ∀ n, T n = ∑ k in finset.range n, b (k+1)

theorem sum_first_n_terms_b
  {b : ℕ → ℕ} (a : ℕ → ℕ) (T : ℕ → ℕ)
  (h1: sequence_b b a)
  (h2: geometric_sequence a) :
  sum_b T b → ∀ n, T n = (3^n + n^2 + n - 1) / 2 :=
sorry

end geometric_sequence_general_formula_sum_first_n_terms_b_l184_184248


namespace truthful_dwarfs_count_l184_184150

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l184_184150


namespace least_addition_to_palindrome_l184_184023

theorem least_addition_to_palindrome : 
  ∃ n : ℕ, is_palindrome (54321 + n) ∧ n = 54445 - 54321 := 
begin
  use 54445 - 54321,
  split,
  { 
    -- proof that (54321 + (54445 - 54321)) is a palindrome
    sorry 
  },
  { 
    -- proof that n = 54445 - 54321
    refl
  }
end

end least_addition_to_palindrome_l184_184023


namespace no_real_solutions_for_equation_l184_184126

theorem no_real_solutions_for_equation :
  ¬ (∃ x : ℝ, (2 * x - 3 * x + 7)^2 + 2 = -|2 * x|) :=
by 
-- proof will go here
sorry

end no_real_solutions_for_equation_l184_184126


namespace mod_product_l184_184713

theorem mod_product (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 50) : 
  173 * 927 % 50 = n := 
  by
    sorry

end mod_product_l184_184713


namespace sum_of_b_for_unique_solution_l184_184199

theorem sum_of_b_for_unique_solution :
  (∑ b in {b : ℝ | (b + 12)^2 - 4 * 3 * 10 = 0}, b) = 24 := by
  sorry

end sum_of_b_for_unique_solution_l184_184199


namespace football_players_count_l184_184305

-- Define the given conditions
def total_students : ℕ := 39
def long_tennis_players : ℕ := 20
def both_sports : ℕ := 17
def play_neither : ℕ := 10

-- Define a theorem to prove the number of football players is 26
theorem football_players_count : 
  ∃ (F : ℕ), F = 26 ∧ 
  (total_students - play_neither) = (F - both_sports) + (long_tennis_players - both_sports) + both_sports :=
by {
  sorry
}

end football_players_count_l184_184305


namespace transformed_function_is_odd_l184_184978

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184978


namespace speed_difference_l184_184735

variables (V_X13 V_Y14 D_X13 D_Y14 t : ℝ)

-- Definitions derived from conditions
def condition1 : Prop := t > 0
def condition2 : Prop := D_X13 = 2000
def condition3 : Prop := V_Y14 = 3 * V_X13
def condition4 (V_Z15 : ℝ) (D_Z15 : ℝ) (t_Z15 : ℝ) : Prop := 
  (V_Z15 = V_X13) ∧ (t_Z15 = t + 2) ∧ (D_Z15 = 5 * D_X13)

-- Prove the speed difference
theorem speed_difference (h_c1 : condition1) (h_c2 : condition2) (h_c3 : condition3) :
  V_Y14 - V_X13 = 2000 / t :=
by {
  sorry
}

end speed_difference_l184_184735


namespace largest_value_2_lt_x_lt_3_l184_184289

variable {x : ℝ}

theorem largest_value_2_lt_x_lt_3 (h : 2 < x ∧ x < 3) :
  ∀ y ∈ {x, x^2, 3*x, real.sqrt x, 1/x}, y ≤ x^2 := 
sorry

end largest_value_2_lt_x_lt_3_l184_184289


namespace find_a_l184_184637

-- Definitions derived from the conditions
def isPureImaginary (z : Complex) : Prop := z.re = 0
def complex_expr (a : ℝ) : Complex := ((a + Complex.i) * (1 + Complex.i)) / ((1 - Complex.i) * (1 + Complex.i))

-- The main statement
theorem find_a (a : ℝ) (h : isPureImaginary (complex_expr a)) : a = 1 :=
sorry

end find_a_l184_184637


namespace cannot_pay_exactly_500_can_pay_exactly_600_l184_184733

-- Defining the costs and relevant equations
def price_of_bun : ℕ := 15
def price_of_croissant : ℕ := 12

-- Proving the non-existence for the 500 Ft case
theorem cannot_pay_exactly_500 : ¬ ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 500 :=
sorry

-- Proving the existence for the 600 Ft case
theorem can_pay_exactly_600 : ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 600 :=
sorry

end cannot_pay_exactly_500_can_pay_exactly_600_l184_184733


namespace equilateral_triangle_l184_184727

theorem equilateral_triangle (ABC A' B' C' : Type)
  (incircle_touch : ∀ (s : set ABC), -- some condition representing the incircle touching ABC at A', B', and C'
    ∃ (I : ABC), -- center of the incircle
    -- represent the in-touch properties between incircle and sides of triangle ABC
    (I ∈ s) /\ (∀ (x : s), x ∉ {A', B', C'}) )
  (orthocenter_coincide : ∃ (H : ABC), -- orthocenter of triangles ABC and A'B'C' coincide
    H = orthocenter ABC /\ H = orthocenter (insert A' (insert B' (singleton C')))) :
  equilateral ABC :=
by
  sorry

end equilateral_triangle_l184_184727


namespace speed_ratio_l184_184281

variables (S S' : ℝ)
-- Conditions: usual time 32 minutes, slower time 40 minutes, same distance
def usual_time : ℝ := 32
def slower_time : ℝ := 40
def same_distance := S * usual_time = S' * slower_time

-- Theorem: ratio of slower speed to usual speed is 4/5
theorem speed_ratio (h : same_distance) : S' / S = 4 / 5 :=
sorry

end speed_ratio_l184_184281


namespace john_behind_steve_l184_184662

theorem john_behind_steve
  (vJ : ℝ) (vS : ℝ) (ahead : ℝ) (t : ℝ) (d : ℝ)
  (hJ : vJ = 4.2) (hS : vS = 3.8) (hA : ahead = 2) (hT : t = 42.5)
  (h1 : vJ * t = d + ahead)
  (h2 : vS * t + ahead = vJ * t - ahead) :
  d = 15 :=
by
  -- Proof omitted
  sorry

end john_behind_steve_l184_184662


namespace dwarfs_truthful_count_l184_184159

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184159


namespace dwarfs_truthful_count_l184_184158

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l184_184158


namespace total_fish_sold_l184_184441

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end total_fish_sold_l184_184441


namespace perfect_squares_less_than_500_with_digits_6_7_8_l184_184628

theorem perfect_squares_less_than_500_with_digits_6_7_8 : 
  { n : ℕ // n ^ 2 < 500 ∧ (n ^ 2 % 10 = 6 ∨ n ^ 2 % 10 = 7 ∨ n ^ 2 % 10 = 8) } = 4 :=
sorry

end perfect_squares_less_than_500_with_digits_6_7_8_l184_184628


namespace evaluate_expression_l184_184762

def a : ℕ := 3^1
def b : ℕ := 3^2
def c : ℕ := 3^3
def d : ℕ := 3^4
def e : ℕ := 3^10
def S : ℕ := a + b + c + d

theorem evaluate_expression : e - S = 58929 := 
by
  sorry

end evaluate_expression_l184_184762


namespace arithmetic_sequence_general_term_sequence_sum_formula_l184_184688

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_general_term :
  (∃ a_1 d : ℕ, (a 1 = a_1) ∧ (∀ n, a (n+1) = a n + d) ∧ 
    (sum (range 5).map a = 15) ∧
    let a2 := a 2, a6 := a 6, a8 := a 8 in
    ∃ r : ℕ, r > 1 ∧ 2 * a2 = r * a6 ∧ r * a6 = a8 + 1) →
  ∀ n, a n = n :=
begin
  sorry
end

noncomputable def b (n : ℕ) : ℕ :=
  2^n * a n

theorem sequence_sum_formula (T : ℕ → ℕ) :
  (∀ n, T n = (finset.range n).sum (λ i, b i)) →
  (∀ n, T n = (n-1) * 2^(n + 1) + 2) :=
begin
  sorry
end

end arithmetic_sequence_general_term_sequence_sum_formula_l184_184688


namespace angle_subtraction_l184_184592

theorem angle_subtraction (a b : ℝ) (h₁ : a = 13) (h₂ : b = 180 - (13 * 13)) : b = 11 :=
by {
  rw [h₁],
  linarith,
}

end angle_subtraction_l184_184592


namespace no_rational_numbers_satisfy_l184_184864

theorem no_rational_numbers_satisfy :
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
    (1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014) :=
by
  sorry

end no_rational_numbers_satisfy_l184_184864


namespace transformed_function_is_odd_l184_184962

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184962


namespace smallest_positive_period_f_max_and_min_values_f_l184_184269

def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 + x)) ^ 2 - Real.sqrt 3 * Real.cos (2 * x)

theorem smallest_positive_period_f : Real.Periodic f π :=
by
  sorry

theorem max_and_min_values_f : 
  let interval_start := π / 6
  let interval_end := π / 4
  ∃ a b : ℝ, a = 1 ∧ b = 2 ∧ 
  ∀ x ∈ Set.Icc interval_start interval_end, a ≤ f x ∧ f x ≤ b :=
by
  sorry

end smallest_positive_period_f_max_and_min_values_f_l184_184269


namespace odd_function_g_l184_184986

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184986


namespace problem1_problem2_l184_184828

-- Problem (1)
theorem problem1 : 
  (sin (2 * π / 3))^2 + cos π + tan (π / 4) - (cos (-11 * π / 6))^2 + sin (-7 * π / 6) = 1 / 2 := 
sorry

-- Problem (2)
theorem problem2 (α : ℝ) : 
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α)) = 
  -tan α := 
sorry

end problem1_problem2_l184_184828


namespace painting_stability_l184_184817

-- Define the nails and their colors
inductive Color
| Red | Blue | Green

structure Nail :=
(color : Color)

-- Define the nails
def a1 : Nail := ⟨Color.Red⟩
def a2 : Nail := ⟨Color.Red⟩
def a3 : Nail := ⟨Color.Blue⟩
def a4 : Nail := ⟨Color.Blue⟩
def a5 : Nail := ⟨Color.Green⟩
def a6 : Nail := ⟨Color.Green⟩

-- Define the condition when the painting falls
def painting_falls (n1 n2 : Nail) : Prop :=
(n1.color = Color.Red ∧ n2.color = Color.Blue) ∨
(n1.color = Color.Red ∧ n2.color = Color.Green) ∨
(n1.color = Color.Blue ∧ n1.color = Color.Green)

-- Define the condition for the painting to remain hanging
def painting_hangs_stable (nails : List Nail) : Prop :=
(∀ n, nails.contains n → (∃ c, n.color = c)) ∧
(∀ n1 n2, nails.contains n1 ∧ nails.contains n2 → n1.color = n2.color)

-- Prove that the arrangement satisfies the conditions
theorem painting_stability :
  ∀ nails : List Nail, -- set of pins
  (painting_falls nails[0] nails[1] ↔ 
   (nails[0].color ≠ nails[1].color)) → 
  (painting_hangs_stable nails ↔ 
   (nails[0].color = nails[1].color ∧ ∃ n, nails.contains n)) →
  sorry

end painting_stability_l184_184817


namespace coefficient_x2_in_expansion_l184_184323

theorem coefficient_x2_in_expansion :
  ∑ k in finset.range 3, binomial 2 k * (-1) ^ k * binomial 4 (2 - k) * (-1) ^ (2 - k) = -14 :=
by sorry

end coefficient_x2_in_expansion_l184_184323


namespace transformed_function_is_odd_l184_184961

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184961


namespace reaction_in_extracellular_fluid_l184_184412

-- Define each condition
def condition_A : Prop := ∃ (glucose lactate : Type), glucose → lactate
def condition_B : Prop := ∃ (proteins ribosomes : Type), proteins → ribosomes
def condition_C : Prop := ∃ (lactate sodium_bicarbonate sodium_lactate carbonic_acid : Type), 
  lactate → sodium_bicarbonate → sodium_lactate ∧ carbonic_acid
def condition_D : Prop := ∃ (pyruvate carbon_dioxide water : Type), 
  pyruvate → carbon_dioxide ∧ water

-- Define the condition that the biochemical reaction occurs in the extracellular fluid
def occurs_in_extracellular_fluid (reaction : Prop) : Prop := 
  reaction = condition_C

-- Statement to prove
theorem reaction_in_extracellular_fluid : occurs_in_extracellular_fluid condition_C := sorry

end reaction_in_extracellular_fluid_l184_184412


namespace smallest_yummy_integer_l184_184389

noncomputable def sequence_sum := (λ (a n : ℤ), n * (2 * a + n - 1) / 2)

theorem smallest_yummy_integer :
  ∃ B : ℤ, (∃ (a n : ℤ), sequence_sum a n = 500 ∧ a ≤ B ∧ B < a + n) ∧ 
  ∀ B' : ℤ, (∃ (a n : ℤ), sequence_sum a n = 500 ∧ a ≤ B' ∧ B' < a + n) → B ≤ B' :=
begin
  use -499,
  split,
  { use [-499, 999],
    split,
    { rw sequence_sum,
      norm_num
    },
    split,
    { norm_num
    },
    { norm_num
    }
  },
  { intros B hB,
    cases hB with a hB',
    cases hB' with n hB'',
    cases hB'' with hsum hB'',
    cases hB'' with ha hb,
    have hlower : a ≥ -499,
    { sorry
    },
    linarith
  }
end

end smallest_yummy_integer_l184_184389


namespace incorrect_D_l184_184625

-- Define the conditions and incorrect answer
variables (a b : Line)
variable (α : Plane)

-- condition: lines a and b are non-coplanar
axiom non_coplanar : ¬(∃ α, a ∈ α ∧ b ∈ α)

-- condition: lines a and b form an acute angle
axiom acute_angle : ∃ θ, 0 < θ ∧ θ < 90

-- proof goal: show that it is incorrect to assert "There exists a plane α such that a ∥ α and b ⧫ α"
theorem incorrect_D : (∃ α, (a ∥ α ∧ b ⧫ α)) → false :=
by
  sorry

end incorrect_D_l184_184625


namespace cone_cylinder_volume_ratio_l184_184113

theorem cone_cylinder_volume_ratio (r h_cylinder h_cone : ℝ) (hcyl : h_cylinder = 20) (hcone : h_cone = 10) (hr : r = 5) :
  (1/3 * π * r^2 * h_cone) / (π * r^2 * h_cylinder) = 1 / 6 := 
by 
  rw [hcyl, hcone, hr, ←mul_assoc, ←mul_assoc, ←mul_assoc, ←mul_assoc, ←mul_div, div_self, ←one_div, mul_comm (1/3:ℝ), mul_assoc, ←div_eq_mul_one_div, mul_one_div, mul_one_div] 
  exact mul_div_assoc' (1/3 * π) (5 ^ 2) 20 (10) 500 sorry

end cone_cylinder_volume_ratio_l184_184113


namespace hyperbola_eccentricity_bounds_l184_184544

theorem hyperbola_eccentricity_bounds (a b : ℝ) (h0_a : 0 < a) (h0_b : 0 < b)
  (h_hyper_eq : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1) → 
                      ∀ c, |(c, b^2 / a) - (c, -b^2 / a)| ≥ 3 / 5 * |(c, b*c / a) - (c, -b*c / a)|) :
  let e := sqrt (1 + (b^2 / a^2)) in e ≥ 5 / 4 := 
sorry

end hyperbola_eccentricity_bounds_l184_184544


namespace transformed_function_is_odd_l184_184968

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184968


namespace sum_of_first_2000_terms_l184_184908

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 2) = a (n + 1) - a n

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
∑ i in finset.range n, a i

theorem sum_of_first_2000_terms (a : ℕ → ℤ) (h₁ : sequence a) (h₂ : a 2 = 1) (h₃ : sum_of_first_n_terms a 1999 = 2000) :
  sum_of_first_n_terms a 2000 = 2001 := sorry

end sum_of_first_2000_terms_l184_184908


namespace maximize_expression_l184_184356

noncomputable def max_value (c d : ℝ) : ℝ :=
  (3 / 2) * (c^2 + d^2)

theorem maximize_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ y : ℝ, 3 * (c - y) * (y + real.sqrt (y^2 + d^2)) = max_value c d :=
sorry

end maximize_expression_l184_184356


namespace points_covered_by_semicircle_l184_184309

/-- Given a right-angled triangle with hypotenuse of 1 unit and angles 30°, 60°, and 90°, 
    and given 25 arbitrary points inside this triangle, prove that there exist at least 
    9 points among them that can be covered by a semicircle with a radius of 3/10. -/
theorem points_covered_by_semicircle :
  ∀ (T : Triangle) (points : Fin 25 → Point),
    T.right_triangle 
    → T.hypotenuse = 1 
    → T.angle30 
    → T.angle60 
    → T.angle90 
    → ∃ (S : Finset Point), S.card = 9 ∧ semicircle_with_radius_3_10_covers S :=
by
  sorry

def Triangle : Type := sorry
def Point : Type := sorry
def semicircle_with_radius_3_10_covers (S : Finset Point) : Prop := sorry
instance (T : Triangle) : right_triangle T := sorry
instance (T : Triangle) : angle30 T := sorry
instance (T : Triangle) : angle60 T := sorry
instance (T : Triangle) : angle90 T := sorry

end points_covered_by_semicircle_l184_184309


namespace edge_length_of_cubical_box_l184_184821

noncomputable def volume_of_cube (edge_length_cm : ℝ) : ℝ :=
  edge_length_cm ^ 3

noncomputable def number_of_cubes : ℝ := 8000
noncomputable def edge_of_small_cube_cm : ℝ := 5

noncomputable def total_volume_of_cubes_cm3 : ℝ :=
  volume_of_cube edge_of_small_cube_cm * number_of_cubes

noncomputable def volume_of_box_cm3 : ℝ := total_volume_of_cubes_cm3
noncomputable def edge_length_of_box_m : ℝ :=
  (volume_of_box_cm3)^(1 / 3) / 100

theorem edge_length_of_cubical_box :
  edge_length_of_box_m = 1 := by 
  sorry

end edge_length_of_cubical_box_l184_184821


namespace banana_cost_l184_184497

theorem banana_cost (x : ℚ) : 
  (let bananas := 16 * 12 * 12 in
   let cost_per_sixpences := bananas * x / 6 in
   let fiver := 5 * 20 * 12 in
   let bananas_per_fiver := fiver / x in
   cost_per_sixpences = 1 / 2 * bananas_per_fiver) → 
  x = 1.25 := by
  let bananas := 16 * 12 * 12
  let cost_per_sixpences := bananas * x / 6
  let fiver := 5 * 20 * 12
  let bananas_per_fiver := fiver / x
  sorry

end banana_cost_l184_184497


namespace selection_with_minimum_one_boy_l184_184645

theorem selection_with_minimum_one_boy (n_boys n_girls : ℕ) (h_boys : n_boys = 6) (h_girls : n_girls = 4) :
  (2^(n_boys + n_girls) - 2^n_girls) = 1008 :=
by
  rw [h_boys, h_girls]
  calc
    2^(6 + 4) - 2^4 = 2^10 - 2^4 : by rw [add_comm]
                  ... = 1024 - 16 : by norm_num
                  ... = 1008     : by norm_num

end selection_with_minimum_one_boy_l184_184645


namespace valid_parameterizations_l184_184546

def line := set (ℝ × ℝ)
def parametric_form (p d : ℝ × ℝ) := {xy : ℝ × ℝ | ∃ t : ℝ, xy = (p.1 + t * d.1, p.2 + t * d.2)}

def pointA : ℝ × ℝ := (0, -4)
def pointB : ℝ × ℝ := (4/3, 0)
def dir := (1, 3)

def param_opt_A : line := parametric_form (0, -4) (1, 3)
def param_opt_B : line := parametric_form (4/3, 0) (-1, -3)
def param_opt_C : line := parametric_form (-2, -10) (2, 6)
def param_opt_D : line := parametric_form (-1, 1) (1/3, 1)
def param_opt_E : line := parametric_form (4, -4) (3/10, 1/10)

def line_equation (xy : ℝ × ℝ) : Prop := xy.2 = 3 * xy.1 - 4

theorem valid_parameterizations : 
  (∀ xy : ℝ × ℝ, xy ∈ param_opt_A ↔ line_equation xy) ∧
  (∀ xy : ℝ × ℝ, xy ∈ param_opt_B ↔ line_equation xy) ∧
  (∀ xy : ℝ × ℝ, xy ∈ param_opt_C ↔ line_equation xy) ∧
  ¬ (∀ xy : ℝ × ℝ, xy ∈ param_opt_D ↔ line_equation xy) ∧
  (∀ xy : ℝ × ℝ, xy ∈ param_opt_E ↔ line_equation xy) := 
  sorry

end valid_parameterizations_l184_184546


namespace set_diff_equality_l184_184343

-- Definitions of sets A and B
def A := {x : ℝ | x > 4}
def B := {x : ℝ | -6 < x ∧ x < 6}

-- Definitions of set differences
def set_diff (S T : set ℝ) := {x | x ∈ S ∧ x ∉ T}

-- Desired property to prove
theorem set_diff_equality :
  set_diff A (set_diff A B) = set_diff B (set_diff B A) :=
sorry

end set_diff_equality_l184_184343


namespace find_omega_l184_184614

theorem find_omega (ω : ℝ) (hω : ω > 0)
  (hmin : ∀ x ∈ set.Icc (0:ℝ) (Real.pi / 6),
                   (sin (ω * x) + sqrt 3 * cos (ω * x)) ≥ -1) :
  ω = 5 := by
  sorry

end find_omega_l184_184614


namespace floor_abs_neg_45_7_l184_184182

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184182


namespace area_of_transformed_region_l184_184345

-- Define the region area and transformation matrix
def area_T : ℝ := 15
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![3, -2],
  ![4, 1]
]

-- Define a function to compute the determinant of a 2x2 matrix
def det_2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

-- Calculating determinant of the given matrix
def det_transformation_matrix : ℝ := det_2x2 transformation_matrix

-- Define the area of T'
def area_T' := area_T * |det_transformation_matrix|

-- The theorem to be proved
theorem area_of_transformed_region : area_T' = 165 := by
  unfold area_T'
  unfold area_T
  unfold transformation_matrix
  unfold det_transformation_matrix
  unfold det_2x2
  sorry  -- Proof of the calculations

end area_of_transformed_region_l184_184345


namespace second_alloy_weight_l184_184648

variable {x : ℝ}

-- Conditions
def c1 := (cr_alloy1 : ℝ) := 0.12
def c2 := (cr_alloy2 : ℝ) := 0.10
def weight_alloy1 := 15.0
def cr_new_alloy := 0.106
def weight_new_alloy := weight_alloy1 + x

-- Condition equations
def eq1 := 0.12 * weight_alloy1 + 0.1 * x = 0.106 * (weight_alloy1 + x)

-- Theorem to prove
theorem second_alloy_weight : eq1 → x = 35 :=
by
  intro h
  sorry

end second_alloy_weight_l184_184648


namespace greatest_product_of_slopes_l184_184750

theorem greatest_product_of_slopes (m_1 m_2 : ℝ) (h1 : m_2 = 3 * m_1) (h2 : abs ((m_2 - m_1) / (1 + m_1 * m_2)) = 1 / real.sqrt 3) : 
  m_1 * m_2 = 1 :=
sorry

end greatest_product_of_slopes_l184_184750


namespace courier_packages_l184_184503

theorem courier_packages (yesterday today : ℕ) (h1 : yesterday = 80) (h2 : today = 2 * yesterday) :
  yesterday + today = 240 :=
by
  rw [h1, h2]
  norm_num

end courier_packages_l184_184503


namespace area_of_triangle_l184_184112

-- Define the sides of the triangle
def a : ℝ := 65
def b : ℝ := 60
def c : ℝ := 25

-- Define the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Use Heron's formula to express the area of the triangle
def triangle_area : ℝ := (s * (s - a) * (s - b) * (s - c)).sqrt

-- Statement of the theorem
theorem area_of_triangle : triangle_area = 750 := by
  sorry

end area_of_triangle_l184_184112


namespace genuine_items_count_l184_184444

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l184_184444


namespace evaluate_expression_l184_184872

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l184_184872


namespace definite_integral_evaluation_l184_184559

def integrand (x : ℝ) : ℝ := x + cos (2 * x)

theorem definite_integral_evaluation : ∫ (x : ℝ) in -π/2..π/2, integrand x = 0 :=
by
  sorry

end definite_integral_evaluation_l184_184559


namespace books_count_l184_184457

theorem books_count (books_per_box : ℕ) (boxes : ℕ) (total_books : ℕ) 
  (h1 : books_per_box = 3)
  (h2 : boxes = 8)
  (h3 : total_books = books_per_box * boxes) : 
  total_books = 24 := 
by 
  rw [h1, h2] at h3
  exact h3

end books_count_l184_184457


namespace log_eq_two_solves_integers_l184_184428

/-- The set of solutions of the equation log₁₀(a² - 15a) = 2 consists of two integers. -/
theorem log_eq_two_solves_integers :
  ∃ (a1 a2 : ℤ), (log 10 (a1^2 - 15 * a1) = 2 ∧ log 10 (a2^2 - 15 * a2) = 2) ∧ a1 ≠ a2 :=
sorry

end log_eq_two_solves_integers_l184_184428


namespace find_salary_J_l184_184050

variables {J F M A May : ℝ}
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (h3 : May = 6500)

theorem find_salary_J : J = 5700 :=
by
  sorry

end find_salary_J_l184_184050


namespace g_recurrence_l184_184120

/-- Definition of g(n) --/
def g (n : ℕ) : ℝ :=
  (2 + Real.sqrt 2) / 4 * (1 + Real.sqrt 2) ^ n + (2 - Real.sqrt 2) / 4 * (1 - Real.sqrt 2) ^ n

/-- Theorem to prove the relationship between g(n+1) and g(n-1) in terms of g(n) --/
theorem g_recurrence (n : ℕ) : g (n + 1) - g (n - 1) = 2 * g n := by
  sorry

end g_recurrence_l184_184120


namespace ball_reaches_less_than_5_l184_184791

noncomputable def height_after_bounces (initial_height : ℕ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (ratio ^ bounces)

theorem ball_reaches_less_than_5 (initial_height : ℕ) (ratio : ℝ) (k : ℕ) (target_height : ℝ) (stop_height : ℝ) 
  (h_initial : initial_height = 500) (h_ratio : ratio = 0.6) (h_target : target_height = 5) (h_stop : stop_height = 0.1) :
  ∃ n, height_after_bounces initial_height ratio n < target_height ∧ 500 * (0.6 ^ 17) < stop_height := by
  sorry

end ball_reaches_less_than_5_l184_184791


namespace allison_marbles_l184_184093

theorem allison_marbles (A B C : ℕ) (h1 : B = A + 8) (h2 : C = 3 * B) (h3 : C + A = 136) : 
  A = 28 :=
by
  sorry

end allison_marbles_l184_184093


namespace peter_and_susan_dollars_l184_184697

theorem peter_and_susan_dollars :
  (2 / 5 : Real) + (1 / 4 : Real) = 0.65 := 
by
  sorry

end peter_and_susan_dollars_l184_184697


namespace inv_5_mod_31_l184_184567

theorem inv_5_mod_31 : ∃ k : ℕ, 0 ≤ k ∧ k < 31 ∧ (5 * k) % 31 = 1 :=
begin
  use [25],
  split,
  exact le_refl 25,
  split,
  exact dec_trivial,
  exact dec_trivial
end

end inv_5_mod_31_l184_184567


namespace shaded_area_correct_l184_184325

noncomputable def radius_ADB := 2
noncomputable def radius_BEC := 3
noncomputable def radius_DFE := (radius_ADB + radius_BEC) / 2

noncomputable def area_of_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * Real.pi * radius^2

noncomputable def area_ADB := area_of_semicircle radius_ADB
noncomputable def area_BEC := area_of_semicircle radius_BEC
noncomputable def area_DFE := area_of_semicircle radius_DFE

noncomputable def shaded_area := area_ADB + area_BEC - area_DFE

theorem shaded_area_correct : shaded_area = 3.375 * Real.pi :=
by
  have h1 : area_ADB = 2 * Real.pi, by sorry
  have h2 : area_BEC = (9 / 2) * Real.pi, by sorry
  have h3 : area_DFE = 3.125 * Real.pi, by sorry
  have h4 : shaded_area = 2 * Real.pi + (9 / 2) * Real.pi - 3.125 * Real.pi, by sorry
  show shaded_area = 3.375 * Real.pi, by sorry

end shaded_area_correct_l184_184325


namespace maximize_sum_l184_184390

theorem maximize_sum :
  ∃ (x : ℕ → ℝ), x 1 = 1 ∧
  (∀ n < 100, 0 ≤ x (n + 1) ∧ x (n + 1) ≤ 2 * x n) ∧
  (S = Sum (λ i, if i % 2 = 0 then x i else -x i)) ∧
  (S = 2^98 - 2^97 + 2^96 - ... + 2^2 - 2^1 + 2^0) := 
begin
  sorry
end

end maximize_sum_l184_184390


namespace perp_bisector_MN_l184_184327

-- Define the conditions for the problem
variables {A B C D M N P : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D]
variables [affine_space ℝ M] [affine_space ℝ N] [affine_space ℝ P]

-- Assume given lengths are equal
variables (h1 : dist B C = dist A D)
-- Assume M is the midpoint of AD and N is the midpoint of BC
variables (hM : midpoint ℝ A D M) (hN : midpoint ℝ B C N)
-- Assume P is the intersection of the perpendicular bisectors of AB and CD
variables (hP1 : is_perp_bisector ℝ P A B) (hP2 : is_perp_bisector ℝ P C D)

-- Define the theorem to be proved
theorem perp_bisector_MN :
  is_perp_bisector ℝ P M N :=
sorry -- Proof to be filled in later

end perp_bisector_MN_l184_184327


namespace number_of_groupings_l184_184752

theorem number_of_groupings (n m : ℕ) (h1 : n = 8) (h2 : m = 2) :
  let total_choices := 2^n in
  let invalid_choices := 2 in
  let valid_choices := total_choices - invalid_choices in
  valid_choices = 254 :=
by {
  sorry
}

end number_of_groupings_l184_184752


namespace number_of_truthful_dwarfs_is_correct_l184_184137

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l184_184137


namespace transformed_function_is_odd_l184_184964

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184964


namespace option_A_correct_l184_184097

def f_A (x : ℝ) := sin (2 * x + π / 2)
def has_period_pi (f : ℝ → ℝ) := ∀ x, f (x + π) = f x
def is_monotone_on (f : ℝ → ℝ) (interval : Set ℝ) := ∀ x y ∈ interval, x ≤ y → f x ≤ f y

theorem option_A_correct :
  has_period_pi f_A ∧ is_monotone_on f_A (Set.Icc (π / 2) π) :=
sorry

end option_A_correct_l184_184097


namespace sum_of_roots_l184_184288

theorem sum_of_roots (p q : ℝ) (h_eq : 2 * p + 3 * q = 6) (h_roots : ∀ x : ℝ, x ^ 2 - p * x + q = 0) : p = 2 := by
sorry

end sum_of_roots_l184_184288


namespace value_of_S_10_l184_184491

noncomputable theory
open_locale classical

-- Definitions based on conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : arithmetic_seq a
axiom h2 : sum_seq a S
axiom h3 : a 3 = 16
axiom h4 : S 20 = 20

-- Question and answer
theorem value_of_S_10 : S 10 = 110 :=
sorry

end value_of_S_10_l184_184491


namespace cos_squared_alpha_minus_pi_over_4_val_l184_184238

-- Define the condition
def sin_2alpha (α : ℝ) : Prop := Math.sin (2 * α) = 1 / 3

-- Define the question
def cos_squared_alpha_minus_pi_over_4 (α : ℝ) : ℝ := Math.cos (α - Real.pi / 4) ^ 2

-- The proof statement
theorem cos_squared_alpha_minus_pi_over_4_val (α : ℝ) (h : sin_2alpha α) :
  cos_squared_alpha_minus_pi_over_4 α = 2 / 3 :=
sorry

end cos_squared_alpha_minus_pi_over_4_val_l184_184238


namespace transformed_function_is_odd_l184_184967

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184967


namespace fraction_of_students_claim_but_enjoys_soccer_l184_184533

theorem fraction_of_students_claim_but_enjoys_soccer (
  (total_students : ℕ) 
  (enjoy_soccer_percentage : ℝ) 
  (enjoy_soccer_say_dont_percentage : ℝ) 
  (dont_enjoy_soccer_admit_percentage : ℝ)
  (claim_dont_enjoy_soccer_but_enjoy_fraction : ℝ)
) :
  total_students > 0 ∧
  enjoy_soccer_percentage = 0.70 ∧
  enjoy_soccer_say_dont_percentage = 0.25 ∧
  dont_enjoy_soccer_admit_percentage = 0.85 →
  claim_dont_enjoy_soccer_but_enjoy_fraction = 35 / 86 :=
by
  sorry

end fraction_of_students_claim_but_enjoys_soccer_l184_184533


namespace sin_decreasing_in_intervals_l184_184707

theorem sin_decreasing_in_intervals {a b : ℝ} (h : a < b) :
  (a ∈ set.Icc (0 : ℝ) (real.pi / 2) ∧ b ∈ set.Icc (0 : ℝ) (real.pi / 2) ∨
   a ∈ set.Icc (real.pi) (3 * real.pi / 2) ∧ b ∈ set.Icc (real.pi) (3 * real.pi / 2)) →
  (a - real.sin a < b - real.sin b) :=
by sorry

end sin_decreasing_in_intervals_l184_184707


namespace transformed_function_is_odd_l184_184971

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184971


namespace cubic_polynomials_exist_l184_184877

-- Define that P and Q are cubic polynomials with real coefficients
def is_cubic_poly (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (P Q : ℝ → ℝ) : Prop :=
  (∀ x, x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 → P x = 0 ∨ P x = 1) ∧
  ((P 1 = 0 ∨ P 2 = 1) → Q 1 = 1 ∧ Q 3 = 1) ∧
  ((P 2 = 0 ∨ P 4 = 0) → Q 2 = 0 ∧ Q 4 = 0) ∧
  ((P 3 = 1 ∨ P 4 = 1) → Q 1 = 0)

-- Define the specific cubic polynomials R1 to R6
noncomputable def R1 : ℝ → ℝ := λ x, -1/2 * x^3 + 7/2 * x^2 - 7 * x + 4
noncomputable def R2 : ℝ → ℝ := λ x, 1/2 * x^3 - 4 * x^2 + 19/2 * x - 6
noncomputable def R3 : ℝ → ℝ := λ x, -1/6 * x^3 + 3/2 * x^2 - 13/3 * x + 4
noncomputable def R4 : ℝ → ℝ := λ x, -2/3 * x^3 + 5 * x^2 - 34/3 * x + 8
noncomputable def R5 : ℝ → ℝ := λ x, -1/2 * x^3 + 4 * x^2 - 19/2 * x + 7
noncomputable def R6 : ℝ → ℝ := λ x, 1/3 * x^3 - 5/2 * x^2 + 31/6 * x - 2

-- Define the set of polynomial pairs
def polynomial_pairs : List (ℝ → ℝ × ℝ → ℝ) :=
  [(R2, R4), (R3, R1), (R3, R3), (R3, R4), (R4, R1), (R5, R1), (R6, R4)]

-- The main statement to prove
theorem cubic_polynomials_exist :
  ∀ (P Q : ℝ → ℝ), is_cubic_poly P → is_cubic_poly Q → satisfies_conditions P Q →
    (P, Q) ∈ polynomial_pairs :=
by
  intros P Q hP hQ cond
  sorry

end cubic_polynomials_exist_l184_184877


namespace necessary_but_not_sufficient_l184_184632

theorem necessary_but_not_sufficient (a : ℝ) (ha : a > 1) : a^2 > a :=
sorry

end necessary_but_not_sufficient_l184_184632


namespace ellipse_range_k_l184_184261

theorem ellipse_range_k (k : ℝ) : 
  (∃ (x y : ℝ) (hk : \(\frac{x^2}{3+k} + \frac{y^2}{2-k} = 1\)), (3 + k > 0) ∧ (2 - k > 0) ∧ (3+k ≠ 2-k)) ↔ 
  k ∈ set.Ioo (-3) (-1/2) ∪ set.Ioo (-1/2) 2 := 
sorry

end ellipse_range_k_l184_184261


namespace probability_two_digit_greater_than_40_l184_184705

theorem probability_two_digit_greater_than_40 : 
  let digits := [1, 2, 3, 4, 5]
  let two_digit_numbers := (digits.product digits).filter (λ pair, pair.fst ≠ pair.snd)
  let favorable_outcomes := two_digit_numbers.filter (λ pair, pair.fst * 10 + pair.snd > 40)
  let probability := (favorable_outcomes.length : ℝ) / (two_digit_numbers.length : ℝ)
  probability = 2 / 5 :=
by {
  sorry
}

end probability_two_digit_greater_than_40_l184_184705


namespace odd_function_check_l184_184949

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184949


namespace john_belt_size_l184_184334

theorem john_belt_size (inches_to_feet : Real := 1 / 10) (feet_to_cm : Real := 25) (waist_inches : Real := 42) :
  Real.round (waist_inches * inches_to_feet * feet_to_cm) = 105 :=
by
  sorry

end john_belt_size_l184_184334


namespace not_decreasing_on_neg1_1_l184_184818

def f (x : ℝ) : ℝ := x^2 - x + 1

theorem not_decreasing_on_neg1_1 : ¬ (∀ x ∈ set.Ioo (-1 : ℝ) (1 : ℝ), f (x) > f (x + 1)) :=
by sorry

end not_decreasing_on_neg1_1_l184_184818


namespace typist_speeds_l184_184557

noncomputable def num_pages : ℕ := 72
noncomputable def ratio : ℚ := 6 / 5
noncomputable def time_difference : ℚ := 1.5

theorem typist_speeds :
  ∃ (x y : ℚ), (x = 9.6 ∧ y = 8) ∧ 
                (num_pages / x - num_pages / y = time_difference) ∧
                (x / y = ratio) :=
by
  -- Let's skip the proof for now
  sorry

end typist_speeds_l184_184557


namespace function_properties_l184_184772

theorem function_properties (k : ℤ) :
  let f : ℝ → ℝ := λ x, sin (2 * x - π / 3)
  in
  -- Statement A: The period is π
  (∀ x, f (x + π) = f x) ∧
  -- Statement B: The increasing interval is [kπ-π/12, kπ+5π/12]
  (∀ k : ℤ, ∀ x ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12), (f (x + ε) > f x) ∧ (f (x - ε) < f x) for all sufficiently small ε > 0) ∧
  -- Statement C: The graph is symmetric about the point (-π/3, 0)
  (f (-π / 3) = 0) ∧ 
  -- Statement D: The graph is not symmetric about the line x=2π/3
  (¬ ∀ x, f (x) = f (2 * π / 3 - x))
  :=
sorry

end function_properties_l184_184772


namespace transformed_function_is_odd_l184_184985

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184985


namespace range_of_m_l184_184204

theorem range_of_m (m : ℝ) : (∀ x : ℝ, exp (|2 * x + 1|) + m ≥ 0) → m ≥ -1 :=
by
  sorry

end range_of_m_l184_184204


namespace ratio_of_inradii_l184_184328

theorem ratio_of_inradii (A B C D : Point) (AC BC AB AD BD r_a r_b : ℝ) 
  (h1 : triangle ABC)
  (h2 : AC = 5)
  (h3 : BC = 12)
  (h4 : AB = 13)
  (h5 : segment_cd_bisects_right_angle ABC D r_a r_b)
  : r_a / r_b = 15 / 44 :=
sorry

end ratio_of_inradii_l184_184328


namespace first_three_decimal_digits_l184_184715

theorem first_three_decimal_digits :
  ∀ (x y r : ℝ), x = 10^2002 → y = 1 → r = 10 / 7 → |x| > |y| →
  let expansion := x^r + r * x^(r-1) * y + (r * (r - 1) / 2) * x^(r-2) * y^2 + (r * (r - 1) * (r - 2) / 6) * x^(r-3) * y^3
  in (10 ^ 2002 + 1)^(10 / 7) = 10^2860 + (1.428571 * 10^858) := by
  intros x y r h1 h2 h3 h4 expansion
  sorry

end first_three_decimal_digits_l184_184715


namespace quadrilateral_is_trapezoid_l184_184588

open EuclideanGeometry

variables {A B C D M N : Point}

def is_midpoint (M : Point) (P Q : Point) : Prop :=
  P + Q = 2 • M

def is_trapezoid (A B C D : Point) : Prop :=
  parallel (line_through A B) (line_through C D) ∨
  parallel (line_through A D) (line_through B C)

theorem quadrilateral_is_trapezoid
  (hABCD : quadrilateral A B C D)
  (hM : is_midpoint M A D)
  (hN : is_midpoint N B C)
  (h_length : dist M N = (dist A B + dist C D) / 2) :
  is_trapezoid A B C D :=
begin
  sorry
end

end quadrilateral_is_trapezoid_l184_184588


namespace li_bai_initial_wine_l184_184816

theorem li_bai_initial_wine (x : ℕ) 
  (h : (((((x * 2 - 2) * 2 - 2) * 2 - 2) * 2 - 2) = 2)) : 
  x = 2 :=
by
  sorry

end li_bai_initial_wine_l184_184816


namespace roots_cubic_reciprocal_sum_l184_184263

theorem roots_cubic_reciprocal_sum (a b c : ℝ) 
(h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 27) (h₃ : a * b * c = 18) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 13 / 24 :=
by
  sorry

end roots_cubic_reciprocal_sum_l184_184263


namespace all_x_multiples_of_5_l184_184805

theorem all_x_multiples_of_5 {a : Fin 2007 → ℕ} {x : Fin 2007 → ℤ}
  (h_a : ∀ i, a i ∈ {2, 3})
  (h_x : ∀ i, (a i * x i + x (i + 2)) % 5 = 0) :
  ∀ i, x i % 5 = 0 :=
by
  sorry

end all_x_multiples_of_5_l184_184805


namespace count_eight_letter_good_words_l184_184852

def is_good_word (w : List Char) : Prop :=
  ∀ i, i < w.length - 1 →
  (w[i] = 'A' → w[i+1] ≠ 'B') ∧
  (w[i] = 'B' → w[i+1] ≠ 'C') ∧
  (w[i] = 'C' → w[i+1] ≠ 'D') ∧
  (w[i] = 'D' → w[i+1] ≠ 'A')

theorem count_eight_letter_good_words : 
  {w : List Char // w.length = 8 ∧ is_good_word w}.length = 8748 :=
sorry

end count_eight_letter_good_words_l184_184852


namespace octahedron_path_count_l184_184511

-- Define the structure and properties of a regular octahedron
structure Octahedron where
  faces : Finset (Fin 8) -- A set of 8 faces
  top : Fin 8 -- The top face
  bottom : Fin 8 -- The bottom face
  middle_ring : Finset (Fin 8) -- The middle ring consisting of 6 faces
  adjacent : (Fin 8) → Finset (Fin 8) -- A function that provides adjacent faces

variables {O : Octahedron}

-- Conditions specific to the given problem
axiom regular_octahedron (H : O)
  (h_faces : O.faces = {0, 1, 2, 3, 4, 5, 6, 7})
  (h_top_bottom : O.top ∈ O.faces ∧ O.bottom ∈ O.faces)
  (h_middle_ring : O.middle_ring = {1, 2, 3, 4, 5, 6})
  (h_adjacent : ∀ (f : Fin 8), (f ∈ O.middle_ring) → (O.adjacent f).card ≤ 3) : 
  -- The proof problem we need to solve: calculate the number of valid paths
  ∃ (valid_paths : Finset (List (Fin 8))), valid_paths.card = 21

-- Main theorem asserting the number of valid paths from top to bottom face
theorem octahedron_path_count {H : O} : 
  ∃ valid_paths : Finset (List (Fin 8)), valid_paths.card = 21 :=
sorry -- The actual proof is not required per the instructions

end octahedron_path_count_l184_184511


namespace largest_possible_A_l184_184766

theorem largest_possible_A : ∃ A B : ℕ, 13 = 4 * A + B ∧ B < A ∧ A = 3 := by
  sorry

end largest_possible_A_l184_184766


namespace gcd_lcm_sum_l184_184829

theorem gcd_lcm_sum (a b : ℕ) (ha : a = 45) (hb : b = 4050) :
  Nat.gcd a b + Nat.lcm a b = 4095 := by
  sorry

end gcd_lcm_sum_l184_184829


namespace dwarfs_truthful_count_l184_184167

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184167


namespace math_proof_problem_l184_184226

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def b : ℝ := 2
noncomputable def c : ℝ := 2 * Real.sqrt 2

def ellipse_eq : ℝ → ℝ → Prop :=
  λ x y => (x^2 / (a^2)) + (y^2 / (b^2)) = 1

def triangle_PAB_area (P A B : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := P
  0.5 * Real.sqrt((x2 - x1)^2 + (y2 - y1)^2) * abs ((y3 - y1) - (x3 - x1))

problem_statement_1 : Prop :=
  ellipse_eq 2 0 ∧
  a = 2 * Real.sqrt 3 ∧ 
  b = 2 ∧ 
  (c / a) = Real.sqrt 6 / 3 → 
  ellipse_eq

problem_statement_2 : Prop :=
  ∃ A B, 
  (2, 0) = A ∧
  P = (-3, 2) ∧ 
  (line_eq := fun x => x + 2) ∧ 
  triangle_PAB_area P A B = 9 / 2

theorem math_proof_problem : 
  problem_statement_1 ∧ problem_statement_2 :=
by 
  sorry

end math_proof_problem_l184_184226


namespace question1_question2_l184_184219

theorem question1 (m : ℝ) (h₁ : m ≠ 0) :
  (∀ n : ℕ, n = 3 → ((x + m) ^ (2 * n + 1)).coeff 3 = ((m * x + 1) ^ (2 * n)).coeff 3) → m = 4 / 7 :=
sorry

theorem question2 (m : ℝ) (n : ℕ) (h₁ : n > 0) (h₂ : m ≠ 0) :
  (((x + m) ^ (2 * n + 1)).coeff n = ((m * x + 1) ^ (2 * n)).coeff n) →
  1 / 2 < (↑(n + 1) / (2 * ↑n + 1) : ℝ) ∧ (↑(n + 1) / (2 * ↑n + 1) : ℝ) ≤ 2 / 3 :=
sorry

end question1_question2_l184_184219


namespace nonagon_isosceles_triangle_count_l184_184222

theorem nonagon_isosceles_triangle_count (N : ℕ) (hN : N = 9) : 
  ∃(k : ℕ), k = 30 := 
by 
  have h := hN
  sorry      -- Solution steps would go here if we were proving it

end nonagon_isosceles_triangle_count_l184_184222


namespace total_gulbis_l184_184739

theorem total_gulbis (dureums fish_per_dureum : ℕ) (h1 : dureums = 156) (h2 : fish_per_dureum = 20) : dureums * fish_per_dureum = 3120 :=
by
  sorry

end total_gulbis_l184_184739


namespace inscribed_circle_radius_l184_184464

noncomputable def radius_of_inscribed_circle (d1 d2 : ℝ) : ℝ :=
let a := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
let area_diagonals := (d1 * d2) / 2 in
let radius := area_diagonals / (2 * a) in
radius

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 16 30 = 120 / 17 :=
by
  sorry

end inscribed_circle_radius_l184_184464


namespace n_plus_floor_sqrt2_plus1_pow_n_is_odd_l184_184382

theorem n_plus_floor_sqrt2_plus1_pow_n_is_odd (n : ℕ) (h : n > 0) : 
  Odd (n + ⌊(Real.sqrt 2 + 1) ^ n⌋) :=
by sorry

end n_plus_floor_sqrt2_plus1_pow_n_is_odd_l184_184382


namespace base8_subtraction_l184_184312

theorem base8_subtraction : nat.of_digits 8 [1, 2, 6] - nat.of_digits 8 [5, 7] = nat.of_digits 8 [0, 4, 7] :=
sorry

end base8_subtraction_l184_184312


namespace polynomial_degree_l184_184857

theorem polynomial_degree (b c e f g : ℂ) (hb : b ≠ 0) (hc : c ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) :
  degree ((X^5 + C b * X^8 + C c) * (C 2 * X^4 + C e * X^3 + C f) * (C 3 * X^2 + C g)) = 14 := 
  sorry

end polynomial_degree_l184_184857


namespace max_dragon_cards_and_sum_l184_184435

-- Declarations and definitions based on the conditions.
def is_card (n : ℕ) : Prop := n > 0
def is_dragon_card (n : ℕ) : Prop := n > 7

constant cards : Fin 20 → ℕ
constant sum_of_any_9_cards_le_63 : ∀ (f : Fin 9 → Fin 20), (∑ i in Finset.image Subtype.val (Finset.univ : Finset (Fin 9)), cards i) ≤ 63

-- Prove the required statements
theorem max_dragon_cards_and_sum :
  ∃ max_dragon_cards max_sum_of_dragon_cards,
    max_dragon_cards = 7 ∧
    (∀ (card_indices : Finset (Fin 20)), card_indices.cardines.count (λ i, is_dragon_card (cards i.val) ≥ max_dragon_cards) ∧
     (∑ i in card_indices.filter (λ i, is_dragon_card (cards i.val)), cards i) ≤ max_sum_of_dragon_cards) ∧
    max_sum_of_dragon_cards = 61 :=
by
  sorry

end max_dragon_cards_and_sum_l184_184435


namespace fishmonger_total_sales_l184_184439

theorem fishmonger_total_sales (first_week_sales : ℕ) (multiplier : ℕ) : 
  first_week_sales = 50 → multiplier = 3 → first_week_sales + first_week_sales * multiplier = 200 :=
by
  intros h_first h_mult
  rw [h_first, h_mult]
  simp
  sorry

end fishmonger_total_sales_l184_184439


namespace divide_AB_into_segments_l184_184303

noncomputable def point_K_divides_AB (O K : Point) (r : ℝ) (CH : ℝ) :=
  let A := (O.x - r, O.y)
  let B := (O.x + r, O.y)
  let C := (O.x, O.y + r)
  let D := (O.x, O.y - r)
  let chord_length := CH
  let AN := (K.x - O.x).abs
  let NB := (K.x - O.x + r).abs

  chord_length = 8 ∧ r = 5 ∧ ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (2 * r)^2 ∧
  AN.abs + NB.abs = 2 * r ∧
  (AN.abs (2 * r) ∧ NB.abs (2 * r) = chord_length^2

theorem divide_AB_into_segments (O K : Point) (r : ℝ) (CH : ℝ) :
  point_K_divides_AB O K r CH →
  r = 5 →
  CH = 8 →
  AN = 2 ∧ NB = 8 :=
by 
sorry

end divide_AB_into_segments_l184_184303


namespace perfect_square_n_eq_3_l184_184426

def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 20
  else if n = 1 then 30
  else 3 * a_seq (n - 1) - a_seq (n - 2)

theorem perfect_square_n_eq_3 :
  (∃ m : ℕ, 5 * a_seq n * a_seq (n + 1) + 1 = m^2) ↔ n = 3 :=
begin
  sorry
end

end perfect_square_n_eq_3_l184_184426


namespace angle_of_inclination_l184_184881

theorem angle_of_inclination (θ : ℝ) :
  (∃ x y : ℝ, 2 * x - 2 * real.sqrt 3 * y + 1 = 0) →
  θ = real.atan (real.sqrt 3 / 3) →
  θ = 30 :=
by
  intro h1 h2
  sorry

end angle_of_inclination_l184_184881


namespace inequality_always_holds_l184_184285

theorem inequality_always_holds (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end inequality_always_holds_l184_184285


namespace mean_score_is_74_l184_184480

theorem mean_score_is_74 (σ q : ℝ)
  (h1 : 58 = q - 2 * σ)
  (h2 : 98 = q + 3 * σ) :
  q = 74 :=
by
  sorry

end mean_score_is_74_l184_184480


namespace tilly_counts_total_stars_l184_184012

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l184_184012


namespace transformed_function_is_odd_l184_184970

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184970


namespace find_BC_length_l184_184657

open Real

variables {A B C D : Type}

structure Triangle (A B C : Type) :=
    (a b c : ℝ)   -- sides opposite to vertices A, B, C respectively

variables [triangle : Triangle ℝ]

noncomputable def BC_length (b c : ℝ) : ℝ :=
  sqrt (b * (b + c))

theorem find_BC_length (b c : ℝ) (AD_equivalence : triangle.a = triangle.b) :
  BC_length b c = sqrt (b * (b + c)) :=
sorry

end find_BC_length_l184_184657


namespace inv_5_mod_31_l184_184568

theorem inv_5_mod_31 : ∃ k : ℕ, 0 ≤ k ∧ k < 31 ∧ (5 * k) % 31 = 1 :=
begin
  use [25],
  split,
  exact le_refl 25,
  split,
  exact dec_trivial,
  exact dec_trivial
end

end inv_5_mod_31_l184_184568


namespace album_cost_l184_184793

-- Definition of the cost variables
variable (B C A : ℝ)

-- Conditions given in the problem
axiom h1 : B = C + 4
axiom h2 : B = 18
axiom h3 : C = 0.70 * A

-- Theorem to prove the cost of the album
theorem album_cost : A = 20 := sorry

end album_cost_l184_184793


namespace negation_of_existential_proposition_l184_184620

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, x > Real.sin x)) ↔ (∀ x : ℝ, x ≤ Real.sin x) :=
by 
  sorry

end negation_of_existential_proposition_l184_184620


namespace tan_sum_of_angles_l184_184920

open Real

theorem tan_sum_of_angles
    (α β : ℝ)
    (hα : α ∈ Ioo (-π/2) (π/2))
    (hβ : β ∈ Ioo (-π/2) (π/2))
    (h_root : ∀ t, t = tan α ∨ t = tan β ↔ t^2 - 3 * sqrt 3 * t + 4 = 0) :
    tan (α + β) = -sqrt 3 :=
by
  sorry

end tan_sum_of_angles_l184_184920


namespace remaining_perimeter_of_square_with_cutouts_l184_184806

theorem remaining_perimeter_of_square_with_cutouts 
  (square_side : ℝ) (green_square_side : ℝ) (init_perimeter : ℝ) 
  (green_square_perimeter_increase : ℝ) (final_perimeter : ℝ) :
  square_side = 10 → green_square_side = 2 →
  init_perimeter = 4 * square_side → green_square_perimeter_increase = 4 * green_square_side →
  final_perimeter = init_perimeter + green_square_perimeter_increase →
  final_perimeter = 44 :=
by
  intros hsquare_side hgreen_square_side hinit_perimeter hgreen_incr hfinal_perimeter
  -- Proof steps can be added here
  sorry

end remaining_perimeter_of_square_with_cutouts_l184_184806


namespace tangency_of_circles_l184_184669

theorem tangency_of_circles 
  {I I_A A B C L Y Z : Type} 
  [incircle : incenter I (triangle A B C)]
  [excircle : excenter I_A (triangle A B C opposite A)]
  [antipode : antipode A' (circle A B C)]
  [midpoint : midpoint L (arc B A C)]
  (inters_lb_ai : ∃ Y, intersection (line L B) (line I A))
  (inters_lc_ai : ∃ Z, intersection (line L C) (line I A)) :
  tangent (circle L Y Z) (circle A' I I_A) :=
sorry

end tangency_of_circles_l184_184669


namespace hyperbola_eccentricity_l184_184545

-- Definitions based on problem conditions
def hyperbola := {x : ℝ × ℝ // (x.1^2 / a^2) - (x.2^2 / b^2) = 1}
def parabola := {x : ℝ × ℝ // x.2^2 = 2 * p * x.1}
def focus_parabola := (p / 2, 0 : ℝ)
def intersection := {x : ℝ × ℝ // x ∈ hyperbola ∧ x ∈ parabola}
def on_x_axis (M F : ℝ × ℝ) := M.2 = 0

-- Theorem statement
theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) 
  (h_focus : (p / 2, 0 : ℝ) = (c, 0)) (h_intersect : intersection.1 ∈ first_quadrant)
  (h_on_axis : on_x_axis (intersection.1, (p / 2, 0 : ℝ))) :
  let e := c / a in e = sqrt 2 + 1 := sorry

end hyperbola_eccentricity_l184_184545


namespace problem_solutions_count_l184_184886

def countSolutions (x1 x2 x3 x4 : ℕ) : ℕ :=
  if (x1 + x2 + x3 + x4 = 23) ∧ (1 ≤ x1 ∧ x1 ≤ 9) ∧ (1 ≤ x2 ∧ x2 ≤ 8) ∧ (1 ≤ x3 ∧ x3 ≤ 7) ∧ (1 ≤ x4 ∧ x4 ≤ 6) then
    1
  else
    0

theorem problem_solutions_count :
  ∑ x1 in Finset.range 10, ∑ x2 in Finset.range 9, ∑ x3 in Finset.range 8, ∑ x4 in Finset.range 7, countSolutions x1 x2 x3 x4 = 115 :=
by
  sorry

end problem_solutions_count_l184_184886


namespace sum_of_x_coords_Q3_l184_184494

theorem sum_of_x_coords_Q3 {x_coords : Fin 44 → ℝ}
  (hQ1 : (∑ i, x_coords i) = 132) :
  let Q2_coords := λ i : Fin 44, (x_coords i + x_coords (i + 1)) / 2,
      Q3_coords := λ i : Fin 44, (Q2_coords i + Q2_coords (i + 1)) / 2 in
  (∑ i, Q3_coords i) = 132 := by
sorry

end sum_of_x_coords_Q3_l184_184494


namespace bus_wheel_radius_l184_184423

theorem bus_wheel_radius:
  let speed_kmh := 66
  let rpm := 175.15923566878982
  let speed_cmm := 110000  -- converting km/h to cm/min as 66 * 1000 * 100 / 60
  let circumference r := 2 * Real.pi * r
  let total_distance_per_min r := rpm * circumference r
  in ∃ r, total_distance_per_min r = speed_cmm ∧ r ≈ 99.88757742343576 := by
    sorry

end bus_wheel_radius_l184_184423


namespace concavity_and_inflection_points_l184_184571

noncomputable def gaussian_curve (x : ℝ) : ℝ := real.exp (-x^2)

theorem concavity_and_inflection_points :
  (∀ x, x < -1/real.sqrt 2 → (4*x^2 - 2) * real.exp (-x^2) > 0) ∧ 
  (∀ x, (-1/real.sqrt 2 < x ∧ x < 1/real.sqrt 2) → (4*x^2 - 2) * real.exp (-x^2) < 0) ∧ 
  (∀ x, x > 1/real.sqrt 2 → (4*x^2 - 2) * real.exp (-x^2) > 0) ∧
  (gaussian_curve (-1/real.sqrt 2) = 1/real.sqrt real.exp 1) ∧
  (gaussian_curve (1/real.sqrt 2) = 1/real.sqrt real.exp 1) :=
sorry

end concavity_and_inflection_points_l184_184571


namespace PU_squared_l184_184017

/-- 
Triangle PQR is inscribed in circle Ω with PQ = 6, QR = 8, PR = 4. 
The bisector of angle P meets side QR at S and circle Ω at a second point T. 
Let δ be the circle with diameter ST. Circles Ω and δ meet at T and a second point U.
--/
variables {P Q R S T U : Type}
variable [metric_space P]
variables (PQ QR PR : ℝ)
variables (Ω δ : set P)
variables (ST_diam : diameter (ST : P × P))

axiom PQ_length : PQ = 6
axiom QR_length : QR = 8
axiom PR_length : PR = 4

axiom bisector_P_meets_QR_at_S : metric.bisector (angle P) QR S
axiom bisector_P_meets_Ω_at_T : metric.circle Ω P Q R T
axiom circle_δ_diameter_ST : metric.circle δ S T
axiom circles_Ω_δ_meet_T_U : metric.circle_meets_at Ω δ T U

theorem PU_squared : ∀ P Q R S T U, PU^2 = 9 :=
by
  -- Assume the conditions mentioned in the problem
  assume P Q R S T U,
  sorry

end PU_squared_l184_184017


namespace committee_selection_problem_l184_184073

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection_problem :
  let total_ways := binomial 7 3 - binomial 5 1 + binomial 7 5 - binomial 5 3 in
  total_ways = 41 :=
by
  -- Step 1: Count the scenarios with Biff and Jacob on the committee
  -- total_ways := binomial 7 3 -- total without exclusion
  -- total_ways := total_ways - binomial 5 1 -- exclude the cases where Alice and Jane are both together

  -- Step 2: Count the scenarios with Biff and Jacob not on the committee
  -- total_ways := total_ways + binomial 7 5 -- total without exclusion
  -- total_ways := total_ways - binomial 5 3 -- exclude the cases where Alice and Jane are both together

  sorry

end committee_selection_problem_l184_184073


namespace train_length_l184_184525

def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (5 / 18)

theorem train_length (speed_km_per_hr : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_km_per_hr = 90 → time_s = 15 → length_m = 375 :=
by
  intros h1 h2
  have speed_m_per_s : ℝ := km_per_hr_to_m_per_s speed_km_per_hr
  rw [h1] at speed_m_per_s
  have speed_m_per_s_calculated : speed_m_per_s = 25 := by sorry
  rw [h2]
  have length_calculated : length_m = speed_m_per_s_calculated * time_s := by sorry
  exact length_calculated

#check train_length

end train_length_l184_184525


namespace most_likely_wins_l184_184449

theorem most_likely_wins {N : ℕ} (h : N > 0) :
  let p := 1 / 2
  let n := 2 * N
  let E := n * p
  E = N := 
by
  sorry

end most_likely_wins_l184_184449


namespace arithmetic_sequence_fifth_term_l184_184431

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15) 
  (h2 : a + 11 * d = 21) :
  a + 4 * d = 0 :=
by
  sorry

end arithmetic_sequence_fifth_term_l184_184431


namespace peter_parakeets_l184_184698

def parakeet_diet : Nat := 2 -- grams/day/parakeet
def parrot_diet : Nat := 14 -- grams/day/parrot
def finch_diet : Nat := parakeet_diet / 2 -- grams/day/finch
def weekly_seed : Nat := 266 -- grams/week required
def parrot_count : Nat := 2 -- number of parrots
def finch_count : Nat := 4 -- number of finches

theorem peter_parakeets : ∃ (parakeet_count : Nat), 
  let 
    parrot_weekly := parrot_diet * parrot_count * 7, 
    finch_weekly := finch_diet * finch_count * 7, 
    remaining_seed := weekly_seed - parrot_weekly - finch_weekly,
    parakeet_weekly := parakeet_diet * 7 
  in remaining_seed = parakeet_weekly * parakeet_count ∧ parakeet_count = 3 :=
by
  -- Proof goes here
  sorry

end peter_parakeets_l184_184698


namespace equal_circumradii_of_isosceles_split_l184_184080

   variables {A B C D : Type*} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D]

   -- Assume the triangle ABC is Isosceles with AB = AC
   def is_isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C

   -- Assume D is a point on BC and line AD intersects BC at D
   def line_intersects_base (A B C D : Point) : Prop := collinear A D B ∧ collinear A D C

   -- Define triangle circumradius
   def circumradius (A B C : Point) : ℝ := (dist A B * dist B C * dist C A) / (4 * triangle_area A B C)

   -- Main statement
   theorem equal_circumradii_of_isosceles_split (A B C D : Point) 
     [is_isosceles_triangle A B C] 
     [line_intersects_base A B C D] : 
     circumradius A B D = circumradius A C D :=
     sorry
   
end equal_circumradii_of_isosceles_split_l184_184080


namespace complex_number_problem_l184_184904

noncomputable def z (c : ℂ) (hz : z * (1 - 3 * complex.i) = 10) := z

theorem complex_number_problem (z : ℂ) (h : z * (1 - 3 * complex.i) = 10) :
  (|z| = real.sqrt 10) ∧ (z - 3 * (complex.of_real (real.cos (real.pi / 4)) 
  + complex.i * complex.of_real (real.sin (real.pi / 4)))^2 = 1) :=
by {
  sorry
}

end complex_number_problem_l184_184904


namespace constant_ratio_of_distances_equal_distances_if_equal_angles_l184_184486

-- Definitions and conditions
def circle := Type
def sphere := Type
def line := Type
def point := Type

variables (k : circle)
variables (α β γ : sphere)
variables (P P1 A B C A1 B1 C1 : point)
variables (g g1 : line)

-- Assumptions
axiom spheres_through_circle : ∀ s : sphere, s = α ∨ s = β ∨ s = γ → s.passes_through k
axiom lines_through_points : g.passes_through P ∧ g1.passes_through P1 ∧
                             (∀ s : sphere, s = α ∨ s = β ∨ s = γ → g.intersects_circle s k P A B C ∧
                                                                  g1.intersects_circle s k P1 A1 B1 C1)
axiom equal_angles : g.angle_with_plane k = g1.angle_with_plane k

-- Proof statements
theorem constant_ratio_of_distances :
  (AB / A1 B1) = (BC / B1 C1) := sorry

theorem equal_distances_if_equal_angles :
  (AB = A1 B1) ∧ (BC = B1 C1) := sorry

end constant_ratio_of_distances_equal_distances_if_equal_angles_l184_184486


namespace batsman_average_after_12th_innings_l184_184792

-- Defining the conditions
def before_12th_innings_average (A : ℕ) : Prop :=
11 * A + 80 = 12 * (A + 2)

-- Defining the question and expected answer
def after_12th_innings_average : ℕ := 58

-- Proving the equivalence
theorem batsman_average_after_12th_innings (A : ℕ) (h : before_12th_innings_average A) : after_12th_innings_average = 58 :=
by
sorry

end batsman_average_after_12th_innings_l184_184792


namespace sum_of_roots_l184_184352

variable (x1 x2 k m : ℝ)
variable (h1 : x1 ≠ x2)
variable (h2 : 4 * x1^2 - k * x1 = m)
variable (h3 : 4 * x2^2 - k * x2 = m)

theorem sum_of_roots (x1 x2 k m : ℝ) (h1 : x1 ≠ x2)
  (h2 : 4 * x1 ^ 2 - k * x1 = m) (h3 : 4 * x2 ^ 2 - k * x2 = m) :
  x1 + x2 = k / 4 := sorry

end sum_of_roots_l184_184352


namespace find_sum_a_b_l184_184235

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l184_184235


namespace cos_arcsin_eq_l184_184846

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l184_184846


namespace ellipse_range_k_l184_184260

theorem ellipse_range_k (k : ℝ) : 
  (∃ (x y : ℝ) (hk : \(\frac{x^2}{3+k} + \frac{y^2}{2-k} = 1\)), (3 + k > 0) ∧ (2 - k > 0) ∧ (3+k ≠ 2-k)) ↔ 
  k ∈ set.Ioo (-3) (-1/2) ∪ set.Ioo (-1/2) 2 := 
sorry

end ellipse_range_k_l184_184260


namespace units_digit_2_pow_2015_l184_184526

def units_digit_of_power_of_2 (n : ℕ) : ℕ :=
  [2, 4, 8, 6].getOrd (n % 4)

theorem units_digit_2_pow_2015 : units_digit_of_power_of_2 2015 = 8 :=
by
  sorry

end units_digit_2_pow_2015_l184_184526


namespace equal_segments_l184_184745

-- Definitions of the circles and given points and tangents
variables (O A B C : Point)
variables (ω1 ω2 : Circle)
variables (tangent : Line)

-- Given conditions
axiom center_ω1 : is_center O ω1
axiom passes_through_O : ω2.contains O
axiom intersects_at_A_and_B : ω1.contains A ∧ ω1.contains B ∧ ω2.contains A ∧ ω2.contains B
axiom tangent_at_B : is_tangent tangent ω2 B ∧ intersects tangent ω1 C

-- Prove that AB = BC
theorem equal_segments : distance A B = distance B C :=
sorry

end equal_segments_l184_184745


namespace problem_l184_184902

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem problem (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 :=
by
  sorry

end problem_l184_184902


namespace cos_arcsin_eq_l184_184839

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l184_184839


namespace ratio_of_areas_l184_184797

-- Definitions based on given conditions
def initial_square_side_length := 4
def first_circle_radius := initial_square_side_length / 2
def second_square_side_length := (2 * first_circle_radius) / Real.sqrt 2
def final_circle_radius := second_square_side_length / 2

-- Areas based on definitions
def initial_square_area := initial_square_side_length ^ 2
def final_circle_area := Real.pi * (final_circle_radius ^ 2)

-- The statement to prove
theorem ratio_of_areas :
  final_circle_area / initial_square_area = Real.pi / 8 :=
sorry

end ratio_of_areas_l184_184797


namespace find_multiplication_value_l184_184807

-- Define the given conditions
def student_chosen_number : ℤ := 63
def subtracted_value : ℤ := 142
def result_after_subtraction : ℤ := 110

-- Define the value he multiplied the number by
def multiplication_value (x : ℤ) : Prop := 
  (student_chosen_number * x) - subtracted_value = result_after_subtraction

-- Statement to prove that the value he multiplied the number by is 4
theorem find_multiplication_value : 
  ∃ x : ℤ, multiplication_value x ∧ x = 4 :=
by 
  -- Placeholder for the actual proof
  sorry

end find_multiplication_value_l184_184807


namespace value_of_OMON_l184_184242

open Real

-- Define the hyperbola equation and tangent notion
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

def line_tangent_to_hyperbola (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
hyperbola_eq P.1 P.2 ∧ ∀ x y, l x y → is_tangent_to_hyperbola P (x, y)

-- Define the asymptotes for the hyperbola
def asymptote_eq1 (x y : ℝ) : Prop := 2 * y = x
def asymptote_eq2 (x y : ℝ) : Prop := -2 * y = x

-- Define the condition that line intersects the asymptotes at points M and N
def line_intersects_asymptotes (l : ℝ → ℝ → Prop) (M N : ℝ × ℝ) : Prop :=
l M.1 M.2 ∧ asymptote_eq1 M.1 M.2 ∧ l N.1 N.2 ∧ asymptote_eq2 N.1 N.2

-- Define the final proof theorem
theorem value_of_OMON 
  (l : ℝ → ℝ → Prop) (P M N : ℝ × ℝ) 
  (h₁ : line_tangent_to_hyperbola l P) 
  (h₂ : line_intersects_asymptotes l M N) : 
  (M.1 * N.1) + (M.2 * N.2) = 3 := 
sorry

end value_of_OMON_l184_184242


namespace square_area_in_complex_plane_l184_184320

noncomputable def is_square_in_complex_plane (z z4 z5 : ℂ) : Prop :=
  (z4 = z^4 ∧ z5 = z^5) ∧ (z ≠ 0) ∧ (z5 - z = i * (z4 - z) ∨ z5 - z = -i * (z4 - z))

theorem square_area_in_complex_plane (z z4 z5 : ℂ) (h : is_square_in_complex_plane z z4 z5) : 
  |z4 - z|^2 = 1 :=
sorry

end square_area_in_complex_plane_l184_184320


namespace simplify_expression_l184_184703

def expression (x y : ℤ) : ℤ := 
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y)

theorem simplify_expression {x y : ℤ} (hx : x = 1) (hy : y = -2) :
  expression x y = -16 :=
by 
  -- This proof will involve algebraic manipulation and substitution.
  sorry

end simplify_expression_l184_184703


namespace value_of_f_neg2_l184_184030

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l184_184030


namespace dwarfs_truthful_count_l184_184165

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l184_184165


namespace angle_PQR_val_l184_184917

-- Definitions based on the conditions in the problem
variables (A B C P Q R : Type)
variables [linear_ordered_ring A] {AB PQ BC QR : A}
variable (angle : A → A → A → A)

-- Given conditions
axiom AB_parallel_PQ : AB = PQ
axiom BC_parallel_QR : BC = QR
axiom angle_ABC_eq_30 : angle A B C = 30

-- The proof problem statement
theorem angle_PQR_val :
  angle P Q R = 30 ∨ angle P Q R = 150 :=
sorry

end angle_PQR_val_l184_184917


namespace binomial_identity_l184_184694

-- Define the binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_identity (n : ℕ) :
  C (4*n + 1) 1 + C (4*n + 1) 5 + C (4*n + 1) 9 + ⋯ + C (4*n + 1) (4*n + 1) =
  2^(4*n - 1) + (-1)^n * 2^(2*n - 1) :=
sorry

end binomial_identity_l184_184694


namespace equal_number_of_digits_l184_184064

noncomputable def probability_equal_digits : ℚ := (20 * (9/16)^3 * (7/16)^3)

theorem equal_number_of_digits :
  probability_equal_digits = 3115125 / 10485760 := by
  sorry

end equal_number_of_digits_l184_184064


namespace right_triangle_inequality_l184_184702

theorem right_triangle_inequality {a b c : ℝ} (h₁ : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l184_184702


namespace odd_function_check_l184_184935

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184935


namespace ratio_QA_AR_l184_184071

section Geometry

/-- Definitions for the geometric setup -/
variables {P Q R O C A B K L : Type}
variables [MetricSpace PQR]
variables {s1 s2 s3 : ℝ}
variables (hInscribed : InscribedCircle O PQR)
variables (hTouchesPQ : Tangency O PQR PQ C)
variables (hTouchesQR : Tangency O PQR QR A)
variables (hTouchesRP : Tangency O PQR RP B)
variables (hIntersectBO : Intersects BO PQ K)
variables (hKQ : KQ = 1)
variables (hIntersectCO : Intersects CO PR L)
variables (hLR : LR = 2)
variables (hQR : QR = 11)

/-- The theorem to prove -/
theorem ratio_QA_AR : QA / AR = 5 / 6 :=
by
  sorry

end Geometry

end ratio_QA_AR_l184_184071


namespace total_handshakes_l184_184535

theorem total_handshakes (team1 team2 refs : ℕ) (players_per_team : ℕ) :
  team1 = 11 → team2 = 11 → refs = 3 → players_per_team = 11 →
  (players_per_team * players_per_team + (players_per_team * 2 * refs) = 187) :=
by
  intros h_team1 h_team2 h_refs h_players_per_team
  -- Now we want to prove that
  -- 11 * 11 + (11 * 2 * 3) = 187
  -- However, we can just add sorry here as the purpose is to write the statement
  sorry

end total_handshakes_l184_184535


namespace correct_propositions_l184_184607

variables (P1 P2 P3 P4 : Prop)

-- Defining the conditions
def Proposition_1 : Prop := 
  ∀ (l1 l2 : Line) (p : Plane), Parallel l1 l2 → AngleWithPlane l1 p = AngleWithPlane l2 p

def Proposition_2 : Prop := 
  ∀ (l1 l2 : Line) (p : Plane), AngleWithPlane l1 p = AngleWithPlane l2 p → Parallel l1 l2

def Proposition_3 : Prop := 
  ∀ (l : Line) (p1 p2 : Plane), Parallel p1 p2 → AngleWithLine p1 l = AngleWithLine p2 l

def Proposition_4 : Prop := 
  ∀ (l : Line) (p1 p2 : Plane), AngleWithLine p1 l = AngleWithLine p2 l → Parallel p1 p2

-- The proof problem statement
theorem correct_propositions : 
  (Proposition_1 P1 P2 P3 P4) ∧ (¬ Proposition_2 P1 P2 P3 P4) ∧ (Proposition_3 P1 P2 P3 P4) ∧ (¬ Proposition_4 P1 P2 P3 P4) :=
sorry

end correct_propositions_l184_184607


namespace find_inclination_angle_l184_184315

noncomputable theory

-- We define the problem by establishing the given conditions and asserting the proof of the desired result.
def problem (alpha : ℝ) : Prop :=
  ∃ M : (ℝ × ℝ), 
  M = (-2, -4) ∧ 
  (∃ (rho theta : ℝ),
    (rho * (Real.sin theta)^2 = 2 * (Real.cos theta)) ∧
    (∃ (A B: ℝ × ℝ),
      line_through M alpha ∧ 
      curve C rho theta ∧ 
      distance_between_points M A * distance_between_points M B = 40) → 
    alpha = (Real.pi / 4))

-- Placeholder definitions for line, given point, curve, and point distances
def line_through (M : (ℝ × ℝ)) (alpha : ℝ) : Prop := sorry
def curve (C: ℝ → ℝ → Prop) (rho theta: ℝ) : Prop := sorry
def distance_between_points (P Q : (ℝ × ℝ)) : ℝ := sorry 

-- Main theorem to prove
theorem find_inclination_angle : ∀ (alpha : ℝ), problem alpha := sorry

end find_inclination_angle_l184_184315


namespace length_of_train_correct_l184_184777

-- Definition for km/hr to m/s conversion
def km_per_hr_to_m_per_s (speed_km_hr : Float) : Float := 
  speed_km_hr * 1000 / 3600

-- The speed of the train in km/hr
def train_speed_km_hr : Float := 60.0

-- The time taken to cross the pole in seconds
def time_to_cross_pole_s : Float := 3.0

-- The calculated speed in m/s
def train_speed_m_s := km_per_hr_to_m_per_s train_speed_km_hr

-- The length of the train
def length_of_train := train_speed_m_s * time_to_cross_pole_s

-- Stating the theorem
theorem length_of_train_correct :
  abs (length_of_train - 50.01) < 0.01 := sorry

end length_of_train_correct_l184_184777


namespace omitted_digits_correct_l184_184757

theorem omitted_digits_correct :
  (287 * 23 = 6601) := by
  sorry

end omitted_digits_correct_l184_184757


namespace path_length_times_paths_number_l184_184409

def grid_dist (p1 p2 : ℕ × ℕ) : ℚ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_growing_path (path : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i < j → i < path.length → j < path.length → grid_dist (path.nth_le i sorry) (path.nth_le (i + 1) sorry) < grid_dist (path.nth_le j sorry) (path.nth_le (j + 1) sorry)

def max_points_in_growing_path (grid : List (ℕ × ℕ)) : ℕ :=
  13

def num_growing_paths_of_max_length (grid : List (ℕ × ℕ)) : ℕ :=
  32

theorem path_length_times_paths_number : 
∀ (grid : List (ℕ × ℕ)), 
  grid = [(i,j) | i in [0,1,2,3,4], j in [0,1,2,3,4]] →
  max_points_in_growing_path grid * num_growing_paths_of_max_length grid = 416 :=
begin
  intros grid hgrid,
  unfold max_points_in_growing_path num_growing_paths_of_max_length,
  rw [mul_comm],
  exact eq.refl 416,
end

end path_length_times_paths_number_l184_184409


namespace distance_from_sphere_center_to_triangle_plane_l184_184520

/-- Lean 4 statement of the given math problem -/
theorem distance_from_sphere_center_to_triangle_plane (r : ℝ) (s₁ s₂ s₃ : ℝ) (d : ℝ) 
  (hs₁ : s₁ = 13) (hs₂ : s₂ = 14) (hs₃ : s₃ = 15) (hr : r = 10) :
  (d = 6 * real.sqrt 2) :=
sorry

end distance_from_sphere_center_to_triangle_plane_l184_184520


namespace odd_function_g_l184_184990

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184990


namespace ratio_of_b_to_c_l184_184775

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- Given conditions:
-- 1. a is two years older than b
def condition1 : Prop := a = b + 2 

-- 2. b's age is given directly
def condition2 : Prop := b = 6

-- 3. The total of the ages of a, b, and c is 17
def condition3 : Prop := a + b + c = 17

-- The proof should show that the ratio of b's age to c's age is 2:1
theorem ratio_of_b_to_c (h1 : condition1) (h2 : condition2) (h3 : condition3) : b = 2 * c :=
by {
  sorry
}

end ratio_of_b_to_c_l184_184775


namespace problem_1_problem_2_l184_184266

-- Definitions
def line_l (m : ℝ) : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (2 + m) * p.1 + (1 - 2 * m) * p.2 + (4 - 3 * m) = 0

def fixed_point_M : ℝ × ℝ := (-1, -2)

def line_l1_through_M (p : ℝ × ℝ) : (ℝ × ℝ) → Prop :=
  λ (p1 : ℝ × ℝ), 2 * p1.1 + p1.2 + 4 = 0

-- Theorem Statements
theorem problem_1 : ∀ (m : ℝ), line_l m fixed_point_M := by
  sorry

theorem problem_2 : ∃ (k : ℝ) (l1 : (ℝ × ℝ) → Prop), 
  (∀ (p : ℝ × ℝ), l1 p → line_l1_through_M p) 
  ∧ (∃ (p1 p2 : ℝ × ℝ), l1 p1 ∧ p1 ≠ (0,0) ∧ l1 p2 ∧ p2 ≠ (0,0) ∧ 
      area_of_triangle p1 p2 = (1/2) * 4 * 2) 
    := by sorry

end problem_1_problem_2_l184_184266


namespace hyperbola_eccentricity_l184_184906

theorem hyperbola_eccentricity : 
  ∀ (a b : ℝ) (c : ℝ), 
    (c > 0) → 
    (b^2 = c^2 - a^2) → 
    (2 * b^2 = √2 * a * c) → 
    let e := c / a in 
    e = √2 :=
by
  intro a b c h_c_pos h_b_eq h_cond e_def_eq
  have e := c / a
  sorry

end hyperbola_eccentricity_l184_184906


namespace find_sample_size_l184_184643

theorem find_sample_size : ∃ n : ℕ, n ∣ 36 ∧ (n + 1) ∣ 35 ∧ n = 6 := by
  use 6
  simp
  sorry

end find_sample_size_l184_184643


namespace sin_two_alpha_plus_pi_over_three_l184_184583

variable (α : ℝ)

theorem sin_two_alpha_plus_pi_over_three (h : cos (π / 3 - α) = 2 * cos (α + π / 6)) :
  sin (2 * α + π / 3) = 4 / 5 :=
sorry

end sin_two_alpha_plus_pi_over_three_l184_184583


namespace correct_exponentiation_l184_184034

theorem correct_exponentiation : (a : ℝ) → ((a^3)^2 = a^6) := 
by
  intros a
  -- Mathematically equivalent proof to show
  -- (a^3)^2 = a^(3*2) = a^6
  rw [pow_mul]
  sorry

end correct_exponentiation_l184_184034


namespace transformed_function_is_odd_l184_184974

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184974


namespace floor_abs_neg_45_7_l184_184186

theorem floor_abs_neg_45_7 : (Int.floor (Real.abs (-45.7))) = 45 :=
by
  sorry

end floor_abs_neg_45_7_l184_184186


namespace Helen_taller_than_Amy_l184_184529

-- Definitions from conditions
def Angela_height : ℕ := 157
def Amy_height : ℕ := 150
def Helen_height := Angela_height - 4

-- Question as a theorem
theorem Helen_taller_than_Amy : Helen_height - Amy_height = 3 := by
  sorry

end Helen_taller_than_Amy_l184_184529


namespace t_100_mod_7_l184_184853

noncomputable def T : ℕ → ℕ 
| 0     := 8
| (n+1) := 5 ^ T n

theorem t_100_mod_7 : T 99 % 7 = 4 :=
sorry

end t_100_mod_7_l184_184853


namespace find_radius_P_l184_184786

noncomputable def radius_of_circle_P : ℝ :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let O := (1, 1)
  let r : ℝ := 3 - 2 * real.sqrt 2
  r

theorem find_radius_P :
  ∃ (P : ℝ × ℝ) (r : ℝ), P = (2 - r, r) ∧ r = 3 - 2 * real.sqrt 2 :=
begin
  use (2 - (3 - 2 * real.sqrt 2), 3 - 2 * real.sqrt 2),
  split,
  { refl },
  { exact (eq.refl (3 - 2 * real.sqrt 2)) }
end

end find_radius_P_l184_184786


namespace smallest_number_divisible_by_618_3648_60_l184_184484

theorem smallest_number_divisible_by_618_3648_60 :
  ∃ n : ℕ, (∀ m, (m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0 → m = 1038239) :=
by
  use 1038239
  sorry

end smallest_number_divisible_by_618_3648_60_l184_184484


namespace three_ints_sum_divisible_by_3_l184_184393

theorem three_ints_sum_divisible_by_3 (a1 a2 a3 a4 a5 : ℤ) : 
  ∃ x y z, (x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4 ∨ x = a5) ∧ 
           (y = a1 ∨ y = a2 ∨ y = a3 ∨ y = a4 ∨ y = a5) ∧ 
           (z = a1 ∨ z = a2 ∨ z = a3 ∨ z = a4 ∨ z = a5) ∧ 
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
           (x + y + z) % 3 = 0 :=
begin
  sorry
end

end three_ints_sum_divisible_by_3_l184_184393


namespace square_possible_length_l184_184436

theorem square_possible_length (sticks : Finset ℕ) (H : sticks = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ s, s = 9 ∧
  ∃ (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a + b + c = 9 :=
by
  sorry

end square_possible_length_l184_184436


namespace table_mat_length_l184_184799

noncomputable def calculate_y (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let y_side := 2 * r * Real.sin (θ / 2)
  y_side

theorem table_mat_length :
  calculate_y 6 8 1 = 3 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end table_mat_length_l184_184799


namespace polynomial_divisibility_check_l184_184855

theorem polynomial_divisibility_check
    (f : ℚ[X] := 4 * X^2 - 6 * X - 18)
    (a : ℚ := 3)
    (m : ℚ := -18)
    (d : ℚ := 36) :
    (eval a f = 0) ∧ (d % m = 0) :=
    by
    sorry

end polynomial_divisibility_check_l184_184855


namespace cos_arcsin_l184_184836

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l184_184836


namespace number_of_truthful_dwarfs_l184_184141

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l184_184141


namespace simplify_expression_l184_184399

theorem simplify_expression :
  (16 / 54) * (27 / 8) * (64 / 81) = 64 / 9 :=
by sorry

end simplify_expression_l184_184399


namespace eval_floor_abs_neg_45_7_l184_184175

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l184_184175


namespace max_value_PA_PF_l184_184918

noncomputable def ellipse_semi_major_axis : ℝ := 2
noncomputable def ellipse_semi_minor_axis : ℝ := √3
noncomputable def ellipse_focus_right : (ℝ × ℝ) := (1, 0)
noncomputable def point_A : (ℝ × ℝ) := (1, 2 * √2)
def ellipse_eq : ∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 := sorry

theorem max_value_PA_PF :
  ∀ P : ℝ × ℝ, 
  (ellipse_eq P.1 P.2) → 
  max (λ P, (dist P point_A) + (dist P ellipse_focus_right)) = 4 + 2 * (√3) := 
sorry

end max_value_PA_PF_l184_184918


namespace movie_theatre_total_seats_l184_184081

theorem movie_theatre_total_seats (A C : ℕ) 
  (hC : C = 188) 
  (hRevenue : 6 * A + 4 * C = 1124) 
  : A + C = 250 :=
by
  sorry

end movie_theatre_total_seats_l184_184081


namespace general_term_a_max_value_S_l184_184246

def S (n : ℕ) : ℤ := -n^2 + 7*n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 7 else -2 * n + 8

theorem general_term_a :
  ∀ n : ℕ, (a n =
  if n = 1 then 7 else -2 * n + 8) := by
  intro n
  sorry

theorem max_value_S :
  ∃ n : ℕ, S n  = 13 := by
  use 3
  sorry

end general_term_a_max_value_S_l184_184246


namespace joanna_initial_gumballs_l184_184332

theorem joanna_initial_gumballs (J : ℕ) : 
  let Jacques_initial := 60
  let total_gumballs := J + Jacques_initial
  let purchased_gumballs := 4 * total_gumballs
  let final_total_gumballs := total_gumballs + purchased_gumballs
  let shared_gumballs := final_total_gumballs / 2
  in shared_gumballs = 250 → J = 40 :=
by {
  intros,
  sorry,
}

end joanna_initial_gumballs_l184_184332


namespace solve_floor_fractional_l184_184894

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem solve_floor_fractional (x : ℝ) :
  floor x * fractional_part x = 2019 * x ↔ x = 0 ∨ x = -1 / 2020 :=
by
  sorry

end solve_floor_fractional_l184_184894


namespace total_sum_of_money_l184_184774

theorem total_sum_of_money (A B C : ℝ) (h_ratio : A / 1 = B / 0.65 ∧ B / 0.65 = C / 0.40) (h_C_share : C = 56) : A + B + C = 287 :=
by
  have h_value_of_one_part : 140 = 56 / 0.40 := sorry -- Skipping the proof
  have h_total_parts : 2.05 = 1 + 0.65 + 0.40 := sorry -- Skipping the proof
  have h_total_sum : (A + B + C) = 140 * 2.05 := sorry -- Skipping the proof
  exact h_total_sum -- Which results in Rs. 287

end total_sum_of_money_l184_184774


namespace probability_grid_black_after_rotation_l184_184789

theorem probability_grid_black_after_rotation :
  let probability := (1 / 2) ^ 16 * ((1 / 2) ^ 16) * ((1 / 2) ^ 12 + 0) * ((1 / 2) ^ 12 + 0) in
  probability = (1 / 65536) :=
by
  sorry

end probability_grid_black_after_rotation_l184_184789


namespace length_of_ae_l184_184476

theorem length_of_ae
  (a b c d e : ℝ)
  (bc : ℝ)
  (cd : ℝ)
  (de : ℝ := 8)
  (ab : ℝ := 5)
  (ac : ℝ := 11)
  (h1 : bc = 2 * cd)
  (h2 : bc = ac - ab)
  : ab + bc + cd + de = 22 := 
by
  sorry

end length_of_ae_l184_184476


namespace distance_between_stations_proof_l184_184452

-- Definitions from conditions
def train_rate_1 : ℝ := 16
def train_rate_2 : ℝ := 21
def extra_distance : ℝ := 60

-- Let D be the distance traveled by the slower train when they meet
-- The distance traveled by faster train will be D + 60
def distance_by_slower_train (D : ℝ) : Prop := (D / train_rate_1) = ((D + extra_distance) / train_rate_2)

-- Now, based on the condition, we need the actual distance between the two stations
def distance_between_stations : ℝ := λ D : ℝ, 2 * D + extra_distance

theorem distance_between_stations_proof : ∃ D : ℝ, distance_by_slower_train D ∧ distance_between_stations D = 444 :=
by
  use 192
  split
  sorry -- The actual proof goes here

end distance_between_stations_proof_l184_184452


namespace water_tank_depth_l184_184076

theorem water_tank_depth (h : ℝ) : 
  ∃ h, 
    let radius := 4 in
    let length := 12 in
    let rectangular_surface_area := 48 in
    (4 - 2 * Real.sqrt 3 = h ∨ 4 + 2 * Real.sqrt 3 = h) ∧
    let c := rectangular_surface_area / length in
    (c = 2 * Real.sqrt (2 * radius * h - h ^ 2)) :=
sorry

end water_tank_depth_l184_184076


namespace geometric_sequence_sum_l184_184652

theorem geometric_sequence_sum (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : S 4 = 1)
  (h2 : S 8 = 3)
  (h3 : ∀ n, S (n + 4) - S n = a (n + 1) + a (n + 2) + a (n + 3) + a (n + 4)) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
by
  -- Insert your proof here.
  sorry

end geometric_sequence_sum_l184_184652


namespace william_time_on_road_l184_184468

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l184_184468


namespace total_votes_cast_l184_184477

theorem total_votes_cast 
  (V : ℕ) -- Total number of votes cast
  (h_cand: 0.35 * V % 1 = 0) -- Candidate's votes as 35% of the total are integers
  (h_v_difference: (0.35 * V) + 1350 = (0.65 * V)) -- Rival received 1350 more votes than the candidate
: V = 4500 :=
sorry

end total_votes_cast_l184_184477


namespace brothers_work_rate_l184_184743

theorem brothers_work_rate (A B C : ℝ) :
  (1 / A + 1 / B = 1 / 8) ∧ (1 / A + 1 / C = 1 / 9) ∧ (1 / B + 1 / C = 1 / 10) →
  A = 160 / 19 ∧ B = 160 / 9 ∧ C = 32 / 3 :=
by
  sorry

end brothers_work_rate_l184_184743


namespace odd_function_check_l184_184954

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184954


namespace infinite_n_exist_l184_184684

-- Define the function s(n) which is the sum of the positive divisors of n.
def sum_divisors (n : ℕ) : ℕ :=
  (finset.range (n+1)).filter (λ d, d ∣ n).sum id

-- Theorem to prove there exist infinitely many n such that (sum_divisors n / n) > (sum_divisors m / m) for all m < n.
theorem infinite_n_exist : ∃∞ n, ∀ m < n, (sum_divisors n) / n > (sum_divisors m) / m :=
sorry

end infinite_n_exist_l184_184684


namespace odd_function_g_l184_184993

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l184_184993


namespace least_n_exceeds_product_l184_184619

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product_l184_184619


namespace triangle_at_most_one_right_angle_l184_184014

-- Definition of a triangle with its angles adding up to 180 degrees
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

-- The main theorem stating that a triangle can have at most one right angle.
theorem triangle_at_most_one_right_angle (α β γ : ℝ) 
  (h₁ : triangle α β γ) 
  (h₂ : α = 90 ∨ β = 90 ∨ γ = 90) : 
  (α = 90 → β ≠ 90 ∧ γ ≠ 90) ∧ 
  (β = 90 → α ≠ 90 ∧ γ ≠ 90) ∧ 
  (γ = 90 → α ≠ 90 ∧ β ≠ 90) :=
sorry

end triangle_at_most_one_right_angle_l184_184014


namespace Ana_shorter_than_Bev_l184_184747

-- Definitions of the given conditions
variable (people : Fin 5 → Fin 5 → ℝ) -- Heights of the 25 people in a 5x5 grid
variable (Ana Bev : ℝ) -- Heights of Ana and Bev
variable (Ana_row : Fin 5) -- Row where Ana is located
variable (Bev_col : Fin 5) -- Column where Bev is located
variable (Ana_shortest_row : Fin 5 → ℝ) -- Shortest persons in each row
variable (Bev_tallest_col : Fin 5 → ℝ) -- Tallest persons in each column

-- Conditions as definitions/statements
def condition1 : Prop := ∀ (i j : Fin 5), people i j ≠ people i.succ j.succ
def condition2 : Prop := Ana = max (Ana_shortest_row 0) (max (Ana_shortest_row 1) (max (Ana_shortest_row 2) (max (Ana_shortest_row 3) (Ana_shortest_row 4))))
def condition3 : Prop := Bev = min (Bev_tallest_col 0) (min (Bev_tallest_col 1) (min (Bev_tallest_col 2) (min (Bev_tallest_col 3) (Bev_tallest_col 4))))
def condition4 : Prop := Ana ≠ Bev

-- Proof statement
theorem Ana_shorter_than_Bev (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : Ana < Bev :=
by
  sorry

end Ana_shorter_than_Bev_l184_184747


namespace transformed_function_is_odd_l184_184976

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184976


namespace abc_value_l184_184631

theorem abc_value 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (hab : a * b = 24) 
  (hac : a * c = 40) 
  (hbc : b * c = 60) : 
  a * b * c = 240 := 
by sorry

end abc_value_l184_184631


namespace transformed_function_is_odd_l184_184966

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184966


namespace TotalGenuineItems_l184_184445

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l184_184445


namespace two_trains_meet_at_distance_l184_184450

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end two_trains_meet_at_distance_l184_184450


namespace greater_number_l184_184425

theorem greater_number (x: ℕ) (h1 : 3 * x + 4 * x = 21) : 4 * x = 12 := by
  sorry

end greater_number_l184_184425


namespace car_speed_l184_184212

def distance (A B : Type) [MetricSpace A] (d : ℝ) : Prop := dist A B = d

def car_departs (A B : Type) [MetricSpace A] [MetricSpace B] : Prop := 
  ∃ (v : ℝ), ∀ (t : ℝ), t >= 0 → dist A (A + t * v) = t * v

def motorcyclist_follows (A M : Type) [MetricSpace A] [MetricSpace M] 
  (start_delay : ℝ) (speed : ℝ) : Prop :=
  ∃ (t : ℝ), t >= start_delay → dist A (A + (t - start_delay) * speed) = (t - start_delay) * speed

def halfway_condition (M A B : Type) [MetricSpace M] [MetricSpace A] [MetricSpace B] 
  (meet_time : ℝ) : Prop :=
  ∃ (v : ℝ), dist A M = meet_time * v ∧ (v * (meet_time / 2) = dist B (B - (meeting_distance / 2) * 60))

theorem car_speed : 
  let A B M : Type := ℕ in
  let start_delay := 1/3 in
  let motor_speed := 60 in
  let distance_AB := 82.5 in
  distance A B distance_AB → 
  car_departs A B →
  motorcyclist_follows A M start_delay motor_speed →
  halfway_condition M A B (distance_AB / (2 * motor_speed)) →
  ∃ v : ℝ, v = 45 :=
begin
  sorry
end

end car_speed_l184_184212


namespace odd_function_check_l184_184950

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184950


namespace matrix_multiplication_solution_l184_184880

theorem matrix_multiplication_solution :
  let M := ![[6, -5], [0, 2]] in
  (M ⬝ ![[1], [2]] = ![[-4], [4]]) ∧ (M ⬝ ![[-3], [1]] = ![[-23], [2]]) :=
by
  let M := ![[6, -5], [0, 2]]
  have h1 : M ⬝ ![[1], [2]] = ![[-4], [4]] := by sorry
  have h2 : M ⬝ ![[-3], [1]] = ![[-23], [2]] := by sorry
  exact ⟨h1, h2⟩

end matrix_multiplication_solution_l184_184880


namespace lcm_of_n_and_24_l184_184018

-- Define the given conditions
def n : ℕ := 16
def twenty_four : ℕ := 24
def gcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem to be proved
theorem lcm_of_n_and_24:
  gcf n twenty_four = 8 →
  lcm n twenty_four = 48 :=
by
  sorry

end lcm_of_n_and_24_l184_184018


namespace find_b_l184_184324

theorem find_b (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a ∈ {1, 2, 4}) (h5 : b ∈ {1, 2, 4}) (h6 : c ∈ {1, 2, 4})
(h7 : (a / 2 : Rat) / (b / c : Rat) = 4) : b = 1 :=
by
  sorry

end find_b_l184_184324


namespace lesser_solution_of_quadratic_l184_184462

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l184_184462


namespace sum_g_from_1_to_1000_l184_184205

def f (n : ℕ) : ℕ :=
  (n.digits 10).count (λ d, d ≥ 5)

def g (n : ℕ) : ℕ :=
  3 ^ (f n)

theorem sum_g_from_1_to_1000 : (∑ i in Finset.range 1001 \ {0}, g i) = 8001 :=
by
  sorry

end sum_g_from_1_to_1000_l184_184205


namespace fraction_simplification_l184_184860

theorem fraction_simplification (x : ℝ) (h : x = 0.5 * 106) : 18 / x = 18 / 53 := by
  rw [h]
  norm_num

end fraction_simplification_l184_184860


namespace sufficient_but_not_necessary_condition_for_negative_root_l184_184056

def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem sufficient_but_not_necessary_condition_for_negative_root 
  (a : ℝ) (h : a < 0) : 
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) ∧ 
  (∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) → a ≤ 0) :=
sorry

end sufficient_but_not_necessary_condition_for_negative_root_l184_184056


namespace range_y_l184_184543

-- Define the function f(x) = log_{1/2}(x)
def f (x : ℝ) : ℝ := log x / log (1/2 : ℝ)

-- Define the inverse function g(x) = (1/2)^x
def g (x : ℝ) : ℝ := (1/2 : ℝ) ^ x

-- Define the function y = f(x) + g(x)
def y (x : ℝ) : ℝ := f x + g x

-- The main theorem to prove
theorem range_y : set.range (λ x, y x) = set.Icc (-3/4 : ℝ) (1/2 : ℝ) :=
by
  sorry

end range_y_l184_184543


namespace travel_time_correct_l184_184467

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l184_184467


namespace painting_frame_ratio_proof_l184_184508

def framed_painting_ratio (x : ℝ) : Prop :=
  let width := 20
  let height := 20
  let side_border := x
  let top_bottom_border := 3 * x
  let framed_width := width + 2 * side_border
  let framed_height := height + 2 * top_bottom_border
  let painting_area := width * height
  let frame_area := painting_area
  let total_area := framed_width * framed_height - painting_area
  total_area = frame_area ∧ (width + 2 * side_border) ≤ (height + 2 * top_bottom_border) → 
  framed_width / framed_height = 4 / 7

theorem painting_frame_ratio_proof (x : ℝ) (h : framed_painting_ratio x) : (20 + 2 * x) / (20 + 6 * x) = 4 / 7 :=
  sorry

end painting_frame_ratio_proof_l184_184508


namespace main_theorem_l184_184922

noncomputable def necessary_and_sufficient_condition (a b : ℝ) (S T : set ℝ) : Prop :=
  (∀ x, x ∈ T → x ∈ S) ↔ ∃ p q ∈ ℤ, b = a * (p / q)

theorem main_theorem (a b : ℝ) (S T : set ℝ)
  (hS : S = {x | ∃ n : Int, n < 0 ∧ x = a * n + b})
  (hT : ∃ x0 q, T = {x | ∃ n : Nat, 0 < n ∧ x = x0 * q ^ n}) :
  necessary_and_sufficient_condition a b S T :=
by
  intro h
  sorry

end main_theorem_l184_184922


namespace length_of_segment_ST_l184_184659

noncomputable def isosceles_triangle (P Q R : ℝ) : Prop := sorry

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ := sorry
noncomputable def altitude_of_triangle (P Q R : ℝ) (P_to_R : ℝ) : ℝ := sorry
noncomputable def segment_length (S T : ℝ) : ℝ := sorry

theorem length_of_segment_ST
  (P Q R S T : ℝ)
  (h1 : isosceles_triangle P Q R)
  (h2 : area_of_triangle P Q R = 144)
  (h3 : isosceles_triangle P S T)
  (h4 : area_of_triangle S T Q = 108)
  (h5 : altitude_of_triangle P Q R 24) :
  segment_length S T = 6 :=
sorry

end length_of_segment_ST_l184_184659


namespace number_of_groupings_l184_184751

theorem number_of_groupings (n m : ℕ) (h1 : n = 8) (h2 : m = 2) :
  let total_choices := 2^n in
  let invalid_choices := 2 in
  let valid_choices := total_choices - invalid_choices in
  valid_choices = 254 :=
by {
  sorry
}

end number_of_groupings_l184_184751


namespace possible_shapes_l184_184006

def is_valid_shapes (T S C : ℕ) : Prop :=
  T + S + C = 24 ∧ T = 7 * S

theorem possible_shapes :
  ∃ (T S C : ℕ), is_valid_shapes T S C ∧ 
    (T = 0 ∧ S = 0 ∧ C = 24) ∨
    (T = 7 ∧ S = 1 ∧ C = 16) ∨
    (T = 14 ∧ S = 2 ∧ C = 8) ∨
    (T = 21 ∧ S = 3 ∧ C = 0) :=
by
  sorry

end possible_shapes_l184_184006


namespace max_diff_x1_x2_l184_184392

noncomputable def f (x : ℝ) := 2 * Real.sin(2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) := 2 * Real.sin(2 * x + Real.pi / 6) + 1

theorem max_diff_x1_x2 : ∀ (x1 x2 : ℝ), g x1 + g x2 = 6 ∧ x1 ∈ Icc (-2 * Real.pi) (2 * Real.pi) ∧ x2 ∈ Icc (-2 * Real.pi) (2 * Real.pi) → x1 - x2 = 3 * Real.pi :=
sorry

end max_diff_x1_x2_l184_184392


namespace find_equal_temperatures_l184_184505

-- Define the conversion functions in Lean
def to_kelvin (F : ℤ) : ℤ :=
  Int.floor ((5.0 / 9.0) * (F.toReal - 32.0) + 273.15)

def to_fahrenheit (K : ℤ) : ℤ :=
  Int.floor ((9.0 / 5.0) * (K.toReal - 273.15) + 32.0)

-- The theorem statement
theorem find_equal_temperatures :
  (Finset.card (Finset.filter (λ F: ℤ, F = to_fahrenheit (to_kelvin F))
                (Finset.range (500 - 100 + 1) |>.map (λ n, n + 100)))) = 401 :=
sorry

end find_equal_temperatures_l184_184505


namespace cos_arcsin_eq_l184_184837

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l184_184837


namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k_l184_184197

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k_l184_184197


namespace problem_l184_184876

noncomputable def d : ℝ := -8.63

theorem problem :
  let floor_d := ⌊d⌋
  let frac_d := d - floor_d
  (3 * floor_d^2 + 20 * floor_d - 67 = 0) ∧
  (4 * frac_d^2 - 15 * frac_d + 5 = 0) → 
  d = -8.63 :=
by {
  sorry
}

end problem_l184_184876


namespace perpendicular_distance_D_to_plane_ABC_l184_184541

-- Define the points D, A, B, C
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (5, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 3, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 2)

-- Define a function to calculate the perpendicular distance from a point to a plane
def perp_distance_to_plane (P Q R S : ℝ × ℝ × ℝ) : ℝ :=
  abs ((S.1 - P.1) * (Q.2 - P.2) * (R.3 - P.3) + (S.2 - P.2) * (Q.3 - P.3) * (R.1 - P.1) + (S.3 - P.3) * (Q.1 - P.1) * (R.2 - P.2)
  - (S.3 - P.3) * (Q.2 - P.2) * (R.1 - P.1) - (S.1 - P.1) * (Q.3 - P.3) * (R.2 - P.2) - (S.2 - P.2) * (Q.1 - P.1) * (R.3 - P.3)) /
  (real.sqrt ((Q.2 - P.2) * (R.3 - P.3) - (Q.3 - P.3) * (R.2 - P.2))^2 + ((Q.3 - P.3) * (R.1 - P.1) - (Q.1 - P.1) * (R.3 - P.3))^2 +
  ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))^2))

-- The statement to show:
theorem perpendicular_distance_D_to_plane_ABC : perp_distance_to_plane D A B C = 1.9 :=
sorry

end perpendicular_distance_D_to_plane_ABC_l184_184541


namespace compute_ab_bc_ca_ratio_l184_184348

variable {a b c : ℝ}
variable (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
variable (h4 : a + b + c = 1)

theorem compute_ab_bc_ca_ratio : 
  (ab_plus_bc_plus_ca (ab | bc)
  (ca (a^2 + b^2 + c^2) = 0 :=
by
  sorry

end compute_ab_bc_ca_ratio_l184_184348


namespace vec_eq_l184_184278

def a : ℝ × ℝ := (-1, 0)
def b : ℝ × ℝ := (0, 2)

theorem vec_eq : (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (-2, -6) := by
  sorry

end vec_eq_l184_184278


namespace height_passes_through_circumcenter_l184_184729

theorem height_passes_through_circumcenter
  (A B C D : Point)
  (h_eq : dist A D = dist B D ∧ dist B D = dist C D) :
  let H := foot_point D (triangle_plane A B C) in
  is_circumcenter H A B C :=
by
  sorry

end height_passes_through_circumcenter_l184_184729


namespace part_I_part_II_l184_184608

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x)) / x

theorem part_I (a : ℝ) (ha : a > 1) : 
  if 1 < a ∧ a ≤ 2 then ∃ xs ∈ Icc 1 a, isMinOn f (Icc 1 a) xs ∧ f xs = Real.log 2
  else ∃ xs ∈ Icc 1 a, isMinOn f (Icc 1 a) xs ∧ f xs = (Real.log (2 * a)) / a :=
sorry

theorem part_II (m : ℝ) (hm : ∃! n ∈ ℤ, f n ^ 2 + m * f n > 0) : 
  (-Real.log 2 < m ∧ m ≤ -Real.log 6 / 3) :=
sorry

end part_I_part_II_l184_184608


namespace card_probability_l184_184522

-- Define the total number of cards
def total_cards : ℕ := 52

-- Define the number of Kings in the deck
def kings_in_deck : ℕ := 4

-- Define the number of Aces in the deck
def aces_in_deck : ℕ := 4

-- Define the probability of the top card being a King
def prob_top_king : ℚ := kings_in_deck / total_cards

-- Define the probability of the second card being an Ace given the first card is a King
def prob_second_ace_given_king : ℚ := aces_in_deck / (total_cards - 1)

-- Define the combined probability of both events happening in sequence
def combined_probability : ℚ := prob_top_king * prob_second_ace_given_king

-- Theorem statement that the combined probability is equal to 4/663
theorem card_probability : combined_probability = 4 / 663 := by
  -- Proof to be filled in
  sorry

end card_probability_l184_184522


namespace percentage_of_palindromes_with_seven_l184_184764

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def contains_seven (n : ℕ) : Prop :=
  let digits := n.toString in
  '7' ∈ digits.toList

theorem percentage_of_palindromes_with_seven :
  let palindromes := {n | 1000 ≤ n ∧ n < 5000 ∧ is_palindrome n}
  let palindromes_with_seven := {n | contains_seven n ∧ n ∈ palindromes}
  let total_palindromes := (palindromes.toFinset.card : ℕ)
  let total_palindromes_with_seven := (palindromes_with_seven.toFinset.card : ℕ)
  (total_palindromes != 0) →
  (total_palindromes_with_seven * 100) / total_palindromes = 19 :=
by
  sorry

end percentage_of_palindromes_with_seven_l184_184764


namespace steps_to_walk_l184_184519

theorem steps_to_walk (total_floors : ℕ) (steps_per_floor : ℕ) : 
  total_floors = 7 → steps_per_floor = 13 → 
  (steps_per_floor * (total_floors - 1)) = 78 :=
by
  intros h_floors h_steps
  rw [h_floors, h_steps]
  norm_num
  sorry

end steps_to_walk_l184_184519


namespace greatest_a_no_integral_solution_l184_184355

theorem greatest_a_no_integral_solution (a : ℤ) :
  (∀ x : ℤ, |x + 1| ≥ a - 3 / 2) → a = 1 :=
by
  sorry

end greatest_a_no_integral_solution_l184_184355


namespace find_complex_number_l184_184032

theorem find_complex_number (z : ℂ) : (5 - 3 * complex.i) + z = (-4 + 9 * complex.i) ↔ z = -9 + 12 * complex.i :=
by
  sorry

end find_complex_number_l184_184032


namespace max_area_with_150_feet_fencing_l184_184094

theorem max_area_with_150_feet_fencing : 
  ∃ (x : ℕ), 2 * x + 2 * (75 - x) = 150 ∧ x > 0 ∧ (75 - x) > 0 ∧
  (x * (75 - x) = 1406) :=
begin
  sorry
end

end max_area_with_150_feet_fencing_l184_184094


namespace min_value_in_interval_l184_184554

theorem min_value_in_interval (a : ℝ) :
  (∀ x : ℝ, x < 1 → (∃ m : ℝ, ∀ y : ℝ, y < 1 → f y ≥ m)) ↔ a < 1 :=
by
  let f : ℝ → ℝ := λ x, x^2 - 2*a*x + a
  sorry

end min_value_in_interval_l184_184554


namespace transformed_function_is_odd_l184_184984

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184984


namespace exists_polynomial_P_l184_184551

open Int Nat

/-- Define a predicate for a value is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Define the polynomial P(x, y, z) --/
noncomputable def P (x y z : ℕ) : ℤ := 
  (1 - 2013 * (z - 1) * (z - 2)) * 
  ((x + y - 1) * (x + y - 1) + 2 * y - 2 + z)

/-- The main theorem to prove --/
theorem exists_polynomial_P :
  ∃ (P : ℕ → ℕ → ℕ → ℤ), 
  (∀ n : ℕ, (¬ is_square n) ↔ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) := 
sorry

end exists_polynomial_P_l184_184551


namespace find_range_of_k_l184_184254

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l184_184254


namespace area_of_fourth_rectangle_l184_184079

-- Define the dimensions and areas
variables (a b : ℝ)
def width1 := 2 * a
def width2 := 3 * a
def height1 := b
def height2 := 2 * b

def area_R1 := 2 * a * b
def area_R2 := 6 * a * b
def area_R3 := 4 * a * b

-- Define the area of the fourth rectangle
def area_R4 := width2 * height2

-- The main theorem we need to prove
theorem area_of_fourth_rectangle (a b : ℝ) :
  area_R4 a b = 6 * a * b :=
by
  unfold area_R4
  unfold width2 height2
  sorry

end area_of_fourth_rectangle_l184_184079


namespace sum_of_sampled_types_l184_184498

-- Define the types of books in each category
def Chinese_types := 20
def Mathematics_types := 10
def Liberal_Arts_Comprehensive_types := 40
def English_types := 30

-- Define the total types of books
def total_types := Chinese_types + Mathematics_types + Liberal_Arts_Comprehensive_types + English_types

-- Define the sample size and stratified sampling ratio
def sample_size := 20
def sampling_ratio := sample_size / total_types

-- Define the number of types sampled from each category
def Mathematics_sampled := Mathematics_types * sampling_ratio
def Liberal_Arts_Comprehensive_sampled := Liberal_Arts_Comprehensive_types * sampling_ratio

-- Define the proof statement
theorem sum_of_sampled_types : Mathematics_sampled + Liberal_Arts_Comprehensive_sampled = 10 :=
by
  -- Your proof here
  sorry

end sum_of_sampled_types_l184_184498


namespace sum_even_powers_is_rational_l184_184537

theorem sum_even_powers_is_rational : 
  (∑ n in -4..4, (-2 : ℚ)^(2 * n)) ∈ ℚ :=
by
  sorry

end sum_even_powers_is_rational_l184_184537


namespace expansion_contains_x4_l184_184598

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def expansion_term (x : ℂ) (i : ℂ) : ℂ :=
  binomial_coeff 6 2 * x^4 * i^2

theorem expansion_contains_x4 (x i : ℂ) (hi : i = Complex.I) : 
  expansion_term x i = -15 * x^4 := by
  sorry

end expansion_contains_x4_l184_184598


namespace product_fraction_l184_184564

theorem product_fraction :
  (∏ k in finset.range (52 - 2 + 1), (1 - (1 : ℝ) / (k + 3))) = (1 / 26 : ℝ) :=
by sorry

end product_fraction_l184_184564


namespace eval_f_neg2_l184_184028

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l184_184028


namespace train_speed_kmh_l184_184810

theorem train_speed_kmh (train_length: ℕ) (bridge_length: ℕ) (time_seconds: ℕ) (speed_kmh: ℕ): 
  train_length = 510 → bridge_length = 140 → time_seconds = 52 →
  speed_kmh = 45 :=
begin
  intros,
  sorry,
end

end train_speed_kmh_l184_184810


namespace arctan_sum_of_roots_eq_pi_div_4_l184_184685

theorem arctan_sum_of_roots_eq_pi_div_4 (x₁ x₂ x₃ : ℝ) 
  (h₁ : Polynomial.eval x₁ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₂ : Polynomial.eval x₂ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₃ : Polynomial.eval x₃ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h_intv : -5 < x₁ ∧ x₁ < 5 ∧ -5 < x₂ ∧ x₂ < 5 ∧ -5 < x₃ ∧ x₃ < 5) :
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = Real.pi / 4 :=
sorry

end arctan_sum_of_roots_eq_pi_div_4_l184_184685


namespace fraction_sum_condition_l184_184891

theorem fraction_sum_condition 
  (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : x + y = x * y): 
  (1/x + 1/y = 1) :=
by
  sorry

end fraction_sum_condition_l184_184891


namespace frogs_equilibrium_l184_184823

axiom circular_pool_divided (n : ℕ) (hn : n ≥ 5) : Prop := True

axiom cell_properties {n : ℕ} (A : ℕ) (ha : 1 ≤ A ∧ A ≤ 2*n) (hn : n ≥ 5) :
  ∀ (B : ℕ), (1 ≤ B ∧ B ≤ 2*n) → Prop := 
  ∃ (neighbors : list ℕ), (neighbors.length = 3) ∧ (∀ b ∈ neighbors, b ≠ A ∧ 1 ≤ b ∧ b ≤ 2*n)

axiom frogs_in_pool {n : ℕ} (hn : n ≥ 5) : 
  ∃ (cells_with_frogs : list (ℕ × ℕ)), cells_with_frogs.length = 2*n ∧
  list.sum (cells_with_frogs.map (λ c, c.2)) = 4*n + 1

noncomputable def eventual_equilibrium_state {n : ℕ} (hn : n ≥ 5) :=
  ∀ (A : ℕ), ∃ (frogs_in_cell_A : ℕ),
    (frogs_in_cell_A > 0) ∨ 
    ∀ (neighbors_A : list ℕ), 
    (neighbors_A.length = 3) ∧ 
    (∀ B ∈ neighbors_A, ∃ (frogs_in_cell_B : ℕ), frogs_in_cell_B > 0)

theorem frogs_equilibrium {n : ℕ} (hn : n ≥ 5) :
  circular_pool_divided n hn →
  cell_properties {n} → 
  frogs_in_pool {n} → 
  eventual_equilibrium_state {n} hn :=
by sorry

end frogs_equilibrium_l184_184823


namespace lesser_solution_of_quadratic_eq_l184_184460

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l184_184460


namespace rectangle_diagonals_perpendicular_is_square_l184_184083

theorem rectangle_diagonals_perpendicular_is_square (R : Type) [rect : Rectangle R] (h : diagonals_perpendicular rect) : Square R :=
sorry

end rectangle_diagonals_perpendicular_is_square_l184_184083


namespace william_time_on_road_l184_184469

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l184_184469


namespace fraction_of_married_men_l184_184105

-- We start by defining the conditions given in the problem.
def only_single_women_and_married_couples (total_women total_married_women : ℕ) :=
  total_women - total_married_women + total_married_women * 2

def probability_single_woman_single (total_women total_single_women : ℕ) :=
  total_single_women / total_women = 3 / 7

-- The main theorem we need to prove under the given conditions.
theorem fraction_of_married_men (total_women total_married_women : ℕ)
  (h1 : probability_single_woman_single total_women (total_women - total_married_women))
  : (total_married_women * 2) / (total_women + total_married_women) = 4 / 11 := sorry

end fraction_of_married_men_l184_184105


namespace f_f_neg1_eq_1_l184_184638

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem f_f_neg1_eq_1 : f(f(-1)) = 1 := by
  sorry

end f_f_neg1_eq_1_l184_184638


namespace has_propertyT_f1_no_propertyT_f2_no_propertyT_f3_no_propertyT_f4_l184_184299

-- Definitions
def PropertyT (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), deriv f x1 * deriv f x2 = -1

-- Functions
def f1 (x : ℝ) := Real.sin x
def f2 (x : ℝ) := Real.log x
def f3 (x : ℝ) := Real.exp x
def f4 (x : ℝ) := x^3

-- Prove PropertyT for f1, f2, f3, and f4
theorem has_propertyT_f1 : PropertyT f1 :=
sorry

theorem no_propertyT_f2 : ¬ PropertyT f2 :=
sorry

theorem no_propertyT_f3 : ¬ PropertyT f3 :=
sorry

theorem no_propertyT_f4 : ¬ PropertyT f4 :=
sorry

end has_propertyT_f1_no_propertyT_f2_no_propertyT_f3_no_propertyT_f4_l184_184299


namespace sum_arith_seq_elems_l184_184239

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_arith_seq_elems (a d : ℝ) 
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 48) :
  arithmetic_seq a d 6 + arithmetic_seq a d 7 = 24 := 
by 
  sorry

end sum_arith_seq_elems_l184_184239


namespace line_through_point_with_equal_intercepts_l184_184570

theorem line_through_point_with_equal_intercepts :
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    ((y = m * x + b ∧ ((x = 0 ∨ y = 0) → (x = y))) ∧ 
    (1 = m * 1 + b ∧ 1 + 1 = b)) → 
    (m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 2) :=
by
  sorry

end line_through_point_with_equal_intercepts_l184_184570


namespace find_a_plus_2b_l184_184233

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l184_184233


namespace coprime_or_multiple_or_double_subset_exists_l184_184391

theorem coprime_or_multiple_or_double_subset_exists
  (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (P : Finset ℕ) (hP : P ⊆ S) (hcard : P.card = 5):
  (∃ a b ∈ P, Nat.gcd a b = 1) ∧ 
  (¬ ∃ a b ∈ P, a ≠ b ∧ a ∣ b) ∧
  (∃ a b ∈ P, a ≠ b ∧ (a * 2 = b ∨ b * 2 = a)) :=
by
  sorry

end coprime_or_multiple_or_double_subset_exists_l184_184391


namespace largest_real_number_l184_184008

theorem largest_real_number (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = -7) 
  (h3 : a * b * c = -14) : 
  max a (max b c) = sqrt 7 :=
by
  sorry

end largest_real_number_l184_184008


namespace find_sum_a_b_l184_184234

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l184_184234


namespace find_radius_of_circle_l184_184618

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 2 * t + 1)

def circle_parametric (a θ : ℝ) (h : a > 0) : ℝ × ℝ :=
  (a * Real.cos θ, a * Real.sin θ)

theorem find_radius_of_circle 
  (h : ∀ P ∈ { (x, y) | ∃ θ, P = (a * Real.cos θ, a * Real.sin θ) }, 
     ∃ t, |(2 * (fst P) + 1 - snd P) / sqrt (2^2 + 1^2)| = (sqrt 5 / 5) + 1 )
  : a = 1 :=
sorry

end find_radius_of_circle_l184_184618


namespace f_2021_l184_184594

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom period_f : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_neg1 : f (-1) = 1

theorem f_2021 : f (2021) = -1 :=
by
  sorry

end f_2021_l184_184594


namespace odd_function_check_l184_184938

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_check :
  ∀ x : ℝ, (f (x - 1) + 1) = - (f (-x - 1) + 1) := 
by
  intro x
  sorry

end odd_function_check_l184_184938


namespace part_a_n_gon_vertices_at_lattice_points_part_b_polyhedra_vertices_at_lattice_points_l184_184779

-- Definition of what it means for n-gon vertices to lie on integer lattice points
def exists_regular_ngon_with_lattice_vertices (n : ℕ) : Prop :=
  ∃ (vertices : list (ℤ × ℤ × ℤ)), vertices.length = n ∧
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → 
    (vertices.nth i).is_some ∧ (vertices.nth j).is_some ∧ 
    dist (vertices.nth_le i sorry) (vertices.nth_le j sorry) = 1)

theorem part_a_n_gon_vertices_at_lattice_points :
  ∀ (n : ℕ), exists_regular_ngon_with_lattice_vertices n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

-- Definition of what it means for polyhedron vertices to lie on integer lattice points
def exists_regular_polyhedra_with_lattice_vertices (poly : Type) [polyhedron poly] : Prop :=
  ∃ (vertices : list (ℤ × ℤ × ℤ)), 
    (∀ (v ∈ vertices), is_lattice_point v) ∧ 
    regular_polyhedron poly vertices

theorem part_b_polyhedra_vertices_at_lattice_points :
  ∀ (poly : Type) [polyhedron poly], exists_regular_polyhedra_with_lattice_vertices poly ↔ 
    (poly = Cube ∨ poly = Tetrahedron ∨ poly = Octahedron) :=
sorry

end part_a_n_gon_vertices_at_lattice_points_part_b_polyhedra_vertices_at_lattice_points_l184_184779


namespace combination_x_l184_184630
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_x (x : ℕ) (H : C 25 (2 * x) = C 25 (x + 4)) : x = 4 ∨ x = 7 :=
by sorry

end combination_x_l184_184630


namespace line_sum_of_slope_and_intercept_l184_184116

theorem line_sum_of_slope_and_intercept :
  (let m := ((-1 - 3) : ℝ) / ((-3 - 1) : ℝ) in
   let b := (3 : ℝ) - m * 1 in
   m + b = 3) := 
by
  sorry

end line_sum_of_slope_and_intercept_l184_184116


namespace solve_equation_l184_184734

theorem solve_equation (x : ℝ) (h : x * (x - 3) = 10) : x = 5 ∨ x = -2 :=
by sorry

end solve_equation_l184_184734


namespace area_of_region_l184_184547

theorem area_of_region :
  (∫ x, ∫ y in {y : ℝ | x^4 + y^4 = |x|^3 + |y|^3}, (1 : ℝ)) = 4 :=
sorry

end area_of_region_l184_184547


namespace limit_example_l184_184488

noncomputable def my_limit_problem (f : ℝ → ℝ) (a A : ℝ) :=
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → |f x - A| < ε

theorem limit_example : my_limit_problem (λ x, (15 * x^2 - 2 * x - 1) / (x + 1/5)) (-1/5) (-8) :=
by
  sorry

end limit_example_l184_184488


namespace transformed_function_is_odd_l184_184965

-- Define the given function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define a transformation of the given function
def f_transformed (x : ℝ) : ℝ := f (x - 1) + 1

-- Prove that the transformed function is odd
theorem transformed_function_is_odd : ∀ x : ℝ, f_transformed (-x) = -f_transformed (x) :=
by
    sorry

end transformed_function_is_odd_l184_184965
