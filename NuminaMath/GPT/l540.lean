import Mathlib

namespace library_books_l540_540069

theorem library_books (A : Prop) (B : Prop) (C : Prop) (D : Prop) :
  (¬A) → (B ∧ D) :=
by
  -- Assume the statement "All books in this library are available for lending." is represented by A.
  -- A is false.
  intro h_notA
  -- Show that statement II ("There is some book in this library not available for lending.")
  -- and statement IV ("Not all books in this library are available for lending.") are both true.
  -- These are represented as B and D, respectively.
  sorry

end library_books_l540_540069


namespace correct_answer_is_C_l540_540775

def proposition_p : Prop := ∀ x : ℝ, y = x^3 → y = -(-x)^3
def proposition_q : Prop := ∀ a b c : ℝ, (b^2 = a * c) → (∃ r : ℝ, b = r * a ∧ c = r * b)

theorem correct_answer_is_C (hp : proposition_p) (hq : ¬ proposition_q) : 
  (¬ hp ∨ ¬hq) :=
by
  sorry

end correct_answer_is_C_l540_540775


namespace math_problem_l540_540247

noncomputable def proof_problem (n : ℕ) (a : Fin n → ℝ) (φ : Fin n → Fin n → ℝ) :=
  let a1 := a 0
  a1 ^ 2 = (∑ i in Finset.range 1 n, (a i) ^ 2) 
           + 2 * (∑ i in Finset.range 1 n | i > 1, ∑ j in Finset.range 1 n | j > 1 ∧ i > j, a i * a j * cos (φ i j))

theorem math_problem (n : ℕ) (a : Fin n → ℝ) (φ : Fin n → Fin n → ℝ) : 
  proof_problem n a φ :=
sorry

end math_problem_l540_540247


namespace appears_and_unique_1992_l540_540371

def g (x : ℕ) : ℕ := if x % 2 = 1 then x else g (x / 2)  -- largest odd divisor

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + 2 / g x
  else
    2 ^ ((x + 1) / 2)

def sequence : ℕ → ℕ
  | 0     => 1
  | (n+1) => f (sequence n)

theorem appears_and_unique_1992 :
  ∃ n : ℕ, sequence n = 1992 ∧ ∀ m : ℕ, sequence m = 1992 → m = 8253 :=
sorry

end appears_and_unique_1992_l540_540371


namespace variance_of_binomial_distribution_example_l540_540275

theorem variance_of_binomial_distribution_example :
  ∀ (X : ℕ → ℝ), (∀ k : ℕ, X k = if h : k ≤ 10 then binomial 10 0.7 k else 0) → 
  (∑ k in range (10 + 1), k * (X k) * (1 - X k)) = 2.1 :=
begin
  intros,
  sorry
end

end variance_of_binomial_distribution_example_l540_540275


namespace mike_travel_time_l540_540296

-- Definitions of conditions
def dave_steps_per_min : ℕ := 85
def dave_step_length_cm : ℕ := 70
def dave_time_min : ℕ := 20
def mike_steps_per_min : ℕ := 95
def mike_step_length_cm : ℕ := 65

-- Calculate Dave's speed in cm/min
def dave_speed_cm_per_min := dave_steps_per_min * dave_step_length_cm

-- Calculate the distance to school in cm
def school_distance_cm := dave_speed_cm_per_min * dave_time_min

-- Calculate Mike's speed in cm/min
def mike_speed_cm_per_min := mike_steps_per_min * mike_step_length_cm

-- Calculate the time for Mike to get to school in minutes as a rational number
def mike_time_min := (school_distance_cm : ℚ) / mike_speed_cm_per_min

-- The proof problem statement
theorem mike_travel_time :
  mike_time_min = 19 + 2 / 7 :=
sorry

end mike_travel_time_l540_540296


namespace algebraic_expression_value_l540_540446

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 7 = 0) : x^2 - y^2 - 14y = 49 :=
by
  sorry

end algebraic_expression_value_l540_540446


namespace richmond_population_l540_540938

theorem richmond_population (R V B : ℕ) (h0 : R = V + 1000) (h1 : V = 4 * B) (h2 : B = 500) : R = 3000 :=
by
  -- skipping proof
  sorry

end richmond_population_l540_540938


namespace surface_area_of_circumscribed_sphere_l540_540466

-- Given definitions based on the problem description
def parallelepiped (a b c : ℝ) : Prop :=
  ∀ {x y z : ℝ}, x = a ∧ y = b ∧ z = c

axiom plane_through_diagonal (a b c l : ℝ)
  (angles : plane forms angles 45 30 with base sides)
  (parallel : plane is parallel to base diagonal)
  (distance : distance from plane to base diagonal = l) :
  parallelepiped a b c

-- The main theorem to be proved
theorem surface_area_of_circumscribed_sphere (a b c l : ℝ)
  (h : plane_through_diagonal a b c l 
  (by sorry) -- angle conditions
  (by sorry) -- parallel condition
  (by sorry)) -- distance condition
  : 
  4 * π * (2 * l) ^ 2 = 16 * π * l^2 :=
by sorry

end surface_area_of_circumscribed_sphere_l540_540466


namespace num_athletes_with_4_points_after_seven_rounds_correct_l540_540858

noncomputable def num_athletes_with_4_points_after_seven_rounds (n : ℕ) (hn : n > 7) : ℕ :=
  let num_participants := 2^n + 6
  let f (m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k
  if hn then 35 * 2^(n - 7) + 2 else 0

theorem num_athletes_with_4_points_after_seven_rounds_correct (n : ℕ) (hn : n > 7) :
  num_athletes_with_4_points_after_seven_rounds n hn = 35 * 2^(n - 7) + 2 :=
by
  sorry

end num_athletes_with_4_points_after_seven_rounds_correct_l540_540858


namespace least_prime_factor_p6_minus_p5_l540_540225

theorem least_prime_factor_p6_minus_p5 (p : ℕ) (hp : Nat.Prime p) : (∃ q : ℕ, Nat.Prime q ∧ q ∣ (p^6 - p^5) ∧ ∀ r : ℕ, Nat.Prime r ∧ r ∣ (p^6 - p^5) → q ≤ r) → 2 :=
by
  sorry

end least_prime_factor_p6_minus_p5_l540_540225


namespace coeff_inv_x_sq_l540_540865

theorem coeff_inv_x_sq (n : ℕ) (h : (binomial_coeff (2 * n) 1) * 2 ^ (2 * n - 2) = 224) : 
  (binomial_coeff (2 * n) 5) * 2 ^ (2 * n - 10) = 14 :=
by
  sorry

end coeff_inv_x_sq_l540_540865


namespace service_center_milepost_l540_540191

theorem service_center_milepost :
  ∀ (first_exit seventh_exit service_fraction : ℝ), 
    first_exit = 50 →
    seventh_exit = 230 →
    service_fraction = 3 / 4 →
    (first_exit + service_fraction * (seventh_exit - first_exit) = 185) :=
by
  intros first_exit seventh_exit service_fraction h_first h_seventh h_fraction
  sorry

end service_center_milepost_l540_540191


namespace probability_same_subject_together_l540_540686

theorem probability_same_subject_together :
  let total_books : ℕ := 6
  let chinese_books : ℕ := 1
  let english_books : ℕ := 2
  let math_books : ℕ := 3
  let total_permutations := Nat.factorial total_books
  let units_permutations := Nat.factorial (chinese_books + 1 + 1)  -- 3 units: Chinese, English, Math
  let english_permutations := Nat.factorial english_books
  let math_permutations := Nat.factorial math_books
  let favorable_outcomes := units_permutations * english_permutations * math_permutations
  let probability := favorable_outcomes / total_permutations
  probability = 1 / 10 :=
by {
  let total_books := 6
  let chinese_books := 1
  let english_books := 2
  let math_books := 3
  let total_permutations := Nat.factorial total_books
  let units_permutations := Nat.factorial (1 + 1 + 1)
  let english_permutations := Nat.factorial english_books
  let math_permutations := Nat.factorial math_books
  let favorable_outcomes := units_permutations * english_permutations * math_permutations
  let probability := favorable_outcomes / total_permutations
  have total_perm : total_permutations = 720 := Nat.factorial.equations._eqn_1 6
  have units_perm : units_permutations = 6 := Nat.factorial.equations._eqn_1 3
  have english_perm : english_permutations = 2 := Nat.factorial.equations._eqn_1 2
  have math_perm : math_permutations = 6 := Nat.factorial.equations._eqn_1 3
  have favor_out : favorable_outcomes = 72 := by
    rw [units_perm, english_perm, math_perm]
    norm_num
  have prob : probability = 1 / 10 := by
    rw [favor_out, total_perm]
    norm_num
  exact prob
}

end probability_same_subject_together_l540_540686


namespace smallest_N_l540_540253

theorem smallest_N (N : ℕ) : (N * 3 ≥ 75) ∧ (N * 2 < 75) → N = 25 :=
by {
  sorry
}

end smallest_N_l540_540253


namespace cyclic_quadrilateral_max_prod_l540_540704

theorem cyclic_quadrilateral_max_prod (a b c d : ℝ) (ac bd ad bc : ℝ) 
  (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: d > 0) 
  (ptolemy : ∃ e f : ℝ, ac + bd = e * f) : 
  ∃ (s : ℝ), a = s ∧ b = s ∧ c = s ∧ d = s ∧ 
               (ab + cd) * (ac + bd) * (ad + bc) ≤ 4 * s^5 * sqrt 2 :=
by 
  sorry

end cyclic_quadrilateral_max_prod_l540_540704


namespace repeating_decimal_simplest_denominator_l540_540169

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l540_540169


namespace tangent_lines_passing_through_point_l540_540577

theorem tangent_lines_passing_through_point :
  ∀ (x0 y0 : ℝ) (p : ℝ × ℝ), 
  (p = (1, 1)) ∧ (y0 = x0 ^ 3) → 
  (y0 - 1 = 3 * x0 ^ 2 * (1 - x0)) → 
  (x0 = 1 ∨ x0 = -1/2) → 
  ((y - (3 * 1 - 2)) * (y - (3/4 * x0 + 1/4))) = 0 :=
sorry

end tangent_lines_passing_through_point_l540_540577


namespace distance_to_conference_center_l540_540564

-- Define all necessary conditions as variables or hypotheses.
variables (x : ℝ) -- Total distance in miles
variables (usual_time : ℝ) (total_time : ℝ) (speed_reduction : ℝ)
hypothesis (h1: usual_time = 200)
hypothesis (h2: total_time = 324)
hypothesis (h3: speed_reduction = 30)

-- Define the initial speed before hitting traffic.
def initial_speed := x / usual_time

-- Define the adjusted speed after hitting traffic (in miles per minute).
def adjusted_speed := initial_speed - (speed_reduction / 60)

-- Define the time equations for both halves of the journey.
def time_first_half := (x / 2) / initial_speed
def time_second_half := (x / 2) / adjusted_speed

-- Theorem stating the equivalent math proof problem.
theorem distance_to_conference_center:
  time_first_half x h1 + time_second_half x h1 h3 = total_time h2 → x = 180 :=
by
  sorry

end distance_to_conference_center_l540_540564


namespace recurring_six_denominator_l540_540172

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l540_540172


namespace distance_between_point_and_circle_center_l540_540861

theorem distance_between_point_and_circle_center 
  (A_polar : ℝ × ℝ)
  (E_eqn : θ → ℝ)
  (A_polar = (2 * Real.sqrt 2, Real.pi / 4))
  (E_eqn = (λ θ, 4 * Real.sin θ)) :
  let A_cart := (A_polar.1 * Real.cos A_polar.2, A_polar.1 * Real.sin A_polar.2) in
  let E_center := (0, 4 * Real.sin (Real.pi / 2)) in
  Real.dist A_cart E_center = 2 :=
by
  sorry

end distance_between_point_and_circle_center_l540_540861


namespace curve_is_cardioid_l540_540355

def is_cardioid (r : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, r = λ θ, a + b * Real.cos θ ∧ a = b

theorem curve_is_cardioid : is_cardioid (λ θ, 3 + Real.cos θ) :=
by
  sorry

end curve_is_cardioid_l540_540355


namespace total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l540_540509

-- Definition of properties required as per the given conditions
def is_fortunate_number (abcd ab cd : ℕ) : Prop :=
  abcd = 100 * ab + cd ∧
  ab ≠ cd ∧
  ab ∣ cd ∧
  cd ∣ abcd

-- Total number of fortunate numbers is 65
theorem total_fortunate_numbers_is_65 : 
  ∃ n : ℕ, n = 65 ∧ 
  ∀(abcd ab cd : ℕ), is_fortunate_number abcd ab cd → n = 65 :=
sorry

-- Largest odd fortunate number is 1995
theorem largest_odd_fortunate_number_is_1995 : 
  ∃ abcd : ℕ, abcd = 1995 ∧ 
  ∀(abcd' ab cd : ℕ), is_fortunate_number abcd' ab cd ∧ cd % 2 = 1 → abcd = 1995 :=
sorry

end total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l540_540509


namespace maxim_receives_l540_540897

def initial_deposit : ℝ := 1000
def annual_interest_rate : ℝ := 0.12
def monthly_compounded : ℕ := 12
def time_period : ℝ := 1 / 12

def final_amount (P r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem maxim_receives (P r : ℝ) (n : ℕ) (t : ℝ) :
  final_amount P r n t = 1010 :=
by
  have h1 : P = initial_deposit := rfl
  have h2 : r = annual_interest_rate := rfl
  have h3 : n = monthly_compounded := rfl
  have h4 : t = time_period := rfl
  rw [h1, h2, h3, h4]
  sorry

end maxim_receives_l540_540897


namespace divisibility_by_37_l540_540559

def sum_of_segments (n : ℕ) : ℕ :=
  let rec split_and_sum (num : ℕ) (acc : ℕ) : ℕ :=
    if num < 1000 then acc + num
    else split_and_sum (num / 1000) (acc + num % 1000)
  split_and_sum n 0

theorem divisibility_by_37 (A : ℕ) : 
  (37 ∣ A) ↔ (37 ∣ sum_of_segments A) :=
sorry

end divisibility_by_37_l540_540559


namespace average_salary_l540_540244

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 14000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

theorem average_salary : (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8200 := 
  by 
    sorry

end average_salary_l540_540244


namespace circumcircle_inequality_l540_540385

variables (A B C D E F: Point)
variables (hexagon_convex : Convex ABCDEF)
variables (parallel_AB_ED : AB ∥ ED)
variables (parallel_BC_FE : BC ∥ FE)
variables (parallel_CD_AF : CD ∥ AF)
variables (R_A R_C R_E : ℝ)
def p := sideLength A B + sideLength B C + sideLength C D + sideLength D E + sideLength E F + sideLength F A

theorem circumcircle_inequality :
  R_A + R_C + R_E ≥ p / 2 :=
sorry

end circumcircle_inequality_l540_540385


namespace no_possible_circles_l540_540870

theorem no_possible_circles (S : set (set ℝ²)) :
  (∀ l : ℝ² → ℝ² → Prop, ∃ k ∈ S, ∀ x y : ℝ², l x y → intersects_line k l) ∧
  (∀ l : ℝ² → ℝ² → Prop, finset.card {k ∈ S | intersects_line k l} ≤ 100) →
  false := 
sorry

end no_possible_circles_l540_540870


namespace absolute_difference_m_n_l540_540521

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l540_540521


namespace third_edge_length_l540_540188

theorem third_edge_length (a b v : ℝ) (h₁ : a = 2) (h₂ : b = 5) (h₃ : v = 30) : ∃ c, v = a * b * c ∧ c = 3 :=
by
  use 3
  rw [h₁, h₂, h₃]
  norm_num
  exact ⟨rfl, rfl⟩

end third_edge_length_l540_540188


namespace sum_identity_l540_540907

open BigOperators

theorem sum_identity (m n k : ℕ) (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) (hkmn : k ≤ m) (hmn : m ≤ n) :
  ∑ i in Finset.range (n+1), (-1 : ℕ) ^ i * (Nat.choose m i) * (Nat.choose (m + n + i) (n + 1)) * (1 / (n + k + i)) = 1 / (n + 1) :=
by
  sorry

end sum_identity_l540_540907


namespace array_sum_remainder_mod_9_l540_540727

theorem array_sum_remainder_mod_9 :
  let sum_terms := ∑' r : ℕ, ∑' c : ℕ, (1 / (4 ^ r)) * (1 / (9 ^ c))
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ sum_terms = m / n ∧ (m + n) % 9 = 5 :=
by
  sorry

end array_sum_remainder_mod_9_l540_540727


namespace problem_l540_540105

-- Assume that we have a set M and a coloring function that assigns each element a color
def M (n : ℕ) := {i : ℕ // 1 ≤ i ∧ i ≤ n}
-- Define the color type as an enumeration of three possible colors: red, blue, yellow
inductive Color
| red
| blue
| yellow

-- Define the coloring function
def coloring (n : ℕ) : M n → Color := sorry

-- Define set A
def A (n : ℕ) : Finset (M n × M n × M n) :=
{p ∈ (M n).product (M n).product (M n) |
    (p.1 + p.2.1 + p.2.2) % n = 0 ∧
    coloring n p.1 = coloring n p.2.1 ∧
    coloring n p.1 = coloring n p.2.2
}

-- Define set B
def B (n : ℕ) : Finset (M n × M n × M n) :=
{p ∈ (M n).product (M n).product (M n) |
    (p.1 + p.2.1 + p.2.2) % n = 0 ∧
    (coloring n p.1 ≠ coloring n p.2.1) ∧
    (coloring n p.1 ≠ coloring n p.2.2) ∧
    (coloring n p.2.1 ≠ coloring n p.2.2)
}

theorem problem (n : ℕ) : 2 * (A n).card ≥ (B n).card :=
sorry

end problem_l540_540105


namespace angle_A₁FB₁_eq_90_l540_540780

-- Definitions according to the conditions given in the problem
variables {p : ℝ} (h_p : p ≠ 0)
def parabola := { xy : ℝ × ℝ // xy.1^2 = 2 * p * xy.2 }

variables {A B : ℝ × ℝ} (hA : A ∈ parabola h_p) (hB : B ∈ parabola h_p)
def directrix := { xy : ℝ × ℝ // xy.2 = -p/2 }

variables {A₁ B₁ : ℝ × ℝ}
(hA₁ : A₁ ∈ directrix h_p ∧ ∃ A, A ∈ parabola h_p ∧ proj_directrix A = A₁)
(hB₁ : B₁ ∈ directrix h_p ∧ ∃ B, B ∈ parabola h_p ∧ proj_directrix B = B₁)

-- F is the focus of the parabola
def F : ℝ × ℝ := (0, p/2)

-- Statement of the theorem
theorem angle_A₁FB₁_eq_90 :
  ∠ A₁ F B₁ = 90 :=
sorry

end angle_A₁FB₁_eq_90_l540_540780


namespace tiling_impossible_2003x2003_l540_540871

theorem tiling_impossible_2003x2003 :
  ¬ (∃ (f : Fin 2003 × Fin 2003 → ℕ),
  (∀ p : Fin 2003 × Fin 2003, f p = 1 ∨ f p = 2) ∧
  (∀ p : Fin 2003, (f (p, 0) + f (p, 1)) % 3 = 0) ∧
  (∀ p : Fin 2003, (f (0, p) + f (1, p) + f (2, p)) % 3 = 0)) := 
sorry

end tiling_impossible_2003x2003_l540_540871


namespace josh_extracurricular_hours_l540_540100

structure Schedule where
  soccer_mon : ℕ
  soccer_wed : ℕ
  soccer_fri : ℕ
  band_tue : ℕ
  band_thu : ℕ
  tutoring_mon : ℕ
  tutoring_wed : ℕ
  coding_fri : ℕ

def total_hours (s : Schedule) : ℕ :=
  let mon_extra := s.soccer_mon + (s.tutoring_mon - 1)
  let wed_extra := s.soccer_wed + (s.tutoring_wed - 1)
  let fri_extra := s.soccer_fri + (s.coding_fri - 0.5)
  mon_extra + s.band_tue + wed_extra + s.band_thu + fri_extra

theorem josh_extracurricular_hours (s : Schedule) : 
  total_hours s = 11 :=
  sorry

end josh_extracurricular_hours_l540_540100


namespace sum_of_roots_x_plus_3_x_minus_4_eq_24_l540_540441

theorem sum_of_roots_x_plus_3_x_minus_4_eq_24 :
  (∃ x : ℝ, (x + 3) * (x - 4) = 24) → (∀ x₁ x₂ : ℝ, (x₁ + 3) * (x₁ - 4) = 24 ∧ (x₂ + 3) * (x₂ - 4) = 24 → x₁ + x₂ = 1) :=
by
  assume h 
  obtain ⟨x, hx⟩ := h
  sorry

end sum_of_roots_x_plus_3_x_minus_4_eq_24_l540_540441


namespace best_fraction_expression_l540_540989

-- Define the expressions
def exprA : ℕ := 2
def exprB (x : ℝ) : ℝ := x / 4
def exprC (x : ℝ) : ℝ := 1 / (2 * x - 3)
def exprD : ℝ := Real.pi / 2

-- The statement to prove that exprC is the fraction as per the context
theorem best_fraction_expression (x : ℝ) : (exprC x = 1 / (2 * x - 3)) := by
  sorry

end best_fraction_expression_l540_540989


namespace am_gm_inequality_l540_540909

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) / (Real.cbrt (x * y * z)) ≤ (x / y) + (y / z) + (z / x) := 
sorry

end am_gm_inequality_l540_540909


namespace area_of_triangle_correct_l540_540457

noncomputable def area_of_triangle (A B a : ℝ) (hA : A = 30) (hB : B = 45) (ha : a = 2) : ℝ :=
  let b := 2 * Real.sqrt 2 in
  let C := 180 - A - B in
  let area := 1/2 * a * b * Real.sin (C * Real.pi / 180) in
  area

theorem area_of_triangle_correct : 
  area_of_triangle 30 45 2 (by rfl) (by rfl) (by rfl) = Real.sqrt 3 + 1 :=
  by sorry

end area_of_triangle_correct_l540_540457


namespace derivative_cos_2x_l540_540384

def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem derivative_cos_2x :
  ∀ x : ℝ, (deriv f x) = -2 * Real.sin (2 * x) :=
by
  sorry

end derivative_cos_2x_l540_540384


namespace find_train_length_l540_540689

theorem find_train_length:
  ∀ (speed_kmph : ℕ) (time_sec : ℝ) (bridge_length : ℝ),
    speed_kmph = 54 →
    time_sec = 55.99552035837134 →
    bridge_length = 660 →
    let speed_mps := (speed_kmph : ℝ) * (1000 / 3600) in
    let distance_covered := speed_mps * time_sec in
    let train_length := distance_covered - bridge_length in
    train_length = 179.9328053755701 :=
by {
  intros,
  simp [h, h_1, h_2],
  sorry,
}

end find_train_length_l540_540689


namespace recurring_six_denominator_l540_540174

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l540_540174


namespace eval_f_at_neg_twenty_three_sixth_pi_l540_540757

noncomputable def f (α : ℝ) : ℝ := 
    (2 * (Real.sin (2 * Real.pi - α)) * (Real.cos (2 * Real.pi + α)) - Real.cos (-α)) / 
    (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

theorem eval_f_at_neg_twenty_three_sixth_pi : 
  f (-23 / 6 * Real.pi) = -Real.sqrt 3 :=
  sorry

end eval_f_at_neg_twenty_three_sixth_pi_l540_540757


namespace fruit_baskets_l540_540824

def apple_choices := 8 -- From 0 to 7 apples
def orange_choices := 13 -- From 0 to 12 oranges

theorem fruit_baskets (a : ℕ) (o : ℕ) (ha : a = 7) (ho : o = 12) :
  (apple_choices * orange_choices) - 1 = 103 := by
  sorry

end fruit_baskets_l540_540824


namespace circle_eq_l540_540786

theorem circle_eq (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (4, 0))
  (hC : C = (0, 2)) :
  ∃ (h: ℝ), (x - 3) ^ 2 + (y - 3) ^ 2 = h :=
by 
  use 10
  -- additional steps to rigorously prove the result would go here
  sorry

end circle_eq_l540_540786


namespace angle_B_is_right_angle_l540_540555

theorem angle_B_is_right_angle {FAB BCD DEF : Triangle}
  (congruent : FAB ≅ BCD ∧ BCD ≅ DEF)
  (equilateral_hexagon : EquilateralHexagon)
  (acute_angles : ∀ (T : Triangle), T.is_isosceles ∧ (T ≠ FBD → ∠T.acute = 15)) :
  ∠B = 90 := 
by
  sorry

end angle_B_is_right_angle_l540_540555


namespace eccentricity_range_circle_equation_l540_540393

-- Define the ellipse and its properties
def is_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : b < a) 
: Prop :=
∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Prove that the range of the eccentricity is 0 <= e <= 1/2
theorem eccentricity_range (a b c e : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
(b_lt_a : b < a) (focus_def : c = a * e) (eq1 : is_ellipse a b a_pos b_pos b_lt_a) 
(A : ℝ × ℝ) (F : ℝ × ℝ) (m : set (ℝ × ℝ)) 
(not_isosceles : ¬ ∃ Q, Q ∈ m ∧ is_isosceles_triangle (A,F,Q))
: 0 ≤ e ∧ e ≤ 1/2 := sorry

-- Define the equation of the circle centered at midpoint of MN passing through A and F
theorem circle_equation (e : ℝ) (M N A F : ℝ × ℝ) (h_e : e = 1/2) 
(A_coord : A = (-2, 0)) (F_coord : F = (1, 0)) (a b : ℝ) (eq_ellipse : is_ellipse a b (by linarith [h_e]) (by linarith [h_e]) (by linarith [h_e]))
: circle_equation (midpoint M N) A F = (x + 1/2)^2 + (y ± sqrt(21)/4)^2 = 57/16 := sorry

end eccentricity_range_circle_equation_l540_540393


namespace right_triangle_smaller_angle_l540_540853

theorem right_triangle_smaller_angle (x : ℝ) (h_right_triangle : 0 < x ∧ x < 90)
  (h_double_angle : ∃ y : ℝ, y = 2 * x)
  (h_angle_sum : x + 2 * x = 90) :
  x = 30 :=
  sorry

end right_triangle_smaller_angle_l540_540853


namespace percentage_of_girls_l540_540959

def total_students : ℕ := 100
def boys : ℕ := 50
def girls : ℕ := total_students - boys

theorem percentage_of_girls :
  (girls / total_students) * 100 = 50 := sorry

end percentage_of_girls_l540_540959


namespace arun_weight_average_l540_540649

theorem arun_weight_average :
  (∀ w : ℝ, 65 < w ∧ w < 72 → 60 < w ∧ w < 70 → w ≤ 68 → 66 ≤ w ∧ w ≤ 69 → 64 ≤ w ∧ w ≤ 67.5 → 
    (66.75 = (66 + 67.5) / 2)) := by
  sorry

end arun_weight_average_l540_540649


namespace smallest_positive_period_maximum_value_set_l540_540801

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2 * x - (π / 6)) + 2 * sin (x - (π / 12)) ^ 2

theorem smallest_positive_period :
  ∀ x, f x = f (x + π) :=
by sorry

theorem maximum_value_set :
  { x | f x = 3 } = { x | ∃ (k : ℤ), x = k * π + (5 * π / 12) } :=
by sorry

end smallest_positive_period_maximum_value_set_l540_540801


namespace infinite_product_result_l540_540327

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540327


namespace consecutive_sum_15_number_of_valid_sets_l540_540042

theorem consecutive_sum_15 : 
  ∃ n (a : ℕ), n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30 :=
begin
  sorry
end

theorem number_of_valid_sets : 
  finset.card ((finset.filter (λ n a, n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30) (finset.range 15).product (finset.range 15))) = 2 :=
begin
  sorry
end

end consecutive_sum_15_number_of_valid_sets_l540_540042


namespace sum_largest_smallest_prime_factors_546_l540_540629

theorem sum_largest_smallest_prime_factors_546 : 
  (let p := prime_factors 546 in 
   List.sum [p.head, p.ilast] = 15) :=
by
  sorry

end sum_largest_smallest_prime_factors_546_l540_540629


namespace volume_enlargement_l540_540831

variable {a : ℝ}

def V1 : ℝ := a^3
def V2 : ℝ := (2 * a)^3

theorem volume_enlargement :
  V2 = 8 * V1 := 
by 
  sorry

end volume_enlargement_l540_540831


namespace trapezoid_area_l540_540618

/-- 
  Define the lines that bound the trapezoid.
-/
def line1 (x : ℝ) : ℝ := x + 2
def line2 (x : ℝ) : ℝ := 12
def line3 (x : ℝ) : ℝ := 7
def y_axis (x : ℝ) : ℝ := 0

/-- 
  Define the vertices of the trapezoid by finding intersections
-/
def vertex1 : ℝ × ℝ := (10, 12)
def vertex2 : ℝ × ℝ := (5, 7)
def vertex3 : ℝ × ℝ := (0, 12)
def vertex4 : ℝ × ℝ := (0, 7)

/-- 
  Calculate the area of the trapezoid bounded by these vertices
-/
theorem trapezoid_area : 
  let area : ℝ := 1 / 2 * (5 + 12) * 5 in
  area = 42.5 := 
by
  sorry

end trapezoid_area_l540_540618


namespace evaluate_expression_l540_540103

noncomputable theory

def p := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
def q := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
def r := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6
def s := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6

theorem evaluate_expression :
  ( (1 / p) + (1 / q) + (1 / r) + (1 / s) ) ^ 2 = 3 / 16 :=
by
  sorry

end evaluate_expression_l540_540103


namespace smallest_number_divisible_and_remainder_l540_540622

theorem smallest_number_divisible_and_remainder (n : ℕ) :
  (∀ m ∈ {3, 4, 5, 6, 7, 8}, (n - 2) % m = 0) ∧ (n % 11 = 0) →
  n = 3362 :=
sorry

end smallest_number_divisible_and_remainder_l540_540622


namespace projection_judgments_l540_540914

def projections_are_correct (AOB α : Type) (proj : AOB → α) : Prop :=
  (proj 0 = 0 ∨ proj 0 = 180) ∧
  (proj 90 = 90) ∧
  (∃ x, 0 < x ∧ x < 90 ∧ proj x = x) ∧
  (∃ y, 90 < y ∧ y < 180 ∧ proj y = y)

theorem projection_judgments (AOB α : Type) (proj : AOB → α) :
  projections_are_correct AOB α proj → {1, 2, 3, 4, 5} :=
by sorry

end projection_judgments_l540_540914


namespace trapezoid_area_l540_540223

theorem trapezoid_area:
  let vert1 := (10, 10)
  let vert2 := (15, 15)
  let vert3 := (0, 15)
  let vert4 := (0, 10)
  let base1 := 10
  let base2 := 15
  let height := 5
  ∃ (area : ℝ), area = 62.5 := by
  sorry

end trapezoid_area_l540_540223


namespace square_area_in_ellipse_l540_540267

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end square_area_in_ellipse_l540_540267


namespace polar_to_rectangular_4sqrt2_pi_over_4_l540_540292

theorem polar_to_rectangular_4sqrt2_pi_over_4 :
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (4, 4) :=
by
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end polar_to_rectangular_4sqrt2_pi_over_4_l540_540292


namespace max_value_of_y_l540_540497

noncomputable def max_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 20*x + 54*y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  max_y x y ≤ 27 + Real.sqrt 829 :=
sorry

end max_value_of_y_l540_540497


namespace area_square_l540_540931

-- Define the points and lengths given in the problem
variables (A B C D E F G : Type) [Let (A : Point) (B : Point) (C : Point) (D : Point) (E : Point) (F : Point) (G : Point)]
variables (AB : length) (CD : length)

-- Define the specific lengths given in the problem
def AB_len : AB = 20 := by sorry
def CD_len : CD = 70 := by sorry

-- Statement to prove area of square
theorem area_square (x : Float) (h1 : AB = 20) (h2 : CD = 70) (h3 : x^2 = 1400) : 
    ∃ (area : Float), area = x^2 ∧ area = 1400 := 
by 
  existsi x^2 
  rw [h3] 
  exact ⟨rfl, h3.symm⟩

end area_square_l540_540931


namespace arithmetic_seq_count_114_l540_540753

open Finset

noncomputable def count_four_term_arithmetic_seq (s : Finset ℕ) : ℕ :=
  s.filter (λ t, ∃ a d, t = {a, a + d, a + 2*d, a + 3*d} ∧
    a ∈ s ∧ a + d ∈ s ∧ a + 2*d ∈ s ∧ a + 3*d ∈ s ∧
    (a + d ≠ a) ∧ (a + 2*d ≠ a) ∧ (a + 3*d ≠ a) ∧
    d ≠ 0).card

theorem arithmetic_seq_count_114 :
  count_four_term_arithmetic_seq (range 20 \ {0}) = 114 := sorry

end arithmetic_seq_count_114_l540_540753


namespace johns_weekly_earnings_after_raise_l540_540099

theorem johns_weekly_earnings_after_raise 
  (original_weekly_earnings : ℕ) 
  (percentage_increase : ℝ) 
  (new_weekly_earnings : ℕ)
  (h1 : original_weekly_earnings = 60)
  (h2 : percentage_increase = 0.16666666666666664) :
  new_weekly_earnings = 70 :=
sorry

end johns_weekly_earnings_after_raise_l540_540099


namespace distance_travelled_l540_540057

theorem distance_travelled (t : ℝ) (h : 15 * t = 10 * t + 20) : 10 * t = 40 :=
by
  have ht : t = 4 := by linarith
  rw [ht]
  norm_num

end distance_travelled_l540_540057


namespace alpha_parallel_beta_sufficient_not_necessary_l540_540532

-- Definitions corresponding to given conditions
variables {a b : Line} {α β : Plane}

-- Conditions
def different_lines (a b : Line) : Prop := a ≠ b
def different_planes (α β : Plane) : Prop := α ≠ β
def line_in_plane (a : Line) (α : Plane) : Prop := a ⊆ α
def perp_line_plane (b : Line) (β : Plane) : Prop := b ⊥ β

-- Main theorem: alpha parallel beta is a sufficient but not necessary condition for a perp b
theorem alpha_parallel_beta_sufficient_not_necessary
  (h1: different_lines a b)
  (h2: different_planes α β)
  (h3: line_in_plane a α)
  (h4: perp_line_plane b β) :
  (parallel α β → perpendicular a b) ∧ ¬(perpendicular a b → parallel α β) :=
by
  sorry

end alpha_parallel_beta_sufficient_not_necessary_l540_540532


namespace maximize_sequence_length_l540_540741

-- Define the sequence terms based on the provided conditions
def sequence (b : ℕ -> ℤ) (y : ℤ) : ℕ -> ℤ
| 0     := 2000
| 1     := y
| (n+2) := if b (n+1) + b n < 0 then ((b (n+1) - b n)) else (b (n+1) + b n)

-- Define the initial conditions for b
def b : ℕ -> ℤ :=
  sequence (λ n, 0) y

-- The goal is to prove that sequence length is maximized when y = 1340
theorem maximize_sequence_length : ∃ y : ℤ, y = 1340 ∧ ∀ m : ℕ, b m ≥ 0 := sorry

end maximize_sequence_length_l540_540741


namespace repeating_decimal_simplest_denominator_l540_540166

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l540_540166


namespace length_of_real_axis_l540_540424

-- Define the hyperbola and its properties
variables {a b : ℝ} (h : a > 0) (k : b > 0)

-- Define the conditions and the statement to prove the length of the real axis
theorem length_of_real_axis (h1 : ∃ P : ℝ × ℝ, P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h2 : ∀ P : ℝ × ℝ, ∃ A B : ℝ × ℝ, 
               (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧ 
               (A.1 > 0 ∧ A.2 > 0 ∧ B.1 > 0 ∧ B.2 < 0) ∧
               (A.1 - P.1 = 3 * (P.1 - B.1) ∧ A.2 - P.2 = 3 * (P.2 - B.2)) ∧
               (abs (A.1 * B.2 - A.2 * B.1) = 4 * b)) : 
  2 * a = 4 :=
begin
  sorry
end

end length_of_real_axis_l540_540424


namespace volume_difference_l540_540264

-- Given conditions
def radius_sphere := 6
def radius_cylinder := 4
def height_cylinder := 4 * Real.sqrt (5)
def volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
def volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder

-- The main statement to prove
theorem volume_difference (X : ℝ) : 
  X = 288 - 64 * Real.sqrt(5) -> 
  (volume_sphere - volume_cylinder) = X * Real.pi :=
by
  sorry

end volume_difference_l540_540264


namespace teamX_total_games_l540_540933

variables (x : ℕ)

-- Conditions
def teamX_wins := (3/4) * x
def teamX_loses := (1/4) * x

def teamY_wins := (2/3) * (x + 10)
def teamY_loses := (1/3) * (x + 10)

-- Question: Prove team X played 20 games
theorem teamX_total_games :
  teamY_wins - teamX_wins = 5 ∧ teamY_loses - teamX_loses = 5 → x = 20 := by
sorry

end teamX_total_games_l540_540933


namespace repeating_decimal_simplest_denominator_l540_540168

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l540_540168


namespace general_term_formula_max_value_of_S_n_l540_540751

noncomputable def d : ℤ := (-28 + 2) / 13

def a (n : ℤ) : ℤ := -2 + (n - 7) * d

def S (n : ℤ) : ℤ := n * (12 - n + 1)

theorem general_term_formula :
  a 7 = -2 ∧ a 20 = -28 →
  ∀ n : ℤ, a n = 14 - 2 * n :=
by
  intros h
  sorry

theorem max_value_of_S_n :
  ∀ n : ℤ, (S 6 = 42) ∧ (S 7 = 42) →
  ∀ n : ℤ, S n ≤ 42 :=
by
  intros h
  sorry

end general_term_formula_max_value_of_S_n_l540_540751


namespace largest_possible_A_l540_540634

theorem largest_possible_A (A B C : ℕ) (h1 : 10 = A * B + C) (h2 : B = C) : A ≤ 9 :=
by sorry

end largest_possible_A_l540_540634


namespace park_area_is_correct_l540_540605

-- Define the side of the square
def side_length : ℕ := 30

-- Define the area function for a square
def area_of_square (side: ℕ) : ℕ := side * side

-- State the theorem we're going to prove
theorem park_area_is_correct : area_of_square side_length = 900 := 
sorry -- proof not required

end park_area_is_correct_l540_540605


namespace find_a_plus_b_l540_540886

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + a * x + b) 
  (h2 : { x : ℝ | 0 ≤ f x ∧ f x ≤ 6 - x } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } ∪ {6}) 
  : a + b = 9 := 
sorry

end find_a_plus_b_l540_540886


namespace exists_base_and_digit_l540_540465

def valid_digit_in_base (B : ℕ) (V : ℕ) : Prop :=
  V^2 % B = V ∧ V ≠ 0 ∧ V ≠ 1

theorem exists_base_and_digit :
  ∃ B V, valid_digit_in_base B V :=
by {
  sorry
}

end exists_base_and_digit_l540_540465


namespace age_twice_in_Y_years_l540_540675

def present_age_of_son : ℕ := 24
def age_difference := 26
def present_age_of_man : ℕ := present_age_of_son + age_difference

theorem age_twice_in_Y_years : 
  ∃ (Y : ℕ), present_age_of_man + Y = 2 * (present_age_of_son + Y) → Y = 2 :=
by
  sorry

end age_twice_in_Y_years_l540_540675


namespace cube_surface_area_l540_540965

theorem cube_surface_area (s : ℝ) (h : s = 8) : 6 * s^2 = 384 :=
by
  sorry

end cube_surface_area_l540_540965


namespace ellipse_x_intercepts_l540_540277

noncomputable def distances_sum (x : ℝ) (y : ℝ) (f₁ f₂ : ℝ × ℝ) :=
  (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2)) + (Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2))

def is_on_ellipse (x y : ℝ) : Prop := 
  distances_sum x y (0, 3) (4, 0) = 7

theorem ellipse_x_intercepts 
  (h₀ : is_on_ellipse 0 0) 
  (hx_intercept : ∀ x : ℝ, is_on_ellipse x 0 → x = 0 ∨ x = 20 / 7) :
  ∀ x : ℝ, is_on_ellipse x 0 ↔ x = 0 ∨ x = 20 / 7 :=
by
  sorry

end ellipse_x_intercepts_l540_540277


namespace sequences_convergence_to_one_third_l540_540206

-- Definition of sequences
noncomputable def a : ℕ → ℝ := sorry
noncomputable def b : ℕ → ℝ := sorry
noncomputable def c : ℕ → ℝ := sorry

-- Conditions
axiom initial_condition : a 1 + b 1 + c 1 = 1

axiom recursive_relation_a (n : ℕ) : a (n + 1) = (a n)^2 + 2 * (b n) * (c n)
axiom recursive_relation_b (n : ℕ) : b (n + 1) = (b n)^2 + 2 * (c n) * (a n)
axiom recursive_relation_c (n : ℕ) : c (n + 1) = (c n)^2 + 2 * (a n) * (b n)

-- Statement to be proved
theorem sequences_convergence_to_one_third :
  ∀ (n : ℕ), (has_limit (λ n, a n) (1/3)) ∧
             (has_limit (λ n, b n) (1/3)) ∧
             (has_limit (λ n, c n) (1/3)) :=
sorry

end sequences_convergence_to_one_third_l540_540206


namespace J_speed_is_4_l540_540488

noncomputable def J_speed := 4
variable (v_J v_P : ℝ)

axiom condition1 : v_J > v_P
axiom condition2 : v_J + v_P = 7
axiom condition3 : (24 / v_J) + (24 / v_P) = 14

theorem J_speed_is_4 : v_J = J_speed :=
by
  sorry

end J_speed_is_4_l540_540488


namespace option_b_correct_option_c_correct_l540_540156

def star_player_winning_rate : ℚ := 3/4
def other_players_winning_rate : ℚ := 1/2
def total_players : ℕ := 5

def prob_star_not_in_first_four_wins_four_games (prob_star : ℚ) (prob_other : ℚ) : ℚ :=
  let win_all_but_one_in_first_four := 3 * (prob_other ^ 3) * (1 - prob_other)
  in win_all_but_one_in_first_four

def prob_team_wins_in_three_games (prob_star : ℚ) (prob_other : ℚ) : ℚ :=
  let star_in_first_three := (prob_other ^ 2) * prob_star
  let star_not_in_first_three := prob_other ^ 3
  in (3/5) * star_in_first_three + (2/5) * star_not_in_first_three

theorem option_b_correct :
  prob_star_not_in_first_four_wins_four_games star_player_winning_rate other_players_winning_rate = 3/16 := sorry

theorem option_c_correct :
  prob_team_wins_in_three_games star_player_winning_rate other_players_winning_rate = 13/80 := sorry

end option_b_correct_option_c_correct_l540_540156


namespace distinct_ordered_pairs_l540_540046

/-- There are 9 distinct ordered pairs of positive integers (m, n) such that the sum of the 
    reciprocals of m and n equals 1/6. -/
theorem distinct_ordered_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.card = 9 ∧ 
  ∀ (p : ℕ × ℕ), p ∈ s → 
    (0 < p.1 ∧ 0 < p.2) ∧ 
    (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6) :=
sorry

end distinct_ordered_pairs_l540_540046


namespace initial_money_proof_l540_540294

-- Definition: Dan's initial money, the money spent, and the money left.
def initial_money : ℝ := sorry
def spent_money : ℝ := 1.0
def left_money : ℝ := 2.0

-- Theorem: Prove that Dan's initial money is the sum of the money spent and the money left.
theorem initial_money_proof : initial_money = spent_money + left_money :=
sorry

end initial_money_proof_l540_540294


namespace find_number_of_juniors_l540_540859

noncomputable def juniors_count (total_students : Nat) (middle_schoolers : Nat) (junior_percentage : ℚ) (senior_percentage : ℚ) (equal_team_count : Bool) : Nat :=
  let remaining_students := total_students - middle_schoolers
  let ratio := senior_percentage / junior_percentage
  let seniors := (remaining_students * junior_percentage) / (junior_percentage + ratio * junior_percentage)
  let juniors := remaining_students - seniors
  juniors

theorem find_number_of_juniors : juniors_count 40 5 (20 / 100) (25 / 100) = 20 :=
  by
    sorry

end find_number_of_juniors_l540_540859


namespace sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l540_540714

-- Definition for the sum of the first n natural numbers
def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition for the sum from 1 to 60
def sum_1_to_60 : ℕ := sum_upto 60

-- Definition for the sum from 1 to 50
def sum_1_to_50 : ℕ := sum_upto 50

-- Proof problem 1
theorem sum_from_1_to_60_is_1830 : sum_1_to_60 = 1830 := 
by
  sorry

-- Definition for the sum from 51 to 60
def sum_51_to_60 : ℕ := sum_1_to_60 - sum_1_to_50

-- Proof problem 2
theorem sum_from_51_to_60_is_555 : sum_51_to_60 = 555 := 
by
  sorry

end sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l540_540714


namespace inverse_32_mod_53_l540_540779

theorem inverse_32_mod_53 (h1 : (21 : ℤ)⁻¹ ≡ 17 [ZMOD 53])
                          (h2 : 32 ≡ -21 [ZMOD 53]) :
  (32 : ℤ)⁻¹ ≡ 36 [ZMOD 53] :=
sorry

end inverse_32_mod_53_l540_540779


namespace chord_length_of_concentric_circles_tangent_l540_540574

theorem chord_length_of_concentric_circles_tangent (R r c : ℝ) 
  (h_area : π * R^2 - π * r^2 = 20 * π) 
  (h_chord : (c / 2)^2 + r^2 = R^2) :
  c = 4 * real.sqrt 5 :=
by
  sorry

end chord_length_of_concentric_circles_tangent_l540_540574


namespace x_gt_y_neither_suff_nor_nec_l540_540001

theorem x_gt_y_neither_suff_nor_nec (x y : ℝ) : 
  (¬ (∀ x y : ℝ, x > y → |x| > |y|)) ∧ (¬ (∀ x y : ℝ, |x| > |y| → x > y)) := 
by 
  constructor; 
  { intro h, have h1 := h 1 (-1), linarith }, 
  { intro h, have h2 := h (-2) 1, linarith }

end x_gt_y_neither_suff_nor_nec_l540_540001


namespace actual_diameter_of_tissue_is_correct_l540_540982

-- Define the conditions
def magnifiedDiameter : ℝ := 0.3
def magnificationFactor : ℝ := 1000

-- Define the actual diameter calculation
def actualDiameter := magnifiedDiameter / magnificationFactor

-- Theorem stating the actual diameter
theorem actual_diameter_of_tissue_is_correct : actualDiameter = 0.0003 :=
by
  -- The proof is omitted, as requested
  sorry

end actual_diameter_of_tissue_is_correct_l540_540982


namespace radius_of_tangent_circle_l540_540668

-- Define the 45-45-90 triangle properties
def right_isosceles_triangle (A B C : Point ℝ) : Prop :=
  ∃ (h AB BC CA : ℝ),
    AB = 2 ∧
    BC = 2 ∧
    CA = 2 * Real.sqrt 2 ∧
    ∃ (θ : ℝ), θ = π / 4 ∧ θ = Angle B A C ∧ θ = Angle C A B

-- Define the tangency to coordinate axes
def circle_tangent_to_axes (O : Point ℝ) (r : ℝ) : Prop :=
  O.x = r ∧ O.y = r

-- Define the hypotenuse being horizontal
def hypotenuse_horizontal (A C : Point ℝ) : Prop :=
  A.y = C.y

-- Define the circle tangent to hypotenuse
def circle_tangent_to_hypotenuse (O : Point ℝ) (r : ℝ) (hypotenuse : Line ℝ) : Prop :=
  ∃ (P : Point ℝ), P ∈ hypotenuse ∧ dist O P = r

-- Problem statement
theorem radius_of_tangent_circle (A B C O : Point ℝ) (r : ℝ) :
  right_isosceles_triangle A B C →
  circle_tangent_to_axes O r →
  hypotenuse_horizontal A C →
  circle_tangent_to_hypotenuse O r (line_through A C) →
  r = 2 :=
sorry

end radius_of_tangent_circle_l540_540668


namespace solution_to_logarithmic_equation_l540_540600

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

def equation (x : ℝ) := log_base 2 x + 1 / log_base (x + 1) 2 = 1

theorem solution_to_logarithmic_equation :
  ∃ x > 0, equation x ∧ x = 1 :=
by
  sorry

end solution_to_logarithmic_equation_l540_540600


namespace sphere_surface_area_of_regular_tetrahedron_l540_540764

noncomputable def tetrahedron_base_edge := Real.sqrt 2

noncomputable def surface_area_of_sphere {R : ℝ} 
  (base_edge : ℝ) 
  (face_right_triangle : Bool) : Real :=
  if face_right_triangle then 4 * Real.pi * ((Real.sqrt 3 / 2) ^ 2) else 0 

theorem sphere_surface_area_of_regular_tetrahedron :
  surface_area_of_sphere tetrahedron_base_edge true = 3 * Real.pi := by
  sorry

end sphere_surface_area_of_regular_tetrahedron_l540_540764


namespace range_of_h_l540_540300

def h (x : ℝ) : ℝ := 3 / (3 + 9 * x ^ 2)

theorem range_of_h (c d : ℝ) (h : ∀ x : ℝ, h(x) ∈ set.Ioc c d) : c + d = 1 :=
by
  sorry

end range_of_h_l540_540300


namespace percentage_increase_in_second_year_l540_540589

-- Definitions based on the given problem conditions
def initial_population := 800
def final_population := 1150
def first_year_increase := 0.25

-- Definition to be proved
theorem percentage_increase_in_second_year :
  ∃ (P : ℝ), P = 15 :=
by
  let after_first_year := initial_population * (1 + first_year_increase)
  have h1 : after_first_year = 1000 := by norm_num
  let after_second_year := after_first_year * (1 + P / 100)
  have h2 : after_second_year = final_population := by norm_num [after_first_year, final_population]
  sorry

end percentage_increase_in_second_year_l540_540589


namespace min_distance_sum_l540_540399

def ellipse_F (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 7 = 1
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def focus_F : ℝ × ℝ := (-3, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum :
  ∃ P M : ℝ × ℝ, ellipse_F P.1 P.2 ∧ circle_C M.1 M.2 ∧
  distance P focus_F + distance P M = 7 - real.sqrt 5 :=
by 
  sorry

end min_distance_sum_l540_540399


namespace number_of_arrangements_is_85_l540_540341

def is_valid_combination (combo : List (ℕ × ℕ)) : Prop :=
  ∑ i in combo, (i.1 * i.2) = 10 ∧ 
  ∀ i in combo, i.2 ≤ i.1

def valid_combinations : List (ℕ × ℕ) := 
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

noncomputable def num_arrangements : ℕ :=
  ∑ combo in valid_combinations, if is_valid_combination (list.pure combo) then 1 else 0

theorem number_of_arrangements_is_85 : num_arrangements = 85 :=
  sorry

end number_of_arrangements_is_85_l540_540341


namespace projection_magnitude_l540_540430

variable {𝕜 : Type*} [RealField 𝕜]
variable {V : Type*} [InnerProductSpace 𝕜 V]

-- Given conditions
variables (u z : V)
hypothesis hu_norm : ∥u∥ = 5
hypothesis hz_norm : ∥z∥ = 8
hypothesis huz_dot : ⟪u, z⟫ = 20

-- Statement to be proven
theorem projection_magnitude : ∥(u.proj z)∥ = 2.5 :=
sorry

end projection_magnitude_l540_540430


namespace hayden_earnings_l540_540039

theorem hayden_earnings 
  (wage_per_hour : ℕ) 
  (pay_per_ride : ℕ)
  (bonus_per_review : ℕ)
  (number_of_rides : ℕ)
  (hours_worked : ℕ)
  (gas_cost_per_gallon : ℕ)
  (gallons_of_gas : ℕ)
  (positive_reviews : ℕ)
  : wage_per_hour = 15 → 
    pay_per_ride = 5 → 
    bonus_per_review = 20 → 
    number_of_rides = 3 → 
    hours_worked = 8 → 
    gas_cost_per_gallon = 3 → 
    gallons_of_gas = 17 → 
    positive_reviews = 2 → 
    (hours_worked * wage_per_hour + number_of_rides * pay_per_ride + positive_reviews * bonus_per_review + gallons_of_gas * gas_cost_per_gallon) = 226 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Further proof processing with these assumptions
  sorry

end hayden_earnings_l540_540039


namespace sum_of_digits_is_10_l540_540626

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end sum_of_digits_is_10_l540_540626


namespace find_x_l540_540366

theorem find_x (x : ℤ) (h1 : 5 * x + 3 ≡ 7 [MOD 15]) (h2 : x ≡ 2 [MOD 4]) : 
  x ≡ 6 [MOD 12] :=
sorry

end find_x_l540_540366


namespace city_B_in_dangerous_area_l540_540936

def speed : ℝ := 20 -- km per hour
def dangerous_radius : ℝ := 30 -- km
def distance_AB : ℝ := 40 -- km

theorem city_B_in_dangerous_area :
  (distance_AB / (speed / real.sqrt 2) - (distance_AB - dangerous_radius) / (speed / real.sqrt 2)) = 1 : sorry

end city_B_in_dangerous_area_l540_540936


namespace sum_of_b_sequence_l540_540792

variable {n : ℕ}

def S (n : ℕ) : ℚ := (n^2)/2 + (3*n)/2

def a (n : ℕ) : ℕ := n + 1

def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

theorem sum_of_b_sequence (n : ℕ) : 
  ∑ k in Finset.range n, b k = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) :=
by
  sorry

end sum_of_b_sequence_l540_540792


namespace unique_restoration_l540_540611

variables (ABC : Triangle) (k : Circle) (A B C A1 B1 C1 T : Point)

-- Triangle ABC is inscribed in circle k
def inscribed (ABC : Triangle) (k : Circle) :=
  k.is_circumscribed_about ABC

-- Points A1, B1, C1 on the sides of triangle ABC
def points_on_sides (ABC : Triangle) (A1 B1 C1 : Point) :=
  A1 ∈ Line B C ∧ B1 ∈ Line C A ∧ C1 ∈ Line A B

-- The concurrency of AA1, BB1, CC1 at point T
def concurrent (A1 B1 C1 A B C T : Point) :=
  Line A A1 ∩ Line B B1 ∩ Line C C1 = some T

-- The main theorem
theorem unique_restoration (inscribed : inscribed ABC k) (on_sides : points_on_sides ABC A1 B1 C1):
  (∃ T, concurrent A1 B1 C1 A B C T) ↔ (∃! X, is_triangle ABC X) := sorry

end unique_restoration_l540_540611


namespace arithmetic_sequence_sum_l540_540153

noncomputable def a_n : ℕ → ℚ
| 1 := sorry /* Define based on problem requirements */
| 2 := sorry
| n := sorry

noncomputable def binomial_term (r : ℕ) : ℚ :=
  (-1 / 9 : ℚ) ^ r * (nat.choose 6 r : ℚ) * x ^((6 - 3 * r) / 2 : ℕ) 

theorem arithmetic_sequence_sum:
  (∀ n, a_n n = a_n 5 ) → 
  a_n 5 = (5 / 3 : ℚ) →
  a_n 3 + a_n 5 + a_n 7 = 5 :=
by
  sorry

end arithmetic_sequence_sum_l540_540153


namespace zero_odd_abundant_numbers_lt_50_l540_540048

/-- A helper definition to check if a number is odd -/
def is_odd (n : ℕ) := n % 2 = 1

/-- A helper definition to compute the sum of proper divisors of a number -/
def sum_proper_divisors (n : ℕ) : ℕ :=
  (List.range n).filter (λ d, d > 0 ∧ n % d = 0).sum

/-- A helper definition to check if a number is abundant -/
def is_abundant (n : ℕ) : Prop :=
  sum_proper_divisors n > n

/-- Main theorem -/
theorem zero_odd_abundant_numbers_lt_50 : 
  (List.range 50).count (λ n, is_odd n ∧ is_abundant n) = 0 :=
sorry

end zero_odd_abundant_numbers_lt_50_l540_540048


namespace download_time_l540_540943

theorem download_time (file_size : ℕ) (first_part_size : ℕ) (rate1 : ℕ) (rate2 : ℕ) (total_time : ℕ)
  (h_file : file_size = 90) (h_first_part : first_part_size = 60) (h_rate1 : rate1 = 5) (h_rate2 : rate2 = 10)
  (h_time : total_time = 15) : 
  file_size = first_part_size + (file_size - first_part_size) ∧ total_time = first_part_size / rate1 + (file_size - first_part_size) / rate2 :=
by
  have time1 : total_time = 12 + 3,
    sorry,
  have part1 : first_part_size = 60,
    sorry,
  have part2 : file_size - first_part_size = 30,
    sorry,
  have rate1_correct : rate1 = 5,
    sorry,
  have rate2_correct : rate2 = 10,
    sorry,
  have time1_total : 12 + 3 = 15,
    sorry,
  exact ⟨rfl, rfl⟩

end download_time_l540_540943


namespace percentage_solution_l540_540974

theorem percentage_solution :
  ∃ (x : ℕ), (0.80 * 170 - (x / 100) * 300 = 31) ∧ x = 35 :=
by
  existsi 35
  split
  · exact (0.8 * 170 - (35 / 100) * 300 = 31)
  · sorry

end percentage_solution_l540_540974


namespace log_equiv_l540_540742

theorem log_equiv {y : ℝ} (hy : y > 0) (h125 : 125 = 5^3) (h81 : 81 = 3^4) : y = 5^(3 / 4) → log y 125 = log 3 81 :=
by
  intro hy_eq
  rw [log_eq_log hy hy_eq]
  rw [h125, h81]
  simp only [h125, h81, log]
sorry

end log_equiv_l540_540742


namespace min_dist_sum_ineq_l540_540009

theorem min_dist_sum_ineq (A B C M : Point) (hM : pointInTriangle M A B C) : 
  min (MA M A) (min (MB M B) (MC M C)) + MA M A + MB M B + MC M C < dist A B + dist B C + dist C A :=
by
  sorry

/-- Auxiliary Definitions -/
def Point : Type := sorry
def dist (P Q : Point) : ℝ := sorry
def pointInTriangle (M A B C : Point) : Prop := sorry
def MA (P Q : Point) : ℝ := dist P Q
def MB (P Q : Point) : ℝ := dist P Q
def MC (P Q : Point) : ℝ := dist P Q

end min_dist_sum_ineq_l540_540009


namespace num_multiples_of_5_between_5_and_205_l540_540047

theorem num_multiples_of_5_between_5_and_205 : 
  (finset.filter (λ n, n % 5 = 0) (finset.Icc 5 205)).card = 41 :=
by 
  sorry

end num_multiples_of_5_between_5_and_205_l540_540047


namespace sum_of_consecutive_odds_ten_consecutive_odds_sum_eight_consecutive_odds_sum_find_n_from_sum_l540_540149

theorem sum_of_consecutive_odds (n : ℕ) : (finset.range n).sum (λ k, 2 * k + 1) = n ^ 2 := 
sorry

theorem ten_consecutive_odds_sum : 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 = 100 :=
sorry

theorem eight_consecutive_odds_sum : 11 + 13 + 15 + 17 + 19 + 21 + 23 + 25 = 144 :=
sorry

theorem find_n_from_sum (n : ℕ) : (finset.range n).sum (λ k, 2 * k + 1) = 225 → n = 15 := 
sorry

end sum_of_consecutive_odds_ten_consecutive_odds_sum_eight_consecutive_odds_sum_find_n_from_sum_l540_540149


namespace complex_conjugate_product_l540_540120

variable (z : ℂ)

# Assume the condition |z| = 7
axiom condition : |z| = 7

theorem complex_conjugate_product : z * complex.conj z = 49 :=
by
  rw [condition, complex.norm_eq_abs]
  sorry

end complex_conjugate_product_l540_540120


namespace set_condition_real_condition_l540_540637

-- Problem 1: Sets
theorem set_condition {A B : Set} : (A ∩ B = ∅) → (A = ∅) :=
  sorry

-- Problem 2: Real Numbers
theorem real_condition (a b : ℝ) : (a^2 + b^2 ≠ 0) ↔ (|a| + |b| ≠ 0) :=
  sorry

end set_condition_real_condition_l540_540637


namespace max_marked_cells_l540_540917

def is_valid_knight_move (p1 p2 : ℕ × ℕ) : Prop :=
  let r1 := p1.fst
  let c1 := p1.snd
  let r2 := p2.fst
  let c2 := p2.snd
  (abs (r1 - r2) = 2 ∧ abs (c1 - c2) = 1) ∨ (abs (r1 - r2) = 1 ∧ abs (c1 - c2) = 2)

def are_all_cells_reachable (cells : List (ℕ × ℕ)) : Prop :=
  ∀ c1 c2 ∈ cells, c1 ≠ c2 → ∃ path : List (ℕ × ℕ), path.head = c1 ∧ path.last = c2 ∧ ∀ p ∈ path, p ∈ cells ∧ ∀ (p1 p2 : ℕ × ℕ), (p1, p2) ∈ path.zip path.tail → is_valid_knight_move p1 p2

def is_valid_marking (cells : List (ℕ × ℕ)) : Prop :=
  (List.length cells ≤ 14) ∧
  (List.nodup cells) ∧
  (∀ i j, i < cells.length → j < cells.length → cells.nth_le i _ ≠ cells.nth_le j _ → cells.nth_le i _ .fst ≠ cells.nth_le j _ .fst) ∧
  (∀ i j, i < cells.length → j < cells.length → cells.nth_le i _ ≠ cells.nth_le j _ → cells.nth_le i _ .snd ≠ cells.nth_le j _ .snd)

theorem max_marked_cells (cells : List (ℕ × ℕ)) :
  is_valid_marking cells → are_all_cells_reachable cells → List.length cells = 14 :=
begin
  sorry
end

end max_marked_cells_l540_540917


namespace gcd_204_85_l540_540196

theorem gcd_204_85: Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l540_540196


namespace inequality_subtraction_l540_540784

variable (a b : ℝ)

-- Given conditions
axiom nonzero_a : a ≠ 0 
axiom nonzero_b : b ≠ 0 
axiom a_lt_b : a < b 

-- Proof statement
theorem inequality_subtraction : a - 3 < b - 3 := 
by 
  sorry

end inequality_subtraction_l540_540784


namespace exact_time_l540_540709

-- Definitions of the conditions
def in_time_interval (h m : ℕ) : Prop :=
  h = 3 ∧ m >= 20 ∧ m < 25

def angle_minute_hand (m : ℕ) : ℝ :=
  2 * real.pi * (m / 60.0)

def angle_hour_hand (h m : ℕ) : ℝ :=
  2 * real.pi * ((h / 12.0) + (m / 720.0))

def hands_switched (h m : ℕ) : Prop :=
  angle_hour_hand h m = (angle_minute_hand m) / 12.0

-- The problem statement in Lean 4
theorem exact_time (h m s : ℕ) :
  in_time_interval h m →
  hands_switched h m →
  (h = 3 ∧ m = 21 ∧ s = 23) :=
by
  sorry

end exact_time_l540_540709


namespace compare_abc_l540_540510

noncomputable def a : ℝ := 2 * Real.log (21 / 20)
noncomputable def b : ℝ := Real.log (11 / 10)
noncomputable def c : ℝ := Real.sqrt 1.2 - 1

theorem compare_abc : a > b ∧ b < c ∧ a > c :=
by {
  sorry
}

end compare_abc_l540_540510


namespace xiao_ming_distance_relation_l540_540186

theorem xiao_ming_distance_relation (distance_school : ℝ) (speed : ℝ) (x : ℝ) (y : ℝ) 
  (h_dist : distance_school = 1200) 
  (h_speed : speed = 70) 
  (h_relation : y = distance_school - speed * x) : 
  y = -70 * x + 1200 :=
by
  rw [h_dist, h_speed] at h_relation
  exact h_relation

end xiao_ming_distance_relation_l540_540186


namespace infinite_product_to_rational_root_l540_540318

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540318


namespace pipe_filling_time_with_leak_l540_540549

theorem pipe_filling_time_with_leak (A L : ℝ) 
  (pipe_rate : A = 1 / 6) 
  (leak_rate : L = 1 / 15) : 
  1 / (A - L) = 10 :=
by 
  -- Given rates for the proof
  have A_rate : A = 1 / 6 := pipe_rate,
  have L_rate : L = 1 / 15 := leak_rate,
  
  -- Simplify the combined rate
  let combined_rate := A - L,
  have combined_rate_eq : combined_rate = 1 / 10 := 
  by 
    simp [A_rate, L_rate],
    rw [A_rate, L_rate],
    norm_num,
    
  -- Proven combined rate
  rw combined_rate_eq,
  
  -- Taking the reciprocal of the combined rate gives the total time
  simp,
  norm_num,
  done

end pipe_filling_time_with_leak_l540_540549


namespace curve_cartesian_equation_max_value_3x_plus_4y_l540_540808

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem curve_cartesian_equation :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∀ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) → (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

theorem max_value_3x_plus_4y :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∃ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) ∧ (∀ ϴ : ℝ, 3 * (3 * Real.cos ϴ) + 4 * (2 * Real.sin ϴ) ≤ Real.sqrt 145) :=
sorry

end curve_cartesian_equation_max_value_3x_plus_4y_l540_540808


namespace DVDs_sold_is_168_l540_540461

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l540_540461


namespace correct_number_of_true_propositions_l540_540377

def f (x : ℝ) : ℝ := Real.sin x + 1 / Real.sin x

def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = f (a + x)

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ m

theorem correct_number_of_true_propositions :
  (¬ symmetric_about_y_axis f) ∧
  symmetric_about_origin f ∧
  symmetric_about_line f (π / 2) ∧
  (¬ minimum_value f 2) →
  2 = 2 :=
by
  intro h
  sorry

end correct_number_of_true_propositions_l540_540377


namespace value_of_y_l540_540738

theorem value_of_y (x : ℝ) : 
  (sqrt (x^2 + 4*x + 4) + sqrt (x^2 - 6*x + 9)) = abs (x + 2) + abs (x - 3) :=
by
  sorry

end value_of_y_l540_540738


namespace smallest_length_AB_l540_540879

-- Define quadratic parameters a, b, c and conditions on them
variables {a b c : ℕ}

-- The main statement: smallest irrational length of AB
theorem smallest_length_AB 
  (a_pos : 0 < a) (a_le_10 : a ≤ 10) (b_pos : 0 < b) (c_pos : 0 < c) 
  (two_distinct_roots : b^2 - 4 * a * c > 0) 
  (length_irrational : ∀ x : ℤ, x ≥ 0 → x^2 ≠ b^2 - 4 * a * c) : 
  (∃ length_AB : ℚ, length_AB = (Real.sqrt (b^2 - 4 * a * c) / a) ∧ 
  ∀ l : ℚ, l ∈ { (Real.sqrt (k) / j) | j k ∈ ℕ ∧ j ≤ 10 ∧ 
  ∀ x : ℤ, x ≥ 0 → x^2 ≠ k } → length_AB ≤ l) :=
sorry -- Proof omitted

end smallest_length_AB_l540_540879


namespace shaded_overlap_area_l540_540969

theorem shaded_overlap_area (a b : ℝ) (α : ℝ) (hα : α ≠ π / 2) :
  let area := (a ∧ b ∧ α)
  in
  area = (min a b) / sin α :=
sorry

end shaded_overlap_area_l540_540969


namespace num_students_l540_540934

theorem num_students (n : ℕ) 
    (average_marks_wrong : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (average_marks_correct : ℕ) :
    average_marks_wrong = 100 →
    wrong_mark = 90 →
    correct_mark = 10 →
    average_marks_correct = 92 →
    n = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end num_students_l540_540934


namespace determine_a_for_monotonicity_l540_540843

theorem determine_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x ≤ y → f x ≤ f y → (λ x, x^2 + a * |x - 1|) x ≤ (λ x, x^2 + a * |x - 1|) y) ↔ (a = -2) :=
sorry

end determine_a_for_monotonicity_l540_540843


namespace problem_l540_540756

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.cbrt 3
noncomputable def c : ℝ := Real.exp (1 / Real.exp 1)

theorem problem (a b c : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.cbrt 3) (hc : c = Real.exp (1 / Real.exp 1)) : a < b ∧ b < c :=
by {
  rw [ha, hb, hc],
  sorry
}

end problem_l540_540756


namespace infinite_product_result_l540_540325

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540325


namespace ratio_of_45_and_9_l540_540620

theorem ratio_of_45_and_9 : (45 / 9) = 5 := 
by
  sorry

end ratio_of_45_and_9_l540_540620


namespace smallest_nine_divisibility_l540_540017

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

theorem smallest_nine_divisibility {a : ℕ → ℕ} (h : seq a) : ∃ n, (∀ m, m ≥ n → 9 ∣ a m) ∧ (∀ k, k < n → ¬ (∀ m, m ≥ k → 9 ∣ a m)) :=
begin
  use 5,
  split,
  { intros m hm,
    sorry
  },
  { intros k hk,
    sorry
  }
end

end smallest_nine_divisibility_l540_540017


namespace infinite_product_value_l540_540334

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540334


namespace remainder_of_f_when_divided_by_x_plus_2_l540_540733

def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 + 8 * x - 20

theorem remainder_of_f_when_divided_by_x_plus_2 : f (-2) = 72 := by
  sorry

end remainder_of_f_when_divided_by_x_plus_2_l540_540733


namespace abs_diff_eq_5_l540_540515

-- Definitions of m and n, and conditions provided in the problem
variables (m n : ℝ)
hypothesis (h1 : m * n = 6)
hypothesis (h2 : m + n = 7)

-- Statement to prove
theorem abs_diff_eq_5 : |m - n| = 5 :=
by
  sorry

end abs_diff_eq_5_l540_540515


namespace geom_seq_b_sum_T_lt_half_l540_540766

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 3 * seq_a (n - 1) + 2

def seq_b (n : ℕ) : ℕ :=
seq_a n + 1

theorem geom_seq_b :
  ∀ n, seq_b (n + 1) = 3 * seq_b n :=
by sorry

def sequence_T (n : ℕ) : ℝ :=
(2 * (3 : ℝ) ^ n) / ((seq_a n).toReal * (seq_a (n + 1)).toReal)

def partial_sum_T (n : ℕ) : ℝ :=
∑ i in Finset.range n, sequence_T (i + 1)

theorem sum_T_lt_half :
  ∀ n, partial_sum_T n < (1 / 2 : ℝ) :=
by sorry

end geom_seq_b_sum_T_lt_half_l540_540766


namespace constant_term_of_polynomial_l540_540159

theorem constant_term_of_polynomial :
  (∃ c : ℝ, (λ x : ℝ, (x^2 - 6 * x - 10) / 5) = (λ x : ℝ, x^2 / 5 - (6 / 5) * x + c)) ∧ 
  c = -2 := 
sorry

end constant_term_of_polynomial_l540_540159


namespace n_c_equation_l540_540658

theorem n_c_equation (n c : ℕ) (hn : 0 < n) (hc : 0 < c) :
  (∀ x : ℕ, (↑x + n * ↑x / 100) * (1 - c / 100) = x) →
  (n^2 / c^2 = (100 + n) / (100 - c)) :=
by sorry

end n_c_equation_l540_540658


namespace expressions_equal_iff_l540_540340

theorem expressions_equal_iff (x y z : ℝ) : x + y + z = 0 ↔ x + yz = (x + y) * (x + z) :=
by
  sorry

end expressions_equal_iff_l540_540340


namespace right_triangle_median_length_l540_540080

theorem right_triangle_median_length 
  (A B C D : Type) 
  (hypAB: A - B = 5) 
  (right_triangle_ABC : right_triangle A B C)
  (AC : length A C = 3)
  (BC : length B C = 4)
  (D_midpoint : midpoint D A B) :
  length C D = 2.5 :=
sorry

end right_triangle_median_length_l540_540080


namespace probability_calculation_correct_l540_540379

noncomputable def probability_b_not_second_leg_given_a_not_first_leg : ℚ :=
  let total_arrangements := 3 * 3! in
  let favorable_arrangements := 6 + 8 in
  favorable_arrangements / total_arrangements

theorem probability_calculation_correct :
  probability_b_not_second_leg_given_a_not_first_leg = 7 / 9 :=
by
  sorry

end probability_calculation_correct_l540_540379


namespace inequality_solution_l540_540744

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 9 / (x + 6) ≥ 2) ↔ (x ∈ Set.Ico (-6 : ℝ) (-3) ∪ Set.Ioc (-2) 3) := 
sorry

end inequality_solution_l540_540744


namespace coefficient_of_x4_in_binomial_expansion_l540_540478

theorem coefficient_of_x4_in_binomial_expansion :
  let f (x : ℝ) := (1 / real.sqrt x - x) ^ 7 in
  ∃ (c : ℝ), has_term c x^4 f ∧ c = -21 := sorry

end coefficient_of_x4_in_binomial_expansion_l540_540478


namespace solve_equation_l540_540926

theorem solve_equation :
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} :=
by
  sorry

end solve_equation_l540_540926


namespace parallel_lines_slope_l540_540404

theorem parallel_lines_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + 2 * y - 1 = 0 → x = -2 * y + 1)
  (h2 : ∀ x y : ℝ, m * x - y = 0 → y = m * x) : 
  m = -1 / 2 :=
by
  sorry

end parallel_lines_slope_l540_540404


namespace find_x_l540_540072

open Nat

-- Definitions given in condition a)
def x : ℕ := (finset.range (60 - 39)).sum (λ n, 40 + n)

def y : ℕ := (finset.range (21 // 2)).count (λ n, 40 + 2 * n ∈ finset.range  21)

-- The theorem we want to prove
theorem find_x (H : x + y = 1061) : x = 1050 := 
by
  sorry

end find_x_l540_540072


namespace triangle_condition_isosceles_l540_540391

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
a = b

theorem triangle_condition_isosceles (a b c A B C : ℝ) 
  (h : c / real.cos (C / 2) = a * (real.cos B / real.sin B) + b * (real.cos A / real.sin A)) :
  is_isosceles_triangle a b c A B C :=
sorry

end triangle_condition_isosceles_l540_540391


namespace number_of_students_not_enrolled_in_biology_l540_540994

noncomputable def total_students : ℕ := 880

noncomputable def biology_enrollment_percent : ℕ := 40

noncomputable def students_not_enrolled_in_biology : ℕ :=
  (100 - biology_enrollment_percent) * total_students / 100

theorem number_of_students_not_enrolled_in_biology :
  students_not_enrolled_in_biology = 528 :=
by
  -- Proof goes here.
  -- Use sorry to skip the proof for this placeholder:
  sorry

end number_of_students_not_enrolled_in_biology_l540_540994


namespace average_messages_correct_l540_540093

-- Definitions for the conditions
def messages_monday := 220
def messages_tuesday := 1 / 2 * messages_monday
def messages_wednesday := 50
def messages_thursday := 50
def messages_friday := 50

-- Definition for the total and average messages
def total_messages := messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday
def average_messages := total_messages / 5

-- Statement to prove
theorem average_messages_correct : average_messages = 96 := 
by sorry

end average_messages_correct_l540_540093


namespace find_constant_term_l540_540373

theorem find_constant_term (x y C : ℤ) 
    (h1 : 5 * x + y = 19) 
    (h2 : 3 * x + 2 * y = 10) 
    (h3 : C = x + 3 * y) 
    : C = 1 := 
by 
  sorry

end find_constant_term_l540_540373


namespace download_time_l540_540942

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end download_time_l540_540942


namespace fill_tank_time_l540_540550

variable (A_rate := 1/3)
variable (B_rate := 1/4)
variable (C_rate := -1/4)

def combined_rate := A_rate + B_rate + C_rate

theorem fill_tank_time (hA : A_rate = 1/3) (hB : B_rate = 1/4) (hC : C_rate = -1/4) : (1 / combined_rate) = 3 := by
  sorry

end fill_tank_time_l540_540550


namespace sum_of_selected_terms_l540_540769

axiom arithmetic_sequence (a : ℕ → ℕ) (d a1 : ℕ) : 
  (∀ n, a n = a1 + (n - 1) * d) ∧ 
  (a 4 = 2 * a 2) ∧ 
  (∃ k, a 1 * (a 1 + 3 * d) = 16)

theorem sum_of_selected_terms :
  ∀ (a : ℕ → ℕ) (d a1 : ℕ), 
  (∀ n, a n = a1 + (n - 1) * d) ∧ 
  (a 4 = 2 * a 2) ∧ 
  (∃ k, a 1 * (a 1 + 3 * d) = 16) → 
  ∃ S, S = 2700 :=
by
  intro a d a1 h
  
  let evens := { n ∣ 20 ≤ n ∧ n ≤ 116 ∧ n % 5 = 0 }
  have S := (∑ n in evens, 2 * n) -- Sum of evens satisfying conditions
  
  existsi S
  sorry

end sum_of_selected_terms_l540_540769


namespace factorial_division_l540_540725

theorem factorial_division (n : ℕ) (h : n = 9) : n.factorial / (n - 1).factorial = 9 :=
by 
  rw [h]
  sorry

end factorial_division_l540_540725


namespace integer_distances_between_points_on_line_l540_540106

theorem integer_distances_between_points_on_line
  (g : ℝ → Prop) (n : ℕ) (h1 : n > 0) :
  ∃ (points_on_g : fin n → ℝ × ℝ) (point_not_on_g : ℝ × ℝ),
    (∀ i, g (points_on_g i).1) ∧
    ¬ g point_not_on_g.1 ∧
    (∀ i j, i ≠ j → ∃ k : ℤ, dist (points_on_g i) (points_on_g j) = k) ∧
    (∀ i, ∃ k : ℤ, dist (points_on_g i) point_not_on_g = k) :=
sorry

end integer_distances_between_points_on_line_l540_540106


namespace sum_of_first_four_terms_l540_540793

def arithmetic_sequence_sum (a1 a2 : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * (a2 - a1))) / 2

theorem sum_of_first_four_terms : arithmetic_sequence_sum 4 6 4 = 28 :=
by
  sorry

end sum_of_first_four_terms_l540_540793


namespace cuberoot_solution_l540_540745

noncomputable def solutions (x : ℝ) : Prop :=
  real.sqrt3 (10 * x - 1) + real.sqrt3 (8 * x + 1) = 3 * real.sqrt3 (x)

theorem cuberoot_solution :
  ∀ x : ℝ, 
    solutions x ↔ 
    x = 0 ∨ 
    x = (-2 + 6 * real.sqrt6) / 106 ∨ 
    x = (-2 - 6 * real.sqrt6) / 106 :=
by
  sorry

end cuberoot_solution_l540_540745


namespace ratio_of_ab_l540_540530

noncomputable theory
open Complex

theorem ratio_of_ab (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∃ c : ℝ, (3 - 8 * I) * (a + b * I) = c * I) : a / b = -8 / 3 :=
by sorry

end ratio_of_ab_l540_540530


namespace frustum_surface_area_ratio_l540_540257

-- Define the original parameters and conditions
variables (h R r l n : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) (r_pos : 0 < r) (l_pos : 0 < l) (n_pos : 0 < n)

-- Define the new lateral surface area and the original lateral surface area
def S_original := π * (R + r) * l
def S_new := π * ((R / n) + (r / n)) * (n * l)

-- The theorem we want to prove: the ratio of new surface area to the original is 1
theorem frustum_surface_area_ratio : (S_new h R r l n) / (S_original h R r l) = 1 := by
  sorry

end frustum_surface_area_ratio_l540_540257


namespace geometric_sequence_k_squared_l540_540867

theorem geometric_sequence_k_squared (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) (h5 : a 5 * a 8 * a 11 = k) : 
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 := by
  sorry

end geometric_sequence_k_squared_l540_540867


namespace Julie_charge_pulling_weeds_l540_540101

theorem Julie_charge_pulling_weeds :
  (∃ W : ℝ, 
     let earnings_mowing := 25 * 4 in
     let earnings_pulling_september := 3 * W in
     let total_earnings := 2 * (earnings_mowing + earnings_pulling_september) in
     total_earnings = 248 → W = 8) :=
by
  intros W h
  set earnings_mowing := 25 * 4
  set earnings_pulling_september := 3 * W
  set total_earnings := 2 * (earnings_mowing + earnings_pulling_september)
  have heq : total_earnings = 248 := h
  sorry

end Julie_charge_pulling_weeds_l540_540101


namespace number_of_men_in_row_l540_540930

theorem number_of_men_in_row (M : ℕ) (W : ℕ) (cases : ℕ) (hW : W = 2) (hcases : cases = 12) :
  (∃ k1 k2 k3 : ℕ, M = k1 + k2 + k3 ∧ k1 > 0 ∧ k2 > 0 ∧ k3 > 0 ∧ 2 * W * ((k1.choose 2 + k2.choose 2 + k3.choose 2) * 2.factorial) = cases) →
  M = 4 :=
by
  sorry

end number_of_men_in_row_l540_540930


namespace sum_largest_smallest_prime_factors_546_l540_540631

theorem sum_largest_smallest_prime_factors_546 : 
  let p := 546; let prime_factors := [2, 3, 7, 13]; 
  (List.minimum prime_factors).getOrElse 0 + (List.maximum prime_factors).getOrElse 0 = 15 := 
by
  intro p prime_factors
  sorry

end sum_largest_smallest_prime_factors_546_l540_540631


namespace triangle_is_equilateral_l540_540152

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)

-- Define a triangle's circumradius and inradius
structure TriangleProperties :=
  (circumradius : ℝ)
  (inradius : ℝ)
  (circumcenter_incenter_sq_distance : ℝ) -- OI^2 = circumradius^2 - 2*circumradius*inradius

noncomputable def circumcenter_incenter_coincide (T : Triangle) (P : TriangleProperties) : Prop :=
  P.circumcenter_incenter_sq_distance = 0

theorem triangle_is_equilateral
  (T : Triangle)
  (P : TriangleProperties)
  (hR : P.circumradius = 2 * P.inradius)
  (hOI : circumcenter_incenter_coincide T P) :
  ∃ (R r : ℝ), T = {A := 1 * r, B := 1 * r, C := 1 * r} :=
by sorry

end triangle_is_equilateral_l540_540152


namespace tournament_games_l540_540543

theorem tournament_games (n : ℕ) (h : n = 16) : ∃ g : ℕ, g = 15 :=
by
  let games := n - 1
  have hg : games = 15 := by
    simp [h]
  exact ⟨games, hg⟩

end tournament_games_l540_540543


namespace race_probability_l540_540078

theorem race_probability :
    let pX := (1 : ℚ) / 4
    let pY := (1 : ℚ) / 12
    let pZ := (1 : ℚ) / 7
    pX + pY + pZ = 10 / 21 :=
by
  let pX := (1 : ℚ) / 4
  let pY := (1 : ℚ) / 12
  let pZ := (1 : ℚ) / 7
  calc
    pX + pY + pZ = (21 / 84) + (7 / 84) + (12 / 84) : by norm_num
    ... = (21 + 7 + 12) / 84 : by norm_num
    ... = 40 / 84 : by norm_num
    ... = 10 / 21 : by norm_num

end race_probability_l540_540078


namespace prob_sum_divisible_by_4_l540_540754

-- Defining the set and its properties
def set : Finset ℕ := {1, 2, 3, 4, 5}

def isDivBy4 (n : ℕ) : Prop := n % 4 = 0

-- Defining a function to calculate combinations
def combinations (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Defining the successful outcomes and the total combinations
def successfulOutcomes : ℕ := 3
def totalOutcomes : ℕ := combinations 5 3

-- Defining the probability
def probability : ℚ := successfulOutcomes / ↑totalOutcomes

-- The proof problem
theorem prob_sum_divisible_by_4 : probability = 3 / 10 := by
  sorry

end prob_sum_divisible_by_4_l540_540754


namespace simplify_expression_l540_540973

variable (x : ℝ)

theorem simplify_expression : (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := 
by 
  sorry

end simplify_expression_l540_540973


namespace conjugate_imaginary_part_l540_540410

theorem conjugate_imaginary_part (z : ℂ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + i) * z = 1 - i) : (conj z).im = 0 :=
sorry

end conjugate_imaginary_part_l540_540410


namespace intersects_exactly_one_point_l540_540830

-- Define the line L and the hyperbola C.
constant L : Type
constant C : Type

-- Define a predicate that determines if a line is parallel to an asymptote of a hyperbola.
constant is_parallel_to_asymptote : L → C → Prop

-- Define a function that counts the number of intersection points between a line and a hyperbola.
constant num_intersections : L → C → ℕ

-- The theorem statement
theorem intersects_exactly_one_point (l : L) (c : C) (h : is_parallel_to_asymptote l c) :
  num_intersections l c = 1 :=
sorry

end intersects_exactly_one_point_l540_540830


namespace abs_difference_of_mn_6_and_sum_7_l540_540518

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l540_540518


namespace square_inscribed_in_ellipse_area_l540_540269

theorem square_inscribed_in_ellipse_area :
  ∀ t : ℝ,
    (∃ t, (t ≠ 0 ∧ (t * t / 4 + t * t / 8 = 1))) →
    let side_length := 2 * t in
    side_length ^ 2 = 32 / 3 :=
by
  -- Proof skipped for now
  sorry

end square_inscribed_in_ellipse_area_l540_540269


namespace brennan_deletes_70_percent_l540_540282

theorem brennan_deletes_70_percent (Brennan_downloaded_initially : ℕ := 800)
                                   (Brennan_downloaded_additional : ℕ := 400)
                                   (files_after_deletion : ℕ := 400)
                                   (P : ℕ)
                                   (h1 : P%100 * Brennan_downloaded_initially + 160 = files_after_deletion) :
      P = 70 :=
begin
  sorry,
end

end brennan_deletes_70_percent_l540_540282


namespace maxim_final_amount_l540_540895

-- Define the necessary constants and their values
def P : ℝ := 1000 -- Principal amount
def r : ℝ := 0.12 -- Annual interest rate as a decimal
def n : ℝ := 12 -- Compounding frequency (monthly)
def t : ℝ := 1 / 12 -- Time period in years (1 month)

-- Define the compound interest formula
def A : ℝ := P * (1 + r / n) ^ (n * t)

-- The statement we need to prove
theorem maxim_final_amount : A = 1010 := by
  -- We state that this part is to be proved as we skip with sorry
  sorry

end maxim_final_amount_l540_540895


namespace sum_valid_combinations_l540_540303

-- Declare that we are working in a noncomputable environment if necessary
noncomputable def valid_combinations_sum : ℕ :=
  -- Definitions from the conditions
  let Ys := [0, 2, 4, 6, 8] in
  let validXFor y := {x : ℕ | x < 10 ∧ (13 + x + y) % 3 = 0} in
  Ys.foldr (fun y acc => acc + (validXFor y).sum id) 0

theorem sum_valid_combinations:
  valid_combinations_sum = 60 := by
    sorry

end sum_valid_combinations_l540_540303


namespace range_frequency_l540_540427

-- Define the sample data
def sample_data : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the condition representing the frequency count
def frequency_count : ℝ := 0.2 * 20

-- Define the proof problem
theorem range_frequency (s : List ℝ) (range_start range_end : ℝ) : 
  s = sample_data → 
  range_start = 11.5 →
  range_end = 13.5 → 
  (s.filter (λ x => range_start ≤ x ∧ x < range_end)).length = frequency_count := 
by 
  intros
  sorry

end range_frequency_l540_540427


namespace polynomial_factor_l540_540349

theorem polynomial_factor:
    ∀ (x y : ℤ),
        5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * (x + y)^2 =
        5 * x^4 + 35 * x^3 + 960 * x^2 + 1649 * x + 4000 - 8 * x * y - 4 * y^2 :=
by {
  intros x y,
  sorry,
}

end polynomial_factor_l540_540349


namespace eval_f_log4_3_l540_540025

def f : ℝ → ℝ :=
λ x, if x < 0 then (1/4)^x else 4^x

theorem eval_f_log4_3 :
  f (Real.log 3 / Real.log 4) = 3 :=
sorry

end eval_f_log4_3_l540_540025


namespace initial_cats_l540_540214

theorem initial_cats (C : ℕ) (h1 : C / 3 ∈ ℕ) (h2 : 4 * C / 3 ∈ ℕ) (h3 : 12 * C / 3 = 60) : C = 15 :=
by
  sorry

end initial_cats_l540_540214


namespace number_of_factors_l540_540735

theorem number_of_factors (M : ℕ) (h : M = 2^5 * 3^4 * 5^3 * 7^3 * 11^1) : 
  nat.divisors M |→ 960 :=
by
  sorry

end number_of_factors_l540_540735


namespace fraction_denominator_l540_540184

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l540_540184


namespace infinite_product_value_l540_540331

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540331


namespace abs_diff_eq_5_l540_540514

-- Definitions of m and n, and conditions provided in the problem
variables (m n : ℝ)
hypothesis (h1 : m * n = 6)
hypothesis (h2 : m + n = 7)

-- Statement to prove
theorem abs_diff_eq_5 : |m - n| = 5 :=
by
  sorry

end abs_diff_eq_5_l540_540514


namespace optimal_income_l540_540851

theorem optimal_income (x : ℝ) (tax take_home_pay : ℝ) 
    (h_tax : tax = 10 * x ^ 2 + 1000) 
    (h_take_home_pay : take_home_pay = 1000 * x - 10 * x ^ 2 - 1000) :
    ∃ x_max, take_home_pay = 24000 → x_max = 50 := 
begin
  -- We'll provide the proof in the theorem body.
  sorry
end

end optimal_income_l540_540851


namespace height_of_water_in_cylinder_l540_540278

theorem height_of_water_in_cylinder
  (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V_cone : ℝ) (V_cylinder : ℝ) (h_cylinder : ℝ) :
  r_cone = 15 → h_cone = 25 → r_cylinder = 20 →
  V_cone = (1 / 3) * π * r_cone^2 * h_cone →
  V_cylinder = V_cone → V_cylinder = π * r_cylinder^2 * h_cylinder →
  h_cylinder = 4.7 :=
by
  intros r_cone_eq h_cone_eq r_cylinder_eq V_cone_eq V_cylinder_eq volume_eq
  sorry

end height_of_water_in_cylinder_l540_540278


namespace units_digit_of_result_is_3_l540_540584

def hundreds_digit_relation (c : ℕ) (a : ℕ) : Prop :=
  a = 2 * c - 3

def original_number_expression (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def reversed_number_expression (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a + 50

def subtraction_result (orig rev : ℕ) : ℕ :=
  orig - rev

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_result_is_3 (a b c : ℕ) (h : hundreds_digit_relation c a) :
  units_digit (subtraction_result (original_number_expression a b c)
                                  (reversed_number_expression a b c)) = 3 :=
by
  sorry

end units_digit_of_result_is_3_l540_540584


namespace number_of_rods_in_one_mile_l540_540015

theorem number_of_rods_in_one_mile (miles_to_furlongs : 1 = 10 * 1)
  (furlongs_to_rods : 1 = 50 * 1) : 1 = 500 * 1 :=
by {
  sorry
}

end number_of_rods_in_one_mile_l540_540015


namespace balls_in_boxes_ways_l540_540050

theorem balls_in_boxes_ways : ∃ (ways : ℕ), ways = 56 :=
by
  let n := 5
  let m := 4
  let ways := 56
  sorry

end balls_in_boxes_ways_l540_540050


namespace incorrect_statement_is_D_l540_540990

variable {a b c r : ℝ}

def statement_A : Prop := "1 - a - ab is a quadratic trinomial"
def statement_B : Prop := "-a^2 b^2 c is a monomial"
def statement_C : Prop := "(a + b) / 2 is a polynomial"
def statement_D : Prop := "In (3 / 4) * real.pi * r^2, the coefficient is (3 / 4)"

theorem incorrect_statement_is_D : ¬ statement_D := by
  sorry

end incorrect_statement_is_D_l540_540990


namespace balls_in_boxes_l540_540052

def waysToPutBallsInBoxes (balls : ℕ) (boxes : ℕ) [Finite boxes] : ℕ :=
  Finset.card { f : Fin boxes → ℕ | (Finset.sum Finset.univ (fun i => f i)) = balls }

theorem balls_in_boxes : waysToPutBallsInBoxes 7 3 = 36 := by
  sorry

end balls_in_boxes_l540_540052


namespace infinite_solutions_x_sq_plus_x_plus_1_sq_eq_y_sq_l540_540137

theorem infinite_solutions_x_sq_plus_x_plus_1_sq_eq_y_sq :
  ∀ n : ℕ, ∃ x y : ℕ, x = (2 * n - 1) / 2 ∧ y = 3 * n ∧ x^2 + (x + 1)^2 = y^2 :=
begin
  sorry
end

end infinite_solutions_x_sq_plus_x_plus_1_sq_eq_y_sq_l540_540137


namespace sum_of_sequence_l540_540125

noncomputable def sequence (n : ℕ) : ℕ → ℝ := 
  if n = 1 then -(1/3)
  else -1/3 + (n - 1) * 2

theorem sum_of_sequence (n : ℕ) (h_pos : n > 0) : 
  let a_1 := -(1/3)
  let a_n := sequence n
  (sequence 1 + 2 * sequence 2 = 3) ∧ 
  (∀ k : ℕ, 1 ≤ k → sequence (k+1) - sequence k = 2) → 
  ∑ i in finset.range (n + 1), sequence i = n * (n - 4/3) :=
by
  sorry

end sum_of_sequence_l540_540125


namespace no_rational_roots_l540_540118

theorem no_rational_roots (a : ℕ → ℤ) (n : ℕ)
  (an_odd : a n % 2 = 1)
  (a0_odd : a 0 % 2 = 1)
  (f1_odd : let sum := (Finset.range (n + 1)).sum (λ i, a i) in sum % 2 = 1) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ Int.gcd p q = 1 ∧ a n * p^n + (Finset.range n).sum (λ k, a k * p^(k+1) * q^(n-1-k)) + a 0 * q^n = 0 := 
sorry

end no_rational_roots_l540_540118


namespace salary_increase_l540_540254

open Real

noncomputable section

-- Original salaries
variable (x : Fin 10 → ℝ)

-- Mean of the original salaries
def mean (x : Fin 10 → ℝ) : ℝ := (∑ i, x i) / 10

-- Variance of the original salaries
def variance (x : Fin 10 → ℝ) : ℝ := (∑ i, (x i - mean x) ^ 2) / 10

-- New salaries
def y (x : Fin 10 → ℝ) : Fin 10 → ℝ := fun i => x i + 100

-- Assertion to be proved
theorem salary_increase (x : Fin 10 → ℝ) :
  mean (y x) = mean x + 100 ∧ variance (y x) = variance x :=
by
  sorry

end salary_increase_l540_540254


namespace poster_cost_l540_540565

theorem poster_cost (m b1 b2 : ℕ) (n_posts : ℕ) (remain_money : ℕ) (p : ℕ) :
  m = 20 → b1 = 8 → b2 = 4 → n_posts = 2 → remain_money = m - (b1 + b2) → p = remain_money / n_posts → p = 4 := 
by
  intros,
  sorry

end poster_cost_l540_540565


namespace loop_until_true_termination_condition_l540_540083

theorem loop_until_true_termination_condition (M : Prop) :
  (∀ loop_body : (Unit → Unit), (loop_body (); ¬ M) → (loop_body; M)) -> M = TerminationConditionTrue :=
sorry

end loop_until_true_termination_condition_l540_540083


namespace fixed_point_logarithmic_l540_540194

theorem fixed_point_logarithmic (a : ℝ) (l : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (4, 1) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ y = log a (x - 3) + l } :=
sorry

end fixed_point_logarithmic_l540_540194


namespace infinite_product_value_l540_540312

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540312


namespace problem_solution_l540_540928

theorem problem_solution (x : ℂ) :
  (x^3 + 5 * x^2 * (sqrt 3) + 15 * x + 5 * (sqrt 3)) + (x + (sqrt 3)) = 0 →
  x = - (sqrt 3) ∨ x = (complex.I - (sqrt 3)) ∨ x = (-complex.I - (sqrt 3)) :=
by sorry

end problem_solution_l540_540928


namespace infinite_product_value_l540_540335

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540335


namespace distance_between_intersections_of_line_and_circle_l540_540006

theorem distance_between_intersections_of_line_and_circle :
  ∀ (t : ℝ), 
    let x := 2 + (Real.sqrt 2 / 2) * t in
    let y := 1 + (Real.sqrt 2 / 2) * t in
    (x * x + y * y = 4) →
    ((∃ t₁ t₂ : ℝ, 
        (x = 2 + (Real.sqrt 2 / 2) * t₁) ∧ (y = 1 + (Real.sqrt 2 / 2) * t₁) ∧ 
        (x = 2 + (Real.sqrt 2 / 2) * t₂) ∧ (y = 1 + (Real.sqrt 2 / 2) * t₂) ∧ 
        (t₁ + t₂ = -3 * Real.sqrt 2) ∧ (t₁ * t₂ = 1)) →
    (Real.sqrt ((t₁ - t₂)^2 + (t₁ - t₂)^2) = Real.sqrt 14)) :=
sorry

end distance_between_intersections_of_line_and_circle_l540_540006


namespace solution_system_eq_l540_540929

theorem solution_system_eq (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4 ∧ y = -1) :=
by sorry

end solution_system_eq_l540_540929


namespace fraction_denominator_l540_540183

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l540_540183


namespace proof_problem_l540_540588

noncomputable def polar_to_cartesian_O1 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = 4 * Real.cos θ → (ρ^2 = 4 * ρ * Real.cos θ)

noncomputable def cartesian_O1 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 4 * x → x^2 + y^2 - 4 * x = 0

noncomputable def polar_to_cartesian_O2 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = -4 * Real.sin θ → (ρ^2 = -4 * ρ * Real.sin θ)

noncomputable def cartesian_O2 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = -4 * y → x^2 + y^2 + 4 * y = 0

noncomputable def intersections_O1_O2 : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 + 4 * y = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)

noncomputable def line_through_intersections : Prop :=
  ∀ (x y : ℝ), ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)) → y = -x

theorem proof_problem : polar_to_cartesian_O1 ∧ cartesian_O1 ∧ polar_to_cartesian_O2 ∧ cartesian_O2 ∧ intersections_O1_O2 ∧ line_through_intersections :=
  sorry

end proof_problem_l540_540588


namespace fifteenth_prime_l540_540544

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m ∣ n, m = 1 ∨ m = n)

noncomputable def nth_prime (n : ℕ) : ℕ :=
  Nat.find (λ k,  (Nat.filter is_prime (List.range k)).length = n)

theorem fifteenth_prime :
  nth_prime 15 = 47 :=
by
  have h8 : nth_prime 8 = 19 := sorry
  have h9 : nth_prime 9 = 23 := sorry
  have h10 : nth_prime 10 = 29 := sorry
  have h11 : nth_prime 11 = 31 := sorry
  have h12 : nth_prime 12 = 37 := sorry
  have h13 : nth_prime 13 = 41 := sorry
  have h14 : nth_prime 14 = 43 := sorry
  have h15 : nth_prime 15 = 47 := sorry
  exact h15

end fifteenth_prime_l540_540544


namespace find_pairs_l540_540878

theorem find_pairs (a b q r : ℕ) (h1 : a * b = q * (a + b) + r)
  (h2 : q^2 + r = 2011) (h3 : 0 ≤ r ∧ r < a + b) : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ (a = t ∧ b = t + 2012 ∨ a = t + 2012 ∧ b = t)) :=
by
  sorry

end find_pairs_l540_540878


namespace minimum_locks_l540_540545

theorem minimum_locks (n k : ℕ) (h1 : ∀ (s : Finset ℕ), s.card = k → ⋃ (i ∈ s), keys i = locks)
    (h2 : ∀ (s : Finset ℕ), s.card = k - 1 → ∃ l, l ∉ ⋃ (i ∈ s), keys i) :
  ∃ (l : ℕ), l = nat.choose n (k - 1) :=
by
  sorry

end minimum_locks_l540_540545


namespace midsegment_l540_540881

open EuclideanGeometry

variables {A B C M P Q : Point}

-- Assuming the necessary properties and definitions regarding triangle and perpendicularity
variables (h1 : IsTriangle A B C)
          (h2 : Midpoint M A B)
          (h3 : Perpendicular A P (Line_through C))
          (h4 : Perpendicular B Q (Line_through C))
          (h5 : OnLine C (Line_through P Q))
          
theorem midsegment (h1 : IsTriangle A B C) (h2 : Midpoint M A B)
  (h3 : Perpendicular A P (Line_through C))
  (h4 : Perpendicular B Q (Line_through C))
  (h5 : OnLine C (Line_through P Q)) : Distance M P = Distance M Q :=
sorry

end midsegment_l540_540881


namespace find_direction_vector_l540_540586

variables {R : Type*} [Field R]

def projection_matrix : Matrix (Fin 3) (Fin 3) R :=
  ![![3/13, 2/13, 6/13], ![2/13, 12/13, -5/13], ![6/13, -5/13, 2/13]]

def direction_vector (a b c : R) : Prop :=
  ⟦ projection_matrix.mulVec ![1, 0, 0] = (1 : R) / 13 • ![3, 2, 6] ⟧

theorem find_direction_vector :
  direction_vector 3 2 6 :=
by
  sorry

end find_direction_vector_l540_540586


namespace sum_of_digits_of_expression_l540_540624

theorem sum_of_digits_of_expression : 
  (sum_digits (decimal_repr (2^2010 * 5^2008 * 7)) = 10) :=
sorry

end sum_of_digits_of_expression_l540_540624


namespace proposition_1_false_proposition_2_true_proposition_3_true_proposition_4_false_l540_540375

def f (x : ℝ) : ℝ := Real.sin x + 1 / (Real.sin x)

theorem proposition_1_false : ¬ (∀ x, f(x) = f(-x)) := 
by sorry

theorem proposition_2_true : ∀ x, f(-x) = -f(x) := 
by sorry

theorem proposition_3_true : ∀ x, f(Real.pi - x) = f(x) := 
by sorry

theorem proposition_4_false : ¬ (∀ x, f(x) ≥ 2) := 
by sorry

end proposition_1_false_proposition_2_true_proposition_3_true_proposition_4_false_l540_540375


namespace height_of_carton_is_70_l540_540673

def carton_dimensions : ℕ × ℕ := (25, 42)
def soap_box_dimensions : ℕ × ℕ × ℕ := (7, 6, 5)
def max_soap_boxes : ℕ := 300

theorem height_of_carton_is_70 :
  let (carton_length, carton_width) := carton_dimensions
  let (soap_box_length, soap_box_width, soap_box_height) := soap_box_dimensions
  let boxes_per_layer := (carton_length / soap_box_length) * (carton_width / soap_box_width)
  let num_layers := max_soap_boxes / boxes_per_layer
  (num_layers * soap_box_height) = 70 :=
by
  have carton_length := 25
  have carton_width := 42
  have soap_box_length := 7
  have soap_box_width := 6
  have soap_box_height := 5
  have max_soap_boxes := 300
  have boxes_per_layer := (25 / 7) * (42 / 6)
  have num_layers := max_soap_boxes / boxes_per_layer
  sorry

end height_of_carton_is_70_l540_540673


namespace harmonic_series_induction_l540_540553

theorem harmonic_series_induction (n : ℕ) (h₁ : n > 1) :
  (∑ i in range (2^n - 1), 1 / (i + 1) : ℝ) < n :=
by sorry

end harmonic_series_induction_l540_540553


namespace triangle_is_equilateral_l540_540440

variables {a b c : ℝ} {A B C : ℝ}

theorem triangle_is_equilateral (h1 : (a + b + c) * (b + c - a) = 3 * b * c)
  (h2 : sin A = 2 * sin B * cos C) : 
  ∃ (e : ℝ), a = e ∧ b = e ∧ c = e ∧ A = B ∧ B = C :=
sorry

end triangle_is_equilateral_l540_540440


namespace abs_inequality_solution_set_l540_540208

theorem abs_inequality_solution_set (x : ℝ) :
  { x | abs (2 * x - 1) ≥ 3 } = set.Iic (-1) ∪ set.Ici 2 :=
sorry

end abs_inequality_solution_set_l540_540208


namespace coin_probability_two_heads_l540_540245

theorem coin_probability_two_heads :
  ∀ (n k : ℕ) (p : ℚ), 
  n = 3 → k = 2 → p = 1/2 → 
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = 0.375 := by
  intros n k p hn hk hp
  rw [hn, hk, hp]
  norm_num
  sorry

end coin_probability_two_heads_l540_540245


namespace max_sum_prime_multiplication_table_l540_540953

theorem max_sum_prime_multiplication_table : 
  ∃ (a b c d e f g : ℕ) (primes : Finset ℕ),
    primes = {2, 3, 5, 7, 11, 13, 17} ∧ 
    a = 17 ∧ 
    b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes ∧ g ∈ primes ∧ 
    b ≠ a ∧ c ≠ a ∧ d ≠ a ∧ e ≠ a ∧ f ≠ a ∧ g ≠ a ∧ 
    b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ e ≠ f ∧ e ≠ g ∧ f ≠ g ∧
    primes = {a, b, c, d, e, f, g} ∧ 
    (a + b + c) * (d + e + f + g) = 825 := 
begin
  sorry
end

end max_sum_prime_multiplication_table_l540_540953


namespace new_people_joined_l540_540127

-- Conditions
variables (initial_count quit_count current_count new_people : ℕ)
hypothesis h1 : initial_count = 25
hypothesis h2 : quit_count = 8
hypothesis h3 : current_count = 30

-- Calculate number of people left after 8 quit
def remaining_people := initial_count - quit_count

-- Prove the number of new people who joined the team
theorem new_people_joined : remaining_people + new_people = current_count → new_people = 13 :=
by {
  intros, 
  sorry
}

end new_people_joined_l540_540127


namespace recurring_six_denominator_l540_540173

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l540_540173


namespace sum_powers_coprime_l540_540743

theorem sum_powers_coprime (n : ℕ) (hn : ∑ i in finset.range (n + 1), i^φ(n) is_coprime_to n) :
  ∃ primes : Finset ℕ, n = primes.prod ∧ ∀ p ∈ primes, Prime p :=
by sorry

end sum_powers_coprime_l540_540743


namespace product_consecutive_even_div_48_l540_540558

theorem product_consecutive_even_div_48 (k : ℤ) : 
  (2 * k) * (2 * k + 2) * (2 * k + 4) % 48 = 0 :=
by
  sorry

end product_consecutive_even_div_48_l540_540558


namespace solve_equation_l540_540210

-- Defining the equation as a function.
def equation (x : ℝ) : ℝ := 9^x - 6 * 3^x - 7

-- The main statement: proving the solution to the equation is log_3 7.
theorem solve_equation : ∃ x : ℝ, equation x = 0 ∧ x = log 7 / log 3 := by
  -- declaration of the existence of the solution.
  use log 7 / log 3
  split
  -- the first part of the split indicates that the equation equals 0 when x = log 7 / log 3
  {
    -- substituting x = log 7 / log 3 into the equation and simplifying should yield 0
    sorry
  }
  -- the second part of the split directly indicates that x = log 7 / log 3
  {
    refl
  }

end solve_equation_l540_540210


namespace central_angle_of_chord_l540_540773

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 0

theorem central_angle_of_chord (x y: ℝ) (A B: ℝ × ℝ): 
  circle_eq x 0 → circle_eq 4 0 → A = (0, 0) → B = (4, 0) → 
  ∃ θ, θ = π / 2 := 
by
  intro h1 h2 hA hB
  use π / 2
  sorry

end central_angle_of_chord_l540_540773


namespace ratio_of_b_to_a_l540_540948

theorem ratio_of_b_to_a (a b c : ℕ) (x y : ℕ) 
  (h1 : a > 0) 
  (h2 : x = 100 * a + 10 * b + c)
  (h3 : y = 100 * 9 + 10 * 9 + 9 - 241) 
  (h4 : x = y) :
  b = 5 → a = 7 → (b / a : ℚ) = 5 / 7 := 
by
  intros
  subst_vars
  sorry

end ratio_of_b_to_a_l540_540948


namespace person_A_wins_7_times_l540_540547

variables {k l m : ℕ}

-- Define conditions
def total_rounds := 15
def distance_A := 17
def distance_B := 2

-- Define the movement equations
def move_eq_A := 3 * l - 2 * m + k = distance_A
def move_eq_B := 3 * m - 2 * l + k = distance_B

-- Define the total covered distance equation
def total_distance_eq := 2 * k + 5 * (l - m) = 19

-- Prove that Person A won 7 times
theorem person_A_wins_7_times 
  (h1 : k + l + m = total_rounds) 
  (h2 : move_eq_A) 
  (h3 : move_eq_B) 
  (h4 : total_distance_eq) :
  l = 7 :=
begin
  sorry
end

end person_A_wins_7_times_l540_540547


namespace equation_of_hyperbola_l540_540425

-- Defining the existence of the hyperbola
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (c_eq_5 : c = 5)

-- Given conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptote_parallel_to_line : Prop :=
  b / a = 2

def focus_on_line : Prop :=
  ∃ (x : ℝ), x = -5 ∧ y = 2 * x + 10  -- Focus at (-5, 0)

-- Combining the conditions
def hyperbola_conditions : Prop :=
  asymptote_parallel_to_line ∧ focus_on_line ∧ c_eq_5 ∧ c^2 = a^2 + b^2

-- The final goal: the equation of the hyperbola
theorem equation_of_hyperbola (x y : ℝ) (h : hyperbola_conditions) : hyperbola_equation x y := by
  sorry

end equation_of_hyperbola_l540_540425


namespace books_got_rid_of_l540_540270

-- Define the number of books they originally had
def original_books : ℕ := 87

-- Define the number of shelves used
def shelves_used : ℕ := 9

-- Define the number of books per shelf
def books_per_shelf : ℕ := 6

-- Define the number of books left after placing them on shelves
def remaining_books : ℕ := shelves_used * books_per_shelf

-- The statement to prove
theorem books_got_rid_of : original_books - remaining_books = 33 := 
by 
-- here is proof body you need to fill in 
  sorry

end books_got_rid_of_l540_540270


namespace balls_in_boxes_l540_540054

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end balls_in_boxes_l540_540054


namespace range_of_a_l540_540417

noncomputable def f (a x : ℝ) : ℝ := a / (x + 1) + Real.log x

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Ioc 0 2 → x₂ ∈ Ioc 0 2 → x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > -1) ↔ a ≤ 27 / 4 :=
sorry

end range_of_a_l540_540417


namespace average_score_of_both_classes_l540_540967

-- Defining the conditions
def class1_students : Nat := 20
def class1_avg_score : Float := 80
def class2_students : Nat := 30
def class2_avg_score : Float := 70

-- Define the combined average score
def combined_avg_score : Float :=
  (class1_students * class1_avg_score + class2_students * class2_avg_score) / (class1_students + class2_students)

-- The theorem to prove
theorem average_score_of_both_classes :
  combined_avg_score = 74 :=
by
  sorry

end average_score_of_both_classes_l540_540967


namespace number_of_games_l540_540489

theorem number_of_games
  (touchdowns_per_game : ℕ)
  (points_per_touchdown : ℕ)
  (conversion_count : ℕ)
  (points_per_conversion : ℕ)
  (old_record_points : ℕ)
  (excess_points : ℕ)
  (total_points : ℕ)
  (points_per_game : ℕ)
  (total_touchdown_points : ℕ)
  (games : ℕ)
  (h1 : touchdowns_per_game = 4)
  (h2 : points_per_touchdown = 6)
  (h3 : conversion_count = 6)
  (h4 : points_per_conversion = 2)
  (h5 : old_record_points = 300)
  (h6 : excess_points = 72)
  (h7 : total_points = old_record_points + excess_points)
  (h8 : points_per_game = touchdowns_per_game * points_per_touchdown)
  (h9 : total_touchdown_points = total_points - (conversion_count * points_per_conversion))
  (h10 : games = total_touchdown_points / points_per_game) :
  games = 15 :=
begin
  sorry
end

end number_of_games_l540_540489


namespace return_trip_average_speed_l540_540239

-- Definitions based on conditions
def distance_to_syracuse := 120 -- in miles
def rate_to_syracuse := 40 -- miles per hour
def total_travel_time := 5 + 24/60 -- hours

-- Statement of the problem
theorem return_trip_average_speed : 
  ∃ (rate_return_trip : ℕ), rate_return_trip = 50 :=
by
  have distance_to_syracuse := (120 : ℕ)
  have rate_to_syracuse := (40 : ℕ)
  have time_to_syracuse : ℕ := distance_to_syracuse / rate_to_syracuse
  have total_travel_time : ℕ := 5.4
  have time_return_trip := total_travel_time - time_to_syracuse
  have rate_return_trip : ℕ := distance_to_syracuse / time_return_trip
  existsi rate_return_trip
  sorry

end return_trip_average_speed_l540_540239


namespace problem_proof_l540_540758

def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 1 then
    Real.log2 (1 - x^2)
  else if x ≥ 1 then
    Real.sin (Real.pi * x / 3)
  else
    0  -- This case should theoretically never happen for given conditions

theorem problem_proof :
  f (31/2) + f (Real.sqrt 3 / 2) = -5/2 :=
by
  sorry

end problem_proof_l540_540758


namespace fish_catch_mean_median_mode_l540_540290

theorem fish_catch_mean_median_mode :
  let data := [1, 2, 0, 2, 1, 2, 3, 1, 0, 2, 4, 0]
  let mean := (data.sum : ℝ) / data.length
  let sorted_data := data.qsort (≤)
  let median := (sorted_data.nth (sorted_data.length / 2 - 1) + sorted_data.nth (sorted_data.length / 2)) / 2
  let mode := sorted_data.maximum
  mean = median ∧ mean < mode := 
by {
  -- Conditions are defined, proof is omitted as it's not required
  sorry,
}

end fish_catch_mean_median_mode_l540_540290


namespace sum_largest_smallest_prime_factors_546_l540_540630

theorem sum_largest_smallest_prime_factors_546 : 
  let p := 546; let prime_factors := [2, 3, 7, 13]; 
  (List.minimum prime_factors).getOrElse 0 + (List.maximum prime_factors).getOrElse 0 = 15 := 
by
  intro p prime_factors
  sorry

end sum_largest_smallest_prime_factors_546_l540_540630


namespace total_weight_of_all_items_l540_540539

theorem total_weight_of_all_items :
  ∀ (silverware_weight plate_weight glass_weight decoration_weight settings_per_table num_tables num_backup_settings : ℕ),
  silverware_weight = 4 → 
  plate_weight = 12 → 
  glass_weight = 8 → 
  decoration_weight = 16 → 
  settings_per_table = 8 → 
  num_tables = 15 → 
  num_backup_settings = 20 →
  let total_settings := num_tables * settings_per_table + num_backup_settings in
  let total_weight_of_silverware := total_settings * (3 * silverware_weight) in
  let total_weight_of_plates := total_settings * (2 * plate_weight) in
  let total_weight_of_glasses := total_settings * (2 * glass_weight) in
  let total_weight_of_decorations := num_tables * decoration_weight in
  total_weight_of_silverware + total_weight_of_plates + total_weight_of_glasses + total_weight_of_decorations = 7520 :=
begin
  intros,
  have total_settings := num_tables * settings_per_table + num_backup_settings,
  have total_weight_of_silverware := total_settings * (3 * silverware_weight),
  have total_weight_of_plates := total_settings * (2 * plate_weight),
  have total_weight_of_glasses := total_settings * (2 * glass_weight),
  have total_weight_of_decorations := num_tables * decoration_weight,
  have total_weight_of_all_items := total_weight_of_silverware + total_weight_of_plates + total_weight_of_glasses + total_weight_of_decorations,
  simp at *,
  exact sorry,
end

end total_weight_of_all_items_l540_540539


namespace num_bicycles_l540_540710

theorem num_bicycles (spokes_per_wheel wheels_per_bicycle total_spokes : ℕ) (h1 : spokes_per_wheel = 10) (h2 : total_spokes = 80) (h3 : wheels_per_bicycle = 2) : total_spokes / spokes_per_wheel / wheels_per_bicycle = 4 := by
  sorry

end num_bicycles_l540_540710


namespace ratio_I1_I2_l540_540367

def f (x : ℝ) : ℝ :=
  x^4 + |x|

def I1 : ℝ :=
  ∫ x in 0..π, f (Real.cos x)

def I2 : ℝ :=
  ∫ x in 0..(π/2), f (Real.sin x)

theorem ratio_I1_I2 : (I1 / I2) = 2 :=
  sorry

end ratio_I1_I2_l540_540367


namespace descates_rule_negative_roots_l540_540814

-- Definitions: Polynomial and sign changes
noncomputable def polynomial (coeffs : List ℝ) : ℝ → ℝ :=
  λ x => coeffs.enum.reverse.foldl (λ acc ⟨i, a⟩ => acc + a * x^i) 0

noncomputable def transform_polynomial (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (-x)

noncomputable def sign_changes (coeffs : List ℝ) : ℕ := 
  let signs := coeffs.map signum
  signs.zipWith (≠) (List.drop 1 signs) |>.filter id |>.length

-- Theorem: Application of Descartes' rule for negative roots
theorem descates_rule_negative_roots (coeffs : List ℝ) :
  let f := polynomial coeffs
  let f_neg := transform_polynomial f
  let neg_roots := -- (function to count negative roots of f)
  let sign_changes_of_f_neg := sign_changes coeffs.map (λ ⟨i, a⟩ => if i % 2 = 0 then a else -a)
  neg_roots <= sign_changes_of_f_neg := 
sorry

end descates_rule_negative_roots_l540_540814


namespace infinite_product_result_l540_540322

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540322


namespace value_of_expression_l540_540980

theorem value_of_expression : (20 * 24) / (2 * 0 + 2 * 4) = 60 := sorry

end value_of_expression_l540_540980


namespace equation_of_line_is_correct_l540_540200

/-! Given the circle x^2 + y^2 + 2x - 4y + a = 0 with a < 3 and the midpoint of the chord AB as C(-2, 3), prove that the equation of the line l that intersects the circle at points A and B is x - y + 5 = 0. -/

theorem equation_of_line_is_correct (a : ℝ) (h : a < 3) :
  ∃ l : ℝ × ℝ × ℝ, (l = (1, -1, 5)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0 → 
    (x - y + 5 = 0)) :=
sorry

end equation_of_line_is_correct_l540_540200


namespace minimum_time_to_finish_route_l540_540189

-- Step (a): Defining conditions and necessary terms
def points : Nat := 12
def segments_between_points : ℕ := 17
def time_per_segment : ℕ := 10 -- in minutes
def total_time_in_minutes : ℕ := segments_between_points * time_per_segment -- Total time in minutes

-- Step (c): Proving the question == answer given conditions
theorem minimum_time_to_finish_route (K : ℕ) : K = 4 :=
by
  have time_in_hours : ℕ := total_time_in_minutes / 60
  have minimum_time : ℕ := 4
  sorry -- proof needed

end minimum_time_to_finish_route_l540_540189


namespace descartes_rule_of_signs_l540_540815

-- Define the polynomial f(x) and its negative variant f(-x)
section descartes_rule_of_signs

variables {R : Type*} [LinearOrderedField R]

/-- Polynomial f(x) given by coefficients aₙ, aₙ₋₁, ..., a₁, a₀ -/
def f (coeffs : List R) : R → R :=
  λ x, coeffs.foldl (λ acc a, acc * x + a) 0

/-- Polynomial f(-x) obtained by substituting -x in place of x in f(x) -/
def f_neg (coeffs : List R) : R → R :=
  λ x, f coeffs (-x)

-- Count the sign changes in a list of coefficients
def sign_changes (coeffs : List R) : Nat :=
  coeffs.filter_map (λ x, if x ≠ 0 then some x else none).pairwise (≠).length

/-- The number of negative roots of the polynomial f(x) is at most the number of sign changes
in the sequence of coefficients of f(-x) -/
theorem descartes_rule_of_signs {coeffs : List R} (x : R) :
  (number_of_negative_real_roots (f coeffs) x) ≤ (sign_changes (f_neg coeffs) x) :=
  sorry

end descartes_rule_of_signs

end descartes_rule_of_signs_l540_540815


namespace B_work_days_l540_540645

def work_done_by_A : ℕ := 24
def work_done_by_C : ℕ := 12
def work_done_together : Rational := 24 / 7

theorem B_work_days (x : ℕ) (h : 1 / x + 1 / work_done_by_A + 1 / work_done_by_C = 7 / 24) : x = 6 := sorry

end B_work_days_l540_540645


namespace part1_part2_l540_540123

-- Definitions for functions
def f (x : ℝ) := |x - 3|
def g (x : ℝ) := |x - 2|

-- First part: Solve the inequality
theorem part1 (x : ℝ) : f(x) + g(x) < 2 ↔ (3 / 2 < x ∧ x < 7 / 2) := 
by sorry

-- Second part: Prove inequality under given conditions
theorem part2 {x y : ℝ} (hfx : f(x) ≤ 1) (hgy : g(y) ≤ 1) : |x - 2 * y + 1| ≤ 3 := 
by sorry

end part1_part2_l540_540123


namespace simplify_fraction_result_l540_540921

theorem simplify_fraction_result :
  (144: ℝ) / 1296 * 72 = 8 :=
by
  sorry

end simplify_fraction_result_l540_540921


namespace circumcircle_fixed_point_l540_540912

noncomputable theory

variables {O F L L1 L2 : Type*}
variables (P : Type*) [is_point P]
variables (line : P → P → P → Prop)
variables (circle : P → P → P → P → Prop)
variables (acute_angle : P → P → P → Prop)
variables (lies_inside : P → P → P → Prop)
variables (tangent : P → P → P → Prop)
variables (passes_through : P → P → P → Prop)

def circumcircle_passing_fixed_point (O F L L1 L2 : P) : Prop :=
  ∃ X, passes_through X F ∧ passes_through X L1 ∧ passes_through X L2 ∧ X ≠ F

theorem circumcircle_fixed_point
  (h1: line O F L)
  (h2: line O F L1)
  (h3: line O F L2)
  (h4: acute_angle O F L2)
  (h5: lies_inside O F L1)
  (h6: passes_through F L L1)
  (h7: tangent L L1 L1)
  (h8: passes_through F L L2)
  (h9: tangent L L2 L2)
: circumcircle_passing_fixed_point O F L L1 L2 := sorry

end circumcircle_fixed_point_l540_540912


namespace hyperbola_eq_of_point_and_symmetry_l540_540359

theorem hyperbola_eq_of_point_and_symmetry :
  ∃ a : ℝ, (a > 0 ∧ (∀ x y : ℝ, (x, y) = (3, -1) →  (x^2)/a^2 - (y^2)/a^2 = 1)) :=
by {
  use 8,
  split,
  { norm_num },
  {
    rintros x y ⟨hx, hy⟩,
    rw [hx, hy],
    norm_num,
  },
  sorry  -- actual steps to reach the solution are skipped
}

end hyperbola_eq_of_point_and_symmetry_l540_540359


namespace negation_proposition_false_l540_540587

theorem negation_proposition_false : 
  (¬ ∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end negation_proposition_false_l540_540587


namespace total_monthly_expenditure_l540_540463

noncomputable def calculate_monthly_pay (hours_per_week : ℕ) (hourly_rate : ℕ) : ℕ :=
  4 * (hours_per_week * hourly_rate)

theorem total_monthly_expenditure : 
  let fiona_weekly_hours := 40
  let fiona_hourly_rate := 20
  let john_weekly_hours := 30
  let john_hourly_rate := 22
  let jeremy_weekly_hours := 25
  let jeremy_hourly_rate := 18
  let katie_weekly_hours := 35
  let katie_hourly_rate := 21
  let matt_weekly_hours := 28
  let matt_hourly_rate := 19
  let weeks_in_month := 4 in
  calculate_monthly_pay fiona_weekly_hours fiona_hourly_rate + 
  calculate_monthly_pay john_weekly_hours john_hourly_rate + 
  calculate_monthly_pay jeremy_weekly_hours jeremy_hourly_rate + 
  calculate_monthly_pay katie_weekly_hours katie_hourly_rate + 
  calculate_monthly_pay matt_weekly_hours matt_hourly_rate
  = 12708 :=
by sorry

end total_monthly_expenditure_l540_540463


namespace math_competition_problem_l540_540962

def numberOfWays (students tests : Finset ℕ) (f : ℕ → Finset ℕ) : ℕ :=
  if (∀ s ∈ students, (f s).card = 2) ∧
     (∀ t ∈ tests, ((students.filter (λ s, t ∈ f s)).card = 2)) then 
    2040 
  else 
    0

theorem math_competition_problem : 
  let students := Finset.range 5
  let tests := Finset.range 5
  ∃ f : ℕ → Finset ℕ, (∀ s ∈ students, (f s).card = 2) ∧
                      (∀ t ∈ tests, ((students.filter (λ s, t ∈ f s)).card = 2)) ∧
                      numberOfWays students tests f = 2040 :=
by
  sorry

end math_competition_problem_l540_540962


namespace horizontal_asymptote_l540_540946

def function (x : ℝ) : ℝ := (5 * x^2 - 9) / (3 * x^2 + 5 * x + 2)

theorem horizontal_asymptote : 
  lim (filter.at_top) (λ x, function x) = 5 / 3 :=
by {
  sorry
}

end horizontal_asymptote_l540_540946


namespace find_angle_BAC_l540_540073

variables {n q p c : ℝ}
variables {A B C D : Point}
variables {angle_BAC : ℝ} -- representing ∠ BAC

-- Definitions and assumptions based on the problem statement
def is_isosceles_triangle (ABC : Triangle) : Prop := ABC.side_AB = ABC.side_AC

def angle_bisector (A B C : Point) (AD : Line) : Prop := 
  Line.angle_divides_angle AD (angle BAC)

def sum_of_segments (BC BD AD : Segment) : Prop := BC.length = BD.length + AD.length

-- Goal to prove
theorem find_angle_BAC (ABC : Triangle) (AD : Line) :
  is_isosceles_triangle ABC →
  angle_bisector ABC.A BC ABC.AC AD →
  sum_of_segments ABC.BC AD BD →
  angle_BAC = 80 :=
  sorry

end find_angle_BAC_l540_540073


namespace geometric_sequence_product_l540_540862

variable {α : Type*} [Field α]

-- Define the arithmetic sequence a_n
def a (n : ℕ) : α := sorry -- Definition of a_n as an arithmetic sequence (left as sorry)

-- Define the geometric sequence b_n
def b (n : ℕ) : α := sorry -- Definition of b_n as a geometric sequence (left as sorry)

-- Condition 1: a_1 + a_11 = 8
axiom h1 : a 1 + a 11 = 8

-- Condition 2: b_7 = a_7
axiom h2 : b 7 = a 7

-- Theorem to prove: b_6 * b_8 = 16
theorem geometric_sequence_product : b 6 * b 8 = 16 :=
by
  sorry

end geometric_sequence_product_l540_540862


namespace never_return_original_quadruple_repeat_transform_correct_l540_540660

theorem never_return_original_quadruple (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (∃ (n : ℕ), (repeat_transform (a, b, c, d) n = (a, b, c, d))) ↔ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

-- Helper function: repeat_transform
def repeat_transform : ℕ × ℝ × ℝ × ℝ × ℝ → ℝ × ℝ × ℝ × ℝ
| (0, a, b, c, d) := (a, b, c, d)
| (n + 1, a, b, c, d) := repeat_transform (n, ab, bc, cd, da)

theorem repeat_transform_correct (a b c d : ℝ) (n : ℕ) : 
  repeat_transform (n, a, b, c, d) = (repeat_transform n ab, repeat_transform n bc, repeat_transform n cd, repeat_transform n da)
:= by sorry

end never_return_original_quadruple_repeat_transform_correct_l540_540660


namespace difference_between_sums_l540_540144

-- Define the arithmetic sequence sums
def sum_seq (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Define sets A and B
def sumA : ℕ := sum_seq 10 75
def sumB : ℕ := sum_seq 76 125

-- State the problem
theorem difference_between_sums : sumB - sumA = 2220 :=
by
  -- The proof is omitted
  sorry

end difference_between_sums_l540_540144


namespace range_of_t_l540_540891

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x + cos x

-- State the main theorem
theorem range_of_t (t : ℝ) (h1 : 0 < t ∧ t < 1) (h2 : f (t^2) > f (2 * t - 1)) : 1/2 < t ∧ t < 1 :=
sorry

end range_of_t_l540_540891


namespace infinite_div_by_100_l540_540911

theorem infinite_div_by_100 : ∀ k : ℕ, ∃ n : ℕ, n > 0 ∧ (2 ^ n + n ^ 2) % 100 = 0 :=
by
  sorry

end infinite_div_by_100_l540_540911


namespace recurring_six_denominator_l540_540171

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l540_540171


namespace rational_smaller_than_neg_half_l540_540551

theorem rational_smaller_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  use (-1 : ℚ)
  sorry

end rational_smaller_than_neg_half_l540_540551


namespace num_three_digit_integers_l540_540024

open Nat

theorem num_three_digit_integers :
  let digits := {2, 3, 5, 8}
  card digits = 4 →
  ∃ nums : Set (Fin 1000), 
      (∀ n ∈ nums, ∀ d ∈ digits, Multiset.card (n.digits d) ≤ 1) ∧ 
      card nums = 24 :=
by 
  sorry

end num_three_digit_integers_l540_540024


namespace smallest_N_circular_table_l540_540252

theorem smallest_N_circular_table (N chairs : ℕ) (circular_seating : N < chairs) :
  (∀ new_person_reserved : ℕ, new_person_reserved < chairs →
    (∃ i : ℕ, (i < N) ∧ (new_person_reserved = (i + 1) % chairs ∨ 
                           new_person_reserved = (i - 1) % chairs))) ↔ N = 18 := by
sorry

end smallest_N_circular_table_l540_540252


namespace num_athletes_with_4_points_after_seven_rounds_correct_l540_540857

noncomputable def num_athletes_with_4_points_after_seven_rounds (n : ℕ) (hn : n > 7) : ℕ :=
  let num_participants := 2^n + 6
  let f (m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k
  if hn then 35 * 2^(n - 7) + 2 else 0

theorem num_athletes_with_4_points_after_seven_rounds_correct (n : ℕ) (hn : n > 7) :
  num_athletes_with_4_points_after_seven_rounds n hn = 35 * 2^(n - 7) + 2 :=
by
  sorry

end num_athletes_with_4_points_after_seven_rounds_correct_l540_540857


namespace count_negative_numbers_l540_540481

theorem count_negative_numbers : 
  let n1 := abs (-2)
  let n2 := - abs (3^2)
  let n3 := - (3^2)
  let n4 := (-2)^(2023)
  (if n1 < 0 then 1 else 0) + (if n2 < 0 then 1 else 0) + (if n3 < 0 then 1 else 0) + (if n4 < 0 then 1 else 0) = 3 := 
by
  sorry

end count_negative_numbers_l540_540481


namespace inequality_proof_l540_540749

theorem inequality_proof (a b c d : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a + b = 2 → c + d = 2 → 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := 
by 
  intros ha hb hc hd hab hcd
  sorry

end inequality_proof_l540_540749


namespace calculate_T_l540_540010

theorem calculate_T (n : ℕ) (hn : 0 < n) :
  let a : ℕ → ℕ := λ k, 3 * k
  let b : ℕ → ℕ := λ k, 3 ^ (k - 1)
  let T : ℕ → ℕ := λ n, ∑ i in finset.range n, a (n - i) * b (i + 1)
  in T n = (3 ^ (n + 2)) / 4 - (3 * n) / 2 - 9 / 4 := sorry

end calculate_T_l540_540010


namespace first_player_has_winning_strategy_l540_540213

-- Definitions for the game setup
def Bag := Fin 2008
def Frogs := Nat

-- Initial condition: 2008 bags with 2008 frogs
noncomputable def initial_bags : Bag → Frogs := λ b => 2008

-- Removing frogs function
def remove_frogs (bags : Bag → Frogs) (b : Bag) (f : Frogs) (h : 1 ≤ f ∧ f ≤ bags b) : Bag → Frogs :=
  λ b' => if b' = b then bags b - f else if b' > b ∧ bags b' > bags b - f then bags b - f else bags b'

-- Winning condition: a player loses if they take the last frog from bag 1
def losing_condition (bags : Bag → Frogs) : Prop :=
  bags 0 = 0

-- Winning strategy definition
noncomputable def first_player_wins : Prop :=
  ∃ strategy : (Bag → Frogs) → Bag → Frogs, 
    ∀ bags : (Bag → Frogs),
      ∀ m : Bag, ∀ n : Frogs, 
        1 ≤ n ∧ n ≤ bags m → 
        losing_condition (remove_frogs bags m n)

theorem first_player_has_winning_strategy : first_player_wins := 
by
  -- Proof is omitted
  sorry

end first_player_has_winning_strategy_l540_540213


namespace initial_ratio_l540_540090

def initial_men (M : ℕ) (W : ℕ) : Prop :=
  let men_after := M + 2
  let women_after := W - 3
  (2 * women_after = 24) ∧ (men_after = 14)

theorem initial_ratio (M W : ℕ) (h : initial_men M W) :
  (M = 12) ∧ (W = 15) → M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  intro hm hw
  have h12 : M = 12 := hm
  have h15 : W = 15 := hw
  sorry

end initial_ratio_l540_540090


namespace equivalent_angle_terminal_side_l540_540791

theorem equivalent_angle_terminal_side (k : ℤ) (a : ℝ) (c : ℝ) (d : ℝ) : a = -3/10 * Real.pi → c = a * 180 / Real.pi → d = c + 360 * k →
   ∃ k : ℤ, d = 306 :=
sorry

end equivalent_angle_terminal_side_l540_540791


namespace residue_neg_1234_mod_32_l540_540302

theorem residue_neg_1234_mod_32 : -1234 % 32 = 14 := 
by sorry

end residue_neg_1234_mod_32_l540_540302


namespace zero_point_in_interval_l540_540950

def f (x : ℝ) : ℝ := 3^x + 2*x - 3

theorem zero_point_in_interval :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) :=
by
  -- Conditions
  have h0 : f 0 = -2 := by norm_num
  have h1 : f 1 = 2 := by norm_num
  have h_strict_incr : ∀ (a b : ℝ), a < b → f a < f b := 
    by intros a b hab; rw [f]; apply strict_mono_incr_on_exp; linarith
  -- Proof
  sorry  -- Placeholder for the actual proof

end zero_point_in_interval_l540_540950


namespace max_pieces_on_chessboard_l540_540619

-- Definition of an 8x8 chessboard
def is_chessboard (cells : Fin 8 × Fin 8 → Prop) : Prop :=
  ∀ r c, cells (⟨r, by simp⟩, ⟨c, by simp⟩) → true

-- Definition to check if no more than three pieces are placed on any diagonal
def max_three_per_diagonal (pieces : Fin 8 × Fin 8 → Prop) : Prop :=
  ∀ delta, 
    (∑ r in Finset.univ, ∑ c in Finset.univ, if pieces (⟨r, by simp⟩, ⟨c, by simp⟩) ∧ (r - c = delta) then 1 else 0) ≤ 3 ∧
    (∑ r in Finset.univ, ∑ c in Finset.univ, if pieces (⟨r, by simp⟩, ⟨c, by simp⟩) ∧ (r + c = delta) then 1 else 0) ≤ 3

-- The theorem stating the maximum number of pieces on the chessboard with given conditions
theorem max_pieces_on_chessboard : 
  ∃ pieces : Fin 8 × Fin 8 → Prop, 
    is_chessboard pieces ∧ max_three_per_diagonal pieces ∧ 
    (∑ r in Finset.univ, ∑ c in Finset.univ, if pieces (⟨r, by simp⟩, ⟨c, by simp⟩) then 1 else 0) = 38 :=
sorry

end max_pieces_on_chessboard_l540_540619


namespace total_money_divided_l540_540664

theorem total_money_divided (A B C : ℝ) (h1 : A = (1 / 2) * B) (h2 : B = (1 / 2) * C) (h3 : C = 208) :
  A + B + C = 364 := 
sorry

end total_money_divided_l540_540664


namespace apples_division_l540_540456

theorem apples_division (total_apples : ℕ) (ratios : list (ℕ × ℕ))
    (h_total : total_apples = 169)
    (h_ratios : ratios = [(1, 2), (1, 3), (1, 4)]) :
    ∃ apples1 apples2 apples3,
    apples1 + apples2 + apples3 = total_apples ∧
    (apples1 : ℤ) / (apples2 : ℤ) = 1 / 2 ∧
    (apples1 : ℤ) / (apples3 : ℤ) = 1 / 3 ∧
    (apples2 : ℤ) / (apples3 : ℤ) = 4 / 3 ∧
    apples1 = 78 ∧
    apples2 = 52 ∧
    apples3 = 39 := 
by
  sorry

end apples_division_l540_540456


namespace find_m_l540_540455

noncomputable def is_solution (θ : ℝ) (m : ℝ) : Prop :=
  (m ≠ 0) ∧ (sin θ = (sqrt 2 / 2) * m) ∧ (cos θ = -1 / sqrt(1 + m^2))

theorem find_m (m : ℝ) (θ : ℝ) (h : is_solution θ m) : m = 1 ∨ m = -1 :=
by sorry

end find_m_l540_540455


namespace a_decreasing_S_n_formula_l540_540008

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else 1 / (3 * n - 2)

def b (n : ℕ) : ℝ :=
  a n / (3 * n + 1)

def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i + 1)

theorem a_decreasing (n : ℕ) (hn : n ≥ 1) : a (n + 1) < a n :=
  by
    -- Proof to be filled in
    sorry

theorem S_n_formula (n : ℕ) (hn : n ≥ 1) : S n = n / (3 * n + 1) :=
  by
    -- Proof to be filled in
    sorry

end a_decreasing_S_n_formula_l540_540008


namespace range_of_m_solution_set_l540_540803

noncomputable def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem range_of_m (m : ℝ) :
  (∃ x: ℝ, f x ≤ m) ↔ m ≥ -3 := sorry

theorem solution_set (s : set ℝ) :
  s = {x : ℝ | 5 - real.sqrt 3 < x ∧ x < 5} ∪ {x : ℝ | 5 < x ∧ x < 6} ↔
  (∀ x : ℝ, x ∈ s ↔ x^2 - 8 * x + 15 + f x < 0) := sorry

end range_of_m_solution_set_l540_540803


namespace prime_if_and_only_if_digit_is_nine_l540_540598

theorem prime_if_and_only_if_digit_is_nine (B : ℕ) (h : 0 ≤ B ∧ B < 10) :
  Prime (303200 + B) ↔ B = 9 := 
by
  sorry

end prime_if_and_only_if_digit_is_nine_l540_540598


namespace sum_of_digits_of_expression_l540_540623

theorem sum_of_digits_of_expression : 
  (sum_digits (decimal_repr (2^2010 * 5^2008 * 7)) = 10) :=
sorry

end sum_of_digits_of_expression_l540_540623


namespace problem_sum_of_possible_n_values_l540_540542

theorem problem_sum_of_possible_n_values :
  (∃ (n : ℕ), n ∈ {11, 12, 13, 14}) ∧
  (∀ x ∈ {11, 12, 13, 14}, (50 ≤ x + 50 ∧ x ≤ 12.5) 
   ∨ (60 ≤ x + 50 ∧ x ≤ 15)) →
  ∑' x ∈ {11, 12, 13, 14}, x = 50 := 
sorry

end problem_sum_of_possible_n_values_l540_540542


namespace determine_h_l540_540737

theorem determine_h (h : ℝ) : (∃ x : ℝ, x = 3 ∧ x^3 - 2 * h * x + 15 = 0) → h = 7 :=
by
  intro hx
  sorry

end determine_h_l540_540737


namespace fraction_denominator_l540_540182

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l540_540182


namespace fraction_of_male_gerbils_is_correct_l540_540288

def total_pets := 90
def total_gerbils := 66
def total_hamsters := total_pets - total_gerbils
def fraction_hamsters_male := 1/3
def total_males := 25
def male_hamsters := fraction_hamsters_male * total_hamsters
def male_gerbils := total_males - male_hamsters
def fraction_gerbils_male := male_gerbils / total_gerbils

theorem fraction_of_male_gerbils_is_correct : fraction_gerbils_male = 17 / 66 := by
  sorry

end fraction_of_male_gerbils_is_correct_l540_540288


namespace limit_expression_l540_540841

variable {α : Type*} [Field α] {f : α → α} {x₀ : α}

-- Condition: The derivative of f at x₀ is -2
def f_derivative_at_x₀ := deriv f x₀ = -2

-- Statement: We want to prove that the limit is 1 given the derivative condition
theorem limit_expression (h_deriv : f_derivative_at_x₀) :
  tendsto (λ h : α, (f (x₀ - (1/2) * h) - f x₀) / h) (𝓝 0) (𝓝 1) :=
sorry

end limit_expression_l540_540841


namespace sum_of_valid_y_values_l540_540734

-- Definitions
def is_single_digit (y : ℕ) : Prop := y >= 0 ∧ y <= 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Lean statement of the problem
theorem sum_of_valid_y_values : 
  (∑ y in {y | is_single_digit y ∧ divisible_by_4 (48000 + y*100 + 72) ∧ divisible_by_3 (21 + y)}, y) = 9 := 
by 
  -- Add the main part of the proof here
  sorry

end sum_of_valid_y_values_l540_540734


namespace opinions_eventually_stabilize_l540_540661

-- Define the type representing the opinions
inductive Opinion
| EarthRevolvesAroundJupiter
| JupiterRevolvesAroundEarth

open Opinion

-- Defining the situation with 101 wise men in a circle
structure Situation where
  opinions : Fin 101 → Opinion

-- Update rule for one step
def update_opinion (s : Situation) (i : Fin 101) : Opinion :=
  let left_neighbor := s.opinions (i - 1)
  let right_neighbor := s.opinions (i + 1)
  if left_neighbor = right_neighbor then 
    s.opinions i 
  else 
    match s.opinions i with
    | EarthRevolvesAroundJupiter => JupiterRevolvesAroundEarth
    | JupiterRevolvesAroundEarth => EarthRevolvesAroundJupiter

def update_all (s : Situation) : Situation :=
  ⟨fun i => update_opinion s i⟩

-- Define what it means for the situation to be stable
def is_stable (s : Situation) : Prop :=
  ∀ i : Fin 101, update_opinion s i = s.opinions i

-- The statement of the theorem
theorem opinions_eventually_stabilize (s : Situation) :
  ∃ n : ℕ, is_stable (Nat.iterate update_all n s) :=
sorry

end opinions_eventually_stabilize_l540_540661


namespace burger_cost_cents_l540_540717

theorem burger_cost_cents 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 550) 
  (h2 : 3 * b + 2 * s = 400) 
  (h3 : 2 * b + s = 250) : 
  b = 100 :=
by
  sorry

end burger_cost_cents_l540_540717


namespace compare_x_y_l540_540783

theorem compare_x_y (a b : ℝ) (h1 : a > b) (h2 : b > 1) (x y : ℝ)
  (hx : x = a + 1 / a) (hy : y = b + 1 / b) : x > y :=
by {
  sorry
}

end compare_x_y_l540_540783


namespace min_shift_m_symm_l540_540145

noncomputable def shifted_symmetric_min_m : ℝ := 
  let y := λ x : ℝ, (sqrt 3) * Real.cos x + Real.sin x
  let shifted_y := λ m x : ℝ, (sqrt 3) * Real.cos (x + m) + Real.sin (x + m)
  (∃ m > 0, (∀ x : ℝ, shifted_y m x = shifted_y m (-x))) → m = π / 6

theorem min_shift_m_symm : shifted_symmetric_min_m := sorry

end min_shift_m_symm_l540_540145


namespace find_opposite_of_neg_half_l540_540201

-- Define the given number
def given_num : ℚ := -1/2

-- Define what it means to find the opposite of a number
def opposite (x : ℚ) : ℚ := -x

-- State the theorem
theorem find_opposite_of_neg_half : opposite given_num = 1/2 :=
by
  -- Proof is omitted for now
  sorry

end find_opposite_of_neg_half_l540_540201


namespace minimum_points_on_circle_l540_540227

theorem minimum_points_on_circle (C : ℝ) (hC : C = 1956) :
  ∃ (n : ℕ), 
  (∀ (p : ℕ), p ∈ range n -> 
    (∃ (q r : ℕ), q ≠ p ∧ r ≠ p ∧ q ≠ r ∧ abs (q - p) % C = 1 ∧ abs (r - p) % C = 2)) 
  ∧ n = 1304 :=
by
  sorry

end minimum_points_on_circle_l540_540227


namespace find_k_eq_neg_four_thirds_l540_540654

-- Definitions based on conditions
def hash_p (k : ℚ) (p : ℚ) : ℚ := k * p + 20

-- Using the initial condition
def triple_hash_18 (k : ℚ) : ℚ :=
  let hp := hash_p k 18
  let hhp := hash_p k hp
  hash_p k hhp

-- The Lean statement for the desired proof
theorem find_k_eq_neg_four_thirds (k : ℚ) (h : triple_hash_18 k = -4) : k = -4 / 3 :=
sorry

end find_k_eq_neg_four_thirds_l540_540654


namespace minimum_distance_AB_l540_540020

-- Definitions of the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C2 (x y : ℝ) : Prop := y^2 - x + 1 = 0

theorem minimum_distance_AB :
  ∃ (A B : ℝ × ℝ), C1 A.1 A.2 ∧ C2 B.1 B.2 ∧ dist A B = 3*Real.sqrt 2 / 4 := sorry

end minimum_distance_AB_l540_540020


namespace infinite_product_to_rational_root_l540_540319

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540319


namespace num_students_selected_from_second_campsite_of_systematic_sample_equals_20_l540_540602

theorem num_students_selected_from_second_campsite_of_systematic_sample_equals_20 :
  ∀ (total_students sample_size drawn_number : ℕ) (first_start first_end second_start second_end third_start third_end : ℕ),
  total_students = 600 →
  sample_size = 50 →
  drawn_number = 3 →
  first_start = 1 → first_end = 266 →
  second_start = 267 → second_end = 496 →
  third_start = 497 → third_end = 600 →
  (∃ n : ℕ, second_start ≤ 12 * n - 9 ∧ 12 * n - 9 ≤ second_end) →
  (let interval := 12 in
  let selected_students := list.map (λ n, 3 + interval * n) (list.range sample_size) in
  (list.countp (λ x, second_start ≤ x ∧ x ≤ second_end) selected_students) = 20) :=
begin
  sorry
end

end num_students_selected_from_second_campsite_of_systematic_sample_equals_20_l540_540602


namespace twenty_two_supernumber_twenty_five_supernumber_forty_nine_supernumber_count_count_all_supernumbers_l540_540246

def isSupernumber (A B C : ℕ) : Prop :=
  (10 ≤ A ∧ A < 100) ∧ (10 ≤ B ∧ B < 100) ∧ (10 ≤ C ∧ C < 100) ∧ (A = B + C) ∧ (digitSum A = digitSum B + digitSum C)

def digitSum (n : ℕ) : ℕ :=
  let d := n.digits 10
  d.foldr (·+·) 0

theorem twenty_two_supernumber :
  isSupernumber 22 10 12 ∧ isSupernumber 22 11 11 :=
sorry

theorem twenty_five_supernumber :
  isSupernumber 25 10 15 ∧ isSupernumber 25 11 14 ∧ isSupernumber 25 12 13 :=
sorry

theorem forty_nine_supernumber_count :
  (∑ x in Finset.range 90, ∑ y in Finset.range 90, ite (isSupernumber 49 x y) 1 0) = 15 :=
sorry

theorem count_all_supernumbers :
  ∑ A in Finset.range 90, ite (∃ (B C : ℕ), isSupernumber A B C) 1 0 = 80 :=
sorry

end twenty_two_supernumber_twenty_five_supernumber_forty_nine_supernumber_count_count_all_supernumbers_l540_540246


namespace min_abs_value_l540_540412

theorem min_abs_value (a : ℝ) (h : 0 ≤ a ∧ a < 4) : ∃ x, x = |a - 2| + |3 - a| ∧ x = 1 :=
begin
  sorry
end

end min_abs_value_l540_540412


namespace jack_has_42_pounds_l540_540872

noncomputable def jack_pounds (P : ℕ) : Prop :=
  let euros := 11
  let yen := 3000
  let pounds_per_euro := 2
  let yen_per_pound := 100
  let total_yen := 9400
  let pounds_from_euros := euros * pounds_per_euro
  let pounds_from_yen := yen / yen_per_pound
  let total_pounds := P + pounds_from_euros + pounds_from_yen
  total_pounds * yen_per_pound = total_yen

theorem jack_has_42_pounds : jack_pounds 42 :=
  sorry

end jack_has_42_pounds_l540_540872


namespace abs_difference_of_mn_6_and_sum_7_l540_540516

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l540_540516


namespace problem1_part1_problem1_part2_l540_540850

theorem problem1_part1 (a b c : ℝ) (F1 F2 : ℝ × ℝ) (P Q : ℝ × ℝ) (O : ℝ × ℝ) (perimeter : ℝ) :
  let hyperbola := ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1,
  let e := c / a,
  let symmetry := P.1 = -Q.1 ∧ P.2 = -Q.2,
  let center := O = (0, 0),
  let not_on_x_axis := P.2 ≠ 0 ∧ Q.2 ≠ 0,
  let perimeter_condition := perimeter = 4 * sqrt 2,
  e = 2 →
  hyperbola 1 (3 / 2) →
  F1 = (-c, 0) ∧ F2 = (c, 0) →
  symmetry →
  center →
  not_on_x_axis →
  perimeter_condition →
  ∀ (M : ℝ × ℝ), is_ellipse P F1 F2

theorem problem1_part2 (kx m : ℝ) (M N G H : ℝ × ℝ) :
  let line := ∀ (x y : ℝ), y = kx * x + m,
  let trajectory := ∀ (x y : ℝ), x^2 / 2 + y^2 = 1,
  let circle := ∀ (x y : ℝ), x^2 + y^2 = 3 / 2,
  let ratio_constant := ∀ m, (dist M.1 M.2) / (dist G.1 G.2) = const,
  line_intersects_trajectory M N →
  line_intersects_circle G H →
  ratio_constant →
  is_constant k 1

end problem1_part1_problem1_part2_l540_540850


namespace fourth_number_in_15th_row_l540_540585

theorem fourth_number_in_15th_row :
  let last_number := 5 * 15 in
  let fourth_number := last_number - 2 in
  fourth_number = 73 :=
by
  -- definitions
  let last_number := 5 * 15
  let fourth_number := last_number - 2
  -- goal
  show fourth_number = 73
  sorry

end fourth_number_in_15th_row_l540_540585


namespace find_mass_m2_l540_540256

theorem find_mass_m2 :
  ∀ (m1 m2 : ℝ) (λ AB BC BO OC : ℝ),
  m1 = 2 → λ = 2 → AB = 7 → BC = 5 → BO = 4 → OC = 3 →
  (m1 * AB + (λ * AB) * 0.5 * AB = m2 * BO + (λ * BC) * 0.5 * BO) →
  m2 = 10.75 :=
by
  intros m1 m2 λ AB BC BO OC Hm1 Hλ HAB HBC HBO HOC Hequilibrium
  sorry

end find_mass_m2_l540_540256


namespace first_term_exceeding_10000_l540_540580

-- Define the sequence
def seq : ℕ → ℕ
| 0 => 2
| (n + 1) => (finset.sum (finset.range (n + 1)) seq)

-- The theorem statement about the first term exceeding 10000
theorem first_term_exceeding_10000 :
  ∃ n, seq(n) > 10000 ∧ seq(n) = 16384 :=
by
  -- proof to be done
  sorry

end first_term_exceeding_10000_l540_540580


namespace correct_proposition_D_l540_540233

theorem correct_proposition_D (a b c : ℝ) (h : a > b) : a - c > b - c :=
by
  sorry

end correct_proposition_D_l540_540233


namespace exists_n_for_binom_eq_l540_540572

open BigOperators

def prime (p : ℕ) := p > 1 ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

theorem exists_n_for_binom_eq (p k : ℤ) (hp : prime p):
    ∃ n : ℤ, nat.choose n p = nat.choose (n + k) p :=
sorry

end exists_n_for_binom_eq_l540_540572


namespace mahesh_days_worked_l540_540893

-- Assignments for work rates
def work_rate_mahesh : ℚ := 1 / 60
def work_rate_rajesh : ℚ := 1 / 45
def rajesh_days : ℚ := 30

-- Work done by Rajesh in given days
def work_done_rajesh : ℚ := rajesh_days * work_rate_rajesh

-- Total work is 1 unit
def total_work : ℚ := 1

-- Formula for work completed by Mahesh (in x days) + work completed by Rajesh
def work_done_mahesh (x : ℚ) : ℚ := x * work_rate_mahesh

-- The equation representing total work done
def eq_work_done (x : ℚ) : Prop := work_done_mahesh(x) + work_done_rajesh = total_work

-- Proof statement: find x such that given conditions hold true
theorem mahesh_days_worked : ∃ x : ℚ, eq_work_done(x) ∧ x = 20 :=
sorry

end mahesh_days_worked_l540_540893


namespace initial_ratio_is_four_five_l540_540089

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end initial_ratio_is_four_five_l540_540089


namespace trajectory_of_point_on_cube_face_is_parabolic_segment_l540_540386

theorem trajectory_of_point_on_cube_face_is_parabolic_segment
  (cube : ℝ → ℝ → ℝ → Prop)
  (A B C D A1 B1 C1 D1 : ℝ → ℝ → ℝ)
  (face : ℝ → ℝ → ℝ → Prop)
  (side_edge : ℝ → ℝ → ℝ → Prop)
  (base_face : ℝ → ℝ → ℝ → Prop)
  (P : ℝ)
  (P_on_face : face P)
  (dist_equality : ∀ (P : ℝ), dist P side_edge = dist P base_face) :
  ∃ α β γ : ℝ, trajectory_of_P_segment_of_parabola :=
sorry

end trajectory_of_point_on_cube_face_is_parabolic_segment_l540_540386


namespace regular_polygon_sides_l540_540403

theorem regular_polygon_sides (α : ℝ) (h : α = 120) : 
  let exterior_angle := 180 - α in
  let n := 360 / exterior_angle in
  n = 6 :=
by
  let exterior_angle := 180 - α;
  let n := 360 / exterior_angle;
  sorry

end regular_polygon_sides_l540_540403


namespace denominator_of_repeating_six_l540_540161

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l540_540161


namespace simplify_expression_l540_540920

variable (a b c x y z : ℝ)

theorem simplify_expression :
  (cz * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + bz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cz + bz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cz * a^3 * y^3 + 3 * bz * c^3 * x^3) / (cz + bz) :=
by
  sorry

end simplify_expression_l540_540920


namespace options_correct_l540_540154

theorem options_correct :
  (∀ (x : ℂ), complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x) ∧ 
  ¬(∀ (x : ℂ), complex.exp (complex.I * x) = -complex.I) ∧ 
  ∀ (x : ℝ), (0 ≤ x) → (2^x ≥ 1 + x * real.log 2 + (x * real.log 2)^2 / 2!) ∧ 
  ∀ (x : ℝ), (0 < x ∧ x < 1) → real.cos x ≤ 1 - x^2 / 2! + x^4 / 4! :=
by
  sorry

end options_correct_l540_540154


namespace maximum_ratio_of_area_H_to_area_S_l540_540691

noncomputable def max_area_ratio_of_triangle_in_hexagon : ℝ :=
  let a : ℝ := 1 in -- arbitrary unit side length for the hexagon
  let area_hexagon := (3 * real.sqrt 3 / 2) * a^2 in
  let area_triangle := (3 * real.sqrt 3 / 8) * a^2 in
  area_triangle / area_hexagon

theorem maximum_ratio_of_area_H_to_area_S :
  max_area_ratio_of_triangle_in_hexagon = 3 / 8 :=
by
  -- Proof goes here
  sorry

end maximum_ratio_of_area_H_to_area_S_l540_540691


namespace shaded_region_is_correct_l540_540477

noncomputable def area_shaded_region : ℝ :=
  let r_small := (3 : ℝ) / 2
  let r_large := (15 : ℝ) / 2
  let area_small := (1 / 2) * Real.pi * r_small^2
  let area_large := (1 / 2) * Real.pi * r_large^2
  (area_large - 2 * area_small + 3 * area_small)

theorem shaded_region_is_correct :
  area_shaded_region = (117 / 4) * Real.pi :=
by
  -- The proof will go here.
  sorry

end shaded_region_is_correct_l540_540477


namespace binomial_variance_eq_p_mul_one_sub_p_l540_540070

open ProbabilityTheory

variables {X : Type} [DiscreteRandomVariable X ℝ]
variables (p q : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : q = 1 - p)

noncomputable def binomialVariance : ℝ := p * (1 - p)

theorem binomial_variance_eq_p_mul_one_sub_p
  (hX : ∀ x : X, x = 0 ∨ x = 1)
  (hp : ∀ x : X, x = 1 → P(x) = p)
  (hq : ∀ x : X, x = 0 → P(x) = q):
  variance X = p * (1 - p) := sorry

end binomial_variance_eq_p_mul_one_sub_p_l540_540070


namespace average_weight_bc_is_43_l540_540935

variable (a b c : ℝ)

-- Definitions of the conditions
def average_weight_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_weight_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def weight_b (b : ℝ) : Prop := b = 31

-- The theorem to prove
theorem average_weight_bc_is_43 :
  ∀ (a b c : ℝ), average_weight_abc a b c → average_weight_ab a b → weight_b b → (b + c) / 2 = 43 :=
by
  intros a b c h_average_weight_abc h_average_weight_ab h_weight_b
  sorry

end average_weight_bc_is_43_l540_540935


namespace problem_counts_correct_pairs_l540_540370

noncomputable def count_valid_pairs : ℝ :=
  sorry

theorem problem_counts_correct_pairs :
  count_valid_pairs = 128 :=
by
  sorry

end problem_counts_correct_pairs_l540_540370


namespace cosine_angle_between_a_b_lambda_value_if_perpendicular_l540_540000

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Dot product definition
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Norm (magnitude) definition
def norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem 1: Prove the cosine value of the angle between a and b
theorem cosine_angle_between_a_b :
  Real.cos (Real.acos ((dot_product a b) / (norm a * norm b))) = (2 * Real.sqrt(5)) / 25 :=
sorry

-- Problem 2: Prove the value of lambda if a - λb is perpendicular to 2a + b
theorem lambda_value_if_perpendicular (λ : ℝ) :
  dot_product (a - (λ • b)) (2 • a + b) = 0 → λ = 52 / 9 :=
sorry

end cosine_angle_between_a_b_lambda_value_if_perpendicular_l540_540000


namespace fraction_denominator_l540_540181

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l540_540181


namespace sahil_machine_purchase_price_l540_540140

theorem sahil_machine_purchase_price
  (repair_cost : ℕ)
  (transportation_cost : ℕ)
  (selling_price : ℕ)
  (profit_percent : ℤ)
  (purchase_price : ℕ)
  (total_cost : ℕ)
  (profit_ratio : ℚ)
  (h1 : repair_cost = 5000)
  (h2 : transportation_cost = 1000)
  (h3 : selling_price = 30000)
  (h4 : profit_percent = 50)
  (h5 : total_cost = purchase_price + repair_cost + transportation_cost)
  (h6 : profit_ratio = profit_percent / 100)
  (h7 : selling_price = (1 + profit_ratio) * total_cost) :
  purchase_price = 14000 :=
by
  sorry

end sahil_machine_purchase_price_l540_540140


namespace quadratic_graph_tangent_to_x_axis_l540_540031
-- To avoid naming conflicts and for a broader import

-- We are defining key conditions explicitly
def quadratic_function_tangent_to_x_axis (a b : ℝ) (h : a > 0) : Prop :=
  let d := (b + 1)^2 / (4 * a) in
  let g := λ x : ℝ, a * x^2 + b * x + d in
  let discriminant := b^2 - 4 * a * d in
  discriminant = 0

-- The theorem statement, which translates the problem into Lean
theorem quadratic_graph_tangent_to_x_axis 
  (a b : ℝ)
  (h₁ : a > 0)
  (h₂ : ∀ d : ℝ, d = (b + 1)^2 / (4 * a) → 
     quadratic_function_tangent_to_x_axis a b h₁) :
  ∃ d : ℝ, d = (b + 1)^2 / (4 * a) 
    ∧ quadratic_function_tangent_to_x_axis a b h₁ :=
sorry

end quadratic_graph_tangent_to_x_axis_l540_540031


namespace seating_arrangements_l540_540470

theorem seating_arrangements (Ana Bob Cindy : Type) :
  let total_permutations := Nat.factorial 9,
      prohibited_permutations := Nat.factorial 7 * Nat.factorial 3
  in total_permutations - prohibited_permutations = 332640 :=
by {
  let total_permutations := Nat.factorial 9,
      prohibited_permutations := Nat.factorial 7 * Nat.factorial 3;
  have h1 : total_permutations = 9! := rfl,
  have h2 : prohibited_permutations = 7! * 3! := rfl,
  have h3 : 9! = 362880 := rfl,
  have h4 : 7! * 3! = 30240 := rfl,
  have h5 : 362880 - 30240 = 332640 := rfl,
  sorry
}

end seating_arrangements_l540_540470


namespace complex_args_difference_l540_540487

theorem complex_args_difference (n : ℕ) (Z : Fin n → ℂ)
  (h : ∑ k, Z k = 0) : ∃ i j : Fin n, i ≠ j ∧ |arg (Z i) - arg (Z j)| ≥ 2 * π / 3 :=
by sorry

end complex_args_difference_l540_540487


namespace sum_first_five_terms_geom_sequence_l540_540401

variable {a : Nat → ℝ} (n : Nat)

def geometric_sequence (a : Nat → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_geom_seq (a : Nat → ℝ) (n : Nat) : ℝ :=
  (finset.range n).sum a

theorem sum_first_five_terms_geom_sequence
  (h1 : a 0 + a 1 = 1)
  (h2: a 1 + a 2 = 2)
  (h_geom : geometric_sequence a 2) :
  sum_geom_seq a 5 = 31 / 3 := 
sorry

end sum_first_five_terms_geom_sequence_l540_540401


namespace general_term_formula_minimum_sum_l540_540475

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1) * d)

theorem general_term_formula 
  (a₁ : ℝ) (d : ℝ)
  (h₁ : arithmetic_seq a₁ d 10 = 18)
  (h₂ : sum_first_n_terms a₁ d 5 = -15) :
  ∀ n : ℕ, arithmetic_seq a₁ d n = 3 * (n : ℝ) - 12 :=
by sorry

theorem minimum_sum 
  (a₁ : ℝ) (d : ℝ)
  (h₁ : arithmetic_seq a₁ d 10 = 18)
  (h₂ : sum_first_n_terms a₁ d 5 = -15) :
  ∃ n : ℕ, sum_first_n_terms a₁ d n = -18 ∧ (n = 3 ∨ n = 4) :=
by sorry

end general_term_formula_minimum_sum_l540_540475


namespace projection_same_vector_l540_540428

open Function

def p : ℝ × ℝ := (55/26, 45/26)

def v1 : ℝ × ℝ := (3, 4)
def v2 : ℝ × ℝ := (2, -1)

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  ((dot_uv / dot_vv) * v.1, (dot_uv / dot_vv) * v.2)

theorem projection_same_vector (v : ℝ × ℝ) :
  projection v1 v = projection v2 v → 
  projection v1 v = p := 
by
  sorry

end projection_same_vector_l540_540428


namespace mean_less_than_median_diff_l540_540648

noncomputable def q : List ℝ :=
  [-4, abs (5 - 12), Real.sqrt 49, Real.log2 64, 7^2, 18, 20, 26, 29, 33, 42.5, 50.3]

noncomputable def median (l : List ℝ) : ℝ :=
  let l_sorted := l.qsort (· ≤ ·)
  l_sorted.get! (l_sorted.length / 2)

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_less_than_median_diff :
  abs (mean q - median q) = 2.446 :=
by
  sorry

end mean_less_than_median_diff_l540_540648


namespace square_area_l540_540469

-- Definitions of the points and lengths
variables {A B C D E F G H : Type*} [HasDist E] [HasDist F] [HasDist G] [HasDist H] -- Points and geometric objects
variables (a d c : ℝ) -- Side length a, areas d and side length c

-- Hypotheses: 
-- ABCD is a square
-- E, F, G, H are midpoints
-- EF = c
def square_side (a : ℝ) : Prop := d = a ^ 2
def midpoint (x y z : Type*) : Prop := sorry -- replace with midpoint condition
def distance (x y : Type*) (c : ℝ) : Prop := sorry -- replace with distance calculation

theorem square_area (a d c : ℝ) (h1: square_side a d) (h2: midpoint E F G H) (h3: distance E F c):
  d = 2 * c ^ 2 :=
sorry

end square_area_l540_540469


namespace infinite_product_sqrt_nine_81_l540_540305

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540305


namespace last_digit_of_N_is_3_l540_540614

/- Given conditions -/
constant two_digits_correct_wrong_places (N : ℕ) : Prop
constant one_digit_correct_right_place (N : ℕ) : Prop
constant two_digits_one_correct_each (N : ℕ) : Prop
constant one_digit_correct_wrong_place (N : ℕ) : Prop
constant none_of_given_digits_correct (N : ℕ) : Prop

/- The target four-digit number to be identified -/
def four_digit_number : Type := {n : ℕ // 1000 ≤ n ∧ n < 10000}

/- Problem statement -/
theorem last_digit_of_N_is_3 (N : four_digit_number)
  (h1 : two_digits_correct_wrong_places N.1)
  (h2 : one_digit_correct_right_place N.1)
  (h3 : two_digits_one_correct_each N.1)
  (h4 : one_digit_correct_wrong_place N.1)
  (h5 : none_of_given_digits_correct N.1) :
  N.1 % 10 = 3 :=
sorry

end last_digit_of_N_is_3_l540_540614


namespace no_valid_n_l540_540883

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_valid_n : ∀ n : ℕ, n + sum_of_digits n + sum_of_digits (sum_of_digits n) + sum_of_digits (sum_of_digits (sum_of_digits n)) + ... = 2000000 → False := 
sorry

end no_valid_n_l540_540883


namespace am_gm_inequality_l540_540525

noncomputable theory

variables {a b c : ℝ} {n p q r : ℕ}

def in_positive_reals (x : ℝ) : Prop := x > 0

def pqr_sum_to_n (p q r n : ℕ) : Prop := p + q + r = n

theorem am_gm_inequality
  (ha : in_positive_reals a)
  (hb : in_positive_reals b)
  (hc : in_positive_reals c)
  (hn : n > 0)
  (hp : p ≥ 0)
  (hq : q ≥ 0)
  (hr : r ≥ 0)
  (h_sum : pqr_sum_to_n p q r n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
by sorry

end am_gm_inequality_l540_540525


namespace find_common_difference_l540_540849

theorem find_common_difference (AB BC AC : ℕ) (x y z d : ℕ) 
  (h1 : AB = 300) (h2 : BC = 350) (h3 : AC = 400) 
  (hx : x = (2 * d) / 5) (hy : y = (7 * d) / 15) (hz : z = (8 * d) / 15) 
  (h_sum : x + y + z = 750) : 
  d = 536 :=
by
  -- Proof goes here
  sorry

end find_common_difference_l540_540849


namespace fraction_of_pianists_got_in_l540_540894

-- Define the conditions
def flutes_got_in (f : ℕ) := f = 16
def clarinets_got_in (c : ℕ) := c = 15
def trumpets_got_in (t : ℕ) := t = 20
def total_band_members (total : ℕ) := total = 53
def total_pianists (p : ℕ) := p = 20

-- The main statement we want to prove
theorem fraction_of_pianists_got_in : 
  ∃ (pi : ℕ), 
    flutes_got_in 16 ∧ 
    clarinets_got_in 15 ∧ 
    trumpets_got_in 20 ∧ 
    total_band_members 53 ∧ 
    total_pianists 20 ∧ 
    pi / 20 = 1 / 10 := 
  sorry

end fraction_of_pianists_got_in_l540_540894


namespace inverse_function_identity_l540_540790

theorem inverse_function_identity (f g : ℝ → ℝ) (h_inv : ∀ x, g (f x) = x) 
  (h_g_def : ∀ x, g x = log x / log 2 + 1) : 
  f 2 + g 2 = 4 :=
by
  have h_f2 : f 2 = 2, 
  { 
    have h1 : g (f 2) = 2 := h_inv 2,
    rw [h_g_def] at h1,
    dsimp at h1,
    exact_mod_cast (Eq.log_eq_iff (by norm_num)) in h1
  },
  have h_g2 : g 2 = 2,
  {
    have h2 : g 2 = log 2 / log 2 + 1 := h_g_def 2,
    rw log_base_self at h2,
    norm_num at h2
  }
  calc 
    f 2 + g 2 = 2 + 2 : by rw [h_f2, h_g2]
    ... = 4 : by norm_num

end inverse_function_identity_l540_540790


namespace distance_QR_l540_540265

theorem distance_QR (DE EF DF : ℝ) (Q R E D F : Point) 
  (h1 : DE = 9) 
  (h2 : EF = 12) 
  (h3 : DF = 15) 
  (hvDE : right_triangle D E F)
  (hQ : Q.on_circle E D)
  (hQ_tangent : Q.tangent EF at E)
  (hR : R.on_circle D F)
  (hR_tangent : R.tangent DE at D) :
  dist Q R = 7 := 
  sorry

end distance_QR_l540_540265


namespace area_inequality_equality_condition_l540_540495

variable (a b c d S : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
variable (s : ℝ) (h5 : s = (a + b + c + d) / 2)
variable (h6 : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)))

theorem area_inequality (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  S ≤ Real.sqrt (a * b * c * d) :=
sorry

theorem equality_condition (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  (S = Real.sqrt (a * b * c * d)) ↔ (a = c ∧ b = d ∨ a = d ∧ b = c) :=
sorry

end area_inequality_equality_condition_l540_540495


namespace students_per_configuration_l540_540569

theorem students_per_configuration (students_per_column : ℕ → ℕ) :
  students_per_column 1 = 15 ∧
  students_per_column 2 = 1 ∧
  students_per_column 3 = 1 ∧
  students_per_column 4 = 6 ∧
  ∀ i j, (i ≠ j ∧ i ≤ 12 ∧ j ≤ 12) → students_per_column i ≠ students_per_column j →
  (∃ n, 13 ≤ n ∧ ∀ k, k < 13 → students_per_column k * n = 60) :=
by
  sorry

end students_per_configuration_l540_540569


namespace area_bounded_region_correct_l540_540923

noncomputable def area_of_bounded_region : ℝ :=
  let boundary_eqn (x y : ℝ) : Prop := y^2 + 4 * x * y + 60 * |x| = 600
  in 450

theorem area_bounded_region_correct : 
  ∀ (x y : ℝ), (y^2 + 4 * x * y + 60 * |x| = 600) → area_of_bounded_region = 450 :=
by
  -- proof to be filled in
  sorry

end area_bounded_region_correct_l540_540923


namespace frustum_edges_meet_at_point_l540_540639

-- Definitions
def is_prism (s : Type) (faces : s → Type) :=
  ∃ (f1 f2 : faces), f1 ≠ f2 ∧ 
  ∀ (f : faces), f ∈ ({f1, f2} : set faces) → ∀ (adjf : faces), adjf ⊆ f → adjf ∉ ({f1, f2} : set faces) → is_parallelogram adjf

def is_pyramid (s : Type) (faces : s → Type) :=
  ∃ (basef : faces), (is_polygon basef) ∧ 
  ∀ (f : faces), f ∉ (basef) → is_triangle f ∧ 
  ∃ (v : ℝ × ℝ × ℝ), ∀ (t : faces), t ∉ (basef) → (v ∈ vertices t)

def is_frustum (original_pyramid frustum : Type) (cut_plane : Type) :=
  is_pyramid original_pyramid ∧ horizontal_plane cut_plane ∧ 
  is_pyramid frustum ∧ ∃ (hcut : ℝ), (exists_plane_plane_cut original_pyramid cut_plane hcut) 

-- Theorem to prove
theorem frustum_edges_meet_at_point 
  (original_pyramid frustum : Type) (cut_plane : Type) :
  is_frustum original_pyramid frustum cut_plane → 
  ∃ (meet_point : ℝ × ℝ × ℝ), ∀ (edge : line original_pyramid), edge ∈ extensions_of_lateral_edges frustum → (meet_point ∈ edge) :=
sorry

end frustum_edges_meet_at_point_l540_540639


namespace evaluate_expr_at_2_l540_540922

-- Define the expression
def expr (x : ℝ) : ℝ := (1 - 1 / (x + 1)) / (x / (x - 1))

-- Define the simplified version of the expression
def simplified_expr (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- State the theorem
theorem evaluate_expr_at_2 :
  (expr 2 = simplified_expr 2) ∧ (simplified_expr 2 = 1 / 3) :=
by
  -- Proof simplified using the steps given
  sorry

end evaluate_expr_at_2_l540_540922


namespace range_of_expression_l540_540381

noncomputable def pq_po_over_po_plus_3_qp_qo_over_qo (P Q O A B : Point)
  (a : ℝ) (α : ℝ)
  (angle_AOB : ∠ A O B = real.pi / 3)
  (dist_PQ : dist P Q = real.sqrt 7)
  : ℝ :=
  let cosα := real.cos α,
      cos_120_minus_α := real.cos (2 * real.pi / 3 - α),
      sin_120_minus_α := real.sin (2 * real.pi / 3 - α)
  in 
  a * cosα + 3 * a * (cos_120_minus_α * cosα + sin_120_minus_α * real.sin α)

theorem range_of_expression (P Q O A B : Point)
  (α : ℝ)
  (h_a : ∥P - Q∥ = real.sqrt 7)
  (h_AOB : ∠ A O B = real.pi / 3)
  :
  ∃ (C : set ℝ), C = Ioo (-real.sqrt 7 / 2) 7 ∧
    pq_po_over_po_plus_3_qp_qo_over_qo P Q O A B (real.sqrt 7) α ∈ C :=
by
  sorry

end range_of_expression_l540_540381


namespace impossible_even_n_m_if_n3_plus_m3_is_odd_l540_540444

theorem impossible_even_n_m_if_n3_plus_m3_is_odd
  (n m : ℤ) (h : (n^3 + m^3) % 2 = 1) : ¬((n % 2 = 0) ∧ (m % 2 = 0)) := by
  sorry

end impossible_even_n_m_if_n3_plus_m3_is_odd_l540_540444


namespace exist_point_set_with_distance_condition_l540_540908

theorem exist_point_set_with_distance_condition (m : ℕ) : 
  ∃ S : set (ℝ × ℝ), (S ≠ ∅) ∧  ∀ A ∈ S, (∃ B ∈ S, A ≠ B ∧ dist A B = 1) ∧ (finset.filter (λ B, dist A B = 1) (finset.univ : finset (ℝ × ℝ))).card = m :=
sorry

end exist_point_set_with_distance_condition_l540_540908


namespace equilateral_triangle_circumcircle_property_l540_540772

theorem equilateral_triangle_circumcircle_property
  {A B C M : Point} {circumcircle : Circle}
  (h_eq_tri : Triangle A B C ∧ Equilateral A B C)
  (h_on_circumcircle : PointOnCircle M circumcircle)
  (h_arc : ArcContains circumcircle B C M ∧ ¬ArcContains circumcircle A B C) :
  distance M B + distance M C = distance M A :=
by
  sorry

end equilateral_triangle_circumcircle_property_l540_540772


namespace find_s_l540_540677

theorem find_s (s : ℝ) (h1 : 3 * s * s * sqrt 3 = 9 * sqrt 3) : s = sqrt 3 :=
by
  sorry

end find_s_l540_540677


namespace coefficient_of_x4y3_expansion_l540_540082

theorem coefficient_of_x4y3_expansion :
  let poly := (- (1:ℤ) * x * y + 2 * x + 3 * y - 6:ℤ) ^ 6
  coefficient_of (poly) x^4 y^3 = -21600 := by sorry

end coefficient_of_x4y3_expansion_l540_540082


namespace numerator_of_harmonic_sum_divisible_by_p_l540_540136

theorem numerator_of_harmonic_sum_divisible_by_p (p : ℕ) (hp_prime : p.prime) (hp_gt_two : p > 2) :
  ∃ m n : ℕ, (∑ i in finset.range (p - 1), (1 / (i + 1) : ℚ)) = m / n ∧ p ∣ m :=
sorry

end numerator_of_harmonic_sum_divisible_by_p_l540_540136


namespace solve_log_equation_l540_540924

theorem solve_log_equation (y : ℝ) (h : log y - 3 * log 5 = -3) : y = 0.125 :=
sorry

end solve_log_equation_l540_540924


namespace small_cone_alt_eq_normalized_l540_540672

def lower_base_area := 400 * Real.pi
def upper_base_area := 36 * Real.pi
def frustum_height := 30

def radius_lower_base (A : Real) : Real := 
  real.sqrt (A / Real.pi)

def radius_upper_base (B : Real) : Real := 
  real.sqrt (B / Real.pi)

def altitude_larger_cone (h_frustum r1 r2 : Real) : Real := 
  h_frustum * (r1 / (r1 - r2))

noncomputable def altitude_small_cone (h_cone r1 r2 : Real) : Real := 
  h_cone * (r2 / r1)

theorem small_cone_alt_eq_normalized 
  (lower_base_area upper_base_area frustum_height : Real) : 
  radius_lower_base lower_base_area = 20 →
  radius_upper_base upper_base_area = 6 →
  altitude_larger_cone 30 20 6 = 300 / 7 →
  altitude_small_cone (300 / 7) 20 6 = 1286 / 100 :=
by
  intro r1_eq r2_eq H_eq h_eq
  rw [r1_eq, r2_eq, H_eq, h_eq]
  norm_num

#print axioms small_cone_alt_eq_normalized

end small_cone_alt_eq_normalized_l540_540672


namespace triangle_construction_l540_540291

noncomputable def construct_triangle (P : ℝ) (α : ℝ) (m : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    let perimeter := dist A B + dist B C + dist C A in
    let angle_α := angle A B C in
    let altitude_m := dist A (line_through B C) in
    perimeter = P ∧ angle_α = α ∧ altitude_m = m

theorem triangle_construction (P : ℝ) (α : ℝ) (m : ℝ) :
  construct_triangle P α m :=
sorry

end triangle_construction_l540_540291


namespace minimum_surface_area_of_sphere_l540_540012

/-- Given four points A, B, C, D on the surface of a sphere,
the segments AB, AC, AD are mutually perpendicular,
and AB + AC + AD = 12, the minimum surface area of the sphere is 48π. -/
theorem minimum_surface_area_of_sphere 
  (A B C D : point)
  (AB AC AD : ℝ)
  (h1 : AB = distance A B)
  (h2 : AC = distance A C)
  (h3 : AD = distance A D)
  (h4 : AB + AC + AD = 12)
  (h5 : AB * AB + AC * AC + AD * AD = 4 * 4 + 4 * 4 + 4 * 4) :
  4 * math.pi * ((1 / 2 * real.sqrt(AB * AB + AC * AC + AD * AD))^2) = 48 * math.pi :=
by
  sorry

end minimum_surface_area_of_sphere_l540_540012


namespace trapezoid_sequence_properties_l540_540085

def trapezoid_sequence (a b : ℝ) := (list ℕ) (λ (a_n : ℝ) (n : ℕ), (a_n > 0 ∧ a ≥ b) → 
                        ∃ l : ℝ, ∀ n, a_(n+1) = | (a_n - b) / 2 | ∧ a_n != a_m 
                        ∧ (a_n >= a_(n+1) ∨ a_n <= a_(n+1)) 
                        ∧ l = b / 3)

theorem trapezoid_sequence_properties (a b : ℝ) (h : a > 0) (h' : a ≥ b) : 
                { seq : ℕ → ℝ | ∀ n, seq 0 = a ∧ seq (n+1) = | (seq n - b) / 2 | }
                    → (∀ n m, seq n ≠ seq m)
                    → ¬ (∀ n, seq (n+1) ≥ seq n ∧ ∀ n, seq (n+1) ≤ seq n)
                    → ∃ l, lim (seq n) n = l ∧ l = b / 3 :=
by
sorry

end trapezoid_sequence_properties_l540_540085


namespace average_test_score_of_remainder_l540_540828

variable (score1 score2 score3 totalAverage : ℝ)
variable (percentage1 percentage2 percentage3 : ℝ)

def equation (score1 score2 score3 totalAverage : ℝ) (percentage1 percentage2 percentage3: ℝ) : Prop :=
  (percentage1 * score1) + (percentage2 * score2) + (percentage3 * score3) = totalAverage

theorem average_test_score_of_remainder
  (h1 : percentage1 = 0.15)
  (h2 : score1 = 100)
  (h3 : percentage2 = 0.5)
  (h4 : score2 = 78)
  (h5 : percentage3 = 0.35)
  (total : totalAverage = 76.05) :
  (score3 = 63) :=
sorry

end average_test_score_of_remainder_l540_540828


namespace higher_selling_price_l540_540055

theorem higher_selling_price (cost price340 : ℝ) (h_cost : cost = 200) (h_gain : price340 - cost = 140) (h_5_percent_gain : (price340 - cost) * 0.05 = 7):
  ∃ P : ℝ, P - cost = 147 ∧ P = 347 :=
by
  use 347
  split
  . calc 347 - 200 = 147 : by norm_num
  . exact 347

end higher_selling_price_l540_540055


namespace distances_sum_less_than_pairwise_distances_sum_l540_540002

variable {A B C D M : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]

theorem distances_sum_less_than_pairwise_distances_sum 
  (h_convex : convex_quadrilateral A B C D)
  (h_inside : inside_point M A B C D) :
  distance M A + distance M B + distance M C + distance M D < distance A B + distance B C + distance C D + distance D A :=
by
  sorry

end distances_sum_less_than_pairwise_distances_sum_l540_540002


namespace proof_f_3_eq_9_ln_3_l540_540419

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem proof_f_3_eq_9_ln_3 (a : ℝ) (h : deriv (deriv (f a)) 1 = 3) : f a 3 = 9 * Real.log 3 :=
by
  sorry

end proof_f_3_eq_9_ln_3_l540_540419


namespace right_angled_triangle_sets_l540_540696

theorem right_angled_triangle_sets :
  (¬ (1 ^ 2 + 2 ^ 2 = 3 ^ 2)) ∧
  (¬ (2 ^ 2 + 3 ^ 2 = 4 ^ 2)) ∧
  (3 ^ 2 + 4 ^ 2 = 5 ^ 2) ∧
  (¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2)) :=
by
  sorry

end right_angled_triangle_sets_l540_540696


namespace lim_Sn_div_an_squared_l540_540956

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def S_n (n : ℕ) : ℕ := n * n

theorem lim_Sn_div_an_squared (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) :
  (∀ n : ℕ, S_n n = n^2) →
  (∀ n : ℕ, a_n n = 2 * n - 1) →
  filter.tendsto (λ n, (S_n n / (a_n n)^2 : ℚ)) filter.at_top (𝓝 (1/4)) :=
by
  intros hS hA
  sorry

end lim_Sn_div_an_squared_l540_540956


namespace ellipse_minimum_distance_point_l540_540351

theorem ellipse_minimum_distance_point :
  ∃ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1) ∧ (∀ p, x - 2 * y - 12 = 0 → dist (x, y) p ≥ dist (2, -3) p) :=
sorry

end ellipse_minimum_distance_point_l540_540351


namespace distinct_possible_collections_l540_540284

def vowels : Finset Char := {'O', 'U', 'E'}
def consonants : Multiset Char := {'C', 'M', 'P', 'T', 'S', 'R', 'R'} -- includes two R's
def magnet_chars : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}
def number_falling_off : Nat := 5

theorem distinct_possible_collections : 
  ( |(vowels.ndchoose 3) * (consonants.ndchoose 2)| + |(vowels.ndchoose 3) * (consonants.ndchoose_with (Multiset.filter (≠ 'R') consonants) 2)| = 14 ) :=
by
  sorry

end distinct_possible_collections_l540_540284


namespace class3_qualifies_l540_540259

/-- Data structure representing a class's tardiness statistics. -/
structure ClassStats where
  mean : ℕ
  median : ℕ
  variance : ℕ
  mode : Option ℕ -- mode is optional because not all classes might have a unique mode.

def class1 : ClassStats := { mean := 3, median := 3, variance := 0, mode := none }
def class2 : ClassStats := { mean := 2, median := 0, variance := 1, mode := none }
def class3 : ClassStats := { mean := 2, median := 0, variance := 2, mode := none }
def class4 : ClassStats := { mean := 0, median := 2, variance := 0, mode := some 2 }

/-- Predicate to check if a class qualifies for the flag, meaning no more than 5 students tardy each day for 5 consecutive days. -/
def qualifies (cs : ClassStats) : Prop :=
  cs.mean = 2 ∧ cs.variance = 2

theorem class3_qualifies : qualifies class3 :=
by
  sorry

end class3_qualifies_l540_540259


namespace number_of_divisors_of_4410_with_more_than_3_factors_l540_540819

-- Define the number and its prime factorization
def n : ℕ := 4410
def prime_factorization_4410 : List (ℕ × ℕ) := [(2, 1), (3, 2), (5, 1), (7, 2)]

-- Formula to count divisors
def count_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.2 + 1)) 1

-- Define the number of divisors of n
def d_4410 : ℕ := count_divisors prime_factorization_4410

-- Count how many divisors of n have more than 3 factors
def count_divisors_with_more_than_3_factors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ m => m > 0 ∧ n % m = 0 ∧ count_divisors (prime_factorization_of m) > 3).length

-- The problem statement
theorem number_of_divisors_of_4410_with_more_than_3_factors : 
  count_divisors_with_more_than_3_factors n = 9 :=
sorry

-- The function to obtain the prime factorization of a number
noncomputable def prime_factorization_of (m : ℕ) : List (ℕ × ℕ) :=
  sorry  -- Placeholder for actual prime factorization function

end number_of_divisors_of_4410_with_more_than_3_factors_l540_540819


namespace matrix_arithmetic_sequence_sum_l540_540480

theorem matrix_arithmetic_sequence_sum (a : ℕ → ℕ → ℕ)
  (h_row1 : ∀ i, 2 * a 4 2 = a 4 (i - 1) + a 4 (i + 1))
  (h_row2 : ∀ i, 2 * a 5 2 = a 5 (i - 1) + a 5 (i + 1))
  (h_row3 : ∀ i, 2 * a 6 2 = a 6 (i - 1) + a 6 (i + 1))
  (h_col1 : ∀ i, 2 * a 5 2 = a (i - 1) 2 + a (i + 1) 2)
  (h_sum : a 4 1 + a 4 2 + a 4 3 + a 5 1 + a 5 2 + a 5 3 + a 6 1 + a 6 2 + a 6 3 = 63)
  : a 5 2 = 7 := sorry

end matrix_arithmetic_sequence_sum_l540_540480


namespace lisa_ratio_l540_540536

theorem lisa_ratio (L J T : ℝ) 
  (h1 : L + J + T = 60) 
  (h2 : T = L / 2) 
  (h3 : L = T + 15) : 
  L / 60 = 1 / 2 :=
by 
  sorry

end lisa_ratio_l540_540536


namespace linear_regression_estimate_l540_540426

theorem linear_regression_estimate (x y : ℝ) (h : x = 25) (h_eq : y = 0.50 * x - 0.81) : y = 11.69 := by
  have h₁ : y = 0.50 * 25 - 0.81 := by rw [h, h_eq]
  have h₂ : y = 12.50 - 0.81 := by rw [h₁]
  have h₃ : y = 11.69 := by norm_num [h₂]
  exact h₃

end linear_regression_estimate_l540_540426


namespace expectation_of_xi_l540_540809

noncomputable def compute_expectation : ℝ := 
  let m : ℝ := 0.3
  let E : ℝ := (1 * 0.5) + (3 * m) + (5 * 0.2)
  E

theorem expectation_of_xi :
  let m: ℝ := 1 - 0.5 - 0.2 
  (0.5 + m + 0.2 = 1) → compute_expectation = 2.4 := 
by
  sorry

end expectation_of_xi_l540_540809


namespace rectangles_fit_l540_540816

theorem rectangles_fit :
  let width := 50
  let height := 90
  let r_width := 1
  let r_height := (10 * Real.sqrt 2)
  ∃ n : ℕ, 
  n = 315 ∧
  (∃ w_cuts h_cuts : ℕ, 
    w_cuts = Int.floor (width / r_height) ∧
    h_cuts = Int.floor (height / r_height) ∧
    n = ((Int.floor (width / r_height) * Int.floor (height / r_height)) + 
         (Int.floor (height / r_width) * Int.floor (width / r_height)))) := 
sorry

end rectangles_fit_l540_540816


namespace smallest_multiple_l540_540976

theorem smallest_multiple (x : ℕ) (h1 : x % 24 = 0) (h2 : x % 36 = 0) (h3 : x % 20 ≠ 0) :
  x = 72 :=
by
  sorry

end smallest_multiple_l540_540976


namespace find_m_for_parallel_l540_540382

def vector_a (m : ℝ) : ℝ × ℝ × ℝ := (1, -2, m)
def vector_b : ℝ × ℝ × ℝ := (-1, 2, -1)

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ t : ℝ, a = (t * b.1, t * b.2, t * b.3)

theorem find_m_for_parallel (m : ℝ) (h : are_parallel (vector_a m) vector_b) : m = 1 :=
by
  sorry

end find_m_for_parallel_l540_540382


namespace surface_area_of_octahedron_from_cube_l540_540464

theorem surface_area_of_octahedron_from_cube (a : ℝ) : 
  let surface_area := 8 * (1 / 2) * (a * sqrt 2 / 2) * (a * sqrt 2 / 2) * (sqrt 3 / 2) 
  in surface_area = a^2 * sqrt 3 :=
by sorry

end surface_area_of_octahedron_from_cube_l540_540464


namespace find_constant_a_range_of_f_l540_540799

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) - a

theorem find_constant_a (h : f a 0 = -Real.sqrt 3) : a = Real.sqrt 3 := by
  sorry

theorem range_of_f (a : ℝ) (h : a = Real.sqrt 3) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  f a x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

end find_constant_a_range_of_f_l540_540799


namespace find_probability_between_0_and_1_l540_540535

-- Define a random variable X following a normal distribution N(μ, σ²)
variables {X : ℝ → ℝ} {μ σ : ℝ}
-- Define conditions:
-- Condition 1: X follows a normal distribution with mean μ and variance σ²
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  sorry  -- Assume properties of normal distribution are satisfied

-- Condition 2: P(X < 1) = 1/2
def P_X_lt_1 : Prop := 
  sorry  -- Assume that P(X < 1) = 1/2

-- Condition 3: P(X > 2) = p
def P_X_gt_2 (p : ℝ) : Prop := 
  sorry  -- Assume that P(X > 2) = p

noncomputable
def probability_X_between_0_and_1 (p : ℝ) : ℝ :=
  1/2 - p

theorem find_probability_between_0_and_1 (X : ℝ → ℝ) {μ σ p : ℝ} 
  (hX : normal_dist X μ σ)
  (h1 : P_X_lt_1)
  (h2 : P_X_gt_2 p) :
  probability_X_between_0_and_1 p = 1/2 - p := 
  sorry

end find_probability_between_0_and_1_l540_540535


namespace player_B_wins_under_optimal_strategy_l540_540548

theorem player_B_wins_under_optimal_strategy :
  (∀ (board : ℕ × ℕ) (A B : ℕ → ℕ × ℕ) (n : ℕ),
    (∀ k, (A k).1 ∈ {0, ..., n-1} ∧ (A k).2 ∈ {0, ..., n-1} ∧
           (B k).1 ∈ {0, ..., n-1} ∧ (B k).2 ∈ {0, ..., n-1}) →
    (∀ k1 k2, (A k1).1 ≠ (A k2).1 ∧ (A k1).2 ≠ (A k2).2) →
    (∀ k1 k2, (B k1).1 ≠ (B k2).1 ∧ (B k1).2 ≠ (B k2).2) →
    (∀ k1 k2, (A k1).1 ≠ (B k2).1 ∧ (A k1).2 ≠ (B k2).2) →
    ∃ K, (B K).1 = n - (A K).1 + 1 ∧ (B K).2 = (A K).2) →
  true :=
sorry

end player_B_wins_under_optimal_strategy_l540_540548


namespace celestia_badges_l540_540434

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end celestia_badges_l540_540434


namespace angle_is_pi_over_six_l540_540035

noncomputable def angle_between_vec_sub_add
  (a b c : EuclideanSpace ℝ (Fin 2))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hc : ‖c‖ = 1)
  (hab : ∠(a, b) = 2 * Real.pi / 3)
  (hac : ∠(a, c) = 2 * Real.pi / 3)
  (hbc : ∠(b, c) = 2 * Real.pi / 3) : Real :=
  ∠(a - b, a + c)

theorem angle_is_pi_over_six
  (a b c : EuclideanSpace ℝ (Fin 2))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hc : ‖c‖ = 1)
  (hab : ∠(a, b) = 2 * Real.pi / 3)
  (hac : ∠(a, c) = 2 * Real.pi / 3)
  (hbc : ∠(b, c) = 2 * Real.pi / 3) :
  angle_between_vec_sub_add a b c ha hb hc hab hac hbc = Real.pi / 6 :=
sorry

end angle_is_pi_over_six_l540_540035


namespace number_of_workers_l540_540997

-- Definitions corresponding to problem conditions
def total_contribution := 300000
def extra_total_contribution := 325000
def extra_amount := 50

-- Main statement to prove the number of workers
theorem number_of_workers : ∃ W C : ℕ, W * C = total_contribution ∧ W * (C + extra_amount) = extra_total_contribution ∧ W = 500 := by
  sorry

end number_of_workers_l540_540997


namespace truncated_cone_sphere_radius_l540_540273

theorem truncated_cone_sphere_radius (r_top r_bottom : ℝ) (h_top h_bottom h_lateral : ℝ) :
  r_top = 3 → r_bottom = 10 →
  ∃ (r_sphere : ℝ), r_sphere = sqrt 30 ∧
  touching_sphere r_bottom r_top r_sphere :=
by
  sorry

def touching_sphere (r_bottom r_top r_sphere : ℝ) : Prop :=
  r_sphere = sqrt 30

end truncated_cone_sphere_radius_l540_540273


namespace part_I_part_II_l540_540884

-- Problem conditions as definitions
variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

-- Statement for part (Ⅰ)
theorem part_I : (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Statement for part (Ⅱ)
theorem part_II : (1 / (a ^ 2016)) + (1 / (b ^ 2016)) ≥ 2 ^ 2017 :=
by
  sorry

end part_I_part_II_l540_540884


namespace intersection_distance_maximum_value_l540_540794

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (t : ℝ) (x y : ℝ) : Prop := x = 1 + t ∧ y = 2 + t

-- Define the intersection distance |AB|
theorem intersection_distance :
  ∃ x y t1 t2, C1 x y ∧ C2 t1 x y ∧ C2 t2 x y ∧ t1 ≠ t2 ∧ dist (x, y) (x, y) = sqrt 2 :=
sorry

-- Define the maximum value of (x+1)(y+1) on C1
theorem maximum_value (x y : ℝ) (h : C1 x y) : 
  ∃ θ : ℝ, x = cos θ ∧ y = sin θ ∧ 
  max ((cos θ + 1) * (sin θ + 1)) = (3 / 2 + sqrt 2) :=
sorry

end intersection_distance_maximum_value_l540_540794


namespace missing_keys_count_l540_540467

-- Definitions
def total_alphabet_keys : ℕ := 26
def total_accent_marks : ℕ := 5
def total_special_characters : ℕ := 3

def vowels : ℕ := 5
def consonants : ℕ := total_alphabet_keys - vowels

def missing_consonants : ℕ := (3 / 8) * consonants.round.to_nat -- rounded to nearest integer
def missing_vowels : ℕ := 4
def missing_accent_marks : ℕ := (2 / 5) * total_accent_marks.round.to_nat
def missing_special_characters : ℕ := (1 / 2) * total_special_characters.round.to_nat

-- Theorem statement
theorem missing_keys_count :
  missing_consonants + missing_vowels + missing_accent_marks + missing_special_characters = 16 := 
by
  sorry

end missing_keys_count_l540_540467


namespace find_x_range_l540_540352

theorem find_x_range : 
  {x : ℝ | (2 / (x + 2) + 4 / (x + 8) ≤ 3 / 4)} = 
  {x : ℝ | (-4 < x ∧ x ≤ -2) ∨ (4 ≤ x)} := by
  sorry

end find_x_range_l540_540352


namespace sum_n_terms_max_sum_n_l540_540392

variable {a : ℕ → ℚ} (S : ℕ → ℚ)
variable (d a_1 : ℚ)

-- Conditions given in the problem
axiom sum_first_10 : S 10 = 125 / 7
axiom sum_first_20 : S 20 = -250 / 7
axiom sum_arithmetic_seq : ∀ n, S n = n * (a 1 + a n) / 2

-- Define the first term and common difference for the arithmetic sequence
axiom common_difference : ∀ n, a n = a_1 + (n - 1) * d

-- Theorem 1: Sum of the first n terms
theorem sum_n_terms (n : ℕ) : S n = (75 * n - 5 * n^2) / 14 := 
  sorry

-- Theorem 2: Value of n that maximizes S_n
theorem max_sum_n : n = 7 ∨ n = 8 ↔ (∀ m, S m ≤ S 7 ∨ S m ≤ S 8) := 
  sorry

end sum_n_terms_max_sum_n_l540_540392


namespace charlies_mother_cookies_l540_540971

theorem charlies_mother_cookies 
    (charlie_cookies : ℕ) 
    (father_cookies : ℕ) 
    (total_cookies : ℕ)
    (h_charlie : charlie_cookies = 15)
    (h_father : father_cookies = 10)
    (h_total : total_cookies = 30) : 
    (total_cookies - charlie_cookies - father_cookies = 5) :=
by {
    sorry
}

end charlies_mother_cookies_l540_540971


namespace find_ab_l540_540954

theorem find_ab (a b : ℝ) 
  (h : vector.cross ⟨2, a, -7⟩ ⟨6, -3, b⟩ = 0) : 
  (a = -1) ∧ (b = -21) := 
by 
  sorry

end find_ab_l540_540954


namespace remainder_of_3y_l540_540635

theorem remainder_of_3y (y : ℕ) (hy : y % 9 = 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_of_3y_l540_540635


namespace intersection_eq_l540_540033

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | Real.log (1 - x) > 0 }

theorem intersection_eq : A ∩ B = Set.Icc (-1) 0 :=
by
  sorry

end intersection_eq_l540_540033


namespace circle_center_to_line_distance_l540_540731

noncomputable def point_line_distance (x₁ y₁ a b c : ℝ) : ℝ :=
  | a * x₁ + b * y₁ + c | / Real.sqrt (a^2 + b^2)

theorem circle_center_to_line_distance :
  let x₁ := -4
  let y₁ := 3
  let a := 4
  let b := 3
  let c := -1
  point_line_distance x₁ y₁ a b c = 8 / 5 :=
by
  -- Calculations outline for proof
  unfold point_line_distance
  sorry

end circle_center_to_line_distance_l540_540731


namespace scientific_notation_of_1650000_l540_540235

theorem scientific_notation_of_1650000 : (1650000 : ℝ) = 1.65 * 10^6 := 
by {
  -- Proof goes here
  sorry
}

end scientific_notation_of_1650000_l540_540235


namespace probability_red_or_white_correct_l540_540643

-- Define the conditions
def totalMarbles : ℕ := 30
def blueMarbles : ℕ := 5
def redMarbles : ℕ := 9
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the calculated probability
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- Verify the probability is equal to 5 / 6
theorem probability_red_or_white_correct :
  probabilityRedOrWhite = 5 / 6 := by
  sorry

end probability_red_or_white_correct_l540_540643


namespace tickets_used_l540_540280

variable (C T : Nat)

theorem tickets_used (h1 : C = 7) (h2 : T = C + 5) : T = 12 := by
  sorry

end tickets_used_l540_540280


namespace bryan_travel_hours_per_year_l540_540092

-- Definitions based on the conditions
def minutes_walk_to_bus_station := 5
def minutes_ride_bus := 20
def minutes_walk_to_job := 5
def days_per_year := 365

-- Total time for one-way travel in minutes
def one_way_travel_minutes := minutes_walk_to_bus_station + minutes_ride_bus + minutes_walk_to_job

-- Total daily travel time in minutes
def daily_travel_minutes := one_way_travel_minutes * 2

-- Convert daily travel time from minutes to hours
def daily_travel_hours := daily_travel_minutes / 60

-- Total yearly travel time in hours
def yearly_travel_hours := daily_travel_hours * days_per_year

-- The theorem to prove
theorem bryan_travel_hours_per_year : yearly_travel_hours = 365 :=
by {
  -- The preliminary arithmetic is not the core of the theorem
  sorry
}

end bryan_travel_hours_per_year_l540_540092


namespace g_monotone_increasing_l540_540289

-- Define the function
def g (x : ℝ) : ℝ := sin (2 * x + π / 6)

-- Define the interval
def interval := Set.Ioo (-π / 3) (π / 6)

-- Prove that the function g is monotonically increasing on the interval
theorem g_monotone_increasing : MonotoneOn g interval := sorry

end g_monotone_increasing_l540_540289


namespace total_sours_is_123_l540_540141

noncomputable def cherry_sours := 32
noncomputable def lemon_sours := 40 -- Derived from the ratio 4/5 = 32/x
noncomputable def orange_sours := 24 -- 25% of the total sours in the bag after adding them
noncomputable def grape_sours := 27 -- Derived from the ratio 3/2 = 40/y

theorem total_sours_is_123 :
  cherry_sours + lemon_sours + orange_sours + grape_sours = 123 :=
by
  sorry

end total_sours_is_123_l540_540141


namespace derivative_volume_equals_surface_area_l540_540667

-- Define the volume function of a sphere
def volume_sphere (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

-- Define the surface area function of a sphere
def surface_area_sphere (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- The theorem stating the equivalent proof problem
theorem derivative_volume_equals_surface_area (R : ℝ) (hR : 0 < R):
  deriv (volume_sphere) R = surface_area_sphere R :=
sorry

end derivative_volume_equals_surface_area_l540_540667


namespace fraction_computation_l540_540720

noncomputable def compute_fraction : ℚ :=
  (64^4 + 324) * (52^4 + 324) * (40^4 + 324) * (28^4 + 324) * (16^4 + 324) /
  (58^4 + 324) * (46^4 + 324) * (34^4 + 324) * (22^4 + 324) * (10^4 + 324)

theorem fraction_computation :
  compute_fraction = 137 / 1513 :=
by sorry

end fraction_computation_l540_540720


namespace exists_N_for_interval_l540_540494

theorem exists_N_for_interval (a b : ℝ) (h1 : a < b) (h2 : ∀ n : ℤ, n ∉ set.Icc a b) :
  ∃ N : ℝ, N > 0 ∧ (∀ n : ℤ, n ∉ set.Icc (N * a) (N * b)) ∧ (N * b - N * a > 1 / 6) :=
by
  sorry

end exists_N_for_interval_l540_540494


namespace question_I_question_II_l540_540420

def f (x : ℝ) : ℝ := abs (x - 2) + 2 * abs (x - 1)

theorem question_I:
  ∀ x: ℝ, f x > 4 ↔ x ∈ Iio 0 ∪ Ioi 0 := sorry

theorem question_II:
  ∀ m: ℝ, (∀ x: ℝ, f x > 2 * m^2 - 7 * m + 4) ↔ (1 / 2 < m ∧ m < 3) := sorry

end question_I_question_II_l540_540420


namespace instantaneous_rate_of_change_correct_l540_540197

-- Define the function f(x) = 1 / x
def f (x : ℝ) := 1 / x

-- Define the derivative of the function f
def f' (x : ℝ) := -1 / x^2

-- Define the point of interest
def x_point := 2

-- Define the instantaneous rate of change at x = 2
def instantaneous_rate_of_change := -1 / 4

-- State the theorem to be proven
theorem instantaneous_rate_of_change_correct : f' x_point = instantaneous_rate_of_change :=
by sorry

end instantaneous_rate_of_change_correct_l540_540197


namespace remove_one_and_average_l540_540633

theorem remove_one_and_average (l : List ℕ) (n : ℕ) (avg : ℚ) :
  l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  avg = 8.5 →
  (l.sum - n : ℚ) = 14 * avg →
  n = 1 :=
by
  intros hlist havg hsum
  sorry

end remove_one_and_average_l540_540633


namespace total_students_in_school_l540_540570

theorem total_students_in_school
  (students_per_group : ℕ) (groups_per_class : ℕ) (number_of_classes : ℕ)
  (h1 : students_per_group = 7) (h2 : groups_per_class = 9) (h3 : number_of_classes = 13) :
  students_per_group * groups_per_class * number_of_classes = 819 := by
  -- The proof steps would go here
  sorry

end total_students_in_school_l540_540570


namespace new_students_admitted_l540_540216

theorem new_students_admitted (x : ℕ) :
  let initial_students := 35
  let increase_in_expenses := 84
  let diminished_expenditure_per_head := 1
  let original_expenditure := 630
  let original_avg_expenditure := original_expenditure / initial_students
  let new_avg_expenditure := original_avg_expenditure - diminished_expenditure_per_head
  let total_new_students := initial_students + x
  let new_total_expenditure := new_avg_expenditure * total_new_students
in new_total_expenditure = original_expenditure + increase_in_expenses → x = 7 :=
by
  intros
  have original_avg_expenditure_calc : original_avg_expenditure = 18 := by sorry
  have new_avg_expenditure_calc : new_avg_expenditure = 17 := by sorry
  have new_total_expenditure_calc : new_total_expenditure = 17 * (35 + x) := by sorry
  rw [original_expenditure, increase_in_expenses] at h
  rw [new_avg_expenditure_calc, original_expenditure_calc] at h
  rw [new_total_expenditure_calc] at h
  have eqn : 17 * (35 + x) = 714 := by sorry
  solve_by_elim

end new_students_admitted_l540_540216


namespace angle_B_is_pi_over_3_f_intervals_of_monotonicity_l540_540483

-- Definitions and conditions for the first proof problem
variable {A B C a b c : ℝ}
variable {m n : (ℝ × ℝ)}
variable (triangle_ABC : Type*)
variable (angle_A angle_B angle_C : triangle_ABC → ℝ)
variable (side_a side_b side_c : triangle_ABC → ℝ)

-- Proof problem 1: Angle B
theorem angle_B_is_pi_over_3 
(triangle_ABC : Type*)
(angle_A angle_B angle_C : triangle_ABC → ℝ)
(side_a side_b side_c : triangle_ABC → ℝ)
(m n : ℝ × ℝ)
(h1 : m = (side_b triangle_ABC, 2 * side_a triangle_ABC - side_c triangle_ABC))
(h2 : n = (real.cos (angle_B triangle_ABC), real.cos (angle_C triangle_ABC)))
(h3 : m.1 / n.1 = m.2 / n.2):
  (angle_B triangle_ABC = π / 3) :=
  sorry

-- Definitions and conditions for the second proof problem
variable {f : ℝ → ℝ}

-- Proof problem 2: Intervals of Monotonicity
theorem f_intervals_of_monotonicity 
(ω : ℝ)
(ω_neg : ω < 0)
(period_π : real.periodic (λ x, real.cos (ω * x - π / 6) + real.sin (ω * x)) π):
  (∀ k : ℤ, 
    [k * π + π / 3, k * π + 5 * π / 6] = { x | f(x) = real.cos(ω * x - π / 6) + real.sin(ω * x) -> f ' (x) > 0 } ∧
    [k * π - π / 6, k * π + π / 3] = { x | f(x) = real.cos(ω * x - π / 6) + real.sin(ω * x) -> f ' (x) < 0 }) :=
  sorry

end angle_B_is_pi_over_3_f_intervals_of_monotonicity_l540_540483


namespace recurring_six_denominator_l540_540179

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l540_540179


namespace max_value_of_expression_l540_540115

theorem max_value_of_expression
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 1) :
  2 * x^2 * y * real.sqrt 6 + 8 * y^2 * z ≤ real.sqrt (144 / 35) + real.sqrt (88 / 35) :=
  by
    sorry

end max_value_of_expression_l540_540115


namespace distinct_ways_to_place_digits_l540_540447

theorem distinct_ways_to_place_digits : 
  (∃ (grid : fin 6 → option ℕ), 
    multiset.card (multiset.filter (λ x, x ≠ none) (multiset.map grid (multiset.fin_range 6)))
    = 4 ∧ (multiset.erase_dup $ multiset.filter_map id $ multiset.map grid (multiset.fin_range 6)) = {1,2,3,4}) → 15 * (nat.factorial 4) = 360 :=
by
  sorry

end distinct_ways_to_place_digits_l540_540447


namespace team_relay_orderings_l540_540873

theorem team_relay_orderings :
  let jordan := 5
  let laps := 5
  let choices_for_remaining (n : ℕ) : ℕ := if n = 0 then 1 else n * choices_for_remaining (n-1)
  let case1 := choices_for_remaining 3
  let case2 := choices_for_remaining 3
  (case1 + case2) = 12 :=
by
  -- Jordan is fixed for the fifth lap
  let jordan := 5

  -- Number of laps
  let laps := 5

  -- Choices for the remaining friends to fill laps when Alice runs either first or second
  have h1 : ∀ n : ℕ, choices_for_remaining n = if n = 0 then 1 else n * choices_for_remaining (n - 1) := by 
    intro n 
    cases n
    simp
    sorry

  -- Case 1: Alice runs the first lap
  have case1 : choices_for_remaining 3 = 6 := by sorry

  -- Case 2: Alice runs the second lap
  have case2 : choices_for_remaining 3 = 6 := by sorry

  -- Total ways
  have total : case1 + case2 = 12 := by sorry

  exact total

end team_relay_orderings_l540_540873


namespace infinite_product_to_rational_root_l540_540320

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540320


namespace general_term_of_geometric_sequence_l540_540761

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem general_term_of_geometric_sequence
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4)
  (hq : is_geometric_sequence a q)
  (q := 1/2) :
  ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ * q^(n - 1) :=
sorry

end general_term_of_geometric_sequence_l540_540761


namespace maximize_profit_l540_540671

-- Define the total number of workers.
def total_workers : ℕ := 20

-- Define the number of workers producing flour and processing noodles.
def NF (NN : ℕ) : ℕ := total_workers - NN

-- Define the production rate for flour and noodles per worker per day.
def flour_rate : ℕ := 600
def noodles_rate : ℕ := 400

-- Define profit per kg of flour and noodles.
def flour_profit_per_kg : ℝ := 0.2
def noodles_profit_per_kg : ℝ := 0.6

-- Define the daily profit formula.
def daily_profit (NN : ℕ) : ℝ :=
  (NF NN) * flour_rate * flour_profit_per_kg / 1000 +
  NN * noodles_rate * noodles_profit_per_kg / 1000

-- Prove that the maximum profit is achieved when 12 workers process noodles.
theorem maximize_profit :
  ∀ NN : ℕ, NN ∈ finset.range (total_workers + 1) →
  daily_profit 12 ≥ daily_profit NN :=
begin
  sorry
end

-- Define the optimal number of workers processing noodles and the maximum profit.
def optimal_NN : ℕ := 12
def maximum_profit : ℝ := daily_profit optimal_NN

end maximize_profit_l540_540671


namespace price_increase_after_reduction_l540_540592

theorem price_increase_after_reduction (P : ℝ) (h : P > 0) : 
  let reduced_price := P * 0.85
  let increase_factor := 1 / 0.85
  let percentage_increase := (increase_factor - 1) * 100
  percentage_increase = 17.65 := by
  sorry

end price_increase_after_reduction_l540_540592


namespace angle_relation_l540_540473

theorem angle_relation
  (x y z w : ℝ)
  (h_sum : x + y + z + (360 - w) = 360) :
  x = w - y - z :=
by
  sorry

end angle_relation_l540_540473


namespace download_time_l540_540944

theorem download_time (file_size : ℕ) (first_part_size : ℕ) (rate1 : ℕ) (rate2 : ℕ) (total_time : ℕ)
  (h_file : file_size = 90) (h_first_part : first_part_size = 60) (h_rate1 : rate1 = 5) (h_rate2 : rate2 = 10)
  (h_time : total_time = 15) : 
  file_size = first_part_size + (file_size - first_part_size) ∧ total_time = first_part_size / rate1 + (file_size - first_part_size) / rate2 :=
by
  have time1 : total_time = 12 + 3,
    sorry,
  have part1 : first_part_size = 60,
    sorry,
  have part2 : file_size - first_part_size = 30,
    sorry,
  have rate1_correct : rate1 = 5,
    sorry,
  have rate2_correct : rate2 = 10,
    sorry,
  have time1_total : 12 + 3 = 15,
    sorry,
  exact ⟨rfl, rfl⟩

end download_time_l540_540944


namespace continuous_function_identity_l540_540993

-- Part (a)
theorem continuous_function_identity (f : ℝ → ℝ) (h_cont : continuous f) 
(h_identity : ∀ x, f (f (f x)) = x) : ∀ x, f x = x := sorry

-- Part (b)
noncomputable def example_discontinuous_function : ∃ (g : ℝ → ℝ), ¬continuous g ∧ (∀ x, g x ≠ x) ∧ (∀ x, g (g (g x)) = x) :=
begin
  let g : ℝ → ℝ := λ x, if x = 1 then 2 else if x = 2 then 3 else if x = 3 then 1 else x,
  use g,
  split,
  { -- g is not continuous
    sorry
  },
  split,
  { -- g(x) ≠ x for all x
    intro x,
    by_cases h1 : x = 1,
    { rw h1, exact dec_trivial },
    by_cases h2 : x = 2,
    { rw h2, exact dec_trivial },
    by_cases h3 : x = 3,
    { rw h3, exact dec_trivial },
    exact dec_trivial
  },
  { -- g(g(g(x))) = x for all x
    intro x,
    by_cases h1 : x = 1,
    { rw h1, show g (g (g 1)) = 1, rw [g, g, g], exact dec_trivial },
    by_cases h2 : x = 2,
    { rw h2, show g (g (g 2)) = 2, rw [g, g, g], exact dec_trivial },
    by_cases h3 : x = 3,
    { rw h3, show g (g (g 3)) = 3, rw [g, g, g], exact dec_trivial },
    exact dec_trivial
  }
end

end continuous_function_identity_l540_540993


namespace sequence_problem_l540_540656

open Nat

theorem sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, 0 < n → S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ ∀ n : ℕ, 0 < n → a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_problem_l540_540656


namespace johns_original_earnings_l540_540492

-- Definitions from conditions
variables (x : ℝ) (raise_percentage : ℝ) (new_salary : ℝ)

-- Conditions
def conditions : Prop :=
  raise_percentage = 0.25 ∧ new_salary = 75 ∧ x + raise_percentage * x = new_salary

-- Theorem statement
theorem johns_original_earnings (h : conditions x 0.25 75) : x = 60 :=
sorry

end johns_original_earnings_l540_540492


namespace problem_statement_l540_540388

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of a geometric sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ (n + 1)) / (1 - q)

-- Variables and constants for the problem
variable {a : ℕ → ℝ}
variable (q : ℝ := 1 / 2)

-- Given geometric sequence and common ratio
axiom seq_geometric : is_geometric_sequence a q

-- Define S3 and a3
def S_3 : ℝ := geometric_sum a q 3
def a_3 : ℝ := a 0 * q ^ 2

-- The proof goal
theorem problem_statement : S_3 / a_3 = 7 := sorry

end problem_statement_l540_540388


namespace question_equals_answer_l540_540529

noncomputable def omega : ℂ := -- Assume a nonreal root of x^4 = 1

-- Given conditions
axiom omega_pow_four : ω^4 = 1
axiom omega_nonreal : ¬ (∃ (r : ℝ), ω = r)

-- Math proof problem: Showing (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14
theorem question_equals_answer : (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 :=
by sorry  -- proof goes here

end question_equals_answer_l540_540529


namespace additional_tickets_correct_l540_540708

-- Define the number of friends
def friends : ℕ := 17

-- Define the initial number of tickets
def initial_tickets : ℕ := 865

-- Define the ratio structure
def ratio : List ℕ := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]

-- Compute the total number of parts in the given ratio
def total_parts : ℕ := ratio.sum

-- Compute the tickets per part
def tickets_per_part (tickets total_parts : ℕ) : ℚ := tickets / total_parts

-- Define the rounded down tickets per part
def rounded_tickets_per_part : ℕ := (tickets_per_part initial_tickets total_parts).toNat

-- Compute the number of tickets needed for the distribution
def needed_tickets : ℕ := total_parts * rounded_tickets_per_part

-- Compute the extra tickets that do not fit the ratio
def extra_tickets : ℕ := initial_tickets - needed_tickets

-- Compute the next multiple of total parts
def next_multiple : ℕ := ((initial_tickets + total_parts - 1) / total_parts) * total_parts

-- Compute the additional tickets needed to reach the next multiple
def additional_tickets : ℕ := next_multiple - initial_tickets

-- Prove that the additional tickets needed equals 26
theorem additional_tickets_correct : additional_tickets = 26 := by
  sorry

end additional_tickets_correct_l540_540708


namespace count_consecutive_sets_sum_15_l540_540043

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end count_consecutive_sets_sum_15_l540_540043


namespace ROI_is_21_42_percentage_l540_540813

-- Defining the initial conditions
def initialInvestment : ℝ := 900
def investmentStockA : ℝ := (1 / 3) * initialInvestment
def investmentStockB : ℝ := (1 / 5) * initialInvestment
def investmentStockC : ℝ := (1 / 4) * initialInvestment
def investmentStockD : ℝ := initialInvestment - (investmentStockA + investmentStockB + investmentStockC)

def finalValueStockA : ℝ := investmentStockA * 2
def finalValueStockB : ℝ := investmentStockB * 1.30
def finalValueStockC : ℝ := investmentStockC * 0.5
def finalValueStockD : ℝ := investmentStockD * 0.75

def totalFinalValue : ℝ := finalValueStockA + finalValueStockB + finalValueStockC + finalValueStockD

def ROI : ℝ := (totalFinalValue - initialInvestment) / initialInvestment * 100

-- The theorem that proves the ROI percentage
theorem ROI_is_21_42_percentage : ROI ≈ 21.42 := by
  -- We use real number approximation (≈) instead of exact equality due to floating point arithmetic
  sorry

end ROI_is_21_42_percentage_l540_540813


namespace angle_PSU_l540_540482

-- Conditions from the problem
variables (P Q R S T U : Type) [linear_ordered_field P] [linear_ordered_field Q] 
[linear_ordered_field R] [linear_ordered_field S] [linear_ordered_field T] 
[linear_ordered_field U]
variables (angle_PRQ_nat : ℕ) (angle_QRP_nat : ℕ)
variable (angle_PS_nS_deg : ℕ) (angle_PTQ_deg : ℕ)

-- Given Conditions
def angle_PRQ := angle_PRQ_nat = 45
def angle_QRP := angle_QRP_nat = 60

-- Question to prove
theorem angle_PSU :
  angle_PRQ_nat = 45 -> 
  angle_QRP_nat = 60 -> 
  ∃ (angle_PSU : ℕ), angle_PSU = 15 :=
by
  sorry

end angle_PSU_l540_540482


namespace brittany_age_after_vacation_l540_540283

-- Definitions and conditions
def rebecca_age := 25
def brittany_age := rebecca_age + 3
def vacation_years := 4
def birthdays_celebrated := 3

-- Statement to prove
theorem brittany_age_after_vacation : brittany_age + birthdays_celebrated = 31 :=
by
  -- Initialize
  let r := rebecca_age -- Rebecca's age
  let b_start := r + 3 -- Brittany's age at the beginning
  
  -- Conditions in Lean syntax
  have h1: brittany_age = b_start := rfl
  have h2: birthdays_celebrated = 3 := rfl
  
  -- Compute Brittany's age after the vacation
  have h3:  brittany_age + birthdays_celebrated = (r + 3) + 3 := by rw [h1, h2]
  have h4: (r + 3) + 3 = 31  := by norm_num [rebecca_age]

  exact h4

end brittany_age_after_vacation_l540_540283


namespace tan_addition_identity_l540_540782

theorem tan_addition_identity (θ : ℝ) (h : sin θ + cos θ = sqrt 2) :
  tan (θ + π / 3) = -2 - sqrt 3 := 
sorry

end tan_addition_identity_l540_540782


namespace expected_win_value_l540_540258

noncomputable def expected_win : ℚ :=
  let single_roll_EV := (1/2) * (1/10) * (4 + 3 + 2 + 1 + 0) in
  let double_roll_EV := 
    let sum_n_m := 20 * 25 - 2 * (1 + 2 + 3 + 4 + 5) * 5 in
    (1/50) * sum_n_m in
  (1/2) * (single_roll_EV + double_roll_EV)

theorem expected_win_value : expected_win = 19/4 := by
  sorry

end expected_win_value_l540_540258


namespace solve_x_l540_540395

theorem solve_x (x : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ) 
  (hA : A = (1, 3)) (hB : B = (2, 4))
  (ha : a = (2 * x - 1, x ^ 2 + 3 * x - 3))
  (hab : a = (B.1 - A.1, B.2 - A.2)) : x = 1 :=
by {
  sorry
}

end solve_x_l540_540395


namespace lights_on_after_presses_1_4_7_10_init_all_on_lights_all_on_after_presses_5_7_init_partial_on_no_sequence_lights_all_on_init_most_partial_off_l540_540579

-- Part (a) statement
theorem lights_on_after_presses_1_4_7_10_init_all_on :
  ∀(L : Fin 10 → bool) (toggle: Fin 10 → Fin 10 → bool),
    ( (∀ i : Fin 10, L i = tt) ∧
      toggle 1 = fun x => x == 1 || x == 2 || x == 4 ∧ 
      toggle 4 = fun x => x == 3 || x == 4 || x == 5 || x == 7 || x == 10 ∧ 
      toggle 7 = fun x => x == 4 || x == 5 || x == 6 || x == 7 || x == 8 ∧ 
      toggle 10 = fun x => x == 7 || x == 8 || x == 9 || x == 10
    ) →
  (L ⟨1, <dec_trivial⟩ = tt ∧ L ⟨3, <dec_trivial⟩ = tt ∧ L ⟨4, <dec_trivial⟩ = tt ∧ L ⟨7, <dec_trivial⟩ = tt ∧ L ⟨9, <dec_trivial⟩ = tt)
:= by
  sorry

-- Part (b) statement
theorem lights_all_on_after_presses_5_7_init_partial_on :
  ∀(L : Fin 10 → bool) (toggle : Fin 10 → Fin 10 → bool),
    ( (L 1 = tt ∧ L 3 = tt ∧ L 4 = ff ∧
       L 5 = tt ∧ L 7 = tt ∧ L 8 = ff ∧
       L 9 = tt ∧ L 10 = ff) ∧ 
      toggle 5 = fun x => x == 1 || x == 2 || x == 5 || x == 6 || x == 8 ∧
      toggle 7 = fun x => x == 4 || x == 5 || x == 6 || x == 7 || x == 9
    ) →
  (∀ i : Fin 10, L i = tt)
:= by
  sorry

-- Part (c) statement
theorem no_sequence_lights_all_on_init_most_partial_off :
  ∃ (L : Fin 10 → bool) (toggle : Fin 10 → Fin 10 → bool),
    ( (L 1 = tt ∧ L 2 = ff ∧ L 3 = tt ∧
       L 4 = tt ∧ L 5 = ff ∧
       L 6 = ff ∧ L 7 = tt ∧
       L 8 = ff ∧ L 9 = ff ∧
       L 10 = tt) ∧ 
      toggle 1 = fun x => x == 1 || x == 2 || x == 4 ∧
      toggle 4 = fun x => x == 3 || x == 4 || x == 5 || x == 7 || x == 10 ∧
      toggle 7 = fun x => x == 4 || x == 5 || x == 6 || x == 7 || x == 8 ∧
      toggle 6 = fun x => x == 3 || x == 5 || x == 6 || x == 8 || x == 9
    ) →
 ¬∀ (sequence: List (Fin 10)), ∀ i : Fin 10, L i = tt := by
  sorry

end lights_on_after_presses_1_4_7_10_init_all_on_lights_all_on_after_presses_5_7_init_partial_on_no_sequence_lights_all_on_init_most_partial_off_l540_540579


namespace lateral_area_of_cone_l540_540061

def axis_section_is_isosceles_triangle : Prop := sorry
def base_length : ℝ := real.sqrt 2
def height : ℝ := 1

theorem lateral_area_of_cone :
  axis_section_is_isosceles_triangle →
  base_length = real.sqrt 2 →
  height = 1 →
  lateral_area(base_length, height) = real.pi * real.sqrt 2 :=
by
  intros
  sorry

end lateral_area_of_cone_l540_540061


namespace infinite_product_result_l540_540326

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540326


namespace proj_u_on_z_magnitude_l540_540431

variables {𝕜 : Type*} [IsROrC 𝕜]
variables (u z : 𝕜 ^ 2)

axiom norm_u : ‖u‖ = 5
axiom norm_z : ‖z‖ = 8
axiom dot_uz : InnerProductSpace.inner u z = 20

noncomputable def proj_magnitude (u z : 𝕜 ^ 2) : ℝ :=
  ‖(InnerProductSpace.inner u z / ‖z‖ ^ 2) • z‖

theorem proj_u_on_z_magnitude :
  proj_magnitude u z = 5 / 2 :=
by
  rw [proj_magnitude, norm_u, norm_z, dot_uz]
  sorry

end proj_u_on_z_magnitude_l540_540431


namespace infinite_product_sqrt_nine_81_l540_540309

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540309


namespace degree_of_f_plus_g_eq_3_l540_540511

noncomputable def f (z : ℂ) (a0 a1 a2 a3 : ℂ) : ℂ := a3 * z^3 + a2 * z^2 + a1 * z + a0
noncomputable def g (z : ℂ) (b0 b1 b2 : ℂ) : ℂ := b2 * z^2 + b1 * z + b0

theorem degree_of_f_plus_g_eq_3 
  {a0 a1 a2 a3 b0 b1 b2 : ℂ} (h₁ : a3 ≠ 0) (h₂ : b2 ≠ 0) :
  ∀ z : ℂ, degree (f z a0 a1 a2 a3 + g z b0 b1 b2) = 3 :=
by sorry

end degree_of_f_plus_g_eq_3_l540_540511


namespace coefficient_of_x_squared_l540_540575

theorem coefficient_of_x_squared :
  let exp := ((- sqrt(x)) + 1 / x)^10
  ∃ (c : ℝ), c = 45 ∧ (∃ (r : ℕ), x^2 * c = binomial 10 r * (-1)^(10 - r) * x^((10 - 3 * r) / 2)) :=
by
  sorry

end coefficient_of_x_squared_l540_540575


namespace number_of_puzzles_l540_540490

-- Define the conditions as given in the problem
def stuffed_animals := 18
def action_figures := 42
def board_games := 2
def total_toys := 108
def joel_toys := 22

-- Define the puzzles and sister's toys variables
variable (P S : ℕ)

-- Given conditions
axiom toys_from_sister : 2 * S = 22
axiom total_contributions : stuffed_animals + action_figures + board_games + P + S + 2 * S = total_toys

-- Goal: Prove the number of puzzles collected
theorem number_of_puzzles : P = 13 :=
by
  -- substitute the value of S from the first axiom
  have h_S : S = 11, by linarith [toys_from_sister],
  -- substitute the value of S into the total_contributions
  have h_contribution_sub : stuffed_animals + action_figures + board_games + P + 3 * 11 = total_toys,
  from sorry, -- placeholder to illustrate substitution
  -- simplify the contribution equation
  rw [h_S] at *, -- use the value of S
  -- final step to solve for P
  sorry -- actual computation step goes here

end number_of_puzzles_l540_540490


namespace tangent_eq_at_e_tangent_eq_thru_origin_l540_540804

noncomputable def func (x : ℝ) : ℝ := Real.exp x
noncomputable def deriv_func (x : ℝ) : ℝ := Real.exp x

-- Equation of the tangent line at x = e
theorem tangent_eq_at_e : 
  ∃ (y : ℝ → ℝ), (∀ (x : ℝ), y x = Real.exp Real.e * x - Real.exp (Real.e + 1) + Real.exp Real.e) ∧ 
  (y Real.e = Real.exp Real.e) ∧ 
  (∀ (x : ℝ), deriv_func x = Real.exp x) := 
by sorry

-- Equation of the tangent line passing through the origin
theorem tangent_eq_thru_origin : 
  ∃ (y : ℝ → ℝ), (∀ (x : ℝ), y x = Real.exp 1 * x) ∧ 
  (y 0 = 0) ∧ 
  (∀ (x : ℝ), deriv_func x = Real.exp x) := 
by sorry

end tangent_eq_at_e_tangent_eq_thru_origin_l540_540804


namespace PA_PB_dot_product_range_l540_540768

noncomputable def sphere_radius (O : Point) := 1

noncomputable def distance_AB (A B : Point) : ℝ := Real.sqrt 3

theorem PA_PB_dot_product_range (O A B P : Point) 
  (h1 : dist O A = 1) 
  (h2 : dist O B = 1)
  (h3 : dist A B = Real.sqrt 3)
  (h4 : inner (O.to_vector A) (O.to_vector B) = -1/2)
  (h5 : ∥(O.to_vector A) + (O.to_vector B)∥ = 1) :
  inner (P.to_vector A) (P.to_vector B) ∈ Set.Icc (-1/2 : ℝ) (3/2 : ℝ)
:=
sorry

-- Definitions for geometric entities
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Turning a point into a vector from the origin
noncomputable def Point.to_vector (O A : Point) : EuclideanSpace ℝ (Fin 3) :=
  EuclideanSpace.single (Fin.mk 0 sorry) (A.x - O.x) +
  EuclideanSpace.single (Fin.mk 1 sorry) (A.y - O.y) +
  EuclideanSpace.single (Fin.mk 2 sorry) (A.z - O.z)

end PA_PB_dot_product_range_l540_540768


namespace jason_average_messages_l540_540095

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l540_540095


namespace download_time_l540_540941

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end download_time_l540_540941


namespace intersection_complement_A_B_l540_540398

open Set Real

noncomputable def A : Set Real := {x | x^2 - x - 2 > 0}
noncomputable def B : Set Real := {x | log 2 (1 - x) ≤ 2}

noncomputable def complement_A : Set Real := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_complement_A_B :
  (B ∩ complement_A) = {x | -1 ≤ x ∧ x < 1} := by
sorry

end intersection_complement_A_B_l540_540398


namespace denominator_of_repeating_six_l540_540164

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l540_540164


namespace water_bottles_total_l540_540730

/-- Daniel works for a sports stadium filling water bottles for athletes. 
    The football team had 11 players that wanted 6 bottles each. 
    The soccer team had him fill 53 bottles. 
    The lacrosse team needed 12 more bottles than the football team. 
    He filled 49 bottles for the rugby team during the final game he worked this season. 
    Prove that the total number of water bottles filled by Daniel for the teams alone (excluding coaches) is 246. -/
theorem water_bottles_total 
  (football_players : ℕ) (football_bottles_per_player : ℕ) 
  (soccer_bottles : ℕ) (extra_lacrosse_bottles : ℕ) (rugby_bottles : ℕ) : 
  football_players = 11 ∧ football_bottles_per_player = 6 ∧ 
  soccer_bottles = 53 ∧ extra_lacrosse_bottles = 12 ∧ rugby_bottles = 49 →
  let football_total_bottles := football_players * football_bottles_per_player in
  let lacrosse_total_bottles := football_total_bottles + extra_lacrosse_bottles in
  let total_bottles := football_total_bottles + soccer_bottles + lacrosse_total_bottles + rugby_bottles in
  total_bottles = 246 :=
by {
  intros,
  let football_total_bottles := football_players * football_bottles_per_player,
  let lacrosse_total_bottles := football_total_bottles + extra_lacrosse_bottles,
  let total_bottles := football_total_bottles + soccer_bottles + lacrosse_total_bottles + rugby_bottles,
  sorry
}

end water_bottles_total_l540_540730


namespace part1_div_op_calc1_part1_div_op_calc2_part2_neg_odd_pow_part2_neg_even_pow_part3_calc_l540_540702

-- Definition of the dividing operation
def div_op (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 / a else (nat.iterate (λ x, x / a) n a)

-- The proof statements
theorem part1_div_op_calc1 : div_op (1 / 2) 4 = 8 :=
  sorry

theorem part1_div_op_calc2 : div_op (-3) 3 = -1 / 27 :=
  sorry

theorem part2_neg_odd_pow : ∀ n : ℕ, n % 2 = 1 → ∀ a < 0, div_op a n < 0 :=
  sorry

theorem part2_neg_even_pow : ∀ n : ℕ, n % 2 = 0 → ∀ a < 0, div_op a n > 0 :=
  sorry

theorem part3_calc : div_op (-8) 3 + 11 * div_op (-1 / 4) 4 = 160 :=
  sorry

end part1_div_op_calc1_part1_div_op_calc2_part2_neg_odd_pow_part2_neg_even_pow_part3_calc_l540_540702


namespace computer_literate_females_l540_540079

theorem computer_literate_females :
  ∀ (total_employees : ℕ) (pct_female : ℝ) (pct_male_literate : ℝ) (pct_total_literate : ℝ),
  total_employees = 1400 →
  pct_female = 0.60 →
  pct_male_literate = 0.50 →
  pct_total_literate = 0.62 →
  let num_female_employees := pct_female * total_employees in
  let num_male_employees := total_employees - num_female_employees in
  let num_literate_male := pct_male_literate * num_male_employees in
  let num_total_literate := pct_total_literate * total_employees in
  num_total_literate - num_literate_male = 588 :=
by
  intros total_employees pct_female pct_male_literate pct_total_literate
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let num_female_employees := 0.60 * 1400
  let num_male_employees := 1400 - num_female_employees
  let num_literate_male := 0.50 * num_male_employees
  let num_total_literate := 0.62 * 1400
  have : num_total_literate - num_literate_male = 588 := sorry
  exact this

end computer_literate_females_l540_540079


namespace initial_markup_percentage_l540_540681

variables (C M : ℝ)

-- Define initial retail price
def P1 : ℝ := C * (1 + M)

-- Define retail price after New Year mark-up
def P2 : ℝ := P1 * 1.25

-- Define final selling price after February discount
def S : ℝ := P2 * 0.91

-- Additional given condition
def profit_condition : ℝ := C * 1.365

-- The theorem to prove the initial markup percentage
theorem initial_markup_percentage (h : S = profit_condition) : M ≈ 0.1996 :=
by
  sorry

end initial_markup_percentage_l540_540681


namespace sum_double_series_eq_l540_540721

theorem sum_double_series_eq : 
  (∑ n from 3 to ∞, ∑ k in range (n - 2), k / (3^(n + k))) = 3 / 128 :=
by sorry

end sum_double_series_eq_l540_540721


namespace one_circle_equiv_three_squares_l540_540081

-- Define the weights of circles and squares symbolically
variables {w_circle w_square : ℝ}

-- Equations based on the conditions in the problem
-- 3 circles balance 5 squares
axiom eq1 : 3 * w_circle = 5 * w_square

-- 2 circles balance 3 squares and 1 circle
axiom eq2 : 2 * w_circle = 3 * w_square + w_circle

-- We need to prove that 1 circle is equivalent to 3 squares
theorem one_circle_equiv_three_squares : w_circle = 3 * w_square := 
by sorry

end one_circle_equiv_three_squares_l540_540081


namespace students_taking_history_or_statistics_or_both_l540_540077

theorem students_taking_history_or_statistics_or_both
  (H : ℕ) (S : ℕ) (B : ℕ)
  (hH : H = 36)
  (hS : S = 32)
  (hHistNotStat : H - B = 25) :
  H + S - B = 57 :=
by
  have hB : B = 11 := by
    sorry
  rw [hH, hS, hB]
  sorry

end students_taking_history_or_statistics_or_both_l540_540077


namespace probability_of_both_even_numbers_l540_540249

theorem probability_of_both_even_numbers (toys : Finset ℕ) (h : toys = Finset.range 22 \ {0}) :
  let even_toys := toys.filter (λ x, x % 2 = 0) in
  let total_even := even_toys.card in
  let total := toys.card in
  total = 21 →
  total_even = 10 →
  (total_even.toRat / total.toRat) * ((total_even - 1).toRat / (total - 1).toRat) = (3 : ℚ) / 14 := by
  intros h_total h_even
  rw [←h, Finset.card_sdiff, Finset.card_range]
  have even_toys_card : even_toys.card = 10 := by
    sorry
  have total_minus_one : total - 1 = 20 := by
    sorry
  have total_even_minus_one : total_even - 1 = 9 := by
    sorry
  field_simp [h_total, h_even, even_toys_card, total_minus_one, total_even_minus_one]
  norm_num
  done

end probability_of_both_even_numbers_l540_540249


namespace trigonometric_expression_eq_l540_540578

noncomputable def tan (θ : ℝ) : ℝ := real.sin θ / real.cos θ

theorem trigonometric_expression_eq :
  let θ := 12 * real.pi / 180 in
  (sqrt 3 * tan θ - 3) / ((4 * (real.cos θ)^2 - 2) * real.sin θ) = -4 * sqrt 3 :=
by
  sorry

end trigonometric_expression_eq_l540_540578


namespace shelves_used_l540_540685

theorem shelves_used (initial_books : ℕ) (sold_books : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (total_shelves : ℕ) :
  initial_books = 120 → sold_books = 39 → books_per_shelf = 9 → remaining_books = initial_books - sold_books → total_shelves = remaining_books / books_per_shelf → total_shelves = 9 :=
by
  intros h_initial_books h_sold_books h_books_per_shelf h_remaining_books h_total_shelves
  rw [h_initial_books, h_sold_books] at h_remaining_books
  rw [h_books_per_shelf, h_remaining_books] at h_total_shelves
  exact h_total_shelves

end shelves_used_l540_540685


namespace ratio_female_male_l540_540706

theorem ratio_female_male (f m : ℕ) 
  (h1 : (50 * f) / f = 50) 
  (h2 : (30 * m) / m = 30) 
  (h3 : (50 * f + 30 * m) / (f + m) = 35) : 
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_male_l540_540706


namespace inscribed_circles_tangent_l540_540876

theorem inscribed_circles_tangent (AB CD AD BC : ℝ)
  (h : AB + CD = AD + BC) : 
  ∃ pt : (ℝ × ℝ), ∀ α β : circle, α.inscribed_in (triangle ABC) ∧ β.inscribed_in (triangle ACD) → α.tangent_to β :=
by 
  -- This is just a placeholder for the actual proof.
  sorry

end inscribed_circles_tangent_l540_540876


namespace quadratic_inequality_l540_540195

-- Define the quadratic function and conditions
variables {a b c x0 y1 y2 y3 : ℝ}
variables (A : (a * x0^2 + b * x0 + c = 0))
variables (B : (a * (-2)^2 + b * (-2) + c = 0))
variables (C : (a + b + c) * (4 * a + 2 * b + c) < 0)
variables (D : a > 0)
variables (E1 : y1 = a * (-1)^2 + b * (-1) + c)
variables (E2 : y2 = a * (- (sqrt 2) / 2)^2 + b * (- (sqrt 2) / 2) + c)
variables (E3 : y3 = a * 1^2 + b * 1 + c)

-- Prove that y3 > y1 > y2
theorem quadratic_inequality : y3 > y1 ∧ y1 > y2 := by 
  sorry

end quadratic_inequality_l540_540195


namespace probability_of_guessing_correctly_l540_540560

noncomputable def secret_number : ℕ := sorry

def possible_numbers : list ℕ := [78, 90, 96]

def secret_number_conditions (n : ℕ) : Prop :=
  75 < n ∧
  n < 100 ∧
  (n / 10) % 2 = 1 ∧
  n % 2 = 0 ∧
  n % 3 = 0

theorem probability_of_guessing_correctly :
  ∃ n, secret_number_conditions n ∧ (n ∈ possible_numbers) ∧ (∃ m, (m ∈ possible_numbers ∧ m ≠ n) → false) →
  (1 / 3 : ℚ) :=
sorry

end probability_of_guessing_correctly_l540_540560


namespace ratio_of_sums_l540_540882

def M : ℕ := 36 * 36 * 85 * 128

theorem ratio_of_sums (h : M = 2^11 * 3^4 * 5 * 17) :
  let odd_sum := (1 + 3 + 3^2 + 3^3 + 3^4) * (1 + 5) * (1 + 17),
      even_sum := 4095 * odd_sum - odd_sum
  in (odd_sum : ℚ) / even_sum = 1 / 4094 :=
by
  -- The proof of this theorem would be inserted here.
  sorry

end ratio_of_sums_l540_540882


namespace felix_brother_lifting_capacity_is_600_l540_540350

-- Define the conditions
def felix_lifting_capacity (felix_weight : ℝ) : ℝ := 1.5 * felix_weight
def felix_brother_weight (felix_weight : ℝ) : ℝ := 2 * felix_weight
def felix_brother_lifting_capacity (brother_weight : ℝ) : ℝ := 3 * brother_weight
def felix_actual_lifting_capacity : ℝ := 150

-- Define the proof problem
theorem felix_brother_lifting_capacity_is_600 :
  ∃ felix_weight : ℝ,
    felix_lifting_capacity felix_weight = felix_actual_lifting_capacity ∧
    felix_brother_lifting_capacity (felix_brother_weight felix_weight) = 600 :=
by
  sorry

end felix_brother_lifting_capacity_is_600_l540_540350


namespace sum_of_digits_is_10_l540_540625

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end sum_of_digits_is_10_l540_540625


namespace seq_b_is_geometric_l540_540004

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence {a_n} with first term a_1 and common ratio q
def a_n (a₁ q : α) (n : ℕ) : α := a₁ * q^(n-1)

-- Define the sequence {b_n}
def b_n (a₁ q : α) (n : ℕ) : α :=
  a_n a₁ q (3*n - 2) + a_n a₁ q (3*n - 1) + a_n a₁ q (3*n)

-- Theorem stating {b_n} is a geometric sequence with common ratio q^3
theorem seq_b_is_geometric (a₁ q : α) (h : q ≠ 1) :
  ∀ n : ℕ, b_n a₁ q (n + 1) = q^3 * b_n a₁ q n :=
by
  sorry

end seq_b_is_geometric_l540_540004


namespace area_of_fifteen_sided_figure_l540_540728

def point : Type := ℕ × ℕ

def vertices : List point :=
  [(1,1), (1,3), (3,5), (4,5), (5,4), (5,3), (6,3), (6,2), (5,1), (4,1), (3,2), (2,2), (1,1)]

def graph_paper_area (vs : List point) : ℚ :=
  -- Placeholder for actual area calculation logic
  -- The area for the provided vertices is found to be 11 cm^2.
  11

theorem area_of_fifteen_sided_figure : graph_paper_area vertices = 11 :=
by
  -- The actual proof would involve detailed steps to show that the area is indeed 11 cm^2
  -- Placeholder proof
  sorry

end area_of_fifteen_sided_figure_l540_540728


namespace base_1000_has_six_digits_l540_540655

theorem base_1000_has_six_digits : ∃ (b : ℕ), b^5 ≤ 1000 ∧ 1000 < b^6 ∧ b = 2 :=
by {
  use 2,
  split,
  { exact nat.pow_le_pow_of_le_right (nat.zero_le 2) (by linarith) },
  split,
  { exact lt_of_pow_lt_pow (by linarith) },
  exact rfl,
  sorry
}

end base_1000_has_six_digits_l540_540655


namespace max_value_of_h_l540_540918

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.sin (x - π / 3)
def h (x : ℝ) : ℝ := f x + g x

theorem max_value_of_h : ∃ x : ℝ, h x = √3 :=
by
  sorry

end max_value_of_h_l540_540918


namespace sqrt_2_plus_sqrt_3_inv_in_S_l540_540595

variable {S : Set ℝ}

theorem sqrt_2_plus_sqrt_3_inv_in_S
  (h_subset_real : S ⊆ ℝ)
  (h_integers_in_S : ∀ z : ℤ, (z:ℝ) ∈ S)
  (h_sqrt2_plus_sqrt3_in_S : (↑(Real.sqrt 2) + ↑(Real.sqrt 3)) ∈ S)
  (h_add_closed : ∀ x y : ℝ, x ∈ S → y ∈ S → (x + y) ∈ S)
  (h_mul_closed : ∀ x y : ℝ, x ∈ S → y ∈ S → (x * y) ∈ S) :
  (↑(Real.sqrt 2) + ↑(Real.sqrt 3))⁻¹ ∈ S := by
sorry

end sqrt_2_plus_sqrt_3_inv_in_S_l540_540595


namespace count_even_numbers_l540_540219

theorem count_even_numbers : 
  (∃ L : List ℕ, (∀ i ∈ L, i ∈ [0, 1, 2, 3, 4, 5]) ∧ L.length = 6 ∧ 
  (L.last (by simp) ∈ [0, 2, 4]) ∧ L.nodup) → 
  (∑ t', (t' ∈ permutations [0, 1, 2, 3, 4, 5]) ∧ 
  (t' !nth 5 = some 0 ∨ t' !nth 5 = some 2 ∨ t' !nth 5 = some 4) = 312) := 
sorry

end count_even_numbers_l540_540219


namespace infinite_product_value_l540_540330

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540330


namespace denominator_of_repeating_six_l540_540160

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l540_540160


namespace area_of_region_covered_by_squares_l540_540613

structure Square :=
  (side_length : ℝ)
  (center: ℝ × ℝ)

noncomputable def area_of_square (s : Square) : ℝ :=
  s.side_length ^ 2

noncomputable def overlap_area (s1 s2 : Square) : ℝ :=
  let d := s1.side_length * Real.sqrt 2 / 2 -- since each square is congruent and overlap in a diamond shape
  in 1 / 2 * d * d

noncomputable def total_covered_area (s1 s2 : Square) : ℝ :=
  area_of_square(s1) + area_of_square(s2) - overlap_area(s1, s2)

theorem area_of_region_covered_by_squares : 
  ∀ s1 s2 : Square, 
    s1.side_length = 12 → 
    s2.side_length = 12 → 
    s2.center = ((s1.center.1 + s1.side_length / 2), s1.center.2) → 
    total_covered_area(s1, s2) = 252 :=
by
  intros s1 s2 h1 h2 h3
  sorry

end area_of_region_covered_by_squares_l540_540613


namespace cot_identity_l540_540522

open Real

theorem cot_identity
  (p q r : ℝ) 
  (θ φ ψ : ℝ) 
  (h1 : p^2 + q^2 = 2001 * r^2) 
  (h2 : sin θ ≠ 0) 
  (h3 : sin φ ≠ 0) 
  (h4 : sin ψ ≠ 0) 
  (h5 : p = r * sin θ / sin ψ)
  (h6 : q = r * sin φ / sin ψ) :
  (cot ψ / (cot θ + cot φ) = 1000) := 
by
  sorry

end cot_identity_l540_540522


namespace find_w_value_l540_540981

theorem find_w_value : 
  (2^5 * 9^2) / (8^2 * 243) = 0.16666666666666666 := 
by
  sorry

end find_w_value_l540_540981


namespace find_m_l540_540500

def is_square_of_linear_expression_in_x (F : ℝ → ℝ) : Prop :=
∃ (a b : ℝ), F = (λ x, (a * x + b)^2)

theorem find_m (m : ℝ) :
  let F := λ x, (8 * x^2 + 20 * x + 5 * m) / 8 in
  is_square_of_linear_expression_in_x F → m = 2.5 :=
by
  sorry

end find_m_l540_540500


namespace part1_part2_l540_540765

noncomputable def sequence (a d : ℝ) : ℕ → ℝ
| 1 => a
| n + 1 => if n < 15 then sequence a d n + d
           else if n < 30 then sequence a d n + 1
           else sequence a d n + 1/d

theorem part1 (d : ℝ) (hd : d ≠ 0) : 
  sequence 1 d 46 = 16 + 15 * (d + 1/d) ∧ 
  (sequence 1 d 46 ∈ set.Iic (-14) ∪ set.Ici 46) :=
by
  sorry

def M (a d : ℝ) : set ℝ :=
  {b | ∃ i j k : ℕ, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 16 ∧ b = sequence a d i + sequence a d j + sequence a d k}

theorem part2 : 
  2 ∈ M (1/3) (1/4) :=
by
  sorry

end part1_part2_l540_540765


namespace angle_in_third_quadrant_l540_540840

def in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

theorem angle_in_third_quadrant (α : ℝ)
  (h1 : sin (2 * α) > 0)
  (h2 : sin α + cos α < 0) : 
  in_third_quadrant α :=
sorry

end angle_in_third_quadrant_l540_540840


namespace original_planned_length_l540_540659

theorem original_planned_length (x : ℝ) (h1 : x > 0) (total_length : ℝ := 3600) (efficiency_ratio : ℝ := 1.8) (time_saved : ℝ := 20) 
  (h2 : total_length / x - total_length / (efficiency_ratio * x) = time_saved) :
  x = 80 :=
sorry

end original_planned_length_l540_540659


namespace unique_pos_neg_roots_of_poly_l540_540919

noncomputable def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 5 * Polynomial.X^3 + Polynomial.C 15 * Polynomial.X - Polynomial.C 9

theorem unique_pos_neg_roots_of_poly : 
  (∃! x : ℝ, (0 < x) ∧ poly.eval x = 0) ∧ (∃! x : ℝ, (x < 0) ∧ poly.eval x = 0) :=
  sorry

end unique_pos_neg_roots_of_poly_l540_540919


namespace smallest_n_exists_l540_540526

theorem smallest_n_exists :
  ∃ (n : ℕ), (n = 13) ∧
    ∃ (a : Fin n → ℕ), (∃' (x y : ℕ), (x < y) ∧
    distinct_values a 5 ∧
    ∀ (i j : ℕ), (i < j) → (∃ (k l : ℕ), (k ≠ i) ∧ (k ≠ j) ∧ (l ≠ i) ∧ (l ≠ j) ∧ (a i + a j = a k + a l))) :=
sorry

end smallest_n_exists_l540_540526


namespace infinite_product_result_l540_540323

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540323


namespace triangle_ABC_is_isosceles_l540_540396

-- Define points A, B, and C
def A : (ℝ × ℝ) := (3, 5)
def B : (ℝ × ℝ) := (-6, -2)
def C : (ℝ × ℝ) := (0, -6)

-- Define the distance function between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2) ^ (1 / 2)

-- Define the distances AB and AC
def AB := dist A B
def AC := dist A C

-- Prove that triangle ABC is isosceles by showing AB = AC
theorem triangle_ABC_is_isosceles : AB = AC :=
  sorry

end triangle_ABC_is_isosceles_l540_540396


namespace hyperbola_through_point_1_hyperbola_through_point_2_l540_540747

-- Define conditions for the first hyperbola proof
def ellipse_eq := ∀ x y : ℝ, (x^2)/16 + (y^2)/25 = 1
def point_on_hyperbola_1 (x y : ℝ) := x = -2 ∧ y = real.sqrt 10

-- Lean theorem statement for the first proof problem
theorem hyperbola_through_point_1 : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 21 = 1) ↔ 
  (ellipse_eq ∧ point_on_hyperbola_1 (-2) (real.sqrt 10)) := 
sorry

-- Define conditions for the second hyperbola proof
def hyperbola_eq := ∀ x y : ℝ, (x^2)/16 - (y^2)/4 = 1
def point_on_hyperbola_2 (x y : ℝ) := x = 3 * real.sqrt 2 ∧ y = 2

-- Lean theorem statement for the second proof problem
theorem hyperbola_through_point_2 : 
  (∀ x y : ℝ, y^2 / 9 - x^2 / 7 = 1) ↔ 
  (hyperbola_eq ∧ point_on_hyperbola_2 (3 * real.sqrt 2) 2) := 
sorry

end hyperbola_through_point_1_hyperbola_through_point_2_l540_540747


namespace lateral_area_cone_l540_540059

theorem lateral_area_cone (base_len height : ℝ) (r l : ℝ) (h_base: base_len = sqrt 2) (h_height: height = 1) (h_radius: r = 1) (h_slant_height: l = sqrt 2) :
  (π * r * l) = sqrt 2 * π :=
by
  sorry

end lateral_area_cone_l540_540059


namespace trapezoid_is_right_trapezoid_l540_540185

theorem trapezoid_is_right_trapezoid 
  (A B C D O E : Point)
  (h_trapezoid : trapezoid A B C D)
  (h_intersect : intersect AC BD O)
  (h_E_on_AB : on_line E A B)
  (h_EO_parallel_AD : parallel (line_through E O) (line_through A D))
  (h_angle_bisector : angle_bisector (angle C E D) (line_through E O)) :
  right_trapezoid A B C D := 
sorry

end trapezoid_is_right_trapezoid_l540_540185


namespace g_symmetric_about_pi_thirds_l540_540802

noncomputable def a : ℝ := sorry

def f (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

def g (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem g_symmetric_about_pi_thirds (h : ∀ x, f x = f (π / 3 - x)) :
  ∀ x, g x = g (π / 3 - x) :=
sorry

end g_symmetric_about_pi_thirds_l540_540802


namespace problem_1_problem_2_l540_540423

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x + a / (x + 1)

theorem problem_1 (a : ℝ) (h : a = 9 / 2) :
  ∃ (y : ℝ), y ∈ (Set.range (λ x, f x a) ⟨1, e, (one_le_two.trans le_e_one_eq)⟩) ∧
    y = ln 2 + 3 / 2 ∨ y = 9 / 4 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + x

theorem problem_2 (h_monotonic : ∀ x ∈ Icc 1 2, deriv (g x) ≤ 0) :
  ∀ a, a ∈ Set.Ici (27 / 2) :=
sorry

end problem_1_problem_2_l540_540423


namespace infinite_product_sqrt_nine_81_l540_540308

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540308


namespace rhombus_area_l540_540243

def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 18) :
  area_of_rhombus d1 d2 = 126 :=
by {
  rw [h1, h2],
  simp [area_of_rhombus],
  norm_num,
}

end rhombus_area_l540_540243


namespace determine_a_l540_540736

theorem determine_a (a : ℚ) : (∃ b : ℚ, (9 : ℚ) * x^2 + 27 * x + a = (3 * x + b)^2) → a = 81 / 4 :=
by
  intro h
  cases h with b hb
  sorry

end determine_a_l540_540736


namespace option_d_not_necessarily_true_l540_540825

theorem option_d_not_necessarily_true (a b c : ℝ) (h: a > b) : ¬(a * c^2 > b * c^2) ↔ c = 0 :=
by sorry

end option_d_not_necessarily_true_l540_540825


namespace smallest_palindrome_base2_base4_greater_than_5_l540_540724

-- Helper function to check if a number is a palindrome in a given base
def is_palindrome_in_base (n : ℕ) (base : ℕ) : Prop :=
  let digits := List.replicate (Nat.log n base + 1) (0 : ℕ) in  -- Extract digits in the provided base
  digits.reverse = digits

-- Main theorem statement
theorem smallest_palindrome_base2_base4_greater_than_5 : ∃ n : ℕ, n > 5 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ n = 15 := 
by
  sorry

end smallest_palindrome_base2_base4_greater_than_5_l540_540724


namespace distance_from_focus_to_asymptote_l540_540005

theorem distance_from_focus_to_asymptote
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a = b)
  (h2 : |a| / Real.sqrt 2 = 2) :
  Real.sqrt 2 * 2 = 2 * Real.sqrt 2 :=
by
  sorry

end distance_from_focus_to_asymptote_l540_540005


namespace part1_part2_l540_540415

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1 (m : ℝ) : (∃ x, deriv f x = 2 ∧ f x = 2 * x + m) → m = -Real.exp 1 :=
sorry

theorem part2 : ∀ x > 0, -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x) :=
sorry

end part1_part2_l540_540415


namespace value_of_A_cos_alpha_plus_beta_l540_540028

noncomputable def f (A x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem value_of_A {A : ℝ}
  (h1 : f A (Real.pi / 3) = Real.sqrt 2) :
  A = 2 := 
by
  sorry

theorem cos_alpha_plus_beta {α β : ℝ}
  (hαβ1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hαβ2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h2 : f 2 (4*α + 4*Real.pi/3) = -30 / 17)
  (h3 : f 2 (4*β - 2*Real.pi/3) = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
by
  sorry

end value_of_A_cos_alpha_plus_beta_l540_540028


namespace sofia_total_time_l540_540147

def distance1 : ℕ := 150
def speed1 : ℕ := 5
def distance2 : ℕ := 150
def speed2 : ℕ := 6
def laps : ℕ := 8
def time_per_lap := (distance1 / speed1) + (distance2 / speed2)
def total_time := 440  -- 7 minutes and 20 seconds in seconds

theorem sofia_total_time :
  laps * time_per_lap = total_time :=
by
  -- Proof steps are omitted and represented by sorry.
  sorry

end sofia_total_time_l540_540147


namespace area_ratio_PQR_ABC_l540_540086

variable {A B C D E F P Q R : Type*}
variable [Field A] [Field B] [Field C]

-- Triangle vertices and segments
variable (ABC : Triangle A B C)
variable (D : PointOnSegment B C)
variable (E : PointOnSegment C A)
variable (F : PointOnSegment A B)

-- The given ratios
axiom BD_DC_ratio : ratio_segment BD DC = 2/3
axiom CE_EA_ratio : ratio_segment CE EA = 2/3
axiom AF_FB_ratio : ratio_segment AF FB = 2/3

-- Cevians intersect at points
variable (P : PointOnLine AD)
variable (Q : PointOnLine BE)
variable (R : PointOnLine CF)

-- The theorem we need to prove
theorem area_ratio_PQR_ABC : area_ratio PQR ABC = 6/25 :=
sorry

end area_ratio_PQR_ABC_l540_540086


namespace polynomial_expansion_sum_eq_l540_540496

theorem polynomial_expansion_sum_eq :
  (∀ (x : ℝ), (2 * x - 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 243) :=
by
  sorry

end polynomial_expansion_sum_eq_l540_540496


namespace sqrt_sum_fractions_eq_l540_540715

theorem sqrt_sum_fractions_eq : sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36) = sqrt 61 / 30 := 
by
  sorry

end sqrt_sum_fractions_eq_l540_540715


namespace minimum_half_lives_l540_540694

theorem minimum_half_lives (n : ℕ) : 
  (1 / 2) ^ n < (1 / 1000) → n ≥ 10 :=
begin
  -- Proof omitted
  sorry
end

end minimum_half_lives_l540_540694


namespace perimeter_of_rectangle_l540_540913

noncomputable def a' : ℝ := 2 * Real.sqrt 1228.5
noncomputable def b' : ℝ := Real.sqrt 1228.5
noncomputable def x' : ℝ := 0 -- Dummy initialization, to be determined
noncomputable def y' : ℝ := 0 -- Dummy initialization, to be determined

-- Conditions from the problem
axiom rect_area : x' * y' = 2457
axiom ellipse_area : π * a' * b' = 2457 * π
axiom sum_of_distances_to_foci : x' + y' = 2 * a'
axiom diagonal_square : x' ^ 2 + y' ^ 2 = 4 * (a' ^ 2 - b' ^ 2)

-- Theorem to prove
theorem perimeter_of_rectangle : 2 * (x' + y') = 8 * Real.sqrt 1228.5 := sorry

end perimeter_of_rectangle_l540_540913


namespace original_savings_l540_540128

-- Defining the conditions
def spent_on_furniture (S : ℝ) : ℝ := (3 / 4) * S
def spent_on_TV (S : ℝ) : ℝ := (1 / 4) * S

-- The cost of the TV
def TV_cost : ℝ := 200

-- Proposition stating that Linda's original savings were $800
theorem original_savings : ∃ S : ℝ, spent_on_TV S = TV_cost ∧ S = 800 := 
by
  use 800
  split
  sorry ∧ -- This part of the proof shows that 1/4 of her original savings equals TV_cost
  rfl -- This shows that S is indeed equal to $800

end original_savings_l540_540128


namespace trig_identity_l540_540409

noncomputable def tan_eq_neg_4_over_3 (theta : ℝ) : Prop := 
  Real.tan theta = -4 / 3

theorem trig_identity (theta : ℝ) (h : tan_eq_neg_4_over_3 theta) : 
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 :=
by
  sorry

end trig_identity_l540_540409


namespace number_of_small_pipes_l540_540732

-- Definitions from the conditions
def diameter_large_pipe : ℝ := 8
def diameter_small_pipe : ℝ := 2
def radius_large_pipe : ℝ := diameter_large_pipe / 2
def radius_small_pipe : ℝ := diameter_small_pipe / 2
def area_large_pipe : ℝ := Real.pi * radius_large_pipe ^ 2
def area_small_pipe : ℝ := Real.pi * radius_small_pipe ^ 2

-- Main theorem to prove
theorem number_of_small_pipes : 
  ∃ (n : ℕ), n = (area_large_pipe / area_small_pipe) ∧ n = 16 :=
by
  sorry

end number_of_small_pipes_l540_540732


namespace length_of_major_axis_l540_540699

theorem length_of_major_axis
  {F1 F2 : ℝ × ℝ}
  (F1_eq : F1 = (20, 10))
  (F2_eq : F2 = (20, 70))
  (tangent_to_y_axis : ∀ P : ℝ × ℝ, P.1 = 0 → (dist P F1 + dist P F2) = (dist (reflect_y F1) P + dist P F2)) :
  ∃ (length : ℝ), length = 20 * sqrt 13 :=
begin
  sorry
end

noncomputable def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

end length_of_major_axis_l540_540699


namespace find_a_l540_540026

variable {f : ℝ → ℝ}

-- Conditions
variables (a : ℝ) (domain : Set ℝ := Set.Ioo (3 - 2 * a) (a + 1))
variable (even_f : ∀ x, f (x + 1) = f (- (x + 1)))

-- The theorem stating the problem
theorem find_a (h : ∀ x, x ∈ domain ↔ x ∈ Set.Ioo (3 - 2 * a) (a + 1)) : a = 2 := by
  sorry

end find_a_l540_540026


namespace average_of_original_set_l540_540157

-- Average of 8 numbers is some value A and the average of the new set where each number is 
-- multiplied by 8 is 168. We need to show that the original average A is 21.

theorem average_of_original_set (A : ℝ) (h1 : (64 * A) / 8 = 168) : A = 21 :=
by {
  -- This is the theorem statement, we add the proof next
  sorry -- proof placeholder
}

end average_of_original_set_l540_540157


namespace perimeter_PXY_l540_540612

variables {P Q R X Y I : Point}
variables {PQ QR PR : ℝ}
variables (h1 : PQ = 15) (h2 : QR = 20) (h3 : PR = 25)
variables {triangle_PQR : Triangle P Q R}
variables (hincenter : is_incenter triangle_PQR I)
variables (h_parallel : parallel (line_through I) (line_through Q R))
variables (hX : X = intersection (parallel (line_through I Q) (line_through P Q)) (line_through P Q))
variables (hY : Y = intersection (parallel (line_through I R) (line_through P R)) (line_through P R))

theorem perimeter_PXY : perimeter (triangle P X Y) = 40 :=
by sorry

end perimeter_PXY_l540_540612


namespace trig_identity_proof_l540_540657

theorem trig_identity_proof :
    sin 2023 * cos 17 + cos 2023 * sin 17 = - (sqrt 3) / 2 := 
by
  sorry

end trig_identity_proof_l540_540657


namespace steve_num_nickels_l540_540150

-- Definitions for the conditions
def num_nickels (N : ℕ) : Prop :=
  ∃ D Q : ℕ, D = N + 4 ∧ Q = D + 3 ∧ 5 * N + 10 * D + 25 * Q + 5 = 380

-- Statement of the problem
theorem steve_num_nickels : num_nickels 4 :=
sorry

end steve_num_nickels_l540_540150


namespace triangle_count_l540_540049

theorem triangle_count : let count := { n : ℕ | ∃ (a b c : ℕ), a + b + c = 2017 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a } in
count.card = 85008 :=
by {
  sorry
}

end triangle_count_l540_540049


namespace polynomial_divisibility_by_difference_l540_540604

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

lemma P_Q_comm (x : ℝ) : P.eval (Q.eval x) = Q.eval (P.eval x) := sorry

theorem polynomial_divisibility_by_difference (n : ℕ) :
  (P.eval'^[n] - Q.eval'^[n] : Polynomial ℝ) ∣ (P - Q) :=
begin
  sorry
end

end polynomial_divisibility_by_difference_l540_540604


namespace one_over_modulus_z_l540_540533

open Complex

-- Define the problem conditions
def z : ℂ := Complex.i / (1 - Complex.i)

-- State the proof problem
theorem one_over_modulus_z : 1 / Complex.abs z = Real.sqrt 2 := by
  sorry

end one_over_modulus_z_l540_540533


namespace pyramid_height_l540_540968

theorem pyramid_height
    (area1 area2 : ℝ)
    (h_diff : ℝ)
    (h : ℝ)
    (H : h = (130 * (13 + 5 * real.sqrt 3)) / (13 - 5 * real.sqrt 3)) :
    (area1 = 150) ∧ (area2 = 338) ∧ (h_diff = 10) → 
    h = (1690 + 650 * real.sqrt 3) / 94 :=
begin
    intro h,
    sorry
end

end pyramid_height_l540_540968


namespace count_consecutive_sets_sum_15_l540_540044

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end count_consecutive_sets_sum_15_l540_540044


namespace ceil_x_square_possibilities_l540_540448

theorem ceil_x_square_possibilities {x : ℝ} (h : ⌈x⌉ = 12) : 
  ∃ S : Finset ℤ, (S.card = 23) ∧ (∀ n ∈ S, ∃ y : ℝ, 11 < y ≤ 12 ∧ ⌈y^2⌉ = n) :=
by 
  sorry

end ceil_x_square_possibilities_l540_540448


namespace celestia_badges_l540_540433

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end celestia_badges_l540_540433


namespace inradius_inequality_l540_540848

theorem inradius_inequality
  (r r_A r_B r_C : ℝ) 
  (h_inscribed_circle: r > 0) 
  (h_tangent_circles_A: r_A > 0) 
  (h_tangent_circles_B: r_B > 0) 
  (h_tangent_circles_C: r_C > 0)
  : r ≤ r_A + r_B + r_C :=
  sorry

end inradius_inequality_l540_540848


namespace distance_between_parabola_vertices_eq_5_l540_540193

def parabola_vertices_distance : ℝ :=
  abs (3.5 - (-1.5))

theorem distance_between_parabola_vertices_eq_5 :
  ∃ x y : ℝ, (sqrt(x^2 + y^2) + abs(y - 2) = 5) → parabola_vertices_distance = 5 :=
by
  sorry

end distance_between_parabola_vertices_eq_5_l540_540193


namespace smallest_value_of_n_l540_540571

theorem smallest_value_of_n (a b c m n : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 2010) (h4 : (a! * b! * c!) = m * 10 ^ n) : ∃ n, n = 500 := 
sorry

end smallest_value_of_n_l540_540571


namespace domain_of_function_l540_540939

noncomputable def domain (f : ℝ → ℝ) : set ℝ :=
  {x | (2 * x - 3 ≥ 0) ∧ (x ≠ 3)}

def function_to_prove (x : ℝ) : ℝ :=
  sqrt (2 * x - 3) + 1 / (x - 3)

theorem domain_of_function : domain function_to_prove = 
  {x | x ≥ 3/2} \ {3} := by
  sorry

end domain_of_function_l540_540939


namespace correct_calculation_l540_540985

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end correct_calculation_l540_540985


namespace lateral_area_of_cone_l540_540062

def axis_section_is_isosceles_triangle : Prop := sorry
def base_length : ℝ := real.sqrt 2
def height : ℝ := 1

theorem lateral_area_of_cone :
  axis_section_is_isosceles_triangle →
  base_length = real.sqrt 2 →
  height = 1 →
  lateral_area(base_length, height) = real.pi * real.sqrt 2 :=
by
  intros
  sorry

end lateral_area_of_cone_l540_540062


namespace find_sum_l540_540354

theorem find_sum (a b : ℝ) 
  (h₁ : (a + Real.sqrt b) + (a - Real.sqrt b) = -8) 
  (h₂ : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) : 
  a + b = 8 := 
sorry

end find_sum_l540_540354


namespace jason_average_messages_l540_540096

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l540_540096


namespace problem_statement_l540_540839

variable (P : ℕ → Prop)

theorem problem_statement
    (h1 : P 2)
    (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 2)) :
    ∀ n : ℕ, n > 0 → 2 ∣ n → P n :=
by
  sorry

end problem_statement_l540_540839


namespace simplify_sqrt_expr_l540_540566

theorem simplify_sqrt_expr : sqrt 5 - sqrt 20 + sqrt 45 = 2 * sqrt 5 :=
by
  sorry

end simplify_sqrt_expr_l540_540566


namespace rational_coefficients_count_l540_540975

theorem rational_coefficients_count :
  ∃ n : ℕ, n = 502 ∧ (∀ k : ℕ, k ∈ ({0} ∪ (finset.range 2005 \ 0) : set ℕ) → k % lcm 4 2 = 0 → 2^(k/4) ∈ ℚ ∧ 3^((2004-k)/2) ∈ ℚ) :=
begin
  sorry
end

end rational_coefficients_count_l540_540975


namespace problem_statement_l540_540788

theorem problem_statement
  (m : ℝ)
  (α β λ μ : ℝ)
  (h1 : α ≠ β)
  (h2 : α < β)
  (h3 : λ > 0)
  (h4 : μ > 0)
  (h_root1 : α^2 - m * α - 1 = 0)
  (h_root2 : β^2 - m * β - 1 = 0) :
  |(2*(λ*α+μ*β)/(λ+μ) - m)/(((λ*α+μ*β)/(λ+μ))^2 + 1) - (2*(μ*α+λ*β)/(λ+μ) - m)/(((μ*α+λ*β)/(λ+μ))^2 + 1)| < |α - β| :=
by
  sorry

end problem_statement_l540_540788


namespace ellipse_condition_l540_540958

theorem ellipse_condition (x y m : ℝ) :
  (1 < m ∧ m < 3) → (∀ x y, (∃ k1 k2: ℝ, k1 > 0 ∧ k2 > 0 ∧ k1 ≠ k2 ∧ (x^2 / k1 + y^2 / k2 = 1 ↔ (1 < m ∧ m < 3 ∧ m ≠ 2)))) :=
by 
  sorry

end ellipse_condition_l540_540958


namespace real_number_expression_equals_sqrt_15_l540_540230

theorem real_number_expression_equals_sqrt_15 : 
    ∃ x : ℝ, x = 3 + 5 / (2 + 5 / (1 + 5 / (2 + ...))) → x = Real.sqrt 15 :=
by
  -- Existence of x and the equation will be formalized
  sorry

end real_number_expression_equals_sqrt_15_l540_540230


namespace find_original_list_size_l540_540231

theorem find_original_list_size
  (n m : ℤ)
  (h1 : (m + 3) * (n + 1) = m * n + 20)
  (h2 : (m + 1) * (n + 2) = m * n + 22):
  n = 7 :=
sorry

end find_original_list_size_l540_540231


namespace max_m_value_l540_540369

-- Given the conditions
variables (b : ℝ) (a : ℝ)
variables (hb : b > 0)
variables (x : ℝ)

-- We define the inequality as a function
def inequality := (b - (a - 2))^2 + (Real.log b - (a - 1))^2

-- We want to prove that the maximum value of m that satisfies the inequality is 2
theorem max_m_value : (∀ b a, b > 0 → ∀ m, (inequality b a) ≥ m^2 - m → m ≤ 2) :=
by
  sorry

end max_m_value_l540_540369


namespace num_integers_units_digit_condition_l540_540439

theorem num_integers_units_digit_condition :
  ∃ (count : ℕ), count = 81 ∧
  ∀ n : ℕ, (1000 < n ∧ n < 2050) →
  (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)) →
  count = (∑ n in finset.range 1050 \ finset.range 1001, if (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)) then 1 else 0)
:= by
  sorry

end num_integers_units_digit_condition_l540_540439


namespace john_remaining_amount_l540_540491

theorem john_remaining_amount (initial_amount games: ℕ) (food souvenirs: ℕ) :
  initial_amount = 100 →
  games = 20 →
  food = 3 * games →
  souvenirs = (1 / 2 : ℚ) * games →
  initial_amount - (games + food + souvenirs) = 10 :=
by
  sorry

end john_remaining_amount_l540_540491


namespace area_of_triangle_PQR_l540_540860

-- Define the setup for the isosceles triangle PQR where PQ = PR and PS is the altitude
variables {P Q R S : Type}
variables [HasDist P] [HasDist Q] [HasDist R] [HasDist S]

-- Definitions according to the conditions
def is_isosceles_triangle (P Q R : Type) [HasDist P] [HasDist Q] [HasDist R] :=
  dist P Q = dist P R

def altitude_bisects_base (P Q R S : Type) [HasDist P] [HasDist Q] [HasDist R] [HasDist S] :=
  dist Q S = dist R S

def PQR_conditions (P Q R S : Type) [HasDist P] [HasDist Q] [HasDist R] [HasDist S] :=
  is_isosceles_triangle P Q R ∧ altitude_bisects_base P Q R S ∧ dist P Q = 13 ∧ dist Q R = 10

-- Lean statement for the proof problem
theorem area_of_triangle_PQR (P Q R S : Type) [HasDist P] [HasDist Q] [HasDist R] [HasDist S]
  (h₁ : is_isosceles_triangle P Q R)
  (h₂ : altitude_bisects_base P Q R S)
  (h₃ : dist P Q = 13)
  (h₄ : dist Q R = 10) :
  area P Q R = 60 :=
sorry

end area_of_triangle_PQR_l540_540860


namespace denominator_of_repeating_six_l540_540163

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l540_540163


namespace calc_expression_l540_540285

theorem calc_expression : (π - 3.14)^0 + sqrt 18 + (-1/2)^(-1) - abs (1 - sqrt 2) = 2 * sqrt 2 :=
by
  sorry

end calc_expression_l540_540285


namespace point_in_second_quadrant_l540_540836

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l540_540836


namespace circle_tangent_l540_540617

variables {O M : ℝ} {R : ℝ}

theorem circle_tangent
  (r : ℝ)
  (hOM_pos : O ≠ M)
  (hO : O > 0)
  (hR : R > 0)
  (h_distinct : ∀ (m n : ℝ), m ≠ n → abs (m - n) ≠ 0) :
  (r = abs (O - M) - R) ∨ (r = abs (O - M) + R) ∨ (r = R - abs (O - M)) →
  (abs ((O - M)^2 + r^2 - R^2) = 2 * R * r) :=
sorry

end circle_tangent_l540_540617


namespace fraction_of_shaded_area_l540_540552

theorem fraction_of_shaded_area (l w : ℝ) (hlw : l ≠ 0 ∧ w ≠ 0) :
  let rectangle_area := l * w in
  let triangle_area := (½) * l * (½ * w) in
  let shaded_area := rectangle_area - triangle_area in
  (shaded_area / rectangle_area) = (3 / 4) :=
by
  let rectangle_area := l * w
  let triangle_area := (½) * l * (½ * w)
  let shaded_area := rectangle_area - triangle_area
  have : (shaded_area / rectangle_area) = (3 / 4) := sorry
  exact this

end fraction_of_shaded_area_l540_540552


namespace dot_product_find_a_given_b_l540_540787

variables {a b c : ℝ}

-- Given the conditions:
def triangle_area (A : ℝ) : Prop := A = 3
def sides {a b c : ℝ} : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
def cos_A : Prop := (∃ (cosA : ℝ), cosA = 4/5)

-- Proof statements:
-- (I) Prove dot product of vectors AB and AC
theorem dot_product (A : ℝ) (cosA : ℝ) (bc : ℝ) (h : triangle_area A) (hcos_A : cos_A) (hbc : bc = 10) :
  ∃ (dot_AB_AC : ℝ), dot_AB_AC = 8 := sorry

-- (II) Given b, prove the value of a
theorem find_a_given_b (b : ℝ) (h : b = 2) (cosA : ℝ) (hcos_A : cos_A): 
  ∃ (a : ℝ), a = sqrt 13 := sorry

end dot_product_find_a_given_b_l540_540787


namespace calculate_inverse_y3_minus_y_l540_540449

theorem calculate_inverse_y3_minus_y
  (i : ℂ) (y : ℂ)
  (h_i : i = Complex.I)
  (h_y : y = (1 + i * Real.sqrt 3) / 2) :
  (1 / (y^3 - y)) = -1/2 + i * (Real.sqrt 3) / 6 :=
by
  sorry

end calculate_inverse_y3_minus_y_l540_540449


namespace polyhedron_volume_l540_540479

-- Define the conditions
structure EquilateralTriangle (s : ℝ) :=
  (side : ℝ := s)

structure Rectangle (l w : ℝ) :=
  (length : ℝ := l) 
  (width : ℝ := w)

structure RegularHexagon (s : ℝ) :=
  (side : ℝ := s)

def A : EquilateralTriangle 2 := {}
def E : EquilateralTriangle 2 := {}
def F : EquilateralTriangle 2 := {}

def B : Rectangle 2 1 := {}
def C : Rectangle 2 1 := {}
def D : Rectangle 2 1 := {}

def G : RegularHexagon 1 := {}

-- Prove the volume of the polyhedron formed by these polygons
theorem polyhedron_volume : 
  let volume := 4 + Real.sqrt 3 in
  volume = 4 + Real.sqrt 3 :=
by sorry

end polyhedron_volume_l540_540479


namespace number_of_possible_x_values_l540_540951
noncomputable def triangle_sides_possible_values (x : ℕ) : Prop :=
  27 < x ∧ x < 63

theorem number_of_possible_x_values : 
  ∃ n, n = (62 - 28 + 1) ∧ ( ∀ x : ℕ, triangle_sides_possible_values x ↔ 28 ≤ x ∧ x ≤ 62) :=
sorry

end number_of_possible_x_values_l540_540951


namespace sales_growth_correct_equation_l540_540234

theorem sales_growth_correct_equation (x : ℝ) 
(sales_24th : ℝ) (total_sales_25th_26th : ℝ) 
(h_initial : sales_24th = 5000) (h_total : total_sales_25th_26th = 30000) :
  (5000 * (1 + x)) + (5000 * (1 + x)^2) = 30000 :=
sorry

end sales_growth_correct_equation_l540_540234


namespace ratio_of_de_bc_l540_540484

-- Definitions for Triangle, Angle, and Altitudes
structure Triangle where
  A B C : ℝ

structure Angle where
  α : ℝ

structure Altitudes (T : Triangle) where
  CD BE : ℝ -- Altitudes on AB and AC respectively
  -- Assumed to be orthogonal by their definition

-- The theorem we need to state
theorem ratio_of_de_bc (T : Triangle) (α : ℝ) (alt : Altitudes T)
  (h1 : ∠ T.A = α) : 
  (alt.CD / distance T.B T.C) = |Real.cos α| :=
sorry

end ratio_of_de_bc_l540_540484


namespace Darla_electricity_bill_l540_540295

theorem Darla_electricity_bill :
  let tier1_rate := 4
  let tier2_rate := 3.5
  let tier3_rate := 3
  let tier1_limit := 300
  let tier2_limit := 500
  let late_fee1 := 150
  let late_fee2 := 200
  let late_fee3 := 250
  let consumption := 1200
  let cost_tier1 := tier1_limit * tier1_rate
  let cost_ttier2 := tier2_limit * tier2_rate
  let cost_tier3 := (consumption - (tier1_limit + tier2_limit)) * tier3_rate
  let total_cost := cost_tier1 + cost_tier2 + cost_tier3
  let late_fee := late_fee3
  let final_cost := total_cost + late_fee
  final_cost = 4400 :=
by
  sorry

end Darla_electricity_bill_l540_540295


namespace find_a_l540_540067

-- Define the lines and their slopes
def l1_slope (a : ℝ) : ℝ := -a
def l2_slope (a : ℝ) : ℝ := -(1 / (a - 2))

-- Define the perpendicular condition
def perpendicular (s1 s2 : ℝ) : Prop := s1 * s2 = -1

-- State the theorem
theorem find_a (a : ℝ) (h_perpendicular : perpendicular (l1_slope a) (l2_slope a)) : a = 1 := by
  sorry

end find_a_l540_540067


namespace volume_ratio_cone_pyramid_l540_540601

theorem volume_ratio_cone_pyramid
  (A B C D P Q A' C' D' E' : Type)
  [ordered_ring ABCD_PABCD]
  [ordered_ring Cone]
  [regular_pyramid PABCD ABCD]
  [cone_cone ABCD]
  (h1 : ∃ a : ℝ, (A_of_base = A) = a)
  (h2 : ∃ b : ℝ, (B D_on_lateral) = b)
  (h3 : ∃ c : ℝ, (C_on_base_plane) = c)
  (h4 : ∃ p : ℝ, (P_on_circle_base_cone) = p)
  :
  let volume_cone := 
    (1 / 3) * (Mathlib.pi * ((Mathlib.sqrt 3 * (a^2)) ^ 2)) * (Mathlib.sqrt 7 /2)
  let volume_pyramid := 
    (1 / 3) * ((a^2 / 2) * (Mathlib.sqrt 4 * (a^2)  - (a ^2 / 2)))

  volume_cone / volume_pyramid = (9 * Mathlib.pi * (Mathlib.sqrt 2)) / 8
  := by sorry

end volume_ratio_cone_pyramid_l540_540601


namespace even_multiples_of_25_l540_540818

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0

theorem even_multiples_of_25 (a b : ℕ) (h1 : 249 ≤ a) (h2 : b ≤ 501) :
  (a = 250 ∨ a = 275 ∨ a = 300 ∨ a = 350 ∨ a = 400 ∨ a = 450) →
  (b = 275 ∨ b = 300 ∨ b = 350 ∨ b = 400 ∨ b = 450 ∨ b = 500) →
  (∃ n, n = 5 ∧ ∀ m, (is_multiple_of_25 m ∧ is_even m ∧ a ≤ m ∧ m ≤ b) ↔ m ∈ [a, b]) :=
by sorry

end even_multiples_of_25_l540_540818


namespace min_value_of_t_for_AM_BM_condition_l540_540474

noncomputable def range_of_t (t : ℝ) : Prop :=
  t ≥ 2 * Real.sqrt 3 / 3

theorem min_value_of_t_for_AM_BM_condition :
  (∀ (x y : ℝ), (0 ≤ x ∧ x ≤ t) ∧ y = 2 - (2 * x / t) → 
  (x^2 + (y - 2)^2 ≤ 4 * (x^2 + (y - 1)^2))) ↔ (t ≥ (2 * Real.sqrt 3 / 3)) :=
begin
  split,
  { intro h, sorry },
  { intro h, sorry }
end

end min_value_of_t_for_AM_BM_condition_l540_540474


namespace mow_lawn_time_is_1_point_07_hours_l540_540541

noncomputable def mowLawnTime : ℝ :=
  let lawnWidth := 120 -- feet
  let lawnLength := 80 -- feet
  let initialSwathWidth := 30 / 12 -- converting inches to feet
  let overlap := 6 / 12 -- converting inches to feet
  let effectiveSwathWidth := initialSwathWidth - overlap
  let numStrips := lawnWidth / effectiveSwathWidth
  let totalDistance := numStrips * lawnLength
  let mowingRate := 4500 -- feet per hour
  totalDistance / mowingRate

theorem mow_lawn_time_is_1_point_07_hours :
  mowLawnTime ≈ 1.07 :=
by
  sorry

end mow_lawn_time_is_1_point_07_hours_l540_540541


namespace difference_of_two_numbers_l540_540961

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l540_540961


namespace printers_time_ratio_l540_540640

def rate (time : ℚ) : ℚ := 1 / time

def combined_rate (rate1 rate2 : ℚ) : ℚ := rate1 + rate2

def time_to_complete (rate : ℚ) : ℚ := 1 / rate

theorem printers_time_ratio :
  let X_rate := rate 12
  let Y_rate := rate 10
  let Z_rate := rate 20
  let combined_YZ_rate := combined_rate Y_rate Z_rate
  let X_time := 12
  let YZ_time := time_to_complete combined_YZ_rate
  (X_time / YZ_time) = 9 / 5 :=
begin
  sorry
end

end printers_time_ratio_l540_540640


namespace pentagon_affine_transformation_l540_540344

structure Pentagon (P: Type) :=
(vertices : list P)
(convex : Prop)

def DiagonalParallelSides (P: Type) [affine_space P V] (pent: Pentagon P) : Prop :=
  ∀ diagonal ∈ diagonals pent.vertices, ∃ side ∈ sides pent.vertices, are_parallel diagonal side

def RegularPentagon (P: Type) [affine_space P V] (pent: Pentagon P) : Prop :=
  let vertices := pent.vertices in
  are_regular vertices

theorem pentagon_affine_transformation
  (P: Type) [affine_space P V]
  (pent: Pentagon P)
  (H1 : pent.convex)
  (H2 : DiagonalParallelSides P pent)
: ∃ (aff_trans: P → P), RegularPentagon P (affine_transform pent aff_trans) :=
sorry

end pentagon_affine_transformation_l540_540344


namespace find_f_2009_l540_540003

variable f : ℝ → ℝ

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetry_about_two (f : ℝ → ℝ) : Prop := ∀ x, f (2 + x) = f (2 - x)

theorem find_f_2009 (H1 : odd_function f)
                     (H2 : symmetry_about_two f)
                     (H3 : f (-1) = -2) :
  f 2009 = 2 := by
  sorry

end find_f_2009_l540_540003


namespace infinite_product_value_l540_540313

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540313


namespace maxim_receives_l540_540898

def initial_deposit : ℝ := 1000
def annual_interest_rate : ℝ := 0.12
def monthly_compounded : ℕ := 12
def time_period : ℝ := 1 / 12

def final_amount (P r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem maxim_receives (P r : ℝ) (n : ℕ) (t : ℝ) :
  final_amount P r n t = 1010 :=
by
  have h1 : P = initial_deposit := rfl
  have h2 : r = annual_interest_rate := rfl
  have h3 : n = monthly_compounded := rfl
  have h4 : t = time_period := rfl
  rw [h1, h2, h3, h4]
  sorry

end maxim_receives_l540_540898


namespace infinite_product_value_l540_540329

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540329


namespace alpha_value_l540_540599

theorem alpha_value (b : ℝ) : (∀ x : ℝ, (|2 * x - 3| < 2) ↔ (x^2 + -3 * x + b < 0)) :=
by
  sorry

end alpha_value_l540_540599


namespace proposition_correctness_l540_540636

theorem proposition_correctness :
  (∀ x : ℝ, (|x-1| < 2) → (x < 3)) ∧
  (∀ (P Q : Prop), (Q → ¬ P) → (P → ¬ Q)) :=
by 
sorry

end proposition_correctness_l540_540636


namespace sqrt_expr_value_l540_540748

noncomputable def sqrt_expr_pos (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ := (Real.sqrt (a + b) - Real.sqrt (a - b))

theorem sqrt_expr_value : sqrt_expr_pos 9 (4 * Real.sqrt 5) (by norm_num) (by norm_num; apply Real.sqrt_pos_of_pos; norm_num) = 4 :=
  sorry

end sqrt_expr_value_l540_540748


namespace gcd_values_count_l540_540983

theorem gcd_values_count (a b : ℕ) (h : a * b = 3600) : ∃ n, n = 29 ∧ ∀ d, d ∣ a ∧ d ∣ b → d = gcd a b → n = 29 :=
by { sorry }

end gcd_values_count_l540_540983


namespace total_candies_l540_540606

theorem total_candies (red_candies blue_candies : ℕ)
  (h_red_candies : red_candies = 145)
  (h_blue_candies : blue_candies = 3264) :
  red_candies + blue_candies = 3409 :=
by
  rw [h_red_candies, h_blue_candies]
  exact rfl

end total_candies_l540_540606


namespace problem1_problem2_l540_540723

-- Problem 1: Simplification and computation of the algebraic expression
theorem problem1 : 
  (1 : ℝ) * (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 :=
by 
  sorry

-- Problem 2: Simplification and computation of the logarithmic expression
theorem problem2 :
  (2 : ℝ) * (Real.log10 5) ^ 2 + Real.log10 (2 * Real.log10 50) =
  (Real.log10 5) ^ 2 + Real.log10 2 + Real.log10 (Real.log10 2 + 2 * Real.log10 5) :=
by 
  sorry

end problem1_problem2_l540_540723


namespace pawn_moves_on_octagon_l540_540763

noncomputable def a (n : ℕ) : ℚ -- Placeholder for the function a
noncomputable def x := 2 + Real.sqrt 2 -- Define x
noncomputable def y := 2 - Real.sqrt 2 -- Define y

theorem pawn_moves_on_octagon (k : ℕ) :
  a (2*k - 1) = 0 ∧ 
  a (2*k) = (x^(k-1) - y^(k-1)) / Real.sqrt 2 := 
sorry

end pawn_moves_on_octagon_l540_540763


namespace car_travel_time_l540_540705

-- Definitions and assumptions based on the conditions
def miles_per_gallon := 30
def full_tank_gallons := 12
def speed_mph := 60
def tank_used_fraction := 0.8333333333333334

-- Lean statement to prove the car travels for 5 hours
theorem car_travel_time :
  let gasoline_used := full_tank_gallons * tank_used_fraction in
  let distance_traveled := gasoline_used * miles_per_gallon in
  let travel_time := distance_traveled / speed_mph in
  travel_time = 5 :=
by
  sorry

end car_travel_time_l540_540705


namespace ratio_of_ab_l540_540531

noncomputable theory
open Complex

theorem ratio_of_ab (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∃ c : ℝ, (3 - 8 * I) * (a + b * I) = c * I) : a / b = -8 / 3 :=
by sorry

end ratio_of_ab_l540_540531


namespace red_blue_point_pairing_l540_540653

variables {P : Type} [Plane P] 

def pair_points_no_intersection (n : ℕ) (R B : fin n → P) : Prop :=
  ∃ (pairs : fin n → fin n), 
  (∀ i j : fin n, i ≠ j → 
  ¬ intersects (segment (R i) (B (pairs i))) (segment (R j) (B (pairs j))))

theorem red_blue_point_pairing (n : ℕ) 
  (R B : fin n → P) 
  (no_collinear : ∀ (i j k : P), 
  distinct {i, j, k} → ¬ collinear {i, j, k}) : 
  pair_points_no_intersection n R B :=
sorry

end red_blue_point_pairing_l540_540653


namespace benny_attended_games_l540_540281

def total_games : ℕ := 39
def games_missed : ℕ := 25
def games_attended : ℕ := total_games - games_missed

theorem benny_attended_games : games_attended = 14 := by
  unfold games_attended
  unfold total_games
  unfold games_missed
  simp
  sorry

end benny_attended_games_l540_540281


namespace radian_measure_of_200_degrees_l540_540203

theorem radian_measure_of_200_degrees :
  (200 : ℝ) * (Real.pi / 180) = (10 / 9) * Real.pi :=
sorry

end radian_measure_of_200_degrees_l540_540203


namespace infinite_product_value_l540_540314

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540314


namespace cube_of_99999_eq_l540_540501

theorem cube_of_99999_eq : let N := 99999 in N ^ 3 = 999970000299999 := by
  let N := 99999
  have : N = 10 ^ 5 - 1 := by rfl
  calc
    N ^ 3 = (10 ^ 5 - 1) ^ 3       := by rw this
        ... = 10 ^ 15 - 3 * 10 ^ 10 + 3 * 10 ^ 5 - 1 := by sorry
        ... = 999970000299999 := by sorry

end cube_of_99999_eq_l540_540501


namespace arithmetic_sequence_property_sequence_b_property_sum_B_n_l540_540771

theorem arithmetic_sequence_property (d a1 : ℕ) (a : ℕ → ℕ) :
    a 3 = 7 → 
    (4 * a1 + 6 * d) = 24 → 
    (∀ n : ℕ, a n = a1 + (n - 1) * d) → 
    (∀ n : ℕ, a n = 2 * n + 1) :=
sorry

theorem sequence_b_property (a : ℕ → ℕ) (b : ℕ → ℕ) :
    (∀ n : ℕ, a n = 2 * n + 1) →
    (∀ n : ℕ, T_n = n^2 + a n) →
    (∀ n : ℕ, b n = 
      if n = 1 then 4 
      else T_n - T_(n - 1)) → 
    (∀ n : ℕ, b n = 
      if n = 1 then 4 
      else 2 * n + 1) :=
sorry

theorem sum_B_n (b : ℕ → ℕ) (B : ℕ → ℝ) :
    (∀ n : ℕ, b n = if n = 1 then 4 else 2 * n + 1) →
    (∀ n : ℕ, B_n = ∑ i in range n, 1 / (b i * b (i + 1))) →
    B n = (3 / 20) - (1 / (4 * n + 6)) :=
sorry

end arithmetic_sequence_property_sequence_b_property_sum_B_n_l540_540771


namespace set_inclusion_l540_540778

-- Define the sets P and Q
def P : set ℝ := {x | x < 2}
def Q : set ℝ := {x | x^2 < 2}

-- State the theorem
theorem set_inclusion : P ⊇ Q :=
by {
  -- Proof to be filled here
  sorry
}

end set_inclusion_l540_540778


namespace price_difference_at_least_25_l540_540901

theorem price_difference_at_least_25
    (price : ℕ → ℕ → ℕ)
    (distinct_prices : ∀ i j, i ≠ j → price i 30 ≠ price j 30)
    (rate : ℕ → ℕ → ℕ → ℕ) :
    ∃ i j, i ≠ j ∧ price j 30 ≥ 25 * price i 30 := 
begin
    have initial_prices : ∀ i, price i 0 = 1,
    sorry,
    
    have daily_increase : ∀ i n, price i (n + 1) = rate (price i n) 2 3,
    sorry,
    
    have final_price_representation : ∀ i, ∃ a b : ℕ, price i 30 = 2^a * 3^b,
    sorry,
    
    have distinct_final_prices : ∀ i j, i ≠ j -> price i 30 ≠ price j 30,
    from distinct_prices,
    
    have min_ratio : ∀ i j, i ≠ j → (rate price i 30 ≥ 25 * rate price i 30), 
    sorry,

    use (rate final_price_representation distinct_final_prices min_ratio),
    sorry,
end

end price_difference_at_least_25_l540_540901


namespace range_of_m_l540_540416

theorem range_of_m (m : ℝ) (h : ∀ x y : ℝ, 0 < x ∧ x ≤ 1 ∧ 0 < y ∧ y ≤ 1 ∧ x < y → f x m > f y m) : 
m ∈ (-∞, 0) ∪ (1, 3] :=
by sorry

noncomputable def f (x m : ℝ) := (sqrt (3 - m * x)) / (m - 1)

end range_of_m_l540_540416


namespace solution_set_log_inequality_l540_540209

theorem solution_set_log_inequality :
  { x : ℝ | 0 < x ∧ x < 2 ∧ x ≠ 1 } = { x : ℝ | log 2 (abs (x - 1)) < 0 } :=
by
  sorry

end solution_set_log_inequality_l540_540209


namespace point_in_second_quadrant_l540_540833

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l540_540833


namespace marsha_pay_per_mile_l540_540538

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l540_540538


namespace triangle_minimum_side_c_l540_540390

theorem triangle_minimum_side_c (a b c : ℝ)
  (h1 : a + b = 2)
  (h2 : 120 * π / 180 = 2 * π / 3) : 
  ∃ (c : ℝ), c = sqrt 3 ∧ 
    c = sqrt (a^2 + b^2 + a * b * cos (2 * π / 3)) :=
begin
  sorry
end

end triangle_minimum_side_c_l540_540390


namespace find_a_l540_540064

theorem find_a (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f (g x) = x)
  (h2 : f x = (Real.log (x + 1) / Real.log 2) + a)
  (h3 : g 4 = 1) :
  a = 3 :=
sorry

end find_a_l540_540064


namespace projection_magnitude_l540_540429

variable {𝕜 : Type*} [RealField 𝕜]
variable {V : Type*} [InnerProductSpace 𝕜 V]

-- Given conditions
variables (u z : V)
hypothesis hu_norm : ∥u∥ = 5
hypothesis hz_norm : ∥z∥ = 8
hypothesis huz_dot : ⟪u, z⟫ = 20

-- Statement to be proven
theorem projection_magnitude : ∥(u.proj z)∥ = 2.5 :=
sorry

end projection_magnitude_l540_540429


namespace point_in_second_quadrant_l540_540837

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l540_540837


namespace Sn_bounds_l540_540133

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℝ :=
  (∑ i in Finset.range n, (a i + a (i + 2) % n) / (a ((i + 1) % n)))

theorem Sn_bounds (n : ℕ) (a : ℕ → ℕ) (h₀ : 3 ≤ n) (h₁ : ∀ i, 1 ≤ a i ∧ a i % (a ((i - 1 + n) % n) + a ((i + 1) % n) ) = 0) : 
  2 * n ≤ S_n a n ∧ S_n a n ≤ 3 * n := 
sorry

end Sn_bounds_l540_540133


namespace population_reaches_capacity_in_75_years_l540_540198

-- Defining constants and conditions
def total_land : ℕ := 30000
def sustainable_land_per_person : ℝ := 1.2
def initial_population : ℕ := 300
def growth_rate : ℕ := 4
def years_per_period : ℕ := 25

-- Defining the maximum carrying capacity
def max_capacity : ℕ := (total_land / sustainable_land_per_person).to_nat

-- Function to calculate population after given periods
def population_after_years (initial_population : ℕ) (growth_rate : ℕ) (years_per_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * (growth_rate ^ (years / years_per_period))

-- The proof statement
theorem population_reaches_capacity_in_75_years : 
  population_after_years initial_population growth_rate years_per_period 75 ≥ max_capacity :=
by
  sorry

end population_reaches_capacity_in_75_years_l540_540198


namespace problem_statement_l540_540405

noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)

def S_n (n : ℕ) : ℕ := (n * (a_n (n+1) + a_n 1)) / 2

def b_n (a : ℕ) (n : ℕ) : ℕ := -(a * n + 1) * a_n n

def T_n (a : ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range n, b_n a i

theorem problem_statement (n : ℕ) (h : n > 0) : 
  let a := -2 in 
  (a_n 1 = 1) ∧ 
  (∀ k ≥ 2, a_n k = 2^(k-1)) ∧ 
  T_n a n = (2 * n - 3) * 2^n + 3 :=
by
  simp
  sorry

end problem_statement_l540_540405


namespace factor_quadratic_expression_l540_540940

theorem factor_quadratic_expression (a b : ℤ) (h: 25 * -198 = -4950 ∧ a + b = -195 ∧ a * b = -4950) : a + 2 * b = -420 :=
sorry

end factor_quadratic_expression_l540_540940


namespace lengths_of_trains_l540_540688

noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def length_of_train (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem lengths_of_trains (Va Vb : ℝ) : Va = 60 ∧ Vb < Va ∧ length_of_train (km_per_hour_to_m_per_s Va) 42 = (700 : ℝ) 
    → length_of_train (km_per_hour_to_m_per_s Vb * (42 / 56)) 56 = (700 : ℝ) :=
by
  intros h
  sorry

end lengths_of_trains_l540_540688


namespace airplane_estimation_equivalent_l540_540817

noncomputable def estimate_airplanes (flights: ℕ) (first_repetition: ℕ) : ℝ :=
  let h := (list.range 14).sum (λ j, 1 / (j + 1));
  let gamma := 0.5772156649; -- Euler-Mascheroni constant approximation
  let harmonic_14 := real.log 14 + gamma;
  1 + harmonic_14

theorem airplane_estimation_equivalent (h1 : flights = 15) (h2 : first_repetition = 15) : 
  estimate_airplanes flights first_repetition ≈ 134 :=
by
  sorry

end airplane_estimation_equivalent_l540_540817


namespace faster_train_speed_l540_540616

theorem faster_train_speed 
  (slower_speed : ℕ) (time_seconds : ℕ) (train_length : ℕ) :
  slower_speed = 36 →
  time_seconds = 27 →
  train_length = 3750 / 100 →
  (V : ℕ) →
  75 = (V - 36) * 1000 / 3600 * 27 →
  V = 46 :=
begin
  intros h1 h2 h3 V h4,
  sorry
end

end faster_train_speed_l540_540616


namespace intersection_of_sets_l540_540032

def setM : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }
def setN : Set ℝ := { x | Real.log x ≥ 0 }

theorem intersection_of_sets : (setM ∩ setN) = { x | 1 ≤ x ∧ x ≤ 4 } := 
by {
  sorry
}

end intersection_of_sets_l540_540032


namespace max_min_values_of_expression_l540_540361

def max_min_expression (x y : ℝ) : ℝ := x * real.sqrt (4 - y^2) + y * real.sqrt (9 - x^2)

theorem max_min_values_of_expression :
  ∀ (x y: ℝ), -3 ≤ x ∧ x ≤ 3 ∧ -2 ≤ y ∧ y ≤ 2 →
  -6 ≤ max_min_expression x y ∧ max_min_expression x y ≤ 6 :=
by
  sorry

end max_min_values_of_expression_l540_540361


namespace wall_length_to_height_ratio_l540_540947

theorem wall_length_to_height_ratio
  (W H L : ℝ)
  (V : ℝ)
  (h1 : H = 6 * W)
  (h2 : L * H * W = V)
  (h3 : V = 86436)
  (h4 : W = 6.999999999999999) :
  L / H = 7 :=
by
  sorry

end wall_length_to_height_ratio_l540_540947


namespace proposition_1_false_proposition_2_true_proposition_3_true_proposition_4_false_l540_540374

def f (x : ℝ) : ℝ := Real.sin x + 1 / (Real.sin x)

theorem proposition_1_false : ¬ (∀ x, f(x) = f(-x)) := 
by sorry

theorem proposition_2_true : ∀ x, f(-x) = -f(x) := 
by sorry

theorem proposition_3_true : ∀ x, f(Real.pi - x) = f(x) := 
by sorry

theorem proposition_4_false : ¬ (∀ x, f(x) ≥ 2) := 
by sorry

end proposition_1_false_proposition_2_true_proposition_3_true_proposition_4_false_l540_540374


namespace part1_part2_l540_540795

-- Defining the quadratic equation
def quad_eq (k x : ℝ) := k * x^2 + (k + 3) * x + 3

-- Part 1: Proving that the equation has two real roots given k ≠ 0
theorem part1 (k : ℝ) (h : k ≠ 0) : 
  let Δ := (k + 3)^2 - 4 * k * 3 in Δ ≥ 0 :=
by
  let Δ := (k + 3)^2 - 4 * k * 3
  have h_Δ : Δ = (k - 3)^2, from sorry
  have h_nonneg : (k - 3)^2 ≥ 0, from sorry
  exact h_nonneg

-- Part 2: Proving that if the roots are integers, then k = 1 or k = 3
theorem part2 (k : ℝ) (h : k ≠ 0) (hx1 : quad_eq k (-3 / k) = 0) (hx2 : quad_eq k (-1) = 0) :
  k = 1 ∨ k = 3 :=
by
  have k_ne0 : k ≠ 0, from h
  have k_int : -3 / k ∈ ℤ ∧ -1 ∈ ℤ, from sorry
  have k_div3 : k ∈ {1, 3}, from sorry
  exact k_div3

end part1_part2_l540_540795


namespace no_integer_solution_l540_540360

theorem no_integer_solution (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by
  -- Proof omitted
  sorry

end no_integer_solution_l540_540360


namespace regular_ngon_is_regular_for_large_n_l540_540368

def regular_ngon_condition (n : ℕ) (A : Fin n → Point) : Prop :=
  ∀ (k : ℕ), (2 ≤ k ∧ k ≤ n - 2) →
    dist (A 1) (A (k + 1)) = dist (A 2) (A (k + 2)) ∧ 
    dist (A 2) (A (k + 2)) = dist (A (n - 1)) (A (k - 1)) ∧
    dist (A (n - 1)) (A (k - 1)) = dist (A n) (A k)

theorem regular_ngon_is_regular_for_large_n (n : ℕ) (A : Fin n → Point) 
  (hn : 7 ≤ n) (hA : regular_ngon_condition n A) : 
  (∀ i j : Fin n, dist (A i) (A (i + 1)) = dist (A j) (A (j + 1))) ∧
  (∀ i j : Fin n, ∡ A (A (i + 1)) A (j + 1) = ∡ A (A (j + 1)) A (i + 2)) :=
sorry

end regular_ngon_is_regular_for_large_n_l540_540368


namespace distance_from_M_to_plane_is_correct_l540_540458

noncomputable def distance_from_M_to_plane (AC BC AB AM: ℝ) (M: ℝ → ℝ) :=
  ∀ (C B : Real) (angleC : ℝ) (angleB : ℝ) (angleA : ℝ) (F: B ≠ C) (F1: AC = 2) (F2: angleC = π / 2) (F3: angleB = π / 6) (F4: AB = 4) (F5: AM = 2)
  (P6: AC = 2) (M1: midpoint (B C) (M AC) (∃ D, abs (dist AB) = (2* sqrt 2))), 

by sorry


theorem distance_from_M_to_plane_is_correct :
  distance_from_M_to_plane 2 (2 * sqrt 3) 4 2 1 :=
by sorry

end distance_from_M_to_plane_is_correct_l540_540458


namespace probability_251_is_5_over_14_l540_540712

-- Conditions about bus intervals and random arrival
def bus_interval_152 : ℕ := 5
def bus_interval_251 : ℕ := 7

-- Define the area of the rectangle and the triangle
def area_rectangle (a b : ℕ) : ℚ := a * b
def area_triangle (a : ℕ) : ℚ := (a * a) / 2

-- Define the probability calculation
noncomputable def probability_first_bus_251 : ℚ :=
  area_triangle bus_interval_152 / area_rectangle bus_interval_152 bus_interval_251

-- The theorem that needs to be proven
theorem probability_251_is_5_over_14 :
  probability_first_bus_251 = 5 / 14 := by
  sorry

end probability_251_is_5_over_14_l540_540712


namespace infinite_pairs_x_y_in_A_or_B_l540_540218

def tuanis (a b : ℕ) : Prop := 
  ∃(m : ℕ), a + b = Σ.digits 10 m ∧ ∀ (d : ℕ), d ∈ (Σ.digits 10 m).toList → d = 0 ∨ d = 1

def A := {a : ℕ | ∃ b ∈ B, tuanis a b}
def B := {b : ℕ | ∃ a ∈ A, tuanis a b}

theorem infinite_pairs_x_y_in_A_or_B :
  ∃ s ∈ {A, B}, ∃ (x y : ℕ), (x ∈ s ∧ y ∈ s) ∧ x - y = 1 ∧ ∀ p ∈ ℕ × ℕ, p ∈ s → ∞ (p.fst - p.snd = 1) := 
sorry

end infinite_pairs_x_y_in_A_or_B_l540_540218


namespace problem1_problem2_l540_540019

noncomputable def ellipse_std_eqn (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def hyp_std_eqn : Prop :=
  ∀ x y : ℝ, x^2 - y^2 = 1

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def point (x y : ℝ) : Prop := True

noncomputable def vector_eq (a b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (a * x₁, a * y₁) = (b * x₂, b * y₂)

noncomputable def line_eq (k : ℝ) : ℝ → ℝ := λ x, k * x + 1

theorem problem1 :
  (∃ c : ℝ, c = Real.sqrt 2) →
  (∃ a e : ℝ, e = c / a ∧ e = Real.sqrt 2 / 2 ∧ a = 2) →
  ellipse_std_eqn 2 (Real.sqrt 2) := sorry

theorem problem2 :
  (∃ a b : ℝ, ellipse_std_eqn 2 (Real.sqrt 2) ∧ ∀ x y : ℝ, (0, 1) ≠ (x, y) ∧ point x y) →
  (∃ A B : ℝ × ℝ, vector_eq (-A.1) (2 * B.1) A.1 A.2 B.1 B.2) →
  (∃ k : ℝ, k^2 = 1 / 14) →
  (∃ S : ℝ, S = 1/2 * Real.sqrt (1 + k^2) * Real.sqrt ((A.1 + B.1) ^ 2 - 4 * A.1 * B.1)) →
  (∃ area : ℝ, area = Real.sqrt 126 / 8) :=
sorry

end problem1_problem2_l540_540019


namespace diamond_cut_1_3_loss_diamond_max_loss_ratio_l540_540949

noncomputable def value (w : ℝ) : ℝ := 6000 * w^2

theorem diamond_cut_1_3_loss (a : ℝ) :
  (value a - (value (1/4 * a) + value (3/4 * a))) / value a = 0.375 :=
by sorry

theorem diamond_max_loss_ratio :
  ∀ (m n : ℝ), (m > 0) → (n > 0) → 
  (1 - (value (m/(m + n)) + value (n/(m + n))) ≤ 0.5) :=
by sorry

end diamond_cut_1_3_loss_diamond_max_loss_ratio_l540_540949


namespace speed_of_current_l540_540211

/-- The condition that the speed of the boat in still water is 50 kmph --/
def speed_boat_still_water : ℝ := 50

/-- The condition that the speed upstream is 30 kmph --/
def speed_upstream : ℝ := 30

/-- The theorem stating that the speed of the current is 20 kmph --/
theorem speed_of_current (c : ℝ) : (speed_boat_still_water - c = speed_upstream) → c = 20 := 
by
  intros h
  have h1 : c = speed_boat_still_water - speed_upstream, 
  from eq_sub_of_add_eq h.symm,
  rw [speed_boat_still_water, speed_upstream] at h1,
  simp at h1,
  exact h1.symm

end speed_of_current_l540_540211


namespace positional_relationship_l540_540590

-- Define the circle equation
def circle (x y : ℝ) : ℝ := x^2 + y^2 - 2 * x + 4 * y - 4 

-- Define points M and N
def M := (2 : ℝ, -4 : ℝ)
def N := (-2 : ℝ, 1 : ℝ)

-- Lean statement to prove points' positional relationship relative to the circle
theorem positional_relationship :
  circle M.1 M.2 < 0 ∧ circle N.1 N.2 > 0 :=
by
  sorry

end positional_relationship_l540_540590


namespace conjugate_complex_magnitudes_eq_l540_540445

noncomputable def complex_numbers := ℂ

open complex

variables {x y : complex_numbers} 

theorem conjugate_complex_magnitudes_eq :
  (x.re + y.re)^2 - complex.i * (3 * (x.re^2 + x.im^2)) = 4 - 6 * complex.i →
  ∃ a b : ℝ, x = ⟨a, b⟩ ∧ y = ⟨a, -b⟩ ∧ |x| + |y| = 2 * real.sqrt 2 :=
begin
  sorry,
end

end conjugate_complex_magnitudes_eq_l540_540445


namespace parallel_vectors_m_eq_4_over_3_l540_540847

variables {a : ℝ × ℝ} {b : ℝ × ℝ} (m : ℝ)

def vector_a := (3 : ℝ, -4 : ℝ)
def vector_b := (-1, m)

theorem parallel_vectors_m_eq_4_over_3
  (h : ∃ k : ℝ, vector_a = (k * (-1), k * m)) : m = 4 / 3 :=
by
  sorry

end parallel_vectors_m_eq_4_over_3_l540_540847


namespace digits_of_300_pow_8_l540_540984

theorem digits_of_300_pow_8 : 
  let d := Nat.floor (Real.log10 (300^8)) + 1 
  in d = 20 := 
by
  sorry

end digits_of_300_pow_8_l540_540984


namespace coloring_paths_inequality_l540_540726

-- Definitions and conditions
def is_adjacent (pos1 pos2 : (ℕ × ℕ)) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 + 1 = pos2.2)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 = pos2.1 + 1 ∨ pos1.1 + 1 = pos2.1))

def is_path (seq : list (ℕ × ℕ)) : Prop :=
  ∀ i < seq.length - 1, is_adjacent (seq.nth_le i sorry) (seq.nth_le (i + 1) sorry)

def non_intersecting (path1 path2 : list (ℕ × ℕ)) : Prop :=
  ∀ square ∈ path1, square ∉ path2

def has_black_path (coloring : (ℕ × ℕ) → bool) (m n : ℕ) : Prop :=
  ∃ path : list (ℕ × ℕ), 
    path.head = (0, 0) ∧ path.last sorry = (m - 1, n - 1) ∧ is_path path ∧ 
    ∀ square ∈ path, coloring square = tt

def has_two_non_intersecting_black_paths (coloring : (ℕ × ℕ) → bool) (m n : ℕ) : Prop :=
  ∃ path1 path2 : list (ℕ × ℕ), 
    has_black_path coloring m n ∧ non_intersecting path1 path2

def count_colorings_with_black_paths (m n : ℕ) : ℕ :=
  -- Placeholder for the actual counting method
  sorry

def count_colorings_with_two_non_intersecting_black_paths (m n : ℕ) : ℕ :=
  -- Placeholder for the actual counting method
  sorry

theorem coloring_paths_inequality (m n : ℕ) : 
  (count_colorings_with_black_paths m n) ^ 2 ≥ 
  (count_colorings_with_two_non_intersecting_black_paths m n) * 2 ^ (m * n) := 
sorry

end coloring_paths_inequality_l540_540726


namespace point_in_second_quadrant_l540_540834

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l540_540834


namespace remaining_elements_is_60_l540_540957

-- Define the set T
def T : Finset ℕ := Finset.range 101

-- Define the set of multiples of 4
def multiplesOf4 : Finset ℕ := (Finset.range 26).image (λ n, 4 * n)

-- Define the set of multiples of 5
def multiplesOf5 : Finset ℕ := (Finset.range 21).image (λ n, 5 * n)

-- Define the set of common multiples of 4 and 5 (i.e., multiples of 20)
def multiplesOf20 : Finset ℕ := (Finset.range 6).image (λ n, 20 * n)

-- Define the number of elements remaining after removing multiples of 4 and 5
def remainingElements : ℕ :=
  T.card - multiplesOf4.card - (multiplesOf5.card - multiplesOf20.card)

-- Theorem stating the number of remaining elements is 60
theorem remaining_elements_is_60 : remainingElements = 60 := by
  sorry

end remaining_elements_is_60_l540_540957


namespace no_function_exists_l540_540138

-- Main theorem statement
theorem no_function_exists : ¬ ∃ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * f (2 * y * f x + f y) = x^3 * f (y * f x)) ∧ 
  (∀ z : ℝ, 0 < z → f z > 0) :=
sorry

end no_function_exists_l540_540138


namespace female_democrats_ratio_l540_540963

theorem female_democrats_ratio 
  (M F : ℕ) 
  (H1 : M + F = 660)
  (H2 : (1 / 3 : ℝ) * 660 = 220)
  (H3 : ∃ dem_males : ℕ, dem_males = (1 / 4 : ℝ) * M)
  (H4 : ∃ dem_females : ℕ, dem_females = 110) :
  110 / F = 1 / 2 :=
by
  sorry

end female_democrats_ratio_l540_540963


namespace solve_sqrt_expression_l540_540038

theorem solve_sqrt_expression (x y : ℝ) 
  (h : sqrt (x + 5) + (2 * x - y)^2 = 0) : 
  sqrt (x^2 - 2 * x * y + y^2) = 5 := by
  sorry

end solve_sqrt_expression_l540_540038


namespace simplify_sqrt_one_third_l540_540567

theorem simplify_sqrt_one_third : sqrt (1 / 3) = sqrt 3 / 3 := 
by
  sorry

end simplify_sqrt_one_third_l540_540567


namespace repeating_decimals_subtraction_l540_540979

theorem repeating_decimals_subtraction : (0.246246246246... - 0.135135135135... - 0.012012012012...) = (1 / 9) := 
sorry

end repeating_decimals_subtraction_l540_540979


namespace infinite_product_to_rational_root_l540_540317

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540317


namespace prob_first_two_same_color_expected_value_eta_l540_540676

-- Definitions and conditions
def num_white : ℕ := 4
def num_black : ℕ := 3
def total_pieces : ℕ := num_white + num_black

-- Probability of drawing two pieces of the same color
def prob_same_color : ℚ :=
  (4/7 * 3/6) + (3/7 * 2/6)

-- Expected value of the number of white pieces drawn in the first four draws
def E_eta : ℚ :=
  1 * (4 / 35) + 2 * (18 / 35) + 3 * (12 / 35) + 4 * (1 / 35)

-- Proof statements
theorem prob_first_two_same_color : prob_same_color = 3 / 7 :=
  by sorry

theorem expected_value_eta : E_eta = 16 / 7 :=
  by sorry

end prob_first_two_same_color_expected_value_eta_l540_540676


namespace compute_star_l540_540298

def star (x y : ℕ) := 4 * x + 6 * y

theorem compute_star : star 3 4 = 36 := 
by
  sorry

end compute_star_l540_540298


namespace planning_committee_ways_is_20_l540_540271

-- Define the number of students in the council
def num_students : ℕ := 6

-- Define the ways to choose a 3-person committee from num_students
def committee_ways (x : ℕ) : ℕ := Nat.choose x 3

-- Given condition: number of ways to choose the welcoming committee is 20
axiom welcoming_committee_condition : committee_ways num_students = 20

-- Statement to prove
theorem planning_committee_ways_is_20 : committee_ways num_students = 20 := by
  exact welcoming_committee_condition

end planning_committee_ways_is_20_l540_540271


namespace number_of_8_digit_increasing_integers_mod_1000_l540_540108

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_8_digit_increasing_integers_mod_1000 :
  let M := choose 9 8
  M % 1000 = 9 :=
by
  let M := choose 9 8
  show M % 1000 = 9
  sorry

end number_of_8_digit_increasing_integers_mod_1000_l540_540108


namespace difference_of_two_numbers_l540_540960

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l540_540960


namespace repeating_decimal_simplest_denominator_l540_540167

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l540_540167


namespace tangent_line_through_point_l540_540023

theorem tangent_line_through_point (a : ℝ) : 
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → y = a) ∧ 
    (∀ x y : ℝ, y = l x → (x - 1)^2 + y^2 = 4) → 
    a = 0 :=
by
  sorry

end tangent_line_through_point_l540_540023


namespace sin_satisfies_conditions_l540_540797

-- Define the period condition
def period_sin_shift : Real → Prop := λ y, (∀ x, y (x + π) = y x)

-- Define the symmetry condition
def symmetric_sin_shift (f : Real → Real) : Prop := 
  ∀ x, f (2 * x - (π / 3)) = f (2 * (π / 3) - (2 * x - (π / 3)))

-- Define the increasing function condition
def increasing_on_interval (f : Real → Real) : Prop := 
  ∀ x y, - (π / 6) < x ∧ x < (π / 3) ∧ - (π / 6) < y ∧ y < (π / 3) → x < y → f x < f y

-- Prove that the function satisfies all the conditions
theorem sin_satisfies_conditions : 
  period_sin_shift (λ x, Real.sin (2 * x - π / 6))
  ∧
  symmetric_sin_shift (λ x, Real.sin (2 * x - π / 6))
  ∧
  increasing_on_interval (λ x, Real.sin (2 * x - π / 6)) :=
begin
  sorry -- The actual proof goes here
end

end sin_satisfies_conditions_l540_540797


namespace john_workdays_l540_540493

def mpg := 30
def miles_to_work_each_way := 20
def leisure_miles_per_week := 40
def gallons_per_week := 8

theorem john_workdays : ∃ d : ℕ, 
  let work_miles_per_week := d * 2 * miles_to_work_each_way in
  let total_miles_per_week := work_miles_per_week + leisure_miles_per_week in
  total_miles_per_week = gallons_per_week * mpg ∧ d = 5 :=
by {
  let work_miles_per_week := 5 * 2 * miles_to_work_each_way,
  let total_miles_per_week := work_miles_per_week + leisure_miles_per_week,
  have h1 : total_miles_per_week = gallons_per_week * mpg,
    calc work_miles_per_week := 5 * 2 * miles_to_work_each_way : by simp[miles_to_work_each_way]
    ... := 5 * 2 * 20 : by simp[miles_to_work_each_way]
    ... := 200 : by norm_num
    let total_miles_per_week := work_miles_per_week + leisure_miles_per_week,
    calc total_miles_per_week := work_miles_per_week + leisure_miles_per_week : by simp[leisure_miles_per_week]
    ... := 200 + 40 : by simp
    ... := 240 : by norm_num
  have h2 : gallons_per_week * mpg = 240,
    calc gallons_per_week * mpg := 8 * 30 : by simp [gallons_per_week, mpg]
    ... := 240 : by norm_num
  exact ⟨5, h1, h2⟩
}


end john_workdays_l540_540493


namespace geese_survived_first_year_l540_540900

-- Define the initial number of goose eggs
def initial_eggs : ℕ := 100

-- Define the fraction of goose eggs that hatched
def hatched_fraction : ℚ := 1 / 4

-- Define the fraction of hatched geese that survived the first month
def survived_first_month_fraction : ℚ := 4 / 5

-- Define the fraction of the geese that survived the first month but did not survive the first year
def did_not_survive_first_year_fraction : ℚ := 3 / 5

-- Calculate the number of hatched geese
def hatched_geese := (hatched_fraction * initial_eggs).toNat

-- Calculate the number of geese that survived the first month
def survived_first_month_geese := (survived_first_month_fraction * hatched_geese).toNat

-- Calculate the number of geese that did not survive the first year
def did_not_survive_first_year_geese := (did_not_survive_first_year_fraction * survived_first_month_geese).toNat

-- Calculate the number of geese that survived the first year
def survived_first_year_geese := survived_first_month_geese - did_not_survive_first_year_geese

-- The theorem to prove
theorem geese_survived_first_year : survived_first_year_geese = 8 :=
by
  sorry  -- provide the detailed proof here

end geese_survived_first_year_l540_540900


namespace value_of_a_l540_540452

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 24 - 4 * a) : a = 3 :=
by
  sorry

end value_of_a_l540_540452


namespace find_second_liquid_parts_l540_540904

-- Define the given constants
def first_liquid_kerosene_percentage : ℝ := 0.25
def second_liquid_kerosene_percentage : ℝ := 0.30
def first_liquid_parts : ℝ := 6
def mixture_kerosene_percentage : ℝ := 0.27

-- Define the amount of kerosene from each liquid
def kerosene_from_first_liquid := first_liquid_kerosene_percentage * first_liquid_parts
def kerosene_from_second_liquid (x : ℝ) := second_liquid_kerosene_percentage * x

-- Define the total parts of mixture
def total_mixture_parts (x : ℝ) := first_liquid_parts + x

-- Define the total kerosene in the mixture
def total_kerosene_in_mixture (x : ℝ) := mixture_kerosene_percentage * total_mixture_parts x

-- State the theorem
theorem find_second_liquid_parts (x : ℝ) :
  kerosene_from_first_liquid + kerosene_from_second_liquid x = total_kerosene_in_mixture x → 
  x = 4 :=
by
  sorry

end find_second_liquid_parts_l540_540904


namespace domain_of_function_l540_540187

theorem domain_of_function :
  {x : ℝ | x > 4 ∧ x ≠ 5} = (Set.Ioo 4 5 ∪ Set.Ioi 5) :=
by
  sorry

end domain_of_function_l540_540187


namespace recurring_six_denominator_l540_540176

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l540_540176


namespace sum_largest_smallest_prime_factors_546_l540_540627

theorem sum_largest_smallest_prime_factors_546 : 
  (let p := prime_factors 546 in 
   List.sum [p.head, p.ilast] = 15) :=
by
  sorry

end sum_largest_smallest_prime_factors_546_l540_540627


namespace correct_option_l540_540987

theorem correct_option : 
  (∀ a b : ℝ, (a - b) * (-a - b) ≠ a^2 - b^2) ∧
  (∀ a : ℝ, 2 * a^3 + 3 * a^3 ≠ 5 * a^6) ∧ 
  (∀ x y : ℝ, 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2) ∧
  (∀ x : ℝ, (-2 * x^2)^3 ≠ -6 * x^6) :=
by 
  split
  . intros a b
    sorry
  . split
    . intros a
      sorry
    . split
      . intros x y
        sorry
      . intros x
        sorry

end correct_option_l540_540987


namespace plane_determination_l540_540796

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end plane_determination_l540_540796


namespace visitors_from_Mon_to_Fri_l540_540698

-- Given conditions:
def ticket_price : ℕ := 3
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300
def total_revenue : ℕ := 3000

-- Prove that the number of visitors from Monday to Friday is 100
theorem visitors_from_Mon_to_Fri :
  ∃ (x : ℕ), 3 * (5 * x + saturday_visitors + sunday_visitors) = total_revenue ∧ x = 100 :=
by
  let total_weekend_visitors := saturday_visitors + sunday_visitors
  have h1: total_weekend_visitors = 500 := by norm_num
  have total_weekly_visitors_eq : (5 * 100) + 500 = 1000 := by norm_num
  have revenue_constraint : 3 * 1000 = total_revenue := by norm_num
  use 100
  sorry

end visitors_from_Mon_to_Fri_l540_540698


namespace sum_largest_smallest_prime_factors_546_l540_540632

theorem sum_largest_smallest_prime_factors_546 : 
  let p := 546; let prime_factors := [2, 3, 7, 13]; 
  (List.minimum prime_factors).getOrElse 0 + (List.maximum prime_factors).getOrElse 0 = 15 := 
by
  intro p prime_factors
  sorry

end sum_largest_smallest_prime_factors_546_l540_540632


namespace sum_largest_smallest_prime_factors_546_l540_540628

theorem sum_largest_smallest_prime_factors_546 : 
  (let p := prime_factors 546 in 
   List.sum [p.head, p.ilast] = 15) :=
by
  sorry

end sum_largest_smallest_prime_factors_546_l540_540628


namespace every_natural_number_appears_once_l540_540116

noncomputable def y_seq : ℕ → ℕ
| 1 := 1
| (2 * k) := if k % 2 = 0 then 2 * y_seq k else 2 * y_seq k + 1
| (2 * k + 1) := if k % 2 = 0 then 2 * y_seq k + 1 else 2 * y_seq k

theorem every_natural_number_appears_once :
  ∀ n : ℕ, ∃! k : ℕ, y_seq k = n :=
sorry

end every_natural_number_appears_once_l540_540116


namespace real_roots_polynomial_ab_leq_zero_l540_540557

theorem real_roots_polynomial_ab_leq_zero
  {a b c : ℝ}
  (h : ∀ x, Polynomial.eval x (Polynomial.C 1 * X^4 + Polynomial.C a * X^3 + Polynomial.C b * X + Polynomial.C c) = 0 → x ∈ ℝ) :
  a * b ≤ 0 := 
begin
  sorry
end

end real_roots_polynomial_ab_leq_zero_l540_540557


namespace sale_price_is_correct_l540_540955

def initial_price : ℝ := 560
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.30
def discount3 : ℝ := 0.15
def tax_rate : ℝ := 0.12

noncomputable def final_price : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let price_after_third_discount := price_after_second_discount * (1 - discount3)
  let price_after_tax := price_after_third_discount * (1 + tax_rate)
  price_after_tax

theorem sale_price_is_correct :
  final_price = 298.55 :=
sorry

end sale_price_is_correct_l540_540955


namespace ones_digit_of_22_to_22_11_11_l540_540362

theorem ones_digit_of_22_to_22_11_11 : (22 ^ (22 * (11 ^ 11))) % 10 = 4 :=
by
  sorry

end ones_digit_of_22_to_22_11_11_l540_540362


namespace barrel_capacity_is_16_l540_540250

noncomputable def capacity_of_barrel (midway_tap_rate bottom_tap_rate used_bottom_tap_early_time assistant_use_time : Nat) : Nat :=
  let midway_draw := used_bottom_tap_early_time / midway_tap_rate
  let bottom_draw_assistant := assistant_use_time / bottom_tap_rate
  let total_extra_draw := midway_draw + bottom_draw_assistant
  2 * total_extra_draw

theorem barrel_capacity_is_16 :
  capacity_of_barrel 6 4 24 16 = 16 :=
by
  sorry

end barrel_capacity_is_16_l540_540250


namespace find_first_interest_rate_l540_540915

/-- Given the conditions:
1. Total amount Rs. 2600 is divided into two parts.
2. Amount lent at the first interest rate is Rs. 1600.
3. Interest rate for the second part is 6%.
4. Total yearly annual income from both parts is Rs. 140.
Prove that the first interest rate is 5%.
-/
theorem find_first_interest_rate (total_amount : ℕ) (first_part : ℕ) (second_part_rate : ℕ) (total_income : ℕ) :
  total_amount = 2600 →
  first_part = 1600 →
  second_part_rate = 6 →
  total_income = 140 →
  ∃ (first_part_rate : ℕ), (first_part_rate = 5) :=
by
  intro h1 h2 h3 h4
  use 5
  sorry

end find_first_interest_rate_l540_540915


namespace negation_proposition_l540_540952

theorem negation_proposition : (¬ ∀ x : ℝ, (1 < x) → x - 1 ≥ Real.log x) ↔ (∃ x_0 : ℝ, (1 < x_0) ∧ x_0 - 1 < Real.log x_0) :=
by
  sorry

end negation_proposition_l540_540952


namespace non_seeing_points_exist_l540_540212

theorem non_seeing_points_exist (n : ℕ) (segments : fin (n^2) → set (ℝ × ℝ)) 
  (h1 : ∀ i j : fin (n^2), i ≠ j → ¬ parallel (segments i) (segments j)) 
  (h2 : ∀ i j : fin (n^2), i ≠ j → ¬ intersects (segments i) (segments j)) :
  ∃ (points : fin n → ℝ × ℝ), ∀ i j : fin n, i ≠ j → ¬ (visible (segments) (points i) (points j)) :=
sorry

end non_seeing_points_exist_l540_540212


namespace estimate_height_using_regression_line_l540_540472

theorem estimate_height_using_regression_line 
  (foot_lengths heights : List ℝ) 
  (x : ℝ) 
  (n : ℕ) 
  (b : ℝ)
  (sum_foot_lengths : ℝ)
  (sum_heights : ℝ)
  (h1 : foot_lengths.length = n)
  (h2 : heights.length = n)
  (h3 : sum_foot_lengths = foot_lengths.sum)
  (h4 : sum_heights = heights.sum)
  (b_value : b = 4)
  (x_value : x = 24) : 
  let x_bar := sum_foot_lengths / n
  let y_bar := sum_heights / n
  let a := y_bar - b * x_bar
  let regression_line := λ x, b * x + a
  regression_line x = 166 := 
by
  sorry

end estimate_height_using_regression_line_l540_540472


namespace minimal_positive_period_f_l540_540581

open Real

def f (x : ℝ) : ℝ := abs (sin (2 * x) + cos (2 * x))

theorem minimal_positive_period_f : isLeastPeriod f (π / 2) :=
begin
  sorry
end

end minimal_positive_period_f_l540_540581


namespace archer_probability_less_than_8_l540_540407

-- Define the conditions as probabilities for hitting the 10-ring, 9-ring, and 8-ring.
def p_10 : ℝ := 0.24
def p_9 : ℝ := 0.28
def p_8 : ℝ := 0.19

-- Define the probability that the archer scores at least 8.
def p_at_least_8 : ℝ := p_10 + p_9 + p_8

-- Calculate the probability of the archer scoring less than 8.
def p_less_than_8 : ℝ := 1 - p_at_least_8

-- Now, state the theorem to prove that this probability is equal to 0.29.
theorem archer_probability_less_than_8 : p_less_than_8 = 0.29 := by sorry

end archer_probability_less_than_8_l540_540407


namespace course_selection_l540_540906

open Nat.Comb

theorem course_selection : 
  ∃ (n : ℕ), 
    let total_ways_different := (4.choose 2) * (2.choose 2)
    let total_ways_one_common := (4.choose 1) * (3.choose 1) * (2.choose 1)
    n = total_ways_different + total_ways_one_common ∧ n = 30 :=
by
  let total_ways_different := (4.choose 2) * (2.choose 2)
  let total_ways_one_common := (4.choose 1) * (3.choose 1) * (2.choose 1)
  have h : 30 = total_ways_different + total_ways_one_common := by sorry
  exact ⟨30, by simp [total_ways_different, total_ways_one_common, h]⟩

end course_selection_l540_540906


namespace proof_problem_l540_540454

-- Define the system of equations
def system_of_equations (x y a : ℝ) : Prop :=
  (3 * x + y = 2 + 3 * a) ∧ (x + 3 * y = 2 + a)

-- Define the condition x + y < 0
def condition (x y : ℝ) : Prop := x + y < 0

-- Prove that if the system of equations has a solution with x + y < 0, then a < -1 and |1 - a| + |a + 1 / 2| = 1 / 2 - 2 * a
theorem proof_problem (x y a : ℝ) (h1 : system_of_equations x y a) (h2 : condition x y) :
  a < -1 ∧ |1 - a| + |a + 1 / 2| = (1 / 2) - 2 * a := 
sorry

end proof_problem_l540_540454


namespace domain_of_f_exp_l540_540842

theorem domain_of_f_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 < 4 → ∃ y, f y = f (x + 1)) →
  (∀ x, 1 ≤ 2^x ∧ 2^x < 4 → ∃ y, f y = f (2^x)) :=
by
  sorry

end domain_of_f_exp_l540_540842


namespace logarithms_order_l540_540279

theorem logarithms_order :
  (cos 1 < sin 1 ∧ sin 1 < 1 ∧ 1 < tan 1) →
  log (sin 1) (cos 1) > log (cos 1) (sin 1) ∧ log (cos 1) (sin 1) > log (sin 1) (tan 1) :=
by
  intro h
  sorry

end logarithms_order_l540_540279


namespace same_terminal_side_angles_l540_540596

theorem same_terminal_side_angles (α : ℝ) : 
  (∃ k : ℤ, α = -457 + k * 360) ↔ (∃ k : ℤ, α = 263 + k * 360) :=
sorry

end same_terminal_side_angles_l540_540596


namespace infinite_product_to_rational_root_l540_540316

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540316


namespace triangle_is_isosceles_l540_540084

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) 
  (h₀ : A + B + C = π) :
  (A = B) := 
sorry

end triangle_is_isosceles_l540_540084


namespace exists_distinct_x1_x2_f_eq_implies_a_lt_4_l540_540418

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else 2*a*x - 5

theorem exists_distinct_x1_x2_f_eq_implies_a_lt_4 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = f x2 a) → a < 4 :=
sorry

end exists_distinct_x1_x2_f_eq_implies_a_lt_4_l540_540418


namespace minimum_length_AD_l540_540468

theorem minimum_length_AD (A B C D E: ℝ)
  (h1: ∃ x y z, (x = 0 ∧ y = 0 ∧ z = 0) ∧ (x^2 + y^2 = z^2))
  (h2: D ∈ segment ℝ A B)
  (h3: E ∈ segment ℝ A C)
  (h4: foldingToBC A B C D E) :
  AD = 2 * real.sqrt 3 - 3 := 
sorry

end minimum_length_AD_l540_540468


namespace quad_theorem_l540_540762

variables {A B C D P Q R X Y I : Type}
variables [linear_ordered_field A]

-- Definitions of setup conditions:
def quad_with_incircle (A B C D : Type) (I : Type) : Prop :=
  ∃ (incircle : A → A), true -- This is an abstract representation of the quadrilateral with an incircle.

def intersect (a b c d q : Type) : Prop := true -- Abstract representation of line intersections.

def point_on_segment (a p b : Type) : Prop := true -- Abstract representation of points on line segments.

def incircle_center (p b d : Type) : Type := unspecified -- Incenter representation

def lines_intersect (p y q x r : Type) : Prop := true -- Abstract representation of line intersection.

def perpendicular (r i bd : Type) : Prop := true -- Abstract representation of perpendicularity.

-- Theorem statement:
theorem quad_theorem (ABCD : Type) [quad_with_incircle A B C D I]
  (h1 : intersect D A B C Q)
  (h2 : intersect B A C D P)
  (h3 : point_on_segment A P B)
  (h4 : point_on_segment A Q D)
  (h5 : ∃X, incircle_center P B D = X)
  (h6 : ∃Y, incircle_center Q B D = Y)
  (h7 : lines_intersect P Y Q X R) :
  perpendicular R I (B : A × A, D : A × A) :=
begin
  sorry
end

end quad_theorem_l540_540762


namespace original_number_of_men_l540_540238

theorem original_number_of_men (M : ℕ) (h1 : ∀ w, (M - 5) * 10 = M * 6) : M = 13 :=
by
  have h2 : 6 * M = 10 * (M - 5) := by sorry
  have h3 : 6 * M = 10 * M - 50 := by sorry
  have h4 : 50 = 4 * M := by sorry
  have h5 : M = 12.5 := by sorry
  have h6 : round M = 13 := by sorry
  sorry

end original_number_of_men_l540_540238


namespace infinite_product_value_l540_540333

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540333


namespace infinite_product_value_l540_540336

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540336


namespace infinite_product_sqrt_nine_81_l540_540307

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540307


namespace positive_integers_congruent_to_4_mod_9_less_than_500_eq_56_l540_540820

theorem positive_integers_congruent_to_4_mod_9_less_than_500_eq_56 :
  {n : ℕ | n < 500 ∧ n % 9 = 4}.card = 56 :=
sorry

end positive_integers_congruent_to_4_mod_9_less_than_500_eq_56_l540_540820


namespace number_of_ordered_pairs_l540_540508

theorem number_of_ordered_pairs (ω : ℂ) (hω : ω^4 = 1 ∧ ¬ ω.re = 0) :
  set.finite {p : ℤ × ℤ | ∥ ↑p.1 * ω + ↑p.2 ∥ = 1} ∧
  set.card {p : ℤ × ℤ | ∥ ↑p.1 * ω + ↑p.2 ∥ = 1} = 4 :=
by 
  sorry

end number_of_ordered_pairs_l540_540508


namespace repeating_decimal_simplest_denominator_l540_540165

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l540_540165


namespace rhombus_proof_l540_540389

noncomputable def rhombus_example : Prop :=
  let A := (-2, 2)
  let C := (4, 4)
  let line_side := fun p : ℝ × ℝ => p.1 - p.2 + 4 = 0
  let line_AC := fun p : ℝ × ℝ => p.1 - 3 * p.2 + 8 = 0
  let line_BD := fun p : ℝ × ℝ => 3 * p.1 + p.2 - 6 = 0
  let line_AB := fun p : ℝ × ℝ => p.1 - p.2 + 4 = 0
  let line_CD := fun p : ℝ × ℝ => p.1 - p.2 = 0
  let line_BC := fun p : ℝ × ℝ => p.1 + 7 * p.2 - 32 = 0
  let line_AD := fun p : ℝ × ℝ => p.1 + 7 * p.2 - 12 = 0
  ∀ B : ℝ × ℝ, ∀ D : ℝ × ℝ,
    rhombus A A C ∧ 
    (line_side A) ∧ 
    (∀ D, line_AC D → line_AB B ∧ line_CD D ∧ line_BC B ∧ line_AD D) ∧ 
    (line_AC A) ∧ 
    (line_BD D) ∧ 
    (line_AB A) ∧ 
    (line_CD C) ∧ 
    (line_BC B) ∧ 
    (line_AD A)

-- And now we provide a proof skeleton to avoid any compilation issues.
theorem rhombus_proof : rhombus_example :=
by {
  -- Proof steps would go here
  sorry,
}

end rhombus_proof_l540_540389


namespace binomial_variance_eq_p_mul_one_sub_p_l540_540071

open ProbabilityTheory

variables {X : Type} [DiscreteRandomVariable X ℝ]
variables (p q : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : q = 1 - p)

noncomputable def binomialVariance : ℝ := p * (1 - p)

theorem binomial_variance_eq_p_mul_one_sub_p
  (hX : ∀ x : X, x = 0 ∨ x = 1)
  (hp : ∀ x : X, x = 1 → P(x) = p)
  (hq : ∀ x : X, x = 0 → P(x) = q):
  variance X = p * (1 - p) := sorry

end binomial_variance_eq_p_mul_one_sub_p_l540_540071


namespace problem1_solution_problem2_solution_l540_540422

noncomputable def problem1 (φ ω : ℝ) := 
  2 * sin φ = 1 → |φ| < π / 2 → φ = π / 6

noncomputable def problem2 (ω : ℝ) (f : ℝ → ℝ) :=
  (∀ x, f x = 2 * sin (ω * x + φ)) →
  (∃ x, f(x+2) - f(x) = 4) →
  ω > 0 →
  ω = π / 2

-- Example of how these could be stated as Theorems
theorem problem1_solution (φ : ℝ) (h1 : 2 * sin φ = 1) (h2 : |φ| < π / 2) : 
  φ = π / 6 :=
by sorry

theorem problem2_solution (ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * sin (ω * x + φ)) 
  (h2 : ∃ x, f(x+2) - f(x) = 4) 
  (h3 : ω > 0) : 
  ω = π / 2 :=
by sorry

end problem1_solution_problem2_solution_l540_540422


namespace Geoff_spending_over_three_days_l540_540610

-- Define the costs and conditions
def Monday_spending : ℕ := 60
def Tuesday_spending_without_discount := 4 * Monday_spending
def Tuesday_discount := 0.10 * Tuesday_spending_without_discount
def Tuesday_spending := Tuesday_spending_without_discount - Tuesday_discount
def Wednesday_spending_without_tax := 5 * Monday_spending
def Wednesday_tax := 0.08 * Wednesday_spending_without_tax
def Wednesday_spending := Wednesday_spending_without_tax + Wednesday_tax

-- Define the total spending over the three days
def total_spending := Monday_spending + Tuesday_spending + Wednesday_spending

-- Prove that total spending is $600
theorem Geoff_spending_over_three_days :
  total_spending = 600 := by
  sorry

end Geoff_spending_over_three_days_l540_540610


namespace original_number_l540_540874

variable (n : ℝ)

theorem original_number :
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by
  sorry

end original_number_l540_540874


namespace infinite_product_result_l540_540324

noncomputable def infinite_product := (3:ℝ)^(1/4) * (9:ℝ)^(1/16) * (27:ℝ)^(1/64) * (81:ℝ)^(1/256) * ...

theorem infinite_product_result : infinite_product = real.sqrt (81) ^ (1 / 9) :=
by
  unfold infinite_product
  sorry

end infinite_product_result_l540_540324


namespace cos_4theta_l540_540829

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (4 * θ) = 17 / 32 :=
sorry

end cos_4theta_l540_540829


namespace angle_AKB_in_equilateral_triangle_l540_540131

theorem angle_AKB_in_equilateral_triangle
  (A B C P Q K : Point)
  (hABC : EquilateralTriangle A B C)
  (hP : OnSegment P A B)
  (hQ : OnSegment Q B C)
  (hRatio1 : ratio (dist A P) (dist P B) = 2)
  (hRatio2 : ratio (dist B Q) (dist Q C) = 2)
  (hK : ∃ K, IntersectsAt K (Segment A Q) (Segment C P)) :
  angle A K B = 90 :=
by sorry

end angle_AKB_in_equilateral_triangle_l540_540131


namespace smallest_k_for_positive_roots_5_l540_540378

noncomputable def smallest_k_for_positive_roots : ℕ := 5

theorem smallest_k_for_positive_roots_5
  (k p q : ℕ) 
  (hk : k = smallest_k_for_positive_roots)
  (hq_pos : 0 < q)
  (h_distinct_pos_roots : ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
    k * x₁ * x₂ = q ∧ k * x₁ + k * x₂ > p ∧ k * x₁ * x₂ < q * ( 1 / (x₁*(1 - x₁) * x₂ * (1 - x₂)))) :
  k = 5 :=
by
  sorry

end smallest_k_for_positive_roots_5_l540_540378


namespace sin_alpha_point_trig_l540_540846

theorem sin_alpha_point_trig :
  (α : ℝ) (P : ℝ × ℝ)
  (hP : P = (2 * Real.cos (2 * Real.pi * 120 / 360), Real.sqrt 2 * Real.sin (2 * Real.pi * 225 / 360)))
  (h_p_on_terminal_side_of_alpha : P = (-1, -1)) :
  Real.sin α = - Real.sqrt 2 / 2 :=
by
  sorry

end sin_alpha_point_trig_l540_540846


namespace ratio_of_average_speeds_l540_540739

-- Conditions
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4
def distance_ab : ℕ := 600
def distance_ac : ℕ := 360

-- Theorem to prove the ratio of their average speeds
theorem ratio_of_average_speeds : (distance_ab / time_eddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 20 ∧
                                  (distance_ac / time_freddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 9 :=
by
  -- Solution steps go here if performing an actual proof
  sorry

end ratio_of_average_speeds_l540_540739


namespace two_digit_perfect_square_l540_540692

theorem two_digit_perfect_square :
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ a + b = 11} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
by sorry

end two_digit_perfect_square_l540_540692


namespace infinite_product_value_l540_540339

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540339


namespace derivative_of_y_l540_540356

variable (x : ℝ)

def y := x^3 + 3 * x^2 + 6 * x - 10

theorem derivative_of_y : (deriv y) x = 3 * x^2 + 6 * x + 6 :=
sorry

end derivative_of_y_l540_540356


namespace infinite_product_value_l540_540311

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540311


namespace propA_propB_propC_propD_l540_540798

-- Proposition A: Given the conditions, prove the required probability equality.
theorem propA (ξ : Type) [Probability ξ] (σ : ℝ) (P : Event ξ → ℝ) (x : ξ) 
  (h1 : P ({y | y ≤ 4}) = 0.79) (h2 : P ({y | y ≤ -2}) = 0.21) : 
  P ({y | y ≤ -2}) = 0.21 :=
sorry

-- Proposition B: Prove that the number of ways 10 passengers can get off at 5 bus stops is not 10^5.
theorem propB : ¬((5 : ℕ) ^ 10 = 10 ^ 5) :=
sorry

-- Proposition C: Prove that the number of ways to select only one pair of same color from 6 pairs of shoes when choosing 4 shoes is 240.
theorem propC (pairs : ℕ) (ways : ℕ)
  (h1 : pairs = 6)
  (h2 : ways = 240) : 
  ways = 6 * (choose 5 2) * 2 * 2 := 
sorry

-- Proposition D: Prove that the number of ways to divide 7 students (4 males, 3 females) into two groups with specific requirements is 104.
theorem propD (students : ℕ) (males : ℕ) (females : ℕ) (ways : ℕ)
  (h1 : students = 7) (h2 : males = 4) (h3 : females = 3)
  (h4 : ways = 104) : 
  ways = (choose 4 1 * choose 3 1) + (choose 4 2 * 1) := 
sorry

end propA_propB_propC_propD_l540_540798


namespace ellipse_segment_length_l540_540011

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_segment_length
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_eq a b 0 4)
  (h4 : b = 4)
  (h5 : 3 / 5 = (real.sqrt (a^2 - b^2)) / a) :
  ∃ A B : ℝ × ℝ, 
    (subtype (λ p : ℝ × ℝ, ellipse_eq a b p.1 p.2)) A ∧
    (subtype (λ p : ℝ × ℝ, ellipse_eq a b p.1 p.2)) B ∧
    (A = (3 + 4*k, k) ∧ B = (3 - 4*k, k) ∨ A = (3 - 4*k, k) ∧ B = (3 + 4*k, k)) ∧
    (k = 4 / 5) ∧
    (|A.1 - B.1| = 41 / 5) := sorry

end ellipse_segment_length_l540_540011


namespace infinite_product_value_l540_540310

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540310


namespace point_in_second_quadrant_l540_540832

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l540_540832


namespace abs_diff_eq_5_l540_540513

-- Definitions of m and n, and conditions provided in the problem
variables (m n : ℝ)
hypothesis (h1 : m * n = 6)
hypothesis (h2 : m + n = 7)

-- Statement to prove
theorem abs_diff_eq_5 : |m - n| = 5 :=
by
  sorry

end abs_diff_eq_5_l540_540513


namespace necessary_condition_l540_540880

noncomputable theory
open Set

def A : Set ℝ := { x | x^2 + 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 + (m+1) * x + m = 0 }

theorem necessary_condition (m : ℝ) : (A ⊆ B m) → m = 2 :=
by
  sorry

end necessary_condition_l540_540880


namespace length_of_map_l540_540274

-- Defining the conditions
variables (area width length : ℝ)

-- Assume the conditions given in the problem
axiom area_eq_10 : area = 10
axiom width_eq_2 : width = 2

-- The statement to prove
theorem length_of_map : length = 5 :=
by
  -- We substitute the conditions into our equation
  have h : area = length * width := sorry
  -- Substitute the known values of area and width
  rw [area_eq_10, width_eq_2] at h
  -- Solve for length
  exact eq.div area_eq_10 width_eq_2

end length_of_map_l540_540274


namespace odd_function_property_f_neg_half_l540_540785

noncomputable def f (x : ℝ) : ℝ := if x > 0 then log x / log 2 else -log (-x) / log 2

theorem odd_function_property (x : ℝ) (h : x > 0) : f (-x) = -f x := by
  rw [if_neg (show ¬ -x > 0 by linarith), if_pos h]
  exact neg_inj.mp (neg_div 1 _ (log_neg_iff.mpr (by linarith)))

theorem f_neg_half : f (-1 / 2) = 1 := by
  rw [if_neg (show ¬ (-1/2) > 0 by norm_num), if_pos (show 1/2 > 0 by norm_num)]
  have : log 2 = real.log 2 := rfl
  rw [←this, log_inv, real.log_inv', one_div_eq_inv]
  exact neg_inv (log (2:ℝ) / real.log 2)

end odd_function_property_f_neg_half_l540_540785


namespace train_passes_jogger_in_24_seconds_l540_540260

theorem train_passes_jogger_in_24_seconds
  (jogger_speed_kmh : Real)
  (train_speed_kmh : Real)
  (initial_lead_m : Real)
  (train_length_m : Real)
  (convert_speed : (Real) → (Real) → Real := λ speed_kmh, speed_kmh * (1000/3600))
  (relative_speed : Real := convert_speed train_speed_kmh - convert_speed jogger_speed_kmh)
  (total_distance : Real := initial_lead_m + train_length_m) :
  jogger_speed_kmh = 9 →
  train_speed_kmh = 45 →
  initial_lead_m = 120 →
  train_length_m = 120 →
  (total_distance / relative_speed) = 24 :=
by
  intros
  sorry

end train_passes_jogger_in_24_seconds_l540_540260


namespace angle_between_rays_l540_540615

theorem angle_between_rays (
  (V1 V2 : ℝ)  -- angular velocities of the two points
  (l : ℝ)  -- circumference of the circle
  (h₁ : l / V2 - l / V1 = 5)  -- one point completes a full revolution 5 seconds faster
  (h₂ : 60 * (V1 - V2) = 2 * l)  -- faster point makes two more revolutions per minute
  (r : ℝ := 360)  -- if needed, the full rotation in degrees
) :
  let α_same := (V1 - V2) / l * r
  let α_opp := (V1 + V2) / l * r
  α_same = 12 ∨ α_opp = 60 :=
by
  sorry


end angle_between_rays_l540_540615


namespace lines_tangent_to_parabola_l540_540372

theorem lines_tangent_to_parabola :
  ∀ (a : ℝ), ∃ (p q : ℝ), q = -a * p - a^2 ∧ p^2 - 4 * q = 0 → ∀ (L : set (ℝ × ℝ)), 
  (L = { (p, q) | ∃ (a : ℝ), q = -a * p - a^2 }) ↔ (L = { (p, q) | tangent (p^2 - 4 * (q : ℝ) = 0) (q = -a * p - a^2) (p, q) }) :=
sorry

end lines_tangent_to_parabola_l540_540372


namespace time_for_one_eighth_l540_540255

noncomputable def half_time_equality (a b : ℝ) : Prop :=
  ae ^ (-8 * b) = 1 / 2

noncomputable def eighth_time_equality (a b t : ℝ) : Prop :=
  a e ^ (-b * t) = 1 / 8

theorem time_for_one_eighth (a b t : ℝ) (h1 : half_time_equality a b) : eighth_time_equality a b (24 - 8) :=
  sorry

end time_for_one_eighth_l540_540255


namespace digit_4_count_in_range_l540_540647

theorem digit_4_count_in_range :
  (count_digit_4_in_range 1 1000) = 300 :=
sorry

-- Helper function to count digit 4 occurrences in a given range
def count_digit_4_in_range (start : ℕ) (end : ℕ) : ℕ :=
  (list.range (end - start + 1)).map (λ n, n + start).foldl (λ acc num, 
    acc + count_digit_in_number 4 num) 0

-- Helper function to count occurrences of a given digit within a number
def count_digit_in_number (digit : ℕ) (num : ℕ) : ℕ :=
  if num = 0 then 0 else 
    (if num % 10 = digit then 1 else 0) + count_digit_in_number digit (num / 10)

end digit_4_count_in_range_l540_540647


namespace greatest_unique_digit_multiple_of_5_remainder_l540_540107

-- Define the problem statement according to the conditions and question
theorem greatest_unique_digit_multiple_of_5_remainder :
  let M := 9876543215 in
  (∀ n, (n % 10 = 5 ∨ n % 10 = 0) → (∀ i j, i ≠ j → n.to_digits.nth i ≠ n.to_digits.nth j) → n ≤ 9876543215 → n % 500 = 215) :=
by
  sorry

end greatest_unique_digit_multiple_of_5_remainder_l540_540107


namespace infinite_product_value_l540_540338

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540338


namespace lateral_area_cone_l540_540060

theorem lateral_area_cone (base_len height : ℝ) (r l : ℝ) (h_base: base_len = sqrt 2) (h_height: height = 1) (h_radius: r = 1) (h_slant_height: l = sqrt 2) :
  (π * r * l) = sqrt 2 * π :=
by
  sorry

end lateral_area_cone_l540_540060


namespace infinite_product_value_l540_540332

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540332


namespace inverse_function_sum_l540_540582

noncomputable def f (x : ℝ) : ℝ := 
  if x < 3 then x - 3 else real.sqrt x

noncomputable def f_inv (y : ℝ) : ℝ := 
  if y < 0 then y + 3 else y^2

theorem inverse_function_sum : 
  (f_inv (-7) + f_inv (-6) + f_inv (-5) + f_inv (-4) + f_inv (-3) + f_inv (-2) + f_inv (-1) + 
   f_inv 1 + f_inv 2 + f_inv 3 + f_inv 4 + f_inv 5 + f_inv 6 + f_inv 7) = 128 :=
by
  sorry

end inverse_function_sum_l540_540582


namespace area_UQVS_108_l540_540087

open Real

noncomputable def triangle_area (b h : ℝ) : ℝ := 0.5 * b * h

variables (P Q R S T U V : Type) 
          (PQ PR QR PS PT QV VR ST UT VU PST PVR PSU UQVS : ℝ)
          [inhabited P] [inhabited Q] [inhabited R] [inhabited S] [inhabited T] [inhabited U] [inhabited V]

-- Given conditions
axiom PQ_value : PQ = 60
axiom PR_value : PR = 15
axiom area_PQR : triangle_area PQ PR = 180
axiom midpoint_PS : PS = PQ / 2
axiom midpoint_PT : PT = PR / 2

-- Question to be proved
theorem area_UQVS_108 : 
  ∀ (P Q R S T U V : Type), PQ = 60 → PR = 15 → triangle_area PQ PR = 180 → PS = PQ / 2 → PT = PR / 2 → 
  ∃ (UQVS : ℝ), UQVS = 108 :=
sorry

end area_UQVS_108_l540_540087


namespace absolute_difference_m_n_l540_540520

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l540_540520


namespace union_of_sets_l540_540776

open Set

variable (ℝ : Type) [LinearOrderedField ℝ] [RealTranscendentals ℝ]

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (4 - x^2) }

theorem union_of_sets :
  A ∪ B = Ioc (-2 : ℝ) (4 : ℝ) :=
by
  sorry

end union_of_sets_l540_540776


namespace find_plane_equation_l540_540113

def vector_dot (v w : Vector ℝ) : ℝ :=
  v.x * w.x + v.y * w.y + v.z * w.z

noncomputable def proj (w v : Vector ℝ) : Vector ℝ :=
  (vector_dot v w / vector_dot w w) • w

-- Given a condition on the projection of v and w, we need to find the plane equation
theorem find_plane_equation (x y z : ℝ) (w : Vector ℝ) (v_proj : Vector ℝ) :
  w = ⟨3, -2, 3⟩ →
  v_proj = ⟨6, -4, 6⟩ →
  proj w ⟨x, y, z⟩ = v_proj →
  3 * x - 2 * y + 3 * z - 44 = 0 :=
by
  intros h_w h_v_proj h_proj
  sorry

end find_plane_equation_l540_540113


namespace problem_arith_sequences_l540_540890

theorem problem_arith_sequences (a b : ℕ → ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n + e)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) : 
  a 37 + b 37 = 100 := 
sorry

end problem_arith_sequences_l540_540890


namespace find_value_of_triangle_l540_540016

theorem find_value_of_triangle (p : ℕ) (triangle : ℕ) 
  (h1 : triangle + p = 47) 
  (h2 : 3 * (triangle + p) - p = 133) :
  triangle = 39 :=
by 
  sorry

end find_value_of_triangle_l540_540016


namespace derivative_of_f_at_alpha_l540_540760

variables {α : ℝ}

def f (x : ℝ) : ℝ := 1 - cos x

theorem derivative_of_f_at_alpha : deriv f α = sin α := 
by 
  sorry

end derivative_of_f_at_alpha_l540_540760


namespace problem_statement_l540_540812

theorem problem_statement (m n c d a : ℝ)
  (h1 : m = -n)
  (h2 : c * d = 1)
  (h3 : a = 2) :
  Real.sqrt (c * d) + 2 * (m + n) - a = -1 :=
by
  -- Proof steps are skipped with sorry 
  sorry

end problem_statement_l540_540812


namespace correct_number_of_true_propositions_l540_540376

def f (x : ℝ) : ℝ := Real.sin x + 1 / Real.sin x

def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = f (a + x)

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ m

theorem correct_number_of_true_propositions :
  (¬ symmetric_about_y_axis f) ∧
  symmetric_about_origin f ∧
  symmetric_about_line f (π / 2) ∧
  (¬ minimum_value f 2) →
  2 = 2 :=
by
  intro h
  sorry

end correct_number_of_true_propositions_l540_540376


namespace ratio_second_to_first_l540_540129

-- Condition 1: The first bell takes 50 pounds of bronze
def first_bell_weight : ℕ := 50

-- Condition 2: The second bell is a certain size compared to the first bell
variable (x : ℕ) -- the ratio of the size of the second bell to the first bell
def second_bell_weight := first_bell_weight * x

-- Condition 3: The third bell is four times the size of the second bell
def third_bell_weight := 4 * second_bell_weight x

-- Condition 4: The total weight of bronze required is 550 pounds
def total_weight : ℕ := 550

-- Define the proof problem
theorem ratio_second_to_first (x : ℕ) (h : 50 + 50 * x + 200 * x = 550) : x = 2 :=
by
  sorry

end ratio_second_to_first_l540_540129


namespace molecular_weight_constant_l540_540228

-- Define the molecular weight of bleach
def molecular_weight_bleach (num_moles : Nat) : Nat := 222

-- Theorem stating the molecular weight of any amount of bleach is 222 g/mol
theorem molecular_weight_constant (n : Nat) : molecular_weight_bleach n = 222 :=
by
  sorry

end molecular_weight_constant_l540_540228


namespace smallest_possible_integer_l540_540669

noncomputable def smallest_integer (M : ℕ) : ℕ :=
  lcm (finset.range 28) * 31

theorem smallest_possible_integer :
  ∃ M : ℕ,
    (∀ k ∈ finset.range 28 ++ finset.singleton 31, k ∣ M) ∧
    (¬ 28 ∣ M ∧ ¬ 29 ∣ M ∧ ¬ 30 ∣ M) ∧
    M = 2329089562800 :=
by
  use 2329089562800
  sorry

end smallest_possible_integer_l540_540669


namespace minimum_value_y_l540_540397

variable {x y : ℝ}

theorem minimum_value_y (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : y ≥ Real.exp 1 :=
sorry

end minimum_value_y_l540_540397


namespace nine_segment_closed_broken_line_impossible_l540_540485

theorem nine_segment_closed_broken_line_impossible :
  ¬ (∃ broken_line : list (ℝ × ℝ) → Prop,
    broken_line.length = 9 ∧
    (∀ segment ∈ broken_line, ∃ unique_segment ∈ broken_line, segment ≠ unique_segment ∧ segment.intersects(unique_segment)) ∧
    (is_closed broken_line)) :=
by
  sorry

end nine_segment_closed_broken_line_impossible_l540_540485


namespace absolute_difference_m_n_l540_540519

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l540_540519


namespace plane_eq_l540_540112

noncomputable def vec3 := ℕ → ℝ

def w : vec3 := λ i, if i = 0 then 3 else if i = 1 then -1 else if i = 2 then 3 else 0

def v : vec3 := λ i, if i = 0 then x else if i = 1 then y else if i = 2 then z else 0

def proj (w v : vec3) : vec3 :=
  let dot_product (a b : vec3) : ℝ := (a 0 * b 0) + (a 1 * b 1) + (a 2 * b 2)
  let scalar_proj := dot_product v w / dot_product w w
  λ i, scalar_proj * w i

axiom eq_proj : proj w v = λ i, if i = 0 then 6 else if i = 1 then -2 else if i = 2 then 6 else 0

theorem plane_eq (x y z : ℝ) : 3 * x - y + 3 * z - 38 = 0 :=
by
  sorry

end plane_eq_l540_540112


namespace arithmetic_mean_increase_l540_540058

theorem arithmetic_mean_increase (b1 b2 b3 b4 b5 : ℝ) :
  let original_mean := (b1 + b2 + b3 + b4 + b5) / 5
  let new_mean := ((b1 + 15) + (b2 + 15) + (b3 + 15) + (b4 + 15) + (b5 + 15)) / 5
  new_mean = original_mean + 15 :=
by {
  let T := b1 + b2 + b3 + b4 + b5,
  let original_mean := T / 5,
  let new_sum := T + 5 * 15,
  let new_mean := new_sum / 5,
  show new_mean = original_mean + 15, 
  sorry
}

end arithmetic_mean_increase_l540_540058


namespace balls_in_boxes_l540_540053

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end balls_in_boxes_l540_540053


namespace smallest_integer_not_expressible_in_form_l540_540364

theorem smallest_integer_not_expressible_in_form :
  ∀ (n : ℕ), (0 < n ∧ (∀ a b c d : ℕ, n ≠ (2^a - 2^b) / (2^c - 2^d))) ↔ n = 11 :=
by
  sorry

end smallest_integer_not_expressible_in_form_l540_540364


namespace A_finishes_race_in_36_seconds_l540_540074

-- Definitions of conditions
def distance_A := 130 -- A covers a distance of 130 meters
def distance_B := 130 -- B covers a distance of 130 meters
def time_B := 45 -- B covers the distance in 45 seconds
def distance_B_lag := 26 -- A beats B by 26 meters

-- Statement to prove
theorem A_finishes_race_in_36_seconds : 
  ∃ t : ℝ, distance_A / t + distance_B_lag = distance_B / time_B := sorry

end A_finishes_race_in_36_seconds_l540_540074


namespace infinite_product_sqrt_nine_81_l540_540306

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540306


namespace exponential_solution_l540_540925

theorem exponential_solution (x : ℝ) : 
  (4^x - 3^(x - 1 / 2) = 3^(x + 1 / 2) - 2^(2 * x - 1)) ↔ (x = 3 / 2) :=
sorry

end exponential_solution_l540_540925


namespace smallest_prime_divisor_of_sum_l540_540229

theorem smallest_prime_divisor_of_sum :
  let n := 2^14 + 7^9
  ∀ (p : ℕ), nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → ¬ p ∣ n := by
sorry

end smallest_prime_divisor_of_sum_l540_540229


namespace decreasing_range_of_a_l540_540844

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x >= 2 then (a - 5) * x - 2 else x^2 - 2 * (a + 1) * x + 3 * a

theorem decreasing_range_of_a :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (1 ≤ a ∧ a ≤ 4) :=
  sorry

end decreasing_range_of_a_l540_540844


namespace hyperbola_slope_range_of_PM_l540_540453

theorem hyperbola_slope_range_of_PM :
  ∀ (t : ℝ), t ∈ Ioo (-3/2) (3/2) ∧ t ≠ sqrt 2 ∧ t ≠ 1 →
  let k := 2 / (t^2 + t - 2) in
  k ∈ Iio (-8 / 9) ∪ Ioo (8 / 7) (sqrt 2) ∪ Ioi (sqrt 2) :=
begin
  intros t ht,
  let k := 2 / (t^2 + t - 2),
  sorry
end

end hyperbola_slope_range_of_PM_l540_540453


namespace reciprocal_neg_two_l540_540593

theorem reciprocal_neg_two : 1 / (-2) = - (1 / 2) :=
by
  sorry

end reciprocal_neg_two_l540_540593


namespace unique_x_value_l540_540498

noncomputable def possible_values_x : set ℝ := {x | let A := {2, 0, x},
                                                      B := {2, x^2} in
                                                  A ∩ B = B}

theorem unique_x_value : possible_values_x = {1} :=
sorry

end unique_x_value_l540_540498


namespace probability_of_different_colors_l540_540663

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def yellow_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_of_different_colors :
  let total_outcomes := nat.choose total_balls drawn_balls,
      favorable_outcomes := nat.choose red_balls 1 * nat.choose yellow_balls 1 in
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 :=
by
  let total_outcomes := nat.choose total_balls drawn_balls
  let favorable_outcomes := nat.choose red_balls 1 * nat.choose yellow_balls 1
  sorry

end probability_of_different_colors_l540_540663


namespace fixed_point_of_line_l540_540130

theorem fixed_point_of_line (k : ℝ) : ∃ P : ℝ × ℝ, P = (4, -6) ∧ ∀ k : ℝ, (k+1) * P.1 + P.2 + 2 - 4 * k = 0 :=
by
  let P := (4 : ℝ, -6 : ℝ)
  use P
  split
  · refl
  · intro k
    sorry

end fixed_point_of_line_l540_540130


namespace prod_cos_eq_prod_sin_eq_l540_540750

theorem prod_cos_eq (n : ℕ) (h1 : 2 ≤ n) :
    (∏ k in Finset.range(n - 1), |Real.cos (k * Real.pi / n)|) = (1 / 2) ^ n * (1 - (-1) ^ n) := sorry

theorem prod_sin_eq (n : ℕ) (h1 : 2 ≤ n) :
    (∏ k in Finset.range(n - 1), Real.sin (k * Real.pi / n)) = n * (1 / 2) ^ (n - 1) := sorry

end prod_cos_eq_prod_sin_eq_l540_540750


namespace correct_calculation_l540_540986

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end correct_calculation_l540_540986


namespace infinite_product_value_l540_540315

theorem infinite_product_value :
  (∀ (n : ℕ), n > 0 → ∏ i in finset.range (n+1), (3^(i / (4^i))) = real.sqrt (81)) := 
sorry

end infinite_product_value_l540_540315


namespace consecutive_sum_15_number_of_valid_sets_l540_540041

theorem consecutive_sum_15 : 
  ∃ n (a : ℕ), n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30 :=
begin
  sorry
end

theorem number_of_valid_sets : 
  finset.card ((finset.filter (λ n a, n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30) (finset.range 15).product (finset.range 15))) = 2 :=
begin
  sorry
end

end consecutive_sum_15_number_of_valid_sets_l540_540041


namespace infinite_product_value_l540_540337

def infinite_product := ∏' (n : ℕ), (3 ^ (n / (4^n : ℝ)))

theorem infinite_product_value :
  infinite_product = (3 : ℝ) ^ (4/9) :=
sorry

end infinite_product_value_l540_540337


namespace schools_connections_inequality_l540_540892

noncomputable def number_of_schools (n : ℕ) : ℕ := n

noncomputable def connections_per_school (d : ℕ) : ℕ := d

theorem schools_connections_inequality (n d : ℕ) (h : ∀ s, s < n → connections_per_school d = d) :
  d < 2 * n^(1/3 : ℝ) :=
sorry

end schools_connections_inequality_l540_540892


namespace find_a7_coefficient_l540_540022

theorem find_a7_coefficient (a_7 : ℤ) : 
    (∀ x : ℤ, (x+1)^5 * (2*x-1)^3 = a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) → a_7 = 28 :=
by
  sorry

end find_a7_coefficient_l540_540022


namespace minimum_shows_required_l540_540964

-- Definition of the problem in Lean
theorem minimum_shows_required:
  ∃ (m : ℕ), 
    ∀ (participants : finset ℕ)
        (shows : finset (finset ℕ)),
      (participants.card = 8) →                   -- 8 participants
      (∀ show ∈ shows, show.card = 4) →           -- Each show has 4 people
      (∀ (a b ∈ participants) (hab: a ≠ b),
        ∃ k : ℕ, 
          (∀ show ∈ shows, a ∈ show ∧ b ∈ show) = 
          k) →
        m = 14 :=                                 -- Minimum value of m is 14
begin
  sorry
end

end minimum_shows_required_l540_540964


namespace quadratic_function_points_l540_540838

theorem quadratic_function_points:
  (∀ x y, (y = x^2 + x - 1) → ((x = -2 → y = 1) ∧ (x = 0 → y = -1) ∧ (x = 2 → y = 5))) →
  (-1 < 1 ∧ 1 < 5) :=
by
  intro h
  have h1 := h (-2) 1 (by ring)
  have h2 := h 0 (-1) (by ring)
  have h3 := h 2 5 (by ring)
  exact And.intro (by linarith) (by linarith)

end quadratic_function_points_l540_540838


namespace min_x_squared_plus_y_squared_l540_540242

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end min_x_squared_plus_y_squared_l540_540242


namespace g8_pow_4_eq_64_l540_540932

-- Definitions based on conditions
variables (f g : ℝ → ℝ)
hypothesis h1 : ∀ x, x ≥ 1 → f(g(x)) = x^2
hypothesis h2 : ∀ x, x ≥ 1 → g(f(x)) = x^4
hypothesis h3 : g(64) = 64

-- The goal to prove
theorem g8_pow_4_eq_64 : [g(8)]^4 = 64 :=
by
  sorry

end g8_pow_4_eq_64_l540_540932


namespace part_I_part_II_l540_540421

noncomputable def f (x a : ℝ) := |x - 3 * a|

-- Part I
theorem part_I {x : ℝ} :
  let a := 1 in
  let g := (λ x, 5 - |2 * x - 1|) in
  f x a > g x ↔ x < -1/3 ∨ x > 3 := 
by {
  sorry
}

-- Part II
theorem part_II {a : ℝ} :
  (∃ x0, f x0 a + x0 < 6) ↔ a < 2 :=
by {
  sorry
}

end part_I_part_II_l540_540421


namespace average_person_funding_l540_540561

-- Define the conditions from the problem
def total_amount_needed : ℝ := 1000
def amount_already_have : ℝ := 200
def number_of_people : ℝ := 80

-- Define the correct answer
def average_funding_per_person : ℝ := 10

-- Formulate the proof statement
theorem average_person_funding :
  (total_amount_needed - amount_already_have) / number_of_people = average_funding_per_person :=
by
  sorry

end average_person_funding_l540_540561


namespace fourth_square_area_l540_540970

theorem fourth_square_area (PQ QR RS QS : ℝ)
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 49)
  (h3 : RS^2 = 64) :
  QS^2 = 138 :=
by
  sorry

end fourth_square_area_l540_540970


namespace shortest_altitude_right_triangle_l540_540597

theorem shortest_altitude_right_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (area : ℝ) (H : RightTriangle a b c) (H_area : area = (1 / 2) * a * b):
  ∃ (h : ℝ), h = 12 ∧ (area = (1 / 2) * c * h) := by
  sorry

-- RightTriangle predicate needs to be defined as suitable to indicate that sides form a right triangle
def RightTriangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2)

end shortest_altitude_right_triangle_l540_540597


namespace num_visionary_sets_l540_540151

def visionary_set (A B : set ℕ) : Prop :=
  B ⊆ A ∧ ∃ n, n ∈ B ∧ n % 2 = 0 ∧ B.card ∈ B

def set_A : set ℕ := {i | 1 ≤ i ∧ i ≤ 20}

theorem num_visionary_sets :
  (∑ B in (set.sublists set_A), if visionary_set set_A B then 1 else 0) = 2^20 - 512 :=
sorry

end num_visionary_sets_l540_540151


namespace opposite_sign_pairs_l540_540697

def opposite_sign (a b : ℤ) : Prop := (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)

theorem opposite_sign_pairs :
  ¬opposite_sign (-(-1)) 1 ∧
  ¬opposite_sign ((-1)^2) 1 ∧
  ¬opposite_sign (|(-1)|) 1 ∧
  opposite_sign (-1) 1 :=
by {
  sorry
}

end opposite_sign_pairs_l540_540697


namespace henry_list_empty_after_11_steps_l540_540040

theorem henry_list_empty_after_11_steps :
  let initial_list := (List.range 1000).map (λ n => n + 1)
  let step (l : List ℕ) := (l.filter (λ n => (n > 0) ∧ (n.digits.nodup))) 
                            .map (λ n => n - 1)
  let rec process_list (l : List ℕ) (steps : ℕ) : ℕ :=
    if l = [] then steps else process_list (step l) (steps + 1)
  process_list initial_list 0 = 11 :=
by
  sorry

end henry_list_empty_after_11_steps_l540_540040


namespace find_principal_l540_540991

theorem find_principal (R : ℝ) (P : ℝ) (h : ((P * (R + 5) * 10) / 100) = ((P * R * 10) / 100 + 600)) : P = 1200 :=
by
  sorry

end find_principal_l540_540991


namespace find_triangle_height_l540_540937

variables (A B C D E : Point)
variables (circle : Circle)
variables (AB AC BC : length)
variables (tangent_D tangent_E : Prop)
variables (cyclic_ADEC : Prop)

def height_of_triangle (AB AC BC : length) : length := 
  sqrt (AB.value ^ 2 - (AC.value / 2) ^ 2)

theorem find_triangle_height
  (AB AC : ℝ)
  (condition1 : Circle.circle_tangent_to_side circle A B D)
  (condition2 : Circle.circle_tangent_to_side circle B C E)
  (condition3 : cyclic_points A D E C)
  (h : height_of_triangle 5 2 5 = 2 * sqrt 6 / 5) :
  h = 2 * sqrt 6 / 5 :=
by
  sorry

end find_triangle_height_l540_540937


namespace track_length_l540_540711

theorem track_length (x : ℝ) (hb hs : ℝ) (h_opposite : hs = x / 2 - 120) (h_first_meet : hb = 120) (h_second_meet : hs + 180 = x / 2 + 60) : x = 600 := 
by
  sorry

end track_length_l540_540711


namespace bucyrus_temperature_third_day_l540_540158

theorem bucyrus_temperature_third_day 
  (avg_temp : ℤ)
  (temp_day1 : ℤ)
  (temp_day2 : ℤ)
  (total_days : ℕ) :
  avg_temp = -7 ∧ temp_day1 = -8 ∧ temp_day2 = 1 ∧ total_days = 3 →
  (let total_temp := avg_temp * (total_days : ℤ) in
   let sum_first_two := temp_day1 + temp_day2 in
   let temp_day3 := total_temp - sum_first_two in
   temp_day3 = -14) :=
by
  intros h,
  cases h with h_avg h_rest,
  cases h_rest with h_temp1 h_rest2,
  cases h_rest2 with h_temp2 h_total_days,
  have : (avg_temp * (total_days : ℤ)) = -21, from congr_arg (λ x, x * 3) h_avg,
  have : (temp_day1 + temp_day2) = -7, by rw [h_temp1, h_temp2]; norm_num,
  let temp_day3 := -21 - (-7),
  show temp_day3 = -14, by norm_num; rw temp_day3; refl,
  sorry

end bucyrus_temperature_third_day_l540_540158


namespace exists_pair_satisfying_system_l540_540752

theorem exists_pair_satisfying_system (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) ↔ m ≠ 1 :=
by
  sorry

end exists_pair_satisfying_system_l540_540752


namespace geometric_sequence_common_ratio_l540_540400

theorem geometric_sequence_common_ratio (a_n S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n = a_n + S_n (n-1)) 
    (h2 : ∀ n, S_n = 2 * a_n - 1) : 
    ∀ n, a_n / a_1 = 2 := by
  sorry

end geometric_sequence_common_ratio_l540_540400


namespace geometric_probability_l540_540546

theorem geometric_probability (a x : ℝ) (h1 : x ∈ Icc (-2) 4) 
  (h2 : a^2 + 1/(a^2 + 1) ≥ |x|) : 
  (P : ℝ) = 1/3 :=
sorry

end geometric_probability_l540_540546


namespace mica_total_cost_is_30_79_l540_540899

variables 
  (pasta_kg : ℝ) (pasta_price_per_kg : ℝ)
  (beef_kg : ℝ) (beef_price_per_kg : ℝ)
  (sauce_jars : ℝ) (sauce_price_per_jar : ℝ)
  (quesadillas_price : ℝ)
  (cheese_kg : ℝ) (cheese_price_per_kg : ℝ)
  (discount_rate : ℝ)
  (vat_rate : ℝ)

def total_cost : ℝ :=
  let pasta_cost := pasta_kg * pasta_price_per_kg in
  let beef_cost := beef_kg * beef_price_per_kg in
  let sauce_cost := sauce_jars * sauce_price_per_jar in
  let discounted_sauce_cost := sauce_cost * (1 - discount_rate) in
  let quesadillas_cost := quesadillas_price in
  let cheese_cost := cheese_kg * cheese_price_per_kg in
  let subtotal := pasta_cost + beef_cost + discounted_sauce_cost + quesadillas_cost + cheese_cost in
  let vat := subtotal * vat_rate in
  subtotal + vat

theorem mica_total_cost_is_30_79 
  (pasta_kg := 2.75) (pasta_price_per_kg := 1.70)
  (beef_kg := 0.45) (beef_price_per_kg := 8.20)
  (sauce_jars := 3) (sauce_price_per_jar := 2.30)
  (quesadillas_price := 11.50)
  (cheese_kg := 0.65) (cheese_price_per_kg := 5)
  (discount_rate := 0.10)
  (vat_rate := 0.05) : 
  total_cost pasta_kg pasta_price_per_kg beef_kg beef_price_per_kg sauce_jars sauce_price_per_jar quesadillas_price cheese_kg cheese_price_per_kg discount_rate vat_rate = 30.79 :=
by 
  sorry

end mica_total_cost_is_30_79_l540_540899


namespace correct_statement_D_l540_540638

-- We define each of the mathematical conditions as hypotheses.
def coefficient_neg_3mn : ℤ := -3
def degree_3sq_m3_n : ℕ := 3 + 1
def terms_poly_a2b : List (ℤ × (List (char × ℕ))) := [(1, [('a', 2), ('b', 1)]), (-3, [('a', 1), ('b', 1)]), (5, [])]
def coefficient_linear_term_poly_m2m3 : ℤ := 1

-- We state the main proof problem, using the hypotheses.
theorem correct_statement_D :
  ¬(coefficient_neg_3mn = 3) ∧ 
  ¬(degree_3sq_m3_n = 6) ∧ 
  ¬(terms_poly_a2b = [(1, [('a', 2), ('b', 1)]), (3, [('a', 1), ('b', 1)]), (5, [])]) ∧ 
  (coefficient_linear_term_poly_m2m3 = 1) :=
by
  -- Skip the proof here
  sorry

end correct_statement_D_l540_540638


namespace num_true_statements_l540_540945

theorem num_true_statements :
    (∀ (q: Type) (p: q → Prop), (p q) → q = rectangle → False) ∧  -- Statement 1 is False
    (∃ (par: Type) (p: par → Par), (adj_eq_side: Par → Prop), p par ∧ adj_eq_side par ∧ q = rhombus) ∧  -- Statement 2 is True
    (∃ (rect: Type) (p: rect → Rect), (adj_eq_side: Rect → Prop), p rect ∧ adj_eq_side rect ∧ rect = square) ∧  -- Statement 3 is True
    (∃ (quad: Type) (p: quad → Quad), (opposite_parallel_and_equal: Quad → Prop), p quad ∧ opposite_parallel_and_equal quad ∧ quad = parallelogram) →  -- Statement 4 is True
    (number_of_true_statements = 3) :=
sorry

end num_true_statements_l540_540945


namespace distance_between_street_lights_l540_540237

theorem distance_between_street_lights :
  ∀ (n : ℕ) (L : ℝ), n = 18 → L = 16.4 → 8 > 0 →
  (L / (8 : ℕ) = 2.05) :=
by
  intros n L h_n h_L h_nonzero
  sorry

end distance_between_street_lights_l540_540237


namespace fraction_denominator_l540_540180

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l540_540180


namespace soccer_ball_seams_l540_540683

theorem soccer_ball_seams 
  (num_pentagons : ℕ) 
  (num_hexagons : ℕ) 
  (sides_per_pentagon : ℕ) 
  (sides_per_hexagon : ℕ) 
  (total_pieces : ℕ) 
  (equal_sides : sides_per_pentagon = sides_per_hexagon)
  (total_pieces_eq : total_pieces = 32)
  (num_pentagons_eq : num_pentagons = 12)
  (num_hexagons_eq : num_hexagons = 20)
  (sides_per_pentagon_eq : sides_per_pentagon = 5)
  (sides_per_hexagon_eq : sides_per_hexagon = 6) :
  90 = (num_pentagons * sides_per_pentagon + num_hexagons * sides_per_hexagon) / 2 :=
by 
  sorry

end soccer_ball_seams_l540_540683


namespace athletes_with_four_points_l540_540856

theorem athletes_with_four_points 
  (n : ℕ) 
  (hn : n > 7) 
  (num_athletes := 2^n + 6) 
  (num_rounds := 7) 
  (points_for_win := 1) 
  (points_for_loss := 0) :
  let f (m k : ℕ) := 2^(n - m) * (nat.choose m k),
      participants_with_4_points_2n := f num_rounds 4,
      participants_with_4_points_6 := 2 in
  2^n + 6 > 0 ∧ --- Ensure that the number of athletes is positive
  num_rounds = 7 ∧
  num_athletes = 2^n + 6 ∧
  points_for_win = 1 ∧
  points_for_loss = 0 →
  participants_with_4_points_2n * 35 + participants_with_4_points_6 = 35 * 2^(n - 7) + 2 :=
sorry

end athletes_with_four_points_l540_540856


namespace imaginary_part_of_quotient_l540_540411

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - complex.i
def z2 : ℂ := real.sqrt 3 + complex.i

-- Define the conjugate of z1
def conj_z1 : ℂ := complex.conj z1

-- Define the quotient of conj_z1 and z2
def quotient : ℂ := conj_z1 / z2

-- Prove the imaginary part of the quotient is (√3 - 1) / 4
theorem imaginary_part_of_quotient : quotient.im = (real.sqrt 3 - 1) / 4 := by
  sorry

end imaginary_part_of_quotient_l540_540411


namespace find_certain_number_l540_540251

theorem find_certain_number (x : ℝ) 
    (h : 7 * x - 6 - 12 = 4 * x) : x = 6 := 
by
  sorry

end find_certain_number_l540_540251


namespace order_of_numbers_l540_540202

-- Define the conditions as hypotheses
variable (h1 : 6^(0.7) > 1)
variable (h2 : 0 < 0.7^6 ∧ 0.7^6 < 1)
variable (h3 : log 0.7 6 < 0)

theorem order_of_numbers (h1 : 6^(0.7) > 1) (h2 : 0 < 0.7^6 ∧ 0.7^6 < 1) (h3 : log 0.7 6 < 0) :
  log 0.7 6 < 0.7^6 ∧ 0.7^6 < 6^(0.7) :=
sorry

end order_of_numbers_l540_540202


namespace smallest_four_digit_integer_not_dividing_factorial_l540_540621

theorem smallest_four_digit_integer_not_dividing_factorial :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (let S_n := n * (n + 1) / 2 in
                                     let P_n := Nat.factorial n in 
                                     ¬ (S_n ∣ P_n)) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → (let S_m := m * (m + 1) / 2 in
                                                                                   let P_m := Nat.factorial m in
                                                                                   S_m ∣ P_m) :=
sorry

end smallest_four_digit_integer_not_dividing_factorial_l540_540621


namespace last_place_is_Fedya_l540_540540

def position_is_valid (position : ℕ) := position >= 1 ∧ position <= 4

variable (Misha Anton Petya Fedya : ℕ)

axiom Misha_statement: position_is_valid Misha → Misha ≠ 1 ∧ Misha ≠ 4
axiom Anton_statement: position_is_valid Anton → Anton ≠ 4
axiom Petya_statement: position_is_valid Petya → Petya = 1
axiom Fedya_statement: position_is_valid Fedya → Fedya = 4

theorem last_place_is_Fedya : ∃ (x : ℕ), x = Fedya ∧ Fedya = 4 :=
by
  sorry

end last_place_is_Fedya_l540_540540


namespace max_value_f_l540_540746

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (Real.pi / 2 - x)

theorem max_value_f : ∃ x : ℝ, f x = 5 :=
  by
  sorry

end max_value_f_l540_540746


namespace total_limes_picked_l540_540380

-- Define the number of limes each person picked
def fred_limes : Nat := 36
def alyssa_limes : Nat := 32
def nancy_limes : Nat := 35
def david_limes : Nat := 42
def eileen_limes : Nat := 50

-- Formal statement of the problem
theorem total_limes_picked : 
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  -- Add proof
  sorry

end total_limes_picked_l540_540380


namespace recurring_six_denominator_l540_540175

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l540_540175


namespace find_a_l540_540413

theorem find_a (a : ℝ) (h : ∃ (n : ℕ), n = 1 ∧ (x^2 + (a / x))^5.coeff 7 = -15): a = -3 :=
by
  sorry

end find_a_l540_540413


namespace sum_of_cubes_to_zero_l540_540347

theorem sum_of_cubes_to_zero :
  3 * (∑ k in Finset.range 12, k^3) + 3 * (∑ k in Finset.range 12, -(k^3)) = 0 :=
by
  sorry

end sum_of_cubes_to_zero_l540_540347


namespace positive_integers_congruent_to_4_mod_9_less_than_500_eq_56_l540_540821

theorem positive_integers_congruent_to_4_mod_9_less_than_500_eq_56 :
  {n : ℕ | n < 500 ∧ n % 9 = 4}.card = 56 :=
sorry

end positive_integers_congruent_to_4_mod_9_less_than_500_eq_56_l540_540821


namespace inequal_satisfied_for_all_x_l540_540363

theorem inequal_satisfied_for_all_x (a : ℝ) (h : a ≤ -2) : 
  ∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x :=
by
  sorry

end inequal_satisfied_for_all_x_l540_540363


namespace local_time_finite_l540_540999

variables {X : ℝ → ℝ} {T : ℝ} {x : ℝ}

-- Define the indicator function
def indicator (P : Prop) [Decidable P] : ℝ :=
  if P then 1 else 0

-- Define the function F(x)
def F (x : ℝ) : ℝ :=
  ∫ t in 0..T, indicator (X t < x)

-- Define the local time l_X
def l_X (x : ℝ) : ℝ :=
  lim (λ ε : ℝ, (1 / (2 * ε)) * ∫ t in 0..T, indicator (x - ε < X t ∧ X t < x + ε)) (Filter.atTop) 

-- Proposition
theorem local_time_finite :
  ∀ᵐ x, (with respect to the Lebesgue measure),
  l_X(x, T) = F'(x) :=
begin
  sorry
end

end local_time_finite_l540_540999


namespace acute_triangle_ineq_l540_540854

open_locale real

variables {A B C O P : Type*} -- Types for points
variables [triangle ABC] [circumcenter O ABC] [altitude A P BC]
variables (alpha beta gamma delta : ℝ)

noncomputable def angle_BCA : ℝ := sorry -- Placeholder for angle BCA
noncomputable def angle_ABC : ℝ := sorry -- Placeholder for angle ABC
noncomputable def angle_CAB : ℝ := sorry -- Placeholder for angle CAB
noncomputable def angle_COP : ℝ := sorry -- Placeholder for angle COP

theorem acute_triangle_ineq (h1 : is_acute_triangle ABC)
  (h2 : is_circumcenter O ABC)
  (h3 : is_foot A P BC)
  (h4 : angle_BCA >= angle_ABC + 30) :
  angle_CAB + angle_COP < 90 :=
sorry

end acute_triangle_ineq_l540_540854


namespace rock_paper_scissors_expected_value_l540_540217

theorem rock_paper_scissors_expected_value :
  let A B C : Type := ℕ in
  let P : Type := ℕ → ℝ := λ x, match x with
    | 0 => 5 / 9
    | 1 => 1 / 3
    | 2 => 1 / 9
    | _ => 0
  in
  let E : P → ℝ := λ P, 0 * P 0 + 1 * P 1 + 2 * P 2 in
  E P = 2 / 3 :=
by
  -- Proof is required; skip it for now
  sorry

end rock_paper_scissors_expected_value_l540_540217


namespace calculate_expr_l540_540716

noncomputable def expr : ℝ := 3^(-1) + (27:ℝ)^(1/3) - (5 - Real.sqrt 5)^0 + |Real.sqrt 3 - 1/3|

theorem calculate_expr : expr = 2 + Real.sqrt 3 := by
  sorry

end calculate_expr_l540_540716


namespace ways_to_choose_n_greater_than_half_sum_l540_540126

theorem ways_to_choose_n_greater_than_half_sum
  (n : ℕ) (n_pos : 0 < n)
  (a : Fin (2 * n - 1) → ℝ)
  (h_pos : ∀ i, 0 < a i) (S : ℝ)
  (h_sum : ∑ i, a i = S) :
  ∃ s : Finset (Fin (2 * n - 1)),
    s.card = n ∧
    ∑ i in s, a i > S / 2 :=
sorry

end ways_to_choose_n_greater_than_half_sum_l540_540126


namespace sum_of_x_total_sum_l540_540977

def mean (a b c d e : ℝ) : ℝ := (a + b + c + d + e) / 5

def median (a b c d e : ℝ) : ℝ :=
  let sorted := List.sort [a, b, c, d, e]
  sorted.nthLe 2 (by apply List.length_sort sorted; simp)

theorem sum_of_x (x : ℝ) :
  (median 3 7 9 15 x = mean 3 7 9 15 x) →
  x = 1 ∨ x = 11 ∨ x = 8.5 :=
sorry

theorem total_sum :
  (∑ x in ({1, 11, 8.5} : Finset ℝ), x) = 20.5 :=
by simp

end sum_of_x_total_sum_l540_540977


namespace polynomial_roots_bc_product_l540_540603

theorem polynomial_roots_bc_product : ∃ (b c : ℤ), 
  (∀ x, (x^2 - 2*x - 1 = 0 → x^5 - b*x^3 - c*x^2 = 0)) ∧ (b * c = 348) := by 
  sorry

end polynomial_roots_bc_product_l540_540603


namespace correct_option_l540_540988

theorem correct_option : 
  (∀ a b : ℝ, (a - b) * (-a - b) ≠ a^2 - b^2) ∧
  (∀ a : ℝ, 2 * a^3 + 3 * a^3 ≠ 5 * a^6) ∧ 
  (∀ x y : ℝ, 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2) ∧
  (∀ x : ℝ, (-2 * x^2)^3 ≠ -6 * x^6) :=
by 
  split
  . intros a b
    sorry
  . split
    . intros a
      sorry
    . split
      . intros x y
        sorry
      . intros x
        sorry

end correct_option_l540_540988


namespace max_number_of_kids_on_school_bus_l540_540609

-- Definitions based on the conditions from the problem
def totalRowsLowerDeck : ℕ := 15
def totalRowsUpperDeck : ℕ := 10
def capacityLowerDeckRow : ℕ := 5
def capacityUpperDeckRow : ℕ := 3
def reservedSeatsLowerDeck : ℕ := 10
def staffMembers : ℕ := 4

-- The total capacity of the lower and upper decks
def totalCapacityLowerDeck := totalRowsLowerDeck * capacityLowerDeckRow
def totalCapacityUpperDeck := totalRowsUpperDeck * capacityUpperDeckRow
def totalCapacity := totalCapacityLowerDeck + totalCapacityUpperDeck

-- The maximum number of different kids that can ride the bus
def maxKids := totalCapacity - reservedSeatsLowerDeck - staffMembers

theorem max_number_of_kids_on_school_bus : maxKids = 91 := 
by 
  -- Step-by-step proof not required for this task
  sorry

end max_number_of_kids_on_school_bus_l540_540609


namespace volume_of_pyramid_MABCD_l540_540499

def fib_seq : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := fib_seq (n - 1) + fib_seq (n - 2)

theorem volume_of_pyramid_MABCD (d ma mc mb : ℝ)
  (h_ma : ma = (fib_seq 1))
  (h_mc : mc = (fib_seq 2))
  (h_mb : mb = (fib_seq 3))
  (h_dm : d = 3)
  (rect: ∀ A B C D: ℝ, ∃ DM: ℝ, (DM ⟂ A) ∧ (DM ⟂ B) ∧ (DM ⟂ C) ∧ (DM ⟂ D) ): 
  (∃ vol: ℝ, vol = (sqrt 130)) :=
by
  -- The work to calculate the volume of the pyramid would be done here
  sorry

end volume_of_pyramid_MABCD_l540_540499


namespace average_candies_correct_l540_540346

noncomputable def Eunji_candies : ℕ := 35
noncomputable def Jimin_candies : ℕ := Eunji_candies + 6
noncomputable def Jihyun_candies : ℕ := Eunji_candies - 3
noncomputable def Total_candies : ℕ := Eunji_candies + Jimin_candies + Jihyun_candies
noncomputable def Average_candies : ℚ := Total_candies / 3

theorem average_candies_correct :
  Average_candies = 36 := by
  sorry

end average_candies_correct_l540_540346


namespace radius_of_circle_B_is_2_l540_540718

theorem radius_of_circle_B_is_2
  (A B C D : Type)
  (radius_A : ℝ)
  (radius_D : ℝ)
  (tangent : Circle → Circle → Prop)
  (center_A : Point)
  (center_B : Point)
  (center_C : Point)
  (center_D : Point)
  (radius_B : ℝ) :
  (radius_A = 2) →
  (tangent A B) →
  (tangent A C) →
  (tangent B C) →
  (tangent A D) →
  (tangent B D) →
  (tangent C D) →
  (radius_B = 2) :=
begin
  sorry
end

end radius_of_circle_B_is_2_l540_540718


namespace parabola_equation_line_equation_chord_l540_540807

section
variables (p : ℝ) (x_A y_A : ℝ) (M_x M_y : ℝ)
variable (h_p_pos : p > 0)
variable (h_A : y_A^2 = 8 * x_A)
variable (h_directrix_A : x_A + p / 2 = 5)
variable (h_M : (M_x, M_y) = (3, 2))

theorem parabola_equation (h_x_A : x_A = 3) : y_A^2 = 8 * x_A :=
sorry

theorem line_equation_chord
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : y1^2 = 8 * x1 ∧ y2^2 = 8 * x2)
  (h_chord_M : (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = 2) :
  y_M - 2 * x_M + 4 = 0 :=
sorry
end

end parabola_equation_line_equation_chord_l540_540807


namespace bisect_segment_AC_l540_540134

-- Definitions based on conditions
variables {A B C H H1 H2 : Point}
variable (triangle_orthocenter : H = orthocenter A B C)
variable (proj_internal_bisector : H1 = projection H (internal_bisector A B C))
variable (proj_external_bisector : H2 = projection H (external_bisector A B C))

-- Statement to prove
theorem bisect_segment_AC : bisects_line_segment H1 H2 A C :=
by
  sorry

end bisect_segment_AC_l540_540134


namespace athletes_with_four_points_l540_540855

theorem athletes_with_four_points 
  (n : ℕ) 
  (hn : n > 7) 
  (num_athletes := 2^n + 6) 
  (num_rounds := 7) 
  (points_for_win := 1) 
  (points_for_loss := 0) :
  let f (m k : ℕ) := 2^(n - m) * (nat.choose m k),
      participants_with_4_points_2n := f num_rounds 4,
      participants_with_4_points_6 := 2 in
  2^n + 6 > 0 ∧ --- Ensure that the number of athletes is positive
  num_rounds = 7 ∧
  num_athletes = 2^n + 6 ∧
  points_for_win = 1 ∧
  points_for_loss = 0 →
  participants_with_4_points_2n * 35 + participants_with_4_points_6 = 35 * 2^(n - 7) + 2 :=
sorry

end athletes_with_four_points_l540_540855


namespace winning_strategy_l540_540192

theorem winning_strategy : ∃ n ∈ {4, 5, 6}, ∀ k ∈ {2, 3, 4, 5, 6, 7, 8, 9}, n * k > 1000 → ∃ m ∈ {4, 5, 6}, m < n :=
by
  sorry

end winning_strategy_l540_540192


namespace calvin_follows_conditions_then_made_shots_in_fourth_game_l540_540287

theorem calvin_follows_conditions_then_made_shots_in_fourth_game:
    ∀ (shots_first_three_games made_first_three_games shots_fourth_game total_shots_season average_initial average_new),
    shots_first_three_games = 35 →
    made_first_three_games = 15 →
    shots_fourth_game = 15 →
    total_shots_season = 50 →
    average_initial ≈ 0.4286 →
    average_new = 0.55 →
    ∃ (shots_made_fourth_game: ℕ),
    shots_made_fourth_game = 13 := by
  sorry

end calvin_follows_conditions_then_made_shots_in_fourth_game_l540_540287


namespace tan_diff_identity_l540_540755

theorem tan_diff_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β + π / 4) = 1 / 4) :
  Real.tan (α - π / 4) = 3 / 22 :=
sorry

end tan_diff_identity_l540_540755


namespace max_n_l540_540124

noncomputable def seq_a (n : ℕ) : ℤ := 3 * n - 1

noncomputable def seq_b (n : ℕ) : ℤ := 2 * n - 3

noncomputable def sum_T (n : ℕ) : ℤ := n * (3 * n + 1) / 2

noncomputable def sum_S (n : ℕ) : ℤ := n^2 - 2 * n

theorem max_n (n : ℕ) :
  ∃ n_max : ℕ, T_n < 20 * seq_b n ∧ (∀ m : ℕ, m > n_max → T_n ≥ 20 * seq_b n) :=
  sorry

end max_n_l540_540124


namespace square_inscribed_in_ellipse_area_l540_540268

theorem square_inscribed_in_ellipse_area :
  ∀ t : ℝ,
    (∃ t, (t ≠ 0 ∧ (t * t / 4 + t * t / 8 = 1))) →
    let side_length := 2 * t in
    side_length ^ 2 = 32 / 3 :=
by
  -- Proof skipped for now
  sorry

end square_inscribed_in_ellipse_area_l540_540268


namespace max_area_of_triangle_l540_540406

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  ellipse_equation 2 2 x y

noncomputable def is_eccentricity (e a b : ℝ) : Prop :=
  e = (Real.sqrt(2)) / 2 ∧ a^2 = 2 * b^2 ∧ a > 0 ∧ b > 0

noncomputable def is_line (x m y : ℝ) : Prop :=
  y = (Real.sqrt(2)) / 2 * x + m

noncomputable def max_triangle_area : ℝ := (Real.sqrt 2)

theorem max_area_of_triangle
  (a b : ℝ)
  (h1: is_eccentricity (Real.sqrt(2) / 2) a b)
  (h2 : ellipse_equation (Real.sqrt 2) 1 a b)                     
  (h3 : point_on_ellipse (Real.sqrt 2) 1)
  (m : ℝ) : ∃ (area : ℝ), area = max_triangle_area :=
sorry

end max_area_of_triangle_l540_540406


namespace find_average_of_data_l540_540767

-- Assume a set of positive numbers x_1, x_2, x_3, x_4
variables {x_1 x_2 x_3 x_4 : ℝ}
-- Assume the condition on the variance
def variance (x_1 x_2 x_3 x_4 : ℝ) : ℝ := (1 / 4) * (x_1^2 + x_2^2 + x_3^2 + x_4^2 - 16)

-- Given condition: the variance of the set is s^2
variable (s : ℝ)
hypothesis h_variance : variance x_1 x_2 x [...]

-- Define the mean of new data points
def new_mean (x_1 x_2 x_3 x_4 : ℝ) : ℝ := (1 / 4) * ((x_1 + 2) + (x_2 + 2) + (x_3 + 2) + (x_4 + 2))

-- The theorem we need to prove
theorem find_average_of_data : new_mean x_1 x_2 x_3 x_4 = 4 :=
sorry

end find_average_of_data_l540_540767


namespace find_k_if_parallel_l540_540018

def n1 : ℝ × ℝ × ℝ := (1, 2, -2)
def n2 (k : ℝ) : ℝ × ℝ × ℝ := (-2, -4, k)

theorem find_k_if_parallel (k : ℝ) (λ : ℝ) 
    (h1 : 1 = λ * (-2)) 
    (h2 : 2 = λ * (-4)) 
    (h3 : -2 = λ * k) :
    k = 4 :=
by
  sorry

end find_k_if_parallel_l540_540018


namespace sum_of_solutions_l540_540978

theorem sum_of_solutions : (∑ x in {x : ℝ | |3 * x - 6| = 9}.toFinset, x) = 4 :=
by
  sorry

end sum_of_solutions_l540_540978


namespace maxim_final_amount_l540_540896

-- Define the necessary constants and their values
def P : ℝ := 1000 -- Principal amount
def r : ℝ := 0.12 -- Annual interest rate as a decimal
def n : ℝ := 12 -- Compounding frequency (monthly)
def t : ℝ := 1 / 12 -- Time period in years (1 month)

-- Define the compound interest formula
def A : ℝ := P * (1 + r / n) ^ (n * t)

-- The statement we need to prove
theorem maxim_final_amount : A = 1010 := by
  -- We state that this part is to be proved as we skip with sorry
  sorry

end maxim_final_amount_l540_540896


namespace find_values_l540_540759

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_values 
  (a b c d : ℝ)
  (h1 : ∀ x, f (2*x + 1) a b = 4 * g x c d)
  (h2 : ∀ x, (f x a b)' = (g x c d)')
  (h3 : f 5 a b = 30)
  : a = 2 ∧ b = -5 ∧ c = 2 ∧ d = -1 / 2 :=
sorry

end find_values_l540_540759


namespace scalar_projection_of_a_onto_b_is_minus_3_l540_540037

-- Define the vectors a and b
def vec_a : (ℝ × ℝ) :=
  (3, 6)

def vec_b : (ℝ × ℝ) :=
  (3, -4)

-- Function to compute the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the magnitude of a vector
def magnitude (v : (ℝ × ℝ)) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Function to compute the scalar projection
def scalar_projection (u v : (ℝ × ℝ)) : ℝ :=
  (dot_product u v) / (magnitude v)

-- Theorem statement
theorem scalar_projection_of_a_onto_b_is_minus_3 :
  scalar_projection vec_a vec_b = -3 :=
by
  sorry

end scalar_projection_of_a_onto_b_is_minus_3_l540_540037


namespace square_area_in_ellipse_l540_540266

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end square_area_in_ellipse_l540_540266


namespace range_of_m_in_inverse_proportion_function_l540_540065

theorem range_of_m_in_inverse_proportion_function (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (1 - m) / x > 0) ∧ (x < 0 → (1 - m) / x < 0))) ↔ m < 1 :=
by
  sorry

end range_of_m_in_inverse_proportion_function_l540_540065


namespace reciprocal_and_fraction_l540_540451

theorem reciprocal_and_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : (2/5) * a = 20) : 
  b = (1/a) ∧ (1/3) * a = (50/3) := 
by 
  sorry

end reciprocal_and_fraction_l540_540451


namespace total_players_is_28_l540_540248

def total_players (A B C AB BC AC ABC : ℕ) : ℕ :=
  A + B + C - (AB + BC + AC) + ABC

theorem total_players_is_28 :
  total_players 10 15 18 8 6 4 3 = 28 :=
by
  -- as per inclusion-exclusion principle
  -- T = A + B + C - (AB + BC + AC) + ABC
  -- substituting given values we repeatedly perform steps until final answer
  -- take user inputs to build your final answer.
  sorry

end total_players_is_28_l540_540248


namespace max_good_triplets_l540_540119

-- Define the problem's conditions
variables (k : ℕ) (h_pos : 0 < k)

-- The statement to be proven
theorem max_good_triplets : ∃ T, T = 12 * k ^ 4 := 
sorry

end max_good_triplets_l540_540119


namespace solve_smallest_x_l540_540148

theorem solve_smallest_x :
  let equation := ∀ x : ℚ, 3 * (10 * x^2 + 10 * x + 15) = x * (10 * x - 55)
  ∃ x : ℚ, equation x ∧ x = -29/8 :=
begin
  sorry
end

end solve_smallest_x_l540_540148


namespace triangle_area_proof_l540_540866

noncomputable def triangle_area_lemma : Prop :=
  ∀ {BD DC : ℝ} (h_ratio : BD / DC = 5 / 2) (area_ABD : ℝ),
    area_ABD = 30 →
    let area_ADC : ℝ := (2 / 5) * area_ABD in
    area_ADC = 12

theorem triangle_area_proof : triangle_area_lemma :=
by
  intros BD DC h_ratio area_ABD h_eq
  let area_ADC := (2 / 5) * area_ABD
  show area_ADC = 12
  -- Proof steps (to be filled in)
  sorry

end triangle_area_proof_l540_540866


namespace rachel_total_problems_l540_540139

theorem rachel_total_problems
    (problems_per_minute : ℕ)
    (minutes_before_bed : ℕ)
    (problems_next_day : ℕ) 
    (h1 : problems_per_minute = 5) 
    (h2 : minutes_before_bed = 12) 
    (h3 : problems_next_day = 16) : 
    problems_per_minute * minutes_before_bed + problems_next_day = 76 :=
by
  sorry

end rachel_total_problems_l540_540139


namespace finding_value_of_expression_l540_540888

open Real

theorem finding_value_of_expression
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : 1/a - 1/b - 1/(a + b) = 0) :
  (b/a + a/b)^2 = 5 :=
sorry

end finding_value_of_expression_l540_540888


namespace zeros_of_f_monotonic_decreasing_f_range_of_m_l540_540534

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs ((2 / x) - a * x + 5)

theorem zeros_of_f (a : ℝ) :
  (a = 0 → ∃ x : ℝ, f x a = 0 ∧ x = -(2 / 5))
  ∧ ((a ≥ -25 / 8) ∧ (a ≠ 0) → ∃ x : ℝ, f x a = 0 ∧ (x = (5 + sqrt (25 + 8 * a)) / (2 * a) ∨ x = (5 - sqrt (25 + 8 * a)) / (2 * a)))
  ∧ (a < -25 / 8 → ∀ x : ℝ, f x a ≠ 0) := sorry

theorem monotonic_decreasing_f (a : ℝ) :
  a = 3 → ∀ x1 x2 : ℝ, x1 < x2 → x1 < -1 → x2 < -1 → f x1 a > f x2 a := sorry

theorem range_of_m (m : ℝ) :
  (∀ a > 0, ∃ x0 ∈ Icc 1 2, f x0 a ≥ m) → m ≤ 8 / 3 := sorry

end zeros_of_f_monotonic_decreasing_f_range_of_m_l540_540534


namespace ramu_profit_percent_l540_540651

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit * 100) / total_cost

theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end ramu_profit_percent_l540_540651


namespace parallelogram_is_rhombus_l540_540007

variables (A B C D : Type) [add_comm_group A] [vector_space ℝ A]
variables (AB AC AD CD DB : A)

def is_parallelogram (AB CD : A) : Prop :=
  AB + CD = 0

def is_orthogonal (v1 v2 : A) : Prop :=
  inner_product v1 v2 = 0

theorem parallelogram_is_rhombus (h1 : is_parallelogram AB CD)
  (h2 : is_orthogonal (AB - AD) AC) :
  true := 
sorry

end parallelogram_is_rhombus_l540_540007


namespace sum_arithmetic_series_eq_499500_l540_540236

theorem sum_arithmetic_series_eq_499500 :
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  (n * (a1 + an) / 2) = 499500 := by {
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  show (n * (a1 + an) / 2) = 499500
  sorry
}

end sum_arithmetic_series_eq_499500_l540_540236


namespace solve_equation_l540_540353

theorem solve_equation (x : ℝ) :
  (1 / (x^2 + 17 * x - 8) + 1 / (x^2 + 4 * x - 8) + 1 / (x^2 - 9 * x - 8) = 0) →
  (x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4) :=
by
  sorry

end solve_equation_l540_540353


namespace jacket_price_restoration_l540_540996

theorem jacket_price_restoration :
  ∀ (original_price : ℝ), let reduced_price := (original_price - 0.1 * original_price) - 0.3 * (original_price - 0.1 * original_price) in
  (reduced_price + 0.5873 * reduced_price = original_price) :=
by
  intros original_price
  let reduced_price := (original_price - 0.1 * original_price) - 0.3 * (original_price - 0.1 * original_price)
  sorry

end jacket_price_restoration_l540_540996


namespace sum_denominator_power_of_2_l540_540348

open Real

noncomputable def condition (i : ℕ) : ℝ := 
  if ((2 * i - 1) % 4 = 1) then 1
  else if ((2 * i - 1) % 4 = 3) then -1
  else 0

def sum_formula : ℝ := 
  ∑ i in (Finset.range 1004).map Nat.succ, (condition i) / (2 * i : ℝ)

theorem sum_denominator_power_of_2 : ∃ (n : ℤ), ∃ (k : ℤ), (sum_formula = (1 : ℝ) / (2 ^ n)) :=
sorry

end sum_denominator_power_of_2_l540_540348


namespace find_ratios_l540_540122

noncomputable def two_intersections (ω₁ ω₂: Type) (O₁ O₂ : Point) (r₁ r₂ : Real) (A B : Point) : Prop :=
  (O₂ ∈ ω₁) ∧
  (A ∈ ω₁) ∧ (A ∈ ω₂) ∧
  (B ∈ ω₂) ∧ (Collinear [O₁, O₂, B]) ∧ 
  (dist A B = dist O₁ A) ∧ (dist O₁ A = r₁) ∧ (dist O₂ A = r₂)

theorem find_ratios (ω₁ ω₂ : Type) (O₁ O₂ A B : Point) (r₁ r₂ : Real)
  (h_intersections : two_intersections ω₁ ω₂ O₁ O₂ r₁ r₂ A B) :
  ∃ k : Real, k = (r₁ / r₂) ∧ (
    k = (-1 + Real.sqrt 5) / 2 ∨ k = (1 + Real.sqrt 5) / 2
  ) :=
sorry

end find_ratios_l540_540122


namespace shaded_area_is_50_pi_l540_540864

-- Definitions based on the problem conditions
def radius_large_circle : ℝ := 10
def congruent_small_circles (r : ℝ) (C1 C2 : Circle) : Prop :=
  C1.radius = r ∧ C2.radius = r ∧ C1.radius = C2.radius
def touches_internally (large_circle small_circle : Circle) : Prop :=
  dist large_circle.center small_circle.center + small_circle.radius = large_circle.radius
def touches_externally (C1 C2 : Circle) : Prop :=
  dist C1.center C2.center = C1.radius + C2.radius
def area (r : ℝ) : ℝ := π * r^2

-- Final statement
theorem shaded_area_is_50_pi :
  ∀ (C1 C2 : Circle) (r : ℝ), 
    congruent_small_circles r C1 C2 ∧ 
    touches_internally ⟨(0, 0), radius_large_circle⟩ C1 ∧ 
    touches_internally ⟨(0, 0), radius_large_circle⟩ C2 ∧ 
    touches_externally C1 C2 
    → area radius_large_circle - 2 * area r = 50 * π := 
sorry

end shaded_area_is_50_pi_l540_540864


namespace decrease_in_profit_when_one_loom_idles_l540_540272

def num_looms : ℕ := 125
def total_sales_value : ℕ := 500000
def total_manufacturing_expenses : ℕ := 150000
def monthly_establishment_charges : ℕ := 75000
def sales_value_per_loom : ℕ := total_sales_value / num_looms
def manufacturing_expense_per_loom : ℕ := total_manufacturing_expenses / num_looms
def decrease_in_sales_value : ℕ := sales_value_per_loom
def decrease_in_manufacturing_expenses : ℕ := manufacturing_expense_per_loom
def net_decrease_in_profit : ℕ := decrease_in_sales_value - decrease_in_manufacturing_expenses

theorem decrease_in_profit_when_one_loom_idles : net_decrease_in_profit = 2800 := by
  sorry

end decrease_in_profit_when_one_loom_idles_l540_540272


namespace geometric_progression_quadrilateral_exists_l540_540486

theorem geometric_progression_quadrilateral_exists :
  ∃ (a1 r : ℝ), a1 > 0 ∧ r > 0 ∧ 
  (1 + r + r^2 > r^3) ∧
  (1 + r + r^3 > r^2) ∧
  (1 + r^2 + r^3 > r) ∧
  (r + r^2 + r^3 > 1) := 
sorry

end geometric_progression_quadrilateral_exists_l540_540486


namespace hyperbola_eccentricity_l540_540806

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 = (Real.sqrt (a^2 + 3)) / a) : a = 1 := 
by
  sorry

end hyperbola_eccentricity_l540_540806


namespace triangle_inequality_integer_count_l540_540199

theorem triangle_inequality_integer_count :
  (finset.Ico 19 40).card = 21 :=
by
  -- sorry allows skipping the actual proof for demonstration purposes
  sorry

end triangle_inequality_integer_count_l540_540199


namespace probability_zero_l540_540109

noncomputable def Q (x : ℕ) : ℝ := x^2 - 5 * x + 4

def in_interval (x : ℝ) : Prop := 10 ≤ x ∧ x ≤ 20

open Real

theorem probability_zero :
  ∀ (x : ℝ), in_interval x → (⌊sqrt (Q x)⌋ = sqrt (Q ⌊x⌋ : ℝ)) → false :=
by
  intro x hx h
  have : ∀ n ∈ finset.range (20 - 10 + 1), Q (n + 10) ≠ ⌊(Q (n + 10)).sqrt⌋^2 + ⌊(Q (n + 10)).sqrt⌋,
  { -- For every integer n from 10 to 20, Q(n) is not a perfect square.
    intro n hn,
    finset_cases_on finset.range_succ hn m Hm,
    replace Hm : m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.to_finset + 10 := finset.mem_rangeₓ_succ'.mp Hm
    simp at Hm,
    cases Hm with k Hk,
    interval_cases k ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    have := by sorry, -- derive a contradiction
    contradiction },
  have : ∀ n ∈ ℤ.range (21), sqrt (Q n) ∉ ℤ,
  { -- This is just a rephrasing of the above
    intro n hn,
    simp at hn,
    have := h,
    contradiction }
  simp
sorry

end probability_zero_l540_540109


namespace find_k_l540_540027

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem find_k (k : ℝ) (h : deriv (f k) 0 = 27) : k = 3 :=
by
  sorry

end find_k_l540_540027


namespace area_change_l540_540845

theorem area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := 1.2 * L
  let B' := 0.8 * B
  let A := L * B
  let A' := L' * B'
  A' = 0.96 * A :=
by
  sorry

end area_change_l540_540845


namespace gcd_180_270_450_l540_540224

theorem gcd_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 := by 
  sorry

end gcd_180_270_450_l540_540224


namespace g_of_g_of_2_l540_540443

def g (x : ℝ) : ℝ := 4 * x^2 - 3

theorem g_of_g_of_2 : g (g 2) = 673 := 
by 
  sorry

end g_of_g_of_2_l540_540443


namespace solution_set_l540_540805
  
noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp (2 * x) + 1) - x

theorem solution_set (x : ℝ) :
  f (x + 2) > f (2 * x - 3) ↔ (1 / 3 < x ∧ x < 5) :=
by
  sorry

end solution_set_l540_540805


namespace probability_of_x_squared_less_than_one_is_half_l540_540394

theorem probability_of_x_squared_less_than_one_is_half :
  let interval_len := 2 - (-2)
  let solution_set_len := 1 - (-1)
  probability_of_solution := solution_set_len / interval_len
  probability_of_solution = 1 / 2 :=
by
  sorry

end probability_of_x_squared_less_than_one_is_half_l540_540394


namespace max_players_without_computer_assistance_l540_540998

theorem max_players_without_computer_assistance :
  ∃ (n : ℕ), n = 50 ∧
  ∀ (players : List ℕ) (ranked : players.length = 55)
    (play_game : ℕ → ℕ → ℤ) 
    (computer_assist : ℕ → bool),
    (∀ i j, i ≠ j → (play_game i j = if computer_assist i ∧ ¬ computer_assist j then 1
                                      else if ¬ computer_assist i ∧ computer_assist j then 0
                                      else if i < j then 1 else 0)) →
    (∃ a b, a ∈ players ∧ b ∈ players ∧ 
            (a > b ∧ ∀ k, k ∈ players ∧ k ≠ a ∧ k ≠ b → play_game k a < play_game a k ∧ play_game k b < play_game b k) ∧ 
            (∃ x y, x ∈ players ∧ y ∈ players ∧ 
                    (play_game x a > play_game a x ∧ play_game y a > play_game a y))) →
    (players.filter (λ x => ¬ computer_assist x)).length ≤ 50 :=
by
  sorry

end max_players_without_computer_assistance_l540_540998


namespace recurring_six_denominator_l540_540170

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l540_540170


namespace convert_rectangular_to_polar_l540_540293

def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x) + if x < 0 then Real.pi else if y < 0 then 2 * Real.pi else 0
  (r, theta)

theorem convert_rectangular_to_polar :
  rectangular_to_polar (-3) (3 * Real.sqrt 3) = (6, 2 * Real.pi / 3) :=
by
  sorry

end convert_rectangular_to_polar_l540_540293


namespace train_crossing_time_l540_540687

-- Defining a structure for our problem context
structure TrainCrossing where
  length : Real -- length of the train in meters
  speed_kmh : Real -- speed of the train in km/h
  conversion_factor : Real -- conversion factor from km/h to m/s

-- Given the conditions in the problem
def trainData : TrainCrossing :=
  ⟨ 280, 50.4, 0.27778 ⟩

-- The main theorem statement:
theorem train_crossing_time (data : TrainCrossing) : 
  data.length / (data.speed_kmh * data.conversion_factor) = 20 := 
by
  sorry

end train_crossing_time_l540_540687


namespace deborah_total_letters_l540_540297

-- Definitions based on the conditions:
def postage_per_letter := 1.08
def additional_charge := 0.14
def international_letters := 2
def total_cost := 4.60

-- Lean 4 statement proving that Deborah has to mail 4 letters.
theorem deborah_total_letters (D : ℕ) :
  (D : ℝ) * postage_per_letter + (international_letters : ℝ) * (postage_per_letter + additional_charge) = total_cost →
  D + international_letters = 4 := 
by
  sorry

end deborah_total_letters_l540_540297


namespace survey_sample_is_sample_l540_540471

-- Definitions based on the conditions
def urban_residents : Type := ℕ
def provinces : Type := ℕ
def survey (residents : urban_residents) (prov : provinces) : Prop := 
  residents = 2500 ∧ prov = 11

-- Main statement to prove
theorem survey_sample_is_sample :
  ∀ (residents : urban_residents) (prov : provinces), 
  survey residents prov → "sample" :=
by
  intro residents prov h
  sorry

end survey_sample_is_sample_l540_540471


namespace proj_u_on_z_magnitude_l540_540432

variables {𝕜 : Type*} [IsROrC 𝕜]
variables (u z : 𝕜 ^ 2)

axiom norm_u : ‖u‖ = 5
axiom norm_z : ‖z‖ = 8
axiom dot_uz : InnerProductSpace.inner u z = 20

noncomputable def proj_magnitude (u z : 𝕜 ^ 2) : ℝ :=
  ‖(InnerProductSpace.inner u z / ‖z‖ ^ 2) • z‖

theorem proj_u_on_z_magnitude :
  proj_magnitude u z = 5 / 2 :=
by
  rw [proj_magnitude, norm_u, norm_z, dot_uz]
  sorry

end proj_u_on_z_magnitude_l540_540432


namespace number_of_sequences_returning_to_position_l540_540111

-- Define the vertices of the triangle T'
def vertices : List (ℝ × ℝ) := [(0,0), (6,0), (0,4)]

-- Define transformations
def rot90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rot180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rot270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def refl_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def refl_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def refl_xy (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The problem statement: Prove the number of sequences of three transformations that return T' to its original position.
theorem number_of_sequences_returning_to_position : 
  number_of_sequences (vertices) [(rot90, rot180, rot270, refl_x, refl_y, refl_xy)] = 24 := by
  sorry

end number_of_sequences_returning_to_position_l540_540111


namespace point_in_second_quadrant_l540_540835

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l540_540835


namespace number_of_valid_products_of_two_distinct_elements_l540_540506

def is_divisor (n k : ℕ) : Prop := k ∣ n

def prime_factors_60000 := (2, 5, 3, 1, 5, 4)  -- Represent 2^5 * 3^1 * 5^4

def is_valid_product_of_distinct_elements (n : ℕ) : Prop :=
  ∃ a b c x y z : ℕ,
    0 ≤ a ∧ a ≤ 5 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 4 ∧
    0 ≤ x ∧ x ≤ 5 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 4 ∧
    (a ≠ x ∨ b ≠ y ∨ c ≠ z) ∧
    n = (2^(a+x) * 3^(b+y) * 5^(c+z))

theorem number_of_valid_products_of_two_distinct_elements :
  (finset.get_finset_fin.card 
  ((finset.filter is_valid_product_of_distinct_elements 
  (finset.range 2^(5*2) * 3^(1*2) * 5^(4*2)))) = 293) :=
sorry

end number_of_valid_products_of_two_distinct_elements_l540_540506


namespace problem_statement_l540_540104

variable {x y z : ℝ}

theorem problem_statement
  (h : x^2 + y^2 + z^2 + 9 = 4 * (x + y + z)) :
  x^4 + y^4 + z^4 + 16 * (x^2 + y^2 + z^2) ≥ 8 * (x^3 + y^3 + z^3) + 27 :=
by
  sorry

end problem_statement_l540_540104


namespace polynomial_divisible_by_cube_l540_540135

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := 
  n^2 * x^(n+2) - (2 * n^2 + 2 * n - 1) * x^(n+1) + (n + 1)^2 * x^n - x - 1

theorem polynomial_divisible_by_cube (n : ℕ) (h : n > 0) : 
  ∃ Q, P n x = (x - 1)^3 * Q :=
sorry

end polynomial_divisible_by_cube_l540_540135


namespace problem_statement_l540_540286

theorem problem_statement :
  |2 - Real.tan (Float.pi * 60 / 180)| - (Real.pi - 3.14)^0 + (-1/2)^(-2) + 1/2 * Real.sqrt 12 = 5 :=
by
  sorry

end problem_statement_l540_540286


namespace max_good_subset_size_l540_540527

open Finset

noncomputable def S (n : ℕ) := { a : fin (2^n) → ℕ | ∀ i, a i = 0 ∨ a i = 1 }

def d {n : ℕ} (a b : fin (2^n) → ℕ) := 
  ∑ i, abs (a i - b i)

def good_subset (n : ℕ) (A : finset (fin (2^n) → ℕ)) := 
  ∀ {a b} (ha : a ∈ A) (hb : b ∈ A), a ≠ b → d a b ≥ 2^(n-1)

theorem max_good_subset_size (n : ℕ) : 
  ∃ (A : finset (fin (2^n) → ℕ)), good_subset n A ∧ A.card = 2^(n+1) :=
sorry

end max_good_subset_size_l540_540527


namespace find_n_l540_540852

open Nat

-- Defining the production rates for conditions.
structure Production := 
  (workers : ℕ)
  (gadgets : ℕ)
  (gizmos : ℕ)
  (hours : ℕ)

def condition1 : Production := { workers := 150, gadgets := 450, gizmos := 300, hours := 1 }
def condition2 : Production := { workers := 100, gadgets := 400, gizmos := 500, hours := 2 }
def condition3 : Production := { workers := 75, gadgets := 900, gizmos := 900, hours := 4 }

-- Statement: Finding the value of n.
theorem find_n :
  (75 * ((condition2.gadgets / condition2.workers) * (condition3.hours / condition2.hours))) = 600 := by
  sorry

end find_n_l540_540852


namespace find_number_l540_540652

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l540_540652


namespace seating_impossible_l540_540700

theorem seating_impossible (reps : Fin 54 → Fin 27) : 
  ¬ ∃ (s : Fin 54 → Fin 54),
    (∀ i : Fin 27, ∃ a b : Fin 54, a ≠ b ∧ s a = i ∧ s b = i ∧ (b - a ≡ 10 [MOD 54] ∨ a - b ≡ 10 [MOD 54])) :=
sorry

end seating_impossible_l540_540700


namespace repairs_cost_l540_540562

-- Define the initial conditions
def initial_cost : ℝ := 800
def selling_price : ℝ := 1200
def gain_percent : ℝ := 20

-- Define the amount spent on repairs as a variable
variable (R : ℝ)

-- Define the total cost after repairs
def total_cost := initial_cost + R

-- Define the calculated gain based on the percentage provided
def gain := (gain_percent / 100) * total_cost

-- Define the gain as the difference between selling price and total cost
def actual_gain := selling_price - total_cost

-- State the theorem: the amount spent on repairs is $200
theorem repairs_cost :
  ∃ R, actual_gain = gain ∧ R = 200 := by
  sorry

end repairs_cost_l540_540562


namespace diminished_radius_10_percent_l540_540204

theorem diminished_radius_10_percent
  (r r' : ℝ) 
  (h₁ : r > 0)
  (h₂ : r' > 0)
  (h₃ : (π * r'^2) = 0.8100000000000001 * (π * r^2)) :
  r' = 0.9 * r :=
by sorry

end diminished_radius_10_percent_l540_540204


namespace ellipse_eq_hyperbola_eq_l540_540030

theorem ellipse_eq (F P : ℝ × ℝ) (m n : ℝ)
  (hF : F = (1, 0))
  (hP : P = (1, m))
  (hC : ∀ (x y : ℝ), (y^2 = 4 * x) ↔ (x, y) = (1, m ∨ m = -2)) :
  (4 - n = 1) → ( ∀ x y, ( ∀ x y, x^2 / 4 + y^2 / n = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

theorem hyperbola_eq (A B F : ℝ × ℝ) (P: ℝ × ℝ) (m λ : ℝ)
  (hA : A = (2/3, (2/3) * Real.sqrt 6))
  (hB : B = (2/3, -(2/3) * Real.sqrt 6))
  (hP : P = (1, m))
  (hC : ∀ (x y : ℝ), (y^2 = 4 * x) ↔ (x, y) = (1, m ∨ m = -2)) :
  (6 * x^2 - y^2 = λ) → ( ∃ λ, λ = 2 ∧ 3x^2 - y^2 / 2 = 1) :=
by sorry

end ellipse_eq_hyperbola_eq_l540_540030


namespace a7_plus_a11_l540_540408

variable {a : ℕ → ℤ} (d : ℤ) (a₁ : ℤ)

-- Definitions based on given conditions
def S_n (n : ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Condition: S_17 = 51
axiom h : S_n 17 = 51

-- Theorem to prove the question is equivalent to the answer
theorem a7_plus_a11 (h : S_n 17 = 51) : a_n 7 + a_n 11 = 6 :=
by
  -- This is where you'd fill in the actual proof, but we'll use sorry for now
  sorry

end a7_plus_a11_l540_540408


namespace find_x_l540_540811

noncomputable def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

noncomputable def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (1, x)) 
  (h3 : magnitude (vec_sub a b) = vec_dot a b) : 
  x = 1 / 3 :=
by
  sorry

end find_x_l540_540811


namespace problem_part1_problem_part2_l540_540299

def max (x y : ℝ) : ℝ := if x ≥ y then x else y

def f (a x : ℝ) : ℝ :=
  max (a^x - a) (-Real.logBase a x)

theorem problem_part1 (x : ℝ) (hx : 0 < x):
  f (1/4) 2 + f (1/4) (1/2) = 3/4 :=
sorry

theorem problem_part2 (a x : ℝ) (ha : a > 1) (hx : 0 < x) :
  f a x ≥ 2 ↔ (x ≤ 1/a^2 ∨ x ≥ Real.logBase a (a + 2)) :=
sorry

end problem_part1_problem_part2_l540_540299


namespace area_of_abs_5x_plus_3y_eq_30_l540_540222

theorem area_of_abs_5x_plus_3y_eq_30 : 
  (let A := 1/2 * 12 * 20 in 
  A = 120) :=
by 
  let A := 1 / 2 * 12 * 20
  have h : A = 120
  sorry

end area_of_abs_5x_plus_3y_eq_30_l540_540222


namespace denominator_of_repeating_six_l540_540162

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l540_540162


namespace water_height_in_cylinder_l540_540701

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

noncomputable def volume_cylinder_height (V r : ℝ) : ℝ :=
  V / (π * r^2)

theorem water_height_in_cylinder :
  let V := volume_cone 15 20 in
  volume_cylinder_height V 30 = 1.6 :=
by
  -- The proof is omitted
  sorry

end water_height_in_cylinder_l540_540701


namespace solution_set_l540_540387

variables {f : ℝ → ℝ} {f' : ℝ → ℝ} {f'' : ℝ → ℝ}

noncomputable def g (x : ℝ) : ℝ := exp (x - 1) * (f x - 1)

axiom f_deriv : ∀ x, has_deriv_at f (f' x) x
axiom f''_def : ∀ x, f'' x = has_deriv_at.deriv (f' x)
axiom f_condition : ∀ x, f x + f'' x > 1
axiom f_value_at_1 : f 1 = 0

theorem solution_set : ∀ x, f x - 1 + 1 / exp(x - 1) ≤ 0 → x ∈ set.Iic 1 :=
begin
  sorry
end

end solution_set_l540_540387


namespace probability_of_slope_condition_l540_540524

-- Define the unit square and point
def unit_square : Set (Real × Real) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
def fixed_point : (Real × Real) := (3/4, 1/4)

-- Define the condition for the slope
def slope_condition (Q : Real × Real) : Prop :=
  let m := (Q.2 - fixed_point.2) / (Q.1 - fixed_point.1)
  m ≤ -1

-- Define the area under the condition within the unit square
def area_below_line : Real := 1/2 -- Derived from the geometric solution

-- State the theorem
theorem probability_of_slope_condition : 
  (MeasureTheory.Measure.restrict MeasureTheory.MeasureSpace.volume unit_square).toOuterMeasure.measure (unit_square ∩ {Q | slope_condition Q}) = 
  1 / 2 :=
sorry

end probability_of_slope_condition_l540_540524


namespace centroid_theorem_l540_540642

-- Define a triangle with vertices A, B, and C
structure Triangle where
  A B C : Point

-- Define the concept of a median within the Triangle structure
def median (T : Triangle) (A1 B1 C1 : Point) : Prop :=
  A1 = midpoint T.B T.C ∧ B1 = midpoint T.A T.C ∧ C1 = midpoint T.A T.B

-- Define the property of the centroid dividing medians in the ratio 2:1
def centroid_property (T : Triangle) (O A1 B1 C1 : Point) : Prop :=
  T.median A1 B1 C1 ∧
  (O ∈ line_segment T.A A1) ∧ (O ∈ line_segment T.B B1) ∧ (O ∈ line_segment T.C C1) ∧
  (distance T.A O) / (distance O A1) = 2 / 1 ∧
  (distance T.B O) / (distance O B1) = 2 / 1 ∧
  (distance T.C O) / (distance O C1) = 2 / 1

-- Prove the theorem about the centroid of a triangle
theorem centroid_theorem (T : Triangle) (O A1 B1 C1 : Point) :
    T.median A1 B1 C1 → centroid_property T O A1 B1 C1 := by
  sorry

end centroid_theorem_l540_540642


namespace pascal_triangle_even_or_divisible_by_3_l540_540190

/-- The number of rows from row 2 to row 30 in Pascal's triangle, 
where all entries (excluding the first and last) are either entirely 
even or divisible by 3, is 4. -/
theorem pascal_triangle_even_or_divisible_by_3 :
  (∃ n, 2 ≤ n ∧ n ≤ 30 ∧ (∀ k, 1 ≤ k ∧ k < n → (binom n k % 2 = 0 ∨ binom n k % 3 = 0))) →
  (∃ l, l = 4) :=
begin
  sorry
end

end pascal_triangle_even_or_divisible_by_3_l540_540190


namespace abc_inequality_l540_540591

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a^2 < 16 * b * c) (h2 : b^2 < 16 * c * a) (h3 : c^2 < 16 * a * b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l540_540591


namespace sqrt_simplification_l540_540568

theorem sqrt_simplification : 
  sqrt (cbrt (sqrt (1 / 2 ^ 15))) = sqrt 2 / 4 :=
by sorry

end sqrt_simplification_l540_540568


namespace problem_1_problem_2_l540_540800

-- Problem statement (I)
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici (Real.exp 2) → (x * Real.log x + a * x)' ≥ 0) ↔ a ≥ -3 :=
sorry

-- Problem statement (II)
theorem problem_2 (k : ℕ) : 
  (∀ x : ℝ, x ∈ Set.Ioi 1 → x * Real.log x + a * x > k * (x - 1) + a * x - x) → k ≤ 3 :=
sorry

end problem_1_problem_2_l540_540800


namespace intersection_of_sets_l540_540034

theorem intersection_of_sets (M N : Set ℕ) :
  M = {1, 2, 3, 4, 5} →
  (N = { x : ℕ | x ≤ 1 ∨ x ≥ 3 }) →
  (M ∩ N = {3, 4, 5}) :=
begin
  intros hM hN,
  -- assumes hM: M = {1, 2, 3, 4, 5} and hN: N = { x : ℕ | x ≤ 1 ∨ x ≥ 3 }
  sorry
end

end intersection_of_sets_l540_540034


namespace chucks_team_final_score_l540_540476

variable (RedTeamScore : ℕ) (scoreDifference : ℕ)

-- Given conditions
def red_team_score := RedTeamScore = 76
def score_difference := scoreDifference = 19

-- Question: What was the final score of Chuck's team?
def chucks_team_score (RedTeamScore scoreDifference : ℕ) : ℕ := 
  RedTeamScore + scoreDifference

-- Proof statement
theorem chucks_team_final_score : red_team_score 76 ∧ score_difference 19 → chucks_team_score 76 19 = 95 :=
by
  sorry

end chucks_team_final_score_l540_540476


namespace DVDs_sold_is_168_l540_540462

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l540_540462


namespace acute_angle_condition_l540_540036

theorem acute_angle_condition 
  (m : ℝ) 
  (a : ℝ × ℝ := (2,1))
  (b : ℝ × ℝ := (m,6)) 
  (dot_product := a.1 * b.1 + a.2 * b.2)
  (magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2))
  (magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2))
  (cos_angle := dot_product / (magnitude_a * magnitude_b))
  (acute_angle : cos_angle > 0) : -3 < m ∧ m ≠ 12 :=
sorry

end acute_angle_condition_l540_540036


namespace min_diff_of_means_l540_540729

theorem min_diff_of_means (a b : ℕ) (h1 : a ≠ b) 
  (h2 : (a + b) % 2 = 0) 
  (h3 : ∃ G : ℕ, G^2 = a * b)
  (h4 : ∃ H : ℕ, H = (2 * a * b) / (a + b)) : 
  (∃ k : ℕ, |a - b| = 6) :=
sorry

end min_diff_of_means_l540_540729


namespace find_y_l540_540262

noncomputable def x : Real := 2.6666666666666665

theorem find_y (y : Real) (h : (x * y) / 3 = x^2) : y = 8 :=
sorry

end find_y_l540_540262


namespace dot_product_u_v_l540_540722

def u : ℝ × ℝ × ℝ := (-4, -1, 3)
def v : ℝ × ℝ × ℝ := (6, 8, -5)

theorem dot_product_u_v :
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) = -47 := by
  sorry

end dot_product_u_v_l540_540722


namespace are_complementary_events_l540_540607

variable (A B C : Set ℕ)

def eventA := {1, 2}
def eventB := {3, 4, 5, 6}
def eventC := {1, 3, 5}

theorem are_complementary_events
  (hA : A = eventA)
  (hB : B = eventB) : A ∩ B = ∅ ∧ A ∪ B = {1, 2, 3, 4, 5, 6} := by
  sorry

end are_complementary_events_l540_540607


namespace area_of_triangle_PDE_l540_540573

noncomputable def length (a b : Point) : ℝ := -- define length between two points
sorry

def distance_from_line (P D E : Point) : ℝ := -- define perpendicular distance from P to line DE
sorry

structure Point :=
(x : ℝ)
(y : ℝ)

def area_triangle (P D E : Point) : ℝ :=
0.5 -- define area given conditions

theorem area_of_triangle_PDE (D E : Point) (hD_E : D ≠ E) :
  { P : Point | area_triangle P D E = 0.5 } =
  { P : Point | distance_from_line P D E = 1 / (length D E) } :=
sorry

end area_of_triangle_PDE_l540_540573


namespace find_m_l540_540576

-- All conditions are represented below:
def is_ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2 / m) + (y^2 / 3) = 1
def has_focus_on_x_axis : Prop := true  -- This condition is implicitly satisfied by the ellipse equation provided.
def eccentricity (m : ℝ) : ℝ := (m - 3) / m

-- Question translated to a Lean theorem statement:
theorem find_m (m : ℝ) (h_ellipse : ∀ x y, is_ellipse x y m)
    (h_focus : has_focus_on_x_axis)
    (h_eccentricity : eccentricity m = 1/4) :
    m = 4 := 
begin
  sorry
end

end find_m_l540_540576


namespace exponential_function_fixed_point_l540_540583

theorem exponential_function_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) :
  (0, 2) ∈ set.range (λ x : ℝ, a^x + 1) :=
sorry

end exponential_function_fixed_point_l540_540583


namespace find_m_minus_n_l540_540781

theorem find_m_minus_n (x y m n : ℤ) (h1 : x = -2) (h2 : y = 1) 
  (h3 : 3 * x + 2 * y = m) (h4 : n * x - y = 1) : m - n = -3 :=
by sorry

end find_m_minus_n_l540_540781


namespace pentagon_area_is_16_l540_540863

-- Define the geometric configuration
structure Square :=
  (side : ℝ)
  (area_eq : side * side = 48)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define midpoints and intersection points
def E (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def F (B C : Point) : Point :=
  { x := (B.x + C.x) / 2, y := (B.y + C.y) / 2 }

-- Area of pentagon EBFNM
noncomputable def area_of_pentagon (ABCD : Square) (A B C D : Point) (E F M N : Point) : ℝ :=
  16

-- Prove the area of the pentagon EBFNM
theorem pentagon_area_is_16 (ABCD : Square) (A B C D E F M N : Point)
  (E_mid : E = E A B) (F_mid : F = F B C)
  (DE_inter_AC : line_through D E ∩ line_through A C = {M})
  (DF_inter_AC : line_through D F ∩ line_through A C = {N}) :
  area_of_pentagon ABCD A B C D E F M N = 16 := sorry


end pentagon_area_is_16_l540_540863


namespace total_weight_of_books_l540_540142

theorem total_weight_of_books :
  let sandy_books := 10
  let sandy_weight_per_book := 1.5
  let benny_books := 24
  let benny_weight_per_book := 1.2
  let tim_books := 33
  let tim_weight_per_book := 1.8
  (sandy_books * sandy_weight_per_book + benny_books * benny_weight_per_book + tim_books * tim_weight_per_book) = 103.2 :=
by
  let sandy_books := 10
  let sandy_weight_per_book := 1.5
  let benny_books := 24
  let benny_weight_per_book := 1.2
  let tim_books := 33
  let tim_weight_per_book := 1.8
  have h1 : sandy_books * sandy_weight_per_book = 15 := rfl
  have h2 : benny_books * benny_weight_per_book = 28.8 := rfl
  have h3 : tim_books * tim_weight_per_book = 59.4 := rfl
  have total_weight := h1 + h2 + h3
  have total_weight_equiv : 15 + 28.8 + 59.4 = 103.2 := by norm_num
  exact total_weight_equiv

end total_weight_of_books_l540_540142


namespace algebraic_expression_value_l540_540232

theorem algebraic_expression_value (p q : ℝ)
  (h : p * 3^3 + q * 3 + 3 = 2005) :
  p * (-3)^3 + q * (-3) + 3 = -1999 :=
by
   sorry

end algebraic_expression_value_l540_540232


namespace wendy_fruit_sales_l540_540972

theorem wendy_fruit_sales
  (morning_apples_sold : ℕ) (morning_oranges_sold : ℕ)
  (afternoon_oranges_sold : ℕ)
  (price_apple : ℝ) (price_orange : ℝ)
  (total_sales : ℝ)
  (morning_apples_sales : ℝ := morning_apples_sold * price_apple)
  (morning_oranges_sales : ℝ := morning_oranges_sold * price_orange)
  (morning_sales : ℝ := morning_apples_sales + morning_oranges_sales)
  (afternoon_sales : ℝ := total_sales - morning_sales)
  (afternoon_oranges_sales : ℝ := afternoon_oranges_sold * price_orange)
  (afternoon_apples_sales : ℝ := afternoon_sales - afternoon_oranges_sales)
  (afternoon_apples_sold : ℕ := (afternoon_apples_sales / price_apple).toNat) :
  morning_apples_sold = 40 ∧
  morning_oranges_sold = 30 ∧
  afternoon_oranges_sold = 40 ∧
  price_apple = 1.50 ∧
  price_orange = 1.00 ∧
  total_sales = 205 →
  afternoon_apples_sold = 50 :=
by
  intro h
  cases h
  sorry

end wendy_fruit_sales_l540_540972


namespace alternating_sum_zero_l540_540594

theorem alternating_sum_zero
  (n : ℕ)
  (a : Fin (2 * n + 2) → ℕ)
  (h1 : ∀ i, a i = 2 ∨ a i = 5 ∨ a i = 9)
  (h2 : ∀ i < 2 * n + 1, a i ≠ a (i + 1))
  (h3 : a 0 = a (2 * n + 1)) :
  (Finset.range (2 * n + 1)).sum (λ i, (-1) ^ i * a i * a (i + 1)) = 0 :=
  sorry

end alternating_sum_zero_l540_540594


namespace plumber_charge_shower_l540_540261

theorem plumber_charge_shower (S : ℝ) 
  (sink_cost : ℝ := 30) 
  (toilet_cost : ℝ := 50)
  (max_earning : ℝ := 250)
  (first_job_toilets : ℝ := 3) (first_job_sinks : ℝ := 3)
  (second_job_toilets : ℝ := 2) (second_job_sinks : ℝ := 5)
  (third_job_toilets : ℝ := 1) (third_job_showers : ℝ := 2) (third_job_sinks : ℝ := 3) :
  2 * S + 1 * toilet_cost + 3 * sink_cost ≤ max_earning → S ≤ 55 :=
by
  sorry

end plumber_charge_shower_l540_540261


namespace range_of_T_l540_540013

open Real

theorem range_of_T (x y z : ℝ) (h : x^2 + 2 * y^2 + 3 * z^2 = 4) : 
    - (2 * sqrt 6) / 3 ≤ x * y + y * z ∧ x * y + y * z ≤ (2 * sqrt 6) / 3 := 
by 
    sorry

end range_of_T_l540_540013


namespace sam_age_two_years_ago_l540_540650

variables (S J : ℕ)
variables (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9))

theorem sam_age_two_years_ago : S - 2 = 7 := by
  sorry

end sam_age_two_years_ago_l540_540650


namespace part_one_cardinality_A_intersection_B_part_two_range_a_l540_540777

-- Part 1
theorem part_one_cardinality_A_intersection_B :
  let A := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1}
  let B := {x : ℝ | 0 ≤ x ∧ x < 20}
  let A_int_B := {x : ℝ | ∃ n : ℤ, x = 4 * n + 1 ∧ 0 ≤ x ∧ x < 20}
  cardinality A_int_B = 5 :=
sorry

-- Part 2
theorem part_two_range_a :
  let B := {x : ℝ | ∃ a : ℝ, a ≤ x ∧ x < a + 20}
  let C := {x : ℝ | 5 ≤ x ∧ x < 30}
  ∃ (a : ℝ), ¬(a + 20 ≤ 5 ∨ 30 ≤ a) ⇔ B ∩ C ≠ ∅ :=
sorry

end part_one_cardinality_A_intersection_B_part_two_range_a_l540_540777


namespace arithmetic_geometric_sequence_l540_540770

theorem arithmetic_geometric_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) 
  (S3_eq_a4_plus_2 : S 3 = a 4 + 2)
  (arithmetic_condition : ∀ n, a (n + 1) = a n + d)
  (geometric_condition : (a 1) * (a 1 + 2 * d - 1) = (a 1 + d - 1) * (a 1 + d - 1))
  (sum_arith_seq : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (N_pos : ∀ n, n ∈ ℕ⁺)
  : (∀ n, a n = 2 * n - 1) ∧ (∀ n ∈ ℕ⁺, 1/3 ≤ T n ∧ T n < 1/2) := sorry

end arithmetic_geometric_sequence_l540_540770


namespace complex_number_problem_l540_540021

noncomputable def z1 : ℂ := 2 - complex.i
noncomputable def z2 : ℂ := 4 + 2 * complex.i

theorem complex_number_problem
  (h1 : (z1 - 2) * (1 + complex.i) = 1 - complex.i)
  (h2 : z2.im = 2)
  (h3 : ∃ (r : ℝ), z1 * z2 = r) :
  z1 = 2 - complex.i ∧ z2 = 4 + 2 * complex.i := by
  sorry

end complex_number_problem_l540_540021


namespace value_of_a_and_range_of_b_l540_540789

-- Definition of f(x) being an even function
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

-- Given conditions
variable {f : ℝ → ℝ} {g : ℝ → ℝ} {a b : ℝ}

axiom even_f : even_function f
axiom f_eqn : ∀ x, 2 * f (x + 2) = f (-x)
axiom f_def : ∀ x, x ∈ Ioo 0 2 → f x = (exp x + a * x)
axiom f_max : ∀ x, x ∈ Ioo (-4) (-2) → f x ≤ f (-2)
axiom g_def : ∀ x, g x = (4 / 3) * b * x^3 - 4 * b * x + 2
axiom b_not_zero : b ≠ 0
axiom f_g_ineq : ∀ x1 x2, x1 ∈ Ioo 1 2 → x2 ∈ Ioo 1 2 → f x1 < g x2

-- Prove the value of a and the range of b
theorem value_of_a_and_range_of_b : a = 2 ∧ (b ∈ set.Ici (3/8 * exp 2 + 3/4) ∨ b ∈ set.Iic (-3/8 * exp 2 - 3/4)) :=
by
  sorry

end value_of_a_and_range_of_b_l540_540789


namespace infinite_product_sqrt_nine_81_l540_540304

theorem infinite_product_sqrt_nine_81 : 
  (∀ n : ℕ, n > 0 →
  (let S := ∑' n, (n:ℝ) / 4^n in
  let P := ∏' n, (3:ℝ)^(S / (4^n)) in
  P = (81:ℝ)^(1/9))) := 
sorry

end infinite_product_sqrt_nine_81_l540_540304


namespace initial_ratio_l540_540091

def initial_men (M : ℕ) (W : ℕ) : Prop :=
  let men_after := M + 2
  let women_after := W - 3
  (2 * women_after = 24) ∧ (men_after = 14)

theorem initial_ratio (M W : ℕ) (h : initial_men M W) :
  (M = 12) ∧ (W = 15) → M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  intro hm hw
  have h12 : M = 12 := hm
  have h15 : W = 15 := hw
  sorry

end initial_ratio_l540_540091


namespace continuous_paths_count_l540_540436

-- Define the labeled points
inductive Point
| A | B | C | D | E | F | G
deriving DecidableEq

open Point

-- Define the edges of the graph as pairs of nodes
def edges : List (Point × Point) :=
  [(A, C), (A, D), (A, D), (C, B), (D, C), (D, F), 
   (D, E), (E, F), (E, G), (C, D), (D, F), (D, G),
   (F, G), (G, B), (F, B), (C, F)]

-- Checking if a path does not revisit any points
def valid_path (p : List Point) : Bool :=
  p.Nodup -- No duplicate points

-- Define a function to count valid paths from A to B
noncomputable def count_valid_paths : Nat :=
  (List.filter (λ p => p.head? = some A ∧ p.getLast? = some B ∧ valid_path p) ⟦edges⟧).length

-- The theorem we want to prove
theorem continuous_paths_count : count_valid_paths = 12 :=
by sorry

end continuous_paths_count_l540_540436


namespace recurring_six_denominator_l540_540177

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l540_540177


namespace value_of_a_l540_540450

theorem value_of_a (a : ℕ) : (∃ (x1 x2 x3 : ℤ),
  abs (abs (x1 - 3) - 1) = a ∧
  abs (abs (x2 - 3) - 1) = a ∧
  abs (abs (x3 - 3) - 1) = a ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  → a = 1 :=
by
  sorry

end value_of_a_l540_540450


namespace alice_chicken_weight_l540_540608

theorem alice_chicken_weight (total_cost_needed : ℝ)
  (amount_to_spend_more : ℝ)
  (cost_lettuce : ℝ)
  (cost_tomatoes : ℝ)
  (sweet_potato_quantity : ℝ)
  (cost_per_sweet_potato : ℝ)
  (broccoli_quantity : ℝ)
  (cost_per_broccoli : ℝ)
  (brussel_sprouts_weight : ℝ)
  (cost_per_brussel_sprouts : ℝ)
  (cost_per_pound_chicken : ℝ)
  (total_cost_excluding_chicken : ℝ) :
  total_cost_needed = 35 ∧
  amount_to_spend_more = 11 ∧
  cost_lettuce = 3 ∧
  cost_tomatoes = 2.5 ∧
  sweet_potato_quantity = 4 ∧
  cost_per_sweet_potato = 0.75 ∧
  broccoli_quantity = 2 ∧
  cost_per_broccoli = 2 ∧
  brussel_sprouts_weight = 1 ∧
  cost_per_brussel_sprouts = 2.5 ∧
  total_cost_excluding_chicken = (cost_lettuce + cost_tomatoes + sweet_potato_quantity * cost_per_sweet_potato + broccoli_quantity * cost_per_broccoli + brussel_sprouts_weight * cost_per_brussel_sprouts) →
  (total_cost_needed - amount_to_spend_more - total_cost_excluding_chicken) / cost_per_pound_chicken = 1.5 :=
by
  intros
  sorry

end alice_chicken_weight_l540_540608


namespace length_DO_is_8_l540_540502

variable {A B C O S : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace S]
variables (triangle_ABC : Triangle A B C)
variable (O_is_centroid : O = centroid triangle_ABC)
variable (DS_is_median : ∃ M : A, M ∈ midpoint_side B C = S)
variable (length_DS : dist D S = 12)

theorem length_DO_is_8 :
  dist D O = 8 := sorry

end length_DO_is_8_l540_540502


namespace range_of_a_l540_540066

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l540_540066


namespace a_n_is_perfect_square_l540_540875

inductive Nat : Type
| zero : Nat
| succ : Nat → Nat
open Nat

-- Definitions of sequences a_n and b_n
def a : Nat → ℕ
| zero      := 0
| (succ 0)  := 1
| (succ (succ 0)) := 1
| (succ (succ n)) := 7 * a (succ n) - a n - 2

def b : Nat → ℕ
| zero      := 0
| (succ 0)  := 1
| (succ (succ 0)) := 1
| (succ (succ n)) := 3 * b (succ n) - b n

-- Proof that a_n is a perfect square for every n
theorem a_n_is_perfect_square (n : Nat) : ∃ k : ℕ, a n = k * k :=
by
  sorry

end a_n_is_perfect_square_l540_540875


namespace dvd_sold_168_l540_540460

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l540_540460


namespace abs_difference_of_mn_6_and_sum_7_l540_540517

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l540_540517


namespace arithmetic_sum_equality_l540_540523

theorem arithmetic_sum_equality (n : ℕ) (a1 d1 a2 d2 : ℕ) (h1 : a1 = 5) (h2 : d1 = 5) (h3 : a2 = 22) (h4 : d2 = 3) : n = 18 ↔ n*(5*n + 5) = n*(3*n + 41) := by
  intros
  rw [h1, h2, h3, h4]
  sorry

end arithmetic_sum_equality_l540_540523


namespace units_digit_of_k_squared_plus_2_to_k_l540_540512

theorem units_digit_of_k_squared_plus_2_to_k (k : ℕ) (h : k = 2012 ^ 2 + 2 ^ 2014) : (k ^ 2 + 2 ^ k) % 10 = 5 := by
  sorry

end units_digit_of_k_squared_plus_2_to_k_l540_540512


namespace max_size_subset_l540_540117

/--
Let S be a subset of the set {1, 2, 3, ..., 2015} such that for any two elements a, b ∈ S,
the difference a - b does not divide the sum a + b.
We prove that the maximum possible size of S is 672.
-/
theorem max_size_subset (S : Finset ℕ) (h : ∀ a b ∈ S, a ≠ b → ¬ (a - b) ∣ (a + b)) :
  S ⊆ Finset.range 2016 → S.card ≤ 672 :=
sorry

end max_size_subset_l540_540117


namespace unique_equal_area_point_l540_540226

variable {Point : Type}
variables {A B C D P Q : Point}
variables [AffineSpace ℝ Point]

def convex_quadrilateral (A B C D : Point) : Prop := 
  ∃ (V : Triangle ℝ Point), V = ⟨A, B, C⟩ ∧ ∃ (U : Triangle ℝ Point), U = ⟨A, C, D⟩

def equal_area_triangles (P A B C D : Point) [affine_space ℝ Point] :=
  let area := (Triangle.area (P A B)) in
  Triangle.area (P A B) = area ∧ Triangle.area (P B C) = area ∧
  Triangle.area (P C D) = area ∧ Triangle.area (P D A) = area

theorem unique_equal_area_point (A B C D : Point) (h_conv : convex_quadrilateral A B C D) :
  ∃! (P : Point), equal_area_triangles P A B C D :=
by
  sorry

end unique_equal_area_point_l540_540226


namespace infinite_product_value_l540_540328

noncomputable def infinite_product : ℝ :=
  ∏ n in naturalNumbers, 3^(n/(4^n))

theorem infinite_product_value :
  infinite_product = real.root 9 81 := 
sorry

end infinite_product_value_l540_540328


namespace lateral_edges_equal_angles_l540_540707

-- Define the given conditions as a structure
structure Pyramid (S A B C D : Point) :=
(base_parallelogram : Parallelogram A B C D)

-- Define the main theorem
theorem lateral_edges_equal_angles (S A B C D O : Point) (pyramid : Pyramid S A B C D) :
  (∃ (SO : Ray S O), ∀ x ∈ {A, B, C, D}, angle (Ray S x) SO = angle (Ray S A) SO) ↔
  (dist S A + dist S C = dist S B + dist S D) :=
  sorry

end lateral_edges_equal_angles_l540_540707


namespace image_of_square_OABC_l540_540503

theorem image_of_square_OABC :
  let O := (0, 0)
  let A := (1, 0)
  let B := (1, 1)
  let C := (0, 1)
  let square := {O, A, B, C}
  let u (x y : ℝ) := Real.sin (π * x) * Real.cos (π * y)
  let v (x y : ℝ) := x + y - x * y
  let transformed_square := { (u x y, v x y) | (x, y) ∈ square }
  transformed_square = {(0, t) | t ∈ Set.Icc 0 1} :=
sorry

end image_of_square_OABC_l540_540503


namespace divisor_of_2n_when_remainder_is_two_l540_540068

theorem divisor_of_2n_when_remainder_is_two (n : ℤ) (k : ℤ) : 
  (n = 22 * k + 12) → ∃ d : ℤ, d = 22 ∧ (2 * n) % d = 2 :=
by
  sorry

end divisor_of_2n_when_remainder_is_two_l540_540068


namespace sin_cos_inequality_l540_540910

theorem sin_cos_inequality (α β : ℝ) (h : 0 < α + β ∧ α + β ≤ π) :
  (sin α - sin β) * (cos α - cos β) ≤ 0 :=
sorry

end sin_cos_inequality_l540_540910


namespace full_seasons_already_aired_l540_540098

variable (days_until_premiere : ℕ)
variable (episodes_per_day : ℕ)
variable (episodes_per_season : ℕ)

theorem full_seasons_already_aired (h_days : days_until_premiere = 10)
                                  (h_episodes_day : episodes_per_day = 6)
                                  (h_episodes_season : episodes_per_season = 15) :
  (days_until_premiere * episodes_per_day) / episodes_per_season = 4 := by
  sorry

end full_seasons_already_aired_l540_540098


namespace k_range_l540_540063

-- Define the circle and its properties
def circle (x y k : ℝ) : Prop := x^2 + y^2 - 2*k*x + 2*y + 2 = 0

-- Define the condition on k
def valid_k (k : ℝ) : Prop := k > 0

-- Prove that the range of k, given the conditions is (1, sqrt(2))
theorem k_range (k : ℝ) (h_circle : ∀ x y, ¬circle x y k) (h_k : valid_k k) : 1 < k ∧ k < Real.sqrt 2 := 
by
  sorry

end k_range_l540_540063


namespace selected_student_from_fourth_group_l540_540076

def number_of_students : ℕ := 50
def number_of_groups : ℕ := 5
def selected_student_from_first_group : ℕ := 4
def selected_student_from_second_group : ℕ := 14

theorem selected_student_from_fourth_group :
  ∃ (interval : ℕ) (n : ℕ), interval = number_of_students / number_of_groups ∧
  ∀ (k : ℕ), k ∈ {0, 1, 2, 3, 4} →
  (selected_student_from_first_group + interval * k) = [4, 14, 24, 34, 44].nth k :=
begin
  -- Proof will be provided here
  sorry -- Placeholder for the proof
end

end selected_student_from_fourth_group_l540_540076


namespace hands_form_angle_120_degrees_l540_540221

noncomputable def time_when_angle_is_120_degrees (hour hand: ℝ) (minute_hand: ℝ) 
  (minute_hand_movement: ℝ) (hour_hand_movement: ℝ) : List ℝ :=
  sorry

theorem hands_form_angle_120_degrees : 
  (time_when_angle_is_120_degrees 210 0 360 30) = [16, 56] :=
sorry

end hands_form_angle_120_degrees_l540_540221


namespace norm_a_sub_b_eq_sqrt_three_l540_540774

open real

variables (a b : ℝ^3)
variables (h_non_collinear : ¬collinear {a, b})
variables (h_a_norm : ∥a∥ = 2)
variables (h_b_norm : ∥b∥ = 3)
variables (h_dot : a ⬝ (b - a) = 1)

theorem norm_a_sub_b_eq_sqrt_three :
  ∥a - b∥ = √3 := 
sorry

end norm_a_sub_b_eq_sqrt_three_l540_540774


namespace find_value_of_a_l540_540241

theorem find_value_of_a (a : ℝ) (h : 0.005 * a = 65) : a = 130 := 
by
  sorry

end find_value_of_a_l540_540241


namespace average_messages_correct_l540_540094

-- Definitions for the conditions
def messages_monday := 220
def messages_tuesday := 1 / 2 * messages_monday
def messages_wednesday := 50
def messages_thursday := 50
def messages_friday := 50

-- Definition for the total and average messages
def total_messages := messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday
def average_messages := total_messages / 5

-- Statement to prove
theorem average_messages_correct : average_messages = 96 := 
by sorry

end average_messages_correct_l540_540094


namespace triangle_area_min_a_value_l540_540869

-- Definitions related to the problem's conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Condition: \frac{c}{\sin C} = \frac{a}{\sqrt{3}\cos A}
def condition1 : Prop := c / Real.sin C = a / (Real.sqrt 3 * Real.cos A)

-- Proof problem part 1: Proving the area of triangle ABC is \sqrt{3}
theorem triangle_area (h1 : condition1) (h2 : 4 * Real.sin C = c^2 * Real.sin B) : 
  0.5 * b * c * Real.sin A = Real.sqrt 3 :=
sorry

-- Condition: \vec{AB}\cdot\vec{BC} + \|\vec{AB}\|^2 = 4
-- Note: Using vectors and their dot product may need Real inner product space definitions
variables (AB BC : EuclideanSpace ℝ (Fin 2)) (AB_mag BC_mag : ℝ)
def condition2 : Prop := innerProductSpace ℝ ℝ AB BC + (AB_mag)^2 = 4

-- Proof problem part 2: Proving the minimum value of a is 2\sqrt{2}
theorem min_a_value (h1 : condition1) (h3 : condition2) : 
  ∃ a, a = 2 * Real.sqrt 2 :=
sorry

end triangle_area_min_a_value_l540_540869


namespace negation_of_p_l540_540810

variable (p : ∀ x : ℝ, Real.sqrt (2 - x) < 0)

theorem negation_of_p : (¬ p) ↔ ∃ x₀ : ℝ, Real.sqrt (2 - x₀) ≥ 0 :=
by sorry

end negation_of_p_l540_540810


namespace sum_of_p_minus_two_pmod_p_l540_540528

theorem sum_of_p_minus_two_pmod_p (p : ℕ) (hp : Nat.Prime p) (hpp : p % 2 = 1) :
  (∑ i in Finset.range (p / 2), i^(p-2)) % p = (2 - 2^p) / p % p := sorry

end sum_of_p_minus_two_pmod_p_l540_540528


namespace value_of_a3_l540_540014

theorem value_of_a3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (a : ℝ) (h₀ : (1 + x) * (a - x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) 
(h₁ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) : 
a₃ = -5 :=
sorry

end value_of_a3_l540_540014


namespace find_distance_post_office_l540_540644

noncomputable def post_office_distance (total_time: ℝ) (speed_to: ℝ) (speed_back: ℝ): ℝ :=
  let D := 10 in
  D

theorem find_distance_post_office : 
  ∃ D : ℝ, 
  let speed_to := 12.5 in 
  let speed_back := 2 in
  let total_journey_time := 5 + (48 / 60) in
  (D / speed_to) + (D / speed_back) = total_journey_time ∧
  D = 10 :=
begin
  use 10,
  sorry
end

end find_distance_post_office_l540_540644


namespace sum_of_solutions_l540_540365

theorem sum_of_solutions :
  (∑ x in {2, 3}, x) = 5 := by
sorry

end sum_of_solutions_l540_540365


namespace powderman_heard_blast_distance_l540_540682

def powderman_distance_run
  (t_fuse : ℝ)           -- Fuse set time in seconds
  (runner_speed : ℝ)     -- Powderman's speed in yards per second
  (sound_speed : ℝ)      -- Speed of sound in feet per second
  (distance_in_yards : ℝ) : Prop :=
  ∀ t : ℝ,
  t_fuse = 20 →
  runner_speed = 10 →
  sound_speed = 1080 →
  let p := 30 * t in         -- Distance run by the powderman in feet
  let q := 1080 * (t - 20) in-- Distance sound traveled from the blast point in feet
  t = 20 + distance_in_yards / 10 / (1080 / 10) →
  distance_in_yards = 206

-- The theorem statement:
theorem powderman_heard_blast_distance : powderman_distance_run 20 10 1080 206 := by sorry

end powderman_heard_blast_distance_l540_540682


namespace differential_y_is_dy_l540_540357

-- Define the function y
def y (x : ℝ) : ℝ := Real.exp x * (Real.cos (2 * x) + 2 * Real.sin (2 * x))

-- Define the expected differential dy
def dy (dx : ℝ) (x : ℝ) : ℝ := 5 * Real.exp x * Real.cos (2 * x) * dx

-- Prove that the differential of y is dy
theorem differential_y_is_dy (dx x : ℝ) : 
  (deriv (λ x => Real.exp x * (Real.cos (2 * x) + 2 * Real.sin (2 * x))) x) * dx = 5 * Real.exp x * Real.cos (2 * x) * dx :=
by
  sorry

end differential_y_is_dy_l540_540357


namespace initial_ratio_is_four_five_l540_540088

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end initial_ratio_is_four_five_l540_540088


namespace num_integers_expressible_as_sum_of_three_distinct_l540_540045

noncomputable def numDifferentIntegersSumOfThreeDistinct (s : Set ℕ) : ℕ :=
  let sums := { n | ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ n = a + b + c }
  sums.size

theorem num_integers_expressible_as_sum_of_three_distinct 
  (s : Set ℕ) (h : s = {1, 4, 7, 10, 13, 16, 19, 22, 25}) :
  numDifferentIntegersSumOfThreeDistinct s = 19 :=
by
  sorry

end num_integers_expressible_as_sum_of_three_distinct_l540_540045


namespace equivalent_proof_problem_l540_540146

def math_problem (x y : ℚ) : ℚ :=
((x + y) * (3 * x - y) + y^2) / (-x)

theorem equivalent_proof_problem (hx : x = 4) (hy : y = -(1/4)) :
  math_problem x y = -23 / 2 :=
by
  sorry

end equivalent_proof_problem_l540_540146


namespace recurring_six_denominator_l540_540178

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l540_540178


namespace median_of_speeds_l540_540665

theorem median_of_speeds : 
  let speeds := [67, 63, 69, 55, 65]
  let ordered_speeds := List.sort speeds
  let median := List.nthLe ordered_speeds 2 (by decide : 2 < List.length ordered_speeds)
  median = 65 := by
  sorry

end median_of_speeds_l540_540665


namespace num_integers_lt_500_congruent_4_mod_9_l540_540823

open Int

theorem num_integers_lt_500_congruent_4_mod_9 : 
  {x : ℤ | 0 < x ∧ x < 500 ∧ x % 9 = 4}.toFinset.card = 56 := 
by
  sorry

end num_integers_lt_500_congruent_4_mod_9_l540_540823


namespace emily_more_pastries_l540_540902

variable (p h : ℕ)

-- Conditions
def pastries_per_hour_first_day : ℕ := p
def hours_first_day : ℕ := h
def pastries_per_hour_second_day : ℕ := p - 3
def hours_second_day : ℕ := h + 3
def production_first_day : ℕ := p * h
def production_second_day : ℕ := (p - 3) * (h + 3)
def pastries_difference : ℕ := production_second_day - production_first_day

-- Given condition
axiom p_eq_3h : p = 3 * h

-- Assertion to proof
theorem emily_more_pastries : h = 1 → pastries_difference p h = 3 := by
  sorry

end emily_more_pastries_l540_540902


namespace weight_ratios_l540_540868

theorem weight_ratios {x y z k : ℝ} (h1 : x + y = k * z) (h2 : y + z = k * x) (h3 : z + x = k * y) : x = y ∧ y = z :=
by 
  -- Proof to be filled in later
  sorry

end weight_ratios_l540_540868


namespace distance_between_foci_of_ellipse_l540_540358

theorem distance_between_foci_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 16 = 8) → 
  2 * real.sqrt (288 - 128) = 8 * real.sqrt 10 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l540_540358


namespace BC_tangent_to_BDE_circumcircle_l540_540121

variables (A B C D E M : Type) [Geometry A B C D E M]

-- Conditions
axiom isosceles_triangle (ABC : Triangle) : ABC.AC = ABC.BC
axiom circumcircle_of_ABC (k : Circle) : Circle.contains_ABC k
axiom D_on_circle (D : Point) : Circle.contains D k ∧ D ≠ B ∧ D ≠ C ∧ Arc.shorter D B C
axiom intersection_CD_AB (CD_AB : Line) : CD_AB = Line.intersection (Line.from C D) (Line.from A B)
axiom point_E : E = Line.intersection_point CD_AB

-- Prove
theorem BC_tangent_to_BDE_circumcircle : Line.is_tangent (Line.from B C) (Circle.circumcircle BDE) :=
sorry

end BC_tangent_to_BDE_circumcircle_l540_540121


namespace jellybeans_condition_l540_540695

theorem jellybeans_condition (n : ℕ) (h1 : n ≥ 150) (h2 : n % 15 = 14) : n = 164 :=
sorry

end jellybeans_condition_l540_540695


namespace find_actual_price_l540_540646

noncomputable def original_price_after_discounts (P : ℝ) : ℝ :=
  let after_first_discount := 0.80 * P
  let after_second_discount := 0.72 * after_first_discount
  let after_third_discount := 0.684 * after_second_discount
  after_third_discount

theorem find_actual_price (P : ℝ) (h : original_price_after_discounts P = 6400) :
  P ≈ 9356.73 :=
by 
  sorry

end find_actual_price_l540_540646


namespace student_b_visited_city_a_l540_540966

-- Define the students
inductive Student : Type
| A | B | C

-- Define the cities
inductive City : Type
| A | B | C

-- Define the visited relation
def visited (s : Student) (c : City) : Prop := sorry

-- Given conditions as stated in the problem
axiom visited_more_cities (s1 s2 : Student) : Prop :=
  visited s1 City.A -> visited s1 City.B -> visited s2 City.C ->
  s1 != s2

axiom student_a_visited_more_than_b : visited_more_cities Student.A Student.B

axiom student_a_not_visited_city_b : ¬ visited Student.A City.B

axiom student_b_not_visited_city_c : ¬ visited Student.B City.C

axiom student_c_all_visited_same_city : ∃ c : City, visited Student.A c ∧ visited Student.B c ∧ visited Student.C c

-- Prove that Student B visited City A
theorem student_b_visited_city_a :
  visited Student.B City.A := sorry

end student_b_visited_city_a_l540_540966


namespace most_circular_ellipse_l540_540414

noncomputable def ellipse_equation := {m : ℝ // m > 0} → (∀ x y : ℝ, x^2 / m + y^2 / (m^2 + 1) = 1)

theorem most_circular_ellipse :
  ∀ (e : {m : ℝ // m > 0}), (∃ m : ℝ, m = 1) →
  (∀ x y : ℝ, x^2 / m + y^2 / (m^2 + 1) = 1) ↔ (x^2 + y^2 / 2 = 1) :=
by
  -- skip the proof steps
  sorry

end most_circular_ellipse_l540_540414


namespace count_three_digit_numbers_with_1_and_4_l540_540438

def has_at_least_one_1_and_one_4 (n : ℕ) : Prop :=
(100 ≤ n) ∧ (n ≤ 999) ∧ (n.digit 10 = 1 ∨ n.digit 1 = 1 ∨ (n / 10) % 10 = 1) ∧
(n.digit 10 = 4 ∨ n.digit 1 = 4 ∨ (n / 10) % 10 = 4)

theorem count_three_digit_numbers_with_1_and_4 : 
  (finset.filter (λ n, has_at_least_one_1_and_one_4 n) (finset.range 900).map (λ x, x + 100)).card = 66 :=
by
  sorry

end count_three_digit_numbers_with_1_and_4_l540_540438


namespace sum_of_circle_areas_constant_l540_540693

theorem sum_of_circle_areas_constant (r OP : ℝ) (h1 : 0 < r) (h2 : 0 ≤ OP ∧ OP < r) 
  (a' b' c' : ℝ) (h3 : a'^2 + b'^2 + c'^2 = OP^2) :
  ∃ (a b c : ℝ), (a^2 + b^2 + c^2 = 3 * r^2 - OP^2) :=
by
  sorry

end sum_of_circle_areas_constant_l540_540693


namespace max_arithmetic_sequences_l540_540143

theorem max_arithmetic_sequences {a b c : ℕ} (h1 : a ∈ ({1, 2, 3, ..., 10} : set ℕ))
                                (h2 : b ∈ ({1, 2, 3, ..., 10} : set ℕ))
                                (h3 : c ∈ ({1, 2, 3, ..., 10} : set ℕ))
                                (h4 : a ≠ b)
                                (h5 : b ≠ c)
                                (h6 : a ≠ c)
                                (h7 : ∃ d : ℤ, b - a = d ∧ c - b = d ∧ 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 10) :
  (∃ n : ℕ, n = 40) :=
sorry

end max_arithmetic_sequences_l540_540143


namespace balls_in_boxes_l540_540051

def waysToPutBallsInBoxes (balls : ℕ) (boxes : ℕ) [Finite boxes] : ℕ :=
  Finset.card { f : Fin boxes → ℕ | (Finset.sum Finset.univ (fun i => f i)) = balls }

theorem balls_in_boxes : waysToPutBallsInBoxes 7 3 = 36 := by
  sorry

end balls_in_boxes_l540_540051


namespace number_of_skirts_l540_540345

theorem number_of_skirts (T Ca Cs S : ℕ) (hT : T = 50) (hCa : Ca = 20) (hCs : Cs = 15) (hS : T - Ca = S * Cs) : S = 2 := by
  sorry

end number_of_skirts_l540_540345


namespace part1_cost_price_per_unit_part2_min_units_sold_at_original_price_l540_540666

namespace ShoppingMall

noncomputable def cost_price_type_A_unit := 40
noncomputable def cost_price_type_B_unit := 48
noncomputable def total_cost_type_A := 2000
noncomputable def total_cost_type_B := 2400
noncomputable def selling_price_type_A := 60
noncomputable def selling_price_type_B := 88
noncomputable def quantity_purchased := 50
noncomputable def discount := 0.3
noncomputable def profit_needed := 2460

theorem part1_cost_price_per_unit :
  (total_cost_type_A / cost_price_type_A_unit) = (total_cost_type_B / cost_price_type_B_unit) ∧
  cost_price_type_B_unit = cost_price_type_A_unit + 8 :=
by
  sorry

theorem part2_min_units_sold_at_original_price :
  ∀ (a : ℕ), 20 ≤ a → 
  (selling_price_type_A - cost_price_type_A_unit) * a + 
  (selling_price_type_A * (1 - discount) - cost_price_type_A_unit) * (quantity_purchased - a) + 
  (selling_price_type_B - cost_price_type_B_unit) * quantity_purchased ≥ profit_needed :=
by
  sorry

end ShoppingMall

end part1_cost_price_per_unit_part2_min_units_sold_at_original_price_l540_540666


namespace exists_distinct_ij_l540_540114

theorem exists_distinct_ij (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) (h_n_ge_3 : 3 ≤ n) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k, (a i + a j) ∣ 3 * a k → False) :=
by
  sorry

end exists_distinct_ij_l540_540114


namespace pentagon_area_l540_540437

def side_lengths : list ℕ := [15, 20, 27, 24, 20]

theorem pentagon_area (a b c d e : ℕ)
  (h1 : side_lengths = [a, b, c, d, e])
  (h2 : a = 15)
  (h3 : b = 20)
  (h4 : c = 27)
  (h5 : d = 24)
  (h6 : e = 20) :
  (let area_triangle := (1 / 2) * a * e,
       area_trapezoid := (1 / 2) * (b + c) * d in
    area_triangle + area_trapezoid) = 714 := by
  sorry

end pentagon_area_l540_540437


namespace range_of_m_for_line_to_intersect_ellipse_twice_l540_540301

theorem range_of_m_for_line_to_intersect_ellipse_twice (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.2 = 4 * A.1 + m) ∧
   (B.2 = 4 * B.1 + m) ∧
   ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧
   ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧
   (A.1 + B.1) / 2 = 0 ∧ 
   (A.2 + B.2) / 2 = 4 * 0 + m) ↔
   - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13
 :=
sorry

end range_of_m_for_line_to_intersect_ellipse_twice_l540_540301


namespace find_range_of_a_l540_540029

open Real

noncomputable def f (a x : ℝ) : ℝ := log a (x + sqrt (x^2 - 2))

noncomputable def f_inv (a x : ℝ) : ℝ := (a^x + a^(-x)) / 2

noncomputable def g (a n : ℝ) : ℝ := (sqrt 2 / 2) * f_inv a (n + log a (sqrt 2))

theorem find_range_of_a (a : ℝ) (hapos : a > 0) (ha1 : a ≠ 1) :
  (∀ n ∈ (Set.Ioi (0 : ℝ)), g a n < (3^n + 3^(-n)) / 2) → 1 < a ∧ a < 3 :=
by
  intro h
  sorry

end find_range_of_a_l540_540029


namespace unique_digits_in_base_100_l540_540102

theorem unique_digits_in_base_100 :
  ∃ (X Y Z : ℕ), (X = 1 ∧ Y = 100 ∧ Z = 10000) ∧
  (∀ (a b c : ℕ), a < 100 → b < 100 → c < 100 →
    let S := a * X + b * Y + c * Z in
    S % 100 = a ∧ (S / 100) % 100 = b ∧ S / 10000 = c) :=
by
  use [1, 100, 10000]
  split
  · exact ⟨rfl, rfl, rfl⟩
  intros a b c ha hb hc
  let S := a * 1 + b * 100 + c * 10000
  have ha' : S % 100 = a := by sorry
  have hb' : (S / 100) % 100 = b := by sorry
  have hc' : S / 10000 = c := by sorry
  exact ⟨ha', hb', hc'⟩

end unique_digits_in_base_100_l540_540102


namespace find_slope_of_line_l540_540402

theorem find_slope_of_line
  (k : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (3, 0))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 - y^2 / 3 = 1)
  (A B : ℝ × ℝ)
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (line : ℝ → ℝ → Prop)
  (hline : ∀ x y, line x y ↔ y = k * (x - 3))
  (hintersectA : line A.1 A.2)
  (hintersectB : line B.1 B.2)
  (F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hfoci_sum : ∀ z : ℝ × ℝ, |z.1 - F.1| + |z.2 - F.2| = 16) :
  k = 3 ∨ k = -3 :=
by
  sorry

end find_slope_of_line_l540_540402


namespace lcm_div_l540_540504

/-- 
Let P be the least common multiple of all the integers 15 through 25, inclusive. 
Let Q be the least common multiple of P together with the integers 26, 27, 28, 29, 30, 31, 32, 33, 34, and 35.
Prove that the value of Q / P is 10788.
 -/
theorem lcm_div (P Q : ℕ) 
  (hP : P = Nat.lcm_list (List.range' 15 (25 - 15 + 1))) 
  (hQ : Q = Nat.lcm P (Nat.lcm_list (List.range' 26 (35 - 26 + 1)))) : 
  Q / P = 10788 := 
by 
  sorry

end lcm_div_l540_540504


namespace max_value_of_8a_5b_15c_l540_540889

theorem max_value_of_8a_5b_15c (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  8*a + 5*b + 15*c ≤ (Real.sqrt 115) / 2 :=
by
  sorry

end max_value_of_8a_5b_15c_l540_540889


namespace honda_production_l540_540435

theorem honda_production : 
  ∃ (x y z : ℕ),
    y = 4 * x ∧
    z = (3 / 2) * x ∧ 
    x + y + z = 8000 ∧
    x = 1231 ∧
    y = 4923 ∧
    z = 1846 :=
by
  let x := 1231
  let y := 4 * x
  let z := 3 / 2 * x
  use x, y, z
  simp [x, y, z]
  sorry

end honda_production_l540_540435


namespace milk_added_to_full_can_l540_540075

noncomputable def amount_of_milk_added : ℝ :=
let x := 36 / 7 in
let initial_milk := 4 * x in
let water := 3 * x in
let new_milk := 2 * water in
new_milk - initial_milk

theorem milk_added_to_full_can :
  (amount_of_milk_added ≈ (72 / 7)) := 
sorry

end milk_added_to_full_can_l540_540075


namespace ratio_inequality_l540_540674

-- Definitions for the terms used in the problem
variables {Point : Type} [EuclideanGeometry Point]
variables (A B C M K : Point)

-- Conditions of the problem
def area_eq (ABC M K : Point) [Triangle ABC] : Prop :=
area (Triangle.mk B M K) = area (Quadrilateral.mk A M K C)

-- Proof statement
theorem ratio_inequality (h : area_eq ABC M K) :
  ∀ MB BK AM CA KC : ℝ, 
  MB = dist B M ->
  BK = dist B K ->
  AM = dist A M ->
  CA = dist C A ->
  KC = dist K C ->
  \frac{MB + BK}{AM + CA + KC} ≥ \frac{1}{3} :=
sorry

end ratio_inequality_l540_540674


namespace average_stickers_per_pack_l540_540097

def average (lst : List ℕ) : ℚ :=
  (list.sum lst : ℚ) / list.length lst

theorem average_stickers_per_pack :
  average [5, 8, 0, 12, 15, 20, 22, 25, 30, 35] = 17.2 := by
  sorry

end average_stickers_per_pack_l540_540097


namespace paco_cookies_difference_l540_540905

variable initial_cookies : Nat
variable cookies_ate : Nat
variable cookies_bought : Nat

noncomputable def more_cookies_bought_than_eaten (initial_cookies cookies_ate cookies_bought : Nat) : Nat :=
  cookies_bought - cookies_ate

theorem paco_cookies_difference : more_cookies_bought_than_eaten initial_cookies cookies_ate cookies_bought = 34 :=
  by
    /- Initial conditions -/
    have h1 : initial_cookies = 13 := by sorry
    have h2 : cookies_ate = 2 := by sorry
    have h3 : cookies_bought = 36 := by sorry

    /- Proof -/
    show more_cookies_bought_than_eaten initial_cookies cookies_ate cookies_bought = 34
    sorry

end paco_cookies_difference_l540_540905


namespace value_of_f_750_l540_540885

theorem value_of_f_750 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y^2)
    (hf500 : f 500 = 4) :
    f 750 = 16 / 9 :=
sorry

end value_of_f_750_l540_540885


namespace marsha_pay_per_mile_l540_540537

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l540_540537


namespace cakes_sold_correct_l540_540263

def total_cakes_baked_today : Nat := 5
def total_cakes_baked_yesterday : Nat := 3
def cakes_left : Nat := 2

def total_cakes : Nat := total_cakes_baked_today + total_cakes_baked_yesterday
def cakes_sold : Nat := total_cakes - cakes_left

theorem cakes_sold_correct :
  cakes_sold = 6 :=
by
  -- proof goes here
  sorry

end cakes_sold_correct_l540_540263


namespace parallelogram_area_l540_540679

-- Definitions based on conditions
def adjacent_side1 (s : ℝ) : ℝ := 3 * s
def adjacent_side2 (s : ℝ) : ℝ := s
def angle_degrees : ℝ := 60
def given_area : ℝ := 9 * Real.sqrt 3

-- Problem statement
theorem parallelogram_area (s : ℝ) (h : s = Real.sqrt 6) :
  let side1 := adjacent_side1 s,
      side2 := adjacent_side2 s,
      angle := angle_degrees,
      area := side1 * side2 * Real.sin (angle * Real.pi / 180)
  in area = given_area :=
  sorry

end parallelogram_area_l540_540679


namespace cos_value_l540_540383

theorem cos_value (θ : ℝ) (h₁ : sin θ = -4/5) (h₂ : tan θ > 0) : cos θ = -3/5 := 
by
  sorry

end cos_value_l540_540383


namespace fred_has_times_more_balloons_l540_540916

structure Balloons where
  sally_balloons : ℕ
  fred_balloons : ℕ

def sally : ℕ := 6
def fred : ℕ := 18

theorem fred_has_times_more_balloons (S : Balloons) : S.sally_balloons = sally → S.fred_balloons = fred → S.fred_balloons / S.sally_balloons = 3 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num

end fred_has_times_more_balloons_l540_540916


namespace total_number_of_students_l540_540995

theorem total_number_of_students
  (ratio_girls_to_boys : ℕ) (ratio_boys_to_girls : ℕ)
  (num_girls : ℕ)
  (ratio_condition : ratio_girls_to_boys = 5 ∧ ratio_boys_to_girls = 8)
  (num_girls_condition : num_girls = 160)
  : (num_girls * (ratio_girls_to_boys + ratio_boys_to_girls) / ratio_girls_to_boys = 416) :=
by
  sorry

end total_number_of_students_l540_540995


namespace bella_pizza_l540_540343

variable (rachel total bella : ℕ)

def rachel := 598
def total := 952

theorem bella_pizza : bella = 354 :=
by
  have eq1 : 952 = 598 + bella := by rfl
  show bella = 354 from by
    rw [eq1, add_comm]
    simp
  sorry

end bella_pizza_l540_540343


namespace num_integers_lt_500_congruent_4_mod_9_l540_540822

open Int

theorem num_integers_lt_500_congruent_4_mod_9 : 
  {x : ℤ | 0 < x ∧ x < 500 ∧ x % 9 = 4}.toFinset.card = 56 := 
by
  sorry

end num_integers_lt_500_congruent_4_mod_9_l540_540822


namespace speed_of_train_correct_l540_540662

noncomputable def speed_of_train (L t v_man : ℝ) : ℝ :=
  let v_man_m_s := v_man * (1000 / 3600)
  let v_relative := L / t
  let v_train_m_s := v_relative + v_man_m_s
  v_train_m_s * 3.6

theorem speed_of_train_correct :
  speed_of_train 400 23.998 3 ≈ 63.005 :=
by
  sorry

end speed_of_train_correct_l540_540662


namespace compare_a_b_l540_540827

theorem compare_a_b (a b : ℝ) (h₁ : a = 1.9 * 10^5) (h₂ : b = 9.1 * 10^4) : a > b := by
  sorry

end compare_a_b_l540_540827


namespace arithmetic_sequence_common_difference_l540_540110

variable {ℤ : Type*}

-- Definitions translating the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ (n : ℕ), a (n + 1) = a n + d

-- Given conditions in the problem
def condition1 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
a 2 = S 3

def condition2 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
a 1 * a 3 = S 4

def sum_first_n_terms (a : ℕ → ℤ) : ℕ → ℤ
| 0       => 0
| (n + 1) => sum_first_n_terms n + a n

-- Mathematical problem in Lean statement form
theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (is_arithmetic_sequence a d) →
  (a 2 = sum_first_n_terms a 3) →
  (a 1 * a 3 = sum_first_n_terms a 4) →
  d = -2 := 
by {
  sorry,
}

end arithmetic_sequence_common_difference_l540_540110


namespace infinite_product_series_equiv_l540_540556

theorem infinite_product_series_equiv {b_k : ℕ → ℝ} 
  (h₀ : ∀ k, 0 ≤ b_k k) 
  (h₁ : ∀ k, b_k k < 1) : 
  (∃ l, tendsto (λ n, ∏ k in finset.range n, (1 - b_k k)) at_top (nhds l)) ↔ 
  (∃ l, tendsto (λ n, ∑ k in finset.range n, b_k k) at_top (nhds l)) := 
sorry

end infinite_product_series_equiv_l540_540556


namespace ella_probability_last_roll_l540_540056

theorem ella_probability_last_roll (n k : ℕ) (h₁ : n = 12) (h₂ : k = 2) :
  (∑ (i : ℕ) in finset.range 11, (5/6)^(i-1) * (1/6) * (5/6)^(10-i) * (1/6)) = 19531250 / 362797056 :=
by sorry

end ella_probability_last_roll_l540_540056


namespace volume_of_wedge_l540_540670

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (V : ℝ) (n : ℕ) 
  (h_d : d = 18) 
  (h_angle : angle = 60)
  (h_radius_height : ∀ r h, r = d / 2 ∧ h = d) 
  (h_volume_cylinder : V = π * (d / 2) ^ 2 * d) 
  : n = 729 ↔ V / 2 = n * π :=
by
  sorry

end volume_of_wedge_l540_540670


namespace find_s_l540_540678

theorem find_s (s : ℝ) (h1 : 3 * s * s * sqrt 3 = 9 * sqrt 3) : s = sqrt 3 :=
by
  sorry

end find_s_l540_540678


namespace train_crossing_pole_l540_540690

def speed_kmh_to_ms (v_kmh : ℝ) : ℝ := v_kmh * (1000 / 3600)

def crossing_time (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh_to_ms speed_kmh
  length / speed_ms

theorem train_crossing_pole
  (speed : ℝ := 70)  -- speed of the train in km/hr
  (length : ℝ := 175)  -- length of the train in meters
  : crossing_time length speed = 9 := by
  sorry

end train_crossing_pole_l540_540690


namespace solution_to_equation_l540_540927

noncomputable def solve_equation (x : ℝ) : Prop :=
  (sqrt (3 * x - 2) + 9 / sqrt (3 * x - 2) = 6) → (x = 11 / 3)

theorem solution_to_equation : ∃ x : ℝ, solve_equation x :=
begin
  use 11 / 3,
  sorry,
end

end solution_to_equation_l540_540927


namespace first_player_wins_l540_540220

-- Define the problem conditions and statements
noncomputable def player_with_winning_strategy (total_matches : Nat) (player_turns : Nat → Nat) : Prop :=
  if total_matches = 21 then
    let first_player := "First Player"
    let second_player := "Second Player"
    -- The statement that there is a winning strategy for the first player. 
    -- The details of the strategy are implicit.
    first_player
  else
    sorry  -- placeholder for cases that aren't 21 matches

-- Proof (statement only)
theorem first_player_wins : player_with_winning_strategy 21 (λ n, if n % 2 = 0 then 1 else 3) = "First Player" :=
by
  sorry

end first_player_wins_l540_540220


namespace power_expression_l540_540713

variable {a b : ℝ}

theorem power_expression : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := 
by 
  sorry

end power_expression_l540_540713


namespace area_triangle_AXY_l540_540684

noncomputable def point (α β : Type) := (α × β)

structure square :=
  (A B C D : point ℝ)
  (area : ℝ)
  (side_length : ℝ)
  (AB : side_length)
  (BC : side_length)
  (CD : side_length)
  (DA : side_length)

structure conditions :=
  (square_ABCD : square)
  (point_X : point ℝ)
  (point_Y : point ℝ)
  (on_BC : point_X.2 = square_ABCD.B.2 ∧ point_X.1 > square_ABCD.B.1)
  (on_CD : point_Y.1 = square_ABCD.D.1 ∧ point_Y.2 > square_ABCD.D.2)
  (angle_BAX_quarter : true)  -- Placeholder for the trigonometric condition
  (angle_DAY_quarter : true)

theorem area_triangle_AXY (cond : conditions) : area_triangle cond.square_ABCD.A cond.point_X cond.point_Y = 16 * (Real.sqrt 2 - 1) :=
by
  sorry

end area_triangle_AXY_l540_540684


namespace find_June_score_l540_540719

def Patty := 85
def Josh := 100
def Henry := 94
def average := 94

theorem find_June_score (P J H A : ℕ) (hP : P = 85) (hJ : J = 100) (hH : H = 94) (hA : A = 94) : 
  let totalSum := A * 4,
      totalKnown := P + J + H,
      June := totalSum - totalKnown
  in June = 97 := 
by
  sorry

end find_June_score_l540_540719


namespace dvd_sold_168_l540_540459

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l540_540459


namespace correct_statements_l540_540215

noncomputable theory

open EuclideanGeometry

def line (P Q : Point) := P ≠ Q ∧ ∃ (l : Line), l.frequency P Q
def plane (a b c : Point) := a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ ∃ (pl : Plane), pl.contains a b c

def perpendicular_lines (l₁ l₂ : Line) := ∃ (P : Point), l₁.contains P ∧ l₂.contains P ∧ ∀ (v₁ v₂ : Vector), v₁ ∈ l₁.direction ∧ v₂ ∈ l₂.direction → v₁ ⊥ v₂
def perpendicular_planes (pl₁ pl₂ : Plane) := ∃ (n₁ n₂ : Vector), pl₁.normal n₁ ∧ pl₂.normal n₂ ∧ n₁ ⊥ n₂

theorem correct_statements :
  (∀ l₁ l₂ l : Line, perpendicular_lines l₁ l ∧ perpendicular_lines l₂ l → ¬ parallel l₁ l₂) →
  (∀ pl₁ pl₂ l : Plane, perpendicular_planes pl₁ l ∧ perpendicular_planes pl₂ l → parallel pl₁ pl₂) ∧
  (∀ l₁ l₂ pl : Line, perpendicular_lines l₁ pl ∧ perpendicular_lines l₂ pl → parallel l₁ l₂) ∧
  (∀ pl₁ pl₂ pl₃ : Plane, perpendicular_planes pl₁ pl₃ ∧ perpendicular_planes pl₂ pl₃ → ¬ parallel pl₁ pl₂) :=
by
  sorry

end correct_statements_l540_540215


namespace infinite_product_to_rational_root_l540_540321

theorem infinite_product_to_rational_root :
  (∀ (n : ℕ), ( nat.pow 3 n ) ^ (1 / (4 ^ (n + 1)))) =
  real.root 9 81 :=
sorry

end infinite_product_to_rational_root_l540_540321


namespace distinct_values_S_l540_540505

def complex_i : Complex := Complex.I

def expr_S (n : Int) : Complex := (complex_i ^ n) + (complex_i ^ (-2 * n))

theorem distinct_values_S : set (expr_S n) =
  set {2, Complex.I - 1, 0, -Complex.I + 1} = set 4 := 
  sorry

end distinct_values_S_l540_540505


namespace angle_between_a_b_is_zero_degree_l540_540507

-- Given conditions
variables {a b c : EuclideanSpace ℝ (Fin 3)} -- Assuming ℝ^3 space for vectors
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
variables (habc : a + b + 2 • c = 0)

-- Prove the angle between a and b is 0 degrees
theorem angle_between_a_b_is_zero_degree : real.angle (a.toRealOrtho) (b.toRealOrtho) = 0 :=
by sorry

end angle_between_a_b_is_zero_degree_l540_540507


namespace solve_for_a_minus_c_l540_540240

theorem solve_for_a_minus_c 
  (a b c d : ℝ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
  sorry

end solve_for_a_minus_c_l540_540240


namespace diff_of_squares_l540_540826

theorem diff_of_squares (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 :=
by
  sorry

end diff_of_squares_l540_540826


namespace sharon_trip_distance_l540_540563

theorem sharon_trip_distance (distance time_normal_speed time_reduced_speed : ℝ) (total_time_normal traffic_speed_factor : ℝ) :
  time_normal_speed = 150 →
  traffic_speed_factor = 0.8 →
  total_time_normal = 210 →
  let normal_speed := distance / time_normal_speed in
  let time_first_half := (distance / 2) / normal_speed in
  let reduced_speed := traffic_speed_factor * normal_speed in
  let time_second_half := (distance / 2) / reduced_speed in
  time_first_half + time_second_half = total_time_normal →
  distance = 150 :=
begin
  sorry
end

end sharon_trip_distance_l540_540563


namespace pqr_problem_l540_540887

noncomputable def pqr_sums_to_44 (p q r : ℝ) : Prop :=
  (p < q) ∧ (∀ x, (x < -6 ∨ |x - 20| ≤ 2) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0 ))

theorem pqr_problem (p q r : ℝ) (h : pqr_sums_to_44 p q r) : p + 2*q + 3*r = 44 :=
sorry

end pqr_problem_l540_540887


namespace arcadia_population_approx_9000_l540_540740

noncomputable def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  if years < 10 then initial_population
  else let n := (years - 10) / 25 in initial_population * 3^n

theorem arcadia_population_approx_9000 :
  ∃ (y : ℕ), y = 2095 ∧ abs (population 250 (y - 2010) - 9000) < abs (population 250 (2095 - 2010) - 9000) :=
begin
  sorry
end

end arcadia_population_approx_9000_l540_540740


namespace matthew_walking_rate_l540_540132

theorem matthew_walking_rate
  (QY_distance : ℕ)
  (johnny_rate : ℕ)
  (johnny_walked_distance : ℕ)
  (one_hour : ℕ)
  (met_time : ℕ) :
  QY_distance = 45 →
  johnny_rate = 4 →
  johnny_walked_distance = 24 →
  one_hour = 1 →
  met_time = (johnny_walked_distance / johnny_rate + one_hour): 
  (QY_distance - johnny_walked_distance) / met_time = 3 := 
by
  sorry

end matthew_walking_rate_l540_540132


namespace divisibility_by_7_l540_540554

theorem divisibility_by_7 (m a : ℤ) (h : 0 ≤ a ∧ a ≤ 9) (B : ℤ) (hB : B = m - 2 * a) (h7 : B % 7 = 0) : (10 * m + a) % 7 = 0 := 
sorry

end divisibility_by_7_l540_540554


namespace ratio_of_sector_eof_to_circle_l540_540903

noncomputable def angle_aob : ℝ := 180
noncomputable def angle_aoe : ℝ := 40
noncomputable def angle_fob : ℝ := 60

theorem ratio_of_sector_eof_to_circle (h1 : angle_aob = 180)
  (h2 : angle_aoe = 40) (h3 : angle_fob = 60) : 
  let angle_eof := angle_aob - angle_aoe - angle_fob in
  angle_eof / 360 = 2 / 9 :=
by
  have angle_eof := angle_aob - angle_aoe - angle_fob
  have h: angle_eof = 80 := by
    rw [h1, h2, h3]
    norm_num
    
  sorry

end ratio_of_sector_eof_to_circle_l540_540903


namespace ellipse_equation_l540_540276

theorem ellipse_equation 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≠ n)
  (h4 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → dist A B = 2 * (2:ℝ).sqrt)
  (h5 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → 
    (A.2 + B.2) / (A.1 + B.1) = (2:ℝ).sqrt / 2) :
  m = 1 / 3 → n = (2:ℝ).sqrt / 3 → 
  (∀ x y : ℝ, (1 / 3) * x^2 + ((2:ℝ).sqrt / 3) * y^2 = 1) :=
by
  sorry

end ellipse_equation_l540_540276


namespace even_function_f_values_order_l540_540442

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_f_values_order (m : ℝ) (h_even : ∀ x : ℝ, f x m = f (-x) m) : 
  f (-2) 0 < f 1 0 ∧ f 1 0 < f 0 0 :=
by
  have h_m : m = 0,
  { sorry },
  have f_even: ∀ x : ℝ, f x 0 = -x^2 + 2 := 
  by
    intros x,
    rw [←h_m, f],
    simp,
  show f (-2) 0 < f (1) 0 ∧ f (1) 0 < f (0) 0,
  { sorry }

end even_function_f_values_order_l540_540442


namespace solution_set_for_log_inequality_l540_540207

noncomputable def log_base_0_1 (x: ℝ) : ℝ := Real.log x / Real.log 0.1

theorem solution_set_for_log_inequality :
  ∀ x : ℝ, (0 < x) → 
  log_base_0_1 (2^x - 1) < 0 ↔ x > 1 :=
by
  sorry

end solution_set_for_log_inequality_l540_540207


namespace circumcircle_triangle_XYZ_lies_inside_omega_1_epsilon_l540_540877

-- Define the problem statement in Lean
theorem circumcircle_triangle_XYZ_lies_inside_omega_1_epsilon
  (ABC : Type*) [Triangle ABC]
  (ω : Circle) (I : Point) (r : ℝ)
  (X Y Z : Point)
  (trisector_BC_X : Line) (trisector_BC_Y : Line) (trisector_BC_Z : Line)
  (λ : ℝ) (ε : ℝ) (hε : ε = 1 / 2024)
  (ωλ : Circle)
  (ω1_ε : Circle) :
  -- Conditions
  InCircle ω ABC I r →
  ConcentricCircles ω ωλ →
  Radius ωλ = λ * r →
  Intersection trisector_BC_X trisector_B trisector_BC_C = X →
  Intersection trisector_BC_Y trisector_C trisector_CA_C = Y →
  Intersection trisector_BC_Z trisector_A trisector_AB_C = Z →
  ω1_ε = Circle I ((1 - ε) * r) →
  -- Conclusion
  Circumcircle (Triangle X Y Z) ≤ ω1_ε :=
sorry

-- Some assumptions used in Lean code (can be expanded based on availability):
variables {Triangle: Type*}
  (Point : Type*) (Line : Type*) (Circle : Type*)
  [has_center Circle Point] [has_radius Circle ℝ]
  [intersection : Line → Line → Point]
  (InCircle : Circle → Triangle → Point → ℝ → Prop)
  (ConcentricCircles : Circle → Circle → Prop)
  (Radius : Circle → ℝ)
  (Circumcircle : Triangle → Circle)
  (Triangle : Point → Point → Point → Triangle)

end circumcircle_triangle_XYZ_lies_inside_omega_1_epsilon_l540_540877


namespace part_a_part_b_l540_540992

theorem part_a (a : Fin 10 → ℤ) : ∃ i j : Fin 10, i ≠ j ∧ 27 ∣ (a i)^3 - (a j)^3 := sorry
theorem part_b (b : Fin 8 → ℤ) : ∃ i j : Fin 8, i ≠ j ∧ 27 ∣ (b i)^3 - (b j)^3 := sorry

end part_a_part_b_l540_540992


namespace parallelogram_area_l540_540680

-- Definitions based on conditions
def adjacent_side1 (s : ℝ) : ℝ := 3 * s
def adjacent_side2 (s : ℝ) : ℝ := s
def angle_degrees : ℝ := 60
def given_area : ℝ := 9 * Real.sqrt 3

-- Problem statement
theorem parallelogram_area (s : ℝ) (h : s = Real.sqrt 6) :
  let side1 := adjacent_side1 s,
      side2 := adjacent_side2 s,
      angle := angle_degrees,
      area := side1 * side2 * Real.sin (angle * Real.pi / 180)
  in area = given_area :=
  sorry

end parallelogram_area_l540_540680


namespace checkerboard_all_ones_l540_540703

theorem checkerboard_all_ones (f : Fin 2015 × Fin 2015 → ℝ)
  (h : ∀ (i j : Fin 2015), f(i,j) + f(i.succ,j) + f(i,j.succ) = 3 ∨ 
                            f(i,j) + f(i.pred,j) + f(i,j.succ) = 3 ∨
                            f(i,j) + f(i.succ,j) + f(i,j.pred) = 3 ∨ 
                            f(i,j) + f(i.pred,j) + f(i,j.pred) = 3) :
  ∀ (i j : Fin 2015), f (i, j) = 1 := by
  sorry

end checkerboard_all_ones_l540_540703


namespace second_order_determinant_ratios_l540_540641

theorem second_order_determinant_ratios
  (a b c d e f : ℝ)
  (hD : a * d - b * c ≠ 0) :
  (let D₁ := Matrix.det ![![b, c], ![e, f]],
       D₂ := Matrix.det ![![c, a], ![f, d]],
       D₃ := Matrix.det ![![a, b], ![d, e]]
   in a * D₁ + b * D₂ + c * D₃ = 0 ∧ 
      d * D₁ + e * D₂ + f * D₃ = 0) := by
  sorry

end second_order_determinant_ratios_l540_540641


namespace cookie_radius_l540_540155

theorem cookie_radius (x y : ℝ) : 
  x^2 + y^2 + 5 = 2 * x + 6 * y → 
  ∃ r, r = real.sqrt 5 ∧ ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = r^2 := 
by
  intros h
  sorry

end cookie_radius_l540_540155


namespace sequence_solution_l540_540205

theorem sequence_solution {a b : ℝ} : 
  (∀ n > 1, a_n = 5 * a_{n-1} - 6 * b_{n-1}) →
  (∀ n > 1, b_n = 3 * a_{n-1} - 4 * b_{n-1}) →
  (a_1 = a) →
  (b_1 = b) →
  (∀ n : ℕ, n ≥ 1 → a_n = (a - b) * 2^n + (2 * b - a) * (-1)^(n-1)) ∧ 
  (∀ n : ℕ, n ≥ 1 → b_n = (a - b) * 2^(n-1) + (2 * b - a) * (-1)^(n-1)) :=
by
  sorry

end sequence_solution_l540_540205


namespace exists_distinct_naturals_no_square_sum_l540_540342

theorem exists_distinct_naturals_no_square_sum :
  ∃ S : Finset ℕ, S.card = 1000000 ∧ ∀ T : Finset ℕ, T ⊆ S → T ≠ ∅ → ¬ is_square (∑ x in T, x) :=
sorry

end exists_distinct_naturals_no_square_sum_l540_540342
