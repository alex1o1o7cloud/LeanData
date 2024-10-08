import Mathlib

namespace cylinder_ratio_l137_137312

theorem cylinder_ratio (m r : ℝ) (h1 : m + 2 * r = Real.sqrt (m^2 + (r * Real.pi)^2)) :
  m / (2 * r) = (Real.pi^2 - 4) / 8 := by
  sorry

end cylinder_ratio_l137_137312


namespace smallest_term_at_n_is_4_or_5_l137_137893

def a_n (n : ℕ) : ℝ :=
  n^2 - 9 * n - 100

theorem smallest_term_at_n_is_4_or_5 :
  ∃ n, n = 4 ∨ n = 5 ∧ a_n n = min (a_n 4) (a_n 5) :=
by
  sorry

end smallest_term_at_n_is_4_or_5_l137_137893


namespace probability_at_least_one_correct_l137_137124

theorem probability_at_least_one_correct :
  let p_a := 12 / 20
  let p_b := 8 / 20
  let prob_neither := (1 - p_a) * (1 - p_b)
  let prob_at_least_one := 1 - prob_neither
  prob_at_least_one = 19 / 25 := by
  sorry

end probability_at_least_one_correct_l137_137124


namespace sum_of_primes_eq_100_l137_137399

theorem sum_of_primes_eq_100 : 
  ∃ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S → Nat.Prime x) ∧ S.sum id = 100 ∧ S.card = 9 :=
by
  sorry

end sum_of_primes_eq_100_l137_137399


namespace sqrt_17_irrational_l137_137080

theorem sqrt_17_irrational : ¬ ∃ (q : ℚ), q * q = 17 := sorry

end sqrt_17_irrational_l137_137080


namespace angle_sum_proof_l137_137882

theorem angle_sum_proof (A B C x y : ℝ) 
  (hA : A = 35) 
  (hB : B = 65) 
  (hC : C = 40) 
  (hx : x = 130 - C)
  (hy : y = 90 - A) :
  x + y = 140 := by
  sorry

end angle_sum_proof_l137_137882


namespace arithmetic_sequence_sum_l137_137434

variable {a : ℕ → ℝ}

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Definition of the fourth term condition
def a4_condition (a : ℕ → ℝ) : Prop :=
  a 4 = 2 - a 3

-- Definition of the sum of the first 6 terms
def sum_first_six_terms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

-- Proof statement
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  a4_condition a →
  sum_first_six_terms a = 6 :=
by
  sorry

end arithmetic_sequence_sum_l137_137434


namespace sheela_monthly_income_l137_137100

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (income : ℝ) 
  (h1 : deposit = 2500) (h2 : percentage = 0.25) (h3 : deposit = percentage * income) :
  income = 10000 := 
by
  -- proof steps would go here
  sorry

end sheela_monthly_income_l137_137100


namespace find_x_l137_137166

noncomputable def vector_a : ℝ × ℝ × ℝ := (-3, 2, 5)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-3) * 1 + 2 * x + 5 * (-1) = 2) : x = 5 :=
by 
  sorry

end find_x_l137_137166


namespace largest_n_with_triangle_property_l137_137649

/-- Triangle property: For any subset {a, b, c} with a ≤ b ≤ c, a + b > c -/
def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≤ b → b ≤ c → a + b > c

/-- Definition of the set {3, 4, ..., n} -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1) \ Finset.range 3

/-- The problem statement: The largest possible value of n where all eleven-element
 subsets of {3, 4, ..., n} have the triangle property -/
theorem largest_n_with_triangle_property : ∃ n, (∀ s ⊆ consecutive_set n, s.card = 11 → triangle_property s) ∧ n = 321 := sorry

end largest_n_with_triangle_property_l137_137649


namespace profit_percentage_l137_137118

-- Define the selling price and the cost price
def SP : ℝ := 100
def CP : ℝ := 86.95652173913044

-- State the theorem for profit percentage
theorem profit_percentage :
  ((SP - CP) / CP) * 100 = 15 :=
by
  sorry

end profit_percentage_l137_137118


namespace union_eq_universal_set_l137_137536

-- Define the sets U, M, and N
def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 6}

-- The theorem stating the desired equality
theorem union_eq_universal_set : M ∪ N = U := 
sorry

end union_eq_universal_set_l137_137536


namespace book_selection_l137_137320

theorem book_selection (total_books novels : ℕ) (choose_books : ℕ)
  (h_total : total_books = 15)
  (h_novels : novels = 5)
  (h_choose : choose_books = 3) :
  (Nat.choose 15 3 - Nat.choose 10 3) = 335 :=
by
  sorry

end book_selection_l137_137320


namespace three_gorges_scientific_notation_l137_137116

theorem three_gorges_scientific_notation :
  ∃a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ (798.5 * 10^1 = a * 10^n) ∧ a = 7.985 ∧ n = 2 :=
by
  sorry

end three_gorges_scientific_notation_l137_137116


namespace area_of_rectangle_l137_137499

theorem area_of_rectangle (S R L B A : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S^2 = 1600)
  (h4 : B = 10)
  (h5 : A = L * B) : 
  A = 160 := 
sorry

end area_of_rectangle_l137_137499


namespace angle_degree_measure_l137_137342

theorem angle_degree_measure (x : ℝ) (h1 : (x + (90 - x) = 90)) (h2 : (x = 3 * (90 - x))) : x = 67.5 := by
  sorry

end angle_degree_measure_l137_137342


namespace f_of_1_l137_137282

theorem f_of_1 (f : ℕ+ → ℕ+) (h_mono : ∀ {a b : ℕ+}, a < b → f a < f b)
  (h_fn_prop : ∀ n : ℕ+, f (f n) = 3 * n) : f 1 = 2 :=
sorry

end f_of_1_l137_137282


namespace solution_set_of_inequality_l137_137065

theorem solution_set_of_inequality (x : ℝ) : |2 * x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 :=
by 
  sorry

end solution_set_of_inequality_l137_137065


namespace angle_bisectors_geq_nine_times_inradius_l137_137292

theorem angle_bisectors_geq_nine_times_inradius 
  (r : ℝ) (f_a f_b f_c : ℝ) 
  (h_triangle : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ r = (1 / 2) * (a + b + c) * r ∧ 
      f_a ≥ (2 * a * b / (a + b) + 2 * a * c / (a + c)) / 2 ∧ 
      f_b ≥ (2 * b * a / (b + a) + 2 * b * c / (b + c)) / 2 ∧ 
      f_c ≥ (2 * c * a / (c + a) + 2 * c * b / (c + b)) / 2)
  : f_a + f_b + f_c ≥ 9 * r :=
sorry

end angle_bisectors_geq_nine_times_inradius_l137_137292


namespace find_k_l137_137731

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (-5, 1)

-- Define the condition for parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define the statement to prove
theorem find_k : parallel (a.1 + k * b.1, a.2 + k * b.2) c → k = 1/2 :=
by
  sorry

end find_k_l137_137731


namespace find_intersection_l137_137716

def intersection_point (x y : ℚ) : Prop :=
  3 * x + 4 * y = 12 ∧ 7 * x - 2 * y = 14

theorem find_intersection :
  intersection_point (40 / 17) (21 / 17) :=
by
  sorry

end find_intersection_l137_137716


namespace mul_neg_x_squared_cubed_l137_137950

theorem mul_neg_x_squared_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 :=
sorry

end mul_neg_x_squared_cubed_l137_137950


namespace price_of_shares_l137_137012

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end price_of_shares_l137_137012


namespace smallest_n_property_l137_137992

noncomputable def smallest_n : ℕ := 13

theorem smallest_n_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (x * y * z ∣ (x + y + z) ^ smallest_n) :=
by
  intros x y z hx hy hz hxy hyz hzx
  use smallest_n
  sorry

end smallest_n_property_l137_137992


namespace tim_interest_rate_l137_137299

theorem tim_interest_rate
  (r : ℝ)
  (h1 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000))
  (h2 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000) + 23.5) : 
  r = 0.1 :=
by
  sorry

end tim_interest_rate_l137_137299


namespace loaf_bread_cost_correct_l137_137700

-- Given conditions
def total : ℕ := 32
def candy_bar : ℕ := 2
def final_remaining : ℕ := 18

-- Intermediate calculations as definitions
def remaining_after_candy_bar : ℕ := total - candy_bar
def turkey_cost : ℕ := remaining_after_candy_bar / 3
def remaining_after_turkey : ℕ := remaining_after_candy_bar - turkey_cost
def loaf_bread_cost : ℕ := remaining_after_turkey - final_remaining

-- Theorem stating the problem question and expected answer
theorem loaf_bread_cost_correct : loaf_bread_cost = 2 :=
sorry

end loaf_bread_cost_correct_l137_137700


namespace num_solutions_eq_40_l137_137424

theorem num_solutions_eq_40 : 
  ∀ (n : ℕ), 
  (∃ seq : ℕ → ℕ, seq 1 = 4 ∧ (∀ k : ℕ, 1 ≤ k → seq (k + 1) = seq k + 4) ∧ seq 10 = 40) :=
by
  sorry

end num_solutions_eq_40_l137_137424


namespace find_number_l137_137165

theorem find_number (x : ℝ) (h : 0.6667 * x + 0.75 = 1.6667) : x = 1.375 :=
sorry

end find_number_l137_137165


namespace min_value_function_l137_137627

theorem min_value_function (x y : ℝ) (h1 : -2 < x ∧ x < 2) (h2 : -2 < y ∧ y < 2) (h3 : x * y = -1) :
  ∃ u : ℝ, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) ∧ u = 12 / 5 :=
by
  sorry

end min_value_function_l137_137627


namespace painting_problem_l137_137541

theorem painting_problem
    (H_rate : ℝ := 1 / 60)
    (T_rate : ℝ := 1 / 90)
    (combined_rate : ℝ := H_rate + T_rate)
    (time_worked : ℝ := 15)
    (wall_painted : ℝ := time_worked * combined_rate):
  wall_painted = 5 / 12 := 
by
  sorry

end painting_problem_l137_137541


namespace angle_A_is_60_degrees_value_of_b_plus_c_l137_137189

noncomputable def triangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  let area := (3 * Real.sqrt 3) / 2
  c + 2 * a * Real.cos C = 2 * b ∧
  1/2 * b * c * Real.sin A = area 

theorem angle_A_is_60_degrees (A B C : ℝ) (a b c : ℝ) :
  triangleABC A B C a b c →
  Real.cos A = 1 / 2 → 
  A = 60 :=
by
  intros h1 h2 
  sorry

theorem value_of_b_plus_c (A B C : ℝ) (b c : ℝ) :
  triangleABC A B C (Real.sqrt 7) b c →
  b * c = 6 →
  (b + c) = 5 :=
by 
  intros h1 h2 
  sorry

end angle_A_is_60_degrees_value_of_b_plus_c_l137_137189


namespace quadratic_real_solutions_l137_137653

theorem quadratic_real_solutions (x y : ℝ) :
  (∃ z : ℝ, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end quadratic_real_solutions_l137_137653


namespace henry_income_percent_increase_l137_137421

theorem henry_income_percent_increase :
  let original_income : ℝ := 120
  let new_income : ℝ := 180
  let increase := new_income - original_income
  let percent_increase := (increase / original_income) * 100
  percent_increase = 50 :=
by
  sorry

end henry_income_percent_increase_l137_137421


namespace imaginary_part_of_i_mul_root_l137_137017

theorem imaginary_part_of_i_mul_root
  (z : ℂ) (hz : z^2 - 4 * z + 5 = 0) : (i * z).im = 2 := 
sorry

end imaginary_part_of_i_mul_root_l137_137017


namespace total_cost_correct_l137_137694

-- Define the costs for each repair
def engine_labor_cost := 75 * 16
def engine_part_cost := 1200
def brake_labor_cost := 85 * 10
def brake_part_cost := 800
def tire_labor_cost := 50 * 4
def tire_part_cost := 600

-- Calculate the total costs
def engine_total_cost := engine_labor_cost + engine_part_cost
def brake_total_cost := brake_labor_cost + brake_part_cost
def tire_total_cost := tire_labor_cost + tire_part_cost

-- Calculate the total combined cost
def total_combined_cost := engine_total_cost + brake_total_cost + tire_total_cost

-- The theorem to prove
theorem total_cost_correct : total_combined_cost = 4850 := by
  sorry

end total_cost_correct_l137_137694


namespace trigonometric_identity_l137_137749

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := 
by
  -- proof steps are omitted, using sorry to skip the proof.
  sorry

end trigonometric_identity_l137_137749


namespace stipulated_percentage_l137_137726

theorem stipulated_percentage
  (A B C : ℝ)
  (P : ℝ)
  (hA : A = 20000)
  (h_range : B - C = 10000)
  (hB : B = A + (P / 100) * A)
  (hC : C = A - (P / 100) * A) :
  P = 25 :=
sorry

end stipulated_percentage_l137_137726


namespace convert_angle_l137_137047

theorem convert_angle (α : ℝ) (k : ℤ) :
  -1485 * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π ∧ k = -10 ∧ α = 7 * π / 4 :=
by
  sorry

end convert_angle_l137_137047


namespace trajectory_is_parabola_l137_137372

theorem trajectory_is_parabola
  (P : ℝ × ℝ) : 
  (dist P (0, P.2 + 1) < dist P (0, 2)) -> 
  (P.1^2 = 8 * (P.2 + 2)) :=
by
  sorry

end trajectory_is_parabola_l137_137372


namespace no_four_consecutive_perf_square_l137_137705

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l137_137705


namespace incorrect_option_c_l137_137043

theorem incorrect_option_c (R : ℝ) : 
  let cylinder_lateral_area := 4 * π * R^2
  let sphere_surface_area := 4 * π * R^2
  cylinder_lateral_area = sphere_surface_area :=
  sorry

end incorrect_option_c_l137_137043


namespace sum_of_possible_B_is_zero_l137_137204

theorem sum_of_possible_B_is_zero :
  ∀ B : ℕ, B < 10 → (∃ k : ℤ, 7 * k = 500 + 10 * B + 3) -> B = 0 := sorry

end sum_of_possible_B_is_zero_l137_137204


namespace sum_of_interior_angles_l137_137787

theorem sum_of_interior_angles (n : ℕ) : 
  (∀ θ, θ = 40 ∧ (n = 360 / θ)) → (n - 2) * 180 = 1260 :=
by
  sorry

end sum_of_interior_angles_l137_137787


namespace angle_degrees_l137_137440

-- Define the conditions
def sides_parallel (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = θ₂ ∨ (θ₁ + θ₂ = 180)

def angle_relation (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20

-- Statement of the problem
theorem angle_degrees (θ₁ θ₂ : ℝ) (h_parallel : sides_parallel θ₁ θ₂) (h_relation : angle_relation θ₁ θ₂) :
  (θ₁ = 10 ∧ θ₂ = 10) ∨ (θ₁ = 50 ∧ θ₂ = 130) ∨ (θ₁ = 130 ∧ θ₂ = 50) ∨ θ₁ + θ₂ = 180 ∧ (θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20) :=
by sorry

end angle_degrees_l137_137440


namespace passing_marks_l137_137730

variable (T P : ℝ)

-- condition 1: 0.30T = P - 30
def condition1 : Prop := 0.30 * T = P - 30

-- condition 2: 0.45T = P + 15
def condition2 : Prop := 0.45 * T = P + 15

-- Proof Statement: P = 120 (passing marks)
theorem passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 120 := 
  sorry

end passing_marks_l137_137730


namespace largest_divisor_n4_minus_5n2_plus_6_l137_137011

theorem largest_divisor_n4_minus_5n2_plus_6 :
  ∀ (n : ℤ), (n^4 - 5 * n^2 + 6) % 1 = 0 :=
by
  sorry

end largest_divisor_n4_minus_5n2_plus_6_l137_137011


namespace intersection_of_A_and_B_l137_137260

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := { x : ℝ | -1 < x ∧ x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

end intersection_of_A_and_B_l137_137260


namespace number_of_sides_l137_137093

theorem number_of_sides (n : ℕ) : 
  let a_1 := 6 
  let d := 5
  let a_n := a_1 + (n - 1) * d
  a_n = 5 * n + 1 := 
by
  sorry

end number_of_sides_l137_137093


namespace calc_h_one_l137_137894

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 6
noncomputable def g (x : ℝ) : ℝ := Real.exp (f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- the final theorem that we are proving
theorem calc_h_one : h 1 = 3 * Real.exp 26 - 14 * Real.exp 13 + 21 := by
  sorry

end calc_h_one_l137_137894


namespace variance_of_data_set_l137_137325

theorem variance_of_data_set (m : ℝ) (h_mean : (6 + 7 + 8 + 9 + m) / 5 = 8) :
    (1/5) * ((6-8)^2 + (7-8)^2 + (8-8)^2 + (9-8)^2 + (m-8)^2) = 2 := 
sorry

end variance_of_data_set_l137_137325


namespace third_year_increment_l137_137411

-- Define the conditions
def total_payments : ℕ := 96
def first_year_cost : ℕ := 20
def second_year_cost : ℕ := first_year_cost + 2
def third_year_cost (x : ℕ) : ℕ := second_year_cost + x
def fourth_year_cost (x : ℕ) : ℕ := third_year_cost x + 4

-- The main proof statement
theorem third_year_increment (x : ℕ) 
  (H : first_year_cost + second_year_cost + third_year_cost x + fourth_year_cost x = total_payments) :
  x = 2 :=
sorry

end third_year_increment_l137_137411


namespace chorus_group_membership_l137_137055

theorem chorus_group_membership (n : ℕ) : 
  100 < n ∧ n < 200 →
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  n % 8 = 6 →
  n = 118 ∨ n = 142 ∨ n = 166 ∨ n = 190 :=
by
  sorry

end chorus_group_membership_l137_137055


namespace yarn_for_second_ball_l137_137546

variable (first_ball second_ball third_ball : ℝ) (yarn_used : ℝ)

-- Conditions
variable (h1 : first_ball = second_ball / 2)
variable (h2 : third_ball = 3 * first_ball)
variable (h3 : third_ball = 27)

-- Question: Prove that the second ball used 18 feet of yarn.
theorem yarn_for_second_ball (h1 : first_ball = second_ball / 2) (h2 : third_ball = 3 * first_ball) (h3 : third_ball = 27) :
  second_ball = 18 := by
  sorry

end yarn_for_second_ball_l137_137546


namespace steve_halfway_time_longer_than_danny_l137_137899

theorem steve_halfway_time_longer_than_danny 
  (T_D : ℝ) (T_S : ℝ)
  (h1 : T_D = 33) 
  (h2 : T_S = 2 * T_D):
  (T_S / 2) - (T_D / 2) = 16.5 :=
by sorry

end steve_halfway_time_longer_than_danny_l137_137899


namespace number_of_dogs_l137_137164

theorem number_of_dogs (D C B x : ℕ) (h1 : D = 3 * x) (h2 : B = 9 * x) (h3 : D + B = 204) (h4 : 12 * x = 204) : D = 51 :=
by
  -- Proof skipped
  sorry

end number_of_dogs_l137_137164


namespace necessary_condition_for_inequality_l137_137600

theorem necessary_condition_for_inequality 
  (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x + m > 0) → m > 0 :=
by 
  sorry

end necessary_condition_for_inequality_l137_137600


namespace monotonicity_intervals_range_of_m_l137_137097

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem monotonicity_intervals (m : ℝ) (x : ℝ) (hx : x > 1):
  (m >= 1 → ∀ x' > 1, f m x' ≤ f m x) ∧
  (m < 1 → (∀ x' ∈ Set.Ioo 1 (Real.exp (1 - m)), f m x' > f m x) ∧
            (∀ x' ∈ Set.Ioi (Real.exp (1 - m)), f m x' < f m x)) := by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by
  sorry

end monotonicity_intervals_range_of_m_l137_137097


namespace order_of_values_l137_137153

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_of_values (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  -- Proof would go here
  sorry

end order_of_values_l137_137153


namespace least_cost_flower_bed_divisdes_l137_137159

theorem least_cost_flower_bed_divisdes:
  let Region1 := 5 * 2
  let Region2 := 3 * 5
  let Region3 := 2 * 4
  let Region4 := 5 * 4
  let Region5 := 5 * 3
  let Cost_Dahlias := 2.70
  let Cost_Cannas := 2.20
  let Cost_Begonias := 1.70
  let Cost_Freesias := 3.20
  let total_cost := 
    Region1 * Cost_Dahlias + 
    Region2 * Cost_Cannas + 
    Region3 * Cost_Freesias + 
    Region4 * Cost_Begonias + 
    Region5 * Cost_Cannas
  total_cost = 152.60 :=
by
  sorry

end least_cost_flower_bed_divisdes_l137_137159


namespace evaluate_expression_l137_137144

def cube_root (x : ℝ) := x^(1/3)

theorem evaluate_expression : (cube_root (9 / 32))^2 = (3/8) := 
by
  sorry

end evaluate_expression_l137_137144


namespace ofelia_ratio_is_two_l137_137380

noncomputable def OfeliaSavingsRatio : ℝ :=
  let january_savings := 10
  let may_savings := 160
  let x := (may_savings / january_savings)^(1/4)
  x

theorem ofelia_ratio_is_two : OfeliaSavingsRatio = 2 := by
  sorry

end ofelia_ratio_is_two_l137_137380


namespace probability_at_most_six_distinct_numbers_l137_137086

def roll_eight_dice : ℕ := 6^8

def favorable_cases : ℕ := 3628800

def probability_six_distinct_numbers (n : ℕ) (f : ℕ) : ℚ :=
  f / n

theorem probability_at_most_six_distinct_numbers :
  probability_six_distinct_numbers roll_eight_dice favorable_cases = 45 / 52 := by
  sorry

end probability_at_most_six_distinct_numbers_l137_137086


namespace factorization_count_l137_137120

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l137_137120


namespace circles_disjoint_l137_137693

-- Definitions of the circles
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Prove that the circles are disjoint
theorem circles_disjoint : 
  (¬ ∃ (x y : ℝ), circleM x y ∧ circleN x y) :=
by sorry

end circles_disjoint_l137_137693


namespace temperature_at_midnight_l137_137611

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l137_137611


namespace chocolate_bar_breaks_l137_137091

-- Definition of the problem as per the conditions
def chocolate_bar (rows : ℕ) (cols : ℕ) : ℕ := rows * cols

-- Statement of the proving problem
theorem chocolate_bar_breaks :
  ∀ (rows cols : ℕ), chocolate_bar rows cols = 40 → rows = 5 → cols = 8 → 
  (rows - 1) + (cols * (rows - 1)) = 39 :=
by
  intros rows cols h_bar h_rows h_cols
  sorry

end chocolate_bar_breaks_l137_137091


namespace total_additions_in_2_hours_30_minutes_l137_137090

def additions_rate : ℕ := 15000

def time_in_seconds : ℕ := 2 * 3600 + 30 * 60

def total_additions : ℕ := additions_rate * time_in_seconds

theorem total_additions_in_2_hours_30_minutes :
  total_additions = 135000000 :=
by
  -- Non-trivial proof skipped
  sorry

end total_additions_in_2_hours_30_minutes_l137_137090


namespace BoatWorks_total_canoes_by_April_l137_137251

def BoatWorksCanoes : ℕ → ℕ
| 0 => 5
| (n+1) => 2 * BoatWorksCanoes n

theorem BoatWorks_total_canoes_by_April : (BoatWorksCanoes 0) + (BoatWorksCanoes 1) + (BoatWorksCanoes 2) + (BoatWorksCanoes 3) = 75 :=
by
  sorry

end BoatWorks_total_canoes_by_April_l137_137251


namespace geom_seq_11th_term_l137_137125

/-!
The fifth and eighth terms of a geometric sequence are -2 and -54, respectively. 
What is the 11th term of this progression?
-/
theorem geom_seq_11th_term {a : ℕ → ℤ} (r : ℤ) 
  (h1 : a 5 = -2) (h2 : a 8 = -54) 
  (h3 : ∀ n : ℕ, a (n + 3) = a n * r ^ 3) : 
  a 11 = -1458 :=
sorry

end geom_seq_11th_term_l137_137125


namespace sum_of_squares_of_tom_rates_l137_137026

theorem sum_of_squares_of_tom_rates :
  ∃ r b k : ℕ, 3 * r + 4 * b + 2 * k = 104 ∧
               3 * r + 6 * b + 2 * k = 140 ∧
               r^2 + b^2 + k^2 = 440 :=
by
  sorry

end sum_of_squares_of_tom_rates_l137_137026


namespace range_of_a_l137_137953

variable (a x : ℝ)

def P : Prop := a < x ∧ x < a + 1
def q : Prop := x^2 - 7 * x + 10 ≤ 0

theorem range_of_a (h₁ : P a x → q x) (h₂ : ∃ x, q x ∧ ¬P a x) : 2 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l137_137953


namespace tenth_term_of_arithmetic_sequence_l137_137379

theorem tenth_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 2 * d = 14)
  (h2 : a + 5 * d = 32) : 
  (a + 9 * d = 56) ∧ (d = 6) := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l137_137379


namespace arithmetic_mean_is_one_l137_137266

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 :=
by
  sorry

end arithmetic_mean_is_one_l137_137266


namespace find_f_of_neg_2_l137_137975

theorem find_f_of_neg_2
  (f : ℚ → ℚ)
  (h : ∀ (x : ℚ), x ≠ 0 → 3 * f (1/x) + 2 * f x / x = x^2)
  : f (-2) = 13/5 :=
sorry

end find_f_of_neg_2_l137_137975


namespace find_common_difference_l137_137246

-- Definitions of the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a_n (k + 1) = a_n k + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
  S_n = (n : ℝ) / 2 * (a_n 1 + a_n n)

variables {a_1 d : ℝ}
variables (a_n : ℕ → ℝ)
variables (S_3 S_9 : ℝ)

-- Conditions from the problem statement
axiom a2_eq_3 : a_n 2 = 3
axiom S9_eq_6S3 : S_9 = 6 * S_3

-- The proof we need to write
theorem find_common_difference 
  (h1 : arithmetic_sequence a_n d)
  (h2 : sum_of_first_n_terms a_n 3 S_3)
  (h3 : sum_of_first_n_terms a_n 9 S_9) :
  d = 1 :=
by
  sorry

end find_common_difference_l137_137246


namespace sum_of_diagonal_elements_l137_137675

/-- Odd numbers from 1 to 49 arranged in a 5x5 grid. -/
def table : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 5
| 0, 3 => 7
| 0, 4 => 9
| 1, 0 => 11
| 1, 1 => 13
| 1, 2 => 15
| 1, 3 => 17
| 1, 4 => 19
| 2, 0 => 21
| 2, 1 => 23
| 2, 2 => 25
| 2, 3 => 27
| 2, 4 => 29
| 3, 0 => 31
| 3, 1 => 33
| 3, 2 => 35
| 3, 3 => 37
| 3, 4 => 39
| 4, 0 => 41
| 4, 1 => 43
| 4, 2 => 45
| 4, 3 => 47
| 4, 4 => 49
| _, _ => 0

/-- Proof that the sum of five numbers chosen from the table such that no two of them are in the same row or column equals 125. -/
theorem sum_of_diagonal_elements : 
  (table 0 0 + table 1 1 + table 2 2 + table 3 3 + table 4 4) = 125 := by
  sorry

end sum_of_diagonal_elements_l137_137675


namespace negation_of_universal_l137_137337

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by
  sorry

end negation_of_universal_l137_137337


namespace determine_real_coins_l137_137870

def has_fake_coin (coins : List ℝ) : Prop :=
  ∃ fake_coin ∈ coins, (∀ coin ∈ coins, coin ≠ fake_coin)

theorem determine_real_coins (coins : List ℝ) (h : has_fake_coin coins) (h_length : coins.length = 101) :
  ∃ real_coins : List ℝ, ∀ r ∈ real_coins, r ∈ coins ∧ real_coins.length ≥ 50 :=
by
  sorry

end determine_real_coins_l137_137870


namespace common_ratio_geometric_series_l137_137244

theorem common_ratio_geometric_series {a r S : ℝ} (h₁ : S = (a / (1 - r))) (h₂ : (ar^4 / (1 - r)) = S / 64) (h₃ : S ≠ 0) : r = 1 / 2 :=
sorry

end common_ratio_geometric_series_l137_137244


namespace octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l137_137430

-- Part (a) - Octahedron
/- 
A connected graph representing an octahedron. 
Each vertex has a degree of 4, making the graph Eulerian.
-/
theorem octahedron_has_eulerian_circuit : 
  ∃ circuit : List (ℕ × ℕ), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

-- Part (b) - Cube
/- 
A connected graph representing a cube.
Each vertex has a degree of 3, making it impossible for the graph to be Eulerian.
-/
theorem cube_has_no_eulerian_circuit : 
  ¬ ∃ (circuit : List (ℕ × ℕ)), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

end octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l137_137430


namespace root_ratios_equal_l137_137408

theorem root_ratios_equal (a : ℝ) (ha : 0 < a)
  (hroots : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₂ / x₁ = 2018) :
  ∃ y₁ y₂ : ℝ, 0 < y₁ ∧ 0 < y₂ ∧ y₁^3 + 1 = a * y₁^2 ∧ y₂^3 + 1 = a * y₂^2 ∧ y₂ / y₁ = 2018 :=
sorry

end root_ratios_equal_l137_137408


namespace y_squared_range_l137_137393

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) : 15 ≤ y^2 ∧ y^2 ≤ 25 :=
by
  sorry

end y_squared_range_l137_137393


namespace cocktail_cans_l137_137585

theorem cocktail_cans (prev_apple_ratio : ℝ) (prev_grape_ratio : ℝ) 
  (new_apple_cans : ℝ) : ∃ new_grape_cans : ℝ, new_grape_cans = 15 :=
by
  let prev_apple_per_can := 1 / 6
  let prev_grape_per_can := 1 / 10
  let prev_total_per_can := (1 / 6) + (1 / 10)
  let new_apple_per_can := 1 / 5
  let new_grape_per_can := prev_total_per_can - new_apple_per_can
  let result := 1 / new_grape_per_can
  use result
  sorry

end cocktail_cans_l137_137585


namespace probability_three_primes_l137_137729

def primes : List ℕ := [2, 3, 5, 7]

def is_prime (n : ℕ) : Prop := n ∈ primes

noncomputable def probability_prime : ℚ := 4/10
noncomputable def probability_non_prime : ℚ := 1 - probability_prime

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def calculation :
  ℚ := (choose 5 3) * (probability_prime ^ 3) * (probability_non_prime ^ 2)

theorem probability_three_primes :
  calculation = 720 / 3125 := by
  sorry

end probability_three_primes_l137_137729


namespace evaluate_expression_l137_137918

def my_star (A B : ℕ) : ℕ := (A + B) / 2
def my_hash (A B : ℕ) : ℕ := A * B + 1

theorem evaluate_expression : my_hash (my_star 4 6) 5 = 26 := 
by
  sorry

end evaluate_expression_l137_137918


namespace inequality_absolute_value_l137_137568

theorem inequality_absolute_value (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b :=
sorry

end inequality_absolute_value_l137_137568


namespace valid_x_y_sum_l137_137003

-- Setup the initial conditions as variables.
variables (x y : ℕ)

-- Declare the conditions as hypotheses.
theorem valid_x_y_sum (h1 : 0 < x) (h2 : x < 25)
  (h3 : 0 < y) (h4 : y < 25) (h5 : x + y + x * y = 119) :
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end valid_x_y_sum_l137_137003


namespace find_f_m_l137_137537

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l137_137537


namespace trapezoid_area_l137_137661

def trapezoid_diagonals_and_height (AC BD h : ℕ) :=
  (AC = 17) ∧ (BD = 113) ∧ (h = 15)

theorem trapezoid_area (AC BD h : ℕ) (area1 area2 : ℕ) 
  (H : trapezoid_diagonals_and_height AC BD h) :
  (area1 = 900 ∨ area2 = 780) :=
by
  sorry

end trapezoid_area_l137_137661


namespace triangle_area_proof_l137_137699

def vector2 := ℝ × ℝ

def a : vector2 := (6, 3)
def b : vector2 := (-4, 5)

noncomputable def det (u v : vector2) : ℝ := u.1 * v.2 - u.2 * v.1

noncomputable def parallelogram_area (u v : vector2) : ℝ := |det u v|

noncomputable def triangle_area (u v : vector2) : ℝ := parallelogram_area u v / 2

theorem triangle_area_proof : triangle_area a b = 21 := 
by 
  sorry

end triangle_area_proof_l137_137699


namespace lollipops_Lou_received_l137_137015

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end lollipops_Lou_received_l137_137015


namespace jace_gave_to_neighbor_l137_137645

theorem jace_gave_to_neighbor
  (earnings : ℕ) (debt : ℕ) (remaining : ℕ) (cents_per_dollar : ℕ) :
  earnings = 1000 →
  debt = 358 →
  remaining = 642 →
  cents_per_dollar = 100 →
  earnings - debt - remaining = 0
:= by
  intros h1 h2 h3 h4
  sorry

end jace_gave_to_neighbor_l137_137645


namespace smallest_number_is_33_l137_137883

theorem smallest_number_is_33 (x : ℝ) 
  (h1 : 2 * x = third)
  (h2 : 4 * x = second)
  (h3 : (x + 2 * x + 4 * x) / 3 = 77) : 
  x = 33 := 
by 
  sorry

end smallest_number_is_33_l137_137883


namespace quadratic_inequality_solution_l137_137707

theorem quadratic_inequality_solution : 
  ∀ x : ℝ, (2 * x ^ 2 + 7 * x + 3 > 0) ↔ (x < -3 ∨ x > -0.5) :=
by
  sorry

end quadratic_inequality_solution_l137_137707


namespace find_detergent_volume_l137_137767

variable (B D W : ℕ)
variable (B' D' W': ℕ)
variable (water_volume: unit)
variable (detergent_volume: unit)

def original_ratio (B D W : ℕ) : Prop := B = 2 * W / 100 ∧ D = 40 * W / 100

def altered_ratio (B' D' W' B D W : ℕ) : Prop :=
  B' = 3 * B ∧ D' = D / 2 ∧ W' = W ∧ W' = 300

theorem find_detergent_volume {B D W B' D' W'} (h₀ : original_ratio B D W) (h₁ : altered_ratio B' D' W' B D W) :
  D' = 120 :=
sorry

end find_detergent_volume_l137_137767


namespace sum_of_solutions_l137_137914

theorem sum_of_solutions (x : ℝ) (h : ∀ x, (x ≠ 1) ∧ (x ≠ -1) → ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) )) : 
  (∀ x, (x ≠ 1) ∧ (x ≠ -1) → -15 * x / (x^2 - 1) = 3 * x / (x+1) - 9 / (x-1)) → (x = ( -1 + Real.sqrt 13 ) / 2 ∨ x = ( -1 - Real.sqrt 13 ) / 2) → (x + ( -x ) = -1) :=
by
  sorry

end sum_of_solutions_l137_137914


namespace find_y_l137_137234

variables (x y : ℝ)

theorem find_y (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 515400) : y = 1 / 2 :=
sorry

end find_y_l137_137234


namespace simplify_expression_l137_137608

variable (a b : ℝ)

theorem simplify_expression :
  -2 * (a^3 - 3 * b^2) + 4 * (-b^2 + a^3) = 2 * a^3 + 2 * b^2 :=
by
  sorry

end simplify_expression_l137_137608


namespace noah_has_largest_final_answer_l137_137033

def liam_initial := 15
def liam_final := (liam_initial - 2) * 3 + 3

def mia_initial := 15
def mia_final := (mia_initial * 3 - 4) + 3

def noah_initial := 15
def noah_final := ((noah_initial - 3) + 4) * 3

theorem noah_has_largest_final_answer : noah_final > liam_final ∧ noah_final > mia_final := by
  -- Placeholder for actual proof
  sorry

end noah_has_largest_final_answer_l137_137033


namespace no_such_integers_l137_137474

theorem no_such_integers (x y z : ℤ) : ¬ ((x - y)^3 + (y - z)^3 + (z - x)^3 = 2011) :=
sorry

end no_such_integers_l137_137474


namespace find_m_l137_137256

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m + 1) * x + m = 0}

theorem find_m (m : ℝ) : B m ⊆ A → (m = 1 ∨ m = 2) :=
sorry

end find_m_l137_137256


namespace jessica_speed_last_40_l137_137972

theorem jessica_speed_last_40 
  (total_distance : ℕ)
  (total_time_min : ℕ)
  (first_segment_avg_speed : ℕ)
  (second_segment_avg_speed : ℕ)
  (last_segment_avg_speed : ℕ) :
  total_distance = 120 →
  total_time_min = 120 →
  first_segment_avg_speed = 50 →
  second_segment_avg_speed = 60 →
  last_segment_avg_speed = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end jessica_speed_last_40_l137_137972


namespace inequality_one_inequality_two_l137_137361

theorem inequality_one (a b : ℝ) : 
    a^2 + b^2 ≥ (a + b)^2 / 2 := 
by
    sorry

theorem inequality_two (a b : ℝ) : 
    a^2 + b^2 ≥ 2 * (a - b - 1) := 
by
    sorry

end inequality_one_inequality_two_l137_137361


namespace range_of_c_monotonicity_of_g_l137_137905

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l137_137905


namespace rachel_budget_proof_l137_137233

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end rachel_budget_proof_l137_137233


namespace calculate_expression_l137_137315

variable (a : ℝ)

theorem calculate_expression : 2 * a - 7 * a + 4 * a = -a := by
  sorry

end calculate_expression_l137_137315


namespace subproblem1_l137_137169

theorem subproblem1 (a b c q : ℝ) (h1 : c = b * q) (h2 : c = a * q^2) : 
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 := 
sorry

end subproblem1_l137_137169


namespace average_p_q_l137_137554

theorem average_p_q (p q : ℝ) 
  (h1 : (4 + 6 + 8 + 2 * p + 2 * q) / 7 = 20) : 
  (p + q) / 2 = 30.5 :=
by
  sorry

end average_p_q_l137_137554


namespace partition_sum_le_152_l137_137214

theorem partition_sum_le_152 {S : ℕ} (l : List ℕ) 
  (h1 : ∀ n ∈ l, 1 ≤ n ∧ n ≤ 10) 
  (h2 : l.sum = S) : 
  (∃ l1 l2 : List ℕ, l1.sum ≤ 80 ∧ l2.sum ≤ 80 ∧ l1 ++ l2 = l) ↔ S ≤ 152 := 
by
  sorry

end partition_sum_le_152_l137_137214


namespace webinar_active_minutes_l137_137254

theorem webinar_active_minutes :
  let hours := 13
  let extra_minutes := 17
  let break_minutes := 22
  (hours * 60 + extra_minutes) - break_minutes = 775 := by
  sorry

end webinar_active_minutes_l137_137254


namespace solve_for_x_l137_137985

noncomputable def is_satisfied (x : ℝ) : Prop :=
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2

theorem solve_for_x :
  ∀ x : ℝ, 0 < x → x ≠ 1 ↔ is_satisfied x := by
  sorry

end solve_for_x_l137_137985


namespace carrot_servings_l137_137974

theorem carrot_servings (C : ℕ) 
  (H1 : ∀ (corn_servings : ℕ), corn_servings = 5 * C)
  (H2 : ∀ (green_bean_servings : ℕ) (corn_servings : ℕ), green_bean_servings = corn_servings / 2)
  (H3 : ∀ (plot_plants : ℕ), plot_plants = 9)
  (H4 : ∀ (total_servings : ℕ) 
         (carrot_servings : ℕ)
         (corn_servings : ℕ)
         (green_bean_servings : ℕ), 
         total_servings = carrot_servings + corn_servings + green_bean_servings ∧
         total_servings = 306) : 
  C = 4 := 
    sorry

end carrot_servings_l137_137974


namespace geometric_sequence_a3_value_l137_137027

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem geometric_sequence_a3_value :
  ∃ a : ℕ → ℝ, ∃ r : ℝ,
  geometric_seq a r ∧
  a 1 = 2 ∧
  (a 3) * (a 5) = 4 * (a 6)^2 →
  a 3 = 1 :=
sorry

end geometric_sequence_a3_value_l137_137027


namespace fraction_expression_as_common_fraction_l137_137721

theorem fraction_expression_as_common_fraction :
  ((3 / 7 + 5 / 8) / (5 / 12 + 2 / 15)) = (295 / 154) := 
by
  sorry

end fraction_expression_as_common_fraction_l137_137721


namespace q_value_l137_137662

-- Define the problem conditions
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- Statement of the problem
theorem q_value (p q : ℕ) (hp : prime p) (hq : prime q) (h1 : q = 13 * p + 2) (h2 : is_multiple_of (q - 1) 3) : q = 67 :=
sorry

end q_value_l137_137662


namespace limsup_subset_l137_137687

variable {Ω : Type*} -- assuming a universal sample space Ω for the events A_n and B_n

def limsup (A : ℕ → Set Ω) : Set Ω := 
  ⋂ k, ⋃ n ≥ k, A n

theorem limsup_subset {A B : ℕ → Set Ω} (h : ∀ n, A n ⊆ B n) : 
  limsup A ⊆ limsup B :=
by
  -- here goes the proof
  sorry

end limsup_subset_l137_137687


namespace michelle_phone_bill_l137_137633

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.05
def minute_cost_over_20h : ℝ := 0.20
def messages_sent : ℝ := 150
def hours_talked : ℝ := 22
def allowed_hours : ℝ := 20

theorem michelle_phone_bill :
  base_cost + (messages_sent * text_cost_per_message) +
  ((hours_talked - allowed_hours) * 60 * minute_cost_over_20h) = 51.50 := by
  sorry

end michelle_phone_bill_l137_137633


namespace apple_tree_production_l137_137644

def first_year_production : ℕ := 40
def second_year_production (first_year_production : ℕ) : ℕ := 2 * first_year_production + 8
def third_year_production (second_year_production : ℕ) : ℕ := second_year_production - (second_year_production / 4)
def total_production (first_year_production second_year_production third_year_production : ℕ) : ℕ :=
    first_year_production + second_year_production + third_year_production

-- Proof statement
theorem apple_tree_production : total_production 40 88 66 = 194 := by
  sorry

end apple_tree_production_l137_137644


namespace total_assignments_for_28_points_l137_137803

-- Definitions based on conditions
def assignments_needed (points : ℕ) : ℕ :=
  (points / 7 + 1) * (points % 7) + (points / 7) * (7 - points % 7)

-- The theorem statement, which asserts the answer to the given problem
theorem total_assignments_for_28_points : assignments_needed 28 = 70 :=
by
  -- proof will go here
  sorry

end total_assignments_for_28_points_l137_137803


namespace solve_for_k_l137_137830

theorem solve_for_k (x y k : ℤ) (h1 : x = -3) (h2 : y = 2) (h3 : 2 * x + k * y = 0) : k = 3 :=
by
  sorry

end solve_for_k_l137_137830


namespace thomas_total_blocks_l137_137330

-- Definitions according to the conditions
def a1 : Nat := 7
def a2 : Nat := a1 + 3
def a3 : Nat := a2 - 6
def a4 : Nat := a3 + 10
def a5 : Nat := 2 * a2

-- The total number of blocks
def total_blocks : Nat := a1 + a2 + a3 + a4 + a5

-- The proof statement
theorem thomas_total_blocks :
  total_blocks = 55 := 
sorry

end thomas_total_blocks_l137_137330


namespace blue_stripe_area_l137_137322

def cylinder_diameter : ℝ := 20
def cylinder_height : ℝ := 60
def stripe_width : ℝ := 4
def stripe_revolutions : ℕ := 3

theorem blue_stripe_area : 
  let circumference := Real.pi * cylinder_diameter
  let stripe_length := stripe_revolutions * circumference
  let expected_area := stripe_width * stripe_length
  expected_area = 240 * Real.pi :=
by
  sorry

end blue_stripe_area_l137_137322


namespace no_root_of_equation_l137_137016

theorem no_root_of_equation : ∀ x : ℝ, x - 8 / (x - 4) ≠ 4 - 8 / (x - 4) :=
by
  intro x
  -- Original equation:
  -- x - 8 / (x - 4) = 4 - 8 / (x - 4)
  -- No valid value of x solves the above equation as shown in the given solution
  sorry

end no_root_of_equation_l137_137016


namespace total_surface_area_of_cubes_aligned_side_by_side_is_900_l137_137269

theorem total_surface_area_of_cubes_aligned_side_by_side_is_900 :
  let volumes := [27, 64, 125, 216, 512]
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  (surface_areas.sum = 900) :=
by
  sorry

end total_surface_area_of_cubes_aligned_side_by_side_is_900_l137_137269


namespace gcd_72_108_l137_137626

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l137_137626


namespace always_true_statements_l137_137119

variable (a b c : ℝ)

theorem always_true_statements (h1 : a < 0) (h2 : a < b ∧ b ≤ 0) (h3 : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) :=
by 
  sorry

end always_true_statements_l137_137119


namespace possible_values_of_n_are_1_prime_or_prime_squared_l137_137479

/-- A function that determines if an n x n grid with n marked squares satisfies the condition
    that every rectangle of exactly n grid squares contains at least one marked square. -/
def satisfies_conditions (n : ℕ) (marked_squares : List (ℕ × ℕ)) : Prop :=
  n.succ.succ ≤ marked_squares.length ∧ ∀ (a b : ℕ), a * b = n → ∃ x y, (x, y) ∈ marked_squares ∧ x < n ∧ y < n

/-- The main theorem stating the possible values of n. -/
theorem possible_values_of_n_are_1_prime_or_prime_squared :
  ∀ (n : ℕ), (∃ p : ℕ, Prime p ∧ (n = 1 ∨ n = p ∨ n = p^2)) ↔ satisfies_conditions n marked_squares :=
by
  sorry

end possible_values_of_n_are_1_prime_or_prime_squared_l137_137479


namespace relationship_between_fractions_l137_137161

variable (a a' b b' : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a' > 0)
variable (h₃ : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2)

theorem relationship_between_fractions
  (a : ℝ) (a' : ℝ) (b : ℝ) (b' : ℝ)
  (h1 : a > 0) (h2 : a' > 0)
  (h3 : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2) :
  (b^2) / (a^2) > (b'^2) / (a'^2) :=
by sorry

end relationship_between_fractions_l137_137161


namespace volume_of_pyramid_base_isosceles_right_triangle_l137_137531

theorem volume_of_pyramid_base_isosceles_right_triangle (a h : ℝ) (ha : a = 3) (hh : h = 4) :
  (1 / 3) * (1 / 2) * a * a * h = 6 := by
  sorry

end volume_of_pyramid_base_isosceles_right_triangle_l137_137531


namespace sufficient_but_not_necessary_l137_137561

theorem sufficient_but_not_necessary (a b : ℝ) :
  ((a - b) ^ 3 * b ^ 2 > 0 → a > b) ∧ ¬(a > b → (a - b) ^ 3 * b ^ 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_l137_137561


namespace number_of_mappings_A_to_B_number_of_mappings_B_to_A_l137_137460

theorem number_of_mappings_A_to_B (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (B.card ^ A.card) = 4^5 :=
by sorry

theorem number_of_mappings_B_to_A (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (A.card ^ B.card) = 5^4 :=
by sorry

end number_of_mappings_A_to_B_number_of_mappings_B_to_A_l137_137460


namespace sum_of_solutions_l137_137714

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l137_137714


namespace min_value_at_1_l137_137056

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * a * x + 8 else x + 4 / x + 2 * a

theorem min_value_at_1 (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ (a = 5/4 ∨ a = 2 ∨ a = 4) :=
by
  sorry

end min_value_at_1_l137_137056


namespace ratio_of_albert_to_mary_l137_137504

variables (A M B : ℕ) (s : ℕ) 

-- Given conditions as hypotheses
noncomputable def albert_is_multiple_of_mary := A = s * M
noncomputable def albert_is_4_times_betty := A = 4 * B
noncomputable def mary_is_22_years_younger := M = A - 22
noncomputable def betty_is_11 := B = 11

-- Theorem to prove the ratio of Albert's age to Mary's age
theorem ratio_of_albert_to_mary 
  (h1 : albert_is_multiple_of_mary A M s) 
  (h2 : albert_is_4_times_betty A B) 
  (h3 : mary_is_22_years_younger A M) 
  (h4 : betty_is_11 B) : 
  A / M = 2 :=
by
  sorry

end ratio_of_albert_to_mary_l137_137504


namespace quadratic_properties_l137_137104

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

def min_value_passing_point (f : ℝ → ℝ) : Prop :=
  (f (-1) = -4) ∧ (f 0 = -3)

def intersects_x_axis (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 = p1.2) ∧ (f p2.1 = p2.2)

def max_value_in_interval (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ max_val

theorem quadratic_properties :
  ∃ f : ℝ → ℝ,
    min_value_passing_point f ∧
    intersects_x_axis f (1, 0) (-3, 0) ∧
    max_value_in_interval f (-2) 2 5 :=
by
  sorry

end quadratic_properties_l137_137104


namespace soccer_team_games_played_l137_137085

theorem soccer_team_games_played 
  (players : ℕ) (total_goals : ℕ) (third_players_goals_per_game : ℕ → ℕ) (other_players_goals : ℕ) (G : ℕ)
  (h1 : players = 24)
  (h2 : total_goals = 150)
  (h3 : ∃ n, n = players / 3 ∧ ∀ g, third_players_goals_per_game g = n * g)
  (h4 : other_players_goals = 30)
  (h5 : total_goals = third_players_goals_per_game G + other_players_goals) :
  G = 15 := by
  -- Proof would go here
  sorry

end soccer_team_games_played_l137_137085


namespace container_ratio_l137_137868

theorem container_ratio (V1 V2 V3 : ℝ)
  (h1 : (3 / 4) * V1 = (5 / 8) * V2)
  (h2 : (5 / 8) * V2 = (1 / 2) * V3) :
  V1 / V3 = 1 / 2 :=
by
  sorry

end container_ratio_l137_137868


namespace molecular_weight_calculation_l137_137162

theorem molecular_weight_calculation
    (moles_total_mw : ℕ → ℝ)
    (hw : moles_total_mw 9 = 900) :
    moles_total_mw 1 = 100 :=
by
  sorry

end molecular_weight_calculation_l137_137162


namespace tim_initial_soda_l137_137228

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end tim_initial_soda_l137_137228


namespace rectangle_area_l137_137001

def radius : ℝ := 10
def width : ℝ := 2 * radius
def length : ℝ := 3 * width
def area_of_rectangle : ℝ := length * width

theorem rectangle_area : area_of_rectangle = 1200 :=
  by sorry

end rectangle_area_l137_137001


namespace function_order_l137_137831

theorem function_order (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (2 - x) = f x)
  (h2 : ∀ x : ℝ, f (x + 2) = f (x - 2))
  (h3 : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 → (f x1 - f x2) / (x1 - x2) < 0) :
  f 2016 = f 2014 ∧ f 2014 > f 2015 :=
by
  sorry

end function_order_l137_137831


namespace find_fourth_number_l137_137564

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l137_137564


namespace maximum_expr_value_l137_137961

theorem maximum_expr_value :
  ∃ (x y e f : ℕ), (e = 4 ∧ x = 3 ∧ y = 2 ∧ f = 0) ∧
  (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4) ∧
  (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ∧
  (y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4) ∧
  (f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4) ∧
  (e ≠ x ∧ e ≠ y ∧ e ≠ f ∧ x ≠ y ∧ x ≠ f ∧ y ≠ f) ∧
  (e * x^y - f = 36) :=
by
  sorry

end maximum_expr_value_l137_137961


namespace pattern_equation_l137_137539

theorem pattern_equation (n : ℕ) (hn : n > 0) : n * (n + 2) + 1 = (n + 1) ^ 2 := 
by sorry

end pattern_equation_l137_137539


namespace correct_calculation_l137_137528

-- Define the variables used in the problem
variables (a x y : ℝ)

-- The main theorem statement
theorem correct_calculation : (2 * x * y^2 - x * y^2 = x * y^2) :=
by sorry

end correct_calculation_l137_137528


namespace three_f_x_eq_l137_137743

theorem three_f_x_eq (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 2 / (3 + x)) (x : ℝ) (hx : x > 0) : 
  3 * f x = 18 / (9 + x) := sorry

end three_f_x_eq_l137_137743


namespace find_value_of_expression_l137_137654

theorem find_value_of_expression (a : ℝ) (h : a^2 + 3 * a - 1 = 0) : 2 * a^2 + 6 * a + 2021 = 2023 := 
by
  sorry

end find_value_of_expression_l137_137654


namespace find_m_range_l137_137646

noncomputable def proposition_p (x : ℝ) : Prop := (-2 : ℝ) ≤ x ∧ x ≤ 10
noncomputable def proposition_q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m)

theorem find_m_range (m : ℝ) (h : m > 0) : (¬ ∃ x : ℝ, proposition_p x) → (¬ ∃ x : ℝ, proposition_q x m) → (¬ (¬ (¬ ∃ x : ℝ, proposition_q x m)) → ¬ (¬ ∃ x : ℝ, proposition_p x)) → m ≥ 9 := 
sorry

end find_m_range_l137_137646


namespace shoe_count_l137_137290

theorem shoe_count 
  (pairs : ℕ)
  (total_shoes : ℕ)
  (prob : ℝ)
  (h_pairs : pairs = 12)
  (h_prob : prob = 0.043478260869565216)
  (h_total_shoes : total_shoes = pairs * 2) :
  total_shoes = 24 :=
by
  sorry

end shoe_count_l137_137290


namespace find_a_l137_137639

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (1 + a * 2^x)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x, f x = 2^x / (1 + a * 2^x))
  (h_symm : ∀ x, f x + f (-x) = 1) : a = 1 :=
sorry

end find_a_l137_137639


namespace trig_identity_proof_l137_137044

theorem trig_identity_proof :
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  (Real.sin (Real.pi / 36) * Real.sin (5 * Real.pi / 36) - sin_95 * sin_65) = - (Real.sqrt 3) / 2 :=
by
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  sorry

end trig_identity_proof_l137_137044


namespace compute_complex_power_l137_137555

noncomputable def complex_number := Complex.exp (Complex.I * 125 * Real.pi / 180)

theorem compute_complex_power :
  (complex_number ^ 28) = Complex.ofReal (-Real.cos (40 * Real.pi / 180)) + Complex.I * Real.sin (40 * Real.pi / 180) :=
by
  sorry

end compute_complex_power_l137_137555


namespace max_value_of_expression_l137_137981

theorem max_value_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  (∃ (x : ℝ), x = (ab/(a + b)) + (ac/(a + c)) + (bc/(b + c)) ∧ x = 9/4) :=
  sorry

end max_value_of_expression_l137_137981


namespace h_f_equals_h_g_l137_137298

def f (x : ℝ) := x^2 - x + 1

def g (x : ℝ) := -x^2 + x + 1

def h (x : ℝ) := (x - 1)^2

theorem h_f_equals_h_g : ∀ x : ℝ, h (f x) = h (g x) :=
by
  intro x
  unfold f g h
  sorry

end h_f_equals_h_g_l137_137298


namespace vacation_cost_division_l137_137070

theorem vacation_cost_division (n : ℕ) (h1 : 720 / 4 = 60 + 720 / n) : n = 3 := by
  sorry

end vacation_cost_division_l137_137070


namespace increased_colored_area_l137_137357

theorem increased_colored_area
  (P : ℝ) -- Perimeter of the original convex pentagon
  (s : ℝ) -- Distance from the points colored originally
  : 
  s * P + π * s^2 = 23.14 :=
by
  sorry

end increased_colored_area_l137_137357


namespace original_number_is_8_l137_137784

open Real

theorem original_number_is_8 
  (x : ℝ)
  (h1 : |(x + 5) - (x - 5)| = 10)
  (h2 : (10 / (x + 5)) * 100 = 76.92) : 
  x = 8 := 
by
  sorry

end original_number_is_8_l137_137784


namespace range_of_k_l137_137338

theorem range_of_k (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ ⦃x y⦄, 0 ≤ x → x < y → f y < f x) 
  (h_inequality : ∀ x, f (k * x ^ 2 + 2) + f (k * x + k) ≤ 0) : 0 ≤ k :=
sorry

end range_of_k_l137_137338


namespace bananas_indeterminate_l137_137402

namespace RubyBananaProblem

variables (number_of_candies : ℕ) (number_of_friends : ℕ) (candies_per_friend : ℕ)
           (number_of_bananas : Option ℕ)

-- Given conditions
def Ruby_has_36_candies := number_of_candies = 36
def Ruby_has_9_friends := number_of_friends = 9
def Each_friend_gets_4_candies := candies_per_friend = 4
def Can_distribute_candies := number_of_candies = number_of_friends * candies_per_friend

-- Mathematical statement
theorem bananas_indeterminate (h1 : Ruby_has_36_candies number_of_candies)
                              (h2 : Ruby_has_9_friends number_of_friends)
                              (h3 : Each_friend_gets_4_candies candies_per_friend)
                              (h4 : Can_distribute_candies number_of_candies number_of_friends candies_per_friend) :
  number_of_bananas = none :=
by
  sorry

end RubyBananaProblem

end bananas_indeterminate_l137_137402


namespace chord_bisected_vertically_by_line_l137_137896

theorem chord_bisected_vertically_by_line (p : ℝ) (h : p > 0) (l : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus: focus = (p / 2, 0)) (h_line: ∀ x, l x ≠ 0) :
  ¬ ∃ (A B : ℝ × ℝ), 
     A.1 ≠ B.1 ∧
     A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
     (A.1 + B.1) / 2 = focus.1 ∧ 
     l ((A.1 + B.1) / 2) = focus.2 :=
sorry

end chord_bisected_vertically_by_line_l137_137896


namespace sin_alpha_value_l137_137945

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end sin_alpha_value_l137_137945


namespace correct_judgements_l137_137340

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_period_1 : ∀ x : ℝ, f (x + 1) = -f x
axiom f_increasing_0_1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

theorem correct_judgements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧ 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧ 
  ¬(∀ x y : ℝ, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y) :=
by 
  sorry

end correct_judgements_l137_137340


namespace find_triplet_l137_137060

def ordered_triplet : Prop :=
  ∃ (x y z : ℚ), 
  7 * x + 3 * y = z - 10 ∧ 
  2 * x - 4 * y = 3 * z + 20 ∧ 
  x = 0 ∧ 
  y = -50 / 13 ∧ 
  z = -20 / 13

theorem find_triplet : ordered_triplet := 
  sorry

end find_triplet_l137_137060


namespace license_plate_problem_l137_137356

noncomputable def license_plate_ways : ℕ :=
  let letters := 26
  let digits := 10
  let both_same := letters * digits * 1 * 1
  let digits_adj_same := letters * digits * 1 * letters
  let letters_adj_same := letters * digits * digits * 1
  digits_adj_same + letters_adj_same - both_same

theorem license_plate_problem :
  9100 = license_plate_ways :=
by
  -- Skipping the detailed proof for now
  sorry

end license_plate_problem_l137_137356


namespace woman_age_multiple_l137_137339

theorem woman_age_multiple (S : ℕ) (W : ℕ) (k : ℕ) 
  (h1 : S = 27)
  (h2 : W + S = 84)
  (h3 : W = k * S + 3) :
  k = 2 :=
by
  sorry

end woman_age_multiple_l137_137339


namespace ten_percent_of_fifty_percent_of_five_hundred_l137_137977

theorem ten_percent_of_fifty_percent_of_five_hundred :
  0.10 * (0.50 * 500) = 25 :=
by
  sorry

end ten_percent_of_fifty_percent_of_five_hundred_l137_137977


namespace sin_18_cos_36_eq_quarter_l137_137360

theorem sin_18_cos_36_eq_quarter : Real.sin (Real.pi / 10) * Real.cos (Real.pi / 5) = 1 / 4 :=
by
  sorry

end sin_18_cos_36_eq_quarter_l137_137360


namespace inequality_proof_l137_137400

open Real

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^2 / (b + c + d) + b^2 / (c + d + a) +
   c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2 / 3) :=
by
  sorry

end inequality_proof_l137_137400


namespace poly_perfect_fourth_l137_137798

theorem poly_perfect_fourth (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, (a * x^2 + b * x + c) = k^4) : 
  a = 0 ∧ b = 0 :=
sorry

end poly_perfect_fourth_l137_137798


namespace probability_three_white_two_black_l137_137108

-- Define the total number of balls
def total_balls : ℕ := 17

-- Define the number of white balls
def white_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 9

-- Define the number of balls drawn
def balls_drawn : ℕ := 5

-- Define three white balls drawn
def three_white_drawn : ℕ := 3

-- Define two black balls drawn
def two_black_drawn : ℕ := 2

-- Define the combination formula
noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Calculate the probability
noncomputable def probability : ℚ :=
  (combination white_balls three_white_drawn * combination black_balls two_black_drawn : ℚ) 
  / combination total_balls balls_drawn

-- Statement to prove
theorem probability_three_white_two_black :
  probability = 672 / 2063 := by
  sorry

end probability_three_white_two_black_l137_137108


namespace product_of_squares_is_perfect_square_l137_137563

theorem product_of_squares_is_perfect_square (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
    ∃ k : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = k^2 :=
sorry

end product_of_squares_is_perfect_square_l137_137563


namespace quadratic_function_origin_l137_137038

theorem quadratic_function_origin {a b c : ℝ} :
  (∀ x, y = ax * x + bx * x + c → y = 0 → 0 = c ∧ b = 0) ∨ (c = 0) :=
sorry

end quadratic_function_origin_l137_137038


namespace combine_polynomials_find_value_profit_or_loss_l137_137466

-- Problem 1, Part ①
theorem combine_polynomials (a b : ℝ) : -3 * (a+b)^2 - 6 * (a+b)^2 + 8 * (a+b)^2 = -(a+b)^2 := 
sorry

-- Problem 1, Part ②
theorem find_value (a b c d : ℝ) (h1 : a - 2 * b = 5) (h2 : 2 * b - c = -7) (h3 : c - d = 12) : 
  4 * (a - c) + 4 * (2 * b - d) - 4 * (2 * b - c) = 40 := 
sorry

-- Problem 2
theorem profit_or_loss (initial_cost : ℝ) (selling_prices : ℕ → ℝ) (base_price : ℝ) 
  (h_prices : selling_prices 0 = -3) (h_prices1 : selling_prices 1 = 7) 
  (h_prices2 : selling_prices 2 = -8) (h_prices3 : selling_prices 3 = 9) 
  (h_prices4 : selling_prices 4 = -2) (h_prices5 : selling_prices 5 = 0) 
  (h_prices6 : selling_prices 6 = -1) (h_prices7 : selling_prices 7 = -6) 
  (h_initial_cost : initial_cost = 400) (h_base_price : base_price = 56) : 
  (selling_prices 0 + selling_prices 1 + selling_prices 2 + selling_prices 3 + selling_prices 4 + selling_prices 5 + 
  selling_prices 6 + selling_prices 7 + 8 * base_price) - initial_cost > 0 := 
sorry

end combine_polynomials_find_value_profit_or_loss_l137_137466


namespace Trent_traveled_distance_l137_137620

variable (blocks_length : ℕ := 50)
variables (walking_blocks : ℕ := 4) (bus_blocks : ℕ := 7) (bicycle_blocks : ℕ := 5)
variables (walking_round_trip : ℕ := 2 * walking_blocks * blocks_length)
variables (bus_round_trip : ℕ := 2 * bus_blocks * blocks_length)
variables (bicycle_round_trip : ℕ := 2 * bicycle_blocks * blocks_length)

def total_distance_traveleed : ℕ :=
  walking_round_trip + bus_round_trip + bicycle_round_trip

theorem Trent_traveled_distance :
  total_distance_traveleed = 1600 := by
    sorry

end Trent_traveled_distance_l137_137620


namespace greatest_common_divisor_84_n_l137_137294

theorem greatest_common_divisor_84_n :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∣ 84 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4) ∧ (∀ (x y : ℕ), x ∣ 84 ∧ x ∣ n ∧ y ∣ 84 ∧ y ∣ n → x ≤ y → y = 4) :=
sorry

end greatest_common_divisor_84_n_l137_137294


namespace fraction_simplification_l137_137150

theorem fraction_simplification (x y : ℚ) (h1 : x = 4) (h2 : y = 5) : 
  (1 / y) / (1 / x) = 4 / 5 :=
by
  sorry

end fraction_simplification_l137_137150


namespace range_of_a_l137_137232

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) 
  : -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l137_137232


namespace compute_v_l137_137666

variable (a b c : ℝ)

theorem compute_v (H1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -8)
                  (H2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 12)
                  (H3 : a * b * c = 1) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = -8.5 :=
sorry

end compute_v_l137_137666


namespace disproving_equation_l137_137210

theorem disproving_equation 
  (a b c d : ℚ)
  (h : a / b = c / d)
  (ha : a ≠ 0)
  (hc : c ≠ 0) : 
  a + d ≠ (a / b) * (b + c) := 
by 
  sorry

end disproving_equation_l137_137210


namespace MrWillamTaxPercentage_l137_137193

-- Definitions
def TotalTaxCollected : ℝ := 3840
def MrWillamTax : ℝ := 480

-- Theorem Statement
theorem MrWillamTaxPercentage :
  (MrWillamTax / TotalTaxCollected) * 100 = 12.5 :=
by
  sorry

end MrWillamTaxPercentage_l137_137193


namespace gcd_8885_4514_5246_l137_137458

theorem gcd_8885_4514_5246 : Nat.gcd (Nat.gcd 8885 4514) 5246 = 1 :=
sorry

end gcd_8885_4514_5246_l137_137458


namespace length_of_floor_y_l137_137175

theorem length_of_floor_y
  (A B : ℝ)
  (hx : A = 10)
  (hy : B = 18)
  (width_y : ℝ)
  (length_y : ℝ)
  (width_y_eq : width_y = 9)
  (area_eq : A * B = width_y * length_y) :
  length_y = 20 := 
sorry

end length_of_floor_y_l137_137175


namespace vanya_meets_mother_opposite_dir_every_4_minutes_l137_137996

-- Define the parameters
def lake_perimeter : ℝ := sorry  -- Length of the lake's perimeter, denoted as l
def mother_time_lap : ℝ := 12    -- Time taken by the mother to complete one lap (in minutes)
def vanya_time_overtake : ℝ := 12 -- Time taken by Vanya to overtake the mother (in minutes)

-- Define speeds
noncomputable def mother_speed : ℝ := lake_perimeter / mother_time_lap
noncomputable def vanya_speed : ℝ := 2 * lake_perimeter / vanya_time_overtake

-- Define their relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := mother_speed + vanya_speed

-- Prove that the meeting interval is 4 minutes
theorem vanya_meets_mother_opposite_dir_every_4_minutes :
  (lake_perimeter / relative_speed) = 4 := by
  sorry

end vanya_meets_mother_opposite_dir_every_4_minutes_l137_137996


namespace radio_selling_price_l137_137948

theorem radio_selling_price (CP LP Loss SP : ℝ) (h1 : CP = 1500) (h2 : LP = 11)
  (h3 : Loss = (LP / 100) * CP) (h4 : SP = CP - Loss) : SP = 1335 := 
  by
  -- hint: Apply the given conditions.
  sorry

end radio_selling_price_l137_137948


namespace option_transformations_incorrect_l137_137777

variable {a b x : ℝ}

theorem option_transformations_incorrect (h : a < b) :
  ¬ (3 - a < 3 - b) := by
  -- Here, we would show the incorrectness of the transformation in Option B
  sorry

end option_transformations_incorrect_l137_137777


namespace hose_rate_l137_137025

theorem hose_rate (V : ℝ) (T : ℝ) (r_fixed : ℝ) (total_rate : ℝ) (R : ℝ) :
  V = 15000 ∧ T = 25 ∧ r_fixed = 3 ∧ total_rate = 10 ∧
  (2 * R + 2 * r_fixed = total_rate) → R = 2 :=
by
  -- Given conditions:
  -- Volume V = 15000 gallons
  -- Time T = 25 hours
  -- Rate of fixed hoses r_fixed = 3 gallons per minute each
  -- Total rate of filling the pool total_rate = 10 gallons per minute
  -- Relationship: 2 * rate of first two hoses + 2 * rate of fixed hoses = total rate
  
  sorry

end hose_rate_l137_137025


namespace ab_zero_l137_137647

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l137_137647


namespace books_sold_l137_137503

theorem books_sold (initial_books left_books sold_books : ℕ) (h1 : initial_books = 108) (h2 : left_books = 66) : sold_books = 42 :=
by
  have : sold_books = initial_books - left_books := sorry
  rw [h1, h2] at this
  exact this

end books_sold_l137_137503


namespace smallest_eraser_packs_needed_l137_137853

def yazmin_packs_condition (pencils_packs erasers_packs pencils_per_pack erasers_per_pack : ℕ) : Prop :=
  pencils_packs * pencils_per_pack = erasers_packs * erasers_per_pack

theorem smallest_eraser_packs_needed (pencils_per_pack erasers_per_pack : ℕ) (h_pencils_5 : pencils_per_pack = 5) (h_erasers_7 : erasers_per_pack = 7) : ∃ erasers_packs, yazmin_packs_condition 7 erasers_packs pencils_per_pack erasers_per_pack ∧ erasers_packs = 5 :=
by
  sorry

end smallest_eraser_packs_needed_l137_137853


namespace lcm_gcd_product_difference_l137_137921
open Nat

theorem lcm_gcd_product_difference :
  (Nat.lcm 12 9) * (Nat.gcd 12 9) - (Nat.gcd 15 9) = 105 :=
by
  sorry

end lcm_gcd_product_difference_l137_137921


namespace james_calories_per_minute_l137_137591

-- Define the conditions
def bags : Nat := 3
def ounces_per_bag : Nat := 2
def calories_per_ounce : Nat := 150
def excess_calories : Nat := 420
def run_minutes : Nat := 40

-- Calculate the total consumed calories
def consumed_calories : Nat := (bags * ounces_per_bag) * calories_per_ounce

-- Calculate the calories burned during the run
def run_calories : Nat := consumed_calories - excess_calories

-- Calculate the calories burned per minute
def calories_per_minute : Nat := run_calories / run_minutes

-- The proof problem statement
theorem james_calories_per_minute : calories_per_minute = 12 := by
  -- Due to the proof not required, we use sorry to skip it.
  sorry

end james_calories_per_minute_l137_137591


namespace initial_brownies_l137_137111

theorem initial_brownies (B : ℕ) (eaten_by_father : ℕ) (eaten_by_mooney : ℕ) (new_brownies : ℕ) (total_brownies : ℕ) :
  eaten_by_father = 8 →
  eaten_by_mooney = 4 →
  new_brownies = 24 →
  total_brownies = 36 →
  (B - (eaten_by_father + eaten_by_mooney) + new_brownies = total_brownies) →
  B = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end initial_brownies_l137_137111


namespace negation_proof_l137_137822

theorem negation_proof :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) → (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_proof_l137_137822


namespace plane_equation_through_point_and_parallel_l137_137761

theorem plane_equation_through_point_and_parallel (P : ℝ × ℝ × ℝ) (D : ℝ)
  (normal_vector : ℝ × ℝ × ℝ) (A B C : ℝ)
  (h1 : normal_vector = (2, -1, 3))
  (h2 : P = (2, 3, -1))
  (h3 : A = 2) (h4 : B = -1) (h5 : C = 3)
  (hD : A * 2 + B * 3 + C * -1 + D = 0) :
  A * x + B * y + C * z + D = 0 :=
by
  sorry

end plane_equation_through_point_and_parallel_l137_137761


namespace exists_line_equidistant_from_AB_CD_l137_137309

noncomputable def Line : Type := sorry  -- This would be replaced with an appropriate definition of a line in space

def Point : Type := sorry  -- Similarly, a point in space type definition

variables (A B C D : Point)

def perpendicularBisector (P Q : Point) : Type := sorry  -- Definition for perpendicular bisector plane of two points

def is_perpendicularBisector_of (e : Line) (P Q : Point) : Prop := sorry  -- e is perpendicular bisector plane of P and Q

theorem exists_line_equidistant_from_AB_CD (A B C D : Point) :
  ∃ e : Line, is_perpendicularBisector_of e A C ∧ is_perpendicularBisector_of e B D :=
by
  sorry

end exists_line_equidistant_from_AB_CD_l137_137309


namespace hexagon_coloring_l137_137986

def hex_colorings : ℕ := 2

theorem hexagon_coloring :
  ∃ c : ℕ, c = hex_colorings := by
  sorry

end hexagon_coloring_l137_137986


namespace molly_takes_180_minutes_longer_l137_137758

noncomputable def time_for_Xanthia (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

noncomputable def time_for_Molly (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

theorem molly_takes_180_minutes_longer (pages : ℕ) (Xanthia_speed : ℕ) (Molly_speed : ℕ) :
  (time_for_Molly Molly_speed pages - time_for_Xanthia Xanthia_speed pages) * 60 = 180 :=
by
  -- Definitions specific to problem conditions
  let pages := 360
  let Xanthia_speed := 120
  let Molly_speed := 60

  -- Placeholder for actual proof
  sorry

end molly_takes_180_minutes_longer_l137_137758


namespace football_game_attendance_l137_137498

theorem football_game_attendance :
  ∃ y : ℕ, (∃ x : ℕ, x + y = 280 ∧ 60 * x + 25 * y = 14000) ∧ y = 80 :=
by
  sorry

end football_game_attendance_l137_137498


namespace product_of_coefficients_is_negative_integer_l137_137237

theorem product_of_coefficients_is_negative_integer
  (a b c : ℤ)
  (habc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (discriminant_positive : (b * b - 4 * a * c) > 0)
  (product_cond : a * b * c = (c / a)) :
  ∃ k : ℤ, k < 0 ∧ k = a * b * c :=
by
  sorry

end product_of_coefficients_is_negative_integer_l137_137237


namespace balloon_height_l137_137781

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end balloon_height_l137_137781


namespace total_go_stones_correct_l137_137375

-- Definitions based on the problem's conditions
def stones_per_bundle : Nat := 10
def num_bundles : Nat := 3
def white_stones : Nat := 16

-- A function that calculates the total number of go stones
def total_go_stones : Nat :=
  num_bundles * stones_per_bundle + white_stones

-- The theorem we want to prove
theorem total_go_stones_correct : total_go_stones = 46 :=
by
  sorry

end total_go_stones_correct_l137_137375


namespace distance_from_p_to_center_is_2_sqrt_10_l137_137786

-- Define the conditions
def r : ℝ := 4
def PA : ℝ := 4
def PB : ℝ := 6

-- The conjecture to prove
theorem distance_from_p_to_center_is_2_sqrt_10
  (r : ℝ) (PA : ℝ) (PB : ℝ) 
  (PA_mul_PB : PA * PB = 24) 
  (r_squared : r = 4)  : 
  ∃ d : ℝ, d = 2 * Real.sqrt 10 := 
by sorry

end distance_from_p_to_center_is_2_sqrt_10_l137_137786


namespace parabola_focus_l137_137966

theorem parabola_focus (p : ℝ) (hp : p > 0) :
    ∀ (x y : ℝ), (x = 2 * p * y^2) ↔ (x, y) = (1 / (8 * p), 0) :=
by 
  sorry

end parabola_focus_l137_137966


namespace smallest_perfect_square_divisible_by_3_and_5_l137_137720

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end smallest_perfect_square_divisible_by_3_and_5_l137_137720


namespace circle_equation_proof_l137_137081

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define a predicate for the circle being tangent to the y-axis
def tangent_y_axis (center : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r = abs center.1

-- Define the equation of the circle given center and radius
def circle_eqn (center : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2

-- State the theorem
theorem circle_equation_proof :
  tangent_y_axis circle_center →
  ∃ r, r = 1 ∧ circle_eqn circle_center r :=
sorry

end circle_equation_proof_l137_137081


namespace evaluate_tensor_expression_l137_137128

-- Define the tensor operation
def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- The theorem we want to prove
theorem evaluate_tensor_expression : tensor (tensor 5 3) 2 = 293 / 15 := by
  sorry

end evaluate_tensor_expression_l137_137128


namespace value_of_fraction_l137_137827

theorem value_of_fraction (x y : ℤ) (h1 : x = 3) (h2 : y = 4) : (x^5 + 3 * y^3) / 9 = 48 :=
by
  sorry

end value_of_fraction_l137_137827


namespace simplify_to_x5_l137_137190

theorem simplify_to_x5 (x : ℝ) :
  x^2 * x^3 = x^5 :=
by {
  -- proof goes here
  sorry
}

end simplify_to_x5_l137_137190


namespace calculate_fraction_l137_137847

theorem calculate_fraction (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) : 
  (1 / (x - 1)) - (2 / (x^2 - 1)) = 1 / (x + 1) :=
by
  sorry

end calculate_fraction_l137_137847


namespace num_candidates_above_630_l137_137955

noncomputable def normal_distribution_candidates : Prop :=
  let μ := 530
  let σ := 50
  let total_candidates := 1000
  let probability_above_630 := (1 - 0.954) / 2  -- Probability of scoring above 630
  let expected_candidates_above_630 := total_candidates * probability_above_630
  expected_candidates_above_630 = 23

theorem num_candidates_above_630 : normal_distribution_candidates := by
  sorry

end num_candidates_above_630_l137_137955


namespace overall_percentage_of_favor_l137_137757

theorem overall_percentage_of_favor
    (n_starting : ℕ)
    (n_experienced : ℕ)
    (perc_starting_favor : ℝ)
    (perc_experienced_favor : ℝ)
    (in_favor_from_starting : ℕ)
    (in_favor_from_experienced : ℕ)
    (total_surveyed : ℕ)
    (total_in_favor : ℕ)
    (overall_percentage : ℝ) :
    n_starting = 300 →
    n_experienced = 500 →
    perc_starting_favor = 0.40 →
    perc_experienced_favor = 0.70 →
    in_favor_from_starting = 120 →
    in_favor_from_experienced = 350 →
    total_surveyed = 800 →
    total_in_favor = 470 →
    overall_percentage = (470 / 800) * 100 →
    overall_percentage = 58.75 :=
by
  sorry

end overall_percentage_of_favor_l137_137757


namespace principal_amount_borrowed_l137_137926

theorem principal_amount_borrowed 
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 9000) 
  (h2 : R = 0.12) 
  (h3 : T = 3) 
  (h4 : SI = P * R * T) : 
  P = 25000 :=
sorry

end principal_amount_borrowed_l137_137926


namespace mixed_solution_concentration_l137_137614

-- Defining the conditions as given in the question
def weight1 : ℕ := 200
def concentration1 : ℕ := 25
def saltInFirstSolution : ℕ := (concentration1 * weight1) / 100

def weight2 : ℕ := 300
def saltInSecondSolution : ℕ := 60

def totalSalt : ℕ := saltInFirstSolution + saltInSecondSolution
def totalWeight : ℕ := weight1 + weight2

-- Statement of the proof
theorem mixed_solution_concentration :
  ((totalSalt : ℚ) / (totalWeight : ℚ)) * 100 = 22 :=
by
  sorry

end mixed_solution_concentration_l137_137614


namespace find_large_monkey_doll_cost_l137_137259

-- Define the conditions and the target property
def large_monkey_doll_cost (L : ℝ) (condition1 : 300 / (L - 2) = 300 / L + 25)
                           (condition2 : 300 / (L + 1) = 300 / L - 15) : Prop :=
  L = 6

-- The main theorem with the conditions
theorem find_large_monkey_doll_cost (L : ℝ)
  (h1 : 300 / (L - 2) = 300 / L + 25)
  (h2 : 300 / (L + 1) = 300 / L - 15) : large_monkey_doll_cost L h1 h2 :=
  sorry

end find_large_monkey_doll_cost_l137_137259


namespace edward_friend_scores_l137_137515

theorem edward_friend_scores (total_points friend_points edward_points : ℕ) (h1 : total_points = 13) (h2 : edward_points = 7) (h3 : friend_points = total_points - edward_points) : friend_points = 6 := 
by
  rw [h1, h2] at h3
  exact h3

end edward_friend_scores_l137_137515


namespace andy_last_problem_l137_137365

theorem andy_last_problem (s t : ℕ) (start : s = 75) (total : t = 51) : (s + t - 1) = 125 :=
by
  sorry

end andy_last_problem_l137_137365


namespace contrapositive_l137_137951

theorem contrapositive (x : ℝ) (h : x^2 ≥ 1) : x ≥ 0 ∨ x ≤ -1 :=
sorry

end contrapositive_l137_137951


namespace find_number_and_remainder_l137_137068

theorem find_number_and_remainder :
  ∃ (N r : ℕ), (3927 + 2873) * (3 * (3927 - 2873)) + r = N ∧ r < (3927 + 2873) :=
sorry

end find_number_and_remainder_l137_137068


namespace exists_solution_interval_inequality_l137_137818

theorem exists_solution_interval_inequality :
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) := 
by
  sorry

end exists_solution_interval_inequality_l137_137818


namespace expression_evaluation_l137_137851

theorem expression_evaluation (a b c : ℤ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 :=
by
  have ha : a = 8 := h₁
  have hb : b = 10 := h₂
  have hc : c = 3 := h₃
  rw [ha, hb, hc]
  sorry

end expression_evaluation_l137_137851


namespace perpendicular_lines_a_eq_1_l137_137359

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end perpendicular_lines_a_eq_1_l137_137359


namespace cost_price_per_metre_l137_137635

theorem cost_price_per_metre (total_selling_price : ℕ) (total_metres : ℕ) (loss_per_metre : ℕ)
  (h1 : total_selling_price = 9000)
  (h2 : total_metres = 300)
  (h3 : loss_per_metre = 6) :
  (total_selling_price + (loss_per_metre * total_metres)) / total_metres = 36 :=
by
  sorry

end cost_price_per_metre_l137_137635


namespace part_one_part_two_l137_137490

variable {x : ℝ} {m : ℝ}

-- Question 1
theorem part_one (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) : -4 < m ∧ m <= 0 :=
sorry

-- Question 2
theorem part_two (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → mx^2 - mx - 1 > -m + x - 1) : m > 1 :=
sorry

end part_one_part_two_l137_137490


namespace gcd_n_cube_plus_16_n_plus_4_l137_137467

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l137_137467


namespace ratio_of_areas_l137_137580

theorem ratio_of_areas (aC aD : ℕ) (hC : aC = 48) (hD : aD = 60) : 
  (aC^2 : ℚ) / (aD^2 : ℚ) = (16 : ℚ) / (25 : ℚ) := 
by
  sorry

end ratio_of_areas_l137_137580


namespace find_middle_number_l137_137971

theorem find_middle_number
  (S1 S2 M : ℤ)
  (h1 : S1 = 6 * 5)
  (h2 : S2 = 6 * 7)
  (h3 : 13 * 9 = S1 + M + S2) :
  M = 45 :=
by
  -- proof steps would go here
  sorry

end find_middle_number_l137_137971


namespace paint_cans_needed_l137_137901

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l137_137901


namespace investment_in_stocks_l137_137567

theorem investment_in_stocks (T b s : ℝ) (h1 : T = 200000) (h2 : s = 5 * b) (h3 : T = b + s) :
  s = 166666.65 :=
by sorry

end investment_in_stocks_l137_137567


namespace geometric_sequence_condition_l137_137155

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < a 1) 
  (h2 : ∀ n, a (n + 1) = a n * q) :
  (a 1 < a 3) ↔ (a 1 < a 3) ∧ (a 3 < a 6) :=
sorry

end geometric_sequence_condition_l137_137155


namespace value_of_g_g_2_l137_137037

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_g_g_2 : g (g 2) = 1447 := by
  sorry

end value_of_g_g_2_l137_137037


namespace quadratic_floor_eq_more_than_100_roots_l137_137983

open Int

theorem quadratic_floor_eq_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ (S : Finset ℤ), S.card > 100 ∧ ∀ x ∈ S, ⌊(x : ℝ) ^ 2⌋ + p * x + q = 0 :=
by
  sorry

end quadratic_floor_eq_more_than_100_roots_l137_137983


namespace pyramid_lateral_edge_ratio_l137_137727

variable (h x : ℝ)

-- We state the conditions as hypotheses
axiom pyramid_intersected_by_plane_parallel_to_base (h : ℝ) (S S' : ℝ) :
  S' = S / 2 → (S' / S = (x / h) ^ 2) → (x = h / Real.sqrt 2)

-- The theorem we need to prove
theorem pyramid_lateral_edge_ratio (h x : ℝ) (S S' : ℝ)
  (cond1 : S' = S / 2)
  (cond2 : S' / S = (x / h) ^ 2) :
  x / h = 1 / Real.sqrt 2 :=
by
  -- skip the proof
  sorry

end pyramid_lateral_edge_ratio_l137_137727


namespace original_number_of_men_l137_137195

-- Define the conditions
def work_days_by_men (M : ℕ) (days : ℕ) : ℕ := M * days
def additional_men (M : ℕ) : ℕ := M + 10
def completed_days : ℕ := 9

-- The main theorem
theorem original_number_of_men : ∀ (M : ℕ), 
  work_days_by_men M 12 = work_days_by_men (additional_men M) completed_days → 
  M = 30 :=
by
  intros M h
  sorry

end original_number_of_men_l137_137195


namespace smallest_angle_in_convex_polygon_l137_137415

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : ℕ → ℕ) (d : ℕ), n = 25 → (∀ i, 1 ≤ i ∧ i ≤ n → angles i = 166 - 1 * (13 - i)) 
  → 1 ≤ d ∧ d ≤ 1 → (angles 1 = 154) := 
by
  sorry

end smallest_angle_in_convex_polygon_l137_137415


namespace original_number_eq_nine_l137_137505

theorem original_number_eq_nine (N : ℕ) (h1 : ∃ k : ℤ, N - 4 = 5 * k) : N = 9 :=
sorry

end original_number_eq_nine_l137_137505


namespace park_area_is_correct_l137_137123

-- Define the side of the square
def side_length : ℕ := 30

-- Define the area function for a square
def area_of_square (side: ℕ) : ℕ := side * side

-- State the theorem we're going to prove
theorem park_area_is_correct : area_of_square side_length = 900 := 
sorry -- proof not required

end park_area_is_correct_l137_137123


namespace intersection_of_A_and_B_l137_137779

-- Define sets A and B
def setA : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x < 2}

-- Prove that A ∩ B = (-3, 2)
theorem intersection_of_A_and_B : {x : ℝ | x ∈ setA ∧ x ∈ setB} = {x : ℝ | -3 < x ∧ x < 2} := 
by 
  sorry

end intersection_of_A_and_B_l137_137779


namespace organization_population_after_six_years_l137_137579

theorem organization_population_after_six_years :
  ∀ (b : ℕ → ℕ),
  (b 0 = 20) →
  (∀ k, b (k + 1) = 3 * (b k - 5) + 5) →
  b 6 = 10895 :=
by
  intros b h0 hr
  sorry

end organization_population_after_six_years_l137_137579


namespace volume_of_given_cuboid_l137_137135

-- Definition of the function to compute the volume of a cuboid
def volume_of_cuboid (length width height : ℝ) : ℝ :=
  length * width * height

-- Given conditions and the proof target
theorem volume_of_given_cuboid : volume_of_cuboid 2 5 3 = 30 :=
by
  sorry

end volume_of_given_cuboid_l137_137135


namespace find_number_l137_137468

theorem find_number (x : ℕ) (h : x + 20 + x + 30 + x + 40 + x + 10 = 4100) : x = 1000 := 
by
  sorry

end find_number_l137_137468


namespace type_B_ratio_l137_137842

theorem type_B_ratio
    (num_A : ℕ)
    (total_bricks : ℕ)
    (other_bricks : ℕ)
    (h1 : num_A = 40)
    (h2 : total_bricks = 150)
    (h3 : other_bricks = 90) :
    (total_bricks - num_A - other_bricks) / num_A = 1 / 2 :=
by
  sorry

end type_B_ratio_l137_137842


namespace find_equation_of_line_l137_137184

variable (x y : ℝ)

def line_parallel (x y : ℝ) (m : ℝ) :=
  x - 2*y + m = 0

def line_through_point (x y : ℝ) (px py : ℝ) (m : ℝ) :=
  (px - 2 * py + m = 0)
  
theorem find_equation_of_line :
  let px := -1
  let py := 3
  ∃ m, line_parallel x y m ∧ line_through_point x y px py m ∧ m = 7 :=
by
  sorry

end find_equation_of_line_l137_137184


namespace jay_used_zero_fraction_of_gallon_of_paint_l137_137202

theorem jay_used_zero_fraction_of_gallon_of_paint
    (dexter_used : ℝ := 3/8)
    (gallon_in_liters : ℝ := 4)
    (paint_left_liters : ℝ := 4) :
    dexter_used = 3/8 ∧ gallon_in_liters = 4 ∧ paint_left_liters = 4 →
    ∃ jay_used : ℝ, jay_used = 0 :=
by
  sorry

end jay_used_zero_fraction_of_gallon_of_paint_l137_137202


namespace ab_squared_ab_cubed_ab_power_n_l137_137264

-- Definitions of a and b as real numbers, and n as a natural number
variables (a b : ℝ) (n : ℕ)

theorem ab_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by 
  sorry

theorem ab_cubed (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by 
  sorry

theorem ab_power_n (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by 
  sorry

end ab_squared_ab_cubed_ab_power_n_l137_137264


namespace probability_both_tell_truth_l137_137919

theorem probability_both_tell_truth (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.60) : pA * pB = 0.48 :=
by
  subst hA
  subst hB
  sorry

end probability_both_tell_truth_l137_137919


namespace remaining_volume_correct_l137_137683

noncomputable def diameter_sphere : ℝ := 24
noncomputable def radius_sphere : ℝ := diameter_sphere / 2
noncomputable def height_hole1 : ℝ := 10
noncomputable def diameter_hole1 : ℝ := 3
noncomputable def radius_hole1 : ℝ := diameter_hole1 / 2
noncomputable def height_hole2 : ℝ := 10
noncomputable def diameter_hole2 : ℝ := 3
noncomputable def radius_hole2 : ℝ := diameter_hole2 / 2
noncomputable def height_hole3 : ℝ := 5
noncomputable def diameter_hole3 : ℝ := 4
noncomputable def radius_hole3 : ℝ := diameter_hole3 / 2

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius_sphere ^ 3)
noncomputable def volume_hole1 : ℝ := Real.pi * (radius_hole1 ^ 2) * height_hole1
noncomputable def volume_hole2 : ℝ := Real.pi * (radius_hole2 ^ 2) * height_hole2
noncomputable def volume_hole3 : ℝ := Real.pi * (radius_hole3 ^ 2) * height_hole3

noncomputable def remaining_volume : ℝ := 
  volume_sphere - (2 * volume_hole1 + volume_hole3)

theorem remaining_volume_correct : remaining_volume = 2239 * Real.pi := by
  sorry

end remaining_volume_correct_l137_137683


namespace infinite_solutions_l137_137168

theorem infinite_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (k : ℕ), x = k^3 + 1 ∧ y = (k^3 + 1) * k := 
sorry

end infinite_solutions_l137_137168


namespace find_N_l137_137943

theorem find_N (a b c N : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = a + b) : N = 272 :=
sorry

end find_N_l137_137943


namespace tan_alpha_over_tan_beta_l137_137922

theorem tan_alpha_over_tan_beta (α β : ℝ) (h1 : Real.sin (α + β) = 2 / 3) (h2 : Real.sin (α - β) = 1 / 3) :
  (Real.tan α / Real.tan β = 3) :=
sorry

end tan_alpha_over_tan_beta_l137_137922


namespace camden_dogs_fraction_l137_137024

def number_of_dogs (Justins_dogs : ℕ) (extra_dogs : ℕ) : ℕ := Justins_dogs + extra_dogs
def dogs_from_legs (total_legs : ℕ) (legs_per_dog : ℕ) : ℕ := total_legs / legs_per_dog
def fraction_of_dogs (dogs_camden : ℕ) (dogs_rico : ℕ) : ℚ := dogs_camden / dogs_rico

theorem camden_dogs_fraction (Justins_dogs : ℕ) (extra_dogs : ℕ) (total_legs_camden : ℕ) (legs_per_dog : ℕ) :
  Justins_dogs = 14 →
  extra_dogs = 10 →
  total_legs_camden = 72 →
  legs_per_dog = 4 →
  fraction_of_dogs (dogs_from_legs total_legs_camden legs_per_dog) (number_of_dogs Justins_dogs extra_dogs) = 3 / 4 :=
by
  sorry

end camden_dogs_fraction_l137_137024


namespace solve_for_b_l137_137527

theorem solve_for_b (a b c m : ℚ) (h : m = c * a * b / (a - b)) : b = (m * a) / (m + c * a) :=
by
  sorry

end solve_for_b_l137_137527


namespace find_first_term_l137_137613

variable {a : ℕ → ℕ}

-- Given conditions
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) + a n = 4 * n

-- Question to prove
theorem find_first_term : a 0 = 1 :=
sorry

end find_first_term_l137_137613


namespace triangle_side_count_l137_137305

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l137_137305


namespace int_values_satisfy_condition_l137_137041

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l137_137041


namespace max_neg_expr_l137_137285

theorem max_neg_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (- (1 / (2 * a)) - (2 / b)) ≤ - (9 / 2) :=
sorry

end max_neg_expr_l137_137285


namespace parabola_symmetric_points_l137_137674

-- Define the parabola and the symmetry condition
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

def symmetric_points (P Q : ℝ × ℝ) : Prop :=
  P.1 + P.2 = 0 ∧ Q.1 + Q.2 = 0 ∧ P.1 = -Q.1 ∧ P.2 = -Q.2

-- Problem definition: Prove that if there exist symmetric points on the parabola, then a > 3/4
theorem parabola_symmetric_points (a : ℝ) :
  (∃ P Q : ℝ × ℝ, symmetric_points P Q ∧ parabola a P.1 = P.2 ∧ parabola a Q.1 = Q.2) → a > 3 / 4 :=
by
  sorry

end parabola_symmetric_points_l137_137674


namespace ice_cream_total_sum_l137_137759

noncomputable def totalIceCream (friday saturday sunday monday tuesday : ℝ) : ℝ :=
  friday + saturday + sunday + monday + tuesday

theorem ice_cream_total_sum : 
  let friday := 3.25
  let saturday := 2.5
  let sunday := 1.75
  let monday := 0.5
  let tuesday := 2 * monday
  totalIceCream friday saturday sunday monday tuesday = 9 := by
    sorry

end ice_cream_total_sum_l137_137759


namespace area_of_triangle_l137_137376

theorem area_of_triangle 
  (h : ∀ x y : ℝ, (x / 5 + y / 2 = 1) → ((x = 5 ∧ y = 0) ∨ (x = 0 ∧ y = 2))) : 
  ∃ t : ℝ, t = 1 / 2 * 2 * 5 := 
sorry

end area_of_triangle_l137_137376


namespace simplify_expression_l137_137492

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l137_137492


namespace union_sets_l137_137029

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} :=
by
  sorry

end union_sets_l137_137029


namespace slope_of_line_l137_137252

theorem slope_of_line (a : ℝ) (h : a = (Real.tan (Real.pi / 3))) : a = Real.sqrt 3 := by
sorry

end slope_of_line_l137_137252


namespace barry_wand_trick_l137_137461

theorem barry_wand_trick (n : ℕ) (h : (n + 3 : ℝ) / 3 = 50) : n = 147 := by
  sorry

end barry_wand_trick_l137_137461


namespace line_does_not_pass_second_quadrant_l137_137952

theorem line_does_not_pass_second_quadrant (a : ℝ) (ha : a ≠ 0) :
  ∀ (x y : ℝ), (x - y - a^2 = 0) → ¬(x < 0 ∧ y > 0) :=
sorry

end line_does_not_pass_second_quadrant_l137_137952


namespace min_value_c_l137_137456

-- Define the problem using Lean
theorem min_value_c 
    (a b c d e : ℕ)
    (h1 : a + 1 = b) 
    (h2 : b + 1 = c)
    (h3 : c + 1 = d)
    (h4 : d + 1 = e)
    (h5 : ∃ n : ℕ, 5 * c = n ^ 3)
    (h6 : ∃ m : ℕ, 3 * c = m ^ 2) : 
    c = 675 := 
sorry

end min_value_c_l137_137456


namespace cos_2beta_proof_l137_137373

theorem cos_2beta_proof (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (3 * π / 2) (2 * π)) :
  Real.cos (2 * β) = -7 / 25 :=
by
  sorry

end cos_2beta_proof_l137_137373


namespace weight_of_new_person_l137_137514

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight new_weight : ℝ) 
  (h_avg_increase : avg_increase = 1.5) (h_num_persons : num_persons = 9) (h_old_weight : old_weight = 65) 
  (h_new_weight_increase : new_weight = old_weight + num_persons * avg_increase) : 
  new_weight = 78.5 :=
sorry

end weight_of_new_person_l137_137514


namespace inequality_1_inequality_2_l137_137836

theorem inequality_1 (x : ℝ) : (2 * x^2 - 3 * x + 1 < 0) ↔ (1 / 2 < x ∧ x < 1) := 
by sorry

theorem inequality_2 (x : ℝ) (h : x ≠ -1) : (2 * x / (x + 1) ≥ 1) ↔ (x < -1 ∨ x ≥ 1) := 
by sorry

end inequality_1_inequality_2_l137_137836


namespace repeating_decimal_as_fraction_l137_137772

theorem repeating_decimal_as_fraction :
  ∃ x : ℚ, x = 6 / 10 + 7 / 90 ∧ x = 61 / 90 :=
by
  sorry

end repeating_decimal_as_fraction_l137_137772


namespace rectangle_horizontal_length_l137_137480

theorem rectangle_horizontal_length (s v : ℕ) (h : ℕ) 
  (hs : s = 80) (hv : v = 100) 
  (eq_perimeters : 4 * s = 2 * (v + h)) : h = 60 :=
by
  sorry

end rectangle_horizontal_length_l137_137480


namespace total_worth_of_stock_l137_137839

theorem total_worth_of_stock (X : ℝ) (h1 : 0.1 * X * 1.2 - 0.9 * X * 0.95 = -400) : X = 16000 :=
by
  -- actual proof
  sorry

end total_worth_of_stock_l137_137839


namespace evaluate_expr_l137_137897

theorem evaluate_expr :
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -1 / 6 * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7 / 3 := by
  sorry

end evaluate_expr_l137_137897


namespace find_term_number_l137_137586

theorem find_term_number :
  ∃ n : ℕ, (2 * (5 : ℝ)^(1/2) = (3 * (n : ℝ) - 1)^(1/2)) ∧ n = 7 :=
sorry

end find_term_number_l137_137586


namespace suraj_average_increase_l137_137383

theorem suraj_average_increase
  (A : ℝ)
  (h1 : 9 * A + 200 = 10 * 128) :
  128 - A = 8 :=
by
  sorry

end suraj_average_increase_l137_137383


namespace balls_into_boxes_l137_137925

theorem balls_into_boxes : (4 ^ 5 = 1024) :=
by
  -- The proof is omitted; the statement is required
  sorry

end balls_into_boxes_l137_137925


namespace find_abc_l137_137583

def rearrangements (a b c : ℕ) : List ℕ :=
  [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
   100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]

theorem find_abc (a b c : ℕ) (habc : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (rearrangements a b c).sum = 2017 + habc →
  habc = 425 :=
by
  sorry

end find_abc_l137_137583


namespace joyful_not_blue_l137_137824

variables {Snakes : Type} 
variables (isJoyful : Snakes → Prop) (isBlue : Snakes → Prop)
variables (canMultiply : Snakes → Prop) (canDivide : Snakes → Prop)

-- Conditions
axiom H1 : ∀ s : Snakes, isJoyful s → canMultiply s
axiom H2 : ∀ s : Snakes, isBlue s → ¬ canDivide s
axiom H3 : ∀ s : Snakes, ¬ canDivide s → ¬ canMultiply s

theorem joyful_not_blue (s : Snakes) : isJoyful s → ¬ isBlue s :=
by sorry

end joyful_not_blue_l137_137824


namespace decorate_eggs_time_calculation_l137_137449

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l137_137449


namespace red_tint_percentage_new_mixture_l137_137358

-- Definitions of the initial conditions
def original_volume : ℝ := 50
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℝ := 6

-- Definition for the proof
theorem red_tint_percentage_new_mixture : 
  let original_red_tint := red_tint_percentage * original_volume
  let new_red_tint := original_red_tint + added_red_tint
  let new_total_volume := original_volume + added_red_tint
  (new_red_tint / new_total_volume) * 100 = 28.57 :=
by
  sorry

end red_tint_percentage_new_mixture_l137_137358


namespace arithmetic_series_sum_l137_137101

def first_term (k : ℕ) : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def number_of_terms (k : ℕ) : ℕ := 2 * k + 3
def nth_term (k n : ℕ) : ℕ := (first_term k) + (n - 1) * common_difference
def sum_of_terms (k : ℕ) : ℕ :=
  let n := number_of_terms k
  let a := first_term k
  let l := nth_term k n
  n * (a + l) / 2

theorem arithmetic_series_sum (k : ℕ) : sum_of_terms k = 2 * k^3 + 7 * k^2 + 10 * k + 6 :=
sorry

end arithmetic_series_sum_l137_137101


namespace range_of_a_l137_137825

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l137_137825


namespace sam_final_investment_l137_137959

-- Definitions based on conditions
def initial_investment : ℝ := 10000
def first_interest_rate : ℝ := 0.20
def years_first_period : ℕ := 3
def triple_amount : ℕ := 3
def second_interest_rate : ℝ := 0.15
def years_second_period : ℕ := 1

-- Lean function to accumulate investment with compound interest
def compound_interest (P r: ℝ) (n: ℕ) : ℝ := P * (1 + r) ^ n

-- Sam's investment calculations
def amount_after_3_years : ℝ := compound_interest initial_investment first_interest_rate years_first_period
def new_investment : ℝ := triple_amount * amount_after_3_years
def final_amount : ℝ := compound_interest new_investment second_interest_rate years_second_period

-- Proof goal (statement with the proof skipped)
theorem sam_final_investment : final_amount = 59616 := by
  sorry

end sam_final_investment_l137_137959


namespace certain_number_l137_137279

theorem certain_number (x : ℤ) (h : 12 + x = 27) : x = 15 :=
by
  sorry

end certain_number_l137_137279


namespace harry_pencils_lost_l137_137273

-- Define the conditions
def anna_pencils : ℕ := 50
def harry_initial_pencils : ℕ := 2 * anna_pencils
def harry_current_pencils : ℕ := 81

-- Define the proof statement
theorem harry_pencils_lost :
  harry_initial_pencils - harry_current_pencils = 19 :=
by
  -- The proof is to be filled in
  sorry

end harry_pencils_lost_l137_137273


namespace point_coordinates_l137_137105

theorem point_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : abs y = 5) (h4 : abs x = 2) : x = -2 ∧ y = 5 :=
by
  sorry

end point_coordinates_l137_137105


namespace mary_spent_on_jacket_l137_137768

def shirt_cost : ℝ := 13.04
def total_cost : ℝ := 25.31
def jacket_cost : ℝ := total_cost - shirt_cost

theorem mary_spent_on_jacket :
  jacket_cost = 12.27 := by
  sorry

end mary_spent_on_jacket_l137_137768


namespace sequence_value_G_50_l137_137443

theorem sequence_value_G_50 :
  ∀ G : ℕ → ℚ, (∀ n : ℕ, G (n + 1) = (3 * G n + 1) / 3) ∧ G 1 = 3 → G 50 = 152 / 3 :=
by
  intros
  sorry

end sequence_value_G_50_l137_137443


namespace volleyball_team_total_score_l137_137293

-- Define the conditions
def LizzieScore := 4
def NathalieScore := LizzieScore + 3
def CombinedLizzieNathalieScore := LizzieScore + NathalieScore
def AimeeScore := 2 * CombinedLizzieNathalieScore
def TeammatesScore := 17

-- Prove that the total team score is 50
theorem volleyball_team_total_score :
  LizzieScore + NathalieScore + AimeeScore + TeammatesScore = 50 :=
by
  sorry

end volleyball_team_total_score_l137_137293


namespace problem_sum_of_k_l137_137143

theorem problem_sum_of_k {a b c k : ℂ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_ratio : a / (1 - b) = k ∧ b / (1 - c) = k ∧ c / (1 - a) = k) :
  (if (k^2 - k + 1 = 0) then -(-1)/1 else 0) = 1 :=
sorry

end problem_sum_of_k_l137_137143


namespace resulting_surface_area_l137_137935

-- Defining the initial condition for the cube structure
def cube_surface_area (side_length : ℕ) : ℕ :=
  6 * side_length^2

-- Defining the structure and the modifications
def initial_structure : ℕ :=
  64 * (cube_surface_area 2)

def removed_cubes_exposure : ℕ :=
  4 * (cube_surface_area 2)

-- The final lean statement to prove the surface area after removing central cubes
theorem resulting_surface_area : initial_structure + removed_cubes_exposure = 1632 := by
  sorry

end resulting_surface_area_l137_137935


namespace painters_work_days_l137_137006

noncomputable def work_product (n : ℕ) (d : ℚ) : ℚ := n * d

theorem painters_work_days :
  (work_product 5 2 = work_product 4 (2 + 1/2)) :=
by
  sorry

end painters_work_days_l137_137006


namespace area_excluding_holes_l137_137625

theorem area_excluding_holes (x : ℝ) :
  let A_large : ℝ := (x + 8) * (x + 6)
  let A_hole : ℝ := (2 * x - 4) * (x - 3)
  A_large - 2 * A_hole = -3 * x^2 + 34 * x + 24 := by
  sorry

end area_excluding_holes_l137_137625


namespace inclination_angle_of_line_l137_137385

theorem inclination_angle_of_line
  (α : ℝ) (h1 : α > 0) (h2 : α < 180)
  (hslope : Real.tan α = - (Real.sqrt 3) / 3) :
  α = 150 :=
sorry

end inclination_angle_of_line_l137_137385


namespace mirror_area_l137_137540

/-- The outer dimensions of the frame are given as 100 cm by 140 cm,
and the frame width is 15 cm. We aim to prove that the area of the mirror
inside the frame is 7700 cm². -/
theorem mirror_area (W H F: ℕ) (hW : W = 100) (hH : H = 140) (hF : F = 15) :
  (W - 2 * F) * (H - 2 * F) = 7700 :=
by
  sorry

end mirror_area_l137_137540


namespace negation_of_proposition_p_l137_137841

theorem negation_of_proposition_p :
  (¬(∃ x : ℝ, 0 < x ∧ Real.log x > x - 1)) ↔ (∀ x : ℝ, 0 < x → Real.log x ≤ x - 1) :=
by
  sorry

end negation_of_proposition_p_l137_137841


namespace age_difference_l137_137597

theorem age_difference (sum_ages : ℕ) (eldest_age : ℕ) (age_diff : ℕ) 
(h1 : sum_ages = 50) (h2 : eldest_age = 14) :
  14 + (14 - age_diff) + (14 - 2 * age_diff) + (14 - 3 * age_diff) + (14 - 4 * age_diff) = 50 → age_diff = 2 := 
by
  intro h
  sorry

end age_difference_l137_137597


namespace product_of_possible_values_of_N_l137_137711

theorem product_of_possible_values_of_N (N B D : ℤ) 
  (h1 : B = D - N) 
  (h2 : B + 10 - (D - 4) = 1 ∨ B + 10 - (D - 4) = -1) :
  N = 13 ∨ N = 15 → (13 * 15) = 195 :=
by sorry

end product_of_possible_values_of_N_l137_137711


namespace isosceles_base_angle_eq_43_l137_137464

theorem isosceles_base_angle_eq_43 (α β : ℝ) (h_iso : α = β) (h_sum : α + β + 94 = 180) : α = 43 :=
by
  sorry

end isosceles_base_angle_eq_43_l137_137464


namespace units_digit_l137_137442

noncomputable def C := 20 + Real.sqrt 153
noncomputable def D := 20 - Real.sqrt 153

theorem units_digit (h : ∀ n ≥ 1, 20 ^ n % 10 = 0) :
  (C ^ 12 + D ^ 12) % 10 = 0 :=
by
  -- Proof will be provided based on the outlined solution
  sorry

end units_digit_l137_137442


namespace decagon_area_bisection_ratio_l137_137422

theorem decagon_area_bisection_ratio
  (decagon_area : ℝ := 12)
  (below_PQ_area : ℝ := 6)
  (trapezoid_area : ℝ := 4)
  (b1 : ℝ := 3)
  (b2 : ℝ := 6)
  (h : ℝ := 8/9)
  (XQ : ℝ := 4)
  (QY : ℝ := 2) :
  (XQ / QY = 2) :=
by
  sorry

end decagon_area_bisection_ratio_l137_137422


namespace intersection_of_function_and_inverse_l137_137886

theorem intersection_of_function_and_inverse (b a : Int) 
  (h₁ : a = 2 * (-4) + b) 
  (h₂ : a = (-4 - b) / 2) 
  : a = -4 :=
by
  sorry

end intersection_of_function_and_inverse_l137_137886


namespace solve_eq1_solve_eq2_l137_137103

theorem solve_eq1 (x : ℝ) : 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : x^2 + 4 * x + 3 = 0 ↔ x = -3 ∨ x = -1 :=
by
  sorry

end solve_eq1_solve_eq2_l137_137103


namespace possible_values_of_r_l137_137278

noncomputable def r : ℝ := sorry

def is_four_place_decimal (x : ℝ) : Prop := 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ x = a / 10 + b / 100 + c / 1000 + d / 10000

def is_closest_fraction (x : ℝ) : Prop := 
  abs (x - 3 / 11) < abs (x - 3 / 10) ∧ abs (x - 3 / 11) < abs (x - 1 / 4)

theorem possible_values_of_r :
  (0.2614 <= r ∧ r <= 0.2864) ∧ is_four_place_decimal r ∧ is_closest_fraction r →
  ∃ n : ℕ, n = 251 := 
sorry

end possible_values_of_r_l137_137278


namespace find_a12_a12_value_l137_137004

variable (a : ℕ → ℝ)

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

axiom h1 : a 6 + a 10 = 16
axiom h2 : a 4 = 1

-- Theorem to prove
theorem find_a12 : a 6 + a 10 = a 4 + a 12 := by
  -- Place for the proof
  sorry

theorem a12_value : (∃ a12, a 6 + a 10 = 16 ∧ a 4 = 1 ∧ a 6 + a 10 = a 4 + a12) → a 12 = 15 :=
by
  -- Place for the proof
  sorry

end find_a12_a12_value_l137_137004


namespace base_of_log_is_176_l137_137872

theorem base_of_log_is_176 
    (x : ℕ)
    (h : ∃ q r : ℕ, x = 19 * q + r ∧ q = 9 ∧ r = 5) :
    x = 176 :=
by
  sorry

end base_of_log_is_176_l137_137872


namespace set_different_l137_137616

-- Definitions of the sets ①, ②, ③, and ④
def set1 : Set ℤ := {x | x = 1}
def set2 : Set ℤ := {y | (y - 1)^2 = 0}
def set3 : Set ℤ := {x | x = 1}
def set4 : Set ℤ := {1}

-- Lean statement to prove that set3 is different from the others
theorem set_different : set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  -- Skipping the proof with sorry
  sorry

end set_different_l137_137616


namespace union_of_A_and_B_l137_137900

variable (a b : ℕ)

def A : Set ℕ := {3, 2^a}
def B : Set ℕ := {a, b}
def intersection_condition : A a ∩ B a b = {2} := by sorry

theorem union_of_A_and_B (h : A a ∩ B a b = {2}) : 
  A a ∪ B a b = {1, 2, 3} := by sorry

end union_of_A_and_B_l137_137900


namespace sundae_cost_l137_137617

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end sundae_cost_l137_137617


namespace ten_more_than_twice_number_of_birds_l137_137075

def number_of_birds : ℕ := 20

theorem ten_more_than_twice_number_of_birds :
  10 + 2 * number_of_birds = 50 :=
by
  sorry

end ten_more_than_twice_number_of_birds_l137_137075


namespace train_crossing_time_l137_137533

/-!
## Problem Statement
A train 400 m in length crosses a telegraph post. The speed of the train is 90 km/h. Prove that it takes 16 seconds for the train to cross the telegraph post.
-/

-- Defining the given definitions based on the conditions in a)
def train_length : ℕ := 400
def train_speed_kmh : ℕ := 90
def train_speed_ms : ℚ := 25 -- Converting 90 km/h to 25 m/s

-- Proving the problem statement
theorem train_crossing_time : train_length / train_speed_ms = 16 := 
by
  -- convert conditions and show expected result
  sorry

end train_crossing_time_l137_137533


namespace borgnine_tarantulas_needed_l137_137215

def total_legs_goal : ℕ := 1100
def chimp_legs : ℕ := 12 * 4
def lion_legs : ℕ := 8 * 4
def lizard_legs : ℕ := 5 * 4
def tarantula_legs : ℕ := 8

theorem borgnine_tarantulas_needed : 
  let total_legs_seen := chimp_legs + lion_legs + lizard_legs
  let legs_needed := total_legs_goal - total_legs_seen
  let num_tarantulas := legs_needed / tarantula_legs
  num_tarantulas = 125 := 
by
  sorry

end borgnine_tarantulas_needed_l137_137215


namespace fifteenth_term_l137_137245

noncomputable def seq : ℕ → ℝ
| 0       => 3
| 1       => 4
| (n + 2) => 12 / seq (n + 1)

theorem fifteenth_term :
  seq 14 = 3 :=
sorry

end fifteenth_term_l137_137245


namespace difference_of_numbers_l137_137405

-- Definitions for the digits and the numbers formed
def digits : List ℕ := [5, 3, 1, 4]

def largestNumber : ℕ := 5431
def leastNumber : ℕ := 1345

-- The problem statement
theorem difference_of_numbers (digits : List ℕ) (n_largest n_least : ℕ) :
  n_largest = 5431 ∧ n_least = 1345 → (n_largest - n_least) = 4086 :=
by
  sorry

end difference_of_numbers_l137_137405


namespace not_an_algorithm_option_B_l137_137819

def is_algorithm (description : String) : Prop :=
  description = "clear and finite steps to solve a problem producing correct results when executed by a computer"

def operation_to_string (option : Char) : String :=
  match option with
  | 'A' => "Calculating the area of a circle given its radius"
  | 'B' => "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
  | 'C' => "Finding the equation of a line given two points in the coordinate plane"
  | 'D' => "The rules of addition, subtraction, multiplication, and division"
  | _ => ""

noncomputable def categorize_operation (option : Char) : Prop :=
  option = 'B' ↔ ¬ is_algorithm (operation_to_string option)

theorem not_an_algorithm_option_B :
  categorize_operation 'B' :=
by
  sorry

end not_an_algorithm_option_B_l137_137819


namespace point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l137_137353

-- Define the travel records and the fuel consumption rate
def travel_records : List Int := [18, -9, 7, -14, -6, 13, -6, -8]
def fuel_consumption_rate : Float := 0.4

-- Question 1: Proof that point B is 5 km south of point A
theorem point_B_is_south_of_A : (travel_records.sum = -5) :=
  by sorry

-- Question 2: Proof that total distance traveled is 81 km
theorem total_distance_traveled : (travel_records.map Int.natAbs).sum = 81 :=
  by sorry

-- Question 3: Proof that the fuel consumed is 32 liters (Rounded)
theorem fuel_consumed : Float.floor (81 * fuel_consumption_rate) = 32 :=
  by sorry

end point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l137_137353


namespace tangent_line_at_P_no_zero_points_sum_of_zero_points_l137_137940

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

/-- Given that f(x) = ln(x) - 2x, prove that the tangent line at point P(1, -2) has the equation x + y + 1 = 0. -/
theorem tangent_line_at_P (a : ℝ) (h : a = 2) : ∀ x y : ℝ, x + y + 1 = 0 :=
sorry

/-- Show that for f(x) = ln(x) - ax, the function f(x) has no zero points if a > 1/e. -/
theorem no_zero_points (a : ℝ) (h : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f x a = 0 :=
sorry

/-- For f(x) = ln(x) - ax and x1 ≠ x2 such that f(x1) = f(x2) = 0, prove that x1 + x2 > 2 / a. -/
theorem sum_of_zero_points (a x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) : x₁ + x₂ > 2 / a :=
sorry

end tangent_line_at_P_no_zero_points_sum_of_zero_points_l137_137940


namespace total_nephews_correct_l137_137511

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end total_nephews_correct_l137_137511


namespace log_expression_identity_l137_137291

theorem log_expression_identity :
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 1 :=
by
  sorry

end log_expression_identity_l137_137291


namespace find_K_find_t_l137_137307

-- Proof Problem for G9.2
theorem find_K (x : ℚ) (K : ℚ) (h1 : x = 1.9898989) (h2 : x - 1 = K / 99) : K = 98 :=
sorry

-- Proof Problem for G9.3
theorem find_t (p q r t : ℚ)
  (h_avg1 : (p + q + r) / 3 = 18)
  (h_avg2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : t = 20 :=
sorry

end find_K_find_t_l137_137307


namespace ratio_of_probabilities_l137_137790

-- Define the total number of balls and bins
def balls : ℕ := 20
def bins : ℕ := 6

-- Define the sets A and B based on the given conditions
def A : ℕ := Nat.choose bins 1 * Nat.choose (bins - 1) 1 * (Nat.factorial balls / (Nat.factorial 2 * Nat.factorial 5 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def B : ℕ := Nat.choose bins 2 * (Nat.factorial balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))

-- Define the probabilities p and q
def p : ℚ := A / (Nat.factorial balls * Nat.factorial bins)
def q : ℚ := B / (Nat.factorial balls * Nat.factorial bins)

-- Prove the ratio of probabilities p and q equals 2
theorem ratio_of_probabilities : p / q = 2 := by sorry

end ratio_of_probabilities_l137_137790


namespace increasing_function_l137_137737

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Ici (1 : ℝ)) := by
  sorry

end increasing_function_l137_137737


namespace down_payment_calculation_l137_137678

noncomputable def tablet_price : ℝ := 450
noncomputable def installment_1 : ℝ := 4 * 40
noncomputable def installment_2 : ℝ := 4 * 35
noncomputable def installment_3 : ℝ := 4 * 30
noncomputable def total_savings : ℝ := 70
noncomputable def total_installments := tablet_price + total_savings
noncomputable def installment_payments := installment_1 + installment_2 + installment_3
noncomputable def down_payment := total_installments - installment_payments

theorem down_payment_calculation : down_payment = 100 := by
  unfold down_payment
  unfold total_installments
  unfold installment_payments
  unfold tablet_price
  unfold total_savings
  unfold installment_1
  unfold installment_2
  unfold installment_3
  sorry

end down_payment_calculation_l137_137678


namespace line_through_two_points_l137_137785

theorem line_through_two_points (P Q : ℝ × ℝ) (hP : P = (2, 5)) (hQ : Q = (2, -5)) :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q → x = 2) :=
by
  sorry

end line_through_two_points_l137_137785


namespace side_length_of_S2_l137_137712

theorem side_length_of_S2 (r s : ℝ) 
  (h1 : 2 * r + s = 2025) 
  (h2 : 2 * r + 3 * s = 3320) :
  s = 647.5 :=
by {
  -- proof omitted
  sorry
}

end side_length_of_S2_l137_137712


namespace volleyball_match_probabilities_l137_137126

noncomputable def probability_of_team_A_winning : ℚ := (2 / 3) ^ 3
noncomputable def probability_of_team_B_winning_3_0 : ℚ := 1 / 3
noncomputable def probability_of_team_B_winning_3_1 : ℚ := (2 / 3) * (1 / 3)
noncomputable def probability_of_team_B_winning_3_2 : ℚ := (2 / 3) ^ 2 * (1 / 3)

theorem volleyball_match_probabilities :
  probability_of_team_A_winning = 8 / 27 ∧
  probability_of_team_B_winning_3_0 = 1 / 3 ∧
  probability_of_team_B_winning_3_1 ≠ 1 / 9 ∧
  probability_of_team_B_winning_3_2 ≠ 4 / 9 :=
by
  sorry

end volleyball_match_probabilities_l137_137126


namespace find_remainder_l137_137848

theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by
  sorry

end find_remainder_l137_137848


namespace fraction_subtraction_simplified_l137_137453

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l137_137453


namespace find_a_and_b_l137_137642

variable {x : ℝ}

/-- The problem statement: Given the function y = b + a * sin x (with a < 0), and the maximum value is -1, and the minimum value is -5,
    find the values of a and b. --/
theorem find_a_and_b (a b : ℝ) (h : a < 0) 
  (h1 : ∀ x, b + a * Real.sin x ≤ -1)
  (h2 : ∀ x, b + a * Real.sin x ≥ -5) : 
  a = -2 ∧ b = -3 := sorry

end find_a_and_b_l137_137642


namespace divide_plane_into_regions_l137_137334

theorem divide_plane_into_regions :
  (∀ (x y : ℝ), y = 3 * x ∨ y = x / 3) →
  ∃ (regions : ℕ), regions = 4 :=
by
  sorry

end divide_plane_into_regions_l137_137334


namespace female_democrats_l137_137362

/-
There are 810 male and female participants in a meeting.
Half of the female participants and one-quarter of the male participants are Democrats.
One-third of all the participants are Democrats.
Prove that the number of female Democrats is 135.
-/

theorem female_democrats (F M : ℕ) (h : F + M = 810)
  (female_democrats : F / 2 = F / 2)
  (male_democrats : M / 4 = M / 4)
  (total_democrats : (F / 2 + M / 4) = 810 / 3) : 
  F / 2 = 135 := by
  sorry

end female_democrats_l137_137362


namespace train_length_360_l137_137526

variable (time_to_cross : ℝ) (speed_of_train : ℝ)

theorem train_length_360 (h1 : time_to_cross = 12) (h2 : speed_of_train = 30) :
  speed_of_train * time_to_cross = 360 :=
by
  rw [h1, h2]
  norm_num

end train_length_360_l137_137526


namespace max_k_possible_l137_137928

-- Given the sequence formed by writing all three-digit numbers from 100 to 999 consecutively
def digits_sequence : List Nat := List.join (List.map (fun n => [n / 100, (n / 10) % 10, n % 10]) (List.range' 100 (999 - 100 + 1)))

-- Function to get a k-digit number from the sequence
def get_k_digit_number (seq : List Nat) (start k : Nat) : List Nat := seq.drop start |>.take k

-- Statement to prove the maximum k
theorem max_k_possible : ∃ k : Nat, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 5 = get_k_digit_number digits_sequence start2 5) ∧ (¬ ∃ k' > 5, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 k' = get_k_digit_number digits_sequence start2 k')) :=
sorry

end max_k_possible_l137_137928


namespace cost_of_shoes_is_150_l137_137911

def cost_sunglasses : ℕ := 50
def pairs_sunglasses : ℕ := 2
def cost_jeans : ℕ := 100

def cost_basketball_cards : ℕ := 25
def decks_basketball_cards : ℕ := 2

-- Define the total amount spent by Mary and Rose
def total_mary : ℕ := cost_sunglasses * pairs_sunglasses + cost_jeans
def cost_shoes (total_rose : ℕ) (cost_cards : ℕ) : ℕ := total_rose - cost_cards

theorem cost_of_shoes_is_150 (total_spent : ℕ) :
  total_spent = total_mary →
  cost_shoes total_spent (cost_basketball_cards * decks_basketball_cards) = 150 :=
by
  intro h
  sorry

end cost_of_shoes_is_150_l137_137911


namespace find_biology_marks_l137_137454

theorem find_biology_marks (english math physics chemistry : ℕ) (avg_marks : ℕ) (biology : ℕ)
  (h_english : english = 86) (h_math : math = 89) (h_physics : physics = 82)
  (h_chemistry : chemistry = 87) (h_avg_marks : avg_marks = 85) :
  (english + math + physics + chemistry + biology) = avg_marks * 5 →
  biology = 81 :=
by
  sorry

end find_biology_marks_l137_137454


namespace transformed_roots_equation_l137_137475

theorem transformed_roots_equation (α β : ℂ) (h1 : 3 * α^2 + 2 * α + 1 = 0) (h2 : 3 * β^2 + 2 * β + 1 = 0) :
  ∃ (y : ℂ), (y - (3 * α + 2)) * (y - (3 * β + 2)) = y^2 + 4 := 
sorry

end transformed_roots_equation_l137_137475


namespace find_function_that_satisfies_eq_l137_137040

theorem find_function_that_satisfies_eq :
  ∀ (f : ℕ → ℕ), (∀ (m n : ℕ), f (m + f n) = f (f m) + f n) → (∀ n : ℕ, f n = n) :=
by
  intro f
  intro h
  sorry

end find_function_that_satisfies_eq_l137_137040


namespace Mary_cut_10_roses_l137_137774

-- Defining the initial and final number of roses
def initial_roses := 6
def final_roses := 16

-- Calculating the number of roses cut by Mary
def roses_cut := final_roses - initial_roses

-- The proof problem: Prove that the number of roses cut is 10
theorem Mary_cut_10_roses : roses_cut = 10 := by
  sorry

end Mary_cut_10_roses_l137_137774


namespace total_viewing_time_amaya_l137_137089

/-- The total viewing time Amaya spent, including rewinding, was 170 minutes. -/
theorem total_viewing_time_amaya 
  (u1 u2 u3 u4 u5 r1 r2 r3 r4 : ℕ)
  (h1 : u1 = 35)
  (h2 : u2 = 45)
  (h3 : u3 = 25)
  (h4 : u4 = 15)
  (h5 : u5 = 20)
  (hr1 : r1 = 5)
  (hr2 : r2 = 7)
  (hr3 : r3 = 10)
  (hr4 : r4 = 8) :
  u1 + u2 + u3 + u4 + u5 + r1 + r2 + r3 + r4 = 170 :=
by
  sorry

end total_viewing_time_amaya_l137_137089


namespace fourth_month_sale_is_7200_l137_137280

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

end fourth_month_sale_is_7200_l137_137280


namespace part_a_1_part_a_2_l137_137637

noncomputable def P (x k : ℝ) := x^3 - k*x + 2

theorem part_a_1 (k : ℝ) (h : k = 5) : P 2 k = 0 :=
sorry

theorem part_a_2 {x : ℝ} : P x 5 = (x - 2) * (x^2 + 2*x - 1) :=
sorry

end part_a_1_part_a_2_l137_137637


namespace starting_number_of_sequence_l137_137286

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end starting_number_of_sequence_l137_137286


namespace constant_S13_l137_137631

theorem constant_S13 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a n = a 1 + (n - 1) * d) 
(h_sum : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
(h_constant : ∀ a1 d, (a 2 + a 8 + a 11 = 3 * a1 + 18 * d)) : (S 13 = 91 * d) :=
by
  sorry

end constant_S13_l137_137631


namespace janet_dresses_l137_137734

theorem janet_dresses : 
  ∃ D : ℕ, 
    (D / 2) * (2 / 3) + (D / 2) * (6 / 3) = 32 → D = 24 := 
by {
  sorry
}

end janet_dresses_l137_137734


namespace average_inside_time_l137_137866

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l137_137866


namespace inspection_time_l137_137154

theorem inspection_time 
  (num_digits : ℕ) (num_letters : ℕ) 
  (letter_opts : ℕ) (start_digits : ℕ) 
  (inspection_time_three_hours : ℕ) 
  (probability : ℝ) 
  (num_vehicles : ℕ) 
  (vehicles_inspected : ℕ)
  (cond1 : num_digits = 4)
  (cond2 : num_letters = 2)
  (cond3 : letter_opts = 3)
  (cond4 : start_digits = 2)
  (cond5 : inspection_time_three_hours = 180) 
  (cond6 : probability = 0.02)
  (cond7 : num_vehicles = 900)
  (cond8 : vehicles_inspected = num_vehicles * probability) :
  vehicles_inspected = (inspection_time_three_hours / 10) :=
  sorry

end inspection_time_l137_137154


namespace scalene_triangle_third_side_l137_137542

theorem scalene_triangle_third_side (a b c : ℕ) (h : (a - 3)^2 + (b - 2)^2 = 0) : 
  a = 3 ∧ b = 2 → c = 2 ∨ c = 3 ∨ c = 4 := 
by {
  sorry
}

end scalene_triangle_third_side_l137_137542


namespace simplify_fractions_l137_137351

theorem simplify_fractions : 
  (150 / 225) + (90 / 135) = 4 / 3 := by 
  sorry

end simplify_fractions_l137_137351


namespace residue_of_neg_1237_mod_29_l137_137521

theorem residue_of_neg_1237_mod_29 :
  (-1237 : ℤ) % 29 = 10 :=
sorry

end residue_of_neg_1237_mod_29_l137_137521


namespace plywood_perimeter_difference_l137_137332

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l137_137332


namespace car_speed_l137_137220

theorem car_speed
  (v : ℝ)       -- the unknown speed of the car in km/hr
  (time_80 : ℝ := 45)  -- the time in seconds to travel 1 km at 80 km/hr
  (time_plus_10 : ℝ := 55)  -- the time in seconds to travel 1 km at speed v

  (h1 : time_80 = 3600 / 80)
  (h2 : time_plus_10 = time_80 + 10) :
  v = 3600 / (55 / 3600) := sorry

end car_speed_l137_137220


namespace complement_union_l137_137336

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem complement_union (x : ℝ) : (x ∈ Aᶜ ∪ B) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ioi 0) := by
  sorry

end complement_union_l137_137336


namespace avg_speed_while_climbing_l137_137095

-- Definitions for conditions
def totalClimbTime : ℝ := 4
def restBreaks : ℝ := 0.5
def descentTime : ℝ := 2
def avgSpeedWholeJourney : ℝ := 1.5
def totalDistance : ℝ := avgSpeedWholeJourney * (totalClimbTime + descentTime)

-- The question: Prove Natasha's average speed while climbing to the top, excluding the rest breaks duration.
theorem avg_speed_while_climbing :
  (totalDistance / 2) / (totalClimbTime - restBreaks) = 1.29 := 
sorry

end avg_speed_while_climbing_l137_137095


namespace percent_of_y_l137_137920

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 * y) / 20 + (3 * y) / 10) = (35/100) * y :=
by
  sorry

end percent_of_y_l137_137920


namespace square_completion_form_l137_137242

theorem square_completion_form (x k m: ℝ) (h: 16*x^2 - 32*x - 512 = 0):
  (x + k)^2 = m ↔ m = 65 :=
by
  sorry

end square_completion_form_l137_137242


namespace center_of_circle_from_diameter_l137_137867

theorem center_of_circle_from_diameter (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3) (h2 : y1 = -3) (h3 : x2 = 13) (h4 : y2 = 17) :
  (x1 + x2) / 2 = 8 ∧ (y1 + y2) / 2 = 7 :=
by
  sorry

end center_of_circle_from_diameter_l137_137867


namespace cinco_de_mayo_day_days_between_feb_14_and_may_5_l137_137455

theorem cinco_de_mayo_day {
  feb_14_is_tuesday : ∃ n : ℕ, n % 7 = 2
}: 
∃ n : ℕ, n % 7 = 5 := sorry

theorem days_between_feb_14_and_may_5: 
  ∃ d : ℕ, 
  d = 81 := sorry

end cinco_de_mayo_day_days_between_feb_14_and_may_5_l137_137455


namespace total_students_in_class_l137_137343

-- Definitions based on the conditions
def num_girls : ℕ := 140
def num_boys_absent : ℕ := 40
def num_boys_present := num_girls / 2
def num_boys := num_boys_present + num_boys_absent
def total_students := num_girls + num_boys

-- Theorem to be proved
theorem total_students_in_class : total_students = 250 :=
by
  sorry

end total_students_in_class_l137_137343


namespace fifth_term_of_sequence_is_31_l137_137954

namespace SequenceProof

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem fifth_term_of_sequence_is_31 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 5 = 31 :=
by
  sorry

end SequenceProof

end fifth_term_of_sequence_is_31_l137_137954


namespace ellipse_equation_l137_137313

theorem ellipse_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∃ (P : ℝ × ℝ), P = (0, -1) ∧ P.2^2 = b^2) 
  (h4 : ∃ (C2 : ℝ → ℝ → Prop), (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 = 4) ∧ 2 * a = 4) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1) :=
by
  sorry

end ellipse_equation_l137_137313


namespace probability_divisible_by_3_l137_137281

theorem probability_divisible_by_3 :
  ∀ (n : ℤ), (1 ≤ n) ∧ (n ≤ 99) → 3 ∣ (n * (n + 1)) :=
by
  intros n hn
  -- Detailed proof would follow here
  sorry

end probability_divisible_by_3_l137_137281


namespace pages_read_on_Monday_l137_137944

variable (P : Nat) (W : Nat)
def TotalPages : Nat := P + 12 + W

theorem pages_read_on_Monday :
  (TotalPages P W = 51) → (P = 39) :=
by
  sorry

end pages_read_on_Monday_l137_137944


namespace dog_food_cans_l137_137345

theorem dog_food_cans 
  (packages_cat_food : ℕ)
  (cans_per_package_cat_food : ℕ)
  (packages_dog_food : ℕ)
  (additional_cans_cat_food : ℕ)
  (total_cans_cat_food : ℕ)
  (total_cans_dog_food : ℕ)
  (num_cans_dog_food_package : ℕ) :
  packages_cat_food = 9 →
  cans_per_package_cat_food = 10 →
  packages_dog_food = 7 →
  additional_cans_cat_food = 55 →
  total_cans_cat_food = packages_cat_food * cans_per_package_cat_food →
  total_cans_dog_food = packages_dog_food * num_cans_dog_food_package →
  total_cans_cat_food = total_cans_dog_food + additional_cans_cat_food →
  num_cans_dog_food_package = 5 :=
by
  sorry

end dog_food_cans_l137_137345


namespace find_number_exists_l137_137529

theorem find_number_exists (n : ℤ) : (50 < n ∧ n < 70) ∧
    (n % 5 = 3) ∧
    (n % 7 = 2) ∧
    (n % 8 = 2) → n = 58 := 
sorry

end find_number_exists_l137_137529


namespace probability_red_or_white_l137_137432

-- Define the total number of marbles and the counts of blue and red marbles.
def total_marbles : Nat := 60
def blue_marbles : Nat := 5
def red_marbles : Nat := 9

-- Define the remainder to calculate white marbles.
def white_marbles : Nat := total_marbles - (blue_marbles + red_marbles)

-- Lean proof statement to show the probability of selecting a red or white marble.
theorem probability_red_or_white :
  (red_marbles + white_marbles) / total_marbles = 11 / 12 :=
by
  sorry

end probability_red_or_white_l137_137432


namespace proof_problem_l137_137098

theorem proof_problem
  (x y a b c d : ℝ)
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c + d = 0) :
  (x + y)^3 - (-a * b)^2 + 3 * c + 3 * d = -2 :=
by
  -- The proof steps go here.
  sorry

end proof_problem_l137_137098


namespace choose_18_4_eq_3060_l137_137074

/-- The number of ways to select 4 members from a group of 18 people (without regard to order). -/
theorem choose_18_4_eq_3060 : Nat.choose 18 4 = 3060 := 
by
  sorry

end choose_18_4_eq_3060_l137_137074


namespace roots_eq_squares_l137_137598

theorem roots_eq_squares (p q : ℝ) (h1 : p^2 - 5 * p + 6 = 0) (h2 : q^2 - 5 * q + 6 = 0) :
  p^2 + q^2 = 13 :=
sorry

end roots_eq_squares_l137_137598


namespace sets_are_equal_l137_137669

def setA : Set ℤ := {a | ∃ m n l : ℤ, a = 12 * m + 8 * n + 4 * l}
def setB : Set ℤ := {b | ∃ p q r : ℤ, b = 20 * p + 16 * q + 12 * r}

theorem sets_are_equal : setA = setB := sorry

end sets_are_equal_l137_137669


namespace range_of_x_range_of_a_l137_137212

variable (a x : ℝ)

-- Define proposition p: x^2 - 3ax + 2a^2 < 0
def p (a x : ℝ) : Prop := x^2 - 3 * a * x + 2 * a^2 < 0

-- Define proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- First theorem: Prove the range of x when a = 2 and p ∨ q is true
theorem range_of_x (h : p 2 x ∨ q x) : 2 < x ∧ x < 4 := 
by sorry

-- Second theorem: Prove the range of a when ¬p is necessary but not sufficient for ¬q
theorem range_of_a (h : ∀ x, q x → p a x) : 3/2 ≤ a ∧ a ≤ 2 := 
by sorry

end range_of_x_range_of_a_l137_137212


namespace largest_five_digit_negative_int_congruent_mod_23_l137_137937

theorem largest_five_digit_negative_int_congruent_mod_23 :
  ∃ n : ℤ, 23 * n + 1 < -9999 ∧ 23 * n + 1 = -9994 := 
sorry

end largest_five_digit_negative_int_congruent_mod_23_l137_137937


namespace percent_of_x_is_z_l137_137766

-- Defining the conditions as constants in the Lean environment
variables (x y z : ℝ)

-- Given conditions
def cond1 : Prop := 0.45 * z = 0.90 * y
def cond2 : Prop := y = 0.75 * x

-- The statement of the problem proving z = 1.5 * x under given conditions
theorem percent_of_x_is_z
  (h1 : cond1 z y)
  (h2 : cond2 y x) :
  z = 1.5 * x :=
sorry

end percent_of_x_is_z_l137_137766


namespace ellipse_equation_is_standard_form_l137_137423

theorem ellipse_equation_is_standard_form (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_mn_neq : m ≠ n) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ (∀ x y : ℝ, mx^2 + ny^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_is_standard_form_l137_137423


namespace coeff_of_term_equal_three_l137_137748

theorem coeff_of_term_equal_three (x : ℕ) (h : x = 13) : 
    2^x - 2^(x - 2) = 3 * 2^(11) :=
by
    rw [h]
    sorry

end coeff_of_term_equal_three_l137_137748


namespace total_messages_equation_l137_137723

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l137_137723


namespace weight_difference_l137_137864

noncomputable def W_A : ℝ := 78

variable (W_B W_C W_D W_E : ℝ)

axiom cond1 : (W_A + W_B + W_C) / 3 = 84
axiom cond2 : (W_A + W_B + W_C + W_D) / 4 = 80
axiom cond3 : (W_B + W_C + W_D + W_E) / 4 = 79

theorem weight_difference : W_E - W_D = 6 :=
by
  have h1 : W_A = 78 := rfl
  sorry

end weight_difference_l137_137864


namespace irrational_roots_of_odd_quadratic_l137_137227

theorem irrational_roots_of_odd_quadratic (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ p * p = a * (p / q) * (p / q) + b * (p / q) + c := sorry

end irrational_roots_of_odd_quadratic_l137_137227


namespace intersection_of_sets_l137_137207

noncomputable def universal_set (x : ℝ) := true

def set_A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

def set_B (x : ℝ) : Prop := ∃ y, y = Real.log (1 - x)

def complement_U_B (x : ℝ) : Prop := ¬ set_B x

theorem intersection_of_sets :
  { x : ℝ | set_A x } ∩ { x | complement_U_B x } = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_sets_l137_137207


namespace carnival_tickets_l137_137488

theorem carnival_tickets (x : ℕ) (won_tickets : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ)
  (h1 : won_tickets = 5 * x)
  (h2 : found_tickets = 5)
  (h3 : ticket_value = 3)
  (h4 : total_value = 30)
  (h5 : total_value = (won_tickets + found_tickets) * ticket_value) :
  x = 1 :=
by
  -- Proof omitted
  sorry

end carnival_tickets_l137_137488


namespace sahil_selling_price_l137_137708

-- Define the conditions
def purchased_price := 9000
def repair_cost := 5000
def transportation_charges := 1000
def profit_percentage := 50 / 100

-- Calculate the total cost
def total_cost := purchased_price + repair_cost + transportation_charges

-- Calculate the selling price
def selling_price := total_cost + (profit_percentage * total_cost)

-- The theorem to prove the selling price
theorem sahil_selling_price : selling_price = 22500 :=
by
  -- This is where the proof would go, but we skip it with sorry.
  sorry

end sahil_selling_price_l137_137708


namespace pinky_pies_count_l137_137057

theorem pinky_pies_count (helen_pies : ℕ) (total_pies : ℕ) (h1 : helen_pies = 56) (h2 : total_pies = 203) : 
  total_pies - helen_pies = 147 := by
  sorry

end pinky_pies_count_l137_137057


namespace max_reflections_l137_137247

theorem max_reflections (n : ℕ) (angle_CDA : ℝ) (h_angle : angle_CDA = 12) : n ≤ 7 ↔ 12 * n ≤ 90 := by
    sorry

end max_reflections_l137_137247


namespace gcd_102_238_l137_137741

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l137_137741


namespace aquarium_counts_l137_137783

theorem aquarium_counts :
  ∃ (O S L : ℕ), O + S = 7 ∧ L + S = 6 ∧ O + L = 5 ∧ (O ≤ S ∧ O ≤ L) ∧ O = 5 ∧ S = 7 ∧ L = 6 :=
by
  sorry

end aquarium_counts_l137_137783


namespace equation_true_when_n_eq_2_l137_137522

theorem equation_true_when_n_eq_2 : (2 ^ (2 / 2)) = 2 :=
by
  sorry

end equation_true_when_n_eq_2_l137_137522


namespace min_number_of_participants_l137_137088

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l137_137088


namespace min_value_of_function_l137_137146

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (x + 1/x + x^2 + 1/x^2 + 1 / (x + 1/x + x^2 + 1/x^2)) = 4.25 := by
  sorry

end min_value_of_function_l137_137146


namespace simplify_expression_l137_137391

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l137_137391


namespace remainder_415_pow_420_div_16_l137_137665

theorem remainder_415_pow_420_div_16 : 415^420 % 16 = 1 := by
  sorry

end remainder_415_pow_420_div_16_l137_137665


namespace find_a_l137_137510

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 2 ∨ a = 3 := by
  sorry

end find_a_l137_137510


namespace white_marbles_bagA_eq_fifteen_l137_137096

noncomputable def red_marbles_bagA := 5
def rw_ratio_bagA := (1, 3)
def wb_ratio_bagA := (2, 3)

theorem white_marbles_bagA_eq_fifteen :
  let red_to_white := rw_ratio_bagA.1 * red_marbles_bagA
  red_to_white * rw_ratio_bagA.2 = 15 :=
by
  sorry

end white_marbles_bagA_eq_fifteen_l137_137096


namespace students_math_inequality_l137_137306

variables {n x a b c : ℕ}

theorem students_math_inequality (h1 : x + a ≥ 8 * n / 10) 
                                (h2 : x + b ≥ 8 * n / 10) 
                                (h3 : n ≥ a + b + c + x) : 
                                x * 5 ≥ 4 * (x + c) :=
by
  sorry

end students_math_inequality_l137_137306


namespace min_distance_from_point_to_line_l137_137182

theorem min_distance_from_point_to_line : 
  ∀ (x₀ y₀ : Real), 3 * x₀ - 4 * y₀ - 10 = 0 → Real.sqrt (x₀^2 + y₀^2) = 2 :=
by sorry

end min_distance_from_point_to_line_l137_137182


namespace function_is_decreasing_l137_137176

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2

theorem function_is_decreasing (a b : ℝ) (f_even : ∀ x : ℝ, f a b x = f a b (-x))
  (domain_condition : 1 + a + 2 = 0) :
  ∀ x y : ℝ, 1 ≤ x → x < y → y ≤ 2 → f a 0 x > f a 0 y :=
by
  sorry

end function_is_decreasing_l137_137176


namespace tan_sum_product_l137_137718

theorem tan_sum_product (A B C : ℝ) (h_eq: Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := by
  sorry

end tan_sum_product_l137_137718


namespace total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l137_137572

def sandwiches_cost (s: ℕ) : ℝ := 4 * s
def sodas_cost (d: ℕ) : ℝ := 3 * d
def total_cost_before_tax (s: ℕ) (d: ℕ) : ℝ := sandwiches_cost s + sodas_cost d
def tax (amount: ℝ) : ℝ := 0.10 * amount
def total_cost (s: ℕ) (d: ℕ) : ℝ := total_cost_before_tax s d + tax (total_cost_before_tax s d)

theorem total_cost_of_4_sandwiches_and_6_sodas_is_37_4 :
    total_cost 4 6 = 37.4 :=
sorry

end total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l137_137572


namespace problem_solution_l137_137289

def positive (n : ℕ) : Prop := n > 0
def pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1
def divides (m n : ℕ) : Prop := ∃ k, n = k * m

theorem problem_solution (a b c : ℕ) :
  positive a → positive b → positive c →
  pairwise_coprime a b c →
  divides (a^2) (b^3 + c^3) →
  divides (b^2) (a^3 + c^3) →
  divides (c^2) (a^3 + b^3) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end problem_solution_l137_137289


namespace find_original_sum_of_money_l137_137770

theorem find_original_sum_of_money
  (R : ℝ)
  (P : ℝ)
  (h1 : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 63) :
  P = 2100 :=
sorry

end find_original_sum_of_money_l137_137770


namespace find_fixed_point_c_l137_137201

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := 2 * x ^ 2 - c

theorem find_fixed_point_c (c : ℝ) : 
  (∃ a : ℝ, f a = a ∧ g a c = a) ↔ (c = 3 ∨ c = 6) := sorry

end find_fixed_point_c_l137_137201


namespace combined_loading_time_l137_137032

theorem combined_loading_time (rA rB rC : ℝ) (hA : rA = 1 / 6) (hB : rB = 1 / 8) (hC : rC = 1 / 10) :
  1 / (rA + rB + rC) = 120 / 47 := by
  sorry

end combined_loading_time_l137_137032


namespace solve_for_x_l137_137420

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end solve_for_x_l137_137420


namespace max_lights_correct_l137_137913

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end max_lights_correct_l137_137913


namespace points_lie_on_hyperbola_l137_137978

noncomputable def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * (Real.exp t + Real.exp (-t))
  let y := 4 * (Real.exp t - Real.exp (-t))
  (x^2 / 16) - (y^2 / 64) = 1

theorem points_lie_on_hyperbola (t : ℝ) : point_on_hyperbola t := 
by
  sorry

end points_lie_on_hyperbola_l137_137978


namespace sum_is_two_l137_137846

noncomputable def compute_sum (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1)) + (x^8 / (x^4 - 1)) + (x^10 / (x^5 - 1)) + (x^12 / (x^6 - 1))

theorem sum_is_two (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : compute_sum x hx hx_ne = 2 :=
by
  sorry

end sum_is_two_l137_137846


namespace percentage_of_towns_correct_l137_137889

def percentage_of_towns_with_fewer_than_50000_residents (p1 p2 p3 : ℝ) : ℝ :=
  p1 + p2

theorem percentage_of_towns_correct (p1 p2 p3 : ℝ) (h1 : p1 = 0.15) (h2 : p2 = 0.30) (h3 : p3 = 0.55) :
  percentage_of_towns_with_fewer_than_50000_residents p1 p2 p3 = 0.45 :=
by 
  sorry

end percentage_of_towns_correct_l137_137889


namespace jam_cost_l137_137410

theorem jam_cost (N B J H : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 6 * J + 2 * H) = 342) :
  6 * N * J = 270 := 
sorry

end jam_cost_l137_137410


namespace repeating_decimal_sum_l137_137603

theorem repeating_decimal_sum (c d : ℕ) (h : 7 / 19 = (c * 10 + d) / 99) : c + d = 9 :=
sorry

end repeating_decimal_sum_l137_137603


namespace travel_probability_l137_137742

theorem travel_probability (P_A P_B P_C : ℝ) (hA : P_A = 1/3) (hB : P_B = 1/4) (hC : P_C = 1/5) :
  let P_none_travel := (1 - P_A) * (1 - P_B) * (1 - P_C)
  ∃ (P_at_least_one : ℝ), P_at_least_one = 1 - P_none_travel ∧ P_at_least_one = 3/5 :=
by {
  sorry
}

end travel_probability_l137_137742


namespace triangle_angle_A_l137_137412

theorem triangle_angle_A (A B a b : ℝ) (h1 : b = 2 * a) (h2 : B = A + 60) : A = 30 :=
by sorry

end triangle_angle_A_l137_137412


namespace problem_part_1_problem_part_2_l137_137302

variable (θ : Real)
variable (m : Real)
variable (h_θ : θ ∈ Ioc 0 (2 * Real.pi))
variable (h_eq : ∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ (x = Real.sin θ ∨ x = Real.cos θ))

theorem problem_part_1 : 
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 := 
by
  sorry

theorem problem_part_2 : 
  m = Real.sqrt 3 / 2 := 
by 
  sorry

end problem_part_1_problem_part_2_l137_137302


namespace total_number_of_eggs_l137_137459

theorem total_number_of_eggs 
  (cartons : ℕ) 
  (eggs_per_carton_length : ℕ) 
  (eggs_per_carton_width : ℕ)
  (egg_position_from_front : ℕ)
  (egg_position_from_back : ℕ)
  (egg_position_from_left : ℕ)
  (egg_position_from_right : ℕ) :
  cartons = 28 →
  egg_position_from_front = 14 →
  egg_position_from_back = 20 →
  egg_position_from_left = 3 →
  egg_position_from_right = 2 →
  eggs_per_carton_length = egg_position_from_front + egg_position_from_back - 1 →
  eggs_per_carton_width = egg_position_from_left + egg_position_from_right - 1 →
  cartons * (eggs_per_carton_length * eggs_per_carton_width) = 3696 := 
  by 
  intros
  sorry

end total_number_of_eggs_l137_137459


namespace line_perpendicular_to_two_planes_parallel_l137_137980

-- Declare lines and planes
variables {Line Plane : Type}

-- Define the perpendicular and parallel relationships
variables (perpendicular : Line → Plane → Prop)
variables (parallel : Plane → Plane → Prop)

-- Given conditions
variables (m n : Line) (α β : Plane)
-- The known conditions are:
-- 1. m is perpendicular to α
-- 2. m is perpendicular to β
-- We want to prove:
-- 3. α is parallel to β

theorem line_perpendicular_to_two_planes_parallel (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β :=
sorry

end line_perpendicular_to_two_planes_parallel_l137_137980


namespace impossible_to_repaint_white_l137_137054

-- Define the board as a 7x7 grid 
def boardSize : ℕ := 7

-- Define the initial coloring function (checkerboard with corners black)
def initialColor (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the repainting operation allowed
def repaint (cell1 cell2 : (ℕ × ℕ)) (color1 color2 : Prop) : Prop :=
  ¬color1 = color2 

-- Define the main theorem to prove
theorem impossible_to_repaint_white :
  ¬(∃ f : ℕ × ℕ -> Prop, 
    (∀ i j, (i < boardSize) → (j < boardSize) → (f (i, j) = true)) ∧ 
    (∀ i j, (i < boardSize - 1) → (repaint (i, j) (i, j+1) (f (i, j)) (f (i, j+1))) ∧
             (i < boardSize - 1) → (repaint (i, j) (i+1, j) (f (i, j)) (f (i+1, j)))))
  :=
  sorry

end impossible_to_repaint_white_l137_137054


namespace pages_in_first_chapter_l137_137837

/--
Rita is reading a five-chapter book with 95 pages. Each chapter has three pages more than the previous one. 
Prove the number of pages in the first chapter.
-/
theorem pages_in_first_chapter (h : ∃ p1 p2 p3 p4 p5 : ℕ, p1 + p2 + p3 + p4 + p5 = 95 ∧ p2 = p1 + 3 ∧ p3 = p1 + 6 ∧ p4 = p1 + 9 ∧ p5 = p1 + 12) : 
  ∃ x : ℕ, x = 13 := 
by
  sorry

end pages_in_first_chapter_l137_137837


namespace central_vs_northern_chess_match_l137_137969

noncomputable def schedule_chess_match : Nat :=
  let players_team1 := ["A", "B", "C"];
  let players_team2 := ["X", "Y", "Z"];
  let total_games := 3 * 3 * 3;
  let games_per_round := 4;
  let total_rounds := 7;
  Nat.factorial total_rounds

theorem central_vs_northern_chess_match :
    schedule_chess_match = 5040 :=
by
  sorry

end central_vs_northern_chess_match_l137_137969


namespace correct_operations_result_greater_than_1000_l137_137946

theorem correct_operations_result_greater_than_1000
    (finalResultIncorrectOps : ℕ)
    (originalNumber : ℕ)
    (finalResultCorrectOps : ℕ)
    (H1 : finalResultIncorrectOps = 40)
    (H2 : originalNumber = (finalResultIncorrectOps + 12) * 8)
    (H3 : finalResultCorrectOps = (originalNumber * 8) + (2 * originalNumber) + 12) :
  finalResultCorrectOps > 1000 := 
sorry

end correct_operations_result_greater_than_1000_l137_137946


namespace point_on_parabola_dist_3_from_focus_l137_137300

def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 4 * p.fst

def focus : ℝ × ℝ := (1, 0)

theorem point_on_parabola_dist_3_from_focus :
  ∃ y: ℝ, ∃ x: ℝ, (parabola (x, y) ∧ (x = 2) ∧ (y = 2 * Real.sqrt 2 ∨ y = -2 * Real.sqrt 2) ∧ (Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 3)) :=
by
  sorry

end point_on_parabola_dist_3_from_focus_l137_137300


namespace original_number_count_l137_137401

theorem original_number_count (k S : ℕ) (M : ℚ)
  (hk : k > 0)
  (hM : M = S / k)
  (h_add15 : (S + 15) / (k + 1) = M + 2)
  (h_add1 : (S + 16) / (k + 2) = M + 1) :
  k = 6 :=
by
  -- Proof will go here
  sorry

end original_number_count_l137_137401


namespace tagged_fish_in_second_catch_l137_137924

-- Definitions and conditions
def total_fish_in_pond : ℕ := 1750
def tagged_fish_initial : ℕ := 70
def fish_caught_second_time : ℕ := 50
def ratio_tagged_fish : ℚ := tagged_fish_initial / total_fish_in_pond

-- Theorem statement
theorem tagged_fish_in_second_catch (T : ℕ) : (T : ℚ) / fish_caught_second_time = ratio_tagged_fish → T = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l137_137924


namespace sin_pi_minus_2alpha_l137_137673

theorem sin_pi_minus_2alpha (α : ℝ) (h1 : Real.sin (π / 2 + α) = -3 / 5) (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end sin_pi_minus_2alpha_l137_137673


namespace find_initial_salt_concentration_l137_137810

noncomputable def initial_salt_concentration 
  (x : ℝ) (final_concentration : ℝ) (extra_water : ℝ) (extra_salt : ℝ) (evaporation_fraction : ℝ) : ℝ :=
  let initial_volume : ℝ := x
  let remaining_volume : ℝ := evaporation_fraction * initial_volume
  let mixed_volume : ℝ := remaining_volume + extra_water + extra_salt
  let target_salt_volume_fraction : ℝ := final_concentration / 100
  let initial_salt_volume_fraction : ℝ := (target_salt_volume_fraction * mixed_volume - extra_salt) / initial_volume * 100
  initial_salt_volume_fraction

theorem find_initial_salt_concentration :
  initial_salt_concentration 120 33.333333333333336 8 16 (3 / 4) = 18.333333333333332 :=
by
  sorry

end find_initial_salt_concentration_l137_137810


namespace unique_trivial_solution_of_linear_system_l137_137740

variable {R : Type*} [Field R]

theorem unique_trivial_solution_of_linear_system (a b c x y z : R)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_system : x + a * y + a^2 * z = 0 ∧ x + b * y + b^2 * z = 0 ∧ x + c * y + c^2 * z = 0) :
  x = 0 ∧ y = 0 ∧ z = 0 := sorry

end unique_trivial_solution_of_linear_system_l137_137740


namespace area_of_square_field_l137_137660

theorem area_of_square_field (d : ℝ) (s : ℝ) (A : ℝ) (h_d : d = 28) (h_relation : d = s * Real.sqrt 2) (h_area : A = s^2) :
  A = 391.922 :=
by sorry

end area_of_square_field_l137_137660


namespace statement_a_correct_statement_b_correct_l137_137692

open Real

theorem statement_a_correct (a b c : ℝ) (ha : a > b) (hc : c < 0) : a + c > b + c := by
  sorry

theorem statement_b_correct (a b : ℝ) (ha : a > b) (hb : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end statement_a_correct_statement_b_correct_l137_137692


namespace right_triangle_geo_seq_ratio_l137_137428

theorem right_triangle_geo_seq_ratio (l r : ℝ) (ht : 0 < l)
  (hr : 1 < r) (hgeo : l^2 + (l * r)^2 = (l * r^2)^2) :
  (l * r^2) / l = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end right_triangle_geo_seq_ratio_l137_137428


namespace arithmetic_sequence_a10_l137_137807

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) 
    (h1 : a 2 = 2) 
    (h2 : a 3 = 4) : 
    a 10 = 18 := 
sorry

end arithmetic_sequence_a10_l137_137807


namespace three_digit_avg_permutations_l137_137834

theorem three_digit_avg_permutations (a b c: ℕ) (A: ℕ) (h₀: 1 ≤ a ∧ a ≤ 9) (h₁: 0 ≤ b ∧ b ≤ 9) (h₂: 0 ≤ c ∧ c ≤ 9) (h₃: A = 100 * a + 10 * b + c):
  ((100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)) / 6 = A ↔ 7 * a = 3 * b + 4 * c := by
  sorry

end three_digit_avg_permutations_l137_137834


namespace ratio_c_d_l137_137812

theorem ratio_c_d (x y c d : ℝ) 
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = -2 / 3 :=
sorry

end ratio_c_d_l137_137812


namespace determine_coefficients_l137_137929

theorem determine_coefficients (A B C : ℝ) 
  (h1 : 3 * A - 1 = 0)
  (h2 : 3 * A^2 + 3 * B = 0)
  (h3 : A^3 + 6 * A * B + 3 * C = 0) :
  A = 1 / 3 ∧ B = -1 / 9 ∧ C = 5 / 81 :=
by 
  sorry

end determine_coefficients_l137_137929


namespace last_8_digits_of_product_l137_137621

theorem last_8_digits_of_product :
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  (p % 100000000) = 87654321 :=
by
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  have : p % 100000000 = 87654321 := sorry
  exact this

end last_8_digits_of_product_l137_137621


namespace mixed_solution_concentration_l137_137835

def salt_amount_solution1 (weight1 : ℕ) (concentration1 : ℕ) : ℕ := (concentration1 * weight1) / 100
def salt_amount_solution2 (salt2 : ℕ) : ℕ := salt2
def total_salt (salt1 salt2 : ℕ) : ℕ := salt1 + salt2
def total_weight (weight1 weight2 : ℕ) : ℕ := weight1 + weight2
def concentration (total_salt : ℕ) (total_weight : ℕ) : ℚ := (total_salt : ℚ) / (total_weight : ℚ) * 100

theorem mixed_solution_concentration 
  (weight1 weight2 salt2 : ℕ) (concentration1 : ℕ)
  (h_weight1 : weight1 = 200)
  (h_weight2 : weight2 = 300)
  (h_concentration1 : concentration1 = 25)
  (h_salt2 : salt2 = 60) :
  concentration (total_salt (salt_amount_solution1 weight1 concentration1) (salt_amount_solution2 salt2)) (total_weight weight1 weight2) = 22 := 
sorry

end mixed_solution_concentration_l137_137835


namespace range_of_a_l137_137910

noncomputable def function_with_extreme_at_zero_only (a b : ℝ) : Prop :=
∀ x : ℝ, x ≠ 0 → 4 * x^2 + 3 * a * x + 4 > 0

theorem range_of_a (a b : ℝ) (h : function_with_extreme_at_zero_only a b) : 
  -8 / 3 ≤ a ∧ a ≤ 8 / 3 :=
sorry

end range_of_a_l137_137910


namespace multiplication_even_a_b_multiplication_even_a_a_l137_137226

def a : Int := 4
def b : Int := 3

theorem multiplication_even_a_b : a * b = 12 := by sorry
theorem multiplication_even_a_a : a * a = 16 := by sorry

end multiplication_even_a_b_multiplication_even_a_a_l137_137226


namespace complete_square_k_value_l137_137764

theorem complete_square_k_value : 
  ∃ k : ℝ, ∀ x : ℝ, (x^2 - 8*x = (x - 4)^2 + k) ∧ k = -16 :=
by
  use -16
  intro x
  sorry

end complete_square_k_value_l137_137764


namespace trigonometric_inequality_solution_l137_137800

theorem trigonometric_inequality_solution (k : ℤ) :
  ∃ x : ℝ, x = - (3 * Real.pi) / 2 + 4 * Real.pi * k ∧
           (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) :=
by
  sorry

end trigonometric_inequality_solution_l137_137800


namespace ral_current_age_l137_137605

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l137_137605


namespace evaluate_f_diff_l137_137463

def f (x : ℝ) : ℝ := x^4 + 3 * x^3 + 2 * x^2 + 7 * x

theorem evaluate_f_diff:
  f 6 - f (-6) = 1380 := by
  sorry

end evaluate_f_diff_l137_137463


namespace boat_speed_5_kmh_l137_137796

noncomputable def boat_speed_in_still_water (V_s : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  (d / t) - V_s

theorem boat_speed_5_kmh :
  boat_speed_in_still_water 5 10 100 = 5 :=
by
  sorry

end boat_speed_5_kmh_l137_137796


namespace integers_less_than_2019_divisible_by_18_or_21_but_not_both_l137_137746

theorem integers_less_than_2019_divisible_by_18_or_21_but_not_both :
  ∃ (N : ℕ), (∀ (n : ℕ), (n < 2019 → (n % 18 = 0 ∨ n % 21 = 0) → n % (18 * 21 / gcd 18 21) ≠ 0) ↔ (∀ (m : ℕ), m < N)) ∧ N = 176 :=
by
  sorry

end integers_less_than_2019_divisible_by_18_or_21_but_not_both_l137_137746


namespace train_stop_times_l137_137881

theorem train_stop_times :
  ∀ (speed_without_stops_A speed_with_stops_A speed_without_stops_B speed_with_stops_B : ℕ),
  speed_without_stops_A = 45 →
  speed_with_stops_A = 30 →
  speed_without_stops_B = 60 →
  speed_with_stops_B = 40 →
  (60 * (speed_without_stops_A - speed_with_stops_A) / speed_without_stops_A = 20) ∧
  (60 * (speed_without_stops_B - speed_with_stops_B) / speed_without_stops_B = 20) :=
by
  intros
  sorry

end train_stop_times_l137_137881


namespace synodic_month_is_approx_29_5306_l137_137821

noncomputable def sidereal_month_moon : ℝ := 
27 + 7/24 + 43/1440  -- conversion of 7 hours and 43 minutes to days

noncomputable def sidereal_year_earth : ℝ := 
365 + 6/24 + 9/1440  -- conversion of 6 hours and 9 minutes to days

noncomputable def synodic_month (T_H T_F: ℝ) : ℝ := 
(T_H * T_F) / (T_F - T_H)

theorem synodic_month_is_approx_29_5306 : 
  abs (synodic_month sidereal_month_moon sidereal_year_earth - (29 + 12/24 + 44/1440)) < 0.0001 :=
by 
  sorry

end synodic_month_is_approx_29_5306_l137_137821


namespace a2_range_l137_137284

open Nat

noncomputable def a_seq (a : ℕ → ℝ) := ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)

theorem a2_range (a : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)) 
  (h2 : ∀ (m n : ℕ), m < n → a m ≤ a n) 
  (h3 : a 1 = 2) :
  (2 < a 2) ∧ (a 2 ≤ 4) :=
sorry

end a2_range_l137_137284


namespace perimeter_of_triangle_l137_137113

-- Defining the basic structure of the problem
theorem perimeter_of_triangle (A B C : Type)
  (distance_AB distance_AC distance_BC : ℝ)
  (angle_B : ℝ)
  (h1 : distance_AB = distance_AC)
  (h2 : angle_B = 60)
  (h3 : distance_BC = 4) :
  distance_AB + distance_AC + distance_BC = 12 :=
by 
  sorry

end perimeter_of_triangle_l137_137113


namespace solve_equation_l137_137891

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l137_137891


namespace chess_mixed_games_l137_137670

theorem chess_mixed_games (W M : ℕ) (hW : W * (W - 1) / 2 = 45) (hM : M * (M - 1) / 2 = 190) : M * W = 200 :=
by
  sorry

end chess_mixed_games_l137_137670


namespace partitioning_staircase_l137_137173

def number_of_ways_to_partition_staircase (n : ℕ) : ℕ :=
  2^(n-1)

theorem partitioning_staircase (n : ℕ) : 
  number_of_ways_to_partition_staircase n = 2^(n-1) :=
by 
  sorry

end partitioning_staircase_l137_137173


namespace total_money_shared_l137_137865

-- Define the variables and conditions
def joshua_share : ℕ := 30
def justin_share : ℕ := joshua_share / 3
def total_shared_money : ℕ := joshua_share + justin_share

-- State the theorem to prove
theorem total_money_shared : total_shared_money = 40 :=
by
  -- proof will go here
  sorry

end total_money_shared_l137_137865


namespace find_f_x_squared_l137_137341

-- Define the function f with the given condition
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem find_f_x_squared : f (x^2) = (x^2 + 1)^2 :=
by
  sorry

end find_f_x_squared_l137_137341


namespace linear_equation_m_value_l137_137072

theorem linear_equation_m_value (m : ℝ) (x : ℝ) (h : (m - 1) * x ^ |m| - 2 = 0) : m = -1 :=
sorry

end linear_equation_m_value_l137_137072


namespace total_peanuts_in_box_l137_137545

def initial_peanuts := 4
def peanuts_taken_out := 3
def peanuts_added := 12

theorem total_peanuts_in_box : initial_peanuts - peanuts_taken_out + peanuts_added = 13 :=
by
sorry

end total_peanuts_in_box_l137_137545


namespace teresa_science_marks_l137_137558

-- Definitions for the conditions
def music_marks : ℕ := 80
def social_studies_marks : ℕ := 85
def physics_marks : ℕ := music_marks / 2
def total_marks : ℕ := 275

-- Statement to prove
theorem teresa_science_marks : ∃ S : ℕ, 
  S + music_marks + social_studies_marks + physics_marks = total_marks ∧ S = 70 :=
sorry

end teresa_science_marks_l137_137558


namespace minimum_dot_product_l137_137799

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (1, 2, 0)
def B : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the vector AP
def vector_AP (x : ℝ) := (x - 1, -2, 0)

-- Define the vector BP
def vector_BP (x : ℝ) := (x, -1, 1)

-- Define the dot product of vector AP and vector BP
def dot_product (x : ℝ) : ℝ := (x - 1) * x + (-2) * (-1) + 0 * 1

-- State the theorem
theorem minimum_dot_product : ∃ x : ℝ, dot_product x = (x - 1) * x + 2 ∧ 
  (∀ y : ℝ, dot_product y ≥ dot_product (1/2)) := 
sorry

end minimum_dot_product_l137_137799


namespace least_positive_multiple_24_gt_450_l137_137416

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l137_137416


namespace non_congruent_triangles_perimeter_18_l137_137382

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l137_137382


namespace evaluate_expression_l137_137826

theorem evaluate_expression (c : ℕ) (hc : c = 4) : 
  ((c^c - 2 * c * (c-2)^c + c^2)^c) = 431441456 :=
by
  rw [hc]
  sorry

end evaluate_expression_l137_137826


namespace sqrt_equation_l137_137752

theorem sqrt_equation (n : ℕ) (h : 0 < n) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n + 1) : ℝ) :=
sorry

end sqrt_equation_l137_137752


namespace minimum_number_of_girls_l137_137964

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l137_137964


namespace all_possible_triples_l137_137438

theorem all_possible_triples (x y : ℕ) (z : ℤ) (hz : z % 2 = 1)
                            (h : x.factorial + y.factorial = 8 * z + 2017) :
                            (x = 1 ∧ y = 4 ∧ z = -249) ∨
                            (x = 4 ∧ y = 1 ∧ z = -249) ∨
                            (x = 1 ∧ y = 5 ∧ z = -237) ∨
                            (x = 5 ∧ y = 1 ∧ z = -237) := 
  sorry

end all_possible_triples_l137_137438


namespace range_of_a_l137_137386

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then (1/2 : ℝ) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ a < -1 :=
sorry

end range_of_a_l137_137386


namespace sequence_properties_l137_137109

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n, a (n + 1) = a n + 2

theorem sequence_properties {a : ℕ → ℤ} (h : arithmetic_sequence a) :
  a 2 + a 4 = 6 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end sequence_properties_l137_137109


namespace frac_ab_eq_five_thirds_l137_137671

theorem frac_ab_eq_five_thirds (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 :=
by
  sorry

end frac_ab_eq_five_thirds_l137_137671


namespace operation_evaluation_l137_137181

def my_operation (x y : Int) : Int :=
  x * (y + 1) + x * y

theorem operation_evaluation :
  my_operation (-3) (-4) = 21 := by
  sorry

end operation_evaluation_l137_137181


namespace maximum_possible_value_of_k_l137_137907

theorem maximum_possible_value_of_k :
  ∀ (k : ℕ), 
    (∃ (x : ℕ → ℝ), 
      (∀ i j : ℕ, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → x i > 1 ∧ x i ≠ x j ∧ x i ^ ⌊x j⌋ = x j ^ ⌊x i⌋)) 
      → k ≤ 4 :=
by
  sorry

end maximum_possible_value_of_k_l137_137907


namespace solution_set_of_inequality_l137_137618

theorem solution_set_of_inequality (x : ℝ) : 
  (|x - 1| + |x - 2| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by sorry

end solution_set_of_inequality_l137_137618


namespace eq_relation_q_r_l137_137811

-- Define the angles in the context of the problem
variables {A B C D E F : Type}
variables {angle_BAC angle_BFD angle_ADE angle_FEC : ℝ}
variables (right_triangle_ABC : A → B → C → angle_BAC = 90)

-- Equilateral triangle DEF inscribed in ABC
variables (inscribed_equilateral_DEF : D → E → F)
variables (angle_BFD_eq_p : ∀ p : ℝ, angle_BFD = p)
variables (angle_ADE_eq_q : ∀ q : ℝ, angle_ADE = q)
variables (angle_FEC_eq_r : ∀ r : ℝ, angle_FEC = r)

-- Main statement to be proved
theorem eq_relation_q_r {p q r : ℝ} 
  (right_triangle_ABC : angle_BAC = 90)
  (angle_BFD : angle_BFD = 30 + q)
  (angle_FEC : angle_FEC = 120 - r) :
  q + r = 60 :=
sorry

end eq_relation_q_r_l137_137811


namespace find_range_of_a_l137_137062

-- Define the conditions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4 * x + a^2 > 0
def q (a : ℝ) : Prop := a^2 - 5 * a - 6 ≥ 0

-- Define the proposition that one of p or q is true and the other is false
def p_or_q (a : ℝ) : Prop := p a ∨ q a
def not_p_and_q (a : ℝ) : Prop := ¬(p a ∧ q a)

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (2 < a ∧ a < 6) ∨ (-2 ≤ a ∧ a ≤ -1)

-- Theorem statement
theorem find_range_of_a (a : ℝ) : p_or_q a ∧ not_p_and_q a → range_of_a a :=
by
  sorry

end find_range_of_a_l137_137062


namespace brainiacs_like_neither_l137_137216

variables 
  (total : ℕ) -- Total number of brainiacs.
  (R : ℕ) -- Number of brainiacs who like rebus teasers.
  (M : ℕ) -- Number of brainiacs who like math teasers.
  (both : ℕ) -- Number of brainiacs who like both rebus and math teasers.
  (math_only : ℕ) -- Number of brainiacs who like only math teasers.

-- Given conditions in the problem
def twice_as_many_rebus : Prop := R = 2 * M
def both_teasers : Prop := both = 18
def math_teasers_not_rebus : Prop := math_only = 20
def total_brainiacs : Prop := total = 100

noncomputable def exclusion_inclusion : ℕ := R + M - both

-- Proof statement: The number of brainiacs who like neither rebus nor math teasers totals to 4
theorem brainiacs_like_neither
  (h_total : total_brainiacs total)
  (h_twice : twice_as_many_rebus R M)
  (h_both : both_teasers both)
  (h_math_only : math_teasers_not_rebus math_only)
  (h_M : M = both + math_only) :
  total - exclusion_inclusion R M both = 4 :=
sorry

end brainiacs_like_neither_l137_137216


namespace log_expression_equality_l137_137902

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  (Real.log 8 / Real.log 4) + 
  2 = 11 / 2 :=
by 
  sorry

end log_expression_equality_l137_137902


namespace product_of_integers_eq_expected_result_l137_137833

theorem product_of_integers_eq_expected_result
  (E F G H I : ℚ) 
  (h1 : E + F + G + H + I = 80) 
  (h2 : E + 2 = F - 2) 
  (h3 : F - 2 = G * 2) 
  (h4 : G * 2 = H * 3) 
  (h5 : H * 3 = I / 2) :
  E * F * G * H * I = (5120000 / 81) := 
by 
  sorry

end product_of_integers_eq_expected_result_l137_137833


namespace future_value_option_B_correct_l137_137606

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end future_value_option_B_correct_l137_137606


namespace trip_total_hours_l137_137381

theorem trip_total_hours
    (x : ℕ) -- additional hours of travel
    (dist_1 : ℕ := 30 * 6) -- distance for first 6 hours
    (dist_2 : ℕ := 46 * x) -- distance for additional hours
    (total_dist : ℕ := dist_1 + dist_2) -- total distance
    (total_time : ℕ := 6 + x) -- total time
    (avg_speed : ℕ := total_dist / total_time) -- average speed
    (h : avg_speed = 34) : total_time = 8 :=
by
  sorry

end trip_total_hours_l137_137381


namespace probability_matching_shoes_l137_137562

theorem probability_matching_shoes :
  let total_shoes := 24;
  let total_pairs := 12;
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2;
  let matching_pairs := total_pairs;
  let probability := matching_pairs / total_combinations;
  probability = 1 / 23 :=
by
  let total_shoes := 24
  let total_pairs := 12
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := total_pairs
  let probability := matching_pairs / total_combinations
  have : total_combinations = 276 := by norm_num
  have : matching_pairs = 12 := by norm_num
  have : probability = 1 / 23 := by norm_num
  exact this

end probability_matching_shoes_l137_137562


namespace find_first_purchase_find_max_profit_purchase_plan_l137_137141

-- Defining the parameters for the problem
structure KeychainParams where
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  total_purchase_cost_first : ℕ
  total_keychains_first : ℕ
  total_purchase_cost_second : ℕ
  total_keychains_second : ℕ
  purchase_cap_second : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ

-- Define the initial setup
def params : KeychainParams := {
  purchase_price_A := 30,
  purchase_price_B := 25,
  total_purchase_cost_first := 850,
  total_keychains_first := 30,
  total_purchase_cost_second := 2200,
  total_keychains_second := 80,
  purchase_cap_second := 2200,
  selling_price_A := 45,
  selling_price_B := 37
}

-- Part 1: Prove the number of keychains purchased for each type
theorem find_first_purchase (x y : ℕ)
  (h₁ : x + y = params.total_keychains_first)
  (h₂ : params.purchase_price_A * x + params.purchase_price_B * y = params.total_purchase_cost_first) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2: Prove the purchase plan that maximizes the sales profit
theorem find_max_profit_purchase_plan (m : ℕ)
  (h₃ : m + (params.total_keychains_second - m) = params.total_keychains_second)
  (h₄ : params.purchase_price_A * m + params.purchase_price_B * (params.total_keychains_second - m) ≤ params.purchase_cap_second) :
  m = 40 ∧ (params.selling_price_A - params.purchase_price_A) * m + (params.selling_price_B - params.purchase_price_B) * (params.total_keychains_second - m) = 1080 :=
sorry

end find_first_purchase_find_max_profit_purchase_plan_l137_137141


namespace pref_card_game_arrangements_l137_137793

noncomputable def number_of_arrangements :=
  (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3)

theorem pref_card_game_arrangements :
  number_of_arrangements = (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3) :=
by
  sorry

end pref_card_game_arrangements_l137_137793


namespace rectangular_field_area_l137_137552

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l137_137552


namespace final_answer_is_d_l137_137295

-- Definitions of the propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x > 1
def q : Prop := false  -- since the distance between focus and directrix is not 1/6 but 3/2

-- The statement to be proven
theorem final_answer_is_d : p ∧ ¬ q := by sorry

end final_answer_is_d_l137_137295


namespace coprime_squares_l137_137198

theorem coprime_squares (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ k : ℕ, ab = k^2) : 
  ∃ p q : ℕ, a = p^2 ∧ b = q^2 :=
by
  sorry

end coprime_squares_l137_137198


namespace range_of_a_l137_137590

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0)) ↔ (1 ≤ a ∧ a ≤ 3 ∨ a = -1) :=
by
  sorry

end range_of_a_l137_137590


namespace initial_cats_in_shelter_l137_137199

theorem initial_cats_in_shelter
  (cats_found_monday : ℕ)
  (cats_found_tuesday : ℕ)
  (cats_adopted_wednesday : ℕ)
  (current_cats : ℕ)
  (total_adopted_cats : ℕ)
  (initial_cats : ℕ) :
  cats_found_monday = 2 →
  cats_found_tuesday = 1 →
  cats_adopted_wednesday = 3 →
  total_adopted_cats = cats_adopted_wednesday * 2 →
  current_cats = 17 →
  initial_cats = current_cats + total_adopted_cats - (cats_found_monday + cats_found_tuesday) →
  initial_cats = 20 :=
by
  intros
  sorry

end initial_cats_in_shelter_l137_137199


namespace josh_marbles_l137_137525

theorem josh_marbles (initial_marbles lost_marbles : ℕ) (h_initial : initial_marbles = 9) (h_lost : lost_marbles = 5) :
  initial_marbles - lost_marbles = 4 :=
by
  sorry

end josh_marbles_l137_137525


namespace perpendicular_lines_sum_is_minus_four_l137_137970

theorem perpendicular_lines_sum_is_minus_four 
  (a b c : ℝ) 
  (h1 : (a * 2) / (4 * 5) = 1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * (-2) + b = 0) : 
  a + b + c = -4 := 
sorry

end perpendicular_lines_sum_is_minus_four_l137_137970


namespace remainder_4x_mod_7_l137_137593

theorem remainder_4x_mod_7 (x : ℤ) (k : ℤ) (h : x = 7 * k + 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_4x_mod_7_l137_137593


namespace solve_for_a_l137_137019

theorem solve_for_a (a : ℕ) (h : a > 0) (eqn : a / (a + 37) = 925 / 1000) : a = 455 :=
sorry

end solve_for_a_l137_137019


namespace find_minimum_value_max_value_when_g_half_l137_137223

noncomputable def f (a x : ℝ) : ℝ := 1 - 2 * a - 2 * a * (Real.cos x) - 2 * (Real.sin x) ^ 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a <= 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem find_minimum_value (a : ℝ) :
  ∃ g_val, g_val = g a :=
  sorry

theorem max_value_when_g_half : 
  g (-1) = 1 / 2 →
  ∃ max_val, max_val = (max (f (-1) π) (f (-1) 0)) :=
  sorry

end find_minimum_value_max_value_when_g_half_l137_137223


namespace valid_parameterizations_l137_137213

def point_on_line (x y : ℝ) : Prop := (y = 2 * x - 5)

def direction_vector_valid (vx vy : ℝ) : Prop := (∃ (k : ℝ), vx = k * 1 ∧ vy = k * 2)

def parametric_option_valid (px py vx vy : ℝ) : Prop := 
  point_on_line px py ∧ direction_vector_valid vx vy

theorem valid_parameterizations : 
  (parametric_option_valid 10 15 5 10) ∧ 
  (parametric_option_valid 3 1 0.5 1) ∧ 
  (parametric_option_valid 7 9 2 4) ∧ 
  (parametric_option_valid 0 (-5) 10 20) :=
  by sorry

end valid_parameterizations_l137_137213


namespace donation_to_second_home_l137_137311

-- Definitions of the conditions
def total_donation := 700.00
def first_home_donation := 245.00
def third_home_donation := 230.00

-- Define the unknown donation to the second home
noncomputable def second_home_donation := total_donation - first_home_donation - third_home_donation

-- The theorem to prove
theorem donation_to_second_home :
  second_home_donation = 225.00 :=
by sorry

end donation_to_second_home_l137_137311


namespace total_age_10_years_from_now_is_75_l137_137877

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end total_age_10_years_from_now_is_75_l137_137877


namespace single_elimination_tournament_l137_137482

theorem single_elimination_tournament (teams : ℕ) (prelim_games : ℕ) (post_prelim_teams : ℕ) :
  teams = 24 →
  prelim_games = 4 →
  post_prelim_teams = teams - prelim_games →
  post_prelim_teams - 1 + prelim_games = 23 :=
by
  intros
  sorry

end single_elimination_tournament_l137_137482


namespace who_stole_the_broth_l137_137191

-- Define the suspects
inductive Suspect
| MarchHare : Suspect
| MadHatter : Suspect
| Dormouse : Suspect

open Suspect

-- Define the statements
def stole_broth (s : Suspect) : Prop :=
  s = Dormouse

def told_truth (s : Suspect) : Prop :=
  s = Dormouse

-- The March Hare's testimony
def march_hare_testimony : Prop :=
  stole_broth MadHatter

-- Conditions
def condition1 : Prop := ∃! s, stole_broth s
def condition2 : Prop := ∀ s, told_truth s ↔ stole_broth s
def condition3 : Prop := told_truth MarchHare → stole_broth MadHatter

-- Combining conditions into a single proposition to prove
theorem who_stole_the_broth : 
  (condition1 ∧ condition2 ∧ condition3) → stole_broth Dormouse := sorry

end who_stole_the_broth_l137_137191


namespace percentage_neither_language_l137_137270

def total_diplomats : ℕ := 150
def french_speaking : ℕ := 17
def russian_speaking : ℕ := total_diplomats - 32
def both_languages : ℕ := 10 * total_diplomats / 100

theorem percentage_neither_language :
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  neither_language * 100 / total_diplomats = 20 :=
by
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  sorry

end percentage_neither_language_l137_137270


namespace valid_parameterizations_l137_137069

-- Define the parameterization as a structure
structure LineParameterization where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

-- Define the line equation
def line_eq (p : ℝ × ℝ) : Prop :=
  p.snd = -(2/3) * p.fst + 4

-- Proving which parameterizations are valid
theorem valid_parameterizations :
  (line_eq (3 + t * 3, 4 + t * (-2)) ∧
   line_eq (0 + t * 1.5, 4 + t * (-1)) ∧
   line_eq (1 + t * (-6), 3.33 + t * 4) ∧
   line_eq (5 + t * 1.5, (2/3) + t * (-1)) ∧
   line_eq (-6 + t * 9, 8 + t * (-6))) = 
  false ∧ true ∧ false ∧ true ∧ false :=
by
  sorry

end valid_parameterizations_l137_137069


namespace proportion1_proportion2_l137_137619

theorem proportion1 (x : ℚ) : (x / (5 / 9) = (1 / 20) / (1 / 3)) → x = 1 / 12 :=
sorry

theorem proportion2 (x : ℚ) : (x / 0.25 = 0.5 / 0.1) → x = 1.25 :=
sorry

end proportion1_proportion2_l137_137619


namespace lemons_needed_l137_137447

theorem lemons_needed (lemons32 : ℕ) (lemons4 : ℕ) (h1 : lemons32 = 24) (h2 : (24 : ℕ) / 32 = (lemons4 : ℕ) / 4) : lemons4 = 3 := 
sorry

end lemons_needed_l137_137447


namespace harper_jack_distance_apart_l137_137566

def total_distance : ℕ := 1000
def distance_jack_run : ℕ := 152
def distance_apart (total_distance : ℕ) (distance_jack_run : ℕ) : ℕ :=
  total_distance - distance_jack_run 

theorem harper_jack_distance_apart :
  distance_apart total_distance distance_jack_run = 848 :=
by
  unfold distance_apart
  sorry

end harper_jack_distance_apart_l137_137566


namespace sample_size_is_30_l137_137092

-- Definitions based on conditions
def total_students : ℕ := 700 + 500 + 300
def students_first_grade : ℕ := 700
def students_sampled_first_grade : ℕ := 14
def sample_size (n : ℕ) : Prop := students_sampled_first_grade = (students_first_grade * n) / total_students

-- Theorem stating the proof problem
theorem sample_size_is_30 : sample_size 30 :=
by
  sorry

end sample_size_is_30_l137_137092


namespace converse_inverse_l137_137595

-- Define the properties
def is_parallelogram (polygon : Type) : Prop := sorry -- needs definitions about polygons
def has_two_pairs_of_parallel_sides (polygon : Type) : Prop := sorry -- needs definitions about polygons

-- The given condition
axiom parallelogram_implies_parallel_sides (polygon : Type) :
  is_parallelogram polygon → has_two_pairs_of_parallel_sides polygon

-- Proof of the converse:
theorem converse (polygon : Type) :
  has_two_pairs_of_parallel_sides polygon → is_parallelogram polygon := sorry

-- Proof of the inverse:
theorem inverse (polygon : Type) :
  ¬is_parallelogram polygon → ¬has_two_pairs_of_parallel_sides polygon := sorry

end converse_inverse_l137_137595


namespace molly_ate_11_suckers_l137_137668

/-- 
Sienna gave Bailey half of her suckers.
Jen ate 11 suckers and gave the rest to Molly.
Molly ate some suckers and gave the rest to Harmony.
Harmony kept 3 suckers and passed the remainder to Taylor.
Taylor ate one and gave the last 5 suckers to Callie.
How many suckers did Molly eat?
-/
theorem molly_ate_11_suckers
  (sienna_bailey_suckers : ℕ)
  (jen_ate : ℕ)
  (jens_remainder_to_molly : ℕ)
  (molly_remainder_to_harmony : ℕ) 
  (harmony_kept : ℕ) 
  (harmony_remainder_to_taylor : ℕ)
  (taylor_ate : ℕ)
  (taylor_remainder_to_callie : ℕ)
  (jen_condition : jen_ate = 11)
  (harmony_condition : harmony_kept = 3)
  (taylor_condition : taylor_ate = 1)
  (taylor_final_suckers : taylor_remainder_to_callie = 5) :
  molly_ate = 11 :=
by sorry

end molly_ate_11_suckers_l137_137668


namespace total_number_of_balls_l137_137222

-- Define the conditions
def balls_per_box : Nat := 3
def number_of_boxes : Nat := 2

-- Define the proposition
theorem total_number_of_balls : (balls_per_box * number_of_boxes) = 6 :=
by
  sorry

end total_number_of_balls_l137_137222


namespace simplify_expression_l137_137209

theorem simplify_expression :
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) := by
  sorry

end simplify_expression_l137_137209


namespace bottle_caps_remaining_l137_137638

-- Define the problem using the conditions and the desired proof.
theorem bottle_caps_remaining (original_count removed_count remaining_count : ℕ) 
    (h_original : original_count = 87) 
    (h_removed : removed_count = 47)
    (h_remaining : remaining_count = original_count - removed_count) :
    remaining_count = 40 :=
by 
  rw [h_original, h_removed] at h_remaining 
  exact h_remaining

end bottle_caps_remaining_l137_137638


namespace D_96_equals_112_l137_137884

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end D_96_equals_112_l137_137884


namespace max_sum_of_triplet_product_60_l137_137769

theorem max_sum_of_triplet_product_60 : 
  ∃ a b c : ℕ, a * b * c = 60 ∧ a + b + c = 62 :=
sorry

end max_sum_of_triplet_product_60_l137_137769


namespace concentration_sequences_and_min_operations_l137_137756

theorem concentration_sequences_and_min_operations :
  (a_1 = 1.55 ∧ b_1 = 0.65) ∧
  (∀ n ≥ 1, a_n - b_n = 0.9 * (1 / 2)^(n - 1)) ∧
  (∃ n, 0.9 * (1 / 2)^(n - 1) < 0.01 ∧ n = 8) :=
by
  sorry

end concentration_sequences_and_min_operations_l137_137756


namespace find_b_l137_137789

-- Definitions for conditions
def eq1 (a : ℤ) : Prop := 2 * a + 1 = 1
def eq2 (a b : ℤ) : Prop := 2 * b - 3 * a = 2

-- The theorem statement
theorem find_b (a b : ℤ) (h1 : eq1 a) (h2 : eq2 a b) : b = 1 :=
  sorry  -- Proof to be filled in.

end find_b_l137_137789


namespace graph_does_not_pass_through_third_quadrant_l137_137771

theorem graph_does_not_pass_through_third_quadrant (k x y : ℝ) (hk : k < 0) :
  y = k * x - k → (¬ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_does_not_pass_through_third_quadrant_l137_137771


namespace total_questions_solved_l137_137106

-- Define the number of questions Taeyeon solved in a day and the number of days
def Taeyeon_questions_per_day : ℕ := 16
def Taeyeon_days : ℕ := 7

-- Define the number of questions Yura solved in a day and the number of days
def Yura_questions_per_day : ℕ := 25
def Yura_days : ℕ := 6

-- Define the total number of questions Taeyeon and Yura solved
def Total_questions_Taeyeon : ℕ := Taeyeon_questions_per_day * Taeyeon_days
def Total_questions_Yura : ℕ := Yura_questions_per_day * Yura_days
def Total_questions : ℕ := Total_questions_Taeyeon + Total_questions_Yura

-- Prove that the total number of questions solved by Taeyeon and Yura is 262
theorem total_questions_solved : Total_questions = 262 := by
  sorry

end total_questions_solved_l137_137106


namespace compare_probabilities_l137_137318

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l137_137318


namespace positional_relationship_l137_137634

theorem positional_relationship (r PO QO : ℝ) (h_r : r = 6) (h_PO : PO = 4) (h_QO : QO = 6) :
  (PO < r) ∧ (QO = r) :=
by
  sorry

end positional_relationship_l137_137634


namespace problem1_problem2_problem3_l137_137995

def is_real (m : ℝ) : Prop := (m^2 - 3 * m) = 0
def is_complex (m : ℝ) : Prop := (m^2 - 3 * m) ≠ 0
def is_pure_imaginary (m : ℝ) : Prop := (m^2 - 5 * m + 6) = 0 ∧ (m^2 - 3 * m) ≠ 0

theorem problem1 (m : ℝ) : is_real m ↔ (m = 0 ∨ m = 3) :=
sorry

theorem problem2 (m : ℝ) : is_complex m ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

theorem problem3 (m : ℝ) : is_pure_imaginary m ↔ (m = 2) :=
sorry

end problem1_problem2_problem3_l137_137995


namespace square_of_rational_l137_137407

theorem square_of_rational (b : ℚ) : b^2 = b * b :=
sorry

end square_of_rational_l137_137407


namespace math_score_computation_l137_137754

def comprehensive_score 
  (reg_score : ℕ) (mid_score : ℕ) (fin_score : ℕ) 
  (reg_weight : ℕ) (mid_weight : ℕ) (fin_weight : ℕ) 
  : ℕ :=
  (reg_score * reg_weight + mid_score * mid_weight + fin_score * fin_weight) 
  / (reg_weight + mid_weight + fin_weight)

theorem math_score_computation :
  comprehensive_score 80 80 85 3 3 4 = 82 := by
sorry

end math_score_computation_l137_137754


namespace sum_of_series_l137_137426

open BigOperators

-- Define the sequence a(n) = 2 / (n * (n + 3))
def a (n : ℕ) : ℚ := 2 / (n * (n + 3))

-- Prove the sum of the first 20 terms of sequence a equals 10 / 9.
theorem sum_of_series : (∑ n in Finset.range 20, a (n + 1)) = 10 / 9 := by
  sorry

end sum_of_series_l137_137426


namespace Amelia_weekly_sales_l137_137530

-- Conditions
def monday_sales : ℕ := 45
def tuesday_sales : ℕ := 45 - 16
def remaining_sales : ℕ := 16

-- Question to Answer
def total_weekly_sales : ℕ := 90

-- Lean 4 Statement to Prove
theorem Amelia_weekly_sales : monday_sales + tuesday_sales + remaining_sales = total_weekly_sales :=
by
  sorry

end Amelia_weekly_sales_l137_137530


namespace negation_of_prop_l137_137719

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop_l137_137719


namespace least_amount_to_add_l137_137151

theorem least_amount_to_add (current_amount : ℕ) (n : ℕ) (divisor : ℕ) [NeZero divisor]
  (current_amount_eq : current_amount = 449774) (n_eq : n = 1) (divisor_eq : divisor = 6) :
  ∃ k : ℕ, (current_amount + k) % divisor = 0 ∧ k = n := by
  sorry

end least_amount_to_add_l137_137151


namespace median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l137_137706

-- Definition of points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- The problem statements as Lean theorems
theorem median_on_AB_eq : ∀ (A B : ℝ × ℝ), A = (4, 0) ∧ B = (6, 7) → ∃ (x y : ℝ), x - 10 * y + 30 = 0 := by
  intros
  sorry

theorem altitude_on_BC_eq : ∀ (B C : ℝ × ℝ), B = (6, 7) ∧ C = (0, 3) → ∃ (x y : ℝ), 3 * x + 2 * y - 12 = 0 := by
  intros
  sorry

theorem perp_bisector_on_AC_eq : ∀ (A C : ℝ × ℝ), A = (4, 0) ∧ C = (0, 3) → ∃ (x y : ℝ), 8 * x - 6 * y - 7 = 0 := by
  intros
  sorry

end median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l137_137706


namespace sequence_solution_l137_137172

theorem sequence_solution :
  ∀ (a : ℕ → ℝ), (∀ m n : ℕ, a (m^2 + n^2) = a m ^ 2 + a n ^ 2) →
  (0 ≤ a 0 ∧ a 0 ≤ a 1 ∧ a 1 ≤ a 2 ∧ ∀ n, a n ≤ a (n + 1)) →
  (∀ n, a n = 0) ∨ (∀ n, a n = n) ∨ (∀ n, a n = 1 / 2) :=
sorry

end sequence_solution_l137_137172


namespace positive_integral_solution_l137_137722

theorem positive_integral_solution (n : ℕ) (hn : 0 < n) 
  (h : (n : ℚ) / (n + 1) = 125 / 126) : n = 125 := sorry

end positive_integral_solution_l137_137722


namespace square_presses_exceed_1000_l137_137397

theorem square_presses_exceed_1000:
  ∃ n : ℕ, (n = 3) ∧ (3 ^ (2^n) > 1000) :=
by
  sorry

end square_presses_exceed_1000_l137_137397


namespace simon_legos_l137_137863

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l137_137863


namespace remaining_stickers_l137_137389

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l137_137389


namespace find_a_l137_137750

theorem find_a (a b : ℝ) (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) (h4 : a + b = 4) : a = -1 :=
by 
sorry

end find_a_l137_137750


namespace negation_equiv_exists_l137_137419

theorem negation_equiv_exists : 
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 0 := 
by 
  sorry

end negation_equiv_exists_l137_137419


namespace frank_has_4_five_dollar_bills_l137_137932

theorem frank_has_4_five_dollar_bills
    (one_dollar_bills : ℕ := 7)
    (ten_dollar_bills : ℕ := 2)
    (twenty_dollar_bills : ℕ := 1)
    (change : ℕ := 4)
    (peanut_cost_per_pound : ℕ := 3)
    (days_in_week : ℕ := 7)
    (peanuts_per_day : ℕ := 3) :
    let initial_amount := (one_dollar_bills * 1) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)
    let total_peanuts_cost := (peanuts_per_day * days_in_week) * peanut_cost_per_pound
    let F := (total_peanuts_cost + change - initial_amount) / 5 
    F = 4 :=
by
  repeat { admit }


end frank_has_4_five_dollar_bills_l137_137932


namespace person_income_l137_137804

/-- If the income and expenditure of a person are in the ratio 15:8 and the savings are Rs. 7000, then the income of the person is Rs. 15000. -/
theorem person_income (x : ℝ) (income expenditure : ℝ) (savings : ℝ) 
  (h1 : income = 15 * x) 
  (h2 : expenditure = 8 * x) 
  (h3 : savings = income - expenditure) 
  (h4 : savings = 7000) : 
  income = 15000 := 
by 
  sorry

end person_income_l137_137804


namespace differential_savings_l137_137314

def annual_income_before_tax : ℝ := 42400
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.32

theorem differential_savings :
  annual_income_before_tax * initial_tax_rate - annual_income_before_tax * new_tax_rate = 4240 :=
by
  sorry

end differential_savings_l137_137314


namespace parallelogram_base_l137_137050

theorem parallelogram_base (height area : ℕ) (h_height : height = 18) (h_area : area = 612) : ∃ base, base = 34 :=
by
  -- The proof would go here
  sorry

end parallelogram_base_l137_137050


namespace evaluate_expression_l137_137073

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l137_137073


namespace no_prime_divisible_by_45_l137_137149

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l137_137149


namespace second_player_wins_12_petals_second_player_wins_11_petals_l137_137997

def daisy_game (n : Nat) : Prop :=
  ∀ (p1_move p2_move : Nat → Nat → Prop), n % 2 = 0 → (∃ k, p1_move n k = false) ∧ (∃ ℓ, p2_move n ℓ = true)

theorem second_player_wins_12_petals : daisy_game 12 := sorry
theorem second_player_wins_11_petals : daisy_game 11 := sorry

end second_player_wins_12_petals_second_player_wins_11_petals_l137_137997


namespace program_result_l137_137780

def program_loop (i : ℕ) (s : ℕ) : ℕ :=
if i < 9 then s else program_loop (i - 1) (s * i)

theorem program_result : 
  program_loop 11 1 = 990 :=
by 
  sorry

end program_result_l137_137780


namespace sector_angle_measure_l137_137138

theorem sector_angle_measure
  (r l : ℝ)
  (h1 : 2 * r + l = 4)
  (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 :=
sorry

end sector_angle_measure_l137_137138


namespace abs_diff_eq_l137_137559

theorem abs_diff_eq (a b c d : ℤ) (h1 : a = 13) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  |a - b| - |c - d| = 4 := 
  by
  -- Proof goes here
  sorry

end abs_diff_eq_l137_137559


namespace number_of_students_l137_137186

-- Define the conditions as hypotheses
def ordered_apples : ℕ := 6 + 15   -- 21 apples ordered
def extra_apples : ℕ := 16         -- 16 extra apples after distribution

-- Define the main theorem statement to prove S = 21
theorem number_of_students (S : ℕ) (H1 : ordered_apples = 21) (H2 : extra_apples = 16) : S = 21 := 
by
  sorry

end number_of_students_l137_137186


namespace quadratic_no_real_roots_l137_137324

theorem quadratic_no_real_roots (m : ℝ) (h : ∀ x : ℝ, x^2 - m * x + 1 ≠ 0) : m = 0 :=
by
  sorry

end quadratic_no_real_roots_l137_137324


namespace sum_of_squares_of_consecutive_integers_l137_137736

-- The sum of the squares of three consecutive positive integers equals 770.
-- We aim to prove that the largest integer among them is 17.
theorem sum_of_squares_of_consecutive_integers (n : ℕ) (h_pos : n > 0) 
    (h_sum : (n-1)^2 + n^2 + (n+1)^2 = 770) : n + 1 = 17 :=
sorry

end sum_of_squares_of_consecutive_integers_l137_137736


namespace roots_of_polynomial_l137_137680

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l137_137680


namespace larger_integer_is_neg4_l137_137938

-- Definitions of the integers used in the problem
variables (x y : ℤ)

-- Conditions given in the problem
def condition1 : x + y = -9 := sorry
def condition2 : x - y = 1 := sorry

-- The theorem to prove
theorem larger_integer_is_neg4 (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := 
sorry

end larger_integer_is_neg4_l137_137938


namespace solve_equation_l137_137194

theorem solve_equation (x y z : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9) (h_eq : 1 / (x + y + z) = (x * 100 + y * 10 + z) / 1000) :
  x = 1 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end solve_equation_l137_137194


namespace mean_of_remaining_four_numbers_l137_137094

theorem mean_of_remaining_four_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 → (a + b + c + d) / 4 = 88.75 :=
by
  intro h
  sorry

end mean_of_remaining_four_numbers_l137_137094


namespace eight_p_plus_one_composite_l137_137478

theorem eight_p_plus_one_composite 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h8p_minus_one : Nat.Prime (8 * p - 1))
  : ¬ (Nat.Prime (8 * p + 1)) :=
sorry

end eight_p_plus_one_composite_l137_137478


namespace equivalent_sum_of_exponents_l137_137713

theorem equivalent_sum_of_exponents : 3^3 + 3^3 + 3^3 = 3^4 :=
by
  sorry

end equivalent_sum_of_exponents_l137_137713


namespace find_angle_4_l137_137231

/-- Given angle conditions, prove that angle 4 is 22.5 degrees. -/
theorem find_angle_4 (angle : ℕ → ℝ) 
  (h1 : angle 1 + angle 2 = 180) 
  (h2 : angle 3 = angle 4) 
  (h3 : angle 1 = 85) 
  (h4 : angle 5 = 45) 
  (h5 : angle 1 + angle 5 + angle 6 = 180) : 
  angle 4 = 22.5 :=
sorry

end find_angle_4_l137_137231


namespace number_of_ordered_pairs_l137_137368

-- Define the predicate that defines the condition for the ordered pairs (m, n)
def satisfies_condition (m n : ℕ) : Prop :=
  6 % m = 0 ∧ 3 % n = 0 ∧ 6 / m + 3 / n = 1

-- Define the main theorem for the problem statement
theorem number_of_ordered_pairs : 
  (∃ count : ℕ, count = 6 ∧ 
  (∀ m n : ℕ, satisfies_condition m n → m > 0 ∧ n > 0)) :=
by {
 sorry -- Placeholder for the actual proof
}

end number_of_ordered_pairs_l137_137368


namespace intersection_A_B_union_B_complement_A_l137_137258

open Set

variable (U A B : Set ℝ)

noncomputable def U_def : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
noncomputable def A_def : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def B_def : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : (A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
sorry

theorem union_B_complement_A : B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)} :=
sorry

attribute [irreducible] U_def A_def B_def

end intersection_A_B_union_B_complement_A_l137_137258


namespace citizen_income_l137_137551

theorem citizen_income (I : ℝ) (h1 : ∀ I ≤ 40000, 0.15 * I = 8000) 
  (h2 : ∀ I > 40000, (0.15 * 40000 + 0.20 * (I - 40000)) = 8000) : 
  I = 50000 :=
by
  sorry

end citizen_income_l137_137551


namespace solve_inequalities_solve_linear_system_l137_137880

-- System of Inequalities
theorem solve_inequalities (x : ℝ) (h1 : x + 2 > 1) (h2 : 2 * x < x + 3) : -1 < x ∧ x < 3 :=
by
  sorry

-- System of Linear Equations
theorem solve_linear_system (x y : ℝ) (h1 : 3 * x + 2 * y = 12) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_inequalities_solve_linear_system_l137_137880


namespace sequence_periodicity_l137_137476

theorem sequence_periodicity (a : ℕ → ℝ) (h₁ : ∀ n, a (n + 1) = 1 / (1 - a n)) (h₂ : a 8 = 2) :
  a 1 = 1 / 2 := 
sorry

end sequence_periodicity_l137_137476


namespace max_k_value_l137_137569

noncomputable def max_k (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : ℝ :=
  let k := (-1 + Real.sqrt 7) / 2
  k

theorem max_k_value (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_k_value_l137_137569


namespace correct_division_algorithm_l137_137114

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end correct_division_algorithm_l137_137114


namespace sum_of_numbers_l137_137378

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149)
  (h2 : ab + bc + ca = 70) : 
  a + b + c = 17 :=
sorry

end sum_of_numbers_l137_137378


namespace x_y_z_sum_l137_137384

theorem x_y_z_sum :
  ∃ (x y z : ℕ), (16 / 3)^x * (27 / 25)^y * (5 / 4)^z = 256 ∧ x + y + z = 6 :=
by
  -- Proof can be completed here
  sorry

end x_y_z_sum_l137_137384


namespace max_value_amc_am_mc_ca_l137_137147

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l137_137147


namespace choir_group_students_l137_137596

theorem choir_group_students : ∃ n : ℕ, (n % 5 = 0) ∧ (n % 9 = 0) ∧ (n % 12 = 0) ∧ (∃ m : ℕ, n = m * m) ∧ n ≥ 360 := 
sorry

end choir_group_students_l137_137596


namespace units_digit_sum_42_4_24_4_l137_137523

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l137_137523


namespace find_value_of_expression_l137_137838

variable (α β : ℝ)

-- Defining the conditions
def is_root (α : ℝ) : Prop := α^2 - 3 * α + 1 = 0
def add_roots_eq (α β : ℝ) : Prop := α + β = 3
def mult_roots_eq (α β : ℝ) : Prop := α * β = 1

-- The main statement we want to prove
theorem find_value_of_expression {α β : ℝ} 
  (hα : is_root α) 
  (hβ : is_root β)
  (h_add : add_roots_eq α β)
  (h_mul : mult_roots_eq α β) :
  3 * α^5 + 7 * β^4 = 817 := 
sorry

end find_value_of_expression_l137_137838


namespace cyclic_quadrilateral_area_l137_137862

variable (a b c d R : ℝ)
noncomputable def p : ℝ := (a + b + c + d) / 2
noncomputable def Brahmagupta_area : ℝ := Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem cyclic_quadrilateral_area :
  Brahmagupta_area a b c d = Real.sqrt ((a * b + c * d) * (a * d + b * c) * (a * c + b * d)) / (4 * R) := sorry

end cyclic_quadrilateral_area_l137_137862


namespace mean_age_gauss_family_l137_137066

theorem mean_age_gauss_family :
  let ages := [7, 7, 7, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  mean_age = 10 :=
by
  sorry

end mean_age_gauss_family_l137_137066


namespace find_divisor_exists_four_numbers_in_range_l137_137250

theorem find_divisor_exists_four_numbers_in_range :
  ∃ n : ℕ, (n > 1) ∧ (∀ k : ℕ, 39 ≤ k ∧ k ≤ 79 → ∃ a : ℕ, k = n * a) ∧ (∃! (k₁ k₂ k₃ k₄ : ℕ), 39 ≤ k₁ ∧ k₁ ≤ 79 ∧ 39 ≤ k₂ ∧ k₂ ≤ 79 ∧ 39 ≤ k₃ ∧ k₃ ≤ 79 ∧ 39 ≤ k₄ ∧ k₄ ≤ 79 ∧ k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧ k₁ % n = 0 ∧ k₂ % n = 0 ∧ k₃ % n = 0 ∧ k₄ % n = 0) → n = 19 :=
by sorry

end find_divisor_exists_four_numbers_in_range_l137_137250


namespace calculate_markup_percentage_l137_137739

noncomputable def cost_price : ℝ := 225
noncomputable def profit_percentage : ℝ := 0.25
noncomputable def discount1_percentage : ℝ := 0.10
noncomputable def discount2_percentage : ℝ := 0.15
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)
noncomputable def markup_percentage : ℝ := 63.54

theorem calculate_markup_percentage :
  let marked_price := selling_price / ((1 - discount1_percentage) * (1 - discount2_percentage))
  let calculated_markup_percentage := ((marked_price - cost_price) / cost_price) * 100
  abs (calculated_markup_percentage - markup_percentage) < 0.01 :=
sorry

end calculate_markup_percentage_l137_137739


namespace decimal_multiplication_l137_137433

theorem decimal_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by sorry

end decimal_multiplication_l137_137433


namespace log_expression_value_l137_137333

theorem log_expression_value :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 :=
by
  sorry

end log_expression_value_l137_137333


namespace matthew_initial_crackers_l137_137578

theorem matthew_initial_crackers :
  ∃ C : ℕ,
  (∀ (crackers_per_friend cakes_per_friend : ℕ), cakes_per_friend * 4 = 98 → crackers_per_friend = cakes_per_friend → crackers_per_friend * 4 + 8 * 4 = C) ∧ C = 128 :=
sorry

end matthew_initial_crackers_l137_137578


namespace sum_of_three_different_squares_l137_137217

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def existing_list (ns : List Nat) : Prop :=
  ∀ n ∈ ns, is_perfect_square n

theorem sum_of_three_different_squares (a b c : Nat) :
  existing_list [a, b, c] →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 128 →
  false :=
by
  intros
  sorry

end sum_of_three_different_squares_l137_137217


namespace reduced_price_of_oil_is_40_l137_137850

variables 
  (P R : ℝ) 
  (hP : 0 < P)
  (hR : R = 0.75 * P)
  (hw : 800 / (0.75 * P) = 800 / P + 5)

theorem reduced_price_of_oil_is_40 : R = 40 :=
sorry

end reduced_price_of_oil_is_40_l137_137850


namespace problem_gets_solved_prob_l137_137132

-- Define conditions for probabilities
def P_A_solves := 2 / 3
def P_B_solves := 3 / 4

-- Calculate the probability that the problem is solved
theorem problem_gets_solved_prob :
  let P_A_not_solves := 1 - P_A_solves
  let P_B_not_solves := 1 - P_B_solves
  let P_both_not_solve := P_A_not_solves * P_B_not_solves
  let P_solved := 1 - P_both_not_solve
  P_solved = 11 / 12 :=
by
  -- Skip proof
  sorry

end problem_gets_solved_prob_l137_137132


namespace solve_trigonometric_inequality_l137_137030

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0

theorem solve_trigonometric_inequality :
  ∀ x, x ∈ Set.Ioo 0 (2 * Real.pi) → (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0 ↔ x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3)) :=
by
  intros x hx
  sorry

end solve_trigonometric_inequality_l137_137030


namespace range_of_a_l137_137042

variable {α : Type*}

def in_interval (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | in_interval x 0 1}

theorem range_of_a (a : ℝ) (hA_B_nonempty : (A a ∩ B).Nonempty) : 0 < a ∧ a < 1 := 
sorry

end range_of_a_l137_137042


namespace geometric_sequence_common_ratio_l137_137502

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 = 36)
  (h2 : a 4 = 54)
  (h_pos : ∀ n, a n > 0) :
  ∃ q, q > 0 ∧ ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l137_137502


namespace percent_flamingos_among_non_parrots_l137_137249

theorem percent_flamingos_among_non_parrots
  (total_birds : ℝ) (flamingos : ℝ) (parrots : ℝ) (eagles : ℝ) (owls : ℝ)
  (h_total : total_birds = 100)
  (h_flamingos : flamingos = 40)
  (h_parrots : parrots = 20)
  (h_eagles : eagles = 15)
  (h_owls : owls = 25) :
  ((flamingos / (total_birds - parrots)) * 100 = 50) :=
by sorry

end percent_flamingos_among_non_parrots_l137_137249


namespace total_students_in_school_l137_137371

-- Definitions and conditions
def number_of_blind_students (B : ℕ) : Prop := ∃ B, 3 * B = 180
def number_of_other_disabilities (O : ℕ) (B : ℕ) : Prop := O = 2 * B
def total_students (T : ℕ) (D : ℕ) (B : ℕ) (O : ℕ) : Prop := T = D + B + O

theorem total_students_in_school : 
  ∃ (T B O : ℕ), number_of_blind_students B ∧ 
                 number_of_other_disabilities O B ∧ 
                 total_students T 180 B O ∧ 
                 T = 360 :=
by
  sorry

end total_students_in_school_l137_137371


namespace maximum_area_of_rectangle_with_fixed_perimeter_l137_137615

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l137_137615


namespace value_of_abc_l137_137418

-- Conditions
def cond1 (a b : ℤ) : Prop := ∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)
def cond2 (b c : ℤ) : Prop := ∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)

-- Theorem statement
theorem value_of_abc (a b c : ℤ) (h₁ : cond1 a b) (h₂ : cond2 b c) : a + b + c = 31 :=
sorry

end value_of_abc_l137_137418


namespace incorrect_statement_l137_137622

def consecutive_interior_angles_are_supplementary (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 180 → l1 = l2

def alternate_interior_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def corresponding_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def complementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 90

def supplementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 180

theorem incorrect_statement :
  ¬ (∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2) →
    consecutive_interior_angles_are_supplementary l1 l2 →
    alternate_interior_angles_are_equal l1 l2 →
    corresponding_angles_are_equal l1 l2 →
    (∀ (θ₁ θ₂ : ℝ), supplementary_angles θ₁ θ₂) →
    (∀ (θ₁ θ₂ : ℝ), complementary_angles θ₁ θ₂) :=
sorry

end incorrect_statement_l137_137622


namespace complete_square_l137_137942

theorem complete_square (a b c : ℕ) (h : 49 * x ^ 2 + 70 * x - 121 = 0) :
  a = 7 ∧ b = 5 ∧ c = 146 ∧ a + b + c = 158 :=
by sorry

end complete_square_l137_137942


namespace sale_in_fifth_month_l137_137658

theorem sale_in_fifth_month (Sale1 Sale2 Sale3 Sale4 Sale6 AvgSale : ℤ) 
(h1 : Sale1 = 6435) (h2 : Sale2 = 6927) (h3 : Sale3 = 6855) (h4 : Sale4 = 7230) 
(h5 : Sale6 = 4991) (h6 : AvgSale = 6500) : (39000 - (Sale1 + Sale2 + Sale3 + Sale4 + Sale6)) = 6562 :=
by
  sorry

end sale_in_fifth_month_l137_137658


namespace abs_neg_three_l137_137686

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l137_137686


namespace quadrilateral_area_is_22_5_l137_137331

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (3, -1)
def vertex2 : ℝ × ℝ := (-1, 4)
def vertex3 : ℝ × ℝ := (2, 3)
def vertex4 : ℝ × ℝ := (9, 9)

-- Define the function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  0.5 * (abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) 
        - (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)))

-- State that the area of the quadrilateral with given vertices is 22.5
theorem quadrilateral_area_is_22_5 :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 22.5 :=
by 
  -- We skip the proof here.
  sorry

end quadrilateral_area_is_22_5_l137_137331


namespace induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l137_137930

open Nat

theorem induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25 :
  ∀ n : ℕ, n > 0 → 25 ∣ (2^(n+2) * 3^n + 5*n - 4) :=
by
  intro n hn
  sorry

end induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l137_137930


namespace total_pebbles_l137_137701

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end total_pebbles_l137_137701


namespace sin_identity_l137_137208

theorem sin_identity {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.sin (π / 6 - 2 * α) = -7 / 8 := 
by 
  sorry

end sin_identity_l137_137208


namespace evaluate_expression_l137_137398

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 :=
by
  sorry

end evaluate_expression_l137_137398


namespace correct_operations_l137_137651

theorem correct_operations : 
  (∀ x y : ℝ, x^2 + x^4 ≠ x^6) ∧
  (∀ x y : ℝ, 2*x + 4*y ≠ 6*x*y) ∧
  (∀ x : ℝ, x^6 / x^3 = x^3) ∧
  (∀ x : ℝ, (x^3)^2 = x^6) :=
by 
  sorry

end correct_operations_l137_137651


namespace map_scale_l137_137500

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l137_137500


namespace maximum_b_value_l137_137623

noncomputable def f (x : ℝ) := Real.exp x - x - 1
def g (x : ℝ) := -x^2 + 4 * x - 3

theorem maximum_b_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : f a = g b) : b ≤ 3 := by
  sorry

end maximum_b_value_l137_137623


namespace lambs_total_l137_137681

/-
Each of farmer Cunningham's lambs is either black or white.
There are 193 white lambs, and 5855 black lambs.
Prove that the total number of lambs is 6048.
-/

theorem lambs_total (white_lambs : ℕ) (black_lambs : ℕ) (h1 : white_lambs = 193) (h2 : black_lambs = 5855) :
  white_lambs + black_lambs = 6048 :=
by
  -- proof goes here
  sorry

end lambs_total_l137_137681


namespace min_distance_from_start_after_9_minutes_l137_137486

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l137_137486


namespace triangle_area_eq_l137_137747

variable (a b c: ℝ) (A B C : ℝ)
variable (h_cosC : Real.cos C = 1/4)
variable (h_c : c = 3)
variable (h_ratio : a / Real.cos A = b / Real.cos B)

theorem triangle_area_eq : (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 :=
by
  sorry

end triangle_area_eq_l137_137747


namespace tan_11_pi_over_4_l137_137084

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l137_137084


namespace trip_drop_probability_l137_137275

-- Definitions
def P_Trip : ℝ := 0.4
def P_Drop_not : ℝ := 0.9

-- Main theorem
theorem trip_drop_probability : ∀ (P_Trip P_Drop_not : ℝ), P_Trip = 0.4 → P_Drop_not = 0.9 → 1 - P_Drop_not = 0.1 :=
by
  intros P_Trip P_Drop_not h1 h2
  rw [h2]
  norm_num

end trip_drop_probability_l137_137275


namespace total_shirts_l137_137958

def initial_shirts : ℕ := 9
def new_shirts : ℕ := 8

theorem total_shirts : initial_shirts + new_shirts = 17 := by
  sorry

end total_shirts_l137_137958


namespace thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l137_137064

theorem thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one : 37 * 23 = 851 := by
  sorry

end thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l137_137064


namespace binomial_coefficient_plus_ten_l137_137157

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end binomial_coefficient_plus_ten_l137_137157


namespace arithmetic_mean_l137_137171

theorem arithmetic_mean (x b : ℝ) (h : x ≠ 0) : 
  (1 / 2) * ((2 + (b / x)) + (2 - (b / x))) = 2 :=
by sorry

end arithmetic_mean_l137_137171


namespace area_of_park_l137_137601

noncomputable def length (x : ℕ) : ℕ := 3 * x
noncomputable def width (x : ℕ) : ℕ := 2 * x
noncomputable def area (x : ℕ) : ℕ := length x * width x
noncomputable def cost_per_meter : ℕ := 80
noncomputable def total_cost : ℕ := 200
noncomputable def perimeter (x : ℕ) : ℕ := 2 * (length x + width x)

theorem area_of_park : ∃ x : ℕ, area x = 3750 ∧ total_cost = (perimeter x) * cost_per_meter / 100 := by
  sorry

end area_of_park_l137_137601


namespace solid_is_cone_l137_137829

-- Definitions of the conditions.
def front_and_side_views_are_equilateral_triangles (S : Type) : Prop :=
∀ (F : S → Prop) (E : S → Prop), (∃ T : S, F T ∧ E T ∧ T = T) 

def top_view_is_circle_with_center (S : Type) : Prop :=
∀ (C : S → Prop), (∃ O : S, C O ∧ O = O)

-- The proof statement that given the above conditions, the solid is a cone
theorem solid_is_cone (S : Type)
  (H1 : front_and_side_views_are_equilateral_triangles S)
  (H2 : top_view_is_circle_with_center S) : 
  ∃ C : S, C = C :=
by 
  sorry

end solid_is_cone_l137_137829


namespace factorize_x_pow_m_minus_x_pow_m_minus_2_l137_137462

theorem factorize_x_pow_m_minus_x_pow_m_minus_2 (x : ℝ) (m : ℕ) (h : m > 1) : 
  x ^ m - x ^ (m - 2) = (x ^ (m - 2)) * (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_pow_m_minus_x_pow_m_minus_2_l137_137462


namespace student_A_selection_probability_l137_137575

def probability_student_A_selected (total_students : ℕ) (students_removed : ℕ) (representatives : ℕ) : ℚ :=
  representatives / (total_students : ℚ)

theorem student_A_selection_probability :
  probability_student_A_selected 752 2 5 = 5 / 752 :=
by
  sorry

end student_A_selection_probability_l137_137575


namespace Frank_can_buy_7_candies_l137_137999

def tickets_whack_a_mole := 33
def tickets_skee_ball := 9
def cost_per_candy := 6

def total_tickets := tickets_whack_a_mole + tickets_skee_ball

theorem Frank_can_buy_7_candies : total_tickets / cost_per_candy = 7 := by
  sorry

end Frank_can_buy_7_candies_l137_137999


namespace total_seashells_l137_137235

-- Define the conditions from part a)
def unbroken_seashells : ℕ := 2
def broken_seashells : ℕ := 4

-- Define the proof problem
theorem total_seashells :
  unbroken_seashells + broken_seashells = 6 :=
by
  sorry

end total_seashells_l137_137235


namespace days_provisions_initially_meant_l137_137962

theorem days_provisions_initially_meant (x : ℕ) (h1 : 250 * x = 200 * 50) : x = 40 :=
by sorry

end days_provisions_initially_meant_l137_137962


namespace solution_set_of_inequality_l137_137814

theorem solution_set_of_inequality :
  { x : ℝ | - (1 : ℝ) / 2 < x ∧ x <= 1 } =
  { x : ℝ | (x - 1) / (2 * x + 1) <= 0 ∧ x ≠ - (1 : ℝ) / 2 } :=
by
  sorry

end solution_set_of_inequality_l137_137814


namespace gcd_168_486_l137_137133

theorem gcd_168_486 : gcd 168 486 = 6 := 
by sorry

end gcd_168_486_l137_137133


namespace root_expression_equals_181_div_9_l137_137762

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end root_expression_equals_181_div_9_l137_137762


namespace calculate_value_l137_137577

-- Given conditions
def n : ℝ := 2.25

-- Lean statement to express the proof problem
theorem calculate_value : (n / 3) * 12 = 9 := by
  -- Proof will be supplied here
  sorry

end calculate_value_l137_137577


namespace distinct_roots_condition_l137_137556

theorem distinct_roots_condition (a : ℝ) : 
  (∃ (x1 x2 x3 x4 : ℝ), (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ 
  (|x1^2 - 4| = a * x1 + 6) ∧ (|x2^2 - 4| = a * x2 + 6) ∧ (|x3^2 - 4| = a * x3 + 6) ∧ (|x4^2 - 4| = a * x4 + 6)) ↔ 
  ((-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3)) := sorry

end distinct_roots_condition_l137_137556


namespace pie_contest_l137_137435

def first_student_pie := 7 / 6
def second_student_pie := 4 / 3
def third_student_eats_from_first := 1 / 2
def third_student_eats_from_second := 1 / 3

theorem pie_contest :
  (first_student_pie - third_student_eats_from_first = 2 / 3) ∧
  (second_student_pie - third_student_eats_from_second = 1) ∧
  (third_student_eats_from_first + third_student_eats_from_second = 5 / 6) :=
by
  sorry

end pie_contest_l137_137435


namespace parabola_vertex_n_l137_137855

theorem parabola_vertex_n (x y : ℝ) (h : y = -3 * x^2 - 24 * x - 72) : ∃ m n : ℝ, (m, n) = (-4, -24) :=
by
  sorry

end parabola_vertex_n_l137_137855


namespace sam_coins_and_value_l137_137744

-- Define initial conditions
def initial_dimes := 9
def initial_nickels := 5
def initial_pennies := 12

def dimes_from_dad := 7
def nickels_taken_by_dad := 3

def pennies_exchanged := 12
def dimes_from_exchange := 2
def pennies_from_exchange := 2

-- Define final counts of coins after transactions
def final_dimes := initial_dimes + dimes_from_dad + dimes_from_exchange
def final_nickels := initial_nickels - nickels_taken_by_dad
def final_pennies := initial_pennies - pennies_exchanged + pennies_from_exchange

-- Define the total count of coins
def total_coins := final_dimes + final_nickels + final_pennies

-- Define the total value in cents
def value_dimes := final_dimes * 10
def value_nickels := final_nickels * 5
def value_pennies := final_pennies * 1

def total_value := value_dimes + value_nickels + value_pennies

-- Proof statement
theorem sam_coins_and_value :
  total_coins = 22 ∧ total_value = 192 := by
  -- Proof details would go here
  sorry

end sam_coins_and_value_l137_137744


namespace janet_total_distance_l137_137174

-- Define the distances covered in each week for each activity
def week1_running := 8 * 5
def week1_cycling := 7 * 3

def week2_running := 10 * 4
def week2_swimming := 2 * 2

def week3_running := 6 * 5
def week3_hiking := 3 * 2

-- Total distances for each activity
def total_running := week1_running + week2_running + week3_running
def total_cycling := week1_cycling
def total_swimming := week2_swimming
def total_hiking := week3_hiking

-- Total distance covered
def total_distance := total_running + total_cycling + total_swimming + total_hiking

-- Prove that the total distance is 141 miles
theorem janet_total_distance : total_distance = 141 := by
  sorry

end janet_total_distance_l137_137174


namespace rectangle_to_total_height_ratio_l137_137888

theorem rectangle_to_total_height_ratio 
  (total_area : ℕ)
  (width : ℕ)
  (area_per_side : ℕ)
  (height : ℕ)
  (triangle_base : ℕ)
  (triangle_area : ℕ)
  (rect_area : ℕ)
  (total_height : ℕ)
  (ratio : ℚ)
  (h_eqn : 3 * height = area_per_side)
  (h_value : height = total_area / (2 * 3))
  (total_height_eqn : total_height = 2 * height)
  (ratio_eqn : ratio = height / total_height) :
  total_area = 12 → width = 3 → area_per_side = 6 → triangle_base = 3 →
  triangle_area = triangle_base * height / 2 → rect_area = width * height →
  rect_area = area_per_side → ratio = 1 / 2 :=
by
  intros
  sorry

end rectangle_to_total_height_ratio_l137_137888


namespace album_photos_proof_l137_137751

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end album_photos_proof_l137_137751


namespace groupB_avg_weight_eq_141_l137_137005

def initial_group_weight (avg_weight : ℝ) : ℝ := 50 * avg_weight
def groupA_weight_gain : ℝ := 20 * 15
def groupB_weight_gain (x : ℝ) : ℝ := 20 * x

def total_weight (avg_weight : ℝ) (x : ℝ) : ℝ :=
  initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x

def total_avg_weight : ℝ := 46
def num_friends : ℝ := 90

def original_avg_weight : ℝ := total_avg_weight - 12
def final_total_weight : ℝ := num_friends * total_avg_weight

theorem groupB_avg_weight_eq_141 : 
  ∀ (avg_weight : ℝ) (x : ℝ),
    avg_weight = original_avg_weight →
    initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x = final_total_weight →
    avg_weight + x = 141 :=
by 
  intros avg_weight x h₁ h₂
  sorry

end groupB_avg_weight_eq_141_l137_137005


namespace diet_soda_ratio_l137_137188

def total_bottles : ℕ := 60
def diet_soda_bottles : ℕ := 14

theorem diet_soda_ratio : (diet_soda_bottles * 30) = (total_bottles * 7) :=
by {
  -- We're given that total_bottles = 60 and diet_soda_bottles = 14
  -- So to prove the ratio 14/60 is equivalent to 7/30:
  -- Multiplying both sides by 30 and 60 simplifies the arithmetic.
  sorry
}

end diet_soda_ratio_l137_137188


namespace greatest_common_divisor_of_72_and_m_l137_137034

-- Definitions based on the conditions
def is_power_of_prime (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p^k

-- Main theorem based on the question and conditions
theorem greatest_common_divisor_of_72_and_m (m : ℕ) :
  (Nat.gcd 72 m = 9) ↔ (m = 3^2) ∨ ∃ k, k ≥ 2 ∧ m = 3^k :=
by
  sorry

end greatest_common_divisor_of_72_and_m_l137_137034


namespace find_a_plus_b_l137_137347

def f (x a b : ℝ) := x^3 + a * x^2 + b * x + a^2

def extremum_at_one (a b : ℝ) : Prop :=
  f 1 a b = 10 ∧ (3 * 1^2 + 2 * a * 1 + b = 0)

theorem find_a_plus_b (a b : ℝ) (h : extremum_at_one a b) : a + b = -7 :=
by
  sorry

end find_a_plus_b_l137_137347


namespace det_2x2_matrix_l137_137697

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end det_2x2_matrix_l137_137697


namespace problem_statement_l137_137887

noncomputable def A := 5 * Real.pi / 12
noncomputable def B := Real.pi / 3
noncomputable def C := Real.pi / 4
noncomputable def b := Real.sqrt 3
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem problem_statement :
  (Set.Icc (-2 : ℝ) 2 = Set.image f Set.univ) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∃ (area : ℝ), area = (3 + Real.sqrt 3) / 4)
:= sorry

end problem_statement_l137_137887


namespace area_of_EPGQ_l137_137002

noncomputable def area_of_region (length_rect width_rect half_length_rect : ℝ) : ℝ :=
  half_length_rect * width_rect

theorem area_of_EPGQ :
  let length_rect := 10.0
  let width_rect := 6.0
  let P_half_length := length_rect / 2
  let Q_half_length := length_rect / 2
  (area_of_region length_rect width_rect P_half_length) = 30.0 :=
by
  sorry

end area_of_EPGQ_l137_137002


namespace arccos_range_l137_137682

theorem arccos_range (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4)) :
  Set.Icc 0 (3 * Real.pi / 4) = Set.image Real.arccos (Set.Icc (-Real.sqrt 2 / 2) 1) :=
by
  sorry

end arccos_range_l137_137682


namespace minimum_sum_of_original_numbers_l137_137238

theorem minimum_sum_of_original_numbers 
  (m n : ℕ) 
  (h1 : m < n) 
  (h2 : 23 * m - 20 * n = 460) 
  (h3 : ∀ m n, 23 * m - 20 * n = 460 → m < n):
  m + n = 321 :=
sorry

end minimum_sum_of_original_numbers_l137_137238


namespace fractions_order_l137_137874

theorem fractions_order :
  (21 / 17) < (18 / 13) ∧ (18 / 13) < (16 / 11) := by
  sorry

end fractions_order_l137_137874


namespace pencil_count_l137_137516

theorem pencil_count (P N X : ℝ) 
  (h1 : 96 * P + 24 * N = 520) 
  (h2 : X * P + 4 * N = 60) 
  (h3 : P + N = 15.512820512820513) :
  X = 3 :=
by
  sorry

end pencil_count_l137_137516


namespace complement_union_A_B_l137_137571

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l137_137571


namespace total_amount_divided_l137_137274

theorem total_amount_divided (P1 : ℝ) (r1 : ℝ) (r2 : ℝ) (interest : ℝ) (T : ℝ) :
  P1 = 1550 →
  r1 = 0.03 →
  r2 = 0.05 →
  interest = 144 →
  (P1 * r1 + (T - P1) * r2 = interest) → T = 3500 :=
by
  intros hP1 hr1 hr2 hint htotal
  sorry

end total_amount_divided_l137_137274


namespace baker_sold_pastries_l137_137815

theorem baker_sold_pastries : 
  ∃ P : ℕ, (97 = P + 89) ∧ P = 8 :=
by 
  sorry

end baker_sold_pastries_l137_137815


namespace total_weight_lifted_l137_137192

-- Definitions based on conditions
def original_lift : ℝ := 80
def after_training : ℝ := original_lift * 2
def specialization_increment : ℝ := after_training * 0.10
def specialized_lift : ℝ := after_training + specialization_increment

-- Statement of the theorem to prove total weight lifted
theorem total_weight_lifted : 
  (specialized_lift * 2) = 352 :=
sorry

end total_weight_lifted_l137_137192


namespace value_of_y_l137_137854

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end value_of_y_l137_137854


namespace factor_expression_l137_137659

theorem factor_expression (a b c d : ℝ) : 
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 
        = ((a - b) * (b - c) * (c - d) * (d - a)) * (a + b + c + d) := 
by
  sorry

end factor_expression_l137_137659


namespace negation_of_existential_l137_137878

theorem negation_of_existential :
  (∀ x : ℝ, x^2 + x - 1 ≤ 0) ↔ ¬ (∃ x : ℝ, x^2 + x - 1 > 0) :=
by sorry

end negation_of_existential_l137_137878


namespace compound_interest_at_least_double_l137_137484

theorem compound_interest_at_least_double :
  ∀ t : ℕ, (0 < t) → (1.05 : ℝ)^t > 2 ↔ t ≥ 15 :=
by sorry

end compound_interest_at_least_double_l137_137484


namespace sum_of_all_ks_l137_137628

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l137_137628


namespace percentage_calculation_l137_137917

-- Define total and part amounts
def total_amount : ℕ := 800
def part_amount : ℕ := 200

-- Define the percentage calculation
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Theorem to show the percentage is 25%
theorem percentage_calculation :
  percentage part_amount total_amount = 25 :=
sorry

end percentage_calculation_l137_137917


namespace find_a2016_l137_137394

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def cond1 : S 1 = 6 := by sorry
def cond2 : S 2 = 4 := by sorry
def cond3 (n : ℕ) : S n > 0 := by sorry
def cond4 (n : ℕ) : S (2 * n - 1) ^ 2 = S (2 * n) * S (2 * n + 2) := by sorry
def cond5 (n : ℕ) : 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1) := by sorry

theorem find_a2016 : a 2016 = -1009 := by
  -- Use the provided conditions to prove the statement
  sorry

end find_a2016_l137_137394


namespace simplify_and_evaluate_l137_137592

theorem simplify_and_evaluate
  (m : ℝ) (hm : m = 2 + Real.sqrt 2) :
  (1 - (m / (m + 2))) / ((m^2 - 4*m + 4) / (m^2 - 4)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l137_137592


namespace solution_to_trig_equation_l137_137973

theorem solution_to_trig_equation (x : ℝ) (k : ℤ) :
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
  (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) →
  (Real.sin (x / 2) = Real.cos (x / 2)) →
  (∃ k : ℤ, x = (π / 2) + 2 * π * ↑k) :=
by sorry

end solution_to_trig_equation_l137_137973


namespace lines_slope_angle_l137_137230

theorem lines_slope_angle (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : L1 = fun x => m * x)
  (h2 : L2 = fun x => n * x)
  (h3 : θ₁ = 3 * θ₂)
  (h4 : m = 3 * n)
  (h5 : θ₂ ≠ 0) :
  m * n = 9 / 4 :=
by
  sorry

end lines_slope_angle_l137_137230


namespace union_M_N_intersection_M_complement_N_l137_137927

open Set

variable (U : Set ℝ) (M N : Set ℝ)

-- Define the universal set
def is_universal_set (U : Set ℝ) : Prop :=
  U = univ

-- Define the set M
def is_set_M (M : Set ℝ) : Prop :=
  M = {x | ∃ y, y = (x - 2).sqrt}  -- or equivalently x ≥ 2

-- Define the set N
def is_set_N (N : Set ℝ) : Prop :=
  N = {x | x < 1 ∨ x > 3}

-- Define the complement of N in U
def complement_set_N (U N : Set ℝ) : Set ℝ :=
  U \ N

-- Prove M ∪ N = {x | x < 1 ∨ x ≥ 2}
theorem union_M_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∪ N = {x | x < 1 ∨ x ≥ 2} :=
  sorry

-- Prove M ∩ (complement of N in U) = {x | 2 ≤ x ≤ 3}
theorem intersection_M_complement_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∩ (complement_set_N U N) = {x | 2 ≤ x ∧ x ≤ 3} :=
  sorry

end union_M_N_intersection_M_complement_N_l137_137927


namespace wolves_heads_count_l137_137107

/-- 
A person goes hunting in the jungle and discovers a pack of wolves.
It is known that this person has one head and two legs, 
an ordinary wolf has one head and four legs, and a mutant wolf has two heads and three legs.
The total number of heads of all the people and wolves combined is 21,
and the total number of legs is 57.
-/
theorem wolves_heads_count :
  ∃ (x y : ℕ), (x + 2 * y = 20) ∧ (4 * x + 3 * y = 55) ∧ (x + y > 0) ∧ (x + 2 * y + 1 = 21) ∧ (4 * x + 3 * y + 2 = 57) := 
by {
  sorry
}

end wolves_heads_count_l137_137107


namespace subtracted_amount_l137_137643

theorem subtracted_amount (N A : ℝ) (h1 : 0.30 * N - A = 20) (h2 : N = 300) : A = 70 :=
by
  sorry

end subtracted_amount_l137_137643


namespace trains_clear_time_l137_137240

noncomputable def length_train1 : ℝ := 150
noncomputable def length_train2 : ℝ := 165
noncomputable def speed_train1_kmh : ℝ := 80
noncomputable def speed_train2_kmh : ℝ := 65
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * (5/18)
noncomputable def speed_train1 : ℝ := kmh_to_mps speed_train1_kmh
noncomputable def speed_train2 : ℝ := kmh_to_mps speed_train2_kmh
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_time : time_to_clear = 7.82 := 
sorry

end trains_clear_time_l137_137240


namespace grantRooms_is_2_l137_137493

/-- Danielle's apartment has 6 rooms. -/
def danielleRooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. -/
def heidiRooms : ℕ := 3 * danielleRooms

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. -/
def grantRooms : ℕ := heidiRooms / 9

/-- Prove that Grant's apartment has 2 rooms. -/
theorem grantRooms_is_2 : grantRooms = 2 := by
  sorry

end grantRooms_is_2_l137_137493


namespace train_speed_l137_137655

-- Define the conditions given in the problem
def train_length : ℝ := 160
def time_to_cross_man : ℝ := 4

-- Define the statement to be proved
theorem train_speed (H1 : train_length = 160) (H2 : time_to_cross_man = 4) : train_length / time_to_cross_man = 40 :=
by
  sorry

end train_speed_l137_137655


namespace bangles_per_box_l137_137257

-- Define the total number of pairs of bangles
def totalPairs : Nat := 240

-- Define the number of boxes
def numberOfBoxes : Nat := 20

-- Define the proof that each box can hold 24 bangles
theorem bangles_per_box : (totalPairs * 2) / numberOfBoxes = 24 :=
by
  -- Here we're required to do the proof but we'll use 'sorry' to skip it
  sorry

end bangles_per_box_l137_137257


namespace find_number_l137_137045

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l137_137045


namespace trigonometric_identity_l137_137509

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l137_137509


namespace abs_eq_two_iff_l137_137225

theorem abs_eq_two_iff (a : ℝ) : |a| = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_two_iff_l137_137225


namespace prime_square_mod_12_l137_137610

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_ne2 : p ≠ 2) (h_ne3 : p ≠ 3) :
    (∃ n : ℤ, p = 6 * n + 1 ∨ p = 6 * n + 5) → (p^2 % 12 = 1) := by
  sorry

end prime_square_mod_12_l137_137610


namespace number_of_common_tangents_l137_137211

/-- Define the circle C1 with center (2, -1) and radius 2. -/
def C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}

/-- Define the symmetry line x + y - 3 = 0. -/
def symmetry_line := {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Circle C2 is symmetric to C1 about the line x + y = 3. -/
def C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 1)^2 = 4}

/-- Circle C3 with the given condition MA^2 + MO^2 = 10 for any point M on the circle. 
    A(0, 2) and O is the origin. -/
def C3 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 4}

/-- The number of common tangents between circle C2 and circle C3 is 3. -/
theorem number_of_common_tangents
  (C1_sym_C2 : ∀ p : ℝ × ℝ, p ∈ C1 ↔ p ∈ C2)
  (M_on_C3 : ∀ M : ℝ × ℝ, M ∈ C3 → ((M.1)^2 + (M.2 - 2)^2) + ((M.1)^2 + (M.2)^2) = 10) :
  ∃ tangents : ℕ, tangents = 3 :=
sorry

end number_of_common_tangents_l137_137211


namespace cos_half_angle_neg_sqrt_l137_137797

theorem cos_half_angle_neg_sqrt (theta m : ℝ) 
  (h1 : (5 / 2) * Real.pi < theta ∧ theta < 3 * Real.pi)
  (h2 : |Real.cos theta| = m) : 
  Real.cos (theta / 2) = -Real.sqrt ((1 - m) / 2) :=
sorry

end cos_half_angle_neg_sqrt_l137_137797


namespace total_cost_l137_137053

theorem total_cost
  (cost_berries   : ℝ := 11.08)
  (cost_apples    : ℝ := 14.33)
  (cost_peaches   : ℝ := 9.31)
  (cost_grapes    : ℝ := 7.50)
  (cost_bananas   : ℝ := 5.25)
  (cost_pineapples: ℝ := 4.62)
  (total_cost     : ℝ := cost_berries + cost_apples + cost_peaches + cost_grapes + cost_bananas + cost_pineapples) :
  total_cost = 52.09 :=
by
  sorry

end total_cost_l137_137053


namespace items_left_in_store_l137_137685

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l137_137685


namespace polygon_sides_eq_nine_l137_137904

theorem polygon_sides_eq_nine (n : ℕ) (h : n - 1 = 8) : n = 9 := by
  sorry

end polygon_sides_eq_nine_l137_137904


namespace evening_water_usage_is_6_l137_137965

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end evening_water_usage_is_6_l137_137965


namespace find_d_value_l137_137283

/-- Let d be an odd prime number. If 89 - (d+3)^2 is the square of an integer, then d = 5. -/
theorem find_d_value (d : ℕ) (h₁ : Nat.Prime d) (h₂ : Odd d) (h₃ : ∃ m : ℤ, 89 - (d + 3)^2 = m^2) : d = 5 := 
by
  sorry

end find_d_value_l137_137283


namespace sum_series_eq_one_l137_137890

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l137_137890


namespace sum_first_eight_geom_terms_eq_l137_137369

noncomputable def S8_geom_sum : ℚ :=
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  a * (1 - r^8) / (1 - r)

theorem sum_first_eight_geom_terms_eq :
  S8_geom_sum = 3280 / 6561 :=
by
  sorry

end sum_first_eight_geom_terms_eq_l137_137369


namespace point_A_final_position_supplement_of_beta_l137_137535

-- Define the initial and final position of point A on the number line
def initial_position := -5
def moved_position_right := initial_position + 4
def final_position := moved_position_right - 1

theorem point_A_final_position : final_position = -2 := 
by 
-- Proof can be added here
sorry

-- Define the angles and the relationship between them
def alpha := 40
def beta := 90 - alpha
def supplement_beta := 180 - beta

theorem supplement_of_beta : supplement_beta = 130 := 
by 
-- Proof can be added here
sorry

end point_A_final_position_supplement_of_beta_l137_137535


namespace line_through_two_points_l137_137392

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) = (1, 3) ∨ (x, y) = (3, 7) → y = m * x + b) ∧ (m + b = 3) := by
{ sorry }

end line_through_two_points_l137_137392


namespace find_r_cubed_and_reciprocal_cubed_l137_137677

variable (r : ℝ)
variable (h : (r + 1 / r) ^ 2 = 5)

theorem find_r_cubed_and_reciprocal_cubed (r : ℝ) (h : (r + 1 / r) ^ 2 = 5) : r ^ 3 + 1 / r ^ 3 = 2 * Real.sqrt 5 := by
  sorry

end find_r_cubed_and_reciprocal_cubed_l137_137677


namespace purchasing_plans_count_l137_137976

theorem purchasing_plans_count :
  ∃ (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0) ∧ (∃! (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0)) := sorry

end purchasing_plans_count_l137_137976


namespace average_words_per_page_l137_137272

theorem average_words_per_page
  (sheets_to_pages : ℕ := 16)
  (total_sheets : ℕ := 12)
  (total_word_count : ℕ := 240000) :
  (total_word_count / (total_sheets * sheets_to_pages)) = 1250 :=
by
  sorry

end average_words_per_page_l137_137272


namespace largest_square_side_length_l137_137180

noncomputable def largestInscribedSquareSide (s : ℝ) (sharedSide : ℝ) : ℝ :=
  let y := (s * Real.sqrt 2 - sharedSide * Real.sqrt 3) / (2 * Real.sqrt 2)
  y

theorem largest_square_side_length :
  let s := 12
  let t := (s * Real.sqrt 6) / 3
  largestInscribedSquareSide s t = 6 - Real.sqrt 6 :=
by
  sorry

end largest_square_side_length_l137_137180


namespace tan_ratio_proof_l137_137823

noncomputable def tan_ratio (a b : ℝ) : ℝ := Real.tan a / Real.tan b

theorem tan_ratio_proof (a b : ℝ) (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 3) : 
tan_ratio a b = 23 / 7 := by
  sorry

end tan_ratio_proof_l137_137823


namespace first_loan_amount_l137_137152

theorem first_loan_amount :
  ∃ (L₁ L₂ : ℝ) (r : ℝ),
  (L₂ = 4700) ∧
  (L₁ = L₂ + 1500) ∧
  (0.09 * L₂ + r * L₁ = 617) ∧
  (L₁ = 6200) :=
by 
  sorry

end first_loan_amount_l137_137152


namespace min_c_plus_d_l137_137473

theorem min_c_plus_d (c d : ℤ) (h : c * d = 36) : c + d = -37 :=
sorry

end min_c_plus_d_l137_137473


namespace min_sum_of_inverses_l137_137444

theorem min_sum_of_inverses 
  (x y z p q r : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h_sum : x + y + z + p + q + r = 10) :
  (1 / x + 9 / y + 4 / z + 25 / p + 16 / q + 36 / r) = 44.1 :=
sorry

end min_sum_of_inverses_l137_137444


namespace no_perpendicular_hatching_other_than_cube_l137_137436

def is_convex_polyhedron (P : Polyhedron) : Prop :=
  -- Definition of a convex polyhedron
  sorry

def number_of_faces (P : Polyhedron) : ℕ :=
  -- Function returning the number of faces of polyhedron P
  sorry

def hatching_perpendicular (P : Polyhedron) : Prop :=
  -- Definition that checks if the hatching on adjacent faces of P is perpendicular
  sorry

theorem no_perpendicular_hatching_other_than_cube :
  ∀ (P : Polyhedron), is_convex_polyhedron P ∧ number_of_faces P ≠ 6 → ¬hatching_perpendicular P :=
by
  sorry

end no_perpendicular_hatching_other_than_cube_l137_137436


namespace math_proof_l137_137267

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof_l137_137267


namespace minimum_one_by_one_squares_l137_137794

theorem minimum_one_by_one_squares :
  ∀ (x y z : ℕ), 9 * x + 4 * y + z = 49 → (z = 3) :=
  sorry

end minimum_one_by_one_squares_l137_137794


namespace binomial_12_6_eq_1848_l137_137354

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l137_137354


namespace pear_counts_after_events_l137_137906

theorem pear_counts_after_events (Alyssa_picked Nancy_picked Carlos_picked : ℕ) (give_away : ℕ)
  (eat_fraction : ℚ) (share_fraction : ℚ) :
  Alyssa_picked = 42 →
  Nancy_picked = 17 →
  Carlos_picked = 25 →
  give_away = 5 →
  eat_fraction = 0.20 →
  share_fraction = 0.5 →
  ∃ (Alyssa_picked_final Nancy_picked_final Carlos_picked_final : ℕ),
    Alyssa_picked_final = 30 ∧
    Nancy_picked_final = 14 ∧
    Carlos_picked_final = 18 :=
by
  sorry

end pear_counts_after_events_l137_137906


namespace ratio_of_men_to_women_l137_137427

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l137_137427


namespace problem1_problem2_problem3_problem4_l137_137028

theorem problem1 : -20 - (-14) + (-18) - 13 = -37 := by
  sorry

theorem problem2 : (-3/4 + 1/6 - 5/8) / (-1/24) = 29 := by
  sorry

theorem problem3 : -3^2 + (-3)^2 + 3 * 2 + |(-4)| = 10 := by
  sorry

theorem problem4 : 16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3 := by
  sorry

end problem1_problem2_problem3_problem4_l137_137028


namespace problem_l137_137630

-- Define the matrix
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 0], ![0, 2, 3], ![3, 0, 2]]

-- Define the condition that there exists a nonzero vector v such that A * v = k * v
def exists_eigenvector (k : ℝ) : Prop :=
  ∃ (v : Fin 3 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v

theorem problem : ∀ (k : ℝ), exists_eigenvector k ↔ (k = 2 + (45)^(1/3)) :=
sorry

end problem_l137_137630


namespace average_speed_l137_137010

section
def flat_sand_speed : ℕ := 60
def downhill_slope_speed : ℕ := flat_sand_speed + 12
def uphill_slope_speed : ℕ := flat_sand_speed - 18

/-- Conner's average speed on flat, downhill, and uphill slopes, each of which he spends one-third of his time traveling on, is 58 miles per hour -/
theorem average_speed : (flat_sand_speed + downhill_slope_speed + uphill_slope_speed) / 3 = 58 := by
  sorry

end

end average_speed_l137_137010


namespace cuboid_can_form_square_projection_l137_137248

-- Definitions and conditions based directly on the problem
def length1 := 3
def length2 := 4
def length3 := 6

-- Statement to prove
theorem cuboid_can_form_square_projection (x y : ℝ) :
  (4 * x * x + y * y = 36) ∧ (x + y = 4) → True :=
by sorry

end cuboid_can_form_square_projection_l137_137248


namespace al_initial_amount_l137_137667

theorem al_initial_amount
  (a b c : ℕ)
  (h₁ : a + b + c = 2000)
  (h₂ : 3 * a + 2 * b + 2 * c = 3500) :
  a = 500 :=
sorry

end al_initial_amount_l137_137667


namespace janet_total_lives_l137_137782

/-
  Janet's initial lives: 38
  Lives lost: 16
  Lives gained: 32
  Prove that total lives == 54 after the changes
-/

theorem janet_total_lives (initial_lives lost_lives gained_lives : ℕ) 
(h1 : initial_lives = 38)
(h2 : lost_lives = 16)
(h3 : gained_lives = 32):
  initial_lives - lost_lives + gained_lives = 54 := by
  sorry

end janet_total_lives_l137_137782


namespace gerald_paid_l137_137507

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l137_137507


namespace find_q_minus_p_values_l137_137099

theorem find_q_minus_p_values (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) 
    (h : (p * (q + 1) + q * (p + 1)) * (n + 2) = 2 * n * p * q) : 
    q - p = 2 ∨ q - p = 3 ∨ q - p = 5 :=
sorry

end find_q_minus_p_values_l137_137099


namespace calculate_f_f_f_l137_137048

def f (x : ℤ) : ℤ := 3 * x + 2

theorem calculate_f_f_f :
  f (f (f 3)) = 107 :=
by
  sorry

end calculate_f_f_f_l137_137048


namespace sheepdog_catches_sheep_l137_137287

-- Define the speeds and the time taken
def v_s : ℝ := 12 -- speed of the sheep in feet/second
def v_d : ℝ := 20 -- speed of the sheepdog in feet/second
def t : ℝ := 20 -- time in seconds

-- Define the initial distance between the sheep and the sheepdog
def initial_distance (v_s v_d t : ℝ) : ℝ :=
  v_d * t - v_s * t

theorem sheepdog_catches_sheep :
  initial_distance v_s v_d t = 160 :=
by
  -- The formal proof would go here, but for now we replace it with sorry
  sorry

end sheepdog_catches_sheep_l137_137287


namespace polynomial_division_quotient_l137_137738

noncomputable def P (x : ℝ) := 8 * x^3 + 5 * x^2 - 4 * x - 7
noncomputable def D (x : ℝ) := x + 3

theorem polynomial_division_quotient :
  ∀ x : ℝ, (P x) / (D x) = 8 * x^2 - 19 * x + 53 := sorry

end polynomial_division_quotient_l137_137738


namespace largest_intersection_value_l137_137840

theorem largest_intersection_value (b c d : ℝ) :
  ∀ x : ℝ, (x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3 = c*x - d) → x ≤ 6 := sorry

end largest_intersection_value_l137_137840


namespace John_l137_137051

theorem John's_score_in_blackjack
  (Theodore_score : ℕ)
  (Zoey_cards : List ℕ)
  (winning_score : ℕ)
  (John_score : ℕ)
  (h1 : Theodore_score = 13)
  (h2 : Zoey_cards = [11, 3, 5])
  (h3 : winning_score = 19)
  (h4 : Zoey_cards.sum = winning_score)
  (h5 : winning_score ≠ Theodore_score) :
  John_score < 19 :=
by
  -- Here we would provide the proof if required
  sorry

end John_l137_137051


namespace rightmost_three_digits_of_7_pow_1997_l137_137197

theorem rightmost_three_digits_of_7_pow_1997 :
  7^1997 % 1000 = 207 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1997_l137_137197


namespace tan_405_eq_1_l137_137588

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 := 
by 
  sorry

end tan_405_eq_1_l137_137588


namespace find_m_of_perpendicular_vectors_l137_137538

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l137_137538


namespace percentage_reduction_is_10_percent_l137_137263

-- Definitions based on the given conditions
def rooms_rented_for_40 : ℕ := sorry
def rooms_rented_for_60 : ℕ := sorry
def total_rent : ℕ := 2000
def rent_per_room_40 : ℕ := 40
def rent_per_room_60 : ℕ := 60
def rooms_switch_count : ℕ := 10

-- Define the hypothetical new total if the rooms were rented at different rates
def new_total_rent : ℕ := (rent_per_room_40 * (rooms_rented_for_40 + rooms_switch_count)) + (rent_per_room_60 * (rooms_rented_for_60 - rooms_switch_count))

-- Calculate the percentage reduction
noncomputable def percentage_reduction : ℝ := (((total_rent: ℝ) - (new_total_rent: ℝ)) / (total_rent: ℝ)) * 100

-- Statement to prove
theorem percentage_reduction_is_10_percent : percentage_reduction = 10 := by
  sorry

end percentage_reduction_is_10_percent_l137_137263


namespace ratio_of_x_to_y_l137_137481

theorem ratio_of_x_to_y (x y : ℤ) (h : (7 * x - 4 * y) * 9 = (20 * x - 3 * y) * 4) : x * 17 = y * -24 :=
by {
  sorry
}

end ratio_of_x_to_y_l137_137481


namespace find_operation_l137_137483

theorem find_operation (a b : Int) (h : a + b = 0) : (7 + (-7) = 0) := 
by
  sorry

end find_operation_l137_137483


namespace area_of_triangle_ABC_l137_137517

theorem area_of_triangle_ABC
  {A B C : Type*} 
  (AC BC : ℝ)
  (B : ℝ)
  (h1 : AC = Real.sqrt (13))
  (h2 : BC = 1)
  (h3 : B = Real.sqrt 3 / 2): 
  ∃ area : ℝ, area = Real.sqrt 3 := 
sorry

end area_of_triangle_ABC_l137_137517


namespace theta_interval_l137_137778

noncomputable def f (x θ: ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_interval (θ: ℝ) (k: ℤ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x θ > 0) → 
  (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) := 
by
  sorry

end theta_interval_l137_137778


namespace birthday_check_value_l137_137218

theorem birthday_check_value : 
  ∃ C : ℝ, (150 + C) / 4 = C ↔ C = 50 :=
by
  sorry

end birthday_check_value_l137_137218


namespace value_of_f2_l137_137993

noncomputable def f : ℕ → ℕ :=
  sorry

axiom f_condition : ∀ x : ℕ, f (x + 1) = 2 * x + 3

theorem value_of_f2 : f 2 = 5 :=
by sorry

end value_of_f2_l137_137993


namespace binom_10_0_eq_1_l137_137957

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l137_137957


namespace angle_supplement_complement_l137_137127

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l137_137127


namespace susana_chocolate_chips_l137_137581

theorem susana_chocolate_chips :
  ∃ (S_c : ℕ), 
  (∃ (V_c V_v S_v : ℕ), 
    V_c = S_c + 5 ∧
    S_v = (3 * V_v) / 4 ∧
    V_v = 20 ∧
    V_c + S_c + V_v + S_v = 90) ∧
  S_c = 25 :=
by
  existsi 25
  sorry

end susana_chocolate_chips_l137_137581


namespace find_n_l137_137696

/-- Given a natural number n such that LCM(n, 12) = 48 and GCF(n, 12) = 8, prove that n = 32. -/
theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 48) (h2 : Nat.gcd n 12 = 8) : n = 32 :=
sorry

end find_n_l137_137696


namespace martha_to_doris_ratio_l137_137963

-- Define the amounts involved
def initial_amount : ℕ := 21
def doris_spent : ℕ := 6
def remaining_after_doris : ℕ := initial_amount - doris_spent
def final_amount : ℕ := 12
def martha_spent : ℕ := remaining_after_doris - final_amount

-- State the theorem about the ratio
theorem martha_to_doris_ratio : martha_spent * 2 = doris_spent :=
by
  -- Detailed proof is skipped
  sorry

end martha_to_doris_ratio_l137_137963


namespace solve_problem_l137_137565

-- Definitions based on conditions
def salty_cookies_eaten : ℕ := 28
def sweet_cookies_eaten : ℕ := 15

-- Problem statement
theorem solve_problem : salty_cookies_eaten - sweet_cookies_eaten = 13 := by
  sorry

end solve_problem_l137_137565


namespace diagonals_from_one_vertex_l137_137587

theorem diagonals_from_one_vertex (x : ℕ) (h : (x - 2) * 180 = 1800) : (x - 3) = 9 :=
  by
  sorry

end diagonals_from_one_vertex_l137_137587


namespace ratio_of_horns_l137_137078

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_harps := 0

def total_instruments := 7

def charlie_instruments := charlie_flutes + charlie_horns + charlie_harps
def carli_instruments := total_instruments - charlie_instruments

def carli_horns := carli_instruments - carli_flutes

theorem ratio_of_horns : (carli_horns : ℚ) / charlie_horns = 1 / 2 := by
  sorry

end ratio_of_horns_l137_137078


namespace find_f_of_9_l137_137425

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_of_3 : f 3 = 4

-- Theorem statement to prove
theorem find_f_of_9 : f 9 = 64 := by
  sorry

end find_f_of_9_l137_137425


namespace even_factors_count_of_n_l137_137477

def n : ℕ := 2^3 * 3^2 * 7 * 5

theorem even_factors_count_of_n : ∃ k : ℕ, k = 36 ∧ ∀ (a b c d : ℕ), 
  1 ≤ a ∧ a ≤ 3 →
  b ≤ 2 →
  c ≤ 1 →
  d ≤ 1 →
  2^a * 3^b * 7^c * 5^d ∣ n :=
sorry

end even_factors_count_of_n_l137_137477


namespace no_injective_function_exists_l137_137396

theorem no_injective_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f (x^2) - (f x)^2 ≥ 1/4) ∧ (∀ x y, f x = f y → x = y) := 
sorry

end no_injective_function_exists_l137_137396


namespace train_speed_l137_137350

/--
Given:
  Length of the train = 500 m
  Length of the bridge = 350 m
  The train takes 60 seconds to completely cross the bridge.

Prove:
  The speed of the train is exactly 14.1667 m/s
-/
theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 500) (h_bridge : length_bridge = 350) (h_time : time = 60) :
  (length_train + length_bridge) / time = 14.1667 :=
by
  rw [h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_l137_137350


namespace B_gain_l137_137496

-- Problem statement and conditions
def principalA : ℝ := 3500
def rateA : ℝ := 0.10
def periodA : ℕ := 2
def principalB : ℝ := 3500
def rateB : ℝ := 0.14
def periodB : ℕ := 3

-- Calculate amount A will receive from B after 2 years
noncomputable def amountA := principalA * (1 + rateA / 1) ^ periodA

-- Calculate amount B will receive from C after 3 years
noncomputable def amountB := principalB * (1 + rateB / 2) ^ (2 * periodB)

-- Calculate B's gain
noncomputable def gainB := amountB - amountA

-- The theorem to prove
theorem B_gain : gainB = 1019.20 := by
  sorry

end B_gain_l137_137496


namespace difference_before_exchange_l137_137046

--Definitions
variables {S B : ℤ}

-- Conditions
axiom h1 : S - 2 = B + 2
axiom h2 : B > S

theorem difference_before_exchange : B - S = 2 :=
by
-- Proof will go here
sorry

end difference_before_exchange_l137_137046


namespace benny_eggs_l137_137255

def dozen := 12

def total_eggs (n: Nat) := n * dozen

theorem benny_eggs:
  total_eggs 7 = 84 := 
by 
  sorry

end benny_eggs_l137_137255


namespace a_and_b_together_work_days_l137_137203

-- Definitions for the conditions:
def a_work_rate : ℚ := 1 / 9
def b_work_rate : ℚ := 1 / 18

-- The theorem statement:
theorem a_and_b_together_work_days : (a_work_rate + b_work_rate)⁻¹ = 6 := by
  sorry

end a_and_b_together_work_days_l137_137203


namespace color_of_85th_bead_l137_137013

/-- Definition for the repeating pattern of beads -/
def pattern : List String := ["red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

/-- Definition for finding the color of the n-th bead -/
def bead_color (n : Nat) : Option String :=
  let index := (n - 1) % pattern.length
  pattern.get? index

theorem color_of_85th_bead : bead_color 85 = some "yellow" := by
  sorry

end color_of_85th_bead_l137_137013


namespace oldest_sister_clothing_l137_137451

-- Define the initial conditions
def Nicole_initial := 10
def First_sister := Nicole_initial / 2
def Next_sister := Nicole_initial + 2
def Nicole_end := 36

-- Define the proof statement
theorem oldest_sister_clothing : 
    (First_sister + Next_sister + Nicole_initial + x = Nicole_end) → x = 9 :=
by
  sorry

end oldest_sister_clothing_l137_137451


namespace sum_of_perimeters_of_squares_l137_137513

theorem sum_of_perimeters_of_squares (x : ℝ) (h₁ : x = 3) :
  let area1 := x^2 + 4 * x + 4
  let area2 := 4 * x^2 - 12 * x + 9
  let side1 := Real.sqrt area1
  let side2 := Real.sqrt area2
  let perim1 := 4 * side1
  let perim2 := 4 * side2
  perim1 + perim2 = 32 :=
by
  sorry

end sum_of_perimeters_of_squares_l137_137513


namespace plankton_consumption_difference_l137_137820

theorem plankton_consumption_difference 
  (x : ℕ) 
  (d : ℕ) 
  (total_hours : ℕ := 9) 
  (total_consumption : ℕ := 360)
  (sixth_hour_consumption : ℕ := 43)
  (total_series_sum : x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) + (x + 5 * d) + (x + 6 * d) + (x + 7 * d) + (x + 8 * d) = total_consumption)
  (sixth_hour_eq : x + 5 * d = sixth_hour_consumption)
  : d = 3 :=
by
  sorry

end plankton_consumption_difference_l137_137820


namespace largest_integer_lt_100_with_rem_4_div_7_l137_137261

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l137_137261


namespace Quincy_sold_more_l137_137512

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l137_137512


namespace arc_length_of_circle_l137_137268

theorem arc_length_of_circle (r θ : ℝ) (h1 : r = 2) (h2 : θ = 5 * Real.pi / 3) : (θ * r) = 10 * Real.pi / 3 :=
by
  rw [h1, h2]
  -- subsequent steps would go here 
  sorry

end arc_length_of_circle_l137_137268


namespace toes_on_bus_is_164_l137_137933

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l137_137933


namespace min_value_inequality_l137_137599

theorem min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  ∃ n : ℝ, n = 9 / 4 ∧ (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 2 → (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ n) :=
sorry

end min_value_inequality_l137_137599


namespace sufficient_but_not_necessary_l137_137395

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 2) (h2 : b > 2) : 
  a + b > 4 ∧ a * b > 4 := 
by
  sorry

end sufficient_but_not_necessary_l137_137395


namespace coupon_redeem_day_l137_137059

theorem coupon_redeem_day (first_day : ℕ) (redeem_every : ℕ) : 
  (∀ n : ℕ, n < 8 → (first_day + n * redeem_every) % 7 ≠ 6) ↔ (first_day % 7 = 2 ∨ first_day % 7 = 5) :=
by
  sorry

end coupon_redeem_day_l137_137059


namespace min_value_of_fraction_l137_137364

theorem min_value_of_fraction 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Real.sqrt 3) = Real.sqrt (3 ^ a * 3 ^ (2 * b))) : 
  ∃ (min : ℝ), min = (2 / a + 1 / b) ∧ min = 8 :=
by
  -- proof will be skipped using sorry
  sorry

end min_value_of_fraction_l137_137364


namespace find_length_of_MN_l137_137441

theorem find_length_of_MN (A B C M N : ℝ × ℝ)
  (AB AC : ℝ) (M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (N_midpoint : N = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (length_AB : abs (B.1 - A.1) + abs (B.2 - A.2) = 15)
  (length_AC : abs (C.1 - A.1) + abs (C.2 - A.2) = 20) :
  abs (N.1 - M.1) + abs (N.2 - M.2) = 40 / 3 := sorry

end find_length_of_MN_l137_137441


namespace fish_estimation_l137_137137

noncomputable def number_caught := 50
noncomputable def number_marked_caught := 2
noncomputable def number_released := 30

theorem fish_estimation (N : ℕ) (h1 : number_caught = 50) 
  (h2 : number_marked_caught = 2) 
  (h3 : number_released = 30) :
  (number_marked_caught : ℚ) / number_caught = number_released / N → 
  N = 750 :=
by
  sorry

end fish_estimation_l137_137137


namespace no_valid_number_l137_137316

theorem no_valid_number (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) : ¬ ∃ (y : ℕ), (x * 100 + 3 * 10 + y) % 11 = 0 :=
by
  sorry

end no_valid_number_l137_137316


namespace Toby_change_l137_137335

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l137_137335


namespace expression_one_expression_two_l137_137802

-- Define the expressions to be proved.
theorem expression_one : (3.6 - 0.8) * (1.8 + 2.05) = 10.78 :=
by sorry

theorem expression_two : (34.28 / 2) - (16.2 / 4) = 13.09 :=
by sorry

end expression_one_expression_two_l137_137802


namespace triangle_inequality_inequality_l137_137076

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 > 0 := 
by
  sorry

end triangle_inequality_inequality_l137_137076


namespace sculpture_height_l137_137183

def base_height: ℝ := 10  -- height of the base in inches
def combined_height_feet: ℝ := 3.6666666666666665  -- combined height in feet
def inches_per_foot: ℝ := 12  -- conversion factor from feet to inches

-- Convert combined height to inches
def combined_height_inches: ℝ := combined_height_feet * inches_per_foot

-- Math proof problem statement
theorem sculpture_height : combined_height_inches - base_height = 34 := by
  sorry

end sculpture_height_l137_137183


namespace good_quadruple_inequality_l137_137148

theorem good_quadruple_inequality {p a b c : ℕ} (hp : Nat.Prime p) (hodd : p % 2 = 1) 
(habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
(hab : (a * b + 1) % p = 0) (hbc : (b * c + 1) % p = 0) (hca : (c * a + 1) % p = 0) :
  p + 2 ≤ (a + b + c) / 3 := 
by
  sorry

end good_quadruple_inequality_l137_137148


namespace part_I_part_II_part_III_l137_137518

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem part_I : 
  ∀ x:ℝ, f x = x^3 - x :=
by sorry

theorem part_II : 
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (-1:ℝ) 1 ∧ x2 ∈ Set.Icc (-1:ℝ) 1 ∧ (3 * x1^2 - 1) * (3 * x2^2 - 1) = -1 :=
by sorry

theorem part_III (x_n y_m : ℝ) (hx : x_n ∈ Set.Icc (-1:ℝ) 1) (hy : y_m ∈ Set.Icc (-1:ℝ) 1) : 
  |f x_n - f y_m| < 1 :=
by sorry

end part_I_part_II_part_III_l137_137518


namespace multiply_negatives_l137_137000

theorem multiply_negatives : (-3) * (-4) * (-1) = -12 := 
by sorry

end multiply_negatives_l137_137000


namespace probability_P_is_1_over_3_l137_137709

-- Definitions and conditions
def A := 0
def B := 3
def C := 1
def D := 2
def length_AB := B - A
def length_CD := D - C

-- Problem statement to prove
theorem probability_P_is_1_over_3 : (length_CD / length_AB) = 1 / 3 := by
  sorry

end probability_P_is_1_over_3_l137_137709


namespace parabola_focus_equals_ellipse_focus_l137_137323

theorem parabola_focus_equals_ellipse_focus (p : ℝ) : 
  let parabola_focus := (p / 2, 0)
  let ellipse_focus := (2, 0)
  parabola_focus = ellipse_focus → p = 4 :=
by
  intros h
  sorry

end parabola_focus_equals_ellipse_focus_l137_137323


namespace range_of_a_l137_137704

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → tensor (x - a) (x + a) < 2) → -1 < a ∧ a < 2 := by
  sorry

end range_of_a_l137_137704


namespace correct_student_answer_l137_137519

theorem correct_student_answer :
  (9 - (3^2) / 8 = 9 - (9 / 8)) ∧
  (24 - (4 * (3^2)) = 24 - 36) ∧
  ((36 - 12) / (3 / 2) = 24 * (2 / 3)) ∧
  ((-3)^2 / (1 / 3) * 3 = 9 * 3 * 3) →
  (24 * (2 / 3) = 16) :=
by
  sorry

end correct_student_answer_l137_137519


namespace weight_ratio_mars_moon_l137_137390

theorem weight_ratio_mars_moon :
  (∀ iron carbon other_elements_moon other_elements_mars wt_moon wt_mars : ℕ, 
    wt_moon = 250 ∧ 
    iron = 50 ∧ 
    carbon = 20 ∧ 
    other_elements_moon + 50 + 20 = 100 ∧ 
    other_elements_moon * wt_moon / 100 = 75 ∧ 
    other_elements_mars = 150 ∧ 
    wt_mars = (other_elements_mars * wt_moon) / other_elements_moon
  → wt_mars / wt_moon = 2) := 
sorry

end weight_ratio_mars_moon_l137_137390


namespace bridge_length_is_correct_l137_137702

noncomputable def speed_km_per_hour_to_m_per_s (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed_m_per_s time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

def bridge_length (total_distance train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem bridge_length_is_correct : 
  let train_length := 110 
  let speed_kmph := 72
  let time_s := 12.099
  let speed_m_per_s := speed_km_per_hour_to_m_per_s speed_kmph
  let total_distance := total_distance_covered speed_m_per_s time_s
  bridge_length total_distance train_length = 131.98 := 
by
  sorry

end bridge_length_is_correct_l137_137702


namespace total_metal_wasted_l137_137236

noncomputable def wasted_metal (a b : ℝ) (h : b ≤ 2 * a) : ℝ := 
  2 * a * b - (b ^ 2 / 2)

theorem total_metal_wasted (a b : ℝ) (h : b ≤ 2 * a) : 
  wasted_metal a b h = 2 * a * b - b ^ 2 / 2 :=
sorry

end total_metal_wasted_l137_137236


namespace lower_limit_for_x_l137_137087

variable {n : ℝ} {x : ℝ} {y : ℝ}

theorem lower_limit_for_x (h1 : x > n) (h2 : x < 8) (h3 : y > 8) (h4 : y < 13) (h5 : y - x = 7) : x = 2 :=
sorry

end lower_limit_for_x_l137_137087


namespace paired_products_not_equal_1000_paired_products_equal_10000_l137_137178

open Nat

theorem paired_products_not_equal_1000 :
  ∀ (a : Fin 1000 → ℤ), (∃ p n : Nat, p + n = 1000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) → False :=
by 
  sorry

theorem paired_products_equal_10000 :
  ∀ (a : Fin 10000 → ℤ), (∃ p n : Nat, p + n = 10000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) ↔ p = 5050 ∨ p = 4950 :=
by 
  sorry

end paired_products_not_equal_1000_paired_products_equal_10000_l137_137178


namespace last_digit_of_large_exponentiation_l137_137023

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l137_137023


namespace matrix_det_example_l137_137112

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l137_137112


namespace perpendicular_vectors_l137_137296

-- Definitions based on the conditions
def vector_a (x : ℝ) := (x, 3)
def vector_b := (3, 1)

-- Statement to prove
theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * (vector_b).1 + (vector_a x).2 * (vector_b).2 = 0 → x = -1 := by
  -- Proof goes here
  sorry

end perpendicular_vectors_l137_137296


namespace area_ratio_trapezoid_l137_137979

/--
In trapezoid PQRS, the lengths of the bases PQ and RS are 10 and 21 respectively.
The legs of the trapezoid are extended beyond P and Q to meet at point T.
Prove that the ratio of the area of triangle TPQ to the area of trapezoid PQRS is 100/341.
-/
theorem area_ratio_trapezoid (PQ RS TPQ PQRS : ℝ) (hPQ : PQ = 10) (hRS : RS = 21) :
  let area_TPQ := TPQ
  let area_PQRS := PQRS
  area_TPQ / area_PQRS = 100 / 341 :=
by
  sorry

end area_ratio_trapezoid_l137_137979


namespace max_average_speed_palindromic_journey_l137_137765

theorem max_average_speed_palindromic_journey
  (initial_odometer : ℕ)
  (final_odometer : ℕ)
  (trip_duration : ℕ)
  (max_speed : ℕ)
  (palindromic : ℕ → Prop)
  (initial_palindrome : palindromic initial_odometer)
  (final_palindrome : palindromic final_odometer)
  (max_speed_constraint : ∀ t, t ≤ trip_duration → t * max_speed ≤ final_odometer - initial_odometer)
  (trip_duration_eq : trip_duration = 5)
  (max_speed_eq : max_speed = 85)
  (initial_odometer_eq : initial_odometer = 69696)
  (final_odometer_max : final_odometer ≤ initial_odometer + max_speed * trip_duration) :
  (max_speed * (final_odometer - initial_odometer) / trip_duration : ℚ) = 82.2 :=
by sorry

end max_average_speed_palindromic_journey_l137_137765


namespace find_a_for_symmetry_l137_137733

theorem find_a_for_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, a * Real.sin x + Real.cos (x + π / 6) = 
                    a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x + π / 6)) 
           ↔ a = 2 :=
by
  sorry

end find_a_for_symmetry_l137_137733


namespace maximum_distance_l137_137584

-- Given conditions for the problem.
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline : ℝ := 23

-- Problem statement: prove the maximum distance on highway mileage.
theorem maximum_distance : highway_mpg * gasoline = 280.6 :=
sorry

end maximum_distance_l137_137584


namespace range_of_values_l137_137058

theorem range_of_values (x y : ℝ) (h : (x + 2)^2 + y^2 / 4 = 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = 28 / 3 ∧ a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b := by
  sorry

end range_of_values_l137_137058


namespace point_on_x_axis_l137_137506

theorem point_on_x_axis (a : ℝ) (h₁ : 1 - a = 0) : (3 * a - 6, 1 - a) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l137_137506


namespace smallest_positive_integer_l137_137417

theorem smallest_positive_integer :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 3 = 1 ∧ x % 7 = 3 ∧ ∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 3 = 1 ∧ y % 7 = 3 → x ≤ y :=
by
  sorry

end smallest_positive_integer_l137_137417


namespace pens_bought_is_17_l137_137409

def number_of_pens_bought (C S : ℝ) (bought_pens : ℝ) : Prop :=
  (bought_pens * C = 12 * S) ∧ (0.4 = (S - C) / C)

theorem pens_bought_is_17 (C S : ℝ) (bought_pens : ℝ) 
  (h1 : bought_pens * C = 12 * S)
  (h2 : 0.4 = (S - C) / C) :
  bought_pens = 17 :=
sorry

end pens_bought_is_17_l137_137409


namespace bar_graph_represents_circle_graph_l137_137501

theorem bar_graph_represents_circle_graph (r b g : ℕ) 
  (h1 : r = g) 
  (h2 : b = 3 * r) : 
  (r = 1 ∧ b = 3 ∧ g = 1) :=
sorry

end bar_graph_represents_circle_graph_l137_137501


namespace smallest_c_in_progressions_l137_137987

def is_arithmetic_progression (a b c : ℤ) : Prop := b - a = c - b

def is_geometric_progression (b c a : ℤ) : Prop := c^2 = a*b

theorem smallest_c_in_progressions :
  ∃ (a b c : ℤ), is_arithmetic_progression a b c ∧ is_geometric_progression b c a ∧ 
  (∀ (a' b' c' : ℤ), is_arithmetic_progression a' b' c' ∧ is_geometric_progression b' c' a' → c ≤ c') ∧ c = 2 :=
by
  sorry

end smallest_c_in_progressions_l137_137987


namespace smallest_x_undefined_l137_137808

theorem smallest_x_undefined :
  (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1 ∨ x = 8) → (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1) :=
by
  sorry

end smallest_x_undefined_l137_137808


namespace number_of_papers_l137_137858

-- Define the conditions
def folded_pieces (folds : ℕ) : ℕ := 2 ^ folds
def notes_per_day : ℕ := 10
def days_per_notepad : ℕ := 4
def notes_per_notepad : ℕ := notes_per_day * days_per_notepad
def notes_per_paper (folds : ℕ) : ℕ := folded_pieces folds

-- Lean statement for the proof problem
theorem number_of_papers (folds : ℕ) (h_folds : folds = 3) :
  (notes_per_notepad / notes_per_paper folds) = 5 :=
by
  rw [h_folds]
  simp [notes_per_notepad, notes_per_paper, folded_pieces]
  sorry

end number_of_papers_l137_137858


namespace avg_height_of_remaining_students_l137_137229

-- Define the given conditions
def avg_height_11_members : ℝ := 145.7
def number_of_members : ℝ := 11
def height_of_two_students : ℝ := 142.1

-- Define what we need to prove
theorem avg_height_of_remaining_students :
  (avg_height_11_members * number_of_members - 2 * height_of_two_students) / (number_of_members - 2) = 146.5 :=
by
  sorry

end avg_height_of_remaining_students_l137_137229


namespace inequalities_consistent_l137_137875

theorem inequalities_consistent (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1) ^ 2) (h3 : y * (y - 1) ≤ x ^ 2) : true := 
by 
  sorry

end inequalities_consistent_l137_137875


namespace chocolates_vs_gums_l137_137869

theorem chocolates_vs_gums 
    (c g : ℝ) 
    (Kolya_claim : 2 * c > 5 * g) 
    (Sasha_claim : ¬ ( 3 * c > 8 * g )) : 
    7 * c ≤ 19 * g := 
sorry

end chocolates_vs_gums_l137_137869


namespace find_a_l137_137909

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {1, 3, a}) (hB : B = {1, a^2 - a + 1}) (h_subset : B ⊆ A) :
  a = -1 ∨ a = 2 := 
by
  sorry

end find_a_l137_137909


namespace simple_interest_rate_l137_137795

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (H : T = 4 ∧ (7 / 6) * P = P + (P * R * T / 100)) :
  R = 4.17 :=
by
  sorry

end simple_interest_rate_l137_137795


namespace correct_transformation_l137_137374

theorem correct_transformation (x : ℝ) :
  3 + x = 7 ∧ ¬ (x = 7 + 3) ∧
  5 * x = -4 ∧ ¬ (x = -5 / 4) ∧
  (7 / 4) * x = 3 ∧ ¬ (x = 3 * (7 / 4)) ∧
  -((x - 2) / 4) = 1 ∧ (-(x - 2)) = 4 :=
by
  sorry

end correct_transformation_l137_137374


namespace find_number_l137_137349

-- Definitions of the fractions involved
def frac_2_15 : ℚ := 2 / 15
def frac_1_5 : ℚ := 1 / 5
def frac_1_2 : ℚ := 1 / 2

-- Condition that the number is greater than the sum of frac_2_15 and frac_1_5 by frac_1_2 
def number : ℚ := frac_2_15 + frac_1_5 + frac_1_2

-- Theorem statement matching the math proof problem
theorem find_number : number = 5 / 6 :=
by
  sorry

end find_number_l137_137349


namespace common_chord_eq_l137_137429

theorem common_chord_eq (x y : ℝ) :
  x^2 + y^2 + 2*x = 0 →
  x^2 + y^2 - 4*y = 0 →
  x + 2*y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_eq_l137_137429


namespace batteries_C_equivalent_l137_137947

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end batteries_C_equivalent_l137_137947


namespace purely_periodic_period_le_T_l137_137406

theorem purely_periodic_period_le_T {a b : ℚ} (T : ℕ) 
  (ha : ∃ m, a = m / (10^T - 1)) 
  (hb : ∃ n, b = n / (10^T - 1)) :
  (∃ T₁, T₁ ≤ T ∧ ∃ p, a = p / (10^T₁ - 1)) ∧ 
  (∃ T₂, T₂ ≤ T ∧ ∃ q, b = q / (10^T₂ - 1)) := 
sorry

end purely_periodic_period_le_T_l137_137406


namespace part1_part2_l137_137036

-- Definitions of y1 and y2 based on given conditions
def y1 (x : ℝ) : ℝ := -x + 3
def y2 (x : ℝ) : ℝ := 2 + x

-- Prove for x such that y1 = y2
theorem part1 (x : ℝ) : y1 x = y2 x ↔ x = 1 / 2 := by
  sorry

-- Prove for x such that y1 = 2y2 + 5
theorem part2 (x : ℝ) : y1 x = 2 * y2 x + 5 ↔ x = -2 := by
  sorry

end part1_part2_l137_137036


namespace driver_net_rate_of_pay_is_30_33_l137_137576

noncomputable def driver_net_rate_of_pay : ℝ :=
  let hours := 3
  let speed_mph := 65
  let miles_per_gallon := 30
  let pay_per_mile := 0.55
  let cost_per_gallon := 2.50
  let total_distance := speed_mph * hours
  let gallons_used := total_distance / miles_per_gallon
  let gross_earnings := total_distance * pay_per_mile
  let fuel_cost := gallons_used * cost_per_gallon
  let net_earnings := gross_earnings - fuel_cost
  let net_rate_per_hour := net_earnings / hours
  net_rate_per_hour

theorem driver_net_rate_of_pay_is_30_33 :
  driver_net_rate_of_pay = 30.33 :=
by
  sorry

end driver_net_rate_of_pay_is_30_33_l137_137576


namespace cos_A_condition_is_isosceles_triangle_tan_sum_l137_137852

variable {A B C a b c : ℝ}

theorem cos_A_condition (h : (3 * b - c) * Real.cos A - a * Real.cos C = 0) :
  Real.cos A = 1 / 3 := sorry

theorem is_isosceles_triangle (ha : a = 2 * Real.sqrt 3)
  (hs : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2) :
  c = 3 ∧ b = 3 := sorry

theorem tan_sum (h_sin : Real.sin B * Real.sin C = 2 / 3)
  (h_cos : Real.cos A = 1 / 3) :
  Real.tan A + Real.tan B + Real.tan C = 4 * Real.sqrt 2 := sorry

end cos_A_condition_is_isosceles_triangle_tan_sum_l137_137852


namespace mouse_jumps_28_inches_further_than_grasshopper_l137_137485

theorem mouse_jumps_28_inches_further_than_grasshopper :
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  g_to_m_difference = 28 :=
by
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  show g_to_m_difference = 28
  sorry

end mouse_jumps_28_inches_further_than_grasshopper_l137_137485


namespace calculation_correct_l137_137636

theorem calculation_correct (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hxy : x = 2 * y) : 
  (x - 2 / x) * (y + 2 / y) = 1 / 2 * (x^2 - 2 * x + 8 - 16 / x) := 
by 
  sorry

end calculation_correct_l137_137636


namespace find_center_and_radius_sum_l137_137873

-- Define the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 16 * x + y^2 + 10 * y = -75

-- Define the center of the circle
def center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x = a) ∧ (y = b)

-- Define the radius of the circle
def radius (r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x^2 - 16 * x + y^2 + 10 * y = r^2)

-- Main theorem to prove a + b + r = 3 + sqrt 14
theorem find_center_and_radius_sum (a b r : ℝ) (h_cen : center a b) (h_rad : radius r) : 
  a + b + r = 3 + Real.sqrt 14 :=
  sorry

end find_center_and_radius_sum_l137_137873


namespace trig_relationship_l137_137589

noncomputable def a := Real.cos 1
noncomputable def b := Real.cos 2
noncomputable def c := Real.sin 2

theorem trig_relationship : c > a ∧ a > b := by
  sorry

end trig_relationship_l137_137589


namespace initial_students_count_l137_137663

theorem initial_students_count (n : ℕ) (T T' : ℚ)
    (h1 : T = n * 61.5)
    (h2 : T' = T - 24)
    (h3 : T' = (n - 1) * 64) :
  n = 16 :=
by
  sorry

end initial_students_count_l137_137663


namespace sqrt_calc1_sqrt_calc2_l137_137346

-- Problem 1 proof statement
theorem sqrt_calc1 : ( (Real.sqrt 2 + Real.sqrt 3) ^ 2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 4 + 2 * Real.sqrt 6 ) :=
  sorry

-- Problem 2 proof statement
theorem sqrt_calc2 : ( (2 - Real.sqrt 3) ^ 2023 * (2 + Real.sqrt 3) ^ 2023 - 2 * abs (-Real.sqrt 3 / 2) - (-Real.sqrt 2) ^ 0 = -Real.sqrt 3 ) :=
  sorry

end sqrt_calc1_sqrt_calc2_l137_137346


namespace distinct_x_sum_l137_137912

theorem distinct_x_sum (x y z : ℂ) 
(h1 : x + y * z = 9) 
(h2 : y + x * z = 12) 
(h3 : z + x * y = 12) : 
(x = 1 ∨ x = 3) ∧ (¬(x = 1 ∧ x = 3) → x ≠ 1 ∧ x ≠ 3) ∧ (1 + 3 = 4) :=
by
  sorry

end distinct_x_sum_l137_137912


namespace probability_of_urn_contains_nine_red_and_four_blue_after_operations_l137_137582

-- Definition of the initial urn state
def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1

-- Definition of the number of operations
def num_operations : ℕ := 5

-- Definition of the final state
def final_red_balls : ℕ := 9
def final_blue_balls : ℕ := 4

-- Definition of total number of balls after five operations
def total_balls_after_operations : ℕ := 13

-- The probability we aim to prove
def target_probability : ℚ := 1920 / 10395

noncomputable def george_experiment_probability_theorem 
  (initial_red_balls initial_blue_balls num_operations final_red_balls final_blue_balls : ℕ)
  (total_balls_after_operations : ℕ) : ℚ :=
if initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ num_operations = 5 ∧ final_red_balls = 9 ∧ final_blue_balls = 4 ∧ total_balls_after_operations = 13 then
  target_probability
else
  0

-- The theorem statement, no proof provided (using sorry).
theorem probability_of_urn_contains_nine_red_and_four_blue_after_operations :
  george_experiment_probability_theorem 2 1 5 9 4 13 = target_probability := sorry

end probability_of_urn_contains_nine_red_and_four_blue_after_operations_l137_137582


namespace number_of_arrangements_l137_137122

theorem number_of_arrangements (A B : Type) (individuals : Fin 6 → Type)
  (adjacent_condition : ∃ (i : Fin 5), individuals i = B ∧ individuals (i + 1) = A) :
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end number_of_arrangements_l137_137122


namespace probability_heads_odd_l137_137861

theorem probability_heads_odd (n : ℕ) (p : ℚ) (Q : ℕ → ℚ) (h : p = 3/4) (h_rec : ∀ n, Q (n + 1) = p * (1 - Q n) + (1 - p) * Q n) :
  Q 40 = 1/2 * (1 - 1/4^40) := 
sorry

end probability_heads_odd_l137_137861


namespace ratio_of_perimeter_to_b_l137_137187

theorem ratio_of_perimeter_to_b (b : ℝ) (hb : b ≠ 0) :
  let p1 := (-2*b, -2*b)
  let p2 := (2*b, -2*b)
  let p3 := (2*b, 2*b)
  let p4 := (-2*b, 2*b)
  let l := (y = b * x)
  let d1 := 4*b
  let d2 := 4*b
  let d3 := 4*b
  let d4 := 4*b*Real.sqrt 2
  let perimeter := d1 + d2 + d3 + d4
  let ratio := perimeter / b
  ratio = 12 + 4 * Real.sqrt 2 := by
  -- Placeholder for proof
  sorry

end ratio_of_perimeter_to_b_l137_137187


namespace parallelepiped_side_lengths_l137_137524

theorem parallelepiped_side_lengths (x y z : ℕ) 
  (h1 : x + y + z = 17) 
  (h2 : 2 * x * y + 2 * y * z + 2 * z * x = 180) 
  (h3 : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 :=
by {
  sorry
}

end parallelepiped_side_lengths_l137_137524


namespace smallest_n_for_sum_exceed_10_pow_5_l137_137801

def a₁ : ℕ := 9
def r : ℕ := 10
def S (n : ℕ) : ℕ := 5 * n^2 + 4 * n
def target_sum : ℕ := 10^5

theorem smallest_n_for_sum_exceed_10_pow_5 : 
  ∃ n : ℕ, S n > target_sum ∧ ∀ m < n, ¬(S m > target_sum) := 
sorry

end smallest_n_for_sum_exceed_10_pow_5_l137_137801


namespace reading_time_per_week_l137_137791

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l137_137791


namespace f_20_plus_f_neg20_l137_137688

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 + 5

theorem f_20_plus_f_neg20 (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end f_20_plus_f_neg20_l137_137688


namespace find_alpha_l137_137110

noncomputable section

open Real 

def curve_C1 (x y : ℝ) : Prop := x + y = 1
def curve_C2 (x y φ : ℝ) : Prop := x = 2 + 2 * cos φ ∧ y = 2 * sin φ 

def polar_coordinate_eq1 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2
def polar_coordinate_eq2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

def line_l (ρ θ α : ℝ)  (hα: α > 0 ∧ α < π / 2) : Prop := θ = α ∧ ρ > 0 

def OB_div_OA_eq_4 (ρA ρB α : ℝ) : Prop := ρB / ρA = 4

theorem find_alpha (α : ℝ) (hα: α > 0 ∧ α < π / 2)
  (h₁: ∀ (x y ρ θ: ℝ), curve_C1 x y → polar_coordinate_eq1 ρ θ) 
  (h₂: ∀ (x y φ ρ θ: ℝ), curve_C2 x y φ → polar_coordinate_eq2 ρ θ) 
  (h₃: ∀ (ρ θ: ℝ), line_l ρ θ α hα) 
  (h₄: ∀ (ρA ρB : ℝ), OB_div_OA_eq_4 ρA ρB α → ρA = 1 / (cos α + sin α) ∧ ρB = 4 * cos α ): 
  α = 3 * π / 8 :=
by
  sorry

end find_alpha_l137_137110


namespace sum_weights_second_fourth_l137_137052

-- Definitions based on given conditions
noncomputable section

def weight (n : ℕ) : ℕ := 4 - (n - 1)

-- Assumption that weights form an arithmetic sequence.
-- 1st foot weighs 4 jin, 5th foot weighs 2 jin, and weights are linearly decreasing.
axiom weight_arith_seq (n : ℕ) : weight n = 4 - (n - 1)

-- Prove the sum of the weights of the second and fourth feet
theorem sum_weights_second_fourth :
  weight 2 + weight 4 = 6 :=
by
  simp [weight_arith_seq]
  sorry

end sum_weights_second_fourth_l137_137052


namespace div_by_13_l137_137388

theorem div_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) % 13 = 0 :=
by
  sorry

end div_by_13_l137_137388


namespace find_x_squared_plus_y_squared_l137_137676

theorem find_x_squared_plus_y_squared (x y : ℝ) (h₁ : x * y = -8) (h₂ : x^2 * y + x * y^2 + 3 * x + 3 * y = 100) : x^2 + y^2 = 416 :=
sorry

end find_x_squared_plus_y_squared_l137_137676


namespace frequency_of_scoring_l137_137703

def shots : ℕ := 80
def goals : ℕ := 50
def frequency : ℚ := goals / shots

theorem frequency_of_scoring : frequency = 0.625 := by
  sorry

end frequency_of_scoring_l137_137703


namespace greatest_third_term_of_arithmetic_sequence_l137_137609

def is_arithmetic_sequence (a b c d : ℤ) : Prop := (b - a = c - b) ∧ (c - b = d - c)

theorem greatest_third_term_of_arithmetic_sequence :
  ∃ a b c d : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  is_arithmetic_sequence a b c d ∧
  (a + b + c + d = 52) ∧
  (c = 17) :=
sorry

end greatest_third_term_of_arithmetic_sequence_l137_137609


namespace sufficient_but_not_necessary_l137_137656

-- Define the equations of the lines
def line1 (a : ℝ) (x y : ℝ) : ℝ := 2 * x + a * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := (a - 1) * x + 3 * y - 2

-- Define the condition for parallel lines by comparing their slopes
def parallel_condition (a : ℝ) : Prop :=  (2 * 3 = a * (a - 1))

theorem sufficient_but_not_necessary (a : ℝ) : 3 ≤ a :=
  sorry

end sufficient_but_not_necessary_l137_137656


namespace find_positive_integers_l137_137329

theorem find_positive_integers
  (a b c : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ a ≥ c)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 + 1 / (a : ℚ)) * (1 + 1 / (b : ℚ)) * (1 + 1 / (c : ℚ)) = 2 →
  (a, b, c) ∈ [(15, 4, 2), (9, 5, 2), (7, 6, 2), (8, 3, 3), (5, 4, 3)] :=
by
  sorry

end find_positive_integers_l137_137329


namespace race_distance_l137_137908

theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l137_137908


namespace bakery_item_count_l137_137934

theorem bakery_item_count : ∃ (s c : ℕ), 5 * s + 25 * c = 500 ∧ s + c = 12 := by
  sorry

end bakery_item_count_l137_137934


namespace opposite_of_neg_6_l137_137121

theorem opposite_of_neg_6 : ∀ (n : ℤ), n = -6 → -n = 6 :=
by
  intro n h
  rw [h]
  sorry

end opposite_of_neg_6_l137_137121


namespace geom_sequence_sum_l137_137629

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end geom_sequence_sum_l137_137629


namespace distance_between_lines_is_two_l137_137205

noncomputable def distance_between_parallel_lines : ℝ := 
  let A1 := 3
  let B1 := 4
  let C1 := -3
  let A2 := 6
  let B2 := 8
  let C2 := 14
  (|C2 - C1| : ℝ) / Real.sqrt (A2^2 + B2^2)

theorem distance_between_lines_is_two :
  distance_between_parallel_lines = 2 := by
  sorry

end distance_between_lines_is_two_l137_137205


namespace pet_store_cages_l137_137366

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h_initial : initial_puppies = 13) (h_sold : sold_puppies = 7) (h_per_cage : puppies_per_cage = 2) : (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l137_137366


namespace more_apples_than_pears_l137_137489

-- Define the variables
def apples := 17
def pears := 9

-- Theorem: The number of apples minus the number of pears equals 8
theorem more_apples_than_pears : apples - pears = 8 :=
by
  sorry

end more_apples_than_pears_l137_137489


namespace alpha_in_second_quadrant_l137_137140

theorem alpha_in_second_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.cos α - Real.sin α < 0) : 
  π / 2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l137_137140


namespace find_f_2015_l137_137816

def f (x : ℝ) := 2 * x - 1 

theorem find_f_2015 (f : ℝ → ℝ)
  (H1 : ∀ a b : ℝ, f ((2 * a + b) / 3) = (2 * f a + f b) / 3)
  (H2 : f 1 = 1)
  (H3 : f 4 = 7) :
  f 2015 = 4029 := by 
  sorry

end find_f_2015_l137_137816


namespace ferry_tourists_total_l137_137715

theorem ferry_tourists_total 
  (n : ℕ)
  (a d : ℕ)
  (sum_arithmetic_series : ℕ → ℕ → ℕ → ℕ)
  (trip_count : n = 5)
  (first_term : a = 85)
  (common_difference : d = 3) :
  sum_arithmetic_series n a d = 455 :=
by
  sorry

end ferry_tourists_total_l137_137715


namespace total_women_attendees_l137_137134

theorem total_women_attendees 
  (adults : ℕ) (adult_women : ℕ) (student_offset : ℕ) (total_students : ℕ)
  (male_students : ℕ) :
  adults = 1518 →
  adult_women = 536 →
  student_offset = 525 →
  total_students = adults + student_offset →
  total_students = 2043 →
  male_students = 1257 →
  (adult_women + (total_students - male_students) = 1322) :=
by
  sorry

end total_women_attendees_l137_137134


namespace overtime_pay_rate_increase_l137_137009

theorem overtime_pay_rate_increase
  (regular_rate : ℝ)
  (total_compensation : ℝ)
  (total_hours : ℝ)
  (overtime_hours : ℝ)
  (expected_percentage_increase : ℝ)
  (h1 : regular_rate = 16)
  (h2 : total_hours = 48)
  (h3 : total_compensation = 864)
  (h4 : overtime_hours = total_hours - 40)
  (h5 : 40 * regular_rate + overtime_hours * (regular_rate + regular_rate * expected_percentage_increase / 100) = total_compensation) :
  expected_percentage_increase = 75 := 
by
  sorry

end overtime_pay_rate_increase_l137_137009


namespace Mark_owes_total_l137_137439

noncomputable def base_fine : ℕ := 50

def additional_fine (speed_over_limit : ℕ) : ℕ :=
  let first_10 := min speed_over_limit 10 * 2
  let next_5 := min (speed_over_limit - 10) 5 * 3
  let next_10 := min (speed_over_limit - 15) 10 * 5
  let remaining := max (speed_over_limit - 25) 0 * 6
  first_10 + next_5 + next_10 + remaining

noncomputable def total_fine (base : ℕ) (additional : ℕ) (school_zone : Bool) : ℕ :=
  let fine := base + additional
  if school_zone then fine * 2 else fine

def court_costs : ℕ := 350

noncomputable def processing_fee (fine : ℕ) : ℕ := fine / 10

def lawyer_fees (hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours

theorem Mark_owes_total :
  let speed_over_limit := 45
  let base := base_fine
  let additional := additional_fine speed_over_limit
  let school_zone := true
  let fine := total_fine base additional school_zone
  let total_fine_with_costs := fine + court_costs
  let processing := processing_fee total_fine_with_costs
  let lawyer := lawyer_fees 100 4
  let total := total_fine_with_costs + processing + lawyer
  total = 1346 := sorry

end Mark_owes_total_l137_137439


namespace g_of_f_roots_reciprocal_l137_137277

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c

theorem g_of_f_roots_reciprocal
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ∃ g : ℝ → ℝ, g 1 = (4 - a) / (4 * c) :=
sorry

end g_of_f_roots_reciprocal_l137_137277


namespace factor_of_land_increase_l137_137895

-- Definitions of the conditions in the problem:
def initial_money_given_by_blake : ℝ := 20000
def money_received_by_blake_after_sale : ℝ := 30000

-- The main theorem to prove
theorem factor_of_land_increase (F : ℝ) : 
  (1/2) * (initial_money_given_by_blake * F) = money_received_by_blake_after_sale → 
  F = 3 :=
by sorry

end factor_of_land_increase_l137_137895


namespace problem_part1_problem_part2_l137_137753

variable (a b : ℝ)

theorem problem_part1 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  9 / a + 1 / b ≥ 4 :=
sorry

theorem problem_part2 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  ∃ a b, (a + 3 / b) * (b + 3 / a) = 12 :=
sorry

end problem_part1_problem_part2_l137_137753


namespace least_n_divisible_by_some_not_all_l137_137679

theorem least_n_divisible_by_some_not_all (n : ℕ) (h : 1 ≤ n):
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ k ∣ (n^2 - n)) ∧ ¬ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ (n^2 - n)) ↔ n = 3 :=
by
  sorry

end least_n_divisible_by_some_not_all_l137_137679


namespace line_through_two_points_l137_137007

theorem line_through_two_points :
  ∀ (A_1 B_1 A_2 B_2 : ℝ),
    (2 * A_1 + 3 * B_1 = 1) →
    (2 * A_2 + 3 * B_2 = 1) →
    (∀ (x y : ℝ), (2 * x + 3 * y = 1) → (x * (B_2 - B_1) + y * (A_1 - A_2) = A_1 * B_2 - A_2 * B_1)) :=
by 
  intros A_1 B_1 A_2 B_2 h1 h2 x y hxy
  sorry

end line_through_two_points_l137_137007


namespace value_of_expression_l137_137239

theorem value_of_expression (x : ℕ) (h : x = 8) : 
  (x^3 + 3 * (x^2) * 2 + 3 * x * (2^2) + 2^3 = 1000) := by
{
  sorry
}

end value_of_expression_l137_137239


namespace circle_radius_condition_l137_137348

theorem circle_radius_condition (c : ℝ) : 
  (∃ x y : ℝ, (x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ 
  (radius = 6) ↔ 
  c = -23 := by
  sorry

end circle_radius_condition_l137_137348


namespace problem1_problem2_1_problem2_2_l137_137241

-- Define the quadratic function and conditions
def quadratic (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

-- Problem 1: Expression of the quadratic function given vertex
theorem problem1 (b c : ℝ) : (quadratic 2 b c = 0) ∧ (∀ x : ℝ, quadratic x b c = (x - 2)^2) ↔ (b = -4) ∧ (c = 4) := sorry

-- Problem 2.1: Given n < -5 and y1 = y2, range of b + c
theorem problem2_1 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : quadratic (3*n - 4) b c = y1)
  (h3 : quadratic (5*n + 6) b c = y2) (h4 : y1 = y2) : b + c < -38 := sorry

-- Problem 2.2: Given n < -5 and c > 0, compare values of y1 and y2
theorem problem2_2 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : c > 0) 
  (h3 : quadratic (3*n - 4) b c = y1) (h4 : quadratic (5*n + 6) b c = y2) : y1 < y2 := sorry

end problem1_problem2_1_problem2_2_l137_137241


namespace trigonometric_identity_l137_137543

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + Real.pi / 6) / Real.sin (4 * α - Real.pi / 6) :=
sorry

end trigonometric_identity_l137_137543


namespace spiders_make_webs_l137_137018

theorem spiders_make_webs :
  (∀ (s d : ℕ), s = 7 ∧ d = 7 → (∃ w : ℕ, w = s)) ∧
  (∀ (d w : ℕ), w = 1 ∧ d = 7 → (∃ s : ℕ, s = w)) →
  (∀ (s : ℕ), s = 1) :=
by
  sorry

end spiders_make_webs_l137_137018


namespace no_pos_int_solutions_l137_137130

theorem no_pos_int_solutions (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + 2^(2 * k) + 1 ≠ y^3 := by
  sorry

end no_pos_int_solutions_l137_137130


namespace smallest_number_of_groups_l137_137672

theorem smallest_number_of_groups
  (participants : ℕ)
  (max_group_size : ℕ)
  (h1 : participants = 36)
  (h2 : max_group_size = 12) :
  participants / max_group_size = 3 :=
by
  sorry

end smallest_number_of_groups_l137_137672


namespace total_leaves_l137_137163

def fernTypeA_fronds := 15
def fernTypeA_leaves_per_frond := 45
def fernTypeB_fronds := 20
def fernTypeB_leaves_per_frond := 30
def fernTypeC_fronds := 25
def fernTypeC_leaves_per_frond := 40

def fernTypeA_count := 4
def fernTypeB_count := 5
def fernTypeC_count := 3

theorem total_leaves : 
  fernTypeA_count * (fernTypeA_fronds * fernTypeA_leaves_per_frond) + 
  fernTypeB_count * (fernTypeB_fronds * fernTypeB_leaves_per_frond) + 
  fernTypeC_count * (fernTypeC_fronds * fernTypeC_leaves_per_frond) = 
  8700 := 
sorry

end total_leaves_l137_137163


namespace pure_imaginary_a_zero_l137_137641

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_a_zero (a : ℝ) (h : is_pure_imaginary (i / (1 + a * i))) : a = 0 :=
sorry

end pure_imaginary_a_zero_l137_137641


namespace formation_enthalpy_benzene_l137_137262

/-- Define the enthalpy changes based on given conditions --/
def ΔH_acetylene : ℝ := 226.7 -- kJ/mol for C₂H₂
def ΔH_benzene_formation : ℝ := 631.1 -- kJ for reactions forming C₆H₆
def ΔH_benzene_phase_change : ℝ := -33.9 -- kJ for phase change of C₆H₆

/-- Define the enthalpy change of formation for benzene --/
def ΔH_formation_benzene : ℝ := 3 * ΔH_acetylene + ΔH_benzene_formation + ΔH_benzene_phase_change

/-- Theorem stating the heat change in the reaction equals the calculated value --/
theorem formation_enthalpy_benzene :
  ΔH_formation_benzene = -82.9 :=
by
  sorry

end formation_enthalpy_benzene_l137_137262


namespace milk_production_days_l137_137297

variable {x : ℕ}

def daily_cow_production (x : ℕ) : ℚ := (x + 4) / ((x + 2) * (x + 3))

def total_daily_production (x : ℕ) : ℚ := (x + 5) * daily_cow_production x

def required_days (x : ℕ) : ℚ := (x + 9) / total_daily_production x

theorem milk_production_days : 
  required_days x = (x + 9) * (x + 2) * (x + 3) / ((x + 5) * (x + 4)) := 
by 
  sorry

end milk_production_days_l137_137297


namespace multiplication_identity_l137_137732

theorem multiplication_identity (x y z w : ℝ) (h1 : x = 2000) (h2 : y = 2992) (h3 : z = 0.2992) (h4 : w = 20) : 
  x * y * z * w = 4 * y^2 :=
by
  sorry

end multiplication_identity_l137_137732


namespace prob1_prob2_l137_137472

theorem prob1 : -2 + 5 - |(-8 : ℤ)| + (-5) = -10 := 
by
  sorry

theorem prob2 : (-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22 := 
by
  sorry

end prob1_prob2_l137_137472


namespace polynomial_divisibility_l137_137760

theorem polynomial_divisibility (A B : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^(205 : ℕ) + A * x + B = 0) : 
    A + B = -1 :=
by
  sorry

end polynomial_divisibility_l137_137760


namespace least_possible_value_expression_l137_137465

theorem least_possible_value_expression :
  ∃ min_value : ℝ, ∀ x : ℝ, ((x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019) ≥ min_value ∧ min_value = 2018 :=
by
  sorry

end least_possible_value_expression_l137_137465


namespace projectile_reaches_24_meters_l137_137548

theorem projectile_reaches_24_meters (h : ℝ) (t : ℝ) (v₀ : ℝ) :
  (h = -4.9 * t^2 + 19.6 * t) ∧ (h = 24) → t = 4 :=
by
  intros
  sorry

end projectile_reaches_24_meters_l137_137548


namespace percentage_democrats_l137_137508

/-- In a certain city, some percent of the registered voters are Democrats and the rest are Republicans. In a mayoral race, 85 percent of the registered voters who are Democrats and 20 percent of the registered voters who are Republicans are expected to vote for candidate A. Candidate A is expected to get 59 percent of the registered voters' votes. Prove that 60 percent of the registered voters are Democrats. -/
theorem percentage_democrats (D R : ℝ) (h : D + R = 100) (h1 : 0.85 * D + 0.20 * R = 59) : 
  D = 60 :=
by
  sorry

end percentage_democrats_l137_137508


namespace common_difference_of_arithmetic_sequence_l137_137845

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l137_137845


namespace intercept_sum_l137_137445

theorem intercept_sum (x0 y0 : ℕ) (h1 : x0 < 17) (h2 : y0 < 17)
  (hx : 7 * x0 ≡ 2 [MOD 17]) (hy : 3 * y0 ≡ 15 [MOD 17]) : x0 + y0 = 17 :=
sorry

end intercept_sum_l137_137445


namespace pencil_cost_l137_137994

-- Definitions of given conditions
def has_amount : ℝ := 5.00  -- Elizabeth has 5 dollars
def borrowed_amount : ℝ := 0.53  -- She borrowed 53 cents
def needed_amount : ℝ := 0.47  -- She needs 47 cents more

-- Theorem to prove the cost of the pencil
theorem pencil_cost : has_amount + borrowed_amount + needed_amount = 6.00 := by 
  sorry

end pencil_cost_l137_137994


namespace find_lcm_of_two_numbers_l137_137196

theorem find_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (prod : ℕ) 
  (h1 : hcf = 22) (h2 : prod = 62216) (h3 : A * B = prod) (h4 : Nat.gcd A B = hcf) :
  Nat.lcm A B = 2828 := 
by
  sorry

end find_lcm_of_two_numbers_l137_137196


namespace repeating_decimal_fraction_l137_137859

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l137_137859


namespace carla_book_count_l137_137139

theorem carla_book_count (tiles_count books_count : ℕ) 
  (tiles_monday : tiles_count = 38)
  (total_tuesday_count : 2 * tiles_count + 3 * books_count = 301) : 
  books_count = 75 :=
by
  sorry

end carla_book_count_l137_137139


namespace smallest_number_l137_137265

theorem smallest_number (a b c d : ℝ) (h1 : a = -5) (h2 : b = 0) (h3 : c = 1/2) (h4 : d = Real.sqrt 2) : a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by
  sorry

end smallest_number_l137_137265


namespace charlie_cost_per_gb_l137_137022

noncomputable def total_data_usage (w1 w2 w3 w4 : ℕ) : ℕ := w1 + w2 + w3 + w4

noncomputable def data_over_limit (total_data usage_limit: ℕ) : ℕ :=
  if total_data > usage_limit then total_data - usage_limit else 0

noncomputable def cost_per_gb (extra_cost data_over_limit: ℕ) : ℕ :=
  if data_over_limit > 0 then extra_cost / data_over_limit else 0

theorem charlie_cost_per_gb :
  let D := 8
  let w1 := 2
  let w2 := 3
  let w3 := 5
  let w4 := 10
  let C := 120
  let total_data := total_data_usage w1 w2 w3 w4
  let data_over := data_over_limit total_data D
  C / data_over = 10 := by
  -- Sorry to skip the proof
  sorry

end charlie_cost_per_gb_l137_137022


namespace daughter_and_child_weight_l137_137067

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight_l137_137067


namespace sin_B_value_cos_A_value_l137_137470

theorem sin_B_value (A B C S : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) : 
  Real.sin B = 4/5 :=
sorry

theorem cos_A_value (A B C : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) 
  (h2: A - C = π/4)
  (h3: Real.sin B = 4/5) 
  (h4: Real.cos B = -3/5): 
  Real.cos A = Real.sqrt (50 + 5 * Real.sqrt 2) / 10 :=
sorry

end sin_B_value_cos_A_value_l137_137470


namespace t_shirts_left_yesterday_correct_l137_137008

-- Define the conditions
def t_shirts_left_yesterday (x : ℕ) : Prop :=
  let t_shirts_sold_morning := (3 / 5) * x
  let t_shirts_sold_afternoon := 180
  t_shirts_sold_morning = t_shirts_sold_afternoon

-- Prove that x = 300 given the above conditions
theorem t_shirts_left_yesterday_correct (x : ℕ) (h : t_shirts_left_yesterday x) : x = 300 :=
by
  sorry

end t_shirts_left_yesterday_correct_l137_137008


namespace percentage_of_40_l137_137129

theorem percentage_of_40 (P : ℝ) (h1 : 8/100 * 24 = 1.92) (h2 : P/100 * 40 + 1.92 = 5.92) : P = 10 :=
sorry

end percentage_of_40_l137_137129


namespace bank_teller_bills_l137_137876

theorem bank_teller_bills (x y : ℕ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
by
  sorry

end bank_teller_bills_l137_137876


namespace tangent_line_a_value_l137_137115

theorem tangent_line_a_value (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y - 1 = 0 → x^2 + y^2 + 4 * x = 0) → a = -1 / 4 :=
by
  sorry

end tangent_line_a_value_l137_137115


namespace libraryRoomNumber_l137_137988

-- Define the conditions
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def isPrime (n : ℕ) : Prop := Nat.Prime n
def isEven (n : ℕ) : Prop := n % 2 = 0
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def hasDigit7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Main theorem
theorem libraryRoomNumber (n : ℕ) (h1 : isTwoDigit n)
  (h2 : (isPrime n ∧ isEven n ∧ isDivisibleBy5 n ∧ hasDigit7 n) ↔ false)
  : n % 10 = 0 := 
sorry

end libraryRoomNumber_l137_137988


namespace calc_pow_l137_137170

-- Definitions used in the conditions
def base := 2
def exp := 10
def power := 2 / 5

-- Given condition
def given_identity : Pow.pow base exp = 1024 := by sorry

-- Statement to be proved
theorem calc_pow : Pow.pow 1024 power = 16 := by
  -- Use the given identity and known exponentiation rules to derive the result
  sorry

end calc_pow_l137_137170


namespace value_at_2013_l137_137557

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f x = -f (-x)
axiom periodic_5 : ∀ x : ℝ, f (x + 5) ≥ f x
axiom periodic_1 : ∀ x : ℝ, f (x + 1) ≤ f x

theorem value_at_2013 : f 2013 = 0 :=
by
  -- Proof goes here
  sorry

end value_at_2013_l137_137557


namespace surface_area_of_cube_l137_137648

-- Define the condition: volume of the cube is 1728 cubic centimeters
def volume_cube (s : ℝ) : ℝ := s^3
def given_volume : ℝ := 1728

-- Define the question: surface area of the cube
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- The statement that needs to be proved
theorem surface_area_of_cube :
  ∃ s : ℝ, volume_cube s = given_volume → surface_area_cube s = 864 :=
by
  sorry

end surface_area_of_cube_l137_137648


namespace distinct_digits_unique_D_l137_137020

theorem distinct_digits_unique_D 
  (A B C D : ℕ)
  (hA : A ≠ B)
  (hB : B ≠ C)
  (hC : C ≠ D)
  (hD : D ≠ A)
  (h1 : D < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : A < 10)
  (h_add : A * 1000 + A * 100 + C * 10 + B + B * 1000 + C * 100 + B * 10 + D = B * 1000 + D * 100 + A * 10 + B) :
  D = 0 :=
by sorry

end distinct_digits_unique_D_l137_137020


namespace billiard_ball_radius_unique_l137_137860

noncomputable def radius_of_billiard_balls (r : ℝ) : Prop :=
  let side_length := 292
  let lhs := (8 + 2 * Real.sqrt 3) * r
  lhs = side_length

theorem billiard_ball_radius_unique (r : ℝ) : radius_of_billiard_balls r → r = (146 / 13) * (4 - Real.sqrt 3 / 3) :=
by
  intro h1
  sorry

end billiard_ball_radius_unique_l137_137860


namespace roses_cut_l137_137871

variable (initial final : ℕ) -- Declare variables for initial and final numbers of roses

-- Define the theorem stating the solution
theorem roses_cut (h1 : initial = 6) (h2 : final = 16) : final - initial = 10 :=
sorry -- Use sorry to skip the proof

end roses_cut_l137_137871


namespace statement_B_statement_C_statement_D_l137_137989

variables (a b : ℝ)

-- Condition: a > 0
axiom a_pos : a > 0

-- Condition: e^a + ln b = 1
axiom eq1 : Real.exp a + Real.log b = 1

-- Statement B: a + ln b < 0
theorem statement_B : a + Real.log b < 0 :=
  sorry

-- Statement C: e^a + b > 2
theorem statement_C : Real.exp a + b > 2 :=
  sorry

-- Statement D: a + b > 1
theorem statement_D : a + b > 1 :=
  sorry

end statement_B_statement_C_statement_D_l137_137989


namespace Roger_first_bag_candies_is_11_l137_137690

-- Define the conditions
def Sandra_bags : ℕ := 2
def Sandra_candies_per_bag : ℕ := 6
def Roger_bags : ℕ := 2
def Roger_second_bag_candies : ℕ := 3
def Extra_candies_Roger_has_than_Sandra : ℕ := 2

-- Define the total candy for Sandra
def Sandra_total_candies : ℕ := Sandra_bags * Sandra_candies_per_bag

-- Using the conditions, we define the total candy for Roger
def Roger_total_candies : ℕ := Sandra_total_candies + Extra_candies_Roger_has_than_Sandra

-- Define the candy in Roger's first bag
def Roger_first_bag_candies : ℕ := Roger_total_candies - Roger_second_bag_candies

-- The proof statement we need to prove
theorem Roger_first_bag_candies_is_11 : Roger_first_bag_candies = 11 := by
  sorry

end Roger_first_bag_candies_is_11_l137_137690


namespace other_acute_angle_measure_l137_137652

-- Definitions based on the conditions
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90
def is_right_triangle (a b : ℝ) : Prop := right_triangle_sum a b ∧ a = 20

-- The statement to prove
theorem other_acute_angle_measure {a b : ℝ} (h : is_right_triangle a b) : b = 70 :=
sorry

end other_acute_angle_measure_l137_137652


namespace balloon_arrangements_l137_137452

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

end balloon_arrangements_l137_137452


namespace find_fourth_intersection_point_l137_137367

theorem find_fourth_intersection_point 
  (a b r: ℝ) 
  (h4 : ∃ a b r, ∀ x y, (x - a)^2 + (y - b)^2 = r^2 → (x, y) = (4, 1) ∨ (x, y) = (-2, -2) ∨ (x, y) = (8, 1/2) ∨ (x, y) = (-1/4, -16)):
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → x * y = 4 → (x, y) = (-1/4, -16) := 
sorry

end find_fourth_intersection_point_l137_137367


namespace remainder_sum_mod_15_l137_137717

variable (k j : ℤ) -- these represent any integers

def p := 60 * k + 53
def q := 75 * j + 24

theorem remainder_sum_mod_15 :
  (p k + q j) % 15 = 2 :=  
by 
  sorry

end remainder_sum_mod_15_l137_137717


namespace simplify_fraction_l137_137469

theorem simplify_fraction (a : ℝ) (h : a = 2) : (24 * a^5) / (72 * a^3) = 4 / 3 := by
  sorry

end simplify_fraction_l137_137469


namespace residue_mod_13_l137_137574

theorem residue_mod_13 :
  (250 ≡ 3 [MOD 13]) → 
  (20 ≡ 7 [MOD 13]) → 
  (5^2 ≡ 12 [MOD 13]) → 
  ((250 * 11 - 20 * 6 + 5^2) % 13 = 3) :=
by 
  sorry

end residue_mod_13_l137_137574


namespace find_f_2_l137_137968

def f (x : ℕ) : ℤ := sorry

axiom func_def : ∀ x : ℕ, f (x + 1) = x^2 - x

theorem find_f_2 : f 2 = 0 :=
by
  sorry

end find_f_2_l137_137968


namespace ax_plus_by_equals_d_set_of_solutions_l137_137344

theorem ax_plus_by_equals_d (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  ∀ (x y : ℤ), (a * x + b * y = d) ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a :=
by
  sorry

theorem set_of_solutions (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by
  sorry

end ax_plus_by_equals_d_set_of_solutions_l137_137344


namespace sets_of_consecutive_integers_summing_to_20_l137_137684

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l137_137684


namespace polygon_sides_l137_137317

def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360

theorem polygon_sides (n : ℕ) (h : 1/4 * sum_interior_angles n - sum_exterior_angles = 90) : n = 12 := 
by
  -- sorry to skip the proof
  sorry

end polygon_sides_l137_137317


namespace sets_equal_l137_137892

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }
def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem sets_equal : E = F :=
  sorry

end sets_equal_l137_137892


namespace pilot_fish_final_speed_relative_to_ocean_l137_137403

-- Define conditions
def keanu_speed : ℝ := 20 -- Keanu's speed in mph
def wind_speed : ℝ := 5 -- Wind speed in mph
def shark_speed (initial_speed: ℝ) : ℝ := 2 * initial_speed -- Shark doubles its speed

-- The pilot fish increases its speed by half the shark's increase
def pilot_fish_speed (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  initial_pilot_fish_speed + 0.5 * shark_initial_speed

-- Define the speed of the pilot fish relative to the ocean
def pilot_fish_speed_relative_to_ocean (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  pilot_fish_speed initial_pilot_fish_speed shark_initial_speed - wind_speed

-- Initial assumptions
def initial_pilot_fish_speed : ℝ := keanu_speed -- Pilot fish initially swims at the same speed as Keanu
def initial_shark_speed : ℝ := keanu_speed -- Let us assume the shark initially swims at the same speed as Keanu for simplicity

-- Prove the final speed of the pilot fish relative to the ocean
theorem pilot_fish_final_speed_relative_to_ocean : 
  pilot_fish_speed_relative_to_ocean initial_pilot_fish_speed initial_shark_speed = 25 := 
by sorry

end pilot_fish_final_speed_relative_to_ocean_l137_137403


namespace tan_cos_identity_l137_137131

theorem tan_cos_identity :
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  (tanθ^2 - cosθ^2) / (tanθ^2 * cosθ^2) = -5 / 3 :=
by
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  have h_tan : tanθ^2 = (Real.sin θ)^2 / (Real.cos θ)^2 := by sorry; -- Given condition 1
  have h_cos : cosθ^2 = 3 / 4 := by sorry; -- Given condition 2
  -- Prove the statement
  sorry

end tan_cos_identity_l137_137131


namespace polynomial_integer_roots_l137_137998

theorem polynomial_integer_roots (b1 b2 : ℤ) (x : ℤ) (h : x^3 + b2 * x^2 + b1 * x + 18 = 0) :
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
sorry

end polynomial_integer_roots_l137_137998


namespace factorize_problem1_factorize_problem2_l137_137177

-- Problem 1
theorem factorize_problem1 (a b : ℝ) : 
    -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := 
by sorry

-- Problem 2
theorem factorize_problem2 (a b x y : ℝ) : 
    9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) := 
by sorry

end factorize_problem1_factorize_problem2_l137_137177


namespace negated_roots_quadratic_reciprocals_roots_quadratic_l137_137326

-- For (1)
theorem negated_roots_quadratic (x y : ℝ) : 
    (x^2 + 3 * x - 2 = 0) ↔ (y^2 - 3 * y - 2 = 0) :=
sorry

-- For (2)
theorem reciprocals_roots_quadratic (a b c x y : ℝ) (h : a ≠ 0) :
    (a * x^2 - b * x + c = 0) ↔ (c * y^2 - b * y + a = 0) :=
sorry

end negated_roots_quadratic_reciprocals_roots_quadratic_l137_137326


namespace find_original_number_l137_137725

def original_number (x : ℝ) : Prop :=
  let step1 := 1.20 * x
  let step2 := step1 * 0.85
  let final_value := step2 * 1.30
  final_value = 1080

theorem find_original_number : ∃ x : ℝ, original_number x :=
by
  use 1080 / (1.20 * 0.85 * 1.30)
  sorry

end find_original_number_l137_137725


namespace eric_days_waited_l137_137206

def num_chickens := 4
def eggs_per_chicken_per_day := 3
def total_eggs := 36

def eggs_per_day := num_chickens * eggs_per_chicken_per_day
def num_days := total_eggs / eggs_per_day

theorem eric_days_waited : num_days = 3 :=
by
  sorry

end eric_days_waited_l137_137206


namespace at_least_one_genuine_l137_137544

theorem at_least_one_genuine :
  ∀ (total_products genuine_products defective_products selected_products : ℕ),
  total_products = 12 →
  genuine_products = 10 →
  defective_products = 2 →
  selected_products = 3 →
  (∃ g d : ℕ, g + d = selected_products ∧ g = 0 ∧ d = selected_products) = false :=
by
  intros total_products genuine_products defective_products selected_products
  intros H_total H_gen H_def H_sel
  sorry

end at_least_one_genuine_l137_137544


namespace power_function_value_l137_137915

-- Given conditions
def f : ℝ → ℝ := fun x => x^(1 / 3)

theorem power_function_value :
  f (Real.log 5 / (Real.log 2 * 8) + Real.log 160 / (Real.log (1 / 2))) = -2 := by
  sorry

end power_function_value_l137_137915


namespace negation_correct_l137_137145

-- Define the initial statement
def initial_statement (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |x| ≥ 3

-- Define the negated statement
def negated_statement (s : Set ℝ) : Prop :=
  ∃ x ∈ s, |x| < 3

-- The theorem to be proven
theorem negation_correct (s : Set ℝ) :
  ¬(initial_statement s) ↔ negated_statement s := by
  sorry

end negation_correct_l137_137145


namespace circus_tent_capacity_l137_137219

theorem circus_tent_capacity (num_sections : ℕ) (people_per_section : ℕ) 
  (h1 : num_sections = 4) (h2 : people_per_section = 246) :
  num_sections * people_per_section = 984 :=
by
  sorry

end circus_tent_capacity_l137_137219


namespace log_identity_l137_137363

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_identity
    (a b c : ℝ)
    (h1 : a ^ 2 + b ^ 2 = c ^ 2)
    (h2 : a > 0)
    (h3 : c > 0)
    (h4 : b > 0)
    (h5 : c > b) :
    log_base (c + b) a + log_base (c - b) a = 2 * log_base (c + b) a * log_base (c - b) a :=
sorry

end log_identity_l137_137363


namespace find_diagonal_length_l137_137923

noncomputable def parallelepiped_diagonal_length 
  (s : ℝ) -- Side length of square face
  (h : ℝ) -- Length of vertical edge
  (θ : ℝ) -- Angle between vertical edge and square face edges
  (hsq : s = 5) -- Length of side of the square face ABCD
  (hedge : h = 5) -- Length of vertical edge AA1
  (θdeg : θ = 60) -- Angle in degrees
  : ℝ :=
5 * Real.sqrt 3

-- The main theorem to be proved
theorem find_diagonal_length
  (s : ℝ)
  (h : ℝ)
  (θ : ℝ)
  (hsq : s = 5)
  (hedge : h = 5)
  (θdeg : θ = 60)
  : parallelepiped_diagonal_length s h θ hsq hedge θdeg = 5 * Real.sqrt 3 := 
sorry

end find_diagonal_length_l137_137923


namespace find_t_l137_137553

variables (t : ℝ)

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, t)
def a_plus_b : ℝ × ℝ := (2, 1 + t)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_t (t : ℝ) :
  are_parallel (3, t) (2, 1 + t) ↔ t = -3 :=
sorry

end find_t_l137_137553


namespace time_difference_in_minutes_l137_137035

def speed := 60 -- speed of the car in miles per hour
def distance1 := 360 -- distance of the first trip in miles
def distance2 := 420 -- distance of the second trip in miles
def hours_to_minutes := 60 -- conversion factor from hours to minutes

theorem time_difference_in_minutes :
  ((distance2 / speed) - (distance1 / speed)) * hours_to_minutes = 60 :=
by
  -- proof to be provided
  sorry

end time_difference_in_minutes_l137_137035


namespace group_size_l137_137185

def total_people (I N B Ne : ℕ) : ℕ := I + N - B + B + Ne

theorem group_size :
  let I := 55
  let N := 43
  let B := 61
  let Ne := 63
  total_people I N B Ne = 161 :=
by
  sorry

end group_size_l137_137185


namespace range_of_a_l137_137735

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1/2 * x^2

theorem range_of_a (a : ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a (x1 + a) - f a (x2 + a)) / (x1 - x2) ≥ 3) :
  a ≥ 9 / 4 :=
sorry

end range_of_a_l137_137735


namespace eggs_per_basket_l137_137039

theorem eggs_per_basket (red_eggs : ℕ) (orange_eggs : ℕ) (min_eggs : ℕ) :
  red_eggs = 30 → orange_eggs = 45 → min_eggs = 5 →
  (∃ k, (30 % k = 0) ∧ (45 % k = 0) ∧ (k ≥ 5) ∧ k = 15) :=
by
  intros h1 h2 h3
  use 15
  sorry

end eggs_per_basket_l137_137039


namespace new_perimeter_of_rectangle_l137_137664

theorem new_perimeter_of_rectangle (w : ℝ) (A : ℝ) (new_area_factor : ℝ) (L : ℝ) (L' : ℝ) (P' : ℝ) 
  (h_w : w = 10) (h_A : A = 150) (h_new_area_factor: new_area_factor = 4 / 3)
  (h_orig_length : L = A / w) (h_new_area: A' = new_area_factor * A) (h_A' : A' = 200)
  (h_new_length : L' = A' / w) (h_perimeter : P' = 2 * (L' + w)) 
  : P' = 60 :=
sorry

end new_perimeter_of_rectangle_l137_137664


namespace bamboo_fifth_section_volume_l137_137487

theorem bamboo_fifth_section_volume
  (a₁ q : ℝ)
  (h1 : a₁ * (a₁ * q) * (a₁ * q^2) = 3)
  (h2 : (a₁ * q^6) * (a₁ * q^7) * (a₁ * q^8) = 9) :
  a₁ * q^4 = Real.sqrt 3 :=
sorry

end bamboo_fifth_section_volume_l137_137487


namespace percentage_cd_only_l137_137763

noncomputable def percentage_power_windows : ℝ := 0.60
noncomputable def percentage_anti_lock_brakes : ℝ := 0.40
noncomputable def percentage_cd_player : ℝ := 0.75
noncomputable def percentage_gps_system : ℝ := 0.50
noncomputable def percentage_pw_and_abs : ℝ := 0.10
noncomputable def percentage_abs_and_cd : ℝ := 0.15
noncomputable def percentage_pw_and_cd : ℝ := 0.20
noncomputable def percentage_gps_and_abs : ℝ := 0.12
noncomputable def percentage_gps_and_cd : ℝ := 0.18
noncomputable def percentage_pw_and_gps : ℝ := 0.25

theorem percentage_cd_only : 
  percentage_cd_player - (percentage_abs_and_cd + percentage_pw_and_cd + percentage_gps_and_cd) = 0.22 := 
by
  sorry

end percentage_cd_only_l137_137763


namespace sufficient_not_necessary_l137_137117

theorem sufficient_not_necessary (a b : ℝ) : (a^2 + b^2 ≤ 2) → (-1 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((-1 ≤ a * b ∧ a * b ≤ 1) → a^2 + b^2 ≤ 2) := 
by
  sorry

end sufficient_not_necessary_l137_137117


namespace solve_furniture_factory_l137_137288

variable (num_workers : ℕ) (tables_per_worker : ℕ) (legs_per_worker : ℕ) 
variable (tabletop_workers legs_workers : ℕ)

axiom worker_capacity : tables_per_worker = 3 ∧ legs_per_worker = 6
axiom total_workers : num_workers = 60
axiom table_leg_ratio : ∀ (x : ℕ), tabletop_workers = x → legs_workers = (num_workers - x)
axiom daily_production_eq : ∀ (x : ℕ), (4 * tables_per_worker * x = 6 * legs_per_worker * (num_workers - x))

theorem solve_furniture_factory : 
  ∃ (x y : ℕ), num_workers = x + y ∧ 
            4 * 3 * x = 6 * (num_workers - x) ∧ 
            x = 20 ∧ y = (num_workers - 20) := by
  sorry

end solve_furniture_factory_l137_137288


namespace probability_of_specific_sequence_l137_137437

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l137_137437


namespace calculate_expression_l137_137640

theorem calculate_expression :
  (1/4 * 6.16^2) - (4 * 1.04^2) = 5.16 :=
by
  sorry

end calculate_expression_l137_137640


namespace solve_for_x_l137_137594

theorem solve_for_x : ∃ x : ℤ, 25 - 7 = 3 + x ∧ x = 15 := by
  sorry

end solve_for_x_l137_137594


namespace compute_value_l137_137632

variable (p q : ℚ)
variable (h : ∀ x, 3 * x^2 - 7 * x - 6 = 0 → x = p ∨ x = q)

theorem compute_value (h_pq : p ≠ q) : (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  -- We assume p and q are the roots of the polynomial and p ≠ q.
  have sum_roots : p + q = 7 / 3 := sorry
  have prod_roots : p * q = -2 := sorry
  -- Additional steps to derive the required result (proof) are ignored here.
  sorry

end compute_value_l137_137632


namespace x_intercept_of_perpendicular_line_is_16_over_3_l137_137990

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l137_137990


namespace number_of_red_balls_eq_47_l137_137271

theorem number_of_red_balls_eq_47
  (T : ℕ) (white green yellow purple : ℕ)
  (neither_red_nor_purple_prob : ℚ)
  (hT : T = 100)
  (hWhite : white = 10)
  (hGreen : green = 30)
  (hYellow : yellow = 10)
  (hPurple : purple = 3)
  (hProb : neither_red_nor_purple_prob = 0.5)
  : T - (white + green + yellow + purple) = 47 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end number_of_red_balls_eq_47_l137_137271


namespace cost_per_square_meter_l137_137082

theorem cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (hlength : length = 20) 
  (hwidth : width = 15) 
  (hheight : height = 5) 
  (hmoney : total_expenditure = 38000) : 
  58.46 = total_expenditure / (length * width + 2 * length * height + 2 * width * height) :=
by 
  -- Let's assume our definitions and use sorry to skip the proof
  sorry

end cost_per_square_meter_l137_137082


namespace number_of_ways_to_form_committee_with_president_l137_137982

open Nat

def number_of_ways_to_choose_members (total_members : ℕ) (committee_size : ℕ) (president_required : Bool) : ℕ :=
  if president_required then choose (total_members - 1) (committee_size - 1) else choose total_members committee_size

theorem number_of_ways_to_form_committee_with_president :
  number_of_ways_to_choose_members 30 5 true = 23741 :=
by
  -- Given that total_members = 30, committee_size = 5, and president_required = true,
  -- we need to show that the number of ways to choose the remaining members is 23741.
  sorry

end number_of_ways_to_form_committee_with_president_l137_137982


namespace monotonic_intervals_l137_137021

open Set

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * a * x^3 + x^2 + 1

theorem monotonic_intervals (a : ℝ) (h : a ≤ 0) :
  (a = 0 → (∀ x : ℝ, (x < 0 → deriv (f a) x < 0) ∧ (0 < x → deriv (f a) x > 0))) ∧
  (a < 0 → (∀ x : ℝ, (x < 2 / a → deriv (f a) x > 0 ∨ deriv (f a) x = 0) ∧ 
                     (2 / a < x → deriv (f a) x < 0 ∨ deriv (f a) x = 0))) :=
by
  sorry

end monotonic_intervals_l137_137021


namespace percentage_of_cash_is_20_l137_137200

theorem percentage_of_cash_is_20
  (raw_materials : ℕ)
  (machinery : ℕ)
  (total_amount : ℕ)
  (h_raw_materials : raw_materials = 35000)
  (h_machinery : machinery = 40000)
  (h_total_amount : total_amount = 93750) :
  (total_amount - (raw_materials + machinery)) * 100 / total_amount = 20 :=
by
  sorry

end percentage_of_cash_is_20_l137_137200


namespace probability_at_most_one_correct_in_two_rounds_l137_137844

theorem probability_at_most_one_correct_in_two_rounds :
  let pA := 3 / 5
  let pB := 2 / 3
  let pA_incorrect := 1 - pA
  let pB_incorrect := 1 - pB
  let p_0_correct := pA_incorrect * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A1 := pA * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A2 := pA_incorrect * pA * pB_incorrect * pB_incorrect
  let p_1_correct_B1 := pA_incorrect * pA_incorrect * pB * pB_incorrect
  let p_1_correct_B2 := pA_incorrect * pA_incorrect * pB_incorrect * pB
  let p_at_most_one := p_0_correct + p_1_correct_A1 + p_1_correct_A2 + 
      p_1_correct_B1 + p_1_correct_B2
  p_at_most_one = 32 / 225 := 
  sorry

end probability_at_most_one_correct_in_two_rounds_l137_137844


namespace right_angle_locus_l137_137560

noncomputable def P (x y : ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

theorem right_angle_locus (x y : ℝ) : P x y → x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2 :=
by
  sorry

end right_angle_locus_l137_137560


namespace distance_travelled_downstream_in_12_minutes_l137_137224

noncomputable def speed_boat_still : ℝ := 15 -- in km/hr
noncomputable def rate_current : ℝ := 3 -- in km/hr
noncomputable def time_downstream : ℝ := 12 / 60 -- in hr (since 12 minutes is 12/60 hours)
noncomputable def effective_speed_downstream : ℝ := speed_boat_still + rate_current -- in km/hr
noncomputable def distance_downstream := effective_speed_downstream * time_downstream -- in km

theorem distance_travelled_downstream_in_12_minutes :
  distance_downstream = 3.6 := 
by
  sorry

end distance_travelled_downstream_in_12_minutes_l137_137224


namespace paint_cost_l137_137650

theorem paint_cost (l : ℝ) (b : ℝ) (rate : ℝ) (area : ℝ) (cost : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : l = 18.9999683334125) 
  (h3 : rate = 3.00001) 
  (h4 : area = l * b) 
  (h5 : cost = area * rate) : 
  cost = 361.00 :=
by
  sorry

end paint_cost_l137_137650


namespace sum_odd_implies_parity_l137_137071

theorem sum_odd_implies_parity (a b c: ℤ) (h: (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 := 
sorry

end sum_odd_implies_parity_l137_137071


namespace B_completes_work_in_12_hours_l137_137806

theorem B_completes_work_in_12_hours:
  let A := 1 / 4
  let C := (1 / 2) - A
  let B := (1 / 3) - C
  (1 / B) = 12 :=
by
  -- placeholder for the proof
  sorry

end B_completes_work_in_12_hours_l137_137806


namespace combined_salaries_of_A_B_C_D_l137_137775

theorem combined_salaries_of_A_B_C_D (salaryE : ℕ) (avg_salary : ℕ) (num_people : ℕ)
    (h1 : salaryE = 9000) (h2 : avg_salary = 8800) (h3 : num_people = 5) :
    (avg_salary * num_people) - salaryE = 35000 :=
by
  sorry

end combined_salaries_of_A_B_C_D_l137_137775


namespace initial_population_l137_137079

theorem initial_population (P : ℝ) (h1 : ∀ t : ℕ, P * (1.10 : ℝ) ^ t = 26620 → t = 3) : P = 20000 := by
  have h2 : P * (1.10) ^ 3 = 26620 := sorry
  sorry

end initial_population_l137_137079


namespace Diane_age_when_conditions_met_l137_137657

variable (Diane_current : ℕ) (Alex_current : ℕ) (Allison_current : ℕ)
variable (D : ℕ)

axiom Diane_current_age : Diane_current = 16
axiom Alex_Allison_sum : Alex_current + Allison_current = 47
axiom Diane_half_Alex : D = (Alex_current + (D - 16)) / 2
axiom Diane_twice_Allison : D = 2 * (Allison_current + (D - 16))

theorem Diane_age_when_conditions_met : D = 78 :=
by
  sorry

end Diane_age_when_conditions_met_l137_137657


namespace max_marks_paper_I_l137_137494

variable (M : ℝ)

theorem max_marks_paper_I (h1 : 0.65 * M = 112 + 58) : M = 262 :=
  sorry

end max_marks_paper_I_l137_137494


namespace arithmetic_sequence_sum_l137_137142

-- Define the arithmetic sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := sorry

-- Given condition
axiom h1 : a_n 3 + a_n 7 = 37

-- Proof statement
theorem arithmetic_sequence_sum : a_n 2 + a_n 4 + a_n 6 + a_n 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l137_137142


namespace skittles_total_correct_l137_137102

def number_of_students : ℕ := 9
def skittles_per_student : ℕ := 3
def total_skittles : ℕ := 27

theorem skittles_total_correct : number_of_students * skittles_per_student = total_skittles := by
  sorry

end skittles_total_correct_l137_137102


namespace percentage_of_students_enrolled_is_40_l137_137547

def total_students : ℕ := 880
def not_enrolled_in_biology : ℕ := 528
def enrolled_in_biology : ℕ := total_students - not_enrolled_in_biology
def percentage_enrolled : ℕ := (enrolled_in_biology * 100) / total_students

theorem percentage_of_students_enrolled_is_40 : percentage_enrolled = 40 := by
  -- Beginning of the proof
  sorry

end percentage_of_students_enrolled_is_40_l137_137547


namespace exterior_angle_DEG_l137_137604

-- Define the degree measures of angles in a square and a pentagon.
def square_interior_angle := 90
def pentagon_interior_angle := 108

-- Define the sum of the adjacent interior angles at D
def adjacent_interior_sum := square_interior_angle + pentagon_interior_angle

-- Statement to prove the exterior angle DEG
theorem exterior_angle_DEG :
  360 - adjacent_interior_sum = 162 := by
  sorry

end exterior_angle_DEG_l137_137604


namespace problem_inequality_I_problem_inequality_II_l137_137967

theorem problem_inequality_I (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  1 / a + 1 / b ≥ 4 := sorry

theorem problem_inequality_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := sorry

end problem_inequality_I_problem_inequality_II_l137_137967


namespace find_y_value_l137_137404

theorem find_y_value (a y : ℕ) (h1 : (15^2) * y^3 / 256 = a) (h2 : a = 450) : y = 8 := 
by 
  sorry

end find_y_value_l137_137404


namespace find_a_l137_137984

theorem find_a (a : ℝ) (h : ∃ x, x = 3 ∧ x^2 + a * x + a - 1 = 0) : a = -2 :=
sorry

end find_a_l137_137984


namespace calculate_expression_l137_137532

theorem calculate_expression : 6 * (8 + 1/3) = 50 := by
  sorry

end calculate_expression_l137_137532


namespace all_numbers_divisible_by_5_l137_137570

variable {a b c d e f g : ℕ}

-- Seven natural numbers and the condition that the sum of any six is divisible by 5
axiom cond_a : (a + b + c + d + e + f) % 5 = 0
axiom cond_b : (b + c + d + e + f + g) % 5 = 0
axiom cond_c : (a + c + d + e + f + g) % 5 = 0
axiom cond_d : (a + b + c + e + f + g) % 5 = 0
axiom cond_e : (a + b + c + d + f + g) % 5 = 0
axiom cond_f : (a + b + c + d + e + g) % 5 = 0
axiom cond_g : (a + b + c + d + e + f) % 5 = 0

theorem all_numbers_divisible_by_5 :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end all_numbers_divisible_by_5_l137_137570


namespace m_squared_n_minus_1_l137_137602

theorem m_squared_n_minus_1 (a b m n : ℝ)
  (h1 : a * m^2001 + b * n^2001 = 3)
  (h2 : a * m^2002 + b * n^2002 = 7)
  (h3 : a * m^2003 + b * n^2003 = 24)
  (h4 : a * m^2004 + b * n^2004 = 102) :
  m^2 * (n - 1) = 6 := by
  sorry

end m_squared_n_minus_1_l137_137602


namespace blue_to_yellow_ratio_is_half_l137_137728

noncomputable section

def yellow_fish := 12
def blue_fish : ℕ := by 
  have total_fish := 42
  have green_fish := 2 * yellow_fish
  exact total_fish - (yellow_fish + green_fish)
def fish_ratio (x y : ℕ) := x / y

theorem blue_to_yellow_ratio_is_half : fish_ratio blue_fish yellow_fish = 1 / 2 := by
  sorry

end blue_to_yellow_ratio_is_half_l137_137728


namespace prime_5p_plus_4p4_is_perfect_square_l137_137607

theorem prime_5p_plus_4p4_is_perfect_square (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ q : ℕ, 5^p + 4 * p^4 = q^2 ↔ p = 5 :=
by
  sorry

end prime_5p_plus_4p4_is_perfect_square_l137_137607


namespace product_of_reciprocals_plus_one_geq_nine_l137_137243

theorem product_of_reciprocals_plus_one_geq_nine
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hab : a + b = 1) :
  (1 / a + 1) * (1 / b + 1) ≥ 9 :=
sorry

end product_of_reciprocals_plus_one_geq_nine_l137_137243


namespace percentage_less_than_l137_137308

theorem percentage_less_than (x y : ℝ) (h1 : y = x * 1.8181818181818181) : (∃ P : ℝ, P = 45) :=
by
  sorry

end percentage_less_than_l137_137308


namespace unique_solution_l137_137327

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l137_137327


namespace jacob_calories_l137_137939

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l137_137939


namespace trucks_have_160_containers_per_truck_l137_137497

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l137_137497


namespace value_of_expression_l137_137960

theorem value_of_expression (x y z : ℝ) (h : (x * y * z) / (|x * y * z|) = 1) :
  (|x| / x + y / |y| + |z| / z) = 3 ∨ (|x| / x + y / |y| + |z| / z) = -1 :=
sorry

end value_of_expression_l137_137960


namespace area_ratio_of_squares_l137_137903

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 16 * b) : a ^ 2 = 16 * b ^ 2 := by
  sorry

end area_ratio_of_squares_l137_137903


namespace sum_of_first_11_terms_l137_137695

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Condition: the sequence is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 1 + a 5 + a 9 = 39
axiom h2 : a 3 + a 7 + a 11 = 27
axiom h3 : is_arithmetic_sequence a d

-- Proof statement
theorem sum_of_first_11_terms : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = 121 := 
sorry

end sum_of_first_11_terms_l137_137695


namespace solve_for_a_l137_137689

theorem solve_for_a (a : ℝ) 
  (h : (2 * a + 16 + (3 * a - 8)) / 2 = 89) : 
  a = 34 := 
sorry

end solve_for_a_l137_137689


namespace monotonic_increasing_interval_l137_137310

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 2 < x → (f x < f (x + 1)) :=
by
  intros x h
  sorry

end monotonic_increasing_interval_l137_137310


namespace right_triangle_leg_length_l137_137431

theorem right_triangle_leg_length
  (A : ℝ)
  (b h : ℝ)
  (hA : A = 800)
  (hb : b = 40)
  (h_area : A = (1 / 2) * b * h) :
  h = 40 :=
by
  sorry

end right_triangle_leg_length_l137_137431


namespace total_people_on_playground_l137_137691

open Nat

-- Conditions
def num_girls := 28
def num_boys := 35
def num_3rd_grade_girls := 15
def num_3rd_grade_boys := 18
def num_teachers := 4

-- Derived values (from conditions)
def num_4th_grade_girls := num_girls - num_3rd_grade_girls
def num_4th_grade_boys := num_boys - num_3rd_grade_boys
def num_3rd_graders := num_3rd_grade_girls + num_3rd_grade_boys
def num_4th_graders := num_4th_grade_girls + num_4th_grade_boys

-- Total number of people
def total_people := num_3rd_graders + num_4th_graders + num_teachers

-- Proof statement
theorem total_people_on_playground : total_people = 67 :=
  by
     -- This is where the proof would go
     sorry

end total_people_on_playground_l137_137691


namespace geese_initial_formation_l137_137710

theorem geese_initial_formation (G : ℕ) 
  (h1 : G / 2 + 4 = 12) : G = 16 := 
sorry

end geese_initial_formation_l137_137710


namespace gcd_153_68_eq_17_l137_137817

theorem gcd_153_68_eq_17 : Int.gcd 153 68 = 17 :=
by
  sorry

end gcd_153_68_eq_17_l137_137817


namespace sum_of_cubes_form_l137_137321

theorem sum_of_cubes_form (a b : ℤ) (x1 y1 x2 y2 : ℤ)
  (h1 : a = x1^2 + 3 * y1^2) (h2 : b = x2^2 + 3 * y2^2) :
  ∃ x y : ℤ, a^3 + b^3 = x^2 + 3 * y^2 := sorry

end sum_of_cubes_form_l137_137321


namespace counterexample_to_proposition_l137_137624

theorem counterexample_to_proposition : ∃ (a : ℝ), a^2 > 0 ∧ a ≤ 0 :=
  sorry

end counterexample_to_proposition_l137_137624


namespace toothpaste_duration_l137_137776

theorem toothpaste_duration 
  (toothpaste_grams : ℕ)
  (dad_usage_per_brushing : ℕ) 
  (mom_usage_per_brushing : ℕ) 
  (anne_usage_per_brushing : ℕ) 
  (brother_usage_per_brushing : ℕ) 
  (brushes_per_day : ℕ) 
  (total_usage : ℕ) 
  (days : ℕ) 
  (h1 : toothpaste_grams = 105) 
  (h2 : dad_usage_per_brushing = 3) 
  (h3 : mom_usage_per_brushing = 2) 
  (h4 : anne_usage_per_brushing = 1) 
  (h5 : brother_usage_per_brushing = 1) 
  (h6 : brushes_per_day = 3)
  (h7 : total_usage = (3 * brushes_per_day) + (2 * brushes_per_day) + (1 * brushes_per_day) + (1 * brushes_per_day)) 
  (h8 : days = toothpaste_grams / total_usage) : 
  days = 5 :=
  sorry

end toothpaste_duration_l137_137776


namespace car_speed_l137_137276

variable (fuel_efficiency : ℝ) (fuel_decrease_gallons : ℝ) (time_hours : ℝ) 
          (gallons_to_liters : ℝ) (kilometers_to_miles : ℝ)
          (car_speed_mph : ℝ)

-- Conditions given in the problem
def fuelEfficiency : ℝ := 40 -- km per liter
def fuelDecreaseGallons : ℝ := 3.9 -- gallons
def timeHours : ℝ := 5.7 -- hours
def gallonsToLiters : ℝ := 3.8 -- liters per gallon
def kilometersToMiles : ℝ := 1.6 -- km per mile

theorem car_speed (fuel_efficiency fuelDecreaseGallons timeHours gallonsToLiters kilometersToMiles : ℝ) : 
  let fuelDecreaseLiters := fuelDecreaseGallons * gallonsToLiters
  let distanceKm := fuelDecreaseLiters * fuel_efficiency
  let distanceMiles := distanceKm / kilometersToMiles
  let averageSpeed := distanceMiles / timeHours
  averageSpeed = 65 := sorry

end car_speed_l137_137276


namespace unique_sum_of_two_cubes_lt_1000_l137_137879

theorem unique_sum_of_two_cubes_lt_1000 
  : ∃ (sums : Finset ℕ), 
    (∀ x ∈ sums, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ x = a^3 + b^3) 
    ∧ sums.card = 40 
    ∧ ∀ x ∈ sums, x < 1000 := 
by sorry

end unique_sum_of_two_cubes_lt_1000_l137_137879


namespace shauna_fifth_test_score_l137_137788

theorem shauna_fifth_test_score :
  ∀ (a1 a2 a3 a4: ℕ), a1 = 76 → a2 = 94 → a3 = 87 → a4 = 92 →
  (∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / 5 = 85 ∧ a5 = 76) :=
by
  sorry

end shauna_fifth_test_score_l137_137788


namespace mary_baking_cups_l137_137061

-- Conditions
def flour_needed : ℕ := 9
def sugar_needed : ℕ := 11
def flour_added : ℕ := 4
def sugar_added : ℕ := 0

-- Statement to prove
theorem mary_baking_cups : sugar_needed - (flour_needed - flour_added) = 6 := by
  sorry

end mary_baking_cups_l137_137061


namespace four_digit_sum_l137_137843

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end four_digit_sum_l137_137843


namespace graph_of_g_contains_1_0_and_sum_l137_137941

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : Prop := 3 * y = 2 * f (3 * x) + 4

theorem graph_of_g_contains_1_0_and_sum :
  f 3 = -2 → g 1 0 ∧ (1 + 0 = 1) :=
by
  intro h
  sorry

end graph_of_g_contains_1_0_and_sum_l137_137941


namespace hockey_team_ties_l137_137156

theorem hockey_team_ties (W T : ℕ) (h1 : 2 * W + T = 60) (h2 : W = T + 12) : T = 12 :=
by
  sorry

end hockey_team_ties_l137_137156


namespace quadratic_equation_solution_l137_137457

-- We want to prove that for the conditions given, the only possible value of m is 3
theorem quadratic_equation_solution (m : ℤ) (h1 : m^2 - 7 = 2) (h2 : m + 3 ≠ 0) : m = 3 :=
sorry

end quadratic_equation_solution_l137_137457


namespace height_percentage_difference_l137_137136

theorem height_percentage_difference (H : ℝ) (p r q : ℝ) 
  (hp : p = 0.60 * H) 
  (hr : r = 1.30 * H) : 
  (r - p) / p * 100 = 116.67 :=
by
  sorry

end height_percentage_difference_l137_137136


namespace largest_variable_l137_137936

theorem largest_variable {x y z w : ℤ} 
  (h1 : x + 3 = y - 4)
  (h2 : x + 3 = z + 2)
  (h3 : x + 3 = w - 1) :
  y > x ∧ y > z ∧ y > w :=
by sorry

end largest_variable_l137_137936


namespace fifteen_percent_of_x_l137_137253

variables (x : ℝ)

-- Condition: Given x% of 60 is 12
def is_x_percent_of_60 : Prop := (x / 100) * 60 = 12

-- Prove: 15% of x is 3
theorem fifteen_percent_of_x (h : is_x_percent_of_60 x) : (15 / 100) * x = 3 :=
by
  sorry

end fifteen_percent_of_x_l137_137253


namespace max_tickets_jane_can_buy_l137_137414

def ticket_price : ℝ := 15.75
def processing_fee : ℝ := 1.25
def jane_money : ℝ := 150.00

theorem max_tickets_jane_can_buy : ⌊jane_money / (ticket_price + processing_fee)⌋ = 8 := 
by
  sorry

end max_tickets_jane_can_buy_l137_137414


namespace farmer_initial_apples_l137_137809

variable (initial_apples given_away_apples remaining_apples : ℕ)

def initial_apple_count (given_away_apples remaining_apples : ℕ) : ℕ :=
  given_away_apples + remaining_apples

theorem farmer_initial_apples : initial_apple_count 88 39 = 127 := by
  -- Given conditions
  let given_away_apples := 88
  let remaining_apples := 39

  -- Calculate the initial apples
  let initial_apples := initial_apple_count given_away_apples remaining_apples

  -- We are supposed to prove initial apples count is 127
  show initial_apples = 127
  sorry

end farmer_initial_apples_l137_137809


namespace people_sharing_cookies_l137_137885

theorem people_sharing_cookies (total_cookies : ℕ) (cookies_per_person : ℕ) (people : ℕ) 
  (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) (h3 : total_cookies = cookies_per_person * people) : 
  people = 6 :=
by
  sorry

end people_sharing_cookies_l137_137885


namespace abs_neg_five_is_five_l137_137849

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l137_137849


namespace perpendicular_condition_l137_137301

theorem perpendicular_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → (m * x + y + 1 = 0 → (2 * m - 1 = 0))) ↔ (m = 1/2) :=
by sorry

end perpendicular_condition_l137_137301


namespace combined_swim_time_l137_137698

theorem combined_swim_time 
    (freestyle_time: ℕ)
    (backstroke_without_factors: ℕ)
    (backstroke_with_factors: ℕ)
    (butterfly_without_factors: ℕ)
    (butterfly_with_factors: ℕ)
    (breaststroke_without_factors: ℕ)
    (breaststroke_with_factors: ℕ) :
    freestyle_time = 48 ∧
    backstroke_without_factors = freestyle_time + 4 ∧
    backstroke_with_factors = backstroke_without_factors + 2 ∧
    butterfly_without_factors = backstroke_without_factors + 3 ∧
    butterfly_with_factors = butterfly_without_factors + 3 ∧
    breaststroke_without_factors = butterfly_without_factors + 2 ∧
    breaststroke_with_factors = breaststroke_without_factors - 1 →
    freestyle_time + backstroke_with_factors + butterfly_with_factors + breaststroke_with_factors = 216 :=
by
  sorry

end combined_swim_time_l137_137698


namespace tank_capacities_l137_137573

theorem tank_capacities (x y z : ℕ) 
  (h1 : x + y + z = 1620)
  (h2 : z = x + y / 5) 
  (h3 : z = y + x / 3) :
  x = 540 ∧ y = 450 ∧ z = 630 := 
by 
  sorry

end tank_capacities_l137_137573


namespace find_pairs_l137_137724

theorem find_pairs (a b : ℕ) (h1: a > 0) (h2: b > 0) (q r : ℕ)
  (h3: a^2 + b^2 = q * (a + b) + r) (h4: q^2 + r = 1977) : 
  (a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50) :=
sorry

end find_pairs_l137_137724


namespace sum_of_x_coords_l137_137158

theorem sum_of_x_coords (x : ℝ) (y : ℝ) :
  y = abs (x^2 - 6*x + 8) ∧ y = 6 - x → (x = (5 + Real.sqrt 17) / 2 ∨ x = (5 - Real.sqrt 17) / 2 ∨ x = 2)
  →  ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) :=
by
  intros h1 h2
  have H : ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) := sorry
  exact H

end sum_of_x_coords_l137_137158


namespace total_toes_on_bus_l137_137377

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l137_137377


namespace sum_diff_square_cube_l137_137471

/-- If the sum of two numbers is 25 and the difference between them is 15,
    then the difference between the square of the larger number and the cube of the smaller number is 275. -/
theorem sum_diff_square_cube (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x - y = 15) :
  x^2 - y^3 = 275 :=
sorry

end sum_diff_square_cube_l137_137471


namespace unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l137_137828

theorem unit_digit_of_product_of_nine_consecutive_numbers_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) % 10 = 0 :=
by
  sorry

end unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l137_137828


namespace num_possible_values_for_n_l137_137304

open Real

noncomputable def count_possible_values_for_n : ℕ :=
  let log2 := log 2
  let log2_9 := log 9 / log2
  let log2_50 := log 50 / log2
  let range_n := ((6 : ℕ), 450)
  let count := range_n.2 - range_n.1 + 1
  count

theorem num_possible_values_for_n :
  count_possible_values_for_n = 445 :=
by
  sorry

end num_possible_values_for_n_l137_137304


namespace minimum_value_l137_137352

theorem minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 5) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a) + (1 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l137_137352


namespace find_f_five_thirds_l137_137898

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l137_137898


namespace eighth_term_is_79_l137_137549

variable (a d : ℤ)

def fourth_term_condition : Prop := a + 3 * d = 23
def sixth_term_condition : Prop := a + 5 * d = 51

theorem eighth_term_is_79 (h₁ : fourth_term_condition a d) (h₂ : sixth_term_condition a d) : a + 7 * d = 79 :=
sorry

end eighth_term_is_79_l137_137549


namespace largest_number_of_stores_visited_l137_137931

theorem largest_number_of_stores_visited
  (stores : ℕ) (total_visits : ℕ) (total_peopled_shopping : ℕ)
  (people_visiting_2_stores : ℕ) (people_visiting_3_stores : ℕ)
  (people_visiting_4_stores : ℕ) (people_visiting_1_store : ℕ)
  (everyone_visited_at_least_one_store : ∀ p : ℕ, 0 < people_visiting_1_store + people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores)
  (h1 : stores = 15) (h2 : total_visits = 60) (h3 : total_peopled_shopping = 30)
  (h4 : people_visiting_2_stores = 12) (h5 : people_visiting_3_stores = 6)
  (h6 : people_visiting_4_stores = 4) (h7 : people_visiting_1_store = total_peopled_shopping - (people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores + 2)) :
  ∃ p : ℕ, ∀ person, person ≤ p ∧ p = 4 := sorry

end largest_number_of_stores_visited_l137_137931


namespace value_of_a_plus_b_l137_137805

theorem value_of_a_plus_b (a b : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b) 
  (hg : ∀ x, g x = -3 * x + 2)
  (hgf : ∀ x, g (f x) = -2 * x - 3) :
  a + b = 7 / 3 :=
by
  sorry

end value_of_a_plus_b_l137_137805


namespace discount_double_time_l137_137413

theorem discount_double_time (TD FV : ℝ) (h1 : TD = 10) (h2 : FV = 110) : 
  2 * TD = 20 :=
by
  sorry

end discount_double_time_l137_137413


namespace annual_rent_per_square_foot_is_172_l137_137813

def monthly_rent : ℕ := 3600
def local_taxes : ℕ := 500
def maintenance_fees : ℕ := 200
def length_of_shop : ℕ := 20
def width_of_shop : ℕ := 15

def total_monthly_cost : ℕ := monthly_rent + local_taxes + maintenance_fees
def annual_cost : ℕ := total_monthly_cost * 12
def area_of_shop : ℕ := length_of_shop * width_of_shop
def annual_rent_per_square_foot : ℕ := annual_cost / area_of_shop

theorem annual_rent_per_square_foot_is_172 :
  annual_rent_per_square_foot = 172 := by
    sorry

end annual_rent_per_square_foot_is_172_l137_137813


namespace polynomials_with_three_different_roots_count_l137_137745

theorem polynomials_with_three_different_roots_count :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6: ℕ), 
    a_0 = 0 ∧ 
    (a_6 = 0 ∨ a_6 = 1) ∧
    (a_5 = 0 ∨ a_5 = 1) ∧
    (a_4 = 0 ∨ a_4 = 1) ∧
    (a_3 = 0 ∨ a_3 = 1) ∧
    (a_2 = 0 ∨ a_2 = 1) ∧
    (a_1 = 0 ∨ a_1 = 1) ∧
    (1 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1) % 2 = 0 ∧
    (1 - a_6 + a_5 - a_4 + a_3 - a_2 + a_1) % 2 = 0) -> 
  ∃ (n : ℕ), n = 8 :=
sorry

end polynomials_with_three_different_roots_count_l137_137745


namespace product_of_p_r_s_l137_137077

theorem product_of_p_r_s (p r s : ℕ) 
  (h1 : 4^p + 4^3 = 280)
  (h2 : 3^r + 29 = 56) 
  (h3 : 7^s + 6^3 = 728) : 
  p * r * s = 27 :=
by
  sorry

end product_of_p_r_s_l137_137077


namespace problem_solution_l137_137328

theorem problem_solution :
  ∃ x y z : ℕ,
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x^2 + y^2 + z^2 = 2 * (y * z + 1) ∧
    x + y + z = 4032 ∧
    x^2 * y + z = 4031 :=
by
  sorry

end problem_solution_l137_137328


namespace binomial_constant_term_l137_137550

theorem binomial_constant_term (n : ℕ) (h : n > 0) :
  (∃ r : ℕ, n = 2 * r) ↔ (n = 6) :=
by
  sorry

end binomial_constant_term_l137_137550


namespace range_of_m_for_roots_greater_than_1_l137_137949

theorem range_of_m_for_roots_greater_than_1:
  ∀ m : ℝ, 
  (∀ x : ℝ, 8 * x^2 - (m - 1) * x + (m - 7) = 0 → 1 < x) ↔ 25 ≤ m :=
by
  sorry

end range_of_m_for_roots_greater_than_1_l137_137949


namespace eliminate_denominators_eq_l137_137031

theorem eliminate_denominators_eq :
  ∀ (x : ℝ), 1 - (x + 3) / 6 = x / 2 → 6 - x - 3 = 3 * x :=
by
  intro x
  intro h
  -- Place proof steps here.
  sorry

end eliminate_denominators_eq_l137_137031


namespace ab_is_zero_l137_137446

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l137_137446


namespace sufficient_but_not_necessary_pi_l137_137370

theorem sufficient_but_not_necessary_pi (x : ℝ) : 
  (x = Real.pi → Real.sin x = 0) ∧ (Real.sin x = 0 → ∃ k : ℤ, x = k * Real.pi) → ¬(Real.sin x = 0 → x = Real.pi) :=
by
  sorry

end sufficient_but_not_necessary_pi_l137_137370


namespace f_at_7_l137_137857

noncomputable def f (x : ℝ) (a b c d : ℝ) := a * x^7 + b * x^5 + c * x^3 + d * x + 5

theorem f_at_7 (a b c d : ℝ) (h : f (-7) a b c d = -7) : f 7 a b c d = 17 := 
by
  sorry

end f_at_7_l137_137857


namespace first_payment_amount_l137_137221

-- The number of total payments
def total_payments : Nat := 65

-- The number of the first payments
def first_payments : Nat := 20

-- The number of remaining payments
def remaining_payments : Nat := total_payments - first_payments

-- The extra amount added to the remaining payments
def extra_amount : Int := 65

-- The average payment
def average_payment : Int := 455

-- The total amount paid over the year
def total_amount_paid : Int := average_payment * total_payments

-- The variable we want to solve for: amount of each of the first 20 payments
variable (x : Int)

-- The equation for total amount paid
def total_payments_equation : Prop :=
  20 * x + 45 * (x + 65) = 455 * 65

-- The theorem stating the amount of each of the first 20 payments
theorem first_payment_amount : x = 410 :=
  sorry

end first_payment_amount_l137_137221


namespace at_least_half_team_B_can_serve_l137_137991

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l137_137991


namespace certain_number_minus_two_l137_137755

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := 
sorry

end certain_number_minus_two_l137_137755


namespace subscription_total_eq_14036_l137_137303

noncomputable def total_subscription (x : ℕ) : ℕ :=
  3 * x + 14000

theorem subscription_total_eq_14036 (c : ℕ) (profit_b : ℕ) (total_profit : ℕ) 
  (h1 : profit_b = 10200)
  (h2 : total_profit = 30000) 
  (h3 : (profit_b : ℝ) / (total_profit : ℝ) = (c + 5000 : ℝ) / (total_subscription c : ℝ)) :
  total_subscription c = 14036 :=
by
  sorry

end subscription_total_eq_14036_l137_137303


namespace trig_identity_cos_sin_l137_137792

theorem trig_identity_cos_sin : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
sorry

end trig_identity_cos_sin_l137_137792


namespace range_of_m_l137_137916

theorem range_of_m (m : ℝ) :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * m * x_0 + m + 2 < 0) ↔ (-1 : ℝ) ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l137_137916


namespace max_n_consecutive_sum_2014_l137_137450

theorem max_n_consecutive_sum_2014 : 
  ∃ (k n : ℕ), (2 * k + n - 1) * n = 4028 ∧ n = 53 ∧ k > 0 := sorry

end max_n_consecutive_sum_2014_l137_137450


namespace sum_of_roots_eq_h_over_4_l137_137491

theorem sum_of_roots_eq_h_over_4 (x1 x2 h b : ℝ) (h_ne : x1 ≠ x2)
  (hx1 : 4 * x1 ^ 2 - h * x1 = b) (hx2 : 4 * x2 ^ 2 - h * x2 = b) : x1 + x2 = h / 4 :=
sorry

end sum_of_roots_eq_h_over_4_l137_137491


namespace almonds_addition_l137_137014

theorem almonds_addition (walnuts almonds total_nuts : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_total_nuts : total_nuts = 0.5)
  (h_sum : total_nuts = walnuts + almonds) : 
  almonds = 0.25 := by
  sorry

end almonds_addition_l137_137014


namespace beaver_group_count_l137_137495

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l137_137495


namespace valid_shirt_tie_combinations_l137_137612

theorem valid_shirt_tie_combinations
  (num_shirts : ℕ)
  (num_ties : ℕ)
  (restricted_shirts : ℕ)
  (restricted_ties : ℕ)
  (h_shirts : num_shirts = 8)
  (h_ties : num_ties = 7)
  (h_restricted_shirts : restricted_shirts = 3)
  (h_restricted_ties : restricted_ties = 2) :
  num_shirts * num_ties - restricted_shirts * restricted_ties = 50 := by
  sorry

end valid_shirt_tie_combinations_l137_137612


namespace find_certain_number_l137_137448

theorem find_certain_number (x y : ℝ)
  (h1 : (28 + x + 42 + y + 104) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) :
  y = 78 :=
by
  sorry

end find_certain_number_l137_137448


namespace product_of_four_consecutive_integers_l137_137856

theorem product_of_four_consecutive_integers (n : ℤ) : ∃ k : ℤ, k^2 = (n-1) * n * (n+1) * (n+2) + 1 :=
by
  sorry

end product_of_four_consecutive_integers_l137_137856


namespace complete_the_square_example_l137_137773

theorem complete_the_square_example (x : ℝ) : 
  ∃ c d : ℝ, (x^2 - 6 * x + 5 = 0) ∧ ((x + c)^2 = d) ∧ (d = 4) :=
sorry

end complete_the_square_example_l137_137773


namespace total_elephants_in_two_parks_l137_137049

theorem total_elephants_in_two_parks (n1 n2 : ℕ) (h1 : n1 = 70) (h2 : n2 = 3 * n1) : n1 + n2 = 280 := by
  sorry

end total_elephants_in_two_parks_l137_137049


namespace largest_no_solution_l137_137387

theorem largest_no_solution (a : ℕ) (h_odd : a % 2 = 1) (h_pos : a > 0) :
  ∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → a * x + (a + 1) * y + (a + 2) * z ≠ n :=
sorry

end largest_no_solution_l137_137387


namespace train_seats_count_l137_137063

theorem train_seats_count 
  (Standard Comfort Premium : ℝ)
  (Total_SEATS : ℝ)
  (hs : Standard = 36)
  (hc : Comfort = 0.20 * Total_SEATS)
  (hp : Premium = (3/5) * Total_SEATS)
  (ht : Standard + Comfort + Premium = Total_SEATS) :
  Total_SEATS = 180 := sorry

end train_seats_count_l137_137063


namespace friction_coefficient_example_l137_137319

variable (α : ℝ) (mg : ℝ) (μ : ℝ)

theorem friction_coefficient_example
    (hα : α = 85 * Real.pi / 180) -- converting degrees to radians
    (hN : ∀ (N : ℝ), N = 6 * mg) -- Normal force in the vertical position
    (F : ℝ) -- Force applied horizontally by boy
    (hvert : F * Real.sin α - mg + (6 * mg) * Real.cos α = 0) -- vertical equilibrium
    (hhor : F * Real.cos α - μ * (6 * mg) - (6 * mg) * Real.sin α = 0) -- horizontal equilibrium
    : μ = 0.08 :=
by
  sorry

end friction_coefficient_example_l137_137319


namespace expression_in_multiply_form_l137_137083

def a : ℕ := 3 ^ 1005
def b : ℕ := 7 ^ 1006
def m : ℕ := 114337548

theorem expression_in_multiply_form : 
  (a + b)^2 - (a - b)^2 = m * 10 ^ 1006 :=
by
  sorry

end expression_in_multiply_form_l137_137083


namespace shaded_area_of_hexagon_with_quarter_circles_l137_137832

noncomputable def area_inside_hexagon_outside_circles
  (s : ℝ) (h : s = 4) : ℝ :=
  let hex_area := (3 * Real.sqrt 3) / 2 * s^2
  let quarter_circle_area := (1 / 4) * Real.pi * s^2
  let total_quarter_circles_area := 6 * quarter_circle_area
  hex_area - total_quarter_circles_area

theorem shaded_area_of_hexagon_with_quarter_circles :
  area_inside_hexagon_outside_circles 4 rfl = 48 * Real.sqrt 3 - 24 * Real.pi := by
  sorry

end shaded_area_of_hexagon_with_quarter_circles_l137_137832


namespace min_value_reciprocals_l137_137520

open Real

theorem min_value_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + b = 1) :
  (1 / a + 1 / b) = 4 :=
by
  sorry

end min_value_reciprocals_l137_137520


namespace boat_distance_against_stream_l137_137534

theorem boat_distance_against_stream 
  (v_b : ℝ)
  (v_s : ℝ)
  (distance_downstream : ℝ)
  (t : ℝ)
  (speed_downstream : v_s + v_b = 11)
  (speed_still_water : v_b = 8)
  (time : t = 1) :
  (v_b - (11 - v_b)) * t = 5 :=
by
  -- Here we're given the initial conditions and have to show the final distance against the stream is 5 km
  sorry

end boat_distance_against_stream_l137_137534


namespace cards_left_l137_137160

def number_of_initial_cards : ℕ := 67
def number_of_cards_taken : ℕ := 9

theorem cards_left (l : ℕ) (d : ℕ) (hl : l = number_of_initial_cards) (hd : d = number_of_cards_taken) : l - d = 58 :=
by
  sorry

end cards_left_l137_137160


namespace divisor_inequality_l137_137355

-- Definition of our main inequality theorem
theorem divisor_inequality (n : ℕ) (h1 : n > 0) (h2 : n % 8 = 4)
    (divisors : List ℕ) (h3 : divisors = (List.range (n + 1)).filter (λ x => n % x = 0)) 
    (i : ℕ) (h4 : i < divisors.length - 1) (h5 : i % 3 ≠ 0) : 
    divisors[i + 1] ≤ 2 * divisors[i] := sorry

end divisor_inequality_l137_137355


namespace quadratic_distinct_real_roots_l137_137167

theorem quadratic_distinct_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a = 0 → x^2 - 2*x - a = 0 ∧ (∀ y : ℝ, y ≠ x → y^2 - 2*y - a = 0)) → 
  a > -1 :=
by
  sorry

end quadratic_distinct_real_roots_l137_137167


namespace gather_half_of_nuts_l137_137956

open Nat

theorem gather_half_of_nuts (a b c : ℕ) (h₀ : (a + b + c) % 2 = 0) : ∃ k, k = (a + b + c) / 2 :=
  sorry

end gather_half_of_nuts_l137_137956


namespace sqrt_four_eq_pm_two_l137_137179

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l137_137179
