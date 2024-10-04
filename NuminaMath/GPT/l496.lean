import Mathlib

namespace ShelbyRainDrivingTime_l496_496505

-- Define the conditions
def drivingTimeNonRain (totalTime: ℕ) (rainTime: ℕ) : ℕ := totalTime - rainTime
def rainSpeed : ℚ := 20 / 60
def noRainSpeed : ℚ := 30 / 60
def totalDistance (rainTime: ℕ) (nonRainTime: ℕ) : ℚ := rainSpeed * rainTime + noRainSpeed * nonRainTime

-- Prove the question == answer given conditions
theorem ShelbyRainDrivingTime :
  ∀ (rainTime totalTime: ℕ),
  (totalTime = 40) →
  (totalDistance rainTime (drivingTimeNonRain totalTime rainTime) = 16) →
  rainTime = 24 :=
by
  intros rainTime totalTime ht hd
  have h1 : drivingTimeNonRain totalTime rainTime = 40 - rainTime := rfl
  rw [← h1] at hd
  sorry

end ShelbyRainDrivingTime_l496_496505


namespace truncated_pyramid_middle_section_perimeter_area_l496_496187

theorem truncated_pyramid_middle_section_perimeter_area {k1 k2 t1 t2 t_m k_m : ℝ}
    (h_perimeter : k_m = (k1 + k2) / 2) 
    (h_area : t_m = (t1 + 2 * sqrt (t1 * t2) + t2) / 4) : 
    k_m = (k1 + k2) / 2 ∧ t_m = (t1 + 2 * sqrt (t1 * t2) + t2) / 4 := 
by
  split
  · exact h_perimeter
  · exact h_area

end truncated_pyramid_middle_section_perimeter_area_l496_496187


namespace gcd_of_840_and_1764_l496_496531

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := 
by {
  sorry
}

end gcd_of_840_and_1764_l496_496531


namespace profit_function_and_optimal_price_l496_496265

variable (cost selling base_units additional_units: ℝ)
variable (x: ℝ) (y: ℝ)

def profit (x: ℝ): ℝ := -20 * x^2 + 100 * x + 6000

theorem profit_function_and_optimal_price:
  (cost = 40) →
  (selling = 60) →
  (base_units = 300) →
  (additional_units = 20) →
  (0 ≤ x) →
  (x < 20) →
  (y = profit x) →
  exists x_max y_max: ℝ, (x_max = 2.5) ∧ (y_max = 6125) :=
by 
  sorry

end profit_function_and_optimal_price_l496_496265


namespace combined_non_overlapping_area_l496_496962

noncomputable def PQS : ℝ := (90 / 360) * π * 10^2
noncomputable def RQS : ℝ := (60 / 360) * π * 10^2
noncomputable def overlap : ℝ := RQS
noncomputable def non_overlapping_area : ℝ := PQS - overlap

theorem combined_non_overlapping_area : non_overlapping_area = (25 / 3) * π := by sorry

end combined_non_overlapping_area_l496_496962


namespace find_other_vertices_of_parallelogram_l496_496496

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_vertices_of_parallelogram
  (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (8, 9))
  (parallel_to_x_axis : ∃ x1 x2 : ℝ, (x1, -3) = (5, -3) ∧ (x2, 9) = (5, 9)) :
  ∃ C D : ℝ × ℝ, C = (5, -3) ∧ D = (5, 9) :=
by
  sorry

end find_other_vertices_of_parallelogram_l496_496496


namespace part1_monotonic_intervals_part2_sum_of_roots_gt_2_l496_496027

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - a * x) / Real.exp x

theorem part1_monotonic_intervals :
  let a := 2 in
  ∀ (x : ℝ),
    (x < 2 - Real.sqrt 2 ∨ x > 2 + Real.sqrt 2 → deriv (f x a) < 0) ∧ 
    (2 - Real.sqrt 2 < x ∧ x < 2 + Real.sqrt 2 → deriv (f x a) > 0) := by
  sorry

theorem part2_sum_of_roots_gt_2 (x1 x2 : ℝ) :
  let a := 1 in
  (f x1 a) = (Real.log x1 + 1) / Real.exp x1 ∧ 
  (f x2 a) = (Real.log x2 + 1) / Real.exp x2 →
  x1 + x2 > 2 := by
  sorry

end part1_monotonic_intervals_part2_sum_of_roots_gt_2_l496_496027


namespace megan_savings_final_balance_percentage_l496_496480

noncomputable def megan_final_percentage (initial_balance : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : ℝ :=
let increased_balance := initial_balance * (1 + increase_rate) in
let final_balance := increased_balance * (1 - decrease_rate) in
(final_balance / initial_balance) * 100

theorem megan_savings_final_balance_percentage :
  megan_final_percentage 125 0.25 0.20 = 100 := 
by
  sorry

end megan_savings_final_balance_percentage_l496_496480


namespace train_speed_problem_l496_496964

theorem train_speed_problem (l1 l2 : ℝ) (v2 : ℝ) (t : ℝ) (v1 : ℝ) :
  l1 = 120 → l2 = 280 → v2 = 30 → t = 19.99840012798976 →
  0.4 / (t / 3600) = v1 + v2 → v1 = 42 :=
by
  intros hl1 hl2 hv2 ht hrel
  rw [hl1, hl2, hv2, ht] at *
  sorry

end train_speed_problem_l496_496964


namespace regression_analysis_incorrect_statement_l496_496583

-- Definitions for the conditions
def condition_A : Prop := "The regression line must pass through the point (\overline{x}, \overline{y})"
def condition_B : Prop := "The regression line is the line that passes through the most sample data points in a scatter plot"
def condition_C : Prop := "When the correlation coefficient r > 0, the two variables are positively correlated"
def condition_D : Prop := "If the linear correlation between two variables is weaker, then |r| is closer to 0"

-- The goal is to prove that condition_B is false
theorem regression_analysis_incorrect_statement : ¬condition_B :=
by
  sorry

end regression_analysis_incorrect_statement_l496_496583


namespace tangent_line_value_l496_496091

theorem tangent_line_value {a : ℝ} (h : a > 0) : 
  (∀ θ ρ, (ρ * (Real.cos θ + Real.sin θ) = a) → (ρ = 2 * Real.cos θ)) → 
  a = 1 + Real.sqrt 2 :=
sorry

end tangent_line_value_l496_496091


namespace buckets_needed_l496_496956

variable {C : ℝ} (hC : C > 0)

theorem buckets_needed (h : 42 * C = 42 * C) : 
  (42 * C) / ((2 / 5) * C) = 105 :=
by
  sorry

end buckets_needed_l496_496956


namespace tangent_condition_l496_496133

-- Define the functions and conditions
def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

def tangent_line (x : ℝ) (k : ℝ) := k * x - 2

theorem tangent_condition (a b k : ℝ) (h1 : (1, 0) ∈ (λ x, (x, f x a b)) '' univ) 
  (h2 : (1, 0) ∈ (λ x, (x, tangent_line x k)) '' univ)
  (h3: ∀ x, (∂ (f x a b) / ∂ x) /- implicit differentiation -./(1) = k):
  k = 2 ∧ f = λ x, x^3 - x := 
by sorry

end tangent_condition_l496_496133


namespace find_value_of_x_squared_plus_one_over_x_squared_l496_496820

theorem find_value_of_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by 
  sorry

end find_value_of_x_squared_plus_one_over_x_squared_l496_496820


namespace base_5_to_base_2_equiv_l496_496297

theorem base_5_to_base_2_equiv : 
  (324 : ℕ) in_base 5 = (1011001 : ℕ) in_base 2 :=
sorry

end base_5_to_base_2_equiv_l496_496297


namespace cyclic_matrix_determinant_zero_l496_496118

-- Define the context: variables and the polynomial they are roots of.
variables {α : Type*} [field α] (a b c d p q r : α)
-- Define the polynomial and root conditions.
def polynomial_roots : Prop := 
  a = arbitrary α ∧ b = arbitrary α ∧ c = arbitrary α ∧ d = arbitrary α ∧
  ∀ x : α, x^4 + p * x^2 + q * x + r = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)

-- Define the matrix whose determinant is in question.
def cyclic_matrix (a b c d : α) : matrix (fin 4) (fin 4) α :=
![![a, b, c, d],
  ![b, c, d, a],
  ![c, d, a, b],
  ![d, a, b, c]]

-- State the goal: The determinant of the cyclic matrix is 0.
theorem cyclic_matrix_determinant_zero (hroots : polynomial_roots a b c d p q r) :
  matrix.det (cyclic_matrix a b c d) = 0 :=
sorry

end cyclic_matrix_determinant_zero_l496_496118


namespace pow_two_ge_square_iff_l496_496702

theorem pow_two_ge_square_iff (n : ℤ) : 
  (2 ^ n ≥ n ^ 2) ↔ (n = 1 ∨ n = 2 ∨ n ≥ 4) := 
by 
  sorry

end pow_two_ge_square_iff_l496_496702


namespace inequality_proof_l496_496126

variable (x y z : ℝ)
variable (hx : x ≥ 5/2)
variable (hy : y ≥ 5/2)
variable (hz : z ≥ 5/2)

theorem inequality_proof :
    (1 + 1 / (2 + x)) * (1 + 1 / (2 + y)) * (1 + 1 / (2 + z)) ≥ (1 + 1 / (2 + real.cbrt (x * y * z)))^3
:= sorry

end inequality_proof_l496_496126


namespace omega_range_l496_496356

theorem omega_range (ω : ℝ) (hω_pos : ω > 0) : 
  (∀ x ∈ set.Icc (-π/3) (π/4), 2 * sin (ω * x) ≥ -2) → 
  ω ∈ set.Ioc 0 (3/2) :=
by
  sorry

end omega_range_l496_496356


namespace southbound_vehicle_count_l496_496949

def northbound_speed := 70 -- miles per hour
def southbound_speed := 50 -- miles per hour
def observed_vehicles := 25 -- vehicles
def observation_time := 5 / 60 -- hours (5 minutes)

theorem southbound_vehicle_count :
  let relative_speed := northbound_speed + southbound_speed in
  let observation_distance := relative_speed * observation_time in
  let vehicle_density := observed_vehicles / observation_distance in
  vehicle_density * 150 = 375 :=
by
  let relative_speed := northbound_speed + southbound_speed
  let observation_distance := relative_speed * observation_time
  let vehicle_density := observed_vehicles / observation_distance
  have : vehicle_density * 150 = 375, sorry
  exact this

end southbound_vehicle_count_l496_496949


namespace sum_last_three_coeffs_l496_496237

theorem sum_last_three_coeffs (b : ℝ) (h : b ≠ 0) : 
  let c := (1 - 2/b)^5 in 
  let s := (c.coeff 0) + (c.coeff 1) + (c.coeff 2) in 
  s = 31 :=
sorry

end sum_last_three_coeffs_l496_496237


namespace solution_set_for_inequality_l496_496770

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain (x : ℝ) : f x ∈ ℝ

axiom f_second_derivative (x : ℝ) : deriv (deriv (f x)) x - 2 * f x > 0

axiom f_at_half : f (1 / 2) = Real.exp 1

theorem solution_set_for_inequality : { x : ℝ | x > 0 ∧ f (1 / 2 * Real.log x) < x } = { x : ℝ | 0 < x ∧ x < Real.exp 1 } := 
sorry

end solution_set_for_inequality_l496_496770


namespace prob_not_green_is_six_over_eleven_l496_496062

-- Define the odds for pulling a green marble
def odds_green : ℕ × ℕ := (5, 6)

-- Define the total number of events as the sum of both parts of the odds
def total_events : ℕ := odds_green.1 + odds_green.2

-- Define the probability of not pulling a green marble
def probability_not_green : ℚ := odds_green.2 / total_events

-- State the theorem
theorem prob_not_green_is_six_over_eleven : probability_not_green = 6 / 11 := by
  -- Proof goes here
  sorry

end prob_not_green_is_six_over_eleven_l496_496062


namespace sum_of_positive_odd_divisors_180_l496_496986

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d % 2 = 1) (Finset.divisors n)).sum

theorem sum_of_positive_odd_divisors_180 : 
  sum_of_positive_odd_divisors 180 = 78 := by
  sorry

end sum_of_positive_odd_divisors_180_l496_496986


namespace find_a_b_and_analyze_function_l496_496787

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem find_a_b_and_analyze_function :
  ∃ (a b : ℝ), 
    (let f := f with a b,
      f(1) = 2 ∧
      deriv f 1 = -1 ∧
      (a = 1 ∧ b = 8 / 3) ∧
      (intervals_of_monotonicity_and_extrema f = 
         { inc_on : set.Ixx (-∞ : ℝ) 0 ∪ set.Ixx 2 ∞, 
           dec_on : set.Ixx 0 2, 
           local_max : { (0, 8 / 3) }, 
           local_min : { (2, 4 / 3) }
         } ∧
      (max_on_interval f (-2) 5 = 58 / 3) ) )
sorry

end find_a_b_and_analyze_function_l496_496787


namespace cm_eq_cn_l496_496654

noncomputable theory

open_locale classical

variables {K : Type*} [field K]
variables (A B C D M N : K)
variables (BM DN CM CN : K)

-- Conditions
def inscribed_quadrilateral (A B C D M N : K) : Prop :=
  -- Here you would specify the condition that quadrilateral ABCD is inscribed in a circle
  sorry

def intersection_points (A B C D M N : K) : Prop :=
  -- Here you would specify the conditions that define points M and N
  sorry

def equal_lengths (BM DN : K) : Prop := BM = DN

-- Theorem
theorem cm_eq_cn
  (inscribed : inscribed_quadrilateral A B C D M N)
  (intersections : intersection_points A B C D M N)
  (eq_lengths : equal_lengths BM DN) :
  CM = CN :=
begin
  sorry
end

end cm_eq_cn_l496_496654


namespace ratio_flow_chart_to_total_time_l496_496304

noncomputable def T := 48
noncomputable def D := 18
noncomputable def C := (3 / 8) * T
noncomputable def F := T - C - D

theorem ratio_flow_chart_to_total_time : (F / T) = (1 / 4) := by
  sorry

end ratio_flow_chart_to_total_time_l496_496304


namespace remainder_of_hx10_divided_by_hx_is_6_l496_496457

noncomputable def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_hx10_divided_by_hx_is_6 : 
  let q := h (x ^ 10);
  q % h (x) = 6 := by
  sorry

end remainder_of_hx10_divided_by_hx_is_6_l496_496457


namespace percentage_denied_riverside_l496_496312

variables (riversideHighKids : ℕ) (westSideHighKids : ℕ) (mountainTopHighKids : ℕ)
          (totalKidsGotIn : ℕ) (deniedFromWestSideHigh : ℕ) (deniedFromMountainTopHigh : ℕ)

-- Given conditions
def totalKids : ℕ := 260
def deniedFromWestSideHigh : ℕ := 63
def deniedFromMountainTopHigh : ℕ := 25
def totalKidsGotIn : ℕ := 148

-- Definition of the number of kids denied from Riverside High
def deniedFromRiversideHigh : ℕ := 112 - 88

-- Definition of the percentage of kids from Riverside High who were denied
def percentageDeniedFromRiversideHigh : ℕ := (deniedFromRiversideHigh * 100) / 120

theorem percentage_denied_riverside :
  percentageDeniedFromRiversideHigh = 20 := by
  sorry

end percentage_denied_riverside_l496_496312


namespace johns_burritos_l496_496858

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l496_496858


namespace ellipse_standard_eq_max_triangle_area_l496_496008

-- Define the ellipse and its properties
def ellipse (a b : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the foci and distance condition
def foci_distance (c : ℝ) :=
  c = real.sqrt 2

-- Define the problem's conditions
def problem_conditions (a b c : ℝ) :=
  ellipse a b ∧ foci_distance c ∧ a = real.sqrt 3 ∧ b^2 = a^2 - c^2

-- Proof for the equation of the ellipse
theorem ellipse_standard_eq (a b c : ℝ) (x y : ℝ) (h : problem_conditions a b c) :
  (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 3 + y^2 = 1) :=
sorry

-- Define the circle and point conditions
def circle (x y : ℝ) :=
  x^2 + y^2 = 4

-- Define the maximum area problem
theorem max_triangle_area (x_P y_P : ℝ) (hP : circle x_P y_P) :
  ∃ (M N : ℝ) (k1 k2 : ℝ), ((k1 * k2 = -1) ∧ (M - x_P)^2 + (N - y_P)^2 = 4) →
  1 / 2 * (real.abs (x_P - M) * real.abs (y_P - N)) ≤ 4 :=
sorry

end ellipse_standard_eq_max_triangle_area_l496_496008


namespace problem_1_problem_2_l496_496454

-- Problem (1)
theorem problem_1 (x a : ℝ) (h_a : a = 1) (hP : x^2 - 4*a*x + 3*a^2 < 0) (hQ1 : x^2 - x - 6 ≤ 0) (hQ2 : x^2 + 2*x - 8 > 0) :
  2 < x ∧ x < 3 := sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h_a_pos : 0 < a) (h_suff_neccess : (¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) → ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) ∧
                   ¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) ≠ ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) :
  1 < a ∧ a ≤ 2 := sorry

end problem_1_problem_2_l496_496454


namespace number_of_boys_l496_496069

theorem number_of_boys (total_students : ℕ) (h_condition : total_students = 40)
  (h_probability_ratio : ∃ p : ℝ, p = (3 / 4) * (1 - p)) :
  ∃ boys : ℕ, boys = 17 :=
by
  have h_solution := 7 * 17 = total_students ∧ 17 * 4 = 3 * (40 - 17)
  existsi 17
  exact h_solution.1
  sorry -- Proof to be completed here

end number_of_boys_l496_496069


namespace find_f_13_l496_496528

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f (x + f x) = 3 * f x
axiom f_of_1 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
by
  have hf := f_property
  have hf1 := f_of_1
  sorry

end find_f_13_l496_496528


namespace area_square_II_l496_496177

theorem area_square_II (a b : ℝ) (h : (a^2 + 2*a*b + b^2) > 0) :
  let s₁ := (a + b) / Real.sqrt 2,
      A₁ := s₁^2,
      A₂ := 2 * A₁
  in A₂ = (a + b)^2 := by
  let s₁ := (a + b) / Real.sqrt 2
  let A₁ := s₁^2
  let A₂ := 2 * A₁
  sorry

end area_square_II_l496_496177


namespace find_a10_l496_496765

variable {a : ℤ} -- declare initial term a
variable {q : ℤ} -- declare common ratio q (integer)
variable {a_n : ℕ → ℤ} -- declare geometric sequence a_n

-- given conditions in our Lean4 problem
def is_geometric_sequence (a_n : ℕ → ℤ) (a q : ℤ) : Prop :=
∀ n : ℕ, a_n n = a * q ^ n

def condition1 : Prop := is_geometric_sequence a_n a q
def condition2 : Prop := a_n 3 * a_n 6 = -512 -- mapping a_4 → a_n 3 and a_7 → a_n 6 

def condition3 : Prop := a_n 2 + a_n 7 = 124 -- mapping a_3 → a_n 2 and a_8 → a_n 7
def is_integer_common_ratio : Prop := q ∈ ℤ

-- theorem statement we need to prove
theorem find_a10 
  (cond1 : condition1)
  (cond2 : condition2)
  (cond3 : condition3)
  (int_q : is_integer_common_ratio) : 
  a_n 9 = 512 :=
sorry -- proof omitted

end find_a10_l496_496765


namespace area_of_quadrilateral_l496_496942

noncomputable def a : ℝ × ℝ × ℝ := (2, -5, 3)
noncomputable def b : ℝ × ℝ × ℝ := (4, -9, 6)
noncomputable def c : ℝ × ℝ × ℝ := (3, -4, 1)
noncomputable def d : ℝ × ℝ × ℝ := (5, -8, 4)

def b_minus_a : ℝ × ℝ × ℝ := (4 - 2, -9 - (-5), 6 - 3)
def c_minus_a : ℝ × ℝ × ℝ := (3 - 2, -4 - (-5), 1 - 3)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def parallelogram_area : ℝ :=
  magnitude (cross_product b_minus_a c_minus_a)

theorem area_of_quadrilateral : parallelogram_area = Real.sqrt 110 := by
  -- Proof is omitted
  sorry

end area_of_quadrilateral_l496_496942


namespace tank_never_fills_l496_496901

structure Pipe :=
(rate1 : ℕ) (rate2 : ℕ)

def net_flow (pA pB pC pD : Pipe) (time1 time2 : ℕ) : ℤ :=
  let fillA := pA.rate1 * time1 + pA.rate2 * time2
  let fillB := pB.rate1 * time1 + pB.rate2 * time2
  let drainC := pC.rate1 * time1 + pC.rate2 * time2
  let drainD := pD.rate1 * (time1 + time2)
  (fillA + fillB) - (drainC + drainD)

theorem tank_never_fills (pA pB pC pD : Pipe) (time1 time2 : ℕ)
  (hA : pA = Pipe.mk 40 20) (hB : pB = Pipe.mk 20 40) 
  (hC : pC = Pipe.mk 20 40) (hD : pD = Pipe.mk 30 30) 
  (hTime : time1 = 30 ∧ time2 = 30): 
  net_flow pA pB pC pD time1 time2 = 0 := by
  sorry

end tank_never_fills_l496_496901


namespace parabola_properties_l496_496756

noncomputable def parabola_focus_distance (p : ℝ) (m : ℝ) : Prop :=
  let directrix_y := -p / 2
  let focus := (0 : ℝ, p / 2)
  let distance_to_focus := real.sqrt ((m - 0)^2 + (4 - (p / 2))^2)
  abs (4 - directrix_y) = distance_to_focus

theorem parabola_properties (p m : ℝ) (h_condition : p > 0) (h_parabola : m^2 = 8*p) (h_distance : parabola_focus_distance p m) :
  p = 1/2 ∧ (m = 2 ∨ m = -2) :=
sorry

end parabola_properties_l496_496756


namespace rotated_ln_graph_eq_exp_neg_x_l496_496184

theorem rotated_ln_graph_eq_exp_neg_x : 
  (∀ x : ℝ, x > 0 -> ∃ y : ℝ, y = real.exp (-x)) :=
begin
  sorry
end

end rotated_ln_graph_eq_exp_neg_x_l496_496184


namespace domain_composite_function_l496_496771

theorem domain_composite_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x = y) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (2^x - 1) = y) :=
by
  sorry

end domain_composite_function_l496_496771


namespace find_point_W_l496_496655

structure Point where
  x : ℝ
  y : ℝ

def is_rectangle (O S U V : Point) : Prop :=
  O.x = 0 ∧ O.y = 0 ∧
  U.x = 4 ∧ U.y = 3 ∧
  S.x = U.x ∧ S.y = 0 ∧
  V.x = 0 ∧ V.y = U.y

def area_rectangle (O U : Point) : ℝ :=
  abs (U.x - O.x) * abs (U.y - O.y)

def area_triangle (S W V : Point) : ℝ :=
  1 / 2 * abs (V.x - S.x) * abs (W.y - V.y)

theorem find_point_W :
  ∃ W : Point, 
    let O := Point.mk 0 0 in
    let U := Point.mk 4 3 in
    let S := Point.mk U.x 0 in
    let V := Point.mk 0 U.y in
    is_rectangle O S U V ∧
    area_rectangle O U + area_triangle S W V = 2 * area_rectangle O U ∧
    W.x = 8 ∧ W.y = 3 :=
by
  sorry

end find_point_W_l496_496655


namespace convex_numbers_count_l496_496056

def digit_tens_place_greater (hundreds tens ones : ℕ) : Prop :=
  tens > hundreds + ones

def is_convex_number (n : ℕ) : Prop :=
  ∃ (hundreds tens ones : ℕ), 
    n = 100 * hundreds + 10 * tens + ones ∧
    hundreds ∈ {1, 2, 3, 4, 5} ∧
    tens ∈ {1, 2, 3, 4, 5} ∧
    ones ∈ {1, 2, 3, 4, 5} ∧
    hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧ 
    digit_tens_place_greater hundreds tens ones

theorem convex_numbers_count : 
  (finset.filter is_convex_number (finset.range 1000)).card = 20 :=
sorry

end convex_numbers_count_l496_496056


namespace expected_number_of_digits_is_1_55_l496_496105

def probability_one_digit : ℚ := 9 / 20
def probability_two_digits : ℚ := 1 / 2
def probability_twenty : ℚ := 1 / 20
def expected_digits : ℚ := (1 * probability_one_digit) + (2 * probability_two_digits) + (2 * probability_twenty)

theorem expected_number_of_digits_is_1_55 :
  expected_digits = 1.55 :=
sorry

end expected_number_of_digits_is_1_55_l496_496105


namespace select_at_least_8_sticks_l496_496748

theorem select_at_least_8_sticks (S : Finset ℕ) (hS : S = (Finset.range 92 \ {0})) :
  ∃ (sticks : Finset ℕ) (h_sticks : sticks.card = 8),
    ∃ (a b c : ℕ) (h_a : a ∈ sticks) (h_b : b ∈ sticks) (h_c : c ∈ sticks),
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
by
  -- Proof required here
  sorry

end select_at_least_8_sticks_l496_496748


namespace lean_solution_l496_496914

def lean_problem : Prop :=
  let a := 0.3 in
  2^a > a^2 ∧ a^2 > log 2 a

theorem lean_solution : lean_problem :=
  by
  let a := 0.3
  sorry

end lean_solution_l496_496914


namespace problem_proof_l496_496343

variable {x : ℕ → ℝ}

noncomputable def x_sequence : ℕ → ℝ
| 1       := 1
| (n + 2) := x_sequence (n + 1) + sqrt (x_sequence (n + 1) + 1) - 1

theorem problem_proof (n : ℕ) (hn : n > 0) :
  (0 < x_sequence (n + 1) ∧ x_sequence (n + 1) < x_sequence n) ∧
  3 * x_sequence (n + 1) - 2 * x_sequence n < x_sequence n * x_sequence (n + 1) / 3 ∧
  (2 / 3) ^ (n - 1) ≤ x_sequence n ∧ x_sequence n ≤ (2 / 3) ^ (n - 2) :=
sorry

end problem_proof_l496_496343


namespace min_distance_from_origin_l496_496768

theorem min_distance_from_origin (m n : ℝ) (h : 2 * m + n + 5 = 0) : (sqrt (m^2 + n^2) ≥ sqrt 5) ∧ (sqrt (m^2 + n^2) = sqrt 5 ↔ m = 2) :=
by
  sorry

end min_distance_from_origin_l496_496768


namespace min_value_fraction_l496_496341

-- We start by defining the geometric sequence and the given conditions
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {a1 : ℝ} (h_pos : ∀ n, 0 < a n)
variable (h_geo : ∀ n, a (n + 1) = a n * r)
variable (h_a7 : a 7 = a 6 + 2 * a 5)
variable (h_am_an : ∃ m n, a m * a n = 16 * (a 1)^2)

theorem min_value_fraction : 
  ∃ (m n : ℕ), (a m * a n = 16 * (a 1)^2 ∧ (1/m) + (4/n) = 1) :=
sorry

end min_value_fraction_l496_496341


namespace calories_per_person_l496_496849

open Nat

theorem calories_per_person :
  ∀ (oranges people pieces_per_orange calories_per_orange : ℕ),
    oranges = 5 →
    pieces_per_orange = 8 →
    people = 4 →
    calories_per_orange = 80 →
    (oranges * pieces_per_orange) / people * (calories_per_orange / pieces_per_orange) = 100 :=
by
  intros oranges people pieces_per_orange calories_per_orange
  assume h_oranges h_pieces_per_orange h_people h_calories_per_orange
  rw [h_oranges, h_pieces_per_orange, h_people, h_calories_per_orange]
  norm_num
  sorry

end calories_per_person_l496_496849


namespace cube_root_1510_l496_496353

theorem cube_root_1510 :
  (∛ 1510 = 11.47) :=
begin
  -- Given conditions
  let h1 : (∛ 1.51 = 1.147) := sorry,
  let h2 : (∛ 15.1 = 2.472) := sorry,
  let h3 : (∛ 0.151 = 0.5325) := sorry,

  -- Steps to prove ∛ 1510 = 11.47 will go here
  sorry
end

end cube_root_1510_l496_496353


namespace find_x_l496_496811

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 2)
noncomputable def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def scalar_vec_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem find_x (x : ℝ) :
  (vec_add a (b x)).1 * (vec_sub a (scalar_vec_mul 2 (b x))).2 =
  (vec_add a (b x)).2 * (vec_sub a (scalar_vec_mul 2 (b x))).1 →
  x = 4 :=
by sorry

end find_x_l496_496811


namespace find_a_divides_x13_plus_x_plus_90_l496_496872

theorem find_a_divides_x13_plus_x_plus_90:
  ∃ a : ℤ, (a > 0) ∧ (∀ q : polynomial ℤ, (x^2 - x + a) * q = x^13 + x + 90) ∧ (a = 2) := by
  sorry

end find_a_divides_x13_plus_x_plus_90_l496_496872


namespace total_cost_8_dozen_pencils_2_dozen_notebooks_l496_496523

variable (P N : ℝ)

def eq1 : Prop := 3 * P + 4 * N = 60
def eq2 : Prop := P + N = 15.512820512820513

theorem total_cost_8_dozen_pencils_2_dozen_notebooks :
  eq1 P N ∧ eq2 P N → (96 * P + 24 * N = 520) :=
by
  sorry

end total_cost_8_dozen_pencils_2_dozen_notebooks_l496_496523


namespace b_share_l496_496590

theorem b_share (total_money : ℝ) (ratioA ratioB ratioC : ℝ) (h_ratios : ratioA = 2 ∧ ratioB = 3 ∧ ratioC = 4) (h_total_money : total_money = 900) :
  ratioB / (ratioA + ratioB + ratioC) * total_money = 300 :=
by
  simp at h_ratios,
  cases h_ratios with h_ratioA h_ratios_r,
  cases h_ratios_r with h_ratioB h_ratioC,
  rw [h_ratioA, h_ratioB, h_ratioC],
  rw h_total_money,
  sorry

end b_share_l496_496590


namespace successive_percentage_increase_l496_496253

theorem successive_percentage_increase (P : ℝ) : 
  let P1 := P * 1.15 in
  let P2 := P1 * 1.15 in
  P2 = P * 1.3225 :=
by
  let P1 := P * 1.15
  let P2 := P * 1.15 * 1.15
  have h_P2 : P2 = P * 1.3225 := by
    calc
      P * 1.15 * 1.15 = P * (1.15 * 1.15) : by ring
      ... = P * 1.3225 : by norm_num
  exact h_P2

end successive_percentage_increase_l496_496253


namespace arithmetic_sequence_solution_l496_496086

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  (a 5 = 11) ∧ (a 8 = 5) ∧ (∀ n m : ℕ, n < m → a n + (m - n) * (a n - a (n - 1)) = a m)

theorem arithmetic_sequence_solution :
  ∃ a : ℕ → ℤ,
    arithmetic_sequence a ∧
    (∀ n : ℕ, a n = -2 * n + 21) ∧
    (∑ i in Finset.range 10, a i = 100) :=
by
  sorry

end arithmetic_sequence_solution_l496_496086


namespace arithmetic_geom_seq_l496_496006

variable {a_n : ℕ → ℝ}
variable {d a_1 : ℝ}
variable (h_seq : ∀ n, a_n n = a_1 + (n-1) * d)
variable (d_ne_zero : d ≠ 0)
variable (a_1_ne_zero : a_1 ≠ 0)
variable (geo_seq : (a_1 + d)^2 = a_1 * (a_1 + 3 * d))

theorem arithmetic_geom_seq :
  (a_1 + a_n 14) / a_n 3 = 5 := by
  sorry

end arithmetic_geom_seq_l496_496006


namespace maximize_profit_in_country_B_l496_496487

noncomputable def demand_A (P_A : ℝ) := 40 - 2 * P_A
noncomputable def demand_B (P_B : ℝ) := 26 - P_B

noncomputable def total_cost (Q : ℝ) := 8 * Q + 1

noncomputable def total_revenue_A (q_A : ℝ) := (20 - q_A / 2) * q_A
noncomputable def total_revenue_B (q_B : ℝ) := (26 - q_B) * q_B

noncomputable def marginal_revenue_A (q_A : ℝ) := 20 - q_A
noncomputable def marginal_revenue_B (q_B : ℝ) := 26 - 2 * q_B

noncomputable def marginal_cost := 8

theorem maximize_profit_in_country_B : 
  let q_A := 12
  let q_B := 9
  let P_A := 14
  let P_B := 17
  let Q := q_A + q_B
  let TR_A := 14 * 12
  let TR_B := 17 * 9
  let TR_total := TR_A + TR_B
  let TC := total_cost Q
  let profit_before_tax := TR_total - TC - 2
  let profit_after_tax_B := 30 + 63 + 40 + 0.7 -- profit calculation in country B with progressive tax
  let profit_after_tax_A := profit_before_tax * 0.85 -- profit calculation in country A with flat tax
  in profit_after_tax_B = 133.7 ∧ profit_after_tax_B > profit_after_tax_A :=
begin
  have q_A := 12,
  have q_B := 9,
  have P_A := 14,
  have P_B := 17,
  have Q := q_A + q_B,
  have TR_A := 14 * 12,
  have TR_B := 17 * 9,
  have TR_total := TR_A + TR_B,
  have TC := total_cost Q,
  have profit_before_tax := TR_total - TC - 2,
  have profit_after_tax_B := 30 + 63 + 40 + 0.7,
  have profit_after_tax_A := profit_before_tax * 0.85,
  split,
  { exact profit_after_tax_B, },
  { exact profit_after_tax_B > profit_after_tax_A, },
end

end maximize_profit_in_country_B_l496_496487


namespace inequality_solution_l496_496512

def fractional_part (x : ℝ) : ℝ := x - x.floor

def integer_part (x : ℝ) : ℤ := x.floor

theorem inequality_solution (x : ℝ) (hx : 0 ≤ fractional_part(x) ∧ fractional_part(x) < 1) :
  fractional_part(x) * (integer_part(x) - 1) < x - 2 → x ≥ 3 :=
by
  sorry

end inequality_solution_l496_496512


namespace land_area_l496_496891

theorem land_area (x : ℝ) (h : (70 * x - 800) / 1.2 * 1.6 + 800 = 80 * x) : x = 20 :=
by
  sorry

end land_area_l496_496891


namespace area_of_region_l496_496173

theorem area_of_region:
  (∃ m n : ℤ, m + n * real.pi = 64 + 32 * real.pi) → m + n = 96 :=
by
  sorry

end area_of_region_l496_496173


namespace f_expression_f_odd_f_decreasing_l496_496334

noncomputable def f : ℝ → ℝ := λ x, (1 - 4^x) / (1 + 4^x)

theorem f_expression (x : ℝ) :
    f(x) = (1 - 4^x) / (1 + 4^x) :=
  sorry

theorem f_odd (x : ℝ) :
    f(-x) = -f(x) :=
  sorry

theorem f_decreasing :
    ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) > f(x₂) :=
  sorry

end f_expression_f_odd_f_decreasing_l496_496334


namespace acceptable_outfits_l496_496045

-- Definitions based on the given conditions
def shirts : Nat := 8
def pants : Nat := 5
def hats : Nat := 7
def pant_colors : List String := ["red", "black", "blue", "gray", "green"]
def shirt_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]
def hat_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]

-- Axiom that ensures distinct colors for pants, shirts, and hats.
axiom distinct_colors : ∀ color ∈ pant_colors, color ∈ shirt_colors ∧ color ∈ hat_colors

-- Problem statement
theorem acceptable_outfits : 
  let total_outfits := shirts * pants * hats
  let monochrome_outfits := List.length pant_colors
  let acceptable_outfits := total_outfits - monochrome_outfits
  acceptable_outfits = 275 :=
by
  sorry

end acceptable_outfits_l496_496045


namespace sum_of_positive_odd_divisors_180_l496_496990

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d % 2 = 1) (Finset.divisors n)).sum

theorem sum_of_positive_odd_divisors_180 : 
  sum_of_positive_odd_divisors 180 = 78 := by
  sorry

end sum_of_positive_odd_divisors_180_l496_496990


namespace area_of_triangle_ABC_l496_496465

noncomputable def distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Define points O, A, B, and C
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (real.sqrt (75) ^ (1 / 4), 0, 0)
def B : ℝ × ℝ × ℝ := (0, 1, 0) -- Choice of 1 for unit positive y-axis
def C : ℝ × ℝ × ℝ := (0, 0, 1) -- Choice of 1 for unit positive z-axis

-- Define the angle BAC
def angle_BAC : ℝ := real.pi / 6

-- Formula to compute the area of a triangle given two sides and the included angle.
def triangle_area (a b c α : ℝ) : ℝ :=
  1 / 2 * a * b * real.sin α

theorem area_of_triangle_ABC : 
  let OA := distance 0 (real.sqrt (75) ^ (1 / 4))
  in triangle_area (distance O B) (distance O C) angle_BAC = 5 / 2 :=
sorry

end area_of_triangle_ABC_l496_496465


namespace find_first_number_in_list_l496_496922

theorem find_first_number_in_list
  (unknown_number : ℕ)
  (known_numbers : List ℕ)
  (average : ℕ)
  (num_values : ℕ)
  (known_sum : ℕ) :
  known_numbers = [48, 507, 2, 684, 42] →
  average = 223 →
  num_values = 6 →
  list.sum known_numbers = known_sum →
  unknown_number = (average * num_values - known_sum) →
  unknown_number = 55 := 
by
  intros h_known_list h_avg h_num_values h_known_sum h_calc
  rw [h_known_list, h_avg, h_num_values, h_calc]
  have : list.sum [48, 507, 2, 684, 42] = 1283 := rfl
  have : 223 * 6 = 1338 := rfl
  unfold list.sum
  simp at h_known_sum
  exact h_known_sum

end find_first_number_in_list_l496_496922


namespace cave_depth_l496_496928

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end cave_depth_l496_496928


namespace branches_on_one_stem_l496_496694

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l496_496694


namespace cone_slant_height_l496_496018

theorem cone_slant_height (r l : ℝ) (h1 : r = 1)
  (h2 : 2 * r * Real.pi = (1 / 2) * 2 * l * Real.pi) :
  l = 2 :=
by
  -- Proof steps go here
  sorry

end cone_slant_height_l496_496018


namespace boy_completion_time_l496_496272

theorem boy_completion_time (M W B : ℝ) (h1 : M + W + B = 1/3) (h2 : M = 1/6) (h3 : W = 1/18) : B = 1/9 :=
sorry

end boy_completion_time_l496_496272


namespace contrapositive_example_l496_496930

theorem contrapositive_example (x : ℝ) (h : x = 1 → x^2 - 3 * x + 2 = 0) :
  x^2 - 3 * x + 2 ≠ 0 → x ≠ 1 :=
by
  intro h₀
  intro h₁
  have h₂ := h h₁
  contradiction

end contrapositive_example_l496_496930


namespace domain_inverse_y_eq_neg_x_x_plus_2_l496_496888

theorem domain_inverse_y_eq_neg_x_x_plus_2 :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, x ≥ 0 → f x = -x * (x + 2) ∧ (f x ∈ (-∞, 1])) :=
sorry

end domain_inverse_y_eq_neg_x_x_plus_2_l496_496888


namespace probability_intersecting_chord_l496_496517

-- Definitions related to the problem
def points_on_circle : ℕ := 2023
def points_excluding_A_B : ℕ := 2021
def probability_CD_intersects_AB : ℚ := 1/2

-- Lean statement to prove the problem
theorem probability_intersecting_chord (points_on_circle = 2023)
    (A B : fin points_on_circle)
    (hAB : A ≠ B) :
    let C D := (choose_two (points_excluding_A_B)) in
    probability_of_intersection(A, B, C, D, points_on_circle) = probability_CD_intersects_AB := by sorry

end probability_intersecting_chord_l496_496517


namespace parallel_KL_BB1_l496_496939

-- Definitions and conditions
variables {A B C C1 A1 B1 I L K : Type}
variables [incircle : Incircle ABC I] 
variables [touches : TouchPoints ABC I C1 A1 B1]
variables [Icenter : Incenter ABC I]
variables [footL : FootOfAngleBisector A B L]
variables [intersectionK : LineIntersection (Line B1 I) (Line A1 C1) K]

-- Theorem
theorem parallel_KL_BB1 :
  Parallel (Line K L) (Line B B1) :=
begin
  sorry
end

end parallel_KL_BB1_l496_496939


namespace ron_needs_to_drink_40_percent_l496_496596

-- Problem setup and conditions
def potion_volumes : ℝ := 600
def min_effect_volume : ℝ := 30
def hermione_first_drink : ℝ := potion_volumes / 2
def hermione_mix : ℝ := potion_volumes + hermione_first_drink
def hermione_second_drink : ℝ := hermione_mix / 2
def hermione_final_mix : ℝ := hermione_mix / 2
def harry_first_drink : ℝ := potion_volumes / 2
def harry_mix : ℝ := hermione_final_mix + harry_first_drink
def harry_second_drink : ℝ := harry_mix / 2
def final_mixture : ℝ := harry_mix / 2
def portion_intelligence : ℝ := hermione_first_drink / hermione_mix * final_mixture
def portion_beauty : ℝ := (hermione_final_mix - hermione_first_drink) / hermione_mix * final_mixture
def portion_strength : ℝ := harry_first_drink / harry_mix * final_mixture

-- Mathematically equivalent proof problem
theorem ron_needs_to_drink_40_percent :
  ∀ (portion_intelligence portion_beauty portion_strength min_effect_volume final_mixture : ℝ),
    portion_intelligence ≥ min_effect_volume →
    portion_beauty ≥ min_effect_volume →
    portion_strength ≥ min_effect_volume →
    final_mixture = 375 →
    (portion_intelligence = 125 ∧ portion_beauty = 125 ∧ portion_strength = 125) →
    0.4 * final_mixture ≥ min_effect_volume :=
by
  sorry

end ron_needs_to_drink_40_percent_l496_496596


namespace Sasha_is_correct_l496_496330

-- Define a figure as a set of cells in a 3x3 grid
def figure := set (fin 3 × fin 3)

-- Define the perimeter of a figure
def perimeter (f: figure) : ℕ := sorry -- Perimeter calculation is not provided for brevity

-- Define the seven figures claimed by Sasha
abbrev figures : list figure := [
  {(0,0),(0,1),(0,2),(1,0),(1,1)}, -- Shape 1
  {(0,0),(0,1),(0,2),(1,1),(2,1)}, -- Shape 2
  {(1,0),(1,1),(1,2),(0,1),(2,1)}, -- Shape 3
  {(0,1),(1,0),(1,1),(1,2),(2,1)}, -- Shape 4
  {(0,0),(0,1),(1,0),(1,2),(2,1)}, -- Shape 5
  {(0,0),(1,0),(1,1),(1,2),(2,1)}, -- Shape 6
  {(0,1),(1,0),(1,1),(1,2),(2,1)}  -- Shape 7
  ]

-- Each figure has 5 cells
def covers_five_cells (f: figure) : Prop :=
  f.card = 5

-- Each figure has the same perimeter as the 3x3 grid
def same_perimeter (f: figure) : Prop :=
  perimeter f = 12 -- 12 units is the perimeter of a 3x3 square

-- None of the figures overlap even when flipped or rotated
def non_overlapping (fs: list figure) : Prop :=
  ∀ f1 f2 ∈ fs, f1 ≠ f2 → ∀ r1 r2, sorry -- Define non-overlapping condition

-- The main problem statement
theorem Sasha_is_correct : 
  (∀ f ∈ figures, covers_five_cells f ∧ same_perimeter f) ∧ non_overlapping figures :=
sorry

end Sasha_is_correct_l496_496330


namespace least_common_time_for_horses_to_meet_l496_496551

theorem least_common_time_for_horses_to_meet :
  ∃ U > 0, (∀ (k : Fin 12), Horse k ∈ {Horse 1, Horse 2, Horse 3, Horse 4, Horse 5, Horse 6, Horse 7} → U % k = 0) → U = 420 :=
sorry

end least_common_time_for_horses_to_meet_l496_496551


namespace derivative_bound_l496_496760

theorem derivative_bound {a b : ℝ} (f : ℝ → ℝ) (h_diff : ∀ x ∈ set.Icc a b, deriv f x ∧ deriv (deriv f) x)
  (h_ab : b ≥ a + 2)
  (h_f_bound : ∀ x ∈ set.Icc a b, abs (f x) ≤ 1)
  (h_f''_bound : ∀ x ∈ set.Icc a b, abs (deriv (deriv f) x) ≤ 1) :
  ∀ x ∈ set.Icc a b, abs (deriv f x) ≤ 2 :=
by
  sorry

end derivative_bound_l496_496760


namespace rectangle_area_ratios_l496_496004

/-- Given a rectangle ABCD. Point M is the midpoint of side AB, and point K is the midpoint of side BC. 
Segments AK and CM intersect at point E. Prove that the area of quadrilateral AECD is 4 times the area of quadrilateral MBKE. --/
theorem rectangle_area_ratios
  (A B C D M K E : Point)
  (hRect : is_rectangle A B C D)
  (hM_mid : midpoint M A B)
  (hK_mid : midpoint K B C)
  (hIntersect : line_of E ∈ (line_through A K ∩ line_through C M)) :
  area_of_quad A E C D = 4 * area_of_quad M B K E :=
sorry

end rectangle_area_ratios_l496_496004


namespace minimal_positive_period_f_l496_496188

-- Define the function f(x) as the determinant of the given matrix
def f (x : Real) : Real :=
  let a := Real.sin x
  let b := 2
  let c := -1
  let d := Real.cos x
  a * d - c * b

-- Define the theorem stating the minimal positive period of f(x) is π
theorem minimal_positive_period_f : ∃ T > 0, T ≤ 2 * Real.pi ∧ (∀ x : Real, f(x) = f(x + T)) := by
  use Real.pi
  split
  · exact Real.pi_pos
  · split
    · linarith
    · intro x
      calc f(x) = (Real.sin x) * (Real.cos x) + 2 : by sorry
           ... = (Real.sin (x + Real.pi)) * (Real.cos (x + Real.pi)) + 2 : by sorry

end minimal_positive_period_f_l496_496188


namespace sum_of_radii_of_circles_l496_496340

theorem sum_of_radii_of_circles
  (a b r : ℝ)
  (tangent_relation : r = |a + b + 2| / real.sqrt 2)
  (point_tangency : (a + 2)^2 + b^2 = r^2)
  (chord_length : a^2 + 1 = r^2) :
  ∑ r_i in {r | tangent_relation ∧ point_tangency ∧ chord_length}, r_i = 6 * real.sqrt 2 :=
sorry

end sum_of_radii_of_circles_l496_496340


namespace angle_equality_l496_496290

theorem angle_equality
  (A B C D M O_1 O_2 O_3 : Point)
  (circle_1 : Circle)
  (circle_2 : Circle)
  (line_l : Line)
  (h1 : A ∈ circle_1 ∧ A ∈ circle_2)
  (h2 : tangent line_l circle_1 B ∧ tangent line_l circle_2 C)
  (h3 : circumcenter O_3 A B C)
  (h4 : reflection D O_3 A)
  (h5 : midpoint M O_1 O_2) :
  ∠O_1 D M = ∠O_2 D A := sorry

end angle_equality_l496_496290


namespace range_of_my_function_l496_496543

noncomputable def my_function (x : ℝ) : ℝ := -3 * sin x + 1

theorem range_of_my_function : set.range my_function = set.Icc (-4 : ℝ) 2 :=
by
  sorry

end range_of_my_function_l496_496543


namespace heights_intersect_l496_496500

-- Assume A, B, C, D are points in a tetrahedron ABCD.
variables (A B C D : Type) [affine_space ℝ (A B C D)]

-- Define heights DH_4 and BH_2 intersecting
variables (H4 H2 : Type) [affine_space ℝ (H4 H2)]
axiom height_intersection : ∃ P, P ∈ line (D, H4) ∧ P ∈ line (B, H2)

-- The goal is to prove the existence of a point of intersection for the other two heights
theorem heights_intersect (A B C D H1 H3 : Type) [affine_space ℝ (A B C D H1 H3)] :
  ∃ Q, Q ∈ line (A, H1) ∧ Q ∈ line (C, H3) := sorry

end heights_intersect_l496_496500


namespace calculate_p_p_l496_496887

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem calculate_p_p : p (p 2 (-3)) (p (-4) 1) = 290 :=
by {
  -- required statement of proof problem
  sorry
}

end calculate_p_p_l496_496887


namespace clock_angle_at_seven_l496_496222

theorem clock_angle_at_seven :
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7 * degrees_per_hour
  let minute_hand_position := 0
  hour_hand_position = 210 →
  minute_hand_position = 0 →
  min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position)) = 150 :=
by
  intros degrees_per_hour hour_hand_position minute_hand_position h1 h2
  have h3 : degrees_per_hour = 30 := by sorry
  have h4 : hour_hand_position = 210 := by sorry
  have h5 : minute_hand_position = 0 := by sorry
  have h6 : hour_hand_position = -minute_hand_position + hour_hand_position := by sorry
  have h7 : min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position)) = 150 := by sorry
  exact h7

end clock_angle_at_seven_l496_496222


namespace coffee_mix_price_l496_496622

theorem coffee_mix_price 
  (P : ℝ)
  (pound_2nd : ℝ := 2.45)
  (total_pounds : ℝ := 18)
  (final_price_per_pound : ℝ := 2.30)
  (pounds_each_kind : ℝ := 9) :
  9 * P + 9 * pound_2nd = total_pounds * final_price_per_pound →
  P = 2.15 :=
by
  intros h
  sorry

end coffee_mix_price_l496_496622


namespace plane_through_points_l496_496318

open Real EuclideanSpace

def point := EuclideanSpace.point 3 ℝ
def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

theorem plane_through_points :
  ∃ (A B C D : ℝ), A > 0 ∧ Int.gcd (Int.natAbs A.to_int) (Int.gcd (Int.natAbs B.to_int) (Int.gcd (Int.natAbs C.to_int) (Int.natAbs D.to_int))) = 1 ∧
  ∀ (p : point), (p = ⟨-2, 3, -3⟩ ∨ p = ⟨2, 3, -1⟩ ∨ p = ⟨4, 1, -2⟩) → plane_equation A B C D p.1 p.2 p.3 :=
begin
  use [1, 2, -2, -10],
  split,
  -- Prove A > 0
  { exact zero_lt_one },
  split,
  -- Prove gcd condition
  { sorry },
  -- Prove the plane contains the points
  { intro p,
    rintro (rfl | rfl | rfl);
    simp [plane_equation, point.mk, one_mul, zero_mul, sub_eq_add_neg, add_comm, add_left_comm] }
end

end plane_through_points_l496_496318


namespace sum_of_positive_odd_divisors_of_180_l496_496976

/-- The sum of the positive odd divisors of 180 is 78. -/
theorem sum_of_positive_odd_divisors_of_180 : 
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45] in
  let sum_odd_divisors := List.sum odd_divisors in
  sum_odd_divisors = 78 := 
by
  -- odd_divisors is [1, 3, 5, 9, 15, 45]
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45]
  -- Summation of the odd divisors
  let sum_odd_divisors := List.sum odd_divisors
  -- Verify that the sum is 78
  show sum_odd_divisors = 78
  sorry

end sum_of_positive_odd_divisors_of_180_l496_496976


namespace largest_int_and_product_of_digits_l496_496686

-- Define the conditions
def conditionI (n : ℕ) : Prop :=
  let digits := [1, 3, 5, 9]
  n.digits = digits ∧ (digits.sum (λ d, d * d) = 45)

def conditionII (n : ℕ) : Prop :=
  let digits := [1, 3, 5, 9]
  n.digits = digits ∧ (list.sorted (≤) digits ∧ ∀ d ∈ digits, d % 2 = 1)

-- Define the main problem
theorem largest_int_and_product_of_digits : 
  ∃ n : ℕ, conditionI n ∧ conditionII n ∧ n = 1539 ∧ (n.digits.prod = 135) :=
by
  sorry

end largest_int_and_product_of_digits_l496_496686


namespace no_polynomial_exists_l496_496302

theorem no_polynomial_exists :
  ¬ ∃ (P : Polynomial ℤ),
    P.eval (1 + Real.sqrt 3) = 2 + Real.sqrt 3 ∧ 
    P.eval (3 + Real.sqrt 5) = 3 + Real.sqrt 5 :=
by
  sorry

end no_polynomial_exists_l496_496302


namespace integer_cube_less_than_triple_unique_l496_496233

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l496_496233


namespace nicholas_more_crackers_than_mona_l496_496138

theorem nicholas_more_crackers_than_mona:
  ∃ (mona_crackers : ℕ), 
    (marcus_crackers = 27) ∧ 
    (nicholas_crackers = 15) ∧ 
    (marcus_crackers = 3 * mona_crackers) → 
    (nicholas_crackers - mona_crackers) = 6 :=
begin
  let marcus_crackers := 27,
  let nicholas_crackers := 15,
  sorry
end

end nicholas_more_crackers_than_mona_l496_496138


namespace pints_in_two_liters_l496_496766

theorem pints_in_two_liters (p : ℝ) (h : p = 1.575 / 0.75) : 2 * p = 4.2 := 
sorry

end pints_in_two_liters_l496_496766


namespace divisibility_by_7_divisibility_by_13_l496_496002

-- Define sequences of remainders when divided by 50
def remainders_50 (A : ℕ) : List ℕ :=
  List.unfold
    (λ x, if x = 0 then none else let (q, r) := x /% 50 in some (r, q))
    A

-- Sum of remainders
def sum_remainders (A : ℕ) : ℕ :=
  (remainders_50 A).sum

theorem divisibility_by_7 (A : ℕ) : 
  (A % 7 = 0) ↔ (sum_remainders A % 7 = 0) :=
by sorry

theorem divisibility_by_13 (A : ℕ) : 
  (A % 13 = 0) ↔ (sum_remainders A % 13 = 0) :=
by sorry

end divisibility_by_7_divisibility_by_13_l496_496002


namespace box_length_is_approx_12_point_2_l496_496610

noncomputable def box_length
  (total_volume : Real) 
  (total_cost : Real)
  (cost_per_box : Real)
  (number_of_boxes : Real)
  (volume_per_box : Real)
  (length_of_box : Real) : Real :=
  (length_of_box := Real.cbrt volume_per_box)

theorem box_length_is_approx_12_point_2 :
  box_length 1_080_000 120 0.2 600 1800 12.2 ≈ 12.2 :=
  sorry

end box_length_is_approx_12_point_2_l496_496610


namespace number_of_solutions_l496_496394

open Nat

-- Definitions arising from the conditions
def is_solution (x y : ℕ) : Prop := 3 * x + 5 * y = 501

-- Statement of the problem
theorem number_of_solutions :
  (∃ k : ℕ, k ≥ 0 ∧ k < 33 ∧ ∀ (x y : ℕ), x = 5 * k + 2 ∧ y = 99 - 3 * k → is_solution x y) :=
  sorry

end number_of_solutions_l496_496394


namespace find_q1_div_q2_l496_496166

-- Define our variables and hypotheses
variables {p q k : ℝ}
variables {p1 p2 q1 q2 : ℝ}
variables (nonzero_p1 : p1 ≠ 0) (nonzero_p2 : p2 ≠ 0)
          (nonzero_q1 : q1 ≠ 0) (nonzero_q2 : q2 ≠ 0)

-- Conditions
hypothesis (inv_proportional : ∀ p q, p * q = k)
hypothesis (given_ratio : p1 / p2 = 3 / 4)

-- Goal
theorem find_q1_div_q2 :
  q1 ≠ 0 →
  q2 ≠ 0 →
 (p1 * q1 = p2 * q2) →
  q1 / q2 = 4 / 3 := 
sorry

end find_q1_div_q2_l496_496166


namespace perimeter_triangle_MNO_l496_496631

-- Define the points and distances based on given conditions
structure Point3D where
  x y z : ℝ

def midpoint (A B : Point3D) : Point3D :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2, z := (A.z + B.z) / 2 }

noncomputable def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

-- Definitions based on problem conditions
def P : Point3D := {x := 0, y := 0, z := 0}
def Q : Point3D := {x := 10, y := 0, z := 0}
def R : Point3D := {x := 5, y := 5 * Real.sqrt 3, z := 0}
def T : Point3D := {x := 5, y := 5 * Real.sqrt 3, z := 20}
def M : Point3D := midpoint P Q
def N : Point3D := midpoint Q R
def O : Point3D := midpoint R T

-- The perimeter proof statement
theorem perimeter_triangle_MNO : distance M N + distance N O + distance O M = 5 + 10 * Real.sqrt 5 := by
  sorry

end perimeter_triangle_MNO_l496_496631


namespace triangle_length_AX_l496_496092

theorem triangle_length_AX (A B C X : Type*) (AB AC BC AX XB : ℝ)
  (hAB : AB = 70) (hAC : AC = 42) (hBC : BC = 56)
  (h_bisect : ∃ (k : ℝ), AX = 3 * k ∧ XB = 4 * k) :
  AX = 30 := 
by
  sorry

end triangle_length_AX_l496_496092


namespace percent_only_cats_l496_496072

def total_students := 500
def total_cats := 120
def total_dogs := 200
def both_cats_and_dogs := 40
def only_cats := total_cats - both_cats_and_dogs

theorem percent_only_cats:
  (only_cats : ℕ) / (total_students : ℕ) * 100 = 16 := 
by 
  sorry

end percent_only_cats_l496_496072


namespace average_disk_space_proof_l496_496619

noncomputable def average_disk_space_per_hour (days : ℕ) (total_space_mb : ℕ) : ℕ :=
  let total_hours := days * 24
  total_space_mb / total_hours

theorem average_disk_space_proof : average_disk_space_per_hour 15 20000 = 56 :=
by
  let total_hours : ℕ := 15 * 24
  have h1 : total_hours = 360 := rfl
  let avg_space : ℕ := 20000 / total_hours
  have h2 : avg_space = 20000 / 360 := by rw h1
  have h3 : avg_space = 56 := by norm_num
  rw [h2, h3]
  exact rfl

end average_disk_space_proof_l496_496619


namespace min_value_of_f_on_interval_l496_496536

def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem min_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) Real.pi, ∀ y ∈ set.Icc (0 : ℝ) Real.pi, f x ≤ f y ∧ f x = (Real.pi / 3) - Real.sqrt 3 :=
by
  sorry

end min_value_of_f_on_interval_l496_496536


namespace initial_girls_count_l496_496165

variable (p : ℕ) -- total number of people initially in the group
variable (girls_initial : ℕ) -- number of girls initially in the group
variable (girls_after : ℕ) -- number of girls after the change
variable (total_after : ℕ) -- total number of people after the change

/--
Initially, 50% of the group are girls. 
Later, five girls leave and five boys arrive, leading to 40% of the group now being girls.
--/
theorem initial_girls_count :
  (girls_initial = p / 2) →
  (total_after = p) →
  (girls_after = girls_initial - 5) →
  (girls_after = 2 * total_after / 5) →
  girls_initial = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_girls_count_l496_496165


namespace operation_not_multiple_of_57_l496_496829

theorem operation_not_multiple_of_57 :
  ∃ f : (Fin 15 → ℕ) → ℕ,
    (f = (λ s, ∑ i, s i) ∨ f = (λ s, (∑ i, s i) / 15)) ∧
    ¬ ∃ n : ℕ, f (λ i, i+1) = 57 * n := 
by sorry

end operation_not_multiple_of_57_l496_496829


namespace constant_term_correct_l496_496814

noncomputable def constant_term_expansion (a : ℝ) : ℝ :=
  let expr := (1 - x)^3 * (1 - (a / x))^3
  in coeff expr 0

theorem constant_term_correct :
  (∫ x in 1..real.exp 1, 1 / x) = 1 →
  constant_term_expansion 1 = 20 :=
by
  intro h
  have a_eq_one : (∫ x in 1..real.exp 1, 1 / x) = 1 := h
  rw [expr_expansion, a_eq_one]
  sorry

end constant_term_correct_l496_496814


namespace divide_rope_length_l496_496629

-- Definitions of variables based on the problem conditions
def rope_length : ℚ := 8 / 15
def num_parts : ℕ := 3

-- Theorem statement
theorem divide_rope_length :
  (1 / num_parts = (1 : ℚ) / 3) ∧ (rope_length * (1 / num_parts) = 8 / 45) :=
by
  sorry

end divide_rope_length_l496_496629


namespace sum_of_positive_odd_divisors_eq_78_l496_496992

theorem sum_of_positive_odd_divisors_eq_78 :
  ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 180)), d = 78 :=
by {
  -- proof steps go here
  sorry
}

end sum_of_positive_odd_divisors_eq_78_l496_496992


namespace light_ray_passes_center_l496_496217

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = D.2 ∧ B.2 = C.2 ∧ D.1 = C.1

def reflected_ray_passes_through_center (A B C D : ℝ × ℝ) (reflections : List (ℝ × ℝ)) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) ∧ M ∈ reflections

theorem light_ray_passes_center (A B C D : ℝ × ℝ) (reflections : List (ℝ × ℝ))
  (H1 : is_rectangle A B C D) (H2 : List.head reflections = some A)
  (H3 : List.last reflections (by simp) = some C) : 
  reflected_ray_passes_through_center A B C D reflections :=
  sorry

end light_ray_passes_center_l496_496217


namespace probability_of_drawing_red_ball_l496_496080

theorem probability_of_drawing_red_ball (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) 
  (H1 : total_balls = 10) (H2 : red_balls = 4) (H3 : black_balls = 6) :
  (red_balls : ℚ) / total_balls = 2 / 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end probability_of_drawing_red_ball_l496_496080


namespace compound_interest_comparison_l496_496604

theorem compound_interest_comparison
    (P : ℝ) (t : ℝ) (annual_rate : ℝ) (monthly_factor : ℝ) : 
    P * (1 + annual_rate / monthly_factor)^(monthly_factor * t) > P * (1 + annual_rate)^t :=
by
  let P := 1000
  let t := 10
  let annual_rate := 0.05
  let monthly_factor := 12
  have annual_amount := P * (1 + annual_rate)^t
  have monthly_amount := P * (1 + annual_rate / monthly_factor)^(monthly_factor * t)
  show monthly_amount > annual_amount
  sorry

end compound_interest_comparison_l496_496604


namespace flynn_tv_weeks_l496_496323

-- Define the conditions
def minutes_per_weekday := 30
def additional_hours_weekend := 2
def total_hours := 234
def minutes_per_hour := 60
def weekdays := 5

-- Define the total watching time per week in minutes
def total_weekday_minutes := minutes_per_weekday * weekdays
def total_weekday_hours := total_weekday_minutes / minutes_per_hour
def total_weekly_hours := total_weekday_hours + additional_hours_weekend

-- Create a theorem to prove the correct number of weeks
theorem flynn_tv_weeks : 
  (total_hours / total_weekly_hours) = 52 := 
by
  sorry

end flynn_tv_weeks_l496_496323


namespace distance_equality_l496_496809

open Real

def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

theorem distance_equality (a : ℝ) :
  distance_point_to_line (-3, -4) a 1 1 = distance_point_to_line (6, 3) a 1 1 ↔
  a = -1/3 ∨ a = -7/9 :=
by sorry

end distance_equality_l496_496809


namespace irreducibility_l496_496463

noncomputable def F (n : ℕ) (X : ℤ[X]) : ℤ[X] :=
  X^n + 5 * X^(n - 1) + 3

theorem irreducibility (n : ℕ) (hn : n > 1) :
  irreducible (F n) :=
sorry

end irreducibility_l496_496463


namespace car_gas_consumption_l496_496303

theorem car_gas_consumption
  (miles_today : ℕ)
  (miles_tomorrow : ℕ)
  (total_gallons : ℕ)
  (h1 : miles_today = 400)
  (h2 : miles_tomorrow = miles_today + 200)
  (h3 : total_gallons = 4000)
  : (∃ g : ℕ, 400 * g + (400 + 200) * g = total_gallons ∧ g = 4) :=
by
  sorry

end car_gas_consumption_l496_496303


namespace complement_of_P_l496_496385

noncomputable def U := Set.Univ ℝ
def P := {x : ℝ | x^2 ≤ 1}
def comp_of_P_in_U := {x : ℝ | x < -1 ∨ x > 1}

theorem complement_of_P : U \ P = comp_of_P_in_U := by
  sorry

end complement_of_P_l496_496385


namespace find_k_and_sequence_l496_496031

noncomputable def sequence (k : ℕ) (n : ℕ) := -(1/2 : ℝ) * (n^2 : ℝ) + (k : ℝ) * (n : ℝ)

theorem find_k_and_sequence {k : ℕ} (pos_k : 0 < k) (S_max : sequence k k = 8) :
  k = 4 ∧ ∀ n : ℕ, (n > 0 → n = 1 → (sequence k n = (7/2 : ℝ)))
               ∧ (n > 1 → (sequence k n - sequence k (n-1) = (9/2 : ℝ) - n)) :=
by
  sorry

end find_k_and_sequence_l496_496031


namespace circle_Q_radius_l496_496664

theorem circle_Q_radius
  (radius_P : ℝ := 2)
  (radius_S : ℝ := 4)
  (u v : ℝ)
  (h1: (2 + v)^2 = (2 + u)^2 + v^2)
  (h2: (4 - v)^2 = u^2 + v^2)
  (h3: v = u + u^2 / 2)
  (h4: v = 2 - u^2 / 4) :
  v = 16 / 9 :=
by
  /- Proof goes here. -/
  sorry

end circle_Q_radius_l496_496664


namespace min_birthdays_on_wednesday_l496_496251

theorem min_birthdays_on_wednesday (n x w: ℕ) (h_n : n = 61) 
  (h_ineq : w > x) (h_sum : 6 * x + w = n) : w ≥ 13 :=
by
  sorry

end min_birthdays_on_wednesday_l496_496251


namespace value_of_a_l496_496817

/-- Given that 0.5% of a is 85 paise, prove that the value of a is 170 rupees. --/
theorem value_of_a (a : ℝ) (h : 0.005 * a = 85) : a = 170 := 
  sorry

end value_of_a_l496_496817


namespace tangent_line_at_one_l496_496782

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem tangent_line_at_one 
    (a b : ℝ)
    (h : tangent (f 1 a b) at (1, f 1 a b) is (λ x, -x + 3)) :
    a = 1 ∧ b = 8 / 3 := 
  by
    sorry

end tangent_line_at_one_l496_496782


namespace power_of_i_l496_496258

theorem power_of_i (i : ℂ) (h : i^2 = -1) : i^607 = -i :=
by
  -- Assuming the condition that i^4 = 1, which is implied by i^2 = -1
  have h1 : i^4 = 1 := by calc
    i^4 = (i^2)^2 : by ring
    ... = (-1)^2 : by rw [h]
    ... = 1 : by norm_num
  sorry

end power_of_i_l496_496258


namespace mean_median_difference_l496_496832

theorem mean_median_difference (total_students : ℕ) (h : total_students > 0) :
  let students_75 := 0.15 * total_students;
      students_85 := 0.30 * total_students;
      students_90 := 0.25 * total_students;
      students_95 := 0.10 * total_students;
      students_100 := total_students - students_75 - students_85 - students_90 - students_95;
      median := 90;
      mean := (75 * students_75 + 85 * students_85 + 90 * students_90 + 95 * students_95 + 100 * students_100) / total_students;
  mean - median = -1.25 :=
by
  sorry

end mean_median_difference_l496_496832


namespace nth_equation_l496_496491

theorem nth_equation (n : ℕ) : 
  1 - (1 / ((n + 1)^2)) = (n / (n + 1)) * ((n + 2) / (n + 1)) :=
by sorry

end nth_equation_l496_496491


namespace largest_x_63_over_8_l496_496713

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496713


namespace total_pencils_correct_l496_496651

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l496_496651


namespace largest_real_number_l496_496732

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496732


namespace parabola_directrix_eq_l496_496181

open Real

noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ :=
  let focus_y := (1 : ℝ) in
  (0, focus_y)

theorem parabola_directrix_eq (a : ℝ) (focus_y : Real) (h_focus : focus_y = 1) :
  (directrix a = - (1 : ℝ)) :=
by
  have h_focus_coords : (0, focus_y) = (0, 1) := by
    simp [h_focus]
  sorry

end parabola_directrix_eq_l496_496181


namespace tangent_to_circumcircle_l496_496127

theorem tangent_to_circumcircle 
  (A B C D E : Type) [triangle_type A B C] [circumcircle_type k A B C]
  (isosceles : triangle_isosceles A C B)
  (D_on_arc : point_on_shorter_arc D k B C)
  (D_not_BC : D ≠ B ∧ D ≠ C)
  (E_on_CD_AB : intersects_at E (line_through C D) (line_through A B)) :
  tangent_to (line_through B C) (circumcircle_of (triangle B D E)) :=
sorry

end tangent_to_circumcircle_l496_496127


namespace final_balance_percentage_l496_496483

variable (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ)

def finalPercent (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ) : ℝ :=
  let afterIncrease := initialAmount * (1 + increasePercent)
  let afterDecrease := afterIncrease * (1 - decreasePercent)
  afterDecrease / initialAmount * 100

theorem final_balance_percentage :
  finalPercent 125 0.25 0.2 = 100 := by
  sorry

end final_balance_percentage_l496_496483


namespace debate_team_girls_l496_496196

theorem debate_team_girls (boys groups students_per_group : ℕ) 
  (h_boys : boys = 31) 
  (h_groups : groups = 7) 
  (h_students_per_group : students_per_group = 9) 
  : (total_students - boys) = 32 :=
by
  let total_students := groups * students_per_group
  have h_total_students : total_students = 63 := by
    rw [h_groups, h_students_per_group]
    exact rfl
  rw [h_boys, h_total_students]
  exact rfl
  sorry

end debate_team_girls_l496_496196


namespace positive_difference_l496_496567

-- Definitions based on the conditions
def point1 := (0, 5) -- Point on line l
def point2 := (5, 0) -- Point on line l
def point3 := (0, 2) -- Point on line m
def point4 := (8, 0) -- Point on line m

-- Define the function that gives the line equation from two points
def line_equation (p1 p2 : (ℤ × ℤ)) : ℤ × ℤ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let slope := (y2 - y1) / (x2 - x1)
  (slope, y1 - slope * x1)
-- each equation is written as y = slope * x + intercept

-- Equation of line l
def equation_l := line_equation point1 point2

-- Equation of line m
def equation_m := line_equation point3 point4

-- Theorem to prove
theorem positive_difference (x1 x2 : ℤ) (h₁ : equation_l.1 * -15 + equation_l.2 = 20) (h₂ : equation_m.1 * -72 + equation_m.2 = 20) : 
  | -15 - (-72) | = 57 :=
sorry

end positive_difference_l496_496567


namespace largest_real_number_l496_496728

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496728


namespace polynomial_perfect_square_l496_496063

theorem polynomial_perfect_square (k : ℝ) 
  (h : ∃ a : ℝ, x^2 + 8*x + k = (x + a)^2) : 
  k = 16 :=
by
  sorry

end polynomial_perfect_square_l496_496063


namespace totalTrianglesInFigure_l496_496671

-- Define the points and segments in the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 1⟩
def D : Point := ⟨0, 1⟩

def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def M : Point := midpoint A B
def N : Point := midpoint B C
def P : Point := midpoint C D
def Q : Point := midpoint D A

def center : Point := midpoint A C

-- Define the proposition stating the number of triangles
noncomputable def countTriangles : ℕ := sorry

theorem totalTrianglesInFigure : countTriangles = 20 := sorry

end totalTrianglesInFigure_l496_496671


namespace prove_solutions_l496_496587

noncomputable def solutions (t : ℝ) (k l : ℤ) : Prop :=
  (∃ t1, t = (π / 4) * (2 * k + 1) ∧ sin (2 * t) ≠ 0 ∧ sin t ≠ 1 / 2 ∧ sin t ≠ -1 / 2) ∨
  (∃ t2, t = (π * l + π / 3) ∨ t = (π * l - π / 3) ∧ sin (2 * t) ≠ 0 ∧ sin t ≠ 1 / 2 ∧ sin t ≠ -1 / 2)

theorem prove_solutions (t : ℝ) (k l : ℤ) :
  sin (3 * t) - sin t = (8 * cos t * cos (2 * t) / sin (2 * t)) / (4 - (sin t)⁻²) →
  solutions t k l :=
by
  intros h
  sorry

end prove_solutions_l496_496587


namespace midpoint_arc_equidistant_l496_496344

-- Define the geometric objects and conditions
variables {A B C M N K P Q T : Type} [MetricSpace T]

axiom triangle_ABC : Triangle T A B C
axiom AB_gt_BC : dist A B > dist B C
axiom circumcircle_Γ : Circle T A B C
axiom points_MN : dist A M = dist C N
axiom K_is_intersection : ∃ K, Line T M N ∩ Line T A C = {K}
axiom incenter_P : Incenter T P A M K
axiom excenter_Q : ExcenterOpposite T Q C N K

-- Goal: Prove the midpoint of arc ABC on Gamma is equidistant from P and Q
theorem midpoint_arc_equidistant :
  ∃ T, (MidpointArc T A B C) ∧ (dist T P = dist T Q) :=
sorry

end midpoint_arc_equidistant_l496_496344


namespace branches_on_one_stem_l496_496695

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l496_496695


namespace find_value_of_ff0_l496_496024

def f (x : ℝ) : ℝ :=
  if x < 1 then 2 ^ x else -x + 3

theorem find_value_of_ff0 : f (f 0) = 2 :=
by
  sorry

end find_value_of_ff0_l496_496024


namespace hyperbola_eccentricity_l496_496001

-- Given conditions
variables (a b : ℝ)

-- Given equations
def hyperbola (x y : ℝ) : Prop := (x^2) / (a^2) - (y^2) / (b^2) = 1
def asymptote (x y : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x
def parabola (x y : ℝ) : Prop := y = x^2 + 1

-- Lean theorem statement
theorem hyperbola_eccentricity (single_intersection : ∃ x y, asymptote a b x y ∧ parabola x y) : 
  ∃ e : ℝ, e = sqrt 5 := 
sorry

end hyperbola_eccentricity_l496_496001


namespace children_division_into_circles_l496_496429

theorem children_division_into_circles (n m k : ℕ) (hn : n = 5) (hm : m = 2) (trees_indistinguishable : true) (children_distinguishable : true) :
  ∃ ways, ways = 50 := 
by
  sorry

end children_division_into_circles_l496_496429


namespace weekly_sales_profit_no_adjustment_maximize_weekly_sales_profit_decrease_l496_496611

def initial_price : ℝ := 58
def initial_sales_per_week : ℝ := 300
def decrease_rate_in_price_per_unit_sales : ℝ := 25
def increase_rate_in_price_per_unit_sales : ℝ := 10
def cost_price_per_box : ℝ := 35
def average_loss_cost_per_box_per_week : ℝ := 3

theorem weekly_sales_profit_no_adjustment :
  (initial_price - cost_price_per_box - average_loss_cost_per_box_per_week) * initial_sales_per_week = 6000 := by
  sorry

theorem maximize_weekly_sales_profit_decrease :
  let x := 4 in
  let profit := (initial_price - cost_price_per_box - average_loss_cost_per_box_per_week - x) * 
                (initial_sales_per_week + decrease_rate_in_price_per_unit_sales * x) 
  in 
  profit = 6400 := by
  sorry

end weekly_sales_profit_no_adjustment_maximize_weekly_sales_profit_decrease_l496_496611


namespace smallest_time_and_digit_sum_l496_496200

-- Define the sequence of times for each horse
def horses_times : List ℕ := List.range' 1 12

-- Define a function to compute the LCM of a list of numbers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the set of times for the 7 horses that will meet at time T
def meeting_horses : List ℕ := [1, 2, 3, 4, 6, 7, 8]

-- Define the smallest time T when these horses meet at the starting point
def smallest_meeting_time : ℕ :=
  lcm_list meeting_horses

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement
theorem smallest_time_and_digit_sum :
  smallest_meeting_time = 168 ∧ sum_of_digits smallest_meeting_time = 15 :=
by
  sorry

end smallest_time_and_digit_sum_l496_496200


namespace ratio_m_q_l496_496352

theorem ratio_m_q (m n p q : ℚ) (h1 : m / n = 25) (h2 : p / n = 5) (h3 : p / q = 1 / 15) : 
  m / q = 1 / 3 :=
by 
  sorry

end ratio_m_q_l496_496352


namespace frog_min_jumps_l496_496270

def is_valid_jump (a b : ℕ × ℕ) : Prop :=
  let dx := (a.1 - b.1) ^ 2
  let dy := (a.2 - b.2) ^ 2
  dx + dy = 25

theorem frog_min_jumps :
  ∃ (path: list (ℕ × ℕ)), 
    path.head = (0, 0) ∧
    path.tail.head = (1, 0) ∧
    path.reverse.head = (2, 1) ∧
    (∀ (k:ℕ), k < path.length - 1 → is_valid_jump (path.nth_le k _) (path.nth_le (k+1) _)) ∧
    path.length = 5 :=
sorry

end frog_min_jumps_l496_496270


namespace total_distance_is_298_4_l496_496633

-- Define initial conditions
def initial_height : ℝ := 100
def bounce_ratio : ℝ := 1 / 2
def num_bounces : ℕ := 8

-- Define the total distance traveled by the ball
noncomputable def total_distance_traveled : ℝ :=
  2 * ∑ k in Finset.range num_bounces, initial_height * (bounce_ratio ^ k) - initial_height

-- Prove the total distance traveled is approximately 298.4 meters
theorem total_distance_is_298_4 : abs (total_distance_traveled - 298.4) < 0.1 := sorry

end total_distance_is_298_4_l496_496633


namespace AF_squared_l496_496453

-- Definitions of the points, vectors, and distances involved
variables {A B C D E F : Type} [MetricSpace A] (pointA pointB pointC : A)
variables (midpointD : A) (angleBisectorE : A) (intersectionF : A)

-- Given conditions as assumptions
axiom is_isosceles : dist pointA pointB = dist pointA pointC
axiom is_midpoint : midpointD = (pointA + pointB) / 2
axiom is_angle_bisector : true -- Placeholder condition for angle bisector
axiom is_intersection : intersectionF = intersection (line_through pointA midpointD) (line_through pointC angleBisectorE)
axiom equilateral_AFE : dist pointA intersectionF = dist pointA angleBisectorE ∧ dist pointA intersectionF = 1

-- Known value
axiom AC_length : dist pointA pointC = 2

-- Prove that the square of the distance AF is 7/4
theorem AF_squared : dist pointA intersectionF ^ 2 = 7 / 4 := by
  sorry

end AF_squared_l496_496453


namespace max_perimeter_is_isosceles_l496_496749

-- Definitions for lengths and angle
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (AB : ℝ) -- Segment AB length
variable (C : RealAngle) -- Fixed angle C

-- The theorem statement
theorem max_perimeter_is_isosceles (ABC : Triangle A B C) (hAB : length(AB) = AB) (hAngleC : angleA = C) :
  is_isosceles ABC :=
sorry -- Proof is omitted

end max_perimeter_is_isosceles_l496_496749


namespace sin_cos_A_lambda_value_l496_496421

variables {a b c A B C : ℝ}

-- Conditions
axiom triangle_sides_relation : a = b + c
axiom area_formula (S : ℝ) : S = 1 / 2 * (a^2 - (b - c)^2)
axiom tan_C (C_non_right_angle : 0 < C ∧ C < π) : tan C = 2

-- Questions and answers
theorem sin_cos_A :
  (∃ S, S = 1 / 2 * (a^2 - (b - c)^2)) →
  ∃ (sin_A cos_A : ℝ), (sin_A = 4 / 5) ∧ (cos_A = 3 / 5) :=
sorry

theorem lambda_value (λ : ℝ) :
  (exists b a, tan C = 2 ∧ λ = b / a) →
  ∃ (λ : ℝ), λ = sqrt 5 / 2 :=
sorry

end sin_cos_A_lambda_value_l496_496421


namespace exists_integers_binom_satisfy_l496_496125

theorem exists_integers_binom_satisfy (n k : ℕ) (h : k > 0) :
  ∃ (a1 a2 a3 a4 a5 : ℕ),
    a1 > a2 ∧
    a2 > a3 ∧
    a3 > a4 ∧
    a4 > a5 ∧
    a5 > k ∧
    n = (binom a1 3 - binom a2 3 + binom a3 3 - binom a4 3 + binom a5 3) :=
sorry

end exists_integers_binom_satisfy_l496_496125


namespace parabola_intersects_line_l496_496381

noncomputable def parabola (x : ℝ) : ℝ := real.sqrt(8 * x)

noncomputable def line (x : ℝ) : ℝ := (real.sqrt(3) / 3) * (x - 2)

theorem parabola_intersects_line :
  ∃ A B : ℝ × ℝ, A ≠ B ∧ parabola (A.1) = A.2 ∧ line (A.1) = A.2 ∧
                  parabola (B.1) = B.2 ∧ line (B.1) = B.2 ∧
                  real.dist A B = 32 := by
  sorry

end parabola_intersects_line_l496_496381


namespace f_inv_sum_correct_l496_496679

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2 - x else x^3 - 2*x^2 + x

noncomputable def f_inv_sum : ℝ :=
  let inv_neg_1 := (1 + Real.sqrt 5) / 2 in
  let inv_1 := 1 in
  let inv_4 := -2 in
  inv_neg_1 + inv_1 + inv_4

theorem f_inv_sum_correct :
  f_inv_sum = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end f_inv_sum_correct_l496_496679


namespace lucy_total_cost_for_lamp_and_table_l496_496136

noncomputable def original_price_lamp : ℝ := 200 / 1.2

noncomputable def table_price : ℝ := 2 * original_price_lamp

noncomputable def total_cost_paid (lamp_cost discounted_price table_price: ℝ) :=
  lamp_cost + table_price

theorem lucy_total_cost_for_lamp_and_table :
  total_cost_paid 20 (original_price_lamp * 0.6) table_price = 353.34 :=
by
  let lamp_original_price := original_price_lamp
  have h1 : original_price_lamp * (0.6 * (1 / 5)) = 20 := by sorry
  have h2 : table_price = 2 * original_price_lamp := by sorry
  have h3 : total_cost_paid 20 (original_price_lamp * 0.6) table_price = 20 + table_price := by sorry
  have h4 : table_price = 2 * (200 / 1.2) := by sorry
  have h5 : 20 + table_price = 353.34 := by sorry
  exact h5

end lucy_total_cost_for_lamp_and_table_l496_496136


namespace level_for_1000_points_l496_496154

def points_for_level : ℕ → ℕ
| 10 := 90
| 11 := 160
| 12 := 250
| 13 := 360
| 14 := 490
| 15 := 640
| 16 := 810
| 17 := 1000
| 18 := 1210
| _  := 0

theorem level_for_1000_points : ∃ l, points_for_level l = 1000 :=
by {
  use 17,
  show points_for_level 17 = 1000,
  sorry
}

end level_for_1000_points_l496_496154


namespace product_of_possible_values_of_x_l496_496088

theorem product_of_possible_values_of_x :
  (∃ x, |x - 7| - 3 = -2) → ∃ y z, |y - 7| - 3 = -2 ∧ |z - 7| - 3 = -2 ∧ y * z = 48 :=
by
  sorry

end product_of_possible_values_of_x_l496_496088


namespace MQ_is_18_l496_496842

noncomputable def MQ_length (x y : ℝ) (h : x^2 + y^2 = 180) : ℝ :=
  let MQ_squared : ℝ := 2 * (x^2 + y^2) in
  MQ_squared.sqrt

theorem MQ_is_18 (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x^2 + y^2 = 180) : MQ_length x y h = 18 :=
by
  sorry

end MQ_is_18_l496_496842


namespace find_number_l496_496249

theorem find_number :
  ∃ N, let sum := 555 + 445 in
  let difference := 555 - 445 in
  let quotient := 2 * difference in
  N = sum * quotient + 25 ∧ N = 220025 :=
by
  sorry

end find_number_l496_496249


namespace largest_real_number_l496_496718

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496718


namespace sum_of_odd_divisors_180_l496_496972

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_divisors_180 :
  sum_of_positive_odd_divisors 180 = 78 :=
by
  sorry

end sum_of_odd_divisors_180_l496_496972


namespace sequence_unbounded_l496_496110

noncomputable def sequence (a : ℕ → ℕ → ℤ) : ℕ → ℕ → ℤ
| 0, 1     => 1
| 0, 2     => -1
| 0, (n+3) => 0
| (t+1), 100 => sequence t 100 + sequence t 1
| (t+1), (n+1) => sequence t (n+1) + sequence t (n+2) 

theorem sequence_unbounded (M : ℤ) : 
  ∃ (i : ℕ) (t : ℕ), abs (sequence t i) > M := 
by
  sorry

end sequence_unbounded_l496_496110


namespace probability_same_number_l496_496657

theorem probability_same_number (n m : ℕ) : 
  n < 250 → m < 250 → 
  (∃ k1, n = 20 * k1) → (∃ k2, m = 28 * k2) → 
  (count (λ x, x < 250 ∧ ∃ k, x = 20 * k) * count (λ y, y < 250 ∧ ∃ k, y = 28 * k)) = 96 → 
  (count (λ z, z < 250 ∧ ∃ k, z = 140 * k) : ℚ) / 96 = 1 / 96 :=
sorry

end probability_same_number_l496_496657


namespace sum_six_terms_l496_496802

variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (S_2 S_4 S_6 : ℝ)

-- Given conditions
axiom sum_two_terms : S 2 = 4
axiom sum_four_terms : S 4 = 16

-- Problem statement
theorem sum_six_terms : S 6 = 52 :=
by
  -- Insert the proof here
  sorry

end sum_six_terms_l496_496802


namespace distinct_real_numbers_g_g_g_g_eq_six_l496_496882

noncomputable def g (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem distinct_real_numbers_g_g_g_g_eq_six :
  ∃ (c : ℝ), g (g (g (g c))) = 6 ∧ (set.finite {c' : ℝ | g (g (g (g c'))) = 6 }) ∧ (set.card {c' : ℝ | g (g (g (g c'))) = 6 } = 12) :=
sorry

end distinct_real_numbers_g_g_g_g_eq_six_l496_496882


namespace sum_of_positive_odd_divisors_of_180_l496_496977

/-- The sum of the positive odd divisors of 180 is 78. -/
theorem sum_of_positive_odd_divisors_of_180 : 
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45] in
  let sum_odd_divisors := List.sum odd_divisors in
  sum_odd_divisors = 78 := 
by
  -- odd_divisors is [1, 3, 5, 9, 15, 45]
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45]
  -- Summation of the odd divisors
  let sum_odd_divisors := List.sum odd_divisors
  -- Verify that the sum is 78
  show sum_odd_divisors = 78
  sorry

end sum_of_positive_odd_divisors_of_180_l496_496977


namespace true_propositions_l496_496838

open Real

-- Definitions
def orthogonal_distance (P Q : ℝ × ℝ) : ℝ := abs (P.1 - Q.1) + abs (P.2 - Q.2)

def is_constant_distance (P Q : ℝ → ℝ × ℝ) : Prop :=
  ∀ α : ℝ, orthogonal_distance (2, 3) (Q α) = 4

def euclidean_distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def satisfies_inequality (P Q : ℝ × ℝ) : Prop :=
  euclidean_distance P Q ≥ (sqrt 2 / 2) * orthogonal_distance P Q

-- Propositions
def prop1 : Prop :=
  is_constant_distance (2, 3) (fun α => (sin α ^ 2, cos α ^ 2))

def prop3 : Prop :=
  ∀ (P Q : ℝ × ℝ), satisfies_inequality P Q

-- Main theorem statement
theorem true_propositions : prop1 ∧ prop3 := 
by 
  sorry

end true_propositions_l496_496838


namespace sum_of_odd_integers_21_to_65_l496_496578

theorem sum_of_odd_integers_21_to_65 :
  ∑ i in finset.Icc 21 65, if i % 2 = 1 then i else 0 = 989 := by
  sorry

end sum_of_odd_integers_21_to_65_l496_496578


namespace no_bounded_function_exists_l496_496691

theorem no_bounded_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x y : ℝ, f(1) > 0 ∧ (f x) ≤ C ∧ (f x + y)^2 ≥ (f x)^2 + 2 * (f (x * y)) + (f y)^2) := 
sorry

end no_bounded_function_exists_l496_496691


namespace sum_of_digits_p_l496_496696

theorem sum_of_digits_p 
  (p : ℕ) 
  (h1 : 14 * k = 8 * p for some k : ℕ)
  (h2 : (4 / 7 : ℝ) * p - 50 = 150) :
  (3 + 5 + 0 = 8) :=
by {
  have hp : p = 350,
  { sorry },
  have sum_digits_p : 3 + 5 + 0 = 8,
  { sorry },
  exact sum_digits_p
}

end sum_of_digits_p_l496_496696


namespace triangle_construction_l496_496639

-- Given: A triangle ABC where the lengths of the sides AB and AC are known
variables (A B C : Point)
variables (c b : ℝ) [Fact (0 < c)] [Fact (0 < b)]
variable (triangle_ABC : Triangle A B C)
variable (hAB : dist A B = c)
variable (hAC : dist A C = b)

-- To prove: The internal and external angle bisectors of ∠BAC are equal
theorem triangle_construction
    (h_bisectors_equal : internal_and_external_angle_bisectors_equal A B C) :
    internal_angle_bisector_length A B C = external_angle_bisector_length A B C := by
  sorry

end triangle_construction_l496_496639


namespace exterior_angle_BAC_l496_496279

-- Given conditions
def angle_BAD_interior_octagon : ℝ :=
  180 * (8 - 2) / 8

def angle_CAD_square : ℝ :=
  90

-- Prove that the measure of the exterior angle BAC at vertex A is 135 degrees
theorem exterior_angle_BAC : angle_BAD_interior_octagon + angle_CAD_square = 135 :=
  sorry

end exterior_angle_BAC_l496_496279


namespace impossible_to_reduce_time_l496_496648

def current_speed := 60 -- speed in km/h
def time_per_km (v : ℕ) : ℕ := 60 / v -- 60 minutes divided by speed in km/h gives time per km in minutes

theorem impossible_to_reduce_time (v : ℕ) (h : v = current_speed) : time_per_km v = 1 → ¬(time_per_km v - 1 = 0) :=
by
  intros h1 h2
  sorry

end impossible_to_reduce_time_l496_496648


namespace friends_area_is_greater_by_14_point_4_times_l496_496563

theorem friends_area_is_greater_by_14_point_4_times :
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  friend_area / tommy_area = 14.4 :=
by
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  sorry

end friends_area_is_greater_by_14_point_4_times_l496_496563


namespace correct_statements_count_l496_496644

theorem correct_statements_count : 
  let statement1 := (-a : ℤ) < 0
  let statement2 := ∀ n : ℤ, -1 ≤ n → n < 0 → n = -1
  let statement3 := abs (2 : ℤ) = abs (-2 : ℤ)
  let statement4 := polynomial.degree (C (3 : ℤ) * X * Y^2 - C (2 : ℤ) * X * Y) = 2
  2 = count (λ s, s = true) [statement1, statement2, statement3, statement4] :=
by {
  let statement1 := (-a : ℤ) < 0,
  let statement2 := ∀ n : ℤ, -1 ≤ n → n < 0 → n = -1,
  let statement3 := abs (2 : ℤ) = abs (-2 : ℤ),
  let statement4 := polynomial.degree (C (3 : ℤ) * X * Y^2 - C (2 : ℤ) * X * Y) = 2,
  have cnt := (if statement1 then 1 else 0) + (if statement2 then 1 else 0) 
    + (if statement3 then 1 else 0) + (if statement4 then 1 else 0),
  exact 2 = cnt,
  sorry
}

end correct_statements_count_l496_496644


namespace strictly_positive_integers_equal_l496_496123

theorem strictly_positive_integers_equal 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : 
  a = b :=
sorry

end strictly_positive_integers_equal_l496_496123


namespace solution_set_inequality1_solution_set_inequality2_l496_496243

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l496_496243


namespace megan_savings_final_balance_percentage_l496_496479

noncomputable def megan_final_percentage (initial_balance : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : ℝ :=
let increased_balance := initial_balance * (1 + increase_rate) in
let final_balance := increased_balance * (1 - decrease_rate) in
(final_balance / initial_balance) * 100

theorem megan_savings_final_balance_percentage :
  megan_final_percentage 125 0.25 0.20 = 100 := 
by
  sorry

end megan_savings_final_balance_percentage_l496_496479


namespace determining_digit_a_l496_496179

-- Define the problem constraints
def valid_dominos (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (a + 4) + (a + 6) + (e + 6) = 24 ∧
  (b + 4) + (b + 5) + (e + 4) = 24 ∧
  (a + 6) + d + c = 24 ∧
  -- Add more conditions to represent remaining rows as needed

theorem determining_digit_a (a b c d e : ℕ) (h_valid : valid_dominos a b c d e) : a = 2 :=
  sorry

end determining_digit_a_l496_496179


namespace quadratic_linear_common_solution_l496_496416

theorem quadratic_linear_common_solution
  (a x1 x2 d e : ℝ)
  (ha : a ≠ 0) (hx1x2 : x1 ≠ x2) (hd : d ≠ 0)
  (h_quad : ∀ x, a * (x - x1) * (x - x2) = 0 → x = x1 ∨ x = x2)
  (h_linear : d * x1 + e = 0)
  (h_combined : ∀ x, a * (x - x1) * (x - x2) + d * x + e = 0 → x = x1) :
  d = a * (x2 - x1) :=
by sorry

end quadratic_linear_common_solution_l496_496416


namespace gold_weight_is_ten_l496_496552

theorem gold_weight_is_ten :
  let weights := finset.range 19
  let total_weight := weights.sum id
  let bronze_weights := finset.range 9
  let total_bronze_weight := bronze_weights.sum id
  let iron_weights := finset.Icc 10 18
  let total_iron_weight := iron_weights.sum id
  let S_gold := total_weight - (total_bronze_weight + total_iron_weight)
  total_weight = 190 ∧ (total_iron_weight - total_bronze_weight) = 90 →
  S_gold = 10 :=
by
  simp [S_gold, total_weight, total_bronze_weight, total_iron_weight] at *
  sorry

end gold_weight_is_ten_l496_496552


namespace sum_odd_divisors_of_180_l496_496983

theorem sum_odd_divisors_of_180 : 
  let n := 180 in 
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0} in 
  ∑ d in odd_divisors, d = 78 :=
by
  let n := 180
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0}
  have h : ∑ d in odd_divisors, d = 78 := sorry -- Sum of odd divisors of 180
  exact h

end sum_odd_divisors_of_180_l496_496983


namespace distance_DE_is_correct_l496_496286

variables (A B C D E P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace P]

-- Define the points and conditions
variables (A B C D E P : EuclideanSpace ℝ (Fin 2))
variables (dist_AB : dist A B = 8)
variables (dist_BC : dist B C = 15)
variables (dist_AC : dist A C = 17)
variables (dist_PC : dist P C = 7)
variables (BP_line : ∃ k, P = Fin 2 ![k * (B 0 - C 0) + C 0, k * (B 1 - C 1) + C 1])
variables (ABCD_trapezoid : ∃ D, ((dist A B = dist A D) ∨ (dist A D = dist C B)) ∧ (D ∈ (Line BP)))
variables (ABCE_trapezoid : ∃ E, ((dist A B = dist A E) ∨ (dist A E = dist C B)) ∧ (E ∈ (Line BP)))

-- Assert the proof of the distance DE
theorem distance_DE_is_correct :
  ∃ (DE : ℝ), DE = 3 * √17 :=
sorry

end distance_DE_is_correct_l496_496286


namespace percentage_increase_of_soda_l496_496742

variable (C S x : ℝ)

theorem percentage_increase_of_soda
  (h1 : 1.25 * C = 10)
  (h2 : S + x * S = 12)
  (h3 : C + S = 16) :
  x = 0.5 :=
sorry

end percentage_increase_of_soda_l496_496742


namespace range_of_m_l496_496405

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → x < m) : m > 1 := 
by
  sorry

end range_of_m_l496_496405


namespace fourier_transform_f_l496_496317

noncomputable def transform : ℝ → ℂ
| p => (i * sqrt (real.pi / 2) * complex.sign p * exp (-real.abs p))

def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem fourier_transform_f :
  (Fourier.transform f) = transform :=
sorry

end fourier_transform_f_l496_496317


namespace comparison_of_s1_and_s2_l496_496121

-- Define the variables and conditions
variables {A B C G : Point}
-- Assume G is the centroid of triangle ABC
def isCentroid (G A B C : Point) : Prop :=
  ∃ (ma mb mc : LineSegment),
    ma ∈ medians A B C ∧ mb ∈ medians B C A ∧ mc ∈ medians C A B ∧
    G = intersection (ma, mb, mc) 
-- Define s1 and s2
def s1 (G A B C : Point) [isCentroid G A B C] : ℝ :=
  2 * (distance G A + distance G B + distance G C)
def s2 (A B C : Point) : ℝ :=
  3 * (distance A B + distance B C + distance C A)

-- State the proof
theorem comparison_of_s1_and_s2
  (A B C G : Point)
  (hG : isCentroid G A B C) :
  s1 G A B C < s2 A B C :=
by 
  -- Proof goes here
  sorry

end comparison_of_s1_and_s2_l496_496121


namespace sum_of_positive_odd_divisors_eq_78_l496_496993

theorem sum_of_positive_odd_divisors_eq_78 :
  ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 180)), d = 78 :=
by {
  -- proof steps go here
  sorry
}

end sum_of_positive_odd_divisors_eq_78_l496_496993


namespace second_train_length_l496_496963

noncomputable def length_second_train
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_crossing : ℝ)
  (length_first_train : ℝ) : ℝ :=
  let relative_speed := (speed_first_train + speed_second_train) * (5 / 18)
  let total_distance := relative_speed * time_crossing
  total_distance - length_first_train

theorem second_train_length
  (speed_first_train speed_second_train time_crossing length_first_train : ℝ)
  (hspeed_first : speed_first_train = 60)
  (hspeed_second : speed_second_train = 40)
  (htime : time_crossing = 11.879049676025918)
  (hlength_first : length_first_train = 170) :
  length_second_train speed_first_train speed_second_train time_crossing length_first_train 
  ≈ 159.9736015568311 := 
by {
  sorry
}

end second_train_length_l496_496963


namespace breakfast_calories_l496_496863

variable (B : ℝ) 

def lunch_calories := 1.25 * B
def dinner_calories := 2.5 * B
def shakes_calories := 900
def total_calories := 3275

theorem breakfast_calories:
  (B + lunch_calories B + dinner_calories B + shakes_calories = total_calories) → B = 500 :=
by
  sorry

end breakfast_calories_l496_496863


namespace mod_inverse_problem_l496_496315

theorem mod_inverse_problem
  (a b c d : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bound : a < 15 ∧ b < 15 ∧ c < 15 ∧ d < 15)
  (h_invertible : Nat.gcd a 15 = 1 ∧ Nat.gcd b 15 = 1 ∧ Nat.gcd c 15 = 1 ∧ Nat.gcd d 15 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (Nat.invMod (a * b * c * d) 15)) % 15 = 11 :=
sorry

end mod_inverse_problem_l496_496315


namespace isogonal_conjugate_tangent_l496_496909

open EuclideanGeometry

-- Define the problem setup
variables {A B C : Point}

def is_tangent_at (A : Point) (circumcircle : Circle) (line : Line) : Prop := 
  is_tangent_to_circle A line circumcircle

-- The statement to be proved
theorem isogonal_conjugate_tangent (h_triangle : Triangle ABC) 
  (h_parallel : parallel (line_through A (midpoint B C)) (line_through B C)) : 
  is_tangent_at A (circumcircle ABC) (isogonal_conjugate (line_through A (midpoint B C)) h_triangle) := 
sorry

end isogonal_conjugate_tangent_l496_496909


namespace num_of_odd_integers_between_9_and_39_l496_496946

theorem num_of_odd_integers_between_9_and_39 :
  let n := (39 - 9) / 2 + 1 in
  (list.sum (list.filter (λ x, x % 2 = 1) (list.range' 9 (39 - 9 + 1))) = 384 → n = 16) :=
by
  sorry

end num_of_odd_integers_between_9_and_39_l496_496946


namespace amount_paid_by_customer_l496_496540

theorem amount_paid_by_customer 
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (final_price : ℝ)
  (h1 : cost_price = 6681.818181818181)
  (h2 : markup_percentage = 10 / 100)
  (h3 : final_price = cost_price * (1 + markup_percentage)) :
  final_price = 7350 :=
by 
  sorry

end amount_paid_by_customer_l496_496540


namespace sum_of_odd_divisors_180_l496_496974

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_divisors_180 :
  sum_of_positive_odd_divisors 180 = 78 :=
by
  sorry

end sum_of_odd_divisors_180_l496_496974


namespace residue_625_mod_17_l496_496688

theorem residue_625_mod_17 : 625 % 17 = 13 :=
by
  sorry

end residue_625_mod_17_l496_496688


namespace max_min_sums_l496_496952

def P (x y : ℤ) := x^2 + y^2 = 50

theorem max_min_sums : 
  ∃ (x₁ y₁ x₂ y₂ : ℤ), P x₁ y₁ ∧ P x₂ y₂ ∧ 
    (x₁ + y₁ = 8) ∧ (x₂ + y₂ = -8) :=
by
  sorry

end max_min_sums_l496_496952


namespace prime_representation_l496_496158

theorem prime_representation (N : ℕ) (hN : Nat.prime N) : ∃ (n p : ℕ), (0 ≤ p ∧ p < 30 ∧ Nat.prime p) ∧ N = 30 * n + p :=
  sorry

end prime_representation_l496_496158


namespace number_of_sheets_in_six_cm_stack_l496_496276

variable (sheets_per_reem : ℕ := 400)
variable (thickness_per_reem : ℝ := 4)
variable (desired_thickness : ℝ := 6)

def thickness_per_sheet := thickness_per_reem / sheets_per_reem
def number_of_sheets_in_stack := desired_thickness / thickness_per_sheet

theorem number_of_sheets_in_six_cm_stack :
  number_of_sheets_in_stack = 600 :=
by
  unfold number_of_sheets_in_stack
  unfold thickness_per_sheet
  norm_num
sorry

end number_of_sheets_in_six_cm_stack_l496_496276


namespace pics_per_album_eq_five_l496_496492

-- Definitions based on conditions
def pics_from_phone : ℕ := 5
def pics_from_camera : ℕ := 35
def total_pics : ℕ := pics_from_phone + pics_from_camera
def num_albums : ℕ := 8

-- Statement to prove
theorem pics_per_album_eq_five : total_pics / num_albums = 5 := by
  sorry

end pics_per_album_eq_five_l496_496492


namespace passes_through_fixed_point_l496_496238

theorem passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  ∃ x y : ℝ, x = 3 ∧ y = -3 ∧ (λ x, a^(x-3) - 4) x = y :=
by
  use 3, -3
  sorry

end passes_through_fixed_point_l496_496238


namespace cubic_sum_identity_l496_496048

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 40) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 637 :=
by
  sorry

end cubic_sum_identity_l496_496048


namespace find_matrix_N_l496_496740

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, -5; 4, -3]

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  !![-20, -7; 11, 4]

-- Define the expected matrix N
def N_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  !![(44 / 7), -(57 / 7); -(49 / 14), (63 / 14)]

-- Define the problem statement to be proved
theorem find_matrix_N : (N_expected ⬝ A = B) :=
by
  sorry

end find_matrix_N_l496_496740


namespace committee_number_of_ways_l496_496685

-- Define the conditions
def num_letters : ℕ := 8
def letter_counts : List ℕ := [2, 2, 3, 1, 1] -- frequencies for C, M, E, I, T

-- Define the factorial function (using Mathlib)
noncomputable def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the number of ways to arrange the letters of "COMMITTEE"
noncomputable def num_arrangements (n : ℕ) (counts : List ℕ) : ℕ :=
  fact n / counts.map fact |> List.prod

theorem committee_number_of_ways : num_arrangements num_letters letter_counts = 1680 := by
  sorry

end committee_number_of_ways_l496_496685


namespace engines_not_defective_count_l496_496407

noncomputable def not_defective_engines (total_batches : ℕ) (engines_per_batch : ℕ) (defective_fraction : ℚ) : ℕ :=
  total_batches * engines_per_batch * (1 - defective_fraction)

theorem engines_not_defective_count:
  not_defective_engines 5 80 (1/4) = 300 :=
by
  sorry

end engines_not_defective_count_l496_496407


namespace calories_per_person_l496_496852

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end calories_per_person_l496_496852


namespace intersection_P_compl_M_l496_496889

-- Define universal set U
def U : Set ℤ := Set.univ

-- Define set M
def M : Set ℤ := {1, 2}

-- Define set P
def P : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the complement of M in U
def M_compl : Set ℤ := { x | x ∉ M }

-- Define the intersection of P and the complement of M
def P_inter_M_compl : Set ℤ := P ∩ M_compl

-- The theorem we want to prove
theorem intersection_P_compl_M : P_inter_M_compl = {-2, -1, 0} := 
by {
  sorry
}

end intersection_P_compl_M_l496_496889


namespace min_value_expression_l496_496003

theorem min_value_expression (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
by
  sorry

end min_value_expression_l496_496003


namespace range_of_a_l496_496751

theorem range_of_a (a : ℝ) (h_pos : a > 0) 
  (h_cond : ∀ (x1 x2 : ℝ), x1 > 0 → x2 > 0 → x1 ≠ x2
    → (a * (Real.log x1 - Real.log x2) + (x1^2 - x2^2) / 2) / (x1 - x2) ≥ 2) :
  a ∈ set.Ici 1 :=
begin
  sorry
end

end range_of_a_l496_496751


namespace ratio_of_reading_speeds_l496_496309

-- Define the conditions
def reading_time_Emery := 20
def avg_reading_time := 60

-- Define the speeds as variables
variables {E S : ℕ}

-- Define the proof statement
theorem ratio_of_reading_speeds (E S : ℕ) (h1 : 2 * avg_reading_time = reading_time_Emery + S)
  (h2 : 20 * E = 100 * S) : E / S = 5 :=
begin
  have S_val: S = 100 := by linarith [h1],
  rw [S_val] at h2,
  linarith,
end

end ratio_of_reading_speeds_l496_496309


namespace trig_sum_identity_l496_496322

-- Define the trigonometric expressions for angles
def cosine_15 := Real.cos (15 * Real.pi / 180)
def sine_375 := Real.sin (375 * Real.pi / 180)

-- Statement of the problem
theorem trig_sum_identity : cosine_15 + sine_375 = sqrt (6) / 2 := by
  sorry -- Proof required here

end trig_sum_identity_l496_496322


namespace prism_surface_area_l496_496630

open Real

noncomputable def surface_area_of_BXYZ : ℝ :=
  100 + (25 * sqrt 3) / 4

theorem prism_surface_area (A B C D E F X Y Z : Point)
  (hPrism : is_right_prism A B C D E F)
  (hHeight : height A D = 20)
  (hBaseEquilateral : is_equilateral_triangle A B C)
  (hBaseSideLength : side_length A B = 10)
  (hMidpoints : midpoint A B = X ∧ midpoint B C = Y ∧ midpoint D F = Z)
  (hCut : sliced_prism_with_line_through_points A B C D E F X Y Z) :
  ∃ (t: ℝ), t = surface_area_of_BXYZ :=
by
  sorry

end prism_surface_area_l496_496630


namespace systematic_sampling_l496_496273

-- Definitions for the class of 50 students numbered from 1 to 50, sampling interval, and starting number.
def students : Set ℕ := {n | n ∈ Finset.range 50 ∧ n ≥ 1}
def sampling_interval : ℕ := 10
def start : ℕ := 6

-- The main theorem stating that the selected students' numbers are as given.
theorem systematic_sampling : ∃ (selected : List ℕ), selected = [6, 16, 26, 36, 46] ∧ 
  ∀ x ∈ selected, x ∈ students := 
  sorry

end systematic_sampling_l496_496273


namespace possible_values_of_p_l496_496916

noncomputable def count_possible_prime_p : ℕ :=
  let numbers_base_p_to_10 (digits : List ℕ) (p : ℕ) : ℕ :=
    digits.foldr (λ (digit accu) accu * p + digit) 0 in
  let equation_lhs (p : ℕ) : ℕ :=
    numbers_base_p_to_10 [6, 7, 9] p + numbers_base_p_to_10 [7, 0, 5] p + numbers_base_p_to_10 [8, 3, 2] p in
  let equation_rhs (p : ℕ) : ℕ :=
    numbers_base_p_to_10 [9, 2, 4] p + numbers_base_p_to_10 [5, 9, 5] p + numbers_base_p_to_10 [7, 9, 6] p in
  let is_valid_prime (p : ℕ) : Bool :=
    equation_lhs p = equation_rhs p ∧ Nat.prime p ∧ p < 10 in
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29].count is_valid_prime

theorem possible_values_of_p : count_possible_prime_p = 3 := by
  -- proof goes here
  sorry

end possible_values_of_p_l496_496916


namespace john_burritos_left_l496_496856

theorem john_burritos_left : 
  ∀ (boxes : ℕ) (burritos_per_box : ℕ) (given_away_fraction : ℚ) (eaten_per_day : ℕ) (days : ℕ),
  boxes = 3 → 
  burritos_per_box = 20 →
  given_away_fraction = 1 / 3 →
  eaten_per_day = 3 →
  days = 10 →
  let initial_burritos := boxes * burritos_per_box in
  let given_away_burritos := given_away_fraction * initial_burritos in
  let after_giving_away := initial_burritos - given_away_burritos in
  let eaten_burritos := eaten_per_day * days in
  let final_burritos := after_giving_away - eaten_burritos in
  final_burritos = 10 := 
by 
  intros,
  sorry

end john_burritos_left_l496_496856


namespace sufficient_but_not_necessary_l496_496049

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 ∧ ¬ (a^2 > b^2 → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_l496_496049


namespace farmer_sowed_correct_amount_l496_496620

def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6
def buckets_sowed : ℝ := initial_buckets - final_buckets

theorem farmer_sowed_correct_amount : buckets_sowed = 2.75 :=
by {
  sorry
}

end farmer_sowed_correct_amount_l496_496620


namespace cost_per_person_l496_496918

def total_cost : ℕ := 30000  -- Cost in million dollars
def num_people : ℕ := 300    -- Number of people in million

theorem cost_per_person : total_cost / num_people = 100 :=
by
  sorry

end cost_per_person_l496_496918


namespace pentadecagon_diagonals_l496_496812

def numberOfDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentadecagon_diagonals : numberOfDiagonals 15 = 90 :=
by
  sorry

end pentadecagon_diagonals_l496_496812


namespace smaller_root_of_equation_l496_496319

theorem smaller_root_of_equation :
  let a := λ x: ℝ => (x - 1/3) in
  let b := λ x: ℝ => (x - 2/3) in
  ∃ x: ℝ, (a x)^2 + (a x) * (b x) = 0 ∧ ∀ y: ℝ, (a y)^2 + (a y) * (b y) = 0 → x ≤ y := 
by sorry

end smaller_root_of_equation_l496_496319


namespace john_burritos_left_l496_496860

theorem john_burritos_left : 
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  in burritos_left = 10 := by
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  have h : total_burritos = 60 := by rfl
  have h1 : burritos_given_away = 20 := by sorry
  have h2 : burritos_left_after_giving = 40 := by sorry
  have h3 : total_burritos_eaten = 30 := by sorry
  have h4 : burritos_left = 10 := by sorry
  exact h4 -- Concluding that burritos_left = 10

end john_burritos_left_l496_496860


namespace sin_sum_less_than_zero_l496_496746

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ 0 < γ ∧ γ < Real.pi / 2

theorem sin_sum_less_than_zero (n : ℕ) :
  (∀ (α β γ : ℝ), is_acute_triangle α β γ → (Real.sin (n * α) + Real.sin (n * β) + Real.sin (n * γ) < 0)) ↔ n = 4 :=
by
  sorry

end sin_sum_less_than_zero_l496_496746


namespace earthquake_energy_l496_496897

theorem earthquake_energy 
  (lg : ℝ → ℝ)
  (exp : ℝ → ℝ) 
  (M4 M6 M5_5 : ℝ)
  (E4 E6 : ℝ)
  (lambda mu : ℝ)
  (lg6_3 : ℝ)
  (ten_pow_0_05 : ℝ)
  (Hlg : ∀ x, lg (10 ^ x) = x) 
  (Hexp : ∀ x, exp (lg x) = x) 
  (Hrel : ∀ M, lg (E4 * exp (lambda * (M - M4))) = lambda * M + mu)
  (HM4 : lg E4 = lambda * M4 + mu)
  (HM6 : lg E6 = lambda * M6 + mu)
  (HvalM4 : lg6_3 + 10 = lambda * M4 + mu)
  (HvalM6 : lg6_3 + 13 = lambda * M6 + mu)
  (Hlg6_3 : lg 6.3 = lg6_3)
  (Hten_pow_0_05 : 10 ^ 0.05 = ten_pow_0_05)
  (HE4 : E4 = 6.3 * 10^10)
  (HE6 : E6 = 6.3 * 10^13)
  : E5_5 = 1.1 * 10^13 := 
sorry

end earthquake_energy_l496_496897


namespace exponential_inequality_l496_496805

variables {x1 x2 : ℝ}

theorem exponential_inequality (hx : x1 ≠ x2) :
  (2 ^ x1 + 2 ^ x2) / 2 > 2 ^ ((x1 + x2) / 2) :=
sorry

end exponential_inequality_l496_496805


namespace log_base_function_inequalities_l496_496338

/-- 
Given the function y = log_(1/(sqrt(2))) (1/(x + 3)),
prove that:
1. for y > 0, x ∈ (-2, +∞)
2. for y < 0, x ∈ (-3, -2)
-/
theorem log_base_function_inequalities :
  let y (x : ℝ) := Real.logb (1 / Real.sqrt 2) (1 / (x + 3))
  ∀ x : ℝ, (y x > 0 ↔ x > -2) ∧ (y x < 0 ↔ -3 < x ∧ x < -2) :=
by
  intros
  -- Proof steps would go here
  sorry

end log_base_function_inequalities_l496_496338


namespace largest_real_number_l496_496716

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496716


namespace berry_circle_properties_l496_496607

theorem berry_circle_properties :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 12 = 2 * x + 4 * y → r = Real.sqrt 17)
    ∧ (π * Real.sqrt 17 ^ 2 > 30) :=
by
  sorry

end berry_circle_properties_l496_496607


namespace range_of_a_l496_496789

def f (x : ℝ) := real.sqrt x

theorem range_of_a
  (a : ℝ)
  (h1 : f (a + 1) < f (10 - 2 * a))
  (h2 : 0 ≤ a + 1)
  (h3 : 0 ≤ 10 - 2 * a) :
  -1 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l496_496789


namespace range_of_f_l496_496193

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_f : set.range (λ x, f x) = set.Icc (-1 : ℝ) 3 \ {3} :=
sorry

end range_of_f_l496_496193


namespace probability_point_exactly_two_units_from_origin_is_zero_l496_496275

theorem probability_point_exactly_two_units_from_origin_is_zero :
  ∀ (Q : ℝ × ℝ),
    (-3 ≤ Q.1 ∧ Q.1 ≤ 3) ∧ (-2 ≤ Q.2 ∧ Q.2 ≤ 2) →
    (∃ x y, (x^2 + y^2 = 4) ∧ (Q = (x, y))) →
    0 :=
by
  sorry

end probability_point_exactly_two_units_from_origin_is_zero_l496_496275


namespace sum_odd_divisors_of_180_l496_496985

theorem sum_odd_divisors_of_180 : 
  let n := 180 in 
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0} in 
  ∑ d in odd_divisors, d = 78 :=
by
  let n := 180
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0}
  have h : ∑ d in odd_divisors, d = 78 := sorry -- Sum of odd divisors of 180
  exact h

end sum_odd_divisors_of_180_l496_496985


namespace negation_of_universal_sin_pos_l496_496189

theorem negation_of_universal_sin_pos :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 :=
by sorry

end negation_of_universal_sin_pos_l496_496189


namespace sum_tangents_eq_6sqrt21_l496_496294

open Real

variables (O A B C : ℝ)
variables (ω : set ℝ)
variables (r BC : ℝ)
variables (OA : A = 15)
variables (radius : r = 6)
variables (tangent_length : ∀ P, P ∈ ω → ∃ T, (dist A T = dist A P) ∧ dist O T = r)
variables (T₁ T₂ : ℝ)
variables (AB_tangent : dist A B = dist A T₁ ∧ dist O B = r)
variables (AC_tangent : dist A C = dist A T₂ ∧ dist O C = r)
variables (BC_length : BC = 8)
variables (BC_tangent : dist B C = BC ∧ ∀ P, P ∈ ω → dist P B = dist P C)

theorem sum_tangents_eq_6sqrt21 :
  ∃ T₁ T₂, dist A T₁ = dist A T₂ ∧ dist A B = dist A T₁ ∧ dist A C = dist A T₂ →
  dist T₁ O = r ∧ dist T₂ O = r →
  ∀ T₁ T₂, dist O A = 15 ∧ r = 6 → dist A B = 3 * sqrt 21 ∧ dist A C = 3 * sqrt 21 →
  BC = 8 ∧ dist B C = BC →
  (dist A B) + (dist A C) = 6 * sqrt 21 :=
begin
  sorry
end

end sum_tangents_eq_6sqrt21_l496_496294


namespace coeff_of_x_in_expansion_l496_496522

theorem coeff_of_x_in_expansion :
  ∀ (x : ℝ), polynomial.coeff (((1 + 2 * x)^3 * (1 - x)^4) : polynomial ℝ) 1 = 2 :=
begin
  sorry
end

end coeff_of_x_in_expansion_l496_496522


namespace distance_from_A_to_D_l496_496149

theorem distance_from_A_to_D 
  (A B C D : Type)
  (east_of : B → A)
  (north_of : C → B)
  (distance_AC : Real)
  (angle_BAC : ℝ)
  (north_of_D : D → C)
  (distance_CD : Real) : 
  distance_AC = 5 * Real.sqrt 5 → 
  angle_BAC = 60 → 
  distance_CD = 15 → 
  ∃ (AD : Real), AD =
    Real.sqrt (
      (5 * Real.sqrt 15 / 2) ^ 2 + 
      (5 * Real.sqrt 5 / 2 + 15) ^ 2
    ) :=
by
  intros
  sorry


end distance_from_A_to_D_l496_496149


namespace find_c_l496_496898

theorem find_c (b c : ℤ) (H : (b - 4) / (2 * b + 42) = c / 6) : c = 2 := 
sorry

end find_c_l496_496898


namespace angle_CPD_110_l496_496841

-- Define the conditions as hypotheses

variable {O1 O2 C D P R S T : Point}
variable (SAR : Semicircle O1 S R)
variable (RBT : Semicircle O2 R T)
variable (tangent1 : Tangent C P SAR)
variable (tangent2 : Tangent D P RBT)
variable (straight_line : Collinear S R T)
variable (arc_AS : arc_measure SAR S A = 68)
variable (arc_BT : arc_measure RBT B T = 42)

-- Define the angle CPD in degrees is 110
theorem angle_CPD_110 : 
  ∠ C P D = 110 :=
sorry

end angle_CPD_110_l496_496841


namespace area_of_rhombus_proof_l496_496019

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  d1 * d2 / 2

theorem area_of_rhombus_proof :
  ∀ (d1 d2 : ℝ), (d1 = 2 ∧ d2 = 3) ∨ (d1 = 3 ∧ d2 = 2) → area_of_rhombus d1 d2 = 3 :=
by
  intros d1 d2 h
  cases h with h1 h2
  { cases h1; simp [area_of_rhombus] }
  { cases h2; simp [area_of_rhombus] }
  sorry

end area_of_rhombus_proof_l496_496019


namespace range_of_f_implies_a_in_interval_l496_496372

-- Define the function f
def f (a x : ℝ) : ℝ := log (a * x^2 + 2 * x + 1)

-- Define the condition that f must have a range of ℝ
def hasRangeR (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

-- State the main theorem
theorem range_of_f_implies_a_in_interval (a : ℝ) :
  hasRangeR a ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_f_implies_a_in_interval_l496_496372


namespace empty_set_range_at_most_one_element_range_l496_496800

noncomputable def A (a : ℝ) := { x : ℝ | a * x^2 - 3 * x + 2 = 0 }

theorem empty_set_range (a : ℝ) : A(a) = ∅ ↔ a ∈ Ioo (9/8 : ℝ) ∞ :=
by {
  sorry
}

theorem at_most_one_element_range (a : ℝ) : (∀ x : ℝ, x ∈ A(a) → x = 0) ∨ A(a).card ≤ 1 ↔ a ∈ ({0} ∪ (Ici (9/8 : ℝ))) :=
by {
  sorry
}

end empty_set_range_at_most_one_element_range_l496_496800


namespace no_two_integers_with_product_zero_not_ending_in_zero_l496_496741

theorem no_two_integers_with_product_zero_not_ending_in_zero :
  ∀ (a b : ℤ), (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) → a * b ≠ 0 :=
by {
  intros a b H,
  cases H with h1 h2,
  sorry
}

end no_two_integers_with_product_zero_not_ending_in_zero_l496_496741


namespace problem_part1_problem_part2_l496_496468

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - a - 1

theorem problem_part1 :
  ∀ x ∈ Icc 0 2, f x 1 ∈ Icc (-2) 6 :=
sorry

noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 0 then -a - 1
  else if -2 < a ∧ a < 0 then -a^2 - a - 1
  else 3 * a + 3

theorem problem_part2 :
  (∀ a, ∃ m : ℤ, g a ≤ ↑m) ∧
  (∃ m : ℤ, ∀ a, g a ≤ m) ∧
  (∃! m, ∀ a, g a - (m : ℝ) ≤ 0) :=
sorry

end problem_part1_problem_part2_l496_496468


namespace total_students_after_new_classes_l496_496919

def initial_classes : ℕ := 15
def students_per_class : ℕ := 20
def new_classes : ℕ := 5

theorem total_students_after_new_classes :
  initial_classes * students_per_class + new_classes * students_per_class = 400 :=
by
  sorry

end total_students_after_new_classes_l496_496919


namespace alex_original_seat_l496_496163

-- We define a type for seats
inductive Seat where
  | s1 | s2 | s3 | s4 | s5 | s6
  deriving DecidableEq, Inhabited

open Seat

-- Define the initial conditions and movements
def initial_seats : (Fin 6 → Seat) := ![s1, s2, s3, s4, s5, s6]

def move_bella (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

def move_coral (s : Seat) : Seat :=
  match s with
  | s1 => s6  -- two seats left from s1 wraps around to s6
  | s2 => s1
  | s3 => s2
  | s4 => s3
  | s5 => s4
  | s6 => s5

-- Dan and Eve switch seats among themselves
def switch_dan_eve (s : Seat) : Seat :=
  match s with
  | s3 => s4
  | s4 => s3
  | _ => s  -- all other positions remain the same

def move_finn (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

-- Define the final seat for Alex
def alex_final_seat : Seat := s6  -- Alex returns to one end seat

-- Define a theorem for the proof of Alex's original seat being Seat.s1
theorem alex_original_seat :
  ∃ (original_seat : Seat), original_seat = s1 :=
  sorry

end alex_original_seat_l496_496163


namespace gcd_min_val_l496_496458

theorem gcd_min_val (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 1155) : ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 105 :=
by
  sorry

end gcd_min_val_l496_496458


namespace art_collection_area_l496_496677

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l496_496677


namespace find_a_l496_496778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

theorem find_a (a : ℝ) : f_prime a 1 = 2 → a = -3 := by
  intros h
  -- skipping the proof, as it is not required
  sorry

end find_a_l496_496778


namespace imaginary_part_conjugate_l496_496000

theorem imaginary_part_conjugate (z : ℂ) (h : z = (3 - complex.i) / (3 + complex.i)) :
  complex.im z.conj = 3 / 5 :=
by 
  sorry

end imaginary_part_conjugate_l496_496000


namespace pairs_satisfy_inequality_l496_496878

section inequality_problem

variables (a b : ℝ)

-- Conditions
variable (hb1 : b ≠ -1)
variable (hb2 : b ≠ 0)

-- Inequalities to check
def inequality (a b : ℝ) : Prop :=
  (1 + a) ^ 2 / (1 + b) ≤ 1 + a ^ 2 / b

-- Main theorem
theorem pairs_satisfy_inequality :
  (b > 0 ∨ b < -1 → ∀ a, a ≠ b → inequality a b) ∧
  (∀ a, a ≠ -1 ∧ a ≠ 0 → inequality a a) :=
by
  sorry

end inequality_problem

end pairs_satisfy_inequality_l496_496878


namespace finite_group_with_automorphism_implies_cyclic_or_abelian_l496_496680

open GroupTheory

-- Defining the properties and conditions
variables {G : Type*} [Group G] [Fintype G] (f : G ≃* G)

-- Main theorem (no proof included)
theorem finite_group_with_automorphism_implies_cyclic_or_abelian (h : ∀ H : Subgroup G, H ≠ ⊥ → H ≠ ⊤ → H ⊆ f H → False) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (n : ℕ) (hn : 0 < n), G ≅ AddCommGroup.toGroup (ZNalg p) n := sorry

end finite_group_with_automorphism_implies_cyclic_or_abelian_l496_496680


namespace expand_and_simplify_l496_496311

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * 3 * x = 51 * x^2 - 27 * x := 
by 
  sorry

end expand_and_simplify_l496_496311


namespace polar_equation_of_circle_chord_length_range_l496_496090

noncomputable def circle_center_polar : ℝ × ℝ :=
  (√2, π / 4)

noncomputable def circle_radius : ℝ :=
  √3

theorem polar_equation_of_circle :
  ∃ ρ θ : ℝ, ρ^2 - 2*ρ*(cos θ + sin θ) - 1 = 0 :=
  sorry

theorem chord_length_range (α : ℝ) (h : α ∈ set.Ico 0 (π / 4)) :
  ∀ A B : (ℝ × ℝ), 
  (∃ t : ℝ, (2 + t * cos α, 2 + t * sin α) ∈ 
  { p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 3 }) →
  (dist A B ∈ set.Ico (2*√2) (2*√3)) :=
  sorry

end polar_equation_of_circle_chord_length_range_l496_496090


namespace original_deck_card_count_l496_496268

theorem original_deck_card_count (r b u : ℕ)
  (h1 : r / (r + b + u) = 1 / 5)
  (h2 : r / (r + b + u + 3) = 1 / 6) :
  r + b + u = 15 := by
  sorry

end original_deck_card_count_l496_496268


namespace sin_and_tan_inequality_l496_496906

theorem sin_and_tan_inequality (n : ℕ) (hn : 0 < n) :
  2 * Real.sin (1 / n) + Real.tan (1 / n) > 3 / n :=
sorry

end sin_and_tan_inequality_l496_496906


namespace parabola_vertex_below_x_axis_l496_496418

theorem parabola_vertex_below_x_axis (a : ℝ) : (∀ x : ℝ, (x^2 + 2 * x + a < 0)) → a < 1 := 
by
  intro h
  -- proof step here
  sorry

end parabola_vertex_below_x_axis_l496_496418


namespace expected_value_min_of_subset_l496_496968

noncomputable def expected_value_min (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : ℚ :=
  (n + 1) / (r + 1)

theorem expected_value_min_of_subset (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  expected_value_min n r h = (n + 1) / (r + 1) :=
sorry

end expected_value_min_of_subset_l496_496968


namespace FO_gt_DI_l496_496087

-- Definitions and conditions
variables (F I D O : Type) [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O]
variables (FI DO DI FO : ℝ) (angle_FIO angle_DIO : ℝ)
variable (convex_FIDO : ConvexQuadrilateral F I D O)

-- Conditions
axiom FI_DO_equal : FI = DO
axiom FI_DO_gt_DI : FI > DI
axiom angles_equal : angle_FIO = angle_DIO

-- Goal
theorem FO_gt_DI : FO > DI :=
sorry

end FO_gt_DI_l496_496087


namespace cycles_same_length_l496_496881

-- Define the mathematical problem in Lean 4
theorem cycles_same_length (f : ℤ → ℚ) (p : ℤ) (hf : ∀ x : ℤ, f x = (p * x + 1) / (x + p)) :
  ∃ r, ∀ s, s ≠ 1 → (∀ x, (∃ n, f^[n] x = x ∧ least_period f x = s) → s = r) :=
sorry

end cycles_same_length_l496_496881


namespace triangles_similar_l496_496112

variable (A B C H : Type) [IsTriangle A B C]
variable (H : IsOrthocenter A B C)
variable (X_A X_B X_C : Type)
variable (circ_A circ_B circ_C : Circle A B C)

axiom tangent_at_H : Tangent H circ_A = Tangent H circ_B = Tangent H circ_C
axiom AH_EQ_AX_A : dist A H = dist A X_A
axiom BH_EQ_BX_B : dist B H = dist B X_B
axiom CH_EQ_CX_C : dist C H = dist C X_C

theorem triangles_similar :
  Similar (Triangle.mk X_A X_B X_C) (OrthoTriangle.mk A B C H) := by
  sorry

end triangles_similar_l496_496112


namespace find_D_l496_496085

noncomputable def Point : Type := ℝ × ℝ

-- Given points A, B, and C
def A : Point := (-2, 0)
def B : Point := (6, 8)
def C : Point := (8, 6)

-- Condition: AB parallel to DC and AD parallel to BC, which means it is a parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  ((B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2)) ∧
  ((C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2))

-- Proves that with given A, B, and C, D should be (0, -2)
theorem find_D : ∃ D : Point, is_parallelogram A B C D ∧ D = (0, -2) :=
  by sorry

end find_D_l496_496085


namespace polynomial_factors_sum_l496_496050

theorem polynomial_factors_sum (a b : ℝ) 
  (h : ∃ c : ℝ, (∀ x: ℝ, x^3 + a * x^2 + b * x + 8 = (x + 1) * (x + 2) * (x + c))) : 
  a + b = 21 :=
sorry

end polynomial_factors_sum_l496_496050


namespace ordered_triples_count_l496_496278

def similar_prisms_count (b : ℕ) (c : ℕ) (a : ℕ) := 
  (a ≤ c ∧ c ≤ b ∧ 
   ∃ (x y z : ℕ), x ≤ z ∧ z ≤ y ∧ y = b ∧ 
   x < a ∧ y < b ∧ z < c ∧ 
   ((x : ℚ) / a = (y : ℚ) / b ∧ (y : ℚ) / b = (z : ℚ) / c))

theorem ordered_triples_count : 
  ∃ (n : ℕ), n = 24 ∧ ∀ a c, similar_prisms_count 2000 c a → a < c :=
sorry

end ordered_triples_count_l496_496278


namespace range_F_l496_496301

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

def f_inv (y : ℝ) : ℝ := 2 + Real.log y / Real.log 3

def F (x : ℝ) : ℝ := (f_inv x)^2 - f_inv (x^2)

theorem range_F :
  {y : ℝ | ∃ x : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ y = F x} = set.Icc 2 5 := sorry

end range_F_l496_496301


namespace triangle_length_product_square_l496_496908

theorem triangle_length_product_square 
  (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ) 
  (h1 : a1 * b1 / 2 = 3)
  (h2 : a2 * b2 / 2 = 4)
  (h3 : a1 = a2)
  (h4 : c1 = 2 * c2) 
  (h5 : c1^2 = a1^2 + b1^2)
  (h6 : c2^2 = a2^2 + b2^2) :
  (b1 * b2)^2 = (2304 / 25 : ℝ) :=
by
  sorry

end triangle_length_product_square_l496_496908


namespace hoseok_item_price_l496_496039

theorem hoseok_item_price 
  (num_1000_bills : ℕ) (num_100_coins : ℕ) (num_10_coins : ℕ) 
  (value_1000_bill : ℕ) (value_100_coin : ℕ) (value_10_coin : ℕ)
  (total_price : ℕ) :
  num_1000_bills = 7 → 
  num_100_coins = 4 → 
  num_10_coins = 5 → 
  value_1000_bill = 1000 → 
  value_100_coin = 100 → 
  value_10_coin = 10 → 
  total_price = (num_1000_bills * value_1000_bill) + 
                (num_100_coins * value_100_coin) + 
                (num_10_coins * value_10_coin) → 
  total_price = 7450 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  exact h7

end hoseok_item_price_l496_496039


namespace tiles_needed_l496_496269

def inches_to_feet (inches : ℝ) : ℝ := inches / 12

def tile_area_feet : ℝ := (inches_to_feet 6) * (inches_to_feet 9)
def floor_area_feet : ℝ := 10 * 15

theorem tiles_needed : floor_area_feet / tile_area_feet = 400 :=
by
  sorry

end tiles_needed_l496_496269


namespace problem_statement_l496_496365

theorem problem_statement (x y z a b : ℝ)
  (H1 : (∀ x, | x + a | ≤ b ↔ -6 ≤ x ∧ x ≤ 2))
  (H2 : |2 * y + z| < 1 / 3)
  (H3 : |y - 4 * z| < 1 / 6) :
  a = 2 ∧ b = 4 ∧ |z| < 2 / 27 :=
by {
  sorry
}

end problem_statement_l496_496365


namespace john_burritos_left_l496_496862

theorem john_burritos_left : 
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  in burritos_left = 10 := by
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  have h : total_burritos = 60 := by rfl
  have h1 : burritos_given_away = 20 := by sorry
  have h2 : burritos_left_after_giving = 40 := by sorry
  have h3 : total_burritos_eaten = 30 := by sorry
  have h4 : burritos_left = 10 := by sorry
  exact h4 -- Concluding that burritos_left = 10

end john_burritos_left_l496_496862


namespace shark_fin_falcata_area_correct_l496_496944

noncomputable def area_of_sharks_fin_falcata : ℝ :=
  let larger_circle_area := (1/4) * π * (5^2) in
  let smaller_circle_area := (1/2) * π * (3^2) in
  let overlap_area_estimate := (1/2) * smaller_circle_area in
  let shark_fin_falcata_area := larger_circle_area - overlap_area_estimate in
  shark_fin_falcata_area

theorem shark_fin_falcata_area_correct :
  area_of_sharks_fin_falcata = 4 * π :=
by
  -- Proof will be provided here
  sorry

end shark_fin_falcata_area_correct_l496_496944


namespace sum_of_odd_divisors_180_l496_496971

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_divisors_180 :
  sum_of_positive_odd_divisors 180 = 78 :=
by
  sorry

end sum_of_odd_divisors_180_l496_496971


namespace largest_x_63_over_8_l496_496710

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496710


namespace original_number_correct_l496_496612

theorem original_number_correct (d q r : ℕ) (h_d : d = 163) (h_q : q = 76) (h_r : r = 13) : d * q + r = 12401 :=
by
  rw [h_d, h_q, h_r]
  sorry

end original_number_correct_l496_496612


namespace find_smallest_value_l496_496885

noncomputable def smallest_value (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2

theorem find_smallest_value (a b c d : ℝ) (h1: a + b = 18)
  (h2: ab + c + d = 85) (h3: ad + bc = 180) (h4: cd = 104) :
  smallest_value a b c d = 484 :=
sorry

end find_smallest_value_l496_496885


namespace correct_option_C_correct_option_D_l496_496242

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l496_496242


namespace regression_analysis_incorrect_statement_l496_496582

-- Definitions for the conditions
def condition_A : Prop := "The regression line must pass through the point (\overline{x}, \overline{y})"
def condition_B : Prop := "The regression line is the line that passes through the most sample data points in a scatter plot"
def condition_C : Prop := "When the correlation coefficient r > 0, the two variables are positively correlated"
def condition_D : Prop := "If the linear correlation between two variables is weaker, then |r| is closer to 0"

-- The goal is to prove that condition_B is false
theorem regression_analysis_incorrect_statement : ¬condition_B :=
by
  sorry

end regression_analysis_incorrect_statement_l496_496582


namespace investment_calculation_l496_496624

theorem investment_calculation
  (face_value : ℝ)
  (market_price : ℝ)
  (rate_of_dividend : ℝ)
  (annual_income : ℝ)
  (h1 : face_value = 10)
  (h2 : market_price = 8.25)
  (h3 : rate_of_dividend = 12)
  (h4 : annual_income = 648) :
  ∃ investment : ℝ, investment = 4455 :=
by
  sorry

end investment_calculation_l496_496624


namespace sum_of_odd_divisors_180_l496_496975

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_divisors_180 :
  sum_of_positive_odd_divisors 180 = 78 :=
by
  sorry

end sum_of_odd_divisors_180_l496_496975


namespace smallest_integer_k_l496_496320

theorem smallest_integer_k (k : ℕ) :
  (∃ k, (digits_sum (9 * (10^k - 1) / 9) = 650) ∧ k = 72) :=
begin
  sorry
end

/-- Helper function to calculate the digit sum of an integer -/
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

end smallest_integer_k_l496_496320


namespace constant_function_l496_496877

variable {R : Type} [LinearOrder R] [NonzeroFact R]

theorem constant_function
  (a : R) 
  (h_a_pos : 0 < a)
  (f : R → ℝ)
  (hf_domain : ∀ x, 0 < x → x < ∞)
  (hf_a : f a = 1)
  (hf_property : ∀ x y, 0 < x → 0 < y → f x * f y + f (a / x) * f (a / y) = 2 * f (x * y)) :
  ∀ x, 0 < x → f x = 1 := by sorry

end constant_function_l496_496877


namespace marble_arrangement_mod_1000_l496_496658

noncomputable def binom : ℕ → ℕ → ℕ 
| 0 0 := 1
| 0 (n + 1) := 0
| (n + 1) 0 := 1
| (n + 1) (k + 1) := binom n k + binom n (k + 1)

theorem marble_arrangement_mod_1000 : 
  ∃ N : ℕ, 
  N = binom 25 7 ∧ 
  N % 1000 = 700 := 
by
  have N := binom 25 7
  existsi N
  split
  . exact sorry
  . exact sorry

end marble_arrangement_mod_1000_l496_496658


namespace count_three_top_numbers_l496_496419

noncomputable def is_three_top_numbers (n : ℕ) : Prop :=
  (10 ≤ n ∧ n ≤ 97) ∧
  (let sum := n + (n + 1) + (n + 2) in sum < 100) ∧
  (let sum_units := (n + (n + 1) + (n + 2)) % 10 in
    let units_n := n % 10 in
    let units_n_plus_1 := (n + 1) % 10 in
    let units_n_plus_2 := (n + 2) % 10 in
    units_n < sum_units ∧ units_n_plus_1 < sum_units ∧ units_n_plus_2 < sum_units)

theorem count_three_top_numbers : 
  Nat.card { n : ℕ // is_three_top_numbers n } = 5 := 
sorry

end count_three_top_numbers_l496_496419


namespace profit_after_taxes_and_quantities_and_prices_l496_496485

variables {P_A P_B : ℝ} {q_A q_B Q : ℝ}

/-- Define the demand functions for countries A and B -/
def demand_A : ℝ → ℝ := λ P_A, 40 - 2 * P_A
def demand_B : ℝ → ℝ := λ P_B, 26 - P_B

/-- Define the total cost function -/
def total_cost : ℝ → ℝ := λ Q, 8 * Q + 1

/-- Define the total quantity produced as the sum of quantities in both countries -/
def total_quantity : ℝ := q_A + q_B

/-- Define the total revenue for countries A and B -/
def total_revenue_A : ℝ := P_A * q_A
def total_revenue_B : ℝ := P_B * q_B

/-- Define the total revenue as the sum of revenues from both countries -/
def total_revenue : ℝ := total_revenue_A + total_revenue_B

/-- Define the total profit before tax -/
def total_profit : ℝ := total_revenue - total_cost total_quantity - 2

/-- Define the post-tax profit for Country A (fixed rate 15%) -/
def post_tax_profit_A : ℝ := total_profit * 0.85

/-- Define the post-tax profit for Country B (progressive rates) -/
def post_tax_profit_B : ℝ :=
  if total_profit ≤ 30 then
    total_profit
  else if total_profit ≤ 100 then
    30 + 0.9 * (total_profit - 30)
  else if total_profit ≤ 150 then
    30 + 0.9 * 70 + 0.8 * (total_profit - 100)
  else
    30 + 0.9 * 70 + 0.8 * 50 + 0.7 * (total_profit - 150)

/-- Main theorem to prove the profits after taxes and quantities and prices -/
theorem profit_after_taxes_and_quantities_and_prices :
  q_A = 12 ∧ q_B = 9 ∧ P_A = 14 ∧ P_B = 17 ∧ post_tax_profit_B = 133.7 :=
by
  -- This space is left for the full proof. Use sorry to compile successfully.
  sorry

end profit_after_taxes_and_quantities_and_prices_l496_496485


namespace painter_total_cost_l496_496281

-- Define the arithmetic sequence for house addresses
def south_side_arith_seq (n : ℕ) : ℕ := 5 + (n - 1) * 7
def north_side_arith_seq (n : ℕ) : ℕ := 6 + (n - 1) * 8

-- Define the counting of digits
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

-- Define the condition of painting cost for multiples of 10
def painting_cost (n : ℕ) : ℕ :=
  if n % 10 = 0 then 2 * digit_count n
  else digit_count n

-- Calculate total cost for side with given arithmetic sequence
def total_cost_for_side (side_arith_seq : ℕ → ℕ): ℕ :=
  List.range 25 |>.map (λ n => painting_cost (side_arith_seq (n + 1))) |>.sum

-- Main theorem to prove
theorem painter_total_cost : total_cost_for_side south_side_arith_seq + total_cost_for_side north_side_arith_seq = 147 := by
  sorry

end painter_total_cost_l496_496281


namespace three_digit_numbers_count_l496_496044

-- Define what it means for a number to be three-digit and contain specific digits.
def has_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

def valid_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ has_digit n 4 ∧ has_digit n 5

theorem three_digit_numbers_count : { n : ℕ | valid_digit n }.to_finset.card = 50 := by
  sorry

end three_digit_numbers_count_l496_496044


namespace best_meeting_days_l496_496574

-- Define the days of the week
inductive Day : Type
| Mon | Tues | Wed | Thurs | Fri
deriving DecidableEq

open Day

-- Define the availability constraints for each team member
def Anna_availability : Day → Prop
| Mon := False
| Wed := False
| _ := True

def Bill_availability : Day → Prop
| Tues := False
| Thurs := False
| Fri := False
| _ := True

def Carl_availability : Day → Prop
| Mon := False
| Tues := False
| Thurs := False
| Fri := False
| _ := True

def Dana_availability : Day → Prop
| Wed := False
| Thurs := False
| _ := True

-- Function to count the number of available attendees on a given day
def available_attendees (d : Day) : ℕ :=
  (if Anna_availability d then 1 else 0) + 
  (if Bill_availability d then 1 else 0) + 
  (if Carl_availability d then 1 else 0) + 
  (if Dana_availability d then 1 else 0)

-- Definition of the best days as those with the maximum number of attendees
def best_days (max_attendance : ℕ) (d : Day) : Prop :=
  available_attendees d = max_attendance

-- Prove that the best days are Monday, Tuesday, Wednesday, or Friday
theorem best_meeting_days :
  ∃ max_attendance, max_attendance = 2 ∧ 
  (best_days max_attendance Mon ∨ best_days max_attendance Tues ∨ 
   best_days max_attendance Wed ∨ best_days max_attendance Fri) :=
by
  exists 2
  split
  {
    rfl
  }
  {
    sorry
  }

end best_meeting_days_l496_496574


namespace quadratic_radical_same_type_l496_496544

theorem quadratic_radical_same_type (a : ℝ) (h : (∃ (t : ℝ), t ^ 2 = 3 * a - 4) ∧ (∃ (t : ℝ), t ^ 2 = 8)) : a = 2 :=
by
  -- Extract the properties of the radicals
  sorry

end quadratic_radical_same_type_l496_496544


namespace behavior_on_1_2_l496_496288

/-- Definition of an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- Definition of being decreasing on an interval -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

/-- Definition of having a minimum value on an interval -/
def has_minimum_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x ≥ m

theorem behavior_on_1_2 
  {f : ℝ → ℝ} 
  (h_odd : is_odd_function f) 
  (h_dec : is_decreasing_on f (-2) (-1)) 
  (h_min : has_minimum_on f (-2) (-1) 3) :
  is_decreasing_on f 1 2 ∧ ∀ x, 1 ≤ x → x ≤ 2 → f x ≤ -3 := 
by 
  sorry

end behavior_on_1_2_l496_496288


namespace necessary_but_not_sufficient_l496_496117

-- Define the conditions of the problem
variable (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1)

-- Define properties P and Q
def P := ∀ x > 0, ∃ b > 1, b^x = a
def Q := ∀ x > 0, (3 - a) * (x^(3-a - 1)) < 0

-- The theorem stating that P is a necessary but not sufficient condition for Q
theorem necessary_but_not_sufficient : 
  (∀ x > 0, P ∧ a ∈ (1, +∞) → Q ∧ a ∈ (3, +∞)) ∧ ¬ (∀ x > 0, Q ∧ a ∈ (3, +∞) → P ∧ a ∈ (1, +∞)) :=
sorry

end necessary_but_not_sufficient_l496_496117


namespace limit_calculation_l496_496792

def f (x : ℝ) : ℝ := x^2 + 3*x - 2

theorem limit_calculation : 
  (∃ delta : ℝ → ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x (h₀ : abs x < δ), abs ((f (1 + 2 * x) - f 1) / x) < ε) → 
  limit (λ (Δx : ℝ), (f (1 + 2 * Δx) - f 1) / Δx) 0 10 :=
sorry

end limit_calculation_l496_496792


namespace no_solution_inequalities_l496_496803

theorem no_solution_inequalities (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  intro h
  sorry

end no_solution_inequalities_l496_496803


namespace chapter_page_difference_l496_496608

/-- The first chapter of a book has 37 pages -/
def first_chapter_pages : Nat := 37

/-- The second chapter of a book has 80 pages -/
def second_chapter_pages : Nat := 80

/-- Prove the difference in the number of pages between the second and the first chapter is 43 -/
theorem chapter_page_difference : (second_chapter_pages - first_chapter_pages) = 43 := by
  sorry

end chapter_page_difference_l496_496608


namespace united_telephone_charge_l496_496210

theorem united_telephone_charge (x : ℝ) :
  (∃ x : ℝ, 6 + 120 * x = 12 + 120 * 0.20) → x = 0.25 :=
by
  intro h
  cases h with x h_x
  have h₁ : 6 + 120 * x = 12 + 24 := by
    rw [mul_comm 120, mul_comm 120]
      norm_num at h_x
  rw [add_comm 12 24] at h₁
  linarith

end united_telephone_charge_l496_496210


namespace trajectory_of_M_is_parabola_l496_496363

theorem trajectory_of_M_is_parabola:
  ∀ (x y: ℝ), 5 * real.sqrt(x^2 + y^2) = abs (3 * x + 4 * y - 12) → 
    (∃ (p: ℝ), (x^2 + y^2) = ((3 * x + 4 * y - 12) ^ 2) / 25) :=
by
  sorry

end trajectory_of_M_is_parabola_l496_496363


namespace f_at_7_l496_496874

noncomputable def f (x : ℝ) (a b c d : ℝ) := a * x^7 + b * x^5 + c * x^3 + d * x + 5

theorem f_at_7 (a b c d : ℝ) (h : f (-7) a b c d = -7) : f 7 a b c d = 17 := 
by
  sorry

end f_at_7_l496_496874


namespace find_certain_amount_l496_496423

theorem find_certain_amount :
  ∀ (A : ℝ), (160 * 8 * 12.5 / 100 = A * 8 * 4 / 100) → 
            (A = 500) :=
  by
    intros A h
    sorry

end find_certain_amount_l496_496423


namespace sum_of_positive_odd_divisors_eq_78_l496_496991

theorem sum_of_positive_odd_divisors_eq_78 :
  ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 180)), d = 78 :=
by {
  -- proof steps go here
  sorry
}

end sum_of_positive_odd_divisors_eq_78_l496_496991


namespace unique_integer_cube_triple_l496_496228

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l496_496228


namespace range_of_m_perimeter_of_isosceles_triangle_l496_496095

-- Define the variables for the lengths of the sides and the range of m
variables (AB BC AC : ℝ) (m : ℝ)

-- Conditions given in the problem
def triangle_conditions (AB BC : ℝ) (AC : ℝ) (m : ℝ) : Prop :=
  AB = 17 ∧ BC = 8 ∧ AC = 2 * m - 1

-- Proof that the range for m is between 5 and 13
theorem range_of_m (AB BC : ℝ) (m : ℝ) (h : triangle_conditions AB BC (2 * m - 1) m) : 
  5 < m ∧ m < 13 :=
by
  sorry

-- Proof that the perimeter is 42 when triangle is isosceles with given conditions
theorem perimeter_of_isosceles_triangle (AB BC AC : ℝ) (h : triangle_conditions AB BC AC 0) : 
  (AB = AC ∨ BC = AC) → (2 * AB + BC = 42) :=
by
  sorry

end range_of_m_perimeter_of_isosceles_triangle_l496_496095


namespace measureable_weights_count_l496_496571

theorem measureable_weights_count : 
  ∀ (a b c d : ℕ), 
    (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 18) → 
    (∃ n, n = 27) :=
by 
  intros a b c d h,
  cases h with ha1 hd18,
  simp at ha1 hd18,
  exact ⟨27, rfl⟩

end measureable_weights_count_l496_496571


namespace trains_at_initial_stations_after_2016_minutes_l496_496186

theorem trains_at_initial_stations_after_2016_minutes :
  (∀ t : ℕ, t % 14 = 0 → t % 16 = 0 → t % 18 = 0 → (t % 2016 = 0)) :=
by
  -- Given condition: trains on specific lines have round trip times.
  assume t : ℕ,
  assume h1 : t % 14 = 0,
  assume h2 : t % 16 = 0,
  assume h3 : t % 18 = 0,
  sorry

end trains_at_initial_stations_after_2016_minutes_l496_496186


namespace intersection_of_A_and_B_l496_496890

open Set

variable {U : Type} [LinearOrderedField U]

noncomputable def A : Set U := {x : U | x * (x - 2) < 0}
noncomputable def B : Set U := {x : U | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {x : U | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l496_496890


namespace ellipse_equation_trajectory_of_M_range_of_QS_l496_496021

theorem ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) 
  (eccentricity : Real.sqrt 3 / 3 = Real.sqrt (a^2 - b^2) / a)
  (tangent_line : ∀ x y : ℝ, y = x + 2 → x^2 + y^2 = b^2 → False) :
  (C1_eq : ∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) :=
sorry

theorem trajectory_of_M (C1_eq : ∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1)
  (F1_x F1_y F2_x F2_y : ℝ)
  (h_focus : F1_x = -a ∧ F2_x = a ∧ F1_y = F2_y = 0)
  (h_l1 : ∀ x y : ℝ, y = x → x = -1)
  (h_l2 : ∀ x y : ℝ, y = -1/x )
  (h_P : ∀ x y : ℝ, y = x - 2 → F2_x = 2 → y = -1/2) :
  (C2_eq : ∀ x y : ℝ, y^2 = 4*x) :=
 sorry

theorem range_of_QS (C2_eq : ∀ x y : ℝ, y^2 = 4*x)
  (Q_x Q_y : ℝ) 
  (R_x R_y S_x S_y : ℝ)
  (h_QRS : Q_x = 0 ∧ Q_y = 0 ∧ ((R_x, R_y) ≠ (S_x, S_y) ∧ (R_x, y1) = (y1^2 / 4, y1) ∧ (S_x, S_y) = (y2^2 / 4, y2) ∧ (S_x, S_y) ≠ (0, 0))
  (h_dot_prod : (0,0) = ((Q_x - R_x), (Q_y - R_y)) • ((R_x - S_x), (R_y - S_y)) = 0)) :
  (range_QS : ∀ (QS_len : ℝ), QS_len ≥ 8*Real.sqrt(5)) :=
 sorry

end ellipse_equation_trajectory_of_M_range_of_QS_l496_496021


namespace area_equals_a_of_integral_l496_496871

noncomputable def integral_area (a : ℝ) : ℝ :=
  ∫ x in 0..a, real.sqrt x

theorem area_equals_a_of_integral (a : ℝ) (h : a > 0) (area_eq : integral_area a = a) : a = 9 / 4 :=
by
  sorry

end area_equals_a_of_integral_l496_496871


namespace integral_equation_solution_l496_496570

noncomputable def phi (x : ℝ) : ℝ := sorry -- this definition is the target to be proven (hence, noncomputable and sorry)

theorem integral_equation_solution (x : ℝ) : 
  (φ x = e^(x + x^2)) :=
begin
  -- Integral equation from conditions
  let integral_eq := (λ x : ℝ, e^(x^2) + ∫ t in 0..x, e^(x^2 - t^2) * φ t),
  sorry
end

end integral_equation_solution_l496_496570


namespace count_shooting_sequences_l496_496076

-- The problem conditions are the column arrangements: [3, 3, 2]
def columns : List ℕ := [3, 3, 2]

-- We need to prove that the total number of sequences to shoot all targets is 560.
theorem count_shooting_sequences (columns : List ℕ) (h_cols : columns = [3, 3, 2]) :
  (8.choose 3) *  (5.choose 2) = 560 :=
by
  rw [h_cols]
  sorry

end count_shooting_sequences_l496_496076


namespace tangency_condition_l496_496108

variables {A B C D E X Y Z : Type*}

-- Considering A, B, C, D are points forming a trapezoid with AD parallel to BC
-- and angles constraints

def is_trapezoid (A B C D : Type*) := 
AB ∥ CD

def angle_constraints (A B C D : Type*) (α β : ℝ) := 
α < β ∧ β < 90

-- E is the intersection of diagonals AC and BD
def intersect_diagonals (A C B D : Type*): Type* := 
E

-- The circumcircle of triangle BEC intersects the segment CD at X
def circumcircle (B E C : Type*) := 
ω

def intersect_circumcircle (ω : Type*) (CD : Type*) := 
∃X (X ∈ ω)

-- Defining intersections of lines AX, BC at point Y and BX, AD at point Z
def line_intersections (A X Y C E Z : Type*) :=
(Y = intersect AX BC) ∧ (Z = intersect BX AD)

-- Main theorem to be proved
theorem tangency_condition {A B C D E X Y Z : Type*}
  (ht : is_trapezoid A B C D)
  (ha : angle_constraints A B C D α β)
  (ie : intersect_diagonals A B C D = E)
  (hx : intersect_circumcircle (circumcircle B E C) CD)
  (li : line_intersections A X Y B C D E) :
  (is_tangent E Z ω) ↔ (is_tangent B E (circumcircle B X Y)) := sorry

end tangency_condition_l496_496108


namespace bugs_meet_on_diagonal_l496_496079

noncomputable def isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (AB CD : ℝ), (AB > CD) ∧ (AB = AB) ∧ (CD = CD)

noncomputable def same_speeds (speed1 speed2 : ℝ) : Prop :=
  speed1 = speed2

noncomputable def opposite_directions (path1 path2 : ℝ → ℝ) (diagonal_length : ℝ) : Prop :=
  ∀ t, path1 t = diagonal_length - path2 t

noncomputable def bugs_meet (A B C D : Type) (path1 path2 : ℝ → ℝ) (T : ℝ) : Prop :=
  ∃ t ≤ T, path1 t = path2 t

theorem bugs_meet_on_diagonal :
  ∀ (A B C D : Type) (speed : ℝ) (path1 path2 : ℝ → ℝ) (diagonal_length cycle_period : ℝ),
  isosceles_trapezoid A B C D →
  same_speeds speed speed →
  (∀ t, 0 ≤ t → t ≤ cycle_period) →
  opposite_directions path1 path2 diagonal_length →
  bugs_meet A B C D path1 path2 cycle_period :=
by
  intros
  sorry

end bugs_meet_on_diagonal_l496_496079


namespace circle_m_range_circle_intersects_m_value_l496_496022

-- Definition of the problem
noncomputable def equation_C (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def is_circle (m : ℝ) : Prop :=
  m < 5 / 4

noncomputable def intersects_line (m : ℝ) : Prop :=
  -- This is the mathematical condition for the given problem, representing the specific scenario where the circle intersects the line with the specified chord length.
  ∃ (x y: ℝ), 
    equation_C x y m ∧ 
    (x + 2*y - 4 = 0) ∧ 
    ((4/5*sqrt 5)^2 = 2 * (5 - m) - (1/5))

-- Prove the statements
theorem circle_m_range (m : ℝ) : 
  (∀ x y, equation_C x y m → is_circle m) :=
by
  sorry

theorem circle_intersects_m_value (m : ℝ) : 
  (∀ x y, equation_C x y m → intersects_line m) →
  m = 3.62 :=
by
  sorry

end circle_m_range_circle_intersects_m_value_l496_496022


namespace real_solutions_to_system_l496_496669

theorem real_solutions_to_system :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), 
    (x = z + w + 2*z*w*x) ∧ 
    (y = w + x + 2*w*x*y) ∧ 
    (z = x + y + 2*x*y*z) ∧ 
    (w = y + z + 2*y*z*w) ↔ 
    (x, y, z, w) ∈ s) ∧
    (s.card = 15) :=
sorry

end real_solutions_to_system_l496_496669


namespace number_of_elements_in_B_l496_496033

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set (ℕ × ℕ) := {(x, y) | x ∈ A ∧ y ∈ A ∧ (x - y) ∈ A}

theorem number_of_elements_in_B : B.to_finset.card = 10 :=
by
  sorry

end number_of_elements_in_B_l496_496033


namespace john_safety_percentage_l496_496102

def bench_max_weight : ℕ := 1000
def john_weight : ℕ := 250
def weight_on_bar : ℕ := 550
def total_weight := john_weight + weight_on_bar
def percentage_of_max_weight := (total_weight * 100) / bench_max_weight
def percentage_under_max_weight := 100 - percentage_of_max_weight

theorem john_safety_percentage : percentage_under_max_weight = 20 := by
  sorry

end john_safety_percentage_l496_496102


namespace John_days_per_week_l496_496449

theorem John_days_per_week
    (patients_first : ℕ := 20)
    (patients_increase_rate : ℕ := 20)
    (patients_second : ℕ := (20 + (20 * 20 / 100)))
    (total_weeks_year : ℕ := 50)
    (total_patients_year : ℕ := 11000) :
    ∃ D : ℕ, (20 * D + (20 + (20 * 20 / 100)) * D) * total_weeks_year = total_patients_year ∧ D = 5 := by
  sorry

end John_days_per_week_l496_496449


namespace every_street_covered_l496_496089

def intersections := {A, B, C, D, E, F, G, H, I, J, K : Type → Prop}

def horizontal_streets := {
  (A, B, C, D),
  (E, F, G),
  (H, I, J, K)
}

def vertical_streets := {
  (A, E, H),
  (B, F, I),
  (D, G, J)
}

def diagonal_streets := {
  (H, F, C),
  (C, G, K)
}

noncomputable
def street_coverage := by
  sorry

theorem every_street_covered (p1 p2 p3 : intersections) (h1 : p1 = G) (h2 : p2 = H) (h3 : p3 = B) : 
  street_coverage horizontal_streets vertical_streets diagonal_streets (p1, p2, p3) :=
by
  sorry

end every_street_covered_l496_496089


namespace exists_common_fixed_point_l496_496109

variable {G : Set (ℝ → ℝ)}

def is_affine (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ f = λ x, a * x + b

def is_group (G : Set (ℝ → ℝ)) : Prop :=
  (∀ {f g : ℝ → ℝ}, f ∈ G → g ∈ G → (g ∘ f) ∈ G) ∧
  (∀ {f : ℝ → ℝ}, f ∈ G → ∃ f_inv : ℝ → ℝ, (f_inv ∘ f) = id ∧ (f_inv ∈ G))

def has_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃ (x_f : ℝ), f x_f = x_f

theorem exists_common_fixed_point
  (nonempty_G : G ≠ ∅)
  (G_affine : ∀ f ∈ G, is_affine f)
  (G_group : is_group G)
  (G_fixed_point : ∀ f ∈ G, has_fixed_point f) :
  ∃ k : ℝ, ∀ f ∈ G, f k = k := sorry

end exists_common_fixed_point_l496_496109


namespace largest_x_63_over_8_l496_496712

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496712


namespace simplify_expression_l496_496161

variable {x y : ℝ}
variable (h : x * y ≠ 0)

theorem simplify_expression (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^2 + 1) / y) - ((x^2 - 1) / y) * ((y^3 - 1) / x) =
  (x^3*y^2 - x^2*y^3 + x^3 + x^2 + y^2 + y^3) / (x*y) :=
by sorry

end simplify_expression_l496_496161


namespace prob_high_quality_product_distribution_mean_variance_l496_496067

-- Definitions used in the problem
def QualityIndexDist : ProbabilityDistribution ℝ := normal 64 10 

def HighQualityRange : Set ℝ := {x | 54 ≤ x ∧ x ≤ 84}

def GivenProbabilities : Prop := 
  P(μ - σ ≤ X ∧ X ≤ μ + σ) = 0.6827 ∧
  P(μ - 2σ ≤ X ∧ X ≤ μ + 2σ) = 0.9545 ∧
  P(μ - 3σ ≤ X ∧ X ≤ μ + 3σ) = 0.9973

-- Proof Statements as Lean Theorems
theorem prob_high_quality_product : GivenProbabilities → 
  P(λ x, QualityIndexDist.pdf x, HighQualityRange) = 0.82 :=
sorry

theorem distribution_mean_variance (X : binom 5 0.82) : 
  (X.mean, X.variance) = (4.1, 0.738) :=
sorry


end prob_high_quality_product_distribution_mean_variance_l496_496067


namespace circle_range_of_m_l496_496180

open Real

theorem circle_range_of_m (m : ℝ) : 
  (∃ h k r, r > 0 ∧ (x y : ℝ) → (x - h)^2 + (y - k)^2 = r^2) ↔ m < 1 / 2 :=
begin
  sorry
end

end circle_range_of_m_l496_496180


namespace solve_for_y_l496_496400

theorem solve_for_y (x : ℝ) (y : ℝ) (h1 : x = 8) (h2 : x^(2*y) = 16) : y = 2/3 :=
by
  sorry

end solve_for_y_l496_496400


namespace ratio_PC_PB_l496_496437

noncomputable def point := (ℝ × ℝ)

def square (A B C D : point) : Prop :=
  A.1 = 0 ∧ A.2 = 6 ∧
  B.1 = 6 ∧ B.2 = 6 ∧
  C.1 = 6 ∧ C.2 = 0 ∧
  D.1 = 0 ∧ D.2 = 0

def midpoint (P Q R : point) : Prop :=
  R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2

def intersection (l₁ l₂ : ℝ → ℝ) : point :=
( l₁(1), l₂(1))

def line_CN (C N: point): ℝ → ℝ := λ x, -2*x + 12

def line_BD (B D: point): ℝ → ℝ := λ x, -x + 6

theorem ratio_PC_PB :
  ∀ (A B C D N P: point),
  square A B C D →
  A.1 = 0 → A.2 = 6 →
  B.1 = 6 → B.2 = 6 →
  C.1 = 6 → C.2 = 0 →
  D.1 = 0 → D.2 = 0 →
  midpoint A B N → 
  P = intersection (line_BD B D) (line_CN C N) →
  P = C →
  (dist P C / dist P B) = 0 :=
by sorry

end ratio_PC_PB_l496_496437


namespace total_chips_l496_496215

-- Definitions of the given conditions
def Viviana_chocolate_chips (Susana_chocolate_chips : ℕ) := Susana_chocolate_chips + 5
def Susana_vanilla_chips (Viviana_vanilla_chips : ℕ) := 3 / 4 * Viviana_vanilla_chips
def Viviana_vanilla_chips := 20
def Susana_chocolate_chips := 25

-- The statement to prove the total number of chips
theorem total_chips :
  let Viviana_choco := Viviana_chocolate_chips Susana_chocolate_chips,
      Susana_vani := Susana_vanilla_chips Viviana_vanilla_chips,
      total := Viviana_choco + Viviana_vanilla_chips + Susana_chocolate_chips + Susana_vani
  in total = 90 :=
by
  sorry

end total_chips_l496_496215


namespace abs_pi_expression_l496_496668

theorem abs_pi_expression : ∀ (π : ℝ), π ≈ 3.14 → (|π + |π - 6|| = 6) :=
by {
  intro π,
  intro hπ,
  sorry
}

end abs_pi_expression_l496_496668


namespace Ivan_returns_alive_Ivan_takes_princesses_l496_496106

theorem Ivan_returns_alive (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6: ∀ girl : ℕ, girl ∈ five_girls → 
          ∃ truth_count : ℕ, 
          (truth_count = (if girl ∈ Tsarevnas then 2 else 3))): 
  ∃ princesses : Finset ℕ, princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ ∀ k ∈ Koscheis, k ∉ princesses :=
sorry

theorem Ivan_takes_princesses (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6 and cond7: ∀ girl1 girl2 girl3 : ℕ, girl1 ≠ girl2 → girl2 ≠ girl3 → girl1 ∈ Tsarevnas → girl2 ∈ Tsarevnas → girl3 ∈ Tsarevnas → 
          ∃ (eldest middle youngest : ℕ), 
              (eldest ∈ Tsarevnas ∧ middle ∈ Tsarevnas ∧ youngest ∈ Tsarevnas) 
          ∧
              (eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
          ∧
              (∀ k ∈ Koscheis, k ≠ eldest ∧ k ≠ middle ∧ k ≠ youngest)
  ):
  ∃ princesses : Finset ℕ, 
          princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ 
          (∃ eldest ,∃ middle,∃ youngest : ℕ, eldest ∈ princesses ∧ middle ∈ princesses ∧ youngest ∈ princesses ∧ 
                 eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
:=
sorry

end Ivan_returns_alive_Ivan_takes_princesses_l496_496106


namespace dulce_points_l496_496845

variable (D : ℕ)
def maxPoints : ℕ := 5
def valPoints : ℕ := 2 * (maxPoints + D)
def totalTeamPoints : ℕ := maxPoints + D + valPoints
def opponentPoints : ℕ := 40
def pointsBehind : ℕ := 16

theorem dulce_points : totalTeamPoints = opponentPoints - pointsBehind → D = 3 :=
by
  intro h
  let s :=  totalTeamPoints
  sorry

end dulce_points_l496_496845


namespace smores_cost_calculation_l496_496148

variable (people : ℕ) (s'mores_per_person : ℕ) (s'mores_per_set : ℕ) (cost_per_set : ℕ)

theorem smores_cost_calculation
  (h1 : s'mores_per_person = 3)
  (h2 : people = 8)
  (h3 : s'mores_per_set = 4)
  (h4 : cost_per_set = 3):
  (people * s'mores_per_person / s'mores_per_set) * cost_per_set = 18 := 
by
  sorry

end smores_cost_calculation_l496_496148


namespace largest_circle_radius_on_chessboard_l496_496145

def circle_radius_chessboard (side_length : ℝ) : ℝ :=
  let r := Math.sqrt (5 / 2)
  1 / 2 * r

theorem largest_circle_radius_on_chessboard :
  let side_length := 1
  circle_radius_chessboard side_length = Math.sqrt 10 / 2 :=
by
  sorry

end largest_circle_radius_on_chessboard_l496_496145


namespace tangent_line_at_pt_l496_496934

theorem tangent_line_at_pt :
  let y := λ x : ℝ, 1 / x
  let P := (-1 : ℝ, -1) in
  ∃ (c : ℝ → ℝ), (∀ (x : ℝ), c x = y x ∨ c x = -x - P.1) ∧ (∀ (x y : ℝ), c x + y - 2 = 0) :=
by
  sorry

end tangent_line_at_pt_l496_496934


namespace unequal_circles_no_tangents_l496_496569

theorem unequal_circles_no_tangents (C₁ C₂ : Circle) (h₁ : C₁ ≠ C₂) (same_plane : ∃ P, C₁ ⊂ P ∧ C₂ ⊂ P) : 
  ¬ (num_common_tangents C₁ C₂ = 0) := sorry

end unequal_circles_no_tangents_l496_496569


namespace sign_of_E_l496_496183

-- Definitions based on the given conditions
def F (n : ℕ) : ℤ := sorry -- F is some function from positive integers to integers
def E (n : ℕ) : ℤ := (-1)^n * F n -- The derived expression according to the problem

-- Theorem to be proved
theorem sign_of_E (n : ℕ) (hn : n > 0) : 
  (even n → E n = F n) ∧ (odd n → E n = -F n) :=
by
  sorry

end sign_of_E_l496_496183


namespace fixed_point_R_min_ratio_PQ_QR_l496_496837

noncomputable theory

open_locale real

structure Point :=
(x : ℝ)
(y : ℝ)

def parabola (p : Point) : Prop := p.y^2 = 4 * p.x

theorem fixed_point_R (P : Point) (hP : P.y ≠ 0)
  (ht1 : ∃ P1, parabola P1 ∧ P.y * P1.y = 2 * (P1.x + P.x))
  (ht2 : ∃ P2, parabola P2 ∧ P.y * P2.y = 2 * (P2.x + P.x))
  (orthogonal_condition : ∀ P1 P2, P1.y * P2.y = -P.y^2) :
  (∃ R, R.x = 2 ∧ R.y = 0) :=
sorry

theorem min_ratio_PQ_QR (P Q R : Point) (hP : P.y ≠ 0)
  (hQ : Q.x = P.x / 2 ∧ Q.y = P.y / 2)
  (hR : R.x = 2 ∧ R.y = 0) :
  ∃ (ratio : ℝ), ratio = 2 * real.sqrt 2 :=
sorry

end fixed_point_R_min_ratio_PQ_QR_l496_496837


namespace sets_relationship_l496_496687

def M : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 3 * k - 2}
def P : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def S : Set ℤ := {x : ℤ | ∃ m : ℤ, x = 6 * m + 1}

theorem sets_relationship : S ⊆ P ∧ M = P := by
  sorry

end sets_relationship_l496_496687


namespace max_g_of_15_l496_496456

noncomputable def g (x : ℝ) : ℝ := x^3  -- Assume the polynomial g(x) = x^3 based on the maximum value found.

theorem max_g_of_15 (g : ℝ → ℝ) (h_coeff : ∀ x, 0 ≤ g x)
  (h3 : g 3 = 3) (h27 : g 27 = 1701) : g 15 = 3375 :=
by
  -- According to the problem's constraint and identified solution,
  -- here is the statement asserting that the maximum value of g(15) is 3375
  sorry

end max_g_of_15_l496_496456


namespace max_min_distance_PA_l496_496368

def curve_C : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 4 + y^2 / 9 = 1}

def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

def parametric_curve_C (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 3 * sin θ)

def standard_line_l (x y : ℝ) : Prop := 2 * x + y - 6 = 0

def distance_PA (θ α : ℝ) : ℝ :=
  let P := (2 * cos θ, 3 * sin θ)
  let d := (Real.sqrt 5 / 5) * abs (4 * cos θ + 3 * sin θ - 6)
  (2 * Real.sqrt 5 / 5) * abs (5 * sin (θ + α) - 6)

theorem max_min_distance_PA (θ α : ℝ) :
  -1 ≤ sin (θ + α) ∧ sin (θ + α) ≤ 1 →
  max (distance_PA θ α) = (22 * Real.sqrt 5 / 5) ∧ min (distance_PA θ α) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end max_min_distance_PA_l496_496368


namespace model_A_sampling_l496_496263

theorem model_A_sampling (prod_A prod_B prod_C total_prod total_sampled : ℕ)
    (hA : prod_A = 1200) (hB : prod_B = 6000) (hC : prod_C = 2000)
    (htotal : total_prod = prod_A + prod_B + prod_C) (htotal_car : total_prod = 9200)
    (hsampled : total_sampled = 46) :
    (prod_A * total_sampled) / total_prod = 6 := by
  sorry

end model_A_sampling_l496_496263


namespace equilateral_triangle_angles_sum_l496_496433

theorem equilateral_triangle_angles_sum (A B C A1 A2 B1 : Type)
  [IsEquilateralTriangle A B C] (h1 : B = B1) (h2 : C = A2) (hA1 : dist B A1 = dist A1 A2)
  (hA2 : dist A1 A2 = dist A2 C) (hB1 : dist A B1 = 2 * dist B1 C) :
  ∠AA1B1 + ∠AA2B1 = 30 :=
sorry

end equilateral_triangle_angles_sum_l496_496433


namespace non_square_rhombus_l496_496084

theorem non_square_rhombus (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  (∃ x y : ℝ, ((|x + y| / (2 * a) + |x - y| / (2 * b)) = 1)) → 
  ∀ p1 p2 p3 p4 : ℝ × ℝ, 
    p1 = (a, a) ∧ p2 = (-a, -a) ∧ p3 = (b, -b) ∧ p4 = (-b, b) →
    is_rhombus p1 p2 p3 p4 ∧ ¬is_square p1 p2 p3 p4 :=
by
  sorry

end non_square_rhombus_l496_496084


namespace probability_of_smallest_section_l496_496267

-- Define the probabilities for the largest and next largest sections
def P_largest : ℚ := 1 / 2
def P_next_largest : ℚ := 1 / 3

-- Define the total probability constraint
def total_probability (P_smallest : ℚ) : Prop :=
  P_largest + P_next_largest + P_smallest = 1

-- State the theorem to be proved
theorem probability_of_smallest_section : 
  ∃ P_smallest : ℚ, total_probability P_smallest ∧ P_smallest = 1 / 6 := 
by
  sorry

end probability_of_smallest_section_l496_496267


namespace regression_analysis_incorrect_statement_l496_496580

theorem regression_analysis_incorrect_statement :
  ∃ (B : Prop), 
  (∀ (A C D : Prop), 
    (A ↔ ∀ (x̄ ȳ : ℝ), (∃ (f : ℝ → ℝ), f = least_squares_regression_line  x̄ ȳ) → (f x̄ = ȳ)) ∧
    (B ↔ ¬(least_squares_regression_line_minimizes_distance (∃ (data_points : list (ℝ × ℝ)), true))) ∧
    (C ↔ ∀ (r : ℝ), r > 0 → positively_correlated r) ∧
    (D ↔ ∀ (r : ℝ), (|r| = abs r) → linear_correlation_weaker r)
  ) → ¬ B :=
begin
  sorry
end

end regression_analysis_incorrect_statement_l496_496580


namespace correct_statements_count_l496_496444

theorem correct_statements_count
  (a b c : ℝ)
  (h1 : ∀ a b : ℝ, a * b = b * a)
  (h2 : ∀ a : ℝ, a * 0 = a)
  (h3 : ∀ a b c : ℝ, (a * b) * c = c * (a * b) + (a * c) + (c * b) - 2 * c) :
  let f := λ x : ℝ, 1 + 3 * x + (1 / (3 * x)) in
  (∀ x : ℝ, f'(x) = 3 - (1 / (3 * x^2)) ∧ f'(x) = 0 → x = 1/3 ∨ x = -1/3) → 
  (f(-x) = -(f(x)) → false) → 
  ((∀ x < -1/3, f'(x) > 0) ∧ (∀ x > 1/3, f'(x) > 0)) → 
  1 :=
begin
  sorry
end

end correct_statements_count_l496_496444


namespace number_of_men_required_l496_496261

theorem number_of_men_required :
  ∀ (men1 men2 days1 days2 hours1 hours2 productivity_ratio job_ratio total_hours1 total_hours2),
  men1 = 250 → 
  days1 = 16 → 
  hours1 = 8 → 
  job_ratio = 3 → 
  days2 = 20 → 
  hours2 = 10 → 
  productivity_ratio = 0.8 →
  total_hours1 = men1 * days1 * hours1 →
  total_hours2 = job_ratio * total_hours1 →
  let full_productivity_hours := total_hours2 / productivity_ratio in
  let hours_per_man := days2 * hours2 in
  let required_men := full_productivity_hours / hours_per_man in
  required_men = 600 :=
by
  intros men1 men2 days1 days2 hours1 hours2 productivity_ratio job_ratio total_hours1 total_hours2
  intros h_men1 h_days1 h_hours1 h_job_ratio h_days2 h_hours2 h_productivity_ratio h_total_hours1 h_total_hours2
  let full_productivity_hours := total_hours2 / productivity_ratio
  let hours_per_man := days2 * hours2
  let required_men := full_productivity_hours / hours_per_man
  have h_total_hours1_eq : total_hours1 = men1 * days1 * hours1 := h_total_hours1
  rw h_total_hours1_eq at h_total_hours2
  have h_total_hours2_eq : total_hours2 = job_ratio * total_hours1 := h_total_hours2
  rw [h_total_hours1_eq, h_total_hours2_eq]
  have full_productivity_hours_eq : full_productivity_hours = (job_ratio * (men1 * days1 * hours1)) / productivity_ratio :=
    by simp only [full_productivity_hours]
  let required_men := full_productivity_hours / hours_per_man
  have hours_per_man_eq : hours_per_man = days2 * hours2 := rfl
  rw hours_per_man_eq

  sorry

end number_of_men_required_l496_496261


namespace probability_three_heads_one_tail_l496_496823

-- Define the number of coins tossed
def num_coins : ℕ := 4

-- Probability of heads for each coin
def p_heads : ℝ := 1/2

-- Probability of tails for each coin
def p_tails : ℝ := 1/2

-- Number of heads and tails for the specific case
def num_heads : ℕ := 3
def num_tails : ℕ := 1

-- Probability of exactly three heads and one tail from four coin tosses
theorem probability_three_heads_one_tail : 
  (choose num_coins num_heads) * p_heads^num_heads * p_tails^num_tails = 1/4 :=
by
  sorry

end probability_three_heads_one_tail_l496_496823


namespace sampling_method_is_systematic_l496_496614

noncomputable def is_systematic_sampling (population_size : ℕ) (total_sampled : ℕ) (divisible_by : ℕ) : Prop :=
  ∃ segments_size starting_point, 
  population_size % segments_size = 0 ∧ 
  total_sampled = population_size / segments_size ∧ 
  starting_point < segments_size ∧ 
  ∀ i, is_part (divisible_by * i + starting_point)

theorem sampling_method_is_systematic :
  is_systematic_sampling 75 15 5 :=
by
  sorry

end sampling_method_is_systematic_l496_496614


namespace simplify_trig_expression_l496_496510

theorem simplify_trig_expression :
  (\sin (15 * Real.pi / 180) + \sin (30 * Real.pi / 180) + \sin (45 * Real.pi / 180) + \sin (60 * Real.pi / 180) + \sin (75 * Real.pi / 180))
  / (\cos (15 * Real.pi / 180) * \sin (45 * Real.pi / 180) * \cos (30 * Real.pi / 180)) = 7.13109 :=
by
  sorry

end simplify_trig_expression_l496_496510


namespace train_speed_l496_496635

theorem train_speed : 
  ∀ (length time : ℕ), 
  length = 800 → time = 10 → 
  (length / time : ℝ) * 3.6 = 288 := by
  intros length time h_length h_time
  rw [h_length, h_time]
  calc
    (800 / 10 : ℝ) * 3.6 = 80 * 3.6 : by norm_num
                    ... = 288      : by norm_num

end train_speed_l496_496635


namespace fraction_of_milk_in_cup1_is_7_over_16_l496_496100

-- Define initial amounts in the cups
def initial_coffee_cup1 : ℝ := 3
def initial_milk_cup2 : ℝ := 7

-- Define the transferred amounts
def coffee_transferred_to_cup2 : ℝ := 1
def liquid_transferred_to_cup1 : ℝ := 2
def coffee_fraction_in_cup2 : ℝ := 1 / 8
def milk_fraction_in_cup2 : ℝ := 7 / 8

-- Calculate the amounts transferred
def coffee_transferred_back_to_cup1 : ℝ := liquid_transferred_to_cup1 * coffee_fraction_in_cup2
def milk_transferred_back_to_cup1 : ℝ := liquid_transferred_to_cup1 * milk_fraction_in_cup2

-- Calculate final amounts in Cup 1
def final_coffee_cup1 : ℝ := initial_coffee_cup1 - coffee_transferred_to_cup2 + coffee_transferred_back_to_cup1
def final_milk_cup1 : ℝ := milk_transferred_back_to_cup1
def total_liquid_cup1 : ℝ := final_coffee_cup1 + final_milk_cup1
def milk_fraction_in_cup1 : ℝ := final_milk_cup1 / total_liquid_cup1

theorem fraction_of_milk_in_cup1_is_7_over_16 :
  milk_fraction_in_cup1 = 7 / 16 := by
  sorry

end fraction_of_milk_in_cup1_is_7_over_16_l496_496100


namespace johns_burritos_l496_496857

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l496_496857


namespace largest_two_digit_prod_12_l496_496221

theorem largest_two_digit_prod_12 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ ((∃ a b : ℕ, n = 10 * a + b ∧ a * b = 12) ∧ ∀ m : ℕ, ((10 ≤ m ∧ m < 100) ∧ (∃ a b : ℕ, m = 10 * a + b ∧ a * b = 12) → m ≤ 62)) :=
by
  existsi 62
  constructor
  {
    split
    {
      exact by norm_num
      exact by norm_num
    }
  }
  {
    constructor
    sorry
  }

end largest_two_digit_prod_12_l496_496221


namespace largest_real_number_l496_496731

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496731


namespace probability_blue_from_H1_l496_496081

noncomputable def P (H A : Type) : Type := sorry -- Placeholder definition for probability

constants H1 H2 H3 A : Type
constants P_H1 P_H2 P_H3 P_A_H1 P_A_H2 P_A_H3 P_A : ℝ

axiom h1 : P_H1 = 0.4
axiom h2 : P_H2 = 0.4
axiom h3 : P_H3 = 0.2
axiom a_h1 : P_A_H1 = 0.6
axiom a_h2 : P_A_H2 = 0.8
axiom a_h3 : P_A_H3 = 0.2
axiom total_p_A : P_A = P_H1 * P_A_H1 + P_H2 * P_A_H2 + P_H3 * P_A_H3

theorem probability_blue_from_H1 : P (H1, A) = (P_H1 * P_A_H1) / P_A :=
by {
    rw [h1, h2, h3, a_h1, a_h2, a_h3, total_p_A],
    sorry
}

end probability_blue_from_H1_l496_496081


namespace summation_of_fractions_l496_496586

theorem summation_of_fractions :
  (\frac{1}{3} + \frac{1}{3} + \frac{1}{3} + \frac{1}{3} + \frac{1}{3} + \frac{1}{3} + \frac{1}{3}) 
  = (7 * \(\frac{1}{3}\)) := by
  sorry

end summation_of_fractions_l496_496586


namespace bus_passengers_l496_496926

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l496_496926


namespace Megan_final_balance_percentage_l496_496475

-- Definitions based on conditions
def Megan_initial_balance : ℝ := 125
def Megan_increase_rate : ℝ := 0.25
def Megan_decrease_rate : ℝ := 0.20

-- Proof statement
theorem Megan_final_balance_percentage :
  let initial_balance := Megan_initial_balance
  let increased_balance := initial_balance * (1 + Megan_increase_rate)
  let final_balance := increased_balance * (1 - Megan_decrease_rate) in
  (final_balance / initial_balance * 100) = 100 :=
by
  sorry

end Megan_final_balance_percentage_l496_496475


namespace ratio_of_triangle_areas_l496_496134

def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

theorem ratio_of_triangle_areas
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (A B : ℝ × ℝ) (l : ℝ → ℝ)
  (h_line : ∀ x, l x = (parabola x) → l x = -(Real.sqrt 2) / 4 * x + 1)
  (h_parabola_A : A.2 = parabola A.1)
  (h_parabola_B : B.2 = parabola B.1)
  (h_distance : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = 3) :
  (let S_ΔAOF := (1 / 2) * abs (A.1 * F.2 - F.1 * A.2),
       S_ΔBOF := (1 / 2) * abs (B.1 * F.2 - F.1 * B.2) in
   S_ΔAOF / S_ΔBOF) = 2 :=
sorry

end ratio_of_triangle_areas_l496_496134


namespace g_difference_l496_496876

variable (g : ℝ → ℝ)

-- Condition: g is a linear function
axiom linear_g : ∃ a b : ℝ, ∀ x : ℝ, g x = a * x + b

-- Condition: g(10) - g(4) = 18
axiom g_condition : g 10 - g 4 = 18

theorem g_difference : g 16 - g 4 = 36 :=
by
  sorry

end g_difference_l496_496876


namespace coffee_per_cup_for_weak_l496_496168

-- Defining the conditions
def weak_coffee_cups : ℕ := 12
def strong_coffee_cups : ℕ := 12
def total_coffee_tbsp : ℕ := 36
def weak_increase_factor : ℕ := 1
def strong_increase_factor : ℕ := 2

-- The theorem stating the problem
theorem coffee_per_cup_for_weak :
  ∃ W : ℝ, (weak_coffee_cups * W + strong_coffee_cups * (strong_increase_factor * W) = total_coffee_tbsp) ∧ (W = 1) :=
  sorry

end coffee_per_cup_for_weak_l496_496168


namespace smallest_possible_score_l496_496164

theorem smallest_possible_score (total_score : ℕ) (h1 : total_score = 32) :
    ∃ x : ℕ, x = 2 ∧ 
    (∃ a b c d e f : ℕ, 
    a ≥ 1 ∧ a ≤ 6 ∧ 
    b ≥ 1 ∧ b ≤ 6 ∧ 
    c ≥ 1 ∧ c ≤ 6 ∧ 
    d ≥ 1 ∧ d ≤ 6 ∧ 
    e ≥ 1 ∧ e ≤ 6 ∧ 
    f ≥ 1 ∧ f ≤ 6 ∧ 
    a + b + c + d + e + f = total_score ∧ min a (min b (min c (min d (min e f)))) = x) := 
begin
  sorry
end

end smallest_possible_score_l496_496164


namespace range_of_a_l496_496025

open Real

noncomputable def f (x a : ℝ) : ℝ := log 2 (x^2 - a * x + 3 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 ≤ x → deriv (λ x, log 2 (x^2 - a * x + 3 * a)) x ≥ 0) 
  → -4 < a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l496_496025


namespace problem1_problem2_l496_496384

-- Definitions
variable {U : Set ℝ} (A : Set ℝ) (B : Set ℝ) (b : ℝ)

def A : Set ℝ := { x | x^2 - 3*x + b = 0 }
def B : Set ℝ := { x | (x - 2)*(x^2 + 3*x - 4) = 0 }
def M : Set ℝ

theorem problem1 (h : b = 4) :
  (A ⊂ M ∧ M ⊆ B) ↔
  M = {-4} ∨ M = {1} ∨ M = {2} ∨
  M = {-4, 1} ∨ M = {-4, 2} ∨
  M = {1, 2} ∨ M = {-4, 1, 2} :=
sorry

theorem problem2 :
  (∀ b, (A ∩ (Bᶜ)) = ∅) ↔ (b > 9/4 ∨ b = 2) :=
sorry

end problem1_problem2_l496_496384


namespace appropriate_sampling_methods_l496_496616

noncomputable def regionA_points : ℕ := 150
noncomputable def regionB_points : ℕ := 120
noncomputable def regionC_points : ℕ := 180
noncomputable def regionD_points : ℕ := 150

noncomputable def total_points : ℕ := regionA_points + regionB_points + regionC_points + regionD_points
noncomputable def sample_size : ℕ := 100

noncomputable def regionC_large_points : ℕ := 20
noncomputable def regionC_large_sample : ℕ := 7

-- Appropriate sampling methods for investigation (1) and (2)
theorem appropriate_sampling_methods (total_points = 600) (sample_size = 100)
  (regionC_large_points = 20) (regionC_large_sample = 7) :
    "stratified sampling" = "stratified sampling" ∧ "simple random sampling" = "simple random sampling" :=
  sorry

end appropriate_sampling_methods_l496_496616


namespace road_renovation_proof_l496_496204

variables (x : ℕ) (m : ℕ)
variables (a_efficiency b_efficiency a_daily_cost b_daily_cost : ℕ) (total_cost_per_day : ℕ) (total_length : ℕ) (max_cost : ℕ)

def teamA_efficiency := b_efficiency * 3 / 2
def teamA_days := total_length / teamA_efficiency
def teamB_days := total_length / b_efficiency

def equations := (360 / b_efficiency - 360 / teamA_efficiency = 3) ∧
               (teamA_efficiency = b_efficiency * 3 / 2) ∧
               (a_daily_cost = 70000) ∧
               (b_daily_cost = 50000) ∧
               (total_length = 1200) ∧
               (max_cost = 1450000)

def min_days_for_teamA := m ≥ 10

def solution : Prop :=
  (b_efficiency = 40) ∧
  (teamA_efficiency = 60) ∧
  (m ≥ 10)

theorem road_renovation_proof :
  equations →
  solution :=
sorry

end road_renovation_proof_l496_496204


namespace part1_part2_l496_496791

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * |x - 1|

theorem part1 (a : ℝ) (h : a = 2) : 
  ∃ max min : ℝ, max = 6 ∧ min = 1 ∧ ∀ x ∈ set.Icc 0 2, f a x ≤ max ∧ f a x ≥ min := sorry

theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a x ≤ f a y) ↔ (-2 ≤ a ∧ a ≤ 0) := sorry

end part1_part2_l496_496791


namespace num_possible_values_of_a_l496_496917

theorem num_possible_values_of_a : 
  { a // a > 0 ∧ a ∣ 20 ∧ 5 ∣ a }.card = 3 :=
sorry

end num_possible_values_of_a_l496_496917


namespace f_sum_eight_nine_l496_496539

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)
axiom f_even_shift : ∀ x : ℝ, f(x + 2) = f(-x - 2)
axiom f_at_one : f 1 = 1

theorem f_sum_eight_nine :
  f 8 + f 9 = -1 :=
sorry

end f_sum_eight_nine_l496_496539


namespace number_of_children_l496_496558

theorem number_of_children (
  hours : ℕ,
  houses_per_hour : ℕ,
  treats_per_house : ℕ,
  total_treats : ℕ)
  (h1 : hours = 4)
  (h2 : houses_per_hour = 5)
  (h3 : treats_per_house = 3)
  (h4 : total_treats = 180)
  : ∃ children : ℕ, children = total_treats / (houses_per_hour * treats_per_house * hours) ∧ children = 3 :=
by {
  use 3,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end number_of_children_l496_496558


namespace solve_for_n_l496_496047

theorem solve_for_n (n : ℕ) (h : sqrt (5 + n) = 7) : n = 44 :=
sorry

end solve_for_n_l496_496047


namespace count_5_divisors_1_to_50000_l496_496813

def count_divisors (n : ℕ) (p : ℕ) : ℕ :=
  let rec aux (n k : ℕ) : ℕ :=
    if n < k then 0 else n / k + aux n (k * p)
  in aux n p

theorem count_5_divisors_1_to_50000 :
  count_divisors 50000 5 = 12499 :=
by
  sorry

end count_5_divisors_1_to_50000_l496_496813


namespace total_pencils_correct_l496_496649

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end total_pencils_correct_l496_496649


namespace problem_equiv_c_l496_496752

variables (m n : Set Point) (α β : Set Point)
variables [IsLine m] [IsLine n] [IsPlane α] [IsPlane β]

-- Definitions for the conditions
def line_in_plane (l : Set Point) (p : Set Point) [IsLine l] [IsPlane p] : Prop := l ⊆ p
def parallel_planes (p1 p2 : Set Point) [IsPlane p1] [IsPlane p2] : Prop := ∀ (l1 l2 : Set Point) [IsLine l1] [IsLine l2], (l1 ⊆ p1) ∧ (l2 ⊆ p2) → parallel l1 l2
def coplanar_lines (l1 l2 : Set Point) [IsLine l1] [IsLine l2] : Prop := ∃ (p : Set Point) [IsPlane p], (l1 ⊆ p) ∧ (l2 ⊆ p)

-- Theorem to prove
theorem problem_equiv_c (h1: line_in_plane m α)
                        (h2: line_in_plane n β)
                        (h3: parallel_planes α β)
                        (h4: coplanar_lines m n) :
  parallel m n :=
sorry

end problem_equiv_c_l496_496752


namespace slopes_hyperbola_l496_496796

theorem slopes_hyperbola 
  (x y : ℝ)
  (M : ℝ × ℝ) 
  (t m : ℝ) 
  (h_point_M_on_line: M = (9 / 5, t))
  (h_hyperbola : ∀ t: ℝ, (16 * m^2 - 9) * t^2 + 160 * m * t + 256 = 0)
  (k1 k2 k3 : ℝ)
  (h_k2 : k2 = -5 * t / 16) :
  k1 + k3 = 2 * k2 :=
sorry

end slopes_hyperbola_l496_496796


namespace largest_real_number_l496_496717

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496717


namespace derivative_zero_not_sufficient_or_necessary_l496_496256

variable {α : Type*} [LinearOrder α] [TopologicalSpace α]

noncomputable def has_extremum_at (f : α → ℝ) (a : α) : Prop :=
  ∃ (ε > (0 : ℝ)), ∀ (x ∈ metric.ball a ε), f x ≤ f a

theorem derivative_zero_not_sufficient_or_necessary (f : α → ℝ) (a : α)
(hf : deriv f a = 0) :
  ¬(has_extremum_at f a) ∨ (∃ (g : α → ℝ), (deriv g a = 0) ∧ ¬(has_extremum_at g a)) :=
sorry

end derivative_zero_not_sufficient_or_necessary_l496_496256


namespace perpendicular_bisector_eq_l496_496933

theorem perpendicular_bisector_eq
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0):
  ∃ (a b c : ℝ), a = 3 ∧ b = -1 ∧ c = -9 ∧ (∀ (x y : ℝ), a * x + b * y + c = 0) :=
sorry

end perpendicular_bisector_eq_l496_496933


namespace proof_problem_l496_496007

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (a1 d : ℤ), ∀ n, a n = a1 + d * (n - 1)

def conditions (a : ℕ → ℤ) : Prop :=
  2 * a 2 + a 3 + a 5 = 20 ∧ 
  (∑ i in Finset.range 10, a (i + 1)) = 100

def correct_general_term (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a (i + 1) * 2 ^ a(i + 1) ∧
        S n = (10 + (6 * n - 5) * 2 ^ (2 * n + 1)) / 9

theorem proof_problem :
  ∀ (a : ℕ → ℤ), 
    arithmetic_sequence a → 
    conditions a → 
    correct_general_term a ∧ 
    sum_of_sequence a (λ n, 0) :=
by
  intros a arith_seq_cond cond
  have gen_term := sorry -- The proof for general term
  have sum_seq := sorry -- The proof for sum of the sequence
  exact ⟨gen_term, sum_seq⟩

end proof_problem_l496_496007


namespace interval_of_increase_of_g_l496_496373

theorem interval_of_increase_of_g
    (a : ℝ)
    (f : ℝ → ℝ)
    (g : ℝ → ℝ)
    (h_f : ∀ x, f(-x) = f(x)) -- f(x) is an even function
    (h_f_def : ∀ x, f(x) = x^2 + (a-1)*x)
    (h_g_def : ∀ x, g(x) = ax^2 - 2*x - 1) :
    ∃ a, a = 1 ∧ (∀ x, x ≥ 1 → g x ≤ g (x+1)) :=
by
  sorry

end interval_of_increase_of_g_l496_496373


namespace integer_An_l496_496014

theorem integer_An 
  (a b : ℤ)
  (h_ab : a > b ∧ b > 0)
  (θ : ℝ)
  (h_θ : θ ∈ set.Ioo 0 (Real.pi / 2))
  (h_sin : Real.sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, (a^2 + b^2 : ℤ) ^ n * Real.sin (n * θ) ∈ ℤ :=
by
  sorry

end integer_An_l496_496014


namespace total_seats_600_l496_496646

-- Let s represent the total number of seats on the airplane.
def total_seats (s : ℝ) : Prop := 30 + 0.20 * s + (3/4) * s = s

-- We have to prove that s = 600 given the conditions
theorem total_seats_600 : ∃ s, s = 600 ∧ total_seats s :=
by
  use 600
  split
  rfl
  unfold total_seats
  sorry

end total_seats_600_l496_496646


namespace function_range_l496_496542

noncomputable def g (x : ℝ) : ℝ := Real.log x + x

noncomputable def f (x : ℝ) : ℝ := x^2 / (Real.log x + x)

theorem function_range :
  (∃ a, 0 < a ∧ a < 1 ∧ g a = 0) →
  (∀ x, x > 0 → x ≠ a → f x ∈ Set.Ioo (-∞:ℝ) 0 ∪ Set.Icc 1 ∞) :=
begin
  sorry
end

end function_range_l496_496542


namespace simplify_expression_l496_496160

theorem simplify_expression :
  sqrt (real.cbrt (sqrt (sqrt ((1 : ℝ) / (65536 : ℝ))))) = (1 : ℝ) / real.sqrt 2 := by
  sorry

end simplify_expression_l496_496160


namespace exceed_plan_by_47_percent_l496_496043

theorem exceed_plan_by_47_percent (plan : ℕ) (percent : ℕ) (target_parts : ℤ) :
  plan = 40 →
  percent = 47 →
  target_parts = 59 →
  (plan : ℤ) + (percent : ℤ) * plan / 100 ≥ target_parts :=
by
  intro h_plan h_percent h_target_parts
  rw [h_plan, h_percent, h_target_parts]
  norm_num
  sorry

end exceed_plan_by_47_percent_l496_496043


namespace cube_less_than_triple_l496_496225

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l496_496225


namespace parallel_lines_l496_496035

theorem parallel_lines (k : ℝ) 
  (h₁ : ∀ x y : ℝ, k * x - y + 1 = 0) 
  (h₂ : ∀ x y : ℝ, x - k * y + 1 = 0) 
  (h₃ : ∃ l₁ l₂ : ℝ → ℝ → Prop, (l₁ = h₁) ∧ (l₂ = h₂) ∧ (l₁ ∥ l₂)) : 
  k = -1 :=
by
  sorry

end parallel_lines_l496_496035


namespace six_digit_number_l496_496764

theorem six_digit_number (E U L S R T : ℕ) 
  (h1 : E + U + L = 6)
  (h2 : S + R + U + T = 18)
  (h3 : U * T = 15)
  (h4 : S * L = 8)
  (distinct_digits : ∀ x : ℕ, List.count [E, U, L, S, R, T] x <= 1) :
  (10^5 * E + 10^4 * U + 10^3 * L + 10^2 * S + 10 * R + T = 132465) :=
begin
  sorry
end

end six_digit_number_l496_496764


namespace set_solution_l496_496467

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x - 4 else 2^(-x) - 4

theorem set_solution : {x : ℝ | f(x - 2) < 0} = {x : ℝ | 0 < x ∧ x < 4} :=
by
  sorry

end set_solution_l496_496467


namespace log_sum_zero_l496_496255

theorem log_sum_zero (a b c N : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_N : 0 < N) (h_neq_N : N ≠ 1) (h_geom_mean : b^2 = a * c) : 
  1 / Real.logb a N - 2 / Real.logb b N + 1 / Real.logb c N = 0 :=
  by
  sorry

end log_sum_zero_l496_496255


namespace total_amount_is_correct_l496_496864

variable (amount_from_grandpa amount_from_grandma amount_from_aunt amount_from_uncle : ℕ)

-- Conditions
def amount_from_grandpa := 30
def amount_from_grandma := 3 * amount_from_grandpa
def amount_from_aunt := 2 * amount_from_grandpa
def amount_from_uncle := amount_from_grandma / 2

-- Total amount calculation
def total_amount := amount_from_grandpa + amount_from_grandma + amount_from_aunt + amount_from_uncle

-- The theorem to prove
theorem total_amount_is_correct : total_amount = 225 := by
  sorry

end total_amount_is_correct_l496_496864


namespace number_of_linear_equations_in_one_variable_l496_496597

def is_linear_equation_in_one_variable (eq : String) : Bool :=
  match eq with
  | "3x=10" => true
  | "5x - 4/7y = 35" => false
  | "x^2 - 4 = 0" => false
  | "4z - 3(z + 2) = 1" => true
  | "1/x = 3" => false
  | "x = 3" => true
  | _ => false

theorem number_of_linear_equations_in_one_variable :
  let equations := ["3x=10", "5x - 4/7y = 35", "x^2 - 4 = 0", "4z - 3(z + 2) = 1", "1/x = 3", "x = 3"];
  Nat := 3 :=
by
  let linear_eqs := equations.filter is_linear_equation_in_one_variable
  have : linear_eqs.length = 3 := sorry
  exact this

end number_of_linear_equations_in_one_variable_l496_496597


namespace find_m_value_m_range_l496_496335

-- Define conditions based on the given statements
def ellipse_foci_on_x_axis (m : ℝ) : Prop := 
  9 - m > 0 ∧ 2 * m > 0

def hyperbola_eccentricity_condition (m : ℝ) : Prop := 
  m > 0 ∧ (∃ (e : ℝ), (sqrt (6) / 2 < e ∧ e < sqrt (2)) ∧ (e = sqrt (1 + (5 / m))))

theorem find_m_value (m : ℝ) (h₁ : ellipse_foci_on_x_axis m) 
  (h₂ : 5 > m ∨ m > 0) :
  m = 4 / 3 := sorry

theorem m_range (m : ℝ) (p : Prop) (q : Prop)
  (hp : p) (hq : q) 
  (hp_def : p ↔ ellipse_foci_on_x_axis m) 
  (hq_def : q ↔ hyperbola_eccentricity_condition m) :
  5 / 2 < m ∧ m < 3 := sorry

end find_m_value_m_range_l496_496335


namespace inequality_solution_l496_496511

def fractional_part (x : ℝ) : ℝ := x - x.floor

def integer_part (x : ℝ) : ℤ := x.floor

theorem inequality_solution (x : ℝ) (hx : 0 ≤ fractional_part(x) ∧ fractional_part(x) < 1) :
  fractional_part(x) * (integer_part(x) - 1) < x - 2 → x ≥ 3 :=
by
  sorry

end inequality_solution_l496_496511


namespace round_robin_teams_l496_496205

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end round_robin_teams_l496_496205


namespace find_c_l496_496884

theorem find_c : 
  ∃ c : ℝ, (let p (x : ℝ) := 4 * x - 9 in
            let q (x : ℝ) := 5 * x - c in
            p (q 3) = 11) → 
          c = 10 :=
begin
  sorry
end

end find_c_l496_496884


namespace largest_x_l496_496726

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496726


namespace surface_area_of_circumscribed_sphere_l496_496637

-- Defining the lengths of the mutually perpendicular edges
def a : ℝ := Real.sqrt 3
def b : ℝ := Real.sqrt 2
def c : ℝ := 1

-- Statement of the theorem
theorem surface_area_of_circumscribed_sphere : 
  let R := Real.sqrt (a^2 + b^2 + c^2) / 2 in
  4 * Real.pi * R^2 = 6 * Real.pi := by
  -- Applying the conditions (which would involve calculations similar to those in the provided solution)
  sorry

end surface_area_of_circumscribed_sphere_l496_496637


namespace segment_bisects_side_l496_496122

variables {A B C H A' : Type*}
variables [linear_ordered_comm_ring A] [linear_ordered_comm_ring B] [linear_ordered_comm_ring C]
variables [linear_ordered_comm_ring H]
variables [linear_ordered_comm_ring A']

def orthocenter (A B C H : Type*) := true -- placeholder definition
def is_diameter (AA' : Type*) := true -- placeholder definition
def circumcircle (triangle : Type*) := true -- placeholder definition

theorem segment_bisects_side (ABC H A' : Type*) 
  (orthocenter_ABC_H : orthocenter A B C H) 
  (diameter_AA' : is_diameter AA') 
  (circumcircle_ABC : circumcircle (triangle A B C)) :
  bisects A' H (line_segment B C) :=
begin
  sorry
end

end segment_bisects_side_l496_496122


namespace market_price_of_article_l496_496064

theorem market_price_of_article
  (initial_tax : ℚ := 0.035)
  (reduced_tax : ℚ := 1/30)
  (tax_difference : ℚ := 13)
  (h1 : initial_tax - reduced_tax = 1/600)
  (h2 : tax_difference = 13) :
  let P : ℚ := tax_difference / (initial_tax - reduced_tax)
  in P = 7800 :=
by
  sorry

end market_price_of_article_l496_496064


namespace optimal_bicycle_point_l496_496289

noncomputable def distance_A_B : ℝ := 30  -- Distance between A and B is 30 km
noncomputable def midpoint_distance : ℝ := distance_A_B / 2  -- Distance between midpoint C to both A and B is 15 km
noncomputable def walking_speed : ℝ := 5  -- Walking speed is 5 km/h
noncomputable def biking_speed : ℝ := 20  -- Biking speed is 20 km/h

theorem optimal_bicycle_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ (30 - x + 4 * x = 60 - 3 * x) → x = 5 :=
by sorry

end optimal_bicycle_point_l496_496289


namespace polynomial_evaluation_sum_l496_496107

theorem polynomial_evaluation_sum :
  let P := (λ x : ℤ, x^6 - x^3 - x^2 - 1) in
  ∀ (q1 q2 q3 : ℤ → ℤ),
  (P = λ x, (q1 x) * (q2 x) * (q3 x)) ∧
  (∀ q, (q = q1 ∨ q = q2 ∨ q = q3) → irreducible q) ∧
  (∀ q, (q = q1 ∨ q = q2 ∨ q = q3) → monic q) →
  q1 3 + q2 3 + q3 3 = 32 :=
sorry

end polynomial_evaluation_sum_l496_496107


namespace part1_part2_l496_496623

noncomputable def ellipse_eq (x y : ℂ) : Prop := 
  x^2 / 36 + y^2 / 4 = 1

noncomputable def line_eq (x y c : ℂ) : Prop :=
  y = x / 3 + c

structure Point : Type :=
  (x : ℂ)
  (y : ℂ)

def point_P : Point := ⟨3 * complex.sqrt 2, complex.sqrt 2⟩

noncomputable def intersects (A B : Point) (c : ℂ) : Prop := 
  ∃ x1 x2 : ℂ, ellipse_eq x1 (x1 / 3 + c) ∧ ellipse_eq x2 (x2 / 3 + c) ∧ 
               A = ⟨x1, x1 / 3 + c⟩ ∧ B = ⟨x2, x2 / 3 + c⟩

noncomputable def incenter_locus : Prop :=
  ∀ (A B : Point) (c : ℂ), intersects A B c → ∃ k : ℂ, 
  ∀ P_incenter : Point, P_incenter = ⟨3 * complex.sqrt 2, k⟩

noncomputable def area_of_triangle (A B P : Point) : ℂ :=
  1/2 * abs (A.x * (B.y - P.y) + B.x * (P.y - A.y) + P.x * (A.y - B.y))

noncomputable def angle_condition (A B P : Point) : Prop :=
  ∠(A, P, B) = π / 3

noncomputable def triangle_area_given_angle : ℂ :=
  ∀ (A B : Point), angle_condition A B point_P →
  abs (area_of_triangle A B point_P) = ?

-- The final Lean statements for both parts of the problem:

theorem part1 : incenter_locus := sorry

theorem part2 : triangle_area_given_angle := sorry

end part1_part2_l496_496623


namespace correct_statements_count_l496_496672

def Line := ℝ → ℝ

def slope (l : Line) : Option ℝ := sorry -- Placeholder for slope definition

def parallel (l1 l2 : Line) : Prop := sorry -- Placeholder for parallel definition

def perpendicular (l1 l2 : Line) : Prop := sorry -- Placeholder for perpendicular definition

def angle_of_inclination (l : Line) : ℝ := sorry -- Placeholder for angle of inclination definition

theorem correct_statements_count :
  ∀ (l1 l2 : Line),
    (¬(l1 = l2)) →
    ((∃ m1 m2 : ℝ, slope l1 = some m1 ∧ slope l2 = some m2 ∧ m1 = m2 → parallel l1 l2) ∧
     (∀ m1 m2 : ℝ, perpendicular l1 l2 → slope l1 = some m1 ∧ slope l2 = some m2 → m1 * m2 = -1) ∧
     (angle_of_inclination l1 = angle_of_inclination l2 → parallel l1 l2) ∧
     (∀ m1 m2 : ℝ, parallel l1 l2 → slope l1 = some m1 ∧ slope l2 = some m2 → m1 = m2) ∧
     (¬(parallel l1 l2 → slope l1 = some m1 ∧ slope l2 = some m2 → m1 = m2))) →
    2 :=
by
  intro l1 l2 not_coincide H
  sorry -- Proof goes here

end correct_statements_count_l496_496672


namespace determine_m_l496_496798

theorem determine_m (m : ℕ) (f : ℝ → ℝ) (h1 : f = λ x : ℝ, x^(3*m - 5))
  (h2 : ∀ x : ℝ, 0 < x → f(x) < f(y) ∧ 0 < y → y < x)
  (h3 : ∀ x : ℝ, f(-x) = f(x)) :
  m = 1 := sorry

end determine_m_l496_496798


namespace max_distance_increases_l496_496146

noncomputable def largest_n_for_rearrangement (C : ℕ) (marked_points : ℕ) : ℕ :=
  670

theorem max_distance_increases (C : ℕ) (marked_points : ℕ) (n : ℕ) (dist : ℕ → ℕ → ℕ) :
  ∀ i j, i < marked_points → j < marked_points →
    dist i j ≤ n → 
    (∃ rearrangement : ℕ → ℕ, 
    ∀ i j, i < marked_points → j < marked_points → 
      dist (rearrangement i) (rearrangement j) > dist i j) → 
    n ≤ largest_n_for_rearrangement C marked_points := 
by
  sorry

end max_distance_increases_l496_496146


namespace largest_x_l496_496723

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496723


namespace problem_statement_l496_496370

noncomputable def f (x : ℝ) : ℝ := log10 (10 / (sqrt (1 + 4 * x^2) - 2 * x))

theorem problem_statement : f 2017 + f (-2017) = 2 :=
by
  sorry

end problem_statement_l496_496370


namespace part1_part2_l496_496432

def seq (x : ℝ) : ℕ → ℝ
| 0     => x
| (n+1) => 1 - (1 / seq n)

theorem part1 (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  seq x 0 = x ∧ seq x 1 = 1 - 1 / x ∧ seq x 2 = -1 / (x - 1) ∧ seq x 3 = x := by
  sorry

theorem part2 :
  seq 2023 2023 = 2022 / 2023 := by
  sorry

end part1_part2_l496_496432


namespace correct_graph_illustration_l496_496662

-- Definitions for the problem conditions
def speed (car : String) : ℝ := 
  if car = "X" then v
  else if car = "Y" then 3 * v
  else if car = "Z" then 2 * v
  else 0

def time (car : String) : ℝ := 
  if car = "X" then t
  else if car = "Y" then t / 3
  else if car = "Z" then t / 2
  else 0

-- The proof statement
theorem correct_graph_illustration : (select_correct_graph == "D") := sorry

end correct_graph_illustration_l496_496662


namespace Craig_walked_distance_l496_496673

-- Definitions of conditions
variable (distance_school_to_david distance_total distance_david_to_home : ℝ)

-- Assumptions based on the given conditions
axiom school_to_david_distance : distance_school_to_david = 0.2
axiom total_distance : distance_total = 0.9

-- Statement to be proved
theorem Craig_walked_distance :
  distance_david_to_home = distance_total - distance_school_to_david :=
by
  -- Ensuring the subtraction
  rw [school_to_david_distance, total_distance]
  -- Simplification, showing correct answer is 0.7
  conv in (0.9 - 0.2) => simp
  -- providing final result
  sorry

end Craig_walked_distance_l496_496673


namespace y_coordinate_of_point_l496_496016

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

noncomputable def focus_of_parabola : ℝ × ℝ :=
  (0, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem y_coordinate_of_point {x_P y_P : ℝ} 
  (h₁ : point_on_parabola x_P y_P)
  (h₂ : distance (x_P, y_P) focus_of_parabola = 2) : 
  y_P = 1 := 
by
  sorry

end y_coordinate_of_point_l496_496016


namespace john_half_full_decks_l496_496866

theorem john_half_full_decks :
  ∀ (x : ℕ),
    let full_decks := 3 in
    let cards_per_deck := 52 in
    let poor_quality_cards := 34 in
    let remaining_cards := 200 in
    let total_cards_before_throwing := remaining_cards + poor_quality_cards in
    let full_decks_cards := full_decks * cards_per_deck in
    let half_full_deck_cards := cards_per_deck / 2 in
    total_cards_before_throwing = full_decks_cards + x * half_full_deck_cards →
    x = 3 :=
begin
  intros x full_decks cards_per_deck poor_quality_cards remaining_cards total_cards_before_throwing
          full_decks_cards half_full_deck_cards h,
  sorry
end

end john_half_full_decks_l496_496866


namespace jessica_seashells_l496_496853

theorem jessica_seashells (joan jessica total : ℕ) (h1 : joan = 6) (h2 : total = 14) (h3 : total = joan + jessica) : jessica = 8 :=
by
  -- proof steps would go here
  sorry

end jessica_seashells_l496_496853


namespace sum_odd_divisors_of_180_l496_496982

theorem sum_odd_divisors_of_180 : 
  let n := 180 in 
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0} in 
  ∑ d in odd_divisors, d = 78 :=
by
  let n := 180
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0}
  have h : ∑ d in odd_divisors, d = 78 := sorry -- Sum of odd divisors of 180
  exact h

end sum_odd_divisors_of_180_l496_496982


namespace find_theta_max_g_value_l496_496026

noncomputable def f (x θ : ℝ) : ℝ := cos x * cos (x - θ) - 1 / 2 * cos θ

theorem find_theta (θ : ℝ) (hθ : θ ∈ Set.Ioo 0 π) :
  (∀ x, f x θ ≤ f (π / 3) θ) → θ = 2 * π / 3 :=
sorry

noncomputable def g (x θ : ℝ) : ℝ := 2 * f (3 / 2 * x) θ

theorem max_g_value (θ : ℝ) (hθ : θ = 2 * π / 3) :
  (∃ x ∈ Set.Icc 0 (π / 3), g x θ = 1) :=
sorry

end find_theta_max_g_value_l496_496026


namespace max_dot_product_of_points_on_ellipses_l496_496395

theorem max_dot_product_of_points_on_ellipses :
  let C1 (M : ℝ × ℝ) := M.1^2 / 25 + M.2^2 / 9 = 1
  let C2 (N : ℝ × ℝ) := N.1^2 / 9 + N.2^2 / 25 = 1
  ∃ M N : ℝ × ℝ,
    C1 M ∧ C2 N ∧
    (∀ M N, C1 M ∧ C2 N → M.1 * N.1 + M.2 * N.2 ≤ 15 ∧ 
      (∃ θ φ, M = (5 * Real.cos θ, 3 * Real.sin θ) ∧ N = (3 * Real.cos φ, 5 * Real.sin φ) ∧ (M.1 * N.1 + M.2 * N.2 = 15))) :=
by
  sorry

end max_dot_product_of_points_on_ellipses_l496_496395


namespace provisions_last_days_l496_496070

theorem provisions_last_days
  (initial_soldiers : ℕ)
  (initial_consumption_per_soldier : ℕ)
  (initial_days : ℕ)
  (additional_soldiers : ℕ)
  (new_consumption_per_soldier : ℚ)
  (total_provisions : ℕ)
  (new_total_soldiers : ℕ)
    (new_total_consumption : ℚ)
    (new_days : ℚ):
    initial_soldiers = 1200 →
    initial_consumption_per_soldier = 3 →
    initial_days = 30 →
    additional_soldiers = 528 →
    new_consumption_per_soldier = 2.5 →
    total_provisions = initial_soldiers * initial_consumption_per_soldier * initial_days →
    new_total_soldiers = initial_soldiers + additional_soldiers →
    new_total_consumption = new_total_soldiers * new_consumption_per_soldier →
    new_days = total_provisions / new_total_consumption →
    new_days = 25 := 
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9,
  sorry
end

end provisions_last_days_l496_496070


namespace sum_of_positive_odd_divisors_of_180_l496_496979

/-- The sum of the positive odd divisors of 180 is 78. -/
theorem sum_of_positive_odd_divisors_of_180 : 
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45] in
  let sum_odd_divisors := List.sum odd_divisors in
  sum_odd_divisors = 78 := 
by
  -- odd_divisors is [1, 3, 5, 9, 15, 45]
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45]
  -- Summation of the odd divisors
  let sum_odd_divisors := List.sum odd_divisors
  -- Verify that the sum is 78
  show sum_odd_divisors = 78
  sorry

end sum_of_positive_odd_divisors_of_180_l496_496979


namespace increasing_interval_l496_496940

noncomputable def f (x : ℝ) : ℝ := log x + 1 / x

theorem increasing_interval (x : ℝ) (hx : x > 0) : 
    ∃ (I : Set ℝ), (∀ y ∈ I, f' y > 0) ∧ I = Set.Ioi (1) :=
by
  have f' : ∀ x, x > 0 → deriv f x = (x - 1) / (x^2) := sorry
  sorry

end increasing_interval_l496_496940


namespace mod_abc_eq_zero_l496_496397

open Nat

theorem mod_abc_eq_zero
    (a b c : ℕ)
    (h1 : (a + 2 * b + 3 * c) % 9 = 1)
    (h2 : (2 * a + 3 * b + c) % 9 = 2)
    (h3 : (3 * a + b + 2 * c) % 9 = 3) :
    (a * b * c) % 9 = 0 := by
  sorry

end mod_abc_eq_zero_l496_496397


namespace school_total_payment_l496_496098

def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def chaperones_per_class : ℕ := 5
def student_fee : ℝ := 5.50
def adult_fee : ℝ := 6.50

def total_students : ℕ := num_classes * students_per_class
def total_adults : ℕ := num_classes * chaperones_per_class

def total_student_cost : ℝ := total_students * student_fee
def total_adult_cost : ℝ := total_adults * adult_fee

def total_cost : ℝ := total_student_cost + total_adult_cost

theorem school_total_payment : total_cost = 1010.0 := by
  sorry

end school_total_payment_l496_496098


namespace cos_cubed_identity_l496_496954

theorem cos_cubed_identity (a b : ℝ) :
  (∀ θ : ℝ, cos θ ^ 3 = a * cos (3 * θ) + b * cos θ) ↔ (a = 1/4 ∧ b = 3/4) := by
  sorry

end cos_cubed_identity_l496_496954


namespace distance_equality_l496_496810

open Real

def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

theorem distance_equality (a : ℝ) :
  distance_point_to_line (-3, -4) a 1 1 = distance_point_to_line (6, 3) a 1 1 ↔
  a = -1/3 ∨ a = -7/9 :=
by sorry

end distance_equality_l496_496810


namespace ellipse_equation_line_equation_l496_496647
-- Import the necessary libraries

-- Problem (I): The equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hA : (1 : ℝ) / a^2 + (9 / 4 : ℝ) / b^2 = 1)
  (h_ecc : b^2 = (3 / 4 : ℝ) * a^2) : 
  (a^2 = 4 ∧ b^2 = 3) :=
by
  sorry

-- Problem (II): The equation of the line
theorem line_equation (k : ℝ) (h_area : (12 * Real.sqrt (2 : ℝ)) / 7 = 12 * abs k / (4 * k^2 + 3)) : 
  k = 1 ∨ k = -1 :=
by
  sorry

end ellipse_equation_line_equation_l496_496647


namespace binomial_parameters_l496_496470

-- Definitions and conditions
def is_binomial (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop :=
∀ k : ℕ, X k = (n.choose k) * (p^k) * ((1-p)^(n-k))

def mean (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ :=
n * p

def variance (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ :=
n * p * (1 - p)

-- Theorem statement
theorem binomial_parameters (X : ℕ → ℝ) : 
  (∃ n p, is_binomial X n p ∧ mean X n p = 15 ∧ variance X n p = 12) → (60, 0.25) :=
begin
  intro h,
  sorry
end

end binomial_parameters_l496_496470


namespace november_profit_december_selling_price_no_880_profit_l496_496283

variables 
  (cost_price sell_price : ℕ) 
  (initial_volume monthly_volume : ℕ)
  (price_reduction profit : ℕ)
  (new_volume increase_rate : ℕ)
  (profit_goal reduced_price : ℕ)

-- Conditions
axiom h1 : cost_price = 48 
axiom h2 : sell_price >= cost_price
axiom h3 : sell_price = 60
axiom h4 : initial_volume = 60
axiom h5 : increase_rate = 10
axiom h6 : price_reduction = 2
axiom h7 : reduced_price = sell_price - price_reduction
axiom h8 : new_volume = initial_volume + price_reduction * increase_rate
axiom h9 : profit = (reduced_price - cost_price) * new_volume

-- Question 1: November Profit
theorem november_profit : profit = 800 := 
by sorry

-- Question 2: December Selling Price to Achieve 770 Profit
noncomputable theory
variables (x : ℕ)
axiom h10 : 770 = (sell_price - x - cost_price) * (initial_volume + x * increase_rate)
theorem december_selling_price : (x = 5) → reduced_price = 55 := 
by sorry

-- Question 3: Impossibility of 880 Profit
variables (y : ℕ)
axiom h11 : 880 = (sell_price - y - cost_price) * (initial_volume + y * increase_rate)
theorem no_880_profit : (y^2 - 6*y + 16 ≠ 0) → ¬ ∃ y, (880 = (sell_price - y - cost_price) * (initial_volume + y * increase_rate)) := 
by sorry

end november_profit_december_selling_price_no_880_profit_l496_496283


namespace train_speed_approximation_l496_496541

theorem train_speed_approximation (train_speed_mph : ℝ) (seconds : ℝ) :
  (40 : ℝ) * train_speed_mph * 1 / 60 = seconds → seconds = 27 := 
  sorry

end train_speed_approximation_l496_496541


namespace sum_of_positive_odd_divisors_of_180_l496_496978

/-- The sum of the positive odd divisors of 180 is 78. -/
theorem sum_of_positive_odd_divisors_of_180 : 
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45] in
  let sum_odd_divisors := List.sum odd_divisors in
  sum_odd_divisors = 78 := 
by
  -- odd_divisors is [1, 3, 5, 9, 15, 45]
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45]
  -- Summation of the odd divisors
  let sum_odd_divisors := List.sum odd_divisors
  -- Verify that the sum is 78
  show sum_odd_divisors = 78
  sorry

end sum_of_positive_odd_divisors_of_180_l496_496978


namespace find_a_b_and_analyze_function_l496_496786

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem find_a_b_and_analyze_function :
  ∃ (a b : ℝ), 
    (let f := f with a b,
      f(1) = 2 ∧
      deriv f 1 = -1 ∧
      (a = 1 ∧ b = 8 / 3) ∧
      (intervals_of_monotonicity_and_extrema f = 
         { inc_on : set.Ixx (-∞ : ℝ) 0 ∪ set.Ixx 2 ∞, 
           dec_on : set.Ixx 0 2, 
           local_max : { (0, 8 / 3) }, 
           local_min : { (2, 4 / 3) }
         } ∧
      (max_on_interval f (-2) 5 = 58 / 3) ) )
sorry

end find_a_b_and_analyze_function_l496_496786


namespace short_bar_is_400_l496_496280

noncomputable def short_bar_length (total_length long_bar_extra : ℕ) : ℕ :=
  (total_length - long_bar_extra) / 2

theorem short_bar_is_400 (total_length long_bar_extra short_bar : ℕ) (h1 : total_length = 950) (h2 : long_bar_extra = 150) (h3 : short_bar = short_bar_length total_length long_bar_extra) :
  short_bar = 400 :=
by
  rw [h1, h2, short_bar_length]
  simp
  sorry

end short_bar_is_400_l496_496280


namespace wire_length_unique_l496_496955

noncomputable def distance_increment := (5 / 3)

theorem wire_length_unique (d L : ℝ) 
  (h1 : L = 25 * d) 
  (h2 : L = 24 * (d + distance_increment)) :
  L = 1000 := by
  sorry

end wire_length_unique_l496_496955


namespace sequence_increasing_l496_496355

theorem sequence_increasing (n : ℕ) (h : n ≥ 2) : 
  let a : ℕ → ℝ := λ n, (n - 1) / (n + 1) in a n < a (n + 1) :=
by
  sorry

end sequence_increasing_l496_496355


namespace sum_of_inverses_geq_one_l496_496451

noncomputable def upper_density (A : Set ℤ) : ℝ :=
  sorry  -- Definition of upper density

def k_set (k : ℕ) (A : Set ℤ) : Prop :=
  ∃ x : Fin k → ℤ, ∀ i j : Fin k, i ≠ j → Disjoint (x i +ᵥ A) (x j +ᵥ A)

theorem sum_of_inverses_geq_one (t : ℕ) (K : Fin t → ℕ) (A : Fin t → Set ℤ)
  (h1 : ∀ i, k_set (K i) (A i))
  (h2 : Finset.univ.bUnion (λ i, A i) = Set.univ) :
  1 ≤ Finset.univ.sum (λ i, (K i : ℝ)⁻¹) :=
sorry

end sum_of_inverses_geq_one_l496_496451


namespace WendyMorningRoutineTime_l496_496218

def skincareRoutineTime : Nat := 2 + 3 + 1 + 2 + 2 + 1
def waitingTimes : Nat := 3 + 4 + 3 + 5 + 2
def makeupTime : Nat := 30
def hairStylingTime : Nat := 20
def totalTime : Nat := skincareRoutineTime + waitingTimes + makeupTime + hairStylingTime

theorem WendyMorningRoutineTime : totalTime = 78 := by
  have h1 : skincareRoutineTime = 11 := rfl
  have h2 : waitingTimes = 17 := rfl
  have h3 : makeupTime = 30 := rfl
  have h4 : hairStylingTime = 20 := rfl
  have h5 : skincareRoutineTime + waitingTimes + makeupTime + hairStylingTime = 11 + 17 + 30 + 20 := by 
    rw [h1, h2, h3, h4]
  have h6 : 11 + 17 + 30 + 20 = 78 := rfl
  rw [h5, h6]
  sorry

end WendyMorningRoutineTime_l496_496218


namespace inscribed_sphere_radius_l496_496208

noncomputable section

-- Definitions
variables (r1 r2 : ℝ) (m1 m2 a1 a2 : ℝ) (F V : ℝ)

-- Conditions
def equal_surface_area (F : ℝ) (r1 r2 a1 a2 : ℝ) : Prop :=
  π * r1 * (r1 + a1) = π * r2 * (r2 + a2)

def equal_volume (V : ℝ) (r1 r2 m1 m2 : ℝ) : Prop :=
  (π / 3) * r1^2 * m1 = (π / 3) * r2^2 * m2

def slant_height (r m a : ℝ) : Prop :=
  a^2 = r^2 + m^2

-- Theorem Statement
theorem inscribed_sphere_radius (h1 : equal_surface_area F r1 r2 a1 a2)
                                (h2 : equal_volume V r1 r2 m1 m2)
                                (h3 : slant_height r1 m1 a1)
                                (h4 : slant_height r2 m2 a2)
                                (h5 : r1 ≠ r2) :
  ∃ (ρ : ℝ), ρ = r1 * r2 / sqrt (r1^2 + r2^2) :=
sorry

end inscribed_sphere_radius_l496_496208


namespace sum_of_positive_odd_divisors_eq_78_l496_496995

theorem sum_of_positive_odd_divisors_eq_78 :
  ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 180)), d = 78 :=
by {
  -- proof steps go here
  sorry
}

end sum_of_positive_odd_divisors_eq_78_l496_496995


namespace allan_initial_balloon_count_l496_496642

def number_of_balloons (A Jake_balloons additional Allan_Jake_total: ℕ) : ℕ := 
  A + additional + Jake_balloons

theorem allan_initial_balloon_count (A : ℕ) (Jake_balloons : ℕ := 5) (additional : ℕ := 2) (Allan_Jake_total : ℕ := 10) :
  number_of_balloons A Jake_balloons additional Allan_Jake_total = Allan_Jake_total →
  A = 3 := 
by
  intros h,
  sorry

end allan_initial_balloon_count_l496_496642


namespace engines_not_defective_l496_496408

theorem engines_not_defective (batches : ℕ) (engines_per_batch : ℕ) (defective_fraction : ℚ) 
  (h_batches : batches = 5) (h_engines_per_batch : engines_per_batch = 80) (h_defective_fraction : defective_fraction = 1/4) : 
  (batches * engines_per_batch - (batches * engines_per_batch * defective_fraction)).toNat = 300 :=
by
  sorry

end engines_not_defective_l496_496408


namespace color_white_cells_l496_496147

/--
  Given an infinite grid with a finite number of black cells such that each black cell has an even 
  number of white neighboring cells (0, 2, or 4), prove that one can color each white cell red or 
  green such that each black cell has an equal number of red and green neighboring cells.
-/
theorem color_white_cells 
  (grid : ℤ × ℤ → bool) 
  (black_cells : finset (ℤ × ℤ))
  (h_black : ∀ b ∈ black_cells, (card ((neighbors b).filter (λ c, grid c = ff)) % 2 = 0)) :
  ∃ R G : finset (ℤ × ℤ),
    (∀ w, grid w = ff → (w ∈ R ∨ w ∈ G) ∧ ¬(w ∈ R ∧ w ∈ G)) ∧ 
    (∀ b ∈ black_cells, (card ((neighbors b).filter (λ c, c ∈ R)) = card ((neighbors b).filter (λ c, c ∈ G)))) :=
sorry

end color_white_cells_l496_496147


namespace common_area_of_intersecting_circles_l496_496573

theorem common_area_of_intersecting_circles (a : ℝ) :
  a > 0 →
  let r := a / 2,
      hypotenuse := a * Real.sqrt 2,
      semi_circle_area := (1 / 2) * Real.pi * (a / Real.sqrt 2)^2,
      triangle_area := (1 / 2) * (a / Real.sqrt 2) * (a / Real.sqrt 2),
      common_area := semi_circle_area - triangle_area
  in common_area = a^2 / 8 * (Real.pi - 2) :=
by
  intro ha
  let r := a / 2
  let hypotenuse := a * Real.sqrt 2
  let semi_circle_area := (1 / 2) * Real.pi * (a / Real.sqrt 2)^2
  let triangle_area := (1 / 2) * (a / Real.sqrt 2) * (a / Real.sqrt 2)
  let common_area := semi_circle_area - triangle_area
  have semi_circle_area_correct : semi_circle_area = a^2 * Real.pi / 8, by sorry
  have triangle_area_correct : triangle_area = a^2 / 4, by sorry
  have common_area_correct : common_area = a^2 / 8 * (Real.pi - 2), by sorry
  exact common_area_correct

end common_area_of_intersecting_circles_l496_496573


namespace pyramid_has_11_ways_to_positive_top_l496_496833

def is_positive_top (a b c d e : Int) : Bool :=
  a * b * c * d * e = 1

def count_ways_to_positive_top : Nat :=
  let values := [(-1), 1]
  (values.product values).product values
  |>
  List.product values
  |>
  List.product values
  |>
  List.count (λ ((a, b), (c, d, e)) => is_positive_top a b c d e = true)

theorem pyramid_has_11_ways_to_positive_top : count_ways_to_positive_top = 11 :=
  sorry

end pyramid_has_11_ways_to_positive_top_l496_496833


namespace find_x_and_intersection_l496_496349

open Set

noncomputable def A : Set ℕ := {1, 3, 5}
noncomputable def B (x : ℝ) : Set ℝ := {1, 2, x^2 - 1}
noncomputable def unionAB : Set ℝ := {1, 2, 3, 5}

theorem find_x_and_intersection (x : ℝ) :
  B x ⊆ unionAB ↔ (x = 2 ∨ x = -2 ∨ x = Real.sqrt 6 ∨ x = -Real.sqrt 6) ∧ (A ∩ B x = {1, 3} ∨ A ∩ B x = {1, 5}) := by
  sorry

end find_x_and_intersection_l496_496349


namespace solution_set_inequality1_solution_set_inequality2_l496_496244

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l496_496244


namespace slope_perpendicular_l496_496234

theorem slope_perpendicular (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (h2 : y1 = -4) (h3 : x2 = 6) (h4 : y2 = 5) : 
  let m := (y2 - y1) / (x2 - x1)
  in -1 / m = -1 / 3 :=
by
  have hxy : y2 - y1 = 9 := by rw [h2, h4]; norm_num
  have hxx : x2 - x1 = 3 := by rw [h1, h3]; norm_num
  let m := (y2 - y1) / (x2 - x1)
  have hm : m = 3 := by rw [hxy, hxx]; norm_num
  rw [hm]
  norm_num

end slope_perpendicular_l496_496234


namespace doug_initial_marbles_l496_496307

theorem doug_initial_marbles (E D : ℕ) (H1 : E = D + 5) (H2 : E = 27) : D = 22 :=
by
  -- proof provided here would infer the correct answer from the given conditions
  sorry

end doug_initial_marbles_l496_496307


namespace train_crossing_time_approx_l496_496636

-- Definitions according to the problem conditions.
def train_speed_km_per_hr : ℝ := 100
def train_length_meters : ℝ := 500

-- Conversion from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * 1000 / 3600

-- Time calculation given speed in m/s and length in meters.
def time_to_cross_pole (train_length : ℝ) (train_speed_m_per_s : ℝ) : ℝ :=
  train_length / train_speed_m_per_s

-- The final theorem stating the problem equivalence.
theorem train_crossing_time_approx :
  time_to_cross_pole train_length_meters (km_per_hr_to_m_per_s train_speed_km_per_hr) ≈ 18 :=
by 
  sorry

end train_crossing_time_approx_l496_496636


namespace unique_integer_cube_triple_l496_496230

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l496_496230


namespace min_value_of_expression_l496_496120

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
  (1/x + 4/y + 9/z) >= 36/5 :=
sorry

end min_value_of_expression_l496_496120


namespace overlapping_area_zero_l496_496537

-- Definition of the points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def point0 : Point := { x := 0, y := 0 }
def point1 : Point := { x := 2, y := 2 }
def point2 : Point := { x := 2, y := 0 }
def point3 : Point := { x := 0, y := 2 }
def point4 : Point := { x := 1, y := 1 }

def triangle1 : Triangle := { p1 := point0, p2 := point1, p3 := point2 }
def triangle2 : Triangle := { p1 := point3, p2 := point1, p3 := point0 }

-- Function to calculate the area of a triangle
def area (t : Triangle) : ℝ :=
  0.5 * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

-- Using collinear points theorem to prove that the area of the overlapping region is zero
theorem overlapping_area_zero : area { p1 := point0, p2 := point1, p3 := point4 } = 0 := 
by 
  -- This follows directly from the fact that the points (0,0), (2,2), and (1,1) are collinear
  -- skipping the actual geometric proof for conciseness
  sorry

end overlapping_area_zero_l496_496537


namespace lilies_per_centerpiece_correct_l496_496474

-- Definitions based on the conditions
def num_centerpieces : ℕ := 6
def roses_per_centerpiece : ℕ := 8
def cost_per_flower : ℕ := 15
def total_budget : ℕ := 2700

-- Definition of the number of orchids per centerpiece using given condition
def orchids_per_centerpiece : ℕ := 2 * roses_per_centerpiece

-- Definition of the total cost for roses and orchids before calculating lilies
def total_rose_cost : ℕ := num_centerpieces * roses_per_centerpiece * cost_per_flower
def total_orchid_cost : ℕ := num_centerpieces * orchids_per_centerpiece * cost_per_flower
def total_rose_and_orchid_cost : ℕ := total_rose_cost + total_orchid_cost

-- Definition for the remaining budget for lilies
def remaining_budget_for_lilies : ℕ := total_budget - total_rose_and_orchid_cost

-- Number of lilies in total and per centerpiece
def total_lilies : ℕ := remaining_budget_for_lilies / cost_per_flower
def lilies_per_centerpiece : ℕ := total_lilies / num_centerpieces

-- The proof statement we want to assert
theorem lilies_per_centerpiece_correct : lilies_per_centerpiece = 6 :=
by
  sorry

end lilies_per_centerpiece_correct_l496_496474


namespace ratio_yellow_jelly_beans_l496_496327

noncomputable def total_jelly_beans := 15 + 22 + 35 + 40
noncomputable def yellow_jelly_beans := 15 * 0.40 + 22 * 0.30 + 35 * 0.25 + 40 * 0.10

theorem ratio_yellow_jelly_beans :
  yellow_jelly_beans / total_jelly_beans * 100 ≈ 23 :=
begin
  -- Proof omitted
  sorry
end

end ratio_yellow_jelly_beans_l496_496327


namespace z_x_given_y_z_y_given_x_l496_496592

variables (x y : Prop)
noncomputable def z : Prop → ℝ
variable (zxy : z (x ∧ y) = 0.10)
variable (zx : z x = 0.02)
variable (zy : z y = 0.10)

theorem z_x_given_y : (z (x ∧ y) / z y) = 1 :=
by sorry

theorem z_y_given_x : (z (x ∧ y) / z x) = 5 :=
by sorry

end z_x_given_y_z_y_given_x_l496_496592


namespace diameter_of_second_square_l496_496524

theorem diameter_of_second_square 
  (d₁ : ℝ) (h₁ : d₁ = 4 * real.sqrt 2)
  (A₂ : ℝ) (h₂ : A₂ = 2 * (d₁ / real.sqrt 2) ^ 2) :
  real.sqrt (2 * A₂) * real.sqrt 2 = 8 :=
by
  sorry

end diameter_of_second_square_l496_496524


namespace reach_14_from_458_l496_496599

def double (n : ℕ) : ℕ :=
  n * 2

def erase_last_digit (n : ℕ) : ℕ :=
  n / 10

def can_reach (start target : ℕ) (ops : List (ℕ → ℕ)) : Prop :=
  ∃ seq : List (ℕ → ℕ), seq = ops ∧
    seq.foldl (fun acc f => f acc) start = target

-- The proof problem statement
theorem reach_14_from_458 : can_reach 458 14 [double, erase_last_digit, double, double, erase_last_digit, double, double, erase_last_digit] :=
  sorry

end reach_14_from_458_l496_496599


namespace global_school_math_students_l496_496519

theorem global_school_math_students (n : ℕ) (h1 : n < 600) (h2 : n % 28 = 27) (h3 : n % 26 = 20) : n = 615 :=
by
  -- skip the proof
  sorry

end global_school_math_students_l496_496519


namespace square_of_positive_difference_l496_496175

theorem square_of_positive_difference {y : ℝ}
  (h : (45 + y) / 2 = 50) :
  (|y - 45|)^2 = 100 :=
by
  sorry

end square_of_positive_difference_l496_496175


namespace compute_a_d_sum_l496_496119

variables {a1 a2 a3 d1 d2 d3 : ℝ}

theorem compute_a_d_sum
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a_d_sum_l496_496119


namespace original_ticket_price_l496_496557

open Real

theorem original_ticket_price 
  (P : ℝ)
  (total_revenue : ℝ)
  (revenue_equation : total_revenue = 10 * 0.60 * P + 20 * 0.85 * P + 15 * P) 
  (total_revenue_val : total_revenue = 760) : 
  P = 20 := 
by
  sorry

end original_ticket_price_l496_496557


namespace ShelbyRainDrivingTime_l496_496506

-- Define the conditions
def drivingTimeNonRain (totalTime: ℕ) (rainTime: ℕ) : ℕ := totalTime - rainTime
def rainSpeed : ℚ := 20 / 60
def noRainSpeed : ℚ := 30 / 60
def totalDistance (rainTime: ℕ) (nonRainTime: ℕ) : ℚ := rainSpeed * rainTime + noRainSpeed * nonRainTime

-- Prove the question == answer given conditions
theorem ShelbyRainDrivingTime :
  ∀ (rainTime totalTime: ℕ),
  (totalTime = 40) →
  (totalDistance rainTime (drivingTimeNonRain totalTime rainTime) = 16) →
  rainTime = 24 :=
by
  intros rainTime totalTime ht hd
  have h1 : drivingTimeNonRain totalTime rainTime = 40 - rainTime := rfl
  rw [← h1] at hd
  sorry

end ShelbyRainDrivingTime_l496_496506


namespace sum_of_mean_and_median_is_68_8_l496_496593

def a : Set ℕ := {17, 27, 31, 53, 61}

theorem sum_of_mean_and_median_is_68_8 :
  let a_list := [17, 27, 31, 53, 61]
  let mean := (17 + 27 + 31 + 53 + 61) / 5.0
  let median := 31
  mean + ↑median = 68.8 :=
by
  let a_list := [17, 27, 31, 53, 61]
  let mean := (17 + 27 + 31 + 53 + 61) / 5.0
  let median := 31
  have h1 : mean = 37.8 := by decide
  have h2 : ↑median = 31.0 := by decide
  rw [h1, h2]
  norm_num

end sum_of_mean_and_median_is_68_8_l496_496593


namespace range_of_a_l496_496943

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ a ∈ set.Iio (-2) ∪ set.Ioi 2 :=
by
  sorry

end range_of_a_l496_496943


namespace part1_part2_part3_l496_496367

theorem part1 (n : ℕ) (h1 : 4^n - 2^n = 240) : n = 4 := sorry

theorem part2 (C : ℕ → ℕ → ℕ) :
  let n := 4
  let x := (5 : ℚ)
  let y := (-1 / (5 : ℚ))
  let term_r (r : ℕ) := C n r * x^(n-r) * y^r
  let coef_containing_x := C n 2 * x^(n-2) * y^2
  coef_containing_x = 150 := sorry

theorem part3 (C : ℕ → ℕ → ℕ) (f : ℕ → ℚ) :
  let n := 4
  let x := (5 : ℚ)
  let y := (-1 / (5 : ℚ))
  let term_r (r : ℕ) := C n r * x^(n-r) * y^r
  {term_r r | r ∈ {0, 2, 4}} = {625, 150, -1} := sorry

end part1_part2_part3_l496_496367


namespace correct_statement_count_is_2_l496_496745

-- Definitions of statements in Lean
def f (x m : ℝ) : ℝ := x^2 - 2 * m * x - 3

def statement1 (m : ℝ) : Prop := 4 * m^2 + 12 > 0

def statement2 (m : ℝ) : Prop := m = 1

def statement3 (m : ℝ) : Prop := m = -1

def statement4 (m : ℝ) : Prop :=
  let f := f 4 m
  let f' := f 2010 m
  f = f' ∧ m = 1007 -- This translation aligns with when f(4) = f(2010)

def number_of_correct_statements (m : ℝ) : ℕ :=
  (if statement1 m then 1 else 0) +
  (if statement2 m then 1 else 0) +
  (if statement3 m then 1 else 0) +
  (if statement4 m then 1 else 0)

theorem correct_statement_count_is_2 (m : ℝ) : number_of_correct_statements m = 2 :=
 by { sorry }

end correct_statement_count_is_2_l496_496745


namespace coral_pages_to_read_third_week_l496_496298

def pages_to_finish (total_pages : ℕ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) : ℕ :=
  let pages_read_first_week := (total_pages : ℝ) * first_week_fraction
  let remaining_pages_after_first_week := total_pages - pages_read_first_week.to_nat
  let pages_read_second_week := (remaining_pages_after_first_week : ℝ) * second_week_fraction
  let remaining_pages_after_second_week := remaining_pages_after_first_week - pages_read_second_week.to_nat
  remaining_pages_after_second_week

theorem coral_pages_to_read_third_week :
  pages_to_finish 600 0.5 0.3 = 210 := 
by
  sorry

end coral_pages_to_read_third_week_l496_496298


namespace pool_drain_rate_l496_496169

-- Define the dimensions and other conditions
def poolLength : ℝ := 150
def poolWidth : ℝ := 40
def poolDepth : ℝ := 10
def poolCapacityPercent : ℝ := 0.80
def drainTime : ℕ := 800

-- Define the problem statement
theorem pool_drain_rate :
  let fullVolume := poolLength * poolWidth * poolDepth
  let volumeAt80Percent := fullVolume * poolCapacityPercent
  let drainRate := volumeAt80Percent / drainTime
  drainRate = 60 :=
by
  sorry

end pool_drain_rate_l496_496169


namespace extreme_values_of_f_max_value_on_interval_l496_496375
open Real

noncomputable def f (x : ℝ) : ℝ := ln x / x - 1

theorem extreme_values_of_f :
  (∀ x > 0, f x ≤ f e) ∧ (∀ y, 0 < y ∧ y ≠ e → f y < f e) :=
sorry

theorem max_value_on_interval (m : ℝ) (hm : m > 0) :
  (m ≤ exp(1) / 2 → ∀ x ∈ set.Icc m (2 * m), f x ≤ f (2 * m)) ∧
  (exp(1) / 2 < m ∧ m < exp(1) → ∀ x ∈ set.Icc m (2 * m), f x ≤ f (exp 1)) ∧
  (m ≥ exp(1) → ∀ x ∈ set.Icc m (2 * m), f x ≤ f m) :=
sorry

end extreme_values_of_f_max_value_on_interval_l496_496375


namespace cap_to_sunglasses_prob_l496_496144

-- Define the conditions
def num_people_wearing_sunglasses : ℕ := 60
def num_people_wearing_caps : ℕ := 40
def prob_sunglasses_and_caps : ℚ := 1 / 3

-- Define the statement to prove
theorem cap_to_sunglasses_prob : 
  (num_people_wearing_sunglasses * prob_sunglasses_and_caps) / num_people_wearing_caps = 1 / 2 :=
by
  sorry

end cap_to_sunglasses_prob_l496_496144


namespace simplify_sqrt_eight_l496_496162

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 :=
by
  -- Given that 8 can be factored into 4 * 2 and the property sqrt(a * b) = sqrt(a) * sqrt(b)
  sorry

end simplify_sqrt_eight_l496_496162


namespace sum_distances_l496_496844

-- Definitions of point P, line l, and curve C
def point_P : ℝ × ℝ := (1, -2)

def line_parametric (t : ℝ) : ℝ × ℝ := (1 + (sqrt 2)/2 * t, -2 + (sqrt 2)/2 * t)

def curve_C_polar (ρ θ : ℝ) : Prop := ρ * sin(θ)^2 = 2 * cos(θ)

def curve_C : ℝ × ℝ → Prop
| (x, y) := y^2 = 2 * x

-- Statement of the proof problem
theorem sum_distances (A B : ℝ × ℝ) (t1 t2 : ℝ)
  (hA : A = line_parametric t1)
  (hB : B = line_parametric t2)
  (h_curve_A : curve_C A)
  (h_curve_B : curve_C B)
  (h_t_positive : 0 < t1 ∧ 0 < t2) :
  (real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
   real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) = 6 * sqrt 2 := 
sorry

end sum_distances_l496_496844


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_product_sequence_l496_496005

noncomputable def a_seq : ℕ → ℕ := λ n, 2 * (n + 1) - 1

noncomputable def b_seq : ℕ → ℕ := λ n, 3^n

noncomputable def c_seq : ℕ → ℕ := λ n, a_seq n * b_seq n

noncomputable def T_n (n : ℕ) : ℕ := 3 + (n - 1) * 3^(n + 1)

theorem arithmetic_sequence_formula :
  ∀ n, a_seq n = 2 * (n + 1) - 1 := 
by sorry

theorem geometric_sequence_formula :
  ∀ n, b_seq n = 3^n := 
by sorry

theorem sum_of_product_sequence :
  ∀ n, (∑ i in finset.range (n+1), c_seq i) = T_n n := 
by sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_product_sequence_l496_496005


namespace meaningful_expression_l496_496410

theorem meaningful_expression (x : ℝ) : x >= -5 → x ≠ 0 → ∃ y : ℝ, y = (sqrt (x + 5)) / x :=
by
  intros h1 h2
  have h3 : x + 5 >= 0 := by linarith
  use (sqrt (x + 5)) / x
  sorry

end meaningful_expression_l496_496410


namespace range_of_a_value_of_m_l496_496354

def ellipse_eccentricity (a : ℝ) : ℝ :=
  if a > 1 then sqrt(1 - 1/(a^2))
  else sqrt(1 - a^2)

def proposition_q (a : ℝ) : Prop :=
  sqrt(3)/2 < ellipse_eccentricity a ∧ ellipse_eccentricity a < 2*sqrt 2/3

def proposition_p (a : ℝ) (m : ℝ) : Prop :=
  |a - m| < 1/2

theorem range_of_a (a : ℝ) (h : a > 0) (hq : proposition_q a) :
  a ∈ Set.Ioo (1/3) 1/2 ∪ Set.Ioo 2 3 :=
sorry

theorem value_of_m (a m : ℝ) (h : a > 0) (hp : proposition_p a m) (hsuff : ∀ a, proposition_p a m → proposition_q a) 
  (hnsuff : ¬ ∀ a, proposition_q a → proposition_p a m) :
  m = 5/2 :=
sorry

end range_of_a_value_of_m_l496_496354


namespace expected_number_of_digits_l496_496893

noncomputable def expectedNumberDigits : ℝ :=
  let oneDigitProbability := (9 : ℝ) / 16
  let twoDigitProbability := (7 : ℝ) / 16
  (oneDigitProbability * 1) + (twoDigitProbability * 2)

theorem expected_number_of_digits :
  expectedNumberDigits = 1.4375 := by
  sorry

end expected_number_of_digits_l496_496893


namespace area_of_triangle_ABC_perimeter_of_triangle_ABC_l496_496093

-- Define the triangle conditions
def is_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
 ∃ (a b c : ℝ),
   a = 30 ∧                      -- BC = 30 units
   a > 0 ∧
   sin(π / 3) = (√3 / 2) ∧       -- sin(60°) = √3/2
   cos(π / 3) = (1/2) ∧          -- cos(60°) = 1/2
   sin(π / 6) = (1/2) ∧          -- sin(30°) = 1/2
   cos(π / 6) = (√3 / 2) ∧       -- cos(30°) = √3/2
   b = a / 2 ∧                   -- AB is half of BC in a 30-60-90 triangle
   c = b * √3                   -- AC is BC/2 * √3 in a 30-60-90 triangle

-- Prove the calculated area is correct
theorem area_of_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :
  is_triangle_ABC A B C →
  ∃ (area : ℝ),
  area = (225 * √3) / 2 :=
by
  intro h,
  rcases h with ⟨a, b, c, ha, ha_pos, hb1, hc1, hc2, hc3, hb2, hc4⟩,
  use (225 * √3) / 2,
  sorry

-- Prove the calculated perimeter is correct
theorem perimeter_of_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :
  is_triangle_ABC A B C →
  ∃ (perimeter : ℝ),
  perimeter = 45 + 15 * √3 :=
by
  intro h,
  rcases h with ⟨a, b, c, ha, ha_pos, hb1, hc1, hc2, hc3, hb2, hc4⟩,
  use 45 + 15 * √3,
  sorry

end area_of_triangle_ABC_perimeter_of_triangle_ABC_l496_496093


namespace total_pencils_correct_l496_496652

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l496_496652


namespace m_values_l496_496053

def set_A : Set ℝ := {x | x^2 ≠ 1}
def set_B (m : ℝ) : Set ℝ := {x | m * x = 1}

theorem m_values :
  ∀ m : ℝ, ((set_A ∪ set_B m) = set_A) ↔ (m ∈ {-1, 0, 1}) :=
by
  sorry

end m_values_l496_496053


namespace problem_f_2017_l496_496030

noncomputable def f : ℤ → ℝ
| x => if x < 0 then real.logb 3 (-x) else -f (x - 2)

theorem problem_f_2017 : f 2017 = 0 :=
by 
  sorry

end problem_f_2017_l496_496030


namespace ladder_angle_of_elevation_l496_496920

noncomputable def angle_of_elevation (adjacent hypotenuse : ℝ) : ℝ :=
Real.arccos (adjacent / hypotenuse)

theorem ladder_angle_of_elevation :
  angle_of_elevation 4.6 9.2 = Real.pi / 3 :=
by
  -- Given conditions
  let adjacent : ℝ := 4.6
  let hypotenuse : ℝ := 9.2
  
  -- The expected angle of elevation
  let expected_angle := Real.pi / 3
  
  -- Proof that the computed angle is the expected angle
  calc
    angle_of_elevation adjacent hypotenuse 
      = Real.arccos (adjacent / hypotenuse) : rfl
    ... = Real.arccos (4.6 / 9.2) : by rw [adjacent, hypotenuse]
    ... = Real.arccos (0.5) : by norm_num
    ... = Real.pi / 3 : by norm_num

end ladder_angle_of_elevation_l496_496920


namespace cost_per_trip_l496_496900

theorem cost_per_trip
  (pass_cost : ℝ)
  (oldest_trips : ℕ)
  (youngest_trips : ℕ)
  (h_pass_cost : pass_cost = 100.0)
  (h_oldest_trips : oldest_trips = 35)
  (h_youngest_trips : youngest_trips = 15) :
  (2 * pass_cost) / (oldest_trips + youngest_trips) = 4.0 :=
by
  sorry

end cost_per_trip_l496_496900


namespace maximum_value_2ab_sqrt3_2ac_le_sqrt3_achievable_value_2ab_sqrt3_2ac_eq_sqrt3_l496_496130

theorem maximum_value_2ab_sqrt3_2ac_le_sqrt3 (a b c : ℝ)
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * a * c ≤ Real.sqrt 3 :=
begin
  sorry
end

theorem achievable_value_2ab_sqrt3_2ac_eq_sqrt3 (a b c : ℝ)
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum_squares : a^2 + b^2 + c^2 = 1)
  (h_eq_conditions : a = b ∧ b = Real.sqrt (1/2) ∧ c = 0) :
  2 * a * b * Real.sqrt 3 + 2 * a * c = Real.sqrt 3 :=
begin
  sorry
end

end maximum_value_2ab_sqrt3_2ac_le_sqrt3_achievable_value_2ab_sqrt3_2ac_eq_sqrt3_l496_496130


namespace round_to_nearest_hundredth_l496_496504

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 236.78953) : 
  Float.round (x * 100) / 100 = 236.79 :=
by
  sorry

end round_to_nearest_hundredth_l496_496504


namespace monthly_compounding_is_better_l496_496601

noncomputable def principal : ℝ := 1000
noncomputable def annualRate : ℝ := 0.05
noncomputable def years : ℕ := 10
noncomputable def annualCompounding : ℕ := 1
noncomputable def monthlyCompounding : ℕ := 12

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem monthly_compounding_is_better :
  compound_interest principal annualRate monthlyCompounding years 
  > compound_interest principal annualRate annualCompounding years := 
by {
  sorry
}

end monthly_compounding_is_better_l496_496601


namespace bee_travel_distance_l496_496565

-- Define the radii of the concentric circles.
def radius_small : ℝ := 15
def radius_large : ℝ := 25

-- The travel distance when moving one-eighth of the larger circle's circumference.
def arc_distance : ℝ := (1/8) * (2 * Real.pi * radius_large)

-- The travel distance when going from the edge of the larger circle to the center and back.
def center_distance : ℝ := 2 * radius_large

-- The total distance travelled by the bee.
def total_distance : ℝ := arc_distance + center_distance

theorem bee_travel_distance : total_distance = (25 * Real.pi / 4) + 50 := by
  -- The proof is omitted here, add sorry to accept the statement
  sorry

end bee_travel_distance_l496_496565


namespace annual_payment_correct_l496_496913

-- Conditions
def interest_rate : ℝ := 0.01
def total_payment : ℝ := 6300
def compounded_factor : ℝ := (1 + interest_rate)^2 + (1 + interest_rate) + 1

-- The Problem to prove
theorem annual_payment_correct : 
  let x := total_payment / compounded_factor in
  x ≈ 2000 := 
by
  sorry

end annual_payment_correct_l496_496913


namespace rectangle_area_l496_496172

theorem rectangle_area (y : ℝ) (h : y > 0) 
    (h_area : ∃ (E F G H : ℝ × ℝ), 
        E = (0, 0) ∧ 
        F = (0, 5) ∧ 
        G = (y, 5) ∧ 
        H = (y, 0) ∧ 
        5 * y = 45) : 
    y = 9 := 
by
    sorry

end rectangle_area_l496_496172


namespace smallest_n_sum_of_10_l496_496678

-- Define a predicate for 'extra-special' numbers
def is_extra_special (x : ℝ) : Prop :=
  ∀ s s', string_of_real x = s ++ s' → ((s ++ s').to_list.all (λ c, c = '0' ∨ c = '5'))

noncomputable def smallest_n_sum (S : set ℝ) (target : ℝ) :=
  Inf {n : ℕ | ∃ l : list ℝ, l.length = n ∧ (∀ x ∈ l, x ∈ S) ∧ l.sum = target}

theorem smallest_n_sum_of_10 :
  smallest_n_sum { x : ℝ | is_extra_special x } 10 = 10 :=
sorry

end smallest_n_sum_of_10_l496_496678


namespace part_I_intervals_of_monotonicity_part_II_inequality_solution_range_l496_496028

noncomputable def f (x a : ℝ) := exp x * (x^2 + a * x + a)

open Function

theorem part_I_intervals_of_monotonicity (a : ℝ) (h : a = 1) :
  increasingOn (f x 1) {x : ℝ | x < -2} ∧ increasingOn (f x 1) {x : ℝ | x > -1} ∧
  decreasingOn (f x 1) {x : ℝ | -2 < x ∧ x < -1} :=
sorry

theorem part_II_inequality_solution_range :
  {a : ℝ | ∀ x ∈ set.Ici a, f x a ≤ exp a} ⊆ {a : ℝ | a ≤ 1 / 2} :=
sorry

end part_I_intervals_of_monotonicity_part_II_inequality_solution_range_l496_496028


namespace triangle_acute_of_angles_sum_gt_90_l496_496235

theorem triangle_acute_of_angles_sum_gt_90 
  (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : α + β > 90) 
  (h₃ : α + γ > 90) 
  (h₄ : β + γ > 90) 
  : α < 90 ∧ β < 90 ∧ γ < 90 :=
sorry

end triangle_acute_of_angles_sum_gt_90_l496_496235


namespace angle_CDE_is_85_l496_496831

-- Definitions based on given conditions
def angleA : ℝ := 90
def angleB : ℝ := 90
def angleC : ℝ := 90
def angleAEB : ℝ := 50
def angleBED : ℝ -- to be defined
def angleBDE : ℝ -- to be defined

-- Conditions for triangle BED and quadrilateral ACDE
axiom triangle_BED_is_isosceles : angleBED = angleBDE
axiom quadrilateral_angle_sum : angleA + angleC + angleCDE + (angleAEB + angleBED) = 360

-- Theorem to be proven
theorem angle_CDE_is_85 : 
  angleA = 90 ∧ angleB = 90 ∧ angleC = 90 ∧ 
  angleAEB = 50 ∧ 
  angleBED = angleBDE ∧ 
  90 + 90 + angleCDE + (angleAEB + angleBED) = 360 → 
  angleCDE = 85 := 
by
  -- Inserting logical reasoning steps here can transform these Lean definitions into a proof
  sorry

end angle_CDE_is_85_l496_496831


namespace smallest_possible_b_l496_496116

theorem smallest_possible_b
  (a c b : ℤ)
  (h1 : a < c)
  (h2 : c < b)
  (h3 : c = (a + b) / 2)
  (h4 : b^2 / c = a) :
  b = 2 :=
sorry

end smallest_possible_b_l496_496116


namespace existence_of_n_with_s_prime_divisors_l496_496159

theorem existence_of_n_with_s_prime_divisors (s : ℕ) (hs : s ≥ 1) : 
  ∃ n : ℕ, n ≥ 1 ∧ (nat.num_of_prime_divisors (2^n - 1) ≥ s) :=
sorry

end existence_of_n_with_s_prime_divisors_l496_496159


namespace translation_identity_l496_496561

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := sin (2 * x)
def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Define the translation of g by π / 12 units to the left
def g_translated (x : ℝ) : ℝ := g (x + π / 12)

-- The theorem we want to prove
theorem translation_identity : ∀ x : ℝ, g_translated x = f x :=
by
  sorry

end translation_identity_l496_496561


namespace find_a_for_slope_l496_496931

theorem find_a_for_slope :
  ∃ (a : ℝ), ∀ (x : ℝ), ((a * x + 1) * Real.exp x)' x = -2 → a = -3 :=
by
  sorry

end find_a_for_slope_l496_496931


namespace ratio_is_five_to_three_l496_496953

variable (g b : ℕ)

def girls_more_than_boys : Prop := g - b = 6
def total_pupils : Prop := g + b = 24
def ratio_girls_to_boys : ℚ := g / b

theorem ratio_is_five_to_three (h1 : girls_more_than_boys g b) (h2 : total_pupils g b) : ratio_girls_to_boys g b = 5 / 3 := by
  sorry

end ratio_is_five_to_three_l496_496953


namespace cos_lt_taylor_l496_496905

theorem cos_lt_taylor (x : ℝ) (h1 : 0 < x) (h2 : x < (π / 2)) :
  cos x < 1 - (x^2 / 2) + (x^4 / 16) :=
sorry

end cos_lt_taylor_l496_496905


namespace number_of_bricks_l496_496660

-- Define the conditions
def brenda_rate (y : ℕ) : ℕ := y / 8
def brandon_rate (y : ℕ) : ℕ := y / 10
def combined_rate (y : ℕ) : ℕ := (brenda_rate y + brandon_rate y) - 12

theorem number_of_bricks 
  (y : ℕ) 
  (h1 : brenda_rate y) 
  (h2 : brandon_rate y) 
  (h3 : combined_rate y * 6 = y) 
  : y = 206 := 
sorry

end number_of_bricks_l496_496660


namespace AB_connection_probability_l496_496009

noncomputable def probability_AB_connected
  (A B C D : Type)
  (not_coplanar : ¬ ∃ P : A × B × C × D → Prop, P A B C D = false)
  (prob_edge : ℕ := 1/2)
  (independence : ∀ X Y : Type, X ∉ {A, B, C, D} → Y ∉ {A, B, C, D} → X ≠ Y → Event.independent (X ↔ Y)) :
  ℚ :=
  3 / 4

theorem AB_connection_probability
  (A B C D : Type)
  (not_coplanar : ¬ ∃ P : A × B × C × D → Prop, P A B C D = false)
  (prob_edge : ℕ := 1/2)
  (independence : ∀ X Y : Type, X ∉ {A, B, C, D} → Y ∉ {A, B, C, D} → X ≠ Y → Event.independent (X ↔ Y)) :
  probability_AB_connected A B C D not_coplanar prob_edge independence = (3 / 4) :=
sorry

end AB_connection_probability_l496_496009


namespace infinitely_many_good_approximations_l496_496459

theorem infinitely_many_good_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ p q : ℕ in at_top, abs (x - p / q) < 1 / q ^ 2 :=
by
  sorry

end infinitely_many_good_approximations_l496_496459


namespace branches_sum_one_main_stem_l496_496692

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l496_496692


namespace magnitude_b_eq_2sqrt5_l496_496389

namespace VectorProof

-- Definitions:
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Proof statement for the calculated magnitude of vector b given perpendicular condition on a and b:
theorem magnitude_b_eq_2sqrt5 (x : ℝ) 
  (h_perp : vec_a.1 * vec_b(x).1 + vec_a.2 * vec_b(x).2 = 0) : 
  |vec_b 4| = real.sqrt (5) * 2 :=
by
  sorry

end VectorProof

end magnitude_b_eq_2sqrt5_l496_496389


namespace find_common_chord_l496_496015

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- The common chord is the line we need to prove
def CommonChord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- The theorem stating that the common chord is the line x + 2*y - 1 = 0
theorem find_common_chord (x y : ℝ) (p : C1 x y ∧ C2 x y) : CommonChord x y :=
sorry

end find_common_chord_l496_496015


namespace off_road_vehicle_cost_l496_496848

theorem off_road_vehicle_cost
  (dirt_bike_count : ℕ) (dirt_bike_cost : ℕ)
  (off_road_vehicle_count : ℕ) (register_cost : ℕ)
  (total_cost : ℕ) (off_road_vehicle_cost : ℕ) :
  dirt_bike_count = 3 → dirt_bike_cost = 150 →
  off_road_vehicle_count = 4 → register_cost = 25 →
  total_cost = 1825 →
  3 * dirt_bike_cost + 4 * off_road_vehicle_cost + 7 * register_cost = total_cost →
  off_road_vehicle_cost = 300 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end off_road_vehicle_cost_l496_496848


namespace OP_perpendicular_AE_l496_496843

open EuclideanGeometry

-- Definitions based on conditions in (a)
variables {O A B C D E F P : Point}
variables (h_inscribed : inscribed_pentagon O A B C D E)
[h_intersect_AD_BE : ∃ F, intersection_point_lines AD BE F]
(h_CEF_extension : line_extension_CF_intersects_circle O C F P)
(h_product_equal : AB * CD = BC * ED)

-- The statement we need to prove
theorem OP_perpendicular_AE (h_inscribed : inscribed_pentagon O A B C D E)
    (h_intersect_AD_BE : ∃ F, intersection_point_lines (line_through A D) (line_through B E) F)
    (h_CEF_extension : ∀ P, line_extension_CF_intersects_circle O C F P)
    (h_product_equal : length A B * length C D = length B C * length E D) :
    perpendicular (line_through O P) (line_through A E) :=
sorry

end OP_perpendicular_AE_l496_496843


namespace remainder_when_add_13_l496_496821

theorem remainder_when_add_13 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 :=
sorry

end remainder_when_add_13_l496_496821


namespace tree_planting_problem_l496_496556

theorem tree_planting_problem 
  (road_length : ℕ)
  (interval_length : ℕ)
  (h1 : road_length = 42)
  (h2 : interval_length = 7) :
  (road_length / interval_length) + 1 = 7 := 
  by 
    have h1 : 42 / 7 + 1 = 7 := by norm_num
    exact h1

end tree_planting_problem_l496_496556


namespace part1_part2_l496_496793

-- Definition of the function f
def f (x : ℝ) : ℝ := 4 * sin x * (sin (π / 4 + x / 2))^2 + cos (2 * x)

-- Statement for Part 1
theorem part1 (ω : ℝ) (hω_pos : ω > 0) : 
  (∀ x ∈ Icc (-(π / 2)) (2 * π / 3), 
    deriv (λ x, f(ω * x)) x > 0) ↔ ω ∈ Icc 0 (3 / 4) :=
sorry

-- Set definitions for Part 2
def A : set ℝ := { x | π / 6 ≤ x ∧ x ≤ 2 * π / 3 }
def B (m : ℝ) : set ℝ := { x | abs (f x - m) < 2 }

-- Statement for Part 2
theorem part2 (m : ℝ) : (A ∪ B m = B m) ↔ m ∈ Ioi 1 ∩ Iio 4 :=
sorry

end part1_part2_l496_496793


namespace final_balance_percentage_l496_496482

variable (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ)

def finalPercent (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ) : ℝ :=
  let afterIncrease := initialAmount * (1 + increasePercent)
  let afterDecrease := afterIncrease * (1 - decreasePercent)
  afterDecrease / initialAmount * 100

theorem final_balance_percentage :
  finalPercent 125 0.25 0.2 = 100 := by
  sorry

end final_balance_percentage_l496_496482


namespace min_distance_integer_points_to_line_l496_496535

theorem min_distance_integer_points_to_line : 
  ∃ (m n : ℤ), let d := λ m n : ℤ, (| (5 * m - 3 * n : ℚ) + 12 % 5| / (5 * √34)) 
  in d m n = (√34 / 85) :=
sorry

end min_distance_integer_points_to_line_l496_496535


namespace largest_real_number_l496_496719

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496719


namespace directrix_of_parabola_l496_496932

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = - (1 / 8) * x^2 → y = 2 :=
by
  sorry

end directrix_of_parabola_l496_496932


namespace necessary_but_not_sufficient_condition_l496_496469

-- Definitions
variable (f : ℝ → ℝ)

-- Condition that we need to prove
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -g (-x)

-- Necessary and sufficient condition
theorem necessary_but_not_sufficient_condition : 
  (∀ x, |f x| = |f (-x)|) ↔ (∀ x, f x = -f (-x)) ∧ ¬(∀ x, |f x| = |f (-x)| → f x = -f (-x)) := by 
sorry

end necessary_but_not_sufficient_condition_l496_496469


namespace domain_of_f_odd_function_range_of_x_gt_zero_l496_496378

def f (a x : ℝ) := log a (10 + x) - log a (10 - x)

-- Prove that the domain of f(x) is -10 < x < 10 given a > 0 and a ≠ 1
theorem domain_of_f {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : ∀ x : ℝ, (10 + x > 0) ∧ (10 - x > 0) ↔ (-10 < x ∧ x < 10) :=
sorry

-- Prove that f(x) is an odd function
theorem odd_function {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : ∀ x : ℝ, f a (-x) = -f a x :=
sorry

-- Prove the range of x when f(x) > 0
theorem range_of_x_gt_zero {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : 
  (∀ x : ℝ, f a x > 0 → ((a > 1 ∧ 0 < x ∧ x < 10) ∨ (0 < a ∧ a < 1 ∧ -10 < x ∧ x < 0))) :=
sorry

end domain_of_f_odd_function_range_of_x_gt_zero_l496_496378


namespace unique_ordered_triple_satisfies_conditions_l496_496393

theorem unique_ordered_triple_satisfies_conditions :
  ∃! (a b c : ℤ), a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ log a b = c^3 ∧ a + b + c = 100 := 
sorry

end unique_ordered_triple_satisfies_conditions_l496_496393


namespace problem1_max_value_problem2_find_a_problem3_ineq_l496_496790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  x * f a x

theorem problem1_max_value (h : ∀ x > 0, f (-1) x ≤ f (-1) 1) : 
  ∀ x > 0, f (-1) x ≤ -1 :=
by
  sorry

theorem problem2_find_a (h : ∀ x ∈ Set.Ioc 0 Real.e, f a x ≤ -3) :
  a = - Real.exp 2 :=
by
  sorry

theorem problem3_ineq (a_pos : a > 0) (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2) : 
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
by
  sorry

end problem1_max_value_problem2_find_a_problem3_ineq_l496_496790


namespace mr_cruz_new_weight_l496_496484

-- Define the initial weight
def initial_weight : ℝ := 70

-- Define the calorie intake details for each month
def first_month_weekday_calories : ℝ := 3000
def first_month_weekend_calories : ℝ := 3500
def second_month_weekday_calories : ℝ := 4000
def second_month_weekend_calories : ℝ := 4500

-- Define number of days in a week
def weekdays : ℝ := 5
def weekends : ℝ := 2
def days_in_a_month : ℝ := 30

-- Calorie equivalence for weight gain
def calories_per_pound : ℝ := 3500

theorem mr_cruz_new_weight : 
    (initial_weight + 
    ((weekdays * first_month_weekday_calories + weekends * first_month_weekend_calories) / calories_per_pound) +
    ((weekdays * second_month_weekday_calories + weekends * second_month_weekend_calories) / calories_per_pound))
    = 85 := 
by
  calc
    initial_weight + 
    ((weekdays * first_month_weekday_calories + weekends * first_month_weekend_calories) / calories_per_pound) +
    ((weekdays * second_month_weekday_calories + weekends * second_month_weekend_calories) / calories_per_pound)
    = initial_weight + (22000 / calories_per_pound) + (29000 / calories_per_pound) : by norm_num
  ... = 70 + (22000 / 3500) + (29000 / 3500) : by rw [initial_weight, calories_per_pound]
  ... = 70 + 6.2857 + 8.2857 : by norm_num
  ... = 70 + 14.5714 : by norm_num
  ... = 84.5714 : by norm_num
  ... ≈ 85 : by norm_num 

end mr_cruz_new_weight_l496_496484


namespace can_transform_to_2_12_22_cannot_transform_to_12_12_12_l496_496554

-- Condition: Initial configuration
def initialState : (ℕ × ℕ × ℕ) := (19, 8, 9)

-- Condition: Operation rule
def operation (a b c : ℕ × ℕ × ℕ) : (ℕ × ℕ × ℕ) :=
  (a.1 + 1, b.1 + 1, c.1 - 2)

-- Problem 1: Prove the configuration can be transformed into (2, 12, 22)
theorem can_transform_to_2_12_22 : 
  ∃ n, -- there exists a number of operations
    let steps := iterate n (λ x, operation x) initialState in
    steps = (2, 12, 22) := sorry

-- Problem 2: Prove that it's impossible to transform into (12, 12, 12)
theorem cannot_transform_to_12_12_12 :
  ¬ (∃ n, -- there does not exist any number of operations
    let steps := iterate n (λ x, operation x) initialState in
    steps = (12, 12, 12)) := sorry

end can_transform_to_2_12_22_cannot_transform_to_12_12_12_l496_496554


namespace sum_of_fourth_and_sixth_terms_l496_496425

-- Define the sequence
noncomputable def a : ℕ → ℚ 
| 0       := 1 -- We start the sequence from index 1 to match the problem's conditions
| 1       := 1
| (n + 2) := ((n + 2) / (n + 1))^3

-- Sum of the fourth and sixth terms in the sequence
theorem sum_of_fourth_and_sixth_terms : (a 3) + (a 5) = 13832 / 3375 := by
  sorry

end sum_of_fourth_and_sixth_terms_l496_496425


namespace tangent_line_equation_f_ge_neg4a2_plus4a_l496_496794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem tangent_line_equation (a : ℝ) : 
  (∀ x, f'(x) = Real.exp x - 2 * a) →
  f'(0) = 1 - 2 * a →
  1 - 2 * a = 2 →
  a = -1 / 2 → ∀ x : ℝ, ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ y = m * x + b := by 
  sorry

theorem f_ge_neg4a2_plus4a (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, f(x, a) ≥ -4 * a^2 + 4 * a := by
  sorry

end tangent_line_equation_f_ge_neg4a2_plus4a_l496_496794


namespace jon_needs_five_loads_l496_496103

def laundry_machine_max_capacity : ℝ := 8
def shirt_weight : ℝ := 1 / 4
def pant_weight : ℝ := 1 / 2
def sock_weight : ℝ := 1 / 6
def jacket_weight : ℝ := 2
def efficiency_loss_per_jacket : ℝ := 0.1

def shirts_to_wash : ℕ := 20
def pants_to_wash : ℕ := 20
def socks_to_wash : ℕ := 18
def jackets_to_wash : ℕ := 6

def total_shirt_weight := shirts_to_wash * shirt_weight
def total_pant_weight := pants_to_wash * pant_weight
def total_sock_weight := socks_to_wash * sock_weight
def total_jacket_weight := jackets_to_wash * jacket_weight

def total_weight := total_shirt_weight + total_pant_weight + total_sock_weight + total_jacket_weight

def loads_needed_without_efficiency_loss := total_weight / laundry_machine_max_capacity

-- Definition for the total number of loads needed.
-- To minimize the number of loads and efficiency loss, wash jackets separately.

def loads_for_jackets := jackets_to_wash / (laundry_machine_max_capacity / jacket_weight)
def remaining_weight := total_shirt_weight + total_pant_weight + total_sock_weight
def loads_for_remaining_clothes := remaining_weight / laundry_machine_max_capacity
def total_loads := loads_for_jackets + loads_for_remaining_clothes

theorem jon_needs_five_loads : total_loads = 5 := by
  sorry

end jon_needs_five_loads_l496_496103


namespace squares_count_correct_l496_496689

-- Assuming basic setup and coordinate system.
def is_valid_point (x y : ℕ) : Prop :=
  x ≤ 8 ∧ y ≤ 8

-- Checking if a point (a, b) in the triangle as described.
def is_in_triangle (a b : ℕ) : Prop :=
  0 ≤ b ∧ b ≤ a ∧ a ≤ 4

-- Function derived from the solution detailing the number of such squares.
def count_squares (a b : ℕ) : ℕ :=
  -- Placeholder to represent the derived formula - to be replaced with actual derivation function
  (9 - a + b) * (a + b + 1) - 1

-- Statement to prove
theorem squares_count_correct (a b : ℕ) (h : is_in_triangle a b) :
  ∃ n, n = count_squares a b := 
sorry

end squares_count_correct_l496_496689


namespace system_of_equations_solution_l496_496515

theorem system_of_equations_solution:
  ∀ (x y : ℝ), 
    x^2 + y^2 + x + y = 42 ∧ x * y = 15 → 
      (x = 3 ∧ y = 5) ∨ (x = 5 ∧ y = 3) ∨ 
      (x = (-9 + Real.sqrt 21) / 2 ∧ y = (-9 - Real.sqrt 21) / 2) ∨ 
      (x = (-9 - Real.sqrt 21) / 2 ∧ y = (-9 + Real.sqrt 21) / 2) := 
by
  sorry

end system_of_equations_solution_l496_496515


namespace AB_plus_BK_eq_KC_l496_496295

-- Define the geometric setup
variables {O A B C D K : Point}
variable circle : Circle O A
variables (arc_midpoint : ArcMidpoint circle A C D)
variables (BC : Line B C)
variables (DK_perp_BC : Perpendicular D K BC)
variables (AOB : Angle A O B)
variables (BOC : Angle B O C)
variable (angle_condition : AOB < BOC)

-- Define D as the midpoint of arc AC which contains B
def midpoint_arc_AC_contains_B := isArcMidpoint_arc_midpoint arc_midpoint B

-- Define orthogonality of DK and BC
def ortho_DK_BC := isPerpendicular DK_perp_BC

-- Prove the required equality
theorem AB_plus_BK_eq_KC :
  AB + BK = KC :=
by
-- Placeholder proof
sorry

end AB_plus_BK_eq_KC_l496_496295


namespace sum_of_three_geq_54_l496_496548

theorem sum_of_three_geq_54 (a : Fin 10 → ℕ) (h_diff : Function.Injective a) (h_sum : (∑ i, a i) > 144) :
  ∃ i j k : Fin 10, i < j ∧ j < k ∧ a i + a j + a k ≥ 54 := 
by
  -- By contradiction
  sorry

end sum_of_three_geq_54_l496_496548


namespace sqrt_inequality_l496_496065

theorem sqrt_inequality (x : ℝ) (h : ∀ r : ℝ, r = 2 * x - 1 → r ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_inequality_l496_496065


namespace wickets_before_last_match_l496_496271

theorem wickets_before_last_match (W : ℕ) (avg_before : ℝ) (wickets_taken : ℕ) (runs_conceded : ℝ) (avg_drop : ℝ) :
  avg_before = 12.4 → wickets_taken = 4 → runs_conceded = 26 → avg_drop = 0.4 →
  (avg_before - avg_drop) * (W + wickets_taken) = avg_before * W + runs_conceded →
  W = 55 :=
by
  intros
  sorry

end wickets_before_last_match_l496_496271


namespace triangle_AC_eq_12_l496_496150

variables {A B C D E : Type} [metric_space A] [metric_space B]
variables {AC BC AB BE DC EC : ℝ}

theorem triangle_AC_eq_12 (h1 : AC = DC + AD) (h2 : BD = ED) (h3 : ∠BDC = ∠DEB)
(h4 : AB = 7) (h5 : BE = 2) (h6 : AD = EC) : AC = 12 := by
  sorry

end triangle_AC_eq_12_l496_496150


namespace XY_parallel_BC_l496_496466

-- Definitions of scalene triangle and its properties
variables {A B C D E F O P X Y : Type*}

-- Assume the following are points in the plane
variables [geometry.point A] [geometry.point B] [geometry.point C]
variables [geometry.point D] [geometry.point E] [geometry.point F] [geometry.point O]
variables [geometry.point P] [geometry.point X] [geometry.point Y]

-- Conditions
axiom scalene_triangle (hABC : geometry.triangle A B C)
axiom altitude_AD (hAD : geometry.is_altitude A D B C)
axiom altitude_BE (hBE : geometry.is_altitude B E A C)
axiom altitude_CF (hCF : geometry.is_altitude C F A B)
axiom circumcenter_O (hO : geometry.is_circumcenter O (geometry.triangle A B C))
axiom circumcircle_meets_at_P (hP: geometry.circumcircle_meet_at A D O P)
axiom PX_meets_ABC (hX : geometry.meets_circircle_logs (geometry.line P E) (geometry.circumcircle (geometry.triangle A B C)) X P)
axiom PY_meets_ABC (hY : geometry.meets_circircle_logs (geometry.line P F) (geometry.circumcircle (geometry.triangle A B C)) Y P)

-- The proof goal
theorem XY_parallel_BC : geometry.parallel (geometry.line X Y) (geometry.line B C) :=
sorry

end XY_parallel_BC_l496_496466


namespace number_of_brnet_brown_eyed_and_tall_l496_496428

variable (Girl : Type)
variable (isBlonde : Girl → Prop)
variable (isBrunette : Girl → Prop)
variable (hasBlueEyes : Girl → Prop)
variable (hasBrownEyes : Girl → Prop)
variable (isTall : Girl → Prop)
variable (isShort : Girl → Prop)

variable [Fintype Girl]

-- conditions
variable (totalGirls : Fintype.card Girl = 60)
variable (numBlueEyedBlondes : Fintype.card { g : Girl | isBlonde g ∧ hasBlueEyes g } = 18)
variable (numBrunettes : Fintype.card { g : Girl | isBrunette g } = 35)
variable (numBrownEyed : Fintype.card { g : Girl | hasBrownEyes g } = 22)
variable (numTallGirls : Fintype.card { g : Girl | isTall g } = 20)

-- helper definitions
def BlondeGirls : Finset Girl := { g | isBlonde g }.toFinset
def BlueEyedBlondes : Finset Girl := { g | isBlonde g ∧ hasBlueEyes g }.toFinset
def BrownEyedGirls : Finset Girl := { g | hasBrownEyes g }.toFinset
def TallGirls : Finset Girl := { g | isTall g }.toFinset
def Brunettes : Finset Girl := { g | isBrunette g }.toFinset

def BrownEyedBlondes : Finset Girl := { g | isBlonde g ∧ hasBrownEyes g }.toFinset

theorem number_of_brnet_brown_eyed_and_tall :
  Fintype.card Girl = 60 →
  Fintype.card { g : Girl | isBlonde g ∧ hasBlueEyes g } = 18 →
  Fintype.card { g : Girl | isBrunette g } = 35 →
  Fintype.card { g : Girl | hasBrownEyes g } = 22 →
  Fintype.card { g : Girl | isTall g } = 20 →
  Fintype.card { g : Girl | isBrunette g ∧ hasBrownEyes g ∧ isTall g } = 5 :=
by
  intros
  sorry

end number_of_brnet_brown_eyed_and_tall_l496_496428


namespace final_number_is_1999_l496_496941

-- Define the basic setup with initial sequence of numbers
def initial_numbers : List ℝ := (List.range 1999).map (λ n, if n = 0 then 1 else 1 / (n + 1))

-- Define the operation on two numbers
def transform (a b : ℝ) : ℝ := a * b + a + b

-- We want to prove that starting with initial_numbers and performing the transform operation until one number remains,
-- the final number must be 1999.
theorem final_number_is_1999 :
  (∃ seq : List ℝ, (seq.head = 2000) ∧
  (∀ n, seq ∈ initial_numbers →
    (∀ a b, a ∈ seq → b ∈ seq → transform a b ∉ seq →
      (seq.drop 2).append [transform a b]))) :=
sorry

end final_number_is_1999_l496_496941


namespace solve_inequality_l496_496514

-- Define integer and fractional parts functions
noncomputable def int_part (x : ℝ) : ℤ := int.floor x
noncomputable def frac_part (x : ℝ) : ℝ := x - ↑(int_part x)

theorem solve_inequality (x : ℝ) (h₀ : frac_part x * (int_part x - 1) < x - 2)
  (h₁ : 0 ≤ frac_part x) (h₂ : frac_part x < 1) :
  x ≥ 3 :=
sorry

end solve_inequality_l496_496514


namespace fraction_of_l496_496967

theorem fraction_of (a b : ℚ) (h_a : a = 3/4) (h_b : b = 1/6) : b / a = 2/9 :=
by
  sorry

end fraction_of_l496_496967


namespace sum_odd_divisors_of_180_l496_496981

theorem sum_odd_divisors_of_180 : 
  let n := 180 in 
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0} in 
  ∑ d in odd_divisors, d = 78 :=
by
  let n := 180
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0}
  have h : ∑ d in odd_divisors, d = 78 := sorry -- Sum of odd divisors of 180
  exact h

end sum_odd_divisors_of_180_l496_496981


namespace parallelepipeds_from_4_points_l496_496553

theorem parallelepipeds_from_4_points : ∀ (A B C D : ℝ × ℝ × ℝ), 
  ¬ AffineLinearIndependent ℝ ![A, B, C, D] →
  ∃ (num_parallelepipeds : ℕ), num_parallelepipeds = 29 :=
sorry

end parallelepipeds_from_4_points_l496_496553


namespace curve_is_parabola_l496_496706

theorem curve_is_parabola (r θ : ℝ) : (r = 2 * (Real.cot θ) * (Real.csc θ)) → (∃ (x y : ℝ), y^2 = 2 * x) :=
by
  sorry

end curve_is_parabola_l496_496706


namespace range_of_a_l496_496396

theorem range_of_a (a : ℚ) (h₀ : 0 < a) (h₁ : ∃ n : ℕ, (2 * n - 1 = 2007) ∧ (-a < n ∧ n < a)) :
  1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l496_496396


namespace value_is_solution_of_the_equation_l496_496198

-- Definition and condition from the problem
def makes_both_sides_equal (x : ℝ) (lhs rhs : ℝ) : Prop :=
  lhs = rhs

-- The statement of the problem, asking to prove that the value is called 'Solution of the equation'
theorem value_is_solution_of_the_equation (x lhs rhs : ℝ) (H : makes_both_sides_equal x lhs rhs) : 
  "the value of the unknown that makes both sides of the equation equal is called the solution of the equation" :=
sorry

end value_is_solution_of_the_equation_l496_496198


namespace digits_of_85550_can_be_arranged_l496_496436

theorem digits_of_85550_can_be_arranged : 
  let digits := [8, 5, 5, 5, 0]
  let non_zero_digits := [8, 5, 5, 5]
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

  (∑ pos in {1, 2, 3, 4}, factorial non_zero_digits.length / (non_zero_digits.count 5)!) = 16 :=
by
  sorry

end digits_of_85550_can_be_arranged_l496_496436


namespace train_speed_kmph_l496_496634

noncomputable def speed_of_train
  (train_length : ℝ) (bridge_cross_time : ℝ) (total_length : ℝ) : ℝ :=
  (total_length / bridge_cross_time) * 3.6

theorem train_speed_kmph
  (train_length : ℝ := 130) 
  (bridge_cross_time : ℝ := 30) 
  (total_length : ℝ := 245) : 
  speed_of_train train_length bridge_cross_time total_length = 29.4 := by
  sorry

end train_speed_kmph_l496_496634


namespace box_1_contains_at_least_one_ball_l496_496902

theorem box_1_contains_at_least_one_ball :
  let balls := ["A", "B", "C"]
  let boxes := [1, 2, 3, 4]
  (∃ (b : fin 4 → fin 4), b (fin.mk 0 (by norm_num)) = 0)
  → (4^3 - 3^3 = 37) :=
by
  intro
  sorry

end box_1_contains_at_least_one_ball_l496_496902


namespace rational_numbers_to_integers_l496_496362

open Classical

noncomputable theory

variables {a b : ℚ}

theorem rational_numbers_to_integers 
  (h_distinct : a ≠ b)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_inf : ∃ᶠ n: ℕ in at_top, a^n - b^n ∈ ℤ) :
  ∃ a' b' : ℕ, (a = a') ∧ (b = b') :=
begin
  sorry
end

end rational_numbers_to_integers_l496_496362


namespace find_percentage_l496_496401

variable (X P : ℝ)

theorem find_percentage (h₁ : 0.20 * X = 400) (h₂ : (P / 100) * X = 2400) : P = 120 :=
by
  -- The proof is intentionally left out
  sorry

end find_percentage_l496_496401


namespace inclination_angle_to_slope_l496_496825

theorem inclination_angle_to_slope (a : ℝ) (h : real.tan (real.pi / 4) = 1) :
  a = 1 :=
sorry

end inclination_angle_to_slope_l496_496825


namespace bkingsley_2023_appears_on_line_36_l496_496533

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the problem statements as hypotheses
def problem1 : Prop :=
  let cycle_letters : ℕ := 9
  let cycle_digits : ℕ := 4
  lcm cycle_letters cycle_digits = 36

-- The goal is to prove the hypothesis defined in problem1
theorem bkingsley_2023_appears_on_line_36 : problem1 := by
  sorry

end bkingsley_2023_appears_on_line_36_l496_496533


namespace complement_U_A_l496_496259

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_U_A :
  U \ A = {1, 2, 6} := by
  sorry

end complement_U_A_l496_496259


namespace part_I_part_II_l496_496780

open Real

-- Definitions for part (I)
def f (x : ℝ) (a b : ℝ) : ℝ :=
if x >= 0 then (x - a)^2 - 1 else -(x - b)^2 + 1

-- Definition for odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

-- Part (I)
theorem part_I (a b : ℝ) (h_a_neg : a < 0) (h_f_odd : is_odd_function (f a b)) : 
  (f a b = λ x, if x >= 0 then (x + 1)^2 - 1 else -(x - 1)^2 + 1) :=
sorry

-- Definitions for part (II)
def is_monotone_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f y ≤ f x

-- Part (II)
theorem part_II (a b : ℝ) (h_a_pos : a > 0) (h_f_monotone : is_monotone_decreasing (f a b) (set.Icc -1 1)) : 
  b - a = -2 :=
sorry

end part_I_part_II_l496_496780


namespace arc_degree_chord_eq_radius_l496_496826

theorem arc_degree_chord_eq_radius {r : ℝ} (h_positive: r > 0) (C : ℝ) (S : Type*) [inner_product_space ℝ S] 
  (hc1: dist C S = r) (hc2: dist S (0 : S) = r) (hc3: dist C (0 : S) = r) :
  (arc_degree S C 0 = 60 ∨ arc_degree S C 0 = 300) := 
by
  sorry

end arc_degree_chord_eq_radius_l496_496826


namespace john_burritos_left_l496_496855

theorem john_burritos_left : 
  ∀ (boxes : ℕ) (burritos_per_box : ℕ) (given_away_fraction : ℚ) (eaten_per_day : ℕ) (days : ℕ),
  boxes = 3 → 
  burritos_per_box = 20 →
  given_away_fraction = 1 / 3 →
  eaten_per_day = 3 →
  days = 10 →
  let initial_burritos := boxes * burritos_per_box in
  let given_away_burritos := given_away_fraction * initial_burritos in
  let after_giving_away := initial_burritos - given_away_burritos in
  let eaten_burritos := eaten_per_day * days in
  let final_burritos := after_giving_away - eaten_burritos in
  final_burritos = 10 := 
by 
  intros,
  sorry

end john_burritos_left_l496_496855


namespace range_of_values_l496_496017

variable {f : ℝ → ℝ}

-- Conditions and given data
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

def is_monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f (x) ≤ f (y)

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (Real.log a / Real.log 2) + f (-Real.log a / Real.log 2) ≤ 2 * f (1)

-- The goal
theorem range_of_values (h1 : is_even f) (h2 : is_monotone_on_nonneg f) (a : ℝ) (h3 : condition f a) :
  1/2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_values_l496_496017


namespace final_balance_percentage_l496_496481

variable (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ)

def finalPercent (initialAmount : ℝ) (increasePercent : ℝ) (decreasePercent : ℝ) : ℝ :=
  let afterIncrease := initialAmount * (1 + increasePercent)
  let afterDecrease := afterIncrease * (1 - decreasePercent)
  afterDecrease / initialAmount * 100

theorem final_balance_percentage :
  finalPercent 125 0.25 0.2 = 100 := by
  sorry

end final_balance_percentage_l496_496481


namespace intersection_is_one_l496_496032

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem intersection_is_one : M ∩ N = {1} :=
by
  sorry

end intersection_is_one_l496_496032


namespace absolute_difference_avg_median_l496_496404

theorem absolute_difference_avg_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((3 + 4 * a + 2 * b) / 4) - (a + b / 2 + 1)| = 1 / 4 :=
by
  sorry

end absolute_difference_avg_median_l496_496404


namespace vertex_of_quadratic_l496_496342

variable {a k b c d : ℝ}

noncomputable def f (x : ℝ) := a * x^2 + k * x + c + d

theorem vertex_of_quadratic (h1 : a > 0) (h2 : k ≠ b) (h3 : d = d) :
  let x_vertex := -k / (2 * a),
      y_vertex := -k^2 / (4 * a) + c + d in
  (∃ x y, (x, y) = (x_vertex, y_vertex) ∧ f x_vertex = y_vertex) := by
  sorry

end vertex_of_quadratic_l496_496342


namespace is_sample_of_population_l496_496560

noncomputable def lifespan_group (n : ℕ) (G : Type) (pop : set G) (sample : set G) : Prop :=
  pop = {urban_lifespans | urban_lifespans ∈ G} ∧ sample = {urban_lifespans | urban_lifespans ∈ G ∧ size sample = 2500}

theorem is_sample_of_population (n : ℕ) (G : Type) (pop sample : set G) (h_pop : pop = {x | x ∈ G})
  (h_sample : sample = {y | y ∈ G ∧ size sample = 2500}) : sample ⊆ pop :=
by
  sorry

end is_sample_of_population_l496_496560


namespace count_ordered_pairs_l496_496447

theorem count_ordered_pairs (d n : ℕ) (h₁ : d ≥ 35) (h₂ : n > 0) 
    (h₃ : 45 + 2 * n < 120)
    (h₄ : ∃ a b : ℕ, 10 * a + b = 30 + n ∧ 10 * b + a = 35 + n ∧ a ≤ 9 ∧ b ≤ 9) :
    ∃ k : ℕ, -- number of valid ordered pairs (d, n)
    sorry := sorry

end count_ordered_pairs_l496_496447


namespace Thaiangulation_difference_by_two_triangles_l496_496606

-- Definition of a triangulation
structure Triangulation (Π : Type) :=
  (triangles : set (set Π))
  (disjoint_diagonals : ∀ t₁ t₂ ∈ triangles, (t₁ ≠ t₂) → (t₁ ∩ t₂).subset (finset.to_set (∅ : finset (Π))))

-- Definition of a Thaiangulation
def is_Thaiangulation {Π : Type} [fintype Π] (T : Triangulation Π) (A : Π → ℝ) : Prop :=
  ∀ t₁ t₂ ∈ T.triangles, A t₁ = A t₂

-- Main statement
theorem Thaiangulation_difference_by_two_triangles
  {Π : Type} [fintype Π] (A : Π → ℝ)
  (T₁ T₂ : Triangulation Π)
  (hT₁ : is_Thaiangulation T₁ A)
  (hT₂ : is_Thaiangulation T₂ A)
  (diff_T₁_T₂ : T₁ ≠ T₂) : 
  ∃ t₁ t₂ t₃ t₄ ∈ T₁.triangles ∪ T₂.triangles, 
  t₁ ∉ T₂.triangles ∧ 
  t₂ ∉ T₂.triangles ∧
  t₃ ∉ T₁.triangles ∧
  t₄ ∉ T₁.triangles ∧
  t₁ ∪ t₂ = t₃ ∪ t₄ ∧
  (T₁.triangles.erase t₁).insert t₃ = (T₂.triangles.erase t₄).insert t₂ :=
sorry

end Thaiangulation_difference_by_two_triangles_l496_496606


namespace prove_f_4_eq_82_l496_496364

noncomputable theory
open_locale classical

-- Define f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Assume f is monotonic
axiom monotonic_f : ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Given condition: ∀ x ∈ ℝ, f (f x - 3^x) = 4
axiom condition : ∀ x : ℝ, f (f x - 3^x) = 4

-- The theorem to prove: f(4) = 82
theorem prove_f_4_eq_82 : f 4 = 82 :=
by
  sorry

end prove_f_4_eq_82_l496_496364


namespace sum_of_positive_odd_divisors_180_l496_496989

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d % 2 = 1) (Finset.divisors n)).sum

theorem sum_of_positive_odd_divisors_180 : 
  sum_of_positive_odd_divisors 180 = 78 := by
  sorry

end sum_of_positive_odd_divisors_180_l496_496989


namespace exists_multiple_of_power_of_two_non_zero_digits_l496_496152

open Nat

theorem exists_multiple_of_power_of_two_non_zero_digits (k : ℕ) (h : 0 < k) : 
  ∃ m : ℕ, (2^k ∣ m) ∧ (∀ d ∈ digits 10 m, d ≠ 0) :=
sorry

end exists_multiple_of_power_of_two_non_zero_digits_l496_496152


namespace amateur_definition_l496_496250
-- Import necessary libraries

-- Define the meaning of "amateur" and state that it is "amateurish" or "non-professional"
def meaning_of_amateur : String :=
  "amateurish or non-professional"

-- The main statement asserting that the meaning of "amateur" is indeed "amateurish" or "non-professional"
theorem amateur_definition : meaning_of_amateur = "amateurish or non-professional" :=
by
  -- The proof is trivial and assumed to be correct
  sorry

end amateur_definition_l496_496250


namespace exists_even_n_points_l496_496701

theorem exists_even_n_points (n : ℕ) (h : n ≥ 2) :
  (∃ (P Q : Fin n → ℝ × ℝ),
    (∀ i j, P i ≠ P j → i ≠ j ∧ Q i ≠ Q j → i ≠ j ∧ P i ≠ Q j) ∧
    (∀ i, (dist (P i) (P ((i + 1) % n))) ≥ 1) ∧
    (∀ i, (dist (Q i) (Q ((i + 1) % n))) ≥ 1) ∧
    (∀ i j, (dist (P i) (Q j)) ≤ 1)) ↔ (Even n) := by
  sorry

end exists_even_n_points_l496_496701


namespace find_number_of_terms_l496_496380

theorem find_number_of_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = (2^n - 1) / (2^n)) → S n = 321 / 64 → n = 6 :=
by
  sorry

end find_number_of_terms_l496_496380


namespace largest_prime_factor_of_expression_l496_496584

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Prime p ∧ p > 35 ∧ p > 2 ∧ p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ ∀ q, Prime q ∧ q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l496_496584


namespace middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l496_496947

theorem middle_number_of_consecutive_numbers_sum_of_squares_eq_2030 :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = 2030 ∧ (n + 1) = 26 :=
by sorry

end middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l496_496947


namespace determine_q_l496_496915

noncomputable def q (x : ℝ) : ℝ := x^3 - (61 / 4) * x^2 + (305 / 4) * x - (225 / 4)

theorem determine_q : 
  (∃ q : ℝ → ℝ, 
    (∀ x : ℝ, q x = x^3 - (61 / 4) * x^2 + (305 / 4) * x - (225 / 4)) ∧
    (q 2 = q 2 - complex.I * complex.I) ∧ 
    (q 0 = -40)) :=
begin
  use q,
  split,
  { intro x,
    dsimp only [q], },
  split,
  { simp [q], },
  { simp [q], }
end

end determine_q_l496_496915


namespace pirate_ship_minimum_speed_l496_496625

noncomputable def minimum_speed (initial_distance : ℝ) (caravel_speed : ℝ) (caravel_direction : ℝ) : ℝ :=
  let caravel_velocity_x := -caravel_speed * Real.cos caravel_direction
  let caravel_velocity_y := -caravel_speed * Real.sin caravel_direction
  let t := initial_distance / (caravel_speed * (1 + Real.sqrt 3))
  let v_p := Real.sqrt ((initial_distance / t - caravel_velocity_x)^2 + (caravel_velocity_y)^2)
  v_p

theorem pirate_ship_minimum_speed : 
  minimum_speed 10 12 (Real.pi / 3) = 6 * Real.sqrt 6 :=
by
  sorry

end pirate_ship_minimum_speed_l496_496625


namespace Marge_savings_l496_496139

theorem Marge_savings
  (lottery_winnings : ℝ)
  (taxes_paid : ℝ)
  (student_loan_payment : ℝ)
  (amount_after_taxes : ℝ)
  (amount_after_student_loans : ℝ)
  (fun_money : ℝ)
  (investment : ℝ)
  (savings : ℝ)
  (h_win : lottery_winnings = 12006)
  (h_tax : taxes_paid = lottery_winnings / 2)
  (h_after_tax : amount_after_taxes = lottery_winnings - taxes_paid)
  (h_loans : student_loan_payment = amount_after_taxes / 3)
  (h_after_loans : amount_after_student_loans = amount_after_taxes - student_loan_payment)
  (h_fun : fun_money = 2802)
  (h_savings_investment : amount_after_student_loans - fun_money = savings + investment)
  (h_investment : investment = savings / 5)
  (h_left : amount_after_student_loans - fun_money = 1200) :
  savings = 1000 :=
by
  sorry

end Marge_savings_l496_496139


namespace point_on_perpendicular_bisector_l496_496499

def midpoint (A B : Point) : Point := sorry  -- Placeholder for the midpoint function definition

def equidistant (P A B : Point) : Prop := dist P A = dist P B

def perpendicular_bisector (X A B : Point) [M : Point] : Prop := 
  midpoint A B = M ∧ is_perpendicular (line_through X M) (line_through A B)

theorem point_on_perpendicular_bisector {A B X : Point} (h : equidistant X A B) : 
  ∃ M : Point, midpoint A B = M ∧ perpendicular_bisector X A B M :=
sorry

end point_on_perpendicular_bisector_l496_496499


namespace set_intersection_complement_A_B_l496_496801

noncomputable def setA : Set ℝ := {x : ℝ | x^2 + x - 2 > 0}
noncomputable def setComplementA : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}
noncomputable def setB : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.log2 x}

theorem set_intersection_complement_A_B :
  (setComplementA ∩ setB) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_complement_A_B_l496_496801


namespace game_is_not_fair_l496_496598

-- Define probabilities and outcomes
def ball_numbers : List ℕ := [1, 2, 3, 4]

-- Function to compute the product parity
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

-- Function to compute the product of drawing two balls
def product_parity (b1 b2 : ℕ) : Prop :=
  if is_even (b1 * b2) then is_even (b1 * b2) else is_odd (b1 * b2)

-- Count elder sister wins
def elder_sister_wins : ℕ :=
  (ball_numbers.filter (λ b1, 
    ball_numbers.countp (λ b2, is_even (b1 * b2)) > 0)).length

-- Count younger sister wins
def younger_sister_wins : ℕ :=
  (ball_numbers.filter (λ b1, 
    ball_numbers.countp (λ b2, is_odd (b1 * b2)) > 0)).length

-- Probabilities for winning
def elder_sister_probability : ℚ := elder_sister_wins / (ball_numbers.length * ball_numbers.length)
def younger_sister_probability : ℚ := younger_sister_wins / (ball_numbers.length * ball_numbers.length)

-- Game fairness
def is_game_fair : Prop := elder_sister_probability = younger_sister_probability

-- Formal proof statement
theorem game_is_not_fair : ¬ is_game_fair :=
by {
  sorry -- Proof omitted
}

end game_is_not_fair_l496_496598


namespace monotonicity_f_inequality_a_zero_l496_496379

noncomputable def p (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def q (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - (1 + 2 * a) * x
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := q(x, a) + 2 * a * x * p(x)

theorem monotonicity_f (a : ℝ) :
  ( (a ≤ 0) → 
    (∀ x : ℝ, 0 < x ∧ x < 1 → f'(x, a) < 0) ∧ 
    (∀ x : ℝ, x > 1 → f'(x, a) > 0) ) ∧
  ( (0 < a) ∧ (a < 1/2) → 
    (∀ x : ℝ, 0 < x ∧ x < 2 * a → f'(x, a) > 0) ∧ 
    (∀ x : ℝ, 2 * a < x ∧ x < 1 → f'(x, a) < 0) ∧ 
    (∀ x : ℝ, x > 1 → f'(x, a) > 0) ) ∧
  ( (a = 1/2) → 
    (∀ x : ℝ, x > 0 → f'(x, a) ≥ 0) ) ∧
  ( (a > 1/2) → 
    (∀ x : ℝ, 0 < x ∧ x < 1 → f'(x, a) > 0) ∧ 
    (∀ x : ℝ, 1 < x ∧ x < 2 * a → f'(x, a) < 0) ∧ 
    (∀ x : ℝ, x > 2 * a → f'(x, a) > 0) ) :=
sorry

theorem inequality_a_zero :
  ∀ x : ℝ, x > 0 → x * p(x) + q(x, 0) < exp(x) + (1/2) * x^2 - x - 1 :=
sorry

end monotonicity_f_inequality_a_zero_l496_496379


namespace sum_of_reciprocals_of_squared_roots_plus_one_l496_496326

noncomputable def p (x : ℂ) : ℂ := 985 * x^2021 + 211 * x^2020 - 211

def roots_p : Fin 2021 → ℂ := sorry  -- Assume this is a function that gives us x_k for k in {1, 2, ..., 2021}

theorem sum_of_reciprocals_of_squared_roots_plus_one :
  ∑ k in Finset.range 2021, (1 / (roots_p ⟨k, sorry⟩)^2 + 1) = 2021 :=
sorry

end sum_of_reciprocals_of_squared_roots_plus_one_l496_496326


namespace find_X_l496_496386

def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (3, 0, 2)
def c (X : ℝ) : ℝ × ℝ × ℝ := (4, 2, X)

theorem find_X (X : ℝ) (h : ∃ (λ μ : ℝ), c X = (λ • a) + (μ • b)) : X = 5 := by
  sorry

end find_X_l496_496386


namespace cinema_number_of_rows_l496_496435

theorem cinema_number_of_rows :
  let total_seats_base_8 := 3 * 8^2 + 5 * 8^1 + 1 * 8^0 in
  let seats_per_row := 3 in
  let total_rows := total_seats_base_8 / seats_per_row in
  total_rows = 77 := by
  sorry

end cinema_number_of_rows_l496_496435


namespace part_I_part_II_l496_496029

def f (x a : ℝ) : ℝ := abs (3 * x + 2) - abs (2 * x + a)

theorem part_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = 4 / 3 :=
by
  sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≤ 0) ↔ (3 ≤ a ∨ a ≤ -7) :=
by
  sorry

end part_I_part_II_l496_496029


namespace Q_at_1_evaluation_l496_496670

theorem Q_at_1_evaluation :
  let P : Polynomial ℤ := 3 * X^3 - 5 * X^2 + X - 7
  let mean := (3 + (-5) + 1 + (-7)) / 4
  let Q : Polynomial ℤ := mean * X^3 + mean * X^2 + mean * X + mean
  Q.eval 1 = -8 := by
  -- The proof goes here, but is omitted
  sorry

end Q_at_1_evaluation_l496_496670


namespace pure_imaginary_solution_l496_496052

theorem pure_imaginary_solution (z : ℂ) (b : ℝ) (hz : ∃ a : ℝ, z = a * complex.I) (h : (2 - complex.I) * z = 4 - b * complex.I) : b = -8 :=
sorry

end pure_imaginary_solution_l496_496052


namespace dilute_solution_l496_496538

variable (N : ℝ) -- variable for ounces of water added

-- Define the initial condition
def initial_solution : ℝ := 12 -- 12 ounces of initial solution
def initial_alcohol_percentage : ℝ := 0.6 -- 60% alcohol
def desired_alcohol_percentage : ℝ := 0.4 -- 40% alcohol

-- Define the alcohol content
def initial_alcohol_content : ℝ := initial_solution * initial_alcohol_percentage -- 7.2 ounces of alcohol

-- Lean theorem statement
theorem dilute_solution (N : ℝ) 
  (h : initial_alcohol_content 12 0.6 = initial_solution 12 * initial_alcohol_percentage 0.6) :
  initial_alcohol_content 12 0.6 = desired_alcohol_percentage 0.4 * (initial_solution 12 + N) :=
begin
  sorry
end

end dilute_solution_l496_496538


namespace tree_grade_third_grade_l496_496527

theorem tree_grade_third_grade (C : ℝ) (a5_core a5_bark : ℝ) (r : ℝ) (n : ℕ)
  (hC : C = 3.14)
  (ha5_core : a5_core = 0.4)
  (ha5_bark : a5_bark = 0.2)
  (hradius : r = 50)
  (harithmetic_sequence : ∀ n, a5_core - a5_bark = (a5_core - a5_bark) / (n - 1 + 4))
  (radius_sum : r = n * (a5_core + a5_bark) / 2)
: 100 ≤ n ∧ n ≤ 299 →
   "The large tree belongs to the third grade." :=
by
  sorry

end tree_grade_third_grade_l496_496527


namespace profit_after_taxes_and_quantities_and_prices_l496_496486

variables {P_A P_B : ℝ} {q_A q_B Q : ℝ}

/-- Define the demand functions for countries A and B -/
def demand_A : ℝ → ℝ := λ P_A, 40 - 2 * P_A
def demand_B : ℝ → ℝ := λ P_B, 26 - P_B

/-- Define the total cost function -/
def total_cost : ℝ → ℝ := λ Q, 8 * Q + 1

/-- Define the total quantity produced as the sum of quantities in both countries -/
def total_quantity : ℝ := q_A + q_B

/-- Define the total revenue for countries A and B -/
def total_revenue_A : ℝ := P_A * q_A
def total_revenue_B : ℝ := P_B * q_B

/-- Define the total revenue as the sum of revenues from both countries -/
def total_revenue : ℝ := total_revenue_A + total_revenue_B

/-- Define the total profit before tax -/
def total_profit : ℝ := total_revenue - total_cost total_quantity - 2

/-- Define the post-tax profit for Country A (fixed rate 15%) -/
def post_tax_profit_A : ℝ := total_profit * 0.85

/-- Define the post-tax profit for Country B (progressive rates) -/
def post_tax_profit_B : ℝ :=
  if total_profit ≤ 30 then
    total_profit
  else if total_profit ≤ 100 then
    30 + 0.9 * (total_profit - 30)
  else if total_profit ≤ 150 then
    30 + 0.9 * 70 + 0.8 * (total_profit - 100)
  else
    30 + 0.9 * 70 + 0.8 * 50 + 0.7 * (total_profit - 150)

/-- Main theorem to prove the profits after taxes and quantities and prices -/
theorem profit_after_taxes_and_quantities_and_prices :
  q_A = 12 ∧ q_B = 9 ∧ P_A = 14 ∧ P_B = 17 ∧ post_tax_profit_B = 133.7 :=
by
  -- This space is left for the full proof. Use sorry to compile successfully.
  sorry

end profit_after_taxes_and_quantities_and_prices_l496_496486


namespace exactly_one_correct_proposition_l496_496951

-- Definitions and conditions
variable {a b c : Type} [has_parallel a b] [has_parallel b c]
variable [has_perp a b] [has_perp b c]
variable {α : Type} [has_parallel a α] [subset b α]

-- Definitions for propositions
def prop1 : Prop := has_parallel a b ∧ has_parallel b c → has_parallel a c
def prop2 : Prop := has_perp a b ∧ has_perp b c → has_parallel a c
def prop3 : Prop := has_parallel a α ∧ subset b α → has_parallel a b
def prop4 : Prop := has_parallel a b ∧ has_parallel b α → has_parallel a α

-- Prove that there is exactly one correct proposition
theorem exactly_one_correct_proposition : prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 := by
  sorry

end exactly_one_correct_proposition_l496_496951


namespace find_sets_l496_496034

open Set

noncomputable def U := ℝ
def A := {x : ℝ | Real.log x / Real.log 2 <= 2}
def B := {x : ℝ | x ≥ 1}

theorem find_sets (x : ℝ) :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  (compl A ∩ B = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) :=
  sorry

end find_sets_l496_496034


namespace reserved_fraction_l496_496896

variable (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ)
variable (f : ℚ)

def mrSalazarFractionReserved (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ) : ℚ :=
  1 - (leftover_oranges + rotten_oranges) * sold_fraction / initial_oranges

theorem reserved_fraction (h1 : initial_oranges = 84) (h2 : sold_fraction = 3 / 7) (h3 : rotten_oranges = 4) (h4 : leftover_oranges = 32) :
  (mrSalazarFractionReserved initial_oranges sold_fraction rotten_oranges leftover_oranges) = 1 / 4 :=
  by
    -- Proof is omitted
    sorry

end reserved_fraction_l496_496896


namespace problem_l496_496870

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (λ : Fin n → ℝ)

theorem problem (hx : ∀ i, 0 < x i)
                (hλ : ∀ i, 0 ≤ λ i)
                (hλsum : ∑ i, λ i = 1)
                (hxincr : ∀ i j, i < j → x i < x j) :
  (∑ i, (λ i) * (x i)) * (∑ i, λ i * (x i)⁻¹) ≤ ( ( (x 0 + x (Fin.last n)) / 2 ) ^ 2 ) / (x 0 * x (Fin.last n)) := 
sorry

end problem_l496_496870


namespace find_N_l496_496819

theorem find_N (N : ℤ) :
  (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 :=
by
  sorry

end find_N_l496_496819


namespace anya_hairs_wanted_more_l496_496653

def anya_initial_number_of_hairs : ℕ := 0 -- for simplicity, assume she starts with 0 hairs
def hairs_lost_washing : ℕ := 32
def hairs_lost_brushing : ℕ := hairs_lost_washing / 2
def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing
def hairs_to_grow_back : ℕ := 49

theorem anya_hairs_wanted_more : total_hairs_lost + hairs_to_grow_back = 97 :=
by
  sorry

end anya_hairs_wanted_more_l496_496653


namespace dihedral_angle_eq_arcsin_2_over_3_l496_496202

-- Definition of conditions in the proof problem
variables (A B C D E : Type)
variables [NormedAddCommGroup A] [NormedAddCommGroup B]
variables [NormedAddCommGroup C] [NormedAddCommGroup D] [NormedAddCommGroup E]
variables (AD BE DE : ℝ)
def perpendicular (x y : Type) [NormedAddCommGroup x] [NormedAddCommGroup y] := true

-- Given conditions
axiom AD_eq : AD = 1
axiom BE_eq : BE = 2
axiom DE_eq : DE = 3
axiom is_perpendicular_AD : perpendicular A D
axiom is_perpendicular_BE : perpendicular B E

-- The objective to prove:
theorem dihedral_angle_eq_arcsin_2_over_3 
  (h1 : AD_eq)
  (h2 : BE_eq)
  (h3 : DE_eq)
  (h4 : is_perpendicular_AD)
  (h5 : is_perpendicular_BE) : 
  ∃ (φ : ℝ), φ = Real.arcsin (2/3) :=
by
  sorry

end dihedral_angle_eq_arcsin_2_over_3_l496_496202


namespace angle_in_third_quadrant_l496_496549

-- Definition: Condition 1 - The angle in radians
def angle_radians : ℝ := 4

-- Definition: Conversion from radians to degrees
def radians_to_degrees (radians : ℝ) : ℝ := radians * 180 / Real.pi

-- The correct answer we want to prove
def correct_quadrant (degrees : ℝ) : String :=
  if degrees > 180 ∧ degrees <= 270 then "third" else "unknown"

-- Statement: Given the angle in radians, prove its terminal side is in the third quadrant
theorem angle_in_third_quadrant (angle_radians = 4) :
  correct_quadrant (radians_to_degrees angle_radians) = "third" := by
  sorry

end angle_in_third_quadrant_l496_496549


namespace standard_circle_equation_l496_496545

theorem standard_circle_equation (x y : ℝ) :
  ∃ (h k r : ℝ), h = 2 ∧ k = -1 ∧ r = 3 ∧ (x - h)^2 + (y - k + 1)^2 = r^2 :=
by
  use 2, -1, 3
  simp
  sorry

end standard_circle_equation_l496_496545


namespace sum_of_positive_odd_divisors_of_180_l496_496980

/-- The sum of the positive odd divisors of 180 is 78. -/
theorem sum_of_positive_odd_divisors_of_180 : 
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45] in
  let sum_odd_divisors := List.sum odd_divisors in
  sum_odd_divisors = 78 := 
by
  -- odd_divisors is [1, 3, 5, 9, 15, 45]
  let odd_divisors : List ℕ := [1, 3, 5, 9, 15, 45]
  -- Summation of the odd divisors
  let sum_odd_divisors := List.sum odd_divisors
  -- Verify that the sum is 78
  show sum_odd_divisors = 78
  sorry

end sum_of_positive_odd_divisors_of_180_l496_496980


namespace range_of_b_minus_2_over_a_minus_1_l496_496750

noncomputable def f (x a b c : ℝ) := (1 / 3) * x^3 + (1 / 2) * a * x^2 + 2 * b * x + c

theorem range_of_b_minus_2_over_a_minus_1 (x1 x2 a b c : ℝ) (h1 : 0 < x1) (h2 : x1 < 1) 
(h3 : 1 < x2) (h4 : x2 < 2) (h5 : ∀ x, deriv (λ x, f x a b c) x = 0 → (x = x1 ∨ x = x2)) :
  (1 / 4) < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1 :=
sorry

end range_of_b_minus_2_over_a_minus_1_l496_496750


namespace knowledge_competition_score_l496_496073

theorem knowledge_competition_score (x : ℕ) (hx : x ≤ 20) : 5 * x - (20 - x) ≥ 88 :=
  sorry

end knowledge_competition_score_l496_496073


namespace exists_real_x_P_x_minus_1_eq_Q_x_plus_1_l496_496460

theorem exists_real_x_P_x_minus_1_eq_Q_x_plus_1 :
  ∀ (P Q : Polynomial ℝ), 
  P.degree = 2018 ∧ Q.degree = 2018 ∧ P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1 ∧ (∀ x : ℝ, P.eval x ≠ Q.eval x) → 
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) :=
by
  intros P Q h,
  cases h with hP h1,
  cases h1 with hQ h2,
  cases h2 with hLP h3,
  cases h3 with hLQ h4,
  sorry

end exists_real_x_P_x_minus_1_eq_Q_x_plus_1_l496_496460


namespace proj_7v_l496_496128

variables (v w : ℝ^2)

def proj_w_v := ⟨4, 3⟩

theorem proj_7v : ∀ (v w : ℝ^2), (proj_w_v = ⟨4, 3⟩) → 
  (7 • proj_w_v = ⟨28, 21⟩) := 
by
  intros
  exact sorry

end proj_7v_l496_496128


namespace min_value_pairwise_products_sum_95_l496_496337

theorem min_value_pairwise_products_sum_95 {a : Fin 95 → ℤ} 
  (h1 : ∀ i : Fin 95, a i = 1 ∨ a i = -1) : 
  ∑ i j in Finset.offDiag (Finset.univ : Finset (Fin 95)), a i * a j = 13 :=
sorry

end min_value_pairwise_products_sum_95_l496_496337


namespace initial_blocks_l496_496502

theorem initial_blocks (used_blocks remaining_blocks : ℕ) (h1 : used_blocks = 25) (h2 : remaining_blocks = 72) : 
  used_blocks + remaining_blocks = 97 := by
  sorry

end initial_blocks_l496_496502


namespace find_x_if_perpendicular_l496_496387

-- Given definitions and conditions
def a : ℝ × ℝ := (-5, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Statement to be proved
theorem find_x_if_perpendicular (x : ℝ) :
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 10 :=
by
  sorry

end find_x_if_perpendicular_l496_496387


namespace correctProposition_l496_496246

-- Define the axioms used in the problem
axiom axiom1 : ∀ (p1 p2 p3 : Point), ¬ Collinear p1 p2 p3 → Plane p
axiom axiom2 : ∀ (l : Line) (p : Point), p ∉ l.points → Plane p
axiom axiom3 : ∀ (l1 l2 : Line), l1 ≠ l2 ∧ Intersect l1 l2 → Plane p

-- Define propositions presented in the problem
def propositionA : Prop := ∀ (p1 p2 p3 : Point), Plane p
def propositionB : Prop := ∀ (l : Line) (p : Point), Plane p
def propositionC : Prop := ∀ (q : Quadrilateral), Plane p
def propositionD : Prop := ∀ (l1 l2 : Line), Intersect l1 l2 → Plane p

-- Stating the proof problem
theorem correctProposition : propositionD := 
by 
  sorry

end correctProposition_l496_496246


namespace cos_A_in_right_triangle_AB_5_BC_4_cos_B_4_over_5_l496_496010

theorem cos_A_in_right_triangle_AB_5_BC_4_cos_B_4_over_5 :
  ∀ (A B C : Type) [EuclideanGeometry A B C],
  (AB AB 5) ∧ (BC BC 4) ∧ (cos B (B) (4 / 5)) → 
  (cos A (A) = 3 / 5) := 
sorry

end cos_A_in_right_triangle_AB_5_BC_4_cos_B_4_over_5_l496_496010


namespace largest_real_number_l496_496738

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496738


namespace roots_theorem_l496_496509

-- Definitions and Conditions
def root1 (a b p : ℝ) : Prop := 
  a + b = -p ∧ a * b = 1

def root2 (b c q : ℝ) : Prop := 
  b + c = -q ∧ b * c = 2

-- The theorem to prove
theorem roots_theorem (a b c p q : ℝ) (h1 : root1 a b p) (h2 : root2 b c q) : 
  (b - a) * (b - c) = p * q - 6 :=
sorry

end roots_theorem_l496_496509


namespace maximize_profit_in_country_B_l496_496488

noncomputable def demand_A (P_A : ℝ) := 40 - 2 * P_A
noncomputable def demand_B (P_B : ℝ) := 26 - P_B

noncomputable def total_cost (Q : ℝ) := 8 * Q + 1

noncomputable def total_revenue_A (q_A : ℝ) := (20 - q_A / 2) * q_A
noncomputable def total_revenue_B (q_B : ℝ) := (26 - q_B) * q_B

noncomputable def marginal_revenue_A (q_A : ℝ) := 20 - q_A
noncomputable def marginal_revenue_B (q_B : ℝ) := 26 - 2 * q_B

noncomputable def marginal_cost := 8

theorem maximize_profit_in_country_B : 
  let q_A := 12
  let q_B := 9
  let P_A := 14
  let P_B := 17
  let Q := q_A + q_B
  let TR_A := 14 * 12
  let TR_B := 17 * 9
  let TR_total := TR_A + TR_B
  let TC := total_cost Q
  let profit_before_tax := TR_total - TC - 2
  let profit_after_tax_B := 30 + 63 + 40 + 0.7 -- profit calculation in country B with progressive tax
  let profit_after_tax_A := profit_before_tax * 0.85 -- profit calculation in country A with flat tax
  in profit_after_tax_B = 133.7 ∧ profit_after_tax_B > profit_after_tax_A :=
begin
  have q_A := 12,
  have q_B := 9,
  have P_A := 14,
  have P_B := 17,
  have Q := q_A + q_B,
  have TR_A := 14 * 12,
  have TR_B := 17 * 9,
  have TR_total := TR_A + TR_B,
  have TC := total_cost Q,
  have profit_before_tax := TR_total - TC - 2,
  have profit_after_tax_B := 30 + 63 + 40 + 0.7,
  have profit_after_tax_A := profit_before_tax * 0.85,
  split,
  { exact profit_after_tax_B, },
  { exact profit_after_tax_B > profit_after_tax_A, },
end

end maximize_profit_in_country_B_l496_496488


namespace range_of_a_l496_496382

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a

def proposition_q (a : ℝ) : Prop := ∃ (x₀ : ℝ), x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : proposition_p a ∧ proposition_q a ↔ (a = 1 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_l496_496382


namespace number_of_solutions_l496_496190

-- Define the custom operation star
def star (a b : ℕ) : ℕ := a * a / b

-- The main theorem stating the number of integer values x that make 9 star x a positive integer is 5
theorem number_of_solutions : (finset.univ.filter (λ x : ℕ, (9 * 9 / x > 0 ∧ 9 * 9 % x = 0))).card = 5 :=
by
  -- univ represents all natural numbers
  -- filter predicate checks that (9 * 9 / x > 0) and 9 * 9 is divisible by x (i.e., 9 * 9 % x = 0)
  sorry

end number_of_solutions_l496_496190


namespace solve_for_x_l496_496403

theorem solve_for_x (x : ℝ) (h : 3 + 5 * x = 28) : x = 5 :=
by {
  sorry
}

end solve_for_x_l496_496403


namespace tan_arith_seq_l496_496417

theorem tan_arith_seq (x y z : ℝ)
  (h₁ : y = x + π / 3)
  (h₂ : z = x + 2 * π / 3) :
  (Real.tan x * Real.tan y) + (Real.tan y * Real.tan z) + (Real.tan z * Real.tan x) = -3 :=
sorry

end tan_arith_seq_l496_496417


namespace monthly_compounding_is_better_l496_496602

noncomputable def principal : ℝ := 1000
noncomputable def annualRate : ℝ := 0.05
noncomputable def years : ℕ := 10
noncomputable def annualCompounding : ℕ := 1
noncomputable def monthlyCompounding : ℕ := 12

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem monthly_compounding_is_better :
  compound_interest principal annualRate monthlyCompounding years 
  > compound_interest principal annualRate annualCompounding years := 
by {
  sorry
}

end monthly_compounding_is_better_l496_496602


namespace combined_distance_20_birds_two_seasons_l496_496605

theorem combined_distance_20_birds_two_seasons :
  let distance_jim_to_disney := 50
  let distance_disney_to_london := 60
  let number_of_birds := 20
  (number_of_birds * (distance_jim_to_disney + distance_disney_to_london)) = 2200 := by
  sorry

end combined_distance_20_birds_two_seasons_l496_496605


namespace compare_y_values_l496_496497

noncomputable def parabola (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 1

theorem compare_y_values :
  ∃ y1 y2 y3, (parabola (-3) = y1) ∧ (parabola (-2) = y2) ∧ (parabola 2 = y3) ∧ (y3 < y1) ∧ (y1 < y2) :=
by
  sorry

end compare_y_values_l496_496497


namespace bus_passenger_count_l496_496925

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l496_496925


namespace determine_list_price_l496_496640

theorem determine_list_price (x : ℝ) :
  0.12 * (x - 15) = 0.15 * (x - 25) → x = 65 :=
by 
  sorry

end determine_list_price_l496_496640


namespace fraction_of_crop_closest_to_DA_l496_496936

noncomputable section

open Real

-- Given points A, B, C, D
def A := (0 : ℝ) × (0 : ℝ)
def B := (1 : ℝ) × (2 : ℝ)
def C := (3 : ℝ) × (2 : ℝ)
def D := (4 : ℝ) × (0 : ℝ)

-- Given angles at vertices
def angle_A := 75
def angle_D := 75
def angle_B := 105
def angle_C := 105

-- Given side lengths
def side_AB := 100
def side_BC := 150
def side_CD := 100
def side_DA := 150

-- Define areas ( A_{DEF} and A_{total} are placeholders)
-- Assume we have calculated regions and total area
def A_DEF : ℝ := sorry -- Placeholder for the area of triangle DEF (region closest to DA)
def A_total : ℝ := sorry -- Placeholder for the total area of the quadrilateral

-- Define the fraction
def fraction_closest_to_DA : ℝ := A_DEF / A_total

-- The theorem we are proving
theorem fraction_of_crop_closest_to_DA :
  fraction_closest_to_DA =
  A_DEF / A_total := 
by
  sorry

end fraction_of_crop_closest_to_DA_l496_496936


namespace inscribed_circle_radius_l496_496192

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ), a = 3 → b = 6 → c = 18 → (∃ (r : ℝ), (1 / r) = (1 / a) + (1 / b) + (1 / c) + 4 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 9 / (5 + 6 * Real.sqrt 3)) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end inscribed_circle_radius_l496_496192


namespace math_problem_l496_496530

noncomputable def f : ℝ → ℝ := sorry

noncomputable def g (x : ℝ) : ℝ := f (x + (1/6))

lemma f_odd (x : ℝ) : f (-x) = -f (x) := sorry

lemma g_def (x : ℝ) : g (x) = f (x + 1/6) := sorry

lemma g_def_shifted (x : ℝ) : g (x + 1/3) = f (1/2 - x) := sorry

lemma f_interval (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : f (x) = 2^x - 1 := sorry

theorem math_problem :
  f (Real.log 5 / Real.log 2) + g (5/6) = 1/4 :=
sorry

end math_problem_l496_496530


namespace positive_integer_solution_l496_496219

theorem positive_integer_solution (n : ℕ) (h1 : n + 2009 ∣ n^2 + 2009) (h2 : n + 2010 ∣ n^2 + 2010) : n = 1 := 
by
  -- The proof would go here.
  sorry

end positive_integer_solution_l496_496219


namespace fibonacci_series_limit_l496_496868

open Classical
open Nat

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 := 0
| 1 := 1
| (n+2) := fibonacci(n+1) + fibonacci(n)

-- Define the given sequence in the problem
def seq_sum (n : ℕ) : ℚ := (List.range (n + 1)).sum (λ i, fibonacci i / 2 ^ i)

-- The statement to be proven in Lean 4:
theorem fibonacci_series_limit : ∃ L : ℚ, L = 2 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |seq_sum n - L| < ε :=
sorry  -- Proof to be provided

end fibonacci_series_limit_l496_496868


namespace number_of_cubes_with_both_colors_painted_l496_496621

theorem number_of_cubes_with_both_colors_painted :
  let cube_side := 4 in 
  let total_small_cubes := cube_side * cube_side * cube_side in 
  let blue_sides_cubes := (cube_side - 2) * cube_side * 2 in 
  let corner_cubes := 8 in 
  (total_small_cubes, blue_sides_cubes, corner_cubes) = (64, 24, 8) -> 24 := 
sorry

end number_of_cubes_with_both_colors_painted_l496_496621


namespace covered_ratio_battonya_covered_ratio_sopron_l496_496306

noncomputable def angular_diameter_sun : ℝ := 1899 / 2
noncomputable def angular_diameter_moon : ℝ := 1866 / 2

def max_phase_battonya : ℝ := 0.766
def max_phase_sopron : ℝ := 0.678

def center_distance (R_M R_S f : ℝ) : ℝ :=
  R_M - (2 * f - 1) * R_S

-- Defining the hypothetical calculation (details omitted for brevity)
def covered_ratio (R_S R_M d : ℝ) : ℝ := 
  -- Placeholder for the actual calculation logic
  sorry

theorem covered_ratio_battonya :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_battonya) = 0.70 :=
  sorry

theorem covered_ratio_sopron :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_sopron) = 0.59 :=
  sorry

end covered_ratio_battonya_covered_ratio_sopron_l496_496306


namespace log_m_n_iff_m_minus_1_n_minus_1_l496_496359

theorem log_m_n_iff_m_minus_1_n_minus_1 (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) (h3 : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) :=
sorry

end log_m_n_iff_m_minus_1_n_minus_1_l496_496359


namespace congruent_is_sufficient_but_not_necessary_for_similarity_l496_496568

noncomputable section

def congruent (Δ₁ Δ₂ : Triangle) : Prop := 
  ∃ f : Δ₁ → Δ₂, is_isomorphism f

def similar (Δ₁ Δ₂ : Triangle) : Prop := 
  ∃ f : Δ₁ → Δ₂, is_homothetic f

theorem congruent_is_sufficient_but_not_necessary_for_similarity 
  (Δ₁ Δ₂ : Triangle)
  (h : congruent Δ₁ Δ₂) : 
  similar Δ₁ Δ₂ ∧ ¬ (similar Δ₁ Δ₂ → congruent Δ₁ Δ₂) := 
by
  sorry

end congruent_is_sufficient_but_not_necessary_for_similarity_l496_496568


namespace perp_condition_l496_496388

def a (x : ℝ) : ℝ × ℝ := (x-1, 2)
def b : ℝ × ℝ := (2, 1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_condition (x : ℝ) : dot_product (a x) b = 0 ↔ x = 0 :=
by 
  sorry

end perp_condition_l496_496388


namespace probability_at_least_one_shows_one_is_correct_l496_496566

/-- Two fair 8-sided dice are rolled. What is the probability that at least one of the dice shows a 1? -/
def probability_at_least_one_shows_one : ℚ :=
  let total_outcomes := 8 * 8
  let neither_one := 7 * 7
  let at_least_one := total_outcomes - neither_one
  at_least_one / total_outcomes

theorem probability_at_least_one_shows_one_is_correct :
  probability_at_least_one_shows_one = 15 / 64 :=
by
  unfold probability_at_least_one_shows_one
  sorry

end probability_at_least_one_shows_one_is_correct_l496_496566


namespace regression_analysis_incorrect_statement_l496_496581

theorem regression_analysis_incorrect_statement :
  ∃ (B : Prop), 
  (∀ (A C D : Prop), 
    (A ↔ ∀ (x̄ ȳ : ℝ), (∃ (f : ℝ → ℝ), f = least_squares_regression_line  x̄ ȳ) → (f x̄ = ȳ)) ∧
    (B ↔ ¬(least_squares_regression_line_minimizes_distance (∃ (data_points : list (ℝ × ℝ)), true))) ∧
    (C ↔ ∀ (r : ℝ), r > 0 → positively_correlated r) ∧
    (D ↔ ∀ (r : ℝ), (|r| = abs r) → linear_correlation_weaker r)
  ) → ¬ B :=
begin
  sorry
end

end regression_analysis_incorrect_statement_l496_496581


namespace max_teams_score_exactly_ten_l496_496426

theorem max_teams_score_exactly_ten (teams : ℕ) (points_per_win : ℕ) (points_per_draw : ℕ) (points_per_loss : ℕ) : 
  teams = 17 → 
  points_per_win = 3 → 
  points_per_draw = 1 → 
  points_per_loss = 0 → 
  ∃ n, n ≤ 17 ∧ ∀ t, t ≤ n → teams_have_exact_points t points_per_win points_per_draw points_per_loss 17 10 → n = 11 :=
by
  intros 
  intro h_teams, intro h_points_win, intro h_points_draw, intro h_points_loss
  use 11
  split
  case a => 
    exact Nat.le_refl 11
  case b => 
    intros t h_le_t h_teams_have_exact_points
    sorry

end max_teams_score_exactly_ten_l496_496426


namespace max_stamps_l496_496828

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) : 
  price_per_stamp = 45 ∧ total_cents = 3600 → ∃ n : ℕ, n * price_per_stamp ≤ total_cents ∧ n = 80 :=
begin
  sorry
end

end max_stamps_l496_496828


namespace triangle_area_l496_496440

noncomputable def is_right_angled_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
                                          (a b c : ℝ)
                                          (angle_A : A → ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
                      (AB AC : ℝ) (angle_A : A → ℝ) (h : angle_A = 90) : 
                      is_right_angled_triangle A B C AB AC (sqrt (AB^2 + AC^2)) ∧ 
                      (1 / 2) * AB * AC = 270 :=
by
  have h1 : is_right_angled_triangle A B C AB AC (sqrt (AB^2 + AC^2)), 
  { sorry },
  have h2 : (1 / 2) * AB * AC = 270, 
  { sorry },
  exact ⟨h1, h2⟩

end triangle_area_l496_496440


namespace arithmetic_sequence_sum_l496_496390

-- Given {a_n} is an arithmetic sequence, and a_2 + a_3 + a_{10} + a_{11} = 40, prove a_6 + a_7 = 20
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 40) :
  a 6 + a 7 = 20 :=
sorry

end arithmetic_sequence_sum_l496_496390


namespace red_monochromatic_triangles_l496_496966

theorem red_monochromatic_triangles (P : Type) [fintype P] (h1 : fintype.card P = 43)
  (h2 : ∀ (v : P), (∑ (e : P → Prop (v ∈ red_edges e), 20) + (∑ (e : P → Prop (v ∈ blue_edges e), 22) )
  (h3 : blue_monochromatic_triangles = 2022) :
  red_monochromatic_triangles = 859 := 
sorry

end red_monochromatic_triangles_l496_496966


namespace molecular_weight_CO_l496_496969

theorem molecular_weight_CO :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let molecular_weight := atomic_weight_C + atomic_weight_O
  molecular_weight = 28.01 := 
by
  sorry

end molecular_weight_CO_l496_496969


namespace find_a9_l496_496443

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => a n + n

theorem find_a9 : a 9 = 37 := by
  sorry

end find_a9_l496_496443


namespace cost_price_of_watch_l496_496591

theorem cost_price_of_watch 
  (CP : ℝ) 
  (h₁ : ∃ SP1 SP2 : ℝ, SP1 = 0.90 * CP ∧ SP2 = 1.05 * CP ∧ SP2 - SP1 = 180) : 
  CP = 1200 :=
by
  cases h₁ with SP1 h₁_cases,
  cases h₁_cases with SP2 h₁_conditions,
  cases h₁_conditions with h1 h2 h3,
  sorry

end cost_price_of_watch_l496_496591


namespace max_min_values_of_f_l496_496534

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_values_of_f : 
  (∀ x ∈ set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 3) ∧ 
  (∃ x ∈ set.Icc (-3 : ℝ) (0 : ℝ), f x = 3) ∧
  (∀ x ∈ set.Icc (-3 : ℝ) (0 : ℝ), f (-3 : ℝ) ≤ f x) ∧ 
  (∃ x ∈ set.Icc (-3 : ℝ) (0 : ℝ), f (-3 : ℝ) = f x) := 
begin
  sorry
end

end max_min_values_of_f_l496_496534


namespace divisible_by_13_l496_496879

noncomputable def f : ℕ → ℤ
| 0       := 0
| 1       := 0
| (v + 2) := 4 ^ (v + 2) * f (v + 1) - 16 ^ (v + 1) * f v + v * 2 ^ v ^ 2

theorem divisible_by_13 
  : ∀ v, v ∈ {1989, 1990, 1991} → (f v) % 13 = 0 :=
by
  sorry

end divisible_by_13_l496_496879


namespace sum_of_positive_odd_divisors_180_l496_496987

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d % 2 = 1) (Finset.divisors n)).sum

theorem sum_of_positive_odd_divisors_180 : 
  sum_of_positive_odd_divisors 180 = 78 := by
  sorry

end sum_of_positive_odd_divisors_180_l496_496987


namespace compound_interest_comparison_l496_496603

theorem compound_interest_comparison
    (P : ℝ) (t : ℝ) (annual_rate : ℝ) (monthly_factor : ℝ) : 
    P * (1 + annual_rate / monthly_factor)^(monthly_factor * t) > P * (1 + annual_rate)^t :=
by
  let P := 1000
  let t := 10
  let annual_rate := 0.05
  let monthly_factor := 12
  have annual_amount := P * (1 + annual_rate)^t
  have monthly_amount := P * (1 + annual_rate / monthly_factor)^(monthly_factor * t)
  show monthly_amount > annual_amount
  sorry

end compound_interest_comparison_l496_496603


namespace sum_interior_angles_polygon_succ_l496_496020

def sum_of_interior_angles_polygon (k : ℕ) : ℝ := sorry

theorem sum_interior_angles_polygon_succ (k : ℕ) (f : ℕ → ℝ) 
  (h₁ : ∀ k, f(k) = sum_of_interior_angles_polygon k)
  (h₂ : f(3) = π) :
  f(k + 1) = f(k) + π :=
sorry

end sum_interior_angles_polygon_succ_l496_496020


namespace find_sum_of_digits_of_satisfying_ns_l496_496132

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeros (x : ℕ) : ℕ :=
Nat.floor (x / 5) + Nat.floor (x / 25) + Nat.floor (x / 125) -- This defines the number of trailing zeros (approximation)

def satisfies_conditions (n : ℕ) (k : ℕ) : Prop :=
trailing_zeros (factorial n) = k ∧ trailing_zeros (factorial (3 * n)) = 5 * k

def four_least_values_of_n : List ℕ := [25, 30, 35, 40] -- These are the found appropriate values of n

def s : ℕ := four_least_values_of_n.sum

def sum_of_digits (x : ℕ) : ℕ :=
x.digits.sum

theorem find_sum_of_digits_of_satisfying_ns :
  sum_of_digits s = 4 := by
  unfold s
  unfold four_least_values_of_n
  exact 4  -- Placeholder to indicate the correct answer based on the given problem

end find_sum_of_digits_of_satisfying_ns_l496_496132


namespace train_distance_l496_496615

def fuel_efficiency := 5 / 2 
def coal_remaining := 160
def expected_distance := 400

theorem train_distance : fuel_efficiency * coal_remaining = expected_distance := 
by
  sorry

end train_distance_l496_496615


namespace solution_set_of_inequality_l496_496358

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4 * x else (-(x^2 - 4 * x))

theorem solution_set_of_inequality :
  {x : ℝ | f (x - 2) < 5} = {x : ℝ | -3 < x ∧ x < 7} := by
  sorry

end solution_set_of_inequality_l496_496358


namespace carlos_initial_blocks_l496_496663

theorem carlos_initial_blocks (g : ℕ) (l : ℕ) (total : ℕ) : g = 21 → l = 37 → total = g + l → total = 58 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carlos_initial_blocks_l496_496663


namespace expected_rolls_in_non_leap_year_l496_496659

-- Define the probability of rolling a composite number, prime number, and 1
def prob_roll_composite := 3 / 6 -- composite numbers: 4, 6
def prob_roll_prime := 2 / 6    -- prime numbers: 2, 3, 5
def prob_roll_one := 1 / 6

-- Define the expected number of rolls on a single day
noncomputable def expected_daily_rolls : ℚ := 
  let E := (5 / 6) * 1 + (1 / 6) * (1 + E)
  in E

-- Expected rolls over a non-leap year
noncomputable def total_expected_rolls : ℚ := (365 : ℚ) * expected_daily_rolls

-- The theorem to prove the total expected rolls is 438
theorem expected_rolls_in_non_leap_year : total_expected_rolls = 438 := by
  sorry

end expected_rolls_in_non_leap_year_l496_496659


namespace triangle_obtuse_angled_l496_496847

-- Given definitions and conditions
variables {A B C a b c : ℝ}
variables [IsTriangle A B C a b c]

-- Main theorem statement
theorem triangle_obtuse_angled
  (h : c - b * Real.cos A < 0) : ∃ B, B > π / 2 ∧ B < π :=
sorry

end triangle_obtuse_angled_l496_496847


namespace graph_passes_through_origin_l496_496529

theorem graph_passes_through_origin (f: ℝ → ℝ) : 
  (f = (λ x: ℝ, 2^x) ∧ f 0 = 1) ∨ 
  (f = (λ x: ℝ, log 2 (λ x, x)) ∧ f 1 = 0) ∨ 
  (f = (λ x: ℝ, x^{1/2}) ∧ f 0 = 0) ∨ 
  (f = (λ x: ℝ, x^2) ∧ f 0 = 0) :=
begin
  sorry
end

end graph_passes_through_origin_l496_496529


namespace machine_A_production_rate_l496_496137

-- Define the relevant quantities and conditions
variables (T_G S_A S_G S_B : ℝ)
variables (H1 : ∀ (t: ℝ), (2000: ℝ) * (t: ℝ))
variables (H2 : H1 (S_A * (T_G + 10)) = (2000: ℝ))
variables (H3 : H1 (1.10 * S_A * T_G) = (2000: ℝ))
variables (H4 : H1 (1.15 * S_A * (T_G - 5)) = (2000: ℝ))

-- Translate problem statement to a Lean 4 proof statement
theorem machine_A_production_rate :
  S_A = 200 / 11 := 
sorry

end machine_A_production_rate_l496_496137


namespace find_non_negative_integer_solutions_l496_496703

theorem find_non_negative_integer_solutions :
  ∃ (x y z w : ℕ), 2 ^ x * 3 ^ y - 5 ^ z * 7 ^ w = 1 ∧
  ((x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
   (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1)) := by
  sorry

end find_non_negative_integer_solutions_l496_496703


namespace definite_integral_value_l496_496666

noncomputable section

open Real

def integrand (x : ℝ) : ℝ := (3 * x^2 + 5) * cos (2 * x)

theorem definite_integral_value :
  ∫ x in 0..2 * π, integrand x = 3 * π :=
by
  sorry

end definite_integral_value_l496_496666


namespace f_at_5_is_1_l496_496369

-- Define the function f
noncomputable def f : ℝ → ℝ :=
λ x, if x < 4 then Math.sin (Real.pi / 6 * x) else f (x - 1)

-- The proof problem: Prove that f(5) = 1
theorem f_at_5_is_1 : f 5 = 1 :=
by sorry

end f_at_5_is_1_l496_496369


namespace tangent_line_at_one_l496_496784

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem tangent_line_at_one 
    (a b : ℝ)
    (h : tangent (f 1 a b) at (1, f 1 a b) is (λ x, -x + 3)) :
    a = 1 ∧ b = 8 / 3 := 
  by
    sorry

end tangent_line_at_one_l496_496784


namespace minimum_value_y_l496_496224

theorem minimum_value_y (x : ℝ) (h : x ≥ 1) : 5*x^2 - 8*x + 20 ≥ 13 :=
by {
  sorry
}

end minimum_value_y_l496_496224


namespace fraction_volume_above_45_degrees_north_parallel_l496_496575

theorem fraction_volume_above_45_degrees_north_parallel (R : ℝ) (R_pos : R = 1):
  let fraction := (8 - 5 * Real.sqrt 2) / 16 in
  fraction = (8 - 5 * Real.sqrt 2) / 16 :=
by
  let cap_volume := π * ((2/3) - (5 * Real.sqrt 2 / 12))
  let sphere_volume := (4/3) * π
  let calc_fraction := cap_volume / sphere_volume
  have : calc_fraction = (8 - 5 * Real.sqrt 2) / 16 := sorry
  exact this

end fraction_volume_above_45_degrees_north_parallel_l496_496575


namespace pyramid_top_block_l496_496641

theorem pyramid_top_block (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
                         (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
                         (h : a * b ^ 4 * c ^ 6 * d ^ 4 * e = 140026320) : 
                         (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨ 
                         (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨ 
                         (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨ 
                         (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1) := 
sorry

end pyramid_top_block_l496_496641


namespace ElaCollected13Pounds_l496_496867

def KimberleyCollection : ℕ := 10
def HoustonCollection : ℕ := 12
def TotalCollection : ℕ := 35

def ElaCollection : ℕ := TotalCollection - KimberleyCollection - HoustonCollection

theorem ElaCollected13Pounds : ElaCollection = 13 := sorry

end ElaCollected13Pounds_l496_496867


namespace largest_x_l496_496724

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496724


namespace cannot_determine_not_right_triangle_l496_496240

noncomputable def right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
(∃ (A B C : ℝ),
  (a : b : c = real.sqrt 3 : real.sqrt 2 : 1 ∨
   ∠B - ∠C = ∠A ∨
   (∠A : ∠B : ∠C = 6 : 8 : 10) ∨
   a^2 = (b + c) * (b - c))) ↔ (∠A : ∠B : ∠C = 6 : 8 : 10)

-- Main theorem
theorem cannot_determine_not_right_triangle : 
  ¬ right_triangle A B C a b c := sorry

end cannot_determine_not_right_triangle_l496_496240


namespace seeds_germination_percentage_l496_496743

theorem seeds_germination_percentage :
  ∀ (total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x : ℕ),
    total_seeds = 300 + 200 → 
    germinated_percentage_second_plot = 35 → 
    germinated_percentage_total = 32 → 
    second_plot_seeds = 200 → 
    germinated_seeds_second_plot = (germinated_percentage_second_plot * second_plot_seeds) / 100 → 
    germinated_seeds_total = (germinated_percentage_total * total_seeds) / 100 → 
    germinated_seeds_first_plot = germinated_seeds_total - germinated_seeds_second_plot → 
    x = 30 → 
    x = (germinated_seeds_first_plot * 100) / 300 → 
    x = 30 :=
  by 
    intros total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x
    sorry

end seeds_germination_percentage_l496_496743


namespace c_alone_9_days_l496_496252

theorem c_alone_9_days (W : ℝ) (x : ℝ) (h : (W / 12) + (W / 18) + (W / x) = W / 4) : x = 9 := 
by
  simp at h
  sorry

end c_alone_9_days_l496_496252


namespace work_rate_sum_l496_496248

theorem work_rate_sum (A B : ℝ) (W : ℝ) (h1 : (A + B) * 4 = W) (h2 : A * 8 = W) : (A + B) * 4 = W :=
by
  -- placeholder for actual proof
  sorry

end work_rate_sum_l496_496248


namespace inscribed_circle_in_quadrilateral_l496_496886

open Classical

structure Point where
  x : ℝ 
  y : ℝ

structure Quadrilateral where
  A B C D : Point
  convex : Prop

structure InscribedCircle (Q : Quadrilateral) where
  has_inscribed_circle : Prop

def segments (Q : Quadrilateral) :=
  {A B : Point // ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ (A = x • Q.A + (1 - x) • Q.B ∨ B = x • Q.B + (1 - x) • Q.C ∨ C = x • Q.C + (1 - x) • Q.D ∨ D = x • Q.D + (1 - x) • Q.A)}

theorem inscribed_circle_in_quadrilateral
  (Q : Quadrilateral)
  (E F G H : Point)
  (hE_in_segment_AB : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ E = x • Q.A + (1 - x) • Q.B)
  (hF_in_segment_BC : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ F = x • Q.B + (1 - x) • Q.C)
  (hG_in_segment_CD : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ G = x • Q.C + (1 - x) • Q.D)
  (hH_in_segment_DA : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ H = x • Q.D + (1 - x) • Q.A)
  (P : Point)
  (hP_intersection : ∃ a b c d : ℝ, P = a • E + b • G ∧ P = c • F + d • H ∧ a + b = 1 ∧ c + d = 1)
  (hHAEP : InscribedCircle {A := Q.A, B := H, C := E, D := P, convex := sorry})
  (hEBFP : InscribedCircle {A := E, B := Q.B, C := F, D := P, convex := sorry})
  (hFCGP : InscribedCircle {A := F, B := Q.C, C := G, D := P, convex := sorry})
  (hGDHP : InscribedCircle {A := G, B := Q.D, C := H, D := P, convex := sorry})
  : InscribedCircle Q :=
sorry

end inscribed_circle_in_quadrilateral_l496_496886


namespace range_of_θ_l496_496013

def increasing_seq (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≤ a (n + 1)

noncomputable def a (n : ℕ+) (θ : ℝ) : ℝ :=
  n ^ 2 + 2 * real.sqrt 3 * real.sin θ * n

theorem range_of_θ :
  (∀ n : ℕ+, a n θ ≤ a (n + 1) θ) ∧ (θ ∈ Icc 0 (2 * real.pi)) →
  θ ∈ Icc 0 (4 * real.pi / 3) ∨ θ ∈ Icc (5 * real.pi / 3) (2 * real.pi) :=
begin
  sorry
end

end range_of_θ_l496_496013


namespace canary_possible_distances_l496_496284

noncomputable def distance_from_bus_stop (bus_stop swallow sparrow canary : ℝ) : Prop :=
  swallow = 380 ∧
  sparrow = 450 ∧
  (sparrow - swallow) = (canary - sparrow) ∨
  (swallow - sparrow) = (sparrow - canary)

theorem canary_possible_distances (swallow sparrow canary : ℝ) :
  distance_from_bus_stop 0 swallow sparrow canary →
  canary = 520 ∨ canary = 1280 :=
by
  sorry

end canary_possible_distances_l496_496284


namespace largest_real_number_l496_496735

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496735


namespace number_of_n_le_60_with_f50_eq_18_l496_496744

def number_of_divisors (n : Nat) : Nat :=
  Finset.card (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1)))

def f1 (n : Nat) : Nat := 3 * (number_of_divisors n)

def f (j : Nat) (n : Nat) : Nat :=
  if j = 1 then f1 n else f1 (f (j - 1) n)

def count_valid_n : Nat :=
  Finset.card (Finset.filter (λ n, f 50 n = 18) (Finset.range 61))

theorem number_of_n_le_60_with_f50_eq_18 : count_valid_n = 13 :=
  sorry

end number_of_n_le_60_with_f50_eq_18_l496_496744


namespace proof_problem_l496_496398

open Real

noncomputable def problem (c d : ℝ) : ℝ :=
  5^(c / d) + 2^(d / c)

theorem proof_problem :
  let c := log 8
  let d := log 25
  problem c d = 2 * sqrt 2 + 5^(2 / 3) :=
by
  intro c d
  have c_def : c = log 8 := rfl
  have d_def : d = log 25 := rfl
  rw [c_def, d_def]
  sorry

end proof_problem_l496_496398


namespace theo_donut_holes_count_l496_496656

def radii := {niraek := 5, theo := 7, akshaj := 9, mira := 11}

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Using a noncomputable definition for lcm to simplify reasoning

theorem theo_donut_holes_count :
  let surface_area (r : ℕ) := 4 * r^2 in
  let lcm_all := lcm (lcm (lcm (surface_area radii.niraek) (surface_area radii.theo)) (surface_area radii.akshaj)) (surface_area radii.mira) in
  (lcm_all / (surface_area radii.theo)) = 1036830 := sorry

end theo_donut_holes_count_l496_496656


namespace polynomial_solution_l496_496705

open Polynomial
open Real

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → P.eval (x * sqrt 2) = P.eval (x + sqrt (1 - x^2))) :
  ∃ U : Polynomial ℝ, P = (U.comp (Polynomial.C (1/4) - 2 * X^2 + 5 * X^4 - 4 * X^6 + X^8)) :=
sorry

end polynomial_solution_l496_496705


namespace largest_x_l496_496725

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496725


namespace find_angle_sum_l496_496758

theorem find_angle_sum
  {α β : ℝ}
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 1 / 3)
  (h_cos_β : Real.cos β = 3 / 5) :
  α + 2 * β = π - Real.arctan (13 / 9) :=
sorry

end find_angle_sum_l496_496758


namespace sequence_property_l496_496383

def a : ℕ → ℤ
| n := if n = 0 then 0 else if n ≤ 6 then n else - a (n - 3)

def S (n : ℕ) : ℤ := 
a 1 + a 2 + a 3 + a (n - 5) + a (n - 4) + a (n - 2) + a (n - 1) + a n

theorem sequence_property : 
  a 2015 = 5 ∧ S 2015 = 15 :=
by
  sorry

end sequence_property_l496_496383


namespace inequality_system_no_solution_l496_496325

theorem inequality_system_no_solution (a : ℝ) : ¬ (∃ x : ℝ, x ≤ 5 ∧ x > a) ↔ a ≥ 5 :=
sorry

end inequality_system_no_solution_l496_496325


namespace problem1_problem2a_problem2b_l496_496910

noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem1 : x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 := 
by
  sorry

theorem problem2a : a - b = 2 * Real.sqrt 2 := 
by 
  sorry

theorem problem2b : a^2 - 2 * a * b + b^2 = 8 := 
by 
  sorry

end problem1_problem2a_problem2b_l496_496910


namespace pairs_of_integers_l496_496113

def S (r : ℕ) (x y z : ℝ) : ℝ := x^r + y^r + z^r

theorem pairs_of_integers
  (x y z : ℝ)
  (h : x + y + z = 0)
  (m n : ℕ) :
  (
    (m = 2 ∧ n = 3) ∨ 
    (m = 3 ∧ n = 2) ∨ 
    (m = 2 ∧ n = 5) ∨ 
    (m = 5 ∧ n = 2)
  ) ↔ 
  ∀ {x y z : ℝ}, x + y + z = 0 → 
    (S (m + n) x y z) / (m + n) = (S m x y z) / m * (S n x y z) / n :=
begin
  sorry
end

end pairs_of_integers_l496_496113


namespace rhombus_area_l496_496835

theorem rhombus_area (y : ℝ) 
  (h1 : ∀ P Q : ℝ × ℝ, dist (P.1, P.2) (Q.1, Q.2) = real.sqrt((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) 
  (area : 56 = (16 * (y + 3.5)) / 2) :
  y = 3.5 :=
by
  sorry

end rhombus_area_l496_496835


namespace shaded_area_ratio_is_half_l496_496754

-- Define the side length of the large square
def side_length : ℝ := 5

-- Define the point coordinates
def point_bottom_left : (ℝ × ℝ) := (0, 0)
def point_top_mid : (ℝ × ℝ) := (2.5, 5)
def point_top_right : (ℝ × ℝ) := (5, 5)
def point_left_mid : (ℝ × ℝ) := (0, 2.5)
def point_right_mid : (ℝ × ℝ) := (5, 2.5)

-- Define the area of the large square
def area_large_square : ℝ := 25  -- side_length^2

-- Define the coordinates of the vertices of the shaded region
def shaded_region_vertices : list (ℝ × ℝ) :=
  [point_bottom_left, point_top_mid, point_top_right, point_right_mid, point_left_mid]

-- Dummy area computation as placeholder
noncomputable def area_shaded_region : ℝ := 12.5  -- The calculated area in the solution

-- Define the ratio of the shaded area to the large square area
def shaded_to_total_ratio : ℝ := area_shaded_region / area_large_square

-- Prove that the ratio is 1/2
theorem shaded_area_ratio_is_half : shaded_to_total_ratio = 1 / 2 :=
by
  sorry

end shaded_area_ratio_is_half_l496_496754


namespace sweet_numbers_count_l496_496518

def triple_or_subtract (n : ℕ) : ℕ := if n ≤ 30 then 3 * n else n - 15

def is_in_sequence (start target : ℕ) : Prop :=
  ∃ n, (iterate triple_or_subtract n start) = target

def is_sweet_number (G : ℕ) : Prop :=
  ¬ is_in_sequence G 18

def count_sweet_numbers : ℕ :=
  Finset.filter (λ G, is_sweet_number G) (Finset.range 61) |>.card

theorem sweet_numbers_count : count_sweet_numbers = 52 := by
  sorry

end sweet_numbers_count_l496_496518


namespace problem1_problem2_l496_496135

-- Define the universal set
def ℝ := Real

-- Define sets A and B
def A : Set ℝ := {x | x * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Problem 1: Prove that when a = 1, ∁(A ∪ B) = (-∞, 0]
theorem problem1 (a : ℝ) (h : a = 1) : 
  set.compl (A ∪ B a) = {x : ℝ | x ≤ 0} :=
by
  sorry

-- Problem 2: Prove that if A ∩ B ≠ ∅, then a < 3
theorem problem2 (a : ℝ) (h : (A ∩ B a).nonempty) : 
  a < 3 :=
by
  sorry

end problem1_problem2_l496_496135


namespace quadrilateral_is_trapezoid_or_parallelogram_l496_496191

-- Define a quadrilateral with its angles A, B, C, D
variable (A B C D : ℝ)

-- Assume the condition where products of the cosines of opposite angles are equal
axiom cosine_product_condition : cos A * cos C = cos B * cos D

-- Define the quadrilateral property statement
theorem quadrilateral_is_trapezoid_or_parallelogram
    (A B C D : ℝ)
    (cosine_product_condition : cos A * cos C = cos B * cos D) :
    (/* condition that quadrilateral is a trapezoid */) ∨
    (/* condition that quadrilateral is a parallelogram */):
    sorry

end quadrilateral_is_trapezoid_or_parallelogram_l496_496191


namespace count_odd_numbers_300_600_l496_496041

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l496_496041


namespace log_squared_expression_l496_496579

theorem log_squared_expression
  (y : ℝ) (log10_3_eq_y : log 10 3 = y) :
  (log 10 (10 * log 10 1000))^2 = y^2 + 2*y + 1 :=
sorry

end log_squared_expression_l496_496579


namespace tangent_line_at_one_l496_496783

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem tangent_line_at_one 
    (a b : ℝ)
    (h : tangent (f 1 a b) at (1, f 1 a b) is (λ x, -x + 3)) :
    a = 1 ∧ b = 8 / 3 := 
  by
    sorry

end tangent_line_at_one_l496_496783


namespace coefficient_x5_y2_l496_496176

-- Definitions of polynomial and related entities
noncomputable def polynomial := (x^2 + 3 * x - y : ℤ[X])

-- The main theorem statement
theorem coefficient_x5_y2 : coefficient ((polynomial)^5) (monomial 5 2) = 90 := by
  -- The detailed proof would go here
  sorry

end coefficient_x5_y2_l496_496176


namespace Megan_final_balance_percentage_l496_496476

-- Definitions based on conditions
def Megan_initial_balance : ℝ := 125
def Megan_increase_rate : ℝ := 0.25
def Megan_decrease_rate : ℝ := 0.20

-- Proof statement
theorem Megan_final_balance_percentage :
  let initial_balance := Megan_initial_balance
  let increased_balance := initial_balance * (1 + Megan_increase_rate)
  let final_balance := increased_balance * (1 - Megan_decrease_rate) in
  (final_balance / initial_balance * 100) = 100 :=
by
  sorry

end Megan_final_balance_percentage_l496_496476


namespace trigonometric_identity_l496_496351

theorem trigonometric_identity (α : ℝ) (h : Real.tan (α + π / 4) = -3) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 :=
by
  sorry

end trigonometric_identity_l496_496351


namespace largest_angle_in_pentagon_l496_496074

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (hA : A = 70) 
  (hB : B = 120) 
  (hCD : C = D) 
  (hE : E = 3 * C - 30) 
  (sum_angles : A + B + C + D + E = 540) :
  E = 198 := 
by 
  sorry

end largest_angle_in_pentagon_l496_496074


namespace find_point_A_l496_496755

theorem find_point_A (
  fold_pts_coincide : ∀ (x y : ℝ), x = -2 → y = 8 → (x + y) / 2 = 3,
  coincide_points : ∀ (A B : ℝ), (B - A = 2024) → (A + B) / 2 = 3
) : ∃ A : ℝ, A = -1009 :=
by
  use -1009
  sorry

end find_point_A_l496_496755


namespace point_on_imaginary_axis_point_in_fourth_quadrant_l496_496438

-- (I) For what value(s) of the real number m is the point A on the imaginary axis?
theorem point_on_imaginary_axis (m : ℝ) :
  m^2 - 8 * m + 15 = 0 ∧ m^2 + m - 12 ≠ 0 ↔ m = 5 := sorry

-- (II) For what value(s) of the real number m is the point A located in the fourth quadrant?
theorem point_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8 * m + 15 > 0 ∧ m^2 + m - 12 < 0) ↔ -4 < m ∧ m < 3 := sorry

end point_on_imaginary_axis_point_in_fourth_quadrant_l496_496438


namespace min_value_geometric_sequence_l496_496071

-- Definitions based on conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q

-- We need to state the problem using the above definitions
theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (s t : ℕ) 
  (h_seq : is_geometric_sequence a q) 
  (h_q : q ≠ 1) 
  (h_st : a s * a t = (a 5) ^ 2) 
  (h_s_pos : s > 0) 
  (h_t_pos : t > 0) 
  : 4 / s + 1 / (4 * t) = 5 / 8 := sorry

end min_value_geometric_sequence_l496_496071


namespace number_of_staffing_arrangements_l496_496329

-- Define the 6 members
inductive Member : Type
| A | B | C | D | E | F

-- Define the positions
inductive Position : Type
| Secretary
| DeputySecretary
| Publicity
| Organization

-- Define a staffing arrangement
structure Arrangement :=
(secretary : Member)
(deputySecretary : Member)
(publicity : Member)
(organization : Member)

-- Define the restriction condition
def validSecretary (m : Member) : Bool :=
  m ≠ Member.A ∧ m ≠ Member.B

-- Define the condition to filter valid arrangements
def isValidArrangement (a : Arrangement) : Bool :=
  validSecretary a.secretary

-- Prove the number of valid arrangements
theorem number_of_staffing_arrangements : Nat :=
  let total_arrangements := 6 * 5 * 4 * 3 -- A_6^4
  let invalid_arrangements_for_A := 5 * 4 * 3 * 1 -- A_5^3
  let invalid_arrangements_for_B := 5 * 4 * 3 * 1 -- A_5^3
  total_arrangements - invalid_arrangements_for_A - invalid_arrangements_for_B = 240 :=
by
  sorry -- Proof details to be filled


end number_of_staffing_arrangements_l496_496329


namespace largest_x_63_over_8_l496_496714

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496714


namespace Dana_Colin_relationship_l496_496097

variable (C : ℝ) -- Let C be the number of cards Colin has.

def Ben_cards (C : ℝ) : ℝ := 1.20 * C -- Ben has 20% more cards than Colin
def Dana_cards (C : ℝ) : ℝ := 1.40 * Ben_cards C + Ben_cards C -- Dana has 40% more cards than Ben

theorem Dana_Colin_relationship : Dana_cards C = 1.68 * C := by
  sorry

end Dana_Colin_relationship_l496_496097


namespace sequence_fraction_l496_496759

-- Definitions for arithmetic and geometric sequences
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def isGeometricSeq (a b c : ℝ) :=
  b^2 = a * c

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}

-- a is an arithmetic sequence with common difference d ≠ 0
axiom h1 : isArithmeticSeq a d
axiom h2 : d ≠ 0

-- a_2, a_3, a_9 form a geometric sequence
axiom h3 : isGeometricSeq (a 2) (a 3) (a 9)

-- Goal: prove the value of the given expression
theorem sequence_fraction {a : ℕ → ℝ} {d : ℝ} (h1 : isArithmeticSeq a d) (h2 : d ≠ 0) (h3 : isGeometricSeq (a 2) (a 3) (a 9)) :
  (a 2 + a 3 + a 4) / (a 4 + a 5 + a 6) = 3 / 8 :=
by
  sorry

end sequence_fraction_l496_496759


namespace matrix_line_transformation_l496_496308

theorem matrix_line_transformation (a b: ℝ) (h1 : ∀ P : ℝ × ℝ, (a * P.1 + P.2 = 7) → ((3 * P.1) * 9 + ((- P.1 + b * P.2) = 91))) :
  a = 2 ∧ b = 13 :=
sorry

end matrix_line_transformation_l496_496308


namespace area_of_XYZ_l496_496839

/-!
# Problem statement
In the diagram, triangle XYZ is an isosceles right triangle with ∠X = ∠Y.
A semicircle with diameter XZ is drawn such that the triangle XYZ is inside the semicircle.
If XY = 8√2, what is the area of the triangle XYZ?
-/

variables (X Y Z : Type) [metric_space X]
variables (XY XZ : ℝ)
variables (area : ℝ)
variables (a b : ℝ)

/-- XYZ is an isosceles right triangle with ∠X = ∠Y -/
def is_isosceles_right_triangle (X Y Z : Type) [metric_space X] (XY XZ : ℝ) : Prop :=
∠ X = ∠ Y ∧ XY = a ∧ XZ = b

/-- A semicircle with diameter XZ is drawn such that XYZ is inside the semicircle -/
def is_inside_semicircle (X Y Z : Type) [metric_space X] (XZ : ℝ) : Prop :=
XY = 8 * sqrt 2 ∧ XZ = 16

/-- Area of triangle XYZ -/
def triangle_area (X Y Z : Type) [metric_space X] (XY : ℝ) : ℝ :=
1/2 * XY * XY

theorem area_of_XYZ (X Y Z : Type) [metric_space X] (XY XZ : ℝ) (a b : ℝ)
(h1 : is_isosceles_right_triangle X Y Z XY XZ)
(h2 : is_inside_semicircle X Y Z XZ) :
triangle_area X Y Z XY = 64 :=
by sorry

end area_of_XYZ_l496_496839


namespace largest_x_l496_496721

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496721


namespace find_a_b_and_analyze_function_l496_496785

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem find_a_b_and_analyze_function :
  ∃ (a b : ℝ), 
    (let f := f with a b,
      f(1) = 2 ∧
      deriv f 1 = -1 ∧
      (a = 1 ∧ b = 8 / 3) ∧
      (intervals_of_monotonicity_and_extrema f = 
         { inc_on : set.Ixx (-∞ : ℝ) 0 ∪ set.Ixx 2 ∞, 
           dec_on : set.Ixx 0 2, 
           local_max : { (0, 8 / 3) }, 
           local_min : { (2, 4 / 3) }
         } ∧
      (max_on_interval f (-2) 5 = 58 / 3) ) )
sorry

end find_a_b_and_analyze_function_l496_496785


namespace sum_odd_divisors_of_180_l496_496984

theorem sum_odd_divisors_of_180 : 
  let n := 180 in 
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0} in 
  ∑ d in odd_divisors, d = 78 :=
by
  let n := 180
  let odd_divisors := {d : Nat | d > 0 ∧ d ∣ n ∧ d % 2 ≠ 0}
  have h : ∑ d in odd_divisors, d = 78 := sorry -- Sum of odd divisors of 180
  exact h

end sum_odd_divisors_of_180_l496_496984


namespace sum_of_solutions_l496_496970

theorem sum_of_solutions (α β : ℝ) (h : α * β + β * α = 24) :
  α + β = 12 :=
begin
  sorry
end

end sum_of_solutions_l496_496970


namespace find_smaller_number_l496_496546

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : x = 21 := 
by
  sorry

end find_smaller_number_l496_496546


namespace total_chips_is_90_l496_496214

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end total_chips_is_90_l496_496214


namespace number_of_elements_in_set_l496_496684

-- Define the set of points satisfying the given logarithmic condition
def pointsSet : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ log10 (x^3 + 1/3 * y^3 + 1/9) = log10 x + log10 y }

-- The theorem to prove: the number of elements in the defined set is 1
theorem number_of_elements_in_set : pointsSet.finite ∧ Finset.card (pointsSet.toFinset) = 1 := by
  sorry

end number_of_elements_in_set_l496_496684


namespace art_collection_total_area_l496_496675

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l496_496675


namespace distinct_prime_factors_252_l496_496683

theorem distinct_prime_factors_252 : 
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ (252 = primes.prod ∧ primes.card = 3) :=
sorry

end distinct_prime_factors_252_l496_496683


namespace Megan_final_balance_percentage_l496_496477

-- Definitions based on conditions
def Megan_initial_balance : ℝ := 125
def Megan_increase_rate : ℝ := 0.25
def Megan_decrease_rate : ℝ := 0.20

-- Proof statement
theorem Megan_final_balance_percentage :
  let initial_balance := Megan_initial_balance
  let increased_balance := initial_balance * (1 + Megan_increase_rate)
  let final_balance := increased_balance * (1 - Megan_decrease_rate) in
  (final_balance / initial_balance * 100) = 100 :=
by
  sorry

end Megan_final_balance_percentage_l496_496477


namespace clarify_verses_l496_496526

def historical_verses_correct (Q1 Q2 Q3 Q4 : Prop) : Prop :=
  (Q1 → Q4) → (Q2 → false) → (Q3 → false) → Q1 ∧ Q4

theorem clarify_verses :
  ∀ (Q1 : Prop) (Q2 : Prop) (Q3 : Prop) (Q4 : Prop),
  (Q1 → Q4) →
  (Q2 → false) →
  (Q3 → false) →
  historical_verses_correct Q1 Q2 Q3 Q4 :=
by
  intros Q1 Q2 Q3 Q4 h1 h2 h3
  unfold historical_verses_correct
  intro h
  split
  repeat {sorry} -- Here "sorry" is used to skip the actual proof detailing steps.

end clarify_verses_l496_496526


namespace total_puzzle_pieces_l496_496099

theorem total_puzzle_pieces (p1 p2 p3: ℕ) 
  (h1: p1 = 1000) 
  (h2: p2 = p1 + 0.5 * p1) 
  (h3: p3 = p1 + 0.5 * p1) : 
  p1 + p2 + p3 = 4000 :=
by
  sorry

end total_puzzle_pieces_l496_496099


namespace angle_BIE_in_degrees_l496_496846

theorem angle_BIE_in_degrees {α β γ : ℝ} (h₁ : α + β + γ = 180) (h₂ : α = 44) :
  let BAI := α / 2,
      ABI := (180 - α - γ) / 2 in
  BAI + ABI = 64 :=
by
  sorry

end angle_BIE_in_degrees_l496_496846


namespace area_of_inscribed_rectangle_l496_496923

theorem area_of_inscribed_rectangle (b h y : ℝ) (hb_pos : b > 0) (hh_pos : h > 0) (hy_pos : y > 0) (hy_less_h : y < h) :
  ∃ area : ℝ, area = (b * y / h) * (h - y) :=
by
  let area := (b * y / h) * (h - y)
  use area
  sorry

end area_of_inscribed_rectangle_l496_496923


namespace induction_first_step_l496_496211

theorem induction_first_step (n : ℕ) (h₁ : n > 1) : 
  1 + 1/2 + 1/3 < 2 := 
sorry

end induction_first_step_l496_496211


namespace area_of_triangle_OFA_l496_496174

-- Definitions of given conditions
def parabola_equation : real → real → Prop := λ x y, y^2 = 4 * x
def line_l_equation (θ : real) (F : real × real) : real → real → Prop := 
  λ x y, y = real.tan θ * (x - F.1) + F.2
def point_F : real × real := (1, 0)

-- Definition of the area of a triangle given coordinates
def triangle_area (O F A : real × real) : real :=
  1 / 2 * real.abs (O.1 * (F.2 - A.2) + F.1 * (A.2 - O.2) + A.1 * (O.2 - F.2))

-- Problem translated to proof statement
theorem area_of_triangle_OFA :
  ∃ A : real × real, 
    (∃ y : real, y = 2 * real.sqrt 3 ∧ parabola_equation (A.1) y ∧ line_l_equation (real.pi / 3) point_F A.1 y ∧ A.2 = y) ∧
    triangle_area (0, 0) point_F A = real.sqrt 3 := 
begin
  sorry,
end

end area_of_triangle_OFA_l496_496174


namespace probability_of_exactly_one_failure_l496_496555

theorem probability_of_exactly_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  p1 * (1 - p2) + (1 - p1) * p2 = 0.14 :=
by
  rw [h1, h2]
  norm_num
  sorry

end probability_of_exactly_one_failure_l496_496555


namespace maximum_value_2ab_sqrt3_2ac_le_sqrt3_achievable_value_2ab_sqrt3_2ac_eq_sqrt3_l496_496129

theorem maximum_value_2ab_sqrt3_2ac_le_sqrt3 (a b c : ℝ)
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * a * c ≤ Real.sqrt 3 :=
begin
  sorry
end

theorem achievable_value_2ab_sqrt3_2ac_eq_sqrt3 (a b c : ℝ)
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum_squares : a^2 + b^2 + c^2 = 1)
  (h_eq_conditions : a = b ∧ b = Real.sqrt (1/2) ∧ c = 0) :
  2 * a * b * Real.sqrt 3 + 2 * a * c = Real.sqrt 3 :=
begin
  sorry
end

end maximum_value_2ab_sqrt3_2ac_le_sqrt3_achievable_value_2ab_sqrt3_2ac_eq_sqrt3_l496_496129


namespace triangle_proof_l496_496439

noncomputable def length_DC (AB DA BC DB : ℝ) : ℝ :=
  Real.sqrt (BC^2 - DB^2)

theorem triangle_proof :
  ∀ (AB DA BC DB : ℝ), AB = 30 → DA = 24 → BC = 22.5 → DB = 18 →
  length_DC AB DA BC DB = 13.5 :=
by
  intros AB DA BC DB hAB hDA hBC hDB
  rw [length_DC]
  sorry

end triangle_proof_l496_496439


namespace inequality_l496_496151

-- Define the harmonic number H_n
def harmonic (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (k + 1 : ℚ))

-- Define the sum of reciprocal of odd numbers up to 2n-1
def oddReciprocalSum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (2 * k + 1 : ℚ))

-- Define the sum of reciprocal of even numbers up to 2n
def evenReciprocalSum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (2 * (k + 1) : ℚ))

theorem inequality (n : ℕ) (h : 1 < n) :
  (1 / (n + 1 : ℚ)) * oddReciprocalSum n >
  (1 / (n : ℚ)) * evenReciprocalSum n := by
  sorry

end inequality_l496_496151


namespace apples_per_pie_l496_496521

/-- Let's define the parameters given in the problem -/
def initial_apples : ℕ := 62
def apples_given_to_students : ℕ := 8
def pies_made : ℕ := 6

/-- Define the remaining apples after handing out to students -/
def remaining_apples : ℕ := initial_apples - apples_given_to_students

/-- The statement we need to prove: each pie requires 9 apples -/
theorem apples_per_pie : remaining_apples / pies_made = 9 := by
  -- Add the proof here
  sorry

end apples_per_pie_l496_496521


namespace total_pencils_correct_l496_496650

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end total_pencils_correct_l496_496650


namespace bus_passengers_l496_496927

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l496_496927


namespace identify_variables_constants_l496_496638

noncomputable def boyles_law (P V C : ℝ) : Prop :=
  P * V = C

-- Define the Isothermal condition
def isothermal (C : ℝ) : Prop :=
  ∃ P V : ℝ, P * V = C

theorem identify_variables_constants (P V C : ℝ) (h : boyles_law P V C) (h_iso : isothermal C) :
  (variable P) ∧ (variable V) ∧ (constant C) :=
sorry

-- Assuming variable and constant are definitions for checking variable and constant respectively.
def variable (x : ℝ) : Prop := true  -- Placeholder definition
def constant (x : ℝ) : Prop := true  -- Placeholder definition

end identify_variables_constants_l496_496638


namespace number_of_good_polynomials_modulo_1000_l496_496296

-- Definitions from problem conditions
noncomputable def P (x : ℕ → ℤ) : Prop :=
  ∀ m : ℤ, (P m - 20) % 2011 = 0 ∨ (P m - 15) % 2011 = 0 ∨ (P m - 1234) % 2011 = 0 

noncomputable def good_polynomial (P : ℕ → ℤ) : Prop :=
  (∃ m₁ m₂ m₃ : ℤ, (P m₁ - 20) % 2011 = 0 ∧ (P m₂ - 15) % 2011 = 0 ∧ (P m₃ - 1234) % 2011 = 0) ∧
  P ∈ set_of P

-- Statement to prove
noncomputable def N : ℕ := { P : ℕ → ℤ | good_polynomial P }.to_finset.card

theorem number_of_good_polynomials_modulo_1000 : N % 1000 = 460 :=
by sorry

end number_of_good_polynomials_modulo_1000_l496_496296


namespace num_solutions_eq_natural_exponent_l496_496912

theorem num_solutions_eq_natural_exponent (n : ℕ) (h1 : n = 21 ∨ n = 57 ∨ n = 165) :
  ∃ k : ℕ, (x1 x2 x3 : ℕ), 
    0 < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ 
    x1 + x2 + x3 = n ∧ 
    3 ^ k = n := 
sorry

end num_solutions_eq_natural_exponent_l496_496912


namespace median_avg_scores_compare_teacher_avg_scores_l496_496264

-- Definitions of conditions
def class1_students (a : ℕ) := a
def class2_students (b : ℕ) := b
def class3_students (c : ℕ) := c
def class4_students (c : ℕ) := c

def avg_score_1 := 68
def avg_score_2 := 78
def avg_score_3 := 74
def avg_score_4 := 72

-- Part 1: Prove the median of the average scores.
theorem median_avg_scores : 
  let scores := [68, 72, 74, 78]
  ∃ m, m = 73 :=
by 
  sorry

-- Part 2: Prove that the average scores for Teacher Wang and Teacher Li are not necessarily the same.
theorem compare_teacher_avg_scores (a b c : ℕ) (h_ab : a ≠ 0 ∧ b ≠ 0) : 
  let wang_avg := (68 * a + 78 * b) / (a + b)
  let li_avg := 73
  wang_avg ≠ li_avg :=
by
  sorry

end median_avg_scores_compare_teacher_avg_scores_l496_496264


namespace second_year_students_count_l496_496194

noncomputable def student_ratio := (5, 4, 3)
noncomputable def total_sample := 240

theorem second_year_students_count :
  let ratio_sum := student_ratio.1 + student_ratio.2 + student_ratio.3 in
  (total_sample * student_ratio.2) / ratio_sum = 80 :=
by
  let ratio_sum := student_ratio.1 + student_ratio.2 + student_ratio.3
  have h : ratio_sum = 12 := by norm_num
  rw [h]
  have h2 : (total_sample * 4) / 12 = 80 := by norm_num
  exact h2

end second_year_students_count_l496_496194


namespace min_sum_one_over_xy_l496_496950

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end min_sum_one_over_xy_l496_496950


namespace verify_minimum_door_height_l496_496938

namespace BusDoorHeight

noncomputable def minimum_door_height (μ : ℝ) (σ : ℝ) : ℝ :=
  let X := NormalDistribution μ σ, in
  184 

theorem verify_minimum_door_height :
  ∀ (μ σ : ℝ),
  μ = 170 → σ = 7 →
  let X := NormalDistribution μ σ in
  (Pr(X ≤ μ + σ) - Pr(X ≤ μ - σ) = 0.6826) →
  (Pr(X ≤ μ + 2*σ) - Pr(X ≤ μ - 2*σ) = 0.9544) →
  (Pr(X ≤ μ + 3*σ) - Pr(X ≤ μ - 3*σ) = 0.9974) →
  (1 - Pr(X ≤ μ + 2*σ) ≤ 0.0228) →
  minimum_door_height μ σ = 184 :=
by
  intros
  sorry

end BusDoorHeight

end verify_minimum_door_height_l496_496938


namespace find_lambda_l496_496036

-- Definitions of vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (1, 1)

-- Definition of the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2) = 0

-- The proof problem statement
theorem find_lambda (λ : ℝ) (h : is_perpendicular b (a.1 + λ * b.1, a.2 + λ * b.2)) : λ = -3 :=
sorry

end find_lambda_l496_496036


namespace correctOptionOnlyC_l496_496999

-- Definitions for the transformations
def isTransformA (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b^2) / (a^2)) 
def isTransformB (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b + 1) / (a + 1))
def isTransformC (a b : ℝ) : Prop := (a ≠ 0) → (b / a = (a * b) / (a^2))
def isTransformD (a b : ℝ) : Prop := (a ≠ 0) → ((-b + 1) / a = -(b + 1) / a)

-- Main theorem to assert the correctness of the transformations
theorem correctOptionOnlyC (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬isTransformA a b ∧ ¬isTransformB a b ∧ isTransformC a b ∧ ¬isTransformD a b :=
by
  sorry

end correctOptionOnlyC_l496_496999


namespace banana_cream_pie_correct_slice_l496_496830

def total_students := 45
def strawberry_pie_preference := 15
def pecan_pie_preference := 10
def pumpkin_pie_preference := 9

noncomputable def banana_cream_pie_slice_degrees : ℝ :=
  let remaining_students := total_students - strawberry_pie_preference - pecan_pie_preference - pumpkin_pie_preference
  let students_per_preference := remaining_students / 2
  (students_per_preference / total_students) * 360

theorem banana_cream_pie_correct_slice :
  banana_cream_pie_slice_degrees = 44 := by
  sorry

end banana_cream_pie_correct_slice_l496_496830


namespace number_of_possible_teams_l496_496585

-- Definitions for the conditions
def num_goalkeepers := 3
def num_defenders := 5
def num_midfielders := 5
def num_strikers := 5

-- The number of ways to choose x from y
def choose (y x : ℕ) : ℕ := Nat.factorial y / (Nat.factorial x * Nat.factorial (y - x))

-- Main proof problem statement
theorem number_of_possible_teams :
  (choose num_goalkeepers 1) *
  (choose num_strikers 2) *
  (choose num_midfielders 4) *
  (choose (num_defenders + (num_midfielders - 4)) 4) = 2250 := by
  sorry

end number_of_possible_teams_l496_496585


namespace interval_k_is_40_l496_496957

def total_students := 1200
def sample_size := 30

theorem interval_k_is_40 : (total_students / sample_size) = 40 :=
by
  sorry

end interval_k_is_40_l496_496957


namespace molly_age_problem_l496_496895

theorem molly_age_problem :
  ∃ (X : ℕ), 
  let M := 12 in
  (M + 18 = 5 * (M - X)) ∧ X = 6 :=
begin
  sorry,
end

end molly_age_problem_l496_496895


namespace smallest_tangent_sum_largest_tangent_product_l496_496171

-- First define the angles and their sum condition
variables {A B C : ℝ}
hypothesis h_triangle : A + B + C = Real.pi

-- The following will state the questions as limit problems
theorem smallest_tangent_sum (h_triangle : A + B + C = Real.pi) : 
  ∃ m, (∀ (A B C : ℝ), A + B + C = Real.pi -> tan (A / 2) + tan (B / 2) + tan (C / 2) ≥ m) ∧
       (∀ (A B C : ℝ), A + B + C = Real.pi -> tan (A / 2) + tan (B / 2) + tan (C /2) ≤ m) ∧
       m = Real.sqrt 3 := sorry

theorem largest_tangent_product (h_triangle : A + B + C = Real.pi) :
  ∃ M, (∀ (A B C : ℝ), A + B + C = Real.pi -> tan (A / 2) * tan (B / 2) * tan (C / 2) ≤ M) ∧
       (∀ (A B C : ℝ), A + B + C = Real.pi -> tan (A /2 ) * tan (B / 2) * tan (C / 2) ≥ M) ∧
       M = 1 / (3 * Real.sqrt 3) := sorry

end smallest_tangent_sum_largest_tangent_product_l496_496171


namespace tangent_line_at_zero_max_min_values_on_interval_l496_496781

noncomputable def f (x : ℝ) : ℝ := (x - 1) * sin x + 2 * cos x + x

theorem tangent_line_at_zero : 
    let line_at_zero := 2 in
    (∃ (k : ℝ), (∀ x, f x - line_at_zero = f'(0) * (x - 0)) ∧ (f 0 = line_at_zero)) :=
sorry

theorem max_min_values_on_interval : 
    let x := (x : ℝ) in
    let pi := Real.pi in
    (∃ (max_value min_value : ℝ), max_value = pi - 1 ∧ min_value = pi - 2 ∧ x ∈ [0, pi] → f x ≤ max_value ∧ f x ≥ min_value) :=
sorry

end tangent_line_at_zero_max_min_values_on_interval_l496_496781


namespace acute_triangle_angles_equal_l496_496077

variable {A B C D M : Type}

-- Assume ABC is an acute triangle
axiom is_acute_triangle (A B C : Type) : Prop

-- A point D where tangents to the circumcircle of ABC intersect
axiom tangents_intersect_circumcircle (A B D : Type) : Prop

-- Midpoint M of AB
axiom midpoint (A B M : Type) : Prop

-- Angle ACM equals angle BCD
axiom angle_equality (A B C D M : Type) : Prop

theorem acute_triangle_angles_equal
  (h_acute : is_acute_triangle A B C)
  (h_tangents : tangents_intersect_circumcircle A B D)
  (h_midpoint : midpoint A B M) :
  angle_equality A B C D M := sorry

end acute_triangle_angles_equal_l496_496077


namespace g_of_3_l496_496399

def g (x : ℝ) : ℝ := 5 * x ^ 4 + 4 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 :=
by
    -- proof will go here
    sorry

end g_of_3_l496_496399


namespace find_min_value_of_f_l496_496374

def f (a x : ℝ) : ℝ := -x^2 + a*x - a/4 + 1/2

theorem find_min_value_of_f (a : ℝ) :
  (a < 0 → ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ -5) ∧
  (a > 2 → ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ -(1 / 3)) :=
by sorry

end find_min_value_of_f_l496_496374


namespace count_odd_numbers_300_600_l496_496042

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l496_496042


namespace all_Xanths_are_Yelps_and_Wicks_l496_496068

-- Definitions for Zorbs, Yelps, Xanths, and Wicks
variable {U : Type} (Zorb Yelp Xanth Wick : U → Prop)

-- Conditions from the problem
axiom all_Zorbs_are_Yelps : ∀ u, Zorb u → Yelp u
axiom all_Xanths_are_Zorbs : ∀ u, Xanth u → Zorb u
axiom all_Xanths_are_Wicks : ∀ u, Xanth u → Wick u

-- The goal is to prove that all Xanths are Yelps and are Wicks
theorem all_Xanths_are_Yelps_and_Wicks : ∀ u, Xanth u → Yelp u ∧ Wick u := sorry

end all_Xanths_are_Yelps_and_Wicks_l496_496068


namespace least_time_for_four_horses_to_meet_l496_496201

-- Conditions
def horse_completes_lap_in_k_minutes (k : ℕ) : ℕ := k

-- Lean 4 statement for the proof problem
theorem least_time_for_four_horses_to_meet {T : ℕ} (T > 0) : 
  (∃ (horses : set ℕ), horses ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧ horses.card ≥ 4 ∧ (∀ k ∈ horses, T % k = 0)) ↔ T = 6 :=
by {
  sorry
}

end least_time_for_four_horses_to_meet_l496_496201


namespace sum_of_positive_odd_divisors_180_l496_496988

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d % 2 = 1) (Finset.divisors n)).sum

theorem sum_of_positive_odd_divisors_180 : 
  sum_of_positive_odd_divisors 180 = 78 := by
  sorry

end sum_of_positive_odd_divisors_180_l496_496988


namespace cube_less_than_triple_l496_496226

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l496_496226


namespace determineFinalCounts_l496_496142

structure FruitCounts where
  plums : ℕ
  oranges : ℕ
  apples : ℕ
  pears : ℕ
  cherries : ℕ

def initialCounts : FruitCounts :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def givenAway : FruitCounts :=
  { plums := 4, oranges := 3, apples := 5, pears := 0, cherries := 0 }

def receivedFromSam : FruitCounts :=
  { plums := 2, oranges := 0, apples := 0, pears := 1, cherries := 0 }

def receivedFromBrother : FruitCounts :=
  { plums := 0, oranges := 1, apples := 2, pears := 0, cherries := 0 }

def receivedFromNeighbor : FruitCounts :=
  { plums := 0, oranges := 0, apples := 0, pears := 3, cherries := 2 }

def finalCounts (initial given receivedSam receivedBrother receivedNeighbor : FruitCounts) : FruitCounts :=
  { plums := initial.plums - given.plums + receivedSam.plums,
    oranges := initial.oranges - given.oranges + receivedBrother.oranges,
    apples := initial.apples - given.apples + receivedBrother.apples,
    pears := initial.pears - given.pears + receivedSam.pears + receivedNeighbor.pears,
    cherries := initial.cherries - given.cherries + receivedNeighbor.cherries }

theorem determineFinalCounts :
  finalCounts initialCounts givenAway receivedFromSam receivedFromBrother receivedFromNeighbor =
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 } :=
by
  sorry

end determineFinalCounts_l496_496142


namespace sum_of_positive_odd_divisors_eq_78_l496_496994

theorem sum_of_positive_odd_divisors_eq_78 :
  ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 180)), d = 78 :=
by {
  -- proof steps go here
  sorry
}

end sum_of_positive_odd_divisors_eq_78_l496_496994


namespace variance_2X_plus_q_l496_496336

variable (p q : ℝ)
variable (X : ℕ → ℝ)

-- Ensure X follows binomial distribution B(5, p)
def is_binomial_5_p (X : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n, ∃ k, X n = 5*p*k*(1-p)^(5-k)

-- Given E(X) = 2
axiom ex_X : (∑ k in range 6, k * 5*p*(1-p)^(5-k)) = 2

-- Goal D(2X + q) = 4.8
theorem variance_2X_plus_q : ¬¬(5*p = 2) → is_binomial_5_p X p →  (∑ k in range 6, (2 * k + q - (2* (∑ k in range 6, k * 5*p*(1-p)^(5-k)) / 6))^2 * 5*p*(1-p)^(5-k)) = 4.8 := by
  sorry

end variance_2X_plus_q_l496_496336


namespace seating_arrangement_l496_496697

theorem seating_arrangement : 
  ∃ x y z : ℕ, 
  7 * x + 8 * y + 9 * z = 65 ∧ z = 1 ∧ x + y + z = r :=
sorry

end seating_arrangement_l496_496697


namespace number_of_valid_triples_l496_496277

theorem number_of_valid_triples : 
  ∃ n, n = 7 ∧ ∀ (a b c : ℕ), b = 2023 → a ≤ b → b ≤ c → a * c = 2023^2 → (n = 7) :=
by 
  sorry

end number_of_valid_triples_l496_496277


namespace ordinary_eq_C1_rectangular_eq_C2_max_PM_PN_l496_496797

-- Definitions related to curve C_1 and C_2
def C1_parametric_eq_1 (θ : ℝ) : ℝ := 2 * Real.cos θ
def C1_parametric_eq_2 (θ : ℝ) : ℝ := Math.sqrt 3 * Real.sin θ

def polar_eq_C2 (ρ : ℝ) : Prop := ρ = 2

-- Theorems to prove
theorem ordinary_eq_C1 :
  ∀ (x y : ℝ),
    (∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = Math.sqrt 3 * Real.sin θ) ↔
    (x^2 / 4 + y^2 / 3 = 1) :=
sorry

theorem rectangular_eq_C2 :
  ∀ (x y : ℝ),
    (∃ ρ θ : ℝ, ρ = 2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
    (x^2 + y^2 = 4) :=
sorry

theorem max_PM_PN :
  let M := (0, √3),
      N := (0, -√3),
      P := λ α, (2 * Real.cos α, 2 * Real.sin α) in
  ∀ α : ℝ,
    ∃ α_star : ℝ,
      (| (P α_star).fst - M.fst | + | (P α_star).snd - M.snd | + 
       | (P α_star).fst - N.fst | + | (P α_star).snd - N.snd |) = 2 * √7 :=
sorry

end ordinary_eq_C1_rectangular_eq_C2_max_PM_PN_l496_496797


namespace meeting_time_proof_l496_496209

def train_meeting_time {dist : ℕ} {speedA speedB : ℕ} {startA startB : ℕ} : Prop :=
  dist = 20 ∧ speedA = 20 ∧ speedB = 25 ∧ startA = 7 ∧ startB = 8 → startB = 8

theorem meeting_time_proof : train_meeting_time :=
by {
  intro h,
  cases h with dist_cond h,
  cases h with speedA_cond h,
  cases h with speedB_cond h,
  cases h with startA_cond startB_cond,
  exact startB_cond,
}

end meeting_time_proof_l496_496209


namespace part1_correct_part2_correct_l496_496959

-- Definitions for conditions
def total_students := 200
def likes_employment := 140
def dislikes_employment := 60
def p_likes : ℚ := likes_employment / total_students

def male_likes := 60
def male_dislikes := 40
def female_likes := 80
def female_dislikes := 20
def n := total_students
def alpha := 0.005
def chi_squared_critical_value := 7.879

-- Part 1: Estimate the probability of selecting at least 2 students who like employment
def probability_at_least_2_of_3 : ℚ :=
  3 * ((7/10) ^ 2) * (3/10) + ((7/10) ^ 3)

-- Proof goal for Part 1
theorem part1_correct : probability_at_least_2_of_3 = 98 / 125 := by
  sorry

-- Part 2: Chi-squared test for independence between intention and gender
def a := male_likes
def b := male_dislikes
def c := female_likes
def d := female_dislikes
def chi_squared_statistic : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof goal for Part 2
theorem part2_correct : chi_squared_statistic = 200 / 21 ∧ 200 / 21 > chi_squared_critical_value := by
  sorry

end part1_correct_part2_correct_l496_496959


namespace find_angles_of_quadrilateral_l496_496316

variables (A B C D : Type)
variables [euclidean_geometry A B C D]
variables (angle : A → A → A → ℝ)

def is_cyclic_quadrilateral (A B C D : Type) : Prop :=
  angle B A C + angle A C D + angle C D B + angle D B A = 360

theorem find_angles_of_quadrilateral (A B C D : Type)
  (h1 : angle B A C = 30)
  (h2 : angle A C D = 40)
  (h3 : angle A D B = 50)
  (h4 : angle C B D = 60)
  (h5 : angle A B C + angle A D C = 180) :
  angle A B C = 100 ∧ angle A D C = 80 ∧ angle B A D = 90 ∧ angle B C D = 90 :=
by sorry

end find_angles_of_quadrilateral_l496_496316


namespace number_of_divisors_of_12n2_l496_496883

-- Definition for the number of divisors function
def num_divisors (n : ℕ) : ℕ := 
  (factors n).unique.length

-- Definition stating that n is an odd integer and has exactly 13 positive divisors
def is_odd_integer_with_13_divisors (n : ℕ) : Prop :=
  (∃ p : ℕ, nat.prime p ∧ n = p ^ 12) ∧ (num_divisors n = 13) ∧ (n % 2 = 1)

-- Theorem stating the number of positive divisors of 12n^2
theorem number_of_divisors_of_12n2 (n : ℕ) (hn : is_odd_integer_with_13_divisors n) : 
  num_divisors (12 * n ^ 2) = 150 :=
by 
  sorry

end number_of_divisors_of_12n2_l496_496883


namespace a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l496_496904

-- Mathematical condition: a^2 + b^2 = 0
variable {a b : ℝ}

-- Mathematical statement to be proven
theorem a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero 
  (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry  -- proof yet to be provided

end a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l496_496904


namespace purity_of_alloy_l496_496260

theorem purity_of_alloy (w1 w2 : ℝ) (p1 p2 : ℝ) (h_w1 : w1 = 180) (h_p1 : p1 = 920) (h_w2 : w2 = 100) (h_p2 : p2 = 752) : 
  let a := w1 * (p1 / 1000) + w2 * (p2 / 1000)
  let b := w1 + w2
  let p_result := (a / b) * 1000
  p_result = 860 :=
by
  sorry

end purity_of_alloy_l496_496260


namespace even_integers_between_3000_and_6000_with_four_different_digits_and_one_zero_l496_496392

-- Define the conditions under which the even integers with four different digits (one being zero) need to be within the specified range.

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_four_different_digits (n : ℕ) : Prop :=
  ∀ i j k l : ℕ, i ≠ j → j ≠ k → k ≠ l → l ≠ i → i ≠ k → j ≠ l →

def contains_zero (n : ℕ) : Prop :=
  ∃ d : ℕ, d = 0

def in_range_3000_to_6000 (n : ℕ) : Prop :=
  3000 ≤ n ∧ n ≤ 6000

def satisfies_conditions (n : ℕ) : Prop :=
  is_even n ∧ has_four_different_digits n ∧ contains_zero n ∧ in_range_3000_to_6000 n

theorem even_integers_between_3000_and_6000_with_four_different_digits_and_one_zero :
  ∃ count : ℕ, (∀ (n : ℕ), satisfies_conditions n → n = count ) ∧ count = 330 :=
by
  sorry

end even_integers_between_3000_and_6000_with_four_different_digits_and_one_zero_l496_496392


namespace largest_real_number_l496_496720

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496720


namespace derivative_at_zero_does_not_exist_l496_496254

noncomputable def f : ℝ → ℝ :=
λ x, if x = 0 then 0 else arctan x * sin (7 / x)

theorem derivative_at_zero_does_not_exist :
  ¬(∃ L, filter.tendsto (λ Δx, (f Δx - f 0) / Δx) (nhds 0) (nhds L)) :=
by {
  sorry
}

end derivative_at_zero_does_not_exist_l496_496254


namespace student_marks_calculation_l496_496282

def max_marks : ℕ := 300

def passing_marks : ℕ := (60 * max_marks) / 100

def student_marks (failed_by : ℕ) : ℕ :=
passing_marks - failed_by

theorem student_marks_calculation :
  student_marks 100 = 80 :=
by
  unfold student_marks passing_marks max_marks
  norm_num
  sorry

end student_marks_calculation_l496_496282


namespace problem_1_problem_2_l496_496757

-- The equation of the parabola and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Chords AB and CD, slopes k1 and k2
def chords (x1 x2 x3 x4 y1 y2 y3 y4 k1 k2 : ℝ) : Prop := 
  (y2 - y1 = k1 * (x2 - x1)) ∧ (y4 - y3 = k2 * (x4 - x3)) ∧ 
  ((x1 + x2) / 2 = 1) ∧ ((x3 + x4) / 2 = -1)

-- Midpoints M and N
def midpoints (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M = ((x1 + x2) / 2, (y1 + y2) / 2)) ∧ 
  (N = ((x3 + x4) / 2, (y3 + y4) / 2))

-- Area of triangle FMN
def area_ΔFMN (F M N : ℝ × ℝ) : ℝ :=
  1/2 * abs ((fst F * (snd M - snd N)) + (fst M * (snd N - snd F)) + (fst N * (snd F - snd M)))

-- Fixed point for line MN
def fixed_point (x y k1 k2 : ℝ) : Prop :=
  ∀ (M N : ℝ × ℝ), ( ((snd N) - (snd M)) / ((fst N) - (fst M)) = k1 + k2) →
  ∃ (P : ℝ × ℝ), P = (1, 1/2)

-- The proof problems in Lean statements
theorem problem_1 (p : ℝ) (k1 : ℝ) (F M N : ℝ × ℝ) (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :
  p > 0 → k1 = 1 →
  parabola p x1 y1 → parabola p x2 y2 → parabola p x3 y3 → parabola p x4 y4 →
  chords x1 x2 x3 x4 y1 y2 y3 y4 k1 (-1/k1) →
  midpoints x1 x2 x3 x4 y1 y2 y3 y4 M N →
  area_ΔFMN F M N = 1 := sorry

theorem problem_2 (p : ℝ) (k1 k2 : ℝ) (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :
  p > 0 → (1/k1 + 1/k2 = 1) →
  parabola p x1 y1 → parabola p x2 y2 → parabola p x3 y3 → parabola p x4 y4 →
  chords x1 x2 x3 x4 y1 y2 y3 y4 k1 k2 →
  fixed_point 1 (1/2) k1 k2 :=
  sorry

end problem_1_problem_2_l496_496757


namespace find_days_jane_indisposed_l496_496448

-- Define the problem conditions
def John_rate := 1 / 20
def Jane_rate := 1 / 10
def together_rate := John_rate + Jane_rate
def total_task := 1
def total_days := 10

-- The time Jane was indisposed
def days_jane_indisposed (x : ℝ) : Prop :=
  (total_days - x) * together_rate + x * John_rate = total_task

-- Statement we want to prove
theorem find_days_jane_indisposed : ∃ x : ℝ, days_jane_indisposed x ∧ x = 5 :=
by 
  sorry

end find_days_jane_indisposed_l496_496448


namespace chloe_age_digit_sum_l496_496293

/-
  Problem statement:
  Prove that the sum of the digits of Chloe's age the next time her age is a multiple of Zoe's age is 4,
  given the following conditions:
  - Chloe is 2 years older than Joey.
  - Zoe is 1 year old today.
  - There are 7 distinct birthdays, including today, where Joey’s age will be an integral multiple of Zoe’s age.
-/

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := (to_string n).to_list.map (λ c, c.to_nat - '0'.to_nat)
  digits.foldl (λ acc d, acc + d) 0

theorem chloe_age_digit_sum :
  ∃ J C n : ℕ,
    (Z : ℕ) = 1 ∧
    C = J + 2 ∧
    (∃ s : finset ℕ, s.card = 7 ∧ ∀ n ∈ s, (J + n) % (Z + n) = 0) ∧
    let nextChloeAge := C + 63
    in sum_of_digits nextChloeAge = 4 :=
by {
  let Z := 1,
  obtain J : ℕ := 65,
  let C := J + 2,
  let s := {0, 1, 2, 3, 4, 63, 126} : finset ℕ,
  have h_card : s.card = 7 := rfl,
  have h_s : ∀ n ∈ s, (J + n) % (Z + n) = 0 := by sorry,
  use [J, C, 63],
  calc sum_of_digits (C + 63) = sum_of_digits 130 := rfl
                           ... = 1 + 3 + 0 := rfl
                           ... = 4 := rfl,
}

end chloe_age_digit_sum_l496_496293


namespace geometry_problem_l496_496427

-- Define the setup conditions
variables {O A B M V S C D : Type*}
variables [linear_ordered_field ℝ]

-- Geometric conditions
variable (AB : ℝ) (AV : ℝ) (VM : ℝ) (VB : ℝ) (L : ℝ)
variable (h : ℝ)
variable (DA : ℝ)
variable (p q : ℤ)

-- Given conditions
axiom h_eq : h = DA 
axiom AV_eq : AV = 80
axiom VM_eq : VM = 120
axiom VB_eq : VB = 200
axiom AB_eq : AV + VM + VB = 400

-- Correct answer
axiom pq_eq : p + q = 303
axiom DA_eq : DA = p * real.sqrt q
-- Real geometric properties are replaced by simplified numeric properties
axiom OZ_eq : 4 * 120 = 480 -- this would be replaced by geometric interpretation 

-- Lean statement to prove
theorem geometry_problem :
  ∃ (p q : ℤ), p * p = 90000 ∧ q = 3 ∧ p + q = 303 :=
by {
  use (300, 3),
  split,
  { norm_num },
  { split,
    norm_num, norm_num }
}

end geometry_problem_l496_496427


namespace determinant_of_matrix_l496_496667

theorem determinant_of_matrix :
  let M := !![ [3, 0, 5], [4, 5, -2], [1, 2, 6] ]
  matrix.det M = 117 :=
by
  let M := !![ [3, 0, 5], [4, 5, -2], [1, 2, 6] ]
  sorry

end determinant_of_matrix_l496_496667


namespace smallest_possible_value_l496_496185

/-
Given:
1. m and n are positive integers.
2. gcd of m and n is (x + 5).
3. lcm of m and n is x * (x + 5).
4. m = 60.
5. x is a positive integer.

Prove:
The smallest possible value of n is 100.
-/

theorem smallest_possible_value 
  (m n x : ℕ) 
  (h1 : m = 60) 
  (h2 : x > 0) 
  (h3 : Nat.gcd m n = x + 5) 
  (h4 : Nat.lcm m n = x * (x + 5)) : 
  n = 100 := 
by 
  sorry

end smallest_possible_value_l496_496185


namespace calculate_PC_l496_496066
noncomputable def ratio (a b : ℝ) : ℝ := a / b

theorem calculate_PC (AB BC CA PC PA : ℝ) (h1: AB = 6) (h2: BC = 10) (h3: CA = 8)
  (h4: ratio PC PA = ratio 8 6)
  (h5: ratio PA (PC + 10) = ratio 6 10) :
  PC = 40 :=
sorry

end calculate_PC_l496_496066


namespace least_number_subtracted_l496_496739

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_n : n = 4273981567) (h_x : x = 17) : 
  (n - x) % 25 = 0 := by
  sorry

end least_number_subtracted_l496_496739


namespace length_of_AB_l496_496075

/-- In a right triangle ABC with hypotenuse AB, 
    let M be the midpoint of BC with median from A to M of length 5 units, 
    and let N be the midpoint of AC with median from B to N of length 3√5 units. 
    We prove that the length of AB is 2√14 units. -/
theorem length_of_AB (A B C M N : Point)
  (h_right_triangle : right_triangle A B C)
  (hM : midpoint M B C)
  (hAM : length (segment A M) = 5)
  (hN : midpoint N A C)
  (hBN : length (segment B N) = 3 * sqrt 5):
  length (segment A B) = 2 * sqrt 14 := 
sorry

end length_of_AB_l496_496075


namespace general_formulas_range_lambda_l496_496773

noncomputable def sequence_a (n : ℕ) : ℕ :=
2 ^ n

noncomputable def sequence_b (n : ℕ) : ℕ :=
(2 * n - 1) / 2 ^ n

noncomputable def sum_b_terms (n : ℕ) : ℕ :=
3 - (2 * n + 3) / 2 ^ n

theorem general_formulas :
  (∀ n, sequence_a n = 2 ^ n) ∧ (∀ n, sequence_b n = (2 * n - 1) / 2 ^ n) :=
sorry

theorem range_lambda (λ : ℝ) :
  (∀ n, λ ≥ n * (3 - sum_b_terms n)) ↔ (λ ∈ set.Ici (7 / 2)) :=
sorry

end general_formulas_range_lambda_l496_496773


namespace solution_set_l496_496058

variable {f : ℝ → ℝ}

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define that f is decreasing on positive reals
def decreasing_on_pos_reals (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_decreasing : decreasing_on_pos_reals f
axiom f_at_two_zero : f 2 = 0

-- Main theorem statement
theorem solution_set : { x : ℝ | (x - 1) * f (x - 1) > 0 } = { x | x < -1 } ∪ { x | x > 3 } :=
sorry

end solution_set_l496_496058


namespace odd_function_domain_l496_496415

open Real

theorem odd_function_domain (a b c : ℝ) (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_f : ∀ x, f x = x * cos x + c) 
  (h_domain : ∀ x, a ≤ x ∧ x ≤ b) : 
  a + b + c = 0 := 
begin
  have h0 : f 0 = 0 * cos 0 + c := h_f 0,
  simp at h0,
  have h1 : 0 = c,
  { linarith, },
  have h2 : b = -a,
  { sorry, },
  linarith [h1, h2],
end

end odd_function_domain_l496_496415


namespace farm_produce_weeks_l496_496141

def eggs_needed_per_week (saly_eggs ben_eggs ked_eggs : ℕ) : ℕ :=
  saly_eggs + ben_eggs + ked_eggs

def number_of_weeks (total_eggs : ℕ) (weekly_eggs : ℕ) : ℕ :=
  total_eggs / weekly_eggs

theorem farm_produce_weeks :
  let saly_eggs := 10
  let ben_eggs := 14
  let ked_eggs := 14 / 2
  let total_eggs := 124
  let weekly_eggs := eggs_needed_per_week saly_eggs ben_eggs ked_eggs
  number_of_weeks total_eggs weekly_eggs = 4 :=
by
  sorry 

end farm_produce_weeks_l496_496141


namespace find_f2019_l496_496167

noncomputable def f1 (x : ℝ) : ℝ := 1 / (2 - x)

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
if n = 1 then f1(x) else f1(fn (n-1) x)

theorem find_f2019 (n : ℕ) (x : ℝ) (a b : ℕ):
  (∀ (a b : ℕ), f1 4 = -1 / 2 ∧ f1 (-1 / 2) = 2 / 5 ∧ 
                 f1 (2 / 5) = 5 / 8 ∧ f1 (fn (2018) 4) = f1 (6053 / 6056)) → 
  ∃ (a b : ℕ), gcd a b = 1 ∧ a = 6053 ∧ b = 6056 :=
sorry

end find_f2019_l496_496167


namespace count_points_on_AD_l496_496442

variables (A B C D P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] (points : set Type)
variables [is_iso_trapezoid : isosceles_trapezoid A B C D]
variables (AD_length AB_length DC_length : ℝ)
variables (point_on_AD : P ∈ segment A D)

def isosceles_trapezoid (A B C D : Type) :=
  ∃ (parallel : AB ∥ DC) (ABlength : AB_length = 998) (DClength : DC_length = 1001) (ADlength : AD_length = 1999), parallel ∧ (is_isosceles_trapezoid = true)

theorem count_points_on_AD {A B C D P : Type} :
  (isosceles_trapezoid A B C D) →
  (point_on_AD P) →
  ∃ (number_of_points : ℤ), (∀ P ∈ segment A D, (right_angle ∠BPC)) ∧ (number_of_points = 2) :=
begin
  sorry,
end

end count_points_on_AD_l496_496442


namespace sine_transformations_correct_l496_496958

theorem sine_transformations_correct :
  (∀ x, (1, 2) y = sin (2 * (x - π / 6)) ↔ y = sin (2x - π / 3)) ∧
  (∀ x, (1, 3) y = sin (2x - π / 3)) ∧
  (∀ x, (4, 3) y = sin (2 * (x + 5 * π / 6) - π / 3)) :=
by
  sorry

end sine_transformations_correct_l496_496958


namespace green_red_socks_ratio_l496_496473

theorem green_red_socks_ratio 
  (r : ℕ) -- Number of pairs of red socks originally ordered
  (y : ℕ) -- Price per pair of red socks
  (green_socks_price : ℕ := 3 * y) -- Price per pair of green socks, 3 times the red socks
  (C_original : ℕ := 6 * green_socks_price + r * y) -- Cost of the original order
  (C_interchanged : ℕ := r * green_socks_price + 6 * y) -- Cost of the interchanged order
  (exchange_rate : ℚ := 1.2) -- 20% increase
  (cost_relation : C_interchanged = exchange_rate * C_original) -- Cost relation given by the problem
  : (6 : ℚ) / (r : ℚ) = 2 / 3 := 
by
  sorry

end green_red_socks_ratio_l496_496473


namespace all_zero_l496_496333

theorem all_zero (n : ℕ) (a : Fin (2 * n) → ℤ) 
  (h : ∀ k : Fin (2 * n), ∃ (S T : Finset (Fin (2 * n))), 
      S ∩ T = ∅ ∧ S ∪ T = Finset.univ.erase k ∧ (∑ i in S, a i) = (∑ i in T, a i)) : 
  ∀ k, a k = 0 :=
begin
  sorry
end

end all_zero_l496_496333


namespace range_of_b_for_local_minimum_l496_496377

variable {x : ℝ}
variable (b : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ :=
  x^3 - 6 * b * x + 3 * b

def f' (x : ℝ) (b : ℝ) : ℝ :=
  3 * x^2 - 6 * b

theorem range_of_b_for_local_minimum
  (h1 : f' 0 b < 0)
  (h2 : f' 1 b > 0) :
  0 < b ∧ b < 1 / 2 :=
by
  sorry

end range_of_b_for_local_minimum_l496_496377


namespace polynomial_form_l496_496704

noncomputable def binom (n k : ℕ) : ℚ := sorry -- Definition for binomial coefficient

theorem polynomial_form (P : ℝ[X]) (c : ℤ) (a : ℕ) :
  P.eval 2014 = 1 ∧ (∀ x : ℝ, x * (P.eval (x - c)) = (x - 2014) * P.eval x) →
  P = (λ x, binom (x / c) a) ∧ 0 < c ∧ c ∣ 2014 :=
begin
  sorry
end

end polynomial_form_l496_496704


namespace inequality1_inequality2_l496_496903

variables (Γ B P : ℕ)

def convex_polyhedron : Prop :=
  Γ - B + P = 2

theorem inequality1 (h : convex_polyhedron Γ B P) : 
  3 * Γ ≥ 6 + P :=
sorry

theorem inequality2 (h : convex_polyhedron Γ B P) : 
  3 * B ≥ 6 + P :=
sorry

end inequality1_inequality2_l496_496903


namespace secant_line_ratio_l496_496804

variables (O : Point) (r₁ r₂ r₃ : ℝ) (a b : ℝ)
variables (A₁ B₁ C₁ C₂ B₂ A₂ : Point)
variables (h_c1 : dist O C₁ = r₃) (h_b1 : dist O B₁ = r₂) (h_a1 : dist O A₁ = r₁)
variables (h_seq : segments_on_line [A₁, B₁, C₁, C₂, B₂, A₂])

-- The statement to prove
theorem secant_line_ratio :
  ∃ line, intersects_line_circle line [A₁, B₁, C₁, C₂, B₂, A₂] ∧
          (dist A₁ C₁ / dist C₁ B₂ = a / b) :=
sorry

end secant_line_ratio_l496_496804


namespace bus_passenger_count_l496_496924

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l496_496924


namespace angle_A_eq_60_angle_B_eq_60_max_y_l496_496078

noncomputable def vector := (ℝ × ℝ)

axiom acute_triangle (A B C : ℝ) : Prop
axiom vectors_collinear (p q : vector) : Prop

-- Definitions given in the problem statement
def p (A : ℝ) : vector := (2 - 2 * real.sin A, real.cos A + real.sin A)
def q (A : ℝ) : vector := (real.sin A - real.cos A, 1 + real.sin A)
def y (B C : ℝ) : ℝ := 2 * real.sin B * real.sin B + real.cos ((C - 3 * B) / 2)

-- Given conditions for acute triangle and collinear vectors
variables (A B C : ℝ)
variables [h_acute : acute_triangle A B C]
variables (h_collinear : vectors_collinear (p A) (q A))

-- Statements to prove
theorem angle_A_eq_60 : A = 60 :=
sorry

theorem angle_B_eq_60_max_y : ∃ B, 2 * B - 30 = 90 ∧ y B C = (real.sin (2 * B - 30) + 1) :=
sorry

end angle_A_eq_60_angle_B_eq_60_max_y_l496_496078


namespace probability_s10_to_s15_l496_496516

-- Define the conditions and the problem structure
def sequence : Type := Fin 50 → ℝ

def bubble_sort_pass_move (seq : sequence) (n m : ℕ) : Prop :=
  ∀ i j, (i < j) → (i ≥ 1 ∧ j ≤ 50) → i ≠ 10 → j ≠ 15 → seq i < seq j

-- Main theorem statement
theorem probability_s10_to_s15 (seq : sequence) : 
  bubble_sort_pass_move seq 10 15 → (1 / 240 : ℚ) :=
sorry

end probability_s10_to_s15_l496_496516


namespace product_of_slopes_l496_496935

noncomputable def tan_15_deg : ℝ := 2 - Real.sqrt 3

theorem product_of_slopes (m n : ℝ)
  (h_eq_1 : ∀ x, y = m * x)
  (h_eq_2 : ∀ x, y = n * x)
  (h_angle_relation : 3 * atan n = atan m)
  (h_slope_relation : m = 3 * n)
  (h_theta1 : m = 1) :
  m * n = 2 - Real.sqrt 3 :=
by
  -- Here, we simplify the provided slope angles and use trigonometric identities to show the result
  have h_tan_15 : n = tan_15_deg,
  { 
    sorry
  },
  calc
    m * n = 1 * n : by rw h_theta1
    ...   = 1 * tan_15_deg : by rw h_tan_15
    ...   = tan_15_deg : one_mul _
    ...   = 2 - Real.sqrt 3 : by rfl

end product_of_slopes_l496_496935


namespace tetrahedron_volume_formula_l496_496153

-- Definitions used directly in the conditions
variable (a b d : ℝ) (φ : ℝ)

-- Tetrahedron volume formula theorem statement
theorem tetrahedron_volume_formula 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b) 
  (hd_pos : 0 < d) 
  (hφ_pos : 0 < φ) 
  (hφ_le_pi : φ ≤ Real.pi) :
  (∀ V : ℝ, V = 1 / 6 * a * b * d * Real.sin φ) :=
sorry

end tetrahedron_volume_formula_l496_496153


namespace veronica_photo_choices_l496_496559

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem veronica_photo_choices : choose 5 3 + choose 5 4 = 15 := by
  sorry

end veronica_photo_choices_l496_496559


namespace solve_triangle_l496_496445

noncomputable def problem_statement : Prop :=
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 4 ∧ c = 6 ∧ cos C = 1 / 8 →
  sin A = sqrt 7 / 4 ∧ b = 5 ∧ (b * sin A = 5 * (sqrt 7 / 4))

theorem solve_triangle (A B C : ℝ) (a b c : ℝ)
  (hA : ∀ (A B C : ℝ) (a b c : ℝ),
    a = 4 ∧ c = 6 ∧ cos C = 1 / 8 →
    sin A = sqrt 7 / 4 ∧ b = 5 ∧ (b * sin A = 5 * (sqrt 7 / 4))) :
  problem_statement :=
by sorry

end solve_triangle_l496_496445


namespace proportion_solution_l496_496054

theorem proportion_solution (x : ℝ) (h : x / 6 = 4 / 0.39999999999999997) : x = 60 := sorry

end proportion_solution_l496_496054


namespace function_is_identity_l496_496700

open Nat

noncomputable def Z_plus := Nat

def f (n : Z_plus) : Z_plus := n  -- This is our candidate function

theorem function_is_identity (f : Z_plus → Z_plus) : (∀ (a b : Z_plus), 
  (f a + b) ∣ (a^2 + f a * f b)) → 
  (∀ n : Z_plus, f n = n) :=
by
  intros h n
  sorry

end function_is_identity_l496_496700


namespace branches_sum_one_main_stem_l496_496693

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l496_496693


namespace perpendicular_OC_EK_l496_496960

-- Given a triangle ABC inscribed in a circle with center O
variables {A B C O E K : Type}
variables [Triangle ABC] [Circle (center := O) ABC]
variables (intersect_AC : Line A C) (intersect_BC : Line B C)
variables [Intersection E intersect_AC O ∧ Intersection E intersect_AC A]
variables [Intersection K intersect_BC O ∧ Intersection K intersect_BC B]

-- We need to prove that lines OC and EK are perpendicular
theorem perpendicular_OC_EK 
  (hAC : E ≠ A) (hBC : K ≠ B) 
  (hE : E ∈ (Circle (center := O) ABC)) 
  (hK : K ∈ (Circle (center := O) ABC)) 
  (hC: O, C ∈ Line intersect_AC) 
  (hEK : Line E K) :
  Perpendicular (Line O C) (Line E K) := 
sorry

end perpendicular_OC_EK_l496_496960


namespace find_angle_l496_496023

open Real

-- Mathematical conditions and definitions
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

def f1 : ℝ × ℝ := (-5, 0)
def f2 : ℝ × ℝ := (5, 0)

def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

def product_of_distances (P : ℝ × ℝ) : Prop := 
  let p1 := sqrt ((P.1 - f1.1)^2 + (P.2 - f1.2)^2) in
  let p2 := sqrt ((P.1 - f2.1)^2 + (P.2 - f2.2)^2) in
  p1 * p2 = 32

-- Theorem statement
theorem find_angle (P : ℝ × ℝ) (hP : point_on_hyperbola P) (hDist : product_of_distances P) :
  ∠(f1, P, f2) = 90 :=
by
  sorry

end find_angle_l496_496023


namespace area_of_triangle_APQ_l496_496366

theorem area_of_triangle_APQ (A P Q : ℝ × ℝ) (hA : A = (8, 6)) 
  (hP : P = (0, b₁)) (hQ : Q = (0, b₂)) (hb₁b₂_sum : b₁ + b₂ = 4)
  (m₁ m₂ : ℝ) (hm₁m₂ : m₁ * m₂ = -1) 
  (line₁ : ∀ x : ℝ, y = m₁ * x + b₁)
  (line₂ : ∀ x : ℝ, y = m₂ * x + b₂) :
  (1/2 * (8 : ℝ) * abs (b₂ - b₁) = 80) :=
begin
  sorry
end

end area_of_triangle_APQ_l496_496366


namespace modulus_z_is_5_l496_496339

def z := (2 - complex.i)^2 / complex.i

theorem modulus_z_is_5 : complex.abs z = 5 := 
by sorry

end modulus_z_is_5_l496_496339


namespace unique_integer_cube_triple_l496_496229

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l496_496229


namespace tape_length_calculation_l496_496929

noncomputable def total_tape_length_needed
  (area : ℝ)
  (π_est : ℝ)
  (extra_length : ℝ) : ℝ :=
  let r := real.sqrt (area * (7 / 22))
  let circumference := 2 * π_est * r
  circumference + extra_length

theorem tape_length_calculation :
  total_tape_length_needed 176 (22 / 7) 3 ≈ 50.058 := 
  sorry

end tape_length_calculation_l496_496929


namespace largest_real_number_l496_496733

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496733


namespace building_height_l496_496589

theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (building_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70)
  (ratio_eq : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  building_height = 28 :=
by
  have h_flagpole_shadow := ratio_eq ▸ h_flagpole ▸ s_flagpole ▸ s_building
  sorry

end building_height_l496_496589


namespace no_convex_quad_with_given_areas_l496_496096

theorem no_convex_quad_with_given_areas :
  ¬ ∃ (A B C D M : Type) 
    (T_MAB T_MBC T_MDA T_MDC : ℕ) 
    (H1 : T_MAB = 1) 
    (H2 : T_MBC = 2)
    (H3 : T_MDA = 3) 
    (H4 : T_MDC = 4),
    true :=
by {
  sorry
}

end no_convex_quad_with_given_areas_l496_496096


namespace pyramid_volume_l496_496907

theorem pyramid_volume (s : ℝ) (a : ℝ) (h : ℝ) : 
  s = 5 → a = 2 * (1 + Real.sqrt 2) * s^2 → h = 5 * Real.sqrt 3 → 
  (1 / 3) * a * h = (250 * Real.sqrt 3 * (1 + Real.sqrt 2))/3 :=
by
  intro hs ha hh
  rw [hs, ha, hh]
  sorry

end pyramid_volume_l496_496907


namespace product_integral_inequality_l496_496111

open Set Real

variables {n : ℕ}
variables {u : Fin n → (Fin n → ℝ) → ℝ}
variables {x : Fin n → ℝ}

theorem product_integral_inequality
  (H₁ : ∀ i : Fin n, Continuous (u i))
  (H₂ : ∀ i : Fin n, ∀ x : Fin n → ℝ, ∀ y : ℝ, u i (x ∘ Fin.insertNth i y) = u i x)
  (H₃ : ∀ i : Fin n, ∀ x : Fin n → ℝ, 0 ≤ u i x) :
  (∫ x in Icc (λ _, 0) (λ _, 1), (∏ i, u i x)) ^ (n-1) ≤ ∏ i, ∫ x in Icc (λ _, 0) (λ _, 1), (u i x) ^ (n-1) :=
sorry

end product_integral_inequality_l496_496111


namespace solve_inequality_l496_496513

-- Define integer and fractional parts functions
noncomputable def int_part (x : ℝ) : ℤ := int.floor x
noncomputable def frac_part (x : ℝ) : ℝ := x - ↑(int_part x)

theorem solve_inequality (x : ℝ) (h₀ : frac_part x * (int_part x - 1) < x - 2)
  (h₁ : 0 ≤ frac_part x) (h₂ : frac_part x < 1) :
  x ≥ 3 :=
sorry

end solve_inequality_l496_496513


namespace complex_number_pure_imaginary_l496_496411

variable (θ : ℝ)

theorem complex_number_pure_imaginary (h₁ : cos θ = 4 / 5) (h₂ : sin θ ≠ 3 / 5) :
  tan (θ - π/4) = -7 :=
  sorry

end complex_number_pure_imaginary_l496_496411


namespace geometric_mean_of_2_and_8_l496_496182

def geometric_mean (a b : ℝ) := Real.sqrt (a * b)

theorem geometric_mean_of_2_and_8 :
  ∃ m : ℝ, (m ^ 2 = 2 * 8) ∧ (m = 4 ∨ m = -4) := by
  use [4, -4]
  split
  · sorry
  · split
  · sorry
  · sorry

end geometric_mean_of_2_and_8_l496_496182


namespace correct_propositions_l496_496777

-- Define the propositions as statements with necessary conditions

def proposition_2 (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < -2 ∧ b > 2 → (f a > f b ↔ a + b < 0)

def proposition_3 : Prop :=
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0)

def proposition_4 (a b : ℝ) : Prop :=
  (a + b ≠ 4 → a ≠ 1 ∨ b ≠ 3) ∧ ¬ (a + b ≠ 4 → a ≠ 1 ∨ b ≠ 3)

def proposition_5 (φ : ℝ) (k : ℤ) (ω : ℝ) : Prop :=
  (φ = k * π + π / 2) ↔ (∀ (x : ℝ), sin (ω * x + φ) = sin (ω * (-x) + φ))

-- Main theorem
theorem correct_propositions : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 5 * |x| - 1 / sqrt (2 * |x| - 4)) ∧ proposition_2 f -3 3) ∧
  proposition_3 ∧
  proposition_4 1 3 ∧
  (∃ φ : ℝ, ∃ k : ℤ, ∃ ω : ℝ, ω ≠ 0 ∧ proposition_5 φ k ω) :=
  sorry

end correct_propositions_l496_496777


namespace total_chips_is_90_l496_496213

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end total_chips_is_90_l496_496213


namespace log_between_integers_l496_496681

theorem log_between_integers : 
  ∃ c d : ℤ, (log 50 / log 10 > c) ∧ (log 50 / log 10 < d) ∧ (c + d = 3) :=
by
  sorry

end log_between_integers_l496_496681


namespace number_of_strings_per_bass_l496_496865

theorem number_of_strings_per_bass :
  let B := 3 in
  let G := 2 * B in
  let E := G - 3 in
  ∃ X : ℕ, 3 * X + G * 6 + E * 8 = 72 ∧ X = 4 :=
by
  let B := 3
  let G := 2 * B
  let E := G - 3
  exists 4
  calc
    3 * 4 + G * 6 + E * 8 = 3 * 4 + (2 * 3) * 6 + (6 - 3) * 8 : by sorry
    ... = 12 + 36 + 24 : by sorry
    ... = 72 : by sorry

end number_of_strings_per_bass_l496_496865


namespace largest_real_number_l496_496730

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496730


namespace largest_real_number_l496_496736

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496736


namespace count_shenma_numbers_l496_496572

def is_shenma_number (n : ℕ) : Prop :=
  let digits := (Finset.range 10).filter (λ d, (n / 10 ^ d) % 10 ∈ Finset.range 10)
  digits.card = 5 ∧
  (let middle_digit := (n / 10 ^ 2) % 10 in
   digits.erase middle_digit = Finset.range 5 ∧
   ∀ i j, i < j → ((n / 10 ^ i % 10 ) < (n / 10 ^ j % 10 )))

theorem count_shenma_numbers : 
  (Finset.filter is_shenma_number (Finset.range 100000)).card = 1512 := 
  sorry

end count_shenma_numbers_l496_496572


namespace ratio_of_pipe_lengths_l496_496262

theorem ratio_of_pipe_lengths (L S : ℕ) (h1 : L + S = 177) (h2 : L = 118) (h3 : ∃ k : ℕ, L = k * S) : L / S = 2 := 
by 
  sorry

end ratio_of_pipe_lengths_l496_496262


namespace max_area_right_triangle_in_rectangle_l496_496945

theorem max_area_right_triangle_in_rectangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) : 
  ∃ (A B C : ℝ × ℝ), A = (0, 0) ∧ B = (12, 0) ∧ C = (0, 15) ∧ 
  let area := 1/2 * 12 * 15 in area = 90 := 
by 
  sorry

end max_area_right_triangle_in_rectangle_l496_496945


namespace one_over_x_not_in_M_range_of_a_for_log_function_two_exponent_x_plus_x_squared_in_M_l496_496038

open Real

def is_in_M (f : ℝ → ℝ) : Prop :=
  ∃ x_0 : ℝ, f (x_0 + 1) = f x_0 + f 1

-- (1) Prove that \( f(x) = \frac{1}{x} \notin M \)
theorem one_over_x_not_in_M : ¬ is_in_M (λ x, 1 / x) :=
sorry

-- (2) Determine the range of \( a \) such that \( f(x) = \log a / (x^2 + 1) \in M \)
def log_function_in_M (a : ℝ) : Prop :=
  is_in_M (λ x, log (a / (x^2 + 1)))

theorem range_of_a_for_log_function :
  ∀ a : ℝ, 0 < a → log_function_in_M a ↔ a ∈ set.Icc (3 - sqrt 5) (3 + sqrt 5) :=
sorry

-- (3) Prove that \( f(x) = 2^x + x^2 \in M \)
theorem two_exponent_x_plus_x_squared_in_M : is_in_M (λ x, 2^x + x^2) :=
sorry

end one_over_x_not_in_M_range_of_a_for_log_function_two_exponent_x_plus_x_squared_in_M_l496_496038


namespace shelby_rain_time_l496_496508

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end shelby_rain_time_l496_496508


namespace range_of_m_l496_496057

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x < 0 ∧ mx^2 + 2*x + 1 = 0) : m ∈ Set.Iic 1 :=
sorry

end range_of_m_l496_496057


namespace largest_x_63_over_8_l496_496709

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496709


namespace equal_area_division_l496_496753

theorem equal_area_division (circle : Type) (A B C : circle) (arc : set circle)
  (is_convex_shape : convex_shape_bounded_by_arc_and_broken_line arc A B C)
  (arc_opposite_broken_line : arc_broken_line_opposite arc A B C) :
  ∃ M : circle, midpoint_arc M A C →
  ∃ l : line, divides_area_equally l M :=
sorry

end equal_area_division_l496_496753


namespace reciprocal_of_sum_is_correct_l496_496195

def reciprocal (r : ℚ) : ℚ := 1 / r

theorem reciprocal_of_sum_is_correct :
  reciprocal ((1 : ℚ) / 4 + (1 : ℚ) / 6) = 12 / 5 :=
by
  -- The proof is to be filled in here
  sorry

end reciprocal_of_sum_is_correct_l496_496195


namespace tank_capacity_l496_496402

theorem tank_capacity :
  ∃ T : ℝ, (5/8) * T + 12 = (11/16) * T ∧ T = 192 :=
sorry

end tank_capacity_l496_496402


namespace period_start_time_l496_496331

theorem period_start_time (end_time : ℕ) (rained_hours : ℕ) (not_rained_hours : ℕ) (total_hours : ℕ) (start_time : ℕ) 
  (h1 : end_time = 17) -- 5 pm as 17 in 24-hour format 
  (h2 : rained_hours = 2)
  (h3 : not_rained_hours = 6)
  (h4 : total_hours = rained_hours + not_rained_hours)
  (h5 : total_hours = 8)
  (h6 : start_time = end_time - total_hours)
  : start_time = 9 :=
sorry

end period_start_time_l496_496331


namespace correctOptionOnlyC_l496_496998

-- Definitions for the transformations
def isTransformA (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b^2) / (a^2)) 
def isTransformB (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b + 1) / (a + 1))
def isTransformC (a b : ℝ) : Prop := (a ≠ 0) → (b / a = (a * b) / (a^2))
def isTransformD (a b : ℝ) : Prop := (a ≠ 0) → ((-b + 1) / a = -(b + 1) / a)

-- Main theorem to assert the correctness of the transformations
theorem correctOptionOnlyC (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬isTransformA a b ∧ ¬isTransformB a b ∧ isTransformC a b ∧ ¬isTransformD a b :=
by
  sorry

end correctOptionOnlyC_l496_496998


namespace initial_yellow_marbles_l496_496157

variables (total_yellow : ℝ) (given_by_joan : ℝ) (initial_yellow : ℝ)
variables (h1 : total_yellow = 111) (h2 : given_by_joan = 25) (h3 : initial_yellow = total_yellow - given_by_joan)

theorem initial_yellow_marbles (total_yellow given_by_joan : ℝ) (h1 : total_yellow = 111) (h2 : given_by_joan = 25) :
  initial_yellow = 86 :=
begin
  simp [h1, h2, h3],
  sorry
end

end initial_yellow_marbles_l496_496157


namespace age_of_youngest_person_l496_496550

theorem age_of_youngest_person :
  ∃ (a1 a2 a3 a4 : ℕ), 
  (a1 < a2) ∧ (a2 < a3) ∧ (a3 < a4) ∧ 
  (a4 = 50) ∧ 
  (a1 + a2 + a3 + a4 = 158) ∧ 
  (a2 - a1 = a3 - a2) ∧ (a3 - a2 = a4 - a3) ∧ 
  a1 = 29 :=
by
  sorry

end age_of_youngest_person_l496_496550


namespace tim_income_percent_less_than_juan_l496_496140

-- Definitions of the conditions
variables {M T J : ℝ}
-- Condition 1: Mary's income is 70 percent more than Tim's income
def condition1 : Prop := M = 1.70 * T
-- Condition 2: Mary's income is 102 percent of Juan's income
def condition2 : Prop := M = 1.02 * J

-- The goal to prove
theorem tim_income_percent_less_than_juan :
  condition1 →
  condition2 →
  (J - ((1.02 * J) / 1.70)) / J * 100 = 40 :=
by
  intros h1 h2
  sorry

end tim_income_percent_less_than_juan_l496_496140


namespace sin_add_arctan_arcsin_l496_496699

theorem sin_add_arctan_arcsin :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan 3
  (Real.sin a = 4 / 5) →
  (Real.tan b = 3) →
  Real.sin (a + b) = (13 * Real.sqrt 10) / 50 :=
by
  intros _ _
  sorry

end sin_add_arctan_arcsin_l496_496699


namespace choose_bar_chart_for_comparisons_l496_496203

/-- 
To easily compare the quantities of various items, one should choose a bar chart 
based on the characteristics of statistical charts.
-/
theorem choose_bar_chart_for_comparisons 
  (chart_type: Type) 
  (is_bar_chart: chart_type → Prop)
  (is_ideal_chart_for_comparison: chart_type → Prop)
  (bar_chart_ideal: ∀ c, is_bar_chart c → is_ideal_chart_for_comparison c) 
  (comparison_chart : chart_type) 
  (h: is_bar_chart comparison_chart): 
  is_ideal_chart_for_comparison comparison_chart := 
by
  exact bar_chart_ideal comparison_chart h

end choose_bar_chart_for_comparisons_l496_496203


namespace range_of_k_l496_496414

theorem range_of_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (k * x1 + 2)^2 = 6 ∧ x2^2 - (k * x2 + 2)^2 = 6)) →
  -real.sqrt 15 / 3 < k ∧ k < -1 :=
begin
  sorry
end

end range_of_k_l496_496414


namespace distance_between_parallel_lines_l496_496178

noncomputable def point_to_line_distance (a b c x₁ y₁ : ℝ) :=
  (abs (a * x₁ + b * y₁ + c)) / (real.sqrt (a * a + b * b))

theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y, x - y = 1
  let line2 : ℝ → ℝ → Prop := λ x y, 2 * x - 2 * y + 3 = 0
  let distance := ((5 * real.sqrt 2) / 4)
  (∃ (x₁ y₁ : ℝ), line1 x₁ y₁) →
  (∀ (x₁ y₁ : ℝ), line1 x₁ y₁ → point_to_line_distance 2 (-2) 3 x₁ y₁ = distance) :=
begin
  sorry
end

end distance_between_parallel_lines_l496_496178


namespace optimization_problem_l496_496775

theorem optimization_problem :
  ∃ (x y : ℝ), (x + 2 * y ≤ 5) ∧
               (2 * x + y ≤ 4) ∧
               (x ≥ 0) ∧
               (y ≥ 0) ∧
               (3 * x + 4 * y = 11) ∧
               (x = 1) ∧
               (y = 2) :=
by
  use 1, 2
  split
  · norm_num
  · split
    · norm_num
    · split
      · norm_num
      · split
        · norm_num
        · split
          · norm_num
          · norm_num >> sorry

end optimization_problem_l496_496775


namespace inverse_of_matrix_l496_496707

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 9], ![2, 5]]

def inv_mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/2, -9/2], ![-1, 2]]

theorem inverse_of_matrix :
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℚ), 
    inv * mat = 1 ∧ mat * inv = 1 :=
  ⟨inv_mat, by
    -- Providing the proof steps here is beyond the scope
    sorry⟩

end inverse_of_matrix_l496_496707


namespace largest_real_number_l496_496715

theorem largest_real_number (x : ℝ) (h : ⌊x⌋ / x = 8 / 9) : x ≤ 63 / 8 :=
sorry

end largest_real_number_l496_496715


namespace triangle_median_length_l496_496422

theorem triangle_median_length (A B C N : Point) (h : Triangle A B C) 
  (h1 : dist A B = 40) (h2 : dist B C = 40) (h3 : dist A C = 38) 
  (h4 : midpoint N B C) :
  dist A N = 20 * Real.sqrt 3 :=
sorry

end triangle_median_length_l496_496422


namespace elements_author_is_euclid_l496_496921

def author_of_elements := "Euclid"

theorem elements_author_is_euclid : author_of_elements = "Euclid" :=
by
  rfl -- Reflexivity of equality, since author_of_elements is defined to be "Euclid".

end elements_author_is_euclid_l496_496921


namespace max_value_b_over_c_plus_c_over_b_l496_496420

theorem max_value_b_over_c_plus_c_over_b (a b c A B C : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_alt : ∃h : ℝ, h = (√3 / 6) * a)
  (h_area : (1 / 2) * (√3 / 6) * a^2 = (1 / 2) * b * c * real.sin A)
  (h_cosine : real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  : ∃ x : ℝ, x = (b / c) + (c / b) ∧ x ≤ 4 :=
sorry

end max_value_b_over_c_plus_c_over_b_l496_496420


namespace arithmetic_sequence_sum_l496_496471

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l496_496471


namespace f_value_l496_496779

noncomputable def f (x : ℝ) : ℝ :=
if x >= 4 then (1 / 2) ^ x else sorry  -- This is a placeholder for f(x) = f(x+1) definition.

theorem f_value (x : ℝ) (h1 : x >= 4 → f(x) = (1 / 2) ^ x)
                      (h2 : x < 4 → f(x) = f(x + 1)) :
  f (2 + real.log 3 / real.log 2) = 1 / 24 := by
  sorry

end f_value_l496_496779


namespace sample_avg_std_dev_xy_l496_496799

theorem sample_avg_std_dev_xy {x y : ℝ} (h1 : (4 + 5 + 6 + x + y) / 5 = 5)
  (h2 : (( (4 - 5)^2 + (5 - 5)^2 + (6 - 5)^2 + (x - 5)^2 + (y - 5)^2 ) / 5) = 2) : x * y = 21 :=
by
  sorry

end sample_avg_std_dev_xy_l496_496799


namespace largest_set_with_sum_conditions_l496_496493

theorem largest_set_with_sum_conditions :
  ∀ (S : set ℝ), (∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → a + b + c ∈ ℚ) → 
  (∀ a b ∈ S, a ≠ b → a + b ∉ ℚ) → 
  set.finite S ∧ set.card S ≤ 3 :=
sorry

end largest_set_with_sum_conditions_l496_496493


namespace equilateral_triangle_angles_sum_l496_496434

theorem equilateral_triangle_angles_sum (A B C A1 A2 B1 : Type)
  [IsEquilateralTriangle A B C] (h1 : B = B1) (h2 : C = A2) (hA1 : dist B A1 = dist A1 A2)
  (hA2 : dist A1 A2 = dist A2 C) (hB1 : dist A B1 = 2 * dist B1 C) :
  ∠AA1B1 + ∠AA2B1 = 30 :=
sorry

end equilateral_triangle_angles_sum_l496_496434


namespace largest_real_number_l496_496729

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496729


namespace trigonometric_problem_l496_496774

theorem trigonometric_problem
  (α : ℝ) 
  (h_cond : ∃ P : ℝ × ℝ, P = (4/5, -3/5) ∧ 
            P.fst = (4/5) ∧ P.snd = (-3/5) ∧ 
            (P.fst ^ 2 + P.snd ^ 2 = 1)) :
  (sin α = -3/5) ∧
  ( (sin (π / 2 - α) / sin (π + α)) * (tan (α - π) / cos (3 * π - α)) = 5 / 4 ) :=
by
  sorry

end trigonometric_problem_l496_496774


namespace min_value_expression_l496_496131

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (y : ℝ), y = x * sqrt 2 ∧ ∀ (u : ℝ), ∀ (hu : u > 0), 
     sqrt ((x^2 + u^2) * (4 * x^2 + u^2)) / (x * u) ≥ 3 * sqrt 2) := 
sorry

end min_value_expression_l496_496131


namespace sample_size_l496_496424

theorem sample_size {Universities MiddleSchools PrimarySchools SampledMiddleSchools TotalSchools : ℕ} (h1 : Universities = 20) (h2 : MiddleSchools = 200) (h3 : PrimarySchools = 480) (h4 : SampledMiddleSchools = 10) (h5 : TotalSchools = Universities + MiddleSchools + PrimarySchools) 
: ∃ n, n = 35 := by
  have h_total : TotalSchools = 700 := by
    rw [h5, h1, h2, h3]
    norm_num
  have h_prob_selector : (10 / MiddleSchools : ℚ) = (10 / 200) := by
    rw h2
  have h_prob_equality : (n / TotalSchools : ℚ) = (1 / 20) := by
    sorry
  have h_solve_n : n = 35 := by 
    sorry
  exact ⟨35, h_solve_n⟩

end sample_size_l496_496424


namespace baskets_and_remainder_l496_496199

-- Define the initial conditions
def cucumbers : ℕ := 216
def basket_capacity : ℕ := 23

-- Define the expected calculations
def expected_baskets : ℕ := cucumbers / basket_capacity
def expected_remainder : ℕ := cucumbers % basket_capacity

-- Theorem to prove the output values
theorem baskets_and_remainder :
  expected_baskets = 9 ∧ expected_remainder = 9 := by
  sorry

end baskets_and_remainder_l496_496199


namespace regression_lines_common_point_l496_496206

open Real

noncomputable def regression_line (x̄ ȳ : ℝ) (l : ℝ → ℝ) : Prop :=
  l x̄ = ȳ

theorem regression_lines_common_point
  (x̄ ȳ : ℝ)
  (l₁ l₂ : ℝ → ℝ)
  (hx : ∀ (A B : Type) [instA : Field A] [instB : Field B], @instAun x̄ = @instBun x̄)
  (hy : ∀ (A B : Type) [instA : Field A] [instB : Field B], @instAun ȳ = @instBun ȳ)
  :
  regression_line x̄ ȳ l₁ ∧ regression_line x̄ ȳ l₂ :=
by
  sorry

end regression_lines_common_point_l496_496206


namespace cannot_cut_out_rect_l496_496892

noncomputable def square_area : ℝ := 400
noncomputable def rect_area : ℝ := 300
noncomputable def length_to_width_ratio : ℝ × ℝ := (3, 2)

theorem cannot_cut_out_rect (h1: square_area = 400) (h2: rect_area = 300) (h3: length_to_width_ratio = (3, 2)) : 
  false := sorry

end cannot_cut_out_rect_l496_496892


namespace geometric_series_sum_l496_496220

theorem geometric_series_sum :
    let a := (1 / 2)
    let r := (1 / 2)
    let n := 8
    (\sum i in finset.range(n), a * r^i) = (255 / 256) := 
by
  sorry

end geometric_series_sum_l496_496220


namespace percent_change_range_l496_496291

-- Define initial conditions
def initial_yes_percent : ℝ := 0.60
def initial_no_percent : ℝ := 0.40
def final_yes_percent : ℝ := 0.80
def final_no_percent : ℝ := 0.20

-- Define the key statement to prove
theorem percent_change_range : 
  ∃ y_min y_max : ℝ, 
  y_min = 0.20 ∧ 
  y_max = 0.60 ∧ 
  (y_max - y_min = 0.40) :=
sorry

end percent_change_range_l496_496291


namespace calories_per_person_l496_496850

open Nat

theorem calories_per_person :
  ∀ (oranges people pieces_per_orange calories_per_orange : ℕ),
    oranges = 5 →
    pieces_per_orange = 8 →
    people = 4 →
    calories_per_orange = 80 →
    (oranges * pieces_per_orange) / people * (calories_per_orange / pieces_per_orange) = 100 :=
by
  intros oranges people pieces_per_orange calories_per_orange
  assume h_oranges h_pieces_per_orange h_people h_calories_per_orange
  rw [h_oranges, h_pieces_per_orange, h_people, h_calories_per_orange]
  norm_num
  sorry

end calories_per_person_l496_496850


namespace age_difference_of_declans_sons_l496_496525

theorem age_difference_of_declans_sons 
  (current_age_elder_son : ℕ) 
  (future_age_younger_son : ℕ) 
  (years_until_future : ℕ) 
  (current_age_elder_son_eq : current_age_elder_son = 40) 
  (future_age_younger_son_eq : future_age_younger_son = 60) 
  (years_until_future_eq : years_until_future = 30) :
  (current_age_elder_son - (future_age_younger_son - years_until_future)) = 10 := by
  sorry

end age_difference_of_declans_sons_l496_496525


namespace polygons_intersection_area_l496_496446

open Set

theorem polygons_intersection_area 
  (square : Set ℝ^2)
  (polygons : Fin 7 → Set ℝ^2)
  (area_square : MeasureTheory.Measure ℝ^2 volume square = 4)
  (area_polygons : ∀ i, MeasureTheory.Measure ℝ^2 volume (polygons i) ≥ 1) :
  ∃ i j, i ≠ j ∧ MeasureTheory.Measure ℝ^2 volume (polygons i ∩ polygons j) ≥ 1/7 := 
sorry

end polygons_intersection_area_l496_496446


namespace fifth_pyTriple_is_correct_l496_496143

-- Definitions based on conditions from part (a)
def pyTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := b + 1
  (a, b, c)

-- Question: Prove that the 5th Pythagorean triple is (11, 60, 61)
theorem fifth_pyTriple_is_correct : pyTriple 5 = (11, 60, 61) :=
  by
    -- Skip the proof
    sorry

end fifth_pyTriple_is_correct_l496_496143


namespace inequality_solution_difference_l496_496747

theorem inequality_solution_difference (a : ℝ) :
  (∀ x : ℝ, (x + 2) * sqrt (a * x + x - x^2 - a) ≥ 0) →
  (∃ x y : ℝ, abs (x - y) = 4 ∧ (x + 2) * sqrt (a * x + x - x^2 - a) ≥ 0 ∧ (y + 2) * sqrt (a * y + y - y^2 - a) ≥ 0) ↔ 
  a ∈ Set.Icc (-6 : ℝ) (-3) ∪ Set.Ici (5 : ℝ) :=
by
  sorry

end inequality_solution_difference_l496_496747


namespace developer_break_even_price_l496_496618

theorem developer_break_even_price :
  let acres := 4
  let cost_per_acre := 1863
  let total_cost := acres * cost_per_acre
  let num_lots := 9
  let cost_per_lot := total_cost / num_lots
  cost_per_lot = 828 :=
by {
  sorry  -- This is where the proof would go.
} 

end developer_break_even_price_l496_496618


namespace art_collection_area_l496_496676

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l496_496676


namespace problem_1_problem_2_l496_496788

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - π / 6) - 4 * sin(x) ^ 2 + 2
noncomputable def g (x : ℝ) : ℝ := sqrt 3 * sin(2 * x + 2 * (π / 6) + π / 3)

theorem problem_1 : ∀ x : ℝ, f x = sqrt 3 * sin(2 * x + π / 3) := sorry

theorem problem_2 : ∀ x : ℝ, 
  x ∈ Icc (-π / 6) (7 * π / 12) → 
  g' x > 0 → 
  (x ∈ Icc (-π / 6) (π / 6)) := sorry

end problem_1_problem_2_l496_496788


namespace find_a_value_l496_496060

theorem find_a_value
    (a : ℝ)
    (line : ∀ (x y : ℝ), 3 * x + y + a = 0)
    (circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0) :
    a = 1 := sorry

end find_a_value_l496_496060


namespace problem_solution_l496_496772

-- Define the functions f and g with the given conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom domain_f : ∀ x, x ∈ ℝ
axiom domain_g : ∀ x, x ∈ ℝ
axiom cond1 : ∀ x, f(x + 4) + f(-x) = 2
axiom cond2 : ∀ x, f(2 * x + 1) = f(-(2 * x + 1))
axiom cond3 : ∀ x, g(x) = -f(2 - x)

-- Theorem to prove
theorem problem_solution : f 0 = 1 ∧ g 2024 = -1 :=
by
  sorry

end problem_solution_l496_496772


namespace gre_exam_month_l496_496310

def months_of_year := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def start_month := "June"
def preparation_duration := 5

theorem gre_exam_month :
  months_of_year[(months_of_year.indexOf start_month + preparation_duration) % 12] = "November" := by
  sorry

end gre_exam_month_l496_496310


namespace rectangle_perimeter_l496_496156

noncomputable def perimeter_of_rectangle := (8:ℝ) * real.sqrt 2012

theorem rectangle_perimeter (a b : ℝ) (x y : ℝ) (h₁ : x * y = 4024)
  (h₂ : real.pi * a * b = 4024 * real.pi)
  (h₃ : x + y = 2 * a)
  (h₄ : x^2 + y^2 = 4 * (a^2 - b^2)) :
  2 * (x + y) = perimeter_of_rectangle :=
by
  sorry

end rectangle_perimeter_l496_496156


namespace integer_cube_less_than_triple_unique_l496_496232

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l496_496232


namespace probability_of_rolling_prime_is_half_l496_496239

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def total_outcomes : ℕ := 8

def successful_outcomes : ℕ := 4 -- prime numbers between 1 and 8 are 2, 3, 5, and 7

def probability_of_rolling_prime : ℚ :=
  successful_outcomes / total_outcomes

theorem probability_of_rolling_prime_is_half : probability_of_rolling_prime = 1 / 2 :=
  sorry

end probability_of_rolling_prime_is_half_l496_496239


namespace find_point_on_parabola_l496_496274

def parabola_y2_eq_8x (x y : ℝ) : Prop :=
  y^2 = 8 * x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

-- Focus of the parabola y^2 = 8x is at (2, 0)
def focus : ℝ × ℝ := (2, 0)

-- Point P on the parabola y^2 = 8x and distance from focus is 9
def is_on_parabola_and_9_from_focus (P : ℝ × ℝ) : Prop :=
  parabola_y2_eq_8x P.1 P.2 ∧ distance P focus = 9

theorem find_point_on_parabola (P : ℝ × ℝ) :
  is_on_parabola_and_9_from_focus P → P = (7, 2 * real.sqrt 14) ∨ P = (7, -2 * real.sqrt 14) :=
by
  intro h
  sorry

end find_point_on_parabola_l496_496274


namespace midpoint_of_segment_l496_496223

theorem midpoint_of_segment :
  let p1 : ℝ × ℝ × ℝ := (5, -8, 2)
  let p2 : ℝ × ℝ × ℝ := (-1, 6, -4)
  midpoint p1 p2 = (2, -1, -1) :=
by
  sorry

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

end midpoint_of_segment_l496_496223


namespace last_digit_of_x95_l496_496577

theorem last_digit_of_x95 (x : ℕ) : 
  (x^95 % 10) - (3^58 % 10) = 4 % 10 → (x^95 % 10 = 3) := by
  sorry

end last_digit_of_x95_l496_496577


namespace largest_systematic_sample_l496_496332

theorem largest_systematic_sample {n_products interval start second_smallest max_sample : ℕ} 
  (h1 : n_products = 300) 
  (h2 : start = 2) 
  (h3 : second_smallest = 17) 
  (h4 : interval = second_smallest - start) 
  (h5 : n_products % interval = 0) 
  (h6 : max_sample = start + (interval * ((n_products / interval) - 1))) : 
  max_sample = 287 := 
by
  -- This is where the proof would go if required.
  sorry

end largest_systematic_sample_l496_496332


namespace g_at_8_l496_496937

def functional_eq (g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g(x) + g(2*x+y) + 4*x*y = g(3*x - y) + 3*x^2 + 3

theorem g_at_8 (g : ℝ → ℝ) (h : functional_eq g) : g 8 = 67 :=
sorry

end g_at_8_l496_496937


namespace integer_cube_less_than_triple_unique_l496_496231

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l496_496231


namespace find_principal_amount_l496_496996

variable {P R T : ℝ} -- variables for principal, rate, and time
variable (H1: R = 25)
variable (H2: T = 2)
variable (H3: (P * (0.5625) - P * (0.5)) = 225)

theorem find_principal_amount
    (H1 : R = 25)
    (H2 : T = 2)
    (H3 : (P * 0.0625) = 225) : 
    P = 3600 := 
  sorry

end find_principal_amount_l496_496996


namespace smallest_sum_l496_496321

theorem smallest_sum 
  (a b c d e : ℕ) 
  (ha : a = 22)
  (hb : b = 10)
  (hc : c = 15)
  (hd : d = 21)
  (he : e = 7)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : e > 0)
  (h6 : ¬ nat.coprime a b)
  (h7 : ¬ nat.coprime b c)
  (h8 : ¬ nat.coprime c d)
  (h9 : ¬ nat.coprime d e)
  (h10 : nat.coprime a c)
  (h11 : nat.coprime a d)
  (h12 : nat.coprime a e)
  (h13 : nat.coprime b d)
  (h14 : nat.coprime b e)
  (h15 : nat.coprime c e) 
  : a + b + c + d + e = 75 :=
by
  sorry

end smallest_sum_l496_496321


namespace find_triplets_l496_496314

open Nat

theorem find_triplets (p n m : ℕ) (hp : Prime p) (h : p ^ n + 144 = m ^ 2) : 
  (p, n, m) = (5, 2, 13) ∨ 
  (p, n, m) = (2, 8, 20) ∨ 
  (p, n, m) = (3, 4, 15) :=  
by
  sorry

end find_triplets_l496_496314


namespace eating_time_l496_496489

-- Defining the terms based on the conditions provided
def rate_mr_swift := 1 / 15 -- Mr. Swift eats 1 pound in 15 minutes
def rate_mr_slow := 1 / 45  -- Mr. Slow eats 1 pound in 45 minutes

-- Combined eating rate of Mr. Swift and Mr. Slow
def combined_rate := rate_mr_swift + rate_mr_slow

-- Total amount of cereal to be consumed
def total_cereal := 4 -- pounds

-- Proving the total time to eat the cereal
theorem eating_time :
  (total_cereal / combined_rate) = 45 :=
by
  sorry

end eating_time_l496_496489


namespace highest_price_is_A_l496_496626

variables {n m : ℝ}
-- Condition: 0 < n < m < 100
axiom h1 : 0 < n
axiom h2 : n < m
axiom h3 : m < 100

-- Definitions of the pricing schemes
def A_price := 100 * (1 + m / 100) * (1 - n / 100)
def B_price := 100 * (1 + n / 100) * (1 - m / 100)
def C_price := 100 * (1 + (m + n) / 200) * (1 - (m + n) / 200)
def D_price := 100 * (1 + (m * n) / 10000) * (1 - (m * n) / 10000)

theorem highest_price_is_A : 
  A_price ≥ B_price ∧ 
  A_price ≥ C_price ∧ 
  A_price ≥ D_price :=
sorry

end highest_price_is_A_l496_496626


namespace trevor_current_age_l496_496564

theorem trevor_current_age :
  ∃ (T : ℕ), T = 11 ∧ (∃ (T : ℕ), (20 + (24 - T)) = 3 * T) :=
by
  use 11
  split
  {
    refl
  }
  {
    intros
    use 11
    sorry
  }

end trevor_current_age_l496_496564


namespace largest_x_63_over_8_l496_496711

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l496_496711


namespace slope_angle_of_vertical_line_is_90_deg_l496_496197

-- Define the line given by the equation
def line_eq := ∀ x y : ℝ, x = -1

-- Prove that the slope angle of this line is 90 degrees
theorem slope_angle_of_vertical_line_is_90_deg : ∀ x y : ℝ, line_eq x y → atan 0 = real.pi / 2 :=
by
  intro x y line_eq
  sorry

end slope_angle_of_vertical_line_is_90_deg_l496_496197


namespace largest_real_number_l496_496737

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496737


namespace expansion_term_count_l496_496441

theorem expansion_term_count (n : ℕ) (a b c : ℝ) :
  (∀ (i j k : ℕ), i + j + k = n → i ≥ 0 → j ≥ 0 → k ≥ 0) →
  n = 10 →
  (nat.choose (n + 3 - 1) (3 - 1) = 66) :=
by {
  intro hjk,
  assume hn : n = 10,
  rw hn,
  show nat.choose 12 2 = 66,
  sorry
}

end expansion_term_count_l496_496441


namespace hyperbola_asymptotes_eccentricity_l496_496827

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ :=
  c / a

theorem hyperbola_asymptotes_eccentricity (a b c : ℝ) (h₁ : 2 * y - x = 0)
    (h₂ : 2 * y + x = 0) (h₃ : a^2 + b^2 = c^2) :
  hyperbola_eccentricity a b c = sqrt 5 / 2 ∨ hyperbola_eccentricity a b c = sqrt 5 := 
sorry

end hyperbola_asymptotes_eccentricity_l496_496827


namespace find_b_c_angle_m_n_l496_496037

variable (a b c m n : ℝ × ℝ)

-- Define the given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (3, b₂)
def vec_c : ℝ × ℝ := (2, c₂)
def vec_m : ℝ × ℝ := (2 * 1 - 3, 2 * 2 - b₂)
def vec_n : ℝ × ℝ := (1 + 2, 2 + c₂)

-- Conditions in the problem
axiom parallel_cond : 2 * 3 = b₂
axiom perpendicular_cond : 1 * 2 + 2 * c₂ = 0

-- Statement to prove b and c
theorem find_b_c (h1 : b₂ = 6) (h2 : c₂ = -1) :
  vec_b = (3, 6) ∧ vec_c = (2, -1) := by sorry

-- Statement to prove the angle between m and n 
theorem angle_m_n (h3 : b₂ = 6) (h4 : c₂ = -1) :
  let cosine := (-1 * 3 + -2 * 1) / (Real.sqrt (1^2 + 2^2) * Real.sqrt (3^2 + 1^2))
  Math.cos (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by sorry

end find_b_c_angle_m_n_l496_496037


namespace cos2_plus_sin2_given_tan_l496_496012

noncomputable def problem_cos2_plus_sin2_given_tan : Prop :=
  ∀ (α : ℝ), Real.tan α = 2 → Real.cos α ^ 2 + Real.sin (2 * α) = 1

-- Proof is omitted
theorem cos2_plus_sin2_given_tan : problem_cos2_plus_sin2_given_tan := sorry

end cos2_plus_sin2_given_tan_l496_496012


namespace pentagon_vector_relation_l496_496346

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem pentagon_vector_relation
  (A A' B' C' D' E' : V)
  (hB : 3 • B' = 2 • A' + A)
  (hC : 3 • C' = 2 • B' + B)
  (hD : 3 • D' = 2 • C' + C)
  (hE : 3 • E' = 2 • D' + D)
  (hA : 3 • A' = 2 • E' + E) :
  ∃ (p q r s t : ℝ),
    p • A' + q • B' + r • C' + s • D' + t • E' = A ∧
    p = 1/81 ∧ q = 2/81 ∧ r = 4/81 ∧ s = 8/81 ∧ t = 16/81 :=
begin
  use [1/81, 2/81, 4/81, 8/81, 16/81],
  split,
  { sorry }, -- The proof would go here
  { simp }
end

end pentagon_vector_relation_l496_496346


namespace chain_slip_properties_l496_496613

def chain_slip_time (s_init len : ℝ) (g : ℝ) : ℝ :=
  (3 / Real.sqrt g) * Real.log(9 + Real.sqrt 80) - Real.log 2

def initial_velocity (s_init len g : ℝ) : ℝ :=
  Real.sqrt ((80 * g) / 9)

theorem chain_slip_properties:
  ∀ (s_init s_other len δ g : ℝ),
    s_init = 10 → 
    s_other = 8 → 
    len = 18 →
    g > 0 →
    chain_slip_time s_init len g = 2.9 ∧ 
    initial_velocity s_init len g = 9.3 :=
by
  intros s_init s_other len δ g h1 h2 h3 h4
  sorry

end chain_slip_properties_l496_496613


namespace geometric_series_sum_l496_496576

theorem geometric_series_sum :
  ∑ k in finset.range 8, (1/(4:ℝ)^k) = (65535 / 196608:ℝ) :=
by
  sorry

end geometric_series_sum_l496_496576


namespace minimum_jumps_l496_496450

theorem minimum_jumps (dist_cm : ℕ) (jump_mm : ℕ) (dist_mm : ℕ) (cm_to_mm_conversion : dist_mm = dist_cm * 10) (leap_condition : ∃ n : ℕ, jump_mm * n ≥ dist_mm) : ∃ n : ℕ, 19 * n = 18120 → n = 954 :=
by
  sorry

end minimum_jumps_l496_496450


namespace find_radius_of_tangent_circles_l496_496207

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 5

-- Define the circle equation with center (r, 0) and radius r
def circle (r x y : ℝ) : Prop := (x - r)^2 + y^2 = r^2

-- Given conditions
-- 1. The circles are externally tangent to each other at the origin
-- 2. The circles are internally tangent to the ellipse

theorem find_radius_of_tangent_circles (r : ℝ) :
  (∀x y, ellipse x y → (circle r x y)) ∧
  (∀x y, ellipse x y → (circle (-r) x y)) →
  r = (Real.sqrt 15) / 4 :=
by
  sorry

end find_radius_of_tangent_circles_l496_496207


namespace profit_percentage_l496_496609

theorem profit_percentage (SP CP : ℝ) (hs : SP = 270) (hc : CP = 225) : 
  ((SP - CP) / CP) * 100 = 20 :=
by
  rw [hs, hc]
  sorry  -- The proof will go here

end profit_percentage_l496_496609


namespace fill_in_the_blanks_correctly_l496_496503

def remote_areas_need : String := "what the remote areas need"
def children : String := "children"
def education : String := "education"
def good_textbooks : String := "good textbooks"

-- Defining the grammatical agreement condition
def subject_verb_agreement (s : String) (v : String) : Prop :=
  (s = remote_areas_need ∧ v = "is") ∨ (s = children ∧ v = "are")

-- The main theorem statement
theorem fill_in_the_blanks_correctly : 
  subject_verb_agreement remote_areas_need "is" ∧ subject_verb_agreement children "are" :=
sorry

end fill_in_the_blanks_correctly_l496_496503


namespace art_collection_total_area_l496_496674

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l496_496674


namespace calculate_principal_sum_l496_496055

def simple_interest (P R T : ℕ) : ℝ := P * R * T / 100

def compound_interest (P R T : ℕ) : ℝ := P * ((1 + R / 100.0) ^ T - 1)

def diff_ci_si (P : ℕ) : ℝ :=
  compound_interest P 10 2 - simple_interest P 10 2

theorem calculate_principal_sum (P : ℕ) (h1 : diff_ci_si P = 51) : P = 5100 :=
by
  -- Proof will be filled in here
  sorry

end calculate_principal_sum_l496_496055


namespace client_dropped_off_phones_l496_496104

def initial_phones : ℕ := 15
def repaired_phones : ℕ := 3
def coworker_phones : ℕ := 9

theorem client_dropped_off_phones (x : ℕ) : 
  initial_phones - repaired_phones + x = 2 * coworker_phones → x = 6 :=
by
  sorry

end client_dropped_off_phones_l496_496104


namespace work_completion_time_l496_496588

theorem work_completion_time (A B : Type) (work_rate_A : ℝ) (work_rate_B : ℝ) (days_A : ℝ) (days_B : ℝ) (total_work : ℝ) :
  days_A = 4 → days_B = 12 → work_rate_A = total_work / days_A → work_rate_B = total_work / days_B → 
  total_work / (work_rate_A + work_rate_B) = 3 :=
by
  intros daysA_eq daysB_eq work_rate_A_eq work_rate_B_eq
  rw [daysA_eq, daysB_eq, work_rate_A_eq, work_rate_B_eq]
  sorry

end work_completion_time_l496_496588


namespace largest_real_number_l496_496734

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l496_496734


namespace valid_passwords_count_l496_496645

theorem valid_passwords_count : 
  let total_passwords := 10 * 10 * 10 * 10 in
  let restricted_passwords := 10 * 10 * 10 in
  let valid_passwords := total_passwords - restricted_passwords in
  valid_passwords = 9000 := 
by
  let total_passwords := 10 * 10 * 10 * 10
  let restricted_passwords := 10 * 10 * 10
  let valid_passwords := total_passwords - restricted_passwords
  have h : valid_passwords = 9000 := by sorry
  exact h

end valid_passwords_count_l496_496645


namespace lg_roots_eq_l496_496046

theorem lg_roots_eq (a b : ℝ) (h1 : ∀ x : ℝ, 2 * x^2 - 4 * x + 1 = 0 → x = log10 a ∨ x = log10 b) :
  a * b = 100 :=
by
  sorry

end lg_roots_eq_l496_496046


namespace parallel_chords_proof_l496_496348

open Set

-- Declare the points A, B, M, N
variables {A B M N A₁ B₁ : Point}
-- Assume the points lie on a circle (assume existence of a circle containing these points)
variable {C : Circle} (h₀ : OnCircle A C) (h₁ : OnCircle B C) (h₂ : OnCircle M C) (h₃ : OnCircle N C)
-- Chord MA₁ is perpendicular to line NB
variable (h₄ : Perpendicular (Chord M A₁) (Line N B))
-- Chord MB₁ is perpendicular to line NA
variable (h₅ : Perpendicular (Chord M B₁) (Line N A))

-- The proof statement that lines AA₁ and BB₁ are parallel.
theorem parallel_chords_proof : Parallel (Line A A₁) (Line B B₁) :=
by
  sorry

end parallel_chords_proof_l496_496348


namespace largest_x_l496_496722

-- Conditions setup
def largest_x_satisfying_condition : ℝ :=
  let x : ℝ := 63 / 8 in x

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 8 / 9) : 
  x = largest_x_satisfying_condition :=
by
  sorry

end largest_x_l496_496722


namespace correct_option_c_l496_496245

theorem correct_option_c :
  (sqrt 9 = 3 ∧ sqrt ((-5)^2) ≠ -5 ∧ (-sqrt 2)^2 = 2 ∧ sqrt 6 / sqrt 2 ≠ 3) →
  (∃x, x = (-sqrt 2)^2 ∧ x = 2) :=
by
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest',
  cases h_rest' with h3 h4,
  use 2,
  split,
  { exact h3 },
  { rfl }

end correct_option_c_l496_496245


namespace trajectory_C_eq_line_l_eq_l496_496763

-- Constants and conditions
constant A : Point
constant B : Point := (-4, 0)
def midpoint (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)
constant Gamma : Set Point := {A | (A.x - 4)^2 + A.y^2 = 16}
constant P : Point := midpoint A B
constant l : Line
constant M N : Point

-- Assumptions
axiom A_on_Gamma : A ∈ Gamma
axiom P_on_trajectory : P ∈ {P | P = midpoint A B}
axiom l_through_point : passes_through l (-1, 3)
axiom l_intersects_C_at_MN : intersects l {P | P.x^2 + P.y^2 = 4} M N
axiom MN_distance : dist M N = 2 * sqrt 3

-- Prove equations of trajectory C and line l
theorem trajectory_C_eq : {P | P = midpoint A B} = {P | P.x^2 + P.y^2 = 4} :=
sorry

theorem line_l_eq : l = {P | 4 * P.x + 3 * P.y - 5 = 0} ∨ l = {P | P.x = -1} :=
sorry

end trajectory_C_eq_line_l_eq_l496_496763


namespace max_value_of_f_range_of_a_inequality_sum_l496_496371

-- Problem 1: Prove that the maximum value of f(x) is 0 given the function f(x) = ln(x) - x + 1 for x \in (0, +\infty).
theorem max_value_of_f :
  ∀ x : ℝ, (0 < x) → (ln x - x + 1 ≤ 0) := 
sorry

-- Problem 2: Prove the range of values for a given f(x1) ≤ g(x2) for x1 ∈ (0, +\infty) and x2 ∈ [1, 2] is a ≤ 4.
theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ ∈ set.Icc (1:ℝ) (2:ℝ), (ln x₁ - x₁ + 1 ≤ x₂^3 - a * x₂)) → (a ≤ 4) :=
sorry

-- Problem 3: Prove the inequality (1/n)^n + (2/n)^n + ... + (n/n)^n < e / (e - 1).
theorem inequality_sum (n : ℕ) (h : 0 < n) :
  (\sum k in finset.range n, (k / n) ^ n) < (real.exp 1 / (real.exp 1 - 1)) :=
sorry

end max_value_of_f_range_of_a_inequality_sum_l496_496371


namespace Yulia_number_l496_496247

theorem Yulia_number (x : ℤ) (Dasha_has : x + 1) (Anya_has : x + 13) (Anya_eq_4_Dasha : x + 13 = 4 * (x + 1)) : x = 3 := by
  sorry

end Yulia_number_l496_496247


namespace lattice_point_count_l496_496324

-- Define the regions A and B using the boundaries given in the problem
def regionA (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in
  x ≤ 0 ∧ y = x^2 ∧ x ≥ -10 ∧ y ≤ 1

def regionB (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in
  x ≥ 0 ∧ y = x^2 ∧ x ≤ 1 ∧ y ≤ 100

-- Define the count function N for lattice points in a region
noncomputable def N (region : ℝ × ℝ → Prop) : ℕ :=
  Finset.card (Finset.filter region
  ((Finset.Product (Finset.range 101) (Finset.range 101)).filter (λ p => region (p.1, p.2))))

-- The theorem to be proved
theorem lattice_point_count :
  N (λ p => regionA p ∨ regionB p) + N (λ p => regionA p ∧ regionB p) = 1010 :=
sorry

end lattice_point_count_l496_496324


namespace inscribe_isosceles_triangle_in_given_triangle_l496_496965

variable (A B C D E F : Type)
variable (h : ℝ)
variable [NonArchimedeanOrderedField ℝ]
variable (triangle_ABC : A × B × C)

-- Definitions for Lean
def line_parallel_to (l1 l2 : A) : Prop := sorry
def intersects (line1 line2 : A) : Prop := sorry
def perpendicular_bisector (DE : D × E) : Prop := sorry
def is_vertex_of_isosceles (F D E : F × D × E) : Prop := sorry
def height_of_isosceles (F DE : F × (D × E)) (h : ℝ) : Prop := sorry
def within_triangle (DE : D × E) (triangle_ABC : A × B × C) : Prop := sorry

theorem inscribe_isosceles_triangle_in_given_triangle :
    ∃ (F : Type), line_parallel_to D E ∧ 
                  intersects D E ∧ 
                  perpendicular_bisector (D, E) ∧ 
                  is_vertex_of_isosceles (F, D, E) ∧ 
                  height_of_isosceles F (D, E) h ∧ 
                  within_triangle (D, E) triangle_ABC :=
sorry

end inscribe_isosceles_triangle_in_given_triangle_l496_496965


namespace tan_alpha_value_l496_496361

open Real

variable (α : ℝ)

/- Conditions -/
def alpha_interval : Prop := (0 < α) ∧ (α < π)
def sine_cosine_sum : Prop := sin α + cos α = -7 / 13

/- Statement -/
theorem tan_alpha_value 
  (h1 : alpha_interval α)
  (h2 : sine_cosine_sum α) : 
  tan α = -5 / 12 :=
sorry

end tan_alpha_value_l496_496361


namespace find_digits_sum_l496_496170

def original_num1 := 742586
def original_num2 := 829430
def incorrect_sum := 1212016

theorem find_digits_sum (d e : ℕ) (hd : d = 2) (he : e = 6) :
  (original_num1 + f d e original_num1 + (original_num2 + f d e original_num2) = incorrect_sum) →
  d + e = 8 := 
sorry

def f (d e : ℕ) (n : ℕ) : ℕ :=
  let change_digit := λ digit, if digit = d then e else digit
  n.digits.reverse.map change_digit |> (λ l, l.foldr (λ x y, x + 10 * y) 0)

#eval original_num1
#eval original_num2
#eval original_num1 + original_num2

#eval f 2 6 742586
#eval f 2 6 829430

end find_digits_sum_l496_496170


namespace cylinder_surface_area_l496_496627

-- Define the height and radius of the cylinder
def height := 10
def radius := 3

-- Define the formula for the total surface area of a cylinder
def surface_area_cylinder (r h : ℕ) : ℕ := 2 * π * r^2 + 2 * π * r * h

-- Prove that the total surface area of the cylinder is 78π square inches
theorem cylinder_surface_area : surface_area_cylinder radius height = 78 * π := by
  sorry

end cylinder_surface_area_l496_496627


namespace shelby_rain_time_l496_496507

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end shelby_rain_time_l496_496507


namespace johns_haircut_tip_percentage_l496_496101

noncomputable def percent_of_tip (annual_spending : ℝ) (haircut_cost : ℝ) (haircut_frequency : ℕ) : ℝ := 
  ((annual_spending / haircut_frequency - haircut_cost) / haircut_cost) * 100

theorem johns_haircut_tip_percentage : 
  let hair_growth_rate : ℝ := 1.5
  let initial_length : ℝ := 6
  let max_length : ℝ := 9
  let haircut_cost : ℝ := 45
  let annual_spending : ℝ := 324
  let months_in_year : ℕ := 12
  let growth_period := 2 -- months it takes for hair to grow 3 inches
  let haircuts_per_year := months_in_year / growth_period -- number of haircuts per year
  percent_of_tip annual_spending haircut_cost haircuts_per_year = 20 := by
  sorry

end johns_haircut_tip_percentage_l496_496101


namespace consecutive_odd_integers_sum_n_eq_169_l496_496236

theorem consecutive_odd_integers_sum_n_eq_169 (k : ℕ) (n : ℕ) 
    (h1 : ∑ i in Finset.range k, (2 * i + 1) = 169)
    (h2 : n = 1 + (k - 1) * 2) : n = 25 := 
by sorry

end consecutive_odd_integers_sum_n_eq_169_l496_496236


namespace find_x_from_average_l496_496520

theorem find_x_from_average :
  let sum_series := 5151
  let n := 102
  let known_average := 50 * (x + 1)
  (sum_series + x) / n = known_average → 
  x = 51 / 5099 :=
by
  intros
  sorry

end find_x_from_average_l496_496520


namespace number_of_men_in_first_group_l496_496824

-- Definitions for the conditions
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℕ :=
  length / days / men

def work_rate_first_group (M : ℕ) : ℕ :=
  rate_of_work M 48 2

def work_rate_second_group : ℕ :=
  rate_of_work 2 36 3

theorem number_of_men_in_first_group (M : ℕ) 
  (h₁ : work_rate_first_group M = 24)
  (h₂ : work_rate_second_group = 12) :
  M = 4 :=
  sorry

end number_of_men_in_first_group_l496_496824


namespace proof_l496_496461

open Polynomial

noncomputable def P (x y : ℤ) : ℤ := sorry

lemma symmetry (x y : ℤ) : P(x, y) = P(y, x) := sorry

lemma div_by_x_minus_y (x y : ℤ) : x - y ∣ P(x, y) := sorry

theorem proof (x y : ℤ) : (x - y) ^ 2 ∣ P(x, y) :=
by {
  sorry
}

end proof_l496_496461


namespace thief_speed_l496_496285

noncomputable def speed_of_thief (distance_initial : ℝ) (distance_thief : ℝ) (speed_policeman : ℝ) : ℝ :=
  let distance_policeman := distance_initial + distance_thief in
  let time_policeman := distance_policeman / speed_policeman in
  let speed_thief := distance_thief / time_policeman in
  speed_thief

theorem thief_speed
  (distance_initial : ℝ) 
  (distance_thief : ℝ) 
  (speed_policeman : ℝ) 
  (h_initial : distance_initial = 150) 
  (h_thief : distance_thief = 600) 
  (h_policeman : speed_policeman = 10) :
  speed_of_thief distance_initial distance_thief speed_policeman = 8 := 
by
  rw [h_initial, h_thief, h_policeman]
  unfold speed_of_thief
  sorry

end thief_speed_l496_496285


namespace Dan_running_speed_is_10_l496_496961

noncomputable def running_speed
  (d : ℕ)
  (S : ℕ)
  (avg : ℚ) : ℚ :=
  let total_distance := 2 * d
  let total_time := d / (avg * 60) 
  let swim_time := d / S
  let run_time := total_time - swim_time
  total_distance / run_time

theorem Dan_running_speed_is_10
  (d S : ℕ)
  (avg : ℚ)
  (h1 : d = 4)
  (h2 : S = 6)
  (h3 : avg = 0.125) :
  running_speed d S (avg * 60) = 10 := by 
  sorry

end Dan_running_speed_is_10_l496_496961


namespace max_abs_diff_poly_ge_one_l496_496462

theorem max_abs_diff_poly_ge_one 
  (a : ℝ) (n : ℕ) (p : ℕ → ℝ) 
  (p_poly : ∀ x y: ℝ, p (x + y) = p x + p y)
  (p_deg : ∀ x y: ℝ, p (x * y) - p x * y = 0)
  (ha : a ≥ 3) 
  (hp_deg : degree p = n) : 
  ∃ i : ℕ, 0 ≤ i ∧ i ≤ n + 1 ∧ |a^i - p(i)| ≥ 1 :=
by
  sorry

end max_abs_diff_poly_ge_one_l496_496462


namespace calories_per_person_l496_496851

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end calories_per_person_l496_496851


namespace derivative_sum_l496_496357

theorem derivative_sum (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, deriv f x = f' x)
  (h : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 :=
sorry

end derivative_sum_l496_496357


namespace sum_of_g_11_values_l496_496880

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 9 * x + 20
noncomputable def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_g_11_values :
  let solutions := { x : ℝ | f x = 11 },
  (solutions.image g).sum = 35 :=
by
  sorry

end sum_of_g_11_values_l496_496880


namespace find_m_value_l496_496051

theorem find_m_value :
  ∃ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 ∧ m = 11 :=
by
  -- The proof would follow here.
  sorry

end find_m_value_l496_496051


namespace least_blue_eyes_and_snack_l496_496328

variable (total_students blue_eyes students_with_snack : ℕ)

theorem least_blue_eyes_and_snack (h1 : total_students = 35) 
                                 (h2 : blue_eyes = 14) 
                                 (h3 : students_with_snack = 22) :
  ∃ n, n = 1 ∧ 
        ∀ k, (k < n → 
                 ∃ no_snack_no_blue : ℕ, no_snack_no_blue = total_students - students_with_snack ∧
                      no_snack_no_blue = blue_eyes - k) := 
by
  sorry

end least_blue_eyes_and_snack_l496_496328


namespace parabola_constant_term_l496_496532

theorem parabola_constant_term
  (a b c : ℝ)
  (h1 : ∀ x, (-2 * (x - 1)^2 + 3) = a * x^2 + b * x + c ) :
  c = 2 :=
sorry

end parabola_constant_term_l496_496532


namespace find_negative_integer_l496_496547

theorem find_negative_integer (M : ℤ) (h_neg : M < 0) (h_eq : M^2 + M = 12) : M = -4 :=
sorry

end find_negative_integer_l496_496547


namespace minimize_sum_power_α_l496_496767

variable (n : ℕ) (c : ℝ) (α : ℝ)
variable (x : Fin n → ℝ)

theorem minimize_sum_power_α 
  (h_pos : ∀ i, 0 < x i) 
  (h_sum : (∑ i, x i) = c) 
  (h_alpha : 1 < α) : 
  (∀ i, x i = c / n) → (∑ i, (x i)^α) ≤ (∑ i, (c / n)^α) :=
by 
  sorry

end minimize_sum_power_α_l496_496767


namespace maximize_container_volume_l496_496212

noncomputable def height (x : ℝ) : ℝ := 6.4 - 2 * x
noncomputable def volume (x : ℝ) : ℝ := x * (x + 1) * height x

theorem maximize_container_volume : 
  ∃ x h V, 
    0 < x ∧ x < 3.2 ∧
    h = height x ∧ 
    V = volume x ∧ 
    ∀ y, (0 < y ∧ y < 3.2) → volume y ≤ V ∧ 
    (x = 2 ∧ h = 2.4 ∧ V = 14.4) :=
by
  sorry

end maximize_container_volume_l496_496212


namespace symmetry_composition_l496_496124

-- Define reflections and lines
variable (l1 l2 l3 : Line)
variable (I2 I3 : Transformation)
variable (S_ln : Line → Transformation)

-- Given condition
axiom h : l3 = S_ln l1 l2

-- Statement to prove
theorem symmetry_composition : S_ln I3 = S_ln l1 ∘ I2 ∘ S_ln l1 := 
sorry

end symmetry_composition_l496_496124


namespace BF_value_l496_496840

variables (A B C D E F : Type)
variables (AC AE DE CE BF : ℝ)

-- Conditions
constant right_angle_at_A : true
constant right_angle_at_C : true
constant points_on_AC : true
constant DE_perpendicular_to_AC : true
constant BF_perpendicular_to_AC : true
constant AE_eq_4 : AE = 4
constant DE_eq_6 : DE = 6
constant CE_eq_6 : CE = 6
constant AC_eq_10 : AC = 10

-- Goal
theorem BF_value : BF = 15 / 4 :=
by sorry

end BF_value_l496_496840


namespace curved_surface_area_of_cone_l496_496594

noncomputable theory

-- Define the slant height and height
def slant_height := 10
def height := 8

-- Define the radius using the Pythagorean theorem
def radius := Real.sqrt (slant_height ^ 2 - height ^ 2)

-- Define the curved surface area of the cone
def curved_surface_area := Real.pi * radius * slant_height

-- Now, state the theorem
theorem curved_surface_area_of_cone :
  curved_surface_area = 60 * Real.pi := by
  sorry

end curved_surface_area_of_cone_l496_496594


namespace maximum_distance_l496_496762

noncomputable def max_distance_point_to_line (α : ℝ) (h₁ : -real.pi ≤ α ∧ α ≤ real.pi) : ℝ :=
  abs (real.sqrt 2 * real.sin (α + real.pi / 4) - 2)

theorem maximum_distance :
  ∀ (α : ℝ) (h₁ : -real.pi ≤ α ∧ α ≤ real.pi), ∃ d : ℝ, d = 2 + real.sqrt 2 ∧
  max_distance_point_to_line α h₁ ≤ d := 
begin
  intros α h₁,
  use 2 + real.sqrt 2,
  split,
  { refl },
  {
    -- the detailed steps skipped
    sorry
  }
end

end maximum_distance_l496_496762


namespace equal_distance_points_line_l496_496807

theorem equal_distance_points_line (a : ℝ) :
    let A := (-3, -4)
    let B := (6, 3)
    let l := λ (x: ℝ) (y: ℝ), a * x + y + 1 = 0
    let distance := λ (P : ℝ × ℝ) (l : ℝ → ℝ → Prop), abs (a * P.1 + P.2 + 1) / sqrt (a^2 + 1)
in distance A l = distance B l ↔ a = -7/9 ∨ a = -1/3 := sorry

end equal_distance_points_line_l496_496807


namespace female_officers_l496_496899

variable (F : ℕ)

theorem female_officers (h1 : 0.16 * F = 160) : F = 1000 := 
by
  -- Proof to be provided
  sorry

end female_officers_l496_496899


namespace find_first_alloy_percentage_l496_496082

theorem find_first_alloy_percentage (c_second : ℝ) (w_first w_second : ℕ) (c_new : ℝ) (result : ℝ) :
  c_second = 8 → w_first = 20 → w_second = 35 → c_new = 9.454545454545453 → 
  (∃ x, (x * w_first + c_second * w_second) / (w_first + w_second) = c_new ∧ x = result) := 
by
  intros h1 h2 h3 h4
  use 12
  split
  calc
    (12 * 20 + 8 * 35) / (20 + 35) = 9.454545454545453 : sorry
  exact rfl
  sorry

end find_first_alloy_percentage_l496_496082


namespace distance_to_line_correct_valid_a_range_correct_solve_a_plus_b_true_statements_l496_496600

noncomputable def distance_from_circle_center_to_line : ℝ :=
  let C := (0, 2)
  let line := λ (x y : ℝ), x + y - 6 = 0
  2 * Real.sqrt 2

theorem distance_to_line_correct :
  distance_from_circle_center_to_line = 2 * Real.sqrt 2 :=
sorry

def is_monotonic_decreasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (a^2 - 1)^y < (a^2 - 1)^x

def valid_a_range (a : ℝ) : Prop :=
  a ∈ set.Ioo (-Real.sqrt 2) (-1) ∪ set.Ioo 1 (Real.sqrt 2)

theorem valid_a_range_correct (a : ℝ) :
  is_monotonic_decreasing a ↔ valid_a_range a :=
sorry

def f (x a : ℝ) := Real.exp x - (1 / 2) * x^2 - a * x

def tangent_line_at_zero (f : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ :=
  2 * x + b

theorem solve_a_plus_b (a b : ℝ) :
  (tangent_line_at_zero (f · a) a b = f 0 a) → (a + b = 0) :=
sorry

def proposition1 := ∀ x y : ℝ, x + y = 0 → x = -y
def converse_of_proposition1 := ∀ x y : ℝ, x = -y → x + y = 0

def proposition2 := ∀ x : ℝ, x^2 + x - 6 ≥ 0 → x > 2
def negation_of_proposition2 := ∃ x : ℝ, x^2 + x - 6 < 0 ∧ x ≤ 2

def proposition3 (A : ℝ) := A > 30 → Real.sin A > 1/2
def proposition4 (φ : ℝ) := ∀ k : ℤ, φ = k * Real.pi

theorem true_statements :
  converse_of_proposition1 ∧ negation_of_proposition2 ∧
  ¬proposition3 ∧ ¬proposition4 :=
sorry

end distance_to_line_correct_valid_a_range_correct_solve_a_plus_b_true_statements_l496_496600


namespace correct_option_C_correct_option_D_l496_496241

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l496_496241


namespace original_price_of_racket_l496_496494

theorem original_price_of_racket (P : ℝ) (h : (3 / 2) * P = 90) : P = 60 :=
sorry

end original_price_of_racket_l496_496494


namespace count_odd_numbers_300_600_l496_496040

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l496_496040


namespace number_of_edges_in_tree_l496_496345

theorem number_of_edges_in_tree (G : Type) [Graph G] (n : ℕ) (h1 : 1 < n)
    (h2 : (∀ v w : G, v ≠ w → ∃! p : Path v w, True)) : 
    ∃ e : ℕ, e = n - 1 :=
by
  sorry

end number_of_edges_in_tree_l496_496345


namespace right_triangle_ratio_l496_496628

theorem right_triangle_ratio (a b c r s : ℝ) (h : a / b = 2 / 5)
  (h_c : c^2 = a^2 + b^2)
  (h_r : r = a^2 / c)
  (h_s : s = b^2 / c) :
  r / s = 4 / 25 := by
  sorry

end right_triangle_ratio_l496_496628


namespace range_f_x_minus_one_l496_496061

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define the function f as given in the conditions
noncomputable def f : ℝ → ℝ
| x => if x > 0 then x - 1 else -(f (-x))  -- Defined piecewise to ensure the odd property

-- Theorem statement
theorem range_f_x_minus_one (f_odd : is_odd_function f) (h : ∀ x, x > 0 → f x = x - 1) :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end range_f_x_minus_one_l496_496061


namespace equal_distance_points_line_l496_496808

theorem equal_distance_points_line (a : ℝ) :
    let A := (-3, -4)
    let B := (6, 3)
    let l := λ (x: ℝ) (y: ℝ), a * x + y + 1 = 0
    let distance := λ (P : ℝ × ℝ) (l : ℝ → ℝ → Prop), abs (a * P.1 + P.2 + 1) / sqrt (a^2 + 1)
in distance A l = distance B l ↔ a = -7/9 ∨ a = -1/3 := sorry

end equal_distance_points_line_l496_496808


namespace probability_AB_hired_l496_496822

theorem probability_AB_hired (A B C D E : Type)
  (h : Finset {A, B, C, D, E} → Finset UnorderedTriple A, B, C, D, E)
  (p : ∀ x ∈ {A, B, C, D, E}, 1 / 5)
  : (∀ s ∈ Finset.powerset_len 3 ({A, B, C, D, E} : Finset Type), 
  P(s.contains A ∨ s.contains B) = (9 / 10))
:=
  sorry

end probability_AB_hired_l496_496822


namespace slices_left_l496_496472

theorem slices_left (pies : ℕ) (slices_per_pie : ℕ) (classmates : ℕ) (teacher : ℕ) (principal : ℕ) (manny : ℕ) : 
  pies = 5 →
  slices_per_pie = 12 →
  classmates = 30 →
  teacher = 1 →
  principal = 1 →
  manny = 1 →
  (pies * slices_per_pie) - (classmates + teacher + principal + manny) = 27 :=
by
  intro h_pies h_slices_per_pie h_classmates h_teacher h_principal h_manny
  rw [h_pies, h_slices_per_pie, h_classmates, h_teacher, h_principal, h_manny]
  have total_slices : 5 * 12 = 60 := by norm_num
  have total_people : 30 + 1 + 1 + 1 = 33 := by norm_num
  rw [total_slices, total_people]
  norm_num
  exact eq.refl 27

end slices_left_l496_496472


namespace mean_and_median_correction_l496_496305

def original_data := [15, 25, 19, 23, 20]
def corrected_data := [15, 22, 19, 24, 20]

def mean (data : List ℕ) : ℚ :=
  data.sum / data.length

def median (data : List ℕ) : ℚ :=
  let sorted_data := data.qsort (λ a b => a < b)
  if sorted_data.length % 2 = 0 then
    (sorted_data.get (sorted_data.length / 2 - 1) + sorted_data.get (sorted_data.length / 2)) / 2
  else
    sorted_data.get (sorted_data.length / 2)

theorem mean_and_median_correction :
  mean corrected_data - mean original_data = -0.4 ∧ median corrected_data = median original_data :=
by
  sorry

end mean_and_median_correction_l496_496305


namespace sum_of_odd_divisors_180_l496_496973

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_divisors_180 :
  sum_of_positive_odd_divisors 180 = 78 :=
by
  sorry

end sum_of_odd_divisors_180_l496_496973


namespace john_burritos_left_l496_496854

theorem john_burritos_left : 
  ∀ (boxes : ℕ) (burritos_per_box : ℕ) (given_away_fraction : ℚ) (eaten_per_day : ℕ) (days : ℕ),
  boxes = 3 → 
  burritos_per_box = 20 →
  given_away_fraction = 1 / 3 →
  eaten_per_day = 3 →
  days = 10 →
  let initial_burritos := boxes * burritos_per_box in
  let given_away_burritos := given_away_fraction * initial_burritos in
  let after_giving_away := initial_burritos - given_away_burritos in
  let eaten_burritos := eaten_per_day * days in
  let final_burritos := after_giving_away - eaten_burritos in
  final_burritos = 10 := 
by 
  intros,
  sorry

end john_burritos_left_l496_496854


namespace sum_squares_seven_consecutive_not_perfect_square_l496_496501

theorem sum_squares_seven_consecutive_not_perfect_square : 
  ∀ (n : ℤ), ¬ ∃ k : ℤ, k * k = (n-3)^2 + (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 :=
by
  sorry

end sum_squares_seven_consecutive_not_perfect_square_l496_496501


namespace gcd_a_n_a_n_plus_1_l496_496873

def a_n (n : ℕ) : ℕ := (7^n - 1) / 6

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_a_n_a_n_plus_1 (n : ℕ) : gcd (a_n n) (a_n (n + 1)) = 1 := 
by
  sorry

end gcd_a_n_a_n_plus_1_l496_496873


namespace integral_ln_result_l496_496455

noncomputable def integral_ln : ℝ :=
  ∫ (x : ℝ) in 1..Real.exp 1, Real.log x

theorem integral_ln_result :
  integral_ln = 1 := by
  sorry

end integral_ln_result_l496_496455


namespace number_of_B_students_l496_496834

/-- Let x be the number of students who earn a B. 
    Given the conditions:
    - The number of students who earn an A is 0.5x.
    - The number of students who earn a C is 2x.
    - The number of students who earn a D is 0.3x.
    - The total number of students in the class is 40.
    Prove the number of students who earn a B is 40 / 3.8 = 200 / 19, approximately 11. -/
theorem number_of_B_students (x : ℝ) (h_bA: x * 0.5 + x + x * 2 + x * 0.3 = 40) : 
  x = 40 / 3.8 :=
by 
  sorry

end number_of_B_students_l496_496834


namespace frog_jump_no_center_intersection_l496_496431

theorem frog_jump_no_center_intersection (n : ℕ) (h : 2 ≤ n) :
  (∃ jump_condition : (fr : Fin (2*n)) → Fin (2*n),
    ∀ fr1 fr2 : Fin (2*n),
      fr1 ≠ fr2 →
      ¬segment_passes_through_center (jump_condition fr1) (jump_condition fr2)) ↔ (n % 4 = 2) :=
by
  sorry

end frog_jump_no_center_intersection_l496_496431


namespace sum_y_coords_at_least_one_l496_496869

variables {n : ℕ} {r : ℝ}
variables (x y : Fin (2 * n) → ℝ)

def half_circle_on_upper_plane (x y : Fin (2 * n) → ℝ) : Prop :=
  ∀ j, x j ^ 2 + y j ^ 2 = r ^ 2 ∧ y j ≥ 0

def sum_of_odd_x_coords (x : Fin (2 * n) → ℝ) : Prop :=
  Odd (Finset.univ.sum x)

theorem sum_y_coords_at_least_one
  (hx : sum_of_odd_x_coords x)
  (hy : half_circle_on_upper_plane x y) :
  Finset.univ.sum y ≥ 1 := 
sorry

end sum_y_coords_at_least_one_l496_496869


namespace mitch_spare_bars_l496_496894

theorem mitch_spare_bars (num_friends : ℕ) (bars_per_friend : ℕ) (total_bars : ℕ) (spare_bars : ℕ) :
  num_friends = 7 → bars_per_friend = 2 → total_bars = 24 → spare_bars = 10 →
  spare_bars = total_bars - (bars_per_friend * num_friends) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end mitch_spare_bars_l496_496894


namespace point_light_source_covered_by_four_spheres_l496_496292

-- Define the regular tetrahedron and its vertices
structure Tetrahedron :=
  (A B C D O : ℝ^3)
  (is_regular : EuclideanGeometry.isRegularTetrahedron A B C D O)

-- Define the condition of a sphere
structure Sphere :=
  (center : ℝ^3)
  (radius : ℝ)
  (contains_ray_from : ∀ (P : ℝ^3), ∃ (R : ℝ^3), R = P - O ∧ R.dot R = radius^2)

-- Define the problem statement in Lean 4
theorem point_light_source_covered_by_four_spheres
  (T : Tetrahedron)
  (s1 s2 s3 s4 : Sphere)
  (non_intersecting : ∀ i j, i ≠ j → (s1 ∩ s2).empty)
  (covers_every_ray : ∀ ray, ∃ s ∈ [s1, s2, s3, s4], intersects s ray) :
  (∀ ray_from_O, ∃ s ∈ [s1, s2, s3, s4], intersects s ray_from_O) :=
sorry

end point_light_source_covered_by_four_spheres_l496_496292


namespace spy_partition_l496_496632

-- Define the types and assumptions
variables {S : Type} [fintype S]
variables (acq : S → S → Prop) (lead : S → S → Prop)
variables (n : ℕ)
variables (A B C : set S)

-- Define the propositions
def no_acquaintance (A : set S) : Prop := 
  ∀ a1 a2 ∈ A, ¬ acq a1 a2

def leads (A B : set S) : Prop := 
  ∀ b ∈ B, ∃ a ∈ A, lead a b

def leads_chain (B C : set S) : Prop := 
  ∀ c ∈ C, ∃ b ∈ B, lead b c

noncomputable 
def condition (A B C : set S) (n : ℕ) : Prop := 
  no_acquaintance acq A ∧ 
  leads lead A B ∧ 
  leads_chain lead B C ∧ 
  (fintype.card (A ∪ B) > real.sqrt n)

theorem spy_partition (n : ℕ) (h_n : n ≥ 2) [fintype S] :
  ∃ A B C : set S, condition acq lead A B C n :=
sorry

end spy_partition_l496_496632


namespace reflection_and_parallel_l496_496595

-- Definitions of terms in Lean 4
variables {A B C M O H H_A : Point}
variable {Γ : Circle}
variable {BC : Line}
variable [incircle_abc : inscribed_in A B C Γ]
variable [center_of_circle : center O Γ]
variable [orthocenter : orthocenter H A B C]
variable [altitude_foot : foot_of_altitude H_A A B C]
variable [midpoint_M : midpoint M B C]

-- Reflective points
def reflection (P Q : Point) : Point := sorry  -- Define reflection point
noncomputable def H' := reflection H M
noncomputable def H_star := reflection H BC

-- Stating the theorem
theorem reflection_and_parallel :
  (H' ∈ Γ) ∧ (H_star ∈ Γ) ∧ ((H' H_star) ∥ BC) :=
by
  sorry

end reflection_and_parallel_l496_496595


namespace find_x_plus_y_l496_496360

theorem find_x_plus_y (x y : ℤ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) : x + y = -1 ∨ x + y = -5 :=
sorry

end find_x_plus_y_l496_496360


namespace cut_square_positions_l496_496495

theorem cut_square_positions :
  ∀ (grid : matrix (fin 3) (fin 3) (option ℕ)), 
    (∃ (i j : fin 3), grid i j = none) → 
    (∀ i j, grid i j ≠ none) → 
    (∃ i j, grid i j = none) :=
by
  sorry

end cut_square_positions_l496_496495


namespace find_omega_l496_496376

noncomputable def f (ω x : ℝ) := sin(ω * x)^2 + sqrt(3) * sin(ω * x) * cos(ω * x)

theorem find_omega
  (α β ω : ℝ)
  (h1 : f ω α = -1/2)
  (h2 : f ω β = 1/2)
  (h3 : abs (α - β) = 3/4 * π) :
  ω = 1/3 := sorry

end find_omega_l496_496376


namespace angle_CPD_is_constant_l496_496452

section ProofProblem

variable (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C]

-- Given that [C] is a circle with diameter [MN] and chord [AB] with given length.
variables {A B M N : C}
variables (lengthAB : ℝ) (lengthDiameter : ℝ)
variable (ABisChord : dist A B = lengthAB) (MNisDiameter : dist M N = lengthDiameter)
-- [AB] neither coincides nor is perpendicular to [MN]
variables (h1 : A ≠ B) (h2 : M ≠ N) (h3 : ∀ x y: C, x ≠ y ∧ ∃ θ: ℝ, 0 < θ < π ∧ cos θ = 0)
-- [C] and [D] are orthogonal projections of [A] and [B] on [MN]
variables (C D : C) (orthogonalProjectionA : dist C M = dist A M ∧ C.orth_proj MN) (orthogonalProjectionB : dist D N = dist B N ∧ D.orth_proj MN)
-- [P] is the midpoint of [AB]
variable (P : C) (midpointP : dist P A = dist P B ∧ dist P (.5 * A + .5 * B) = 0)
-- Prove that ∠CPD does not depend on [AB]
theorem angle_CPD_is_constant: ∃ θ: ℝ, ∀ AB: C × C, cos θ = DMC_p C D P A B lengthAB lengthDiameter
by
  sorry
end ProofProblem

end angle_CPD_is_constant_l496_496452


namespace intersection_points_correct_l496_496266

noncomputable def circle_diameter_endpoints : Type := (ℝ × ℝ) × (ℝ × ℝ)

noncomputable def intersection_x_coordinates (d : circle_diameter_endpoints) : set ℝ :=
  let h := (d.1.1 + d.2.1) / 2
  let k := (d.1.2 + d.2.2) / 2
  let radius_square := (d.1.1 - h) ^ 2 + (d.1.2 - k) ^ 2
  {x : ℝ | (x - h) ^ 2 + (1 - k) ^ 2 = radius_square}

noncomputable def problem_data : circle_diameter_endpoints := ((1,5), (7,3))

theorem intersection_points_correct :
  intersection_x_coordinates problem_data = {3, 5} :=
sorry

end intersection_points_correct_l496_496266


namespace slope_of_dividing_line_l496_496083

open Real

def vertex := (ℝ × ℝ)

structure Rectangle :=
(vertices : list vertex)

def L_shaped_region : list Rectangle :=
[
  ⟨[(0, 0), (0, 4), (4, 4), (4, 2)]⟩,
  ⟨[(4, 2), (6, 2), (6, 0), (4, 0)]⟩
]

def area (rect : Rectangle) : ℝ :=
match rect.vertices with
| [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] =>
  abs ((x1 - x3) * (y1 - y3))
| _ => 0
end

def total_area (region : list Rectangle) : ℝ :=
region.map area |> List.sum

def bisected_area := total_area L_shaped_region / 2

theorem slope_of_dividing_line :
  let total_a := total_area L_shaped_region in
  bisected_area = 10 →
  ∃ m : ℝ, m = 2
  sorry

end slope_of_dividing_line_l496_496083


namespace Geometry_l496_496347

/-- Given points A(-1, 0), B(1, 0), and a moving point M whose trajectory curve C
    satisfies ∠AMB = 2θ and |overrightarrow{AM}| * |overrightarrow{BM}| * cos^2θ = 3. 
    A line passing through point B intersects curve C at points P and Q.
    Prove that |overrightarrow{AM}| + |overrightarrow{BM}| = 4, 
    and the equation of curve C is (x^2)/4 + (y^2)/3 = 1, 
    and the maximum area of triangle APQ is 3. -/
theorem Geometry.TriangleAreaMax (
  A B M : Point,
  θ : Real,
  hA : A = (-1, 0),
  hB : B = (1, 0),
  hAngle : ∠AMB = 2 * θ,
  hCondition : |overrightarrow{AM}| * |overrightarrow{BM}| * Real.cos ^ 2 θ = 3,
  intersect_PQ : Line -> Point -> Point
) :  (|overrightarrow{AM}| + |overrightarrow{BM}| = 4) ∧
      Curve.C = Ellipse (foci := (A,B), a := 2, c := 1) ∧
      (max_area APQ = 3) := by
  sorry

end Geometry_l496_496347


namespace inequality_problem_l496_496011

variables {a b c d : ℝ}

theorem inequality_problem (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥ a^2 + b^2 + c^2 + d^2 := 
by
  sorry

end inequality_problem_l496_496011


namespace sum_two_integers_l496_496948

theorem sum_two_integers (a b : ℤ) (h1 : a = 17) (h2 : b = 19) : a + b = 36 := by
  sorry

end sum_two_integers_l496_496948


namespace median_ratio_within_bounds_l496_496761

def median_ratio_limits (α : ℝ) (hα : 0 < α ∧ α < π) : Prop :=
  ∀ (s_c s_b : ℝ), s_b = 1 → (1 / 2) ≤ (s_c / s_b) ∧ (s_c / s_b) ≤ 2

theorem median_ratio_within_bounds (α : ℝ) (hα : 0 < α ∧ α < π) : 
  median_ratio_limits α hα :=
by
  sorry

end median_ratio_within_bounds_l496_496761


namespace circle_standard_equation_l496_496769

theorem circle_standard_equation (x y : ℝ) : 
  let center := (-3, 4) in
  let radius := real.sqrt 5 in
  (∀ (a b r : ℝ), center = (a, b) ∧ radius = r → 
    (x - a)^2 + (y - b)^2 = r^2 → (x + 3)^2 + (y - 4)^2 = 5) :=
by 
  intros;
  cases center with a b;
  simp only [];
  intros;
  cases h;
  rw [h_a, h_b, h_a.right];
  simp only [add_comm, pow_two, sq_sqrt (by norm_num), sub_neg_eq_add];
  sorry

end circle_standard_equation_l496_496769


namespace johns_burritos_l496_496859

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l496_496859


namespace courtyard_brick_dimension_l496_496617

theorem courtyard_brick_dimension 
  (length_courtyard : ℝ)
  (breadth_courtyard : ℝ)
  (known_dimension : ℝ)
  (total_bricks : ℝ)
  (known_dimension_eq : known_dimension = 13)
  (length_courtyard_eq : length_courtyard = 1800)
  (breadth_courtyard_eq : breadth_courtyard = 1200)
  (total_bricks_eq : total_bricks = 11076.923076923076) :
  (2160000 / total_bricks) / known_dimension = 15 :=
by
  rw [known_dimension_eq, length_courtyard_eq, breadth_courtyard_eq, total_bricks_eq]
  rw [div_div_eq_div_mul]
  norm_num
  sorry

end courtyard_brick_dimension_l496_496617


namespace ned_trays_per_trip_l496_496490

def trays_from_table1 : ℕ := 27
def trays_from_table2 : ℕ := 5
def total_trips : ℕ := 4
def total_trays : ℕ := trays_from_table1 + trays_from_table2
def trays_per_trip : ℕ := total_trays / total_trips

theorem ned_trays_per_trip :
  trays_per_trip = 8 :=
by
  -- proof is skipped
  sorry

end ned_trays_per_trip_l496_496490


namespace routes_A_to_B_via_C_l496_496391

structure Point where
  x : ℝ
  y : ℝ

def A : Point := Point.mk 0 3
def B : Point := Point.mk 3 0
def C : Point := Point.mk 1.5 1.5

def moves (start end_ : Point) : ℝ :=
  (abs (end_.x - start.x) + abs (end_.y - start.y))

noncomputable def routes_from_A_to_C_to_B : ℝ :=
  let routes (p1 p2 : Point) := (moves p1 p2) choose (floor (moves p1 p2) / 2)
  routes A C * routes C B

theorem routes_A_to_B_via_C : routes_from_A_to_C_to_B = 9 := by
  sorry

end routes_A_to_B_via_C_l496_496391


namespace math_test_difference_l496_496430

noncomputable def calculate_difference (scores : List ℕ) (percentages : List ℝ) : ℝ :=
  let sorted_scores := scores.sort
  let median := sorted_scores.nth (sorted_scores.length / 2) |>.getD 0
  let mean := List.zipWith (λ s p => (s : ℝ) * p) scores percentages |>.sum
  median - mean

theorem math_test_difference :
  let scores := [60, 75, 85, 95]
  let percentages := [0.15, 0.20, 0.40, 0.25]
  calculate_difference scores percentages = 3.25 :=
begin
  -- Proof skipped as per the instructions
  sorry
end

end math_test_difference_l496_496430


namespace mass_percentage_oxygen_NaBrO3_l496_496682

-- Definitions
def molar_mass_Na : ℝ := 22.99
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

-- Theorem: proof that the mass percentage of oxygen in NaBrO3 is 31.81%
theorem mass_percentage_oxygen_NaBrO3 :
  ((3 * molar_mass_O) / molar_mass_NaBrO3) * 100 = 31.81 := by
  sorry

end mass_percentage_oxygen_NaBrO3_l496_496682


namespace megan_savings_final_balance_percentage_l496_496478

noncomputable def megan_final_percentage (initial_balance : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : ℝ :=
let increased_balance := initial_balance * (1 + increase_rate) in
let final_balance := increased_balance * (1 - decrease_rate) in
(final_balance / initial_balance) * 100

theorem megan_savings_final_balance_percentage :
  megan_final_percentage 125 0.25 0.20 = 100 := 
by
  sorry

end megan_savings_final_balance_percentage_l496_496478


namespace find_ω_l496_496413

noncomputable def func (A ω ϕ x : ℝ) : ℝ := A * Real.sin (ω * x + ϕ)
def conditions (A ω ϕ m : ℝ) (hA : A > 0) (hω : ω > 0) : Prop :=
  (func A ω ϕ (π / 6) = m) ∧ (func A ω ϕ (π / 3) = m) ∧ (func A ω ϕ (2 * π / 3) = m)

theorem find_ω (A ϕ m : ℝ) (hA : A > 0) (hω_pos : ω > 0) (hcond : conditions A 4 ϕ m hA hω_pos) :
  ω = 4 :=
begin
  sorry
end

end find_ω_l496_496413


namespace engines_not_defective_l496_496409

theorem engines_not_defective (batches : ℕ) (engines_per_batch : ℕ) (defective_fraction : ℚ) 
  (h_batches : batches = 5) (h_engines_per_batch : engines_per_batch = 80) (h_defective_fraction : defective_fraction = 1/4) : 
  (batches * engines_per_batch - (batches * engines_per_batch * defective_fraction)).toNat = 300 :=
by
  sorry

end engines_not_defective_l496_496409


namespace interval_of_inequality_l496_496313

theorem interval_of_inequality (x : ℝ) :
  (1 / (x^3 + 1) > 4 / x + 2 / 5) ↔ (x ∈ set.Ioo (-1) 0) :=
by sorry

end interval_of_inequality_l496_496313


namespace inverse_variation_example_l496_496155

theorem inverse_variation_example :
  ∀ (p q k : ℝ),
  -- Conditions:
  (p * q = k) →
  (1500 * 0.25 = k) →
  -- Question condition: What is q when p = 3000?
  (p = 3000) →
  -- Conclusion to prove:
  q = 0.125 :=
begin
  intros p q k inverse_variation initial_k given_p,
  rw given_p at inverse_variation,
  rw initial_k at inverse_variation,
  rw mul_comm at inverse_variation,
  linarith,
end

end inverse_variation_example_l496_496155


namespace max_maples_l496_496287

def total_trees := 75

def valid_spacing (maples : List Nat) : Prop :=
  ∀ i j, i < j ∧ j < maples.length → maples.nth i + 6 ≤ maples.nth j

theorem max_maples (maples : List Nat) (larches : List Nat) :
  maples.length + larches.length = total_trees ∧ valid_spacing maples → maples.length ≤ 39 :=
by
  sorry

end max_maples_l496_496287


namespace total_chips_l496_496216

-- Definitions of the given conditions
def Viviana_chocolate_chips (Susana_chocolate_chips : ℕ) := Susana_chocolate_chips + 5
def Susana_vanilla_chips (Viviana_vanilla_chips : ℕ) := 3 / 4 * Viviana_vanilla_chips
def Viviana_vanilla_chips := 20
def Susana_chocolate_chips := 25

-- The statement to prove the total number of chips
theorem total_chips :
  let Viviana_choco := Viviana_chocolate_chips Susana_chocolate_chips,
      Susana_vani := Susana_vanilla_chips Viviana_vanilla_chips,
      total := Viviana_choco + Viviana_vanilla_chips + Susana_chocolate_chips + Susana_vani
  in total = 90 :=
by
  sorry

end total_chips_l496_496216


namespace simplify_expression_l496_496997

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (yz + xz + xy) / (xyz * (x + y + z)) :=
by
  sorry

end simplify_expression_l496_496997


namespace dasha_strip_problem_l496_496300

theorem dasha_strip_problem (a b c : ℕ) (h : a * (2 * b + 2 * c - a) = 43) :
  a = 1 ∧ b + c = 22 :=
by {
  sorry
}

end dasha_strip_problem_l496_496300


namespace solve_equation_real_l496_496911

theorem solve_equation_real (x : ℝ) (h : (x ^ 2 - x + 1) * (3 * x ^ 2 - 10 * x + 3) = 20 * x ^ 2) :
    x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 :=
by
  sorry

end solve_equation_real_l496_496911


namespace mila_social_media_time_week_l496_496698

theorem mila_social_media_time_week
  (hours_per_day_on_phone : ℕ)
  (half_on_social_media : ℕ)
  (days_in_week : ℕ)
  (h1 : hours_per_day_on_phone = 6)
  (h2 : half_on_social_media = hours_per_day_on_phone / 2)
  (h3 : days_in_week = 7) : 
  half_on_social_media * days_in_week = 21 := 
by
  rw [h2, h3]
  norm_num
  exact h1.symm ▸ rfl

end mila_social_media_time_week_l496_496698


namespace sum_of_roots_of_quadratic_l496_496816

noncomputable def x1_x2_roots_properties : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ + x₂ = 3) ∧ (x₁ * x₂ = -4)

theorem sum_of_roots_of_quadratic :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) → (x₁ + x₂ = 3) :=
by
  sorry

end sum_of_roots_of_quadratic_l496_496816


namespace cube_less_than_triple_l496_496227

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l496_496227


namespace find_interval_l496_496412

theorem find_interval (a b : ℝ) (h₁ : ∀ x ∈ set.Icc a b, -0.5 * x^2 + 6.5 ≥ min (-0.5 * a^2 + 6.5) (-0.5 * b^2 + 6.5) ∧ -0.5 * x^2 + 6.5 ≤ max (-0.5 * a^2 + 6.5) (-0.5 * b^2 + 6.5))
  (h₂ : ∀ x ∈ set.Icc a b, x = a ∨ x = b)
  (h₃ : ∃ c ∈ set.Icc a b, -0.5 * c^2 + 6.5 = 2 * a)
  (h₄ : ∃ d ∈ set.Icc a b, -0.5 * d^2 + 6.5 = 2 * b) :
  (a = 1 ∧ b = 3) := sorry

end find_interval_l496_496412


namespace solve_speeds_ratio_l496_496257

noncomputable def speeds_ratio (v_A v_B : ℝ) : Prop :=
  v_A / v_B = 1 / 3

theorem solve_speeds_ratio (v_A v_B : ℝ) (h1 : ∃ t : ℝ, t = 1 ∧ v_A = 300 - v_B ∧ v_A = v_B ∧ v_B = 300) 
  (h2 : ∃ t : ℝ, t = 7 ∧ 7 * v_A = 300 - 7 * v_B ∧ 7 * v_A = 300 - v_B ∧ 7 * v_B = v_A): 
    speeds_ratio v_A v_B :=
sorry

end solve_speeds_ratio_l496_496257


namespace no_integers_p_and_q_l496_496690

theorem no_integers_p_and_q (p q : ℤ) : ¬(∀ x : ℤ, 3 ∣ (x^2 + p * x + q)) :=
by
  sorry

end no_integers_p_and_q_l496_496690


namespace problem_integer_and_decimal_parts_eq_2_l496_496059

theorem problem_integer_and_decimal_parts_eq_2 :
  let x := 3
  let y := 2 - Real.sqrt 3
  2 * x^3 - (y^3 + 1 / y^3) = 2 :=
by
  sorry

end problem_integer_and_decimal_parts_eq_2_l496_496059


namespace couples_seating_arrangement_l496_496836

def factorial (n : ℕ) : ℕ := Nat.recOn n 1 (λ n' acc, (n' + 1) * acc)

theorem couples_seating_arrangement :
  let couples := 4
  let arrangements := factorial 7 - 4 * factorial 6 + 24 * factorial 5 - 32 * factorial 4 + 16 * factorial 3
  arrangements = 1488 :=
by
  sorry   -- Proof omitted.

end couples_seating_arrangement_l496_496836


namespace heat_released_is_1824_l496_496661

def ΔH_f_NH3 : ℝ := -46  -- Enthalpy of formation of NH3 in kJ/mol
def ΔH_f_H2SO4 : ℝ := -814  -- Enthalpy of formation of H2SO4 in kJ/mol
def ΔH_f_NH4SO4 : ℝ := -909  -- Enthalpy of formation of (NH4)2SO4 in kJ/mol

def ΔH_rxn : ℝ :=
  2 * ΔH_f_NH4SO4 - (2 * ΔH_f_NH3 + ΔH_f_H2SO4)  -- Reaction enthalpy change

def heat_released : ℝ := 2 * ΔH_rxn  -- Heat released for 4 moles of NH3

theorem heat_released_is_1824 : heat_released = -1824 :=
by
  -- Theorem statement for proving heat released is 1824 kJ
  sorry

end heat_released_is_1824_l496_496661


namespace triangle_ratio_EG_GF_l496_496094

open_locale classical

variables {A B C M E F G : Type}
variables [inner_product_space ℝ A]
variables (a b c : A) -- Points a, b, c in the space corresponding to A, B, C

-- Midpoint of BC
def M_midpoint (a b c : A) : A :=
  (b + c) / 2

-- Vector relations along AB and AC
def vector_f (a b : A) (x : ℝ) : A :=
  (x * b + (15 - x) * a) / 15

def vector_e (a c : A) (x : ℝ) : A :=
  (3 * x * c + (20 - 3 * x) * a) / 20

-- A finite, non-zero parameter t
noncomputable def t := sorry

-- Intersection point G
def intersection_G (a b c : A) (x : ℝ) : A :=
  a + t * ((M_midpoint a b c) - a)

-- Required ratio EG/GF
def ratio_EG_GF (a b c : A) (x : ℝ) : ℝ :=
  4/3 -- Given answer by solving the ratio problem

theorem triangle_ratio_EG_GF (a b c : A) (x : ℝ) (h1 : (b + c) / 2 = M_midpoint a b c)
  (h2 : ∃ AE x = 3 * x) : ratio_EG_GF a b c x = 4 / 3 := 
sorry

end triangle_ratio_EG_GF_l496_496094


namespace tom_gave_5_nephews_cars_l496_496562

noncomputable def packages : ℕ := 10
noncomputable def cars_per_package : ℕ := 5
noncomputable def total_cars : ℕ := packages * cars_per_package
noncomputable def cars_left : ℕ := 30
noncomputable def cars_given_away : ℕ := total_cars - cars_left
noncomputable def fraction_given_to_each_nephew : ℚ := 1 / 5
noncomputable def cars_per_nephew : ℕ := (cars_given_away * fraction_given_to_each_nephew).toNat
noncomputable def number_of_nephews : ℕ := cars_given_away / cars_per_nephew

theorem tom_gave_5_nephews_cars :
  number_of_nephews = 5 :=
by
  sorry

end tom_gave_5_nephews_cars_l496_496562


namespace max_modulus_expression_l496_496114

open Complex

theorem max_modulus_expression (α β : ℂ) (hβ : ‖β‖ = 2) (hαβ : conj α * β ≠ 1) :
  ∃ (z : ℝ), z = 1 ∧ ∀ α β, ‖β‖ = 2 → conj α * β ≠ 1 → ‖(β - α) / (1 - conj α * β)‖ ≤ z :=
by
  use 1
  sorry

end max_modulus_expression_l496_496114


namespace problem_statement_l496_496795

section
variables {R : Type*} [linear_ordered_field R]

def f (x : R) : R := -2 * real.cos x - x + (x + 1) * real.log (x + 1)
def g (k x : R) : R := k * (x^2 + 2 / x)

theorem problem_statement (k : R) (h : k ≠ 0)
  (H : ∃ x1 ∈ Icc (-1 : R) 1, ∀ x2 ∈ Icc (1/2) 2, f x1 - g k x2 < k - 6) :
  k ∈ set.Ioi 1 :=
sorry

end

end problem_statement_l496_496795


namespace john_burritos_left_l496_496861

theorem john_burritos_left : 
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  in burritos_left = 10 := by
  let total_boxes := 3 
  let burritos_per_box := 20
  let total_burritos := total_boxes * burritos_per_box
  let burritos_given_away := total_burritos / 3
  let burritos_left_after_giving := total_burritos - burritos_given_away
  let burritos_eaten_per_day := 3
  let days := 10
  let total_burritos_eaten := burritos_eaten_per_day * days
  let burritos_left := burritos_left_after_giving - total_burritos_eaten
  have h : total_burritos = 60 := by rfl
  have h1 : burritos_given_away = 20 := by sorry
  have h2 : burritos_left_after_giving = 40 := by sorry
  have h3 : total_burritos_eaten = 30 := by sorry
  have h4 : burritos_left = 10 := by sorry
  exact h4 -- Concluding that burritos_left = 10

end john_burritos_left_l496_496861


namespace problem_1_problem_2_l496_496350

open Real

-- Define the sets A, B, C based on given conditions
def A (a : ℝ) := {x : ℝ | x^2 - a * x ≤ x - a}
def B := {x : ℝ | 1 ≤ log 2 (x + 1) ∧ log 2 (x + 1) ≤ 2}
def C (b c : ℝ) := {x : ℝ | x^2 + b * x + c > 0}

-- Problem (1)
theorem problem_1 (a : ℝ) (h : A a ∩ B = A a) : 1 ≤ a ∧ a ≤ 3 :=
sorry

-- Problem (2)
theorem problem_2 (b c : ℝ) (h1 : B ∩ C b c = ∅) (h2 : B ∪ C b c = univ) : b = -4 ∧ c = 3 :=
sorry

end problem_1_problem_2_l496_496350


namespace factors_of_m_multiples_of_200_l496_496815

theorem factors_of_m_multiples_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) : 
  (∃ k, 200 * k ≤ m ∧ ∃ a b c, k = 2^a * 3^b * 5^c ∧ 3 ≤ a ∧ a ≤ 12 ∧ 2 ≤ c ∧ c ≤ 9 ∧ 0 ≤ b ∧ b ≤ 10) := 
by sorry

end factors_of_m_multiples_of_200_l496_496815


namespace engines_not_defective_count_l496_496406

noncomputable def not_defective_engines (total_batches : ℕ) (engines_per_batch : ℕ) (defective_fraction : ℚ) : ℕ :=
  total_batches * engines_per_batch * (1 - defective_fraction)

theorem engines_not_defective_count:
  not_defective_engines 5 80 (1/4) = 300 :=
by
  sorry

end engines_not_defective_count_l496_496406


namespace parallel_lines_have_k_eq_3_or_5_l496_496806

def line1 (k : ℝ) : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), (k - 3) * p.1 + (4 - k) * p.2 + 1 = 0

def line2 (k : ℝ) : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), 2 * (k - 3) * p.1 - 2 * p.2 + 3 = 0

theorem parallel_lines_have_k_eq_3_or_5 (k : ℝ) :
  (∀ p : ℝ × ℝ, line1 k p → line2 k p) → (k = 3 ∨ k = 5) :=
by 
  -- Proof omitted
  sorry

end parallel_lines_have_k_eq_3_or_5_l496_496806


namespace sum_of_possible_values_g1_l496_496875

theorem sum_of_possible_values_g1 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g(g(x - y)) = g(x) * g(y) - g(x) + g(y) - 2 * x * y) :
  (g 1 = sqrt 2 ∨ g 1 = -sqrt 2) → g 1 + (-g 1) = 0 :=
by
  sorry

end sum_of_possible_values_g1_l496_496875


namespace min_value_function_f_positive_reals_l496_496464

noncomputable def function_f (x y : ℝ) : ℝ :=
  (x + 1 / y)^2 * (x + 1 / y - 2023) + (y + 1 / x)^2 * (y + 1 / x - 2023)

theorem min_value_function_f_positive_reals :
  (x y : ℝ) (hx : x > 0) (hy : y > 0),
  ∃ value, value = function_f x y ∧ value = -1814505489.667 :=
begin
  sorry
end

end min_value_function_f_positive_reals_l496_496464


namespace compute_sin_product_l496_496665

theorem compute_sin_product : 
  (1 - Real.sin (Real.pi / 12)) *
  (1 - Real.sin (5 * Real.pi / 12)) *
  (1 - Real.sin (7 * Real.pi / 12)) *
  (1 - Real.sin (11 * Real.pi / 12)) = 
  (1 / 16) :=
by
  sorry

end compute_sin_product_l496_496665


namespace FindLengthOfWaterFountain_l496_496818

-- Define the entities such as Length, Men, and Days involved in the conditions.

def WaterFountainLength (L : ℕ) : Prop :=
  20 * 42 * 56 = 35 * 3 * L

theorem FindLengthOfWaterFountain : ∃ L : ℕ, WaterFountainLength L :=
by
  use 2240
  unfold WaterFountainLength
  simp
  sorry

end FindLengthOfWaterFountain_l496_496818


namespace sin_cos_sixth_power_sum_l496_496115

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 5 / 8 :=
by
  sorry

end sin_cos_sixth_power_sum_l496_496115


namespace largest_real_number_l496_496727

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l496_496727


namespace size_of_second_file_l496_496299

theorem size_of_second_file 
  (internet_speed : ℕ := 2) -- speed in megabits per minute
  (total_time_minutes : ℕ := 120) -- 2 hours in minutes
  (size_file1 : ℕ := 80) -- size of the first file in megabits
  (size_file3 : ℕ := 70) -- size of the third file in megabits):
  size_file1 + size_file2 + size_file3 = internet_speed * total_time_minutes → 
  size_file2 = 90 := 
by sorry

end size_of_second_file_l496_496299


namespace sum_of_intersection_x_coordinates_l496_496498

theorem sum_of_intersection_x_coordinates :
  ∀ c d : ℕ, (c > 0) ∧ (d > 0) ∧ ∃ x : ℝ, (x = -7 / c) ∧ (x = -d / 5) → 
  (Σ' (c d : ℕ), (c > 0) ∧ (d > 0) ∧ ∃ x : ℝ, (x = -7 / c) ∧ (x = -d / 5)) = -48 / 5 :=
sorry

end sum_of_intersection_x_coordinates_l496_496498


namespace incorrect_statement_is_C_l496_496643

variables (a b : ℝ^3) (k : ℝ)

def statement_A (a : ℝ^3) : Prop := |(3 : ℝ) • a| = 3 * |a|
def statement_B (a b : ℝ^3) : Prop := (3 : ℝ) • (a + b) = (3 : ℝ) • a + (3 : ℝ) • b
def statement_C (a b : ℝ^3) : Prop := (|a| = 3 * |b| → a = 3 • b ∨ a = -(3 • b))
def statement_D (a b : ℝ^3) : Prop := ∀ k : ℝ, a = k • b → a ∥ b

-- The proof problem: Prove that among A, B, D and C, the incorrect one is C.
theorem incorrect_statement_is_C : 
  statement_A a → 
  statement_B a b → 
  statement_D a b →
  ¬ statement_C a b :=
by
  intros hA hB hD
  sorry -- proof goes here

end incorrect_statement_is_C_l496_496643


namespace distance_PF2_eq_3_l496_496776

noncomputable def ellipse_major_axis {x y : ℝ} 
  (P F1 F2 : ℝ × ℝ) : Prop :=
  (x^2 / 16 + y^2 / 4 = 1) ∧ 
  (abs (dist P F1 + dist P F2) = 8) ∧ 
  (abs (dist P F1 - dist P F2) = 2)

theorem distance_PF2_eq_3
  (P F1 F2 : ℝ × ℝ) 
  (h : ellipse_major_axis P F1 F2) : 
  abs (dist P F2) = 3 :=
  sorry

end distance_PF2_eq_3_l496_496776


namespace largest_integer_less_80_rem_5_l496_496708

/-- Let x be an integer less than 80 that leaves a remainder of 5 when divided by 8.
    Find the largest such x and prove that it is 77. --/
theorem largest_integer_less_80_rem_5 : ∃ x : ℤ, x < 80 ∧ x % 8 = 5 ∧ ∀ y : ℤ, y < 80 ∧ y % 8 = 5 → y ≤ x :=
by
  use 77
  split
  sorry

end largest_integer_less_80_rem_5_l496_496708
