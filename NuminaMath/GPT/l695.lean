import Mathlib

namespace degrees_manufacturing_section_l695_695921

def percentage_to_degrees (p : ℝ) : ℝ := (360 / 100) * p

theorem degrees_manufacturing_section (p : ℝ) (h : p = 45) : percentage_to_degrees p = 162 := by
  rw [h]
  norm_num
  done

end degrees_manufacturing_section_l695_695921


namespace linear_func_3_5_l695_695115

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_func_3_5 (f : ℝ → ℝ) (h_linear: linear_function f) 
  (h_diff: ∀ d : ℝ, f (d + 1) - f d = 3) : f 3 - f 5 = -6 :=
by
  sorry

end linear_func_3_5_l695_695115


namespace hyperbola_equation_l695_695788

theorem hyperbola_equation (foci_ellipse : set ℝ) (asymptote : ℝ → ℝ) :
  (foci_ellipse = {p | (p.1)^2 = 9}) →
  (asymptote = λ x, (real.sqrt 5 / 2) * x) →
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 5 ∧ ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) :=
begin
  intros h_foci h_asymptote,
  use [2, real.sqrt 5],
  split,
  { norm_num },
  split,
  { norm_num },
  assume x y,
  sorry
end

end hyperbola_equation_l695_695788


namespace smallest_positive_angle_l695_695729

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l695_695729


namespace find_imaginary_part_of_z_l695_695765

noncomputable def z_imaginary_part : Prop :=
  ∃ (b : ℝ), (|complex.mk 2 b| = 2 * real.sqrt 2) ∧ (b = 2 ∨ b = -2)

theorem find_imaginary_part_of_z : z_imaginary_part := 
  sorry

end find_imaginary_part_of_z_l695_695765


namespace bob_needs_change_probability_l695_695947

/-- Bob begins with 10 dimes and one five-dollar bill, trying to buy a toy from the machine.
    The toys prices are $[0.10, 0.35, 0.60, 0.85, 1.10, 1.35, 1.60, 1.85, 2.10, 2.35]$.
    What is the probability that Bob will need to break his five-dollar bill before he can buy
    his favorite toy of price $2.35$?
    The number of possible sequences to dispense toys are $10!$, and the number of favorable
    sequences are $9! + 9 \times 8! + 36 \times 7! + 84 \times 6!$. Therefore, the answer
    is $1 - \frac{9! + 9 \times 8! + 36 \times 7! + 84 \times 6!}{10!}$ which equals $\frac{758}{1009}$. -/
theorem bob_needs_change_probability : 
  let num_toys := 10
  let toy_prices := [0.10, 0.35, 0.60, 0.85, 1.10, 1.35, 1.60, 1.85, 2.10, 2.35]
  let bob_dimes := 10
  let bob_dollars := 5
  let favorite_toy_price := 2.35
  let total_possible_orders := (num_toys.factorial : ℚ)
  let favorable_orders := (9.factorial : ℚ) + 9 * (8.factorial : ℚ) + 36 * (7.factorial : ℚ) + 84 * (6.factorial : ℚ)
  let probability_no_change := favorable_orders / total_possible_orders
  let probability_change := 1 - probability_no_change
  probability_change = (758 / 1009 : ℚ) :=
by sorry

end bob_needs_change_probability_l695_695947


namespace find_acute_angle_l695_695356

theorem find_acute_angle (α : ℝ) (a b : ℝ × ℝ)
  (h_a : a = (3/2, Real.sin α))
  (h_b : b = (Real.cos α, 1/3))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  α = Real.pi / 4 := sorry

end find_acute_angle_l695_695356


namespace triangle_third_side_possible_lengths_l695_695648

theorem triangle_third_side_possible_lengths (a b : ℕ) (h₁ : a = 4) (h₂ : b = 10) (c : ℕ) :
  6 < c ∧ c < 14 ↔ c ∈ {7, 8, 9, 10, 11, 12, 13} :=
by sorry

end triangle_third_side_possible_lengths_l695_695648


namespace probability_non_square_non_cube_l695_695590

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695590


namespace intersection_of_M_and_N_l695_695811

-- Define the sets M and N based on the given conditions
def M : Set ℝ := { x | |x| ≤ 2 }
def N : Set ℕ := { x | 0 < x }

-- Define the intersection of M and N
def M_inter_N : Set ℝ := { x | x ∈ M ∧ x ∈ N.map (Nat.cast) }

-- The proof statement
theorem intersection_of_M_and_N : M_inter_N = {1, 2} := 
by 
sorry

end intersection_of_M_and_N_l695_695811


namespace arithmetic_progression_product_difference_le_one_l695_695221

theorem arithmetic_progression_product_difference_le_one 
  (a b : ℝ) :
  ∃ (m n k l : ℤ), |(a + b * m) * (a + b * n) - (a + b * k) * (a + b * l)| ≤ 1 :=
sorry

end arithmetic_progression_product_difference_le_one_l695_695221


namespace fabulous_integers_l695_695683

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (a^n - a) % n = 0

theorem fabulous_integers (n : ℕ) : is_fabulous n ↔ ¬(∃ k : ℕ, n = 2^k ∧ k ≥ 1) := 
sorry

end fabulous_integers_l695_695683


namespace cistern_fill_time_l695_695183

def pipe_rate (time : ℕ) : ℚ := 1 / time

def combined_rate (rateA rateB rateC : ℚ) : ℚ := rateA + rateB - rateC

theorem cistern_fill_time :
  let rateA := pipe_rate 10 in
  let rateB := pipe_rate 15 in
  let rateC := pipe_rate 12 in
  let combined := combined_rate rateA rateB rateC in
  1 / combined = 12 :=
by
  sorry

end cistern_fill_time_l695_695183


namespace Megan_bought_24_eggs_l695_695066

def eggs_problem : Prop :=
  ∃ (p c b : ℕ),
    b = 3 ∧
    c = 2 * b ∧
    p - c = 9 ∧
    p + c + b = 24

theorem Megan_bought_24_eggs : eggs_problem :=
  sorry

end Megan_bought_24_eggs_l695_695066


namespace area_enclosed_by_curve_l695_695903

theorem area_enclosed_by_curve (a : ℝ) (h : a > 0) : 
  let curve := λ x y => (a + x) * y^2 = (a - x) * x^2
  in ∃ S : ℝ, S = (a^2 / 2) * (4 - real.pi) ∧ ∀ x y : ℝ, 0 ≤ x → curve x y → 0 ≤ y ∧ S = 2 * ∫ x in 0..a, sqrt ((a - x) * x^2 / (a + x))  :=
begin
  intro a,
  intro h,
  let curve := λ x y : ℝ, (a + x) * y^2 = (a - x) * x^2,
  use (a^2 / 2) * (4 - real.pi),
  split,
  exact (by norm_num : (a^2 / 2) * (4 - real.pi) = _),
  intros x y hx hcurve,
  sorry -- proof steps are omitted
end

end area_enclosed_by_curve_l695_695903


namespace system_solution_l695_695487

theorem system_solution (a x y z : ℂ) (k l : ℤ) :
  (| a + 1 / a | ≥ 2) ∧ (| a | = 1) ∧ (sin y = 1 ∨ sin y = -1) ∧ (cos z = 0) →
  (x = π / 2 + k * π) ∧ (y = π / 2 + k * π) ∧ (z = π / 2 + l * π) :=
by
  sorry

end system_solution_l695_695487


namespace correct_calculated_value_l695_695155

theorem correct_calculated_value (n : ℕ) (h1 : n = 32 * 3) : n / 4 = 24 := 
by
  -- proof steps will be filled here
  sorry

end correct_calculated_value_l695_695155


namespace june_ride_time_l695_695872

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end june_ride_time_l695_695872


namespace find_x_l695_695629

def cube_volume (s : ℝ) := s^3
def cube_surface_area (s : ℝ) := 6 * s^2

theorem find_x (x : ℝ) (s : ℝ) 
  (hv : cube_volume s = 7 * x)
  (hs : cube_surface_area s = x) : 
  x = 42 := 
by
  sorry

end find_x_l695_695629


namespace num_odd_even_subsets_equal_sum_capacities_odd_even_equal_sum_capacities_odd_subsets_l695_695063

-- Definitions from the conditions
def Sn (n : ℕ) := (finset.range n).image (λ x, x + 1)
def capacity (X : finset ℕ) := X.sum id
def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

-- Part 1
theorem num_odd_even_subsets_equal {n : ℕ} : 
  finset.filter (λ X, is_odd (capacity X)) (finset.powerset (Sn n)).card =
  finset.filter (λ X, is_even (capacity X)) (finset.powerset (Sn n)).card := 
sorry

-- Part 2
theorem sum_capacities_odd_even_equal {n : ℕ} (h : 3 ≤ n) : 
  (finset.filter (λ X, is_odd (capacity X)) (finset.powerset (Sn n))).sum capacity = 
  (finset.filter (λ X, is_even (capacity X)) (finset.powerset (Sn n))).sum capacity := 
sorry

-- Part 3
theorem sum_capacities_odd_subsets {n : ℕ} (h : 3 ≤ n) : 
  (finset.filter (λ X, is_odd (capacity X)) (finset.powerset (Sn n))).sum capacity = 
  2^(n-3) * n * (n+1) := 
sorry

end num_odd_even_subsets_equal_sum_capacities_odd_even_equal_sum_capacities_odd_subsets_l695_695063


namespace sum_first_9_terms_l695_695505

theorem sum_first_9_terms (d : ℚ) (h1 : d ≠ 0) (h2 : (1 + d)^2 = 1 * (1 + 4 * d)) :
  (finset.range 9).sum (λ n, 1 + n * d) = 81 :=
sorry

end sum_first_9_terms_l695_695505


namespace disjoint_translates_l695_695461

open Set

variable (S : Finset ℕ := Finset.range 1000000)
variable (A : Finset ℕ) (hA : A.card = 101)

theorem disjoint_translates :
  ∃ (x : Finset ℕ), x.card = 100 ∧ ∀ (xi xj ∈ x), xi ≠ xj → (A.map ⟨(· + xi : ℕ → ℕ), sorry⟩).disjoint (A.map ⟨(· + xj : ℕ → ℕ), sorry⟩) :=
sorry

end disjoint_translates_l695_695461


namespace phi_value_l695_695332

theorem phi_value (φ : ℝ) (h1 : 0 ≤ φ) (h2 : φ < π) 
  (h3 : ∃ x : ℝ, x = π / 3 ∧ cos x = sin (2 * x + φ)) : φ = π / 6 :=
by
  sorry

end phi_value_l695_695332


namespace min_value_A2_minus_B2_l695_695891

noncomputable def A (x y z : ℝ) : ℝ := ℝ.sqrt (x + 3) + ℝ.sqrt (y + 6) + ℝ.sqrt (z + 11)
noncomputable def B (x y z : ℝ) : ℝ := ℝ.sqrt (x + 2) + ℝ.sqrt (y + 2) + ℝ.sqrt (z + 2)

theorem min_value_A2_minus_B2 (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) : A x y z ^ 2 - B x y z ^ 2 = 36 := by
  sorry

end min_value_A2_minus_B2_l695_695891


namespace longest_side_length_l695_695204

-- Definitions for the coordinates of the vertices.
def A := (3 : ℝ, 3 : ℝ)
def B := (7 : ℝ, -1 : ℝ)
def C := (4 : ℝ, 5 : ℝ)

-- Distance formula between two points.
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Distances between the given points.
def d_AB := dist A B
def d_AC := dist A C
def d_BC := dist B C

-- The goal is to prove that the longest side's length is 3√5.
theorem longest_side_length : max d_AB (max d_AC d_BC) = 3 * Real.sqrt 5 := by
  sorry

end longest_side_length_l695_695204


namespace mobius_trip_proof_l695_695440

noncomputable def mobius_trip_time : ℝ :=
  let speed_no_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let distance_total := 257
  let distance_typical := 120
  let distance_light := distance_total - distance_typical
  let time_typical := distance_typical / speed_typical_load
  let time_light := distance_light / speed_light_load
  let time_return := distance_total / speed_no_load
  let rest_first := (20 + 25 + 35) / 60.0
  let rest_second := (45 + 30) / 60.0
  time_typical + time_light + time_return + rest_first + rest_second

theorem mobius_trip_proof : mobius_trip_time = 44.6783 :=
  by sorry

end mobius_trip_proof_l695_695440


namespace probability_non_square_non_cube_l695_695587

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695587


namespace jacob_current_age_l695_695708

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l695_695708


namespace sum_num_den_cos_gamma_eq_25_l695_695382

-- Define the setup with the given conditions
variables (γ δ : ℝ)
variables (hγδ : γ + δ < Real.pi)
variables (hcosγ : ∃ (p q : ℕ), Rational.mk' p q = Real.cos γ ∧ Nat.gcd p q = 1)

-- Prove that the sum of the numerator and denominator of the positive rational cos γ is 25
theorem sum_num_den_cos_gamma_eq_25 (hchords : ∀ (r : ℝ), ∃ (x y z : ℕ), x = 5 ∧ y = 7 ∧ z = 9 
                                              ∧ x^2 = y^2 + z^2 - 2 * y * z * Real.cos (γ / 2)) :
  ∃ (p q : ℕ), Rational.mk' p q = Real.cos γ ∧ p + q = 25 :=
by 
  sorry

end sum_num_den_cos_gamma_eq_25_l695_695382


namespace sqrt_9_eq_pm3_l695_695935

theorem sqrt_9_eq_pm3 : ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_9_eq_pm3_l695_695935


namespace length_of_plot_57_meters_l695_695517

section RectangleProblem

variable (b : ℝ) -- breadth of the plot
variable (l : ℝ) -- length of the plot
variable (cost_per_meter : ℝ) -- cost per meter
variable (total_cost : ℝ) -- total cost

-- Given conditions
def length_eq_breadth_plus_14 (b l : ℝ) : Prop := l = b + 14
def cost_eq_perimeter_cost_per_meter (cost_per_meter total_cost perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

-- Definition of perimeter
def perimeter (b l : ℝ) : ℝ := 2 * l + 2 * b

-- Problem statement
theorem length_of_plot_57_meters
  (h1 : length_eq_breadth_plus_14 b l)
  (h2 : cost_eq_perimeter_cost_per_meter cost_per_meter total_cost (perimeter b l))
  (h3 : cost_per_meter = 26.50)
  (h4 : total_cost = 5300) :
  l = 57 :=
by
  sorry

end RectangleProblem

end length_of_plot_57_meters_l695_695517


namespace percentage_improvement_is_15_percent_l695_695420

noncomputable def annual_yield_improvement (old_range new_range : ℝ) : ℝ :=
  ((new_range - old_range) / old_range) * 100

theorem percentage_improvement_is_15_percent (old_range : ℝ) (new_range : ℝ) : 
  old_range = 10000 → 
  new_range = 11500 → 
  annual_yield_improvement old_range new_range = 15 := 
by 
  intros 
  rw [annual_yield_improvement, h, h_1]
  norm_num
  sorry

end percentage_improvement_is_15_percent_l695_695420


namespace sum_binomial_1_l695_695277

theorem sum_binomial_1 (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (3^(k + 1) / (k + 1)) * nat.choose n k = 
  (4^(n + 1) - 1) / (n + 1) :=
by
  sorry

end sum_binomial_1_l695_695277


namespace numbers_are_even_or_even_number_of_odds_l695_695936

theorem numbers_are_even_or_even_number_of_odds (a : Fin 24 → ℤ) (h : (∑ i, a i) = 576) : 
  (∀ i, a i % 2 = 0) ∨ (∃ k, k <= 24 ∧ k % 2 = 0 ∧ (∃ S, S.card = k ∧ ∀ i ∈ S, a i % 2 = 1)) := 
by
  sorry

end numbers_are_even_or_even_number_of_odds_l695_695936


namespace min_value_2a_minus_ab_l695_695909

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_lt_11 : a < 11) (hb_lt_11 : b < 11) : 
  ∃ (min_val : ℤ), min_val = -80 ∧ ∀ x y : ℕ, 0 < x → 0 < y → x < 11 → y < 11 → 2 * x - x * y ≥ min_val :=
by
  use -80
  sorry

end min_value_2a_minus_ab_l695_695909


namespace proof_problem_l695_695008

noncomputable def problem : Prop :=
  let total_schools := 21 + 14 + 7 in
  let primary_schools := 21 in
  let middle_schools := 14 in
  let universities := 7 in

  -- Calculation with stratified sampling
  let selected_primary := 3 in
  let selected_middle := 2 in
  let selected_university := 1 in
  
  -- Total selected schools
  let total_selected := selected_primary + selected_middle + selected_university in
  
  -- Ensure total selected schools is 6
  total_selected = 6 ∧

  -- Probability calculation
  let outcomes := (total_selected * (total_selected - 1)) / 2 in        -- Combinations of 2 from 6
  let favorable_outcomes := (selected_primary * (selected_primary - 1)) / 2 in -- Combinations of 2 from 3 primary schools
  let probability := (favorable_outcomes : ℚ) / outcomes in
  probability = 1/5

theorem proof_problem : problem :=
by
  sorry

end proof_problem_l695_695008


namespace real_number_set_condition_l695_695128

theorem real_number_set_condition (x : ℝ) :
  (x ≠ 1) ∧ (x^2 - x ≠ 1) ∧ (x^2 - x ≠ x) →
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 := 
by
  sorry

end real_number_set_condition_l695_695128


namespace trigonometric_values_terminal_side_l695_695797

theorem trigonometric_values_terminal_side (α : ℝ)
  (h1 : ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, x = k → y = 2 * x)) :
  (sin α = (2 * Real.sqrt 5) / 5 ∧ cos α = Real.sqrt 5 / 5 ∧ tan α = 2) ∨
  (sin α = (-2 * Real.sqrt 5) / 5 ∧ cos α = -Real.sqrt 5 / 5 ∧ tan α = 2) :=
by
  sorry

end trigonometric_values_terminal_side_l695_695797


namespace total_cost_is_correct_l695_695418

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l695_695418


namespace max_area_of_garden_l695_695673

theorem max_area_of_garden (L : ℝ) (hL : 0 ≤ L) :
  ∃ x y : ℝ, x + 2 * y = L ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y = L^2 / 8 :=
by
  sorry

end max_area_of_garden_l695_695673


namespace probability_neither_square_nor_cube_l695_695534

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695534


namespace harry_morning_ratio_l695_695716

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio_l695_695716


namespace jacob_age_proof_l695_695710

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l695_695710


namespace find_unknown_fractions_l695_695602

-- conditions, in Lean definitions
def fractions := [1/3, 1/7, 1/9, 1/11, 1/33]
def unknown_fractions := List.filter (λ x : ℚ => x.denom % 10 = 5) [1/5, 1/15, 1/45, 1/385]

-- the statement to prove
theorem find_unknown_fractions : 
    List.sum (fractions ++ unknown_fractions) = 1 :=
by
  -- placeholder for the proof
  sorry

end find_unknown_fractions_l695_695602


namespace expression_equivalence_l695_695269

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by
  sorry

end expression_equivalence_l695_695269


namespace ratio_of_perimeters_of_squares_l695_695618

theorem ratio_of_perimeters_of_squares (A B : ℝ) (h: A / B = 16 / 25) : ∃ (P1 P2 : ℝ), P1 / P2 = 4 / 5 :=
by
  sorry

end ratio_of_perimeters_of_squares_l695_695618


namespace quadratic_expression_decreasing_y_range_range_of_a_l695_695054

-- 1. Given y = ax^2 + bx + 1 with m = 4
variable {a b : ℝ}
variable {m n p : ℝ}
variable {y : ℝ → ℝ}

-- Condition: a ≠ 0
axiom a_nonzero : a ≠ 0

-- Condition: y(x) = ax^2 + bx + 1
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Condition: y(-1) = m
axiom y_neg1_eq_m : quadratic (-1) = 4

-- Condition: y(0) = 1
axiom y_0_eq_1 : quadratic 0 = 1

-- Condition: y(1) = n
axiom y_1_eq_n : quadratic 1 = n

-- Condition: y(2) = 1
axiom y_2_eq_1 : quadratic 2 = 1

-- Condition: y(3) = p
axiom y_3_eq_p : quadratic 3 = p

-- Prove the quadratic function is y = x^2 - 2x + 1
theorem quadratic_expression : ∀ x : ℝ, quadratic x = x^2 - 2 * x + 1 := by
  sorry

-- Prove the range of x where y decreases as x increases
theorem decreasing_y_range : ∀ x : ℝ, x < 1 → ∃ y : ℝ, quadratic x = y ∧ ∀ x' > x, quadratic x' < y := by
  sorry

-- Prove the range of values for a given only one of m, n, p is positive
theorem range_of_a : (∃! (m n p : ℝ), (quadratic (-1) = m ∧ quadratic (1) = n ∧ quadratic (3) = p) ∧ (m > 0 → n ≤ 0 ∧ p ≤ 0) ∧ (n > 0 → m ≤ 0 ∧ p ≤ 0) ∧ (p > 0 → m ≤ 0 ∧ n ≤ 0)) ↔ a ≤ -1/3 := by
  sorry

end quadratic_expression_decreasing_y_range_range_of_a_l695_695054


namespace find_other_factor_l695_695998

theorem find_other_factor (n : ℕ) (hn : n = 75) :
    ( ∃ k, k = 25 ∧ ∃ m, (k * 3^3 * m = 75 * 2^5 * 6^2 * 7^3) ) :=
by
  sorry

end find_other_factor_l695_695998


namespace sum_of_x_values_l695_695437

variable (x y z : ℂ)

-- Conditions
def condition1 := x + y * z = 9
def condition2 := y + x * z = 14
def condition3 := z + x * y = 14

-- Theorem statement
theorem sum_of_x_values : 
  (∑ (x y z : ℂ) in {⟨x, y, z⟩ | condition1 ∧ condition2 ∧ condition3}, x) = 22 :=
sorry

end sum_of_x_values_l695_695437


namespace tournament_draws_l695_695615

open Finset

theorem tournament_draws (n : ℕ) (hn : n = 12) :
  let matches := n.choose 2
  let wins := n
  matches - wins = 54 :=
by
  sorry

end tournament_draws_l695_695615


namespace no_real_solution_complex_solution_l695_695290

open Matrix

noncomputable def determinant (b : ℂ) (y : ℂ) : ℂ := 
det ![
  ![y^2 + b, y, y],
  ![y, y^2 + b, y],
  ![y, y, y^2 + b]
]

theorem no_real_solution (b : ℝ) (hb : b ≠ 0) : ¬∃ y : ℝ, determinant b y = 0 := 
sorry

theorem complex_solution (b : ℂ) (hb : b ≠ 0) : 
  ∃ y : ℂ, (y = (complex.I * (real.sqrt b) / 2) ∨ y = (-complex.I * (real.sqrt b) / 2)) ∧ determinant b y = 0 := 
sorry

end no_real_solution_complex_solution_l695_695290


namespace solve_sum_of_first_9_l695_695776

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ)
hypothesis (h_seq : arithmetic_sequence a)
hypothesis (h_sum : sum_first_n_terms a S)
hypothesis (h_cond : a 3 + a 7 = 6)

-- Proof statement
theorem solve_sum_of_first_9 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (h_cond : a 3 + a 7 = 6) : S 9 = 27 :=
sorry

end solve_sum_of_first_9_l695_695776


namespace comparison_of_products_l695_695966

def A : ℕ := 8888888888888888888 -- 19 digits, all 8's
def B : ℕ := 3333333333333333333333333333333333333333333333333333333333333333 -- 68 digits, all 3's
def C : ℕ := 4444444444444444444 -- 19 digits, all 4's
def D : ℕ := 6666666666666666666666666666666666666666666666666666666666666667 -- 68 digits, first 67 are 6's, last is 7

theorem comparison_of_products : C * D > A * B ∧ C * D - A * B = 4444444444444444444 := sorry

end comparison_of_products_l695_695966


namespace bus_speed_including_stoppages_l695_695268

theorem bus_speed_including_stoppages :
  ∀ (speed_stoppages : ℕ) (stop_time : ℕ),
  speed_stoppages = 60 → stop_time = 15 → 
  let actual_running_time := (60 - stop_time) in
  let speed_including_stoppages := speed_stoppages * actual_running_time / 60 in
  speed_including_stoppages = 45 := 
by
  intros speed_stoppages stop_time h_speed h_stop
  simp only [h_speed, h_stop]
  let actual_running_time := (60 - 15)
  have speed_including_stoppages : ℕ := 60 * actual_running_time / 60
  calc 
    60 * actual_running_time / 60 = 60 * 45 / 60   : by simp [actual_running_time]
                      ...      = 45                : by norm_num
  sorry

end bus_speed_including_stoppages_l695_695268


namespace total_boxes_l695_695940
namespace AppleBoxes

theorem total_boxes (initial_boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ)
  (apples_per_bag : ℕ) (bags_per_box : ℕ) (good_apples : ℕ) (final_boxes : ℕ) :
  initial_boxes = 14 →
  apples_per_box = 105 →
  rotten_apples = 84 →
  apples_per_bag = 6 →
  bags_per_box = 7 →
  final_boxes = (initial_boxes * apples_per_box - rotten_apples) / (apples_per_bag * bags_per_box) →
  final_boxes = 33 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  exact h6

end AppleBoxes

end total_boxes_l695_695940


namespace combined_distance_all_birds_two_seasons_l695_695172

-- Definition of the given conditions
def number_of_birds : Nat := 20
def distance_jim_to_disney : Nat := 50
def distance_disney_to_london : Nat := 60

-- The conclusion we need to prove
theorem combined_distance_all_birds_two_seasons :
  (distance_jim_to_disney + distance_disney_to_london) * number_of_birds = 2200 :=
by
  sorry

end combined_distance_all_birds_two_seasons_l695_695172


namespace ratio_of_voters_l695_695846

open Real

theorem ratio_of_voters (X Y : ℝ) (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) : X / Y = 2 :=
by
  sorry

end ratio_of_voters_l695_695846


namespace max_area_of_triangle_l695_695775

noncomputable def max_triangle_area (v1 v2 v3 : ℝ) (S : ℝ) : Prop :=
  2 * S + Real.sqrt 3 * (v1 * v2 + v3) = 0 ∧ v3 = Real.sqrt 3 → S ≤ Real.sqrt 3 / 4

theorem max_area_of_triangle (v1 v2 v3 S : ℝ) :
  max_triangle_area v1 v2 v3 S :=
by
  sorry

end max_area_of_triangle_l695_695775


namespace f_x_inequality_l695_695510

theorem f_x_inequality (f : ℝ → ℝ) (h_deriv : ∀ x, has_deriv_at f (f' x) x)
  (h_ineq : ∀ x, 2 * (f' x) > f x)
  (h_f_ln4 : f (Real.log 4) = 2) :
  {x : ℝ | f x > Real.exp (x / 2)} = Ioi (Real.log 4) :=
by sorry

end f_x_inequality_l695_695510


namespace distinct_real_roots_l695_695791

theorem distinct_real_roots (ω : ℂ) (n : ℕ) (hω : abs ω = 1) (hn : n > 0) :
  ∃ (x : ℕ → ℝ), (∀ k : ℕ, k < n → (1 + Complex.I * x k) / (1 - Complex.I * x k)) ^ n = ω ∧
  (∀ i j : ℕ, i < n → j < n → i ≠ j → x i ≠ x j) :=
  sorry

end distinct_real_roots_l695_695791


namespace probability_neither_square_nor_cube_l695_695539

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695539


namespace stage_8_area_l695_695364

theorem stage_8_area :
  let area_square := 4 * 4
  let area_rect (width : ℕ) := 4 * width
  let total_area_stage_1_to_4 := 4 * area_square
  let total_area_stage_5_to_8 := (area_rect 4) + (area_rect 6) + (area_rect 8) + (area_rect 10)
  let total_area_stage_8 := total_area_stage_1_to_4 + total_area_stage_5_to_8
  total_area_stage_8 = 176 := by
  let area_square := 4 * 4
  let area_rect (width : ℕ) := 4 * width
  let total_area_stage_1_to_4 := 4 * area_square
  let total_area_stage_5_to_8 := (area_rect 4) + (area_rect 6) + (area_rect 8) + (area_rect 10)
  let total_area_stage_8 := total_area_stage_1_to_4 + total_area_stage_5_to_8
  show total_area_stage_8 = 176 from sorry

end stage_8_area_l695_695364


namespace simplify_expression_l695_695468

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ( (x^6 - 1) / (3 * x^3) )^2) = Real.sqrt (x^12 + 7 * x^6 + 1) / (3 * x^3) :=
by sorry

end simplify_expression_l695_695468


namespace total_earnings_for_the_day_l695_695465

-- Definitions from the conditions
def kem_hourly_rate := 4
def shem_hourly_rate := 2.5 * kem_hourly_rate
def tiff_hourly_rate := kem_hourly_rate + 3
def kem_daily_hours := 6
def shem_daily_hours := 8
def tiff_daily_hours := 10

-- Calculations from definitions
def kem_daily_earnings := kem_hourly_rate * kem_daily_hours
def shem_daily_earnings := shem_hourly_rate * shem_daily_hours
def tiff_daily_earnings := tiff_hourly_rate * tiff_daily_hours
def total_daily_earnings := kem_daily_earnings + shem_daily_earnings + tiff_daily_earnings

-- Theorem statement
theorem total_earnings_for_the_day : total_daily_earnings = 174 := by
  sorry

end total_earnings_for_the_day_l695_695465


namespace sphere_radius_correct_l695_695609

-- Definitions reflecting the conditions of the problem
def Cone (radius : ℝ) (apex_angle : ℝ)

-- Parameters for the three cones
def cone1 := Cone 32 (π / 3)
def cone2 := Cone 48 (2 * π / 3)
def cone3 := Cone 48 (2 * π / 3)

-- Proving the radius of the sphere
theorem sphere_radius_correct (equi_dist : ∀ (O₁ O₂ O₃ : Cone), dist (center O) (center O₁) = dist (center O) (center O₂)
  ∧ dist (center O) (center O₂) = dist (center O) (center O₃))
  : ∃ (R : ℝ), R = 13 * (sqrt 3 + 1) := 
sorry

end sphere_radius_correct_l695_695609


namespace optimal_selection_exists_l695_695798

-- Define the given 5x5 matrix
def matrix : Matrix (Fin 5) (Fin 5) ℕ := ![
  ![11, 17, 25, 19, 16],
  ![24, 10, 13, 15, 3],
  ![12, 5, 14, 2, 18],
  ![23, 4, 1, 8, 22],
  ![6, 20, 7, 21, 9]
]

-- Define the statement of the proof problem
theorem optimal_selection_exists:
  ∃ (s : Finset (Fin 5 × Fin 5)), 
    (s.card = 5) ∧
    (∀ (i j : Fin 5 × Fin 5), i ∈ s → j ∈ s → i.1 ≠ j.1 ∧ i.2 ≠ j.2) ∧
    (∀ (t : Finset ℕ), (t = s.image (λ ij, matrix ij.1 ij.2)) → t.min' (by decide) = 15) :=
sorry

end optimal_selection_exists_l695_695798


namespace find_largest_even_integer_l695_695938

-- Define the sum of the first 30 positive even integers
def sum_first_30_even : ℕ := 2 * (30 * 31 / 2)

-- Assume five consecutive even integers and their sum
def consecutive_even_sum (m : ℕ) : ℕ := (m - 8) + (m - 6) + (m - 4) + (m - 2) + m

-- Statement of the theorem to be proven
theorem find_largest_even_integer : ∃ (m : ℕ), consecutive_even_sum m = sum_first_30_even ∧ m = 190 :=
by
  sorry

end find_largest_even_integer_l695_695938


namespace find_lambda_l695_695040

noncomputable def arithmetic_sequence_S (n : ℕ) : ℚ := (3 * n + 2) * n / 2
noncomputable def arithmetic_sequence_T (n : ℕ) : ℚ := (4 * n + 5) * n / 2

variables (a1 a4 b3 : ℚ)
axiom a1_a4_eq : a1 + a4 = 14
axiom b3_val : b3 = 25 / 2

theorem find_lambda (λ : ℚ) : 
  let S4 := arithmetic_sequence_S 4 in
  let T5 := arithmetic_sequence_T 5 in
  let a1a4 := a1 + a4 in
  let b3 := b3 in
  (a1a4 / b3) + λ = 1 → λ = -3 / 25 :=
by
  intros
  sorry

end find_lambda_l695_695040


namespace meeting_point_l695_695896

def midpoint (x1 y1 x2 y2 : ℤ) : ℤ × ℤ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem meeting_point :
  let mark := (2, 9)
  let sandy := (-6, 3)
  midpoint mark.1 mark.2 sandy.1 sandy.2 = (-2, 6) :=
by
  let mark := (2, 9)
  let sandy := (-6, 3)
  show midpoint mark.1 mark.2 sandy.1 sandy.2 = (-2, 6)
  sorry

end meeting_point_l695_695896


namespace point_symmetry_wrt_x_axis_l695_695112

theorem point_symmetry_wrt_x_axis :
  ∀ (P : ℝ × ℝ) (P_sym : ℝ × ℝ), P = (-3, 5) → P_sym = (P.1, -P.2) → P_sym = (-3, -5) :=
by
  intros P P_sym hP_symm1 hP_symm2
  rw [hP_symm2, hP_symm1]
  simp
  sorry

end point_symmetry_wrt_x_axis_l695_695112


namespace num_triples_satisfying_gcd_lcm_l695_695275

open Nat

theorem num_triples_satisfying_gcd_lcm : 
  (∃ (a b c : ℕ), gcd (gcd a b) c = 14 ∧ lcm (lcm a b) c = 2^17 * 7^18) →
  (set.finite { p : ℕ × ℕ × ℕ | gcd (gcd p.1 p.2.1) p.2.2 = 14 ∧ lcm (lcm p.1 p.2.1) p.2.2 = 2^17 * 7^18 }) ∧ 
  (set.to_finset { p : ℕ × ℕ × ℕ | gcd (gcd p.1 p.2.1) p.2.2 = 14 ∧ lcm (lcm p.1 p.2.1) p.2.2 = 2^17 * 7^18 }).card = 9792 :=
sorry

end num_triples_satisfying_gcd_lcm_l695_695275


namespace intersection_points_of_lines_l695_695148

theorem intersection_points_of_lines :
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ x + 3 * y = 3 ∧ x = 10 / 11 ∧ y = 13 / 11) ∧
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ 5 * x - 3 * y = 6 ∧ x = 24 ∧ y = 38) :=
by
  sorry

end intersection_points_of_lines_l695_695148


namespace line_interparabola_length_l695_695666

theorem line_interparabola_length :
  (∀ (x y : ℝ), y = x - 2 → y^2 = 4 * x) →
  ∃ (A B : ℝ × ℝ), (∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2)) →
  (dist A B = 4 * Real.sqrt 6) :=
by
  intros
  sorry

end line_interparabola_length_l695_695666


namespace inverse_at_1_l695_695060

def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem inverse_at_1 : (∃ g : ℝ → ℝ, ∀ y, f (g y) = y ∧ g (f y) = y) → ∃ g : ℝ → ℝ, g 1 = 1 :=
by
  intros h_exists
  sorry

end inverse_at_1_l695_695060


namespace solve_exp_equation_l695_695493

theorem solve_exp_equation (x : ℝ) : 4^x - 6 * 2^x - 16 = 0 → x = 3 :=
by
  intro h
  sorry

end solve_exp_equation_l695_695493


namespace combined_weight_difference_l695_695873

def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := -5.25
def biology_weight : ℝ := 3.755

theorem combined_weight_difference :
  (chemistry_weight - calculus_weight) - (geometry_weight + biology_weight) = 7.995 :=
by
  sorry

end combined_weight_difference_l695_695873


namespace smallest_positive_integer_solves_congruence_l695_695964

theorem smallest_positive_integer_solves_congruence :
  ∃ x : ℕ, x > 0 ∧ (4 * x ≡ 17 [MOD 31]) ∧ x = 12 :=
by
  use 12
  split
  { linarith }
  split
  { sorry }
  { rfl }

end smallest_positive_integer_solves_congruence_l695_695964


namespace telephone_number_solution_l695_695203

theorem telephone_number_solution
  (A B C D E F G H I J : ℕ)
  (h_distinct : list.nodup [A, B, C, D, E, F, G, H, I, J])
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_DEF : D = E + 1 ∧ E = F + 1)
  (h_consecutive_GHIJ : (G = H + 1 ∧ H = I + 1 ∧ I = J + 1) ∨ (G = H + 2 ∧ H = I + 2 ∧ I = J + 2))
  (h_sum_ABC : A + B + C = 10) :
  A = 5 :=
begin
  sorry,
end

end telephone_number_solution_l695_695203


namespace sum_of_circle_areas_is_correct_l695_695139

variable (O : Point)
variable (half_line1 half_line2 : Line)
variable (C : ℕ → Circle)
variable (d : ℕ → ℝ)  -- distance of the center of circles from O
variable (r : ℕ → ℝ)  -- radius of the circles
variable (r1 d1 : ℝ)  -- initial radius and distance of circle C1

-- Conditions
axiom (C1_tangent : TangentToBothHalfLines O half_line1 half_line2 (C 1))
axiom (C1_distance : (C 1).center.dist O = d1)
axiom (C1_radius : (C 1).radius = r1)
axiom (Cn_tangent (n : ℕ) : TangentToBothHalfLines O half_line1 half_line2 (C n) ∧ TangentExternal (C (n - 1)) (C n) ∧ (C n).center.dist O < (C (n - 1)).center.dist O)

noncomputable def sum_of_areas_infinite_series : ℝ :=
  let α := r 1 / d 1 in
  let r_geo := (d 1 - r 1) / (d 1 + r 1) in
  π * r 1^2 / (1 - r_geo^2)

theorem sum_of_circle_areas_is_correct :
  sum_of_areas_infinite_series O half_line1 half_line2 C d r r1 d1 = 
  π / 4 * r 1 * (d 1 + r 1)^2 / d 1 := sorry

end sum_of_circle_areas_is_correct_l695_695139


namespace value_of_a_l695_695699

def g (x : ℝ) : ℝ := 5 * x - 7

theorem value_of_a (a : ℝ) : g(a) = 0 ↔ a = 7 / 5 := by
  sorry

end value_of_a_l695_695699


namespace find_b_l695_695594

open Set

-- Define the points
def P1 : ℝ × ℝ := (6, -10)
def P2 (b : ℝ) : ℝ × ℝ := (-b + 4, 3)
def P3 (b : ℝ) : ℝ × ℝ := (3b + 6, 3)

-- Define the collinearity condition for the points
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_b (b : ℝ) (h : collinear P1 (P2 b) (P3 b)) : b = -1 / 2 := by
  sorry

end find_b_l695_695594


namespace smallest_even_sum_is_102_l695_695937

theorem smallest_even_sum_is_102 (s : Int) (h₁ : ∃ a b c d e f g : Int, s = a + b + c + d + e + f + g)
    (h₂ : ∀ a b c d e f g : Int, b = a + 2 ∧ c = a + 4 ∧ d = a + 6 ∧ e = a + 8 ∧ f = a + 10 ∧ g = a + 12)
    (h₃ : s = 756) : ∃ a : Int, a = 102 :=
  by
    sorry

end smallest_even_sum_is_102_l695_695937


namespace scientific_notation_of_10900_l695_695133

theorem scientific_notation_of_10900 : ∃ (x : ℝ) (n : ℤ), 10900 = x * 10^n ∧ x = 1.09 ∧ n = 4 := by
  use 1.09
  use 4
  sorry

end scientific_notation_of_10900_l695_695133


namespace embankment_height_bounds_l695_695197

theorem embankment_height_bounds
  (a : ℝ) (b : ℝ) (h : ℝ)
  (a_eq : a = 5)
  (b_lower_bound : 2 ≤ b)
  (vol_lower_bound : 400 ≤ (25 * (a^2 - b^2)))
  (vol_upper_bound : (25 * (a^2 - b^2)) ≤ 500) :
  1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by
  sorry

end embankment_height_bounds_l695_695197


namespace probability_neither_square_nor_cube_l695_695559

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695559


namespace incorrect_calculation_l695_695632

theorem incorrect_calculation :
  ¬ (sqrt (3 ^ 2) + sqrt (3 ^ 2) = 3 ^ 2) :=
by
  -- We can see that
  -- sqrt (3 ^ 2) = 3
  -- so sqrt (3 ^ 2) + sqrt (3 ^ 2) = 3 + 3 = 6
  -- but 3 ^ 2 = 9
  -- and hence 6 ≠ 9
  sorry

end incorrect_calculation_l695_695632


namespace part1_part2_l695_695766

def z : ℂ := (3 + 2 * complex.I) / (2 - 3 * complex.I)

theorem part1 : |z - 1 - 2 * complex.I| = real.sqrt 2 :=
by sorry

theorem part2 : ∑ k in finset.range 2021, z ^ (k + 1) = complex.I :=
by sorry

end part1_part2_l695_695766


namespace probability_neither_square_nor_cube_l695_695564

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695564


namespace fraction_to_decimal_l695_695246

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
sorry

end fraction_to_decimal_l695_695246


namespace cone_base_radius_l695_695794

theorem cone_base_radius
  (R : ℝ) -- radius of the base of the cone
  (l : ℝ) -- slant height of the cone
  (A : ℝ) -- lateral surface area of the cone
  (h_l : l = 6) -- given slant height is 6 cm
  (h_A : A = 18 * real.pi) -- given lateral surface area is 18π cm²
  (h_LSA : A = (1 / 2) * (2 * real.pi * R) * l) : -- formula for the lateral surface area
  R = 3 :=
by
  sorry

end cone_base_radius_l695_695794


namespace last_two_nonzero_digits_80_fact_l695_695124

-- Define the factorial function 
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def last_two_nonzero_digits (n : ℕ) : ℕ := sorry

theorem last_two_nonzero_digits_80_fact : last_two_nonzero_digits (factorial 80) = 76 := sorry

end last_two_nonzero_digits_80_fact_l695_695124


namespace wheel_distance_covered_l695_695208

noncomputable def diameter : ℝ := 15
noncomputable def revolutions : ℝ := 11.210191082802547
noncomputable def pi : ℝ := Real.pi -- or you can use the approximate value if required: 3.14159
noncomputable def circumference : ℝ := pi * diameter
noncomputable def distance_covered : ℝ := circumference * revolutions

theorem wheel_distance_covered :
  distance_covered = 528.316820577 := 
by
  unfold distance_covered
  unfold circumference
  unfold diameter
  unfold revolutions
  norm_num
  sorry

end wheel_distance_covered_l695_695208


namespace prob_rain_both_days_l695_695596

-- Declare the probabilities involved
def P_Monday : ℝ := 0.40
def P_Tuesday : ℝ := 0.30
def P_Tuesday_given_Monday : ℝ := 0.30

-- Prove the probability of it raining on both days
theorem prob_rain_both_days : P_Monday * P_Tuesday_given_Monday = 0.12 :=
by
  sorry

end prob_rain_both_days_l695_695596


namespace ellipse_foci_y_axis_iff_l695_695111

theorem ellipse_foci_y_axis_iff (m n : ℝ) (h : m > n ∧ n > 0) :
  (m > n ∧ n > 0) ↔ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → ∃ a b : ℝ, a^2 - b^2 = 1 ∧ x^2/b^2 + y^2/a^2 = 1 ∧ a > b) :=
sorry

end ellipse_foci_y_axis_iff_l695_695111


namespace total_volume_is_30_l695_695179

-- Define the amounts of the solutions
def volume1 : ℝ := 15
def volume2 : ℝ := 15

-- Define the total volume of the mixture
def total_volume : ℝ := volume1 + volume2

-- The formal statement
theorem total_volume_is_30
    (v1 : ℝ = volume1)
    (v2 : ℝ = volume2)
    : total_volume = 30 :=
    by
    sorry

end total_volume_is_30_l695_695179


namespace sector_area_eq_l695_695624

theorem sector_area_eq (d : ℝ) (θ : ℝ) (hd : d = 8) (hθ : θ = 60) : 
  (1 / 6) * d^2 * Real.pi = (8 * Real.pi) / 3 :=
by
  -- Assumptions needed for the proof
  have hr : d / 2 = 4, from sorry,
  have ha : Real.pi * (d / 2)^2 = 16 * Real.pi, from sorry,
  -- Actual proof 
  sorry

end sector_area_eq_l695_695624


namespace probability_neither_square_nor_cube_l695_695541

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695541


namespace general_term_of_an_l695_695341

theorem general_term_of_an (a : ℕ → ℕ) (h1 : a 1 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) :
    ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end general_term_of_an_l695_695341


namespace product_ab_range_l695_695830

theorem product_ab_range (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ab = a + b + 3) : ab ≥ (7 * real.sqrt 3) / 2 := 
sorry

end product_ab_range_l695_695830


namespace det1_value_system_solution_l695_695913

-- Definitions from conditions
def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

-- Given values from the problem
def det1 := determinant (-2) 3 2 1

def a1 := 2
def b1 := -1
def c1 := 1
def a2 := 3
def b2 := 2
def c2 := 11

def D := determinant a1 b1 a2 b2
def Dx := determinant c1 b1 c2 b2
def Dy := determinant a1 c1 a2 c2

def x := Dx / D
def y := Dy / D

-- Theorem statements to be proved
theorem det1_value : det1 = -8 := 
by 
  unfold det1 determinant 
  sorry

theorem system_solution : x = 13 / 7 ∧ y = 19 / 7 := 
by 
  unfold x y Dx Dy D determinant 
  sorry

end det1_value_system_solution_l695_695913


namespace sqrt_equal_iff_l695_695759

-- Definitions for even positive integers
def even_positive (k : ℕ) : Prop := (k > 0) ∧ (∃ d, k = 2 * d)

-- The main theorem statement
theorem sqrt_equal_iff (m n p : ℕ) (hm : even_positive m) (hn : even_positive n) (hp : even_positive p)
  (hmp : m ≥ p) (hnp : n ≥ p) : 
  sqrt m n (m - p) = sqrt n m (n - p) ↔ m = n ∧ m ≥ p :=
sorry

end sqrt_equal_iff_l695_695759


namespace variance_is_three_halves_l695_695176

noncomputable def variance_of_remaining_scores (scores : List ℕ) : ℚ :=
  let scores' := (scores.sorted.drop 1) -- drop the lowest score
                 |>.take (scores.length - 2) -- drop the highest score
  let mean := (scores'.sum : ℚ) / scores'.length
  (scores'.map (λ x => (x - mean) ^ 2)).sum / scores'.length

theorem variance_is_three_halves (x : ℕ) (h : x ≥ 3) :
  let scores := [89, 90, 90, 91, 93, 90 + x]
  let remaining_scores := List.erase (List.erase scores (scores.maximum? h).getOrElse 0) (scores.minimum? h).getOrElse 0
  remaining_scores.sum / (remaining_scores.length : ℚ) = 91 →
  variance_of_remaining_scores scores = 3 / 2 :=
by
  sorry

end variance_is_three_halves_l695_695176


namespace axis_of_symmetry_parabola_l695_695109

theorem axis_of_symmetry_parabola :
  ∀ x y : ℝ, y = (x - 5)^2 + 4 → (∃ y', (5, y') ∈ set_of (λ p : ℝ × ℝ, y = (fst p - 5)^2 + snd p)) := 
begin
  sorry
end

end axis_of_symmetry_parabola_l695_695109


namespace ellipse_intersects_x_axis_at_four_l695_695220

theorem ellipse_intersects_x_axis_at_four
    (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 0))
    (h2 : f2 = (4, 0))
    (h3 : ∃ P : ℝ × ℝ, P = (1, 0) ∧ (dist P f1 + dist P f2 = 4)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ (dist Q f1 + dist Q f2 = 4) :=
sorry

end ellipse_intersects_x_axis_at_four_l695_695220


namespace shift_function_correct_l695_695331

variables (φ x : ℝ) (k : ℤ)

-- Define the function f(x)
noncomputable def f (x : ℝ) := sin (1/2 * x + φ)

-- Define the condition that x = π/3 is the symmetric axis of f(x)
axiom symmetric_axis : x = π / 3

-- Define the condition that φ is bound within |φ| < π/2
axiom phi_bound : |φ| < π / 2

-- Express the correctness statement
theorem shift_function_correct :
  (∀ φ, |φ| < π / 2 → f (π / 3) = sin (π / 6 + φ) → φ = π / 3) →
  g (x - π / 3) = cos (1/2 * x) :=
sorry

end shift_function_correct_l695_695331


namespace total_payment_l695_695070

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l695_695070


namespace max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l695_695670

noncomputable def point_in_circle : Prop :=
  let P := (-Real.sqrt 3, 2)
  ∃ (x y : ℝ), x^2 + y^2 = 12 ∧ x = -Real.sqrt 3 ∧ y = 2

theorem max_min_AB_length (α : ℝ) (h1 : -Real.sqrt 3 ≤ α ∧ α ≤ Real.pi / 2) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let R := Real.sqrt 12
  ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12 ∧ (P.1, P.2) = (-Real.sqrt 3, 2)) →
    ((max (dist A B) (dist P P)) = 4 * Real.sqrt 3 ∧ (min (dist A B) (dist P P)) = 2 * Real.sqrt 5) :=
sorry

theorem chord_length_at_angle (α : ℝ) (h2 : α = 120 / 180 * Real.pi) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let A := (Real.sqrt 12, 0)
  let B := (-Real.sqrt 12, 0)
  let AB := (dist A B)
  AB = Real.sqrt 47 :=
sorry

theorem trajectory_midpoint_chord :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  ∀ (M : ℝ × ℝ), (∀ k : ℝ, P.2 - 2 = k * (P.1 + Real.sqrt 3) ∧ M.2 = - 1 / k * M.1) → 
  (M.1^2 + M.2^2 + Real.sqrt 3 * M.1 + 2 * M.2 = 0) :=
sorry

end max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l695_695670


namespace expand_product_correct_l695_695717

noncomputable def expand_product (x : ℝ) : ℝ :=
  3 * (x + 4) * (x + 5)

theorem expand_product_correct (x : ℝ) :
  expand_product x = 3 * x^2 + 27 * x + 60 :=
by
  unfold expand_product
  sorry

end expand_product_correct_l695_695717


namespace fish_count_and_total_l695_695448

-- Definitions of each friend's number of fish
def max_fish : ℕ := 6
def sam_fish : ℕ := 3 * max_fish
def joe_fish : ℕ := 9 * sam_fish
def harry_fish : ℕ := 5 * joe_fish

-- Total number of fish for all friends combined
def total_fish : ℕ := max_fish + sam_fish + joe_fish + harry_fish

-- The theorem stating the problem and corresponding solution
theorem fish_count_and_total :
  max_fish = 6 ∧
  sam_fish = 3 * max_fish ∧
  joe_fish = 9 * sam_fish ∧
  harry_fish = 5 * joe_fish ∧
  total_fish = (max_fish + sam_fish + joe_fish + harry_fish) :=
by
  repeat { sorry }

end fish_count_and_total_l695_695448


namespace david_math_homework_time_l695_695703

def total_time_spent_on_homework : ℕ := 60
def time_spent_on_spelling_homework : ℕ := 18
def time_spent_reading : ℕ := 27
def time_spent_on_math_homework : ℕ := total_time_spent_on_homework - time_spent_on_spelling_homework - time_spent_reading

theorem david_math_homework_time : time_spent_on_math_homework = 15 :=
by
  -- given conditions
  have h1 : total_time_spent_on_homework = 60 := rfl
  have h2 : time_spent_on_spelling_homework = 18 := rfl
  have h3 : time_spent_reading = 27 := rfl
  -- use conditions to calculate
  have h4 : time_spent_on_math_homework = 60 - 18 - 27 := rfl
  -- conclude
  finish

end david_math_homework_time_l695_695703


namespace rita_bought_4_jackets_l695_695085

/-
Given:
  - Rita bought 5 short dresses costing $20 each.
  - Rita bought 3 pairs of pants costing $12 each.
  - The jackets cost $30 each.
  - She spent an additional $5 on transportation.
  - Rita had $400 initially.
  - Rita now has $139.

Prove that the number of jackets Rita bought is 4.
-/

theorem rita_bought_4_jackets :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let transportation_cost := 5
  let initial_amount := 400
  let remaining_amount := 139
  let jackets_cost_per_unit := 30
  let total_spent := initial_amount - remaining_amount
  let total_clothes_transportation_cost := dresses_cost + pants_cost + transportation_cost
  let jackets_cost := total_spent - total_clothes_transportation_cost
  let number_of_jackets := jackets_cost / jackets_cost_per_unit
  number_of_jackets = 4 :=
by
  sorry

end rita_bought_4_jackets_l695_695085


namespace distinct_possible_lunches_l695_695226

namespace SchoolCafeteria

def main_courses : List String := ["Hamburger", "Veggie Burger", "Chicken Sandwich", "Pasta"]
def beverages_when_meat_free : List String := ["Water", "Soda"]
def beverages_when_meat : List String := ["Water"]
def snacks : List String := ["Apple Pie", "Fruit Cup"]

-- Count the total number of distinct possible lunches
def count_distinct_lunches : Nat :=
  let count_options (main_course : String) : Nat :=
    if main_course = "Hamburger" ∨ main_course = "Chicken Sandwich" then
      beverages_when_meat.length * snacks.length
    else
      beverages_when_meat_free.length * snacks.length
  (main_courses.map count_options).sum

theorem distinct_possible_lunches : count_distinct_lunches = 12 := by
  sorry

end SchoolCafeteria

end distinct_possible_lunches_l695_695226


namespace probability_neither_square_nor_cube_l695_695570

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695570


namespace ratio_of_min_max_S_l695_695779

theorem ratio_of_min_max_S {a b c : ℝ}
  (h1 : (a - 1) / 2 = (b - 2) / 3)
  (h2 : (b - 2) / 3 = (3 - c) / 4)
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  let S := a + 2 * b + c in
  let n := 6 in
  let m := 11 in
  n / m = 6 / 11 :=
by
  sorry

end ratio_of_min_max_S_l695_695779


namespace admission_plans_count_l695_695943

/-- Given 4 students and 3 schools, where each school must admit at least one student,
    the total number of different admission plans is 36. -/
theorem admission_plans_count : 
  let students := {A, B, C, D}
  let schools := {Alpha, Beta, Gamma}
  (∀ school ∈ schools, ∃ student ∈ students, student ∈ admits school) →
  (total_admission_plans students schools = 36) :=
sorry

end admission_plans_count_l695_695943


namespace find_angle_ACB_l695_695839

-- Variables and conditions
variables (A B C D : Type) 
variables [IsTriangle A B C]
variables (angle_ABC : ∠ABC = 45)
variables (BD_CD_ratio : ∃ D : Type, BD = 2 * CD)
variables (angle_DAB : ∠DAB = 30)

-- Goal
theorem find_angle_ACB : ∠ACB = 60 := 
by
  sorry

end find_angle_ACB_l695_695839


namespace tangent_line_equation_at_point_l695_695926

def curve (x : ℝ) : ℝ := -x^2 + 1
def point : ℝ × ℝ := (1, 0)

theorem tangent_line_equation_at_point :
  ∀ x y : ℝ, point = (1, 0) → y = curve x → 
  ∃ m b : ℝ, m = -2 ∧ b = 2 ∧ m * x + b = y :=
by
  sorry

end tangent_line_equation_at_point_l695_695926


namespace how_many_women_left_l695_695395

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l695_695395


namespace max_value_200_max_value_attained_l695_695888

noncomputable def max_value (X Y Z : ℕ) : ℕ := 
  X * Y * Z + X * Y + Y * Z + Z * X

theorem max_value_200 (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  max_value X Y Z ≤ 200 :=
sorry

theorem max_value_attained (X Y Z : ℕ) (h : X = 5) (h1 : Y = 5) (h2 : Z = 5) : 
  max_value X Y Z = 200 :=
sorry

end max_value_200_max_value_attained_l695_695888


namespace probability_neither_perfect_square_nor_cube_l695_695522

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695522


namespace vector_proof_l695_695144

variables
  (A B C D E H K P M N Q : Type*)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]
  [AddGroup H] [AddGroup K] [AddGroup P] [AddGroup M] [AddGroup N] [AddGroup Q]

variables
  (overline_AE overline_AD overline_AC overline_AB overline_HK : A)
  (overline_AM overline_AP overline_AN overline_AQ overline_AH overline_AK : B)

axiom h1 : overline_HK = (overline_AK - overline_AH)
axiom h2 : overline_AK = 1/2 * (overline_AQ + overline_AN)
axiom h3 : overline_AH = 1/2 * (overline_AM + overline_AP)
axiom h4 : overline_AQ = 1/2 * (overline_AE + overline_AD)
axiom h5 : overline_AN = 1/2 * (overline_AB + overline_AC)
axiom h6 : overline_AM = 1/2 * overline_AB
axiom h7 : overline_AP = 1/2 * (overline_AC + overline_AD)

theorem vector_proof : overline_HK = 1/4 * overline_AE :=
by
  sorry

end vector_proof_l695_695144


namespace diagonal_sum_difference_l695_695853

def table_value (n : ℕ) (i j : ℕ) : ℕ := j + 1

def sum_below_diagonal (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, ∑ j in Finset.range i, table_value n (i + 1) (j + 1)

def sum_above_diagonal (n : ℕ) : ℕ :=
  ∑ j in Finset.range n, ∑ i in Finset.range j, table_value n (i + 1) (j + 1)

theorem diagonal_sum_difference (n : ℕ) :
  sum_above_diagonal n = 2 * sum_below_diagonal n :=
sorry

end diagonal_sum_difference_l695_695853


namespace peter_wins_l695_695906

-- Define the type for the 8x8 board and the types of moves
inductive CellColor
| none | red | green

structure Board :=
  (cells : Fin 8 → Fin 8 → CellColor)
  (valid_moves : list ((Fin 8 × Fin 8) × (Fin 8 × Fin 8))) -- Moves are pairs of cell positions

-- Introducing the notion of symmetry on the board
def symmetrical {α : Type*} (f : α → α) (b : Board) : Prop := ∀ i j, f (f (b.cells i j)) = b.cells i j

-- Define the initial condition of the board
def initial_board : Board :=
  { cells := λ _ _, CellColor.none,
    valid_moves := [((⟨0, sorry⟩, ⟨0, sorry⟩), (⟨0, sorry⟩, ⟨1, sorry⟩)), -- Example moves, need actual valid ones
                    ((⟨0, sorry⟩, ⟨0, sorry⟩), (⟨1, sorry⟩, ⟨1, sorry⟩)),
                    ((⟨0, sorry⟩, ⟨0, sorry⟩), (⟨2, sorry⟩, ⟨2, sorry⟩))] }

-- The main theorem - Peter has a winning strategy
theorem peter_wins : 
  ∃ strategy : Board → Board, (∀ b : Board, symmetrical strategy b) →
  (∃ b' : Board, strategy initial_board = b') :=
sorry

end peter_wins_l695_695906


namespace my_age_is_five_times_son_age_l695_695442

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l695_695442


namespace SmallestPositiveAngle_l695_695737

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l695_695737


namespace age_proof_l695_695445

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l695_695445


namespace solve_fraction_equation_l695_695095

theorem solve_fraction_equation (x : ℝ) : (3/5)^x = (5/3)^9 ↔ x = -9 :=
by sorry

end solve_fraction_equation_l695_695095


namespace remove_four_digits_and_sum_divisible_by_six_l695_695123

noncomputable def number_str : String := "2458710411"
noncomputable def repetitions : Nat := 98
noncomputable def num_len : Nat := String.length number_str * repetitions

theorem remove_four_digits_and_sum_divisible_by_six :
  ∃ count : ℕ, 
    (count = 90894 ∧ 
    (∀ l, l.length = num_len → 
    (∃ m k, m + k = list_sum l ∧ m % 6 = 0 ∧ k % 6 = 0))) :=
sorry

end remove_four_digits_and_sum_divisible_by_six_l695_695123


namespace concyclic_points_l695_695168

-- Definitions of points A, B, C, D and intersections M, K
variables {A B C D M K : Type}

-- Convex quadrilateral and segment conditions
axiom h1 : convex_quadrilateral A B C D
axiom h2 : dist A B = dist B C
axiom h3 : dist B C = dist C D
axiom h4 : M = intersection (line A C) (line B D)
axiom h5 : K = intersection (angle_bisector ∠ A) (angle_bisector ∠ D)

-- The proof we need to show
theorem concyclic_points (h1 : convex_quadrilateral A B C D)
                         (h2 : dist A B = dist B C)
                         (h3 : dist B C = dist C D)
                         (h4 : M = intersection (line A C) (line B D))
                         (h5 : K = intersection (angle_bisector ∠ A) (angle_bisector ∠ D)) :
                         ∠ A K D = ∠ A M D := 
begin
  sorry
end

end concyclic_points_l695_695168


namespace sum_abs_diff_eq_2500_l695_695372

theorem sum_abs_diff_eq_2500 :
  ∀ (A B : Fin 50 → ℕ) (U : Finset ℕ),
    (∀ i, A i ∈ U) ∧ (∀ i, B i ∈ U) ∧
    (∀ i j, i < j → A i < A j) ∧
    (∀ i j, i < j → B i > B j) ∧
    U = Finset.range 101 →
    (Finset.range 50).sum (λ i, abs (A ⟨i, sorry⟩ - B ⟨i, sorry⟩)) = 2500 :=
by sorry

end sum_abs_diff_eq_2500_l695_695372


namespace consecutive_even_numbers_average_35_greatest_39_l695_695500

-- Defining the conditions of the problem
def average_of_even_numbers (n : ℕ) (S : ℕ) : ℕ := (n * S + (2 * n * (n - 1)) / 2) / n

-- Main statement to be proven
theorem consecutive_even_numbers_average_35_greatest_39 : 
  ∃ (n : ℕ), average_of_even_numbers n (38 - (n - 1) * 2) = 35 ∧ (38 - (n - 1) * 2) + (n - 1) * 2 = 38 :=
by
  sorry

end consecutive_even_numbers_average_35_greatest_39_l695_695500


namespace sum_of_distinct_prime_factors_of_1320_l695_695965

theorem sum_of_distinct_prime_factors_of_1320 :
  let n := 1320
  let prime_factors := [2, 3, 5, 11]
  ∀ (distinct_pf : List ℕ) (h : distinct_pf = prime_factors),
    distinct_pf.sum = 21 :=
by
  let n := 1320
  let prime_factors := [2, 3, 5, 11]
  intros distinct_pf h
  simp [h, List.sum]
  sorry

end sum_of_distinct_prime_factors_of_1320_l695_695965


namespace find_number_of_violas_l695_695655

theorem find_number_of_violas (cellos : ℕ) (pairs : ℕ) (probability : ℚ) 
    (h1 : cellos = 800) 
    (h2 : pairs = 100) 
    (h3 : probability = 0.00020833333333333335) : 
    ∃ V : ℕ, V = 600 := 
by 
    sorry

end find_number_of_violas_l695_695655


namespace wire_leftover_after_hexagon_l695_695175

theorem wire_leftover_after_hexagon
    (original_length : ℝ)
    (side_length : ℝ)
    (hexagon_sides : ℕ) :
    original_length = 50 → 
    side_length = 8 → 
    hexagon_sides = 6 → 
    (original_length - (hexagon_sides * side_length)) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end wire_leftover_after_hexagon_l695_695175


namespace payment_correct_l695_695067

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l695_695067


namespace negative_number_from_operations_l695_695969

theorem negative_number_from_operations :
  (∀ (a b : Int), a + b < 0 → a = -1 ∧ b = -3) ∧
  (∀ (a b : Int), a - b < 0 → a = 1 ∧ b = 4) ∧
  (∀ (a b : Int), a * b > 0 → a = 3 ∧ b = -2) ∧
  (∀ (a b : Int), a / b = 0 → a = 0 ∧ b = -7) :=
by
  sorry

end negative_number_from_operations_l695_695969


namespace quotient_of_fifths_l695_695352

theorem quotient_of_fifths : (2 / 5) / (1 / 5) = 2 := 
  by 
    sorry

end quotient_of_fifths_l695_695352


namespace intersected_cubes_half_nonintersected_l695_695956

theorem intersected_cubes_half_nonintersected (n : ℕ) :
  (4 * n = (n^3 - 4 * n) / 2 ∧ even n) ∨
  (4 * n - 3 = (n^3 - 4 * n + 3) / 2 ∧ odd n) → n = 3 :=
begin
  sorry
end

end intersected_cubes_half_nonintersected_l695_695956


namespace lcm_of_two_numbers_hcf_and_product_l695_695140

theorem lcm_of_two_numbers_hcf_and_product (a b : ℕ) (h_hcf : Nat.gcd a b = 20) (h_prod : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_of_two_numbers_hcf_and_product_l695_695140


namespace hydrogen_atomic_weight_is_correct_l695_695727

-- Definitions and assumptions based on conditions
def molecular_weight : ℝ := 68
def number_of_hydrogen_atoms : ℕ := 1
def number_of_chlorine_atoms : ℕ := 1
def number_of_oxygen_atoms : ℕ := 2
def atomic_weight_chlorine : ℝ := 35.45
def atomic_weight_oxygen : ℝ := 16.00

-- Definition for the atomic weight of hydrogen to be proved
def atomic_weight_hydrogen (w : ℝ) : Prop :=
  w * number_of_hydrogen_atoms
  + atomic_weight_chlorine * number_of_chlorine_atoms
  + atomic_weight_oxygen * number_of_oxygen_atoms = molecular_weight

-- The theorem to prove the atomic weight of hydrogen
theorem hydrogen_atomic_weight_is_correct : atomic_weight_hydrogen 1.008 :=
by
  unfold atomic_weight_hydrogen
  simp
  sorry

end hydrogen_atomic_weight_is_correct_l695_695727


namespace function_identity_l695_695188

theorem function_identity (f : ℕ → ℕ)
  (h : ∀ m n : ℕ, (m^2 + n)^2 % (f(m)^2 + f(n)) = 0) :
  ∀ n : ℕ, f(n) = n :=
sorry

end function_identity_l695_695188


namespace parabola_focus_symmetry_l695_695509

theorem parabola_focus_symmetry :
  let focus_orig := (-2, 0)
  ∃ a b : ℝ, ((b / 2 = (-2 + a) / 2 - 1) ∧ (b / (a + 2) = -1)) → (a, b) = (1, -3) :=
begin
  intro focus_orig,
  use (1: ℝ),
  use (-3: ℝ),
  intro h,
  split,
  { exact real.eq_iff (h.1) (by linarith using h.2) },
  { exact h },
end

end parabola_focus_symmetry_l695_695509


namespace parallel_lines_condition_l695_695347

theorem parallel_lines_condition (k1 k2 b : ℝ) (l1 l2 : ℝ → ℝ) (H1 : ∀ x, l1 x = k1 * x + 1)
  (H2 : ∀ x, l2 x = k2 * x + b) : (∀ x, l1 x = l2 x ↔ k1 = k2 ∧ b = 1) → (k1 = k2) ↔ (∀ x, l1 x ≠ l2 x ∧ l1 x - l2 x = 1 - b) := 
by
  sorry

end parallel_lines_condition_l695_695347


namespace initial_customers_l695_695207

/-- Given that 5 customers left and 9 customers remain, prove that the initial number of customers was 14. -/
theorem initial_customers (num_left : ℕ) (num_remaining : ℕ) (h1 : num_left = 5) (h2 : num_remaining = 9) :
  num_remaining + num_left = 14 :=
by
  rw [h1, h2]
  norm_num
  sorry

end initial_customers_l695_695207


namespace blue_horses_count_l695_695922

theorem blue_horses_count :
  ∃ (B : ℕ), 
  let P := 3 * B in
  let G := 2 * P in
  let Gold := (1 / 6) * G in
  B + P + G + Gold = 33 ∧ B = 3 :=
by
  -- let variables be defined first
  let B := 3
  let P := 3 * B
  let G := 2 * P
  let Gold := (1 / 6) * G
  have h1 : B + P + G + Gold = 33 := by sorry
  have h2 : B = 3 := by sorry
  existsi B
  exact ⟨h1, h2⟩

end blue_horses_count_l695_695922


namespace square_area_proof_l695_695201

theorem square_area_proof
  (x s : ℝ)
  (h1 : x^2 = 3 * s)
  (h2 : 4 * x = s^2) :
  x^2 = 6 :=
  sorry

end square_area_proof_l695_695201


namespace initial_water_amount_l695_695995

theorem initial_water_amount (x : ℝ) (h : x + 6.8 = 9.8) : x = 3 := 
by
  sorry

end initial_water_amount_l695_695995


namespace three_pow_12_mul_three_pow_8_equals_243_pow_4_l695_695361

theorem three_pow_12_mul_three_pow_8_equals_243_pow_4 : 3^12 * 3^8 = 243^4 := 
by sorry

end three_pow_12_mul_three_pow_8_equals_243_pow_4_l695_695361


namespace min_ab_minus_cd_l695_695340

theorem min_ab_minus_cd (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9) (h5 : a^2 + b^2 + c^2 + d^2 = 21) : ab - cd ≥ 2 := sorry

end min_ab_minus_cd_l695_695340


namespace coffee_equals_milk_l695_695752

variable (C : ℝ) (M : ℝ)

-- Defining the initial condition
def initial_coffee (C : ℝ) : ℝ := 1
def initial_milk (M : ℝ) : ℝ := 0

-- Defining the conditions for each step and refill
def step1_coffee (C : ℝ) : ℝ := 1 - (1 / 6)
def step1_milk (M : ℝ) : ℝ := 1 / 6

def step2_coffee (C : ℝ) : ℝ := step1_coffee C - ((step1_coffee C) / 3)
def step2_milk (M : ℝ) : ℝ := step1_milk M - ((step1_milk M) / 3) + 1 / 3

def step3_coffee (C : ℝ) : ℝ := step2_coffee C - ((step2_coffee C) / 2)
def step3_milk (M : ℝ) : ℝ := step2_milk M - ((step2_milk M) / 2) + 1 / 2

def final_coffee (C : ℝ) : ℝ := step3_coffee C
def final_milk (M : ℝ) : ℝ := step3_milk M

-- Assert that the amounts consumed are equal
theorem coffee_equals_milk :
  initial_coffee C - final_coffee C = initial_milk M - final_milk M :=
sorry

end coffee_equals_milk_l695_695752


namespace jacob_current_age_l695_695707

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l695_695707


namespace general_formula_sum_of_terms_l695_695772

-- Definitions of the initial conditions and sequence.
def seq (a : ℕ → ℕ) := a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- General formula proof
theorem general_formula (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
sorry

-- Sum of the first n terms proof
theorem sum_of_terms (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : seq a) (h_S : ∀ n, S n = (n - 1) * 2^(n + 1) + 2 - n^2 / 2 - n / 2) :
  (∀ n, S n = ∑ i in Finset.range n, (i + 1) * a (i + 1)) :=
sorry

end general_formula_sum_of_terms_l695_695772


namespace largest_n_sum_is_11_l695_695052

def single_digit_primes := [2, 3, 5, 7]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (d e : ℕ) : Prop :=
  d ≠ e ∧ d ∈ single_digit_primes ∧ e ∈ single_digit_primes ∧ is_prime (d^2 + e^2)

def n_value (d e : ℕ) : ℕ := d * e * (d^2 + e^2)

def sum_digits (n : ℕ) : ℕ := n.digits.sum

theorem largest_n_sum_is_11 :
  ∃ (d e : ℕ), valid_triplet d e ∧ sum_digits (n_value d e) = 11 := sorry

end largest_n_sum_is_11_l695_695052


namespace square_diff_l695_695355

-- Definitions and conditions from the problem
def three_times_sum_eq (a b : ℝ) : Prop := 3 * (a + b) = 18
def diff_eq (a b : ℝ) : Prop := a - b = 4

-- Goal to prove that a^2 - b^2 = 24 under the given conditions
theorem square_diff (a b : ℝ) (h₁ : three_times_sum_eq a b) (h₂ : diff_eq a b) : a^2 - b^2 = 24 :=
sorry

end square_diff_l695_695355


namespace delegates_without_name_badges_l695_695224

theorem delegates_without_name_badges
  (total_delegates : Nat)
  (pre_printed_badges : Nat)
  (hand_written_badges : Nat)
  (H1 : total_delegates = 36)
  (H2 : pre_printed_badges = 16)
  (H3 : hand_written_badges = (total_delegates - pre_printed_badges) / 2)
  (H4 : (total_delegates - pre_printed_badges - hand_written_badges = 10)) :
  total_delegates - pre_printed_badges - hand_written_badges = 10 :=
by
  have : total_delegates - pre_printed_badges = 36 - 16 := by rw [H1, H2]
  have : total_delegates - pre_printed_badges - hand_written_badges = 20 - hand_written_badges := by rw [this]
  have : hand_written_badges = 10 := by rw [H3, ←this]
  rw [this]
  sorry

end delegates_without_name_badges_l695_695224


namespace min_distance_on_minor_arc_l695_695879

variable {ω : Type*} [MetricSpace ω] [NormedAddCommGroup ω]
variable {O X A S T : ω}
variable (circle : Set ω) (chord : Set ω)

-- Definitions
def is_center (O : ω) (circle : Set ω) : Prop := ∀ (X : ω), X ∈ circle → dist O X = radius
def is_chord (S T : ω) (chord : Set ω) (circle : Set ω) : Prop := S ∈ chord ∧ T ∈ chord ∧ ∀ (P : ω), P ∈ chord → P ∈ circle
def minor_arc (S T : ω) (circle : Set ω) (arc : Set ω) : Prop := ∀ (X : ω), X ∈ arc → X ∈ circle

-- Problem Statement
theorem min_distance_on_minor_arc (circle ω : Type*) [MetricSpace ω] [NormedAddCommGroup ω]
  {O A S T X : ω} (h_circ : is_center O circle)
  (h_chord : is_chord S T chord circle) (h_arc : minor_arc S T circle arc)
  (h_dist : ∀ (P : ω), dist A P ≥ dist A X) : 
  X ∈ segment O A := sorry

end min_distance_on_minor_arc_l695_695879


namespace area_of_triangle_abc_l695_695394

theorem area_of_triangle_abc :
  ∀ (S₁ S₂ S₃ : ℕ), 
  S₁ = 6 → S₂ = 24 → S₃ = 54 →
  let S_ABC := S₁ + S₂ + S₃ + 2 * (Int.sqrt S₁ * Int.sqrt S₃ + Int.sqrt S₂ * Int.sqrt S₃ + Int.sqrt S₁ * Int.sqrt S₂) in
  S_ABC = 216 :=
by
  intros S₁ S₂ S₃ h₁ h₂ h₃
  let S_ABC := S₁ + S₂ + S₃ + 2 * (Int.sqrt S₁ * Int.sqrt S₃ + Int.sqrt S₃ * Int.sqrt S₂ + Int.sqrt S₁ * Int.sqrt S₂)
  have hS₁ : S₁ = 6 := by assumption
  have hS₂ : S₂ = 24 := by assumption
  have hS₃ : S₃ = 54 := by assumption
  sorry

end area_of_triangle_abc_l695_695394


namespace parameterize_solutions_l695_695490

theorem parameterize_solutions (a x y z : ℝ) (k l : ℤ) (h1 : |a + 1/a| = 2)
  (h2 : |a| = 1) (h3 : cos z = 0) :
  (∃ k l : ℤ, x = π/2 + k * π ∧ y = π/2 + k * π ∧ z = π/2 + l * π) :=
by
  sorry

end parameterize_solutions_l695_695490


namespace tri_connected_collection_count_l695_695660

def isCongruent (s1 s2 : Square) : Prop := sorry
def isVertex (P : Point) (s : Square) : Prop := sorry
def touchesThreeOtherSquares (s : Square) (collection : Set Square) : Prop := sorry

def triConnected (collection : Set Square) : Prop :=
  (∀ s ∈ collection, ∀ t ∈ collection, isCongruent s t) ∧
  (∀ s₁ s₂ ∈ collection, ∀ P, P ∈ s₁ ∩ s₂ → isVertex P s₁ ∧ isVertex P s₂) ∧
  (∀ s ∈ collection, touchesThreeOtherSquares s collection)

def even (n : ℕ) : Prop := n % 2 = 0

theorem tri_connected_collection_count :
  ∃ (n : ℕ), 2018 ≤ n ∧ n ≤ 3018 ∧ even n ∧ triConnected (set_of (λ s : Square, s ∈ finset.range n)) ∧ n = 501 := sorry

end tri_connected_collection_count_l695_695660


namespace constant_k_chord_circle_l695_695720

theorem constant_k_chord_circle :
  ∀ (P : ℝ × ℝ) (C : set (ℝ × ℝ)), 
    C = { p | p.1^2 + p.2^2 = 1 } → 
    P = (0, 1/2) → 
    ∀ (A B : ℝ × ℝ), 
      A ∈ C → B ∈ C → P.1 * A.1 + P.2 * A.2 = 1/2 → P.1 * B.1 + P.2 * B.2 = 1/2 → 
        (let PA := real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2),
             PB := real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
         in (1 / PA) + (1 / PB) = 4) :=
begin
  intros P C hC hP A B hA hB hPA hPB,
  -- proof goes here
  sorry
end

end constant_k_chord_circle_l695_695720


namespace smallest_number_of_students_l695_695214

theorem smallest_number_of_students (n : ℕ) 
  (five_students_score_100 : 5 * 100 ≤ total_score)
  (each_student_scores_at_least_60 : ∀ i, score(i) ≥ 60)
  (mean_score_is_76 : total_score / n = 76) 
  : n ≥ 13 :=
by
  sorry

def total_score := sum_scores n
def sum_scores (n : ℕ) : ℕ := sorry
def score (i : ℕ) : ℕ := sorry

end smallest_number_of_students_l695_695214


namespace line_through_unit_circle_points_l695_695303

noncomputable theory

open Complex

def on_unit_circle (z : ℂ) : Prop := abs z = 1

theorem line_through_unit_circle_points (a b z : ℂ) 
  (ha : on_unit_circle a) 
  (hb : on_unit_circle b) 
  (hz : ∃ t : ℝ, z = t * a + (1 - t) * b) : 
  z = a + b - a * b * conj z := 
sorry

end line_through_unit_circle_points_l695_695303


namespace age_proof_l695_695444

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l695_695444


namespace area_of_EFGH_l695_695611

-- Define the dimensions of the smaller rectangles
def smaller_rectangle_short_side : ℕ := 7
def smaller_rectangle_long_side : ℕ := 2 * smaller_rectangle_short_side

-- Define the configuration of rectangles
def width_EFGH : ℕ := 2 * smaller_rectangle_short_side
def length_EFGH : ℕ := smaller_rectangle_long_side

-- Prove that the area of rectangle EFGH is 196 square feet
theorem area_of_EFGH : width_EFGH * length_EFGH = 196 := by
  sorry

end area_of_EFGH_l695_695611


namespace problem_statement_l695_695826

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 14) : 6 * (x + 3) = 4.8 :=
sorry

end problem_statement_l695_695826


namespace prime_factors_of_M_l695_695816

theorem prime_factors_of_M :
  ∀ (M : ℕ), (log 2 (log 3 (log 5 (log 11 M)))) = 10 → (nat.prime_factors M).length = 1 :=
by
  intros M h
  sorry

end prime_factors_of_M_l695_695816


namespace min_xyz_product_l695_695433

theorem min_xyz_product 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (hxyz : x + y + z = 1)
  (h₁ : x ≤ 2 * (y + z)) 
  (h₂ : y ≤ 2 * (x + z)) 
  (h₃ : z ≤ 2 * (x + y)) : 
  xyz = ⅓² :=
sorry

end min_xyz_product_l695_695433


namespace simplify_sqrt3_7_pow6_l695_695473

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695473


namespace problem_statement_l695_695301

theorem problem_statement (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n : ℕ, a (n + 2) = a n)
  (h_b : ∀ n : ℕ, b (n + 1) - b n = a n)
  (h_repeat : ∀ k : ℕ, ∃ m : ℕ, (b (2 * m) / a m) = k)
  : b 1 = 2 :=
sorry

end problem_statement_l695_695301


namespace average_grade_entire_class_greater_than_4_l695_695494

-- Define the conditions
variables (students : ℕ) (grade : students → ℝ)
variable (watchesCartoons : students → Prop)
variable (watchesFootball : students → Prop)

-- Assume the total number of students is 7
def total_students : Prop := students = 7

-- Define average grade for cartoon watchers and football watchers
def avg_grade_cartoon_watchers : Prop :=
  let cartoon_grades := { g : ℝ | ∃ s, watchesCartoons s ∧ grade s = g } in
  ∑ g in cartoon_grades, g / (card cartoon_grades) < 4

def avg_grade_football_watchers : Prop :=
  let football_grades := { g : ℝ | ∃ s, watchesFootball s ∧ grade s = g } in
  ∑ g in football_grades, g / (card football_grades) < 4

-- Define the proof problem
theorem average_grade_entire_class_greater_than_4 :
  total_students ∧ avg_grade_cartoon_watchers ∧ avg_grade_football_watchers →
  ∃ g : ℝ, average (grade g) students > 4 :=
by {
  intros,
  sorry  -- Proof to be constructed.
}

end average_grade_entire_class_greater_than_4_l695_695494


namespace correct_option_d_l695_695790

variable (m t x1 x2 y1 y2 : ℝ)

theorem correct_option_d (h_m : m > 0)
  (h_y1 : y1 = m * x1^2 - 2 * m * x1 + 1)
  (h_y2 : y2 = m * x2^2 - 2 * m * x2 + 1)
  (h_x1 : t < x1 ∧ x1 < t + 1)
  (h_x2 : t + 2 < x2 ∧ x2 < t + 3)
  (h_t_geq1 : t ≥ 1) :
  y1 < y2 := sorry

end correct_option_d_l695_695790


namespace largest_integer_not_exceeding_50x_l695_695053

def sum_cos (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), real.cos (k * real.pi / 180)
def sum_sin (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), real.sin (k * real.pi / 180)

noncomputable def x : ℝ := sum_cos 60 / sum_sin 60

theorem largest_integer_not_exceeding_50x : ⌊50 * x⌋ = 186 := sorry

end largest_integer_not_exceeding_50x_l695_695053


namespace range_of_k_l695_695831

-- Define the curve and the line
def curve (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
def line (k x : ℝ) : ℝ := k * (x - 1) + 1

-- State the theorem about the range of k
theorem range_of_k {k : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ curve x1 = line k x1 ∧ curve x2 = line k x2) ↔ 0 < k ∧ k ≤ 1/2 :=
by
  sorry

end range_of_k_l695_695831


namespace solve_for_x_l695_695491

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 8 * x - 16 = 0) : x = 4 / 3 :=
by
  sorry

end solve_for_x_l695_695491


namespace sum_first_10_common_elements_l695_695745

-- Define the arithmetic progression (AP)
def arithmetic_prog (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression (GP)
def geometric_prog (k : ℕ) : ℕ := 10 * 2 ^ k

-- Find the sum of the first 10 elements present in both sequences
theorem sum_first_10_common_elements : 
  (Σ x in {x | ∃ n k, arithmetic_prog n = x ∧ geometric_prog k = x}, x).take 10 = 6990500 :=
sorry

end sum_first_10_common_elements_l695_695745


namespace abc_sum_l695_695890

noncomputable def x : ℝ := real.sqrt ((real.sqrt 65) / 2 + 5 / 2)

theorem abc_sum : ∃ (a b c : ℕ), 
  (∀ (x : ℝ), x = real.sqrt ((real.sqrt 65) / 2 + 5 / 2) →
   x ^ 100 = 3 * x ^ 98 + 18 * x ^ 96 + 13 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40) ∧ 
  a + b + c = 105 :=
begin
  sorry
end

end abc_sum_l695_695890


namespace standard_equation_of_ellipse_slope_of_line_AB_proof_l695_695777

-- Define the parameters and conditions given in the problem
variables (a b : ℝ) (e : ℝ) (C : ℝ × ℝ) (A B : ℝ × ℝ)

-- Conditions
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity : Prop := e = 2 / 3
def point_C_on_ellipse : Prop := C = (2, 5 / 3)
def standard_ellipse_conditions : Prop := a > b ∧ b > 0 ∧ ellipse 2 (5 / 3)
def eqn_from_eccentricity : Prop := sqrt (1 - b^2 / a^2) = e

-- Proof for the standard equation of the ellipse
theorem standard_equation_of_ellipse : standard_ellipse_conditions ∧ eqn_from_eccentricity →
  ellipse x y → (ellipse 2 (5 / 3) → (a^2 = 9 ∧ b^2 = 5)) → (x / a)^2 + (y / b)^2 = 1 :=
sorry

-- Parameters and conditions for slope problem
def vector_AB := (A.1 + a, A.2) = (1 / 2 * C.1, 1 / 2 * C.2)
def slope_of_line_AB (k : ℝ) : Prop := k = (5 * sqrt 3) / 3

-- Proof for the slope of line AB
theorem slope_of_line_AB_proof (m : ℝ) :
  standard_ellipse_conditions ∧ eqn_from_eccentricity ∧
  vector_AB ∧ (a = 3 ∧ b = 5) →
  slope_of_line_AB m :=
sorry

end standard_equation_of_ellipse_slope_of_line_AB_proof_l695_695777


namespace multiply_vars_l695_695235

variables {a b : ℝ}

theorem multiply_vars : -3 * a * b * 2 * a = -6 * a^2 * b := by
  sorry

end multiply_vars_l695_695235


namespace sandra_coffee_l695_695088

theorem sandra_coffee (S : ℕ) (H1 : 2 + S = 8) : S = 6 :=
by
  sorry

end sandra_coffee_l695_695088


namespace probability_sum_20_four_6faced_dice_l695_695828

theorem probability_sum_20_four_6faced_dice : 
  (∃ (d1 d2 d3 d4 : ℕ), 
    (d1 = 6 ∨ d2 = 6 ∨ d3 = 6 ∨ d4 = 6) ∧ 
    d1 + d2 + d3 + d4 = 20 ∧ 
    1 ≤ d1 ∧ d1 ≤ 6 ∧ 
    1 ≤ d2 ∧ d2 ≤ 6 ∧ 
    1 ≤ d3 ∧ d3 ≤ 6 ∧ 
    1 ≤ d4 ∧ d4 ≤ 6) → 
  (prob_condition_20 : ℚ,
   prob_condition_20 = 15 / 1296) := 
sorry

end probability_sum_20_four_6faced_dice_l695_695828


namespace find_common_ratio_l695_695769

-- We need to state that q is the common ratio of the geometric sequence

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first three terms for the geometric sequence
def S_3 (a : ℕ → ℝ) := a 0 + a 1 + a 2

-- State the Lean 4 declaration of the proof problem
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : (S_3 a) / (a 2) = 3) :
  q = 1 := 
sorry

end find_common_ratio_l695_695769


namespace sum_first_100_terms_of_sequence_l695_695860

theorem sum_first_100_terms_of_sequence (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n ≥ 2, a (n + 1) = a n - a (n - 1)) : 
  ∑ i in Finset.range 100, a (i + 1) = 5 := 
by 
  sorry

end sum_first_100_terms_of_sequence_l695_695860


namespace proof_statements_l695_695426

variables {α β : Type*} [Plane α] [Plane β]

def intersecting_lines_parallel (h1 : ∃ (a b : Line α), a ≠ b ∧ a ∥ b)
                                (h2 : ∃ (c d : Line β), c ≠ d ∧ c ∥ d) : Prop :=
Plane.parallel α β

def line_parallel_plane (l : Line) 
                        (h1 : ∃ (a : Line α), a ∥ l ∧ l ∉ α) : Prop :=
l ∥ α

theorem proof_statements (h1 : ∃ (a b : Line α), a ≠ b ∧ a ∥ b)
                         (h2 : ∃ (c d : Line β), c ≠ d ∧ c ∥ d)
                         (l : Line)
                         (h3 : ∃ (a : Line α), a ∥ l ∧ l ∉ α) :
  (intersecting_lines_parallel h1 h2) ∧ (line_parallel_plane l h3) :=
sorry

end proof_statements_l695_695426


namespace aquarium_goldfish_count_l695_695439

-- Definition of the conditions
def total_goldfish (G : ℕ) : Prop :=
  let allowed_to_take_home := G / 2 in
  let caught := (3 * allowed_to_take_home) / 5 in
  (allowed_to_take_home - caught = 20)

-- The theorem stating the problem and expected answer
theorem aquarium_goldfish_count : ∃ G : ℕ, total_goldfish G ∧ G = 100 :=
by {
  -- We would seek to show that the only G that makes the above proposition true is 100
  sorry
}

end aquarium_goldfish_count_l695_695439


namespace chord_length_polar_coordinate_l695_695857

theorem chord_length_polar_coordinate (theta : ℝ) (rho : ℝ) (h_theta : theta = π / 4) (h_rho : rho = 4 * sin theta) :
  chord_length h_theta h_rho = 2 * sqrt 2 :=
sorry

end chord_length_polar_coordinate_l695_695857


namespace find_triplets_l695_695721

theorem find_triplets (x y z : ℝ) : 
  (sqrt (3^x * (5^y + 7^z)) + sqrt (5^y * (7^z + 3^x)) + sqrt (7^z * (3^x + 5^y)) = sqrt 2 * (3^x + 5^y + 7^z)) →
  ∃ t : ℝ, x = t / real.log 3 ∧ y = t / real.log 5 ∧ z = t / real.log 7 :=
begin
  sorry
end

end find_triplets_l695_695721


namespace perpendicular_lines_l695_695702

def direction_vector_line1 (k : ℝ) : ℝ × ℝ × ℝ :=
  (2, -1, k)

def direction_vector_line2 (k : ℝ) : ℝ × ℝ × ℝ :=
  (k, 3, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem perpendicular_lines (k : ℝ) :
  dot_product (direction_vector_line1 k) (direction_vector_line2 k) = 0 ↔ k = 3 / 4 :=
by
  sorry

end perpendicular_lines_l695_695702


namespace xy_yz_zx_value_l695_695099

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 9) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + z * x + x^2 = 25) :
  x * y + y * z + z * x = 8 * Real.sqrt 3 :=
by sorry

end xy_yz_zx_value_l695_695099


namespace ratio_of_areas_l695_695152
-- Define the conditions and the ratio to be proven
theorem ratio_of_areas (t r : ℝ) (h : 3 * t = 2 * π * r) : 
  (π^2 / 18) = (π^2 * r^2 / 9) / (2 * r^2) :=
by 
  sorry

end ratio_of_areas_l695_695152


namespace thue_morse_min_square_subwords_l695_695993

-- Define what constitutes a binary word
def is_binary_word (w : list ℕ) : Prop :=
  ∀ x ∈ w, x = 0 ∨ x = 1

-- Define what constitutes a square subword
def is_square_subword (w : list ℕ) : Prop :=
  ∃ a b, a = b ∧ w = a ++ b

-- The Thue-Morse sequence of length 50
def thue_morse_50 : list ℕ :=
  [0, 1, 1, 0, 1, 0, 0, 1,
   0, 1, 1, 0, 0, 1, 0, 1,
   0, 0, 1, 1, 0, 1, 0, 0,
   1, 0, 1, 1, 0, 0, 1, 0,
   1, 0, 0, 1, 0, 1, 1, 0,
   1, 0, 0, 1, 0, 1, 1, 0]

-- Main proof statement
theorem thue_morse_min_square_subwords : 
  is_binary_word thue_morse_50 ∧
  (n : ℕ), n = 50 →
  ∀ w : list ℕ, length w = n →
    countp is_square_subword thue_morse_50 ≤ countp is_square_subword w :=
begin
  sorry
end

end thue_morse_min_square_subwords_l695_695993


namespace hyperbola_asymptotes_l695_695805

-- Defining the problem parameters
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (eccentricity : ℝ)
variables (hyperbola_eq : ∀ x y : ℝ, (y^2 / a^2) - (x^2 / b^2) = 1)
variables (ecc_eq : eccentricity = √3)

-- The theorem statement
theorem hyperbola_asymptotes :
  (x : ℝ) → ((y : ℝ) → (x ± √2 * y = 0)) :=
by
  sorry

end hyperbola_asymptotes_l695_695805


namespace court_salary_l695_695645

noncomputable section 
def is_magic_square (arr : List (List ℤ)) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → arr[i][k] + arr[j][k] + arr[k][k] = arr[0][k] + arr[1][k] + arr[2][k] 

def all_prime (l : List ℤ) : Prop := ∀ x, x ∈ l → Nat.Prime x

def valid_magic_square (e x y : ℤ) : Prop := 
  let a := e + x 
  let b := e - x - y
  let c := e + y
  let d := e - x + y
  let f := e + x - y
  let g := e - y
  let h := e + x + y
  let i := e - x
  is_magic_square [[a, b, c], [d, e, f], [g, h, i]]

def valid_primes_magic_square (e x y : ℤ) : Prop :=
  let lst_with1 := [e - x - y + 1, e - y + 1, e + x - y + 1, e - x + 1, e + 1, e + x + 1, e - x + y + 1, e + y + 1, e + x + y + 1]
  let lst_minus1 := [e - x - y - 1, e - y - 1, e + x - y - 1, e - x - 1, e - 1, e + x - 1, e - x + y - 1, e + y - 1, e + x + y - 1]
  valid_magic_square e x y ∧ all_prime lst_with1 ∧ all_prime lst_minus1

def court_mathematician_salary (salary : ℤ) : Prop := salary = 1490

theorem court_salary : ∃ e x y, valid_primes_magic_square e x y → court_mathematician_salary (9 * e) :=
by
 sorry

end court_salary_l695_695645


namespace largest_three_digit_starting_with_eight_divisible_by_digits_l695_695724

theorem largest_three_digit_starting_with_eight_divisible_by_digits : 
  ∃ n : ℕ, 800 ≤ n ∧ n < 900 ∧ 
           (∀ d ∈ {8, (n / 10) % 10, n % 10}, d ≠ 0 ∧ n % d = 0) ∧ 
           (∀ m : ℕ, 800 ≤ m ∧ m < 900 ∧
                    (∀ d ∈ {8, (m / 10) % 10, m % 10}, d ≠ 0 ∧ m % d = 0) → m ≤ n) := 
  ∃ n : ℕ, n = 864

end largest_three_digit_starting_with_eight_divisible_by_digits_l695_695724


namespace probability_neither_square_nor_cube_l695_695532

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695532


namespace van_helsing_earnings_l695_695955

theorem van_helsing_earnings (V W : ℕ) 
  (h1 : W = 4 * V) 
  (h2 : W = 8) :
  let E_v := 5 * (V / 2)
  let E_w := 10 * 8
  let E_total := E_v + E_w
  E_total = 85 :=
by
  sorry

end van_helsing_earnings_l695_695955


namespace pure_imaginary_implies_a_zero_l695_695822

theorem pure_imaginary_implies_a_zero (a : ℂ) (h : (a - complex.I) ^ 2 * complex.I ^ 3).im = 0 :
  a.re = 0 := sorry

end pure_imaginary_implies_a_zero_l695_695822


namespace phase_shift_sin_l695_695950

-- Defining the base function and the target function
def base_function (x : ℝ) : ℝ := Real.sin (4 * x)
def target_function (x : ℝ) : ℝ := Real.sin (4 * x - (Real.pi / 4))

-- Defining what it means to shift the base function by a phase shift to get the target function
def phase_shift (y₁ y₂ : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, y₁ (x + shift) = y₂ x

-- The problem statement to prove
theorem phase_shift_sin :
  phase_shift base_function target_function (Real.pi / 16) :=
sorry

end phase_shift_sin_l695_695950


namespace number_of_multisets_is_correct_l695_695504

noncomputable def number_of_multisets (b₀ b₈ : ℤ) (s : Fin 8 → ℤ) : ℕ :=
  -- Define the conditions in Lean
  if (b₈ ≠ 0 ∧ b₀ ≠ 0 ∧
    (∀ i, s i = 1 ∨ s i = -1)) -- Ensure roots are only 1 or -1
  then 9 -- The number of possible multisets
  else 0 -- Error case, not likely necessary in mathematical context

theorem number_of_multisets_is_correct (b₀ b₈ : ℤ) (s : Fin 8 → ℤ)
  (h1 : b₈ ≠ 0) (h2: b₀ ≠ 0)
  (h3 : ∀ i, s i = 1 ∨ s i = -1) :
  number_of_multisets b₀ b₈ s = 9 :=
begin
  -- Proof will go here
  sorry
end

end number_of_multisets_is_correct_l695_695504


namespace arithmetic_sequence_general_formula_and_sum_max_l695_695852

theorem arithmetic_sequence_general_formula_and_sum_max :
  ∀ (a : ℕ → ℤ), 
  (a 7 = -8) → (a 17 = -28) → 
  (∀ n, a n = -2 * n + 6) ∧ 
  (∀ S : ℕ → ℤ, (∀ n, S n = -n^2 + 5 * n) → ∀ n, S n ≤ 6) :=
by
  sorry

end arithmetic_sequence_general_formula_and_sum_max_l695_695852


namespace sum_roots_fraction_l695_695786

noncomputable theory

open Complex Polynomial

def poly : Polynomial ℂ := 
  Polynomial.C 6 + Polynomial.C 5 * Polynomial.X + Polynomial.C 4 * (Polynomial.X ^ 2) 
  + Polynomial.C 3 * (Polynomial.X ^ 3) + Polynomial.C 2 * (Polynomial.X ^ 4) + Polynomial.C 1 * (Polynomial.X ^ 5)
  
-- Statement of the problem
theorem sum_roots_fraction :
  let zs := poly.roots.to_finset in
  zs.card = 5 →
  (∑ z in zs, z / (z^2 + 1)) = 4 / 17 :=
by {
  intro zs hzs_card,
  have : ∀ z ∈ zs, z^5 + 2*z^4 + 3*z^3 + 4*z^2 + 5*z + 6 = 0,
  { sorry }, -- Showing the roots condition
  sorry
}

end sum_roots_fraction_l695_695786


namespace max_handshakes_l695_695102

theorem max_handshakes (n : ℕ) (h : n = 20) :
  ∀ G : SimpleGraph (Fin n), 
    ¬(∃ (v₁ v₂ v₃ : G.vertex), v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧ G.Adj v₁ v₂ ∧ G.Adj v₂ v₃ ∧ G.Adj v₁ v₃) →
    G.edge_finset.card ≤ 100 :=
by
  intro G hG
  have key : (∀ G : SimpleGraph (Fin n), 
    ¬(∃ (v₁ v₂ v₃ : G.vertex), v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧ G.Adj v₁ v₂ ∧ G.Adj v₂ v₃ ∧ G.Adj v₁ v₃) →
    G.edge_finset.card ≤ ⌊ (20 * 20) / 4 ⌋),
  from by sorry,
  exact key

#check max_handshakes

end max_handshakes_l695_695102


namespace option_A_option_C_option_D_l695_695927

def D (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem option_A : ∀ x : ℝ, D (-x) = D x :=
by
  sorry

theorem option_C : ∃! x : ℝ, D x - x^3 = 0 :=
by
  sorry

theorem option_D : ∀ x : ℝ, D (D x) = 1 :=
by
  sorry

end option_A_option_C_option_D_l695_695927


namespace factor_polynomial_l695_695644

theorem factor_polynomial (x y : ℤ) : 
  x^7 + x^6 * y + x^5 * y^2 + x^4 * y^3 + x^3 * y^4 + x^2 * y^5 + x * y^6 + y^7 = 
  (x + y) * (x^2 + y^2) * (x^4 + y^4) :=
begin
  sorry
end

end factor_polynomial_l695_695644


namespace sum_first_10_common_elements_l695_695746

-- Define the arithmetic progression (AP)
def arithmetic_prog (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression (GP)
def geometric_prog (k : ℕ) : ℕ := 10 * 2 ^ k

-- Find the sum of the first 10 elements present in both sequences
theorem sum_first_10_common_elements : 
  (Σ x in {x | ∃ n k, arithmetic_prog n = x ∧ geometric_prog k = x}, x).take 10 = 6990500 :=
sorry

end sum_first_10_common_elements_l695_695746


namespace convert_to_polar_coordinates_l695_695251

open Real

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * π - arctan (abs y / abs x) else arctan (abs y / abs x)
  (r, θ)

theorem convert_to_polar_coordinates : 
  polar_coordinates 3 (-3) = (3 * sqrt 2, 7 * π / 4) :=
by
  sorry

end convert_to_polar_coordinates_l695_695251


namespace exceptional_points_on_same_circle_l695_695893

noncomputable theory

-- Definitions for the problem
variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]
variables (Γ₁ Γ₂ Γ₃ : set ℝ)
variables (P : ℝ) (A₁ B₁ A₂ B₂ A₃ B₃ : ℝ)
variables (tangent_to_circle : ℝ → set ℝ → Prop)
variables (non_overlapping : set ℝ → set ℝ → Prop)
variables (mutually_external : set ℝ → set ℝ → Prop)
variables (exceptional_point : ℝ → Prop)
variables (concurrent : ℝ → ℝ → ℝ → Prop)

-- Assumptions
axiom three_circles : (non_overlapping Γ₁ Γ₂) ∧ (non_overlapping Γ₂ Γ₃) ∧ (non_overlapping Γ₃ Γ₁) ∧ 
                      (mutually_external Γ₁ Γ₂) ∧ (mutually_external Γ₂ Γ₃) ∧ (mutually_external Γ₃ Γ₁)

axiom construction : ∀ (P : ℝ), P ∉ Γ₁ ∧ P ∉ Γ₂ ∧ P ∉ Γ₃ → 
                      (tangent_to_circle P Γ₁) ∧ (tangent_to_circle P Γ₂) ∧ (tangent_to_circle P Γ₃)

axiom exceptional_concurrent : ∀ (P : ℝ), exceptional_point P → 
                            (concurrent (A₁B₁) (A₂B₂) (A₃B₃))

-- Definition of a circle
def circle (center : ℝ) (radius : ℝ) := {x : ℝ | dist x center = radius}

-- The theorem to prove
theorem exceptional_points_on_same_circle : ∀ P, exceptional_point P → 
  ∃ O r, circle O r P :=
begin
  sorry
end

end exceptional_points_on_same_circle_l695_695893


namespace value_of_x4_plus_1_div_x4_l695_695373

theorem value_of_x4_plus_1_div_x4 (x : ℝ) (hx : x^2 + 1 / x^2 = 2) : x^4 + 1 / x^4 = 2 := 
sorry

end value_of_x4_plus_1_div_x4_l695_695373


namespace five_points_no_three_collinear_has_one_additional_intersection_l695_695011

theorem five_points_no_three_collinear_has_one_additional_intersection (P : Finset (EuclideanSpace ℝ (Fin 2))) 
    (h₁ : P.card = 5) 
    (h₂ : ∀ (A B C : EuclideanSpace ℝ (Fin 2)), A ∈ P → B ∈ P → C ∈ P → A ≠ B → B ≠ C → A ≠ C → ¬Collinear ℝ ({A, B, C} : Set (EuclideanSpace ℝ (Fin 2)))) : 
  ∃ (Q : Set (EuclideanSpace ℝ (Fin 2))), Q.card = 5 ∧ (∀ p, p ∈ Q → ∃ A B, A ∈ P ∧ B ∈ P ∧ A ≠ B ∧ IsIntersectionPoint A B p) :=
sorry

end five_points_no_three_collinear_has_one_additional_intersection_l695_695011


namespace circle_area_from_diameter_points_l695_695454

theorem circle_area_from_diameter_points (C D : ℝ × ℝ)
    (hC : C = (-2, 3)) (hD : D = (4, -1)) :
    ∃ (A : ℝ), A = 13 * Real.pi :=
by
  let distance := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  have diameter : distance = Real.sqrt (6^2 + (-4)^2) := sorry -- this follows from the coordinates
  have radius : distance / 2 = Real.sqrt 13 := sorry -- half of the diameter
  exact ⟨13 * Real.pi, sorry⟩ -- area of the circle

end circle_area_from_diameter_points_l695_695454


namespace simplify_sqrt3_7_pow6_l695_695472

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695472


namespace cross_section_area_of_triangular_prism_l695_695844

theorem cross_section_area_of_triangular_prism
  (a : ℝ) (α : ℝ) (is_acute : 0 < α ∧ α < π / 2) :
  ∃ area : ℝ, area = (a^2 * Real.sqrt 3) / (12 * Real.cos α) :=
by
  use (a^2 * Real.sqrt 3) / (12 * Real.cos α)
  -- Proof steps would go here
  sorry

end cross_section_area_of_triangular_prism_l695_695844


namespace unique_zero_function_l695_695260

theorem unique_zero_function {f : ℕ → ℕ} (h : ∀ m n, f (m + f n) = f m + f n + f (n + 1)) : ∀ n, f n = 0 :=
by {
  sorry
}

end unique_zero_function_l695_695260


namespace probability_non_square_non_cube_l695_695585

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695585


namespace sophia_pages_difference_l695_695097

theorem sophia_pages_difference (total_pages : ℕ) (f_fraction : ℚ) (l_fraction : ℚ) 
  (finished_pages : ℕ) (left_pages : ℕ) :
  f_fraction = 2/3 ∧ 
  l_fraction = 1/3 ∧
  total_pages = 270 ∧
  finished_pages = f_fraction * total_pages ∧
  left_pages = l_fraction * total_pages
  →
  finished_pages - left_pages = 90 :=
by
  intro h
  sorry

end sophia_pages_difference_l695_695097


namespace exists_good_point_l695_695989

theorem exists_good_point (labels : Fin 1991 → Int)
  (h_label_range : ∀ i, labels i = 1 ∨ labels i = -1)
  (h_neg_count : (Finset.univ.filter (λ i : Fin 1991, labels i = -1)).card < 664) :
  ∃ i : Fin 1991, (∀ j, ∑ k in Finset.range j, labels ((i + k) % 1991) > 0) :=
by sorry

end exists_good_point_l695_695989


namespace constant_term_in_binomial_expansion_l695_695023

theorem constant_term_in_binomial_expansion : 
  ∀ (x : ℝ) (n : ℕ), (∑ i in Finset.range (n + 1), nat.choose n i) = 64 → 
  (let r := 2 in (nat.choose 6 r) * (3^r) = 135) :=
by
  intros x n sum_eq
  sorry

end constant_term_in_binomial_expansion_l695_695023


namespace snooker_tournament_vip_vs_general_l695_695638

theorem snooker_tournament_vip_vs_general :
  (∃ (V G : ℕ), V + G = 320 ∧ 40 * V + 15 * G = 7500 ∧ G - V = 104) :=
by
  exists 108, 212
  split
  { sorry }
  split
  { sorry }
  { sorry }

end snooker_tournament_vip_vs_general_l695_695638


namespace generalTermAndSumCondition_l695_695318
-- Import required Lean libraries

noncomputable def arithmeticSequence {a b : ℕ} 
  (h1 : 2 * a + 3 * b = 8) (h2 : a + 4 * b = 3 * a + 3 * b) : ℕ → ℤ :=
λ n, (2 * n - 1 : ℤ)

theorem generalTermAndSumCondition 
  {a b : ℕ} {an : ℕ → ℤ} 
  (h1: 2 * a + 3 * b = 8) 
  (h2: a + 4 * b = 3 * a + 3 * b)
  (h3: an = λ n, 2 * n - 1) 
  (S : ℕ → ℚ)
  (h4 : (S n = (∑ i in Finset.range n, (λ i, 1 / (coe (2 * i + 1 : ℕ) : ℚ) - (1 / (2 * (i + 1 : ℕ) + 1))))) 
    (Sn := λ n, (S n : ℚ) = 1 - 1 / (2 * n + 1))
  : 
  S 1009 > 2016 / 2017 :=
sorry

end generalTermAndSumCondition_l695_695318


namespace sum_first_10_common_terms_eq_6990500_l695_695742

-- Define the arithmetic progression
def is_arithmetic_term (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression
def is_geometric_term (k : ℕ) : ℕ := 10 * 2^k

-- Predicate to check if a term is common in both progressions
def is_common_term (m : ℕ) : Prop :=
  ∃ n k, m = is_arithmetic_term n ∧ m = is_geometric_term k ∧ k % 2 = 1

-- Sum of the first 10 common terms
def sum_of_first_10_common_terms : ℕ :=
  let common_terms := [20, 80, 320, 1280, 5120, 20480, 81920, 327680, 1310720, 5242880] in
  common_terms.sum

-- Main theorem statement
theorem sum_first_10_common_terms_eq_6990500 :
  sum_of_first_10_common_terms = 6990500 :=
by
  sorry

end sum_first_10_common_terms_eq_6990500_l695_695742


namespace magnitude_of_complex_l695_695367

theorem magnitude_of_complex (z : ℂ) (hz : z = (2 - complex.i) / complex.i) : complex.abs z = real.sqrt 5 := by
  sorry

end magnitude_of_complex_l695_695367


namespace range_of_t_l695_695704

noncomputable def f : ℝ → ℝ
| x => if x ∈ [-2, 0] then x^2 + 2*x
       else if x ∈ [0, 2] then 2 * (x-2)^2 + 4 * (x-2)
       else if x ∈ [2, 4] then 4 * (x-4)^2 + 8 * (x-4)
       else 0 -- just a placeholder for other values

theorem range_of_t (t : ℝ) : 
  (∀ x ∈ set.Icc 2 4, f x ≥ 2 * Real.log (t + 1) / Real.log 2) → 
  -1 < t ∧ t ≤ -3/4 :=
by
  sorry

end range_of_t_l695_695704


namespace probability_neither_square_nor_cube_l695_695544

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695544


namespace sum_of_roots_l695_695751

theorem sum_of_roots : 
  let a := 1
  let b := 2001
  let c := -2002
  ∀ x y: ℝ, (x^2 + b*x + c = 0) ∧ (y^2 + b*y + c = 0) -> (x + y = -b) :=
by
  sorry

end sum_of_roots_l695_695751


namespace difference_of_squares_divisible_by_9_l695_695089

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : 
  9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) :=
by
  sorry

end difference_of_squares_divisible_by_9_l695_695089


namespace twelve_circles_possible_l695_695028

theorem twelve_circles_possible :
  ∃ G : SimpleGraph (Fin 12), ∀ v : Fin 12, G.degree v = 5 :=
sorry

end twelve_circles_possible_l695_695028


namespace brad_net_profit_l695_695227

theorem brad_net_profit 
  (gallons : ℕ) (glasses_per_gallon : ℕ) (cost_per_gallon : ℝ) (price_per_glass : ℝ) 
  (gallons_made : ℕ) (glasses_drank : ℕ) (glasses_unsold : ℕ) :
  gallons = 16 →
  cost_per_gallon = 3.50 →
  price_per_glass = 1.00 →
  gallons_made = 2 →
  glasses_drank = 5 →
  glasses_unsold = 6 →
  (price_per_glass * (gallons_made * glasses_per_gallon - glasses_drank - glasses_unsold) - 
   gallons_made * cost_per_gallon) = 14 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end brad_net_profit_l695_695227


namespace polar_to_rectangular_coordinates_l695_695250

-- Define the given conditions in Lean 4
def r : ℝ := 4
def theta : ℝ := Real.pi / 3

-- Define the conversion formulas
def x : ℝ := r * Real.cos theta
def y : ℝ := r * Real.sin theta

-- State the proof problem
theorem polar_to_rectangular_coordinates : (x = 2) ∧ (y = 2 * Real.sqrt 3) := by
  -- Sorry is used to indicate the proof is omitted
  sorry

end polar_to_rectangular_coordinates_l695_695250


namespace john_spending_at_supermarket_l695_695225

variable (X : ℝ)

theorem john_spending_at_supermarket (fresh_fruits_vegetables : X / 5)
                                      (meat_products : X / 3)
                                      (bakery_products : X / 10)
                                      (dairy_products : X / 6)
                                      (candy_magazine : ℝ)
                                      (remaining_amount : candy_magazine = 13)
                                      (magazine_cost : ℝ)
                                      (magazine_cost_def : magazine_cost = 4) :
                                      (X = 13 * 5) :=
by
  sorry

end john_spending_at_supermarket_l695_695225


namespace ajax_store_price_l695_695087

theorem ajax_store_price (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ)
    (h_original: original_price = 180)
    (h_first_discount : first_discount_rate = 0.5)
    (h_second_discount : second_discount_rate = 0.2) :
    let first_discount_price := original_price * (1 - first_discount_rate)
    let saturday_price := first_discount_price * (1 - second_discount_rate)
    saturday_price = 72 :=
by
    sorry

end ajax_store_price_l695_695087


namespace odd_square_diff_div_by_eight_l695_695456

theorem odd_square_diff_div_by_eight (n p : ℤ) : 
  (2 * n + 1)^2 - (2 * p + 1)^2 % 8 = 0 := 
by 
-- Here we declare the start of the proof.
  sorry

end odd_square_diff_div_by_eight_l695_695456


namespace wine_division_l695_695620

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end wine_division_l695_695620


namespace emma_calculation_l695_695621

theorem emma_calculation : (37^2 = 38^2 - 75) :=
by
  have a := 38
  calc
    (a - 1)^2 = a^2 - 2 * a + 1      : by rw pow_two_sub1
    ...       = 38^2 - 2 * 38 + 1    : by rw a
    ...       = 38^2 - 75            : by norm_num
  sorry

end emma_calculation_l695_695621


namespace unique_root_l695_695267

variable {R : Type*} [LinearOrderedField R]
variable f : R → R

theorem unique_root (h_increasing : ∀ a b : R, a < b → f(a) < f(b)) :
  ∀ x₁ x₂ : R, f x₁ = 0 → f x₂ = 0 → x₁ = x₂ :=
by
  intros x₁ x₂ h1 h2
  -- Proof would proceed here
  sorry

end unique_root_l695_695267


namespace circle_B_area_l695_695239

theorem circle_B_area
  (r R : ℝ)
  (h1 : ∀ (x : ℝ), x = 5)  -- derived from r = 5
  (h2 : R = 2 * r)
  (h3 : 25 * Real.pi = Real.pi * r^2)
  (h4 : R = 10)  -- derived from diameter relation
  : ∃ A_B : ℝ, A_B = 100 * Real.pi :=
by
  sorry

end circle_B_area_l695_695239


namespace max_3cosx_4sinx_l695_695273

theorem max_3cosx_4sinx (x : ℝ) : (3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧ (∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5) :=
  sorry

end max_3cosx_4sinx_l695_695273


namespace smaller_circle_x_coordinate_l695_695952

theorem smaller_circle_x_coordinate (h : ℝ) 
  (P : ℝ × ℝ) (S : ℝ × ℝ)
  (H1 : P = (9, 12))
  (H2 : S = (h, 0))
  (r_large : ℝ)
  (r_small : ℝ)
  (H3 : r_large = 15)
  (H4 : r_small = 10) :
  S.1 = 10 ∨ S.1 = -10 := 
sorry

end smaller_circle_x_coordinate_l695_695952


namespace area_quadrilateral_eq_l695_695337

noncomputable def area_of_quadrilateral (p : ℝ) (theta : ℝ) (h_p : 0 < p) (h_theta : 0 < theta ∧ theta < Real.pi) : ℝ :=
  2 * p^2 * (1 + Real.cot theta ^ 2) ^ (3 / 2)

theorem area_quadrilateral_eq (p : ℝ) (theta : ℝ) (h_p : 0 < p) (h_theta : 0 < theta ∧ theta < Real.pi) :
  let area := area_of_quadrilateral p theta h_p h_theta in
    area = 2 * p^2 * (1 + Real.cot theta ^ 2) ^ (3 / 2) := 
by
  sorry

end area_quadrilateral_eq_l695_695337


namespace determine_b_l695_695326

-- Define the function
def f (x : ℝ) (b : ℝ) : ℝ := x^2 + 2*(b-1)*x + 2

-- Statements about the function and the interval
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y ≤ f x

-- The set (-∞, 4]
def interval : Set ℝ := {x | x ≤ 4}

-- Problem statement
theorem determine_b (b : ℝ) :
  is_decreasing_on (λ x, f x b) interval → b ≤ -3 :=
by { sorry }

end determine_b_l695_695326


namespace probability_non_square_non_cube_l695_695592

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695592


namespace number_of_hexagons_l695_695252

-- Define the recursive relation for S_m
def S : ℕ → ℕ 
| 1       := 1
| (m+1) := 6 * m + S m

-- State the main theorem to prove
theorem number_of_hexagons (m : ℕ) : S (m+1) = 3 * (m+1)^2 - 3 * (m+1) + 1 :=
by sorry

end number_of_hexagons_l695_695252


namespace exists_perfect_symmetry_day_in_century_l695_695447

def is_symmetrical (date : String) : Prop :=
  date = date.reverse

def is_valid_date (date : String) : Prop :=
  -- Assuming a basic check that length is 8 and it represents a valid date
  date.length = 8

theorem exists_perfect_symmetry_day_in_century :
  ∃ date : String, is_valid_date date ∧ date.toInt ≥ 20000101 ∧ date.toInt ≤ 20991231 ∧ is_symmetrical date :=
sorry

end exists_perfect_symmetry_day_in_century_l695_695447


namespace complement_of_A_in_U_l695_695812

def U : Set ℤ := {x | -2 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {x | ∃ n : ℕ, (x = 2 * n ∧ n ≤ 3)}

theorem complement_of_A_in_U : (U \ A) = {-2, -1, 1, 3, 5} :=
by
  sorry

end complement_of_A_in_U_l695_695812


namespace part1_part2_l695_695195

-- Part 1: Proving the solutions for (x-1)^2 = 49
theorem part1 (x : ℝ) (h : (x - 1)^2 = 49) : x = 8 ∨ x = -6 :=
sorry

-- Part 2: Proving the time for the object to reach the ground
theorem part2 (t : ℝ) (h : 4.9 * t^2 = 10) : t = 10 / 7 :=
sorry

end part1_part2_l695_695195


namespace isosceles_trapezoid_perimeter_l695_695245

-- Definitions specific to the problem
def isosceles_trapezoid (A B C D : Type) :=
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (Parallel : AB = 3 * CD) (AB_v : AB > 0) 
  (BC_v : BC > 0) (CD_v : CD > 0) (DA_v : DA > 0) (Eq1 : AB = 60) (Eq2 : BC = 52)
  (Eq3 : CD = 20) (Eq4 : DA = 52)

-- Define the perimeter computation
def trapezoid_perimeter (AB BC CD DA : ℝ) : ℝ := AB + BC + CD + DA

-- The final proof problem statement
theorem isosceles_trapezoid_perimeter :
  ∀ (A B C D : Type), 
    ∀ (AB BC CD DA : ℝ),
      isosceles_trapezoid A B C D AB BC CD DA ->
      trapezoid_perimeter AB BC CD DA = 184 :=
by
  intros A B C D AB BC CD DA h
  dsimp [isosceles_trapezoid, trapezoid_perimeter] at h
  simp [h.Eq1, h.Eq2, h.Eq3, h.Eq4]
  done

end isosceles_trapezoid_perimeter_l695_695245


namespace compute_expression_l695_695240

theorem compute_expression : (Real.pi - 3.14)^0 + 3^(-1) = 4 / 3 := by
  sorry

end compute_expression_l695_695240


namespace find_a_negative_l695_695881

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then -x^2 else 3 * x - 62

theorem find_a_negative :
  ∃ a : ℝ, a < 0 ∧ h (h (h 15)) = h (h (h a)) ∧ a = -(83521^(1/4)) :=
begin
  sorry
end

end find_a_negative_l695_695881


namespace john_pant_cost_l695_695414

def compute_pant_cost (tshirt_cost : ℕ) (tshirt_count : ℕ) (total_spent : ℕ) : ℕ :=
  total_spent - (tshirt_cost * tshirt_count)

theorem john_pant_cost :
  compute_pant_cost 20 3 110 = 50 :=
by
  unfold compute_pant_cost
  norm_num -- Verifies the computation directly
  sorry -- Proof here (omitted)

end john_pant_cost_l695_695414


namespace determine_d_l695_695597

noncomputable def roots_in_form (d : ℚ) : Prop :=
  ∃ (x : ℚ), x^2 - 3*x + d = 0 ∧ (x = (3 + (√d)) / 2 ∨ x = (3 - (√d)) / 2)

theorem determine_d :
  ∃ (d : ℚ), d = 9 / 5 ∧ roots_in_form d :=
sorry

end determine_d_l695_695597


namespace euler_formula_convex_polyhedron_l695_695242

variable (S A F : ℕ)
variable (convex : Prop)

/-- Euler's formula for convex polyhedra -/
theorem euler_formula_convex_polyhedron (h_convex : convex) (h_vertices : S) (h_edges : A) (h_faces : F) : 
  S + F = A + 2 :=
sorry

end euler_formula_convex_polyhedron_l695_695242


namespace smallest_angle_in_cyclic_quad_l695_695841

theorem smallest_angle_in_cyclic_quad :
  ∀ (a d : ℝ), let largest := 140 in
    a + 3 * d = largest → 
    (2 * a + 3 * d = 180) → 
    a = 40 :=
by
  intros a d largest h1 h2
  sorry

end smallest_angle_in_cyclic_quad_l695_695841


namespace sequences_converge_to_points_l695_695169

variable (a : ℝ)

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, a⟩
def B : Point := ⟨a, 0⟩
def C : Point := ⟨0, 0⟩

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def P_seq : ℕ → Point
| 0     := A a
| 1     := midpoint (B a) (C a)
| (2*k) := midpoint (A a) (P_seq (2*k-1))
| (2*k+1) := midpoint (B a) (P_seq (2*k))

-- Define the limit points
def lim_P2k := ⟨a / 3, 2 * a / 3⟩
def lim_P2k1 := ⟨2 * a / 3, a / 3⟩

theorem sequences_converge_to_points :
  (∀ k, P_seq (2*k) = lim_P2k) ∧ (∀ k, P_seq (2*k+1) = lim_P2k1) :=
  sorry

end sequences_converge_to_points_l695_695169


namespace probability_no_distinct_positive_real_roots_l695_695222

theorem probability_no_distinct_positive_real_roots :
  let possible_b := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let total_pairs := 121
  let valid_b_c_pairs :=
    [(-5, 1), (-5, 2), (-5, 3), (-5, 4), (-5, 5), 
     (-4, 1), (-4, 2), (-4, 3), 
     (-3, 1), (-3, 2)]
  probable := 121 - (List.length valid_b_c_pairs)
  let p := probable / total_pairs
  p = 111 / 121 := by
  let possible_pairs := List.product possible_b possible_b
  have num_pairs : List.length possible_pairs = total_pairs := by
    -- proof for the total number of pairs (b, c)
    sorry
  have num_valid : List.length valid_b_c_pairs = 10 := by
    -- proof for the number of valid pairs
    sorry
  let num_invalid := total_pairs - 10
  have h: num_invalid = 111 := by
    -- proof for the number of invalid pairs
    sorry
  have prob : p = 111 / 121 := by
    -- expression for the probability
    sorry
  -- conclusion
  exact prob

end probability_no_distinct_positive_real_roots_l695_695222


namespace sequence_formula_l695_695343

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = 2 * n

theorem sequence_formula (a : ℕ → ℕ) (h : sequence a) : ∀ n, a n = n^2 + n := 
  sorry

end sequence_formula_l695_695343


namespace probability_within_interval_l695_695787

-- Definitions based on the problem conditions
def mu : ℝ := 0
def sigma : ℝ := 2
def prob_interval (a b : ℝ) : ℝ :=
  if (a, b) = (-2, 2) then 0.6826 else
  if (a, b) = (-4, 4) then 0.9544 else 0

-- Theorem stating the computed probability for the interval [-4, -2]
theorem probability_within_interval : prob_interval (-4, 4) = 0.9544 →
                                     prob_interval (-2, 2) = 0.6826 →
                                     (prob_interval (-4, 4) - prob_interval (-2, 2)) / 2 = 0.1359 :=
by
  intros h1 h2
  have h : (0.9544 - 0.6826) / 2 = 0.1359 := sorry
  exact h

end probability_within_interval_l695_695787


namespace speed_ratio_l695_695616

variable (v1 v2 : ℝ) -- Speeds of A and B respectively
variable (dA dB : ℝ) -- Distances to destinations A and B respectively

-- Conditions:
-- 1. Both reach their destinations in 1 hour
def condition_1 : Prop := dA = v1 ∧ dB = v2

-- 2. When they swap destinations, A takes 35 minutes more to reach B's destination
def condition_2 : Prop := dB / v1 = dA / v2 + 35 / 60

-- Given these conditions, prove that the ratio of v1 to v2 is 3
theorem speed_ratio (h1 : condition_1 v1 v2 dA dB) (h2 : condition_2 v1 v2 dA dB) : v1 = 3 * v2 :=
sorry

end speed_ratio_l695_695616


namespace find_b_value_l695_695062

theorem find_b_value (b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + complex.i) (h2 : z2 = 2 + b * complex.i) (h3 : ∃ r : ℝ, z1 * z2 = r) : b = -2 :=
sorry

end find_b_value_l695_695062


namespace find_a_l695_695697

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a : ∃ a : ℝ, g a = 0 ∧ a = 7 / 5 :=
by {
  existsi (7 / 5),
  split,
  { show g (7 / 5) = 0, sorry },
  { refl }
}

end find_a_l695_695697


namespace boat_stream_speed_l695_695178

/-- A boat can travel with a speed of 22 km/hr in still water. 
If the speed of the stream is unknown, the boat takes 7 hours 
to go 189 km downstream. What is the speed of the stream?
-/
theorem boat_stream_speed (v : ℝ) : (22 + v) * 7 = 189 → v = 5 :=
by
  intro h
  sorry

end boat_stream_speed_l695_695178


namespace simplified_expression_correct_l695_695091

-- Define the necessary powers and intermediate values
noncomputable def power_values : ℕ × ℕ × ℕ × ℕ :=
  (7^5, 2^7, 2^3, (-2)^3)

-- Compute the intermediate subtraction and subsequent exponentiation
noncomputable def intermediate_result : ℕ :=
  (2^3 - (-2)^3)

noncomputable def exponentiated_result : ℕ :=
  intermediate_result ^ 8

-- Compute the sum of the original powers and final multiplication
noncomputable def sum_of_powers : ℕ :=
  (7^5 + 2^7)

-- Verify the final result
noncomputable def final_result : ℕ :=
  sum_of_powers * exponentiated_result

theorem simplified_expression_correct : final_result = 72778137514496 := by
  -- proof here
  sorry

end simplified_expression_correct_l695_695091


namespace distance_ahead_when_crossing_finish_l695_695237

noncomputable def CatrinaSpeed : ℝ := 100 / 10
noncomputable def SedraSpeed : ℝ := 400 / 44
noncomputable def raceDistance : ℝ := 1000

theorem distance_ahead_when_crossing_finish :
  let timeCatrina := raceDistance / CatrinaSpeed
  let distanceSedraInCatrinaTime := SedraSpeed * timeCatrina
  (raceDistance - distanceSedraInCatrinaTime).round = 91 :=
by
  let timeCatrina := raceDistance / CatrinaSpeed
  let distanceSedraInCatrinaTime := SedraSpeed * timeCatrina
  have h: (raceDistance - distanceSedraInCatrinaTime).round = 91 := sorry
  exact h

end distance_ahead_when_crossing_finish_l695_695237


namespace circle_passing_through_points_l695_695422

noncomputable def parabola (x: ℝ) (a b: ℝ) : ℝ :=
  x^2 + a * x + b

theorem circle_passing_through_points (a b α β k: ℝ) :
  parabola 0 a b = b ∧ parabola α a b = 0 ∧ parabola β a b = 0 ∧
  ((0 - (α + β) / 2)^2 + (1 - k)^2 = ((α + β) / 2)^2 + (k - b)^2) →
  b = 1 :=
by
  sorry

end circle_passing_through_points_l695_695422


namespace LeapDay2044_l695_695035

-- Define a type for the days of the week
inductive DayOfWeek : Type
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define dates
constant Feb29_2024 : DayOfWeek
constant Feb29_2044 : DayOfWeek

-- Given condition
axiom H1 : Feb29_2024 = Thursday

-- The theorem to prove
theorem LeapDay2044 : Feb29_2044 = Monday :=
by sorry

end LeapDay2044_l695_695035


namespace parabola_focus_symmetry_l695_695508

theorem parabola_focus_symmetry :
  let focus_orig := (-2, 0)
  ∃ a b : ℝ, ((b / 2 = (-2 + a) / 2 - 1) ∧ (b / (a + 2) = -1)) → (a, b) = (1, -3) :=
begin
  intro focus_orig,
  use (1: ℝ),
  use (-3: ℝ),
  intro h,
  split,
  { exact real.eq_iff (h.1) (by linarith using h.2) },
  { exact h },
end

end parabola_focus_symmetry_l695_695508


namespace betty_beads_l695_695685

theorem betty_beads (red blue green : ℕ) (h1 : red = 60) (h2 : 7.5 * blue = 3.5 * red) (h3 : 7.5 * green = 2 * red) :
  blue + green = 44 :=
by
  sorry

end betty_beads_l695_695685


namespace eccentricity_of_ellipse_lambda_mu_squared_sum_one_l695_695778

open Real

-- Definitions and conditions
variables (a b c : ℝ) (h : a > b ∧ b > 0)
variable (F : ℝ × ℝ) -- focus F coordinates
variable (O : ℝ × ℝ := (0, 0)) -- center at the origin
variable (A B : ℝ × ℝ) -- intersection points

-- Given conditions
noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
noncomputable def line_eq (x y : ℝ) : Prop := y = x - (F.1)
def collinear (u v : ℝ × ℝ) (a : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * a.1, k * a.2)

-- Given information from the problem statement
variable (c_comp : c = sqrt (a^2 - b^2))
variable (a_collinear : collinear (A.1 + B.1, A.2 + B.2) (3, -1))

-- Main theorem statement
theorem eccentricity_of_ellipse : ∃ e, e = sqrt 6 / 3 :=
sorry

theorem lambda_mu_squared_sum_one (λ μ : ℝ) :
  ellipse_eq (λ * A.1 + μ * B.1) (λ * A.2 + μ * B.2) →
  λ^2 + μ^2 = 1 :=
sorry

end eccentricity_of_ellipse_lambda_mu_squared_sum_one_l695_695778


namespace john_total_cost_l695_695868

-- Definitions based on given conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_next_10_years : ℕ := 20000
def university_tuition : ℕ := 250000
def years_first_phase : ℕ := 8
def years_second_phase : ℕ := 10

-- We need to prove the total cost John pays
theorem john_total_cost : 
  (years_first_phase * yearly_cost_first_8_years + years_second_phase * yearly_cost_next_10_years + university_tuition) / 2 = 265000 :=
by sorry

end john_total_cost_l695_695868


namespace parameterize_solutions_l695_695489

theorem parameterize_solutions (a x y z : ℝ) (k l : ℤ) (h1 : |a + 1/a| = 2)
  (h2 : |a| = 1) (h3 : cos z = 0) :
  (∃ k l : ℤ, x = π/2 + k * π ∧ y = π/2 + k * π ∧ z = π/2 + l * π) :=
by
  sorry

end parameterize_solutions_l695_695489


namespace zongzi_packing_l695_695904

theorem zongzi_packing (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (8 * x + 10 * y = 200) ↔ (x, y) = (5, 16) ∨ (x, y) = (10, 12) ∨ (x, y) = (15, 8) ∨ (x, y) = (20, 4) := 
sorry

end zongzi_packing_l695_695904


namespace at_least_six_neighbors_l695_695674

def Polygon (P : Type) := {s : set P // ∃ d : ℝ, d > 0 ∧ ∀ p ∈ s, Metric.ball p d = s }

axiom square : set (ℝ × ℝ)
axiom square_subdivision (side_length : ℝ) (P : set (ℝ × ℝ)) : Prop :=
  square = {p | 0 ≤ p.1 ∧ p.1 ≤ side_length ∧ 0 ≤ p.2 ∧ p.2 ≤ side_length } ∧ 
  ∃ polys : set (Polygon P), (∀ poly ∈ polys, poly.val ⊆ square) ∧ 
  (∀ poly ∈ polys, ∃ d : ℝ, d ≤ side_length / 30 ∧ ∀ p ∈ poly.val, Metric.ball p d = poly.val) ∧ 
  ∀ p ∈ square, ∃ poly ∈ polys, p ∈ poly.val

theorem at_least_six_neighbors (P : set (ℝ × ℝ)) {side_length : ℝ} 
  (h1 : side_length = 1) 
  (h2 : square_subdivision side_length P) : ∃ poly ∈ P, ∃ neighbors : set (Polygon P), neighbors.card ≥ 6 := 
by 
  sorry

end at_least_six_neighbors_l695_695674


namespace bisector_theorem_problem_l695_695019

theorem bisector_theorem_problem 
  (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 12)
  (D1F : ℝ) (h_D1F : D1F = 12 / 5) (D1E : ℝ) (h_D1E : D1E = 13 / 5)
  (XZ : ℝ) (h_XZ : XZ = 12 / 5) (XY : ℝ) (h_XY : XY = 13 / 5) :
  ∃ (XX1 : ℝ), XX1 = 12 / 25 :=
by
  use 12 / 25
  sorry

end bisector_theorem_problem_l695_695019


namespace find_m_l695_695667

noncomputable section

variables (p q : ℝ) (m : ℝ)

def line (p q : ℝ) (t : ℝ) := p + t * (q - p)

theorem find_m (p q : ℝ) (hpq : p ≠ q) :
  (∃ m, (m * p + (5 / 6) * q) = line p q (5 / 6)) ↔ m = 1 / 6 :=
by
  sorry

end find_m_l695_695667


namespace my_age_is_five_times_son_age_l695_695443

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l695_695443


namespace largest_five_digit_palindromic_number_l695_695312

def is_five_digit_palindrome (n : ℕ) : Prop := n / 10000 = n % 10 ∧ (n / 1000) % 10 = (n / 10) % 10

def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

theorem largest_five_digit_palindromic_number :
  ∃ (abcba deed : ℕ), is_five_digit_palindrome abcba ∧ 10000 ≤ abcba ∧ abcba < 100000 ∧ is_four_digit_palindrome deed ∧ 1000 ≤ deed ∧ deed < 10000 ∧ abcba = 45 * deed ∧ abcba = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l695_695312


namespace maddie_watched_8_episodes_l695_695438

def minutes_per_episode : ℕ := 44
def minutes_monday : ℕ := 138
def minutes_tuesday_wednesday : ℕ := 0
def minutes_thursday : ℕ := 21
def episodes_friday : ℕ := 2
def minutes_per_episode_friday := episodes_friday * minutes_per_episode
def minutes_weekend : ℕ := 105
def total_minutes := minutes_monday + minutes_tuesday_wednesday + minutes_thursday + minutes_per_episode_friday + minutes_weekend
def answer := total_minutes / minutes_per_episode

theorem maddie_watched_8_episodes : answer = 8 := by
  sorry

end maddie_watched_8_episodes_l695_695438


namespace combined_weight_of_two_new_students_l695_695383

theorem combined_weight_of_two_new_students (W : ℕ) (X : ℕ) 
  (cond1 : (W - 150 + X) / 8 = (W / 8) - 2) :
  X = 134 := 
sorry

end combined_weight_of_two_new_students_l695_695383


namespace problem_solution_l695_695294

noncomputable def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_xplus1_even (x : ℝ) : f (x + 1) = f (-x + 1) := sorry

theorem problem_solution : f 2015 = -1 := 
by 
  sorry

end problem_solution_l695_695294


namespace maximum_area_of_triangle_l695_695377

theorem maximum_area_of_triangle 
  (a b c : ℝ)
  (A B C : α)
  (h1 : a * real.cos B + b * real.cos A = real.sqrt 3)
  (h2 : let R := 1 in real.pi * R^2 = real.pi) 
  (h3 : a = b) 
  (h4 : c = real.sqrt 3) 
  (h5 : real.sin C = real.sqrt 3 / 2) 
  : 
  let area := (1 / 2) * a * b * real.sin C in 
  area ≤ (3 * real.sqrt 3) / 4 :=
sorry

end maximum_area_of_triangle_l695_695377


namespace probability_not_square_or_cube_l695_695552

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695552


namespace star_m_eq_33_l695_695880

-- Define the star function for sum of digits of a positive integer.
def star (x : ℕ) : ℕ :=
  x.digits.foldl (· + ·) 0

-- Define the set S with the given conditions.
def S : set ℕ :=
  {n | star n = 15 ∧ (∀ d ∈ n.digits, 2 ≤ d) ∧ 0 ≤ n ∧ n < 10^8}

-- Define m to be the number of elements in S.
def m : ℕ := fintype.card S.to_finset

-- State the theorem that proves the question equals the correct answer given the conditions.
theorem star_m_eq_33 : star m = 33 :=
  by
  sorry

end star_m_eq_33_l695_695880


namespace total_amount_paid_l695_695867

variable (bikes : ℕ) (price_per_bike : ℕ) (vehicles : ℕ) (price_per_vehicle : ℕ) (registration_cost : ℕ)

-- Define the number of bikes, price per bike, number of off-road vehicles, price per vehicle, and registration cost
def bikes := 3
def price_per_bike := 150
def vehicles := 4
def price_per_vehicle := 300
def registration_cost := 25

-- Calculate the total cost
def total_cost :=
  let bikes_cost := bikes * price_per_bike
  let vehicles_cost := vehicles * price_per_vehicle
  let total_registration_cost := (bikes + vehicles) * registration_cost
  bikes_cost + vehicles_cost + total_registration_cost

-- The proof problem
theorem total_amount_paid : total_cost = 1825 :=
by
  sorry

end total_amount_paid_l695_695867


namespace sum_of_real_solutions_l695_695741

theorem sum_of_real_solutions : 
  let a := 1
  let b := -12
  let c := 16
  let discriminant := b^2 - 4 * a * c
  let x1 := (12 + Real.sqrt discriminant) / (2 * a)
  let x2 := (12 - Real.sqrt discriminant) / (2 * a)
  (x1 + x2 = 12) :=
by
  let a := 1
  let b := -12
  let c := 16
  let discriminant := b^2 - 4 * a * c
  let x1 := (12 + Real.sqrt discriminant) / (2 * a)
  let x2 := (12 - Real.sqrt discriminant) / (2 * a)
  have h : x1 + x2 = 12
  sorry

end sum_of_real_solutions_l695_695741


namespace place_abs_values_correctly_l695_695854

theorem place_abs_values_correctly : abs (abs (1 - 2) - abs (4 - 8) - 16) = 19 := by
  sorry

end place_abs_values_correctly_l695_695854


namespace inequality_proof_l695_695457

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + 1) ^ 2 / b + (b + 1) ^ 2 / a ≥ 8 :=
by
    have H₁ : (a + 1) ^ 2 = a^2 + 2*a + 1 := by ring
    have H₂ : (b + 1) ^ 2 = b^2 + 2*b + 1 := by ring
    have H3 : (a^2 + 2*a + 1) / b + (b^2 + 2*b + 1) / a =
                    a^2/b + 2*a/b + 1/b + b^2/a + 2*b/a + 1/a := by
      -- Proof by term expansion
      sorry
    have H4 : a^2 / b + b^2 / a ≥ 2 * (a * b / (a * b)) := by
      -- AM-GM Inequality Application
      sorry
    have H5 : 2 * (a*b / (a*b)) = 2 := by simp
    have H6 : 2*a / b + 2*b / a ≥ 4 := sorry
    have H7 : a^2/b + 2*a/b + 1/b + b^2/a + 2*b/a + 1/a ≥ 8 := by
      nlinarith [H3, H4, H6]
    assumption

end inequality_proof_l695_695457


namespace expected_value_of_winnings_l695_695991

theorem expected_value_of_winnings :
  let P_H := 2 / 5
  let P_T := 3 / 5
  let G_H := 5
  let L_T := 1
  E = P_H * G_H + P_T * (- L_T) →  E = 7 / 5 :=
begin
  intro E,
  sorry,
end

end expected_value_of_winnings_l695_695991


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l695_695047

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l695_695047


namespace exists_solution_in_range_l695_695705

open Function

theorem exists_solution_in_range : ∃ z ∈ set.Icc (-10 : ℝ) 10, exp (2 * z) = (z - 2) / (z + 2) := 
sorry

end exists_solution_in_range_l695_695705


namespace line_intersection_up_to_four_lines_l695_695607

variables (R1 R2 : ℝ) (h1 : R1 > 0) (h2 : R2 > 0) (h3 : R1 > R2)
variables (O P : ℝ×ℝ×ℝ) (hP : O ≠ P)
variables (α β : ℝ)

-- Statement of the problem in Lean
theorem line_intersection_up_to_four_lines :
  ∃ l : ℝ, 
    (∃ A1 B1 B2 A2 : ℝ×ℝ×ℝ, 
      -- Conditions to be met by points A1, B1, B2, A2
      (dist (0,0,0) A1 = R2) ∧
      (dist (0,0,0) B1 = R2) ∧
      (dist (0,0,0) B2 = R1) ∧
      (dist (0,0,0) A2 = R1) ∧
      -- Angle condition
      (∠ A1 O B1 = α) ∧
      -- Line intersection condition
      (∃ l1 : line, l1.through P A1 ∧ l1.through P B1 ∧ l1.through P B2 ∧ l1.through P A2 ∧ 
                     azimuth l1 = β)) := sorry

end line_intersection_up_to_four_lines_l695_695607


namespace triangle_side_length_l695_695863

-- Define the given conditions
def angle_B : ℝ := 60 * Real.pi / 180  -- converting degrees to radians
def a : ℝ := 3
def b : ℝ := Real.sqrt 13

-- Define Law of Cosines
def law_of_cosines (a b c angle_B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos angle_B

-- State the theorem
theorem triangle_side_length (c : ℝ) : 
  law_of_cosines a b c angle_B → c = 4 :=
by
  sorry

end triangle_side_length_l695_695863


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l695_695046

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l695_695046


namespace quadratic_exists_g_h_l695_695057

noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def h (x : ℝ) : ℝ := (x + b / (2 * a) + 1 / 2)^2
noncomputable def g (x : ℝ) : ℝ := a^2 * x^2 + (2 * a * d - a^2) * x + d^2

theorem quadratic_exists_g_h (a b c : ℝ) (ha : a ≠ 0) (d : ℝ) :
  ∃ (g h : ℝ → ℝ), f(x) = a * x^2 + b * x + c ∧ f(x + 1) = a * (x + 1)^2 + b * (x + 1) + c ∧ 
  g x = a^2 * x^2 + (2 * a * d - a^2) * x + d^2 ∧ 
  h x = (x + b / (2 * a) + 1 / 2)^2 ∧ 
  (f x) * (f (x + 1)) = g (h x) :=
by {
  sorry
}

end quadratic_exists_g_h_l695_695057


namespace estimated_population_correct_correlation_coefficient_correct_l695_695213

noncomputable def estimated_population (sum_y : ℕ) (num_plots : ℕ) (num_samples : ℕ) : ℕ :=
  (sum_y / num_samples) * num_plots

theorem estimated_population_correct :
  estimated_population (∑ i in range 20, y i) 200 20 = 12000 := by
  sorry

noncomputable def correlation_coefficient (sum_x_square : ℝ) (sum_y_square : ℝ) (sum_xy : ℝ) : ℝ :=
  sum_xy / (Real.sqrt (sum_x_square * sum_y_square))

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 := by
  sorry

end estimated_population_correct_correlation_coefficient_correct_l695_695213


namespace infinite_symm_colored_subset_exists_l695_695262

theorem infinite_symm_colored_subset_exists :
  (∀ (f : ℤ × ℤ → Prop), 
      (∀ x, f x ∨ ¬f x) → 
      (∃ g : ℤ × ℤ → Bool, 
        ∀ x, g x = tt ∨ g x = ff) →
  (∃ (h : ℤ × ℤ → Prop),
    (∀ x, (h x ↔ f x) ∧ 
    (∃ c : ℤ × ℤ, (∀ x, (h x → h (2 * c.1 - x.1, 2 * c.2 - x.2)) ∨ (∃ k : ℤ, c = (k, k))))) :=
by sorry

end infinite_symm_colored_subset_exists_l695_695262


namespace age_difference_constant_l695_695511

theorem age_difference_constant (a b x : ℕ) : (a + x) - (b + x) = a - b :=
by
  sorry

end age_difference_constant_l695_695511


namespace mow_lawn_payment_l695_695975

theorem mow_lawn_payment (bike_cost weekly_allowance babysitting_rate babysitting_hours money_saved target_savings mowing_payment : ℕ) 
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : babysitting_rate = 7)
  (h4 : babysitting_hours = 2)
  (h5 : money_saved = 65)
  (h6 : target_savings = 6) :
  mowing_payment = 10 :=
sorry

end mow_lawn_payment_l695_695975


namespace graph_location_l695_695930

theorem graph_location (k : ℝ) (H : k > 0) :
    (∀ x : ℝ, (0 < x → 0 < y) → (y = 2/x) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
    sorry

end graph_location_l695_695930


namespace probability_neither_square_nor_cube_l695_695531

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695531


namespace circles_are_externally_tangent_l695_695126

-- Define the first circle
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the second circle
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define a predicate that checks if two circles are externally tangent
def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  let (x1, y1) := center1
  let (x2, y2) := center2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = radius1 + radius2

-- Define the statement to be proved
theorem circles_are_externally_tangent : externally_tangent (0, 0) (3, 4) 2 3 :=
  sorry

end circles_are_externally_tangent_l695_695126


namespace geometric_seq_sum_l695_695333

noncomputable def a_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

theorem geometric_seq_sum :
  let a1 := a_n 1
  let a2 := a_n 2
  let a3 := a_n 3
  let a4 := a_n 4
  let a5 := a_n 5
  a1 + |a2| + a3 + |a4| + a5 = 121 :=
by
  sorry

end geometric_seq_sum_l695_695333


namespace find_smallest_angle_l695_695736

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l695_695736


namespace probability_neither_perfect_square_nor_cube_l695_695575

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695575


namespace rationalize_denominator_l695_695458

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = (Real.sqrt 3) / 5 :=
by
  sorry

end rationalize_denominator_l695_695458


namespace point_relationship_l695_695366

theorem point_relationship (a b c : ℝ) 
  (ha : a = -\frac{1}{-1})
  (hb : 1 = -\frac{1}{b})
  (hc : c = -\frac{1}{2}) :
  a > c ∧ c > b :=
by
  have h_a : a = 1 := by simp [ha]
  have h_b : b = -1 := by simp [hb, eq_comm, neg_inv]
  have h_c : c = -1/2 := by simp [hc]
  have : 1 > -1/2 := by linarith
  have : -1/2 > -1 := by linarith
  exact ⟨this, this⟩

end point_relationship_l695_695366


namespace initial_mixture_amount_l695_695200

/-- A solution initially contains an unknown amount of a mixture consisting of 15% sodium chloride
(NaCl), 30% potassium chloride (KCl), 35% sugar, and 20% water. To this mixture, 50 grams of sodium chloride
and 80 grams of potassium chloride are added. If the new salt content of the solution (NaCl and KCl combined)
is 47.5%, how many grams of the mixture were present initially?

Given:
  * The initial mixture consists of 15% NaCl and 30% KCl.
  * 50 grams of NaCl and 80 grams of KCl are added.
  * The new mixture has 47.5% NaCl and KCl combined.
  
Prove that the initial amount of the mixture was 2730 grams. -/
theorem initial_mixture_amount
    (x : ℝ)
    (h_initial_mixture : 0.15 * x + 50 + 0.30 * x + 80 = 0.475 * (x + 130)) :
    x = 2730 := by
  sorry

end initial_mixture_amount_l695_695200


namespace cos_C_in_triangle_l695_695376

theorem cos_C_in_triangle 
  {A B C : ℝ} {a b c : ℝ} (h1 : a = 3 * b / 2) (h2 : a = 3 * c / 4) 
  (h3 : b = 2 * c / 4) 
  (h4 : 0 < b) (h5 : 0 < c) (h6 : 0 < a) :
  ∃ x : ℝ, x > 0 ∧ a = 3 * x ∧ b = 2 * x ∧ c = 4 * x ∧ (cos C = -1 / 4) :=
begin
  sorry
end

end cos_C_in_triangle_l695_695376


namespace rows_of_pies_l695_695988

theorem rows_of_pies (baked_pecan_pies : ℕ) (baked_apple_pies : ℕ) (pies_per_row : ℕ) : 
  baked_pecan_pies = 16 ∧ baked_apple_pies = 14 ∧ pies_per_row = 5 → 
  (baked_pecan_pies + baked_apple_pies) / pies_per_row = 6 :=
by
  sorry

end rows_of_pies_l695_695988


namespace mode_class_scores_l695_695017

theorem mode_class_scores :
  let scores := [95, 90, 85, 90, 92]
  ∃ mode : ℕ, mode = 90 ∧ (∀ m ∈ scores, list.count scores m ≤ list.count scores 90) :=
begin
  let scores := [95, 90, 85, 90, 92],
  use 90,
  split,
  { refl },
  { intros m hm,
    sorry
  }
end

end mode_class_scores_l695_695017


namespace area_ratio_of_pentagons_l695_695156

noncomputable theory

theorem area_ratio_of_pentagons
    (a : ℝ) -- side length of the larger pentagon
    (b : ℝ) -- side length of the smaller pentagon
    (h : ℝ := a * real.cos (π / 10)) -- distance due to cosine of π/10
    (H : ℝ := a * real.cos (3 * π / 10) + h) -- full distance of a side of the larger pentagon to the opposite vertex
    (k : ℝ := b / a) -- ratio of side lengths
    : k^2 = (7 - 3 * real.sqrt 5) / 2 :=
sorry

end area_ratio_of_pentagons_l695_695156


namespace polynomial_a_not_factorable_l695_695911

theorem polynomial_a_not_factorable (P : Polynomial ℤ) :
  P = ∑ i in (Finset.range (1111 + 1)), (2 * i + 2) * X^(2222 - 2 * i) →
  ¬ ∃ f g : Polynomial ℤ, f * g = P := 
by
  sorry

end polynomial_a_not_factorable_l695_695911


namespace probability_non_square_non_cube_l695_695589

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695589


namespace find_C_and_D_l695_695719

theorem find_C_and_D :
  (∀ x, x^2 - 3 * x - 10 ≠ 0 → (4 * x - 3) / (x^2 - 3 * x - 10) = (17 / 7) / (x - 5) + (11 / 7) / (x + 2)) :=
by
  sorry

end find_C_and_D_l695_695719


namespace find_unique_n_l695_695874

-- Define the sequence
def sequence (n : ℕ) : ℕ := 2^n + 49

-- Main proof theorem stating there exists a unique n satisfying the given conditions
theorem find_unique_n :
  ∃! n : ℕ,
  (∃ p q r s : ℕ, p < q ∧ r < s ∧ Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ q - p = s - r ∧
  sequence n = p * q ∧ sequence (n + 1) = r * s) :=
begin
  sorry
end

end find_unique_n_l695_695874


namespace total_songs_purchased_is_162_l695_695971

variable (c_country : ℕ) (c_pop : ℕ) (c_jazz : ℕ) (c_rock : ℕ)
variable (s_country : ℕ) (s_pop : ℕ) (s_jazz : ℕ) (s_rock : ℕ)

-- Setting up the conditions
def num_country_albums := 6
def num_pop_albums := 2
def num_jazz_albums := 4
def num_rock_albums := 3

-- Number of songs per album
def country_album_songs := 9
def pop_album_songs := 9
def jazz_album_songs := 12
def rock_album_songs := 14

theorem total_songs_purchased_is_162 :
  num_country_albums * country_album_songs +
  num_pop_albums * pop_album_songs +
  num_jazz_albums * jazz_album_songs +
  num_rock_albums * rock_album_songs = 162 := by
  sorry

end total_songs_purchased_is_162_l695_695971


namespace mushroom_count_l695_695135

theorem mushroom_count (x y : ℕ) (h1 : x + y = 30) (h2 : ∀ s : finset ℕ, s.card = 12 -> s.exists (λ m, m = x)) (h3 : ∀ t : finset ℕ, t.card = 20 -> t.exists (λ n, n = y)) : x = 19 ∧ y = 11 :=
sorry

end mushroom_count_l695_695135


namespace probability_neither_square_nor_cube_l695_695543

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695543


namespace face_opposite_teal_is_yellow_l695_695662

/-- Definition of the cube with each face labeled according to the first view described --/
noncomputable def cube :=
  { top := "B", front_1_view := "V", right_1_view := "Y", front_2_view := "O", front_3_view := "L",
    opposite_teal := "Y"} -- Position of colors around the cube

/--Proving the face opposite teal is yellow given the conditions of the cube views--/
theorem face_opposite_teal_is_yellow :
  ∀ (top front_1_view right_1_view front_2_view front_3_view : string) (opposite_teal : string),
  top = "B" → right_1_view = "Y" → front_1_view = "V" → front_2_view = "O" → front_3_view = "L" →
  opposite_teal = "Y" →
  (cube.opposite_teal = "Y") :=
by
  intros top front_1_view right_1_view front_2_view front_3_view opposite_teal,
  assume ht : top = "B",
  assume hr : right_1_view = "Y",
  assume hv : front_1_view = "V",
  assume ho : front_2_view = "O",
  assume hl : front_3_view = "L",
  assume hy : opposite_teal = "Y",
  sorry

end face_opposite_teal_is_yellow_l695_695662


namespace total_students_l695_695622

def Varsity_students : ℕ := 1300
def Northwest_students : ℕ := 1400
def Central_students : ℕ := 1800
def Greenbriar_students : ℕ := 1650

theorem total_students : Varsity_students + Northwest_students + Central_students + Greenbriar_students = 6150 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end total_students_l695_695622


namespace simplify_sqrt_product_l695_695092

theorem simplify_sqrt_product : (Real.sqrt (3 * 5) * Real.sqrt (3 ^ 5 * 5 ^ 5) = 3375) :=
  sorry

end simplify_sqrt_product_l695_695092


namespace construction_continues_indefinitely_no_triple_sums_to_zero_l695_695773

noncomputable def sequence (x0 y0 z0 : ℝ) (n : ℕ) : ℝ × ℝ × ℝ :=
nat.iterate (λ ⟨x, y, z⟩, (2 * x / (x^2 - 1), 2 * y / (y^2 - 1), 2 * z / (z^2 - 1))) n (x0, y0, z0)

theorem construction_continues_indefinitely :
    ∀ (n : ℕ), let (x_n, y_n, z_n) := sequence 2 4 (6/7) n in
    (x_n ≠ 1 ∧ x_n ≠ -1) ∧ (y_n ≠ 1 ∧ y_n ≠ -1) ∧ (z_n ≠ 1 ∧ z_n ≠ -1) := 
by
  sorry

theorem no_triple_sums_to_zero :
    ∀ (k : ℕ), let (x_k, y_k, z_k) := sequence 2 4 (6/7) k in
    x_k + y_k + z_k ≠ 0 := 
by
  sorry

end construction_continues_indefinitely_no_triple_sums_to_zero_l695_695773


namespace pants_cost_correct_l695_695686

-- Define the conditions as variables
def initial_money : ℕ := 71
def shirt_cost : ℕ := 5
def num_shirts : ℕ := 5
def remaining_money : ℕ := 20

-- Define intermediates necessary to show the connection between conditions and the question
def money_spent_on_shirts : ℕ := num_shirts * shirt_cost
def money_left_after_shirts : ℕ := initial_money - money_spent_on_shirts
def pants_cost : ℕ := money_left_after_shirts - remaining_money

-- The main theorem to prove the question is equal to the correct answer
theorem pants_cost_correct : pants_cost = 26 :=
by
  sorry

end pants_cost_correct_l695_695686


namespace probability_neither_perfect_square_nor_cube_l695_695529

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695529


namespace mean_days_calculation_l695_695900

-- Definitions for the problem conditions
def num_students_5_days : ℕ := 2
def num_students_10_days : ℕ := 4
def num_students_15_days : ℕ := 3
def num_students_20_days : ℕ := 7
def num_students_25_days : ℕ := 5
def num_students_0_days : ℕ := 4

def days_consumed_5_days : ℕ := 5
def days_consumed_10_days : ℕ := 10
def days_consumed_15_days : ℕ := 15
def days_consumed_20_days : ℕ := 20
def days_consumed_25_days : ℕ := 25

-- Computation for the problem
def total_days_consumed : ℕ :=
  (num_students_5_days * days_consumed_5_days) +
  (num_students_10_days * days_consumed_10_days) +
  (num_students_15_days * days_consumed_15_days) +
  (num_students_20_days * days_consumed_20_days) +
  (num_students_25_days * days_consumed_25_days) +
  (num_students_0_days * 0)

def total_students : ℕ :=
  num_students_5_days +
  num_students_10_days +
  num_students_15_days +
  num_students_20_days +
  num_students_25_days +
  num_students_0_days

def mean_days : ℝ :=
  (total_days_consumed : ℝ) / (total_students : ℝ)

theorem mean_days_calculation :
  mean_days = 14.4 :=
by 
  have h1 : total_days_consumed = 360 := rfl
  have h2 : total_students = 25 := rfl
  have h3 : mean_days = (360 : ℝ) / 25 := rfl
  have h4 : (360 : ℝ) / 25 = 14.4 := by norm_num
  exact eq.trans h3 h4

end mean_days_calculation_l695_695900


namespace geometric_progression_identical_numbers_l695_695942

theorem geometric_progression_identical_numbers (n : ℕ) 
    (numbers : Fin 4n → ℝ) (h_pos : ∀ i, 0 < numbers i) 
    (h_geo : ∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i < 4n ∧ j < 4n ∧ k < 4n ∧ l < 4n →
        ∃ r : ℝ, r ≠ 0 ∧ numbers j = r * numbers i ∧ numbers k = r * numbers j ∧ numbers l = r * numbers k) : 
    ∃ (x : ℝ), (∃ (t : Fin n → Fin 4n), ∀ (i j : Fin n), numbers (t i) = numbers (t j) ∧ t i ≠ t j) :=
sorry

end geometric_progression_identical_numbers_l695_695942


namespace cube_root_power_simplify_l695_695481

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695481


namespace sum_of_ages_five_years_from_now_l695_695623

noncomputable def viggo_age_when_brother_was_2 (brother_age: ℕ) : ℕ :=
  10 + 2 * brother_age

noncomputable def current_viggo_age (viggo_age_at_2: ℕ) (current_brother_age: ℕ) : ℕ :=
  viggo_age_at_2 + (current_brother_age - 2)

def sister_age (viggo_age: ℕ) : ℕ :=
  viggo_age + 5

noncomputable def cousin_age (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) : ℕ :=
  ((viggo_age + brother_age + sister_age) / 3)

noncomputable def future_ages_sum (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) (cousin_age: ℕ) : ℕ :=
  viggo_age + 5 + brother_age + 5 + sister_age + 5 + cousin_age + 5

theorem sum_of_ages_five_years_from_now :
  let current_brother_age := 10
  let viggo_age_at_2 := viggo_age_when_brother_was_2 2
  let current_viggo_age := current_viggo_age viggo_age_at_2 current_brother_age
  let current_sister_age := sister_age current_viggo_age
  let current_cousin_age := cousin_age current_viggo_age current_brother_age current_sister_age
  future_ages_sum current_viggo_age current_brother_age current_sister_age current_cousin_age = 99 := sorry

end sum_of_ages_five_years_from_now_l695_695623


namespace problem_statement_l695_695781

theorem problem_statement (a b : ℝ) (c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (4/b) = 1) :
  c < a + b :=
begin
  sorry -- Proof to be provided
end

end problem_statement_l695_695781


namespace probability_not_square_or_cube_l695_695554

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695554


namespace wealth_numbers_count_l695_695608

theorem wealth_numbers_count :
  let N := 100 in
  (finset.filter (λ x : ℕ, ∃ a b : ℕ, 4 * a + 9 * b = x) (finset.range (N + 1))).card = 88 :=
by
  sorry

end wealth_numbers_count_l695_695608


namespace interval_length_of_m_values_l695_695425

noncomputable def count_lattice_points_below_line (m : ℝ) : ℕ :=
  -- Count the number of lattice points (x, y) in the range [1, 50] x [1, 50] satisfying y ≤ mx - 5.
  sorry

theorem interval_length_of_m_values :
  let S := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50}
  ∃ m₁ m₂ : ℝ, 
  m₁ = 0.8 ∧ m₂ = 0.9 ∧ count_lattice_points_below_line m₁ + count_lattice_points_below_line m₂ = 2500 ∧
  m₂ - m₁ = 1 / 10 ∧ 
  nat.gcd 1 10 = 1 ∧ 
  1 + 10 = 11 :=
by sorry

end interval_length_of_m_values_l695_695425


namespace video_games_per_shelf_l695_695682

theorem video_games_per_shelf (total_games : ℕ) (shelves : ℕ) (approx_games_per_shelf : ℤ) 
  (h1 : total_games = 163) (h2 : shelves = 2) (h3 : approx_games_per_shelf = 82) : 
  (total_games / shelves : ℤ) = approx_games_per_shelf :=
begin
  sorry
end

end video_games_per_shelf_l695_695682


namespace cost_price_computer_table_chair_l695_695189

-- Defining the cost price of the items as a variable
variable (C : ℝ)

-- Defining the conditions as given in the problem
def markup_price (C : ℝ) : ℝ := 1.20 * C
def discounted_price (C : ℝ) : ℝ := 0.90 * markup_price C

-- Given that the discounted price is Rs. 8400
axiom given_disounted_price : discounted_price C = 8400

-- Proving the cost price before the discount was applied
theorem cost_price_computer_table_chair (C : ℝ) : discounted_price C = 8400 → C = 7777.78 := 
by
  intros h,
  rw [discounted_price, markup_price] at h,
  -- Solve the equation
  have h1 : 0.90 * (1.20 * C) = 8400 := h,
  have h2 : 1.08 * C = 8400 := h1,
  have h3 : C = 8400 / 1.08 := (eq_div_iff (by norm_num)).mp h2,
  norm_num at h3,
  exact h3

end cost_price_computer_table_chair_l695_695189


namespace magnitude_of_combination_l695_695815

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (3, 1)

open_locale real_inner_product_space

-- Definition of the operation
def scaled_vector (a b : ℝ × ℝ) (n m : ℝ) : ℝ × ℝ :=
  (n * a.1 + m * b.1, n * a.2 + m * b.2)

-- Magnitude calculation
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- The theorem to prove
theorem magnitude_of_combination :
  magnitude (scaled_vector vector_a vector_b 2 3) = real.sqrt 170 := by
  sorry

end magnitude_of_combination_l695_695815


namespace probability_neither_square_nor_cube_l695_695568

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695568


namespace num_valid_n_between_101_and_1000_l695_695756

noncomputable def count_terminating_decimals : ℕ :=
  Nat.card {n // (n ≥ 101 ∧ n ≤ 1000 ∧ (∃ a b : ℕ, n = 2^a * 5^b) ∧ (1 / n).numerator % 10 ≠ 0)}

theorem num_valid_n_between_101_and_1000 : count_terminating_decimals = 10 := sorry

end num_valid_n_between_101_and_1000_l695_695756


namespace probability_neither_perfect_square_nor_cube_l695_695576

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695576


namespace total_rainfall_l695_695407

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l695_695407


namespace min_distance_between_circles_l695_695430

noncomputable def a : ℂ := 2 - 5 * complex.I
noncomputable def b : ℂ := -3 + 4 * complex.I
def circle (c : ℂ) (r : ℝ) : set ℂ := {z | complex.abs (z - c) = r}

theorem min_distance_between_circles :
  ∀ z w : ℂ, z ∈ circle a 2 → w ∈ circle b 4 → complex.abs (z - w) ≥ real.sqrt 106 - 6 :=
by
  sorry

end min_distance_between_circles_l695_695430


namespace solve_equation_l695_695429

theorem solve_equation (n : ℕ) (x : ℝ) 
  (h : x + (2 * ⌊x⌋) + (3 * ⌊x⌋) + ... + (n * ⌊x⌋) = ((1 + 2 + ... + n) ^ 2)) :
  x = (1 + n) * n / 2 :=
sorry

end solve_equation_l695_695429


namespace max_roots_l695_695803

-- Define the functions f and g as provided
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

def g (x : ℝ) : ℝ :=
  if x > 0 then (x - 1/2)^2 + 1
  else -(x + 3)^2 + 1

-- Define a as a positive real number
variable (a : ℝ)
hypothesis (ha : a > 0)

-- Theorem statement
theorem max_roots (ha : a > 0) : ∃ (n : ℕ), n = 6 ∧ (λ x => g (f x) - a = 0).roots.count = 6 :=
sorry

end max_roots_l695_695803


namespace Ed_lost_marble_count_l695_695712

variable (D : ℕ) -- Number of marbles Doug has

noncomputable def Ed_initial := D + 19 -- Ed initially had D + 19 marbles
noncomputable def Ed_now := D + 8 -- Ed now has D + 8 marbles
noncomputable def Ed_lost := Ed_initial D - Ed_now D -- Ed lost Ed_initial - Ed_now marbles

theorem Ed_lost_marble_count : Ed_lost D = 11 := by 
  sorry

end Ed_lost_marble_count_l695_695712


namespace playerA_first_move_l695_695946

-- Define the initial state (11, 13) of the game
def initialState : ℕ × ℕ := (11, 13)

-- Define the move function for the game
def move (s t : ℕ × ℕ) := 
  (s.1 = t.1 ∧ s.2 > t.2) ∨
  (s.1 > t.1 ∧ s.2 = t.2) ∨
  (s.1 > t.1 ∧ s.2 > t.2 ∧ s.1 - t.1 = s.2 - t.2)

-- Define the winning state definition
def is_win_state (s : ℕ × ℕ) := s = (0, 0)

-- Define the $\star$ states known by our analysis
def is_star_state (s : ℕ × ℕ) : bool :=
  s = (3, 5) ∨ s = (8, 13)

-- State the main theorem
theorem playerA_first_move : ∃ t : ℕ × ℕ, move initialState t ∧ is_star_state t := 
by
  sorry

end playerA_first_move_l695_695946


namespace find_y_intercept_at_origin_l695_695191

def point (α : Type) := (α × α)

def slope (p1 p2 : point ℝ) : ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  (y2 - y1) / (x2 - x1)

def y_intercept (p1 p2 : point ℝ) : ℝ :=
  let m := slope p1 p2 in
  let (x1, y1) := p1 in
  y1 - m * x1

theorem find_y_intercept_at_origin : 
  y_intercept (3, 20) (-7, -2) = 13.4 :=
by
  -- proof goes here
  sorry

end find_y_intercept_at_origin_l695_695191


namespace polynomial_coefficients_sum_l695_695354

theorem polynomial_coefficients_sum :
  let (a : ℕ → ℝ) := (λ n => (1 - 2*x)^4.coeff n) in
  a 0 + a 1 + a 3 = -39 :=
by
  sorry

end polynomial_coefficients_sum_l695_695354


namespace probability_min_max_l695_695985

variable {Ω : Type*} {ι : Type*}
variables (σ : Ω → ι → ℝ) [Fintype ι] [DecidableEq ι]

noncomputable def xi_min (ξ : ι → ℝ) : ℝ := fintype.min ι ξ  
noncomputable def xi_max (ξ : ι → ℝ) : ℝ := fintype.max ι ξ  

theorem probability_min_max (ξ : ι → MeasureTheory.Measure Ω) (x : ℝ)
  (h_indep : ∀ i j, i ≠ j → MeasureTheory.Measure.Indep (ξ i) (ξ j)) :
  (MeasureTheory.Prob {ω : Ω | xi_min (λ i, σ ω i) ≥ x} = ∏ i, MeasureTheory.Prob {ω : Ω | σ ω i ≥ x}) ∧
  (MeasureTheory.Prob {ω : Ω | xi_max (λ i, σ ω i) < x} = ∏ i, MeasureTheory.Prob {ω : Ω | σ ω i < x}) :=
  sorry

end probability_min_max_l695_695985


namespace probability_neither_perfect_square_nor_cube_l695_695525

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695525


namespace total_cost_l695_695416

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l695_695416


namespace arctan_tan_eq_three_fourths_x_l695_695093

theorem arctan_tan_eq_three_fourths_x (x : ℝ) : 
  (- (2 * Real.pi) / 3 ≤ x ∧ x ≤ (2 * Real.pi) / 3) → (Real.arctan (Real.tan x) = (3 * x) / 4) ↔ x = 0 := 
by
  intro h
  sorry

end arctan_tan_eq_three_fourths_x_l695_695093


namespace geometric_sequence_product_l695_695024

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)

theorem geometric_sequence_product (h : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_product_l695_695024


namespace probability_neither_perfect_square_nor_cube_l695_695579

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695579


namespace probability_neither_square_nor_cube_l695_695533

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695533


namespace focus_reflection_is_correct_l695_695507

-- Define the original parabola and the reflection line
def parabola (x y : ℝ) : Prop := y^2 = -8 * x
def reflection_line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-2, 0)

-- Define the function for reflecting a point over the line y = x - 1
def reflect_over_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in
  let s := (y - x + 1) / 2 in
  (x + 2 * s, y - 2 * s)

-- Assertion about the focus point after reflection
theorem focus_reflection_is_correct :
  reflect_over_line focus = (1, -3) :=
sorry

end focus_reflection_is_correct_l695_695507


namespace polygon_inequality_l695_695814

open Real

theorem polygon_inequality 
  (n : ℕ) (h : Fin n → ℝ) (R r : ℝ) 
  (hn_ge_3: n ≥ 3) 
  (h_nonneg : ∀ i, 0 ≤ h i) :
  R * cos (π / n) ≥ (∑ i, h i) / n ∧ (∑ i, h i) / n ≥ r :=
by
  sorry

end polygon_inequality_l695_695814


namespace pizzas_ordered_l695_695147

-- Definitions for the conditions
def total_slices : ℕ := 14
def slices_per_pizza : ℕ := 2

-- Theorem to prove the number of pizzas ordered
theorem pizzas_ordered : total_slices / slices_per_pizza = 7 := by
  rw [total_slices, slices_per_pizza]
  sorry

end pizzas_ordered_l695_695147


namespace spadesuit_value_l695_695254

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l695_695254


namespace correct_option_is_C_l695_695158

section
  variable {x : ℝ}

  -- Define functions for options A, B, C, D
  def f_A := λ x : ℝ, x - 1
  def g_A := λ x : ℝ, x^2 / x - 1

  def f_B := λ x : ℝ, |x|
  def g_B := λ x : ℝ, (sqrt x)^2

  def f_C := λ x : ℝ, x
  def g_C := λ x : ℝ, real.cbrt (x^3)

  def f_D := λ x : ℝ, 2 * x
  def g_D := λ x : ℝ, sqrt (4 * x^2)

  -- Define their equivalence in terms of domain and function value correspondence
  def equiv_A : Prop := (∀ x, x ≠ 0 → f_A x = g_A x) ∧ (∀ x, (f_A x = g_A x) → x ≠ 0)
  def equiv_B : Prop := (∀ x, x ≥ 0 → f_B x = g_B x) ∧ (∀ x, f_B x = g_B x → x ≥ 0)
  def equiv_C : Prop := (∀ x, f_C x = g_C x)
  def equiv_D : Prop := (∀ x, f_D x = g_D x)

  -- Theorem to prove for the correct option
  theorem correct_option_is_C : equiv_C := by sorry
end

end correct_option_is_C_l695_695158


namespace sarah_starting_amount_l695_695914

def toyCar1_cost := 12
def toyCar2_originalCost := 15
def toyCar2_discount := 0.10
def scarf_originalCost := 10
def scarf_discount := 0.20
def beanie_cost := 14
def beanie_tax := 0.08
def necklace_cost := 20
def necklace_discount := 0.05
def gloves_cost := 12
def book_cost := 15
def remaining_amount := 7

def car2_cost := toyCar2_originalCost * (1 - toyCar2_discount)
def scarf_cost := scarf_originalCost * (1 - scarf_discount)
def beanie_total_cost := beanie_cost * (1 + beanie_tax)
def necklace_total_cost := necklace_cost * (1 - necklace_discount)
def total_cost := toyCar1_cost + car2_cost + scarf_cost + beanie_total_cost + necklace_total_cost + gloves_cost + book_cost

theorem sarah_starting_amount: total_cost + remaining_amount = 101.62 :=
by
  sorry

end sarah_starting_amount_l695_695914


namespace complex_modulus_l695_695293

theorem complex_modulus (z : ℂ) (h : (1 + (sqrt 3) * Complex.I) * z = 1 + Complex.I) : 
  Complex.abs z = sqrt 2 / 2 :=
by
  sorry

end complex_modulus_l695_695293


namespace dark_vs_light_diff_in_9x9_grid_l695_695649

def is_dark_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

def dark_squares (n : ℕ) :=
  { (i, j) | i < n ∧ j < n ∧ is_dark_square i j }

def light_squares (n : ℕ) :=
  { (i, j) | i < n ∧ j < n ∧ ¬ is_dark_square i j }

theorem dark_vs_light_diff_in_9x9_grid :
  dark_squares 9 .card = light_squares 9 .card + 1 :=
sorry

end dark_vs_light_diff_in_9x9_grid_l695_695649


namespace find_possible_K_l695_695837

theorem find_possible_K (K : ℕ) (N : ℕ) (h1 : K * (K + 1) / 2 = N^2) (h2 : N < 150)
  (h3 : ∃ m : ℕ, N^2 = m * (m + 1) / 2) : K = 1 ∨ K = 8 ∨ K = 39 ∨ K = 92 ∨ K = 168 := by
  sorry

end find_possible_K_l695_695837


namespace brad_net_profit_l695_695229

noncomputable def lemonade_glasses_per_gallon := 16
noncomputable def cost_per_gallon := 3.50
noncomputable def gallons_made := 2
noncomputable def price_per_glass := 1.00
noncomputable def glasses_drunk := 5
noncomputable def glasses_left_unsold := 6

theorem brad_net_profit :
  let total_glasses := gallons_made * lemonade_glasses_per_gallon in
  let glasses_not_sold := glasses_drunk + glasses_left_unsold in
  let glasses_sold := total_glasses - glasses_not_sold in
  let total_cost := gallons_made * cost_per_gallon in
  let total_revenue := glasses_sold * price_per_glass in
  let net_profit := total_revenue - total_cost in
  net_profit = 14 :=
by
  sorry

end brad_net_profit_l695_695229


namespace probability_neither_perfect_square_nor_cube_l695_695523

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695523


namespace tan_2x_eq_sin_x_has_3_solutions_l695_695818

-- Define the functions involved
def f (x : ℝ) := Real.tan (2 * x)
def g (x : ℝ) := Real.sin x

-- Define the interval of interest
def interval := Set.Icc 0 (2 * Real.pi)

-- The theorem stating the problem
theorem tan_2x_eq_sin_x_has_3_solutions :
  Set.card ((Set.Icc 0 (2 * Real.pi)).inter (SetOf λ x => f x = g x)) = 3 := sorry

end tan_2x_eq_sin_x_has_3_solutions_l695_695818


namespace nancy_first_counted_l695_695073

theorem nancy_first_counted (x : ℤ) (h : (x + 12 + 1 + 12 + 7 + 3 + 8) / 6 = 7) : x = -1 := 
by 
  sorry

end nancy_first_counted_l695_695073


namespace selected_student_id_l695_695606

theorem selected_student_id :
  ∃ (n : ℕ), n = 16 ∧ (∃ interval, interval ∈ {i : ℕ | 1 ≤ i ∧ i ≤ 54} ∧
  ∃ ids, id3 = 3 ∧ id29 = 29 ∧ id42 = 42 ∧ ids = [id3, id29, id42, n] ∧
  ∃ id_seq, id_seq = list.unfold (λ x, if x > 54 then none else some (x, x + interval)) 3 ∧ 
  {3, 29, 42} ⊆ id_seq.set ∧ n ∈ id_seq.set) :=
sorry

end selected_student_id_l695_695606


namespace difference_of_numbers_l695_695925

theorem difference_of_numbers : 
  ∃ (L S : ℕ), L = 1631 ∧ L = 6 * S + 35 ∧ L - S = 1365 := 
by
  sorry

end difference_of_numbers_l695_695925


namespace valid_seating_arrangements_count_l695_695714

/-- 
  Eight people are seated in eight chairs arranged in a circle. 
  Prove that the number of valid seating arrangements, such that
  no one sits in their original chair, in the chairs directly adjacent
  to it, or in the chairs two positions away from it, is 32.
--/
theorem valid_seating_arrangements_count : 
  let positions := {1, 2, 3, 4, 5, 6, 7, 8}
  let constraints (i : ℕ) (new_pos : ℕ) := new_pos ≠ i ∧ new_pos ≠ (i+1) % 8 ∧ new_pos ≠ (i-1) % 8 
    ∧ new_pos ≠ (i+2) % 8 ∧ new_pos ≠ (i-2) % 8
  ∃ arrangements : List (Π i, ℕ), 
    (∀ p ∈ arrangements, ∀ i, constraints i (p i)) 
    ∧ arrangements.length = 32 := 
sorry

end valid_seating_arrangements_count_l695_695714


namespace isosceles_triangle_sine_angle_l695_695728

theorem isosceles_triangle_sine_angle {A B C : Type*} [inhabited A] [inhabited B] [inhabited C]
  (isosceles : ∀ (A B C : ℝ), A = C)
  (constant_perimeter : ∀ (DEPL : set (ℝ × ℝ)), 
    (∃ (a h : ℝ), ∀ (x : ℝ), P (DEPL) = P (KNQR))) :
  sin A = 4 / 5 :=
by 
  sorry

end isosceles_triangle_sine_angle_l695_695728


namespace dash_cam_mounts_max_profit_l695_695076

noncomputable def monthly_profit (x t : ℝ) : ℝ :=
  (48 + t / (2 * x)) * x - 32 * x - 3 - t

theorem dash_cam_mounts_max_profit :
  ∃ (x t : ℝ), 1 < x ∧ x < 3 ∧ x = 3 - 2 / (t + 1) ∧
  monthly_profit x t = 37.5 := by
sorry

end dash_cam_mounts_max_profit_l695_695076


namespace probability_non_square_non_cube_l695_695591

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695591


namespace caroline_cannot_end_with_1_l695_695236

theorem caroline_cannot_end_with_1 :
  let nums := (finset.range 2023).erase 0, -- numbers from 1 to 2022
      sum := nums.sum id,
      sum_init := 2022 * 1011 in
  (∀ a b ∈ nums, 
    (∃ nums' : finset ℕ, nums' = (nums.erase a).erase b ∪ {a - b}) → -- erasing the numbers a and b and adding a - b
    ((nums \ {a, b}).sum id + (a - b)) % 2 = 1) → -- resulting parity of sum is always odd
  1 ∉ nums :=
begin
  sorry
end

end caroline_cannot_end_with_1_l695_695236


namespace min_distance_on_ellipse_l695_695314

-- Defining the problem conditions and the final proof goal
theorem min_distance_on_ellipse :
  let ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 2) = 1
  let distance (x y : ℝ) := real.sqrt ((x - 1)^2 + y^2)
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
  ∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → distance P.1 P.2 ≤ distance Q.1 Q.2 →
  distance P.1 P.2 = real.sqrt (15) / 3 := sorry

end min_distance_on_ellipse_l695_695314


namespace sum_of_solutions_eq_neg2_l695_695289

noncomputable def sum_of_real_solutions (a : ℝ) (h : a > 2) : ℝ :=
  -2

theorem sum_of_solutions_eq_neg2 (a : ℝ) (h : a > 2) :
  sum_of_real_solutions a h = -2 := sorry

end sum_of_solutions_eq_neg2_l695_695289


namespace probability_neither_square_nor_cube_l695_695572

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695572


namespace how_many_women_left_l695_695396

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l695_695396


namespace sequence_a_general_formula_sequence_b_sum_l695_695298

-- Define the sequence a_n using the provided condition
def sequence_a (n : ℕ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  ∀ n, 3 * S_n n = 1 - a_n n

-- General formula for a_n to be proven
theorem sequence_a_general_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
    sequence_a n a_n S_n → (a_n n = (1 / 4) ^ n) :=
by
  -- Proof logic to be filled in here
  sorry

-- Definition of the sequence b_n
def sequence_b (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  ∀ n, b_n n = 1 / (Real.logb 2 (a_n n) * Real.logb 2 (a_n (n + 1)))

-- Sum of the first n terms of b_n to be proven
theorem sequence_b_sum (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :
    (sequence_a n a_n S_n) ∧ (sequence_b a_n b_n) →
    (T_n n = ∑ k in Finset.range n, b_n k → (T_n n = n / (4 * (n + 1)))) :=
by
  -- Proof logic to be filled in here
  sorry

end sequence_a_general_formula_sequence_b_sum_l695_695298


namespace largest_three_digit_number_l695_695238

def digit_set : set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def digits : Type := finset ℕ

def valid_digits (d : digits) : Prop :=
  d.card = 6 ∧ ∀ x ∈ d, x ∈ digit_set

noncomputable def satisfies_equation (a b c d e f : ℕ) : Prop :=
  a + b * 10 + c = d * 100 + e * 10 + f

-- Prove largest valid three-digit number satisfying the conditions
theorem largest_three_digit_number :
  ∀ d : digits, valid_digits d →
    ∃ a b c d e f ∈ d, satisfies_equation a b c d e f → d * 100 + e * 10 + f = 105 :=
sorry

end largest_three_digit_number_l695_695238


namespace reciprocal_roots_k_value_l695_695353

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end reciprocal_roots_k_value_l695_695353


namespace smallest_positive_angle_l695_695731

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l695_695731


namespace min_colors_shapes_l695_695661

def representable_centers (C S : Nat) : Nat :=
  C + (C * (C - 1)) / 2 + S + S * (S - 1)

theorem min_colors_shapes (C S : Nat) :
  ∀ (C S : Nat), (C + (C * (C - 1)) / 2 + S + S * (S - 1)) ≥ 12 → (C, S) = (3, 3) :=
sorry

end min_colors_shapes_l695_695661


namespace negation_of_universal_statement_l695_695520
open Classical

variable (p : Prop)
variable (x y : ℝ)
variable (x0 y0 : ℝ)

-- Conditions
def condition_x_y_in_interval : Prop := x ∈ (0, 1)
def condition_x_y_sum : Prop := x + y < 2

-- Negation of the proposition
def negation_of_p : Prop := ∃ x0 y0 : ℝ, x0 ∈ (0, 1) ∧ y0 ∈ (0, 1) ∧ x0 + y0 ≥ 2

-- Proof problem statement
theorem negation_of_universal_statement : 
  (∀ x y : ℝ, x ∈ (0, 1) → y ∈ (0, 1) → x + y < 2) →
   (∃ x0 y0 : ℝ, x0 ∈ (0, 1) ∧ y0 ∈ (0, 1) ∧ x0 + y0 ≥ 2) :=
by
  -- Proof goes here
  sorry

end negation_of_universal_statement_l695_695520


namespace initial_hamburgers_l695_695196

-- Define the conditions as hypotheses
variables (x : ℕ) (served left_over : ℕ)
hypothesis h_served : served = 3
hypothesis h_left_over : left_over = 6
hypothesis h_initial : x = served + left_over

-- Prove the initial number of hamburgers
theorem initial_hamburgers : x = 9 :=
by
  rw [h_initial, h_served, h_left_over]
  simp [h_initial]
  sorry

end initial_hamburgers_l695_695196


namespace least_number_to_add_l695_695962

theorem least_number_to_add (a : ℕ) (p q r : ℕ) (h : a = 1076) (hp : p = 41) (hq : q = 59) (hr : r = 67) :
  ∃ k : ℕ, k = 171011 ∧ (a + k) % (lcm p (lcm q r)) = 0 :=
sorry

end least_number_to_add_l695_695962


namespace inequality_neg_mul_l695_695358

theorem inequality_neg_mul (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
    sorry

end inequality_neg_mul_l695_695358


namespace max_profit_l695_695209

theorem max_profit : ∃ v p : ℝ, 
  v + p ≤ 5 ∧
  v + 3 * p ≤ 12 ∧
  100000 * v + 200000 * p = 850000 :=
by
  sorry

end max_profit_l695_695209


namespace train_speed_kmph_l695_695671

-- Define the conditions of the problem
def train_length : ℝ := 20
def passing_time : ℝ := 1.9998400127989762
def conversion_factor : ℝ := 3.6

-- Define the speed calculation
def speed_mps : ℝ := train_length / passing_time
def speed_kmph : ℝ := speed_mps * conversion_factor

-- State the theorem to prove the speed in kmph
theorem train_speed_kmph : speed_kmph = 36.00287986320432 := by
  sorry

end train_speed_kmph_l695_695671


namespace area_N1N2N3_fraction_ABC_l695_695635

/-- Define points and conditions for the given problem -/
def A : Point := ...
def B : Point := ...
def C : Point := ...
def D : Point := midpoint B C
def E : Point := midpoint A C
def F : Point := midpoint A B
def N1 : Point := ...
def N2 : Point := ...
def N3 : Point := ...
/-- The ratio AN2 : N2N1 : N1D = 4 : 4 : 1 and similar conditions on other lines -/
axiom AN2_N2N1_N1D_ratio : ∀ P Q R : Point, ratio (A, N2, N1, D) = 4:4:1
axiom BE_and_CF_similar_ratios : ∀ P Q R : Point, ratio (B, E, N3, F) = 4:4:1

/-- Prove that the area of triangle N1N2N3 is 5/32 of triangle ABC -/
theorem area_N1N2N3_fraction_ABC {K : ℝ} (h : area (triangle A B C) = K) :
  ∃ (K : ℝ). area (triangle N1 N2 N3) / area (triangle A B C) = 5 / 32 :=
sorry

end area_N1N2N3_fraction_ABC_l695_695635


namespace percentage_of_error_l695_695027

theorem percentage_of_error (x : ℝ) (hx : x > 0) :
  let correct_result := 3 * x in
  let mistaken_result := x / (5/2) in
  let absolute_error := abs (correct_result - mistaken_result) in
  let percentage_error := (absolute_error / correct_result) * 100 in
  percentage_error = 86.67 :=
by
  sorry

end percentage_of_error_l695_695027


namespace evaluate_expression_l695_695265

variable (x : ℝ)

theorem evaluate_expression (x : ℝ) : 
  let y := (Real.sqrt (x^2 - 4 * x + 4)) + (Real.sqrt (x^2 + 4 * x + 4))
  in y = |x - 2| + |x + 2| :=
by
  sorry

end evaluate_expression_l695_695265


namespace number_of_solutions_l695_695974

theorem number_of_solutions :
  {x : ℕ // x > 1 ∧ x^4 + x^3 + 4*x^2 - 60 < 500} = {2, 3, 4} :=
by
  sorry

end number_of_solutions_l695_695974


namespace real_and_imaginary_numbers_intersection_is_empty_l695_695825

-- Definitions for the sets
def C : Set ℂ := {z : ℂ | True}
def R : Set ℂ := {z : ℂ | ∃ (a : ℝ), z = a}
def I : Set ℂ := {z : ℂ | ∃ (b : ℝ), z = complex.I * b}

-- To be proven:
theorem real_and_imaginary_numbers_intersection_is_empty :
  R ∩ I = ∅ :=
by
  sorry

end real_and_imaginary_numbers_intersection_is_empty_l695_695825


namespace maximum_distance_PQ_probability_minor_arc_AB_l695_695770

open Real

-- Define the given conditions
def line (x : ℝ) : ℝ := (sqrt 3 / 3) * x
def circle_center : ℝ × ℝ := (2, 0)
def circle_radius : ℝ := 2
def distance_AB : ℝ := 2 * sqrt 3
def point_P : ℝ × ℝ := (-1, sqrt 7)

-- Function to calculate distance between two points (x1, y1) and (x2, y2)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Maximum distance PQ equivalent proof statement
theorem maximum_distance_PQ : 
  ∀ Q : ℝ × ℝ, distance circle_center Q = circle_radius → distance point_P Q ≤ 6 :=
by sorry

-- Probability of M lying on minor arc AB equivalent proof statement
theorem probability_minor_arc_AB : 
  let angle_AOB := 120 in
  (angle_AOB.toReal / 360) = 1 / 3 :=
by sorry

end maximum_distance_PQ_probability_minor_arc_AB_l695_695770


namespace polygon_sides_l695_695771

theorem polygon_sides (n : ℕ) (sum_of_angles : ℕ) (missing_angle : ℕ) 
  (h1 : sum_of_angles = 3240) 
  (h2 : missing_angle * n / (n - 1) = 2 * sum_of_angles) : 
  n = 20 := 
sorry

end polygon_sides_l695_695771


namespace inscribed_circle_locus_l695_695813

theorem inscribed_circle_locus {A B C : Point} {R r : ℝ} (hR : R > r)
  (h_tangent : tangent smaller_circle at_point A to_larger_circle at_points B C)
  : ∀ P, is_center_of_inscribed_circle_in_triangle_ABC P ↔ forms_circle_touches_given_circles_at_A P :=
by sorry

end inscribed_circle_locus_l695_695813


namespace probability_neither_square_nor_cube_l695_695547

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695547


namespace water_spill_volume_when_tilted_by_30_deg_angle_to_spill_half_water_l695_695678

noncomputable def tan (angle : ℝ) : ℝ := Real.tan angle
noncomputable def atan (x : ℝ) : ℝ := Real.atan x

constant radius : ℝ := 10
constant height : ℝ := 25
constant initial_angle : ℝ := Real.pi / 6 -- 30 degrees in radians

def tilted_drop := radius * 2 * tan initial_angle / Real.sqrt 3

def volume_of_wedge := (radius ^ 2) * Real.pi * tilted_drop / sqrt 3

def half_full_angle := atan (height / (radius * 2))

theorem water_spill_volume_when_tilted_by_30_deg :
    volume_of_wedge = 1000 / sqrt 3 * Real.pi :=
sorry

theorem angle_to_spill_half_water :
    half_full_angle = atan (5 / 4) :=
sorry

end water_spill_volume_when_tilted_by_30_deg_angle_to_spill_half_water_l695_695678


namespace average_visitors_per_day_l695_695637

theorem average_visitors_per_day (total_days : ℕ) (sundays : ℕ) (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (starts_with_sunday : total_days = 30 ∧ sundays = 5 ∧ sunday_visitors = 1000 ∧ other_day_visitors = 700) : 
  (5000 + 700 * (total_days - sundays)) / total_days = 750 :=
by
  rw [starts_with_sunday.1, starts_with_sunday.2.1, starts_with_sunday.2.2.1, starts_with_sunday.2.2.2]
  sorry

end average_visitors_per_day_l695_695637


namespace equidistant_planes_spheres_l695_695389

variable (A B C D E : Point)

/- 
Assume the following conditions:
- The points A, B, C, D, and E do not all lie on the same plane.
- The points A, B, C, D, and E do not all lie on the same sphere.
-/

/-- There are exactly 15 planes or spheres equidistant from the given points A, B, C, D, and E. -/
theorem equidistant_planes_spheres : 
  (¬ collinear {A, B, C, D}) ∧ (¬ cocircular {A, B, C, D, E}) → 
  ∃! S : Set Point, equidistant S {A, B, C, D, E} := 
sorry

end equidistant_planes_spheres_l695_695389


namespace problem1_problem2_problem3_l695_695308

noncomputable def A : set ℝ := {x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def B (m : ℝ) : set ℝ := {x | x^2 - 2*m*x + m^2 - 9 ≤ 0 }
noncomputable def C (b : ℝ) : set ℝ := { y | ∃ x : ℝ, y = 2^x + b }

-- Problem 1
theorem problem1 (m : ℝ) (h : A ∩ B m = Icc 0 4) : m = 3 := sorry

-- Problem 2
theorem problem2 (b : ℝ) (h : A ∩ C b = ∅) : 4 ≤ b := sorry

-- Problem 3
theorem problem3 (m : ℝ) (h : A ∪ B m = B m) : 1 ≤ m ∧ m ≤ 2 := sorry

end problem1_problem2_problem3_l695_695308


namespace range_of_m_l695_695807

theorem range_of_m (h : ¬ ∃ (x : ℝ), 1 < x ∧ x < 3 ∧ x^2 - m * x - 1 = 0) :
  m ∈ (-∞, 0] ∪ [8 / 3, ∞) := 
sorry

end range_of_m_l695_695807


namespace range_of_x_satisfying_f_l695_695782

variable {f : ℝ → ℝ}
variable (hf : ∀ x y : ℝ, x < y → f x ≥ f y)

theorem range_of_x_satisfying_f (x : ℝ) : f (x^2 - 2 * x) < f 3 ↔ x ∈ set.Iio (-1) ∪ set.Ioi 3 :=
by
  sorry

end range_of_x_satisfying_f_l695_695782


namespace trigonometric_identity_l695_695916

open Real

theorem trigonometric_identity :
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  4 * cos_18 ^ 2 - 1 = 1 / (4 * sin_18 ^ 2) :=
by
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  sorry

end trigonometric_identity_l695_695916


namespace SmallestPositiveAngle_l695_695739

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l695_695739


namespace sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l695_695599

variable (p q : ℝ) (x1 x2 : ℝ)

-- Define the condition: Roots of the quadratic equation
def quadratic_equation_condition : Prop :=
  x1^2 + p * x1 + q = 0 ∧ x2^2 + p * x2 + q = 0

-- Define the identities for calculations based on properties of roots
def properties_of_roots : Prop :=
  x1 + x2 = -p ∧ x1 * x2 = q

-- First proof problem
theorem sum_of_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                           (h2 : properties_of_roots p q x1 x2) :
  1 / x1 + 1 / x2 = -p / q := 
by sorry

-- Second proof problem
theorem sum_of_square_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                  (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^2) + 1 / (x2^2) = (p^2 - 2*q) / (q^2) := 
by sorry

-- Third proof problem
theorem sum_of_cubic_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                 (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^3) + 1 / (x2^3) = p * (3*q - p^2) / (q^3) := 
by sorry

end sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l695_695599


namespace women_left_room_is_3_l695_695402

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l695_695402


namespace number_exceeds_twenty_percent_by_forty_l695_695978

theorem number_exceeds_twenty_percent_by_forty (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 :=
by
  sorry

end number_exceeds_twenty_percent_by_forty_l695_695978


namespace total_selling_price_l695_695820

theorem total_selling_price (total_cost : ℝ) (cost_loss_book : ℝ) (loss_percent : ℝ) 
  (gain_percent : ℝ) (selling_price_expected : ℝ) :
  total_cost = 300 →
  cost_loss_book = 175 →
  loss_percent = 0.15 →
  gain_percent = 0.19 →
  selling_price_expected = 297.50 →
  let cost_gain_book := total_cost - cost_loss_book in
  let selling_price_loss_book := cost_loss_book - (loss_percent * cost_loss_book) in
  let selling_price_gain_book := cost_gain_book + (gain_percent * cost_gain_book) in
  selling_price_loss_book + selling_price_gain_book = selling_price_expected :=
by
  intros h1 h2 h3 h4 h5
  let cost_gain_book := total_cost - cost_loss_book
  let selling_price_loss_book := cost_loss_book - (loss_percent * cost_loss_book)
  let selling_price_gain_book := cost_gain_book + (gain_percent * cost_gain_book)
  have h6 : cost_gain_book = 125, from sorry
  have h7 : selling_price_loss_book = 148.75, from sorry
  have h8 : selling_price_gain_book = 148.75, from sorry
  exact sorry

end total_selling_price_l695_695820


namespace smallest_positive_angle_l695_695730

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l695_695730


namespace cannonball_maximum_height_l695_695997

def height_function (t : ℝ) := -20 * t^2 + 100 * t + 36

theorem cannonball_maximum_height :
  ∃ t₀ : ℝ, ∀ t : ℝ, height_function t ≤ height_function t₀ ∧ height_function t₀ = 161 :=
by
  sorry

end cannonball_maximum_height_l695_695997


namespace find_XY_length_in_30_60_90_triangle_l695_695859

noncomputable theory

def length_of_adjacent_side_30_60_90 (YZ : ℝ) (h1 : YZ = 12) : ℝ :=
    let XY := 12 * Real.sqrt 3 in
    XY

theorem find_XY_length_in_30_60_90_triangle (YZ : ℝ) (hYZ : YZ = 12) : 
    length_of_adjacent_side_30_60_90 YZ hYZ = 12 * Real.sqrt 3 :=
by
    unfold length_of_adjacent_side_30_60_90
    rw hYZ
    simp
    sorry

end find_XY_length_in_30_60_90_triangle_l695_695859


namespace quadratic_real_roots_m_range_l695_695836

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l695_695836


namespace collinear_GFE_l695_695453

/-- Pentagon ABCDE is inscribed in a circle. Its diagonals AC and BD intersect at F. -/
variables (A B C D E F G H I J : Point)
variables (circle : Circle A B C D E) (intersect_AC_BD : Intersects (Line A C) (Line B D) F)
variables (bisector_BAC_CDB : AngleBisector (A B C) (C D B) G)
variables (intersect_AG_BD : Intersects (Line A G) (Line B D) H)
variables (intersect_DG_AC : Intersects (Line D G) (Line A C) I)
variables (intersect_EG_AD : Intersects (Line E G) (Line A D) J)

/-- The quadrilateral FHGI is cyclic. -/
variable (cyclic_FHGI : CyclicQuadrilateral F H G I)

/-- The product equality condition. -/
variable (product_equality : (Distance J A) * (Distance F C) * (Distance G H) 
                             = (Distance J D) * (Distance F B) * (Distance G I))

theorem collinear_GFE : Collinear G F E := 
by
  sorry

end collinear_GFE_l695_695453


namespace solve_sine_problem_l695_695763

theorem solve_sine_problem (α : ℝ)
  (h : sin (α + π / 3) - sin α = 4 / 5) :
  sin (2 * α - π / 6) = -7 / 25 :=
by
  sorry

end solve_sine_problem_l695_695763


namespace probability_neither_perfect_square_nor_cube_l695_695524

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695524


namespace probability_neither_perfect_square_nor_cube_l695_695577

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695577


namespace solve_for_r_l695_695098

theorem solve_for_r (r : ℚ) (h : 8 = 2^(3 * r + 2)) : r = 1 / 3 :=
by
  sorry

end solve_for_r_l695_695098


namespace values_of_x_for_g_l695_695049

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l695_695049


namespace sin_A_eq_one_half_l695_695375

theorem sin_A_eq_one_half (a b : ℝ) (sin_B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_B = 2/3) : 
  ∃ (sin_A : ℝ), sin_A = 1/2 := 
by
  let sin_A := a * sin_B / b
  existsi sin_A
  sorry

end sin_A_eq_one_half_l695_695375


namespace point_not_on_graph_l695_695160

-- Define the function y = (x - 1) / (x + 2)
def f (x : ℝ) : Option ℝ :=
  if x + 2 = 0 then none else some ((x - 1) / (x + 2))

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1/2)
def C : ℝ × ℝ := (3, 1/2)
def D : ℝ × ℝ := (2, 3)
def E : ℝ × ℝ := (-2, 5)

-- The proof statement
theorem point_not_on_graph : ∀ (y : ℝ), y ≠ f (E.1) :=
by
  intro y
  have h : f E.1 = none := by rfl  -- f(-2) = none since denominator is zero
  intro hyp
  have : some y = none := by rw [←hyp, h]
  contradiction

end point_not_on_graph_l695_695160


namespace general_formula_and_min_n_l695_695793

namespace ArithmeticSequence

theorem general_formula_and_min_n (a : ℕ → ℤ) (c : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ) :
  (∀ n : ℕ, a n = 3 + (n - 1) * (-2)) ∧ 
  (∀ n : ℕ, c n = (5 - a n) / 2) ∧ 
  (∀ n : ℕ, b n = 2 ^ (c n)) ∧
  (∀ n : ℕ, T n = (finset.range n).sum (λ k => 1 + k)) →
  (T 63 ≥ 2016 ∧ (∀ m < 63, T m < 2016)) :=
by sorry

end ArithmeticSequence

end general_formula_and_min_n_l695_695793


namespace num_ways_award_medals_l695_695941

-- There are 8 sprinters in total
def num_sprinters : ℕ := 8

-- Three of the sprinters are Americans
def num_americans : ℕ := 3

-- The number of non-American sprinters
def num_non_americans : ℕ := num_sprinters - num_americans

-- The question to prove: the number of ways the medals can be awarded if at most one American gets a medal
theorem num_ways_award_medals 
  (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = num_sprinters) (h2 : m = num_americans) 
  (h3 : k = num_non_americans) 
  (no_american : ℕ := k * (k - 1) * (k - 2)) 
  (one_american : ℕ := m * 3 * k * (k - 1)) 
  : no_american + one_american = 240 :=
sorry

end num_ways_award_medals_l695_695941


namespace probability_neither_perfect_square_nor_cube_l695_695521

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695521


namespace bridge_length_l695_695977

-- Conditions
def walking_speed : ℝ := 10 -- km/hr
def crossing_time_minutes : ℝ := 15 -- minutes
def crossing_time_hours : ℝ := crossing_time_minutes / 60 -- converting minutes to hours

-- Proof problem: Prove the length of the bridge is 2.5 kilometers
theorem bridge_length : walking_speed * crossing_time_hours = 2.5 := by
  sorry

end bridge_length_l695_695977


namespace jonas_tshirts_count_l695_695032

def pairs_to_individuals (pairs : Nat) : Nat := pairs * 2

variable (num_pairs_socks : Nat := 20)
variable (num_pairs_shoes : Nat := 5)
variable (num_pairs_pants : Nat := 10)
variable (num_additional_pairs_socks : Nat := 35)

def total_individual_items_without_tshirts : Nat :=
  pairs_to_individuals num_pairs_socks +
  pairs_to_individuals num_pairs_shoes +
  pairs_to_individuals num_pairs_pants

def total_individual_items_desired : Nat :=
  total_individual_items_without_tshirts +
  pairs_to_individuals num_additional_pairs_socks

def tshirts_jonas_needs : Nat :=
  total_individual_items_desired - total_individual_items_without_tshirts

theorem jonas_tshirts_count : tshirts_jonas_needs = 70 := by
  sorry

end jonas_tshirts_count_l695_695032


namespace fraction_product_l695_695131

theorem fraction_product (a b : ℕ) 
  (h1 : 1/5 < a / b)
  (h2 : a / b < 1/4)
  (h3 : b ≤ 19) :
  ∃ a1 a2 b1 b2, 4 * a2 < b1 ∧ b1 < 5 * a2 ∧ b2 ≤ 19 ∧ 4 * a2 < b2 ∧ b2 < 20 ∧ a = 4 ∧ b = 19 ∧ a1 = 2 ∧ b1 = 9 ∧ 
  (a + b = 23 ∨ a + b = 11) ∧ (23 * 11 = 253) := by
  sorry

end fraction_product_l695_695131


namespace vertical_asymptote_x_value_l695_695757

theorem vertical_asymptote_x_value (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 :=
by
  sorry

end vertical_asymptote_x_value_l695_695757


namespace find_x_l695_695106

theorem find_x 
  (x y z : ℝ)
  (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
  (h2 : y + z = 110) 
  : x = 106 := 
by 
  sorry

end find_x_l695_695106


namespace probability_not_square_or_cube_l695_695548

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695548


namespace probability_neither_perfect_square_nor_cube_l695_695580

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695580


namespace tan_half_odd_and_periodic_l695_695515

-- Definition of the function
def f (x : ℝ) : ℝ := Real.tan (x / 2)

-- Proof statement (without the actual proof)
theorem tan_half_odd_and_periodic :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 2 * π) = f x) := by
  sorry

end tan_half_odd_and_periodic_l695_695515


namespace problem_part_1_problem_part_2_problem_part_3_l695_695287

open Nat

-- Define the problem conditions and prove the required results
theorem problem_part_1 (n : ℕ) (h : ∀ k, n = k →  2 * binomial k 1 = (1 / 5) * 2^2 * binomial k 2) : n = 6 :=
by
  sorry

theorem problem_part_2 : binomial 6 3 * 2^3  = 160 :=
by
  sorry

theorem problem_part_3 (a : Fin 7 → ℕ) 
    (h : (∑ i in Finset.range 7, a i * (1 ^ (6 - i))) = 2^6) : 
    ∑ i in Finset.range 7, a i = 64 :=
by
  sorry

end problem_part_1_problem_part_2_problem_part_3_l695_695287


namespace max_value_3ab_sqrt2_plus_9bc_l695_695043

def a_nonnegative (a : ℝ) := 0 ≤ a
def b_nonnegative (b : ℝ) := 0 ≤ b
def c_nonnegative (c : ℝ) := 0 ≤ c
def sum_of_squares_eq_three (a b c : ℝ) := a^2 + b^2 + c^2 = 3

theorem max_value_3ab_sqrt2_plus_9bc (a b c : ℝ) (ha : a_nonnegative a) (hb : b_nonnegative b) (hc : c_nonnegative c) (hsum : sum_of_squares_eq_three a b c) : 
  3 * a * b * real.sqrt 2 + 9 * b * c ≤ 3 * real.sqrt 11 :=
sorry

end max_value_3ab_sqrt2_plus_9bc_l695_695043


namespace probability_C_value_l695_695651

axiom probability_A : ℚ := 1/4
axiom probability_B : ℚ := 1/3
axiom probability_D : ℚ := 1/6
axiom total_probability : (probability_A + probability_B + probability_C + probability_D) = 1

theorem probability_C_value : probability_C = 1/4 :=
by
  sorry

end probability_C_value_l695_695651


namespace race_distance_l695_695680

variable (distance : ℝ)

theorem race_distance :
  (0.25 * distance = 50) → (distance = 200) :=
by
  intro h
  sorry

end race_distance_l695_695680


namespace FQ_equals_6_l695_695243

noncomputable def FQ_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85) : ℝ :=
  let EF := Real.sqrt(DF^2 - DE^2) in
  EF

theorem FQ_equals_6 (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85) :
  FQ_length DE DF h1 h2 = 6 := by
  let EF := Real.sqrt (DF^2 - DE^2)
  have h3 : EF = 6
  { 
    have h4 : DE^2 + EF^2 = DF^2 := by sorry
    rw [h1, h2] at h4
    exact Real.sqrt (85 - 49) -- this calculates EF = 6.
  }
  rw [h3]
  sorry

end FQ_equals_6_l695_695243


namespace rain_on_both_days_l695_695162

-- Define the events probabilities
variables (P_M P_T P_N P_MT : ℝ)

-- Define the initial conditions
axiom h1 : P_M = 0.6
axiom h2 : P_T = 0.55
axiom h3 : P_N = 0.25

-- Define the statement to prove
theorem rain_on_both_days : P_MT = 0.4 :=
by
  -- The proof is omitted for now
  sorry

end rain_on_both_days_l695_695162


namespace probability_non_square_non_cube_l695_695584

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695584


namespace cloth_woven_approx_15_l695_695654

-- Define the rate of weaving as per condition 1
def rate_of_weaving : ℝ := 0.127

-- Define the time taken as per condition 2
def time_taken : ℝ := 118.11

-- The amount of cloth woven is the product of the rate of weaving and time taken
-- We prove that this is approximately equal to 15 meters
theorem cloth_woven_approx_15 : rate_of_weaving * time_taken ≈ 15 :=
by
  -- The product of rate_of_weaving and time_taken should be around 15, incorporate the mathematical computation here
  have h : rate_of_weaving * time_taken = 15.00097 := by norm_num
  -- Given norm_num's computation we say this is approximately 15
  sorry

end cloth_woven_approx_15_l695_695654


namespace values_of_x_for_g_l695_695048

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l695_695048


namespace minimum_clerks_l695_695110

theorem minimum_clerks :
  (3 * (nat.choose n 3) = nat.choose n 4) → (n ≥ 4) → n = 15 :=
by
  intros h₁ h₂
  sorry

end minimum_clerks_l695_695110


namespace projection_of_a_onto_b_correct_l695_695321

noncomputable def projection_of_a_onto_b
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : (bit0 1 • a + bit1 1 • b) ⬝ b = 0) 
  (h2 : ‖b‖ = 2 * Real.sqrt 2) : ℝ :=
(λ (a b : EuclideanSpace ℝ (Fin 3)) (h1 : (bit0 1 • a + bit1 1 • b) ⬝ b = 0) (h2 : ‖b‖ = 2 * Real.sqrt 2), -3 * Real.sqrt 2) a b h1 h2 

theorem projection_of_a_onto_b_correct
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : (bit0 1 • a + bit1 1 • b) ⬝ b = 0) 
  (h2 : ‖b‖ = 2 * Real.sqrt 2) :
  projection_of_a_onto_b a b h1 h2 = -3 * Real.sqrt 2 :=
sorry

end projection_of_a_onto_b_correct_l695_695321


namespace kendall_total_distance_l695_695033

def distance_mother : ℝ := 0.17
def distance_father : ℝ := 0.5
def distance_friend : ℝ := 0.68
def mile_to_km : ℝ := 1.60934

def total_distance_miles : ℝ := distance_mother + distance_father + distance_friend
def total_distance_km : ℝ := total_distance_miles * mile_to_km

theorem kendall_total_distance :
  total_distance_km ≈ 2.17 := by
  sorry

end kendall_total_distance_l695_695033


namespace player_A_loses_l695_695617

-- Define a function to check if a given position is a losing position
def is_losing_position (n : Nat) : Prop :=
  n % 7 == 0 ∨ n % 7 == 1

-- The number of chips at the start
def initial_chips : Nat := 2016

-- Main theorem stating the losing condition for player A (first player)
theorem player_A_loses : is_losing_position initial_chips :=
  by
    -- Since 2016 modulo 7 equals 0
    have h : initial_chips % 7 = 0 := Nat.mod_eq_zero_of_dvd (by norm_num : 7 ∣ 2016)
    exact Or.inl h

end player_A_loses_l695_695617


namespace distance_squared_from_B_to_origin_l695_695010

-- Conditions:
-- 1. the radius of the circle is 10 cm
-- 2. the length of AB is 8 cm
-- 3. the length of BC is 3 cm
-- 4. the angle ABC is a right angle
-- 5. the center of the circle is at the origin
-- a^2 + b^2 is the square of the distance from B to the center of the circle (origin)

theorem distance_squared_from_B_to_origin
  (a b : ℝ)
  (h1 : a^2 + (b + 8)^2 = 100)
  (h2 : (a + 3)^2 + b^2 = 100)
  (h3 : 6 * a - 16 * b = 55) : a^2 + b^2 = 50 :=
sorry

end distance_squared_from_B_to_origin_l695_695010


namespace extreme_points_inequality_l695_695324

noncomputable def f (a x : ℝ) : ℝ := a * x - (a / x) - 2 * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := (a * x^2 - 2 * x + a) / x^2

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : a > 0) (h2 : 1 < x1) (h3 : x1 < Real.exp 1)
  (h4 : f a x1 = 0) (h5 : f a x2 = 0) (h6 : x1 ≠ x2) : |f a x1 - f a x2| < 1 :=
by
  sorry

end extreme_points_inequality_l695_695324


namespace probability_non_square_non_cube_l695_695586

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695586


namespace T_n_sum_l695_695795

section
variables {α : Type*} (a b : ℕ → ℕ)

-- Conditions
def S (n : ℕ) : ℕ := 2 * a n - 2
def a2_eq_b3 : b 3 = a 2 := rfl
def b2_b6_sum : b 2 + b 6 = 10 := rfl

-- General terms of the sequences
def a_n_general (n : ℕ) : ℕ := 2^n
def b_n_general (n : ℕ) : ℕ := n + 1

-- Problem to prove
def T_n (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6

theorem T_n_sum (n : ℕ) (hS : ∀ n, S n = 2 * (a n) - 2)
                (hb3 : b 3 = a 2)
                (hb2b6 : b 2 + b 6 = 10) :
  T_n n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry
end

end T_n_sum_l695_695795


namespace sum_of_reciprocals_l695_695134

-- Define the conditions formally
def total_students : ℕ := 1000
def num_classes : ℕ := 35

-- Class sizes as a list of natural numbers
variable (class_sizes : Fin num_classes → ℕ)
-- Constraint ensuring the total number of students across all classes
axiom class_sizes_sum : ∑ i, class_sizes i = total_students

-- Main theorem stating that the sum of the reciprocals of class sizes is 35
theorem sum_of_reciprocals (class_sizes_proper : ∀ i, class_sizes i > 0):
  (∑ i, (1 : ℝ) / (class_sizes i)) = 35 := by
  sorry

end sum_of_reciprocals_l695_695134


namespace exists_coprime_positive_sum_le_m_l695_695636

theorem exists_coprime_positive_sum_le_m (m : ℕ) (a b : ℤ) 
  (ha : 0 < a) (hb : 0 < b) (hcoprime : Int.gcd a b = 1)
  (h1 : a ∣ (m + b^2)) (h2 : b ∣ (m + a^2)) 
  : ∃ a' b', 0 < a' ∧ 0 < b' ∧ Int.gcd a' b' = 1 ∧ a' ∣ (m + b'^2) ∧ b' ∣ (m + a'^2) ∧ a' + b' ≤ m + 1 :=
by
  sorry

end exists_coprime_positive_sum_le_m_l695_695636


namespace gergonne_point_concurrency_l695_695886

noncomputable def are_concurrent (A B C D E F : Point) (triangle_prop : TriangleProp (A B C)) 
  (tangent_D : Tangent D (Segment B C) (Incircle (A B C)))
  (tangent_E : Tangent E (Segment C A) (Incircle (A B C)))
  (tangent_F : Tangent F (Segment A B) (Incircle (A B C))) : Prop :=
  Ceva (Line A D) (Line B E) (Line C F)

theorem gergonne_point_concurrency (A B C D E F : Point) 
  (triangle_prop : TriangleProp (A B C)) 
  (tangent_D : Tangent D (Segment B C) (Incircle (A B C)))
  (tangent_E : Tangent E (Segment C A) (Incircle (A B C)))
  (tangent_F : Tangent F (Segment A B) (Incircle (A B C))) :
  are_concurrent A B C D E F triangle_prop tangent_D tangent_E tangent_F :=
sorry

end gergonne_point_concurrency_l695_695886


namespace wrench_weight_relation_l695_695639

variables (h w : ℕ)

theorem wrench_weight_relation (h w : ℕ) 
  (cond : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) : w = 2 * h := 
by sorry

end wrench_weight_relation_l695_695639


namespace average_marks_l695_695165

-- Given conditions
variables (M P C : ℝ)
variables (h1 : M + P = 32) (h2 : C = P + 20)

-- Statement to be proved
theorem average_marks : (M + C) / 2 = 26 :=
by
  -- The proof will be inserted here
  sorry

end average_marks_l695_695165


namespace presidency_meeting_arrangements_l695_695659

theorem presidency_meeting_arrangements : 
  let total_ways := 3 * (nat.choose 6 3) * (nat.choose 6 1) * (nat.choose 6 1) in 
  total_ways = 2160 := 
by
  sorry

end presidency_meeting_arrangements_l695_695659


namespace derivative_of_f_2_derivative_of_f_1_l695_695327

def f (x : ℝ) : ℝ := x^2 - 3 * x - 1

theorem derivative_of_f_2 : deriv (λ x, f 2) 0 := 
sorry

theorem derivative_of_f_1 : deriv f 1 = -1 :=
sorry

end derivative_of_f_2_derivative_of_f_1_l695_695327


namespace distinct_sets_L_l695_695101

universe u
variables {V : Type u} [fintype V] (G : simple_graph V) [decidable_rel G.adj] [is_complete_graph G]

noncomputable def L (u v : V) : set V :=
{u, v} ∪ {w | ∃ r b, r ≠ u ∧ r ≠ v ∧ b ≠ u ∧ b ≠ v ∧ w = r ∧ {(u, v), (u, r), (v, r)} ⊆ (G.edge_set \ {(u, v), (u, r), (v, r)})}

theorem distinct_sets_L (hG : G.order = 2015) (h_coloring : ∀ e, G.edge_coloring e) :
  ∃ S : finset (set V), S.card ≥ 120 ∧ (∀ x ∈ S, ∃ u v, x = L u v) :=
sorry

end distinct_sets_L_l695_695101


namespace minimum_value_2_only_in_option_b_l695_695217

noncomputable def option_a (x : ℝ) : ℝ := x + 1 / x
noncomputable def option_b (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def option_c (x : ℝ) : ℝ := (Real.log x) + 1 / (Real.log x)
noncomputable def option_d (x : ℝ) : ℝ := (Real.sin x) + 1 / (Real.sin x)

theorem minimum_value_2_only_in_option_b :
  (∀ x > 0, option_a x ≠ 2) ∧
  (∃ x, option_b x = 2) ∧
  (∀ x (h: 0 < x) (h' : x < 1), option_c x ≠ 2) ∧
  (∀ x (h: 0 < x) (h' : x < π / 2), option_d x ≠ 2) :=
by
  sorry

end minimum_value_2_only_in_option_b_l695_695217


namespace second_polygon_sides_l695_695953

theorem second_polygon_sides (a b n m : ℕ) (s : ℝ) 
  (h1 : a = 45) 
  (h2 : b = 3 * s)
  (h3 : n * b = m * s)
  (h4 : n = 45) : m = 135 := 
by
  sorry

end second_polygon_sides_l695_695953


namespace smallest_positive_angle_l695_695732

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l695_695732


namespace total_rainfall_over_3_days_l695_695409

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l695_695409


namespace order_of_logs_l695_695286

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem order_of_logs : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l695_695286


namespace shooter_hit_rate_l695_695758

theorem shooter_hit_rate :
  ∀ (p : ℝ),
  (1 - (1 - p)^4 = 80 / 81) → p = 2 / 3 :=
by
  intros p h,
  sorry

end shooter_hit_rate_l695_695758


namespace inequality_sol_set_a_eq_2_inequality_sol_set_general_l695_695335

theorem inequality_sol_set_a_eq_2 :
  ∀ x : ℝ, (x^2 - x + 2 - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem inequality_sol_set_general (a : ℝ) :
  (∀ x : ℝ, (x^2 - x + a - a^2 ≤ 0) ↔
    (if a < 1/2 then a ≤ x ∧ x ≤ 1 - a
    else if a > 1/2 then 1 - a ≤ x ∧ x ≤ a
    else x = 1/2)) :=
by sorry

end inequality_sol_set_a_eq_2_inequality_sol_set_general_l695_695335


namespace median_list_sum_of_squares_l695_695387

theorem median_list_sum_of_squares : 
  let N := (100 * 101 * 201) / 6
  let position1 := N / 2
  let position2 := N / 2 + 1
  let cumulative_count := fun n => (n * (n + 1) * (2 * n + 1)) / 6
  ∃ n : ℕ, (1 ≤ n ∧ n ≤ 100) ∧ (cumulative_count 71 < position1) ∧ (position1 ≤ cumulative_count 72) ∧
    (cumulative_count 71 < position2) ∧ (position2 <= cumulative_count 72) ∧
    median = 72 :=
by
  let N := (100 * 101 * 201) / 6
  let position1 := N / 2
  let position2 := N / 2 + 1
  let cumulative_count := fun n => (n * (n + 1) * (2 * n + 1)) / 6
  use 72
  have hc1 : 1 ≤ 72 ∧ 72 ≤ 100 := by decide
  have hc2 : cumulative_count 71 < position1 := by decide
  have hc3 : position1 ≤ cumulative_count 72 := by decide
  have hc4 : cumulative_count 71 < position2 := by decide
  have hc5 : position2 <= cumulative_count 72 := by decide
  exact ⟨hc1, hc2, hc3, hc4, hc5, rfl⟩
  sorry

end median_list_sum_of_squares_l695_695387


namespace find_a_from_constant_term_l695_695368

-- Given conditions
def term_in_expansion (a : ℝ) : ℝ :=
  let expansion_term := (ax + 1) * (2x - 1 / x)^5
  -- Assuming the specific manner it is interpreted in the problem,
  -- We take the constant term from the expansion as -40
  if ... then -40 else sorry

-- Problem statement
theorem find_a_from_constant_term (h : term_in_expansion a = -40) : a = 1 :=
by
  sorry

end find_a_from_constant_term_l695_695368


namespace even_function_B_l695_695968

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

def fA (x : ℝ) : ℝ := x^2 * sin x
def fB (x : ℝ) : ℝ := x^2 * cos x
def fC (x : ℝ) : ℝ := |real.log x|
def fD (x : ℝ) : ℝ := 2^(-x)

theorem even_function_B : is_even fB :=
by
  intros x
  dsimp [fB]
  rw [cos_neg]
  sorry

end even_function_B_l695_695968


namespace sum_reciprocal_k_ge_one_l695_695432

variables {Z : Type} [AddGroup Z] [DecidableEq Z] [Fintype Z]

def k_set (A : Set Z) (k : ℕ) : Prop :=
  ∃ (x : Fin k → Z), ∀ i j, i ≠ j → (x i +ˢ A) ∩ (x j +ˢ A) = ∅

def translates (x : Z) (A : Set Z) : Set Z := 
  {z | ∃ a ∈ A, z = x + a}

theorem sum_reciprocal_k_ge_one {t : ℕ}
  (A : Fin t → Set Z) (k : Fin t → ℕ)
  (hk : ∀ i, k_set (A i) (k i))
  (hA : (⋃ i, A i) = Set.univ) :
  (∑ i, 1 / (k i : ℝ)) ≥ 1 :=
by
  sorry

end sum_reciprocal_k_ge_one_l695_695432


namespace sum_first_10_common_terms_l695_695750

def AP_term (n : ℕ) : ℕ := 5 + 3 * n
def GP_term (k : ℕ) : ℕ := 10 * 2^k
def common_terms (m : ℕ) : ℕ := 20 * 4^m

theorem sum_first_10_common_terms :
  ∑ m in Finset.range 10, common_terms m = 6990500 :=
by sorry

end sum_first_10_common_terms_l695_695750


namespace example_table_tennis_probability_l695_695600

def table_tennis_game (A_serves_first : Bool) (score_probability_A : ℝ) 
  (independent_serves : ∀ n m : ℕ, A n ≠ A m → Independent (events! n) (events! m))
  (serve_points_A : Fin 4 → Fin 5)
  (serve_points_B : Fin 4 → Fin 5) : Prop :=
P(A₀) = 0.16 ∧
P(A₁) = 0.48 ∧
P(B₀) = 0.36 ∧
P(B₁) = 0.48 ∧
P(B₂) = 0.16 ∧
P(A₂) = 0.36 ∧
P(B) = 0.352 ∧
P(C) = 0.3072

theorem example_table_tennis_probability :
  ∀ (prob_A : ℝ) (prob_B : ℝ) (score_A : ℕ) (score_B : ℕ),
  table_tennis_game true 0.6
  (λ n m h, by sorry)
  (λ x, by sorry)
  (λ x, by sorry) :=
by sorry

end example_table_tennis_probability_l695_695600


namespace company_profit_l695_695184

theorem company_profit: ∃ (n : ℕ), 600 + 200 * n < 280 * n :=
begin
  sorry,
end

end company_profit_l695_695184


namespace cubic_polynomial_distinct_integer_roots_l695_695695

theorem cubic_polynomial_distinct_integer_roots (b c d : ℤ) :
  ∃ (m : ℕ), m ≤ 3 ∧ set.finite { x : ℤ | x^3 + b * x^2 + c * x + d = 0} ∧ 
    set.card ({ x : ℤ | x^3 + b * x^2 + c * x + d = 0}.to_finset) = m := by
sorry

end cubic_polynomial_distinct_integer_roots_l695_695695


namespace roots_are_imaginary_l695_695279

theorem roots_are_imaginary (k : ℚ) (h : 2 * k^2 - 1 = 8) : 
  let a := (2 * k^2) - 1
  let Δ := (-(3 * k)) ^ 2 - 4 * (1 : ℚ) * (2 * k^2 - 1)
in Δ < 0 :=
by 
  let a := (2 * k^2) - 1
  let Δ := (-(3 * k)) ^ 2 - 4 * (1 : ℚ) * (2 * k^2 - 1)
  sorry

end roots_are_imaginary_l695_695279


namespace probability_neither_square_nor_cube_l695_695561

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695561


namespace common_chord_circumcircles_passes_through_F_l695_695501

-- Definitions based on problem conditions
variables (S1 S2 : Circle)
variables (F A B C D E : Point)
variables (l : Line)

-- Conditions
axiom circles_touch (F : Point) : touches S1 S2 F
axiom common_tangent (l : Line) (A B : Point) : tangent l S1 A ∧ tangent l S2 B
axiom parallel_line (AB : Line) (C D E : Point) : parallel AB (line_through C D E) ∧ tangent (line_through C) S2 C ∧ intersect (line_through C) S1 D E

-- Proof Goal
theorem common_chord_circumcircles_passes_through_F :
  ∃ (common_chord : Chord), passes_through common_chord F ∧
    circumcircle (triangle A B C) * circumcircle (triangle B D E) = common_chord :=
sorry

end common_chord_circumcircles_passes_through_F_l695_695501


namespace avoid_loss_maximize_profit_max_profit_per_unit_l695_695187

-- Definitions of the functions as per problem conditions
noncomputable def C (x : ℝ) : ℝ := 2 + x
noncomputable def R (x : ℝ) : ℝ := if x ≤ 4 then 4 * x - (1 / 2) * x^2 - (1 / 2) else 7.5
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Proof statements

-- 1. Range to avoid loss
theorem avoid_loss (x : ℝ) : 1 ≤ x ∧ x ≤ 5.5 ↔ L x ≥ 0 :=
by
  sorry

-- 2. Production to maximize profit
theorem maximize_profit (x : ℝ) : x = 3 ↔ ∀ y, L y ≤ L 3 :=
by
  sorry

-- 3. Maximum profit per unit selling price
theorem max_profit_per_unit (x : ℝ) : x = 3 ↔ (R 3 / 3 = 2.33) :=
by
  sorry

end avoid_loss_maximize_profit_max_profit_per_unit_l695_695187


namespace segment_longer_than_incircle_diameter_l695_695905

variable {A B C : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]

-- Define the triangle ABC and points on its sides
variable (triangle_ABC : Triangle ℝ A B C)
variable {A' B' C' : ℝ}
variable (on_side_A'C : SidesOfTriangle A' C (AB triangle_ABC))
variable (on_side_B'A : SidesOfTriangle B' A (BC triangle_ABC))
variable (on_side_C'B : SidesOfTriangle A' B (CA triangle_ABC))

-- Define the condition of the right angle
def right_angle_condition : Prop :=
  ∃ (a b c : ℝ), (VecAdd A B' = a) ∧ (VecAdd C' B = b) ∧ (angle A' C' B' = 90)

-- Define the radius of the incircle
variable (r : ℝ)
-- Define the inradius of triangle ABC
variable (inradius_triangle_ABC : Inradius triangle_ABC r)

-- Prove the required inequality
theorem segment_longer_than_incircle_diameter
  (h_right_angle : right_angle_condition)
  (h_inradius : inradius_triangle_ABC) :
  A'B' > 2 * r :=
sorry

end segment_longer_than_incircle_diameter_l695_695905


namespace fixed_point_f_fixed_point_g_fixed_point_h_l695_695827

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

theorem fixed_point_f : ∃ x : ℝ, f x = x := sorry

def g (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 3
def g_derivative (a b c x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem fixed_point_g 
  (a b c : ℝ) (a_zero : a = 0) : 
  ∃ x0 ∈ (set.Icc (1/2 : ℝ) 2), 
    (g a b c x0 = x0 ∧ g_derivative a b c x0 = x0) ∧ (5/4 ≤ b ∧ b ≤ 11) := 
sorry

def h (a b c x : ℝ) : ℝ := g_derivative a b c x

theorem fixed_point_h 
  (a b c : ℝ) (a_nonzero : a ≠ 0) 
  (m : ℝ) (pos_geometric_seq : ∃ q > 0, (∀ k : ℕ, function.iterate (λ x, h a b c x) k m) = q ^ k * m) : 
  ∃ x : ℝ, h a b c x = x := 
sorry

end fixed_point_f_fixed_point_g_fixed_point_h_l695_695827


namespace nth_term_arithmetic_progression_l695_695281

theorem nth_term_arithmetic_progression (r : ℕ) :
  (∑ k in Finset.range (r + 1), (2 * (k + 1)^2 + 3 * (k + 1))) - 
  (∑ k in Finset.range r, (2 * (k + 1)^2 + 3 * (k + 1))) = 4 * r + 1 :=
by
  sorry

end nth_term_arithmetic_progression_l695_695281


namespace find_C_l695_695907

noncomputable def point := (ℚ × ℚ)

def A : point := (-2, 4)
def B : point := (3, -1)
def D : point := (0, 2)
def ratio := (1, 2)

-- Formalize the declaration needed to articulate the proof related to the section formula
def section_formula (B : point) (ratio : ℚ × ℚ) (A : point) : point :=
  (((ratio.2 * B.1 + ratio.1 * A.1) / (ratio.2 + ratio.1)), 
   ((ratio.2 * B.2 + ratio.1 * A.2) / (ratio.2 + ratio.1)))

def expected_C : point := (4 / 3, 2 / 3)

theorem find_C : expected_C = section_formula B ratio D :=
sorry

end find_C_l695_695907


namespace SmallestPositiveAngle_l695_695738

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l695_695738


namespace probability_non_square_non_cube_l695_695588

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l695_695588


namespace cluster_example_l695_695003

def f1 (x : ℝ) : ℝ := 1/2 * sin (2 * x)
def f2 (x : ℝ) : ℝ := 2 * sin (x + π / 4)
def f3 (x : ℝ) : ℝ := 2 * sin (x + π / 3)
def f4 (x : ℝ) : ℝ := √2 * sin (2 * x) + 1

def sameCluster (f g : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, ∀ x : ℝ, f x = g (x + T)

theorem cluster_example : sameCluster f2 f3 :=
by
  sorry

end cluster_example_l695_695003


namespace ratio_N1N2N3_ABC_l695_695976

noncomputable def point := ℝ × ℝ
noncomputable def triangle_area (A B C : point) : ℝ :=
  0.5 * ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A))

theorem ratio_N1N2N3_ABC {A B C D E F N1 N2 N3 : point}
  (hD_midpoint : D = (fst B + fst C) / 2, (snd B + snd C) / 2)
  (hE_divides_AC : E = (2 * fst A + 3 * fst C) / 5, (2 * snd A + 3 * snd C) / 5)
  (hF_midpoint : F = (fst A + fst B) / 2, (snd A + snd B) / 2)
  (hN1_intersect : some geometric condition for N1)
  (hN2_intersect : some geometric condition for N2)
  (hN3_intersect : some geometric condition for N3) :
  triangle_area N1 N2 N3 = 0.5 * triangle_area A B C :=
sorry

end ratio_N1N2N3_ABC_l695_695976


namespace FAH_45_degrees_l695_695142

open Real

noncomputable def unit_square_conditions (A B C D E F G H : ℝ × ℝ) : Prop :=
  A = (0, 1) ∧
  B = (0, 0) ∧
  C = (1, 0) ∧
  D = (1, 1) ∧
  F = (1/4, 0) ∧
  (BF = 1/4) ∧
  (EF_parallel_AB : ∀ x y : ℝ × ℝ, x ≠ y → ((x.2 = y.2) ↔ (EF ∈ line_through x y))) ∧
  (GH_parallel_BC : ∀ x y : ℝ × ℝ, x ≠ y → ((x.1 = y.1) ↔ (GH ∈ line_through x y))) ∧
  (BF + DH = FH : ∀ x, ((BF.1 = 1/4) ∧ (DH.2 = x)) → 
    FH = sqrt((3/4)^2 + (1 - x)^2))

noncomputable def angle_FAH (A F H : ℝ × ℝ) : ℝ :=
  Real.arctan (abs (4 + 3 / 5) / (1 + (4 * (3 / 5))))

theorem FAH_45_degrees : ∃ (A B C D E F G H : ℝ × ℝ), unit_square_conditions A B C D E F G H ∧
  abs (angle_FAH A F H * 180 / π - 45) < 1 :=
begin
  sorry
end

end FAH_45_degrees_l695_695142


namespace assignment_increase_l695_695957

-- Define what an assignment statement is
def assignment_statement (lhs rhs : ℕ) : ℕ := rhs

-- Define the conditions and the problem
theorem assignment_increase (n : ℕ) : assignment_statement n (n + 1) = n + 1 :=
by
  -- Here we would prove that the assignment statement increases n by 1
  sorry

end assignment_increase_l695_695957


namespace women_left_room_is_3_l695_695401

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l695_695401


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l695_695266

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) 
  (h1 : ∃ (angles : Fin n → ℝ), 
    (∀ i : Fin n, 0 < angles i ∧ angles i < 180) ∧ 
    (∀ i : Fin 4, 90 < angles i) ∧ 
    (∑ i, angles i = 180 * (n - 2))) : 
  n ≤ 7 :=
by {
  sorry
}

end max_sides_of_convex_polygon_with_four_obtuse_angles_l695_695266


namespace inverse_difference_2007_1_l695_695359

-- Definitions/Conditions
variable {α β : Type*}  -- α is the type for domain and β for codomain
variable [AddGroup β]   -- ensuring operation + is available for β

-- Function f with inverse f_inv defined
variable (f : α → β)
variable (f_inv : β → α)

-- Condition 1: f has an inverse function f_inv
axiom f_has_inverse : ∀ (x : α), f_inv (f x) = x

-- Condition 2: f_inv(x-1) = f_inv(x) - 2 for all x in the domain
axiom f_inv_property : ∀ (x : β), f_inv (x - 1) = f_inv (x) - 2

-- Proposition
theorem inverse_difference_2007_1 : f_inv 2007 - f_inv 1 = 4012 :=
sorry

end inverse_difference_2007_1_l695_695359


namespace log_product_zero_l695_695233

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end log_product_zero_l695_695233


namespace outdoor_tables_count_l695_695996

theorem outdoor_tables_count (num_indoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) : ℕ :=
  let num_outdoor_tables := (total_chairs - (num_indoor_tables * chairs_per_indoor_table)) / chairs_per_outdoor_table
  num_outdoor_tables

example (h₁ : num_indoor_tables = 9)
        (h₂ : chairs_per_indoor_table = 10)
        (h₃ : chairs_per_outdoor_table = 3)
        (h₄ : total_chairs = 123) :
        outdoor_tables_count 9 10 3 123 = 11 :=
by
  -- Only the statement has to be provided; proof steps are not needed
  sorry

end outdoor_tables_count_l695_695996


namespace find_radius_of_sphere_l695_695075

noncomputable def radius_of_sphere : ℝ :=
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let h := λ (x: ℝ), x -- Equivalent heights of the cones
  let d12 := 3
  let d13 := 4
  let d23 := 5
  let R := 1
  R

theorem find_radius_of_sphere :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let h := λ (x: ℝ), x -- Equivalent heights of the cones
  let d12 := 3
  let d13 := 4
  let d23 := 5
  ∃ R: ℝ, R = 1 :=
by {
  -- skipping the proof
  sorry,
}

end find_radius_of_sphere_l695_695075


namespace cube_root_power_simplify_l695_695482

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695482


namespace find_a_decreasing_l695_695928

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem find_a_decreasing : 
  (∀ x : ℝ, x < 6 → f x a ≤ f (x - 1) a) → a ≥ 6 := 
sorry

end find_a_decreasing_l695_695928


namespace complement_union_example_l695_695171

open Set

theorem complement_union_example :
  ∀ (U A B : Set ℕ), 
  U = {1, 2, 3, 4, 5, 6, 7, 8} → 
  A = {1, 3, 5, 7} → 
  B = {2, 4, 5} → 
  (U \ (A ∪ B)) = {6, 8} := by 
  intros U A B hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_union_example_l695_695171


namespace find_a_l695_695696

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a : ∃ a : ℝ, g a = 0 ∧ a = 7 / 5 :=
by {
  existsi (7 / 5),
  split,
  { show g (7 / 5) = 0, sorry },
  { refl }
}

end find_a_l695_695696


namespace determine_even_condition_l695_695694

theorem determine_even_condition (x : ℤ) (m : ℤ) (h : m = x % 2) : m = 0 ↔ x % 2 = 0 :=
by sorry

end determine_even_condition_l695_695694


namespace count_divisible_subsets_l695_695058

open Finset BigOperators

theorem count_divisible_subsets (p : ℕ) (hp : Nat.Prime p) (hp_odd: p % 2 = 1) :
  let F := (Finset.range p).erase 0
  let s (M : Finset ℕ) := M.sum id
  let count_subsets := (F.powerset.filter (λ T, (¬ T = ∅ ∧ p ∣ s T))).card
  count_subsets = (2^(p-1) - 1) / p := by
  sorry

end count_divisible_subsets_l695_695058


namespace symmetry_and_sums_l695_695435

theorem symmetry_and_sums
  (f g : ℝ → ℝ)
  (f' g': ℝ → ℝ)
  (h1 : ∀ x, f'(x) = (deriv f) x)
  (h2 : ∀ x, g'(x) = (deriv g) x)
  (h3 : ∀ x, f(x + 2) - g(1 - x) = 2)
  (h4 : ∀ x, f'(x) = g'(x + 1))
  (h5 : ∀ x, g(x + 1) = -g(1 - x)) :
  (∀ x, g(4 - x) = g(x)) ∧
  (∀ x, g'(2 + x) = -g'(2 - x)) ∧
  (∑ k in finset.range 2022, g(k)) = 0 :=
sorry

end symmetry_and_sums_l695_695435


namespace probability_neither_square_nor_cube_l695_695557

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695557


namespace average_age_of_team_l695_695163

/--
The captain of a cricket team of 11 members is 26 years old and the wicket keeper is 
3 years older. If the ages of these two are excluded, the average age of the remaining 
players is one year less than the average age of the whole team. Prove that the average 
age of the whole team is 32 years.
-/
theorem average_age_of_team 
  (captain_age : Nat) (wicket_keeper_age : Nat) (remaining_9_average_age : Nat)
  (team_size : Nat) (total_team_age : Nat) (remaining_9_total_age : Nat)
  (A : Nat) :
  captain_age = 26 →
  wicket_keeper_age = captain_age + 3 →
  team_size = 11 →
  total_team_age = team_size * A →
  total_team_age = remaining_9_total_age + captain_age + wicket_keeper_age →
  remaining_9_total_age = 9 * (A - 1) →
  A = 32 :=
by
  sorry

end average_age_of_team_l695_695163


namespace prob1_prob2_l695_695348

noncomputable def a : ℝ × ℝ :=
  (real.sqrt 2 / 2, -real.sqrt 2 / 2)

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.sin x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a₁ := real.sqrt 2 / 2
  let a₂ := -real.sqrt 2 / 2
  a₁ * Real.sin x + a₂ * Real.cos x

theorem prob1 (x : ℝ) (h : 0 < x ∧ x < π / 2) (h_angle : Real.pi / 3 = Real.arccos (f x)) :
  x = 5 * π / 12 :=
sorry

theorem prob2 (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∀ m : ℝ, (f x ≤ m) ↔ m ∈ Set.Ici (-real.sqrt 2 / 2) :=
sorry

end prob1_prob2_l695_695348


namespace person_A_number_sums_l695_695986

theorem person_A_number_sums :
  let S := {n | ∃ k : ℕ, k ≤ 4 ∧ n = (5^k) * 1} -- defines the set of all numbers written by A.
  in ∀ x y ∈ S, x ≠ y → ∃ sums : Finset ℕ, sums.card = 206 ∧ (∀ s ∈ sums, ∃ a b ∈ S, a ≠ b ∧ s = a + b) :=
begin
  sorry
end

end person_A_number_sums_l695_695986


namespace brad_net_profit_l695_695228

theorem brad_net_profit 
  (gallons : ℕ) (glasses_per_gallon : ℕ) (cost_per_gallon : ℝ) (price_per_glass : ℝ) 
  (gallons_made : ℕ) (glasses_drank : ℕ) (glasses_unsold : ℕ) :
  gallons = 16 →
  cost_per_gallon = 3.50 →
  price_per_glass = 1.00 →
  gallons_made = 2 →
  glasses_drank = 5 →
  glasses_unsold = 6 →
  (price_per_glass * (gallons_made * glasses_per_gallon - glasses_drank - glasses_unsold) - 
   gallons_made * cost_per_gallon) = 14 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end brad_net_profit_l695_695228


namespace stella_dolls_count_l695_695920

variables (D : ℕ) (clocks glasses P_doll P_clock P_glass cost profit : ℕ)

theorem stella_dolls_count (h_clocks : clocks = 2)
                     (h_glasses : glasses = 5)
                     (h_P_doll : P_doll = 5)
                     (h_P_clock : P_clock = 15)
                     (h_P_glass : P_glass = 4)
                     (h_cost : cost = 40)
                     (h_profit : profit = 25) :
  D = 3 :=
by sorry

end stella_dolls_count_l695_695920


namespace find_initial_divisor_l695_695451

theorem find_initial_divisor (N D : ℤ) (h1 : N = 2 * D) (h2 : N % 4 = 2) : D = 3 :=
by
  sorry

end find_initial_divisor_l695_695451


namespace distance_climbing_down_l695_695821

-- Definitions based on the conditions given in the problem.
variables (x : ℝ)

-- The conditions of the problem
def time_up (x : ℝ) : ℝ := x / 2
def time_down (x : ℝ) : ℝ := (x + 2) / 3

-- The total time condition
def total_time_condition (x : ℝ) : Prop := time_up x + time_down x = 4

-- Proving the main statement
theorem distance_climbing_down : total_time_condition x → x + 2 = 6 :=
by
  intro h
  sorry

end distance_climbing_down_l695_695821


namespace range_of_a_l695_695345

variable (P M : Set ℝ)
variable (a : ℝ)

def P := {x | -1 ≤ x ∧ x ≤ 1}
def M := {a}

theorem range_of_a (h : P ∩ M = ∅) : a ∈ Set.Iio (-1) ∪ Set.Ioi 1 :=
by
  sorry

end range_of_a_l695_695345


namespace relationship_among_a_b_c_l695_695285

noncomputable def a : ℝ := Real.log 6 / Real.log 3  -- log base 3 of 6
noncomputable def b : ℝ := 1 + 3^(-Real.log (Real.exp 1) / Real.log 3) -- 1 + 3^(-log base 3 of e)
noncomputable def c : ℝ := 1 / (2 / 3)  -- (2/3)^(-1)

theorem relationship_among_a_b_c : a > c ∧ c > b := by
  rw [← Real.log_div_log 6 3, ← Real.log_div_log (Real.exp 1) 3]
  sorry

end relationship_among_a_b_c_l695_695285


namespace Asaf_age_is_90_l695_695497

variable (A B P : ℕ)

theorem Asaf_age_is_90 
  (h1 : A + B = 140)
  (h2 : 2 * P + 60 = 220)
  (h3 : |A - B| = P / 2) :
  A = 90 := 
sorry

end Asaf_age_is_90_l695_695497


namespace line_equation_and_distance_l695_695319

noncomputable def cos_alpha := -1/2
noncomputable def point_A : ℝ × ℝ := (sqrt(3) / 3, 2)
def line_passing_through_fixed_point (k : ℝ) := k * (sqrt(3)) - 1

theorem line_equation_and_distance :
  ∀ (B : ℝ × ℝ),
    B = (sqrt(3), 1) →
    (∀ (k : ℝ), k * (B.1 - sqrt(3)) - B.2 + 1 = 0) →
    (∃ (l : ℝ × ℝ → Prop),
      l = λ P, sqrt(3) * P.1 + P.2 - 3 = 0 ∧
      (dist (B : ℝ × ℝ) l = 1/2)) :=
begin
  intro B,
  intro B_eq,
  intro line_cond,
  use (λ P, sqrt(3) * P.1 + P.2 - 3 = 0),
  split,
  { refl },
  { sorry }
end

#check line_equation_and_distance

end line_equation_and_distance_l695_695319


namespace related_possibility_greater_l695_695278

theorem related_possibility_greater {a b c d : ℝ} (H : ¬(a = 0 ∧ b = 0) ∧ ¬(c = 0 ∧ d = 0)) 
    (Δ : (a / (a + b)) - (c / (c + d)) > 0) :
    ∃ χ² : ℝ, χ² > 0 ∧ (χ² > some_threshold → (possibility_related x Y > 0)) := 
sorry

end related_possibility_greater_l695_695278


namespace simplify_root_power_l695_695478

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695478


namespace portion_of_platinum_limit_unspent_l695_695462

variables (G : ℝ)

def gold_card_limit := G
def platinum_card_limit := 2 * G
def diamond_card_limit := 3 * G

def gold_card_initial_balance := G / 3
def platinum_card_initial_balance := G / 3  -- since 2G / 6 = G / 3
def diamond_card_initial_balance := G / 3  -- since 3G / 9 = G / 3

def platinum_card_balance_after_transfer :=
  platinum_card_initial_balance + gold_card_initial_balance -- which is 2G / 3

def platinum_card_balance_after_half_transfer :=
  platinum_card_balance_after_transfer / 2 -- which is G / 3

def remaining_platinum_card_balance :=
  platinum_card_balance_after_transfer - platinum_card_balance_after_half_transfer -- which is G / 3

def unspent_platinum_limit :=
  platinum_card_limit - remaining_platinum_card_balance -- which is (2G - G / 3)

def portion_of_unspent_limit :=
  (unspent_platinum_limit / platinum_card_limit) == (5 / 6)

theorem portion_of_platinum_limit_unspent :
  portion_of_unspent_limit G :=
by
  sorry

end portion_of_platinum_limit_unspent_l695_695462


namespace find_s2_l695_695884

def t(x : ℝ) := 2 * x - 6
def s(y : ℝ) := 2 * y * y + 2 * y - 4

theorem find_s2 : s(2) = 36 :=
by
  have h: t(4) = 2, from calc
    t 4 = 2 * 4 - 6 : by rw t
         ... = 8 - 6 : by norm_num
         ... = 2 : by norm_num
  rw ←h
  have h2: s(t(4)) = s(2), from calc
    s (t 4) = s 2 : by rw h
  rw h2
  dsimp [s, t]
  norm_num

end find_s2_l695_695884


namespace cube_problem_proof_l695_695663

-- Definitions of the cube and its properties
structure Cube (n : ℕ) :=
  (unit_cubes : ℕ)
  (rods : set (ℕ × ℕ))
  (rod_pierce : ∀ r ∈ rods, ∃ i j, r = (i, j))
  (each_unit_cube_pierced : ∀ cube_id, ∃ r ∈ rods, cube_id ∈ {p.1 | p ∈ r} ∪ {p.2 | p ∈ r})

-- Define the propositions for proof
def proposition_a (n : ℕ) (C : Cube n) : Prop :=
  ∃ rods_selected : set (ℕ × ℕ),
  (∀ r1 r2 ∈ rods_selected, r1 ≠ r2 → disjoint {u | ∃ i j, u = (i, j ∈ r1)} {u | ∃ i j, u = (i, j ∈ r2)}) ∧
  (∃ d1 d2, ∀ r ∈ rods_selected, rod_direction r = d1 ∨ rod_direction r = d2)

def proposition_b (n : ℕ) (C : Cube n) : Prop :=
  ∃ rods_selected : set (ℕ × ℕ),
  (∀ r1 r2 ∈ rods_selected, r1 ≠ r2 → disjoint {u | ∃ i j, u = (i, j ∈ r1)} {u | ∃ i j, u = (i, j ∈ r2)}) ∧
  rods_selected.card = 2 * n^2

-- The final theorem to state our problem similarly to given problem and solutions
theorem cube_problem_proof (n : ℕ) (C : Cube n) : 
  proposition_a n C ∧ proposition_b n C :=
begin
  sorry, -- Proof is not required
end

end cube_problem_proof_l695_695663


namespace digit_sum_after_2015_operations_l695_695211

def initial_number : ℕ := 2015

def next_digit_sum (n : ℕ) : ℕ :=
  let tens := (n / 10) % 10
  let hundreds := (n / 100) % 10
  let thousands := (n / 1000) % 10
  tens + hundreds + thousands

def append_digit_sum (n : ℕ) : ℕ :=
  10 * n + next_digit_sum n

def repeated_process (n : ℕ) (times : ℕ) : ℕ :=
  (List.repeat append_digit_sum times).foldl id n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem digit_sum_after_2015_operations :
  sum_of_digits (repeated_process initial_number 2015) = 8065 :=
by
  sorry

end digit_sum_after_2015_operations_l695_695211


namespace sin_sum_angle_correct_l695_695850

noncomputable def sin_sum_angle (α β : ℝ) : ℝ :=
  if h : (cos α, sin α) = (-2/3 : ℝ, sqrt 5 / 3 : ℝ) ∧ (cos β, sin β) = (sqrt 5 / 3 : ℝ, -2/3 : ℝ) then
    sin (α + β)
  else 
    0

theorem sin_sum_angle_correct (α β : ℝ)
    (hα : cos α = -2 / 3 ∧ sin α = sqrt 5 / 3)
    (hβ : cos β = sqrt 5 / 3 ∧ sin β = -2 / 3) :
  sin (α + β) = 1 :=
sorry

end sin_sum_angle_correct_l695_695850


namespace x_seq_converges_to_zero_y_seq_converges_l695_695036

def x_seq : ℕ → ℝ
| 1       := x1
| (n + 1) := 3 * (x_seq n)^2 - 2 * n * (x_seq n)^3

def y_seq (x_seq : ℕ → ℝ) : ℕ → ℝ
| 1       := x_seq 1
| (n + 1) := y_seq x_seq n + (n + 1) * (x_seq (n + 1))

theorem x_seq_converges_to_zero (x1 : ℝ) (h : 0 < x1 ∧ x1 < 1/2) :
  ∃ l, filter.tendsto x_seq filter.at_top (𝓝 l) ∧ l = 0 :=
sorry

theorem y_seq_converges (x1 : ℝ) (h : 0 < x1 ∧ x1 < 1/2) :
  ∃ l, filter.tendsto (y_seq x_seq) filter.at_top (𝓝 l) :=
sorry

end x_seq_converges_to_zero_y_seq_converges_l695_695036


namespace log_eq_implies_m_l695_695823

noncomputable def m_value (c n : ℝ) : ℝ := sqrt (exp c / n)

theorem log_eq_implies_m (c n m : ℝ)
  (h : log (m^2) = c - 2 * log n) : m = sqrt (exp c / n) :=
by
  sorry

end log_eq_implies_m_l695_695823


namespace sum_S10_equals_10_div_21_l695_695796

def a (n : ℕ) : ℚ := 1 / (4 * n^2 - 1)
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_S10_equals_10_div_21 : S 10 = 10 / 21 :=
by
  sorry

end sum_S10_equals_10_div_21_l695_695796


namespace total_cost_is_correct_l695_695417

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l695_695417


namespace problem_statement_l695_695633

theorem problem_statement :
  (¬ (∃ x₀ : ℝ, real.exp x₀ ≥ 1) → ∀ x : ℝ, real.exp x < 1) ∧
  (∀ (a b m : ℝ), a < b → a * m^2 < b * m^2) ∧
  (∀ (x : ℝ), sin x = 1 / 2 → x = real.pi / 6) ∧
  (¬ (∀ p q : Prop, ¬ (p ∨ q) → ¬ p ∧ ¬ q)) →
  false := 
sorry

end problem_statement_l695_695633


namespace range_BA_dot_BP_l695_695392

noncomputable def P (α : ℝ) := (Real.cos α, Real.sin α)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)
def BA : ℝ × ℝ := (1, 1)
def BP (α : ℝ) := (Real.cos α, Real.sin α + 1)

def BA_dot_BP (α : ℝ) : ℝ := (Real.cos α) + (Real.sin α) + 1

theorem range_BA_dot_BP : ∀ α ∈ Set.Icc 0 Real.pi, 0 ≤ BA_dot_BP α ∧ BA_dot_BP α ≤ 1 + Real.sqrt 2 :=
sorry

end range_BA_dot_BP_l695_695392


namespace subtracted_value_l695_695104

theorem subtracted_value (s : ℕ) (h : s = 4) (x : ℕ) (h2 : (s + s^2 - x = 4)) : x = 16 :=
by
  sorry

end subtracted_value_l695_695104


namespace distance_pskov_volkhov_l695_695113

theorem distance_pskov_volkhov 
  (d_LV : ℕ) (d_VL : ℕ) (d_LP : ℕ) (d_PL : ℕ)
  (h1 : d_LV = 194)
  (h2 : d_VL = 116)
  (h3 : d_LP = 451)
  (h4 : d_PL = 141) 
  : (∃ d_PV, d_PV = 335) :=
by
  use 335
  sorry

end distance_pskov_volkhov_l695_695113


namespace sum_first_10_common_elements_l695_695747

-- Define the arithmetic progression (AP)
def arithmetic_prog (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression (GP)
def geometric_prog (k : ℕ) : ℕ := 10 * 2 ^ k

-- Find the sum of the first 10 elements present in both sequences
theorem sum_first_10_common_elements : 
  (Σ x in {x | ∃ n k, arithmetic_prog n = x ∧ geometric_prog k = x}, x).take 10 = 6990500 :=
sorry

end sum_first_10_common_elements_l695_695747


namespace proof_problem_exists_R1_R2_l695_695015

def problem (R1 R2 : ℕ) : Prop :=
  let F1_R1 := (4 * R1 + 5) / (R1^2 - 1)
  let F2_R1 := (5 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (3 * R2 + 2) / (R2^2 - 1)
  let F2_R2 := (2 * R2 + 3) / (R2^2 - 1)
  F1_R1 = F1_R2 ∧ F2_R1 = F2_R2 ∧ R1 + R2 = 14

theorem proof_problem_exists_R1_R2 : ∃ (R1 R2 : ℕ), problem R1 R2 :=
sorry

end proof_problem_exists_R1_R2_l695_695015


namespace jacob_age_proof_l695_695709

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l695_695709


namespace num_palindromes_between_1000_3000_l695_695351

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def valid_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 2999 ∧ is_palindrome n

theorem num_palindromes_between_1000_3000 : ∃! (n : ℕ), valid_palindrome n → n = 30 := 
sorry

end num_palindromes_between_1000_3000_l695_695351


namespace parabola_line_intersection_l695_695806

/-- 
Given the parabola y^2 = -x and the line l: y = k(x + 1) intersect at points A and B,
(Ⅰ) Find the range of values for k;
(Ⅱ) Let O be the vertex of the parabola, prove that OA ⟂ OB.
-/
theorem parabola_line_intersection (k : ℝ) (A B : ℝ × ℝ)
  (hA : A.2 ^ 2 = -A.1) (hB : B.2 ^ 2 = -B.1)
  (hlineA : A.2 = k * (A.1 + 1)) (hlineB : B.2 = k * (B.1 + 1)) :
  (k ≠ 0) ∧ ((A.2 * B.2 = -1) → A.1 * B.1 * (A.2 * B.2) = -1) :=
by
  sorry

end parabola_line_intersection_l695_695806


namespace probability_neither_perfect_square_nor_cube_l695_695578

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695578


namespace quadratic_real_roots_m_range_l695_695835

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l695_695835


namespace probability_neither_square_nor_cube_l695_695565

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695565


namespace palindrome_count_l695_695350

-- Definitions for the problem
def digits : List ℕ := [3, 3, 4, 4, 6, 6, 6]

-- A 7-digit palindrome has the form [a, b, c, d, c, b, a] where d is the middle digit.
def is_palindrome (n : List ℕ) : Prop :=
  n.length = 7 ∧ (∀ i < 3, n[i] = n[6 - i]) ∧ n[3] = 6

-- Calculate the number of valid palindromes for the given digits
def count_palindromes (digits : List ℕ) : ℕ :=
  -- body of the proof is skipped
  sorry

-- Statement of the theorem
theorem palindrome_count : count_palindromes digits = 6 :=
by
  -- Proof is omitted
  sorry

end palindrome_count_l695_695350


namespace Generalized_Helly_up_to_3D_l695_695761

noncomputable def n_dimensional_helly (n : ℕ) : Prop :=
  ∀ (C : fin (n + 2) → set ℝ^n),
    (∀ (s : finset (fin (n + 2))), s.card = n + 1 → (⋂ i ∈ s, C i).nonempty) →
    (⋂ i, C i).nonempty

axiom helly_1d : n_dimensional_helly 1

axiom helly_2d : n_dimensional_helly 2

axiom helly_3d : n_dimensional_helly 3

-- The generalized theorem up to 3-dimensional space
theorem Generalized_Helly_up_to_3D :
  ∀ n, n <= 3 → n_dimensional_helly n :=
by
  intros n hn
  rcases hn with ⟨ ⟨ ⟩  | mk lb1 (lt_nat.lb2_lb_of_nat z) ⟩;
  [exact helly_1d, exact helly_2d, exact helly_3d]

end Generalized_Helly_up_to_3D_l695_695761


namespace f_transformation_l695_695365

def transformation_condition_1 (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f y = f (x * 2)

def transformation_condition_2 (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f y = f (y - (x + π / 3))

def transformation_condition_3 (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x, f x = f (x - y + 1)

theorem f_transformation :
  (∀ x, transformation_condition_1 f x ∧ transformation_condition_2 f x ∧ transformation_condition_3 f x) →
  (λ x, f (x + π / 3) - 1 = (1 / 2) * sin x) →
  f = (λ x, (1 / 2) * sin (2 * x - π / 3) + 1) :=
by
  intros h_trans h_eq
  sorry

end f_transformation_l695_695365


namespace count_x0_such_that_x0_equals_x6_l695_695423

def sequence_condition (x_n: ℕ → ℝ) (n: ℕ) : Prop :=
  (x_n n = if 2 * x_n (n-1) < 1 then 2 * x_n (n-1) else 2 * x_n (n-1) - 1)

theorem count_x0_such_that_x0_equals_x6 : 
  (∃ x0 ∈ Ico 0 1, ∃ x_n : ℕ → ℝ, (∀ n > 0, sequence_condition x_n n) ∧ x_n 0 = x0 ∧ x_n 6 = x0) → set.finite { x0 ∈ Ico 0 1 | ∃ x_n : ℕ → ℝ, (∀ n > 0, sequence_condition x_n n) ∧ x_n 0 = x0 ∧ x_n 6 = x0 } ∧ set.card { x0 ∈ Ico 0 1 | ∃ x_n : ℕ → ℝ, (∀ n > 0, sequence_condition x_n n) ∧ x_n 0 = x0 ∧ x_n 6 = x0 } = 64 :=
sorry

end count_x0_such_that_x0_equals_x6_l695_695423


namespace property_tax_increase_is_800_l695_695980

-- Define conditions as constants
def tax_rate : ℝ := 0.10
def initial_value : ℝ := 20000
def new_value : ℝ := 28000

-- Define the increase in property tax
def tax_increase : ℝ := (new_value * tax_rate) - (initial_value * tax_rate)

-- Statement to be proved
theorem property_tax_increase_is_800 : tax_increase = 800 :=
by
  sorry

end property_tax_increase_is_800_l695_695980


namespace probability_neither_square_nor_cube_l695_695569

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695569


namespace simplify_root_power_l695_695475

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695475


namespace flies_before_9_minute_rest_total_rest_before_98_flies_flies_after_1999_minutes_rest_l695_695181

def rest_period (n : ℕ) : ℕ :=
  Nat.bitCount n

def total_rest (n : ℕ) : ℕ :=
  (Nat.range n).sum (λ i, rest_period (i + 1))

theorem flies_before_9_minute_rest :
  (∀ n, rest_period n = 9 → n - 1 = 510) :=
by
  sorry

theorem total_rest_before_98_flies :
  total_rest 98 = 312 :=
by
  sorry

theorem flies_after_1999_minutes_rest :
  (n : ℕ) → total_rest n = 1999 → n - 1 = 462 :=
by
  sorry

end flies_before_9_minute_rest_total_rest_before_98_flies_flies_after_1999_minutes_rest_l695_695181


namespace geo_seq_arith_seq_l695_695929

theorem geo_seq_arith_seq (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_gp : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, a_n n > 0) (h_arith : a_n 4 - a_n 3 = a_n 5 - a_n 4) 
  (hq_pos : q > 0) (hq_neq1 : q ≠ 1) :
  S 6 / S 3 = 2 := by
  sorry

end geo_seq_arith_seq_l695_695929


namespace value_of_a_l695_695601

theorem value_of_a (a : ℝ) (h : 1 ∈ ({a, a ^ 2} : Set ℝ)) : a = -1 :=
sorry

end value_of_a_l695_695601


namespace rectangle_ratio_l695_695084

theorem rectangle_ratio (a b c d : ℝ)
  (h1 : (a * b) / (c * d) = 0.16)
  (h2 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 :=
by 
  sorry

end rectangle_ratio_l695_695084


namespace projection_of_a_on_b_l695_695315

theorem projection_of_a_on_b (a b : ℝ) (θ : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 1)
  (hθ : θ = 60) : 
  (|a| * Real.cos (θ * Real.pi / 180)) = 1 := 
sorry

end projection_of_a_on_b_l695_695315


namespace f_not_monotonic_l695_695768

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x:ℝ)) = -f x

def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ¬ ( (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x) )

variable (f : ℝ → ℝ)

axiom periodicity : ∀ x, f (x + 3/2) = -f x 
axiom odd_shifted : is_odd_function (λ x => f (x - 3/4))

theorem f_not_monotonic : is_not_monotonic f := by
  sorry

end f_not_monotonic_l695_695768


namespace crayons_selection_l695_695605

theorem crayons_selection : 
  ∃ (n : ℕ), n = Nat.choose 14 4 ∧ n = 1001 := by
  sorry

end crayons_selection_l695_695605


namespace polar_to_rectangular_l695_695248

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 3) :
    (r * Real.cos θ, r * Real.sin θ) = (2, 2 * Real.sqrt 3) :=
by
  rw [h_r, h_θ]
  norm_num
  rw [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  exact ⟨rfl, rfl⟩

end polar_to_rectangular_l695_695248


namespace probability_neither_square_nor_cube_l695_695571

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695571


namespace find_original_number_l695_695668

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_original_number_l695_695668


namespace function_properties_l695_695832

-- Definitions for the conditions
noncomputable def f (x : ℝ) : ℝ := Real.logBase 2 (Real.sqrt (x^2 + 1) - x)

-- The proof statement
theorem function_properties : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) :=
by 
  sorry

end function_properties_l695_695832


namespace proof_problem_l695_695877

noncomputable def alpha : ℕ := (2 ^ 1000) % 13
noncomputable def beta (alpha : ℕ) : ℝ := 
  ( (7 + 4 * Real.sqrt alpha) ^ (1 / 2) - (7 - 4 * Real.sqrt alpha) ^ (1 / 2) ) / Real.sqrt alpha
def f (a : ℝ) (beta : ℝ) : ℝ := a - beta
def F (a : ℝ) (b : ℝ) : ℝ := b ^ 2 + a
noncomputable def delta : ℝ := 
  let (x1 := 10 ^ Real.sqrt (Real.log 7))
  let (x2 := 10 ^ (-Real.sqrt (Real.log 7)))
  x1 * x2

theorem proof_problem : 
  alpha = 3 ∧ 
  beta 3 = 2 ∧ 
  F 3 (f 4 2) = 7 ∧ 
  delta = 1 :=
by
  sorry

end proof_problem_l695_695877


namespace range_of_f_minus_2_l695_695044

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_minus_2 (a b : ℝ) (h1 : 1 ≤ f (-1) a b) (h2 : f (-1) a b ≤ 2) (h3 : 2 ≤ f 1 a b) (h4 : f 1 a b ≤ 4) :
  6 ≤ f (-2) a b ∧ f (-2) a b ≤ 10 :=
sorry

end range_of_f_minus_2_l695_695044


namespace tangent_parallel_line_l695_695371

open Function

def f (x : ℝ) : ℝ := x^4 - x

def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel_line {P : ℝ × ℝ} (hP : ∃ x y, P = (x, y) ∧ f' x = 3) :
  P = (1, 0) := by
  sorry

end tangent_parallel_line_l695_695371


namespace fish_caught_in_second_catch_l695_695381

theorem fish_caught_in_second_catch
  (tagged_fish_released : Int)
  (tagged_fish_in_second_catch : Int)
  (total_fish_in_pond : Int)
  (C : Int)
  (h_tagged_fish_count : tagged_fish_released = 60)
  (h_tagged_in_second_catch : tagged_fish_in_second_catch = 2)
  (h_total_fish : total_fish_in_pond = 1800) :
  C = 60 :=
by
  sorry

end fish_caught_in_second_catch_l695_695381


namespace find_smallest_angle_l695_695733

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l695_695733


namespace women_left_room_is_3_l695_695400

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l695_695400


namespace triangle_base_length_l695_695498

/-
Theorem: Given a triangle with height 5.8 meters and area 24.36 square meters,
the length of the base is 8.4 meters.
-/

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) :
  h = 5.8 ∧ A = 24.36 ∧ A = (b * h) / 2 → b = 8.4 :=
by
  sorry

end triangle_base_length_l695_695498


namespace inclination_angle_of_line_l695_695932

noncomputable def slope_of_line (a b : ℝ) : ℝ := -a / b

theorem inclination_angle_of_line {c : ℝ} :
  ∃ θ : ℝ, θ ∈ set.Ico 0 real.pi ∧ tan θ = - (1 / sqrt 3) ∧ θ = (5 * real.pi) / 6 :=
by sorry

end inclination_angle_of_line_l695_695932


namespace two_trains_meet_distance_l695_695619

noncomputable def distance_between_stations : ℕ := 65

theorem two_trains_meet_distance :
  let D := distance_between_stations in
  let distance_covered_by_first_train := 20 * 2 in
  let distance_covered_by_second_train := 25 * 1 in
  D = distance_covered_by_first_train + distance_covered_by_second_train :=
by
  sorry

end two_trains_meet_distance_l695_695619


namespace SmallestPositiveAngle_l695_695740

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l695_695740


namespace integer_coefficient_equation_calculate_expression_l695_695231

noncomputable def a : ℝ := (Real.sqrt 5 - 1) / 2

theorem integer_coefficient_equation :
  a ^ 2 + a - 1 = 0 :=
sorry

theorem calculate_expression :
  a ^ 3 - 2 * a + 2015 = 2014 :=
sorry

end integer_coefficient_equation_calculate_expression_l695_695231


namespace vertex_angle_third_cone_l695_695948

theorem vertex_angle_third_cone (A : Point) (plane : Plane) 
  (cone1 cone2 cone3 : Cone) 
  (common_vertex : cone1.vertex = A ∧ cone2.vertex = A ∧ cone3.vertex = A)
  (touch_externally : (Cone.tangent_externally cone1 cone2) ∧ (Cone.tangent_externally cone2 cone3) ∧ (Cone.tangent_externally cone3 cone1))
  (vertex_angle_cone1 : cone1.vertex_angle = π / 3)
  (vertex_angle_cone2 : cone2.vertex_angle = π / 3)
  (cones_touch_plane : (Cone.tangent_plane cone1 plane) ∧ (Cone.tangent_plane cone2 plane) ∧ (Cone.tangent_plane cone3 plane))
  (cones_same_side : (Cone.same_side plane cone1 cone2 cone3)) :
  cone3.vertex_angle = 2 * real.arctan(2 * (sqrt 3 - sqrt 2)) :=
sorry

end vertex_angle_third_cone_l695_695948


namespace triangle_area_correct_l695_695150

-- Define base and height
def base : ℝ := 18
def height : ℝ := 6

-- Define the area function
def triangle_area (b h : ℝ) : ℝ := (b * h) / 2

-- Proof statement
theorem triangle_area_correct : triangle_area base height = 54 :=
by 
  sorry

end triangle_area_correct_l695_695150


namespace af_squared_l695_695862

theorem af_squared (A B C D E F : Type) (ω : A -> B -> C -> Type) (γ : D -> E -> Type)
  (AB BC AC : ℕ) (angle : A -> C -> B)
  (h1 : AB = 6) (h2 : BC = 8) (h3 : AC = 4)
  (h4 : inscribed_triangle ω A B C)
  (h5 : angle_bisector_intersects BC angle D)
  (h6 : intersects_angle_bisector_again ω angle E)
  (h7 : circle_with_diameter γ D E)
  (h8 : intersects_second_time γ ω F) :
  AF^2 = 1024 / 25 := sorry

end af_squared_l695_695862


namespace remainder_is_zero_l695_695434

theorem remainder_is_zero :
  let p₁ (x : ℝ) := (x^6 - (1/2)^6) / (x - 1/2)
  let s₁ := (1/2)^6
  let p₂ (x : ℝ) := (p₁ (x) - (1/2)^5) / (x - 1/2)
  let s₂ := p₁ (1/2)
  in s₂ = 0 := by
sorry

end remainder_is_zero_l695_695434


namespace range_k_plus_b_l695_695328

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def k (x₀ : ℝ) : ℝ := 1 / x₀

noncomputable def b (x₀ : ℝ) : ℝ := Real.log x₀ - 1

def g (x : ℝ) : ℝ := (Real.log x - 1) + 1 / x

theorem range_k_plus_b : ∀ x₀ > 0, k x₀ + b x₀ ∈ Set.Ici 0 :=
by
  intros
  have h1 : k x₀ = 1 / x₀ := by sorry
  have h2 : b x₀ = Real.log x₀ - 1 := by sorry
  have h_g : g x₀ = Real.log x₀ - 1 + 1 / x₀ := by sorry
  sorry

end range_k_plus_b_l695_695328


namespace strawberries_to_grapes_ratio_l695_695419

-- Define initial conditions
def initial_grapes : ℕ := 100
def fruits_left : ℕ := 96

-- Define the number of strawberries initially
def strawberries_init (S : ℕ) : Prop :=
  (S - (2 * (1/5) * S) = fruits_left - initial_grapes + ((2 * (1/5)) * initial_grapes))

-- Define the ratio problem in Lean
theorem strawberries_to_grapes_ratio (S : ℕ) (h : strawberries_init S) : (S / initial_grapes = 3 / 5) :=
sorry

end strawberries_to_grapes_ratio_l695_695419


namespace trajectory_of_M_l695_695305

variables {x y : ℝ}

theorem trajectory_of_M (h : y / (x + 2) + y / (x - 2) = 2) (hx : x ≠ 2) (hx' : x ≠ -2) :
  x * y - x^2 + 4 = 0 :=
by sorry

end trajectory_of_M_l695_695305


namespace seashells_calculation_l695_695899

theorem seashells_calculation :
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  leigh_seashells = 16 :=
by
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  show leigh_seashells = 16
  sorry

end seashells_calculation_l695_695899


namespace percent_of_marigolds_l695_695994

-- Definition of the conditions
def half_of (x : ℚ) : ℚ := x / 2
def two_thirds_of (x : ℚ) : ℚ := (2 / 3) * x

-- Assume fractions representing yellow and white flowers
def fraction_yellow := 4/7
def fraction_white := 3/7

-- Conditions
def half_yellow_are_daisies (y : ℚ) : Prop := half_of(y) = y / 2
def two_thirds_white_are_marigolds (w : ℚ) : Prop := two_thirds_of(w) = (2 / 3) * w

-- Proof statement for the total percentage of marigolds
theorem percent_of_marigolds :
  half_yellow_are_daisies fraction_yellow →
  two_thirds_white_are_marigolds fraction_white →
  (fraction_yellow * half_of(1) + fraction_white * two_thirds_of(1)) * 100 = 57 := 
by
  intro h1 h2
  -- here the proof would be constructed
  sorry

end percent_of_marigolds_l695_695994


namespace women_left_room_is_3_l695_695399

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l695_695399


namespace cube_root_power_simplify_l695_695486

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695486


namespace payment_correct_l695_695068

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l695_695068


namespace concyclic_iff_power_of_point_l695_695431

structure Point (α : Type) := 
  (x : α) (y : α)

def concyclic {α : Type} [Field α] (A B C D : Point α) : Prop := sorry

def pow_of_point {α : Type} [Field α] (A B C D M : Point α) := 
  let MA := (A.x - M.x) * (A.y - M.y)
  let MB := (B.x - M.x) * (B.y - M.y)
  let MC := (C.x - M.x) * (C.y - M.y)
  let MD := (D.x - M.x) * (D.y - M.y)
  MA * MB = MC * MD

theorem concyclic_iff_power_of_point {α : Type} [Field α] (A B C D M : Point α) (h_noncollinear : ¬Concollinear (Set.toFinset {A, B, C, D})) (h_intersect : Intersect (Line A B) (Line C D) = M) :
  concyclic A B C D ↔ pow_of_point A B C D M := 
by {
  sorry
}

end concyclic_iff_power_of_point_l695_695431


namespace largest_four_digit_odd_digits_sum_19_l695_695961

theorem largest_four_digit_odd_digits_sum_19 : 
  ∃ n : ℕ, n = 9711 ∧ (1000 ≤ n ∧ n < 10000) ∧ 
            (∑ d in [ (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10], True) = 19 ∧
            ∀ k, k ∈ [ (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10] → k % 2 = 1 :=
begin
  sorry
end

end largest_four_digit_odd_digits_sum_19_l695_695961


namespace valentines_count_l695_695450

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 52) : x * y = 108 :=
by sorry

end valentines_count_l695_695450


namespace quadratic_has_real_roots_iff_l695_695834

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l695_695834


namespace probability_sum_is_perfect_square_l695_695954

-- Define the conditions: Two 6-sided dice
def num_faces := 6
def total_outcomes := num_faces * num_faces

-- Define the problem: The probability that the sum is a perfect square
theorem probability_sum_is_perfect_square : ∃ p : ℚ, 
  (p = 7 / 36) ∧ 
  (let perfect_squares := {4, 9},
       valid_outcomes := 
         { (d1, d2) | (d1 + d2 ∈ perfect_squares) } in
    set.size valid_outcomes / total_outcomes = p) :=
begin
  sorry
end

end probability_sum_is_perfect_square_l695_695954


namespace percentage_exceed_l695_695640

variable {x y : ℝ}

-- The condition x = 0.60 * y
def condition : Prop := x = 0.60 * y

-- The proof problem statement
theorem percentage_exceed (h : condition) : ((y - x) / x) * 100 = 66.67 :=
by
  sorry

end percentage_exceed_l695_695640


namespace girls_in_class_l695_695386

theorem girls_in_class (g b : ℕ) (h1 : g + b = 28) (h2 : g * 4 = b * 3) : g = 12 := by
  sorry

end girls_in_class_l695_695386


namespace complex_product_in_polar_form_l695_695688

theorem complex_product_in_polar_form :
  let z1 := Complex.ofPolar 4 (Real.pi / 6)
  let z2 := Complex.ofPolar (-3) (Real.pi / 4)
  let z3 := Complex.ofPolar 5 (Real.pi / 3)
  let z_product := z1 * z2 * z3
  let r := Complex.abs z_product
  let theta := Complex.arg z_product
  (r, Real.toDegrees theta) = (60, 315) :=
by
  sorry

end complex_product_in_polar_form_l695_695688


namespace triangle_bc_length_l695_695390

theorem triangle_bc_length {A B C G O : Point} 
    (h_triangle_acute : acute_triangle A B C)
    (h_angle_acb : ∠ACB = 45)
    (h_centroid : centroid A B C = G)
    (h_circumcenter : circumcenter A B C = O)
    (h_OG_length : distance O G = 1)
    (h_OG_parallel_BC : parallel (line_through O G) (line_through B C)) :
    distance B C = 2 := 
sorry

end triangle_bc_length_l695_695390


namespace line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l695_695114

-- Define the point (M) and the properties of the line
def point_M : ℝ × ℝ := (-1, 3)

def parallel_to_y_axis (line : ℝ × ℝ → Prop) : Prop :=
  ∃ b : ℝ, ∀ y : ℝ, line (b, y)

-- Statement we need to prove
theorem line_through_point_parallel_to_y_axis_eq_x_eq_neg1 :
  (∃ line : ℝ × ℝ → Prop, line point_M ∧ parallel_to_y_axis line) → ∀ p : ℝ × ℝ, (p.1 = -1 ↔ (∃ line : ℝ × ℝ → Prop, line p ∧ line point_M ∧ parallel_to_y_axis line)) :=
by
  sorry

end line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l695_695114


namespace max_elements_of_T_l695_695198

noncomputable def arithmetic_mean_is_integer (T : Set ℕ) (y : ℕ) : Prop :=
  ∃ (m M : ℕ), M = ∑ x in T, x ∧ m + 1 = T.card ∧ ∀ y ∈ T, (M - y) % m = 0

theorem max_elements_of_T :
  ∃ (T : Set ℕ), 
    2 ∈ T ∧ 
    3003 ∈ T ∧ 
    (∀ y ∈ T, arithmetic_mean_is_integer T y) ∧ 
    T.card = 30 := sorry

end max_elements_of_T_l695_695198


namespace median_inequality_l695_695910

open Real

theorem median_inequality
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b > c)
  :  let CM3 := sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4) in
     CM3 < (a + b) / 2 :=
by
  -- Definitions from conditions
  let CM3 := sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  have semiperimeter : (a + b) / 2 > 0, from sorry,
  sorry

end median_inequality_l695_695910


namespace eagles_points_l695_695380

theorem eagles_points (x y : ℕ) (h₁ : x + y = 82) (h₂ : x - y = 18) : y = 32 :=
sorry

end eagles_points_l695_695380


namespace probability_neither_square_nor_cube_l695_695546

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695546


namespace lasagna_cheese_quantity_l695_695613

theorem lasagna_cheese_quantity : 
  ∃ x : ℝ, (6 * x) + 4 = 13 ∧ x = 1.5 :=
by
  use 1.5
  split
  { linarith }
  { refl }

end lasagna_cheese_quantity_l695_695613


namespace solve_system_eq_l695_695918

theorem solve_system_eq (x y z : ℝ) 
  (h1 : x * y = 6 * (x + y))
  (h2 : x * z = 4 * (x + z))
  (h3 : y * z = 2 * (y + z)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = -24 ∧ y = 24 / 5 ∧ z = 24 / 7) :=
  sorry

end solve_system_eq_l695_695918


namespace change_in_expression_l695_695374

theorem change_in_expression {x b : ℝ} (hx : 0 ≤ b) : 
  let initial_expr := 2*x^2 + 5
  let increased_expr := 2*(x + b)^2 + 5
  let decreased_expr := 2*(x - b)^2 + 5
  in increased_expr - initial_expr = 4*x*b + 2*b^2 ∨ 
     decreased_expr - initial_expr = -4*x*b + 2*b^2 :=
by
  sorry

end change_in_expression_l695_695374


namespace linear_polynomial_divisible_49_l695_695595

theorem linear_polynomial_divisible_49 {P : ℕ → Polynomial ℚ} :
    let Q := Polynomial.C 1 * (Polynomial.X ^ 8) + Polynomial.C 1 * (Polynomial.X ^ 7)
    ∃ a b x, (P x) = Polynomial.C a * Polynomial.X + Polynomial.C b ∧ a ≠ 0 ∧ 
              (∀ i, P (i + 1) = (Polynomial.C 1 * Polynomial.X + Polynomial.C 1) * P i ∨ 
                            P (i + 1) = Polynomial.derivative (P i)) →
              (a - b) % 49 = 0 :=
by
  sorry

end linear_polynomial_divisible_49_l695_695595


namespace greatest_integer_of_set_is_152_l695_695519

-- Define the conditions
def median (s : Set ℤ) : ℤ := 150
def smallest_integer (s : Set ℤ) : ℤ := 140
def consecutive_even_integers (s : Set ℤ) : Prop := 
  ∀ x ∈ s, ∃ y ∈ s, x = y ∨ x = y + 2

-- The main theorem
theorem greatest_integer_of_set_is_152 (s : Set ℤ) 
  (h_median : median s = 150)
  (h_smallest : smallest_integer s = 140)
  (h_consecutive : consecutive_even_integers s) : 
  ∃ greatest : ℤ, greatest = 152 := 
sorry

end greatest_integer_of_set_is_152_l695_695519


namespace find_group_weight_l695_695653

def molecular_weight := 74
def atomic_weight_Ca := 40.08
def num_groups := 2

noncomputable def group_weight := (molecular_weight - atomic_weight_Ca) / num_groups

theorem find_group_weight :
  molecular_weight - atomic_weight_Ca = group_weight * num_groups := by
  sorry

end find_group_weight_l695_695653


namespace part1_part2_l695_695436

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp x - x

theorem part1 (a : ℝ) (h1 : 0 < a) (h2 : ∀ x, 1 < x → f a x < f a (x + 1)) (h3 : ∃ x > 2, g a x = Real.min (g a '' (Set.Ioi 2))) :
  a ∈ Set.Ioo 0 (1 / Real.exp 2) := sorry

theorem part2 (a : ℝ) (h1 : 0 < a) (h2 : ∀ x, f a x ≠ 0) (h3 : ∀ x, g a x ≠ 0) :
  a ∈ Set.Ioo (1 / Real.exp 1) Real.exp 1 := sorry

end part1_part2_l695_695436


namespace visitors_previous_day_l695_695212

theorem visitors_previous_day (total_visitors_85_days : ℕ) (visitors_today : ℕ) (total_visitors_85_days = 829) (visitors_today = 784) : (total_visitors_85_days - visitors_today = 45) :=
by
  intros
  sorry

end visitors_previous_day_l695_695212


namespace total_house_rent_l695_695915

theorem total_house_rent (P S R : ℕ)
  (h1 : S = 5 * P)
  (h2 : R = 3 * P)
  (h3 : R = 1800) : 
  S + P + R = 5400 :=
by
  sorry

end total_house_rent_l695_695915


namespace polar_to_rectangular_coordinates_l695_695249

-- Define the given conditions in Lean 4
def r : ℝ := 4
def theta : ℝ := Real.pi / 3

-- Define the conversion formulas
def x : ℝ := r * Real.cos theta
def y : ℝ := r * Real.sin theta

-- State the proof problem
theorem polar_to_rectangular_coordinates : (x = 2) ∧ (y = 2 * Real.sqrt 3) := by
  -- Sorry is used to indicate the proof is omitted
  sorry

end polar_to_rectangular_coordinates_l695_695249


namespace different_algorithms_for_same_problem_l695_695157

-- Define the basic concept of a problem
def Problem := Type

-- Define what it means for something to be an algorithm solving a problem
def Algorithm (P : Problem) := P -> Prop

-- Define the statement to be true: Different algorithms can solve the same problem
theorem different_algorithms_for_same_problem (P : Problem) (A1 A2 : Algorithm P) :
  P = P -> A1 ≠ A2 -> true :=
by
  sorry

end different_algorithms_for_same_problem_l695_695157


namespace probability_neither_square_nor_cube_l695_695566

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695566


namespace problem_proof_l695_695691

noncomputable def problem_statement : ℝ :=
  (0.027 ^ (-1 / 3)) - (Real.logBase 3 2 * Real.logBase 8 3)

theorem problem_proof : problem_statement = 3 := by
  sorry

end problem_proof_l695_695691


namespace probability_neither_perfect_square_nor_cube_l695_695527

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695527


namespace probability_not_square_or_cube_l695_695553

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695553


namespace one_fourth_difference_l695_695687

theorem one_fourth_difference :
  (1 / 4) * ((9 * 5) - (7 + 3)) = 35 / 4 :=
by sorry

end one_fourth_difference_l695_695687


namespace number_of_correct_statements_l695_695159

theorem number_of_correct_statements :
  let statements := [
    fractional.is_fraction_iff (sqrt 3) 2,
    ∀ (a b : ℝ), irrational a → irrational b → irrational (a * b),
    ∀ (a b : ℝ), irrational a → irrational b → a > 0 → b > 0 → irrational (a + b),
    irrational (0.1010010001 : ℝ)
  ] in
  (∃ n, list.count, (statements, λ statement, statement) = n ∧ n = 1) :=
by
  sorry

end number_of_correct_statements_l695_695159


namespace probability_neither_square_nor_cube_l695_695558

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695558


namespace women_left_l695_695404

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l695_695404


namespace locus_of_vertex_C_l695_695984

theorem locus_of_vertex_C :
  ∀ (A B C D : Point), 
    dist A B = 2 → 
    is_median A D C B → 
    dist A D = 3 / 2 → 
  ∃ (center : Point) (r : ℝ), 
    is_circle (locus_of C) center r ∧ 
    center = ⟨-2, 0⟩ ∧ 
    r = 3 :=
by sorry

end locus_of_vertex_C_l695_695984


namespace work_completion_days_l695_695650

structure WorkProblem :=
  (total_work : ℝ := 1) -- Assume total work to be 1 unit
  (days_A : ℝ := 30)
  (days_B : ℝ := 15)
  (days_together : ℝ := 5)

noncomputable def total_days_taken (wp : WorkProblem) : ℝ :=
  let work_per_day_A := 1 / wp.days_A
  let work_per_day_B := 1 / wp.days_B
  let work_per_day_together := work_per_day_A + work_per_day_B
  let work_done_together := wp.days_together * work_per_day_together
  let remaining_work := wp.total_work - work_done_together
  let days_for_A := remaining_work / work_per_day_A
  wp.days_together + days_for_A

theorem work_completion_days (wp : WorkProblem) : total_days_taken wp = 20 :=
by
  sorry

end work_completion_days_l695_695650


namespace focus_reflection_is_correct_l695_695506

-- Define the original parabola and the reflection line
def parabola (x y : ℝ) : Prop := y^2 = -8 * x
def reflection_line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-2, 0)

-- Define the function for reflecting a point over the line y = x - 1
def reflect_over_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in
  let s := (y - x + 1) / 2 in
  (x + 2 * s, y - 2 * s)

-- Assertion about the focus point after reflection
theorem focus_reflection_is_correct :
  reflect_over_line focus = (1, -3) :=
sorry

end focus_reflection_is_correct_l695_695506


namespace count_good_permutations_le_l695_695643

-- Mathematical definitions based on the problem conditions
def is_good_permutation (l : List ℕ) : Prop :=
  ¬ ∃ (sub : List ℕ), sub.length = 10 ∧ sub.is_strict_anti

-- Main theorem statement
theorem count_good_permutations_le (n : ℕ) :
  (Finset.filter is_good_permutation (Finset.permutations (Finset.range n))).card ≤ 81^n :=
sorry

end count_good_permutations_le_l695_695643


namespace women_left_l695_695403

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l695_695403


namespace solve_for_a_l695_695051

noncomputable theory

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_a (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  is_pure_imaginary ((1 + i) * (1 + a * i)) ↔ a = 1 :=
sorry

end solve_for_a_l695_695051


namespace arithmetic_example_l695_695628

theorem arithmetic_example : 4 * (9 - 6) - 8 = 4 := by
  sorry

end arithmetic_example_l695_695628


namespace cylinder_original_radius_l695_695264

theorem cylinder_original_radius
  (r : ℝ)
  (h_original : ℝ := 4)
  (h_increased : ℝ := 3 * h_original)
  (volume_eq : π * (r + 8)^2 * h_original = π * r^2 * h_increased) :
  r = 4 + 4 * Real.sqrt 5 :=
sorry

end cylinder_original_radius_l695_695264


namespace log2_arithmetic_extremum_l695_695851

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x^2 + 6 * x

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

theorem log2_arithmetic_extremum :
  ∃ (a d : ℕ), 
    let a1 := arithmetic_sequence a d 0 in
    let a4031 := arithmetic_sequence a d 4030 in
    a1 = a4031 ∧ 
    (∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ a1 + a4031 = 8 ∧ 
    2 * (arithmetic_sequence a d 2015) = a1 + a4031 ∧
    log 2 (arithmetic_sequence a d 2015) = 2) :=
sorry

end log2_arithmetic_extremum_l695_695851


namespace perpendicular_bisector_chord_AB_l695_695336

theorem perpendicular_bisector_chord_AB :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 5}
  let line := {p : ℝ × ℝ | 2 * p.1 + 3 * p.2 + 1 = 0}
  let center := (1, -2)
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ circle ∧ B ∈ circle ∧ A ∈ line ∧ B ∈ line ∧
    ∃ L : ℝ → ℝ × ℝ → ℝ, (L center = 0) ∧ L = λ (x : ℝ) (p : ℝ × ℝ), 3 * p.1 - 2 * p.2 - 7 :=
by sorry

end perpendicular_bisector_chord_AB_l695_695336


namespace ratio_kaydence_brother_father_age_l695_695378

-- Statement of the proof problem in Lean
theorem ratio_kaydence_brother_father_age 
  (total_age : ℕ)
  (father_age : ℕ)
  (mother_age : ℕ)
  (sister_age : ℕ)
  (kaydence_age : ℕ)
  (brother_age : ℕ)
  (h1 : total_age = 200)
  (h2 : father_age = 60)
  (h3 : mother_age = father_age - 2)
  (h4 : sister_age = 40)
  (h5 : kaydence_age = 12)
  (h6 : brother_age = total_age - (father_age + mother_age + sister_age + kaydence_age)) :
  brother_age / father_age = 1 / 2 :=
begin
  sorry
end

end ratio_kaydence_brother_father_age_l695_695378


namespace probability_neither_square_nor_cube_l695_695573

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695573


namespace solve_AE_length_l695_695951

open Real

noncomputable def length_of_AE (AB CD AC : ℝ) (eq_areas : TriangleAreaEqual AED BEC) : ℝ :=
  if h : (AB = 10 ∧ CD = 15 ∧ AC = 17) then
    34 / 5
  else 
    0

theorem solve_AE_length (AB CD AC : ℝ) (E : Point) (AED BEC : Triangle) (h1 : AB = 10) (h2: CD = 15) (h3: AC = 17) (h4: TriangleAreaEqual AED BEC) :
  length_of_AE AB CD AC h4 = 34 / 5 :=
by
  rw [length_of_AE]
  split_ifs
  . rfl
  . contradiction

end solve_AE_length_l695_695951


namespace cube_root_power_simplify_l695_695484

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695484


namespace true_proposition_l695_695306

variable (x : ℝ)

def p := ∀ x : ℝ, 2^x > 0
def q := (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)

theorem true_proposition (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end true_proposition_l695_695306


namespace eggs_in_each_basket_l695_695029

theorem eggs_in_each_basket
  (total_red_eggs : ℕ)
  (total_orange_eggs : ℕ)
  (h_red : total_red_eggs = 30)
  (h_orange : total_orange_eggs = 45)
  (eggs_in_each_basket : ℕ)
  (h_at_least : eggs_in_each_basket ≥ 5) :
  (total_red_eggs % eggs_in_each_basket = 0) ∧ 
  (total_orange_eggs % eggs_in_each_basket = 0) ∧
  eggs_in_each_basket = 15 := sorry

end eggs_in_each_basket_l695_695029


namespace area_ratio_KLM_ABC_l695_695864

-- Triangle definitions and conditions
variables {K L M A B C : Type}
variables [triangle KLM] -- Denotes a triangle with vertices K, L, M
variables [incircle_touches_ACB KLM A B C] -- Denotes incircle touching at points A, B, C

-- Radii conditions
parameter (r R : ℝ) (hR3r : R = 3 * r)

-- Areas of the triangles KLM and ABC
noncomputable def area_KLM (a b c : ℝ) : ℝ := 
   (1 / 2) * b * c * sin (α K L M)

noncomputable def area_ABC (a b c : ℝ) : ℝ := 
   (1 / 2) * r * r * (sin (α K L M) + sin (β K L M) + sin (γ K L M))

-- Final theorem statement
theorem area_ratio_KLM_ABC (a b c : ℝ) (α β γ : ℝ) :
  (area_KLM a b c) / (area_ABC a b c) = 6 :=
sorry

end area_ratio_KLM_ABC_l695_695864


namespace slant_angle_range_l695_695313

theorem slant_angle_range (P : ℝ → ℝ) (x : ℝ) (α : ℝ) (hP : P x = 4 / (Real.exp x + 1))
  (hα : ∃ t, α = Real.arctan (-(4 * Real.exp t) / (Real.exp t + 1)^2)) :
  α ∈ Ico (3 * Real.pi / 4) Real.pi :=
by sorry

end slant_angle_range_l695_695313


namespace problem_statement_l695_695310

-- Define the given conditions
def directly_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k * (3 * x - 2)

-- Define the main problem as a theorem
theorem problem_statement : 
  (∀ x : ℝ, ∃ y : ℝ, directly_proportional y x) →
  (∃ y : ℝ, directly_proportional y 2 ∧ y = 8) →
  (∀ x : ℝ, y = 6 * x - 4) →
  ∃ (k : ℝ), k = 2 ∧
  let y := 6 * x - 4 in 
  (∃ (y_val : ℝ), x = -2 → y_val = y ∧ y_val = -16) ∧
  (let A := (2 / 3, 0 : ℝ) in
  let B := (0, -4 : ℝ) in
  let O := (0, 0 : ℝ) in
  ∃ (area : ℝ), area = 1 / 2 * ((2 : ℝ) / 3) * 4 ∧ area = 4 / 3) :=
by sorry

end problem_statement_l695_695310


namespace part1_part2_l695_695001

-- Given sequences a and b
def sequence_a : ℕ → ℝ
| 0 => 1
| 1 => 3
| 2 => 5
| 3 => 6
| _ => 0 -- for out-of-bound indices

def sequence_b : ℕ → ℝ
| 0 => 2
| 1 => 3
| 2 => 10
| 3 => 7
| _ => 0 -- for out-of-bound indices

-- Function to calculate distance between sequences
def distance (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in (Finset.range n), abs (a i - b i)

-- Proof statements
theorem part1 : distance sequence_a sequence_b 4 = 7 :=
by
  sorry

def b_sequence : ℕ → ℝ
| (4*k - 3) => 2
| (4*k - 2) => -3
| (4*k - 1) => -1/2
| 4*k => 1/3
| _ => 0

def c_sequence : ℕ → ℝ
| (4*k - 3) => 3
| (4*k - 2) => -2
| (4*k - 1) => -1/3
| 4*k => 1/2
| _ => 0

theorem part2 (m : ℕ) (h : m < 3456) : distance b_sequence c_sequence m < 2016 :=
by
  sorry

end part1_part2_l695_695001


namespace right_triangle_area_l695_695934

theorem right_triangle_area (a b c p : ℝ) (h1 : a = b) (h2 : 3 * p = a + b + c)
  (h3 : c = Real.sqrt (2 * a ^ 2)) :
  (1/2) * a ^ 2 = (9 * p ^ 2 * (3 - 2 * Real.sqrt 2)) / 4 :=
by
  sorry

end right_triangle_area_l695_695934


namespace exists_sigma_sum_lt_half_l695_695038

theorem exists_sigma_sum_lt_half (n : ℕ) (h_pos : n > 0) : 
  ∃ (σ : Fin n → Fin n), (Function.Injective σ) ∧ (∑ k in Finset.range n, (k + 1) / ((k + 1 + σ ⟨k, Nat.lt_of_lt_of_le (Fin.is_lt k) n⟩).val + 1)^2 < 1 / 2) :=
sorry

end exists_sigma_sum_lt_half_l695_695038


namespace tim_kittens_l695_695949

theorem tim_kittens (initial_kittens : ℕ) (given_to_jessica_fraction : ℕ) (saras_kittens : ℕ) (adopted_fraction : ℕ) 
  (h_initial : initial_kittens = 12)
  (h_fraction_to_jessica : given_to_jessica_fraction = 3)
  (h_saras_kittens : saras_kittens = 14)
  (h_adopted_fraction : adopted_fraction = 2) :
  let kittens_after_jessica := initial_kittens - initial_kittens / given_to_jessica_fraction
  let total_kittens_after_sara := kittens_after_jessica + saras_kittens
  let adopted_kittens := saras_kittens / adopted_fraction
  let final_kittens := total_kittens_after_sara - adopted_kittens
  final_kittens = 15 :=
by {
  sorry
}

end tim_kittens_l695_695949


namespace complex_number_solution_l695_695002

theorem complex_number_solution (a : ℝ) (z : ℂ) (h_im : ∀ r : ℝ, z = r * complex.I) (h_eq : (2 - complex.I) * z = a + complex.I) : a = 1 / 2 :=
  sorry

end complex_number_solution_l695_695002


namespace preimage_of_3_1_is_2_half_l695_695296

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * p.2, p.1 - 2 * p.2)

theorem preimage_of_3_1_is_2_half :
  (∃ x y : ℝ, f (x, y) = (3, 1) ∧ (x = 2 ∧ y = 1/2)) :=
by
  sorry

end preimage_of_3_1_is_2_half_l695_695296


namespace cost_of_producing_one_component_l695_695185

-- Define the conditions as constants
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_cost : ℕ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℕ := 195

-- Define the cost of producing one component as a variable
variable (C : ℕ)

/-- Prove that C must be less than or equal to 80 given the conditions -/
theorem cost_of_producing_one_component : 
  150 * C + 150 * shipping_cost_per_unit + fixed_monthly_cost ≤ 150 * selling_price_per_component → C ≤ 80 :=
by
  sorry

end cost_of_producing_one_component_l695_695185


namespace min_value_of_function_product_inequality_l695_695647

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality_l695_695647


namespace sum_first_10_common_terms_l695_695749

def AP_term (n : ℕ) : ℕ := 5 + 3 * n
def GP_term (k : ℕ) : ℕ := 10 * 2^k
def common_terms (m : ℕ) : ℕ := 20 * 4^m

theorem sum_first_10_common_terms :
  ∑ m in Finset.range 10, common_terms m = 6990500 :=
by sorry

end sum_first_10_common_terms_l695_695749


namespace average_age_of_team_l695_695641

theorem average_age_of_team
  (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age : ℕ) (remaining_players_count : ℕ)
  (A : ℕ) :
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age = 28 →
  remaining_players_count = team_size - 2 →
  (11 * A - (captain_age + wicket_keeper_age) = remaining_players_count * (A - 1)) →
  A = 22 :=
by
  intros team_size_eq captain_age_eq wicket_keeper_age_eq remaining_players_eq age_eq
  rw team_size_eq at *
  rw captain_age_eq at *
  rw wicket_keeper_age_eq at *
  rw remaining_players_eq at *
  sorry

end average_age_of_team_l695_695641


namespace sarah_father_double_age_in_2030_l695_695449

noncomputable def sarah_birthday_double_age_year : ℕ :=
let sarah_age_2010 := 10 in
let father_age_2010 := 6 * sarah_age_2010 in
let year := 2010 in
let x := 20 in -- Solved value as per the problem statement
year + x

/-- 
  Suppose Sarah's age in 2010 is 10 years and her father's age in 2010 is 60 years.
  Prove that the year in which Sarah's father's age will be double her age is 2030.
--/
theorem sarah_father_double_age_in_2030 :
  let sarah_age_2010 := 10 in
  let father_age_2010 := 6 * sarah_age_2010 in
  let year := 2010 in
  let x := 20 in
  year + x = 2030 :=
by {
  let sarah_age_2010 := 10,
  let father_age_2010 := 6 * sarah_age_2010,
  let year := 2010,
  let x := 20,
  have : father_age_2010 + x = 2 * (sarah_age_2010 + x) := by {
    simp [sarah_age_2010, father_age_2010, x],
    linarith,
  },
  exact rfl,
}

end sarah_father_double_age_in_2030_l695_695449


namespace even_plus_abs_odd_is_even_l695_695061

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Proof problem statement
theorem even_plus_abs_odd_is_even (f g : ℝ → ℝ) 
  (hf : is_even_function f) (hg : is_odd_function g) : is_even_function (λ x, f x + |g x|) :=
sorry

end even_plus_abs_odd_is_even_l695_695061


namespace intersecting_circles_distance_l695_695006

variables (R r d : ℝ)
variable (hR : R > r)

theorem intersecting_circles_distance :
  (R - r < d ∧ d < R + r) → d < R + r ∧ R - r < d := 
begin
  sorry
end

end intersecting_circles_distance_l695_695006


namespace angle_is_orthogonal_l695_695041

noncomputable def a : ℝ^3 := ⟨1, -2, -5⟩
noncomputable def b : ℝ^3 := ⟨Real.sqrt 7, 4, -1⟩
noncomputable def c : ℝ^3 := ⟨13, -4, 17⟩

theorem angle_is_orthogonal :
  let ac := a • c
  let ab := a • b
  ac • b - ab • c = 0 :=
by
  sorry

end angle_is_orthogonal_l695_695041


namespace median_interval_60_64_l695_695604

theorem median_interval_60_64 
  (students : ℕ) 
  (f_45_49 f_50_54 f_55_59 f_60_64 : ℕ) :
  students = 105 ∧ 
  f_45_49 = 8 ∧ 
  f_50_54 = 15 ∧ 
  f_55_59 = 20 ∧ 
  f_60_64 = 18 ∧ 
  (8 + 15 + 20 + 18) ≥ (105 + 1) / 2
  → 60 ≤ (105 + 1) / 2  ∧ (105 + 1) / 2 ≤ 64 :=
sorry

end median_interval_60_64_l695_695604


namespace trees_distance_approx_l695_695843

noncomputable def distance_between_trees (L : ℝ) (n : ℕ) : ℝ :=
  L / (n - 1)

theorem trees_distance_approx (L : ℝ) (n : ℕ) (hL : L = 800) (hn : n = 52) :
  distance_between_trees L n ≈ 15.686 := by
  have h_distance : distance_between_trees 800 52 = 800 / 51 := by
    unfold distance_between_trees
    rw [hL, hn]
  rw [h_distance]
  -- Prove that 800 / 51 is approximately 15.686
  sorry

end trees_distance_approx_l695_695843


namespace proper_subsets_of_S_l695_695344

-- Define the set S
def S : Set ℕ := {1, 2}

-- Define a function to count the number of proper subsets
def num_proper_subsets (S : Set ℕ) : ℕ := 2 ^ S.card - 1

-- The statement we want to prove
theorem proper_subsets_of_S : num_proper_subsets S = 3 := by
  sorry

end proper_subsets_of_S_l695_695344


namespace f_monotonicity_l695_695785

noncomputable def f (a x : ℝ) : ℝ :=
  abs (x^2 - a * x) - log x

theorem f_monotonicity (a : ℝ) :
  (if a < 1 then
     ∃ b : ℝ, b = (a + sqrt (a^2 + 8)) / 4 ∧
              (∀ x ∈ set.Ioo 0 b, deriv (f a) x < 0) ∧
              (∀ x ∈ set.Ioi b, deriv (f a) x > 0)
   else if 1 ≤ a ∧ a ≤ 2 * sqrt 2 then
     (∀ x ∈ set.Ioo 0 a, deriv (f a) x < 0) ∧
     (∀ x ∈ set.Ioi a, deriv (f a) x > 0)
   else
     ∃ c d : ℝ, c = (a - sqrt (a^2 - 8)) / 4 ∧
                d = (a + sqrt (a^2 - 8)) / 4 ∧
                (∀ x ∈ set.Ioo 0 c, deriv (f a) x < 0) ∧
                (∀ x ∈ set.Ioo c d, deriv (f a) x > 0) ∧
                (∀ x ∈ set.Ioo d a, deriv (f a) x < 0) ∧
                (∀ x ∈ set.Ioi a, deriv (f a) x > 0))
  := sorry

end f_monotonicity_l695_695785


namespace center_of_mass_is_centroid_l695_695610

-- Define the conditions of the problem
variables {α : Type*} [AddCommGroup α] [VectorSpace ℝ α]
variables {A B C : α}
variables (flies : Fin 3 → α)
variable  (fixed_pt : α)

-- Define equal mass and center of mass conditions
def equal_mass (flies : Fin 3 → α) : Prop := 
  flies 0 = flies 1 ∧ flies 1 = flies 2

def center_of_mass_fixed (flies : Fin 3 → α) (fixed_pt : α) : Prop :=
  (flies 0 + flies 1 + flies 2) / 3 = fixed_pt

-- Lean theorem statement
theorem center_of_mass_is_centroid (flies : Fin 3 → α) (fixed_pt : α) :
  equal_mass flies →
  center_of_mass_fixed flies fixed_pt →
  fixed_pt = (((A + B + C) / 3) : α) :=
by
  sorry

end center_of_mass_is_centroid_l695_695610


namespace jerome_ratio_l695_695412

-- Definitions for the conditions
def Jerome_gave_meg : ℕ := 8
def Jerome_gave_bianca : ℕ := 3 * Jerome_gave_meg
def Jerome_left : ℕ := 54
def Jerome_initial : ℕ := Jerome_left + Jerome_gave_meg + Jerome_gave_bianca

-- The theorem we aim to prove
theorem jerome_ratio :
  Jerome_initial.toNat.gcd Jerome_left.toNat = 1 → 
  Jerome_initial.toNat / 27 = 43 ∧ Jerome_left.toNat / 27 = 2 :=
by sorry

end jerome_ratio_l695_695412


namespace rain_difference_l695_695902

variable (rain_mondays : ℕ → ℕ)
variable (rain_tuesdays : ℕ → ℕ)

def total_rain_mondays (n : ℕ) : ℕ := 1.5 * 7
def total_rain_tuesdays (n : ℕ) : ℕ := 2.5 * 9

theorem rain_difference (n : ℕ) : 
  total_rain_tuesdays n - total_rain_mondays n = 12 :=
  sorry

end rain_difference_l695_695902


namespace length_segment_ZZ_l695_695138

variable (Z : ℝ × ℝ) (Z' : ℝ × ℝ)

theorem length_segment_ZZ' 
  (h_Z : Z = (-5, 3)) (h_Z' : Z' = (5, 3)) : 
  dist Z Z' = 10 := by
  sorry

end length_segment_ZZ_l695_695138


namespace remaining_card_number_l695_695282

theorem remaining_card_number (A B C D E F G H : ℕ) (cards : Finset ℕ) 
  (hA : A + B = 10) 
  (hB : C - D = 1) 
  (hC : E * F = 24) 
  (hD : G / H = 3) 
  (hCards : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : A ∉ cards ∧ B ∉ cards ∧ C ∉ cards ∧ D ∉ cards ∧ E ∉ cards ∧ F ∉ cards ∧ G ∉ cards ∧ H ∉ cards) :
  7 ∈ cards := 
by
  sorry

end remaining_card_number_l695_695282


namespace bounded_iff_summable_l695_695421

theorem bounded_iff_summable (p : ℕ → ℝ) (x v : ℕ → ℝ) 
  (h_p_nonneg : ∀ n, 0 ≤ p n)
  (h_ini : x 0 = 0 ∧ v 0 = 1)
  (h_rec_x : ∀ n, x (n + 1) = x n + v n)
  (h_rec_v : ∀ n, v (n + 1) = v n - p (n + 1) * x (n + 1))
  (h_v_lim : ∀ ε > 0, ∃ N, ∀ n ≥ N, v n < ε) :
  (∃ C, ∀ n, |x n| ≤ C) ↔ (∑' n, n * p n < ∞) :=
sorry

end bounded_iff_summable_l695_695421


namespace min_sum_AM_GM_l695_695706

open Real

theorem min_sum_AM_GM (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a)) ≥ real.root 3 (3 / 54) :=
by
  sorry

end min_sum_AM_GM_l695_695706


namespace compute_XY_distance_l695_695037

-- Definitions of the points and triangle
def Point := (ℝ × ℝ)
def A : Point := (0, 0)
def B : Point := (20, 0)
def C : Point := (11, 11 * Real.sqrt 3)

def is_perpendicular (P Q R : Point) : Prop :=
  let ⟨x1, y1⟩ := P
  let ⟨x2, y2⟩ := Q
  let ⟨x3, y3⟩ := R
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

def distance (P Q : Point) : ℝ :=
  let ⟨x1, y1⟩ := P
  let ⟨x2, y2⟩ := Q
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem compute_XY_distance :
  ∃ X Y : Point, is_perpendicular B X X ∧ is_perpendicular C Y Y ∧ X.2 = 11 * Real.sqrt 3 ∧ Y.2 = 11 * Real.sqrt 3 ∧ distance X Y = 21 :=
by {
  sorry
}

end compute_XY_distance_l695_695037


namespace find_number_l695_695631

theorem find_number (x : ℤ) : 17 * (x + 99) = 3111 → x = 84 :=
by
  sorry

end find_number_l695_695631


namespace tan_angle_A1_PA2_correct_l695_695804

-- Define the hyperbola with parameters and constraints
structure Hyperbola (a b : ℝ) (e : ℝ) :=
(h_pos : a > 0)
(k_pos : b > 0)
(eccentricity : e = 2)
(hyp_eq : ∀ x y, (x^2) / (a^2) - (y^2) / (b^2) = 1)
(focus_and_point : ∃ F P : ℝ × ℝ, (P.fst ^ 2 / a ^ 2 - P.snd ^ 2 / b ^ 2 = 1) ∧
                   (F.fst ^ 2 + F.snd ^ 2 = c ^ 2) ∧ -- where c = 2a
                   ∃ A2 : ℝ × ℝ, (P - F).snd * (P - A2).snd = - (P - F).fst * (P - A2).fst)  

-- Define the problem conditions
def problem_conditions : Prop :=
∃ a b : ℝ, ∃ C : Hyperbola a b 2, C.h_pos ∧ C.k_pos

-- Define the mathematical problem
def tan_A1_PA2 (a b : ℝ) (C : Hyperbola a b 2) : ℝ :=
sorry

-- The main goal in Lean is to solve this problem under the given conditions
theorem tan_angle_A1_PA2_correct : problem_conditions → ∀ a b : ℝ, ∀ C : Hyperbola a b 2, tan_A1_PA2 a b C = 1 / 2 :=
by
  intro h a b C
  sorry

end tan_angle_A1_PA2_correct_l695_695804


namespace cube_root_power_simplify_l695_695485

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695485


namespace paint_used_l695_695413

theorem paint_used (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) 
  (first_week_paint : ℚ) (remaining_paint : ℚ) (second_week_paint : ℚ) (total_used_paint : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  first_week_paint = first_week_fraction * total_paint →
  remaining_paint = total_paint - first_week_paint →
  second_week_paint = second_week_fraction * remaining_paint →
  total_used_paint = first_week_paint + second_week_paint →
  total_used_paint = 120 := sorry

end paint_used_l695_695413


namespace probability_neither_perfect_square_nor_cube_l695_695526

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695526


namespace least_positive_integer_div_by_four_distinct_primes_l695_695963

theorem least_positive_integer_div_by_four_distinct_primes :
  ∃ (n : ℕ), (∀ p ∈ {2, 3, 5, 7}, p ∣ n) ∧ (∀ m, (∀ p ∈ {2, 3, 5, 7}, p ∣ m) → m ≥ n) ∧ n = 210 :=
by
  sorry

end least_positive_integer_div_by_four_distinct_primes_l695_695963


namespace sum_of_selected_balls_is_odd_l695_695284

-- Define the set of balls numbered from 1 to 11.
def balls : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

-- Define a function to partition the balls into odd and even numbers.
def partitionBalls (balls : List ℤ) : List ℤ × List ℤ :=
  (balls.filter (λ n, n % 2 = 1), balls.filter (λ n, n % 2 = 0))

-- Define combinations function.
noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of ways to choose 5 balls so the sum is odd
def numberOfWays : ℕ :=
  let (odds, evens) := partitionBalls balls
  combinations odds.length 1 * combinations evens.length 4 +
  combinations odds.length 3 * combinations evens.length 2 +
  combinations odds.length 5

theorem sum_of_selected_balls_is_odd : numberOfWays = 236 :=
  by
    sorry

end sum_of_selected_balls_is_odd_l695_695284


namespace length_OC_in_terms_of_s_and_c_l695_695657

-- Declare the necessary definitions and assumptions
variables {O A B C : Type} 
variables [has_dist O A] [has_angle A O B] [has_distance A B]
noncomputable def circle_centered_O (O : Point) (radius : ℝ) : Circle :=
  sorry -- Define a circle centered at O with given radius

noncomputable def point_on_circle (A : Point) (C : Circle) : Prop :=
  sorry -- State that A lies on Circle C

noncomputable def segment_tangent (A B : Point) (C : Circle) : Prop :=
  sorry -- State that AB is tangent to circle at A

noncomputable def angle_AOB := 2 * theta
noncomputable def point_on_OA_ratio := lambda OC OA, 4 * OC = 3 * OA

noncomputable def angle_bisector (O A B : Point) (θ : ℝ) :=
  sorry -- Define the angle bisector condition

theorem length_OC_in_terms_of_s_and_c (O A B C : Point) (θ : ℝ) (s : ℝ) (c : ℝ) :
  (angle_AOB = 2 * θ) →
  (OC_on_OA_ratio OC OA) →
  (angle_bisector O A B θ) →
  OC = 3 / 4 :=
by sorry

end length_OC_in_terms_of_s_and_c_l695_695657


namespace find_expression_and_area_l695_695764

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := - x^2 - 4 * x + 1

theorem find_expression_and_area :
  (∃ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c = 0 ∧ (∃ x : ℝ, f'(x) = 2 * x + 2) ∧ ∀ x, (f(x) = a * x^2 + b * x + c)) ∧
  let f (x : ℝ) := x^2 + 2 * x + 1 in
  let g (x : ℝ) := - x^2 - 4 * x + 1 in
  (∃ (area : ℝ), area = ∫ x in (-3)..0, (g x - f x) dx ∧ area = 9) :=
begin
  have h_deriv : ∀ x : ℝ, has_deriv_at f (2 * x + 2) x := sorry,
  use [1, 2, 1],
  split,
  { split,
    { norm_num },
    { split,
      { sorry },
      { intro x,
        simp [f] } } },
  { unfold f g,
    use 9,
    split,
    { sorry },
    { norm_num } }
end

end find_expression_and_area_l695_695764


namespace tangent_line_inclination_range_l695_695516

theorem tangent_line_inclination_range:
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 → (0 ≤ 2*x ∧ 2*x ≤ 1 ∨ -1 ≤ 2*x ∧ 2*x < 0) →
    ∃ (α : ℝ), (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π) :=
sorry

end tangent_line_inclination_range_l695_695516


namespace fourth_derivative_l695_695270

noncomputable def y (x : ℝ) : ℝ := (x^2 + 3) * Real.log(x - 3)

theorem fourth_derivative (x : ℝ) (h : x > 3) : 
  Real.deriv^[4] (λ x, (x^2 + 3) * Real.log(x - 3)) x = (-2 * x^2 + 24 * x - 126) / (x - 3)^4 :=
sorry

end fourth_derivative_l695_695270


namespace growing_path_product_l695_695241

structure Point where
  x : ℕ
  y : ℕ

/-- Definition of the distance between two points in a 5x5 grid -/
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Definition of a growing path -/
def is_growing_path (path : List Point) : Prop :=
  ∀ (i j : ℕ), i < j → j < path.length → distance (path.get i) (path.get (i+1)) < distance (path.get (j-1)) (path.get j)

/-- The main theorem stating the product of the maximum number of points in a growing path (m) 
    and the number of growing paths consisting of exactly m points (r) in the 5x5 grid is equal to 1020. -/
theorem growing_path_product :
  ∃ (m r : ℕ), 
    (∀ path, is_growing_path path → path.length ≤ m) ∧
    (∃ path, is_growing_path path ∧ path.length = m) ∧
    (number_of_growing_paths_of_length m = r) ∧
    m * r = 1020 :=
sorry

end growing_path_product_l695_695241


namespace probability_neither_square_nor_cube_l695_695530

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695530


namespace find_a_b_l695_695020

open Real

-- Define the polar equation of circle C1
def circle_C1 (θ : ℝ) : ℝ := 4 * sin θ

-- Define the polar equation of line C2
def line_C2 (θ : ℝ) : ℝ := 2 * sqrt 2 / cos (θ - π / 4)

-- Provided polar coordinates of intersection points
def intersection_point1_polar : ℝ × ℝ := (4, π / 2)
def intersection_point2_polar : ℝ × ℝ := (2 * sqrt 2, π / 4)

-- Convert polar to Cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

-- Cartesian coordinates of intersection points
def intersection_point1_cartesian := polar_to_cartesian 4 (π / 2)
def intersection_point2_cartesian := polar_to_cartesian (2 * sqrt2) (π / 4)

-- Define the parametric equations
noncomputable def parametric_x (t a : ℝ) : ℝ := t^3 + a
noncomputable def parametric_y (t b : ℝ) : ℝ := b / 2 * t^3 + 1

-- Given points P and Q in Cartesian coordinates
def point_P : ℝ × ℝ := (0, 2)
def point_Q : ℝ × ℝ := (1, 3)

-- Prove that a = -1 and b = 2 given the parametric equation constraints
theorem find_a_b (t : ℝ) : 
  let a := -1, b := 2 in
  (parametric_y t b = (b / 2) * (parametric_x t a - a)) ∧
  (point_Q.fst = parametric_x 1 a) ∧ (point_Q.snd = parametric_y 1 b) :=
sorry

end find_a_b_l695_695020


namespace tangent_line_eq_l695_695271

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp x

theorem tangent_line_eq :
  ∀ (x y : ℝ), y = f x → x = 0 → y = -1 → 1 = 2 - Real.exp 0 →
  ∃ (m : ℝ), m = 1 ∧ (x - y - 1 = 0) :=
by {
  intros x y hy hx heq1 htang,
  use [(2 - Real.exp 0)],
  dsimp,
  rw [hx, hy],
  use heq1,
  use htang,
  exact ⟨hx, by rw [hx, hy, htang]⟩,
  sorry -- proof steps to be filled in
}

end tangent_line_eq_l695_695271


namespace women_left_l695_695406

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l695_695406


namespace fraction_of_distance_traveled_by_bus_is_one_quarter_l695_695861

-- Given conditions
constant D : ℝ
constant distance_by_foot : ℝ
constant distance_by_car : ℝ

-- Proving the fraction of the distance traveled by bus is 1/4
theorem fraction_of_distance_traveled_by_bus_is_one_quarter
  (hD : D = 24)
  (hfoot : distance_by_foot = (1 / 2) * D)
  (hcar : distance_by_car = 6) :
  (D - (distance_by_foot + distance_by_car)) / D = (1 / 4) :=
by
  sorry

end fraction_of_distance_traveled_by_bus_is_one_quarter_l695_695861


namespace estimate_of_high_scores_l695_695658

open ProbabilityTheory

noncomputable def number_of_students_with_high_scores : ℝ :=
  let n := 100 in -- Number of students
  let mu := 100 in -- Mean of the normal distribution
  let sigma := 10 in -- Standard deviation of the normal distribution
  let ξ := NormalDistr μ σ in -- Normal distribution with mean 100 and standard deviation 10
  let p90_100 := 0.3 in -- Given probability P(90 ≤ ξ ≤ 100)
  let p110_plus := (0.5 - p90_100) in -- Calculated probability P(ξ ≥ 110)
  p110_plus * n -- Number of students with math scores ≥ 110

theorem estimate_of_high_scores : number_of_students_with_high_scores = 20 := sorry

end estimate_of_high_scores_l695_695658


namespace area_of_triangle_l695_695513

noncomputable def ellipse_equation (a b : ℕ) : Prop := 
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def are_foci (a b c : ℕ) : Prop := 
  c = √(a^2 - b^2)

noncomputable def perpendicular (PF1 PF2 : ℝ) : Prop := 
  PF1 * PF2 = 0

theorem area_of_triangle {a b c : ℕ} (h1 : ellipse_equation 5 3)
  (h2 : are_foci 5 3 c) 
  (h3 : ∀ P F1 F2 : ℝ, P ∈ ellipse 5 3 ∧ perpendicular (P - F1) (P - F2)) :
  ∀ S : ℝ, S = 9 := 
sorry

end area_of_triangle_l695_695513


namespace geometric_sequence_expression_l695_695357

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (h_q : q = 4)
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_sum : a 0 + a 1 + a 2 = 21) :
  ∀ n, a n = 4 ^ n :=
by sorry

end geometric_sequence_expression_l695_695357


namespace smallest_product_is_neg280_l695_695762

theorem smallest_product_is_neg280 :
  let S := {-8, -6, -4, 0, 3, 5, 7}
  in ∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = -280 := by
sorry

end smallest_product_is_neg280_l695_695762


namespace sum_of_digits_R50_div_R8_l695_695701

def R (k : ℕ) : ℕ := (10^k - 1) / 9

def digit_sum (n : ℕ) : ℕ := 
  n.digits.map (λ x => x).sum

theorem sum_of_digits_R50_div_R8 : 
  digit_sum (R 50 / R 8) = 6 :=
  sorry

end sum_of_digits_R50_div_R8_l695_695701


namespace cube_root_power_simplify_l695_695483

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l695_695483


namespace Rajesh_saves_correct_amount_l695_695083

variable (monthly_salary : ℝ) (spend_food_percentage spend_medicines_percentage save_percentage : ℝ)

def amount_spent_on_food (salary : ℝ) (percentage : ℝ) : ℝ :=
  (percentage / 100) * salary

def amount_spent_on_medicines (salary : ℝ) (percentage : ℝ) : ℝ :=
  (percentage / 100) * salary

def remaining_amount (salary food_spending medicine_spending : ℝ) : ℝ :=
  salary - (food_spending + medicine_spending)

def amount_saved (remaining : ℝ) (percentage : ℝ) : ℝ :=
  (percentage / 100) * remaining

theorem Rajesh_saves_correct_amount :
  monthly_salary = 15000 → 
  spend_food_percentage = 40 → 
  spend_medicines_percentage = 20 →
  save_percentage = 60 →
  amount_saved (remaining_amount monthly_salary (amount_spent_on_food monthly_salary spend_food_percentage) 
  (amount_spent_on_medicines monthly_salary spend_medicines_percentage)) save_percentage = 3600 :=
by
  intros
  sorry

end Rajesh_saves_correct_amount_l695_695083


namespace determine_hash_value_l695_695042

-- Define the operations '@' and '!' based on the problem statement
def at (a b : ℕ) : ℕ := max a b
def excl (a b : ℕ) : ℕ := min a b
def hash (a b : ℕ) : ℕ := excl (at a b) b

-- Given conditions 
def a := 5
def b := 3

-- Statement of the theorem
theorem determine_hash_value : hash a b = 3 := by
  -- This proof should be filled out in Lean
  sorry

end determine_hash_value_l695_695042


namespace farm_section_areas_l695_695681

theorem farm_section_areas (n : ℕ) (total_area : ℕ) (sections : ℕ) 
  (hn : sections = 5) (ht : total_area = 300) : total_area / sections = 60 :=
by
  sorry

end farm_section_areas_l695_695681


namespace carson_gardening_l695_695692

theorem carson_gardening : 
  ∀ (lines_to_mow : ℕ) (time_per_line : ℕ) (total_gardening_time : ℕ)
    (flowers_per_row : ℕ) (time_per_flower : ℚ),
    lines_to_mow = 40 →
    time_per_line = 2 →
    total_gardening_time = 108 →
    flowers_per_row = 7 →
    time_per_flower = (1/2 : ℚ) →
    let time_mowing := lines_to_mow * time_per_line in
    let remaining_time := total_gardening_time - time_mowing in
    let total_flowers := remaining_time / time_per_flower in
    let rows_of_flowers := total_flowers / flowers_per_row in
    rows_of_flowers = 8 :=
begin
  intros lines_to_mow time_per_line total_gardening_time flowers_per_row time_per_flower,
  intros h1 h2 h3 h4 h5,
  let time_mowing := lines_to_mow * time_per_line,
  let remaining_time := total_gardening_time - time_mowing,
  let total_flowers := remaining_time / time_per_flower,
  let rows_of_flowers := total_flowers / flowers_per_row,
  sorry,
end

end carson_gardening_l695_695692


namespace octagon_area_l695_695603

theorem octagon_area (O A B C D E F G H : Point) 
  (squares_centered_at_O : ∀ (p : Point), p ∈ {A, B, C, D, E, F, G, H} → dist O p = 1)
  (AB_length : dist A B = 25 / 99) 
  (area_Octagon : Area.Octagon A B C D E F G H = 100 / 99) 
  : 100 + 99 = 199 :=
by
  sorry

end octagon_area_l695_695603


namespace circle_diameter_l695_695999

theorem circle_diameter (A : ℝ) (h : A = π / 4) : ∃ d : ℝ, d = 1 :=
by
  let r := Real.sqrt (A / π)
  have h1 : r = 1/2, from sorry
  let d := 2 * r
  have d_eq : d = 1, from sorry
  use d
  exact d_eq
    

end circle_diameter_l695_695999


namespace hexagon_side_equalities_l695_695885

variables {A B C D E F : Type}

-- Define the properties and conditions of the problem
noncomputable def convex_hexagon (A B C D E F : Type) : Prop :=
  True -- Since we neglect geometric properties in this abstract.

def parallel (a b : Type) : Prop := True -- placeholder for parallel condition
def equal_length (a b : Type) : Prop := True -- placeholder for length

-- Given conditions
variables (h1 : convex_hexagon A B C D E F)
variables (h2 : parallel AB DE)
variables (h3 : parallel BC FA)
variables (h4 : parallel CD FA)
variables (h5 : equal_length AB DE)

-- Statement to prove
theorem hexagon_side_equalities : equal_length BC DE ∧ equal_length CD FA := sorry

end hexagon_side_equalities_l695_695885


namespace seating_arrangements_around_round_table_fixed_guest_l695_695848

theorem seating_arrangements_around_round_table_fixed_guest :
  (∃ (n : ℕ), n = 9) →
  (∃ (factorial : ℕ → ℕ), factorial 9 = 362880) :=
by
  intros n_exists factorial_def
  obtain ⟨n, hn⟩ := n_exists
  rw hn at factorial_def
  obtain ⟨factorial, hfactorial⟩ := factorial_def
  rw hfactorial
  exact ⟨factorial, rfl⟩

end seating_arrangements_around_round_table_fixed_guest_l695_695848


namespace num_sum_of_three_l695_695817

-- Define the arithmetic sequence with first term 4 and common difference 3
def arithmetic_seq (n : ℕ) : ℕ := 3 * n + 1

-- The set defined
def set := {arithmetic_seq n | n ∈ finset.range 15}

-- Sum of three distinct elements from the set
def sum_of_three : set ℕ :=
{ s | ∃ (k1 k2 k3 : ℕ), k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3 ∧
  k1 ∈ finset.range 15 ∧ k2 ∈ finset.range 15 ∧ k3 ∈ finset.range 15 ∧
  s = arithmetic_seq k1 + arithmetic_seq k2 + arithmetic_seq k3 }

-- The proof: the number of integers that can be expressed as a sum of three distinct numbers
theorem num_sum_of_three : finset.card sum_of_three = 37 :=
sorry

end num_sum_of_three_l695_695817


namespace women_left_l695_695405

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l695_695405


namespace daYanSeq_properties_l695_695495

-- Define the Da Yan sequence
def daYanSeq : ℕ → ℕ
| 1 := 0
| (n + 1) := 
  if n % 2 = 1 then daYanSeq n + n + 1  -- n is odd
  else daYanSeq n + n  -- n is even

-- Prove properties about the Da Yan sequence
theorem daYanSeq_properties :
  (daYanSeq 3 = 4) ∧
  (∀ n, daYanSeq (n + 2) = daYanSeq n + 2 * n + 2) ∧
  (∀ n, daYanSeq n = 
  if n % 2 = 1 then (n^2 - 1) / 2  -- n is odd
  else n^2 / 2) ∧
  (∀ n, ∃ k, ∑ i in (finset.range (2 * n)).filter (λ i, i % 2 = 1), daYanSeq (i + 1) - 
            ∑ i in (finset.range (2 * n)).filter (λ i, i % 2 = 0), daYanSeq (i + 1) = n * (n + 1) ∧ n * (n + 1) = 2) := 
  by
  -- Proof steps here
  sorry

end daYanSeq_properties_l695_695495


namespace probability_neither_square_nor_cube_l695_695540

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695540


namespace solution_set_of_inequality_l695_695792

noncomputable theory
open_locale classical

-- Definition of the function f
def f (x : ℝ) : ℝ := if x > 0 then 1 - real.log x / real.log 2 else if x < 0 then -(1 - real.log (-x) / real.log 2) else 0

lemma odd_function_property (x : ℝ) : f (-x) = - f (x) := 
by
  -- Proof of odd function property
  sorry

theorem solution_set_of_inequality : {x : ℝ | f x < 0} = {x | -2 < x ∧ x < 0 ∨ x > 2} := 
by
  -- Formal proof goes here
  sorry

end solution_set_of_inequality_l695_695792


namespace brad_net_profit_l695_695230

noncomputable def lemonade_glasses_per_gallon := 16
noncomputable def cost_per_gallon := 3.50
noncomputable def gallons_made := 2
noncomputable def price_per_glass := 1.00
noncomputable def glasses_drunk := 5
noncomputable def glasses_left_unsold := 6

theorem brad_net_profit :
  let total_glasses := gallons_made * lemonade_glasses_per_gallon in
  let glasses_not_sold := glasses_drunk + glasses_left_unsold in
  let glasses_sold := total_glasses - glasses_not_sold in
  let total_cost := gallons_made * cost_per_gallon in
  let total_revenue := glasses_sold * price_per_glass in
  let net_profit := total_revenue - total_cost in
  net_profit = 14 :=
by
  sorry

end brad_net_profit_l695_695230


namespace sum_first_10_common_terms_eq_6990500_l695_695744

-- Define the arithmetic progression
def is_arithmetic_term (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression
def is_geometric_term (k : ℕ) : ℕ := 10 * 2^k

-- Predicate to check if a term is common in both progressions
def is_common_term (m : ℕ) : Prop :=
  ∃ n k, m = is_arithmetic_term n ∧ m = is_geometric_term k ∧ k % 2 = 1

-- Sum of the first 10 common terms
def sum_of_first_10_common_terms : ℕ :=
  let common_terms := [20, 80, 320, 1280, 5120, 20480, 81920, 327680, 1310720, 5242880] in
  common_terms.sum

-- Main theorem statement
theorem sum_first_10_common_terms_eq_6990500 :
  sum_of_first_10_common_terms = 6990500 :=
by
  sorry

end sum_first_10_common_terms_eq_6990500_l695_695744


namespace train_speed_is_108_kmh_l695_695676

noncomputable def train_speed_kmh (distance_m : ℕ) (time_s : ℕ) :=
  (distance_m / time_s) * 3.6

theorem train_speed_is_108_kmh :
  train_speed_kmh 480 16 = 108 :=
by
  sorry

end train_speed_is_108_kmh_l695_695676


namespace probability_not_square_or_cube_l695_695551

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695551


namespace probability_neither_square_nor_cube_l695_695560

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695560


namespace paul_work_days_l695_695077

theorem paul_work_days (P : ℕ) (h : 1 / P + 1 / 120 = 1 / 48) : P = 80 := 
by 
  sorry

end paul_work_days_l695_695077


namespace find_a_and_period_and_decreasing_intervals_l695_695802

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4 * cos x * sin (x + 7 * Real.pi / 6) + a

theorem find_a_and_period_and_decreasing_intervals :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ 2) ∧ -- Condition to ensure the function's maximum value is 2
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) 1 = f x 1) ∧ -- Smallest positive period when a = 1
  (∀ k : ℤ, ∃ I : set ℝ, I = set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi) ∧
    ∀ x : ℝ, x ∈ I → f x 1 ≥ f (x + 0.1) 1): -- Intervals where f(x) is strictly decreasing when a = 1
  sorry

end find_a_and_period_and_decreasing_intervals_l695_695802


namespace partitions_count_l695_695274

open Finset

noncomputable def numberOfPartitions (n : ℕ) : ℕ :=
  (choose (n+2) 2) - (numberOfInvalidPartitions n)

def validPartition (A1 A2 A3 : Finset ℕ) : Prop :=
  (∀ (a ∈ A1) (b ∈ A1), a < b → ((a % 2) ≠ (b % 2))) ∧
  (∀ (a ∈ A2) (b ∈ A2), a < b → ((a % 2) ≠ (b % 2))) ∧
  (∀ (a ∈ A3) (b ∈ A3), a < b → ((a % 2) ≠ (b % 2))) ∧
  (A1.nonempty → A2.nonempty → A3.nonempty → (∃! (m ∈ {A1, A2, A3}), ∃ x ∈ m, x % 2 = 0))

def numberOfInvalidPartitions (n : ℕ) : ℕ :=
  sorry -- Define specific counting of invalid partitions as explained

theorem partitions_count (n : ℕ) :
  (∃ A1 A2 A3 : Finset ℕ, (A1 ∪ A2 ∪ A3 = range n) ∧ disjoint A1 A2 ∧ disjoint A2 A3 ∧ disjoint A1 A3 ∧ validPartition A1 A2 A3) →
  numberOfPartitions n = choose (n+2) 2 - numberOfInvalidPartitions n :=
sorry -- Proof construction

end partitions_count_l695_695274


namespace max_f_value_l695_695272

def f (x : ℝ) : ℝ := real.sqrt (x + 16) + real.sqrt (25 - x) + 2 * real.sqrt x

theorem max_f_value : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 25 → f x ≤ 11 + 4 * real.sqrt 2 :=
by {
  intros x hx,
  sorry
}

end max_f_value_l695_695272


namespace rational_triplets_solution_l695_695256

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def solutions_set (s : set (ℚ × ℚ × ℚ)) :=
  s = { (1,1,1), (1,2,2), (2,3,6), (2,4,4), (3,3,3) }

theorem rational_triplets_solution :
  ∀ (x y z : ℚ), 0 < x → 0 < y → 0 < z →
  is_integer (x + y + z) →
  is_integer (1/x + 1/y + 1/z) →
  is_integer (x * y * z) →
  x ≤ y → y ≤ z →
  solutions_set {(x, y, z)} :=
begin
  intros x y z hx hy hz hxyz1 hxyz2 hxyz3 hxy hyz,
  sorry -- the proof goes here.
end

end rational_triplets_solution_l695_695256


namespace explicit_formula_l695_695342

def sequence_a (n : ℕ) : ℝ :=
match n with
| 0 => 1
| (n+1) => 1 + sequence_a n + real.sqrt (1 + 4 * sequence_a n)

theorem explicit_formula (n : ℕ) (hn : n > 0) : 
  sequence_a n = 1 + (n - 1) * (n + real.sqrt 5 - 1) :=
sorry

end explicit_formula_l695_695342


namespace sin_2alpha_minus_half_pi_eq_neg_half_l695_695021

theorem sin_2alpha_minus_half_pi_eq_neg_half : 
  let α := real.atan2 (-1) (real.sqrt 3) in
  sin (2 * α - real.pi / 2) = -1 / 2 :=
by 
  let α := real.atan2 (-1) (real.sqrt 3)
  sorry

end sin_2alpha_minus_half_pi_eq_neg_half_l695_695021


namespace infinite_natural_solutions_l695_695080

theorem infinite_natural_solutions : ∀ n : ℕ, ∃ x y z : ℕ, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
by
  sorry

end infinite_natural_solutions_l695_695080


namespace cost_of_one_jacket_l695_695120

theorem cost_of_one_jacket
  (S J : ℝ)
  (h1 : 10 * S + 20 * J = 800)
  (h2 : 5 * S + 15 * J = 550) : J = 30 :=
sorry

end cost_of_one_jacket_l695_695120


namespace total_votes_l695_695388

theorem total_votes (votes_brenda : ℕ) (total_votes : ℕ) 
  (h1 : votes_brenda = 50) 
  (h2 : votes_brenda = (1/4 : ℚ) * total_votes) : 
  total_votes = 200 :=
by 
  sorry

end total_votes_l695_695388


namespace determine_x_l695_695154

variable (a b c d x : ℝ)
variable (h1 : (a^2 + x)/(b^2 + x) = c/d)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : d ≠ c) -- added condition from solution step

theorem determine_x : x = (a^2 * d - b^2 * c) / (c - d) := by
  sorry

end determine_x_l695_695154


namespace angle_MDN_45_l695_695125

/-- Given a square ABCD, with M on diagonal AC and N on side BC,
and satisfying MN = MD, we aim to prove that angle MDN is 45 degrees. -/

theorem angle_MDN_45 {a : ℝ} {x y : ℝ} (hM : ∀ M, M ∈ diag_AC (A (0,0)) (C (a,a)) → M = (x, x))
  (hN : ∀ N, N ∈ side_BC (B (a, 0)) (C (a, a)) → N = (a, y))
  (h_condition : dist (M (x, x)) (N (a, y)) = dist (M (x, x)) (D (a, 0))) :
  angle M D N = 45 := 
sorry

end angle_MDN_45_l695_695125


namespace percent_of_dollar_in_pocket_l695_695145

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

theorem percent_of_dollar_in_pocket :
  let total_cents := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  total_cents = 91 := by
  sorry

end percent_of_dollar_in_pocket_l695_695145


namespace train_speed_l695_695677

theorem train_speed (time_to_cross : ℚ) (length_of_train : ℚ) : 
  time_to_cross = 7 → length_of_train = 105 → (length_of_train / time_to_cross) * 3.6 = 54 := 
by
  intros h_time h_length
  have h1 : length_of_train / time_to_cross = 15,
  { rw [h_time, h_length],
    norm_num },
  rw h1,
  norm_num,
  sorry

end train_speed_l695_695677


namespace inequality_solution_l695_695970

theorem inequality_solution (x : ℝ) :
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3 / 2) :=
by
  sorry

end inequality_solution_l695_695970


namespace tetrahedron_area_sum_squares_l695_695455

theorem tetrahedron_area_sum_squares (A B C D : Type*) 
  (tetrahedron : Tetrahedron A B C D) 
  (h_AB_CD_90 : tetrahedron.dihedral_angle A B = 90 ∧ tetrahedron.dihedral_angle C D = 90) :
  tetrahedron.area (A, B, C)^2 + tetrahedron.area (A, B, D)^2 = 
  tetrahedron.area (B, C, D)^2 + tetrahedron.area (A, C, D)^2 :=
by 
  sorry

end tetrahedron_area_sum_squares_l695_695455


namespace sum_g_20_eq_40_l695_695316

theorem sum_g_20_eq_40 
  (f g : ℝ → ℝ)
  (hf_dom : ∀ x, f x ∈ ℝ)
  (hg_dom : ∀ x, g x ∈ ℝ)
  (hf_odd : ∀ x, f (-x + 1) = -f (x + 1))
  (h1 : ∀ x, f (1 - x) + g x = 2)
  (h2 : ∀ x, f x + g (x - 3) = 2) : 
  (∑ k in finset.range 20, g (k + 1)) = 40 := 
sorry

end sum_g_20_eq_40_l695_695316


namespace ms_marvel_jump_l695_695103

theorem ms_marvel_jump :
  ∃ n, (2 * 3^(n - 1) > 2000) ∧ (∀ m, m < n → 2 * 3^(m - 1) ≤ 2000) :=
begin
  sorry
end

end ms_marvel_jump_l695_695103


namespace maximize_sum_of_arithmetic_seq_l695_695218

theorem maximize_sum_of_arithmetic_seq (a d : ℤ) (n : ℤ) : d < 0 → a^2 = (a + 10 * d)^2 → n = 5 ∨ n = 6 :=
by
  intro h_d_neg h_a1_eq_a11
  have h_a1_5d_neg : a + 5 * d = 0 := sorry
  have h_sum_max : n = 5 ∨ n = 6 := sorry
  exact h_sum_max

end maximize_sum_of_arithmetic_seq_l695_695218


namespace george_room_painting_l695_695016

-- Define the number of ways to choose 2 colors out of 9 without considering the restriction
def num_ways_total : ℕ := Nat.choose 9 2

-- Define the restriction that red and pink should not be combined
def num_restricted_ways : ℕ := 1

-- Define the final number of permissible combinations
def num_permissible_combinations : ℕ := num_ways_total - num_restricted_ways

theorem george_room_painting :
  num_permissible_combinations = 35 :=
by
  sorry

end george_room_painting_l695_695016


namespace original_number_is_45_l695_695362

theorem original_number_is_45 (x : ℕ) (h : x - 30 = x / 3) : x = 45 :=
by {
  sorry
}

end original_number_is_45_l695_695362


namespace range_of_t_l695_695329

noncomputable def f (x : ℝ) : ℝ := abs (x * real.exp x)

theorem range_of_t : 
  {t : ℝ | ∃ x₁ x₂ x₃ x₄ : ℝ, f(x₁)^2 + t * f(x₁) + 1 = 0 ∧ 
                             f(x₂)^2 + t * f(x₂) + 1 = 0 ∧ 
                             f(x₃)^2 + t * f(x₃) + 1 = 0 ∧ 
                             f(x₄)^2 + t * f(x₄) + 1 = 0 ∧ 
                             x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧
                             x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ 
                             x₃ ≠ x₄ } = 
  {t : ℝ | t < -((real.exp 2) + 1) / real.exp 1 } :=
sorry

end range_of_t_l695_695329


namespace probability_neither_perfect_square_nor_cube_l695_695583

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695583


namespace width_of_room_is_3_75_l695_695119

-- Define the given constants
def length_of_room : ℝ := 5.5
def cost_per_sq_meter : ℝ := 1400
def total_cost : ℝ := 28875

-- Define the area of the floor based on total cost and cost per square meter.
def area_of_floor : ℝ := total_cost / cost_per_sq_meter

-- State the theorem that the width of the room is 3.75 meters given the conditions.
theorem width_of_room_is_3_75 : total_cost / cost_per_sq_meter / length_of_room = 3.75 :=
by
  -- leave the proof for further elaboration
  sorry

end width_of_room_is_3_75_l695_695119


namespace manolo_rate_change_after_one_hour_l695_695895

variable (masks_in_first_hour : ℕ)
variable (masks_in_remaining_time : ℕ)
variable (total_masks : ℕ)

-- Define conditions as Lean definitions
def first_hour_rate := 1 / 4  -- masks per minute
def remaining_time_rate := 1 / 6  -- masks per minute
def total_time := 4  -- hours
def masks_produced_in_first_hour (t : ℕ) := t * 15  -- t hours, 60 minutes/hour, at 15 masks/hour
def masks_produced_in_remaining_time (t : ℕ) := t * 10 -- (total_time - 1) hours, 60 minutes/hour, at 10 masks/hour

-- Main proof problem statement
theorem manolo_rate_change_after_one_hour :
  masks_in_first_hour = masks_produced_in_first_hour 1 →
  masks_in_remaining_time = masks_produced_in_remaining_time (total_time - 1) →
  total_masks = masks_in_first_hour + masks_in_remaining_time →
  (∃ t : ℕ, t = 1) :=
by
  -- Placeholder, proof not required
  sorry

end manolo_rate_change_after_one_hour_l695_695895


namespace division_quotient_l695_695842

theorem division_quotient (D d R : ℤ) (hD : D = 689) (hd : d = 36) (hR : R = 5) :
  let Q := (D - R) / d in Q = 19 :=
by
  sorry

end division_quotient_l695_695842


namespace total_loss_150_l695_695675

-- Define selling prices
def selling_price_A : ℝ := 1800
def selling_price_B : ℝ := 1800

-- Define profit and loss percentages
def profit_percentage_A : ℝ := 0.2
def loss_percentage_B : ℝ := 0.2

-- Assume cost prices
def cost_price_A : ℝ := selling_price_A / (1 + profit_percentage_A)
def cost_price_B : ℝ := selling_price_B / (1 - loss_percentage_B)

-- Total cost and selling prices
def total_cost_price : ℝ := cost_price_A + cost_price_B
def total_selling_price : ℝ := selling_price_A + selling_price_B

-- Prove the total loss
theorem total_loss_150 (h1: total_cost_price = 3750) (h2: total_selling_price = 3600) :
      total_cost_price - total_selling_price = 150 := by
  sorry

end total_loss_150_l695_695675


namespace correct_time_on_bus_l695_695464

def leave_time := (7 : ℕ, 0 : ℕ) -- 7:00 a.m.
def bus_time := (7 : ℕ, 45 : ℕ) -- 7:45 a.m.
def total_classes := 8
def class_duration := 55 -- in minutes
def other_activities := (1 : ℕ, 45 : ℕ) -- 1 hour and 45 minutes
def arrive_time := (17 : ℕ, 15 : ℕ) -- 5:15 p.m.

def total_time_away : ℕ :=
  let start := bus_time.1 * 60 + bus_time.2 in -- convert bus_time to minutes
  let end := arrive_time.1 * 60 + arrive_time.2 in -- convert arrive_time to minutes
  end - start

def total_class_time : ℕ := total_classes * class_duration

def total_other_activities : ℕ := other_activities.1 * 60 + other_activities.2

def total_school_time : ℕ := total_class_time + total_other_activities

def time_on_bus : ℕ := total_time_away - total_school_time

theorem correct_time_on_bus : time_on_bus = 25 := by
  sorry

end correct_time_on_bus_l695_695464


namespace infinite_lively_subset_l695_695146

noncomputable def is_lively (n : ℕ) : Prop :=
  ∃ p : ℕ, prime p ∧ p > 10^(10^100) ∧ p ∣ n

theorem infinite_lively_subset
  (S : Set ℕ) (hS1 : ∀ s ∈ S, is_lively s)
  (hS2 : Set.Infinite S) :
  ∃ T ⊆ S, Set.Infinite T ∧ ∀ (F : Finset ℕ), (∀ t ∈ F, t ∈ T) → is_lively (F.sum id) :=
sorry

end infinite_lively_subset_l695_695146


namespace amount_alloy_B_mixed_l695_695990

/-- Define the ratio properties and the amount of alloys A and B --/
def ratio_lead_tin_A : ℝ := 3 / 4 
def ratio_tin_copper_B : ℝ := 2 / 5 
def weight_alloy_A : ℝ := 90 
def weight_tin_new_alloy : ℝ := 91.42857142857143

/-- Define the function to calculate the weight of tin in alloy A --/
def weight_tin_in_A (weight_A : ℝ) (ratio_A : ℝ) : ℝ := (ratio_A / (1 + ratio_A)) * weight_A

/-- Define the function to calculate the weight of tin in alloy B --/
def weight_tin_in_B (weight_B : ℝ) (ratio_B : ℝ) : ℝ := (ratio_B / (1 + ratio_B)) * weight_B

/-- The final Lean statement to validate the amount of alloy B added --/
theorem amount_alloy_B_mixed (x : ℝ) : 
  weight_tin_in_A weight_alloy_A ratio_lead_tin_A + weight_tin_in_B x ratio_tin_copper_B = weight_tin_new_alloy → 
  x = 140 := 
sorry

end amount_alloy_B_mixed_l695_695990


namespace measure_of_angle_B_and_area_of_triangle_l695_695026

theorem measure_of_angle_B_and_area_of_triangle 
    (a b c : ℝ) 
    (A B C : ℝ) 
    (condition : 2 * c = a + (Real.cos A * (b / (Real.cos B))))
    (sum_sides : a + c = 3 * Real.sqrt 2)
    (side_b : b = 4)
    (angle_B : B = Real.pi / 3) :
    B = Real.pi / 3 ∧ 
    (1/2 * a * c * (Real.sin B) = Real.sqrt 3 / 6) :=
by
    sorry

end measure_of_angle_B_and_area_of_triangle_l695_695026


namespace value_of_a_l695_695698

def g (x : ℝ) : ℝ := 5 * x - 7

theorem value_of_a (a : ℝ) : g(a) = 0 ↔ a = 7 / 5 := by
  sorry

end value_of_a_l695_695698


namespace remaining_stock_weight_l695_695064

def green_beans_weight : ℕ := 80
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 20
def flour_weight : ℕ := 2 * sugar_weight
def lentils_weight : ℕ := flour_weight - 10

def rice_remaining_weight : ℕ := rice_weight - rice_weight / 3
def sugar_remaining_weight : ℕ := sugar_weight - sugar_weight / 5
def flour_remaining_weight : ℕ := flour_weight - flour_weight / 4
def lentils_remaining_weight : ℕ := lentils_weight - lentils_weight / 6

def total_remaining_weight : ℕ :=
  rice_remaining_weight + sugar_remaining_weight + flour_remaining_weight + lentils_remaining_weight + green_beans_weight

theorem remaining_stock_weight :
  total_remaining_weight = 343 := by
  sorry

end remaining_stock_weight_l695_695064


namespace quadrilateral_is_trapezoid_l695_695908

noncomputable def divides_segment_in_equal_ratio (A B : Point) (M : Point) (λ : Real) : Prop :=
  ∃ t : Real, 0 < t ∧ t < 1 ∧
    (M = Point_on_segment A B t) ∧
    t = (λ / (1 - λ))

noncomputable def equal_area_partition (A B C D M N : Point) : Prop :=
  area (quadrilateral A M N D) = area (quadrilateral M B C N)

theorem quadrilateral_is_trapezoid
  (A B C D M N : Point)
  (λ : Real)
  (hM : divides_segment_in_equal_ratio A B M λ)
  (hN : divides_segment_in_equal_ratio D C N λ)
  (hMN : equal_area_partition A B C D M N) :
  parallel AB DC :=
sorry

end quadrilateral_is_trapezoid_l695_695908


namespace exist_perpendicular_line_in_plane_l695_695295

variables {Point : Type*} [EuclideanSpace Point] 

-- Definitions of line and plane to be used
def Line (P : Type*) [EuclideanSpace P] := set P 
def Plane (P : Type*) [EuclideanSpace P] := set P

variables (l : Line Point) (α : Plane Point)

-- Statement of the problem
theorem exist_perpendicular_line_in_plane : ∃ l' : Line Point, l' ⊆ α ∧ ⊥ (l ∩ l') :=
sorry

end exist_perpendicular_line_in_plane_l695_695295


namespace log_eq_solution_l695_695094

theorem log_eq_solution (x : ℝ) (hx : log 8 x - 3 * log 2 x = 6) :
  x = 1 / Real.sqrt (Real.sqrt 512) :=
sorry

end log_eq_solution_l695_695094


namespace part1_part2_l695_695311

theorem part1 (x₁ x₂ : ℝ)
  (h₁ : x₁ * x₁ - 5 * x₁ - 3 = 0)
  (h₂ : x₂ * x₂ - 5 * x₂ - 3 = 0) :
  x₁^2 + x₂^2 = 31 :=
sorry

theorem part2 (x₁ x₂ : ℝ)
  (h₁ : x₁ * x₁ - 5 * x₁ - 3 = 0)
  (h₂ : x₂ * x₂ - 5 * x₂ - 3 = 0) :
  (1 / x₁) - (1 / x₂) = (sqrt 37 / -3) ∨ (1 / x₁) - (1 / x₂) = (- sqrt 37 / -3) :=
sorry

end part1_part2_l695_695311


namespace find_a_for_chord_length_l695_695322

theorem find_a_for_chord_length (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0 ∧ x + y + 2 = 0) →
  (chord_length : ℝ) (chord_length = 4) →
  a = -4 := 
by 
  sorry

end find_a_for_chord_length_l695_695322


namespace at_most_three_differences_l695_695672

noncomputable def fractional_part (x : ℝ) := x - x.floor

theorem at_most_three_differences (a : ℝ) (h : 0 < a ∧ a < 1) (seq : ℕ → ℕ)
  (h_seq_ordered : ∀ n, seq n < seq (n + 1))
  (h_seq_condition : ∀ n, fractional_part (seq n * a) < 1 / 10) :
  ∃ (s : Finset ℕ), s.card ≤ 3 ∧ (∀ n, (seq (n + 1) - seq n) ∈ s) :=
sorry

end at_most_three_differences_l695_695672


namespace cyclic_quad_equal_sides_l695_695167

open Set

variable (A B C D P E : Point) (circle₁ : Circle) (circle₂ : Circle)

-- Circles passing through the given points
variables (circ₁ : circle_contains circle₁ A ∧ circle_contains circle₁ B ∧ circle_contains circle₁ P)
variables (circ₂ : circle_contains circle₂ A ∧ circle_contains circle₂ B ∧ circle_contains circle₂ E ∧ circle_contains circle₂ C)

-- Quadrilateral \(ABCD\) is cyclic
variable (cyclic_quad : CyclicQuadrilateral A B C D)

-- Conditions
variable (AB_eq_AD : distance A B = distance A D)
variable (circle_meets_BC_at_E : line_contains_segment (line_through B C) E)
variable (diagonal_intersect_in_P : diagonals_intersect_in A C B D P)

theorem cyclic_quad_equal_sides :
  AB_eq_AD → CD = CE :=
by
  intro h
  sorry

end cyclic_quad_equal_sides_l695_695167


namespace hyperbola_eccentricity_correct_l695_695304

noncomputable def hyperbola_eccentricity (a b : ℝ) (P F1 F2: ℝ × ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: (P.1 / a) ^ 2 - (P.2 / b) ^ 2 = 1)
  (h4: F1.1 = - sqrt (a^2 + b^2))
  (h5: F2.1 = sqrt (a^2 + b^2))
  (h6: P.1 = 4 / 3 * a)
  (h7: (F1.1 - P.1) * (F2.1 - P.1) + (F1.2 - P.2) * (F2.2 - P.2) = 0)
  : ℝ :=
sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct :
  ∀ (a b: ℝ), ∀ (P F1 F2 : ℝ × ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: (P.1 / a) ^ 2 - (P.2 / b) ^ 2 = 1)
  (h4: F1.1 = - sqrt (a^2 + b^2))
  (h5: F2.1 = sqrt (a^2 + b^2))
  (h6: P.1 = 4 / 3 * a)
  (h7: (F1.1 - P.1) * (F2.1 - P.1) + (F1.2 - P.2) * (F2.2 - P.2) = 0),
  hyperbola_eccentricity a b P F1 F2 h1 h2 h3 h4 h5 h6 h7 = 3 * sqrt 2 / 2 :=
sorry

end hyperbola_eccentricity_correct_l695_695304


namespace number_of_valid_pairs_l695_695810

open Finset

-- Define set A
def A := {x | 1 ≤ x ∧ x ≤ 9}

-- Define set B as the Cartesian product of A with itself
def B := {ab : ℤ × ℤ | ab.1 ∈ A ∧ ab.2 ∈ A}

-- Define the function f
def f (ab : ℤ × ℤ) : ℤ := ab.1 * ab.2 - ab.1 - ab.2

-- The main theorem to be proved
theorem number_of_valid_pairs : (filter (λ ab, f ab = 11) B).card = 4 :=
by sorry

end number_of_valid_pairs_l695_695810


namespace quadratic_decreasing_l695_695297

-- Define the quadratic function and the condition a < 0
def quadratic_function (a x : ℝ) := a * x^2 - 2 * a * x + 1

-- Define the main theorem to be proven
theorem quadratic_decreasing (a m : ℝ) (ha : a < 0) : 
  (∀ x, x > m → quadratic_function a x < quadratic_function a (x+1)) ↔ m ≥ 1 :=
by
  sorry

end quadratic_decreasing_l695_695297


namespace max_value_in_interval_l695_695309

variable {R : Type*} [OrderedCommRing R]

variables (f : R → R)
variables (odd_f : ∀ x, f (-x) = -f (x))
variables (f_increasing : ∀ x y, 0 < x → x < y → f x < f y)
variables (additive_f : ∀ x y, f (x + y) = f x + f y)
variables (f1_eq_2 : f 1 = 2)

theorem max_value_in_interval : ∀ x ∈ Set.Icc (-3 : R) (-2 : R), f x ≤ f (-2) ∧ f (-2) = -4 :=
by
  sorry

end max_value_in_interval_l695_695309


namespace invitations_meet_everyone_each_lady_attendance_l695_695096

open Finset

-- Definitions and assumptions related to the problem conditions
variable (ladies gentlemen : Finset ℕ) (num_invitations : ℕ)
variable (nLadies nGentlemen nTotal : ℕ := 20)

-- Proving that under the given conditions, 11 invitations are necessary and sufficient
theorem invitations_meet_everyone (ladies_card : ladies.card = 9)
  (gentlemen_card : gentlemen.card = 11)
  (total_people_card : nTotal = ladies.card + gentlemen.card)
  (each_invitation_ladies : ∀ invitation, (invitation.card = 3) → (invitation ⊂ ladies))
  (each_invitation_gentlemen : ∀ invitation, (invitation.card = 2) → (invitation ⊂ gentlemen))
  (every_meets_each_other : ∀ p q ∈ (ladies ∪ gentlemen), ∃ invitation, (p ∈ invitation) ∧ (q ∈ invitation)) :
  num_invitations = 11 :=
  sorry

-- Proving that each lady must attend exactly 7 times
theorem each_lady_attendance (n_attends : ℕ) (each_lady_count : n_attends = 7)
  (ladies_attendance : ∀ lady ∈ ladies, attendance_count lady = n_attends) :
  n_attends = 7 :=
  sorry

end invitations_meet_everyone_each_lady_attendance_l695_695096


namespace final_price_percentage_l695_695219

-- Define the original price
def original_price : ℝ := 350.00

-- Define the first discount rate as 10%
def first_discount_rate : ℝ := 0.10

-- Define the second discount rate as 12%
def second_discount_rate : ℝ := 0.12

-- Formula to calculate the price after the first discount
def price_after_first_discount (price : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate)

-- Formula to calculate the price after the second discount
def price_after_second_discount (price : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate)

-- Proof statement
theorem final_price_percentage :
  let final_price := price_after_second_discount 
                      (price_after_first_discount original_price first_discount_rate)
                      second_discount_rate 
  in (final_price / original_price) * 100 = 79.2 := by
  sorry

end final_price_percentage_l695_695219


namespace real_root_of_sqrt_eq_l695_695276

theorem real_root_of_sqrt_eq : ∃ x : ℝ, sqrt (x - 2) + sqrt (x + 4) = 12 ∧ x = 35.0625 :=
by
  sorry

end real_root_of_sqrt_eq_l695_695276


namespace bread_products_wasted_l695_695411

theorem bread_products_wasted :
  (50 * 8 - (20 * 5 + 15 * 4 + 10 * 10 * 1.5)) / 1.5 = 60 := by
  -- The proof steps are omitted here
  sorry

end bread_products_wasted_l695_695411


namespace max_f_eq_3_plus_sqrt2_l695_695801

def f (x : ℝ) : ℝ := sin x + cos x + 2 * sin x * cos x + 2

theorem max_f_eq_3_plus_sqrt2 : ∃ x : ℝ, f x = 3 + real.sqrt 2 := 
sorry

end max_f_eq_3_plus_sqrt2_l695_695801


namespace no_infinitely_many_m_l695_695754

def f (n : ℕ) : ℕ :=
  (∑ s in (powerset (finset.range n) | ∑ x in s, x = n, 1) : ℕ)

theorem no_infinitely_many_m (f : ℕ → ℕ) :
  ¬ ∃∞ (m : ℕ), f m = f (m + 1) := by
  sorry

end no_infinitely_many_m_l695_695754


namespace simplify_root_power_l695_695480

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695480


namespace gaeun_taller_than_nana_l695_695072

def nana_height_m : ℝ := 1.618
def gaeun_height_cm : ℝ := 162.3
def nana_height_cm : ℝ := nana_height_m * 100

theorem gaeun_taller_than_nana : gaeun_height_cm - nana_height_cm = 0.5 := by
  sorry

end gaeun_taller_than_nana_l695_695072


namespace total_cost_l695_695415

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l695_695415


namespace M_eq_l695_695459

def M : Set Nat := { m | 0 < m ∧ ∃ k : Z, 10 = k * (m + 1) }

theorem M_eq : M = {1, 4, 9} :=
by
  sorry

end M_eq_l695_695459


namespace not_all_pairs_product_minus_one_are_perfect_squares_l695_695244

-- Definitions
def is_not_in_set (d : ℕ) : Prop :=
  d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13

def is_not_perfect_square (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ≠ n

-- Theorem Statement
theorem not_all_pairs_product_minus_one_are_perfect_squares {d : ℕ} 
  (h : is_not_in_set d) :
  ∃ (a b : ℕ), a ≠ b ∧ (a ∈ ({2, 5, 13, d} : set ℕ)) ∧ (b ∈ ({2, 5, 13, d} : set ℕ)) ∧ is_not_perfect_square (a * b - 1) :=
sorry

end not_all_pairs_product_minus_one_are_perfect_squares_l695_695244


namespace problem_solution_l695_695690

theorem problem_solution :
  50000 - ((37500 / 62.35) ^ 2 + Real.sqrt 324) = -311752.222 :=
by
  sorry

end problem_solution_l695_695690


namespace sum_of_possible_b_l695_695898

/-
  Problem Statement:
  Prove that the sum of all possible distinct integers \( b \) such that \( x^2 + bx + 24 = 0 \)
  has two distinct negative integer solutions is equal to 60.
-/

theorem sum_of_possible_b : 
  (∑ b in {b : ℤ | ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ r ≠ s ∧ r * s = 24 ∧ b = r + s}.to_finset, b) = 60 :=
sorry

end sum_of_possible_b_l695_695898


namespace comic_books_exclusive_count_l695_695223

theorem comic_books_exclusive_count 
  (shared_comics : ℕ) 
  (total_andrew_comics : ℕ) 
  (john_exclusive_comics : ℕ) 
  (h_shared_comics : shared_comics = 15) 
  (h_total_andrew_comics : total_andrew_comics = 22) 
  (h_john_exclusive_comics : john_exclusive_comics = 10) : 
  (total_andrew_comics - shared_comics + john_exclusive_comics = 17) := by 
  sorry

end comic_books_exclusive_count_l695_695223


namespace range_of_f_l695_695780

-- Define the floor function for real numbers
def floor (x : ℝ) : ℝ := Real.floor x

-- Define the function f(x, y) given the problem's conditions
def f (x y : ℝ) : ℝ :=
  (x + y) / (floor x * floor y + floor x + floor y + 1)

-- The main theorem to prove the range of the function
theorem range_of_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) :
  f(x, y) ∈ {1/2} ∪ Icc (5/6 : ℝ) (5/4 : ℝ) :=
sorry

end range_of_f_l695_695780


namespace find_AD_l695_695379

variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable [inhabited A] [inhabited B] [inhabited C] [inhabited D]

def is_isosceles_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ (AB AC : ℝ), AB = 20 ∧ AC = 20 ∧ dist B C = 18

def is_midpoint (D B C : Type) [metric_space D] [metric_space B] [metric_space C] :=
  ∃ (BD DC : ℝ), BD = 9 ∧ DC = 9

theorem find_AD (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [is_isosceles_triangle A B C] [is_midpoint D B C] : 
  dist A D = real.sqrt 319 := by
  sorry

end find_AD_l695_695379


namespace sum_p_from_1_to_21_l695_695924

def p (x : ℕ) : ℤ :=
  sorry -- Definition of the cubic polynomial goes here, based on interpolation constraints.

theorem sum_p_from_1_to_21 :
  -- Given the conditions
  (p 3 = 1) ∧ 
  (p 8 = 20) ∧ 
  (p 12 = 10) ∧ 
  (p 18 = 28) →
  -- Prove the sum equals the calculated value
  (∑ x in Finset.range 22, p x) = 2046 * a + 1540 * b + 231 * c + 21 * d :=
sorry -- Proof goes here.

end sum_p_from_1_to_21_l695_695924


namespace sum_first_10_common_terms_l695_695748

def AP_term (n : ℕ) : ℕ := 5 + 3 * n
def GP_term (k : ℕ) : ℕ := 10 * 2^k
def common_terms (m : ℕ) : ℕ := 20 * 4^m

theorem sum_first_10_common_terms :
  ∑ m in Finset.range 10, common_terms m = 6990500 :=
by sorry

end sum_first_10_common_terms_l695_695748


namespace value_of_a_2019_l695_695130

-- Defining the sequence
def sequence (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 2
  else 
    let rec a : ℕ → ℚ
    | 1 => 2
    | (n+1) => 1 - (1 / a n)
    a n

theorem value_of_a_2019 : sequence 2019 = -1 := 
by {
  sorry
}

end value_of_a_2019_l695_695130


namespace summation_of_g_l695_695767

noncomputable def g (x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 + 3 * x - (5 / 12)

theorem summation_of_g :
  (∑ k in (finset.range 2017).map (λ k : ℕ, k + 1) , g (k / 2018)) = 2017 := by
  sorry

end summation_of_g_l695_695767


namespace product_of_numbers_l695_695164

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43.05 := by
  sorry

end product_of_numbers_l695_695164


namespace find_smallest_angle_l695_695734

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l695_695734


namespace quadratic_has_real_roots_iff_l695_695833

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l695_695833


namespace simplify_expression_l695_695883

variables {p q r x y z : ℝ}
variables (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)

def x_def : ℝ := q / r + r / q
def y_def : ℝ := p / r + r / p
def z_def : ℝ := p / q + q / p

theorem simplify_expression (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  (x_def q r) ^ 2 + (y_def p r) ^ 2 + (z_def p q) ^ 2 - 2 * (x_def q r) * (y_def p r) * (z_def p q) = 0 :=
sorry

end simplify_expression_l695_695883


namespace probability_neither_square_nor_cube_l695_695542

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695542


namespace is_quadratic_l695_695967

variable (x y : ℝ)

def equationA : Prop := x + 2 * y = 1
def equationB : Prop := x^2 + x - 1 = x^2
def equationC : Prop := x^2 + 3 / x = 8
def equationD : Prop := x^2 - 5 * x = 0

theorem is_quadratic : equationD → ∀ a b c : ℝ, a = 1 → b = -5 → c = 0 → a * x^2 + b * x + c = 0 :=
by
  intro hD a b c ha hb hc
  rw [ha, hb, hc]
  exact hD

end is_quadratic_l695_695967


namespace variance_of_temperatures_l695_695117

def temperatures : List ℕ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_temperatures : variance temperatures = 22 / 3 := 
by
  sorry

end variance_of_temperatures_l695_695117


namespace triangle_problem_l695_695840

noncomputable def triangle_area (a b C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

theorem triangle_problem 
  (a b c : ℝ)
  (C : ℝ)
  (h1 : Real.tan (π / 4 - C) = √3 - 2)
  (h2 : c = √7)
  (h3 : a + b = 5)
  (hC : C = π / 3) :
  triangle_area a b C = (3 * √3) / 2 :=
by 
  sorry

end triangle_problem_l695_695840


namespace compute_square_of_BD_l695_695039

-- Definitions for the problem
variables (A B C D P Q R S : Type) [AffineSpace A B] [AffineSpace A D] [AffineSpace P Q] [AffineSpace R S] 
variables (area_BCAC : ℝ) (PQ_r RS_s : ℝ)
variables (BD : ℝ) (d : ℝ)

-- Given conditions
def parallelogram : Prop := Parallelogram area_BCAC

def projections : Prop := PQ_r = 4 ∧ RS_s = 10

-- Square of the length of BD
def square_diagonal_length_BD : Prop := (d*2)*(d*2) = 196

-- The mathematically equivalent proof problem
theorem compute_square_of_BD 
  (h1 : parallelogram)
  (h2 : projections)
  : square_diagonal_length_BD :=
sorry

end compute_square_of_BD_l695_695039


namespace smallest_number_of_rectangles_l695_695979

theorem smallest_number_of_rectangles (board_size : ℕ) (board_size_eq : board_size = 2008) :
  ∃ (M : ℕ), (∀ (rectangles : ℕ → Prop),
  (∀ (i j : ℕ), (i < board_size ∧ j < board_size) → ∃ r, rectangles r ∧ (i, j) ∈ r) →
  ∀ (r : ℕ), rectangles r → (sides_of_cells_covered r)) ∧ M = 2009 := sorry

end smallest_number_of_rectangles_l695_695979


namespace infinite_positive_sequence_geometric_l695_695866

theorem infinite_positive_sequence_geometric {a : ℕ → ℝ} (h : ∀ n ≥ 1, a (n + 2) = a n - a (n + 1)) 
  (h_pos : ∀ n, a n > 0) :
  ∃ (a1 : ℝ) (q : ℝ), q = (Real.sqrt 5 - 1) / 2 ∧ (∀ n, a n = a1 * q^(n - 1)) := by
  sorry

end infinite_positive_sequence_geometric_l695_695866


namespace probability_three_heads_l695_695630

noncomputable def fair_coin_flip: ℝ := 1 / 2

theorem probability_three_heads :
  (fair_coin_flip * fair_coin_flip * fair_coin_flip) = 1 / 8 :=
by
  -- proof would go here
  sorry

end probability_three_heads_l695_695630


namespace find_sum_l695_695202

noncomputable def principal_sum (P R : ℝ) := 
  let I := (P * R * 10) / 100
  let new_I := (P * (R + 5) * 10) / 100
  I + 600 = new_I

theorem find_sum (P R : ℝ) (h : principal_sum P R) : P = 1200 := 
  sorry

end find_sum_l695_695202


namespace problem_proof_l695_695346

variables {A B C A₀ B₀ C₀ O : Type} [metric_space O]
variables (r R : ℝ)
variables [has_dist O O] -- for distance calculation

def inradius (t : triangle O) : ℝ := r
def circumradius (t : triangle O) : ℝ := R

def A₀_in_t : Prop := angle_bisector_point A B C A₀ 
def B₀_in_t : Prop := angle_bisector_point B A C B₀
def C₀_in_t : Prop := angle_bisector_point C A B C₀

def OA₀ := dist O A₀
def OB₀ := dist O B₀
def OC₀ := dist O C₀

noncomputable def proof : Prop :=
  ∀ (t : triangle O),
    inradius t = r →
    circumradius t = R →
    A₀_in_t →
    B₀_in_t →
    C₀_in_t →
    OA₀ * OB₀ * OC₀ ≤ r^2 * R / 2

theorem problem_proof (t : triangle O) :
  proof r R := sorry

end problem_proof_l695_695346


namespace correct_mean_l695_695121

theorem correct_mean (mean incorrect_values correct_values : list ℤ) (n : ℕ)
  (hmean : mean = 120) (hincorrect_values : incorrect_values = [-30, 320, 120, 60, -100])
  (hcorrect_values : correct_values = [-50, 350, 100, 25, -80]) (hn : n = 40) :
  (∑ x in incorrect_values, x) - (∑ x in correct_values, x) + n * mean = 4775 →
  4775 / n = 119.375 :=
begin
  sorry
end

end correct_mean_l695_695121


namespace rate_of_mixed_oil_is_correct_l695_695360

def rate_of_mixed_oil : ℝ :=
  let cost1 := 10 * 40 -- Total cost for the first oil
  let cost2 := 5 * 66  -- Total cost for the second oil
  let total_cost := cost1 + cost2 -- Total cost of the mixed oil
  let total_volume := 10 + 5  -- Total volume of the mixed oil
  total_cost / total_volume  -- Rate of the mixed oil per litre

theorem rate_of_mixed_oil_is_correct :
  rate_of_mixed_oil = 48.67 :=
by
  -- The proof is omitted
  sorry

end rate_of_mixed_oil_is_correct_l695_695360


namespace math_problem_l695_695132

variable (a b c : ℝ)

theorem math_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  sqrt (1 / a - 1) * sqrt (1 / b - 1) + sqrt (1 / b - 1) * sqrt (1 / c - 1) + 
  sqrt (1 / c - 1) * sqrt (1 / a - 1) ≥ 6 := 
sorry

end math_problem_l695_695132


namespace category_B_count_solution_hiring_probability_l695_695190

-- Definitions and conditions
def category_A_count : Nat := 12

def total_selected_housekeepers : Nat := 20
def category_B_selected_housekeepers : Nat := 16
def category_A_selected_housekeepers := total_selected_housekeepers - category_B_selected_housekeepers

-- The value of x
def category_B_count (x : Nat) : Prop :=
  (category_A_selected_housekeepers * x) / category_A_count = category_B_selected_housekeepers

-- Assertion for the value of x
theorem category_B_count_solution : category_B_count 48 :=
by sorry

-- Conditions for the second part of the problem
def remaining_category_A : Nat := 3
def remaining_category_B : Nat := 2
def total_remaining := remaining_category_A + remaining_category_B

def possible_choices := remaining_category_A * (remaining_category_A - 1) / 2 + remaining_category_A * remaining_category_B + remaining_category_B * (remaining_category_B - 1) / 2
def successful_choices := remaining_category_A * remaining_category_B

def probability (a b : Nat) := (successful_choices % total_remaining) / (possible_choices % total_remaining)

-- Assertion for the probability
theorem hiring_probability : probability remaining_category_A remaining_category_B = 3 / 5 :=
by sorry

end category_B_count_solution_hiring_probability_l695_695190


namespace number_of_ordered_pairs_problem_solution_l695_695889

noncomputable def nonreal_root_of_z4 : ℂ := complex.I

theorem number_of_ordered_pairs (a b : ℤ) (ω : ℂ) (hω : ω ^ 4 = 1) (h_nr : ω ≠ 1 ∧ ω ≠ -1) :
  (|a * ω + b| = real.sqrt 2) ↔ (a^2 + b^2 = 2) :=
begin
  sorry
end

theorem problem_solution :
  (finset.card ((finset.univ : finset (ℤ × ℤ)).filter (λ p, 
    |p.1 * nonreal_root_of_z4 + p.2| = real.sqrt 2)) = 4) :=
begin
  sorry
end

end number_of_ordered_pairs_problem_solution_l695_695889


namespace num_valid_five_digit_numbers_l695_695349

-- Definitions for the problem
def is_even_digit (n : ℕ) : Prop := n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def is_valid_five_digit_number (n : ℕ) : Prop := 
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (∀ d ∈ (n.digits 10), is_even_digit d) ∧ 
  (n % 10 = 0) ∧ 
  ((n.digits 10).sum % 3 = 0)

-- Theorem stating the number of such valid numbers
theorem num_valid_five_digit_numbers : 
  {n : ℕ | is_valid_five_digit_number n}.card = 500 := 
sorry

end num_valid_five_digit_numbers_l695_695349


namespace rotation_angle_l695_695463

theorem rotation_angle (P Q R : Point) (y : ℝ) (h1 : rotate P Q 480 R) 
  (h2 : rotate P Q y R) (h3 : y < 360) (h4 : dist P R = 10) : y = 240 := 
by
  sorry

end rotation_angle_l695_695463


namespace range_of_x_l695_695856

theorem range_of_x (x : ℝ) :
  (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (x ≤ 1) ∧ (x ≠ 0) :=
by
  apply Iff.intro
  sorry

end range_of_x_l695_695856


namespace arc_length_problem_l695_695933

noncomputable def arc_length (r : ℝ) (theta : ℝ) : ℝ :=
  r * theta

theorem arc_length_problem :
  ∀ (r : ℝ) (theta_deg : ℝ), r = 1 ∧ theta_deg = 150 → 
  arc_length r (theta_deg * (Real.pi / 180)) = (5 * Real.pi / 6) :=
by
  intro r theta_deg h
  sorry

end arc_length_problem_l695_695933


namespace sqrt_mul_l695_695689

theorem sqrt_mul (h₁ : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3) : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 :=
by
  sorry

end sqrt_mul_l695_695689


namespace find_m_n_l695_695330

noncomputable def func (m n : ℝ) (x : ℝ) : ℝ :=
log 3 ((m * x^2 + 8 * x + n) / (x^2 + 1))

theorem find_m_n :
  (∃ (m n : ℝ), (∀ x : ℝ, 0 ≤ func m n x ∧ func m n x ≤ 2) ∧ m = 5 ∧ n = 5) :=
by
  sorry

end find_m_n_l695_695330


namespace simplify_sqrt3_7_pow6_l695_695469

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695469


namespace maximum_absolute_value_l695_695808

theorem maximum_absolute_value (x y : ℝ) 
  (h1 : x + y - 2 ≤ 0) 
  (h2 : x - y + 4 ≥ 0) 
  (h3 : y ≥ 0) : 
  ∃ z : ℝ, z = |x - 2 * y + 2| ∧ z ≤ 5 :=
begin
  sorry
end

end maximum_absolute_value_l695_695808


namespace trains_cross_time_l695_695166

theorem trains_cross_time (L : ℝ) (T1 T2 : ℝ) (T : ℝ) :
  (L = 120) →
  (T1 = 9) →
  (T2 = 15) →
  (T ≈ 11.25) :=
by
  sorry

end trains_cross_time_l695_695166


namespace valid_permutations_count_l695_695819

theorem valid_permutations_count (n : ℕ) (h : n > 0) : 
  (∃ P : list (list ℕ), ∀ l ∈ P, l.length = n ∧ 
    ∀ (i : ℕ), i ∈ (list.range (n - 1)) → 
    (abs (l.nth_le (i + 1) (by sorry) - l.nth_le i (by sorry)) = 1)) → 
  list.length P = 2^(n-1) :=
sorry

end valid_permutations_count_l695_695819


namespace x_is_neg_g_neg_y_l695_695428

def g (t : ℝ) : ℝ := t / (2 - t)

theorem x_is_neg_g_neg_y (x y : ℝ) (h : y = g x) (hx : x ≠ 2) : x = -g (-y) :=
by
  sorry

end x_is_neg_g_neg_y_l695_695428


namespace weight_ratio_l695_695065

-- Conditions
def initial_weight : ℕ := 99
def initial_loss : ℕ := 12
def weight_added_back (x : ℕ) : Prop := x = 81 + 30 - initial_weight
def times_lost : ℕ := 3 * initial_loss
def final_gain : ℕ := 6
def final_weight : ℕ := 81

-- Question
theorem weight_ratio (x : ℕ)
  (H1 : weight_added_back x)
  (H2 : initial_weight - initial_loss + x - times_lost + final_gain = final_weight) :
  x / initial_loss = 2 := by
  sorry

end weight_ratio_l695_695065


namespace intersection_A_B_l695_695307

def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def B : Set ℝ := { -1, 0, 1, 2, 3 } 

theorem intersection_A_B : A ∩ B = { 0, 1 } :=
by trivial -- placeholder for proper proof

end intersection_A_B_l695_695307


namespace incorrect_multiplier_l695_695664

theorem incorrect_multiplier (x : ℕ) : 
  let correct_result := 134 * 43 in
  let incorrect_result := 134 * x in
  correct_result - incorrect_result = 1206 → x = 34 :=
by
  sorry

end incorrect_multiplier_l695_695664


namespace quadratic_poly_solution_l695_695446

variable (a b c : ℝ)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_poly_solution
  (h_poly : ∀ x : ℝ, (a * x^2 + b * x + c) = 0)
  (h_dist : (sqrt (discriminant a b c) / |a|) = 1)
  (h_numbers : ∃ p q r : ℝ, ({-a, b, c, discriminant a b c} = {p, q, r, 1/4} ∨ {-a, b, c, discriminant a b c} = {p, q, r, -1} ∨ {-a, b, c, discriminant a b c} = {p, q, r, -3/2})) :
  a = -1/2 :=
sorry

end quadratic_poly_solution_l695_695446


namespace area_between_sin_curves_l695_695983

noncomputable def area_between_curves : ℝ :=
  (1/2) * ∫ (φ : ℝ) in -Real.pi / 2 .. Real.pi / 2, (4 * sin φ)^2 - (2 * sin φ)^2

theorem area_between_sin_curves :
  area_between_curves = 3 * Real.pi :=
by
  sorry

end area_between_sin_curves_l695_695983


namespace translation_preserves_coordinates_l695_695614

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end translation_preserves_coordinates_l695_695614


namespace largest_even_integer_product_l695_695283

theorem largest_even_integer_product (n : ℕ) (h : 2 * n * (2 * n + 2) * (2 * n + 4) * (2 * n + 6) = 5040) :
  2 * n + 6 = 20 :=
by
  sorry

end largest_even_integer_product_l695_695283


namespace complete_the_square_l695_695492

-- Define the initial condition
def initial_eqn (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0

-- Theorem statement for completing the square
theorem complete_the_square (x : ℝ) : initial_eqn x → (x - 3)^2 = 4 :=
by sorry

end complete_the_square_l695_695492


namespace div_problem_l695_695086

variables (A B C : ℝ)

theorem div_problem (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : B = 93 :=
by {
  sorry
}

end div_problem_l695_695086


namespace approximate_probability_l695_695161

-- Define the conditions
def probability_of_success : ℝ := 0.40
def probability_of_failure : ℝ := 1 - probability_of_success
def number_of_trials : ℕ := 5
def number_of_successes : ℕ := 3

-- Use the binomial probability formula

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem approximate_probability :
  binomial_probability number_of_trials number_of_successes probability_of_success = 0.2304 :=
by
  unfold binomial_probability
  rw [Nat.choose, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial, Nat.factorial]
  norm_num
  sorry

end approximate_probability_l695_695161


namespace log_incorrect_reasoning_l695_695180

theorem log_incorrect_reasoning (a : ℝ) (x : ℝ) (h0 : 0 < a) (h1 : a ≠ 1) (h2 : x > 0) :
  (∀ a, 0 < a → a < 1 → ∀ x, x > 0 → log a x < 0) → 
  (∀ x, x > 0 → log 2 x > 0) := 
begin
  -- proof omitted
  sorry
end

end log_incorrect_reasoning_l695_695180


namespace RS_distance_l695_695007

-- Define the points and their coordinates based on the given problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 0, y := 0 }
def Q : Point := { x := 6, y := 0 }
def R : Point := { x := 6, y := 8 }

-- Define S as the intersection of the perpendiculars from P and R.
def S : Point := { x := 6, y := 0 }

-- Define a function to calculate the distance between two points.
def distance (A B : Point) : ℝ :=
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Theorem statement: The distance RS is 8 units.
theorem RS_distance : distance R S = 8 := 
by
  -- The full proof is required here, but we are stating the problem as required.
  sorry

end RS_distance_l695_695007


namespace pants_bought_eq_3_l695_695460

-- The conditions as definitions
def dresses := 5
def cost_per_dress := 20
def jackets := 4
def cost_per_jacket := 30
def transport_cost := 5
def initial_amount := 400
def remaining_amount := 139
def cost_per_pant := 12

-- The mathematical proof problem statement
theorem pants_bought_eq_3 : 
  let total_dresses_cost := dresses * cost_per_dress in
  let total_jackets_cost := jackets * cost_per_jacket in
  let known_expenses := total_dresses_cost + total_jackets_cost + transport_cost in
  let total_spent := initial_amount - remaining_amount in
  let amount_spent_on_pants := total_spent - known_expenses in
  let number_of_pants := amount_spent_on_pants / cost_per_pant in
  number_of_pants = 3 := by
  sorry

end pants_bought_eq_3_l695_695460


namespace range_of_a_l695_695325

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), f(x) = real.exp x + real.exp (2 - x)) →
  (∀ (x : ℝ), ((f x) ^ 2 - a * f x ≤ 0 ↔ f x ≤ a)) →
  (∃! x : ℤ, x = 0 ∨ x = 1 ∨ x = 2) →
  (1 + real.exp 2 ≤ a ∧ a < real.exp (-1) + real.exp 3) :=
by
  -- Proof not required
  sorry

-- Definitions used in conditions
noncomputable def f (x : ℝ) := real.exp x + real.exp (2 - x)

end range_of_a_l695_695325


namespace total_questions_attempted_l695_695847

/-- 
In an examination, a student scores 3 marks for every correct answer and loses 1 mark for
every wrong answer. He attempts some questions and secures 180 marks. The number of questions
he attempts correctly is 75. Prove that the total number of questions he attempts is 120. 
-/
theorem total_questions_attempted
  (marks_per_correct : ℕ := 3)
  (marks_lost_per_wrong : ℕ := 1)
  (total_marks : ℕ := 180)
  (correct_answers : ℕ := 75) :
  ∃ (wrong_answers total_questions : ℕ), 
    total_marks = (marks_per_correct * correct_answers) - (marks_lost_per_wrong * wrong_answers) ∧
    total_questions = correct_answers + wrong_answers ∧
    total_questions = 120 := 
by {
  sorry -- proof omitted
}

end total_questions_attempted_l695_695847


namespace no_such_function_exists_l695_695865

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := 
by
  sorry

end no_such_function_exists_l695_695865


namespace inclusion_exclusion_principle_l695_695288

-- Define our conditions first
variable {N : ℕ}
variable {n : ℕ}
variable {properties : Fin n → Set}
variable {N_i : Fin n → ℕ}
variable {N_ij : Fin n → Fin n → ℕ}

-- The theorem stating the principle of inclusion-exclusion
theorem inclusion_exclusion_principle (N : ℕ) (n : ℕ)
  (properties : Fin n → Set ℕ)
  (N_i : Fin n → ℕ)
  (N_ij : Fin n → Fin n → ℕ)
  (N_ijk : ∀ (i j k: Fin n), ℕ) -- Assuming a generic multi-order case
  -- Add similar generic assumptions for higher order intersections (4, 5, ...) if necessary
  :
  (N - ∑ i, N_i i + ∑ i j, if i < j then N_ij i j else 0 
  - ∑ i j k, if i < j ∧ j < k then N_ijk i j k else 0
  + ∑ i j k₁ k₂, if i < j ∧ j < k₁ ∧ k₁ < k₂ then N_ijk i j k₁ else 0) =
  -- Continue the series as needed
  sorry

end inclusion_exclusion_principle_l695_695288


namespace total_teachers_in_school_l695_695009

theorem total_teachers_in_school (
  senior_teachers : ℕ,
  intermediate_teachers : ℕ,
  other_teachers_sampled : ℕ,
  total_sampled : ℕ,
  remaining_sampled_from_senior_and_intermediate : ℕ) :
  senior_teachers = 26 →
  intermediate_teachers = 104 →
  other_teachers_sampled = 16 →
  total_sampled = 56 →
  remaining_sampled_from_senior_and_intermediate = total_sampled - other_teachers_sampled →
  let total_teachers := senior_teachers + intermediate_teachers + total_sampled * (senior_teachers + intermediate_teachers) / remaining_sampled_from_senior_and_intermediate in
  total_teachers = 182 :=
by
  intros
  sorry

end total_teachers_in_school_l695_695009


namespace unique_fish_total_l695_695897

-- Define the conditions as stated in the problem
def Micah_fish : ℕ := 7
def Kenneth_fish : ℕ := 3 * Micah_fish
def Matthias_fish : ℕ := Kenneth_fish - 15
def combined_fish : ℕ := Micah_fish + Kenneth_fish + Matthias_fish
def Gabrielle_fish : ℕ := 2 * combined_fish

def shared_fish_Micah_Matthias : ℕ := 4
def shared_fish_Kenneth_Gabrielle : ℕ := 6

-- Define the total unique fish computation
def total_unique_fish : ℕ := (Micah_fish + Kenneth_fish + Matthias_fish + Gabrielle_fish) - (shared_fish_Micah_Matthias + shared_fish_Kenneth_Gabrielle)

-- State the theorem
theorem unique_fish_total : total_unique_fish = 92 := by
  -- Proof omitted
  sorry

end unique_fish_total_l695_695897


namespace how_many_women_left_l695_695397

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l695_695397


namespace compute_matrix_vector_l695_695878

variable {R : Type*} [Ring R]
variable {M : Matrix (Fin 2) (Fin 2) R}
variable {u z : Vector (Fin 2) R}

-- Assume the conditions given in the problem
def condition1 : (M.mul_vec u = ![(3 : R), (-4 : R)]) := sorry
def condition2 : (M.mul_vec z = ![(-2 : R), (5 : R)]) := sorry

-- Define the final statement to be proven
theorem compute_matrix_vector :
  M.mul_vec (4 • u - 3 • z) = ![(18 : R), (-31 : R)] :=
by
  -- Proof to be completed
  sorry

end compute_matrix_vector_l695_695878


namespace magnitude_AB_l695_695022

noncomputable def vector_magnitude (z1 z2 : Complex) : Real :=
  Complex.abs (z2 - z1)

theorem magnitude_AB :
  let z1 := (2 : Complex) + Complex.i
  let z2 := (4 : Complex) - 3 * Complex.i
  vector_magnitude z1 z2 = 2 * Real.sqrt 5 :=
by
  sorry

end magnitude_AB_l695_695022


namespace june_ride_time_l695_695871

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end june_ride_time_l695_695871


namespace logical_statements_evaluation_l695_695216

theorem logical_statements_evaluation (p q : Prop) :
  (¬((p ∧ q) → ((p ∨ q) ∧ ¬(p ∨ q)))) ∧
  (¬((¬(p ∧ q)) → ((p ∨ q) → (¬(p ∨ q)))) ∧
  (((p ∨ q) → ¬(¬p))) ∧
  (¬((¬p) → (¬(p ∧ q))))  :=
by
  sorry

end logical_statements_evaluation_l695_695216


namespace n_eq_d_n_squared_l695_695000

noncomputable def d (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Finset.range (n + 1)).filter (λ x, x ∣ n).card

theorem n_eq_d_n_squared (n : ℕ) : n = d(n) * d(n) ↔ n = 1 ∨ n = 9 := by
  sorry

end n_eq_d_n_squared_l695_695000


namespace length_of_CD_l695_695129

theorem length_of_CD
  (radius : ℝ)
  (length : ℝ)
  (total_volume : ℝ)
  (cylinder_volume : ℝ := π * radius^2 * length)
  (hemisphere_volume : ℝ := (2 * (2/3) * π * radius^3))
  (h1 : radius = 4)
  (h2 : total_volume = 432 * π)
  (h3 : total_volume = cylinder_volume + hemisphere_volume) :
  length = 22 := by
sorry

end length_of_CD_l695_695129


namespace find_m_value_l695_695369

noncomputable def power_function_decreasing (m : ℝ) : Prop :=
  let f := λ x: ℝ, (m^2 - m - 1) * x^(m^2 - 2*m - 3)
  (∀ x > 0, (m^2 - m - 1) ≠ 0) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x)

theorem find_m_value : ∃ m : ℝ, power_function_decreasing m ∧ m = 2 :=
by { sorry }

end find_m_value_l695_695369


namespace probability_neither_perfect_square_nor_cube_l695_695528

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l695_695528


namespace number_put_in_machine_l695_695669

theorem number_put_in_machine (x : ℕ) (y : ℕ) (h1 : y = x + 15 - 6) (h2 : y = 77) : x = 68 :=
by
  sorry

end number_put_in_machine_l695_695669


namespace triangle_similarity_l695_695693

-- Definitions based on identified conditions
variables {α : Type*} [normed_field α]

-- Assume a circle centered at some point O with radius r and necessary points on the circle.
variables (O : α) (R : ℝ) (P Q R S N V T : α)

-- Chord PQ bisects chord RS at N
def bisector (PQ RS : α) (N : α) : Prop :=
  dist P N = dist Q N ∧ dist R N = dist S N

-- V is a point between R and N
def between (R N V : α) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ V = t • R + (1 - t) • N

-- Angle PQR is 30 degrees
def ang_PQR_30 (P Q R : α) : Prop :=
  ∃ θ : ℝ, θ = 30 ∧ angle (P - Q) (R - Q) = θ

-- Definitions and assumptions based on the problem conditions
axiom PQ_bisects_RS : bisector P Q R S N
axiom V_between_RN : between R N V
axiom angle_PQR_is_30 : ang_PQR_30 P Q R

-- The proof statement
theorem triangle_similarity : ∃ t : ℝ, t ≠ 0 →
  let angle_PVN = 30, angle_PNV = 15 in
  angle (P - V) (N - V) = angle_PVN ∧ angle (N - V) (P - V) = angle_PNV →
  (triangle PVN) ~ (triangle PVT) :=
sorry

end triangle_similarity_l695_695693


namespace range_of_a_for_two_unequal_roots_l695_695323

theorem range_of_a_for_two_unequal_roots (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * Real.log x₁ = x₁ ∧ a * Real.log x₂ = x₂) ↔ a > Real.exp 1 :=
sorry

end range_of_a_for_two_unequal_roots_l695_695323


namespace geometric_sequence_eighth_term_l695_695514

theorem geometric_sequence_eighth_term (a r : ℝ) (h1 : a * r ^ 3 = 12) (h2 : a * r ^ 11 = 3) : 
  a * r ^ 7 = 6 * Real.sqrt 2 :=
sorry

end geometric_sequence_eighth_term_l695_695514


namespace find_r_l695_695887

theorem find_r (r : ℝ) (h_curve : r = -2 * r^2 + 5 * r - 2) : r = 1 :=
sorry

end find_r_l695_695887


namespace sequence_general_formula_l695_695634

theorem sequence_general_formula (n : ℕ) : 
  let a_n := [9, 99, 999, 9999, ...].nth n in
  a_n = 10^n - 1 := 
sorry

end sequence_general_formula_l695_695634


namespace scientific_notation_2023_l695_695718

def scientific_notation (n : Float) (s : Float) (e : Int) : Prop :=
  n = s * 10^e

def round_to_two_significant_figures (x : Float) : Float :=
  Float.round (10^1 * x) / 10^1

theorem scientific_notation_2023 : scientific_notation 2023 2.0 3 :=
by
  sorry

end scientific_notation_2023_l695_695718


namespace red_balls_probability_l695_695177

-- Definitions of the conditions
def red_balls : ℕ := 4
def blue_balls : ℕ := 5
def green_balls : ℕ := 3
def total_balls : ℕ := red_balls + blue_balls + green_balls
def balls_picked : ℕ := 3

-- Probability calculation functions
def first_draw_prob : ℚ := red_balls / total_balls
def second_draw_prob : ℚ := (red_balls - 1) / (total_balls - 1)
def third_draw_prob : ℚ := (red_balls - 2) / (total_balls - 2)

-- Combined probability
def combined_prob : ℚ := first_draw_prob * second_draw_prob * third_draw_prob

-- Lean statement to prove the probability
theorem red_balls_probability : combined_prob = 1 / 55 :=
by
  sorry

end red_balls_probability_l695_695177


namespace athletes_meet_second_time_at_l695_695987

-- Define the conditions given in the problem
def distance_AB : ℕ := 110

def man_uphill_speed : ℕ := 3
def man_downhill_speed : ℕ := 5

def woman_uphill_speed : ℕ := 2
def woman_downhill_speed : ℕ := 3

-- Define the times for the athletes' round trips
def man_round_trip_time : ℚ := (distance_AB / man_uphill_speed) + (distance_AB / man_downhill_speed)
def woman_round_trip_time : ℚ := (distance_AB / woman_uphill_speed) + (distance_AB / woman_downhill_speed)

-- Lean statement for the proof
theorem athletes_meet_second_time_at :
  ∀ (t : ℚ), t = lcm (man_round_trip_time) (woman_round_trip_time) →
  ∃ d : ℚ, d = 330 / 7 := 
by sorry

end athletes_meet_second_time_at_l695_695987


namespace inverse_proposition_l695_695118

theorem inverse_proposition (q_1 q_2 : ℚ) :
  (q_1 ^ 2 = q_2 ^ 2 → q_1 = q_2) ↔ (q_1 = q_2 → q_1 ^ 2 = q_2 ^ 2) :=
sorry

end inverse_proposition_l695_695118


namespace container_capacity_l695_695992

theorem container_capacity (C : ℝ) (h1 : C > 0) (h2 : 0.40 * C + 14 = 0.75 * C) : C = 40 := 
by 
  -- Would contain the proof here
  sorry

end container_capacity_l695_695992


namespace initial_bananas_on_tree_l695_695082

-- Definitions based on the problem conditions
def bananas_left_on_tree : ℕ := 430
def raj_eaten_bananas : ℕ := 120
def raj_basket_bananas : ℕ := 2 * raj_eaten_bananas
def asha_eaten_bananas : ℕ := 100
def asha_basket_bananas : ℕ := 3 * asha_eaten_bananas
def vijay_eaten_bananas : ℕ := 80
def vijay_basket_bananas : ℕ := 4 * vijay_eaten_bananas

-- Statement we need to prove
theorem initial_bananas_on_tree :
  raj_basket_bananas + asha_basket_bananas + vijay_basket_bananas + bananas_left_on_tree = 1290 :=
begin
  sorry,
end

end initial_bananas_on_tree_l695_695082


namespace probability_not_square_or_cube_l695_695549

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695549


namespace solve_for_x_l695_695917

theorem solve_for_x : ∀ (x : ℕ), (1000 = 10^3) → (40 = 2^3 * 5) → 1000^5 = 40^x → x = 15 :=
by
  intros x h1 h2 h3
  sorry

end solve_for_x_l695_695917


namespace cos_sum_seventh_roots_of_unity_l695_695291

noncomputable def cos_sum (α : ℝ) : ℝ := 
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)

theorem cos_sum_seventh_roots_of_unity (z : ℂ) (α : ℝ)
  (hz : z^7 = 1) (hz_ne_one : z ≠ 1) (hα : z = Complex.exp (Complex.I * α)) :
  cos_sum α = -1/2 :=
by
  sorry

end cos_sum_seventh_roots_of_unity_l695_695291


namespace monomial_properties_l695_695503

def monomial := (3/2) * (a^2 * b)

theorem monomial_properties (a b : ℕ) :
  monomial = (3/2) * (a^2 * b) ∧
  degree(monomial) = 3 :=
by
  -- Proof is not required.
  sorry

end monomial_properties_l695_695503


namespace decaf_percent_second_batch_l695_695665

theorem decaf_percent_second_batch (initial_stock : ℕ) (initial_decaf_percent : ℚ) (second_batch : ℕ) (total_stock_decaf_percent : ℚ) : 
  initial_stock = 400 → 
  initial_decaf_percent = 20% → 
  second_batch = 100 → 
  total_stock_decaf_percent = 28% → 
  ((total_stock_decaf_percent * (initial_stock + second_batch) - initial_decaf_percent * initial_stock) / second_batch) = 60 := 
by 
  intros 
  sorry

end decaf_percent_second_batch_l695_695665


namespace number_of_integer_values_P_is_square_l695_695882

theorem number_of_integer_values_P_is_square :
  let P (x : ℤ) := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 31 in
  {x : ℤ | ∃ k : ℤ, P x = k * k}.card = 1 :=
sorry

end number_of_integer_values_P_is_square_l695_695882


namespace part1_first_5_terms_part2_sufficiency_part2_necessity_counterexample_part3_general_formula_l695_695302

-- Definitions for part (1)
def seq_bn_part1 : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 2
| 4 := 2
| 5 := 3
| _ := 0 -- hasn't been determined for n > 5

-- Sequence definition for part (2)
def strictly_increasing_condition (a : ℕ → ℤ) : Prop :=
  (a 1 % 2 = 1) ∧ (∀ i, i ≥ 2 → a i % 2 = 0)

-- The mathematical proof statements
theorem part1_first_5_terms (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 5) : b_n = seq_bn_part1 n := sorry

theorem part2_sufficiency (a : ℕ → ℤ) (h : strictly_increasing_condition a) :
  ∀ n, b_n < b_(n+1) := sorry

theorem part2_necessity_counterexample : ∃ a : ℕ → ℤ,
  (∀ n, a 1 % 2 ≠ 1 ∨ ∃ i, i ≥ 2 ∧ a i % 2 ≠ 0) ∧ 
  (∀ n, b_n < b_(n+1)) := sorry

theorem part3_general_formula (a : ℕ → ℤ) (h : ∀ i, a i = b_i) : ∀ n, a n = 0 := sorry

end part1_first_5_terms_part2_sufficiency_part2_necessity_counterexample_part3_general_formula_l695_695302


namespace find_n_l695_695958

/-- Let \( n \) be an integer. If \( 0 \le n < 144 \) and \( 143n \equiv 105 \pmod{144} \), then \( n = 39 \). -/
theorem find_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 144 ∧ 143 * n % 144 = 105 := 
begin
  use 39,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end find_n_l695_695958


namespace sum_first_10_common_terms_eq_6990500_l695_695743

-- Define the arithmetic progression
def is_arithmetic_term (n : ℕ) : ℕ := 5 + 3 * n

-- Define the geometric progression
def is_geometric_term (k : ℕ) : ℕ := 10 * 2^k

-- Predicate to check if a term is common in both progressions
def is_common_term (m : ℕ) : Prop :=
  ∃ n k, m = is_arithmetic_term n ∧ m = is_geometric_term k ∧ k % 2 = 1

-- Sum of the first 10 common terms
def sum_of_first_10_common_terms : ℕ :=
  let common_terms := [20, 80, 320, 1280, 5120, 20480, 81920, 327680, 1310720, 5242880] in
  common_terms.sum

-- Main theorem statement
theorem sum_first_10_common_terms_eq_6990500 :
  sum_of_first_10_common_terms = 6990500 :=
by
  sorry

end sum_first_10_common_terms_eq_6990500_l695_695743


namespace three_not_in_range_iff_l695_695760

theorem three_not_in_range_iff (c : ℝ) : (∀ x : ℝ, g(x) = x^2 + 2*x + c → g(x) ≠ 3) ↔ c > 4 :=
by
sorry

end three_not_in_range_iff_l695_695760


namespace distance_city_A_C_l695_695713

-- Define the conditions
def starts_simultaneously (A : Prop) (Eddy Freddy : Prop) := Eddy ∧ Freddy
def travels (A B C : Prop) (Eddy Freddy : Prop) := Eddy → 3 = 3 ∧ Freddy → 4 = 4
def distance_AB (A B : Prop) := 600
def speed_ratio (Eddy_speed Freddy_speed : ℝ) := Eddy_speed / Freddy_speed = 1.7391304347826086

noncomputable def distance_AC (Eddy_time Freddy_time : ℝ) (Eddy_speed Freddy_speed : ℝ) 
  := (Eddy_speed / 1.7391304347826086) * Freddy_time

theorem distance_city_A_C 
  (A B C Eddy Freddy : Prop)
  (Eddy_time Freddy_time : ℝ) 
  (Eddy_speed effective_Freddy_speed : ℝ)
  (h1 : starts_simultaneously A Eddy Freddy)
  (h2 : travels A B C Eddy Freddy)
  (h3 : distance_AB A B = 600)
  (h4 : speed_ratio Eddy_speed effective_Freddy_speed)
  (h5 : Eddy_speed = 200)
  (h6 : effective_Freddy_speed = 115)
  : distance_AC Eddy_time Freddy_time Eddy_speed effective_Freddy_speed = 460 := 
  by sorry

end distance_city_A_C_l695_695713


namespace limit_fraction_is_2_l695_695232

noncomputable def limit_fraction : ℝ :=
  lim_n_to_inf (λ n, (2 * ↑n - 5) / (↑n + 1))

theorem limit_fraction_is_2 :
  limit_fraction = 2 :=
sorry

end limit_fraction_is_2_l695_695232


namespace build_wall_30_persons_l695_695173

-- Defining the conditions
def work_rate (persons : ℕ) (days : ℕ) : ℚ := 1 / (persons * days)

-- Total work required to build the wall by 8 persons in 42 days
def total_work : ℚ := work_rate 8 42 * 8 * 42

-- Work rate for 30 persons
def combined_work_rate (persons : ℕ) : ℚ := persons * work_rate 8 42

-- Days required for 30 persons to complete the same work
def days_required (persons : ℕ) (work : ℚ) : ℚ := work / combined_work_rate persons

-- Expected result is 11.2 days for 30 persons
theorem build_wall_30_persons : days_required 30 total_work = 11.2 := 
by
  sorry

end build_wall_30_persons_l695_695173


namespace adam_paper_tearing_l695_695210

theorem adam_paper_tearing (n : ℕ) :
  let starts_with_one_piece : ℕ := 1
  let increment_to_four : ℕ := 3
  let increment_to_ten : ℕ := 9
  let target_pieces : ℕ := 20000
  let start_modulo : ℤ := 1

  -- Modulo 3 analysis
  starts_with_one_piece % 3 = start_modulo ∧
  increment_to_four % 3 = 0 ∧ 
  increment_to_ten % 3 = 0 ∧ 
  target_pieces % 3 = 2 → 
  n % 3 = start_modulo ∧ ∀ m, m % 3 = 0 → n + m ≠ target_pieces :=
sorry

end adam_paper_tearing_l695_695210


namespace greatest_b_value_l695_695151

theorem greatest_b_value (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 15 ≠ -9) ↔ b = 9 :=
sorry

end greatest_b_value_l695_695151


namespace locus_of_orthocenters_is_polar_of_P_l695_695292

open_locale classical

-- Define the circle and its properties
variables {O P A B : Type} [metric_space O] [metric_space P] [metric_space A] [metric_space B]
          {circle : Type} [metric_space circle]

-- Define the circle and the point P not on the circle
axiom h_circle : ∃ O, is_circle O
axiom h_not_on_circle : ∃ P, ¬ (is_on_circle P O)

-- Define the diameter AB
axiom h_diameter : ∃ A B, (is_on_circle A O ∧ is_on_circle B O ∧ dist O A = dist O B)

-- The main theorem to be proved, stating that the locus of orthocenters is the polar of P
theorem locus_of_orthocenters_is_polar_of_P : 
  ∀ ABP (h1 : ABP = triangle A B P), is_orthocenter ABP → is_polar P O :=
sorry

end locus_of_orthocenters_is_polar_of_P_l695_695292


namespace mass_percentage_of_H_in_ascorbic_acid_l695_695725

-- Definitions based on the problem conditions
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

def ascorbic_acid_molecular_formula_C : ℝ := 6
def ascorbic_acid_molecular_formula_H : ℝ := 8
def ascorbic_acid_molecular_formula_O : ℝ := 6

noncomputable def ascorbic_acid_molar_mass : ℝ :=
  ascorbic_acid_molecular_formula_C * molar_mass_C + 
  ascorbic_acid_molecular_formula_H * molar_mass_H + 
  ascorbic_acid_molecular_formula_O * molar_mass_O

noncomputable def hydrogen_mass_in_ascorbic_acid : ℝ :=
  ascorbic_acid_molecular_formula_H * molar_mass_H

noncomputable def hydrogen_mass_percentage_in_ascorbic_acid : ℝ :=
  (hydrogen_mass_in_ascorbic_acid / ascorbic_acid_molar_mass) * 100

theorem mass_percentage_of_H_in_ascorbic_acid :
  hydrogen_mass_percentage_in_ascorbic_acid = 4.588 :=
by
  sorry

end mass_percentage_of_H_in_ascorbic_acid_l695_695725


namespace probability_engineers_crimson_l695_695973

theorem probability_engineers_crimson (teams : Fin 128) (equally_strong : ∀ (a b : Fin 128), (1 / 2)) 
  (randomly_paired : ∀ (a b : Fin 128), (1 / (128.choose 2))) :
  let matches := 127 
  let total_pairs := 128.choose 2 
  (matches / total_pairs : ℝ) = (1 / 64 : ℝ) :=
by
  sorry

end probability_engineers_crimson_l695_695973


namespace min_le_mult_l695_695876

theorem min_le_mult {x y z m : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : m ≤ x * y^2 * z^3 :=
by
  sorry

end min_le_mult_l695_695876


namespace probability_neither_perfect_square_nor_cube_l695_695582

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695582


namespace T_is_positive_l695_695809

noncomputable def a : ℕ → ℕ
| 1     := 3
| (n+1) := 2 * a n + 2 ^ (n+1) - 1

def b (n : ℕ) : ℕ := (a n - 1) / 2^n

def c (n : ℕ) : ℚ := (-1)^n / b n

def T (n : ℕ) : ℚ := ∑ k in Finset.range n, c (k + 1)

theorem T_is_positive (n : ℕ) : T (2 * n) > - (Real.sqrt 2) / 2 :=
sorry

end T_is_positive_l695_695809


namespace transformation_of_f_to_g_l695_695800

def f (x : ℝ) : ℝ := sin (2 * x + π / 3)
def g (x : ℝ) : ℝ := sqrt 3 * sin (2 * x)

-- Stating the transformation
theorem transformation_of_f_to_g : 
  ∀ x, g x = sqrt 3 * f (x - π / 6) :=
sorry

end transformation_of_f_to_g_l695_695800


namespace simplify_sqrt3_7_pow6_l695_695474

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695474


namespace polygon_affinity_l695_695646

theorem polygon_affinity
    (S S1 S2 : Type) 
    [polygon : is_polygon S]
    [plane1 : projection_plane S1]
    [plane2 : projection_plane S2] 
    (x12 : intersection_line S1 S2) 
    (not_perpendicular_S_S1_S2 : ¬(is_perpendicular S S1 ∧ is_perpendicular S S2)) :
    ∃ affinity : S1 → S2, ∀ (P P' P'' : point_in_polygon S), 
        (is_first_image P P' S1) ∧ (is_second_image P P'' S2) → 
            (lines_connecting_corresponding_points_are_parallel P' P'') ∧
            (intersection_points_of_corresponding_lines_lie_on_a_straight_line P' P'') :=
sorry

end polygon_affinity_l695_695646


namespace card_one_on_top_l695_695774

theorem card_one_on_top (n : ℕ) (cards : List ℕ) :
  (1 ∈ cards) → ∃ seq : List (List ℕ) → List ℕ, card_one_on_top :=

sorry

end card_one_on_top_l695_695774


namespace reporters_cover_local_politics_l695_695656

theorem reporters_cover_local_politics (total_reporters politics_reporters local_politics_percent not_politics_percent not_local_politics_percent: ℕ)
  (h1 : politics_reporters = total_reporters * (1 - not_politics_percent / 100))
  (h2 : local_politics_percent = politics_reporters * (1 - not_local_politics_percent / 100 / total_reporters)):
  local_politics_percent / total_reporters * 100 = 28 :=
by
  -- Here we assume some initial values and validate using the conditions provided above.
  let total_reporters := 100
  let not_politics_percent := 60
  let not_local_politics_percent := 30

  have politics_reporters_eq : total_reporters * (1 - not_politics_percent / 100) = 40 :=
    by
      calc total_reporters * (1 - not_politics_percent / 100)
        = 100 * (1 - 60 / 100) : by rfl
        ... = 100 * 0.4 : by norm_num
        ... = 40 : by norm_num

  have local_politics_percent_eq : 40 * (1 - not_local_politics_percent / 100) = 28 :=
    by
      calc 40 * (1 - not_local_politics_percent / 100)
        = 40 * (1 - 30 / 100) : by rfl
        ... = 40 * 0.7 : by norm_num
        ... = 28 : by norm_num

  have result : 28 / 100 * 100 = 28 :=
    by
      calc 28 / 100 * 100
        = 28 / 1 : by norm_num
        ... = 28 : by norm_num

  exact result

end reporters_cover_local_politics_l695_695656


namespace box_weights_l695_695441

theorem box_weights (a b c : ℕ) (h1 : a + b = 132) (h2 : b + c = 135) (h3 : c + a = 137) (h4 : a > 40) (h5 : b > 40) (h6 : c > 40) : a + b + c = 202 :=
by 
  sorry

end box_weights_l695_695441


namespace total_rainfall_l695_695408

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l695_695408


namespace probability_neither_square_nor_cube_l695_695567

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695567


namespace triangle_ADE_right_iff_sides_ABC_ratio_3_4_5_l695_695055

-- Given conditions
variable (A B C E D : Type) [Point A] [Point B] [Point C] [Point E] [Point D]
variable (AB AC BC AE BD : ℝ)
variable (h1 : right_triangle A B C)
variable (h2 : AC > AB)
variable (h3 : AE = AB)
variable (h4 : BD = AB)

-- Correct answer as theorem statement
theorem triangle_ADE_right_iff_sides_ABC_ratio_3_4_5 :
  (right_triangle A D E) ↔ ((AB/BC) = 3/5 ∧ (AC/BC) = 4/5) :=
sorry

end triangle_ADE_right_iff_sides_ABC_ratio_3_4_5_l695_695055


namespace probability_neither_square_nor_cube_l695_695536

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695536


namespace part1_monotonicity_part2_minimum_range_l695_695799

noncomputable def f (k x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

theorem part1_monotonicity (x : ℝ) (h : x ≠ 1) :
    k = 0 → f k x = (x / (x - 1)) * Real.log x ∧ 
    (0 < x ∧ x < 1 ∨ 1 < x) → Monotone (f k) :=
sorry

theorem part2_minimum_range (k : ℝ) :
    (∃ x ∈ Set.Ioi 1, IsLocalMin (f k) x) ↔ k ∈ Set.Ioi 1 :=
sorry

end part1_monotonicity_part2_minimum_range_l695_695799


namespace find_MC_length_l695_695013

variables {A B C D E M : Type}
variables [rect : is_rectangle A B C D]
variables {BC AE BM : ℝ}
variables (E_diag : E ∈ AC ∧ dist B C = dist E C)
variables (M_on_BC : M ∈ BC ∧ dist E M = dist M C)
variables (BM_val : BM = 6)
variables (AE_val : AE = 3)

theorem find_MC_length : ∃ MC : ℝ, MC = 9 :=
by
  have := E_diag,
  have := M_on_BC,
  have := BM_val,
  have := AE_val,
  sorry

end find_MC_length_l695_695013


namespace soccer_team_players_l695_695199

theorem soccer_team_players
  (first_half_starters : ℕ)
  (first_half_subs : ℕ)
  (second_half_mult : ℕ)
  (did_not_play : ℕ)
  (players_prepared : ℕ) :
  first_half_starters = 11 →
  first_half_subs = 2 →
  second_half_mult = 2 →
  did_not_play = 7 →
  players_prepared = 20 :=
by
  -- Proof steps go here
  sorry

end soccer_team_players_l695_695199


namespace fish_vs_tadpoles_l695_695012

/-- Initial conditions -/
def initial_fish : ℕ := 150
def initial_tadpoles : ℕ := 5 * initial_fish
def initial_snails : ℕ := 200

/-- Curtis catches -/
def caught_fish : ℕ := 20
def caught_tadpoles : ℕ := (3 / 5 : ℚ) * initial_tadpoles
def caught_snails : ℕ := 30

/-- Curtis releases -/
def released_fish : ℕ := caught_fish / 2
def released_snails : ℕ := caught_snails / 2
def released_tadpoles : ℕ := released_fish + released_snails

/-- Numbers after releases -/
def remaining_fish : ℕ := initial_fish - caught_fish + released_fish
def remaining_tadpoles : ℕ := initial_tadpoles - caught_tadpoles + released_tadpoles
def remaining_snails : ℕ := initial_snails - caught_snails + released_snails

/-- Tadpoles develop into frogs -/
def developed_tadpoles : ℕ := (2 / 3 : ℚ) * remaining_tadpoles
def final_tadpoles : ℕ := remaining_tadpoles - developed_tadpoles

theorem fish_vs_tadpoles :
  final_tadpoles - remaining_fish = -31 := by
  sorry

end fish_vs_tadpoles_l695_695012


namespace eggs_total_l695_695078

-- Definitions of the conditions in Lean
def num_people : ℕ := 3
def omelets_per_person : ℕ := 3
def eggs_per_omelet : ℕ := 4

-- The claim we need to prove
theorem eggs_total : (num_people * omelets_per_person) * eggs_per_omelet = 36 :=
by
  sorry

end eggs_total_l695_695078


namespace find_angle_ECB_l695_695855

noncomputable theory
open_locale classical

variables {A B C D E : Type}
variables [geometry A B C D E]
-- angles in degrees
variables (angle_DCA angle_ABC angle_ABE : ℝ)
-- parallel lines
variables (line_AB line_DC : ℕ)

-- conditions
def parallel_lines : Prop := line_DC ∥ line_AB
def given_angles : Prop := angle_DCA = 50 ∧ angle_ABC = 60 ∧ angle_ABE = 40

theorem find_angle_ECB
  (h1 : parallel_lines)
  (h2 : given_angles):
  ∠ECB = 50 := sorry

end find_angle_ECB_l695_695855


namespace fraction_exponent_product_l695_695149

theorem fraction_exponent_product :
  ( (5/6: ℚ)^2 * (2/3: ℚ)^3 = 50/243 ) :=
by
  sorry

end fraction_exponent_product_l695_695149


namespace ratio_of_areas_l695_695018

-- Define the setup for a regular decagon
structure RegularDecagon :=
  (A B C D E F G H I J : Point)
  (is_regular_decagon : regular_polygon A B C D E F G H I J 10)

-- Define the midpoints P and Q
def P (dec : RegularDecagon) := midpoint (dec.C) (dec.D)
def Q (dec : RegularDecagon) := midpoint (dec.H) (dec.I)

-- Define the areas in question
noncomputable def area_ABCPQ (dec : RegularDecagon) : ℝ := sorry
noncomputable def area_DEFGHPQ (dec : RegularDecagon) : ℝ := sorry

-- Define the statement to prove the ratio of the areas is 1/3
theorem ratio_of_areas (dec : RegularDecagon) :
  (area_ABCPQ dec) / (area_DEFGHPQ dec) = (1 / 3) :=
sorry

end ratio_of_areas_l695_695018


namespace simplify_and_evaluate_expr_l695_695467

-- Defining the condition
def x := Real.sqrt 3 + 1

-- Defining the expression
def expr := ((1 / (x - 2) + 1) / ((x^2 - 2 * x + 1) / (x - 2)))

-- The theorem that we need to prove
theorem simplify_and_evaluate_expr : expr = Real.sqrt 3 / 3 :=
by 
  -- Skipping the proof as requested
  sorry

end simplify_and_evaluate_expr_l695_695467


namespace arc_length_correct_l695_695502

noncomputable def length_of_arc_bc (circumference : ℝ) (angle_degrees : ℝ) : ℝ :=
  (angle_degrees / 360) * circumference

theorem arc_length_correct :
  ∀ (circumference : ℝ) (angle_degrees : ℝ),
  circumference = 80 →
  angle_degrees = 120 →
  length_of_arc_bc circumference angle_degrees = 80 / 3 :=
by
  intros
  rw [length_of_arc_bc, ‹circumference = 80›, ‹angle_degrees = 120›]
  norm_num
  sorry

end arc_length_correct_l695_695502


namespace probability_neither_square_nor_cube_l695_695562

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695562


namespace final_configuration_l695_695030

def initial_configuration : (String × String) :=
  ("bottom-right", "bottom-left")

def first_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("bottom-right", "bottom-left") => ("top-right", "top-left")
  | _ => conf

def second_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("top-right", "top-left") => ("top-left", "top-right")
  | _ => conf

theorem final_configuration :
  second_transformation (first_transformation initial_configuration) =
  ("top-left", "top-right") :=
by
  sorry

end final_configuration_l695_695030


namespace cost_price_of_product_l695_695518

theorem cost_price_of_product (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  (marked_price = 120) → 
  (discount_rate = 0.1) → 
  (profit_rate = 0.2) → 
  let cost_price := 90 in
  marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end cost_price_of_product_l695_695518


namespace incorrect_average_initially_l695_695499

theorem incorrect_average_initially (S : ℕ) :
  (S + 25) / 10 = 46 ↔ (S + 65) / 10 = 50 := by
  sorry

end incorrect_average_initially_l695_695499


namespace AQ_eq_CO_l695_695612

variable {Point : Type*} [AffineSpace ℝ Point]

variables (A B C D O P Q L : Point)
variables (AB CD AC BD AD BC AQ CO : ℝ)

-- Hypotheses
variables (is_trapezium : true)  -- This will need your definition of a trapezium in Lean
variables (parallel_AB_BCP : true)  -- Line through C parallel to AB
variables (intersects_BD_at_P : true)  -- Line through C intersects BD at P
variables (intersection_O : true)  -- O is intersection of AC and BD
variables (DP_BO : true)  -- DP = BO
variables (parallel_CD_BQ : true)  -- Line through B parallel to CD intersects AC at Q

-- Define the theorem statement
theorem AQ_eq_CO (h1 : is_trapezium)
                 (h2 : parallel_AB_BCP)
                 (h3 : intersects_BD_at_P)
                 (h4 : intersection_O)
                 (h5 : DP_BO)
                 (h6 : parallel_CD_BQ) :
    AQ = CO :=
sorry

end AQ_eq_CO_l695_695612


namespace chinese_characters_digits_l695_695789

theorem chinese_characters_digits:
  ∃ (a b g s t : ℕ), -- Chinese characters represented by digits
    -- Different characters represent different digits
    a ≠ b ∧ a ≠ g ∧ a ≠ s ∧ a ≠ t ∧
    b ≠ g ∧ b ≠ s ∧ b ≠ t ∧
    g ≠ s ∧ g ≠ t ∧
    s ≠ t ∧
    -- Equation: 业步高 * 业步高 = 高升抬步高
    (a * 100 + b * 10 + g) * (a * 100 + b * 10 + g) = (g * 10000 + s * 1000 + t * 100 + b * 10 + g) :=
by {
  -- We need to prove that the number represented by "高升抬步高" is 50625.
  sorry
}

end chinese_characters_digits_l695_695789


namespace cos_monotonic_increasing_interval_l695_695122

open Real

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

theorem cos_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ,
    (∃ y, y = cos (π / 3 - 2 * x)) →
    (monotonic_increasing_interval k x) :=
by
  sorry

end cos_monotonic_increasing_interval_l695_695122


namespace total_area_of_squares_l695_695385

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨12, 0⟩
def C : Point := ⟨12, 12⟩

def length (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def total_area (A B C : Point) : ℝ :=
  let side_length_1 := length A B
  let side_length_2 := length B C
  area_of_square side_length_1 + area_of_square side_length_2

theorem total_area_of_squares :
  total_area A B C = 288 := by
  sorry

end total_area_of_squares_l695_695385


namespace quadratic_function_proof_l695_695339

theorem quadratic_function_proof (a c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^2 - 4 * x + c)
  (h_sol_set : ∀ x, f x < 0 → (-1 < x ∧ x < 5)) :
  (a = 1 ∧ c = -5) ∧ (∀ x, 0 ≤ x ∧ x ≤ 3 → -9 ≤ f x ∧ f x ≤ -5) :=
by
  sorry

end quadratic_function_proof_l695_695339


namespace make_graph_monochromatic_l695_695711

-- Define the complete graph K_30
structure CompleteGraph (n : Nat) :=
  (edges : List (Fin n × Fin n))
  (complete : ∀ (v u : Fin n), (v, u) ∈ edges ∨ (u, v) ∈ edges ∧ v ≠ u)

-- Define the concept of edge coloring
inductive Color
| red
| blue

-- Define the edge coloring for K_30
structure EdgeColoring (G : CompleteGraph 30) :=
  (color : (Fin 30 × Fin 30) → Color)
  (symmetric : ∀ {a b : Fin 30}, color (a, b) = color (b, a))

-- Define a non-monochromatic triangle and the color changing operation
def isNonMonochromaticTriangle (G : CompleteGraph 30) (c : EdgeColoring G) (u v w : Fin 30) : Prop :=
  let cuv := c.color (u, v)
  let cvw := c.color (v, w)
  let cwu := c.color (w, u)
  (cuv ≠ cvw) ∧ (cvw ≠ cwu) ∧ (cwu ≠ cuv)

def makeTriangleMonochromatic (G : CompleteGraph 30) (c : EdgeColoring G) (u v w : Fin 30) 
                              (h : isNonMonochromaticTriangle G c u v w) : EdgeColoring G :=
  let cuv := c.color (u, v)
  let newColor := if cuv = c.color (w, u) then c.color (v, w) else cuv
  { color := 
      fun e => if e = (u, v) ∨ e = (v, u) ∨ e = (v, w) ∨ e = (w, v)
               then newColor
               else c.color e,
    symmetric := 
      by {
        intros a b,
        rw [←(EdgeColoring.symmetric c)],
        simp,
      }
  }

-- Define the statement to prove the entire graph can be made monochromatic
theorem make_graph_monochromatic (G : CompleteGraph 30) (c : EdgeColoring G) :
  ∃ c' : EdgeColoring G, ∀ (u v: Fin 30), c'.color (u, v) = c'.color (v, u) :=
by
  -- the proof goes here
  sorry

end make_graph_monochromatic_l695_695711


namespace problem_k_value_l695_695100

theorem problem_k_value (a b c : ℕ) (h1 : a + b / c = 101) (h2 : a / c + b = 68) :
  (a + b) / c = 13 :=
sorry

end problem_k_value_l695_695100


namespace probability_neither_perfect_square_nor_cube_l695_695581

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l695_695581


namespace total_rainfall_over_3_days_l695_695410

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l695_695410


namespace power_of_i_sum_l695_695715

theorem power_of_i_sum : 
  let i := Complex.I in
  (i^22 + i^222) = -2 := by 
  sorry

end power_of_i_sum_l695_695715


namespace regular_ngon_in_grid_l695_695206

theorem regular_ngon_in_grid (n : ℕ) (h₁ : n ≥ 3) 
  (grid : ℤ × ℤ → Prop) (vertices_in_grid : ∀ i : Fin n, grid (v i)) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_ngon_in_grid_l695_695206


namespace hyperbola_equation_correct_l695_695722

theorem hyperbola_equation_correct:
  ∃ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) ∧
  (∀ x y : ℝ, (x, y) = (2, -2) → (x^2 / a - y^2 / b = 1) ∧ (y^2 / 2 - x^2 / 4 = 1)) :=
by
  -- Assume a and b represents the coefficient for x^2 and y^2 respectively
  use [2, -4]
  split
  sorry  -- this proof will check the assumptions and prove the final equation

end hyperbola_equation_correct_l695_695722


namespace correct_option_for_ruler_length_l695_695071

theorem correct_option_for_ruler_length (A B C D : String) (correct_answer : String) : 
  A = "two times as longer as" ∧ 
  B = "twice the length of" ∧ 
  C = "three times longer of" ∧ 
  D = "twice long than" ∧ 
  correct_answer = B := 
by
  sorry

end correct_option_for_ruler_length_l695_695071


namespace sum_of_valid_x_values_l695_695259

-- Define the given numbers
def numbers (x : ℝ) : List ℝ := [3, 5, 7, 15, x]

-- Function to compute the mean of the list of numbers
def mean (x : ℝ) : ℝ := (3 + 5 + 7 + 15 + x) / 5

-- Function to determine the median of a list of 5 sorted real numbers
def median (lst : List ℝ) : ℝ :=
  if lst.length = 5 then nth_le lst 2 (by simp) else 0 -- assuming the list is sorted

-- Define the problem conditions
def valid_x_condition (x : ℝ) : Prop :=
  let sorted_numbers := (numbers x).qsort (≤)
  median sorted_numbers = mean x

-- Theorem statement
theorem sum_of_valid_x_values : ∑ x in ({x : ℝ | valid_x_condition x}.toFinset), x = -5 := sorry

end sum_of_valid_x_values_l695_695259


namespace measure_of_larger_angle_l695_695141

variables (x : ℝ) (smaller_angle larger_angle : ℝ)

def supplementary (a b : ℝ) : Prop := a + b = 180

def ratio_4_5 (a b : ℝ) : Prop := 4 * b = 5 * a

theorem measure_of_larger_angle :
  supplementary (4 * x) (5 * x) →
  ratio_4_5 (4 * x) (5 * x) →
  larger_angle = 100 :=
by
  intros h1 h2
  have h3 : 9 * x = 180 := sorry
  have h4 : x = 20 := sorry
  have h5 : 5 * x = 100 := sorry
  exact h5

end measure_of_larger_angle_l695_695141


namespace average_age_increase_l695_695107

theorem average_age_increase (average_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ)
                             (h1 : average_age_students = 26) (h2 : num_students = 25) (h3 : teacher_age = 52)
                             (h4 : new_avg_age = (650 + teacher_age) / (num_students + 1))
                             (h5 : 650 = average_age_students * num_students) :
  new_avg_age - average_age_students = 1 := 
by
  sorry

end average_age_increase_l695_695107


namespace range_of_a_l695_695004

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1) := 
sorry

end range_of_a_l695_695004


namespace classroom_chairs_l695_695384

-- Definitions of the given conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def white_chairs : ℕ := (blue_chairs + green_chairs) - 13
def total_chairs : ℕ := blue_chairs + green_chairs + white_chairs

-- Statement to prove
theorem classroom_chairs : total_chairs = 67 := by
  -- We substitute the conditions and compute the total number of chairs
  have h1 : total_chairs = blue_chairs + green_chairs + white_chairs := rfl
  have h2 : blue_chairs = 10 := rfl
  have h3 : green_chairs = 3 * 10 := rfl
  have h4 : green_chairs = 30 := by simp [h3]
  have h5 : blue_chairs + green_chairs = 10 + 30 := by simp [h2, h4]
  have h6 : blue_chairs + green_chairs = 40 := by simp [h5]
  have h7 : white_chairs = 40 - 13 := rfl
  have h8 : white_chairs = 27 := by simp [h7]
  have h9 : total_chairs = 10 + 30 + 27 := by simp [h2, h4, h8]
  show total_chairs = 67 from by simp [h9]

end classroom_chairs_l695_695384


namespace necessary_but_not_sufficient_condition_l695_695755

def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem necessary_but_not_sufficient_condition (x y : ℝ) :
  |x - y| < 1 → ceil x = ceil y := sorry

end necessary_but_not_sufficient_condition_l695_695755


namespace payment_of_employee_B_l695_695981

-- Define the variables and conditions
variables (A B : ℝ) (total_payment : ℝ) (payment_ratio : ℝ)

-- Assume the given conditions
def conditions : Prop := 
  (A + B = total_payment) ∧ 
  (A = payment_ratio * B) ∧ 
  (total_payment = 550) ∧ 
  (payment_ratio = 1.5)

-- Prove the payment of employee B is 220 given the conditions
theorem payment_of_employee_B : conditions A B total_payment payment_ratio → B = 220 := 
by
  sorry

end payment_of_employee_B_l695_695981


namespace calculate_series_l695_695234

theorem calculate_series : 20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 200 := 
by
  sorry

end calculate_series_l695_695234


namespace robotics_club_students_neither_cs_nor_elec_l695_695901

theorem robotics_club_students_neither_cs_nor_elec 
  (total_students : ℕ)
  (cs_students : ℕ)
  (elec_students : ℕ)
  (both_cs_elec : ℕ)
  (total_students = 60)
  (cs_students = 45)
  (elec_students = 33)
  (both_cs_elec = 25) 
  : (total_students - (cs_students - both_cs_elec + elec_students - both_cs_elec + both_cs_elec) = 7) :=
by 
  sorry

end robotics_club_students_neither_cs_nor_elec_l695_695901


namespace probability_neither_square_nor_cube_l695_695537

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695537


namespace _l695_695912

-- Definitions that are directly used in the conditions
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

-- The theorem we want to prove
lemma lcm_inequality (k m n : ℕ) : lcm k m * lcm m n * lcm n k ≥ lcm3 k m n ^ 2 :=
sorry

end _l695_695912


namespace sin_cos_sum_l695_695320

open Real -- Open the real number namespace

-- Define the key elements for this proof problem
variables (x y : ℝ) (r : ℝ)
noncomputable def P := (-5 : ℝ, 12 : ℝ)
def alpha := classical.some (exists_angle P)

-- Translation of given conditions
axiom terminal_side_condition : ∃ (α : ℝ), P = (r * cos α, r * sin α) ∧ r = sqrt (x^2 + y^2)

-- Lean statement for the problem
theorem sin_cos_sum : (sin alpha + cos alpha = 7 / 13) :=
by 
  have x_val : x = -5 := rfl
  have y_val : y = 12 := rfl
  have r_val : r = 13 := by rw [← sqrt_sq, pow_two, pow_two, add_comm]; linarith
  sorry

end sin_cos_sum_l695_695320


namespace twice_x_minus_y_neg_l695_695972

theorem twice_x_minus_y_neg (x y : ℝ) : 
  2 * x - y < 0 ↔ "twice x minus y is a negative number" := sorry

end twice_x_minus_y_neg_l695_695972


namespace simplify_sqrt3_7_pow6_l695_695471

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695471


namespace minimum_value_expression_l695_695726

theorem minimum_value_expression :
  ∃ x y : ℝ, (∀ a b : ℝ, (a^2 + 4*a*b + 5*b^2 - 8*a - 6*b) ≥ -41) ∧ (x^2 + 4*x*y + 5*y^2 - 8*x - 6*y) = -41 := 
sorry

end minimum_value_expression_l695_695726


namespace max_student_numbers_l695_695014

open Nat

theorem max_student_numbers (S : Finset ℕ) (h : ∀ x y ∈ S, x ≠ y → x.gcd y ∉ S) :
  S.card ≤ 3721 :=
sorry

end max_student_numbers_l695_695014


namespace problem_statement_l695_695427

variable (a b c : ℝ)

-- Define proposition p
def p : Prop := a > b → a + c > b + c

-- Define proposition q
def q : Prop := a > b ∧ b > 0 → a * c > b * c

-- Define the target proposition (not p or not q is true)
def target_proposition : Prop := ¬p ∨ ¬q

-- The theorem stating the problem:
theorem problem_statement : p → ¬q → target_proposition := by
  intros hp hnq
  exact Or.inr hnq -- Or of two, use right component ¬q
  sorry

end problem_statement_l695_695427


namespace june_bernard_travel_time_l695_695869

-- Defining the given conditions
def distance_june_julia : ℝ := 2
def time_june_julia : ℝ := 6
def distance_june_bernard : ℝ := 5

-- Rate calculation premise
def travel_rate : ℝ := distance_june_julia / time_june_julia

-- Proof problem statement
theorem june_bernard_travel_time :
  travel_rate * distance_june_bernard = 15 := sorry

end june_bernard_travel_time_l695_695869


namespace inequalities_hold_l695_695253

variables {ℝ : Type*}
variables (a b : ℝ) (f g : ℝ → ℝ)

-- Assume the given conditions:
-- 1. f is odd and increasing
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- 2. g is even and coincides with f on [0, +∞)
def even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def coincides_on_nonneg (f g : ℝ → ℝ) := ∀ x, 0 ≤ x → f x = g x

-- 3. a > b > 0
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0

-- Define the hypotheses:
axiom f_is_odd : odd f
axiom f_is_increasing : increasing f
axiom g_is_even : even g
axiom g_coincides_with_f : coincides_on_nonneg f g

-- Define the proof required:
theorem inequalities_hold : 
  (f b - f (-a) > g a - g (-b)) ∧
  (f a - f (-b) > g b - g (-a)) :=
sorry

end inequalities_hold_l695_695253


namespace quadrilateral_has_incircle_l695_695081

-- Define the problem conditions in Lean
variable {A B C D O A1 B1 C1 D1 : Point}
variable (inscribed_quad : Quadrilateral A B C D)  -- A, B, C, D form an inscribed quadrilateral
variable (O_is_intersection : is_intersection O (Diagonal A C) (Diagonal B D))  -- O is the intersection of diagonals AC and BD
variable (A1_projection : is_projection A1 O (Line SEG AB))  -- A1 is the projection of O onto AB
variable (B1_projection : is_projection B1 O (Line SEG BC))  -- B1 is the projection of O onto BC
variable (C1_projection : is_projection C1 O (Line SEG CD))  -- C1 is the projection of O onto CD
variable (D1_projection : is_projection D1 O (Line SEG DA))  -- D1 is the projection of O onto DA
variable (no_extension : not_on_extension A1 A B ∧ not_on_extension B1 B C ∧ not_on_extension C1 C D ∧ not_on_extension D1 D A) -- The points do not lie on the extensions of sides

-- Formalize the statement we want to prove
theorem quadrilateral_has_incircle : has_incircle (Quadrilateral A1 B1 C1 D1) :=
sorry

end quadrilateral_has_incircle_l695_695081


namespace largest_square_area_with_four_interior_lattice_points_l695_695194

/-- A point (x, y) in the plane is called a lattice point if both x and y are integers. -/
def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (x, y)

/-- The square must have side length √(2) < s ≤ 2*√(2) to enclose exactly four lattice points
in its interior, where the largest such square has side length 2*√(2) yielding an area of 8. -/
theorem largest_square_area_with_four_interior_lattice_points :
  ∃ (s : ℝ), s = 2 * Real.sqrt 2 ∧ s^2 = 8 :=
by
  use 2 * Real.sqrt 2
  split
  { sorry }
  { sorry }

end largest_square_area_with_four_interior_lattice_points_l695_695194


namespace sum_of_lengths_PS_TV_eq_25_l695_695391

def number_line_segment_sum_length (P V : ℝ) (n : ℕ) (PS_parts TV_parts : ℕ) : ℝ := 
  let segment_length := V - P
  let part_length := segment_length / n
  let PS_length := PS_parts * part_length
  let TV_length := TV_parts * part_length
  PS_length + TV_length

theorem sum_of_lengths_PS_TV_eq_25 : 
  number_line_segment_sum_length 3 33 6 3 2 = 25 := 
by 
  sorry

end sum_of_lengths_PS_TV_eq_25_l695_695391


namespace probability_neither_square_nor_cube_l695_695538

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695538


namespace monotonic_increasing_interval_l695_695116

theorem monotonic_increasing_interval (a : ℝ) : 
  (∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x = x^2 - 2*a*x - 4*a → f y = y^2 - 2*a*y - 4*a → f x ≤ f y) ↔ (a ≤ 1) :=
sorry

end monotonic_increasing_interval_l695_695116


namespace probability_not_square_or_cube_l695_695555

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695555


namespace expected_wide_right_misses_l695_695261

-- Definitions for the problem conditions
def total_attempts_over_season : ℕ := 80
def miss_rate : ℚ := 1 / 3
def distribution_of_misses (misses : ℕ) : ℕ := misses / 4
def total_game_attempts : ℕ := 12
def increased_wind_speed : ℚ := 18 / 12 -- This is the ratio of increased wind speed to average wind speed
def under_40_yards_attempts : ℕ := 9
def wario_success_rate_under_40_yards : ℚ := 0.70
def waluigi_success_rate_under_40_yards : ℚ := 0.65

-- Assumption: distribution remains the same under increased wind
theorem expected_wide_right_misses : distribution_of_misses (total_game_attempts * miss_rate) = 1 := by
  sorry

end expected_wide_right_misses_l695_695261


namespace volume_of_prism_l695_695105

variables (a b c : ℝ)
variables (ab_prod : a * b = 36) (ac_prod : a * c = 48) (bc_prod : b * c = 72)

theorem volume_of_prism : a * b * c = 352.8 :=
by
  sorry

end volume_of_prism_l695_695105


namespace x_percent_more_than_y_l695_695838

theorem x_percent_more_than_y (z : ℝ) (hz : z ≠ 0) (y : ℝ) (x : ℝ)
  (h1 : y = 0.70 * z) (h2 : x = 0.84 * z) :
  x = y + 0.20 * y :=
by
  -- proof goes here
  sorry

end x_percent_more_than_y_l695_695838


namespace reach_any_point_l695_695074

theorem reach_any_point (infinite_line : ℕ → Prop)
  (stones_initial : ∀ n, stones_initial n → n = 1)
  (one_step_move : ∀ i, infinite_line i ∧ infinite_line (i+1) ∧ ¬ infinite_line (i+2) 
    → infinite_line (i+2)) : 
  ∀ n, infinite_line n := 
sorry

end reach_any_point_l695_695074


namespace roots_sum_reciprocal_squares_l695_695512

theorem roots_sum_reciprocal_squares (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + bc + ca = 20) (h3 : abc = 3) :
  (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2) = 328 / 9 := 
by
  sorry

end roots_sum_reciprocal_squares_l695_695512


namespace induction_sum_inequality_induction_step_l695_695143

theorem induction_sum_inequality (n : ℕ) : 
  (∑ i in Finset.range (2 * n + 1), 1 / (n + i + 1) : ℝ) >= (5 / 6) :=
sorry

theorem induction_step (k : ℕ) : 
  ((∑ i in Finset.range (2 * k + 3), 1 / (k + i + 2)) - 
   (∑ i in Finset.range (2 * k + 1), 1 / (k + i + 1))) = 
  (1 / (3 * k + 1) + 1 / (3 * k + 2) + 1 / (3 * k + 3) - 1 / (k + 1)) :=
sorry

end induction_sum_inequality_induction_step_l695_695143


namespace standard_equation_of_ellipse_find_slope_k_dot_product_constant_l695_695300

-- Define the conditions
variable (a b c : ℝ) (h_ab : a > b) (h_a_pos: a > 0) (h_b_pos: b > 0)
variable (h_len_major_minor : 2*a = sqrt(3) * 2*b)
variable (h_area_triangle : (1 / 2) * 2 * b * 2 * c = (5 * sqrt(2)) / 3)

-- Define the standard equation of the ellipse
def ellipse_C := ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Lean statement for Question 1
theorem standard_equation_of_ellipse : ellipse_C :=
by sorry

-- Define the line and intersection points A, B
variable (k : ℝ)
def line_eqn (x : ℝ) := k * (x + 1)

-- Define midpoint condition
variable (hx_mid : ∀ x1 x2 : ℝ, (x1 + x2) / 2 = -1 / 2)

-- Lean statement for Question 2(i)
theorem find_slope_k : k = sqrt(3) / 3 ∨ k = -sqrt(3) / 3 :=
by sorry

-- Define points and midpoint dot product condition
variable (x1 y1 x2 y2 : ℝ)
variable (M : ℝ × ℝ := (-7/3, 0))

-- Lean statement for Question 2(ii)
theorem dot_product_constant :
  let MA := (x1 + 7/3, y1)
  let MB := (x2 + 7/3, y2)
  (MA.1 * MB.1 + MA.2 * MB.2 = 4 / 9) :=
by sorry

end standard_equation_of_ellipse_find_slope_k_dot_product_constant_l695_695300


namespace hyperbola_equation_l695_695005

/-- The equation of a hyperbola given its foci and vertices. -/
theorem hyperbola_equation (a b c : ℝ) :
  let ellipse_vertices : ℝ := 4
  let ellipse_foci : ℝ := sqrt 7
  let hyperbola_vertices : ℝ := sqrt 7
  let hyperbola_foci : ℝ := 4
  let a_squared : ℝ := a^2
  let b_squared : ℝ := b^2
  let c_squared : ℝ := c^2
  in 
  (c = 4) ∧ (a = sqrt 7) ∧ (c^2 = a^2 + b^2) ∧ (b = 3) → (a_squared = 7) ∧ (b_squared = 9) → (eq (1) (1)) :=
begin
  intros,
  sorry
end

end hyperbola_equation_l695_695005


namespace next_sales_amount_l695_695192

theorem next_sales_amount
  (royalties1: ℝ)
  (sales1: ℝ)
  (royalties2: ℝ)
  (percentage_decrease: ℝ)
  (X: ℝ)
  (h1: royalties1 = 4)
  (h2: sales1 = 20)
  (h3: royalties2 = 9)
  (h4: percentage_decrease = 58.333333333333336 / 100)
  (h5: royalties2 / X = royalties1 / sales1 - ((royalties1 / sales1) * percentage_decrease)): 
  X = 108 := 
  by 
    -- Proof omitted
    sorry

end next_sales_amount_l695_695192


namespace area_of_triangle_ABC_l695_695363

-- Define points and triangle ABC
variables (A B C E : Type) [real_inner_product_space ℝ A] [real_inner_product_space ℝ B] [real_inner_product_space ℝ C] [real_inner_product_space ℝ E]


-- Define the conditions
variables (CE : ℝ) (hCE : CE = 2)
variables (angleBAC : real.angle) (hangleBAC : angleBAC.degrees = 30)

-- Prove that the area of ΔABC is 8√3/3 square centimeters
theorem area_of_triangle_ABC 
  (hCE : CE = 2)
  (hangleBAC : angleBAC.degrees = 30) :
  let AC := 4 in     -- Using the property of 30-60-90 triangle ACE with CE = 2
  let BC := (4 / real.sqrt 3) in
  (1 / 2) * AC * BC = (8 * real.sqrt 3) / 3 :=
sorry

end area_of_triangle_ABC_l695_695363


namespace simplify_and_rationalize_l695_695466

theorem simplify_and_rationalize :
  ( (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) *
    (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 ) :=
by
  sorry

end simplify_and_rationalize_l695_695466


namespace infinite_odd_even_V_l695_695280

noncomputable def sqrt2020 : ℝ := real.sqrt 2020
noncomputable def sqrt2021 : ℝ := real.sqrt 2021

def V (n : ℕ) : ℤ := int.floor (2^n * sqrt2020) + int.floor (2^n * sqrt2021)

theorem infinite_odd_even_V :
  ∃ (odd_indices : ℕ → Prop) (even_indices : ℕ → Prop),
  (∀ n, odd_indices n ∨ even_indices n) ∧
    infinite {n | odd_indices n} ∧
    infinite {n | even_indices n} :=
sorry

end infinite_odd_even_V_l695_695280


namespace shipping_cost_per_unit_l695_695186

-- Define the conditions
def cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def num_components : ℝ := 150
def lowest_selling_price : ℝ := 196.67

-- Define the revenue and total cost
def total_cost (S : ℝ) : ℝ := (cost_per_component * num_components) + fixed_monthly_cost + (num_components * S)
def total_revenue : ℝ := lowest_selling_price * num_components

-- Define the proposition to be proved
theorem shipping_cost_per_unit (S : ℝ) :
  total_cost S ≤ total_revenue → S ≤ 6.67 :=
by sorry

end shipping_cost_per_unit_l695_695186


namespace find_smallest_angle_l695_695735

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l695_695735


namespace k_h_neg3_eq_l695_695050

noncomputable def h (x : ℝ) : ℝ := 5 * x^2 - 9

-- Given condition
axiom k_h3_eq : k (h 3) = 15

-- Theorem to prove
theorem k_h_neg3_eq : k (h (-3)) = 15 := by
  sorry

end k_h_neg3_eq_l695_695050


namespace find_general_formula_find_sum_formula_l695_695299

-- Defining the properties for the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (S 12 = 78) ∧ (a 8 = 4 * a 2)

-- Proving the general formula for {a_n}
theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : arithmetic_sequence a S) : ∀ n, a n = n := 
by sorry

-- Defining the properties for the sequence {b_n} and its sum T_n
def geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) :=
  ∀ n, b n = (a n) / (3^n)

-- Proving the sum of the first n terms of {b_n}
theorem find_sum_formula (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h1 : arithmetic_sequence a S)
  (h2 : geometric_sequence b T) :
  ∀ n, T n = (3^(n+1) - 2*n - 3) / (4 * 3^n) := 
by sorry

end find_general_formula_find_sum_formula_l695_695299


namespace automobile_finance_companies_credit_l695_695684

noncomputable def auto_credit_share : ℝ := 0.36
noncomputable def total_consumer_credit : ℝ := 291.6666666666667
noncomputable def finance_company_share : ℝ := 1 / 3
noncomputable def finance_company_credit : ℝ := finance_company_share * (auto_credit_share * total_consumer_credit)

theorem automobile_finance_companies_credit : finance_company_credit ≈ 35 := by
  sorry

end automobile_finance_companies_credit_l695_695684


namespace pizza_boxes_sold_l695_695263

variables (P : ℕ) -- Representing the number of pizza boxes sold

def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2

def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def goal_amount : ℝ := 500
def more_needed : ℝ := 258
def current_amount : ℝ := goal_amount - more_needed

-- Total earnings calculation
def total_earnings : ℝ := (P : ℝ) * pizza_price + fries_sold * fries_price + soda_sold * soda_price

theorem pizza_boxes_sold (h : total_earnings P = current_amount) : P = 15 := 
by
  sorry

end pizza_boxes_sold_l695_695263


namespace sin_cos_identity_l695_695919

theorem sin_cos_identity (n : ℕ) (h : n ≥ 1) (a : Fin n → ℤ) 
  (h_a : ∀ i, a i = 1 ∨ a i = -1) :
  2 * sin ((a 0) + (∑ i in Finset.range (n-1), (a 0) * ∏ j in Finset.range (i+1), (a (Fin (j+1) n))) / 2 ^ i) * (π / 4) 
  = (a 0) * sqrt (2 + (∑ i in Finset.range (n-1), a (i+1) * sqrt (2 + ∏ j in Finset.range (i+1), sqrt 2))) := 
sorry

end sin_cos_identity_l695_695919


namespace eighth_term_sum_of_first_15_terms_l695_695960

-- Given definitions from the conditions
def a1 : ℚ := 5
def a30 : ℚ := 100
def n8 : ℕ := 8
def n15 : ℕ := 15
def n30 : ℕ := 30

-- Formulate the arithmetic sequence properties
def common_difference : ℚ := (a30 - a1) / (n30 - 1)

def nth_term (n : ℕ) : ℚ :=
  a1 + (n - 1) * common_difference

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * common_difference)

-- Statements to be proven
theorem eighth_term :
  nth_term n8 = 25 + 1/29 := by sorry

theorem sum_of_first_15_terms :
  sum_of_first_n_terms n15 = 393 + 2/29 := by sorry

end eighth_term_sum_of_first_15_terms_l695_695960


namespace probability_not_square_or_cube_l695_695556

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695556


namespace simplify_root_power_l695_695479

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695479


namespace domain_of_f_l695_695257

open Set Real

def f (x : ℝ) : ℝ := (1 / (1 - x)) + log (x + 1)

theorem domain_of_f :
  (∀ x : ℝ, (f x ∈ ℝ) ↔ ((-1 < x ∧ x < 1) ∨ (1 < x))) :=
by sorry

end domain_of_f_l695_695257


namespace x_is_integer_l695_695982

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ k1 : ℤ, x^2 - x = k1)
  (h2 : ∃ (n : ℕ) (_ : n > 2) (k2 : ℤ), x^n - x = k2) : 
  ∃ (m : ℤ), x = m := 
sorry

end x_is_integer_l695_695982


namespace area_of_triangle_find_angle_C_l695_695025

-- Definition of the problem conditions
variable {A B C : Type}
variable (a b c : ℝ)
variable (cos_B : ℝ) (dot_AB_BC : ℝ)
variable (triangle_ABC : Δ A B C)

-- Specific problem conditions
def conditions := cos_B = 3 / 5 ∧ dot_AB_BC = -21

-- Statements to be proven in Lean 4 (no proof provided)
theorem area_of_triangle (h : conditions) : 
  area_of_triangle triangle_ABC = 14 :=
sorry

theorem find_angle_C (h : conditions) (a_eq_seven : a = 7) : 
  angle_C triangle_ABC = π / 4 :=
sorry

end area_of_triangle_find_angle_C_l695_695025


namespace proof_inscribed_sphere_area_cross_section_l695_695598

-- Definitions derived from the conditions
variables (R : ℝ) (α : ℝ)
def inscribed_sphere_area_cross_section : ℝ :=
  (π * (R * (Real.tan (α / 2)) * (Real.sin (2 * α)))^2)

-- The theorem to prove the equivalency
theorem proof_inscribed_sphere_area_cross_section :
  inscribed_sphere_area_cross_section R α = π * R^2 * (Real.tan (α / 2))^2 * (Real.sin (2 * α))^2 :=
sorry

end proof_inscribed_sphere_area_cross_section_l695_695598


namespace probability_neither_square_nor_cube_l695_695535

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l695_695535


namespace dot_product_eq_negative_12_sqrt_3_l695_695784

variables {a b : ℝ}

-- Given conditions
def magnitude_a := 6
def magnitude_b := 4
def angle := real.pi * (5 / 6) -- 150 degrees in radians

-- Prove the dot product is -12√3
theorem dot_product_eq_negative_12_sqrt_3
  (ha : ∥a∥ = magnitude_a)
  (hb : ∥b∥ = magnitude_b)
  (angle_eq : ∠a b = angle) : 
  a • b = -12 * real.sqrt 3 :=
sorry

end dot_product_eq_negative_12_sqrt_3_l695_695784


namespace dave_winfield_home_runs_correct_l695_695215

def dave_winfield_home_runs (W : ℕ) : Prop :=
  755 = 2 * W - 175

theorem dave_winfield_home_runs_correct : dave_winfield_home_runs 465 :=
by
  -- The proof is omitted as requested
  sorry

end dave_winfield_home_runs_correct_l695_695215


namespace percentage_of_other_items_l695_695496

theorem percentage_of_other_items (notebooks markers total : ℝ) (h_notebooks : notebooks = 42) (h_markers : markers = 21) (h_total : total = 100) : 
    total - (notebooks + markers) = 37 :=
by
  -- Given conditions
  have h1 : notebooks = 42 := h_notebooks
  have h2 : markers = 21 := h_markers
  have h3 : total = 100 := h_total

  -- Calculating the percentage of other items
  have h4 : total - (notebooks + markers) = 100 - (42 + 21) := 
    by sorry
  have h5 : 100 - (42 + 21) = 100 - 63 := by sorry
  have h6 : 100 - 63 = 37 := by sorry

  -- Conclusion
  exact (Eq.trans (Eq.trans h4 h5) h6)

end percentage_of_other_items_l695_695496


namespace probability_neither_square_nor_cube_l695_695545

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l695_695545


namespace boat_travel_time_l695_695652

theorem boat_travel_time (x : ℝ) (T : ℝ) (h0 : 0 ≤ x) (h1 : x ≠ 15.6) 
    (h2 : 96 = (15.6 - x) * T) 
    (h3 : 96 = (15.6 + x) * 5) : 
    T = 8 :=
by 
  sorry

end boat_travel_time_l695_695652


namespace triangle_area_l695_695593

theorem triangle_area (p r : ℝ) (hp : p = 32) (hr : r = 2.5) : 
  (r * (p / 2)) = 40 :=
by 
  -- Given conditions
  have hp_div2 : p / 2 = 16, by linarith [hp],
  have hr_val : r = 2.5, by exact hr,
  -- Calculate area using the formula
  have area : r * (p / 2) = 2.5 * 16, by rw [hr_val, hp_div2],
  -- Prove area equals 40
  linarith [area]

end triangle_area_l695_695593


namespace simplify_root_power_l695_695477

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695477


namespace number_of_groups_l695_695170

-- Define constants
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6

-- Define the theorem to be proven
theorem number_of_groups :
  (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end number_of_groups_l695_695170


namespace linear_function_expression_point_P_on_graph_point_Q_on_graph_l695_695317

theorem linear_function_expression (k b : ℝ) (h₀ : k ≠ 0) 
  (hA : 2 = k * 1 + b) (hB : 4 = k * (-1) + b) : 
  (∀ x, (x, -x + 3) = (1, 2) ∨ (x, 2)) :=
by sorry

theorem point_P_on_graph (k b : ℝ) (h₀ : k ≠ 0) 
  (hA : 2 = k * 1 + b) (hB : 4 = k * (-1) + b) (xP : 2) (yP : 3) :
  yP ≠ k * xP + b :=
by sorry

theorem point_Q_on_graph (k b : ℝ) (h₀ : k ≠ 0) 
  (hA : 2 = k * 1 + b) (hB : 4 = k * (-1) + b) (xQ : 3) (yQ : 0) :
  yQ = k * xQ + b :=
by sorry

end linear_function_expression_point_P_on_graph_point_Q_on_graph_l695_695317


namespace probability_of_answering_second_question_l695_695824

theorem probability_of_answering_second_question
  (P_A : ℝ = 0.63)
  (P_A_and_B : ℝ = 0.32)
  (P_neither : ℝ = 0.20) :
  let P_B := 0.49
  P_B = 1 - P_neither - P_A + P_A_and_B :=
begin
  sorry
end

end probability_of_answering_second_question_l695_695824


namespace weighted_asymptotes_sum_l695_695700

noncomputable def function := λ x : ℝ, (x^2 - 4*x + 4) / (x^3 - 2*x^2 - x + 2)

theorem weighted_asymptotes_sum : 
  let a := 1 in 
  let b := 2 in 
  let c := 1 in 
  let d := 0 in 
  a + 2 * b + 3 * c + 4 * d = 8 := 
by
  -- proof will be here
  sorry

end weighted_asymptotes_sum_l695_695700


namespace molar_weight_of_BaF2_l695_695626

theorem molar_weight_of_BaF2 (Ba_weight : Real) (F_weight : Real) (num_moles : ℕ) 
    (Ba_weight_val : Ba_weight = 137.33) (F_weight_val : F_weight = 18.998) 
    (num_moles_val : num_moles = 6) 
    : (137.33 + 2 * 18.998) * 6 = 1051.956 := 
by
  sorry

end molar_weight_of_BaF2_l695_695626


namespace sequence_constant_iff_perfect_square_l695_695753

noncomputable def S (n : ℕ) : ℕ :=
  let m := Nat.floor (Real.sqrt n)
  n - m^2

def seq (a : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => seq n + S (seq n)

theorem sequence_constant_iff_perfect_square (A : ℕ) :
  (∃ c : ℕ, ∀ n : ℕ, seq A n = c) ↔ (∃ k : ℕ, A = k^2) :=
sorry

end sequence_constant_iff_perfect_square_l695_695753


namespace triangle_area_proof_l695_695182

noncomputable def segment_squared (a b : ℝ) : ℝ := a ^ 2 - b ^ 2

noncomputable def triangle_conditions (a b c : ℝ): Prop :=
  segment_squared b a = a ^ 2 - c ^ 2

noncomputable def area_triangle_OLK (r a b c : ℝ) (cond : triangle_conditions a b c): ℝ :=
  (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3))

theorem triangle_area_proof (r a b c : ℝ) (cond : triangle_conditions a b c) :
  area_triangle_OLK r a b c cond = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3)) :=
sorry

end triangle_area_proof_l695_695182


namespace probability_neither_square_nor_cube_l695_695574

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l695_695574


namespace number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l695_695136

theorem number_of_apples (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (apples_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    apples_mult = 5 → 
    (apples_mult * peaches_fraction * oranges_fraction * total_fruit) = 35 :=
by
  intros h1 h2 h3
  sorry

theorem ratio_of_mixed_fruits (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (mixed_fruits_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    mixed_fruits_mult = 2 → 
    (mixed_fruits_mult * peaches_fraction * oranges_fraction * total_fruit) / total_fruit = 1/4 :=
by
  intros h1 h2 h3
  sorry

theorem total_weight_of_oranges (total_fruit : ℕ) (oranges_fraction : ℚ) (orange_weight : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    orange_weight = 200 → 
    (orange_weight * oranges_fraction * total_fruit) = 2800 :=
by
  intros h1 h2
  sorry

end number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l695_695136


namespace probability_two_late_one_on_time_l695_695845

noncomputable def p_late : ℚ := 1 / 40
noncomputable def p_on_time : ℚ := 1 - p_late

theorem probability_two_late_one_on_time :
  let p_two_late := (p_late * p_late * p_on_time)
  let total_probability := 3 * p_two_late
  (total_probability * 100).round() = 0.2 :=
by
  sorry

end probability_two_late_one_on_time_l695_695845


namespace polar_to_rectangular_l695_695247

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 3) :
    (r * Real.cos θ, r * Real.sin θ) = (2, 2 * Real.sqrt 3) :=
by
  rw [h_r, h_θ]
  norm_num
  rw [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  exact ⟨rfl, rfl⟩

end polar_to_rectangular_l695_695247


namespace barrel_contents_lost_l695_695174

theorem barrel_contents_lost (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 220) 
  (h2 : remaining_amount = 198) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 10 :=
by
  rw [h1, h2]
  sorry

end barrel_contents_lost_l695_695174


namespace spadesuit_value_l695_695255

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l695_695255


namespace arithmetic_progression_numbers_l695_695944

theorem arithmetic_progression_numbers :
  ∃ (a d : ℚ), (3 * (2 * a - d) = 2 * (a + d)) ∧ ((a - d) * (a + d) = (a - 2)^2) ∧
  ((a = 5 ∧ d = 4 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 ∧ c = 9) 
   ∨ (a = 5 / 4 ∧ d = 1 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 / 4 ∧ c = 9 / 4)) :=
by
  sorry

end arithmetic_progression_numbers_l695_695944


namespace polygon_area_l695_695858

theorem polygon_area (sides : ℕ) (perpendicular_adjacent : Bool) (congruent_sides : Bool) (perimeter : ℝ) (area : ℝ) :
  sides = 32 → 
  perpendicular_adjacent = true → 
  congruent_sides = true →
  perimeter = 64 →
  area = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end polygon_area_l695_695858


namespace how_many_women_left_l695_695398

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l695_695398


namespace simplify_root_power_l695_695476

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l695_695476


namespace sum_of_intervals_l695_695892

-- Define the given function f
def f (x : ℝ) := ⌊x⌋ * (2013 ^ (x - ⌊x⌋) - 1)

-- Define the main theorem to prove
theorem sum_of_intervals (h : ∀ x, 1 ≤ x ∧ x < 2013 → f x ≤ 1) :
  (∑ k in (finset.range (2012 + 1)).erase 0, log 2013 ((↑k + 1 : ℝ) / k)) = 1 :=
sorry

end sum_of_intervals_l695_695892


namespace cube_sum_gt_l695_695875

variable (a b c d : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
variable (h1 : a + b = c + d)
variable (h2 : a^2 + b^2 > c^2 + d^2)

theorem cube_sum_gt : a^3 + b^3 > c^3 + d^3 := by
  sorry

end cube_sum_gt_l695_695875


namespace calculate_selling_price_l695_695923

-- Definitions of the conditions
def cost_price : ℝ := 4500
def loss_percentage : ℝ := 28.888888888888886

-- Calculation of the selling price based on the conditions
def loss_amount : ℝ := (loss_percentage / 100) * cost_price
def selling_price : ℝ := cost_price - loss_amount

-- Statement of the proof problem
theorem calculate_selling_price : selling_price = 3200 := 
by
  -- You should provide an actual proof here
  sorry

end calculate_selling_price_l695_695923


namespace x_eq_1_sufficient_but_not_necessary_l695_695783

theorem x_eq_1_sufficient_but_not_necessary (x : ℝ) : x^2 - 3 * x + 2 = 0 → (x = 1 ↔ true) ∧ (x ≠ 1 → ∃ y : ℝ, y ≠ x ∧ y^2 - 3 * y + 2 = 0) :=
by
  sorry

end x_eq_1_sufficient_but_not_necessary_l695_695783


namespace events_complementary_l695_695137

def event_A (n : ℕ) : Prop := n ≤ 3
def event_B (n : ℕ) : Prop := n ≥ 4
def valid_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem events_complementary :
  (∀ n, n ∈ valid_outcomes → event_A n ∨ event_B n) ∧ (∀ n, n ∈ valid_outcomes → ¬(event_A n ∧ event_B n)) :=
by
  sorry

end events_complementary_l695_695137


namespace determine_digits_l695_695127

def product_consecutive_eq_120_times_ABABAB (n A B : ℕ) : Prop :=
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 * (A * 101010101 + B * 10101010 + A * 1010101 + B * 101010 + A * 10101 + B * 1010 + A * 101 + B * 10 + A)

theorem determine_digits (A B : ℕ) (h : ∃ n, product_consecutive_eq_120_times_ABABAB n A B):
  A = 5 ∧ B = 7 :=
sorry

end determine_digits_l695_695127


namespace weight_of_new_person_l695_695108

theorem weight_of_new_person
  (average_increase : ℝ)
  (old_person_weight : ℝ) 
  (new_persons_weight : ℝ)
  (num_persons : ℕ)
  (h1 : num_persons = 7)
  (h2 : average_increase = 3.5)
  (h3 : old_person_weight = 75) :
  new_persons_weight = old_person_weight + average_increase * ↑ num_persons :=
by
  sorry

def main : IO Unit :=
  IO.println s!"The weight of the new person might be {weight_of_new_person 3.5 75 99.5 7 sorry sorry sorry} kg"

end weight_of_new_person_l695_695108


namespace probability_not_square_or_cube_l695_695550

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l695_695550


namespace max_folded_triangle_area_l695_695205

open Real

theorem max_folded_triangle_area :
  let a := (3: ℝ)/2
  let b := real.sqrt 5 / 2
  let c := real.sqrt 2
  let s := (a + b + c) / 2
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := 2 * area / a
  ∃ (max_area: ℝ), max_area = 9 / 28 :=
by
  let a := (3: ℝ)/2
  let b := real.sqrt 5 / 2
  let c := real.sqrt 2
  let s := (a + b + c) / 2
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := 2 * area / a
  have coincident_area : ℝ := area
  existsi (9: ℝ)/28
  sorry

end max_folded_triangle_area_l695_695205


namespace coeff_of_x_squared_in_binomial_expansion_l695_695370

theorem coeff_of_x_squared_in_binomial_expansion :
  (∃ n : ℕ, ((∑ i in range (n + 1), (2:ℚ) ^ (n - i) * (1 / 3:ℚ) ^ i * choose n i) = 729)) →
  n = 6 →
  (coeff (expand (2 * x + 1 / (3 * x)) 6) 2 = 160) :=
by
  sorry

end coeff_of_x_squared_in_binomial_expansion_l695_695370


namespace probability_neither_square_nor_cube_l695_695563

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l695_695563


namespace probability_of_red_or_green_l695_695627

theorem probability_of_red_or_green :
  let red := 5;
      blue := 2;
      yellow := 3;
      green := 4;
      total := red + blue + yellow + green
  in total = 14 ∧ (red + green) / total = 9 / 14 :=
by
  let red := 5;
  let blue := 2;
  let yellow := 3;
  let green := 4;
  let total := red + blue + yellow + green;
  have h1: total = 14, by
  {
    sorry
  },
  have h2: (red + green) / total = 9 / 14, by
  {
    sorry
  },
  exact ⟨h1, h2⟩

end probability_of_red_or_green_l695_695627


namespace percent_equivalence_l695_695153

theorem percent_equivalence (x : ℝ) : (0.6 * 0.3 * x - 0.1 * x) / x * 100 = 8 := by
  sorry

end percent_equivalence_l695_695153


namespace y_coordinate_of_equidistant_point_on_y_axis_l695_695959

theorem y_coordinate_of_equidistant_point_on_y_axis :
  (∃ y : ℝ, sqrt ((-3 - 0)^2 + (1 - y)^2) = sqrt ((-2 - 0)^2 + (5 - y)^2) ∧ y = 19 / 8) :=
sorry

end y_coordinate_of_equidistant_point_on_y_axis_l695_695959


namespace total_spent_after_three_years_l695_695894

def iPhone_cost : ℝ := 1000
def contract_cost_per_month : ℝ := 200
def case_cost_before_discount : ℝ := 0.20 * iPhone_cost
def headphones_cost_before_discount : ℝ := 0.5 * case_cost_before_discount
def charger_cost : ℝ := 60
def warranty_cost_for_two_years : ℝ := 150
def discount_rate : ℝ := 0.10
def time_in_years : ℝ := 3

def contract_cost_for_three_years := contract_cost_per_month * 12 * time_in_years
def case_cost_after_discount := case_cost_before_discount * (1 - discount_rate)
def headphones_cost_after_discount := headphones_cost_before_discount * (1 - discount_rate)

def total_cost : ℝ :=
  iPhone_cost +
  contract_cost_for_three_years +
  case_cost_after_discount +
  headphones_cost_after_discount +
  charger_cost +
  warranty_cost_for_two_years

theorem total_spent_after_three_years : total_cost = 8680 :=
  by
    sorry

end total_spent_after_three_years_l695_695894


namespace irreducible_polynomial_l695_695056

theorem irreducible_polynomial 
  {n : ℕ} 
  {a : Fin (n+1) → ℤ} 
  (P : ℤ) 
  (hp : Prime P) 
  (ha0 : a 0 ≠ 0) 
  (han : a n ≠ 0) 
  (hP : P > Finset.sum (Finset.range n) (λ i, |a i| * |a n|^i)) : 
  irreducible (λ x : ℤ, a 0 * x^n + a 1 * x^(n-1) + ∙∙ + a (n-1) * x + P * a n) :=
sorry

end irreducible_polynomial_l695_695056


namespace length_increase_50x_l695_695452

theorem length_increase_50x (x : ℕ) (n : ℕ) : 
  ((n + 3) / 3) * x = 50 * x → n = 147 := 
by
  assume h : ((n + 3) / 3) * x = 50 * x
  sorry

end length_increase_50x_l695_695452


namespace total_payment_l695_695069

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l695_695069


namespace abs_x_lt_2_sufficient_but_not_necessary_l695_695642

theorem abs_x_lt_2_sufficient_but_not_necessary (x : ℝ) :
  (|x| < 2) → (x ^ 2 - x - 6 < 0) ∧ ¬ ((x ^ 2 - x - 6 < 0) → (|x| < 2)) := by
  sorry

end abs_x_lt_2_sufficient_but_not_necessary_l695_695642


namespace latus_rectum_parabola_l695_695723

theorem latus_rectum_parabola : 
  ∀ (x y : ℝ), (x = 4 * y^2) → (x = -1/16) :=
by 
  sorry

end latus_rectum_parabola_l695_695723


namespace system_solution_l695_695488

theorem system_solution (a x y z : ℂ) (k l : ℤ) :
  (| a + 1 / a | ≥ 2) ∧ (| a | = 1) ∧ (sin y = 1 ∨ sin y = -1) ∧ (cos z = 0) →
  (x = π / 2 + k * π) ∧ (y = π / 2 + k * π) ∧ (z = π / 2 + l * π) :=
by
  sorry

end system_solution_l695_695488


namespace gcd_pow_minus_one_l695_695625

theorem gcd_pow_minus_one (n m : ℕ) (hn : n = 1030) (hm : m = 1040) :
  Nat.gcd (2^n - 1) (2^m - 1) = 1023 := 
by
  sorry

end gcd_pow_minus_one_l695_695625


namespace probability_between_C_and_E_l695_695079

theorem probability_between_C_and_E
  (AB AD BC BE : ℝ)
  (h₁ : AB = 4 * AD)
  (h₂ : AB = 8 * BC)
  (h₃ : AB = 2 * BE) : 
  (AB / 2 - AB / 8) / AB = 3 / 8 :=
by 
  sorry

end probability_between_C_and_E_l695_695079


namespace cheese_mouse_distance_l695_695193

-- Define the coordinates of the cheese and the mouse's line
def cheese : ℝ × ℝ := (13, 8)
def mouse_line (x : ℝ) : ℝ := -4 * x + 9

-- Define the conditions and hypothesis for the problem
theorem cheese_mouse_distance :
  ∃ (a b : ℝ), b = mouse_line a ∧
   a = 1 / 17 ∧ b = 149 / 17 ∧ (a + b = 150 / 17) :=
by {
  existsi (1 / 17),
  existsi (149 / 17),
  split,
  { simp [mouse_line],
    field_simp,
    linarith },
  split,
  { refl },
  split,
  { refl },
  { linarith [show (1 : ℝ) / 17 + 149 / 17 = 150 / 17, by { field_simp, linarith }] }
}

end cheese_mouse_distance_l695_695193


namespace is_necessary_and_sufficient_condition_l695_695393

theorem is_necessary_and_sufficient_condition (A B C : ℝ) (a b c : ℝ)
  (hA : 0 < A < π) (hB : 0 < B < π) (hC : 0 < C < π)
  (h_sum : A + B + C = π)
  (h1 : a / sin A = b / sin B) :
  (a > b) ↔ (sin A > sin B) :=
by
  sorry

end is_necessary_and_sufficient_condition_l695_695393


namespace find_f_neg3_l695_695045

variable (f : ℝ → ℝ)

-- Definitions and conditions
def even_function : Prop := ∀ x, f (-x) = f x

def condition_for_f (x : ℝ) : Prop := ∀ x > 0, f (2 + x) = -2 * f (2 - x)

-- Specific value given
def specific_value : Prop := f (-1) = 4

-- Goal
theorem find_f_neg3 (h_even : even_function f) (h_cond : condition_for_f f) (h_val : specific_value f) : f (-3) = -8 := by
  sorry

end find_f_neg3_l695_695045


namespace circumscribed_circle_diameter_l695_695829

noncomputable def diameter_of_circumscribed_circle 
  (b : ℝ) (B : ℝ) (sin_B : ℝ) : ℝ :=
b / sin_B

theorem circumscribed_circle_diameter 
  (b : ℝ) (B : ℂ) (hb : b = 15) (hB : B = 45) :
  diameter_of_circumscribed_circle b (Real.sin (Real.pi * B / 180)) = 15 * Real.sqrt 2 :=
by
  have hsin45 : Real.sin (Real.pi * 45 / 180) = Real.sqrt 2 / 2 := sorry
  rw [hb, hsin45]
  field_simp
  rw [← mul_assoc, ← Real.sqrt_mul, mul_div_cancel, mul_comm]
  hint


end circumscribed_circle_diameter_l695_695829


namespace cost_of_goods_l695_695945

theorem cost_of_goods
  (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 315)
  (h2 : 4 * x + 10 * y + z = 420) :
  x + y + z = 105 :=
by
  sorry

end cost_of_goods_l695_695945


namespace find_m_polynomial_simplifies_no_x2_l695_695338

def simplify_polynomial (m : ℝ) : ℝ :=
(2 * m * (x^2) + 4 * (x^2) + 3 * x + 1) - (6 * (x^2) - 4 * (y^2) + 3 * x)

theorem find_m_polynomial_simplifies_no_x2 (m : ℝ) : ∃ m, (2 * m - 2) * x^2 + 4 * (y^2) + 1 = 4 * (y^2) + 1 :=
begin
  use 1,
  sorry
end

end find_m_polynomial_simplifies_no_x2_l695_695338


namespace june_bernard_travel_time_l695_695870

-- Defining the given conditions
def distance_june_julia : ℝ := 2
def time_june_julia : ℝ := 6
def distance_june_bernard : ℝ := 5

-- Rate calculation premise
def travel_rate : ℝ := distance_june_julia / time_june_julia

-- Proof problem statement
theorem june_bernard_travel_time :
  travel_rate * distance_june_bernard = 15 := sorry

end june_bernard_travel_time_l695_695870


namespace imaginary_part_is_neg_one_l695_695931

noncomputable def complex_expression : ℂ := (1 - complex.I)^2 / (1 + complex.I)

-- Define the theorem to state that the imaginary part of the complex_expression is -1.
theorem imaginary_part_is_neg_one : complex.im complex_expression = -1 :=
sorry

end imaginary_part_is_neg_one_l695_695931


namespace simplify_sqrt_expression_l695_695090

theorem simplify_sqrt_expression :
  (Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)) = 6 :=
by
  sorry

end simplify_sqrt_expression_l695_695090


namespace arrangement_11250_multiple_of_5_l695_695849

theorem arrangement_11250_multiple_of_5 : 
  ∃ (n : ℕ), n = 69 ∧ 
  (∑ d in {d | (d ∈ {0, 1, 1, 2, 5})} ∧ (d ≠ 0) ∧ (d ≠ 5), true) = (card {p | permutation {1, 1, 2, 5, 0} p ∧ (p.last = 0 ∨ p.last = 5)})) :=
begin
  use 69,
  split,
  { reflexivity, },
  { sorry, }, -- proof goes here
end

end arrangement_11250_multiple_of_5_l695_695849


namespace length_of_XY_l695_695059

theorem length_of_XY : 
  (∀ (A B C D E X Y : Type) 
    (AB BC CA : ℝ) 
    (angle_A angle_B angle_C : ℝ)
    (is_right_triangle : angle_C = 90 ∧ AB = 7 ∧ BC = 1 ∧ CA = 4 * Real.sqrt 3) 
    (at_D_E : ∃ D E, trisectors_of_C AD BE) 
    (X_Y_circumcircle_intersect : (AC ∩ circumcircle CDE = X) ∧ (BC ∩ circumcircle CDE = Y)),
    length XY = 112 / 65 )
:= sorry

end length_of_XY_l695_695059


namespace smallest_positive_period_l695_695258

-- Define the function
def f (x : Real) : Real := Real.sin (2 * x + Real.pi / 3)

-- Define the period of a function
def period (f : Real → Real) (T : Real) : Prop := ∀ x, f (x + T) = f x

-- State the theorem
theorem smallest_positive_period : ∃ T > 0, period f T ∧ ∀ T', period f T' → T' ≥ T :=
by
  exists π
  split
  exact Real.pi_pos
  split
  { intro x
    sorry },  -- Proof for periodicity
  { intros T' hT'
    sorry }  -- Proof that T is the smallest period

end smallest_positive_period_l695_695258


namespace simplify_sqrt3_7_pow6_l695_695470

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l695_695470


namespace billion_scientific_notation_l695_695939

theorem billion_scientific_notation (x : ℝ) (hbillion : x = 27.58 * 10^9) :
  x = 2.758 * 10^10 :=
by {
  rw [hbillion],
  have : 27.58 = 2.758 * 10 := by norm_num,
  rw [this, mul_assoc],
  congr,
  norm_num,
  sorry,
}

end billion_scientific_notation_l695_695939


namespace staircase_steps_l695_695031

theorem staircase_steps (x : ℕ) (h1 : x + 2 * x + (2 * x - 10) = 2 * 45) : x = 20 :=
by 
  -- The proof is skipped
  sorry

end staircase_steps_l695_695031


namespace hyperbola_eccentricity_l695_695334

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x, y = - (frac √5 2) x) (h_eqn : ∀ x y, ((x^2) / (a^2)) - (y^2 / (b^2)) = 1) :
  let e : ℝ := (sqrt (a^2 + b^2)) / a 
  in e = 3 / 2 :=
sorry

end hyperbola_eccentricity_l695_695334


namespace trigonometric_identity_problem_l695_695424

theorem trigonometric_identity_problem : 
  ∀ (A B C : ℝ) (p q r s : ℕ),
  (C > 90) ∧ (* C is obtuse *)
  (cos A^2 + cos B^2 + 2 * sin A * sin B * cos C = 21 / 11) ∧
  (cos A^2 + cos C^2 + 2 * sin A * sin C * cos B = 17 / 10) ∧
  (cos B^2 + cos C^2 + 2 * sin B * sin C * cos A = (p - q * real.sqrt r) / s) ∧
  (nat.gcd (p + q) s = 1) ∧
  (r % (n^2) ≠ 0 ∀ n > 1) -> 
  p + q + r + s = 349 := 
by
  sorry

end trigonometric_identity_problem_l695_695424


namespace gift_weight_l695_695679

-- Given conditions
def small_bag_weight : ℕ := 220
def num_gifts_per_small_bag : ℕ := 6
def large_bag_weight : ℕ := 250
def num_small_bags_in_large_bag : ℕ := 9
def total_weight_kg : ℕ := 13.3 * 1000  -- Convert kg to grams

-- Calculate the total weight of gifts
theorem gift_weight :
  let total_weight_g := total_weight_kg
  let total_small_bags_weight := num_small_bags_in_large_bag * small_bag_weight
  let total_bags_weight := total_small_bags_weight + large_bag_weight
  let weight_of_gifts := total_weight_g - total_bags_weight
  let total_num_gifts := num_gifts_per_small_bag * num_small_bags_in_large_bag
  weight_of_gifts / total_num_gifts = 205 :=
by
  sorry

end gift_weight_l695_695679


namespace jorge_land_fraction_clay_rich_soil_l695_695034

theorem jorge_land_fraction_clay_rich_soil 
  (total_acres : ℕ) 
  (yield_good_soil_per_acre : ℕ) 
  (yield_clay_soil_factor : ℕ)
  (total_yield : ℕ) 
  (fraction_clay_rich_soil : ℚ) :
  total_acres = 60 →
  yield_good_soil_per_acre = 400 →
  yield_clay_soil_factor = 2 →
  total_yield = 20000 →
  fraction_clay_rich_soil = 1/3 :=
by
  intro h_total_acres h_yield_good_soil_per_acre h_yield_clay_soil_factor h_total_yield
  -- math proof will be here
  sorry

end jorge_land_fraction_clay_rich_soil_l695_695034
