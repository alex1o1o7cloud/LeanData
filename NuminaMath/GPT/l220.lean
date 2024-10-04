import Mathlib

namespace total_distance_this_week_l220_220611

def daily_distance : ℕ := 6 + 7
def num_days : ℕ := 5

theorem total_distance_this_week : daily_distance * num_days = 65 := by
  rw [daily_distance, num_days]
  norm_num
  sorry

end total_distance_this_week_l220_220611


namespace count_three_digit_integers_end7_divby21_l220_220787

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220787


namespace gina_hourly_rate_l220_220747

theorem gina_hourly_rate:
  ∀ (paint_rose_rate paint_lily_rate order_rose order_lily total_payment : ℝ),
    paint_rose_rate = 6 ∧ paint_lily_rate = 7 ∧
    order_rose = 6 ∧ order_lily = 14 ∧ total_payment = 90 →
    (total_payment / ((order_rose / paint_rose_rate) + (order_lily / paint_lily_rate))) = 30 :=
by
  intros paint_rose_rate paint_lily_rate order_rose order_lily total_payment h
  obtain ⟨h_rose_rate, h_lily_rate, h_order_rose, h_order_lily, h_payment⟩ := h
  have h1 : order_rose / paint_rose_rate = 1 := by rw [h_rose_rate, h_order_rose]; norm_num
  have h2 : order_lily / paint_lily_rate = 2 := by rw [h_lily_rate, h_order_lily]; norm_num
  have h_total_time : (order_rose / paint_rose_rate) + (order_lily / paint_lily_rate) = 3 :=
    by rw [h1, h2]; norm_num
  have h_hourly : total_payment / ((order_rose / paint_rose_rate) + (order_lily / paint_lily_rate)) = 30 :=
    by rw [←h_total_time, h_payment]; norm_num
  exact h_hourly

end gina_hourly_rate_l220_220747


namespace polynomial_A_l220_220420

variables {a b : ℝ} (A : ℝ)
variables (h1 : 2 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem polynomial_A (h : A / (2 * a * b) = 1 - 4 * a ^ 2) : 
  A = 2 * a * b - 8 * a ^ 3 * b :=
by
  sorry

end polynomial_A_l220_220420


namespace probability_three_correct_packages_l220_220312

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220312


namespace probability_three_correct_out_of_five_l220_220275

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220275


namespace greatest_divisor_of_arithmetic_sequence_sum_l220_220108

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), (∃ (n : ℕ), n = 12 * x + 66 * c) → (6 ∣ n) :=
by
  intro x c _ h
  cases h with n h_eq
  use 6
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l220_220108


namespace probability_three_correct_deliveries_is_one_sixth_l220_220319

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220319


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220798

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220798


namespace find_m_l220_220400

variable (f : ℝ → ℝ)
variable (m : ℝ)

-- Definition of the function
def func_def (x : ℝ) : ℝ := 2^x + m

-- Condition: inverse function passing through (3, 1)
def passes_through : Prop := (∃ x, func_def x = 1) ∧ func_def 1 = 3

-- The theorem we want to prove
theorem find_m (h : passes_through) : m = 1 := by
  sorry

end find_m_l220_220400


namespace general_formula_for_sequence_a_l220_220388

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end general_formula_for_sequence_a_l220_220388


namespace odd_terms_in_expansion_l220_220829

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l220_220829


namespace sum_of_squares_of_binom_equals_single_binom_l220_220016

theorem sum_of_squares_of_binom_equals_single_binom (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (Nat.choose n k)^2 = Nat.choose (2 * n) n :=
by 
  sorry

end sum_of_squares_of_binom_equals_single_binom_l220_220016


namespace number_of_odd_terms_in_expansion_l220_220831

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l220_220831


namespace number_of_subsets_of_intersection_l220_220759

open Set Finset

def A := {1, 2, 3, 5}
def B := {2, 3, 5, 6, 7}

theorem number_of_subsets_of_intersection :
  card (powerset (A ∩ B)) = 8 :=
by
  sorry

end number_of_subsets_of_intersection_l220_220759


namespace truncated_cone_surface_area_l220_220455

noncomputable def surfaceArea (a : ℝ) (α : ℝ) : ℝ :=
  π * a^2 * (Real.sin (α / 2 + real.pi / 12) * Real.sin (α / 2 - real.pi / 12)) / (Real.sin α)^2

theorem truncated_cone_surface_area (a α : ℝ) (h1 : 0 < a) (h2 : 0 < α) (h3 : α < π) :
  let S := surfaceArea a α in
  S = π * a^2 * (Real.sin (α / 2 + real.pi / 12) * Real.sin (α / 2 - real.pi / 12)) / (Real.sin α)^2 :=
by
  sorry

end truncated_cone_surface_area_l220_220455


namespace smallest_arith_prog_term_l220_220263

-- Define the conditions of the problem as a structure
structure ArithProgCondition (a d : ℝ) :=
  (sum_of_squares : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (sum_of_cubes : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)

-- Define the theorem we want to prove
theorem smallest_arith_prog_term :
  ∃ (a d : ℝ), ArithProgCondition a d ∧ (a = 0 ∧ (d = sqrt 7 ∨ d = -sqrt 7) → -2 * sqrt 7) := 
sorry

end smallest_arith_prog_term_l220_220263


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220818

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220818


namespace large_planks_count_l220_220486

-- Define the primary conditions
def total_planks := 29        -- Total number of planks needed
def small_planks := 17        -- Number of small planks

-- The proof problem statement to show that the number of large planks is 12
theorem large_planks_count : ∃ (L : ℕ), L + small_planks = total_planks ∧ L = 12 :=
by {
  use 12,
  split,
  { -- First part of the statement: L + small_planks = total_planks
    exact Eq.refl (12 + small_planks) },
  { -- Second part of the statement: L = 12
    exact Eq.refl 12 }
}

end large_planks_count_l220_220486


namespace equivalent_problem_l220_220763

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem equivalent_problem
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h2 : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 := 
sorry

end equivalent_problem_l220_220763


namespace general_formula_a_n_max_min_T_n_l220_220372

noncomputable def a_n (n : ℕ+) : ℝ := (-1)^(n - 1) * (3 / 2^n)

noncomputable def S_n (n : ℕ+) : ℝ := 1 - (-1 / 2)^n

noncomputable def T_n (n : ℕ+) : ℝ := S_n n - 1 / (S_n n)

theorem general_formula_a_n (n : ℕ+) :
  a_n n = (-1)^(n - 1) * 3 / 2^n :=
sorry

theorem max_min_T_n :
  -7 / 12 ≤ ∀ n, T_n n ∧ ∀ n, T_n n ≤ 5 / 6 :=
sorry

end general_formula_a_n_max_min_T_n_l220_220372


namespace mari_made_64_buttons_l220_220025

def mari_buttons (kendra : ℕ) : ℕ := 5 * kendra + 4

theorem mari_made_64_buttons (sue kendra mari : ℕ) (h_sue : sue = 6) (h_kendra : kendra = 2 * sue) (h_mari : mari = mari_buttons kendra) : mari = 64 :=
by 
  rw [h_sue, h_kendra, h_mari]
  simp [mari_buttons]
  sorry

end mari_made_64_buttons_l220_220025


namespace initial_population_l220_220456

-- Define the initial population
variable (P : ℝ)

-- Define the conditions
theorem initial_population
  (h1 : P * 1.25 * 0.8 * 1.1 * 0.85 * 1.3 + 150 = 25000) :
  P = 24850 :=
by
  sorry

end initial_population_l220_220456


namespace positive_difference_between_median_and_mode_l220_220230

def dataset : List ℕ := [20, 20, 21, 21, 22, 33, 36, 36, 37, 43, 45, 47, 49, 62, 64, 65, 66, 68, 71, 73, 75, 79]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! (l.length / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ n m, if l.count n > l.count m then n else m) 0

theorem positive_difference_between_median_and_mode :
  median dataset - mode dataset = 23 := by
  sorry

end positive_difference_between_median_and_mode_l220_220230


namespace determinant_one_l220_220246

noncomputable def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos α * cos (2 * α), cos α * sin (2 * α), -sin α],
    ![-sin (2 * α), cos (2 * α), 0],
    ![sin α * cos (2 * α), sin α * sin (2 * α), cos α]]

theorem determinant_one (α : ℝ) : (matrix α).det = 1 := by
  sorry

end determinant_one_l220_220246


namespace work_problem_l220_220155

theorem work_problem (W : ℕ) (T_AB T_A T_B together_worked alone_worked remaining_work : ℕ)
  (h1 : T_AB = 30)
  (h2 : T_A = 60)
  (h3 : together_worked = 20)
  (h4 : T_B = 30)
  (h5 : remaining_work = W / 3)
  (h6 : alone_worked = 20)
  : alone_worked = 20 :=
by
  /- Proof is not required -/
  sorry

end work_problem_l220_220155


namespace number_of_solutions_divisible_by_l220_220897

theorem number_of_solutions_divisible_by (α : ℕ) (q : ℕ) (m : ℕ) (n : ℕ)
    (hq_odd : q % 2 = 1) (hn_eq : n = 2^α * q) (hm_pos : 0 < m) :
    ∃ k, (k * 2^(α + 1)) = (∑ x in (finset.Icc 0 m), if (x^2) ≤ m then 1 else 0) :=
sorry

end number_of_solutions_divisible_by_l220_220897


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220801

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220801


namespace sum_f_values_l220_220397

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 / (1 + x^2)

-- Definition of the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- Conditions
axiom geom_seq : ∃ r : ℝ, ∀ n : ℕ, (a (n+1)) = a n * r
axiom prod_a1_a2019 : a 1 * a 2019 = 1

-- Statement of the problem
theorem sum_f_values : (f (a 1) + f (a 2) + f (a 3) + ... + f (a 2019)) = 2019 :=
by
  sorry

end sum_f_values_l220_220397


namespace probability_three_correct_deliveries_is_one_sixth_l220_220321

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220321


namespace ratio_of_triangle_areas_l220_220094

theorem ratio_of_triangle_areas {n : ℝ} (h₀ : n > 0) :
  let side_sq := 2 in
  let area_sq := side_sq * side_sq in
  let area_triangle1 := n * area_sq in
  let r := 4 * n in
  let s := 1 / n in
  let area_triangle2 := (1 / 2) * side_sq * (1 / n) in
  area_triangle2 / area_sq = (1 / (4 * n)) :=
by sorry

end ratio_of_triangle_areas_l220_220094


namespace distance_Mia_skates_l220_220032

-- Definition of distances and speeds
def distance_XY : ℝ := 150
def speed_Mia : ℝ := 10
def speed_Noah : ℝ := 9
def angle_XZ_XY : ℝ := real.pi / 4  -- 45 degrees in radians

-- Using the necessary parameters, we assert the distance Mia travels
theorem distance_Mia_skates : 
  ∃ (t : ℝ), 
    t = (3000 * real.sqrt 2 - real.sqrt (3000^2 * 2 - 4 * 19 * 22500)) / 38 ∧
    10 * t ≈ 111.2 :=
sorry

end distance_Mia_skates_l220_220032


namespace product_mod_7_zero_l220_220733

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220733


namespace three_digit_numbers_l220_220598

theorem three_digit_numbers (digits : Finset ℕ) (n : ℕ) (h₀ : digits = {0, 1, 2, 3}) (h₁ : n = 3) :
  ∃ num : ℕ, ∀ d1 d2 d3 ∈ digits, d1 ≠ 0 → d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → num = 18 :=
by
  sorry

end three_digit_numbers_l220_220598


namespace sum_smallest_largest_l220_220557

theorem sum_smallest_largest (z b : ℤ) (n : ℤ) (h_even_n : (n % 2 = 0)) (h_mean : z = (n * b + ((n - 1) * n) / 2) / n) : 
  (2 * (z - (n - 1) / 2) + n - 1) = 2 * z := by
  sorry

end sum_smallest_largest_l220_220557


namespace f_expression_tangent_perpendicular_exists_f_xn_ym_bound_l220_220904

noncomputable def f (x : ℝ) : ℝ := (1 : ℝ)/3 * x^3 - x

def x_n (n : ℕ) : ℝ := 1 - 2^(-n)
def y_m (m : ℕ) : ℝ := sqrt 2 * (3^(-m) - 1)

theorem f_expression :
  (f = λ x, (1 : ℝ)/3 * x^3 - x) :=
sorry

theorem tangent_perpendicular_exists :
  ∃ (x1 x2 : ℝ), -sqrt 2 ≤ x1 ∧ x1 ≤ sqrt 2 ∧ -sqrt 2 ≤ x2 ∧ x2 ≤ sqrt 2 ∧
  (derivative f x1) * (derivative f x2) = -1 ∧
  x1 ≠ x2 :=
sorry

theorem f_xn_ym_bound (n m : ℕ) :
  |f (x_n n) - f (y_m m)| < 4 / 3 :=
sorry

end f_expression_tangent_perpendicular_exists_f_xn_ym_bound_l220_220904


namespace john_finishes_ahead_l220_220480

theorem john_finishes_ahead (
    initial_gap : ℝ := 14,
    speed_john : ℝ := 4.2,
    speed_steve : ℝ := 3.7,
    time : ℝ := 32
) : 
    let Distance_John := speed_john * time
    let Distance_Steve := speed_steve * time
    let Total_Distance_John := Distance_John + initial_gap
    Total_Distance_John - Distance_Steve = 30 := 
by
    sorry -- Proof omitted

end john_finishes_ahead_l220_220480


namespace find_omega_and_g_l220_220398

-- Define the function f(x) with the given conditions
def f (x : ℝ) (ω : ℝ) : ℝ :=
  √3 * sin (ω * x) * sin (ω * x + π / 2) - cos (ω * x) ^ 2 + 1 / 2

-- Given: f(x) has a period of π
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the function g(x) based on transformations
def g (x : ℝ) : ℝ :=
  sin (1 / 2 * x + π / 6)

theorem find_omega_and_g :
  (∃ ω > 0, has_period (λ x, f x ω) π ∧ ω = 1) ∧ 
  (∀ x, g x = sin (1 / 2 * x + π / 6)) :=
  by
  -- Solution structure skipped
  sorry

end find_omega_and_g_l220_220398


namespace prime_divisor_exponent_l220_220015

theorem prime_divisor_exponent (a n : ℕ) (p : ℕ) 
    (ha : a ≥ 2)
    (hn : n ≥ 1) 
    (hp : Nat.Prime p) 
    (hdiv : p ∣ a^(2^n) + 1) :
    2^(n+1) ∣ (p-1) :=
by
  sorry

end prime_divisor_exponent_l220_220015


namespace brayden_gavin_touchdowns_l220_220673

theorem brayden_gavin_touchdowns :
  (∃ (T : ℕ), 7 * T + 14 = 63) → ∃ (T : ℕ), T = 7 :=
by
  intro h
  cases h with T hT
  use T
  change 7 * T + 14 = 63 at hT
  sorry

end brayden_gavin_touchdowns_l220_220673


namespace f_divides_f_succ_pow_two_l220_220742

def smallest_positive_integer_with_d_divisors (d : ℕ) (h : d > 0) : ℕ :=
sorry -- The actual definition is omitted.

def f (d : ℕ) (h : d > 0) : ℕ :=
(smallest_positive_integer_with_d_divisors d h)

theorem f_divides_f_succ_pow_two (k : ℕ) : 
  (f (2^k) (nat.zero_lt_pow two_pos k)) ∣ (f (2^(k+1)) (nat.zero_lt_pow two_pos (k+1))) := 
sorry

end f_divides_f_succ_pow_two_l220_220742


namespace find_P_on_y_axis_l220_220460

def point := (ℝ × ℝ)

def O : point := (0, 0)
def A : point := (2, -2)

def is_on_y_axis (P : point) : Prop :=
  ∃ y, P = (0, y)

def is_isosceles (O A P : point) : Prop :=
  (P.1 = 0) ∧ ((dist O A = dist O P) ∨ (dist O A = dist A P))

theorem find_P_on_y_axis :
  ∃ (P : point), is_on_y_axis P ∧ is_isosceles O A P ∧ (4 = 4) :=
sorry

end find_P_on_y_axis_l220_220460


namespace distance_between_points_l220_220385

theorem distance_between_points {A B : ℝ}
  (hA : abs A = 3)
  (hB : abs B = 9) :
  abs (A - B) = 6 ∨ abs (A - B) = 12 :=
sorry

end distance_between_points_l220_220385


namespace range_of_data_set_l220_220970

def data_set : Set ℕ := {3, 4, 4, 6}

theorem range_of_data_set : 
  (range (data_set)) = 3 :=
sorry

end range_of_data_set_l220_220970


namespace lines_are_perpendicular_l220_220892

-- Definitions for sides and angles in triangle
variables {R : Type*} [linear_ordered_field R] [nonzero R] 
          (a b c x y : R) (A B C : R)

def line1 := sin A * x + a * y + c
def line2 := b * x - sin B * y + sin C

-- Line slopes
def slope1 := - (sin A) / a
def slope2 := b / (sin B)

theorem lines_are_perpendicular : slope1 * slope2 = -1 :=
by {
  -- Treatments formulas for line1 and line2
  sorry -- Detailed proof skipped
}

end lines_are_perpendicular_l220_220892


namespace initial_population_l220_220073

theorem initial_population (P : ℝ) (h : P * 1.21 = 12000) : P = 12000 / 1.21 :=
by sorry

end initial_population_l220_220073


namespace count_three_digit_integers_end7_divby21_l220_220788

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220788


namespace odd_terms_in_expansion_l220_220835

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l220_220835


namespace tangent_line_if_and_only_if_isosceles_l220_220499

theorem tangent_line_if_and_only_if_isosceles
  (A B C D X P Q : Point)
  (circumcircle_ABC : Circle)
  (γ : Circle)
  (hD_on_BC : D ∈ segment B C)
  (hAD_meets_circumcircle_ABC_at_X : line_through A D ∩ circumcircle_ABC = {X})
  (hP_foot_perpendicular_X_AB : is_foot_of_perpendicular X P A B)
  (hQ_foot_perpendicular_X_AC : is_foot_of_perpendicular X Q A C)
  (hγ_diameter_XD : γ = circle_with_diameter X D) :
  (is_tangent PQ γ ↔ segment_length A B = segment_length A C) :=
begin
  sorry
end

end tangent_line_if_and_only_if_isosceles_l220_220499


namespace number_of_odd_terms_in_expansion_l220_220830

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l220_220830


namespace solution_set_inequality_l220_220956

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain (x : ℝ) : x ∈ set.univ
axiom f_value : f 1 = 3
axiom f_inequality (x : ℝ) : f(x) + deriv.[2] f x < 2

theorem solution_set_inequality (x : ℝ) : (∀ x, (↑e ^ x * f x > 2 * ↑e ^ x + ↑e) ↔ x < 1) :=
by
  sorry

end solution_set_inequality_l220_220956


namespace remainder_prod_mod_7_l220_220712

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220712


namespace probability_of_exactly_three_correct_packages_l220_220286

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220286


namespace range_sin4_cos4_l220_220218

theorem range_sin4_cos4 : 
  ∀ x : ℝ, 0.5 ≤ (sin x) ^ 4 + (cos x) ^ 4 ∧ (sin x) ^ 4 + (cos x) ^ 4 ≤ 1 := by
  -- Placeholder for proof
  sorry

end range_sin4_cos4_l220_220218


namespace conjugate_of_Z_l220_220769

noncomputable def Z := (1 + 3 * Complex.i) / (1 - Complex.i)

theorem conjugate_of_Z : Complex.conj Z = -1 - 2 * Complex.i := by
  sorry

end conjugate_of_Z_l220_220769


namespace terminal_side_in_third_quadrant_l220_220756

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : tan α > 0) (h2 : sin α < 0) : 
    (α > π) ∧ (α < 3 * π / 2) :=
begin
  sorry
end

end terminal_side_in_third_quadrant_l220_220756


namespace geometric_sum_n_eq_3_l220_220976

theorem geometric_sum_n_eq_3 :
  (∃ n : ℕ, (1 / 2) * (1 - (1 / 3) ^ n) = 728 / 2187) ↔ n = 3 :=
by
  sorry

end geometric_sum_n_eq_3_l220_220976


namespace C_le_D_and_D_le_2C_l220_220132

variable (n : ℕ) (a : Fin n → ℝ)

def b (k : ℕ) : ℝ := (Finset.range k).sum (λ i, a ⟨i, Nat.lt_of_lt_of_le i.property (Nat.succ_le_of_lt (Nat.lt.base n)) ⟩) / k

def C : ℝ := (Finset.range n).sum (λ k, (a ⟨k, Nat.lt.base n⟩ - b (k + 1))^2)

def D : ℝ := (Finset.range n).sum (λ k, (a ⟨k, Nat.lt.base n⟩ - b n)^2)

theorem C_le_D_and_D_le_2C : C ≤ D ∧ D ≤ 2 * C := 
by 
  sorry

end C_le_D_and_D_le_2C_l220_220132


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220814

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220814


namespace probability_three_correct_deliveries_is_one_sixth_l220_220315

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220315


namespace neg_neg_eq_pos_l220_220136

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l220_220136


namespace count_three_digit_integers_end7_divby21_l220_220791

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220791


namespace min_price_is_750_l220_220226

noncomputable def min_selling_price (products : ℕ) (avg_price : ℕ) (min_price : ℕ) (num_less_1000 : ℕ) (max_price : ℕ) : ℕ :=
  let total_value := products * avg_price in
  let known_contributors := num_less_1000 * min_price + (max_price - min_price) in
  let unknown_contributors := (products - num_less_1000 - 1) * min_price in
  let equation := total_value - known_contributors - unknown_contributors in
  equation / (products - 1)

theorem min_price_is_750 
  (products: ℕ) 
  (avg_price: ℕ) 
  (num_less_1000: ℕ) 
  (max_price: ℕ) 
  (h_avg_price: avg_price = 1200) 
  (h_products: products = 25) 
  (h_less_1000: num_less_1000 = 10) 
  (h_max_price: max_price = 12000) 
  (min_price: ℕ)
  (h_min_price_eq: min_selling_price products avg_price min_price num_less_1000 max_price = min_price) 
  : min_price = 750 :=
by
  sorry

end min_price_is_750_l220_220226


namespace alex_score_correct_l220_220913

variable (n : ℕ) (avg17 avg18 : ℕ) (total17 total18 alex_score : ℕ)
variable (h_n : n = 18) (h_avg17 : avg17 = 75) (h_avg18 : avg18 = 76)
variable (h_total17 : total17 = 17 * avg17) (h_total18 : total18 = n * avg18) 
variable (h_alex_score : alex_score = total18 - total17)

theorem alex_score_correct :
  alex_score = 93 := by
  rw [h_total17, h_total18, h_n, h_avg17, h_avg18, ←h_alex_score]
  namespace

    have h1 : total17 = 17 * 75 := by rw h_total17; rw h_avg17
    have h2 : total18 = 18 * 76 := by rw h_total18; rw h_n; rw h_avg18
    have h3 : 17 * 75 = 1275 := by norm_num
    have h4 : 18 * 76 = 1368 := by norm_num
    have h5 : alex_score = 1368 - 1275 := by rw ← h_alex_score; rw h_total18; rw h_total17
    have h6 : 1368 - 1275 = 93 := by norm_num

    exact by
      rw h5
      rw h3
      rw h4
      rw h6
      apply h6

#<invalid.gen>
end alex_score_correct_l220_220913


namespace prove_f_zero_l220_220221

open Real

def centroid (A B C : Point) : Point := (1 / 3) • (A + B + C)

theorem prove_f_zero (f : Point → ℝ) (h : ∀ A B C : Point, f (centroid A B C) = f A + f B + f C) : ∀ A : Point, f A = 0 := 
by
  sorry

end prove_f_zero_l220_220221


namespace negation_proof_l220_220575

theorem negation_proof : ¬ (∃ x : ℝ, (x ≤ -1) ∨ (x ≥ 2)) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 := 
by 
  -- proof skipped
  sorry

end negation_proof_l220_220575


namespace john_total_time_spent_l220_220003

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end john_total_time_spent_l220_220003


namespace range_of_data_set_l220_220972

theorem range_of_data_set : 
  let data := [3, 4, 4, 6] in
  data.maximum! - data.minimum! = 3 :=
by
  sorry

end range_of_data_set_l220_220972


namespace neg_neg_eq_pos_l220_220137

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l220_220137


namespace angle_BAE_is_90_degrees_l220_220524

theorem angle_BAE_is_90_degrees
  (A B C E : Type)
  [IsTriangle ABC]
  (hE_on_AC : E ∈ lineSegment A C)
  (angle_ABE : angle A B E = 25)
  (angle_EBC : angle E B C = 65) :
  angle B A E = 90 := 
by sorry

end angle_BAE_is_90_degrees_l220_220524


namespace calculate_mari_buttons_l220_220020

theorem calculate_mari_buttons (Sue_buttons : ℕ) (Kendra_buttons : ℕ) (Mari_buttons : ℕ)
  (h_sue : Sue_buttons = 6)
  (h_kendra : Sue_buttons = 0.5 * Kendra_buttons)
  (h_mari : Mari_buttons = 4 + 5 * Kendra_buttons) :
  Mari_buttons = 64 := 
by
  sorry

end calculate_mari_buttons_l220_220020


namespace area_ASD_is_sqrt_38_l220_220951

noncomputable theory

-- Definitions of areas of the triangular faces
def area_ASB : ℝ := 25
def area_BSC : ℝ := 36
def area_CSD : ℝ := 49

-- The conditions of the problem.
-- For simplicity, we assume the areas are given by some function.

-- Proving the area of the face ASD
theorem area_ASD_is_sqrt_38 (area_ASB area_BSC area_CSD : ℝ) (h1 : area_ASB = 25) (h2 : area_BSC = 36) (h3 : area_CSD = 49) : sqrt 38 = sqrt (area_ASB + area_CSD - area_BSC) :=
begin
  simp [h1, h2, h3],
  sorry
end

end area_ASD_is_sqrt_38_l220_220951


namespace probability_exactly_three_correct_l220_220354

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220354


namespace triangle_BEF_angles_l220_220170

theorem triangle_BEF_angles 
  (circle : Type)
  (center : circle)
  (radius : ℝ)
  (l m : Π (p : circle), Prop)
  (A B C D E F: circle)
  (CD_diameter : diameter C D)
  (tangent_A : tangent l A)
  (tangent_B : tangent m B)
  (parallel_lm : ∀ (p q : circle), l p → m q → parallel p q)
  (BC_parallel : ∀ (p q : circle), parallel p C → parallel q D)
  (BC_intersect_E : intersects BC l E)
  (ED_intersect_F : intersects ED m F)
  (angle_EBD : angle E B D = 90)
  (angle_DBF : angle D B F = 45) :
  angles_triangle BEF 135 22.5 22.5 :=
sorry

end triangle_BEF_angles_l220_220170


namespace probability_of_exactly_three_correct_packages_l220_220285

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220285


namespace pete_nickels_spent_l220_220927

-- Definitions based on conditions
def initial_amount_per_person : ℕ := 250 -- 250 cents for $2.50
def total_initial_amount : ℕ := 2 * initial_amount_per_person
def total_expense : ℕ := 200 -- they spent 200 cents in total
def raymond_dimes_left : ℕ := 7
def value_of_dime : ℕ := 10
def raymond_remaining_amount : ℕ := raymond_dimes_left * value_of_dime
def raymond_spent_amount : ℕ := total_expense - raymond_remaining_amount
def value_of_nickel : ℕ := 5

-- Theorem to prove Pete spent 14 nickels
theorem pete_nickels_spent : 
  (total_expense - raymond_spent_amount) / value_of_nickel = 14 :=
by
  sorry

end pete_nickels_spent_l220_220927


namespace point_D_rotated_l220_220526

structure Point :=
  (x : ℤ)
  (y : ℤ)

def rotate_90_clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

theorem point_D_rotated : 
  let D := ⟨-2, 3⟩ in
  rotate_90_clockwise D = ⟨3, 2⟩ :=
by
  let D := ⟨-2, 3⟩
  show rotate_90_clockwise D = ⟨3, 2⟩
  sorry

end point_D_rotated_l220_220526


namespace evaluate_at_minus_three_l220_220896

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 9 * x^3 - 6 * x^2 + 15 * x - 210

theorem evaluate_at_minus_three : g (-3) = -1686 :=
by
  sorry

end evaluate_at_minus_three_l220_220896


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220811

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220811


namespace product_mod_7_zero_l220_220731

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220731


namespace three_correct_deliveries_probability_l220_220270

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220270


namespace number_of_odd_terms_in_expansion_l220_220833

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l220_220833


namespace neg_neg_eq_l220_220139

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l220_220139


namespace sum_of_y_where_g_4y_eq_11_l220_220503

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

theorem sum_of_y_where_g_4y_eq_11 : 
  let y_values := {y : ℝ | g(4 * y) = 11} in
  y_values.sum = 3 / 32 :=
by
  sorry

end sum_of_y_where_g_4y_eq_11_l220_220503


namespace magnitude_a_plus_3b_parallel_condition_cosine_angle_l220_220779

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 0)
def vector_b : ℝ × ℝ := (2, 1)

-- 1. Prove the magnitude of a + 3b
theorem magnitude_a_plus_3b : 
  |((vector_a.1 + 3 * vector_b.1), (vector_a.2 + 3 * vector_b.2))| = real.sqrt 58 := by
  sorry

-- 2. Prove k = -1/3 makes k*a - b parallel to a + 3b and they are in opposite direction
theorem parallel_condition (k : ℝ) :
  (k * vector_a.1 - vector_b.1, k * vector_a.2 - vector_b.2) = (-1/3 : ℝ) * ((vector_a.1 + 3 * vector_b.1), (vector_a.2 + 3 * vector_b.2)) →
  k = -1/3 := by
  sorry

-- 3. Prove cos(angle between a - b and b) when k*a - b is perpendicular to 3*a - b
theorem cosine_angle :
  let vector_c := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  ∃ k : ℝ, k = 1 ∧ (vector_c.1 * vector_b.1 + vector_c.2 * vector_b.2) / (real.sqrt (vector_c.1^2 + vector_c.2^2) * real.sqrt (vector_b.1^2 + vector_b.2^2)) = - (3 * real.sqrt 10 / 10) := by
  sorry

end magnitude_a_plus_3b_parallel_condition_cosine_angle_l220_220779


namespace smallest_possible_rounded_cost_l220_220204

theorem smallest_possible_rounded_cost :
  ∃ x k : ℕ, 
    let total_cost_in_dollars := (1.05 * x : ℚ) / 100 in
    (total_cost_in_dollars).round_to_nearest_multiple_of 5 = 55 ∧ 
    (5 * k * 2000 / 21 = x) ∧
    (k > 0) :=
  sorry

end smallest_possible_rounded_cost_l220_220204


namespace probability_three_correct_deliveries_is_one_sixth_l220_220322

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220322


namespace solve_for_y_l220_220045

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y_l220_220045


namespace triangle_sine_inequality_l220_220875

theorem triangle_sine_inequality
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a + b > c)
  (hbac : b + c > a)
  (hact : c + a > b)
  : |(a / (a + b)) + (b / (b + c)) + (c / (c + a)) - (3 / 2)| < (8 * Real.sqrt 2 - 5 * Real.sqrt 5) / 6 := 
sorry

end triangle_sine_inequality_l220_220875


namespace area_of_triangle_XYZ_l220_220105

noncomputable def area_triangle {A B C : Type*} [EuclideanGeometry A B C] (X Y Z : A) : ℝ :=
  1 / 2 * (dist X Y) * (dist X Z) * (sin (angle Y X Z))

theorem area_of_triangle_XYZ 
  (X Y Z W : Type*)
  [EuclideanGeometry X Y Z W]
  (coplanar : collinear X Y Z W)
  (angle_W_right : angle W X Y = π / 2)
  (XZ_length : dist X Z = 17)
  (XY_length : dist X Y = 25)
  (WZ_length : dist W Z = 8) :
  area_triangle X Y Z = 90 :=
begin
  sorry
end

end area_of_triangle_XYZ_l220_220105


namespace circles_tangent_l220_220672

/--
Two equal circles each with a radius of 5 are externally tangent to each other and both are internally tangent to a larger circle with a radius of 13. 
Let the points of tangency be A and B. Let AB = m/n where m and n are positive integers and gcd(m, n) = 1. 
We need to prove that m + n = 69.
-/
theorem circles_tangent (r1 r2 r3 : ℝ) (tangent_external : ℝ) (tangent_internal : ℝ) (AB : ℝ) (m n : ℕ) 
  (hmn_coprime : Nat.gcd m n = 1) (hr1 : r1 = 5) (hr2 : r2 = 5) (hr3 : r3 = 13) 
  (ht_external : tangent_external = r1 + r2) (ht_internal : tangent_internal = r3 - r1) 
  (hAB : AB = (130 / 8)): m + n = 69 :=
by
  sorry

end circles_tangent_l220_220672


namespace four_points_C_l220_220854

def point (ℝ : Type) := (ℝ, ℝ)

def triangle_perimeter (A B C : point ℝ) : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  AB + AC + BC

def triangle_area (A B C : point ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)).abs / 2

theorem four_points_C :
  ∃ C₁ C₂ C₃ C₄ : point ℝ,
    let A := (0 : ℝ, 0 : ℝ)
    let B := (12 : ℝ, 0 : ℝ)
    triangle_perimeter A B C₁ = 60 ∧
    triangle_area A B C₁ = 72 ∧
    triangle_perimeter A B C₂ = 60 ∧
    triangle_area A B C₂ = 72 ∧
    triangle_perimeter A B C₃ = 60 ∧
    triangle_area A B C₃ = 72 ∧
    triangle_perimeter A B C₄ = 60 ∧
    triangle_area A B C₄ = 72 :=
by
  sorry

end four_points_C_l220_220854


namespace sector_area_l220_220559

theorem sector_area 
  (central_angle : ℝ) 
  (radius : ℝ) 
  (h_angle : central_angle = 2 * π / 3) 
  (h_radius : radius = sqrt 3) : 
  (1 / 2 * radius^2 * central_angle) = π :=
by
  -- proof omitted
  sorry

end sector_area_l220_220559


namespace minimum_value_of_f_l220_220772

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x + (1 / 3)

theorem minimum_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ 1) → (∀ x : ℝ, f 1 = -(1 / 3)) :=
by
  sorry

end minimum_value_of_f_l220_220772


namespace problem_inequality_l220_220582

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l220_220582


namespace find_range_l220_220457

noncomputable def range_trig_expr (A B C : ℝ) (a b c : ℝ) : Prop :=
  let sin := Real.sin
  let cos := Real.cos
  let sqrt3 := Real.sqrt 3
  a = 1 ∧ (b * cos A - cos B = 1) ∧ (0 < A) ∧ (A < Real.pi / 2) ∧ (0 < B) ∧ (B < Real.pi / 2) ∧ (0 < C) ∧ (C < Real.pi / 2)

theorem find_range :
  ∀ (A B C a b c : ℝ), range_trig_expr A B C a b c →
  sqrt 3 < sin B + 2 * sqrt 3 * sin(A) ^ 2 ∧
  sin B + 2 * sqrt 3 * sin(A) ^ 2 < 1 + sqrt 3 :=
by
  intros A B C a b c h,
  sorry

end find_range_l220_220457


namespace eighteen_tuple_permutations_l220_220694

theorem eighteen_tuple_permutations :
  ∃ (x : Fin 18 → ℕ), (∀ i j : Fin 18, i ≤ j → x i ≤ x j)
  ∧ (set.range x = set.univ) ∧ fintype.card (finset.univ : finset (Fin 18)) = 18! :=
by {
  sorry
}

end eighteen_tuple_permutations_l220_220694


namespace positive_integers_with_cube_roots_less_than_12_l220_220416

theorem positive_integers_with_cube_roots_less_than_12 :
  {n : ℕ // 0 < n ∧ n < 1728}.card = 1727 := sorry

end positive_integers_with_cube_roots_less_than_12_l220_220416


namespace product_mod_five_remainder_is_one_l220_220676

theorem product_mod_five_remainder_is_one :
  (∏ i in finset.range 20, 7 + 10 * i) % 5 = 1 :=
by
  sorry

end product_mod_five_remainder_is_one_l220_220676


namespace percent_increase_proof_l220_220669

-- We will define the conditions
def salary_increase : ℝ := 25000
def new_salary : ℝ := 90000

-- Definition of the original salary based on the conditions
def original_salary : ℝ := new_salary - salary_increase

-- Definition of the percent increase calculation
def percent_increase : ℝ := (salary_increase / original_salary) * 100

-- Proof statement
theorem percent_increase_proof : percent_increase = 38.46 := by
  -- Proof should go here, but we are adding sorry for now
  sorry

end percent_increase_proof_l220_220669


namespace correct_meteor_passing_time_l220_220980

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_l220_220980


namespace range_of_g_l220_220008

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arctan x)^2 - (Real.arctan x) * (Real.arccot x) + (Real.arccot x)^2

theorem range_of_g : set.range g = set.Icc (π^2 / 16) (π^2 / 2) := by
  sorry

end range_of_g_l220_220008


namespace sum_of_possible_values_l220_220693

theorem sum_of_possible_values (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9)
  (h : ∃ (A B : ℕ), A + 4 + 5 + 3 + B + 6 + 2 ≡ 0 [MOD 9]) :
  ∀ s ∈ {7, 16}.sum = 23 := 
by sorry

end sum_of_possible_values_l220_220693


namespace product_mod_7_zero_l220_220732

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220732


namespace probability_three_correct_packages_l220_220314

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220314


namespace eating_ways_eq_6720_l220_220518

-- Define the problem context
def is_valid_chocolate_bar (R : ℕ) (C : ℕ) := R = 2 ∧ C = 4

def valid_eating_condition (R C : ℕ) (grid : Matrix (Fin R) (Fin C) Bool) :=
  ∀ (r c : Fin R) (hrc : grid[r, c] = true), (neighboring_unvisited r c grid ≤ 2)

-- Function that computes number of neighbor cells that are unvisited
def neighboring_unvisited (r c : Fin 2) (grid : Matrix (Fin 2) (Fin 4) Bool) : ℕ :=
  let directions := [(1, 0), (0, 1), (-1, 0), (0, -1)]  -- down, right, up, left
  directions.foldl (λ acc (dx, dy), 
    let nr := r + dx,
        nc := c + dy in 
    if nr < 2 ∧ nr >= 0 ∧ nc < 4 ∧ nc >= 0 ∧ grid[nr, nc] = true then
      acc + 1
    else
      acc) 0

noncomputable def number_of_ways_to_eat_chocolate : ℕ :=
  6720

-- The theorem statement
theorem eating_ways_eq_6720 :
  ∀ (R C : ℕ), is_valid_chocolate_bar R C → ∃ k : ℕ, valid_eating_condition R C grid → k = 6720 :=
begin
  intros R C hRC,
  existsi number_of_ways_to_eat_chocolate,
  intro h,
  sorry -- The proof steps would go here
end

end eating_ways_eq_6720_l220_220518


namespace rationalize_denominator_l220_220041

theorem rationalize_denominator : 
  (∃ A B C D : ℤ, D > 0 ∧ ¬(∃ p : ℕ, prime p ∧ p^2 ∣ B) ∧ gcd A C D = 1 ∧ A = 3 ∧ B = 8 ∧ C = 15 ∧ D = 17 ∧ 3 + 8 + 15 + 17 = 43) :=
by 
  use [3, 8, 15, 17]
  have h1 : 17 > 0 := by norm_num
  have h2 : ¬ (∃ p : ℕ, prime p ∧ p ^ 2 ∣ 8) := by 
    intro hp
    cases hp with p hp
    cases hp with prime_p p_square_divides_8
    cases p_square_divides_8 with k hk
    interval_cases p with h_cases using [2]
    { simp at hk, sorry }
    { cases h_cases with k2, contradiction }
  have h3 : gcd 3 15 17 = 1 := by norm_num
  use h1, h2, h3
  tauto
  sorry

end rationalize_denominator_l220_220041


namespace monotonic_decreasing_interval_l220_220962

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, f' x < 0 ↔ x > -1 ∧ x < 11 := 
sorry

end monotonic_decreasing_interval_l220_220962


namespace students_not_taking_languages_l220_220853

theorem students_not_taking_languages (total_students : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ) 
    (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
    total_students = 150 → french = 58 → german = 40 → spanish = 35 →
    french_german = 20 → french_spanish = 15 → german_spanish = 10 → all_three = 5 →
    total_students - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 62 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  rw [h1, h2, h3, h4, h5, h6, h7, h8],
  norm_num,
end

end students_not_taking_languages_l220_220853


namespace correct_transformation_l220_220642

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l220_220642


namespace probability_three_correct_deliveries_l220_220325

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220325


namespace sum_of_squares_eq_product_l220_220570

noncomputable def sequence : ℕ → ℕ
| 0 := 1
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 5
| (n+1) := (Finset.range (n+1)).prod sequence - 1

theorem sum_of_squares_eq_product :
  (Finset.range 70).sum (λ n, (sequence n) ^ 2) = (Finset.range 70).prod sequence :=
sorry

end sum_of_squares_eq_product_l220_220570


namespace problem_1_solution_problem_2_solution_l220_220678

noncomputable def problem_1 : Real :=
  (-3) + (2 - Real.pi)^0 - (1 / 2)⁻¹

theorem problem_1_solution :
  problem_1 = -4 :=
by
  sorry

noncomputable def problem_2 (a : Real) : Real :=
  (2 * a)^3 - a * a^2 + 3 * a^6 / a^3

theorem problem_2_solution (a : Real) :
  problem_2 a = 10 * a^3 :=
by
  sorry

end problem_1_solution_problem_2_solution_l220_220678


namespace total_students_is_37_l220_220864

-- Let b be the number of blue swim caps 
-- Let r be the number of red swim caps
variables (b r : ℕ)

-- The number of blue swim caps according to the male sports commissioner
def condition1 : Prop := b = 4 * r + 1

-- The number of blue swim caps according to the female sports commissioner
def condition2 : Prop := b = r + 24

-- The total number of students in the 3rd grade
def total_students : ℕ := b + r

theorem total_students_is_37 (h1 : condition1 b r) (h2 : condition2 b r) : total_students b r = 37 :=
by sorry

end total_students_is_37_l220_220864


namespace leaves_count_l220_220158

theorem leaves_count {m n L : ℕ} (h1 : m + n = 10) (h2 : L = 5 * m + 2 * n) :
  ¬(L = 45 ∨ L = 39 ∨ L = 37 ∨ L = 31) :=
by
  sorry

end leaves_count_l220_220158


namespace find_number_l220_220585

theorem find_number (x : ℝ) 
(h : (x^3 - 0.125) / (x^2 + 0.65) = 0.3000000000000001) : x ≈ 0.7 :=
sorry

end find_number_l220_220585


namespace correct_transformation_l220_220643

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l220_220643


namespace marco_strawberries_weight_l220_220906

variable (total_weight dads_weight marcos_weight : ℕ)

-- Conditions
def total_weight := 20
def dads_weight := 17
def marcos_weight := total_weight - dads_weight

-- Theorem stating the problem
theorem marco_strawberries_weight : marcos_weight = 3 := by
  sorry

end marco_strawberries_weight_l220_220906


namespace distinguishable_rearrangements_vowels_first_l220_220407

-- Definitions from the conditions
def word : String := "EXAMPLE"
def vowels : List Char := ['E', 'E', 'A']
def consonants : List Char := ['X', 'M', 'P', 'L']

-- The Lean 4 proof problem statement
theorem distinguishable_rearrangements_vowels_first : 
  (calculate_rearrangements_vowels_first vowels consonants) = 72 := 
by 
  sorry

-- Auxiliary definition to partition and arrange letters
def calculate_rearrangements_vowels_first (vowels consonants : List Char) : ℕ :=
  let vowel_permutations := (Permutations.count_unique vowels).nat_div (Permutations.count_repeats vowels)
  let consonant_permutations := Permutations.count_unique consonants
  vowel_permutations * consonant_permutations

end distinguishable_rearrangements_vowels_first_l220_220407


namespace geometric_sequence_sum_of_reciprocal_terms_l220_220368

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)

-- Condition: positive sequence
axiom pos_sequence : ∀ n : ℕ, 0 < a n

-- Condition: sum of first n terms
axiom sum_terms : ∀ n : ℕ, S n = ∑ i in range (n + 1), a i

-- Condition: S_n, a_n, and 1/2 form an arithmetic sequence
axiom arithmetic_seq : ∀ n : ℕ, 2 * a n = S n + (1 / 2)

-- Question 1: Prove that {a_n} is a geometric sequence
theorem geometric_sequence : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
  sorry

-- Condition for question 2: b_n = log_2 a_n + 3
axiom b_def : ∀ n : ℕ, b n = log 2 (a n) + 3

-- Question 2: Prove that the sum of the first n terms T_n of the sequence {1 / (b_n * b_(n + 1))} equals n / (2 * (n + 2))
theorem sum_of_reciprocal_terms : ∃ T : ℕ → ℝ, T = (λ n, ∑ i in range n, 1 / (b i * b (i + 1))) ∧ T n = n / (2 * (n + 2)) :=
  sorry

end geometric_sequence_sum_of_reciprocal_terms_l220_220368


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220806

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220806


namespace sum_of_six_consecutive_integers_l220_220553

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l220_220553


namespace distance_B_from_original_position_l220_220202

noncomputable theory

-- Let s be the side length of the square
def side_length_square (A : ℝ) : ℝ := real.sqrt A

-- Condition 1: Area of the square
def square_area : ℝ := 18

-- Condition 2: The visible red area is three times the visible white area
def red_to_white_area_ratio (red_area white_area : ℝ) : Prop :=
  red_area = 3 * white_area

-- Hypothesis: the given conditions
h : side_length_square square_area ^ 2 = 18
rx : ∀ x : ℝ, red_to_white_area_ratio ((1/2)* x^2) (18 - (1/2) * x^2)

-- The proof problem to show the distance, B, from its original position
theorem distance_B_from_original_position : ∃ x : ℝ, x = 6 := 
sorry

end distance_B_from_original_position_l220_220202


namespace reflected_ray_equation_l220_220194

-- Define the initial point
def point_of_emanation : (ℝ × ℝ) := (-1, 3)

-- Define the point after reflection which the ray passes through
def point_after_reflection : (ℝ × ℝ) := (4, 6)

-- Define the expected equation of the line in general form
def expected_line_equation (x y : ℝ) : Prop := 9 * x - 5 * y - 6 = 0

-- The theorem we need to prove
theorem reflected_ray_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) → expected_line_equation x y :=
sorry

end reflected_ray_equation_l220_220194


namespace arithmetic_geometric_sequence_S8_l220_220768

theorem arithmetic_geometric_sequence_S8 (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n = 1 + (n - 1) * d) → -- arithmetic sequence definition
  (S 8 = 8 * a 1 + (8 * 7 / 2) * d) → -- sum of first 8 terms calculation
  (a 1 = 1) → -- given a1 = 1
  (a 2 = 1 + d) → -- relationship for a2
  (a 5 = 1 + 4 d) → -- relationship for a5
  (a 1 * a 5 = (a 2) ^ 2) → -- geometric sequence condition
  (S 8 = 8 ∨ S 8 = 64) := -- conclusion: S8 can be 8 or 64
by sorry

end arithmetic_geometric_sequence_S8_l220_220768


namespace James_uses_150_sheets_of_paper_l220_220881

-- Define the conditions
def number_of_books := 2
def pages_per_book := 600
def pages_per_side := 4
def sides_per_sheet := 2

-- Statement to prove
theorem James_uses_150_sheets_of_paper :
  number_of_books * pages_per_book / (pages_per_side * sides_per_sheet) = 150 :=
by sorry

end James_uses_150_sheets_of_paper_l220_220881


namespace probability_of_exactly_three_correct_packages_l220_220289

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220289


namespace ratio_of_sleep_l220_220682

theorem ratio_of_sleep (connor_sleep : ℝ) (luke_extra : ℝ) (puppy_sleep : ℝ) 
    (h1 : connor_sleep = 6)
    (h2 : luke_extra = 2)
    (h3 : puppy_sleep = 16) :
    puppy_sleep / (connor_sleep + luke_extra) = 2 := 
by 
  sorry

end ratio_of_sleep_l220_220682


namespace forest_volume_scientific_notation_l220_220687

theorem forest_volume_scientific_notation :
  ∃ a : ℕ, 26900000 = 2.69 * 10^a :=
by
  use 7
  sorry

end forest_volume_scientific_notation_l220_220687


namespace probability_three_correct_deliveries_l220_220334

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220334


namespace rectangle_area_increase_l220_220245

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A_original := L * W
  let A_new := (2 * L) * (2 * W)
  (A_new - A_original) / A_original * 100 = 300 := by
  sorry

end rectangle_area_increase_l220_220245


namespace total_worksheets_l220_220206

theorem total_worksheets (worksheets_graded : ℕ) (problems_per_worksheet : ℕ) (problems_remaining : ℕ)
  (h1 : worksheets_graded = 7)
  (h2 : problems_per_worksheet = 2)
  (h3 : problems_remaining = 14): 
  worksheets_graded + (problems_remaining / problems_per_worksheet) = 14 := 
by 
  sorry

end total_worksheets_l220_220206


namespace distance_to_ceiling_l220_220453

noncomputable def fly_distance_from_ceiling : ℝ :=
  let P := (0, 0, 0)  -- point P at the origin
  let slope_ceiling (x y : ℝ) : ℝ := (1 / 3) * x + (1 / 3) * y  -- ceiling equation
  let dist (a b : ℝ × ℝ × ℝ) : ℝ := real.sqrt ((a.1.1 - b.1.1)^2 + (a.1.2 - b.1.2)^2 + (a.2 - b.2)^2)  -- Euclidean distance
  let fly := (2, 6, real.sqrt 60)  -- Fly's coordinates
  let fly_dist := dist P fly  -- Distance of the fly from P
  let ceiling_height := slope_ceiling 2 6  -- Ceiling height at fly's (x, y) position
  (fly.2 - ceiling_height)  -- Vertical distance from the fly to the ceiling

theorem distance_to_ceiling : 
  ∀ P : ℝ × ℝ × ℝ, 
  ∀ slope_ceiling : ℝ → ℝ → ℝ, 
  ∀ dist : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ,
  ∀ fly : ℝ × ℝ × ℝ,
  ∀ fly_dist : ℝ,
  ∀ ceiling_height : ℝ,
  P = (0, 0, 0) →
  slope_ceiling = (λ x y, (1 / 3) * x + (1 / 3) * y) →
  dist = (λ a b, real.sqrt ((a.1.1 - b.1.1)^2 + (a.1.2 - b.1.2)^2 + (a.2 - b.2)^2)) →
  fly = (2, 6, real.sqrt 60) →
  fly_dist = dist P fly →
  ceiling_height = slope_ceiling 2 6 →
  fly.2 - ceiling_height = (3 * real.sqrt 60 - 8) / 3 :=
by
  intros
  -- Proof steps go here
  sorry

end distance_to_ceiling_l220_220453


namespace find_numbers_l220_220663

def seven_digit_number (n : ℕ) : Prop := 10^6 ≤ n ∧ n < 10^7

theorem find_numbers (x y : ℕ) (hx: seven_digit_number x) (hy: seven_digit_number y) :
  10^7 * x + y = 3 * x * y → x = 1666667 ∧ y = 3333334 :=
by
  sorry

end find_numbers_l220_220663


namespace registration_methods_count_l220_220605

def students : Finset ℕ := {0, 1, 2, 3, 4}  -- Representing students A, B, C, D, E as 0, 1, 2, 3, 4.

def courses : Finset ℕ := {0, 1, 2}  -- Representing courses A, B, C as 0, 1, 2.

def enrollments (s : Finset ℕ) (c : Finset ℕ) :=
  { f : ℕ → ℕ // ∀ x ∈ s, f x ∈ c ∧ ∀ y ∈ c, ∃ x ∈ s, f x = y }

noncomputable def number_of_registration_methods : ℕ :=
  (enrollments students courses).card -- The number of valid enrollment mappings.

theorem registration_methods_count : number_of_registration_methods = 150 := 
sorry

end registration_methods_count_l220_220605


namespace generate_function_table_l220_220625

theorem generate_function_table :
    ∃ (x_values : List ℝ), ( (∀ x ∈ x_values, 1 ≤ x ∧ x ≤ 2.5) ∧ 
    (List.length x_values = ((2.5 - 1) / 0.05).floor + 1) ∧
    ∀ x ∈ x_values, ∃ y z w : ℝ,
    y = Real.sqrt x ∧
    z = x^2 ∧
    w = x - Real.floor x ) :=
by
  sorry

end generate_function_table_l220_220625


namespace tangent_length_l220_220995

noncomputable def length_PQ (AB AC: ℝ) (P B R Q : Type) [MetricSpace P B] [MetricSpace P R] [MetricSpace P Q] : ℝ := sorry

theorem tangent_length (AB AC: ℝ) (P B R Q : Type) [MetricSpace P B] [MetricSpace P R] [MetricSpace P Q]  
    (h₁ : tangent P B) 
    (h₂ : tangent P R) 
    (h₃ : tangent P Q) 
    (equal_segments : PB = PR)
    (length_of_AB : AB = 24) :
    (length_PQ PQ) = 12 := sorry

end tangent_length_l220_220995


namespace probability_three_correct_deliveries_is_one_sixth_l220_220316

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220316


namespace simplify_expr1_simplify_expr2_l220_220675

theorem simplify_expr1 : (1 + 1 / 2 : ℝ)^0 - (1 - (0.5 : ℝ)^(-2)) / ((27 / 8 : ℝ)^(2 / 3)) = 7 / 3 :=
by sorry

theorem simplify_expr2 : (Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) = (2 : ℝ)^(7 / 8) :=
by sorry

end simplify_expr1_simplify_expr2_l220_220675


namespace coefficient_of_expansion_l220_220381

open Real

theorem coefficient_of_expansion (a b c : ℝ) :
  let m := 3 * ∫ x in 0..π, sin x
  let expansion_term := m = 6
  expansion_term → 
  coefficient_of_ab2c_m_3 (a + 2 * b - 3 * c) m = -6480 := 
by
  sorry

end coefficient_of_expansion_l220_220381


namespace incorrect_statement_about_categorical_variables_l220_220624

theorem incorrect_statement_about_categorical_variables {X Y : Type} (k : ℕ) (K2 : X → Y → ℕ) :
  (∀ x y, K2 x y = k) → ¬ (k < by sorry → ∀ x y, confidence_in_judging "X is related to Y" k < by sorry) :=
by
  sorry

end incorrect_statement_about_categorical_variables_l220_220624


namespace darcy_folded_shirts_l220_220232

-- Defining the conditions as Lean definitions
def S_total := 20
def P_total := 8
def P_folded := 5
def C_remaining := 11

-- The Lean statement
theorem darcy_folded_shirts : 
  let C_total := S_total + P_total in
  let C_after_folding_shorts := C_total - P_folded in
  let C_folded := C_after_folding_shorts - C_remaining in
  let shirts_folded := C_folded - P_folded in
  shirts_folded = 7 :=
by sorry

-- This Lean statement formalizes the proof problem: Given the conditions specified, we prove that the number of shirts Darcy folded is 7.

end darcy_folded_shirts_l220_220232


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220797

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220797


namespace mistaken_operation_l220_220183

variable (x : ℝ)

theorem mistaken_operation :
  (exists O : ℝ → ℝ, (x → 10 * x) ∧ (percentage_error = 99)) → 
  (10 * x > 0.01 * (10 * x) ∧ O x = 0.1 * x) :=
sorry

end mistaken_operation_l220_220183


namespace sum_of_1984_consecutive_integers_not_square_l220_220938

theorem sum_of_1984_consecutive_integers_not_square :
  ∀ n : ℕ, ¬ ∃ k : ℕ, 992 * (2 * n + 1985) = k * k := by
  sorry

end sum_of_1984_consecutive_integers_not_square_l220_220938


namespace three_correct_deliveries_probability_l220_220267

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220267


namespace avg_rate_first_half_l220_220910

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l220_220910


namespace day_after_72_days_from_Friday_l220_220614

def day_of_week_after_n_days(start_day : ℕ, n : ℕ) : ℕ :=
  (start_day + n) % 7

theorem day_after_72_days_from_Friday : day_of_week_after_n_days 5 72 = 0 := 
by
  sorry

end day_after_72_days_from_Friday_l220_220614


namespace find_divisor_l220_220248

def Dividend : ℕ := 56
def Quotient : ℕ := 4
def Divisor : ℕ := 14

theorem find_divisor (h1 : Dividend = 56) (h2 : Quotient = 4) : Dividend / Quotient = Divisor := by
  sorry

end find_divisor_l220_220248


namespace Vins_bike_trip_distance_l220_220609

theorem Vins_bike_trip_distance (d1 d2 d3 : ℕ) (m1 m2 trips : ℕ) : 
  d1 = 6 → d2 = 7 → trips = 5 → d3 = d1 + d2 → m1 = d3 * trips → m2 = 65 → 
  m1 = m2 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h4, h1, h2, h5] at h6
  exact h6
  sorry

end Vins_bike_trip_distance_l220_220609


namespace present_age_of_son_l220_220177

/-- A man is 46 years older than his son and in two years, the man's age will be twice the age of his son. Prove that the present age of the son is 44. -/
theorem present_age_of_son (M S : ℕ) (h1 : M = S + 46) (h2 : M + 2 = 2 * (S + 2)) : S = 44 :=
by {
  sorry
}

end present_age_of_son_l220_220177


namespace probability_of_exactly_three_correct_packages_l220_220288

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220288


namespace monthly_cost_per_iguana_l220_220783

theorem monthly_cost_per_iguana
  (gecko_cost snake_cost annual_cost : ℕ)
  (monthly_cost_per_iguana : ℕ)
  (gecko_count iguana_count snake_count : ℕ)
  (annual_cost_eq : annual_cost = 1140)
  (gecko_count_eq : gecko_count = 3)
  (iguana_count_eq : iguana_count = 2)
  (snake_count_eq : snake_count = 4)
  (gecko_cost_eq : gecko_cost = 15)
  (snake_cost_eq : snake_cost = 10)
  (total_annual_cost_eq : gecko_count * gecko_cost + iguana_count * monthly_cost_per_iguana * 12 + snake_count * snake_cost * 12 = annual_cost) :
  monthly_cost_per_iguana = 5 :=
by
  sorry

end monthly_cost_per_iguana_l220_220783


namespace at_most_four_distinct_numbers_l220_220944

def property_P (a b : ℝ) : Prop :=
  (a + b) ∈ ℤ ∧ (1 / a + 1 / b) ∈ ℤ 

theorem at_most_four_distinct_numbers 
  (nums : List ℝ) 
  (h_nonzero : ∀ x ∈ nums, x ≠ 0)
  (h_circle : ∀ i, property_P (nums.nthLe i sorry) (nums.nthLe ((i + 1) % nums.length) sorry)) :
  nums.toFinset.card ≤ 4 := 
sorry

end at_most_four_distinct_numbers_l220_220944


namespace vasya_password_combinations_l220_220997

theorem vasya_password_combinations : 
  (∃ (A B C : ℕ), A ≠ 2 ∧ B ≠ 2 ∧ C ≠ 2 ∧ A = 0 ∨ A = 1 ∨ A = 3 ∨ A = 4 ∨ A = 5 ∨ A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9 ∧
                   B = 0 ∨ B = 1 ∨ B = 3 ∨ B = 4 ∨ B = 5 ∨ B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9 ∧
                   C = 0 ∨ C = 1 ∨ C = 3 ∨ C = 4 ∨ C = 5 ∨ C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9 ∧
                   A ≠ B ∧ B ≠ C ∧ A = lastDigit ∧ firstDigit = lastDigit) →
  (classical.some (9 * 8 * 7) = 504) :=
by
  sorry

end vasya_password_combinations_l220_220997


namespace family_reunion_soda_cost_l220_220259

theorem family_reunion_soda_cost :
  let people := 60 in
  let cans_per_person := 2 in
  let cola_preference := 20 in
  let lemon_lime_preference := 15 in
  let orange_preference := 18 in
  let no_preference := people - cola_preference - lemon_lime_preference - orange_preference in
  let cola_cans_per_box := 10 in
  let lemon_lime_cans_per_box := 12 in
  let orange_cans_per_box := 8 in
  let cola_box_cost := 2 in
  let lemon_lime_box_cost := 3 in
  let orange_box_cost := 2.5 in
  let sales_tax := 0.06 in
  let family_members := 6 in

  let total_cans := people * cans_per_person in
  let cola_cans_needed := cola_preference * cans_per_person in
  let lemon_lime_cans_needed := lemon_lime_preference * cans_per_person in
  let orange_cans_needed := orange_preference * cans_per_person in
  let remaining_cans_needed := no_preference * cans_per_person in
  let cola_boxes_needed := (cola_cans_needed + cola_cans_per_box - 1) / cola_cans_per_box in
  let lemon_lime_boxes_needed := (lemon_lime_cans_needed + lemon_lime_cans_per_box - 1) / lemon_lime_cans_per_box in
  let orange_boxes_needed := (orange_cans_needed + orange_cans_per_box - 1) / orange_cans_per_box in

  let total_cost_before_tax := cola_boxes_needed * cola_box_cost +
                               lemon_lime_boxes_needed * lemon_lime_box_cost +
                               orange_boxes_needed * orange_box_cost in

  let total_cost_with_tax := total_cost_before_tax + total_cost_before_tax * sales_tax in

  let cost_per_family_member := total_cost_with_tax / family_members in

  total_cost_after_tax < 31.27 + 0.01 ∧
  total_cost_after_tax > 31.27 - 0.01 ∧
  cost_per_family_member < 5.21 + 0.01 ∧
  cost_per_family_member > 5.21 - 0.01 
  := by
    sorry

end family_reunion_soda_cost_l220_220259


namespace num_cube_roots_less_than_12_l220_220413

/-- There are 1727 positive whole numbers whose cube roots are less than 12 -/
theorem num_cube_roots_less_than_12 : 
  { n : ℕ // 0 < n ∧ ∃ x : ℕ, x = n ∧ (x : ℝ)^(1/3) < 12 }.card = 1727 := 
sorry

end num_cube_roots_less_than_12_l220_220413


namespace count_lines_l220_220470

open set

def distinct_elements (s : set ℤ) : Prop :=
  (s = {-3, -2, -1, 0, 1, 2, 3})

def is_valid_line (a b c: ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_acute_angle (a b : ℤ) : Prop :=
  a > 0 ∧ b < 0

noncomputable def count_distinct_lines : ℕ :=
  card { l | ∃ a b c, 
    a ≠ 0 ∧ a > 0 ∧ distinct_elements {-3, -2, -1, 0, 1, 2, 3} ∧
    is_valid_line a b c ∧ is_acute_angle a b ∧ l = (λ x y, a * x + b * y + c) }

theorem count_lines : count_distinct_lines = 43 := 
by sorry

end count_lines_l220_220470


namespace max_value_of_f_find_a_when_Ma_eq_2_l220_220685

noncomputable def f (x a : ℝ) : ℝ := 
  (Real.cos x)^2 + a * (Real.sin x) - a / 4 - 1 / 2

def M (a : ℝ) := if 0 < a ∧ a ≤ 2 then (1 / 4 * a^2) - (1 / 4 * a) + 1 / 2 else (3 / 4 * a) - 1 / 2

theorem max_value_of_f (a : ℝ) (h₁ : 0 < a) : 
  ∃ x (hx : 0 ≤ x ∧ x ≤ (Real.pi / 2)), f x a = M a :=
sorry

theorem find_a_when_Ma_eq_2 (a : ℝ) (h₁ : 0 < a) (h₂ : M a = 2) : a = 10 / 3 :=
sorry

end max_value_of_f_find_a_when_Ma_eq_2_l220_220685


namespace double_neg_eq_pos_l220_220144

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l220_220144


namespace simplify_complex_expression_l220_220940

-- Given complex numbers and their operations
variable (a : ℂ) (b : ℂ) (c : ℂ)

-- The specific problem values
def z1 := -5 + 3 * complex.i
def z2 := 2 - 7 * complex.i
def z3 := 1 + 2 * complex.i

-- The desired result
def result := -6 + 12 * complex.i

-- The theorem to prove
theorem simplify_complex_expression (z1 z2 z3 : ℂ) :
  (z1 - z2 + z3 = result) :=
by
  -- Sorry is used to indicate that the proof is omitted
  sorry

end simplify_complex_expression_l220_220940


namespace probability_of_three_correct_packages_l220_220303

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220303


namespace john_payoff_probability_l220_220634

theorem john_payoff_probability : 
    let prob_single_6 := 1 / 6 
    let prob_john_18 := (prob_single_6) ^ 3
    let prob_not_18 := 1 - (prob_single_6) ^ 3
    let prob_others_not_18 := (prob_not_18) ^ 22
    in  prob_john_18 * prob_others_not_18 =
        (1 / 216) * (1 - (1 / 216)) ^ 22 := 
by 
 -- let definitions for all the probabilities as described in the problem
 sorry

end john_payoff_probability_l220_220634


namespace natural_numbers_equal_power_l220_220755

theorem natural_numbers_equal_power
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n :=
by
  sorry

end natural_numbers_equal_power_l220_220755


namespace nested_radicals_simplification_l220_220542

noncomputable def simplify_nested_radicals : Prop :=
  sqrt (8 + 4 * sqrt 3) + sqrt (8 - 4 * sqrt 3) = 2 * sqrt 6

theorem nested_radicals_simplification : simplify_nested_radicals := by
  sorry

end nested_radicals_simplification_l220_220542


namespace anne_age_ratio_l220_220217

-- Define the given conditions and prove the final ratio
theorem anne_age_ratio (A M : ℕ) (h1 : A = 4 * (A - 4 * M) + M) 
(h2 : A - M = 3 * (A - 4 * M)) : (A : ℚ) / (M : ℚ) = 5.5 := 
sorry

end anne_age_ratio_l220_220217


namespace john_time_spent_on_pictures_l220_220002

noncomputable def time_spent (num_pictures : ℕ) (draw_time color_time : ℝ) : ℝ := 
  (num_pictures * draw_time) + (num_pictures * color_time)

theorem john_time_spent_on_pictures :
  ∀ (num_pictures : ℕ) (draw_time : ℝ) (color_percentage_less : ℝ)
    (color_time : ℝ),
    num_pictures = 10 →
    draw_time = 2 →
    color_percentage_less = 0.30 →
    color_time = draw_time - (color_percentage_less * draw_time) →
    time_spent num_pictures draw_time color_time = 34 :=
begin
  sorry,
end

end john_time_spent_on_pictures_l220_220002


namespace remaining_squares_count_l220_220985

-- Define initial conditions
def initial_matchsticks : ℕ := 40
def initial_squares : ℕ := 30

-- Define claims made by individuals
def claim_A (remaining_squares_1x1 : ℕ) : Prop := remaining_squares_1x1 = 5
def claim_B (remaining_squares_2x2 : ℕ) : Prop := remaining_squares_2x2 = 3
def claim_C (remaining_squares_3x3 : ℕ) : Prop := true -- All remain
def claim_D (all_different_lines : Bool) : Prop := all_different_lines
def claim_E (four_same_line : Bool) : Prop := four_same_line

-- Two claims are incorrect
def two_incorrect_claims := 2

-- Removed matchsticks and their impact:
def removed_matchsticks : ℕ := 5

theorem remaining_squares_count
  (initial_matchsticks squares_removed
  remaining_squares_1x1 remaining_squares_2x2 remaining_squares_3x3 remaining_squares_4x4 : ℕ)
  (all_different_lines four_same_line : Bool)
  (condition1 : initial_matchsticks = 40)
  (condition2 : squares_removed = removed_matchsticks)
  (condition3 : remaining_squares_1x1 + remaining_squares_2x2 + remaining_squares_3x3 + remaining_squares_4x4 = 14)
  (condition4 : (claim_A remaining_squares_1x1 ∧ claim_B remaining_squares_2x2 ∧ claim_C ∧ claim_D all_different_lines ∧ claim_E four_same_line) ∇ two_incorrect_claims) :
  (remaining_squares_1x1 + remaining_squares_2x2 + remaining_squares_3x3 + remaining_squares_4x4 = 14) :=
sorry

end remaining_squares_count_l220_220985


namespace probability_two_red_two_blue_one_green_l220_220260

def total_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_choose_red (r : ℕ) : ℕ := total_ways_to_choose 4 r
def ways_to_choose_blue (b : ℕ) : ℕ := total_ways_to_choose 3 b
def ways_to_choose_green (g : ℕ) : ℕ := total_ways_to_choose 2 g

def successful_outcomes (r b g : ℕ) : ℕ :=
  ways_to_choose_red r * ways_to_choose_blue b * ways_to_choose_green g

def total_outcomes : ℕ := total_ways_to_choose 9 5

def probability_of_selection (r b g : ℕ) : ℚ :=
  (successful_outcomes r b g : ℚ) / (total_outcomes : ℚ)

theorem probability_two_red_two_blue_one_green :
  probability_of_selection 2 2 1 = 2 / 7 := by
  sorry

end probability_two_red_two_blue_one_green_l220_220260


namespace distance_from_origin_l220_220449

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end distance_from_origin_l220_220449


namespace xy_parallel_to_ad_l220_220931

-- Definitions related to the problem
variables {α : Type*} [linear_ordered_field α]
variables (A B C D K X Y : Point α)
variables (circle : Circle α)
variables (line1 : Line α) (line2 : Line α)
variables (diag1 : Line α) (diag2 : Line α)

-- Conditions of the problem
def inscribed_quadrilateral (A B C D : Point α) (circle : Circle α) : Prop :=
  circle.contains A ∧ circle.contains B ∧ circle.contains C ∧ circle.contains D

def midpoint_of_arc (K A D : Point α) (circle : Circle α) : Prop :=
  circle.contains K ∧ 
  ∃ arc1 arc2, 
    arc1.contains A ∧ arc1.contains D ∧ 
    arc2.contains A ∧ arc2.contains D ∧ 
    arc1 ≠ arc2 ∧ 
    arc1.length = arc2.length ∧ 
    arc1 ∩ arc2 = {K}

def intersection_with_diagonal (K diag : Line α) : Point α := sorry

-- Given the configurations
noncomputable def configure : Prop :=
  ∃ (A B C D K X Y : Point α) (circle : Circle α) (line1 line2 diag1 diag2 : Line α),
  inscribed_quadrilateral A B C D circle ∧
  midpoint_of_arc K A D circle ∧
  K = intersection_with_arc_midpoint A D circle ∧
  X = intersection_with_diagonal K diag1 ∧
  Y = intersection_with_diagonal K diag2

-- Prove our claim
theorem xy_parallel_to_ad : configure → parallel X Y A D := by
  sorry

end xy_parallel_to_ad_l220_220931


namespace sequence_product_modulo_7_l220_220717

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220717


namespace integer_root_of_polynomial_l220_220579

/-- Prove that -6 is a root of the polynomial equation x^3 + bx + c = 0,
    where b and c are rational numbers and 3 - sqrt(5) is a root
 -/
theorem integer_root_of_polynomial (b c : ℚ)
  (h : ∀ x : ℝ, (x^3 + (b : ℝ)*x + (c : ℝ) = 0) → x = (3 - Real.sqrt 5) ∨ x = (3 + Real.sqrt 5) ∨ x = -6) :
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6 :=
by
  sorry

end integer_root_of_polynomial_l220_220579


namespace transform_apples_and_pears_l220_220986

theorem transform_apples_and_pears (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (∃ f : fin (a+b) → ℕ, (∀ i, f i = 0 ∨ f i = 1) ∧ 
    (∀ i < a, f i = 0) ∧ 
    (∀ j ∈ set.Ico (a) (a + b), f j = 1) → 
      (∃ g : fin (a+b) → ℕ, (∀ j, g j = 0 ∨ g j = 1) ∧ 
        (∀ j < b, g j = 1) ∧ 
        (∀ k ∈ set.Ico b (a + b), g k = 0))
  ↔ (a * b) % 2 = 0) :=
sorry

end transform_apples_and_pears_l220_220986


namespace direct_variation_y_value_l220_220430

theorem direct_variation_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x) (h2 : y = 15) (h3 : x = 5) : 
  y = 3 * -10 :=
by
  revert k x y h1 h2 h3
  intros k x y h1 h2 h3
  have : k = 3 := by
    have : 15 = k * 5 := by rw [h1, h2, h3]
    exact eq_of_mul_eq_mul_right (ne_of_gt zero_lt_five) this
  rw [this, neg_mul_eq_neg_mul]
  exact eq.refl -30

end direct_variation_y_value_l220_220430


namespace range_of_a_l220_220149

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l220_220149


namespace num_cube_roots_less_than_12_l220_220412

/-- There are 1727 positive whole numbers whose cube roots are less than 12 -/
theorem num_cube_roots_less_than_12 : 
  { n : ℕ // 0 < n ∧ ∃ x : ℕ, x = n ∧ (x : ℝ)^(1/3) < 12 }.card = 1727 := 
sorry

end num_cube_roots_less_than_12_l220_220412


namespace _l220_220851

def Ivanight_moves (x y : ℕ) : list (ℕ × ℕ) := [(x + 2) % 8, (y + 1) % 8, (x + 2) % 8, (y - 1 + 8) % 8,
                                                (x - 2 + 8) % 8, (y + 1) % 8, (x - 2 + 8) % 8, (y - 1 + 8) % 8,
                                                (x + 1) % 8, (y + 2) % 8, (x + 1) % 8, (y - 2 + 8) % 8,
                                                (x - 1 + 8) % 8, (y + 2) % 8, (x - 1 + 8) % 8, (y - 2 + 8) % 8]

def color_distribution (board : ℕ → ℕ → ℕ) (moves : list (ℕ × ℕ)) : Prop :=
  ∀ c, c < n → (list.count moves (λ (_, y) → board x y = c) = 8 / n)

noncomputable def ZS_chessboard_exists (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 8 → 
    ∃ (board : ℕ → ℕ → ℕ), 
      ∀ (x y : ℕ), let moves := Ivanight_moves x y in color_distribution board moves

axiom main_theorem : ∀ n, ZS_chessboard_exists n

end _l220_220851


namespace find_value_of_f2_sub_f3_l220_220766

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem find_value_of_f2_sub_f3 (h_odd : is_odd_function f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 :=
by
  sorry

end find_value_of_f2_sub_f3_l220_220766


namespace fraction_equality_l220_220421

theorem fraction_equality (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := 
by
  -- Use the hypthesis to derive that a = 2k, b = 3k, c = 4k and show the equality.
  sorry

end fraction_equality_l220_220421


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220805

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220805


namespace gcd_of_product_diff_is_12_l220_220690

theorem gcd_of_product_diff_is_12
  (a b c d : ℤ) : ∃ (D : ℤ), D = 12 ∧
  ∀ (a b c d : ℤ), D ∣ (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b) :=
by
  use 12
  sorry

end gcd_of_product_diff_is_12_l220_220690


namespace point_cannot_exist_on_line_l220_220049

theorem point_cannot_exist_on_line (m k : ℝ) (h : m * k > 0) : ¬ (2000 * m + k = 0) :=
sorry

end point_cannot_exist_on_line_l220_220049


namespace correct_operation_l220_220115

theorem correct_operation (a : ℝ) :
  (a^2)^3 = a^6 :=
by
  sorry

end correct_operation_l220_220115


namespace FG_tangent_to_inscribed_circle_l220_220578

theorem FG_tangent_to_inscribed_circle
  (A B C D E F G: Point)
  (h_square: is_square A B C D)
  (h_E_midpoint: midpoint E A B)
  (h_F_on_BC: F ∈ BC)
  (h_G_on_CD: G ∈ CD)
  (h_parallel: parallel (line A G) (line E F)) :
  tangent (segment F G) (circle_inscribed_in_square A B C D) :=
sorry

end FG_tangent_to_inscribed_circle_l220_220578


namespace solve_equation_l220_220546

theorem solve_equation : 
  ∃ x y : ℝ, (3 * x^2 + 14 * y^2 - 12 * x * y + 6 * x - 20 * y + 11 = 0) ∧
            (x = 3) ∧ (y = 2) := 
by
  have x := 3
  have y := 2
  use x, y
  split
  { sorry }
  split
  { exact rfl }
  { exact rfl }

end solve_equation_l220_220546


namespace collinear_vectors_imply_x_eq_sqrt2_l220_220405

noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * Real.pi / 180

-- Conditions definitions
def vector_a : ℝ × ℝ := (1, Real.sqrt (1 + Real.sin (deg_to_rad 40)))
def vector_b (x : ℝ) : ℝ × ℝ := (1 / Real.sin (deg_to_rad 65), x)

-- Statement
theorem collinear_vectors_imply_x_eq_sqrt2 (x : ℝ) :
  (∃ k : ℝ, vector_a = (k * (1 / Real.sin (deg_to_rad 65)), k * x)) → x = Real.sqrt 2 :=
by
  sorry

end collinear_vectors_imply_x_eq_sqrt2_l220_220405


namespace largest_possible_perimeter_l220_220602

noncomputable def max_perimeter (a b c: ℕ) : ℕ := 2 * (a + b + c - 6)

theorem largest_possible_perimeter :
  ∃ (a b c : ℕ), (a = c) ∧ ((a - 2) * (b - 2) = 8) ∧ (max_perimeter a b c = 42) := by
  sorry

end largest_possible_perimeter_l220_220602


namespace base_length_of_prism_l220_220848

theorem base_length_of_prism (V : ℝ) (hV : V = 36 * Real.pi) : ∃ (AB : ℝ), AB = 3 * Real.sqrt 3 :=
by
  sorry

end base_length_of_prism_l220_220848


namespace hyperbola_eccentricity_l220_220949

theorem hyperbola_eccentricity (a b c e : ℝ) (h_asymptote : ∀ x : ℝ, y = a / 2, b / 2) :
  (e = sqrt 5 / 2) ∨ (e = sqrt 5) :=
sorry

end hyperbola_eccentricity_l220_220949


namespace degree_g_l220_220497

noncomputable def f (x : ℚ) : ℚ := -9 * x ^ 4 + 2 * x ^ 3 - 7 * x + 8

def g (x : ℚ) : ℚ

theorem degree_g (h : ∀ x, degree (f x + g x) = 1) : ∃ n : ℕ, degree (g x) = 4 :=
sorry

end degree_g_l220_220497


namespace standard_equation_of_ellipse_exists_line_intersecting_ellipse_with_circle_passing_origin_l220_220373

-- Question part (1)
theorem standard_equation_of_ellipse :
  let e := (1 : ℝ) / 2 in
  let a := 2 * 1 in
  let c := 1 in
  let b_squared := a^2 - c^2 in
  (b_squared = 3) →
  ∃ a b : ℝ, (a = 2) ∧ (b = sqrt 3) ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ∧ (eccentricity_eq e a c) ∧ (a - c = 1) ∧ (a > 0) ∧
  ∃ x y : ℝ, ((x^2 / 4) + (y^2 / 3) = 1) :=
by
  sorry

-- Question part (2)
theorem exists_line_intersecting_ellipse_with_circle_passing_origin :
  let F : Type := ℝ in -- Field Type
  let k m : F in
  let h := - (2 : F) * Real.sqrt (21 : F) / 7 in
  let g := (2 : F) * Real.sqrt (21 : F) / 7 in
  ∃ (A B : F × F), (m - g = - (21 : ℝ) / 7) ∧ (k ∈ (- k / (21 : F)) ∨ (7 * k = 12) ) :=
by
  sorry

end standard_equation_of_ellipse_exists_line_intersecting_ellipse_with_circle_passing_origin_l220_220373


namespace mari_made_64_buttons_l220_220024

def mari_buttons (kendra : ℕ) : ℕ := 5 * kendra + 4

theorem mari_made_64_buttons (sue kendra mari : ℕ) (h_sue : sue = 6) (h_kendra : kendra = 2 * sue) (h_mari : mari = mari_buttons kendra) : mari = 64 :=
by 
  rw [h_sue, h_kendra, h_mari]
  simp [mari_buttons]
  sorry

end mari_made_64_buttons_l220_220024


namespace contrapositive_example_l220_220054

theorem contrapositive_example (x : ℝ) : 
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := 
by
  sorry

end contrapositive_example_l220_220054


namespace min_value_cyclic_sum_l220_220500

theorem min_value_cyclic_sum (a b c d : ℝ) :
  let Σ (f : ℝ → ℝ) := f a + f b + f c + f d in
  Σ (λ x, x^2) + Σ (λ x, match x with | a => b | b => c | c => d | d => a end * x) + Σ (λ x, x) >= -2/5 :=
sorry

end min_value_cyclic_sum_l220_220500


namespace probability_of_three_correct_packages_l220_220299

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220299


namespace area_of_quadrilateral_ABCD_l220_220953

theorem area_of_quadrilateral_ABCD : 
  (∃ (x y : ℝ), x^2 + y^2 - x * y + x - 4 * y = 12) →
  (let A := (0, 6) in
   let C := (0, -2) in
   let B := (3, 0) in
   let D := (-4, 0) in
   let AC := abs(6 - (-2)) in
   let BD := abs(3 - (-4)) in
   (1 / 2) * AC * BD = 28) :=
by
  sorry

end area_of_quadrilateral_ABCD_l220_220953


namespace probability_three_correct_deliveries_l220_220327

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220327


namespace isosceles_triangles_with_perimeter_21_l220_220086

theorem isosceles_triangles_with_perimeter_21 : 
  ∃ n : ℕ, n = 5 ∧ (∀ (a b c : ℕ), a ≤ b ∧ b = c ∧ a + 2*b = 21 → 1 ≤ a ∧ a ≤ 10) :=
sorry

end isosceles_triangles_with_perimeter_21_l220_220086


namespace sum_of_squares_of_distances_l220_220974

theorem sum_of_squares_of_distances {s : ℝ} (h : s = 1) : 
  let square_center := (1/2 : ℝ, 1/2 : ℝ) in 
  let vertices := {(0 : ℝ, 0 : ℝ), (1, 0), (1, 1), (0, 1)} in 
  ∀ (line : ℝ × ℝ → Prop), 
  (∀ (x y : ℝ), line (1/2, 1/2) → line (x, y)) → 
  let distances := (λ v : ℝ × ℝ, Classical.choose (Exists.intro (λ p : ℝ × ℝ, p ∈ line v) (sorry : ∃ p : ℝ × ℝ, p ∈ line v))) in
  (∑ v in vertices, (dist (distances v) v)^2) = 2 :=
sorry

end sum_of_squares_of_distances_l220_220974


namespace sum_of_midpoints_l220_220080

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l220_220080


namespace difference_students_l220_220517

variables {A B AB : ℕ}

theorem difference_students (h1 : A + AB + B = 800)
  (h2 : AB = 20 * (A + AB) / 100)
  (h3 : AB = 25 * (B + AB) / 100) :
  A - B = 100 :=
sorry

end difference_students_l220_220517


namespace find_a1_q_exists_k_l220_220496

-- Definitions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Conditions from the problem
variables (a : ℕ → ℝ) (q : ℝ) (k : ℝ)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_a5_eq_2_pow_9 : a 3 * a 4 * a 5 = 2^9

-- Conditions for sequence {b_n}
def b (n : ℕ) := (1 : ℝ) / n * (log (a 1) + log (a 2) + ... + log (a (n - 1)) + log (k * a n))

-- Questions translated to Lean
theorem find_a1_q : 
  a 1 = 1 ∧ q = 2 :=
sorry

theorem exists_k :
  ∃ k > 0, ∀ n, b n = 1 / n * 
  (log (a 1) + log (a 2) + ... + log (a (n - 1)) + log (k * a n)) → 
  b (n + 1) - b n = (1 : ℝ) / 2 :=
sorry

end find_a1_q_exists_k_l220_220496


namespace probability_no_overlap_correct_l220_220937

-- Define the structure and probabilities
structure PentagonalBipyramid where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)

-- Define the initial assumptions
def initial_conditions (P : PentagonalBipyramid) := 
  P.vertices = Finset.range 7 ∧
  ∀ v, v ∈ P.vertices → 
    Finset.card (Finset.filter (λ p, p.1 = v ∨ p.2 = v) P.edges) = 4

-- Define the movement of ants
def ant_move (P : PentagonalBipyramid) (pos : ℕ → ℕ) : ℕ → ℕ := 
  λ i, ite (i < 7) (some (P.edges.filter (λ e, e.1 = pos i ∨ e.2 = pos i))) sorry

-- Define the probability calculation
def probability_no_overlap (P : PentagonalBipyramid) (move : ℕ → ℕ) : ℚ := 
  let total_moves := (4 : ℚ) ^ 7 in
  let favorable_moves := 5 / 256 in
  favorable_moves / total_moves

-- The final theorem
theorem probability_no_overlap_correct :
  ∀ (P : PentagonalBipyramid) (pos : ℕ → ℕ), 
    initial_conditions P →
    probability_no_overlap P (ant_move P pos) = 5 / 256 := 
by
  intros
  unfold initial_conditions
  unfold probability_no_overlap
  sorry

end probability_no_overlap_correct_l220_220937


namespace meteor_shower_problem_l220_220978

theorem meteor_shower_problem (encounter_time_towards : ℝ) (encounter_time_same : ℝ)
  (h_towards : encounter_time_towards = 7)
  (h_same : encounter_time_same = 13) : 
  let H := 2 * encounter_time_towards * encounter_time_same / (encounter_time_towards + encounter_time_same) in
  H = 9.1 :=
by {
  rw [h_towards, h_same],
  simp [H],
  norm_num,
  sorry
}

end meteor_shower_problem_l220_220978


namespace find_angle_A_triangle_area_l220_220443

variable {A B C : Real} -- Angles
variable {a b c : Real} -- Sides

-- Problem A
theorem find_angle_A (h1 : (a - b) / c = (Real.sin B + Real.sin C) / (Real.sin B + Real.sin A)) :
  A = 2 * Real.pi / 3 :=
sorry

-- Problem B
theorem triangle_area (h1 : a = Real.sqrt 7)
                     (h2 : b = 2 * c)
                     (h3 : A = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end find_angle_A_triangle_area_l220_220443


namespace xanthia_hot_dogs_l220_220626

theorem xanthia_hot_dogs (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) :
  ∃ n m : ℕ, n * a = m * b ∧ n = 7 := by 
sorry

end xanthia_hot_dogs_l220_220626


namespace sum_six_consecutive_integers_l220_220550

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l220_220550


namespace sum_of_squares_of_3_consecutive_integers_sum_of_squares_of_6_consecutive_integers_sum_of_squares_of_11_consecutive_integers_is_square_l220_220939

-- Problem 1: Sum of the squares of 3 consecutive positive integers cannot be a square
theorem sum_of_squares_of_3_consecutive_integers (n : ℤ) (k : ℤ) : 
  (n-1)^2 + n^2 + (n+1)^2 ≠ k^2 :=
sorry

-- Problem 2: Sum of the squares of 6 consecutive positive integers cannot be a square
theorem sum_of_squares_of_6_consecutive_integers (n : ℤ) (k : ℤ) : 
  (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 ≠ k^2 :=
sorry

-- Problem 3: Sum of the squares of 11 consecutive positive integers from 18 to 28 is a square
theorem sum_of_squares_of_11_consecutive_integers_is_square : 
  ∑ i in finset.range (28 - 18 + 1), (18 + i)^2 = 77^2 :=
sorry

end sum_of_squares_of_3_consecutive_integers_sum_of_squares_of_6_consecutive_integers_sum_of_squares_of_11_consecutive_integers_is_square_l220_220939


namespace no_tetrahedron_obtuse_each_edge_l220_220531

/-- 
  Prove that there does not exist a tetrahedron in which each edge is a side of an obtuse plane angle.
-/
theorem no_tetrahedron_obtuse_each_edge :
  ¬ ∃ (T : Tetrahedron), 
    (∀ edge ∈ T.edges, ∃ triangle ∈ T.faces, 
      (edge ∈ triangle.edges ∧ 
        (exists obtuse_angle ∈ triangle.angles, obtuse_angle > 90 ∧ opposite_side = edge))) :=
sorry

end no_tetrahedron_obtuse_each_edge_l220_220531


namespace equilateral_triangle_angle_sum_l220_220493

noncomputable def equilateral_triangle (A B C : Type) (AB AC BC : A) : Prop :=
  AB = AC ∧ AC = BC

def divides_equally (A B D E : Type) (AB AD DE EB : A) : Prop :=
  AB = AD + DE + EB ∧ AD = DE ∧ DE = EB

def point_on_segment (F B C : Type) (BC CF : B) : Prop :=
  BC = CF + F

def angle_sum (ABCDEF : Type) (angle_CDF angle_CEF : ABCDEF) : Prop :=
  angle_CDF + angle_CEF = 30

theorem equilateral_triangle_angle_sum
  (A B C D E F : Type)
  (AB AC BC : A)
  (AD DE EB : B)
  (BC CF : C)
  (CDF CEF : ABCDEF) :
  equilateral_triangle A B C AB AC BC →
  divides_equally A B D E AB AD DE EB →
  point_on_segment F B C BC CF →
  angle_sum ABCDEF CDF CEF :=
begin
  sorry
end

end equilateral_triangle_angle_sum_l220_220493


namespace intersection_eq_range_of_a_l220_220777

noncomputable def A : set ℝ := {x | ∃ y, y = real.sqrt (3^x - 3)}
noncomputable def B : set ℝ := {x | ∃ y, y = real.sqrt (real.log x / real.log 0.5 + 1)}
def C (a : ℝ) : set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2 * a - 1}
def IntersectionAB : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Part (1)
theorem intersection_eq : (A ∩ B) = IntersectionAB := 
  sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : (C a ∪ IntersectionAB) = C a) : 2 ≤ a ∧ a ≤ 3 :=
  sorry

end intersection_eq_range_of_a_l220_220777


namespace expected_steps_to_color_interval_l220_220915

/-- Natalie colors the interval [0, 1] with certain rules. We aim to prove the expected value
    of the number of steps to color the entire interval black is 5 given the conditions. -/
theorem expected_steps_to_color_interval: 
  (expected_steps : ℝ) :=
begin
  -- Define the initial conditions
  let interval := set.Icc (0:ℝ) (1:ℝ),
  let interval_colored_initially := 0,
  let color_step := 
    λ x : ℝ, 
      if x ≤ 0.5 
      then (set.Icc x (x + 0.5))
      else (set.Icc x 1) ∪ (set.Icc 0 (x - 0.5)),

  -- Expected steps required by Natalie to fully color the interval [0, 1]
  have h1: expected_steps = 5, sorry,
end

end expected_steps_to_color_interval_l220_220915


namespace cost_for_15_pounds_of_apples_l220_220671

-- Axiom stating the cost of apples per weight
axiom cost_of_apples (pounds : ℕ) : ℕ

-- Condition given in the problem
def rate_apples : Prop := cost_of_apples 5 = 4

-- Statement of the problem
theorem cost_for_15_pounds_of_apples : rate_apples → cost_of_apples 15 = 12 :=
by
  intro h
  -- Proof to be filled in here
  sorry

end cost_for_15_pounds_of_apples_l220_220671


namespace mike_mowed_lawns_proof_l220_220513

def mike_mowed_lawns_money (M : ℕ) : Prop :=
  M + 26 = 5 * 8

theorem mike_mowed_lawns_proof : ∃ M : ℕ, mike_mowed_lawns_money M ∧ M = 14 :=
by
  use 14
  split
  . dsimp [mike_mowed_lawns_money]
    norm_num
  . norm_num

end mike_mowed_lawns_proof_l220_220513


namespace part_a_part_b_l220_220520

/-
Define conditions: board of 3x53, placing numbers 1 to 159 with positions and adjacency constraints.
-/

def board := Fin 3 × Fin 53

-- Definition for the adjacency
def adjacent (c1 c2 : board) : Prop :=
  c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c2.2 = c1.2 + 1) ∨
  c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c2.1 = c1.1 + 1)

-- Definition for valid placement of numbers
def valid_placement (f : Fin 159 → board) : Prop :=
  f 0 = (⟨2, sorry⟩ : board) ∧ ∀ a : Fin 158, adjacent (f a) (f (a + 1))

-- Definition for good cells
def good (f : Fin 159 → board) (c : board) : Prop :=
  ∃ a : Fin 158, c = f a ∧ adjacent c (f a.succ)

-- The largest number adjacent to 1 is 2
theorem part_a (f : Fin 159 → board) (h : valid_placement f) : 
  ∃ (c : board), adjacent (f 0) c ∧ f 1 = c := sorry

-- The number of good cells is 79
theorem part_b (f : Fin 159 → board) (h : valid_placement f) : 
  {c : board | good f c}.to_finset.card = 79 := sorry

end part_a_part_b_l220_220520


namespace average_infection_rate_infected_computers_exceed_700_l220_220161

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l220_220161


namespace correct_transformation_l220_220644

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l220_220644


namespace sum_of_logs_l220_220824

open Real

noncomputable def log_base (b a : ℝ) : ℝ := log a / log b

theorem sum_of_logs (x y z : ℝ)
  (h1 : log_base 2 (log_base 4 (log_base 5 x)) = 0)
  (h2 : log_base 3 (log_base 5 (log_base 2 y)) = 0)
  (h3 : log_base 4 (log_base 2 (log_base 3 z)) = 0) :
  x + y + z = 666 := sorry

end sum_of_logs_l220_220824


namespace three_correct_deliveries_probability_l220_220268

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220268


namespace number_of_intersections_l220_220066

   -- Definitions corresponding to conditions
   def C1 (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
   def C2 (a x y : ℝ) : Prop := y = a*x^2
   def positive_real (a : ℝ) : Prop := a > 0

   -- Final statement converting the question, conditions, and correct answer into Lean code
   theorem number_of_intersections (a : ℝ) (ha : positive_real a) :
     ∃ (count : ℕ), (count = 4) ∧
     (∀ x y : ℝ, C1 x y → C2 a x y → True) := sorry
   
end number_of_intersections_l220_220066


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220793

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220793


namespace race_length_l220_220448

open Real

-- Conditions
variables (v L : ℝ)
variables (speedA : ℝ := 4 * v)
variables (speedB : ℝ := v)
variables (speedC : ℝ := 2 * v)

-- Distances each runner covers
def distanceA := L + 60
def distanceB := L
def distanceC := L + 30

-- Time equations
def timeA := distanceA / speedA
def timeB := distanceB / speedB
def timeC := distanceC / speedC

-- Goal: Prove L = 60
theorem race_length : timeA = timeB ∧ timeA = timeC → L = 60 :=
by
    intros h
    sorry

end race_length_l220_220448


namespace sum_not_divisible_by_three_times_any_number_l220_220928

theorem sum_not_divisible_by_three_times_any_number (n : ℕ) (a : Fin n → ℕ) (h : n ≥ 3) (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k : Fin n, ¬ (a i + a j) ∣ (3 * a k)) :=
sorry

end sum_not_divisible_by_three_times_any_number_l220_220928


namespace probability_function_has_root_l220_220932

theorem probability_function_has_root : ∀ (a b : ℝ), (-π ≤ a ∧ a ≤ π) → (-π ≤ b ∧ b ≤ π) →
  (finset.Icc (-π:ℝ) π).product (finset.Icc (-π:ℝ) π)
    .measure (set_of (λ (p : ℝ × ℝ), p.1^2 + p.2^2 ≥ π)) =
  (3 / 4) * ((2 * π) * (2 * π)) :=
by
  sorry

end probability_function_has_root_l220_220932


namespace range_of_m_l220_220034

theorem range_of_m (m n : ℝ) (h1 : n = 2 / m) (h2 : n ≥ -1) :
  m ≤ -2 ∨ m > 0 := 
sorry

end range_of_m_l220_220034


namespace probability_exactly_three_correct_l220_220351

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220351


namespace meteor_shower_problem_l220_220979

theorem meteor_shower_problem (encounter_time_towards : ℝ) (encounter_time_same : ℝ)
  (h_towards : encounter_time_towards = 7)
  (h_same : encounter_time_same = 13) : 
  let H := 2 * encounter_time_towards * encounter_time_same / (encounter_time_towards + encounter_time_same) in
  H = 9.1 :=
by {
  rw [h_towards, h_same],
  simp [H],
  norm_num,
  sorry
}

end meteor_shower_problem_l220_220979


namespace integer_part_m_eq_3_l220_220750

def m (x : Real) : Real :=
  3^(Real.cos x ^ 2) + 3^(Real.sin x ^ 4)

theorem integer_part_m_eq_3 {x : Real} (hx : 0 < x ∧ x < Real.pi / 2) :
  Real.floor (m x) = 3 :=
sorry

end integer_part_m_eq_3_l220_220750


namespace pentagon_area_l220_220957

theorem pentagon_area (p q : ℕ) (area_pentagon : ℝ)
  (h1 : ∀ (seg_len : ℝ), seg_len = 3 → 15 * seg_len ∈ ℕ)
  (h2 : ∃ p q : ℕ, area_pentagon = sqrt p + sqrt q)
  (h3 : area_pentagon = sqrt 48) :
  p + q = 48 :=
sorry

end pentagon_area_l220_220957


namespace shell_weight_l220_220883

theorem shell_weight :
  let P_initial := 5
  let P_added := 12
  P_initial + P_added = 17 :=
by
  let P_initial := 5
  let P_added := 12
  show P_initial + P_added = 17 by
    sorry

end shell_weight_l220_220883


namespace root_polynomial_l220_220505

noncomputable def roots : ℂ := sorry

theorem root_polynomial (roots: ℂ) :
    (root.polynomial (x^4 - 4 * x^3 + 8 * x^2 - 7 * x + 3)) = roots ->
    (root.sum (roots) = 4) ->
    (root.sum_products (roots) 2 = 8) ->
  \[
  \frac{roots[1]^2}{roots[2]^2 + roots[3]^2 + roots[4]^2} + 
  \frac{roots[2]^2}{roots[1]^2 + roots[3]^2 + roots{4]^2} + 
  \frac{roots[3]^2}{roots[1]^2 + roots[2]^2 + + roots[4]^2} + 
  \frac{roots[4]^2}{roots[1]^2 + roots[2]^2 + + roots[3]^2} == -4
  sorry

end root_polynomial_l220_220505


namespace solve_tan_pi_x_eq_floor_log10_pix_minus_floor_log10_floor_pix_solution_l220_220383

def floor (a : ℝ) : ℤ := Int.floor a

theorem solve_tan_pi_x_eq_floor_log10_pix_minus_floor_log10_floor_pix (x : ℝ) (h : 1 ≤ Real.exp (x * Real.log pi)) :
  tan (π * x) = floor (Real.log10 (Real.exp (x * Real.log pi))) - floor (Real.log10 (floor (Real.exp (x * Real.log pi)))) :=
begin
  sorry,
end

theorem solution (x : ℝ) :
  (solve_tan_pi_x_eq_floor_log10_pix_minus_floor_log10_floor_pix x (begin
    have h₀ : Real.exp (x * Real.log pi) = π^x := by rw [Real.exp, Real.log];
    sorry,
  end)) →
  ∃ n : ℕ, x = n := by
  sorry

end solve_tan_pi_x_eq_floor_log10_pix_minus_floor_log10_floor_pix_solution_l220_220383


namespace three_correct_deliveries_probability_l220_220265

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220265


namespace sequence_general_formula_l220_220391

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  3^n + 1

theorem sequence_general_formula (a : ℕ → ℕ) (n : ℕ) (h₁ : sequence_sum n = ∑ i in Finset.range n, a i) :
  a n = if n = 1 then 4 else 2 * 3^(n - 1) := sorry

end sequence_general_formula_l220_220391


namespace john_total_shirts_l220_220882

-- Define initial conditions
def initial_shirts : ℕ := 12
def additional_shirts : ℕ := 4

-- Statement of the problem
theorem john_total_shirts : initial_shirts + additional_shirts = 16 := by
  sorry

end john_total_shirts_l220_220882


namespace minimum_value_of_S_l220_220748

def S (n : ℤ) : ℤ :=
  |n - 1| + 2 * |n - 2| + 3 * |n - 3| + 4 * |n - 4| + 5 * |n - 5| +
  6 * |n - 6| + 7 * |n - 7| + 8 * |n - 8| + 9 * |n - 9| + 10 * |n - 10|

theorem minimum_value_of_S: (∃ n : ℤ, n ∈ finset.range 4 ∧ S n = 112) ∧ (∀ n : ℤ, n ∈ finset.range 4 → S n ≥ 112) :=
sorry

end minimum_value_of_S_l220_220748


namespace probability_three_correct_deliveries_l220_220328

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220328


namespace projection_magnitude_is_neg_one_l220_220760

-- Define the vectors a and b and the condition of perpendicularity
def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (k, 3)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Define the magnitude of the projection
def projection_magnitude (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / Math.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- State the theorem
theorem projection_magnitude_is_neg_one : 
  ∀ (k : ℝ), is_perpendicular (a.1 + b k .1, a.2 + b k .2) a → projection_magnitude a (b k) = -1 :=
sorry

end projection_magnitude_is_neg_one_l220_220760


namespace neg_neg_eq_l220_220140

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l220_220140


namespace winner_won_by_288_votes_l220_220600

theorem winner_won_by_288_votes (V : ℝ) (votes_won : ℝ) (perc_won : ℝ) 
(h1 : perc_won = 0.60)
(h2 : votes_won = 864)
(h3 : votes_won = perc_won * V) : 
votes_won - (1 - perc_won) * V = 288 := 
sorry

end winner_won_by_288_votes_l220_220600


namespace table_relation_l220_220061

theorem table_relation :
  ∀ (x y : ℕ), (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) → y = x^2 :=
by
  intros x y h
  cases h with
  | inl h1 => cases h1 with; rw [←h_left, ←h_right]
  | inr h2 => cases h2 with
    | inl h3 => cases h3 with; rw [←h_left, ←h_right]
    | inr h4 => cases h4 with
      | inl h5 => cases h5 with; rw [←h_left, ←h_right]
      | inr h6 => cases h6 with
        | inl h7 => cases h7 with; rw [←h_left, ←h_right]
        | inr h8 => cases h8 with; rw [←h_left, ←h_right]
  simp
sorry

end table_relation_l220_220061


namespace length_CD_l220_220441

theorem length_CD
  {A B C D : Type}
  [point : Point A B C D]
  (triangle_ABC : Triangle A B C)
  (angle_ABC_eq : angle triangle_ABC B = 120 * 180/pi)
  (AB_eq : length (segment A B) = 3)
  (BC_eq : length (segment B C) = 4)
  (D_is_intersection : D_is_interception_pt D (perpendicular_from A B) (perpendicular_from C B)) :
  length (segment C D) = 10 / sqrt(3) :=
by
  sorry

end length_CD_l220_220441


namespace plant_supplier_earnings_l220_220185

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l220_220185


namespace pure_imaginary_solution_l220_220841

theorem pure_imaginary_solution (m : ℝ) (z : ℂ) (h1 : z = (m^2 - m - 2) + (m + 1) * complex.I) (h2 : z.im = z) : 
    m = 2 := 
by 
  sorry

end pure_imaginary_solution_l220_220841


namespace find_t_l220_220459

theorem find_t (t : ℝ) (θ : ℝ) (h1 : sin θ = t / Real.sqrt (4 + t^2))
  (h2 : cos θ = -2 / Real.sqrt (4 + t^2))
  (h3 : sin θ + cos θ = Real.sqrt 5 / 5) :
  t = 4 := 
begin
  sorry,
end

end find_t_l220_220459


namespace smallest_arith_prog_l220_220262

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end smallest_arith_prog_l220_220262


namespace arun_age_l220_220838

theorem arun_age (A G M : ℕ) (h1 : (A - 6) / 18 = G) (h2 : G = M - 2) (h3 : M = 5) : A = 60 :=
by
  sorry

end arun_age_l220_220838


namespace probability_three_correct_deliveries_is_one_sixth_l220_220318

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220318


namespace min_phi_shift_symmetric_l220_220959

theorem min_phi_shift_symmetric :
  ∃ φ > 0, ∀ x, (∀ x', (sqrt 2) * sin (x' + (π/4) + φ) = -(sqrt 2) * sin ((-x') + (π/4) + φ)) ↔ φ = (3 * π / 4) :=
by sorry

end min_phi_shift_symmetric_l220_220959


namespace probability_of_three_correct_packages_l220_220297

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220297


namespace sum_binom_eq_binom_l220_220743

theorem sum_binom_eq_binom (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range (n + 1), 2^k * (Nat.choose n k) * 
  (Nat.choose (n - k) (Nat.floor ((n - k) / 2)))) = Nat.choose (2 * n + 1) n :=
by
  sorry

end sum_binom_eq_binom_l220_220743


namespace number_of_boys_in_school_l220_220197

theorem number_of_boys_in_school (total_students : ℕ) (sample_size : ℕ) 
(number_diff : ℕ) (ratio_boys_sample_girls_sample : ℚ) : 
total_students = 1200 → sample_size = 200 → number_diff = 10 →
ratio_boys_sample_girls_sample = 105 / 95 →
∃ (boys_in_school : ℕ), boys_in_school = 630 := by 
  sorry

end number_of_boys_in_school_l220_220197


namespace probability_of_three_correct_packages_l220_220302

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220302


namespace polynomial_factorization_evaluation_at_3_sum_l220_220017

-- Define the polynomial to be factored and the evaluation asked for
noncomputable def poly : ℤ[X] := X^5 - 2 * X^3 - X^2 - 2

def q1 : ℤ[X] := X + 1
def q2 : ℤ[X] := X^2 - X + 1
def q3 : ℤ[X] := X^2 - 2

-- The factorization of the polynomial
theorem polynomial_factorization :
  poly = q1 * q2 * q3 := by 
  sorry

-- Evaluating each factor at y = 3 and summing them up
theorem evaluation_at_3_sum :
  (q1.eval 3) + (q2.eval 3) + (q3.eval 3) = 18 := by 
  sorry

end polynomial_factorization_evaluation_at_3_sum_l220_220017


namespace triangle_right_l220_220850

variable {a b c : ℝ}

theorem triangle_right (h : a * Real.cos (b / a * Real.pi) + a * Real.cos (c / a * Real.pi) = b + c) :
  a^2 = b^2 + c^2 := by
sorrrrrry

end triangle_right_l220_220850


namespace sign_of_f_based_on_C_l220_220895

def is_triangle (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem sign_of_f_based_on_C (a b c : ℝ) (R r : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) 
  (h3 : c = 2 * R * Real.sin C)
  (h4 : r = 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2))
  (h5 : A + B + C = Real.pi)
  (h_triangle : is_triangle a b c)
  : (a + b - 2 * R - 2 * r > 0 ↔ C < Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r = 0 ↔ C = Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r < 0 ↔ C > Real.pi / 2) :=
sorry

end sign_of_f_based_on_C_l220_220895


namespace trapezoid_angles_l220_220473

theorem trapezoid_angles (ABCD : trapezoid)
  (M : midpoint CD)
  (h1 : divides_angle BD BM ABC 3)
  (h2 : angle_bisector AC BAD) :
  angle A = 72 ∧ angle B = 108 ∧ angle C = 54 ∧ angle D = 126 := 
sorry

end trapezoid_angles_l220_220473


namespace miles_driven_on_monday_l220_220007

variables (monday miles_tuesday miles_wednesday miles_thursday miles_friday total_reimbursement reimbursement_rate total_miles : ℝ)

-- Define the conditions
def miles_tuesday := 26
def miles_wednesday := 20 
def miles_thursday := 20
def miles_friday := 16
def reimbursement_rate := 0.36
def total_reimbursement := 36
def total_miles := total_reimbursement / reimbursement_rate

-- Proving the number of miles driven on Monday
theorem miles_driven_on_monday : 
  monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday = total_miles →
  monday = 18 := 
by 
  sorry

end miles_driven_on_monday_l220_220007


namespace distance_from_origin_l220_220450

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end distance_from_origin_l220_220450


namespace number_of_polygons_with_odd_exterior_angles_l220_220189

theorem number_of_polygons_with_odd_exterior_angles
  (n : ℕ) (h1 : n ≥ 4)
  (h2 : ∀ (i j : ℕ), i ≠ j → (360 / n) % 2 = 1 → (360/n : ℕ) = 2*i+1 ∨ (360/n : ℕ) = 4*i + 1 ∨ (360/n : ℕ) = 6*i + 1 ∨ (360/n : ℕ) = 8*i + 1 ∨ (360/n : ℕ) = 10*i + 1) :
  ∃ i ∈ {1, 3, 5, 9, 15, 45}, (360 / i = n) :=
  sorry

end number_of_polygons_with_odd_exterior_angles_l220_220189


namespace g_lt_one_l220_220396

noncomputable def f (x a : ℝ) : ℝ := x - 2 / x - a * (Real.log x - 1 / x^2)

def g (a : ℝ) : ℝ := a - a * (Real.log a) - 1 / a

theorem g_lt_one (a : ℝ) (ha : a > 0) : g(a) < 1 := 
sorry

end g_lt_one_l220_220396


namespace remainder_prod_mod_7_l220_220714

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220714


namespace correct_option_is_D_l220_220116

variable (a : ℝ)

def A := (a^2 / a^8) = a^(-4)
def B := (a * a^2) = a^2
def C := (a^3)^2 = a^9
def D := (3 * a)^3 = 27 * a^3

theorem correct_option_is_D : D := by 
  unfold D -- unfolds the definition of D
  calculate -- triggers the mathematical calculations which proves D

end correct_option_is_D_l220_220116


namespace length_MN_l220_220873

-- Define the conditions of the problem
variable (A B C D M N E : Type)
variable [BC_parallel_AD : Parallel B C D]
variable [BC_length : Length B C = 900]
variable [AD_length : Length A D = 1800]
variable [angle_A_45 : Angle A = 45]
variable [angle_D_45 : Angle D = 45]
variable [M_midpoint : Midpoint M B C]
variable [N_midpoint : Midpoint N A D]

-- The theorem stating the result
theorem length_MN : Length M N = 450 := by
  sorry

end length_MN_l220_220873


namespace cos_double_angle_l220_220395

theorem cos_double_angle (x y r : ℕ) (h1 : x = 3) (h2 : y = 4) (h3 : r = 5) : 
  let cos_alpha := (x : ℚ) / r in
  let cos_2alpha := 2 * cos_alpha^2 - 1 in
  cos_2alpha = -7 / 25 := 
by
  sorry

end cos_double_angle_l220_220395


namespace velocity_zero_at_0_4_8_l220_220188

def distance (t : ℝ) : ℝ :=
  (1/4) * t^4 - 4 * t^3 + 16 * t^2

def velocity (t : ℝ) : ℝ :=
  derivative (distance t)

theorem velocity_zero_at_0_4_8 :
  ∀ (t : ℝ), velocity t = 0 ↔ t = 0 ∨ t = 4 ∨ t = 8 :=
by
  sorry

end velocity_zero_at_0_4_8_l220_220188


namespace product_mod_seven_l220_220705

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220705


namespace race_time_l220_220445

theorem race_time (t : ℝ) (h1 : 100 / t = 66.66666666666667 / 45) : t = 67.5 :=
by
  sorry

end race_time_l220_220445


namespace parabola_locus_l220_220010

variables (a c k : ℝ) (a_pos : 0 < a) (c_pos : 0 < c) (k_pos : 0 < k)

theorem parabola_locus :
  ∀ t : ℝ, ∃ x y : ℝ,
    x = -kt / (2 * a) ∧ y = - k^2 * t^2 / (4 * a) + c ∧
    y = - (k^2 / (4 * a)) * x^2 + c :=
sorry

end parabola_locus_l220_220010


namespace three_correct_deliveries_probability_l220_220269

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220269


namespace pos_product_inequality_l220_220929

theorem pos_product_inequality (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) :
  (∏ i : Fin n, 1 + (a i)^2 / a ((i + 1) % n)) 
  ≥ (∏ i : Fin n, 1 + a i) := 
sorry

end pos_product_inequality_l220_220929


namespace smallest_arith_prog_term_l220_220264

-- Define the conditions of the problem as a structure
structure ArithProgCondition (a d : ℝ) :=
  (sum_of_squares : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (sum_of_cubes : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)

-- Define the theorem we want to prove
theorem smallest_arith_prog_term :
  ∃ (a d : ℝ), ArithProgCondition a d ∧ (a = 0 ∧ (d = sqrt 7 ∨ d = -sqrt 7) → -2 * sqrt 7) := 
sorry

end smallest_arith_prog_term_l220_220264


namespace perimeter_eq_2sqrt10_add_4_max_value_eq_sqrt21_over_3_l220_220476

-- Defining the triangle with given conditions
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (altitude_from_AB : ℝ)
  (area : ℝ)

-- Given conditions
axiom triangle_ABC : Triangle
axiom angle_C_eq_pi_over_3 : triangle_ABC.C = Real.pi / 3
axiom altitude_from_AB_eq_sqrt3 : triangle_ABC.altitude_from_AB = Real.sqrt 3
axiom area_eq_2sqrt3 : triangle_ABC.area = 2 * Real.sqrt 3

-- Problem 1: Perimeter
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem perimeter_eq_2sqrt10_add_4 :
  perimeter triangle_ABC = 2 * Real.sqrt 10 + 4 :=
sorry

-- Problem 2: Maximum value of (2/a + 1/b)
noncomputable def value (t : Triangle) : ℝ :=
  (2 / t.a) + (1 / t.b)

theorem max_value_eq_sqrt21_over_3 :
  (∃ A B, triangle_ABC.A + triangle_ABC.B + triangle_ABC.C = Real.pi)
  ∧ angle_C_eq_pi_over_3
  ∧ altitude_from_AB_eq_sqrt3
  ∧ area_eq_2sqrt3
  ∧ (∀ x, value x ≤ Real.sqrt 21 / 3) := sorry

end perimeter_eq_2sqrt10_add_4_max_value_eq_sqrt21_over_3_l220_220476


namespace determine_cards_per_friend_l220_220943

theorem determine_cards_per_friend (n_cards : ℕ) (n_friends : ℕ) (h : n_cards = 12) : n_friends > 0 → (n_cards / n_friends) = (12 / n_friends) :=
by
  sorry

end determine_cards_per_friend_l220_220943


namespace Evan_books_proof_l220_220698

variable (E_current E_future : ℕ)

def Evan_books_conditions : Prop :=
  E_current = 200 - 40 ∧
  E_future = 860

theorem Evan_books_proof (h : Evan_books_conditions E_current E_future) :
  E_future - 5 * E_current = 60 :=
by
  obtain ⟨h1, h2⟩ := h
  have h3: E_current = 160 := h1
  have h4: E_future = 860 := h2
  calc
    E_future - 5 * E_current
        = 860 - 5 * 160 : by rw [h3, h4]
    _   = 60 : by norm_num

end Evan_books_proof_l220_220698


namespace constant_term_is_60_l220_220869

-- Define the binomial coefficient
noncomputable def binom (n k : ℕ) := Nat.choose n k

-- Define the general term for the expansion of (x + 1/x)^6
def general_term (r : ℕ) := binom 6 r * (3 : ℤ)^(6 - 2 * r)

-- Calculate the constant term for (3 + x)(x + 1/x)^6
def constant_term := 3 * general_term 3

-- Prove the constant term is 60
theorem constant_term_is_60 : constant_term = 60 := by
  sorry

end constant_term_is_60_l220_220869


namespace original_number_l220_220123

theorem original_number (x : ℝ) (h : 1.4 * x = 700) : x = 500 :=
sorry

end original_number_l220_220123


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220820

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220820


namespace count_valid_x_l220_220234

def my_oplus (a b : ℕ) : ℕ := a^3 / b

theorem count_valid_x :
  {x : ℕ | my_oplus 5 x > 0}.toFinset.card = 4 :=
by 
  sorry

end count_valid_x_l220_220234


namespace min_value_of_f_l220_220251

def f (x : ℝ) : ℝ :=
  x + (1 / x) + (x + (1 / x))^2

theorem min_value_of_f : ∀ (x : ℝ), (0 < x) → f x ≥ 6 :=
by
  intro x hx
  sorry

end min_value_of_f_l220_220251


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220812

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220812


namespace yasmin_children_l220_220483

-- Definitions based on the conditions
def has_twice_children (children_john children_yasmin : ℕ) := children_john = 2 * children_yasmin
def total_grandkids (children_john children_yasmin : ℕ) (total : ℕ) := total = children_john + children_yasmin

-- Problem statement in Lean 4
theorem yasmin_children (children_john children_yasmin total_grandkids : ℕ) 
  (h1 : has_twice_children children_john children_yasmin) 
  (h2 : total_grandkids children_john children_yasmin 6) : 
  children_yasmin = 2 :=
by
  sorry

end yasmin_children_l220_220483


namespace sum_of_powers_l220_220640

noncomputable def imaginary_unit : ℂ := Complex.i

theorem sum_of_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) 
  (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8 + 9 * i^9 = 4 + 5 * i :=
by
  sorry

end sum_of_powers_l220_220640


namespace largest_option_is_C_l220_220118

noncomputable def option_A : ℝ := real.sqrt (real.sqrt ((5 + 6)^2)^(1/3))
noncomputable def option_B : ℝ := real.sqrt ((6 + 1) * (5^(1/3)))
noncomputable def option_C : ℝ := real.sqrt ((5 + 2) * (6^(1/3)))
noncomputable def option_D : ℝ := (8 * (6^(1/2)))^(1/3)
noncomputable def option_E : ℝ := (10 * (5^(1/2)))^(1/3)

theorem largest_option_is_C : option_C ≥ option_A ∧ option_C ≥ option_B ∧ option_C >= option_D ∧ option_C ≥ option_E := 
by
  sorry

end largest_option_is_C_l220_220118


namespace pens_per_student_l220_220478

theorem pens_per_student (n_students n_pens_taken_per_student : ℕ) 
                         (monthly_taken : list ℕ) 
                         (total_pens initial_red_pens initial_black_pens : ℕ) 
                         (h1 : n_students = 6) 
                         (h2 : initial_red_pens = 85) 
                         (h3 : initial_black_pens = 92) 
                         (h4 : total_pens = n_students * (initial_red_pens + initial_black_pens)) 
                         (h5 : monthly_taken.sum = 336) 
                         (h6 : total_pens - monthly_taken.sum = n_pens_taken_per_student * n_students) :
  n_pens_taken_per_student = 121 :=
begin
  sorry
end

end pens_per_student_l220_220478


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220802

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220802


namespace largest_median_is_eight_l220_220111

theorem largest_median_is_eight (y : ℤ) : 
  let s := [y, 3 * y, 6, 4, 8].sort (≤) in
  s.nth 2 = some 8 :=
sorry

end largest_median_is_eight_l220_220111


namespace vasya_password_combinations_l220_220998

theorem vasya_password_combinations : 
  let D := {0, 1, 3, 4, 5, 6, 7, 8, 9} in
  let valid_passwords := {p : Finset (Fin 10) | 
                             p.val < 10000 ∧ 
                             ∀ i, p.val.digits 10 (i) ∉ {2} ∧ 
                             (∀ i < 3, p.val.digits 10 i ≠ p.val.digits 10 (i + 1)) ∧ 
                             p.val.digits 10 0 = p.val.digits 10 3
  } in
  valid_passwords.card = 504 :=
by
  sorry

end vasya_password_combinations_l220_220998


namespace symmetric_points_tangent_line_l220_220757

theorem symmetric_points_tangent_line (k : ℝ) (hk : 0 < k) :
  (∃ P Q : ℝ × ℝ, P.2 = Real.exp P.1 ∧ ∃ x₀ : ℝ, 
    Q.2 = k * Q.1 ∧ Q = (P.2, P.1) ∧ 
    Q.1 = x₀ ∧ k = 1 / x₀ ∧ x₀ = Real.exp 1) → k = 1 / Real.exp 1 := 
by 
  sorry

end symmetric_points_tangent_line_l220_220757


namespace product_mod_7_zero_l220_220729

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220729


namespace dryer_less_than_washing_machine_by_30_l220_220782

-- Definitions based on conditions
def washing_machine_price : ℝ := 100
def discount_rate : ℝ := 0.10
def total_paid_after_discount : ℝ := 153

-- The equation for price of the dryer
def original_dryer_price (D : ℝ) : Prop :=
  washing_machine_price + D - discount_rate * (washing_machine_price + D) = total_paid_after_discount

-- The statement we need to prove
theorem dryer_less_than_washing_machine_by_30 (D : ℝ) (h : original_dryer_price D) :
  washing_machine_price - D = 30 :=
by 
  sorry

end dryer_less_than_washing_machine_by_30_l220_220782


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220796

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220796


namespace slopes_angle_l220_220590

theorem slopes_angle (k_1 k_2 : ℝ) (θ : ℝ) 
  (h1 : 6 * k_1^2 + k_1 - 1 = 0)
  (h2 : 6 * k_2^2 + k_2 - 1 = 0) :
  θ = π / 4 ∨ θ = 3 * π / 4 := 
by sorry

end slopes_angle_l220_220590


namespace neg_neg_eq_pos_l220_220135

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l220_220135


namespace range_of_x_l220_220074

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  1 - x > 0

theorem range_of_x (x : ℝ) : range_of_independent_variable x → x < 1 :=
by sorry

end range_of_x_l220_220074


namespace arrangement_possible_l220_220878

noncomputable def exists_a_b : Prop :=
  ∃ a b : ℝ, a + 2*b > 0 ∧ 7*a + 13*b < 0

theorem arrangement_possible : exists_a_b := by
  sorry

end arrangement_possible_l220_220878


namespace find_n_l220_220403

noncomputable def a : ℕ → ℕ
| 0     := 5
| 1     := 13
| (n+2) := 5 * a (n+1) - 6 * a n

def S (n : ℕ) : ℕ := (Finset.range n).sum a

theorem find_n (n : ℕ) (h₁ : a 0 = 5) (h₂ : a 1 = 13) (h₃ : ∀ n, a(n+2) = 5 * a(n+1) - 6 * a(n)) :
  ∃ n, S n ≥ 2016 :=
by
  sorry

end find_n_l220_220403


namespace not_both_squares_l220_220898

theorem not_both_squares (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : 
  ¬ (Nat.is_square (p^2 + q) ∧ Nat.is_square (p + q^2)) :=
by
  sorry

end not_both_squares_l220_220898


namespace sum_of_midpoints_y_coordinates_l220_220078

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l220_220078


namespace tangents_product_equality_l220_220889

variables {α : Type*} [linear_ordered_field α]

-- Let A, B, C be points of the triangle ABC with circumcircle Γ.
-- Let I be the incenter of the triangle ABC.
variables {A B C I D E F R S T : Point α}

-- Assume AI, BI, and CI intersect $\Gamma$ at D, E, F respectively.
-- Assume the tangents to $\Gamma$ at F, D, E intersect AI, BI, and CI at R, S, T respectively.
-- Define the incidence structure for the points in terms of lines and intersection conditions.
variables (AI BI CI : Line α) (Γ : Circle α)

-- Definitions for the intersections
def intersects_circumcircle (line : Line α) (point1 point2 : Point α) : Prop :=
  is_on_circle point1 Γ ∧ is_on_circle point2 Γ ∧ point1 ≠ point2 ∧ is_on_line point1 line ∧ is_on_line point2 line

def tangent_intersects_line (circle_point line_point : Point α) : Prop :=
  is_tangent_to circle_point Γ ∧ is_on_line line_point (line circle_point I)

-- The statement to prove
theorem tangents_product_equality :
  intersects_circumcircle AI A D →
  intersects_circumcircle BI B E →
  intersects_circumcircle CI C F →
  tangent_intersects_line F R →
  tangent_intersects_line D S →
  tangent_intersects_line E T →
  |distance A R| * |distance B S| * |distance C T| = |distance I D| * |distance I E| * |distance I F| :=
sorry

end tangents_product_equality_l220_220889


namespace percentage_increase_surface_area_l220_220432

theorem percentage_increase_surface_area :
  let L : ℝ := 1 -- original edge length can be assumed as any positive number, here set to 1 for simplicity
  let orig_surface_area := 6 * L^2
  let new_edge_A := 1.10 * L
  let new_edge_B := 1.15 * L
  let new_edge_C := 1.20 * L
  let new_surface_area := 
    2 * (new_edge_B * new_edge_C) + 
    2 * (new_edge_A * new_edge_C) + 
    2 * (new_edge_A * new_edge_B)
  let percentage_increase := 
    ((new_surface_area - orig_surface_area) / orig_surface_area) * 100
  percentage_increase ≈ 32.17 := 
by
  simp [orig_surface_area, new_edge_A, new_edge_B, new_edge_C, new_surface_area, percentage_increase]
  have h : new_surface_area = 7.93 * L^2 := by
    dsimp [new_surface_area, new_edge_A, new_edge_B, new_edge_C]
    ring
  sorry

end percentage_increase_surface_area_l220_220432


namespace probability_three_correct_deliveries_l220_220339

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220339


namespace arithmetic_sequence_geometric_ratio_sum_sn_l220_220463

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℕ)
  (ha2 : a 2 = 5) (ha_sum : a 1 + a 4 = 12) :
  a n = 2 * n + 1 := 
  sorry

theorem geometric_ratio (n : ℕ) (b : ℕ → ℕ)
  (hb_pos : ∀ n, 0 < b n) (hb_prod : ∀ n, b n * b (n + 1) = 2^(2 * n + 1)) :
  ∃ q, q = 2 := 
  sorry

theorem sum_sn (n : ℕ) (a b : ℕ → ℕ)
  (ha : ∀ n, a n = 2 * n + 1) (hb : ∀ n, b n = 2^n) :
  (∑ i in Finset.range n, a i + b i) = n^2 + 2 * n + 2^(n+1) - 2 :=
  sorry

end arithmetic_sequence_geometric_ratio_sum_sn_l220_220463


namespace work_completion_time_l220_220127

theorem work_completion_time
  (W : ℝ) -- Total work
  (p_rate : ℝ := W / 40) -- p's work rate
  (q_rate : ℝ := W / 24) -- q's work rate
  (work_done_by_p_alone : ℝ := 8 * p_rate) -- Work done by p in first 8 days
  (remaining_work : ℝ := W - work_done_by_p_alone) -- Remaining work after 8 days
  (combined_rate : ℝ := p_rate + q_rate) -- Combined work rate of p and q
  (time_to_complete_remaining_work : ℝ := remaining_work / combined_rate) -- Time to complete remaining work
  : (8 + time_to_complete_remaining_work) = 20 :=
by
  sorry

end work_completion_time_l220_220127


namespace probability_three_correct_packages_l220_220305

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220305


namespace sum_of_solutions_is_nine_l220_220509

-- Definitions of the conditions
variable (x y z w : ℂ)
variable (solutions : Set (ℂ × ℂ × ℂ × ℂ))
variable H1 : ∀ (sol : ℂ × ℂ × ℂ × ℂ), sol ∈ solutions → (sol.1 + sol.2.1 * sol.2.2) = 8
variable H2 : ∀ (sol : ℂ × ℂ × ℂ × ℂ), sol ∈ solutions → (sol.2.1 + sol.1 * sol.2.2.1.2) = 11
variable H3 : ∀ (sol : ℂ × ℂ × ℂ × ℂ), sol ∈ solutions → (sol.2.2.1.1 + sol.1 * sol.2.1) = 11
variable H4 : ∀ (sol : ℂ × ℂ × ℂ × ℂ), sol ∈ solutions → (sol.2.2.2 + sol.2.1 * sol.2.2.1.1) = 5

-- Prove that the sum of all x_i values in the solutions is 9
theorem sum_of_solutions_is_nine : (∑ sol in solutions, sol.1) = 9 := 
by
  sorry

end sum_of_solutions_is_nine_l220_220509


namespace evaluate_sets_are_equal_l220_220212

theorem evaluate_sets_are_equal :
  (-3^5) = ((-3)^5) ∧
  ¬ ((-2^2) = ((-2)^2)) ∧
  ¬ ((-4 * 2^3) = (-4^2 * 3)) ∧
  ¬ ((- (-3)^2) = (- (-2)^3)) :=
by
  sorry

end evaluate_sets_are_equal_l220_220212


namespace john_total_time_spent_l220_220004

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end john_total_time_spent_l220_220004


namespace coefficient_of_xy6_eq_one_l220_220840

theorem coefficient_of_xy6_eq_one (a : ℚ) (h : (7 : ℚ) * a = 1) : a = 1 / 7 :=
by sorry

end coefficient_of_xy6_eq_one_l220_220840


namespace speed_in_still_water_l220_220122

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (hs1 : upstream_speed = 25) (hs2 : downstream_speed = 37) : 
  (upstream_speed + downstream_speed) / 2 = 31 :=
by
  rw [hs1, hs2]
  norm_num
  sorry

end speed_in_still_water_l220_220122


namespace triangle_inequality_l220_220894

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 :=
sorry

end triangle_inequality_l220_220894


namespace fractional_parts_equal_implies_integer_l220_220888

open Real

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem fractional_parts_equal_implies_integer (n : ℕ) (hn : n ≥ 3) (x : ℝ)
  (h1 : fractional_part x = fractional_part (x^2))
  (h2 : fractional_part x = fractional_part (x^n)) :
  x ∈ ℤ := 
sorry

end fractional_parts_equal_implies_integer_l220_220888


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220803

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220803


namespace paper_length_l220_220561

theorem paper_length :
  ∃ (L : ℝ), (2 * (11 * L) = 2 * (8.5 * 11) + 100 ∧ L = 287 / 22) :=
sorry

end paper_length_l220_220561


namespace problem_given_conditions_l220_220765

noncomputable theory
open Real

-- Define the polynomial function
def f (a b c x : ℝ) : ℝ := -x^3 + a * x^2 + b * x + c

-- Define the derivative of the polynomial function
def f' (a b c x : ℝ) : ℝ := -3 * x^2 + 2 * a * x + b

-- The given problem's proof statement
theorem problem_given_conditions (a b c : ℝ)
  (h1 : ∀ x < 0, f a b c x < f a b c 0)
  (h2 : ∀ x ∈ Ioc 0 1, f a b c x > f a b c 0)
  (h3 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ f a b c r1 = 0 ∧ f a b c r2 = 0 ∧ f a b c r3 = 0)
  (h4 : f a b c 1 = 0) :
  (b = 0) ∧ (f a 0 (1 - a) 2 > -5/2) :=
by
  sorry

end problem_given_conditions_l220_220765


namespace _l220_220577

noncomputable def pyramid_theorem (S1 S2 S3 Q : ℝ) (α β γ : ℝ) 
  (h_plane_angles : ∀ (D : Point), angle_at_apex D = 90°)
  (h_areas : ∀ (S1 : Area), S1 = area_face ABD)
  (h_areas : ∀ (S2 : Area), S2 = area_face BCD)
  (h_areas : ∀ (S3 : Area), S3 = area_face CAD)
  (h_base_area : ∀ (Q : Area), Q = area_base ABC)
  (h_dihedral_angles : ∀ (α : Angle), α = dihedral_angle AB)
  (h_dihedral_angles : ∀ (β : Angle), β = dihedral_angle BC)
  (h_dihedral_angles : ∀ (γ : Angle), γ = dihedral_angle AC)
  : Prop :=
  (cos α = S1 / Q) ∧ 
  (cos β = S2 / Q) ∧ 
  (cos γ = S3 / Q) ∧ 
  (S1^2 + S2^2 + S3^2 = Q^2) ∧ 
  (cos (2 * α) + cos (2 * β) + cos (2 * γ) = 1)

end _l220_220577


namespace village_food_sales_l220_220102

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l220_220102


namespace photocopy_cost_l220_220056

def costOfOnePhotocopy (x : ℝ) : Prop :=
  -- Condition 1: The combined savings is $0.80
  let combined_savings := 0.80 in
  -- Condition 2: Full price for 160 copies
  let full_price := 160 * x in
  -- Condition 3: Discounted cost for 160 copies
  let discounted_cost := 0.75 * full_price in
  -- The equation given by the difference in full and discounted price should match the savings
  full_price - discounted_cost = combined_savings

theorem photocopy_cost :
  costOfOnePhotocopy 0.02 :=
by
  -- The actual proof will go here
  sorry

end photocopy_cost_l220_220056


namespace palindromic_count_l220_220858

def reverse_string (s : String) : String :=
  s.reverse

def is_palindrome (s : String) : Bool :=
  s = reverse_string s

def sequence_a : Nat → String
| 1         => "A"
| 2         => "B"
| (n + 3)   => (sequence_a (n + 2)) ++ (reverse_string (sequence_a (n + 1)))

noncomputable def count_palindromes (n : Nat) : Nat :=
  (List.range n).countp (fun i => is_palindrome (sequence_a (i + 1)))

theorem palindromic_count : count_palindromes 1000 = 667 :=
  sorry

end palindromic_count_l220_220858


namespace sum_of_second_and_third_largest_numbers_l220_220479

theorem sum_of_second_and_third_largest_numbers : 
  ∃ x y z : ℕ, (x = 681 ∧ y = 816 ∧ z = 861 ∧ x + y = 1497) :=
by
  use 681, 816, 861
  split; try { rfl }
  split; try { rfl }
  split; try { rfl }
  sorry

end sum_of_second_and_third_largest_numbers_l220_220479


namespace shape_is_cylinder_l220_220739

noncomputable def shape_desc (r θ z a : ℝ) : Prop := r = a

theorem shape_is_cylinder (a : ℝ) (h_a : a > 0) :
  ∀ (r θ z : ℝ), shape_desc r θ z a → ∃ c : Set (ℝ × ℝ × ℝ), c = {p : ℝ × ℝ × ℝ | ∃ θ z, p = (a, θ, z)} :=
by
  sorry

end shape_is_cylinder_l220_220739


namespace product_mod_7_zero_l220_220734

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220734


namespace option_B_correct_l220_220364

theorem option_B_correct (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) : 
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by 
  have hc1_pos : (c^2 + 1) > 0 := by linarith
  sorry

end option_B_correct_l220_220364


namespace probability_three_correct_packages_l220_220311

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220311


namespace chewbacca_gum_packs_l220_220225

theorem chewbacca_gum_packs (x : ℝ) :
  (45 - x) / 55 = 45 / (55 + 4 * x) → x = 31.25 :=
by
  intro h
  have h1 : (45 - x) * (55 + 4 * x) = 45 * 55, from sorry
  have h2 : 2475 + 180 * x - 55 * x - 4 * x ^ 2 = 2475, from sorry
  have h3 : 125 * x - 4 * x ^ 2 = 0, from sorry
  have h4 : x * (125 - 4 * x) = 0, from sorry
  have h5 : x = 0 ∨ x = 31.25, from sorry
  exact (or.resolve_left h5 (by linarith))

end chewbacca_gum_packs_l220_220225


namespace gcd_of_powers_specific_gcd_2_1020_1031_l220_220616

theorem gcd_of_powers (a : ℕ) (k m : ℕ) : Nat.gcd (a^k - 1) (a^m - 1) = a^(Nat.gcd k m) - 1 := by
  sorry

theorem specific_gcd_2_1020_1031 : Nat.gcd (2^1020 - 1) (2^1031 - 1) = 1 := by
  have h := gcd_of_powers 2 1020 1031
  rw [Nat.gcd_eq_left_iff, Nat.gcd_comm] at h
  exact h
  sorry

end gcd_of_powers_specific_gcd_2_1020_1031_l220_220616


namespace zoo_visitors_sunday_l220_220030

-- Definitions based on conditions
def friday_visitors : ℕ := 1250
def saturday_multiplier : ℚ := 3
def sunday_decrease_percent : ℚ := 0.15

-- Assert the equivalence
theorem zoo_visitors_sunday : 
  let saturday_visitors := friday_visitors * saturday_multiplier
  let sunday_visitors := saturday_visitors * (1 - sunday_decrease_percent)
  round (sunday_visitors : ℚ) = 3188 :=
by
  sorry

end zoo_visitors_sunday_l220_220030


namespace tan_of_second_quadrant_angle_l220_220825

noncomputable def tan_value (θ : ℝ) (h1 : Real.sin θ = 4/5) (h2 : π/2 < θ ∧ θ < π) : Real :=
Real.tan θ

theorem tan_of_second_quadrant_angle : 
  ∀ (θ : ℝ), Real.sin θ = 4/5 ∧ (π/2 < θ ∧ θ < π) → tan_value θ (by simp) (by simp) = -4/3 :=
by
  intro θ h
  sorry

end tan_of_second_quadrant_angle_l220_220825


namespace xy_equals_nine_l220_220426

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l220_220426


namespace sin_alpha_value_l220_220380

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Ioo (-π / 2) (π / 2))
  (h2 : tan α = Real.sin (76 * π / 180) * Real.cos (46 * π / 180) - Real.cos (76 * π / 180) * Real.sin (46 * π / 180)) :
  Real.sin α = sqrt 5 / 5 :=
by
  sorry

end sin_alpha_value_l220_220380


namespace integral_sol_l220_220635

noncomputable def integrand := (λ x : ℝ, (x^3 - 6 * x^2 + 14 * x - 4) / ((x + 2) * (x - 2)^3))

theorem integral_sol : ∀ (x : ℝ), 
  ∫ (t : ℝ) in 0..x, integrand t = log (abs (x + 2)) - (1 / (x - 2)^2) + C :=
sorry

end integral_sol_l220_220635


namespace mike_corvette_average_speed_l220_220908

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l220_220908


namespace odd_terms_in_expansion_l220_220836

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l220_220836


namespace digit_count_of_499849_l220_220168

theorem digit_count_of_499849 : ∀ n : ℕ,
  n = 499849 ∧ n < 500000 ∧ (n.digits.sum = 43) → n.digits.length = 6 :=
by
  intros n h,
  cases h with h₁ h₂,
  cases h₂ with h₃ h₄,
  rw h₁,
  simp,
  sorry

end digit_count_of_499849_l220_220168


namespace number_of_ways_l220_220361

def num_ways_distribute_pencils (a b c d : ℕ) : Prop := 
  a + b + c + d = 10 ∧ 
  a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 ∧ 
  a ≤ 2 * b ∧ a ≤ 2 * c ∧ a ≤ 2 * d ∧
  b ≤ 2 * a ∧ b ≤ 2 * c ∧ b ≤ 2 * d ∧
  c ≤ 2 * a ∧ c ≤ 2 * b ∧ c ≤ 2 * d ∧
  d ≤ 2 * a ∧ d ≤ 2 * b ∧ d ≤ 2 * c

theorem number_of_ways : 
∃ (n : ℕ), n = 15 ∧ 
  (set.univ.filter (λ (abcd : ℕ × ℕ × ℕ × ℕ), num_ways_distribute_pencils abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2)).card = n := 
begin
  sorry
end

end number_of_ways_l220_220361


namespace probability_of_three_correct_packages_l220_220295

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220295


namespace mapping_proposition_prism_proposition_right_prism_proposition_existence_of_zero_proposition_solid_geometry_axiom_three_proposition_l220_220535

-- Definitions used directly from the problem conditions
def mapping_definition (A B : Set) (f : A → B) (x : A) (y : B) :=
  ∀ x ∈ A, ∃! y ∈ B, f(x) = y

def prism_definition (faces : Set) (quadrilaterals : Set) :=
  ∃ two_faces ∈ faces, (two_faces_parallel : two_faces ∈ faces) ∧ (common_edge_parallel : ∀ quadrilateral ∈ quadrilaterals, quadrilateral.edge ∈ quadrilaterals.edge ∧ adjacent quadrilateral edge parallel)

def right_prism_definition (prism : Set) (lateral_edges : Set) :=
  ∀ lateral_edge ∈ lateral_edges, (lateral_edge_perpendicular : lateral_edge.perpendicular_to_base ∈ prism.edges.base) ∧ (base_regular : prism.base.polygon = regular_polygon)

def existence_of_zero_theorem (f : ℝ → ℝ) (a b : ℝ) :=
  continuous_on f (a, b) ∧ (f(a) * f(b) < 0) → ∃ x ∈ (a, b), f(x) = 0

def solid_geometry_axiom_three (planes : Set) (common_point : point) :=
  ∃ plane1 plane2 ∈ planes, (plane1 ≠ plane2) ∧ (common_point ∈ plane1 ∧ common_point ∈ plane2) → ∃! common_line, common_line ∈ planes through common_point

-- Lean statements proving them
theorem mapping_proposition (A B : Set) (f : A → B) (x : A) (y : B) :
  mapping_definition A B f x y → (∀ x ∈ A, ∃! y ∈ B, f(x) = y) :=
sorry

theorem prism_proposition (faces quadrilaterals : Set) :
  prism_definition faces quadrilaterals → (∃ two_faces ∈ faces, two_faces_parallel ∧ common_edge_parallel) :=
sorry

theorem right_prism_proposition (prism lateral_edges : Set) :
  right_prism_definition prism lateral_edges → (∀ lateral_edge ∈ lateral_edges, lateral_edge.perpendicular_to_base ∧ prism.base.regular_polygon) :=
sorry

theorem existence_of_zero_proposition (f : ℝ → ℝ) (a b : ℝ) :
  existence_of_zero_theorem f a b → (∃ x ∈ (a, b), f(x) = 0) :=
sorry

theorem solid_geometry_axiom_three_proposition (planes : Set) (common_point : point) :
  solid_geometry_axiom_three planes common_point → (∃! common_line, common_line ∈ planes through common_point) :=
sorry

end mapping_proposition_prism_proposition_right_prism_proposition_existence_of_zero_proposition_solid_geometry_axiom_three_proposition_l220_220535


namespace remainder_prod_mod_7_l220_220716

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220716


namespace sequence_product_modulo_7_l220_220722

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220722


namespace cookie_selling_price_l220_220485

noncomputable def selling_price_per_cookie (dozens : ℕ) (cost_per_cookie : ℝ) (profit_per_charity : ℝ) : ℝ :=
  let total_cookies := dozens * 12
  let total_cost := total_cookies * cost_per_cookie
  let total_profit := profit_per_charity * 2
  let total_money_made := total_profit + total_cost
  total_money_made / total_cookies

theorem cookie_selling_price :
  selling_price_per_cookie 6 0.25 45 = 1.50 :=
begin
  sorry
end

end cookie_selling_price_l220_220485


namespace xy_equals_nine_l220_220427

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l220_220427


namespace find_a_range_l220_220365

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * a * x^2 + 2 * x + 1
def f' (x a : ℝ) : ℝ := x^2 - a * x + 2

theorem find_a_range (a : ℝ) :
  (0 < x1) ∧ (x1 < 1) ∧ (1 < x2) ∧ (x2 < 3) ∧
  (f' 0 a > 0) ∧ (f' 1 a < 0) ∧ (f' 3 a > 0) →
  3 < a ∧ a < 11 / 3 :=
by
  sorry

end find_a_range_l220_220365


namespace mean_of_remaining_set_l220_220052

def arithmetic_mean_of_list (l : List ℝ) : ℝ :=
  (l.sum / l.length : ℝ)

def original_set : List ℝ := List.replicate 57 45 ++ [40, 50, 60]

theorem mean_of_remaining_set :
  arithmetic_mean_of_list (original_set.diff [40, 50, 60]) = 44.74 :=
by
  sorry

end mean_of_remaining_set_l220_220052


namespace value_of_a_l220_220764

theorem value_of_a (x a : ℤ) (h : x = 4) (h_eq : 5 * (x - 1) - 3 * a = -3) : a = 6 :=
by {
  sorry
}

end value_of_a_l220_220764


namespace chord_length_calculation_l220_220573

noncomputable def lengthOfChord : ℝ :=
  let line (t : ℝ) := (2 - (1/2)*t, -1 + (1/2)*t)
  let circle : ℝ × ℝ → Prop := λ ⟨x, y⟩, x^2 + y^2 = 4
  let a := 1
  let b := 1
  let c := -2
  let r := 2
  let d := (| a * 0 + b * 0 + c |) / real.sqrt (a^2 + b^2)
  let chord_length := 2 * real.sqrt (r^2 - d^2)
  chord_length

theorem chord_length_calculation
  (line : ℝ → ℝ × ℝ)
  (circle : ℝ × ℝ → Prop)
  (a b c r : ℝ) 
  (d : ℝ) 
  (chord_length : ℝ) : 
  chord_length_calculation line circle a b c r d chord_length := 
  by 
    let line := λ t : ℝ, (2 - (1/2)*t, -1 + (1/2)*t)
    let circle := λ ⟨x, y⟩, x^2 + y^2 = 4
    let a := 1 
    let b := 1 
    let c := -2 
    let r := 2 
    let d := (| a * 0 + b * 0 + c |) / real.sqrt (a^2 + b^2)
    let chord_length := 2 * real.sqrt (r^2 - d^2)
    sorry

end chord_length_calculation_l220_220573


namespace min_restoration_time_l220_220166

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l220_220166


namespace least_three_digit_with_factors_2_3_5_7_l220_220619

theorem least_three_digit_with_factors_2_3_5_7 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ [2, 3, 5, 7], d ∣ n) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < n → ∀ d ∈ [2, 3, 5, 7], ¬ d ∣ m :=
  ⟨210, by
     apply And.intro _,
     apply And.intro _,
     apply And.intro _,
     iterate 4 { split },
sorry⟩

end least_three_digit_with_factors_2_3_5_7_l220_220619


namespace babysitter_cost_difference_l220_220514

def rate_current : ℝ := 16
def rate_new : ℝ := 12
def extra_new : ℝ := 3
def hours : ℝ := 6
def screams : ℝ := 2

theorem babysitter_cost_difference : 
  let cost_current := rate_current * hours in
  let base_new := rate_new * hours in
  let extra_new_total := extra_new * screams in
  let cost_new := base_new + extra_new_total in
  cost_current - cost_new = 18 :=
by
  let cost_current := rate_current * hours
  let base_new := rate_new * hours
  let extra_new_total := extra_new * screams
  let cost_new := base_new + extra_new_total
  show cost_current - cost_new = 18
  sorry

end babysitter_cost_difference_l220_220514


namespace volume_of_sphere_in_cone_l220_220195

/-- The volume of a sphere inscribed in a right circular cone with
a base diameter of 16 inches and a cross-section with a vertex angle of 45 degrees
is 4096 * sqrt 2 * π / 3 cubic inches. -/
theorem volume_of_sphere_in_cone :
  let d := 16 -- the diameter of the base of the cone in inches
  let angle := 45 -- the vertex angle of the cross-section triangle in degrees
  let r := 8 * Real.sqrt 2 -- the radius of the sphere in inches
  let V := 4 / 3 * Real.pi * r^3 -- the volume of the sphere in cubic inches
  V = 4096 * Real.sqrt 2 * Real.pi / 3 :=
by
  simp only [Real.sqrt]
  sorry -- proof goes here

end volume_of_sphere_in_cone_l220_220195


namespace problem_integer_solution_l220_220252

def satisfies_condition (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem problem_integer_solution :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 20200 ∧ satisfies_condition n :=
sorry

end problem_integer_solution_l220_220252


namespace probability_three_correct_deliveries_l220_220337

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220337


namespace arithmetic_seq_50th_term_l220_220237

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_50th_term : 
  arithmetic_seq_nth_term 3 7 50 = 346 :=
by
  -- Intentionally left as sorry
  sorry

end arithmetic_seq_50th_term_l220_220237


namespace solve_trig_system_l220_220047

open Real

theorem solve_trig_system (x y : ℝ) (k : ℤ) (hx : tan x * tan y = 1/6) (hy : sin x * sin y = 1/(5 * sqrt 2)) :
  (x + y = ±(π / 4) + 2 * π * k) ∧ (x - y = ±(arccos ((7 / (5 * sqrt 2))) + 2 * π * k)) :=
sorry

end solve_trig_system_l220_220047


namespace charlie_scored_25_points_l220_220857

theorem charlie_scored_25_points 
    (team_total_points : ℕ) 
    (num_other_players : ℕ) 
    (avg_points_other_players : ℕ) 
    (team_total_points_eq : team_total_points = 60)
    (num_other_players_eq : num_other_players = 7) 
    (avg_points_other_players_eq : avg_points_other_players = 5) :
    ∃ (charlie_points : ℕ), charlie_points = 25 :=
by 
    have total_points_other_players := num_other_players * avg_points_other_players,
    rw [num_other_players_eq, avg_points_other_players_eq] at total_points_other_players,
    have total_points_other_players_eq : total_points_other_players = 35,
    from total_points_other_players,
    have charlie_points := team_total_points - total_points_other_players,
    rw [team_total_points_eq, total_points_other_players_eq] at charlie_points,
    use charlie_points,
    linarith,
    sorry

end charlie_scored_25_points_l220_220857


namespace mari_made_64_buttons_l220_220023

def mari_buttons (kendra : ℕ) : ℕ := 5 * kendra + 4

theorem mari_made_64_buttons (sue kendra mari : ℕ) (h_sue : sue = 6) (h_kendra : kendra = 2 * sue) (h_mari : mari = mari_buttons kendra) : mari = 64 :=
by 
  rw [h_sue, h_kendra, h_mari]
  simp [mari_buttons]
  sorry

end mari_made_64_buttons_l220_220023


namespace three_correct_deliveries_probability_l220_220272

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220272


namespace min_possible_value_max_M_max_possible_value_min_m_l220_220130

theorem min_possible_value_max_M :
  ∀ (a b : fin 10 → ℕ), 
    (∀ i, a i ∈ finset.range 1 11) →
    (∀ i, b i ∈ finset.range 1 11) →
    (∀ i j, a i = a j → i = j) →
    (∀ i j, b i = b j → i = j) →
    ∃ M, 
        M = max (finset.image (λ i, (a i)^2 + (b i)^2) finset.univ) 
    ∧ M = 101 :=
begin
  sorry
end

theorem max_possible_value_min_m :
  ∀ (a b : fin 10 → ℕ), 
    (∀ i, a i ∈ finset.range 1 11) →
    (∀ i, b i ∈ finset.range 1 11) →
    (∀ i j, a i = a j → i = j) →
    (∀ i j, b i = b j → i = j) →
    ∃ m, 
        m = min (finset.image (λ i, (a i)^2 + (b i)^2) finset.univ) 
    ∧ m = 61 :=
begin
  sorry
end

end min_possible_value_max_M_max_possible_value_min_m_l220_220130


namespace probability_three_correct_packages_l220_220310

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220310


namespace problem_statement_l220_220594

theorem problem_statement (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 4 / 5) : y - x = 500 / 9 := 
by
  sorry

end problem_statement_l220_220594


namespace prime_gt_7_divisibility_l220_220236

theorem prime_gt_7_divisibility (p : ℕ) (h_prime : p.prime) (h_geq_7 : p ≥ 7) :
  ¬ (∀ p, p.prime ∧ p ≥ 7 → 30 ∣ (p^2 - 1)) :=
by
  sorry

end prime_gt_7_divisibility_l220_220236


namespace probability_three_correct_out_of_five_l220_220277

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220277


namespace min_shots_to_hit_terrorist_l220_220645

theorem min_shots_to_hit_terrorist : ∀ terrorist_position : ℕ, (1 ≤ terrorist_position ∧ terrorist_position ≤ 10) →
  ∃ shots : ℕ, shots ≥ 6 ∧ (∀ move : ℕ, (shots - move) ≥ 1 → (terrorist_position + move ≤ 10 → terrorist_position % 2 = move % 2)) :=
by
  sorry

end min_shots_to_hit_terrorist_l220_220645


namespace quadratic_inequality_solution_l220_220845

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ ax^2 + bx - 1 > 0) : 
  (∀ x : ℝ, x ∈ (set.Ioo (2 / 3) 2) ↔ (ax + 1) / (bx - 1) > 0) :=
by
  sorry

end quadratic_inequality_solution_l220_220845


namespace find_log_pqr_l220_220131

-- Conditions provided in the problem.
variables {p q r x : ℝ}
variables (d : ℝ)
hypothesis (h1 : log x / log p = 2)
hypothesis (h2 : log x / log q = 3)
hypothesis (h3 : log x / log r = 6)
hypothesis (h4 : log x / log (p * q * r) = d)

-- The goal: d = 1.
theorem find_log_pqr : d = 1 :=
by {
  sorry
}

end find_log_pqr_l220_220131


namespace neg_abs_value_eq_neg_three_l220_220982

theorem neg_abs_value_eq_neg_three : -|-3| = -3 := 
by sorry

end neg_abs_value_eq_neg_three_l220_220982


namespace horizontal_asymptote_cross_l220_220746

def f (x : ℝ) : ℝ := (2 * x^2 - 5 * x - 7) / (x^2 - 4 * x + 1)

theorem horizontal_asymptote_cross : f 3 = 2 :=
by
  sorry

end horizontal_asymptote_cross_l220_220746


namespace total_distance_this_week_l220_220610

def daily_distance : ℕ := 6 + 7
def num_days : ℕ := 5

theorem total_distance_this_week : daily_distance * num_days = 65 := by
  rw [daily_distance, num_days]
  norm_num
  sorry

end total_distance_this_week_l220_220610


namespace probability_three_correct_packages_l220_220308

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220308


namespace length_of_XY_l220_220866

-- Definitions based on conditions
def AOB_angle : ℝ := 45
def OY_length : ℝ := 10
def XY_length := 10 - 5 * Real.sqrt 2

-- Theorem statement
theorem length_of_XY 
  (AOB : ℝ)
  (OY : ℝ)
  (ha : AOB = 45)
  (ho : OY = 10) :
  ∃ XY : ℝ, XY = 10 - 5 * Real.sqrt 2 := 
begin
  use 10 - 5 * Real.sqrt 2,
  tauto,
sorry -- Proof is omitted
end

end length_of_XY_l220_220866


namespace norris_money_left_l220_220922

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l220_220922


namespace region_area_l220_220521

-- Variables and assumptions
variables {z : ℂ}

-- Define the region
def region :=
  {z : ℂ | 0 ≤ complex.arg (z - 1) ∧ complex.arg (z - 1) ≤ π / 4 ∧ z.re ≤ 2}

-- Statement of the theorem
theorem region_area : 
  ∃ (S : ℝ), 
  (S = 1) ∧ 
  (area of the region defined by {z | 0 ≤ complex.arg (z - 1) ∧ complex.arg (z - 1) ≤ π / 4 ∧ z.re ≤ 2} = S) :=
sorry

end region_area_l220_220521


namespace product_ab_l220_220568

noncomputable def a : ℂ := -7 - 2i
noncomputable def b : ℂ := 7 + 2i
noncomputable def line : ℂ × ℂ → ℂ := λ z, a * z.1 + b * conj z.1

theorem product_ab :
  (let u := (3 : ℂ) - 4i
       v := (-4 : ℂ) + 2i
       line_eqn := line (u, v)
   in line_eqn = 20 → a * b = 53) := sorry

end product_ab_l220_220568


namespace grid_has_11_distinct_integers_l220_220527

theorem grid_has_11_distinct_integers
  (grid : Fin 101 → Fin 101 → Fin 102)
  (H1 : ∀ k : Fin 102, 
          (∃ r : Fin 101, ∃ c : Fin 101, grid r c = k) ∧
          (∑ r : Fin 101, ∑ c : Fin 101, (if grid r c = k then 1 else 0)) = 101) :
  ∃ r : Fin 101, ∃ S : Finset (Fin 102), S.card ≥ 11 ∧ ∀ c : Fin 101, grid r c ∈ S ∨
  ∃ c : Fin 101, ∃ S : Finset (Fin 102), S.card ≥ 11 ∧ ∀ r : Fin 101, grid r c ∈ S := 
sorry

end grid_has_11_distinct_integers_l220_220527


namespace find_f_2008_l220_220506

-- Define the set B
def B : Set ℚ := { x : ℚ | x ≠ -1 ∧ x ≠ 2 }

-- Define the function h
def h (x : ℚ) : ℚ := 2 - (1 / (x + 1))

-- Define the function f with the given property
def f (x : ℚ) : ℝ :=
  if x ∈ B then (1 / 3) * (Real.log (|x + 1|) - Real.log (|h x + 1|) + Real.log (|h (h x) + 1|))
  else 0  -- f is undefined outside B, but we provide 0 to have a total function

-- State the problem
theorem find_f_2008 : f 2008 = Real.log ((2009 * 2002) / 2005) := sorry

end find_f_2008_l220_220506


namespace number_of_toys_sold_l220_220179

theorem number_of_toys_sold (n : ℕ) 
  (selling_price : ℕ = 23100)
  (cost_price_single_toy : ℕ = 1100)
  (gain : ℕ = 3 * cost_price_single_toy)
  (total_cost_price : ℕ = n * cost_price_single_toy)
  (sale_eq_cost_price_plus_gain : selling_price = total_cost_price + gain) :
  n = 18 :=
by 
  sorry

end number_of_toys_sold_l220_220179


namespace rectangular_sheet_integer_side_l220_220660

theorem rectangular_sheet_integer_side
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_cut_a : ∀ x, x ≤ a → ∃ n : ℕ, x = n ∨ x = n + 1)
  (h_cut_b : ∀ y, y ≤ b → ∃ n : ℕ, y = n ∨ y = n + 1) :
  ∃ n m : ℕ, a = n ∨ b = m := 
sorry

end rectangular_sheet_integer_side_l220_220660


namespace trisectors_middle_segment_not_longest_l220_220667

theorem trisectors_middle_segment_not_longest
  (A B C H1 H2 : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq H1] [DecidableEq H2]
  (triangle_ABC : Triangle A B C)
  (angle_ACB : Angle A C B)
  (trisectors_H1 H2 : Trisector C (segment A B) H1 H2)
  (angles_pos : 0 < angle_ACB ∧ angle_ACB < 180) :
  ¬ longest_segment (segment H1 H2) (segments_divided A H1 H2 B) :=
sorry

end trisectors_middle_segment_not_longest_l220_220667


namespace x_pow_y_equals_nine_l220_220425

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l220_220425


namespace sequence_product_modulo_7_l220_220719

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220719


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220808

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220808


namespace probability_exactly_three_correct_l220_220348

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220348


namespace det_A_minus_dA_star_l220_220490

-- Define the matrix type for 2x2 real matrices
def Matrix2Real := Matrix (Fin 2) (Fin 2) ℝ

-- Define the problem conditions in Lean
variables (A : Matrix2Real) (d : ℝ)

-- Hypotheses based on problem statement
hypothesis h1 : det A = d
hypothesis h2 : d ≠ 0
hypothesis h3 : det (A + d • Aᵀ) = 0

-- Statement of the theorem to be proved
theorem det_A_minus_dA_star : det (A - d • Aᵀ) = 4 := 
by
  sorry

end det_A_minus_dA_star_l220_220490


namespace solve_for_x_l220_220545

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l220_220545


namespace part1_part2_l220_220356

-- Definitions following the conditions in the problem
def f (x : ℕ) : ℕ :=
  if x = 1 then 1
  else ∑ p in (factors x).toFinset.filter (λ p, nat.Prime p ∧ (multiplicity p x % 2 = multiplicity p x % 2)), p ^ (multiplicity p x).get

noncomputable def a_seq (m : ℕ) (a : ℕ → ℕ) : ℕ → ℕ
| n := if n ≤ m then a n else max {f (a n) + k | k : ℕ, k ≤ m}

-- First part: Existence of constants A and B
theorem part1 (A B : ℝ) (hA : 0 < A ∧ A < 1) :
  (∀ (x : ℕ), x > 1 ∧ x prime_factors.size > 1 → f x < A * x + B) :=
sorry

-- Second part: Existence of an integer Q
theorem part2 (m : ℕ) (a : ℕ → ℕ) :
  ∃ (Q : ℕ), ∀ (n : ℕ), a_seq m a n < Q :=
sorry

end part1_part2_l220_220356


namespace lines_are_concurrent_l220_220124

noncomputable def concurrent_lines (A B C D E F A1 D1 B1 E1 C1 F1 : Type*)
    [convex_hexagon ABCDEF inscribed_circle circumcircle] 
    (lFA : external_tangentωFωA) (lAB : external_tangentωAωB) 
    (lBC : external_tangentωBωC) (lCD : external_tangentωCωD) 
    (lDE : external_tangentωDωE) (lEF : external_tangentωEωF)
    (A1 := intersection lFA lAB) (D1 := intersection lDE lEF)
    (B1 := intersection lAB lBC) (E1 := intersection lBC lCD)
    (C1 := intersection lCD lDE) (F1 := intersection lEF lFA) :
    Prop :=
  concurrent A1 D1 B1 E1 C1 F1

theorem lines_are_concurrent
  (A B C D E F : Type*)
  [convex_hexagon ABCDEF inscribed_circle circumcircle] :
  concurrent_lines A B C D E F sorry sorry sorry sorry sorry sorry := sorry

end lines_are_concurrent_l220_220124


namespace tangent_line_value_l220_220961

theorem tangent_line_value 
    (k a b : ℝ)
    (h1 : ∀ x y, y = k * x + 1 → (x, y) = (1, 3))
    (h2 : ∀ x y, y = x^3 + a * x + b → (x, y) = (1, 3))
    (h3 : ∀ x, deriv (λ x, x^3 + a * x + b) x = 3 * x^2 + a) :
    2 * a + b = 1 :=
sorry

end tangent_line_value_l220_220961


namespace find_MD_l220_220960

-- Definitions based on the problem's conditions
variables (A B C D K M : Point)
variables (BK AD AC : Line)
variable (h_rhombus : is_rhombus A B C D)
variable (h_BK_perp_AD : is_perpendicular BK AD)
variable (h_BK_length : length BK = 4)
variable (h_M_on_BK : M ∈ BK)
variable (h_M_on_AC : M ∈ AC)
variable (h_AK_KD_ratio : ratio AK KD = 1 / 2)

-- The main statement to be proved
theorem find_MD :
  length MD = 2 :=
sorry

end find_MD_l220_220960


namespace probability_three_correct_deliveries_l220_220336

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220336


namespace inequality_iff_l220_220547

def f (x : ℝ) : ℝ :=
  if x >= 0 then x else 0

def g (x : ℝ) : ℝ :=
  if x < 2 then 2 - x else 0

theorem inequality_iff (x : ℝ) : f (g x) > g (f x) ↔ x < 0 := by
  sorry

end inequality_iff_l220_220547


namespace problem_l220_220774

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (2 * m + 1) * x - 1
noncomputable def h (m : ℝ) (x : ℝ) := f m x + g m x

noncomputable def h_deriv (m : ℝ) (x : ℝ) : ℝ := m * x - (2 * m + 1) + (2 / x)

theorem problem (m : ℝ) : h_deriv m 1 = h_deriv m 3 → m = 2 / 3 :=
by
  sorry

end problem_l220_220774


namespace internet_service_ratio_l220_220026

theorem internet_service_ratio :
  ∀ (current_bill P20 P30 : ℝ),
    current_bill = 20 →
    P20 = 30 →
    (P30 - P20) * 12 = 120 →
    P30 / current_bill = 2 := 
by {
  intros current_bill P20 P30 h_current_bill h_P20 h_savings,
  rw h_current_bill at *,
  rw h_P20 at *,
  simp at h_savings,
  linarith
 }

end internet_service_ratio_l220_220026


namespace probability_three_correct_deliveries_l220_220338

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220338


namespace sufficient_condition1_sufficient_condition2_sufficient_condition3_l220_220762

variables {X : Type} [Preorder X] [TopologicalSpace X] [OrderTopology X] [ConditionallyCompleteLinearOrder X]
variables {f : X → ℝ}

-- Condition definitions:
def odd_function (f : X → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : X → ℝ) : Prop := ∀ x, g (-x) = g x
def periodic (f : X → ℝ) (T : X) : Prop := ∀ x, f (x + T) = f x
def extremum_at (f : X → ℝ) (x : X) : Prop := ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → f y ≤ f x
def derivative_zero_at (f' : X → ℝ) (x : X) : Prop := f' x = 0

-- The sufficient condition proofs:
theorem sufficient_condition1 (h : odd_function f) : even_function (derivative f) := sorry
theorem sufficient_condition2 {T : X} (h : periodic f T) : periodic (derivative f) T := sorry
theorem sufficient_condition3 {x : X} (h : extremum_at f x) : derivative_zero_at (derivative f) x := sorry

end sufficient_condition1_sufficient_condition2_sufficient_condition3_l220_220762


namespace _l220_220661

noncomputable def Pythagorean_theorem (a b: ℕ) : ℝ := real.sqrt (a * a + b * b)

lemma similar_triangles_corresponding_leg_length : 
  ∀ (a b c d: ℕ), c * a = d * 15 → c = 3 → (a = 15 → d = 45) := 
by {
  intros a b c d h1 h2 h3,
  rw h3 at h1,
  linarith [h2, h3],
  sorry,
}

end _l220_220661


namespace trapezoid_slopes_sum_abs_l220_220466

theorem trapezoid_slopes_sum_abs 
  (E H : ℝ × ℝ)
  (e_coord : E = (11, 31))
  (h_coord : H = (12, 38))
  (no_horiz_or_vert_sides : ¬ ∃ F G : ℝ × ℝ, (F.1 = E.1 ∨ F.2 = E.2) ∧ (G.1 = H.1 ∨ G.2 = H.2))
  (only_parallel_sides : ∃ F G : ℝ × ℝ, (slope F E = slope G H ∧ slope F E ≠ slope F H)) :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ (∑ k in (abs (λ m, some (slope F E) k)).to_finset, k = 26) ∧ (p + q = 27) := sorry

end trapezoid_slopes_sum_abs_l220_220466


namespace problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l220_220861

theorem problem1421_part1 (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_yellow : yellow_balls = 15) :
  (red_balls < yellow_balls) := by 
  sorry  -- Solution Proof for Part 1

theorem problem1421_part2 (total_balls : ℕ) (red_balls : ℕ) (h_total : total_balls = 20) 
  (h_red : red_balls = 5) :
  (red_balls / total_balls = 1 / 4) := by 
  sorry  -- Solution Proof for Part 2

theorem problem1421_part3 (red_balls total_balls m : ℕ) (h_red : red_balls = 5) 
  (h_total : total_balls = 20) :
  ((red_balls + m) / (total_balls + m) = 3 / 4) → (m = 40) := by 
  sorry  -- Solution Proof for Part 3

theorem problem1421_part4 (total_balls red_balls additional_balls x : ℕ) 
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_additional : additional_balls = 18):
  (total_balls + additional_balls = 38) → ((red_balls + x) / 38 = 1 / 2) → 
  (x = 14) ∧ ((additional_balls - x) = 4) := by 
  sorry  -- Solution Proof for Part 4

end problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l220_220861


namespace iterative_average_difference_l220_220214

-- Definition of iterative average
def iterative_average (seq : List ℚ) : ℚ :=
  (seq.tail.foldl (λ acc x, (acc + x) / 2) seq.head)

-- List of all permutations of the set {1, 2, 3, 4, 5}
def all_permutations : List (List ℚ) := [1, 2, 3, 4, 5].permutations

-- Maximum and Minimum iterative average from all permutations
def max_iterative_average : ℚ := all_permutations.map iterative_average |> List.maximum' (by decide)
def min_iterative_average : ℚ := all_permutations.map iterative_average |> List.minimum' (by decide)

-- Theorem stating the difference
theorem iterative_average_difference : max_iterative_average - min_iterative_average = 17/8 := 
by sorry

end iterative_average_difference_l220_220214


namespace find_a_such_that_l220_220011

def f (x : ℝ) : ℝ := if x ≤ 0 then 2 * x else 3 * x - 52

theorem find_a_such_that (a : ℝ) (h : a < 0) : f (f (f 11)) = f (f (f a)) ↔ a = -19 :=
by
  have := f
  sorry

end find_a_such_that_l220_220011


namespace find_length_QR_l220_220475

variable {P Q R : Type} [Point P] [Point Q] [Point R]

def angle_Q : Angle := 30
def angle_R : Angle := 105
def PR : ℝ := 4 * Real.sqrt 2

theorem find_length_QR (QR : ℝ) :
  let angle_P : Angle := 180 - angle_Q - angle_R,
      sin_P := Real.sin (angle_P * Real.pi / 180),
      sin_Q := Real.sin (angle_Q * Real.pi / 180)
  in QR = PR * (sin_P / sin_Q) → QR = 8 :=
by
  sorry

end find_length_QR_l220_220475


namespace complex_purely_imaginary_condition_l220_220423

theorem complex_purely_imaginary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) ∧
  ¬(a = 1 ∧ ¬a = -2 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) :=
  sorry

end complex_purely_imaginary_condition_l220_220423


namespace sequence_is_natural_l220_220886

noncomputable def sequence (n : ℕ) : ℕ := sorry

variables (c : ℝ) (h_c : c > 1) (a : ℕ → ℕ)
          (h_a1 : a 1 = 1)
          (h_a2 : a 2 = 2)
          (h_mult : ∀ m n, a (m * n) = a m * a n)
          (h_add : ∀ m n, a (m + n) ≤ c * (a m + a n))

theorem sequence_is_natural (n : ℕ) : a n = n := sorry

end sequence_is_natural_l220_220886


namespace price_after_9_years_l220_220169

-- Assume the initial conditions
def initial_price : ℝ := 640
def decrease_factor : ℝ := 0.75
def years : ℕ := 9
def period : ℕ := 3

-- Define the function to calculate the price after a certain number of years, given the period and decrease factor
def price_after_years (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_price * (decrease_factor ^ (years / period))

-- State the theorem that we intend to prove
theorem price_after_9_years : price_after_years initial_price decrease_factor 9 period = 270 := by
  sorry

end price_after_9_years_l220_220169


namespace sandwiches_per_person_l220_220151

open Nat

theorem sandwiches_per_person (total_sandwiches : ℕ) (total_people : ℕ) (h1 : total_sandwiches = 657) (h2 : total_people = 219) : 
(total_sandwiches / total_people) = 3 :=
by
  -- a proof would go here
  sorry

end sandwiches_per_person_l220_220151


namespace probability_three_correct_deliveries_is_one_sixth_l220_220317

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220317


namespace concentric_tangent_product_l220_220584

variable {R r : ℝ} -- radii of the circles
variable (p : ℝ) (h_p : p = R / r)
variable {A D B C P : Point} -- points as described in the problem

-- Conditions: geometric relationships
variable (large_circle : Circle) (small_circle : Circle)
variable (concentric : large_circle.center = small_circle.center)
variable (diameter_AD : is_diameter A D large_circle)
variable (intersect_BC : is_intersection B C diameter_AD small_circle)
variable (on_smaller_circle : on_circle P small_circle)

-- Prove the given statement
theorem concentric_tangent_product (p : ℝ) (h_p : p = R / r) :
  ∃ A D B C P : Point,
  is_diameter A D large_circle ∧ 
  is_intersection B C diameter_AD small_circle ∧ 
  on_circle P small_circle →
  tan (angle A P B) * tan (angle C P D) = ( (p - 1) / (p + 1))^2 := 
by
  sorry

end concentric_tangent_product_l220_220584


namespace calculate_expression_l220_220224

theorem calculate_expression : (Real.pi - 2023)^0 - |1 - Real.sqrt 2| + 2 * Real.cos (Real.pi / 4) - (1 / 2)⁻¹ = 0 :=
by
  sorry

end calculate_expression_l220_220224


namespace equilateral_triangle_area_l220_220071

theorem equilateral_triangle_area (p : ℝ) : 
  let s := p in 
  (3 * s = 3 * p) → 
  (∃ A, A = (sqrt 3 / 4) * p^2) :=
by 
  intro h
  use (sqrt 3 / 4) * p^2
  have h1 : s = p := by linarith
  assumption

end equilateral_triangle_area_l220_220071


namespace triangle_area_l220_220208

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) :
  (∃ (a b c : ℕ), (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 4 ∧ b = 3 ∧ c = 5) ∨ (a = 5 ∧ b = 4 ∧ c = 3)) →
  (1/2 * a * b : ℝ = 6) :=
by
  sorry

end triangle_area_l220_220208


namespace smallest_m_plus_n_l220_220504

theorem smallest_m_plus_n : 
  ∃ (m n : ℕ) (hm : Nat.gcd m n = 1),
    (∑ k in Finset.range 1004, 1 / ((2 * k + 10) * (2 * k + 12))) = (m / n) ∧
    m + n = 10571 :=
by
  sorry

end smallest_m_plus_n_l220_220504


namespace odd_number_diff_squares_representation_l220_220528

noncomputable def number_of_ways {n : ℕ} (p : Fin n → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_odd : ∀ i, p i % 2 = 1) : ℕ :=
  let m := ∏ i, p i
  2 ^ (n - 1)

theorem odd_number_diff_squares_representation {n : ℕ} (p : Fin n → ℕ)
  (h_prime : ∀ i, Nat.Prime (p i)) (h_odd : ∀ i, p i % 2 = 1) :
  let m := ∏ i, p i
  (m % 2 = 1) → -- m is odd
  (∃ N, (N = number_of_ways p h_prime h_odd) ∧ (N = 2^(n-1))) :=
by
  intros
  let m := ∏ i, p i
  have : m % 2 = 1 := by
    sorry
  have : ∃ N, N = 2^(n-1) := by
    sorry
  use 2^(n-1)
  split
  · rfl
  · rfl

end odd_number_diff_squares_representation_l220_220528


namespace rectangle_perimeter_l220_220201

theorem rectangle_perimeter (P : ℕ) (s : ℕ): (P = 4 * s) → (P = 120) → 
  (4 * s = 120) → 
  let half_side := s / 2 in 
  let rect_perimeter := 2 * (s + half_side) in
  rect_perimeter = 90 :=
by
  intros P s hP hP120 h4s120 half_side rect_perimeter
  rw [hP120, h4s120] at hP
  have s_value : s = 30 := by 
    linarith
  rw [s_value] at *
  let calculated_perimeter : ℕ := 2 * (s + s / 2)
  have perimeter_def : rect_perimeter = calculated_perimeter := rfl
  rw [s_value, ← perimeter_def] at *
  simp
  exact rfl

end rectangle_perimeter_l220_220201


namespace Vins_bike_trip_distance_l220_220608

theorem Vins_bike_trip_distance (d1 d2 d3 : ℕ) (m1 m2 trips : ℕ) : 
  d1 = 6 → d2 = 7 → trips = 5 → d3 = d1 + d2 → m1 = d3 * trips → m2 = 65 → 
  m1 = m2 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h4, h1, h2, h5] at h6
  exact h6
  sorry

end Vins_bike_trip_distance_l220_220608


namespace sum_of_cn_l220_220394

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3

def sequence_b : ℕ → ℤ 
| 1 := -1
| (n + 1) := sequence_b n + (2 * n - 1)

def sequence_c (n : ℕ) : ℚ :=
  (sequence_a n * sequence_b n : ℚ) / n

def T (n : ℕ) : ℚ :=
  (3 / 2) * n^2 - (9 / 2) * n + 1

theorem sum_of_cn {n : ℕ} : 
  (∑ i in Finset.range n, sequence_c (i + 1)) = T n := sorry

end sum_of_cn_l220_220394


namespace roots_of_polynomial_in_range_l220_220847

theorem roots_of_polynomial_in_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 * x2 = m^2 - 2 ∧ (x1 + x2) = -(m - 1)) 
  -> 0 < m ∧ m < 1 :=
by
  sorry

end roots_of_polynomial_in_range_l220_220847


namespace min_restoration_time_l220_220165

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l220_220165


namespace find_DF_l220_220862

variables (D E F : Type) [IsRightTriangle D E F] (cosE : Real) (EF : Real) (sqrt221 : Real)

/-- condition: In a right triangle DEF, where E is the right angle,
Cos(E) = (5 * sqrt(221)) / 221. EF is the hypotenuse with EF = sqrt(221).
Find DF -/
theorem find_DF
  (cos_eq : cosE = (5 * sqrt221) / 221)
  (EF_eq : EF = sqrt221)
  (cos_definition : cosE = DF / EF) :
  DF = 5 :=
begin
  sorry
end

end find_DF_l220_220862


namespace fractionProcessedByTeamB_l220_220647

variable (C : ℝ) -- Number of calls processed by each member of team B
variable (N : ℝ) -- Number of agents in team B

-- Condition: Each member of team A processes 7/5 times the number of calls as each member of team B
def callsProcessedByMemberA := (7 / 5 : ℝ) * C

-- Condition: Team A has 5/8 as many agents as team B
def agentsInTeamA := (5 / 8 : ℝ) * N

-- Total calls processed by team A
def totalCallsByTeamA := callsProcessedByMemberA * agentsInTeamA

-- Total calls processed by team B
def totalCallsByTeamB := C * N

-- Total calls processed by both teams
def totalCalls := totalCallsByTeamA + totalCallsByTeamB

-- The fraction of total calls processed by team B
def fractionOfCallsByTeamB := totalCallsByTeamB / totalCalls

-- The theorem to prove
theorem fractionProcessedByTeamB : fractionOfCallsByTeamB = (8 / 15 : ℝ) :=
by
  sorry

end fractionProcessedByTeamB_l220_220647


namespace inequality_geq_8_l220_220581

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l220_220581


namespace find_dividend_l220_220844

def quotient : ℝ := 9.3
def divisor : ℝ := 4 / 3
def remainder : ℝ := 5
def dividend : ℝ := 17.4

theorem find_dividend :
  divisor * quotient + remainder = dividend :=
by
  sorry

end find_dividend_l220_220844


namespace probability_three_correct_deliveries_l220_220341

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220341


namespace fencing_cost_approx_l220_220125

noncomputable def π : Real := Math.pi

-- Define the problem conditions
def diameter : Real := 36
def rate_per_meter : Real := 3.50

-- Calculation of circumference
def circumference : Real := π * diameter

-- Calculation of fencing cost
def cost_of_fencing : Real := circumference * rate_per_meter

-- The proof problem
theorem fencing_cost_approx : Real.round cost_of_fencing = 396 := by
  sorry

end fencing_cost_approx_l220_220125


namespace solve_for_x_l220_220357

-- Definition of the operation
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- The mathematical statement to be proved
theorem solve_for_x (x : ℝ) (h : otimes x (x - 1) = 3) : x = 2 ∨ x = -1 := 
by 
  sorry

end solve_for_x_l220_220357


namespace range_of_data_set_l220_220971

theorem range_of_data_set : 
  let data := [3, 4, 4, 6] in
  data.maximum! - data.minimum! = 3 :=
by
  sorry

end range_of_data_set_l220_220971


namespace system_of_equations_has_integer_solutions_l220_220529

theorem system_of_equations_has_integer_solutions (a b : ℤ) :
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_of_equations_has_integer_solutions_l220_220529


namespace three_correct_deliveries_probability_l220_220271

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220271


namespace w_pow_six_l220_220489

-- Define the complex number w
def w : ℂ := (-1 + real.sqrt 3 * complex.I) / 2

-- State the theorem
theorem w_pow_six : w^6 = 1 :=
by sorry

end w_pow_six_l220_220489


namespace probability_exactly_three_correct_l220_220352

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220352


namespace cyclic_quadrilateral_l220_220040

-- Define the points A, B, C, D, M, E, K on the circle
variables {P : Type*} [MetricSpace P] [InnerProductSpace ℝ P] [normed_space.create_finite_dimensional ℝ P]
variables (A B C D M E K : P)

-- Define the circle that contains points A, B, C, D with M being the midpoint of arc AB
axiom on_circle : ∀ (P : P), P ∈ circle ↔ dist O P = radius
axiom midpoint_arc : is_midpoint_of_arc A B M

-- Define the intersection conditions
axiom intersection_E : E ∈ line_segment M C ∧ E ∈ line_segment A B
axiom intersection_K : K ∈ line_segment M D ∧ K ∈ line_segment A B

-- Prove that KECD is a cyclic quadrilateral
theorem cyclic_quadrilateral :
  ∃ (O : P) (circle : Circle P), quadrilateral_cyclic (E, K, C, D) :=
sorry

end cyclic_quadrilateral_l220_220040


namespace five_step_staircase_toothpicks_l220_220219

theorem five_step_staircase_toothpicks (toothpicks_three_steps: ℕ) (toothpicks_five_steps: ℕ) :
  toothpicks_three_steps = 18 → toothpicks_five_steps - toothpicks_three_steps = 22 :=
  begin
    sorry  -- Proof omitted
  end

end five_step_staircase_toothpicks_l220_220219


namespace ln_abs_x_minus_1_increasing_l220_220703

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 1|)

theorem ln_abs_x_minus_1_increasing :
  ∀ x : ℝ, 1 < x → ∀ y : ℝ, 1 < y → x < y → f(x) < f(y) := 
by
  sorry

end ln_abs_x_minus_1_increasing_l220_220703


namespace gray_region_area_l220_220454

theorem gray_region_area (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 3) : 
  (π * (3 * r) * (3 * r) - π * r * r) = 18 * π := by
  sorry

end gray_region_area_l220_220454


namespace wally_not_all_numbers_l220_220934

def next_wally_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2
  else
    (n + 1001) / 2

def eventually_print(n: ℕ) : Prop :=
  ∃ k: ℕ, (next_wally_number^[k]) 1 = n

theorem wally_not_all_numbers :
  ¬ ∀ n, n ≤ 100 → eventually_print n :=
by
  sorry

end wally_not_all_numbers_l220_220934


namespace find_tunnel_length_l220_220596

-- Define the conditions
variables (speed_mph : ℕ) (speed_mpm : ℝ) (train_length : ℝ) (time_minutes : ℕ) (distance_travelled : ℝ)

-- Assign the given values from the conditions
def train_speed : Prop := speed_mph = 90
def train_speed_mpm : Prop := speed_mpm = 1.5
def train_length_def : Prop := train_length = 2
def time_def : Prop := time_minutes = 4
def distance_travelled_def : Prop := distance_travelled = 6

-- Theorem stating the length of the tunnel
theorem find_tunnel_length (h1 : train_speed) (h2 : train_speed_mpm) (h3 : train_length_def) 
  (h4 : time_def) (h5 : distance_travelled_def) : 
  distance_travelled - train_length = 4 :=
by 
  sorry

end find_tunnel_length_l220_220596


namespace amanda_round_trip_time_l220_220211

variable (time_bus_to_beach : ℝ)
variable (distance_to_beach : ℝ)
variable (time_differential : ℝ)
variable (detour_distance : ℝ)
variable (traffic_factor : ℝ)

-- Given conditions
axiom h1 : time_bus_to_beach = 40
axiom h2 : distance_to_beach = 120
axiom h3 : time_differential = 5
axiom h4 : detour_distance = 15
axiom h5 : traffic_factor = 1.3

-- Proof statement
theorem amanda_round_trip_time :
  let car_time_to_beach := time_bus_to_beach - time_differential in
  let car_speed := distance_to_beach / car_time_to_beach in
  let normal_detour_time := detour_distance / car_speed in
  let adjusted_detour_time := normal_detour_time * traffic_factor in
  let total_detour_time := 2 * adjusted_detour_time in
  let normal_round_trip_time := 2 * car_time_to_beach in
  let total_round_trip_time := normal_round_trip_time + total_detour_time in
  abs(total_round_trip_time - 81.375) < 0.001 :=
by
  sorry

end amanda_round_trip_time_l220_220211


namespace probability_three_correct_deliveries_l220_220326

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220326


namespace even_product_probability_l220_220607

def number_on_first_spinner := [3, 6, 5, 10, 15]
def number_on_second_spinner := [7, 6, 11, 12, 13, 14]

noncomputable def probability_even_product : ℚ :=
  1 - (3 / 5) * (3 / 6)

theorem even_product_probability :
  probability_even_product = 7 / 10 :=
by
  sorry

end even_product_probability_l220_220607


namespace neg_neg_eq_pos_l220_220138

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l220_220138


namespace probability_of_three_primes_l220_220093

-- Defining the range of integers
def range := {n : ℕ | 1 ≤ n ∧ n ≤ 30}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Defining the set of prime numbers from 1 to 30
def primes : set ℕ := {n | n ∈ range ∧ is_prime n}

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := n.choose k

-- Define probability function
def probability_three_primes : ℚ :=
  binomial 10 3 / binomial 30 3

theorem probability_of_three_primes : 
  probability_three_primes = 6 / 203 :=
by sorry

end probability_of_three_primes_l220_220093


namespace count_three_digit_integers_end7_divby21_l220_220786

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220786


namespace probability_three_correct_out_of_five_l220_220280

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220280


namespace greatest_divisor_of_arithmetic_sum_l220_220109

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end greatest_divisor_of_arithmetic_sum_l220_220109


namespace ellipse_equation_and_lambda_range_l220_220374

theorem ellipse_equation_and_lambda_range :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b = a / 2 ∧ a * b = 1 / 2 ∧
    (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 + 4 * y^2 = 1))) ∧
  (∀ (λ : ℝ), (∃ m : ℝ, m > 0 ∧ x = m * y + λ ∧ 
    ∀ y1 y2 : ℝ, (2 * (m^2 + 4) * y2 = -2 * λ * m ∧ λ ∈ ℝ) ∧ 
    y1 + y2 = -λ * m / (m^2 + 4) ∧ y1 * y2 = -(λ^2 - 1) / (m^2 + 4)) ↔ 
    λ ∈ set.Icc (-1) (-1/3) ∪ set.Icc (1/3) 1) :=
begin
  sorry
end

end ellipse_equation_and_lambda_range_l220_220374


namespace probability_three_primes_from_1_to_30_l220_220091

noncomputable def prob_three_primes : ℚ := 
  let primes := {x ∈ Finset.range 31 | Nat.Prime x }.card
  let total_combinations := (Finset.range 31).card.choose 3
  let prime_combinations := primes.choose 3
  prime_combinations / total_combinations

theorem probability_three_primes_from_1_to_30 : prob_three_primes = 10 / 339 := 
  by
    sorry

end probability_three_primes_from_1_to_30_l220_220091


namespace probability_of_exactly_three_correct_packages_l220_220292

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220292


namespace count_three_digit_integers_end7_divby21_l220_220789

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220789


namespace jellybeans_in_jar_orig_l220_220243

noncomputable def original_jellybeans (remaining :: ℝ)(decay_rate :: ℝ) : ℝ := remaining / decay_rate^3

theorem jellybeans_in_jar_orig : 
    let remain := 29.16 in
    let rate := 0.7 in
    original_jellybeans remain (rate) = 85 := 
by
    sorry

end jellybeans_in_jar_orig_l220_220243


namespace sum_of_squares_le_neg_nab_l220_220593

theorem sum_of_squares_le_neg_nab (n : ℕ) (a b : ℝ) (x : Fin n → ℝ)
  (h1 : ∑ i, x i = 0)
  (h2 : ∀ i, a ≤ x i ∧ x i ≤ b) :
  ∑ i, (x i) ^ 2 ≤ - (n : ℝ) * a * b := 
sorry

end sum_of_squares_le_neg_nab_l220_220593


namespace alice_minimum_speed_l220_220128

-- Conditions
def distance : ℝ := 60 -- The distance from City A to City B in miles
def bob_speed : ℝ := 40 -- Bob's constant speed in miles per hour
def alice_delay : ℝ := 0.5 -- Alice's delay in hours before she starts

-- Question as a proof statement
theorem alice_minimum_speed : ∀ (alice_speed : ℝ), alice_speed > 60 → 
  (alice_speed * (1.5 - alice_delay) < distance) → true :=
by
  sorry

end alice_minimum_speed_l220_220128


namespace stamps_in_last_page_l220_220028

-- Define the total number of books, pages per book, and stamps per original page.
def total_books : ℕ := 6
def pages_per_book : ℕ := 30
def original_stamps_per_page : ℕ := 7

-- Define the new stamps per page after reorganization.
def new_stamps_per_page : ℕ := 9

-- Define the number of fully filled books and pages in the fourth book.
def filled_books : ℕ := 3
def pages_in_fourth_book : ℕ := 26

-- Define the total number of stamps originally.
def total_original_stamps : ℕ := total_books * pages_per_book * original_stamps_per_page

-- Prove that the last page in the fourth book contains 9 stamps under the given conditions.
theorem stamps_in_last_page : 
  total_original_stamps / new_stamps_per_page - (filled_books * pages_per_book + pages_in_fourth_book) * new_stamps_per_page = 9 :=
by
  sorry

end stamps_in_last_page_l220_220028


namespace cube_root_inequality_solution_l220_220701

theorem cube_root_inequality_solution (x : ℝ) :
    (∛x + 4 / (∛x + 4) ≤ 0) ↔ (x ∈ Ioo (-∞) (-64) ∪ ({-8} : set ℝ)) :=
by
  sorry

end cube_root_inequality_solution_l220_220701


namespace scholarship_covers_percentage_l220_220222

variable (tuition_fee : ℝ) (job_pay : ℝ) (months : ℕ) (remaining_fee : ℝ)

def scholarship_coverage_percentage (tuition_fee job_pay : ℝ) (months remaining_fee : ℝ) : ℝ :=
  let total_earnings := job_pay * months
  let remaining_tuition := tuition_fee - total_earnings
  let scholarship_amount := remaining_tuition - remaining_fee
  (scholarship_amount / tuition_fee) * 100

theorem scholarship_covers_percentage :
  tuition_fee = 90 →
  job_pay = 15 →
  months = 3 →
  remaining_fee = 18 →
  scholarship_coverage_percentage 90 15 3 18 = 30 :=
by
  intros
  sorry    

end scholarship_covers_percentage_l220_220222


namespace gcd_2952_1386_l220_220617

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end gcd_2952_1386_l220_220617


namespace num_cube_roots_less_than_12_l220_220411

/-- There are 1727 positive whole numbers whose cube roots are less than 12 -/
theorem num_cube_roots_less_than_12 : 
  { n : ℕ // 0 < n ∧ ∃ x : ℕ, x = n ∧ (x : ℝ)^(1/3) < 12 }.card = 1727 := 
sorry

end num_cube_roots_less_than_12_l220_220411


namespace rectangle_circle_area_ratio_l220_220070

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end rectangle_circle_area_ratio_l220_220070


namespace number_of_odd_terms_in_expansion_l220_220832

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l220_220832


namespace line_segment_parametric_curve_l220_220059

noncomputable def parametric_curve (θ : ℝ) := 
  (2 + Real.cos θ ^ 2, 1 - Real.sin θ ^ 2)

theorem line_segment_parametric_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    ∃ x y : ℝ, (x, y) = parametric_curve θ ∧ 2 ≤ x ∧ x ≤ 3 ∧ x - y = 2) := 
sorry

end line_segment_parametric_curve_l220_220059


namespace number_of_intersections_l220_220691

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def line (x : ℝ) : Prop := x = 3

theorem number_of_intersections :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ circle 3 y1 ∧ circle 3 y2 :=
by
  sorry

end number_of_intersections_l220_220691


namespace trader_loss_percentage_l220_220633

theorem trader_loss_percentage :
  let SP := 325475
  let gain := 14 / 100
  let loss := 14 / 100
  let CP1 := SP / (1 + gain)
  let CP2 := SP / (1 - loss)
  let TCP := CP1 + CP2
  let TSP := SP + SP
  let profit_or_loss := TSP - TCP
  let profit_or_loss_percentage := (profit_or_loss / TCP) * 100
  profit_or_loss_percentage = -1.958 :=
by
  sorry

end trader_loss_percentage_l220_220633


namespace cricket_team_members_l220_220989

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (team_avg_age : ℕ) 
  (remaining_avg_age : ℕ) 
  (h1 : captain_age = 26)
  (h2 : wicket_keeper_age = 29)
  (h3 : team_avg_age = 23)
  (h4 : remaining_avg_age = 22) 
  (h5 : team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age) : 
  n = 11 := 
sorry

end cricket_team_members_l220_220989


namespace unit_cubes_with_red_vertices_l220_220492

open Set

noncomputable def lattice_cube (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  { p | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n ∧ 1 ≤ p.3 ∧ p.3 ≤ n }

def red_points (n k : ℕ) (T : Set (ℕ × ℕ × ℕ)) : Set (ℕ × ℕ × ℕ) :=
  { p | p ∈ T ∧ ... -- some property to ensure the points are red according to the problem statement }

def is_unit_cube (pts : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (x y z : ℕ), { (x, y, z), (x+1, y, z), (x, y+1, z), (x, y, z+1), 
                   (x+1, y+1, z), (x+1, y, z+1), (x, y+1, z+1),
                   (x+1, y+1, z+1) } ⊆ pts

theorem unit_cubes_with_red_vertices (n k : ℕ) 
  (T : Set (ℕ × ℕ × ℕ) := lattice_cube n)
  (R : Set (ℕ × ℕ × ℕ) := red_points n k T) 
  (h1 : 3 * n^2 - 3 * n + 1 + k ≤ Finset.card R)
  (h2 : ∀ (a b : ℕ × ℕ × ℕ), a ∈ R → b ∈ R → (parallel a b → line_segment a b ⊆ R)) :
  ∃ (cubes : Finset (Set (ℕ × ℕ × ℕ))), 
    ∀ cube ∈ cubes, is_unit_cube cube ∧ Finset.card cubes ≥ k := 
sorry

end unit_cubes_with_red_vertices_l220_220492


namespace sum_angles_180_l220_220039

-- Points A, B, R, E, and C lie on a circle.
variables (A B R E C : Type)

-- Measures of arcs BR and RE are given.
def arc_BR : ℝ := 48
def arc_RE : ℝ := 54

-- Angles S and R subtend arcs BE and EC respectively.
def arc_BE := arc_BR + arc_RE
def arc_EC := 360 - arc_BE
def angle_S := 1 / 2 * arc_BE
def angle_R := 1 / 2 * arc_EC

-- Sum of the measures of angles S and R.
def sum_angles := angle_S + angle_R

theorem sum_angles_180 : sum_angles = 180 := by
  sorry

end sum_angles_180_l220_220039


namespace remainder_prod_mod_7_l220_220728

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220728


namespace probability_three_correct_deliveries_is_one_sixth_l220_220323

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220323


namespace last_digit_to_appear_is_seven_l220_220229

def modified_fibonacci (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 then 2
  else (modified_fibonacci (n - 1) + modified_fibonacci (n - 2)) % 10

theorem last_digit_to_appear_is_seven :
  ∃ n, ((∀ m, modified_fibonacci m % 10 ≠ 7) ∧ modified_fibonacci n % 10 = 7) :=
sorry

end last_digit_to_appear_is_seven_l220_220229


namespace point_in_third_quadrant_l220_220437

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 1 + 2 * m < 0) : m < -1 / 2 := 
by 
  sorry

end point_in_third_quadrant_l220_220437


namespace sequence_general_formula_l220_220390

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  3^n + 1

theorem sequence_general_formula (a : ℕ → ℕ) (n : ℕ) (h₁ : sequence_sum n = ∑ i in Finset.range n, a i) :
  a n = if n = 1 then 4 else 2 * 3^(n - 1) := sorry

end sequence_general_formula_l220_220390


namespace norris_money_left_l220_220917

theorem norris_money_left :
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  total_savings - amount_spent = 10 :=
by
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  have h1 : total_savings = 85 := by rfl
  have h2 : total_savings - amount_spent = 85 - amount_spent := by rw h1
  have h3 : 85 - amount_spent = 85 - 75 := rfl
  have h4 : 85 - 75 = 10 := rfl
  exact eq.trans (eq.trans h2 h3) h4

end norris_money_left_l220_220917


namespace positive_integers_with_cube_roots_less_than_12_l220_220414

theorem positive_integers_with_cube_roots_less_than_12 :
  {n : ℕ // 0 < n ∧ n < 1728}.card = 1727 := sorry

end positive_integers_with_cube_roots_less_than_12_l220_220414


namespace probability_second_class_first_given_first_class_second_l220_220156

theorem probability_second_class_first_given_first_class_second :
  let total_items := 6
  let first_class_items := 4
  let second_class_items := 2
  let total_outcomes := (first_class_items * (first_class_items - 1)) + 
                        (second_class_items * first_class_items)
  let conditional_outcomes := second_class_items * first_class_items
  total_outcomes = 20 ∧ conditional_outcomes = 8
  → (conditional_outcomes : ℚ) / total_outcomes = (2 : ℚ) / 5 :=
by 
  intros
  cases total_outcomes with _ total_outcome_eq
  cases conditional_outcomes with _ conditional_outcome_eq
  rw [total_outcome_eq, conditional_outcome_eq]
  norm_num
  sorry

end probability_second_class_first_given_first_class_second_l220_220156


namespace product_mod_7_zero_l220_220730

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l220_220730


namespace stephanie_soy_sauce_bottles_l220_220048

theorem stephanie_soy_sauce_bottles :
  let ounces_per_bottle := 16
      ounces_per_cup := 8
      first_recipe_cups := 1 -- half of original 2 cups
      second_recipe_cups := 1 * 1.5 -- 1 cup increased by 1.5 times
      third_recipe_cups := 3 * 0.75 -- 3 cups reduced by 25%
      total_cups := first_recipe_cups + second_recipe_cups + third_recipe_cups
      total_ounces := total_cups * ounces_per_cup
      bottles_needed := Nat.ceil (total_ounces / ounces_per_bottle)
  in bottles_needed = 3 := sorry

end stephanie_soy_sauce_bottles_l220_220048


namespace freight_rails_intersection_l220_220987

-- Description of the conditions
def total_freight_wagons : ℕ := 30
def total_passenger_wagons : ℕ := 20
def adjacent_rails_per_wagon : ℕ := 2
def movable_passenger_wagons : ℕ := 10
def movable_freight_wagons : ℕ := 9
def non_movable_passenger_wagons : ℕ := total_passenger_wagons - movable_passenger_wagons
def non_movable_freight_wagons : ℕ := total_freight_wagons - movable_freight_wagons
def total_rails_for_passenger : ℕ := non_movable_passenger_wagons * adjacent_rails_per_wagon
def minimum_rails_for_freight : ℕ := non_movable_freight_wagons

-- The proof problem
theorem freight_rails_intersection : 
  ∃ (r : ℕ), r ∈ (finset.range total_rails_for_passenger) ∧ 
              (finset.count r (finset.range minimum_rails_for_freight)) ≥ 2 :=
sorry

end freight_rails_intersection_l220_220987


namespace find_t_l220_220422

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.root 4 (8 - t)) : 
  t = 7 / 2 :=
sorry

end find_t_l220_220422


namespace probability_three_correct_deliveries_l220_220340

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220340


namespace area_ratio_ABD_BCD_l220_220523

-- Define the geometric setup and conditions
variables {A B C D : Type} [triangle : EquilateralTriangle A B C] 
open EquilateralTriangle

-- Define point D on AC and the given angle condition
variables (D : Point) (hD_on_AC : D ∈ Segment A C) 
variables (h_angle : ∠DBC = 30)

-- State the theorem
theorem area_ratio_ABD_BCD (A B C D : Point) 
  [hABC : EquilateralTriangle A B C] 
  (hD_on_AC : D ∈ Segment A C) 
  (h_angle : ∠DBC = 30) : 
  area (Triangle.mk A B D) / area (Triangle.mk B C D) = 1 + Real.sqrt 3 := 
sorry

end area_ratio_ABD_BCD_l220_220523


namespace minimum_value_of_M_l220_220977

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

noncomputable def b_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 ^ (n - 1)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (a_n (i + 1) : ℝ) / (b_n (i + 1) : ℝ)

theorem minimum_value_of_M : ∀ n : ℕ, n > 0 → T_n n < 10 := by
  sorry

end minimum_value_of_M_l220_220977


namespace constant_term_binomial_expansion_l220_220106

theorem constant_term_binomial_expansion :
    let f (x : ℤ) := x ^ 2 + 3 / x ^ 2 in
    let term := (10.choose 5) * (3 ^ 5) in
    term = 61236 :=
by
  sorry

end constant_term_binomial_expansion_l220_220106


namespace zero_point_log_function_l220_220849

theorem zero_point_log_function (f : ℝ → ℝ) (a : ℝ) (h : f x = log 2 (x + a))
  (h_zero : f (-2) = 0) : a = 3 := 
  by 
  sorry

end zero_point_log_function_l220_220849


namespace painted_fraction_l220_220157

theorem painted_fraction : ∃ (x : ℚ), x + (1 - x) = 1 ∧ 1.3 * x + 0.5 * (1 - x) = 1 ∧ x = 5 / 8 :=
by 
  use (5 / 8)
  split
  · simp
  split
  · have h : 1.3 * (5 / 8 : ℚ) + 0.5 * (1 - 5 / 8 : ℚ) = 1 :=
      by norm_num
    assumption
  · simp

end painted_fraction_l220_220157


namespace inequality_holds_for_all_real_x_l220_220501

def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

def f (a x : ℝ) := 4*x^2 - 4*(a + 1)*x + 3*a + 3

def distinct_real_roots (a : ℝ) := (a^2 - a - 2 > 0) ∧ (a < 1) ∧ (11 - 5*a > 0)

theorem inequality_holds_for_all_real_x (a : ℝ) :
  distinct_real_roots a → (∀ x : ℝ, quadratic (a + 1) (-a) (a - 1) x < 0) ↔ a < -2/3 * sqrt 3 :=
by
  sorry

end inequality_holds_for_all_real_x_l220_220501


namespace dryer_sheets_per_load_l220_220548

theorem dryer_sheets_per_load (loads_per_week : ℕ) (cost_of_box : ℝ) (sheets_per_box : ℕ)
  (annual_savings : ℝ) (weeks_in_year : ℕ) (x : ℕ)
  (h1 : loads_per_week = 4)
  (h2 : cost_of_box = 5.50)
  (h3 : sheets_per_box = 104)
  (h4 : annual_savings = 11)
  (h5 : weeks_in_year = 52)
  (h6 : annual_savings = 2 * cost_of_box)
  (h7 : sheets_per_box * 2 = weeks_in_year * (loads_per_week * x)):
  x = 1 :=
by
  sorry

end dryer_sheets_per_load_l220_220548


namespace prod_ab_ac_eq_3sqrt48848_l220_220860

variable (A B C P Q X Y : Type)
variables [AcuteTriangle A B C] (foot_perp_C_AB : FootPerpendicular C A B P)
          (foot_perp_B_AC : FootPerpendicular B A C Q)
          (PQ_intersects_circumcircle : IntersectsCircumcircle PQ X Y)
          (XP : Real) (PQ : Real) (QY : Real)

def m_sqrt_n_sum : Nat := 22

theorem prod_ab_ac_eq_3sqrt48848 (hXP : XP = 12) (hPQ : PQ = 28) (hQY : QY = 16) :
  ∃ m n : ℕ, (AB * AC = m * Real.sqrt n) ∧ (m + n = m_sqrt_n_sum) := 
sorry

end prod_ab_ac_eq_3sqrt48848_l220_220860


namespace wicket_keeper_age_difference_l220_220058

def cricket_team_average_age : Nat := 24
def total_members : Nat := 11
def remaining_members : Nat := 9
def age_difference : Nat := 1

theorem wicket_keeper_age_difference :
  let total_age := cricket_team_average_age * total_members
  let remaining_average_age := cricket_team_average_age - age_difference
  let remaining_total_age := remaining_average_age * remaining_members
  let combined_age := total_age - remaining_total_age
  let average_age := cricket_team_average_age
  let wicket_keeper_age := combined_age - average_age
  wicket_keeper_age - average_age = 9 := 
by
  sorry

end wicket_keeper_age_difference_l220_220058


namespace students_not_take_test_l220_220912

theorem students_not_take_test
  (total_students : ℕ)
  (q1_correct : ℕ)
  (q2_correct : ℕ)
  (both_correct : ℕ)
  (h_total : total_students = 29)
  (h_q1 : q1_correct = 19)
  (h_q2 : q2_correct = 24)
  (h_both : both_correct = 19)
  : (total_students - (q1_correct + q2_correct - both_correct) = 5) :=
by
  sorry

end students_not_take_test_l220_220912


namespace norris_money_left_l220_220921

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l220_220921


namespace triangle_hypotenuse_length_l220_220097

noncomputable def length_of_hypotenuse (PQ PR U V : ℝ) :=
  let a := PQ in
  let b := PR in
  let PU := a / 4 in
  let UQ := 3 * a / 4 in
  let PV := b / 4 in
  let VR := 3 * b / 4 in
  UQ = 18 ∧ VR = 45 → Real.sqrt (a * a + b * b)

theorem triangle_hypotenuse_length :
  length_of_hypotenuse 24 60 1 1 = 12 * Real.sqrt 29 := by
  sorry

end triangle_hypotenuse_length_l220_220097


namespace students_in_all_three_activities_l220_220085

open Set

variable (U : Type) [Fintype U]
variable (students : Finset U)
variable (chess soccer music : Finset U)
variable (a : Finset U)

-- Define the conditions from the problem
def total_students := students.card = 30
def num_chess := chess.card = 15
def num_soccer := soccer.card = 18
def num_music := music.card = 12
def at_least_one_activity := ∀ s ∈ students, s ∈ chess ∨ s ∈ soccer ∨ s ∈ music
def at_least_two_activities := (chess ∩ soccer).card + (soccer ∩ music).card + (chess ∩ music).card - 2 * (chess ∩ soccer ∩ music).card = 14

-- Prove the number of students participating in all three activities is 1
theorem students_in_all_three_activities :
  total_students ∧ num_chess ∧ num_soccer ∧ num_music ∧ at_least_one_activity ∧ at_least_two_activities → (chess ∩ soccer ∩ music).card = 1 := sorry

end students_in_all_three_activities_l220_220085


namespace inequality_geq_8_l220_220580

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l220_220580


namespace triangle_right_angle_l220_220891

variable {α : Type} [LinearOrderedField α] [Trigonometric α]

/--
Given a △ABC with internal angles A, B, C at vertices A, B, C, and opposite sides of lengths a, b, c respectively,
if b * cos C + c * cos B = a * sin A, then △ABC is a right triangle.
-/
theorem triangle_right_angle
  {a b c : α} {A B C : α}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : 0 < A) (h₅ : 0 < B) (h₆ : 0 < C)
  (h₇ : A + B + C = π)
  (h₈ : b * cos C + c * cos B = a * sin A) :
  A = π / 2 :=
sorry

end triangle_right_angle_l220_220891


namespace difference_of_interchanged_digits_l220_220954

variable (x y : ℕ) (h₀ : x - y = 6)

theorem difference_of_interchanged_digits : (10 * x + y) - (10 * y + x) = 54 := by
  calc
    (10 * x + y) - (10 * y + x) = 9 * (x - y) : by linarith
                               ... = 9 * 6     : by rw [h₀]
                               ... = 54        : by norm_num

end difference_of_interchanged_digits_l220_220954


namespace both_dislike_radio_and_music_l220_220036

theorem both_dislike_radio_and_music (total_people : ℕ) (perc_dislike_radio perc_both : ℚ)
  (h_total : total_people = 1500)
  (h_perc_dislike_radio : perc_dislike_radio = 0.4)
  (h_perc_both : perc_both = 0.15) :
  ∃ (num_both : ℕ), num_both = 90 :=
by
  have h_dislike_radio : total_people * perc_dislike_radio = 600 := by sorry
  have h_both_dislike : 600 * perc_both = 90 := by sorry
  use 90
  rw [h_both_dislike]
  norm_num
  sorry

end both_dislike_radio_and_music_l220_220036


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220817

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220817


namespace pipe_cut_l220_220646

theorem pipe_cut (x : ℝ) (h1 : x + 2 * x = 177) : 2 * x = 118 :=
by
  sorry

end pipe_cut_l220_220646


namespace find_ellipse_equation_l220_220753

noncomputable def ellipse_equation (a b c : ℝ) : Prop :=
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)) ∧ a > b ∧ b > 0 ∧ (c = 1) ∧ (c / a = 1 / 2) ∧ (c^2 = a^2 - b^2)

theorem find_ellipse_equation : 
  ∃ a b : ℝ, a = 2 ∧ b = sqrt 3 ∧ ellipse_equation 2 (sqrt 3) 1 :=
sorry

end find_ellipse_equation_l220_220753


namespace problem_statement_l220_220387

theorem problem_statement :
  (∑ i in Finset.range (if i % 2 = 1 then 1 else 0) \binom 5 i) = 16 →
  ∃ T3 T4 T5 : Int,
    (T3 = 90 ∧ T4 = -270 ∧ T5 = 405) ∧
    ((binomial 5 2 * (x^(2/3))^3 * (3 * x^2)^2 = T3 * x^6) ∧
     (binomial 5 3 * (x^(2/3))^2 * (-3 * x^2)^3 = T4 * x^((22/3))) ∧
     (binomial 5 4 * (x^(2/3)) * (-3 * x^2)^4 = T5 * x^((26/3)))) :=
begin
  sorry
end

end problem_statement_l220_220387


namespace work_days_for_A_l220_220648

theorem work_days_for_A (x : ℕ) : 
  (∀ a b, 
    (a = 1 / (x : ℚ)) ∧ 
    (b = 1 / 20) ∧ 
    (8 * (a + b) = 14 / 15) → 
    x = 15) :=
by
  intros a b h
  have ha : a = 1 / (x : ℚ) := h.1
  have hb : b = 1 / 20 := h.2.1
  have hab : 8 * (a + b) = 14 / 15 := h.2.2
  sorry

end work_days_for_A_l220_220648


namespace least_number_to_subtract_l220_220618

theorem least_number_to_subtract (x : ℕ) (h : 509 - x = 45 * n) : ∃ x, (509 - x) % 9 = 0 ∧ (509 - x) % 15 = 0 ∧ x = 14 := by
  sorry

end least_number_to_subtract_l220_220618


namespace average_infection_rate_infected_computers_exceed_700_l220_220162

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l220_220162


namespace Yasmin_children_count_l220_220481

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end Yasmin_children_count_l220_220481


namespace transformed_sum_is_2051_l220_220591

variable (a : ℕ → ℤ)
variable (sum_a : ℤ)
variable (n : ℕ)

-- Conditions
def sum_of_100_numbers_is_2001: Prop := (∑ i in Finset.range 100, a i) = 2001

-- Transformation function
def transform (i : ℕ) : ℤ := a i + if i % 2 = 0 then (i + 1) else -(i + 1)

-- Proof statement
theorem transformed_sum_is_2051
  (h1: sum_of_100_numbers_is_2001)
  : (∑ i in Finset.range 100, transform i) = 2051 := 
sorry

end transformed_sum_is_2051_l220_220591


namespace three_correct_deliveries_probability_l220_220274

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220274


namespace both_trains_cross_bridge_in_time_l220_220153

noncomputable def speed_in_mps (speed_kmph : Float) : Float :=
  speed_kmph * 1000 / 3600

noncomputable def time_to_cross_bridge (length_train1 length_train2 : Float)
(speed_train1_kmph speed_train2_kmph : Float) (bridge_length : Float) : Float :=
let speed_train1 := speed_in_mps speed_train1_kmph
let speed_train2 := speed_in_mps speed_train2_kmph
let combined_length := length_train1 + length_train2 + bridge_length
let relative_speed := speed_train1 + speed_train2
combined_length / relative_speed

theorem both_trains_cross_bridge_in_time (
  length_train1 length_train2 : Float := 120
  speed_train1_kmph speed_train2_kmph : Float := 60
  length_train3 : Float := 150
  bridge_length : Float := 250
) : time_to_cross_bridge length_train1 length_train3 speed_train1_kmph speed_train2_kmph bridge_length ≈ 17.83 :=
by
  sorry

end both_trains_cross_bridge_in_time_l220_220153


namespace geometric_sequence_characterization_l220_220113

theorem geometric_sequence_characterization (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

end geometric_sequence_characterization_l220_220113


namespace cricket_overs_remaining_l220_220467

theorem cricket_overs_remaining 
  (run_rate_first_10_overs : ℝ) 
  (run_rate_remaining : ℝ) 
  (target : ℕ) 
  (total_overs : ℕ) 
  (played_overs : ℕ) 
  (runs_first_10_overs := run_rate_first_10_overs * 10)
  (remaining_runs := target - runs_first_10_overs)
  (required_overs := remaining_runs / run_rate_remaining) :
  total_overs - played_overs = 40 :=
by
  -- Given conditions
  have h1 : run_rate_first_10_overs = 4.8 := sorry,
  have h2 : run_rate_remaining = 5.85 := sorry,
  have h3 : target = 282 := sorry,
  have h4 : total_overs = 50 := sorry,
  have h5 : played_overs = 10 := sorry,
  -- Calculate intermediate values
  have h6 : runs_first_10_overs = 48 := sorry,
  have h7 : remaining_runs = 234 := sorry,
  have h8 : required_overs ≈ 40 := sorry,
  -- Conclusion
  sorry

end cricket_overs_remaining_l220_220467


namespace simplify_sqrt_expression_l220_220540

theorem simplify_sqrt_expression : sqrt(8 + 4 * sqrt 3) + sqrt(8 - 4 * sqrt 3) = 2 * sqrt 6 :=
by
  sorry

end simplify_sqrt_expression_l220_220540


namespace min_value_nS_l220_220592

def S (a₁ d n : ℕ) : ℕ := (n * a₁ + (n * (n - 1) / 2) * d)

theorem min_value_nS (a₁ d : ℕ)
  (h₁ : S a₁ d 10 = 0)
  (h₂ : S a₁ d 15 = 25) :
  ∃ n : ℕ, nS a₁ d n = -49 := sorry

end min_value_nS_l220_220592


namespace right_triangle_hypotenuse_enlargement_l220_220846

theorem right_triangle_hypotenuse_enlargement
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  ((5 * a)^2 + (5 * b)^2 = (5 * c)^2) :=
by sorry

end right_triangle_hypotenuse_enlargement_l220_220846


namespace distance_between_trees_l220_220632

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 255) (h2 : num_trees = 18) : yard_length / (num_trees - 1) = 15 := by
  sorry

end distance_between_trees_l220_220632


namespace binary_10011_is_19_l220_220558

-- Define the binary number
def binary_10011 := [1, 0, 0, 1, 1]  -- representing the binary number 10011

-- Function to convert a binary number represented as a list of bits to a decimal number
def binary_to_decimal (b : List Nat) : Nat :=
  b.foldr (λ (digit power : Nat) acc => acc + digit * 2^power) 0

-- Now state the theorem
theorem binary_10011_is_19 : binary_to_decimal [1, 0, 0, 1, 1] = 19 := by
  sorry

end binary_10011_is_19_l220_220558


namespace probability_three_correct_out_of_five_l220_220276

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220276


namespace train_speed_is_15_mps_l220_220572

variable (train_length : ℕ) (h1 : train_length = 450) (h2 : platform_length = train_length)
variable (time : ℕ) (h3 : time = 60)

def total_distance := train_length + platform_length

noncomputable def speed := total_distance / time

theorem train_speed_is_15_mps  : train_length = 450 → platform_length = train_length → time = 60 → speed = 15 := 
by
  intro h1 h2 h3
  sorry

end train_speed_is_15_mps_l220_220572


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220819

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220819


namespace largest_n_dividing_20_factorial_by_12_l220_220704

theorem largest_n_dividing_20_factorial_by_12 :
  ∃ n : ℕ, 12^n ∣ (20.factorial) ∧ (∀ m : ℕ, 12^m ∣ (20.factorial) → m ≤ 8) :=
begin
  sorry
end

end largest_n_dividing_20_factorial_by_12_l220_220704


namespace neg_neg_eq_l220_220141

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l220_220141


namespace product_of_invertible_function_labels_l220_220231

noncomputable def f6 : ℝ → ℝ := λ x, x^3 - 3 * x
def f7 : set (ℝ × ℝ) := {(-6, 4), (-5, 5), (-4, 1), (-3, 0), (-2, -1), (-1, -2), (0, -3), (1, -4), (2, -5), (3, -6)}
noncomputable def f8 : ℝ → ℝ := λ x, Real.tan x
noncomputable def f9 : ℝ → ℝ := λ x, 5 / x

theorem product_of_invertible_function_labels :
  (Function.bijective f6) ∧ ¬Function.bijective (λ x, if (x, f7)∈ f7 then f7.to_finset.find x else 0) ∧
  (∀ x ∈ ({-6, -5, -4, -3, -2, -1, 0, 1, 2, 3} : set ℤ), Real.tan x ≠ Real.nan → Function.bijective (λ (x : ℝ), Real.tan x)) ∧
  (Function.bijective f9) →
  6 * 8 * 9 = 432 :=
by sorry

end product_of_invertible_function_labels_l220_220231


namespace inequality_proof_equality_condition_l220_220822

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b ∧ a < 1) :=
sorry

end inequality_proof_equality_condition_l220_220822


namespace number_of_possible_r_l220_220576

theorem number_of_possible_r : 
  let r := (a : ℕ) / 10000 + (b : ℕ) / 1000 + (c : ℕ) / 100 + (d : ℕ) / 10 in
  r ∈ range 0.3125 0.3542 → 
  r.is_integer_arithmetic 418 :=
begin
  sorry
end

end number_of_possible_r_l220_220576


namespace symmetry_of_shifted_sine_l220_220773

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ :=
  Real.sin (ω * x + φ)
  
theorem symmetry_of_shifted_sine (ω : ℝ) (φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2)  
(hperiod : (2 * Real.pi) / ω = Real.pi)
(hshift_sym : ∀ x : ℝ, f (x - Real.pi / 3) ω φ = -f (-x) ω φ) :
  ∃ k : ℤ, x = (5 * Real.pi) / 12 + k * (Real.pi / 2) := 
begin
  sorry -- proof goes here
end

end symmetry_of_shifted_sine_l220_220773


namespace masha_nonnegative_l220_220522

theorem masha_nonnegative (a b c d : ℝ) (h1 : a + b = c * d) (h2 : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := 
by
  -- Proof is omitted
  sorry

end masha_nonnegative_l220_220522


namespace constant_term_is_60_l220_220868

noncomputable def constant_term_expansion (x : ℤ) : ℤ :=
  let term := (3 + x) * (x + 1/x)^6 in
  nexpr term 0  -- nexpr is a hypothetical function to retrieve the n-th power term (here 0-th power) from the expression

theorem constant_term_is_60 : 
  constant_term_expansion x = 60 := 
by 
  sorry

end constant_term_is_60_l220_220868


namespace intersection_A_B_l220_220377

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_A_B :
  A ∩ B = {-3, 1} := by
  sorry

end intersection_A_B_l220_220377


namespace max_sum_of_squares_diff_l220_220975

theorem max_sum_of_squares_diff {x y : ℕ} (h : x > 0 ∧ y > 0) (h_diff : x^2 - y^2 = 2016) :
  x + y ≤ 1008 ∧ ∃ x' y' : ℕ, x'^2 - y'^2 = 2016 ∧ x' + y' = 1008 :=
sorry

end max_sum_of_squares_diff_l220_220975


namespace smallest_arith_prog_l220_220261

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end smallest_arith_prog_l220_220261


namespace probability_of_exactly_three_correct_packages_l220_220287

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220287


namespace intersection_of_sets_l220_220148

open Set

theorem intersection_of_sets :
  let A := {x : ℝ | 2 < x ∧ x < 4}
  let B := {x : ℝ | x < 3 ∨ x > 5}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_sets_l220_220148


namespace three_digit_integers_product_24_count_l220_220244

-- Define a predicate stating that a three-digit number has digits (a, b, c) 
-- such that the product of these digits is 24.
def digits_product_is_24 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
               1 ≤ a ∧ a ≤ 9 ∧    -- a is a single digit
               0 ≤ b ∧ b ≤ 9 ∧    -- b is a single digit
               0 ≤ c ∧ c ≤ 9 ∧    -- c is a single digit
               a * b * c = 24 

-- Define the proof problem stating that there are exactly 21 three-digit integers 
-- whose digits' product is 24.
theorem three_digit_integers_product_24_count : 
  (finset.filter digits_product_is_24 (finset.Icc 100 999)).card = 21 := 
sorry

end three_digit_integers_product_24_count_l220_220244


namespace segment_in_triangle_length_le_longest_side_l220_220629

theorem segment_in_triangle_length_le_longest_side
  (ABC : Type) [triangle ABC] (MN : segment ABC) : 
  length(MN) ≤ max (length (side1 ABC)) (max (length (side2 ABC)) (length (side3 ABC))) :=
sorry

end segment_in_triangle_length_le_longest_side_l220_220629


namespace probability_three_correct_out_of_five_l220_220281

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220281


namespace solve_log_eq_l220_220700

noncomputable def log_base (b x : ℝ) : ℝ :=
  log x / log b

theorem solve_log_eq : 
  (log_base (√(3:ℝ) (16)) 256 = log_base 2 64 ) := 
by
  sorry

end solve_log_eq_l220_220700


namespace PQRS_is_parallelogram_l220_220494

variables {A B C D E P Q R S : Type} [EuclideanGeometry]

-- Definitions of points and their relationships
def isIntersectionPoint (E : Type) (A B C D : Type) := 
  E = intersection (diagonal A B) (diagonal C D)

def isCircumcenter (P : Type) (A B C E : Type) := 
  P = circumcenter (triangle A B E)

def areCircumcenters (P Q R S : Type) (A B C D E : Type) :=
  isCircumcenter P A B E ∧ isCircumcenter Q B C E ∧ 
  isCircumcenter R C D E ∧ isCircumcenter S A D E

theorem PQRS_is_parallelogram (hE : isIntersectionPoint E A B C D) (hCircum : areCircumcenters P Q R S A B C D E) :
  is_parallelogram P Q R S :=
sorry

end PQRS_is_parallelogram_l220_220494


namespace sum_of_midpoints_l220_220081

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l220_220081


namespace product_mod_seven_l220_220710

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220710


namespace find_area_MOI_l220_220874

noncomputable def incenter_coords (a b c : ℝ) (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

noncomputable def shoelace_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem find_area_MOI :
  let A := (0, 0)
  let B := (8, 0)
  let C := (0, 17)
  let O := (4, 8.5)
  let I := incenter_coords 8 15 17 A B C
  let M := (6.25, 6.25)
  shoelace_area M O I = 25.78125 :=
by
  sorry

end find_area_MOI_l220_220874


namespace apples_rate_per_kg_l220_220096

variable (A : ℝ)

theorem apples_rate_per_kg (h : 8 * A + 9 * 65 = 1145) : A = 70 :=
sorry

end apples_rate_per_kg_l220_220096


namespace yellow_flowers_killed_by_fungus_l220_220654

variable (r y o p : ℕ) -- red, yellow, orange, purple flowers that survived
variable (k_y : ℕ) -- number of yellow flowers killed by fungus
variable (total_bouquets : ℕ := 36)
variable (flowers_per_bouquet : ℕ := 9)
variable (initial_seeds : ℕ := 125)
variable (k_r : ℕ := 45)
variable (k_o : ℕ := 30)
variable (k_p : ℕ := 40)

theorem yellow_flowers_killed_by_fungus :
  let total_flowers_required := total_bouquets * flowers_per_bouquet
  let red_survived := initial_seeds - k_r
  let orange_survived := initial_seeds - k_o
  let purple_survived := initial_seeds - k_p
  let total_survived_other_than_yellow := red_survived + orange_survived + purple_survived
  let yellow_needed := total_flowers_required - total_survived_other_than_yellow
  let yellow_survived := initial_seeds - k_y
  red_survived = 80 → orange_survived = 95 → purple_survived = 85 →
  total_flowers_required = 324 →
  total_survived_other_than_yellow = 260 →
  yellow_needed = 64 →
  yellow_survived = yellow_needed →
  (initial_seeds - yellow_survived = k_y) :=
by
  intros 
  rw [total_survived_other_than_yellow, yellow_needed, red_survived, orange_survived, purple_survived, total_flowers_required]
  sorry

end yellow_flowers_killed_by_fungus_l220_220654


namespace mushroom_distribution_l220_220538

-- Define the total number of mushrooms
def total_mushrooms : ℕ := 120

-- Define the number of girls
def number_of_girls : ℕ := 5

-- Auxiliary function to represent each girl receiving pattern
def mushrooms_received (n :ℕ) (total : ℕ) : ℝ :=
  (n + 20) + 0.04 * (total - (n + 20))

-- Define the equality function to check distribution condition
def equal_distribution (girls : ℕ) (total : ℕ) : Prop :=
  ∀ i j : ℕ, i < girls → j < girls → mushrooms_received i total = mushrooms_received j total

-- Main proof statement about the total mushrooms and number of girls following the distribution
theorem mushroom_distribution :
  total_mushrooms = 120 ∧ number_of_girls = 5 ∧ equal_distribution number_of_girls total_mushrooms := 
by 
  sorry

end mushroom_distribution_l220_220538


namespace prize_sector_area_l220_220173

-- Define the radius of the circle
def radius : ℝ := 8

-- Define the probability of landing on the "Prize" sector
def probability : ℝ := 3 / 8

-- Calculate and prove the area of the "Prize" sector expressed in terms of π
theorem prize_sector_area :
  let total_area := Real.pi * radius^2
  ∃ (area : ℝ), probability = area / total_area ∧ area = 24 * Real.pi :=
by
  let total_area := Real.pi * radius^2
  use 24 * Real.pi
  constructor
  . sorry
  . sorry

end prize_sector_area_l220_220173


namespace good_number_iff_gcd_phi_l220_220637

def is_good_number (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ a : ℤ, ∃ m : ℕ, m > 0 ∧ m^m % n = a % n 

theorem good_number_iff_gcd_phi (n : ℕ) (phi : ℕ := Nat.totient n) :
  is_good_number n ↔ Nat.gcd n phi = 1 := 
sorry

end good_number_iff_gcd_phi_l220_220637


namespace intersection_sum_l220_220471

-- Define the curve C in polar coordinates
def C (θ : ℝ) : ℝ := 2 * cos θ

-- Define the transformation to get C1
def translated_stretched_C1 (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define the parametric line l
def l (t : ℝ) : ℝ × ℝ :=
  (t * cos (π / 3), sqrt 3 + t * sin (π / 3))

-- Define the point P
def P : ℝ × ℝ := (0, sqrt 3)

theorem intersection_sum (A B : ℝ × ℝ) (hA : translated_stretched_C1 A.1 A.2) (hB : translated_stretched_C1 B.1 B.2) 
(hA_line : ∃ t₁, l t₁ = A) (hB_line : ∃ t₂, l t₂ = B) : 
  (1 / dist P A) + (1 / dist P B) = 3 / 2 := 
sorry

end intersection_sum_l220_220471


namespace sum_binom_solutions_25_l220_220112

theorem sum_binom_solutions_25 :
  (∑ n in { n : ℕ | n ≤ 25 ∧ binom 25 n + binom 25 13 = binom 26 13}.toFinset, n) = 25 := by
  sorry

end sum_binom_solutions_25_l220_220112


namespace nikola_jobs_l220_220516

theorem nikola_jobs (ants_needed : ℕ) (food_per_ant : ℕ) (food_cost_per_ounce : ℚ) 
  (job_start_cost : ℚ) (leaf_cost : ℚ) (leaves_raked : ℕ) 
  (total_savings : ℚ)
  (h1 : ants_needed = 400)
  (h2 : food_per_ant = 2)
  (h3 : food_cost_per_ounce = 0.1)
  (h4 : job_start_cost = 5)
  (h5 : leaf_cost = 0.01)
  (h6 : leaves_raked = 6000)
  (h7 : total_savings = 80) :
  let total_food := ants_needed * food_per_ant,
      total_cost_food := total_food * food_cost_per_ounce,
      earnings_from_leaves := leaves_raked * leaf_cost,
      earnings_from_jobs := total_savings - earnings_from_leaves,
      number_of_jobs := earnings_from_jobs / job_start_cost
  in number_of_jobs = 4 :=
by {
  sorry
}

end nikola_jobs_l220_220516


namespace avg_rate_first_half_l220_220911

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l220_220911


namespace math_problem_l220_220887

noncomputable def S (k : ℕ) (a : ℕ → ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, (Nat.choose k i) * a i)

theorem math_problem
  {n : ℕ} (hn : n > 1) (a : ℕ → ℝ)
  (h0 : ∀ i ≤ n, 0 ≤ a i) :
  (1 / n) * (Finset.range n).sum (λ k, (S k a) ^ 2) - (1 / (n ^ 2)) * ((Finset.range (n + 1)).sum (λ k, S k a)) ^ 2
  ≤ (4 / 45) * ((S n a) - (S 0 a)) ^ 2 :=
  sorry

end math_problem_l220_220887


namespace three_correct_deliveries_probability_l220_220273

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220273


namespace range_of_derivative_at_one_l220_220903

theorem range_of_derivative_at_one :
  ∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ 5 * Real.pi / 12 →
  let f (x : ℝ) := (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ in
  ∃ y : ℝ, y ∈ set.Icc (sqrt 2) 2 ∧ deriv f 1 = y :=
by
  intros θ hθ f,
  sorry

end range_of_derivative_at_one_l220_220903


namespace root_in_interval_l220_220063

def f (x : ℝ) := log x + 2 * x - 6

theorem root_in_interval : ∃ x ∈ Ioo 2.5 2.75, f x = 0 :=
by
  have continuous_f : ContinuousOn f (Ioo 2 (3 : ℝ)) := sorry
  have monotonic_f : StrictMonoOn f (Ioo 2 3) := sorry
  have f_2_5_neg : f 2.5 < 0 := by
    sorry
  have f_2_75_neg : f 2.75 < 0 := by
    sorry
  have f_3_pos : f 3 > 0 := by
    sorry
  -- Use the Intermediate Value Theorem for finding the root in interval
  obtain ⟨c, hc1, hc2⟩ : ∃ c ∈ Ioo 2.75 3, f c = 0 := sorry
  exact hc1

end root_in_interval_l220_220063


namespace intersecting_lines_sum_c_d_l220_220064

theorem intersecting_lines_sum_c_d 
  (c d : ℚ)
  (h1 : 2 = 1 / 5 * (3 : ℚ) + c)
  (h2 : 3 = 1 / 5 * (2 : ℚ) + d) : 
  c + d = 4 :=
by sorry

end intersecting_lines_sum_c_d_l220_220064


namespace find_xyz_l220_220839

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
    (h4 : x * y = 16 * real.cbrt 4)
    (h5 : x * z = 28 * real.cbrt 4)
    (h6 : y * z = 112 / real.cbrt 4) :
    x * y * z = 112 * real.sqrt 7 := 
sorry

end find_xyz_l220_220839


namespace neg_neg_eq_l220_220142

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l220_220142


namespace slope_of_line_l220_220438

theorem slope_of_line (θ : ℝ) (h : θ = 30) :
  ∃ k, k = Real.tan (60 * (π / 180)) ∨ k = Real.tan (120 * (π / 180)) := by
    sorry

end slope_of_line_l220_220438


namespace polynomial_subtraction_l220_220120

variables {a b : ℝ}

theorem polynomial_subtraction :
  (3 * a^2 * b - 6 * a * b^2) - (2 * a^2 * b - 3 * a * b^2) = - a^2 * b :=
by 
  calc
    (3 * a^2 * b - 6 * a * b^2) - (2 * a^2 * b - 3 * a * b^2)
    = 3 * a^2 * b - 6 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 : by ring
    ... = (3 * a^2 * b - 2 * a^2 * b) + (- 6 * a * b^2 + 3 * a * b^2) : by ring
    ... = a^2 * b + (- 3 * a * b^2) : by ring
    ... = a^2 * b - 3 * a * b^2 : by ring
    ... = - a^2 * b : by ring
  sorry  -- Omit the actual proof for this exercise

end polynomial_subtraction_l220_220120


namespace bean_seedlings_l220_220175

theorem bean_seedlings
  (beans_per_row : ℕ)
  (pumpkins : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (radishes_per_row : ℕ)
  (rows_per_bed : ℕ) (beds : ℕ)
  (H_beans_per_row : beans_per_row = 8)
  (H_pumpkins : pumpkins = 84)
  (H_pumpkins_per_row : pumpkins_per_row = 7)
  (H_radishes : radishes = 48)
  (H_radishes_per_row : radishes_per_row = 6)
  (H_rows_per_bed : rows_per_bed = 2)
  (H_beds : beds = 14) :
  (beans_per_row * ((beds * rows_per_bed) - (pumpkins / pumpkins_per_row) - (radishes / radishes_per_row)) = 64) :=
by
  sorry

end bean_seedlings_l220_220175


namespace max_non_right_triangle_centers_l220_220367

theorem max_non_right_triangle_centers (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  ∃ k, k = m + n - 2 ∧ (∀ (centers : finset (ℕ × ℕ)), centers.card = k → ¬∃ (a b c : (ℕ × ℕ)), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2) = 0) :=
begin
  sorry
end

end max_non_right_triangle_centers_l220_220367


namespace range_of_function_l220_220241

theorem range_of_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → ∃ (y : ℝ), y = 2 - real.sqrt (-(x^2) + 4*x) ∧ 0 ≤ y ∧ y ≤ 2 :=
by
  intro x hx
  sorry

end range_of_function_l220_220241


namespace jackson_paintable_area_l220_220880

namespace PaintWallCalculation

def length := 14
def width := 11
def height := 9
def windowArea := 70
def bedrooms := 4

def area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

def paintable_area_one_bedroom : ℕ :=
  area_one_bedroom - windowArea

def total_paintable_area : ℕ :=
  bedrooms * paintable_area_one_bedroom

theorem jackson_paintable_area :
  total_paintable_area = 1520 :=
sorry

end PaintWallCalculation

end jackson_paintable_area_l220_220880


namespace sequence_general_term_l220_220588

-- Define the sequence
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 1

-- State the theorem
theorem sequence_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

end sequence_general_term_l220_220588


namespace x_pow_y_equals_nine_l220_220424

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l220_220424


namespace greatest_value_of_b_l220_220249

theorem greatest_value_of_b : ∃ b, (∀ a, (-a^2 + 7 * a - 10 ≥ 0) → (a ≤ b)) ∧ b = 5 :=
by
  sorry

end greatest_value_of_b_l220_220249


namespace solve_picnic_problem_l220_220659

def picnic_problem : Prop :=
  ∃ (M W A C : ℕ), 
    M = W + 80 ∧ 
    A = C + 80 ∧ 
    M + W = A ∧ 
    A + C = 240 ∧ 
    M = 120

theorem solve_picnic_problem : picnic_problem :=
  sorry

end solve_picnic_problem_l220_220659


namespace count_three_digit_integers_end7_divby21_l220_220790

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220790


namespace probability_three_correct_packages_l220_220313

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220313


namespace stratified_sampling_b_members_l220_220649

variable (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) (sampleSize : ℕ)

-- Conditions from the problem
def condition1 : groupA = 45 := by sorry
def condition2 : groupB = 45 := by sorry
def condition3 : groupC = 60 := by sorry
def condition4 : sampleSize = 10 := by sorry

-- The proof problem statement
theorem stratified_sampling_b_members : 
  (sampleSize * groupB) / (groupA + groupB + groupC) = 3 :=
by sorry

end stratified_sampling_b_members_l220_220649


namespace range_of_m_l220_220439

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = -3) (h3 : x + y > 0) : m > 2 :=
by
  sorry

end range_of_m_l220_220439


namespace sum_six_consecutive_integers_l220_220552

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l220_220552


namespace binomial_expansion_coefficient_l220_220238

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coefficient :
  let expr := (1 / 2 : ℝ) * (x^2) - (1 / x^(1/2)) in
  let r := 4 in
  let term_coeff := (-1)^r * (1 / 2)^(6 - r) * binomial_coefficient 6 r in
  term_coeff = (15 / 4) := by
  -- Proof goes here
  sorry

end binomial_expansion_coefficient_l220_220238


namespace probability_of_exactly_three_correct_packages_l220_220291

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220291


namespace positive_integers_with_cube_roots_less_than_12_l220_220415

theorem positive_integers_with_cube_roots_less_than_12 :
  {n : ℕ // 0 < n ∧ n < 1728}.card = 1727 := sorry

end positive_integers_with_cube_roots_less_than_12_l220_220415


namespace worker_late_time_l220_220612

theorem worker_late_time (T : ℕ) (norm_speed : ℝ) (new_speed : ℝ) (D : ℕ) :
  T = 60 →
  new_speed = 5 / 6 * norm_speed →
  D = norm_speed * T →
  let T_new := D / new_speed in 
  T_new - T = 12 :=
by
  intros hT hnew_speed hD
  rw [hT, hnew_speed, hD]
  -- Insert the steps to show T_new = 72 and resulting T_new - T = 12
  sorry

end worker_late_time_l220_220612


namespace octagon_diagonals_l220_220991

theorem octagon_diagonals : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by
  intros n hn
  rw hn
  sorry

end octagon_diagonals_l220_220991


namespace remainder_prod_mod_7_l220_220724

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220724


namespace iterative_average_difference_l220_220213

-- Definition of iterative average
def iterative_average (seq : List ℚ) : ℚ :=
  (seq.tail.foldl (λ acc x, (acc + x) / 2) seq.head)

-- List of all permutations of the set {1, 2, 3, 4, 5}
def all_permutations : List (List ℚ) := [1, 2, 3, 4, 5].permutations

-- Maximum and Minimum iterative average from all permutations
def max_iterative_average : ℚ := all_permutations.map iterative_average |> List.maximum' (by decide)
def min_iterative_average : ℚ := all_permutations.map iterative_average |> List.minimum' (by decide)

-- Theorem stating the difference
theorem iterative_average_difference : max_iterative_average - min_iterative_average = 17/8 := 
by sorry

end iterative_average_difference_l220_220213


namespace exists_two_lucky_tickets_in_ten_consecutive_tickets_l220_220877

def is_lucky (n : ℕ) : Prop :=
  let digits := (n % 1000000).digits 10 in
  let first_three := digits.take 3 in
  let last_three := digits.drop 3 in
  first_three.sum = last_three.sum

theorem exists_two_lucky_tickets_in_ten_consecutive_tickets :
  ∃ (a b : ℕ), a < b ∧ b < a + 10 ∧ is_lucky a ∧ is_lucky b :=
sorry

end exists_two_lucky_tickets_in_ten_consecutive_tickets_l220_220877


namespace no_integer_coordinates_between_A_and_B_l220_220655

section
variable (A B : ℤ × ℤ)
variable (Aeq : A = (2, 3))
variable (Beq : B = (50, 305))

theorem no_integer_coordinates_between_A_and_B :
  (∀ P : ℤ × ℤ, P.1 > 2 ∧ P.1 < 50 ∧ P.2 = (151 * P.1 - 230) / 24 → False) :=
by
  sorry
end

end no_integer_coordinates_between_A_and_B_l220_220655


namespace quadratic_root_in_interval_l220_220560

theorem quadratic_root_in_interval 
  (a b c : ℝ) 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_root_in_interval_l220_220560


namespace composite_with_most_divisors_less_than_20_l220_220029

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

def number_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem composite_with_most_divisors_less_than_20 :
  ∃ n, is_composite n ∧ n < 20 ∧ (∀ m, is_composite m ∧ m < 20 → number_of_divisors n ≥ number_of_divisors m) ∧ (n = 12 ∨ n = 18) :=
sorry

end composite_with_most_divisors_less_than_20_l220_220029


namespace perfect_cubes_count_540_l220_220785

theorem perfect_cubes_count_540 : 
  let n := 540 in
  let factors := {d | d > 0 ∧ d ∣ n} in
  let perfect_cubes := {d | d ∈ factors ∧ ∀ p k, prime p → p^k ∣ d → k % 3 = 0} in
  fintype.card perfect_cubes = 2 := sorry

end perfect_cubes_count_540_l220_220785


namespace magnitude_of_complex_root_l220_220434

theorem magnitude_of_complex_root
  (p : ℝ) 
  (h_discriminant : p^2 - 16 < 0) 
  (x₁ x₂ : ℂ)
  (h_roots : ∃ (a b : ℂ), a = x₁ ∧ b = x₂ ∧ (∀ (x : ℂ), x^2 + p*x + 4 = 0 → x = x₁ ∨ x = x₂)) :
  |x₁| = 2 :=
by
  -- Begin proof here
  sorry

end magnitude_of_complex_root_l220_220434


namespace sets_relationship_l220_220992

variables {U : Type*} (A B C : Set U)

theorem sets_relationship (h1 : A ∩ B = C) (h2 : B ∩ C = A) : A = C ∧ ∃ B, A ⊆ B := by
  sorry

end sets_relationship_l220_220992


namespace monotonic_intervals_max_b_value_l220_220771

noncomputable theory

def f (a x : ℝ) := a * x - 1 - real.log x

theorem monotonic_intervals (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < (1 / a) → f a x < 0) ∧ 
  (∀ x : ℝ, x ≥ (1 / a) → f a x ≥ 0) :=
sorry

theorem max_b_value (b : ℝ) :
  (∀ x : ℝ, 0 < x → f 1 x ≥ b * x - 2) →
  b ≤ 1 - (1 / real.exp 2) :=
sorry

end monotonic_intervals_max_b_value_l220_220771


namespace number_of_spheres_l220_220167

theorem number_of_spheres
  (diameter_cylinder : ℝ) (height_cylinder : ℝ) (diameter_sphere : ℝ)
  (h1 : diameter_cylinder = 8) (h2 : height_cylinder = 48) (h3 : diameter_sphere = 8) :
  768 * 3 / 256 = 9 :=
by
  -- Definitions from conditions
  let radius_cylinder := diameter_cylinder / 2
  let volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder

  let radius_sphere := diameter_sphere / 2
  let volume_sphere := 4 / 3 * Real.pi * radius_sphere^3

  -- Calculate number of spheres
  have volume_cylinder_expr : volume_cylinder = Real.pi * 4^2 * 48 := by sorry
  have volume_sphere_expr : volume_sphere = (4/3) * Real.pi * 4^3 := by sorry

  have num_spheres : volume_cylinder / volume_sphere = 9 := by sorry

  exact num_spheres

end number_of_spheres_l220_220167


namespace area_covered_by_AP_l220_220375

noncomputable def area_swept_by_segment (A : ℝ × ℝ) (P : ℝ → ℝ × ℝ) (t1 t2 : ℝ) : ℝ :=
sorry

def A : ℝ × ℝ := (2, 0)

def P (t : ℝ) : ℝ × ℝ :=
  (Real.sin (2 * t - Real.pi / 3), Real.cos (2 * t - Real.pi / 3))

def t1 : ℝ := Real.pi / 12 -- 15 degrees in radians
def t2 : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem area_covered_by_AP : area_swept_by_segment A P t1 t2 = sorry :=
by 
  -- Using t1 and t2 as the bounds
  have h1 : P t1 = (Real.sin ((2 * Real.pi / 12) - (Real.pi / 3)), Real.cos ((2 * Real.pi / 12) - (Real.pi / 3))) := sorry,
  have h2 : P t2 = (Real.sin ((2 * Real.pi / 4) - (Real.pi / 3)), Real.cos ((2 * Real.pi / 4) - (Real.pi / 3))) := sorry,
  sorry

end area_covered_by_AP_l220_220375


namespace probability_of_two_distinct_extreme_points_l220_220770

open Set
open Finset

noncomputable def function_has_two_distinct_extreme_points_probability : ℚ :=
  let a_values := {1, 2, 3}
  let b_values := {0, 1, 2}
  let all_combinations := a_values.product b_values
  let favourable_combinations := all_combinations.filter (λ (ab : ℕ × ℕ), ab.1 > ab.2)
  favourable_combinations.card / all_combinations.card

theorem probability_of_two_distinct_extreme_points :
  function_has_two_distinct_extreme_points_probability = 2 / 3 := by
  sorry

end probability_of_two_distinct_extreme_points_l220_220770


namespace product_mod_seven_l220_220706

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220706


namespace part1_focus_passes_through_part2_line_equation_l220_220893

-- Part 1: Define conditions and the theorem for the first question
theorem part1_focus_passes_through {x1 x2 : ℝ} 
  (A : ( ℝ × ℝ )) (B : ( ℝ × ℝ )) (on_parabola : ∀ z ∈ {A, B}, 2 * (z.1)^2 = z.2) 
  (perpendicular_bisector : ∀ x y, (A.1 + B.1) / 2 = x ∧ (A.2 + B.2) / 2 = y) :
  x1 + x2 = 0 ↔ (∃ k b : ℝ, ∀ x y, l = (y = k*x + b) ∧ l passes_through F(0, (1/8))) :=
sorry

-- Part 2: Define conditions and the theorem for the second question
theorem part2_line_equation {x1 x2 : ℝ} 
  (A : ( ℝ × ℝ )) (B : ( ℝ × ℝ )) (on_parabola : ∀ z ∈ {A, B}, 2 * (z.1)^2 = z.2) 
  (perpendicular_bisector : ∀ x y, (A.1 + B.1) / 2 = x ∧ (A.2 + B.2) / 2 = y) 
  (x1_val : x1 = 1) (x2_val : x2 = -3) :
  ∃ k b : ℝ, l = (y = k*x + b) ∧ l.equation = (x - 4*y + 41 = 0) :=
sorry

end part1_focus_passes_through_part2_line_equation_l220_220893


namespace ratio_of_areas_l220_220068

variable (l w r : ℝ)
variable (h1 : l = 2 * w)
variable (h2 : 6 * w = 2 * real.pi * r)

theorem ratio_of_areas : (l * w) / (real.pi * r^2) = 2 * real.pi / 9 :=
by sorry

end ratio_of_areas_l220_220068


namespace maximal_correlation_sin_nU_sin_mU_eq_zero_l220_220359

noncomputable def maximal_correlation_of_sine (n m : ℕ) : ℝ :=
  let U := uniform_of_real ([0, 2 * real.pi])
  let sin_nU := fun (u : ℝ) => real.sin(n * u)
  let sin_mU := fun (u : ℝ) => real.sin(m * u)
  let correlation (X Y : ℝ → ℝ) := 
    (E[X U * Y U] - E[X U] * E[Y U]) / (sqrt (E[(X U)^2] - (E[X U])^2) * sqrt (E[(Y U)^2] - (E[Y U])^2))
  sup {f g : ℝ → ℝ | measurable f ∧ measurable g ∧ ∀ u, is_finite (f (U u)) ∧ is_finite (g (U u))} 
    (correlation f g)

theorem maximal_correlation_sin_nU_sin_mU_eq_zero (n m : ℕ) (h_n : n > 0) (h_m : m > 0) :
  maximal_correlation_of_sine n m = 0 := 
sorry

end maximal_correlation_sin_nU_sin_mU_eq_zero_l220_220359


namespace village_foods_sales_l220_220099

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l220_220099


namespace value_of_c_in_base8_perfect_cube_l220_220736

theorem value_of_c_in_base8_perfect_cube (c : ℕ) (h : 0 ≤ c ∧ c < 8) :
  4 * 8^2 + c * 8 + 3 = x^3 → c = 0 := by
  sorry

end value_of_c_in_base8_perfect_cube_l220_220736


namespace Yasmin_children_count_l220_220482

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end Yasmin_children_count_l220_220482


namespace gen_term_an_min_value_Tn_l220_220392

-- Definition of the arithmetic sequence and the sum of the first n terms
def arith_seq (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 d : ℤ) (n : ℕ) : ℤ := n * a_1 + n * (n - 1) / 2 * d

-- Given conditions
variables (a_3 : ℤ) (S_6 : ℤ) (h_a3 : a_3 = 10) (h_S6 : S_6 = 72)
noncomputable def d : ℤ := 4
noncomputable def a_1 : ℤ := 2
def a : ℕ → ℤ := arith_seq a_1 d
def S : ℕ → ℤ := sum_arith_seq a_1 d

-- Equivalent proof problems
theorem gen_term_an (n : ℕ) : a n = 4 * n - 2 := by 
  -- Given h_a3 and h_S6
  sorry

-- Definition of the sequence b_n
def b (n : ℕ) : ℤ := 1 / 2 * (a n) - 30
def T (n : ℕ) : ℤ := ∑ i in finset.range n, b (i + 1)

-- Proof of the minimum value of T_n
theorem min_value_Tn (n : ℕ) : T 15 = -225 := by 
  sorry

end gen_term_an_min_value_Tn_l220_220392


namespace probability_three_correct_deliveries_l220_220332

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220332


namespace inequality_condition_l220_220574

theorem inequality_condition {x : ℝ} (h : -1/2 ≤ x ∧ x < 1) : (2 * x + 1) / (1 - x) ≥ 0 :=
sorry

end inequality_condition_l220_220574


namespace algebraic_expression_value_l220_220749

theorem algebraic_expression_value (x : Fin 2004 → ℝ)
  (h : ∑ i in Finset.range 2004, |x i - (i + 1)| = 0) :
  (2 * x 0 - 2 * x 1 + ∑ i in Finset.range (2001), (-2) * x (i + 1)) + 2 * x 2003 = 6 :=
sorry

end algebraic_expression_value_l220_220749


namespace trees_needed_l220_220172

theorem trees_needed
  (C : ℝ) (d : ℝ)
  (hC : C = 1200)
  (hd : d = 10) :
  let W := C / d,
      P := W - 1 in
  W = 120 ∧ P = 119 := sorry

end trees_needed_l220_220172


namespace base_number_l220_220433

theorem base_number (a x : ℕ) (h1 : a ^ x - a ^ (x - 2) = 3 * 2 ^ 11) (h2 : x = 13) : a = 2 :=
by
  sorry

end base_number_l220_220433


namespace combined_salaries_l220_220973

theorem combined_salaries (A B C D E : ℝ) 
  (hC : C = 11000) 
  (hAverage : (A + B + C + D + E) / 5 = 8200) : 
  A + B + D + E = 30000 := 
by 
  sorry

end combined_salaries_l220_220973


namespace square_area_l220_220702

theorem square_area (P : ℝ) (hP : P = 32) : ∃ A : ℝ, A = 64 ∧ A = (P / 4) ^ 2 :=
by {
  sorry
}

end square_area_l220_220702


namespace count_three_digit_numbers_with_2_before_3_l220_220043

theorem count_three_digit_numbers_with_2_before_3 : 
  ∃ n : ℕ, 
  n = 12 ∧ 
  (∀ (a b c : ℕ), 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
    a ∈ {1, 2, 3, 4, 5} ∧ 
    b ∈ {1, 2, 3, 4, 5} ∧ 
    c ∈ {1, 2, 3, 4, 5} ∧ 
    ((a = 2 ∧ b ≠ 3 ∧ c = 3) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (b = 2 ∧ c = 3) ∨ 
     (a ≠ 2 ∧ b = 2 ∧ c = 3)) 
    → n = 12 ) :=
by {
  sorry
}

end count_three_digit_numbers_with_2_before_3_l220_220043


namespace sum_even_102_to_200_l220_220082

theorem sum_even_102_to_200 :
  let sum_first_50_even := (50 / 2) * (2 + 100)
  let sum_102_to_200 := 3 * sum_first_50_even
  sum_102_to_200 = 7650 :=
by
  let sum_first_50_even := (50 / 2) * (2 + 100)
  let sum_102_to_200 := 3 * sum_first_50_even
  have : sum_102_to_200 = 7650 := by
    rw [sum_first_50_even, sum_102_to_200]
    sorry
  exact this

end sum_even_102_to_200_l220_220082


namespace all_planes_land_at_specific_airports_l220_220174

/-- Conditions / Parameters -/
structure AirportScenario where
  airports_count : ℕ
  specific_airports : ℕ
  farthest_airport_landing : ∀ (a b : Fin airports_count), a ≠ b → ℕ
  unique_distances : ∀ (a b c d : Fin airports_count), a ≠ b → c ≠ d → farthest_airport_landing a b ≠ farthest_airport_landing c d

/-- The problem statement is transformed to this theorem -/
theorem all_planes_land_at_specific_airports (scenario : AirportScenario) : 
  scenario.airports_count = 1985 → 
  scenario.specific_airports = 50 → 
  (∀ a : Fin scenario.airports_count, ∃ b : Fin scenario.specific_airports, scenario.farthest_airport_landing a b) :=
sorry

end all_planes_land_at_specific_airports_l220_220174


namespace find_x_l220_220152

theorem find_x (x : ℝ) (h : 5020 - (x / 100.4) = 5015) : x = 502 :=
sorry

end find_x_l220_220152


namespace num_participation_schemes_l220_220362
-- Import the complete math library

theorem num_participation_schemes : 
  ∀ (n r : ℕ) (A B : ℕ) (total_no_A_B no_A_B : ℕ),
  (n = 5) → (r = 3) → 
  (total_no_A_B = finset.card (finset.permutations (finset.range n) r)) →
  (no_A_B = finset.card (finset.permutations (finset.range (n - 2)) r)) →
  total_no_A_B - no_A_B = 54 := 
by
  intros _ _ A B total_no_A_B no_A_B n_def r_def total_def no_def
  sorry

end num_participation_schemes_l220_220362


namespace probability_exactly_three_correct_l220_220350

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220350


namespace divisor_count_1800_1800_l220_220247

theorem divisor_count_1800_1800 :
  let prime_factorization_1800 := (2, 3) :: (3, 2) :: (5, 2) :: []
  let prime_factorization_1800_1800 := (2, 5400) :: (3, 3600) :: (5, 3600) :: []
  the number of positive integer divisors of (2^(5400) * 3^(3600) * 5^(3600)) 
  that are divisible by exactly 1000 positive integers
  = 36 :=
by
  let prime_factorization_1800 := (2, 3) :: (3, 2) :: (5, 2) :: []
  let prime_factorization_1800_1800 := (2, 5400) :: (3, 3600) :: (5, 3600) :: []
  have h_prime_fact_1800 : 1800 = 2^3 * 3^2 * 5^2 := by sorry
  have h_prime_fact_1800_1800 : 1800 ^ 1800 = 2 ^ 5400 * 3 ^ 3600 * 5 ^ 3600 := by sorry
  have h_divisor_count : 
    ∃ (a b c : ℕ), (2^a * 3^b * 5^c ∣ 1800 ^ 1800) ∧ ((a+1)*(b+1)*(c+1) = 1000) := by sorry
  exact 36

end divisor_count_1800_1800_l220_220247


namespace toll_for_18_wheel_truck_l220_220597

noncomputable def toll (x : ℕ) : ℝ :=
  2.50 + 0.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let num_wheels := 18
  let wheels_on_front_axle := 2
  let wheels_per_other_axle := 4
  let num_other_axles := (num_wheels - wheels_on_front_axle) / wheels_per_other_axle
  let total_num_axles := num_other_axles + 1
  toll total_num_axles = 4.00 :=
by
  sorry

end toll_for_18_wheel_truck_l220_220597


namespace vasya_password_combinations_l220_220996

theorem vasya_password_combinations : 
  (∃ (A B C : ℕ), A ≠ 2 ∧ B ≠ 2 ∧ C ≠ 2 ∧ A = 0 ∨ A = 1 ∨ A = 3 ∨ A = 4 ∨ A = 5 ∨ A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9 ∧
                   B = 0 ∨ B = 1 ∨ B = 3 ∨ B = 4 ∨ B = 5 ∨ B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9 ∧
                   C = 0 ∨ C = 1 ∨ C = 3 ∨ C = 4 ∨ C = 5 ∨ C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9 ∧
                   A ≠ B ∧ B ≠ C ∧ A = lastDigit ∧ firstDigit = lastDigit) →
  (classical.some (9 * 8 * 7) = 504) :=
by
  sorry

end vasya_password_combinations_l220_220996


namespace problem_inequality_l220_220583

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l220_220583


namespace find_antiderivative_l220_220384

-- Define our function and the conditions
variable {ℝ : Type} [LinearOrderedField ℝ]

theorem find_antiderivative (f : ℝ → ℝ) : 
  (∀ x : ℝ, deriv f x = 4 * x^3) → f 1 = -1 → 
  ∃ C : ℝ, f = λ x, x^4 + C :=
by
  intro h_deriv h_init
  use (-2) -- Given our solution steps
  ext x
  have : f = λ x, x^4 - 2 := 
    sorry -- This part represents the skipped proof
  exact this

end find_antiderivative_l220_220384


namespace solve_quadratic_roots_l220_220077

theorem solve_quadratic_roots : ∀ x : ℝ, (x - 1)^2 = 1 → (x = 2 ∨ x = 0) :=
by
  sorry

end solve_quadratic_roots_l220_220077


namespace probability_exactly_three_correct_l220_220349

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220349


namespace greatest_divisor_of_arithmetic_sequence_sum_l220_220107

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), (∃ (n : ℕ), n = 12 * x + 66 * c) → (6 ∣ n) :=
by
  intro x c _ h
  cases h with n h_eq
  use 6
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l220_220107


namespace probability_three_common_books_l220_220914

open Finset

/-- Probability that Jenna and Marco select exactly 3 common books out of 12 -/
theorem probability_three_common_books :
  let total_ways := (choose 12 4) * (choose 12 4),
      successful_ways := (choose 12 3) * (choose 9 1) * (choose 8 1) in
  (successful_ways : ℚ) / total_ways = (32 / 495 : ℚ) :=
by
  sorry

end probability_three_common_books_l220_220914


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220815

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220815


namespace log_equation_solution_l220_220104

theorem log_equation_solution (x : ℝ) : 
  (x > 0) ∧ (log 2 (x^2) + 2 * log x 8 = 392 / (log 2 (x^3) + 20 * log x 32)) ↔ 
  (x = 1/32 ∨ x = 32 ∨ x = 1/4 ∨ x = 4) := 
sorry 

end log_equation_solution_l220_220104


namespace ratio_a_to_c_l220_220510

-- Define the variables and ratios
variables (x y z a b c d : ℝ)

-- Define the conditions as given ratios
variables (h1 : a / b = 2 * x / (3 * y))
variables (h2 : b / c = z / (5 * z))
variables (h3 : a / d = 4 * x / (7 * y))
variables (h4 : d / c = 7 * y / (3 * z))

-- Statement to prove the ratio of a to c
theorem ratio_a_to_c (x y z a b c d : ℝ) 
  (h1 : a / b = 2 * x / (3 * y)) 
  (h2 : b / c = z / (5 * z)) 
  (h3 : a / d = 4 * x / (7 * y)) 
  (h4 : d / c = 7 * y / (3 * z)) : a / c = 2 * x / (15 * y) :=
sorry

end ratio_a_to_c_l220_220510


namespace constant_term_in_expansion_l220_220767

theorem constant_term_in_expansion : 
  (∃ n : ℕ, (∀ k : ℕ, ratio_coeff_k (x^2 - (1 : ℂ) / x^(1 / 2)) n 2 / ratio_coeff_k (x^2 - (1 : ℂ) / x^(1 / 2)) n 4 = 3 / 14) ∧ 
        n = 10) →
  constant_term (x^2 - (1 : ℂ) / x^(1 / 2)) 10 = 45 :=
by
  sorry

end constant_term_in_expansion_l220_220767


namespace final_jacket_price_is_correct_l220_220203

-- Define the initial price, the discounts, and the tax rate
def initial_price : ℝ := 120
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def sales_tax : ℝ := 0.05

-- Calculate the final price using the given conditions
noncomputable def price_after_first_discount := initial_price * (1 - first_discount)
noncomputable def price_after_second_discount := price_after_first_discount * (1 - second_discount)
noncomputable def final_price := price_after_second_discount * (1 + sales_tax)

-- The theorem to prove
theorem final_jacket_price_is_correct : final_price = 75.60 := by
  -- The proof is omitted
  sorry

end final_jacket_price_is_correct_l220_220203


namespace mike_corvette_average_speed_l220_220909

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l220_220909


namespace radius_of_circle_l220_220966

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem radius_of_circle : 
  ∃ r : ℝ, circle_area r = circle_circumference r → r = 2 := 
by 
  sorry

end radius_of_circle_l220_220966


namespace percentage_decrease_in_area_l220_220628

constant L B : ℝ -- Assume that L and B are positive real numbers representing the original length and breadth.

-- Conditions
def new_length (L : ℝ) := 0.80 * L
def new_breadth (B : ℝ) := 0.90 * B

-- Definition of areas
def original_area (L B : ℝ) := L * B
def new_area (L B : ℝ) := (new_length L) * (new_breadth B)

-- Proof of percentage decrease in area
theorem percentage_decrease_in_area (L B : ℝ) (hL : 0 < L) (hB : 0 < B)
  : (original_area L B - new_area L B) / original_area L B * 100 = 28 := 
by sorry

end percentage_decrease_in_area_l220_220628


namespace remainder_prod_mod_7_l220_220726

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220726


namespace total_distance_traveled_l220_220360

theorem total_distance_traveled (x : ℕ) (d_1 d_2 d_3 d_4 d_5 d_6 : ℕ) 
  (h1 : d_1 = 60 / x) 
  (h2 : d_2 = 60 / (x + 3)) 
  (h3 : d_3 = 60 / (x + 6)) 
  (h4 : d_4 = 60 / (x + 9)) 
  (h5 : d_5 = 60 / (x + 12)) 
  (h6 : d_6 = 60 / (x + 15)) 
  (hx1 : x ∣ 60) 
  (hx2 : (x + 3) ∣ 60) 
  (hx3 : (x + 6) ∣ 60) 
  (hx4 : (x + 9) ∣ 60) 
  (hx5 : (x + 12) ∣ 60) 
  (hx6 : (x + 15) ∣ 60) :
  d_1 + d_2 + d_3 + d_4 + d_5 + d_6 = 39 := 
sorry

end total_distance_traveled_l220_220360


namespace balance_two_diamonds_three_bullets_l220_220089

-- Define the variables
variables (a b c : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * a + b = 9 * c
def condition2 : Prop := a = b + c

-- Goal is to prove two diamonds (2 * b) balance three bullets (3 * c)
theorem balance_two_diamonds_three_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 
  2 * b = 3 * c := 
by 
  sorry

end balance_two_diamonds_three_bullets_l220_220089


namespace probability_three_correct_out_of_five_l220_220278

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220278


namespace line_parallel_to_skew_is_intersecting_or_skew_l220_220205

variable {R : Type*} [EuclideanSpace R]

-- Define what it means for lines to be skew
def skew (l1 l2 : Line R) : Prop :=
  ¬ (l1 = l2) ∧ ¬ (l1 ∥ l2) ∧ ¬ (∃ P, P ∈ l1 ∧ P ∈ l2)

-- Define what it means for lines to be parallel
def parallel (l1 l2 : Line R) : Prop :=
l1 ∥ l2

-- Define the specific problem
theorem line_parallel_to_skew_is_intersecting_or_skew (l1 l2 l3 : Line R) (h1 : skew l1 l2) (h2 : parallel l3 l1) : 
  (∃ P, P ∈ l3 ∧ P ∈ l2) ∨ skew l3 l2 :=
sorry

end line_parallel_to_skew_is_intersecting_or_skew_l220_220205


namespace total_games_proof_l220_220446

variable (score1 score2 score3 games1 games2 games3 total_games : ℕ)
variable (avg_runs total_runs : ℕ)

-- Conditions
def conditions : Prop :=
  score1 = 1 ∧ games1 = 1 ∧
  score2 = 4 ∧ games2 = 2 ∧
  score3 = 5 ∧ games3 = 3 ∧
  avg_runs = 4 ∧
  total_runs = (score1 * games1 + score2 * games2 + score3 * games3) ∧
  total_games * avg_runs = total_runs

-- Proof problem statement
theorem total_games_proof (h : conditions) : total_games = 6 :=
by
  sorry

end total_games_proof_l220_220446


namespace max_candies_eaten_l220_220087

theorem max_candies_eaten : 
  ∃ candies : ℕ, candies = 351 ∧ 
  (∀ (n : ℕ) (num_board : list ℕ), num_board.length = 27 → 
   (∀ t ≤ n, ∃ x y : ℕ, 
    x ∈ num_board ∧ y ∈ num_board ∧ 
    num_board = erase num_board x ∧ num_board = erase num_board y ∧ 
    num_board.length = 27 - (2 * t) + t + 1 ∧  
    candies = sum (erase_all num_board (x * y))))
:=
begin
  sorry
end

end max_candies_eaten_l220_220087


namespace number_of_ways_to_distribute_balls_l220_220532

theorem number_of_ways_to_distribute_balls : 
  ∃ n : ℕ, n = 81 ∧ n = 3^4 := 
by sorry

end number_of_ways_to_distribute_balls_l220_220532


namespace superdomino_probability_l220_220664

-- Definitions based on conditions
def is_superdomino (a b : ℕ) : Prop := 0 ≤ a ∧ a ≤ 12 ∧ 0 ≤ b ∧ b ≤ 12
def is_superdouble (a b : ℕ) : Prop := a = b
def total_superdomino_count : ℕ := 13 * 13
def superdouble_count : ℕ := 13

-- Proof statement
theorem superdomino_probability : (superdouble_count : ℚ) / total_superdomino_count = 13 / 169 :=
by
  sorry

end superdomino_probability_l220_220664


namespace complex_expression_identity_l220_220018

noncomputable section

variable (x y : ℂ) 

theorem complex_expression_identity (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x^2 + x*y + y^2 = 0) : 
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 := 
by 
  sorry

end complex_expression_identity_l220_220018


namespace secret_known_on_monday_l220_220031

def students_know_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem secret_known_on_monday :
  ∃ n : ℕ, students_know_secret n = 3280 ∧ (n + 1) % 7 = 0 :=
by
  sorry

end secret_known_on_monday_l220_220031


namespace painted_cubes_unique_l220_220652

-- Definitions of conditions
def is_painted_cubes_equiv (c1 c2 : Cube) : Prop :=
  ∃ r : Rotation, r.rotate c1 = c2

def painted_cube (c: Cube) : Prop :=
  c.faces.count Red = 1 ∧
  c.faces.count Blue = 4 ∧
  c.faces.count Green = 1 ∧
  c.faces.count otherColors = 0

-- Statement of the problem
theorem painted_cubes_unique : 
  (∃ c1 : Cube, painted_cube c1 ∧ 
  ∀ c2 : Cube, painted_cube c2 → is_painted_cubes_equiv c1 c2 → false) 
  = 4 :=
sorry

end painted_cubes_unique_l220_220652


namespace no_zeros_iff_k_gt_inv_e_two_distinct_zeros_ln_sum_gt_2_l220_220012

open Real

-- Define the function f as given in the problem.
def f (x k : ℝ) := log x - k * x

-- Statement for part (1)
theorem no_zeros_iff_k_gt_inv_e {k : ℝ} :
  (∀ x : ℝ, x > 0 → f x k ≠ 0) ↔ k > 1 / exp 1 := sorry

-- Statement for part (2)
theorem two_distinct_zeros_ln_sum_gt_2 {k x1 x2 : ℝ} (hx1 : x1 > 0) (hx2 : x2 > 0) (hDistinct : x1 ≠ x2)
  (hf_x1 : f x1 k = 0) (hf_x2 : f x2 k = 0) :
  log x1 + log x2 > 2 := sorry

end no_zeros_iff_k_gt_inv_e_two_distinct_zeros_ln_sum_gt_2_l220_220012


namespace necess_suff_cond_odd_function_l220_220019

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.sin (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def P (ω ϕ : ℝ) : Prop := f ω ϕ 0 = 0
def Q (ω ϕ : ℝ) : Prop := is_odd (f ω ϕ)

theorem necess_suff_cond_odd_function (ω ϕ : ℝ) : P ω ϕ ↔ Q ω ϕ := by
  sorry

end necess_suff_cond_odd_function_l220_220019


namespace lines_are_perpendicular_l220_220404

variables {m n : Line} {α β : Plane}

-- Assume the conditions from the problem
axiom line_perpendicular_to_plane : (m ⊥ α)
axiom another_line_perpendicular_to_plane : (n ⊥ β)
axiom planes_are_perpendicular : (α ⊥ β)

-- Define the theorem we need to prove
theorem lines_are_perpendicular (line_perpendicular_to_plane : m ⊥ α) 
                                (another_line_perpendicular_to_plane : n ⊥ β) 
                                (planes_are_perpendicular : α ⊥ β) : m ⊥ n := 
sorry

end lines_are_perpendicular_l220_220404


namespace no_integer_y_makes_Q_perfect_square_l220_220240

def Q (y : ℤ) : ℤ := y^4 + 8 * y^3 + 18 * y^2 + 10 * y + 41

theorem no_integer_y_makes_Q_perfect_square :
  ¬ ∃ y : ℤ, ∃ b : ℤ, Q y = b^2 :=
by
  intro h
  rcases h with ⟨y, b, hQ⟩
  sorry

end no_integer_y_makes_Q_perfect_square_l220_220240


namespace eel_count_l220_220695

theorem eel_count 
  (x y z : ℕ)
  (h1 : y + z = 12)
  (h2 : x + z = 14)
  (h3 : x + y = 16) : 
  x + y + z = 21 := 
by 
  sorry

end eel_count_l220_220695


namespace cost_of_four_enchiladas_and_five_tacos_l220_220534

noncomputable def t : ℝ := 2.40 / 9
noncomputable def e : ℝ := 2.6665 / 4

theorem cost_of_four_enchiladas_and_five_tacos : 4 * e + 5 * t = 4.00 :=
by
  have h1 : 4 * e + 5 * t = 4.00 := by linarith { t := t, e := e }
  have h2 : 5 * e + 4 * t = 4.40 := by linarith { t := t, e := e }
  rw [← h1, ← h2]
  linarith

end cost_of_four_enchiladas_and_five_tacos_l220_220534


namespace probability_three_correct_deliveries_l220_220343

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220343


namespace combined_water_leak_l220_220114

theorem combined_water_leak
  (largest_rate : ℕ)
  (medium_rate : ℕ)
  (smallest_rate : ℕ)
  (time_minutes : ℕ)
  (h1 : largest_rate = 3)
  (h2 : medium_rate = largest_rate / 2)
  (h3 : smallest_rate = medium_rate / 3)
  (h4 : time_minutes = 120) :
  largest_rate * time_minutes + medium_rate * time_minutes + smallest_rate * time_minutes = 600 := by
  sorry

end combined_water_leak_l220_220114


namespace behavior_for_x_less_than_zero_l220_220639

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y
def is_convex (f : ℝ → ℝ) : Prop := ∀ x y, f'' x > 0

theorem behavior_for_x_less_than_zero
  (h1 : is_odd_function f)
  (h2 : ∀ x > 0, f' x > 0)
  (h3 : ∀ x > 0, f'' x > 0) :
  ∀ x < 0, f' x > 0 ∧ f'' x < 0 :=
sorry

end behavior_for_x_less_than_zero_l220_220639


namespace tetrahedron_pairs_l220_220821

theorem tetrahedron_pairs (tetra_edges : ℕ) (h_tetra : tetra_edges = 6) :
  ∀ (num_pairs : ℕ), num_pairs = (tetra_edges * (tetra_edges - 1)) / 2 → num_pairs = 15 :=
by
  sorry

end tetrahedron_pairs_l220_220821


namespace calculate_fraction_l220_220741

noncomputable def f : ℝ → ℝ := sorry

axiom f_recursion (x : ℝ) (hx : x ≥ 0) : f(x + 1) + 1 = f(x) + 43 / ((x + 1) * (x + 2))
axiom f_initial : f(0) = 2020

theorem calculate_fraction : 101 / f(2020) = 2.35 := by
  sorry

end calculate_fraction_l220_220741


namespace denomination_of_four_bills_l220_220027

theorem denomination_of_four_bills (X : ℕ) (h1 : 10 * 20 + 8 * 10 + 4 * X = 300) : X = 5 :=
by
  -- proof goes here
  sorry

end denomination_of_four_bills_l220_220027


namespace remainder_prod_mod_7_l220_220725

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220725


namespace least_naturial_number_exists_l220_220250

def partition_has_product_triple (s : Finset ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b = c

theorem least_naturial_number_exists :
  ∃ n : ℕ, (∀ (s1 s2 : Finset ℕ), s1 ∪ s2 = (Finset.range n).map Nat.succ ∧ s1 ∩ s2 = ∅ →
    partition_has_product_triple s1 ∨ partition_has_product_triple s2) ∧
    (∀ k < n, ¬ (∀ (s1 s2 : Finset ℕ), s1 ∪ s2 = (Finset.range k).map Nat.succ ∧ s1 ∩ s2 = ∅ →
    partition_has_product_triple s1 ∨ partition_has_product_triple s2)) :=
sorry

end least_naturial_number_exists_l220_220250


namespace remainder_prod_mod_7_l220_220723

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220723


namespace Alpha_Beta_meet_at_Alpha_Beta_meet_again_l220_220033

open Real

-- Definitions and conditions
def A : ℝ := -24
def B : ℝ := -10
def C : ℝ := 10
def Alpha_speed : ℝ := 4
def Beta_speed : ℝ := 6

-- Question 1: Prove that Alpha and Beta meet at -10.4
theorem Alpha_Beta_meet_at : 
  ∃ t : ℝ, (A + Alpha_speed * t = C - Beta_speed * t) ∧ (A + Alpha_speed * t = -10.4) :=
  sorry

-- Question 2: Prove that after reversing at t = 2, Alpha and Beta meet again at -44
theorem Alpha_Beta_meet_again :
  ∃ t z : ℝ, 
    ((t = 2) ∧ (4 * t + (14 - 4 * t) + (14 - 4 * t + 20) = 40) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = C - Beta_speed * t - Beta_speed * z) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = -44)) :=
  sorry  

end Alpha_Beta_meet_at_Alpha_Beta_meet_again_l220_220033


namespace sum_inequality_l220_220126

theorem sum_inequality (n : ℕ) (hn : n > 2) (a : ℕ → ℝ) (ha : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≤ 1) :
  let A (k : ℕ) := (∑ i in finset.range k, a i.succ) / k in 
  abs (∑ k in finset.range n, a k.succ - ∑ k in finset.range n, A (k + 1)) < (n-1) / 2 :=
sorry

end sum_inequality_l220_220126


namespace nth_term_sequence_l220_220199

theorem nth_term_sequence (n : ℕ) (hn : n > 0) : 
  ∃ a b, (a = n) ∧ (b = n^2 + 2) ∧ (a / b = n / (n^2 + 2)) :=
by
  exists n
  exists (n^2 + 2)
  sorry

end nth_term_sequence_l220_220199


namespace park_road_area_l220_220657

theorem park_road_area 
  (side_length_road : ℝ) 
  (outer_perimeter : ℝ)
  (road_width : ℝ)
  (h_road_width : road_width = 3)
  (h_outer_perimeter : outer_perimeter = 600)
  (h_side_length_road : side_length_road = outer_perimeter / 4 + 2 * road_width) :
  let park_side_length := side_length_road + 2 * road_width in
  let park_area := park_side_length^2 in
  let inner_area := (side_length_road)^2 in
  let road_area := park_area - inner_area in
  road_area = 1836 :=
by 
  sorry

end park_road_area_l220_220657


namespace probability_three_correct_packages_l220_220307

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220307


namespace minimum_pieces_to_guarantee_fish_and_sausage_l220_220656

theorem minimum_pieces_to_guarantee_fish_and_sausage :
  ∀ (f_squares s_squares : set (ℕ × ℕ)),
  (∀ (x y : ℕ × ℕ), x ∈ f_squares ∧ y ∈ f_squares ∧ x ≠ y → 
    x.1 ≤ 8 ∧ x.2 ≤ 8 ∧ y.1 ≤ 8 ∧ y.2 ≤ 8 ∧ ∃ (a b : ℕ × ℕ), 
    set.disjoint {a, b} f_squares ∧ (∀ (i j : ℕ × ℕ), (i, j) ∈ f_squares) ∨ (two_fish_in_6x6 ∧ one_sausage_in_3x3))),
  (∃ (a b : ℕ × ℕ),
  a ∈ f_squares ∧ a ∈ s_squares ∧ b ∉ f_squares ∧ b ∉ s_squares ∧ ∃ (A B : finset (ℕ × ℕ)), 
  A.card = 2 ∧ B.card = 1 ∧ ∀  (u v: ℕ × ℕ), 
  (u, v) ∈ A ∧ (u, v) ∈ B ∧ u ≠ v → set_disjoint A B) → set.nonempty
(sorry)

end minimum_pieces_to_guarantee_fish_and_sausage_l220_220656


namespace rocky_knockout_percentage_l220_220935

theorem rocky_knockout_percentage:
  ∀ (K: ℕ), 
  (190 * 20 * K = 190 * 19 * 100) → 
  (K / 190.0) * 100 = 50 :=
by
  intro K
  intro hk
  sorry

end rocky_knockout_percentage_l220_220935


namespace distance_from_origin_to_point_P_l220_220452

def origin : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (12, -5)

theorem distance_from_origin_to_point_P :
  (Real.sqrt ((12 - 0)^2 + (-5 - 0)^2)) = 13 := by
  simp [Real.sqrt, *]
  sorry

end distance_from_origin_to_point_P_l220_220452


namespace num_ordered_pairs_divisible_by_11_l220_220491

/- 
 Problem statement: 
 Prove that the number of ordered pairs (i, j) of divisors of 6^8 such that d_i - d_j is divisible by 11 is 665.
-/ 

theorem num_ordered_pairs_divisible_by_11 :
  let m := 6^8 in
  let divisors := {d : ℕ | d ∣ m} in
  let condition := λ (i j : ℕ), ((i ∈ divisors) ∧ (j ∈ divisors) ∧ ((i - j) % 11 = 0)) in
  finset.card ((finset.filter (λ p : ℕ × ℕ, condition p.1 p.2) 
                  (finset.product (finset.filter (λ d, d ∣ m) finset.univ) 
                                  (finset.filter (λ d, d ∣ m) finset.univ))) = 665

sorry

end num_ordered_pairs_divisible_by_11_l220_220491


namespace total_climbing_time_l220_220000

def first_flight_time : ℕ := 25
def time_increment : ℕ := 7
def stop_time : ℕ := 10
def total_flights : ℕ := 7
noncomputable def total_time : ℕ :=
  (List.sum (List.map (λ n, first_flight_time + (n - 1) * time_increment)
                       (List.range total_flights))) + 2 * stop_time

theorem total_climbing_time : total_time = 342 := by
  sorry

end total_climbing_time_l220_220000


namespace a_eq_zero_l220_220062

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 :=
sorry

end a_eq_zero_l220_220062


namespace village_food_sales_l220_220101

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l220_220101


namespace probability_three_correct_deliveries_l220_220342

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220342


namespace probability_of_three_primes_l220_220092

-- Defining the range of integers
def range := {n : ℕ | 1 ≤ n ∧ n ≤ 30}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Defining the set of prime numbers from 1 to 30
def primes : set ℕ := {n | n ∈ range ∧ is_prime n}

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := n.choose k

-- Define probability function
def probability_three_primes : ℚ :=
  binomial 10 3 / binomial 30 3

theorem probability_of_three_primes : 
  probability_three_primes = 6 / 203 :=
by sorry

end probability_of_three_primes_l220_220092


namespace quadratic_increasing_l220_220468

theorem quadratic_increasing (x : ℝ) (hx : x > 1) : ∃ y : ℝ, y = (x-1)^2 + 1 ∧ ∀ (x₁ x₂ : ℝ), x₁ > x ∧ x₂ > x₁ → (x₁ - 1)^2 + 1 < (x₂ - 1)^2 + 1 := by
  sorry

end quadratic_increasing_l220_220468


namespace rhombus_area_of_intersecting_triangles_l220_220606

theorem rhombus_area_of_intersecting_triangles (s : ℝ) (h1 : s = 4 * real.sqrt 3) : 
  let d1 := s, d2 := 12 - s 
  in (1/2) * d1 * d2 = 24 * real.sqrt 3 - 24 :=
by
  sorry

end rhombus_area_of_intersecting_triangles_l220_220606


namespace lines_passing_through_neg1_0_l220_220566

theorem lines_passing_through_neg1_0 (k : ℝ) :
  ∀ x y : ℝ, (y = k * (x + 1)) ↔ (x = -1 → y = 0 ∧ k ≠ 0) :=
by
  sorry

end lines_passing_through_neg1_0_l220_220566


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220804

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220804


namespace sum_six_consecutive_integers_l220_220551

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l220_220551


namespace number_of_lines_in_grid_equal_156_l220_220410

-- Each point (i, j, k) in the grid where i, j, k are in the range 1 to 5
def point := (ℕ × ℕ × ℕ)

def is_point_in_grid (p : point) : Prop :=
  ∃ (i j k : ℕ), 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ 1 ≤ k ∧ k ≤ 5 ∧ p = (i, j, k)

-- Defining a line that passes through four distinct points in the grid
def is_line (p1 p2 p3 p4 : point) : Prop :=
  is_point_in_grid p1 ∧ is_point_in_grid p2 ∧ is_point_in_grid p3 ∧ is_point_in_grid p4 ∧
  ∃ (a b c : ℤ), (∀ (n : ℕ), n ∈ {0, 1, 2, 3} → (p1 = (a + n * (p2.1 - p1.1), b + n * (p2.2 - p1.2), c + n * (p2.3 - p1.3)))) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)

noncomputable def count_lines_passing_through_four_points : ℕ :=
  156

theorem number_of_lines_in_grid_equal_156 :
  ∃ (n : ℕ), n = count_lines_passing_through_four_points ∧ n = 156 :=
by
  use count_lines_passing_through_four_points
  split
  · refl
  · sorry

end number_of_lines_in_grid_equal_156_l220_220410


namespace find_fx_find_extrema_l220_220399

def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 + x ^ 2 + b * x
def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x + b
def g (x : ℝ) (a b : ℝ) : ℝ := f x a b + f' x a b
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

theorem find_fx (a b : ℝ) (h_g_odd : odd_function (g a b)) : f = (λ x, -1/3 * x ^ 3 + x ^ 2) :=
sorry

theorem find_extrema 
  (a b : ℝ) 
  (h_f : f = (λ x, -1/3 * x ^ 3 + x ^ 2)) 
  (x ∈ Icc 1 2) : 
  g 2 - g 1 ≥ 0 ∧ g 2 - g (sqrt 2) ≤ 0 := 
sorry

end find_fx_find_extrema_l220_220399


namespace simplify_polynomial_subtraction_l220_220941

variable (x : ℝ)

def P1 : ℝ := 2*x^6 + x^5 + 3*x^4 + x^3 + 5
def P2 : ℝ := x^6 + 2*x^5 + x^4 - x^3 + 7
def P3 : ℝ := x^6 - x^5 + 2*x^4 + 2*x^3 - 2

theorem simplify_polynomial_subtraction : (P1 x - P2 x) = P3 x :=
by
  sorry

end simplify_polynomial_subtraction_l220_220941


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220799

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220799


namespace polynomial_real_roots_l220_220255

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end polynomial_real_roots_l220_220255


namespace sequence_an_sequence_Tn_l220_220751

open Nat

theorem sequence_an (a : ℕ → ℝ) (Sn : ℕ → ℝ) (n : ℕ) (hp : n > 0) :
  (∀ n, a (n + 1) = 2 * real.sqrt (Sn n) + 1) ∧ a 1 = 1 ∧
  (∀ n, Sn n = ∑ i in range n, a i) →
  a n = 2 * n - 1 :=
by sorry

theorem sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) :
  (∀ n, a n = 2 * n - 1) ∧
  (T n = ∑ i in range n, 1 / (a i * a (i + 1))) →
  T n = n / (2 * n + 1) :=
by sorry

end sequence_an_sequence_Tn_l220_220751


namespace probability_three_correct_deliveries_l220_220331

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220331


namespace sum_of_arcs_limit_l220_220565

-- Given definitions
variable (D : ℝ) (n : ℕ) (hn : n > 0)

-- The length of one quarter circle arc when the diameter D is divided into n parts
def quarterCircleArcLength : ℝ := (π * D) / (8 * n)

-- Sum of the lengths of the quarter circle arcs
def totalArcLength (n : ℕ) : ℝ := n * quarterCircleArcLength D n

-- The expected limiting value
def expectedLimitingValue : ℝ := π * D / 8

-- The proposition to prove
theorem sum_of_arcs_limit : (∃ n : ℕ, ∀ ε > 0, ∃ N, ∀ n ≥ N, |totalArcLength D n - expectedLimitingValue D| < ε) :=
by
  sorry

end sum_of_arcs_limit_l220_220565


namespace runners_meet_again_l220_220603

-- Definitions based on the problem conditions
def track_length : ℝ := 500 
def speed_runner1 : ℝ := 4.4
def speed_runner2 : ℝ := 4.8
def speed_runner3 : ℝ := 5.0

-- The time at which runners meet again at the starting point
def time_when_runners_meet : ℝ := 2500

theorem runners_meet_again :
  ∀ t : ℝ, t = time_when_runners_meet → 
  (∀ n1 n2 n3 : ℤ, 
    ∃ k : ℤ, 
    speed_runner1 * t = n1 * track_length ∧ 
    speed_runner2 * t = n2 * track_length ∧ 
    speed_runner3 * t = n3 * track_length) :=
by 
  sorry

end runners_meet_again_l220_220603


namespace line_inclination_angle_l220_220571

theorem line_inclination_angle (θ : ℝ) :
  (∃ (A B C : ℝ), A = sqrt 3 ∧ B = -3 ∧ C = 1 ∧ 
    ∃ (x y : ℝ), A * x + B * y + C = 0) ∧ θ ∈ Ico 0 180 →
  θ = 30 :=
by
  sorry

end line_inclination_angle_l220_220571


namespace median_length_BC_l220_220758

noncomputable def length_of_median (Ax Ay Bx By Cx Cy : ℝ) : ℝ :=
  let Dx := (Bx + Cx)/2
  let Dy := (By + Cy)/2
  real.sqrt ((Ax - Dx)^2 + (Ay - Dy)^2)
  
theorem median_length_BC :
  length_of_median 2 1 (-2) 3 0 1 = real.sqrt 10 :=
by 
  sorry

end median_length_BC_l220_220758


namespace sequence_inequality_infinite_l220_220507

open Nat Real

theorem sequence_inequality_infinite {a : ℕ → ℝ} (h_pos : ∀ n, 0 < a n) :
  ∃ᶠ n in at_top, 1 + a n > a (n - 1) * real.exp (real.log 2 / n) :=
sorry

end sequence_inequality_infinite_l220_220507


namespace remaining_money_l220_220186

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l220_220186


namespace suitable_survey_for_comprehensive_investigation_l220_220117

theorem suitable_survey_for_comprehensive_investigation (A B C D : Prop) 
  (hA : A = "Understanding the amount of time each classmate spends on physical exercise per week is suitable for a comprehensive investigation")  
  (hB : B ≠ "Investigating the crash resistance of a batch of cars is suitable for a comprehensive investigation due to the impracticality of testing every car")
  (hC : C ≠ "Surveying the viewership rate of the Spring Festival Gala is suitable for a comprehensive investigation due to the impracticality of surveying every viewer")
  (hD : D ≠ "Testing the number of times the soles of shoes produced by a shoe factory can withstand bending is suitable for a comprehensive investigation due to the impracticality of testing every sole") :
  A = "Understanding the amount of time each classmate spends on physical exercise per week is suitable for a comprehensive investigation" :=
sorry

end suitable_survey_for_comprehensive_investigation_l220_220117


namespace damaged_cartons_per_customer_l220_220488

theorem damaged_cartons_per_customer (total_cartons : ℕ) (num_customers : ℕ) (total_accepted : ℕ) 
    (h1 : total_cartons = 400) (h2 : num_customers = 4) (h3 : total_accepted = 160) 
    : (total_cartons - total_accepted) / num_customers = 60 :=
by
  sorry

end damaged_cartons_per_customer_l220_220488


namespace total_salaries_proof_l220_220586

def total_salaries (A_salary B_salary : ℝ) :=
  A_salary + B_salary

theorem total_salaries_proof : ∀ A_salary B_salary : ℝ,
  A_salary = 3000 →
  (0.05 * A_salary = 0.15 * B_salary) →
  total_salaries A_salary B_salary = 4000 :=
by
  intros A_salary B_salary h1 h2
  rw [h1] at h2
  sorry

end total_salaries_proof_l220_220586


namespace solve_star_equation_l220_220689

def star (a b : ℝ) : ℝ := (a + b) / a

theorem solve_star_equation : ∀ x : ℝ, star (star 4 3) x = 13 → x = 21 :=
by
  intros x h
  sorry

end solve_star_equation_l220_220689


namespace no_positive_integer_solutions_l220_220692

theorem no_positive_integer_solutions:
    ∀ x y : ℕ, x > 0 → y > 0 → x^2 + 2 * y^2 = 2 * x^3 - x → false :=
by
  sorry

end no_positive_integer_solutions_l220_220692


namespace parallel_lines_slope_l220_220402

theorem parallel_lines_slope (n : ℝ) :
  (∀ x y : ℝ, 2 * x + 2 * y - 5 = 0 → 4 * x + n * y + 1 = 0 → -1 = - (4 / n)) →
  n = 4 :=
by sorry

end parallel_lines_slope_l220_220402


namespace product_mod_seven_l220_220707

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220707


namespace solve_for_x_l220_220543

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l220_220543


namespace minimum_restoration_time_l220_220163

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l220_220163


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220807

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220807


namespace isosceles_triangle_sum_l220_220564

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

def valid_triangle_sides (n : ℕ) : Prop :=
  0 < n^2 + n ∧ 0 < 2 * n + 12 ∧ 0 < 3 * n + 3

theorem isosceles_triangle_sum :
  (∑ n in {n : ℕ | valid_triangle_sides n ∧ 
              is_isosceles (n^2 + n) (2 * n + 12) (3 * n + 3)}, n) = 7 :=
by
  sorry

end isosceles_triangle_sum_l220_220564


namespace product_of_terms_l220_220223

theorem product_of_terms : (∏ n in finset.range (11) (λ n, 1 - 1 / (n + 2) ^ 2)) = 13 / 24 :=
  sorry

end product_of_terms_l220_220223


namespace three_correct_deliveries_probability_l220_220266

/-- Five packages are delivered to five houses randomly, with each house getting exactly one package.
    Prove that the probability that exactly three packages are delivered to the correct houses is 1/12. 
-/ 
theorem three_correct_deliveries_probability :
  let houses := Finset.range 5 in
  let packages := Finset.range 5 in
  let correct_deliveries : Nat := 3 in
  ∃ (prob : ℚ), prob = 1/12 := 
sorry

end three_correct_deliveries_probability_l220_220266


namespace probability_three_correct_deliveries_l220_220344

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220344


namespace iterative_average_difference_l220_220216

def iterative_average (lst : List ℚ) : ℚ :=
  match lst with
  | []      => 0
  | [x]     => x
  | x::y::l => iterative_average ((x + y)/2 :: l)

theorem iterative_average_difference :
  let numbers := [1, 2, 3, 4, 5].map (λ x => (x : ℚ)) in
  (iterative_average [5, 4, 3, 2, 1].map (λ x => (x : ℚ)) -
   iterative_average [1, 2, 3, 4, 5].map (λ x => (x : ℚ))) = 17/8 := 
  sorry

end iterative_average_difference_l220_220216


namespace two_possible_combinations_l220_220198

-- Defining variables and parameters
def number_of_possible_combinations : ℕ :=
  ({p : ℕ × ℕ | 42 * p.1 + 20 * p.2 = 482}).to_finset.card

-- The theorem statement
theorem two_possible_combinations : number_of_possible_combinations = 2 :=
by
  sorry

end two_possible_combinations_l220_220198


namespace matt_total_vibrations_l220_220512

noncomputable def vibrations_lowest : ℕ := 1600
noncomputable def vibrations_highest : ℕ := vibrations_lowest + (6 * vibrations_lowest / 10)
noncomputable def time_seconds : ℕ := 300
noncomputable def total_vibrations : ℕ := vibrations_highest * time_seconds

theorem matt_total_vibrations :
  total_vibrations = 768000 := by
  sorry

end matt_total_vibrations_l220_220512


namespace probability_even_number_from_draws_l220_220670

/-- 
There is an urn with 9 balls numbered 1 to 9. José and Maria each draw one ball simultaneously.
José's ball will form the tens digit while Maria's ball will form the units digit of a two-digit number.
We want to prove the probability that the number formed is even.
-/
theorem probability_even_number_from_draws : 
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let even_numbers := {2, 4, 6, 8}
  let total_pairs := 72
  let favorable_pairs := 32
  let probability := (favorable_pairs : ℚ) / total_pairs
  probability = (4 : ℚ) / 9 :=
by
  sorry

end probability_even_number_from_draws_l220_220670


namespace sum_of_series_l220_220697

theorem sum_of_series :
  ∑ n in Finset.range (128) + 2, (1 / ((2 * n - 3) * (2 * n + 1))) = 129 / 259 := by
  sorry

end sum_of_series_l220_220697


namespace parallel_OM_AC_l220_220651

-- Definitions
variable {α : Type} [metric_space α]

structure Circle (α : Type) [metric_space α] :=
(center : α) (radius : ℝ)

def tangent_point (O : α) (AO : Circle α) (PointA : α) : Prop :=
dist PointA AO.center = AO.radius ∧ dist PointA O = 0

def diameter_points (A B : α) : Prop :=
dist A B > 0

def chord_touches (B C : α) (AO : Circle α) (PointM : α) : Prop :=
dist B C > 0 ∧ dist PointM AO.center = AO.radius ∧ dist PointM C = 0

-- Theorem Statement
theorem parallel_OM_AC 
(O A B C M : α) (AO : Circle α)
(h1 : tangent_point O AO A)
(h2 : diameter_points A B)
(h3 : chord_touches B C AO M)
: parallel OM AC := sorry

end parallel_OM_AC_l220_220651


namespace probability_three_correct_deliveries_l220_220330

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220330


namespace ratio_of_areas_l220_220067

variable (l w r : ℝ)
variable (h1 : l = 2 * w)
variable (h2 : 6 * w = 2 * real.pi * r)

theorem ratio_of_areas : (l * w) / (real.pi * r^2) = 2 * real.pi / 9 :=
by sorry

end ratio_of_areas_l220_220067


namespace directrix_of_parabola_l220_220060

theorem directrix_of_parabola (p : ℝ) : y^2 = -8 * x → x = 2 :=
by 
  assume h
  sorry

end directrix_of_parabola_l220_220060


namespace cuboid_dimensions_l220_220103

-- Define the problem conditions and the goal
theorem cuboid_dimensions (x y v : ℕ) :
  (v * (x * y - 1) = 602) ∧ (x * (v * y - 1) = 605) →
  v = x + 3 →
  x = 11 ∧ y = 4 ∧ v = 14 :=
by
  sorry

end cuboid_dimensions_l220_220103


namespace two_dice_two_coins_prob_l220_220051

/-- Two dice rolls and two coin flips. Calculate probability that the sum is at least 12. -/
theorem two_dice_two_coins_prob :
  ∀ (die1 die2 : ℕ) (coin1 coin2 : ℕ),
  (1 ≤ die1 ∧ die1 ≤ 6) →
  (1 ≤ die2 ∧ die2 ≤ 6) →
  (coin1 = 0 ∨ coin1 = 5 ∨ coin1 = 25) →
  (coin2 = 0 ∨ coin2 = 5 ∨ coin2 = 25) →
  (11 ≤ die1 + die2 + coin1 + coin2) →
  ∃ (favorable outcomes total outcomes : ℕ),
  total outcomes = 144 ∧
  favorable outcomes = 94 ∧
  (favorable outcomes : ℝ) / (total outcomes : ℝ) = 47 / 72 :=
by 
  intros die1 die2 coin1 coin2 hdie1 hdie2 hcoin1 hcoin2
  sorry

end two_dice_two_coins_prob_l220_220051


namespace cone_surface_ratio_and_angle_l220_220075

-- Definitions of the given conditions
def total_surface_area_ratio (k : ℝ) (α : ℝ) : Prop :=
  k = π * (sin α + 1) / cos α

def permissible_values (k : ℝ) : Prop :=
  k > π

-- Proposition to prove the given questions (angle α and permissible values of k)
theorem cone_surface_ratio_and_angle (k : ℝ) (α : ℝ) :
  total_surface_area_ratio k α →
  permissible_values k →
  α = (π / 2) - 2 * arctan (π / k) ∧ k > π := 
by
  intro h1 h2
  -- Proof will go here
  sorry

end cone_surface_ratio_and_angle_l220_220075


namespace find_b_l220_220737

-- Let's define the real numbers and the conditions given.
variables (b y a : ℝ)

-- Conditions from the problem
def condition1 := abs (b - y) = b + y - a
def condition2 := abs (b + y) = b + a

-- The goal is to find the value of b
theorem find_b (h1 : condition1 b y a) (h2 : condition2 b y a) : b = 1 :=
by
  sorry

end find_b_l220_220737


namespace plant_supplier_earnings_l220_220184

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l220_220184


namespace probability_ray_OA_within_angle_xOT_l220_220462

open MeasureTheory Set

-- Define the context: Cartesian coordinate system and angles
def angle (deg : ℝ) := deg / 360

-- Condition: angle xOT = 60 degrees
def angle_xOT : ℝ := 60

-- Question: What is the probability of the ray OA falling within angle xOT?
theorem probability_ray_OA_within_angle_xOT :
  (angle angle_xOT) = 1 / 6 :=
by
  sorry

end probability_ray_OA_within_angle_xOT_l220_220462


namespace point_P_quadrant_l220_220761

theorem point_P_quadrant 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  (0 < Real.cos θ) ∧ (Real.sin θ < 0) :=
by
  sorry

end point_P_quadrant_l220_220761


namespace median_and_mode_l220_220745

open Set

variable (data_set : List ℝ)
variable (mean : ℝ)

noncomputable def median (l : List ℝ) : ℝ := sorry -- Define medial function
noncomputable def mode (l : List ℝ) : ℝ := sorry -- Define mode function

theorem median_and_mode (x : ℝ) (mean_set : (3 + x + 4 + 5 + 8) / 5 = 5) :
  data_set = [3, 4, 5, 5, 8] ∧ median data_set = 5 ∧ mode data_set = 5 :=
by
  have hx : x = 5 := sorry
  have hdata_set : data_set = [3, 4, 5, 5, 8] := sorry
  have hmedian : median data_set = 5 := sorry
  have hmode : mode data_set = 5 := sorry
  exact ⟨hdata_set, hmedian, hmode⟩

end median_and_mode_l220_220745


namespace train_length_l220_220665

theorem train_length (L V : ℝ) 
  (h1 : V = L / 10) 
  (h2 : V = (L + 870) / 39) 
  : L = 300 :=
by
  sorry

end train_length_l220_220665


namespace other_cat_weight_is_7_l220_220681

variable (C : ℕ) -- Declare a variable for the weight of the other cat

-- Conditions
def cat_weight_1 : ℕ := 10
def dog_weight : ℕ := 34
def sum_cat_weights : ℕ := cat_weight_1 + C

-- The dog's weight is twice the sum of the two cats' weights
axiom dog_weight_condition : 2 * sum_cat_weights = dog_weight

theorem other_cat_weight_is_7 : C = 7 :=
by
  have h1 : 2 * (10 + C) = 34 := dog_weight_condition
  calc
    2 * (10 + C) = 34       : by exact h1
    20 + 2 * C = 34         : by ring
    2 * C = 34 - 20         : by linarith
    2 * C = 14              : by norm_num
    C = 14 / 2              : by linarith
    C = 7                   : by norm_num

end other_cat_weight_is_7_l220_220681


namespace simplify_fraction_1_simplify_resistance_simplify_task_comparison_l220_220129
noncomputable theory

-- Definitions
def complex_fraction_1 (x y : ℝ) : ℝ :=
  (1 + x / y) / (1 - x / y)

def resistance (R1 R2 : ℝ) : ℝ :=
  (1 / R1 + 1 / R2).recip

def task_comparison (x y z : ℝ) : ℝ :=
  x / (1 / (1 / y + 1 / z))

-- Theorems
theorem simplify_fraction_1 (x y : ℝ) : 
  complex_fraction_1 x y = (y + x) / (y - x) := sorry

theorem simplify_resistance (R1 R2 : ℝ) : 
  resistance R1 R2 = R1 * R2 / (R1 + R2) := sorry

theorem simplify_task_comparison (x y z : ℝ) :
  task_comparison x y z = (x * y + x * z) / (y * z) := sorry

end simplify_fraction_1_simplify_resistance_simplify_task_comparison_l220_220129


namespace Nara_is_1_69_meters_l220_220537

-- Define the heights of Sangheon, Chiho, and Nara
def Sangheon_height : ℝ := 1.56
def Chiho_height : ℝ := Sangheon_height - 0.14
def Nara_height : ℝ := Chiho_height + 0.27

-- The statement to be proven
theorem Nara_is_1_69_meters : Nara_height = 1.69 :=
by
  -- the proof goes here
  sorry

end Nara_is_1_69_meters_l220_220537


namespace shading_problem_l220_220469

-- Define the total number of small rectangles
def total_rectangles : ℕ := 12

-- Define the fraction that needs to be shaded
def fraction_to_shade : ℚ := (2 / 3) * (3 / 4)

-- Define the calculation of the number of rectangles to be shaded
def num_shaded_rectangles : ℕ := total_rectangles * fraction_to_shade

-- The statement to prove
theorem shading_problem : num_shaded_rectangles = 6 :=
  by sorry

end shading_problem_l220_220469


namespace systematic_sampling_l220_220193

-- Define the conditions
def total_products : ℕ := 100
def selected_products (n : ℕ) : ℕ := 3 + 10 * n
def is_systematic (f : ℕ → ℕ) : Prop :=
  ∃ k b, ∀ n, f n = b + k * n

-- Theorem to prove that the selection method is systematic sampling
theorem systematic_sampling : is_systematic selected_products :=
  sorry

end systematic_sampling_l220_220193


namespace whitney_money_left_over_l220_220119

def total_cost (posters_cost : ℝ) (notebooks_cost : ℝ) (bookmarks_cost : ℝ) (pencils_cost : ℝ) (tax_rate : ℝ) :=
  let pre_tax := (3 * posters_cost) + (4 * notebooks_cost) + (5 * bookmarks_cost) + (2 * pencils_cost)
  let tax := pre_tax * tax_rate
  pre_tax + tax

def money_left_over (initial_money : ℝ) (total_cost : ℝ) :=
  initial_money - total_cost

theorem whitney_money_left_over :
  let initial_money := 40
  let posters_cost := 7.50
  let notebooks_cost := 5.25
  let bookmarks_cost := 3.10
  let pencils_cost := 1.15
  let tax_rate := 0.08
  money_left_over initial_money (total_cost posters_cost notebooks_cost bookmarks_cost pencils_cost tax_rate) = -26.20 :=
by
  sorry

end whitney_money_left_over_l220_220119


namespace candidate_lost_by_votes_l220_220160

theorem candidate_lost_by_votes :
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  candidate_votes <= 6450 ∧ rival_votes <= 6450 ∧ rival_votes - candidate_votes = 2451 :=
by
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  have h1: candidate_votes <= 6450 := sorry
  have h2: rival_votes <= 6450 := sorry
  have h3: rival_votes - candidate_votes = 2451 := sorry
  exact ⟨h1, h2, h3⟩

end candidate_lost_by_votes_l220_220160


namespace total_revenue_full_price_l220_220662

theorem total_revenue_full_price (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (3 * p) / 4 = 2800) : 
  f * p = 680 :=
by
  -- proof omitted
  sorry

end total_revenue_full_price_l220_220662


namespace seven_card_combinations_l220_220856

theorem seven_card_combinations (B : ℕ) : 
  (nat.choose 40 7 = 186BC0B40) → B = 0 :=
by
  sorry

end seven_card_combinations_l220_220856


namespace smallest_alpha_22_5_l220_220495

noncomputable def smallest_alpha (m n p : ℝ^3) (α : ℝ) : Prop :=
  ∥m∥ = 1 ∧ ∥n∥ = 1 ∧ ∥p∥ = 1 ∧
  (angle m n = α) ∧
  (angle p (cross m n) = α) ∧
  (dot_product n (cross p m) = 1 / (2 * real.sqrt 2))

theorem smallest_alpha_22_5 (m n p : ℝ^3) (α : ℝ) :
  smallest_alpha m n p α → α = 22.5 := 
sorry

end smallest_alpha_22_5_l220_220495


namespace probability_three_correct_packages_l220_220309

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220309


namespace determine_sanity_l220_220674

-- Defining the conditions for sanity based on responses to a specific question

-- Define possible responses
inductive Response
| ball : Response
| yes : Response

-- Define sanity based on logical interpretation of an illogical question
def is_sane (response : Response) : Prop :=
  response = Response.ball

-- The theorem stating asking the specific question determines sanity
theorem determine_sanity (response : Response) : is_sane response ↔ response = Response.ball :=
by
  sorry

end determine_sanity_l220_220674


namespace train_cross_signal_pole_time_l220_220154

theorem train_cross_signal_pole_time
  (train_length : ℕ) (platform_length : ℕ) (total_time : ℕ)
  (total_distance : train_length + platform_length = 325)
  (speed : train_length + platform_length / total_time ≈ 8.333) :
  (train_length / speed) ≈ 36 :=
sorry

end train_cross_signal_pole_time_l220_220154


namespace range_of_a1_l220_220370

theorem range_of_a1 {a : ℕ → ℝ} (h_seq : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h_a1_positive : a 1 > 0) :
  (0 < a 1) ∧ (a 1 < 1) ↔ ∀ m n : ℕ, m < n → a m < a n := by
  sorry

end range_of_a1_l220_220370


namespace no_such_varphi_exists_l220_220508
noncomputable theory

open Classical

-- Define the base 2 representation of x.
def base2_representation (x : ℝ) (H: 0 ≤ x ∧ x < 1) : ℕ → ℕ := λ i, nat.of_num((x * 2 ^ (i + 1)).floor % 2)

-- Define f_n function.
def f_n (x : ℝ) (H: 0 ≤ x ∧ x < 1) (n : ℕ) : ℤ :=
  ∑ j in Finset.range n, (-1) ^ (∑ i in Finset.range (j + 1), base2_representation x H i)

-- Main theorem
theorem no_such_varphi_exists :
  ¬ ∃ (varphi : ℝ → ℝ), (∀ x, 0 ≤ varphi x) ∧ (tendsto varphi at_top at_top)
  ∧ (∃ B : ℝ, ∀ n, ∫ (x : ℝ) in 0..1, varphi (|f_n x ⟨le_refl 0, zero_lt_one⟩ n|) < B) :=
sorry

end no_such_varphi_exists_l220_220508


namespace product_mod_seven_l220_220709

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220709


namespace length_major_axis_of_ellipse_is_2sqrt7_l220_220754

-- Definitions based on the given problem
def major_axis_length (a : ℝ) (h : a > 2) : ℝ := 2 * a

def ellipse (a : ℝ) : Prop := 
  ∃ x y : ℝ, 
  (x^2 / a^2) + (y^2 / (a^2 - 4)) = 1

def line_intersects_ellipse_at_one_point (a : ℝ) : Prop := 
  let delta := 192 * (a^2 - 4)^2 - 16 * (a^2 - 3) * (16 - a^2) * (a^2 - 4) in
  delta = 0

-- The main theorem we need to prove
theorem length_major_axis_of_ellipse_is_2sqrt7 (a : ℝ) (h : a > 2) :
  ellipse a ∧ line_intersects_ellipse_at_one_point a → major_axis_length a h = 2 * Real.sqrt 7 :=
by
  sorry

end length_major_axis_of_ellipse_is_2sqrt7_l220_220754


namespace moon_radius_scientific_notation_l220_220968

noncomputable def moon_radius : ℝ := 1738000

theorem moon_radius_scientific_notation :
  moon_radius = 1.738 * 10^6 :=
by
  sorry

end moon_radius_scientific_notation_l220_220968


namespace find_m_over_n_l220_220781

variable (OA OB OC : ℝ)
variable (m n : ℝ)
variable (a b : ℝ)
variable (angle_OA_OB : ℝ)

noncomputable def vector_magnitude := real.sqrt

def dot_product (u v : ℝ) : ℝ :=
u * v * real.cos angle_OA_OB

-- Conditions
axiom h1 : vector_magnitude OA = 3
axiom h2 : vector_magnitude OB = 2
axiom h3 : OC = m * OA + n * OB
axiom h4 : angle_OA_OB = real.pi / 3  -- 60 degrees in radians
axiom h5 : dot_product OC (OB - OA) = 0  -- ⟂ condition

-- Question (proof statement)
theorem find_m_over_n : (m / n) = (1 / 6) :=
sorry

end find_m_over_n_l220_220781


namespace unique_solution_4x2y_4xy2_2_l220_220253

theorem unique_solution_4x2y_4xy2_2 (x y : ℝ) :
  4^(x^2 + y) + 4^(x + y^2) = 2 → (x, y) = (-1/2, -1/2) := 
sorry

end unique_solution_4x2y_4xy2_2_l220_220253


namespace perimeter_rectangle_l220_220134

variables (AE BE CF m n : ℕ)
variables (ABCD : Type)  -- Representing the rectangle

def E := (8 : ℕ)
def B_E := (17 : ℕ)
def C_F := (3 : ℕ)
def P := (56 : ℕ)

/-- Given a rectangle \(ABCD\) with dimensions AE = 8, BE = 17, CF = 3,
  the perimeter P is such that m and n are relatively prime positive integers
  and the sum m + n equals 57. -/
theorem perimeter_rectangle (hAE : AE = 8) (hBE : BE = 17) (hCF : CF = 3) (hP : P = 56) :
  ∃ m n : ℕ, Nat.coprime m n ∧ P = m / n ∧ m + n = 57 := sorry

end perimeter_rectangle_l220_220134


namespace problem_conclusion_l220_220376

variable (p q : Prop)
variable (h1 : ¬p)
variable (h2 : p ∨ q)

theorem problem_conclusion : ¬p ∧ q :=
by
  exact ⟨h1, or.elim h2 (λ hp, false.elim (h1 hp)) id⟩
  -- sorry  -- Add this if preferring explicit stepwise skipping proof

end problem_conclusion_l220_220376


namespace no_equal_refereed_matches_l220_220872

theorem no_equal_refereed_matches {k : ℕ} (h1 : ∀ {n : ℕ}, n > k → n = 2 * k) 
    (h2 : ∀ {n : ℕ}, n > k → ∃ m, m = k * (2 * k - 1))
    (h3 : ∀ {n : ℕ}, n > k → ∃ r, r = (2 * k - 1) / 2): 
    False := 
by
  sorry

end no_equal_refereed_matches_l220_220872


namespace probability_three_correct_deliveries_is_one_sixth_l220_220320

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220320


namespace sequence_product_modulo_7_l220_220718

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220718


namespace find_x_plus_y_of_parallel_vectors_l220_220780

theorem find_x_plus_y_of_parallel_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (x, 2, -2)) 
  (hb : b = (2, y, 4)) 
  (h_parallel : ∃ k : ℝ, a = k • b) 
  : x + y = -5 := 
by 
  sorry

end find_x_plus_y_of_parallel_vectors_l220_220780


namespace roof_ratio_l220_220076

theorem roof_ratio (L W : ℕ) (h1 : L * W = 768) (h2 : L - W = 32) : L / W = 3 := 
sorry

end roof_ratio_l220_220076


namespace odd_terms_in_expansion_l220_220826

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l220_220826


namespace solve_inequality_l220_220046

theorem solve_inequality (a : ℝ) : 
  let S := {x : ℝ | 6 * x^2 + a * x - a^2 < 0} in
  if h : a > 0 then 
    S = {x : ℝ | -a / 2 < x ∧ x < a / 3} 
  else if h : a = 0 then 
    S = ∅ 
  else 
    S = {x : ℝ | a / 3 < x ∧ x < -a / 2} :=
sorry

end solve_inequality_l220_220046


namespace five_digit_number_divisibility_l220_220930

theorem five_digit_number_divisibility (a : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) : 11 ∣ 100001 * a :=
by
  sorry

end five_digit_number_divisibility_l220_220930


namespace odd_terms_in_expansion_l220_220837

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l220_220837


namespace probability_of_three_correct_packages_l220_220300

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220300


namespace find_constant_l220_220440

theorem find_constant (c : ℝ) (t : ℝ) : (x = 1 - 3 * t) → (y = 2 * t - c) → (t = 0.8) → (x = y) → (c = 3) :=
by
  assume h1 : x = 1 - 3 * t,
  assume h2 : y = 2 * t - c,
  assume ht : t = 0.8,
  assume hxy : x = y,
  sorry

end find_constant_l220_220440


namespace probability_of_exactly_three_correct_packages_l220_220294

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220294


namespace minimal_size_of_S_l220_220133

theorem minimal_size_of_S (n : ℕ) (S : Set ℕ) (h₁: ∀ β : ℝ, β > 0 → S ⊆ {m | ∃ k : ℕ, ⌊β * (k : ℝ)⌋ = m} → 
      {1, 2, ..., n} ⊆ {m | ∃ k : ℕ, ⌊β * (k : ℝ)⌋ = m}) : 
  ∃ (S : Set ℕ), S ⊆ {1, 2, ..., n} ∧ S.card = n.div(2) + 1 :=
begin
  sorry
end

end minimal_size_of_S_l220_220133


namespace wrappers_in_collection_l220_220688

variable (W : ℕ)

/-- 
Danny collects bottle caps and wrappers. He found 22 bottle caps and 8 wrappers at the park. 
Now he has 28 bottle caps and a certain number of wrappers in his collection. Danny had 
6 bottle caps at first. How many wrappers does he have in his collection now?
-/
theorem wrappers_in_collection (W_initial : ℕ) 
  (found_wrappers : 8) : 
  W_initial + found_wrappers = W_initial + 8 := 
by 
  sorry -- proof follows from arithmetic addition of constants

end wrappers_in_collection_l220_220688


namespace angle_RQP_l220_220865

variables (K L P Q R Y : Point)
variables (y : ℝ)

-- Define the parallel lines condition
def parallel (a b c d : Point) : Prop := ∃ m1 m2, m1 ≠ m2 ∧ m1 * (b.x - a.x) + c.x * (b.y - a.y) = m2 * (d.x - c.x) + a.x * (d.y - c.y)

-- Given conditions
axiom parallel_KL_PQ : parallel K L P Q
axiom angle_KQY : ∠KQY = y
axiom angle_YPQ : ∠YPQ = 5 * y
axiom supplementary_angles : ∠KQY + ∠YPQ = 180

-- Question to prove
theorem angle_RQP (h1 : parallel_KL_PQ) (h2 : angle_KQY) (h3 : angle_YPQ) (h4 : supplementary_angles) : ∠RQP = 90 :=
sorry

end angle_RQP_l220_220865


namespace remaining_money_l220_220187

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l220_220187


namespace complex_partition_l220_220536

noncomputable def is_non_zero (z : ℂ) : Prop := z ≠ 0

def angle (z w : ℂ) : ℝ := complex.arg (z / w)

def acute_angle (z w : ℂ) : Prop := angle z w ≤ real.pi / 2

def obtuse_angle (z w : ℂ) : Prop := angle z w > real.pi / 2

theorem complex_partition (S : set ℂ) (hS : S.card = 1993) (hS_nonzero : ∀ z ∈ S, is_non_zero z) :
  ∃ (partition : list (set ℂ)),
    (∃ S_i : set ℂ, S_i ∈ partition) ∧
    (∀ S_i S_j : set ℂ, S_i ≠ S_j ∧ S_i ∈ partition ∧ S_j ∈ partition → obtuse_angle (∑ z in S_i, z) (∑ z in S_j, z)) ∧
    (∀ S_i : set ℂ, S_i ∈ partition → ∀ z ∈ S_i, acute_angle z (∑ z' in S_i, z')) :=
sorry

end complex_partition_l220_220536


namespace num_correct_statements_is_two_l220_220599

theorem num_correct_statements_is_two :
  let statement_A := (∀ (A : ℝ), sin A = 1 / 2 → (deg A = 30 ∨ deg A = 150)) in 
  let statement_B := (∀ (A : ℝ), cos (2 * π - A) = cos A) in
  let statement_C := ∀ (A : ℝ), 0 < sin A ∧ 0 < cos A → (A % π ≠ 0) in
  let statement_D := (sin (130 : ℝ) ^ 2 + sin (140 : ℝ) ^ 2 = 1) in
  (¬ statement_A) ∧ statement_B ∧ (¬ statement_C) ∧ statement_D →  true :=
by
  sorry

end num_correct_statements_is_two_l220_220599


namespace octahedron_side_length_is_three_sqrt_two_over_four_l220_220209

theorem octahedron_side_length_is_three_sqrt_two_over_four 
  (Q1 Q2 Q3 Q4 Q1' Q2' Q3' Q4' : ℝ × ℝ × ℝ)
  (Q1Q2_dist Q1Q3_dist Q1Q4_dist : ℝ)
  (Q1_adj_Q2 : Q2 = (Q1.1 + Q1Q2_dist, Q1.2, Q1.3))
  (Q1_adj_Q3 : Q3 = (Q1.1, Q1.2 + Q1Q3_dist, Q1.3))
  (Q1_adj_Q4 : Q4 = (Q1.1, Q1.2, Q1.3 + Q1Q4_dist))
  (Q1'Q2'_dist Q1'Q3'_dist Q1'Q4'_dist : ℝ)
  (Q1'_adj_Q2': Q2' = (Q1'.1 - Q1'Q2'_dist, Q1'.2, Q1'.3))
  (Q1'_adj_Q3': Q3' = (Q1'.1, Q1'.2 - Q1'Q3'_dist, Q1'.3))
  (Q1'_adj_Q4': Q4' = (Q1'.1, Q1'.2, Q1'.3 - Q1'Q4'_dist))
  (Q_cube_dims: Q1' = (Q1.1 + 1, Q1.2 + 1, Q1.3 + 1))
  (oct_dist : Q1Q2_dist = Q1Q3_dist ∧ Q1Q3_dist = Q1Q4_dist ∧ Q1'Q2'_dist = Q1'Q3'_dist ∧ Q1'Q3'_dist = Q1'Q4'_dist)
  (y : ℝ) :
  Q1Q2_dist = y ∧ Q1Q3_dist = y ∧ Q1Q4_dist = y → 
  (∃ s : ℝ, s = (3 * real.sqrt 2) / 4) :=
by 
  sorry

end octahedron_side_length_is_three_sqrt_two_over_four_l220_220209


namespace ratio_of_increase_to_original_l220_220192

noncomputable def ratio_increase_avg_marks (T : ℝ) : ℝ :=
  let original_avg := T / 40
  let new_total := T + 20
  let new_avg := new_total / 40
  let increase_avg := new_avg - original_avg
  increase_avg / original_avg

theorem ratio_of_increase_to_original (T : ℝ) (hT : T > 0) :
  ratio_increase_avg_marks T = 20 / T :=
by
  unfold ratio_increase_avg_marks
  sorry

end ratio_of_increase_to_original_l220_220192


namespace find_x_proportionally_l220_220013

theorem find_x_proportionally (k m x z : ℝ) (h1 : ∀ y, x = k * y^2) (h2 : ∀ z, y = m / (Real.sqrt z)) (h3 : x = 7 ∧ z = 16) :
  ∃ x, x = 7 / 9 := by
  sorry

end find_x_proportionally_l220_220013


namespace probability_three_correct_deliveries_l220_220329

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220329


namespace smallest_unachievable_total_score_l220_220964

theorem smallest_unachievable_total_score (scores : List ℕ) (total_darts : ℕ) : 
  (scores = [1, 3, 7, 8, 12]) ∧ (total_darts = 3) → 
  ∃ unachievable_score : ℕ, unachievable_score = 22 :=
by
  intro h
  let available_scores := [1, 3, 7, 8, 12]
  have h1 : scores = available_scores, from h.1
  have h2 : total_darts = 3, from h.2
  use 22
  sorry

end smallest_unachievable_total_score_l220_220964


namespace find_BD_length_l220_220474

noncomputable def BD_length (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  let AC := 6
  let BC := 6
  let AD := 9
  let CD := 3
  BD = 9

theorem find_BD_length (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]: 
  BD_length A B C D :=
  sorry

end find_BD_length_l220_220474


namespace count_neg_values_l220_220358

noncomputable def number_of_integers_make_expression_negative : ℕ :=
  -- The number of integer values x such that x^4 - 51x^2 + 100 < 0
  10

-- Statement: Prove that the number of integers x such that the expression x^4 - 51x^2 + 100 is negative is exactly 10.
theorem count_neg_values :
  ∃ (n : ℕ), n = number_of_integers_make_expression_negative ∧ 
  (n = 10 ∧ ∀ (x : ℤ), x^4 - 51 * x^2 + 100 < 0) :=
begin
  use 10,
  split,
  { refl, },
  { split,
    { refl, },
    { sorry, },
  },
end

end count_neg_values_l220_220358


namespace probability_exactly_three_correct_l220_220346

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220346


namespace odd_terms_in_expansion_l220_220827

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l220_220827


namespace terez_cows_l220_220946

theorem terez_cows (N_f N : ℕ) (h1 : N_f = 0.5 * N) (h2 : 0.5 * N_f = 11) : N = 44 :=
by
  sorry

end terez_cows_l220_220946


namespace triangle_exists_l220_220988

theorem triangle_exists (n : ℕ) (h : 2 ≤ n) (a : Fin n → ℝ) (h_sum : (∑ i, a i) = 3) :
  ∃ x y z, x ∈ ({1, 2} ∪ Finset.univ.image a) ∧ y ∈ ({1, 2} ∪ Finset.univ.image a) ∧ z ∈ ({1, 2} ∪ Finset.univ.image a) ∧ 
  x + y > z ∧ y + z > x ∧ z + x > y := 
sorry

end triangle_exists_l220_220988


namespace g_f2_minus_f_g2_eq_zero_l220_220502

def f (x : ℝ) : ℝ := x^2 + 3 * x + 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem g_f2_minus_f_g2_eq_zero : g (f 2) - f (g 2) = 0 := by
  sorry

end g_f2_minus_f_g2_eq_zero_l220_220502


namespace tan_alpha_l220_220363

theorem tan_alpha (α : ℝ) (h1 : Real.sin(α - π) = 2 / 3) (h2 : α ∈ Ioo (-π / 2) 0) : 
  Real.tan α = -2 * Real.sqrt 5 / 5 :=
by
  sorry  

end tan_alpha_l220_220363


namespace Q_at_1_eq_neg_1_l220_220890

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

noncomputable def mean_coefficient : ℝ := (3 - 5 + 2 - 1) / 4

noncomputable def Q (x : ℝ) : ℝ := mean_coefficient * x^3 + mean_coefficient * x^2 + mean_coefficient * x + mean_coefficient

theorem Q_at_1_eq_neg_1 : Q 1 = -1 := by
  sorry

end Q_at_1_eq_neg_1_l220_220890


namespace odd_terms_in_expansion_l220_220834

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l220_220834


namespace dodecagon_ratio_l220_220525

-- Let A_i be the vertices of a regular 12-gon (dodecagon).
def regular_dodecagon (A : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i : Fin 12, dist (A i) (A ((i + 1) % 12)) = dist (A 0) (A 1)

-- Let X be a point in the plane.
variable (X : ℝ × ℝ)

-- Define the marked diagonals.
def diagonal (A : Fin 12 → ℝ × ℝ) (i : Fin 12) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (A i, A ((i + 5) % 12))

-- Feet of the perpendiculars from X to the marked diagonals.
def foot_of_perpendicular (X : ℝ × ℝ) (A : Fin 12 → ℝ × ℝ) (i : Fin 12) : ℝ × ℝ := sorry

-- Distances from X to vertices A_i.
def XA_dist (X : ℝ × ℝ) (A : Fin 12 → ℝ × ℝ) : Fin 12 → ℝ :=
  λ i, dist X (A i)

-- Distances between feet of perpendiculars.
def B_dist (X : ℝ × ℝ) (A : Fin 12 → ℝ × ℝ) : Fin 12 → ℝ :=
  λ i, dist (foot_of_perpendicular X A i) (foot_of_perpendicular X A ((i + 5) % 12))

theorem dodecagon_ratio (A : Fin 12 → ℝ × ℝ) (hA : regular_dodecagon A) :
  (∑ i, XA_dist X A i) / (∑ i, B_dist X A i) = 1 / 2 :=
sorry

end dodecagon_ratio_l220_220525


namespace solve_for_n_l220_220823

theorem solve_for_n (n : ℚ) (h : 4^8 = 8^n) : n = 16 / 3 := by
  sorry

end solve_for_n_l220_220823


namespace angle_BDC_is_75_l220_220556

-- Definitions for the given conditions
variables (A B C D : Type) [EuclideanGeometry]

-- Angles and segment lengths based on the problem conditions
variables (angle_BAC : Angle A B C = 30) 
          (side_AB : Segment A B = Segment A C) 
          (side_AC : Segment A C = Segment A D) 
          (congruence_AB : Congruent (Triangle A B C) (Triangle A C D))

-- To prove
theorem angle_BDC_is_75 (A B C D : Type)
    [EuclideanGeometry]
    (angle_BAC : Angle A B C = 30) 
    (side_AB : Segment A B = Segment A C) 
    (side_AC : Segment A C = Segment A D) 
    (congruence_AB : Congruent (Triangle A B C) (Triangle A C D)) :
    Angle B D C = 75 := 
sorry

end angle_BDC_is_75_l220_220556


namespace angle_BMO_is_90_degrees_l220_220220

-- Assume the geometric setup
variables (A B C K N M O : Point)
variables (⊙O : Circle)
variables (⊙circumABC : Circle)
variables (⊙circumBKN : Circle)

-- Conditions
axiom circle_O_passes_through_A_and_C : A ∈ ⊙O ∧ C ∈ ⊙O
axiom circle_O_intersects_AB_at_K_and_BC_at_N : K ∈ (AB ∩ ⊙O) ∧ N ∈ (BC ∩ ⊙O) ∧ K ≠ N
axiom circumcircles_intersect_at_B_and_M : B ∈ (⊙circumABC ∩ ⊙circumBKN) ∧ M ∈ (⊙circumABC ∩ ⊙circumBKN)

-- The theorem to prove
theorem angle_BMO_is_90_degrees : ∃ (θ : Real), θ = 90 ∧ ∠BMO = θ :=
by
  sorry

end angle_BMO_is_90_degrees_l220_220220


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l220_220816

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l220_220816


namespace division_remainder_l220_220447

theorem division_remainder (q d D R : ℕ) (h_q : q = 40) (h_d : d = 72) (h_D : D = 2944) (h_div : D = d * q + R) : R = 64 :=
by sorry

end division_remainder_l220_220447


namespace norris_money_left_l220_220920

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l220_220920


namespace no_five_coin_combination_for_48_l220_220258

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def coin_values : list ℕ := [penny, nickel, dime, quarter]

noncomputable def can_make_value_with_five_coins (target : ℕ) : bool :=
  list.any (list.partitions 5 coin_values) (λ combo, combo.sum = target)

theorem no_five_coin_combination_for_48 :
  ¬ can_make_value_with_five_coins 48 :=
by
  sorry

end no_five_coin_combination_for_48_l220_220258


namespace difference_of_interests_l220_220515

def investment_in_funds (X Y : ℝ) (total_investment : ℝ) : ℝ := X + Y
def interest_earned (investment_rate : ℝ) (amount : ℝ) : ℝ := investment_rate * amount

variable (X : ℝ) (Y : ℝ)
variable (total_investment : ℝ) (rate_X : ℝ) (rate_Y : ℝ)
variable (investment_X : ℝ) 

axiom h1 : total_investment = 100000
axiom h2 : rate_X = 0.23
axiom h3 : rate_Y = 0.17
axiom h4 : investment_X = 42000
axiom h5 : investment_in_funds X Y total_investment = total_investment - investment_X

-- We need to show the difference in interest is 200
theorem difference_of_interests : 
  let interest_X := interest_earned rate_X investment_X
  let investment_Y := total_investment - investment_X
  let interest_Y := interest_earned rate_Y investment_Y
  interest_Y - interest_X = 200 :=
by
  sorry

end difference_of_interests_l220_220515


namespace minimum_restoration_time_l220_220164

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l220_220164


namespace factory_output_doubling_l220_220843

theorem factory_output_doubling (x : ℝ) (h : (1 + 0.10)^x = 2) : (1 + 0.10)^x = 2 :=
by
  assumption

end factory_output_doubling_l220_220843


namespace sequence_product_modulo_7_l220_220720

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220720


namespace arrows_from_530_to_533_l220_220752

-- Define what it means for the pattern to be cyclic with period 5
def cycle_period (n m : Nat) : Prop := n % m = 0

-- Define the equivalent points on the circular track
def equiv_point (n : Nat) (m : Nat) : Nat := n % m

-- Given conditions
def arrow_pattern : Prop :=
  ∀ n : Nat, cycle_period n 5 ∧
  (equiv_point 530 5 = 0) ∧ (equiv_point 533 5 = 3)

-- The theorem to be proved
theorem arrows_from_530_to_533 :
  (∃ seq : List (Nat × Nat),
    seq = [(0, 1), (1, 2), (2, 3)]) :=
sorry

end arrows_from_530_to_533_l220_220752


namespace area_of_square_l220_220072

theorem area_of_square (A B : ℝ × ℝ) (hA : A = (1, 6)) (hB : B = (5, 2)) (h_adj : ∀ C D, (A = C ∧ B = D) → (C.1 - D.1)^2 + (C.2 - D.2)^2 = 16 * sqrt 2) : 
  let d := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let side_length := d
  in side_length ^ 2 = 32 :=
by sorry

end area_of_square_l220_220072


namespace quadrilateral_is_rhombus_l220_220088

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem quadrilateral_is_rhombus
  (h₁ : Coincide (Triangle.mk A B C) (Triangle.mk A D C))
  (h₂ : Coincide (Triangle.mk A B D) (Triangle.mk C B D)) :
  (distance A B = distance A D) ∧ (distance B C = distance C D) → 
  (distance A B = distance B C) ∧ (distance A D = distance C D) → 
  is_rhombus A B C D :=
begin
  sorry
end

end quadrilateral_is_rhombus_l220_220088


namespace intersection_of_M_and_N_l220_220379

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 3, 5} → N = {2, 3, 4} → M ∩ N = {2, 3} := by
  intro hM hN
  rw [hM, hN]
  simp
  sorry

end intersection_of_M_and_N_l220_220379


namespace solve_trigonometric_equation_l220_220257

theorem solve_trigonometric_equation (x : ℝ) (k : ℤ) :
  (sin x ≠ 0) ∧ (cos x ≠ 0) ∧ (1 / (sin (x / 2))^3 * (cos (x / 2))^3 - 6 / cos x = (tan (x / 2))^3 + (cot (x / 2))^3)
  → ∃ k : ℤ, x = π / 4 + k * π :=
by
  intro h
  sorry

end solve_trigonometric_equation_l220_220257


namespace fraction_of_married_men_is_two_fifths_l220_220859

noncomputable def fraction_of_married_men (W : ℕ) (p : ℚ) (h : p = 1 / 3) : ℚ :=
  let W_s := p * W
  let W_m := W - W_s
  let M_m := W_m
  let T := W + M_m
  M_m / T

theorem fraction_of_married_men_is_two_fifths (W : ℕ) (p : ℚ) (h : p = 1 / 3) (hW : W = 6) : fraction_of_married_men W p h = 2 / 5 :=
by
  sorry

end fraction_of_married_men_is_two_fifths_l220_220859


namespace sine_between_line_and_plane_example_l220_220933

open Real

def normal_vector_plane (a b c d : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def direction_vector_line_intersection (n1 n2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  -- Calculate the cross product (or any other method to find the perpendicular direction vector)
  let (a1, b1, c1) := n1 in
  let (a2, b2, c2) := n2 in
  (b1 * c2 - c1 * b2, c1 * a2 - a1 * c2, a1 * b2 - b1 * a2)

def magnitude_vector (u : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := u in sqrt (x^2 + y^2 + z^2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := u in
  let (x2, y2, z2) := v in
  x1 * x2 + y1 * y2 + z1 * z2

def sine_of_angle_between_line_and_plane
  (plane_normal : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  abs (dot_product plane_normal line_direction) /
    (magnitude_vector plane_normal * magnitude_vector line_direction)

theorem sine_between_line_and_plane_example :
  let plane_normal := normal_vector_plane 3 (-5) 1 (-7) in
  let normal_plane1 := normal_vector_plane 1 (-3) 0 7 in
  let normal_plane2 := normal_vector_plane 0 4 2 1 in
  let line_direction := direction_vector_line_intersection normal_plane1 normal_plane2 in
  sine_of_angle_between_line_and_plane plane_normal line_direction = sqrt 10 / 35 := 
by
  -- proof omitted
  sorry

end sine_between_line_and_plane_example_l220_220933


namespace four_digit_numbers_with_3_or_7_l220_220409

theorem four_digit_numbers_with_3_or_7 : 
  let total_four_digit_numbers := 9000
  let numbers_without_3_or_7 := 3584
  total_four_digit_numbers - numbers_without_3_or_7 = 5416 :=
by
  trivial

end four_digit_numbers_with_3_or_7_l220_220409


namespace integer_solutions_pairs_l220_220784

-- Definition of the problem
def satisfies_equation (x y : ℤ) : Prop :=
  sqrt (x : ℝ) + sqrt (y : ℝ) = sqrt 1968

-- List of valid integer pairs satisfying the equation
def valid_pairs : List (ℤ × ℤ) :=
  [(0, 1968), (123, 1107), (492, 492), (1107, 123), (1968, 0)]

-- Statement of the theorem to be proven
theorem integer_solutions_pairs :
  ∀ (x y : ℤ), satisfies_equation x y ↔ (x, y) ∈ valid_pairs := 
sorry

end integer_solutions_pairs_l220_220784


namespace range_of_x_l220_220902

def f (x : ℝ) : ℝ := (1 / 2) ^ (1 + x ^ 2) + 1 / (1 + |x|)

theorem range_of_x {x : ℝ} : 
  f (2 * x - 1) + f (1 - 2 * x) < 2 * f x ↔ x < 1 / 3 ∨ x > 1 := 
sorry

end range_of_x_l220_220902


namespace pieces_count_leq_n_squared_l220_220925

theorem pieces_count_leq_n_squared (n : ℕ) (B W : set (fin (2*n) × fin (2*n))) :
  (∀ b ∈ B, ∀ w ∈ W, b.2 ≠ w.2) →
  (∀ w ∈ W, ∀ b ∈ B, w.1 ≠ b.1) →
  (B.card ≤ n^2 ∨ W.card ≤ n^2) :=
sorry

end pieces_count_leq_n_squared_l220_220925


namespace correct_meteor_passing_time_l220_220981

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_l220_220981


namespace point_on_graph_l220_220386

theorem point_on_graph (f : ℝ → ℝ) (h : f 3 = 5) :
  2 * 8.5 = 4 * f (3 * 1) - 3 ∧ (1 + 8.5 = 9.5) :=
by
  -- use given information and arithmetic to show the point is on the graph
  have h1 : f (3 * 1) = f 3 := by sorry -- simple arithmetic
  rw [h1, h]
  split
  . -- show it satisfies the equation 2y = 4f(3x) - 3
    calc
      2 * 8.5 = 17 := by norm_num
      4 * 5 - 3 = 17 := by norm_num
  . -- show the sum of coordinates is 9.5
    calc
      1 + 8.5 = 9.5 := by norm_num

end point_on_graph_l220_220386


namespace angle_bisectors_concurrence_l220_220569

noncomputable def acute_triangle (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : Prop :=
  ∡ABC < 90 ∧ ∡BAC < 90 ∧ ∡ACB < 90

theorem angle_bisectors_concurrence 
  {A B C : Type} [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] 
  (ha : acute_triangle A B C)
  (hA1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center A B C)
  (hB1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center B C A)
  (hC1 : ∃ O : Type, is_circumcircle O A B C ∧ O ∈ circle_center C A B) :
  ∃ line : Type, is_bisector line A1 B1 C1 :=
sorry

end angle_bisectors_concurrence_l220_220569


namespace no_counterexample_in_set_l220_220498

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def s_holds (n : ℕ) : Prop := 
  (sum_of_digits n % 9 = 0 ∧ is_perfect_square n) → n % 9 = 0

theorem no_counterexample_in_set : 
  ∀ n ∈ {36, 81, 49, 64}, s_holds n :=
by
  intros n hn
  simp [s_holds, sum_of_digits, is_perfect_square] at hn
  cases hn with h1 h2 h3 h4;
  { sorry }

end no_counterexample_in_set_l220_220498


namespace mechanic_work_hours_l220_220907

theorem mechanic_work_hours :
  let total_spent := 220
  let part_cost := 20
  let labor_rate := 0.5
  let discount_rate := 0.45 -- 10% discount on 0.5
  let break_minutes := 30
  let part_count := 2
  let first_two_hours := 2 * 60
  let cost_of_parts := part_count * part_cost
  let labor_cost := total_spent - cost_of_parts
  let first_two_hour_cost := first_two_hours * labor_rate
  let remaining_labor_cost := labor_cost - first_two_hour_cost
  let discounted_minutes := remaining_labor_cost / discount_rate
  let total_minutes_worked := discounted_minutes + break_minutes
  let total_hours_worked := total_minutes_worked / 60
  total_hours_worked - (break_minutes / 60) = 4.44 :=
begin
  sorry
end

end mechanic_work_hours_l220_220907


namespace yasmin_children_l220_220484

-- Definitions based on the conditions
def has_twice_children (children_john children_yasmin : ℕ) := children_john = 2 * children_yasmin
def total_grandkids (children_john children_yasmin : ℕ) (total : ℕ) := total = children_john + children_yasmin

-- Problem statement in Lean 4
theorem yasmin_children (children_john children_yasmin total_grandkids : ℕ) 
  (h1 : has_twice_children children_john children_yasmin) 
  (h2 : total_grandkids children_john children_yasmin 6) : 
  children_yasmin = 2 :=
by
  sorry

end yasmin_children_l220_220484


namespace norris_money_left_l220_220923

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l220_220923


namespace complex_number_in_second_quadrant_l220_220465

def given_complex_number : ℂ := ⟨0, 1⟩ / ⟨3, -3⟩

theorem complex_number_in_second_quadrant :
  given_complex_number.re < 0
  ∧ given_complex_number.im > 0 := by
  sorry

end complex_number_in_second_quadrant_l220_220465


namespace probability_three_correct_out_of_five_l220_220283

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220283


namespace no_flippy_div_by_33_l220_220408

def is_flippy (n : ℕ) : Prop :=
  (n >= 10000) ∧ (n < 100000) ∧ 
  (let d1 := n / 10000, d2 := (n / 1000) % 10, d3 := (n / 100) % 10, 
   d4 := (n / 10) % 10, d5 := n % 10 in
   d1 ≠ d2 ∧ d1 = d2 ∧ d1 ≠ d3 ∧ d1 = d4 ∧ d1 ≠ d5)

def divisible_by_33 (n : ℕ) : Prop :=
  n % 33 = 0

theorem no_flippy_div_by_33 : ∀ n : ℕ, is_flippy n ∧ divisible_by_33 n → false :=
by 
  intro n
  intro h
  sorry

end no_flippy_div_by_33_l220_220408


namespace polynomial_mod_p_l220_220014

noncomputable def P (a : ℕ → ℤ) (d : ℕ) := λ x : ℤ, ∑ i in finset.range (d + 1), a i * x ^ i

theorem polynomial_mod_p (a : ℕ → ℤ) (d p : ℕ) (hprime : nat.prime p) :
  ∀ x, (P a d x)^p % p = P a d (x^p) % p :=
begin
  sorry
end

end polynomial_mod_p_l220_220014


namespace alvin_marble_count_correct_l220_220668

variable (initial_marble_count lost_marble_count won_marble_count final_marble_count : ℕ)

def calculate_final_marble_count (initial : ℕ) (lost : ℕ) (won : ℕ) : ℕ :=
  initial - lost + won

theorem alvin_marble_count_correct :
  initial_marble_count = 57 →
  lost_marble_count = 18 →
  won_marble_count = 25 →
  final_marble_count = calculate_final_marble_count initial_marble_count lost_marble_count won_marble_count →
  final_marble_count = 64 :=
by
  intros h_initial h_lost h_won h_calculate
  rw [h_initial, h_lost, h_won] at h_calculate
  exact h_calculate

end alvin_marble_count_correct_l220_220668


namespace unbiased_variance_estimate_l220_220740

theorem unbiased_variance_estimate :
  ∀ (n : ℕ) (D_b : ℝ), n = 41 → D_b = 3 → (n * D_b) / (n - 1) = 3.075 := 
by
  intros n D_b hn hDb
  rw [hn, hDb]
  have h1 : 41 * 3 = 123 := by norm_num
  have h2 : 41 - 1 = 40 := by norm_num
  rw [h1, h2]
  norm_num

end unbiased_variance_estimate_l220_220740


namespace greatest_divisor_of_arithmetic_sum_l220_220110

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end greatest_divisor_of_arithmetic_sum_l220_220110


namespace find_p_l220_220563

noncomputable def rectangle_pqrs_value_of_p
  (P : ℝ × ℝ) (S : ℝ × ℝ) (Q : ℝ × ℝ) (area : ℝ) : Prop :=
  P = (2, 3) ∧ S = (12, 3) ∧ Q.fst = p ∧ Q.snd = 15 ∧ area = 120 → p = 15

-- We are given:
def P : ℝ × ℝ := (2, 3)
def S : ℝ × ℝ := (12, 3)
def Area : ℝ := 120

-- We need to prove:
theorem find_p (p : ℝ) :
  rectangle_pqrs_value_of_p P S (p, 15) Area := sorry

end find_p_l220_220563


namespace quadratic_solution_range_l220_220684

theorem quadratic_solution_range (a : ℝ) (h1 : 16 * (a - 3) > 0) (h2 : a > 0) (h3 : a ^ 2 - 16 * a + 48 > 0) :
  a ∈ set.Ioo 3 4 ∨ a ∈ set.Ioo 12 19 ∨ a ∈ set.Ioi 19 :=
sorry

end quadratic_solution_range_l220_220684


namespace max_n_Sn_lt_2023_l220_220393

theorem max_n_Sn_lt_2023 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, a 1 = 4) →
    (∀ n, a n + a (n + 1) = 4 * n + 2) →
    (∀ n, S n = ∑ i in finset.range n, a (i + 1)) →
  ∃ n, S n < 2023 ∧ ∀ m, S m < 2023 → m ≤ n :=
sorry

end max_n_Sn_lt_2023_l220_220393


namespace inscribed_square_side_length_l220_220683

-- Define a right triangle
structure RightTriangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)
  (is_right : PQ^2 + QR^2 = PR^2)

-- Define the triangle PQR
def trianglePQR : RightTriangle :=
  { PQ := 6, QR := 8, PR := 10, is_right := by norm_num }

-- Define the problem statement
theorem inscribed_square_side_length (t : ℝ) (h : RightTriangle) :
  t = 3 :=
  sorry

end inscribed_square_side_length_l220_220683


namespace area_ratio_leq_quarter_l220_220926

theorem area_ratio_leq_quarter
    {A B C A1 B1 C1 : Type*}
    [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
    [InnerProductSpace ℝ A1] [InnerProductSpace ℝ B1] [InnerProductSpace ℝ C1]
    [Triangle ABC] [PointOnSide BC A1] [PointOnSide CA B1] [PointOnSide AB C1]
    (h_intersect : IntersectAtSinglePoint A A1 B B1 C C1) :
    AreaRatioABC A1 B1 C1 ABC ≤ 1 / 4 :=
sorry

end area_ratio_leq_quarter_l220_220926


namespace prime_frequency_in_range_l220_220035

def is_prime (n : Nat) : Prop :=
  1 < n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b : Nat) : List Nat :=
  List.filter is_prime (List.range' a (b - a + 1))

def count_primes_in_range (a b : Nat) : Nat :=
  List.length (primes_in_range a b)

theorem prime_frequency_in_range :
  let a := 1
  let b := 20
  let total_natural_numbers := b - a + 1
  let total_primes := count_primes_in_range a b
  (total_primes : ℚ) / total_natural_numbers = 0.4 :=
sorry

end prime_frequency_in_range_l220_220035


namespace june_biking_time_l220_220005

theorem june_biking_time :
  ∀ (d_jj d_jb : ℕ) (t_jj : ℕ), (d_jj = 2) → (t_jj = 8) → (d_jb = 6) →
  (t_jb : ℕ) → t_jb = (d_jb * t_jj) / d_jj → t_jb = 24 :=
by
  intros d_jj d_jb t_jj h_djj h_tjj h_djb t_jb h_eq
  rw [h_djj, h_tjj, h_djb] at h_eq
  simp at h_eq
  exact h_eq

end june_biking_time_l220_220005


namespace sum_of_six_consecutive_integers_l220_220554

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l220_220554


namespace part1_property_M_part2_property_M_l220_220842

-- Define the property M for a function f
def property_M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f(x₀ + 1) = f(x₀) + f(1)

-- Part (1): f(x) = 3^x has property M with a specific value of x₀
theorem part1_property_M (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3^x) : property_M f :=
  ∃ x₀ : ℝ, x₀ = 1 - real.logb 3 2 ∧ f (x₀ + 1) = f x₀ + f 1

-- Part (2): g(x) = log (a / (2x^2 + 1)) has property M implies a is within a specific range
theorem part2_property_M (g : ℝ → ℝ) (a : ℝ) (ha : 0 < a)
  (h : ∀ x : ℝ, g x = real.log (a / (2 * x^2 + 1))) : property_M g → 
  6 - 3 * real.sqrt 3 ≤ a ∧ a ≤ 6 + 3 * real.sqrt 3 :=
sorry

end part1_property_M_part2_property_M_l220_220842


namespace max_hours_worked_l220_220511

theorem max_hours_worked
  (r : ℝ := 8)  -- Regular hourly rate
  (h_r : ℝ := 20)  -- Hours at regular rate
  (r_o : ℝ := r + 0.25 * r)  -- Overtime hourly rate
  (E : ℝ := 410)  -- Total weekly earnings
  : (h_r + (E - r * h_r) / r_o) = 45 :=
by
  sorry

end max_hours_worked_l220_220511


namespace symmetric_point_on_altitude_l220_220530

variables {A B C O N : Type}
variables {midline_midpoint : A → B → C → Type}
variables {circumcenter : A → B → C → O → Prop}
variables {foot_perpendicular : O → A → B → N → Prop}
variables {midpoint_of_midline : A → B → C → A → B → A → B → midline_midpoint A B C}

theorem symmetric_point_on_altitude
  (circumcenter : circumcenter A B C O)
  (midline := midline_midpoint A B C (Classical.some midpoint (Classical.some midpoint (Classical.some midpoint (Classical.some midpoint)))))
  (foot := foot_perpendicular O A B N)
  (midpoint_theorem : ∀M : midline_midpoint A B C, 
                      ∃ t (α β : ℝ), 
                        α * (Classical.some midpoint (Classical.some midpoint)) + β * (Classical.some midpoint (Classical.some O)) = t * M)
  (symmetry : ∀ x y, 
                let P := x + (x - y) in
                (y - x = x - y) ↔ foot_perpendicular y P)
  : sorry

end symmetric_point_on_altitude_l220_220530


namespace sequence_a_n_bounded_sequence_b_n_unbounded_l220_220638

def a_n (n : ℕ) : ℕ :=
  (Finset.range (n / 3 + 1)).sum (λ k, Nat.choose n (3 * k))

def b_n (n : ℕ) : ℕ :=
  (Finset.range (n / 5 + 1)).sum (λ k, Nat.choose n (5 * k))

theorem sequence_a_n_bounded :
  ∃ M, ∀ n, abs ((a_n n : ℤ) - (2^n : ℤ) / 3) ≤ M := sorry

theorem sequence_b_n_unbounded :
  ¬ (∃ M, ∀ n, abs ((b_n n : ℤ) - (2^n : ℤ) / 5) ≤ M) := sorry

end sequence_a_n_bounded_sequence_b_n_unbounded_l220_220638


namespace alligator_population_l220_220519

/-
Problem Statement:
On Lagoon Island, the alligator population consists of males (m), adult females (af), and juvenile females (jf).
The ratio of males to adult females to juvenile females is 2:3:5.
During the mating season, the number of adult females doubles due to migration.
There are 15 adult females during the non-mating season.
Prove that the number of male alligators during the mating season is equal to 10.
-/

theorem alligator_population:
  let ratio_m := 2
  let ratio_af := 3
  let ratio_jf := 5
  let af_non_mating := 15
  let af_mating := 2 * af_non_mating
  let males := af_non_mating * (ratio_m / ratio_af) in
  males = 10 :=
by
  sorry  -- proof omitted

end alligator_population_l220_220519


namespace digging_foundation_l220_220879

-- Define given conditions
variable (m1 d1 m2 d2 k : ℝ)
variable (md_proportionality : m1 * d1 = k)
variable (k_value : k = 20 * 6)

-- Prove that for 30 men, it takes 4 days to dig the foundation
theorem digging_foundation : m1 = 20 ∧ d1 = 6 ∧ m2 = 30 → d2 = 4 :=
by
  sorry

end digging_foundation_l220_220879


namespace count_positive_whole_numbers_with_cube_root_less_than_12_l220_220419

theorem count_positive_whole_numbers_with_cube_root_less_than_12 :
  {n : ℕ | ∃ (k : ℕ), n = k ∧ k < 12 ∧ 1 ≤ k}.card = 1727 := sorry

end count_positive_whole_numbers_with_cube_root_less_than_12_l220_220419


namespace both_dislike_radio_and_music_l220_220037

theorem both_dislike_radio_and_music (total_people : ℕ) (perc_dislike_radio perc_both : ℚ)
  (h_total : total_people = 1500)
  (h_perc_dislike_radio : perc_dislike_radio = 0.4)
  (h_perc_both : perc_both = 0.15) :
  ∃ (num_both : ℕ), num_both = 90 :=
by
  have h_dislike_radio : total_people * perc_dislike_radio = 600 := by sorry
  have h_both_dislike : 600 * perc_both = 90 := by sorry
  use 90
  rw [h_both_dislike]
  norm_num
  sorry

end both_dislike_radio_and_music_l220_220037


namespace triangle_has_integer_area_l220_220589

def triangle_diagonal_and_area (a b c m n : ℝ) : Prop :=
  (a = real.sqrt 377) ∧ (b = real.sqrt 153) ∧ (c = real.sqrt 80) ∧ 
  ((∃ y₁ y₂ : ℤ, 377 = y₁^2 + y₂^2 ∧ m = real.sqrt (y₁^2) ∧ n = real.sqrt (y₂^2))) ∧
  ((∃ y₁ y₂ : ℤ, 153 = y₁^2 + y₂^2)) ∧ 
  ((∃ y₁ y₂ : ℤ, 80 = y₁^2 + y₂^2)) ∧ 
  (real.sqrt 377 * real.sqrt 153 >= 0) ∧ 
  (∀ (x1 x2 x3 : ℤ), (√377 * √153 * 2 = 42))

theorem triangle_has_integer_area : ∃ (a b c : ℝ), triangle_diagonal_and_area a b c _ _ :=
begin
  sorry
end

end triangle_has_integer_area_l220_220589


namespace volume_of_pyramid_l220_220950

noncomputable def pyramid_volume (l α β : ℝ) := 
  (2 / 3) * l ^ 3 * Real.tan (α / 2) * Real.sin β * Real.sin (2 * β)

theorem volume_of_pyramid 
  (α β l : ℝ)
  (hα : α > 0) 
  (hβ : 0 < β ∧ β < Real.pi / 2) 
  (hl : l > 0) :
  ∃ V : ℝ, V = pyramid_volume l α β :=
begin
  use pyramid_volume l α β,
  sorry,
end

end volume_of_pyramid_l220_220950


namespace sum_of_cube_faces_is_42_l220_220965

theorem sum_of_cube_faces_is_42 
  (x : ℕ) (h_even : x % 2 = 0)
  (h_faces: ∀ y ∈ {x, x + 2, x + 4, x + 6, x + 8, x + 10}, y % 2 = 0) 
  (h_pairs_equal : ∀ (a b : ℕ), 
                     a ∈ {x, x + 2, x + 4, x + 6, x + 8, x + 10} → 
                     b ∈ {x + 10, x + 8, x + 6, x + 4, x + 2, x} → 
                     a + b = 2 * x + 10) 
  : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 42) :=
by
  sorry

end sum_of_cube_faces_is_42_l220_220965


namespace int_square_last_digit_no_perfect_square_5n_plus_2_5n_plus_3_no_int_n_squared_plus_3_div_by_5_find_n_for_perfect_square_of_sequence_l220_220630

-- Part (a)
theorem int_square_last_digit (n : ℤ) : 
  ∃ d ∈ ({0, 1, 4, 5, 6, 9} : set ℤ), n^2 % 10 = d :=
sorry

-- Part (b)
theorem no_perfect_square_5n_plus_2_5n_plus_3 (n : ℕ) : 
  ¬ (∃ k : ℕ, (5 * n + 2 = k * k) ∨ (5 * n + 3 = k * k)) :=
sorry

-- Part (c)
theorem no_int_n_squared_plus_3_div_by_5 (n : ℤ) : 
  ¬ (n^2 + 3) % 5 = 0 :=
sorry

-- Part (d)
theorem find_n_for_perfect_square_of_sequence :
  ∃ n : ℕ, (∏ i in finset.range(n + 1), (i + 1)) + 97 = m * m for some m in ℕ :=
sorry

end int_square_last_digit_no_perfect_square_5n_plus_2_5n_plus_3_no_int_n_squared_plus_3_div_by_5_find_n_for_perfect_square_of_sequence_l220_220630


namespace dart_probability_inside_circle_l220_220653

noncomputable def side_len : ℝ := 2

noncomputable def radius : ℝ := side_len / (2 * tan (Real.pi / 10))

noncomputable def area_decagon : ℝ := (1 / 2) * 10 * side_len * radius

noncomputable def area_circle : ℝ := Real.pi * radius ^ 2

noncomputable def probability : ℝ := area_circle / area_decagon

theorem dart_probability_inside_circle :
  probability = Real.pi / (10 * tan (Real.pi / 10)) :=
by
  sorry

end dart_probability_inside_circle_l220_220653


namespace b_days_to_complete_work_l220_220631

theorem b_days_to_complete_work (x : ℕ) 
  (A : ℝ := 1 / 30) 
  (B : ℝ := 1 / x) 
  (C : ℝ := 1 / 40)
  (work_eq : 8 * (A + B + C) + 4 * (A + B) = 1) 
  (x_ne_0 : x ≠ 0) : 
  x = 30 := 
by
  sorry

end b_days_to_complete_work_l220_220631


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220813

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220813


namespace rotate_point_OA_90_counterclockwise_l220_220993

theorem rotate_point_OA_90_counterclockwise :
  let O := (0, 0)
  let B := (8, 0)
  ∃ A : ℝ × ℝ,
    (A.1 > 0 ∧ A.2 > 0) ∧
    (∠ A B O = 90) ∧
    (∠ A O B = 45) ∧
    let A_rotated := (-A.2, A.1)
    A_rotated = (-8, 8) :=
  
sorry

end rotate_point_OA_90_counterclockwise_l220_220993


namespace probability_of_three_correct_packages_l220_220304

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220304


namespace find_C_coordinates_l220_220461

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (0, -2)

-- Define the linear function passing through A and B
def is_linear_func (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = -2 * x - 2

-- Formulate the problem of finding the coordinate of C based on the area condition
def on_line_AB (C : ℝ × ℝ) : Prop :=
  C.2 = -2 * C.1 - 2

def area_triangle (C : ℝ × ℝ) : Prop :=
  let bx := B.1 in
  let by := B.2 in
  4 = (1 / 2) * (Real.abs (bx * C.2 - by * C.1))

-- Formalize the final problem statement
theorem find_C_coordinates (C : ℝ × ℝ) (h_line : on_line_AB C) (h_area : area_triangle C) :
  C = (4, -10) ∨ C = (-4, 6) :=
sorry

end find_C_coordinates_l220_220461


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220795

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220795


namespace remainder_prod_mod_7_l220_220715

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220715


namespace initial_amount_correct_l220_220053

-- Definitions based on given conditions
def r : ℝ := 0.10
def n : ℕ := 2
def t : ℝ := 1.5
def A : ℝ := 157.625

-- Question: Prove that the initial amount was approximately Rs. 136.082
theorem initial_amount_correct (P : ℝ) : 
  P * (1 + r / n) ^ (n * t) = A → P ≈ 136.082 := 
by
  sorry

end initial_amount_correct_l220_220053


namespace angle_CED_is_90_l220_220994

theorem angle_CED_is_90 
  (A B C D E : Point) 
  (h1 : Circle A r ∧ Circle B r) 
  (h2 : C ∈ Circle A r ∧ D ∈ Circle B r) 
  (h3 : E ∈ Circle A r ∧ E ∈ Circle B r) 
  (h4 : ∃ M : Point, M = midpoint A B) 
  (h5 : ∃ line_CD : Line, perpendicular bisector M CD ∧ CD passes through M
        ∧ CD intersects Circle A r at C 
        ∧ CD intersects Circle B r at D) 
  : angle CED = 90 :=
begin
  sorry
end

end angle_CED_is_90_l220_220994


namespace probability_of_exactly_three_correct_packages_l220_220290

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220290


namespace count_positive_whole_numbers_with_cube_root_less_than_12_l220_220417

theorem count_positive_whole_numbers_with_cube_root_less_than_12 :
  {n : ℕ | ∃ (k : ℕ), n = k ∧ k < 12 ∧ 1 ≤ k}.card = 1727 := sorry

end count_positive_whole_numbers_with_cube_root_less_than_12_l220_220417


namespace norm_linear_combination_l220_220406

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

variable (a b : ℝ × ℝ)
variable (h_angle : ∀ a b : ℝ × ℝ, real.arccos (dot_product a b / (vector_norm a * vector_norm b)) = real.pi / 4)
variable (h_norm_a : vector_norm a = real.sqrt 2)
variable (h_norm_b : vector_norm b = 1)

theorem norm_linear_combination :
  vector_norm (3 * (a.1, a.2) - 4 * (b.1, b.2)) = real.sqrt 22 :=
sorry

end norm_linear_combination_l220_220406


namespace heat_more_games_than_bulls_l220_220444

theorem heat_more_games_than_bulls (H : ℕ) 
(h1 : 70 + H = 145) :
H - 70 = 5 :=
sorry

end heat_more_games_than_bulls_l220_220444


namespace interval_of_n_l220_220355

theorem interval_of_n (n : ℕ) (h1 : n < 1000) 
  (h2 : (∀ k : ℕ, (1 / n : ℝ) * (10 ^ k) % 1 = (1 / n : ℝ) * (10 ^ (k + 6) % 1))
  (h3 : (∀ k : ℕ, (1 / (n + 6) : ℝ) * (10 ^ k) % 1 = (1 / (n + 6) : ℝ) * (10 ^ (k + 4) % 1)) : 
  201 ≤ n ∧ n ≤ 400 :=
sorry

end interval_of_n_l220_220355


namespace ratio_josh_to_doug_l220_220487

theorem ratio_josh_to_doug (J D B : ℕ) (h1 : J + D + B = 68) (h2 : J = 2 * B) (h3 : D = 32) : J / D = 3 / 4 := 
by
  sorry

end ratio_josh_to_doug_l220_220487


namespace rectangle_length_l220_220990

theorem rectangle_length (area width : ℝ) (h_area : area = 215.6) (h_width : width = 14) : 
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by 
  use 15.4
  split
  · rfl
  · rw [h_area, h_width]
    norm_num

end rectangle_length_l220_220990


namespace sequence_product_modulo_7_l220_220721

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l220_220721


namespace simplify_fraction_l220_220983

theorem simplify_fraction : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := 
by {
  sorry
}

end simplify_fraction_l220_220983


namespace keegan_total_classes_l220_220885

theorem keegan_total_classes (school_hours_per_day : ℝ)
                            (history_chemistry_hours : ℝ)
                            (other_class_minutes : ℝ)
                            (h₁ : school_hours_per_day = 7.5)
                            (h₂ : history_chemistry_hours = 1.5)
                            (h₃ : other_class_minutes = 72) :
  let total_minutes_school := school_hours_per_day * 60
      total_minutes_history_chemistry := history_chemistry_hours * 60
      remaining_minutes := total_minutes_school - total_minutes_history_chemistry
      num_other_classes := remaining_minutes / other_class_minutes
  in num_other_classes + 2 = 7 :=
by sorry

end keegan_total_classes_l220_220885


namespace problem_y_minus_x_l220_220595

theorem problem_y_minus_x (x y : ℝ) (h₁ : x + y = 580) (h₂ : x / y = 0.75) : y - x = 83 :=
begin
  sorry
end

end problem_y_minus_x_l220_220595


namespace number_of_proper_divisors_30_pow_4_l220_220955

theorem number_of_proper_divisors_30_pow_4 : 
  let n := 30^4
  let numberOfFactors := (do_factors_count n) - 2
  numberOfFactors = 123 := 
by
  sorry

def do_factors_count (n: ℕ) : ℕ := 
  -- Counts number of divisors of n
  sorry

end number_of_proper_divisors_30_pow_4_l220_220955


namespace intersection_of_sets_l220_220378

def setA : Set ℝ := {x | (x - 2) / x ≤ 0}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def setC : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets : setA ∩ setB = setC :=
by
  sorry

end intersection_of_sets_l220_220378


namespace arithmetic_sequence_6th_term_l220_220084

theorem arithmetic_sequence_6th_term :
  let a : ℕ → ℤ := λ n, -3 + n * 4 in a 5 = 17 :=
by
  let a : ℕ → ℤ := λ n, -3 + n * 4
  show a 5 = 17
  sorry

end arithmetic_sequence_6th_term_l220_220084


namespace remainder_prod_mod_7_l220_220727

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l220_220727


namespace equal_sides_of_triangle_l220_220901

-- Definitions of points and circles involved in the problem
variables {A B C D E F G X : Point} -- Points on the plane
variables (Γ1 : Circle) -- Circumcircle Γ_1
variables (incircle : Circle) -- Incircle of triangle ABC

-- Conditions given in the problem
variables (H1 : is_circumcircle Γ1 ΔABC) -- Γ1 is the circumcircle of ΔABC
variables (H2 : midpoint_arc_D Γ1 A B ≠ C) -- D is the midpoint of arc AB not containing C
variables (H3 : midpoint_arc_E Γ1 A C ≠ B) -- E is the midpoint of arc AC not containing B
variables (H4 : is_incircle incircle ΔABC) -- Incircle of ΔABC
variables (H5 : touches_side_a AB incircle F) -- incircle touches AB at F
variables (H6 : touches_side_b AC incircle G) -- incircle touches AC at G
variables (H7 : intersects_at_line_eg_df E G DF X) -- lines EG and DF intersect at X
variables (H8 : XD = XE) -- Given XD = XE

-- Theorem to prove AB = AC
theorem equal_sides_of_triangle
    (Γ1 : Circle) (ABC : Triangle) (incircle : Circle) (D E F G X : Point)
    (H1 : is_circumcircle Γ1 ABC)
    (H2 : midpoint_arc_D Γ1 A B ≠ C)
    (H3 : midpoint_arc_E Γ1 A C ≠ B)
    (H4 : is_incircle incircle ABC)
    (H5 : touches_side_a AB incircle F)
    (H6 : touches_side_b AC incircle G)
    (H7 : intersects_at_line_eg_df E G DF X)
    (H8 : XD = XE) : 
  AB = AC := 
sorry

end equal_sides_of_triangle_l220_220901


namespace B_days_to_complete_work_l220_220159

theorem B_days_to_complete_work:
  ∀ (x : ℕ), 
    (∀ (f A B : ℝ),
      A = 1 / 15 ∧ 
      B = 1 / x ∧ 
      6 * (A + B) = 0.7 → 
      x = 20) :=
begin
  sorry
end

end B_days_to_complete_work_l220_220159


namespace sampling_method_is_systematic_l220_220604

/-- 
  To conduct a spot check on the annual inspection of cars in a certain city, 
  cars with license plates ending in the digit 6 are selected for inspection 
  on the main roads of the city. This sampling method is systematic sampling.
-/
theorem sampling_method_is_systematic :
  (∀ car, car.has_license_plate_ending_in_6 → car.selected_for_inspection) →
  systematic_sampling_method :=
by
  sorry

end sampling_method_is_systematic_l220_220604


namespace lasagna_ground_mince_l220_220196

theorem lasagna_ground_mince (total_ground_mince : ℕ) (num_cottage_pies : ℕ) (ground_mince_per_cottage_pie : ℕ) 
  (num_lasagnas : ℕ) (L : ℕ) : 
  total_ground_mince = 500 ∧ num_cottage_pies = 100 ∧ ground_mince_per_cottage_pie = 3 
  ∧ num_lasagnas = 100 ∧ total_ground_mince - num_cottage_pies * ground_mince_per_cottage_pie = num_lasagnas * L 
  → L = 2 := 
by sorry

end lasagna_ground_mince_l220_220196


namespace absolute_prime_digit_condition_l220_220431

-- Define absolute prime
def is_absolute_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ ∀ (perm : List ℕ), (perm.permutes n.digits → Nat.Prime (List.natOfDigits perm))

-- Define statement for the proof problem
theorem absolute_prime_digit_condition : ∀ n : ℕ, is_absolute_prime n → ∃ k ≤ 3, (∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d → False) :=
by
  intros n h,
  sorry

end absolute_prime_digit_condition_l220_220431


namespace new_train_distance_l220_220181

theorem new_train_distance (older_train_distance : ℝ) (percentage_increase : ℝ) (new_train_distance : ℝ) :
  older_train_distance = 300 → percentage_increase = 0.30 → 
  new_train_distance = older_train_distance + older_train_distance * percentage_increase → 
  new_train_distance = 390 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end new_train_distance_l220_220181


namespace calculate_mari_buttons_l220_220021

theorem calculate_mari_buttons (Sue_buttons : ℕ) (Kendra_buttons : ℕ) (Mari_buttons : ℕ)
  (h_sue : Sue_buttons = 6)
  (h_kendra : Sue_buttons = 0.5 * Kendra_buttons)
  (h_mari : Mari_buttons = 4 + 5 * Kendra_buttons) :
  Mari_buttons = 64 := 
by
  sorry

end calculate_mari_buttons_l220_220021


namespace general_term_sum_first_n_terms_l220_220371

noncomputable def a : ℕ → ℝ
| 1 := 1
| (n + 2) := -a (n + 1) + (a (n + 1) - 1) * a (n + 1)

theorem general_term (n : ℕ) (h : n > 0) : a (n + 1) = 1 / (n + 1) :=
by
  sorry

noncomputable def c : ℕ → ℝ
| n := 3 ^ n * (a (n + 1))⁻¹

noncomputable def S (n : ℕ) : ℝ :=
finset.sum (finset.range (n + 1)) c

theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 1) / 4 * 3^(n + 1) + 3 / 4 :=
by
  sorry

end general_term_sum_first_n_terms_l220_220371


namespace calculate_mari_buttons_l220_220022

theorem calculate_mari_buttons (Sue_buttons : ℕ) (Kendra_buttons : ℕ) (Mari_buttons : ℕ)
  (h_sue : Sue_buttons = 6)
  (h_kendra : Sue_buttons = 0.5 * Kendra_buttons)
  (h_mari : Mari_buttons = 4 + 5 * Kendra_buttons) :
  Mari_buttons = 64 := 
by
  sorry

end calculate_mari_buttons_l220_220022


namespace sum_of_midpoints_y_coordinates_l220_220079

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l220_220079


namespace cube_root_neg_eighth_l220_220679

theorem cube_root_neg_eighth : ∃ x : ℚ, x^3 = -1 / 8 ∧ x = -1 / 2 :=
by
  sorry

end cube_root_neg_eighth_l220_220679


namespace least_three_digit_divisible_by_2_3_5_7_l220_220621

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end least_three_digit_divisible_by_2_3_5_7_l220_220621


namespace radius_inscribed_circle_correct_l220_220171

noncomputable def radius_of_inscribed_circle
  (O : Point) (A : Point) (B : Point) (C : Point) (M : Point)
  (R : ℝ) (r : ℝ) (angle_A : ℝ) (AM_ratio_MO : ℝ) (BC_length : ℝ) (sqrt3 : ℝ) : Prop :=
  angle_A = 60 ∧
  tangent_to_circle B O ∧
  tangent_to_circle C O ∧
  (BC_length = 7) ∧
  (AM_ratio_MO = 2 / 3) ∧
  (BC = Line B C) ∧
  (AO = Line A O) ∧
  (AO ∩ BC = M) ∧
  (AM = Segment A M) ∧
  (MO = Segment M O) ∧
  (∃ p, 
     let S_ABC := 1 / 2 * 7 * (2 / 3 * R) in
     let S_alt := (R * sqrt3) * r in 
     S_ABC = S_alt ∧
     r = (7 * sqrt3) / 9 )

theorem radius_inscribed_circle_correct
  (O A B C M : Point)
  (R r angle_A AM_ratio_MO BC_length sqrt3 : ℝ)
  (h1 : angle_A = 60)
  (h2 : tangent_to_circle B O)
  (h3 : tangent_to_circle C O)
  (h4 : BC_length = 7)
  (h5 : AM_ratio_MO = 2 / 3)
  (h6 : BC = Line B C)
  (h7 : AO = Line A O)
  (h8 : AO ∩ BC = M)
  (h9 : AM = Segment A M)
  (h10 : MO = Segment M O)
  : r = (7 * sqrt3) / 9 :=
begin
  -- proof goes here
  sorry
end

end radius_inscribed_circle_correct_l220_220171


namespace solve_for_x_l220_220544

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l220_220544


namespace constant_term_is_60_l220_220867

noncomputable def constant_term_expansion (x : ℤ) : ℤ :=
  let term := (3 + x) * (x + 1/x)^6 in
  nexpr term 0  -- nexpr is a hypothetical function to retrieve the n-th power term (here 0-th power) from the expression

theorem constant_term_is_60 : 
  constant_term_expansion x = 60 := 
by 
  sorry

end constant_term_is_60_l220_220867


namespace problem1_simplify_expr_problem2_simplify_expression_l220_220147

-- Problem (1)
theorem problem1_simplify_expr : (sqrt 2 + 1 + 2 * (sqrt 2 / 2) + 4 = 2 * sqrt 2 + 5) :=
by
  sorry

-- Problem (2)
theorem problem2_simplify_expression (x y : ℝ) (hx : x = sqrt 2 + 1) (hy : y = sqrt 2 - 1) :
  ((x^2 + y^2 - 2*x*y) / (x - y) / (x / y - y / x) = (sqrt 2) / 4) :=
by
  sorry

end problem1_simplify_expr_problem2_simplify_expression_l220_220147


namespace simplify_sqrt_expression_l220_220539

theorem simplify_sqrt_expression : sqrt(8 + 4 * sqrt 3) + sqrt(8 - 4 * sqrt 3) = 2 * sqrt 6 :=
by
  sorry

end simplify_sqrt_expression_l220_220539


namespace probability_of_three_correct_packages_l220_220296

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220296


namespace product_of_real_r_with_one_solution_l220_220254

theorem product_of_real_r_with_one_solution : 
  (∀ x : ℝ, x ≠ 0 → (∃! (r : ℝ), (1 / (3 * x)) = ((r - x) / 9))) → (∏ (r : ℝ) in {r | ∃ x : ℝ, x ≠ 0 ∧ (1 / (3 * x)) = ((r - x) / 9)}.to_finset id = 0) :=
by
  sorry

end product_of_real_r_with_one_solution_l220_220254


namespace girl_weaves_on_tenth_day_l220_220458

theorem girl_weaves_on_tenth_day 
  (a1 d : ℝ)
  (h1 : 7 * a1 + 21 * d = 28)
  (h2 : a1 + d + a1 + 4 * d + a1 + 7 * d = 15) :
  a1 + 9 * d = 10 :=
by sorry

end girl_weaves_on_tenth_day_l220_220458


namespace probability_exactly_three_correct_l220_220353

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220353


namespace expected_bounces_l220_220464

-- Definitions for the conditions stated in the problem
def rectangular_billiard_table := { length := 3, width := 1 }
def center := (rectangular_billiard_table.length / 2, rectangular_billiard_table.width / 2)
def travel_distance := 2

-- The mathematical statement
theorem expected_bounces (table : rectangular_billiard_table) 
  (center_pos : center) (distance_traveled : travel_distance):
  let p2 := (2 / Real.pi) * (Real.arccos (3 / 4) + Real.arccos (1 / 4) - Real.arcsin (3 / 4)),
  p1 := 1 - p2,
  expected_reflections := 1 + p2
  in 
    expected_reflections ≈ 1.7594 := 
by
  sorry

end expected_bounces_l220_220464


namespace perpendicular_lines_parallel_l220_220042

variables (a b : Type) [line_space a] [line_space b]
variables (α : Type) [plane_space α]

axiom perpendicular (a : line_space) (α : plane_space) : Prop
axiom parallel (a : line_space) (b : line_space) : Prop

theorem perpendicular_lines_parallel (a b : Type) [line_space a] [line_space b] 
  (α : Type) [plane_space α] 
  (h1 : perpendicular a α) (h2 : perpendicular b α) : parallel a b := 
sorry

end perpendicular_lines_parallel_l220_220042


namespace versatile_nontrivial_network_iff_odd_l220_220050

-- Define some basic prerequisites.
variables {V E : Type*} [Fintype V] [Fintype E] [DecidableEq E]

-- Network represented as a graph.
structure Network (n : ℕ) :=
  (points : Fin n → V)
  (adj : ∀ (i j : Fin n), i ≠ j → Prop)
  (network_condition : ∀ (a b : Fin n), connected (adj) a.val b.val)

-- Definitions of spin and labels.
def spin (network : Network n) (v : V) : Network n := sorry

def is_versatile (network : Network n) : Prop :=
  ∀ label1 label2 : Fin (n - 1) → E, 
    ∃ (spins : (Network n → Network n) → ℕ), 
      (iterate spins network (Network.spin network v))label1 = label2

-- Define nontrivial network.
def is_nontrivial (network : Network n) : Prop :=
  ∃ v : Fin n, 2 ≤ Fintype.card {e : E // e ∈ network.points v}

-- Final statement: find all odd n ≥ 5 such that any nontrivial network with n points is versatile.
theorem versatile_nontrivial_network_iff_odd (n : ℕ) (hn: n ≥ 5) : 
  (∀ network : Network n, is_nontrivial network → is_versatile network) ↔ (n % 2 = 1) :=
sorry

end versatile_nontrivial_network_iff_odd_l220_220050


namespace trajectory_of_vertex_C_l220_220083

theorem trajectory_of_vertex_C (x y : ℝ) (h1 : x ≠ 5) (h2 : x ≠ -5) :
  let k_AC := y / (x + 5)
  let k_BC := y / (x - 5)
  let slope_condition : k_AC * k_BC = -1 / 2
  slope_condition → (x ^ 2 / 25 + y ^ 2 / (25 / 2) = 1) := 
sorry

end trajectory_of_vertex_C_l220_220083


namespace probability_divisible_by_5_l220_220963

/- 
  We need to prove that the probability of a randomly chosen divisor of 15! 
  being divisible by 5 is equal to 3/4, given the prime factorization of 15!.
-/ 

theorem probability_divisible_by_5 :
  let prime_factorization := (2^11) * (3^6) * (5^3) * (7^2) * 11 * 13 in
  let total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) in
  let divisors_divisible_by_5 := (11 + 1) * (6 + 1) * (2 + 1) * (2 + 1) * (1 + 1) * (1 + 1) in
  let probability := divisors_divisible_by_5 / total_divisors in
  prime_factorization = 15! ∧ total_divisors = 4032 ∧ divisors_divisible_by_5 = 3024 → 
  probability = 3 / 4 
  :=
by
  intros
  split
  sorry

end probability_divisible_by_5_l220_220963


namespace proof_math_problem_l220_220121

-- Definitions for the given values
def a : ℝ := 0.32
def x : ℝ := 0.08

-- Main theorem stating the equality
theorem proof_math_problem :
  (sqrt 2 * (x - a) / (2 * x - a)
   - sqrt ((sqrt x / (sqrt (2 * x) + sqrt a)) ^ 2 + (2 * sqrt a / (sqrt (2 * x) + sqrt a))) = 1 :=
by
  -- Skip the proof here
  sorry

end proof_math_problem_l220_220121


namespace probability_three_correct_deliveries_l220_220335

theorem probability_three_correct_deliveries :
  let total_deliveries := 5!
  let ways_choose_three_correct := Nat.choose 5 3
  let derangement_two := 1
  let favorable_outcomes := ways_choose_three_correct * derangement_two
  (favorable_outcomes.to_rat / total_deliveries.to_rat) = (1 / 12) := by
  sorry

end probability_three_correct_deliveries_l220_220335


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220794

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l220_220794


namespace smallest_delightful_integer_l220_220936

-- Definition of "delightful" integer
def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ ((n + 1) * (2 * B + n)) / 2 = 3050

-- Proving the smallest delightful integer
theorem smallest_delightful_integer : ∃ (B : ℤ), is_delightful B ∧ ∀ (B' : ℤ), is_delightful B' → B ≤ B' :=
  sorry

end smallest_delightful_integer_l220_220936


namespace triangle_median_inequality_l220_220876

-- Defining the parameters and the inequality theorem.
theorem triangle_median_inequality
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (Δ : ℝ)
  (median_medians : ∀ {a b c : ℝ}, ma ≤ mb ∧ mb ≤ mc ∧ a ≥ b ∧ b ≥ c)  :
  a * (-ma + mb + mc) + b * (ma - mb + mc) + c * (ma + mb - mc) ≥ 6 * Δ := 
sorry

end triangle_median_inequality_l220_220876


namespace remainder_prod_mod_7_l220_220713

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220713


namespace center_of_circle_l220_220952

theorem center_of_circle :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → (x, y) = (1, 1) :=
by
  sorry

end center_of_circle_l220_220952


namespace compare_sine_values_1_compare_sine_values_2_l220_220227

theorem compare_sine_values_1 (h1 : 0 < Real.pi / 10) (h2 : Real.pi / 10 < Real.pi / 8) (h3 : Real.pi / 8 < Real.pi / 2) :
  Real.sin (- Real.pi / 10) > Real.sin (- Real.pi / 8) :=
by
  sorry

theorem compare_sine_values_2 (h1 : 0 < Real.pi / 8) (h2 : Real.pi / 8 < 3 * Real.pi / 8) (h3 : 3 * Real.pi / 8 < Real.pi / 2) :
  Real.sin (7 * Real.pi / 8) < Real.sin (5 * Real.pi / 8) :=
by
  sorry

end compare_sine_values_1_compare_sine_values_2_l220_220227


namespace monotonic_decreasing_range_a_l220_220436

theorem monotonic_decreasing_range_a (a : ℝ) :
  (∀ x : ℝ, deriv (λ x, sin (2 * x) + 4 * cos x + a * x) ≤ 0) → a ≤ -3 :=
by
  sorry

end monotonic_decreasing_range_a_l220_220436


namespace binary_rep_235_l220_220686

theorem binary_rep_235 :
  let x := (nat.binaryDigits 235).count (eq 0),
      y := (nat.binaryDigits 235).count (eq 1)
  in 2 * y - 3 * x = 11 := by
  sorry

end binary_rep_235_l220_220686


namespace area_of_circle_l220_220863

-- Given condition as a Lean definition
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 + 9 * x - 12 * y - 27 = 0

-- Theorem stating the goal
theorem area_of_circle : ∀ (x y : ℝ), circle_eq x y → ∃ r : ℝ, r = 15.25 ∧ ∃ a : ℝ, a = π * r := 
sorry

end area_of_circle_l220_220863


namespace general_formula_for_sequence_a_l220_220389

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end general_formula_for_sequence_a_l220_220389


namespace sum_of_six_consecutive_integers_l220_220555

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l220_220555


namespace minimize_distance_sum_l220_220055

def point (α : Type*) := prod α α

def distance (P Q : point ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

noncomputable def minimized_k : ℝ :=
  let A : point ℝ := (3, 3)
  let B : point ℝ := (-1, -1)
  let C : ℝ → point ℝ := λ k, (-3, k)
  classical.some (real.exists_local_min_on_Icc (λ k, distance A (C k) + distance B (C k))
    (-100) 100) -- Assuming some reasonable bounds for searching

theorem minimize_distance_sum :
  minimized_k = -9 :=
by
  sorry

end minimize_distance_sum_l220_220055


namespace range_of_data_set_l220_220969

def data_set : Set ℕ := {3, 4, 4, 6}

theorem range_of_data_set : 
  (range (data_set)) = 3 :=
sorry

end range_of_data_set_l220_220969


namespace probability_three_primes_from_1_to_30_l220_220090

noncomputable def prob_three_primes : ℚ := 
  let primes := {x ∈ Finset.range 31 | Nat.Prime x }.card
  let total_combinations := (Finset.range 31).card.choose 3
  let prime_combinations := primes.choose 3
  prime_combinations / total_combinations

theorem probability_three_primes_from_1_to_30 : prob_three_primes = 10 / 339 := 
  by
    sorry

end probability_three_primes_from_1_to_30_l220_220090


namespace number_of_pizza_varieties_l220_220210

-- Definitions for the problem conditions
def number_of_flavors : Nat := 8
def toppings : List String := ["C", "M", "O", "J", "L"]

-- Function to count valid combinations of toppings
def valid_combinations (n : Nat) : Nat :=
  match n with
  | 1 => 5
  | 2 => 10 - 1 -- Subtracting the invalid combination (O, J)
  | 3 => 10 - 3 -- Subtracting the 3 invalid combinations containing (O, J)
  | _ => 0

def total_topping_combinations : Nat :=
  valid_combinations 1 + valid_combinations 2 + valid_combinations 3

-- The final proof stating the number of pizza varieties
theorem number_of_pizza_varieties : total_topping_combinations * number_of_flavors = 168 := by
  -- Calculation steps can be inserted here, we use sorry for now
  sorry

end number_of_pizza_varieties_l220_220210


namespace quadratic_complete_square_l220_220942

theorem quadratic_complete_square (x m n : ℝ) 
  (h : 9 * x^2 - 36 * x - 81 = 0) :
  (x + m)^2 = n ∧ m + n = 11 :=
sorry

end quadratic_complete_square_l220_220942


namespace probability_three_correct_out_of_five_l220_220282

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220282


namespace remainder_8_pow_900_mod_29_l220_220735

theorem remainder_8_pow_900_mod_29 : 8^900 % 29 = 7 :=
by sorry

end remainder_8_pow_900_mod_29_l220_220735


namespace pave_square_with_tiles_l220_220945

theorem pave_square_with_tiles (b c : ℕ) (h_right_triangle : (b > 0) ∧ (c > 0)) :
  (∃ (k : ℕ), k^2 = b^2 + c^2) ↔ (∃ (m n : ℕ), m * c * b = 2 * n^2 * (b^2 + c^2)) := 
sorry

end pave_square_with_tiles_l220_220945


namespace cos_960_eq_neg_half_l220_220242

theorem cos_960_eq_neg_half (cos : ℝ → ℝ) (h1 : ∀ x, cos (x + 360) = cos x) 
  (h_even : ∀ x, cos (-x) = cos x) (h_cos120 : cos 120 = - cos 60)
  (h_cos60 : cos 60 = 1 / 2) : cos 960 = -(1 / 2) := by
  sorry

end cos_960_eq_neg_half_l220_220242


namespace find_greater_number_l220_220256

def HCF (a b : ℕ) : ℕ := a.gcd b

theorem find_greater_number (a b : ℕ) 
  (h1 : HCF a b = 23)
  (h2 : a * b = 98596)
  (h3 : ∀ x y : ℕ, HCF x y = 23 → x * y = 98596 → abs (x - y) ≥ abs (a - b)) : max a b = 713 := 
sorry

end find_greater_number_l220_220256


namespace circle_symmetry_l220_220567

def initial_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 1 = 0

def standard_form_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

def symmetric_circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

theorem circle_symmetry :
  (∀ x y : ℝ, initial_circle_eq x y ↔ standard_form_eq x y) →
  (∀ x y : ℝ, standard_form_eq x y → symmetric_circle_eq (-x) (-y)) →
  ∀ x y : ℝ, initial_circle_eq x y → symmetric_circle_eq x y :=
by
  intros h1 h2 x y hxy
  sorry

end circle_symmetry_l220_220567


namespace emma_missing_fraction_l220_220696

theorem emma_missing_fraction (x : ℝ) : 
  (1/3 * x) - (2/3 * (1/3 * x)) = 1/9 * x :=
by
  -- Define the number of coins initially lost
  let lost := 1/3 * x
  -- Define the number of coins found
  let found := 2/3 * lost
  -- Define the number of coins still missing
  let still_missing := lost - found
  -- Simplify the fraction
  calc 
    still_missing = (1/3 * x) - (2/3 * (1/3 * x)) : by rfl
    ...         = 1/9 * x : by sorry

end emma_missing_fraction_l220_220696


namespace product_expression_value_l220_220636

theorem product_expression_value (m : ℕ) : 
  (finset.range m).prod (λ k, (1 + 1 / (2 * k + 1) : ℚ) * (1 - 1 / (2 * k + 2) : ℚ)) = 1 :=
sorry

end product_expression_value_l220_220636


namespace odd_terms_in_expansion_l220_220828

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l220_220828


namespace sphere_curvilinear_quadrilateral_areas_l220_220200

theorem sphere_curvilinear_quadrilateral_areas (R : ℝ) : 
  let area1 := (2 * π * R^2 / 3) * (4 * sqrt (2 / 3) - 3),
      area2 := π * R^2 * (16 / 3 * sqrt (2 / 3) - 2) in 
  (quadrilateral_area_1 = area1) ∧ (quadrilateral_area_2 = area2) := 
sorry

end sphere_curvilinear_quadrilateral_areas_l220_220200


namespace sqrt_expression_meaningful_l220_220948

/--
When is the algebraic expression √(x + 2) meaningful?
To ensure the algebraic expression √(x + 2) is meaningful, 
the expression under the square root, x + 2, must be greater than or equal to 0.
Thus, we need to prove that this condition is equivalent to x ≥ -2.
-/
theorem sqrt_expression_meaningful (x : ℝ) : (x + 2 ≥ 0) ↔ (x ≥ -2) :=
by
  sorry

end sqrt_expression_meaningful_l220_220948


namespace coefficient_of_a2b3_in_expansion_l220_220615

theorem coefficient_of_a2b3_in_expansion (a b c : ℂ) :
  (coeff a b c (a + b)^5 * (c^2 + (1/c^2))^3 (a^2 * b^3) = 0) :=
by
  sorry

end coefficient_of_a2b3_in_expansion_l220_220615


namespace fractions_of_group_money_l220_220905

def moneyDistribution (m l n o : ℕ) (moeGave : ℕ) (lokiGave : ℕ) (nickGave : ℕ) : Prop :=
  moeGave = 1 / 5 * m ∧
  lokiGave = 1 / 4 * l ∧
  nickGave = 1 / 3 * n ∧
  moeGave = lokiGave ∧
  lokiGave = nickGave ∧
  o = moeGave + lokiGave + nickGave

theorem fractions_of_group_money (m l n o total : ℕ) :
  moneyDistribution m l n o 1 1 1 →
  total = m + l + n →
  (o : ℚ) / total = 1 / 4 :=
by sorry

end fractions_of_group_money_l220_220905


namespace weight_of_mixture_l220_220057

variable (C : ℝ) -- Cost per pound of milk powder and coffee in June
variable (weight : ℝ) -- Weight of the mixture in July
variable (milk_price_july : ℝ := 0.4) -- Cost per pound of milk powder in July

-- Define the conditions
def cost_coffee_july := 3 * C
def cost_milk_july := 0.4 * C
def equal_cost_july (cost : ℝ) := cost_coffee_july C + cost_milk_july C
def cost_per_pound_mixture_july := (cost_coffee_july C + cost_milk_july C) / 2
def mixture_cost_july := 5.10

theorem weight_of_mixture : C = 1 ∧ ((0.4 : ℝ) / C) = 1 ∧ cost_per_pound_mixture_july = 1.7 ∧ mixture_cost_july = 5.10 → weight = 3 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end weight_of_mixture_l220_220057


namespace number_of_valid_arrangements_l220_220180

   def factorial : ℕ → ℕ
   | 0       := 1
   | (n + 1) := (n + 1) * factorial n

   def comb (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

   def circular_arrangements (total identicals : ℕ) : ℕ :=
     factorial (total - 1) / (identicals - 1)!

   def valid_plate_arrangements : ℕ :=
     circular_arrangements 10 4 3 3 - circular_arrangements 7 4 3

   theorem number_of_valid_arrangements :
     valid_plate_arrangements = 24780 :=
   by
     sorry
   
end number_of_valid_arrangements_l220_220180


namespace not_prime_x_plus_y_minus_one_l220_220899

theorem not_prime_x_plus_y_minus_one (x y : ℤ) (hx : x > 1) (hy : y > 1) 
  (hdiv : x + y - 1 ∣ x^2 + y^2 - 1) : ¬ Nat.Prime (x + y - 1) := by
  sorry

end not_prime_x_plus_y_minus_one_l220_220899


namespace probability_exactly_three_correct_l220_220345

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220345


namespace geometric_sequence_ordered_pairs_l220_220587

/-- The sequence b_1, b_2, ..., is geometric with b_1 = b and common ratio s, where b and s are 
positive integers. Given that log_4 b_1 + log_4 b_2 + ... + log_4 b_{10} = 2010, find the
number of possible ordered pairs (b, s). --/
theorem geometric_sequence_ordered_pairs (b s : ℕ) (h1 : b > 0) (h2 : s > 0)
  (h3 : ∑ i in finset.range 10, real.logb 4 (b * s^i) = 2010) :
  (finset.filter (λ (p : ℕ × ℕ), p.1 > 0 ∧ p.2 > 0 ∧ 
    ∑ i in finset.range 10, real.logb 4 (p.1 * p.2^i) = 2010) 
    (finset.Icc (1, 1) (b, s))).card = 45 :=
sorry

end geometric_sequence_ordered_pairs_l220_220587


namespace cos_B_in_triangle_l220_220442

theorem cos_B_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = (Real.sqrt 5 / 2) * b)
  (h2 : A = 2 * B)
  (h_triangle: A + B + C = Real.pi) : 
  Real.cos B = Real.sqrt 5 / 4 :=
sorry

end cos_B_in_triangle_l220_220442


namespace problem1_problem2_l220_220150

-- Problem 1
theorem problem1 : 
  let I := 1/27
  in (I^(1/3))^(1/3) - (6 + 1/4) * (1/2) + (2 * Real.sqrt 2) - (2/3) + Real.pi^0 - 3^(-1) = -1 := 
by 
  have h1 : I = (1/27) := rfl
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1) (h : x + x⁻¹ = 4) : 
  (x^2 - x^(-2)) / (x^(1/2) + x^(-1/2)) = -4 * Real.sqrt 2 := 
by 
  have h1 : x + x⁻¹ = 4 := h
  have h2 : 0 < x ∧ x < 1 := ⟨hx1, hx2⟩
  sorry

end problem1_problem2_l220_220150


namespace part1_part2_l220_220401

-- Given hyperbola C
def hyperbola (x y: ℝ) : Prop := x^2 - y^2 = 1

-- Given line l
def line (x y k: ℝ) : Prop := y = k * x + 1

-- Midpoint condition
def midpoint_condition (x1 x2: ℝ) : Prop := (x1 + x2) / 2 = Real.sqrt 2

-- Length function
def length_AB (x1 x2 y1 y2: ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Part 1: Range of k
theorem part1 (k: ℝ) :
  (∃ x y, hyperbola x y ∧ line x y k ∧ x ≠ y) ↔ k ∈ Ioo (-Real.sqrt 2) (-1) ∪ Ioo (-1) 1 ∪ Ioo 1 (Real.sqrt 2) :=
sorry

-- Part 2: Length of AB
theorem part2 (x1 x2 y1 y2 k: ℝ) (h1: hyperbola x1 y1) (h2: hyperbola x2 y2) (h3: line x1 y1 k) (h4: line x2 y2 k) (h5: midpoint_condition x1 x2):
  length_AB x1 x2 y1 y2 = 6 := 
sorry

end part1_part2_l220_220401


namespace measure_of_minor_arc_XB_l220_220852

theorem measure_of_minor_arc_XB (O : Type*) [Circle O] (X Y B : O) 
  (angle_XBY : MeasureAngle X B Y = 30)
  (arc_XBY : MeasureArc X B Y = 220) : 
  MeasureArc X B = 160 := by
sorry

end measure_of_minor_arc_XB_l220_220852


namespace problem_l220_220429

theorem problem (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x ^ 3 / y ^ 2) + (y ^ 3 / x ^ 2) + y = 440 := by
  sorry

end problem_l220_220429


namespace sum_of_first_15_odd_positive_integers_l220_220677

theorem sum_of_first_15_odd_positive_integers :
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  S_n = 225 :=
by
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  have : S_n = 225 := sorry
  exact this

end sum_of_first_15_odd_positive_integers_l220_220677


namespace calc_expr_l220_220680

theorem calc_expr : abs (-3) + real.sqrt 4 + (-2) * 1 = 3 :=
by
  sorry

end calc_expr_l220_220680


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220810

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220810


namespace project_completion_l220_220191

theorem project_completion (x : ℝ) :
  let A_rate := (1 / 12 : ℝ)
  let B_rate := (1 / 16 : ℝ)
  let A_days_alone := 5 : ℝ
  (x * B_rate + (A_days_alone + x) * A_rate = 1) ↔ (x / 16 + (5 + x) / 12 = 1) :=
by
  sorry

end project_completion_l220_220191


namespace x_power_1994_condition_l220_220009
noncomputable theory

theorem x_power_1994_condition (x : ℂ) (h : x + 1/x = -1) : x^1994 + 1/x^1994 = -1 :=
sorry

end x_power_1994_condition_l220_220009


namespace count_positive_whole_numbers_with_cube_root_less_than_12_l220_220418

theorem count_positive_whole_numbers_with_cube_root_less_than_12 :
  {n : ℕ | ∃ (k : ℕ), n = k ∧ k < 12 ∧ 1 ≤ k}.card = 1727 := sorry

end count_positive_whole_numbers_with_cube_root_less_than_12_l220_220418


namespace least_three_digit_with_factors_2_3_5_7_l220_220620

theorem least_three_digit_with_factors_2_3_5_7 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ [2, 3, 5, 7], d ∣ n) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < n → ∀ d ∈ [2, 3, 5, 7], ¬ d ∣ m :=
  ⟨210, by
     apply And.intro _,
     apply And.intro _,
     apply And.intro _,
     iterate 4 { split },
sorry⟩

end least_three_digit_with_factors_2_3_5_7_l220_220620


namespace count_three_digit_integers_with_7_units_place_div_by_21_l220_220800

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l220_220800


namespace percentage_of_females_l220_220095

def number_of_turtles : ℕ := 100
def percentage_babies_of_m_4 : Rat := 0.40
def number_of_babies : ℕ := 4

theorem percentage_of_females (total_turtles : ℕ) (percentage_babies : Rat) (no_of_babies : ℕ) :
    total_turtles = 100 ∧ percentage_babies = 0.40 ∧ no_of_babies = 4 → 
    let M := no_of_babies / (percentage_babies * (1/4 : Rat)) in
    let F := total_turtles - M in
    (F / total_turtles) * 100 = 60 :=
begin
  sorry
end

end percentage_of_females_l220_220095


namespace correct_transformation_l220_220641

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l220_220641


namespace leak_empty_tank_time_l220_220038

-- Definitions based on given conditions
def rate_A := 1 / 2 -- Rate of Pipe A (1 tank per 2 hours)
def rate_A_plus_L := 2 / 5 -- Combined rate of Pipe A and leak

-- Theorem states the time leak takes to empty full tank is 10 hours
theorem leak_empty_tank_time : 1 / (rate_A - rate_A_plus_L) = 10 :=
by
  -- Proof steps would go here
  sorry

end leak_empty_tank_time_l220_220038


namespace nested_radicals_simplification_l220_220541

noncomputable def simplify_nested_radicals : Prop :=
  sqrt (8 + 4 * sqrt 3) + sqrt (8 - 4 * sqrt 3) = 2 * sqrt 6

theorem nested_radicals_simplification : simplify_nested_radicals := by
  sorry

end nested_radicals_simplification_l220_220541


namespace sequence_tuple_l220_220472

/-- Prove the unique solution to the system of equations derived from the sequence pattern. -/
theorem sequence_tuple (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 7) : (x, y) = (8, 1) :=
by
  sorry

end sequence_tuple_l220_220472


namespace probability_of_three_correct_packages_l220_220301

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220301


namespace probability_three_correct_out_of_five_l220_220284

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220284


namespace least_three_digit_divisible_by_2_3_5_7_l220_220622

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end least_three_digit_divisible_by_2_3_5_7_l220_220622


namespace general_formula_a_sum_b_l220_220369

noncomputable theory

open Nat

def sequence_a : ℕ → ℕ
| 0       := 4
| (n + 1) := sequence_a n + 2

def sequence_b (n : ℕ) : ℤ :=
  (Int.ofNat (2 ^ n)) - (3 * (Int.ofNat n))

-- Here we use absolute value function from Mathlib and summation
def abs_sum : ℕ → ℤ
| 0       := Int.natAbs (sequence_b 0)
| (n + 1) := abs_sum n + Int.natAbs (sequence_b (n + 1))

-- Now we write the theorem statements based on these functions above
theorem general_formula_a (n : ℕ) : sequence_a n = 2 * n + 4 := sorry

theorem sum_b : abs_sum 9 = 1889 := sorry

end general_formula_a_sum_b_l220_220369


namespace double_neg_eq_pos_l220_220143

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l220_220143


namespace board_problem_l220_220477

theorem board_problem :
  let initial := 44
  let n := 4^30
  (exists (seq : ℕ → ℕ → ℕ), seq 0 initial = initial ∧ 
  (∀ k a, seq (k+1) a = 
    let (a1, a2, a3, a4) := four_distinct_integers a in -- A function that ensures four distinct integers summing 4a
    (∀ k, ∃ a1 a2 a3 a4, a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4 ∧ (a1 + a2 + a3 + a4) / 4 = a)) ∧
    (∀ m < 30, 
    (seq 30 a) = (b1 b2 b3 ... b n) →
    (seq m a).seq (m.last).sum_sq / n ≥ 2011) :=
sorry

-- Helper definitions go here.

end board_problem_l220_220477


namespace initial_percentage_decrease_l220_220967

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₁ : P > 0) (h₂ : 1.55 * (1 - x / 100) = 1.24) :
    x = 20 :=
by
  sorry

end initial_percentage_decrease_l220_220967


namespace problem_statement_l220_220065

theorem problem_statement
  (m n : ℕ)
  (a b : ℕ → ℕ)
  (h2550 : 2550 = (List.ofFn (λ i, if i < m then a i else b (i - m)).factorial.prod) / (List.ofFn (λ i, if i < n then b i else a (i - n)).factorial.prod))
  (ha_sorted : ∀ i j, i < j → j < m → a i ≥ a j)
  (hb_sorted : ∀ i j, i < j → j < n → b i ≥ b j)
  (ha : ∀ i, i < m → a i > 0)
  (hb : ∀ i, i < n → b i > 0)
  (ha1_plus_b1_min : ∀ x y, 2550 = (x.factorial * ((List.ofFn (λ i, if i = 1 then x else a (i - 2)).factorial.prod)))
                           / (y.factorial * ((List.ofFn (lambda j, if j = 1 then y else b (j - 2)).factorial.prod))))
                           → x + y ≥ a 0 + b 0) :
  |a 0 - b 0| = 4 :=
sorry

end problem_statement_l220_220065


namespace identical_dice_probability_l220_220098

def num_ways_to_paint_die : ℕ := 3^6

def total_ways_to_paint_dice (n : ℕ) : ℕ := (num_ways_to_paint_die ^ n)

def count_identical_ways : ℕ := 1 + 324 + 8100

def probability_identical_dice : ℚ :=
  (count_identical_ways : ℚ) / (total_ways_to_paint_dice 2 : ℚ)

theorem identical_dice_probability : probability_identical_dice = 8425 / 531441 := by
  sorry

end identical_dice_probability_l220_220098


namespace area_of_triangle_l220_220666

theorem area_of_triangle :
  let line_eq := λ (x y : ℝ), 2 * x + y = 6
  let x_intercept := (3, 0)  -- The x-intercept occurs when y = 0
  let y_intercept := (0, 6)  -- The y-intercept occurs when x = 0
  ∃ (area : ℝ), area = 9 :=
by
  sorry

end area_of_triangle_l220_220666


namespace swimming_speed_l220_220658

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 14) 
  (h3 : time = 3.5) 
  (h4 : distance = (v - water_speed) * time) : 
  v = 6 := 
by
  sorry

end swimming_speed_l220_220658


namespace norris_money_left_l220_220916

theorem norris_money_left :
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  total_savings - amount_spent = 10 :=
by
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  have h1 : total_savings = 85 := by rfl
  have h2 : total_savings - amount_spent = 85 - amount_spent := by rw h1
  have h3 : 85 - amount_spent = 85 - 75 := rfl
  have h4 : 85 - 75 = 10 := rfl
  exact eq.trans (eq.trans h2 h3) h4

end norris_money_left_l220_220916


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220809

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l220_220809


namespace count_three_digit_integers_end7_divby21_l220_220792

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l220_220792


namespace probability_three_correct_packages_l220_220306

def derangement_count (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

theorem probability_three_correct_packages 
  (total_packages : ℕ)
  (correct_packages : ℕ)
  (total_arrangements : ℕ)
  (correct_choice_count : ℕ)
  (probability : ℚ) :
  total_packages = 5 →
  correct_packages = 3 →
  total_arrangements = total_packages.factorial →
  correct_choice_count = nat.choose total_packages correct_packages →
  probability = (correct_choice_count * derangement_count (total_packages - correct_packages)) / total_arrangements →
  probability = 1 / 12 :=
by
  intros hp tp ta cc p
  rw [hp, tp, ta, cc] at *
  sorry

end probability_three_correct_packages_l220_220306


namespace digit_equation_l220_220239

-- Definitions for digits and the equation components
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x ≤ 9

def three_digit_number (A B C : ℤ) : ℤ := 100 * A + 10 * B + C
def two_digit_number (A D : ℤ) : ℤ := 10 * A + D
def four_digit_number (A D C : ℤ) : ℤ := 1000 * A + 100 * D + 10 * D + C

-- Statement of the theorem
theorem digit_equation (A B C D : ℤ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) :
  three_digit_number A B C * two_digit_number A D = four_digit_number A D C :=
sorry

end digit_equation_l220_220239


namespace problem1_problem2_problem3_l220_220776

-- Conditions
def a_seq : ℕ → ℝ 
| 0       => 1/2
| (n + 1) => (a_seq n ^ 2) / 2016 + a_seq n

-- 1. Prove a_{n+1} > a_n
theorem problem1 (n : ℕ) : a_seq (n + 1) > a_seq n := 
sorry

-- 2. Prove a_{2017} < 1
theorem problem2 : a_seq 2017 < 1 := 
sorry

-- 3. Find the minimum value of k such that a_k > 1
theorem problem3 : ∃ k : ℕ, k > 0 ∧ a_seq k > 1 ∧ ∀ m > 0, m < k → a_seq m ≤ 1 :=
⟨2018, by exact dec_trivial; sorry; sorry⟩

end problem1_problem2_problem3_l220_220776


namespace boat_speed_in_still_water_l220_220178

variable (Vb Vs tu td D : ℝ)

-- Conditions
axiom condition1 : Vs = 8
axiom condition2 : tu = 2 * td
axiom condition3 : (Vb - Vs) * tu = (Vb + Vs) * td
-- Distance covered upstream and downstream are the same.

theorem boat_speed_in_still_water : Vb = 24 := by
  -- introducing the conditions
  have h1 : Vs = 8 := condition1
  have h2 : tu = 2 * td := condition2
  have h3 : (Vb - Vs) * tu = (Vb + Vs) * td := condition3
  -- the proof is omitted with sorry
  sorry

end boat_speed_in_still_water_l220_220178


namespace probability_exactly_three_correct_l220_220347

/-- There are five packages and five houses. The probability that exactly three packages are delivered to the correct houses is 1/12. -/
theorem probability_exactly_three_correct : ∀ (n : ℕ), n = 5 → (num_exactly_three_correct n) = 1 / 12 :=
by
  intros n h
  sorry

/-- Helper function to determine the numerator of the probability where exactly three out of five packages are delivered to the correct houses. -/
def num_exactly_three_correct (n : ℕ) : ℝ := 
if n = 5 then (10 * (1/60) * (1/2))
else 0

end probability_exactly_three_correct_l220_220347


namespace norris_money_left_l220_220918

theorem norris_money_left :
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  total_savings - amount_spent = 10 :=
by
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  have h1 : total_savings = 85 := by rfl
  have h2 : total_savings - amount_spent = 85 - amount_spent := by rw h1
  have h3 : 85 - amount_spent = 85 - 75 := rfl
  have h4 : 85 - 75 = 10 := rfl
  exact eq.trans (eq.trans h2 h3) h4

end norris_money_left_l220_220918


namespace karthik_weight_average_l220_220006

theorem karthik_weight_average
  (weight : ℝ)
  (hKarthik: 55 < weight )
  (hBrother: weight < 58 )
  (hFather : 56 < weight )
  (hSister: 54 < weight ∧ weight < 57) :
  (56 < weight ∧ weight < 57) → (weight = 56.5) :=
by 
  sorry

end karthik_weight_average_l220_220006


namespace possible_last_two_digits_of_x_2009_l220_220235

structure Sequence {α : Type*} (n : ℕ) :=
  (x : Fin (n + 1) → α)
  (initial : x ⟨0, Nat.zero_lt_succ n⟩ ∈ {5, 7})
  (recurrence : ∀ k (hk : k < n), 
    x ⟨k+1, by linarith⟩ ∈ {5^(x ⟨k, hk⟩), 7^(x ⟨k, hk⟩)})

-- Define the sequence x_n
noncomputable def x : Sequence 2009 :=
{
  x := sorry,
  initial := sorry,
  recurrence := sorry,
}

-- Theorem to prove the possible last two digits for x_2009
theorem possible_last_two_digits_of_x_2009 : 
  ∃ m ∈ {25, 7, 43}, (x.x ⟨2009, sorry⟩ % 100 = m) :=
  sorry

end possible_last_two_digits_of_x_2009_l220_220235


namespace factor_expression_l220_220228

variable (x : ℝ)

def e : ℝ := (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5)

theorem factor_expression : e x = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by
  sorry

end factor_expression_l220_220228


namespace double_neg_eq_pos_l220_220146

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l220_220146


namespace solution_set_of_inequality_l220_220233

def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x > 0, x^2 * f_prime x + 1 > 0) → 
  f 1 = 5 →
  { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x ∧ f x < 1 / x + 4 } :=
by 
  intros h1 h2 
  sorry

end solution_set_of_inequality_l220_220233


namespace probability_of_three_correct_packages_l220_220298

noncomputable def probability_three_correct_deliveries : ℚ :=
  let num_ways_to_choose_correct := Nat.choose 5 3 in
  let prob_per_arrangement := (1 / 5) * (1 / 4) * (1 / 3) * (1 / 2) in
  let total_prob := num_ways_to_choose_correct * prob_per_arrangement in
  total_prob

theorem probability_of_three_correct_packages :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end probability_of_three_correct_packages_l220_220298


namespace constant_term_is_60_l220_220870

-- Define the binomial coefficient
noncomputable def binom (n k : ℕ) := Nat.choose n k

-- Define the general term for the expansion of (x + 1/x)^6
def general_term (r : ℕ) := binom 6 r * (3 : ℤ)^(6 - 2 * r)

-- Calculate the constant term for (3 + x)(x + 1/x)^6
def constant_term := 3 * general_term 3

-- Prove the constant term is 60
theorem constant_term_is_60 : constant_term = 60 := by
  sorry

end constant_term_is_60_l220_220870


namespace s_at_5_l220_220190

noncomputable def q : ℚ[X] := sorry -- We don't need the exact polynomial, just its properties.
noncomputable def s : ℚ[X] := (λ x, x^2 + (4/5)*x + (3/5))

lemma q_values_at_points :
  q.eval 1 = 2 ∧ q.eval 3 = 5 ∧ q.eval (-2) = -3 :=
begin
  sorry
end

lemma s_is_remainder :
  ∃ (k : ℚ[X]), q = (X - 1) * (X - 3) * (X + 2) * k + s :=
begin
  sorry
end

theorem s_at_5 : s.eval 5 = 29.6 :=
by {
  -- Direct application
  simp [s],
  norm_num,
}

end s_at_5_l220_220190


namespace village_foods_sales_l220_220100

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l220_220100


namespace intersection_of_function_and_inverse_l220_220958

theorem intersection_of_function_and_inverse (m : ℝ) :
  (∀ x y : ℝ, y = Real.sqrt (x - m) ↔ x = y^2 + m) →
  (∃ x : ℝ, Real.sqrt (x - m) = x) ↔ (m ≤ 1 / 4) :=
by
  sorry

end intersection_of_function_and_inverse_l220_220958


namespace find_a_l220_220382

variable {x n : ℝ}

theorem find_a (hx : x > 0) (hn : n > 0) :
    (∀ n > 0, x + n^n / x^n ≥ n + 1) ↔ (∀ n > 0, a = n^n) :=
sorry

end find_a_l220_220382


namespace norris_money_left_l220_220919

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l220_220919


namespace convert_kahs_to_zahs_l220_220428

def zahs := ℕ
def tols := ℕ
def kahs := ℕ

variable (zah_to_tol : ζ → ζ)   -- 15 zahs are equal to 24 tols
variable (tol_to_kah : τ → κ)   -- 9 tols are equal to 15 kahs

theorem convert_kahs_to_zahs (zah_to_tol_tol : zah_to_tol 15 = 24)
                              (tol_to_kah_kah : tol_to_kah 9 = 15)
                              (n : ζ) : zah_to_tol (tol_to_kah 2000) = 750 :=
by 
sorry

end convert_kahs_to_zahs_l220_220428


namespace distance_from_origin_to_point_P_l220_220451

def origin : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (12, -5)

theorem distance_from_origin_to_point_P :
  (Real.sqrt ((12 - 0)^2 + (-5 - 0)^2)) = 13 := by
  simp [Real.sqrt, *]
  sorry

end distance_from_origin_to_point_P_l220_220451


namespace complex_number_m_is_2_l220_220562

noncomputable def is_real (z : ℂ) : Prop :=
  z.im = 0

theorem complex_number_m_is_2 (m : ℝ) (h : is_real ((2 + m * complex.I) / (1 + complex.I))) : m = 2 :=
  sorry

end complex_number_m_is_2_l220_220562


namespace probability_three_correct_deliveries_l220_220333

theorem probability_three_correct_deliveries (h : ∀ i, i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) :
  (∃ (h₁ h₂ : Set ℕ), h₁ ∩ h₂ = ∅ ∧ h₁ ∪ h₂ = {i | i ∈ ({1, 2, 3, 4, 5} : Set ℕ)} ∧
  |h₁| = 3 ∧ |h₂| = 2 ∧
  (∀ x y ∈ h₂, x ≠ y → x ≠ y.swap) ∧
  (choose 5 3) * (1/5) * (1/4) * (1/3) * (1/2) = 1/12) :=
sorry

end probability_three_correct_deliveries_l220_220333


namespace female_students_in_first_class_l220_220601

theorem female_students_in_first_class
  (females_in_second_class : ℕ)
  (males_in_first_class : ℕ)
  (males_in_second_class : ℕ)
  (males_in_third_class : ℕ)
  (females_in_third_class : ℕ)
  (extra_students : ℕ)
  (total_students_need_partners : ℕ)
  (total_males : ℕ := males_in_first_class + males_in_second_class + males_in_third_class)
  (total_females : ℕ := females_in_second_class + females_in_third_class)
  (females_in_first_class : ℕ)
  (females : ℕ := females_in_first_class + total_females) :
  (females_in_second_class = 18) →
  (males_in_first_class = 17) →
  (males_in_second_class = 14) →
  (males_in_third_class = 15) →
  (females_in_third_class = 17) →
  (extra_students = 2) →
  (total_students_need_partners = total_males - extra_students) →
  females = total_students_need_partners →
  females_in_first_class = 9 :=
by
  intros
  sorry

end female_students_in_first_class_l220_220601


namespace empty_set_subset_zero_set_l220_220623

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end empty_set_subset_zero_set_l220_220623


namespace line_equation_l220_220176

noncomputable section

def point (P : ℝ × ℝ) := P = (1, 4)

def positive_intercepts (L : ℝ → ℝ) : Prop :=
  ∃ m b, L = λ x, m * x + b ∧  m < 0 ∧ b > 0 ∧ m * (1:ℝ) + b = 4

def min_sum_of_intercepts (L : ℝ → ℝ) : Prop :=
  ∃ m b, L = λ x, m * x + b ∧ (b + - b / m) = 0

theorem line_equation (L : ℝ → ℝ) :
  (∀ P, point P → L (P.1) = P.2) →
  positive_intercepts L →
  min_sum_of_intercepts L →
  ∃ A B C, A * P.1 + B * P.2 + C = 0 ∧ (A, B, C) ≠ (0, 0, 0) :=
begin
  sorry
end

end line_equation_l220_220176


namespace part_a_part_b_l220_220984

-- Define the number of lamps and the flip size
def num_lamps : ℕ := 111
def flip_size : ℕ := 13

-- Part (a): Always possible to turn off all the lamps
theorem part_a (initial_state : Fin num_lamps → Bool) : 
  ∃ flips : Nat, flips > 0 ∧ (∀ flip_set : Fin flip_size → Fin num_lamps, initial_state (flip_set) = !initial_state (flip_set)) → ∀ (i : Fin num_lamps), initial_state i = false :=
sorry

-- Part (b): Minimum flips required if all lamps are initially on
theorem part_b : 
  ∀ initial_state : Fin num_lamps → Bool, (∀ i : Fin num_lamps, initial_state i = true) → 
  ∃ flips : Nat, flips = 9 ∧ (∀ flip_set : Fin flip_size → Fin num_lamps, initial_state (flip_set) = !initial_state (flip_set)) → ∀ (i : Fin num_lamps), initial_state i = false :=
sorry

end part_a_part_b_l220_220984


namespace double_neg_eq_pos_l220_220145

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l220_220145


namespace even_function_A_value_l220_220435

-- Given function definition
def f (x : ℝ) (A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement to prove
theorem even_function_A_value (A : ℝ) (h : ∀ x : ℝ, f x A = f (-x) A) : A = 1 :=
by
  sorry

end even_function_A_value_l220_220435


namespace probability_three_correct_deliveries_is_one_sixth_l220_220324

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- define the derangement for 2 items
def derangement_2 : ℕ := 1

def prob_exactly_three_correct (total_packages : ℕ) (correct_deliveries : ℕ) : ℚ :=
  choose total_packages correct_deliveries * (derangement_2 : ℚ) / (factorial total_packages : ℚ)

theorem probability_three_correct_deliveries_is_one_sixth :
  prob_exactly_three_correct 5 3 = 1 / 6 :=
by
  sorry

end probability_three_correct_deliveries_is_one_sixth_l220_220324


namespace minimum_w_coincide_after_translation_l220_220366

noncomputable def period_of_cosine (w : ℝ) : ℝ := (2 * Real.pi) / w

theorem minimum_w_coincide_after_translation
  (w : ℝ) (h_w_pos : 0 < w) :
  period_of_cosine w = (4 * Real.pi) / 3 → w = 3 / 2 :=
by
  sorry

end minimum_w_coincide_after_translation_l220_220366


namespace speed_of_man_correct_l220_220207

noncomputable def speed_of_man_in_kmph (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := (train_length_m / time_pass_sec)
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man_correct : 
  speed_of_man_in_kmph 77.993280537557 140 6 = 6.00871946444388 := 
by simp [speed_of_man_in_kmph]; sorry

end speed_of_man_correct_l220_220207


namespace norris_money_left_l220_220924

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l220_220924


namespace units_digit_base8_l220_220182

theorem units_digit_base8 (a b : ℕ) (h1 : a = 198 - 3) (h2 : b = 53 + 7)
  : ((a * b) % 8) = 4 :=
by
  -- Convert the conditions to actual values
  have h_a : a = 195, by { rw h1, refl },
  have h_b : b = 60, by { rw h2, refl },
  -- Calculate the product
  let product := a * b,
  have h_product : product = 195 * 60, by rwa [h_a, h_b],
  -- Calculate the remainder when product is divided by 8
  have h_remainder : product % 8 = 11700 % 8, by rwa h_product,
  calc 
    product % 8
        = 11700 % 8 : by assumption
    ... = 4 : sorry -- Calculation of the remainder

end units_digit_base8_l220_220182


namespace remainder_prod_mod_7_l220_220711

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l220_220711


namespace equivalent_proof_problem_l220_220778

-- Definitions
def p := ∀ x : ℝ, 2^x + 2^(-x) ≥ 2
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → f 0 = 0

-- Theorem to be proved
theorem equivalent_proof_problem : p ∧ ¬ q := by
  unfold p q
  sorry

end equivalent_proof_problem_l220_220778


namespace count_valid_n_values_l220_220744

-- Define the problem conditions
def holds_for_all_real_t (n : ℕ) : Prop := 
  ∀ (t : ℝ), (sin t + complex.I * cos t)^n = sin (n * t) + complex.I * cos (n * t)

-- Define the statement to prove
theorem count_valid_n_values :
  { n : ℕ | n ≤ 500 ∧ holds_for_all_real_t n }.to_finset.card = 125 :=
sorry

end count_valid_n_values_l220_220744


namespace vasya_password_combinations_l220_220999

theorem vasya_password_combinations : 
  let D := {0, 1, 3, 4, 5, 6, 7, 8, 9} in
  let valid_passwords := {p : Finset (Fin 10) | 
                             p.val < 10000 ∧ 
                             ∀ i, p.val.digits 10 (i) ∉ {2} ∧ 
                             (∀ i < 3, p.val.digits 10 i ≠ p.val.digits 10 (i + 1)) ∧ 
                             p.val.digits 10 0 = p.val.digits 10 3
  } in
  valid_passwords.card = 504 :=
by
  sorry

end vasya_password_combinations_l220_220999


namespace iterative_average_difference_l220_220215

def iterative_average (lst : List ℚ) : ℚ :=
  match lst with
  | []      => 0
  | [x]     => x
  | x::y::l => iterative_average ((x + y)/2 :: l)

theorem iterative_average_difference :
  let numbers := [1, 2, 3, 4, 5].map (λ x => (x : ℚ)) in
  (iterative_average [5, 4, 3, 2, 1].map (λ x => (x : ℚ)) -
   iterative_average [1, 2, 3, 4, 5].map (λ x => (x : ℚ))) = 17/8 := 
  sorry

end iterative_average_difference_l220_220215


namespace exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l220_220699

theorem exists_half_perimeter_area_rectangle_6x1 :
  ∃ x₁ x₂ : ℝ, (6 * 1 / 2 = (6 + 1) / 2) ∧
                x₁ * x₂ = 3 ∧
                (x₁ + x₂ = 3.5) ∧
                (x₁ = 2 ∨ x₁ = 1.5) ∧
                (x₂ = 2 ∨ x₂ = 1.5)
:= by
  sorry

theorem not_exists_half_perimeter_area_rectangle_2x1 :
  ¬(∃ x : ℝ, x * (1.5 - x) = 1)
:= by
  sorry

end exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l220_220699


namespace sum_of_perpendicular_distances_constant_l220_220044

theorem sum_of_perpendicular_distances_constant (a h₁ h₂ : ℝ) 
  (D : Type) 
  (h₁_eq : ∀ (D : D), h₁ = (fun x y => x)(a, D)) 
  (h₂_eq : ∀ (D : D), h₂ = (fun x y => x)(a, D)) : 
  h₁ + h₂ = (a * Real.sqrt 3) / 2 :=
by
  sorry

end sum_of_perpendicular_distances_constant_l220_220044


namespace john_time_spent_on_pictures_l220_220001

noncomputable def time_spent (num_pictures : ℕ) (draw_time color_time : ℝ) : ℝ := 
  (num_pictures * draw_time) + (num_pictures * color_time)

theorem john_time_spent_on_pictures :
  ∀ (num_pictures : ℕ) (draw_time : ℝ) (color_percentage_less : ℝ)
    (color_time : ℝ),
    num_pictures = 10 →
    draw_time = 2 →
    color_percentage_less = 0.30 →
    color_time = draw_time - (color_percentage_less * draw_time) →
    time_spent num_pictures draw_time color_time = 34 :=
begin
  sorry,
end

end john_time_spent_on_pictures_l220_220001


namespace distance_centroid_circumcenter_l220_220775

-- Define the given parameters and their types
variables (S M : Type) [point : S] [point : M]
variables (R : ℝ) (m_a m_b m_c : ℝ)

-- Define the theorem stating the equivalence
theorem distance_centroid_circumcenter 
  (S M : Type) [point : S] [point : M] 
  (R : ℝ) (m_a m_b m_c : ℝ) : 
  let SM : ℝ := real.sqrt (R^2 - (4 / 27) * (m_a^2 + m_b^2 + m_c^2)) 
  in SM = real.sqrt (R^2 - (4 / 27) * (m_a^2 + m_b^2 + m_c^2)) :=
sorry

end distance_centroid_circumcenter_l220_220775


namespace geometric_sequence_term_l220_220871

theorem geometric_sequence_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 3^n - 1) →
  (a_n n = S_n n - S_n (n - 1)) →
  (a_n n = 2 * 3^(n - 1)) :=
by
  intros h1 h2
  sorry

end geometric_sequence_term_l220_220871


namespace root_cubic_of_quadratic_l220_220900

variable (a b c d x₁ x₂ : ℝ)

-- Define the conditions:
def quadratic_roots : Prop :=
  ∀ (x : ℝ), x^2 - (a + d) * x + (ad - bc) = 0 → x = x₁ ∨ x = x₂

-- Define the target statement to prove:
def target_equation (y : ℝ) : ℝ :=
  y^2 - (a^3 + d^3 + 3 * a * b * c + 3 * b * c * d) * y + (ad - bc)^3

-- The main proof statement:
theorem root_cubic_of_quadratic :
  quadratic_roots a b c d x₁ x₂ →
  ∀ (y : ℝ), target_equation a b c d y = 0 →
  (y = x₁^3 ∨ y = x₂^3) := by
  intros h_quad h_target
  sorry

end root_cubic_of_quadratic_l220_220900


namespace rectangle_circle_area_ratio_l220_220069

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end rectangle_circle_area_ratio_l220_220069


namespace smallest_moves_to_all_heads_up_l220_220947

/-- Initial configuration: Vera has a row of 2023 coins, alternately tails-up and heads-up,
with the leftmost coin tails-up.

Move rules:
1. On the first move, Vera may flip over any of the 2023 coins.
2. On all subsequent moves, Vera may only flip over a coin adjacent to the one she flipped on the previous move.

Question: Determine the smallest possible number of moves Vera can make to reach a state
in which every coin is heads-up. -/
def initial_state (n : ℕ) := (fin n → bool) := λ i, i % 2 = 0

def is_adjacent (a b : ℕ) : Prop := (a = b + 1) ∨ (a + 1 = b)

theorem smallest_moves_to_all_heads_up (n : ℕ) (initial : fin n → bool) :
  n = 2023 →
  initial_state n initial →
  ∃ (moves : list ℕ), (∀ i ∈ moves, 0 ≤ i ∧ i < n) ∧
                        (∀ k, k ∈ moves.tail → is_adjacent (moves.nth k) (moves.nth (k - 1))) ∧
                        (∀ i, initial (i) = tt) ∧
                        list.length moves = 4044 := sorry

end smallest_moves_to_all_heads_up_l220_220947


namespace probability_of_exactly_three_correct_packages_l220_220293

noncomputable def probability_exactly_three_correct : ℚ :=
  let total_permutations := (5!).to_rat
  let choose_3 := nat.choose 5 3
  let derangement_2 := 1 / (2!).to_rat
  (choose_3 * derangement_2) / total_permutations

theorem probability_of_exactly_three_correct_packages : 
  probability_exactly_three_correct = 1 / 12 := by
  sorry

end probability_of_exactly_three_correct_packages_l220_220293


namespace product_mod_seven_l220_220708

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l220_220708


namespace jovana_initial_shells_l220_220884

theorem jovana_initial_shells (x : ℕ) (h₁ : x + 12 = 17) : x = 5 :=
by
  -- Proof omitted
  sorry

end jovana_initial_shells_l220_220884


namespace insulation_cost_l220_220627

def rectangular_prism_surface_area (l w h : ℕ) : ℕ :=
2 * l * w + 2 * l * h + 2 * w * h

theorem insulation_cost
  (l w h : ℕ) (cost_per_square_foot : ℕ)
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost : cost_per_square_foot = 20) :
  rectangular_prism_surface_area l w h * cost_per_square_foot = 1440 := 
sorry

end insulation_cost_l220_220627


namespace find_CD_l220_220650

variable (A B C D : Point)
variable (h d : ℝ)
variable [inhabited Point]

-- Given conditions
def angle_eq : angle A C B = angle A D B := sorry
def perp_to_plane : perp D C (plane A B C) := sorry
def height_h : height A B C = h := sorry
def distance_d : distance circumcenter (side A B) = d := sorry

-- To prove
theorem find_CD (h d : ℝ) : CD = 2 * sqrt(d * (d - h)) := sorry

end find_CD_l220_220650


namespace walter_time_spent_on_seals_l220_220613

noncomputable def time_spent_at_zoo : ℕ := 185

variables (S : ℕ)

def time_spent_looking_at_seals := S
def time_spent_looking_at_penguins := 8 * S
def time_spent_looking_at_elephants := 13
def time_spent_looking_at_giraffes := S / 2

axiom total_time_spent_at_zoo (S : ℕ) :
  time_spent_looking_at_seals S + 
  time_spent_looking_at_penguins S + 
  time_spent_looking_at_elephants + 
  time_spent_looking_at_giraffes S = time_spent_at_zoo

theorem walter_time_spent_on_seals (S : ℕ) (h : total_time_spent_at_zoo S) :
  S = 16 :=
by sorry

end walter_time_spent_on_seals_l220_220613


namespace geometric_progression_l220_220549

-- Definitions and conditions from the problem

variable (a b c : ℝ)
variable (ABC_non_isosceles : ¬(a = b ∨ b = c ∨ c = a))
variable (circle_tangent_to_side : ∀ D E F.  (Circle DEF ∧ Tangent Circle DEF Side))

-- Lean statement (theorem) to prove the given problem
theorem geometric_progression :
  (a^2 + b^2) * (a^2 + c^2) = (b^2 + c^2)^2 :=
by
  sorry

end geometric_progression_l220_220549


namespace minimum_measurements_required_l220_220738

noncomputable def error_in_distribution (n : ℕ) : ℝ :=
  4 / n

noncomputable def probability_condition : ℝ :=
  0.683

theorem minimum_measurements_required (n : ℕ) : 0 < n → 
  (∀ ε_n : ℝ, ε_n ~ N(0, error_in_distribution n) → 
  (∀ X : ℝ, X ~ N(0, error_in_distribution n) → 
    P(0 - sqrt(error_in_distribution n) < X ∧ X < 0 + sqrt(error_in_distribution n)) = probability_condition)) → 
  n ≥ 16 :=
by
  sorry

end minimum_measurements_required_l220_220738


namespace hall_height_is_correct_l220_220855

noncomputable def height_of_hall (length width cost_per_sqm total_expenditure : ℝ) : ℝ :=
  let total_area := total_expenditure / cost_per_sqm in
  let floor_area := length * width in
  let area_for_walls := total_area - floor_area in
  let perimeter := 2 * (length + width) in
  area_for_walls / perimeter

theorem hall_height_is_correct :
  height_of_hall 20 15 40 38000 ≈ 9.2857 :=
by
  sorry

end hall_height_is_correct_l220_220855


namespace rachel_baked_brownies_l220_220533

theorem rachel_baked_brownies (b : ℕ) (h : 3 * b / 5 = 18) : b = 30 :=
by
  sorry

end rachel_baked_brownies_l220_220533


namespace probability_three_correct_out_of_five_l220_220279

-- Define the number of packages and houses
def n : ℕ := 5
def k : ℕ := 3

-- Define the number of combinations for choosing k correct houses out of n
def combinations : ℕ := Nat.choose n k

-- Define the number of derangements for the remaining n-k items
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total permutations of n items
def total_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Probability calculation
def probability (n : ℕ) (k : ℕ) : ℚ :=
  (combinations * derangements (n - k)) / total_permutations n

-- Main theorem statement
theorem probability_three_correct_out_of_five :
  probability n k = 1 / 12 :=
by
  sorry

end probability_three_correct_out_of_five_l220_220279
