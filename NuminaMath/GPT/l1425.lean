import Mathlib

namespace tank_capacities_l1425_142500

theorem tank_capacities (x y z : ℕ) 
  (h1 : x + y + z = 1620)
  (h2 : z = x + y / 5) 
  (h3 : z = y + x / 3) :
  x = 540 ∧ y = 450 ∧ z = 630 := 
by 
  sorry

end tank_capacities_l1425_142500


namespace repeating_decimal_as_fraction_l1425_142541

theorem repeating_decimal_as_fraction :
  ∃ x : ℚ, x = 6 / 10 + 7 / 90 ∧ x = 61 / 90 :=
by
  sorry

end repeating_decimal_as_fraction_l1425_142541


namespace range_of_a_l1425_142587

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 16 * Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, a - 1 ≤ x ∧ x ≤ a + 2 → (fderiv ℝ f x) x < 0)
  ↔ (1 < a) ∧ (a ≤ 2) :=
by
  sorry

end range_of_a_l1425_142587


namespace range_of_a_l1425_142556

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1/2 * x^2

theorem range_of_a (a : ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a (x1 + a) - f a (x2 + a)) / (x1 - x2) ≥ 3) :
  a ≥ 9 / 4 :=
sorry

end range_of_a_l1425_142556


namespace sum_of_squares_of_consecutive_integers_l1425_142572

-- The sum of the squares of three consecutive positive integers equals 770.
-- We aim to prove that the largest integer among them is 17.
theorem sum_of_squares_of_consecutive_integers (n : ℕ) (h_pos : n > 0) 
    (h_sum : (n-1)^2 + n^2 + (n+1)^2 = 770) : n + 1 = 17 :=
sorry

end sum_of_squares_of_consecutive_integers_l1425_142572


namespace problem1_l1425_142599

theorem problem1 (a b : ℝ) : (a - b)^3 + 3 * a * b * (a - b) + b^3 - a^3 = 0 :=
sorry

end problem1_l1425_142599


namespace points_on_quadratic_l1425_142581

theorem points_on_quadratic (c y₁ y₂ : ℝ) 
  (hA : y₁ = (-1)^2 - 6*(-1) + c) 
  (hB : y₂ = 2^2 - 6*2 + c) : y₁ > y₂ := 
  sorry

end points_on_quadratic_l1425_142581


namespace probability_three_primes_l1425_142533

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

end probability_three_primes_l1425_142533


namespace overall_percentage_of_favor_l1425_142547

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

end overall_percentage_of_favor_l1425_142547


namespace find_ratio_l1425_142594

-- Given conditions
variable (x y a b : ℝ)
variable (h1 : 2 * x - y = a)
variable (h2 : 4 * y - 8 * x = b)
variable (h3 : b ≠ 0)

theorem find_ratio (a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 4 := by
  sorry

end find_ratio_l1425_142594


namespace total_dolls_l1425_142589

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end total_dolls_l1425_142589


namespace find_a_for_symmetry_l1425_142514

theorem find_a_for_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, a * Real.sin x + Real.cos (x + π / 6) = 
                    a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x + π / 6)) 
           ↔ a = 2 :=
by
  sorry

end find_a_for_symmetry_l1425_142514


namespace find_a_l1425_142544

theorem find_a (a b : ℝ) (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) (h4 : a + b = 4) : a = -1 :=
by 
sorry

end find_a_l1425_142544


namespace gcd_102_238_l1425_142559

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l1425_142559


namespace stipulated_percentage_l1425_142557

theorem stipulated_percentage
  (A B C : ℝ)
  (P : ℝ)
  (hA : A = 20000)
  (h_range : B - C = 10000)
  (hB : B = A + (P / 100) * A)
  (hC : C = A - (P / 100) * A) :
  P = 25 :=
sorry

end stipulated_percentage_l1425_142557


namespace option_transformations_incorrect_l1425_142534

variable {a b x : ℝ}

theorem option_transformations_incorrect (h : a < b) :
  ¬ (3 - a < 3 - b) := by
  -- Here, we would show the incorrectness of the transformation in Option B
  sorry

end option_transformations_incorrect_l1425_142534


namespace pattern_equation_l1425_142505

theorem pattern_equation (n : ℕ) (hn : n > 0) : n * (n + 2) + 1 = (n + 1) ^ 2 := 
by sorry

end pattern_equation_l1425_142505


namespace max_average_speed_palindromic_journey_l1425_142509

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

end max_average_speed_palindromic_journey_l1425_142509


namespace complete_square_k_value_l1425_142538

theorem complete_square_k_value : 
  ∃ k : ℝ, ∀ x : ℝ, (x^2 - 8*x = (x - 4)^2 + k) ∧ k = -16 :=
by
  use -16
  intro x
  sorry

end complete_square_k_value_l1425_142538


namespace smallest_perfect_square_divisible_by_3_and_5_l1425_142531

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end smallest_perfect_square_divisible_by_3_and_5_l1425_142531


namespace percentage_of_students_enrolled_is_40_l1425_142504

def total_students : ℕ := 880
def not_enrolled_in_biology : ℕ := 528
def enrolled_in_biology : ℕ := total_students - not_enrolled_in_biology
def percentage_enrolled : ℕ := (enrolled_in_biology * 100) / total_students

theorem percentage_of_students_enrolled_is_40 : percentage_enrolled = 40 := by
  -- Beginning of the proof
  sorry

end percentage_of_students_enrolled_is_40_l1425_142504


namespace ratio_of_areas_l1425_142502

theorem ratio_of_areas (aC aD : ℕ) (hC : aC = 48) (hD : aD = 60) : 
  (aC^2 : ℚ) / (aD^2 : ℚ) = (16 : ℚ) / (25 : ℚ) := 
by
  sorry

end ratio_of_areas_l1425_142502


namespace coefficient_c_nonzero_l1425_142583

-- We are going to define the given polynomial and its conditions
def P (x : ℝ) (a b c d e : ℝ) : ℝ :=
  x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Given conditions
def five_x_intercepts (P : ℝ → ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧ P x5 = 0

def double_root_at_zero (P : ℝ → ℝ) : Prop :=
  P 0 = 0 ∧ deriv P 0 = 0

-- Equivalent proof problem
theorem coefficient_c_nonzero (a b c d e : ℝ)
  (h1 : P 0 a b c d e = 0)
  (h2 : deriv (P · a b c d e) 0 = 0)
  (h3 : ∀ x, P x a b c d e = x^2 * (x - 1) * (x - 2) * (x - 3))
  (h4 : ∀ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) : 
  c ≠ 0 := 
sorry

end coefficient_c_nonzero_l1425_142583


namespace correct_calculation_l1425_142591

theorem correct_calculation (x a : Real) :
  (3 * x^2 - x^2 ≠ 3) → 
  (-3 * a^2 - 2 * a^2 ≠ -a^2) →
  (x^3 / x ≠ 3) → 
  ((-x)^3 = -x^3) → 
  true :=
by
  intros _ _ _ _
  trivial

end correct_calculation_l1425_142591


namespace sum_of_solutions_l1425_142550

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l1425_142550


namespace union_eq_universal_set_l1425_142506

-- Define the sets U, M, and N
def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 6}

-- The theorem stating the desired equality
theorem union_eq_universal_set : M ∪ N = U := 
sorry

end union_eq_universal_set_l1425_142506


namespace combined_salaries_of_A_B_C_D_l1425_142540

theorem combined_salaries_of_A_B_C_D (salaryE : ℕ) (avg_salary : ℕ) (num_people : ℕ)
    (h1 : salaryE = 9000) (h2 : avg_salary = 8800) (h3 : num_people = 5) :
    (avg_salary * num_people) - salaryE = 35000 :=
by
  sorry

end combined_salaries_of_A_B_C_D_l1425_142540


namespace percentage_cd_only_l1425_142569

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

end percentage_cd_only_l1425_142569


namespace theta_interval_l1425_142535

noncomputable def f (x θ: ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_interval (θ: ℝ) (k: ℤ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x θ > 0) → 
  (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) := 
by
  sorry

end theta_interval_l1425_142535


namespace total_messages_equation_l1425_142529

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l1425_142529


namespace all_numbers_divisible_by_5_l1425_142508

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

end all_numbers_divisible_by_5_l1425_142508


namespace math_score_computation_l1425_142567

def comprehensive_score 
  (reg_score : ℕ) (mid_score : ℕ) (fin_score : ℕ) 
  (reg_weight : ℕ) (mid_weight : ℕ) (fin_weight : ℕ) 
  : ℕ :=
  (reg_score * reg_weight + mid_score * mid_weight + fin_score * fin_weight) 
  / (reg_weight + mid_weight + fin_weight)

theorem math_score_computation :
  comprehensive_score 80 80 85 3 3 4 = 82 := by
sorry

end math_score_computation_l1425_142567


namespace shortest_distance_parabola_to_line_l1425_142580

open Real

theorem shortest_distance_parabola_to_line :
  ∃ (d : ℝ), 
    (∀ (P : ℝ × ℝ), (P.1 = (P.2^2) / 8) → 
      ((2 * P.1 - P.2 - 4) / sqrt 5 ≥ d)) ∧ 
    (d = 3 * sqrt 5 / 5) :=
sorry

end shortest_distance_parabola_to_line_l1425_142580


namespace find_intersection_l1425_142521

def intersection_point (x y : ℚ) : Prop :=
  3 * x + 4 * y = 12 ∧ 7 * x - 2 * y = 14

theorem find_intersection :
  intersection_point (40 / 17) (21 / 17) :=
by
  sorry

end find_intersection_l1425_142521


namespace find_m_of_perpendicular_vectors_l1425_142503

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l1425_142503


namespace polynomials_with_three_different_roots_count_l1425_142526

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

end polynomials_with_three_different_roots_count_l1425_142526


namespace three_f_x_eq_l1425_142516

theorem three_f_x_eq (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 2 / (3 + x)) (x : ℝ) (hx : x > 0) : 
  3 * f x = 18 / (9 + x) := sorry

end three_f_x_eq_l1425_142516


namespace molly_takes_180_minutes_longer_l1425_142519

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

end molly_takes_180_minutes_longer_l1425_142519


namespace certain_number_minus_two_l1425_142568

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := 
sorry

end certain_number_minus_two_l1425_142568


namespace passing_marks_l1425_142517

variable (T P : ℝ)

-- condition 1: 0.30T = P - 30
def condition1 : Prop := 0.30 * T = P - 30

-- condition 2: 0.45T = P + 15
def condition2 : Prop := 0.45 * T = P + 15

-- Proof Statement: P = 120 (passing marks)
theorem passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 120 := 
  sorry

end passing_marks_l1425_142517


namespace sqrt_equation_l1425_142565

theorem sqrt_equation (n : ℕ) (h : 0 < n) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n + 1) : ℝ) :=
sorry

end sqrt_equation_l1425_142565


namespace matrix_power_101_l1425_142593

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

theorem matrix_power_101 :
  B ^ (101 : ℕ) = B := sorry

end matrix_power_101_l1425_142593


namespace coeff_of_term_equal_three_l1425_142537

theorem coeff_of_term_equal_three (x : ℕ) (h : x = 13) : 
    2^x - 2^(x - 2) = 3 * 2^(11) :=
by
    rw [h]
    sorry

end coeff_of_term_equal_three_l1425_142537


namespace original_price_of_shoes_l1425_142590

noncomputable def original_price (final_price : ℝ) (sales_tax : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  final_price / sales_tax / (discount1 * discount2)

theorem original_price_of_shoes :
  original_price 51 1.07 0.40 0.85 = 140.18 := by
    have h_pre_tax_price : 47.66 = 51 / 1.07 := sorry
    have h_price_relation : 47.66 = 0.85 * 0.40 * 140.18 := sorry
    sorry

end original_price_of_shoes_l1425_142590


namespace positive_integral_solution_l1425_142518

theorem positive_integral_solution (n : ℕ) (hn : 0 < n) 
  (h : (n : ℚ) / (n + 1) = 125 / 126) : n = 125 := sorry

end positive_integral_solution_l1425_142518


namespace find_original_sum_of_money_l1425_142551

theorem find_original_sum_of_money
  (R : ℝ)
  (P : ℝ)
  (h1 : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 63) :
  P = 2100 :=
sorry

end find_original_sum_of_money_l1425_142551


namespace total_wristbands_proof_l1425_142584

-- Definitions from the conditions
def wristbands_per_person : ℕ := 2
def total_wristbands : ℕ := 125

-- Theorem statement to be proved
theorem total_wristbands_proof : total_wristbands = 125 :=
by
  sorry

end total_wristbands_proof_l1425_142584


namespace polynomial_division_quotient_l1425_142510

noncomputable def P (x : ℝ) := 8 * x^3 + 5 * x^2 - 4 * x - 7
noncomputable def D (x : ℝ) := x + 3

theorem polynomial_division_quotient :
  ∀ x : ℝ, (P x) / (D x) = 8 * x^2 - 19 * x + 53 := sorry

end polynomial_division_quotient_l1425_142510


namespace all_equal_l1425_142596

theorem all_equal (n : ℕ) (a : ℕ → ℝ) (h1 : 3 < n)
  (h2 : ∀ k : ℕ, k < n -> (a k)^3 = (a (k + 1 % n))^2 + (a (k + 2 % n))^2 + (a (k + 3 % n))^2) : 
  ∀ i j : ℕ, i < n -> j < n -> a i = a j :=
by
  sorry

end all_equal_l1425_142596


namespace f_of_7_l1425_142586

theorem f_of_7 (f : ℝ → ℝ) (h : ∀ (x : ℝ), f (4 * x - 1) = x^2 + 2 * x + 2) :
    f 7 = 10 := by
  sorry

end f_of_7_l1425_142586


namespace polynomial_divisibility_l1425_142522

theorem polynomial_divisibility (A B : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^(205 : ℕ) + A * x + B = 0) : 
    A + B = -1 :=
by
  sorry

end polynomial_divisibility_l1425_142522


namespace blue_to_yellow_ratio_is_half_l1425_142571

noncomputable section

def yellow_fish := 12
def blue_fish : ℕ := by 
  have total_fish := 42
  have green_fish := 2 * yellow_fish
  exact total_fish - (yellow_fish + green_fish)
def fish_ratio (x y : ℕ) := x / y

theorem blue_to_yellow_ratio_is_half : fish_ratio blue_fish yellow_fish = 1 / 2 := by
  sorry

end blue_to_yellow_ratio_is_half_l1425_142571


namespace concentration_sequences_and_min_operations_l1425_142546

theorem concentration_sequences_and_min_operations :
  (a_1 = 1.55 ∧ b_1 = 0.65) ∧
  (∀ n ≥ 1, a_n - b_n = 0.9 * (1 / 2)^(n - 1)) ∧
  (∃ n, 0.9 * (1 / 2)^(n - 1) < 0.01 ∧ n = 8) :=
by
  sorry

end concentration_sequences_and_min_operations_l1425_142546


namespace multiplication_identity_l1425_142554

theorem multiplication_identity (x y z w : ℝ) (h1 : x = 2000) (h2 : y = 2992) (h3 : z = 0.2992) (h4 : w = 20) : 
  x * y * z * w = 4 * y^2 :=
by
  sorry

end multiplication_identity_l1425_142554


namespace infinitely_many_good_pairs_l1425_142592

def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

theorem infinitely_many_good_pairs :
  ∃ (a b : ℕ), (0 < a) ∧ (0 < b) ∧ 
  ∀ t : ℕ, is_triangular t ↔ is_triangular (a * t + b) :=
sorry

end infinitely_many_good_pairs_l1425_142592


namespace total_gumballs_l1425_142585

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l1425_142585


namespace travel_probability_l1425_142560

theorem travel_probability (P_A P_B P_C : ℝ) (hA : P_A = 1/3) (hB : P_B = 1/4) (hC : P_C = 1/5) :
  let P_none_travel := (1 - P_A) * (1 - P_B) * (1 - P_C)
  ∃ (P_at_least_one : ℝ), P_at_least_one = 1 - P_none_travel ∧ P_at_least_one = 3/5 :=
by {
  sorry
}

end travel_probability_l1425_142560


namespace find_detergent_volume_l1425_142520

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

end find_detergent_volume_l1425_142520


namespace tan_sum_product_l1425_142513

theorem tan_sum_product (A B C : ℝ) (h_eq: Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := by
  sorry

end tan_sum_product_l1425_142513


namespace graph_does_not_pass_through_third_quadrant_l1425_142552

theorem graph_does_not_pass_through_third_quadrant (k x y : ℝ) (hk : k < 0) :
  y = k * x - k → (¬ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_does_not_pass_through_third_quadrant_l1425_142552


namespace janet_dresses_l1425_142555

theorem janet_dresses : 
  ∃ D : ℕ, 
    (D / 2) * (2 / 3) + (D / 2) * (6 / 3) = 32 → D = 24 := 
by {
  sorry
}

end janet_dresses_l1425_142555


namespace incorrect_value_at_x5_l1425_142576

theorem incorrect_value_at_x5 
  (f : ℕ → ℕ) 
  (provided_values : List ℕ) 
  (h_f : ∀ x, f x = 2 * x ^ 2 + 3 * x + 5)
  (h_provided_values : provided_values = [10, 18, 29, 44, 63, 84, 111, 140]) : 
  ¬ (f 5 = provided_values.get! 4) := 
by
  sorry

end incorrect_value_at_x5_l1425_142576


namespace monotonic_decreasing_interval_l1425_142588

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x + Real.pi / 4))

theorem monotonic_decreasing_interval :
  ∀ (x1 x2 : ℝ), (-Real.pi / 8) < x1 ∧ x1 < Real.pi / 8 ∧ (-Real.pi / 8) < x2 ∧ x2 < Real.pi / 8 ∧ x1 < x2 →
  f x1 > f x2 :=
sorry

end monotonic_decreasing_interval_l1425_142588


namespace trigonometric_identity_l1425_142573

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := 
by
  -- proof steps are omitted, using sorry to skip the proof.
  sorry

end trigonometric_identity_l1425_142573


namespace album_photos_proof_l1425_142566

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

end album_photos_proof_l1425_142566


namespace problem_part1_problem_part2_l1425_142511

variable (a b : ℝ)

theorem problem_part1 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  9 / a + 1 / b ≥ 4 :=
sorry

theorem problem_part2 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  ∃ a b, (a + 3 / b) * (b + 3 / a) = 12 :=
sorry

end problem_part1_problem_part2_l1425_142511


namespace remainder_sum_mod_15_l1425_142512

variable (k j : ℤ) -- these represent any integers

def p := 60 * k + 53
def q := 75 * j + 24

theorem remainder_sum_mod_15 :
  (p k + q j) % 15 = 2 :=  
by 
  sorry

end remainder_sum_mod_15_l1425_142512


namespace sam_coins_and_value_l1425_142525

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

end sam_coins_and_value_l1425_142525


namespace find_pairs_l1425_142530

theorem find_pairs (a b : ℕ) (h1: a > 0) (h2: b > 0) (q r : ℕ)
  (h3: a^2 + b^2 = q * (a + b) + r) (h4: q^2 + r = 1977) : 
  (a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50) :=
sorry

end find_pairs_l1425_142530


namespace Mary_cut_10_roses_l1425_142539

-- Defining the initial and final number of roses
def initial_roses := 6
def final_roses := 16

-- Calculating the number of roses cut by Mary
def roses_cut := final_roses - initial_roses

-- The proof problem: Prove that the number of roses cut is 10
theorem Mary_cut_10_roses : roses_cut = 10 := by
  sorry

end Mary_cut_10_roses_l1425_142539


namespace find_equation_of_line_midpoint_find_equation_of_line_vector_l1425_142595

-- Definition for Problem 1
def equation_of_line_midpoint (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (x / (-6) + y / 2 = 1) ∧ l (-3) = 1)

-- Proof Statement for Problem 1
theorem find_equation_of_line_midpoint : equation_of_line_midpoint (-6) 2 :=
sorry

-- Definition for Problem 2
def equation_of_line_vector (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (y - 1) / (-1) = (x + 3) / (-6) ∧ l (-3) = 1)

-- Proof Statement for Problem 2
theorem find_equation_of_line_vector : equation_of_line_vector (-9) (3 / 2) :=
sorry

end find_equation_of_line_midpoint_find_equation_of_line_vector_l1425_142595


namespace unique_trivial_solution_of_linear_system_l1425_142528

variable {R : Type*} [Field R]

theorem unique_trivial_solution_of_linear_system (a b c x y z : R)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_system : x + a * y + a^2 * z = 0 ∧ x + b * y + b^2 * z = 0 ∧ x + c * y + c^2 * z = 0) :
  x = 0 ∧ y = 0 ∧ z = 0 := sorry

end unique_trivial_solution_of_linear_system_l1425_142528


namespace percent_of_x_is_z_l1425_142515

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

end percent_of_x_is_z_l1425_142515


namespace snowballs_made_by_brother_l1425_142597

/-- Janet makes 50 snowballs and her brother makes the remaining snowballs. Janet made 25% of the total snowballs. 
    Prove that her brother made 150 snowballs. -/
theorem snowballs_made_by_brother (total_snowballs : ℕ) (janet_snowballs : ℕ) (fraction_janet : ℚ)
  (h1 : janet_snowballs = 50) (h2 : fraction_janet = 25 / 100) (h3 : janet_snowballs = fraction_janet * total_snowballs) :
  total_snowballs - janet_snowballs = 150 :=
by
  sorry

end snowballs_made_by_brother_l1425_142597


namespace mary_spent_on_jacket_l1425_142543

def shirt_cost : ℝ := 13.04
def total_cost : ℝ := 25.31
def jacket_cost : ℝ := total_cost - shirt_cost

theorem mary_spent_on_jacket :
  jacket_cost = 12.27 := by
  sorry

end mary_spent_on_jacket_l1425_142543


namespace find_original_number_l1425_142563

def original_number (x : ℝ) : Prop :=
  let step1 := 1.20 * x
  let step2 := step1 * 0.85
  let final_value := step2 * 1.30
  final_value = 1080

theorem find_original_number : ∃ x : ℝ, original_number x :=
by
  use 1080 / (1.20 * 0.85 * 1.30)
  sorry

end find_original_number_l1425_142563


namespace toothpaste_duration_l1425_142564

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

end toothpaste_duration_l1425_142564


namespace complete_the_square_example_l1425_142542

theorem complete_the_square_example (x : ℝ) : 
  ∃ c d : ℝ, (x^2 - 6 * x + 5 = 0) ∧ ((x + c)^2 = d) ∧ (d = 4) :=
sorry

end complete_the_square_example_l1425_142542


namespace ice_cream_total_sum_l1425_142548

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

end ice_cream_total_sum_l1425_142548


namespace ferry_tourists_total_l1425_142561

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

end ferry_tourists_total_l1425_142561


namespace dilation_image_l1425_142574

open Complex

theorem dilation_image (z₀ : ℂ) (c : ℂ) (k : ℝ) (z : ℂ)
    (h₀ : z₀ = 0 - 2*I) (h₁ : c = 1 + 2*I) (h₂ : k = 2) :
    z = -1 - 6*I :=
by
  sorry

end dilation_image_l1425_142574


namespace find_natural_numbers_l1425_142578

theorem find_natural_numbers (x y z : ℕ) (hx : x ≤ y) (hy : y ≤ z) : 
    (1 + 1 / x) * (1 + 1 / y) * (1 + 1 / z) = 3 
    → (x = 1 ∧ y = 3 ∧ z = 8) 
    ∨ (x = 1 ∧ y = 4 ∧ z = 5) 
    ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
sorry

end find_natural_numbers_l1425_142578


namespace max_k_value_l1425_142507

noncomputable def max_k (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : ℝ :=
  let k := (-1 + Real.sqrt 7) / 2
  k

theorem max_k_value (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_k_value_l1425_142507


namespace plane_equation_through_point_and_parallel_l1425_142523

theorem plane_equation_through_point_and_parallel (P : ℝ × ℝ × ℝ) (D : ℝ)
  (normal_vector : ℝ × ℝ × ℝ) (A B C : ℝ)
  (h1 : normal_vector = (2, -1, 3))
  (h2 : P = (2, 3, -1))
  (h3 : A = 2) (h4 : B = -1) (h5 : C = 3)
  (hD : A * 2 + B * 3 + C * -1 + D = 0) :
  A * x + B * y + C * z + D = 0 :=
by
  sorry

end plane_equation_through_point_and_parallel_l1425_142523


namespace range_of_m_l1425_142598

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0 
def neg_q_sufficient_for_neg_p (m : ℝ) : Prop :=
  ∀ x : ℝ, p x → q x m

theorem range_of_m (m : ℝ) : neg_q_sufficient_for_neg_p m → m ≥ 9 :=
by
  sorry

end range_of_m_l1425_142598


namespace find_k_l1425_142553

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

end find_k_l1425_142553


namespace root_expression_equals_181_div_9_l1425_142545

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

end root_expression_equals_181_div_9_l1425_142545


namespace increasing_function_l1425_142562

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Ici (1 : ℝ)) := by
  sorry

end increasing_function_l1425_142562


namespace tomTotalWeightMoved_is_525_l1425_142575

-- Tom's weight
def tomWeight : ℝ := 150

-- Weight in each hand
def weightInEachHand : ℝ := 1.5 * tomWeight

-- Weight vest
def weightVest : ℝ := 0.5 * tomWeight

-- Total weight moved
def totalWeightMoved : ℝ := (weightInEachHand * 2) + weightVest

theorem tomTotalWeightMoved_is_525 : totalWeightMoved = 525 := by
  sorry

end tomTotalWeightMoved_is_525_l1425_142575


namespace distance_problem_l1425_142579

-- Define the problem
theorem distance_problem
  (x y : ℝ)
  (h1 : x + y = 21)
  (h2 : x / 60 + 21 / 60 = 10 / 60 + y / 4) :
  x = 19 ∧ y = 2 :=
by
  sorry

end distance_problem_l1425_142579


namespace organization_population_after_six_years_l1425_142501

theorem organization_population_after_six_years :
  ∀ (b : ℕ → ℕ),
  (b 0 = 20) →
  (∀ k, b (k + 1) = 3 * (b k - 5) + 5) →
  b 6 = 10895 :=
by
  intros b h0 hr
  sorry

end organization_population_after_six_years_l1425_142501


namespace carter_drum_stick_sets_l1425_142577

theorem carter_drum_stick_sets (sets_per_show sets_tossed_per_show nights : ℕ) :
  sets_per_show = 5 →
  sets_tossed_per_show = 6 →
  nights = 30 →
  (sets_per_show + sets_tossed_per_show) * nights = 330 := by
  intros
  sorry

end carter_drum_stick_sets_l1425_142577


namespace triangle_area_eq_l1425_142536

variable (a b c: ℝ) (A B C : ℝ)
variable (h_cosC : Real.cos C = 1/4)
variable (h_c : c = 3)
variable (h_ratio : a / Real.cos A = b / Real.cos B)

theorem triangle_area_eq : (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 :=
by
  sorry

end triangle_area_eq_l1425_142536


namespace pyramid_lateral_edge_ratio_l1425_142570

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

end pyramid_lateral_edge_ratio_l1425_142570


namespace max_sum_of_triplet_product_60_l1425_142558

theorem max_sum_of_triplet_product_60 : 
  ∃ a b c : ℕ, a * b * c = 60 ∧ a + b + c = 62 :=
sorry

end max_sum_of_triplet_product_60_l1425_142558


namespace fraction_expression_as_common_fraction_l1425_142524

theorem fraction_expression_as_common_fraction :
  ((3 / 7 + 5 / 8) / (5 / 12 + 2 / 15)) = (295 / 154) := 
by
  sorry

end fraction_expression_as_common_fraction_l1425_142524


namespace integers_less_than_2019_divisible_by_18_or_21_but_not_both_l1425_142549

theorem integers_less_than_2019_divisible_by_18_or_21_but_not_both :
  ∃ (N : ℕ), (∀ (n : ℕ), (n < 2019 → (n % 18 = 0 ∨ n % 21 = 0) → n % (18 * 21 / gcd 18 21) ≠ 0) ↔ (∀ (m : ℕ), m < N)) ∧ N = 176 :=
by
  sorry

end integers_less_than_2019_divisible_by_18_or_21_but_not_both_l1425_142549


namespace calculate_markup_percentage_l1425_142527

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

end calculate_markup_percentage_l1425_142527


namespace negation_of_prop_l1425_142532

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop_l1425_142532


namespace g_neg_3_eq_neg_9_l1425_142582

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Given functions and values
variables (f g : ℝ → ℝ) (h_even : is_even_function f) (h_f_g : ∀ x, f x = g x - 2 * x)
variables (h_g3 : g 3 = 3)

-- Goal: Prove that g (-3) = -9
theorem g_neg_3_eq_neg_9 : g (-3) = -9 :=
sorry

end g_neg_3_eq_neg_9_l1425_142582
