import Mathlib

namespace NUMINAMATH_GPT_number_of_B_is_14_l2089_208922

-- Define the problem conditions
variable (num_students : ℕ)
variable (num_A num_B num_C num_D : ℕ)
variable (h1 : num_A = 8 * num_B / 10)
variable (h2 : num_C = 13 * num_B / 10)
variable (h3 : num_D = 5 * num_B / 10)
variable (h4 : num_students = 50)
variable (h5 : num_A + num_B + num_C + num_D = num_students)

-- Formalize the statement to be proved
theorem number_of_B_is_14 :
  num_B = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_B_is_14_l2089_208922


namespace NUMINAMATH_GPT_triangle_third_side_length_l2089_208911

theorem triangle_third_side_length (a b : ℕ) (h1 : a = 2) (h2 : b = 3) 
(h3 : ∃ x, x^2 - 10 * x + 21 = 0 ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)) :
  ∃ x, x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_third_side_length_l2089_208911


namespace NUMINAMATH_GPT_math_problem_l2089_208916

variable (a : ℝ) (m n : ℝ)

theorem math_problem
  (h1 : a^m = 3)
  (h2 : a^n = 2) :
  a^(2*m + 3*n) = 72 := 
  sorry

end NUMINAMATH_GPT_math_problem_l2089_208916


namespace NUMINAMATH_GPT_prob_two_fours_l2089_208945

-- Define the sample space for a fair die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The probability of rolling a 4 on a fair die
def prob_rolling_four : ℚ := 1 / 6

-- Probability of two independent events both resulting in rolling a 4
def prob_both_rolling_four : ℚ := (prob_rolling_four) * (prob_rolling_four)

-- Prove that the probability of rolling two 4s in two independent die rolls is 1/36
theorem prob_two_fours : prob_both_rolling_four = 1 / 36 := by
  sorry

end NUMINAMATH_GPT_prob_two_fours_l2089_208945


namespace NUMINAMATH_GPT_Vikki_take_home_pay_is_correct_l2089_208914

noncomputable def Vikki_take_home_pay : ℝ :=
  let hours_worked : ℝ := 42
  let hourly_pay_rate : ℝ := 12
  let gross_earnings : ℝ := hours_worked * hourly_pay_rate

  let fed_tax_first_300 : ℝ := 300 * 0.15
  let amount_over_300 : ℝ := gross_earnings - 300
  let fed_tax_excess : ℝ := amount_over_300 * 0.22
  let total_federal_tax : ℝ := fed_tax_first_300 + fed_tax_excess

  let state_tax : ℝ := gross_earnings * 0.07
  let retirement_contribution : ℝ := gross_earnings * 0.06
  let insurance_cover : ℝ := gross_earnings * 0.03
  let union_dues : ℝ := 5

  let total_deductions : ℝ := total_federal_tax + state_tax + retirement_contribution + insurance_cover + union_dues
  let take_home_pay : ℝ := gross_earnings - total_deductions
  take_home_pay

theorem Vikki_take_home_pay_is_correct : Vikki_take_home_pay = 328.48 :=
by
  sorry

end NUMINAMATH_GPT_Vikki_take_home_pay_is_correct_l2089_208914


namespace NUMINAMATH_GPT_area_ratio_l2089_208969

theorem area_ratio (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AB BC AC AD AE : ℝ) (ADE_ratio : ℝ) :
  AB = 25 ∧ BC = 39 ∧ AC = 42 ∧ AD = 19 ∧ AE = 14 →
  ADE_ratio = 19 / 56 :=
by sorry

end NUMINAMATH_GPT_area_ratio_l2089_208969


namespace NUMINAMATH_GPT_largest_non_expressible_number_l2089_208996

theorem largest_non_expressible_number :
  ∀ (x y z : ℕ), 15 * x + 18 * y + 20 * z ≠ 97 :=
by sorry

end NUMINAMATH_GPT_largest_non_expressible_number_l2089_208996


namespace NUMINAMATH_GPT_unicorn_tether_l2089_208912

theorem unicorn_tether (a b c : ℕ) (h_c_prime : Prime c) :
  (∃ (a b c : ℕ), c = 1 ∧ (25 - 15 = 10 ∧ 10^2 + 10^2 = 15^2 ∧ 
  a = 10 ∧ b = 125) ∧ a + b + c = 136) :=
  sorry

end NUMINAMATH_GPT_unicorn_tether_l2089_208912


namespace NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2089_208953

def U : Set ℤ := {1, 2, 3, 4, 5}
def A : Set ℤ := {x | abs (x - 3) < 2}
def C_UA : Set ℤ := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A_with_respect_to_U :
  C_UA = {1, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2089_208953


namespace NUMINAMATH_GPT_cylinder_surface_area_is_128pi_l2089_208954

noncomputable def cylinder_total_surface_area (h r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_surface_area_is_128pi :
  cylinder_total_surface_area 12 4 = 128 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_is_128pi_l2089_208954


namespace NUMINAMATH_GPT_area_of_small_parallelograms_l2089_208968

theorem area_of_small_parallelograms (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  (1 : ℝ) / (m * n : ℝ) = 1 / (m * n) :=
by
  sorry

end NUMINAMATH_GPT_area_of_small_parallelograms_l2089_208968


namespace NUMINAMATH_GPT_candy_sharing_l2089_208949

theorem candy_sharing (Hugh_candy Tommy_candy Melany_candy shared_candy : ℕ) 
  (h1 : Hugh_candy = 8) (h2 : Tommy_candy = 6) (h3 : shared_candy = 7) :
  Hugh_candy + Tommy_candy + Melany_candy = 3 * shared_candy →
  Melany_candy = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_candy_sharing_l2089_208949


namespace NUMINAMATH_GPT_inscribed_circle_radius_l2089_208933

theorem inscribed_circle_radius (a b c r : ℝ) (h : a^2 + b^2 = c^2) (h' : r = (a + b - c) / 2) : r = (a + b - c) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l2089_208933


namespace NUMINAMATH_GPT_find_divisor_l2089_208904

theorem find_divisor (n d : ℤ) (k : ℤ)
  (h1 : n % d = 3)
  (h2 : n^2 % d = 4) : d = 5 :=
sorry

end NUMINAMATH_GPT_find_divisor_l2089_208904


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l2089_208932

theorem first_term_of_geometric_sequence (a r : ℚ) 
  (h1 : a * r = 18) 
  (h2 : a * r^2 = 24) : 
  a = 27 / 2 := 
sorry

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l2089_208932


namespace NUMINAMATH_GPT_derivative_at_3_l2089_208935

noncomputable def f (x : ℝ) := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end NUMINAMATH_GPT_derivative_at_3_l2089_208935


namespace NUMINAMATH_GPT_four_digit_integer_5533_l2089_208995

theorem four_digit_integer_5533
  (a b c d : ℕ)
  (h1 : a + b + c + d = 16)
  (h2 : b + c = 8)
  (h3 : a - d = 2)
  (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  1000 * a + 100 * b + 10 * c + d = 5533 :=
by {
  sorry
}

end NUMINAMATH_GPT_four_digit_integer_5533_l2089_208995


namespace NUMINAMATH_GPT_part1_part2_l2089_208977

theorem part1 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ a b : students, a ≠ b ∧
  (∀ c : students, c ≠ a → d a c > d a b) ∧ 
  (∀ c : students, c ≠ b → d b c > d b a) :=
sorry

theorem part2 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ c : students, ∀ a : students, ¬ (∀ b : students, b ≠ a → d b a < d b c ∧ d a c < d a b) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2089_208977


namespace NUMINAMATH_GPT_max_value_m_l2089_208915

theorem max_value_m (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ (m : ℝ), (4 / (1 - x) ≥ m - 1 / x)) ↔ (∃ (m : ℝ), m ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_max_value_m_l2089_208915


namespace NUMINAMATH_GPT_math_problem_l2089_208987

variables (x y z : ℝ)

theorem math_problem
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( (x^2 / (x + y) >= (3 * x - y) / 4) ) ∧ 
  ( (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) >= (x * y + y * z + z * x) / 2 ) :=
by sorry

end NUMINAMATH_GPT_math_problem_l2089_208987


namespace NUMINAMATH_GPT_angle_C_in_triangle_l2089_208957

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end NUMINAMATH_GPT_angle_C_in_triangle_l2089_208957


namespace NUMINAMATH_GPT_minnie_takes_longer_l2089_208944

def minnie_speed_flat := 25 -- kph
def minnie_speed_downhill := 35 -- kph
def minnie_speed_uphill := 10 -- kph

def penny_speed_flat := 35 -- kph
def penny_speed_downhill := 45 -- kph
def penny_speed_uphill := 15 -- kph

def distance_flat := 25 -- km
def distance_downhill := 20 -- km
def distance_uphill := 15 -- km

noncomputable def minnie_time := 
  (distance_uphill / minnie_speed_uphill) + 
  (distance_downhill / minnie_speed_downhill) + 
  (distance_flat / minnie_speed_flat) -- hours

noncomputable def penny_time := 
  (distance_uphill / penny_speed_uphill) + 
  (distance_downhill / penny_speed_downhill) + 
  (distance_flat / penny_speed_flat) -- hours

noncomputable def minnie_time_minutes := minnie_time * 60 -- minutes
noncomputable def penny_time_minutes := penny_time * 60 -- minutes

noncomputable def time_difference := minnie_time_minutes - penny_time_minutes -- minutes

theorem minnie_takes_longer : time_difference = 130 :=
  sorry

end NUMINAMATH_GPT_minnie_takes_longer_l2089_208944


namespace NUMINAMATH_GPT_root_of_quadratic_l2089_208913

theorem root_of_quadratic (b : ℝ) : 
  (-9)^2 + b * (-9) - 45 = 0 -> b = 4 :=
by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_l2089_208913


namespace NUMINAMATH_GPT_find_sum_of_pqr_l2089_208990

theorem find_sum_of_pqr (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) 
(h4 : q = p * (4 - p)) (h5 : r = q * (4 - q)) (h6 : p = r * (4 - r)) : 
p + q + r = 6 :=
sorry

end NUMINAMATH_GPT_find_sum_of_pqr_l2089_208990


namespace NUMINAMATH_GPT_remainder_poly_div_l2089_208992

theorem remainder_poly_div 
    (x : ℤ) 
    (h1 : (x^2 + x + 1) ∣ (x^3 - 1)) 
    (h2 : x^5 - 1 = (x^3 - 1) * (x^2 + x + 1) - x * (x^2 + x + 1) + 1) : 
  ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_poly_div_l2089_208992


namespace NUMINAMATH_GPT_tangent_line_equation_at_point_l2089_208986

theorem tangent_line_equation_at_point 
  (x y : ℝ) (h_curve : y = x^3 - 2 * x) (h_point : (x, y) = (1, -1)) : 
  (x - y - 2 = 0) := 
sorry

end NUMINAMATH_GPT_tangent_line_equation_at_point_l2089_208986


namespace NUMINAMATH_GPT_rectangular_prism_diagonals_l2089_208948

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_diagonals_l2089_208948


namespace NUMINAMATH_GPT_sin_330_correct_l2089_208980

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_sin_330_correct_l2089_208980


namespace NUMINAMATH_GPT_moores_law_2000_l2089_208962

noncomputable def number_of_transistors (year : ℕ) : ℕ :=
  if year = 1990 then 1000000
  else 1000000 * 2 ^ ((year - 1990) / 2)

theorem moores_law_2000 :
  number_of_transistors 2000 = 32000000 :=
by
  unfold number_of_transistors
  rfl

end NUMINAMATH_GPT_moores_law_2000_l2089_208962


namespace NUMINAMATH_GPT_expression_multiple_l2089_208941

theorem expression_multiple :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  (a - b) / (1/78) = 13 :=
by
  sorry

end NUMINAMATH_GPT_expression_multiple_l2089_208941


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2089_208955

def A : Set ℝ := { x | x ≥ 0 }
def B : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 ≤ x ∧ x < 2 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2089_208955


namespace NUMINAMATH_GPT_perfect_power_transfer_l2089_208924

-- Given Conditions
variables {x y z : ℕ}

-- Definition of what it means to be a perfect seventh power
def is_perfect_seventh_power (n : ℕ) :=
  ∃ k : ℕ, n = k^7

-- The proof problem
theorem perfect_power_transfer 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : is_perfect_seventh_power (x^3 * y^5 * z^6)) :
  is_perfect_seventh_power (x^5 * y^6 * z^3) := by
  sorry

end NUMINAMATH_GPT_perfect_power_transfer_l2089_208924


namespace NUMINAMATH_GPT_solution_set_inequality_l2089_208902

theorem solution_set_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
  ∀ (x : ℝ), (0 < x ∧ x ≤ 1 ↔ ∀ a > 0, ∀ b ≤ 1, (2/x + (1-x) ^ (1/2) ≥ 1 + (1-x)^(1/2))) := sorry

end NUMINAMATH_GPT_solution_set_inequality_l2089_208902


namespace NUMINAMATH_GPT_triple_hash_100_l2089_208989

def hash (N : ℝ) : ℝ :=
  0.5 * N + N

theorem triple_hash_100 : hash (hash (hash 100)) = 337.5 :=
by
  sorry

end NUMINAMATH_GPT_triple_hash_100_l2089_208989


namespace NUMINAMATH_GPT_smallest_c_such_that_one_in_range_l2089_208907

theorem smallest_c_such_that_one_in_range :
  ∃ c : ℝ, (∀ x : ℝ, ∃ y : ℝ, y =  x^2 - 2 * x + c ∧ y = 1) ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_such_that_one_in_range_l2089_208907


namespace NUMINAMATH_GPT_Kevin_crates_per_week_l2089_208938

theorem Kevin_crates_per_week (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 20) (h₃ : c = 17) :
  a + b + c = 50 :=
by 
  sorry

end NUMINAMATH_GPT_Kevin_crates_per_week_l2089_208938


namespace NUMINAMATH_GPT_parabola_focus_on_line_l2089_208908

theorem parabola_focus_on_line (p : ℝ) (h₁ : 0 < p) (h₂ : (2 * (p / 2) + 0 - 2 = 0)) : p = 2 :=
sorry

end NUMINAMATH_GPT_parabola_focus_on_line_l2089_208908


namespace NUMINAMATH_GPT_factor_poly_l2089_208930

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end NUMINAMATH_GPT_factor_poly_l2089_208930


namespace NUMINAMATH_GPT_mul_112_54_l2089_208998

theorem mul_112_54 : 112 * 54 = 6048 :=
by
  sorry

end NUMINAMATH_GPT_mul_112_54_l2089_208998


namespace NUMINAMATH_GPT_solve_system_l2089_208978

theorem solve_system (x y : ℝ) (h1 : 2 * x - y = 0) (h2 : x + 2 * y = 1) : 
  x = 1 / 5 ∧ y = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2089_208978


namespace NUMINAMATH_GPT_hypotenuse_length_l2089_208985

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end NUMINAMATH_GPT_hypotenuse_length_l2089_208985


namespace NUMINAMATH_GPT_number_of_cheeses_per_pack_l2089_208974

-- Definitions based on the conditions
def packs : ℕ := 3
def cost_per_cheese : ℝ := 0.10
def total_amount_paid : ℝ := 6

-- Theorem statement to prove the number of string cheeses in each pack
theorem number_of_cheeses_per_pack : 
  (total_amount_paid / (packs : ℝ)) / cost_per_cheese = 20 :=
sorry

end NUMINAMATH_GPT_number_of_cheeses_per_pack_l2089_208974


namespace NUMINAMATH_GPT_base_8_to_base_4_l2089_208947

theorem base_8_to_base_4 (n : ℕ) (h : n = 6 * 8^2 + 5 * 8^1 + 3 * 8^0) : 
  (n : ℕ) = 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 2 * 4^1 + 3 * 4^0 :=
by
  -- Conversion proof goes here
  sorry

end NUMINAMATH_GPT_base_8_to_base_4_l2089_208947


namespace NUMINAMATH_GPT_dawn_monthly_payments_l2089_208951

theorem dawn_monthly_payments (annual_salary : ℕ) (saved_per_month : ℕ)
  (h₁ : annual_salary = 48000)
  (h₂ : saved_per_month = 400)
  (h₃ : ∀ (monthly_salary : ℕ), saved_per_month = (10 * monthly_salary) / 100):
  annual_salary / saved_per_month = 12 :=
by
  sorry

end NUMINAMATH_GPT_dawn_monthly_payments_l2089_208951


namespace NUMINAMATH_GPT_arithmetic_sequence_index_l2089_208905

theorem arithmetic_sequence_index (a : ℕ → ℕ) (n : ℕ) (first_term comm_diff : ℕ):
  (∀ k, a k = first_term + comm_diff * (k - 1)) → a n = 2016 → n = 404 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_index_l2089_208905


namespace NUMINAMATH_GPT_floor_width_l2089_208997

theorem floor_width (W : ℕ) (hAreaFloor: 10 * W - 64 = 16) : W = 8 :=
by
  -- the proof should be added here
  sorry

end NUMINAMATH_GPT_floor_width_l2089_208997


namespace NUMINAMATH_GPT_wendy_facial_products_l2089_208927

def total_time (P : ℕ) : ℕ :=
  5 * (P - 1) + 30

theorem wendy_facial_products :
  (total_time 6 = 55) :=
by
  sorry

end NUMINAMATH_GPT_wendy_facial_products_l2089_208927


namespace NUMINAMATH_GPT_divisor_proof_l2089_208984

def original_number : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

def remainder : ℕ := 36

theorem divisor_proof (D : ℕ) (Q : ℕ) (h : original_number = D * Q + remainder) : original_number % D = remainder :=
by 
  sorry

end NUMINAMATH_GPT_divisor_proof_l2089_208984


namespace NUMINAMATH_GPT_least_number_to_add_l2089_208988

theorem least_number_to_add (n : ℕ) (h : n = 17 * 23 * 29) : 
  ∃ k, k + 1024 ≡ 0 [MOD n] ∧ 
       (∀ m, (m + 1024) ≡ 0 [MOD n] → k ≤ m) ∧ 
       k = 10315 :=
by 
  sorry

end NUMINAMATH_GPT_least_number_to_add_l2089_208988


namespace NUMINAMATH_GPT_gcd_of_a_and_b_lcm_of_a_and_b_l2089_208971

def a : ℕ := 2 * 3 * 7
def b : ℕ := 2 * 3 * 3 * 5

theorem gcd_of_a_and_b : Nat.gcd a b = 6 := by
  sorry

theorem lcm_of_a_and_b : Nat.lcm a b = 630 := by
  sorry

end NUMINAMATH_GPT_gcd_of_a_and_b_lcm_of_a_and_b_l2089_208971


namespace NUMINAMATH_GPT_prime_bound_l2089_208910

-- The definition for the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry  -- placeholder for the primorial definition

-- The main theorem to prove
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ 2 ^ (n - 1) := sorry

end NUMINAMATH_GPT_prime_bound_l2089_208910


namespace NUMINAMATH_GPT_find_a_l2089_208928

theorem find_a 
  (a : ℝ)
  (h : ∀ n : ℕ, (n.choose 2) * 2^(5-2) * a^2 = 80 → n = 5) :
  a = 1 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2089_208928


namespace NUMINAMATH_GPT_playground_area_l2089_208909

theorem playground_area (w l : ℕ) (h1 : 2 * l + 2 * w = 72) (h2 : l = 3 * w) : l * w = 243 := by
  sorry

end NUMINAMATH_GPT_playground_area_l2089_208909


namespace NUMINAMATH_GPT_failed_english_is_45_l2089_208940

-- Definitions of the given conditions
def total_students : ℝ := 1 -- representing 100%
def failed_hindi : ℝ := 0.35
def failed_both : ℝ := 0.2
def passed_both : ℝ := 0.4

-- The goal is to prove that the percentage of students who failed in English is 45%

theorem failed_english_is_45 :
  let failed_at_least_one := total_students - passed_both
  let failed_english := failed_at_least_one - failed_hindi + failed_both
  failed_english = 0.45 :=
by
  -- The steps and manipulation will go here, but for now we skip with sorry
  sorry

end NUMINAMATH_GPT_failed_english_is_45_l2089_208940


namespace NUMINAMATH_GPT_sum_of_x_and_y_l2089_208965

theorem sum_of_x_and_y (x y : ℝ) 
  (h₁ : |x| + x + 5 * y = 2)
  (h₂ : |y| - y + x = 7) : 
  x + y = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l2089_208965


namespace NUMINAMATH_GPT_which_polygon_covers_ground_l2089_208939

def is_tessellatable (n : ℕ) : Prop :=
  let interior_angle := (n - 2) * 180 / n
  360 % interior_angle = 0

theorem which_polygon_covers_ground :
  is_tessellatable 6 ∧ ¬is_tessellatable 5 ∧ ¬is_tessellatable 8 ∧ ¬is_tessellatable 12 :=
by
  sorry

end NUMINAMATH_GPT_which_polygon_covers_ground_l2089_208939


namespace NUMINAMATH_GPT_check_point_on_curve_l2089_208936

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x * y + 2 * y + 1 = 0

theorem check_point_on_curve :
  point_on_curve 0 (-1/2) :=
by
  sorry

end NUMINAMATH_GPT_check_point_on_curve_l2089_208936


namespace NUMINAMATH_GPT_rationalize_denominator_l2089_208929

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2089_208929


namespace NUMINAMATH_GPT_required_integer_l2089_208921

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4 = sum

def middle_digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  d2 + d3 = sum

def thousands_minus_units (n : ℕ) (diff : ℕ) : Prop :=
  let d1 := n / 1000
  let d4 := n % 10
  d1 - d4 = diff

def divisible_by (n : ℕ) (d : ℕ) : Prop :=
  n % d = 0

theorem required_integer : 
  ∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    digits_sum_to n 18 ∧ 
    middle_digits_sum_to n 9 ∧ 
    thousands_minus_units n 3 ∧ 
    divisible_by n 9 ∧ 
    n = 6453 :=
by
  sorry

end NUMINAMATH_GPT_required_integer_l2089_208921


namespace NUMINAMATH_GPT_solve_for_a_and_b_range_of_f_when_x_lt_zero_l2089_208950

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1 + a * (2 ^ x)) / (2 ^ x + b)

theorem solve_for_a_and_b (a b : ℝ) :
  f a b 1 = 3 ∧
  f a b (-1) = -3 →
  a = 1 ∧ b = -1 :=
by
  sorry

theorem range_of_f_when_x_lt_zero (x : ℝ) :
  ∀ x < 0, f 1 (-1) x < -1 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_a_and_b_range_of_f_when_x_lt_zero_l2089_208950


namespace NUMINAMATH_GPT_weight_of_6m_rod_l2089_208900

theorem weight_of_6m_rod (r ρ : ℝ) (h₁ : 11.25 > 0) (h₂ : 6 > 0) (h₃ : 0 < r) (h₄ : 42.75 = π * r^2 * 11.25 * ρ) : 
  (π * r^2 * 6 * (42.75 / (π * r^2 * 11.25))) = 22.8 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_6m_rod_l2089_208900


namespace NUMINAMATH_GPT_original_square_area_l2089_208979

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end NUMINAMATH_GPT_original_square_area_l2089_208979


namespace NUMINAMATH_GPT_compound_interest_for_2_years_l2089_208973

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_for_2_years 
  (P : ℝ) (R : ℝ) (T : ℝ) (S : ℝ)
  (h1 : S = 600)
  (h2 : R = 5)
  (h3 : T = 2)
  (h4 : simple_interest P R T = S)
  : compound_interest P R T = 615 := 
sorry

end NUMINAMATH_GPT_compound_interest_for_2_years_l2089_208973


namespace NUMINAMATH_GPT_cylinder_volume_increase_l2089_208906

variable (r h : ℝ)

theorem cylinder_volume_increase :
  (π * (4 * r) ^ 2 * (2 * h)) = 32 * (π * r ^ 2 * h) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_increase_l2089_208906


namespace NUMINAMATH_GPT_angle_solution_l2089_208983

/-!
  Given:
  k + 90° = 360°

  Prove:
  k = 270°
-/

theorem angle_solution (k : ℝ) (h : k + 90 = 360) : k = 270 :=
by
  sorry

end NUMINAMATH_GPT_angle_solution_l2089_208983


namespace NUMINAMATH_GPT_percentage_reduced_l2089_208956

theorem percentage_reduced (P : ℝ) (h : (85 * P / 100) - 11 = 23) : P = 40 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_reduced_l2089_208956


namespace NUMINAMATH_GPT_net_income_on_15th_day_l2089_208959

noncomputable def net_income_15th_day : ℝ :=
  let earnings_15th_day := 3 * (3 ^ 14)
  let tax := 0.10 * earnings_15th_day
  let earnings_after_tax := earnings_15th_day - tax
  earnings_after_tax - 100

theorem net_income_on_15th_day :
  net_income_15th_day = 12913916.3 := by
  sorry

end NUMINAMATH_GPT_net_income_on_15th_day_l2089_208959


namespace NUMINAMATH_GPT_number_of_ways_to_score_l2089_208982

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end NUMINAMATH_GPT_number_of_ways_to_score_l2089_208982


namespace NUMINAMATH_GPT_probability_order_correct_l2089_208942

inductive Phenomenon
| Certain
| VeryLikely
| Possible
| Impossible
| NotVeryLikely

open Phenomenon

def probability_order : Phenomenon → ℕ
| Certain       => 5
| VeryLikely    => 4
| Possible      => 3
| NotVeryLikely => 2
| Impossible    => 1

theorem probability_order_correct :
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] =
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] :=
by
  -- skips the proof
  sorry

end NUMINAMATH_GPT_probability_order_correct_l2089_208942


namespace NUMINAMATH_GPT_paul_lost_crayons_l2089_208958

theorem paul_lost_crayons :
  ∀ (initial_crayons given_crayons left_crayons lost_crayons : ℕ),
    initial_crayons = 1453 →
    given_crayons = 563 →
    left_crayons = 332 →
    lost_crayons = (initial_crayons - given_crayons) - left_crayons →
    lost_crayons = 558 :=
by
  intros initial_crayons given_crayons left_crayons lost_crayons
  intros h_initial h_given h_left h_lost
  sorry

end NUMINAMATH_GPT_paul_lost_crayons_l2089_208958


namespace NUMINAMATH_GPT_vector_projection_condition_l2089_208918

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 3 + 2 * t)
noncomputable def line_m (s : ℝ) : ℝ × ℝ := (4 + 2 * s, 5 + 3 * s)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_projection_condition 
  (t s : ℝ)
  (C : ℝ × ℝ := line_l t)
  (D : ℝ × ℝ := line_m s)
  (Q : ℝ × ℝ)
  (hQ : is_perpendicular (Q.1 - C.1, Q.2 - C.2) (2, 3))
  (v1 v2 : ℝ)
  (hv_sum : v1 + v2 = 3)
  (hv_def : ∃ k : ℝ, v1 = 3 * k ∧ v2 = -2 * k)
  : (v1, v2) = (9, -6) := 
sorry

end NUMINAMATH_GPT_vector_projection_condition_l2089_208918


namespace NUMINAMATH_GPT_sample_processing_l2089_208981

-- Define sample data
def standard: ℕ := 220
def samples: List ℕ := [230, 226, 218, 223, 214, 225, 205, 212]

-- Calculate deviations
def deviations (samples: List ℕ) (standard: ℕ) : List ℤ :=
  samples.map (λ x => x - standard)

-- Total dosage of samples
def total_dosage (samples: List ℕ): ℕ :=
  samples.sum

-- Total cost to process to standard dosage
def total_cost (deviations: List ℤ) (cost_per_ml_adjustment: ℤ) : ℤ :=
  cost_per_ml_adjustment * (deviations.map Int.natAbs).sum

-- Theorem statement
theorem sample_processing :
  let deviation_vals := deviations samples standard;
  let total_dosage_val := total_dosage samples;
  let total_cost_val := total_cost deviation_vals 10;
  deviation_vals = [10, 6, -2, 3, -6, 5, -15, -8] ∧
  total_dosage_val = 1753 ∧
  total_cost_val = 550 :=
by
  sorry

end NUMINAMATH_GPT_sample_processing_l2089_208981


namespace NUMINAMATH_GPT_sheila_paintings_l2089_208960

theorem sheila_paintings (a b : ℕ) (h1 : a = 9) (h2 : b = 9) : a + b = 18 :=
by
  sorry

end NUMINAMATH_GPT_sheila_paintings_l2089_208960


namespace NUMINAMATH_GPT_length_BD_l2089_208920

/-- Points A, B, C, and D lie on a line in that order. We are given:
  AB = 2 cm,
  AC = 5 cm, and
  CD = 3 cm.
Then, we need to show that the length of BD is 6 cm. -/
theorem length_BD :
  ∀ (A B C D : ℕ),
  A + B = 2 → A + C = 5 → C + D = 3 →
  D - B = 6 :=
by
  intros A B C D h1 h2 h3
  -- Proof steps to be filled in
  sorry

end NUMINAMATH_GPT_length_BD_l2089_208920


namespace NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l2089_208970

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (h1 : (1 + x) / 2 = 5)
  (h2 : (2 + y) / 2 = 6) :
  x + y = 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l2089_208970


namespace NUMINAMATH_GPT_find_triple_l2089_208999

theorem find_triple (A B C : ℕ) (h1 : A^2 + B - C = 100) (h2 : A + B^2 - C = 124) : 
  (A, B, C) = (12, 13, 57) := 
  sorry

end NUMINAMATH_GPT_find_triple_l2089_208999


namespace NUMINAMATH_GPT_divisible_iff_l2089_208903

-- Definitions from the conditions
def a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a (n + 1) + a n

-- Main theorem statement.
theorem divisible_iff (n k : ℕ) : 2^k ∣ a n ↔ 2^k ∣ n := by
  sorry

end NUMINAMATH_GPT_divisible_iff_l2089_208903


namespace NUMINAMATH_GPT_fish_in_third_tank_l2089_208919

-- Definitions of the conditions
def first_tank_goldfish : ℕ := 7
def first_tank_beta_fish : ℕ := 8
def first_tank_fish : ℕ := first_tank_goldfish + first_tank_beta_fish

def second_tank_fish : ℕ := 2 * first_tank_fish

def third_tank_fish : ℕ := second_tank_fish / 3

-- The statement to prove
theorem fish_in_third_tank : third_tank_fish = 10 := by
  sorry

end NUMINAMATH_GPT_fish_in_third_tank_l2089_208919


namespace NUMINAMATH_GPT_no_solution_to_equation_l2089_208993

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, 8 / (x ^ 2 - 4) + 1 = x / (x - 2) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_to_equation_l2089_208993


namespace NUMINAMATH_GPT_students_in_neither_l2089_208925

def total_students := 60
def students_in_art := 40
def students_in_music := 30
def students_in_both := 15

theorem students_in_neither : total_students - (students_in_art - students_in_both + students_in_music - students_in_both + students_in_both) = 5 :=
by
  sorry

end NUMINAMATH_GPT_students_in_neither_l2089_208925


namespace NUMINAMATH_GPT_solid_brick_height_l2089_208961

theorem solid_brick_height (n c base_perimeter height : ℕ) 
  (h1 : n = 42) 
  (h2 : c = 1) 
  (h3 : base_perimeter = 18)
  (h4 : n % base_area = 0)
  (h5 : 2 * (length + width) = base_perimeter)
  (h6 : base_area * height = n) : 
  height = 3 :=
by sorry

end NUMINAMATH_GPT_solid_brick_height_l2089_208961


namespace NUMINAMATH_GPT_find_x_for_condition_l2089_208931

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x_for_condition :
  (2 * f 1 - 16 = f (1 - 6)) :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_condition_l2089_208931


namespace NUMINAMATH_GPT_simplify_fraction_expr_l2089_208901

theorem simplify_fraction_expr (a : ℝ) (h : a ≠ 1) : (a / (a - 1) + 1 / (1 - a)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_expr_l2089_208901


namespace NUMINAMATH_GPT_center_of_circle_polar_eq_l2089_208963

theorem center_of_circle_polar_eq (ρ θ : ℝ) : 
    (∀ ρ θ, ρ = 2 * Real.cos θ ↔ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) → 
    ∃ x y : ℝ, x = 1 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_polar_eq_l2089_208963


namespace NUMINAMATH_GPT_isosceles_triangle_sine_base_angle_l2089_208966

theorem isosceles_triangle_sine_base_angle (m : ℝ) (θ : ℝ) 
  (h1 : m > 0)
  (h2 : θ > 0 ∧ θ < π / 2)
  (h_base_height : m * (Real.sin θ) = (m * 2 * (Real.sin θ) * (Real.cos θ))) :
  Real.sin θ = (Real.sqrt 15) / 4 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_sine_base_angle_l2089_208966


namespace NUMINAMATH_GPT_five_b_value_l2089_208976

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 :=
by
  sorry

end NUMINAMATH_GPT_five_b_value_l2089_208976


namespace NUMINAMATH_GPT_max_distance_l2089_208923

-- Definition of curve C₁ in rectangular coordinates.
def C₁_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Definition of curve C₂ in its general form.
def C₂_general (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Coordinates of point M, the intersection of C₂ with x-axis.
def M : ℝ × ℝ := (2, 0)

-- Condition that N is a moving point on curve C₁.
def N (x y : ℝ) : Prop := C₁_rectangular x y

-- Maximum distance |MN|.
theorem max_distance (x y : ℝ) (hN : N x y) : 
  dist (2, 0) (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_GPT_max_distance_l2089_208923


namespace NUMINAMATH_GPT_simplify_tan_expression_l2089_208994

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end NUMINAMATH_GPT_simplify_tan_expression_l2089_208994


namespace NUMINAMATH_GPT_remaining_length_after_cut_l2089_208917

/- Definitions -/
def original_length (a b : ℕ) : ℕ := 5 * a + 4 * b
def rectangle_perimeter (a b : ℕ) : ℕ := 2 * (a + b)
def remaining_length (a b : ℕ) : ℕ := original_length a b - rectangle_perimeter a b

/- Theorem statement -/
theorem remaining_length_after_cut (a b : ℕ) : remaining_length a b = 3 * a + 2 * b := 
by 
  sorry

end NUMINAMATH_GPT_remaining_length_after_cut_l2089_208917


namespace NUMINAMATH_GPT_product_of_000412_and_9243817_is_closest_to_3600_l2089_208937

def product_closest_to (x y value: ℝ) : Prop := (abs (x * y - value) < min (abs (x * y - 350)) (min (abs (x * y - 370)) (min (abs (x * y - 3700)) (abs (x * y - 4000)))))

theorem product_of_000412_and_9243817_is_closest_to_3600 :
  product_closest_to 0.000412 9243817 3600 :=
by
  sorry

end NUMINAMATH_GPT_product_of_000412_and_9243817_is_closest_to_3600_l2089_208937


namespace NUMINAMATH_GPT_proof_remove_terms_sum_is_one_l2089_208926

noncomputable def remove_terms_sum_is_one : Prop :=
  let initial_sum := (1/2) + (1/4) + (1/6) + (1/8) + (1/10) + (1/12)
  let terms_to_remove := (1/8) + (1/10)
  initial_sum - terms_to_remove = 1

theorem proof_remove_terms_sum_is_one : remove_terms_sum_is_one :=
by
  -- proof will go here but is not required
  sorry

end NUMINAMATH_GPT_proof_remove_terms_sum_is_one_l2089_208926


namespace NUMINAMATH_GPT_extreme_points_f_l2089_208972

theorem extreme_points_f (a b : ℝ)
  (h1 : 3 * (-2)^2 + 2 * a * (-2) + b = 0)
  (h2 : 3 * 4^2 + 2 * a * 4 + b = 0) :
  a - b = 21 :=
sorry

end NUMINAMATH_GPT_extreme_points_f_l2089_208972


namespace NUMINAMATH_GPT_solve_ordered_pair_l2089_208943

theorem solve_ordered_pair : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^y + 3 = y^x ∧ 2 * x^y = y^x + 11 ∧ x = 14 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_ordered_pair_l2089_208943


namespace NUMINAMATH_GPT_find_a_minus_b_l2089_208952

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 3 * a * x + 4

-- Define the condition for the function being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function f(x) with given parameters
theorem find_a_minus_b (a b : ℝ) (h_dom_range : ∀ x : ℝ, b - 3 ≤ x → x ≤ 2 * b) (h_even_f : is_even (f a)) :
  a - b = -1 :=
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l2089_208952


namespace NUMINAMATH_GPT_intersection_eq_l2089_208991

open Set

variable {α : Type*}

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_eq : M ∩ N = {2, 3} := by
  apply Set.ext
  intro x
  simp [M, N]
  sorry

end NUMINAMATH_GPT_intersection_eq_l2089_208991


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_geometric_sequence_is_two_l2089_208934

theorem sum_of_reciprocals_of_geometric_sequence_is_two
  (a1 q : ℝ)
  (pos_terms : 0 < a1)
  (S P M : ℝ)
  (sum_eq : S = 9)
  (product_eq : P = 81 / 4)
  (sum_of_terms : S = a1 * (1 - q^4) / (1 - q))
  (product_of_terms : P = a1 * a1 * q * q * (a1*q*q) * (q*a1) )
  (sum_of_reciprocals : M = (q^4 - 1) / (a1 * (q^4 - q^3)))
  : M = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_geometric_sequence_is_two_l2089_208934


namespace NUMINAMATH_GPT_correct_answers_proof_l2089_208964

variable (n p q s c : ℕ)
variable (total_questions points_per_correct penalty_per_wrong total_score correct_answers : ℕ)

def num_questions := 20
def points_correct := 5
def penalty_wrong := 1
def total_points := 76

theorem correct_answers_proof :
  (total_questions * points_per_correct - (total_questions - correct_answers) * penalty_wrong) = total_points →
  correct_answers = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_answers_proof_l2089_208964


namespace NUMINAMATH_GPT_intersection_eq_l2089_208975

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem intersection_eq : M ∩ N = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l2089_208975


namespace NUMINAMATH_GPT_find_interest_rate_l2089_208967

theorem find_interest_rate (P r : ℝ) 
  (h1 : 100 = P * (1 + 2 * r)) 
  (h2 : 200 = P * (1 + 6 * r)) : 
  r = 0.5 :=
sorry

end NUMINAMATH_GPT_find_interest_rate_l2089_208967


namespace NUMINAMATH_GPT_train2_length_is_230_l2089_208946

noncomputable def train_length_proof : Prop :=
  let speed1_kmph := 120
  let speed2_kmph := 80
  let length_train1 := 270
  let time_cross := 9
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time_cross
  let length_train2 := total_distance - length_train1
  length_train2 = 230

theorem train2_length_is_230 : train_length_proof :=
  by
    sorry

end NUMINAMATH_GPT_train2_length_is_230_l2089_208946
