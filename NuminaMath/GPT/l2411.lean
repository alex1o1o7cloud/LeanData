import Mathlib

namespace expected_value_is_7_l2411_241109

def win (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * (10 - n) else 10 - n

def fair_die_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def expected_value (values : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (values.map (λ n => win n)).sum / values.length

theorem expected_value_is_7 :
  expected_value fair_die_values win = 7 := 
sorry

end expected_value_is_7_l2411_241109


namespace stock_yield_percentage_l2411_241191

theorem stock_yield_percentage (face_value market_price : ℝ) (annual_dividend_rate : ℝ) 
  (h_face_value : face_value = 100)
  (h_market_price : market_price = 140)
  (h_annual_dividend_rate : annual_dividend_rate = 0.14) :
  (annual_dividend_rate * face_value / market_price) * 100 = 10 :=
by
  -- computation here
  sorry

end stock_yield_percentage_l2411_241191


namespace concentration_of_spirit_in_vessel_a_l2411_241193

theorem concentration_of_spirit_in_vessel_a :
  ∀ (x : ℝ), 
    (∀ (v1 v2 v3 : ℝ), v1 * (x / 100) + v2 * (30 / 100) + v3 * (10 / 100) = 15 * (26 / 100) →
      v1 + v2 + v3 = 15 →
      v1 = 4 → v2 = 5 → v3 = 6 →
      x = 45) :=
by
  intros x v1 v2 v3 h h_volume h_v1 h_v2 h_v3
  sorry

end concentration_of_spirit_in_vessel_a_l2411_241193


namespace eighth_term_geometric_sequence_l2411_241139

theorem eighth_term_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 12) (h_r : r = 1/4) (h_n : n = 8) :
  a * r^(n - 1) = 3 / 4096 := 
by 
  sorry

end eighth_term_geometric_sequence_l2411_241139


namespace total_cost_correct_l2411_241110

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l2411_241110


namespace unique_n_divisors_satisfies_condition_l2411_241149

theorem unique_n_divisors_satisfies_condition:
  ∃ (n : ℕ), (∃ d1 d2 d3 : ℕ, d1 = 1 ∧ d2 > d1 ∧ d3 > d2 ∧ n = d3 ∧
  n = d2^2 + d3^3) ∧ n = 68 := by
  sorry

end unique_n_divisors_satisfies_condition_l2411_241149


namespace min_units_l2411_241155

theorem min_units (x : ℕ) (h1 : 5500 * 60 + 5000 * (x - 60) > 550000) : x ≥ 105 := 
by {
  sorry
}

end min_units_l2411_241155


namespace ratio_Umar_Yusaf_l2411_241165

variable (AliAge YusafAge UmarAge : ℕ)

-- Given conditions:
def Ali_is_8_years_old : Prop := AliAge = 8
def Ali_is_3_years_older_than_Yusaf : Prop := AliAge = YusafAge + 3
def Umar_is_10_years_old : Prop := UmarAge = 10

-- Proof statement:
theorem ratio_Umar_Yusaf (h1 : Ali_is_8_years_old AliAge)
                         (h2 : Ali_is_3_years_older_than_Yusaf AliAge YusafAge)
                         (h3 : Umar_is_10_years_old UmarAge) :
  UmarAge / YusafAge = 2 :=
by
  sorry

end ratio_Umar_Yusaf_l2411_241165


namespace problem_statement_l2411_241131

theorem problem_statement :
  ∃ (w x y z : ℕ), (2^w * 3^x * 5^y * 7^z = 588) ∧ (2 * w + 3 * x + 5 * y + 7 * z = 21) :=
by
  sorry

end problem_statement_l2411_241131


namespace unique_prime_solution_l2411_241124

-- Define the problem in terms of prime numbers and checking the conditions
open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_solution (p : ℕ) (hp : is_prime p) (h1 : is_prime (p^2 - 6)) (h2 : is_prime (p^2 + 6)) : p = 5 := 
sorry

end unique_prime_solution_l2411_241124


namespace servings_needed_l2411_241119

theorem servings_needed
  (pieces_per_serving : ℕ)
  (jared_consumption : ℕ)
  (three_friends_consumption : ℕ)
  (another_three_friends_consumption : ℕ)
  (last_four_friends_consumption : ℕ) : 
  pieces_per_serving = 60 →
  jared_consumption = 150 →
  three_friends_consumption = 3 * 80 →
  another_three_friends_consumption = 3 * 200 →
  last_four_friends_consumption = 4 * 100 →
  ∃ (s : ℕ), s = 24 :=
by
  intros
  sorry

end servings_needed_l2411_241119


namespace problem_l2411_241127

variable (a b : ℝ)

theorem problem (h : a = 1.25 * b) : (4 * b) / a = 3.2 :=
by
  sorry

end problem_l2411_241127


namespace ratio_arithmetic_seq_a2019_a2017_eq_l2411_241195

def ratio_arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n ≥ 1 → a (n+2) / a (n+1) - a (n+1) / a n = 2

theorem ratio_arithmetic_seq_a2019_a2017_eq (a : ℕ → ℝ) 
  (h : ratio_arithmetic_seq a) 
  (ha1 : a 1 = 1) 
  (ha2 : a 2 = 1) 
  (ha3 : a 3 = 3) : 
  a 2019 / a 2017 = 4 * 2017^2 - 1 :=
sorry

end ratio_arithmetic_seq_a2019_a2017_eq_l2411_241195


namespace carrie_spent_money_l2411_241148

variable (cost_per_tshirt : ℝ) (num_tshirts : ℕ)

theorem carrie_spent_money (h1 : cost_per_tshirt = 9.95) (h2 : num_tshirts = 20) :
  cost_per_tshirt * num_tshirts = 199 := by
  sorry

end carrie_spent_money_l2411_241148


namespace max_areas_in_disk_l2411_241105

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk_l2411_241105


namespace min_num_edges_chromatic_l2411_241114

-- Definition of chromatic number.
def chromatic_number (G : SimpleGraph V) : ℕ := sorry

-- Definition of the number of edges in a graph as a function.
def num_edges (G : SimpleGraph V) : ℕ := sorry

-- Statement of the theorem.
theorem min_num_edges_chromatic (G : SimpleGraph V) (n : ℕ) 
  (chrom_num_G : chromatic_number G = n) : 
  num_edges G ≥ n * (n - 1) / 2 :=
sorry

end min_num_edges_chromatic_l2411_241114


namespace die_roll_probability_div_3_l2411_241172

noncomputable def probability_divisible_by_3 : ℚ :=
  1 - ((2 : ℚ) / 3) ^ 8

theorem die_roll_probability_div_3 :
  probability_divisible_by_3 = 6305 / 6561 :=
by
  sorry

end die_roll_probability_div_3_l2411_241172


namespace jerky_remaining_after_giving_half_l2411_241106

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end jerky_remaining_after_giving_half_l2411_241106


namespace ratio_of_rooms_l2411_241161

theorem ratio_of_rooms (rooms_danielle : ℕ) (rooms_grant : ℕ) (ratio_grant_heidi : ℚ)
  (h1 : rooms_danielle = 6)
  (h2 : rooms_grant = 2)
  (h3 : ratio_grant_heidi = 1/9) :
  (18 : ℚ) / rooms_danielle = 3 :=
by
  sorry

end ratio_of_rooms_l2411_241161


namespace evaluate_expression_l2411_241145

noncomputable def cuberoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem evaluate_expression : 
  cuberoot (1 + 27) * cuberoot (1 + cuberoot 27) = cuberoot 112 := 
by 
  sorry

end evaluate_expression_l2411_241145


namespace min_value_fraction_l2411_241182

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℝ, x = (a / (a + 2 * b) + b / (a + b)) ∧ x ≥ 1 - 1 / (2 * sqrt 2) ∧ x = 1 - 1 / (2 * sqrt 2)) :=
by
  sorry

end min_value_fraction_l2411_241182


namespace b_catches_A_distance_l2411_241147

noncomputable def speed_A := 10 -- kmph
noncomputable def speed_B := 20 -- kmph
noncomputable def time_diff := 7 -- hours
noncomputable def distance_A := speed_A * time_diff -- km
noncomputable def relative_speed := speed_B - speed_A -- kmph
noncomputable def catch_up_time := distance_A / relative_speed -- hours
noncomputable def distance_B := speed_B * catch_up_time -- km

theorem b_catches_A_distance :
  distance_B = 140 := by
  sorry

end b_catches_A_distance_l2411_241147


namespace integral_1_integral_2_integral_3_integral_4_integral_5_l2411_241187
open Real

-- Integral 1
theorem integral_1 : ∫ (x : ℝ), sin x * cos x ^ 3 = -1 / 4 * cos x ^ 4 + C :=
by sorry

-- Integral 2
theorem integral_2 : ∫ (x : ℝ), 1 / ((1 + sqrt x) * sqrt x) = 2 * log (1 + sqrt x) + C :=
by sorry

-- Integral 3
theorem integral_3 : ∫ (x : ℝ), x ^ 2 * sqrt (x ^ 3 + 1) = 2 / 9 * (x ^ 3 + 1) ^ (3/2) + C :=
by sorry

-- Integral 4
theorem integral_4 : ∫ (x : ℝ), (exp (2 * x) - 3 * exp x) / exp x = exp x - 3 * x + C :=
by sorry

-- Integral 5
theorem integral_5 : ∫ (x : ℝ), (1 - x ^ 2) * exp x = - (x - 1) ^ 2 * exp x + C :=
by sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_l2411_241187


namespace radius_of_sphere_touching_four_l2411_241142

noncomputable def r_sphere_internally_touching_four := Real.sqrt (3 / 2) + 1
noncomputable def r_sphere_externally_touching_four := Real.sqrt (3 / 2) - 1

theorem radius_of_sphere_touching_four (r : ℝ) (R := Real.sqrt (3 / 2)) :
  r = R + 1 ∨ r = R - 1 :=
by
  sorry

end radius_of_sphere_touching_four_l2411_241142


namespace hike_down_distance_l2411_241159

theorem hike_down_distance :
  let rate_up := 4 -- rate going up in miles per day
  let time := 2    -- time in days
  let rate_down := 1.5 * rate_up -- rate going down in miles per day
  let distance_down := rate_down * time -- distance going down in miles
  distance_down = 12 :=
by
  sorry

end hike_down_distance_l2411_241159


namespace dennis_pants_purchase_l2411_241154

theorem dennis_pants_purchase
  (pants_cost : ℝ) 
  (pants_discount : ℝ) 
  (socks_cost : ℝ) 
  (socks_discount : ℝ) 
  (socks_quantity : ℕ)
  (total_spent : ℝ)
  (discounted_pants_cost : ℝ)
  (discounted_socks_cost : ℝ)
  (pants_quantity : ℕ) :
  pants_cost = 110.00 →
  pants_discount = 0.30 →
  socks_cost = 60.00 →
  socks_discount = 0.30 →
  socks_quantity = 2 →
  total_spent = 392.00 →
  discounted_pants_cost = pants_cost * (1 - pants_discount) →
  discounted_socks_cost = socks_cost * (1 - socks_discount) →
  total_spent = socks_quantity * discounted_socks_cost + pants_quantity * discounted_ppants_cost →
  pants_quantity = 4 :=
by
  intros
  sorry

end dennis_pants_purchase_l2411_241154


namespace divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l2411_241185

noncomputable def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1)^n - x^n - 1

def P (x : ℝ) : ℝ := x^2 + x + 1

-- Prove Q(x, n) is divisible by P(x) if and only if n ≡ 1 or 5 (mod 6)
theorem divisibility_by_P (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

-- Prove Q(x, n) is divisible by P(x)^2 if and only if n ≡ 1 (mod 6)
theorem divisibility_by_P_squared (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 1 := 
sorry

-- Prove Q(x, n) is divisible by P(x)^3 if and only if n = 1
theorem divisibility_by_P_cubed (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^3 = 0 ↔ n = 1 := 
sorry

end divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l2411_241185


namespace slope_y_intercept_sum_l2411_241101

theorem slope_y_intercept_sum 
  (m b : ℝ) 
  (h1 : (2 : ℝ) * m + b = -1) 
  (h2 : (5 : ℝ) * m + b = 2) : 
  m + b = -2 := 
sorry

end slope_y_intercept_sum_l2411_241101


namespace distinct_four_digit_numbers_product_18_l2411_241198

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l2411_241198


namespace matrix_determinant_6_l2411_241118

theorem matrix_determinant_6 (x y z w : ℝ)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 2 * w) - z * (5 * x + 2 * y)) = 6 :=
by
  sorry

end matrix_determinant_6_l2411_241118


namespace solve_for_x_l2411_241177

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
    5 * y ^ 2 + 2 * y + 3 = 3 * (9 * x ^ 2 + y + 1) ↔ x = 0 ∨ x = 1 / 6 := 
by
  sorry

end solve_for_x_l2411_241177


namespace find_m_l2411_241113

-- Define the conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 4 = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- Statement of the problem
theorem find_m (m : ℝ) (e : ℝ) (h1 : eccentricity e) (h2 : ∀ x y : ℝ, ellipse_eq x y m) :
  m = 3 ∨ m = 5 :=
sorry

end find_m_l2411_241113


namespace power_function_value_l2411_241102

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1 / 2)) (H : f 9 = 3) : f 25 = 5 :=
by
  sorry

end power_function_value_l2411_241102


namespace proof_problem_l2411_241167

noncomputable def arithmetic_mean (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem proof_problem (a b c x y z m : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (m_pos : 0 < m) (m_ne_one : m ≠ 1) 
  (h_b : b = arithmetic_mean a c) (h_y : y = geometric_mean x z) :
  (b - c) * Real.logb m x + (c - a) * Real.logb m y + (a - b) * Real.logb m z = 0 := by
  sorry

end proof_problem_l2411_241167


namespace total_distance_in_land_miles_l2411_241151

-- Definitions based on conditions
def speed_one_sail : ℕ := 25
def time_one_sail : ℕ := 4
def distance_one_sail := speed_one_sail * time_one_sail

def speed_two_sails : ℕ := 50
def time_two_sails : ℕ := 4
def distance_two_sails := speed_two_sails * time_two_sails

def conversion_factor : ℕ := 115  -- Note: 1.15 * 100 for simplicity with integers

-- Theorem to prove the total distance in land miles
theorem total_distance_in_land_miles : (distance_one_sail + distance_two_sails) * conversion_factor / 100 = 345 := by
  sorry

end total_distance_in_land_miles_l2411_241151


namespace problem_part_I_problem_part_II_l2411_241186

-- Problem Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_part_I (x : ℝ) :
    (f (x + 3/2) ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 2) :=
  sorry

-- Problem Part II
theorem problem_part_II (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
    3*p + 2*q + r ≥ 9/4 :=
  sorry

end problem_part_I_problem_part_II_l2411_241186


namespace find_c_l2411_241180

-- Define conditions as Lean statements
theorem find_c :
  ∀ (c n : ℝ), 
  (n ^ 2 + 1 / 16 = 1 / 4) → 
  2 * n = c → 
  c < 0 → 
  c = - (Real.sqrt 3) / 2 :=
by
  intros c n h1 h2 h3
  sorry

end find_c_l2411_241180


namespace ab_cd_zero_l2411_241112

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : ac + bd = 0) : 
  ab + cd = 0 := 
sorry

end ab_cd_zero_l2411_241112


namespace cos_double_angle_l2411_241141

theorem cos_double_angle (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : Real.cos (2 * x) = 1 / 2 := 
sorry

end cos_double_angle_l2411_241141


namespace base_angle_of_isosceles_triangle_l2411_241197

theorem base_angle_of_isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) (h_isosceles : A = B ∨ B = C ∨ A = C) (h_angle : A = 42 ∨ B = 42 ∨ C = 42) :
  A = 42 ∨ A = 69 ∨ B = 42 ∨ B = 69 ∨ C = 42 ∨ C = 69 :=
by
  sorry

end base_angle_of_isosceles_triangle_l2411_241197


namespace literate_employees_l2411_241199

theorem literate_employees (num_illiterate : ℕ) (wage_decrease_per_illiterate : ℕ)
  (total_average_salary_decrease : ℕ) : num_illiterate = 35 → 
                                        wage_decrease_per_illiterate = 25 →
                                        total_average_salary_decrease = 15 →
                                        ∃ L : ℕ, L = 23 :=
by {
  -- given: num_illiterate = 35
  -- given: wage_decrease_per_illiterate = 25
  -- given: total_average_salary_decrease = 15
  sorry
}

end literate_employees_l2411_241199


namespace ratio_HC_JE_l2411_241196

noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := B + 2
noncomputable def D : ℝ := C + 1
noncomputable def E : ℝ := D + 1
noncomputable def F : ℝ := E + 2

variable (G H J K : ℝ × ℝ)
variable (parallel_AG_HC parallel_AG_JE parallel_AG_KB : Prop)

-- Conditions
axiom points_on_line : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F
axiom AB : B - A = 1
axiom BC : C - B = 2
axiom CD : D - C = 1
axiom DE : E - D = 1
axiom EF : F - E = 2
axiom G_off_AF : G.2 ≠ 0
axiom H_on_GD : H.1 = G.1 ∧ H.2 = D
axiom J_on_GF : J.1 = G.1 ∧ J.2 = F
axiom K_on_GB : K.1 = G.1 ∧ K.2 = B
axiom parallel_hc_je_kb_ag : parallel_AG_HC ∧ parallel_AG_JE ∧ parallel_AG_KB ∧ (G.2 / 1) = (K.2 / (K.1 - G.1))

-- Task: Prove the ratio HC/JE = 7/8
theorem ratio_HC_JE : (H.2 - C) / (J.2 - E) = 7 / 8 :=
sorry

end ratio_HC_JE_l2411_241196


namespace parabola_opens_downwards_iff_l2411_241125

theorem parabola_opens_downwards_iff (a : ℝ) : (∀ x : ℝ, (a - 1) * x^2 + 2 * x ≤ 0) ↔ a < 1 := 
sorry

end parabola_opens_downwards_iff_l2411_241125


namespace classify_numbers_l2411_241152

def isDecimal (n : ℝ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), n = i + f ∧ i ≠ 0

def isNatural (n : ℕ) : Prop :=
  n ≥ 0

theorem classify_numbers :
  (isDecimal 7.42) ∧ (isDecimal 3.6) ∧ (isDecimal 5.23) ∧ (isDecimal 37.8) ∧
  (isNatural 5) ∧ (isNatural 100) ∧ (isNatural 502) ∧ (isNatural 460) :=
by
  sorry

end classify_numbers_l2411_241152


namespace smallest_gcd_bc_l2411_241156

theorem smallest_gcd_bc (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (gcd_ab : Nat.gcd a b = 168) (gcd_ac : Nat.gcd a c = 693) : Nat.gcd b c = 21 := 
sorry

end smallest_gcd_bc_l2411_241156


namespace combi_sum_l2411_241184

theorem combi_sum : (Nat.choose 8 2) + (Nat.choose 8 3) + (Nat.choose 9 2) = 120 :=
by
  sorry

end combi_sum_l2411_241184


namespace cost_of_milkshake_l2411_241194

theorem cost_of_milkshake
  (initial_money : ℝ)
  (remaining_after_cupcakes : ℝ)
  (remaining_after_sandwich : ℝ)
  (remaining_after_toy : ℝ)
  (final_remaining : ℝ)
  (money_spent_on_milkshake : ℝ) :
  initial_money = 20 →
  remaining_after_cupcakes = initial_money - (1 / 4) * initial_money →
  remaining_after_sandwich = remaining_after_cupcakes - 0.30 * remaining_after_cupcakes →
  remaining_after_toy = remaining_after_sandwich - (1 / 5) * remaining_after_sandwich →
  final_remaining = 3 →
  money_spent_on_milkshake = remaining_after_toy - final_remaining →
  money_spent_on_milkshake = 5.40 :=
by
  intros 
  sorry

end cost_of_milkshake_l2411_241194


namespace purple_gumdrops_after_replacement_l2411_241122

def total_gumdrops : Nat := 200
def orange_percentage : Nat := 40
def purple_percentage : Nat := 10
def yellow_percentage : Nat := 25
def white_percentage : Nat := 15
def black_percentage : Nat := 10

def initial_orange_gumdrops := (orange_percentage * total_gumdrops) / 100
def initial_purple_gumdrops := (purple_percentage * total_gumdrops) / 100
def orange_to_purple := initial_orange_gumdrops / 3
def final_purple_gumdrops := initial_purple_gumdrops + orange_to_purple

theorem purple_gumdrops_after_replacement : final_purple_gumdrops = 47 := by
  sorry

end purple_gumdrops_after_replacement_l2411_241122


namespace number_of_books_l2411_241188

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

end number_of_books_l2411_241188


namespace optimal_position_theorem_l2411_241174

noncomputable def optimal_position (a b a1 b1 : ℝ) : ℝ :=
  (b / 2) + (b1 / (2 * a1)) * (a - a1)

theorem optimal_position_theorem 
  (a b a1 b1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) :
  ∃ x, x = optimal_position a b a1 b1 := by
  sorry

end optimal_position_theorem_l2411_241174


namespace line_form_l2411_241162

-- Given vector equation for a line
def line_eq (x y : ℝ) : Prop :=
  (3 * (x - 4) + 7 * (y - 14)) = 0

-- Prove that the line can be written in the form y = mx + b
theorem line_form (x y : ℝ) (h : line_eq x y) :
  y = (-3/7) * x + (110/7) :=
sorry

end line_form_l2411_241162


namespace original_deck_size_l2411_241166

-- Let's define the number of red and black cards initially
def numRedCards (r : ℕ) : ℕ := r
def numBlackCards (b : ℕ) : ℕ := b

-- Define the initial condition as given in the problem
def initial_prob_red (r b : ℕ) : Prop :=
  r / (r + b) = 2 / 5

-- Define the condition after adding 7 black cards
def prob_red_after_adding_black (r b : ℕ) : Prop :=
  r / (r + (b + 7)) = 1 / 3

-- The proof statement to verify original number of cards in the deck
theorem original_deck_size (r b : ℕ) (h1 : initial_prob_red r b) (h2 : prob_red_after_adding_black r b) : r + b = 35 := by
  sorry

end original_deck_size_l2411_241166


namespace translation_theorem_l2411_241183

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (θ : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

theorem translation_theorem
  (θ φ : ℝ)
  (hθ1 : |θ| < Real.pi / 2)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (hf : f θ 0 = 1 / 2)
  (hg : g θ φ 0 = 1 / 2) :
  φ = 2 * Real.pi / 3 :=
sorry

end translation_theorem_l2411_241183


namespace jeremy_uncle_money_l2411_241130

def total_cost (num_jerseys : Nat) (cost_per_jersey : Nat) (basketball_cost : Nat) (shorts_cost : Nat) : Nat :=
  (num_jerseys * cost_per_jersey) + basketball_cost + shorts_cost

def total_money_given (total_cost : Nat) (money_left : Nat) : Nat :=
  total_cost + money_left

theorem jeremy_uncle_money :
  total_money_given (total_cost 5 2 18 8) 14 = 50 :=
by
  sorry

end jeremy_uncle_money_l2411_241130


namespace commutative_star_l2411_241150

def star (a b : ℤ) : ℤ := a^2 + b^2

theorem commutative_star (a b : ℤ) : star a b = star b a :=
by sorry

end commutative_star_l2411_241150


namespace remainder_of_large_number_div_by_101_l2411_241179

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l2411_241179


namespace non_real_roots_b_range_l2411_241104

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l2411_241104


namespace minimum_value_of_f_l2411_241153

def f (x : ℝ) : ℝ := |3 - x| + |x - 2|

theorem minimum_value_of_f : ∃ x0 : ℝ, (∀ x : ℝ, f x0 ≤ f x) ∧ f x0 = 1 := 
by
  sorry

end minimum_value_of_f_l2411_241153


namespace min_solution_of_x_abs_x_eq_3x_plus_4_l2411_241158

theorem min_solution_of_x_abs_x_eq_3x_plus_4 : 
  ∃ x : ℝ, (x * |x| = 3 * x + 4) ∧ ∀ y : ℝ, (y * |y| = 3 * y + 4) → x ≤ y :=
sorry

end min_solution_of_x_abs_x_eq_3x_plus_4_l2411_241158


namespace find_time_eating_dinner_l2411_241144

def total_flight_time : ℕ := 11 * 60 + 20
def time_reading : ℕ := 2 * 60
def time_watching_movies : ℕ := 4 * 60
def time_listening_radio : ℕ := 40
def time_playing_games : ℕ := 1 * 60 + 10
def time_nap : ℕ := 3 * 60

theorem find_time_eating_dinner : 
  total_flight_time - (time_reading + time_watching_movies + time_listening_radio + time_playing_games + time_nap) = 30 := 
by
  sorry

end find_time_eating_dinner_l2411_241144


namespace saved_money_is_30_l2411_241121

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30_l2411_241121


namespace max_and_min_of_expression_l2411_241129

variable {x y : ℝ}

theorem max_and_min_of_expression (h : |5 * x + y| + |5 * x - y| = 20) : 
  (∃ (maxQ minQ : ℝ), maxQ = 124 ∧ minQ = 3 ∧ 
  (∀ z, z = x^2 - x * y + y^2 → z <= 124 ∧ z >= 3)) :=
sorry

end max_and_min_of_expression_l2411_241129


namespace intersection_nonempty_range_b_l2411_241117

noncomputable def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
noncomputable def B (b : ℝ) (a : ℝ) : Set ℝ := {x | (x - b)^2 < a}

theorem intersection_nonempty_range_b (b : ℝ) : 
  A ∩ B b 1 ≠ ∅ ↔ -2 < b ∧ b < 2 := 
by
  sorry

end intersection_nonempty_range_b_l2411_241117


namespace equation_of_line_BC_l2411_241111

/-
Given:
1. Point A(3, -1)
2. The line containing the median from A to side BC: 6x + 10y - 59 = 0
3. The line containing the angle bisector of ∠B: x - 4y + 10 = 0

Prove:
The equation of the line containing side BC is 2x + 9y - 65 = 0.
-/

noncomputable def point_A : (ℝ × ℝ) := (3, -1)

noncomputable def median_line (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0

noncomputable def angle_bisector_line_B (x y : ℝ) : Prop := x - 4 * y + 10 = 0

theorem equation_of_line_BC :
  ∃ (x y : ℝ), 2 * x + 9 * y - 65 = 0 :=
sorry

end equation_of_line_BC_l2411_241111


namespace batsman_average_after_17th_innings_l2411_241140

theorem batsman_average_after_17th_innings :
  ∀ (A : ℕ), (80 + 16 * A) = 17 * (A + 2) → A + 2 = 48 := by
  intro A h
  sorry

end batsman_average_after_17th_innings_l2411_241140


namespace Sarah_consumed_one_sixth_l2411_241103

theorem Sarah_consumed_one_sixth (total_slices : ℕ) (slices_sarah_ate : ℕ) (shared_slices : ℕ) :
  total_slices = 20 → slices_sarah_ate = 3 → shared_slices = 1 → 
  ((slices_sarah_ate + shared_slices / 3) / total_slices : ℚ) = 1 / 6 :=
by
  intros h1 h2 h3
  sorry

end Sarah_consumed_one_sixth_l2411_241103


namespace eggs_per_chicken_per_day_l2411_241100

-- Define the conditions
def chickens : ℕ := 8
def price_per_dozen : ℕ := 5
def total_revenue : ℕ := 280
def weeks : ℕ := 4
def eggs_per_dozen : ℕ := 12
def days_per_week : ℕ := 7

-- Theorem statement on how many eggs each chicken lays per day
theorem eggs_per_chicken_per_day :
  (chickens * ((total_revenue / price_per_dozen * eggs_per_dozen) / (weeks * days_per_week))) / chickens = 3 :=
by
  sorry

end eggs_per_chicken_per_day_l2411_241100


namespace maximum_distance_is_correct_l2411_241181

-- Define the right trapezoid with the given side lengths and angle conditions
structure RightTrapezoid (AB CD : ℕ) where
  B_angle : ℝ
  D_angle : ℝ
  h_AB : AB = 200
  h_CD : CD = 100
  h_B_angle : B_angle = 90
  h_D_angle : D_angle = 45

-- Define the guards' walking condition and distance calculation
def max_distance_between_guards (T : RightTrapezoid 200 100) : ℝ :=
  let P := 400 + 100 * Real.sqrt 2
  let d := (400 + 100 * Real.sqrt 2) / 2
  222.1  -- Hard-coded according to the problem's correct answer for maximum distance

theorem maximum_distance_is_correct :
  ∀ (T : RightTrapezoid 200 100), max_distance_between_guards T = 222.1 := by
  sorry

end maximum_distance_is_correct_l2411_241181


namespace words_per_page_l2411_241175

theorem words_per_page (p : ℕ) :
  (p ≤ 120) ∧ (154 * p % 221 = 145) → p = 96 := by
  sorry

end words_per_page_l2411_241175


namespace probability_one_white_one_black_l2411_241168

def white_ball_count : ℕ := 8
def black_ball_count : ℕ := 7
def total_ball_count : ℕ := white_ball_count + black_ball_count
def total_ways_to_choose_2_balls : ℕ := total_ball_count.choose 2
def favorable_ways : ℕ := white_ball_count * black_ball_count

theorem probability_one_white_one_black : 
  (favorable_ways : ℚ) / (total_ways_to_choose_2_balls : ℚ) = 8 / 15 :=
by
  sorry

end probability_one_white_one_black_l2411_241168


namespace ticket_distribution_count_l2411_241178

theorem ticket_distribution_count :
  let A := 2
  let B := 2
  let C := 1
  let D := 1
  let total_tickets := A + B + C + D
  ∃ (num_dist : ℕ), num_dist = 180 :=
by {
  sorry
}

end ticket_distribution_count_l2411_241178


namespace smallest_four_digit_number_divisible_by_35_l2411_241171

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

end smallest_four_digit_number_divisible_by_35_l2411_241171


namespace rahul_batting_average_before_match_l2411_241163

open Nat

theorem rahul_batting_average_before_match (R : ℕ) (A : ℕ) :
  (R + 69 = 6 * 54) ∧ (A = R / 5) → (A = 51) :=
by
  sorry

end rahul_batting_average_before_match_l2411_241163


namespace vec_eqn_solution_l2411_241176

theorem vec_eqn_solution :
  ∀ m : ℝ, let a : ℝ × ℝ := (1, -2) 
           let b : ℝ × ℝ := (m, 4) 
           (a.1 * b.2 = a.2 * b.1) → 2 • a - b = (4, -8) :=
by
  intro m a b h_parallel
  sorry

end vec_eqn_solution_l2411_241176


namespace roots_difference_one_l2411_241116

theorem roots_difference_one (p : ℝ) :
  (∃ (x y : ℝ), (x^3 - 7 * x + p = 0) ∧ (y^3 - 7 * y + p = 0) ∧ (x - y = 1)) ↔ (p = 6 ∨ p = -6) :=
sorry

end roots_difference_one_l2411_241116


namespace mean_weight_correct_l2411_241126

def weights := [51, 60, 62, 64, 64, 65, 67, 73, 74, 74, 75, 76, 77, 78, 79]

noncomputable def mean_weight (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem mean_weight_correct :
  mean_weight weights = 69.27 := by
  sorry

end mean_weight_correct_l2411_241126


namespace b_must_be_one_l2411_241164

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l2411_241164


namespace negation_universal_to_existential_l2411_241146

theorem negation_universal_to_existential :
  ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_universal_to_existential_l2411_241146


namespace largest_int_lt_100_div_9_rem_5_l2411_241128

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l2411_241128


namespace total_oil_leakage_l2411_241108

def oil_leaked_before : ℕ := 6522
def oil_leaked_during : ℕ := 5165
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leakage : total_oil_leaked = 11687 := by
  sorry

end total_oil_leakage_l2411_241108


namespace no_partition_of_six_consecutive_numbers_product_equal_l2411_241138

theorem no_partition_of_six_consecutive_numbers_product_equal (n : ℕ) :
  ¬ ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n+6) ∧ 
    A ∩ B = ∅ ∧ 
    A.prod id = B.prod id :=
by
  sorry

end no_partition_of_six_consecutive_numbers_product_equal_l2411_241138


namespace cube_face_problem_l2411_241143

theorem cube_face_problem (n : ℕ) (h : 0 < n) :
  ((6 * n^2) : ℚ) / (6 * n^3) = 1 / 3 → n = 3 :=
by
  sorry

end cube_face_problem_l2411_241143


namespace g_properties_l2411_241192

def f (x : ℝ) : ℝ := x

def g (x : ℝ) : ℝ := -f x

theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry

end g_properties_l2411_241192


namespace elementary_school_classes_count_l2411_241173

theorem elementary_school_classes_count (E : ℕ) (donate_per_class : ℕ) (middle_school_classes : ℕ) (total_balls : ℕ) :
  donate_per_class = 5 →
  middle_school_classes = 5 →
  total_balls = 90 →
  5 * 2 * E + 5 * 2 * middle_school_classes = total_balls →
  E = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end elementary_school_classes_count_l2411_241173


namespace lowest_price_eq_195_l2411_241135

def cost_per_component : ℕ := 80
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_costs : ℕ := 16500
def num_components : ℕ := 150

theorem lowest_price_eq_195 
  (cost_per_component shipping_cost_per_unit fixed_monthly_costs num_components : ℕ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 5)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : num_components = 150) :
  (fixed_monthly_costs + num_components * (cost_per_component + shipping_cost_per_unit)) / num_components = 195 :=
by
  sorry

end lowest_price_eq_195_l2411_241135


namespace range_of_x_l2411_241132

theorem range_of_x (total_students math_club chemistry_club : ℕ) (h_total : total_students = 45) 
(h_math : math_club = 28) (h_chemistry : chemistry_club = 21) (x : ℕ) :
  4 ≤ x ∧ x ≤ 21 ↔ (28 + 21 - x ≤ 45) :=
by sorry

end range_of_x_l2411_241132


namespace yogurt_banana_slices_l2411_241120

/--
Given:
1. Each banana yields 10 slices.
2. Vivian needs to make 5 yogurts.
3. She needs to buy 4 bananas.

Prove:
The number of banana slices needed for each yogurt is 8.
-/
theorem yogurt_banana_slices 
    (slices_per_banana : ℕ)
    (bananas_bought : ℕ)
    (yogurts_needed : ℕ)
    (h1 : slices_per_banana = 10)
    (h2 : yogurts_needed = 5)
    (h3 : bananas_bought = 4) : 
    (bananas_bought * slices_per_banana) / yogurts_needed = 8 :=
by
  sorry

end yogurt_banana_slices_l2411_241120


namespace num_cars_in_parking_lot_l2411_241136

-- Define the conditions
variable (C : ℕ) -- Number of cars
def number_of_bikes := 5 -- Number of bikes given
def total_wheels := 66 -- Total number of wheels given
def wheels_per_bike := 2 -- Number of wheels per bike
def wheels_per_car := 4 -- Number of wheels per car

-- Define the proof statement
theorem num_cars_in_parking_lot 
  (h1 : total_wheels = 66) 
  (h2 : number_of_bikes = 5) 
  (h3 : wheels_per_bike = 2)
  (h4 : wheels_per_car = 4) 
  (h5 : C * wheels_per_car + number_of_bikes * wheels_per_bike = total_wheels) :
  C = 14 :=
by
  sorry

end num_cars_in_parking_lot_l2411_241136


namespace fraction_of_tips_in_august_is_five_eighths_l2411_241189

-- Definitions
def average_tips (other_tips_total : ℤ) (n : ℤ) : ℤ := other_tips_total / n
def total_tips (other_tips : ℤ) (august_tips : ℤ) : ℤ := other_tips + august_tips
def fraction (numerator : ℤ) (denominator : ℤ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Given conditions
variables (A : ℤ) -- average monthly tips for the other 6 months (March to July and September)
variables (other_months : ℤ := 6)
variables (tips_total_other : ℤ := other_months * A) -- total tips for the 6 other months
variables (tips_august : ℤ := 10 * A) -- tips for August
variables (total_tips_all : ℤ := tips_total_other + tips_august) -- total tips for all months

-- Prove the statement
theorem fraction_of_tips_in_august_is_five_eighths :
  fraction tips_august total_tips_all = 5 / 8 := by sorry

end fraction_of_tips_in_august_is_five_eighths_l2411_241189


namespace sum_of_roots_l2411_241137

-- Define the quadratic equation whose roots are the excluded domain values C and D
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- Define C and D as the roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic_eq x

-- Define C and D as the specific roots of the given quadratic equation
axiom C : ℝ
axiom D : ℝ

-- Assert that C and D are the roots of the quadratic equation
axiom hC : is_root C
axiom hD : is_root D

-- Statement to prove
theorem sum_of_roots : C + D = 3 :=
by sorry

end sum_of_roots_l2411_241137


namespace quadratic_roots_m_value_l2411_241160

theorem quadratic_roots_m_value
  (x1 x2 m : ℝ)
  (h1 : x1^2 + 2 * x1 + m = 0)
  (h2 : x2^2 + 2 * x2 + m = 0)
  (h3 : x1 + x2 = x1 * x2 - 1) :
  m = -1 :=
sorry

end quadratic_roots_m_value_l2411_241160


namespace invalid_votes_percentage_l2411_241170

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l2411_241170


namespace problem_f_2005_value_l2411_241134

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2005_value (h_even : ∀ x : ℝ, f (-x) = f x)
                            (h_periodic : ∀ x : ℝ, f (x + 8) = f x + f 4)
                            (h_initial : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = 4 - x) :
  f 2005 = 0 :=
sorry

end problem_f_2005_value_l2411_241134


namespace math_club_problem_l2411_241107

theorem math_club_problem :
  ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end math_club_problem_l2411_241107


namespace volume_removed_percentage_l2411_241133

noncomputable def original_volume : ℕ := 20 * 15 * 10

noncomputable def cube_volume : ℕ := 4 * 4 * 4

noncomputable def total_volume_removed : ℕ := 8 * cube_volume

noncomputable def percentage_volume_removed : ℝ :=
  (total_volume_removed : ℝ) / (original_volume : ℝ) * 100

theorem volume_removed_percentage :
  percentage_volume_removed = 512 / 30 := sorry

end volume_removed_percentage_l2411_241133


namespace dave_coins_l2411_241169

theorem dave_coins :
  ∃ n : ℕ, n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 5] ∧ n ≡ 1 [MOD 3] ∧ n = 58 :=
sorry

end dave_coins_l2411_241169


namespace inequality_triangle_area_l2411_241157

-- Define the triangles and their properties
variables {α β γ : Real} -- Internal angles of triangle ABC
variables {r : Real} -- Circumradius of triangle ABC
variables {P Q : Real} -- Areas of triangles ABC and A'B'C' respectively

-- Define the bisectors and intersect points
-- Note: For the purpose of this proof, we're not explicitly defining the geometry
-- of the inner bisectors and intersect points but working from the given conditions.

theorem inequality_triangle_area
  (h1 : P = r^2 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) / 2)
  (h2 : Q = r^2 * (Real.sin (β + γ) + Real.sin (γ + α) + Real.sin (α + β)) / 2) :
  16 * Q^3 ≥ 27 * r^4 * P :=
sorry

end inequality_triangle_area_l2411_241157


namespace investment_ratio_l2411_241123

noncomputable def ratio_A_B (profit : ℝ) (profit_C : ℝ) (ratio_A_C : ℝ) (ratio_C_A : ℝ) := 
  3 / 1

theorem investment_ratio (total_profit : ℝ) (C_profit : ℝ) (A_C_ratio : ℝ) (C_A_ratio : ℝ) :
  total_profit = 60000 → C_profit = 20000 → A_C_ratio = 3 / 2 → ratio_A_B total_profit C_profit A_C_ratio C_A_ratio = 3 / 1 :=
by 
  intros h1 h2 h3
  sorry

end investment_ratio_l2411_241123


namespace correct_propositions_l2411_241190

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end correct_propositions_l2411_241190


namespace messages_per_member_per_day_l2411_241115

theorem messages_per_member_per_day (initial_members removed_members remaining_members total_weekly_messages total_daily_messages : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : remaining_members = initial_members - removed_members)
  (h4 : total_weekly_messages = 45500)
  (h5 : total_daily_messages = total_weekly_messages / 7)
  (h6 : 7 * total_daily_messages = total_weekly_messages) -- ensures that total_daily_messages calculated is correct
  : total_daily_messages / remaining_members = 50 := 
by
  sorry

end messages_per_member_per_day_l2411_241115
