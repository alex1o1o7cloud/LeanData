import Mathlib

namespace value_of_fraction_l2052_205278

variables {a b c : ℝ}

-- Conditions
def quadratic_has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def person_A_roots (a' b c : ℝ) : Prop :=
  b = -6 * a' ∧ c = 8 * a'

def person_B_roots (a b' c : ℝ) : Prop :=
  b' = -3 * a ∧ c = -4 * a

-- Proof Statement
theorem value_of_fraction (a b c a' b' : ℝ)
  (hnr : quadratic_has_no_real_roots a b c)
  (hA : person_A_roots a' b c)
  (hB : person_B_roots a b' c) :
  (2 * b + 3 * c) / a = 6 :=
by
  sorry

end value_of_fraction_l2052_205278


namespace pipe_individual_empty_time_l2052_205240

variable (a b c : ℝ)

noncomputable def timeToEmptyFirstPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * c + b * c - a * b)

noncomputable def timeToEmptySecondPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + b * c - a * c)

noncomputable def timeToEmptyThirdPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + a * c - b * c)

theorem pipe_individual_empty_time
  (x y z : ℝ)
  (h1 : 1 / x + 1 / y = 1 / a)
  (h2 : 1 / x + 1 / z = 1 / b)
  (h3 : 1 / y + 1 / z = 1 / c) :
  x = timeToEmptyFirstPipe a b c ∧ y = timeToEmptySecondPipe a b c ∧ z = timeToEmptyThirdPipe a b c :=
sorry

end pipe_individual_empty_time_l2052_205240


namespace number_of_red_socks_l2052_205222

-- Definitions:
def red_sock_pairs (R : ℕ) := R
def red_sock_cost (R : ℕ) := 3 * R
def blue_socks_pairs : ℕ := 6
def blue_sock_cost : ℕ := 5
def total_amount_spent := 42

-- Proof Statement
theorem number_of_red_socks (R : ℕ) (h : red_sock_cost R + blue_socks_pairs * blue_sock_cost = total_amount_spent) : 
  red_sock_pairs R = 4 :=
by 
  sorry

end number_of_red_socks_l2052_205222


namespace fruit_seller_l2052_205295

theorem fruit_seller (A P : ℝ) (h1 : A = 700) (h2 : A * (100 - P) / 100 = 420) : P = 40 :=
sorry

end fruit_seller_l2052_205295


namespace petes_original_number_l2052_205260

theorem petes_original_number (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y - 5) (h3 : 3 * z = 96) :
  x = 12.33 :=
by
  -- Proof goes here
  sorry

end petes_original_number_l2052_205260


namespace inequality_proof_l2052_205286

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b + b^2 / c + c^2 / a) + (a + b + c) ≥ (6 * (a^2 + b^2 + c^2) / (a + b + c)) :=
by
  sorry

end inequality_proof_l2052_205286


namespace childSupportOwed_l2052_205263

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l2052_205263


namespace ninety_percent_of_population_is_expected_number_l2052_205206

/-- Define the total population of the village -/
def total_population : ℕ := 9000

/-- Define the percentage rate as a fraction -/
def percentage_rate : ℕ := 90

/-- Define the expected number of people representing 90% of the population -/
def expected_number : ℕ := 8100

/-- The proof problem: Prove that 90% of the total population is 8100 -/
theorem ninety_percent_of_population_is_expected_number :
  (percentage_rate * total_population / 100) = expected_number :=
by
  sorry

end ninety_percent_of_population_is_expected_number_l2052_205206


namespace factorization_correct_l2052_205209

theorem factorization_correct {x : ℝ} : (x - 15)^2 = x^2 - 30*x + 225 :=
by
  sorry

end factorization_correct_l2052_205209


namespace roots_polynomial_identity_l2052_205205

theorem roots_polynomial_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a * b + b * c + c * a = 22) (h3 : a * b * c = 8) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  sorry

end roots_polynomial_identity_l2052_205205


namespace real_and_imaginary_parts_of_z_l2052_205289

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2 + i

-- State the theorem
theorem real_and_imaginary_parts_of_z :
  z.re = -1 ∧ z.im = 1 :=
by
  -- Provide the proof or placeholder
  sorry

end real_and_imaginary_parts_of_z_l2052_205289


namespace correct_probability_statement_l2052_205276

-- Define the conditions
def impossible_event_has_no_probability : Prop := ∀ (P : ℝ), P < 0 ∨ P > 0
def every_event_has_probability : Prop := ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1
def not_all_random_events_have_probability : Prop := ∃ (P : ℝ), P < 0 ∨ P > 1
def certain_events_do_not_have_probability : Prop := (∀ (P : ℝ), P ≠ 1)

-- The main theorem asserting that every event has a probability
theorem correct_probability_statement : every_event_has_probability :=
by sorry

end correct_probability_statement_l2052_205276


namespace cinema_total_cost_l2052_205246

theorem cinema_total_cost 
  (total_students : ℕ)
  (ticket_cost : ℕ)
  (half_price_interval : ℕ)
  (free_interval : ℕ)
  (half_price_cost : ℕ)
  (free_cost : ℕ)
  (total_cost : ℕ)
  (H_total_students : total_students = 84)
  (H_ticket_cost : ticket_cost = 50)
  (H_half_price_interval : half_price_interval = 12)
  (H_free_interval : free_interval = 35)
  (H_half_price_cost : half_price_cost = ticket_cost / 2)
  (H_free_cost : free_cost = 0)
  (H_total_cost : total_cost = 3925) :
  total_cost = ((total_students / half_price_interval) * half_price_cost +
                (total_students / free_interval) * free_cost +
                (total_students - (total_students / half_price_interval + total_students / free_interval)) * ticket_cost) :=
by 
  sorry

end cinema_total_cost_l2052_205246


namespace parabola_distance_l2052_205223

theorem parabola_distance (y : ℝ) (h : y ^ 2 = 24) : |-6 - 1| = 7 :=
by { sorry }

end parabola_distance_l2052_205223


namespace intersection_M_complement_N_l2052_205244

noncomputable def M := {y : ℝ | 1 ≤ y ∧ y ≤ 2}
noncomputable def N_complement := {x : ℝ | 1 ≤ x}

theorem intersection_M_complement_N : M ∩ N_complement = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_M_complement_N_l2052_205244


namespace mail_per_house_l2052_205204

theorem mail_per_house (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : 
  total_mail / total_houses = 6 := 
by 
  sorry

end mail_per_house_l2052_205204


namespace probability_of_successful_meeting_l2052_205248

noncomputable def successful_meeting_probability : ℝ :=
  let volume_hypercube := 16.0
  let volume_pyramid := (1.0/3.0) * 2.0^3 * 2.0
  let volume_reduced_base := volume_pyramid / 4.0
  let successful_meeting_volume := volume_reduced_base
  successful_meeting_volume / volume_hypercube

theorem probability_of_successful_meeting : successful_meeting_probability = 1 / 12 :=
  sorry

end probability_of_successful_meeting_l2052_205248


namespace triangle_area_l2052_205268

/-- Proof that the area of a triangle with side lengths 9 cm, 40 cm, and 41 cm is 180 square centimeters, 
    given that these lengths form a right triangle. -/
theorem triangle_area : ∀ (a b c : ℕ), a = 9 → b = 40 → c = 41 → a^2 + b^2 = c^2 → (a * b) / 2 = 180 := by
  intros a b c ha hb hc hpyth
  sorry

end triangle_area_l2052_205268


namespace floor_width_l2052_205283

theorem floor_width (tile_length tile_width floor_length max_tiles : ℕ) (h1 : tile_length = 25) (h2 : tile_width = 65) (h3 : floor_length = 150) (h4 : max_tiles = 36) :
  ∃ floor_width : ℕ, floor_width = 450 :=
by
  sorry

end floor_width_l2052_205283


namespace unique_solution_m_n_eq_l2052_205267

theorem unique_solution_m_n_eq (m n : ℕ) (h : m^2 = (10 * n + 1) * n + 2) : (m, n) = (11, 7) := by
  sorry

end unique_solution_m_n_eq_l2052_205267


namespace min_value_of_m_l2052_205215

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → x ≠ ⌊x⌋ → mx < Real.log x) ↔ m = (1 / 2) * Real.log 2 :=
by
  sorry

end min_value_of_m_l2052_205215


namespace coupon_savings_difference_l2052_205208

theorem coupon_savings_difference {P : ℝ} (hP : P > 200)
  (couponA_savings : ℝ := 0.20 * P) 
  (couponB_savings : ℝ := 50)
  (couponC_savings : ℝ := 0.30 * (P - 200)) :
  (200 ≤ P - 200 + 50 → 200 ≤ P ∧ P ≤ 200 + 400 → 600 - 250 = 350) :=
by
  sorry

end coupon_savings_difference_l2052_205208


namespace candy_division_l2052_205211

theorem candy_division (total_candy : ℕ) (students : ℕ) (per_student : ℕ) 
  (h1 : total_candy = 344) (h2 : students = 43) : 
  total_candy / students = per_student ↔ per_student = 8 := 
by 
  sorry

end candy_division_l2052_205211


namespace least_positive_three_digit_multiple_of_7_l2052_205275

theorem least_positive_three_digit_multiple_of_7 : ∃ n : ℕ, n % 7 = 0 ∧ n ≥ 100 ∧ n < 1000 ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m ≥ 100 ∧ m < 1000) → n ≤ m := 
by
  sorry

end least_positive_three_digit_multiple_of_7_l2052_205275


namespace arithmetic_expression_l2052_205282

theorem arithmetic_expression : (-9) + 18 + 2 + (-1) = 10 :=
by 
  sorry

end arithmetic_expression_l2052_205282


namespace slope_angle_of_vertical_line_l2052_205247

theorem slope_angle_of_vertical_line :
  ∀ {θ : ℝ}, (∀ x, (x = 3) → x = 3) → θ = 90 := by
  sorry

end slope_angle_of_vertical_line_l2052_205247


namespace cosine_of_third_angle_l2052_205232

theorem cosine_of_third_angle 
  (α β γ : ℝ) 
  (h1 : α < 40 * Real.pi / 180) 
  (h2 : β < 80 * Real.pi / 180) 
  (h3 : Real.sin γ = 5 / 8) :
  Real.cos γ = -Real.sqrt 39 / 8 := 
sorry

end cosine_of_third_angle_l2052_205232


namespace work_days_of_b_l2052_205217

theorem work_days_of_b (d : ℕ) 
  (A B C : ℕ)
  (h_ratioA : A = (3 * 115) / 5)
  (h_ratioB : B = (4 * 115) / 5)
  (h_C : C = 115)
  (h_total_wages : 1702 = (A * 6) + (B * d) + (C * 4)) :
  d = 9 := 
sorry

end work_days_of_b_l2052_205217


namespace total_cost_l2052_205210

/-- There are two types of discs, one costing 10.50 and another costing 8.50.
You bought a total of 10 discs, out of which 6 are priced at 8.50.
The task is to determine the total amount spent. -/
theorem total_cost (price1 price2 : ℝ) (num1 num2 : ℕ) 
  (h1 : price1 = 10.50) (h2 : price2 = 8.50) 
  (h3 : num1 = 6) (h4 : num2 = 10) 
  (h5 : num2 - num1 = 4) : 
  (num1 * price2 + (num2 - num1) * price1) = 93.00 := 
by
  sorry

end total_cost_l2052_205210


namespace nine_wolves_nine_sheep_seven_days_l2052_205243

theorem nine_wolves_nine_sheep_seven_days
    (wolves_sheep_seven_days : ∀ {n : ℕ}, 7 * n / 7 = n) :
    9 * 9 / 9 = 7 := by
  sorry

end nine_wolves_nine_sheep_seven_days_l2052_205243


namespace polar_eq_parabola_l2052_205200

/-- Prove that the curve defined by the polar equation is a parabola. -/
theorem polar_eq_parabola :
  ∀ (r θ : ℝ), r = 1 / (2 * Real.sin θ + Real.cos θ) →
    ∃ (x y : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (x + 2 * y = r^2) :=
by 
  sorry

end polar_eq_parabola_l2052_205200


namespace race_head_start_l2052_205287

theorem race_head_start (v_A v_B : ℕ) (h : v_A = 4 * v_B) (d : ℕ) : 
  100 / v_A = (100 - d) / v_B → d = 75 :=
by
  sorry

end race_head_start_l2052_205287


namespace complex_purely_imaginary_a_eq_3_l2052_205236

theorem complex_purely_imaginary_a_eq_3 (a : ℝ) :
  (∀ (a : ℝ), (a^2 - 2*a - 3) + (a + 1)*I = 0 + (a + 1)*I → a = 3) :=
by
  sorry

end complex_purely_imaginary_a_eq_3_l2052_205236


namespace max_intersections_intersections_ge_n_special_case_l2052_205231

variable {n m : ℕ}

-- Conditions: n points on a circumference, m and n are positive integers, relatively prime, 6 ≤ 2m < n
def valid_conditions (n m : ℕ) : Prop := Nat.gcd m n = 1 ∧ 6 ≤ 2 * m ∧ 2 * m < n

-- Maximum intersections I = (m-1)n
theorem max_intersections (h : valid_conditions n m) : ∃ I, I = (m - 1) * n :=
by
  sorry

-- Prove I ≥ n
theorem intersections_ge_n (h : valid_conditions n m) : ∃ I, I ≥ n :=
by
  sorry

-- Special case: m = 3 and n is even
theorem special_case (h : valid_conditions n 3) (hn : Even n) : ∃ I, I = n :=
by
  sorry

end max_intersections_intersections_ge_n_special_case_l2052_205231


namespace consecutive_integers_product_sum_l2052_205253

theorem consecutive_integers_product_sum (a b c d : ℕ) :
  a * b * c * d = 3024 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 → a + b + c + d = 30 :=
by
  sorry

end consecutive_integers_product_sum_l2052_205253


namespace christina_speed_l2052_205212

theorem christina_speed
  (d v_j v_l t : ℝ)
  (D_l : ℝ)
  (h_d : d = 360)
  (h_v_j : v_j = 5)
  (h_v_l : v_l = 12)
  (h_D_l : D_l = 360)
  (h_t : t = D_l / v_l)
  (h_distance : d = v_j * t + c * t) :
  c = 7 :=
by
  sorry

end christina_speed_l2052_205212


namespace triangle_properties_l2052_205214

theorem triangle_properties
  (K : ℝ) (α β : ℝ)
  (hK : K = 62.4)
  (hα : α = 70 + 20/60 + 40/3600)
  (hβ : β = 36 + 50/60 + 30/3600) :
  ∃ (a b T : ℝ), 
    a = 16.55 ∧
    b = 30.0 ∧
    T = 260.36 :=
by
  sorry

end triangle_properties_l2052_205214


namespace problem_statement_l2052_205270

theorem problem_statement 
  (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 945) :
  2 * w + 3 * x + 5 * y + 7 * z = 21 :=
by
  sorry

end problem_statement_l2052_205270


namespace quadratic_root_value_m_l2052_205271

theorem quadratic_root_value_m (m : ℝ) : ∃ x, x = 1 ∧ x^2 + x - m = 0 → m = 2 := by
  sorry

end quadratic_root_value_m_l2052_205271


namespace expand_expression_l2052_205279

variable (x y : ℝ)

theorem expand_expression :
  ((6 * x + 8 - 3 * y) * (4 * x - 5 * y)) = 
  (24 * x^2 - 42 * x * y + 32 * x - 40 * y + 15 * y^2) :=
by
  sorry

end expand_expression_l2052_205279


namespace combined_weight_of_boxes_l2052_205235

-- Defining the weights of each box as constants
def weight1 : ℝ := 2.5
def weight2 : ℝ := 11.3
def weight3 : ℝ := 5.75
def weight4 : ℝ := 7.2
def weight5 : ℝ := 3.25

-- The main theorem statement
theorem combined_weight_of_boxes : weight1 + weight2 + weight3 + weight4 + weight5 = 30 := by
  sorry

end combined_weight_of_boxes_l2052_205235


namespace ratio_sub_add_l2052_205203

theorem ratio_sub_add (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 :=
sorry

end ratio_sub_add_l2052_205203


namespace shem_wage_multiple_kem_l2052_205297

-- Define the hourly wages and conditions
def kem_hourly_wage : ℝ := 4
def shem_daily_wage : ℝ := 80
def shem_workday_hours : ℝ := 8

-- Prove the multiple of Shem's hourly wage compared to Kem's hourly wage
theorem shem_wage_multiple_kem : (shem_daily_wage / shem_workday_hours) / kem_hourly_wage = 2.5 := by
  sorry

end shem_wage_multiple_kem_l2052_205297


namespace train_crosses_bridge_in_approximately_21_seconds_l2052_205290

noncomputable def length_of_train : ℝ := 110  -- meters
noncomputable def speed_of_train_kmph : ℝ := 60  -- kilometers per hour
noncomputable def length_of_bridge : ℝ := 240  -- meters

noncomputable def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def required_time : ℝ := total_distance / speed_of_train_mps

theorem train_crosses_bridge_in_approximately_21_seconds :
  |required_time - 21| < 1 :=
by sorry

end train_crosses_bridge_in_approximately_21_seconds_l2052_205290


namespace max_u_plus_2v_l2052_205291

theorem max_u_plus_2v (u v : ℝ) (h1 : 2 * u + 3 * v ≤ 10) (h2 : 4 * u + v ≤ 9) : u + 2 * v ≤ 6.1 :=
sorry

end max_u_plus_2v_l2052_205291


namespace cory_can_eat_fruits_in_105_ways_l2052_205288

-- Define the number of apples, oranges, and bananas Cory has
def apples := 4
def oranges := 1
def bananas := 2

-- Define the total number of fruits Cory has
def total_fruits := apples + oranges + bananas

-- Calculate the number of distinct orders in which Cory can eat the fruits
theorem cory_can_eat_fruits_in_105_ways :
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 :=
by
  -- Provide a sorry to skip the proof
  sorry

end cory_can_eat_fruits_in_105_ways_l2052_205288


namespace triangle_area_is_15_l2052_205249

def Point := (ℝ × ℝ)

def A : Point := (2, 2)
def B : Point := (7, 2)
def C : Point := (4, 8)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem triangle_area_is_15 : area_of_triangle A B C = 15 :=
by
  -- The proof goes here
  sorry

end triangle_area_is_15_l2052_205249


namespace approx_cube_of_331_l2052_205242

noncomputable def cube (x : ℝ) : ℝ := x * x * x

theorem approx_cube_of_331 : 
  ∃ ε > 0, abs (cube 0.331 - 0.037) < ε :=
by
  sorry

end approx_cube_of_331_l2052_205242


namespace ratio_four_l2052_205226

variable {x y : ℝ}

theorem ratio_four : y = 0.25 * x → x / y = 4 := by
  sorry

end ratio_four_l2052_205226


namespace charge_increase_percentage_l2052_205213

variable (P R G : ℝ)

def charge_relation_1 : Prop := P = 0.45 * R
def charge_relation_2 : Prop := P = 0.90 * G

theorem charge_increase_percentage (h1 : charge_relation_1 P R) (h2 : charge_relation_2 P G) : 
  (R/G - 1) * 100 = 100 :=
by
  sorry

end charge_increase_percentage_l2052_205213


namespace positive_value_of_A_l2052_205245

-- Define the relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- State the main theorem
theorem positive_value_of_A (A : ℝ) : hash A 7 = 72 → A = 11 :=
by
  -- Placeholder for the proof
  sorry

end positive_value_of_A_l2052_205245


namespace theta_in_second_quadrant_l2052_205234

open Real

-- Definitions for conditions
def cond1 (θ : ℝ) : Prop := sin θ > cos θ
def cond2 (θ : ℝ) : Prop := tan θ < 0

-- Main theorem statement
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : cond1 θ) 
  (h2 : cond2 θ) : 
  θ > π/2 ∧ θ < π :=
sorry

end theta_in_second_quadrant_l2052_205234


namespace smallest_x_satisfies_abs_eq_l2052_205285

theorem smallest_x_satisfies_abs_eq (x : ℝ) :
  (|2 * x + 5| = 21) → (x = -13) :=
sorry

end smallest_x_satisfies_abs_eq_l2052_205285


namespace booksJuly_l2052_205251

-- Definitions of the conditions
def booksMay : ℕ := 2
def booksJune : ℕ := 6
def booksTotal : ℕ := 18

-- Theorem statement proving how many books Tom read in July
theorem booksJuly : (booksTotal - (booksMay + booksJune)) = 10 :=
by
  sorry

end booksJuly_l2052_205251


namespace multiple_of_people_l2052_205219

-- Define the conditions
variable (P : ℕ) -- number of people who can do the work in 8 days

-- define a function that represents the work capacity of M * P people in days, 
-- we abstract away the solving steps into one declaration.

noncomputable def work_capacity (M P : ℕ) (days : ℕ) : ℚ :=
  M * (1/8) * days

-- Set up the problem to prove that the multiple of people is 2
theorem multiple_of_people (P : ℕ) : ∃ M : ℕ, work_capacity M P 2 = 1/2 :=
by
  use 2
  unfold work_capacity
  sorry

end multiple_of_people_l2052_205219


namespace john_payment_l2052_205296

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l2052_205296


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l2052_205255

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l2052_205255


namespace find_m_l2052_205257

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 + m
noncomputable def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem find_m (m : ℝ) : 
  ∃ a b : ℝ, (0 < a) ∧ (f a m = b) ∧ (g a = b) ∧ (2 * a = (6 / a) - 4) → m = -5 := 
by
  sorry

end find_m_l2052_205257


namespace find_painted_stencils_l2052_205237

variable (hourly_wage racquet_wage grommet_wage stencil_wage total_earnings hours_worked racquets_restrung grommets_changed : ℕ)
variable (painted_stencils: ℕ)

axiom condition_hourly_wage : hourly_wage = 9
axiom condition_racquet_wage : racquet_wage = 15
axiom condition_grommet_wage : grommet_wage = 10
axiom condition_stencil_wage : stencil_wage = 1
axiom condition_total_earnings : total_earnings = 202
axiom condition_hours_worked : hours_worked = 8
axiom condition_racquets_restrung : racquets_restrung = 7
axiom condition_grommets_changed : grommets_changed = 2

theorem find_painted_stencils :
  painted_stencils = 5 :=
by
  -- Given:
  -- hourly_wage = 9
  -- racquet_wage = 15
  -- grommet_wage = 10
  -- stencil_wage = 1
  -- total_earnings = 202
  -- hours_worked = 8
  -- racquets_restrung = 7
  -- grommets_changed = 2

  -- We need to prove:
  -- painted_stencils = 5
  
  sorry

end find_painted_stencils_l2052_205237


namespace christopher_avg_speed_l2052_205254

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

end christopher_avg_speed_l2052_205254


namespace arithmetic_mean_of_fractions_l2052_205272

theorem arithmetic_mean_of_fractions :
  (3 / 8 + 5 / 9 + 7 / 12) / 3 = 109 / 216 :=
by
  sorry

end arithmetic_mean_of_fractions_l2052_205272


namespace increasing_condition_l2052_205224

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 2 * x - 2 * a

-- Prove that f is increasing on the interval [2, +∞) if and only if a ≤ 2
theorem increasing_condition (a : ℝ) : (∀ x ≥ 2, f' x a ≥ 0) ↔ (a ≤ 2) := 
sorry

end increasing_condition_l2052_205224


namespace find_f_five_l2052_205225

noncomputable def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

theorem find_f_five (y : ℝ) (h : f 2 y = 50) : f 5 y = 92 := by
  sorry

end find_f_five_l2052_205225


namespace total_cost_of_shirt_and_coat_l2052_205277

-- Definition of the conditions
def shirt_cost : ℕ := 150
def one_third_of_coat (coat_cost: ℕ) : Prop := shirt_cost = coat_cost / 3

-- Theorem stating the problem to prove
theorem total_cost_of_shirt_and_coat (coat_cost : ℕ) (h : one_third_of_coat coat_cost) : shirt_cost + coat_cost = 600 :=
by 
  -- Proof goes here, using sorry as placeholder
  sorry

end total_cost_of_shirt_and_coat_l2052_205277


namespace eccentricity_of_hyperbola_l2052_205292

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := 2 * b
  let e := c / a
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_cond : hyperbola_eccentricity a b h_a h_b = 2 * (b / a)) :
  hyperbola_eccentricity a b h_a h_b = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l2052_205292


namespace inequality_proof_l2052_205230

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  (a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b) :=
by
  sorry

end inequality_proof_l2052_205230


namespace bus_stops_for_28_minutes_per_hour_l2052_205241

-- Definitions based on the conditions
def without_stoppages_speed : ℕ := 75
def with_stoppages_speed : ℕ := 40
def speed_difference : ℕ := without_stoppages_speed - with_stoppages_speed

-- Theorem statement
theorem bus_stops_for_28_minutes_per_hour : 
  ∀ (T : ℕ), (T = (speed_difference*60)/(without_stoppages_speed))  → 
  T = 28 := 
by
  sorry

end bus_stops_for_28_minutes_per_hour_l2052_205241


namespace math_club_team_selection_l2052_205202

open Nat

-- Lean statement of the problem
theorem math_club_team_selection : 
  (choose 7 3) * (choose 9 3) = 2940 :=
by 
  sorry

end math_club_team_selection_l2052_205202


namespace kickball_students_l2052_205256

theorem kickball_students (w t : ℕ) (hw : w = 37) (ht : t = w - 9) : w + t = 65 :=
by
  sorry

end kickball_students_l2052_205256


namespace obtuse_angle_probability_l2052_205228

noncomputable def probability_obtuse_angle : ℝ :=
  let F : ℝ × ℝ := (0, 3)
  let G : ℝ × ℝ := (5, 0)
  let H : ℝ × ℝ := (2 * Real.pi + 2, 0)
  let I : ℝ × ℝ := (2 * Real.pi + 2, 3)
  let rectangle_area : ℝ := (2 * Real.pi + 2) * 3
  let semicircle_radius : ℝ := Real.sqrt (2.5^2 + 1.5^2)
  let semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius^2
  semicircle_area / rectangle_area

theorem obtuse_angle_probability :
  probability_obtuse_angle = 17 / (24 + 4 * Real.pi) :=
by
  sorry

end obtuse_angle_probability_l2052_205228


namespace calculate_value_l2052_205259

def a : ℤ := 3 * 4 * 5
def b : ℚ := 1/3 + 1/4 + 1/5

theorem calculate_value :
  (a : ℚ) * b = 47 := by
sorry

end calculate_value_l2052_205259


namespace lamplighter_monkey_distance_traveled_l2052_205261

-- Define the parameters
def running_speed : ℕ := 15
def running_time : ℕ := 5
def swinging_speed : ℕ := 10
def swinging_time : ℕ := 10

-- Define the proof statement
theorem lamplighter_monkey_distance_traveled :
  (running_speed * running_time) + (swinging_speed * swinging_time) = 175 := by
  sorry

end lamplighter_monkey_distance_traveled_l2052_205261


namespace necessary_condition_l2052_205229

theorem necessary_condition (m : ℝ) (h : ∀ x : ℝ, x^2 - x + m > 0) : m > 0 := 
sorry

end necessary_condition_l2052_205229


namespace rope_total_in_inches_l2052_205265

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l2052_205265


namespace clara_age_l2052_205298

theorem clara_age (x : ℕ) (n m : ℕ) (h1 : x - 2 = n^2) (h2 : x + 3 = m^3) : x = 123 :=
by sorry

end clara_age_l2052_205298


namespace trigonometric_identity_l2052_205216

open Real

theorem trigonometric_identity (α : ℝ) : 
  sin α * sin α + cos (π / 6 + α) * cos (π / 6 + α) + sin α * cos (π / 6 + α) = 3 / 4 :=
sorry

end trigonometric_identity_l2052_205216


namespace find_share_of_A_l2052_205207

noncomputable def investment_share_A (initial_investment_A initial_investment_B withdraw_A add_B after_months end_of_year_profit : ℝ) : ℝ :=
  let investment_months_A := (initial_investment_A * after_months) + ((initial_investment_A - withdraw_A) * (12 - after_months))
  let investment_months_B := (initial_investment_B * after_months) + ((initial_investment_B + add_B) * (12 - after_months))
  let total_investment_months := investment_months_A + investment_months_B
  let ratio_A := investment_months_A / total_investment_months
  ratio_A * end_of_year_profit

theorem find_share_of_A : 
  investment_share_A 3000 4000 1000 1000 8 630 = 240 := 
by 
  sorry

end find_share_of_A_l2052_205207


namespace magnitude_of_b_l2052_205239

variable (a b : ℝ)

-- Defining the given conditions as hypotheses
def condition1 : Prop := (a - b) * (a - b) = 9
def condition2 : Prop := (a + 2 * b) * (a + 2 * b) = 36
def condition3 : Prop := a^2 + (a * b) - 2 * b^2 = -9

-- Defining the theorem to prove
theorem magnitude_of_b (ha : condition1 a b) (hb : condition2 a b) (hc : condition3 a b) : b^2 = 3 := 
sorry

end magnitude_of_b_l2052_205239


namespace gcd_168_54_264_l2052_205299

theorem gcd_168_54_264 : Nat.gcd (Nat.gcd 168 54) 264 = 6 :=
by
  -- proof goes here and ends with sorry for now
  sorry

end gcd_168_54_264_l2052_205299


namespace vera_first_place_l2052_205269

noncomputable def placement (anna vera katya natasha : ℕ) : Prop :=
  (anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)

theorem vera_first_place :
  ∃ (anna vera katya natasha : ℕ),
    (placement anna vera katya natasha) ∧ 
    (vera = 1) ∧ 
    (1 ≠ 4) → 
    ((anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)) ∧ 
    (1 = 1) ∧ 
    (∃ i j k l : ℕ, (i ≠ 1 ∧ i ≠ 4) ∧ (j = 1) ∧ (k ≠ 1) ∧ (l = 4)) ∧ 
    (vera = 1) :=
sorry

end vera_first_place_l2052_205269


namespace find_m_value_l2052_205238

noncomputable def x0 : ℝ := sorry

noncomputable def m : ℝ := x0^3 + 2 * x0^2 + 2

theorem find_m_value :
  (x0^2 + x0 - 1 = 0) → (m = 3) :=
by
  intro h
  have hx : x0 = sorry := sorry
  have hm : m = x0 ^ 3 + 2 * x0^2 + 2 := rfl
  rw [hx] at hm
  sorry

end find_m_value_l2052_205238


namespace sum_of_digits_of_4_plus_2_pow_21_l2052_205220

theorem sum_of_digits_of_4_plus_2_pow_21 :
  let x := (4 + 2)
  (x^(21) % 100).div 10 + (x^(21) % 100).mod 10 = 6 :=
by
  let x := (4 + 2)
  sorry

end sum_of_digits_of_4_plus_2_pow_21_l2052_205220


namespace number_of_numbers_is_ten_l2052_205262

open Nat

-- Define the conditions as given
variable (n : ℕ) -- Total number of numbers
variable (incorrect_average correct_average incorrect_value correct_value : ℤ)
variable (h1 : incorrect_average = 16)
variable (h2 : correct_average = 17)
variable (h3 : incorrect_value = 25)
variable (h4 : correct_value = 35)

-- Define the proof problem
theorem number_of_numbers_is_ten
  (h1 : incorrect_average = 16)
  (h2 : correct_average = 17)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 35)
  (h5 : ∀ (x : ℤ), x ≠ incorrect_value → incorrect_average * (n : ℤ) + x = correct_average * (n : ℤ) + correct_value - incorrect_value)
  : n = 10 := 
sorry

end number_of_numbers_is_ten_l2052_205262


namespace geom_seq_sum_l2052_205281

theorem geom_seq_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = 2) (h2 : a 1 * a 5 = 64) :
  (a 1 * (1 - q^n)) / (1 - q) = 2^(n+1) - 2 := 
sorry

end geom_seq_sum_l2052_205281


namespace sequence_nth_term_l2052_205227

/-- The nth term of the sequence {a_n} defined by a_1 = 1 and
    the recurrence relation a_{n+1} = 2a_n + 2 for all n ∈ ℕ*,
    is given by the formula a_n = 3 * 2 ^ (n - 1) - 2. -/
theorem sequence_nth_term (n : ℕ) (h : n > 0) : 
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n > 0, a (n + 1) = 2 * a n + 2) ∧ a n = 3 * 2 ^ (n - 1) - 2 :=
  sorry

end sequence_nth_term_l2052_205227


namespace pants_cost_l2052_205201

theorem pants_cost (P : ℝ) : 
(80 + 3 * P + 300) * 0.90 = 558 → P = 80 :=
by
  sorry

end pants_cost_l2052_205201


namespace third_motorcyclist_speed_l2052_205294

theorem third_motorcyclist_speed 
  (t₁ t₂ : ℝ)
  (x : ℝ)
  (h1 : t₁ - t₂ = 1.25)
  (h2 : 80 * t₁ = x * (t₁ - 0.5))
  (h3 : 60 * t₂ = x * (t₂ - 0.5))
  (h4 : x ≠ 60)
  (h5 : x ≠ 80):
  x = 100 :=
by
  sorry

end third_motorcyclist_speed_l2052_205294


namespace trapezoid_shorter_base_length_l2052_205221

theorem trapezoid_shorter_base_length
  (L B : ℕ)
  (hL : L = 125)
  (hB : B = 5)
  (h : ∀ x, (L - x) / 2 = B → x = 115) :
  ∃ x, x = 115 := by
    sorry

end trapezoid_shorter_base_length_l2052_205221


namespace towels_per_pack_l2052_205273

open Nat

-- Define the given conditions
def packs : Nat := 9
def total_towels : Nat := 27

-- Define the property to prove
theorem towels_per_pack : total_towels / packs = 3 := by
  sorry

end towels_per_pack_l2052_205273


namespace polynomial_divisibility_l2052_205266

theorem polynomial_divisibility (a : ℤ) : 
  (∀x : ℤ, x^2 - x + a ∣ x^13 + x + 94) → a = 2 := 
by 
  sorry

end polynomial_divisibility_l2052_205266


namespace system_solution_l2052_205250

theorem system_solution :
  (∀ x y : ℝ, (2 * x + 3 * y = 19) ∧ (3 * x + 4 * y = 26) → x = 2 ∧ y = 5) →
  (∃ x y : ℝ, (2 * (2 * x + 4) + 3 * (y + 3) = 19) ∧ (3 * (2 * x + 4) + 4 * (y + 3) = 26) ∧ x = -1 ∧ y = 2) :=
by
  sorry

end system_solution_l2052_205250


namespace ratio_of_men_to_women_l2052_205284

theorem ratio_of_men_to_women
  (M W : ℕ)
  (h1 : W = M + 6)
  (h2 : M + W = 16) :
  M * 11 = 5 * W :=
by
    -- We can explicitly construct the necessary proof here, but according to instructions we add sorry to bypass for now
    sorry

end ratio_of_men_to_women_l2052_205284


namespace range_of_a_l2052_205218

variable (a : ℝ)
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a ∧ a > 0}

def p (a : ℝ) := 1 ∈ A a
def q (a : ℝ) := 2 ∈ A a

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : 1 < a ∧ a ≤ 2 := sorry

end range_of_a_l2052_205218


namespace problem_C_l2052_205293

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0

theorem problem_C (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0 :=
by
  sorry

end problem_C_l2052_205293


namespace number_added_is_59_l2052_205264

theorem number_added_is_59 (x : ℤ) (h1 : -2 < 0) (h2 : -3 < 0) (h3 : -2 * -3 + x = 65) : x = 59 :=
by sorry

end number_added_is_59_l2052_205264


namespace surface_area_of_glued_cubes_l2052_205233

noncomputable def calculate_surface_area (large_cube_edge_length : ℕ) : ℕ :=
sorry

theorem surface_area_of_glued_cubes :
  calculate_surface_area 4 = 136 :=
sorry

end surface_area_of_glued_cubes_l2052_205233


namespace exists_consecutive_numbers_divisible_by_3_5_7_l2052_205274

theorem exists_consecutive_numbers_divisible_by_3_5_7 :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 200 ∧
    a % 3 = 0 ∧ (a + 1) % 5 = 0 ∧ (a + 2) % 7 = 0 :=
by
  sorry

end exists_consecutive_numbers_divisible_by_3_5_7_l2052_205274


namespace circle_through_points_l2052_205252

-- Definitions of the points
def O : (ℝ × ℝ) := (0, 0)
def M1 : (ℝ × ℝ) := (1, 1)
def M2 : (ℝ × ℝ) := (4, 2)

-- Definition of the center and radius of the circle
def center : (ℝ × ℝ) := (4, -3)
def radius : ℝ := 5

-- The circle equation function
def circle_eq (x y : ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  (x - c.1)^2 + (y + c.2)^2 = r^2

theorem circle_through_points :
  circle_eq 0 0 center radius ∧ circle_eq 1 1 center radius ∧ circle_eq 4 2 center radius :=
by
  -- This is where the proof would go
  sorry

end circle_through_points_l2052_205252


namespace pirate_coins_total_l2052_205280

def total_coins (y : ℕ) := 6 * y

theorem pirate_coins_total : 
  (∃ y : ℕ, y ≠ 0 ∧ y * (y + 1) / 2 = 5 * y) →
  total_coins 9 = 54 :=
by
  sorry

end pirate_coins_total_l2052_205280


namespace sum_of_natural_numbers_l2052_205258

theorem sum_of_natural_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_natural_numbers_l2052_205258
