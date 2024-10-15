import Mathlib

namespace NUMINAMATH_GPT_division_of_fractions_l1290_129075

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_GPT_division_of_fractions_l1290_129075


namespace NUMINAMATH_GPT_double_people_half_work_l1290_129006

-- Definitions
def initial_person_count (P : ℕ) : Prop := true
def initial_time (T : ℕ) : Prop := T = 16

-- Theorem
theorem double_people_half_work (P T : ℕ) (hP : initial_person_count P) (hT : initial_time T) : P > 0 → (2 * P) * (T / 2) = P * T / 2 := by
  sorry

end NUMINAMATH_GPT_double_people_half_work_l1290_129006


namespace NUMINAMATH_GPT_market_value_of_stock_l1290_129054

-- Define the given conditions.
def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.09 * face_value
def yield : ℝ := 0.08

-- State the problem: proving the market value of the stock.
theorem market_value_of_stock : (dividend_per_share / yield) * 100 = 112.50 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l1290_129054


namespace NUMINAMATH_GPT_green_beans_to_onions_ratio_l1290_129026

def cut_conditions
  (potatoes : ℕ)
  (carrots : ℕ)
  (onions : ℕ)
  (green_beans : ℕ) : Prop :=
  carrots = 6 * potatoes ∧ onions = 2 * carrots ∧ potatoes = 2 ∧ green_beans = 8

theorem green_beans_to_onions_ratio (potatoes carrots onions green_beans : ℕ) :
  cut_conditions potatoes carrots onions green_beans →
  green_beans / gcd green_beans onions = 1 ∧ onions / gcd green_beans onions = 3 :=
by
  sorry

end NUMINAMATH_GPT_green_beans_to_onions_ratio_l1290_129026


namespace NUMINAMATH_GPT_total_driving_time_is_40_l1290_129050

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end NUMINAMATH_GPT_total_driving_time_is_40_l1290_129050


namespace NUMINAMATH_GPT_share_apples_l1290_129085

theorem share_apples (h : 9 / 3 = 3) : true :=
sorry

end NUMINAMATH_GPT_share_apples_l1290_129085


namespace NUMINAMATH_GPT_polynomial_division_l1290_129008

noncomputable def polynomial_div_quotient (p q : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.divByMonic p q

theorem polynomial_division 
  (p q : Polynomial ℚ)
  (hq : q = Polynomial.C 3 * Polynomial.X - Polynomial.C 4)
  (hp : p = 10 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 8 * Polynomial.X - 9) :
  polynomial_div_quotient p q = (10 / 3) * Polynomial.X ^ 2 - (55 / 9) * Polynomial.X - (172 / 27) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_division_l1290_129008


namespace NUMINAMATH_GPT_total_cost_is_63_l1290_129089

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_63_l1290_129089


namespace NUMINAMATH_GPT_cube_root_of_neg_eight_l1290_129051

theorem cube_root_of_neg_eight : ∃ x : ℝ, x ^ 3 = -8 ∧ x = -2 := by 
  sorry

end NUMINAMATH_GPT_cube_root_of_neg_eight_l1290_129051


namespace NUMINAMATH_GPT_snow_total_inches_l1290_129059

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end NUMINAMATH_GPT_snow_total_inches_l1290_129059


namespace NUMINAMATH_GPT_shift_parabola_left_l1290_129069

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end NUMINAMATH_GPT_shift_parabola_left_l1290_129069


namespace NUMINAMATH_GPT_sequence_bounded_l1290_129097

theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_dep : ∀ k n m l, k + n = m + l → (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ m M : ℝ, ∀ n, m ≤ a n ∧ a n ≤ M :=
sorry

end NUMINAMATH_GPT_sequence_bounded_l1290_129097


namespace NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l1290_129055

theorem sum_of_coordinates_of_other_endpoint :
  ∀ (x y : ℤ), (7, -15) = ((x + 3) / 2, (y - 5) / 2) → x + y = -14 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l1290_129055


namespace NUMINAMATH_GPT_evaluate_expression_l1290_129083

theorem evaluate_expression : (1 / (2 + (1 / (3 + (1 / 4))))) = (13 / 30) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1290_129083


namespace NUMINAMATH_GPT_geometric_series_first_term_l1290_129047

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1290_129047


namespace NUMINAMATH_GPT_intersection_A_B_subset_A_B_l1290_129052

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
noncomputable def set_B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 22}

theorem intersection_A_B (a : ℝ) (ha : a = 10) : set_A a ∩ set_B = {x : ℝ | 21 ≤ x ∧ x ≤ 22} := by
  sorry

theorem subset_A_B (a : ℝ) : set_A a ⊆ set_B → a ≤ 9 := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_subset_A_B_l1290_129052


namespace NUMINAMATH_GPT_largest_angle_in_consecutive_integer_hexagon_l1290_129022

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end NUMINAMATH_GPT_largest_angle_in_consecutive_integer_hexagon_l1290_129022


namespace NUMINAMATH_GPT_exercise_l1290_129029

theorem exercise (x y z : ℕ) (h1 : x * y * z = 1) : (7 ^ ((x + y + z) ^ 3) / 7 ^ ((x - y + z) ^ 3)) = 7 ^ 6 := 
by
  sorry

end NUMINAMATH_GPT_exercise_l1290_129029


namespace NUMINAMATH_GPT_determine_b_l1290_129042

theorem determine_b (a b : ℤ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l1290_129042


namespace NUMINAMATH_GPT_factor_difference_of_squares_196_l1290_129096

theorem factor_difference_of_squares_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_196_l1290_129096


namespace NUMINAMATH_GPT_new_student_weight_l1290_129067

theorem new_student_weight :
  let avg_weight_29 := 28
  let num_students_29 := 29
  let avg_weight_30 := 27.4
  let num_students_30 := 30
  let total_weight_29 := avg_weight_29 * num_students_29
  let total_weight_30 := avg_weight_30 * num_students_30
  let new_student_weight := total_weight_30 - total_weight_29
  new_student_weight = 10 :=
by
  sorry

end NUMINAMATH_GPT_new_student_weight_l1290_129067


namespace NUMINAMATH_GPT_square_side_length_equals_5_sqrt_pi_l1290_129092

theorem square_side_length_equals_5_sqrt_pi :
  ∃ s : ℝ, ∃ r : ℝ, (r = 5) ∧ (s = 2 * r) ∧ (s ^ 2 = 25 * π) ∧ (s = 5 * Real.sqrt π) :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_equals_5_sqrt_pi_l1290_129092


namespace NUMINAMATH_GPT_algebra_ineq_example_l1290_129032

theorem algebra_ineq_example (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ( ( (xy + yz + zx) / 3 ) ^ (1/3) ) :=
by
  sorry

end NUMINAMATH_GPT_algebra_ineq_example_l1290_129032


namespace NUMINAMATH_GPT_centroid_quad_area_correct_l1290_129036

noncomputable def centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  if h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35 then
    12800 / 9
  else
    sorry

theorem centroid_quad_area_correct (E F G H Q : ℝ × ℝ) (side_length EQ FQ : ℝ) 
  (h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35) :
  centroid_quadrilateral_area E F G H Q side_length EQ FQ = 12800 / 9 :=
sorry

end NUMINAMATH_GPT_centroid_quad_area_correct_l1290_129036


namespace NUMINAMATH_GPT_product_in_base_7_l1290_129012

def base_7_product : ℕ :=
  let b := 7
  Nat.ofDigits b [3, 5, 6] * Nat.ofDigits b [4]

theorem product_in_base_7 :
  base_7_product = Nat.ofDigits 7 [3, 2, 3, 1, 2] :=
by
  -- The proof is formally skipped for this exercise, hence we insert 'sorry'.
  sorry

end NUMINAMATH_GPT_product_in_base_7_l1290_129012


namespace NUMINAMATH_GPT_find_value_of_6b_l1290_129007

theorem find_value_of_6b (a b : ℝ) (h1 : 10 * a = 20) (h2 : 120 * a * b = 800) : 6 * b = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_6b_l1290_129007


namespace NUMINAMATH_GPT_find_a3_plus_a5_l1290_129043

-- Define an arithmetic-geometric sequence
def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, 0 < r ∧ ∃ b : ℝ, a n = b * r ^ n

-- Define the given condition
def given_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

-- Define the target theorem statement
theorem find_a3_plus_a5 (a : ℕ → ℝ) 
  (pos_sequence : is_arithmetic_geometric a) 
  (cond : given_condition a) : 
  a 3 + a 5 = 5 :=
sorry

end NUMINAMATH_GPT_find_a3_plus_a5_l1290_129043


namespace NUMINAMATH_GPT_binomial_30_3_l1290_129095

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_GPT_binomial_30_3_l1290_129095


namespace NUMINAMATH_GPT_compute_difference_of_squares_l1290_129018

theorem compute_difference_of_squares :
  let a := 23
  let b := 12
  (a + b) ^ 2 - (a - b) ^ 2 = 1104 := by
sorry

end NUMINAMATH_GPT_compute_difference_of_squares_l1290_129018


namespace NUMINAMATH_GPT_bucket_capacity_l1290_129078

theorem bucket_capacity (jack_buckets_per_trip : ℕ)
                        (jill_buckets_per_trip : ℕ)
                        (jack_trip_ratio : ℝ)
                        (jill_trips : ℕ)
                        (tank_capacity : ℝ)
                        (bucket_capacity : ℝ)
                        (h1 : jack_buckets_per_trip = 2)
                        (h2 : jill_buckets_per_trip = 1)
                        (h3 : jack_trip_ratio = 3 / 2)
                        (h4 : jill_trips = 30)
                        (h5 : tank_capacity = 600) :
  bucket_capacity = 5 :=
by 
  sorry

end NUMINAMATH_GPT_bucket_capacity_l1290_129078


namespace NUMINAMATH_GPT_find_k_value_l1290_129039

noncomputable def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| n => a + (n - 1) * d

theorem find_k_value (a d : ℤ) (k : ℕ) 
  (h1 : arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 24)
  (h2 : (Finset.range 11).sum (λ i => arithmetic_seq a d (5 + i)) = 110)
  (h3 : arithmetic_seq a d k = 16) : 
  k = 16 :=
sorry

end NUMINAMATH_GPT_find_k_value_l1290_129039


namespace NUMINAMATH_GPT_sales_fifth_month_l1290_129086

theorem sales_fifth_month (s1 s2 s3 s4 s6 s5 : ℝ) (target_avg total_sales : ℝ)
  (h1 : s1 = 4000)
  (h2 : s2 = 6524)
  (h3 : s3 = 5689)
  (h4 : s4 = 7230)
  (h6 : s6 = 12557)
  (h_avg : target_avg = 7000)
  (h_total_sales : total_sales = 42000) :
  s5 = 6000 :=
by
  sorry

end NUMINAMATH_GPT_sales_fifth_month_l1290_129086


namespace NUMINAMATH_GPT_final_remaining_money_l1290_129031

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end NUMINAMATH_GPT_final_remaining_money_l1290_129031


namespace NUMINAMATH_GPT_time_to_eat_potatoes_l1290_129091

theorem time_to_eat_potatoes (rate : ℕ → ℕ → ℝ) (potatoes : ℕ → ℕ → ℝ) 
    (minutes : ℕ) (hours : ℝ) (total_potatoes : ℕ) : 
    rate 3 20 = 9 / 1 -> potatoes 27 9 = 3 := 
by
  intro h1
  -- You can add intermediate steps here as optional comments for clarity during proof construction
  /- 
  Given: 
  rate 3 20 = 9 -> Jason's rate of eating potatoes is 9 potatoes per hour
  time = potatoes / rate -> 27 potatoes / 9 potatoes/hour = 3 hours
  -/
  sorry

end NUMINAMATH_GPT_time_to_eat_potatoes_l1290_129091


namespace NUMINAMATH_GPT_prop_A_prop_B_prop_C_prop_D_l1290_129027

variable {a b : ℝ}

-- Proposition A
theorem prop_A (h : a^2 - b^2 = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b < 1 := sorry

-- Proposition B (negation of the original proposition since B is incorrect)
theorem prop_B (h : (1 / b) - (1 / a) = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b ≥ 1 := sorry

-- Proposition C
theorem prop_C (h : a > b + 1) (a_pos : 0 < a) (b_pos : 0 < b) : a^2 > b^2 + 1 := sorry

-- Proposition D (negation of the original proposition since D is incorrect)
theorem prop_D (h1 : a ≤ 1) (h2 : b ≤ 1) (a_pos : 0 < a) (b_pos : 0 < b) : |a - b| < |1 - a * b| := sorry

end NUMINAMATH_GPT_prop_A_prop_B_prop_C_prop_D_l1290_129027


namespace NUMINAMATH_GPT_visitors_inversely_proportional_l1290_129093

theorem visitors_inversely_proportional (k : ℝ) (v₁ v₂ t₁ t₂ : ℝ) (h1 : v₁ * t₁ = k) (h2 : t₁ = 20) (h3 : v₁ = 150) (h4 : t₂ = 30) : v₂ = 100 :=
by
  -- This is a placeholder line; the actual proof would go here.
  sorry

end NUMINAMATH_GPT_visitors_inversely_proportional_l1290_129093


namespace NUMINAMATH_GPT_arnold_total_protein_l1290_129071

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end NUMINAMATH_GPT_arnold_total_protein_l1290_129071


namespace NUMINAMATH_GPT_team_A_days_additional_people_l1290_129046

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

end NUMINAMATH_GPT_team_A_days_additional_people_l1290_129046


namespace NUMINAMATH_GPT_time_to_see_each_other_again_l1290_129005

variable (t : ℝ) (t_frac : ℚ)
variable (kenny_speed jenny_speed : ℝ)
variable (kenny_initial jenny_initial : ℝ)
variable (building_side distance_between_paths : ℝ)

def kenny_position (t : ℝ) : ℝ := kenny_initial + kenny_speed * t
def jenny_position (t : ℝ) : ℝ := jenny_initial + jenny_speed * t

theorem time_to_see_each_other_again
  (kenny_speed_eq : kenny_speed = 4)
  (jenny_speed_eq : jenny_speed = 2)
  (kenny_initial_eq : kenny_initial = -50)
  (jenny_initial_eq : jenny_initial = -50)
  (building_side_eq : building_side = 100)
  (distance_between_paths_eq : distance_between_paths = 300)
  (t_gt_50 : t > 50)
  (t_frac_eq : t_frac = 50) :
  (t == t_frac) :=
  sorry

end NUMINAMATH_GPT_time_to_see_each_other_again_l1290_129005


namespace NUMINAMATH_GPT_spherical_to_rectangular_example_l1290_129060

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 5 (3 * Real.pi / 2) (Real.pi / 3) = (0, -5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  simp [spherical_to_rectangular, Real.sin, Real.cos]
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_example_l1290_129060


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1290_129009

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1290_129009


namespace NUMINAMATH_GPT_problem_remainder_l1290_129037

theorem problem_remainder :
  ((12095 + 12097 + 12099 + 12101 + 12103 + 12105 + 12107) % 10) = 7 := by
  sorry

end NUMINAMATH_GPT_problem_remainder_l1290_129037


namespace NUMINAMATH_GPT_more_visitors_that_day_l1290_129049

def number_of_visitors_previous_day : ℕ := 100
def number_of_visitors_that_day : ℕ := 666

theorem more_visitors_that_day :
  number_of_visitors_that_day - number_of_visitors_previous_day = 566 :=
by
  sorry

end NUMINAMATH_GPT_more_visitors_that_day_l1290_129049


namespace NUMINAMATH_GPT_books_leftover_l1290_129040

theorem books_leftover :
  (1500 * 45) % 47 = 13 :=
by
  sorry

end NUMINAMATH_GPT_books_leftover_l1290_129040


namespace NUMINAMATH_GPT_linear_function_quadrants_l1290_129087

theorem linear_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = (k + 1) * x + k - 2 → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ (-1 < k ∧ k < 2) := 
sorry

end NUMINAMATH_GPT_linear_function_quadrants_l1290_129087


namespace NUMINAMATH_GPT_end_same_digit_l1290_129034

theorem end_same_digit
  (a b : ℕ)
  (h : (2 * a + b) % 10 = (2 * b + a) % 10) :
  a % 10 = b % 10 :=
by
  sorry

end NUMINAMATH_GPT_end_same_digit_l1290_129034


namespace NUMINAMATH_GPT_find_theta_l1290_129079

variable (x : ℝ) (θ : ℝ) (k : ℤ)

def condition := (3 - 3^(-|x - 3|))^2 = 3 - Real.cos θ

theorem find_theta (h : condition x θ) : ∃ k : ℤ, θ = (2 * k + 1) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_find_theta_l1290_129079


namespace NUMINAMATH_GPT_Ayla_call_duration_l1290_129063

theorem Ayla_call_duration
  (charge_per_minute : ℝ)
  (monthly_bill : ℝ)
  (customers_per_week : ℕ)
  (weeks_in_month : ℕ)
  (calls_duration : ℝ)
  (h_charge : charge_per_minute = 0.05)
  (h_bill : monthly_bill = 600)
  (h_customers : customers_per_week = 50)
  (h_weeks_in_month : weeks_in_month = 4)
  (h_calls_duration : calls_duration = (monthly_bill / charge_per_minute) / (customers_per_week * weeks_in_month)) :
  calls_duration = 60 :=
by 
  sorry

end NUMINAMATH_GPT_Ayla_call_duration_l1290_129063


namespace NUMINAMATH_GPT_inscribed_circle_radius_range_l1290_129070

noncomputable def r_range (AD DB : ℝ) (angle_A : ℝ) : Set ℝ :=
  { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 }

theorem inscribed_circle_radius_range (AD DB : ℝ) (angle_A : ℝ) (h1 : AD = 2 * Real.sqrt 3) 
    (h2 : DB = Real.sqrt 3) (h3 : angle_A > 60) : 
    r_range AD DB angle_A = { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 } :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_range_l1290_129070


namespace NUMINAMATH_GPT_irrational_sqrt_10_l1290_129024

theorem irrational_sqrt_10 : Irrational (Real.sqrt 10) :=
sorry

end NUMINAMATH_GPT_irrational_sqrt_10_l1290_129024


namespace NUMINAMATH_GPT_problem_l1290_129017

variable (x : ℝ)

theorem problem (A B : ℝ) 
  (h : (A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3))): 
  A + B = 15 := by
  sorry

end NUMINAMATH_GPT_problem_l1290_129017


namespace NUMINAMATH_GPT_sum_of_three_largest_l1290_129057

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_largest_l1290_129057


namespace NUMINAMATH_GPT_graph_transformation_l1290_129099

theorem graph_transformation (a b c : ℝ) (h1 : c = 1) (h2 : a + b + c = -2) (h3 : a - b + c = 2) :
  (∀ x, cx^2 + 2 * bx + a = (x - 2)^2 - 5) := 
sorry

end NUMINAMATH_GPT_graph_transformation_l1290_129099


namespace NUMINAMATH_GPT_axis_of_symmetry_condition_l1290_129016

theorem axis_of_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
    (h_sym : ∀ x y, y = -x → y = (p * x + q) / (r * x + s)) : p = s :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_condition_l1290_129016


namespace NUMINAMATH_GPT_calculate_f_sum_l1290_129073

noncomputable def f (n : ℕ) := Real.log (3 * n^2) / Real.log 3003

theorem calculate_f_sum :
  f 7 + f 11 + f 13 = 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_sum_l1290_129073


namespace NUMINAMATH_GPT_solve_for_y_l1290_129094

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1290_129094


namespace NUMINAMATH_GPT_original_number_is_17_l1290_129015

-- Function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  (ones * 10) + tens

-- Problem statement
theorem original_number_is_17 (x : ℕ) (h1 : reverse_digits (2 * x) + 2 = 45) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_17_l1290_129015


namespace NUMINAMATH_GPT_min_CD_squared_diff_l1290_129048

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 2)) + (Real.sqrt (z + 2))
noncomputable def f (x y z : ℝ) : ℝ := (C x y z) ^ 2 - (D x y z) ^ 2

theorem min_CD_squared_diff (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  f x y z ≥ 41.4736 :=
sorry

end NUMINAMATH_GPT_min_CD_squared_diff_l1290_129048


namespace NUMINAMATH_GPT_value_of_a_if_lines_are_parallel_l1290_129064

theorem value_of_a_if_lines_are_parallel (a : ℝ) :
  (∀ (x y : ℝ), x + a*y - 7 = 0 → (a+1)*x + 2*y - 14 = 0) → a = -2 :=
sorry

end NUMINAMATH_GPT_value_of_a_if_lines_are_parallel_l1290_129064


namespace NUMINAMATH_GPT_expression_divisibility_l1290_129082

theorem expression_divisibility (x y : ℝ) : 
  ∃ P : ℝ, (x^2 - x * y + y^2)^3 + (x^2 + x * y + y^2)^3 = (2 * x^2 + 2 * y^2) * P := 
by 
  sorry

end NUMINAMATH_GPT_expression_divisibility_l1290_129082


namespace NUMINAMATH_GPT_slope_parallel_l1290_129053

theorem slope_parallel {x y : ℝ} (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = -1/2 ∧ ( ∀ (x1 x2 : ℝ), 3 * x1 - 6 * y = 15 → ∃ y1 : ℝ, y1 = m * x1) :=
by
  sorry

end NUMINAMATH_GPT_slope_parallel_l1290_129053


namespace NUMINAMATH_GPT_solve_inequality_l1290_129001

noncomputable def inequality (x : ℕ) : Prop :=
  6 * (9 : ℝ)^(1/x) - 13 * (3 : ℝ)^(1/x) * (2 : ℝ)^(1/x) + 6 * (4 : ℝ)^(1/x) ≤ 0

theorem solve_inequality (x : ℕ) (hx : 1 < x) : inequality x ↔ x ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l1290_129001


namespace NUMINAMATH_GPT_geometric_sum_eqn_l1290_129023

theorem geometric_sum_eqn 
  (a1 q : ℝ) 
  (hne1 : q ≠ 1) 
  (hS2 : a1 * (1 - q^2) / (1 - q) = 1) 
  (hS4 : a1 * (1 - q^4) / (1 - q) = 3) :
  a1 * (1 - q^8) / (1 - q) = 15 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_eqn_l1290_129023


namespace NUMINAMATH_GPT_smallest_nuts_in_bag_l1290_129038

theorem smallest_nuts_in_bag :
  ∃ (N : ℕ), N ≡ 1 [MOD 11] ∧ N ≡ 8 [MOD 13] ∧ N ≡ 3 [MOD 17] ∧
             (∀ M, (M ≡ 1 [MOD 11] ∧ M ≡ 8 [MOD 13] ∧ M ≡ 3 [MOD 17]) → M ≥ N) :=
sorry

end NUMINAMATH_GPT_smallest_nuts_in_bag_l1290_129038


namespace NUMINAMATH_GPT_increase_in_area_l1290_129028

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
noncomputable def radius_of_circle (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * (radius ^ 2)

theorem increase_in_area :
  let rectangle_length := 60
  let rectangle_width := 20
  let rectangle_area := area_of_rectangle rectangle_length rectangle_width
  let fence_length := perimeter_of_rectangle rectangle_length rectangle_width
  let circle_radius := radius_of_circle fence_length
  let circle_area := area_of_circle circle_radius
  let area_increase := circle_area - rectangle_area
  837.99 ≤ area_increase :=
by
  sorry

end NUMINAMATH_GPT_increase_in_area_l1290_129028


namespace NUMINAMATH_GPT_af_over_cd_is_025_l1290_129033

theorem af_over_cd_is_025
  (a b c d e f X : ℝ)
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end NUMINAMATH_GPT_af_over_cd_is_025_l1290_129033


namespace NUMINAMATH_GPT_smallest_positive_solution_to_congruence_l1290_129045

theorem smallest_positive_solution_to_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 33] ∧ x = 28 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_positive_solution_to_congruence_l1290_129045


namespace NUMINAMATH_GPT_howard_groups_l1290_129002

theorem howard_groups :
  (18 : ℕ) / (24 / 4) = 3 := sorry

end NUMINAMATH_GPT_howard_groups_l1290_129002


namespace NUMINAMATH_GPT_total_cost_eq_4800_l1290_129014

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_eq_4800_l1290_129014


namespace NUMINAMATH_GPT_box_area_relation_l1290_129088

theorem box_area_relation (a b c : ℕ) (h : a = b + c + 10) :
  (a * b) * (b * c) * (c * a) = (2 * (b + c) + 10)^2 := 
sorry

end NUMINAMATH_GPT_box_area_relation_l1290_129088


namespace NUMINAMATH_GPT_vendor_profit_l1290_129072

theorem vendor_profit {s₁ s₂ c₁ c₂ : ℝ} (h₁ : s₁ = 80) (h₂ : s₂ = 80) (profit₁ : s₁ = c₁ * 1.60) (loss₂ : s₂ = c₂ * 0.80) 
: (s₁ + s₂) - (c₁ + c₂) = 10 := by 
  sorry

end NUMINAMATH_GPT_vendor_profit_l1290_129072


namespace NUMINAMATH_GPT_solution_triple_root_system_l1290_129068

theorem solution_triple_root_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_triple_root_system_l1290_129068


namespace NUMINAMATH_GPT_ice_cream_eaten_on_friday_l1290_129041

theorem ice_cream_eaten_on_friday
  (x : ℝ) -- the amount eaten on Friday night
  (saturday_night : ℝ) -- the amount eaten on Saturday night
  (total : ℝ) -- the total amount eaten
  
  (h1 : saturday_night = 0.25)
  (h2 : total = 3.5)
  (h3 : x + saturday_night = total) : x = 3.25 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_eaten_on_friday_l1290_129041


namespace NUMINAMATH_GPT_arithmetic_sequence_1001th_term_l1290_129013

theorem arithmetic_sequence_1001th_term (p q : ℚ)
    (h1 : p + 3 * q = 12)
    (h2 : 12 + 3 * q = 3 * p - q) :
    (p + (1001 - 1) * (3 * q) = 5545) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_1001th_term_l1290_129013


namespace NUMINAMATH_GPT_charlotte_one_way_journey_time_l1290_129030

def charlotte_distance : ℕ := 60
def charlotte_speed : ℕ := 10

theorem charlotte_one_way_journey_time :
  charlotte_distance / charlotte_speed = 6 :=
by
  sorry

end NUMINAMATH_GPT_charlotte_one_way_journey_time_l1290_129030


namespace NUMINAMATH_GPT_sum_of_interior_angles_heptagon_l1290_129061

theorem sum_of_interior_angles_heptagon (n : ℕ) (h : n = 7) : (n - 2) * 180 = 900 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_heptagon_l1290_129061


namespace NUMINAMATH_GPT_solve_variable_expression_l1290_129056

variable {x y : ℕ}

theorem solve_variable_expression
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (7 * x + 5 * y) / (x - 2 * y) = 26) :
  x = 3 * y :=
sorry

end NUMINAMATH_GPT_solve_variable_expression_l1290_129056


namespace NUMINAMATH_GPT_inscribed_square_side_length_l1290_129021

theorem inscribed_square_side_length (a h : ℝ) (ha_pos : 0 < a) (hh_pos : 0 < h) :
    ∃ x : ℝ, x = (h * a) / (a + h) :=
by
  -- Code here demonstrates the existence of x such that x = (h * a) / (a + h)
  sorry

end NUMINAMATH_GPT_inscribed_square_side_length_l1290_129021


namespace NUMINAMATH_GPT_question_l1290_129098

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Proposition A: x + y ≠ 8
def PropA : Prop := x + y ≠ 8

-- Proposition B: x ≠ 2 ∨ y ≠ 6
def PropB : Prop := x ≠ 2 ∨ y ≠ 6

-- We need to prove that PropA is a sufficient but not necessary condition for PropB.
theorem question : (PropA x y → PropB x y) ∧ ¬ (PropB x y → PropA x y) :=
sorry

end NUMINAMATH_GPT_question_l1290_129098


namespace NUMINAMATH_GPT_distinct_solutions_sub_l1290_129084

open Nat Real

theorem distinct_solutions_sub (p q : Real) (hpq_distinct : p ≠ q) (h_eqn_p : (p - 4) * (p + 4) = 17 * p - 68) (h_eqn_q : (q - 4) * (q + 4) = 17 * q - 68) (h_p_gt_q : p > q) : p - q = 9 := 
sorry

end NUMINAMATH_GPT_distinct_solutions_sub_l1290_129084


namespace NUMINAMATH_GPT_math_problem_l1290_129019

variable (a b c : ℝ)

variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a ≠ -b) (h5 : b ≠ -c) (h6 : c ≠ -a)

theorem math_problem 
    (h₁ : (a * b) / (a + b) = 4)
    (h₂ : (b * c) / (b + c) = 5)
    (h₃ : (c * a) / (c + a) = 7) :
    (a * b * c) / (a * b + b * c + c * a) = 280 / 83 := 
sorry

end NUMINAMATH_GPT_math_problem_l1290_129019


namespace NUMINAMATH_GPT_total_value_after_3_years_l1290_129065

noncomputable def value_after_years (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

def machine1_initial_value : ℝ := 2500
def machine1_depreciation_rate : ℝ := 0.05
def machine2_initial_value : ℝ := 3500
def machine2_depreciation_rate : ℝ := 0.07
def machine3_initial_value : ℝ := 4500
def machine3_depreciation_rate : ℝ := 0.04
def years : ℕ := 3

theorem total_value_after_3_years :
  value_after_years machine1_initial_value machine1_depreciation_rate years +
  value_after_years machine2_initial_value machine2_depreciation_rate years +
  value_after_years machine3_initial_value machine3_depreciation_rate years = 8940 :=
by
  sorry

end NUMINAMATH_GPT_total_value_after_3_years_l1290_129065


namespace NUMINAMATH_GPT_cost_of_article_l1290_129058

noncomputable def find_cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : Prop :=
  C = 168.57

theorem cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : 
  find_cost_of_article C G h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l1290_129058


namespace NUMINAMATH_GPT_no_integer_b_for_four_integer_solutions_l1290_129081

theorem no_integer_b_for_four_integer_solutions :
  ∀ (b : ℤ), ¬ ∃ x1 x2 x3 x4 : ℤ, 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (∀ x : ℤ, (x^2 + b*x + 1 ≤ 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)) :=
by sorry

end NUMINAMATH_GPT_no_integer_b_for_four_integer_solutions_l1290_129081


namespace NUMINAMATH_GPT_spring_length_5kg_weight_l1290_129090

variable {x y : ℝ}

-- Given conditions
def spring_length_no_weight : y = 6 := sorry
def spring_length_4kg_weight : y = 7.2 := sorry

-- The problem: to find the length of the spring for 5 kilograms
theorem spring_length_5kg_weight :
  (∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (b = 6) ∧ (4 * k + b = 7.2)) →
  y = 0.3 * 5 + 6 :=
  sorry

end NUMINAMATH_GPT_spring_length_5kg_weight_l1290_129090


namespace NUMINAMATH_GPT_vertex_of_parabola_l1290_129077

theorem vertex_of_parabola (a b c : ℝ) :
  (∀ x y : ℝ, (x = -2 ∧ y = 5) ∨ (x = 4 ∧ y = 5) ∨ (x = 2 ∧ y = 2) →
    y = a * x^2 + b * x + c) →
  (∃ x_vertex : ℝ, x_vertex = 1) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1290_129077


namespace NUMINAMATH_GPT_math_problem_l1290_129020

theorem math_problem
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2006)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2007)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2006)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2007)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2006)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2007)
  : (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = -1 / 2006 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1290_129020


namespace NUMINAMATH_GPT_find_f_13_l1290_129074

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem find_f_13 (f : ℝ → ℝ) 
  (h_period : periodic f 1.5) 
  (h_val : f 1 = 20) 
  : f 13 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_f_13_l1290_129074


namespace NUMINAMATH_GPT_intercepts_equal_l1290_129000

theorem intercepts_equal (a : ℝ) (ha : (a ≠ 0) ∧ (a ≠ 2)) : 
  (a = 1 ∨ a = 2) ↔ (a = 1 ∨ a = 2) := 
by 
  sorry


end NUMINAMATH_GPT_intercepts_equal_l1290_129000


namespace NUMINAMATH_GPT_find_ax5_plus_by5_l1290_129080

theorem find_ax5_plus_by5 (a b x y : ℝ)
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end NUMINAMATH_GPT_find_ax5_plus_by5_l1290_129080


namespace NUMINAMATH_GPT_probability_recurrence_relation_l1290_129025

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end NUMINAMATH_GPT_probability_recurrence_relation_l1290_129025


namespace NUMINAMATH_GPT_solution_of_equation_l1290_129003

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solution_of_equation (k : ℤ) (h : -1 ≤ k ∧ k ≤ 5) :
  ∃ x : ℝ, 4 * ↑(integer_part x) = 25 * fractional_part x - 4.5 ∧
           x = k + (8 * ↑k + 9) / 50 := 
sorry

end NUMINAMATH_GPT_solution_of_equation_l1290_129003


namespace NUMINAMATH_GPT_triangle_area_is_4_l1290_129062

variable {PQ RS : ℝ} -- lengths of PQ and RS respectively
variable {area_PQRS area_PQS : ℝ} -- areas of the trapezoid and triangle respectively

-- Given conditions
@[simp]
def trapezoid_area_is_12 (area_PQRS : ℝ) : Prop :=
  area_PQRS = 12

@[simp]
def RS_is_twice_PQ (PQ RS : ℝ) : Prop :=
  RS = 2 * PQ

-- To prove: the area of triangle PQS is 4 given the conditions
theorem triangle_area_is_4 (h1 : trapezoid_area_is_12 area_PQRS)
                          (h2 : RS_is_twice_PQ PQ RS)
                          (h3 : area_PQRS = 3 * area_PQS) : area_PQS = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_4_l1290_129062


namespace NUMINAMATH_GPT_longer_piece_length_l1290_129004

-- Conditions
def total_length : ℤ := 69
def is_cuts_into_two_pieces (a b : ℤ) : Prop := a + b = total_length
def is_twice_the_length (a b : ℤ) : Prop := a = 2 * b

-- Question: What is the length of the longer piece?
theorem longer_piece_length
  (a b : ℤ) 
  (H1: is_cuts_into_two_pieces a b)
  (H2: is_twice_the_length a b) :
  a = 46 :=
sorry

end NUMINAMATH_GPT_longer_piece_length_l1290_129004


namespace NUMINAMATH_GPT_sum_xyz_is_sqrt_13_l1290_129044

variable (x y z : ℝ)

-- The conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z

axiom eq1 : x^2 + y^2 + x * y = 3
axiom eq2 : y^2 + z^2 + y * z = 4
axiom eq3 : z^2 + x^2 + z * x = 7 

-- The theorem statement: Prove that x + y + z = sqrt(13)
theorem sum_xyz_is_sqrt_13 : x + y + z = Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_xyz_is_sqrt_13_l1290_129044


namespace NUMINAMATH_GPT_f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l1290_129076

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_2_2_eq_7 : f 2 2 = 7 :=
sorry

theorem f_3_3_eq_61 : f 3 3 = 61 :=
sorry

theorem f_4_4_can_be_evaluated : ∃ n, f 4 4 = n :=
sorry

end NUMINAMATH_GPT_f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l1290_129076


namespace NUMINAMATH_GPT_total_cost_of_items_l1290_129010

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end NUMINAMATH_GPT_total_cost_of_items_l1290_129010


namespace NUMINAMATH_GPT_coeff_of_nxy_n_l1290_129066

theorem coeff_of_nxy_n {n : ℕ} (degree_eq : 1 + n = 10) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_coeff_of_nxy_n_l1290_129066


namespace NUMINAMATH_GPT_log_a_properties_l1290_129011

noncomputable def log_a (a x : ℝ) (h : 0 < a ∧ a < 1) : ℝ := Real.log x / Real.log a

theorem log_a_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 1 < x → log_a a x h < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → log_a a x h > 0) ∧
  (¬ ∀ x1 x2 : ℝ, log_a a x1 h > log_a a x2 h → x1 > x2) ∧
  (∀ x y : ℝ, log_a a (x * y) h = log_a a x h + log_a a y h) :=
by
  sorry

end NUMINAMATH_GPT_log_a_properties_l1290_129011


namespace NUMINAMATH_GPT_right_triangle_area_l1290_129035

theorem right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) 
  (h_angle_sum : a = 45) (h_other_angle : b = 45) (h_right_angle : c = 90)
  (h_altitude : ∃ height : ℝ, height = 4) :
  ∃ area : ℝ, area = 8 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1290_129035
