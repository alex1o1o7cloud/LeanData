import Mathlib

namespace algebra_inequality_l840_84099

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality_l840_84099


namespace maria_zoo_ticket_discount_percentage_l840_84070

theorem maria_zoo_ticket_discount_percentage 
  (regular_price : ℝ) (paid_price : ℝ) (discount_percentage : ℝ)
  (h1 : regular_price = 15) (h2 : paid_price = 9) :
  discount_percentage = 40 :=
by
  sorry

end maria_zoo_ticket_discount_percentage_l840_84070


namespace product_eq_5832_l840_84071

theorem product_eq_5832 (P Q R S : ℕ) 
(h1 : P + Q + R + S = 48)
(h2 : P + 3 = Q - 3)
(h3 : Q - 3 = R * 3)
(h4 : R * 3 = S / 3) :
P * Q * R * S = 5832 := sorry

end product_eq_5832_l840_84071


namespace least_value_l840_84058

-- Define the quadratic function and its conditions
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + q * x + r

-- Define the conditions for p, q, and r
def conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ (q^2 - 4 * p * r < 0)

-- State the theorem that given the conditions the least value is (4pr - q^2) / 4p
theorem least_value (p q r : ℝ) (h : conditions p q r) :
  ∃ x : ℝ, (∀ y : ℝ, quadratic_function p q r y ≥ quadratic_function p q r x) ∧
  quadratic_function p q r x = (4 * p * r - q^2) / (4 * p) :=
sorry

end least_value_l840_84058


namespace train_length_l840_84088

theorem train_length
  (S : ℝ)
  (L : ℝ)
  (h1 : L + 140 = S * 15)
  (h2 : L + 250 = S * 20) :
  L = 190 :=
by
  -- Proof to be provided here
  sorry

end train_length_l840_84088


namespace conversion_problems_l840_84038

-- Define the conversion factors
def square_meters_to_hectares (sqm : ℕ) : ℕ := sqm / 10000
def hectares_to_square_kilometers (ha : ℕ) : ℕ := ha / 100
def square_kilometers_to_hectares (sqkm : ℕ) : ℕ := sqkm * 100

-- Define the specific values from the problem
def value1_m2 : ℕ := 5000000
def value2_km2 : ℕ := 70000

-- The theorem to prove
theorem conversion_problems :
  (square_meters_to_hectares value1_m2 = 500) ∧
  (hectares_to_square_kilometers 500 = 5) ∧
  (square_kilometers_to_hectares value2_km2 = 7000000) :=
by
  sorry

end conversion_problems_l840_84038


namespace problem_statement_l840_84081

theorem problem_statement (x : ℤ) (y : ℝ) (h : y = 0.5) : 
  (⌈x + y⌉ - ⌊x + y⌋ = 1) ∧ (⌈x + y⌉ - (x + y) = 0.5) := 
by 
  sorry

end problem_statement_l840_84081


namespace inequality_of_factorials_and_polynomials_l840_84083

open Nat

theorem inequality_of_factorials_and_polynomials (m n : ℕ) (hm : m ≥ n) :
  2^n * n! ≤ (m+n)! / (m-n)! ∧ (m+n)! / (m-n)! ≤ (m^2 + m)^n :=
by
  sorry

end inequality_of_factorials_and_polynomials_l840_84083


namespace dimensions_of_triangle_from_square_l840_84024

theorem dimensions_of_triangle_from_square :
  ∀ (a : ℝ) (triangle : ℝ × ℝ × ℝ), 
    a = 10 →
    triangle = (a, a, a * Real.sqrt 2) →
    triangle = (10, 10, 10 * Real.sqrt 2) :=
by
  intros a triangle a_eq triangle_eq
  -- Proof
  sorry

end dimensions_of_triangle_from_square_l840_84024


namespace find_7c_plus_7d_l840_84008

noncomputable def f (c d x : ℝ) : ℝ := c * x + d
noncomputable def h (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 1

theorem find_7c_plus_7d (c d : ℝ) (h_def : ∀ x, h x = f_inv x - 5) (f_def : ∀ x, f c d x = c * x + d) (f_inv_def : ∀ x, f_inv x = 7 * x - 1) : 7 * c + 7 * d = 2 := by
  sorry

end find_7c_plus_7d_l840_84008


namespace employee_count_l840_84036

theorem employee_count (avg_salary : ℕ) (manager_salary : ℕ) (new_avg_increase : ℕ) (E : ℕ) :
  (avg_salary = 1500) ∧ (manager_salary = 4650) ∧ (new_avg_increase = 150) →
  1500 * E + 4650 = 1650 * (E + 1) → E = 20 :=
by
  sorry

end employee_count_l840_84036


namespace Vinnie_exceeded_word_limit_l840_84020

theorem Vinnie_exceeded_word_limit :
  let words_limit := 1000
  let words_saturday := 450
  let words_sunday := 650
  let total_words := words_saturday + words_sunday
  total_words - words_limit = 100 :=
by
  sorry

end Vinnie_exceeded_word_limit_l840_84020


namespace max_wickets_in_innings_l840_84062

-- Define the max wickets a bowler can take per over
def max_wickets_per_over : ℕ := 3

-- Define the number of overs bowled by the bowler
def overs_bowled : ℕ := 6

-- Assume the total players in a cricket team
def total_players : ℕ := 11

-- Lean statement that proves the maximum number of wickets the bowler can take in an innings
theorem max_wickets_in_innings :
  3 * 6 ≥ total_players - 1 →
  max_wickets_per_over * overs_bowled ≥ total_players - 1 :=
by
  sorry

end max_wickets_in_innings_l840_84062


namespace fraction_simplifies_l840_84074

-- Define the integers
def a : ℤ := 1632
def b : ℤ := 1625
def c : ℤ := 1645
def d : ℤ := 1612

-- Define the theorem to prove
theorem fraction_simplifies :
  (a^2 - b^2) / (c^2 - d^2) = 7 / 33 := by
  sorry

end fraction_simplifies_l840_84074


namespace center_circle_is_correct_l840_84068

noncomputable def find_center_of_circle : ℝ × ℝ :=
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
  let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
  let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
  let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
  (-18, -11)

theorem center_circle_is_correct (x y : ℝ) :
  (let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
   let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
   let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
   let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
   (x, y) = find_center_of_circle) :=
  sorry

end center_circle_is_correct_l840_84068


namespace oranges_sold_l840_84076

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l840_84076


namespace old_clock_slower_l840_84054

-- Given conditions
def old_clock_coincidence_minutes : ℕ := 66

-- Standard clock coincidences in 24 hours
def standard_clock_coincidences_in_24_hours : ℕ := 22

-- Standard 24 hours in minutes
def standard_24_hours_in_minutes : ℕ := 24 * 60

-- Total time for old clock in minutes over what should be 24 hours
def total_time_for_old_clock : ℕ := standard_clock_coincidences_in_24_hours * old_clock_coincidence_minutes

-- Problem statement: prove that the old clock's 24 hours is 12 minutes slower 
theorem old_clock_slower : total_time_for_old_clock = standard_24_hours_in_minutes + 12 := by
  sorry

end old_clock_slower_l840_84054


namespace solution_exists_l840_84089

def divide_sum_of_squares_and_quotient_eq_seventy_two (x : ℝ) : Prop :=
  (10 - x)^2 + x^2 + (10 - x) / x = 72

theorem solution_exists (x : ℝ) : divide_sum_of_squares_and_quotient_eq_seventy_two x → x = 2 := sorry

end solution_exists_l840_84089


namespace range_of_a_l840_84003

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ Real.exp 1) ↔ ∀ x : ℝ, 0 < x → a * Real.log (a * x) ≤ Real.exp x := 
by 
  sorry

end range_of_a_l840_84003


namespace compare_a_b_l840_84042

theorem compare_a_b (a b : ℝ) (h₁ : a = 1.9 * 10^5) (h₂ : b = 9.1 * 10^4) : a > b := by
  sorry

end compare_a_b_l840_84042


namespace jam_cost_is_162_l840_84043

theorem jam_cost_is_162 (N B J : ℕ) (h1 : N > 1) (h2 : 4 * B + 6 * J = 39) (h3 : N = 9) : 
  6 * N * J = 162 := 
by sorry

end jam_cost_is_162_l840_84043


namespace sum_of_squares_l840_84026

theorem sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 :=
by
  sorry

end sum_of_squares_l840_84026


namespace repeating_decimal_computation_l840_84018

noncomputable def x := 864 / 999
noncomputable def y := 579 / 999
noncomputable def z := 135 / 999

theorem repeating_decimal_computation :
  x - y - z = 50 / 333 :=
by
  sorry

end repeating_decimal_computation_l840_84018


namespace equation_1_solution_equation_2_solution_l840_84034

theorem equation_1_solution (x : ℝ) (h : (2 * x - 3)^2 = 9 * x^2) : x = 3 / 5 ∨ x = -3 :=
sorry

theorem equation_2_solution (x : ℝ) (h : 2 * x * (x - 2) + x = 2) : x = 2 ∨ x = -1 / 2 :=
sorry

end equation_1_solution_equation_2_solution_l840_84034


namespace inequality_not_always_true_l840_84049

-- Declare the variables and conditions
variables {a b c : ℝ}

-- Given conditions
axiom h1 : a < b 
axiom h2 : b < c 
axiom h3 : a * c < 0

-- Statement of the problem
theorem inequality_not_always_true : ¬ (∀ a b c, (a < b ∧ b < c ∧ a * c < 0) → (c^2 / a < b^2 / a)) :=
by { sorry }

end inequality_not_always_true_l840_84049


namespace sum_of_ais_l840_84014

theorem sum_of_ais :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ), 
    (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (a4 > 0) ∧ (a5 > 0) ∧ (a6 > 0) ∧ (a7 > 0) ∧ (a8 > 0) ∧
    a1^2 + (2*a2)^2 + (3*a3)^2 + (4*a4)^2 + (5*a5)^2 + (6*a6)^2 + (7*a7)^2 + (8*a8)^2 = 204 ∧
    a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 :=
by
  sorry

end sum_of_ais_l840_84014


namespace white_balls_in_bag_l840_84033

open BigOperators

theorem white_balls_in_bag (N : ℕ) (N_green : ℕ) (N_yellow : ℕ) (N_red : ℕ) (N_purple : ℕ)
  (prob_not_red_nor_purple : ℝ) (W : ℕ)
  (hN : N = 100)
  (hN_green : N_green = 30)
  (hN_yellow : N_yellow = 10)
  (hN_red : N_red = 47)
  (hN_purple : N_purple = 3)
  (h_prob_not_red_nor_purple : prob_not_red_nor_purple = 0.5) :
  W = 10 :=
sorry

end white_balls_in_bag_l840_84033


namespace find_angle_x_l840_84027

theorem find_angle_x (x : ℝ) (h1 : 3 * x + 2 * x = 90) : x = 18 :=
  by
    sorry

end find_angle_x_l840_84027


namespace constant_k_value_l840_84072

theorem constant_k_value 
  (S : ℕ → ℕ)
  (h : ∀ n : ℕ, S n = 4 * 3^(n + 1) - k) :
  k = 12 :=
sorry

end constant_k_value_l840_84072


namespace fraction_female_attendees_on_time_l840_84086

theorem fraction_female_attendees_on_time (A : ℝ) (h1 : A > 0) :
  let males_fraction := 3/5
  let males_on_time := 7/8
  let not_on_time := 0.155
  let total_on_time_fraction := 1 - not_on_time
  let males := males_fraction * A
  let males_arrived_on_time := males_on_time * males
  let females := (1 - males_fraction) * A
  let females_arrived_on_time_fraction := (total_on_time_fraction * A - males_arrived_on_time) / females
  females_arrived_on_time_fraction = 4/5 :=
by
  sorry

end fraction_female_attendees_on_time_l840_84086


namespace octahedron_tetrahedron_surface_area_ratio_l840_84087

theorem octahedron_tetrahedron_surface_area_ratio 
  (s : ℝ) 
  (h₁ : s = 1)
  (A_octahedron : ℝ := 2 * Real.sqrt 3)
  (A_tetrahedron : ℝ := Real.sqrt 3)
  (h₂ : A_octahedron = 2 * Real.sqrt 3 * s^2 / 2 * Real.sqrt 3 * (1/4) * s^2) 
  (h₃ : A_tetrahedron = Real.sqrt 3 * s^2 / 4)
  :
  A_octahedron / A_tetrahedron = 2 := 
by
  sorry

end octahedron_tetrahedron_surface_area_ratio_l840_84087


namespace ratio_of_areas_GHI_to_JKL_l840_84041

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l840_84041


namespace number_divided_by_21_l840_84057

theorem number_divided_by_21 (x : ℝ) (h : 6000 - (x / 21.0) = 5995) : x = 105 :=
by
  sorry

end number_divided_by_21_l840_84057


namespace sandy_initial_payment_l840_84028

theorem sandy_initial_payment (P : ℝ) (H1 : P + 300 < P + 1320)
  (H2 : 1320 = 1.10 * (P + 300)) : P = 900 :=
sorry

end sandy_initial_payment_l840_84028


namespace interest_rate_annual_l840_84021

theorem interest_rate_annual :
  ∃ R : ℝ, 
    (5000 * 2 * R / 100) + (3000 * 4 * R / 100) = 2640 ∧ 
    R = 12 :=
sorry

end interest_rate_annual_l840_84021


namespace frac_diff_zero_l840_84059

theorem frac_diff_zero (a b : ℝ) (h : a + b = a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a) - (1 / b) = 0 := 
sorry

end frac_diff_zero_l840_84059


namespace transportation_tax_correct_l840_84084

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l840_84084


namespace total_ways_to_choose_president_and_vice_president_of_same_gender_l840_84004

theorem total_ways_to_choose_president_and_vice_president_of_same_gender :
  let boys := 12
  let girls := 12
  (boys * (boys - 1) + girls * (girls - 1)) = 264 :=
by
  sorry

end total_ways_to_choose_president_and_vice_president_of_same_gender_l840_84004


namespace arrangement_count_l840_84047

def number_of_arrangements (n : ℕ) : ℕ :=
  if n = 6 then 5 * (Nat.factorial 5) else 0

theorem arrangement_count : number_of_arrangements 6 = 600 :=
by
  sorry

end arrangement_count_l840_84047


namespace tens_digit_of_sum_l840_84009

theorem tens_digit_of_sum (a b c : ℕ) (h : a = c + 3) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) :
    ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ (202 * c + 20 * b + 303) % 100 = t ∧ t / 10 = 1 :=
by
  use (20 * b + 3)
  sorry

end tens_digit_of_sum_l840_84009


namespace area_region_sum_l840_84022

theorem area_region_sum (r1 r2 : ℝ) (angle : ℝ) (a b c : ℕ) : 
  r1 = 6 → r2 = 3 → angle = 30 → (54 * Real.sqrt 3 + (9 : ℝ) * Real.pi - (9 : ℝ) * Real.pi = a * Real.sqrt b + c * Real.pi) → a + b + c = 10 :=
by
  intros
  -- We fill this with the actual proof steps later
  sorry

end area_region_sum_l840_84022


namespace arithmetic_sequence_sum_minimum_l840_84053

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_minimum (a1 : ℝ) (d : ℝ) :
  a1 = -20 ∧ (∀ n : ℕ, (S_n a1 d n) > (S_n a1 d 6)) → 
  (10 / 3 < d ∧ d < 4) := 
sorry

end arithmetic_sequence_sum_minimum_l840_84053


namespace dot_product_is_4_l840_84060

-- Define the vectors a and b
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (2, 2)

-- Define collinearity condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define k based on the collinearity condition
def k_value : ℝ := 1 -- derived from solving the collinearity condition in the problem

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of a and b is 4 when k = 1
theorem dot_product_is_4 {k : ℝ} (h : k = k_value) : dot_product (a k) b = 4 :=
by
  rw [h]
  sorry

end dot_product_is_4_l840_84060


namespace valid_patents_growth_l840_84064

variable (a b : ℝ)

def annual_growth_rate : ℝ := 0.23

theorem valid_patents_growth (h1 : b = (1 + annual_growth_rate)^2 * a) : b = (1 + 0.23)^2 * a :=
by
  sorry

end valid_patents_growth_l840_84064


namespace min_value_y_l840_84065

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 2) : 4 * a + b ≥ 8 :=
sorry

end min_value_y_l840_84065


namespace element_of_set_l840_84061

theorem element_of_set : -1 ∈ { x : ℝ | x^2 - 1 = 0 } :=
sorry

end element_of_set_l840_84061


namespace gcd_fx_x_l840_84056

noncomputable def f (x : ℕ) : ℕ := (5 * x + 3) * (8 * x + 2) * (12 * x + 7) * (3 * x + 11)

theorem gcd_fx_x (x : ℕ) (h : ∃ k : ℕ, x = 18720 * k) : Nat.gcd (f x) x = 462 :=
sorry

end gcd_fx_x_l840_84056


namespace intersection_points_of_parametric_curve_l840_84012

def parametric_curve_intersection_points (t : ℝ) : Prop :=
  let x := t - 1
  let y := t + 2
  (x = -3 ∧ y = 0) ∨ (x = 0 ∧ y = 3)

theorem intersection_points_of_parametric_curve :
  ∃ t1 t2 : ℝ, parametric_curve_intersection_points t1 ∧ parametric_curve_intersection_points t2 := 
by
  sorry

end intersection_points_of_parametric_curve_l840_84012


namespace percentage_saved_is_10_l840_84037

-- Given conditions
def rent_expenses : ℕ := 5000
def milk_expenses : ℕ := 1500
def groceries_expenses : ℕ := 4500
def education_expenses : ℕ := 2500
def petrol_expenses : ℕ := 2000
def misc_expenses : ℕ := 3940
def savings : ℕ := 2160

-- Define the total expenses
def total_expenses : ℕ := rent_expenses + milk_expenses + groceries_expenses + education_expenses + petrol_expenses + misc_expenses

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage of savings
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- Prove that the percentage saved is 10%
theorem percentage_saved_is_10 :
  percentage_saved = 10 :=
sorry

end percentage_saved_is_10_l840_84037


namespace complement_intersection_l840_84096

-- Definitions for the sets
def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2, 3}

-- Statement to be proved
theorem complement_intersection (hU : U = {0, 1, 2, 3}) (hA : A = {0, 1}) (hB : B = {1, 2, 3}) :
  ((U \ A) ∩ B) = {2, 3} :=
by
  -- Greek delta: skip proof details
  sorry

end complement_intersection_l840_84096


namespace inequality_selection_l840_84010

theorem inequality_selection (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) 
  (h₃ : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 ∧ (∀ x, |x + a| + |x - b| + c = 4 → x = (a - b)/2) ∧ (a = 8 / 7 ∧ b = 18 / 7 ∧ c = 2 / 7) :=
by
  sorry

end inequality_selection_l840_84010


namespace gcd_polynomial_l840_84097

theorem gcd_polynomial (b : ℤ) (h : 570 ∣ b) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 :=
sorry

end gcd_polynomial_l840_84097


namespace Bruce_bought_8_kg_of_grapes_l840_84095

-- Defining the conditions
def rate_grapes := 70
def rate_mangoes := 55
def weight_mangoes := 11
def total_paid := 1165

-- Result to be proven
def cost_mangoes := rate_mangoes * weight_mangoes
def total_cost_grapes (G : ℕ) := rate_grapes * G
def total_cost (G : ℕ) := (total_cost_grapes G) + cost_mangoes

theorem Bruce_bought_8_kg_of_grapes (G : ℕ) (h : total_cost G = total_paid) : G = 8 :=
by
  sorry  -- Proof omitted

end Bruce_bought_8_kg_of_grapes_l840_84095


namespace student_finished_6_problems_in_class_l840_84094

theorem student_finished_6_problems_in_class (total_problems : ℕ) (x y : ℕ) (h1 : total_problems = 15) (h2 : 3 * y = 2 * x) (h3 : x + y = total_problems) : y = 6 :=
sorry

end student_finished_6_problems_in_class_l840_84094


namespace ball_bounce_height_l840_84078

theorem ball_bounce_height
  (k : ℕ) 
  (h1 : 20 * (2 / 3 : ℝ)^k < 2) : 
  k = 7 :=
sorry

end ball_bounce_height_l840_84078


namespace angle_measure_is_60_l840_84017

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l840_84017


namespace largest_multiple_l840_84013

theorem largest_multiple (n : ℤ) (h8 : 8 ∣ n) (h : -n > -80) : n = 72 :=
by 
  sorry

end largest_multiple_l840_84013


namespace wall_height_l840_84090

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℝ := brick_volume * 6400

noncomputable def wall_length : ℝ := 800

noncomputable def wall_width : ℝ := 600

theorem wall_height :
  ∀ (wall_volume : ℝ), 
  wall_volume = total_brick_volume → 
  wall_volume = wall_length * wall_width * 22.48 :=
by
  sorry

end wall_height_l840_84090


namespace residue_of_927_mod_37_l840_84032

-- Define the condition of the problem, which is the modulus and the number
def modulus : ℤ := 37
def number : ℤ := -927

-- Define the statement we need to prove: that the residue of -927 mod 37 is 35
theorem residue_of_927_mod_37 : (number % modulus + modulus) % modulus = 35 := by
  sorry

end residue_of_927_mod_37_l840_84032


namespace evaluate_expression_l840_84045

theorem evaluate_expression (a b x y c : ℝ) (h1 : a = -b) (h2 : x * y = 1) (h3 : |c| = 2) :
  (c = 2 → (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2) ∧
  (c = -2 → (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2) := by
  sorry

end evaluate_expression_l840_84045


namespace solve_arithmetic_seq_l840_84082

theorem solve_arithmetic_seq (x : ℝ) (h : x > 0) (hx : x^2 = (4 + 16) / 2) : x = Real.sqrt 10 :=
sorry

end solve_arithmetic_seq_l840_84082


namespace lisa_quiz_goal_l840_84051

theorem lisa_quiz_goal (total_quizzes earned_A_on_first earned_A_goal remaining_quizzes additional_A_needed max_quizzes_below_A : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : earned_A_on_first = 30)
  (h3 : earned_A_goal = total_quizzes * 85 / 100)
  (h4 : remaining_quizzes = total_quizzes - 40)
  (h5 : additional_A_needed = earned_A_goal - earned_A_on_first)
  (h6 : max_quizzes_below_A = remaining_quizzes - additional_A_needed):
  max_quizzes_below_A = 0 :=
by sorry

end lisa_quiz_goal_l840_84051


namespace solve_for_x_y_l840_84030

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def triangle_ABC (A B C E F : V) (x y : ℝ) : Prop :=
  (E - A) = (1 / 2) • (B - A) ∧
  (C - F) = (2 : ℝ) • (A - F) ∧
  (E - F) = x • (B - A) + y • (C - A)

theorem solve_for_x_y (A B C E F : V) (x y : ℝ) :
  triangle_ABC A B C E F x y →
  x + y = - (1 / 6 : ℝ) :=
by
  sorry

end solve_for_x_y_l840_84030


namespace calculate_fraction_l840_84052

variables (n_bl: ℕ) (deg_warm: ℕ) (total_deg: ℕ) (total_bl: ℕ)

def blanket_fraction_added := total_deg / deg_warm

theorem calculate_fraction (h1: deg_warm = 3) (h2: total_deg = 21) (h3: total_bl = 14) :
  (blanket_fraction_added total_deg deg_warm) / total_bl = 1 / 2 :=
by {
  sorry
}

end calculate_fraction_l840_84052


namespace multiplication_verification_l840_84080

-- Define the variables
variables (P Q R S T U : ℕ)

-- Define the known digits in the numbers
def multiplicand := 60000 + 1000 * P + 100 * Q + 10 * R
def multiplier := 5000000 + 10000 * S + 1000 * T + 100 * U + 5

-- Define the proof statement
theorem multiplication_verification : 
  (multiplicand P Q R) * (multiplier S T U) = 20213 * 732575 :=
  sorry

end multiplication_verification_l840_84080


namespace three_people_on_staircase_l840_84002

theorem three_people_on_staircase (A B C : Type) (steps : Finset ℕ) (h1 : steps.card = 7) 
  (h2 : ∀ step ∈ steps, step ≤ 2) : 
  ∃ (total_ways : ℕ), total_ways = 336 :=
by {
  sorry
}

end three_people_on_staircase_l840_84002


namespace area_of_region_R_l840_84085

open Real

noncomputable def area_of_strip (width : ℝ) (height : ℝ) : ℝ :=
  width * height

noncomputable def area_of_triangle (leg : ℝ) : ℝ :=
  1 / 2 * leg * leg

theorem area_of_region_R :
  let unit_square_area := 1
  let AE_BE := 1 / sqrt 2
  let area_triangle_ABE := area_of_triangle AE_BE
  let strip_width := 1 / 4
  let strip_height := 1
  let area_strip := area_of_strip strip_width strip_height
  let overlap_area := area_triangle_ABE / 2
  let area_R := area_strip - overlap_area
  area_R = 1 / 8 :=
by
  sorry

end area_of_region_R_l840_84085


namespace quadratic_coefficient_c_l840_84006

theorem quadratic_coefficient_c (b c: ℝ) 
  (h_sum: 12 = b) (h_prod: 20 = c) : 
  c = 20 := 
by sorry

end quadratic_coefficient_c_l840_84006


namespace charlie_and_elle_crayons_l840_84055

theorem charlie_and_elle_crayons :
  (∃ (Lizzie Bobbie Billie Charlie Dave Elle : ℕ),
  Billie = 18 ∧
  Bobbie = 3 * Billie ∧
  Lizzie = Bobbie / 2 ∧
  Charlie = 2 * Lizzie ∧
  Dave = 4 * Billie ∧
  Elle = (Bobbie + Dave) / 2 ∧
  Charlie + Elle = 117) :=
sorry

end charlie_and_elle_crayons_l840_84055


namespace class_A_scores_more_uniform_l840_84000

-- Define the variances of the test scores for classes A and B
def variance_A := 13.2
def variance_B := 26.26

-- Theorem: Prove that the scores of the 10 students from class A are more uniform than those from class B
theorem class_A_scores_more_uniform :
  variance_A < variance_B :=
  by
    -- Assume the given variances and state the comparison
    have h : 13.2 < 26.26 := by sorry
    exact h

end class_A_scores_more_uniform_l840_84000


namespace point_on_x_axis_right_of_origin_is_3_units_away_l840_84031

theorem point_on_x_axis_right_of_origin_is_3_units_away :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ P.1 > 0 ∧ dist (P.1, P.2) (0, 0) = 3 ∧ P = (3, 0) := 
by
  sorry

end point_on_x_axis_right_of_origin_is_3_units_away_l840_84031


namespace total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l840_84067

theorem total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = 1001 ∧ 2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l840_84067


namespace parallel_lines_cond_l840_84063

theorem parallel_lines_cond (a c : ℝ) :
    (∀ (x y : ℝ), (a * x - 2 * y - 1 = 0) ↔ (6 * x - 4 * y + c = 0)) → 
        (a = 3 ∧ ∃ (c : ℝ), c ≠ -2) ∨ (a = 3 ∧ c = -2) := 
sorry

end parallel_lines_cond_l840_84063


namespace regular_polygon_sides_l840_84069

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l840_84069


namespace ratio_of_stock_values_l840_84044

/-- Definitions and conditions -/
def value_expensive := 78
def shares_expensive := 14
def shares_other := 26
def total_assets := 2106

/-- The proof problem -/
theorem ratio_of_stock_values : 
  ∃ (V_other : ℝ), 26 * V_other = total_assets - (shares_expensive * value_expensive) ∧ 
  (value_expensive / V_other) = 2 :=
by
  sorry

end ratio_of_stock_values_l840_84044


namespace students_selected_milk_l840_84035

theorem students_selected_milk
    (total_students : ℕ)
    (students_soda students_milk students_juice : ℕ)
    (soda_percentage : ℚ)
    (milk_percentage : ℚ)
    (juice_percentage : ℚ)
    (h1 : soda_percentage = 0.7)
    (h2 : milk_percentage = 0.2)
    (h3 : juice_percentage = 0.1)
    (h4 : students_soda = 84)
    (h5 : total_students = students_soda / soda_percentage)
    : students_milk = total_students * milk_percentage :=
by
    sorry

end students_selected_milk_l840_84035


namespace grasshopper_visit_all_points_min_jumps_l840_84015

noncomputable def grasshopper_min_jumps : ℕ := 18

theorem grasshopper_visit_all_points_min_jumps (n m : ℕ) (h₁ : n = 2014) (h₂ : m = 18) :
  ∃ k : ℕ, k ≤ m ∧ (∀ i : ℤ, 0 ≤ i → i < n → ∃ j : ℕ, j < k ∧ (j * 57 + i * 10) % n = i) :=
sorry

end grasshopper_visit_all_points_min_jumps_l840_84015


namespace f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l840_84007

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (k x : ℝ) : ℝ := x^2 + k * x
noncomputable def a (x1 x2 : ℝ) : ℝ := (f x1 - f x2) / (x1 - x2)
noncomputable def b (z1 z2 k : ℝ) : ℝ := (g k z1 - g k z2) / (z1 - z2)

theorem f_is_increasing (x1 x2 : ℝ) (h : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) : a x1 x2 > 0 := by
  sorry

theorem exists_ratio_two (k : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = 2 * a x1 x2 := by
  sorry

theorem range_k_for_negative_two_ratio (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = -2 * a x1 x2) → k < -4 := by
  sorry

end f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l840_84007


namespace find_k_for_linear_dependence_l840_84050

structure vector2 :=
  (x : ℝ)
  (y : ℝ)

def linear_dependent (v1 v2 : vector2) :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
  c1 * v1.x + c2 * v2.x = 0 ∧
  c1 * v1.y + c2 * v2.y = 0

theorem find_k_for_linear_dependence :
  ∀ (k : ℝ), linear_dependent (vector2.mk 2 3) (vector2.mk 4 k) ↔ k = 6 :=
by sorry

end find_k_for_linear_dependence_l840_84050


namespace tangent_line_circle_midpoint_locus_l840_84040

/-- 
Let O be the circle x^2 + y^2 = 1,
M be the point (-1, -4), and
N be the point (2, 0).
-/
structure CircleTangentMidpointProblem where
  (x y : ℝ)
  (O_eq : x^2 + y^2 = 1)
  (M_eq : x = -1 ∧ y = -4)
  (N_eq : x = 2 ∧ y = 0)

/- Part (1) -/
theorem tangent_line_circle (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                            (Mx My : ℝ) : ((Mx = -1 ∧ My = -4) → 
                          
                            (x = -1 ∨ 15 * x - 8 * y - 17 = 0)) := by
  sorry

/- Part (2) -/
theorem midpoint_locus (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                       (Nx Ny : ℝ) : ((Nx = 2 ∧ Ny = 0) → 
                       
                       ((x-1)^2 + y^2 = 1 ∧ (0 ≤ x ∧ x < 1 / 2))) := by
  sorry

end tangent_line_circle_midpoint_locus_l840_84040


namespace remainder_of_sum_div_17_l840_84046

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l840_84046


namespace remaining_amount_after_shopping_l840_84066

theorem remaining_amount_after_shopping (initial_amount spent_percentage remaining_amount : ℝ)
  (h_initial : initial_amount = 4000)
  (h_spent : spent_percentage = 0.30)
  (h_remaining : remaining_amount = 2800) :
  initial_amount - (spent_percentage * initial_amount) = remaining_amount :=
by
  sorry

end remaining_amount_after_shopping_l840_84066


namespace total_boys_fraction_of_girls_l840_84079

theorem total_boys_fraction_of_girls
  (n : ℕ)
  (b1 g1 b2 g2 : ℕ)
  (h_equal_students : b1 + g1 = b2 + g2)
  (h_ratio_class1 : b1 / g1 = 2 / 3)
  (h_ratio_class2: b2 / g2 = 4 / 5) :
  ((b1 + b2) / (g1 + g2) = 19 / 26) :=
by sorry

end total_boys_fraction_of_girls_l840_84079


namespace find_abc_digits_l840_84092

theorem find_abc_digits (N : ℕ) (abcd : ℕ) (a b c d : ℕ) (hN : N % 10000 = abcd) (hNsq : N^2 % 10000 = abcd)
  (ha_ne_zero : a ≠ 0) (hb_ne_six : b ≠ 6) (hc_ne_six : c ≠ 6) : (a * 100 + b * 10 + c) = 106 :=
by
  -- The proof is omitted.
  sorry

end find_abc_digits_l840_84092


namespace total_students_is_48_l840_84005

-- Definitions according to the given conditions
def boys'_row := 24
def girls'_row := 24

-- Theorem based on the question and the correct answer
theorem total_students_is_48 :
  boys'_row + girls'_row = 48 :=
by
  sorry

end total_students_is_48_l840_84005


namespace remainder_of_3_pow_2023_mod_5_l840_84093

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l840_84093


namespace find_primes_l840_84016

theorem find_primes (A B C : ℕ) (hA : A < 20) (hB : B < 20) (hC : C < 20)
  (hA_prime : Prime A) (hB_prime : Prime B) (hC_prime : Prime C)
  (h_sum : A + B + C = 30) : 
  (A = 2 ∧ B = 11 ∧ C = 17) ∨ (A = 2 ∧ B = 17 ∧ C = 11) ∨ 
  (A = 11 ∧ B = 2 ∧ C = 17) ∨ (A = 11 ∧ B = 17 ∧ C = 2) ∨ 
  (A = 17 ∧ B = 2 ∧ C = 11) ∨ (A = 17 ∧ B = 11 ∧ C = 2) :=
sorry

end find_primes_l840_84016


namespace find_unknown_number_l840_84029

theorem find_unknown_number (x : ℕ) (hx1 : 100 % x = 16) (hx2 : 200 % x = 4) : x = 28 :=
by 
  sorry

end find_unknown_number_l840_84029


namespace sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l840_84077

theorem sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three : 
  ((Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3) := by
  sorry

end sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l840_84077


namespace greatest_drop_in_price_l840_84001

theorem greatest_drop_in_price (jan feb mar apr may jun : ℝ)
  (h_jan : jan = -0.50)
  (h_feb : feb = 2.00)
  (h_mar : mar = -2.50)
  (h_apr : apr = 3.00)
  (h_may : may = -0.50)
  (h_jun : jun = -2.00) :
  mar = -2.50 ∧ (mar ≤ jan ∧ mar ≤ may ∧ mar ≤ jun) :=
by
  sorry

end greatest_drop_in_price_l840_84001


namespace part_a_part_b_l840_84025

theorem part_a (N : ℕ) : ∃ (a : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i → i ≤ N → a i > 0) ∧ (∀ i : ℕ, 2 ≤ i → i ≤ N → a i > a (i - 1)) ∧ 
(∀ i j : ℕ, 1 ≤ i → i < j → j ≤ N → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 1 - (1 : ℚ) / a 2) := sorry

theorem part_b : ¬ ∃ (a : ℕ → ℕ), (∀ i : ℕ, a i > 0) ∧ (∀ i : ℕ, a i < a (i + 1)) ∧ 
(∀ i j : ℕ, i < j → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 0 - (1 : ℚ) / a 1) := sorry

end part_a_part_b_l840_84025


namespace functional_equation_solution_l840_84048

theorem functional_equation_solution (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x + y) * (f x - f y) = a * (x - y) * f (x + y)) :
  (a = 1 → ∃ α β : ℝ, ∀ x : ℝ, f x = α * x^2 + β * x) ∧
  (a ≠ 1 ∧ a ≠ 0 → ∀ x : ℝ, f x = 0) ∧
  (a = 0 → ∃ c : ℝ, ∀ x : ℝ, f x = c) :=
by sorry

end functional_equation_solution_l840_84048


namespace cost_to_fly_A_to_B_l840_84075

noncomputable def flight_cost (distance : ℕ) : ℕ := (distance * 10 / 100) + 100

theorem cost_to_fly_A_to_B :
  flight_cost 3250 = 425 :=
by
  sorry

end cost_to_fly_A_to_B_l840_84075


namespace height_difference_is_correct_l840_84039

-- Define the heights of the trees as rational numbers.
def maple_tree_height : ℚ := 10 + 1 / 4
def spruce_tree_height : ℚ := 14 + 1 / 2

-- Prove that the spruce tree is 19 3/4 feet taller than the maple tree.
theorem height_difference_is_correct :
  spruce_tree_height - maple_tree_height = 19 + 3 / 4 := 
sorry

end height_difference_is_correct_l840_84039


namespace isosceles_triangle_area_l840_84011

noncomputable def area_of_isosceles_triangle (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20) : ℝ :=
  1 / 2 * (2 * b) * 10

theorem isosceles_triangle_area (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20)
  (h3 : 2 * s + 2 * b = 40) : area_of_isosceles_triangle b s h1 h2 = 75 :=
sorry

end isosceles_triangle_area_l840_84011


namespace steinburg_marching_band_l840_84019

theorem steinburg_marching_band :
  ∃ n : ℤ, n > 0 ∧ 30 * n < 1200 ∧ 30 * n % 34 = 6 ∧ 30 * n = 720 := by
  sorry

end steinburg_marching_band_l840_84019


namespace paula_shirts_count_l840_84023

variable {P : Type}

-- Given conditions as variable definitions
def initial_money : ℕ := 109
def shirt_cost : ℕ := 11
def pants_cost : ℕ := 13
def money_left : ℕ := 74
def money_spent : ℕ := initial_money - money_left
def shirts_count : ℕ → ℕ := λ S => shirt_cost * S

-- Main proposition to prove
theorem paula_shirts_count (S : ℕ) (h : money_spent = shirts_count S + pants_cost) : 
  S = 2 := by
  /- 
    Following the steps of the proof:
    1. Calculate money spent is $35.
    2. Set up the equation $11S + 13 = 35.
    3. Solve for S.
  -/
  sorry

end paula_shirts_count_l840_84023


namespace solve_for_x_l840_84073

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (x y : ℝ) (h : 16 * (3:ℝ) ^ x = (7:ℝ) ^ (y + 4)) (hy : y = -4) :
  x = -4 * log 3 2 := by
  sorry

end solve_for_x_l840_84073


namespace least_possible_value_of_d_l840_84091

theorem least_possible_value_of_d
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  (z - x) = 9 := 
sorry

end least_possible_value_of_d_l840_84091


namespace distance_foci_of_hyperbola_l840_84098

noncomputable def distance_between_foci : ℝ :=
  8 * Real.sqrt 5

theorem distance_foci_of_hyperbola :
  ∃ A B : ℝ, (9 * A^2 - 36 * A - B^2 + 4 * B = 40) → distance_between_foci = 8 * Real.sqrt 5 :=
sorry

end distance_foci_of_hyperbola_l840_84098
