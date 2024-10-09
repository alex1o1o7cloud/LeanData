import Mathlib

namespace point_symmetric_to_line_l526_52608

-- Define the problem statement
theorem point_symmetric_to_line (M : ℝ × ℝ) (l : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (1, 4) →
  l = (1, -1) →
  (∃ a b, N = (a, b) ∧ a + b = 5 ∧ a - b = 1) →
  N = (3, 2) :=
by
  sorry

end point_symmetric_to_line_l526_52608


namespace marching_band_total_weight_l526_52688

def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drummer : ℕ := 15
def weight_percussionist : ℕ := 8

def uniform_trumpet : ℕ := 3
def uniform_clarinet : ℕ := 3
def uniform_trombone : ℕ := 4
def uniform_tuba : ℕ := 5
def uniform_drummer : ℕ := 6
def uniform_percussionist : ℕ := 3

def count_trumpet : ℕ := 6
def count_clarinet : ℕ := 9
def count_trombone : ℕ := 8
def count_tuba : ℕ := 3
def count_drummer : ℕ := 2
def count_percussionist : ℕ := 4

def total_weight_band : ℕ :=
  (count_trumpet * (weight_trumpet + uniform_trumpet)) +
  (count_clarinet * (weight_clarinet + uniform_clarinet)) +
  (count_trombone * (weight_trombone + uniform_trombone)) +
  (count_tuba * (weight_tuba + uniform_tuba)) +
  (count_drummer * (weight_drummer + uniform_drummer)) +
  (count_percussionist * (weight_percussionist + uniform_percussionist))

theorem marching_band_total_weight : total_weight_band = 393 :=
  by
  sorry

end marching_band_total_weight_l526_52688


namespace compute_div_mul_l526_52623

noncomputable def a : ℚ := 0.24
noncomputable def b : ℚ := 0.006

theorem compute_div_mul : ((a / b) * 2) = 80 := by
  sorry

end compute_div_mul_l526_52623


namespace price_difference_is_correct_l526_52675

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l526_52675


namespace range_of_a_l526_52643

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ (¬ ∃ x : ℝ, x < 0 ∧ |x| = a * x - a) ↔ (a > 1 ∨ a ≤ -1) :=
sorry

end range_of_a_l526_52643


namespace ana_bonita_age_gap_l526_52660

theorem ana_bonita_age_gap (A B n : ℚ) (h1 : A = 2 * B + 3) (h2 : A - 2 = 6 * (B - 2)) (h3 : A = B + n) : n = 6.25 :=
by
  sorry

end ana_bonita_age_gap_l526_52660


namespace prob_AB_diff_homes_l526_52678

-- Define the volunteers
inductive Volunteer : Type
| A | B | C | D | E

open Volunteer

-- Define the homes
inductive Home : Type
| home1 | home2

open Home

-- Total number of ways to distribute the volunteers
def total_ways : ℕ := 2^5  -- Each volunteer has independently 2 choices

-- Number of ways in which A and B are in different homes
def diff_ways : ℕ := 2 * 4 * 2^3  -- Split the problem down by cases for simplicity

-- Calculate the probability
def probability : ℚ := diff_ways / total_ways

-- The final statement to prove
theorem prob_AB_diff_homes : probability = 8 / 15 := sorry

end prob_AB_diff_homes_l526_52678


namespace problem_l526_52631

def x : ℕ := 660
def percentage_25_of_x : ℝ := 0.25 * x
def percentage_12_of_1500 : ℝ := 0.12 * 1500
def difference_of_percentages : ℝ := percentage_12_of_1500 - percentage_25_of_x

theorem problem : difference_of_percentages = 15 := by
  -- begin proof (content replaced by sorry)
  sorry

end problem_l526_52631


namespace length_of_bridge_l526_52691

-- Define the conditions
def length_of_train : ℝ := 750
def speed_of_train_kmh : ℝ := 120
def crossing_time : ℝ := 45
def wind_resistance_factor : ℝ := 0.10

-- Define the conversion from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 0.27778

-- Define the actual speed considering wind resistance
def actual_speed_ms (v : ℝ) (resistance : ℝ) : ℝ := (kmh_to_ms v) * (1 - resistance)

-- Define the total distance covered
def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem: Length of the bridge
theorem length_of_bridge : total_distance (actual_speed_ms speed_of_train_kmh wind_resistance_factor) crossing_time - length_of_train = 600 := by
  sorry

end length_of_bridge_l526_52691


namespace tetrahedron_edges_midpoint_distances_sum_l526_52638

theorem tetrahedron_edges_midpoint_distances_sum (a b c d e f m1 m2 m3 m4 m5 m6 : ℝ) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (m1^2 + m2^2 + m3^2 + m4^2 + m5^2 + m6^2) :=
sorry

end tetrahedron_edges_midpoint_distances_sum_l526_52638


namespace spending_difference_l526_52634

-- Define the given conditions
def ice_cream_cartons := 19
def yoghurt_cartons := 4
def ice_cream_cost_per_carton := 7
def yoghurt_cost_per_carton := 1

-- Calculate the total cost based on the given conditions
def total_ice_cream_cost := ice_cream_cartons * ice_cream_cost_per_carton
def total_yoghurt_cost := yoghurt_cartons * yoghurt_cost_per_carton

-- The statement to prove
theorem spending_difference :
  total_ice_cream_cost - total_yoghurt_cost = 129 :=
by
  sorry

end spending_difference_l526_52634


namespace inverse_proportion_inequality_l526_52612

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l526_52612


namespace triangular_pyramid_nonexistence_l526_52625

theorem triangular_pyramid_nonexistence
    (h : ℕ)
    (hb : ℕ)
    (P : ℕ)
    (h_eq : h = 60)
    (hb_eq : hb = 61)
    (P_eq : P = 62) :
    ¬ ∃ (a b c : ℝ), a + b + c = P ∧ 60^2 = 61^2 - (a^2 / 3) :=
by 
  sorry

end triangular_pyramid_nonexistence_l526_52625


namespace integer_multiplied_by_b_l526_52662

variable (a b : ℤ) (x : ℤ)

theorem integer_multiplied_by_b (h1 : -11 * a < 0) (h2 : x < 0) (h3 : (-11 * a * x) * (x * b) + a * b = 89) :
  x = -1 :=
by
  sorry

end integer_multiplied_by_b_l526_52662


namespace cody_games_remaining_l526_52627

-- Definitions based on the conditions
def initial_games : ℕ := 9
def games_given_away : ℕ := 4

-- Theorem statement
theorem cody_games_remaining : initial_games - games_given_away = 5 :=
by sorry

end cody_games_remaining_l526_52627


namespace smallest_number_of_coins_to_pay_up_to_2_dollars_l526_52616

def smallest_number_of_coins_to_pay_up_to (max_amount : Nat) : Nat :=
  sorry  -- This function logic needs to be defined separately

theorem smallest_number_of_coins_to_pay_up_to_2_dollars :
  smallest_number_of_coins_to_pay_up_to 199 = 11 :=
sorry

end smallest_number_of_coins_to_pay_up_to_2_dollars_l526_52616


namespace num_comfortable_butterflies_final_state_l526_52670

noncomputable def num_comfortable_butterflies (n : ℕ) : ℕ :=
  if h : 0 < n then
    n
  else
    0

theorem num_comfortable_butterflies_final_state {n : ℕ} (h : 0 < n):
  num_comfortable_butterflies n = n := by
  sorry

end num_comfortable_butterflies_final_state_l526_52670


namespace unknown_diagonal_length_l526_52618

noncomputable def rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem unknown_diagonal_length
  (area : ℝ) (d2 : ℝ) (h_area : area = 150)
  (h_d2 : d2 = 30) :
  rhombus_diagonal_length area d2 = 10 :=
  by
  rw [h_area, h_d2]
  -- Here, the essential proof would go
  -- Since solving would require computation,
  -- which we are omitting, we use:
  sorry

end unknown_diagonal_length_l526_52618


namespace triangle_angle_sum_l526_52614

theorem triangle_angle_sum (A : ℕ) (h1 : A = 55) (h2 : ∀ (B : ℕ), B = 2 * A) : (A + 2 * A = 165) :=
by
  sorry

end triangle_angle_sum_l526_52614


namespace original_number_l526_52664

theorem original_number (x : ℝ) (hx : 100000 * x = 5 * (1 / x)) : x = 0.00707 := 
by
  sorry

end original_number_l526_52664


namespace sqrt_simplify_l526_52655

theorem sqrt_simplify (p : ℝ) :
  (Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5)) =
  6 * p^4 * Real.sqrt (35 * p) :=
by
  sorry

end sqrt_simplify_l526_52655


namespace solve_for_m_l526_52617

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end solve_for_m_l526_52617


namespace sqrt13_decomposition_ten_plus_sqrt3_decomposition_l526_52686

-- For the first problem
theorem sqrt13_decomposition :
  let a := 3
  let b := Real.sqrt 13 - 3
  a^2 + b - Real.sqrt 13 = 6 := by
sorry

-- For the second problem
theorem ten_plus_sqrt3_decomposition :
  let x := 11
  let y := Real.sqrt 3 - 1
  x - y = 12 - Real.sqrt 3 := by
sorry

end sqrt13_decomposition_ten_plus_sqrt3_decomposition_l526_52686


namespace reconstruct_point_A_l526_52651

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A E' F' G' H' : V)

theorem reconstruct_point_A (E F G H : V) (p q r s : ℝ)
  (hE' : E' = 2 • F - E)
  (hF' : F' = 2 • G - F)
  (hG' : G' = 2 • H - G)
  (hH' : H' = 2 • E - H)
  : p = 1/4 ∧ q = 1/4  ∧ r = 1/4  ∧ s = 1/4  :=
by
  sorry

end reconstruct_point_A_l526_52651


namespace problem_statement_l526_52657

noncomputable def f (x : ℝ) (A : ℝ) (ϕ : ℝ) : ℝ := A * Real.cos (2 * x + ϕ)

theorem problem_statement {A ϕ : ℝ} (hA : A > 0) (hϕ : |ϕ| < π / 2)
  (h1 : f (-π / 4) A ϕ = 2 * Real.sqrt 2)
  (h2 : f 0 A ϕ = 2 * Real.sqrt 6)
  (h3 : f (π / 12) A ϕ = 2 * Real.sqrt 2)
  (h4 : f (π / 4) A ϕ = -2 * Real.sqrt 2)
  (h5 : f (π / 3) A ϕ = -2 * Real.sqrt 6) :
  ϕ = π / 6 ∧ f (5 * π / 12) A ϕ = -4 * Real.sqrt 2 := 
sorry

end problem_statement_l526_52657


namespace evaluate_expression_l526_52615

theorem evaluate_expression : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end evaluate_expression_l526_52615


namespace truck_sand_at_arrival_l526_52696

-- Definitions based on conditions in part a)
def initial_sand : ℝ := 4.1
def lost_sand : ℝ := 2.4

-- Theorem statement corresponding to part c)
theorem truck_sand_at_arrival : initial_sand - lost_sand = 1.7 :=
by
  -- "sorry" placeholder to skip the proof
  sorry

end truck_sand_at_arrival_l526_52696


namespace correct_operation_l526_52639

variable (a : ℝ)

theorem correct_operation : 
  (3 * a^2 + 2 * a^4 ≠ 5 * a^6) ∧
  (a^2 * a^3 ≠ a^6) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) ∧
  ((-2 * a^3)^2 = 4 * a^6) := by
  sorry

end correct_operation_l526_52639


namespace James_age_is_11_l526_52697

-- Define the ages of Julio and James.
def Julio_age := 36

-- The age condition in 14 years.
def Julio_age_in_14_years := Julio_age + 14

-- James' age in 14 years and the relation as per the condition.
def James_age_in_14_years (J : ℕ) := J + 14

-- The main proof statement.
theorem James_age_is_11 (J : ℕ) 
  (h1 : Julio_age_in_14_years = 2 * James_age_in_14_years J) : J = 11 :=
by
  sorry

end James_age_is_11_l526_52697


namespace no_solution_fraction_eq_l526_52656

theorem no_solution_fraction_eq (x : ℝ) : 
  (1 / (x - 2) = (1 - x) / (2 - x) - 3) → False := 
by 
  sorry

end no_solution_fraction_eq_l526_52656


namespace abs_val_neg_three_l526_52695

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l526_52695


namespace prime_divisors_difference_l526_52635

def prime_factors (n : ℕ) : ℕ := sorry -- definition placeholder

theorem prime_divisors_difference (n : ℕ) (hn : 0 < n) : 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ k - m = n ∧ prime_factors k - prime_factors m = 1 := 
sorry

end prime_divisors_difference_l526_52635


namespace quadratic_inequality_solution_l526_52642

theorem quadratic_inequality_solution (x : ℝ) : 16 ≤ x ∧ x ≤ 20 → x^2 - 36 * x + 323 ≤ 3 :=
by
  sorry

end quadratic_inequality_solution_l526_52642


namespace isosceles_triangle_perimeter_l526_52646

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₁ : a = 12) (h₂ : b = 12) (h₃ : c = 17) : a + b + c = 41 :=
by
  rw [h₁, h₂, h₃]
  norm_num

end isosceles_triangle_perimeter_l526_52646


namespace sum_of_squares_l526_52606

theorem sum_of_squares (x y z a b c k : ℝ)
  (h₁ : x * y = k * a)
  (h₂ : x * z = b)
  (h₃ : y * z = c)
  (hk : k ≠ 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) :=
by
  sorry

end sum_of_squares_l526_52606


namespace age_difference_l526_52619

variable (A B C : ℕ)

-- Conditions
def ages_total_condition (a b c : ℕ) : Prop :=
  a + b = b + c + 11

-- Proof problem statement
theorem age_difference (a b c : ℕ) (h : ages_total_condition a b c) : a - c = 11 :=
by
  sorry

end age_difference_l526_52619


namespace div_by_eleven_l526_52641

theorem div_by_eleven (a b : ℤ) (h : (a^2 + 9 * a * b + b^2) % 11 = 0) : 
  (a^2 - b^2) % 11 = 0 :=
sorry

end div_by_eleven_l526_52641


namespace reciprocals_sum_eq_neg_one_over_three_l526_52624

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l526_52624


namespace fraction_addition_l526_52689

theorem fraction_addition (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/9) : a + b = 47/36 :=
by
  rw [h1, h2]
  sorry

end fraction_addition_l526_52689


namespace main_theorem_l526_52609

variable (x : ℤ)

def H : ℤ := 12 - (3 + 7) + x
def T : ℤ := 12 - 3 + 7 + x

theorem main_theorem : H - T + x = -14 + x :=
by
  sorry

end main_theorem_l526_52609


namespace simplify_expression_l526_52663

variable (b c : ℝ)

theorem simplify_expression :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 :=
by sorry

end simplify_expression_l526_52663


namespace find_X_l526_52610

theorem find_X (X : ℝ) (h : 0.80 * X - 0.35 * 300 = 31) : X = 170 :=
by
  sorry

end find_X_l526_52610


namespace find_complex_number_l526_52667

open Complex

theorem find_complex_number (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
sorry

end find_complex_number_l526_52667


namespace sum_of_distinct_prime_divisors_1728_l526_52669

theorem sum_of_distinct_prime_divisors_1728 : 
  (2 + 3 = 5) :=
sorry

end sum_of_distinct_prime_divisors_1728_l526_52669


namespace henry_has_more_than_500_seeds_on_saturday_l526_52694

theorem henry_has_more_than_500_seeds_on_saturday :
  (∃ k : ℕ, 5 * 3^k > 500 ∧ k + 1 = 6) :=
sorry

end henry_has_more_than_500_seeds_on_saturday_l526_52694


namespace percentage_increase_is_50_l526_52677

def initialNumber := 80
def finalNumber := 120

theorem percentage_increase_is_50 : ((finalNumber - initialNumber) / initialNumber : ℝ) * 100 = 50 := 
by 
  sorry

end percentage_increase_is_50_l526_52677


namespace angle_B_in_progression_l526_52681

theorem angle_B_in_progression (A B C a b c : ℝ) (h1: A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) 
(h2: B - A = C - B) (h3: b^2 - a^2 = a * c) (h4: A + B + C = Real.pi) : 
B = 2 * Real.pi / 7 := sorry

end angle_B_in_progression_l526_52681


namespace total_items_count_l526_52650

theorem total_items_count :
  let old_women  := 7
  let mules      := 7
  let bags       := 7
  let loaves     := 7
  let knives     := 7
  let sheaths    := 7
  let sheaths_per_loaf := knives * sheaths
  let sheaths_per_bag := loaves * sheaths_per_loaf
  let sheaths_per_mule := bags * sheaths_per_bag
  let sheaths_per_old_woman := mules * sheaths_per_mule
  let total_sheaths := old_women * sheaths_per_old_woman

  let loaves_per_bag := loaves
  let loaves_per_mule := bags * loaves_per_bag
  let loaves_per_old_woman := mules * loaves_per_mule
  let total_loaves := old_women * loaves_per_old_woman

  let knives_per_loaf := knives
  let knives_per_bag := loaves * knives_per_loaf
  let knives_per_mule := bags * knives_per_bag
  let knives_per_old_woman := mules * knives_per_mule
  let total_knives := old_women * knives_per_old_woman

  let total_bags := old_women * mules * bags

  let total_mules := old_women * mules

  let total_items := total_sheaths + total_loaves + total_knives + total_bags + total_mules + old_women

  total_items = 137256 :=
by
  sorry

end total_items_count_l526_52650


namespace final_payment_order_450_l526_52648

noncomputable def finalPayment (orderAmount : ℝ) : ℝ :=
  let serviceCharge := if orderAmount < 500 then 0.04 * orderAmount
                      else if orderAmount < 1000 then 0.05 * orderAmount
                      else 0.06 * orderAmount
  let salesTax := if orderAmount < 500 then 0.05 * orderAmount
                  else if orderAmount < 1000 then 0.06 * orderAmount
                  else 0.07 * orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := if totalBeforeDiscount < 600 then 0.05 * totalBeforeDiscount
                  else if totalBeforeDiscount < 800 then 0.10 * totalBeforeDiscount
                  else 0.15 * totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_payment_order_450 :
  finalPayment 450 = 465.98 := by
  sorry

end final_payment_order_450_l526_52648


namespace urn_probability_four_each_l526_52693

def number_of_sequences := Nat.choose 6 3

def probability_of_sequence := (1/3) * (1/2) * (3/5) * (1/2) * (4/7) * (5/8)

def total_probability := number_of_sequences * probability_of_sequence

theorem urn_probability_four_each :
  total_probability = 5 / 14 := by
  -- proof goes here
  sorry

end urn_probability_four_each_l526_52693


namespace find_abc_l526_52671

open Polynomial

noncomputable def my_gcd_lcm_problem (a b c : ℤ) : Prop :=
  gcd (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X + 1 ∧
  lcm (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X^3 - 5*X^2 + 7*X - 3

theorem find_abc : ∀ (a b c : ℤ),
  my_gcd_lcm_problem a b c → a + b + c = -3 :=
by
  intros a b c h
  sorry

end find_abc_l526_52671


namespace midpoint_trajectory_l526_52658

theorem midpoint_trajectory (x y : ℝ) (x0 y0 : ℝ)
  (h_circle : x0^2 + y0^2 = 4)
  (h_tangent : x0 * x + y0 * y = 4)
  (h_x0 : x0 = 2 / x)
  (h_y0 : y0 = 2 / y) :
  x^2 * y^2 = x^2 + y^2 :=
sorry

end midpoint_trajectory_l526_52658


namespace distinct_values_of_expression_l526_52659

variable {u v x y z : ℝ}

theorem distinct_values_of_expression (hu : u + u⁻¹ = x) (hv : v + v⁻¹ = y)
  (hx_distinct : x ≠ y) (hx_abs : |x| ≥ 2) (hy_abs : |y| ≥ 2) :
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z = u * v + (u * v)⁻¹)) →
  ∃ n, n = 2 := by 
    sorry

end distinct_values_of_expression_l526_52659


namespace sum_of_squares_l526_52637

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hwx : w * x = a^2) 
  (hwy : w * y = b^2) 
  (hwz : w * z = c^2) 
  (hw : w ≠ 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := 
by
  sorry

end sum_of_squares_l526_52637


namespace cost_price_for_fabrics_l526_52640

noncomputable def total_cost_price (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  selling_price - (meters_sold * profit_per_meter)

noncomputable def cost_price_per_meter (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  total_cost_price meters_sold selling_price profit_per_meter / meters_sold

theorem cost_price_for_fabrics :
  cost_price_per_meter 45 6000 12 = 121.33 ∧
  cost_price_per_meter 60 10800 15 = 165 ∧
  cost_price_per_meter 30 3900 10 = 120 :=
by
  sorry

end cost_price_for_fabrics_l526_52640


namespace intersection_of_planes_is_line_l526_52690

-- Define the conditions as Lean 4 statements
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y + z - 8 = 0
def plane2 (x y z : ℝ) : Prop := x - 2 * y - 2 * z + 1 = 0

-- Define the canonical form of the line as a Lean 4 proposition
def canonical_line (x y z : ℝ) : Prop := 
  (x - 3) / -4 = y / 5 ∧ y / 5 = (z - 2) / -7

-- The theorem to state equivalence between conditions and canonical line equations
theorem intersection_of_planes_is_line :
  ∀ (x y z : ℝ), plane1 x y z → plane2 x y z → canonical_line x y z :=
by
  intros x y z h1 h2
  -- TODO: Insert proof here
  sorry

end intersection_of_planes_is_line_l526_52690


namespace part1_part2_l526_52644

noncomputable def f : ℝ → ℝ 
| x => if 0 ≤ x then 2^x - 1 else -2^(-x) + 1

theorem part1 (x : ℝ) (h : x < 0) : f x = -2^(-x) + 1 := sorry

theorem part2 (a : ℝ) : f a ≤ 3 ↔ a ≤ 2 := sorry

end part1_part2_l526_52644


namespace johns_new_total_lift_l526_52613

theorem johns_new_total_lift :
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 30 / 100.0
  let squat_loss := squat_loss_percentage * initial_squat
  let new_squat := initial_squat - squat_loss
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - 200
  new_squat + new_bench + new_deadlift = 1490 := 
by
  -- Proof will go here
  sorry

end johns_new_total_lift_l526_52613


namespace difference_in_cm_l526_52687

def line_length : ℝ := 80  -- The length of the line is 80.0 centimeters
def diff_length_factor : ℝ := 0.35  -- The difference factor in the terms of the line's length

theorem difference_in_cm (l : ℝ) (d : ℝ) (h₀ : l = 80) (h₁ : d = 0.35 * l) : d = 28 :=
by
  sorry

end difference_in_cm_l526_52687


namespace not_rectangle_determined_by_angle_and_side_l526_52698

axiom parallelogram_determined_by_two_sides_and_angle : Prop
axiom equilateral_triangle_determined_by_area : Prop
axiom square_determined_by_perimeter_and_side : Prop
axiom rectangle_determined_by_two_diagonals : Prop
axiom rectangle_determined_by_angle_and_side : Prop

theorem not_rectangle_determined_by_angle_and_side : ¬rectangle_determined_by_angle_and_side := 
sorry

end not_rectangle_determined_by_angle_and_side_l526_52698


namespace volume_of_pyramid_l526_52673

-- Definitions based on conditions
def regular_quadrilateral_pyramid (h r : ℝ) := 
  ∃ a : ℝ, ∃ S : ℝ, ∃ V : ℝ,
  a = 2 * h * ((h^2 - r^2) / r^2).sqrt ∧
  S = (2 * h * ((h^2 - r^2) / r^2).sqrt)^2 ∧
  V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2)

-- Lean 4 theorem statement
theorem volume_of_pyramid (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ∃ V : ℝ, V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2) :=
sorry

end volume_of_pyramid_l526_52673


namespace find_positive_real_solution_l526_52629

theorem find_positive_real_solution (x : ℝ) (h₁ : 0 < x) (h₂ : 1/2 * (4 * x ^ 2 - 4) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 410 :=
by
  sorry

end find_positive_real_solution_l526_52629


namespace minimum_ab_l526_52645

theorem minimum_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : ab + 2 = 2 * (a + b)) : ab ≥ 6 + 4 * Real.sqrt 2 :=
by
  sorry

end minimum_ab_l526_52645


namespace find_four_digit_number_l526_52672

def digits_sum (n : ℕ) : ℕ := (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
def digits_product (n : ℕ) : ℕ := (n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)

theorem find_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (digits_sum n) * (digits_product n) = 3990 :=
by
  -- The proof is omitted as instructed.
  sorry

end find_four_digit_number_l526_52672


namespace g_at_10_l526_52676

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 10
axiom f_inequality_1 : ∀ x : ℝ, f (x + 20) ≥ f x + 20
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1
def g (x : ℝ) : ℝ := f x - x + 1

-- Proof statement (no proof required)
theorem g_at_10 : g 10 = 10 := sorry

end g_at_10_l526_52676


namespace max_tetrahedron_volume_l526_52633

theorem max_tetrahedron_volume 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (right_triangle : ∃ A B C : Type, 
    ∃ (angle_C : ℝ) (h_angle_C : angle_C = π / 2), 
    ∃ (BC CA : ℝ), BC = a ∧ CA = b) : 
  ∃ V : ℝ, V = (a^2 * b^2) / (6 * (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end max_tetrahedron_volume_l526_52633


namespace negation_proposition_l526_52685

theorem negation_proposition : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l526_52685


namespace questionnaire_visitors_l526_52682

theorem questionnaire_visitors
  (V : ℕ)
  (E U : ℕ)
  (h1 : ∀ v : ℕ, v ∈ { x : ℕ | x ≠ E ∧ x ≠ U } → v = 110)
  (h2 : E = U)
  (h3 : 3 * V = 4 * (E + U - 110))
  : V = 440 :=
by
  sorry

end questionnaire_visitors_l526_52682


namespace min_value_of_expression_l526_52600

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) :
  2 * a + b ≥ 4 * Real.sqrt 2 - 3 := 
sorry

end min_value_of_expression_l526_52600


namespace calculator_display_exceeds_1000_after_three_presses_l526_52666

-- Define the operation of pressing the squaring key
def square_key (n : ℕ) : ℕ := n * n

-- Define the initial display number
def initial_display : ℕ := 3

-- Prove that after pressing the squaring key 3 times, the display is greater than 1000.
theorem calculator_display_exceeds_1000_after_three_presses : 
  square_key (square_key (square_key initial_display)) > 1000 :=
by
  sorry

end calculator_display_exceeds_1000_after_three_presses_l526_52666


namespace tiffany_cans_l526_52649

variable {M : ℕ}

theorem tiffany_cans : (M + 12 = 2 * M) → (M = 12) :=
by
  intro h
  sorry

end tiffany_cans_l526_52649


namespace no_solution_fractional_eq_l526_52602

theorem no_solution_fractional_eq (y : ℝ) (h : y ≠ 3) : 
  ¬ ( (y-2)/(y-3) = 2 - 1/(3-y) ) :=
by
  sorry

end no_solution_fractional_eq_l526_52602


namespace ratio_of_cakes_l526_52605

/-- Define the usual number of cheesecakes, muffins, and red velvet cakes baked in a week -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

/-- Define the total number of cakes usually baked in a week -/
def usual_cakes : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet_cakes

/-- Assume Carter baked this week a multiple of usual cakes, denoted as x -/
def multiple (x : ℕ) : Prop := usual_cakes * x = usual_cakes + 38

/-- Assume he baked usual_cakes + 38 equals 57 cakes -/
def total_cakes_this_week : ℕ := 57

/-- The theorem stating the problem: proving the ratio is 3:1 -/
theorem ratio_of_cakes (x : ℕ) (hx : multiple x) : 
  (total_cakes_this_week : ℚ) / (usual_cakes : ℚ) = (3 : ℚ) :=
by
  sorry

end ratio_of_cakes_l526_52605


namespace average_grade_of_female_students_l526_52647

theorem average_grade_of_female_students
  (avg_all_students : ℝ)
  (avg_male_students : ℝ)
  (num_males : ℕ)
  (num_females : ℕ)
  (total_students := num_males + num_females)
  (total_score_all_students := avg_all_students * total_students)
  (total_score_male_students := avg_male_students * num_males) :
  avg_all_students = 90 →
  avg_male_students = 87 →
  num_males = 8 →
  num_females = 12 →
  ((total_score_all_students - total_score_male_students) / num_females) = 92 := by
  intros h_avg_all h_avg_male h_num_males h_num_females
  sorry

end average_grade_of_female_students_l526_52647


namespace car_distance_covered_by_car_l526_52684

theorem car_distance_covered_by_car
  (V : ℝ)                               -- Initial speed of the car
  (D : ℝ)                               -- Distance covered by the car
  (h1 : D = V * 6)                      -- The car takes 6 hours to cover the distance at speed V
  (h2 : D = 56 * 9)                     -- The car takes 9 hours to cover the distance at speed 56
  : D = 504 :=                          -- Prove that the distance D is 504 kilometers
by
  sorry

end car_distance_covered_by_car_l526_52684


namespace value_of_f1_l526_52683

noncomputable def f (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 3

theorem value_of_f1 (m : ℝ) (h_increasing : ∀ x : ℝ, x ≥ -2 → 2 * x^2 - m * x + 3 ≤ 2 * (x + 1)^2 - m * (x + 1) + 3)
  (h_decreasing : ∀ x : ℝ, x ≤ -2 → 2 * (x - 1)^2 - m * (x - 1) + 3 ≤ 2 * x^2 - m * x + 3) : 
  f 1 (-8) = 13 := 
sorry

end value_of_f1_l526_52683


namespace tan_product_l526_52674

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l526_52674


namespace petals_per_rose_l526_52661

theorem petals_per_rose
    (roses_per_bush : ℕ)
    (bushes : ℕ)
    (bottles : ℕ)
    (oz_per_bottle : ℕ)
    (petals_per_oz : ℕ)
    (petals : ℕ)
    (ounces : ℕ := bottles * oz_per_bottle)
    (total_petals : ℕ := ounces * petals_per_oz)
    (petals_per_bush : ℕ := total_petals / bushes)
    (petals_per_rose : ℕ := petals_per_bush / roses_per_bush) :
    petals_per_oz = 320 →
    roses_per_bush = 12 →
    bushes = 800 →
    bottles = 20 →
    oz_per_bottle = 12 →
    petals_per_rose = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end petals_per_rose_l526_52661


namespace sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l526_52692

noncomputable def calculation (x y z : ℝ) : ℝ :=
  (Real.sqrt x * Real.sqrt y) / Real.sqrt z

theorem sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3 :
  calculation 12 27 3 = 6 * Real.sqrt 3 :=
by
  sorry

end sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l526_52692


namespace intersection_M_N_l526_52622

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l526_52622


namespace athlete_stable_performance_l526_52652

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l526_52652


namespace constant_term_of_first_equation_l526_52653

theorem constant_term_of_first_equation
  (y z : ℤ)
  (h1 : 2 * 20 - y - z = 40)
  (h2 : 3 * 20 + y - z = 20)
  (hx : 20 = 20) :
  4 * 20 + y + z = 80 := 
sorry

end constant_term_of_first_equation_l526_52653


namespace percent_of_dollar_in_pocket_l526_52679

theorem percent_of_dollar_in_pocket :
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  (nickel + 2 * dime + quarter + half_dollar = 100) →
  (100 / 100 * 100 = 100) :=
by
  intros
  sorry

end percent_of_dollar_in_pocket_l526_52679


namespace margin_expression_l526_52636

variable (C S M : ℝ)
variable (n : ℕ)

theorem margin_expression (h : M = (C + S) / n) : M = (2 * S) / (n + 1) :=
sorry

end margin_expression_l526_52636


namespace determine_q_l526_52668

theorem determine_q (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 4) : q = 2 := 
sorry

end determine_q_l526_52668


namespace scientific_notation_of_61345_05_billion_l526_52604

theorem scientific_notation_of_61345_05_billion :
  ∃ x : ℝ, (61345.05 * 10^9) = x ∧ x = 6.134505 * 10^12 :=
by
  sorry

end scientific_notation_of_61345_05_billion_l526_52604


namespace quadratic_has_two_distinct_real_roots_l526_52603

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l526_52603


namespace max_weight_of_chocolates_l526_52601

def max_total_weight (chocolates : List ℕ) (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) : ℕ :=
300

theorem max_weight_of_chocolates (chocolates : List ℕ)
  (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) :
  max_total_weight chocolates H_wt H_div = 300 :=
sorry

end max_weight_of_chocolates_l526_52601


namespace arrangement_problem_l526_52632
   
   def numberOfArrangements (n : Nat) : Nat :=
     n.factorial

   def exclusiveArrangements (total people : Nat) (positions : Nat) : Nat :=
     (positions.choose 2) * (total - 2).factorial

   theorem arrangement_problem : 
     (numberOfArrangements 5) - (exclusiveArrangements 5 3) = 84 := 
   by
     sorry
   
end arrangement_problem_l526_52632


namespace polygon_is_octahedron_l526_52621

theorem polygon_is_octahedron (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_is_octahedron_l526_52621


namespace graph_of_equation_l526_52630

theorem graph_of_equation {x y : ℝ} (h : (x - 2 * y)^2 = x^2 - 4 * y^2) :
  (y = 0) ∨ (x = 2 * y) :=
by
  sorry

end graph_of_equation_l526_52630


namespace problem_false_statements_l526_52628

noncomputable def statement_I : Prop :=
  ∀ x : ℝ, ⌊x + Real.pi⌋ = ⌊x⌋ + 3

noncomputable def statement_II : Prop :=
  ∀ x : ℝ, ⌊x + Real.sqrt 2⌋ = ⌊x⌋ + ⌊Real.sqrt 2⌋

noncomputable def statement_III : Prop :=
  ∀ x : ℝ, ⌊x * Real.pi⌋ = ⌊x⌋ * ⌊Real.pi⌋

theorem problem_false_statements : ¬(statement_I ∨ statement_II ∨ statement_III) := 
by
  sorry

end problem_false_statements_l526_52628


namespace trains_crossing_time_l526_52680

noncomputable def timeToCross (L1 L2 : ℕ) (v1 v2 : ℕ) : ℝ :=
  let total_distance := (L1 + L2 : ℝ)
  let relative_speed := ((v1 + v2) * 1000 / 3600 : ℝ) -- converting km/hr to m/s
  total_distance / relative_speed

theorem trains_crossing_time :
  timeToCross 140 160 60 40 = 10.8 := 
  by 
    sorry

end trains_crossing_time_l526_52680


namespace ferris_wheel_cost_l526_52607

theorem ferris_wheel_cost (roller_coaster_cost log_ride_cost zach_initial_tickets zach_additional_tickets total_tickets ferris_wheel_cost : ℕ) 
  (h1 : roller_coaster_cost = 7)
  (h2 : log_ride_cost = 1)
  (h3 : zach_initial_tickets = 1)
  (h4 : zach_additional_tickets = 9)
  (h5 : total_tickets = zach_initial_tickets + zach_additional_tickets)
  (h6 : total_tickets - (roller_coaster_cost + log_ride_cost) = ferris_wheel_cost) :
  ferris_wheel_cost = 2 := 
by
  sorry

end ferris_wheel_cost_l526_52607


namespace vacation_days_l526_52626

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def towels_per_day_per_person : ℕ := 1
def washer_capacity : ℕ := 14
def num_loads : ℕ := 6

def total_people : ℕ := num_families * people_per_family
def towels_per_day : ℕ := total_people * towels_per_day_per_person
def total_towels : ℕ := num_loads * washer_capacity

def days_at_vacation_rental := total_towels / towels_per_day

theorem vacation_days : days_at_vacation_rental = 7 := by
  sorry

end vacation_days_l526_52626


namespace smallest_side_of_triangle_l526_52654

theorem smallest_side_of_triangle (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) : 
  a > c ∧ b > c :=
by
  sorry

end smallest_side_of_triangle_l526_52654


namespace total_items_sold_at_garage_sale_l526_52611

-- Define the conditions for the problem
def items_more_expensive_than_radio : Nat := 16
def items_less_expensive_than_radio : Nat := 23

-- Declare the total number of items using the given conditions
theorem total_items_sold_at_garage_sale 
  (h1 : items_more_expensive_than_radio = 16)
  (h2 : items_less_expensive_than_radio = 23) :
  items_more_expensive_than_radio + 1 + items_less_expensive_than_radio = 40 :=
by
  sorry

end total_items_sold_at_garage_sale_l526_52611


namespace solve_equation_l526_52620

theorem solve_equation (x : ℝ) (hx : x ≠ 0) : 
  x^2 + 36 / x^2 = 13 ↔ (x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3) := by
  sorry

end solve_equation_l526_52620


namespace maximum_cookies_andy_could_have_eaten_l526_52699

theorem maximum_cookies_andy_could_have_eaten :
  ∃ x : ℤ, (x ≥ 0 ∧ 2 * x + (x - 3) + x = 30) ∧ (∀ y : ℤ, 0 ≤ y ∧ 2 * y + (y - 3) + y = 30 → y ≤ 8) :=
by {
  sorry
}

end maximum_cookies_andy_could_have_eaten_l526_52699


namespace complex_number_simplification_l526_52665

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : i * (1 - i) - 1 = i := 
by
  sorry

end complex_number_simplification_l526_52665
