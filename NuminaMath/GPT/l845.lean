import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l845_84510

theorem inequality_proof (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ Real.sqrt 3 * (x * y + y * z + z * x) := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l845_84510


namespace NUMINAMATH_GPT_find_integer_n_l845_84555

theorem find_integer_n (n : ℤ) (h1 : n ≥ 3) (h2 : ∃ k : ℚ, k * k = (n^2 - 5) / (n + 1)) : n = 3 := by
  sorry

end NUMINAMATH_GPT_find_integer_n_l845_84555


namespace NUMINAMATH_GPT_alice_bob_numbers_count_101_l845_84515

theorem alice_bob_numbers_count_101 : 
  ∃ n : ℕ, (∀ x, 3 ≤ x ∧ x ≤ 2021 → (∃ k l, x = 3 + 5 * k ∧ x = 2021 - 4 * l)) → n = 101 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_numbers_count_101_l845_84515


namespace NUMINAMATH_GPT_decreasing_interval_l845_84531

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_l845_84531


namespace NUMINAMATH_GPT_find_k_l845_84577

theorem find_k (k : ℝ) : (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l845_84577


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l845_84501

-- Question 1
theorem problem1 (a b : ℝ) (h : 5 * a + 3 * b = -4) : 2 * (a + b) + 4 * (2 * a + b) = -8 :=
by
  sorry

-- Question 2
theorem problem2 (a : ℝ) (h : a^2 + a = 3) : 2 * a^2 + 2 * a + 2023 = 2029 :=
by
  sorry

-- Question 3
theorem problem3 (a b : ℝ) (h : a - 2 * b = -3) : 3 * (a - b) - 7 * a + 11 * b + 2 = 14 :=
by
  sorry

-- Question 4
theorem problem4 (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) 
  (h2 : a * b - 2 * b^2 = -3) : a^2 + a * b + 2 * b^2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l845_84501


namespace NUMINAMATH_GPT_complex_number_location_in_plane_l845_84567

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem complex_number_location_in_plane :
  is_in_second_quadrant (-2) 5 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_location_in_plane_l845_84567


namespace NUMINAMATH_GPT_find_numbers_l845_84560

theorem find_numbers (N : ℕ) (a b : ℕ) :
  N = 5 * a →
  N = 7 * b →
  N = 35 ∨ N = 70 ∨ N = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l845_84560


namespace NUMINAMATH_GPT_jimin_and_seokjin_total_l845_84504

def Jimin_coins := (5 * 100) + (1 * 50)
def Seokjin_coins := (2 * 100) + (7 * 10)
def total_coins := Jimin_coins + Seokjin_coins

theorem jimin_and_seokjin_total : total_coins = 820 :=
by
  sorry

end NUMINAMATH_GPT_jimin_and_seokjin_total_l845_84504


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_32_l845_84524

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_32_l845_84524


namespace NUMINAMATH_GPT_solve_for_nabla_l845_84596

theorem solve_for_nabla (nabla : ℤ) (h : 5 * (-4) = nabla + 4) : nabla = -24 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_nabla_l845_84596


namespace NUMINAMATH_GPT_polar_to_cartesian_l845_84507

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.sin θ) : 
  ∀ (x y : ℝ) (h₁ : x = ρ * Real.cos θ) (h₂ : y = ρ * Real.sin θ), 
    x^2 + (y - 1)^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l845_84507


namespace NUMINAMATH_GPT_profit_equation_l845_84552

noncomputable def price_and_profit (x : ℝ) : ℝ :=
  (1 + 0.5) * x * 0.8 - x

theorem profit_equation : ∀ x : ℝ, price_and_profit x = 8 → ((1 + 0.5) * x * 0.8 - x = 8) :=
 by intros x h
    exact h

end NUMINAMATH_GPT_profit_equation_l845_84552


namespace NUMINAMATH_GPT_graveling_cost_correct_l845_84583

-- Define the dimensions of the rectangular lawn
def lawn_length : ℕ := 80 -- in meters
def lawn_breadth : ℕ := 50 -- in meters

-- Define the width of each road
def road_width : ℕ := 10 -- in meters

-- Define the cost per square meter for graveling the roads
def cost_per_sq_m : ℕ := 3 -- in Rs. per sq meter

-- Define the area of the road parallel to the length of the lawn
def area_road_parallel_length : ℕ := lawn_length * road_width

-- Define the effective length of the road parallel to the breadth of the lawn
def effective_road_parallel_breadth_length : ℕ := lawn_breadth - road_width

-- Define the area of the road parallel to the breadth of the lawn
def area_road_parallel_breadth : ℕ := effective_road_parallel_breadth_length * road_width

-- Define the total area to be graveled
def total_area_to_be_graveled : ℕ := area_road_parallel_length + area_road_parallel_breadth

-- Define the total cost of graveling
def total_graveling_cost : ℕ := total_area_to_be_graveled * cost_per_sq_m

-- Theorem: The total cost of graveling the two roads is Rs. 3600
theorem graveling_cost_correct : total_graveling_cost = 3600 := 
by
  unfold total_graveling_cost total_area_to_be_graveled area_road_parallel_length area_road_parallel_breadth effective_road_parallel_breadth_length lawn_length lawn_breadth road_width cost_per_sq_m
  exact rfl

end NUMINAMATH_GPT_graveling_cost_correct_l845_84583


namespace NUMINAMATH_GPT_a9_value_l845_84581

theorem a9_value (a : ℕ → ℝ) (x : ℝ) (h : (1 + x) ^ 10 = 
  (a 0) + (a 1) * (1 - x) + (a 2) * (1 - x)^2 + 
  (a 3) * (1 - x)^3 + (a 4) * (1 - x)^4 + 
  (a 5) * (1 - x)^5 + (a 6) * (1 - x)^6 + 
  (a 7) * (1 - x)^7 + (a 8) * (1 - x)^8 + 
  (a 9) * (1 - x)^9 + (a 10) * (1 - x)^10) : 
  a 9 = -20 :=
sorry

end NUMINAMATH_GPT_a9_value_l845_84581


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l845_84579

noncomputable def quadratic_roots_conditions (x1 x2 m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1)

noncomputable def existence_of_m (x1 x2 : ℝ) (m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1) ∧ ((x1 - 1) * (x2 - 1) = 6 / (m - 5))

theorem problem_part1 : 
  ∃ x2 m, quadratic_roots_conditions 1 x2 m :=
sorry

theorem problem_part2 :
  ∃ m, ∃ x2, existence_of_m 1 x2 m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l845_84579


namespace NUMINAMATH_GPT_average_weight_estimation_exclude_friend_l845_84588

theorem average_weight_estimation_exclude_friend
    (w : ℝ)
    (H1 : 62.4 < w ∧ w < 72.1)
    (H2 : 60.3 < w ∧ w < 70.6)
    (H3 : w ≤ 65.9)
    (H4 : 63.7 < w ∧ w < 66.3)
    (H5 : 75.0 ≤ w ∧ w ≤ 78.5) :
    False ∧ ((63.7 < w ∧ w ≤ 65.9) → (w = 64.8)) :=
by
  sorry

end NUMINAMATH_GPT_average_weight_estimation_exclude_friend_l845_84588


namespace NUMINAMATH_GPT_green_ball_count_l845_84557

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end NUMINAMATH_GPT_green_ball_count_l845_84557


namespace NUMINAMATH_GPT_turnip_count_example_l845_84564

theorem turnip_count_example : 6 + 9 = 15 := 
by
  -- Sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_turnip_count_example_l845_84564


namespace NUMINAMATH_GPT_triangle_area_l845_84582

theorem triangle_area (BC AC : ℝ) (angle_BAC : ℝ) (h1 : BC = 12) (h2 : AC = 5) (h3 : angle_BAC = π / 6) :
  1/2 * BC * (AC * Real.sin angle_BAC) = 15 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l845_84582


namespace NUMINAMATH_GPT_total_cost_of_crayons_l845_84550

-- Definition of the initial conditions
def usual_price : ℝ := 2.5
def discount_rate : ℝ := 0.15
def packs_initial : ℕ := 4
def packs_to_buy : ℕ := 2

-- Calculate the discounted price for one pack
noncomputable def discounted_price : ℝ :=
  usual_price - (usual_price * discount_rate)

-- Calculate the total cost of packs after purchase and validate it
theorem total_cost_of_crayons :
  (packs_initial * usual_price) + (packs_to_buy * discounted_price) = 14.25 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_crayons_l845_84550


namespace NUMINAMATH_GPT_problem_1_problem_2_l845_84548

-- Define the given conditions
variables (a c : ℝ) (cosB : ℝ)
variables (b : ℝ) (S : ℝ)

-- Assuming the values for the variables
axiom h₁ : a = 4
axiom h₂ : c = 3
axiom h₃ : cosB = 1 / 8

-- Prove that b = sqrt(22)
theorem problem_1 : b = Real.sqrt 22 := by
  sorry

-- Prove that the area of triangle ABC is 9 * sqrt(7) / 4
theorem problem_2 : S = 9 * Real.sqrt 7 / 4 := by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l845_84548


namespace NUMINAMATH_GPT_graph_equiv_l845_84594

theorem graph_equiv {x y : ℝ} :
  (x^3 - 2 * x^2 * y + x * y^2 - 2 * y^3 = 0) ↔ (x = 2 * y) :=
sorry

end NUMINAMATH_GPT_graph_equiv_l845_84594


namespace NUMINAMATH_GPT_bobs_fruit_drink_cost_l845_84556

theorem bobs_fruit_drink_cost
  (cost_soda : ℕ)
  (cost_hamburger : ℕ)
  (cost_sandwiches : ℕ)
  (bob_total_spent same_amount : ℕ)
  (andy_spent_eq : same_amount = cost_soda + 2 * cost_hamburger)
  (andy_bob_spent_eq : same_amount = bob_total_spent)
  (bob_sandwich_cost_eq : cost_sandwiches = 3)
  (andy_spent_eq_total : cost_soda = 1)
  (andy_burger_cost : cost_hamburger = 2)
  : bob_total_spent - cost_sandwiches = 2 :=
by
  sorry

end NUMINAMATH_GPT_bobs_fruit_drink_cost_l845_84556


namespace NUMINAMATH_GPT_ara_height_l845_84597

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end NUMINAMATH_GPT_ara_height_l845_84597


namespace NUMINAMATH_GPT_total_money_received_l845_84562

-- Define the given prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def adult_tickets_sold : ℕ := 90
def child_tickets_sold : ℕ := 40

-- Define the theorem to prove the total amount received
theorem total_money_received :
  (adult_ticket_price * adult_tickets_sold + child_ticket_price * child_tickets_sold) = 1240 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_money_received_l845_84562


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l845_84500

variable (a_2 a_4 a_3 : ℤ)

theorem arithmetic_sequence_problem (h : a_2 + a_4 = 16) : a_3 = 8 :=
by
  -- The proof is not needed as per the instructions
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l845_84500


namespace NUMINAMATH_GPT_trig_identity_example_l845_84521

theorem trig_identity_example:
  (Real.sin (63 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) + 
  Real.cos (63 * Real.pi / 180) * Real.cos (108 * Real.pi / 180)) = 
  Real.sqrt 2 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_example_l845_84521


namespace NUMINAMATH_GPT_second_set_parallel_lines_l845_84568

theorem second_set_parallel_lines (n : ℕ) (h : 7 * (n - 1) = 784) : n = 113 := 
by
  sorry

end NUMINAMATH_GPT_second_set_parallel_lines_l845_84568


namespace NUMINAMATH_GPT_number_exceeds_80_by_120_l845_84508

theorem number_exceeds_80_by_120 : ∃ x : ℝ, x = 0.80 * x + 120 ∧ x = 600 :=
by sorry

end NUMINAMATH_GPT_number_exceeds_80_by_120_l845_84508


namespace NUMINAMATH_GPT_remainder_of_99_times_101_divided_by_9_is_0_l845_84585

theorem remainder_of_99_times_101_divided_by_9_is_0 : (99 * 101) % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_99_times_101_divided_by_9_is_0_l845_84585


namespace NUMINAMATH_GPT_total_fencing_l845_84559

open Real

def playground_side_length : ℝ := 27
def garden_length : ℝ := 12
def garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13

theorem total_fencing : 
    4 * playground_side_length + 
    2 * (garden_length + garden_width) + 
    2 * Real.pi * flower_bed_radius + 
    (sandpit_side1 + sandpit_side2 + sandpit_side3) = 211.42 := 
    by sorry

end NUMINAMATH_GPT_total_fencing_l845_84559


namespace NUMINAMATH_GPT_minimum_cost_is_8600_l845_84525

-- Defining the conditions
def shanghai_units : ℕ := 12
def nanjing_units : ℕ := 6
def suzhou_needs : ℕ := 10
def changsha_needs : ℕ := 8
def cost_shanghai_suzhou : ℕ := 400
def cost_shanghai_changsha : ℕ := 800
def cost_nanjing_suzhou : ℕ := 300
def cost_nanjing_changsha : ℕ := 500

-- Defining the function for total shipping cost
def total_shipping_cost (x : ℕ) : ℕ :=
  cost_shanghai_suzhou * x +
  cost_shanghai_changsha * (shanghai_units - x) +
  cost_nanjing_suzhou * (suzhou_needs - x) +
  cost_nanjing_changsha * (x - (shanghai_units - suzhou_needs))

-- Define the minimum shipping cost function
def minimum_shipping_cost : ℕ :=
  total_shipping_cost 10

-- State the theorem to prove
theorem minimum_cost_is_8600 : minimum_shipping_cost = 8600 :=
sorry

end NUMINAMATH_GPT_minimum_cost_is_8600_l845_84525


namespace NUMINAMATH_GPT_count_square_of_integer_fraction_l845_84589

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_count_square_of_integer_fraction_l845_84589


namespace NUMINAMATH_GPT_total_pictures_l845_84576

-- Definitions based on problem conditions
def Randy_pictures : ℕ := 5
def Peter_pictures : ℕ := Randy_pictures + 3
def Quincy_pictures : ℕ := Peter_pictures + 20
def Susan_pictures : ℕ := 2 * Quincy_pictures - 7
def Thomas_pictures : ℕ := Randy_pictures ^ 3

-- The proof statement
theorem total_pictures : Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by
  sorry

end NUMINAMATH_GPT_total_pictures_l845_84576


namespace NUMINAMATH_GPT_interest_percentage_correct_l845_84528

noncomputable def encyclopedia_cost : ℝ := 1200
noncomputable def down_payment : ℝ := 500
noncomputable def monthly_payment : ℝ := 70
noncomputable def final_payment : ℝ := 45
noncomputable def num_monthly_payments : ℕ := 12
noncomputable def total_installment_payments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_cost_paid : ℝ := total_installment_payments + down_payment
noncomputable def amount_borrowed : ℝ := encyclopedia_cost - down_payment
noncomputable def interest_paid : ℝ := total_cost_paid - encyclopedia_cost
noncomputable def interest_percentage : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_percentage_correct : interest_percentage = 26.43 := by
  sorry

end NUMINAMATH_GPT_interest_percentage_correct_l845_84528


namespace NUMINAMATH_GPT_solve_equation_l845_84535

-- Define the equation as a Lean proposition
def equation (x : ℝ) : Prop :=
  (6 * x + 3) / (3 * x^2 + 6 * x - 9) = 3 * x / (3 * x - 3)

-- Define the solution set
def solution (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2

-- Define the condition to avoid division by zero
def valid (x : ℝ) : Prop := x ≠ 1

-- State the theorem
theorem solve_equation (x : ℝ) (h : equation x) (hv : valid x) : solution x :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l845_84535


namespace NUMINAMATH_GPT_min_ab_eq_11_l845_84519

theorem min_ab_eq_11 (a b : ℕ) (h : 23 * a - 13 * b = 1) : a + b = 11 :=
sorry

end NUMINAMATH_GPT_min_ab_eq_11_l845_84519


namespace NUMINAMATH_GPT_gcd_2873_1349_gcd_4562_275_l845_84506

theorem gcd_2873_1349 : Nat.gcd 2873 1349 = 1 := 
sorry

theorem gcd_4562_275 : Nat.gcd 4562 275 = 1 := 
sorry

end NUMINAMATH_GPT_gcd_2873_1349_gcd_4562_275_l845_84506


namespace NUMINAMATH_GPT_abc_inequality_l845_84570

theorem abc_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (a * (a^2 + b * c)) / (b + c) + (b * (b^2 + c * a)) / (c + a) + (c * (c^2 + a * b)) / (a + b) ≥ a * b + b * c + c * a := 
by 
  sorry

end NUMINAMATH_GPT_abc_inequality_l845_84570


namespace NUMINAMATH_GPT_total_sum_of_ages_l845_84584

theorem total_sum_of_ages (Y : ℕ) (interval : ℕ) (age1 age2 age3 age4 age5 : ℕ)
  (h1 : Y = 2) 
  (h2 : interval = 8) 
  (h3 : age1 = Y) 
  (h4 : age2 = Y + interval) 
  (h5 : age3 = Y + 2 * interval) 
  (h6 : age4 = Y + 3 * interval) 
  (h7 : age5 = Y + 4 * interval) : 
  age1 + age2 + age3 + age4 + age5 = 90 := 
by
  sorry

end NUMINAMATH_GPT_total_sum_of_ages_l845_84584


namespace NUMINAMATH_GPT_spadesuit_value_l845_84563

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end NUMINAMATH_GPT_spadesuit_value_l845_84563


namespace NUMINAMATH_GPT_school_avg_GPA_l845_84530

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end NUMINAMATH_GPT_school_avg_GPA_l845_84530


namespace NUMINAMATH_GPT_triangle_sine_inequality_l845_84587

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

end NUMINAMATH_GPT_triangle_sine_inequality_l845_84587


namespace NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l845_84537

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l845_84537


namespace NUMINAMATH_GPT_equation_of_parallel_plane_l845_84538

theorem equation_of_parallel_plane {A B C D : ℤ} (hA : A = 3) (hB : B = -2) (hC : C = 4) (hD : D = -16)
    (point : ℝ × ℝ × ℝ) (pass_through : point = (2, -3, 1)) (parallel_plane : A * 2 + B * (-3) + C * 1 + D = 0)
    (gcd_condition : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) :
    A * 2 + B * (-3) + C + D = 0 ∧ A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parallel_plane_l845_84538


namespace NUMINAMATH_GPT_angle_measure_is_zero_l845_84516

-- Definitions corresponding to conditions
variable (x : ℝ)

def complement (x : ℝ) := 90 - x
def supplement (x : ℝ) := 180 - x

-- Final proof statement
theorem angle_measure_is_zero (h : complement x = (1 / 2) * supplement x) : x = 0 :=
  sorry

end NUMINAMATH_GPT_angle_measure_is_zero_l845_84516


namespace NUMINAMATH_GPT_find_f_f_2_l845_84578

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_f_2 :
  f (f 2) = 14 :=
by
sorry

end NUMINAMATH_GPT_find_f_f_2_l845_84578


namespace NUMINAMATH_GPT_trigonometric_quadrant_l845_84536

theorem trigonometric_quadrant (θ : ℝ) (h1 : Real.sin θ > Real.cos θ) (h2 : Real.sin θ * Real.cos θ < 0) : 
  (θ > π / 2) ∧ (θ < π) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_quadrant_l845_84536


namespace NUMINAMATH_GPT_john_profit_proof_l845_84558

-- Define the conditions
variables 
  (parts_cost : ℝ := 800)
  (selling_price_multiplier : ℝ := 1.4)
  (monthly_build_quantity : ℝ := 60)
  (monthly_rent : ℝ := 5000)
  (monthly_extra_expenses : ℝ := 3000)

-- Define the computed variables based on conditions
def selling_price_per_computer := parts_cost * selling_price_multiplier
def total_revenue := monthly_build_quantity * selling_price_per_computer
def total_cost_of_components := monthly_build_quantity * parts_cost
def total_expenses := monthly_rent + monthly_extra_expenses
def profit_per_month := total_revenue - total_cost_of_components - total_expenses

-- The theorem statement of the proof
theorem john_profit_proof : profit_per_month = 11200 := 
by
  sorry

end NUMINAMATH_GPT_john_profit_proof_l845_84558


namespace NUMINAMATH_GPT_fixed_point_of_tangent_line_l845_84592

theorem fixed_point_of_tangent_line (x y : ℝ) (h1 : x = 3) 
  (h2 : ∃ m : ℝ, (3 - m)^2 + (y - 2)^2 = 4) :
  ∃ (k l : ℝ), k = 4 / 3 ∧ l = 2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_tangent_line_l845_84592


namespace NUMINAMATH_GPT_fraction_evaluation_l845_84514

theorem fraction_evaluation :
  (7 / 18 * (9 / 2) + 1 / 6) / ((40 / 3) - (15 / 4) / (5 / 16)) * (23 / 8) =
  4 + 17 / 128 :=
by
  -- conditions based on mixed number simplification
  have h1 : 4 + 1 / 2 = (9 : ℚ) / 2 := by sorry
  have h2 : 13 + 1 / 3 = (40 : ℚ) / 3 := by sorry
  have h3 : 3 + 3 / 4 = (15 : ℚ) / 4 := by sorry
  have h4 : 2 + 7 / 8 = (23 : ℚ) / 8 := by sorry
  -- the main proof
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l845_84514


namespace NUMINAMATH_GPT_wire_cut_equal_area_l845_84523

theorem wire_cut_equal_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a / b = 2 / Real.sqrt Real.pi) ↔ (a^2 / 16 = b^2 / (4 * Real.pi)) :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_equal_area_l845_84523


namespace NUMINAMATH_GPT_inequality_solution_l845_84505

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end NUMINAMATH_GPT_inequality_solution_l845_84505


namespace NUMINAMATH_GPT_maximum_value_of_k_l845_84513

noncomputable def max_k (m : ℝ) : ℝ := 
  if 0 < m ∧ m < 1 / 2 then 
    1 / m + 2 / (1 - 2 * m) 
  else 
    0

theorem maximum_value_of_k : ∀ m : ℝ, (0 < m ∧ m < 1 / 2) → (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m) ≥ k) → k ≤ 8) :=
  sorry

end NUMINAMATH_GPT_maximum_value_of_k_l845_84513


namespace NUMINAMATH_GPT_geometric_sequence_formula_l845_84599

def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (h_geom : geom_seq a)
  (h1 : a 3 = 2) (h2 : a 6 = 16) :
  ∀ n : ℕ, a n = 2 ^ (n - 2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_formula_l845_84599


namespace NUMINAMATH_GPT_find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l845_84511

theorem find_zeros_of_quadratic {a b : ℝ} (h_a : a = 1) (h_b : b = -2) :
  ∀ x, (a * x^2 + b * x + b - 1 = 0) ↔ (x = 3 ∨ x = -1) := sorry

theorem range_of_a_for_two_distinct_zeros :
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + b - 1 = 0 ∧ a * x2^2 + b * x2 + b - 1 = 0) ↔ (0 < a ∧ a < 1) := sorry

end NUMINAMATH_GPT_find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l845_84511


namespace NUMINAMATH_GPT_expand_polynomial_eq_l845_84571

theorem expand_polynomial_eq :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) = 6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_eq_l845_84571


namespace NUMINAMATH_GPT_percent_of_amount_l845_84575

theorem percent_of_amount (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  rw [hPart, hWhole]
  sorry

end NUMINAMATH_GPT_percent_of_amount_l845_84575


namespace NUMINAMATH_GPT_laura_running_speed_l845_84553

noncomputable def running_speed (x : ℝ) : ℝ := x^2 - 1

noncomputable def biking_speed (x : ℝ) : ℝ := 3 * x + 2

noncomputable def biking_time (x: ℝ) : ℝ := 30 / (biking_speed x)

noncomputable def running_time (x: ℝ) : ℝ := 5 / (running_speed x)

noncomputable def total_motion_time (x : ℝ) : ℝ := biking_time x + running_time x

-- Laura's total workout duration without transition time
noncomputable def required_motion_time : ℝ := 140 / 60

theorem laura_running_speed (x : ℝ) (hx : total_motion_time x = required_motion_time) :
  running_speed x = 83.33 :=
sorry

end NUMINAMATH_GPT_laura_running_speed_l845_84553


namespace NUMINAMATH_GPT_total_amount_after_interest_l845_84503

-- Define the constants
def principal : ℝ := 979.0209790209791
def rate : ℝ := 0.06
def time : ℝ := 2.4

-- Define the formula for interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the formula for the total amount after interest is added
def total_amount (P I : ℝ) : ℝ := P + I

-- State the theorem
theorem total_amount_after_interest : 
    total_amount principal (interest principal rate time) = 1120.0649350649352 :=
by
    -- placeholder for the proof
    sorry

end NUMINAMATH_GPT_total_amount_after_interest_l845_84503


namespace NUMINAMATH_GPT_max_xy_value_l845_84529

theorem max_xy_value (x y : ℕ) (h : 27 * x + 35 * y ≤ 1000) : x * y ≤ 252 :=
sorry

end NUMINAMATH_GPT_max_xy_value_l845_84529


namespace NUMINAMATH_GPT_edward_mowed_lawns_l845_84598

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end NUMINAMATH_GPT_edward_mowed_lawns_l845_84598


namespace NUMINAMATH_GPT_percentage_saved_is_25_l845_84580

def monthly_salary : ℝ := 1000

def increase_percentage : ℝ := 0.10

def saved_amount_after_increase : ℝ := 175

def calculate_percentage_saved (x : ℝ) : Prop := 
  1000 - (1000 - (x / 100) * monthly_salary) * (1 + increase_percentage) = saved_amount_after_increase

theorem percentage_saved_is_25 :
  ∃ x : ℝ, x = 25 ∧ calculate_percentage_saved x :=
sorry

end NUMINAMATH_GPT_percentage_saved_is_25_l845_84580


namespace NUMINAMATH_GPT_simplify_expression_l845_84509

theorem simplify_expression (x y z : ℤ) (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l845_84509


namespace NUMINAMATH_GPT_amount_of_b_l845_84572

variable (A B : ℝ)

theorem amount_of_b (h₁ : A + B = 2530) (h₂ : (3 / 5) * A = (2 / 7) * B) : B = 1714 :=
sorry

end NUMINAMATH_GPT_amount_of_b_l845_84572


namespace NUMINAMATH_GPT_cos_A_equals_one_third_l845_84545

-- Noncomputable context as trigonometric functions are involved.
noncomputable def cosA_in_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  let law_of_cosines : (a * Real.cos B) = (3 * c - b) * Real.cos A := sorry
  (Real.cos A = 1 / 3)

-- Define the problem statement to be proved
theorem cos_A_equals_one_third (a b c A B C : ℝ) 
  (h1 : a = Real.cos B)
  (h2 : a * Real.cos B = (3 * c - b) * Real.cos A) :
  Real.cos A = 1 / 3 := 
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cos_A_equals_one_third_l845_84545


namespace NUMINAMATH_GPT_mask_production_decrease_l845_84593

theorem mask_production_decrease (x : ℝ) : 
  (1 : ℝ) * (1 - x)^2 = 0.64 → 100 * (1 - x)^2 = 64 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mask_production_decrease_l845_84593


namespace NUMINAMATH_GPT_calculate_Delta_l845_84591

-- Define the Delta operation
def Delta (a b : ℚ) : ℚ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Constants for the specific problem
def two := (2 : ℚ)
def three := (3 : ℚ)
def four := (4 : ℚ)

theorem calculate_Delta : Delta (Delta two three) four = 5945 / 4073 := by
  sorry

end NUMINAMATH_GPT_calculate_Delta_l845_84591


namespace NUMINAMATH_GPT_evaluate_fraction_l845_84534

-- Define the custom operations x@y and x#y
def op_at (x y : ℝ) : ℝ := x * y - y^2
def op_hash (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- State the proof goal
theorem evaluate_fraction : (op_at 7 3) / (op_hash 7 3) = -3 :=
by
  -- Calculations to prove the theorem
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l845_84534


namespace NUMINAMATH_GPT_function_neither_even_nor_odd_l845_84549

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) := by
  sorry

end NUMINAMATH_GPT_function_neither_even_nor_odd_l845_84549


namespace NUMINAMATH_GPT_no_finite_set_A_exists_l845_84595

theorem no_finite_set_A_exists (A : Set ℕ) (h : Finite A ∧ ∀ a ∈ A, 2 * a ∈ A ∨ a / 3 ∈ A) : False :=
sorry

end NUMINAMATH_GPT_no_finite_set_A_exists_l845_84595


namespace NUMINAMATH_GPT_ryan_bus_meet_exactly_once_l845_84533

-- Define respective speeds of Ryan and the bus
def ryan_speed : ℕ := 6 
def bus_speed : ℕ := 15 

-- Define bench placement and stop times
def bench_distance : ℕ := 300 
def regular_stop_time : ℕ := 45 
def extra_stop_time : ℕ := 90 

-- Initial positions
def ryan_initial_position : ℕ := 0
def bus_initial_position : ℕ := 300

-- Distance function D(t)
noncomputable def distance_at_time (t : ℕ) : ℤ :=
  let bus_travel_time : ℕ := 15  -- time for bus to travel 225 feet
  let bus_stop_time : ℕ := 45  -- time for bus to stop during regular stops
  let extended_stop_time : ℕ := 90  -- time for bus to stop during 3rd bench stops
  sorry -- calculation of distance function

-- Problem to prove: Ryan and the bus meet exactly once
theorem ryan_bus_meet_exactly_once : ∃ t₁ t₂ : ℕ, t₁ ≠ t₂ ∧ distance_at_time t₁ = 0 ∧ distance_at_time t₂ ≠ 0 := 
  sorry

end NUMINAMATH_GPT_ryan_bus_meet_exactly_once_l845_84533


namespace NUMINAMATH_GPT_overlap_difference_l845_84526

namespace GeometryBiology

noncomputable def total_students : ℕ := 350
noncomputable def geometry_students : ℕ := 210
noncomputable def biology_students : ℕ := 175

theorem overlap_difference : 
    let max_overlap := min geometry_students biology_students;
    let min_overlap := geometry_students + biology_students - total_students;
    max_overlap - min_overlap = 140 := 
by
  sorry

end GeometryBiology

end NUMINAMATH_GPT_overlap_difference_l845_84526


namespace NUMINAMATH_GPT_mean_and_sum_l845_84565

-- Define the sum of five numbers to be 1/3
def sum_of_five_numbers : ℚ := 1 / 3

-- Define the mean of these five numbers
def mean_of_five_numbers : ℚ := sum_of_five_numbers / 5

-- State the theorem
theorem mean_and_sum (h : sum_of_five_numbers = 1 / 3) :
  mean_of_five_numbers = 1 / 15 ∧ (mean_of_five_numbers + sum_of_five_numbers = 2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_mean_and_sum_l845_84565


namespace NUMINAMATH_GPT_car_owners_without_motorcycle_or_bicycle_l845_84569

noncomputable def total_adults := 500
noncomputable def car_owners := 400
noncomputable def motorcycle_owners := 200
noncomputable def bicycle_owners := 150
noncomputable def car_motorcycle_owners := 100
noncomputable def motorcycle_bicycle_owners := 50
noncomputable def car_bicycle_owners := 30

theorem car_owners_without_motorcycle_or_bicycle :
  car_owners - car_motorcycle_owners - car_bicycle_owners = 270 := by
  sorry

end NUMINAMATH_GPT_car_owners_without_motorcycle_or_bicycle_l845_84569


namespace NUMINAMATH_GPT_number_of_valid_pairs_l845_84527

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 4950 ∧ ∀ (x y : ℕ), 
  1 ≤ x ∧ x < y ∧ y ≤ 200 ∧ 
  (Complex.I ^ x + Complex.I ^ y).im = 0 → n = 4950 :=
sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l845_84527


namespace NUMINAMATH_GPT_minimum_value_of_eccentricity_sum_l845_84547

variable {a b m n c : ℝ} (ha : a > b) (hb : b > 0) (hm : m > 0) (hn : n > 0)
variable {e1 e2 : ℝ}

theorem minimum_value_of_eccentricity_sum 
  (h_equiv : a^2 + m^2 = 2 * c^2) 
  (e1_def : e1 = c / a) 
  (e2_def : e2 = c / m) : 
  (2 * e1^2 + (e2^2) / 2) = (9 / 4) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_eccentricity_sum_l845_84547


namespace NUMINAMATH_GPT_nat_numbers_equal_if_divisible_l845_84542

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end NUMINAMATH_GPT_nat_numbers_equal_if_divisible_l845_84542


namespace NUMINAMATH_GPT_number_line_4_units_away_l845_84543

theorem number_line_4_units_away (x : ℝ) : |x + 3.2| = 4 ↔ (x = 0.8 ∨ x = -7.2) :=
by
  sorry

end NUMINAMATH_GPT_number_line_4_units_away_l845_84543


namespace NUMINAMATH_GPT_probability_penny_dime_halfdollar_tails_is_1_over_8_l845_84554

def probability_penny_dime_halfdollar_tails : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_penny_dime_halfdollar_tails_is_1_over_8 :
  probability_penny_dime_halfdollar_tails = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_penny_dime_halfdollar_tails_is_1_over_8_l845_84554


namespace NUMINAMATH_GPT_higher_room_amount_higher_60_l845_84573

variable (higher_amount : ℕ)

theorem higher_room_amount_higher_60 
  (total_rent : ℕ) (amount_credited_50 : ℕ)
  (total_reduction : ℕ)
  (condition1 : total_rent = 400)
  (condition2 : amount_credited_50 = 50)
  (condition3 : total_reduction = total_rent / 4)
  (condition4 : 10 * higher_amount - 10 * amount_credited_50 = total_reduction) :
  higher_amount = 60 := 
sorry

end NUMINAMATH_GPT_higher_room_amount_higher_60_l845_84573


namespace NUMINAMATH_GPT_find_f_2004_l845_84512

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom odd_g : ∀ x : ℝ, g (-x) = -g x
axiom g_eq_f_shift : ∀ x : ℝ, g x = f (x - 1)
axiom g_one : g 1 = 2003

theorem find_f_2004 : f 2004 = 2003 :=
  sorry

end NUMINAMATH_GPT_find_f_2004_l845_84512


namespace NUMINAMATH_GPT_walking_time_l845_84502

theorem walking_time (r s : ℕ) (h₁ : r + s = 50) (h₂ : 2 * s = 30) : 2 * r = 70 :=
by
  sorry

end NUMINAMATH_GPT_walking_time_l845_84502


namespace NUMINAMATH_GPT_gcd_digits_bounded_by_lcm_l845_84566

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end NUMINAMATH_GPT_gcd_digits_bounded_by_lcm_l845_84566


namespace NUMINAMATH_GPT_alpha_inverse_proportional_beta_l845_84546

theorem alpha_inverse_proportional_beta (α β : ℝ) (k : ℝ) :
  (∀ β1 α1, α1 * β1 = k) → (4 * 2 = k) → (β = -3) → (α = -8/3) :=
by
  sorry

end NUMINAMATH_GPT_alpha_inverse_proportional_beta_l845_84546


namespace NUMINAMATH_GPT_find_c_value_l845_84540

def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

theorem find_c_value (c : ℝ) :
  (∀ x < -1, deriv (f c) x < 0) ∧ 
  (∀ x, -1 < x → x < 0 → deriv (f c) x > 0) → 
  c = 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_c_value_l845_84540


namespace NUMINAMATH_GPT_find_cos_2beta_l845_84522

noncomputable def cos_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (htan : Real.tan α = 1 / 7) (hcos : Real.cos (α + β) = 2 * Real.sqrt 5 / 5) : Real :=
  2 * (Real.cos β)^2 - 1

theorem find_cos_2beta (α β : ℝ) (h1: 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2)
  (htan: Real.tan α = 1 / 7) (hcos: Real.cos (α + β) = 2 * Real.sqrt 5 / 5) :
  cos_2beta α β h1 h2 htan hcos = 4 / 5 := 
sorry

end NUMINAMATH_GPT_find_cos_2beta_l845_84522


namespace NUMINAMATH_GPT_number_of_integer_pairs_l845_84539

theorem number_of_integer_pairs (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_ineq : m^2 + m * n < 30) :
  ∃ k : ℕ, k = 48 :=
sorry

end NUMINAMATH_GPT_number_of_integer_pairs_l845_84539


namespace NUMINAMATH_GPT_smallest_base_l845_84517

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_smallest_base_l845_84517


namespace NUMINAMATH_GPT_problem_l845_84574

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : odd_function f
axiom f_property : ∀ x : ℝ, f (x + 2) = -f x
axiom f_at_1 : f 1 = 8

theorem problem : f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end NUMINAMATH_GPT_problem_l845_84574


namespace NUMINAMATH_GPT_calculate_l845_84520

def q (x y : ℤ) : ℤ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x < 0 ∧ y ≤ 0 then x - 3*y
  else 4*x + 2*y

theorem calculate : q (q 2 (-2)) (q (-3) 1) = -4 := 
  by
    sorry

end NUMINAMATH_GPT_calculate_l845_84520


namespace NUMINAMATH_GPT_car_count_is_150_l845_84541

variable (B C K : ℕ)  -- Define the variables representing buses, cars, and bikes

/-- Given conditions: The ratio of buses to cars to bikes is 3:7:10,
    there are 90 fewer buses than cars, and 140 fewer buses than bikes. -/
def conditions : Prop :=
  (C = (7 * B / 3)) ∧ (K = (10 * B / 3)) ∧ (C = B + 90) ∧ (K = B + 140)

theorem car_count_is_150 (h : conditions B C K) : C = 150 :=
by
  sorry

end NUMINAMATH_GPT_car_count_is_150_l845_84541


namespace NUMINAMATH_GPT_inversely_directly_proportional_l845_84586

theorem inversely_directly_proportional (m n z : ℝ) (x : ℝ) (h₁ : x = 4) (hz₁ : z = 16) (hz₂ : z = 64) (hy : ∃ y : ℝ, y = n * Real.sqrt z) (hx : ∃ m y : ℝ, x = m / y^2)
: x = 1 :=
by
  sorry

end NUMINAMATH_GPT_inversely_directly_proportional_l845_84586


namespace NUMINAMATH_GPT_garrison_men_initial_l845_84551

theorem garrison_men_initial (M : ℕ) (P : ℕ):
  (P = M * 40) →
  (P / 2 = (M + 2000) * 10) →
  M = 2000 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_garrison_men_initial_l845_84551


namespace NUMINAMATH_GPT_sum_of_remainders_eq_3_l845_84590

theorem sum_of_remainders_eq_3 (a b c : ℕ) (h1 : a % 59 = 28) (h2 : b % 59 = 15) (h3 : c % 59 = 19) (h4 : a = b + d ∨ b = c + d ∨ c = a + d) : 
  (a + b + c) % 59 = 3 :=
by {
  sorry -- Proof to be constructed
}

end NUMINAMATH_GPT_sum_of_remainders_eq_3_l845_84590


namespace NUMINAMATH_GPT_at_most_two_even_l845_84518

-- Assuming the negation of the proposition
def negate_condition (a b c : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0

-- Proposition to prove by contradiction
theorem at_most_two_even 
  (a b c : ℕ) 
  (h : negate_condition a b c) 
  : False :=
sorry

end NUMINAMATH_GPT_at_most_two_even_l845_84518


namespace NUMINAMATH_GPT_scientific_notation_of_138000_l845_84532

noncomputable def scientific_notation_equivalent (n : ℕ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * (10:ℝ)^exp

theorem scientific_notation_of_138000 : scientific_notation_equivalent 138000 1.38 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_138000_l845_84532


namespace NUMINAMATH_GPT_ball_reaches_20_feet_at_1_75_seconds_l845_84561

noncomputable def ball_height (t : ℝ) : ℝ :=
  60 - 9 * t - 8 * t ^ 2

theorem ball_reaches_20_feet_at_1_75_seconds :
  ∃ t : ℝ, ball_height t = 20 ∧ t = 1.75 ∧ t ≥ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_ball_reaches_20_feet_at_1_75_seconds_l845_84561


namespace NUMINAMATH_GPT_subset_A_has_only_one_element_l845_84544

theorem subset_A_has_only_one_element (m : ℝ) :
  (∀ x y, (mx^2 + 2*x + 1 = 0) → (mx*y^2 + 2*y + 1 = 0) → x = y) →
  (m = 0 ∨ m = 1) :=
by
  sorry

end NUMINAMATH_GPT_subset_A_has_only_one_element_l845_84544
