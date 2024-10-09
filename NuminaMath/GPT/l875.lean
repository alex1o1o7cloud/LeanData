import Mathlib

namespace geometric_sequence_exists_l875_87524

theorem geometric_sequence_exists 
  (a r : ℚ)
  (h1 : a = 3)
  (h2 : a * r = 8 / 9)
  (h3 : a * r^2 = 32 / 81) : 
  r = 8 / 27 :=
by
  sorry

end geometric_sequence_exists_l875_87524


namespace exists_unique_subset_X_l875_87520

theorem exists_unique_subset_X :
  ∃ (X : Set ℤ), ∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n :=
sorry

end exists_unique_subset_X_l875_87520


namespace largest_divisor_39_l875_87562

theorem largest_divisor_39 (m : ℕ) (hm : 0 < m) (h : 39 ∣ m ^ 2) : 39 ∣ m :=
by sorry

end largest_divisor_39_l875_87562


namespace even_sum_probability_l875_87517

-- Conditions
def prob_even_first_wheel : ℚ := 1 / 4
def prob_odd_first_wheel : ℚ := 3 / 4
def prob_even_second_wheel : ℚ := 2 / 3
def prob_odd_second_wheel : ℚ := 1 / 3

-- Statement: Theorem that the probability of the sum being even is 5/12
theorem even_sum_probability : 
  (prob_even_first_wheel * prob_even_second_wheel) + 
  (prob_odd_first_wheel * prob_odd_second_wheel) = 5 / 12 :=
by
  -- Proof steps would go here
  sorry

end even_sum_probability_l875_87517


namespace largest_angle_in_pentagon_l875_87541

def pentagon_angle_sum : ℝ := 540

def angle_A : ℝ := 70
def angle_B : ℝ := 90
def angle_C (x : ℝ) : ℝ := x
def angle_D (x : ℝ) : ℝ := x
def angle_E (x : ℝ) : ℝ := 3 * x - 10

theorem largest_angle_in_pentagon
  (x : ℝ)
  (h_sum : angle_A + angle_B + angle_C x + angle_D x + angle_E x = pentagon_angle_sum) :
  angle_E x = 224 :=
sorry

end largest_angle_in_pentagon_l875_87541


namespace airplane_average_speed_l875_87505

theorem airplane_average_speed :
  ∃ v : ℝ, 
  (1140 = 12 * (0.9 * v) + 26 * (1.2 * v)) ∧ 
  v = 27.14 := 
by
  sorry

end airplane_average_speed_l875_87505


namespace necessary_and_sufficient_condition_l875_87531

variable (p q : Prop)

theorem necessary_and_sufficient_condition (h1 : p → q) (h2 : q → p) : (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l875_87531


namespace largest_n_for_divisibility_l875_87530

theorem largest_n_for_divisibility :
  ∃ n : ℕ, (n + 15) ∣ (n^3 + 250) ∧ ∀ m : ℕ, ((m + 15) ∣ (m^3 + 250)) → (m ≤ 10) → (n = 10) :=
by {
  sorry
}

end largest_n_for_divisibility_l875_87530


namespace cosine_double_angle_tangent_l875_87544

theorem cosine_double_angle_tangent (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
by
  sorry

end cosine_double_angle_tangent_l875_87544


namespace volume_between_spheres_l875_87599

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  volume_of_sphere 10 - volume_of_sphere 4 = (3744 / 3) * Real.pi := by
  sorry

end volume_between_spheres_l875_87599


namespace a_10_is_100_l875_87536

-- Define the sequence a_n as a function from ℕ+ (the positive naturals) to ℤ
axiom a : ℕ+ → ℤ

-- Given assumptions
axiom seq_relation : ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * m.val * n.val
axiom a1 : a 1 = 1

-- Goal statement
theorem a_10_is_100 : a 10 = 100 :=
by
  -- proof goes here, this is just the statement
  sorry

end a_10_is_100_l875_87536


namespace solution_is_correct_l875_87539

-- Define the conditions of the problem.
variable (x y z : ℝ)

-- The system of equations given in the problem
def system_of_equations (x y z : ℝ) :=
  (1/x + 1/(y+z) = 6/5) ∧
  (1/y + 1/(x+z) = 3/4) ∧
  (1/z + 1/(x+y) = 2/3)

-- The desired solution
def solution (x y z : ℝ) := x = 2 ∧ y = 3 ∧ z = 1

-- The theorem to prove
theorem solution_is_correct (h : system_of_equations x y z) : solution x y z :=
sorry

end solution_is_correct_l875_87539


namespace quadratic_inequality_solution_l875_87559

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
  sorry

end quadratic_inequality_solution_l875_87559


namespace salesmans_profit_l875_87529

-- Define the initial conditions and given values
def backpacks_bought : ℕ := 72
def cost_price : ℕ := 1080
def swap_meet_sales : ℕ := 25
def swap_meet_price : ℕ := 20
def department_store_sales : ℕ := 18
def department_store_price : ℕ := 30
def online_sales : ℕ := 12
def online_price : ℕ := 28
def shipping_expenses : ℕ := 40
def local_market_price : ℕ := 24

-- Calculate the total revenue from each channel
def swap_meet_revenue : ℕ := swap_meet_sales * swap_meet_price
def department_store_revenue : ℕ := department_store_sales * department_store_price
def online_revenue : ℕ := (online_sales * online_price) - shipping_expenses

-- Calculate remaining backpacks and local market revenue
def backpacks_sold : ℕ := swap_meet_sales + department_store_sales + online_sales
def backpacks_left : ℕ := backpacks_bought - backpacks_sold
def local_market_revenue : ℕ := backpacks_left * local_market_price

-- Calculate total revenue and profit
def total_revenue : ℕ := swap_meet_revenue + department_store_revenue + online_revenue + local_market_revenue
def profit : ℕ := total_revenue - cost_price

-- State the theorem for the salesman's profit
theorem salesmans_profit : profit = 664 := by
  sorry

end salesmans_profit_l875_87529


namespace driver_license_advantage_l875_87509

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end driver_license_advantage_l875_87509


namespace linear_function_m_value_l875_87556

theorem linear_function_m_value (m : ℝ) (h : abs (m + 1) = 1) : m = -2 :=
sorry

end linear_function_m_value_l875_87556


namespace part_a_part_b_part_c_l875_87511

-- Definitions for the problem
def hard_problem_ratio_a := 2 / 3
def unsolved_problem_ratio_a := 2 / 3
def well_performing_students_ratio_a := 2 / 3

def hard_problem_ratio_b := 3 / 4
def unsolved_problem_ratio_b := 3 / 4
def well_performing_students_ratio_b := 3 / 4

def hard_problem_ratio_c := 7 / 10
def unsolved_problem_ratio_c := 7 / 10
def well_performing_students_ratio_c := 7 / 10

-- Theorems to prove
theorem part_a : 
  ∃ (hard_problem_ratio_a unsolved_problem_ratio_a well_performing_students_ratio_a : ℚ),
  hard_problem_ratio_a == 2 / 3 ∧
  unsolved_problem_ratio_a == 2 / 3 ∧
  well_performing_students_ratio_a == 2 / 3 →
  (True) := sorry

theorem part_b : 
  ∀ (hard_problem_ratio_b : ℚ),
  hard_problem_ratio_b == 3 / 4 →
  (False) := sorry

theorem part_c : 
  ∀ (hard_problem_ratio_c : ℚ),
  hard_problem_ratio_c == 7 / 10 →
  (False) := sorry

end part_a_part_b_part_c_l875_87511


namespace shaded_area_l875_87598

theorem shaded_area (r1 r2 : ℝ) (h1 : r2 = 3 * r1) (h2 : r1 = 2) : 
  π * (r2 ^ 2) - π * (r1 ^ 2) = 32 * π :=
by
  sorry

end shaded_area_l875_87598


namespace sum_squares_6_to_14_l875_87589

def sum_of_squares (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_squares_6_to_14 :
  (sum_of_squares 14) - (sum_of_squares 5) = 960 :=
by
  sorry

end sum_squares_6_to_14_l875_87589


namespace sarah_shaded_area_l875_87582

theorem sarah_shaded_area (r : ℝ) (A_square : ℝ) (A_circle : ℝ) (A_circles : ℝ) (A_shaded : ℝ) :
  let side_length := 27
  let radius := side_length / (3 * 2)
  let area_square := side_length * side_length
  let area_one_circle := Real.pi * (radius * radius)
  let total_area_circles := 9 * area_one_circle
  let shaded_area := area_square - total_area_circles
  shaded_area = 729 - 182.25 * Real.pi := 
by
  sorry

end sarah_shaded_area_l875_87582


namespace avg_payment_correct_l875_87540

def first_payment : ℕ := 410
def additional_amount : ℕ := 65
def num_first_payments : ℕ := 8
def num_remaining_payments : ℕ := 44
def total_installments : ℕ := num_first_payments + num_remaining_payments

def total_first_payments : ℕ := num_first_payments * first_payment
def remaining_payment : ℕ := first_payment + additional_amount
def total_remaining_payments : ℕ := num_remaining_payments * remaining_payment

def total_payment : ℕ := total_first_payments + total_remaining_payments
def average_payment : ℚ := total_payment / total_installments

theorem avg_payment_correct : average_payment = 465 := by
  sorry

end avg_payment_correct_l875_87540


namespace minimize_quadratic_l875_87512

theorem minimize_quadratic (c : ℝ) : ∃ b : ℝ, (∀ x : ℝ, 3 * x^2 + 2 * x + c ≥ 3 * b^2 + 2 * b + c) ∧ b = -1/3 :=
by
  sorry

end minimize_quadratic_l875_87512


namespace angles_on_y_axis_l875_87575

theorem angles_on_y_axis :
  {θ : ℝ | ∃ k : ℤ, (θ = 2 * k * Real.pi + Real.pi / 2) ∨ (θ = 2 * k * Real.pi + 3 * Real.pi / 2)} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by 
  sorry

end angles_on_y_axis_l875_87575


namespace verify_statements_l875_87558

theorem verify_statements (a b : ℝ) :
  ( (ab < 0 ∧ (a < 0 ∧ b > 0 ∨ a > 0 ∧ b < 0)) → (a / b = -1)) ∧
  ( (a + b < 0 ∧ ab > 0) → (|2 * a + 3 * b| = -(2 * a + 3 * b)) ) ∧
  ( (|a - b| + a - b = 0) → (b > a) = False ) ∧
  ( (|a| > |b|) → ((a + b) * (a - b) < 0) = False ) :=
by
  sorry

end verify_statements_l875_87558


namespace school_stats_l875_87552

-- Defining the conditions
def girls_grade6 := 315
def boys_grade6 := 309
def girls_grade7 := 375
def boys_grade7 := 341
def drama_club_members := 80
def drama_club_boys_percent := 30 / 100

-- Calculate the derived numbers
def students_grade6 := girls_grade6 + boys_grade6
def students_grade7 := girls_grade7 + boys_grade7
def total_students := students_grade6 + students_grade7
def drama_club_boys := drama_club_boys_percent * drama_club_members
def drama_club_girls := drama_club_members - drama_club_boys

-- Theorem
theorem school_stats :
  total_students = 1340 ∧
  drama_club_girls = 56 ∧
  boys_grade6 = 309 ∧
  boys_grade7 = 341 :=
by
  -- We provide the proof steps inline with sorry placeholders.
  -- In practice, these would be filled with appropriate proofs.
  sorry

end school_stats_l875_87552


namespace find_a33_l875_87578

theorem find_a33 : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → a 2 = 6 → (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) → a 33 = 3 :=
by
  intros a h1 h2 h_rec
  sorry

end find_a33_l875_87578


namespace sum_factors_30_less_15_l875_87591

theorem sum_factors_30_less_15 : (1 + 2 + 3 + 5 + 6 + 10) = 27 := by
  sorry

end sum_factors_30_less_15_l875_87591


namespace average_of_combined_samples_l875_87532

theorem average_of_combined_samples 
  (a : Fin 10 → ℝ)
  (b : Fin 10 → ℝ)
  (ave_a : ℝ := (1 / 10) * (Finset.univ.sum (fun i => a i)))
  (ave_b : ℝ := (1 / 10) * (Finset.univ.sum (fun i => b i)))
  (combined_average : ℝ := (1 / 20) * (Finset.univ.sum (fun i => a i) + Finset.univ.sum (fun i => b i))) :
  combined_average = (1 / 2) * (ave_a + ave_b) := 
  by
    sorry

end average_of_combined_samples_l875_87532


namespace remainder_of_55_power_55_plus_55_div_56_l875_87545

theorem remainder_of_55_power_55_plus_55_div_56 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  -- to be filled with the proof
  sorry

end remainder_of_55_power_55_plus_55_div_56_l875_87545


namespace resistor_value_l875_87501

/-- Two resistors with resistance R are connected in series to a DC voltage source U.
    An ideal voltmeter connected in parallel to one resistor shows a reading of 10V.
    The voltmeter is then replaced by an ideal ammeter, which shows a reading of 10A.
    Prove that the resistance R of each resistor is 2Ω. -/
theorem resistor_value (R U U_v I_A : ℝ)
  (hU_v : U_v = 10)
  (hI_A : I_A = 10)
  (hU : U = 2 * U_v)
  (hU_total : U = R * I_A) : R = 2 :=
by
  sorry

end resistor_value_l875_87501


namespace total_songs_l875_87522

variable (H : String) (M : String) (A : String) (T : String)

def num_songs (s : String) : ℕ :=
  if s = H then 9 else
  if s = M then 5 else
  if s = A ∨ s = T then 
    if H ≠ s ∧ M ≠ s then 6 else 7 
  else 0

theorem total_songs 
  (hH : num_songs H = 9)
  (hM : num_songs M = 5)
  (hA : 5 < num_songs A ∧ num_songs A < 9)
  (hT : 5 < num_songs T ∧ num_songs T < 9) :
  (num_songs H + num_songs M + num_songs A + num_songs T) / 3 = 10 :=
sorry

end total_songs_l875_87522


namespace sum_of_reciprocals_of_roots_l875_87548

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 * r2 = 7) (h2 : r1 + r2 = 16) :
  (1 / r1) + (1 / r2) = 16 / 7 :=
by
  sorry

end sum_of_reciprocals_of_roots_l875_87548


namespace sum_of_prime_factors_2310_l875_87506

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors_sum (n : Nat) : Nat :=
  (List.filter Nat.Prime (Nat.factors n)).sum

theorem sum_of_prime_factors_2310 :
  prime_factors_sum 2310 = 28 :=
by
  sorry

end sum_of_prime_factors_2310_l875_87506


namespace smallest_value_c_plus_d_l875_87597

noncomputable def problem1 (c d : ℝ) : Prop :=
c > 0 ∧ d > 0 ∧ (c^2 ≥ 12 * d) ∧ ((3 * d)^2 ≥ 4 * c)

theorem smallest_value_c_plus_d : ∃ c d : ℝ, problem1 c d ∧ c + d = 4 / Real.sqrt 3 + 4 / 9 :=
sorry

end smallest_value_c_plus_d_l875_87597


namespace cheaper_store_price_in_cents_l875_87523

/-- List price of Book Y -/
def list_price : ℝ := 24.95

/-- Discount at Readers' Delight -/
def readers_delight_discount : ℝ := 5

/-- Discount rate at Book Bargains -/
def book_bargains_discount_rate : ℝ := 0.2

/-- Calculate sale price at Readers' Delight -/
def sale_price_readers_delight : ℝ := list_price - readers_delight_discount

/-- Calculate sale price at Book Bargains -/
def sale_price_book_bargains : ℝ := list_price * (1 - book_bargains_discount_rate)

/-- Difference in price between Book Bargains and Readers' Delight in cents -/
theorem cheaper_store_price_in_cents :
  (sale_price_book_bargains - sale_price_readers_delight) * 100 = 1 :=
by
  sorry

end cheaper_store_price_in_cents_l875_87523


namespace total_length_of_fence_l875_87576

theorem total_length_of_fence
  (x : ℝ)
  (h1 : (2 : ℝ) * x ^ 2 = 200) :
  (2 * x + 2 * x) = 40 :=
by
sorry

end total_length_of_fence_l875_87576


namespace child_tickets_sold_l875_87573

theorem child_tickets_sold
  (A C : ℕ)
  (h1 : A + C = 130)
  (h2 : 12 * A + 4 * C = 840) : C = 90 :=
  by {
  -- Proof skipped
  sorry
}

end child_tickets_sold_l875_87573


namespace minimum_value_of_c_l875_87553

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2)

noncomputable def tan_formula (a b c B : ℝ) : Prop :=
  24 * (b * c - a) = b * Real.tan B

noncomputable def min_value_c (a b c : ℝ) : ℝ :=
  (2 * Real.sqrt 3) / 3

theorem minimum_value_of_c (a b c B : ℝ) (h₁ : 0 < B ∧ B < π / 2) (h₂ : 24 * (b * c - a) = b * Real.tan B)
  (h₃ : triangle_area a b c = (1/2) * a * b * Real.sin (π / 6)) :
  c ≥ min_value_c a b c :=
by
  sorry

end minimum_value_of_c_l875_87553


namespace max_value_abs_x_sub_3y_l875_87521

theorem max_value_abs_x_sub_3y 
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + 3 * y ≤ 4)
  (h3 : x ≥ -2) : 
  ∃ z, z = |x - 3 * y| ∧ ∀ (x y : ℝ), (y ≥ x) → (x + 3 * y ≤ 4) → (x ≥ -2) → |x - 3 * y| ≤ 4 :=
sorry

end max_value_abs_x_sub_3y_l875_87521


namespace Tony_total_payment_l875_87584

-- Defining the cost of items
def lego_block_cost : ℝ := 250
def toy_sword_cost : ℝ := 120
def play_dough_cost : ℝ := 35

-- Quantities of each item
def total_lego_blocks : ℕ := 3
def total_toy_swords : ℕ := 5
def total_play_doughs : ℕ := 10

-- Quantities purchased on each day
def first_day_lego_blocks : ℕ := 2
def first_day_toy_swords : ℕ := 3
def second_day_lego_blocks : ℕ := total_lego_blocks - first_day_lego_blocks
def second_day_toy_swords : ℕ := total_toy_swords - first_day_toy_swords
def second_day_play_doughs : ℕ := total_play_doughs

-- Discounts and tax rates
def first_day_discount : ℝ := 0.20
def second_day_discount : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Calculating first day purchase amounts
def first_day_cost_before_discount : ℝ := (first_day_lego_blocks * lego_block_cost) + (first_day_toy_swords * toy_sword_cost)
def first_day_discount_amount : ℝ := first_day_cost_before_discount * first_day_discount
def first_day_cost_after_discount : ℝ := first_day_cost_before_discount - first_day_discount_amount
def first_day_sales_tax_amount : ℝ := first_day_cost_after_discount * sales_tax
def first_day_total_cost : ℝ := first_day_cost_after_discount + first_day_sales_tax_amount

-- Calculating second day purchase amounts
def second_day_cost_before_discount : ℝ := (second_day_lego_blocks * lego_block_cost) + (second_day_toy_swords * toy_sword_cost) + 
                                           (second_day_play_doughs * play_dough_cost)
def second_day_discount_amount : ℝ := second_day_cost_before_discount * second_day_discount
def second_day_cost_after_discount : ℝ := second_day_cost_before_discount - second_day_discount_amount
def second_day_sales_tax_amount : ℝ := second_day_cost_after_discount * sales_tax
def second_day_total_cost : ℝ := second_day_cost_after_discount + second_day_sales_tax_amount

-- Total cost
def total_cost : ℝ := first_day_total_cost + second_day_total_cost

-- Lean theorem statement
theorem Tony_total_payment : total_cost = 1516.20 := by
  sorry

end Tony_total_payment_l875_87584


namespace roses_in_february_l875_87596

-- Define initial counts of roses
def roses_oct : ℕ := 80
def roses_nov : ℕ := 98
def roses_dec : ℕ := 128
def roses_jan : ℕ := 170

-- Define the differences
def diff_on : ℕ := roses_nov - roses_oct -- 18
def diff_nd : ℕ := roses_dec - roses_nov -- 30
def diff_dj : ℕ := roses_jan - roses_dec -- 42

-- The increment in differences
def inc : ℕ := diff_nd - diff_on -- 12

-- Express the difference from January to February
def diff_jf : ℕ := diff_dj + inc -- 54

-- The number of roses in February
def roses_feb : ℕ := roses_jan + diff_jf -- 224

theorem roses_in_february : roses_feb = 224 := by
  -- Provide the expected value for Lean to verify
  sorry

end roses_in_february_l875_87596


namespace cubical_cake_l875_87569

noncomputable def cubical_cake_properties : Prop :=
  let a : ℝ := 3
  let top_area := (1 / 2) * 3 * 1.5
  let height := 3
  let volume := top_area * height
  let vertical_triangles_area := 2 * ((1 / 2) * 1.5 * 3)
  let vertical_rectangular_area := 3 * 3
  let iced_area := top_area + vertical_triangles_area + vertical_rectangular_area
  volume + iced_area = 22.5

theorem cubical_cake : cubical_cake_properties := sorry

end cubical_cake_l875_87569


namespace cryptarithm_C_value_l875_87581

/--
Given digits A, B, and C where A, B, and C are distinct and non-repeating,
and the following conditions hold:
1. ABC - BC = A0A
Prove that C = 9.
-/
theorem cryptarithm_C_value (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_non_repeating: (0 <= A ∧ A <= 9) ∧ (0 <= B ∧ B <= 9) ∧ (0 <= C ∧ C <= 9))
  (h_subtraction : 100 * A + 10 * B + C - (10 * B + C) = 100 * A + 0 + A) :
  C = 9 := sorry

end cryptarithm_C_value_l875_87581


namespace arithmetic_sequence_sum_l875_87561

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
def problem_conditions (a : ℕ → ℝ) : Prop :=
  (a 3 + a 8 = 3) ∧ is_arithmetic_sequence a

-- State the theorem to be proved
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : problem_conditions a) : a 1 + a 10 = 3 :=
sorry

end arithmetic_sequence_sum_l875_87561


namespace evaluate_f_at_neg3_l875_87555

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem evaluate_f_at_neg3 : f (-3) = 110 :=
by 
  sorry

end evaluate_f_at_neg3_l875_87555


namespace medicine_duration_l875_87516

theorem medicine_duration (days_per_third_pill : ℕ) (pills : ℕ) (days_per_month : ℕ)
  (h1 : days_per_third_pill = 3)
  (h2 : pills = 90)
  (h3 : days_per_month = 30) :
  ((pills * (days_per_third_pill * 3)) / days_per_month) = 27 :=
sorry

end medicine_duration_l875_87516


namespace part_a_l875_87595

theorem part_a : 
  ∃ (x y : ℕ → ℕ), (∀ n : ℕ, (1 + Real.sqrt 33) ^ n = x n + y n * Real.sqrt 33) :=
sorry

end part_a_l875_87595


namespace good_or_bad_of_prime_divides_l875_87565

-- Define the conditions
variables (k n n' : ℕ)
variables (h1 : k ≥ 2) (h2 : n ≥ k) (h3 : n' ≥ k)
variables (prime_divides : ∀ p, prime p → p ≤ k → (p ∣ n ↔ p ∣ n'))

-- Define what it means for a number to be good or bad
def is_good (m : ℕ) : Prop := ∃ strategy : ℕ → Prop, strategy m

-- Prove that either both n and n' are good or both are bad
theorem good_or_bad_of_prime_divides :
  (is_good n ∧ is_good n') ∨ (¬is_good n ∧ ¬is_good n') :=
sorry

end good_or_bad_of_prime_divides_l875_87565


namespace packages_ratio_l875_87525

theorem packages_ratio (packages_yesterday packages_today : ℕ)
  (h1 : packages_yesterday = 80)
  (h2 : packages_today + packages_yesterday = 240) :
  (packages_today / packages_yesterday) = 2 :=
by
  sorry

end packages_ratio_l875_87525


namespace smallest_coin_remainder_l875_87542

theorem smallest_coin_remainder
  (c : ℕ)
  (h1 : c % 8 = 6)
  (h2 : c % 7 = 5)
  (h3 : ∀ d : ℕ, (d % 8 = 6) → (d % 7 = 5) → d ≥ c) :
  c % 9 = 2 :=
sorry

end smallest_coin_remainder_l875_87542


namespace length_second_platform_l875_87583

-- Define the conditions
def length_train : ℕ := 100
def time_platform1 : ℕ := 15
def length_platform1 : ℕ := 350
def time_platform2 : ℕ := 20

-- Prove the length of the second platform is 500m
theorem length_second_platform : ∀ (speed_train : ℚ), 
  speed_train = (length_train + length_platform1) / time_platform1 →
  (speed_train = (length_train + L) / time_platform2) → 
  L = 500 :=
by 
  intro speed_train h1 h2
  sorry

end length_second_platform_l875_87583


namespace number_of_terms_arithmetic_sequence_l875_87504

theorem number_of_terms_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -36 → d = 6 → l = 66 → ∃ n, l = a + (n-1) * d ∧ n = 18 :=
by
  intros a d l ha hd hl
  exists 18
  rw [ha, hd, hl]
  sorry

end number_of_terms_arithmetic_sequence_l875_87504


namespace determine_y_l875_87594

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (hxy : x = 2 + (1 / y))
variable (hyx : y = 2 + (2 / x))

theorem determine_y (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 2 + (1 / y)) (hyx : y = 2 + (2 / x)) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 := 
sorry

end determine_y_l875_87594


namespace option_A_sufficient_not_necessary_l875_87574

variable (a b : ℝ)

def A : Set ℝ := { x | x^2 - x + a ≤ 0 }
def B : Set ℝ := { x | x^2 - x + b ≤ 0 }

theorem option_A_sufficient_not_necessary : (A = B → a = b) ∧ (a = b → A = B) :=
by
  sorry

end option_A_sufficient_not_necessary_l875_87574


namespace center_of_circle_sum_l875_87572

open Real

theorem center_of_circle_sum (x y : ℝ) (h k : ℝ) :
  (x - h)^2 + (y - k)^2 = 2 → (h = 3) → (k = 4) → h + k = 7 :=
by
  intro h_eq k_eq
  sorry

end center_of_circle_sum_l875_87572


namespace total_students_is_2000_l875_87538

theorem total_students_is_2000
  (S : ℝ) 
  (h1 : 0.10 * S = chess_students) 
  (h2 : 0.50 * chess_students = swimming_students) 
  (h3 : swimming_students = 100) 
  (chess_students swimming_students : ℝ) 
  : S = 2000 := 
by 
  sorry

end total_students_is_2000_l875_87538


namespace positive_difference_of_two_numbers_l875_87502

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l875_87502


namespace system_of_equations_correct_l875_87590

theorem system_of_equations_correct (x y : ℤ) :
  (8 * x - 3 = y) ∧ (7 * x + 4 = y) :=
sorry

end system_of_equations_correct_l875_87590


namespace set_intersection_complement_l875_87508

-- Define the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ ¬ x ∈ A}

-- Define the intersection of B and complement_U_A
def B_inter_complement_U_A : Set ℕ := B ∩ complement_U_A

-- The statement to prove: B ∩ complement_U_A = {6, 7}
theorem set_intersection_complement :
  B_inter_complement_U_A = {6, 7} := by sorry

end set_intersection_complement_l875_87508


namespace average_wage_per_day_l875_87507

theorem average_wage_per_day :
  let num_male := 20
  let num_female := 15
  let num_child := 5
  let wage_male := 35
  let wage_female := 20
  let wage_child := 8
  let total_wages := (num_male * wage_male) + (num_female * wage_female) + (num_child * wage_child)
  let total_workers := num_male + num_female + num_child
  total_wages / total_workers = 26 := by
  sorry

end average_wage_per_day_l875_87507


namespace stratified_sampling_second_class_l875_87528

theorem stratified_sampling_second_class (total_products : ℕ) (first_class : ℕ) (second_class : ℕ) (third_class : ℕ) (sample_size : ℕ) (h_total : total_products = 200) (h_first : first_class = 40) (h_second : second_class = 60) (h_third : third_class = 100) (h_sample : sample_size = 40) :
  (second_class * sample_size) / total_products = 12 :=
by
  sorry

end stratified_sampling_second_class_l875_87528


namespace range_of_a_l875_87577

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.exp x - 1) - Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, 0 < x0 ∧ f (g x0) a > f x0 a) ↔ 1 < a := sorry

end range_of_a_l875_87577


namespace range_of_half_alpha_minus_beta_l875_87514

theorem range_of_half_alpha_minus_beta (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < (1/2) * α - β ∧ (1/2) * α - β < 11/2 :=
by
  -- sorry to skip the proof
  sorry

end range_of_half_alpha_minus_beta_l875_87514


namespace height_percentage_increase_l875_87500

theorem height_percentage_increase (B A : ℝ) (h : A = B - 0.3 * B) : 
  ((B - A) / A) * 100 = 42.857 :=
by
  sorry

end height_percentage_increase_l875_87500


namespace basil_pots_count_l875_87534

theorem basil_pots_count (B : ℕ) (h1 : 9 * 18 + 6 * 30 + 4 * B = 354) : B = 3 := 
by 
  -- This is just the signature of the theorem. The proof is omitted.
  sorry

end basil_pots_count_l875_87534


namespace greatest_partition_l875_87543

-- Define the condition on the partitions of the positive integers
def satisfies_condition (A : ℕ → Prop) (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ b ∧ A a ∧ A b ∧ a + b = n

-- Define what it means for k subsets to meet the requirements
def partition_satisfies (k : ℕ) : Prop :=
∃ A : ℕ → ℕ → Prop,
  (∀ i : ℕ, i < k → ∀ n ≥ 15, satisfies_condition (A i) n)

-- Our conjecture is that k can be at most 3 for the given condition
theorem greatest_partition (k : ℕ) : k ≤ 3 :=
sorry

end greatest_partition_l875_87543


namespace three_distinct_real_solutions_l875_87557

theorem three_distinct_real_solutions (b c : ℝ):
  (∀ x : ℝ, x^2 + b * |x| + c = 0 → x = 0) ∧ (∃! x : ℝ, x^2 + b * |x| + c = 0) →
  b < 0 ∧ c = 0 :=
by {
  sorry
}

end three_distinct_real_solutions_l875_87557


namespace lesser_fraction_of_sum_and_product_l875_87570

open Real

theorem lesser_fraction_of_sum_and_product (a b : ℚ)
  (h1 : a + b = 11 / 12)
  (h2 : a * b = 1 / 6) :
  min a b = 1 / 4 :=
sorry

end lesser_fraction_of_sum_and_product_l875_87570


namespace quadratic_conditions_l875_87510

open Polynomial

noncomputable def exampleQuadratic (x : ℝ) : ℝ :=
-2 * x^2 + 12 * x - 10

theorem quadratic_conditions :
  (exampleQuadratic 1 = 0) ∧ (exampleQuadratic 5 = 0) ∧ (exampleQuadratic 3 = 8) :=
by
  sorry

end quadratic_conditions_l875_87510


namespace compute_f_g_at_2_l875_87580

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4 * x - 1

theorem compute_f_g_at_2 :
  f (g 2) = 49 :=
by
  sorry

end compute_f_g_at_2_l875_87580


namespace lisa_eats_one_candy_on_other_days_l875_87567

def candies_total : ℕ := 36
def candies_per_day_on_mondays_and_wednesdays : ℕ := 2
def weeks : ℕ := 4
def days_in_a_week : ℕ := 7
def mondays_and_wednesdays_in_4_weeks : ℕ := 2 * weeks
def total_candies_mondays_and_wednesdays : ℕ := mondays_and_wednesdays_in_4_weeks * candies_per_day_on_mondays_and_wednesdays
def total_other_candies : ℕ := candies_total - total_candies_mondays_and_wednesdays
def total_other_days : ℕ := weeks * (days_in_a_week - 2)
def candies_per_other_day : ℕ := total_other_candies / total_other_days

theorem lisa_eats_one_candy_on_other_days :
  candies_per_other_day = 1 :=
by
  -- Prove the theorem with conditions defined
  sorry

end lisa_eats_one_candy_on_other_days_l875_87567


namespace domain_of_f_eq_R_l875_87535

noncomputable def f (x m : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

theorem domain_of_f_eq_R (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x + 3 ≠ 0) ↔ (0 ≤ m ∧ m < 3 / 4) :=
by
  sorry

end domain_of_f_eq_R_l875_87535


namespace ratio_consequent_l875_87563

theorem ratio_consequent (a b x : ℕ) (h_ratio : a = 4) (h_b : b = 6) (h_x : x = 30) :
  (a : ℚ) / b = x / 45 := 
by 
  -- add here the necessary proof steps 
  sorry

end ratio_consequent_l875_87563


namespace difference_max_min_planes_l875_87592

open Set

-- Defining the regular tetrahedron and related concepts
noncomputable def tetrahedron := Unit -- Placeholder for the tetrahedron

def union_faces (T : Unit) : Set Point := sorry -- Placeholder for union of faces definition

noncomputable def simple_trace (p : Plane) (T : Unit) : Set Point := sorry -- Placeholder for planes intersecting faces

-- Calculating number of planes
def maximum_planes (T : Unit) : Nat :=
  4 -- One for each face of the tetrahedron

def minimum_planes (T : Unit) : Nat :=
  2 -- Each plane covers traces on two adjacent faces if oriented appropriately

-- Statement of the problem
theorem difference_max_min_planes (T : Unit) :
  maximum_planes T - minimum_planes T = 2 :=
by
  -- Proof skipped
  sorry

end difference_max_min_planes_l875_87592


namespace inequality_solution_l875_87564

theorem inequality_solution :
  {x : ℝ // -1 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 1} = 
  {x : ℝ // x > 1/6} :=
sorry

end inequality_solution_l875_87564


namespace reduced_price_is_correct_l875_87527

-- Definitions for the conditions in the problem
def original_price_per_dozen (P : ℝ) : Prop :=
∀ (X : ℝ), X * P = 40.00001

def reduced_price_per_dozen (P R : ℝ) : Prop :=
R = 0.60 * P

def bananas_purchased_additional (P R : ℝ) : Prop :=
∀ (X Y : ℝ), (Y = X + (64 / 12)) → (X * P = Y * R) 

-- Assertion of the proof problem
theorem reduced_price_is_correct : 
  ∃ (R : ℝ), 
  (∀ P, original_price_per_dozen P ∧ reduced_price_per_dozen P R ∧ bananas_purchased_additional P R) → 
  R = 3.00000075 := 
by sorry

end reduced_price_is_correct_l875_87527


namespace donuts_selection_l875_87551

def number_of_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem donuts_selection : number_of_selections 6 4 = 84 := by
  sorry

end donuts_selection_l875_87551


namespace proof_x1_x2_squared_l875_87546

theorem proof_x1_x2_squared (x1 x2 : ℝ) (h1 : (Real.exp 1 * x1)^x2 = (Real.exp 1 * x2)^x1)
  (h2 : 0 < x1) (h3 : 0 < x2) (h4 : x1 ≠ x2) : x1^2 + x2^2 > 2 :=
sorry

end proof_x1_x2_squared_l875_87546


namespace sum_of_h_values_l875_87537

variable (f h : ℤ → ℤ)

-- Function definition for f and h
def f_def : ∀ x, 0 ≤ x → f x = f (x + 2) := sorry
def h_def : ∀ x, x < 0 → h x = f x := sorry

-- Symmetry condition for f being odd
def f_odd : ∀ x, f (-x) = -f x := sorry

-- Given value
def f_at_5 : f 5 = 1 := sorry

-- The proof statement we need:
theorem sum_of_h_values :
  h (-2022) + h (-2023) + h (-2024) = -1 :=
sorry

end sum_of_h_values_l875_87537


namespace value_of_a3_a5_l875_87566

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem value_of_a3_a5 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 :=
  sorry

end value_of_a3_a5_l875_87566


namespace calculator_sum_l875_87587

theorem calculator_sum :
  let A := 2
  let B := 0
  let C := -1
  let D := 3
  let n := 47
  let A' := if n % 2 = 1 then -A else A
  let B' := B -- B remains 0 after any number of sqrt operations
  let C' := if n % 2 = 1 then -C else C
  let D' := D ^ (3 ^ n)
  A' + B' + C' + D' = 3 ^ (3 ^ 47) - 3
:= by
  sorry

end calculator_sum_l875_87587


namespace find_radioactive_balls_within_7_checks_l875_87585

theorem find_radioactive_balls_within_7_checks :
  ∃ (balls : Finset α), balls.card = 11 ∧ ∃ radioactive_balls ⊆ balls, radioactive_balls.card = 2 ∧
  (∀ (check : Finset α → Prop), (∀ S, check S ↔ (∃ b ∈ S, b ∈ radioactive_balls)) →
  ∃ checks : Finset (Finset α), checks.card ≤ 7 ∧ (∀ b ∈ radioactive_balls, ∃ S ∈ checks, b ∈ S)) :=
sorry

end find_radioactive_balls_within_7_checks_l875_87585


namespace mean_score_of_seniors_l875_87526

variable (s n : ℕ)  -- Number of seniors and non-seniors
variable (m_s m_n : ℝ)  -- Mean scores of seniors and non-seniors
variable (total_mean : ℝ) -- Mean score of all students
variable (total_students : ℕ) -- Total number of students

theorem mean_score_of_seniors :
  total_students = 100 → total_mean = 100 →
  n = 3 * s / 2 →
  s * m_s + n * m_n = total_students * total_mean →
  m_s = (3 * m_n / 2) →
  m_s = 125 :=
by
  intros
  sorry

end mean_score_of_seniors_l875_87526


namespace max_red_balls_l875_87568

theorem max_red_balls (r w : ℕ) (h1 : r = 3 * w) (h2 : r + w ≤ 50) : r = 36 :=
sorry

end max_red_balls_l875_87568


namespace tracy_initial_candies_l875_87571

variable (x : ℕ)
variable (b : ℕ)

theorem tracy_initial_candies : 
  (x % 6 = 0) ∧
  (34 ≤ (1 / 2 * x)) ∧
  ((1 / 2 * x) ≤ 38) ∧
  (1 ≤ b) ∧
  (b ≤ 5) ∧
  (1 / 2 * x - 30 - b = 3) →
  x = 72 := 
sorry

end tracy_initial_candies_l875_87571


namespace pears_for_36_bananas_l875_87518

theorem pears_for_36_bananas (p : ℕ) (bananas : ℕ) (pears : ℕ) (h : 9 * pears = 6 * bananas) :
  36 * pears = 9 * 24 :=
by
  sorry

end pears_for_36_bananas_l875_87518


namespace min_weight_of_lightest_l875_87588

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l875_87588


namespace bus_return_trip_fraction_l875_87515

theorem bus_return_trip_fraction :
  (3 / 4 * 200 + x * 200 = 310) → (x = 4 / 5) := by
  sorry

end bus_return_trip_fraction_l875_87515


namespace smallest_n_for_inequality_l875_87579

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l875_87579


namespace factor_expression_l875_87547

theorem factor_expression (x : ℝ) :
  80 * x ^ 5 - 250 * x ^ 9 = -10 * x ^ 5 * (25 * x ^ 4 - 8) :=
by
  sorry

end factor_expression_l875_87547


namespace speed_is_90_l875_87519

namespace DrivingSpeedProof

/-- Given the observation times and marker numbers, prove the speed of the car is 90 km/hr. -/
theorem speed_is_90 
  (X Y : ℕ)
  (h0 : X ≥ 0) (h1 : X ≤ 9)
  (h2 : Y = 8 * X)
  (h3 : Y ≥ 0) (h4 : Y ≤ 9)
  (noon_marker : 10 * X + Y = 18)
  (second_marker : 10 * Y + X = 81)
  (third_marker : 100 * X + Y = 108)
  : 90 = 90 :=
by {
  sorry
}

end DrivingSpeedProof

end speed_is_90_l875_87519


namespace triangle_angle_not_less_than_60_l875_87554

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l875_87554


namespace true_statements_about_f_l875_87593

noncomputable def f (x : ℝ) := 2 * abs (Real.cos x) * Real.sin x + Real.sin (2 * x)

theorem true_statements_about_f :
  (∀ x y : ℝ, -π/4 ≤ x ∧ x < y ∧ y ≤ π/4 → f x < f y) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → (∃ x : ℝ, f x = y)) :=
by
  sorry

end true_statements_about_f_l875_87593


namespace keep_oranges_per_day_l875_87549

def total_oranges_harvested (sacks_per_day : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  sacks_per_day * oranges_per_sack

def oranges_discarded (discarded_sacks : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  discarded_sacks * oranges_per_sack

def oranges_kept_per_day (total_oranges : ℕ) (discarded_oranges : ℕ) : ℕ :=
  total_oranges - discarded_oranges

theorem keep_oranges_per_day 
  (sacks_per_day : ℕ)
  (oranges_per_sack : ℕ)
  (discarded_sacks : ℕ)
  (h1 : sacks_per_day = 76)
  (h2 : oranges_per_sack = 50)
  (h3 : discarded_sacks = 64) :
  oranges_kept_per_day (total_oranges_harvested sacks_per_day oranges_per_sack) 
  (oranges_discarded discarded_sacks oranges_per_sack) = 600 :=
by
  sorry

end keep_oranges_per_day_l875_87549


namespace socks_selection_l875_87586

theorem socks_selection :
  let red_socks := 120
  let green_socks := 90
  let blue_socks := 70
  let black_socks := 50
  let yellow_socks := 30
  let total_socks :=  red_socks + green_socks + blue_socks + black_socks + yellow_socks 
  (∀ k : ℕ, k ≥ 1 → k ≤ total_socks → (∃ p : ℕ, p = 12 → (p ≥ k / 2)) → k = 28) :=
by
  sorry

end socks_selection_l875_87586


namespace same_terminal_side_angle_in_range_0_to_2pi_l875_87533

theorem same_terminal_side_angle_in_range_0_to_2pi :
  ∃ k : ℤ, 0 ≤ 2 * k * π + (-4) * π / 3 ∧ 2 * k * π + (-4) * π / 3 ≤ 2 * π ∧
  2 * k * π + (-4) * π / 3 = 2 * π / 3 :=
by
  use 1
  sorry

end same_terminal_side_angle_in_range_0_to_2pi_l875_87533


namespace min_max_value_in_interval_l875_87560

theorem min_max_value_in_interval : ∀ (x : ℝ),
  -2 < x ∧ x < 5 →
  ∃ (y : ℝ), (y = -1.5 ∨ y = 1.5) ∧ y = (x^2 - 4 * x + 6) / (2 * x - 4) := 
by sorry

end min_max_value_in_interval_l875_87560


namespace An_nonempty_finite_l875_87513

def An (n : ℕ) : Set (ℕ × ℕ) :=
  { p : ℕ × ℕ | ∃ (k : ℕ), ∃ (a : ℕ), ∃ (b : ℕ), a = Nat.sqrt (p.1^2 + p.2 + n) ∧ b = Nat.sqrt (p.2^2 + p.1 + n) ∧ k = a + b }

theorem An_nonempty_finite (n : ℕ) (h : n ≥ 1) : Set.Nonempty (An n) ∧ Set.Finite (An n) :=
by
  sorry -- The proof goes here

end An_nonempty_finite_l875_87513


namespace history_only_students_l875_87550

theorem history_only_students 
  (total_students : ℕ)
  (history_students stats_students physics_students chem_students : ℕ) 
  (hist_stats hist_phys hist_chem stats_phys stats_chem phys_chem all_four : ℕ) 
  (h1 : total_students = 500)
  (h2 : history_students = 150)
  (h3 : stats_students = 130)
  (h4 : physics_students = 120)
  (h5 : chem_students = 100)
  (h6 : hist_stats = 60)
  (h7 : hist_phys = 50)
  (h8 : hist_chem = 40)
  (h9 : stats_phys = 35)
  (h10 : stats_chem = 30)
  (h11 : phys_chem = 25)
  (h12 : all_four = 20) : 
  (history_students - hist_stats - hist_phys - hist_chem + all_four) = 20 := 
by 
  sorry

end history_only_students_l875_87550


namespace gas_tank_size_l875_87503

-- Conditions from part a)
def advertised_mileage : ℕ := 35
def actual_mileage : ℕ := 31
def total_miles_driven : ℕ := 372

-- Question and the correct answer in the context of conditions
theorem gas_tank_size (h1 : actual_mileage = advertised_mileage - 4) 
                      (h2 : total_miles_driven = 372) 
                      : total_miles_driven / actual_mileage = 12 := 
by sorry

end gas_tank_size_l875_87503
