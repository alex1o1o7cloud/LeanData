import Mathlib

namespace NUMINAMATH_GPT_morgan_first_sat_score_l1055_105540

theorem morgan_first_sat_score (x : ℝ) (h : 1.10 * x = 1100) : x = 1000 :=
sorry

end NUMINAMATH_GPT_morgan_first_sat_score_l1055_105540


namespace NUMINAMATH_GPT_probability_at_least_one_expired_l1055_105563

theorem probability_at_least_one_expired (total_bottles expired_bottles selected_bottles : ℕ) : 
  total_bottles = 10 → expired_bottles = 3 → selected_bottles = 3 → 
  (∃ probability, probability = 17 / 24) :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_expired_l1055_105563


namespace NUMINAMATH_GPT_find_xy_integers_l1055_105595

theorem find_xy_integers (x y : ℤ) (h : x^3 + 2 * x * y = 7) :
  (x, y) = (-7, -25) ∨ (x, y) = (-1, -4) ∨ (x, y) = (1, 3) ∨ (x, y) = (7, -24) :=
sorry

end NUMINAMATH_GPT_find_xy_integers_l1055_105595


namespace NUMINAMATH_GPT_points_lie_on_circle_l1055_105530

theorem points_lie_on_circle (s : ℝ) :
  ( (2 - s^2) / (2 + s^2) )^2 + ( 3 * s / (2 + s^2) )^2 = 1 :=
by sorry

end NUMINAMATH_GPT_points_lie_on_circle_l1055_105530


namespace NUMINAMATH_GPT_find_common_real_root_l1055_105583

theorem find_common_real_root :
  ∃ (m a : ℝ), (a^2 + m * a + 2 = 0) ∧ (a^2 + 2 * a + m = 0) ∧ m = -3 ∧ a = 1 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_find_common_real_root_l1055_105583


namespace NUMINAMATH_GPT_three_digit_ends_in_5_divisible_by_5_l1055_105557

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_5 (n : ℕ) : Prop := n % 10 = 5

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_ends_in_5_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : ends_in_5 N) : is_divisible_by_5 N := 
sorry

end NUMINAMATH_GPT_three_digit_ends_in_5_divisible_by_5_l1055_105557


namespace NUMINAMATH_GPT_eq_solutions_a2_eq_b_times_b_plus_7_l1055_105516

theorem eq_solutions_a2_eq_b_times_b_plus_7 (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h : a^2 = b * (b + 7)) :
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end NUMINAMATH_GPT_eq_solutions_a2_eq_b_times_b_plus_7_l1055_105516


namespace NUMINAMATH_GPT_probability_of_same_color_correct_l1055_105594

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_ways_to_choose_two : ℕ := choose total_plates 2
noncomputable def ways_to_choose_two_red : ℕ := choose red_plates 2
noncomputable def ways_to_choose_two_blue : ℕ := choose blue_plates 2

noncomputable def ways_to_choose_two_same_color : ℕ :=
  ways_to_choose_two_red + ways_to_choose_two_blue

noncomputable def probability_same_color : ℚ :=
  ways_to_choose_two_same_color / total_ways_to_choose_two

theorem probability_of_same_color_correct :
  probability_same_color = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_correct_l1055_105594


namespace NUMINAMATH_GPT_product_of_last_two_digits_l1055_105544

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 8 = 0) : A * B = 32 :=
by
  sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l1055_105544


namespace NUMINAMATH_GPT_percentage_increase_on_friday_l1055_105511

theorem percentage_increase_on_friday (avg_books_per_day : ℕ) (friday_books : ℕ) (total_books_per_week : ℕ) (days_open : ℕ)
  (h1 : avg_books_per_day = 40)
  (h2 : total_books_per_week = 216)
  (h3 : days_open = 5)
  (h4 : friday_books > avg_books_per_day) :
  (((friday_books - avg_books_per_day) * 100) / avg_books_per_day) = 40 :=
sorry

end NUMINAMATH_GPT_percentage_increase_on_friday_l1055_105511


namespace NUMINAMATH_GPT_jill_llamas_count_l1055_105550

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end NUMINAMATH_GPT_jill_llamas_count_l1055_105550


namespace NUMINAMATH_GPT_find_m_l1055_105536

open Nat

def is_arithmetic (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i < n - 1, a (i + 2) - a (i + 1) = a (i + 1) - a i
def is_geometric (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i ≥ n, a (i + 1) * a n = a i * a (n + 1)
def sum_prod_condition (a : ℕ → ℤ) (m : ℕ) : Prop := a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

theorem find_m (a : ℕ → ℤ)
  (h1 : a 3 = -1)
  (h2 : a 7 = 4)
  (h3 : is_arithmetic a 6)
  (h4 : is_geometric a 5) :
  ∃ m : ℕ, m = 1 ∨ m = 3 ∧ sum_prod_condition a m := sorry

end NUMINAMATH_GPT_find_m_l1055_105536


namespace NUMINAMATH_GPT_original_area_of_circle_l1055_105592

theorem original_area_of_circle
  (A₀ : ℝ) -- original area
  (r₀ r₁ : ℝ) -- original and new radius
  (π : ℝ := 3.14)
  (h_area : A₀ = π * r₀^2)
  (h_area_increase : π * r₁^2 = 9 * A₀)
  (h_circumference_increase : 2 * π * r₁ - 2 * π * r₀ = 50.24) :
  A₀ = 50.24 :=
by
  sorry

end NUMINAMATH_GPT_original_area_of_circle_l1055_105592


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_equality_l1055_105500

variables {a_n : ℕ → ℝ} -- the arithmetic sequence
variables (S_n : ℕ → ℝ) -- the sum of the first n terms of the sequence

-- Define the conditions as hypotheses
def condition_1 (S_n : ℕ → ℝ) : Prop := S_n 3 = 3
def condition_2 (S_n : ℕ → ℝ) : Prop := S_n 6 = 15

-- Theorem statement
theorem arithmetic_sequence_sum_equality
  (h1 : condition_1 S_n)
  (h2 : condition_2 S_n)
  (a_n_formula : ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0))
  (S_n_formula : ∀ n, S_n n = n * (a_n 0 + (n - 1) * (a_n 1 - a_n 0) / 2)) :
  a_n 10 + a_n 11 + a_n 12 = 30 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_equality_l1055_105500


namespace NUMINAMATH_GPT_net_sales_revenue_l1055_105552

-- Definition of the conditions
def regression (x : ℝ) : ℝ := 8.5 * x + 17.5

-- Statement of the theorem
theorem net_sales_revenue (x : ℝ) (h : x = 10) : (regression x - x) = 92.5 :=
by {
  -- No proof required as per instruction; use sorry.
  sorry
}

end NUMINAMATH_GPT_net_sales_revenue_l1055_105552


namespace NUMINAMATH_GPT_area_of_ABCD_proof_l1055_105505

noncomputable def point := ℝ × ℝ

structure Rectangle :=
  (A B C D : point)
  (angle_C_trisected_by_CE_CF : Prop)
  (E_on_AB : Prop)
  (F_on_AD : Prop)
  (AF : ℝ)
  (BE : ℝ)

def area_of_rectangle (rect : Rectangle) : ℝ :=
  let (x1, y1) := rect.A
  let (x2, y2) := rect.C
  (x2 - x1) * (y2 - y1)

theorem area_of_ABCD_proof :
  ∀ (ABCD : Rectangle),
    ABCD.angle_C_trisected_by_CE_CF →
    ABCD.E_on_AB →
    ABCD.F_on_AD →
    ABCD.AF = 2 →
    ABCD.BE = 6 →
    abs (area_of_rectangle ABCD - 150) < 1 :=
by
  sorry

end NUMINAMATH_GPT_area_of_ABCD_proof_l1055_105505


namespace NUMINAMATH_GPT_largest_number_among_options_l1055_105555

theorem largest_number_among_options :
  let A := 0.983
  let B := 0.9829
  let C := 0.9831
  let D := 0.972
  let E := 0.9819
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end NUMINAMATH_GPT_largest_number_among_options_l1055_105555


namespace NUMINAMATH_GPT_ordering_of_four_numbers_l1055_105566

variable (m n α β : ℝ)
variable (h1 : m < n)
variable (h2 : α < β)
variable (h3 : 2 * (α - m) * (α - n) - 7 = 0)
variable (h4 : 2 * (β - m) * (β - n) - 7 = 0)

theorem ordering_of_four_numbers : α < m ∧ m < n ∧ n < β :=
by
  sorry

end NUMINAMATH_GPT_ordering_of_four_numbers_l1055_105566


namespace NUMINAMATH_GPT_product_correct_l1055_105535

/-- Define the number and the digit we're interested in -/
def num : ℕ := 564823
def digit : ℕ := 4

/-- Define a function to calculate the local value of the digit 4 in the number 564823 -/
def local_value (n : ℕ) (d : ℕ) := if d = 4 then 40000 else 0

/-- Define a function to calculate the absolute value, although it is trivial for natural numbers -/
def abs_value (d : ℕ) := d

/-- Define the product of local value and absolute value of 4 in 564823 -/
def product := local_value num digit * abs_value digit

/-- Theorem stating that the product is as specified in the problem -/
theorem product_correct : product = 160000 :=
by
  sorry

end NUMINAMATH_GPT_product_correct_l1055_105535


namespace NUMINAMATH_GPT_find_x_plus_y_l1055_105585

theorem find_x_plus_y
  (x y : ℤ)
  (hx : |x| = 2)
  (hy : |y| = 3)
  (hxy : x > y) : x + y = -1 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1055_105585


namespace NUMINAMATH_GPT_quadratic_equation_iff_non_zero_coefficient_l1055_105537

theorem quadratic_equation_iff_non_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_iff_non_zero_coefficient_l1055_105537


namespace NUMINAMATH_GPT_coby_travel_time_l1055_105542

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end NUMINAMATH_GPT_coby_travel_time_l1055_105542


namespace NUMINAMATH_GPT_intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l1055_105513

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

-- Define the conditions
variables (m : ℝ)
theorem intersection_points_of_quadratic :
    (quadratic m 1 = 0) ∧ (quadratic m 3 = 0) ↔ m ≠ 0 :=
sorry

theorem minimum_value_of_quadratic_in_range :
    ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → quadratic (-2) x ≥ -6 :=
sorry

theorem range_of_m_for_intersection_with_segment_PQ :
    ∀ (m : ℝ), (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic m x = (m + 4) / 2) ↔ 
    m ≤ -4 / 3 ∨ m ≥ 4 / 5 :=
sorry

end NUMINAMATH_GPT_intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l1055_105513


namespace NUMINAMATH_GPT_mona_drives_125_miles_l1055_105520

/-- Mona can drive 125 miles with $25 worth of gas, given the car mileage
    and the cost per gallon of gas. -/
theorem mona_drives_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ)
  (h_miles_per_gallon : miles_per_gallon = 25) (h_cost_per_gallon : cost_per_gallon = 5)
  (h_total_money : total_money = 25) :
  (total_money / cost_per_gallon) * miles_per_gallon = 125 :=
by
  sorry

end NUMINAMATH_GPT_mona_drives_125_miles_l1055_105520


namespace NUMINAMATH_GPT_log_6_15_expression_l1055_105521

theorem log_6_15_expression (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 15 / Real.log 6 = (b + 1 - a) / (a + b) :=
sorry

end NUMINAMATH_GPT_log_6_15_expression_l1055_105521


namespace NUMINAMATH_GPT_pow_sum_nineteen_eq_zero_l1055_105581

variable {a b c : ℝ}

theorem pow_sum_nineteen_eq_zero (h₁ : a + b + c = 0) (h₂ : a^3 + b^3 + c^3 = 0) : a^19 + b^19 + c^19 = 0 :=
sorry

end NUMINAMATH_GPT_pow_sum_nineteen_eq_zero_l1055_105581


namespace NUMINAMATH_GPT_equation_solution_l1055_105528

theorem equation_solution (t : ℤ) : 
  ∃ y : ℤ, (21 * t + 2)^3 + 2 * (21 * t + 2)^2 + 5 = 21 * y :=
sorry

end NUMINAMATH_GPT_equation_solution_l1055_105528


namespace NUMINAMATH_GPT_evaluate_expression_l1055_105541

noncomputable def complex_numbers_condition (a b : ℂ) := a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + a * b + b^2 = 0)

theorem evaluate_expression (a b : ℂ) (h : complex_numbers_condition a b) : 
  (a^5 + b^5) / (a + b)^5 = -2 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1055_105541


namespace NUMINAMATH_GPT_find_integer_n_l1055_105503

theorem find_integer_n (n : ℤ) (hn : -150 < n ∧ n < 150) : (n = 80 ∨ n = -100) ↔ (Real.tan (n * Real.pi / 180) = Real.tan (1340 * Real.pi / 180)) :=
by 
  sorry

end NUMINAMATH_GPT_find_integer_n_l1055_105503


namespace NUMINAMATH_GPT_sunzi_classic_l1055_105524

noncomputable def length_of_rope : ℝ := sorry
noncomputable def length_of_wood : ℝ := sorry
axiom first_condition : length_of_rope - length_of_wood = 4.5
axiom second_condition : length_of_wood - (1 / 2) * length_of_rope = 1

theorem sunzi_classic : 
  (length_of_rope - length_of_wood = 4.5) ∧ (length_of_wood - (1 / 2) * length_of_rope = 1) := 
by 
  exact ⟨first_condition, second_condition⟩

end NUMINAMATH_GPT_sunzi_classic_l1055_105524


namespace NUMINAMATH_GPT_residue_11_pow_1234_mod_19_l1055_105545

theorem residue_11_pow_1234_mod_19 : 
  (11 ^ 1234) % 19 = 11 := 
by
  sorry

end NUMINAMATH_GPT_residue_11_pow_1234_mod_19_l1055_105545


namespace NUMINAMATH_GPT_complex_modulus_squared_l1055_105512

open Complex

theorem complex_modulus_squared (w : ℂ) (h : w^2 + abs w ^ 2 = 7 + 2 * I) : abs w ^ 2 = 53 / 14 :=
sorry

end NUMINAMATH_GPT_complex_modulus_squared_l1055_105512


namespace NUMINAMATH_GPT_cylinder_problem_l1055_105572

theorem cylinder_problem (r h : ℝ) (h1 : π * r^2 * h = 2) (h2 : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 :=
sorry

end NUMINAMATH_GPT_cylinder_problem_l1055_105572


namespace NUMINAMATH_GPT_books_loaned_out_during_month_l1055_105569

-- Define the initial conditions
def initial_books : ℕ := 75
def remaining_books : ℕ := 65
def loaned_out_percentage : ℝ := 0.80
def returned_books_ratio : ℝ := loaned_out_percentage
def not_returned_ratio : ℝ := 1 - returned_books_ratio
def difference : ℕ := initial_books - remaining_books

-- Define the main theorem
theorem books_loaned_out_during_month : ∃ (x : ℕ), not_returned_ratio * (x : ℝ) = (difference : ℝ) ∧ x = 50 :=
by
  existsi 50
  simp [not_returned_ratio, difference]
  sorry

end NUMINAMATH_GPT_books_loaned_out_during_month_l1055_105569


namespace NUMINAMATH_GPT_burger_share_l1055_105507

theorem burger_share (burger_length : ℝ) (brother_share : ℝ) (first_friend_share : ℝ) (second_friend_share : ℝ) (valentina_share : ℝ) :
  burger_length = 12 →
  brother_share = burger_length / 3 →
  first_friend_share = (burger_length - brother_share) / 4 →
  second_friend_share = (burger_length - brother_share - first_friend_share) / 2 →
  valentina_share = burger_length - (brother_share + first_friend_share + second_friend_share) →
  brother_share = 4 ∧ first_friend_share = 2 ∧ second_friend_share = 3 ∧ valentina_share = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_burger_share_l1055_105507


namespace NUMINAMATH_GPT_time_to_fill_tank_l1055_105567

-- Define the rates of the pipes
def rate_first_fill : ℚ := 1 / 15
def rate_second_fill : ℚ := 1 / 15
def rate_outlet_empty : ℚ := -1 / 45

-- Define the combined rate
def combined_rate : ℚ := rate_first_fill + rate_second_fill + rate_outlet_empty

-- Define the time to fill the tank
def fill_time (rate : ℚ) : ℚ := 1 / rate

theorem time_to_fill_tank : fill_time combined_rate = 9 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_time_to_fill_tank_l1055_105567


namespace NUMINAMATH_GPT_prove_inequality_l1055_105531

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_inequality_l1055_105531


namespace NUMINAMATH_GPT_benny_lost_books_l1055_105584

-- Define the initial conditions
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def total_books : ℕ := sandy_books + tim_books
def remaining_books : ℕ := 19

-- Define the proof problem to find out the number of books Benny lost
theorem benny_lost_books : total_books - remaining_books = 24 :=
by
  sorry -- Insert proof here

end NUMINAMATH_GPT_benny_lost_books_l1055_105584


namespace NUMINAMATH_GPT_find_number_l1055_105573

theorem find_number (x : ℝ) (h : (3/4 : ℝ) * x = 93.33333333333333) : x = 124.44444444444444 := 
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_find_number_l1055_105573


namespace NUMINAMATH_GPT_prime_numbers_eq_l1055_105590

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_eq 
  (p q r : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (h : p * (p - 7) + q * (q - 7) = r * (r - 7)) :
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 5 ∧ r = 7) ∨
  (p = 7 ∧ q = 5 ∧ r = 5) ∨ (p = 5 ∧ q = 7 ∧ r = 5) ∨
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) ∨
  (p = 7 ∧ q = 3 ∧ r = 3) ∨ (p = 3 ∧ q = 7 ∧ r = 3) ∨
  (∃ (a : ℕ), is_prime a ∧ p = a ∧ q = 7 ∧ r = a) ∨
  (∃ (a : ℕ), is_prime a ∧ p = 7 ∧ q = a ∧ r = a) :=
sorry

end NUMINAMATH_GPT_prime_numbers_eq_l1055_105590


namespace NUMINAMATH_GPT_consecutive_page_sum_l1055_105577

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) = 2156) : n + (n + 1) = 93 :=
sorry

end NUMINAMATH_GPT_consecutive_page_sum_l1055_105577


namespace NUMINAMATH_GPT_length_A_l1055_105523

open Real

theorem length_A'B'_correct {A B C A' B' : ℝ × ℝ} :
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (C.2 - A.2) / (C.1 - A.1) = ((B.2 - C.2) / (B.1 - C.1)) →
  (dist A' B') = 2.5 * sqrt 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_length_A_l1055_105523


namespace NUMINAMATH_GPT_man_l1055_105543

-- Defining the conditions as variables in Lean
variables (S : ℕ) (M : ℕ)
-- Given conditions
def son_present_age := S = 25
def man_present_age := M = S + 27

-- Goal: the ratio of the man's age to the son's age in two years is 2:1
theorem man's_age_ratio_in_two_years (h1 : son_present_age S) (h2 : man_present_age S M) :
  (M + 2) / (S + 2) = 2 := sorry

end NUMINAMATH_GPT_man_l1055_105543


namespace NUMINAMATH_GPT_prime_divisors_of_50_fact_eq_15_l1055_105571

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end NUMINAMATH_GPT_prime_divisors_of_50_fact_eq_15_l1055_105571


namespace NUMINAMATH_GPT_remainder_when_3_pow_2020_div_73_l1055_105556

theorem remainder_when_3_pow_2020_div_73 :
  (3^2020 % 73) = 8 := 
sorry

end NUMINAMATH_GPT_remainder_when_3_pow_2020_div_73_l1055_105556


namespace NUMINAMATH_GPT_coin_loading_impossible_l1055_105506

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end NUMINAMATH_GPT_coin_loading_impossible_l1055_105506


namespace NUMINAMATH_GPT_line_through_two_points_l1055_105515

theorem line_through_two_points (x_1 y_1 x_2 y_2 x y : ℝ) :
  (x - x_1) * (y_2 - y_1) = (y - y_1) * (x_2 - x_1) :=
sorry

end NUMINAMATH_GPT_line_through_two_points_l1055_105515


namespace NUMINAMATH_GPT_circle_area_approx_error_exceeds_one_l1055_105514

theorem circle_area_approx_error_exceeds_one (r : ℝ) : 
  (3.14159 < Real.pi ∧ Real.pi < 3.14160) → 
  2 * r > 25 →  
  |(r * r * Real.pi - r * r * 3.14)| > 1 → 
  2 * r = 51 := 
by 
  sorry

end NUMINAMATH_GPT_circle_area_approx_error_exceeds_one_l1055_105514


namespace NUMINAMATH_GPT_ages_of_patients_l1055_105593

theorem ages_of_patients (x y : ℕ) 
  (h1 : x - y = 44) 
  (h2 : x * y = 1280) : 
  (x = 64 ∧ y = 20) ∨ (x = 20 ∧ y = 64) := by
  sorry

end NUMINAMATH_GPT_ages_of_patients_l1055_105593


namespace NUMINAMATH_GPT_time_to_cross_bridge_l1055_105533

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (time_min : ℝ) :
  speed_km_hr = 5 → length_m = 1250 → time_min = length_m / (speed_km_hr * 1000 / 60) → time_min = 15 :=
by
  intros h_speed h_length h_time
  rw [h_speed, h_length] at h_time
  -- Since 5 km/hr * 1000 / 60 = 83.33 m/min,
  -- substituting into equation gives us 1250 / 83.33 ≈ 15.
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l1055_105533


namespace NUMINAMATH_GPT_probability_neither_prime_nor_composite_l1055_105501

/-- Definition of prime number: A number is prime if it has exactly two distinct positive divisors -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of composite number: A number is composite if it has more than two positive divisors -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

/-- Given the number in the range 1 to 98 -/
def neither_prime_nor_composite (n : ℕ) : Prop := n = 1

/-- Probability function for uniform probability in a discrete sample space -/
def probability (event_occurrences total_possibilities : ℕ) : ℚ := event_occurrences / total_possibilities

theorem probability_neither_prime_nor_composite :
    probability 1 98 = 1 / 98 := by
  sorry

end NUMINAMATH_GPT_probability_neither_prime_nor_composite_l1055_105501


namespace NUMINAMATH_GPT_sum_inf_evaluation_eq_9_by_80_l1055_105553

noncomputable def infinite_sum_evaluation : ℝ := ∑' n, (2 * n) / (n^4 + 16)

theorem sum_inf_evaluation_eq_9_by_80 :
  infinite_sum_evaluation = 9 / 80 :=
by
  sorry

end NUMINAMATH_GPT_sum_inf_evaluation_eq_9_by_80_l1055_105553


namespace NUMINAMATH_GPT_trigonometric_identity_l1055_105574

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A * Real.cos B * Real.cos C + Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = Real.sin A * Real.sin B * Real.sin C :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1055_105574


namespace NUMINAMATH_GPT_grandmother_times_older_l1055_105589

variables (M G Gr : ℕ)

-- Conditions
def MilenasAge : Prop := M = 7
def GrandfatherAgeRelation : Prop := Gr = G + 2
def AgeDifferenceRelation : Prop := Gr - M = 58

-- Theorem to prove
theorem grandmother_times_older (h1 : MilenasAge M) (h2 : GrandfatherAgeRelation G Gr) (h3 : AgeDifferenceRelation M Gr) :
  G / M = 9 :=
sorry

end NUMINAMATH_GPT_grandmother_times_older_l1055_105589


namespace NUMINAMATH_GPT_problem1_problem2_l1055_105525

-- Problem 1
theorem problem1 (a b : ℝ) : (a + 2 * b)^2 - a * (a + 4 * b) = 4 * b^2 :=
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) : 
  (2 / (m - 1) + 1) / (2 * (m + 1) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1055_105525


namespace NUMINAMATH_GPT_sin_x_correct_l1055_105587

noncomputable def sin_x (a b c : ℝ) (x : ℝ) : ℝ :=
  2 * a * b * c / Real.sqrt (a^4 + 2 * a^2 * b^2 * (c^2 - 1) + b^4)

theorem sin_x_correct (a b c x : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : c > 0) 
  (h₄ : 0 < x ∧ x < Real.pi / 2) 
  (h₅ : Real.tan x = 2 * a * b * c / (a^2 - b^2)) :
  Real.sin x = sin_x a b c x :=
sorry

end NUMINAMATH_GPT_sin_x_correct_l1055_105587


namespace NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1055_105564

noncomputable def zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

noncomputable def decreasing_log (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem necessary_condition (a : ℝ) (h : zero_point a) : 0 < a ∧ a < 1 := sorry

theorem not_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) : ¬(zero_point a) := sorry

end NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1055_105564


namespace NUMINAMATH_GPT_probability_one_in_first_20_rows_l1055_105561

theorem probability_one_in_first_20_rows :
  let total_elements := 210
  let number_of_ones := 39
  (number_of_ones / total_elements : ℚ) = 13 / 70 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_in_first_20_rows_l1055_105561


namespace NUMINAMATH_GPT_lucas_fib_relation_l1055_105578

noncomputable def α := (1 + Real.sqrt 5) / 2
noncomputable def β := (1 - Real.sqrt 5) / 2
def Fib : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 2) => Fib n + Fib (n + 1)

def Lucas : ℕ → ℝ
| 0       => 2
| 1       => 1
| (n + 2) => Lucas n + Lucas (n + 1)

theorem lucas_fib_relation (n : ℕ) (hn : 1 ≤ n) :
  Lucas (2 * n + 1) + (-1)^(n+1) = Fib (2 * n) * Fib (2 * n + 1) := sorry

end NUMINAMATH_GPT_lucas_fib_relation_l1055_105578


namespace NUMINAMATH_GPT_number_of_pupils_l1055_105519

theorem number_of_pupils (n : ℕ) (h1 : 79 - 45 = 34)
  (h2 : 34 = 1 / 2 * n) : n = 68 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pupils_l1055_105519


namespace NUMINAMATH_GPT_quilt_cost_calculation_l1055_105560

theorem quilt_cost_calculation :
  let length := 12
  let width := 15
  let cost_per_sq_foot := 70
  let sales_tax_rate := 0.05
  let discount_rate := 0.10
  let area := length * width
  let cost_before_discount := area * cost_per_sq_foot
  let discount_amount := cost_before_discount * discount_rate
  let cost_after_discount := cost_before_discount - discount_amount
  let sales_tax_amount := cost_after_discount * sales_tax_rate
  let total_cost := cost_after_discount + sales_tax_amount
  total_cost = 11907 := by
  {
    sorry
  }

end NUMINAMATH_GPT_quilt_cost_calculation_l1055_105560


namespace NUMINAMATH_GPT_find_C_l1055_105570

noncomputable def A := {x : ℝ | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 6 = 0}
def C := {a : ℝ | (A ∪ (B a)) = A}

theorem find_C : C = {0, 2, 3} := by
  sorry

end NUMINAMATH_GPT_find_C_l1055_105570


namespace NUMINAMATH_GPT_find_A_plus_B_plus_C_plus_D_l1055_105582

noncomputable def A : ℤ := -7
noncomputable def B : ℕ := 8
noncomputable def C : ℤ := 21
noncomputable def D : ℕ := 1

def conditions_satisfied : Prop :=
  D > 0 ∧
  ¬∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ B ∧ p ≠ 1 ∧ p ≠ B ∧ p ≥ 2 ∧
  Int.gcd A (Int.gcd C (Int.ofNat D)) = 1

theorem find_A_plus_B_plus_C_plus_D : conditions_satisfied → A + B + C + D = 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_A_plus_B_plus_C_plus_D_l1055_105582


namespace NUMINAMATH_GPT_cost_of_article_l1055_105562

variable (C : ℝ)
variable (SP1 SP2 : ℝ)
variable (G : ℝ)

theorem cost_of_article (h1 : SP1 = 380) 
                        (h2 : SP2 = 420)
                        (h3 : SP1 = C + G)
                        (h4 : SP2 = C + G + 0.08 * G) :
  C = 120 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l1055_105562


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1055_105517

variable {x y : ℝ}

theorem sum_of_reciprocals (h1 : x + y = 4 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x + 1 / y) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1055_105517


namespace NUMINAMATH_GPT_tailor_time_l1055_105510

theorem tailor_time (x : ℝ) 
  (t_shirt : ℝ := x) 
  (t_pants : ℝ := 2 * x) 
  (t_jacket : ℝ := 3 * x) 
  (h_capacity : 2 * t_shirt + 3 * t_pants + 4 * t_jacket = 10) : 
  14 * t_shirt + 10 * t_pants + 2 * t_jacket = 20 :=
by
  sorry

end NUMINAMATH_GPT_tailor_time_l1055_105510


namespace NUMINAMATH_GPT_M_values_l1055_105575

theorem M_values (m n p M : ℝ) (h1 : M = m / (n + p)) (h2 : M = n / (p + m)) (h3 : M = p / (m + n)) :
  M = 1 / 2 ∨ M = -1 :=
by
  sorry

end NUMINAMATH_GPT_M_values_l1055_105575


namespace NUMINAMATH_GPT_average_of_two_intermediate_numbers_l1055_105526

theorem average_of_two_intermediate_numbers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
(h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_average : (a + b + c + d) / 4 = 5)
(h_max_diff: (max (max a b) (max c d) - min (min a b) (min c d) = 19)) :
  (a + b + c + d) - (max (max a b) (max c d)) - (min (min a b) (min c d)) = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_average_of_two_intermediate_numbers_l1055_105526


namespace NUMINAMATH_GPT_range_of_a_l1055_105502

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a * x + 3 ≥ a) ↔ -7 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1055_105502


namespace NUMINAMATH_GPT_travel_days_l1055_105546

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end NUMINAMATH_GPT_travel_days_l1055_105546


namespace NUMINAMATH_GPT_total_cost_of_topsoil_l1055_105576

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_topsoil_l1055_105576


namespace NUMINAMATH_GPT_find_pairs_l1055_105599

theorem find_pairs (a b : ℕ) (h1 : a + b = 60) (h2 : Nat.lcm a b = 72) : (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := 
sorry

end NUMINAMATH_GPT_find_pairs_l1055_105599


namespace NUMINAMATH_GPT_sara_initial_quarters_l1055_105586

theorem sara_initial_quarters (total_quarters: ℕ) (dad_gave: ℕ) (initial_quarters: ℕ) 
  (h1: dad_gave = 49) (h2: total_quarters = 70) (h3: total_quarters = initial_quarters + dad_gave) :
  initial_quarters = 21 := 
by {
  sorry
}

end NUMINAMATH_GPT_sara_initial_quarters_l1055_105586


namespace NUMINAMATH_GPT_jeffreys_total_steps_l1055_105518

-- Define the conditions
def effective_steps_per_pattern : ℕ := 1
def total_effective_distance : ℕ := 66
def steps_per_pattern : ℕ := 5

-- Define the proof problem
theorem jeffreys_total_steps : ∀ (N : ℕ), 
  N = (total_effective_distance * steps_per_pattern) := 
sorry

end NUMINAMATH_GPT_jeffreys_total_steps_l1055_105518


namespace NUMINAMATH_GPT_Matt_received_more_pencils_than_Lauren_l1055_105529

-- Definitions based on conditions
def total_pencils := 2 * 12
def pencils_to_Lauren := 6
def pencils_after_Lauren := total_pencils - pencils_to_Lauren
def pencils_left := 9
def pencils_to_Matt := pencils_after_Lauren - pencils_left

-- Formulate the problem statement
theorem Matt_received_more_pencils_than_Lauren (total_pencils := 24) (pencils_to_Lauren := 6) (pencils_after_Lauren := 18) (pencils_left := 9) (correct_answer := 3) :
  pencils_to_Matt - pencils_to_Lauren = correct_answer := 
by 
  sorry

end NUMINAMATH_GPT_Matt_received_more_pencils_than_Lauren_l1055_105529


namespace NUMINAMATH_GPT_correct_ordering_of_fractions_l1055_105547

theorem correct_ordering_of_fractions :
  let a := (6 : ℚ) / 17
  let b := (8 : ℚ) / 25
  let c := (10 : ℚ) / 31
  let d := (1 : ℚ) / 3
  b < d ∧ d < c ∧ c < a :=
by
  sorry

end NUMINAMATH_GPT_correct_ordering_of_fractions_l1055_105547


namespace NUMINAMATH_GPT_kenneth_initial_money_l1055_105532

-- Define the costs of the items
def cost_baguette := 2
def cost_water := 1

-- Define the quantities bought
def baguettes_bought := 2
def water_bought := 2

-- Define the amount left after buying the items
def money_left := 44

-- Calculate the total cost
def total_cost := (baguettes_bought * cost_baguette) + (water_bought * cost_water)

-- Define the initial money Kenneth had
def initial_money := total_cost + money_left

-- Prove the initial money is $50
theorem kenneth_initial_money : initial_money = 50 := 
by 
  -- The proof part is omitted because it is not required.
  sorry

end NUMINAMATH_GPT_kenneth_initial_money_l1055_105532


namespace NUMINAMATH_GPT_simplify_expr_to_polynomial_l1055_105558

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end NUMINAMATH_GPT_simplify_expr_to_polynomial_l1055_105558


namespace NUMINAMATH_GPT_total_cats_correct_l1055_105580

-- Jamie's cats
def Jamie_Persian_cats : ℕ := 4
def Jamie_Maine_Coons : ℕ := 2

-- Gordon's cats
def Gordon_Persian_cats : ℕ := Jamie_Persian_cats / 2
def Gordon_Maine_Coons : ℕ := Jamie_Maine_Coons + 1

-- Hawkeye's cats
def Hawkeye_Persian_cats : ℕ := 0
def Hawkeye_Maine_Coons : ℕ := Gordon_Maine_Coons - 1

-- Total cats for each person
def Jamie_total_cats : ℕ := Jamie_Persian_cats + Jamie_Maine_Coons
def Gordon_total_cats : ℕ := Gordon_Persian_cats + Gordon_Maine_Coons
def Hawkeye_total_cats : ℕ := Hawkeye_Persian_cats + Hawkeye_Maine_Coons

-- Proof that the total number of cats is 13
theorem total_cats_correct : Jamie_total_cats + Gordon_total_cats + Hawkeye_total_cats = 13 :=
by sorry

end NUMINAMATH_GPT_total_cats_correct_l1055_105580


namespace NUMINAMATH_GPT_line_equation_l1055_105538

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_l1055_105538


namespace NUMINAMATH_GPT_largest_w_exists_l1055_105508

theorem largest_w_exists (w x y z : ℝ) (h1 : w + x + y + z = 25) (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 2 * y + 2 * z + 193) :
  ∃ (w1 w2 : ℤ), w1 > 0 ∧ w2 > 0 ∧ ((w = w1 / w2) ∧ (w1 + w2 = 27)) :=
sorry

end NUMINAMATH_GPT_largest_w_exists_l1055_105508


namespace NUMINAMATH_GPT_binomial_sum_of_coefficients_l1055_105549

-- Given condition: for the third term in the expansion, the binomial coefficient is 15
def binomial_coefficient_condition (n : ℕ) := Nat.choose n 2 = 15

-- The goal: the sum of the coefficients of all terms in the expansion is 1/64
theorem binomial_sum_of_coefficients (n : ℕ) (h : binomial_coefficient_condition n) :
  (1:ℚ) / (2 : ℚ)^6 = 1 / 64 :=
by 
  have h₁ : n = 6 := by sorry -- Solve for n using the given condition.
  sorry -- Prove the sum of coefficients when x is 1.

end NUMINAMATH_GPT_binomial_sum_of_coefficients_l1055_105549


namespace NUMINAMATH_GPT_part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l1055_105539

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end NUMINAMATH_GPT_part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l1055_105539


namespace NUMINAMATH_GPT_ordering_y1_y2_y3_l1055_105548

-- Conditions
def A (y₁ : ℝ) : Prop := ∃ b : ℝ, y₁ = -4^2 + 2*4 + b
def B (y₂ : ℝ) : Prop := ∃ b : ℝ, y₂ = -(-1)^2 + 2*(-1) + b
def C (y₃ : ℝ) : Prop := ∃ b : ℝ, y₃ = -(1)^2 + 2*1 + b

-- Question translated to a proof problem
theorem ordering_y1_y2_y3 (y₁ y₂ y₃ : ℝ) :
  A y₁ → B y₂ → C y₃ → y₁ < y₂ ∧ y₂ < y₃ :=
sorry

end NUMINAMATH_GPT_ordering_y1_y2_y3_l1055_105548


namespace NUMINAMATH_GPT_central_angle_of_sector_l1055_105554

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def arc_length (r α : ℝ) : ℝ := r * α

theorem central_angle_of_sector :
  ∀ (r α : ℝ),
    circumference r = 2 * Real.pi + 2 →
    arc_length r α = 2 * Real.pi - 2 →
    α = Real.pi - 1 :=
by
  intros r α hcirc harc
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1055_105554


namespace NUMINAMATH_GPT_find_r_power_4_l1055_105579

variable {r : ℝ}

theorem find_r_power_4 (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := 
sorry

end NUMINAMATH_GPT_find_r_power_4_l1055_105579


namespace NUMINAMATH_GPT_ax5_by5_eq_6200_div_29_l1055_105591

variables (a b x y : ℝ)

-- Given conditions
axiom h1 : a * x + b * y = 5
axiom h2 : a * x^2 + b * y^2 = 11
axiom h3 : a * x^3 + b * y^3 = 30
axiom h4 : a * x^4 + b * y^4 = 80

-- Statement to prove
theorem ax5_by5_eq_6200_div_29 : a * x^5 + b * y^5 = 6200 / 29 :=
by
  sorry

end NUMINAMATH_GPT_ax5_by5_eq_6200_div_29_l1055_105591


namespace NUMINAMATH_GPT_number_of_female_students_l1055_105551

variable (n m : ℕ)

theorem number_of_female_students (hn : n ≥ 0) (hm : m ≥ 0) (hmn : m ≤ n) : n - m = n - m :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_students_l1055_105551


namespace NUMINAMATH_GPT_general_term_formula_of_sequence_l1055_105527

theorem general_term_formula_of_sequence {a : ℕ → ℝ} (S : ℕ → ℝ)
  (hS : ∀ n, S n = (2 / 3) * a n + 1 / 3) :
  (∀ n, a n = (-2) ^ (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_of_sequence_l1055_105527


namespace NUMINAMATH_GPT_sum_of_digits_of_multiple_of_990_l1055_105559

theorem sum_of_digits_of_multiple_of_990 (a b c : ℕ) (h₀ : a < 10 ∧ b < 10 ∧ c < 10)
  (h₁ : ∃ (d e f g : ℕ), 123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c = 123000 + 9000 + 900 + 90 + 9 + 0)
  (h2 : (123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c) % 990 = 0) :
  a + b + c = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_digits_of_multiple_of_990_l1055_105559


namespace NUMINAMATH_GPT_number_of_outfits_l1055_105522

-- Definitions based on conditions a)
def trousers : ℕ := 5
def shirts : ℕ := 7
def jackets : ℕ := 3
def specific_trousers : ℕ := 2
def specific_jackets : ℕ := 2

-- Lean 4 theorem statement to prove the number of outfits
theorem number_of_outfits (trousers shirts jackets specific_trousers specific_jackets : ℕ) :
  (3 * jackets + specific_trousers * specific_jackets) * shirts = 91 :=
by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l1055_105522


namespace NUMINAMATH_GPT_percentage_of_25_of_fifty_percent_of_500_l1055_105534

-- Define the constants involved
def fifty_percent_of_500 := 0.50 * 500  -- 50% of 500

-- Prove the equivalence
theorem percentage_of_25_of_fifty_percent_of_500 : (25 / fifty_percent_of_500) * 100 = 10 := by
  -- Place proof steps here
  sorry

end NUMINAMATH_GPT_percentage_of_25_of_fifty_percent_of_500_l1055_105534


namespace NUMINAMATH_GPT_triangle_exists_among_single_color_sticks_l1055_105504

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end NUMINAMATH_GPT_triangle_exists_among_single_color_sticks_l1055_105504


namespace NUMINAMATH_GPT_pens_cost_l1055_105568

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end NUMINAMATH_GPT_pens_cost_l1055_105568


namespace NUMINAMATH_GPT_difference_apples_peaches_pears_l1055_105598

-- Definitions based on the problem conditions
def apples : ℕ := 60
def peaches : ℕ := 3 * apples
def pears : ℕ := apples / 2

-- Statement of the proof problem
theorem difference_apples_peaches_pears : (apples + peaches) - pears = 210 := by
  sorry

end NUMINAMATH_GPT_difference_apples_peaches_pears_l1055_105598


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1055_105509

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

-- State the theorem about the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1055_105509


namespace NUMINAMATH_GPT_exists_n_in_range_multiple_of_11_l1055_105565

def is_multiple_of_11 (n : ℕ) : Prop :=
  (3 * n^5 + 4 * n^4 + 5 * n^3 + 7 * n^2 + 6 * n + 2) % 11 = 0

theorem exists_n_in_range_multiple_of_11 : ∃ n : ℕ, (2 ≤ n ∧ n ≤ 101) ∧ is_multiple_of_11 n :=
sorry

end NUMINAMATH_GPT_exists_n_in_range_multiple_of_11_l1055_105565


namespace NUMINAMATH_GPT_proportional_segments_l1055_105596

-- Define the problem
theorem proportional_segments :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → (a * d = b * c) → d = 18 :=
by
  intros a b c d ha hb hc hrat
  rw [ha, hb, hc] at hrat
  exact sorry

end NUMINAMATH_GPT_proportional_segments_l1055_105596


namespace NUMINAMATH_GPT_nearest_integer_ratio_l1055_105597

variable (a b : ℝ)

-- Given condition and constraints
def condition : Prop := (a > b) ∧ (b > 0) ∧ (a + b) / 2 = 3 * Real.sqrt (a * b)

-- Main statement to prove
theorem nearest_integer_ratio (h : condition a b) : Int.floor (a / b) = 34 ∨ Int.floor (a / b) = 33 := sorry

end NUMINAMATH_GPT_nearest_integer_ratio_l1055_105597


namespace NUMINAMATH_GPT_octagon_area_sum_l1055_105588

theorem octagon_area_sum :
  let A1 := 2024
  let a := 1012
  let b := 506
  let c := 2
  a + b + c = 1520 := by
    sorry

end NUMINAMATH_GPT_octagon_area_sum_l1055_105588
