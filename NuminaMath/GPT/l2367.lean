import Mathlib

namespace system_of_equations_solution_l2367_236757

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x + z = 0) 
  (h3 : y + z = 1) : 
  x = -1 ∧ y = 0 ∧ z = 1 :=
by
  sorry

end system_of_equations_solution_l2367_236757


namespace least_multiple_x_correct_l2367_236748

noncomputable def least_multiple_x : ℕ :=
  let x := 20
  let y := 8
  let z := 5
  5 * y

theorem least_multiple_x_correct (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 33) (h5 : 5 * y = 8 * z) : least_multiple_x = 40 :=
by
  sorry

end least_multiple_x_correct_l2367_236748


namespace four_digit_number_perfect_square_l2367_236742

theorem four_digit_number_perfect_square (abcd : ℕ) (h1 : abcd ≥ 1000 ∧ abcd < 10000) (h2 : ∃ k : ℕ, k^2 = 4000000 + abcd) :
  abcd = 4001 ∨ abcd = 8004 :=
sorry

end four_digit_number_perfect_square_l2367_236742


namespace product_div_by_six_l2367_236731

theorem product_div_by_six (A B C : ℤ) (h1 : A^2 + B^2 = C^2) 
  (h2 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 4 * k + 2) 
  (h3 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 3 * k + 2) : 
  6 ∣ (A * B) :=
sorry

end product_div_by_six_l2367_236731


namespace chris_money_before_birthday_l2367_236798

variables {x : ℕ} -- Assuming we are working with natural numbers (non-negative integers)

-- Conditions
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Question
theorem chris_money_before_birthday : x = total_money_now - (grandmother_money + aunt_uncle_money + parents_money) :=
by
  sorry

end chris_money_before_birthday_l2367_236798


namespace only_integer_solution_l2367_236794

theorem only_integer_solution (a b c d : ℤ) (h : a^2 + b^2 = 3 * (c^2 + d^2)) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
by
  sorry

end only_integer_solution_l2367_236794


namespace empty_set_iff_k_single_element_set_iff_k_l2367_236754

noncomputable def quadratic_set (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

theorem empty_set_iff_k (k : ℝ) : 
  quadratic_set k = ∅ ↔ k > 9/8 := by
  sorry

theorem single_element_set_iff_k (k : ℝ) : 
  (∃ x : ℝ, quadratic_set k = {x}) ↔ (k = 0 ∧ quadratic_set k = {2 / 3}) ∨ (k = 9 / 8 ∧ quadratic_set k = {4 / 3}) := by
  sorry

end empty_set_iff_k_single_element_set_iff_k_l2367_236754


namespace gummy_bear_production_time_l2367_236728

theorem gummy_bear_production_time 
  (gummy_bears_per_minute : ℕ)
  (gummy_bears_per_packet : ℕ)
  (total_packets : ℕ)
  (h1 : gummy_bears_per_minute = 300)
  (h2 : gummy_bears_per_packet = 50)
  (h3 : total_packets = 240) :
  (total_packets / (gummy_bears_per_minute / gummy_bears_per_packet) = 40) :=
sorry

end gummy_bear_production_time_l2367_236728


namespace find_A2_A7_l2367_236706

theorem find_A2_A7 (A : ℕ → ℝ) (hA1A11 : A 11 - A 1 = 56)
  (hAiAi2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → A (i+2) - A i ≤ 12)
  (hAjAj3 : ∀ j, 1 ≤ j ∧ j ≤ 8 → A (j+3) - A j ≥ 17) : 
  A 7 - A 2 = 29 :=
by
  sorry

end find_A2_A7_l2367_236706


namespace jori_water_left_l2367_236710

theorem jori_water_left (initial used : ℚ) (h1 : initial = 3) (h2 : used = 4 / 3) :
  initial - used = 5 / 3 :=
by
  sorry

end jori_water_left_l2367_236710


namespace average_of_numbers_l2367_236702

theorem average_of_numbers (x : ℝ) (h : (5 + -1 + -2 + x) / 4 = 1) : x = 2 :=
by
  sorry

end average_of_numbers_l2367_236702


namespace quadratic_inequality_solution_l2367_236751

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) := sorry

end quadratic_inequality_solution_l2367_236751


namespace regular_polygon_sides_l2367_236788

theorem regular_polygon_sides (n : ℕ) (h1 : ∃ a : ℝ, a = 120 ∧ ∀ i < n, 120 = a) : n = 6 :=
by
  sorry

end regular_polygon_sides_l2367_236788


namespace sum_of_coefficients_l2367_236741

def P (x : ℝ) : ℝ :=
  -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : P 1 = 48 := by
  sorry

end sum_of_coefficients_l2367_236741


namespace factorization_correct_l2367_236733

theorem factorization_correct (x : ℝ) : 
  98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) :=
by
  sorry

end factorization_correct_l2367_236733


namespace find_c_l2367_236790

   variable {a b c : ℝ}
   
   theorem find_c (h1 : 4 * a - 3 * b + c = 0)
     (h2 : (a - 1)^2 + (b - 1)^2 = 4) :
     c = 9 ∨ c = -11 := 
   by
     sorry
   
end find_c_l2367_236790


namespace sequence_difference_l2367_236762

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + n) : a 2017 - a 2016 = 2016 :=
sorry

end sequence_difference_l2367_236762


namespace stock_worth_is_100_l2367_236787

-- Define the number of puppies and kittens
def num_puppies : ℕ := 2
def num_kittens : ℕ := 4

-- Define the cost per puppy and kitten
def cost_per_puppy : ℕ := 20
def cost_per_kitten : ℕ := 15

-- Define the total stock worth function
def stock_worth (num_puppies num_kittens cost_per_puppy cost_per_kitten : ℕ) : ℕ :=
  (num_puppies * cost_per_puppy) + (num_kittens * cost_per_kitten)

-- The theorem to prove that the stock worth is $100
theorem stock_worth_is_100 :
  stock_worth num_puppies num_kittens cost_per_puppy cost_per_kitten = 100 :=
by
  sorry

end stock_worth_is_100_l2367_236787


namespace no_int_representation_l2367_236708

theorem no_int_representation (A B : ℤ) : (99999 + 111111 * Real.sqrt 3) ≠ (A + B * Real.sqrt 3)^2 :=
by
  sorry

end no_int_representation_l2367_236708


namespace compare_sqrt_differences_l2367_236705

theorem compare_sqrt_differences :
  let a := (Real.sqrt 7) - (Real.sqrt 6)
  let b := (Real.sqrt 3) - (Real.sqrt 2)
  a < b :=
by
  sorry -- Proof goes here

end compare_sqrt_differences_l2367_236705


namespace divides_or_l2367_236768

-- Definitions
variables {m n : ℕ} -- using natural numbers (non-negative integers) for simplicity in Lean

-- Hypothesis: m ∨ n + m ∧ n = m + n
theorem divides_or (h : Nat.lcm m n + Nat.gcd m n = m + n) : m ∣ n ∨ n ∣ m :=
sorry

end divides_or_l2367_236768


namespace car_speed_l2367_236772

theorem car_speed (distance time : ℝ) (h1 : distance = 300) (h2 : time = 5) : distance / time = 60 := by
  have h : distance / time = 300 / 5 := by
    rw [h1, h2]
  norm_num at h
  exact h

end car_speed_l2367_236772


namespace inequality_solution_l2367_236735

theorem inequality_solution (x : ℝ) : (x^3 - 12*x^2 + 36*x > 0) ↔ (0 < x ∧ x < 6) ∨ (x > 6) := by
  sorry

end inequality_solution_l2367_236735


namespace estimate_sqrt_interval_l2367_236774

theorem estimate_sqrt_interval : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_sqrt_interval_l2367_236774


namespace middle_school_students_count_l2367_236764

def split_equally (m h : ℕ) : Prop := m = h
def percent_middle (M m : ℕ) : Prop := m = M / 5
def percent_high (H h : ℕ) : Prop := h = 3 * H / 10
def total_students (M H : ℕ) : Prop := M + H = 50
def number_of_middle_school_students (M: ℕ) := M

theorem middle_school_students_count (M H m h : ℕ) 
  (hm_eq : split_equally m h) 
  (hm_percent : percent_middle M m) 
  (hh_percent : percent_high H h) 
  (htotal : total_students M H) : 
  number_of_middle_school_students M = 30 :=
by
  sorry

end middle_school_students_count_l2367_236764


namespace smallest_x_l2367_236779

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_x (x a : ℕ) (h1 : a = 100 * x + 4950)
  (h2 : digitSum a = 50) :
  x = 99950 :=
by sorry

end smallest_x_l2367_236779


namespace complement_union_A_B_l2367_236720

-- Define the sets U, A, and B as per the conditions
def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Specify the statement to prove the complement of A ∪ B with respect to U
theorem complement_union_A_B : (U \ (A ∪ B)) = {2, 4} :=
by
  sorry

end complement_union_A_B_l2367_236720


namespace smallest_perimeter_scalene_triangle_l2367_236755

theorem smallest_perimeter_scalene_triangle (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) :
  a + b + c = 9 := 
sorry

end smallest_perimeter_scalene_triangle_l2367_236755


namespace remainder_of_2356912_div_8_l2367_236789

theorem remainder_of_2356912_div_8 : 912 % 8 = 0 := 
by 
  sorry

end remainder_of_2356912_div_8_l2367_236789


namespace f_even_l2367_236752

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l2367_236752


namespace find_a_l2367_236793

noncomputable def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a (a : ℕ) (h : collinear (a, 0) (0, a + 4) (1, 3)) : a = 4 :=
by
  sorry

end find_a_l2367_236793


namespace sum_fractions_geq_six_l2367_236736

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem sum_fractions_geq_six : 
  (x / y + y / z + z / x + x / z + z / y + y / x) ≥ 6 := 
by
  sorry

end sum_fractions_geq_six_l2367_236736


namespace find_x_l2367_236799

theorem find_x (x : ℕ) (a : ℕ) (h₁: a = 450) (h₂: (15^x * 8^3) / 256 = a) : x = 2 :=
by
  sorry

end find_x_l2367_236799


namespace trailing_zeros_300_factorial_l2367_236781

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l2367_236781


namespace original_number_is_80_l2367_236723

-- Define the existence of the numbers A and B
variable (A B : ℕ)

-- Define the conditions from the problem
def conditions :=
  A = 35 ∧ A / 7 = B / 9

-- Define the statement to prove
theorem original_number_is_80 (h : conditions A B) : A + B = 80 :=
by
  -- Proof is omitted
  sorry

end original_number_is_80_l2367_236723


namespace find_pairs_l2367_236703

theorem find_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m^2 + n^2) ∣ (3 * m * n + 3 * m) ↔ (m, n) = (1, 1) ∨ (m, n) = (4, 2) ∨ (m, n) = (4, 10) :=
sorry

end find_pairs_l2367_236703


namespace factorize_polynomial_l2367_236709
   
   -- Define the polynomial
   def polynomial (x : ℝ) : ℝ :=
     x^3 + 3 * x^2 - 4
   
   -- Define the factorized form
   def factorized_form (x : ℝ) : ℝ :=
     (x - 1) * (x + 2)^2
   
   -- The theorem statement
   theorem factorize_polynomial (x : ℝ) : polynomial x = factorized_form x := 
   by
     sorry
   
end factorize_polynomial_l2367_236709


namespace quadratic_equality_l2367_236763

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l2367_236763


namespace find_last_four_digits_of_N_l2367_236711

def P (n : Nat) : Nat :=
  match n with
  | 0     => 1 -- usually not needed but for completeness
  | 1     => 2
  | _     => 2 + (n - 1) * n

theorem find_last_four_digits_of_N : (P 2011) % 10000 = 2112 := by
  -- we define P(2011) as per the general formula derived and then verify the modulo operation
  sorry

end find_last_four_digits_of_N_l2367_236711


namespace solve_system_of_equations_l2367_236717

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations_l2367_236717


namespace correct_solution_l2367_236783

theorem correct_solution : 
  ∀ (x y a b : ℚ), (a = 1) → (b = 1 / 2) → 
  (a * x + y = 2) → (2 * x - b * y = 1) → 
  (x = 4 / 5 ∧ y = 6 / 5) := 
by
  intros x y a b ha hb h1 h2
  sorry

end correct_solution_l2367_236783


namespace gumball_cost_l2367_236701

theorem gumball_cost (n : ℕ) (T : ℕ) (h₁ : n = 4) (h₂ : T = 32) : T / n = 8 := by
  sorry

end gumball_cost_l2367_236701


namespace find_t1_t2_l2367_236745

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (1, 2)

-- Define the conditions for t1 and t2
def t1_condition (t1 : ℝ) : Prop := (2 / 1) = (t1 / 2)
def t2_condition (t2 : ℝ) : Prop := (2 * 1 + t2 * 2 = 0)

-- The statement to prove
theorem find_t1_t2 (t1 t2 : ℝ) (h1 : t1_condition t1) (h2 : t2_condition t2) : (t1 = 4) ∧ (t2 = -1) :=
by
  sorry

end find_t1_t2_l2367_236745


namespace instantaneous_rate_of_change_at_x1_l2367_236784

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - x^2 + 8

theorem instantaneous_rate_of_change_at_x1 : deriv f 1 = -1 := by
  sorry

end instantaneous_rate_of_change_at_x1_l2367_236784


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l2367_236712

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l2367_236712


namespace weighted_avg_surfers_per_day_l2367_236729

theorem weighted_avg_surfers_per_day 
  (total_surfers : ℕ) 
  (ratio1_day1 ratio1_day2 ratio2_day3 ratio2_day4 : ℕ) 
  (h_total_surfers : total_surfers = 12000)
  (h_ratio_first_two_days : ratio1_day1 = 5 ∧ ratio1_day2 = 7)
  (h_ratio_last_two_days : ratio2_day3 = 3 ∧ ratio2_day4 = 2) 
  : (total_surfers / (ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4)) * 
    ((ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4) / 4) = 3000 :=
by
  sorry

end weighted_avg_surfers_per_day_l2367_236729


namespace vertical_lines_count_l2367_236750

theorem vertical_lines_count (n : ℕ) 
  (h_intersections : (18 * n * (n - 1)) = 756) : 
  n = 7 :=
by 
  sorry

end vertical_lines_count_l2367_236750


namespace phoenix_flight_l2367_236769

theorem phoenix_flight : ∃ n : ℕ, 3 ^ n > 6560 ∧ ∀ m < n, 3 ^ m ≤ 6560 :=
by sorry

end phoenix_flight_l2367_236769


namespace minimum_value_l2367_236753

-- Define geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * ((a 2 / a 1) ^ n)

-- Define the condition for positive geometric sequence
def positive_geometric_sequence (a : ℕ → ℝ) :=
  is_geometric_sequence a ∧ ∀ n : ℕ, a n > 0

-- Condition given in the problem
def condition (a : ℕ → ℝ) :=
  2 * a 4 + a 3 = 2 * a 2 + a 1 + 8

-- Define the problem statement to be proved
theorem minimum_value (a : ℕ → ℝ) (h1 : positive_geometric_sequence a) (h2 : condition a) :
  2 * a 6 + a 5 = 32 :=
sorry

end minimum_value_l2367_236753


namespace power_function_solution_l2367_236744

theorem power_function_solution (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = x ^ α) (h2 : f 4 = 2) : f 3 = Real.sqrt 3 :=
sorry

end power_function_solution_l2367_236744


namespace water_left_after_four_hours_l2367_236776

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l2367_236776


namespace cubic_roots_result_l2367_236722

theorem cubic_roots_result (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 64 + b * 16 + c * 4 + d = 0) (h₃ : a * (-27) + b * 9 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end cubic_roots_result_l2367_236722


namespace necklaces_made_l2367_236738

theorem necklaces_made (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 18) (h2 : beads_per_necklace = 3) : total_beads / beads_per_necklace = 6 := 
by {
  sorry
}

end necklaces_made_l2367_236738


namespace original_square_area_is_correct_l2367_236713

noncomputable def original_square_side_length (s : ℝ) :=
  let original_area := s^2
  let new_width := 0.8 * s
  let new_length := 5 * s
  let new_area := new_width * new_length
  let increased_area := new_area - original_area
  increased_area = 15.18

theorem original_square_area_is_correct (s : ℝ) (h : original_square_side_length s) : s^2 = 5.06 := by
  sorry

end original_square_area_is_correct_l2367_236713


namespace yogurt_cost_l2367_236760

-- Define the conditions given in the problem
def total_cost_ice_cream : ℕ := 20 * 6
def spent_difference : ℕ := 118

theorem yogurt_cost (y : ℕ) 
  (h1 : total_cost_ice_cream = 2 * y + spent_difference) : 
  y = 1 :=
  sorry

end yogurt_cost_l2367_236760


namespace fruit_seller_profit_l2367_236771

theorem fruit_seller_profit 
  (SP : ℝ) (Loss_Percentage : ℝ) (New_SP : ℝ) (Profit_Percentage : ℝ) 
  (h1: SP = 8) 
  (h2: Loss_Percentage = 20) 
  (h3: New_SP = 10.5) 
  (h4: Profit_Percentage = 5) :
  ((New_SP - (SP / (1 - (Loss_Percentage / 100.0))) / (SP / (1 - (Loss_Percentage / 100.0)))) * 100) = Profit_Percentage := 
sorry

end fruit_seller_profit_l2367_236771


namespace number_of_zeros_of_g_is_4_l2367_236767

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ := 
  f (f x + 2) + 2

theorem number_of_zeros_of_g_is_4 : 
  ∃ S : Finset ℝ, S.card = 4 ∧ ∀ x ∈ S, g x = 0 :=
sorry

end number_of_zeros_of_g_is_4_l2367_236767


namespace negation_of_p_l2367_236777

open Classical

variable (p : Prop)

theorem negation_of_p (h : ∀ x : ℝ, x^3 + 2 < 0) : 
  ∃ x : ℝ, x^3 + 2 ≥ 0 :=
by
  sorry

end negation_of_p_l2367_236777


namespace sufficient_but_not_necessary_l2367_236724

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > b + 1) → (a > b) ∧ ¬(a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_l2367_236724


namespace find_d_l2367_236786

theorem find_d (d : ℚ) (h : ∀ x : ℚ, 4*x^3 + 17*x^2 + d*x + 28 = 0 → x = -4/3) : d = 155 / 9 :=
sorry

end find_d_l2367_236786


namespace Brenda_weight_correct_l2367_236718

-- Conditions
def MelWeight : ℕ := 70
def BrendaWeight : ℕ := 3 * MelWeight + 10

-- Proof problem
theorem Brenda_weight_correct : BrendaWeight = 220 := by
  sorry

end Brenda_weight_correct_l2367_236718


namespace cubic_roots_relations_l2367_236780

theorem cubic_roots_relations 
    (a b c d : ℚ) 
    (x1 x2 x3 : ℚ) 
    (h : a ≠ 0)
    (hroots : a * x1^3 + b * x1^2 + c * x1 + d = 0 
      ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 
      ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
    :
    (x1 + x2 + x3 = -b / a) 
    ∧ (x1 * x2 + x1 * x3 + x2 * x3 = c / a) 
    ∧ (x1 * x2 * x3 = -d / a) := 
sorry

end cubic_roots_relations_l2367_236780


namespace rectangle_x_satisfy_l2367_236761

theorem rectangle_x_satisfy (x : ℝ) (h1 : 3 * x = 3 * x) (h2 : x + 5 = x + 5) (h3 : (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5)) : x = 1 :=
sorry

end rectangle_x_satisfy_l2367_236761


namespace range_of_a_l2367_236765

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) : -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l2367_236765


namespace number_of_pairs_101_l2367_236758

theorem number_of_pairs_101 :
  (∃ n : ℕ, (∀ a b : ℕ, (a > 0) → (b > 0) → (a + b = 101) → (b > a) → (n = 50))) :=
sorry

end number_of_pairs_101_l2367_236758


namespace Vikas_submitted_6_questions_l2367_236727

theorem Vikas_submitted_6_questions (R V A : ℕ) (h1 : 7 * V = 3 * R) (h2 : 2 * V = 3 * A) (h3 : R + V + A = 24) : V = 6 :=
by
  sorry

end Vikas_submitted_6_questions_l2367_236727


namespace average_cost_per_individual_before_gratuity_l2367_236746

theorem average_cost_per_individual_before_gratuity
  (total_bill : ℝ)
  (num_people : ℕ)
  (gratuity_percentage : ℝ)
  (bill_including_gratuity : total_bill = 840)
  (group_size : num_people = 7)
  (gratuity : gratuity_percentage = 0.20) :
  (total_bill / (1 + gratuity_percentage)) / num_people = 100 :=
by
  sorry

end average_cost_per_individual_before_gratuity_l2367_236746


namespace permutation_by_transpositions_l2367_236726

-- Formalizing the conditions in Lean
section permutations
  variable {n : ℕ}

  -- Define permutations
  def is_permutation (σ : Fin n → Fin n) : Prop :=
    ∃ σ_inv : Fin n → Fin n, 
      (∀ i, σ (σ_inv i) = i) ∧ 
      (∀ i, σ_inv (σ i) = i)

  -- Define transposition
  def transposition (σ : Fin n → Fin n) (i j : Fin n) : Fin n → Fin n :=
    fun x => if x = i then j else if x = j then i else σ x

  -- Main theorem stating that any permutation can be obtained through a series of transpositions
  theorem permutation_by_transpositions (σ : Fin n → Fin n) (h : is_permutation σ) :
    ∃ τ : ℕ → (Fin n → Fin n),
      (∀ i, is_permutation (τ i)) ∧
      (∀ m, ∃ k, τ m = transposition (τ (m - 1)) (⟨ k, sorry ⟩) (σ (⟨ k, sorry⟩))) ∧
      (∃ m, τ m = σ) :=
  sorry
end permutations

end permutation_by_transpositions_l2367_236726


namespace city_rentals_cost_per_mile_l2367_236785

-- The parameters provided in the problem
def safety_base_rate : ℝ := 21.95
def safety_per_mile_rate : ℝ := 0.19
def city_base_rate : ℝ := 18.95
def miles_driven : ℝ := 150.0

-- The cost expressions based on the conditions
def safety_total_cost (miles: ℝ) : ℝ := safety_base_rate + safety_per_mile_rate * miles
def city_total_cost (miles: ℝ) (city_per_mile_rate: ℝ) : ℝ := city_base_rate + city_per_mile_rate * miles

-- The cost equality condition for 150 miles
def cost_condition : Prop :=
  safety_total_cost miles_driven = city_total_cost miles_driven 0.21

-- Prove that the cost per mile for City Rentals is 0.21 dollars
theorem city_rentals_cost_per_mile : cost_condition :=
by
  -- Start the proof
  sorry

end city_rentals_cost_per_mile_l2367_236785


namespace am_gm_iq_l2367_236721

theorem am_gm_iq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (a + 1/a) * (b + 1/b) ≥ 25/4 := sorry

end am_gm_iq_l2367_236721


namespace total_spots_l2367_236719

variable (P : ℕ)
variable (Bill_spots : ℕ := 2 * P - 1)

-- Given conditions
variable (h1 : Bill_spots = 39)

-- Theorem we need to prove
theorem total_spots (P : ℕ) (Bill_spots : ℕ := 2 * P - 1) (h1 : Bill_spots = 39) : 
  Bill_spots + P = 59 := 
by
  sorry

end total_spots_l2367_236719


namespace multiple_of_regular_rate_is_1_5_l2367_236770

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5_l2367_236770


namespace intersection_eq_l2367_236700

variable {x : ℝ}

def set_A := {x : ℝ | x^2 - 4 * x < 0}
def set_B := {x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5}
def set_intersection := {x : ℝ | 1 / 3 ≤ x ∧ x < 4}

theorem intersection_eq : (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_eq_l2367_236700


namespace remainder_div_by_7_l2367_236704

theorem remainder_div_by_7 (n : ℤ) (k m : ℤ) (r : ℤ) (h₀ : n = 7 * k + r) (h₁ : 3 * n = 7 * m + 3) (hrange : 0 ≤ r ∧ r < 7) : r = 1 :=
by
  sorry

end remainder_div_by_7_l2367_236704


namespace combined_percent_increase_proof_l2367_236743

variable (initial_stock_A_price : ℝ := 25)
variable (initial_stock_B_price : ℝ := 45)
variable (initial_stock_C_price : ℝ := 60)
variable (final_stock_A_price : ℝ := 28)
variable (final_stock_B_price : ℝ := 50)
variable (final_stock_C_price : ℝ := 75)

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  (percent_increase initial_a final_a + percent_increase initial_b final_b + percent_increase initial_c final_c) / 3

theorem combined_percent_increase_proof :
  combined_percent_increase initial_stock_A_price initial_stock_B_price initial_stock_C_price
                            final_stock_A_price final_stock_B_price final_stock_C_price = 16.04 := by
  sorry

end combined_percent_increase_proof_l2367_236743


namespace sum_mod_11_l2367_236714

theorem sum_mod_11 (h1 : 8735 % 11 = 1) (h2 : 8736 % 11 = 2) (h3 : 8737 % 11 = 3) (h4 : 8738 % 11 = 4) :
  (8735 + 8736 + 8737 + 8738) % 11 = 10 :=
by
  sorry

end sum_mod_11_l2367_236714


namespace prod_eq_one_l2367_236782

noncomputable def is_parity_equal (A : Finset ℝ) (a : ℝ) : Prop :=
  (A.filter (fun x => x > a)).card % 2 = (A.filter (fun x => x < 1/a)).card % 2

theorem prod_eq_one
  (A : Finset ℝ)
  (hA : ∀ (a : ℝ), 0 < a → is_parity_equal A a)
  (hA_pos : ∀ x ∈ A, 0 < x) :
  A.prod id = 1 :=
sorry

end prod_eq_one_l2367_236782


namespace focus_of_parabola_l2367_236730

theorem focus_of_parabola : (∃ p : ℝ × ℝ, p = (-1, 35/12)) :=
by
  sorry

end focus_of_parabola_l2367_236730


namespace smallest_integer_remainder_l2367_236732

theorem smallest_integer_remainder :
  ∃ n : ℕ, n > 1 ∧
           (n % 3 = 2) ∧
           (n % 4 = 2) ∧
           (n % 5 = 2) ∧
           (n % 7 = 2) ∧
           n = 422 :=
by
  sorry

end smallest_integer_remainder_l2367_236732


namespace square_side_length_l2367_236740

theorem square_side_length (s : ℝ) (h : s^2 = 1 / 9) : s = 1 / 3 :=
sorry

end square_side_length_l2367_236740


namespace factorization_correct_l2367_236756

theorem factorization_correct (x : ℝ) : 
  (x^2 + 5 * x + 2) * (x^2 + 5 * x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5 * x - 1) :=
by
  sorry

end factorization_correct_l2367_236756


namespace fraction_n_p_l2367_236791

theorem fraction_n_p (m n p : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * r2 = m)
  (h2 : -(r1 + r2) = p)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0)
  (h5 : p ≠ 0)
  (h6 : m = - (r1 + r2) / 2)
  (h7 : n = r1 * r2 / 4) :
  n / p = 1 / 8 :=
by
  sorry

end fraction_n_p_l2367_236791


namespace minimum_expenses_for_Nikifor_to_win_maximum_F_value_l2367_236759

noncomputable def number_of_voters := 35
noncomputable def sellable_voters := 14 -- 40% of 35
noncomputable def preference_voters := 21 -- 60% of 35
noncomputable def minimum_votes_to_win := 18 -- 50% of 35 + 1
noncomputable def cost_per_vote := 9

def vote_supply_function (P : ℕ) : ℕ :=
  if P = 0 then 10
  else if 1 ≤ P ∧ P ≤ 14 then 10 + P
  else 24


theorem minimum_expenses_for_Nikifor_to_win :
  ∃ P : ℕ, P * cost_per_vote = 162 ∧ vote_supply_function P ≥ minimum_votes_to_win := 
sorry

theorem maximum_F_value (F : ℕ) : 
  F = 3 :=
sorry

end minimum_expenses_for_Nikifor_to_win_maximum_F_value_l2367_236759


namespace roberta_listen_days_l2367_236707

-- Define the initial number of records
def initial_records : ℕ := 8

-- Define the number of records received as gifts
def gift_records : ℕ := 12

-- Define the number of records bought
def bought_records : ℕ := 30

-- Define the number of days to listen to 1 record
def days_per_record : ℕ := 2

-- Define the total number of records
def total_records : ℕ := initial_records + gift_records + bought_records

-- Define the total number of days required to listen to all records
def total_days : ℕ := total_records * days_per_record

-- Theorem to prove the total days needed to listen to all records is 100
theorem roberta_listen_days : total_days = 100 := by
  sorry

end roberta_listen_days_l2367_236707


namespace bookstore_purchase_prices_equal_l2367_236766

variable (x : ℝ)

theorem bookstore_purchase_prices_equal
  (h1 : 500 > 0)
  (h2 : 700 > 0)
  (h3 : x > 0)
  (h4 : x + 4 > 0)
  (h5 : ∃ p₁ p₂ : ℝ, p₁ = 500 / x ∧ p₂ = 700 / (x + 4) ∧ p₁ = p₂) :
  500 / x = 700 / (x + 4) :=
by
  sorry

end bookstore_purchase_prices_equal_l2367_236766


namespace complement_union_l2367_236775

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l2367_236775


namespace smaller_two_digit_product_l2367_236792

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l2367_236792


namespace find_k_l2367_236747

theorem find_k (k x : ℝ) (h1 : x + k - 4 = 0) (h2 : x = 2) : k = 2 :=
by
  sorry

end find_k_l2367_236747


namespace convex_pentagons_l2367_236796

theorem convex_pentagons (P : Finset ℝ) (h : P.card = 15) : 
  (P.card.choose 5) = 3003 := 
by
  sorry

end convex_pentagons_l2367_236796


namespace math_problem_l2367_236778

variable (f : ℝ → ℝ)

-- Conditions
axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

-- Proof goals
theorem math_problem :
  (f 0 = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f (x + 6) = f x) :=
by 
  sorry

end math_problem_l2367_236778


namespace area_of_given_triangle_is_32_l2367_236795

noncomputable def area_of_triangle : ℕ :=
  let A := (-8, 0)
  let B := (0, 8)
  let C := (0, 0)
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℤ).natAbs

theorem area_of_given_triangle_is_32 : area_of_triangle = 32 := 
  sorry

end area_of_given_triangle_is_32_l2367_236795


namespace average_is_0_1667X_plus_3_l2367_236797

noncomputable def average_of_three_numbers (X Y Z : ℝ) : ℝ := (X + Y + Z) / 3

theorem average_is_0_1667X_plus_3 (X Y Z : ℝ) 
  (h1 : 2001 * Z - 4002 * X = 8008) 
  (h2 : 2001 * Y + 5005 * X = 10010) : 
  average_of_three_numbers X Y Z = 0.1667 * X + 3 := 
sorry

end average_is_0_1667X_plus_3_l2367_236797


namespace cone_volume_l2367_236773

theorem cone_volume (r h l : ℝ) (π := Real.pi)
  (slant_height : l = 5)
  (lateral_area : π * r * l = 20 * π) :
  (1 / 3) * π * r^2 * h = 16 * π :=
by
  -- Definitions based on conditions
  let slant_height_definition := slant_height
  let lateral_area_definition := lateral_area
  
  -- Need actual proof steps which are omitted using sorry
  sorry

end cone_volume_l2367_236773


namespace questionnaires_drawn_from_unit_D_l2367_236715

theorem questionnaires_drawn_from_unit_D 
  (total_sample: ℕ) 
  (sample_from_B: ℕ) 
  (d: ℕ) 
  (h_total_sample: total_sample = 150) 
  (h_sample_from_B: sample_from_B = 30) 
  (h_arithmetic_sequence: (30 - d) + 30 + (30 + d) + (30 + 2 * d) = total_sample) 
  : 30 + 2 * d = 60 :=
by 
  sorry

end questionnaires_drawn_from_unit_D_l2367_236715


namespace find_total_amount_before_brokerage_l2367_236725

noncomputable def total_amount_before_brokerage (realized_amount : ℝ) (brokerage_rate : ℝ) : ℝ :=
  realized_amount / (1 - brokerage_rate / 100)

theorem find_total_amount_before_brokerage :
  total_amount_before_brokerage 107.25 (1 / 4) = 107.25 * 400 / 399 := by
sorry

end find_total_amount_before_brokerage_l2367_236725


namespace dependence_of_Q_l2367_236716

theorem dependence_of_Q (a d k : ℕ) :
    ∃ (Q : ℕ), Q = (2 * k * (2 * a + 4 * k * d - d)) 
                - (k * (2 * a + (2 * k - 1) * d)) 
                - (k / 2 * (2 * a + (k - 1) * d)) 
                → Q = k * a + 13 * k^2 * d := 
sorry

end dependence_of_Q_l2367_236716


namespace factor_expression_l2367_236734

variable (x y : ℝ)

theorem factor_expression :
(3*x^3 + 28*(x^2)*y + 4*x) - (-4*x^3 + 5*(x^2)*y - 4*x) = x*(x + 8)*(7*x + 1) := sorry

end factor_expression_l2367_236734


namespace gcd_150_450_l2367_236737

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end gcd_150_450_l2367_236737


namespace recurring_decimal_sum_l2367_236749

-- Definitions based on the conditions identified
def recurringDecimal (n : ℕ) : ℚ := n / 9
def r8 := recurringDecimal 8
def r2 := recurringDecimal 2
def r6 := recurringDecimal 6
def r6_simplified : ℚ := 2 / 3

-- The theorem to prove
theorem recurring_decimal_sum : r8 + r2 - r6_simplified = 4 / 9 :=
by
  -- Proof steps will go here (but are omitted because of the problem requirements)
  sorry

end recurring_decimal_sum_l2367_236749


namespace rectangle_area_l2367_236739

-- Definitions of the conditions
variables (Length Width Area : ℕ)
variable (h1 : Length = 4 * Width)
variable (h2 : Length = 20)

-- Statement to prove
theorem rectangle_area : Area = Length * Width → Area = 100 :=
by
  sorry

end rectangle_area_l2367_236739
