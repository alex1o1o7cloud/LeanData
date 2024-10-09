import Mathlib

namespace tutors_next_together_in_360_days_l1998_199856

open Nat

-- Define the intervals for each tutor
def evan_interval := 5
def fiona_interval := 6
def george_interval := 9
def hannah_interval := 8
def ian_interval := 10

-- Statement to prove
theorem tutors_next_together_in_360_days :
  Nat.lcm (Nat.lcm evan_interval fiona_interval) (Nat.lcm george_interval (Nat.lcm hannah_interval ian_interval)) = 360 :=
by
  sorry

end tutors_next_together_in_360_days_l1998_199856


namespace calculate_expression_l1998_199874

theorem calculate_expression :
  3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 :=
by
  sorry

end calculate_expression_l1998_199874


namespace emily_final_score_l1998_199845

theorem emily_final_score :
  16 + 33 - 48 = 1 :=
by
  -- proof skipped
  sorry

end emily_final_score_l1998_199845


namespace fraction_subtraction_l1998_199815

theorem fraction_subtraction : (5 / 6 + 1 / 4 - 2 / 3) = (5 / 12) := by
  sorry

end fraction_subtraction_l1998_199815


namespace k_plus_alpha_is_one_l1998_199867

variable (f : ℝ → ℝ) (k α : ℝ)

-- Conditions from part a)
def power_function := ∀ x : ℝ, f x = k * x ^ α
def passes_through_point := f (1 / 2) = 2

-- Statement to be proven
theorem k_plus_alpha_is_one (h1 : power_function f k α) (h2 : passes_through_point f) : k + α = 1 :=
sorry

end k_plus_alpha_is_one_l1998_199867


namespace add_complex_eq_required_complex_addition_l1998_199833

theorem add_complex_eq (a b c d : ℝ) (i : ℂ) (h : i ^ 2 = -1) :
  (a + b * i) + (c + d * i) = (a + c) + (b + d) * i :=
by sorry

theorem required_complex_addition :
  let a : ℂ := 5 - 3 * i
  let b : ℂ := 2 + 12 * i
  a + b = 7 + 9 * i := 
by sorry

end add_complex_eq_required_complex_addition_l1998_199833


namespace smallest_c_is_52_l1998_199812

def seq (n : ℕ) : ℤ := -103 + (n:ℤ) * 2

theorem smallest_c_is_52 :
  ∃ c : ℕ, 
  (∀ n : ℕ, n < c → (∀ m : ℕ, m < n → seq m < 0) ∧ seq n = 0) ∧
  seq c > 0 ∧
  c = 52 :=
by
  sorry

end smallest_c_is_52_l1998_199812


namespace pizza_pieces_per_person_l1998_199807

theorem pizza_pieces_per_person (total_people : ℕ) (fraction_eat : ℚ) (total_pizza : ℕ) (remaining_pizza : ℕ)
  (H1 : total_people = 15) (H2 : fraction_eat = 3/5) (H3 : total_pizza = 50) (H4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (fraction_eat * total_people) = 4 :=
by
  -- proof goes here
  sorry

end pizza_pieces_per_person_l1998_199807


namespace train_crossing_time_l1998_199877

variable (length_train : ℝ) (time_pole : ℝ) (length_platform : ℝ) (time_platform : ℝ)

-- Given conditions
def train_conditions := 
  length_train = 300 ∧
  time_pole = 14 ∧
  length_platform = 535.7142857142857

-- Theorem statement
theorem train_crossing_time (h : train_conditions length_train time_pole length_platform) :
  time_platform = 39 := sorry

end train_crossing_time_l1998_199877


namespace least_prime_factor_five_power_difference_l1998_199825

theorem least_prime_factor_five_power_difference : 
  ∃ p : ℕ, (Nat.Prime p ∧ p ∣ (5^4 - 5^3)) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (5^4 - 5^3) → p ≤ q) := 
sorry

end least_prime_factor_five_power_difference_l1998_199825


namespace unicorn_journey_length_l1998_199841

theorem unicorn_journey_length (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_flowers : ℕ) (step_length_meters : ℕ) : (num_unicorns = 6) → (flowers_per_step = 4) → (total_flowers = 72000) → (step_length_meters = 3) → 
(total_flowers / flowers_per_step / num_unicorns * step_length_meters / 1000 = 9) :=
by
  intros h1 h2 h3 h4
  sorry

end unicorn_journey_length_l1998_199841


namespace eval_expression_l1998_199873

theorem eval_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 2) * Real.cos (Real.pi + 2)) = Real.sin 2 - Real.cos 2 :=
sorry

end eval_expression_l1998_199873


namespace interest_rate_second_share_l1998_199880

variable (T : ℝ) (r1 : ℝ) (I2 : ℝ) (T_i : ℝ)

theorem interest_rate_second_share 
  (h1 : T = 100000)
  (h2 : r1 = 0.09)
  (h3 : I2 = 24999.999999999996)
  (h4 : T_i = 0.095 * T) : 
  (2750 / I2) * 100 = 11 :=
by {
  sorry
}

end interest_rate_second_share_l1998_199880


namespace sequence_equality_l1998_199866

theorem sequence_equality (a : Fin 1973 → ℝ) (hpos : ∀ n, a n > 0)
  (heq : a 0 ^ a 0 = a 1 ^ a 2 ∧ a 1 ^ a 2 = a 2 ^ a 3 ∧ 
         a 2 ^ a 3 = a 3 ^ a 4 ∧ 
         -- etc., continued for all indices, 
         -- ensuring last index correctly refers back to a 0
         a 1971 ^ a 1972 = a 1972 ^ a 0) :
  a 0 = a 1972 :=
sorry

end sequence_equality_l1998_199866


namespace largest_y_coordinate_ellipse_l1998_199881

theorem largest_y_coordinate_ellipse (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 := 
by
  -- proof to be filled in
  sorry

end largest_y_coordinate_ellipse_l1998_199881


namespace johnny_age_multiple_l1998_199849

theorem johnny_age_multiple
  (current_age : ℕ)
  (age_in_2_years : ℕ)
  (age_3_years_ago : ℕ)
  (k : ℕ)
  (h1 : current_age = 8)
  (h2 : age_in_2_years = current_age + 2)
  (h3 : age_3_years_ago = current_age - 3)
  (h4 : age_in_2_years = k * age_3_years_ago) :
  k = 2 :=
by
  sorry

end johnny_age_multiple_l1998_199849


namespace remaining_lawn_area_l1998_199848

theorem remaining_lawn_area (lawn_length lawn_width path_width : ℕ) 
  (h_lawn_length : lawn_length = 10) 
  (h_lawn_width : lawn_width = 5) 
  (h_path_width : path_width = 1) : 
  (lawn_length * lawn_width - lawn_length * path_width) = 40 := 
by 
  sorry

end remaining_lawn_area_l1998_199848


namespace remainder_division_P_by_D_l1998_199826

def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15
def D (x : ℝ) := 4 * x - 8

theorem remainder_division_P_by_D :
  let remainder := P 2 % D 2
  remainder = 31 :=
by
  -- Proof will be inserted here, but currently skipped
  sorry

end remainder_division_P_by_D_l1998_199826


namespace number_of_moles_of_water_formed_l1998_199842

def balanced_combustion_equation : Prop :=
  ∀ (CH₄ O₂ CO₂ H₂O : ℕ), (CH₄ + 2 * O₂ = CO₂ + 2 * H₂O)

theorem number_of_moles_of_water_formed
  (CH₄_initial moles_of_CH₄ O₂_initial moles_of_O₂ : ℕ)
  (h_CH₄_initial : CH₄_initial = 3)
  (h_O₂_initial : O₂_initial = 6)
  (h_moles_of_H₂O : moles_of_CH₄ * 2 = 2 * moles_of_H₂O) :
  moles_of_H₂O = 6 :=
by
  sorry

end number_of_moles_of_water_formed_l1998_199842


namespace product_units_tens_not_divisible_by_5_l1998_199865

-- Define the list of four-digit numbers
def numbers : List ℕ := [4750, 4760, 4775, 4785, 4790]

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Statement: The product of the units digit and the tens digit of the number
-- that is not divisible by 5 in the list is 0
theorem product_units_tens_not_divisible_by_5 : 
  ∃ n ∈ numbers, ¬divisible_by_5 n ∧ (units_digit n * tens_digit n = 0) :=
by sorry

end product_units_tens_not_divisible_by_5_l1998_199865


namespace solve_equation_l1998_199808

theorem solve_equation : ∀ x : ℝ, -2 * x + 11 = 0 → x = 11 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l1998_199808


namespace range_of_a_l1998_199809

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := -(x + 1)^2 + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f x2 ≤ g x1 a) ↔ a ≥ -1 / Real.exp 1 :=
by
  -- proof would go here
  sorry

end range_of_a_l1998_199809


namespace distance_between_foci_of_hyperbola_l1998_199843

theorem distance_between_foci_of_hyperbola (a b c : ℝ) : (x^2 - y^2 = 4) → (a = 2) → (b = 0) → (c = Real.sqrt (4 + 0)) → 
    dist (2, 0) (-2, 0) = 4 :=
by
  sorry

end distance_between_foci_of_hyperbola_l1998_199843


namespace solve_for_x_l1998_199829

theorem solve_for_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : 
  x = 2 := 
by sorry

end solve_for_x_l1998_199829


namespace difference_in_speed_l1998_199854

theorem difference_in_speed (d : ℕ) (tA tE : ℕ) (vA vE : ℕ) (h1 : d = 300) (h2 : tA = tE - 3) 
    (h3 : vE = 20) (h4 : vE = d / tE) (h5 : vA = d / tA) : vA - vE = 5 := 
    sorry

end difference_in_speed_l1998_199854


namespace cubic_polynomial_root_sum_cube_value_l1998_199840

noncomputable def α : ℝ := (17 : ℝ)^(1 / 3)
noncomputable def β : ℝ := (67 : ℝ)^(1 / 3)
noncomputable def γ : ℝ := (137 : ℝ)^(1 / 3)

theorem cubic_polynomial_root_sum_cube_value
    (p q r : ℝ)
    (h1 : (p - α) * (p - β) * (p - γ) = 1)
    (h2 : (q - α) * (q - β) * (q - γ) = 1)
    (h3 : (r - α) * (r - β) * (r - γ) = 1) :
    p^3 + q^3 + r^3 = 218 := 
by
  sorry

end cubic_polynomial_root_sum_cube_value_l1998_199840


namespace monthly_cost_per_person_is_1000_l1998_199889

noncomputable def john_pays : ℝ := 32000
noncomputable def initial_fee_per_person : ℝ := 4000
noncomputable def total_people : ℝ := 4
noncomputable def john_pays_half : Prop := true

theorem monthly_cost_per_person_is_1000 :
  john_pays_half →
  (john_pays * 2 - (initial_fee_per_person * total_people)) / (total_people * 12) = 1000 :=
by
  intro h
  sorry

end monthly_cost_per_person_is_1000_l1998_199889


namespace max_number_of_different_ages_l1998_199870

theorem max_number_of_different_ages
  (a : ℤ) (s : ℤ)
  (h1 : a = 31)
  (h2 : s = 5) :
  ∃ n : ℕ, n = (36 - 26 + 1) :=
by sorry

end max_number_of_different_ages_l1998_199870


namespace inequality_check_l1998_199851

theorem inequality_check : (-1 : ℝ) / 3 < -1 / 5 := 
by 
  sorry

end inequality_check_l1998_199851


namespace metal_waste_l1998_199831

theorem metal_waste (l w : ℝ) (h : l > w) :
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  wasted_metal = l * w - w ^ 2 / 2 :=
by
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  sorry

end metal_waste_l1998_199831


namespace train_speed_l1998_199878

theorem train_speed (d t : ℝ) (h1 : d = 500) (h2 : t = 3) : d / t = 166.67 := by
  sorry

end train_speed_l1998_199878


namespace required_fencing_l1998_199859

-- Define constants given in the problem
def L : ℕ := 20
def A : ℕ := 720

-- Define the width W based on the area and the given length L
def W : ℕ := A / L

-- Define the total amount of fencing required
def F : ℕ := 2 * W + L

-- State the theorem that this amount of fencing is equal to 92
theorem required_fencing : F = 92 := by
  sorry

end required_fencing_l1998_199859


namespace correct_operation_l1998_199886

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l1998_199886


namespace find_f_zero_l1998_199803

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem find_f_zero (a b : ℝ)
  (h1 : f 3 a b = 7)
  (h2 : f 5 a b = -1) : f 0 a b = 19 :=
by
  sorry

end find_f_zero_l1998_199803


namespace ratio_of_x_and_y_l1998_199813

theorem ratio_of_x_and_y (x y : ℤ) (h : (3 * x - 2 * y) * 4 = 3 * (2 * x + y)) : (x : ℚ) / y = 11 / 6 :=
  sorry

end ratio_of_x_and_y_l1998_199813


namespace perfect_square_trinomial_k_l1998_199898

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l1998_199898


namespace solve_ineq_system_l1998_199868

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end solve_ineq_system_l1998_199868


namespace hyperbola_standard_equation_l1998_199802

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_standard_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (focus_distance_condition : ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * (c a b))
  (circle_intersects_asymptote : ∃ (x y : ℝ), (x, y) = (1, 2) ∧ y = (b/a) * x + 2): 
  (a = 1) ∧ (b = 2) → (x^2 - (y^2 / 4) = 1) := 
sorry

end hyperbola_standard_equation_l1998_199802


namespace P_sufficient_but_not_necessary_for_Q_l1998_199899

-- Definitions based on given conditions
def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

-- The theorem to prove that P is sufficient but not necessary for Q
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l1998_199899


namespace distance_and_speed_l1998_199820

-- Define the conditions given in the problem
def first_car_speed (y : ℕ) := y + 4
def second_car_speed (y : ℕ) := y
def third_car_speed (y : ℕ) := y - 6

def time_relation1 (x : ℕ) (y : ℕ) :=
  x / (first_car_speed y) = x / (second_car_speed y) - 3 / 60

def time_relation2 (x : ℕ) (y : ℕ) :=
  x / (second_car_speed y) = x / (third_car_speed y) - 5 / 60 

-- State the theorem to prove both the distance and the speed of the second car
theorem distance_and_speed : ∃ (x y : ℕ), 
  time_relation1 x y ∧ 
  time_relation2 x y ∧ 
  x = 120 ∧ 
  y = 96 :=
by
  sorry

end distance_and_speed_l1998_199820


namespace like_terms_exponents_l1998_199888

theorem like_terms_exponents (m n : ℤ) (h1 : 2 * n - 1 = m) (h2 : m = 3) : m = 3 ∧ n = 2 :=
by
  sorry

end like_terms_exponents_l1998_199888


namespace probability_fewer_heads_than_tails_is_793_over_2048_l1998_199894

noncomputable def probability_fewer_heads_than_tails (n : ℕ) : ℝ :=
(793 / 2048 : ℚ)

theorem probability_fewer_heads_than_tails_is_793_over_2048 :
  probability_fewer_heads_than_tails 12 = (793 / 2048 : ℚ) :=
sorry

end probability_fewer_heads_than_tails_is_793_over_2048_l1998_199894


namespace fractional_equation_solution_l1998_199885

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l1998_199885


namespace apple_cost_price_orange_cost_price_banana_cost_price_l1998_199835

theorem apple_cost_price (A : ℚ) : 15 = A - (1/6 * A) → A = 18 := by
  intro h
  sorry

theorem orange_cost_price (O : ℚ) : 20 = O + (1/5 * O) → O = 100/6 := by
  intro h
  sorry

theorem banana_cost_price (B : ℚ) : 10 = B → B = 10 := by
  intro h
  sorry

end apple_cost_price_orange_cost_price_banana_cost_price_l1998_199835


namespace total_pots_needed_l1998_199819

theorem total_pots_needed
    (p : ℕ) (s : ℕ) (h : ℕ)
    (hp : p = 5)
    (hs : s = 3)
    (hh : h = 4) :
    p * s * h = 60 := by
  sorry

end total_pots_needed_l1998_199819


namespace hyperbola_eccentricity_l1998_199858

theorem hyperbola_eccentricity (m : ℝ) (h1: ∃ x y : ℝ, (x^2 / 3) - (y^2 / m) = 1) (h2: ∀ a b : ℝ, a^2 = 3 ∧ b^2 = m ∧ (2 = Real.sqrt (1 + b^2 / a^2))) : m = -9 := 
sorry

end hyperbola_eccentricity_l1998_199858


namespace books_arrangement_l1998_199832

/-
  Theorem:
  If there are 4 distinct math books, 6 distinct English books, and 3 distinct science books,
  and each category of books must stay together, then the number of ways to arrange
  these books on a shelf is 622080.
-/

def num_math_books : ℕ := 4
def num_english_books : ℕ := 6
def num_science_books : ℕ := 3

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_arrangements :=
  factorial 3 * factorial num_math_books * factorial num_english_books * factorial num_science_books

theorem books_arrangement : num_arrangements = 622080 := by
  sorry

end books_arrangement_l1998_199832


namespace no_real_solution_l1998_199861

noncomputable def quadratic_eq (x : ℝ) : ℝ := (2*x^2 - 3*x + 5)

theorem no_real_solution : 
  ∀ x : ℝ, quadratic_eq x ^ 2 + 1 ≠ 1 :=
by
  intro x
  sorry

end no_real_solution_l1998_199861


namespace total_players_on_ground_l1998_199879

theorem total_players_on_ground :
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  cricket_players + hockey_players + football_players + softball_players +
  basketball_players + volleyball_players + netball_players + rugby_players = 263 := 
by 
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  sorry

end total_players_on_ground_l1998_199879


namespace jake_sausages_cost_l1998_199890

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end jake_sausages_cost_l1998_199890


namespace elapsed_time_l1998_199827

theorem elapsed_time (x : ℕ) (h1 : 99 > 0) (h2 : (2 : ℚ) / (3 : ℚ) * x = (4 : ℚ) / (5 : ℚ) * (99 - x)) : x = 54 := by
  sorry

end elapsed_time_l1998_199827


namespace arithmetic_sequence_a4_l1998_199822

/-- Given an arithmetic sequence {a_n}, where S₁₀ = 60 and a₇ = 7, prove that a₄ = 5. -/
theorem arithmetic_sequence_a4 (a₁ d : ℝ) 
  (h1 : 10 * a₁ + 45 * d = 60) 
  (h2 : a₁ + 6 * d = 7) : 
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a4_l1998_199822


namespace solve_inequality_l1998_199882

theorem solve_inequality :
  {x : ℝ | x^2 - 9 * x + 14 < 0} = {x : ℝ | 2 < x ∧ x < 7} := sorry

end solve_inequality_l1998_199882


namespace lowest_two_digit_number_whose_digits_product_is_12_l1998_199828

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n < 100 ∧ ∃ d1 d2 : ℕ, 1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ n = 10 * d1 + d2 ∧ d1 * d2 = 12

theorem lowest_two_digit_number_whose_digits_product_is_12 :
  ∃ n : ℕ, is_valid_two_digit_number n ∧ ∀ m : ℕ, is_valid_two_digit_number m → n ≤ m ∧ n = 26 :=
sorry

end lowest_two_digit_number_whose_digits_product_is_12_l1998_199828


namespace ratio_of_areas_l1998_199816

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l1998_199816


namespace orange_juice_fraction_l1998_199830

def capacity_small_pitcher := 500 -- mL
def orange_juice_fraction_small := 1 / 4
def capacity_large_pitcher := 800 -- mL
def orange_juice_fraction_large := 1 / 2

def total_orange_juice_volume := 
  (capacity_small_pitcher * orange_juice_fraction_small) + 
  (capacity_large_pitcher * orange_juice_fraction_large)
def total_volume := capacity_small_pitcher + capacity_large_pitcher

theorem orange_juice_fraction :
  (total_orange_juice_volume / total_volume) = (21 / 52) := 
by 
  sorry

end orange_juice_fraction_l1998_199830


namespace find_smallest_N_l1998_199897

-- Define the sum of digits functions as described
def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

-- Define f(n) which is the sum of digits in base-five representation of n
def f (n : ℕ) : ℕ :=
  sum_of_digits_base n 5

-- Define g(n) which is the sum of digits in base-seven representation of f(n)
def g (n : ℕ) : ℕ :=
  sum_of_digits_base (f n) 7

-- The statement of the problem: find the smallest N such that 
-- g(N) in base-sixteen cannot be represented using only digits 0 to 9
theorem find_smallest_N : ∃ N : ℕ, (g N ≥ 10) ∧ (N % 1000 = 610) :=
by
  sorry

end find_smallest_N_l1998_199897


namespace anthony_pencils_total_l1998_199811

def pencils_initial : Nat := 9
def pencils_kathryn : Nat := 56
def pencils_greg : Nat := 84
def pencils_maria : Nat := 138

theorem anthony_pencils_total : 
  pencils_initial + pencils_kathryn + pencils_greg + pencils_maria = 287 := 
by
  sorry

end anthony_pencils_total_l1998_199811


namespace max_b_c_l1998_199869

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end max_b_c_l1998_199869


namespace find_n_l1998_199884

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end find_n_l1998_199884


namespace relationship_of_y_values_l1998_199801

theorem relationship_of_y_values 
  (k : ℝ) (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_pos : k > 0) 
  (hA : y1 = k / x1) 
  (hB : y2 = k / x2) 
  (hC : y3 = k / x3) 
  (h_order : x1 < 0 ∧ 0 < x2 ∧ x2 < x3) : y1 < y3 ∧ y3 < y2 := 
by
  sorry

end relationship_of_y_values_l1998_199801


namespace find_x_for_g_l1998_199800

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5)/6))^(1/3)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l1998_199800


namespace min_fencing_cost_l1998_199864

theorem min_fencing_cost {A B C : ℕ} (h1 : A = 25) (h2 : B = 35) (h3 : C = 40)
  (h_ratio : ∃ (x : ℕ), 3 * x * 4 * x = 8748) : 
  ∃ (total_cost : ℝ), total_cost = 87.75 :=
by
  sorry

end min_fencing_cost_l1998_199864


namespace problem_statement_l1998_199855

def a : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (2, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement : dot_product (vector_sub a b) a = 0 := 
by 
  -- The proof would go here
  sorry

end problem_statement_l1998_199855


namespace curtain_price_l1998_199810

theorem curtain_price
  (C : ℝ)
  (h1 : 2 * C + 9 * 15 + 50 = 245) :
  C = 30 :=
sorry

end curtain_price_l1998_199810


namespace monica_usd_start_amount_l1998_199863

theorem monica_usd_start_amount (x : ℕ) (H : ∃ (y : ℕ), y = 40 ∧ (8 : ℚ) / 5 * x - y = x) :
  (x / 100) + (x % 100 / 10) + (x % 10) = 2 := 
by
  sorry

end monica_usd_start_amount_l1998_199863


namespace chord_length_cube_l1998_199836

noncomputable def diameter : ℝ := 1
noncomputable def AC (a : ℝ) : ℝ := a
noncomputable def AD (b : ℝ) : ℝ := b
noncomputable def AE (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / 2
noncomputable def AF (b : ℝ) : ℝ := b^2

theorem chord_length_cube (a b : ℝ) (h : AE a b = b^2) : a = b^3 :=
by
  sorry

end chord_length_cube_l1998_199836


namespace waste_scientific_notation_correct_l1998_199804

def total_waste_in_scientific : ℕ := 500000000000

theorem waste_scientific_notation_correct :
  total_waste_in_scientific = 5 * 10^10 :=
by
  sorry

end waste_scientific_notation_correct_l1998_199804


namespace grasshoppers_total_l1998_199872

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l1998_199872


namespace simplify_fraction_l1998_199862

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by
  sorry

end simplify_fraction_l1998_199862


namespace sum_of_products_eq_131_l1998_199857

theorem sum_of_products_eq_131 (a b c : ℝ) 
    (h1 : a^2 + b^2 + c^2 = 222)
    (h2 : a + b + c = 22) : 
    a * b + b * c + c * a = 131 :=
by
  sorry

end sum_of_products_eq_131_l1998_199857


namespace solve_for_x_l1998_199817

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 :=
by {
  sorry
}

end solve_for_x_l1998_199817


namespace Jake_has_8_peaches_l1998_199823

variable (Jake Steven Jill : ℕ)

theorem Jake_has_8_peaches
  (h_steven_peaches : Steven = 15)
  (h_steven_jill : Steven = Jill + 14)
  (h_jake_steven : Jake = Steven - 7) :
  Jake = 8 := by
  sorry

end Jake_has_8_peaches_l1998_199823


namespace prove_A_annual_savings_l1998_199805

noncomputable def employee_A_annual_savings
  (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02) : ℝ :=
  let total_deductions := tax_rate + pension_rate + healthcare_rate
  let Income_after_deductions := A_income * (1 - total_deductions)
  let annual_savings := 12 * Income_after_deductions
  annual_savings

theorem prove_A_annual_savings : 
  ∀ (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02),
  employee_A_annual_savings A_income B_income C_income D_income C_income_val income_ratio tax_rate pension_rate healthcare_rate tax_rate_val pension_rate_val healthcare_rate_val = 232400.16 :=
by
  sorry

end prove_A_annual_savings_l1998_199805


namespace interval_of_monotonic_increase_l1998_199844

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase : {x : ℝ | -1 ≤ x} ⊆ {x : ℝ | 0 < deriv f x} :=
by
  sorry

end interval_of_monotonic_increase_l1998_199844


namespace sin_gt_sub_cubed_l1998_199875

theorem sin_gt_sub_cubed (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  Real.sin x > x - x^3 / 6 := 
by 
  sorry

end sin_gt_sub_cubed_l1998_199875


namespace thirteenth_term_is_correct_l1998_199892

noncomputable def third_term : ℚ := 2 / 11
noncomputable def twenty_third_term : ℚ := 3 / 7

theorem thirteenth_term_is_correct : 
  (third_term + twenty_third_term) / 2 = 47 / 154 := sorry

end thirteenth_term_is_correct_l1998_199892


namespace contrapositive_statement_l1998_199860

-- Definitions derived from conditions
def Triangle (ABC : Type) : Prop := 
  ∃ a b c : ABC, true

def IsIsosceles (ABC : Type) : Prop :=
  ∃ a b c : ABC, a = b ∨ b = c ∨ a = c

def InteriorAnglesNotEqual (ABC : Type) : Prop :=
  ∀ a b : ABC, a ≠ b

-- The contrapositive implication we need to prove
theorem contrapositive_statement (ABC : Type) (h : Triangle ABC) 
  (h_not_isosceles_implies_not_equal : ¬IsIsosceles ABC → InteriorAnglesNotEqual ABC) :
  (∃ a b c : ABC, a = b → IsIsosceles ABC) := 
sorry

end contrapositive_statement_l1998_199860


namespace value_of_m_minus_n_l1998_199895

theorem value_of_m_minus_n (m n : ℝ) (h : (-3)^2 + m * (-3) + 3 * n = 0) : m - n = 3 :=
sorry

end value_of_m_minus_n_l1998_199895


namespace compute_j_in_polynomial_arithmetic_progression_l1998_199838

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end compute_j_in_polynomial_arithmetic_progression_l1998_199838


namespace exists_triangle_sides_l1998_199883

theorem exists_triangle_sides (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c ≤ 1 / 4)
  (h2 : 1 / (a^2) + 1 / (b^2) + 1 / (c^2) < 9) : 
  a + b > c ∧ b + c > a ∧ c + a > b := 
by
  sorry

end exists_triangle_sides_l1998_199883


namespace person_died_at_33_l1998_199887

-- Define the conditions and constants
def start_age : ℕ := 25
def insurance_payment : ℕ := 10000
def premium : ℕ := 450
def loss : ℕ := 1000
def annual_interest_rate : ℝ := 0.05
def half_year_factor : ℝ := 1.025 -- half-yearly compounded interest factor

-- Calculate the number of premium periods (as an integer)
def n := 16 -- (derived from the calculations in the given solution)

-- Define the final age based on the number of premium periods
def final_age : ℕ := start_age + (n / 2)

-- The proof statement
theorem person_died_at_33 : final_age = 33 := by
  sorry

end person_died_at_33_l1998_199887


namespace slower_speed_for_on_time_arrival_l1998_199893

variable (distance : ℝ) (actual_speed : ℝ) (time_early : ℝ)

theorem slower_speed_for_on_time_arrival 
(h1 : distance = 20)
(h2 : actual_speed = 40)
(h3 : time_early = 1 / 15) :
  actual_speed - (600 / 17) = 4.71 :=
by 
  sorry

end slower_speed_for_on_time_arrival_l1998_199893


namespace range_of_a_l1998_199818

variable (a x : ℝ)

-- Condition p: ∀ x ∈ [1, 2], x^2 - a ≥ 0
def p : Prop := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition q: ∃ x ∈ ℝ, x^2 + 2 * a * x + 2 - a = 0
def q : Prop := ∃ x, x^2 + 2 * a * x + 2 - a = 0

-- The proof goal given p ∧ q: a ≤ -2 or a = 1
theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l1998_199818


namespace quadratic_min_value_max_l1998_199821

theorem quadratic_min_value_max (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 - 4 * a * c ≥ 0) :
    (min (min ((b + c) / a) ((c + a) / b)) ((a + b) / c)) ≤ (5 / 4) :=
sorry

end quadratic_min_value_max_l1998_199821


namespace sheilas_family_contribution_l1998_199871

theorem sheilas_family_contribution :
  let initial_amount := 3000
  let monthly_savings := 276
  let duration_years := 4
  let total_after_duration := 23248
  let months_in_year := 12
  let total_months := duration_years * months_in_year
  let savings_over_duration := monthly_savings * total_months
  let sheilas_total_savings := initial_amount + savings_over_duration
  let family_contribution := total_after_duration - sheilas_total_savings
  family_contribution = 7000 :=
by
  sorry

end sheilas_family_contribution_l1998_199871


namespace problem_statement_l1998_199852

variables {A B C O D : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (a b c o d : A)

-- Define the geometric conditions
axiom condition1 : a + 2 • b + 3 • c = 0
axiom condition2 : ∃ (D: A), (∃ (k : ℝ), a = k • d ∧ k ≠ 0) ∧ (∃ (u v : ℝ),  u • b + v • c = d ∧ u + v = 1)

-- Define points
def OA : A := a - o
def OB : A := b - o
def OC : A := c - o
def OD : A := d - o

-- The main statement to prove
theorem problem_statement : 2 • (b - d) + 3 • (c - d) = (0 : A) :=
by
  sorry

end problem_statement_l1998_199852


namespace sum_of_integer_pair_l1998_199850

theorem sum_of_integer_pair (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 10) (h3 : 1 ≤ b) (h4 : b ≤ 10) (h5 : a * b = 14) : a + b = 9 := 
sorry

end sum_of_integer_pair_l1998_199850


namespace taxi_fare_relationship_taxi_fare_relationship_simplified_l1998_199891

variable (x : ℝ) (y : ℝ)

-- Conditions
def starting_fare : ℝ := 14
def additional_fare_per_km : ℝ := 2.4
def initial_distance : ℝ := 3
def total_distance (x : ℝ) := x
def total_fare (x : ℝ) (y : ℝ) := y
def distance_condition (x : ℝ) := x > 3

-- Theorem Statement
theorem taxi_fare_relationship (h : distance_condition x) :
  total_fare x y = additional_fare_per_km * (total_distance x - initial_distance) + starting_fare :=
by
  sorry

-- Simplified Theorem Statement
theorem taxi_fare_relationship_simplified (h : distance_condition x) :
  y = 2.4 * x + 6.8 :=
by
  sorry

end taxi_fare_relationship_taxi_fare_relationship_simplified_l1998_199891


namespace tan_product_in_triangle_l1998_199896

theorem tan_product_in_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = Real.sin B ^ 2) :
  Real.tan A * Real.tan C = 1 :=
sorry

end tan_product_in_triangle_l1998_199896


namespace average_ABC_eq_2A_plus_3_l1998_199876

theorem average_ABC_eq_2A_plus_3 (A B C : ℝ) 
  (h1 : 2023 * C - 4046 * A = 8092) 
  (h2 : 2023 * B - 6069 * A = 10115) : 
  (A + B + C) / 3 = 2 * A + 3 :=
sorry

end average_ABC_eq_2A_plus_3_l1998_199876


namespace sum_of_ages_l1998_199806

theorem sum_of_ages (Petra_age : ℕ) (Mother_age : ℕ)
  (h_petra : Petra_age = 11)
  (h_mother : Mother_age = 36) :
  Petra_age + Mother_age = 47 :=
by
  -- Using the given conditions:
  -- Petra_age = 11
  -- Mother_age = 36
  sorry

end sum_of_ages_l1998_199806


namespace impossible_event_D_l1998_199847

-- Event definitions
def event_A : Prop := true -- This event is not impossible
def event_B : Prop := true -- This event is not impossible
def event_C : Prop := true -- This event is not impossible
def event_D (bag : Finset String) : Prop :=
  if "red" ∈ bag then false else true -- This event is impossible if there are no red balls

-- Bag condition
def bag : Finset String := {"white", "white", "white", "white", "white", "white", "white", "white"}

-- Proof statement
theorem impossible_event_D : event_D bag = true :=
by
  -- The bag contains only white balls, so drawing a red ball is impossible.
  rw [event_D, if_neg]
  sorry

end impossible_event_D_l1998_199847


namespace dresser_clothing_capacity_l1998_199839

theorem dresser_clothing_capacity (pieces_per_drawer : ℕ) (number_of_drawers : ℕ) (total_pieces : ℕ) 
  (h1 : pieces_per_drawer = 5)
  (h2 : number_of_drawers = 8)
  (h3 : total_pieces = 40) :
  pieces_per_drawer * number_of_drawers = total_pieces :=
by {
  sorry
}

end dresser_clothing_capacity_l1998_199839


namespace unique_solution_system_eqns_l1998_199824

theorem unique_solution_system_eqns :
  ∃ (x y : ℝ), (2 * x - 3 * |y| = 1 ∧ |x| + 2 * y = 4 ∧ x = 2 ∧ y = 1) :=
sorry

end unique_solution_system_eqns_l1998_199824


namespace not_kth_power_l1998_199853

theorem not_kth_power (m k : ℕ) (hk : k > 1) : ¬ ∃ a : ℤ, m * (m + 1) = a^k :=
by
  sorry

end not_kth_power_l1998_199853


namespace total_rainbow_nerds_is_36_l1998_199814

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l1998_199814


namespace smallest_nat_number_l1998_199834

theorem smallest_nat_number (n : ℕ) (h1 : ∃ a, 0 ≤ a ∧ a < 20 ∧ n % 20 = a ∧ n % 21 = a + 1) (h2 : n % 22 = 2) : n = 838 := by 
  sorry

end smallest_nat_number_l1998_199834


namespace find_sum_x1_x2_l1998_199837

-- Define sets A and B with given properties
def set_A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def set_B (x1 x2 : ℝ) : Set ℝ := {x | x1 ≤ x ∧ x ≤ x2}

-- Conditions of union and intersection
def union_condition (x1 x2 : ℝ) : Prop := set_A ∪ set_B x1 x2 = {x | x > -2}
def intersection_condition (x1 x2 : ℝ) : Prop := set_A ∩ set_B x1 x2 = {x | 1 < x ∧ x ≤ 3}

-- Main theorem to prove
theorem find_sum_x1_x2 (x1 x2 : ℝ) (h_union : union_condition x1 x2) (h_intersect : intersection_condition x1 x2) :
  x1 + x2 = 2 :=
sorry

end find_sum_x1_x2_l1998_199837


namespace omega_range_l1998_199846

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l1998_199846
