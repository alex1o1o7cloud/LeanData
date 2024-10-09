import Mathlib

namespace coefficient_x7_in_expansion_l1688_168821

theorem coefficient_x7_in_expansion : 
  let n := 10
  let k := 7
  let binom := Nat.choose n k
  let coeff := 1
  coeff * binom = 120 :=
by
  sorry

end coefficient_x7_in_expansion_l1688_168821


namespace base7_addition_l1688_168818

theorem base7_addition (X Y : ℕ) (h1 : Y + 2 = X) (h2 : X + 5 = 8) : X + Y = 4 :=
by
  sorry

end base7_addition_l1688_168818


namespace ratio_twice_width_to_length_l1688_168840

-- Given conditions:
def length_of_field : ℚ := 24
def width_of_field : ℚ := 13.5

-- The problem is to prove the ratio of twice the width to the length of the field is 9/8
theorem ratio_twice_width_to_length : 2 * width_of_field / length_of_field = 9 / 8 :=
by sorry

end ratio_twice_width_to_length_l1688_168840


namespace math_problem_l1688_168891

theorem math_problem :
  101 * 102^2 - 101 * 98^2 = 80800 :=
by
  sorry

end math_problem_l1688_168891


namespace norma_total_cards_l1688_168850

theorem norma_total_cards (initial_cards : ℝ) (additional_cards : ℝ) (total_cards : ℝ) 
  (h1 : initial_cards = 88) (h2 : additional_cards = 70) : total_cards = 158 :=
by
  sorry

end norma_total_cards_l1688_168850


namespace sum_of_roots_l1688_168824

-- States that the sum of the values of x that satisfy the given quadratic equation is 7
theorem sum_of_roots (x : ℝ) :
  (x^2 - 7 * x + 12 = 4) → (∃ a b : ℝ, x^2 - 7 * x + 8 = 0 ∧ a + b = 7) :=
by
  sorry

end sum_of_roots_l1688_168824


namespace is_quadratic_l1688_168885

theorem is_quadratic (A B C D : Prop) :
  (A = (∀ x : ℝ, x + (1 / x) = 0)) ∧
  (B = (∀ x y : ℝ, x + x * y + 1 = 0)) ∧
  (C = (∀ x : ℝ, 3 * x + 2 = 0)) ∧
  (D = (∀ x : ℝ, x^2 + 2 * x = 1)) →
  D := 
by
  sorry

end is_quadratic_l1688_168885


namespace units_digit_7_pow_6_l1688_168889

theorem units_digit_7_pow_6 : (7 ^ 6) % 10 = 9 := by
  sorry

end units_digit_7_pow_6_l1688_168889


namespace trigonometric_identity_l1688_168809

theorem trigonometric_identity
  (α : ℝ) 
  (h : Real.tan α = -1 / 2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := 
by 
  sorry

end trigonometric_identity_l1688_168809


namespace nursing_home_milk_l1688_168856

theorem nursing_home_milk :
  ∃ x y : ℕ, (2 * x + 16 = y) ∧ (4 * x - 12 = y) ∧ (x = 14) ∧ (y = 44) :=
by
  sorry

end nursing_home_milk_l1688_168856


namespace blocks_per_box_l1688_168868

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end blocks_per_box_l1688_168868


namespace find_p_q_l1688_168892

theorem find_p_q (D : ℝ) (p q : ℝ) (h_roots : ∀ x, x^2 + p * x + q = 0 → (x = D ∨ x = 1 - D))
  (h_discriminant : D = p^2 - 4 * q) :
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3 / 16) :=
by
  sorry

end find_p_q_l1688_168892


namespace prop_2_prop_3_l1688_168899

variables {a b c : ℝ}

-- Proposition 2: a > |b| -> a^2 > b^2
theorem prop_2 (h : a > |b|) : a^2 > b^2 := sorry

-- Proposition 3: a > b -> a^3 > b^3
theorem prop_3 (h : a > b) : a^3 > b^3 := sorry

end prop_2_prop_3_l1688_168899


namespace subtraction_like_terms_l1688_168867

variable (a : ℝ)

theorem subtraction_like_terms : 3 * a ^ 2 - 2 * a ^ 2 = a ^ 2 :=
by
  sorry

end subtraction_like_terms_l1688_168867


namespace sin_13pi_over_4_eq_neg_sqrt2_over_2_l1688_168835

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end sin_13pi_over_4_eq_neg_sqrt2_over_2_l1688_168835


namespace simplify_polynomial_l1688_168879

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 :=
by {
  sorry
}

end simplify_polynomial_l1688_168879


namespace cos_alpha_plus_pi_over_3_l1688_168826

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (α + π / 3) = -1 / 3 :=
  sorry

end cos_alpha_plus_pi_over_3_l1688_168826


namespace rolls_sold_to_uncle_l1688_168860

theorem rolls_sold_to_uncle (total_rolls needed_rolls rolls_to_grandmother rolls_to_neighbor rolls_to_uncle : ℕ)
  (h1 : total_rolls = 45)
  (h2 : needed_rolls = 28)
  (h3 : rolls_to_grandmother = 1)
  (h4 : rolls_to_neighbor = 6)
  (h5 : rolls_to_uncle + rolls_to_grandmother + rolls_to_neighbor + needed_rolls = total_rolls) :
  rolls_to_uncle = 10 :=
by {
  sorry
}

end rolls_sold_to_uncle_l1688_168860


namespace man_gets_dividend_l1688_168895

    -- Definitions based on conditions
    noncomputable def investment : ℝ := 14400
    noncomputable def premium_rate : ℝ := 0.20
    noncomputable def face_value : ℝ := 100
    noncomputable def dividend_rate : ℝ := 0.07

    -- Calculate the price per share with premium
    noncomputable def price_per_share : ℝ := face_value * (1 + premium_rate)

    -- Calculate the number of shares bought
    noncomputable def number_of_shares : ℝ := investment / price_per_share

    -- Calculate the dividend per share
    noncomputable def dividend_per_share : ℝ := face_value * dividend_rate

    -- Calculate the total dividend
    noncomputable def total_dividend : ℝ := dividend_per_share * number_of_shares

    -- The proof statement
    theorem man_gets_dividend : total_dividend = 840 := by
        sorry
    
end man_gets_dividend_l1688_168895


namespace total_amount_spent_l1688_168890

-- Define the prices related to John's Star Wars toy collection
def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

-- Problem statement in Lean: Prove the total amount spent is $3000
theorem total_amount_spent : (other_toys_cost + lightsaber_cost) = 3000 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end total_amount_spent_l1688_168890


namespace total_amount_divided_into_two_parts_l1688_168873

theorem total_amount_divided_into_two_parts (P1 P2 : ℝ) (annual_income : ℝ) :
  P1 = 1500.0000000000007 →
  annual_income = 135 →
  (P1 * 0.05 + P2 * 0.06 = annual_income) →
  P1 + P2 = 2500.000000000000 :=
by
  intros hP1 hIncome hInterest
  sorry

end total_amount_divided_into_two_parts_l1688_168873


namespace intersection_lines_k_l1688_168828

theorem intersection_lines_k (k : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -2 :=
by
  sorry

end intersection_lines_k_l1688_168828


namespace planks_needed_l1688_168839

theorem planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (h1 : total_nails = 4) (h2 : nails_per_plank = 2) : total_nails / nails_per_plank = 2 :=
by
  -- Prove that given the conditions, the required result is obtained
  sorry

end planks_needed_l1688_168839


namespace number_of_intersections_l1688_168813

-- Definitions of the given curves.
def curve1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

-- Statement of the theorem
theorem number_of_intersections : ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2 := sorry

end number_of_intersections_l1688_168813


namespace area_of_region_l1688_168877

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 1) → (∃ (A : ℝ), A = 14 * Real.pi) := 
by
  sorry

end area_of_region_l1688_168877


namespace sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l1688_168886

-- Problem 1: Prove the general formula for the sequence of all positive even numbers
theorem sequence_even_numbers (n : ℕ) : ∃ a_n, a_n = 2 * n := by 
  sorry

-- Problem 2: Prove the general formula for the sequence of all positive odd numbers
theorem sequence_odd_numbers (n : ℕ) : ∃ b_n, b_n = 2 * n - 1 := by 
  sorry

-- Problem 3: Prove the general formula for the sequence 1, 4, 9, 16, ...
theorem sequence_square_numbers (n : ℕ) : ∃ a_n, a_n = n^2 := by
  sorry

-- Problem 4: Prove the general formula for the sequence -4, -1, 2, 5, ...
theorem sequence_arithmetic_progression (n : ℕ) : ∃ a_n, a_n = 3 * n - 7 := by
  sorry

end sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l1688_168886


namespace smaller_integer_l1688_168870

noncomputable def m : ℕ := 1
noncomputable def n : ℕ := 1998 * m

lemma two_digit_number (m: ℕ) : 10 ≤ m ∧ m < 100 := by sorry
lemma three_digit_number (n: ℕ) : 100 ≤ n ∧ n < 1000 := by sorry

theorem smaller_integer 
  (two_digit_m: 10 ≤ m ∧ m < 100)
  (three_digit_n: 100 ≤ n ∧ n < 1000)
  (avg_eq_decimal: (m + n) / 2 = m + n / 1000)
  : m = 1 := by 
  sorry

end smaller_integer_l1688_168870


namespace problem1_problem2_l1688_168836

-- Problem 1: Proving the equation
theorem problem1 (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 → x = 1 :=
sorry

-- Problem 2: Proving the solution for the system of equations
theorem problem2 (x y : ℝ) : (x + 2 * y = 8) ∧ (3 * x - 4 * y = 4) → x = 4 ∧ y = 2 :=
sorry

end problem1_problem2_l1688_168836


namespace compute_nested_f_l1688_168881

def f(x : ℤ) : ℤ := x^2 - 4 * x + 3

theorem compute_nested_f : f (f (f (f (f (f 2))))) = f 1179395 := 
  sorry

end compute_nested_f_l1688_168881


namespace additional_stars_needed_l1688_168842

-- Defining the number of stars required per bottle
def stars_per_bottle : Nat := 85

-- Defining the number of bottles Luke needs to fill
def bottles_to_fill : Nat := 4

-- Defining the number of stars Luke has already made
def stars_made : Nat := 33

-- Calculating the number of stars Luke still needs to make
theorem additional_stars_needed : (stars_per_bottle * bottles_to_fill - stars_made) = 307 := by
  sorry  -- Proof to be provided

end additional_stars_needed_l1688_168842


namespace problem_I_problem_II_problem_III_l1688_168857

variables {pA pB : ℝ}

-- Given conditions
def probability_A : ℝ := 0.7
def probability_B : ℝ := 0.6

-- Questions reformulated as proof goals
theorem problem_I : 
  sorry := 
 sorry

theorem problem_II : 
  -- Find: Probability that at least one of A or B succeeds on the first attempt
  sorry := 
 sorry

theorem problem_III : 
  -- Find: Probability that A succeeds exactly one more time than B in two attempts each
  sorry := 
 sorry

end problem_I_problem_II_problem_III_l1688_168857


namespace find_n_l1688_168863

def exp (m n : ℕ) : ℕ := m ^ n

-- Now we restate the problem formally
theorem find_n 
  (m n : ℕ) 
  (h1 : exp 10 m = n * 22) : 
  n = 10^m / 22 := 
sorry

end find_n_l1688_168863


namespace trees_planted_l1688_168848

-- Definitions for the quantities of lindens (x) and birches (y)
variables (x y : ℕ)

-- Definitions matching the given problem conditions
def condition1 := x + y > 14
def condition2 := y + 18 > 2 * x
def condition3 := x > 2 * y

-- The theorem stating that if the conditions hold, then x = 11 and y = 5
theorem trees_planted (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : 
  x = 11 ∧ y = 5 := 
sorry

end trees_planted_l1688_168848


namespace min_value_fraction_l1688_168805

theorem min_value_fraction (x : ℝ) (h : x > 9) : (x^2 + 81) / (x - 9) ≥ 27 := 
  sorry

end min_value_fraction_l1688_168805


namespace find_function_and_max_profit_l1688_168853

noncomputable def profit_function (x : ℝ) : ℝ := -50 * x^2 + 1200 * x - 6400

theorem find_function_and_max_profit :
  (∀ (x : ℝ), (x = 10 → (-50 * x + 800 = 300)) ∧ (x = 13 → (-50 * x + 800 = 150))) ∧
  (∃ (x : ℝ), x = 12 ∧ profit_function x = 800) :=
by
  sorry

end find_function_and_max_profit_l1688_168853


namespace difference_in_floors_l1688_168832

-- Given conditions
variable (FA FB FC : ℕ)
variable (h1 : FA = 4)
variable (h2 : FC = 5 * FB - 6)
variable (h3 : FC = 59)

-- The statement to prove
theorem difference_in_floors : FB - FA = 9 :=
by 
  -- Placeholder proof
  sorry

end difference_in_floors_l1688_168832


namespace number_of_newspapers_l1688_168883

theorem number_of_newspapers (total_reading_materials magazines_sold: ℕ) (h_total: total_reading_materials = 700) (h_magazines: magazines_sold = 425) : 
  ∃ newspapers_sold : ℕ, newspapers_sold + magazines_sold = total_reading_materials ∧ newspapers_sold = 275 :=
by
  sorry

end number_of_newspapers_l1688_168883


namespace product_of_numbers_larger_than_reciprocal_eq_neg_one_l1688_168833

theorem product_of_numbers_larger_than_reciprocal_eq_neg_one :
  ∃ x y : ℝ, x ≠ y ∧ (x = 1 / x + 2) ∧ (y = 1 / y + 2) ∧ x * y = -1 :=
by
  sorry

end product_of_numbers_larger_than_reciprocal_eq_neg_one_l1688_168833


namespace peter_twice_as_old_in_years_l1688_168810

def mother_age : ℕ := 60
def harriet_current_age : ℕ := 13
def peter_current_age : ℕ := mother_age / 2
def years_later : ℕ := 4

theorem peter_twice_as_old_in_years : 
  peter_current_age + years_later = 2 * (harriet_current_age + years_later) :=
by
  -- using given conditions 
  -- Peter's current age is 30
  -- Harriet's current age is 13
  -- years_later is 4
  sorry

end peter_twice_as_old_in_years_l1688_168810


namespace find_number_of_even_numbers_l1688_168865

-- Define the average of the first n even numbers
def average_of_first_n_even (n : ℕ) : ℕ :=
  (n * (1 + n)) / n

-- The given condition: The average is 21
def average_is_21 (n : ℕ) : Prop :=
  average_of_first_n_even n = 21

-- The theorem to prove: If the average is 21, then n = 20
theorem find_number_of_even_numbers (n : ℕ) (h : average_is_21 n) : n = 20 :=
  sorry

end find_number_of_even_numbers_l1688_168865


namespace general_formula_a_n_sum_first_n_b_l1688_168803

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Sequence property
def seq_property (n : ℕ) (S_n : ℕ) : Prop :=
  a_n n ^ 2 + 2 * a_n n = 4 * S_n + 3

-- General formula for {a_n}
theorem general_formula_a_n (n : ℕ) (hpos : ∀ n, a_n n > 0) (S_n : ℕ) (hseq : seq_property n S_n) :
  a_n n = 2 * n + 1 :=
sorry

-- Sum of the first n terms of {b_n}
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def sum_b (n : ℕ) (T_n : ℚ) : Prop :=
  T_n = (1 / 2) * ((1 / (2 * n + 1)) - (1 / (2 * n + 3)))

theorem sum_first_n_b (n : ℕ) (hpos : ∀ n, a_n n > 0) (T_n : ℚ) :
  T_n = (n : ℚ) / (3 * (2 * n + 3)) :=
sorry

end general_formula_a_n_sum_first_n_b_l1688_168803


namespace train_b_speed_l1688_168893

theorem train_b_speed (v : ℝ) (t : ℝ) (d : ℝ) (sA : ℝ := 30) (start_time_diff : ℝ := 2) :
  (d = 180) -> (60 + sA*t = d) -> (v * t = d) -> v = 45 := by 
  sorry

end train_b_speed_l1688_168893


namespace brinley_animal_count_l1688_168871

def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 12 * leopards
def cheetahs : ℕ := snakes / 3  -- rounding down implicitly considered
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem brinley_animal_count : total_animals = 673 :=
by
  -- Mathematical proof would go here.
  sorry

end brinley_animal_count_l1688_168871


namespace milk_revenue_l1688_168875

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end milk_revenue_l1688_168875


namespace wire_ratio_l1688_168814

theorem wire_ratio (bonnie_pieces : ℕ) (length_per_bonnie_piece : ℕ) (roark_volume : ℕ) 
  (unit_cube_volume : ℕ) (bonnie_cube_volume : ℕ) (roark_pieces_per_unit_cube : ℕ)
  (bonnie_total_wire : ℕ := bonnie_pieces * length_per_bonnie_piece)
  (roark_total_wire : ℕ := (bonnie_cube_volume / unit_cube_volume) * roark_pieces_per_unit_cube) :
  bonnie_pieces = 12 →
  length_per_bonnie_piece = 4 →
  unit_cube_volume = 1 →
  bonnie_cube_volume = 64 →
  roark_pieces_per_unit_cube = 12 →
  (bonnie_total_wire / roark_total_wire : ℚ) = 1 / 16 :=
by sorry

end wire_ratio_l1688_168814


namespace translated_vector_ab_l1688_168859

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end translated_vector_ab_l1688_168859


namespace total_patients_in_a_year_l1688_168866

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end total_patients_in_a_year_l1688_168866


namespace quadratic_discriminant_l1688_168820

def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/2) (-2) = 281/4 := by
  sorry

end quadratic_discriminant_l1688_168820


namespace money_distribution_problem_l1688_168849

theorem money_distribution_problem :
  ∃ n : ℕ, (3 * n + n * (n - 1) / 2 = 100 * n) ∧ n = 195 :=
by {
  use 195,
  sorry
}

end money_distribution_problem_l1688_168849


namespace find_a_tangent_line_at_minus_one_l1688_168864

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end find_a_tangent_line_at_minus_one_l1688_168864


namespace probability_4_students_same_vehicle_l1688_168838

-- Define the number of vehicles
def num_vehicles : ℕ := 3

-- Define the probability that 4 students choose the same vehicle
def probability_same_vehicle (n : ℕ) : ℚ :=
  3 / (3^(n : ℤ))

-- Prove that the probability for 4 students is 1/27
theorem probability_4_students_same_vehicle : probability_same_vehicle 4 = 1 / 27 := 
  sorry

end probability_4_students_same_vehicle_l1688_168838


namespace total_cubes_proof_l1688_168807

def Grady_initial_red_cubes := 20
def Grady_initial_blue_cubes := 15
def Gage_initial_red_cubes := 10
def Gage_initial_blue_cubes := 12
def Harper_initial_red_cubes := 8
def Harper_initial_blue_cubes := 10

def Gage_red_received := (2 / 5) * Grady_initial_red_cubes
def Gage_blue_received := (1 / 3) * Grady_initial_blue_cubes

def Grady_red_after_Gage := Grady_initial_red_cubes - Gage_red_received
def Grady_blue_after_Gage := Grady_initial_blue_cubes - Gage_blue_received

def Harper_red_received := (1 / 4) * Grady_red_after_Gage
def Harper_blue_received := (1 / 2) * Grady_blue_after_Gage

def Gage_total_red := Gage_initial_red_cubes + Gage_red_received
def Gage_total_blue := Gage_initial_blue_cubes + Gage_blue_received

def Harper_total_red := Harper_initial_red_cubes + Harper_red_received
def Harper_total_blue := Harper_initial_blue_cubes + Harper_blue_received

def Gage_total_cubes := Gage_total_red + Gage_total_blue
def Harper_total_cubes := Harper_total_red + Harper_total_blue

def Gage_Harper_total_cubes := Gage_total_cubes + Harper_total_cubes

theorem total_cubes_proof : Gage_Harper_total_cubes = 61 := by
  sorry

end total_cubes_proof_l1688_168807


namespace sticker_sum_mod_problem_l1688_168847

theorem sticker_sum_mod_problem :
  ∃ N < 100, (N % 6 = 5) ∧ (N % 8 = 6) ∧ (N = 47 ∨ N = 95) ∧ (47 + 95 = 142) :=
by
  sorry

end sticker_sum_mod_problem_l1688_168847


namespace point_in_third_quadrant_l1688_168855

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l1688_168855


namespace abs_diff_l1688_168804

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l1688_168804


namespace greatest_divisor_l1688_168852

theorem greatest_divisor (d : ℕ) (h1 : 4351 % d = 8) (h2 : 5161 % d = 10) : d = 1 :=
by
  -- Proof goes here
  sorry

end greatest_divisor_l1688_168852


namespace percent_defective_shipped_l1688_168801

theorem percent_defective_shipped
  (P_d : ℝ) (P_s : ℝ)
  (hP_d : P_d = 0.1)
  (hP_s : P_s = 0.05) :
  P_d * P_s = 0.005 :=
by
  sorry

end percent_defective_shipped_l1688_168801


namespace division_remainder_l1688_168817

theorem division_remainder : 4053 % 23 = 5 :=
by
  sorry

end division_remainder_l1688_168817


namespace percentage_boy_scouts_l1688_168896

theorem percentage_boy_scouts (S B G : ℝ) (h1 : B + G = S)
  (h2 : 0.60 * S = 0.50 * B + 0.6818 * G) : (B / S) * 100 = 45 := by
  sorry

end percentage_boy_scouts_l1688_168896


namespace grid_3x3_unique_72_l1688_168844

theorem grid_3x3_unique_72 :
  ∃ (f : Fin 3 → Fin 3 → ℕ), 
    (∀ (i j : Fin 3), 1 ≤ f i j ∧ f i j ≤ 9) ∧
    (∀ (i j k : Fin 3), j < k → f i j < f i k) ∧
    (∀ (i j k : Fin 3), i < k → f i j < f k j) ∧
    f 0 0 = 1 ∧ f 1 1 = 5 ∧ f 2 2 = 8 ∧
    (∃! (g : Fin 3 → Fin 3 → ℕ), 
      (∀ (i j : Fin 3), 1 ≤ g i j ∧ g i j ≤ 9) ∧
      (∀ (i j k : Fin 3), j < k → g i j < g i k) ∧
      (∀ (i j k : Fin 3), i < k → g i j < g k j) ∧
      g 0 0 = 1 ∧ g 1 1 = 5 ∧ g 2 2 = 8) :=
sorry

end grid_3x3_unique_72_l1688_168844


namespace fourth_term_geom_progression_l1688_168812

theorem fourth_term_geom_progression : 
  ∀ (a b c : ℝ), 
    a = 4^(1/2) → 
    b = 4^(1/3) → 
    c = 4^(1/6) → 
    ∃ d : ℝ, d = 1 ∧ b / a = c / b ∧ c / b = 4^(1/6) / 4^(1/3) :=
by
  sorry

end fourth_term_geom_progression_l1688_168812


namespace radian_measure_of_negative_150_degree_l1688_168869

theorem radian_measure_of_negative_150_degree  : (-150 : ℝ) * (Real.pi / 180) = - (5 * Real.pi / 6) := by
  sorry

end radian_measure_of_negative_150_degree_l1688_168869


namespace merchant_gross_profit_l1688_168819

noncomputable def grossProfit (purchase_price : ℝ) (selling_price : ℝ) (discount : ℝ) : ℝ :=
  (selling_price - discount * selling_price) - purchase_price

theorem merchant_gross_profit :
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  grossProfit P S discount = 8 := 
by
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  unfold grossProfit
  sorry

end merchant_gross_profit_l1688_168819


namespace problem_statement_l1688_168827

theorem problem_statement (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (hprod : m * n = 5000) 
  (h_m_not_div_10 : ¬ ∃ k, m = 10 * k) (h_n_not_div_10 : ¬ ∃ k, n = 10 * k) :
  m + n = 633 :=
sorry

end problem_statement_l1688_168827


namespace parabola_properties_l1688_168872

theorem parabola_properties (p m k1 k2 k3 : ℝ)
  (parabola_eq : ∀ x y, y^2 = 2 * p * x ↔ y = m)
  (parabola_passes_through : m^2 = 2 * p)
  (point_distance : ((1 + p / 2)^2 + m^2 = 8) ∨ ((1 + p / 2)^2 + m^2 = 8))
  (p_gt_zero : p > 0)
  (point_P : (1, 2) ∈ { (x, y) | y^2 = 4 * x })
  (slope_eq : k3 = (k1 * k2) / (k1 + k2 - k1 * k2)) :
  (y^2 = 4 * x) ∧ (1/k1 + 1/k2 - 1/k3 = 1) := sorry

end parabola_properties_l1688_168872


namespace shopkeeper_discount_l1688_168841

theorem shopkeeper_discount :
  let CP := 100
  let SP_with_discount := 119.7
  let SP_without_discount := 126
  let discount := SP_without_discount - SP_with_discount
  let discount_percentage := (discount / SP_without_discount) * 100
  discount_percentage = 5 := sorry

end shopkeeper_discount_l1688_168841


namespace chord_length_of_larger_circle_tangent_to_smaller_circle_l1688_168851

theorem chord_length_of_larger_circle_tangent_to_smaller_circle :
  ∀ (A B C : ℝ), B = 5 → π * (A ^ 2 - B ^ 2) = 50 * π → (C / 2) ^ 2 + B ^ 2 = A ^ 2 → C = 10 * Real.sqrt 2 :=
by
  intros A B C hB hArea hChord
  sorry

end chord_length_of_larger_circle_tangent_to_smaller_circle_l1688_168851


namespace Jon_needs_to_wash_20_pairs_of_pants_l1688_168822

theorem Jon_needs_to_wash_20_pairs_of_pants
  (machine_capacity : ℕ)
  (shirts_per_pound : ℕ)
  (pants_per_pound : ℕ)
  (num_shirts : ℕ)
  (num_loads : ℕ)
  (total_pounds : ℕ)
  (weight_of_shirts : ℕ)
  (remaining_weight : ℕ)
  (num_pairs_of_pants : ℕ) :
  machine_capacity = 5 →
  shirts_per_pound = 4 →
  pants_per_pound = 2 →
  num_shirts = 20 →
  num_loads = 3 →
  total_pounds = num_loads * machine_capacity →
  weight_of_shirts = num_shirts / shirts_per_pound →
  remaining_weight = total_pounds - weight_of_shirts →
  num_pairs_of_pants = remaining_weight * pants_per_pound →
  num_pairs_of_pants = 20 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end Jon_needs_to_wash_20_pairs_of_pants_l1688_168822


namespace root_in_interval_iff_a_range_l1688_168831

def f (a x : ℝ) : ℝ := 2 * a * x ^ 2 + 2 * x - 3 - a

theorem root_in_interval_iff_a_range (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0) ↔ (1 ≤ a ∨ a ≤ - (3 + Real.sqrt 7) / 2) :=
sorry

end root_in_interval_iff_a_range_l1688_168831


namespace range_of_a_l1688_168898

noncomputable def f (a x : ℝ) :=
  if x < 0 then
    9 * x + a^2 / x + 7
  else
    9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8 / 7 :=
  sorry

end range_of_a_l1688_168898


namespace cube_vertex_numbering_impossible_l1688_168802

-- Definition of the cube problem
def vertex_numbering_possible : Prop :=
  ∃ (v : Fin 8 → ℕ), (∀ i, 1 ≤ v i ∧ v i ≤ 8) ∧
    (∀ (e1 e2 : (Fin 8 × Fin 8)), e1 ≠ e2 → (v e1.1 + v e1.2 ≠ v e2.1 + v e2.2))

theorem cube_vertex_numbering_impossible : ¬ vertex_numbering_possible :=
sorry

end cube_vertex_numbering_impossible_l1688_168802


namespace driver_weekly_distance_l1688_168830

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l1688_168830


namespace arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l1688_168862

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def seq_sum (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  (Finset.range n).sum seq

noncomputable def T_n (n : ℕ) : ℕ :=
  seq_sum n (λ i => (a_n (i + 1) + 1) * b_n (i + 1))

theorem arithmetic_seq_general_term (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

theorem geometric_seq_general_term (n : ℕ) : b_n n = 2^n := by
  sorry

theorem sequence_sum (n : ℕ) : T_n n = (n - 1) * 2^(n+2) + 4 := by
  sorry

end arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l1688_168862


namespace six_units_away_has_two_solutions_l1688_168882

-- Define point A and its position on the number line
def A_position : ℤ := -3

-- Define the condition for a point x being 6 units away from point A
def is_6_units_away (x : ℤ) : Prop := abs (x + 3) = 6

-- The theorem stating that if x is 6 units away from -3, then x must be either 3 or -9
theorem six_units_away_has_two_solutions (x : ℤ) (h : is_6_units_away x) : x = 3 ∨ x = -9 := by
  sorry

end six_units_away_has_two_solutions_l1688_168882


namespace transformed_function_zero_l1688_168816

-- Definitions based on conditions
def f : ℝ → ℝ → ℝ := sorry  -- Assume this is the given function f(x, y)

-- Transformed function according to symmetry and reflections
def transformed_f (x y : ℝ) : Prop := f (y + 2) (x - 2) = 0

-- Lean statement to be proved
theorem transformed_function_zero (x y : ℝ) : transformed_f x y := sorry

end transformed_function_zero_l1688_168816


namespace common_difference_is_3_l1688_168808

noncomputable def whale_plankton_frenzy (x : ℝ) (y : ℝ) : Prop :=
  (9 * x + 36 * y = 450) ∧
  (x + 5 * y = 53)

theorem common_difference_is_3 :
  ∃ (x y : ℝ), whale_plankton_frenzy x y ∧ y = 3 :=
by {
  sorry
}

end common_difference_is_3_l1688_168808


namespace no_prime_satisfies_polynomial_l1688_168897

theorem no_prime_satisfies_polynomial :
  ∀ p : ℕ, p.Prime → p^3 - 6*p^2 - 3*p + 14 ≠ 0 := by
  sorry

end no_prime_satisfies_polynomial_l1688_168897


namespace XiaoKang_min_sets_pushups_pullups_l1688_168811

theorem XiaoKang_min_sets_pushups_pullups (x y : ℕ) (hx : x ≥ 100) (hy : y ≥ 106) (h : 8 * x + 5 * y = 9050) :
  x ≥ 100 ∧ y ≥ 106 :=
by {
  sorry  -- proof not required as per instruction
}

end XiaoKang_min_sets_pushups_pullups_l1688_168811


namespace solve_x_l1688_168854

def otimes (a b : ℝ) : ℝ := a - 3 * b

theorem solve_x : ∃ x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 :=
by
  use -1
  rw [otimes, otimes]
  sorry

end solve_x_l1688_168854


namespace q_compound_l1688_168846

def q (x y : ℤ) : ℤ :=
  if x ≥ 1 ∧ y ≥ 1 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4 * x - 2 * y

theorem q_compound : q (q 2 (-2)) (q 0 0) = 48 := 
by 
  sorry

end q_compound_l1688_168846


namespace sum_not_complete_residue_system_l1688_168894

theorem sum_not_complete_residue_system {n : ℕ} (hn_even : Even n)
    (a b : Fin n → ℕ) (ha : ∀ k, a k < n) (hb : ∀ k, b k < n) 
    (h_complete_a : ∀ x : Fin n, ∃ k : Fin n, a k = x) 
    (h_complete_b : ∀ y : Fin n, ∃ k : Fin n, b k = y) :
    ¬ (∀ z : Fin n, ∃ k : Fin n, ∃ l : Fin n, z = (a k + b l) % n) :=
by
  sorry

end sum_not_complete_residue_system_l1688_168894


namespace base_n_divisible_by_13_l1688_168806

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 7 + 3 * n + 5 * n^2 + 6 * n^3 + 3 * n^4 + 5 * n^5

-- The main theorem stating the result
theorem base_n_divisible_by_13 : 
  (∃ ns : Finset ℕ, ns.card = 16 ∧ ∀ n ∈ ns, 3 ≤ n ∧ n ≤ 200 ∧ f n % 13 = 0) :=
sorry

end base_n_divisible_by_13_l1688_168806


namespace garden_length_l1688_168884

theorem garden_length (P : ℕ) (breadth : ℕ) (length : ℕ) 
  (h1 : P = 600) (h2 : breadth = 95) (h3 : P = 2 * (length + breadth)) : 
  length = 205 :=
by
  sorry

end garden_length_l1688_168884


namespace find_r_l1688_168880

theorem find_r (k r : ℝ) : 
  5 = k * 3^r ∧ 45 = k * 9^r → r = 2 :=
by 
  sorry

end find_r_l1688_168880


namespace max_value_of_reciprocals_l1688_168878

noncomputable def quadratic (x t q : ℝ) : ℝ := x^2 - t * x + q

theorem max_value_of_reciprocals (α β t q : ℝ) (h1 : α + β = α^2 + β^2)
                                               (h2 : α + β = α^3 + β^3)
                                               (h3 : ∀ n, 1 ≤ n ∧ n ≤ 2010 → α^n + β^n = α + β)
                                               (h4 : α * β = q)
                                               (h5 : α + β = t) :
  ∃ (α β : ℝ), (1 / α^2012 + 1 / β^2012) = 2 := 
sorry

end max_value_of_reciprocals_l1688_168878


namespace isosceles_triangle_angle_l1688_168861

-- Definition of required angles and the given geometric context
variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]
variables (angleBAC : ℝ) (angleBCA : ℝ)

-- Given: shared vertex A, with angle BAC of pentagon
axiom angleBAC_def : angleBAC = 108

-- To Prove: determining the measure of angle BCA in the isosceles triangle
theorem isosceles_triangle_angle (h : 180 > 2 * angleBAC) : angleBCA = (180 - angleBAC) / 2 :=
  sorry

end isosceles_triangle_angle_l1688_168861


namespace cut_half_meter_from_cloth_l1688_168800

theorem cut_half_meter_from_cloth (initial_length : ℝ) (cut_length : ℝ) : 
  initial_length = 8 / 15 → cut_length = 1 / 30 → initial_length - cut_length = 1 / 2 := 
by
  intros h_initial h_cut
  sorry

end cut_half_meter_from_cloth_l1688_168800


namespace linear_equation_condition_l1688_168829

theorem linear_equation_condition (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x ^ (|a|⁻¹ + 3) = 0) ↔ a = -2 := 
by
  sorry

end linear_equation_condition_l1688_168829


namespace simplify_abs_expression_l1688_168845

/-- Simplify the expression: |-4^3 + 5^2 - 6| and prove the result is equal to 45 -/
theorem simplify_abs_expression :
  |(- 4 ^ 3 + 5 ^ 2 - 6)| = 45 :=
by
  sorry

end simplify_abs_expression_l1688_168845


namespace noemi_initial_amount_l1688_168834

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end noemi_initial_amount_l1688_168834


namespace probability_white_then_black_l1688_168825

-- Definition of conditions
def total_balls := 5
def white_balls := 3
def black_balls := 2

def first_draw_white_probability (total white : ℕ) : ℚ :=
  white / total

def second_draw_black_probability (remaining_white remaining_black : ℕ) : ℚ :=
  remaining_black / (remaining_white + remaining_black)

-- The theorem statement
theorem probability_white_then_black :
  first_draw_white_probability total_balls white_balls *
  second_draw_black_probability (total_balls - 1) black_balls
  = 3 / 10 :=
by
  sorry

end probability_white_then_black_l1688_168825


namespace minimum_jumps_l1688_168837

theorem minimum_jumps (a b : ℕ) (h : 2 * a + 3 * b = 2016) : a + b = 673 :=
sorry

end minimum_jumps_l1688_168837


namespace functional_equation_solution_l1688_168843

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧ (∀ x y : ℝ, f (x + y) * f (x + y) = 2 * f x * f y + max (f (x * x) + f (y * y)) (f (x * x + y * y)))

theorem functional_equation_solution (f : ℝ → ℝ) :
  satisfies_conditions f → (∀ x : ℝ, f x = -1 ∨ f x = x - 1) :=
by
  intros h
  sorry

end functional_equation_solution_l1688_168843


namespace num_pos_integers_congruent_to_4_mod_7_l1688_168888

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l1688_168888


namespace maximum_area_of_right_triangle_l1688_168874

theorem maximum_area_of_right_triangle
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2) : 
  ∃ S, S ≤ (3 - 2 * Real.sqrt 2) ∧ S = (1/2) * a * b :=
by
  sorry

end maximum_area_of_right_triangle_l1688_168874


namespace solve_for_a_l1688_168887

-- Given conditions
def x : ℕ := 2
def y : ℕ := 2
def equation (a : ℚ) : Prop := a * x + y = 5

-- Our goal is to prove that "a = 3/2" given the conditions
theorem solve_for_a : ∃ a : ℚ, equation a ∧ a = 3 / 2 :=
by
  sorry

end solve_for_a_l1688_168887


namespace cubic_equation_solution_bound_l1688_168815

theorem cubic_equation_solution_bound (a : ℝ) :
  a ∈ Set.Ici (-15) → ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ → x₂ ≠ x₃ → x₁ ≠ x₃ →
  (x₁^3 + 6 * x₁^2 + a * x₁ + 8 = 0) →
  (x₂^3 + 6 * x₂^2 + a * x₂ + 8 = 0) →
  (x₃^3 + 6 * x₃^2 + a * x₃ + 8 = 0) →
  False := 
sorry

end cubic_equation_solution_bound_l1688_168815


namespace Erica_Ice_Cream_Spend_l1688_168858

theorem Erica_Ice_Cream_Spend :
  (6 * ((3 * 2.00) + (2 * 1.50) + (2 * 3.00))) = 90 := sorry

end Erica_Ice_Cream_Spend_l1688_168858


namespace length_of_plot_l1688_168823

theorem length_of_plot 
  (b : ℝ)
  (H1 : 2 * (b + 20) + 2 * b = 5300 / 26.50)
  : (b + 20 = 60) :=
sorry

end length_of_plot_l1688_168823


namespace owen_profit_l1688_168876

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end owen_profit_l1688_168876
