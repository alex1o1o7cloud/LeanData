import Mathlib

namespace NUMINAMATH_GPT_least_positive_three_digit_multiple_of_9_l870_87010

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end NUMINAMATH_GPT_least_positive_three_digit_multiple_of_9_l870_87010


namespace NUMINAMATH_GPT_brock_peanuts_ratio_l870_87080

theorem brock_peanuts_ratio (initial : ℕ) (bonita : ℕ) (remaining : ℕ) (brock : ℕ)
  (h1 : initial = 148) (h2 : bonita = 29) (h3 : remaining = 82) (h4 : brock = 37)
  (h5 : initial - remaining = bonita + brock) :
  (brock : ℚ) / initial = 1 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_brock_peanuts_ratio_l870_87080


namespace NUMINAMATH_GPT_ptarmigan_environmental_capacity_l870_87033

theorem ptarmigan_environmental_capacity (predators_eradicated : Prop) (mass_deaths : Prop) : 
  (after_predator_eradication : predators_eradicated → mass_deaths) →
  (environmental_capacity_increased : Prop) → environmental_capacity_increased :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ptarmigan_environmental_capacity_l870_87033


namespace NUMINAMATH_GPT_find_possible_first_term_l870_87062

noncomputable def geometric_sequence_first_term (a r : ℝ) : Prop :=
  (a * r^2 = 3) ∧ (a * r^4 = 27)

theorem find_possible_first_term (a r : ℝ) (h : geometric_sequence_first_term a r) :
    a = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_possible_first_term_l870_87062


namespace NUMINAMATH_GPT_lucas_should_give_fraction_l870_87064

-- Conditions as Lean definitions
variables (n : ℕ) -- Number of shells Noah has
def Noah_shells := n
def Emma_shells := 2 * n -- Emma has twice as many shells as Noah
def Lucas_shells := 8 * n -- Lucas has four times as many shells as Emma

-- Desired distribution
def Total_shells := Noah_shells n + Emma_shells n + Lucas_shells n
def Each_person_shells := Total_shells n / 3

-- Fraction calculation
def Shells_needed_by_Emma := Each_person_shells n - Emma_shells n
def Fraction_of_Lucas_shells_given_to_Emma := Shells_needed_by_Emma n / Lucas_shells n 

theorem lucas_should_give_fraction :
  Fraction_of_Lucas_shells_given_to_Emma n = 5 / 24 := 
by
  sorry

end NUMINAMATH_GPT_lucas_should_give_fraction_l870_87064


namespace NUMINAMATH_GPT_number_of_customers_l870_87032

theorem number_of_customers
  (nails_per_person : ℕ)
  (total_sounds : ℕ)
  (trimmed_nails_per_person : nails_per_person = 20)
  (produced_sounds : total_sounds = 100) :
  total_sounds / nails_per_person = 5 :=
by
  -- This is offered as a placeholder to indicate where a Lean proof goes.
  sorry

end NUMINAMATH_GPT_number_of_customers_l870_87032


namespace NUMINAMATH_GPT_find_pos_ints_l870_87065

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end NUMINAMATH_GPT_find_pos_ints_l870_87065


namespace NUMINAMATH_GPT_constant_term_expansion_l870_87048

-- Define the binomial coefficient
noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term in the binomial expansion
noncomputable def general_term (r n : ℕ) (x : ℝ) : ℝ := 
  (2:ℝ)^r * binomial_coeff n r * x^((n-5*r)/2)

-- Given problem conditions
def n := 10
def largest_binomial_term_index := 5  -- Represents the sixth term (r = 5)

-- Statement to prove the constant term equals 180
theorem constant_term_expansion {x : ℝ} : 
  general_term 2 n 1 = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_constant_term_expansion_l870_87048


namespace NUMINAMATH_GPT_polynomial_range_l870_87028

def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 8*x^2 - 8*x + 5

theorem polynomial_range : ∀ x : ℝ, p x ≥ 2 :=
by
sorry

end NUMINAMATH_GPT_polynomial_range_l870_87028


namespace NUMINAMATH_GPT_abs_ineq_solution_l870_87045

theorem abs_ineq_solution (x : ℝ) : (2 ≤ |x - 5| ∧ |x - 5| ≤ 4) ↔ (1 ≤ x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_l870_87045


namespace NUMINAMATH_GPT_parabola_y_axis_symmetry_l870_87055

theorem parabola_y_axis_symmetry (a b c d : ℝ) (r : ℝ) :
  (2019^2 + 2019 * a + b = 0) ∧ (2019^2 + 2019 * c + d = 0) ∧
  (a = -(2019 + r)) ∧ (c = -(2019 - r)) →
  b = -d :=
by
  sorry

end NUMINAMATH_GPT_parabola_y_axis_symmetry_l870_87055


namespace NUMINAMATH_GPT_minimum_value_of_f_l870_87012

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1 / (4 * x - 5)

theorem minimum_value_of_f (x : ℝ) : x > 5 / 4 → ∃ y, ∀ z, f z ≥ y ∧ y = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l870_87012


namespace NUMINAMATH_GPT_twenty_four_game_solution_l870_87013

theorem twenty_four_game_solution :
  let a := 4
  let b := 8
  (a - (b / b)) * b = 24 :=
by
  let a := 4
  let b := 8
  show (a - (b / b)) * b = 24
  sorry

end NUMINAMATH_GPT_twenty_four_game_solution_l870_87013


namespace NUMINAMATH_GPT_sequence_relation_l870_87067

theorem sequence_relation
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h1 : ∀ n, b (n + 1) * a n + b n * a (n + 1) = (-2)^n + 1)
  (h2 : ∀ n, b n = (3 + (-1 : ℚ)^(n-1)) / 2)
  (h3 : a 1 = 2) :
  ∀ n, a (2 * n) = (1 - 4^n) / 2 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sequence_relation_l870_87067


namespace NUMINAMATH_GPT_nina_total_cost_l870_87015

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end NUMINAMATH_GPT_nina_total_cost_l870_87015


namespace NUMINAMATH_GPT_straw_costs_max_packs_type_a_l870_87034

theorem straw_costs (x y : ℝ) (h1 : 12 * x + 15 * y = 171) (h2 : 24 * x + 28 * y = 332) :
  x = 8 ∧ y = 5 :=
  by sorry

theorem max_packs_type_a (m : ℕ) (cA cB : ℕ) (total_packs : ℕ) (max_cost : ℕ)
  (h1 : cA = 8) (h2 : cB = 5) (h3 : total_packs = 100) (h4 : max_cost = 600) :
  m ≤ 33 :=
  by sorry

end NUMINAMATH_GPT_straw_costs_max_packs_type_a_l870_87034


namespace NUMINAMATH_GPT_range_of_k_l870_87081

noncomputable def quadratic_inequality (k x : ℝ) : ℝ :=
  k * x^2 + 2 * k * x - (k + 2)

theorem range_of_k :
  (∀ x : ℝ, quadratic_inequality k x < 0) ↔ -1 < k ∧ k < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l870_87081


namespace NUMINAMATH_GPT_A_share_correct_l870_87089

noncomputable def investment_shares (x : ℝ) (annual_gain : ℝ) := 
  let A_share := x * 12
  let B_share := (2 * x) * 6
  let C_share := (3 * x) * 4
  let total_share := A_share + B_share + C_share
  let total_ratio := 1 + 1 + 1
  annual_gain / total_ratio

theorem A_share_correct (x : ℝ) (annual_gain : ℝ) (h_gain : annual_gain = 18000) : 
  investment_shares x annual_gain / 3 = 6000 := by
  sorry

end NUMINAMATH_GPT_A_share_correct_l870_87089


namespace NUMINAMATH_GPT_tennis_ball_price_l870_87091

theorem tennis_ball_price (x y : ℝ) 
  (h₁ : 2 * x + 7 * y = 220)
  (h₂ : x = y + 83) : 
  y = 6 := 
by 
  sorry

end NUMINAMATH_GPT_tennis_ball_price_l870_87091


namespace NUMINAMATH_GPT_solve_for_y_l870_87066

def G (a y c d : ℕ) := 3 ^ y + 6 * d

theorem solve_for_y (a c d : ℕ) (h1 : G a 2 c d = 735) : 2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l870_87066


namespace NUMINAMATH_GPT_find_angle_A_l870_87088

theorem find_angle_A
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) : A = 50 :=
by 
  sorry

end NUMINAMATH_GPT_find_angle_A_l870_87088


namespace NUMINAMATH_GPT_smallest_x_satisfies_eq_l870_87017

theorem smallest_x_satisfies_eq : ∃ x : ℝ, (1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))) ∧ x = 7 - Real.sqrt 6 :=
by
  -- The proof steps would go here, but we're skipping them with sorry for now.
  sorry

end NUMINAMATH_GPT_smallest_x_satisfies_eq_l870_87017


namespace NUMINAMATH_GPT_optionD_is_equation_l870_87093

-- Definitions for options
def optionA (x : ℕ) := 2 * x - 3
def optionB := 2 + 4 = 6
def optionC (x : ℕ) := x > 2
def optionD (x : ℕ) := 2 * x - 1 = 3

-- Goal: prove that option D is an equation.
theorem optionD_is_equation (x : ℕ) : (optionD x) = True :=
sorry

end NUMINAMATH_GPT_optionD_is_equation_l870_87093


namespace NUMINAMATH_GPT_Hamilton_marching_band_members_l870_87086

theorem Hamilton_marching_band_members (m : ℤ) (k : ℤ) :
  30 * m ≡ 5 [ZMOD 31] ∧ m = 26 + 31 * k ∧ 30 * m < 1500 → 30 * m = 780 :=
by
  intro h
  have hmod : 30 * m ≡ 5 [ZMOD 31] := h.left
  have m_eq : m = 26 + 31 * k := h.right.left
  have hlt : 30 * m < 1500 := h.right.right
  sorry

end NUMINAMATH_GPT_Hamilton_marching_band_members_l870_87086


namespace NUMINAMATH_GPT_mary_total_money_l870_87023

def num_quarters : ℕ := 21
def quarters_worth : ℚ := 0.25
def dimes_worth : ℚ := 0.10

def num_dimes (Q : ℕ) : ℕ := (Q - 7) / 2

def total_money (Q : ℕ) (D : ℕ) : ℚ :=
  Q * quarters_worth + D * dimes_worth

theorem mary_total_money : 
  total_money num_quarters (num_dimes num_quarters) = 5.95 := 
by
  sorry

end NUMINAMATH_GPT_mary_total_money_l870_87023


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l870_87058

theorem trigonometric_identity_proof (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = - (Real.sqrt 3 + 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l870_87058


namespace NUMINAMATH_GPT_find_m_such_that_no_linear_term_in_expansion_l870_87069

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_m_such_that_no_linear_term_in_expansion_l870_87069


namespace NUMINAMATH_GPT_largest_corner_sum_l870_87036

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end NUMINAMATH_GPT_largest_corner_sum_l870_87036


namespace NUMINAMATH_GPT_sum_of_prime_factors_of_91_l870_87020

theorem sum_of_prime_factors_of_91 : 
  (¬ (91 % 2 = 0)) ∧ 
  (¬ (91 % 3 = 0)) ∧ 
  (¬ (91 % 5 = 0)) ∧ 
  (91 = 7 * 13) →
  (7 + 13 = 20) := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_of_91_l870_87020


namespace NUMINAMATH_GPT_range_of_a_l870_87072

-- Given function
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- Derivative of the function
def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

-- Discriminant of the derivative
def discriminant (a : ℝ) : ℝ := 4*a^2 - 12*(a + 6)

-- Proof that the range of 'a' is 'a < -3 or a > 6' for f(x) to have both maximum and minimum values
theorem range_of_a (a : ℝ) : discriminant a > 0 ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l870_87072


namespace NUMINAMATH_GPT_coordinates_of_point_l870_87006

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end NUMINAMATH_GPT_coordinates_of_point_l870_87006


namespace NUMINAMATH_GPT_exactly_one_solves_l870_87003

-- Define the independent probabilities for person A and person B
variables (p₁ p₂ : ℝ)

-- Assume probabilities are between 0 and 1 inclusive
axiom h1 : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom h2 : 0 ≤ p₂ ∧ p₂ ≤ 1

theorem exactly_one_solves : (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = (p₁ * (1 - p₂) + p₂ * (1 - p₁)) := 
by sorry

end NUMINAMATH_GPT_exactly_one_solves_l870_87003


namespace NUMINAMATH_GPT_third_smallest_four_digit_in_pascals_triangle_l870_87050

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end NUMINAMATH_GPT_third_smallest_four_digit_in_pascals_triangle_l870_87050


namespace NUMINAMATH_GPT_average_percentage_difference_in_tail_sizes_l870_87030

-- Definitions for the number of segments in each type of rattlesnake
def segments_eastern : ℕ := 6
def segments_western : ℕ := 8
def segments_southern : ℕ := 7
def segments_northern : ℕ := 9

-- Definition for percentage difference function
def percentage_difference (a : ℕ) (b : ℕ) : ℚ := ((b - a : ℚ) / b) * 100

-- Theorem statement to prove the average percentage difference
theorem average_percentage_difference_in_tail_sizes :
  (percentage_difference segments_eastern segments_western +
   percentage_difference segments_southern segments_western +
   percentage_difference segments_northern segments_western) / 3 = 16.67 := 
sorry

end NUMINAMATH_GPT_average_percentage_difference_in_tail_sizes_l870_87030


namespace NUMINAMATH_GPT_simplify_expression_l870_87092

theorem simplify_expression : 
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l870_87092


namespace NUMINAMATH_GPT_days_to_complete_work_l870_87056

-- Conditions
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def combined_work_rate := work_rate_A + work_rate_B

-- Statement to prove
theorem days_to_complete_work : 1 / combined_work_rate = 16 / 3 := by
  sorry

end NUMINAMATH_GPT_days_to_complete_work_l870_87056


namespace NUMINAMATH_GPT_gcd_228_1995_l870_87031

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_GPT_gcd_228_1995_l870_87031


namespace NUMINAMATH_GPT_greatest_common_divisor_546_180_l870_87009

theorem greatest_common_divisor_546_180 : 
  ∃ d, d < 70 ∧ d > 0 ∧ d ∣ 546 ∧ d ∣ 180 ∧ ∀ x, x < 70 ∧ x > 0 ∧ x ∣ 546 ∧ x ∣ 180 → x ≤ d → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_546_180_l870_87009


namespace NUMINAMATH_GPT_rita_months_needed_l870_87049

def total_hours_required : ℕ := 2500
def backstroke_hours : ℕ := 75
def breaststroke_hours : ℕ := 25
def butterfly_hours : ℕ := 200
def hours_per_month : ℕ := 300

def total_completed_hours : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_hours_required - total_completed_hours
def months_needed (remaining_hours hours_per_month : ℕ) : ℕ := (remaining_hours + hours_per_month - 1) / hours_per_month

theorem rita_months_needed : months_needed remaining_hours hours_per_month = 8 := by
  -- Lean 4 proof goes here
  sorry

end NUMINAMATH_GPT_rita_months_needed_l870_87049


namespace NUMINAMATH_GPT_trip_time_l870_87057

open Real

variables (d T : Real)

theorem trip_time :
  (T = d / 30 + (150 - d) / 6) ∧
  (T = 2 * (d / 30) + 1 + (150 - d) / 30) ∧
  (T - 1 = d / 6 + (150 - d) / 30) →
  T = 20 :=
by
  sorry

end NUMINAMATH_GPT_trip_time_l870_87057


namespace NUMINAMATH_GPT_carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l870_87079

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9
def tom_weight : ℕ := 20

theorem carol_tom_combined_weight :
  carol_weight + tom_weight = 29 := by
  sorry

theorem mildred_heavier_than_carol_tom_combined :
  mildred_weight - (carol_weight + tom_weight) = 30 := by
  sorry

end NUMINAMATH_GPT_carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l870_87079


namespace NUMINAMATH_GPT_quadratic_inequality_l870_87096

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - 3*x - m = 0 ∧ (∃ y : ℝ, y^2 - 3*y - m = 0 ∧ x ≠ y)) ↔ m > - 9 / 4 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l870_87096


namespace NUMINAMATH_GPT_distance_between_foci_l870_87060

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36 = 0

-- Define the distance between the foci of the ellipse
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 2 * Real.sqrt 14.28 = 2 * Real.sqrt 14.28 :=
by sorry

end NUMINAMATH_GPT_distance_between_foci_l870_87060


namespace NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_l870_87084

theorem factorize_1 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

theorem factorize_2 (x y : ℝ) : 25*x^2*y + 20*x*y^2 + 4*y^3 = y * (5*x + 2*y)^2 :=
sorry

theorem factorize_3 (x y a : ℝ) : x^2 * (a - 1) + y^2 * (1 - a) = (a - 1) * (x + y) * (x - y) :=
sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_l870_87084


namespace NUMINAMATH_GPT_original_price_calculation_l870_87002

-- Definitions directly from problem conditions
def price_after_decrease (original_price : ℝ) : ℝ := 0.76 * original_price
def new_price : ℝ := 988

-- Statement embedding our problem
theorem original_price_calculation (x : ℝ) (hx : price_after_decrease x = new_price) : x = 1300 :=
by
  sorry

end NUMINAMATH_GPT_original_price_calculation_l870_87002


namespace NUMINAMATH_GPT_pen_cost_proof_l870_87035

-- Given definitions based on the problem conditions
def is_majority (s : ℕ) := s > 20
def is_odd_and_greater_than_one (n : ℕ) := n > 1 ∧ n % 2 = 1
def is_prime (c : ℕ) := Nat.Prime c

-- The final theorem to prove the correct answer
theorem pen_cost_proof (s n c : ℕ) 
  (h_majority : is_majority s) 
  (h_odd : is_odd_and_greater_than_one n) 
  (h_prime : is_prime c) 
  (h_eq : s * c * n = 2091) : 
  c = 47 := 
sorry

end NUMINAMATH_GPT_pen_cost_proof_l870_87035


namespace NUMINAMATH_GPT_prob1_prob2_prob3_prob4_l870_87052

theorem prob1 : (-20) + (-14) - (-18) - 13 = -29 := sorry

theorem prob2 : (-24) * (-1/2 + 3/4 - 1/3) = 2 := sorry

theorem prob3 : (- (49 + 24/25)) * 10 = -499.6 := sorry

theorem prob4 :
  -3^2 + ((-1/3) * (-3) - 8/5 / 2^2) = -8 - 2/5 := sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_prob4_l870_87052


namespace NUMINAMATH_GPT_central_angle_of_sector_with_area_one_l870_87007

theorem central_angle_of_sector_with_area_one (θ : ℝ):
  (1 / 2) * θ = 1 → θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_with_area_one_l870_87007


namespace NUMINAMATH_GPT_boys_joined_school_l870_87001

theorem boys_joined_school (initial_boys final_boys boys_joined : ℕ) 
  (h1 : initial_boys = 214) 
  (h2 : final_boys = 1124) 
  (h3 : final_boys = initial_boys + boys_joined) : 
  boys_joined = 910 := 
by 
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_boys_joined_school_l870_87001


namespace NUMINAMATH_GPT_find_a_range_l870_87082

-- Definitions as per conditions
def prop_P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0
def prop_Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Given conditions
def P_true (a : ℝ) (h : prop_P a) : Prop :=
  ∀ (a : ℝ), a^2 - 16 < 0

def Q_false (a : ℝ) (h : ¬prop_Q a) : Prop :=
  ∀ (a : ℝ), a > 1

-- Main theorem
theorem find_a_range (a : ℝ) (hP : prop_P a) (hQ : ¬prop_Q a) : 1 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_find_a_range_l870_87082


namespace NUMINAMATH_GPT_compare_a_b_l870_87011

theorem compare_a_b (m : ℝ) (h : m > 1) 
  (a : ℝ := (Real.sqrt (m+1)) - (Real.sqrt m))
  (b : ℝ := (Real.sqrt m) - (Real.sqrt (m-1))) : a < b :=
by
  sorry

end NUMINAMATH_GPT_compare_a_b_l870_87011


namespace NUMINAMATH_GPT_min_fraction_ineq_l870_87059

theorem min_fraction_ineq (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  ∃ z, (z = x * y / (x^2 + 2 * y^2)) ∧ z = 1 / 3 := sorry

end NUMINAMATH_GPT_min_fraction_ineq_l870_87059


namespace NUMINAMATH_GPT_max_min_vec_magnitude_l870_87044

noncomputable def vec_a (θ : ℝ) := (Real.cos θ, Real.sin θ)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def vec_result (θ : ℝ) := (2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ - 1)

noncomputable def vec_magnitude (θ : ℝ) := Real.sqrt ((2 * Real.cos θ - Real.sqrt 3)^2 + (2 * Real.sin θ - 1)^2)

theorem max_min_vec_magnitude : 
  ∃ θ_max θ_min, 
    vec_magnitude θ_max = 4 ∧ 
    vec_magnitude θ_min = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_min_vec_magnitude_l870_87044


namespace NUMINAMATH_GPT_steve_speed_l870_87077

theorem steve_speed (v : ℝ) : 
  (John_initial_distance_behind_Steve = 15) ∧ 
  (John_final_distance_ahead_of_Steve = 2) ∧ 
  (John_speed = 4.2) ∧ 
  (final_push_duration = 34) → 
  v * final_push_duration = (John_speed * final_push_duration) - (John_initial_distance_behind_Steve + John_final_distance_ahead_of_Steve) →
  v = 3.7 := 
by
  intros hconds heq
  exact sorry

end NUMINAMATH_GPT_steve_speed_l870_87077


namespace NUMINAMATH_GPT_parallel_lines_l870_87073

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, x + a * y - (2 * a + 2) = 0 ∧ a * x + y - (a + 1) = 0 → (∀ x y : ℝ, (1 / a = a / 1) ∧ (1 / a ≠ (2 * -a - 2) / (1 * -a - 1)))) → a = 1 := by
sorry

end NUMINAMATH_GPT_parallel_lines_l870_87073


namespace NUMINAMATH_GPT_range_of_f_l870_87039

def diamond (x y : ℝ) := (x + y) ^ 2 - x * y

def f (a x : ℝ) := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∃ b : ℝ, ∀ x : ℝ, x > 0 → f a x > b :=
sorry

end NUMINAMATH_GPT_range_of_f_l870_87039


namespace NUMINAMATH_GPT_pythagorean_consecutive_numbers_unique_l870_87022

theorem pythagorean_consecutive_numbers_unique :
  ∀ (x : ℕ), (x + 2) * (x + 2) = (x + 1) * (x + 1) + x * x → x = 3 :=
by
  sorry 

end NUMINAMATH_GPT_pythagorean_consecutive_numbers_unique_l870_87022


namespace NUMINAMATH_GPT_principal_amount_l870_87068

theorem principal_amount (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (h1 : R = 4) 
  (h2 : T = 5) 
  (h3 : SI = P - 1920) 
  (h4 : SI = (P * R * T) / 100) : 
  P = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_principal_amount_l870_87068


namespace NUMINAMATH_GPT_inequality_proof_l870_87070

theorem inequality_proof (a b : ℝ) (h : a + b > 0) : 
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := 
sorry

end NUMINAMATH_GPT_inequality_proof_l870_87070


namespace NUMINAMATH_GPT_complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l870_87004

-- Definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1)}

-- Part (Ⅰ)
theorem complement_A_union_B_m_eq_4 :
  (m = 4) → compl (A ∪ B 4) = {x | x < -2} ∪ {x | x > 7} := 
by
  sorry

-- Part (Ⅱ)
theorem B_nonempty_and_subset_A_range_m :
  (∃ x, x ∈ B m) ∧ (B m ⊆ A) → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l870_87004


namespace NUMINAMATH_GPT_find_p_q_sum_l870_87099

-- Define the number of trees
def pine_trees := 2
def cedar_trees := 3
def fir_trees := 4

-- Total number of trees
def total_trees := pine_trees + cedar_trees + fir_trees

-- Number of ways to arrange the 9 trees
def total_arrangements := Nat.choose total_trees fir_trees

-- Number of ways to place fir trees so no two are adjacent
def valid_arrangements := Nat.choose (pine_trees + cedar_trees + 1) fir_trees

-- Desired probability in its simplest form
def probability := valid_arrangements / total_arrangements

-- Denominator and numerator of the simplified fraction
def num := 5
def den := 42

-- Statement to prove that the probability is 5/42
theorem find_p_q_sum : (num + den) = 47 := by
  sorry

end NUMINAMATH_GPT_find_p_q_sum_l870_87099


namespace NUMINAMATH_GPT_solve_quadratic_completing_square_l870_87078

theorem solve_quadratic_completing_square :
  ∃ (a b c : ℤ), a > 0 ∧ 25 * a * a + 30 * b - 45 = (a * x + b)^2 - c ∧
                 a + b + c = 62 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_completing_square_l870_87078


namespace NUMINAMATH_GPT_cubic_roots_reciprocal_squares_sum_l870_87021

-- Define the roots a, b, and c
variables (a b c : ℝ)

-- Define the given cubic equation conditions
variables (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6)

-- Define the target statement
theorem cubic_roots_reciprocal_squares_sum :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_reciprocal_squares_sum_l870_87021


namespace NUMINAMATH_GPT_shaded_region_area_l870_87024

noncomputable def area_shaded_region (r_small r_large : ℝ) (A B : ℝ × ℝ) : ℝ := 
  let pi := Real.pi
  let sqrt_5 := Real.sqrt 5
  (5 * pi / 2) - (4 * sqrt_5)

theorem shaded_region_area : 
  ∀ (r_small r_large : ℝ) (A B : ℝ × ℝ), 
  r_small = 2 → 
  r_large = 3 → 
  (A = (0, 0)) → 
  (B = (4, 0)) → 
  area_shaded_region r_small r_large A B = (5 * Real.pi / 2) - (4 * Real.sqrt 5) := 
by
  intros r_small r_large A B h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end NUMINAMATH_GPT_shaded_region_area_l870_87024


namespace NUMINAMATH_GPT_towel_price_40_l870_87005

/-- Let x be the price of each towel bought second by the woman. 
    Given that she bought 3 towels at Rs. 100 each, 5 towels at x Rs. each, 
    and 2 towels at Rs. 550 each, and the average price of the towels was Rs. 160,
    we need to prove that x equals 40. -/
theorem towel_price_40 
    (x : ℝ)
    (h_avg_price : (300 + 5 * x + 1100) / 10 = 160) : 
    x = 40 :=
sorry

end NUMINAMATH_GPT_towel_price_40_l870_87005


namespace NUMINAMATH_GPT_sin_cos_expression_l870_87046

noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def cos_15 := Real.cos (Real.pi / 12)
noncomputable def cos_225 := Real.cos (5 * Real.pi / 4)
noncomputable def sin_15 := Real.sin (Real.pi / 12)

theorem sin_cos_expression :
  sin_45 * cos_15 + cos_225 * sin_15 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_expression_l870_87046


namespace NUMINAMATH_GPT_sphere_volume_increase_l870_87008

theorem sphere_volume_increase 
  (r : ℝ) 
  (S : ℝ := 4 * Real.pi * r^2) 
  (V : ℝ := (4/3) * Real.pi * r^3)
  (k : ℝ := 2) 
  (h : 4 * S = 4 * Real.pi * (k * r)^2) : 
  ((4/3) * Real.pi * (2 * r)^3) = 8 * V := 
by
  sorry

end NUMINAMATH_GPT_sphere_volume_increase_l870_87008


namespace NUMINAMATH_GPT_sum_A_C_l870_87000

theorem sum_A_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : B + C = 340) (h3 : C = 40) : A + C = 200 :=
by
  sorry

end NUMINAMATH_GPT_sum_A_C_l870_87000


namespace NUMINAMATH_GPT_find_x_l870_87029

theorem find_x : ∃ x : ℕ, 6 * 2^x = 2048 ∧ x = 10 := by
  sorry

end NUMINAMATH_GPT_find_x_l870_87029


namespace NUMINAMATH_GPT_min_expression_value_l870_87025

open Real

theorem min_expression_value : ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

end NUMINAMATH_GPT_min_expression_value_l870_87025


namespace NUMINAMATH_GPT_ratio_a_c_l870_87071

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
by
  sorry

end NUMINAMATH_GPT_ratio_a_c_l870_87071


namespace NUMINAMATH_GPT_diorama_factor_l870_87053

theorem diorama_factor (P B factor : ℕ) (h1 : P + B = 67) (h2 : B = P * factor - 5) (h3 : B = 49) : factor = 3 :=
by
  sorry

end NUMINAMATH_GPT_diorama_factor_l870_87053


namespace NUMINAMATH_GPT_walter_exceptional_days_l870_87090

variable (b w : Nat)

-- Definitions of the conditions
def total_days (b w : Nat) : Prop := b + w = 10
def total_earnings (b w : Nat) : Prop := 3 * b + 6 * w = 42

-- The theorem states that given the conditions, the number of days Walter did his chores exceptionally well is 4
theorem walter_exceptional_days : total_days b w → total_earnings b w → w = 4 := 
  by
    sorry

end NUMINAMATH_GPT_walter_exceptional_days_l870_87090


namespace NUMINAMATH_GPT_least_number_of_shoes_needed_on_island_l870_87027

def number_of_inhabitants : ℕ := 10000
def percentage_one_legged : ℕ := 5
def shoes_needed (N : ℕ) : ℕ :=
  let one_legged := (percentage_one_legged * N) / 100
  let two_legged := N - one_legged
  let barefooted_two_legged := two_legged / 2
  let shoes_for_one_legged := one_legged
  let shoes_for_two_legged := (two_legged - barefooted_two_legged) * 2
  shoes_for_one_legged + shoes_for_two_legged

theorem least_number_of_shoes_needed_on_island :
  shoes_needed number_of_inhabitants = 10000 :=
sorry

end NUMINAMATH_GPT_least_number_of_shoes_needed_on_island_l870_87027


namespace NUMINAMATH_GPT_mass_of_man_l870_87054

-- Definitions based on problem conditions
def boat_length : ℝ := 8
def boat_breadth : ℝ := 3
def sinking_height : ℝ := 0.01
def water_density : ℝ := 1000

-- Mass of the man to be proven
theorem mass_of_man : boat_density * (boat_length * boat_breadth * sinking_height) = 240 :=
by
  sorry

end NUMINAMATH_GPT_mass_of_man_l870_87054


namespace NUMINAMATH_GPT_mark_owes_820_l870_87076

-- Definitions of the problem conditions
def base_fine : ℕ := 50
def over_speed_fine (mph_over : ℕ) : ℕ := mph_over * 2
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def lawyer_cost_per_hour : ℕ := 80
def lawyer_hours : ℕ := 3

-- Calculation of the total fine
def total_fine (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let mph_over := actual_speed - speed_limit
  let additional_fine := over_speed_fine mph_over
  let fine_before_multipliers := base_fine + additional_fine
  let fine_after_multipliers := fine_before_multipliers * school_zone_multiplier
  fine_after_multipliers

-- Calculation of the total costs
def total_costs (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let fine := total_fine speed_limit actual_speed
  fine + court_costs + (lawyer_cost_per_hour * lawyer_hours)

theorem mark_owes_820 : total_costs 30 75 = 820 := 
by
  sorry

end NUMINAMATH_GPT_mark_owes_820_l870_87076


namespace NUMINAMATH_GPT_fraction_white_surface_area_l870_87087

theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let black_faces_corners := 6
  let black_faces_centers := 6
  let black_faces_total := 12
  let white_faces_total := total_surface_area - black_faces_total
  white_faces_total / total_surface_area = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_white_surface_area_l870_87087


namespace NUMINAMATH_GPT_jellybean_avg_increase_l870_87019

noncomputable def avg_increase_jellybeans 
  (avg_original : ℕ) (num_bags_original : ℕ) (num_jellybeans_new_bag : ℕ) : ℕ :=
  let total_original := avg_original * num_bags_original
  let total_new := total_original + num_jellybeans_new_bag
  let num_bags_new := num_bags_original + 1
  let avg_new := total_new / num_bags_new
  avg_new - avg_original

theorem jellybean_avg_increase :
  avg_increase_jellybeans 117 34 362 = 7 := by
  let total_original := 117 * 34
  let total_new := total_original + 362
  let num_bags_new := 34 + 1
  let avg_new := total_new / num_bags_new
  let increase := avg_new - 117
  have h1 : total_original = 3978 := by norm_num
  have h2 : total_new = 4340 := by norm_num
  have h3 : num_bags_new = 35 := by norm_num
  have h4 : avg_new = 124 := by norm_num
  have h5 : increase = 7 := by norm_num
  exact h5

end NUMINAMATH_GPT_jellybean_avg_increase_l870_87019


namespace NUMINAMATH_GPT_bananas_to_apples_l870_87041

-- Definitions based on conditions
def bananas := ℕ
def oranges := ℕ
def apples := ℕ

-- Condition 1: 3/4 of 16 bananas are worth 12 oranges
def condition1 : Prop := 3 / 4 * 16 = 12

-- Condition 2: price of one banana equals the price of two apples
def price_equiv_banana_apple : Prop := 1 = 2

-- Proof: 1/3 of 9 bananas are worth 6 apples
theorem bananas_to_apples 
  (c1: condition1)
  (c2: price_equiv_banana_apple) : 1 / 3 * 9 * 2 = 6 :=
by sorry

end NUMINAMATH_GPT_bananas_to_apples_l870_87041


namespace NUMINAMATH_GPT_firecrackers_defective_fraction_l870_87037

theorem firecrackers_defective_fraction (initial_total good_remaining confiscated : ℕ) 
(h_initial : initial_total = 48) 
(h_confiscated : confiscated = 12) 
(h_good_remaining : good_remaining = 15) : 
(initial_total - confiscated - 2 * good_remaining) / (initial_total - confiscated) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_firecrackers_defective_fraction_l870_87037


namespace NUMINAMATH_GPT_triangle_type_l870_87063

theorem triangle_type (a b c : ℝ) (A B C : ℝ) (h1 : A = 30) (h2 : a = 2 * b ∨ b = 2 * c ∨ c = 2 * a) :
  (C > 90 ∨ B > 90) ∨ C = 90 :=
sorry

end NUMINAMATH_GPT_triangle_type_l870_87063


namespace NUMINAMATH_GPT_article_large_font_pages_l870_87040

theorem article_large_font_pages (L S : ℕ) 
  (pages_eq : L + S = 21) 
  (words_eq : 1800 * L + 2400 * S = 48000) : 
  L = 4 := 
by 
  sorry

end NUMINAMATH_GPT_article_large_font_pages_l870_87040


namespace NUMINAMATH_GPT_papaya_production_l870_87074

theorem papaya_production (P : ℕ)
  (h1 : 2 * P + 3 * 20 = 80) :
  P = 10 := 
by sorry

end NUMINAMATH_GPT_papaya_production_l870_87074


namespace NUMINAMATH_GPT_Tom_has_38_photos_l870_87043

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end NUMINAMATH_GPT_Tom_has_38_photos_l870_87043


namespace NUMINAMATH_GPT_part1_part2_l870_87047

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 1) / Real.exp x
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := (-a * x^2 + (2 * a - b) * x + b - 1) / Real.exp x

theorem part1 (a b : ℝ) (h : f a b (-1) + f' a b (-1) = 0) : b = 2 * a :=
sorry

theorem part2 (a : ℝ) (h : a ≤ 1 / 2) (x : ℝ) : f a (2 * a) (abs x) ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l870_87047


namespace NUMINAMATH_GPT_total_amount_shared_l870_87075

-- Given John (J), Jose (Jo), and Binoy (B) and their proportion of money
variables (J Jo B : ℕ)
-- John received 1440 Rs.
variable (John_received : J = 1440)

-- The ratio of their shares is 2:4:6
axiom ratio_condition : J * 2 = Jo * 4 ∧ J * 2 = B * 6

-- The target statement to prove
theorem total_amount_shared : J + Jo + B = 8640 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_amount_shared_l870_87075


namespace NUMINAMATH_GPT_largest_three_digit_number_l870_87083

theorem largest_three_digit_number :
  ∃ n k m : ℤ, 100 ≤ n ∧ n < 1000 ∧ n = 7 * k + 2 ∧ n = 4 * m + 1 ∧ n = 989 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_number_l870_87083


namespace NUMINAMATH_GPT_perfect_square_impossible_l870_87097
noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

theorem perfect_square_impossible (a b c : ℕ) (a_positive : a > 0) (b_positive : b > 0) (c_positive : c > 0) :
  ¬ (is_perfect_square (a^2 + b + c) ∧ is_perfect_square (b^2 + c + a) ∧ is_perfect_square (c^2 + a + b)) :=
sorry

end NUMINAMATH_GPT_perfect_square_impossible_l870_87097


namespace NUMINAMATH_GPT_yogurt_combinations_l870_87094

theorem yogurt_combinations (flavors toppings : ℕ) (hflavors : flavors = 5) (htoppings : toppings = 8) :
  (flavors * Nat.choose toppings 3 = 280) :=
by
  rw [hflavors, htoppings]
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l870_87094


namespace NUMINAMATH_GPT_sequence_formula_l870_87051

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 33) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l870_87051


namespace NUMINAMATH_GPT_train_cross_time_l870_87085

def length_of_train : ℕ := 120 -- the train is 120 m long
def speed_of_train_km_hr : ℕ := 45 -- the train's speed in km/hr
def length_of_bridge : ℕ := 255 -- the bridge is 255 m long

def train_speed_m_s : ℕ := speed_of_train_km_hr * (1000 / 3600)

def total_distance : ℕ := length_of_train + length_of_bridge

def time_to_cross_bridge (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_cross_time :
  time_to_cross_bridge total_distance train_speed_m_s = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_l870_87085


namespace NUMINAMATH_GPT_arithmetic_sequence_term_count_l870_87098

def first_term : ℕ := 5
def common_difference : ℕ := 3
def last_term : ℕ := 203

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 67 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_count_l870_87098


namespace NUMINAMATH_GPT_number_problem_l870_87061

theorem number_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 34) / 10 = 2 := by
  sorry

end NUMINAMATH_GPT_number_problem_l870_87061


namespace NUMINAMATH_GPT_exists_n_satisfying_condition_l870_87038

-- Definition of the divisor function d(n)
def d (n : ℕ) : ℕ := Nat.divisors n |>.card

-- Theorem statement
theorem exists_n_satisfying_condition : ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 :=
by
  sorry

end NUMINAMATH_GPT_exists_n_satisfying_condition_l870_87038


namespace NUMINAMATH_GPT_find_a_find_a_plus_c_l870_87026

-- Define the triangle with given sides and angles
variables (A B C : ℝ) (a b c S : ℝ)
  (h_cosB : cos B = 4/5)
  (h_b : b = 2)
  (h_area : S = 3)

-- Prove the value of the side 'a' when angle A is π/6
theorem find_a (h_A : A = Real.pi / 6) : a = 5 / 3 := 
  sorry

-- Prove the sum of sides 'a' and 'c' when the area of the triangle is 3
theorem find_a_plus_c (h_ac : a * c = 10) : a + c = 2 * Real.sqrt 10 :=
  sorry

end NUMINAMATH_GPT_find_a_find_a_plus_c_l870_87026


namespace NUMINAMATH_GPT_average_salary_l870_87018

theorem average_salary (avg_officer_salary avg_nonofficer_salary num_officers num_nonofficers : ℕ) (total_salary total_employees : ℕ) : 
  avg_officer_salary = 430 → 
  avg_nonofficer_salary = 110 → 
  num_officers = 15 → 
  num_nonofficers = 465 → 
  total_salary = avg_officer_salary * num_officers + avg_nonofficer_salary * num_nonofficers → 
  total_employees = num_officers + num_nonofficers → 
  total_salary / total_employees = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_average_salary_l870_87018


namespace NUMINAMATH_GPT_thirtieth_change_month_is_february_l870_87016

def months_in_year := 12

def months_per_change := 7

def first_change_month := 3 -- March (if we assume January = 1, February = 2, etc.)

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + months_per_change * (n - 1)) % months_in_year

theorem thirtieth_change_month_is_february :
  nth_change_month 30 = 2 := -- February (if we assume January = 1, February = 2, etc.)
by 
  sorry

end NUMINAMATH_GPT_thirtieth_change_month_is_february_l870_87016


namespace NUMINAMATH_GPT_find_number_l870_87014

theorem find_number (x : ℝ) 
  (h : (28 + x / 69) * 69 = 1980) :
  x = 1952 :=
sorry

end NUMINAMATH_GPT_find_number_l870_87014


namespace NUMINAMATH_GPT_cubic_difference_l870_87042

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : a^3 - b^3 = 448 :=
by
  sorry

end NUMINAMATH_GPT_cubic_difference_l870_87042


namespace NUMINAMATH_GPT_find_ratio_of_three_numbers_l870_87095

noncomputable def ratio_of_three_numbers (A B C : ℝ) : Prop :=
  (A + B + C) / (A + B - C) = 4 / 3 ∧
  (A + B) / (B + C) = 7 / 6

theorem find_ratio_of_three_numbers (A B C : ℝ) (h₁ : ratio_of_three_numbers A B C) :
  A / C = 2 ∧ B / C = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_of_three_numbers_l870_87095
