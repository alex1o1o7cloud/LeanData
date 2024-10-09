import Mathlib

namespace simplify_fraction_l745_74582

theorem simplify_fraction (a : ℕ) (h : a = 5) : (15 * a^4) / (75 * a^3) = 1 := 
by
  sorry

end simplify_fraction_l745_74582


namespace students_per_bench_l745_74569

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_l745_74569


namespace distance_to_lateral_face_l745_74528

theorem distance_to_lateral_face 
  (height : ℝ) 
  (angle : ℝ) 
  (h_height : height = 6 * Real.sqrt 6)
  (h_angle : angle = Real.pi / 4) : 
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 30 / 5 :=
by
  sorry

end distance_to_lateral_face_l745_74528


namespace marsha_pay_per_mile_l745_74547

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l745_74547


namespace parabola_vertex_l745_74587

theorem parabola_vertex :
  ∃ (x y : ℤ), ((∀ x : ℝ, 2 * x^2 - 4 * x - 7 = y) ∧ x = 1 ∧ y = -9) := 
sorry

end parabola_vertex_l745_74587


namespace number_of_students_l745_74574

def candiesPerStudent : ℕ := 2
def totalCandies : ℕ := 18
def expectedStudents : ℕ := 9

theorem number_of_students :
  totalCandies / candiesPerStudent = expectedStudents :=
sorry

end number_of_students_l745_74574


namespace number_of_cars_washed_l745_74539

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end number_of_cars_washed_l745_74539


namespace milk_production_days_l745_74518

theorem milk_production_days (y : ℕ) :
  (y + 4) * (y + 2) * (y + 6) / (y * (y + 3) * (y + 4)) = y * (y + 3) * (y + 6) / ((y + 2) * (y + 4)) :=
sorry

end milk_production_days_l745_74518


namespace sector_angle_l745_74559

-- Defining the conditions
def perimeter (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1 / 2) * l * r = 4

-- Lean theorem statement
theorem sector_angle (r l θ : ℝ) :
  (perimeter r l) → (area r l) → (θ = l / r) → |θ| = 2 :=
by sorry

end sector_angle_l745_74559


namespace correctly_calculated_value_l745_74585

theorem correctly_calculated_value (n : ℕ) (h : 5 * n = 30) : n / 6 = 1 :=
sorry

end correctly_calculated_value_l745_74585


namespace probability_first_spade_last_ace_l745_74571

-- Define the problem parameters
def standard_deck : ℕ := 52
def spades_count : ℕ := 13
def aces_count : ℕ := 4
def ace_of_spades : ℕ := 1

-- Probability of drawing a spade but not an ace as the first card
def prob_spade_not_ace_first : ℚ := 12 / 52

-- Probability of drawing any of the four aces among the two remaining cards
def prob_ace_among_two_remaining : ℚ := 4 / 50

-- Probability of drawing the ace of spades as the first card
def prob_ace_of_spades_first : ℚ := 1 / 52

-- Probability of drawing one of three remaining aces among two remaining cards
def prob_three_aces_among_two_remaining : ℚ := 3 / 50

-- Combined probability according to the cases
def final_probability : ℚ := (prob_spade_not_ace_first * prob_ace_among_two_remaining) + (prob_ace_of_spades_first * prob_three_aces_among_two_remaining)

-- The theorem stating that the computed probability matches the expected result
theorem probability_first_spade_last_ace : final_probability = 51 / 2600 := 
  by
    -- inserting proof steps here would solve the theorem
    sorry

end probability_first_spade_last_ace_l745_74571


namespace smallest_number_mod_l745_74514

theorem smallest_number_mod (x : ℕ) :
  (x % 2 = 1) → (x % 3 = 2) → x = 5 :=
by
  sorry

end smallest_number_mod_l745_74514


namespace smallest_positive_debt_l745_74502

theorem smallest_positive_debt :
  ∃ (D : ℕ) (p g : ℤ), 0 < D ∧ D = 350 * p + 240 * g ∧ D = 10 := sorry

end smallest_positive_debt_l745_74502


namespace equalize_costs_l745_74527

variable (L B C : ℝ)
variable (h1 : L < B)
variable (h2 : B < C)

theorem equalize_costs : (B + C - 2 * L) / 3 = ((L + B + C) / 3 - L) :=
by sorry

end equalize_costs_l745_74527


namespace price_reduction_after_markup_l745_74581

theorem price_reduction_after_markup (p : ℝ) (x : ℝ) (h₁ : 0 < p) (h₂ : 0 ≤ x ∧ x < 1) :
  (1.25 : ℝ) * (1 - x) = 1 → x = 0.20 := by
  sorry

end price_reduction_after_markup_l745_74581


namespace find_x_in_equation_l745_74562

theorem find_x_in_equation :
  ∃ x : ℝ, 2.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2000.0000000000002 ∧ x = 0.3 :=
by 
  sorry

end find_x_in_equation_l745_74562


namespace subtract_vectors_l745_74556

def vec_a : ℤ × ℤ × ℤ := (5, -3, 2)
def vec_b : ℤ × ℤ × ℤ := (-2, 4, 1)
def vec_result : ℤ × ℤ × ℤ := (9, -11, 0)

theorem subtract_vectors :
  vec_a - 2 • vec_b = vec_result :=
by sorry

end subtract_vectors_l745_74556


namespace triangular_number_difference_l745_74568

-- Definition of the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Theorem stating the problem
theorem triangular_number_difference :
  triangular_number 2010 - triangular_number 2008 = 4019 :=
by
  sorry

end triangular_number_difference_l745_74568


namespace simplify_and_evaluate_expression_l745_74593

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  (5 * x ^ 2 - 2 * (3 * y ^ 2 + 6 * x) + (2 * y ^ 2 - 5 * x ^ 2)) = 8 :=
by
  sorry

end simplify_and_evaluate_expression_l745_74593


namespace AdvancedVowelSoup_l745_74506

noncomputable def AdvancedVowelSoup.sequence_count : ℕ :=
  let total_sequences := 7^7
  let vowel_only_sequences := 5^7
  let consonant_only_sequences := 2^7
  total_sequences - vowel_only_sequences - consonant_only_sequences

theorem AdvancedVowelSoup.valid_sequences : AdvancedVowelSoup.sequence_count = 745290 := by
  sorry

end AdvancedVowelSoup_l745_74506


namespace average_marks_110_l745_74588

def marks_problem (P C M B E : ℕ) : Prop :=
  (C = P + 90) ∧
  (M = P + 140) ∧
  (P + C + M + B + E = P + 350) ∧
  (B = E) ∧
  (P ≥ 40) ∧
  (C ≥ 40) ∧
  (M ≥ 40) ∧
  (B ≥ 40) ∧
  (E ≥ 40)

theorem average_marks_110 (P C M B E : ℕ) (h : marks_problem P C M B E) : 
    (B + C + M) / 3 = 110 := 
by
  sorry

end average_marks_110_l745_74588


namespace number_of_students_and_average_output_l745_74529

theorem number_of_students_and_average_output 
  (total_potatoes : ℕ)
  (days : ℕ)
  (x y : ℕ) 
  (h1 : total_potatoes = 45715) 
  (h2 : days = 5)
  (h3 : x * y * days = total_potatoes) : 
  x = 41 ∧ y = 223 :=
by
  sorry

end number_of_students_and_average_output_l745_74529


namespace minimum_k_l745_74523

theorem minimum_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  a 1 = (1/2) ∧ (∀ n, 2 * a (n + 1) + S n = 0) ∧ (∀ n, S n ≤ k) → k = (1/2) :=
sorry

end minimum_k_l745_74523


namespace proof_2_fx_minus_11_eq_f_x_minus_d_l745_74557

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 2

theorem proof_2_fx_minus_11_eq_f_x_minus_d :
  2 * (f 5) - 11 = f (5 - d) := by
  sorry

end proof_2_fx_minus_11_eq_f_x_minus_d_l745_74557


namespace Tom_initial_investment_l745_74512

noncomputable def Jose_investment : ℝ := 45000
noncomputable def Jose_investment_time : ℕ := 10
noncomputable def total_profit : ℝ := 36000
noncomputable def Jose_share : ℝ := 20000
noncomputable def Tom_share : ℝ := total_profit - Jose_share
noncomputable def Tom_investment_time : ℕ := 12
noncomputable def proportion_Tom : ℝ := (4 : ℝ) / 5
noncomputable def Tom_expected_investment : ℝ := 6000

theorem Tom_initial_investment (T : ℝ) (h1 : Jose_investment = 45000)
                               (h2 : Jose_investment_time = 10)
                               (h3 : total_profit = 36000)
                               (h4 : Jose_share = 20000)
                               (h5 : Tom_investment_time = 12)
                               (h6 : Tom_share = 16000)
                               (h7 : proportion_Tom = (4 : ℝ) / 5)
                               : T = Tom_expected_investment :=
by
  sorry

end Tom_initial_investment_l745_74512


namespace largest_base5_eq_124_l745_74520

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l745_74520


namespace problem_statement_l745_74525

noncomputable def tangent_sum_formula (x y : ℝ) : ℝ :=
  (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)

theorem problem_statement
  (α β : ℝ)
  (hαβ1 : 0 < α ∧ α < π)
  (hαβ2 : 0 < β ∧ β < π)
  (h1 : Real.tan (α - β) = 1 / 2)
  (h2 : Real.tan β = - 1 / 7)
  : 2 * α - β = - (3 * π / 4) :=
sorry

end problem_statement_l745_74525


namespace rational_inequalities_l745_74535

theorem rational_inequalities (a b c d : ℚ)
  (h : a^3 - 2005 = b^3 + 2027 ∧ b^3 + 2027 = c^3 - 2822 ∧ c^3 - 2822 = d^3 + 2820) :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end rational_inequalities_l745_74535


namespace solve_abs_linear_eq_l745_74595

theorem solve_abs_linear_eq (x : ℝ) : (|x - 1| + x - 1 = 0) ↔ (x ≤ 1) :=
sorry

end solve_abs_linear_eq_l745_74595


namespace ms_smith_books_divided_l745_74515

theorem ms_smith_books_divided (books_for_girls : ℕ) (girls boys : ℕ) (books_per_girl : ℕ)
  (h1 : books_for_girls = 225)
  (h2 : girls = 15)
  (h3 : boys = 10)
  (h4 : books_for_girls / girls = books_per_girl)
  (h5 : books_per_girl * boys + books_for_girls = 375) : 
  books_for_girls / girls * (girls + boys) = 375 := 
by
  sorry

end ms_smith_books_divided_l745_74515


namespace trigonometric_identity_l745_74522

theorem trigonometric_identity (α : ℝ) :
    (1 / Real.sin (-α) - Real.sin (Real.pi + α)) /
    (1 / Real.cos (3 * Real.pi - α) + Real.cos (2 * Real.pi - α)) =
    1 / Real.tan α ^ 3 :=
    sorry

end trigonometric_identity_l745_74522


namespace team_selection_ways_l745_74550

theorem team_selection_ways :
  let ways (n k : ℕ) := Nat.choose n k
  (ways 6 3) * (ways 6 3) = 400 := 
by
  let ways := Nat.choose
  -- Proof is omitted
  sorry

end team_selection_ways_l745_74550


namespace percent_defective_units_shipped_l745_74592

theorem percent_defective_units_shipped (h1 : 8 / 100 * 4 / 100 = 32 / 10000) :
  (32 / 10000) * 100 = 0.32 := 
sorry

end percent_defective_units_shipped_l745_74592


namespace ex1_l745_74511

theorem ex1 (a b : ℕ) (h₀ : a = 3) (h₁ : b = 4) : ∃ n : ℕ, 3^(7*a + b) = n^7 :=
by
  use 27
  sorry

end ex1_l745_74511


namespace polynomial_coeffs_l745_74598

theorem polynomial_coeffs (a b c d e f : ℤ) :
  (((2 : ℤ) * x - 1) ^ 5 = a * x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) →
  (a + b + c + d + e + f = 1) ∧ 
  (b + c + d + e = -30) ∧
  (a + c + e = 122) :=
by
  intro h
  sorry  -- Proof omitted

end polynomial_coeffs_l745_74598


namespace count_integers_divis_by_8_l745_74565

theorem count_integers_divis_by_8 : 
  ∃ k : ℕ, k = 49 ∧ ∀ n : ℕ, 2 ≤ n ∧ n ≤ 80 → (∃ m : ℤ, (n-1) * n * (n+1) = 8 * m) ↔ (∃ m : ℕ, m ≤ k) :=
by 
  sorry

end count_integers_divis_by_8_l745_74565


namespace sin_identity_cos_identity_l745_74594

-- Define the condition that alpha + beta + gamma = 180 degrees.
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

-- Prove that sin 4α + sin 4β + sin 4γ = -4 sin 2α sin 2β sin 2γ.
theorem sin_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.sin (4 * α) + Real.sin (4 * β) + Real.sin (4 * γ) = -4 * Real.sin (2 * α) * Real.sin (2 * β) * Real.sin (2 * γ) := by
  sorry

-- Prove that cos 4α + cos 4β + cos 4γ = 4 cos 2α cos 2β cos 2γ - 1.
theorem cos_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.cos (4 * α) + Real.cos (4 * β) + Real.cos (4 * γ) = 4 * Real.cos (2 * α) * Real.cos (2 * β) * Real.cos (2 * γ) - 1 := by
  sorry

end sin_identity_cos_identity_l745_74594


namespace find_a_l745_74531

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l745_74531


namespace product_of_two_numbers_l745_74548

theorem product_of_two_numbers (x y : ℝ) (h₁ : x + y = 23) (h₂ : x^2 + y^2 = 289) : x * y = 120 := by
  sorry

end product_of_two_numbers_l745_74548


namespace initial_books_count_l745_74538

theorem initial_books_count (x : ℕ) (h : x + 10 = 48) : x = 38 := 
by
  sorry

end initial_books_count_l745_74538


namespace function_fixed_point_l745_74516

theorem function_fixed_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
by
  sorry

end function_fixed_point_l745_74516


namespace max_n_for_factorable_poly_l745_74545

/-- 
  Let p(x) = 6x^2 + n * x + 48 be a quadratic polynomial.
  We want to find the maximum value of n such that p(x) can be factored into
  the product of two linear factors with integer coefficients.
-/
theorem max_n_for_factorable_poly :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * B + A = n → A * B = 48) ∧ n = 289 := 
by
  sorry

end max_n_for_factorable_poly_l745_74545


namespace solution_set_inequality_l745_74554

theorem solution_set_inequality (x : ℝ) : 
  x * (x - 1) ≥ x ↔ x ≤ 0 ∨ x ≥ 2 := 
sorry

end solution_set_inequality_l745_74554


namespace smallest_number_greater_than_500000_has_56_positive_factors_l745_74599

/-- Let n be the smallest number greater than 500,000 
    that is the product of the first four terms of both
    an arithmetic sequence and a geometric sequence.
    Prove that n has 56 positive factors. -/
theorem smallest_number_greater_than_500000_has_56_positive_factors :
  ∃ n : ℕ,
    (500000 < n) ∧
    (∀ a d b r, a > 0 → d > 0 → b > 0 → r > 0 →
      n = (a * (a + d) * (a + 2 * d) * (a + 3 * d)) ∧
          n = (b * (b * r) * (b * r^2) * (b * r^3))) ∧
    (n.factors.length = 56) :=
by sorry

end smallest_number_greater_than_500000_has_56_positive_factors_l745_74599


namespace book_pages_l745_74509

noncomputable def totalPages := 240

theorem book_pages : 
  ∀ P : ℕ, 
    (1 / 2) * P + (1 / 4) * P + (1 / 6) * P + 20 = P → 
    P = totalPages :=
by
  intro P
  intros h
  sorry

end book_pages_l745_74509


namespace andrea_rhinestones_ratio_l745_74552

theorem andrea_rhinestones_ratio :
  (∃ (B : ℕ), B = 45 - (1 / 5 * 45) - 21) →
  (1/5 * 45 : ℕ) + B + 21 = 45 →
  (B : ℕ) / 45 = 1 / 3 := 
sorry

end andrea_rhinestones_ratio_l745_74552


namespace find_minimum_fuse_length_l745_74577

def safeZone : ℝ := 70
def fuseBurningSpeed : ℝ := 0.112
def personSpeed : ℝ := 7
def minimumFuseLength : ℝ := 1.1

theorem find_minimum_fuse_length (x : ℝ) (h1 : x ≥ 0):
  (safeZone / personSpeed) * fuseBurningSpeed ≤ x :=
by
  sorry

end find_minimum_fuse_length_l745_74577


namespace inequality_for_positive_real_numbers_l745_74563

theorem inequality_for_positive_real_numbers 
  (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (a / (b + 2 * c + 3 * d) + 
   b / (c + 2 * d + 3 * a) + 
   c / (d + 2 * a + 3 * b) + 
   d / (a + 2 * b + 3 * c)) ≥ (2 / 3) :=
by
  sorry

end inequality_for_positive_real_numbers_l745_74563


namespace total_books_l745_74558

-- Lean 4 Statement
theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) (albert_books : ℝ) (total_books : ℝ) 
  (h1 : stu_books = 9) 
  (h2 : albert_ratio = 4.5) 
  (h3 : albert_books = stu_books * albert_ratio) 
  (h4 : total_books = stu_books + albert_books) : 
  total_books = 49.5 := 
sorry

end total_books_l745_74558


namespace possible_value_of_a_l745_74589

variable {a b x : ℝ}

theorem possible_value_of_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x :=
sorry

end possible_value_of_a_l745_74589


namespace find_x_l745_74533

noncomputable def arctan := Real.arctan

theorem find_x :
  (∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 5) + arctan (1 / x) = π / 4 ∧ x = -250 / 37) :=
  sorry

end find_x_l745_74533


namespace exists_six_digit_number_l745_74578

theorem exists_six_digit_number : ∃ (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ (∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ x ∧ x < 1000 ∧ 0 ≤ y ∧ y < 1000 ∧ 6 * n = 1000 * y + x) :=
by
  sorry

end exists_six_digit_number_l745_74578


namespace circle_tangent_parabola_height_difference_l745_74504

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (point_of_tangency_left : a ≠ 0)
  (points_of_tangency_on_parabola : (2 * a^2) = (2 * (-a)^2))
  (center_y_coordinate : ∃ c , c = b)
  (circle_equation_tangent_parabola : ∀ x, (x^2 + (2*x^2 - b)^2 = r^2))
  (quartic_double_root : ∀ x, (x = a ∨ x = -a) → (x^2 + (4 - 2*b)*x^2 + b^2 - r^2 = 0)) :
  b - 2 * a^2 = 2 :=
by
  sorry

end circle_tangent_parabola_height_difference_l745_74504


namespace problem1_solution_set_problem2_a_range_l745_74551

section
variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a

-- Problem 1
theorem problem1_solution_set (h : a = 3) : {x | f x a ≤ 6} = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

def g (x : ℝ) := |2 * x - 3|

-- Problem 2
theorem problem2_a_range : ∀ a : ℝ, ∀ x : ℝ, f x a + g x ≥ 5 ↔ 4 ≤ a :=
by
  sorry
end

end problem1_solution_set_problem2_a_range_l745_74551


namespace gcd_of_8a_plus_3_and_5a_plus_2_l745_74541

theorem gcd_of_8a_plus_3_and_5a_plus_2 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 :=
by
  sorry

end gcd_of_8a_plus_3_and_5a_plus_2_l745_74541


namespace first_expression_second_expression_l745_74573

-- Define the variables
variables {a x y : ℝ}

-- Statement for the first expression
theorem first_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := sorry

-- Statement for the second expression
theorem second_expression (x y : ℝ) : (x + 3 * y) * (x - y) = x^2 + 2 * x * y - 3 * y^2 := sorry

end first_expression_second_expression_l745_74573


namespace polynomial_simplification_l745_74519

theorem polynomial_simplification :
  ∃ A B C D : ℤ,
  (∀ x : ℤ, x ≠ D → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C)
  ∧ (A + B + C + D = 8) :=
sorry

end polynomial_simplification_l745_74519


namespace fish_served_l745_74553

theorem fish_served (H E P : ℕ) 
  (h1 : H = E) (h2 : E = P) 
  (fat_herring fat_eel fat_pike total_fat : ℕ) 
  (herring_fat : fat_herring = 40) 
  (eel_fat : fat_eel = 20)
  (pike_fat : fat_pike = 30)
  (total_fat_served : total_fat = 3600) 
  (fat_eq : 40 * H + 20 * E + 30 * P = 3600) : 
  H = 40 ∧ E = 40 ∧ P = 40 := by
  sorry

end fish_served_l745_74553


namespace ralph_socks_l745_74524

theorem ralph_socks (x y z : ℕ) (h1 : x + y + z = 12) (h2 : x + 3 * y + 4 * z = 24) (h3 : 1 ≤ x) (h4 : 1 ≤ y) (h5 : 1 ≤ z) : x = 7 :=
sorry

end ralph_socks_l745_74524


namespace probability_common_letters_l745_74530

open Set

def letters_GEOMETRY : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def letters_RHYME : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

def common_letters : Finset Char := letters_GEOMETRY ∩ letters_RHYME

theorem probability_common_letters :
  (common_letters.card : ℚ) / (letters_GEOMETRY.card : ℚ) = 1 / 2 := by
  sorry

end probability_common_letters_l745_74530


namespace shelves_needed_l745_74513

theorem shelves_needed (initial_stock : ℕ) (additional_shipment : ℕ) (bears_per_shelf : ℕ) (total_bears : ℕ) (shelves : ℕ) :
  initial_stock = 4 → 
  additional_shipment = 10 → 
  bears_per_shelf = 7 → 
  total_bears = initial_stock + additional_shipment →
  total_bears / bears_per_shelf = shelves →
  shelves = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end shelves_needed_l745_74513


namespace choir_average_age_l745_74586

theorem choir_average_age 
  (avg_f : ℝ) (n_f : ℕ)
  (avg_m : ℝ) (n_m : ℕ)
  (h_f : avg_f = 28) 
  (h_nf : n_f = 12) 
  (h_m : avg_m = 40) 
  (h_nm : n_m = 18) 
  : (n_f * avg_f + n_m * avg_m) / (n_f + n_m) = 35.2 := 
by 
  sorry

end choir_average_age_l745_74586


namespace earth_surface_area_scientific_notation_l745_74521

theorem earth_surface_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 780000000 = a * 10^n ∧ a = 7.8 ∧ n = 8 :=
by
  sorry

end earth_surface_area_scientific_notation_l745_74521


namespace real_root_ineq_l745_74536

theorem real_root_ineq (a b : ℝ) (x₀ : ℝ) (h : x₀^4 - a * x₀^3 + 2 * x₀^2 - b * x₀ + 1 = 0) :
  a^2 + b^2 ≥ 8 :=
by
  sorry

end real_root_ineq_l745_74536


namespace no_common_factor_l745_74555

open Polynomial

theorem no_common_factor (f g : ℤ[X]) : f = X^2 + X - 1 → g = X^2 + 2 * X → ∀ d : ℤ[X], d ∣ f ∧ d ∣ g → d = 1 :=
by
  intros h1 h2 d h_dv
  rw [h1, h2] at h_dv
  -- Proof steps would go here
  sorry

end no_common_factor_l745_74555


namespace length_of_square_side_is_correct_l745_74564

noncomputable def length_of_square_side : ℚ :=
  let PQ : ℚ := 7
  let QR : ℚ := 24
  let hypotenuse := (PQ^2 + QR^2).sqrt
  (25 * 175) / (24 * 32)

theorem length_of_square_side_is_correct :
  length_of_square_side = 4375 / 768 := 
by 
  sorry

end length_of_square_side_is_correct_l745_74564


namespace pentagon_PT_length_l745_74505

theorem pentagon_PT_length (QR RS ST : ℝ) (angle_T right_angle_QRS T : Prop) (length_PT := (fun (a b : ℝ) => a + 3 * Real.sqrt b)) :
  QR = 3 →
  RS = 3 →
  ST = 3 →
  angle_T →
  right_angle_QRS →
  (angle_Q angle_R angle_S : ℝ) →
  angle_Q = 135 →
  angle_R = 135 →
  angle_S = 135 →
  ∃ (a b : ℝ), length_PT a b = 6 * Real.sqrt 2 ∧ a + b = 2 :=
by
  sorry

end pentagon_PT_length_l745_74505


namespace speed_of_man_l745_74540

/-
  Problem Statement:
  A train 100 meters long takes 6 seconds to cross a man walking at a certain speed in the direction opposite to that of the train. The speed of the train is 54.99520038396929 kmph. What is the speed of the man in kmph?
-/
 
theorem speed_of_man :
  ∀ (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_kmph : ℝ) (relative_speed_mps : ℝ),
    length_of_train = 100 →
    time_to_cross = 6 →
    speed_of_train_kmph = 54.99520038396929 →
    relative_speed_mps = length_of_train / time_to_cross →
    (relative_speed_mps - (speed_of_train_kmph * (1000 / 3600))) * (3600 / 1000) = 5.00479961403071 :=
by
  intros length_of_train time_to_cross speed_of_train_kmph relative_speed_mps
  intros h1 h2 h3 h4
  sorry

end speed_of_man_l745_74540


namespace page_problem_insufficient_information_l745_74532

theorem page_problem_insufficient_information
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (x y : ℕ)
  (O E : ℕ)
  (h1 : total_problems = 450)
  (h2 : finished_problems = 185)
  (h3 : remaining_pages = 15)
  (h4 : O + E = remaining_pages)
  (h5 : O * x + E * y = total_problems - finished_problems) :
  ∀ (x y : ℕ), O * x + E * y = 265 → x = x ∧ y = y :=
by
  sorry

end page_problem_insufficient_information_l745_74532


namespace dividend_is_correct_l745_74561

def quotient : ℕ := 36
def divisor : ℕ := 85
def remainder : ℕ := 26

theorem dividend_is_correct : divisor * quotient + remainder = 3086 := by
  sorry

end dividend_is_correct_l745_74561


namespace ram_marks_l745_74543

theorem ram_marks (total_marks : ℕ) (percentage : ℕ) (h_total : total_marks = 500) (h_percentage : percentage = 90) : 
  (percentage * total_marks / 100) = 450 := by
  sorry

end ram_marks_l745_74543


namespace tangent_line_parallel_coordinates_l745_74542

theorem tangent_line_parallel_coordinates :
  ∃ (x y : ℝ), y = x^3 + x - 2 ∧ (3 * x^2 + 1 = 4) ∧ (x, y) = (-1, -4) :=
by
  sorry

end tangent_line_parallel_coordinates_l745_74542


namespace solve_expression_l745_74544

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem solve_expression : f (g 3) - g (f 3) = -5 := by
  sorry

end solve_expression_l745_74544


namespace inequality_holds_for_all_real_l745_74583

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 6 * x + 8 ≥ -(x + 4) * (x + 6) :=
  sorry

end inequality_holds_for_all_real_l745_74583


namespace evaluate_fraction_sum_squared_l745_74590

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem evaluate_fraction_sum_squared :
  ( (1 / a + 1 / b + 1 / c + 1 / d)^2 = (11 + 2 * Real.sqrt 30) / 9 ) := 
by
  sorry

end evaluate_fraction_sum_squared_l745_74590


namespace minimum_area_of_quadrilateral_l745_74534

theorem minimum_area_of_quadrilateral
  (ABCD : Type)
  (O : Type)
  (S_ABO : ℝ)
  (S_CDO : ℝ)
  (BC : ℝ)
  (cos_angle_ADC : ℝ)
  (h1 : S_ABO = 3 / 2)
  (h2 : S_CDO = 3 / 2)
  (h3 : BC = 3 * Real.sqrt 2)
  (h4 : cos_angle_ADC = 3 / Real.sqrt 10) :
  ∃ S_ABCD : ℝ, S_ABCD = 6 :=
sorry

end minimum_area_of_quadrilateral_l745_74534


namespace arithmetic_mean_calculation_l745_74546

theorem arithmetic_mean_calculation (x : ℝ) 
  (h : (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30) : 
  x = 14.142857 :=
by
  sorry

end arithmetic_mean_calculation_l745_74546


namespace abs_difference_lt_2t_l745_74596

/-- Given conditions of absolute values with respect to t -/
theorem abs_difference_lt_2t (x y s t : ℝ) (h₁ : |x - s| < t) (h₂ : |y - s| < t) :
  |x - y| < 2 * t :=
sorry

end abs_difference_lt_2t_l745_74596


namespace ms_hatcher_total_students_l745_74570

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l745_74570


namespace shiela_drawings_l745_74526

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
  (h1 : neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
  by 
    have h : total_drawings = neighbors * drawings_per_neighbor := sorry
    rw [h1, h2] at h
    exact h
    -- Proof skipped with sorry.

end shiela_drawings_l745_74526


namespace cost_of_paving_floor_l745_74597

-- Definitions of the constants
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 400

-- Definitions of the calculated area and cost
def area : ℝ := length * width
def cost : ℝ := area * rate_per_sq_meter

-- Statement to prove
theorem cost_of_paving_floor : cost = 8250 := by
  sorry

end cost_of_paving_floor_l745_74597


namespace find_a_l745_74580

variable (a : ℝ) (h_pos : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3)

theorem find_a : a = 3 :=
by sorry

end find_a_l745_74580


namespace last_number_is_four_l745_74575

theorem last_number_is_four (a b c d e last_number : ℕ) (h_counts : a = 6 ∧ b = 12 ∧ c = 1 ∧ d = 12 ∧ e = 7)
    (h_mean : (a + b + c + d + e + last_number) / 6 = 7) : last_number = 4 := 
sorry

end last_number_is_four_l745_74575


namespace trigonometric_identity_l745_74584

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l745_74584


namespace smallest_prime_10_less_than_perfect_square_l745_74576

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_10_less_than_perfect_square :
  ∃ (a : ℕ), is_prime a ∧ (∃ (n : ℕ), a = n^2 - 10) ∧ (∀ (b : ℕ), is_prime b ∧ (∃ (m : ℕ), b = m^2 - 10) → a ≤ b) ∧ a = 71 := 
by
  sorry

end smallest_prime_10_less_than_perfect_square_l745_74576


namespace treaty_signed_on_wednesday_l745_74507

-- This function calculates the weekday after a given number of days since a known weekday.
def weekday_after (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

-- Given the problem conditions:
-- The war started on a Friday: 5th day of the week (considering Sunday as 0)
def war_start_day_of_week : ℕ := 5

-- The number of days after which the treaty was signed
def days_until_treaty : ℕ := 926

-- Expected final day (Wednesday): 3rd day of the week (considering Sunday as 0)
def treaty_day_of_week : ℕ := 3

-- The theorem to be proved:
theorem treaty_signed_on_wednesday :
  weekday_after war_start_day_of_week days_until_treaty = treaty_day_of_week :=
by
  sorry

end treaty_signed_on_wednesday_l745_74507


namespace savings_per_bagel_in_cents_l745_74560

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_per_bagel_in_cents_l745_74560


namespace smallest_number_of_digits_to_append_l745_74591

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l745_74591


namespace number_of_groups_eq_five_l745_74510

-- Define conditions
def total_eggs : ℕ := 35
def eggs_per_group : ℕ := 7

-- Statement to prove the number of groups
theorem number_of_groups_eq_five : total_eggs / eggs_per_group = 5 := by
  sorry

end number_of_groups_eq_five_l745_74510


namespace simplify_sqrt1_simplify_sqrt2_find_a_l745_74572

-- Part 1
theorem simplify_sqrt1 : ∃ m n : ℝ, m^2 + n^2 = 6 ∧ m * n = Real.sqrt 5 ∧ Real.sqrt (6 + 2 * Real.sqrt 5) = m + n :=
by sorry

-- Part 2
theorem simplify_sqrt2 : ∃ m n : ℝ, m^2 + n^2 = 5 ∧ m * n = -Real.sqrt 6 ∧ Real.sqrt (5 - 2 * Real.sqrt 6) = abs (m - n) :=
by sorry

-- Part 3
theorem find_a (a : ℝ) : (Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5) → (a = 3 ∨ a = -3) :=
by sorry

end simplify_sqrt1_simplify_sqrt2_find_a_l745_74572


namespace largest_minus_smallest_eq_13_l745_74567

theorem largest_minus_smallest_eq_13 :
  let a := (-1 : ℤ) ^ 3
  let b := (-1 : ℤ) ^ 2
  let c := -(2 : ℤ) ^ 2
  let d := (-3 : ℤ) ^ 2
  max (max a (max b c)) d - min (min a (min b c)) d = 13 := by
  sorry

end largest_minus_smallest_eq_13_l745_74567


namespace sum_of_possible_values_of_x_l745_74517

-- Conditions
def radius (x : ℝ) : ℝ := x - 2
def semiMajor (x : ℝ) : ℝ := x - 3
def semiMinor (x : ℝ) : ℝ := x + 4

-- Theorem to be proved
theorem sum_of_possible_values_of_x (x : ℝ) :
  (π * semiMajor x * semiMinor x = 2 * π * (radius x) ^ 2) →
  (x = 5 ∨ x = 4) →
  5 + 4 = 9 :=
by
  intros
  rfl

end sum_of_possible_values_of_x_l745_74517


namespace remaining_amount_is_12_l745_74579

-- Define initial amount and amount spent
def initial_amount : ℕ := 90
def amount_spent : ℕ := 78

-- Define the remaining amount after spending
def remaining_amount : ℕ := initial_amount - amount_spent

-- Theorem asserting the remaining amount is 12
theorem remaining_amount_is_12 : remaining_amount = 12 :=
by
  -- Proof omitted
  sorry

end remaining_amount_is_12_l745_74579


namespace damaged_cartons_per_customer_l745_74508

theorem damaged_cartons_per_customer (total_cartons : ℕ) (num_customers : ℕ) (total_accepted : ℕ) 
    (h1 : total_cartons = 400) (h2 : num_customers = 4) (h3 : total_accepted = 160) 
    : (total_cartons - total_accepted) / num_customers = 60 :=
by
  sorry

end damaged_cartons_per_customer_l745_74508


namespace maximum_value_of_f_inequality_holds_for_all_x_l745_74503

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem maximum_value_of_f (a : ℝ) (h : 0 ≤ a) : 
  (∀ x, f a x ≤ f a 1) → f a 1 = 3 / Real.exp 1 → a = 1 := 
by 
  sorry

theorem inequality_holds_for_all_x (b : ℝ) : 
  (∀ a ≤ 0, ∀ x, 0 ≤ x → f a x ≤ b * Real.log (x + 1)) → 1 ≤ b := 
by 
  sorry

end maximum_value_of_f_inequality_holds_for_all_x_l745_74503


namespace original_price_computer_l745_74549

noncomputable def first_store_price (P : ℝ) : ℝ := 0.94 * P

noncomputable def second_store_price (exchange_rate : ℝ) : ℝ := (920 / 0.95) * exchange_rate

theorem original_price_computer 
  (exchange_rate : ℝ)
  (h : exchange_rate = 1.1) 
  (H : (first_store_price P - second_store_price exchange_rate = 19)) :
  P = 1153.47 :=
by
  sorry

end original_price_computer_l745_74549


namespace tank_filled_in_96_minutes_l745_74566

-- conditions
def pipeA_fill_time : ℝ := 6
def pipeB_empty_time : ℝ := 24
def time_with_both_pipes_open : ℝ := 96

-- rate computations and final proof
noncomputable def pipeA_fill_rate : ℝ := 1 / pipeA_fill_time
noncomputable def pipeB_empty_rate : ℝ := 1 / pipeB_empty_time
noncomputable def net_fill_rate : ℝ := pipeA_fill_rate - pipeB_empty_rate
noncomputable def tank_filled_in_time_with_both : ℝ := time_with_both_pipes_open * net_fill_rate

theorem tank_filled_in_96_minutes (HA : pipeA_fill_time = 6) (HB : pipeB_empty_time = 24)
  (HT : time_with_both_pipes_open = 96) : tank_filled_in_time_with_both = 1 :=
by
  sorry

end tank_filled_in_96_minutes_l745_74566


namespace milk_processing_days_required_l745_74501

variable (a m x : ℝ) (n : ℝ)

theorem milk_processing_days_required
  (h1 : (n - a) * (x + m) = nx)
  (h2 : ax + (10 * a / 9) * x + (5 * a / 9) * m = 2 / 3)
  (h3 : nx = 1 / 2) :
  n = 2 * a :=
by sorry

end milk_processing_days_required_l745_74501


namespace quadratic_inequality_solution_l745_74500

def range_of_k (k : ℝ) : Prop := (k ≥ 4) ∨ (k ≤ 2)

theorem quadratic_inequality_solution (k : ℝ) (x : ℝ) (h : x = 1) :
  k^2*x^2 - 6*k*x + 8 ≥ 0 → range_of_k k := 
sorry

end quadratic_inequality_solution_l745_74500


namespace find_multiplier_l745_74537

theorem find_multiplier (x : ℝ) : 3 - 3 * x < 14 ↔ x = -3 :=
by {
  sorry
}

end find_multiplier_l745_74537
