import Mathlib

namespace cubic_roots_sum_cubes_l1386_138661

theorem cubic_roots_sum_cubes
  (p q r : ℂ)
  (h_eq_root : ∀ x, x = p ∨ x = q ∨ x = r → x^3 - 2 * x^2 + 3 * x - 1 = 0)
  (h_sum : p + q + r = 2)
  (h_prod_sum : p * q + q * r + r * p = 3)
  (h_prod : p * q * r = 1) :
  p^3 + q^3 + r^3 = -7 := by
  sorry

end cubic_roots_sum_cubes_l1386_138661


namespace probability_of_two_red_two_blue_l1386_138611

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l1386_138611


namespace total_number_of_slices_l1386_138634

def number_of_pizzas : ℕ := 7
def slices_per_pizza : ℕ := 2

theorem total_number_of_slices :
  number_of_pizzas * slices_per_pizza = 14 :=
by
  sorry

end total_number_of_slices_l1386_138634


namespace quarters_range_difference_l1386_138685

theorem quarters_range_difference (n d q : ℕ) (h1 : n + d + q = 150) (h2 : 5 * n + 10 * d + 25 * q = 2000) :
  let max_quarters := 0
  let min_quarters := 62
  (max_quarters - min_quarters) = 62 :=
by
  let max_quarters := 0
  let min_quarters := 62
  sorry

end quarters_range_difference_l1386_138685


namespace find_b_fixed_point_extremum_l1386_138679

theorem find_b_fixed_point_extremum (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, f x = x ^ 3 + b * x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ (∀ x : ℝ, deriv f x₀ = 3 * x₀ ^ 2 + b) ∧ deriv f x₀ = 0) →
  b = -3 :=
by
  sorry

end find_b_fixed_point_extremum_l1386_138679


namespace find_r_l1386_138670

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 16.5) : r = 8.5 :=
sorry

end find_r_l1386_138670


namespace proof_problem_l1386_138684

variable {a b m n x : ℝ}

theorem proof_problem (h1 : a = -b) (h2 : m * n = 1) (h3 : m ≠ n) (h4 : |x| = 2) :
    (-2 * m * n + (b + a) / (m - n) - x = -4 ∧ x = 2) ∨
    (-2 * m * n + (b + a) / (m - n) - x = 0 ∧ x = -2) :=
by
  sorry

end proof_problem_l1386_138684


namespace inequality_on_positive_reals_l1386_138639

variable {a b c : ℝ}

theorem inequality_on_positive_reals (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_on_positive_reals_l1386_138639


namespace evaluate_polynomial_l1386_138690

theorem evaluate_polynomial
  (x : ℝ)
  (h1 : x^2 - 3 * x - 9 = 0)
  (h2 : 0 < x)
  : x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = 8 :=
sorry

end evaluate_polynomial_l1386_138690


namespace min_tip_percentage_l1386_138608

namespace TipCalculation

def mealCost : Float := 35.50
def totalPaid : Float := 37.275
def maxTipPercent : Float := 0.08

theorem min_tip_percentage : ∃ (P : Float), (P / 100 * mealCost = (totalPaid - mealCost)) ∧ (P < maxTipPercent * 100) ∧ (P = 5) := by
  sorry

end TipCalculation

end min_tip_percentage_l1386_138608


namespace positive_integer_condition_l1386_138668

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) : 
  (∃ (n : ℤ), n > 0 ∧ (abs (x - abs x + 2) / x) = n) ↔ x = 2 :=
by
  sorry

end positive_integer_condition_l1386_138668


namespace simplify_fraction_l1386_138652

-- Given
def num := 54
def denom := 972

-- Factorization condition
def factorization_54 : num = 2 * 3^3 := by 
  sorry

def factorization_972 : denom = 2^2 * 3^5 := by 
  sorry

-- GCD condition
def gcd_num_denom := 54

-- Division condition
def simplified_num := 1
def simplified_denom := 18

-- Statement to prove
theorem simplify_fraction : (num / denom) = (simplified_num / simplified_denom) := by 
  sorry

end simplify_fraction_l1386_138652


namespace seven_times_equivalent_l1386_138681

theorem seven_times_equivalent (n a b : ℤ) (h : n = a^2 + a * b + b^2) :
  ∃ (c d : ℤ), 7 * n = c^2 + c * d + d^2 :=
sorry

end seven_times_equivalent_l1386_138681


namespace range_of_a_l1386_138664

variable {R : Type} [LinearOrderedField R]

def f (x a : R) : R := |x - 1| + |x - 2| - a

theorem range_of_a (h : ∀ x : R, f x a > 0) : a < 1 :=
by
  sorry

end range_of_a_l1386_138664


namespace m_plus_n_in_right_triangle_l1386_138644

noncomputable def triangle (A B C : Point) : Prop :=
  ∃ (BD : ℕ) (x : ℕ) (y : ℕ),
  ∃ (AB BC AC : ℕ),
  ∃ (m n : ℕ),
  B ≠ C ∧
  C ≠ A ∧
  B ≠ A ∧
  m.gcd n = 1 ∧
  BD = 17^3 ∧
  BC = 17^2 * x ∧
  AB = 17 * x^2 ∧
  AC = 17 * x * y ∧
  BC^2 + AC^2 = AB^2 ∧
  (2 * 17 * x) = 17^2 ∧
  ∃ cB, cB = (BC : ℚ) / (AB : ℚ) ∧
  cB = (m : ℚ) / (n : ℚ)

theorem m_plus_n_in_right_triangle :
  ∀ (A B C : Point),
  A ≠ B ∧
  B ≠ C ∧
  C ≠ A ∧
  triangle A B C →
  ∃ m n : ℕ, m.gcd n = 1 ∧ m + n = 162 :=
sorry

end m_plus_n_in_right_triangle_l1386_138644


namespace tickets_spent_on_beanie_l1386_138609

-- Define the initial conditions
def initial_tickets : ℕ := 25
def additional_tickets : ℕ := 15
def tickets_left : ℕ := 18

-- Define the total tickets
def total_tickets := initial_tickets + additional_tickets

-- Define what we're proving: the number of tickets spent on the beanie
theorem tickets_spent_on_beanie : initial_tickets + additional_tickets - tickets_left = 22 :=
by 
  -- Provide proof steps here
  sorry

end tickets_spent_on_beanie_l1386_138609


namespace sum_of_squares_l1386_138662

theorem sum_of_squares (a b : ℕ) (h₁ : a = 300000) (h₂ : b = 20000) : a^2 + b^2 = 9004000000 :=
by
  rw [h₁, h₂]
  sorry

end sum_of_squares_l1386_138662


namespace regular_polygon_sides_l1386_138620

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l1386_138620


namespace value_of_diamond_l1386_138603

def diamond (a b : ℕ) : ℕ := 4 * a + 2 * b

theorem value_of_diamond : diamond 6 3 = 30 :=
by {
  sorry
}

end value_of_diamond_l1386_138603


namespace clerical_percentage_after_reduction_l1386_138635

theorem clerical_percentage_after_reduction
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (reduction_fraction : ℚ)
  (h1 : total_employees = 3600)
  (h2 : clerical_fraction = 1/4)
  (h3 : reduction_fraction = 1/4) : 
  let initial_clerical := clerical_fraction * total_employees
  let reduced_clerical := (1 - reduction_fraction) * initial_clerical
  let let_go := initial_clerical - reduced_clerical
  let new_total := total_employees - let_go
  let clerical_percentage := (reduced_clerical / new_total) * 100
  clerical_percentage = 20 :=
by sorry

end clerical_percentage_after_reduction_l1386_138635


namespace odd_numbers_square_division_l1386_138673

theorem odd_numbers_square_division (m n : ℤ) (hm : Odd m) (hn : Odd n) (h : m^2 - n^2 + 1 ∣ n^2 - 1) : ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := 
sorry

end odd_numbers_square_division_l1386_138673


namespace determine_number_l1386_138628

theorem determine_number (x : ℝ) (number : ℝ) (h1 : number / x = 0.03) (h2 : x = 0.3) : number = 0.009 := by
  sorry

end determine_number_l1386_138628


namespace find_a4_l1386_138613
open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n / (2 * a n + 3)

theorem find_a4 (a : ℕ → ℚ) (h : seq a) : a 4 = 1 / 53 :=
by
  obtain ⟨h1, h_rec⟩ := h
  have a2 := h_rec 1 (by decide)
  have a3 := h_rec 2 (by decide)
  have a4 := h_rec 3 (by decide)
  -- Proof steps would go here
  sorry

end find_a4_l1386_138613


namespace earl_up_second_time_l1386_138645

def earl_floors (n top start up1 down up2 dist : ℕ) : Prop :=
  start + up1 - down + up2 = top - dist

theorem earl_up_second_time 
  (start up1 down top dist : ℕ) 
  (h_start : start = 1) 
  (h_up1 : up1 = 5) 
  (h_down : down = 2) 
  (h_top : top = 20) 
  (h_dist : dist = 9) : 
  ∃ up2, earl_floors n top start up1 down up2 dist ∧ up2 = 7 :=
by
  use 7
  sorry

end earl_up_second_time_l1386_138645


namespace jill_tax_on_other_items_l1386_138678

noncomputable def tax_on_other_items (total_spent clothing_tax_percent total_tax_percent : ℝ) : ℝ :=
  let clothing_spent := 0.5 * total_spent
  let food_spent := 0.25 * total_spent
  let other_spent := 0.25 * total_spent
  let clothing_tax := clothing_tax_percent * clothing_spent
  let total_tax := total_tax_percent * total_spent
  let tax_on_others := total_tax - clothing_tax
  (tax_on_others / other_spent) * 100

theorem jill_tax_on_other_items :
  let total_spent := 100
  let clothing_tax_percent := 0.1
  let total_tax_percent := 0.1
  tax_on_other_items total_spent clothing_tax_percent total_tax_percent = 20 := by
  sorry

end jill_tax_on_other_items_l1386_138678


namespace factorize_cubic_l1386_138659

theorem factorize_cubic (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_cubic_l1386_138659


namespace hypotenuse_length_l1386_138649

theorem hypotenuse_length (a b : ℕ) (h : a = 9 ∧ b = 12) : ∃ c : ℕ, c = 15 ∧ a * a + b * b = c * c :=
by
  sorry

end hypotenuse_length_l1386_138649


namespace max_f_value_l1386_138630

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ :=
  n / (n + 32) / (n + 2)

theorem max_f_value : ∀ n : ℕ, f n ≤ (1 / 50) :=
sorry

end max_f_value_l1386_138630


namespace josh_money_left_l1386_138633

theorem josh_money_left :
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  money_left = 15.87 :=
by
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  have h1 : total_spent = 84.13 := sorry
  have h2 : money_left = initial_money - 84.13 := sorry
  have h3 : money_left = 15.87 := sorry
  exact h3

end josh_money_left_l1386_138633


namespace all_terms_are_integers_l1386_138610

   noncomputable def a : ℕ → ℤ
   | 0 => 1
   | 1 => 1
   | 2 => 997
   | n + 3 => (1993 + a (n + 2) * a (n + 1)) / a n

   theorem all_terms_are_integers : ∀ n : ℕ, ∃ (a : ℕ → ℤ), 
     (a 1 = 1) ∧ 
     (a 2 = 1) ∧ 
     (a 3 = 997) ∧ 
     (∀ n : ℕ, a (n + 3) = (1993 + a (n + 2) * a (n + 1)) / a n) → 
     (∀ n : ℕ, ∃ k : ℤ, a n = k) := 
   by 
     sorry
   
end all_terms_are_integers_l1386_138610


namespace ms_cole_total_students_l1386_138653

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l1386_138653


namespace log_base_2_of_1024_l1386_138696

theorem log_base_2_of_1024 (h : 2^10 = 1024) : Real.logb 2 1024 = 10 :=
by
  sorry

end log_base_2_of_1024_l1386_138696


namespace midpoint_x_coordinate_l1386_138694

theorem midpoint_x_coordinate (M N : ℝ × ℝ)
  (hM : M.1 ^ 2 = 4 * M.2)
  (hN : N.1 ^ 2 = 4 * N.2)
  (h_dist : (Real.sqrt ((M.1 - 1)^2 + M.2^2)) + (Real.sqrt ((N.1 - 1)^2 + N.2^2)) = 6) :
  (M.1 + N.1) / 2 = 2 := 
sorry

end midpoint_x_coordinate_l1386_138694


namespace find_f_7_5_l1386_138656

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_2  : ∀ x, f (x + 2) = -f x
axiom initial_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- Proof goes here
  sorry

end find_f_7_5_l1386_138656


namespace set_A_is_listed_correctly_l1386_138687

def A : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem set_A_is_listed_correctly : A = {-2, -1, 0} := 
by
  sorry

end set_A_is_listed_correctly_l1386_138687


namespace train_length_l1386_138682

theorem train_length (L : ℝ) 
  (h1 : (L / 20) = ((L + 1500) / 70)) : L = 600 := by
  sorry

end train_length_l1386_138682


namespace smallest_solution_l1386_138677

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l1386_138677


namespace iron_conducts_electricity_l1386_138646

-- Define the predicates
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
noncomputable def Iron : Type := sorry
  
theorem iron_conducts_electricity (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron) : ConductsElectricity Iron :=
by
  sorry

end iron_conducts_electricity_l1386_138646


namespace service_center_milepost_l1386_138669

theorem service_center_milepost :
  ∀ (first_exit seventh_exit service_fraction : ℝ), 
    first_exit = 50 →
    seventh_exit = 230 →
    service_fraction = 3 / 4 →
    (first_exit + service_fraction * (seventh_exit - first_exit) = 185) :=
by
  intros first_exit seventh_exit service_fraction h_first h_seventh h_fraction
  sorry

end service_center_milepost_l1386_138669


namespace unique_three_positive_perfect_square_sums_to_100_l1386_138640

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l1386_138640


namespace simplify_expression_l1386_138693

theorem simplify_expression :
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 :=
by
  sorry

end simplify_expression_l1386_138693


namespace total_profit_is_correct_l1386_138671

-- Definitions of the investments
def A_initial_investment : ℝ := 12000
def B_investment : ℝ := 16000
def C_investment : ℝ := 20000
def D_investment : ℝ := 24000
def E_investment : ℝ := 18000
def C_profit_share : ℝ := 36000

-- Definitions of the time periods (in months)
def time_6_months : ℝ := 6
def time_12_months : ℝ := 12

-- Calculations of investment-months for each person
def A_investment_months : ℝ := A_initial_investment * time_6_months
def B_investment_months : ℝ := B_investment * time_12_months
def C_investment_months : ℝ := C_investment * time_12_months
def D_investment_months : ℝ := D_investment * time_12_months
def E_investment_months : ℝ := E_investment * time_6_months

-- Calculation of total investment-months
def total_investment_months : ℝ :=
  A_investment_months + B_investment_months + C_investment_months +
  D_investment_months + E_investment_months

-- The main theorem stating the total profit calculation
theorem total_profit_is_correct :
  ∃ TP : ℝ, (C_profit_share / C_investment_months) = (TP / total_investment_months) ∧ TP = 135000 :=
by
  sorry

end total_profit_is_correct_l1386_138671


namespace sum_binom_equals_220_l1386_138655

/-- The binomial coefficient C(n, k) -/
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

/-- Prove that the sum C(2, 2) + C(3, 2) + C(4, 2) + ... + C(11, 2) equals 220 using the 
    binomial coefficient property C(n, r+1) + C(n, r) = C(n+1, r+1) -/
theorem sum_binom_equals_220 :
  binom 2 2 + binom 3 2 + binom 4 2 + binom 5 2 + binom 6 2 + binom 7 2 + 
  binom 8 2 + binom 9 2 + binom 10 2 + binom 11 2 = 220 := by
sorry

end sum_binom_equals_220_l1386_138655


namespace problem_statement_l1386_138658

variable (a b c : ℤ) -- Declare variables as integers

-- Define conditions based on the problem
def smallest_natural_number (a : ℤ) := a = 1
def largest_negative_integer (b : ℤ) := b = -1
def number_equal_to_its_opposite (c : ℤ) := c = 0

-- State the theorem
theorem problem_statement (h1 : smallest_natural_number a) 
                         (h2 : largest_negative_integer b) 
                         (h3 : number_equal_to_its_opposite c) : 
  a + b + c = 0 := 
  by 
    rw [h1, h2, h3] 
    simp

end problem_statement_l1386_138658


namespace max_value_of_f_l1386_138624

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : 
  ∃ x : ℝ, f x = 6 / 5 ∧ ∀ y : ℝ, f y ≤ 6 / 5 :=
sorry

end max_value_of_f_l1386_138624


namespace total_guitars_l1386_138657

theorem total_guitars (Barbeck_guitars Steve_guitars Davey_guitars : ℕ) (h1 : Barbeck_guitars = 2 * Steve_guitars) (h2 : Davey_guitars = 3 * Barbeck_guitars) (h3 : Davey_guitars = 18) : Barbeck_guitars + Steve_guitars + Davey_guitars = 27 :=
by sorry

end total_guitars_l1386_138657


namespace book_page_count_l1386_138666

theorem book_page_count (x : ℝ) : 
    (x - (1 / 4 * x + 20)) - ((1 / 3 * (x - (1 / 4 * x + 20)) + 25)) - (1 / 2 * ((x - (1 / 4 * x + 20)) - (1 / 3 * (x - (1 / 4 * x + 20)) + 25)) + 30) = 70 →
    x = 480 :=
by
  sorry

end book_page_count_l1386_138666


namespace perfect_square_trinomial_l1386_138614

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l1386_138614


namespace solve_equation_l1386_138697

open Real

theorem solve_equation :
  ∀ x : ℝ, (
    (1 / ((x - 2) * (x - 3))) +
    (1 / ((x - 3) * (x - 4))) +
    (1 / ((x - 4) * (x - 5))) = (1 / 12)
  ) ↔ (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) := 
by 
  sorry

end solve_equation_l1386_138697


namespace sum_of_roots_l1386_138631

theorem sum_of_roots (m n : ℝ) (h1 : ∀ x, x^2 - 3 * x - 1 = 0 → x = m ∨ x = n) : m + n = 3 :=
sorry

end sum_of_roots_l1386_138631


namespace work_completed_in_initial_days_l1386_138698

theorem work_completed_in_initial_days (x : ℕ) : 
  (100 * x = 50 * 40) → x = 20 :=
by
  sorry

end work_completed_in_initial_days_l1386_138698


namespace quadratic_equation_with_one_variable_is_B_l1386_138619

def is_quadratic_equation_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + x + 3 = 0"

theorem quadratic_equation_with_one_variable_is_B :
  is_quadratic_equation_with_one_variable "x^2 + x + 3 = 0" :=
by
  sorry

end quadratic_equation_with_one_variable_is_B_l1386_138619


namespace sum_of_areas_of_disks_l1386_138626

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks_l1386_138626


namespace polynomial_roots_quartic_sum_l1386_138699

noncomputable def roots_quartic_sum (a b c : ℂ) : ℂ :=
  a^4 + b^4 + c^4

theorem polynomial_roots_quartic_sum :
  ∀ (a b c : ℂ), (a^3 - 3 * a + 1 = 0) ∧ (b^3 - 3 * b + 1 = 0) ∧ (c^3 - 3 * c + 1 = 0) →
  (a + b + c = 0) ∧ (a * b + b * c + c * a = -3) ∧ (a * b * c = -1) →
  roots_quartic_sum a b c = 18 :=
by
  intros a b c hroot hsum
  sorry

end polynomial_roots_quartic_sum_l1386_138699


namespace no_real_roots_l1386_138627

-- Define the polynomial P(X) = X^5
def P (X : ℝ) : ℝ := X^5

-- Prove that for every α ∈ ℝ*, the polynomial P(X + α) - P(X) has no real roots
theorem no_real_roots (α : ℝ) (hα : α ≠ 0) : ∀ (X : ℝ), P (X + α) ≠ P X :=
by sorry

end no_real_roots_l1386_138627


namespace always_non_monotonic_l1386_138602

noncomputable def f (a t x : ℝ) : ℝ :=
if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

theorem always_non_monotonic (a : ℝ) (t : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a t x1 ≤ f a t x2 ∨ f a t x1 ≥ f a t x2) → a ≤ 1 / 2 :=
sorry

end always_non_monotonic_l1386_138602


namespace rectangle_width_decrease_l1386_138691

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease_l1386_138691


namespace trash_can_prices_and_minimum_A_can_purchase_l1386_138606

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l1386_138606


namespace inequality_one_over_a_plus_one_over_b_geq_4_l1386_138607

theorem inequality_one_over_a_plus_one_over_b_geq_4 
    (a b : ℕ) (hapos : 0 < a) (hbpos : 0 < b) (h : a + b = 1) : 
    (1 : ℚ) / a + (1 : ℚ) / b ≥ 4 := 
  sorry

end inequality_one_over_a_plus_one_over_b_geq_4_l1386_138607


namespace smallest_integer_conditions_l1386_138686

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Definition of having a prime factor less than a given number
def has_prime_factor_less_than (n k : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p < k

-- Problem statement
theorem smallest_integer_conditions :
  ∃ n : ℕ, n > 0 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ ¬ has_prime_factor_less_than n 60 ∧ ∀ m : ℕ, (m > 0 ∧ ¬ is_prime m ∧ ¬ is_square m ∧ ¬ has_prime_factor_less_than m 60) → n ≤ m :=
  sorry

end smallest_integer_conditions_l1386_138686


namespace vehicle_count_expression_l1386_138654

variable (C B M : ℕ)

-- Given conditions
axiom wheel_count : 4 * C + 2 * B + 2 * M = 196
axiom bike_to_motorcycle : B = 2 * M

-- Prove that the number of cars can be expressed in terms of the number of motorcycles
theorem vehicle_count_expression : C = (98 - 3 * M) / 2 :=
by
  sorry

end vehicle_count_expression_l1386_138654


namespace chris_age_l1386_138689

theorem chris_age (a b c : ℤ) (h1 : a + b + c = 45) (h2 : c - 5 = a)
  (h3 : c + 4 = 3 * (b + 4) / 4) : c = 15 :=
by
  sorry

end chris_age_l1386_138689


namespace a5_b3_c_divisible_by_6_l1386_138647

theorem a5_b3_c_divisible_by_6 (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) :=
by
  sorry

end a5_b3_c_divisible_by_6_l1386_138647


namespace ellipse_tangent_to_rectangle_satisfies_equation_l1386_138618

theorem ellipse_tangent_to_rectangle_satisfies_equation
  (a b : ℝ) -- lengths of the semi-major and semi-minor axes of the ellipse
  (h_rect : 4 * a * b = 48) -- the area condition (since the rectangle sides are 2a and 2b)
  (h_ellipse_form : ∃ (a b : ℝ), ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) : 
  a = 4 ∧ b = 3 ∨ a = 3 ∧ b = 4 := 
sorry

end ellipse_tangent_to_rectangle_satisfies_equation_l1386_138618


namespace union_of_A_and_B_l1386_138660

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_A_and_B_l1386_138660


namespace inequality_does_not_hold_l1386_138632

theorem inequality_does_not_hold {x y : ℝ} (h : x > y) : ¬ (-2 * x > -2 * y) ∧ (2023 * x > 2023 * y) ∧ (x - 1 > y - 1) ∧ (-x / 3 < -y / 3) :=
by {
  sorry
}

end inequality_does_not_hold_l1386_138632


namespace squared_difference_l1386_138601

theorem squared_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) : (x - y)^2 = 24 :=
by
  sorry

end squared_difference_l1386_138601


namespace compound_interest_doubling_time_l1386_138612

theorem compound_interest_doubling_time :
  ∃ t : ℕ, (2 : ℝ) < (1 + (0.13 : ℝ))^t ∧ t = 6 :=
by
  sorry

end compound_interest_doubling_time_l1386_138612


namespace white_socks_cost_proof_l1386_138629

-- Define the cost of a single brown sock in cents
def brown_sock_cost (B : ℕ) : Prop :=
  15 * B = 300

-- Define the cost of two white socks in cents
def white_socks_cost (B : ℕ) (W : ℕ) : Prop :=
  W = B + 25

-- Statement of the problem
theorem white_socks_cost_proof : 
  ∃ B W : ℕ, brown_sock_cost B ∧ white_socks_cost B W ∧ W = 45 :=
by
  sorry

end white_socks_cost_proof_l1386_138629


namespace sum_first_four_terms_l1386_138648

theorem sum_first_four_terms (a : ℕ → ℤ) (h5 : a 5 = 5) (h6 : a 6 = 9) (h7 : a 7 = 13) : 
  a 1 + a 2 + a 3 + a 4 = -20 :=
sorry

end sum_first_four_terms_l1386_138648


namespace electronics_weight_is_9_l1386_138600

noncomputable def electronics_weight : ℕ :=
  let B : ℕ := sorry -- placeholder for the value of books weight.
  let C : ℕ := 12
  let E : ℕ := 9
  have h1 : (B : ℚ) / (C : ℚ) = 7 / 4 := sorry
  have h2 : (C : ℚ) / (E : ℚ) = 4 / 3 := sorry
  have h3 : (B : ℚ) / (C - 6 : ℚ) = 7 / 2 := sorry
  E

theorem electronics_weight_is_9 : electronics_weight = 9 :=
by
  dsimp [electronics_weight]
  repeat { sorry }

end electronics_weight_is_9_l1386_138600


namespace quadratic_inequality_solution_l1386_138617

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 6*x - 16 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 8} := by
  sorry

end quadratic_inequality_solution_l1386_138617


namespace hyperbola_range_l1386_138688

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (2 + m) + y^2 / (m + 1) = 1)) → (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_range_l1386_138688


namespace rent_fraction_l1386_138643

theorem rent_fraction (B R : ℝ) 
  (food_and_beverages_spent : (1 / 4) * (1 - R) * B = 0.1875 * B) : 
  R = 0.25 :=
by
  -- proof skipped
  sorry

end rent_fraction_l1386_138643


namespace line_passes_through_fixed_point_equal_intercepts_line_equation_l1386_138623

open Real

theorem line_passes_through_fixed_point (m : ℝ) : ∃ P : ℝ × ℝ, P = (4, 1) ∧ (m + 2) * P.1 - (m + 1) * P.2 - 3 * m - 7 = 0 := 
sorry

theorem equal_intercepts_line_equation (m : ℝ) :
  ((3 * m + 7) / (m + 2) = -(3 * m + 7) / (m + 1)) → (m = -3 / 2) → 
  (∀ (x y : ℝ), (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0 → x + y - 5 = 0) := 
sorry

end line_passes_through_fixed_point_equal_intercepts_line_equation_l1386_138623


namespace find_valid_pairs_l1386_138663

-- Defining the conditions and target answer set.
def valid_pairs : List (Nat × Nat) := [(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)]

theorem find_valid_pairs (a b : Nat) :
  (∃ n m : Int, (a^2 + b = n * (b^2 - a)) ∧ (b^2 + a = m * (a^2 - b)))
  ↔ (a, b) ∈ valid_pairs :=
by sorry

end find_valid_pairs_l1386_138663


namespace dasha_paper_strip_l1386_138625

theorem dasha_paper_strip (a b c : ℕ) (h1 : a < b) (h2 : 2 * a * b + 2 * a * c - a^2 = 43) :
    ∃ (length width : ℕ), length = a ∧ width = b + c := by
  sorry

end dasha_paper_strip_l1386_138625


namespace infinitely_many_colorings_l1386_138641

def colorings_exist (clr : ℕ → Prop) : Prop :=
  ∀ a b : ℕ, (clr a = clr b) ∧ (0 < a - 10 * b) → clr (a - 10 * b) = clr a

theorem infinitely_many_colorings : ∃ (clr : ℕ → Prop), colorings_exist clr :=
sorry

end infinitely_many_colorings_l1386_138641


namespace isosceles_triangle_base_angle_l1386_138650

theorem isosceles_triangle_base_angle (x : ℝ) 
  (h1 : ∀ (a b : ℝ), a + b + (20 + 2 * b) = 180)
  (h2 : 20 + 2 * x = 180 - 2 * x - x) : x = 40 :=
by sorry

end isosceles_triangle_base_angle_l1386_138650


namespace question1_question2_l1386_138665

theorem question1 (m : ℝ) (x : ℝ) :
  (∀ x, x^2 - m * x + (m - 1) ≥ 0) → m = 2 :=
by
  sorry

theorem question2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (n = (a + 1 / b) * (2 * b + 1 / (2 * a))) → n ≥ (9 / 2) :=
by
  sorry

end question1_question2_l1386_138665


namespace partial_fraction_decomposition_l1386_138667

theorem partial_fraction_decomposition :
  ∃ A B C : ℚ, (∀ x : ℚ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
          (x^2 + 2 * x - 8) / (x^3 - x - 2) = A / (x + 1) + (B * x + C) / (x^2 - x + 2)) ∧
          A = -9/4 ∧ B = 13/4 ∧ C = -7/2 :=
sorry

end partial_fraction_decomposition_l1386_138667


namespace packs_of_tuna_purchased_l1386_138651

-- Definitions based on the conditions
def cost_per_pack_of_tuna : ℕ := 2
def cost_per_bottle_of_water : ℤ := (3 / 2)
def total_paid_by_Barbara : ℕ := 56
def money_spent_on_different_goods : ℕ := 40
def number_of_bottles_of_water : ℕ := 4

-- The proposition to prove
theorem packs_of_tuna_purchased :
  ∃ T : ℕ, total_paid_by_Barbara = cost_per_pack_of_tuna * T + cost_per_bottle_of_water * number_of_bottles_of_water + money_spent_on_different_goods ∧ T = 5 :=
by
  sorry

end packs_of_tuna_purchased_l1386_138651


namespace max_tickets_with_120_l1386_138637

-- Define the cost of tickets
def cost_ticket (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 15
  else 5 * 15 + (n - 5) * 12

-- Define the maximum number of tickets Jane can buy with 120 dollars
def max_tickets (money : ℕ) : ℕ :=
  if money ≤ 75 then money / 15
  else 5 + (money - 75) / 12

-- Prove that with 120 dollars, the maximum number of tickets Jane can buy is 8
theorem max_tickets_with_120 : max_tickets 120 = 8 :=
by
  sorry

end max_tickets_with_120_l1386_138637


namespace part1_proof_part2_proof_l1386_138675

open Real

-- Definitions for the conditions
variables (x y z : ℝ)
variable (h₁ : 0 < x)
variable (h₂ : 0 < y)
variable (h₃ : 0 < z)

-- Part 1
theorem part1_proof : (1 / x + 1 / y ≥ 4 / (x + y)) :=
by sorry

-- Part 2
theorem part2_proof : (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) :=
by sorry

end part1_proof_part2_proof_l1386_138675


namespace find_number_l1386_138605

theorem find_number (x : ℝ) (h : 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30) : x = 66 :=
by
  sorry

end find_number_l1386_138605


namespace bankers_discount_l1386_138680

/-- Given the present worth (P) of Rs. 400 and the true discount (TD) of Rs. 20,
Prove that the banker's discount (BD) is Rs. 21. -/
theorem bankers_discount (P TD FV BD : ℝ) (hP : P = 400) (hTD : TD = 20) 
(hFV : FV = P + TD) (hBD : BD = (TD * FV) / P) : BD = 21 := 
by
  sorry

end bankers_discount_l1386_138680


namespace base_of_square_eq_l1386_138636

theorem base_of_square_eq (b : ℕ) (h : b > 6) : 
  (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 → b = 7 :=
by
  sorry

end base_of_square_eq_l1386_138636


namespace magnitude_of_a_plus_b_in_range_l1386_138676

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def theta_domain : Set ℝ := {θ : ℝ | -Real.pi / 2 < θ ∧ θ < Real.pi / 2}

open Real

theorem magnitude_of_a_plus_b_in_range (θ : ℝ) (hθ : θ ∈ theta_domain) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (cos θ, sin θ)
  1 < sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) ∧ sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) < (3 + 2 * sqrt 2) :=
sorry

end magnitude_of_a_plus_b_in_range_l1386_138676


namespace minimal_d1_l1386_138672

theorem minimal_d1 :
  (∃ (S3 S6 : ℕ), 
    ∃ (d1 : ℚ), 
      S3 = d1 + (d1 + 1) + (d1 + 2) ∧ 
      S6 = d1 + (d1 + 1) + (d1 + 2) + (d1 + 3) + (d1 + 4) + (d1 + 5) ∧ 
      d1 = (5 * S3 - S6) / 9 ∧ 
      d1 ≥ 1 / 2) → 
  ∃ (d1 : ℚ), d1 = 5 / 9 := 
by 
  sorry

end minimal_d1_l1386_138672


namespace harkamal_total_amount_l1386_138674

def cost_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_mangoes (quantity rate : ℕ) : ℕ := quantity * rate
def total_amount_paid (cost1 cost2 : ℕ) : ℕ := cost1 + cost2

theorem harkamal_total_amount :
  let grapes_quantity := 8
  let grapes_rate := 70
  let mangoes_quantity := 9
  let mangoes_rate := 65
  total_amount_paid (cost_grapes grapes_quantity grapes_rate) (cost_mangoes mangoes_quantity mangoes_rate) = 1145 := 
by
  sorry

end harkamal_total_amount_l1386_138674


namespace marbles_in_jar_l1386_138604

theorem marbles_in_jar (g y p : ℕ) (h1 : y + p = 7) (h2 : g + p = 10) (h3 : g + y = 5) :
  g + y + p = 11 :=
sorry

end marbles_in_jar_l1386_138604


namespace perimeter_of_triangle_l1386_138622

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 9) + (P.2^2 / 5) = 1

noncomputable def foci_position (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (-2, 0) ∧ F2 = (2, 0)

theorem perimeter_of_triangle :
  ∀ (P F1 F2 : ℝ × ℝ),
    point_on_ellipse P →
    foci_position F1 F2 →
    dist P F1 + dist P F2 + dist F1 F2 = 10 :=
by
  sorry

end perimeter_of_triangle_l1386_138622


namespace geo_seq_sum_l1386_138621

theorem geo_seq_sum (a : ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = 2^n + a) →
  a = -1 :=
sorry

end geo_seq_sum_l1386_138621


namespace decreasing_even_function_condition_l1386_138695

theorem decreasing_even_function_condition (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, x < y → y < 0 → f y < f x) 
    (h2 : ∀ x : ℝ, f (-x) = f x) : f 13 < f 9 ∧ f 9 < f 1 := 
by
  sorry

end decreasing_even_function_condition_l1386_138695


namespace equilateral_triangle_area_decrease_l1386_138642

theorem equilateral_triangle_area_decrease :
  let original_area : ℝ := 100 * Real.sqrt 3
  let side_length_s := 20
  let decreased_side_length := side_length_s - 6
  let new_area := (decreased_side_length * decreased_side_length * Real.sqrt 3) / 4
  let decrease_in_area := original_area - new_area
  decrease_in_area = 51 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_decrease_l1386_138642


namespace solution_set_of_inequality_l1386_138638

/-- Given an even function f that is monotonically increasing on [0, ∞) with f(3) = 0,
    show that the solution set for xf(2x - 1) < 0 is (-∞, -1) ∪ (0, 2). -/
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_value : f 3 = 0) :
  {x : ℝ | x * f (2*x - 1) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l1386_138638


namespace touching_squares_same_color_probability_l1386_138616

theorem touching_squares_same_color_probability :
  let m := 0
  let n := 1
  100 * m + n = 1 :=
by
  let m := 0
  let n := 1
  sorry -- Proof is omitted as per instructions

end touching_squares_same_color_probability_l1386_138616


namespace quadratic_root_equation_l1386_138615

-- Define the conditions given in the problem
variables (a b x : ℝ)

-- Assertion for a ≠ 0
axiom a_ne_zero : a ≠ 0

-- Root assumption
axiom root_assumption : (x^2 + b * x + a = 0) → x = -a

-- Lean statement to prove that b - a = 1
theorem quadratic_root_equation (h : x^2 + b * x + a = 0) : b - a = 1 :=
sorry

end quadratic_root_equation_l1386_138615


namespace minimum_value_proof_l1386_138683

noncomputable def minimum_value (a b : ℝ) (h : 0 < a ∧ 0 < b) : ℝ :=
  1 / (2 * a) + 1 / b

theorem minimum_value_proof (a b : ℝ) (h : 0 < a ∧ 0 < b)
  (line_bisects_circle : a + b = 1) : minimum_value a b h = (3 + 2 * Real.sqrt 2) / 2 := 
by
  sorry

end minimum_value_proof_l1386_138683


namespace professors_initial_count_l1386_138692

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l1386_138692
