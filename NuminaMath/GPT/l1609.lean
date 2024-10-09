import Mathlib

namespace hypotenuse_length_l1609_160993

theorem hypotenuse_length
  (x : ℝ) 
  (h_leg_relation : 3 * x - 3 > 0) -- to ensure the legs are positive
  (hypotenuse : ℝ)
  (area_eq : 1 / 2 * x * (3 * x - 3) = 84)
  (pythagorean : hypotenuse^2 = x^2 + (3 * x - 3)^2) :
  hypotenuse = Real.sqrt 505 :=
by 
  sorry

end hypotenuse_length_l1609_160993


namespace total_marbles_l1609_160964

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3

theorem total_marbles : Mary_marbles + Joan_marbles = 12 :=
by
  -- Please provide the proof here if needed
  sorry

end total_marbles_l1609_160964


namespace tea_maker_capacity_l1609_160985

theorem tea_maker_capacity (x : ℝ) (h : 0.45 * x = 54) : x = 120 :=
by
  sorry

end tea_maker_capacity_l1609_160985


namespace sum_of_squares_of_roots_l1609_160917

theorem sum_of_squares_of_roots :
  ∀ (p q r : ℚ), (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) ∧
                 (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) ∧
                 (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
                 p^2 + q^2 + r^2 = 34 / 9 := 
by
  sorry

end sum_of_squares_of_roots_l1609_160917


namespace find_x_l1609_160997

theorem find_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) → x = 1.5 := 
by 
  sorry

end find_x_l1609_160997


namespace probability_of_one_black_ball_l1609_160957

theorem probability_of_one_black_ball (total_balls black_balls white_balls drawn_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_black : black_balls = 2)
  (h_white : white_balls = 2)
  (h_drawn : drawn_balls = 2) :
  ((Nat.choose black_balls 1) * (Nat.choose white_balls 1) : ℚ) / (Nat.choose total_balls drawn_balls) = 2 / 3 :=
by {
  -- Insert proof here
  sorry
}

end probability_of_one_black_ball_l1609_160957


namespace solution_set_of_inequalities_l1609_160951

theorem solution_set_of_inequalities (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : ∀ x, mx + n > 0 ↔ x < (1/3)) : ∀ x, nx - m < 0 ↔ x < -3 :=
by
  sorry

end solution_set_of_inequalities_l1609_160951


namespace solve_inequality_l1609_160938

theorem solve_inequality :
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ 
  (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) → 
  0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ 
  (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2 :=
sorry

end solve_inequality_l1609_160938


namespace price_difference_l1609_160991

-- Define the prices of commodity X and Y in the year 2001 + n.
def P_X (n : ℕ) (a : ℝ) : ℝ := 4.20 + 0.45 * n + a * n
def P_Y (n : ℕ) (b : ℝ) : ℝ := 6.30 + 0.20 * n + b * n

-- Define the main theorem to prove
theorem price_difference (n : ℕ) (a b : ℝ) :
  P_X n a = P_Y n b + 0.65 ↔ (0.25 + a - b) * n = 2.75 :=
by
  sorry

end price_difference_l1609_160991


namespace unique_solution_pairs_l1609_160909

theorem unique_solution_pairs :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 = 4 * c) ∧ (c^2 = 4 * b) :=
sorry

end unique_solution_pairs_l1609_160909


namespace right_triangle_perimeter_l1609_160914

-- Given conditions
variable (x y : ℕ)
def leg1 := 11
def right_triangle := (101 * 11 = 121)

-- The question and answer
theorem right_triangle_perimeter :
  (y + x = 121) ∧ (y - x = 1) → (11 + x + y = 132) :=
by
  sorry

end right_triangle_perimeter_l1609_160914


namespace consecutive_integers_sum_l1609_160915

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l1609_160915


namespace parallelogram_area_correct_l1609_160947

noncomputable def parallelogram_area (a b : ℝ) (α : ℝ) (h : a < b) : ℝ :=
  (4 * a^2 - b^2) / 4 * (Real.tan α)

theorem parallelogram_area_correct (a b α : ℝ) (h : a < b) :
  parallelogram_area a b α h = (4 * a^2 - b^2) / 4 * (Real.tan α) :=
by
  sorry

end parallelogram_area_correct_l1609_160947


namespace fencing_cost_l1609_160911

noncomputable def pi_approx : ℝ := 3.14159

theorem fencing_cost 
  (d : ℝ) (r : ℝ)
  (h_d : d = 20) 
  (h_r : r = 1.50) :
  abs (r * pi_approx * d - 94.25) < 1 :=
by
  -- Proof omitted
  sorry

end fencing_cost_l1609_160911


namespace a_2016_is_neg1_l1609_160988

noncomputable def a : ℕ → ℤ
| 0     => 0 -- Arbitrary value for n = 0 since sequences generally start from 1 in Lean
| 1     => 1
| 2     => 2
| n + 1 => a n - a (n - 1)

theorem a_2016_is_neg1 : a 2016 = -1 := sorry

end a_2016_is_neg1_l1609_160988


namespace sum_of_perimeters_of_squares_l1609_160978

theorem sum_of_perimeters_of_squares (x y : ℕ)
  (h1 : x^2 - y^2 = 19) : 4 * x + 4 * y = 76 := 
by
  sorry

end sum_of_perimeters_of_squares_l1609_160978


namespace kyle_gas_and_maintenance_expense_l1609_160996

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l1609_160996


namespace problem1_subproblem1_subproblem2_l1609_160948

-- Problem 1: Prove that a² + b² = 40 given ab = 30 and a + b = 10
theorem problem1 (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) : a^2 + b^2 = 40 := 
sorry

-- Problem 2: Subproblem 1 - Prove that (40 - x)² + (x - 20)² = 420 given (40 - x)(x - 20) = -10
theorem subproblem1 (x : ℝ) (h : (40 - x) * (x - 20) = -10) : (40 - x)^2 + (x - 20)^2 = 420 := 
sorry

-- Problem 2: Subproblem 2 - Prove that (30 + x)² + (20 + x)² = 120 given (30 + x)(20 + x) = 10
theorem subproblem2 (x : ℝ) (h : (30 + x) * (20 + x) = 10) : (30 + x)^2 + (20 + x)^2 = 120 :=
sorry

end problem1_subproblem1_subproblem2_l1609_160948


namespace johnny_marble_combinations_l1609_160953

/-- 
Johnny has 10 different colored marbles. 
The number of ways he can choose four different marbles from his bag is 210.
-/
theorem johnny_marble_combinations : (Nat.choose 10 4) = 210 := by
  sorry

end johnny_marble_combinations_l1609_160953


namespace number_of_terms_added_l1609_160954

theorem number_of_terms_added (k : ℕ) (h : 1 ≤ k) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k :=
by sorry

end number_of_terms_added_l1609_160954


namespace Apollonius_circle_symmetry_l1609_160952

theorem Apollonius_circle_symmetry (a : ℝ) (h : a > 1): 
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let locus_C := {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (Real.sqrt ((x + 1)^2 + y^2) = a * Real.sqrt ((x - 1)^2 + y^2))}
  let symmetric_y := ∀ (P : ℝ × ℝ), P ∈ locus_C → (P.1, -P.2) ∈ locus_C
  symmetric_y := sorry

end Apollonius_circle_symmetry_l1609_160952


namespace probability_of_green_ball_l1609_160920

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l1609_160920


namespace max_pages_within_budget_l1609_160965

-- Definitions based on the problem conditions
def page_cost_in_cents : ℕ := 5
def total_budget_in_cents : ℕ := 5000
def max_expenditure_in_cents : ℕ := 4500

-- Proof problem statement
theorem max_pages_within_budget : 
  ∃ (pages : ℕ), pages = max_expenditure_in_cents / page_cost_in_cents ∧ 
                  pages * page_cost_in_cents ≤ total_budget_in_cents :=
by {
  sorry
}

end max_pages_within_budget_l1609_160965


namespace value_of_expression_l1609_160921

theorem value_of_expression : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 :=
by
  sorry

end value_of_expression_l1609_160921


namespace circle_properties_l1609_160976

theorem circle_properties :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 4 * y - 25 = -y^2 + 10 * x + 49 → (x - 5)^2 + (y - 2)^2 = s^2) ∧
  c = 5 ∧ d = 2 ∧ s = Real.sqrt 103 ∧ c + d + s = 7 + Real.sqrt 103 :=
by
  sorry

end circle_properties_l1609_160976


namespace math_problem_l1609_160928

noncomputable def a (b : ℝ) : ℝ := 
  sorry -- to be derived from the conditions

noncomputable def b : ℝ := 
  sorry -- to be derived from the conditions

theorem math_problem (a b: ℝ) 
  (h1: a - b = 1)
  (h2: a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := 
sorry

end math_problem_l1609_160928


namespace number_of_valid_m_values_l1609_160998

noncomputable def polynomial (m : ℤ) (x : ℤ) : ℤ := 
  2 * (m - 1) * x ^ 2 - (m ^ 2 - m + 12) * x + 6 * m

noncomputable def discriminant (m : ℤ) : ℤ :=
  (m ^ 2 - m + 12) ^ 2 - 4 * 2 * (m - 1) * 6 * m

def is_perfect_square (n : ℤ) : Prop :=
  ∃ (k : ℤ), k * k = n

def has_integral_roots (m : ℤ) : Prop :=
  ∃ (r1 r2 : ℤ), polynomial m r1 = 0 ∧ polynomial m r2 = 0

def valid_m_values (m : ℤ) : Prop :=
  (discriminant m) > 0 ∧ is_perfect_square (discriminant m) ∧ has_integral_roots m

theorem number_of_valid_m_values : 
  (∃ M : List ℤ, (∀ m ∈ M, valid_m_values m) ∧ M.length = 4) :=
  sorry

end number_of_valid_m_values_l1609_160998


namespace least_integer_value_satisfying_inequality_l1609_160904

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end least_integer_value_satisfying_inequality_l1609_160904


namespace card_probability_l1609_160956

theorem card_probability :
  let totalCards := 52
  let kings := 4
  let jacks := 4
  let queens := 4
  let firstCardKing := kings / totalCards
  let secondCardJack := jacks / (totalCards - 1)
  let thirdCardQueen := queens / (totalCards - 2)
  (firstCardKing * secondCardJack * thirdCardQueen) = (8 / 16575) :=
by
  sorry

end card_probability_l1609_160956


namespace find_FC_l1609_160923

theorem find_FC 
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 12)
  (h3 : AB = (1/5) * AD)
  (h4 : ED = (2/3) * AD)
  (h5 : AD = (5/4) * 22)  -- Derived step from solution for full transparency
  (h6 : FC = (ED * (CB + AB)) / AD) : 
  FC = 35 / 3 := 
sorry

end find_FC_l1609_160923


namespace discount_on_soap_l1609_160960

theorem discount_on_soap :
  (let chlorine_price := 10
   let chlorine_discount := 0.20 * chlorine_price
   let discounted_chlorine_price := chlorine_price - chlorine_discount

   let soap_price := 16

   let total_savings := 26

   let chlorine_savings := 3 * chlorine_price - 3 * discounted_chlorine_price
   let soap_savings := total_savings - chlorine_savings

   let discount_per_soap := soap_savings / 5
   let discount_percentage_per_soap := (discount_per_soap / soap_price) * 100
   discount_percentage_per_soap = 25) := sorry

end discount_on_soap_l1609_160960


namespace days_c_worked_l1609_160924

noncomputable def work_done_by_a_b := 1 / 10
noncomputable def work_done_by_b_c := 1 / 18
noncomputable def work_done_by_c_alone := 1 / 45

theorem days_c_worked
  (A B C : ℚ)
  (h1 : A + B = work_done_by_a_b)
  (h2 : B + C = work_done_by_b_c)
  (h3 : C = work_done_by_c_alone) :
  15 = (1/3) / work_done_by_c_alone :=
sorry

end days_c_worked_l1609_160924


namespace similar_polygons_perimeter_ratio_l1609_160925

-- Define the main function to assert the proportional relationship
theorem similar_polygons_perimeter_ratio (x y : ℕ) (h1 : 9 * y^2 = 64 * x^2) : x * 8 = y * 3 :=
by sorry

-- noncomputable if needed (only necessary when computation is involved, otherwise omit)

end similar_polygons_perimeter_ratio_l1609_160925


namespace plan_y_cost_effective_l1609_160934

theorem plan_y_cost_effective (m : ℕ) (h1 : ∀ minutes, cost_plan_x = 15 * minutes)
(h2 : ∀ minutes, cost_plan_y = 3000 + 10 * minutes) :
m ≥ 601 → 3000 + 10 * m < 15 * m :=
by
sorry

end plan_y_cost_effective_l1609_160934


namespace four_minus_x_is_five_l1609_160941

theorem four_minus_x_is_five (x y : ℤ) (h1 : 4 + x = 5 - y) (h2 : 3 + y = 6 + x) : 4 - x = 5 := by
sorry

end four_minus_x_is_five_l1609_160941


namespace sequence_noncongruent_modulo_l1609_160906

theorem sequence_noncongruent_modulo 
  (a : ℕ → ℕ)
  (h0 : a 1 = 1)
  (h1 : ∀ n, a (n + 1) = a n + 2^(a n)) :
  ∀ (i j : ℕ), i ≠ j → i ≤ 32021 → j ≤ 32021 →
  (a i) % (3^2021) ≠ (a j) % (3^2021) := 
by
  sorry

end sequence_noncongruent_modulo_l1609_160906


namespace equation_solution_l1609_160987

noncomputable def solve_equation : Prop :=
∃ (x : ℝ), x^6 + (3 - x)^6 = 730 ∧ (x = 1.5 + Real.sqrt 5 ∨ x = 1.5 - Real.sqrt 5)

theorem equation_solution : solve_equation :=
sorry

end equation_solution_l1609_160987


namespace x_eq_1_sufficient_but_not_necessary_l1609_160980

theorem x_eq_1_sufficient_but_not_necessary (x : ℝ) : x^2 - 3 * x + 2 = 0 → (x = 1 ↔ true) ∧ (x ≠ 1 → ∃ y : ℝ, y ≠ x ∧ y^2 - 3 * y + 2 = 0) :=
by
  sorry

end x_eq_1_sufficient_but_not_necessary_l1609_160980


namespace Andy_collects_16_balls_l1609_160977

-- Define the number of balls collected by Andy, Roger, and Maria.
variables (x : ℝ) (r : ℝ) (m : ℝ)

-- Define the conditions
def Andy_twice_as_many_as_Roger : Prop := r = x / 2
def Andy_five_more_than_Maria : Prop := m = x - 5
def Total_balls : Prop := x + r + m = 35

-- Define the main theorem to prove Andy's number of balls
theorem Andy_collects_16_balls (h1 : Andy_twice_as_many_as_Roger x r) 
                               (h2 : Andy_five_more_than_Maria x m) 
                               (h3 : Total_balls x r m) : 
                               x = 16 := 
by 
  sorry

end Andy_collects_16_balls_l1609_160977


namespace log_value_between_integers_l1609_160931

theorem log_value_between_integers : (1 : ℤ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < (2 : ℤ) → 1 + 2 = 3 :=
by
  sorry

end log_value_between_integers_l1609_160931


namespace initial_distance_l1609_160945

theorem initial_distance (speed_enrique speed_jamal : ℝ) (hours : ℝ) 
  (h_enrique : speed_enrique = 16) 
  (h_jamal : speed_jamal = 23) 
  (h_time : hours = 8) 
  (h_difference : speed_jamal = speed_enrique + 7) : 
  (speed_enrique * hours + speed_jamal * hours = 312) :=
by 
  sorry

end initial_distance_l1609_160945


namespace sqrt_fraction_expression_eq_one_l1609_160962

theorem sqrt_fraction_expression_eq_one :
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 6) = 1 := 
by
  sorry

end sqrt_fraction_expression_eq_one_l1609_160962


namespace find_xz_over_y_squared_l1609_160975

variable {x y z : ℝ}

noncomputable def k : ℝ := 7

theorem find_xz_over_y_squared
    (h1 : x + k * y + 4 * z = 0)
    (h2 : 4 * x + k * y - 3 * z = 0)
    (h3 : x + 3 * y - 2 * z = 0)
    (h_nz : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
    (x * z) / (y ^ 2) = 26 / 9 :=
by sorry

end find_xz_over_y_squared_l1609_160975


namespace chocolates_difference_l1609_160913

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ) (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 5) : robert_chocolates - nickel_chocolates = 2 :=
by sorry

end chocolates_difference_l1609_160913


namespace quadratic_real_roots_range_l1609_160967

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9 / 4 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l1609_160967


namespace dependent_variable_is_temperature_l1609_160974

-- Define the variables involved in the problem
variables (intensity_of_sunlight : ℝ)
variables (temperature_of_water : ℝ)
variables (duration_of_exposure : ℝ)
variables (capacity_of_heater : ℝ)

-- Define the conditions
def changes_with_duration (temp: ℝ) (duration: ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ d, temp = f d) ∧ ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂

-- The theorem we need to prove
theorem dependent_variable_is_temperature :
  changes_with_duration temperature_of_water duration_of_exposure → 
  (∀ t, ∃ d, temperature_of_water = t → duration_of_exposure = d) :=
sorry

end dependent_variable_is_temperature_l1609_160974


namespace range_of_a_l1609_160939

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, ¬ (x^2 - a * x + 1 ≤ 0)) ↔ -2 < a ∧ a < 2 := 
sorry

end range_of_a_l1609_160939


namespace interval_monotonically_increasing_range_g_l1609_160981

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sqrt 3) * Real.sin (x + (Real.pi / 4)) * Real.cos (x + (Real.pi / 4)) + Real.sin (2 * x) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (2 * Real.pi / 3)) - 1

theorem interval_monotonically_increasing :
  ∃ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) → 0 ≤ deriv f x :=
sorry

theorem range_g (m : ℝ) : 
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → g x = m ↔ -3 ≤ m ∧ m ≤ Real.sqrt 3 - 1 :=
sorry

end interval_monotonically_increasing_range_g_l1609_160981


namespace probability_two_boys_l1609_160983

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_pairs : ℕ := Nat.choose number_of_students 2
def boys_pairs : ℕ := Nat.choose number_of_boys 2

theorem probability_two_boys :
  number_of_students = 5 →
  number_of_boys = 2 →
  number_of_girls = 3 →
  (boys_pairs : ℝ) / (total_pairs : ℝ) = 1 / 10 :=
by
  sorry

end probability_two_boys_l1609_160983


namespace female_employees_sampled_l1609_160970

theorem female_employees_sampled
  (T : ℕ) -- Total number of employees
  (M : ℕ) -- Number of male employees
  (F : ℕ) -- Number of female employees
  (S_m : ℕ) -- Number of sampled male employees
  (H_T : T = 140)
  (H_M : M = 80)
  (H_F : F = 60)
  (H_Sm : S_m = 16) :
  ∃ S_f : ℕ, S_f = 12 :=
by
  sorry

end female_employees_sampled_l1609_160970


namespace area_ratio_triangle_l1609_160944

noncomputable def area_ratio (x y : ℝ) (n m : ℕ) : ℝ :=
(x * y) / (2 * n) / ((x * y) / (2 * m))

theorem area_ratio_triangle (x y : ℝ) (n m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  area_ratio x y n m = (m : ℝ) / (n : ℝ) := by
  sorry

end area_ratio_triangle_l1609_160944


namespace find_value_of_expression_l1609_160907

theorem find_value_of_expression (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 :=
sorry

end find_value_of_expression_l1609_160907


namespace batsman_average_after_12th_inning_l1609_160902

theorem batsman_average_after_12th_inning (average_initial : ℕ) (score_12th : ℕ) (average_increase : ℕ) (total_innings : ℕ) 
    (h_avg_init : average_initial = 29) (h_score_12th : score_12th = 65) (h_avg_inc : average_increase = 3) 
    (h_total_innings : total_innings = 12) : 
    (average_initial + average_increase = 32) := 
by
  sorry

end batsman_average_after_12th_inning_l1609_160902


namespace greatest_integer_b_for_no_real_roots_l1609_160942

theorem greatest_integer_b_for_no_real_roots :
  ∃ (b : ℤ), (b * b < 20) ∧ (∀ (c : ℤ), (c * c < 20) → c ≤ 4) :=
by
  sorry

end greatest_integer_b_for_no_real_roots_l1609_160942


namespace kelly_games_giveaway_l1609_160995

theorem kelly_games_giveaway (n m g : ℕ) (h_current: n = 50) (h_left: m = 35) : g = n - m :=
by
  sorry

end kelly_games_giveaway_l1609_160995


namespace max_bars_scenario_a_max_bars_scenario_b_l1609_160968

-- Define the game conditions and the maximum bars Ivan can take in each scenario.

def max_bars_taken (initial_bars : ℕ) : ℕ :=
  if initial_bars = 14 then 13 else 13

theorem max_bars_scenario_a :
  max_bars_taken 13 = 13 :=
by sorry

theorem max_bars_scenario_b :
  max_bars_taken 14 = 13 :=
by sorry

end max_bars_scenario_a_max_bars_scenario_b_l1609_160968


namespace english_alphabet_is_set_l1609_160961

-- Conditions definition: Elements of a set must have the properties of definiteness, distinctness, and unorderedness.
def is_definite (A : Type) : Prop := ∀ (a b : A), a = b ∨ a ≠ b
def is_distinct (A : Type) : Prop := ∀ (a b : A), a ≠ b → (a ≠ b)
def is_unordered (A : Type) : Prop := true  -- For simplicity, we assume unorderedness holds for any set

-- Property that verifies if the 26 letters of the English alphabet can form a set
def english_alphabet_set : Prop :=
  is_definite Char ∧ is_distinct Char ∧ is_unordered Char

theorem english_alphabet_is_set : english_alphabet_set :=
  sorry

end english_alphabet_is_set_l1609_160961


namespace Ted_age_48_l1609_160973

/-- Given ages problem:
 - t is Ted's age
 - s is Sally's age
 - a is Alex's age 
 - The following conditions hold:
   1. t = 2s + 17 
   2. a = s / 2
   3. t + s + a = 72
 - Prove that Ted's age (t) is 48.
-/ 
theorem Ted_age_48 {t s a : ℕ} (h1 : t = 2 * s + 17) (h2 : a = s / 2) (h3 : t + s + a = 72) : t = 48 := by
  sorry

end Ted_age_48_l1609_160973


namespace john_books_purchase_l1609_160950

theorem john_books_purchase : 
  let john_money := 4575
  let book_price := 325
  john_money / book_price = 14 :=
by
  sorry

end john_books_purchase_l1609_160950


namespace number_of_students_l1609_160929

-- Define parameters and conditions
variables (B G : ℕ) -- number of boys and girls

-- Condition: each boy is friends with exactly two girls
axiom boys_to_girls : ∀ (B G : ℕ), 2 * B = 3 * G

-- Condition: total number of children in the class
axiom total_children : ∀ (B G : ℕ), B + G = 31

-- Define the theorem that proves the correct number of students
theorem number_of_students : (B G : ℕ) → 2 * B = 3 * G → B + G = 31 → B + G = 35 :=
by
  sorry

end number_of_students_l1609_160929


namespace large_marshmallows_are_eight_l1609_160963

-- Definition for the total number of marshmallows
def total_marshmallows : ℕ := 18

-- Definition for the number of mini marshmallows
def mini_marshmallows : ℕ := 10

-- Definition for the number of large marshmallows
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

-- Theorem stating that the number of large marshmallows is 8
theorem large_marshmallows_are_eight : large_marshmallows = 8 := by
  sorry

end large_marshmallows_are_eight_l1609_160963


namespace minimum_value_of_tan_sum_l1609_160994

open Real

theorem minimum_value_of_tan_sum :
  ∀ {A B C : ℝ}, 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ 
  2 * sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 ->
  ( ∃ t : ℝ, ( t = 1 / tan A + 1 / tan B + 1 / tan C ) ∧ t = sqrt 13 / 2 ) := 
sorry

end minimum_value_of_tan_sum_l1609_160994


namespace dimensions_multiple_of_three_l1609_160910

theorem dimensions_multiple_of_three (a b c : ℤ) (h : a * b * c = (a + 1) * (b + 1) * (c - 2)) :
  (a % 3 = 0) ∨ (b % 3 = 0) ∨ (c % 3 = 0) :=
sorry

end dimensions_multiple_of_three_l1609_160910


namespace minimum_value_expression_l1609_160946

theorem minimum_value_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 :=
by
  sorry

end minimum_value_expression_l1609_160946


namespace base8_subtraction_l1609_160916

-- Define the base 8 notation for the given numbers
def b8_256 := 256
def b8_167 := 167
def b8_145 := 145

-- Define the sum of 256_8 and 167_8 in base 8
def sum_b8 := 435

-- Define the result of subtracting 145_8 from the sum in base 8
def result_b8 := 370

-- Prove that the result of the entire operation is 370_8
theorem base8_subtraction : sum_b8 - b8_145 = result_b8 := by
  sorry

end base8_subtraction_l1609_160916


namespace line_AB_equation_l1609_160958

theorem line_AB_equation (m : ℝ) (A B : ℝ × ℝ)
  (hA : A = (0, 0)) (hA_line : ∀ (x y : ℝ), A = (x, y) → x + m * y = 0)
  (hB : B = (1, 3)) (hB_line : ∀ (x y : ℝ), B = (x, y) → m * x - y - m + 3 = 0) :
  ∃ (a b c : ℝ), a * 1 - b * 3 + c = 0 ∧ a * x + b * y + c * 0 = 0 ∧ 3 * x - y + 0 = 0 :=
by
  sorry

end line_AB_equation_l1609_160958


namespace unique_increasing_seq_l1609_160905

noncomputable def unique_seq (a : ℕ → ℕ) (r : ℝ) : Prop :=
∀ (b : ℕ → ℕ), (∀ n, b n = 3 * n - 2 → ∑' n, r ^ (b n) = 1 / 2 ) → (∀ n, a n = b n)

theorem unique_increasing_seq {r : ℝ} 
  (hr : 0.4 < r ∧ r < 0.5) 
  (hc : r^3 + 2*r = 1):
  ∃ a : ℕ → ℕ, (∀ n, a n = 3 * n - 2) ∧ (∑'(n), r^(a n) = 1/2) ∧ unique_seq a r :=
by
  sorry

end unique_increasing_seq_l1609_160905


namespace add_base3_numbers_l1609_160984

-- Definitions to represent the numbers in base 3
def base3_num1 := (2 : ℕ) -- 2_3
def base3_num2 := (2 * 3 + 2 : ℕ) -- 22_3
def base3_num3 := (2 * 3^2 + 0 * 3 + 2 : ℕ) -- 202_3
def base3_num4 := (2 * 3^3 + 0 * 3^2 + 2 * 3 + 2 : ℕ) -- 2022_3

-- Summing the numbers in base 10 first
def sum_base10 := base3_num1 + base3_num2 + base3_num3 + base3_num4

-- Expected result in base 10 for 21010_3
def result_base10 := 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3 + 0

-- Proof statement
theorem add_base3_numbers : sum_base10 = result_base10 :=
by {
  -- Proof not required, so we skip it using sorry
  sorry
}

end add_base3_numbers_l1609_160984


namespace time_to_paint_one_room_l1609_160990

theorem time_to_paint_one_room (total_rooms : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) (rooms_left : ℕ) :
  total_rooms = 9 ∧ rooms_painted = 5 ∧ time_remaining = 32 ∧ rooms_left = total_rooms - rooms_painted → time_remaining / rooms_left = 8 :=
by
  intros h
  sorry

end time_to_paint_one_room_l1609_160990


namespace solution_set_of_inequality_l1609_160955

-- Define the conditions and theorem
theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) : (1 / x < x) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x)) :=
by sorry

end solution_set_of_inequality_l1609_160955


namespace year_2013_is_not_special_l1609_160936

def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month * day = year % 100 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

theorem year_2013_is_not_special : ¬ is_special_year 2013 := by
  sorry

end year_2013_is_not_special_l1609_160936


namespace cards_selection_count_l1609_160903

noncomputable def numberOfWaysToChooseCards : Nat :=
  (Nat.choose 4 3) * 3 * (Nat.choose 13 2) * (13 ^ 2)

theorem cards_selection_count :
  numberOfWaysToChooseCards = 158184 := by
  sorry

end cards_selection_count_l1609_160903


namespace race_meeting_time_l1609_160935

noncomputable def track_length : ℕ := 500
noncomputable def first_meeting_from_marie_start : ℕ := 100
noncomputable def time_until_first_meeting : ℕ := 2
noncomputable def second_meeting_time : ℕ := 12

theorem race_meeting_time
  (h1 : track_length = 500)
  (h2 : first_meeting_from_marie_start = 100)
  (h3 : time_until_first_meeting = 2)
  (h4 : ∀ t v1 v2 : ℕ, t * (v1 + v2) = track_length)
  (h5 : 12 = second_meeting_time) :
  second_meeting_time = 12 := by
  sorry

end race_meeting_time_l1609_160935


namespace min_value_b_minus_a_l1609_160992

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_value_b_minus_a :
  ∀ (a : ℝ), ∃ (b : ℝ), b > 0 ∧ f a = g b ∧ ∀ (y : ℝ), b - a = 2 * Real.exp (y - 1 / 2) - Real.log y → y = 1 / 2 → b - a = 2 + Real.log 2 := by
  sorry

end min_value_b_minus_a_l1609_160992


namespace find_x_l1609_160999

theorem find_x (x : ℝ) (h : (20 + 30 + 40 + x) / 4 = 35) : x = 50 := by
  sorry

end find_x_l1609_160999


namespace correct_substitution_l1609_160933

theorem correct_substitution (x : ℝ) : 
    (2 * x - 7)^2 + (5 * x - 17.5)^2 = 0 → 
    x = 7 / 2 :=
by
  sorry

end correct_substitution_l1609_160933


namespace speed_of_man_l1609_160919

open Real Int

/-- 
  A train 110 m long is running with a speed of 40 km/h.
  The train passes a man who is running at a certain speed
  in the direction opposite to that in which the train is going.
  The train takes 9 seconds to pass the man.
  This theorem proves that the speed of the man is 3.992 km/h.
-/
theorem speed_of_man (T_length : ℝ) (T_speed : ℝ) (t_pass : ℝ) (M_speed : ℝ) : 
  T_length = 110 → T_speed = 40 → t_pass = 9 → M_speed = 3.992 :=
by
  intro h1 h2 h3
  sorry

end speed_of_man_l1609_160919


namespace fraction_irreducibility_l1609_160918

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducibility_l1609_160918


namespace correct_inequality_l1609_160937

theorem correct_inequality (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > ab ∧ ab > a :=
sorry

end correct_inequality_l1609_160937


namespace quadratic_inequality_solution_l1609_160979

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + a > 0 ↔ x ≠ -1/a) → a = 1 :=
by
  sorry

end quadratic_inequality_solution_l1609_160979


namespace card_distribution_l1609_160912

-- Definitions of the total cards and distribution rules
def total_cards : ℕ := 363

def ratio_xiaoming_xiaohua (k : ℕ) : Prop := ∃ x y, x = 7 * k ∧ y = 6 * k
def ratio_xiaogang_xiaoming (m : ℕ) : Prop := ∃ x z, z = 8 * m ∧ x = 5 * m

-- Final values to prove
def xiaoming_cards : ℕ := 105
def xiaohua_cards : ℕ := 90
def xiaogang_cards : ℕ := 168

-- The proof statement
theorem card_distribution (x y z k m : ℕ) 
  (hk : total_cards = 7 * k + 6 * k + 8 * m)
  (hx : ratio_xiaoming_xiaohua k)
  (hz : ratio_xiaogang_xiaoming m) :
  x = xiaoming_cards ∧ y = xiaohua_cards ∧ z = xiaogang_cards :=
by
  -- Placeholder for the proof
  sorry

end card_distribution_l1609_160912


namespace age_of_b_l1609_160922

variable (a b : ℕ)
variable (h1 : a * 3 = b * 5)
variable (h2 : (a + 2) * 2 = (b + 2) * 3)

theorem age_of_b : b = 6 :=
by
  sorry

end age_of_b_l1609_160922


namespace tax_free_amount_l1609_160971

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) 
    (tax_rate : ℝ) (exceeds_value : ℝ) :
    total_value = 1720 → 
    tax_rate = 0.11 → 
    tax_paid = 123.2 → 
    total_value - X = exceeds_value → 
    tax_paid = tax_rate * exceeds_value → 
    X = 600 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end tax_free_amount_l1609_160971


namespace smallest_five_digit_number_divisibility_l1609_160932

-- Define the smallest 5-digit number satisfying the conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_number_divisibility :
  ∃ (n : ℕ), isDivisibleBy n 15
          ∧ isDivisibleBy n (2^8)
          ∧ isDivisibleBy n 45
          ∧ isDivisibleBy n 54
          ∧ n >= 10000
          ∧ n < 100000
          ∧ n = 69120 :=
sorry

end smallest_five_digit_number_divisibility_l1609_160932


namespace wall_building_problem_l1609_160989

theorem wall_building_problem 
    (num_workers_1 : ℕ) (length_wall_1 : ℕ) (days_1 : ℕ)
    (num_workers_2 : ℕ) (length_wall_2 : ℕ) (days_2 : ℕ) :
    num_workers_1 = 8 → length_wall_1 = 140 → days_1 = 42 →
    num_workers_2 = 30 → length_wall_2 = 100 →
    (work_done : ℕ → ℕ → ℕ) → 
    (work_done length_wall_1 days_1 = num_workers_1 * days_1 * length_wall_1) →
    (work_done length_wall_2 days_2 = num_workers_2 * days_2 * length_wall_2) →
    (days_2 = 8) :=
by
  intros h1 h2 h3 h4 h5 wf wlen1 wlen2
  sorry

end wall_building_problem_l1609_160989


namespace original_curve_eqn_l1609_160927

-- Definitions based on conditions
def scaling_transformation_formula (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

-- The proof problem to be shown in Lean
theorem original_curve_eqn {x y : ℝ} (h : transformed_curve (2 * x) (3 * y)) :
  4 * x^2 + 9 * y^2 = 1 :=
sorry

end original_curve_eqn_l1609_160927


namespace mike_oranges_l1609_160926

-- Definitions and conditions
variables (O A B : ℕ)
def condition1 := A = 2 * O
def condition2 := B = O + A
def condition3 := O + A + B = 18

-- Theorem to prove that Mike received 3 oranges
theorem mike_oranges (h1 : condition1 O A) (h2 : condition2 O A B) (h3 : condition3 O A B) : 
  O = 3 := 
by 
  sorry

end mike_oranges_l1609_160926


namespace problem_statement_l1609_160930

variables (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def condition (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (2 - x)

theorem problem_statement (h_odd : is_odd f) (h_cond : condition f) : f 2010 = 0 := 
sorry

end problem_statement_l1609_160930


namespace consecutive_integers_divisible_by_12_l1609_160943

theorem consecutive_integers_divisible_by_12 (a b c d : ℤ) 
  (h1 : b = a + 1) (h2 : c = b + 1) (h3 : d = c + 1) : 
  12 ∣ (a * b + a * c + a * d + b * c + b * d + c * d + 1) := 
sorry

end consecutive_integers_divisible_by_12_l1609_160943


namespace todd_ratio_boss_l1609_160986

theorem todd_ratio_boss
  (total_cost : ℕ)
  (boss_contribution : ℕ)
  (employees_contribution : ℕ)
  (num_employees : ℕ)
  (each_employee_pay : ℕ) 
  (total_contributed : ℕ)
  (todd_contribution : ℕ) :
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  each_employee_pay = 11 →
  total_contributed = num_employees * each_employee_pay + boss_contribution →
  todd_contribution = total_cost - total_contributed →
  (todd_contribution : ℚ) / (boss_contribution : ℚ) = 2 := by
  sorry

end todd_ratio_boss_l1609_160986


namespace remaining_student_number_l1609_160901

theorem remaining_student_number (s1 s2 s3 : ℕ) (h1 : s1 = 5) (h2 : s2 = 29) (h3 : s3 = 41) (N : ℕ) (hN : N = 48) :
  ∃ s4, s4 < N ∧ s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3 ∧ (s4 = 17) :=
by
  sorry

end remaining_student_number_l1609_160901


namespace number_of_stanzas_is_correct_l1609_160949

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Define the number of lines per stanza
def lines_per_stanza : ℕ := 10

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Calculate the number of words per stanza
def words_per_stanza : ℕ := lines_per_stanza * words_per_line

-- Define the number of stanzas
def stanzas (total_words words_per_stanza : ℕ) := total_words / words_per_stanza

-- Theorem: Prove that given the conditions, the number of stanzas is 20
theorem number_of_stanzas_is_correct : stanzas total_words words_per_stanza = 20 :=
by
  -- Insert the proof here
  sorry

end number_of_stanzas_is_correct_l1609_160949


namespace intersection_point_of_lines_PQ_RS_l1609_160982

def point := ℝ × ℝ × ℝ

def P : point := (4, -3, 6)
def Q : point := (1, 10, 11)
def R : point := (3, -4, 2)
def S : point := (-1, 5, 16)

theorem intersection_point_of_lines_PQ_RS :
  let line_PQ (u : ℝ) := (4 - 3 * u, -3 + 13 * u, 6 + 5 * u)
  let line_RS (v : ℝ) := (3 - 4 * v, -4 + 9 * v, 2 + 14 * v)
  ∃ u v : ℝ,
    line_PQ u = line_RS v →
    line_PQ u = (19 / 5, 44 / 3, 23 / 3) :=
by
  sorry

end intersection_point_of_lines_PQ_RS_l1609_160982


namespace a7_b7_equals_29_l1609_160908

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29_l1609_160908


namespace evalCeilingOfNegativeSqrt_l1609_160966

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l1609_160966


namespace prime_factors_sum_correct_prime_factors_product_correct_l1609_160969

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end prime_factors_sum_correct_prime_factors_product_correct_l1609_160969


namespace midpoints_distance_l1609_160940

theorem midpoints_distance
  (A B C D M N : ℝ)
  (h1 : M = (A + C) / 2)
  (h2 : N = (B + D) / 2)
  (h3 : D - A = 68)
  (h4 : C - B = 26)
  : abs (M - N) = 21 := 
sorry

end midpoints_distance_l1609_160940


namespace stock_decrease_required_l1609_160972

theorem stock_decrease_required (x : ℝ) (h : x > 0) : 
  (∃ (p : ℝ), (1 - p) * 1.40 * x = x ∧ p * 100 = 28.57) :=
sorry

end stock_decrease_required_l1609_160972


namespace simplify_f_of_alpha_value_of_f_given_cos_l1609_160959

variable (α : Real) (f : Real → Real)

def third_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

noncomputable def f_def : Real → Real := 
  λ α => (Real.sin (α - Real.pi / 2) * 
           Real.cos (3 * Real.pi / 2 + α) * 
           Real.tan (Real.pi - α)) / 
           (Real.tan (-α - Real.pi) * 
           Real.sin (-Real.pi - α))

theorem simplify_f_of_alpha (h : third_quadrant α) :
  f α = -Real.cos α := sorry

theorem value_of_f_given_cos 
  (h : third_quadrant α) 
  (cos_h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := sorry

end simplify_f_of_alpha_value_of_f_given_cos_l1609_160959


namespace find_v_l1609_160900

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
    3, 0]

noncomputable def v : Matrix (Fin 2) (Fin 1) ℝ :=
  !![0;
    1 / 30.333]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_v : 
  (A ^ 10 + A ^ 8 + A ^ 6 + A ^ 4 + A ^ 2 + I) * v = !![0; 12] :=
  sorry

end find_v_l1609_160900
