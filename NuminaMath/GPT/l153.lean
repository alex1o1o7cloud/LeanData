import Mathlib

namespace roots_eqn_values_l153_153270

theorem roots_eqn_values : 
  ∀ (x1 x2 : ℝ), (x1^2 + x1 - 4 = 0) ∧ (x2^2 + x2 - 4 = 0) ∧ (x1 + x2 = -1)
  → (x1^3 - 5 * x2^2 + 10 = -19) := 
by
  intros x1 x2
  intros h
  sorry

end roots_eqn_values_l153_153270


namespace right_triangle_set_l153_153690

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_set :
  (is_right_triangle 3 4 2 = false) ∧
  (is_right_triangle 5 12 15 = false) ∧
  (is_right_triangle 8 15 17 = true) ∧
  (is_right_triangle (3^2) (4^2) (5^2) = false) :=
by
  sorry

end right_triangle_set_l153_153690


namespace largest_integer_is_190_l153_153439

theorem largest_integer_is_190 (A B C D : ℤ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D) 
  (h4 : (A + B + C + D) / 4 = 76) 
  (h5 : A = 37) 
  (h6 : B = 38) 
  (h7 : C = 39) : 
  D = 190 := 
sorry

end largest_integer_is_190_l153_153439


namespace two_degrees_above_zero_l153_153579

-- Define the concept of temperature notation
def temperature_notation (temp: ℝ) : String :=
  if temp < 0 then "-" ++ temp.nat_abs.toString ++ "°C"
  else "+" ++ temp.toString ++ "°C"

-- Given condition: -2 degrees Celsius is denoted as -2°C
def given_condition := temperature_notation (-2) = "-2°C"

-- Proof statement: 2 degrees Celsius above zero is denoted as +2°C given the condition
theorem two_degrees_above_zero : given_condition → temperature_notation 2 = "+2°C" := by
  intro h
  sorry

end two_degrees_above_zero_l153_153579


namespace exponentiation_identity_l153_153509

variable {a : ℝ}

theorem exponentiation_identity : (-a) ^ 2 * a ^ 3 = a ^ 5 := sorry

end exponentiation_identity_l153_153509


namespace circle_and_parabola_no_intersection_l153_153380

theorem circle_and_parabola_no_intersection (m : ℝ) (h : m ≠ 0) :
  (m > 0 ∨ m < -4) ↔
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) → (y^2 = 4 * m * x) → x ≠ -m := 
sorry

end circle_and_parabola_no_intersection_l153_153380


namespace five_twos_make_24_l153_153627

theorem five_twos_make_24 :
  ∃ a b c d e : ℕ, a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  ((a + b + c) * (d + e) = 24) :=
by
  sorry

end five_twos_make_24_l153_153627


namespace production_value_decreased_by_10_percent_l153_153184

variable (a : ℝ)

def production_value_in_January : ℝ := a

def production_value_in_February (a : ℝ) : ℝ := 0.9 * a

theorem production_value_decreased_by_10_percent (a : ℝ) :
  production_value_in_February a = 0.9 * production_value_in_January a := 
by
  sorry

end production_value_decreased_by_10_percent_l153_153184


namespace mapleton_math_team_combinations_l153_153445

open Nat

theorem mapleton_math_team_combinations (girls boys : ℕ) (team_size girl_on_team boy_on_team : ℕ)
    (h_girls : girls = 4) (h_boys : boys = 5) (h_team_size : team_size = 4)
    (h_girl_on_team : girl_on_team = 3) (h_boy_on_team : boy_on_team = 1) :
    (Nat.choose girls girl_on_team) * (Nat.choose boys boy_on_team) = 20 := by
  sorry

end mapleton_math_team_combinations_l153_153445


namespace slope_product_is_neg_one_l153_153296

noncomputable def slope_product (m n : ℝ) : ℝ := m * n

theorem slope_product_is_neg_one 
  (m n : ℝ)
  (eqn1 : ∀ x, ∃ y, y = m * x)
  (eqn2 : ∀ x, ∃ y, y = n * x)
  (angle : ∃ θ1 θ2 : ℝ, θ1 = θ2 + π / 4)
  (neg_reciprocal : m = -1 / n):
  slope_product m n = -1 := 
sorry

end slope_product_is_neg_one_l153_153296


namespace total_sleep_per_week_l153_153970

namespace TotalSleep

def hours_sleep_wd (days: Nat) : Nat := 6 * days
def hours_sleep_wknd (days: Nat) : Nat := 10 * days

theorem total_sleep_per_week : 
  hours_sleep_wd 5 + hours_sleep_wknd 2 = 50 := by
  sorry

end TotalSleep

end total_sleep_per_week_l153_153970


namespace total_cost_proof_l153_153652

def tuition_fee : ℕ := 1644
def room_and_board_cost : ℕ := tuition_fee - 704
def total_cost : ℕ := tuition_fee + room_and_board_cost

theorem total_cost_proof : total_cost = 2584 := 
by
  sorry

end total_cost_proof_l153_153652


namespace rohan_age_is_25_l153_153488

-- Define the current age of Rohan
def rohan_current_age (x : ℕ) : Prop :=
  x + 15 = 4 * (x - 15)

-- The goal is to prove that Rohan's current age is 25 years old
theorem rohan_age_is_25 : ∃ x : ℕ, rohan_current_age x ∧ x = 25 :=
by
  existsi (25 : ℕ)
  -- Proof is omitted since this is a statement only
  sorry

end rohan_age_is_25_l153_153488


namespace expected_value_of_8_sided_die_l153_153824

noncomputable def expected_value_winnings_8_sided_die : ℚ :=
  let probabilities := [1, 1, 1, 1, 1, 1, 1, 1].map (λ x => (1 : ℚ) / 8)
  let winnings := [0, 0, 0, 0, 2, 4, 6, 8]
  let expected_value := (List.zipWith (*) probabilities winnings).sum
  expected_value

theorem expected_value_of_8_sided_die : expected_value_winnings_8_sided_die = 2.5 :=
by
  sorry

end expected_value_of_8_sided_die_l153_153824


namespace cheeseburger_cost_l153_153925

-- Definitions for given conditions
def milkshake_price : ℝ := 5
def cheese_fries_price : ℝ := 8
def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money := jim_money + cousin_money
def spending_percentage : ℝ := 0.80
def total_spent := spending_percentage * combined_money
def number_of_milkshakes : ℝ := 2
def number_of_cheeseburgers : ℝ := 2

-- Prove the cost of one cheeseburger
theorem cheeseburger_cost : (total_spent - (number_of_milkshakes * milkshake_price) - cheese_fries_price) / number_of_cheeseburgers = 3 :=
by
  sorry

end cheeseburger_cost_l153_153925


namespace find_constants_l153_153717

noncomputable section

theorem find_constants (P Q R : ℝ)
  (h : ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
    (5*x^2 + 7*x) / ((x - 2) * (x - 4)^2) =
    P / (x - 2) + Q / (x - 4) + R / (x - 4)^2) :
  P = 3.5 ∧ Q = 1.5 ∧ R = 18 :=
by
  sorry

end find_constants_l153_153717


namespace cos_pi_plus_2alpha_l153_153371

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin ((Real.pi / 2) + α) = 1 / 3) : Real.cos (Real.pi + 2 * α) = 7 / 9 :=
by
  sorry

end cos_pi_plus_2alpha_l153_153371


namespace math_problem_l153_153226

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l153_153226


namespace new_average_of_remaining_students_l153_153438

theorem new_average_of_remaining_students 
  (avg_initial_score : ℝ)
  (num_initial_students : ℕ)
  (dropped_score : ℝ)
  (num_remaining_students : ℕ)
  (new_avg_score : ℝ) 
  (h_avg : avg_initial_score = 62.5)
  (h_num_initial : num_initial_students = 16)
  (h_dropped : dropped_score = 55)
  (h_num_remaining : num_remaining_students = 15)
  (h_new_avg : new_avg_score = 63) :
  let total_initial_score := avg_initial_score * num_initial_students
  let total_remaining_score := total_initial_score - dropped_score
  let calculated_new_avg := total_remaining_score / num_remaining_students
  calculated_new_avg = new_avg_score := 
by
  -- The proof will be provided here
  sorry

end new_average_of_remaining_students_l153_153438


namespace right_triangle_acute_angle_ratio_l153_153911

theorem right_triangle_acute_angle_ratio (A B : ℝ) (h_ratio : A / B = 5 / 4) (h_sum : A + B = 90) :
  min A B = 40 :=
by
  -- Conditions are provided
  sorry

end right_triangle_acute_angle_ratio_l153_153911


namespace shorter_side_of_rectangle_l153_153300

variable (R : Type) [LinearOrderedField R]

noncomputable def findShorterSide (a b : R) : Prop :=
  let d : R := real.sqrt (a^2 + b^2)
  let x : R := a / 3
  let y : R := b / 4
  a = 3 * x ∧ b = 4 * x ∧ (d = 9) →

  (3 * (9 / 5) = 5.4)

theorem shorter_side_of_rectangle : ∀ (a b : R), a = 3 * (9 / 5) → findShorterSide a b :=
by
  sorry

end shorter_side_of_rectangle_l153_153300


namespace max_area_height_l153_153091

theorem max_area_height (h : ℝ) (x : ℝ) 
  (right_trapezoid : True) 
  (angle_30_deg : True) 
  (perimeter_eq_6 : 3 * (x + h) = 6) : 
  h = 1 :=
by 
  sorry

end max_area_height_l153_153091


namespace inequality_solution_set_l153_153394

theorem inequality_solution_set (a b : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, ax^2 + bx - 1 < 0 ↔ -1/2 < x ∧ x < 1) :
  ∀ x : ℝ, (2 * x + 2) / (-x + 1) < 0 ↔ (x < -1 ∨ x > 1) :=
by sorry

end inequality_solution_set_l153_153394


namespace koala_fiber_eaten_l153_153405

def koala_fiber_absorbed (fiber_eaten : ℝ) : ℝ := 0.30 * fiber_eaten

theorem koala_fiber_eaten (absorbed : ℝ) (fiber_eaten : ℝ) 
  (h_absorbed : absorbed = koala_fiber_absorbed fiber_eaten) : fiber_eaten = 40 :=
by {
  have h1 : fiber_eaten * 0.30 = absorbed,
  rw h_absorbed,
  have : 12 = absorbed,
  rw this,
  sorry,
}

end koala_fiber_eaten_l153_153405


namespace complement_intersection_l153_153574

open Set

-- Define the universal set I, and sets M and N
def I : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

-- Lean statement to prove the desired result
theorem complement_intersection : (I \ N) ∩ M = {1, 2} := by
  sorry

end complement_intersection_l153_153574


namespace smallest_k_divides_l153_153539

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l153_153539


namespace max_geq_four_ninths_sum_min_leq_quarter_sum_l153_153776

theorem max_geq_four_ninths_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  max a (max b c) >= 4 / 9 * (a + b + c) :=
by 
  sorry

theorem min_leq_quarter_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  min a (min b c) <= 1 / 4 * (a + b + c) :=
by 
  sorry

end max_geq_four_ninths_sum_min_leq_quarter_sum_l153_153776


namespace compute_Z_value_l153_153897

def operation_Z (c d : ℕ) : ℤ := c^2 - 3 * c * d + d^2

theorem compute_Z_value : operation_Z 4 3 = -11 := by
  sorry

end compute_Z_value_l153_153897


namespace more_boys_than_girls_l153_153759

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l153_153759


namespace female_managers_count_l153_153904

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end female_managers_count_l153_153904


namespace least_four_digit_with_factors_l153_153466

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l153_153466


namespace sqrt_three_pow_divisible_l153_153090

/-- For any non-negative integer n, (1 + sqrt 3)^(2*n + 1) is divisible by 2^(n + 1) -/
theorem sqrt_three_pow_divisible (n : ℕ) :
  ∃ k : ℕ, (⌊(1 + Real.sqrt 3)^(2 * n + 1)⌋ : ℝ) = k * 2^(n + 1) :=
sorry

end sqrt_three_pow_divisible_l153_153090


namespace nth_equation_holds_l153_153018

theorem nth_equation_holds (n : ℕ) (h : 0 < n) :
  1 / (n + 2) + 2 / (n^2 + 2 * n) = 1 / n :=
by
  sorry

end nth_equation_holds_l153_153018


namespace pasta_ratio_l153_153990

theorem pasta_ratio (students_surveyed : ℕ) (spaghetti_preferred : ℕ) (manicotti_preferred : ℕ) 
(h_total : students_surveyed = 800) 
(h_spaghetti : spaghetti_preferred = 320) 
(h_manicotti : manicotti_preferred = 160) : 
(spaghetti_preferred / manicotti_preferred : ℚ) = 2 := by
  sorry

end pasta_ratio_l153_153990


namespace each_girl_gets_2_dollars_l153_153639

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l153_153639


namespace solution_set_l153_153013

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f_pos (x : ℝ) : deriv f x > 1 - f x
axiom f_at_zero : f 0 = 3

theorem solution_set (x : ℝ) : e^x * f x > e^x + 2 ↔ x > 0 :=
by sorry

end solution_set_l153_153013


namespace problem_solution_l153_153591

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l153_153591


namespace boys_in_2nd_l153_153458

def students_in_3rd := 19
def students_in_4th := 2 * students_in_3rd
def girls_in_2nd := 19
def total_students := 86
def students_in_2nd := total_students - students_in_3rd - students_in_4th

theorem boys_in_2nd : students_in_2nd - girls_in_2nd = 10 := by
  sorry

end boys_in_2nd_l153_153458


namespace statement_books_per_shelf_l153_153498

/--
A store initially has 40.0 coloring books.
Acquires 20.0 more books.
Uses 15 shelves to store the books equally.
-/
def initial_books : ℝ := 40.0
def acquired_books : ℝ := 20.0
def total_shelves : ℝ := 15.0

/-- 
Theorem statement: The number of coloring books on each shelf.
-/
theorem books_per_shelf : (initial_books + acquired_books) / total_shelves = 4.0 := by
  sorry

end statement_books_per_shelf_l153_153498


namespace find_quotient_l153_153292

theorem find_quotient
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1200)
  (h2 : larger = 1495)
  (rem : ℕ := 4)
  (h3 : larger % smaller = rem) :
  larger / smaller = 5 := 
by 
  sorry

end find_quotient_l153_153292


namespace ab_greater_than_a_plus_b_l153_153778

theorem ab_greater_than_a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a - b = a / b) : ab > a + b :=
sorry

end ab_greater_than_a_plus_b_l153_153778


namespace factor_and_sum_coeffs_l153_153712

noncomputable def sum_of_integer_coeffs_of_factorization (x y : ℤ) : ℤ :=
  let factors := ([(1 : ℤ), (-1 : ℤ), (5 : ℤ), (1 : ℤ), (6 : ℤ), (1 : ℤ), (1 : ℤ), (5 : ℤ), (-1 : ℤ), (6 : ℤ)])
  factors.sum

theorem factor_and_sum_coeffs (x y : ℤ) :
  (125 * (x^9:ℤ) - 216 * (y^9:ℤ) = (x - y) * (5 * x^2 + x * y + 6 * y^2) * (x + y) * (5 * x^2 - x * y + 6 * y^2))
  ∧ (sum_of_integer_coeffs_of_factorization x y = 24) :=
by
  sorry

end factor_and_sum_coeffs_l153_153712


namespace simplify_expression_l153_153773

variables {x p q r : ℝ}

theorem simplify_expression (h1 : p ≠ q) (h2 : p ≠ r) (h3 : q ≠ r) :
   ( (x + p)^4 / ((p - q) * (p - r)) + (x + q)^4 / ((q - p) * (q - r)) + (x + r)^4 / ((r - p) * (r - q)) 
   ) = p + q + r + 4 * x :=
sorry

end simplify_expression_l153_153773


namespace find_b_l153_153111

-- Definitions based on the conditions in the problem
def eq1 (a : ℝ) := 3 * a + 3 = 0
def eq2 (a b : ℝ) := 2 * b - a = 4

-- Statement of the proof problem
theorem find_b (a b : ℝ) (h1 : eq1 a) (h2 : eq2 a b) : b = 3 / 2 :=
by
  sorry

end find_b_l153_153111


namespace smallest_n_with_digits_315_l153_153028

-- Defining the conditions
def relatively_prime (m n : ℕ) := Nat.gcd m n = 1
def valid_fraction (m n : ℕ) := (m < n) ∧ relatively_prime m n

-- Predicate for the sequence 3, 1, 5 in the decimal representation of m/n
def contains_digits_315 (m n : ℕ) : Prop :=
  ∃ k d : ℕ, 10^k * m % n = 315 * 10^(d - 3) ∧ d ≥ 3

-- The main theorem: smallest n for which the conditions are satisfied
theorem smallest_n_with_digits_315 :
  ∃ n : ℕ, valid_fraction m n ∧ contains_digits_315 m n ∧ n = 159 :=
sorry

end smallest_n_with_digits_315_l153_153028


namespace find_correct_value_l153_153671

-- Definitions based on the problem's conditions
def incorrect_calculation (x : ℤ) : Prop := 7 * x = 126
def correct_value (x : ℤ) (y : ℤ) : Prop := x / 6 = y

theorem find_correct_value :
  ∃ (x y : ℤ), incorrect_calculation x ∧ correct_value x y ∧ y = 3 := by
  sorry

end find_correct_value_l153_153671


namespace sum_of_roots_l153_153377

theorem sum_of_roots (x1 x2 : ℝ) (h1 : x1^2 + 5*x1 - 3 = 0) (h2 : x2^2 + 5*x2 - 3 = 0) (h3 : x1 ≠ x2) :
  x1 + x2 = -5 :=
sorry

end sum_of_roots_l153_153377


namespace middle_part_of_proportion_l153_153519

theorem middle_part_of_proportion (x : ℚ) (h : x + (1/4) * x + (1/8) * x = 104) : (1/4) * x = 208 / 11 :=
by
  sorry

end middle_part_of_proportion_l153_153519


namespace prime_sum_of_digits_base_31_l153_153721

-- Define the sum of digits function in base k
def sum_of_digits_in_base (k n : ℕ) : ℕ :=
  let digits := (Nat.digits k n)
  digits.foldr (· + ·) 0

theorem prime_sum_of_digits_base_31 (p : ℕ) (hp : Nat.Prime p) (h_bound : p < 20000) : 
  sum_of_digits_in_base 31 p = 49 ∨ sum_of_digits_in_base 31 p = 77 :=
by
  sorry

end prime_sum_of_digits_base_31_l153_153721


namespace alpha_beta_range_l153_153198

theorem alpha_beta_range (α β : ℝ) (P : ℝ × ℝ)
  (h1 : α > 0) 
  (h2 : β > 0) 
  (h3 : P = (α, 3 * β))
  (circle_eq : (α - 1)^2 + 9 * (β^2) = 1) :
  1 < α + β ∧ α + β < 5 / 3 :=
sorry

end alpha_beta_range_l153_153198


namespace percentage_of_millet_in_Brand_A_l153_153994

variable (A B : ℝ)
variable (B_percent : B = 0.65)
variable (mix_millet_percent : 0.60 * A + 0.40 * B = 0.50)

theorem percentage_of_millet_in_Brand_A :
  A = 0.40 :=
by
  sorry

end percentage_of_millet_in_Brand_A_l153_153994


namespace purchase_probability_l153_153054

/--
A batch of products from a company has packages containing 10 components each.
Each package has either 1 or 2 second-grade components. 10% of the packages
contain 2 second-grade components. Xiao Zhang will decide to purchase
if all 4 randomly selected components from a package are first-grade.

We aim to prove the probability that Xiao Zhang decides to purchase the company's
products is \( \frac{43}{75} \).
-/
theorem purchase_probability : true := sorry

end purchase_probability_l153_153054


namespace triangle_side_length_l153_153920

theorem triangle_side_length (A B C : Type*) [inner_product_space ℝ (A × B)]
  (AB AC BC : ℝ) (h1 : AB = 2)
  (h2 : AC = 3) (h3 : BC = sqrt(5.2)) :
  ACS = sqrt(5.2) :=
sorry

end triangle_side_length_l153_153920


namespace cake_sector_chord_length_l153_153327

noncomputable def sector_longest_chord_square (d : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let theta := (360 : ℝ) / n
  let chord_length := 2 * r * Real.sin (theta / 2 * Real.pi / 180)
  chord_length ^ 2

theorem cake_sector_chord_length :
  sector_longest_chord_square 18 5 = 111.9473 := by
  sorry

end cake_sector_chord_length_l153_153327


namespace longest_side_of_garden_l153_153940

theorem longest_side_of_garden (l w : ℝ) (h1 : 2 * l + 2 * w = 225) (h2 : l * w = 8 * 225) :
  l = 93.175 ∨ w = 93.175 :=
by
  sorry

end longest_side_of_garden_l153_153940


namespace geometric_sequence_root_product_l153_153253

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l153_153253


namespace vertical_distance_l153_153145

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l153_153145


namespace find_n_series_sum_l153_153042

theorem find_n_series_sum 
  (first_term_I : ℝ) (second_term_I : ℝ) (first_term_II : ℝ) (second_term_II : ℝ) (sum_multiplier : ℝ) (n : ℝ)
  (h_I_first_term : first_term_I = 12)
  (h_I_second_term : second_term_I = 4)
  (h_II_first_term : first_term_II = 12)
  (h_II_second_term : second_term_II = 4 + n)
  (h_sum_multiplier : sum_multiplier = 5) :
  n = 152 :=
by
  sorry

end find_n_series_sum_l153_153042


namespace parabola_vertex_l153_153295

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x y : ℝ, y^2 + 8*y + 4*x + 9 = 0 → x = -1/4 * (y + 4)^2 + 7/4)
  := 
  ⟨7/4, -4, sorry⟩

end parabola_vertex_l153_153295


namespace smallest_k_l153_153540

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l153_153540


namespace common_chord_line_l153_153444

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line_l153_153444


namespace circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l153_153076

-- Circle 1 with center (8, -3) and passing through point (5, 1)
theorem circle_centered_at_8_neg3_passing_through_5_1 :
  ∃ r : ℝ, (r = 5) ∧ ((x - 8: ℝ)^2 + (y + 3)^2 = r^2) := by
  sorry

-- Circle passing through points A(-1, 5), B(5, 5), and C(6, -2)
theorem circle_passing_through_ABC :
  ∃ D E F : ℝ, (D = -4) ∧ (E = -2) ∧ (F = -20) ∧
    ( ∀ (x : ℝ) (y : ℝ), (x = -1 ∧ y = 5) 
      ∨ (x = 5 ∧ y = 5) 
      ∨ (x = 6 ∧ y = -2) 
      → (x^2 + y^2 + D*x + E*y + F = 0)) := by
  sorry

end circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l153_153076


namespace ways_to_make_50_cents_without_dimes_or_quarters_l153_153894

theorem ways_to_make_50_cents_without_dimes_or_quarters : 
  ∃ (n : ℕ), n = 1024 := 
by
  let num_ways := (2 ^ 10)
  existsi num_ways
  sorry

end ways_to_make_50_cents_without_dimes_or_quarters_l153_153894


namespace Marcy_spears_l153_153015

def makeSpears (saplings: ℕ) (logs: ℕ) (branches: ℕ) (trunks: ℕ) : ℕ :=
  3 * saplings + 9 * logs + 7 * branches + 15 * trunks

theorem Marcy_spears :
  makeSpears 12 1 6 0 - (3 * 2) + makeSpears 0 4 0 0 - (9 * 4) + makeSpears 0 0 6 1 - (7 * 0) + makeSpears 0 0 0 2 = 81 := by
  sorry

end Marcy_spears_l153_153015


namespace find_required_water_amount_l153_153210

-- Definitions based on the conditions
def sanitizer_volume : ℝ := 12
def initial_alcohol_concentration : ℝ := 0.60
def desired_alcohol_concentration : ℝ := 0.40

-- Statement of the proof problem
theorem find_required_water_amount : 
  ∃ (x : ℝ), x = 6 ∧ sanitizer_volume * initial_alcohol_concentration = desired_alcohol_concentration * (sanitizer_volume + x) :=
sorry

end find_required_water_amount_l153_153210


namespace sin_x_correct_l153_153385

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

end sin_x_correct_l153_153385


namespace daily_earnings_r_l153_153482

theorem daily_earnings_r (p q r s : ℝ)
  (h1 : p + q + r + s = 300)
  (h2 : p + r = 120)
  (h3 : q + r = 130)
  (h4 : s + r = 200)
  (h5 : p + s = 116.67) : 
  r = 75 :=
by
  sorry

end daily_earnings_r_l153_153482


namespace square_perimeter_l153_153914

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end square_perimeter_l153_153914


namespace range_of_t_l153_153885

noncomputable section

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 3^x
else if 1 < x ∧ x ≤ 3 then (9 / 2) - (3 / 2) * x
else 0

theorem range_of_t (t : ℝ) (h1 : 0 ≤ t ∧ t ≤ 1) (h2 : (0 ≤ f (f t)) ∧ (f (f t) ≤ 1)) :
  (real.log (7 / 3) / real.log 3) ≤ t ∧ t ≤ 1 :=
sorry

end range_of_t_l153_153885


namespace certain_number_l153_153119

theorem certain_number (x y a : ℤ) (h1 : 4 * x + y = a) (h2 : 2 * x - y = 20) 
  (h3 : y ^ 2 = 4) : a = 46 :=
sorry

end certain_number_l153_153119


namespace problem_inequality_l153_153223

theorem problem_inequality 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m + n = 1) : 
  (m + 1 / m) * (n + 1 / n) ≥ 25 / 4 := 
sorry

end problem_inequality_l153_153223


namespace find_divisor_l153_153987

theorem find_divisor (d : ℕ) : 15 = (d * 4) + 3 → d = 3 := by
  intros h
  have h1 : 15 - 3 = 4 * d := by
    linarith
  have h2 : 12 = 4 * d := by
    linarith
  have h3 : d = 3 := by
    linarith
  exact h3

end find_divisor_l153_153987


namespace minimum_value_of_expression_l153_153616

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (3 / (a + b + c)^2) ≥ 2 :=
sorry

end minimum_value_of_expression_l153_153616


namespace cristine_initial_lemons_l153_153851

theorem cristine_initial_lemons (L : ℕ) (h : (3 / 4 : ℚ) * L = 9) : L = 12 :=
sorry

end cristine_initial_lemons_l153_153851


namespace ordered_pair_solution_l153_153516

theorem ordered_pair_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end ordered_pair_solution_l153_153516


namespace necessarily_true_statement_l153_153063

-- Define the four statements as propositions
def Statement1 (d : ℕ) : Prop := d = 2
def Statement2 (d : ℕ) : Prop := d ≠ 3
def Statement3 (d : ℕ) : Prop := d = 5
def Statement4 (d : ℕ) : Prop := d % 2 = 0

-- The main theorem stating that given one of the statements is false, Statement3 is necessarily true
theorem necessarily_true_statement (d : ℕ) 
  (h1 : Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ ¬ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ ¬ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ ¬ Statement2 d ∧ Statement3 d ∧ Statement4 d)):
  Statement2 d :=
sorry

end necessarily_true_statement_l153_153063


namespace evaluate_expression_l153_153081

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l153_153081


namespace gcd_determinant_l153_153895

theorem gcd_determinant (a b : ℤ) (h : Int.gcd a b = 1) :
  Int.gcd (a + b) (a^2 + b^2 - a * b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a * b) = 3 :=
sorry

end gcd_determinant_l153_153895


namespace decagon_side_length_in_rectangle_l153_153399

theorem decagon_side_length_in_rectangle
  (AB CD : ℝ)
  (AE FB : ℝ)
  (s : ℝ)
  (cond1 : AB = 10)
  (cond2 : CD = 15)
  (cond3 : AE = 5)
  (cond4 : FB = 5)
  (regular_decagon : ℝ → Prop)
  (h : regular_decagon s) : 
  s = 5 * (Real.sqrt 2 - 1) :=
by 
  sorry

end decagon_side_length_in_rectangle_l153_153399


namespace merchant_articles_l153_153332

theorem merchant_articles (N CP SP : ℝ) 
  (h1 : N * CP = 16 * SP)
  (h2 : SP = CP * 1.375) : 
  N = 22 :=
by
  sorry

end merchant_articles_l153_153332


namespace primes_digit_sum_difference_l153_153889

def is_prime (a : ℕ) : Prop := Nat.Prime a

def sum_digits (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

theorem primes_digit_sum_difference (p q r : ℕ) (n : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (hneq : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (hpqr : p * q * r = 1899 * 10^n + 962) :
  (sum_digits p + sum_digits q + sum_digits r - sum_digits (p * q * r) = 8) := 
sorry

end primes_digit_sum_difference_l153_153889


namespace greatest_integer_gcd_is_4_l153_153043

theorem greatest_integer_gcd_is_4 : 
  ∀ (n : ℕ), n < 150 ∧ (Nat.gcd n 24 = 4) → n ≤ 148 := 
by
  sorry

end greatest_integer_gcd_is_4_l153_153043


namespace surface_area_of_cube_l153_153303

-- Define the condition: volume of the cube is 1728 cubic centimeters
def volume_cube (s : ℝ) : ℝ := s^3
def given_volume : ℝ := 1728

-- Define the question: surface area of the cube
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- The statement that needs to be proved
theorem surface_area_of_cube :
  ∃ s : ℝ, volume_cube s = given_volume → surface_area_cube s = 864 :=
by
  sorry

end surface_area_of_cube_l153_153303


namespace negation_correct_l153_153793

def original_statement (a : ℝ) : Prop :=
  a > 0 → a^2 > 0

def negated_statement (a : ℝ) : Prop :=
  a ≤ 0 → a^2 ≤ 0

theorem negation_correct (a : ℝ) : ¬ (original_statement a) ↔ negated_statement a :=
by
  sorry

end negation_correct_l153_153793


namespace john_and_lisa_meet_at_midpoint_l153_153612

-- Define the conditions
def john_position : ℝ × ℝ := (2, 9)
def lisa_position : ℝ × ℝ := (-6, 1)

-- Assertion for their meeting point
theorem john_and_lisa_meet_at_midpoint :
  ∃ (x y : ℝ), (x, y) = ((john_position.1 + lisa_position.1) / 2,
                         (john_position.2 + lisa_position.2) / 2) :=
sorry

end john_and_lisa_meet_at_midpoint_l153_153612


namespace find_positive_integral_solution_l153_153860

theorem find_positive_integral_solution :
  ∃ n : ℕ, n > 0 ∧ (n - 1) * 101 = (n + 1) * 100 := by
sorry

end find_positive_integral_solution_l153_153860


namespace greatest_s_property_l153_153265

noncomputable def find_greatest_s (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
if h : m > 0 ∧ n > 0 then m else 0

theorem greatest_s_property (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] (H : 0 < m) (H1 : 0 < n)  :
  ∃ s, (s = find_greatest_s m n p) ∧ s * n * p ≤ m * n * p :=
by 
  sorry

end greatest_s_property_l153_153265


namespace sum_of_roots_l153_153817

theorem sum_of_roots {a b : Real} (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) : a + b = 4 :=
by
  sorry

end sum_of_roots_l153_153817


namespace Arman_worked_last_week_l153_153007

variable (H : ℕ) -- hours worked last week
variable (wage_last_week wage_this_week : ℝ)
variable (hours_this_week worked_this_week two_weeks_earning : ℝ)
variable (worked_last_week : Prop)

-- Define assumptions based on the problem conditions
def condition1 : wage_last_week = 10 := by sorry
def condition2 : wage_this_week = 10.5 := by sorry
def condition3 : hours_this_week = 40 := by sorry
def condition4 : worked_this_week = wage_this_week * hours_this_week := by sorry
def condition5 : worked_this_week = 420 := by sorry -- 10.5 * 40
def condition6 : two_weeks_earning = wage_last_week * (H : ℝ) + worked_this_week := by sorry
def condition7 : two_weeks_earning = 770 := by sorry

-- Proof statement
theorem Arman_worked_last_week : worked_last_week := by
  have h1 : wage_last_week * (H : ℝ) + worked_this_week = two_weeks_earning := sorry
  have h2 : wage_last_week * (H : ℝ) + 420 = 770 := sorry
  have h3 : wage_last_week * (H : ℝ) = 350 := sorry
  have h4 : (10 : ℝ) * (H : ℝ) = 350 := sorry
  have h5 : H = 35 := sorry
  sorry

end Arman_worked_last_week_l153_153007


namespace unique_passenger_counts_l153_153838

def train_frequencies : Nat × Nat × Nat := (6, 4, 3)
def train_passengers_leaving : Nat × Nat × Nat := (200, 300, 150)
def train_passengers_taking : Nat × Nat × Nat := (320, 400, 280)
def trains_per_hour (freq : Nat) : Nat := 60 / freq

def total_passengers_leaving : Nat :=
  let t1 := (trains_per_hour 10) * 200
  let t2 := (trains_per_hour 15) * 300
  let t3 := (trains_per_hour 20) * 150
  t1 + t2 + t3

def total_passengers_taking : Nat :=
  let t1 := (trains_per_hour 10) * 320
  let t2 := (trains_per_hour 15) * 400
  let t3 := (trains_per_hour 20) * 280
  t1 + t2 + t3

theorem unique_passenger_counts :
  total_passengers_leaving = 2850 ∧ total_passengers_taking = 4360 := by
  sorry

end unique_passenger_counts_l153_153838


namespace total_ticket_cost_is_correct_l153_153507

-- Definitions based on the conditions provided
def child_ticket_cost : ℝ := 4.25
def adult_ticket_cost : ℝ := child_ticket_cost + 3.50
def senior_ticket_cost : ℝ := adult_ticket_cost - 1.75

def number_adult_tickets : ℕ := 2
def number_child_tickets : ℕ := 4
def number_senior_tickets : ℕ := 1

def total_ticket_cost_before_discount : ℝ := 
  number_adult_tickets * adult_ticket_cost + 
  number_child_tickets * child_ticket_cost + 
  number_senior_tickets * senior_ticket_cost

def total_tickets : ℕ := number_adult_tickets + number_child_tickets + number_senior_tickets
def discount : ℝ := if total_tickets >= 5 then 3.0 else 0.0

def total_ticket_cost_after_discount : ℝ := total_ticket_cost_before_discount - discount

-- The proof statement: proving the total ticket cost after the discount is $35.50
theorem total_ticket_cost_is_correct : total_ticket_cost_after_discount = 35.50 := by
  -- Note: The exact solution is omitted and replaced with sorry to denote where the proof would be.
  sorry

end total_ticket_cost_is_correct_l153_153507


namespace dave_winfield_home_runs_l153_153068

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l153_153068


namespace smallest_k_divides_l153_153551

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l153_153551


namespace find_S_l153_153052

theorem find_S (R S T : ℝ) (c : ℝ)
  (h1 : R = c * (S / T))
  (h2 : R = 2) (h3 : S = 1/2) (h4 : T = 4/3) (h_c : c = 16/3)
  (h_R : R = Real.sqrt 75) (h_T : T = Real.sqrt 32) :
  S = 45/4 := by
  sorry

end find_S_l153_153052


namespace find_n_in_geom_series_l153_153454

noncomputable def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem find_n_in_geom_series :
  ∃ n : ℕ, geom_sum 1 (1/2) n = 31 / 16 :=
sorry

end find_n_in_geom_series_l153_153454


namespace smallest_k_divides_l153_153552

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l153_153552


namespace cos_pi_div_4_minus_alpha_l153_153561

theorem cos_pi_div_4_minus_alpha (α : ℝ) (h : Real.sin (α + π/4) = 5/13) : 
  Real.cos (π/4 - α) = 5/13 :=
by
  sorry

end cos_pi_div_4_minus_alpha_l153_153561


namespace non_rain_hours_correct_l153_153092

def total_hours : ℕ := 9
def rain_hours : ℕ := 4

theorem non_rain_hours_correct : (total_hours - rain_hours) = 5 := 
by
  sorry

end non_rain_hours_correct_l153_153092


namespace older_brother_pocket_money_l153_153140

-- Definitions of the conditions
axiom sum_of_pocket_money (O Y : ℕ) : O + Y = 12000
axiom older_brother_more (O Y : ℕ) : O = Y + 1000

-- The statement to prove
theorem older_brother_pocket_money (O Y : ℕ) (h1 : O + Y = 12000) (h2 : O = Y + 1000) : O = 6500 :=
by
  exact sorry  -- Placeholder for the proof

end older_brother_pocket_money_l153_153140


namespace scientific_notation_of_million_l153_153686

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end scientific_notation_of_million_l153_153686


namespace sweets_ratio_l153_153191

theorem sweets_ratio (total_sweets : ℕ) (mother_ratio : ℚ) (eldest_sweets second_sweets : ℕ)
  (h1 : total_sweets = 27) (h2 : mother_ratio = 1 / 3) (h3 : eldest_sweets = 8) (h4 : second_sweets = 6) :
  let mother_sweets := mother_ratio * total_sweets
  let remaining_sweets := total_sweets - mother_sweets
  let other_sweets := eldest_sweets + second_sweets
  let youngest_sweets := remaining_sweets - other_sweets
  youngest_sweets / eldest_sweets = 1 / 2 :=
by
  sorry

end sweets_ratio_l153_153191


namespace total_trees_after_planting_l153_153305

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l153_153305


namespace square_pyramid_properties_l153_153313

-- Definitions for the square pyramid with a square base
def square_pyramid_faces : Nat := 4 + 1
def square_pyramid_edges : Nat := 4 + 4
def square_pyramid_vertices : Nat := 4 + 1

-- Definition for the number of diagonals in a square
def diagonals_in_square_base (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem statement
theorem square_pyramid_properties :
  (square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18) ∧ (diagonals_in_square_base 4 = 2) :=
by
  sorry

end square_pyramid_properties_l153_153313


namespace geralds_average_speed_l153_153151

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end geralds_average_speed_l153_153151


namespace polygon_sides_l153_153966

theorem polygon_sides (s : ℕ) (h : 180 * (s - 2) = 720) : s = 6 :=
by
  sorry

end polygon_sides_l153_153966


namespace average_distinct_u_l153_153106

theorem average_distinct_u :
  ∀ (u : ℕ), (∃ a b : ℕ, a + b = 6 ∧ ab = u) →
  {u | ∃ a b : ℕ, a + b = 6 ∧ ab = u}.to_finset.val.sum / 3 = 22 / 3 :=
sorry

end average_distinct_u_l153_153106


namespace nat_power_digit_condition_l153_153088

theorem nat_power_digit_condition (n k : ℕ) : 
  (10^(k-1) < n^n ∧ n^n < 10^k) → (10^(n-1) < k^k ∧ k^k < 10^n) → 
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end nat_power_digit_condition_l153_153088


namespace f_value_third_quadrant_l153_153372

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3 * Real.pi / 2 + α))

theorem f_value_third_quadrant (α : ℝ) (h1 : (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)) (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end f_value_third_quadrant_l153_153372


namespace total_weight_correct_l153_153993

-- Definitions of the given weights of materials
def weight_concrete : ℝ := 0.17
def weight_bricks : ℝ := 0.237
def weight_sand : ℝ := 0.646
def weight_stone : ℝ := 0.5
def weight_steel : ℝ := 1.73
def weight_wood : ℝ := 0.894

-- Total weight of all materials
def total_weight : ℝ := 
  weight_concrete + weight_bricks + weight_sand + weight_stone + weight_steel + weight_wood

-- The proof statement
theorem total_weight_correct : total_weight = 4.177 := by
  sorry

end total_weight_correct_l153_153993


namespace sufficient_but_not_necessary_condition_l153_153278

theorem sufficient_but_not_necessary_condition (a1 d : ℝ) : 
  (2 * a1 + 11 * d > 0) → (2 * a1 + 11 * d ≥ 0) :=
by
  intro h
  apply le_of_lt
  exact h

end sufficient_but_not_necessary_condition_l153_153278


namespace sum_of_smallest_multiples_l153_153933

def smallest_two_digit_multiple_of_5 := 10
def smallest_three_digit_multiple_of_7 := 105

theorem sum_of_smallest_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_smallest_multiples_l153_153933


namespace jessica_coins_worth_l153_153611

theorem jessica_coins_worth :
  ∃ (n d : ℕ), n + d = 30 ∧ 5 * (30 - d) + 10 * d = 165 :=
by {
  sorry
}

end jessica_coins_worth_l153_153611


namespace non_degenerate_ellipse_condition_l153_153513

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 - 6 * x + 18 * y = k) → k > -9 :=
by
  sorry

end non_degenerate_ellipse_condition_l153_153513


namespace trig_identity_example_l153_153486

theorem trig_identity_example:
  (Real.sin (63 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) + 
  Real.cos (63 * Real.pi / 180) * Real.cos (108 * Real.pi / 180)) = 
  Real.sqrt 2 / 2 := 
by 
  sorry

end trig_identity_example_l153_153486


namespace cost_of_single_room_l153_153839

theorem cost_of_single_room
  (total_rooms : ℕ)
  (double_rooms : ℕ)
  (cost_double_room : ℕ)
  (revenue_total : ℕ)
  (cost_single_room : ℕ)
  (H1 : total_rooms = 260)
  (H2 : double_rooms = 196)
  (H3 : cost_double_room = 60)
  (H4 : revenue_total = 14000)
  (H5 : revenue_total = (total_rooms - double_rooms) * cost_single_room + double_rooms * cost_double_room)
  : cost_single_room = 35 :=
sorry

end cost_of_single_room_l153_153839


namespace opposite_2024_eq_neg_2024_l153_153959

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l153_153959


namespace product_of_ab_l153_153244

theorem product_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 7) : a * b = -10 :=
by
  sorry

end product_of_ab_l153_153244


namespace fraction_habitable_surface_l153_153602

def fraction_exposed_land : ℚ := 3 / 8
def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_exposed_land * fraction_inhabitable_land = 1 / 4 := by
    -- proof steps omitted
    sorry

end fraction_habitable_surface_l153_153602


namespace find_integers_with_conditions_l153_153734

theorem find_integers_with_conditions :
  ∃ a b c d : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (1 ≤ c) ∧ (1 ≤ d) ∧ a * b * c * d = 2002 ∧ a + b + c + d < 40 := sorry

end find_integers_with_conditions_l153_153734


namespace vertex_of_parabola_minimum_value_for_x_ge_2_l153_153742

theorem vertex_of_parabola :
  ∀ x y : ℝ, y = x^2 + 2*x - 3 → ∃ (vx vy : ℝ), (vx = -1) ∧ (vy = -4) :=
by
  sorry

theorem minimum_value_for_x_ge_2 :
  ∀ x : ℝ, x ≥ 2 → y = x^2 + 2*x - 3 → ∃ (min_val : ℝ), min_val = 5 :=
by
  sorry

end vertex_of_parabola_minimum_value_for_x_ge_2_l153_153742


namespace proportion_of_ones_l153_153403

theorem proportion_of_ones (m n : ℕ) (h : Nat.gcd m n = 1) : 
  m + n = 275 :=
  sorry

end proportion_of_ones_l153_153403


namespace not_divisible_by_1980_divisible_by_1981_l153_153155

open Nat

theorem not_divisible_by_1980 (x : ℕ) : ¬ (2^100 * x - 1) % 1980 = 0 := by
sorry

theorem divisible_by_1981 : ∃ x : ℕ, (2^100 * x - 1) % 1981 = 0 := by
sorry

end not_divisible_by_1980_divisible_by_1981_l153_153155


namespace find_fraction_l153_153330

theorem find_fraction (x y : ℕ) (h₁ : x / (y + 1) = 1 / 2) (h₂ : (x + 1) / y = 1) : x = 2 ∧ y = 3 := by
  sorry

end find_fraction_l153_153330


namespace smallest_among_neg2_cube_neg3_square_neg_neg1_l153_153069

def smallest_among (a b c : ℤ) : ℤ :=
if a < b then
  if a < c then a else c
else
  if b < c then b else c

theorem smallest_among_neg2_cube_neg3_square_neg_neg1 :
  smallest_among ((-2)^3) (-(3^2)) (-(-1)) = -(3^2) :=
by
  sorry

end smallest_among_neg2_cube_neg3_square_neg_neg1_l153_153069


namespace gcd_390_455_546_l153_153447

theorem gcd_390_455_546 : Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
by
  sorry    -- this indicates the proof is not included

end gcd_390_455_546_l153_153447


namespace mean_of_four_integers_l153_153648

theorem mean_of_four_integers (x : ℝ) (h : (78 + 83 + 82 + x) / 4 = 80) : x = 77 ∧ x = 80 - 3 :=
by
  have h1 : 78 + 83 + 82 + x = 4 * 80 := by sorry
  have h2 : 78 + 83 + 82 = 243 := by sorry
  have h3 : 243 + x = 320 := by sorry
  have h4 : x = 320 - 243 := by sorry
  have h5 : x = 77 := by sorry
  have h6 : x = 80 - 3 := by sorry
  exact ⟨h5, h6⟩

end mean_of_four_integers_l153_153648


namespace geometric_sequence_sum_l153_153104

theorem geometric_sequence_sum (q a₁ : ℝ) (hq : q > 1) (h₁ : a₁ + a₁ * q^3 = 18) (h₂ : a₁^2 * q^3 = 32) :
  (a₁ * (1 - q^8) / (1 - q) = 510) :=
by
  sorry

end geometric_sequence_sum_l153_153104


namespace max_diff_consecutive_slightly_unlucky_l153_153416

def is_slightly_unlucky (n : ℕ) : Prop := (n.digits 10).sum % 13 = 0

theorem max_diff_consecutive_slightly_unlucky :
  ∃ n m : ℕ, is_slightly_unlucky n ∧ is_slightly_unlucky m ∧ (m > n) ∧ ∀ k, (is_slightly_unlucky k ∧ k > n ∧ k < m) → false → (m - n) = 79 :=
sorry

end max_diff_consecutive_slightly_unlucky_l153_153416


namespace system_has_infinite_solutions_l153_153517

theorem system_has_infinite_solutions :
  ∀ (x y : ℝ), (3 * x - 4 * y = 5) ↔ (6 * x - 8 * y = 10) ∧ (9 * x - 12 * y = 15) :=
by
  sorry

end system_has_infinite_solutions_l153_153517


namespace horner_method_multiplications_additions_count_l153_153350

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - 2 * x^2 + 4 * x - 6

-- Define the property we want to prove
theorem horner_method_multiplications_additions_count : 
  ∃ (multiplications additions : ℕ), multiplications = 4 ∧ additions = 4 := 
by
  sorry

end horner_method_multiplications_additions_count_l153_153350


namespace mark_bread_baking_time_l153_153418

/--
Mark is baking bread. 
He has to let it rise for 120 minutes twice. 
He also needs to spend 10 minutes kneading it and 30 minutes baking it. 
Prove that the total time Mark takes to finish making the bread is 280 minutes.
-/
theorem mark_bread_baking_time :
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  rising_time + kneading_time + baking_time = 280 := 
by
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  have rising_time_eq : rising_time = 240 := rfl
  have kneading_time_eq : kneading_time = 10 := rfl
  have baking_time_eq : baking_time = 30 := rfl
  calc
    rising_time + kneading_time + baking_time
        = 240 + 10 + 30 : by rw [rising_time_eq, kneading_time_eq, baking_time_eq]
    ... = 280 : by norm_num

end mark_bread_baking_time_l153_153418


namespace score_recording_l153_153121

theorem score_recording (avg : ℤ) (h : avg = 0) : 
  (9 = avg + 9) ∧ (-18 = avg - 18) ∧ (-2 = avg - 2) :=
by
  -- Proof steps go here
  sorry

end score_recording_l153_153121


namespace megatek_manufacturing_percentage_l153_153435

theorem megatek_manufacturing_percentage :
  ∀ (total_degrees manufacturing_degrees total_percentage : ℝ),
  total_degrees = 360 → manufacturing_degrees = 216 → total_percentage = 100 →
  (manufacturing_degrees / total_degrees) * total_percentage = 60 :=
by
  intros total_degrees manufacturing_degrees total_percentage H1 H2 H3
  rw [H1, H2, H3]
  sorry

end megatek_manufacturing_percentage_l153_153435


namespace cube_root_of_neg_eight_eq_neg_two_l153_153442

theorem cube_root_of_neg_eight_eq_neg_two : real.cbrt (-8) = -2 :=
by
  sorry

end cube_root_of_neg_eight_eq_neg_two_l153_153442


namespace kiwi_lemon_relationship_l153_153290

open Nat

-- Define the conditions
def total_fruits : ℕ := 58
def mangoes : ℕ := 18
def pears : ℕ := 10
def pawpaws : ℕ := 12
def lemons_in_last_two_baskets : ℕ := 9

-- Define the question and the proof goal
theorem kiwi_lemon_relationship :
  ∃ (kiwis lemons : ℕ), 
  kiwis = lemons_in_last_two_baskets ∧ 
  lemons = lemons_in_last_two_baskets ∧ 
  kiwis + lemons = total_fruits - (mangoes + pears + pawpaws) :=
sorry

end kiwi_lemon_relationship_l153_153290


namespace cuboid_inequality_l153_153152

theorem cuboid_inequality 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end cuboid_inequality_l153_153152


namespace problem_solution_l153_153588

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l153_153588


namespace factorial_division_l153_153206

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l153_153206


namespace vertical_distance_l153_153144

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l153_153144


namespace blue_socks_count_l153_153767

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l153_153767


namespace age_ratio_l153_153352

variable (Cindy Jan Marcia Greg: ℕ)

theorem age_ratio 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3: Greg = 16)
  (h4 : Greg = Marcia + 2)
  (h5 : ∃ k : ℕ, Marcia = k * Jan) 
  : Marcia / Jan = 2 := 
    sorry

end age_ratio_l153_153352


namespace polygon_sides_l153_153902

theorem polygon_sides (n : ℕ) 
  (h : 3240 = 180 * (n - 2) - (360)) : n = 22 := 
by 
  sorry

end polygon_sides_l153_153902


namespace percentage_of_students_enrolled_is_40_l153_153596

def total_students : ℕ := 880
def not_enrolled_in_biology : ℕ := 528
def enrolled_in_biology : ℕ := total_students - not_enrolled_in_biology
def percentage_enrolled : ℕ := (enrolled_in_biology * 100) / total_students

theorem percentage_of_students_enrolled_is_40 : percentage_enrolled = 40 := by
  -- Beginning of the proof
  sorry

end percentage_of_students_enrolled_is_40_l153_153596


namespace local_max_2_l153_153775

noncomputable def f (x m n : ℝ) := 2 * Real.log x - (1 / 2) * m * x^2 - n * x

theorem local_max_2 (m n : ℝ) (h : n = 1 - 2 * m) :
  ∃ m : ℝ, -1/2 < m ∧ (∀ x : ℝ, x > 0 → (∃ U : Set ℝ, IsOpen U ∧ (2 ∈ U) ∧ (∀ y ∈ U, f y m n ≤ f 2 m n))) :=
sorry

end local_max_2_l153_153775


namespace treasure_chest_coins_l153_153196

theorem treasure_chest_coins :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 5) ∧ (n ≥ 0) ∧
  (∀ m : ℕ, (m % 8 = 6) ∧ (m % 9 = 5) → m ≥ 0 → n ≤ m) ∧
  (∃ r : ℕ, n = 11 * (n / 11) + r ∧ r = 3) :=
by
  sorry

end treasure_chest_coins_l153_153196


namespace minimum_p_plus_q_l153_153381

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem minimum_p_plus_q (p q : ℝ) (hpq : p ≠ q) (hf : f p + f q = 2) :
  p + q = 3 - 2 * Real.log 2 := by
  sorry

end minimum_p_plus_q_l153_153381


namespace intersections_of_absolute_value_functions_l153_153853

theorem intersections_of_absolute_value_functions : 
  (∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|4 * x + 3|) → ∃ (x y : ℝ), (x = -1 ∧ y = 1) ∧ ¬(∃ (x' y' : ℝ), y' = |3 * x' + 4| ∧ y' = -|4 * x' + 3| ∧ (x' ≠ -1 ∨ y' ≠ 1)) :=
by
  sorry

end intersections_of_absolute_value_functions_l153_153853


namespace find_remainder_division_l153_153422

/--
Given:
1. A dividend of 100.
2. A quotient of 9.
3. A divisor of 11.

Prove: The remainder \( r \) when dividing 100 by 11 is 1.
-/
theorem find_remainder_division :
  ∀ (q d r : Nat), q = 9 → d = 11 → 100 = (d * q + r) → r = 1 :=
by
  intros q d r hq hd hdiv
  -- Proof steps would go here
  sorry

end find_remainder_division_l153_153422


namespace shyam_weight_increase_l153_153653

theorem shyam_weight_increase (x : ℝ) 
    (h1 : x > 0)
    (ratio : ∀ Ram Shyam : ℝ, (Ram / Shyam) = 7 / 5)
    (ram_increase : ∀ Ram : ℝ, Ram' = Ram + 0.1 * Ram)
    (total_weight_after : Ram' + Shyam' = 82.8)
    (total_weight_increase : 82.8 = 1.15 * total_weight) :
    (Shyam' - Shyam) / Shyam * 100 = 22 :=
by
  sorry

end shyam_weight_increase_l153_153653


namespace number_of_female_officers_l153_153423

theorem number_of_female_officers (total_on_duty : ℕ) (female_on_duty : ℕ) (percentage_on_duty : ℚ) : 
  total_on_duty = 500 → 
  female_on_duty = 250 → 
  percentage_on_duty = 1/4 → 
  (female_on_duty : ℚ) = percentage_on_duty * (total_on_duty / 2 : ℚ) →
  (total_on_duty : ℚ) = 4 * female_on_duty →
  total_on_duty = 1000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_female_officers_l153_153423


namespace eq_root_condition_l153_153884

theorem eq_root_condition (k : ℝ) 
    (h_discriminant : -4 * k + 5 ≥ 0)
    (h_roots : ∃ x1 x2 : ℝ, 
        (x1 + x2 = 1 - 2 * k) ∧ 
        (x1 * x2 = k^2 - 1) ∧ 
        (x1^2 + x2^2 = 16 + x1 * x2)) :
    k = -2 :=
sorry

end eq_root_condition_l153_153884


namespace sell_decision_l153_153496

noncomputable def profit_beginning (a : ℝ) : ℝ :=
(a + 100) * 1.024

noncomputable def profit_end (a : ℝ) : ℝ :=
a + 115

theorem sell_decision (a : ℝ) :
  (a > 525 → profit_beginning a > profit_end a) ∧
  (a < 525 → profit_beginning a < profit_end a) ∧
  (a = 525 → profit_beginning a = profit_end a) :=
by
  sorry

end sell_decision_l153_153496


namespace numbers_combination_to_24_l153_153808

theorem numbers_combination_to_24 :
  (40 / 4) + 12 + 2 = 24 :=
by
  sorry

end numbers_combination_to_24_l153_153808


namespace binary_multiplication_l153_153532

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l153_153532


namespace grid_coloring_count_l153_153564

/-- Let n be a positive integer with n ≥ 2. Each of the 2n vertices in a 2 × n grid need to be 
colored red (R), yellow (Y), or blue (B). The three vertices at the endpoints are already colored 
as shown in the problem description. For the remaining 2n-3 vertices, each vertex must be colored 
exactly one color, and adjacent vertices must be colored differently. We aim to show that the 
number of distinct ways to color the vertices is 3^(n-1). -/
theorem grid_coloring_count (n : ℕ) (hn : n ≥ 2) : 
  ∃ a_n b_n c_n : ℕ, 
    (a_n + b_n + c_n = 3^(n-1)) ∧ 
    (a_n = b_n) ∧ 
    (a_n = 2 * b_n + c_n) := 
by 
  sorry

end grid_coloring_count_l153_153564


namespace painting_time_l153_153494

-- Definitions translated from conditions
def total_weight_tons := 5
def weight_per_ball_kg := 4
def number_of_students := 10
def balls_per_student_per_6_minutes := 5

-- Derived Definitions
def total_weight_kg := total_weight_tons * 1000
def total_balls := total_weight_kg / weight_per_ball_kg
def balls_painted_by_all_students_per_6_minutes := number_of_students * balls_per_student_per_6_minutes
def required_intervals := total_balls / balls_painted_by_all_students_per_6_minutes
def total_time_minutes := required_intervals * 6

-- The theorem statement
theorem painting_time : total_time_minutes = 150 := by
  sorry

end painting_time_l153_153494


namespace cost_of_10_pound_bag_is_correct_l153_153060

noncomputable def cost_of_5_pound_bag : ℝ := 13.80
noncomputable def cost_of_25_pound_bag : ℝ := 32.25
noncomputable def min_pounds_needed : ℝ := 65
noncomputable def max_pounds_allowed : ℝ := 80
noncomputable def least_possible_cost : ℝ := 98.73

def min_cost_10_pound_bag : ℝ := 1.98

theorem cost_of_10_pound_bag_is_correct :
  ∀ (x : ℝ), (x >= min_pounds_needed / cost_of_25_pound_bag ∧ x <= max_pounds_allowed / cost_of_5_pound_bag ∧ least_possible_cost = (3 * cost_of_25_pound_bag + x)) → x = min_cost_10_pound_bag :=
by
  sorry

end cost_of_10_pound_bag_is_correct_l153_153060


namespace fair_dice_roll_six_times_four_not_necessarily_appear_l153_153396

-- Define a fair dice
def fair_dice : Ω := {1, 2, 3, 4, 5, 6}
-- Define the event that the number 4 appears when a fair dice is rolled.
def event_four (ω : Ω) := ω = 4

-- Define the random variable for rolling a fair dice 6 times
def roll_six_times : list Ω := replicate 6 (count fair_dice)

-- Statement to prove that the number 4 does not necessarily appear
theorem fair_dice_roll_six_times_four_not_necessarily_appear :
  ¬ ∀ (ωlist : list Ω), ωlist ∈ roll_six_times → ∃ ω ∈ ωlist, event_four ω :=
sorry

end fair_dice_roll_six_times_four_not_necessarily_appear_l153_153396


namespace evaluate_expression_l153_153360

theorem evaluate_expression (a : ℕ) (h : a = 3) : a^2 * a^5 = 2187 :=
by sorry

end evaluate_expression_l153_153360


namespace rotate_parabola_180_l153_153811

theorem rotate_parabola_180 (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 2) → 
  (∃ x' y', x' = -x ∧ y' = -y ∧ y' = -2 * (x' + 1)^2 - 2) := 
sorry

end rotate_parabola_180_l153_153811


namespace polynomial_has_roots_l153_153089

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7

-- Theorem to prove the existence of roots for the polynomial equation
theorem polynomial_has_roots : ∃ x : ℂ, polynomial x = 0 := by
  sorry

end polynomial_has_roots_l153_153089


namespace scientific_notation_of_million_l153_153685

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end scientific_notation_of_million_l153_153685


namespace max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l153_153772

variable {m x x0 : ℝ}

def proposition_p (m : ℝ) : Prop := ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m
def proposition_q (m : ℝ) : Prop := ∃ x0 : ℝ, x0 ^ 2 - m * x0 + 1 = 0

theorem max_val_of_m_if_p_true (h : proposition_p m) : m ≤ Real.sqrt 2 := by
  sorry

theorem range_of_m_if_one_prop_true_one_false (hp : proposition_p m) (hq : ¬ proposition_q m) : (-2 < m ∧ m ≤ Real.sqrt 2) ∨ (2 ≤ m) := by
  sorry

theorem range_of_m_if_one_prop_false_one_true (hp : ¬ proposition_p m) (hq : proposition_q m) : (m ≥ 2) := by
  sorry

end max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l153_153772


namespace probability_two_correct_positions_sum_mn_l153_153024

noncomputable def permutations (l : List ℕ) : List (List ℕ) := 
List.permutations l

theorem probability_two_correct_positions_sum_mn (d1 d2 d3 d4 : ℕ)
  (h : d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4) :
  ∃ (m n : ℕ), (m + n = 5) :=
by
  let digits := [d1, d2, d3, d4]
  let perms := permutations digits
  have h_total_perms : perms.length = 24 := by sorry
  have h_two_correct : (∃ p ∈ perms, ∑ i, if p[i] = digits[i] then 1 else 0 = 2) := by sorry
  have h_probability : (6 : ℚ) / 24 = 1 / 4 := by norm_num
  have h_m_n := h_probability -- given that m = 1 and n = 4
  existsi 1, 4
  trivial

end probability_two_correct_positions_sum_mn_l153_153024


namespace total_peaches_l153_153071

theorem total_peaches (initial_peaches_Audrey : ℕ) (multiplier_Audrey : ℕ)
                      (initial_peaches_Paul : ℕ) (multiplier_Paul : ℕ)
                      (initial_peaches_Maya : ℕ) (additional_peaches_Maya : ℕ) :
                      initial_peaches_Audrey = 26 →
                      multiplier_Audrey = 3 →
                      initial_peaches_Paul = 48 →
                      multiplier_Paul = 2 →
                      initial_peaches_Maya = 57 →
                      additional_peaches_Maya = 20 →
                      (initial_peaches_Audrey + multiplier_Audrey * initial_peaches_Audrey) +
                      (initial_peaches_Paul + multiplier_Paul * initial_peaches_Paul) +
                      (initial_peaches_Maya + additional_peaches_Maya) = 325 :=
by
  sorry

end total_peaches_l153_153071


namespace Tim_scores_expected_value_l153_153162

theorem Tim_scores_expected_value :
  let LAIMO := 15
  let FARML := 10
  let DOMO := 50
  let p := 1 / 3
  let expected_LAIMO := LAIMO * p
  let expected_FARML := FARML * p
  let expected_DOMO := DOMO * p
  expected_LAIMO + expected_FARML + expected_DOMO = 25 :=
by
  -- The Lean proof would go here
  sorry

end Tim_scores_expected_value_l153_153162


namespace value_of_a_l153_153229

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l153_153229


namespace minimize_time_theta_l153_153837

theorem minimize_time_theta (α θ : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : θ = α / 2) : 
  θ = α / 2 :=
by
  sorry

end minimize_time_theta_l153_153837


namespace butterfly_1023_distance_l153_153182

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def Q (n : ℕ) : Complex :=
  match n with
  | 0     => 0
  | k + 1 => Q k + (k + 1) * omega ^ k

noncomputable def butterfly_distance (n : ℕ) : ℝ := Complex.abs (Q n)

theorem butterfly_1023_distance : butterfly_distance 1023 = 511 * Real.sqrt (2 + Real.sqrt 2) :=
  sorry

end butterfly_1023_distance_l153_153182


namespace second_discount_percentage_l153_153681

-- Define the original price as P
variables {P : ℝ} (hP : P > 0)

-- Define the price increase by 34%
def price_after_increase (P : ℝ) := 1.34 * P

-- Define the first discount of 10%
def price_after_first_discount (P : ℝ) := 0.90 * (price_after_increase P)

-- Define the second discount percentage as D (in decimal form)
variables {D : ℝ}

-- Define the price after the second discount
def price_after_second_discount (P D : ℝ) := (1 - D) * (price_after_first_discount P)

-- Define the overall percentage gain of 2.51%
def final_price (P : ℝ) := 1.0251 * P

-- The main theorem to prove
theorem second_discount_percentage (hP : P > 0) (hD : 0 ≤ D ∧ D ≤ 1) :
  price_after_second_discount P D = final_price P ↔ D = 0.1495 :=
by
  sorry

end second_discount_percentage_l153_153681


namespace initial_eggs_proof_l153_153654

-- Definitions based on the conditions provided
def initial_eggs := 7
def added_eggs := 4
def total_eggs := 11

-- The statement to be proved
theorem initial_eggs_proof : initial_eggs + added_eggs = total_eggs :=
by
  -- Placeholder for proof
  sorry

end initial_eggs_proof_l153_153654


namespace angle_quadrant_l153_153783

theorem angle_quadrant (theta : ℤ) (h_theta : theta = -3290) : 
  ∃ q : ℕ, q = 4 := 
by 
  sorry

end angle_quadrant_l153_153783


namespace max_integer_is_110003_l153_153669

def greatest_integer : Prop :=
  let a := 100004
  let b := 110003
  let c := 102002
  let d := 100301
  let e := 100041
  b > a ∧ b > c ∧ b > d ∧ b > e

theorem max_integer_is_110003 : greatest_integer :=
by
  sorry

end max_integer_is_110003_l153_153669


namespace geese_left_in_the_field_l153_153130

theorem geese_left_in_the_field 
  (initial_geese : ℕ) 
  (geese_flew_away : ℕ) 
  (geese_joined : ℕ)
  (h1 : initial_geese = 372)
  (h2 : geese_flew_away = 178)
  (h3 : geese_joined = 57) :
  initial_geese - geese_flew_away + geese_joined = 251 := by
  sorry

end geese_left_in_the_field_l153_153130


namespace min_value_l153_153248

theorem min_value : ∀ (a b : ℝ), a + b^2 = 2 → (∀ x y : ℝ, x = a^2 + 6 * y^2 → y = b) → (∃ c : ℝ, c = 3) :=
by
  intros a b h₁ h₂
  sorry

end min_value_l153_153248


namespace B_gain_l153_153190

-- Problem statement and conditions
def principalA : ℝ := 3500
def rateA : ℝ := 0.10
def periodA : ℕ := 2
def principalB : ℝ := 3500
def rateB : ℝ := 0.14
def periodB : ℕ := 3

-- Calculate amount A will receive from B after 2 years
noncomputable def amountA := principalA * (1 + rateA / 1) ^ periodA

-- Calculate amount B will receive from C after 3 years
noncomputable def amountB := principalB * (1 + rateB / 2) ^ (2 * periodB)

-- Calculate B's gain
noncomputable def gainB := amountB - amountA

-- The theorem to prove
theorem B_gain : gainB = 1019.20 := by
  sorry

end B_gain_l153_153190


namespace exists_y_less_than_half_p_l153_153936

theorem exists_y_less_than_half_p (p : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) :
  ∃ (y : ℕ), y < p / 2 ∧ ∀ (a b : ℕ), p * y + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by sorry

end exists_y_less_than_half_p_l153_153936


namespace smallest_k_l153_153559

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l153_153559


namespace smallest_intersection_value_l153_153953

theorem smallest_intersection_value (a b : ℝ) (f g : ℝ → ℝ)
    (Hf : ∀ x, f x = x^4 - 6 * x^3 + 11 * x^2 - 6 * x + a)
    (Hg : ∀ x, g x = x + b)
    (Hinter : ∀ x, f x = g x → true):
  ∃ x₀, x₀ = 0 :=
by
  intros
  -- Further steps would involve proving roots and conditions stated but omitted here.
  sorry

end smallest_intersection_value_l153_153953


namespace tom_found_dimes_l153_153461

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes_l153_153461


namespace crackers_shared_equally_l153_153279

theorem crackers_shared_equally : ∀ (matthew_crackers friends_crackers left_crackers friends : ℕ),
  matthew_crackers = 23 →
  left_crackers = 11 →
  friends = 2 →
  matthew_crackers - left_crackers = friends_crackers →
  friends_crackers = friends * 6 :=
by
  intro matthew_crackers friends_crackers left_crackers friends
  sorry

end crackers_shared_equally_l153_153279


namespace least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l153_153736

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem least_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem maximum_value_of_f :
  ∃ x, f x = 3 :=
sorry

theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ ∀ x, a < x ∧ x < b → ∀ x', a ≤ x' ∧ x' ≤ x → f x' < f x :=
sorry

end least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l153_153736


namespace basketball_volleyball_problem_l153_153186

-- Define variables and conditions
variables (x y : ℕ) (m : ℕ)

-- Conditions
def price_conditions : Prop :=
  2 * x + 3 * y = 190 ∧ 3 * x = 5 * y

def price_solutions : Prop :=
  x = 50 ∧ y = 30

def purchase_conditions : Prop :=
  8 ≤ m ∧ m ≤ 10 ∧ 50 * m + 30 * (20 - m) ≤ 800

-- The most cost-effective plan
def cost_efficient_plan : Prop :=
  m = 8 ∧ (20 - m) = 12

-- Conjecture for the problem
theorem basketball_volleyball_problem :
  price_conditions x y ∧ purchase_conditions m →
  price_solutions x y ∧ cost_efficient_plan m :=
by {
  sorry
}

end basketball_volleyball_problem_l153_153186


namespace find_b_for_real_root_l153_153216

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 + b * x^3 - 2 * x^2 + b * x + 2 = 0

theorem find_b_for_real_root :
  ∀ b : ℝ, polynomial_has_real_root b → b ≤ 0 := by
  sorry

end find_b_for_real_root_l153_153216


namespace find_abc_l153_153410

theorem find_abc :
  ∃ a b c : ℝ, (∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - a) * (x - b) / (x - c) ≤ 0)) ∧ a < b ∧ a + 2 * b + 3 * c = 74 :=
by
  sorry

end find_abc_l153_153410


namespace angle_sum_around_point_l153_153045

theorem angle_sum_around_point (y : ℝ) (h1 : 150 + y + y = 360) : y = 105 :=
by sorry

end angle_sum_around_point_l153_153045


namespace angle_acb_after_rotations_is_30_l153_153789

noncomputable def initial_angle : ℝ := 60
noncomputable def rotation_clockwise_540 : ℝ := -540
noncomputable def rotation_counterclockwise_90 : ℝ := 90
noncomputable def final_angle : ℝ := 30

theorem angle_acb_after_rotations_is_30 
  (initial_angle : ℝ)
  (rotation_clockwise_540 : ℝ)
  (rotation_counterclockwise_90 : ℝ) :
  final_angle = 30 :=
sorry

end angle_acb_after_rotations_is_30_l153_153789


namespace min_odd_integers_l153_153463

theorem min_odd_integers 
  (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 15)
  (h3 : e + f = 17)
  (h4 : c + d + e + f = 32) :
  ∃ n : ℕ, (n = 2) ∧ (∃ odd_count, 
  odd_count = (if (a % 2 = 0) then 0 else 1) + 
                     (if (b % 2 = 0) then 0 else 1) + 
                     (if (c % 2 = 0) then 0 else 1) + 
                     (if (d % 2 = 0) then 0 else 1) + 
                     (if (e % 2 = 0) then 0 else 1) + 
                     (if (f % 2 = 0) then 0 else 1) ∧
  odd_count = 2) := sorry

end min_odd_integers_l153_153463


namespace find_x3_plus_y3_l153_153025

theorem find_x3_plus_y3 (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) : x^3 + y^3 = 2005 :=
sorry

end find_x3_plus_y3_l153_153025


namespace largest_sum_is_sum3_l153_153847

-- Definitions of the individual sums given in the conditions
def sum1 : ℚ := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
def sum2 : ℚ := (1/4 : ℚ) - (1/6 : ℚ)
def sum3 : ℚ := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
def sum4 : ℚ := (1/4 : ℚ) - (1/8 : ℚ)
def sum5 : ℚ := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)

-- Theorem to prove that sum3 is the largest
theorem largest_sum_is_sum3 : sum3 = (5/12 : ℚ) ∧ sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := 
by 
  -- The proof would go here
  sorry

end largest_sum_is_sum3_l153_153847


namespace smallest_k_l153_153542

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l153_153542


namespace min_surface_area_of_sphere_l153_153373

theorem min_surface_area_of_sphere (a b c : ℝ) (volume : ℝ) (height : ℝ) 
  (h_volume : a * b * c = volume) (h_height : c = height) 
  (volume_val : volume = 12) (height_val : height = 4) : 
  ∃ r : ℝ, 4 * π * r^2 = 22 * π := 
by
  sorry

end min_surface_area_of_sphere_l153_153373


namespace min_coach_handshakes_l153_153343

-- Definitions based on the problem conditions
def total_gymnasts : ℕ := 26
def total_handshakes : ℕ := 325

/- 
  The main theorem stating that the fewest number of handshakes 
  the coaches could have participated in is 0.
-/
theorem min_coach_handshakes (n : ℕ) (h : 0 ≤ n ∧ n * (n - 1) / 2 = total_handshakes) : 
  n = total_gymnasts → (total_handshakes - n * (n - 1) / 2) = 0 :=
by 
  intros h_n_eq_26
  sorry

end min_coach_handshakes_l153_153343


namespace original_plan_trees_average_l153_153913

-- Definitions based on conditions
def original_trees_per_day (x : ℕ) := x
def increased_trees_per_day (x : ℕ) := x + 5
def time_to_plant_60_trees (x : ℕ) := 60 / (x + 5)
def time_to_plant_45_trees (x : ℕ) := 45 / x

-- The main theorem we need to prove
theorem original_plan_trees_average : ∃ x : ℕ, time_to_plant_60_trees x = time_to_plant_45_trees x ∧ x = 15 :=
by
  -- Placeholder for the proof
  sorry

end original_plan_trees_average_l153_153913


namespace eddy_time_to_B_l153_153078

-- Definitions
def distance_A_to_B : ℝ := 570
def distance_A_to_C : ℝ := 300
def time_C : ℝ := 4
def speed_ratio : ℝ := 2.5333333333333333

-- Theorem Statement
theorem eddy_time_to_B : 
  (distance_A_to_B / (distance_A_to_C / time_C * speed_ratio)) = 3 := 
by
  sorry

end eddy_time_to_B_l153_153078


namespace trapezium_side_length_l153_153869

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l153_153869


namespace mode_and_median_of_survey_l153_153036

/-- A data structure representing the number of students corresponding to each sleep time. -/
structure SleepSurvey :=
  (time7 : ℕ)
  (time8 : ℕ)
  (time9 : ℕ)
  (time10 : ℕ)

def survey : SleepSurvey := { time7 := 6, time8 := 9, time9 := 11, time10 := 4 }

theorem mode_and_median_of_survey (s : SleepSurvey) :
  (mode=9 ∧ median = 8.5) :=
by
  -- proof would go here
  sorry

end mode_and_median_of_survey_l153_153036


namespace range_of_a_l153_153414

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a_l153_153414


namespace number_of_zeros_of_F_l153_153567

-- Define the function f(x) = ln(x)
def f (x : ℝ) : ℝ := Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 1 / x

-- Define the function F(x) = f(x) - f'(x)
def F (x : ℝ) : ℝ := f x - f' x

-- State the theorem to find the number of zeros of F(x)
theorem number_of_zeros_of_F : ∃! x > 0, F x = 0 := sorry

end number_of_zeros_of_F_l153_153567


namespace sum_of_interior_angles_l153_153650

theorem sum_of_interior_angles (h_triangle : ∀ (a b c : ℝ), a + b + c = 180)
    (h_quadrilateral : ∀ (a b c d : ℝ), a + b + c + d = 360) :
  (∀ (n : ℕ), n ≥ 3 → ∀ (angles : Fin n → ℝ), (Finset.univ.sum angles) = (n-2) * 180) :=
by
  intro n h_n angles
  sorry

end sum_of_interior_angles_l153_153650


namespace probability_two_cards_l153_153040

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l153_153040


namespace no_n_nat_powers_l153_153425

theorem no_n_nat_powers (n : ℕ) : ∀ n : ℕ, ¬∃ m k : ℕ, k ≥ 2 ∧ n * (n + 1) = m ^ k := 
by 
  sorry

end no_n_nat_powers_l153_153425


namespace binary_111_eq_7_l153_153440

theorem binary_111_eq_7 : (1 * 2^0 + 1 * 2^1 + 1 * 2^2) = 7 :=
by
  sorry

end binary_111_eq_7_l153_153440


namespace boxes_with_nothing_l153_153939

theorem boxes_with_nothing (h_total : 15 = total_boxes)
    (h_pencils : 9 = pencil_boxes)
    (h_pens : 5 = pen_boxes)
    (h_both_pens_and_pencils : 3 = both_pen_and_pencil_boxes)
    (h_markers : 4 = marker_boxes)
    (h_both_markers_and_pencils : 2 = both_marker_and_pencil_boxes)
    (h_no_markers_and_pens : no_marker_and_pen_boxes = 0)
    (h_no_all_three_items : no_all_three_items = 0) :
    ∃ (neither_boxes : ℕ), neither_boxes = 2 :=
by
  sorry

end boxes_with_nothing_l153_153939


namespace infinite_geometric_series_sum_l153_153520

theorem infinite_geometric_series_sum (a r S : ℚ) (ha : a = 1 / 4) (hr : r = 1 / 3) :
  (S = a / (1 - r)) → (S = 3 / 8) :=
by
  sorry

end infinite_geometric_series_sum_l153_153520


namespace package_weights_l153_153139

theorem package_weights (a b c : ℕ) 
  (h1 : a + b = 108) 
  (h2 : b + c = 132) 
  (h3 : c + a = 138) 
  (h4 : a ≥ 40) 
  (h5 : b ≥ 40) 
  (h6 : c ≥ 40) : 
  a + b + c = 189 :=
sorry

end package_weights_l153_153139


namespace michelle_drives_294_miles_l153_153805

theorem michelle_drives_294_miles
  (total_distance : ℕ)
  (michelle_drives : ℕ)
  (katie_drives : ℕ)
  (tracy_drives : ℕ)
  (h1 : total_distance = 1000)
  (h2 : michelle_drives = 3 * katie_drives)
  (h3 : tracy_drives = 2 * michelle_drives + 20)
  (h4 : katie_drives + michelle_drives + tracy_drives = total_distance) :
  michelle_drives = 294 := by
  sorry

end michelle_drives_294_miles_l153_153805


namespace divides_sequence_l153_153798

theorem divides_sequence (a : ℕ → ℕ) (n k: ℕ) (h0 : a 0 = 0) (h1 : a 1 = 1) 
  (hrec : ∀ m, a (m + 2) = 2 * a (m + 1) + a m) :
  (2^k ∣ a n) ↔ (2^k ∣ n) :=
sorry

end divides_sequence_l153_153798


namespace pool_depth_is_10_feet_l153_153434

-- Definitions based on conditions
def hoseRate := 60 -- cubic feet per minute
def poolWidth := 80 -- feet
def poolLength := 150 -- feet
def drainingTime := 2000 -- minutes

-- Proof goal: the depth of the pool is 10 feet
theorem pool_depth_is_10_feet :
  ∃ (depth : ℝ), depth = 10 ∧ (hoseRate * drainingTime) = (poolWidth * poolLength * depth) :=
by
  use 10
  sorry

end pool_depth_is_10_feet_l153_153434


namespace trapezium_parallel_side_length_l153_153863

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l153_153863


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l153_153204

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l153_153204


namespace bob_speed_before_construction_l153_153842

theorem bob_speed_before_construction:
  ∀ (v : ℝ),
    (1.5 * v + 2 * 45 = 180) →
    v = 60 :=
by
  intros v h
  sorry

end bob_speed_before_construction_l153_153842


namespace max_value_y_l153_153578

open Real

theorem max_value_y (x : ℝ) (h : -1 < x ∧ x < 1) : 
  ∃ y_max, y_max = 0 ∧ ∀ y, y = x / (x - 1) + x → y ≤ y_max :=
by
  have y : ℝ := x / (x - 1) + x
  use 0
  sorry

end max_value_y_l153_153578


namespace remainder_of_division_l153_153443

theorem remainder_of_division (L S R : ℕ) (h1 : L - S = 1365) (h2 : L = 1637) (h3 : L = 6 * S + R) : R = 5 :=
by
  sorry

end remainder_of_division_l153_153443


namespace jacks_walking_rate_l153_153985

theorem jacks_walking_rate :
  let distance := 8
  let time_in_minutes := 1 * 60 + 15
  let time := time_in_minutes / 60.0
  let rate := distance / time
  rate = 6.4 :=
by
  sorry

end jacks_walking_rate_l153_153985


namespace f_at_3_l153_153268

noncomputable def f : ℝ → ℝ := sorry

lemma periodic (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 4) = f x := sorry

lemma odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) + f x = 0 := sorry

lemma given_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = (x - 1)^2 := sorry

theorem f_at_3 : f 3 = 0 := 
by
  sorry

end f_at_3_l153_153268


namespace quadratic_inequality_no_solution_l153_153382

theorem quadratic_inequality_no_solution (a b c : ℝ) (h : a ≠ 0)
  (hnsol : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≥ 0)) :
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
sorry

end quadratic_inequality_no_solution_l153_153382


namespace sin_cos_15_eq_quarter_l153_153053

theorem sin_cos_15_eq_quarter :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
by 
  sorry

end sin_cos_15_eq_quarter_l153_153053


namespace area_of_plot_l153_153297

def cm_to_miles (a : ℕ) : ℕ := a * 9

def miles_to_acres (b : ℕ) : ℕ := b * 640

theorem area_of_plot :
  let bottom := 12
  let top := 18
  let height := 10
  let area_cm2 := ((bottom + top) * height) / 2
  let area_miles2 := cm_to_miles area_cm2
  let area_acres := miles_to_acres area_miles2
  area_acres = 864000 :=
by
  sorry

end area_of_plot_l153_153297


namespace travel_allowance_increase_20_l153_153625

def employees_total : ℕ := 480
def employees_no_increase : ℕ := 336
def employees_salary_increase_percentage : ℕ := 10

def employees_salary_increase : ℕ :=
(employees_salary_increase_percentage * employees_total) / 100

def employees_travel_allowance_increase : ℕ :=
employees_total - (employees_salary_increase + employees_no_increase)

def travel_allowance_increase_percentage : ℕ :=
(employees_travel_allowance_increase * 100) / employees_total

theorem travel_allowance_increase_20 :
  travel_allowance_increase_percentage = 20 :=
by sorry

end travel_allowance_increase_20_l153_153625


namespace power_of_a_l153_153393

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l153_153393


namespace action_figures_per_shelf_l153_153502

theorem action_figures_per_shelf (total_figures shelves : ℕ) (h1 : total_figures = 27) (h2 : shelves = 3) :
  (total_figures / shelves = 9) :=
by
  sorry

end action_figures_per_shelf_l153_153502


namespace ratio_rounded_to_nearest_tenth_l153_153856

theorem ratio_rounded_to_nearest_tenth : 
  (Float.round (11 / 16 : Float) * 10) / 10 = 0.7 :=
by
  -- sorry is used because the proof steps are not required in this task.
  sorry

end ratio_rounded_to_nearest_tenth_l153_153856


namespace pennies_for_washing_clothes_l153_153287

theorem pennies_for_washing_clothes (total_money_cents : ℕ) (num_quarters : ℕ) (value_quarter_cents : ℕ) :
  total_money_cents = 184 → num_quarters = 7 → value_quarter_cents = 25 → (total_money_cents - num_quarters * value_quarter_cents) = 9 :=
by
  intros htm hq hvq
  rw [htm, hq, hvq]
  linarith

end pennies_for_washing_clothes_l153_153287


namespace john_needs_2_sets_l153_153367

-- Definition of the conditions
def num_bars_per_set : ℕ := 7
def total_bars : ℕ := 14

-- The corresponding proof problem statement
theorem john_needs_2_sets : total_bars / num_bars_per_set = 2 :=
by
  sorry

end john_needs_2_sets_l153_153367


namespace inequality_l153_153009

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem inequality : c < a ∧ a < b := 
by 
  sorry

end inequality_l153_153009


namespace conditional_probability_calc_l153_153046

-- Definitions of the events M and N
def eventM (red_die : ℕ) : Prop :=
  red_die = 3 ∨ red_die = 6

def eventN (red_die blue_die : ℕ) : Prop :=
  red_die + blue_die > 8

-- Probability of M and MN
def P_M : ℚ := 12 / 36
def P_MN : ℚ := 5 / 36

-- Conditional Probability
def P_N_given_M := P_MN / P_M

theorem conditional_probability_calc :
  P_N_given_M = 5 / 12 :=
by
  unfold P_N_given_M P_M P_MN
  sorry

end conditional_probability_calc_l153_153046


namespace geralds_average_speed_l153_153150

theorem geralds_average_speed (poly_circuits : ℕ) (poly_time : ℝ) (track_length : ℝ) (gerald_speed_ratio : ℝ) :
  poly_circuits = 12 →
  poly_time = 0.5 →
  track_length = 0.25 →
  gerald_speed_ratio = 0.5 →
  let poly_speed :=  poly_circuits * track_length / poly_time in
  let gerald_speed :=  gerald_speed_ratio * poly_speed in
  gerald_speed = 3 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end geralds_average_speed_l153_153150


namespace jerry_remaining_money_l153_153610

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end jerry_remaining_money_l153_153610


namespace Margo_paired_with_Irma_probability_l153_153164

noncomputable def probability_Margo_paired_with_Irma : ℚ :=
  1 / 29

theorem Margo_paired_with_Irma_probability :
  let total_students := 30
  let number_of_pairings := total_students - 1
  probability_Margo_paired_with_Irma = 1 / number_of_pairings := 
by
  sorry

end Margo_paired_with_Irma_probability_l153_153164


namespace greatest_b_no_minus_six_in_range_l153_153666

open Real

theorem greatest_b_no_minus_six_in_range :
  ∃ (b : ℤ), (b = 8) → (¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 15 = -6) :=
by {
  -- We need to find the largest integer b such that -6 is not in the range of f(x) = x^2 + bx + 15
  sorry
}

end greatest_b_no_minus_six_in_range_l153_153666


namespace exists_prime_q_not_div_n_p_minus_p_l153_153408

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem exists_prime_q_not_div_n_p_minus_p :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p ∧ ∀ n : ℕ, ¬ q ∣ (n ^ p - p) :=
sorry

end exists_prime_q_not_div_n_p_minus_p_l153_153408


namespace dave_winfield_home_runs_l153_153067

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l153_153067


namespace max_streetlights_l153_153338

theorem max_streetlights {road_length streetlight_length : ℝ} 
  (h1 : road_length = 1000)
  (h2 : streetlight_length = 1)
  (fully_illuminated : ∀ (n : ℕ), (n * streetlight_length) < road_length)
  : ∃ max_n, max_n = 1998 ∧ (∀ n, n > max_n → (∃ i, streetlight_length * i > road_length)) :=
sorry

end max_streetlights_l153_153338


namespace negation_of_exists_l153_153956

theorem negation_of_exists (h : ¬ (∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0)) : ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_of_exists_l153_153956


namespace square_area_l153_153340

-- Definition of the vertices' coordinates
def y_coords := ({-3, 2, 2, -3} : Set ℤ)
def x_coords_when_y2 := ({0, 5} : Set ℤ)

-- The statement we need to prove
theorem square_area (h1 : y_coords = {-3, 2, 2, -3}) 
                     (h2 : x_coords_when_y2 = {0, 5}) : 
                     ∃ s : ℤ, s^2 = 25 :=
by
  sorry

end square_area_l153_153340


namespace remainder_when_sum_divided_by_5_l153_153896

/-- Reinterpreting the same conditions and question: -/
theorem remainder_when_sum_divided_by_5 (a b c : ℕ) 
    (ha : a < 5) (hb : b < 5) (hc : c < 5) 
    (h1 : a * b * c % 5 = 1) 
    (h2 : 3 * c % 5 = 2)
    (h3 : 4 * b % 5 = (3 + b) % 5): 
    (a + b + c) % 5 = 4 := 
sorry

end remainder_when_sum_divided_by_5_l153_153896


namespace abs_inequality_solution_l153_153452

theorem abs_inequality_solution (x : ℝ) : (|x + 3| > x + 3) ↔ (x < -3) :=
by
  sorry

end abs_inequality_solution_l153_153452


namespace range_of_m_l153_153899

noncomputable def quadratic_polynomial (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + m^2 - 2

theorem range_of_m (m : ℝ) (h1 : ∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ quadratic_polynomial m x1 = 0 ∧ quadratic_polynomial m x2 = 0) :
  0 < m ∧ m < 1 :=
sorry

end range_of_m_l153_153899


namespace distance_from_A_to_y_axis_l153_153643

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the distance function from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- State the theorem
theorem distance_from_A_to_y_axis :
  distance_to_y_axis point_A = 3 :=
  by
    -- This part will contain the proof, but we omit it with 'sorry' for now.
    sorry

end distance_from_A_to_y_axis_l153_153643


namespace prize_winners_l153_153359

variable (Elaine Frank George Hannah : Prop)

axiom ElaineImpliesFrank : Elaine → Frank
axiom FrankImpliesGeorge : Frank → George
axiom GeorgeImpliesHannah : George → Hannah
axiom OnlyTwoWinners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah)

theorem prize_winners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) → (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) :=
by
  sorry

end prize_winners_l153_153359


namespace power_function_no_origin_l153_153388

theorem power_function_no_origin (m : ℝ) :
  (m = 1 ∨ m = 2) → 
  (m^2 - 3 * m + 3 ≠ 0 ∧ (m - 2) * (m + 1) ≤ 0) :=
by
  intro h
  cases h
  case inl =>
    -- m = 1 case will be processed here
    sorry
  case inr =>
    -- m = 2 case will be processed here
    sorry

end power_function_no_origin_l153_153388


namespace select_4_non_coplanar_points_from_tetrahedron_l153_153957

theorem select_4_non_coplanar_points_from_tetrahedron :
  let points := 10
  let totalWays := Nat.choose points 4
  let sameFaceWays := 4 * Nat.choose 6 4
  let sameEdgeWay := 6
  let parallelogramWay := 3
  totalWays - sameFaceWays - sameEdgeWay - parallelogramWay = 141 :=
by sorry

end select_4_non_coplanar_points_from_tetrahedron_l153_153957


namespace geometric_sequence_product_l153_153254

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l153_153254


namespace patty_coins_value_l153_153943

theorem patty_coins_value (n d q : ℕ) (h₁ : n + d + q = 30) (h₂ : 5 * n + 15 * d - 20 * q = 120) : 
  5 * n + 10 * d + 25 * q = 315 := by
sorry

end patty_coins_value_l153_153943


namespace complex_ratio_of_cubes_l153_153619

theorem complex_ratio_of_cubes (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 10) (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 :=
by
  sorry

end complex_ratio_of_cubes_l153_153619


namespace no_integer_pair_2006_l153_153874

theorem no_integer_pair_2006 : ∀ (x y : ℤ), x^2 - y^2 ≠ 2006 := by
  sorry

end no_integer_pair_2006_l153_153874


namespace unique_solution_l153_153563

noncomputable def f (a b x : ℝ) := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b
noncomputable def g (a b x : ℝ) := 4 * Real.exp (2 * x) + a + b

theorem unique_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃! x, f a b x = ( (a^(1/3) + b^(1/3))/2 )^3 * g a b x :=
sorry

end unique_solution_l153_153563


namespace first_day_price_l153_153195

theorem first_day_price (x n: ℝ) :
  n * x = (n + 100) * (x - 1) ∧ 
  n * x = (n - 200) * (x + 2) → 
  x = 4 :=
by
  sorry

end first_day_price_l153_153195


namespace distance_between_foci_l153_153709

-- Given problem
def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

theorem distance_between_foci :
  ∀ (x y : ℝ),
    hyperbola_eq x y →
    2 * Real.sqrt ((137 / 9) + (137 / 16)) / 72 = 38 * Real.sqrt 7 / 72 :=
by
  intros x y h
  sorry

end distance_between_foci_l153_153709


namespace inequality_amgm_l153_153138

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) : 
  (1 / 2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) <= a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) ∧ 
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) <= (a - b)^2 + (b - c)^2 + (c - a)^2 := 
by 
  sorry

end inequality_amgm_l153_153138


namespace arithmetic_mean_18_27_45_l153_153702

theorem arithmetic_mean_18_27_45 : 
  (18 + 27 + 45) / 3 = 30 :=
by
  -- skipping proof
  sorry

end arithmetic_mean_18_27_45_l153_153702


namespace everett_weeks_worked_l153_153085

theorem everett_weeks_worked (daily_hours : ℕ) (total_hours : ℕ) (days_in_week : ℕ) 
  (h1 : daily_hours = 5) (h2 : total_hours = 140) (h3 : days_in_week = 7) : 
  (total_hours / (daily_hours * days_in_week) = 4) :=
by
  sorry

end everett_weeks_worked_l153_153085


namespace attendance_second_day_l153_153998

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l153_153998


namespace find_unique_function_l153_153361

theorem find_unique_function (f : ℝ → ℝ) (hf1 : ∀ x, 0 ≤ x → 0 ≤ f x)
    (hf2 : ∀ x, 0 ≤ x → f (f x) + f x = 12 * x) :
    ∀ x, 0 ≤ x → f x = 3 * x := 
  sorry

end find_unique_function_l153_153361


namespace brenda_cakes_l153_153694

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l153_153694


namespace income_ratio_l153_153032

variable (U B: ℕ) -- Uma's and Bala's incomes
variable (x: ℕ)  -- Common multiplier for expenditures
variable (savings_amt: ℕ := 2000)  -- Savings amount for both
variable (ratio_expenditure_uma : ℕ := 7)
variable (ratio_expenditure_bala : ℕ := 6)
variable (uma_income : ℕ := 16000)
variable (bala_expenditure: ℕ)

-- Conditions of the problem
-- Uma's Expenditure Calculation
axiom ua_exp_calc : savings_amt = uma_income - ratio_expenditure_uma * x
-- Bala's Expenditure Calculation
axiom bala_income_calc : savings_amt = B - ratio_expenditure_bala * x

theorem income_ratio (h1: U = uma_income) (h2: B = bala_expenditure):
  U * ratio_expenditure_bala = B * ratio_expenditure_uma :=
sorry

end income_ratio_l153_153032


namespace probability_of_specific_sequence_l153_153663

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l153_153663


namespace find_f_of_five_thirds_l153_153411

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_f_of_five_thirds (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_fun : ∀ x : ℝ, f (1 + x) = f (-x))
  (h_val : f (-1 / 3) = 1 / 3) : 
  f (5 / 3) = 1 / 3 :=
  sorry

end find_f_of_five_thirds_l153_153411


namespace compute_x2_y2_l153_153011

theorem compute_x2_y2 (x y : ℝ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := 
by sorry

end compute_x2_y2_l153_153011


namespace is_divisible_by_six_l153_153321

/-- A stingy knight keeps gold coins in six chests. Given that he can evenly distribute the coins by opening any
two chests, any three chests, any four chests, or any five chests, prove that the total number of coins can be 
evenly distributed among all six chests. -/
theorem is_divisible_by_six (n : ℕ) 
  (h2 : ∀ (a b : ℕ), a + b = n → (a % 2 = 0 ∧ b % 2 = 0))
  (h3 : ∀ (a b c : ℕ), a + b + c = n → (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0)) 
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = n → (a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧ d % 4 = 0))
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = n → (a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0)) :
  n % 6 = 0 :=
sorry

end is_divisible_by_six_l153_153321


namespace subtract_and_convert_l153_153156

theorem subtract_and_convert : (3/4 - 1/16 : ℚ) = 0.6875 :=
by
  sorry

end subtract_and_convert_l153_153156


namespace average_of_distinct_u_l153_153105

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l153_153105


namespace lollipop_ratio_l153_153426

/-- Sarah bought 12 lollipops for a total of 3 dollars. Julie gave Sarah 75 cents to pay for the shared lollipops.
Prove that the ratio of the number of lollipops shared to the total number of lollipops bought is 1:4. -/
theorem lollipop_ratio
  (h1 : 12 = lollipops_bought)
  (h2 : 3 = total_cost_dollars)
  (h3 : 75 = amount_paid_cents)
  : (75 / 25) / lollipops_bought = 1/4 :=
sorry

end lollipop_ratio_l153_153426


namespace jogging_track_circumference_l153_153515

theorem jogging_track_circumference (speed_deepak speed_wife : ℝ) (time_meet_minutes : ℝ) 
  (h1 : speed_deepak = 20) (h2 : speed_wife = 16) (h3 : time_meet_minutes = 36) : 
  let relative_speed := speed_deepak + speed_wife
  let time_meet_hours := time_meet_minutes / 60
  let circumference := relative_speed * time_meet_hours
  circumference = 21.6 :=
by
  sorry

end jogging_track_circumference_l153_153515


namespace count_four_digit_numbers_with_thousands_digit_one_l153_153892

theorem count_four_digit_numbers_with_thousands_digit_one : 
  ∃ N : ℕ, N = 1000 ∧ (∀ n : ℕ, 1000 ≤ n ∧ n < 2000 → (n / 1000 = 1)) :=
sorry

end count_four_digit_numbers_with_thousands_digit_one_l153_153892


namespace period_f_axis_of_symmetry_f_max_value_f_l153_153570

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 5)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem axis_of_symmetry_f (k : ℤ) :
  ∀ x, 2 * x - Real.pi / 5 = Real.pi / 4 + k * Real.pi → x = 9 * Real.pi / 40 + k * Real.pi / 2 := sorry

theorem max_value_f :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ x = 7 * Real.pi / 20 := sorry

end period_f_axis_of_symmetry_f_max_value_f_l153_153570


namespace power_of_a_l153_153392

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l153_153392


namespace evaluate_expression_is_41_l153_153084

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l153_153084


namespace value_of_a_l153_153232

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l153_153232


namespace smallest_k_divides_l153_153554

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l153_153554


namespace pond_water_after_45_days_l153_153995

theorem pond_water_after_45_days :
  let initial_amount := 300
  let daily_evaporation := 1
  let rain_every_third_day := 2
  let total_days := 45
  let non_third_days := total_days - (total_days / 3)
  let third_days := total_days / 3
  let total_net_change := (non_third_days * (-daily_evaporation)) + (third_days * (rain_every_third_day - daily_evaporation))
  let final_amount := initial_amount + total_net_change
  final_amount = 285 :=
by
  sorry

end pond_water_after_45_days_l153_153995


namespace difference_is_167_l153_153459

-- Define the number of boys and girls in each village
def A_village_boys : ℕ := 204
def A_village_girls : ℕ := 468
def B_village_boys : ℕ := 334
def B_village_girls : ℕ := 516
def C_village_boys : ℕ := 427
def C_village_girls : ℕ := 458
def D_village_boys : ℕ := 549
def D_village_girls : ℕ := 239

-- Define total number of boys and girls
def total_boys := A_village_boys + B_village_boys + C_village_boys + D_village_boys
def total_girls := A_village_girls + B_village_girls + C_village_girls + D_village_girls

-- Define the difference between total girls and total boys
def difference := total_girls - total_boys

-- The theorem to prove the difference is 167
theorem difference_is_167 : difference = 167 := by
  sorry

end difference_is_167_l153_153459


namespace not_divisible_by_x2_x_1_l153_153961

-- Definitions based on conditions
def P (x : ℂ) (n : ℕ) : ℂ := x^(2 * n) + 1 + (x + 1)^(2 * n)
def divisor (x : ℂ) : ℂ := x^2 + x + 1
def possible_n : List ℕ := [17, 20, 21, 64, 65]

theorem not_divisible_by_x2_x_1 (n : ℕ) (hn : n ∈ possible_n) :
  ¬ ∃ p : ℂ[X], P (C ℂ p) n = divisor p := by
    sorry

end not_divisible_by_x2_x_1_l153_153961


namespace trig_identity_proof_l153_153200

noncomputable def sin_30 : Real := 1 / 2
noncomputable def cos_120 : Real := -1 / 2
noncomputable def cos_45 : Real := Real.sqrt 2 / 2
noncomputable def tan_30 : Real := Real.sqrt 3 / 3

theorem trig_identity_proof : 
  sin_30 + cos_120 + 2 * cos_45 - Real.sqrt 3 * tan_30 = Real.sqrt 2 - 1 := 
by
  sorry

end trig_identity_proof_l153_153200


namespace initial_number_of_persons_l153_153785

-- Define the given conditions
def initial_weights (N : ℕ) : ℝ := 65 * N
def new_person_weight : ℝ := 80
def increased_average_weight : ℝ := 2.5
def weight_increase (N : ℕ) : ℝ := increased_average_weight * N

-- Mathematically equivalent proof problem
theorem initial_number_of_persons 
    (N : ℕ)
    (h : weight_increase N = new_person_weight - 65) : N = 6 :=
by
  -- Place proof here when necessary
  sorry

end initial_number_of_persons_l153_153785


namespace incorrect_statement_l153_153771

open Set

theorem incorrect_statement 
  (M : Set ℝ := {x : ℝ | 0 < x ∧ x < 1})
  (N : Set ℝ := {y : ℝ | 0 < y})
  (R : Set ℝ := univ) : M ∪ N ≠ R :=
by
  sorry

end incorrect_statement_l153_153771


namespace sam_new_crime_books_l153_153692

theorem sam_new_crime_books (used_adventure_books : ℝ) (used_mystery_books : ℝ) (total_books : ℝ) :
  used_adventure_books = 13.0 →
  used_mystery_books = 17.0 →
  total_books = 45.0 →
  total_books - (used_adventure_books + used_mystery_books) = 15.0 :=
by
  intros ha hm ht
  rw [ha, hm, ht]
  norm_num
  -- sorry

end sam_new_crime_books_l153_153692


namespace parabolas_intersect_with_high_probability_l153_153308

noncomputable def high_probability_of_intersection : Prop :=
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 →
  (a - c) ^ 2 + 4 * (b - d) >= 0

theorem parabolas_intersect_with_high_probability : high_probability_of_intersection := sorry

end parabolas_intersect_with_high_probability_l153_153308


namespace george_score_l153_153280

theorem george_score (avg_without_george avg_with_george : ℕ) (num_students : ℕ) 
(h1 : avg_without_george = 75) (h2 : avg_with_george = 76) (h3 : num_students = 20) :
  (num_students * avg_with_george) - ((num_students - 1) * avg_without_george) = 95 :=
by 
  sorry

end george_score_l153_153280


namespace bushels_needed_l153_153209

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end bushels_needed_l153_153209


namespace calc3aMinus4b_l153_153238

theorem calc3aMinus4b (a b : ℤ) (h1 : a * 1 - b * 2 = -1) (h2 : a * 1 + b * 2 = 7) : 3 * a - 4 * b = 1 :=
by
  /- Proof goes here -/
  sorry

end calc3aMinus4b_l153_153238


namespace paul_total_vertical_distance_l153_153142

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l153_153142


namespace number_of_female_democrats_l153_153988

variables (F M D_f : ℕ)

def total_participants := F + M = 660
def female_democrats := D_f = F / 2
def male_democrats := (F / 2) + (M / 4) = 220

theorem number_of_female_democrats 
  (h1 : total_participants F M) 
  (h2 : female_democrats F D_f) 
  (h3 : male_democrats F M) : 
  D_f = 110 := by
  sorry

end number_of_female_democrats_l153_153988


namespace attendance_second_day_l153_153997

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l153_153997


namespace factorization_correct_l153_153048

theorem factorization_correct :
  ∀ (y : ℝ), (y^2 - 1 = (y + 1) * (y - 1)) :=
by
  intro y
  sorry

end factorization_correct_l153_153048


namespace bushels_needed_l153_153208

theorem bushels_needed (cows : ℕ) (bushels_per_cow : ℕ)
                       (sheep : ℕ) (bushels_per_sheep : ℕ)
                       (chickens : ℕ) (bushels_per_chicken : ℕ) :
  (cows = 4) → (bushels_per_cow = 2) →
  (sheep = 3) → (bushels_per_sheep = 2) →
  (chickens = 7) → (bushels_per_chicken = 3) →
  (cows * bushels_per_cow + sheep * bushels_per_sheep + chickens * bushels_per_chicken = 35) :=
begin
  sorry
end

end bushels_needed_l153_153208


namespace must_divide_p_l153_153931

theorem must_divide_p (p q r s : ℕ) 
  (hpq : Nat.gcd p q = 45)
  (hqr : Nat.gcd q r = 75)
  (hrs : Nat.gcd r s = 90)
  (hspt : 150 < Nat.gcd s p)
  (hspb : Nat.gcd s p < 200) : 10 ∣ p := by
  sorry

end must_divide_p_l153_153931


namespace blue_socks_count_l153_153768

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l153_153768


namespace javier_visit_sequences_l153_153924

theorem javier_visit_sequences :
  (finset.perm (finset.mk₀ ["A", "B", "C", "D", "E", "S", "S"])) / (multiset.card (multiset.replicate 2 "S")!) = 360 :=
by
  sorry

end javier_visit_sequences_l153_153924


namespace expression_equals_36_l153_153584

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l153_153584


namespace nancy_tortilla_chips_l153_153420

theorem nancy_tortilla_chips :
  ∀ (total_chips chips_brother chips_herself chips_sister : ℕ),
    total_chips = 22 →
    chips_brother = 7 →
    chips_herself = 10 →
    chips_sister = total_chips - chips_brother - chips_herself →
    chips_sister = 5 :=
by
  intros total_chips chips_brother chips_herself chips_sister
  intro h_total h_brother h_herself h_sister
  rw [h_total, h_brother, h_herself] at h_sister
  simp at h_sister
  assumption

end nancy_tortilla_chips_l153_153420


namespace shortest_path_from_vertex_to_center_of_non_adjacent_face_l153_153347

noncomputable def shortest_path_on_cube (edge_length : ℝ) : ℝ :=
  edge_length + (edge_length * Real.sqrt 2 / 2)

theorem shortest_path_from_vertex_to_center_of_non_adjacent_face :
  shortest_path_on_cube 1 = 1 + Real.sqrt 2 / 2 :=
by
  sorry

end shortest_path_from_vertex_to_center_of_non_adjacent_face_l153_153347


namespace number_of_ways_to_choose_team_l153_153283

-- Define the conditions
def total_players : ℕ := 16
def quadruplets : Finset ℕ := {1, 2, 3, 4}
def other_players : Finset ℕ := (Finset.range 16) \ quadruplets
def choose_three_quadruplets : ℕ := (quadruplets.card.choose 3)
def choose_two_others : ℕ := ((Finset.range 16).card - quadruplets.card).choose 2

-- Prove the number of ways to choose the team
theorem number_of_ways_to_choose_team : choose_three_quadruplets * choose_two_others = 264 := by
  have h1 : quadruplets.card = 4 := by simp [quadruplets]
  have h2 : (Finset.range 16).card = 16 := by simp
  have h3 : other_players.card = 12 := by 
    simp [other_players, quadruplets]
    exact Nat.sub_eq_of_eq_add h1.symm
  simp [h1, h3, choose_three_quadruplets, choose_two_others]
  norm_num
  sorry

end number_of_ways_to_choose_team_l153_153283


namespace cube_root_of_neg_eight_l153_153441

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end cube_root_of_neg_eight_l153_153441


namespace equivalence_of_statements_l153_153194

variable (P Q : Prop)

theorem equivalence_of_statements (h : P → Q) :
  (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalence_of_statements_l153_153194


namespace megan_picture_shelves_l153_153624

def books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 8
def total_books : ℕ := 70
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := total_books - total_mystery_books
def picture_shelves : ℕ := total_picture_books / books_per_shelf

theorem megan_picture_shelves : picture_shelves = 2 := 
by sorry

end megan_picture_shelves_l153_153624


namespace not_factorization_method_l153_153318

theorem not_factorization_method {A B C D : Type} 
  (taking_out_common_factor : A)
  (cross_multiplication_method : B)
  (formula_method : C)
  (addition_subtraction_elimination_method : D) :
  ¬(D) := 
sorry

end not_factorization_method_l153_153318


namespace solve_quartic_equation_l153_153022

theorem solve_quartic_equation (a b c : ℤ) (x : ℤ) : 
  x^4 + a * x^2 + b * x + c = 0 :=
sorry

end solve_quartic_equation_l153_153022


namespace certain_fraction_ratio_l153_153528

theorem certain_fraction_ratio :
  (∃ (x y : ℚ), (x / y) / (6 / 5) = (2 / 5) / 0.14285714285714288) →
  (∃ (x y : ℚ), x / y = 84 / 25) := 
  by
    intros h_ratio
    have h_rat := h_ratio
    sorry

end certain_fraction_ratio_l153_153528


namespace compute_abc_l153_153386

theorem compute_abc (a b c : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h₁ : a + b + c = 30) 
  (h₂ : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c + 300/(a * b * c) = 1) : a * b * c = 768 := 
by 
  sorry

end compute_abc_l153_153386


namespace isabella_exchange_l153_153762

theorem isabella_exchange (d : ℚ) : 
  (8 * d / 5 - 72 = 4 * d) → d = -30 :=
by
  sorry

end isabella_exchange_l153_153762


namespace scientific_notation_of_million_l153_153684

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end scientific_notation_of_million_l153_153684


namespace exists_t_perpendicular_min_dot_product_coordinates_l153_153614

-- Definitions of points
def OA : ℝ × ℝ := (5, 1)
def OB : ℝ × ℝ := (1, 7)
def OC : ℝ × ℝ := (4, 2)

-- Definition of vector OM depending on t
def OM (t : ℝ) : ℝ × ℝ := (4 * t, 2 * t)

-- Definition of vector MA and MB
def MA (t : ℝ) : ℝ × ℝ := (5 - 4 * t, 1 - 2 * t)
def MB (t : ℝ) : ℝ × ℝ := (1 - 4 * t, 7 - 2 * t)

-- Dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Proof that there exists a t such that MA ⊥ MB
theorem exists_t_perpendicular : ∃ t : ℝ, dot_product (MA t) (MB t) = 0 :=
by 
  sorry

-- Proof that coordinates of M minimizing MA ⋅ MB is (4, 2)
theorem min_dot_product_coordinates : ∃ t : ℝ, t = 1 ∧ (OM t) = (4, 2) :=
by
  sorry

end exists_t_perpendicular_min_dot_product_coordinates_l153_153614


namespace solution_z_sq_eq_neg_4_l153_153560

theorem solution_z_sq_eq_neg_4 (x y : ℝ) (i : ℂ) (z : ℂ) (h : z = x + y * i) (hi : i^2 = -1) : 
  z^2 = -4 ↔ z = 2 * i ∨ z = -2 * i := 
by
  sorry

end solution_z_sq_eq_neg_4_l153_153560


namespace find_a_plus_b_l153_153581

theorem find_a_plus_b (a b : ℤ) (h : 2*x^3 - a*x^2 - 5*x + 5 = (2*x^2 + a*x - 1)*(x - b) + 3) : a + b = 4 :=
by {
  -- Proof omitted
  sorry
}

end find_a_plus_b_l153_153581


namespace students_in_section_A_l153_153034

theorem students_in_section_A (x : ℕ) (h1 : (40 : ℝ) * x + 44 * 35 = 37.25 * (x + 44)) : x = 36 :=
by
  sorry

end students_in_section_A_l153_153034


namespace each_girl_gets_2_dollars_l153_153640

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l153_153640


namespace only_solution_for_triplet_l153_153527

theorem only_solution_for_triplet (x y z : ℤ) (h : x^2 + y^2 + z^2 - 2 * x * y * z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end only_solution_for_triplet_l153_153527


namespace trig_identity_l153_153243

theorem trig_identity (A : ℝ) (h : Real.cos (π + A) = -1/2) : Real.sin (π / 2 + A) = 1/2 :=
by 
sorry

end trig_identity_l153_153243


namespace avg_tickets_sold_by_males_100_l153_153395

theorem avg_tickets_sold_by_males_100 
  (female_avg : ℕ := 70) 
  (nonbinary_avg : ℕ := 50) 
  (overall_avg : ℕ := 66) 
  (male_ratio : ℕ := 2) 
  (female_ratio : ℕ := 3) 
  (nonbinary_ratio : ℕ := 5) : 
  ∃ (male_avg : ℕ), male_avg = 100 := 
by 
  sorry

end avg_tickets_sold_by_males_100_l153_153395


namespace F_atoms_in_compound_l153_153329

-- Given conditions
def atomic_weight_Al : Real := 26.98
def atomic_weight_F : Real := 19.00
def molecular_weight : Real := 84

-- Defining the assertion: number of F atoms in the compound
def number_of_F_atoms (n : Real) : Prop :=
  molecular_weight = atomic_weight_Al + n * atomic_weight_F

-- Proving the assertion that the number of F atoms is approximately 3
theorem F_atoms_in_compound : number_of_F_atoms 3 :=
  by
  sorry

end F_atoms_in_compound_l153_153329


namespace polynomial_identity_l153_153592

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l153_153592


namespace perimeter_of_square_l153_153803

variable (s : ℝ) (side_length : ℝ)
def is_square_side_length_5 (s : ℝ) : Prop := s = 5
theorem perimeter_of_square (h: is_square_side_length_5 s) : 4 * s = 20 := sorry

end perimeter_of_square_l153_153803


namespace simplify_expression_l153_153630

theorem simplify_expression (a b : ℝ) :
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 :=
by
  sorry

end simplify_expression_l153_153630


namespace container_capacity_l153_153991

theorem container_capacity
  (C : ℝ)  -- Total capacity of the container in liters
  (h1 : C / 2 + 20 = 3 * C / 4)  -- Condition combining the water added and the fractional capacities
  : C = 80 := 
sorry

end container_capacity_l153_153991


namespace car_pass_time_l153_153481

theorem car_pass_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) :
  length = 10 → 
  speed_kmph = 36 → 
  speed_mps = speed_kmph * (1000 / 3600) → 
  time = length / speed_mps → 
  time = 1 :=
by
  intros h_length h_speed_kmph h_speed_conversion h_time_calculation
  -- Here we would normally construct the proof
  sorry

end car_pass_time_l153_153481


namespace train_crossing_time_l153_153480

-- Define the problem conditions in Lean 4
def train_length : ℕ := 130
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := (speed_kmph * 1000 / 3600)

-- The statement to prove
theorem train_crossing_time : (train_length + bridge_length) / speed_mps = 28 :=
by
  -- The proof starts here
  sorry

end train_crossing_time_l153_153480


namespace mul_value_proof_l153_153431

theorem mul_value_proof :
  ∃ x : ℝ, (8.9 - x = 3.1) ∧ ((x * 3.1) * 2.5 = 44.95) :=
by
  sorry

end mul_value_proof_l153_153431


namespace negation_equivalence_l153_153450

theorem negation_equivalence : (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
  sorry

end negation_equivalence_l153_153450


namespace maisie_flyers_count_l153_153014

theorem maisie_flyers_count (M : ℕ) (h1 : 71 = 2 * M + 5) : M = 33 :=
by
  sorry

end maisie_flyers_count_l153_153014


namespace dave_winfield_home_runs_l153_153065

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l153_153065


namespace solve_quadratic_and_compute_l153_153497

theorem solve_quadratic_and_compute (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) : (8 * y - 2)^2 = 248 := 
sorry

end solve_quadratic_and_compute_l153_153497


namespace cosine_squared_identity_l153_153099

theorem cosine_squared_identity (α : ℝ) (h : sin α - cos α = 1/3) :
  cos (π/4 - α) ^ 2 = 17/18 := by
  sorry

end cosine_squared_identity_l153_153099


namespace find_line_AB_l153_153240

noncomputable def equation_of_line_AB : Prop :=
  ∀ (x y : ℝ), ((x-2)^2 + (y-1)^2 = 10) ∧ ((x+6)^2 + (y+3)^2 = 50) → (2*x + y = 0)

theorem find_line_AB : equation_of_line_AB := by
  sorry

end find_line_AB_l153_153240


namespace correct_option_is_C_l153_153173

theorem correct_option_is_C (x y : ℝ) :
  ¬(3 * x + 4 * y = 12 * x * y) ∧
  ¬(x^9 / x^3 = x^3) ∧
  ((x^2)^3 = x^6) ∧
  ¬((x - y)^2 = x^2 - y^2) :=
by
  sorry

end correct_option_is_C_l153_153173


namespace mark_bread_time_l153_153419

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end mark_bread_time_l153_153419


namespace compute_a_l153_153102

theorem compute_a 
  (a b : ℚ) 
  (h : ∃ (x : ℝ), x^3 + (a : ℝ) * x^2 + (b : ℝ) * x - 37 = 0 ∧ x = 2 - 3 * Real.sqrt 3) : 
  a = -55 / 23 :=
by 
  sorry

end compute_a_l153_153102


namespace sin_120_eq_sqrt3_div_2_l153_153968

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l153_153968


namespace books_sold_over_summer_l153_153954

theorem books_sold_over_summer (n l t : ℕ) (h1 : n = 37835) (h2 : l = 143) (h3 : t = 271) : 
  t - l = 128 :=
by
  sorry

end books_sold_over_summer_l153_153954


namespace gcd_91_49_l153_153807

theorem gcd_91_49 : Nat.gcd 91 49 = 7 :=
by
  -- Using the Euclidean algorithm
  -- 91 = 49 * 1 + 42
  -- 49 = 42 * 1 + 7
  -- 42 = 7 * 6 + 0
  sorry

end gcd_91_49_l153_153807


namespace exponentiation_identity_l153_153665

theorem exponentiation_identity :
  (5^4)^2 = 390625 :=
  by sorry

end exponentiation_identity_l153_153665


namespace ratio_x_y_l153_153123

theorem ratio_x_y (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1 / 2) : 
  x / y = 3 / (6 * x - 1) := 
sorry

end ratio_x_y_l153_153123


namespace maximize_area_l153_153059

noncomputable def optimal_fencing (L W : ℝ) : Prop :=
  (2 * L + W = 1200) ∧ (∀ L1 W1, 2 * L1 + W1 = 1200 → L * W ≥ L1 * W1)

theorem maximize_area : ∃ L W, optimal_fencing L W ∧ L + W = 900 := sorry

end maximize_area_l153_153059


namespace cathy_wins_probability_l153_153836

theorem cathy_wins_probability : 
  (∑' (n : ℕ), (1 / 6 : ℚ)^3 * (5 / 6)^(3 * n)) = 1 / 91 
:= by sorry

end cathy_wins_probability_l153_153836


namespace cricket_bat_cost_l153_153193

noncomputable def CP_A_sol : ℝ := 444.96 / 1.95

theorem cricket_bat_cost (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (SP_D : ℝ) :
  (SP_B = 1.20 * CP_A) →
  (SP_C = 1.25 * SP_B) →
  (SP_D = 1.30 * SP_C) →
  (SP_D = 444.96) →
  CP_A = CP_A_sol :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_bat_cost_l153_153193


namespace negation_of_at_least_three_is_at_most_two_l153_153792

theorem negation_of_at_least_three_is_at_most_two :
  (¬ (∀ n : ℕ, n ≥ 3)) ↔ (∃ n : ℕ, n ≤ 2) :=
sorry

end negation_of_at_least_three_is_at_most_two_l153_153792


namespace problem_solution_l153_153589

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l153_153589


namespace painting_problem_equation_l153_153077

def dougPaintingRate := 1 / 3
def davePaintingRate := 1 / 4
def combinedPaintingRate := dougPaintingRate + davePaintingRate
def timeRequiredToComplete (t : ℝ) : Prop := 
  (t - 1) * combinedPaintingRate = 2 / 3

theorem painting_problem_equation : ∃ t : ℝ, timeRequiredToComplete t :=
sorry

end painting_problem_equation_l153_153077


namespace problem_statement_l153_153237

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x <= -3}
def R (S : Set ℝ) : Set ℝ := {x | ∃ y ∈ S, x = y}

theorem problem_statement : R (M ∪ N) = {x | x >= 1} :=
by
  sorry

end problem_statement_l153_153237


namespace ratio_population_A_to_F_l153_153510

variable (F : ℕ)

def population_E := 6 * F
def population_D := 2 * population_E
def population_C := 8 * population_D
def population_B := 3 * population_C
def population_A := 5 * population_B

theorem ratio_population_A_to_F (F_pos : F > 0) :
  population_A F / F = 1440 := by
sorry

end ratio_population_A_to_F_l153_153510


namespace quadratic_vertex_coordinates_l153_153221

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ :=
  -2 * (x + 1)^2 - 4

-- State the main theorem to be proved: The vertex of the quadratic function is at (-1, -4)
theorem quadratic_vertex_coordinates : 
  ∃ h k : ℝ, ∀ x : ℝ, quadratic x = -2 * (x + h)^2 + k ∧ h = -1 ∧ k = -4 := 
by
  -- proof required here
  sorry

end quadratic_vertex_coordinates_l153_153221


namespace mean_proportional_AC_is_correct_l153_153095

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end mean_proportional_AC_is_correct_l153_153095


namespace find_second_number_l153_153950

theorem find_second_number (x : ℝ) (h : (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8) : x = 40 :=
sorry

end find_second_number_l153_153950


namespace polynomial_identity_l153_153594

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l153_153594


namespace a_2_is_minus_1_l153_153455
open Nat

variable (a S : ℕ → ℤ)

-- Conditions
axiom sum_first_n (n : ℕ) (hn : n > 0) : 2 * S n - n * a n = n
axiom S_20 : S 20 = -360

-- The problem statement to prove
theorem a_2_is_minus_1 : a 2 = -1 :=
by 
  sorry

end a_2_is_minus_1_l153_153455


namespace total_pages_read_l153_153260

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l153_153260


namespace coeff_abs_sum_eq_729_l153_153621

-- Given polynomial (2x - 1)^6 expansion
theorem coeff_abs_sum_eq_729 (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (2 * x - 1) ^ 6 = a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end coeff_abs_sum_eq_729_l153_153621


namespace sequence_is_increasing_l153_153880

theorem sequence_is_increasing :
  ∀ n m : ℕ, n < m → (1 - 2 / (n + 1) : ℝ) < (1 - 2 / (m + 1) : ℝ) :=
by
  intro n m hnm
  have : (2 : ℝ) / (n + 1) > 2 / (m + 1) :=
    sorry
  linarith [this]

end sequence_is_increasing_l153_153880


namespace factor_expression_l153_153512

-- Given conditions (none explicitly stated)
-- Definitions for the expressions
def initial_expr : ℤ[y] := 16 * y^6 + 36 * y^4 - 9 - (4 * y^6 - 6 * y^4 + 9)

-- The goal to prove
theorem factor_expression : initial_expr = 6 * (2 * y^6 + 7 * y^4 - 3) :=
by
  sorry

end factor_expression_l153_153512


namespace probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3_l153_153365

def pow_units_cycle (base : ℕ) : List ℕ :=
  match base % 10 with
  | 2 => [2, 4, 8, 6]
  | 5 => [5]
  | _ => []

noncomputable def probability_units_digit_3 : ℚ :=
  let a_vals := List.range 50
  let b_vals := List.range 50
  let counts := a_vals.filter (fun a =>
    (2^a % 10 - 5^5 % 10) % 10 == 3).length
  counts / (50 * 50)

theorem probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3 :
  probability_units_digit_3 = 6 / 25 := 
sorry

end probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3_l153_153365


namespace point_A_is_closer_to_origin_l153_153019

theorem point_A_is_closer_to_origin (A B : ℤ) (hA : A = -2) (hB : B = 3) : abs A < abs B := by 
sorry

end point_A_is_closer_to_origin_l153_153019


namespace M_intersection_N_eq_N_l153_153886

def M := { x : ℝ | x < 4 }
def N := { x : ℝ | x ≤ -2 }

theorem M_intersection_N_eq_N : M ∩ N = N :=
by
  sorry

end M_intersection_N_eq_N_l153_153886


namespace acres_for_corn_l153_153320

theorem acres_for_corn (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ)
  (total_ratio : beans_ratio + wheat_ratio + corn_ratio = 11)
  (land_parts : total_land / 11 = 94)
  : (corn_ratio = 4) → (total_land = 1034) → 4 * 94 = 376 :=
by
  intros
  sorry

end acres_for_corn_l153_153320


namespace find_A_l153_153314

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 100 * A + 78 - (200 + B) = 364) : A = 5 :=
by
  sorry

end find_A_l153_153314


namespace area_PQRS_l153_153149

-- Define the conditions
def EFGH_area := 36
def side_length_EFGH := Real.sqrt EFGH_area
def side_length_equilateral := side_length_EFGH
def displacement := (side_length_EFGH * Real.sqrt 3) / 2
def side_length_PQRS := side_length_EFGH + 2 * displacement

-- Prove the question (area of PQRS)
theorem area_PQRS : 
  (side_length_PQRS)^2 = 144 + 72 * Real.sqrt 3 := by
  sorry

end area_PQRS_l153_153149


namespace num_divisors_multiple_of_4_9_fact_correct_l153_153576

open Nat

noncomputable def num_divisors_multiple_of_4_9_fact : ℕ :=
  let fact_9 := factorial 9
  let prime_factors := (2 ^ 7) * (3 ^ 4) * 5 * 7
  let choices_for_a := 6 -- 2 to 7 inclusive
  let choices_for_b := 5 -- 0 to 4 inclusive
  let choices_for_c := 2 -- 0 to 1 inclusive
  let choices_for_d := 2 -- 0 to 1 inclusive
  choices_for_a * choices_for_b * choices_for_c * choices_for_d

theorem num_divisors_multiple_of_4_9_fact_correct : num_divisors_multiple_of_4_9_fact = 120 := by
  sorry

end num_divisors_multiple_of_4_9_fact_correct_l153_153576


namespace sandy_paid_for_pants_l153_153780

-- Define the costs and change as constants
def cost_of_shirt : ℝ := 8.25
def amount_paid_with : ℝ := 20.00
def change_received : ℝ := 2.51

-- Define the amount paid for pants
def amount_paid_for_pants : ℝ := 9.24

-- The theorem stating the problem
theorem sandy_paid_for_pants : 
  amount_paid_with - (cost_of_shirt + change_received) = amount_paid_for_pants := 
by 
  -- proof is required here
  sorry

end sandy_paid_for_pants_l153_153780


namespace relationship_of_points_l153_153249

variable (y k b x : ℝ)
variable (y1 y2 : ℝ)

noncomputable def linear_func (x : ℝ) : ℝ := k * x - b

theorem relationship_of_points
  (h_pos_k : k > 0)
  (h_point1 : linear_func k b (-1) = y1)
  (h_point2 : linear_func k b 2 = y2):
  y1 < y2 := 
sorry

end relationship_of_points_l153_153249


namespace range_of_xy_l153_153732

theorem range_of_xy {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y)
    (h₃ : x + 2/x + 3*y + 4/y = 10) : 
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end range_of_xy_l153_153732


namespace total_sheep_l153_153339

variable (x y : ℕ)
/-- Initial condition: After one ram runs away, the ratio of rams to ewes is 7:5. -/
def initial_ratio (x y : ℕ) : Prop := 5 * (x - 1) = 7 * y
/-- Second condition: After the ram returns and one ewe runs away, the ratio of rams to ewes is 5:3. -/
def second_ratio (x y : ℕ) : Prop := 3 * x = 5 * (y - 1)
/-- The total number of sheep in the flock initially is 25. -/
theorem total_sheep (x y : ℕ) 
  (h1 : initial_ratio x y) 
  (h2 : second_ratio x y) : 
  x + y = 25 := 
by sorry

end total_sheep_l153_153339


namespace solve_linear_system_l153_153023

theorem solve_linear_system (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  2 * x₁ + 2 * x₂ - x₃ + x₄ + 4 * x₆ = 0 ∧
  x₁ + 2 * x₂ + 2 * x₃ + 3 * x₅ + x₆ = -2 ∧
  x₁ - 2 * x₂ + x₄ + 2 * x₅ = 0 →
  x₁ = -1 / 4 - 5 / 8 * x₄ - 9 / 8 * x₅ - 9 / 8 * x₆ ∧
  x₂ = -1 / 8 + 3 / 16 * x₄ - 7 / 16 * x₅ + 9 / 16 * x₆ ∧
  x₃ = -3 / 4 + 1 / 8 * x₄ - 11 / 8 * x₅ + 5 / 8 * x₆ :=
by
  sorry

end solve_linear_system_l153_153023


namespace part_I_part_II_l153_153740

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ a b ∈ I, a < b → f a < f b

theorem part_I
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = Real.sin (2 * x - Real.pi / 6))
  (hω : 2 > 0)
  (hϕ : 0 < Real.pi / 6 ∧ Real.pi / 6 < Real.pi / 2) :
  (f = λ x, Real.sin (2 * x - Real.pi / 6)) ∧
  (is_monotonically_increasing_on f (Set.Ioc 0 (Real.pi / 3)) ∧
   is_monotonically_increasing_on f (Set.Ioc (5 * Real.pi / 6) Real.pi)) :=
sorry

theorem part_II
  (A : ℝ)
  (hA : f (A / 2) + Real.cos A = 1 / 2) :
  A = 2 * Real.pi / 3 :=
sorry

end part_I_part_II_l153_153740


namespace yardwork_payment_l153_153368

theorem yardwork_payment :
  let earnings := [15, 20, 25, 40]
  let total_earnings := List.sum earnings
  let equal_share := total_earnings / earnings.length
  let high_earner := 40
  high_earner - equal_share = 15 :=
by
  sorry

end yardwork_payment_l153_153368


namespace math_books_together_l153_153315

theorem math_books_together (math_books english_books : ℕ) (h_math_books : math_books = 2) (h_english_books : english_books = 2) : 
  ∃ ways, ways = 12 := by
  sorry

end math_books_together_l153_153315


namespace expression_calculates_to_l153_153845

noncomputable def mixed_number : ℚ := 3 + 3 / 4

noncomputable def decimal_to_fraction : ℚ := 2 / 10

noncomputable def given_expression : ℚ := ((mixed_number * decimal_to_fraction) / 135) * 5.4

theorem expression_calculates_to : given_expression = 0.03 := by
  sorry

end expression_calculates_to_l153_153845


namespace factorize_expression_l153_153213

theorem factorize_expression (x y : ℝ) :
  9 * x^2 - y^2 - 4 * y - 4 = (3 * x + y + 2) * (3 * x - y - 2) :=
by
  sorry

end factorize_expression_l153_153213


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l153_153205

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l153_153205


namespace currency_notes_total_l153_153005

theorem currency_notes_total (num_50_notes total_amount remaining_amount num_100_notes : ℕ) 
  (h1 : remaining_amount = total_amount - (num_50_notes * 50))
  (h2 : num_50_notes = 3500 / 50)
  (h3 : total_amount = 5000)
  (h4 : remaining_amount = 1500)
  (h5 : num_100_notes = remaining_amount / 100) : 
  num_50_notes + num_100_notes = 85 :=
by sorry

end currency_notes_total_l153_153005


namespace complement_U_A_l153_153114

open Set

def U : Set ℤ := univ
def A : Set ℤ := { x | x^2 - x - 2 ≥ 0 }

theorem complement_U_A :
  (U \ A) = { 0, 1 } := by
  sorry

end complement_U_A_l153_153114


namespace binary_multiplication_l153_153533

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l153_153533


namespace shrink_ray_coffee_l153_153829

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l153_153829


namespace arithmetic_progression_sum_l153_153575

theorem arithmetic_progression_sum (a d S n : ℤ) (h_a : a = 32) (h_d : d = -4) (h_S : S = 132) :
  (n = 6 ∨ n = 11) :=
by
  -- Start the proof here
  sorry

end arithmetic_progression_sum_l153_153575


namespace tan_theta_solution_l153_153875

theorem tan_theta_solution (θ : ℝ) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  Real.tan θ = 0 ∨ Real.tan θ = 4 / 3 :=
sorry

end tan_theta_solution_l153_153875


namespace no_non_integer_point_exists_l153_153613

variable (b0 b1 b2 b3 b4 b5 u v : ℝ)

def q (x y : ℝ) : ℝ := b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2

theorem no_non_integer_point_exists
    (h₀ : q b0 b1 b2 b3 b4 b5 0 0 = 0)
    (h₁ : q b0 b1 b2 b3 b4 b5 1 0 = 0)
    (h₂ : q b0 b1 b2 b3 b4 b5 (-1) 0 = 0)
    (h₃ : q b0 b1 b2 b3 b4 b5 0 1 = 0)
    (h₄ : q b0 b1 b2 b3 b4 b5 0 (-1) = 0)
    (h₅ : q b0 b1 b2 b3 b4 b5 1 1 = 0) :
  ∀ u v : ℝ, (¬ ∃ (n m : ℤ), u = n ∧ v = m) → q b0 b1 b2 b3 b4 b5 u v ≠ 0 :=
by
  sorry

end no_non_integer_point_exists_l153_153613


namespace elasticity_ratio_is_correct_l153_153843

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end elasticity_ratio_is_correct_l153_153843


namespace probability_spade_then_ace_l153_153038

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l153_153038


namespace largest_prime_factor_8250_l153_153667

-- Define a function to check if a number is prime (using an existing library function)
def is_prime (n: ℕ) : Prop := Nat.Prime n

-- Define the given problem statement as a Lean theorem
theorem largest_prime_factor_8250 :
  ∃ p, is_prime p ∧ p ∣ 8250 ∧ 
    ∀ q, is_prime q ∧ q ∣ 8250 → q ≤ p :=
sorry -- The proof will be filled in later

end largest_prime_factor_8250_l153_153667


namespace ratio_problem_l153_153489

theorem ratio_problem (x n : ℕ) (h1 : 5 * x = n) (h2 : n = 65) : x = 13 :=
by
  sorry

end ratio_problem_l153_153489


namespace tangent_line_through_M_to_circle_l153_153341

noncomputable def M : ℝ × ℝ := (2, -1)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem tangent_line_through_M_to_circle :
  ∀ {x y : ℝ}, circle_eq x y → M = (2, -1) → 2*x - y - 5 = 0 :=
sorry

end tangent_line_through_M_to_circle_l153_153341


namespace reciprocal_power_l153_153390

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l153_153390


namespace range_f_domain_g_monotonicity_intervals_g_l153_153676

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.cos x + 1

theorem range_f : 
  (∀ x ∈ Set.Icc (-(Real.pi / 2)) (Real.pi / 2), 2 ≤ f x ∧ f x ≤ 9 / 4) :=
  sorry

noncomputable def g (x : ℝ) : ℝ := Real.tan (x / 2 + Real.pi / 3)

theorem domain_g : 
  {x : ℝ | ∃ k : ℤ, x = Real.pi / 3 + 2 * k * Real.pi} = ∅ :=
  sorry

theorem monotonicity_intervals_g : 
  (∀ k : ℤ, Set.Ioo (-(5 * Real.pi / 3) + 2 * k * Real.pi) 
  (Real.pi / 3 + 2 * k * Real.pi) ⊆ {x | ∃ k : ℤ, x ∈ (-(5 * Real.pi / 3) + 2 * k * Real.pi, Real.pi / 3 + 2 * k * Real.pi)}) :=
  sorry

end range_f_domain_g_monotonicity_intervals_g_l153_153676


namespace return_trip_avg_speed_l153_153064

noncomputable def avg_speed_return_trip : ℝ := 
  let distance_ab_to_sy := 120
  let rate_ab_to_sy := 50
  let total_time := 5.5
  let time_ab_to_sy := distance_ab_to_sy / rate_ab_to_sy
  let time_return_trip := total_time - time_ab_to_sy
  distance_ab_to_sy / time_return_trip

theorem return_trip_avg_speed 
  (distance_ab_to_sy : ℝ := 120)
  (rate_ab_to_sy : ℝ := 50)
  (total_time : ℝ := 5.5) 
  : avg_speed_return_trip = 38.71 :=
by
  sorry

end return_trip_avg_speed_l153_153064


namespace brenda_cakes_l153_153693

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l153_153693


namespace ratio_of_common_differences_l153_153118

variable (x y d1 d2 : ℝ)

theorem ratio_of_common_differences (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0) 
  (seq1 : x + 4 * d1 = y) (seq2 : x + 5 * d2 = y) : d1 / d2 = 5 / 4 := 
sorry

end ratio_of_common_differences_l153_153118


namespace card_sequence_probability_l153_153655

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l153_153655


namespace find_original_number_l153_153607

noncomputable def three_digit_number (d e f : ℕ) := 100 * d + 10 * e + f

/-- Given conditions and the sum S, determine the original three-digit number -/
theorem find_original_number (S : ℕ) (d e f : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9)
  (h2 : 0 ≤ e ∧ e ≤ 9) (h3 : 0 ≤ f ∧ f ≤ 9) (h4 : S = 4321) :
  three_digit_number d e f = 577 :=
sorry


end find_original_number_l153_153607


namespace find_sets_l153_153872

theorem find_sets (A B : Set ℕ) :
  A ∩ B = {1, 2, 3} ∧ A ∪ B = {1, 2, 3, 4, 5} →
    (A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
    (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
    (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
    (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4}) :=
by
  sorry

end find_sets_l153_153872


namespace tiling_possible_if_and_only_if_one_dimension_is_integer_l153_153309

-- Define our conditions: a, b are dimensions of the board and t is the positive dimension of the small rectangles
variable (a b : ℝ) (t : ℝ)

-- Define corresponding properties for these variables
axiom pos_t : t > 0

-- Theorem stating the condition for tiling
theorem tiling_possible_if_and_only_if_one_dimension_is_integer (a_non_int : ¬ ∃ z : ℤ, a = z) (b_non_int : ¬ ∃ z : ℤ, b = z) :
  ∃ n m : ℕ, n * 1 + m * t = a * b :=
sorry

end tiling_possible_if_and_only_if_one_dimension_is_integer_l153_153309


namespace smallest_k_divides_l153_153536

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l153_153536


namespace chimney_bricks_l153_153508

theorem chimney_bricks (x : ℝ) 
  (h1 : ∀ x, Brenda_rate = x / 8) 
  (h2 : ∀ x, Brandon_rate = x / 12) 
  (h3 : Combined_rate = (Brenda_rate + Brandon_rate - 15)) 
  (h4 : x = Combined_rate * 6) 
  : x = 360 := 
by 
  sorry

end chimney_bricks_l153_153508


namespace find_x_plus_y_l153_153413

theorem find_x_plus_y (x y : ℝ)
  (h1 : (x - 1)^3 + 2015 * (x - 1) = -1)
  (h2 : (y - 1)^3 + 2015 * (y - 1) = 1)
  : x + y = 2 :=
sorry

end find_x_plus_y_l153_153413


namespace card_at_42_is_8_spade_l153_153079

-- Conditions Definition
def cards_sequence : List String := 
  ["A♥", "A♠", "2♥", "2♠", "3♥", "3♠", "4♥", "4♠", "5♥", "5♠", "6♥", "6♠", "7♥", "7♠", "8♥", "8♠",
   "9♥", "9♠", "10♥", "10♠", "J♥", "J♠", "Q♥", "Q♠", "K♥", "K♠"]

-- Proposition to be proved
theorem card_at_42_is_8_spade :
  cards_sequence[(41 % 26)] = "8♠" :=
by sorry

end card_at_42_is_8_spade_l153_153079


namespace circle_radius_l153_153490

theorem circle_radius (r A C : Real) (h1 : A = π * r^2) (h2 : C = 2 * π * r) (h3 : A + (Real.cos (π / 3)) * C = 56 * π) : r = 7 := 
by 
  sorry

end circle_radius_l153_153490


namespace value_of_a_l153_153124

theorem value_of_a {a : ℝ} (h : ∀ x y : ℝ, (a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) → x = y) : a = 0 ∨ a = 1 := 
  sorry

end value_of_a_l153_153124


namespace number_corresponding_to_8_minutes_l153_153180

theorem number_corresponding_to_8_minutes (x : ℕ) : 
  (12 / 6 = x / 480) → x = 960 :=
by
  sorry

end number_corresponding_to_8_minutes_l153_153180


namespace loan_percentage_correct_l153_153971

-- Define the parameters and conditions of the problem
def house_initial_value : ℕ := 100000
def house_increase_percentage : ℝ := 0.25
def new_house_cost : ℕ := 500000
def loan_percentage : ℝ := 75.0

-- Define the theorem we want to prove
theorem loan_percentage_correct :
  let increase_value := house_initial_value * house_increase_percentage
  let sale_price := house_initial_value + increase_value
  let loan_amount := new_house_cost - sale_price
  let loan_percentage_computed := (loan_amount / new_house_cost) * 100
  loan_percentage_computed = loan_percentage :=
by
  -- Proof placeholder
  sorry

end loan_percentage_correct_l153_153971


namespace karlson_wins_with_optimal_play_l153_153199

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play_l153_153199


namespace brenda_cakes_l153_153695

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l153_153695


namespace one_over_x_plus_one_over_y_eq_two_l153_153378

theorem one_over_x_plus_one_over_y_eq_two 
  (x y : ℝ)
  (h1 : 3^x = Real.sqrt 12)
  (h2 : 4^y = Real.sqrt 12) : 
  1 / x + 1 / y = 2 := 
by 
  sorry

end one_over_x_plus_one_over_y_eq_two_l153_153378


namespace least_four_digit_with_factors_3_5_7_l153_153468

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l153_153468


namespace range_f_g_f_eq_g_implies_A_l153_153384

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

theorem range_f_g :
  (range f ∩ Icc 1 17 = Icc 1 17) ∧ (range g ∩ Icc 1 17 = Icc 1 17) :=
sorry

theorem f_eq_g_implies_A :
  ∀ A ⊆ Icc 0 4, (∀ x ∈ A, f x = g x) → A = {0} ∨ A = {4} ∨ A = {0, 4} :=
sorry

end range_f_g_f_eq_g_implies_A_l153_153384


namespace train_around_probability_train_present_when_alex_arrives_l153_153457

noncomputable def trainArrivalTime : Set ℝ := Set.Icc 15 45
noncomputable def trainWaitTime (t : ℝ) : Set ℝ := Set.Icc t (t + 15)
noncomputable def alexArrivalTime : Set ℝ := Set.Icc 0 60

theorem train_around (t : ℝ) (h : t ∈ trainArrivalTime) :
  ∀ (x : ℝ), x ∈ alexArrivalTime → x ∈ trainWaitTime t ↔ 15 ≤ t ∧ t ≤ 45 ∧ t ≤ x ∧ x ≤ t + 15 :=
sorry

theorem probability_train_present_when_alex_arrives :
  let total_area := 60 * 60
  let favorable_area := 1 / 2 * (15 + 15) * 15
  (favorable_area / total_area) = 1 / 16 :=
sorry

end train_around_probability_train_present_when_alex_arrives_l153_153457


namespace fox_initial_coins_l153_153972

theorem fox_initial_coins :
  ∃ (x : ℕ), ∀ (c1 c2 c3 : ℕ),
    c1 = 3 * x - 50 ∧
    c2 = 3 * c1 - 50 ∧
    c3 = 3 * c2 - 50 ∧
    3 * c3 - 50 = 20 →
    x = 25 :=
by
  sorry

end fox_initial_coins_l153_153972


namespace am_gm_inequality_example_am_gm_inequality_equality_condition_l153_153930

theorem am_gm_inequality_example (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 :=
sorry

theorem am_gm_inequality_equality_condition (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2) ↔ (x = 0 ∧ y = 0 ∨ x = 1 ∧ y = 1) :=
sorry

end am_gm_inequality_example_am_gm_inequality_equality_condition_l153_153930


namespace polynomial_division_result_l153_153010

-- Define the given polynomials
def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + 2 * x + 3
def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3

-- Define the computed quotient and remainder
def q (x : ℝ) : ℝ := 4 * x ^ 2 + 4
def r (x : ℝ) : ℝ := -12 * x + 42

theorem polynomial_division_result :
  (∀ x : ℝ, f x = q x * d x + r x) ∧ (q 1 + r (-1) = 62) :=
by
  sorry

end polynomial_division_result_l153_153010


namespace kiyana_gives_half_l153_153770

theorem kiyana_gives_half (total_grapes : ℕ) (h : total_grapes = 24) : 
  (total_grapes / 2) = 12 :=
by
  sorry

end kiyana_gives_half_l153_153770


namespace cannot_be_correct_average_l153_153678

theorem cannot_be_correct_average (a : ℝ) (h_pos : a > 0) (h_median : a ≤ 12) : 
  ∀ avg, avg = (12 + a + 8 + 15 + 23) / 5 → avg ≠ 71 / 5 := 
by
  intro avg h_avg
  sorry

end cannot_be_correct_average_l153_153678


namespace alice_total_cost_usd_is_correct_l153_153345

def tea_cost_yen : ℕ := 250
def sandwich_cost_yen : ℕ := 350
def conversion_rate : ℕ := 100
def total_cost_usd (tea_cost_yen sandwich_cost_yen conversion_rate : ℕ) : ℕ :=
  (tea_cost_yen + sandwich_cost_yen) / conversion_rate

theorem alice_total_cost_usd_is_correct :
  total_cost_usd tea_cost_yen sandwich_cost_yen conversion_rate = 6 := 
by
  sorry

end alice_total_cost_usd_is_correct_l153_153345


namespace system_of_equations_solution_l153_153821

theorem system_of_equations_solution (x y : ℚ) :
  (x / 3 + y / 4 = 4 ∧ 2 * x - 3 * y = 12) → (x = 10 ∧ y = 8 / 3) :=
by
  sorry

end system_of_equations_solution_l153_153821


namespace reciprocal_power_l153_153391

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l153_153391


namespace find_fraction_l153_153984

theorem find_fraction (n d : ℕ) (h1 : n / (d + 1) = 1 / 2) (h2 : (n + 1) / d = 1) : n / d = 2 / 3 := 
by 
  sorry

end find_fraction_l153_153984


namespace consecutive_even_product_l153_153795

-- Define that there exist three consecutive even numbers such that the product equals 87526608.
theorem consecutive_even_product (a : ℤ) : 
  (a - 2) * a * (a + 2) = 87526608 → ∃ b : ℤ, b = a - 2 ∧ b % 2 = 0 ∧ ∃ c : ℤ, c = a ∧ c % 2 = 0 ∧ ∃ d : ℤ, d = a + 2 ∧ d % 2 = 0 :=
sorry

end consecutive_even_product_l153_153795


namespace general_formula_a_general_formula_c_l153_153725

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem general_formula_a (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 := sorry

-- Definitions for the second problem
def f (x : ℝ) : ℝ := x^2 + 2 * x
def f' (x : ℝ) : ℝ := 2 * x + 2
def k (n : ℕ) : ℝ := 2 * n + 2

def Q (k : ℝ) : Prop := ∃ (n : ℕ), k = 2 * n + 2
def R (k : ℝ) : Prop := ∃ (n : ℕ), k = 4 * n + 2

def c (n : ℕ) : ℕ := 12 * n - 6

theorem general_formula_c (n : ℕ) (hn1 : 0 < c 10)
    (hn2 : c 10 < 115) : c n = 12 * n - 6 := sorry

end general_formula_a_general_formula_c_l153_153725


namespace product_of_coprime_numbers_l153_153962

variable {a b c : ℕ}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem product_of_coprime_numbers (h1 : coprime a b) (h2 : a * b = c) : Nat.lcm a b = c := by
  sorry

end product_of_coprime_numbers_l153_153962


namespace class_distances_l153_153707

theorem class_distances (x y z : ℕ) 
  (h1 : y = x + 8)
  (h2 : z = 3 * x)
  (h3 : x + y + z = 108) : 
  x = 20 ∧ y = 28 ∧ z = 60 := 
  by sorry

end class_distances_l153_153707


namespace area_of_sine_triangle_l153_153109

-- We define the problem conditions and the statement we want to prove
theorem area_of_sine_triangle (A B C : Real) (area_ABC : ℝ) (unit_circle : ℝ) :
  unit_circle = 1 → area_ABC = 1 / 2 →
  let a := 2 * Real.sin A
  let b := 2 * Real.sin B
  let c := 2 * Real.sin C
  let s := (a + b + c) / 2
  let area_sine_triangle := 
    (s * (s - a) * (s - b) * (s - c)).sqrt / 4 
  area_sine_triangle = 1 / 8 :=
by
  intros
  sorry -- Proof is left as an exercise

end area_of_sine_triangle_l153_153109


namespace find_lengths_of_segments_l153_153218

variable (b c : ℝ)

theorem find_lengths_of_segments (CK AK AB CT AC AT : ℝ)
  (h1 : CK = AK + AB)
  (h2 : CK = (b + c) / 2)
  (h3 : CT = AC - AT)
  (h4 : AC = b) :
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := 
sorry

end find_lengths_of_segments_l153_153218


namespace evaluate_expression_l153_153080

theorem evaluate_expression (a : ℕ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 :=
by
  -- Here would be the proof which is omitted as per instructions
  sorry

end evaluate_expression_l153_153080


namespace gym_class_students_correct_l153_153415

noncomputable def check_gym_class_studens :=
  let P1 := 15
  let P2 := 5
  let P3 := 12.5
  let P4 := 9.166666666666666
  let P5 := 8.333333333333334
  P1 = P2 + 10 ∧
  P2 = 2 * P3 - 20 ∧
  P3 = P4 + P5 - 5 ∧
  P4 = (1 / 2) * P5 + 5

theorem gym_class_students_correct : check_gym_class_studens := by
  simp [check_gym_class_studens]
  sorry

end gym_class_students_correct_l153_153415


namespace simplify_expr_l153_153172

theorem simplify_expr : 
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = (5 : ℚ) / 4 := 
by
  sorry

end simplify_expr_l153_153172


namespace calc_sqrt_mult_l153_153168

theorem calc_sqrt_mult : 
  ∀ (a b c : ℕ), a = 256 → b = 64 → c = 16 → 
  (nat.sqrt (nat.sqrt a) * nat.cbrt b * nat.sqrt c = 64) :=
by 
  intros a b c h1 h2 h3
  rw [h1, nat.sqrt_eq, nat.sqrt_eq, h2, nat.cbrt_eq, h3, nat.sqrt_eq]
  sorry

end calc_sqrt_mult_l153_153168


namespace solve_inequality_l153_153161

def solution_set_of_inequality : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem solve_inequality (x : ℝ) (h : (2 - x) / (x + 4) > 0) : x ∈ solution_set_of_inequality :=
by
  sorry

end solve_inequality_l153_153161


namespace trajectory_is_ellipse_l153_153224

theorem trajectory_is_ellipse (M : ℝ × ℝ) (F : ℝ × ℝ) (line_y : ℝ) (ratio : ℝ) (hF : F = ⟨0, 2⟩) (hline : line_y = 8) (hratio : ratio = 1 / 2) :
  (dist M F / real.abs (M.snd - line_y) = ratio) → (M.fst ^ 2 / 12 + M.snd ^ 2 / 16 = 1) :=
by
  sorry

end trajectory_is_ellipse_l153_153224


namespace megan_folders_l153_153623

theorem megan_folders (initial_files deleted_files files_per_folder : ℕ) (h1 : initial_files = 237)
    (h2 : deleted_files = 53) (h3 : files_per_folder = 12) :
    let remaining_files := initial_files - deleted_files
    let total_folders := (remaining_files / files_per_folder) + 1
    total_folders = 16 := 
by
  sorry

end megan_folders_l153_153623


namespace difference_of_fractions_l153_153120

theorem difference_of_fractions (p q : ℕ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (5/6) - (1/7) = 29/42 := by
sorrr

end difference_of_fractions_l153_153120


namespace average_score_group2_l153_153328

-- Total number of students
def total_students : ℕ := 50

-- Overall average score
def overall_average_score : ℝ := 92

-- Number of students from 1 to 30
def group1_students : ℕ := 30

-- Average score of students from 1 to 30
def group1_average_score : ℝ := 90

-- Total number of students - group1_students = 50 - 30 = 20
def group2_students : ℕ := total_students - group1_students

-- Lean 4 statement to prove the average score of students with student numbers 31 to 50 is 95
theorem average_score_group2 :
  (overall_average_score * total_students = group1_average_score * group1_students + x * group2_students) →
  x = 95 :=
sorry

end average_score_group2_l153_153328


namespace isabella_non_yellow_houses_l153_153132

variable (Green Yellow Red Blue Pink : ℕ)

axiom h1 : 3 * Yellow = Green
axiom h2 : Red = Yellow + 40
axiom h3 : Green = 90
axiom h4 : Blue = (Green + Yellow) / 2
axiom h5 : Pink = (Red / 2) + 15

theorem isabella_non_yellow_houses : (Green + Red + Blue + Pink - Yellow) = 270 :=
by 
  sorry

end isabella_non_yellow_houses_l153_153132


namespace Carol_rectangle_length_l153_153201

theorem Carol_rectangle_length :
  (∃ (L : ℕ), (L * 15 = 4 * 30) → L = 8) :=
by
  sorry

end Carol_rectangle_length_l153_153201


namespace max_sara_tie_fraction_l153_153912

theorem max_sara_tie_fraction :
  let max_wins := 2 / 5
  let sara_wins := 1 / 4
  let postponed_fraction := 1 / 20
  let total_wins := max_wins + (sara_wins * (5 / 5))
  let non_postponed_fraction := 1 - postponed_fraction
  let win_ratio_non_postponed := total_wins * (20 / 19)
  let tie_fraction := 1 - win_ratio_non_postponed
  in tie_fraction = 6 / 19 :=
by
  sorry

end max_sara_tie_fraction_l153_153912


namespace num_factors_of_2_pow_20_minus_1_l153_153116

/-- 
Prove that the number of positive two-digit integers 
that are factors of \(2^{20} - 1\) is 5.
-/
theorem num_factors_of_2_pow_20_minus_1 :
  ∃ (n : ℕ), n = 5 ∧ (∀ (k : ℕ), k ∣ (2^20 - 1) → 10 ≤ k ∧ k < 100 → k = 33 ∨ k = 15 ∨ k = 27 ∨ k = 41 ∨ k = 45) 
  :=
sorry

end num_factors_of_2_pow_20_minus_1_l153_153116


namespace sin_value_l153_153730

theorem sin_value (α : ℝ) (h: cos (π / 6 - α) = (sqrt 3)/3) :
  sin (5 * π / 6 - 2 * α) = -1 / 3 :=
sorry

end sin_value_l153_153730


namespace distance_between_first_and_last_pots_l153_153212

theorem distance_between_first_and_last_pots (n : ℕ) (d : ℕ) 
  (h₁ : n = 8) 
  (h₂ : d = 100) : 
  ∃ total_distance : ℕ, total_distance = 175 := 
by 
  sorry

end distance_between_first_and_last_pots_l153_153212


namespace fraction_of_price_l153_153188

theorem fraction_of_price (d : ℝ) : d * 0.65 * 0.70 = d * 0.455 :=
by
  sorry

end fraction_of_price_l153_153188


namespace area_of_PQRS_l153_153148

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end area_of_PQRS_l153_153148


namespace simple_interest_time_l153_153160

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem simple_interest_time (SI CI : ℝ) (SI_given CI_given P_simp P_comp r_simp r_comp t_comp : ℝ) :
  SI = CI / 2 →
  CI = compound_interest P_comp r_comp 1 t_comp - P_comp →
  SI = simple_interest P_simp r_simp t_comp →
  P_simp = 1272 →
  r_simp = 0.10 →
  P_comp = 5000 →
  r_comp = 0.12 →
  t_comp = 2 →
  t_comp = 5 :=
by
  intros
  sorry

end simple_interest_time_l153_153160


namespace least_multiple_of_17_gt_450_l153_153975

def least_multiple_gt (n x : ℕ) (k : ℕ) : Prop :=
  k * n > x ∧ ∀ m : ℕ, m * n > x → m ≥ k

theorem least_multiple_of_17_gt_450 : ∃ k : ℕ, least_multiple_gt 17 450 k :=
by
  use 27
  sorry

end least_multiple_of_17_gt_450_l153_153975


namespace find_m_for_min_value_l153_153222

theorem find_m_for_min_value :
  ∃ (m : ℝ), ( ∀ x : ℝ, (y : ℝ) = m * x^2 - 4 * x + 1 → (∃ x_min : ℝ, (∀ x : ℝ, (m * x_min^2 - 4 * x_min + 1 ≤ m * x^2 - 4 * x + 1) → y = -3))) :=
sorry

end find_m_for_min_value_l153_153222


namespace geoff_needed_more_votes_to_win_l153_153910

-- Definitions based on the conditions
def total_votes : ℕ := 6000
def percent_to_fraction (p : ℕ) : ℚ := p / 100
def geoff_percent : ℚ := percent_to_fraction 1
def win_percent : ℚ := percent_to_fraction 51

-- Specific values derived from the conditions
def geoff_votes : ℚ := geoff_percent * total_votes
def win_votes : ℚ := win_percent * total_votes + 1

-- The theorem we intend to prove
theorem geoff_needed_more_votes_to_win :
  (win_votes - geoff_votes) = 3001 := by
  sorry

end geoff_needed_more_votes_to_win_l153_153910


namespace biased_coin_probability_l153_153679

theorem biased_coin_probability :
  let P1 := 3 / 4
  let P2 := 1 / 2
  let P3 := 3 / 4
  let P4 := 2 / 3
  let P5 := 1 / 3
  let P6 := 2 / 5
  let P7 := 3 / 7
  P1 * P2 * P3 * P4 * P5 * P6 * P7 = 3 / 560 :=
by sorry

end biased_coin_probability_l153_153679


namespace C_is_14_years_younger_than_A_l153_153800

variable (A B C D : ℕ)

-- Conditions
axiom cond1 : A + B = (B + C) + 14
axiom cond2 : B + D = (C + A) + 10
axiom cond3 : D = C + 6

-- To prove
theorem C_is_14_years_younger_than_A : A - C = 14 :=
by
  sorry

end C_is_14_years_younger_than_A_l153_153800


namespace age_difference_l153_153820

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a - c = 18 :=
by
  sorry

end age_difference_l153_153820


namespace walt_age_l153_153791

variable (W M P : ℕ)

-- Conditions
def condition1 := M = 3 * W
def condition2 := M + 12 = 2 * (W + 12)
def condition3 := P = 4 * W
def condition4 := P + 15 = 3 * (W + 15)

theorem walt_age (W M P : ℕ) (h1 : condition1 W M) (h2 : condition2 W M) (h3 : condition3 W P) (h4 : condition4 W P) : 
  W = 30 :=
sorry

end walt_age_l153_153791


namespace prob_exceeds_175_l153_153369

-- Definitions from the conditions
def prob_less_than_160 (p : ℝ) : Prop := p = 0.2
def prob_160_to_175 (p : ℝ) : Prop := p = 0.5

-- The mathematical equivalence proof we need
theorem prob_exceeds_175 (p₁ p₂ p₃ : ℝ) 
  (h₁ : prob_less_than_160 p₁) 
  (h₂ : prob_160_to_175 p₂) 
  (H : p₃ = 1 - (p₁ + p₂)) :
  p₃ = 0.3 := 
by
  -- Placeholder for proof
  sorry

end prob_exceeds_175_l153_153369


namespace percent_dimes_value_is_60_l153_153478

variable (nickels dimes : ℕ)
variable (value_nickel value_dime : ℕ)
variable (num_nickels num_dimes : ℕ)

def total_value (n d : ℕ) (v_n v_d : ℕ) := n * v_n + d * v_d

def percent_value_dimes (total d_value : ℕ) := (d_value * 100) / total

theorem percent_dimes_value_is_60 :
  num_nickels = 40 →
  num_dimes = 30 →
  value_nickel = 5 →
  value_dime = 10 →
  percent_value_dimes (total_value num_nickels num_dimes value_nickel value_dime) (num_dimes * value_dime) = 60 := 
by sorry

end percent_dimes_value_is_60_l153_153478


namespace smallest_k_l153_153556

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l153_153556


namespace intersection_sets_m_n_l153_153573

theorem intersection_sets_m_n :
  let M := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
  let N := { x : ℝ | x > 0 }
  M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_m_n_l153_153573


namespace determine_sixth_face_l153_153035

-- Define a cube configuration and corresponding functions
inductive Color
| black
| white

structure Cube where
  faces : Fin 6 → Fin 9 → Color

noncomputable def sixth_face_color (cube : Cube) : Fin 9 → Color := sorry

-- The statement of the theorem proving the coloring of the sixth face
theorem determine_sixth_face (cube : Cube) : 
  (exists f : (Fin 9 → Color), f = sixth_face_color cube) := 
sorry

end determine_sixth_face_l153_153035


namespace M_ends_in_two_zeros_iff_l153_153929

theorem M_ends_in_two_zeros_iff (n : ℕ) (h : n > 0) : 
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 :=
by sorry

end M_ends_in_two_zeros_iff_l153_153929


namespace cost_difference_l153_153006

-- Given conditions
def first_present_cost : ℕ := 18
def third_present_cost : ℕ := first_present_cost - 11
def total_cost : ℕ := 50

-- denoting costs of the second present via variable
def second_present_cost (x : ℕ) : Prop :=
  first_present_cost + x + third_present_cost = total_cost

-- Goal statement
theorem cost_difference (x : ℕ) (h : second_present_cost x) : x - first_present_cost = 7 :=
  sorry

end cost_difference_l153_153006


namespace triangle_cos_C_correct_l153_153604

noncomputable def triangle_cos_C (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : ℝ :=
  Real.cos C -- This will be defined correctly in the proof phase.

theorem triangle_cos_C_correct (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : 
  triangle_cos_C A B C hABC hSinA hCosB = 16 / 65 :=
sorry

end triangle_cos_C_correct_l153_153604


namespace traffic_light_probability_change_l153_153501

theorem traffic_light_probability_change :
  let cycle_time := 100
  let intervals := [(0, 50), (50, 55), (55, 100)]
  let time_changing := [((45, 50), 5), ((50, 55), 5), ((95, 100), 5)]
  let total_change_time := time_changing.map Prod.snd |>.sum
  let probability := (total_change_time : ℚ) / cycle_time
  probability = 3 / 20 := sorry

end traffic_light_probability_change_l153_153501


namespace largest_cube_edge_from_cone_l153_153969

theorem largest_cube_edge_from_cone : 
  ∀ (s : ℝ), 
  (s = 2) → 
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 :=
by
  sorry

end largest_cube_edge_from_cone_l153_153969


namespace calculate_expression_l153_153703

theorem calculate_expression : 1000 * 2.998 * 2.998 * 100 = (29980)^2 := 
by
  sorry

end calculate_expression_l153_153703


namespace point_slope_form_of_perpendicular_line_l153_153379

theorem point_slope_form_of_perpendicular_line :
  ∀ (l1 l2 : ℝ → ℝ) (P : ℝ × ℝ),
    (l2 x = x + 1) →
    (P = (2, 1)) →
    (∀ x, l2 x = -1 * l1 x) →
    (∀ x, l1 x = -x + 3) :=
by
  intros l1 l2 P h1 h2 h3
  sorry

end point_slope_form_of_perpendicular_line_l153_153379


namespace geometric_sequence_common_ratio_l153_153907

noncomputable def common_ratio (a : ℕ → ℝ) (q : ℝ) (positive : ∀ n, a n > 0) : Prop :=
  a 2 + a 1 = a 1 * q^2 ∧ q = (1 + Real.sqrt 5) / 2

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (positive : ∀ n, a n > 0)
  (geometric : ∀ n, a (n + 1) = a n * q)
  (arithmetic : a 2 + a 1 = 2 * (a 3 / 2)) : 
  q = (1 + Real.sqrt 5) / 2 :=
by {
  sorry
}

end geometric_sequence_common_ratio_l153_153907


namespace cookies_per_bag_l153_153479

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (H1 : total_cookies = 703) (H2 : num_bags = 37) : total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l153_153479


namespace eddys_climbing_rate_l153_153241

def base_camp_ft := 5000
def departure_time := 6 -- in hours: 6:00 AM
def hillary_climbing_rate := 800 -- ft/hr
def stopping_distance_ft := 1000 -- ft short of summit
def hillary_descending_rate := 1000 -- ft/hr
def passing_time := 12 -- in hours: 12:00 PM

theorem eddys_climbing_rate :
  ∀ (base_ft departure hillary_rate stop_dist descend_rate pass_time : ℕ),
    base_ft = base_camp_ft →
    departure = departure_time →
    hillary_rate = hillary_climbing_rate →
    stop_dist = stopping_distance_ft →
    descend_rate = hillary_descending_rate →
    pass_time = passing_time →
    (pass_time - departure) * hillary_rate - descend_rate * (pass_time - (departure + (base_ft - stop_dist) / hillary_rate)) = 6 * 500 :=
by
  intros
  sorry

end eddys_climbing_rate_l153_153241


namespace least_four_digit_with_factors_3_5_7_l153_153470

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l153_153470


namespace part_a_part_b_part_c_l153_153761

def is_frameable (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 6

theorem part_a : is_frameable 3 ∧ is_frameable 4 ∧ is_frameable 6 :=
  sorry

theorem part_b (n : ℕ) (h : n ≥ 7) : ¬ is_frameable n :=
  sorry

theorem part_c : ¬ is_frameable 5 :=
  sorry

end part_a_part_b_part_c_l153_153761


namespace determine_x_l153_153743

theorem determine_x (x y : ℝ) (h : x / (x - 1) = (y^3 + 2 * y^2 - 1) / (y^3 + 2 * y^2 - 2)) : 
  x = y^3 + 2 * y^2 - 1 :=
by
  sorry

end determine_x_l153_153743


namespace parabola_vertex_sum_l153_153823

variable (a b c : ℝ)

def parabola_eq (x y : ℝ) : Prop :=
  x = a * y^2 + b * y + c

def vertex (v : ℝ × ℝ) : Prop :=
  v = (-3, 2)

def passes_through (p : ℝ × ℝ) : Prop :=
  p = (-1, 0)

theorem parabola_vertex_sum :
  ∀ (a b c : ℝ),
  (∃ v : ℝ × ℝ, vertex v) ∧
  (∃ p : ℝ × ℝ, passes_through p) →
  a + b + c = -7/2 :=
by
  intros a b c
  intro conditions
  sorry

end parabola_vertex_sum_l153_153823


namespace total_pages_read_l153_153263

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l153_153263


namespace female_managers_count_l153_153905

def E : ℕ -- total number of employees E
def M : ℕ := E - 500 -- number of male employees (M = E - 500)
def total_managers : ℕ := (2/5) * E -- total number of managers ((2/5)E)
def male_managers : ℕ := (2/5) * M -- number of male managers ((2/5)M)
def female_managers : ℕ := total_managers - male_managers -- number of female managers (total_managers - male_managers)
def company_total_managers: E : ℕ → total_managers : ℕ→ female_ubalnce_constraints: female_managers
theorem female_managers_count : female_managers = 200 := sorry

end female_managers_count_l153_153905


namespace support_percentage_correct_l153_153128

-- Define the total number of government employees and the percentage supporting the project
def num_gov_employees : ℕ := 150
def perc_gov_support : ℝ := 0.70

-- Define the total number of citizens and the percentage supporting the project
def num_citizens : ℕ := 800
def perc_citizens_support : ℝ := 0.60

-- Calculate the number of supporters among government employees
def gov_supporters : ℝ := perc_gov_support * num_gov_employees

-- Calculate the number of supporters among citizens
def citizens_supporters : ℝ := perc_citizens_support * num_citizens

-- Calculate the total number of people surveyed and the total number of supporters
def total_surveyed : ℝ := num_gov_employees + num_citizens
def total_supporters : ℝ := gov_supporters + citizens_supporters

-- Define the expected correct answer percentage
def correct_percentage_supporters : ℝ := 61.58

-- Prove that the percentage of overall supporters is equal to the expected correct percentage 
theorem support_percentage_correct :
  (total_supporters / total_surveyed * 100) = correct_percentage_supporters :=
by
  sorry

end support_percentage_correct_l153_153128


namespace exists_ratios_eq_l153_153424

theorem exists_ratios_eq (a b z : ℕ) (ha : 0 < a) (hb : 0 < b) (hz : 0 < z) (h : a * b = z^2 + 1) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (a : ℚ) / b = (x^2 + 1) / (y^2 + 1) :=
by
  sorry

end exists_ratios_eq_l153_153424


namespace ratio_is_five_to_one_l153_153645

noncomputable def ratio_of_numbers (A B : ℕ) : ℚ :=
  A / B

theorem ratio_is_five_to_one (A B : ℕ) (hA : A = 20) (hLCM : Nat.lcm A B = 80) : ratio_of_numbers A B = 5 := by
  -- Proof omitted
  sorry

end ratio_is_five_to_one_l153_153645


namespace c_share_l153_153674

theorem c_share (A B C : ℕ) (h1 : A + B + C = 364) (h2 : A = B / 2) (h3 : B = C / 2) : 
  C = 208 := by
  -- Proof omitted
  sorry

end c_share_l153_153674


namespace translation_result_l153_153597

-- Define the initial point A
def A : (ℤ × ℤ) := (-2, 3)

-- Define the translation function
def translate (p : (ℤ × ℤ)) (delta_x delta_y : ℤ) : (ℤ × ℤ) :=
  (p.1 + delta_x, p.2 - delta_y)

-- The theorem stating the resulting point after translation
theorem translation_result :
  translate A 3 1 = (1, 2) :=
by
  -- Skipping proof with sorry
  sorry

end translation_result_l153_153597


namespace derivative_at_one_l153_153951

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_at_one : deriv f 1 = -1 := sorry

end derivative_at_one_l153_153951


namespace more_boys_than_girls_l153_153760

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l153_153760


namespace total_amount_spent_l153_153281

theorem total_amount_spent (half_dollar_value : ℝ) (wednesday_spend : ℕ) (thursday_spend : ℕ) : 
  wednesday_spend = 4 → thursday_spend = 14 → half_dollar_value = 0.5 → (wednesday_spend + thursday_spend) * half_dollar_value = 9 :=
by
  intros wednesday_cond thursday_cond half_dollar_cond
  rw [wednesday_cond, thursday_cond, half_dollar_cond]
  norm_num
  sorry

end total_amount_spent_l153_153281


namespace Brenda_bakes_cakes_l153_153698

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l153_153698


namespace max_b_c_value_l153_153179

theorem max_b_c_value (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c - b = 2) : b + c = 18 :=
sorry

end max_b_c_value_l153_153179


namespace expected_revenue_day_14_plan_1_more_reasonable_plan_l153_153342

-- Define the initial conditions
def initial_valuation : ℕ := 60000
def rain_probability : ℚ := 0.4
def no_rain_probability : ℚ := 0.6
def hiring_cost : ℕ := 32000

-- Calculate the expected revenue if Plan ① is adopted
def expected_revenue_plan_1_day_14 : ℚ :=
  (initial_valuation / 10000) * (1/2 * rain_probability + no_rain_probability)

-- Calculate the total revenue for Plan ①
def total_revenue_plan_1 : ℚ :=
  (initial_valuation / 10000) + 2 * expected_revenue_plan_1_day_14

-- Calculate the total revenue for Plan ②
def total_revenue_plan_2 : ℚ :=
  3 * (initial_valuation / 10000) - (hiring_cost / 10000)

-- Define the lemmas to prove
theorem expected_revenue_day_14_plan_1 :
  expected_revenue_plan_1_day_14 = 4.8 := 
  by sorry

theorem more_reasonable_plan :
  total_revenue_plan_1 > total_revenue_plan_2 :=
  by sorry

end expected_revenue_day_14_plan_1_more_reasonable_plan_l153_153342


namespace smallest_k_l153_153558

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l153_153558


namespace find_abc_l153_153583

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) : 
  abc_value a b c = 762 :=
sorry

end find_abc_l153_153583


namespace cubic_of_cubic_roots_correct_l153_153159

variable (a b c : ℝ) (α β γ : ℝ)

-- Vieta's formulas conditions
axiom vieta1 : α + β + γ = -a
axiom vieta2 : α * β + β * γ + γ * α = b
axiom vieta3 : α * β * γ = -c

-- Define the polynomial whose roots are α³, β³, and γ³
def cubic_of_cubic_roots (x : ℝ) : ℝ :=
  x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3

-- Prove that this polynomial has α³, β³, γ³ as roots
theorem cubic_of_cubic_roots_correct :
  ∀ x : ℝ, cubic_of_cubic_roots a b c x = 0 ↔ (x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
sorry

end cubic_of_cubic_roots_correct_l153_153159


namespace expression_equals_36_l153_153585

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l153_153585


namespace find_value_of_N_l153_153719

theorem find_value_of_N :
  (2 * ((3.6 * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) :=
by {
  sorry
}

end find_value_of_N_l153_153719


namespace find_y_l153_153364

theorem find_y (x y : ℤ) (h1 : x^2 - 5 * x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end find_y_l153_153364


namespace lcm_subtract100_correct_l153_153974

noncomputable def lcm1364_884_subtract_100 : ℕ :=
  let a := 1364
  let b := 884
  let lcm_ab := Nat.lcm a b
  lcm_ab - 100

theorem lcm_subtract100_correct : lcm1364_884_subtract_100 = 1509692 := by
  sorry

end lcm_subtract100_correct_l153_153974


namespace tangent_line_circle_p_l153_153448

theorem tangent_line_circle_p (p : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x + 8 = 0 → (x = -p/2 ∨ y = 0)) → 
  (p = 4 ∨ p = 8) :=
by
  sorry

end tangent_line_circle_p_l153_153448


namespace sum_of_reflected_coordinates_l153_153147

noncomputable def sum_of_coordinates (C D : ℝ × ℝ) : ℝ :=
  C.1 + C.2 + D.1 + D.2

theorem sum_of_reflected_coordinates (y : ℝ) :
  let C := (3, y)
  let D := (3, -y)
  sum_of_coordinates C D = 6 :=
by
  sorry

end sum_of_reflected_coordinates_l153_153147


namespace time_for_2km_l153_153788

def distance_over_time (t : ℕ) : ℝ := 
  sorry -- Function representing the distance walked over time

theorem time_for_2km : ∃ t : ℕ, distance_over_time t = 2 ∧ t = 105 :=
by
  sorry

end time_for_2km_l153_153788


namespace probability_sequence_correct_l153_153658

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l153_153658


namespace field_day_difference_l153_153757

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l153_153757


namespace four_machines_save_11_hours_l153_153181

-- Define the conditions
def three_machines_complete_order_in_44_hours := 3 * (1 / (3 * 44)) * 44 = 1

def additional_machine_reduces_time (T : ℝ) := 4 * (1 / (3 * 44)) * T = 1

-- Define the theorem to prove the number of hours saved
theorem four_machines_save_11_hours : 
  (∃ T : ℝ, additional_machine_reduces_time T ∧ three_machines_complete_order_in_44_hours) → 
  44 - 33 = 11 :=
by
  sorry

end four_machines_save_11_hours_l153_153181


namespace boat_ratio_l153_153056

theorem boat_ratio (b c d1 d2 : ℝ) 
  (h1 : b = 20) 
  (h2 : c = 4) 
  (h3 : d1 = 4) 
  (h4 : d2 = 2) : 
  (d1 + d2) / ((d1 / (b + c)) + (d2 / (b - c))) / b = 36 / 35 :=
by 
  sorry

end boat_ratio_l153_153056


namespace least_four_digit_multiple_3_5_7_l153_153471

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l153_153471


namespace percentage_increase_is_20_l153_153755

def number_of_students_this_year : ℕ := 960
def number_of_students_last_year : ℕ := 800

theorem percentage_increase_is_20 :
  ((number_of_students_this_year - number_of_students_last_year : ℕ) / number_of_students_last_year * 100) = 20 := 
by
  sorry

end percentage_increase_is_20_l153_153755


namespace smallest_x_l153_153946

theorem smallest_x (x : ℚ) (h : 7 * (4 * x^2 + 4 * x + 5) = x * (4 * x - 35)) : 
  x = -5/3 ∨ x = -7/8 := by
  sorry

end smallest_x_l153_153946


namespace find_solutions_l153_153716

theorem find_solutions (x y z : ℝ) :
  (x = 5 / 3 ∧ y = -4 / 3 ∧ z = -4 / 3) ∨
  (x = 4 / 3 ∧ y = 4 / 3 ∧ z = -5 / 3) →
  (x^2 - y * z = abs (y - z) + 1) ∧ 
  (y^2 - z * x = abs (z - x) + 1) ∧ 
  (z^2 - x * y = abs (x - y) + 1) :=
by
  sorry

end find_solutions_l153_153716


namespace n_in_S_implies_n_squared_in_S_l153_153273

-- Definition of the set S
def S : Set ℕ := {n | ∃ a b c d e f : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ 
                      n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2}

-- The proof goal
theorem n_in_S_implies_n_squared_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S :=
by
  sorry

end n_in_S_implies_n_squared_in_S_l153_153273


namespace fruit_basket_combinations_l153_153242

theorem fruit_basket_combinations :
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples+1) * (oranges+1) * (bananas+1)
  let empty_basket := 1
  total_combinations - empty_basket = 159 :=
by
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples + 1) * (oranges + 1) * (bananas + 1)
  let empty_basket := 1
  have h_total_combinations : total_combinations = 4 * 8 * 5 := by sorry
  have h_empty_basket : empty_basket = 1 := by sorry
  have h_subtract : 4 * 8 * 5 - 1 = 159 := by sorry
  exact h_subtract

end fruit_basket_combinations_l153_153242


namespace simplified_expression_correct_l153_153945

def simplify_expression (x : ℝ) : ℝ :=
  4 * (x ^ 2 - 5 * x) - 5 * (2 * x ^ 2 + 3 * x)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = -6 * x ^ 2 - 35 * x :=
by
  sorry

end simplified_expression_correct_l153_153945


namespace rankings_are_correct_l153_153141

-- Define teams:
inductive Team
| A | B | C | D

-- Define the type for ranking
structure Ranking :=
  (first : Team)
  (second : Team)
  (third : Team)
  (last : Team)

-- Define the predictions of Jia, Yi, and Bing
structure Predictions := 
  (Jia : Ranking)
  (Yi : Ranking)
  (Bing : Ranking)

-- Define the condition that each prediction is half right, half wrong
def isHalfRightHalfWrong (pred : Ranking) (actual : Ranking) : Prop :=
  (pred.first = actual.first ∨ pred.second = actual.second ∨ pred.third = actual.third ∨ pred.last = actual.last) ∧
  (pred.first ≠ actual.first ∨ pred.second ≠ actual.second ∨ pred.third ≠ actual.third ∨ pred.last ≠ actual.last)

-- Define the actual rankings
def actualRanking : Ranking := { first := Team.C, second := Team.A, third := Team.D, last := Team.B }

-- Define Jia's Predictions 
def JiaPrediction : Ranking := { first := Team.C, second := Team.C, third := Team.D, last := Team.D }

-- Define Yi's Predictions 
def YiPrediction : Ranking := { first := Team.B, second := Team.A, third := Team.C, last := Team.D }

-- Define Bing's Predictions 
def BingPrediction : Ranking := { first := Team.C, second := Team.B, third := Team.A, last := Team.D }

-- Create an instance of predictions
def pred : Predictions := { Jia := JiaPrediction, Yi := YiPrediction, Bing := BingPrediction }

-- The theorem to be proved
theorem rankings_are_correct :
  isHalfRightHalfWrong pred.Jia actualRanking ∧ 
  isHalfRightHalfWrong pred.Yi actualRanking ∧ 
  isHalfRightHalfWrong pred.Bing actualRanking →
  actualRanking.first = Team.C ∧ actualRanking.second = Team.A ∧ actualRanking.third = Team.D ∧ 
  actualRanking.last = Team.B :=
by
  sorry -- Proof is not required.

end rankings_are_correct_l153_153141


namespace min_value_fx_range_a_seq_inequality_l153_153738

open Real

-- Definitions corresponding to the conditions
def f (x : ℝ) (a : ℝ) : ℝ := ln x + 1/x + a * x

-- Proof problem 1
theorem min_value_fx (a : ℝ) (h1 : ∃ (x : ℝ), x > 0 ∧ f x a = 1) :
  ∃ x, x > 0 ∧ f x a = 1 → f 1 a = 1 :=
by
  sorry

-- Proof problem 2
theorem range_a (h2 : ∃ x ∈ Ioo 2 3, deriv (f x a) = 0) :
  a ∈ Ioo (-1 / 4) (-2 / 9) :=
by
  sorry

-- Proof problem 3
theorem seq_inequality (x : ℕ → ℝ) (hx : ∀ n, ln (x n) + 1 / (x (n + 1)) < 1) :
  x 1 ≤ 1 :=
by
  sorry

end min_value_fx_range_a_seq_inequality_l153_153738


namespace elasticity_ratio_l153_153844

theorem elasticity_ratio (e_QN e_PN : ℝ) (h1 : e_QN = 1.27) (h2 : e_PN = 0.76) : 
  (e_QN / e_PN) ≈ 1.7 :=
by
  rw [h1, h2]
  -- prove the statement using the given conditions
  sorry

end elasticity_ratio_l153_153844


namespace smallest_k_for_divisibility_l153_153546

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l153_153546


namespace limit_for_regular_pay_l153_153331

theorem limit_for_regular_pay 
  (x : ℕ) 
  (regular_pay_rate : ℕ := 3) 
  (overtime_pay_rate : ℕ := 6) 
  (total_pay : ℕ := 186) 
  (overtime_hours : ℕ := 11) 
  (H : 3 * x + (6 * 11) = 186) 
  :
  x = 40 :=
sorry

end limit_for_regular_pay_l153_153331


namespace cricket_run_rate_l153_153986

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target_runs : ℝ) (first_overs : ℝ) (remaining_overs : ℝ):
  run_rate_first_10_overs = 6.2 → 
  target_runs = 282 →
  first_overs = 10 →
  remaining_overs = 40 →
  (target_runs - run_rate_first_10_overs * first_overs) / remaining_overs = 5.5 :=
by
  intros h1 h2 h3 h4
  -- Insert proof here
  sorry

end cricket_run_rate_l153_153986


namespace value_of_a_l153_153231

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l153_153231


namespace prob_both_standard_prob_only_one_standard_l153_153192

-- Given conditions
axiom prob_A1 : ℝ
axiom prob_A2 : ℝ
axiom prob_A1_std : prob_A1 = 0.95
axiom prob_A2_std : prob_A2 = 0.95
axiom prob_not_A1 : ℝ
axiom prob_not_A2 : ℝ
axiom prob_not_A1_std : prob_not_A1 = 0.05
axiom prob_not_A2_std : prob_not_A2 = 0.05
axiom independent_A1_A2 : prob_A1 * prob_A2 = prob_A1 * prob_A2

-- Definitions of events
def event_A1 := true -- Event that the first product is standard
def event_A2 := true -- Event that the second product is standard
def event_not_A1 := not event_A1
def event_not_A2 := not event_A2

-- Proof problems
theorem prob_both_standard :
  prob_A1 * prob_A2 = 0.9025 := by sorry

theorem prob_only_one_standard :
  (prob_A1 * prob_not_A2) + (prob_not_A1 * prob_A2) = 0.095 := by sorry

end prob_both_standard_prob_only_one_standard_l153_153192


namespace professionals_work_days_l153_153926

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end professionals_work_days_l153_153926


namespace rational_inequality_solution_l153_153362

theorem rational_inequality_solution (x : ℝ) (h : x ≠ 4) :
  (4 < x ∧ x ≤ 5) ↔ (x - 2) / (x - 4) ≤ 3 :=
sorry

end rational_inequality_solution_l153_153362


namespace ordered_triples_eq_l153_153531

theorem ordered_triples_eq :
  ∃! (x y z : ℤ), x + y = 4 ∧ xy - z^2 = 3 ∧ (x = 2 ∧ y = 2 ∧ z = 0) :=
by
  -- Proof goes here
  sorry

end ordered_triples_eq_l153_153531


namespace smaller_of_two_integers_l153_153027

noncomputable def smaller_integer (m n : ℕ) : ℕ :=
if m < n then m else n

theorem smaller_of_two_integers :
  ∀ (m n : ℕ),
  100 ≤ m ∧ m < 1000 ∧ 100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200 →
  smaller_integer m n = 891 :=
by
  intros m n h
  -- Assuming m, n are positive three-digit integers and satisfy the condition
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  sorry

end smaller_of_two_integers_l153_153027


namespace suraj_innings_l153_153782

theorem suraj_innings (n A : ℕ) (h1 : A + 6 = 16) (h2 : (n * A + 112) / (n + 1) = 16) : n = 16 :=
by
  sorry

end suraj_innings_l153_153782


namespace solution_l153_153633

noncomputable def problem (x : ℝ) : Prop :=
  2021 * (x ^ (2020/202)) - 1 = 2020 * x

theorem solution (x : ℝ) (hx : x ≥ 0) : problem x → x = 1 := 
begin
  sorry
end

end solution_l153_153633


namespace find_integers_l153_153861

theorem find_integers (x : ℤ) (h₁ : x ≠ 3) (h₂ : (x - 3) ∣ (x ^ 3 - 3)) :
  x = -21 ∨ x = -9 ∨ x = -5 ∨ x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 ∨
  x = 7 ∨ x = 9 ∨ x = 11 ∨ x = 15 ∨ x = 27 :=
sorry

end find_integers_l153_153861


namespace hexadecagon_area_l153_153680

theorem hexadecagon_area (r : ℝ) : 
  let θ := (360 / 16 : ℝ)
  let A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180)
  let total_area := 16 * A_triangle
  3 * r^2 = total_area :=
by
  sorry

end hexadecagon_area_l153_153680


namespace sum_of_nonneg_real_numbers_inequality_l153_153618

open BigOperators

variables {α : Type*} [LinearOrderedField α]

theorem sum_of_nonneg_real_numbers_inequality 
  (a : ℕ → α) (n : ℕ)
  (h_nonneg : ∀ i : ℕ, 0 ≤ a i) : 
  (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j) * (∑ j in Finset.Icc i (n - 1), a j ^ 2))) 
  ≤ (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j)) ^ 2) :=
sorry

end sum_of_nonneg_real_numbers_inequality_l153_153618


namespace card_sequence_probability_l153_153656

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l153_153656


namespace money_problem_l153_153833

theorem money_problem
  (A B C : ℕ)
  (h1 : A + B + C = 450)
  (h2 : B + C = 350)
  (h3 : C = 100) :
  A + C = 200 :=
by
  sorry

end money_problem_l153_153833


namespace can_transform_1220_to_2012_cannot_transform_1220_to_2021_l153_153133

def can_transform (abcd : ℕ) (wxyz : ℕ) : Prop :=
  ∀ a b c d w x y z, 
  abcd = a*1000 + b*100 + c*10 + d ∧ 
  wxyz = w*1000 + x*100 + y*10 + z →
  (∃ (k : ℕ) (m : ℕ), 
    (k = a ∧ a ≠ d  ∧ m = c  ∧ c ≠ w ∧ 
     w = b + (k - b) ∧ x = c + (m - c)) ∨
    (k = w ∧ w ≠ x  ∧ m = y  ∧ y ≠ z ∧ 
     z = a + (k - a) ∧ x = d + (m - d)))
          
theorem can_transform_1220_to_2012 : can_transform 1220 2012 :=
sorry

theorem cannot_transform_1220_to_2021 : ¬ can_transform 1220 2021 :=
sorry

end can_transform_1220_to_2012_cannot_transform_1220_to_2021_l153_153133


namespace find_a_f_greater_than_1_l153_153878

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) := x^2 * Real.exp x - a * Real.log x

-- Condition: Slope at x = 1 is 3e - 1
theorem find_a (a : ℝ) (h : deriv (fun x => f x a) 1 = 3 * Real.exp 1 - 1) : a = 1 := sorry

-- Given a = 1
theorem f_greater_than_1 (x : ℝ) (hx : x > 0) : f x 1 > 1 := sorry

end find_a_f_greater_than_1_l153_153878


namespace vanessa_points_l153_153909

theorem vanessa_points (total_points : ℕ) (num_other_players : ℕ) (avg_points_other : ℕ) 
  (h1 : total_points = 65) (h2 : num_other_players = 7) (h3 : avg_points_other = 5) :
  ∃ vp : ℕ, vp = 30 :=
by
  sorry

end vanessa_points_l153_153909


namespace sin_value_l153_153729

theorem sin_value (α : ℝ) (h : Real.cos (π / 6 - α) = (Real.sqrt 3) / 3) :
    Real.sin (5 * π / 6 - 2 * α) = -1 / 3 :=
by
  sorry

end sin_value_l153_153729


namespace grey_eyed_black_haired_students_l153_153250

theorem grey_eyed_black_haired_students (total_students black_haired green_eyed_red_haired grey_eyed : ℕ) 
(h_total : total_students = 60) 
(h_black_haired : black_haired = 35) 
(h_green_eyed_red_haired : green_eyed_red_haired = 20) 
(h_grey_eyed : grey_eyed = 25) : 
grey_eyed - (total_students - black_haired - green_eyed_red_haired) = 20 :=
by
  sorry

end grey_eyed_black_haired_students_l153_153250


namespace value_of_expression_l153_153691

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1) 
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 12) 
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 123) 
  : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 = 334 :=
by
  sorry

end value_of_expression_l153_153691


namespace probability_age_less_than_20_l153_153606

theorem probability_age_less_than_20 (total : ℕ) (ages_gt_30 : ℕ) (ages_lt_20 : ℕ) 
    (h1 : total = 150) (h2 : ages_gt_30 = 90) (h3 : ages_lt_20 = total - ages_gt_30) :
    (ages_lt_20 : ℚ) / total = 2 / 5 :=
by
  simp [h1, h2, h3]
  sorry

end probability_age_less_than_20_l153_153606


namespace find_divisor_l153_153600

theorem find_divisor (n k : ℤ) (h1 : n % 30 = 16) : (2 * n) % 30 = 2 :=
by
  sorry

end find_divisor_l153_153600


namespace Brenda_bakes_cakes_l153_153697

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l153_153697


namespace dennis_years_of_teaching_l153_153464

variable (V A D E N : ℕ)

def combined_years_taught : Prop :=
  V + A + D + E + N = 225

def virginia_adrienne_relation : Prop :=
  V = A + 9

def virginia_dennis_relation : Prop :=
  V = D - 15

def elijah_adrienne_relation : Prop :=
  E = A - 3

def elijah_nadine_relation : Prop :=
  E = N + 7

theorem dennis_years_of_teaching 
  (h1 : combined_years_taught V A D E N) 
  (h2 : virginia_adrienne_relation V A)
  (h3 : virginia_dennis_relation V D)
  (h4 : elijah_adrienne_relation E A) 
  (h5 : elijah_nadine_relation E N) : 
  D = 65 :=
  sorry

end dennis_years_of_teaching_l153_153464


namespace troll_problem_l153_153857

theorem troll_problem (T : ℕ) (h : 6 + T + T / 2 = 33) : 4 * 6 - T = 6 :=
by sorry

end troll_problem_l153_153857


namespace probability_of_spade_then_king_l153_153165

theorem probability_of_spade_then_king :
  ( (24 / 104) * (8 / 103) + (2 / 104) * (7 / 103) ) = 103 / 5356 :=
sorry

end probability_of_spade_then_king_l153_153165


namespace geometric_sequence_root_product_l153_153252

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l153_153252


namespace percentage_of_uninsured_part_time_l153_153256

noncomputable def number_of_employees := 330
noncomputable def uninsured_employees := 104
noncomputable def part_time_employees := 54
noncomputable def probability_neither := 0.5606060606060606

theorem percentage_of_uninsured_part_time:
  (13 / 104) * 100 = 12.5 := 
by 
  -- Here you can assume proof steps would occur/assertions to align with the solution found
  sorry

end percentage_of_uninsured_part_time_l153_153256


namespace total_valid_votes_l153_153818

theorem total_valid_votes (V : ℕ) (h1 : 0.70 * (V: ℝ) - 0.30 * (V: ℝ) = 184) : V = 460 :=
by sorry

end total_valid_votes_l153_153818


namespace total_tickets_l153_153834

theorem total_tickets (A C total_tickets total_cost : ℕ) 
  (adult_ticket_cost : ℕ := 8) (child_ticket_cost : ℕ := 5) 
  (total_cost_paid : ℕ := 201) (child_tickets_count : ℕ := 21) 
  (ticket_cost_eqn : 8 * A + 5 * 21 = 201) 
  (adult_tickets_count : A = total_cost_paid - (child_ticket_cost * child_tickets_count) / adult_ticket_cost) :
  total_tickets = A + child_tickets_count :=
sorry

end total_tickets_l153_153834


namespace fruit_prob_l153_153608

variable (O A B S : ℕ) 

-- Define the conditions
variables (H1 : O + A + B + S = 32)
variables (H2 : O - 5 = 3)
variables (H3 : A - 3 = 7)
variables (H4 : S - 2 = 4)
variables (H5 : 3 + 7 + 4 + B = 20)

-- Define the proof problem
theorem fruit_prob :
  (O = 8) ∧ (A = 10) ∧ (B = 6) ∧ (S = 6) → (O + S) / (O + A + B + S) = 7 / 16 := 
by
  sorry

end fruit_prob_l153_153608


namespace average_speed_round_trip_l153_153622

noncomputable def average_speed (d : ℝ) (v_to v_from : ℝ) : ℝ :=
  let time_to := d / v_to
  let time_from := d / v_from
  let total_time := time_to + time_from
  let total_distance := 2 * d
  total_distance / total_time

theorem average_speed_round_trip (d : ℝ) :
  average_speed d 60 40 = 48 :=
by
  sorry

end average_speed_round_trip_l153_153622


namespace viewers_difference_l153_153398

theorem viewers_difference :
  let second_game := 80
  let first_game := second_game - 20
  let third_game := second_game + 15
  let fourth_game := third_game + (third_game / 10)
  let total_last_week := 350
  let total_this_week := first_game + second_game + third_game + fourth_game
  total_this_week - total_last_week = -10 := 
by
  sorry

end viewers_difference_l153_153398


namespace volleyball_practice_start_time_l153_153949

def homework_time := 1 * 60 + 59  -- convert 1:59 p.m. to minutes since 12:00 p.m.
def homework_duration := 96        -- duration in minutes
def buffer_time := 25              -- time between finishing homework and practice
def practice_start_time := 4 * 60  -- convert 4:00 p.m. to minutes since 12:00 p.m.

theorem volleyball_practice_start_time :
  homework_time + homework_duration + buffer_time = practice_start_time := 
by
  sorry

end volleyball_practice_start_time_l153_153949


namespace inverse_g_neg1_l153_153934

noncomputable def g (c d x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_neg1 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, g c d y = -1 ∧ y = (-1 - d) / c := 
by
  unfold g
  sorry

end inverse_g_neg1_l153_153934


namespace select_people_english_japanese_l153_153189

-- Definitions based on conditions
def total_people : ℕ := 9
def english_speakers : ℕ := 7
def japanese_speakers : ℕ := 3

-- Theorem statement
theorem select_people_english_japanese (h1 : total_people = 9) 
                                      (h2 : english_speakers = 7) 
                                      (h3 : japanese_speakers = 3) :
  ∃ n, n = 20 :=
by {
  sorry
}

end select_people_english_japanese_l153_153189


namespace cafeteria_extra_fruits_l153_153649

theorem cafeteria_extra_fruits (red_apples green_apples bananas oranges students : ℕ) (fruits_per_student : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : bananas = 17)
  (h4 : oranges = 12)
  (h5 : students = 21)
  (h6 : fruits_per_student = 2) :
  (red_apples + green_apples + bananas + oranges) - (students * fruits_per_student) = 43 :=
by
  sorry

end cafeteria_extra_fruits_l153_153649


namespace average_and_fourth_number_l153_153641

theorem average_and_fourth_number {x : ℝ} (h_avg : ((1 + 2 + 4 + 6 + 9 + 9 + 10 + 12 + x) / 9) = 7) :
  x = 10 ∧ 6 = 6 :=
by
  sorry

end average_and_fourth_number_l153_153641


namespace arithmetic_sequence_formula_l153_153756

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (d : ℤ) :
  (a 3 = 4) → (d = -2) → ∀ n : ℕ, a n = 10 - 2 * n :=
by
  intros h1 h2 n
  sorry

end arithmetic_sequence_formula_l153_153756


namespace twenty_fifty_yuan_bills_unique_l153_153831

noncomputable def twenty_fifty_yuan_bills (x y : ℕ) : Prop :=
  x + y = 260 ∧ 20 * x + 50 * y = 100 * 100

theorem twenty_fifty_yuan_bills_unique (x y : ℕ) (h : twenty_fifty_yuan_bills x y) :
  x = 100 ∧ y = 160 :=
by
  sorry

end twenty_fifty_yuan_bills_unique_l153_153831


namespace least_multiple_25_gt_500_l153_153311

theorem least_multiple_25_gt_500 : ∃ (k : ℕ), 25 * k > 500 ∧ (∀ m : ℕ, (25 * m > 500 → 25 * k ≤ 25 * m)) :=
by
  use 21
  sorry

end least_multiple_25_gt_500_l153_153311


namespace binary_mul_1101_111_eq_1001111_l153_153535

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l153_153535


namespace conic_section_is_ellipse_l153_153518

theorem conic_section_is_ellipse (x y : ℝ) : 
  (x - 3)^2 + 9 * (y + 2)^2 = 144 →
  (∃ h k a b : ℝ, a = 12 ∧ b = 4 ∧ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
by
  intro h_eq
  use 3, -2, 12, 4
  constructor
  { sorry }
  constructor
  { sorry }
  sorry

end conic_section_is_ellipse_l153_153518


namespace optimal_fruit_combination_l153_153002

structure FruitPrices :=
  (price_2_apples : ℕ)
  (price_6_apples : ℕ)
  (price_12_apples : ℕ)
  (price_2_oranges : ℕ)
  (price_6_oranges : ℕ)
  (price_12_oranges : ℕ)

def minCostFruits : ℕ :=
  sorry

theorem optimal_fruit_combination (fp : FruitPrices) (total_fruits : ℕ)
  (mult_2_or_3 : total_fruits = 15) :
  fp.price_2_apples = 48 →
  fp.price_6_apples = 126 →
  fp.price_12_apples = 224 →
  fp.price_2_oranges = 60 →
  fp.price_6_oranges = 164 →
  fp.price_12_oranges = 300 →
  minCostFruits = 314 :=
by
  sorry

end optimal_fruit_combination_l153_153002


namespace piecewise_function_not_composed_of_multiple_functions_l153_153211

theorem piecewise_function_not_composed_of_multiple_functions :
  ∀ (f : ℝ → ℝ), (∃ (I : ℝ → Prop) (f₁ f₂ : ℝ → ℝ),
    (∀ x, I x → f x = f₁ x) ∧ (∀ x, ¬I x → f x = f₂ x)) →
    ¬(∃ (g₁ g₂ : ℝ → ℝ), (∀ x, f x = g₁ x ∨ f x = g₂ x)) :=
by
  sorry

end piecewise_function_not_composed_of_multiple_functions_l153_153211


namespace temperature_notation_l153_153580

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end temperature_notation_l153_153580


namespace jumping_contest_l153_153446

theorem jumping_contest (grasshopper_jump frog_jump : ℕ) (h_grasshopper : grasshopper_jump = 9) (h_frog : frog_jump = 12) : frog_jump - grasshopper_jump = 3 := by
  ----- h_grasshopper and h_frog are our conditions -----
  ----- The goal is to prove frog_jump - grasshopper_jump = 3 -----
  sorry

end jumping_contest_l153_153446


namespace parallel_vectors_x_value_l153_153731

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define the condition that vectors are parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- State the problem: if a and b are parallel, then x = 1/2
theorem parallel_vectors_x_value (x : ℝ) (h : is_parallel a (b x)) : x = 1/2 :=
by
  sorry

end parallel_vectors_x_value_l153_153731


namespace parabola_and_hyperbola_focus_equal_l153_153599

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) :=
(2, 0)

noncomputable def hyperbola_focus : (ℝ × ℝ) :=
(2, 0)

theorem parabola_and_hyperbola_focus_equal
  (p : ℝ)
  (h_parabola : parabola_focus p = (2, 0))
  (h_hyperbola : hyperbola_focus = (2, 0)) :
  p = 4 := by
  sorry

end parabola_and_hyperbola_focus_equal_l153_153599


namespace probability_of_letters_l153_153503

theorem probability_of_letters (total_cards alex_letters jamie_letters : ℕ) :
  total_cards = 12 →
  alex_letters = 4 →
  jamie_letters = 8 →
  (∃ (prob : ℚ), prob = (Nat.choose alex_letters 2 * Nat.choose jamie_letters 1 : ℚ) / (Nat.choose total_cards 3) ∧ prob = 12 / 55) :=
by
  intros h_total h_alex h_jamie
  use (Nat.choose alex_letters 2 * Nat.choose jamie_letters 1 : ℚ) / (Nat.choose total_cards 3)
  split
  · sorry -- Placeholder for the actual calculation which isn't needed in the statement
  · sorry -- Placeholder for the verification which isn't needed in the statement

end probability_of_letters_l153_153503


namespace least_four_digit_multiple_3_5_7_l153_153473

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l153_153473


namespace subtraction_and_multiplication_problem_l153_153812

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem_l153_153812


namespace sum_of_numbers_l153_153917

theorem sum_of_numbers (a b : ℕ) (h : a + 4 * b = 30) : a + b = 12 :=
sorry

end sum_of_numbers_l153_153917


namespace james_out_of_pocket_cost_l153_153000

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l153_153000


namespace sum_of_20th_and_30th_triangular_numbers_l153_153357

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_20th_and_30th_triangular_numbers :
  triangular_number 20 + triangular_number 30 = 675 :=
by
  sorry

end sum_of_20th_and_30th_triangular_numbers_l153_153357


namespace simplest_form_fraction_l153_153316

theorem simplest_form_fraction 
  (m n a : ℤ) (h_f1 : (2 * m) / (10 * m * n) = 1 / (5 * n))
  (h_f2 : (m^2 - n^2) / (m + n) = (m - n))
  (h_f3 : (2 * a) / (a^2) = 2 / a) : 
  ∀ (f : ℤ), f = (m^2 + n^2) / (m + n) → 
    (∀ (k : ℤ), k ≠ 1 → (m^2 + n^2) / (m + n) ≠ k * f) :=
by
  intros f h_eq k h_kneq1
  sorry

end simplest_form_fraction_l153_153316


namespace max_cars_div_10_l153_153421

noncomputable def max_cars (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) : ℕ :=
  let k := 2000
  2000 -- Maximum number of cars passing the sensor

theorem max_cars_div_10 (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) :
  car_length = 5 →
  (∀ k : ℕ, distance_for_speed k = k) →
  (∀ k : ℕ, speed k = 10 * k) →
  (max_cars car_length distance_for_speed speed) = 2000 → 
  (max_cars car_length distance_for_speed speed) / 10 = 200 := by
  intros
  sorry

end max_cars_div_10_l153_153421


namespace polynomials_equal_at_all_x_l153_153274

variable {R : Type} [CommRing R]

def f (a_5 a_4 a_3 a_2 a_1 a_0 : R) (x : R) := a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def g (b_3 b_2 b_1 b_0 : R) (x : R) := b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def h (c_2 c_1 c_0 : R) (x : R) := c_2 * x^2 + c_1 * x + c_0

theorem polynomials_equal_at_all_x 
    (a_5 a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 c_2 c_1 c_0 : ℤ)
    (bound_a : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
    (bound_b : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
    (bound_c : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
    (H : f a_5 a_4 a_3 a_2 a_1 a_0 10 = g b_3 b_2 b_1 b_0 10 * h c_2 c_1 c_0 10) :
    ∀ x, f a_5 a_4 a_3 a_2 a_1 a_0 x = g b_3 b_2 b_1 b_0 x * h c_2 c_1 c_0 x := by
  sorry

end polynomials_equal_at_all_x_l153_153274


namespace probability_two_cards_l153_153041

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l153_153041


namespace min_a_n_l153_153236

def a_n (n : ℕ) : ℤ := n^2 - 8 * n + 5

theorem min_a_n : ∃ n : ℕ, ∀ m : ℕ, a_n n ≤ a_n m ∧ a_n n = -11 :=
by
  sorry

end min_a_n_l153_153236


namespace percentage_of_green_ducks_smaller_pond_l153_153908

-- Definitions of the conditions
def num_ducks_smaller_pond : ℕ := 30
def num_ducks_larger_pond : ℕ := 50
def percentage_green_larger_pond : ℕ := 12
def percentage_green_total : ℕ := 15
def total_ducks : ℕ := num_ducks_smaller_pond + num_ducks_larger_pond

-- Calculation of the number of green ducks
def num_green_larger_pond := percentage_green_larger_pond * num_ducks_larger_pond / 100
def num_green_total := percentage_green_total * total_ducks / 100

-- Define the percentage of green ducks in the smaller pond
def percentage_green_smaller_pond (x : ℕ) :=
  x * num_ducks_smaller_pond / 100 + num_green_larger_pond = num_green_total

-- The theorem to be proven
theorem percentage_of_green_ducks_smaller_pond : percentage_green_smaller_pond 20 :=
  sorry

end percentage_of_green_ducks_smaller_pond_l153_153908


namespace determine_a_l153_153727

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l153_153727


namespace average_of_u_l153_153107

theorem average_of_u :
  (∃ u : ℕ, ∀ r1 r2 : ℕ, (r1 + r2 = 6) ∧ (r1 * r2 = u) → r1 > 0 ∧ r2 > 0) →
  (∃ distinct_u : Finset ℕ, distinct_u = {5, 8, 9} ∧ (distinct_u.sum / distinct_u.card) = 22 / 3) :=
sorry

end average_of_u_l153_153107


namespace volleyball_last_place_score_l153_153257

theorem volleyball_last_place_score (n : ℕ) (h : n ≥ 2) 
  (points : Fin n → ℕ) 
  (h_arith_prog : ∃ a d, ∀ i : Fin n, points i = a + (i : ℕ) * d)
  (h_total_points : ∑ i in Finset.finRange n, points ⟨i, sorry⟩ = n * (n - 1) / 2) :
  ∃ i : Fin n, points i = 0 :=
by 
  use 0
  sorry

end volleyball_last_place_score_l153_153257


namespace no_consecutive_positive_integers_with_no_real_solutions_l153_153356

theorem no_consecutive_positive_integers_with_no_real_solutions :
  ∀ b c : ℕ, (c = b + 1) → (b^2 - 4 * c < 0) → (c^2 - 4 * b < 0) → false :=
by
  intro b c
  sorry

end no_consecutive_positive_integers_with_no_real_solutions_l153_153356


namespace large_hexagon_toothpicks_l153_153493

theorem large_hexagon_toothpicks (n : Nat) (h : n = 1001) : 
  let T_half := (n * (n + 1)) / 2
  let T_total := 2 * T_half + n
  let boundary_toothpicks := 6 * T_half
  let total_toothpicks := 3 * T_total - boundary_toothpicks
  total_toothpicks = 3006003 :=
by
  sorry

end large_hexagon_toothpicks_l153_153493


namespace cannot_achieve_90_cents_l153_153982

theorem cannot_achieve_90_cents :
  ∀ (p n d q : ℕ),        -- p: number of pennies, n: number of nickels, d: number of dimes, q: number of quarters
  (p + n + d + q = 6) →   -- exactly six coins chosen
  (p ≤ 4 ∧ n ≤ 4 ∧ d ≤ 4 ∧ q ≤ 4) →  -- no more than four of each kind of coin
  (p + 5 * n + 10 * d + 25 * q ≠ 90) -- total value should not equal 90 cents
:= by
  sorry

end cannot_achieve_90_cents_l153_153982


namespace paul_has_five_dogs_l153_153626

theorem paul_has_five_dogs
  (w1 w2 w3 w4 w5 : ℕ)
  (food_per_10_pounds : ℕ)
  (total_food_required : ℕ)
  (h1 : w1 = 20)
  (h2 : w2 = 40)
  (h3 : w3 = 10)
  (h4 : w4 = 30)
  (h5 : w5 = 50)
  (h6 : food_per_10_pounds = 1)
  (h7 : total_food_required = 15) :
  (w1 / 10 * food_per_10_pounds) +
  (w2 / 10 * food_per_10_pounds) +
  (w3 / 10 * food_per_10_pounds) +
  (w4 / 10 * food_per_10_pounds) +
  (w5 / 10 * food_per_10_pounds) = total_food_required → 
  5 = 5 :=
by
  intros
  sorry

end paul_has_five_dogs_l153_153626


namespace number_is_two_l153_153247

theorem number_is_two 
  (N : ℝ)
  (h1 : N = 4 * 1 / 2)
  (h2 : (1 / 2) * N = 1) :
  N = 2 :=
sorry

end number_is_two_l153_153247


namespace sum_at_simple_interest_l153_153682

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end sum_at_simple_interest_l153_153682


namespace probability_of_specific_sequence_l153_153661

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l153_153661


namespace radical_product_is_64_l153_153170

theorem radical_product_is_64:
  real.sqrt (16:ℝ) * real.sqrt (real.sqrt 256) * real.n_root 64 3 = 64 :=
sorry

end radical_product_is_64_l153_153170


namespace geometric_sequence_product_l153_153255

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l153_153255


namespace victor_percentage_of_marks_l153_153809

theorem victor_percentage_of_marks (marks_obtained : ℝ) (maximum_marks : ℝ) (h1 : marks_obtained = 285) (h2 : maximum_marks = 300) : 
  (marks_obtained / maximum_marks) * 100 = 95 :=
by
  sorry

end victor_percentage_of_marks_l153_153809


namespace perfect_square_condition_l153_153215

def is_perfect_square (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

noncomputable def a_n (n : ℕ) : ℤ := (10^n - 1) / 9

theorem perfect_square_condition (n b : ℕ) (h1 : 0 < b) (h2 : b < 10) :
  is_perfect_square ((a_n (2 * n)) - b * (a_n n)) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) := by
  sorry

end perfect_square_condition_l153_153215


namespace north_east_paths_no_cross_red_l153_153577

theorem north_east_paths_no_cross_red : 
  let to_column (m n : ℕ) := choose (m + n) m  -- Number of paths to (m, n) in grid
                      /- Paths to critical points C and D with steps constraints -/
  let paths_through_C := to_column 7 1 * to_column 7 1
  let paths_through_D := to_column 7 3 * to_column 7 3
  let total_paths := paths_through_C + paths_through_D
  total_paths = 1274 := 
by 
  sorry -- This line is just a placeholder to indicate the proof is skipped

end north_east_paths_no_cross_red_l153_153577


namespace circle_center_radius_l153_153644

/-
Given:
- The endpoints of a diameter are (2, -3) and (-8, 7).

Prove:
- The center of the circle is (-3, 2).
- The radius of the circle is 5√2.
-/

noncomputable def center_and_radius (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let Cx := ((A.1 + B.1) / 2)
  let Cy := ((A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - Cx) * (A.1 - Cx) + (A.2 - Cy) * (A.2 - Cy))
  (Cx, Cy, radius)

theorem circle_center_radius :
  center_and_radius (2, -3) (-8, 7) = (-3, 2, 5 * Real.sqrt 2) :=
by
  sorry

end circle_center_radius_l153_153644


namespace smallest_k_divides_l153_153550

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l153_153550


namespace trapezium_other_side_length_l153_153866

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l153_153866


namespace remainder_div1_remainder_div2_l153_153312

open Polynomial

noncomputable def polynomial1 := X^1001 - 1
noncomputable def divisor1 := X^4 + X^3 + 2 * X^2 + X + 1
noncomputable def divisor2 := X^8 + X^6 + 2 * X^4 + X^2 + 1

theorem remainder_div1 :
  (polynomial1 % divisor1) = X^2 * (1 - X) :=
sorry

theorem remainder_div2 :
  (polynomial1 % divisor2) = -2 * X^7 - X^5 - 2 * X^3 - 1 :=
sorry

end remainder_div1_remainder_div2_l153_153312


namespace problem_7_sqrt_13_l153_153233

theorem problem_7_sqrt_13 : 
  let m := Int.floor (Real.sqrt 13)
  let n := 10 - Real.sqrt 13 - Int.floor (10 - Real.sqrt 13)
  m + n = 7 - Real.sqrt 13 :=
by
  sorry

end problem_7_sqrt_13_l153_153233


namespace candle_height_relation_l153_153806

theorem candle_height_relation : 
  ∀ (h : ℝ) (t : ℝ), h = 1 → (∀ (h1_burn_rate : ℝ), h1_burn_rate = 1 / 5) → (∀ (h2_burn_rate : ℝ), h2_burn_rate = 1 / 6) →
  (1 - t * 1 / 5 = 3 * (1 - t * 1 / 6)) → t = 20 / 3 :=
by
  intros h t h_init h1_burn_rate h2_burn_rate height_eq
  sorry

end candle_height_relation_l153_153806


namespace largest_prime_divisor_of_sum_of_squares_l153_153363

def a : ℕ := 35
def b : ℕ := 84

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p = 13 ∧ (a^2 + b^2) % p = 0 := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l153_153363


namespace temperature_decrease_l153_153601

-- Define the conditions
def temperature_rise (temp_increase: ℤ) : ℤ := temp_increase

-- Define the claim to be proved
theorem temperature_decrease (temp_decrease: ℤ) : temperature_rise 3 = 3 → temperature_rise (-6) = -6 :=
by
  sorry

end temperature_decrease_l153_153601


namespace playground_children_count_l153_153324

theorem playground_children_count (boys girls : ℕ) (h_boys : boys = 27) (h_girls : girls = 35) : boys + girls = 62 := by
  sorry

end playground_children_count_l153_153324


namespace coordinates_of_point_A_l153_153110

def f (x : ℝ) : ℝ := x^2 + 3 * x

theorem coordinates_of_point_A (a : ℝ) (b : ℝ) 
    (slope_condition : deriv f a = 7) 
    (point_condition : f a = b) : 
    a = 2 ∧ b = 10 := 
by {
    sorry
}

end coordinates_of_point_A_l153_153110


namespace sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l153_153485

theorem sin_and_tan_alpha_in_second_quadrant 
  (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (hcos : Real.cos α = -8 / 17) :
  Real.sin α = 15 / 17 ∧ Real.tan α = -15 / 8 := 
  sorry

theorem expression_value_for_given_tan 
  (α : ℝ) (htan : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := 
  sorry

end sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l153_153485


namespace expression_equals_36_l153_153587

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l153_153587


namespace f_has_exactly_one_zero_point_a_range_condition_l153_153568

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * Real.log x + 2 / (x + 1)

theorem f_has_exactly_one_zero_point :
  ∃! x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ f x = 0 := sorry

theorem a_range_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) 1 → ∀ t : ℝ, t ∈ Set.Icc (1 / 2) 2 → f x ≥ t^3 - t^2 - 2 * a * t + 2) → a ≥ 5 / 4 := sorry

end f_has_exactly_one_zero_point_a_range_condition_l153_153568


namespace union_of_sets_l153_153888

def M : Set Int := { -1, 0, 1 }
def N : Set Int := { 0, 1, 2 }

theorem union_of_sets : M ∪ N = { -1, 0, 1, 2 } := by
  sorry

end union_of_sets_l153_153888


namespace problem_equivalent_proof_l153_153523

theorem problem_equivalent_proof (a : ℝ) (h : a / 2 - 2 / a = 5) :
  (a^8 - 256) / (16 * a^4) * (2 * a / (a^2 + 4)) = 81 :=
sorry

end problem_equivalent_proof_l153_153523


namespace polynomial_remainder_l153_153976

theorem polynomial_remainder (x : ℤ) : (x + 1) ∣ (x^15 + 1) ↔ x = -1 := sorry

end polynomial_remainder_l153_153976


namespace inequality_holds_l153_153582

theorem inequality_holds (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_holds_l153_153582


namespace card_sequence_probability_l153_153657

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l153_153657


namespace inequality_example_l153_153366

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a ^ 2 + 8 * b * c)) + (b / Real.sqrt (b ^ 2 + 8 * c * a)) + (c / Real.sqrt (c ^ 2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_example_l153_153366


namespace gcd_g50_g52_l153_153935

def g (x : ℤ) := x^2 - 2*x + 2022

theorem gcd_g50_g52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end gcd_g50_g52_l153_153935


namespace lowest_value_meter_can_record_l153_153185

theorem lowest_value_meter_can_record (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 6) (h2 : A = 2) : A = 2 :=
by sorry

end lowest_value_meter_can_record_l153_153185


namespace find_r_s_l153_153135

def N : Matrix (Fin 2) (Fin 2) Int := ![![3, 4], ![-2, 0]]
def I : Matrix (Fin 2) (Fin 2) Int := ![![1, 0], ![0, 1]]

theorem find_r_s :
  ∃ (r s : Int), (N * N = r • N + s • I) ∧ (r = 3) ∧ (s = 16) :=
by
  sorry

end find_r_s_l153_153135


namespace automobile_travel_distance_l153_153506

variable (a r : ℝ)

theorem automobile_travel_distance (h : r ≠ 0) :
  (a / 4) * (240 / 1) * (1 / (3 * r)) = (20 * a) / r := 
by
  sorry

end automobile_travel_distance_l153_153506


namespace field_day_difference_l153_153758

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l153_153758


namespace vasya_can_construct_polyhedron_l153_153664

-- Definition of a polyhedron using given set of shapes
-- where the original set of shapes can form a polyhedron
def original_set_can_form_polyhedron (squares triangles : ℕ) : Prop :=
  squares = 1 ∧ triangles = 4

-- Transformation condition: replacing 2 triangles with 2 squares
def replacement_condition (initial_squares initial_triangles replaced_squares replaced_triangles : ℕ) : Prop :=
  initial_squares + 2 = replaced_squares ∧ initial_triangles - 2 = replaced_triangles

-- Proving that new set of shapes can form a polyhedron
theorem vasya_can_construct_polyhedron :
  ∃ (new_squares new_triangles : ℕ),
    (original_set_can_form_polyhedron 1 4)
    ∧ (replacement_condition 1 4 new_squares new_triangles)
    ∧ (new_squares = 3 ∧ new_triangles = 2) :=
by
  sorry

end vasya_can_construct_polyhedron_l153_153664


namespace welders_correct_l153_153635

-- Define the initial number of welders
def initial_welders := 12

-- Define the conditions:
-- 1. Total work is 1 job that welders can finish in 3 days.
-- 2. 9 welders leave after the first day.
-- 3. The remaining work is completed by (initial_welders - 9) in 8 days.

theorem welders_correct (W : ℕ) (h1 : W * 1/3 = 1) (h2 : (W - 9) * 8 = 2 * W) : 
  W = initial_welders :=
by
  sorry

end welders_correct_l153_153635


namespace fraction_planted_of_field_is_correct_l153_153214

/-- Given a right triangle with legs 5 units and 12 units, and a small unplanted square S
at the right-angle vertex such that the shortest distance from S to the hypotenuse is 3 units,
prove that the fraction of the field that is planted is 52761/857430. -/
theorem fraction_planted_of_field_is_correct :
  let area_triangle := (5 * 12) / 2
  let area_square := (180 / 169) ^ 2
  let area_planted := area_triangle - area_square
  let fraction_planted := area_planted / area_triangle
  fraction_planted = 52761 / 857430 :=
sorry

end fraction_planted_of_field_is_correct_l153_153214


namespace multiples_of_8_has_highest_avg_l153_153670

def average_of_multiples (m : ℕ) (a b : ℕ) : ℕ :=
(a + b) / 2

def multiples_of_7_avg := average_of_multiples 7 7 196 -- 101.5
def multiples_of_2_avg := average_of_multiples 2 2 200 -- 101
def multiples_of_8_avg := average_of_multiples 8 8 200 -- 104
def multiples_of_5_avg := average_of_multiples 5 5 200 -- 102.5
def multiples_of_9_avg := average_of_multiples 9 9 189 -- 99

theorem multiples_of_8_has_highest_avg :
  multiples_of_8_avg > multiples_of_7_avg ∧
  multiples_of_8_avg > multiples_of_2_avg ∧
  multiples_of_8_avg > multiples_of_5_avg ∧
  multiples_of_8_avg > multiples_of_9_avg :=
by
  sorry

end multiples_of_8_has_highest_avg_l153_153670


namespace complement_intersection_l153_153877

def U : Set ℤ := {1, 2, 3, 4, 5}
def P : Set ℤ := {2, 4}
def Q : Set ℤ := {1, 3, 4, 6}
def C_U_P : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_intersection :
  (C_U_P ∩ Q) = {1, 3} :=
by sorry

end complement_intersection_l153_153877


namespace piravena_total_round_trip_cost_l153_153850

noncomputable def piravena_round_trip_cost : ℝ :=
  let distance_AB := 4000
  let bus_cost_per_km := 0.20
  let flight_cost_per_km := 0.12
  let flight_booking_fee := 120
  let flight_cost := distance_AB * flight_cost_per_km + flight_booking_fee
  let bus_cost := distance_AB * bus_cost_per_km
  flight_cost + bus_cost

theorem piravena_total_round_trip_cost : piravena_round_trip_cost = 1400 := by
  -- Problem conditions for reference:
  -- distance_AC = 3000
  -- distance_AB = 4000
  -- bus_cost_per_km = 0.20
  -- flight_cost_per_km = 0.12
  -- flight_booking_fee = 120
  -- Piravena decides to fly from A to B but returns by bus
  sorry

end piravena_total_round_trip_cost_l153_153850


namespace probability_one_card_per_suit_l153_153387

theorem probability_one_card_per_suit :
  let total_cards := 52
  let total_suits := 4
  let total_draws := 4
  let first_card_prob := 1
  let second_card_prob := (13 / (total_cards - 1))
  let third_card_prob := (13 / (total_cards - 2 - 1))
  let fourth_card_prob := (13 / (total_cards - 3 - 1))
  in (first_card_prob * second_card_prob * third_card_prob * fourth_card_prob) = (2197 / 20825) :=
by 
  sorry

end probability_one_card_per_suit_l153_153387


namespace num_divisors_m2_less_than_m_not_divide_m_l153_153275

namespace MathProof

def m : ℕ := 2^20 * 3^15 * 5^6

theorem num_divisors_m2_less_than_m_not_divide_m :
  let m2 := m ^ 2
  let total_divisors_m2 := 41 * 31 * 13
  let total_divisors_m := 21 * 16 * 7
  let divisors_m2_less_than_m := (total_divisors_m2 - 1) / 2
  divisors_m2_less_than_m - total_divisors_m = 5924 :=
by sorry

end MathProof

end num_divisors_m2_less_than_m_not_divide_m_l153_153275


namespace fraction_to_terminating_decimal_l153_153086

theorem fraction_to_terminating_decimal :
  (53 : ℚ)/160 = 0.33125 :=
by sorry

end fraction_to_terminating_decimal_l153_153086


namespace coffee_tea_soda_l153_153017

theorem coffee_tea_soda (Pcoffee Ptea Psoda Pboth_no_soda : ℝ)
  (H1 : 0.9 = Pcoffee)
  (H2 : 0.8 = Ptea)
  (H3 : 0.7 = Psoda) :
  0.0 = Pboth_no_soda :=
  sorry

end coffee_tea_soda_l153_153017


namespace repeating_decimal_to_fraction_l153_153858

theorem repeating_decimal_to_fraction :
  (0.3 + 0.206) = (5057 / 9990) :=
sorry

end repeating_decimal_to_fraction_l153_153858


namespace no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l153_153675

-- Part (1): Prove that there do not exist positive integers m and n such that m(m+2) = n(n+1)
theorem no_solutions_m_m_plus_2_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) :=
sorry

-- Part (2): Given k ≥ 3,
-- Case (a): Prove that for k=3, there do not exist positive integers m and n such that m(m+3) = n(n+1)
theorem no_solutions_m_m_plus_3_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 3) = n * (n + 1) :=
sorry

-- Case (b): Prove that for k ≥ 4, there exist positive integers m and n such that m(m+k) = n(n+1)
theorem solutions_exist_m_m_plus_k_eq_n_n_plus_1 (k : ℕ) (h : k ≥ 4) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1) :=
sorry

end no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l153_153675


namespace binary_rep_of_21_l153_153706

theorem binary_rep_of_21 : 
  (Nat.digits 2 21) = [1, 0, 1, 0, 1] := 
by 
  sorry

end binary_rep_of_21_l153_153706


namespace complement_B_intersection_A_complement_B_l153_153239

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x < 0}
noncomputable def B : Set ℝ := {x | x > 1}

theorem complement_B :
  (U \ B) = {x | x ≤ 1} := by
  sorry

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | x < 0} := by
  sorry

end complement_B_intersection_A_complement_B_l153_153239


namespace average_interest_rate_l153_153683

theorem average_interest_rate (total_investment : ℝ) (rate1 rate2 : ℝ) (annual_return1 annual_return2 : ℝ) 
  (h1 : total_investment = 6000) 
  (h2 : rate1 = 0.035) 
  (h3 : rate2 = 0.055) 
  (h4 : annual_return1 = annual_return2) :
  (annual_return1 + annual_return2) / total_investment * 100 = 4.3 :=
by
  sorry

end average_interest_rate_l153_153683


namespace certain_event_l153_153346

-- Definitions of the events
def event1 : Prop := ∀ (P : ℝ), P ≠ 20.0
def event2 : Prop := ∀ (x : ℤ), x ≠ 105 ∧ x ≤ 100
def event3 : Prop := ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ ¬(r = 0 ∨ r = 1)
def event4 (a b : ℝ) : Prop := ∃ (area : ℝ), area = a * b

-- Statement to prove that event4 is the only certain event
theorem certain_event (a b : ℝ) : (event4 a b) := 
by
  sorry

end certain_event_l153_153346


namespace minimum_value_of_z_l153_153115

theorem minimum_value_of_z 
  (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : 2 * x - y - 2 ≤ 0) 
  (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x + y ∧ z = -6 :=
sorry

end minimum_value_of_z_l153_153115


namespace ratio_6_3_to_percent_l153_153819

theorem ratio_6_3_to_percent : (6 / 3) * 100 = 200 := by
  sorry

end ratio_6_3_to_percent_l153_153819


namespace symmetric_points_l153_153900

-- Let points P and Q be symmetric about the origin
variables (m n : ℤ)
axiom symmetry_condition : (m, 4) = (- (-2), -n)

theorem symmetric_points :
  m = 2 ∧ n = -4 := 
  by {
    sorry
  }

end symmetric_points_l153_153900


namespace arithmetic_seq_first_term_l153_153409

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (a : ℚ) (n : ℕ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
  (h2 : ∀ n, S (4 * n) / S n = 16) : a = 5 / 2 := 
sorry

end arithmetic_seq_first_term_l153_153409


namespace probability_sequence_correct_l153_153660

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l153_153660


namespace find_sum_of_squares_l153_153412

theorem find_sum_of_squares (x y : ℝ) (h1: x * y = 16) (h2: x^2 + y^2 = 34) : (x + y) ^ 2 = 66 :=
by sorry

end find_sum_of_squares_l153_153412


namespace binomial_cubes_sum_l153_153720

theorem binomial_cubes_sum (x y : ℤ) :
  let B1 := x^4 + 9 * x * y^3
  let B2 := -(3 * x^3 * y) - 9 * y^4
  (B1 ^ 3 + B2 ^ 3 = x ^ 12 - 729 * y ^ 12) := by
  sorry

end binomial_cubes_sum_l153_153720


namespace find_value_of_x_l153_153154

theorem find_value_of_x (x y z : ℤ) (h1 : x > y) (h2 : y > z) (h3 : z = 3)
  (h4 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h5 : (x = y + 1) ∧ (y = z + 1)) :
  x = 5 := 
sorry

end find_value_of_x_l153_153154


namespace additional_pass_combinations_l153_153456

def original_combinations : ℕ := 4 * 2 * 3 * 3
def new_combinations : ℕ := 6 * 2 * 4 * 3
def additional_combinations : ℕ := new_combinations - original_combinations

theorem additional_pass_combinations : additional_combinations = 72 := by
  sorry

end additional_pass_combinations_l153_153456


namespace prob1_prob2_prob3_l153_153737

-- Problem 1
theorem prob1 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2)
  (tangent_line_slope : ℝ) (perpendicular_line_eq : ℝ) :
  (tangent_line_slope = 1 + m) →
  (perpendicular_line_eq = -1/2) →
  (tangent_line_slope * perpendicular_line_eq = -1) →
  m = 1 := sorry

-- Problem 2
theorem prob2 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2) :
  (∀ x, f x ≤ m * x^2 + (m - 1) * x - 1) →
  ∃ (m_ : ℤ), m_ ≥ 2 := sorry

-- Problem 3
theorem prob3 (f : ℝ → ℝ) (F : ℝ → ℝ) (x1 x2 : ℝ) (m : ℝ) 
  (f_def : ∀ x, f x = Real.log x + (1/2) * x^2)
  (F_def : ∀ x, F x = f x + x)
  (hx1 : 0 < x1) (hx2: 0 < x2) :
  m = 1 →
  F x1 = -F x2 →
  x1 + x2 ≥ Real.sqrt 3 - 1 := sorry

end prob1_prob2_prob3_l153_153737


namespace unique_solution_of_abc_l153_153881

theorem unique_solution_of_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_lt_ab_c : a < b) (h_lt_b_c: b < c) (h_eq_abc : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 :=
by {
  -- Proof skipped, only the statement is provided.
  sorry
}

end unique_solution_of_abc_l153_153881


namespace domain_of_lg_abs_x_minus_1_l153_153787

theorem domain_of_lg_abs_x_minus_1 (x : ℝ) : 
  (|x| - 1 > 0) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end domain_of_lg_abs_x_minus_1_l153_153787


namespace smallest_possible_norm_l153_153266

-- Defining the vector \begin{pmatrix} -2 \\ 4 \end{pmatrix}
def vec_a : ℝ × ℝ := (-2, 4)

-- Condition: the norm of \mathbf{v} + \begin{pmatrix} -2 \\ 4 \end{pmatrix} = 10
def satisfies_condition (v : ℝ × ℝ) : Prop :=
  (Real.sqrt ((v.1 + vec_a.1) ^ 2 + (v.2 + vec_a.2) ^ 2)) = 10

noncomputable def smallest_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_possible_norm (v : ℝ × ℝ) (h : satisfies_condition v) : smallest_norm v = 10 - 2 * Real.sqrt 5 := by
  sorry

end smallest_possible_norm_l153_153266


namespace atomic_weight_of_iodine_is_correct_l153_153718

noncomputable def atomic_weight_iodine (atomic_weight_nitrogen : ℝ) (atomic_weight_hydrogen : ℝ) (molecular_weight_compound : ℝ) : ℝ :=
  molecular_weight_compound - (atomic_weight_nitrogen + 4 * atomic_weight_hydrogen)

theorem atomic_weight_of_iodine_is_correct :
  atomic_weight_iodine 14.01 1.008 145 = 126.958 :=
by
  unfold atomic_weight_iodine
  norm_num

end atomic_weight_of_iodine_is_correct_l153_153718


namespace option_a_option_b_option_c_l153_153227

open Real

-- Define the functions f and g
variable {f g : ℝ → ℝ}

-- Given conditions
axiom cond1 : ∀ x : ℝ, f(x + 3) = g(-x) + 4
axiom cond2 : ∀ x : ℝ, deriv f x + deriv g (1 + x) = 0
axiom cond3 : ∀ x : ℝ, g(2*x + 1) = g(-(2*x) + 1)

-- Prove the statements
theorem option_a : deriv g 1 = 0 :=
sorry

theorem option_b : ∀ x : ℝ, f(x + 4) = f(4 - x) :=
sorry

theorem option_c : ∀ x : ℝ, deriv f (x + 1) = deriv f (1 - x) :=
sorry

end option_a_option_b_option_c_l153_153227


namespace sum_first_1000_b_n_l153_153301

noncomputable def a_n : ℕ → ℕ
| 1 := 1
| n := n

def b_n (n : ℕ) : ℕ :=
  ⌊ log (a_n n) ⌋₊

theorem sum_first_1000_b_n :
  ∑ n in Finset.range 1000, b_n (n + 1) = 1893 :=
by {
  sorry
}

end sum_first_1000_b_n_l153_153301


namespace probability_at_5_5_equals_1_over_243_l153_153333

-- Define the base probability function P
def P : ℕ → ℕ → ℚ
| 0, 0       => 1
| x+1, 0     => 0
| 0, y+1     => 0
| x+1, y+1   => (1/3 : ℚ) * P x (y+1) + (1/3 : ℚ) * P (x+1) y + (1/3 : ℚ) * P x y

-- Theorem statement that needs to be proved
theorem probability_at_5_5_equals_1_over_243 : P 5 5 = 1 / 243 :=
sorry

end probability_at_5_5_equals_1_over_243_l153_153333


namespace number_of_children_l153_153358

def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 16

theorem number_of_children : total_pencils / pencils_per_child = 8 :=
by
  sorry

end number_of_children_l153_153358


namespace solve_price_per_litre_second_oil_l153_153822

variable (P : ℝ)

def price_per_litre_second_oil :=
  10 * 55 + 5 * P = 15 * 58.67

theorem solve_price_per_litre_second_oil (h : price_per_litre_second_oil P) : P = 66.01 :=
  by
  sorry

end solve_price_per_litre_second_oil_l153_153822


namespace units_digit_of_subtraction_is_seven_l153_153647

theorem units_digit_of_subtraction_is_seven (a b c: ℕ) (h1: a = c + 3) (h2: b = 2 * c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  result % 10 = 7 :=
by
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  sorry

end units_digit_of_subtraction_is_seven_l153_153647


namespace range_of_m_l153_153389

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m^2 * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ -2 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l153_153389


namespace Zelda_probability_success_l153_153981

variable (P : ℝ → ℝ)
variable (X Y Z : ℝ)

theorem Zelda_probability_success :
  P X = 1/3 ∧ P Y = 1/2 ∧ (P X) * (P Y) * (1 - P Z) = 0.0625 → P Z = 0.625 :=
by
  sorry

end Zelda_probability_success_l153_153981


namespace smallest_integer_m_l153_153668

theorem smallest_integer_m (m : ℕ) : m > 1 ∧ m % 13 = 2 ∧ m % 5 = 2 ∧ m % 3 = 2 → m = 197 := 
by 
  sorry

end smallest_integer_m_l153_153668


namespace day_of_18th_day_of_month_is_tuesday_l153_153433

theorem day_of_18th_day_of_month_is_tuesday
  (day_of_24th_is_monday : ℕ → ℕ)
  (mod_seven : ∀ n, n % 7 = n)
  (h24 : day_of_24th_is_monday 24 = 1) : day_of_24th_is_monday 18 = 2 :=
by
  sorry

end day_of_18th_day_of_month_is_tuesday_l153_153433


namespace total_cost_alex_had_to_pay_l153_153183

def baseCost : ℝ := 30
def costPerText : ℝ := 0.04 -- 4 cents in dollars
def textsSent : ℕ := 150
def costPerMinuteOverLimit : ℝ := 0.15 -- 15 cents in dollars
def hoursUsed : ℝ := 26
def freeHours : ℝ := 25

def totalCost : ℝ :=
  baseCost + (costPerText * textsSent) + (costPerMinuteOverLimit * (hoursUsed - freeHours) * 60)

theorem total_cost_alex_had_to_pay :
  totalCost = 45 := by
  sorry

end total_cost_alex_had_to_pay_l153_153183


namespace smallest_k_divides_l153_153555

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l153_153555


namespace quadratic_residues_count_l153_153754

theorem quadratic_residues_count (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  ∃ (q_residues : Finset (ZMod p)), q_residues.card = (p - 1) / 2 ∧
  ∃ (nq_residues : Finset (ZMod p)), nq_residues.card = (p - 1) / 2 ∧
  ∀ d ∈ q_residues, ∃ x y : ZMod p, x^2 = d ∧ y^2 = d ∧ x ≠ y :=
by
  sorry

end quadratic_residues_count_l153_153754


namespace smallest_k_l153_153541

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l153_153541


namespace pyramid_volume_inequality_l153_153797

theorem pyramid_volume_inequality
  (k : ℝ)
  (OA1 OB1 OC1 OA2 OB2 OC2 OA3 OB3 OC3 OB2 : ℝ)
  (V1 := k * |OA1| * |OB1| * |OC1|)
  (V2 := k * |OA2| * |OB2| * |OC2|)
  (V3 := k * |OA3| * |OB3| * |OC3|)
  (V := k * |OA1| * |OB2| * |OC3|) :
  V ≤ (V1 + V2 + V3) / 3 := 
  sorry

end pyramid_volume_inequality_l153_153797


namespace syllogism_major_minor_premise_l153_153609

theorem syllogism_major_minor_premise
(people_of_Yaan_strong_unyielding : Prop)
(people_of_Yaan_Chinese : Prop)
(all_Chinese_strong_unyielding : Prop) :
  all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese → (all_Chinese_strong_unyielding = all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese = people_of_Yaan_Chinese) :=
by
  intros h
  exact ⟨rfl, rfl⟩

end syllogism_major_minor_premise_l153_153609


namespace compare_volumes_l153_153037

variables (a b c : ℝ) -- dimensions of the second block
variables (a1 b1 c1 : ℝ) -- dimensions of the first block

-- Length, width, and height conditions
def length_cond := a1 = 1.5 * a
def width_cond := b1 = 0.8 * b
def height_cond := c1 = 0.7 * c

-- Volumes of the blocks
def V1 := a1 * b1 * c1 -- Volume of the first block
def V2 := a * b * c -- Volume of the second block

-- Main theorem
theorem compare_volumes (h1 : length_cond) (h2 : width_cond) (h3 : height_cond) :
  V2 = (25/21) * V1 :=
sorry

end compare_volumes_l153_153037


namespace brenda_cakes_l153_153699

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l153_153699


namespace envelopes_initial_count_l153_153848

noncomputable def initialEnvelopes (given_per_friend : ℕ) (friends : ℕ) (left : ℕ) : ℕ :=
  given_per_friend * friends + left

theorem envelopes_initial_count
  (given_per_friend : ℕ) (friends : ℕ) (left : ℕ)
  (h_given_per_friend : given_per_friend = 3)
  (h_friends : friends = 5)
  (h_left : left = 22) :
  initialEnvelopes given_per_friend friends left = 37 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end envelopes_initial_count_l153_153848


namespace angle_C_is_pi_div_6_f_range_l153_153751

-- Defining the problem in Lean

/- Given conditions -/
variables {A B C : ℝ} {a b c : ℝ}
variable h1 : Real.tan A = -3 * Real.tan B
variable h2 : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * b

/- Problem (1): Proving the measure of angle C -/
theorem angle_C_is_pi_div_6 
  (h1 : Real.tan A = -3 * Real.tan B)
  (h2 : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * b) :
  C = Real.pi / 6 := 
sorry

/- Problem (2): Finding the range of function f(x) -/
def f (x : ℝ) : ℝ := Real.sin (x + 2 * Real.pi / 3) + Real.cos (x + Real.pi / 6) ^ 2

theorem f_range {x : ℝ} (hx : x ∈ set.Icc 0 (5 * Real.pi / 6)) :
  ∀ y ∈ set.Icc 0 (5 * Real.pi / 6), f y ∈ set.Icc (-(1 : ℝ)) ((3 * Real.sqrt 3 + 2) / 4) :=
sorry

end angle_C_is_pi_div_6_f_range_l153_153751


namespace january_1_is_monday_l153_153605

theorem january_1_is_monday
  (days_in_january : ℕ)
  (mondays_in_january : ℕ)
  (thursdays_in_january : ℕ) :
  days_in_january = 31 ∧ mondays_in_january = 5 ∧ thursdays_in_january = 5 → 
  ∃ d : ℕ, d = 1 ∧ (d % 7 = 1) :=
by
  sorry

end january_1_is_monday_l153_153605


namespace trapezium_other_side_length_l153_153865

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l153_153865


namespace diameter_of_double_area_square_l153_153673

-- Define the given conditions and the problem to be solved
theorem diameter_of_double_area_square (d₁ : ℝ) (d₁_eq : d₁ = 4 * Real.sqrt 2) :
  ∃ d₂ : ℝ, d₂ = 8 :=
by
  -- Define the conditions
  let s₁ := d₁ / Real.sqrt 2
  have s₁_sq : s₁ ^ 2 = (d₁ ^ 2) / 2 := by sorry -- Pythagorean theorem

  let A₁ := s₁ ^ 2
  have A₁_eq : A₁ = 16 := by sorry -- Given diagonal, thus area

  let A₂ := 2 * A₁
  have A₂_eq : A₂ = 32 := by sorry -- Double the area

  let s₂ := Real.sqrt A₂
  have s₂_eq : s₂ = 4 * Real.sqrt 2 := by sorry -- Side length of second square

  let d₂ := s₂ * Real.sqrt 2
  have d₂_eq : d₂ = 8 := by sorry -- Diameter of the second square

  -- Prove the theorem
  existsi d₂
  exact d₂_eq

end diameter_of_double_area_square_l153_153673


namespace susie_large_rooms_count_l153_153603

theorem susie_large_rooms_count:
  (∀ small_rooms medium_rooms large_rooms : ℕ,  
    (small_rooms = 4) → 
    (medium_rooms = 3) → 
    (large_rooms = x) → 
    (225 = small_rooms * 15 + medium_rooms * 25 + large_rooms * 35) → 
    x = 2) :=
by
  intros small_rooms medium_rooms large_rooms
  intros h1 h2 h3 h4
  sorry

end susie_large_rooms_count_l153_153603


namespace arithmetic_expr_eval_l153_153178

/-- A proof that the arithmetic expression (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) evaluates to -13122. -/
theorem arithmetic_expr_eval : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 :=
by
  sorry

end arithmetic_expr_eval_l153_153178


namespace blue_socks_count_l153_153766

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l153_153766


namespace symmetric_sufficient_not_necessary_l153_153484

theorem symmetric_sufficient_not_necessary (φ : Real) : 
    φ = - (Real.pi / 6) →
    ∃ f : Real → Real, (∀ x, f x = Real.sin (2 * x - φ)) ∧ 
    ∀ x, f (2 * (Real.pi / 6) - x) = f x :=
by
  sorry

end symmetric_sufficient_not_necessary_l153_153484


namespace calculate_total_travel_time_l153_153302

/-- The total travel time, including stops, from the first station to the last station. -/
def total_travel_time (d1 d2 d3 : ℕ) (s1 s2 s3 : ℕ) (t1 t2 : ℕ) : ℚ :=
  let leg1_time := d1 / s1
  let stop1_time := t1 / 60
  let leg2_time := d2 / s2
  let stop2_time := t2 / 60
  let leg3_time := d3 / s3
  leg1_time + stop1_time + leg2_time + stop2_time + leg3_time

/-- Proof that total travel time is 2 hours and 22.5 minutes. -/
theorem calculate_total_travel_time :
  total_travel_time 30 40 50 60 40 80 10 5 = 2.375 :=
by
  sorry

end calculate_total_travel_time_l153_153302


namespace remaining_marbles_l153_153941

theorem remaining_marbles (initial_marbles : ℕ) (num_customers : ℕ) (marble_range : List ℕ)
  (h_initial : initial_marbles = 2500)
  (h_customers : num_customers = 50)
  (h_range : marble_range = List.range' 1 50)
  (disjoint_range : ∀ (a b : ℕ), a ∈ marble_range → b ∈ marble_range → a ≠ b → a + b ≤ 50) :
  initial_marbles - (num_customers * (50 + 1) / 2) = 1225 :=
by
  sorry

end remaining_marbles_l153_153941


namespace evaluate_f_at_7_l153_153566

theorem evaluate_f_at_7 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4) :
  f 7 = -3 :=
by
  sorry

end evaluate_f_at_7_l153_153566


namespace range_of_f_l153_153031

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ |x + 1|

theorem range_of_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_of_f_l153_153031


namespace inequality_proof_l153_153779

variable {a b c d : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  64 * (abcd + 1) / (a + b + c + d)^2 ≤ a^2 + b^2 + c^2 + d^2 + 1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 :=
by 
  sorry

end inequality_proof_l153_153779


namespace inequality_proof_l153_153774

theorem inequality_proof
  (a b c d : ℝ) (h0 : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0) (h4 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1 / 3 :=
sorry

end inequality_proof_l153_153774


namespace right_triangle_perimeter_l153_153062

-- Conditions
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (h_area : 1 / 2 * 15 * b = 150)
variable (h_pythagorean : a^2 + b^2 = c^2)
variable (h_a : a = 15)

-- The theorem to prove the perimeter is 60 units
theorem right_triangle_perimeter : a + b + c = 60 := by
  sorry

end right_triangle_perimeter_l153_153062


namespace square_area_is_81_l153_153960

def square_perimeter (s : ℕ) : ℕ := 4 * s
def square_area (s : ℕ) : ℕ := s * s

theorem square_area_is_81 (s : ℕ) (h : square_perimeter s = 36) : square_area s = 81 :=
by {
  sorry
}

end square_area_is_81_l153_153960


namespace n_is_prime_l153_153407

theorem n_is_prime (p : ℕ) (h : ℕ) (n : ℕ)
  (hp : Nat.Prime p)
  (hh : h < p)
  (hn : n = p * h + 1)
  (div_n : n ∣ (2^(n-1) - 1))
  (not_div_n : ¬ n ∣ (2^h - 1)) : Nat.Prime n := sorry

end n_is_prime_l153_153407


namespace third_place_prize_is_120_l153_153334

noncomputable def prize_for_third_place (total_prize : ℕ) (first_place_prize : ℕ) (second_place_prize : ℕ) (prize_per_novel : ℕ) (num_novels_receiving_prize : ℕ) : ℕ :=
  let remaining_prize := total_prize - first_place_prize - second_place_prize
  let total_other_prizes := num_novels_receiving_prize * prize_per_novel
  remaining_prize - total_other_prizes

theorem third_place_prize_is_120 : prize_for_third_place 800 200 150 22 15 = 120 := by
  sorry

end third_place_prize_is_120_l153_153334


namespace knight_probability_l153_153307

theorem knight_probability :
  let Q := 1 - ((binom 16 3) / (binom 20 4)) in
  let simplified_Q := (66 : ℚ) / 75 in
  let sum_nd := 66 + 75 in
  Q = simplified_Q ∧ sum_nd = 141 :=
by
  sorry

end knight_probability_l153_153307


namespace festival_second_day_attendance_l153_153999

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l153_153999


namespace xiaohong_height_l153_153049

theorem xiaohong_height 
  (father_height_cm : ℕ)
  (height_difference_dm : ℕ)
  (father_height : father_height_cm = 170)
  (height_difference : height_difference_dm = 4) :
  ∃ xiaohong_height_cm : ℕ, xiaohong_height_cm + height_difference_dm * 10 = father_height_cm :=
by
  use 130
  sorry

end xiaohong_height_l153_153049


namespace solution_of_equation_l153_153632

noncomputable def solve_equation (x : ℝ) : Prop :=
  2021 * x^(2020/202) - 1 = 2020 * x ∧ x ≥ 0

theorem solution_of_equation : solve_equation 1 :=
by {
  sorry,
}

end solution_of_equation_l153_153632


namespace ellipse_centroid_locus_l153_153375

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
noncomputable def centroid_locus (x y : ℝ) : Prop := (9 * x^2) / 4 + 3 * y^2 = 1 ∧ y ≠ 0

theorem ellipse_centroid_locus (x y : ℝ) (h : ellipse_equation x y) : centroid_locus (x / 3) (y / 3) :=
  sorry

end ellipse_centroid_locus_l153_153375


namespace medians_form_right_triangle_medians_inequality_l153_153157

variable {α : Type*}
variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}
variable (orthogonal_medians : m_a * m_b = 0)

-- Part (a)
theorem medians_form_right_triangle
  (orthogonal_medians : m_a * m_b = 0) :
  m_a^2 + m_b^2 = m_c^2 :=
sorry

-- Part (b)
theorem medians_inequality
  (orthogonal_medians : m_a * m_b = 0)
  (triangle_sides : a^2 + b^2 = 5 * c^2): 
  5 * (a^2 + b^2 - c^2) ≥ 8 * a * b :=
sorry

end medians_form_right_triangle_medians_inequality_l153_153157


namespace isolate_urea_decomposing_bacteria_valid_option_l153_153476

variable (KH2PO4 Na2HPO4 MgSO4_7H2O urea glucose agar water : Type)
variable (urea_decomposing_bacteria : Type)
variable (CarbonSource : Type → Prop)
variable (NitrogenSource : Type → Prop)
variable (InorganicSalt : Type → Prop)
variable (bacteria_can_synthesize_urease : urea_decomposing_bacteria → Prop)

axiom KH2PO4_is_inorganic_salt : InorganicSalt KH2PO4
axiom Na2HPO4_is_inorganic_salt : InorganicSalt Na2HPO4
axiom MgSO4_7H2O_is_inorganic_salt : InorganicSalt MgSO4_7H2O
axiom urea_is_nitrogen_source : NitrogenSource urea

theorem isolate_urea_decomposing_bacteria_valid_option :
  (InorganicSalt KH2PO4) ∧
  (InorganicSalt Na2HPO4) ∧
  (InorganicSalt MgSO4_7H2O) ∧
  (NitrogenSource urea) ∧
  (CarbonSource glucose) → (∃ bacteria : urea_decomposing_bacteria, bacteria_can_synthesize_urease bacteria) := sorry

end isolate_urea_decomposing_bacteria_valid_option_l153_153476


namespace sum_of_common_ratios_l153_153269

variable (m x y : ℝ)
variable (h₁ : x ≠ y)
variable (h₂ : a2 = m * x)
variable (h₃ : a3 = m * x^2)
variable (h₄ : b2 = m * y)
variable (h₅ : b3 = m * y^2)
variable (h₆ : a3 - b3 = 3 * (a2 - b2))

theorem sum_of_common_ratios : x + y = 3 :=
by
  sorry

end sum_of_common_ratios_l153_153269


namespace sum_of_decimals_is_fraction_l153_153859

def decimal_to_fraction_sum : ℚ :=
  (1 / 10) + (2 / 100) + (3 / 1000) + (4 / 10000) + (5 / 100000) + (6 / 1000000) + (7 / 10000000)

theorem sum_of_decimals_is_fraction :
  decimal_to_fraction_sum = 1234567 / 10000000 :=
by sorry

end sum_of_decimals_is_fraction_l153_153859


namespace convex_polygon_quadrilateral_division_l153_153174

open Nat

theorem convex_polygon_quadrilateral_division (n : ℕ) : ℕ :=
  if h : n > 0 then
    1 / (2 * n - 1) * (Nat.choose (3 * n - 3) (n - 1))
  else
    0

end convex_polygon_quadrilateral_division_l153_153174


namespace fourth_number_in_12th_row_is_92_l153_153029

-- Define the number of elements per row and the row number
def elements_per_row := 8
def row_number := 12

-- Define the last number in a row function
def last_number_in_row (n : ℕ) := elements_per_row * n

-- Define the starting number in a row function
def starting_number_in_row (n : ℕ) := (elements_per_row * (n - 1)) + 1

-- Define the nth number in a specified row function
def nth_number_in_row (n : ℕ) (k : ℕ) := starting_number_in_row n + (k - 1)

-- Prove that the fourth number in the 12th row is 92
theorem fourth_number_in_12th_row_is_92 : nth_number_in_row 12 4 = 92 :=
by
  -- state the required equivalences
  sorry

end fourth_number_in_12th_row_is_92_l153_153029


namespace cost_of_first_ring_is_10000_l153_153764

theorem cost_of_first_ring_is_10000 (x : ℝ) (h₁ : x + 2*x - x/2 = 25000) : x = 10000 :=
sorry

end cost_of_first_ring_is_10000_l153_153764


namespace not_a_factorization_method_l153_153317

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end not_a_factorization_method_l153_153317


namespace triangle_area_l153_153401

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) 
  (h_c : c = 2) (h_C : C = π / 3)
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l153_153401


namespace max_ratio_is_99_over_41_l153_153137

noncomputable def max_ratio (x y : ℕ) (h1 : x > y) (h2 : x + y = 140) : ℚ :=
  if h : y ≠ 0 then (x / y : ℚ) else 0

theorem max_ratio_is_99_over_41 : ∃ (x y : ℕ), x > y ∧ x + y = 140 ∧ max_ratio x y (by sorry) (by sorry) = (99 / 41 : ℚ) :=
by
  sorry

end max_ratio_is_99_over_41_l153_153137


namespace smallest_k_for_divisibility_l153_153545

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l153_153545


namespace Q_difference_l153_153879

def Q (x n : ℕ) : ℕ :=
  (Finset.range (10^n)).sum (λ k => x / (k + 1))

theorem Q_difference (n : ℕ) : 
  Q (10^n) n - Q (10^n - 1) n = (n + 1)^2 :=
by
  sorry

end Q_difference_l153_153879


namespace sam_pennies_l153_153286

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end sam_pennies_l153_153286


namespace shaded_square_area_l153_153825

theorem shaded_square_area (a b s : ℝ) (h : a * b = 40) :
  ∃ s, s^2 = 2500 / 441 :=
by
  sorry

end shaded_square_area_l153_153825


namespace sin_arithmetic_sequence_180_deg_l153_153524

open Real

theorem sin_arithmetic_sequence_180_deg :
  ∀ (b : ℝ), (0 < b ∧ b < 360) → (sin b + sin (3 * b) = 2 * sin (2 * b)) → b = 180 :=
by
  rintro b ⟨hb1, hb2⟩ h
  sorry

end sin_arithmetic_sequence_180_deg_l153_153524


namespace total_wheels_in_parking_lot_l153_153397

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot_l153_153397


namespace factorial_ratio_l153_153202

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l153_153202


namespace isosceles_triangle_sides_l153_153901

theorem isosceles_triangle_sides (a b c : ℕ) (h₁ : a + b + c = 10) (h₂ : (a = b ∨ b = c ∨ a = c)) 
  (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) : 
  (a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 4) := 
by
  sorry

end isosceles_triangle_sides_l153_153901


namespace find_constant_l153_153903

variable (constant : ℝ)

theorem find_constant (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 1 - 2 * t)
  (h2 : y = constant * t - 2)
  (h3 : x = y) : constant = 2 :=
by
  sorry

end find_constant_l153_153903


namespace solve_inequality_l153_153887

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 2) * x + 2 < 0

-- Prove the solution sets for different values of a
theorem solve_inequality :
  ∀ (a : ℝ),
    (a = -1 → {x : ℝ | inequality a x} = {x | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x | x < 2 / a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x | 1 < x ∧ x < 2 / a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x | 2 / a < x ∧ x < 1}) :=
by sorry

end solve_inequality_l153_153887


namespace total_pages_read_l153_153261

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l153_153261


namespace arithmetic_sequence_geometric_sequence_added_number_l153_153504

theorem arithmetic_sequence_geometric_sequence_added_number 
  (a : ℕ → ℤ)
  (h1 : a 1 = -8)
  (h2 : a 2 = -6)
  (h_arith : ∀ n, a n = -8 + (n-1) * 2)  -- derived from the conditions
  (x : ℤ)
  (h_geo : (-8 + x) * x = (-2 + x) * (-2 + x)) :
  x = -1 := 
sorry

end arithmetic_sequence_geometric_sequence_added_number_l153_153504


namespace factorize_expr_l153_153521

theorem factorize_expr (x y : ℝ) : x^3 - 4 * x * y^2 = x * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l153_153521


namespace cos_alpha_correct_l153_153749

-- Define the point P
def P : ℝ × ℝ := (3, -4)

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def r : ℝ :=
  Real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define x-coordinate of point P
def x : ℝ := P.1

-- Define the cosine of the angle
noncomputable def cos_alpha : ℝ :=
  x / r

-- Prove that cos_alpha equals 3/5 given the conditions
theorem cos_alpha_correct : cos_alpha = 3 / 5 :=
by
  sorry

end cos_alpha_correct_l153_153749


namespace valid_cube_placements_count_l153_153505

-- Define the initial cross configuration and the possible placements for the sixth square.
structure CrossConfiguration :=
  (squares : Finset (ℕ × ℕ)) -- Assume (ℕ × ℕ) represents the positions of the squares.

def valid_placements (config : CrossConfiguration) : Finset (ℕ × ℕ) :=
  -- Placeholder definition to represent the valid placements for the sixth square.
  sorry

theorem valid_cube_placements_count (config : CrossConfiguration) :
  (valid_placements config).card = 4 := 
by 
  sorry

end valid_cube_placements_count_l153_153505


namespace ramu_profit_percent_l153_153051

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_percent :
  profit_percent 42000 13000 64500 = 17.27 :=
by
  -- Placeholder for the proof
  sorry

end ramu_profit_percent_l153_153051


namespace probability_yellow_or_blue_twice_l153_153125

theorem probability_yellow_or_blue_twice :
  let total_faces := 12
  let yellow_faces := 4
  let blue_faces := 2
  let probability_yellow_or_blue := (yellow_faces / total_faces) + (blue_faces / total_faces)
  (probability_yellow_or_blue * probability_yellow_or_blue) = 1 / 4 := 
by
  sorry

end probability_yellow_or_blue_twice_l153_153125


namespace green_duck_percentage_l153_153322

noncomputable def smaller_pond_ducks : ℕ := 45
noncomputable def larger_pond_ducks : ℕ := 55
noncomputable def green_percentage_small_pond : ℝ := 0.20
noncomputable def green_percentage_large_pond : ℝ := 0.40

theorem green_duck_percentage :
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_ducks_smaller := green_percentage_small_pond * (smaller_pond_ducks : ℝ)
  let green_ducks_larger := green_percentage_large_pond * (larger_pond_ducks : ℝ)
  let total_green_ducks := green_ducks_smaller + green_ducks_larger
  (total_green_ducks / total_ducks) * 100 = 31 :=
by {
  -- The proof is omitted.
  sorry
}

end green_duck_percentage_l153_153322


namespace root_equation_solution_l153_153487

-- Given conditions from the problem
def is_root_of_quadratic (m : ℝ) : Prop :=
  m^2 - m - 110 = 0

-- Statement of the proof problem
theorem root_equation_solution (m : ℝ) (h : is_root_of_quadratic m) : (m - 1)^2 + m = 111 := 
sorry

end root_equation_solution_l153_153487


namespace given_conditions_l153_153246

theorem given_conditions :
  ∀ (t : ℝ), t > 0 → t ≠ 1 → 
  let x := t^(2/(t-1))
  let y := t^((t+1)/(t-1))
  ¬ ((y * x^(1/y) = x * y^(1/x)) ∨ (y * x^y = x * y^x) ∨ (y^x = x^y) ∨ (x^(x+y) = y^(x+y))) :=
by
  intros t ht_pos ht_ne_1 x_def y_def
  let x := x_def
  let y := y_def
  sorry

end given_conditions_l153_153246


namespace g_extreme_values_l153_153177

-- Definitions based on the conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

-- Theorem statement
theorem g_extreme_values : 
  (g (1/3) = 31/27) ∧ (g 1 = 1) := sorry

end g_extreme_values_l153_153177


namespace complex_evaluation_l153_153615

theorem complex_evaluation (a b : ℂ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a^2 + a * b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := 
by 
  sorry

end complex_evaluation_l153_153615


namespace sequence_general_term_l153_153883

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, a n = S n - S (n-1) :=
by
  -- The proof will be filled in here
  sorry

end sequence_general_term_l153_153883


namespace triangle_median_equal_bc_l153_153922

-- Let \( ABC \) be a triangle, \( AB = 2 \), \( AC = 3 \), and the median from \( A \) to \( BC \) has the same length as \( BC \).
theorem triangle_median_equal_bc (A B C M : Type) (AB AC BC AM : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) 
  (hMedian : BC = AM) (hM : M = midpoint B C) :
  BC = real.sqrt (26 / 5) :=
by sorry

end triangle_median_equal_bc_l153_153922


namespace ratio_of_areas_eq_nine_sixteenth_l153_153964

-- Definitions based on conditions
def side_length_C : ℝ := 45
def side_length_D : ℝ := 60
def area (s : ℝ) : ℝ := s * s

-- Theorem stating the desired proof problem
theorem ratio_of_areas_eq_nine_sixteenth :
  (area side_length_C) / (area side_length_D) = 9 / 16 :=
by
  sorry

end ratio_of_areas_eq_nine_sixteenth_l153_153964


namespace max_vector_sum_l153_153733

theorem max_vector_sum
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ := (2, 0))
  (hA : A.1^2 + A.2^2 = 1)
  (hB : B.1^2 + B.2^2 = 1)
  (hC : C.1^2 + C.2^2 = 1)
  (h_perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  |(2,0) - A + (2,0) - B + (2,0) - C| = 7 := sorry

end max_vector_sum_l153_153733


namespace pole_length_is_5_l153_153814

theorem pole_length_is_5 (x : ℝ) (gate_width gate_height : ℝ) 
  (h_gate_wide : gate_width = 3) 
  (h_pole_taller : gate_height = x - 1) 
  (h_diagonal : x^2 = gate_height^2 + gate_width^2) : 
  x = 5 :=
by
  sorry

end pole_length_is_5_l153_153814


namespace polynomial_divisible_2520_l153_153288

theorem polynomial_divisible_2520 (n : ℕ) : (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) % 2520 = 0 := 
sorry

end polynomial_divisible_2520_l153_153288


namespace sum_of_reciprocals_is_one_l153_153799

theorem sum_of_reciprocals_is_one (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x : ℚ)) + (1 / (y : ℚ)) + (1 / (z : ℚ)) = 1 ↔ (x, y, z) = (2, 4, 4) ∨ 
                                                    (x, y, z) = (2, 3, 6) ∨ 
                                                    (x, y, z) = (3, 3, 3) :=
by 
  sorry

end sum_of_reciprocals_is_one_l153_153799


namespace sine_double_angle_l153_153876

theorem sine_double_angle (theta : ℝ) (h : Real.tan (theta + Real.pi / 4) = 2) : Real.sin (2 * theta) = 3 / 5 :=
sorry

end sine_double_angle_l153_153876


namespace quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l153_153383

-- Proof Problem (1)
theorem quadratic_inequality_roots_a_eq_neg1
  (a : ℝ)
  (h : ∀ x, (-1 < x ∧ x < 3) → ax^2 - 2 * a * x + 3 > 0) :
  a = -1 :=
sorry

-- Proof Problem (2)
theorem quadratic_inequality_for_all_real_a_range
  (a : ℝ)
  (h : ∀ x, ax^2 - 2 * a * x + 3 > 0) :
  0 ≤ a ∧ a < 3 :=
sorry

end quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l153_153383


namespace calculator_sum_l153_153708

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

end calculator_sum_l153_153708


namespace division_of_mixed_numbers_l153_153167

noncomputable def mixed_to_improper (n : ℕ) (a b : ℕ) : ℚ :=
  n + (a / b)

theorem division_of_mixed_numbers : 
  (mixed_to_improper 7 1 3) / (mixed_to_improper 2 1 2) = 44 / 15 :=
by
  sorry

end division_of_mixed_numbers_l153_153167


namespace shaded_areas_are_different_l153_153354

theorem shaded_areas_are_different :
  let shaded_area_I := 3 / 8
  let shaded_area_II := 1 / 3
  let shaded_area_III := 1 / 2
  (shaded_area_I ≠ shaded_area_II) ∧ (shaded_area_I ≠ shaded_area_III) ∧ (shaded_area_II ≠ shaded_area_III) :=
by
  sorry

end shaded_areas_are_different_l153_153354


namespace evaluate_expression_l153_153082

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l153_153082


namespace set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l153_153153

variable {U : Type} [DecidableEq U]
variables (A B C K : Set U)

theorem set_theorem_1 : (A \ K) ∪ (B \ K) = (A ∪ B) \ K := sorry
theorem set_theorem_2 : A \ (B \ C) = (A \ B) ∪ (A ∩ C) := sorry
theorem set_theorem_3 : A \ (A \ B) = A ∩ B := sorry
theorem set_theorem_4 : (A \ B) \ C = (A \ C) \ (B \ C) := sorry
theorem set_theorem_5 : A \ (B ∩ C) = (A \ B) ∪ (A \ C) := sorry
theorem set_theorem_6 : A \ (B ∪ C) = (A \ B) ∩ (A \ C) := sorry
theorem set_theorem_7 : A \ B = (A ∪ B) \ B ∧ A \ B = A \ (A ∩ B) := sorry

end set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l153_153153


namespace trapezium_other_side_length_l153_153867

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l153_153867


namespace money_needed_to_finish_collection_l153_153264

-- Define the conditions
def initial_action_figures : ℕ := 9
def total_action_figures_needed : ℕ := 27
def cost_per_action_figure : ℕ := 12

-- Define the goal
theorem money_needed_to_finish_collection 
  (initial : ℕ) (total_needed : ℕ) (cost_per : ℕ) 
  (h1 : initial = initial_action_figures)
  (h2 : total_needed = total_action_figures_needed)
  (h3 : cost_per = cost_per_action_figure) :
  ((total_needed - initial) * cost_per = 216) := 
by
  sorry

end money_needed_to_finish_collection_l153_153264


namespace part_a_solution_exists_l153_153947

theorem part_a_solution_exists : ∃ (x y : ℕ), x^2 - y^2 = 31 ∧ x = 16 ∧ y = 15 := 
by 
  sorry

end part_a_solution_exists_l153_153947


namespace scientific_notation_correct_l153_153689

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l153_153689


namespace mean_proportional_AC_is_correct_l153_153094

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end mean_proportional_AC_is_correct_l153_153094


namespace problem_statement_l153_153245

theorem problem_statement (x y : ℕ) (h1 : x = 3) (h2 :y = 5) :
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5 / 7 := 
by 
  sorry

end problem_statement_l153_153245


namespace calculate_l153_153276

def q (x y : ℤ) : ℤ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x < 0 ∧ y ≤ 0 then x - 3*y
  else 4*x + 2*y

theorem calculate : q (q 2 (-2)) (q (-3) 1) = -4 := 
  by
    sorry

end calculate_l153_153276


namespace Brenda_bakes_cakes_l153_153696

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l153_153696


namespace sqrt_sqrt_16_l153_153784

theorem sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := sorry

end sqrt_sqrt_16_l153_153784


namespace trapezium_side_length_l153_153868

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l153_153868


namespace min_neighbor_pairs_l153_153284

theorem min_neighbor_pairs (n : ℕ) (h : n = 2005) :
  ∃ (pairs : ℕ), pairs = 56430 :=
by
  sorry

end min_neighbor_pairs_l153_153284


namespace iggy_wednesday_run_6_l153_153750

open Nat

noncomputable def iggy_miles_wednesday : ℕ :=
  let total_time := 4 * 60    -- Iggy spends 4 hours running (240 minutes)
  let pace := 10              -- Iggy runs 1 mile in 10 minutes
  let monday := 3
  let tuesday := 4
  let thursday := 8
  let friday := 3
  let total_miles_other_days := monday + tuesday + thursday + friday
  let total_time_other_days := total_miles_other_days * pace
  let wednesday_time := total_time - total_time_other_days
  wednesday_time / pace

theorem iggy_wednesday_run_6 :
  iggy_miles_wednesday = 6 := by
  sorry

end iggy_wednesday_run_6_l153_153750


namespace Theresa_video_games_l153_153804

variable (Tory Julia Theresa : ℕ)

def condition1 : Prop := Tory = 6
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Theresa = (Julia * 3) + 5

theorem Theresa_video_games : condition1 Tory → condition2 Tory Julia → condition3 Julia Theresa → Theresa = 11 := by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end Theresa_video_games_l153_153804


namespace book_cost_l153_153629

-- Define the problem parameters
variable (p : ℝ) -- cost of one book in dollars

-- Conditions given in the problem
def seven_copies_cost_less_than_15 (p : ℝ) : Prop := 7 * p < 15
def eleven_copies_cost_more_than_22 (p : ℝ) : Prop := 11 * p > 22

-- The theorem stating the cost is between the given bounds
theorem book_cost (p : ℝ) (h1 : seven_copies_cost_less_than_15 p) (h2 : eleven_copies_cost_more_than_22 p) : 
    2 < p ∧ p < (15 / 7 : ℝ) :=
sorry

end book_cost_l153_153629


namespace correct_exponent_operation_l153_153477

theorem correct_exponent_operation (x : ℝ) : x ^ 3 * x ^ 2 = x ^ 5 :=
by sorry

end correct_exponent_operation_l153_153477


namespace triangle_angle_A_l153_153126

theorem triangle_angle_A (a c C A : Real) (h1 : a = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) 
(h4 : Real.sin A = 1 / 2) : A = Real.pi / 6 :=
sorry

end triangle_angle_A_l153_153126


namespace ab_abs_value_l153_153026

theorem ab_abs_value {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : ∃ r s : ℤ, (x - r)^2 * (x - s) = x^3 + a * x^2 + b * x + 9 * a) :
  |a * b| = 1344 := 
sorry

end ab_abs_value_l153_153026


namespace solution_of_equation_l153_153429

def solve_equation (x : ℚ) : Prop := 
  (x^2 + 3 * x + 4) / (x + 5) = x + 6

theorem solution_of_equation : solve_equation (-13/4) := 
by
  sorry

end solution_of_equation_l153_153429


namespace stratified_sample_l153_153753

theorem stratified_sample 
  (total_households : ℕ) 
  (high_income_households : ℕ) 
  (middle_income_households : ℕ) 
  (low_income_households : ℕ) 
  (sample_size : ℕ)
  (H1 : total_households = 600) 
  (H2 : high_income_households = 150)
  (H3 : middle_income_households = 360)
  (H4 : low_income_households = 90)
  (H5 : sample_size = 100) : 
  (middle_income_households * sample_size / total_households = 60) := 
by 
  sorry

end stratified_sample_l153_153753


namespace wheat_bread_served_l153_153942

noncomputable def total_bread_served : ℝ := 0.6
noncomputable def white_bread_served : ℝ := 0.4

theorem wheat_bread_served : total_bread_served - white_bread_served = 0.2 :=
by
  sorry

end wheat_bread_served_l153_153942


namespace right_triangle_area_l153_153326

theorem right_triangle_area (a b c : ℝ)
    (h1 : a = 16)
    (h2 : ∃ r, r = 6)
    (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a^2 + b^2 = c^2) :
    1/2 * a * b = 240 := 
by
  -- given:
  -- a = 16
  -- ∃ r, r = 6
  -- c = Real.sqrt (a^2 + b^2)
  -- a^2 + b^2 = c^2
  -- Prove: 1/2 * a * b = 240
  sorry

end right_triangle_area_l153_153326


namespace dave_winfield_home_runs_l153_153066

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l153_153066


namespace cardinals_home_runs_second_l153_153752

-- Define the conditions
def cubs_home_runs_third : ℕ := 2
def cubs_home_runs_fifth : ℕ := 1
def cubs_home_runs_eighth : ℕ := 2
def cubs_total_home_runs := cubs_home_runs_third + cubs_home_runs_fifth + cubs_home_runs_eighth
def cubs_more_than_cardinals : ℕ := 3
def cardinals_home_runs_fifth : ℕ := 1

-- Define the proof problem
theorem cardinals_home_runs_second :
  (cubs_total_home_runs = cardinals_total_home_runs + cubs_more_than_cardinals) →
  (cardinals_total_home_runs - cardinals_home_runs_fifth = 1) :=
sorry

end cardinals_home_runs_second_l153_153752


namespace no_infinite_harmonic_mean_sequence_l153_153628

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), (∀ n, a n = a 0 → False) ∧
                   (∀ i, 1 ≤ i → a i = (2 * a (i - 1) * a (i + 1)) / (a (i - 1) + a (i + 1))) :=
sorry

end no_infinite_harmonic_mean_sequence_l153_153628


namespace sum_of_areas_squares_l153_153306

theorem sum_of_areas_squares (a : ℕ) (h1 : (a + 4)^2 - a^2 = 80) : a^2 + (a + 4)^2 = 208 := by
  sorry

end sum_of_areas_squares_l153_153306


namespace betty_additional_money_needed_l153_153841

def wallet_cost : ℝ := 100
def betty_savings : ℝ := wallet_cost / 2
def parents_contribution : ℝ := 15
def grandparents_contribution : ℝ := 2 * parents_contribution

def total_money : ℝ := betty_savings + parents_contribution + grandparents_contribution
def amount_needed : ℝ := wallet_cost - total_money

theorem betty_additional_money_needed : amount_needed = 5 := by
  sorry

end betty_additional_money_needed_l153_153841


namespace square_perimeter_l153_153915

theorem square_perimeter (d : ℝ) (h : d = 2 * real.sqrt 2) : ∃ p, p = 8 :=
by
  sorry

end square_perimeter_l153_153915


namespace no_x_for_rational_sin_cos_l153_153710

-- Define rational predicate
def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

-- Define the statement of the problem
theorem no_x_for_rational_sin_cos :
  ∀ x : ℝ, ¬ (is_rational (Real.sin x + Real.sqrt 2) ∧ is_rational (Real.cos x - Real.sqrt 2)) :=
by
  -- Placeholder for proof
  sorry

end no_x_for_rational_sin_cos_l153_153710


namespace sufficiency_not_necessity_l153_153136

def l1 : Type := sorry
def l2 : Type := sorry

def skew_lines (l1 l2 : Type) : Prop := sorry
def do_not_intersect (l1 l2 : Type) : Prop := sorry

theorem sufficiency_not_necessity (p q : Prop) 
  (hp : p = skew_lines l1 l2)
  (hq : q = do_not_intersect l1 l2) :
  (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end sufficiency_not_necessity_l153_153136


namespace smallest_n_equal_sums_l153_153617

def sum_first_n_arithmetic (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_n_equal_sums : ∀ (n : ℕ), 
  sum_first_n_arithmetic 7 4 n = sum_first_n_arithmetic 15 3 n → n ≠ 0 → n = 7 := by
  intros n h1 h2
  sorry

end smallest_n_equal_sums_l153_153617


namespace solve_equation_2021_2020_l153_153631

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l153_153631


namespace abs_diff_ps_pds_eq_31_100_l153_153057

-- Defining the conditions
def num_red : ℕ := 500
def num_black : ℕ := 700
def num_blue : ℕ := 800
def total_marbles : ℕ := num_red + num_black + num_blue
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculating P_s and P_d
def ways_same_color : ℕ := choose num_red 2 + choose num_black 2 + choose num_blue 2
def total_ways : ℕ := choose total_marbles 2
def P_s : ℚ := ways_same_color / total_ways

def ways_different_color : ℕ := num_red * num_black + num_red * num_blue + num_black * num_blue
def P_d : ℚ := ways_different_color / total_ways

-- Proving the statement
theorem abs_diff_ps_pds_eq_31_100 : |P_s - P_d| = (31 : ℚ) / 100 := by
  sorry

end abs_diff_ps_pds_eq_31_100_l153_153057


namespace opposite_2024_eq_neg_2024_l153_153958

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l153_153958


namespace lena_muffins_l153_153020

theorem lena_muffins (x y z : Real) 
  (h1 : x + 2 * y + 3 * z = 3 * x + z)
  (h2 : 3 * x + z = 6 * y)
  (h3 : x + 2 * y + 3 * z = 6 * y)
  (lenas_spending : 2 * x + 2 * z = 6 * y) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end lena_muffins_l153_153020


namespace polynomial_identity_l153_153593

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l153_153593


namespace trapezium_side_length_l153_153870

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l153_153870


namespace john_paid_more_than_jane_l153_153769

theorem john_paid_more_than_jane :
    let original_price : ℝ := 40.00
    let discount_percentage : ℝ := 0.10
    let tip_percentage : ℝ := 0.15
    let discounted_price : ℝ := original_price - (discount_percentage * original_price)
    let john_tip : ℝ := tip_percentage * original_price
    let john_total : ℝ := discounted_price + john_tip
    let jane_tip : ℝ := tip_percentage * discounted_price
    let jane_total : ℝ := discounted_price + jane_tip
    let difference : ℝ := john_total - jane_total
    difference = 0.60 :=
by
  sorry

end john_paid_more_than_jane_l153_153769


namespace geometric_series_terms_l153_153735

theorem geometric_series_terms 
    (b1 q : ℝ)
    (h₁ : (b1^2 / (1 + q + q^2)) = 12)
    (h₂ : (b1^2 / (1 + q^2)) = (36 / 5)) :
    (b1 = 3 ∨ b1 = -3) ∧ q = -1/2 :=
by
  sorry

end geometric_series_terms_l153_153735


namespace brenda_cakes_l153_153701

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l153_153701


namespace negation_of_existence_proposition_l153_153955

theorem negation_of_existence_proposition :
  ¬ (∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ ∀ x : ℝ, x^2 + 2*x - 8 ≠ 0 := by
  sorry

end negation_of_existence_proposition_l153_153955


namespace value_of_a_l153_153230

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l153_153230


namespace value_of_algebraic_expression_l153_153103

variable {a b : ℝ}

theorem value_of_algebraic_expression (h : b = 4 * a + 3) : 4 * a - b - 2 = -5 := 
by
  sorry

end value_of_algebraic_expression_l153_153103


namespace complete_the_square_problem_l153_153353

theorem complete_the_square_problem :
  ∃ r s : ℝ, (r = -2) ∧ (s = 9) ∧ (r + s = 7) ∧ ∀ x : ℝ, 15 * x ^ 2 - 60 * x - 135 = 0 ↔ (x + r) ^ 2 = s := 
by
  sorry

end complete_the_square_problem_l153_153353


namespace smallest_k_divides_l153_153553

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l153_153553


namespace find_integers_for_perfect_square_l153_153714

theorem find_integers_for_perfect_square (x : ℤ) :
  (∃ k : ℤ, x * (x + 1) * (x + 7) * (x + 8) = k^2) ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end find_integers_for_perfect_square_l153_153714


namespace Carver_school_earnings_l153_153427

noncomputable def total_earnings_Carver_school : ℝ :=
  let base_payment := 20
  let total_payment := 900
  let Allen_days := 7 * 3
  let Balboa_days := 5 * 6
  let Carver_days := 4 * 10
  let total_student_days := Allen_days + Balboa_days + Carver_days
  let adjusted_total_payment := total_payment - 3 * base_payment
  let daily_wage := adjusted_total_payment / total_student_days
  daily_wage * Carver_days

theorem Carver_school_earnings : 
  total_earnings_Carver_school = 369.6 := 
by 
  sorry

end Carver_school_earnings_l153_153427


namespace expressions_equal_iff_l153_153854

theorem expressions_equal_iff (a b c: ℝ) : a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 :=
by 
  sorry

end expressions_equal_iff_l153_153854


namespace find_number_l153_153428

theorem find_number 
  (x : ℚ) 
  (h : (3 / 4) * x - (8 / 5) * x + 63 = 12) : 
  x = 60 := 
by
  sorry

end find_number_l153_153428


namespace smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l153_153813

theorem smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7 :
  ∃ n : ℕ, n % 45 = 0 ∧ (n - 100) % 7 = 0 ∧ n = 135 :=
sorry

end smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l153_153813


namespace length_of_AC_l153_153096

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l153_153096


namespace max_value_of_f_l153_153932

open Real

noncomputable def f (θ : ℝ) : ℝ :=
  sin (θ / 2) * (1 + cos θ)

theorem max_value_of_f : 
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π → f θ' ≤ f θ) ∧ f θ = 4 * sqrt 3 / 9) := 
by
  sorry

end max_value_of_f_l153_153932


namespace find_value_of_N_l153_153451

theorem find_value_of_N (x N : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (N + 3 * x)^4) : N = 1.5 := by
  -- Here we will assume that the proof is filled in and correct.
  sorry

end find_value_of_N_l153_153451


namespace age_ratio_l153_153796

theorem age_ratio (x : ℚ) (h1 : 6 * x - 4 = 3 * x + 4) : 
  let age_A := 6 * x;
  let age_B := 3 * x;
  let age_A_hence := age_A + 4;
  let age_B_ago := age_B - 4;
  ratio : ℚ := age_A_hence / age_B_ago
in ratio = 5 :=
by {
  let age_A := 6 * x;
  let age_B := 3 * x;
  let age_A_hence := age_A + 4;
  let age_B_ago := age_B - 4;
  have h : age_A_hence / age_B_ago = 5, sorry;
  exact h
}

end age_ratio_l153_153796


namespace length_of_train_B_l153_153166

-- Given conditions
def lengthTrainA := 125  -- in meters
def speedTrainA := 54    -- in km/hr
def speedTrainB := 36    -- in km/hr
def timeToCross := 11    -- in seconds

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Relative speed of the trains in m/s
def relativeSpeed := kmhr_to_mps (speedTrainA + speedTrainB)

-- Distance covered in the given time
def distanceCovered := relativeSpeed * timeToCross

-- Proof statement
theorem length_of_train_B : distanceCovered - lengthTrainA = 150 := 
by
  -- Proof will go here
  sorry

end length_of_train_B_l153_153166


namespace spectators_count_l153_153906

theorem spectators_count (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : (total_wristbands / wristbands_per_person = 125) :=
by
  sorry

end spectators_count_l153_153906


namespace area_of_sine_curve_l153_153436

theorem area_of_sine_curve :
  let f := (fun x => Real.sin x)
  let a := -Real.pi
  let b := 2 * Real.pi
  (∫ x in a..b, f x) = 6 :=
by
  sorry

end area_of_sine_curve_l153_153436


namespace subset_N_M_l153_153277

def M : Set ℝ := { x | ∃ (k : ℤ), x = k / 2 + 1 / 3 }
def N : Set ℝ := { x | ∃ (k : ℤ), x = k + 1 / 3 }

theorem subset_N_M : N ⊆ M := 
  sorry

end subset_N_M_l153_153277


namespace sum_of_tangency_points_l153_153514

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 21) (max (2 * x - 3) (5 * x + 1))

theorem sum_of_tangency_points {q : ℝ → ℝ} [IsQuadraticPolynomial q]
  (a1 a2 a3 : ℝ) (h1 : q a1 = f a1)
  (h2 : q a2 = f a2)
  (h3 : q a3 = f a3)
  (tangent1 : ∀ x, ∃ b, q x - (-7 * x - 21) = b * (x - a1) ^ 2)
  (tangent2 : ∀ x, ∃ b, q x - (2 * x - 3) = b * (x - a2) ^ 2)
  (tangent3 : ∀ x, ∃ b, q x - (5 * x + 1) = b * (x - a3) ^ 2) :
  a1 + a2 + a3 = -8 := sorry

end sum_of_tangency_points_l153_153514


namespace solve_inequality_l153_153781

theorem solve_inequality (a : ℝ) (ha_pos : 0 < a) :
  (if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
   else if a = 1 then ∅
   else {x : ℝ | 1 / a < x ∧ x < 1}) =
  {x : ℝ | ax^2 - (a + 1) * x + 1 < 0} :=
by sorry

end solve_inequality_l153_153781


namespace oz_words_lost_l153_153918

theorem oz_words_lost (letters : Fin 64) (forbidden_letter : Fin 64) (h_forbidden : forbidden_letter.val = 6) : 
  let one_letter_words := 64 
  let two_letter_words := 64 * 64
  let one_letter_lost := if letters = forbidden_letter then 1 else 0
  let two_letter_lost := (if letters = forbidden_letter then 64 else 0) + (if letters = forbidden_letter then 64 else 0) 
  1 + two_letter_lost = 129 :=
by
  sorry

end oz_words_lost_l153_153918


namespace gcd_problem_l153_153176

theorem gcd_problem : 
  let a := 690
  let b := 875
  let r1 := 10
  let r2 := 25
  let n1 := a - r1
  let n2 := b - r2
  gcd n1 n2 = 170 :=
by
  sorry

end gcd_problem_l153_153176


namespace find_n_l153_153475

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) = 9) : n = 2 :=
by
  sorry

end find_n_l153_153475


namespace scissor_count_l153_153802

theorem scissor_count :
  let initial_scissors := 54 
  let added_scissors := 22
  let removed_scissors := 15
  initial_scissors + added_scissors - removed_scissors = 61 := by
  sorry

end scissor_count_l153_153802


namespace factorial_division_l153_153207

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l153_153207


namespace total_pages_read_l153_153262

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l153_153262


namespace paul_total_vertical_distance_l153_153143

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l153_153143


namespace rectangle_area_l153_153598

theorem rectangle_area (a b : ℝ) (x : ℝ) 
  (h1 : x^2 + (x / 2)^2 = (a + b)^2) 
  (h2 : x > 0) : 
  x * (x / 2) = (2 * (a + b)^2) / 5 := 
by 
  sorry

end rectangle_area_l153_153598


namespace radius_ratio_l153_153826

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end radius_ratio_l153_153826


namespace connie_remaining_marbles_l153_153074

def initial_marbles : ℕ := 73
def marbles_given : ℕ := 70

theorem connie_remaining_marbles : initial_marbles - marbles_given = 3 := by
  sorry

end connie_remaining_marbles_l153_153074


namespace city_routes_l153_153786

theorem city_routes (h v : ℕ) (H : h = 8) (V : v = 5) : (Nat.choose (h + v) v) = 1287 :=
by
  -- Proof goes here
  sorry

end city_routes_l153_153786


namespace triangle_sum_l153_153948

-- Define the triangle operation
def triangle (a b c : ℕ) : ℕ := a + b + c

-- State the theorem
theorem triangle_sum :
  triangle 2 4 3 + triangle 1 6 5 = 21 :=
by
  sorry

end triangle_sum_l153_153948


namespace sahil_selling_price_l153_153672

-- Define the conditions
def purchased_price := 9000
def repair_cost := 5000
def transportation_charges := 1000
def profit_percentage := 50 / 100

-- Calculate the total cost
def total_cost := purchased_price + repair_cost + transportation_charges

-- Calculate the selling price
def selling_price := total_cost + (profit_percentage * total_cost)

-- The theorem to prove the selling price
theorem sahil_selling_price : selling_price = 22500 :=
by
  -- This is where the proof would go, but we skip it with sorry.
  sorry

end sahil_selling_price_l153_153672


namespace largest_inscribed_rectangle_area_l153_153923

theorem largest_inscribed_rectangle_area : 
  ∀ (width length : ℝ) (a b : ℝ), 
  width = 8 → length = 12 → 
  (a = (8 / Real.sqrt 3) ∧ b = 2 * a) → 
  (area : ℝ) = (12 * (8 - a)) → 
  area = (96 - 32 * Real.sqrt 3) :=
by
  intros width length a b hw hl htr harea
  sorry

end largest_inscribed_rectangle_area_l153_153923


namespace find_a_given_coefficient_l153_153122

theorem find_a_given_coefficient (a : ℝ) (h : (5.choose 2) * ((-1 / 2 : ℝ) ^ 2) * (a ^ 3) = 20) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l153_153122


namespace circle_radius_c_value_l153_153562

theorem circle_radius_c_value (x y c : ℝ) (h₁ : x^2 + 8 * x + y^2 + 10 * y + c = 0) (h₂ : (x+4)^2 + (y+5)^2 = 25) :
  c = -16 :=
by sorry

end circle_radius_c_value_l153_153562


namespace vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l153_153741

open Real

-- Problem 1: Prove the vertex of the parabola is at (1, -a)
theorem vertex_of_parabola (a : ℝ) (h : a ≠ 0) : 
  ∀ x : ℝ, y = a * x^2 - 2 * a * x → (1, -a) = ((1 : ℝ), - a) := 
sorry

-- Problem 2: Prove x_0 = 3 if m = n for given points on the parabola
theorem point_symmetry_on_parabola (a : ℝ) (h : a ≠ 0) (m n : ℝ) :
  m = n → ∀ (x0 : ℝ), y = a * x0 ^ 2 - 2 * a * x0 → x0 = 3 :=
sorry

-- Problem 3: Prove the conditions for y1 < y2 ≤ -a and the range of m
theorem range_of_m (a : ℝ) (h : a < 0) : 
  ∀ (m y1 y2 : ℝ), (y1 < y2) ∧ (y2 ≤ -a) → m < (1 / 2) := 
sorry

end vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l153_153741


namespace picture_distance_l153_153061

theorem picture_distance (wall_width picture_width x y : ℝ)
  (h_wall : wall_width = 25)
  (h_picture : picture_width = 5)
  (h_relation : x = 2 * y)
  (h_total : x + picture_width + y = wall_width) :
  x = 13.34 :=
by
  sorry

end picture_distance_l153_153061


namespace max_min_of_f_find_a_and_theta_l153_153235

noncomputable def f (x θ a : ℝ) : ℝ :=
  Real.sin (x + θ) + a * Real.cos (x + 2 * θ)

theorem max_min_of_f (a θ : ℝ) (h1 : a = Real.sqrt 2) (h2 : θ = π / 4) :
  (∀ x ∈ Set.Icc 0 π, -1 ≤ f x θ a ∧ f x θ a ≤ (Real.sqrt 2) / 2) := sorry

theorem find_a_and_theta (a θ : ℝ) (h1 : f (π / 2) θ a = 0) (h2 : f π θ a = 1) :
  a = -1 ∧ θ = -π / 6 := sorry

end max_min_of_f_find_a_and_theta_l153_153235


namespace smallest_k_for_divisibility_l153_153544

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l153_153544


namespace correct_transformation_l153_153977

theorem correct_transformation (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end correct_transformation_l153_153977


namespace inequality_sinx_plus_y_cosx_plus_y_l153_153526

open Real

theorem inequality_sinx_plus_y_cosx_plus_y (
  y x : ℝ
) (hx : x ∈ Set.Icc (π / 4) (3 * π / 4)) (hy : y ∈ Set.Icc (π / 4) (3 * π / 4)) :
  sin (x + y) + cos (x + y) ≤ sin x + cos x + sin y + cos y :=
sorry

end inequality_sinx_plus_y_cosx_plus_y_l153_153526


namespace part_one_part_two_l153_153134

def M (n : ℤ) : ℤ := n - 3
def M_frac (n : ℚ) : ℚ := - (1 / n^2)

theorem part_one 
    : M 28 * M_frac (1/5) = -1 :=
by {
  sorry
}

theorem part_two 
    : -1 / M 39 / (- M_frac (1/6)) = -1 :=
by {
  sorry
}

end part_one_part_two_l153_153134


namespace cone_rolls_path_l153_153336

theorem cone_rolls_path (r h m n : ℝ) (rotations : ℕ) 
  (h_rotations : rotations = 20)
  (h_ratio : h / r = 3 * Real.sqrt 133)
  (h_m : m = 3)
  (h_n : n = 133) : 
  m + n = 136 := 
by sorry

end cone_rolls_path_l153_153336


namespace coffee_shrinkage_l153_153827

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l153_153827


namespace find_n_l153_153983

-- Definitions based on conditions
def a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
def b (n : ℕ) := 2 * n

-- Theorem stating the problem
theorem find_n (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 10 :=
by sorry

end find_n_l153_153983


namespace nurses_quit_count_l153_153646

-- Initial Definitions
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quit : ℕ := 5
def total_remaining_staff : ℕ := 22

-- Remaining Doctors Calculation
def remaining_doctors : ℕ := initial_doctors - doctors_quit

-- Theorem to prove the number of nurses who quit
theorem nurses_quit_count : initial_nurses - (total_remaining_staff - remaining_doctors) = 2 := by
  sorry

end nurses_quit_count_l153_153646


namespace kim_driving_speed_l153_153927

open Nat
open Real

noncomputable def driving_speed (distance there distance_back time_spent traveling_time total_time: ℝ) : ℝ :=
  (distance + distance_back) / traveling_time

theorem kim_driving_speed:
  ∀ (distance there distance_back time_spent traveling_time total_time: ℝ),
  distance = 30 →
  distance_back = 30 * 1.20 →
  total_time = 2 →
  time_spent = 0.5 →
  traveling_time = total_time - time_spent →
  driving_speed distance there distance_back time_spent traveling_time total_time = 44 :=
by
  intros
  simp only [driving_speed]
  sorry

end kim_driving_speed_l153_153927


namespace area_leq_semiperimeter_l153_153259

-- Definitions
def convex_figure (Φ : set (ℝ × ℝ)) : Prop := 
  convex ℝ Φ ∧ measurable_set Φ

def semiperimeter (Φ : set (ℝ × ℝ)) : ℝ := sorry  -- Needs proper definition for semiperimeter

-- Statement
theorem area_leq_semiperimeter {Φ : set (ℝ × ℝ)} (h1 : convex_figure Φ) 
    (h2 : ∀ (x y : ℤ), ¬ ((↑x, ↑y) ∈ Φ)) :
  measure_theory.measure.volume Φ ≤ semiperimeter Φ := 
sorry

end area_leq_semiperimeter_l153_153259


namespace brenda_cakes_l153_153700

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l153_153700


namespace right_triangle_leg_square_l153_153337

theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = a + 2) : 
  b^2 = 4 * a + 4 := 
by 
  sorry

end right_triangle_leg_square_l153_153337


namespace probability_spade_then_ace_l153_153039

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l153_153039


namespace length_of_AC_l153_153097

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l153_153097


namespace find_u_given_roots_of_quadratic_l153_153746

-- Define the initial conditions
variables (k l : ℝ)
variable (r1 r2 : ℝ)
hypothesis roots_initial : ∀ x, (x = r1 ∨ x = r2) ↔ x^2 + k*x + l = 0

-- Define the proof statement
theorem find_u_given_roots_of_quadratic :
  (r1 + r2 = -k) →
  (r1 * r2 = l) →
  ∃ u : ℝ, ∀ x, (x = r1^2 ∨ x = r2^2) ↔ x^2 + u*x + v = 0 ∧ u = -k^2 + 2*l :=
by
  intros h1 h2
  exists -k^2 + 2*l
  intros x
  split
  sorry

end find_u_given_roots_of_quadratic_l153_153746


namespace blue_pill_cost_l153_153763

theorem blue_pill_cost (y : ℕ) :
  -- Conditions
  (∀ t d : ℕ, t = 21 → 
     d = 14 → 
     (735 - d * 2 = t * ((2 * y) + (y + 2)) / t) →
     2 * y + (y + 2) = 35) →
  -- Conclusion
  y = 11 :=
by
  sorry

end blue_pill_cost_l153_153763


namespace probability_sequence_correct_l153_153659

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l153_153659


namespace sqrt_expression_non_negative_l153_153745

theorem sqrt_expression_non_negative (x : ℝ) : 4 + 2 * x ≥ 0 ↔ x ≥ -2 :=
by sorry

end sqrt_expression_non_negative_l153_153745


namespace patternD_cannot_form_pyramid_l153_153849

-- Define the patterns
inductive Pattern
| A
| B
| C
| D

-- Define the condition for folding into a pyramid with a square base
def canFormPyramidWithSquareBase (p : Pattern) : Prop :=
  p = Pattern.A ∨ p = Pattern.B ∨ p = Pattern.C

-- Goal: Prove that Pattern D cannot be folded into a pyramid with a square base
theorem patternD_cannot_form_pyramid : ¬ canFormPyramidWithSquareBase Pattern.D :=
by
  -- Need to provide the proof here
  sorry

end patternD_cannot_form_pyramid_l153_153849


namespace consecutive_page_sum_l153_153963

theorem consecutive_page_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 285 := by
  sorry

end consecutive_page_sum_l153_153963


namespace geometric_mean_of_4_and_9_l153_153030

theorem geometric_mean_of_4_and_9 : ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := 
by
  sorry

end geometric_mean_of_4_and_9_l153_153030


namespace optimal_position_theorem_l153_153093

noncomputable def optimal_position (a b a1 b1 : ℝ) : ℝ :=
  (b / 2) + (b1 / (2 * a1)) * (a - a1)

theorem optimal_position_theorem 
  (a b a1 b1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) :
  ∃ x, x = optimal_position a b a1 b1 := by
  sorry

end optimal_position_theorem_l153_153093


namespace destroyed_cakes_l153_153704

theorem destroyed_cakes (initial_cakes : ℕ) (half_falls : ℕ) (half_saved : ℕ)
  (h1 : initial_cakes = 12)
  (h2 : half_falls = initial_cakes / 2)
  (h3 : half_saved = half_falls / 2) :
  initial_cakes - half_falls / 2 = 3 :=
by
  sorry

end destroyed_cakes_l153_153704


namespace smallest_k_divides_l153_153538

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l153_153538


namespace score_order_l153_153251

theorem score_order (a b c d : ℕ) 
  (h1 : b + d = a + c)
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := 
by
  sorry

end score_order_l153_153251


namespace least_four_digit_with_factors_3_5_7_l153_153469

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l153_153469


namespace binary_mul_1101_111_eq_1001111_l153_153534

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l153_153534


namespace circle_equation_l153_153965

theorem circle_equation (x y : ℝ) (h_eq : x = 0) (k_eq : y = -2) (r_eq : y = 4) :
  (x - 0)^2 + (y - (-2))^2 = 16 := 
by
  sorry

end circle_equation_l153_153965


namespace train_crossing_time_l153_153832

/-- 
Prove that the time it takes for a train traveling at 90 kmph with a length of 100.008 meters to cross a pole is 4.00032 seconds.
-/
theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 90 → length_meters = 100.008 → (length_meters / (speed_kmph * (1000 / 3600))) = 4.00032 :=
by
  intros h1 h2
  sorry

end train_crossing_time_l153_153832


namespace scientific_notation_correct_l153_153688

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l153_153688


namespace sisterPassesMeInOppositeDirection_l153_153348

noncomputable def numberOfPasses (laps_sister : ℕ) : ℕ :=
if laps_sister > 1 then 2 * laps_sister else 0

theorem sisterPassesMeInOppositeDirection
  (my_laps : ℕ) (laps_sister : ℕ) (passes_in_same_direction : ℕ) :
  my_laps = 1 ∧ passes_in_same_direction = 2 ∧ laps_sister > 1 →
  passes_in_same_direction * 2 = 4 :=
by intros; sorry

end sisterPassesMeInOppositeDirection_l153_153348


namespace distance_from_origin_to_midpoint_l153_153973

theorem distance_from_origin_to_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10) → (y1 = 20) → (x2 = -10) → (y2 = -20) → 
  dist (0 : ℝ × ℝ) ((x1 + x2) / 2, (y1 + y2) / 2) = 0 := 
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- remaining proof goes here
  sorry

end distance_from_origin_to_midpoint_l153_153973


namespace marys_next_birthday_l153_153016

theorem marys_next_birthday (d s m : ℝ) (h1 : s = 0.7 * d) (h2 : m = 1.3 * s) (h3 : m + s + d = 25.2) : m + 1 = 9 :=
by
  sorry

end marys_next_birthday_l153_153016


namespace functional_eq_unique_solution_l153_153713

theorem functional_eq_unique_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_eq_unique_solution_l153_153713


namespace expenditure_july_l153_153835

def avg_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def avg_expenditure_feb_to_jul : ℝ := 4250

theorem expenditure_july 
  (avg_expenditure_jan_to_jun : ℝ) 
  (expenditure_january : ℝ) 
  (avg_expenditure_feb_to_jul : ℝ) :
  let expenditure_feb_to_jun := 6 * avg_expenditure_jan_to_jun - expenditure_january,
      expenditure_feb_to_jul := 6 * avg_expenditure_feb_to_jul in
  expenditure_feb_to_jul - expenditure_feb_to_jun = 1500 :=
by
  sorry

end expenditure_july_l153_153835


namespace expression_equals_36_l153_153586

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l153_153586


namespace f_comp_g_eq_g_comp_f_has_solution_l153_153937

variable {R : Type*} [Field R]

def f (a b x : R) : R := a * x + b
def g (c d x : R) : R := c * x ^ 2 + d

theorem f_comp_g_eq_g_comp_f_has_solution (a b c d : R) :
  (∃ x : R, f a b (g c d x) = g c d (f a b x)) ↔ (c = 0 ∨ a * b = 0) ∧ (a * d - c * b ^ 2 + b - d = 0) := by
  sorry

end f_comp_g_eq_g_comp_f_has_solution_l153_153937


namespace problem_l153_153979

-- Define what it means to be a factor or divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a
def is_divisor (a b : ℕ) : Prop := a ∣ b

-- The specific problem conditions
def statement_A := is_factor 4 28
def statement_B := is_divisor 19 209 ∧ ¬ is_divisor 19 57
def statement_C := ¬ is_divisor 30 90 ∧ ¬ is_divisor 30 76
def statement_D := is_divisor 14 28 ∧ ¬ is_divisor 14 56
def statement_E := is_factor 9 162

-- The proof problem
theorem problem : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D ∧ statement_E :=
by 
  -- You would normally provide the proof here
  sorry

end problem_l153_153979


namespace shrink_ray_coffee_l153_153830

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l153_153830


namespace solve_for_w_l153_153050

theorem solve_for_w (w : ℝ) : (2 : ℝ)^(2 * w) = (8 : ℝ)^(w - 4) → w = 12 := by
  sorry

end solve_for_w_l153_153050


namespace lowest_test_score_dropped_l153_153003

theorem lowest_test_score_dropped (S L : ℕ)
  (h1 : S = 5 * 42) 
  (h2 : S - L = 4 * 48) : 
  L = 18 :=
by
  sorry

end lowest_test_score_dropped_l153_153003


namespace smallest_k_l153_153557

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l153_153557


namespace blue_socks_count_l153_153765

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l153_153765


namespace calculate_expression_l153_153073

theorem calculate_expression :
  |1 - Real.sqrt 2| + (1/2)^(-2 : ℤ) - (Real.pi - 2023)^0 = Real.sqrt 2 + 2 := 
by
  sorry

end calculate_expression_l153_153073


namespace find_f_l153_153113

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 + x) :
  ∀ x : ℤ, f x = x^2 - x :=
by
  intro x
  sorry

end find_f_l153_153113


namespace tan_ratio_proof_l153_153008

noncomputable def tan_ratio (a b : ℝ) : ℝ := Real.tan a / Real.tan b

theorem tan_ratio_proof (a b : ℝ) (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 3) : 
tan_ratio a b = 23 / 7 := by
  sorry

end tan_ratio_proof_l153_153008


namespace eggs_eaten_in_afternoon_l153_153349

theorem eggs_eaten_in_afternoon (initial : ℕ) (morning : ℕ) (final : ℕ) (afternoon : ℕ) :
  initial = 20 → morning = 4 → final = 13 → afternoon = initial - morning - final → afternoon = 3 :=
by
  intros h_initial h_morning h_final h_afternoon
  rw [h_initial, h_morning, h_final] at h_afternoon
  linarith

end eggs_eaten_in_afternoon_l153_153349


namespace least_four_digit_with_factors_l153_153467

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l153_153467


namespace markers_needed_total_l153_153491

noncomputable def markers_needed_first_group : ℕ := 10 * 2
noncomputable def markers_needed_second_group : ℕ := 15 * 4
noncomputable def students_last_group : ℕ := 30 - (10 + 15)
noncomputable def markers_needed_last_group : ℕ := students_last_group * 6

theorem markers_needed_total : markers_needed_first_group + markers_needed_second_group + markers_needed_last_group = 110 :=
by
  sorry

end markers_needed_total_l153_153491


namespace container_capacity_l153_153055

theorem container_capacity 
  (C : ℝ)
  (h1 : 0.75 * C - 0.30 * C = 45) :
  C = 100 := by
  sorry

end container_capacity_l153_153055


namespace problem1_problem2_l153_153634

noncomputable def problem1_solution1 : ℝ := (2 + Real.sqrt 6) / 2
noncomputable def problem1_solution2 : ℝ := (2 - Real.sqrt 6) / 2

theorem problem1 (x : ℝ) : 
  (2 * x ^ 2 - 4 * x - 1 = 0) ↔ (x = problem1_solution1 ∨ x = problem1_solution2) :=
by
  sorry

theorem problem2 : 
  (4 * (x + 2) ^ 2 - 9 * (x - 3) ^ 2 = 0) ↔ (x = 1 ∨ x = 13) :=
by
  sorry

end problem1_problem2_l153_153634


namespace solve_triples_l153_153217

theorem solve_triples 
  (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_expr_int : ∃ k : ℤ, (a + b) ^ 4 / c + (b + c) ^ 4 / a + (c + a) ^ 4 / b = k) 
  (h_prime : Nat.Prime (a + b + c)) : 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 6 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 6 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1) := sorry

end solve_triples_l153_153217


namespace athletes_teams_l153_153258

-- Define the type representing athletes
def Athlete : Type := Fin 10

-- Define the two specific athletes A and B
def A : Athlete := 0
def B : Athlete := 1

-- The theorem we want to prove
theorem athletes_teams (hAB : A ≠ B) : 
  ∃ (teams : Finset (Fin 10) × Finset (Fin 10)), 
    (teams.1.card = 5 ∧ teams.2.card = 5 ∧
     teams.1 ∩ teams.2 = ∅ ∧ teams.1 ∪ teams.2 = Finset.univ ∧
     (A ∈ teams.1 ∧ B ∈ teams.2) ∨ (A ∈ teams.2 ∧ B ∈ teams.1) ∧
     teams.1.choose 4 = 70) := sorry

end athletes_teams_l153_153258


namespace find_m_given_a3_eq_40_l153_153723

theorem find_m_given_a3_eq_40 (m : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 - m * x) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_3 = 40 →
  m = -1 := 
by 
  sorry

end find_m_given_a3_eq_40_l153_153723


namespace polynomial_identity_l153_153595

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l153_153595


namespace right_triangle_perimeter_5_shortest_altitude_1_l153_153197

-- Definition of a right-angled triangle's sides with given perimeter and altitude
def right_angled_triangle (a b c : ℚ) : Prop :=
a^2 + b^2 = c^2 ∧ a + b + c = 5 ∧ a * b = c

-- Statement of the theorem to prove the side lengths of the triangle
theorem right_triangle_perimeter_5_shortest_altitude_1 :
  ∃ (a b c : ℚ), right_angled_triangle a b c ∧ (a = 5 / 3 ∧ b = 5 / 4 ∧ c = 25 / 12) ∨ (a = 5 / 4 ∧ b = 5 / 3 ∧ c = 25 / 12) :=
by
  sorry

end right_triangle_perimeter_5_shortest_altitude_1_l153_153197


namespace smallest_integer_odd_sequence_l153_153790

/-- Given the median of a set of consecutive odd integers is 157 and the greatest integer in the set is 171,
    prove that the smallest integer in the set is 149. -/
theorem smallest_integer_odd_sequence (median greatest : ℤ) (h_median : median = 157) (h_greatest : greatest = 171) :
  ∃ smallest : ℤ, smallest = 149 :=
by
  sorry

end smallest_integer_odd_sequence_l153_153790


namespace triangle_side_eq_median_l153_153921

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l153_153921


namespace race_dead_heat_l153_153816

theorem race_dead_heat (va vb D : ℝ) (hva_vb : va = (15 / 16) * vb) (dist_a : D = D) (dist_b : D = (15 / 16) * D) (race_finish : D / va = (15 / 16) * D / vb) :
  va / vb = 15 / 16 :=
by sorry

end race_dead_heat_l153_153816


namespace find_angle_y_l153_153530

theorem find_angle_y (angle_ABC angle_ABD angle_ADB y : ℝ)
  (h1 : angle_ABC = 115)
  (h2 : angle_ABD = 180 - angle_ABC)
  (h3 : angle_ADB = 30)
  (h4 : angle_ABD + angle_ADB + y = 180) :
  y = 85 := 
sorry

end find_angle_y_l153_153530


namespace exists_a_l153_153748

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a_l153_153748


namespace swap_values_l153_153432

theorem swap_values : ∀ (a b : ℕ), a = 3 → b = 2 → 
  (∃ c : ℕ, c = b ∧ (b = a ∧ (a = c ∨ a = 2 ∧ b = 3))) :=
by
  sorry

end swap_values_l153_153432


namespace Expected_and_Variance_l153_153565

variables (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

def P (xi : ℕ) : ℝ := 
  if xi = 0 then p else if xi = 1 then 1 - p else 0

def E_xi : ℝ := 0 * P p 0 + 1 * P p 1

def D_xi : ℝ := (0 - E_xi p)^2 * P p 0 + (1 - E_xi p)^2 * P p 1

theorem Expected_and_Variance :
  (E_xi p = 1 - p) ∧ (D_xi p = p * (1 - p)) :=
sorry

end Expected_and_Variance_l153_153565


namespace range_of_x_l153_153739

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h : f (x^2 - 4) < 2) : 
  (-Real.sqrt 5 < x ∧ x < -2) ∨ (2 < x ∧ x < Real.sqrt 5) :=
sorry

end range_of_x_l153_153739


namespace large_bucket_capacity_l153_153996

variables (S L : ℝ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
by sorry

end large_bucket_capacity_l153_153996


namespace smallest_k_divides_l153_153549

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l153_153549


namespace arithmetic_sequence_a4_eight_l153_153400

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 2 - a 1)

variable {a : ℕ → ℤ}

theorem arithmetic_sequence_a4_eight (h_arith_sequence : arithmetic_sequence a)
    (h_cond : a 3 + a 5 = 16) : a 4 = 8 :=
by
  sorry

end arithmetic_sequence_a4_eight_l153_153400


namespace k_squared_minus_3k_minus_4_l153_153098

theorem k_squared_minus_3k_minus_4 (a b c d k : ℚ)
  (h₁ : (2 * a) / (b + c + d) = k)
  (h₂ : (2 * b) / (a + c + d) = k)
  (h₃ : (2 * c) / (a + b + d) = k)
  (h₄ : (2 * d) / (a + b + c) = k) :
  k^2 - 3 * k - 4 = -50 / 9 ∨ k^2 - 3 * k - 4 = 6 :=
  sorry

end k_squared_minus_3k_minus_4_l153_153098


namespace maximum_area_of_triangle_l153_153267

theorem maximum_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hC : C = π / 6) (hSum : a + b = 12) :
  ∃ (S : ℝ), S = 9 ∧ ∀ S', S' ≤ S := 
sorry

end maximum_area_of_triangle_l153_153267


namespace smallest_distance_l153_153271

open Complex

variable (z w : ℂ)

def a : ℂ := -2 - 4 * I
def b : ℂ := 5 + 6 * I

-- Conditions
def cond1 : Prop := abs (z + 2 + 4 * I) = 2
def cond2 : Prop := abs (w - 5 - 6 * I) = 4

-- Problem
theorem smallest_distance (h1 : cond1 z) (h2 : cond2 w) : abs (z - w) = Real.sqrt 149 - 6 :=
sorry

end smallest_distance_l153_153271


namespace triangle_side_length_l153_153919

theorem triangle_side_length (A B C M : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 3)
  (hMidM : M = midpoint B C)
  (hAM_BC : dist A M = dist B C) :
  dist B C = Real.sqrt (78) / 3 :=
by
  sorry

end triangle_side_length_l153_153919


namespace smallest_y_value_l153_153044

noncomputable def f (y : ℝ) : ℝ := 3 * y ^ 2 + 27 * y - 90
noncomputable def g (y : ℝ) : ℝ := y * (y + 15)

theorem smallest_y_value (y : ℝ) : (∀ y, f y = g y → y ≠ -9) → false := by
  sorry

end smallest_y_value_l153_153044


namespace sufficient_condition_for_inequality_l153_153815

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  sorry

end sufficient_condition_for_inequality_l153_153815


namespace rebecca_perm_charge_l153_153944

theorem rebecca_perm_charge :
  ∀ (P : ℕ), (4 * 30 + 2 * 60 - 2 * 10 + P + 50 = 310) -> P = 40 :=
by
  intros P h
  sorry

end rebecca_perm_charge_l153_153944


namespace coefficient_of_x5_in_expansion_l153_153158

-- Define the polynomial expansion of (x-1)(x+1)^8
def polynomial_expansion (x : ℚ) : ℚ :=
  (x - 1) * (x + 1) ^ 8

-- Define the binomial coefficient function
def binom_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem: The coefficient of x^5 in the expansion of (x-1)(x+1)^8 is 14
theorem coefficient_of_x5_in_expansion :
  binom_coeff 8 4 - binom_coeff 8 5 = 14 :=
sorry

end coefficient_of_x5_in_expansion_l153_153158


namespace trapezium_parallel_side_length_l153_153864

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l153_153864


namespace sum_squares_l153_153967

theorem sum_squares (a b c : ℝ) (h1 : a + b + c = 22) (h2 : a * b + b * c + c * a = 116) : 
  (a^2 + b^2 + c^2 = 252) :=
by
  sorry

end sum_squares_l153_153967


namespace ratio_books_Pete_Matt_l153_153146

-- Definitions for the number of books read by Pete and Matt last year.
variable {P M : ℕ}

-- Conditions
variable (h1 : 3 * P = 300) -- Pete read 300 books in total over the two years.
variable (h2 : 3 / 2 * M = 75) -- Matt read 75 books in his second year, which is 50% more than last year.

-- Proof statement
theorem ratio_books_Pete_Matt : (P : ℚ) / M = 2 :=
by 
  sorry

end ratio_books_Pete_Matt_l153_153146


namespace smallest_k_l153_153543

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l153_153543


namespace determine_a_l153_153728

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l153_153728


namespace range_of_a_l153_153112

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1
noncomputable def f' (a x : ℝ) : ℝ := x^2 - a * x + a - 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f' a x ≤ 0) ∧ (∀ x, 6 < x → f' a x ≥ 0) ↔ 5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end range_of_a_l153_153112


namespace trapezium_parallel_side_length_l153_153862

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l153_153862


namespace option_c_correct_l153_153047

theorem option_c_correct (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end option_c_correct_l153_153047


namespace johns_share_l153_153175

theorem johns_share (total_amount : ℕ) (r1 r2 r3 : ℕ) (h : total_amount = 6000) (hr1 : r1 = 2) (hr2 : r2 = 4) (hr3 : r3 = 6) :
  let total_ratio := r1 + r2 + r3
  let johns_ratio := r1
  let johns_share := (johns_ratio * total_amount) / total_ratio
  johns_share = 1000 :=
by
  sorry

end johns_share_l153_153175


namespace complex_calculation_l153_153072

theorem complex_calculation (i : ℂ) (hi : i * i = -1) : (1 - i)^2 * i = 2 :=
by
  sorry

end complex_calculation_l153_153072


namespace total_area_of_folded_blankets_l153_153852

-- Define the initial conditions
def initial_area : ℕ := 8 * 8
def folds : ℕ := 4
def num_blankets : ℕ := 3

-- Define the hypothesis about folding
def folded_area (initial_area : ℕ) (folds : ℕ) : ℕ :=
  initial_area / (2 ^ folds)

-- The total area of all folded blankets
def total_folded_area (initial_area : ℕ) (folds : ℕ) (num_blankets : ℕ) : ℕ :=
  num_blankets * folded_area initial_area folds

-- The theorem we want to prove
theorem total_area_of_folded_blankets : total_folded_area initial_area folds num_blankets = 12 := by
  sorry

end total_area_of_folded_blankets_l153_153852


namespace evaluate_expression_is_41_l153_153083

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l153_153083


namespace problem1_problem2_l153_153572

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Prove that for a = 1/2, A ∩ B = { x | 0 < x ∧ x < 1 }
theorem problem1 : setA (1/2) ∩ setB = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

-- Prove that if A ∩ B = ∅, then a ≤ -1/2 or a ≥ 2
theorem problem2 (a : ℝ) (h : setA a ∩ setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by
  sorry

end problem1_problem2_l153_153572


namespace option_d_true_l153_153898

theorem option_d_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr_qr : p * r < q * r) : 1 > q / p :=
sorry

end option_d_true_l153_153898


namespace table_length_is_77_l153_153335

theorem table_length_is_77 :
  ∀ (x : ℝ), 
  (∀ (y : ℝ), 
  (x >= 0) ∧ 
  (y = 80) ∧ 
  (∀ (w : ℝ), (w = 8)) ∧ 
  (∀ (h : ℝ), (h = 5)) ∧ 
  (∀ (i : ℕ), 
    (i₀ := 0) ∧ 
    (j₀ := 0) ∧ 
    (∀ i, w + i * 1 = y) ∧ 
    (∀ i, h + i * 1 = x) ∧ 
    (i = 72))) → 
  (x = 77) :=
by
  intros
  sorry

end table_length_is_77_l153_153335


namespace scientific_notation_correct_l153_153687

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l153_153687


namespace sin_2theta_eq_neg_one_half_max_value_of_m3_squared_plus_n_squared_l153_153101

-- Define points A, B, and C with given coordinates
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (1, -1)
def C (θ : ℝ) : ℝ × ℝ := (sqrt 2 * cos θ, sqrt 2 * sin θ)

-- Define vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- Norm of a vector
def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem 1: Prove that sin 2θ = -1/2 given the conditions
theorem sin_2theta_eq_neg_one_half (θ : ℝ) 
  (h : norm (vec B (C θ) - vec B A) = sqrt 2) : 
  sin (2 * θ) = -1 / 2 := 
sorry

-- Problem 2: Prove the maximum value of (m-3)^2 + n^2 is 16 given the conditions
theorem max_value_of_m3_squared_plus_n_squared 
  (θ θ_real : ℝ) 
  (m n : ℝ)
  (h : m * (vec 0 A).1 + n * (vec 0 B).1 = (vec 0 (C θ)).1 
       ∧ m * (vec 0 A).2 + n * (vec 0 B).2 = (vec 0 (C θ)).2) : 
  ∃ m n : ℝ, 
  (m - 3) ^ 2 + n ^ 2 = 16 :=
sorry

end sin_2theta_eq_neg_one_half_max_value_of_m3_squared_plus_n_squared_l153_153101


namespace diagonal_of_rectangular_solid_l153_153430

-- Define the lengths of the edges
def a : ℝ := 2
def b : ℝ := 3
def c : ℝ := 4

-- Prove that the diagonal of the rectangular solid with edges a, b, and c is sqrt(29)
theorem diagonal_of_rectangular_solid (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : (a^2 + b^2 + c^2) = 29 := 
by 
  rw [h1, h2, h3]
  norm_num

end diagonal_of_rectangular_solid_l153_153430


namespace math_problem_l153_153225

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l153_153225


namespace min_value_pm_pn_l153_153219

theorem min_value_pm_pn (x y : ℝ)
  (h : x ^ 2 - y ^ 2 / 3 = 1) 
  (hx : 1 ≤ x) : (8 * x - 3) = 5 :=
sorry

end min_value_pm_pn_l153_153219


namespace b_eq_6_l153_153272

theorem b_eq_6 (a b : ℤ) (h₁ : |a| = 1) (h₂ : ∀ x : ℝ, a * x^2 - 2 * x - b + 5 = 0 → x < 0) : b = 6 := 
by
  sorry

end b_eq_6_l153_153272


namespace probability_calculation_l153_153163

noncomputable def probability_of_event_A : ℚ := 
  let total_ways := 35 
  let favorable_ways := 6 
  favorable_ways / total_ways

theorem probability_calculation (A_team B_team : Type) [Fintype A_team] [Fintype B_team] [DecidableEq A_team] [DecidableEq B_team] :
  let total_players := 7 
  let selected_players := 4 
  let seeded_A := 2 
  let nonseeded_A := 1 
  let seeded_B := 2 
  let nonseeded_B := 2 
  let event_total_ways := Nat.choose total_players selected_players 
  let event_A_ways := Nat.choose seeded_A 2 * Nat.choose nonseeded_A 2 + Nat.choose seeded_B 2 * Nat.choose nonseeded_B 2 
  probability_of_event_A = 6 / 35 := 
sorry

end probability_calculation_l153_153163


namespace equal_divide_remaining_amount_all_girls_l153_153638

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l153_153638


namespace probability_of_specific_sequence_l153_153662

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l153_153662


namespace flour_needed_for_dozen_cookies_l153_153777

/--
Matt uses 4 bags of flour, each weighing 5 pounds, to make a total of 120 cookies.
Prove that 2 pounds of flour are needed to make a dozen cookies.
-/
theorem flour_needed_for_dozen_cookies :
  ∀ (bags_of_flour : ℕ) (weight_per_bag : ℕ) (total_cookies : ℕ),
  bags_of_flour = 4 →
  weight_per_bag = 5 →
  total_cookies = 120 →
  (12 * (bags_of_flour * weight_per_bag)) / total_cookies = 2 :=
by
  sorry

end flour_needed_for_dozen_cookies_l153_153777


namespace min_value_x1x2_squared_inequality_ab_l153_153571

def D : Set (ℝ × ℝ) := 
  { p | ∃ x1 x2, p = (x1, x2) ∧ x1 + x2 = 2 ∧ x1 > 0 ∧ x2 > 0 }

-- Part 1: Proving the minimum value of x1^2 + x2^2 in set D is 2
theorem min_value_x1x2_squared (x1 x2 : ℝ) (h : (x1, x2) ∈ D) : 
  x1^2 + x2^2 ≥ 2 := 
sorry

-- Part 2: Proving the inequality for any (a, b) in set D
theorem inequality_ab (a b : ℝ) (h : (a, b) ∈ D) : 
  (1 / (a + 2 * b) + 1 / (2 * a + b)) ≥ (2 / 3) := 
sorry

end min_value_x1x2_squared_inequality_ab_l153_153571


namespace only_PropositionB_is_correct_l153_153117

-- Define propositions as functions for clarity
def PropositionA (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : Prop :=
  (1 / a) > (1 / b)

def PropositionB (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : Prop :=
  a ^ 3 < a

def PropositionC (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  (b + 1) / (a + 1) < b / a

def PropositionD (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : Prop :=
  c * b^2 < a * b^2

-- The main theorem stating that the only correct proposition is Proposition B
theorem only_PropositionB_is_correct :
  (∀ a b : ℝ, (a * b ≠ 0 ∧ a < b → ¬ PropositionA a b (a * b ≠ 0) (a < b))) ∧
  (∀ a : ℝ, (0 < a ∧ a < 1 → PropositionB a (0 < a) (a < 1))) ∧
  (∀ a b : ℝ, (a > b ∧ b > 0 → ¬ PropositionC a b (a > b) (b > 0))) ∧
  (∀ a b c : ℝ, (c < b ∧ b < a ∧ a * c < 0 → ¬ PropositionD a b c (c < b) (b < a) (a * c < 0))) :=
by
  -- Proof of the theorem
  sorry

end only_PropositionB_is_correct_l153_153117


namespace count_integer_solutions_l153_153893

theorem count_integer_solutions : 
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ s) ↔ (x^3 + y^2 = 2*y + 1)) ∧ 
  s.card = 3 := 
by
  sorry

end count_integer_solutions_l153_153893


namespace same_terminal_side_l153_153938

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180) % 360 = ((4 * k + 1) * 180) % 360 ∨ ((2 * k + 1) * 180) % 360 = ((4 * k - 1) * 180) % 360 := 
sorry

end same_terminal_side_l153_153938


namespace trajectory_equation_circle_equation_l153_153129

-- Define the variables
variables {x y r : ℝ}

-- Prove the trajectory equation of the circle center P
theorem trajectory_equation (h1 : x^2 + r^2 = 2) (h2 : y^2 + r^2 = 3) : y^2 - x^2 = 1 :=
sorry

-- Prove the equation of the circle P given the distance to the line y = x
theorem circle_equation (h : (|x - y| / Real.sqrt 2) = (Real.sqrt 2) / 2) : 
  (x = y + 1 ∨ x = y - 1) → 
  ((y + 1)^2 + x^2 = 3 ∨ (y - 1)^2 + x^2 = 3) :=
sorry

end trajectory_equation_circle_equation_l153_153129


namespace neg_prop_true_l153_153449

theorem neg_prop_true (a : ℝ) :
  ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) → ∃ a : ℝ, a > 2 ∧ a^2 ≥ 4 :=
by
  intros h
  sorry

end neg_prop_true_l153_153449


namespace price_of_scooter_l153_153417

-- Assume upfront_payment and percentage_upfront are given
def upfront_payment : ℝ := 240
def percentage_upfront : ℝ := 0.20

noncomputable
def total_price (upfront_payment : ℝ) (percentage_upfront : ℝ) : ℝ :=
  (upfront_payment / percentage_upfront)

theorem price_of_scooter : total_price upfront_payment percentage_upfront = 1200 :=
  by
    sorry

end price_of_scooter_l153_153417


namespace terminal_side_in_second_quadrant_l153_153744

theorem terminal_side_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
    0 < α ∧ α < π :=
sorry

end terminal_side_in_second_quadrant_l153_153744


namespace initial_water_amount_l153_153992

theorem initial_water_amount 
  (W : ℝ) 
  (evap_rate : ℝ) 
  (days : ℕ) 
  (percentage_evaporated : ℝ) 
  (evap_rate_eq : evap_rate = 0.012) 
  (days_eq : days = 50) 
  (percentage_evaporated_eq : percentage_evaporated = 0.06) 
  (total_evaporated_eq : evap_rate * days = 0.6) 
  (percentage_condition : percentage_evaporated * W = evap_rate * days) 
  : W = 10 := 
  by sorry

end initial_water_amount_l153_153992


namespace total_amount_spent_l153_153282

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end total_amount_spent_l153_153282


namespace mean_after_removal_l153_153437

variable {n : ℕ}
variable {S : ℝ}
variable {S' : ℝ}
variable {mean_original : ℝ}
variable {size_original : ℕ}
variable {x1 : ℝ}
variable {x2 : ℝ}

theorem mean_after_removal (h_mean_original : mean_original = 42)
    (h_size_original : size_original = 60)
    (h_x1 : x1 = 50)
    (h_x2 : x2 = 60)
    (h_S : S = mean_original * size_original)
    (h_S' : S' = S - (x1 + x2)) :
    S' / (size_original - 2) = 41.55 :=
by
  sorry

end mean_after_removal_l153_153437


namespace num_valid_colorings_l153_153404

namespace ColoringGrid

-- Definition of the grid and the constraint.
-- It's easier to represent with simply 9 nodes and adjacent constraints, however,
-- we will declare the conditions and result as discussed.

def Grid := Fin 3 × Fin 3
def Colors := Fin 2

-- Define adjacency relationship
def adjacent (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Condition stating no two adjacent squares can share the same color
def valid_coloring (f : Grid → Colors) : Prop :=
  ∀ a b : Grid, adjacent a b → f a ≠ f b

-- The main theorem stating the number of valid colorings
theorem num_valid_colorings : ∃ (n : ℕ), n = 2 ∧ ∀ (f : Grid → Colors), valid_coloring f → n = 2 :=
by sorry

end ColoringGrid

end num_valid_colorings_l153_153404


namespace value_of_expression_l153_153376

theorem value_of_expression (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 53) :
  x^3 - y^3 - 2 * (x + y) + 10 = 2011 :=
sorry

end value_of_expression_l153_153376


namespace boys_from_clay_l153_153928

theorem boys_from_clay (total_students jonas_students clay_students pine_students total_boys total_girls : ℕ)
  (jonas_girls pine_boys : ℕ) 
  (H1 : total_students = 150)
  (H2 : jonas_students = 50)
  (H3 : clay_students = 60)
  (H4 : pine_students = 40)
  (H5 : total_boys = 80)
  (H6 : total_girls = 70)
  (H7 : jonas_girls = 30)
  (H8 : pine_boys = 15):
  ∃ (clay_boys : ℕ), clay_boys = 45 :=
by
  have jonas_boys : ℕ := jonas_students - jonas_girls
  have boys_from_clay := total_boys - pine_boys - jonas_boys
  exact ⟨boys_from_clay, by sorry⟩

end boys_from_clay_l153_153928


namespace ahn_largest_number_l153_153344

def largest_number_ahn_can_get : ℕ :=
  let n := 10
  2 * (200 - n)

theorem ahn_largest_number :
  (10 ≤ 99) →
  (10 ≤ 99) →
  largest_number_ahn_can_get = 380 := 
by
-- Conditions: n is a two-digit integer with range 10 ≤ n ≤ 99
-- Proof is skipped
  sorry

end ahn_largest_number_l153_153344


namespace solve_system_of_inequalities_l153_153289

theorem solve_system_of_inequalities 
  (x : ℝ) 
  (h1 : x - 3 * (x - 2) ≥ 4)
  (h2 : (1 + 2 * x) / 3 > x - 1) : 
  x ≤ 1 := 
sorry

end solve_system_of_inequalities_l153_153289


namespace cannot_form_figureB_l153_153298

-- Define the pieces as terms
inductive Piece
| square : Piece
| rectangle : Π (h w : ℕ), Piece   -- h: height, w: width

-- Define the available pieces in a list (assuming these are predefined somewhere)
def pieces : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

-- Define the figures that can be formed
def figureA : List Piece := [Piece.square, Piece.square, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

def figureC : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, 
                             Piece.square, Piece.square]

def figureD : List Piece := [Piece.rectangle 2 2, Piece.square, Piece.square, Piece.square,
                              Piece.square]

def figureE : List Piece := [Piece.rectangle 3 1, Piece.square, Piece.square, Piece.square]

-- Define the figure B that we need to prove cannot be formed
def figureB : List Piece := [Piece.rectangle 5 1, Piece.square, Piece.square, Piece.square,
                              Piece.square]

theorem cannot_form_figureB :
  ¬(∃ arrangement : List Piece, arrangement ⊆ pieces ∧ arrangement = figureB) :=
sorry

end cannot_form_figureB_l153_153298


namespace discount_difference_l153_153187

theorem discount_difference (p : ℝ) (single_discount first_discount second_discount : ℝ) :
    p = 12000 →
    single_discount = 0.45 →
    first_discount = 0.35 →
    second_discount = 0.10 →
    (p * (1 - single_discount) - p * (1 - first_discount) * (1 - second_discount) = 420) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end discount_difference_l153_153187


namespace cone_height_l153_153492

theorem cone_height (S h H Vcone Vcylinder : ℝ)
  (hcylinder_height : H = 9)
  (hvolumes : Vcone = Vcylinder)
  (hbase_areas : S = S)
  (hV_cone : Vcone = (1 / 3) * S * h)
  (hV_cylinder : Vcylinder = S * H) : h = 27 :=
by
  -- sorry is used here to indicate missing proof steps which are predefined as unnecessary
  sorry

end cone_height_l153_153492


namespace floor_width_l153_153677

theorem floor_width (W : ℕ) (hAreaFloor: 10 * W - 64 = 16) : W = 8 :=
by
  -- the proof should be added here
  sorry

end floor_width_l153_153677


namespace jonathan_needs_more_money_l153_153004

def cost_dictionary : ℕ := 11
def cost_dinosaur_book : ℕ := 19
def cost_childrens_cookbook : ℕ := 7
def saved_money : ℕ := 8

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
def amount_needed : ℕ := total_cost - saved_money

theorem jonathan_needs_more_money : amount_needed = 29 := by
  have h1 : total_cost = 37 := by
    show 11 + 19 + 7 = 37
    sorry
  show 37 - 8 = 29
  sorry

end jonathan_needs_more_money_l153_153004


namespace max_fraction_diagonals_sides_cyclic_pentagon_l153_153474

theorem max_fraction_diagonals_sides_cyclic_pentagon (a b c d e A B C D E : ℝ)
  (h1 : b * e + a * A = C * D)
  (h2 : c * a + b * B = D * E)
  (h3 : d * b + c * C = E * A)
  (h4 : e * c + d * D = A * B)
  (h5 : a * d + e * E = B * C) :
  (a * b * c * d * e) / (A * B * C * D * E) ≤ (5 * Real.sqrt 5 - 11) / 2 :=
sorry

end max_fraction_diagonals_sides_cyclic_pentagon_l153_153474


namespace coffee_shrinkage_l153_153828

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l153_153828


namespace y_worked_days_l153_153483

-- Definitions based on conditions
def work_rate_x := 1 / 20 -- x's work rate (W per day)
def work_rate_y := 1 / 16 -- y's work rate (W per day)

def remaining_work_by_x := 5 * work_rate_x -- Work finished by x after y left
def total_work := 1 -- Assume the total work W is 1 unit for simplicity

def days_y_worked (d : ℝ) := d * work_rate_y + remaining_work_by_x = total_work

-- The statement we need to prove
theorem y_worked_days :
  (exists d : ℕ, days_y_worked d ∧ d = 15) :=
sorry

end y_worked_days_l153_153483


namespace water_consumption_and_bill_34_7_l153_153801

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7_l153_153801


namespace larger_number_is_1634_l153_153293

theorem larger_number_is_1634 (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := 
sorry

end larger_number_is_1634_l153_153293


namespace minimum_value_expression_l153_153620

theorem minimum_value_expression 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) 
  (h_cond : x1^3 + x2^3 + x3^3 + x4^3 + x5^3 = 1) : 
  ∃ y, y = (3 * Real.sqrt 3) / 2 ∧ 
  (y = (x1 / (1 - x1^2) + x2 / (1 - x2^2) + x3 / (1 - x3^2) + x4 / (1 - x4^2) + x5 / (1 - x5^2))) :=
sorry

end minimum_value_expression_l153_153620


namespace smallest_k_for_divisibility_l153_153547

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l153_153547


namespace vector_odot_not_symmetric_l153_153075

-- Define the vector operation ⊛
def vector_odot (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

-- Statement: Prove that the operation is not symmetric
theorem vector_odot_not_symmetric (a b : ℝ × ℝ) : vector_odot a b ≠ vector_odot b a := by
  sorry

end vector_odot_not_symmetric_l153_153075


namespace contrapositive_proposition_l153_153642

theorem contrapositive_proposition (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_proposition_l153_153642


namespace sum_of_squares_l153_153711

theorem sum_of_squares (b j s : ℕ) (h : b + j + s = 34) : b^2 + j^2 + s^2 = 406 :=
sorry

end sum_of_squares_l153_153711


namespace slower_train_pass_time_l153_153462

noncomputable def time_to_pass (length_train : ℕ) (speed_faster_kmh : ℕ) (speed_slower_kmh : ℕ) : ℕ :=
  let speed_faster_mps := speed_faster_kmh * 5 / 18
  let speed_slower_mps := speed_slower_kmh * 5 / 18
  let relative_speed := speed_faster_mps + speed_slower_mps
  let distance := length_train
  distance * 18 / (relative_speed * 5)

theorem slower_train_pass_time :
  time_to_pass 500 45 15 = 300 :=
by
  sorry

end slower_train_pass_time_l153_153462


namespace opposite_of_negative_a_is_a_l153_153980

-- Define the problem:
theorem opposite_of_negative_a_is_a (a : ℝ) : -(-a) = a :=
by 
  sorry

end opposite_of_negative_a_is_a_l153_153980


namespace inequality_proof_l153_153989

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hxyz : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l153_153989


namespace angle_measure_l153_153651

theorem angle_measure (x : ℝ) (h1 : (180 - x) = 3*x - 2) : x = 45.5 :=
by
  sorry

end angle_measure_l153_153651


namespace point_on_hyperbola_l153_153234

theorem point_on_hyperbola (p r : ℝ) (h1 : p > 0) (h2 : r > 0)
  (h_el : ∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1)
  (h_par : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_circum : ∀ (a b c : ℝ), a = 2 * r - 2 * p) :
  r^2 - p^2 = 1 := sorry

end point_on_hyperbola_l153_153234


namespace james_out_of_pocket_cost_l153_153001

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l153_153001


namespace factorial_ratio_l153_153203

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l153_153203


namespace rectangle_shorter_side_l153_153299

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end rectangle_shorter_side_l153_153299


namespace total_games_played_l153_153127

-- Define the number of teams and games per matchup condition
def num_teams : ℕ := 10
def games_per_matchup : ℕ := 5

-- Calculate total games played during the season
theorem total_games_played : 
  5 * ((num_teams * (num_teams - 1)) / 2) = 225 := by 
  sorry

end total_games_played_l153_153127


namespace consecutive_integers_exist_l153_153855

def good (n : ℕ) : Prop :=
∃ (k : ℕ) (a : ℕ → ℕ), 
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧ 
  (∀ i j i' j', 1 ≤ i → i < j → j ≤ k → 1 ≤ i' → i' < j' → j' ≤ k → a i + a j = a i' + a j' → i = i' ∧ j = j') ∧ 
  (∃ (t : ℕ), ∀ m, 0 ≤ m → m < n → ∃ i j, 1 ≤ i → i < j → j ≤ k → a i + a j = t + m)

theorem consecutive_integers_exist (n : ℕ) (h : n = 1000) : good n :=
sorry

end consecutive_integers_exist_l153_153855


namespace lines_divide_circle_into_four_arcs_l153_153374

theorem lines_divide_circle_into_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y = x + a ∨ y = x + b) →
  a^2 + b^2 = 2 :=
by
  sorry

end lines_divide_circle_into_four_arcs_l153_153374


namespace least_four_digit_with_factors_l153_153465

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l153_153465


namespace point_B_in_third_quadrant_l153_153916

theorem point_B_in_third_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 1) :
    (y - 1 < 0) ∧ (x < 0) :=
by
  sorry  -- proof to be filled

end point_B_in_third_quadrant_l153_153916


namespace next_chime_time_l153_153500

theorem next_chime_time (chime1_interval : ℕ) (chime2_interval : ℕ) (chime3_interval : ℕ) (start_time : ℕ) 
  (h1 : chime1_interval = 18) (h2 : chime2_interval = 24) (h3 : chime3_interval = 30) (h4 : start_time = 9) : 
  ((start_time * 60 + 6 * 60) % (24 * 60)) / 60 = 15 :=
by
  sorry

end next_chime_time_l153_153500


namespace factor_expression_l153_153511

theorem factor_expression (y : ℤ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) :=
by 
  sorry

end factor_expression_l153_153511


namespace simplify_and_multiply_roots_l153_153169

theorem simplify_and_multiply_roots :
  (256 = 4^4) →
  (64 = 4^3) →
  (16 = 4^2) →
  ∜256 * ∛64 * sqrt 16 = 64 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_and_multiply_roots_l153_153169


namespace least_four_digit_multiple_3_5_7_l153_153472

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l153_153472


namespace tangent_line_at_1_l153_153810

-- Assume the curve and the point of tangency
noncomputable def curve (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

-- Define the point of tangency
def point_of_tangency : ℝ := 1

-- Define the expected tangent line equation in standard form Ax + By + C = 0
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 5 = 0

theorem tangent_line_at_1 :
  tangent_line point_of_tangency (curve point_of_tangency) := 
sorry

end tangent_line_at_1_l153_153810


namespace betty_needs_more_money_l153_153840

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end betty_needs_more_money_l153_153840


namespace sum_of_arithmetic_sequence_is_constant_l153_153873

def is_constant (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, S n = c

theorem sum_of_arithmetic_sequence_is_constant
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 2 + a 6 + a 10 = a 1 + d + a 1 + 5 * d + a 1 + 9 * d)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  is_constant 11 a S :=
by
  sorry

end sum_of_arithmetic_sequence_is_constant_l153_153873


namespace evaluate_expression_l153_153705

noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

theorem evaluate_expression : 
  3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 :=
by
  sorry

end evaluate_expression_l153_153705


namespace find_b_solutions_l153_153525

theorem find_b_solutions (b : ℝ) (hb : 0 < b ∧ b < 360) :
  (sin b + sin (3 * b) = 2 * sin (2 * b)) ↔
  b = 45 ∨ b = 135 ∨ b = 225 ∨ b = 315 :=
by sorry

end find_b_solutions_l153_153525


namespace solve_quadratic_l153_153453

theorem solve_quadratic :
  (x = 0 ∨ x = 2/5) ↔ (5 * x^2 - 2 * x = 0) :=
by
  sorry

end solve_quadratic_l153_153453


namespace find_x_l153_153370

def vector := (ℝ × ℝ)

-- Define the vectors a and b
def a (x : ℝ) : vector := (x, 3)
def b : vector := (3, 1)

-- Define the perpendicular condition
def perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Prove that under the given conditions, x = -1
theorem find_x (x : ℝ) (h : perpendicular (a x) b) : x = -1 :=
  sorry

end find_x_l153_153370


namespace problem_solution_l153_153590

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l153_153590


namespace option_a_option_b_option_c_l153_153228

open Real

-- Define the functions f and g
variable {f g : ℝ → ℝ}

-- Given conditions
axiom cond1 : ∀ x : ℝ, f(x + 3) = g(-x) + 4
axiom cond2 : ∀ x : ℝ, deriv f x + deriv g (1 + x) = 0
axiom cond3 : ∀ x : ℝ, g(2*x + 1) = g(-(2*x) + 1)

-- Prove the statements
theorem option_a : deriv g 1 = 0 :=
sorry

theorem option_b : ∀ x : ℝ, f(x + 4) = f(4 - x) :=
sorry

theorem option_c : ∀ x : ℝ, deriv f (x + 1) = deriv f (1 - x) :=
sorry

end option_a_option_b_option_c_l153_153228


namespace equilibrium_shift_if_K_changes_l153_153319

-- Define the equilibrium constant and its relation to temperature
def equilibrium_constant (T : ℝ) : ℝ := sorry

-- Define the conditions
axiom K_related_to_temp (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → T₁ = T₂ ↔ K₁ = K₂

axiom K_constant_with_concentration_change (T : ℝ) (K : ℝ) (c₁ c₂ : ℝ) :
  equilibrium_constant T = K → equilibrium_constant T = K

axiom K_squared_with_stoichiometric_double (T : ℝ) (K : ℝ) :
  equilibrium_constant (2 * T) = K * K

-- Define the problem to be proved
theorem equilibrium_shift_if_K_changes (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → K₁ ≠ K₂ → T₁ ≠ T₂ := 
sorry

end equilibrium_shift_if_K_changes_l153_153319


namespace b_is_square_of_positive_integer_l153_153285

theorem b_is_square_of_positive_integer 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h : b^2 = a^2 + ab + b) : 
  ∃ k : ℕ, b = k^2 := 
by 
  sorry

end b_is_square_of_positive_integer_l153_153285


namespace inconsistent_fractions_l153_153495

theorem inconsistent_fractions : (3 / 5 : ℚ) + (17 / 20 : ℚ) > 1 := by
  sorry

end inconsistent_fractions_l153_153495


namespace smallest_k_divides_l153_153537

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l153_153537


namespace cylinder_height_same_volume_as_cone_l153_153058

theorem cylinder_height_same_volume_as_cone
    (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V : ℝ)
    (h_volume_cone_eq : V = (1 / 3) * Real.pi * r_cone ^ 2 * h_cone)
    (r_cone_val : r_cone = 2)
    (h_cone_val : h_cone = 6)
    (r_cylinder_val : r_cylinder = 1) :
    ∃ h_cylinder : ℝ, (V = Real.pi * r_cylinder ^ 2 * h_cylinder) ∧ h_cylinder = 8 :=
by
  -- Here you would provide the proof for the theorem.
  sorry

end cylinder_height_same_volume_as_cone_l153_153058


namespace total_trees_after_planting_l153_153304

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l153_153304


namespace smallest_k_divides_l153_153548

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l153_153548


namespace ratio_of_weights_l153_153021

variable (x : ℝ)

-- Conditions as definitions in Lean 4
def seth_loss : ℝ := 17.5
def jerome_loss : ℝ := 17.5 * x
def veronica_loss : ℝ := 17.5 + 1.5 -- 19 pounds
def total_loss : ℝ := seth_loss + jerome_loss x + veronica_loss

-- Statement to prove
theorem ratio_of_weights (h : total_loss x = 89) : jerome_loss x / seth_loss = 3 :=
by sorry

end ratio_of_weights_l153_153021


namespace distinct_positive_integer_triplets_l153_153100

theorem distinct_positive_integer_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hprod : a * b * c = 72^3) : 
  ∃ n, n = 1482 :=
by
  sorry

end distinct_positive_integer_triplets_l153_153100


namespace vector_perpendicular_solution_l153_153891

noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (3, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_perpendicular_solution (m : ℝ) (h : dot_product (a m + b) b = 0) : m = 8 := by
  sorry

end vector_perpendicular_solution_l153_153891


namespace total_cards_l153_153402

theorem total_cards (Brenda_card Janet_card Mara_card Michelle_card : ℕ)
  (h1 : Janet_card = Brenda_card + 9)
  (h2 : Mara_card = 7 * Janet_card / 4)
  (h3 : Michelle_card = 4 * Mara_card / 5)
  (h4 : Mara_card = 210 - 60) :
  Janet_card + Brenda_card + Mara_card + Michelle_card = 432 :=
by
  sorry

end total_cards_l153_153402


namespace no_right_triangle_l153_153978

theorem no_right_triangle (a b c : ℝ) (h₁ : a = Real.sqrt 3) (h₂ : b = 2) (h₃ : c = Real.sqrt 5) : 
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end no_right_triangle_l153_153978


namespace unique_k_for_equal_power_l153_153715

theorem unique_k_for_equal_power (k : ℕ) (hk : 0 < k) (h : ∃ m n : ℕ, n > 1 ∧ (3 ^ k + 5 ^ k = m ^ n)) : k = 1 :=
by
  sorry

end unique_k_for_equal_power_l153_153715


namespace angle_A_value_cos_A_minus_2x_value_l153_153131

open Real

-- Let A, B, and C be the internal angles of triangle ABC.
variable {A B C x : ℝ}

-- Given conditions
axiom triangle_angles : A + B + C = π
axiom sinC_eq_2sinAminusB : sin C = 2 * sin (A - B)
axiom B_is_pi_over_6 : B = π / 6
axiom cosAplusx_is_neg_third : cos (A + x) = -1 / 3

-- Proof goals
theorem angle_A_value : A = π / 3 := by sorry

theorem cos_A_minus_2x_value : cos (A - 2 * x) = 7 / 9 := by sorry

end angle_A_value_cos_A_minus_2x_value_l153_153131


namespace gcd_three_numbers_l153_153529

theorem gcd_three_numbers (a b c : ℕ) (h₁ : a = 13847) (h₂ : b = 21353) (h₃ : c = 34691) : Nat.gcd (Nat.gcd a b) c = 5 := by sorry

end gcd_three_numbers_l153_153529


namespace trajectory_equation_max_value_on_trajectory_l153_153726

-- Given conditions
def distance_ratio (x y : ℝ) : Prop := 
  (Real.sqrt (x^2 + y^2)) = (1 / 2) * (Real.sqrt ((x - 3)^2 + y^2))

-- Prove the equation of the trajectory
theorem trajectory_equation (x y : ℝ) (h : distance_ratio x y) : 
  (x + 1)^2 + (4 / 3) * y^2 = 4 := 
sorry

-- Prove the maximum value of 2x^2 + y^2 on this trajectory
theorem max_value_on_trajectory (x y : ℝ) (h : (x + 1)^2 + (4 / 3) * y^2 = 4) : 
  (2 * x^2 + y^2) ≤ 18 := 
sorry

end trajectory_equation_max_value_on_trajectory_l153_153726


namespace contrapositive_even_addition_l153_153291

theorem contrapositive_even_addition (a b : ℕ) :
  (¬((a % 2 = 0) ∧ (b % 2 = 0)) → (a + b) % 2 ≠ 0) :=
sorry

end contrapositive_even_addition_l153_153291


namespace option_A_option_B_option_D_l153_153108

-- Definitions of sequences
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + n * d

def geometric_seq (b_1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  b_1 * q ^ n

-- Option A: Prove that there exist d and q such that a_n = b_n
theorem option_A : ∃ (d q : ℤ), ∀ (a_1 b_1 : ℤ) (n : ℕ), 
  (arithmetic_seq a_1 d n = geometric_seq b_1 q n) := sorry

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Option B: Prove the differences form an arithmetic sequence
theorem option_B (a_1 : ℤ) (d : ℤ) :
  ∀ n k : ℕ, k > 0 → 
  (sum_arithmetic_seq a_1 d ((k + 1) * n) - sum_arithmetic_seq a_1 d (k * n) =
   (sum_arithmetic_seq a_1 d n + k * n * n * d)) := sorry

-- Option D: Prove there exist real numbers A and a such that A * a^a_n = b_n
theorem option_D (a_1 : ℤ) (d : ℤ) (b_1 : ℤ) (q : ℤ) :
  ∀ n : ℕ, b_1 > 0 → q > 0 → 
  ∃ A a : ℝ, A * a^ (arithmetic_seq a_1 d n) = (geometric_seq b_1 q n) := sorry

end option_A_option_B_option_D_l153_153108


namespace equal_divide_remaining_amount_all_girls_l153_153637

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l153_153637


namespace total_potatoes_l153_153325

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes_l153_153325


namespace rectangle_side_lengths_l153_153355

variables (x y m n S : ℝ) (hx_y_ratio : x / y = m / n) (hxy_area : x * y = S)

theorem rectangle_side_lengths :
  x = Real.sqrt (m * S / n) ∧ y = Real.sqrt (n * S / m) :=
sorry

end rectangle_side_lengths_l153_153355


namespace necessary_but_not_sufficient_condition_l153_153033

theorem necessary_but_not_sufficient_condition (p : ℝ) : 
  p < 2 → (¬(p^2 - 4 < 0) → ∃ q, q < p ∧ q^2 - 4 < 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l153_153033


namespace equation_represents_circle_m_condition_l153_153294

theorem equation_represents_circle_m_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0) → m < 1/2 := 
by
  sorry

end equation_represents_circle_m_condition_l153_153294


namespace acute_angle_sum_l153_153012

theorem acute_angle_sum (n : ℕ) (hn : n ≥ 4) (M m: ℕ) 
  (hM : M = 3) (hm : m = 0) : M + m = 3 := 
by 
  sorry

end acute_angle_sum_l153_153012


namespace power_of_two_grows_faster_l153_153220

theorem power_of_two_grows_faster (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
sorry

end power_of_two_grows_faster_l153_153220


namespace inequality_solution_l153_153569

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem inequality_solution (a b : ℝ) 
  (h1 : ∀ (x : ℝ), f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ (x : ℝ), f a b (-2 * x) < 0 ↔ x < -3 / 2 ∨ x > 1 / 2 :=
sorry

end inequality_solution_l153_153569


namespace bananas_oranges_equivalence_l153_153636

theorem bananas_oranges_equivalence :
  (3 / 4) * 12 * banana_value = 9 * orange_value →
  (2 / 3) * 6 * banana_value = 4 * orange_value :=
by
  intros h
  sorry

end bananas_oranges_equivalence_l153_153636


namespace sufficient_but_not_necessary_condition_l153_153882

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → (x ≤ a))
  → (∃ x : ℝ, (x ≤ a ∧ ¬((-2 ≤ x ∧ x ≤ 2))))
  → (a ≥ 2) :=
by
  intros h1 h2
  sorry

end sufficient_but_not_necessary_condition_l153_153882


namespace number_of_subsets_of_M_l153_153794

def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

theorem number_of_subsets_of_M : M = {1} → ∃ n, n = 2 := by
  sorry

end number_of_subsets_of_M_l153_153794


namespace factor_1_factor_2_factor_3_l153_153087

-- Consider the variables a, b, x, y
variable (a b x y : ℝ)

-- Statement 1: Factorize 3a^3 - 6a^2 + 3a
theorem factor_1 : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
by
  sorry
  
-- Statement 2: Factorize a^2(x - y) + b^2(y - x)
theorem factor_2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a^2 - b^2) :=
by
  sorry

-- Statement 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factor_3 : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
by
  sorry

end factor_1_factor_2_factor_3_l153_153087


namespace number_of_valid_pairs_is_343_l153_153722

-- Define the given problem conditions
def given_number : Nat := 1003003001

-- Define the expression for LCM calculation
def LCM (x y : Nat) : Nat := (x * y) / (Nat.gcd x y)

-- Define the prime factorization of the given number
def is_prime_factorization_correct : Prop :=
  given_number = 7^3 * 11^3 * 13^3

-- Define x and y form as described
def is_valid_form (x y : Nat) : Prop :=
  ∃ (a b c d e f : ℕ), x = 7^a * 11^b * 13^c ∧ y = 7^d * 11^e * 13^f

-- Define the LCM condition for the ordered pairs
def meets_lcm_condition (x y : Nat) : Prop :=
  LCM x y = given_number

-- State the theorem to prove an equivalent problem
theorem number_of_valid_pairs_is_343 : is_prime_factorization_correct →
  (∃ (n : ℕ), n = 343 ∧ 
    (∀ (x y : ℕ), is_valid_form x y → meets_lcm_condition x y → x > 0 → y > 0 → True)
  ) :=
by
  intros h
  use 343
  sorry

end number_of_valid_pairs_is_343_l153_153722


namespace determine_value_of_x_l153_153747

theorem determine_value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 2 = 6 * y) : x = 48 :=
by
  sorry

end determine_value_of_x_l153_153747


namespace exists_point_with_at_most_three_nearest_neighbors_l153_153724

variable {Point : Type} [fintype Point]

-- Distance metric on points
variable (dist : Point → Point → ℝ)

-- Neighborhood set N(p) defined for each point in the set based on the distance
def N (S : finset Point) (p : Point) : finset Point := 
  S.filter (λ q, dist p q = finset.min' (S \ {p}) (λ q, dist p q))

theorem exists_point_with_at_most_three_nearest_neighbors (S : finset Point) (hS : S.nonempty) :
  ∃ p ∈ S, (N dist S p).card ≤ 3 := 
by
  sorry

end exists_point_with_at_most_three_nearest_neighbors_l153_153724


namespace hoses_fill_time_l153_153499

noncomputable def time_to_fill_pool {P A B C : ℝ} (h₁ : A + B = P / 3) (h₂ : A + C = P / 4) (h₃ : B + C = P / 5) : ℝ :=
  (120 / 47 : ℝ)

theorem hoses_fill_time {P A B C : ℝ} 
  (h₁ : A + B = P / 3) 
  (h₂ : A + C = P / 4) 
  (h₃ : B + C = P / 5) 
  : time_to_fill_pool h₁ h₂ h₃ = (120 / 47 : ℝ) :=
sorry

end hoses_fill_time_l153_153499


namespace exponentiation_division_l153_153351

variable {a : ℝ} (h1 : (a^2)^3 = a^6) (h2 : a^6 / a^2 = a^4)

theorem exponentiation_division : (a^2)^3 / a^2 = a^4 := 
by 
  sorry

end exponentiation_division_l153_153351


namespace smallest_integer_of_inequality_l153_153171

theorem smallest_integer_of_inequality :
  ∃ x : ℤ, (8 - 7 * x ≥ 4 * x - 3) ∧ (∀ y : ℤ, (8 - 7 * y ≥ 4 * y - 3) → y ≥ x) ∧ x = 1 :=
sorry

end smallest_integer_of_inequality_l153_153171


namespace problem1_problem2_problem3_problem4_l153_153846

-- Problem 1
theorem problem1 : (-10 + (-5) - (-18)) = 3 := 
by
  sorry

-- Problem 2
theorem problem2 : (-80 * (-(4 / 5)) / (abs 16)) = -4 := 
by 
  sorry

-- Problem 3
theorem problem3 : ((1/2 - 5/9 + 5/6 - 7/12) * (-36)) = -7 := 
by 
  sorry

-- Problem 4
theorem problem4 : (- 3^2 * (-1/3)^2 +(-2)^2 / (- (2/3))^3) = -29 / 27 :=
by 
  sorry

end problem1_problem2_problem3_problem4_l153_153846


namespace find_m_l153_153890

-- Define the given equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := fun p => (3 + m) * p.1 - 4 * p.2 = 5 - 3 * m
def line2 : ℝ × ℝ → Prop := fun p => 2 * p.1 - p.2 = 8

-- Define the condition for parallel lines based on the given equations
def are_parallel (m : ℝ) : Prop := (3 + m) / 4 = 2

-- The main theorem stating the value of m
theorem find_m (m : ℝ) (h1 : ∀ p : ℝ × ℝ, line1 m p) (h2 : ∀ p : ℝ × ℝ, line2 p) (h_parallel : are_parallel m) : m = 5 :=
sorry

end find_m_l153_153890


namespace find_n_l153_153871

theorem find_n (n : ℕ) (S : ℕ) (h1 : S = n * (n + 1) / 2)
  (h2 : ∃ a : ℕ, a > 0 ∧ a < 10 ∧ S = 111 * a) : n = 36 :=
sorry

end find_n_l153_153871


namespace degree_measure_supplement_complement_l153_153310

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end degree_measure_supplement_complement_l153_153310


namespace sum_of_digits_of_multiple_of_990_l153_153522

theorem sum_of_digits_of_multiple_of_990 (a b c : ℕ) (h₀ : a < 10 ∧ b < 10 ∧ c < 10)
  (h₁ : ∃ (d e f g : ℕ), 123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c = 123000 + 9000 + 900 + 90 + 9 + 0)
  (h2 : (123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c) % 990 = 0) :
  a + b + c = 12 :=
by {
  sorry
}

end sum_of_digits_of_multiple_of_990_l153_153522


namespace product_of_two_numbers_l153_153952

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : x * y = 97.9450625 :=
by
  sorry

end product_of_two_numbers_l153_153952


namespace deepak_age_l153_153070

theorem deepak_age (A D : ℕ)
  (h1 : A / D = 2 / 3)
  (h2 : A + 5 = 25) :
  D = 30 := 
by
  sorry

end deepak_age_l153_153070


namespace lana_picked_37_roses_l153_153406

def total_flowers_picked (used : ℕ) (extra : ℕ) := used + extra

def picked_roses (total : ℕ) (tulips : ℕ) := total - tulips

theorem lana_picked_37_roses :
    ∀ (tulips used extra : ℕ), tulips = 36 → used = 70 → extra = 3 → 
    picked_roses (total_flowers_picked used extra) tulips = 37 :=
by
  intros tulips used extra htulips husd hextra
  sorry

end lana_picked_37_roses_l153_153406


namespace weight_of_b_l153_153323

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 135) (h2 : a + b = 80) (h3 : b + c = 82) : b = 27 :=
by
  sorry

end weight_of_b_l153_153323


namespace initial_elephants_l153_153460

theorem initial_elephants (E : ℕ) :
  (E + 35 + 135 + 125 = 315) → (5 * 35 / 7 = 25) → (5 * 25 = 125) → (135 = 125 + 10) →
  E = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_elephants_l153_153460
