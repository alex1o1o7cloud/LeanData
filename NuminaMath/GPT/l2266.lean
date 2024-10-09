import Mathlib

namespace hexagon_division_ratio_l2266_226654

theorem hexagon_division_ratio
  (hex_area : ℝ)
  (hexagon : ∀ (A B C D E F : ℝ), hex_area = 8)
  (line_PQ_splits : ∀ (above_area below_area : ℝ), above_area = 4 ∧ below_area = 4)
  (below_PQ : ℝ)
  (unit_square_area : ∀ (unit_square : ℝ), unit_square = 1)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (triangle_area : ∀ (base height : ℝ), triangle_base = 4 ∧ (base * height) / 2 = 3)
  (XQ QY : ℝ)
  (bases_sum : ∀ (XQ QY : ℝ), XQ + QY = 4) :
  XQ / QY = 2 / 3 :=
sorry

end hexagon_division_ratio_l2266_226654


namespace B_is_left_of_A_l2266_226692

-- Define the coordinates of points A and B
def A_coord : ℚ := 5 / 8
def B_coord : ℚ := 8 / 13

-- The statement we want to prove: B is to the left of A
theorem B_is_left_of_A : B_coord < A_coord :=
  by {
    sorry
  }

end B_is_left_of_A_l2266_226692


namespace equalize_expenses_l2266_226650

variable {x y : ℝ} 

theorem equalize_expenses (h : x > y) : (x + y) / 2 - y = (x - y) / 2 :=
by sorry

end equalize_expenses_l2266_226650


namespace sequence_periodicity_l2266_226656

theorem sequence_periodicity (a : ℕ → ℕ) (n : ℕ) (h : ∀ k, a k = 6^k) :
  a (n + 5) % 100 = a n % 100 :=
by sorry

end sequence_periodicity_l2266_226656


namespace boys_girls_relationship_l2266_226602

theorem boys_girls_relationship (b g : ℕ): (4 + 2 * b = g) → (b = (g - 4) / 2) :=
by
  intros h
  sorry

end boys_girls_relationship_l2266_226602


namespace lives_per_player_l2266_226639

-- Definitions based on the conditions
def initial_players : Nat := 2
def joined_players : Nat := 2
def total_lives : Nat := 24

-- Derived condition
def total_players : Nat := initial_players + joined_players

-- Proof statement
theorem lives_per_player : total_lives / total_players = 6 :=
by
  sorry

end lives_per_player_l2266_226639


namespace original_number_is_76_l2266_226677

-- Define the original number x and the condition given
def original_number_condition (x : ℝ) : Prop :=
  (3 / 4) * x = x - 19

-- State the theorem that the original number x must be 76 if it satisfies the condition
theorem original_number_is_76 (x : ℝ) (h : original_number_condition x) : x = 76 :=
sorry

end original_number_is_76_l2266_226677


namespace jorge_total_spent_l2266_226660

-- Definitions based on the problem conditions
def price_adult_ticket : ℝ := 10
def price_child_ticket : ℝ := 5
def num_adult_tickets : ℕ := 12
def num_child_tickets : ℕ := 12
def discount_adult : ℝ := 0.40
def discount_child : ℝ := 0.30
def extra_discount : ℝ := 0.10

-- The desired statement to prove
theorem jorge_total_spent :
  let total_adult_cost := num_adult_tickets * price_adult_ticket
  let total_child_cost := num_child_tickets * price_child_ticket
  let discounted_adult := total_adult_cost * (1 - discount_adult)
  let discounted_child := total_child_cost * (1 - discount_child)
  let total_cost_before_extra_discount := discounted_adult + discounted_child
  let final_cost := total_cost_before_extra_discount * (1 - extra_discount)
  final_cost = 102.60 :=
by 
  sorry

end jorge_total_spent_l2266_226660


namespace difference_of_lines_in_cm_l2266_226667

def W : ℝ := 7.666666666666667
def B : ℝ := 3.3333333333333335
def inch_to_cm : ℝ := 2.54

theorem difference_of_lines_in_cm :
  (W * inch_to_cm) - (B * inch_to_cm) = 11.005555555555553 := 
sorry

end difference_of_lines_in_cm_l2266_226667


namespace empty_set_condition_l2266_226649

def isEmptySet (s : Set ℝ) : Prop := s = ∅

def A : Set ℕ := {n : ℕ | n^2 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def C : Set ℝ := {x : ℝ | x^2 + x + 1 = 0}
def D : Set ℝ := {0}

theorem empty_set_condition : isEmptySet C := by
  sorry

end empty_set_condition_l2266_226649


namespace part1_part2_l2266_226696

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

-- (1) Given a = -1, prove that the inequality f(x, -1) ≤ 0 implies x ≤ -1/3
theorem part1 (x : ℝ) : (f x (-1) ≤ 0) ↔ (x ≤ -1/3) :=
by
  sorry

-- (2) Given f(x) ≥ 0 for all x ≥ -1, prove that the range for a is a ≤ -3 or a ≥ 1
theorem part2 (a : ℝ) : (∀ x, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l2266_226696


namespace find_s_and_x_l2266_226648

theorem find_s_and_x (s x t : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3.75) :
  s = 0.5 ∧ x = s / 2 → x = 0.25 :=
by
  sorry

end find_s_and_x_l2266_226648


namespace sodium_chloride_solution_l2266_226610

theorem sodium_chloride_solution (n y : ℝ) (h1 : n > 30) 
  (h2 : 0.01 * n * n = 0.01 * (n - 8) * (n + y)) : 
  y = 8 * n / (n + 8) :=
sorry

end sodium_chloride_solution_l2266_226610


namespace total_number_of_lives_l2266_226669

theorem total_number_of_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
                              (h1 : initial_players = 7) (h2 : additional_players = 2) (h3 : lives_per_player = 7) : 
                              initial_players + additional_players * lives_per_player = 63 :=
by
  sorry

end total_number_of_lives_l2266_226669


namespace bug_total_distance_l2266_226600

def total_distance (p1 p2 p3 p4 : ℤ) : ℤ :=
  abs (p2 - p1) + abs (p3 - p2) + abs (p4 - p3)

theorem bug_total_distance : total_distance (-3) (-8) 0 6 = 19 := 
by sorry

end bug_total_distance_l2266_226600


namespace fraction_value_sin_cos_value_l2266_226637

open Real

-- Let alpha be an angle in radians satisfying the given condition
variable (α : ℝ)

-- Given condition
def condition  : Prop := sin α = 2 * cos α

-- First question
theorem fraction_value (h : condition α) : 
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 :=
sorry

-- Second question
theorem sin_cos_value (h : condition α) : 
  sin α ^ 2 + 2 * sin α * cos α = 8 / 5 :=
sorry

end fraction_value_sin_cos_value_l2266_226637


namespace clarence_initial_oranges_l2266_226686

variable (initial_oranges : ℕ)
variable (obtained_from_joyce : ℕ := 3)
variable (total_oranges : ℕ := 8)

theorem clarence_initial_oranges (initial_oranges : ℕ) :
  initial_oranges + obtained_from_joyce = total_oranges → initial_oranges = 5 :=
by
  sorry

end clarence_initial_oranges_l2266_226686


namespace triangle_side_length_uniqueness_l2266_226605

-- Define the conditions as axioms
variable (n : ℕ)
variable (h : n > 0)
variable (A1 : 3 * n + 9 > 5 * n - 4)
variable (A2 : 5 * n - 4 > 4 * n + 6)

-- The theorem stating the constraints and expected result
theorem triangle_side_length_uniqueness :
  (4 * n + 6) + (3 * n + 9) > (5 * n - 4) ∧
  (3 * n + 9) + (5 * n - 4) > (4 * n + 6) ∧
  (5 * n - 4) + (4 * n + 6) > (3 * n + 9) ∧
  3 * n + 9 > 5 * n - 4 ∧
  5 * n - 4 > 4 * n + 6 → 
  n = 11 :=
by {
  -- Proof steps can be filled here
  sorry
}

end triangle_side_length_uniqueness_l2266_226605


namespace contractor_net_amount_l2266_226690

-- Definitions based on conditions
def total_days : ℕ := 30
def pay_per_day : ℝ := 25
def fine_per_absence_day : ℝ := 7.5
def days_absent : ℕ := 6

-- Calculate days worked
def days_worked : ℕ := total_days - days_absent

-- Calculate total earnings
def earnings : ℝ := days_worked * pay_per_day

-- Calculate total fine
def fine : ℝ := days_absent * fine_per_absence_day

-- Calculate net amount received by the contractor
def net_amount : ℝ := earnings - fine

-- Problem statement: Prove that the net amount is Rs. 555
theorem contractor_net_amount : net_amount = 555 := by
  sorry

end contractor_net_amount_l2266_226690


namespace percentage_error_in_calculated_area_l2266_226698

theorem percentage_error_in_calculated_area 
  (s : ℝ) 
  (measured_side : ℝ) 
  (h : measured_side = s * 1.04) :
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 8.16 :=
by
  sorry

end percentage_error_in_calculated_area_l2266_226698


namespace tiffany_lives_problem_l2266_226693

/-- Tiffany's lives problem -/
theorem tiffany_lives_problem (L : ℤ) (h1 : 43 - L + 27 = 56) : L = 14 :=
by {
  sorry
}

end tiffany_lives_problem_l2266_226693


namespace max_prime_product_l2266_226614

theorem max_prime_product : 
  ∃ (x y z : ℕ), 
    Prime x ∧ Prime y ∧ Prime z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x + y + z = 49 ∧ 
    x * y * z = 4199 := 
by
  sorry

end max_prime_product_l2266_226614


namespace area_of_rectangular_plot_l2266_226630

theorem area_of_rectangular_plot (B L : ℕ) (h1 : L = 3 * B) (h2 : B = 18) : L * B = 972 := by
  sorry

end area_of_rectangular_plot_l2266_226630


namespace income_ratio_l2266_226662

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

end income_ratio_l2266_226662


namespace ratio_of_tetrahedrons_volume_l2266_226626

theorem ratio_of_tetrahedrons_volume (d R s s' V_ratio m n : ℕ) (h1 : d = 4)
  (h2 : R = 2)
  (h3 : s = 4 * R / Real.sqrt 6)
  (h4 : s' = s / Real.sqrt 8)
  (h5 : V_ratio = (s' / s) ^ 3)
  (hm : m = 1)
  (hn : n = 32)
  (h_ratio : V_ratio = m / n) :
  m + n = 33 :=
by
  sorry

end ratio_of_tetrahedrons_volume_l2266_226626


namespace triangle_angle_sum_l2266_226670

theorem triangle_angle_sum (angle_Q R P : ℝ)
  (h1 : R = 3 * angle_Q)
  (h2 : angle_Q = 30)
  (h3 : P + angle_Q + R = 180) :
    P = 60 :=
by
  sorry

end triangle_angle_sum_l2266_226670


namespace sum_inequality_l2266_226668

theorem sum_inequality (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2) :
  (x + y + z) * (x⁻¹ + y⁻¹ + z⁻¹) ≥ 6 * (x / (y + z) + y / (z + x) + z / (x + y)) := sorry

end sum_inequality_l2266_226668


namespace no_real_solutions_l2266_226666

theorem no_real_solutions : ∀ (x y : ℝ), ¬ (3 * x^2 + y^2 - 9 * x - 6 * y + 23 = 0) :=
by sorry

end no_real_solutions_l2266_226666


namespace tangent_addition_l2266_226621

open Real

theorem tangent_addition (x : ℝ) (h : tan x = 3) :
  tan (x + π / 6) = - (5 * (sqrt 3 + 3)) / 3 := by
  -- Providing a brief outline of the proof steps is not necessary for the statement
  sorry

end tangent_addition_l2266_226621


namespace quadratic_roots_l2266_226607

-- Define the condition for the quadratic equation
def quadratic_eq (x m : ℝ) : Prop := x^2 - 4*x + m + 2 = 0

-- Define the discriminant condition
def discriminant_pos (m : ℝ) : Prop := (4^2 - 4 * (m + 2)) > 0

-- Define the condition range for m
def m_range (m : ℝ) : Prop := m < 2

-- Define the condition for m as a positive integer
def m_positive_integer (m : ℕ) : Prop := m = 1

-- The main theorem stating the problem
theorem quadratic_roots : 
  (∀ (m : ℝ), discriminant_pos m → m_range m) ∧ 
  (∀ m : ℕ, m_positive_integer m → (∃ x1 x2 : ℝ, quadratic_eq x1 m ∧ quadratic_eq x2 m ∧ x1 = 1 ∧ x2 = 3)) := 
by 
  sorry

end quadratic_roots_l2266_226607


namespace coordinate_relationship_l2266_226624

theorem coordinate_relationship (x y : ℝ) (h : |x| - |y| = 0) : (|x| - |y| = 0) :=
by
    sorry

end coordinate_relationship_l2266_226624


namespace john_hiking_probability_l2266_226673

theorem john_hiking_probability :
  let P_rain := 0.3
  let P_sunny := 0.7
  let P_hiking_if_rain := 0.1
  let P_hiking_if_sunny := 0.9

  let P_hiking := P_rain * P_hiking_if_rain + P_sunny * P_hiking_if_sunny

  P_hiking = 0.66 := by
    sorry

end john_hiking_probability_l2266_226673


namespace contrapositive_l2266_226646

theorem contrapositive (a : ℝ) : (a > 0 → a > 1) → (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_l2266_226646


namespace combined_distance_is_twelve_l2266_226676

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l2266_226676


namespace find_solutions_l2266_226661

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 9 * x ^ 2 + 6

theorem find_solutions :
  ∃ x1 x2 x3 : ℝ, f x1 = Real.sqrt 2 ∧ f x2 = Real.sqrt 2 ∧ f x3 = Real.sqrt 2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end find_solutions_l2266_226661


namespace evaluate_expression_l2266_226651

theorem evaluate_expression : (-1:ℤ)^2022 + |(-2:ℤ)| - (1/2 : ℚ)^0 - 2 * Real.tan (Real.pi / 4) = 0 := 
by
  sorry

end evaluate_expression_l2266_226651


namespace first_stopover_distance_l2266_226642

theorem first_stopover_distance 
  (total_distance : ℕ) 
  (second_stopover_distance : ℕ) 
  (distance_after_second_stopover : ℕ) :
  total_distance = 436 → 
  second_stopover_distance = 236 → 
  distance_after_second_stopover = 68 →
  second_stopover_distance - (total_distance - second_stopover_distance - distance_after_second_stopover) = 104 :=
by
  intros
  sorry

end first_stopover_distance_l2266_226642


namespace quadratic_max_value_l2266_226628

open Real

variables (a b c x : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value (h₀ : a < 0) (x₀ : ℝ) (h₁ : 2 * a * x₀ + b = 0) : 
  ∀ x : ℝ, f a b c x ≤ f a b c x₀ := sorry

end quadratic_max_value_l2266_226628


namespace smallest_three_digit_multiple_of_13_l2266_226694

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n ∧ (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l2266_226694


namespace last_two_digits_of_sum_l2266_226606

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum :
  last_two_digits (factorial 4 + factorial 5 + factorial 6 + factorial 7 + factorial 8 + factorial 9) = 4 :=
by
  sorry

end last_two_digits_of_sum_l2266_226606


namespace problem1_solution_problem2_solution_problem3_solution_l2266_226697

-- Problem 1
theorem problem1_solution (x : ℝ) :
  (6 * x - 1) ^ 2 = 25 ↔ (x = 1 ∨ x = -2 / 3) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) :
  4 * x^2 - 1 = 12 * x ↔ (x = 3 / 2 + (Real.sqrt 10) / 2 ∨ x = 3 / 2 - (Real.sqrt 10) / 2) :=
sorry

-- Problem 3
theorem problem3_solution (x : ℝ) :
  x * (x - 7) = 8 * (7 - x) ↔ (x = 7 ∨ x = -8) :=
sorry

end problem1_solution_problem2_solution_problem3_solution_l2266_226697


namespace quadratic_equation_solutions_l2266_226652

theorem quadratic_equation_solutions : ∀ x : ℝ, x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := 
by sorry

end quadratic_equation_solutions_l2266_226652


namespace angle_D_in_triangle_DEF_l2266_226616

theorem angle_D_in_triangle_DEF 
  (E F D : ℝ) 
  (hEF : F = 3 * E) 
  (hE : E = 15) 
  (h_sum_angles : D + E + F = 180) : D = 120 :=
by
  -- Proof goes here
  sorry

end angle_D_in_triangle_DEF_l2266_226616


namespace pq_plus_qr_plus_rp_cubic_1_l2266_226695

theorem pq_plus_qr_plus_rp_cubic_1 (p q r : ℝ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + p * r + q * r = -2)
  (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -6 :=
by
  sorry

end pq_plus_qr_plus_rp_cubic_1_l2266_226695


namespace f_eq_n_for_all_n_l2266_226681

noncomputable def f : ℕ → ℕ := sorry

axiom f_pos_int_valued (n : ℕ) (h : 0 < n) : f n = f n

axiom f_2_eq_2 : f 2 = 2

axiom f_mul_prop (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n

axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : f m > f n

theorem f_eq_n_for_all_n (n : ℕ) (hn : 0 < n) : f n = n := sorry

end f_eq_n_for_all_n_l2266_226681


namespace find_positive_integer_l2266_226675

def product_of_digits (n : Nat) : Nat :=
  -- Function to compute product of digits, assume it is defined correctly
  sorry

theorem find_positive_integer (x : Nat) (h : x > 0) :
  product_of_digits x = x * x - 10 * x - 22 ↔ x = 12 :=
by
  sorry

end find_positive_integer_l2266_226675


namespace problem1_problem2_l2266_226638

theorem problem1 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : Real.exp x + Real.exp y > 2 * Real.exp 1 :=
by {
  sorry -- Proof goes here
}

theorem problem2 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : x * Real.log x + y * Real.log y > 0 :=
by {
  sorry -- Proof goes here
}

end problem1_problem2_l2266_226638


namespace correct_statement_l2266_226672

-- Define the necessary variables
variables {a b c : ℝ}

-- State the theorem including the condition and the conclusion
theorem correct_statement (h : a > b) : b - c < a - c :=
by linarith


end correct_statement_l2266_226672


namespace exceeds_alpha_beta_l2266_226657

noncomputable def condition (α β p q : ℝ) : Prop :=
  q < 50 ∧ α > 0 ∧ β > 0 ∧ p > 0 ∧ q > 0

theorem exceeds_alpha_beta (α β p q : ℝ) (h : condition α β p q) :
  (1 + p / 100) * (1 - q / 100) > 1 → p > 100 * q / (100 - q) := by
  sorry

end exceeds_alpha_beta_l2266_226657


namespace total_teachers_correct_l2266_226608

-- Define the number of departments and the total number of teachers
def num_departments : ℕ := 7
def total_teachers : ℕ := 140

-- Proving that the total number of teachers is 140
theorem total_teachers_correct : total_teachers = 140 := 
by
  sorry

end total_teachers_correct_l2266_226608


namespace sin_35pi_over_6_l2266_226680

theorem sin_35pi_over_6 : Real.sin (35 * Real.pi / 6) = -1 / 2 := by
  sorry

end sin_35pi_over_6_l2266_226680


namespace greatest_k_l2266_226687

noncomputable def n : ℕ := sorry
def k : ℕ := sorry

axiom d : ℕ → ℕ

axiom h1 : d n = 72
axiom h2 : d (5 * n) = 90

theorem greatest_k : ∃ k : ℕ, (∀ m : ℕ, m > k → ¬(5^m ∣ n)) ∧ 5^k ∣ n ∧ k = 3 :=
by
  sorry

end greatest_k_l2266_226687


namespace find_table_price_l2266_226604

noncomputable def chair_price (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
noncomputable def chair_table_sum (C T : ℝ) : Prop := C + T = 64

theorem find_table_price (C T : ℝ) (h1 : chair_price C T) (h2 : chair_table_sum C T) : T = 56 :=
by sorry

end find_table_price_l2266_226604


namespace archipelago_max_value_l2266_226679

noncomputable def archipelago_max_islands (N : ℕ) : Prop :=
  N ≥ 7 ∧ 
  (∀ (a b : ℕ), a ≠ b → a ≤ N → b ≤ N → ∃ c : ℕ, c ≤ N ∧ (∃ d, d ≠ c ∧ d ≤ N → d ≠ a ∧ d ≠ b)) ∧ 
  (∀ (a : ℕ), a ≤ N → ∃ b, b ≠ a ∧ b ≤ N ∧ (∃ c, c ≤ N ∧ c ≠ b ∧ c ≠ a))

theorem archipelago_max_value : archipelago_max_islands 36 := sorry

end archipelago_max_value_l2266_226679


namespace car_speed_l2266_226603

theorem car_speed (d t : ℝ) (h_d : d = 624) (h_t : t = 3) : d / t = 208 := by
  sorry

end car_speed_l2266_226603


namespace remainder_when_a_plus_b_div_40_is_28_l2266_226685

theorem remainder_when_a_plus_b_div_40_is_28 :
  ∃ k j : ℤ, (a = 80 * k + 74 ∧ b = 120 * j + 114) → (a + b) % 40 = 28 := by
  sorry

end remainder_when_a_plus_b_div_40_is_28_l2266_226685


namespace john_trip_total_time_l2266_226689

theorem john_trip_total_time :
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  t1 + t2 + t3 + t4 + t5 = 872 :=
by
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  have h1: t1 + t2 + t3 + t4 + t5 = 2 + (3 * 2) + (4 * (3 * 2)) + (5 * (4 * (3 * 2))) + (6 * (5 * (4 * (3 * 2)))) := by
    sorry
  have h2: 2 + 6 + 24 + 120 + 720 = 872 := by
    sorry
  exact h2

end john_trip_total_time_l2266_226689


namespace g_at_5_l2266_226688

def g (x : ℝ) : ℝ := sorry -- Placeholder for the function definition, typically provided in further context

theorem g_at_5 : g 5 = 3 / 4 :=
by
  -- Given condition as a hypothesis
  have h : ∀ x: ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1 := sorry
  sorry  -- Full proof should go here

end g_at_5_l2266_226688


namespace students_in_johnsons_class_l2266_226615

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end students_in_johnsons_class_l2266_226615


namespace sequence_general_term_correct_l2266_226613

open Nat

def S (n : ℕ) : ℤ := 3 * (n : ℤ) * (n : ℤ) - 2 * (n : ℤ) + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2
  else 6 * (n : ℤ) - 5

theorem sequence_general_term_correct : ∀ n, (S n - S (n - 1) = a n) :=
by
  intros
  sorry

end sequence_general_term_correct_l2266_226613


namespace polar_line_through_center_perpendicular_to_axis_l2266_226601

-- We define our conditions
def circle_in_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def center_of_circle (C : ℝ × ℝ) : Prop := C = (2, 0)

def line_in_rectangular (x : ℝ) : Prop := x = 2

-- We now state the proof problem
theorem polar_line_through_center_perpendicular_to_axis (ρ θ : ℝ) : 
  (∃ C, center_of_circle C ∧ (∃ x, line_in_rectangular x)) →
  (circle_in_polar ρ θ → ρ * Real.cos θ = 2) :=
by
  sorry

end polar_line_through_center_perpendicular_to_axis_l2266_226601


namespace min_value_PF_PA_l2266_226671

noncomputable def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def focus_left : ℝ × ℝ := (-4, 0)
noncomputable def focus_right : ℝ × ℝ := (4, 0)
noncomputable def point_A : ℝ × ℝ := (1, 4)

theorem min_value_PF_PA (P : ℝ × ℝ)
  (hP : hyperbola_eq P.1 P.2)
  (hP_right_branch : P.1 > 0) :
  ∃ P : ℝ × ℝ, ∀ X : ℝ × ℝ, hyperbola_eq X.1 X.2 → X.1 > 0 → 
               (dist X focus_left + dist X point_A) ≥ 9 ∧
               (dist P focus_left + dist P point_A) = 9 := 
sorry

end min_value_PF_PA_l2266_226671


namespace dans_average_rate_l2266_226617

/-- Dan's average rate for the entire trip, given the conditions, equals 0.125 miles per minute --/
theorem dans_average_rate :
  ∀ (d_run d_swim : ℝ) (r_run r_swim : ℝ) (time_run time_swim : ℝ),
  d_run = 3 ∧ d_swim = 3 ∧ r_run = 10 ∧ r_swim = 6 ∧ 
  time_run = (d_run / r_run) * 60 ∧ time_swim = (d_swim / r_swim) * 60 →
  ((d_run + d_swim) / (time_run + time_swim)) = 0.125 :=
by
  intros d_run d_swim r_run r_swim time_run time_swim h
  sorry

end dans_average_rate_l2266_226617


namespace part1_part2_part3_l2266_226611

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem part1 :
  ∀ x : ℝ, |f x| = |x - 1| → x = -2 ∨ x = 0 ∨ x = 1 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |f x1| = g a x1 ∧ |f x2| = g a x2) ↔ (a = 0 ∨ a = 2) :=
sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ (a ≤ -2) :=
sorry

end part1_part2_part3_l2266_226611


namespace tunnel_connects_land_l2266_226663

noncomputable def surface_area (planet : Type) : ℝ := sorry
noncomputable def land_area (planet : Type) : ℝ := sorry
noncomputable def half_surface_area (planet : Type) : ℝ := surface_area planet / 2
noncomputable def can_dig_tunnel_through_center (planet : Type) : Prop := sorry

variable {TauCeti : Type}

-- Condition: Land occupies more than half of the entire surface area.
axiom land_more_than_half : land_area TauCeti > half_surface_area TauCeti

-- Proof problem statement: Prove that inhabitants can dig a tunnel through the center of the planet.
theorem tunnel_connects_land : can_dig_tunnel_through_center TauCeti :=
sorry

end tunnel_connects_land_l2266_226663


namespace center_cell_value_l2266_226622

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l2266_226622


namespace arithmetic_seq_2a9_a10_l2266_226699

theorem arithmetic_seq_2a9_a10 (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (arith_seq : ∀ n : ℕ, ∃ d : ℕ, a n = a 1 + (n - 1) * d) : 2 * a 9 - a 10 = 15 :=
by
  sorry

end arithmetic_seq_2a9_a10_l2266_226699


namespace points_per_game_without_bonus_l2266_226664

-- Definition of the conditions
def b : ℕ := 82
def n : ℕ := 79
def P : ℕ := 15089

-- Theorem statement
theorem points_per_game_without_bonus :
  (P - b * n) / n = 109 :=
by
  -- Proof will be filled in here
  sorry

end points_per_game_without_bonus_l2266_226664


namespace determine_a_l2266_226683

theorem determine_a (a b c : ℤ) (h : (b + 11) * (c + 11) = 2) (hb : b + 11 = -2) (hc : c + 11 = -1) :
  a = 13 := by
  sorry

end determine_a_l2266_226683


namespace inequality_proof_l2266_226619

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l2266_226619


namespace largest_common_term_l2266_226634

theorem largest_common_term (n m : ℕ) (k : ℕ) (a : ℕ) 
  (h1 : a = 7 + 7 * n) 
  (h2 : a = 8 + 12 * m) 
  (h3 : 56 + 84 * k < 500) : a = 476 :=
  sorry

end largest_common_term_l2266_226634


namespace incircle_angle_b_l2266_226643

open Real

theorem incircle_angle_b
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (angle_AOC_eq_4_MKN : ∀ (MKN : ℝ), 4 * MKN = 180 - (180 - γ) / 2 - (180 - α) / 2) :
    β = 108 :=
by
  -- Proof will be handled here.
  sorry

end incircle_angle_b_l2266_226643


namespace fruit_bowl_oranges_l2266_226612

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l2266_226612


namespace simplify_expression_l2266_226632

theorem simplify_expression (x y : ℝ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 :=
by
  sorry

end simplify_expression_l2266_226632


namespace average_percentage_l2266_226635

theorem average_percentage (x : ℝ) : (60 + x + 80) / 3 = 70 → x = 70 :=
by
  intro h
  sorry

end average_percentage_l2266_226635


namespace perimeter_of_larger_triangle_is_65_l2266_226682

noncomputable def similar_triangle_perimeter : ℝ :=
  let a := 7
  let b := 7
  let c := 12
  let longest_side_similar := 30
  let perimeter_small := a + b + c
  let ratio := longest_side_similar / c
  ratio * perimeter_small

theorem perimeter_of_larger_triangle_is_65 :
  similar_triangle_perimeter = 65 := by
  sorry

end perimeter_of_larger_triangle_is_65_l2266_226682


namespace dot_product_MN_MO_is_8_l2266_226691

-- Define the circle O as a set of points (x, y) such that x^2 + y^2 = 9
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the length of the chord MN in the circle
def chord_length (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (x1 - x2)^2 + (y1 - y2)^2 = 16

-- Define the vector MN and MO
def vector_dot_product (M N O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := N
  let (x0, y0) := O
  let v1 := (x2 - x1, y2 - y1)
  let v2 := (x0 - x1, y0 - y1)
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the origin point O (center of the circle)
def O : ℝ × ℝ := (0, 0)

-- The theorem to prove
theorem dot_product_MN_MO_is_8 (M N : ℝ × ℝ) (hM : is_circle M.1 M.2) (hN : is_circle N.1 N.2) (hMN : chord_length M N) :
  vector_dot_product M N O = 8 :=
sorry

end dot_product_MN_MO_is_8_l2266_226691


namespace emily_initial_marbles_l2266_226636

open Nat

theorem emily_initial_marbles (E : ℕ) (h : 3 * E - (3 * E / 2 + 1) = 8) : E = 6 :=
sorry

end emily_initial_marbles_l2266_226636


namespace juicy_12_juicy_20_l2266_226625

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 1 = (1 / a) + (1 / b) + (1 / c) + (1 / d) ∧ a * b * c * d = n

theorem juicy_12 : is_juicy 12 :=
sorry

theorem juicy_20 : is_juicy 20 :=
sorry

end juicy_12_juicy_20_l2266_226625


namespace remaining_painting_time_l2266_226665

-- Define the given conditions as Lean definitions
def total_rooms : ℕ := 9
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 5

-- Formulate the main theorem to prove the remaining time is 32 hours
theorem remaining_painting_time : 
  (total_rooms - rooms_painted) * hours_per_room = 32 := 
by 
  sorry

end remaining_painting_time_l2266_226665


namespace minimum_questions_needed_a_l2266_226655

theorem minimum_questions_needed_a (n : ℕ) (m : ℕ) (h1 : m = n) (h2 : m < 2 ^ n) :
  ∃Q : ℕ, Q = n := sorry

end minimum_questions_needed_a_l2266_226655


namespace find_f_g_3_l2266_226631

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_f_g_3 :
  f (g 3) = -2 := by
  sorry

end find_f_g_3_l2266_226631


namespace min_sum_real_possible_sums_int_l2266_226609

-- Lean 4 statement for the real numbers case
theorem min_sum_real (x y : ℝ) (hx : x + y + 2 * x * y = 5) (hx_pos : x > 0) (hy_pos : y > 0) :
  x + y ≥ Real.sqrt 11 - 1 := 
sorry

-- Lean 4 statement for the integers case
theorem possible_sums_int (x y : ℤ) (hx : x + y + 2 * x * y = 5) :
  x + y = 5 ∨ x + y = -7 :=
sorry

end min_sum_real_possible_sums_int_l2266_226609


namespace dot_product_OA_OB_l2266_226645

theorem dot_product_OA_OB :
  let A := (Real.cos 110, Real.sin 110)
  let B := (Real.cos 50, Real.sin 50)
  (A.1 * B.1 + A.2 * B.2) = 1 / 2 :=
by
  sorry

end dot_product_OA_OB_l2266_226645


namespace compute_expression_l2266_226623

theorem compute_expression :
  (4 + 8 - 16 + 32 + 64 - 128 + 256) / (8 + 16 - 32 + 64 + 128 - 256 + 512) = 1 / 2 :=
by
  sorry

end compute_expression_l2266_226623


namespace percentage_exceeds_self_l2266_226678

theorem percentage_exceeds_self (N : ℝ) (P : ℝ) (hN : N = 75) (h_condition : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end percentage_exceeds_self_l2266_226678


namespace dog_nails_per_foot_l2266_226629

-- Definitions from conditions
def number_of_dogs := 4
def number_of_parrots := 8
def total_nails_to_cut := 113
def parrots_claws := 8

-- Derived calculations from the solution but only involving given conditions
def dogs_claws (nails_per_foot : ℕ) := 16 * nails_per_foot
def parrots_total_claws := number_of_parrots * parrots_claws

-- The main theorem to prove the number of nails per dog foot
theorem dog_nails_per_foot :
  ∃ x : ℚ, 16 * x + parrots_total_claws = total_nails_to_cut :=
by {
  -- Directly state the expected answer
  use 3.0625,
  -- Placeholder for proof
  sorry
}

end dog_nails_per_foot_l2266_226629


namespace larger_number_l2266_226653

theorem larger_number (t a b : ℝ) (h1 : a + b = t) (h2 : a ^ 2 - b ^ 2 = 208) (ht : t = 104) :
  a = 53 :=
by
  sorry

end larger_number_l2266_226653


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l2266_226618

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ p2 = p1 + 1 ∧ is_prime p3 ∧ p3 = p2 + 1

def sum_divisible_by_5 (p1 p2 p3 : ℕ) : Prop :=
  (p1 + p2 + p3) % 5 = 0

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (p1 p2 p3 : ℕ), consecutive_primes p1 p2 p3 ∧ sum_divisible_by_5 p1 p2 p3 ∧ p1 + p2 + p3 = 10 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l2266_226618


namespace max_tickets_sold_l2266_226627

theorem max_tickets_sold (bus_capacity : ℕ) (num_stations : ℕ) (max_capacity : bus_capacity = 25) 
  (total_stations : num_stations = 14) : 
  ∃ (tickets : ℕ), tickets = 67 :=
by 
  sorry

end max_tickets_sold_l2266_226627


namespace solve_equation_in_integers_l2266_226641

theorem solve_equation_in_integers (a b c : ℤ) (h : 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) :
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
sorry

end solve_equation_in_integers_l2266_226641


namespace randy_used_36_blocks_l2266_226684

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks left
def blocks_left : ℕ := 23

-- Define the number of blocks used
def blocks_used (initial left : ℕ) : ℕ := initial - left

-- Prove that Randy used 36 blocks
theorem randy_used_36_blocks : blocks_used initial_blocks blocks_left = 36 := 
by
  -- Proof will be here
  sorry

end randy_used_36_blocks_l2266_226684


namespace carrie_pays_l2266_226633

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l2266_226633


namespace roots_product_eq_348_l2266_226644

theorem roots_product_eq_348 (d e : ℤ) 
  (h : ∀ (s : ℂ), s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) : 
  d * e = 348 :=
sorry

end roots_product_eq_348_l2266_226644


namespace vitamin_A_supplements_per_pack_l2266_226640

theorem vitamin_A_supplements_per_pack {A x y : ℕ} (h1 : A * x = 119) (h2 : 17 * y = 119) : A = 7 :=
by
  sorry

end vitamin_A_supplements_per_pack_l2266_226640


namespace library_visitors_on_sundays_l2266_226658

theorem library_visitors_on_sundays 
  (average_other_days : ℕ) 
  (average_per_day : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ) 
  (total_visitors_month : ℕ)
  (visitors_other_days : ℕ) 
  (total_visitors_sundays : ℕ) :
  average_other_days = 240 →
  average_per_day = 285 →
  total_days = 30 →
  sundays = 5 →
  other_days = total_days - sundays →
  total_visitors_month = average_per_day * total_days →
  visitors_other_days = average_other_days * other_days →
  total_visitors_sundays + visitors_other_days = total_visitors_month →
  total_visitors_sundays = sundays * (510 : ℕ) :=
by
  sorry


end library_visitors_on_sundays_l2266_226658


namespace y_worked_days_l2266_226647

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

end y_worked_days_l2266_226647


namespace pencil_rows_l2266_226674

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) : (total_pencils / pencils_per_row) = 7 :=
by
  sorry

end pencil_rows_l2266_226674


namespace line_equation_direction_point_l2266_226659

theorem line_equation_direction_point 
  (d : ℝ × ℝ) (A : ℝ × ℝ) :
  d = (2, -1) →
  A = (1, 0) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 ↔ x + 2 * y - 1 = 0 :=
by
  sorry

end line_equation_direction_point_l2266_226659


namespace solve_for_x_l2266_226620

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x_l2266_226620
