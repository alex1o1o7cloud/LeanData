import Mathlib

namespace no_real_solutions_l8_8273

theorem no_real_solutions (x : ℝ) : ¬ (3 * x^2 + 5 = |4 * x + 2| - 3) :=
by
  sorry

end no_real_solutions_l8_8273


namespace remainder_when_divided_by_x_minus_2_l8_8466

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 2 * x^2 + 11 * x - 6

theorem remainder_when_divided_by_x_minus_2 :
  (f 2) = 16 := by
  sorry

end remainder_when_divided_by_x_minus_2_l8_8466


namespace fraction_of_loss_l8_8955

theorem fraction_of_loss
  (SP CP : ℚ) (hSP : SP = 16) (hCP : CP = 17) :
  (CP - SP) / CP = 1 / 17 :=
by
  sorry

end fraction_of_loss_l8_8955


namespace vasya_mushrooms_l8_8030

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l8_8030


namespace proposition_p_q_true_l8_8848

def represents_hyperbola (m : ℝ) : Prop := (1 - m) * (m + 2) < 0

def represents_ellipse (m : ℝ) : Prop := (2 * m > 2 - m) ∧ (2 - m > 0)

theorem proposition_p_q_true (m : ℝ) :
  represents_hyperbola m ∧ represents_ellipse m → (1 < m ∧ m < 2) :=
by
  sorry

end proposition_p_q_true_l8_8848


namespace find_number_of_girls_l8_8219

-- Define the ratio of boys to girls as 8:4.
def ratio_boys_to_girls : ℕ × ℕ := (8, 4)

-- Define the total number of students.
def total_students : ℕ := 600

-- Define what it means for the number of girls given a ratio and total students.
def number_of_girls (ratio : ℕ × ℕ) (total : ℕ) : ℕ :=
  let total_parts := (ratio.1 + ratio.2)
  let part_value := total / total_parts
  ratio.2 * part_value

-- State the goal to prove the number of girls is 200 given the conditions.
theorem find_number_of_girls :
  number_of_girls ratio_boys_to_girls total_students = 200 :=
sorry

end find_number_of_girls_l8_8219


namespace find_k_l8_8107

theorem find_k (k : ℝ) :
  (∃ x y : ℝ, y = x + 2 * k ∧ y = 2 * x + k + 1 ∧ x^2 + y^2 = 4) ↔
  (k = 1 ∨ k = -1/5) := 
sorry

end find_k_l8_8107


namespace sequence_length_l8_8543

theorem sequence_length 
  (a : ℕ)
  (b : ℕ)
  (d : ℕ)
  (steps : ℕ)
  (h1 : a = 160)
  (h2 : b = 28)
  (h3 : d = 4)
  (h4 : (28:ℕ) = (160:ℕ) - steps * 4) :
  steps + 1 = 34 :=
by
  sorry

end sequence_length_l8_8543


namespace exists_points_with_small_distance_l8_8727

theorem exists_points_with_small_distance :
  ∃ A B : ℝ × ℝ, (A.2 = A.1^4) ∧ (B.2 = B.1^4 + B.1^2 + B.1 + 1) ∧ 
  (dist A B < 1 / 100) :=
by
  sorry

end exists_points_with_small_distance_l8_8727


namespace power_function_characterization_l8_8551

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_characterization (f : ℝ → ℝ) (h : f 2 = Real.sqrt 2) : 
  ∀ x : ℝ, f x = x ^ (1 / 2) :=
sorry

end power_function_characterization_l8_8551


namespace rent_percentage_l8_8235

noncomputable def condition1 (E : ℝ) : ℝ := 0.25 * E
noncomputable def condition2 (E : ℝ) : ℝ := 1.35 * E
noncomputable def condition3 (E' : ℝ) : ℝ := 0.40 * E'

theorem rent_percentage (E R R' : ℝ) (hR : R = condition1 E) (hE' : E = condition2 E) (hR' : R' = condition3 E) :
  (R' / R) * 100 = 216 :=
sorry

end rent_percentage_l8_8235


namespace number_of_correct_statements_l8_8817

theorem number_of_correct_statements (stmt1: Prop) (stmt2: Prop) (stmt3: Prop) :
  stmt1 ∧ stmt2 ∧ stmt3 → (∀ n, n = 3) :=
by
  sorry

end number_of_correct_statements_l8_8817


namespace range_of_m_l8_8115

theorem range_of_m (m : ℝ) :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + m ≤ 0) → 1 < m := by
  sorry

end range_of_m_l8_8115


namespace height_of_removed_player_l8_8734

theorem height_of_removed_player (S : ℕ) (x : ℕ) (total_height_11 : S + x = 182 * 11)
  (average_height_10 : S = 181 * 10): x = 192 :=
by
  sorry

end height_of_removed_player_l8_8734


namespace find_value_l8_8704

theorem find_value (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a^2006 + (a + b)^2007 = 2 := 
by
  sorry

end find_value_l8_8704


namespace gcd_540_180_diminished_by_2_eq_178_l8_8408

theorem gcd_540_180_diminished_by_2_eq_178 : gcd 540 180 - 2 = 178 := by
  sorry

end gcd_540_180_diminished_by_2_eq_178_l8_8408


namespace problem1_problem2_l8_8934

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Problem 1: Prove that a = sqrt(3) given that x = 1 is an extremum point for h(x, a)
theorem problem1 (a : ℝ) (h_extremum : ∀ x : ℝ, x = 1 → 0 = (2 - a^2 / x^2 + 1 / x)) : a = Real.sqrt 3 := sorry

-- Problem 2: Prove the range of a is [ (e + 1) / 2, +∞ ) such that for any x1, x2 ∈ [1, e], f(x1, a) ≥ g(x2)
theorem problem2 (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f x1 a ≥ g x2) →
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end problem1_problem2_l8_8934


namespace simplify_expression_l8_8247

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 0) : x⁻¹ - 3 * x + 2 = - (3 * x^2 - 2 * x - 1) / x :=
by
  sorry

end simplify_expression_l8_8247


namespace coordinates_of_A_l8_8656

-- Define initial coordinates of point A
def A : ℝ × ℝ := (-2, 4)

-- Define the transformation of moving 2 units upwards
def move_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

-- Define the transformation of moving 3 units to the left
def move_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Combine the transformations to get point A'
def A' : ℝ × ℝ :=
  move_left (move_up A 2) 3

-- The theorem stating that A' is (-5, 6)
theorem coordinates_of_A' : A' = (-5, 6) :=
by
  sorry

end coordinates_of_A_l8_8656


namespace value_of_f_5_l8_8730

variable (a b c m : ℝ)

-- Conditions: definition of f and given value of f(-5)
def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2
axiom H1 : f a b c (-5) = m

-- Question: Prove that f(5) = -m + 4
theorem value_of_f_5 : f a b c 5 = -m + 4 :=
by
  sorry

end value_of_f_5_l8_8730


namespace equations_solution_l8_8825

-- Definition of the conditions
def equation1 := ∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)
def equation2 := ∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)

-- The main statement combining both problems
theorem equations_solution :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)) := by
  sorry

end equations_solution_l8_8825


namespace solve_quadratic_equation_l8_8914

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l8_8914


namespace smallest_is_B_l8_8299

def A : ℕ := 32 + 7
def B : ℕ := (3 * 10) + 3
def C : ℕ := 50 - 9

theorem smallest_is_B : min A (min B C) = B := 
by 
  have hA : A = 39 := by rfl
  have hB : B = 33 := by rfl
  have hC : C = 41 := by rfl
  rw [hA, hB, hC]
  exact sorry

end smallest_is_B_l8_8299


namespace totalHighlighters_l8_8667

-- Define the number of each type of highlighter
def pinkHighlighters : ℕ := 10
def yellowHighlighters : ℕ := 15
def blueHighlighters : ℕ := 8

-- State the theorem to prove
theorem totalHighlighters :
  pinkHighlighters + yellowHighlighters + blueHighlighters = 33 :=
by
  -- Proof to be filled
  sorry

end totalHighlighters_l8_8667


namespace one_cow_one_bag_l8_8712

def husk_eating (C B D : ℕ) : Prop :=
  C * D / B = D

theorem one_cow_one_bag (C B D n : ℕ) (h : husk_eating C B D) (hC : C = 46) (hB : B = 46) (hD : D = 46) : n = D :=
by
  rw [hC, hB, hD] at h
  sorry

end one_cow_one_bag_l8_8712


namespace find_number_l8_8066

def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_digit_is_three (n : ℕ) : Prop :=
  n / 1000 = 3

def last_digit_is_five (n : ℕ) : Prop :=
  n % 10 = 5

theorem find_number :
  ∃ (x : ℕ), four_digit_number (x^2) ∧ first_digit_is_three (x^2) ∧ last_digit_is_five (x^2) ∧ x = 55 :=
sorry

end find_number_l8_8066


namespace compute_sum_l8_8879
-- Import the necessary library to have access to the required definitions and theorems.

-- Define the integers involved based on the conditions.
def a : ℕ := 157
def b : ℕ := 43
def c : ℕ := 19
def d : ℕ := 81

-- State the theorem that computes the sum of these integers and equate it to 300.
theorem compute_sum : a + b + c + d = 300 := by
  sorry

end compute_sum_l8_8879


namespace simplify_complex_expression_l8_8600

variables (x y : ℝ) (i : ℂ)

theorem simplify_complex_expression (h : i^2 = -1) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 :=
by sorry

end simplify_complex_expression_l8_8600


namespace ellipse_eccentricity_m_l8_8615

theorem ellipse_eccentricity_m (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, x^2 / m + y^2 = 1) (h2 : e = Real.sqrt 3 / 2) :
  m = 4 ∨ m = 1 / 4 :=
by sorry

end ellipse_eccentricity_m_l8_8615


namespace LeRoy_should_pay_30_l8_8556

/-- Define the empirical amounts paid by LeRoy and Bernardo, and the total discount. -/
def LeRoy_paid : ℕ := 240
def Bernardo_paid : ℕ := 360
def total_discount : ℕ := 60

/-- Define total expenses pre-discount. -/
def total_expenses : ℕ := LeRoy_paid + Bernardo_paid

/-- Define total expenses post-discount. -/
def adjusted_expenses : ℕ := total_expenses - total_discount

/-- Define each person's adjusted share. -/
def each_adjusted_share : ℕ := adjusted_expenses / 2

/-- Define the amount LeRoy should pay Bernardo. -/
def leroy_to_pay : ℕ := each_adjusted_share - LeRoy_paid

/-- Prove that LeRoy should pay Bernardo $30 to equalize their expenses post-discount. -/
theorem LeRoy_should_pay_30 : leroy_to_pay = 30 :=
by 
  -- Proof goes here...
  sorry

end LeRoy_should_pay_30_l8_8556


namespace find_base_k_l8_8941

theorem find_base_k (k : ℕ) (h1 : 1 + 3 * k + 2 * k^2 = 30) : k = 4 :=
by sorry

end find_base_k_l8_8941


namespace mike_salary_calculation_l8_8739

theorem mike_salary_calculation
  (F : ℝ) (M : ℝ) (new_M : ℝ) (x : ℝ)
  (F_eq : F = 1000)
  (M_eq : M = x * F)
  (increase_eq : new_M = 1.40 * M)
  (new_M_val : new_M = 15400) :
  M = 11000 ∧ x = 11 :=
by
  sorry

end mike_salary_calculation_l8_8739


namespace percentage_decrease_l8_8754

theorem percentage_decrease (original_price new_price : ℝ) (h₁ : original_price = 700) (h₂ : new_price = 532) : 
  ((original_price - new_price) / original_price) * 100 = 24 := by
  sorry

end percentage_decrease_l8_8754


namespace coins_amount_correct_l8_8562

-- Definitions based on the conditions
def cost_of_flour : ℕ := 5
def cost_of_cake_stand : ℕ := 28
def amount_given_in_bills : ℕ := 20 + 20
def change_received : ℕ := 10

-- Total cost of items
def total_cost : ℕ := cost_of_flour + cost_of_cake_stand

-- Total money given
def total_money_given : ℕ := total_cost + change_received

-- Amount given in loose coins
def loose_coins_given : ℕ := total_money_given - amount_given_in_bills

-- Proposition statement
theorem coins_amount_correct : loose_coins_given = 3 := by
  sorry

end coins_amount_correct_l8_8562


namespace find_second_offset_l8_8119

-- Define the given constants
def diagonal : ℝ := 30
def offset1 : ℝ := 10
def area : ℝ := 240

-- The theorem we want to prove
theorem find_second_offset : ∃ (offset2 : ℝ), area = (1 / 2) * diagonal * (offset1 + offset2) ∧ offset2 = 6 :=
sorry

end find_second_offset_l8_8119


namespace sequence_sum_after_6_steps_l8_8076

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 1435 -- would define how numbers sequence works recursively.
  else sorry -- next steps up to 6
  

theorem sequence_sum_after_6_steps : sequence_sum 6 = 191 := 
by
  sorry

end sequence_sum_after_6_steps_l8_8076


namespace food_per_puppy_meal_l8_8855

-- Definitions for conditions
def mom_daily_food : ℝ := 1.5 * 3
def num_puppies : ℕ := 5
def total_food_needed : ℝ := 57
def num_days : ℕ := 6

-- Total food for the mom dog over the given period
def total_mom_food : ℝ := mom_daily_food * num_days

-- Total food for the puppies over the given period
def total_puppy_food : ℝ := total_food_needed - total_mom_food

-- Total number of puppy meals over the given period
def total_puppy_meals : ℕ := (num_puppies * 2) * num_days

theorem food_per_puppy_meal :
  total_puppy_food / total_puppy_meals = 0.5 :=
  sorry

end food_per_puppy_meal_l8_8855


namespace rectangle_area_inscribed_circle_l8_8266

theorem rectangle_area_inscribed_circle 
  (radius : ℝ) (width len : ℝ) 
  (h_radius : radius = 5) 
  (h_width : width = 2 * radius) 
  (h_len_ratio : len = 3 * width) 
  : width * len = 300 := 
by
  sorry

end rectangle_area_inscribed_circle_l8_8266


namespace total_dolphins_l8_8421

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l8_8421


namespace number_of_badminton_players_l8_8482

-- Definitions based on the given conditions
variable (Total_members : ℕ := 30)
variable (Tennis_players : ℕ := 19)
variable (No_sport_players : ℕ := 3)
variable (Both_sport_players : ℕ := 9)

-- The goal is to prove the number of badminton players is 17
theorem number_of_badminton_players :
  ∀ (B : ℕ), Total_members = B + Tennis_players - Both_sport_players + No_sport_players → B = 17 :=
by
  intro B
  intro h
  sorry

end number_of_badminton_players_l8_8482


namespace expenditure_on_house_rent_l8_8811

theorem expenditure_on_house_rent (I : ℝ) (H1 : 0.30 * I = 300) : 0.20 * (I - 0.30 * I) = 140 :=
by
  -- Skip the proof, the statement is sufficient at this stage.
  sorry

end expenditure_on_house_rent_l8_8811


namespace interest_percentage_calculation_l8_8329

-- Definitions based on problem conditions
def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℕ := 12

-- Theorem statement:
theorem interest_percentage_calculation :
  let total_paid := down_payment + (monthly_payment * number_of_monthly_payments)
  let interest_paid := total_paid - purchase_price
  let interest_percent := (interest_paid / purchase_price) * 100
  interest_percent = 18.2 :=
by sorry

end interest_percentage_calculation_l8_8329


namespace find_value_l8_8572

-- Define the variables and given conditions
variables (x y z : ℚ)
variables (h1 : 2 * x - y = 4)
variables (h2 : 3 * x + z = 7)
variables (h3 : y = 2 * z)

-- Define the goal to prove
theorem find_value : 6 * x - 3 * y + 3 * z = 51 / 4 := by 
  sorry

end find_value_l8_8572


namespace find_constants_l8_8064

theorem find_constants (P Q R : ℚ) (h : ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) = (P / (x - 1)) + (Q / (x - 4)) + (R / (x - 6))) : 
  (P, Q, R) = (-4/5, -1/2, 23/10) := 
  sorry

end find_constants_l8_8064


namespace rectangle_square_ratio_l8_8238

theorem rectangle_square_ratio (l w s : ℝ) (h1 : 0.4 * l * w = 0.25 * s * s) : l / w = 15.625 :=
by
  sorry

end rectangle_square_ratio_l8_8238


namespace mod_pow_eq_l8_8664

theorem mod_pow_eq (m : ℕ) (h1 : 13^4 % 11 = m) (h2 : 0 ≤ m ∧ m < 11) : m = 5 := by
  sorry

end mod_pow_eq_l8_8664


namespace pencils_per_row_cannot_be_determined_l8_8547

theorem pencils_per_row_cannot_be_determined
  (rows : ℕ)
  (total_crayons : ℕ)
  (crayons_per_row : ℕ)
  (h_total_crayons: total_crayons = 210)
  (h_rows: rows = 7)
  (h_crayons_per_row: crayons_per_row = 30) :
  ∀ (pencils_per_row : ℕ), false :=
by
  sorry

end pencils_per_row_cannot_be_determined_l8_8547


namespace scale_down_multiplication_l8_8936

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l8_8936


namespace rowing_upstream_speed_l8_8567

theorem rowing_upstream_speed (V_down V_m : ℝ) (h_down : V_down = 35) (h_still : V_m = 31) : ∃ V_up, V_up = V_m - (V_down - V_m) ∧ V_up = 27 := by
  sorry

end rowing_upstream_speed_l8_8567


namespace remainder_divisibility_l8_8074

theorem remainder_divisibility (n : ℕ) (d : ℕ) (r : ℕ) : 
  let n := 1234567
  let d := 256
  let r := n % d
  r = 933 ∧ ¬ (r % 7 = 0) := by
  sorry

end remainder_divisibility_l8_8074


namespace f_odd_f_decreasing_f_extremum_l8_8450

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_val : f 1 = -2
axiom f_neg : ∀ x > 0, f x < 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem f_extremum : ∃ (max min : ℝ), max = f (-3) ∧ min = f 3 :=
sorry

end f_odd_f_decreasing_f_extremum_l8_8450


namespace correct_statements_are_two_l8_8437

def statement1 : Prop := 
  ∀ (data : Type) (eq : data → data → Prop), 
    (∃ (t : data), eq t t) → 
    (∀ (d1 d2 : data), eq d1 d2 → d1 = d2)

def statement2 : Prop := 
  ∀ (samplevals : Type) (regress_eqn : samplevals → samplevals → Prop), 
    (∃ (s : samplevals), regress_eqn s s) → 
    (∀ (sv1 sv2 : samplevals), regress_eqn sv1 sv2 → sv1 = sv2)

def statement3 : Prop := 
  ∀ (predvals : Type) (pred_eqn : predvals → predvals → Prop), 
    (∃ (p : predvals), pred_eqn p p) → 
    (∀ (pp1 pp2 : predvals), pred_eqn pp1 pp2 → pp1 = pp2)

def statement4 : Prop := 
  ∀ (observedvals : Type) (linear_eqn : observedvals → observedvals → Prop), 
    (∃ (o : observedvals), linear_eqn o o) → 
    (∀ (ov1 ov2 : observedvals), linear_eqn ov1 ov2 → ov1 = ov2)

def correct_statements_count : ℕ := 2

theorem correct_statements_are_two : 
  (statement1 ∧ statement2 ∧ ¬ statement3 ∧ ¬ statement4) → 
  correct_statements_count = 2 := by
  sorry

end correct_statements_are_two_l8_8437


namespace students_play_alto_saxophone_l8_8249

def roosevelt_high_school :=
  let total_students := 600
  let marching_band_students := total_students / 5
  let brass_instrument_students := marching_band_students / 2
  let saxophone_students := brass_instrument_students / 5
  let alto_saxophone_students := saxophone_students / 3
  alto_saxophone_students

theorem students_play_alto_saxophone :
  roosevelt_high_school = 4 :=
  by
    sorry

end students_play_alto_saxophone_l8_8249


namespace inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l8_8293

theorem inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed:
  (∀ a b : ℝ, a > b → a^3 > b^3) → (∀ a b : ℝ, a^3 > b^3 → a > b) :=
  by
  sorry

end inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l8_8293


namespace fraction_evaluation_l8_8304

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem fraction_evaluation :
  (sqrt 2 * (sqrt 3 - sqrt 7)) / (2 * sqrt (3 + sqrt 5)) =
  (30 - 10 * sqrt 5 - 6 * sqrt 21 + 2 * sqrt 105) / 8 :=
by
  sorry

end fraction_evaluation_l8_8304


namespace focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l8_8318

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let p := b^2 / (4 * a) - c / (4 * a)
  (p, 1 / (4 * a))

theorem focus_parabola_y_eq_neg4x2_plus_4x_minus_1 :
  focus_of_parabola (-4) 4 (-1) = (1 / 2, -1 / 8) :=
sorry

end focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l8_8318


namespace probabilities_inequalities_l8_8870

variables (M N : Prop) (P : Prop → ℝ)

axiom P_pos_M : P M > 0
axiom P_pos_N : P N > 0
axiom P_cond_N_M : P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)

theorem probabilities_inequalities :
    (P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)) ∧
    (P (N ∧ M) > P N * P M) ∧
    (P (M ∧ N) / P N > P (M ∧ ¬N) / P (¬N)) :=
by
    sorry

end probabilities_inequalities_l8_8870


namespace minute_hand_travel_distance_l8_8769

theorem minute_hand_travel_distance :
  ∀ (r : ℝ), r = 8 → (45 / 60) * (2 * Real.pi * r) = 12 * Real.pi :=
by
  intros r r_eq
  sorry

end minute_hand_travel_distance_l8_8769


namespace ratio_of_down_payment_l8_8372

theorem ratio_of_down_payment (C D : ℕ) (daily_min : ℕ) (days : ℕ) (balance : ℕ) (total_cost : ℕ) 
  (h1 : total_cost = 120)
  (h2 : daily_min = 6)
  (h3 : days = 10)
  (h4 : balance = daily_min * days) 
  (h5 : D + balance = total_cost) : 
  D / total_cost = 1 / 2 := 
  by
  sorry

end ratio_of_down_payment_l8_8372


namespace find_non_negative_integer_solutions_l8_8578

theorem find_non_negative_integer_solutions :
  ∃ (x y z w : ℕ), 2 ^ x * 3 ^ y - 5 ^ z * 7 ^ w = 1 ∧
  ((x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
   (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1)) := by
  sorry

end find_non_negative_integer_solutions_l8_8578


namespace greatest_possible_gcd_l8_8052

theorem greatest_possible_gcd (d : ℕ) (a : ℕ → ℕ) (h_sum : (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 595)
  (h_gcd : ∀ i, d ∣ a i) : d ≤ 35 :=
sorry

end greatest_possible_gcd_l8_8052


namespace b_alone_work_time_l8_8449

def work_rate_combined (a_rate b_rate : ℝ) : ℝ := a_rate + b_rate

theorem b_alone_work_time
  (a_rate b_rate : ℝ)
  (h1 : work_rate_combined a_rate b_rate = 1/16)
  (h2 : a_rate = 1/20) :
  b_rate = 1/80 := by
  sorry

end b_alone_work_time_l8_8449


namespace probability_of_heart_and_joker_l8_8229

-- Define a deck with 54 cards, including jokers
def total_cards : ℕ := 54

-- Define the count of specific cards in the deck
def hearts_count : ℕ := 13
def jokers_count : ℕ := 2
def remaining_cards (x: ℕ) : ℕ := total_cards - x

-- Define the probability of drawing a specific card
def prob_of_first_heart : ℚ := hearts_count / total_cards
def prob_of_second_joker (first_card_a_heart: Bool) : ℚ :=
  if first_card_a_heart then jokers_count / remaining_cards 1 else 0

-- Calculate the probability of drawing a heart first and then a joker
def prob_first_heart_then_joker : ℚ :=
  prob_of_first_heart * prob_of_second_joker true

-- Proving the final probability
theorem probability_of_heart_and_joker :
  prob_first_heart_then_joker = 13 / 1419 := by
  -- Skipping the proof
  sorry

end probability_of_heart_and_joker_l8_8229


namespace evaluate_trig_expression_l8_8889

theorem evaluate_trig_expression :
  (Real.tan (π / 18) - Real.sqrt 3) * Real.sin (2 * π / 9) = -1 :=
by
  sorry

end evaluate_trig_expression_l8_8889


namespace reciprocal_of_x_l8_8154

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2 * x^2 = 0) (h2 : x ≠ 0) : x = 2 → (1 / x = 1 / 2) :=
by {
  sorry
}

end reciprocal_of_x_l8_8154


namespace inequality_for_a_and_b_l8_8335

theorem inequality_for_a_and_b (a b : ℝ) : 
  (1 / 3 * a - b) ≤ 5 :=
sorry

end inequality_for_a_and_b_l8_8335


namespace jane_reading_speed_second_half_l8_8594

-- Definitions from the problem's conditions
def total_pages : ℕ := 500
def first_half_pages : ℕ := total_pages / 2
def first_half_speed : ℕ := 10
def total_days : ℕ := 75

-- The number of days spent reading the first half
def first_half_days : ℕ := first_half_pages / first_half_speed

-- The number of days spent reading the second half
def second_half_days : ℕ := total_days - first_half_days

-- The number of pages in the second half
def second_half_pages : ℕ := total_pages - first_half_pages

-- The actual theorem stating that Jane's reading speed for the second half was 5 pages per day
theorem jane_reading_speed_second_half :
  second_half_pages / second_half_days = 5 :=
by
  sorry

end jane_reading_speed_second_half_l8_8594


namespace power_equation_l8_8768

theorem power_equation (x a b : ℝ) (ha : 3^x = a) (hb : 5^x = b) : 45^x = a^2 * b :=
sorry

end power_equation_l8_8768


namespace ben_has_20_mms_l8_8737

theorem ben_has_20_mms (B_candies Ben_candies : ℕ) 
  (h1 : B_candies = 50) 
  (h2 : B_candies = Ben_candies + 30) : 
  Ben_candies = 20 := 
by
  sorry

end ben_has_20_mms_l8_8737


namespace compute_54_mul_46_l8_8497

theorem compute_54_mul_46 : (54 * 46 = 2484) :=
by sorry

end compute_54_mul_46_l8_8497


namespace common_root_for_permutations_of_coeffs_l8_8131

theorem common_root_for_permutations_of_coeffs :
  ∀ (a b c d : ℤ), (a = -7 ∨ a = 4 ∨ a = -3 ∨ a = 6) ∧ 
                   (b = -7 ∨ b = 4 ∨ b = -3 ∨ b = 6) ∧
                   (c = -7 ∨ c = 4 ∨ c = -3 ∨ c = 6) ∧
                   (d = -7 ∨ d = 4 ∨ d = -3 ∨ d = 6) ∧
                   (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * 1^3 + b * 1^2 + c * 1 + d = 0) :=
by
  intros a b c d h
  sorry

end common_root_for_permutations_of_coeffs_l8_8131


namespace john_alone_finishes_in_48_days_l8_8990

theorem john_alone_finishes_in_48_days (J R : ℝ) (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 16 / 24) (h3 : ∀ T : ℝ, J * T = 1 → T = 48) : 
  (J = 1 / 48) → (∀ T : ℝ, J * T = 1 → T = 48) :=
by
  intro hJohn
  sorry

end john_alone_finishes_in_48_days_l8_8990


namespace sum_of_sampled_types_l8_8230

-- Define the types of books in each category
def Chinese_types := 20
def Mathematics_types := 10
def Liberal_Arts_Comprehensive_types := 40
def English_types := 30

-- Define the total types of books
def total_types := Chinese_types + Mathematics_types + Liberal_Arts_Comprehensive_types + English_types

-- Define the sample size and stratified sampling ratio
def sample_size := 20
def sampling_ratio := sample_size / total_types

-- Define the number of types sampled from each category
def Mathematics_sampled := Mathematics_types * sampling_ratio
def Liberal_Arts_Comprehensive_sampled := Liberal_Arts_Comprehensive_types * sampling_ratio

-- Define the proof statement
theorem sum_of_sampled_types : Mathematics_sampled + Liberal_Arts_Comprehensive_sampled = 10 :=
by
  -- Your proof here
  sorry

end sum_of_sampled_types_l8_8230


namespace cos_theta_seven_l8_8376

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l8_8376


namespace jumping_contest_l8_8360

variables (G F M K : ℤ)

-- Define the conditions
def condition_1 : Prop := G = 39
def condition_2 : Prop := G = F + 19
def condition_3 : Prop := M = F - 12
def condition_4 : Prop := K = 2 * F - 5

-- The theorem asserting the final distances
theorem jumping_contest 
    (h1 : condition_1 G)
    (h2 : condition_2 G F)
    (h3 : condition_3 F M)
    (h4 : condition_4 F K) :
    G = 39 ∧ F = 20 ∧ M = 8 ∧ K = 35 := by
  sorry

end jumping_contest_l8_8360


namespace salt_solution_mixture_l8_8685

/-- Let's define the conditions and hypotheses required for our proof. -/
def ounces_of_salt_solution 
  (percent_salt : ℝ) (amount : ℝ) : ℝ := percent_salt * amount

def final_amount (x : ℝ) : ℝ := x + 70
def final_salt_content (x : ℝ) : ℝ := 0.40 * (x + 70)

theorem salt_solution_mixture (x : ℝ) :
  0.60 * x + 0.20 * 70 = 0.40 * (x + 70) ↔ x = 70 :=
by {
  sorry
}

end salt_solution_mixture_l8_8685


namespace dimes_difference_l8_8182

theorem dimes_difference (a b c : ℕ) :
  a + b + c = 120 →
  5 * a + 10 * b + 25 * c = 1265 →
  c ≥ 10 →
  (max (b) - min (b)) = 92 :=
sorry

end dimes_difference_l8_8182


namespace max_base_angle_is_7_l8_8666

-- Define the conditions and the problem statement
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isosceles_triangle (x : ℕ) : Prop :=
  is_prime x ∧ ∃ y : ℕ, 2 * x + y = 180 ∧ is_prime y

theorem max_base_angle_is_7 :
  ∃ (x : ℕ), isosceles_triangle x ∧ x = 7 :=
by
  sorry

end max_base_angle_is_7_l8_8666


namespace triangle_circle_property_l8_8847

-- Let a, b, and c be the lengths of the sides of a right triangle, where c is the hypotenuse.
variables {a b c : ℝ}

-- Let varrho_b be the radius of the circle inscribed around the leg b of the triangle.
variable {varrho_b : ℝ}

-- Assume the relationship a^2 + b^2 = c^2 (Pythagorean theorem).
axiom right_triangle : a^2 + b^2 = c^2

-- Prove that b + c = a + 2 * varrho_b
theorem triangle_circle_property (h : a^2 + b^2 = c^2) (radius_condition : varrho_b = (a*b)/(a+c-b)) : 
  b + c = a + 2 * varrho_b :=
sorry

end triangle_circle_property_l8_8847


namespace find_x_when_y_neg_10_l8_8017

def inversely_proportional (x y : ℝ) (k : ℝ) := x * y = k

theorem find_x_when_y_neg_10 (k : ℝ) (h₁ : inversely_proportional 4 (-2) k) (yval : y = -10) 
: ∃ x, inversely_proportional x y k ∧ x = 4 / 5 := by
  sorry

end find_x_when_y_neg_10_l8_8017


namespace upper_limit_of_prime_range_l8_8755

theorem upper_limit_of_prime_range : 
  ∃ x : ℝ, (26 / 3 < 11) ∧ (11 < x) ∧ (x < 17) :=
by
  sorry

end upper_limit_of_prime_range_l8_8755


namespace base9_addition_l8_8404

-- Define the numbers in base 9
def num1 : ℕ := 1 * 9^2 + 7 * 9^1 + 5 * 9^0
def num2 : ℕ := 7 * 9^2 + 1 * 9^1 + 4 * 9^0
def num3 : ℕ := 6 * 9^1 + 1 * 9^0
def result : ℕ := 1 * 9^3 + 0 * 9^2 + 6 * 9^1 + 1 * 9^0

-- State the theorem
theorem base9_addition : num1 + num2 + num3 = result := by
  sorry

end base9_addition_l8_8404


namespace parabolic_arch_height_l8_8093

noncomputable def arch_height (a : ℝ) : ℝ :=
  a * (0 : ℝ)^2

theorem parabolic_arch_height :
  ∃ (a : ℝ), (∫ x in (-4 : ℝ)..4, a * x^2) = (160 : ℝ) ∧ arch_height a = 30 :=
by
  sorry

end parabolic_arch_height_l8_8093


namespace find_larger_number_l8_8468

theorem find_larger_number (x y : ℝ) (h1 : 4 * y = 6 * x) (h2 : x + y = 36) : y = 21.6 :=
by
  sorry

end find_larger_number_l8_8468


namespace total_rainfall_2004_l8_8818

theorem total_rainfall_2004 (average_rainfall_2003 : ℝ) (increase_percentage : ℝ) (months : ℝ) :
  average_rainfall_2003 = 36 →
  increase_percentage = 0.10 →
  months = 12 →
  (average_rainfall_2003 * (1 + increase_percentage) * months) = 475.2 :=
by
  -- The proof is left as an exercise
  sorry

end total_rainfall_2004_l8_8818


namespace tram_speed_l8_8269

theorem tram_speed
  (L v : ℝ)
  (h1 : L = 2 * v)
  (h2 : 96 + L = 10 * v) :
  v = 12 := 
by sorry

end tram_speed_l8_8269


namespace scooterValue_after_4_years_with_maintenance_l8_8284

noncomputable def scooterDepreciation (initial_value : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((3 : ℝ) / 4) ^ years

theorem scooterValue_after_4_years_with_maintenance (M : ℝ) :
  scooterDepreciation 40000 4 - 4 * M = 12656.25 - 4 * M :=
by
  sorry

end scooterValue_after_4_years_with_maintenance_l8_8284


namespace range_of_a_l8_8713

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + 1 / 4 > 0) ↔ (Real.sqrt 5 - 3) / 2 < a ∧ a < (3 + Real.sqrt 5) / 2 :=
by
  sorry

end range_of_a_l8_8713


namespace tree_count_l8_8884

theorem tree_count (m N : ℕ) 
  (h1 : 12 ≡ (33 - m) [MOD N])
  (h2 : (105 - m) ≡ 8 [MOD N]) :
  N = 76 := 
sorry

end tree_count_l8_8884


namespace fraction_subtraction_inequality_l8_8239

theorem fraction_subtraction_inequality (a b n : ℕ) (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a : ℚ) / b > (a - n : ℚ) / (b - n) :=
sorry

end fraction_subtraction_inequality_l8_8239


namespace find_A_l8_8915

def hash_rel (A B : ℝ) := A^2 + B^2

theorem find_A (A : ℝ) (h : hash_rel A 7 = 196) : A = 7 * Real.sqrt 3 :=
by sorry

end find_A_l8_8915


namespace baseball_team_games_l8_8863

theorem baseball_team_games (P Q : ℕ) (hP : P > 3 * Q) (hQ : Q > 3) (hTotal : 2 * P + 6 * Q = 78) :
  2 * P = 54 :=
by
  -- placeholder for the actual proof
  sorry

end baseball_team_games_l8_8863


namespace proof_problem_l8_8709

def work_problem :=
  ∃ (B : ℝ),
  (1 / 6) + (1 / B) + (1 / 24) = (1 / 3) ∧ B = 8

theorem proof_problem : work_problem :=
by
  sorry

end proof_problem_l8_8709


namespace chocolates_total_l8_8981

theorem chocolates_total (x : ℕ)
  (h1 : x - 12 + x - 18 + x - 20 = 2 * x) :
  x = 50 :=
  sorry

end chocolates_total_l8_8981


namespace marble_count_l8_8669

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l8_8669


namespace simplify_expression_l8_8921

variable (a b : ℤ)

theorem simplify_expression : 
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := 
by sorry

end simplify_expression_l8_8921


namespace morning_registration_count_l8_8444

variable (M : ℕ) -- Number of students registered for the morning session
variable (MorningAbsentees : ℕ := 3) -- Absentees in the morning session
variable (AfternoonRegistered : ℕ := 24) -- Students registered for the afternoon session
variable (AfternoonAbsentees : ℕ := 4) -- Absentees in the afternoon session

theorem morning_registration_count :
  (M - MorningAbsentees) + (AfternoonRegistered - AfternoonAbsentees) = 42 → M = 25 :=
by
  sorry

end morning_registration_count_l8_8444


namespace zara_goats_l8_8682

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l8_8682


namespace apple_equation_l8_8649

-- Conditions directly from a)
def condition1 (x : ℕ) : Prop := (x - 1) % 3 = 0
def condition2 (x : ℕ) : Prop := (x + 2) % 4 = 0

theorem apple_equation (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : 
  (x - 1) / 3 = (x + 2) / 4 := 
sorry

end apple_equation_l8_8649


namespace tenth_term_arithmetic_sequence_l8_8488

theorem tenth_term_arithmetic_sequence :
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  aₙ 10 = 3 :=
by
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  show aₙ 10 = 3
  sorry

end tenth_term_arithmetic_sequence_l8_8488


namespace trigonometric_identity_l8_8943

theorem trigonometric_identity (α : Real) (h : Real.sin (Real.pi + α) = -1/3) : 
  (Real.sin (2 * α) / Real.cos α) = 2/3 :=
by
  sorry

end trigonometric_identity_l8_8943


namespace initial_amount_saved_l8_8447

noncomputable section

def cost_of_couch : ℝ := 750
def cost_of_table : ℝ := 100
def cost_of_lamp : ℝ := 50
def amount_still_owed : ℝ := 400

def total_cost : ℝ := cost_of_couch + cost_of_table + cost_of_lamp

theorem initial_amount_saved (initial_amount : ℝ) :
  initial_amount = total_cost - amount_still_owed ↔ initial_amount = 500 :=
by
  -- the proof is omitted
  sorry

end initial_amount_saved_l8_8447


namespace highest_y_coordinate_l8_8160

theorem highest_y_coordinate : 
  (∀ x y : ℝ, ((x - 4)^2 / 25 + y^2 / 49 = 0) → y = 0) := 
by
  sorry

end highest_y_coordinate_l8_8160


namespace find_non_zero_real_x_satisfies_equation_l8_8098

theorem find_non_zero_real_x_satisfies_equation :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 - (18 * x) ^ 9 = 0 ∧ x = 2 :=
by
  sorry

end find_non_zero_real_x_satisfies_equation_l8_8098


namespace probability_of_diff_by_three_is_one_eighth_l8_8428

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l8_8428


namespace find_cost_price_l8_8439

noncomputable def original_cost_price (C S C_new S_new : ℝ) : Prop :=
  S = 1.25 * C ∧
  C_new = 0.80 * C ∧
  S_new = 1.25 * C - 10.50 ∧
  S_new = 1.04 * C

theorem find_cost_price (C S C_new S_new : ℝ) :
  original_cost_price C S C_new S_new → C = 50 :=
by
  sorry

end find_cost_price_l8_8439


namespace completing_the_square_step_l8_8908

theorem completing_the_square_step (x : ℝ) : 
  x^2 + 4 * x + 2 = 0 → x^2 + 4 * x = -2 :=
by
  intro h
  sorry

end completing_the_square_step_l8_8908


namespace sum_of_reciprocals_of_squares_l8_8938

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 19) : 1 / (a * a : ℚ) + 1 / (b * b : ℚ) = 362 / 361 := 
by
  sorry

end sum_of_reciprocals_of_squares_l8_8938


namespace necessary_not_sufficient_condition_l8_8086

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the condition for the problem
def condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the proof problem: Prove that the interval is a necessary but not sufficient condition for f(x) < 0
theorem necessary_not_sufficient_condition : 
  ∀ x : ℝ, condition x → ¬ (∀ y : ℝ, condition y → f y < 0) :=
sorry

end necessary_not_sufficient_condition_l8_8086


namespace number_of_real_solutions_eq_2_l8_8574

theorem number_of_real_solutions_eq_2 :
  ∃! (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3 :=
sorry

end number_of_real_solutions_eq_2_l8_8574


namespace breadthOfRectangularPart_l8_8762

variable (b l : ℝ)

def rectangularAreaProblem : Prop :=
  (l * b + (1 / 12) * b * l = 24 * b) ∧ (l - b = 10)

theorem breadthOfRectangularPart :
  rectangularAreaProblem b l → b = 12.15 :=
by
  intros
  sorry

end breadthOfRectangularPart_l8_8762


namespace train_length_l8_8764

/-
  Given:
  - Speed of the train is 78 km/h
  - Time to pass an electric pole is 5.0769230769230775 seconds
  We need to prove that the length of the train is 110 meters.
-/

def speed_kmph : ℝ := 78
def time_seconds : ℝ := 5.0769230769230775
def expected_length_meters : ℝ := 110

theorem train_length :
  (speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters :=
by {
  -- Proof goes here
  sorry
}

end train_length_l8_8764


namespace yogurt_amount_l8_8897

-- Conditions
def total_ingredients : ℝ := 0.5
def strawberries : ℝ := 0.2
def orange_juice : ℝ := 0.2

-- Question and Answer (Proof Goal)
theorem yogurt_amount : total_ingredients - strawberries - orange_juice = 0.1 := by
  -- Since calculation involves specifics, we add sorry to indicate the proof is skipped
  sorry

end yogurt_amount_l8_8897


namespace Debby_drinks_five_bottles_per_day_l8_8001

theorem Debby_drinks_five_bottles_per_day (total_bottles : ℕ) (days : ℕ) (h1 : total_bottles = 355) (h2 : days = 71) : (total_bottles / days) = 5 :=
by 
  sorry

end Debby_drinks_five_bottles_per_day_l8_8001


namespace max_blocks_that_fit_l8_8905

noncomputable def box_volume : ℕ :=
  3 * 4 * 2

noncomputable def block_volume : ℕ :=
  2 * 1 * 2

noncomputable def max_blocks (box_volume : ℕ) (block_volume : ℕ) : ℕ :=
  box_volume / block_volume

theorem max_blocks_that_fit : max_blocks box_volume block_volume = 6 :=
by
  sorry

end max_blocks_that_fit_l8_8905


namespace triangle_third_side_count_l8_8357

theorem triangle_third_side_count : 
  ∀ (x : ℕ), (3 < x ∧ x < 19) → ∃ (n : ℕ), n = 15 := 
by 
  sorry

end triangle_third_side_count_l8_8357


namespace correct_average_l8_8138

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (wrong_num correct_num : ℝ)
  (h_avg : incorrect_avg = 23)
  (h_n : n = 10)
  (h_wrong : wrong_num = 26)
  (h_correct : correct_num = 36) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 24 :=
by
  -- Proof goes here
  sorry

end correct_average_l8_8138


namespace equivalent_single_discount_calculation_l8_8665

-- Definitions for the successive discounts
def discount10 (x : ℝ) : ℝ := 0.90 * x
def discount15 (x : ℝ) : ℝ := 0.85 * x
def discount25 (x : ℝ) : ℝ := 0.75 * x

-- Final price after applying all discounts
def final_price (x : ℝ) : ℝ := discount25 (discount15 (discount10 x))

-- Equivalent single discount fraction
def equivalent_discount (x : ℝ) : ℝ := 0.57375 * x

theorem equivalent_single_discount_calculation (x : ℝ) : 
  final_price x = equivalent_discount x :=
sorry

end equivalent_single_discount_calculation_l8_8665


namespace simplify_to_ap_minus_b_l8_8346

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((7*p + 3) - 3*p * 2) * 4 + (5 - 2 / 4) * (8*p - 12)

theorem simplify_to_ap_minus_b (p : ℝ) :
  simplify_expression p = 40 * p - 42 :=
by
  -- Proof steps would go here
  sorry

end simplify_to_ap_minus_b_l8_8346


namespace droid_weekly_coffee_consumption_l8_8347

noncomputable def weekly_consumption_A : ℕ :=
  (3 * 5) + 4 + 2 + 1 -- Weekdays + Saturday + Sunday + Monday increase

noncomputable def weekly_consumption_B : ℕ :=
  (2 * 5) + 3 + (1 - 1 / 2) -- Weekdays + Saturday + Sunday decrease

noncomputable def weekly_consumption_C : ℕ :=
  (1 * 5) + 2 + 1 -- Weekdays + Saturday + Sunday

theorem droid_weekly_coffee_consumption :
  weekly_consumption_A = 22 ∧ weekly_consumption_B = 14 ∧ weekly_consumption_C = 8 :=
by 
  sorry

end droid_weekly_coffee_consumption_l8_8347


namespace determine_digit_phi_l8_8531

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l8_8531


namespace sum_of_two_numbers_l8_8899

theorem sum_of_two_numbers (S : ℝ) (L : ℝ) (h1 : S = 3.5) (h2 : L = 3 * S) : S + L = 14 :=
by
  sorry

end sum_of_two_numbers_l8_8899


namespace count_positive_numbers_is_three_l8_8433

def negative_three := -3
def zero := 0
def negative_three_squared := (-3) ^ 2
def absolute_negative_nine := |(-9)|
def negative_one_raised_to_four := -1 ^ 4

def number_list : List Int := [ -negative_three, zero, negative_three_squared, absolute_negative_nine, negative_one_raised_to_four ]

def count_positive_numbers (lst: List Int) : Nat :=
  lst.foldl (λ acc x => if x > 0 then acc + 1 else acc) 0

theorem count_positive_numbers_is_three : count_positive_numbers number_list = 3 :=
by
  -- The proof will go here.
  sorry

end count_positive_numbers_is_three_l8_8433


namespace find_c_l8_8105

-- Define that \( r \) and \( s \) are roots of \( 2x^2 - 4x - 5 \)
variables (r s : ℚ)
-- Condition: sum of roots \( r + s = 2 \)
axiom sum_of_roots : r + s = 2
-- Condition: product of roots \( rs = -5/2 \)
axiom product_of_roots : r * s = -5 / 2

-- Definition of \( c \) based on the roots \( r-3 \) and \( s-3 \)
def c : ℚ := (r - 3) * (s - 3)

-- The theorem to be proved
theorem find_c : c = 1 / 2 :=
by
  sorry

end find_c_l8_8105


namespace angle_terminal_side_equiv_l8_8687

theorem angle_terminal_side_equiv (k : ℤ) : 
  ∀ θ α : ℝ, θ = - (π / 3) → α = 5 * π / 3 → α = θ + 2 * k * π := by
  intro θ α hθ hα
  sorry

end angle_terminal_side_equiv_l8_8687


namespace smallest_rel_prime_210_l8_8563

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l8_8563


namespace milk_leftover_after_milkshakes_l8_8475

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l8_8475


namespace triangle_side_AC_l8_8276

theorem triangle_side_AC 
  (AB BC : ℝ)
  (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13)
  (h2 : BC = 3)
  (h3 : angle_C = Real.pi / 3) :
  ∃ AC : ℝ, AC = 4 :=
by 
  sorry

end triangle_side_AC_l8_8276


namespace golden_ratio_eqn_value_of_ab_value_of_pq_n_l8_8149

-- Part (1): Finding the golden ratio
theorem golden_ratio_eqn {x : ℝ} (h1 : x^2 + x - 1 = 0) : x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

-- Part (2): Finding the value of ab
theorem value_of_ab {a b m : ℝ} (h1 : a^2 + m * a = 1) (h2 : b^2 - 2 * m * b = 4) (h3 : b ≠ -2 * a) : a * b = 2 :=
sorry

-- Part (3): Finding the value of pq - n
theorem value_of_pq_n {p q n : ℝ} (h1 : p ≠ q) (eq1 : p^2 + n * p - 1 = q) (eq2 : q^2 + n * q - 1 = p) : p * q - n = 0 :=
sorry

end golden_ratio_eqn_value_of_ab_value_of_pq_n_l8_8149


namespace marks_deducted_per_wrong_answer_l8_8009

theorem marks_deducted_per_wrong_answer
  (correct_awarded : ℕ)
  (total_marks : ℕ)
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (final_marks : ℕ) :
  correct_awarded = 3 →
  total_marks = 38 →
  total_questions = 70 →
  correct_answers = 27 →
  incorrect_answers = total_questions - correct_answers →
  final_marks = total_marks →
  final_marks = correct_answers * correct_awarded - incorrect_answers * 1 →
  1 = 1
  := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_deducted_per_wrong_answer_l8_8009


namespace samantha_birth_year_l8_8618

theorem samantha_birth_year 
  (first_amc8 : ℕ)
  (amc8_annual : ∀ n : ℕ, n ≥ first_amc8)
  (seventh_amc8 : ℕ)
  (samantha_age : ℕ)
  (samantha_birth_year : ℕ)
  (move_year : ℕ)
  (h1 : first_amc8 = 1983)
  (h2 : seventh_amc8 = first_amc8 + 6)
  (h3 : seventh_amc8 = 1989)
  (h4 : samantha_age = 14)
  (h5 : samantha_birth_year = seventh_amc8 - samantha_age)
  (h6 : move_year = seventh_amc8 - 3) :
  samantha_birth_year = 1975 :=
sorry

end samantha_birth_year_l8_8618


namespace right_triangle_third_side_l8_8658

theorem right_triangle_third_side (x : ℝ) : 
  (∃ (a b c : ℝ), (a = 3 ∧ b = 4 ∧ (a^2 + b^2 = c^2 ∧ (c = x ∨ x^2 + a^2 = b^2)))) → (x = 5 ∨ x = Real.sqrt 7) :=
by 
  sorry

end right_triangle_third_side_l8_8658


namespace problem_statement_l8_8987

theorem problem_statement (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end problem_statement_l8_8987


namespace percent_change_range_l8_8369

-- Define initial conditions
def initial_yes_percent : ℝ := 0.60
def initial_no_percent : ℝ := 0.40
def final_yes_percent : ℝ := 0.80
def final_no_percent : ℝ := 0.20

-- Define the key statement to prove
theorem percent_change_range : 
  ∃ y_min y_max : ℝ, 
  y_min = 0.20 ∧ 
  y_max = 0.60 ∧ 
  (y_max - y_min = 0.40) :=
sorry

end percent_change_range_l8_8369


namespace speed_of_first_train_l8_8533

-- Define the conditions
def distance_pq := 110 -- km
def speed_q := 25 -- km/h
def meet_time := 10 -- hours from midnight
def start_p := 7 -- hours from midnight
def start_q := 8 -- hours from midnight

-- Define the total travel time for each train
def travel_time_p := meet_time - start_p -- hours
def travel_time_q := meet_time - start_q -- hours

-- Define the distance covered by each train
def distance_covered_p (V_p : ℕ) : ℕ := V_p * travel_time_p
def distance_covered_q := speed_q * travel_time_q

-- Theorem to prove the speed of the first train
theorem speed_of_first_train (V_p : ℕ) : distance_covered_p V_p + distance_covered_q = distance_pq → V_p = 20 :=
sorry

end speed_of_first_train_l8_8533


namespace find_a_of_parabola_l8_8363

theorem find_a_of_parabola (a b c : ℤ) (h_vertex : (2, 5) = (2, 5)) (h_point : 8 = a * (3 - 2) ^ 2 + 5) :
  a = 3 :=
sorry

end find_a_of_parabola_l8_8363


namespace quotient_of_f_div_g_l8_8283

-- Define the polynomial f(x) = x^5 + 5
def f (x : ℝ) : ℝ := x ^ 5 + 5

-- Define the divisor polynomial g(x) = x - 1
def g (x : ℝ) : ℝ := x - 1

-- Define the expected quotient polynomial q(x) = x^4 + x^3 + x^2 + x + 1
def q (x : ℝ) : ℝ := x ^ 4 + x ^ 3 + x ^ 2 + x + 1

-- State and prove the main theorem
theorem quotient_of_f_div_g (x : ℝ) :
  ∃ r : ℝ, f x = g x * (q x) + r :=
by
  sorry

end quotient_of_f_div_g_l8_8283


namespace find_a_b_solve_inequality_l8_8565

-- Definitions for the given conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def sol_set1 (x : ℝ) (b : ℝ) : Prop := x < 1 ∨ x > b
def root_eq (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 = 0

-- The final Lean statements for the proofs
theorem find_a_b (a b : ℝ) : (∀ x, (inequality1 a x) ↔ (sol_set1 x b)) → a = 1 ∧ b = 2 :=
sorry

theorem solve_inequality (c : ℝ) : 
  (∀ x, (root_eq 1 x) ↔ (x = 1 ∨ x = 2)) → 
  (c > 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (2 < x ∧ x < c)) ∧
  (c < 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (c < x ∧ x < 2)) ∧
  (c = 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ false) :=
sorry

end find_a_b_solve_inequality_l8_8565


namespace domain_of_h_l8_8935

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_of_h :
  {x : ℝ | 2 * x - 10 ≠ 0} = {x : ℝ | x ≠ 5} :=
by
  sorry

end domain_of_h_l8_8935


namespace remainder_division_l8_8379

theorem remainder_division (n : ℕ) :
  n = 2345678901 →
  n % 102 = 65 :=
by sorry

end remainder_division_l8_8379


namespace value_of_M_correct_l8_8681

noncomputable def value_of_M : ℤ :=
  let d1 := 4        -- First column difference
  let d2 := -7       -- Row difference
  let d3 := 1        -- Second column difference
  let a1 := 25       -- First number in the row
  let a2 := 16 - d1  -- First number in the first column
  let a3 := a1 - d2 * 6  -- Last number in the row
  a3 + d3

theorem value_of_M_correct : value_of_M = -16 :=
  by
    let d1 := 4       -- First column difference
    let d2 := -7      -- Row difference
    let d3 := 1       -- Second column difference
    let a1 := 25      -- First number in the row
    let a2 := 16 - d1 -- First number in the first column
    let a3 := a1 - d2 * 6 -- Last number in the row
    have : a3 + d3 = -16
    · sorry
    exact this

end value_of_M_correct_l8_8681


namespace utility_bills_total_correct_l8_8367

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l8_8367


namespace unique_paintings_count_l8_8515

-- Given the conditions of the problem:
-- - N = 6 disks
-- - 3 disks are blue
-- - 2 disks are red
-- - 1 disk is green
-- - Two paintings that can be obtained from one another by a rotation or a reflection are considered the same

-- Define a theorem to calculate the number of unique paintings.
theorem unique_paintings_count : 
    ∃ n : ℕ, n = 13 :=
sorry

end unique_paintings_count_l8_8515


namespace tonya_large_lemonade_sales_l8_8413

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l8_8413


namespace largest_corner_sum_l8_8248

noncomputable def sum_faces (cube : ℕ → ℕ) : Prop :=
  cube 1 + cube 7 = 8 ∧ 
  cube 2 + cube 6 = 8 ∧ 
  cube 3 + cube 5 = 8 ∧ 
  cube 4 + cube 4 = 8

theorem largest_corner_sum (cube : ℕ → ℕ) 
  (h : sum_faces cube) : 
  ∃ n, n = 17 ∧ 
  ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
            (cube a = 7 ∧ cube b = 6 ∧ cube c = 4 ∨ 
             cube a = 6 ∧ cube b = 4 ∧ cube c = 7 ∨ 
             cube a = 4 ∧ cube b = 7 ∧ cube c = 6)) → 
            a + b + c = n := sorry

end largest_corner_sum_l8_8248


namespace proof_problem_l8_8217

noncomputable def initialEfficiencyOfOneMan : ℕ := sorry
noncomputable def initialEfficiencyOfOneWoman : ℕ := sorry
noncomputable def totalWork : ℕ := sorry

-- Condition (1): 10 men and 15 women together can complete the work in 6 days.
def condition1 := 10 * initialEfficiencyOfOneMan + 15 * initialEfficiencyOfOneWoman = totalWork / 6

-- Condition (2): The efficiency of men to complete the work decreases by 5% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (3): The efficiency of women to complete the work increases by 3% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (4): It takes 100 days for one man alone to complete the same work at his initial efficiency.
def condition4 := initialEfficiencyOfOneMan = totalWork / 100

-- Define the days required for one woman alone to complete the work at her initial efficiency.
noncomputable def daysForWomanToCompleteWork : ℕ := 225

-- Mathematically equivalent proof problem
theorem proof_problem : 
  condition1 ∧ condition4 → (totalWork / daysForWomanToCompleteWork = initialEfficiencyOfOneWoman) :=
by
  sorry

end proof_problem_l8_8217


namespace average_speed_of_trip_l8_8290

theorem average_speed_of_trip 
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_distance : ℝ)
  (second_leg_speed : ℝ)
  (h_dist : total_distance = 50)
  (h_first_leg : first_leg_distance = 25)
  (h_second_leg : second_leg_distance = 25)
  (h_first_speed : first_leg_speed = 60)
  (h_second_speed : second_leg_speed = 30) :
  (total_distance / 
   ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)) = 40) :=
by
  sorry

end average_speed_of_trip_l8_8290


namespace chord_division_ratio_l8_8487

theorem chord_division_ratio (R AB PO DP PC x AP PB : ℝ)
  (hR : R = 11)
  (hAB : AB = 18)
  (hPO : PO = 7)
  (hDP : DP = R - PO)
  (hPC : PC = R + PO)
  (hPower : AP * PB = DP * PC)
  (hChord : AP + PB = AB) :
  AP = 12 ∧ PB = 6 ∨ AP = 6 ∧ PB = 12 :=
by
  -- Structure of the theorem is provided.
  -- Proof steps are skipped and marked with sorry.
  sorry

end chord_division_ratio_l8_8487


namespace tangent_line_at_zero_l8_8187

noncomputable def curve (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, (curve x) = m * x + b) ∧
    m = 2 ∧ b = 1 :=
by 
  sorry

end tangent_line_at_zero_l8_8187


namespace mingyu_change_l8_8204

theorem mingyu_change :
  let eraser_cost := 350
  let pencil_cost := 180
  let erasers_count := 3
  let pencils_count := 2
  let payment := 2000
  let total_eraser_cost := erasers_count * eraser_cost
  let total_pencil_cost := pencils_count * pencil_cost
  let total_cost := total_eraser_cost + total_pencil_cost
  let change := payment - total_cost
  change = 590 := 
by
  -- The proof will go here
  sorry

end mingyu_change_l8_8204


namespace quadratic_has_real_roots_find_value_of_m_l8_8334

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l8_8334


namespace ratio_of_chords_l8_8711

theorem ratio_of_chords 
  (E F G H Q : Type)
  (EQ GQ FQ HQ : ℝ)
  (h1 : EQ = 4)
  (h2 : GQ = 10)
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 5 / 2 := 
by 
  sorry

end ratio_of_chords_l8_8711


namespace problem_equivalent_l8_8602

theorem problem_equivalent
  (x : ℚ)
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289 / 8 := 
by
  sorry

end problem_equivalent_l8_8602


namespace min_value_of_a_l8_8403

/-- Given the inequality |x - 1| + |x + a| ≤ 8, prove that the minimum value of a is -9 -/

theorem min_value_of_a (a : ℝ) (h : ∀ x : ℝ, |x - 1| + |x + a| ≤ 8) : a = -9 :=
sorry

end min_value_of_a_l8_8403


namespace sequence_sum_l8_8294

theorem sequence_sum:
  ∀ (y : ℕ → ℕ), 
  (y 1 = 100) → 
  (∀ k ≥ 2, y k = y (k - 1) ^ 2 + 2 * y (k - 1) + 1) →
  ( ∑' n, 1 / (y n + 1) = 1 / 101 ) :=
by
  sorry

end sequence_sum_l8_8294


namespace find_a5_l8_8129

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, ∃ q : ℝ, a (n + m) = a n * q ^ m

theorem find_a5
  (h : geometric_sequence a)
  (h3 : a 3 = 2)
  (h7 : a 7 = 8) :
  a 5 = 4 :=
sorry

end find_a5_l8_8129


namespace batsman_average_increase_l8_8951

theorem batsman_average_increase
  (A : ℕ)  -- Assume the initial average is a non-negative integer
  (h1 : 11 * A + 70 = 12 * (A + 3))  -- Condition derived from the problem
  : A + 3 = 37 := 
by {
  -- The actual proof would go here, but is replaced by sorry to skip the proof
  sorry
}

end batsman_average_increase_l8_8951


namespace c_is_perfect_square_l8_8995

theorem c_is_perfect_square (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : c = a + b / a - 1 / b) : ∃ m : ℕ, c = m * m :=
by
  sorry

end c_is_perfect_square_l8_8995


namespace function_increasing_interval_l8_8906

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem function_increasing_interval :
  ∀ x : ℝ, x > 0 → deriv f x > 0 := 
sorry

end function_increasing_interval_l8_8906


namespace Y_pdf_from_X_pdf_l8_8608

/-- Given random variable X with PDF p(x), prove PDF of Y = X^3 -/
noncomputable def X_pdf (σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2))

noncomputable def Y_pdf (σ : ℝ) (y : ℝ) : ℝ :=
  (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2))

theorem Y_pdf_from_X_pdf (σ : ℝ) (y : ℝ) :
  ∀ x : ℝ, X_pdf σ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2)) →
  Y_pdf σ y = (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2)) :=
sorry

end Y_pdf_from_X_pdf_l8_8608


namespace woman_complete_time_l8_8430

-- Define the work rate of one man
def man_rate := 1 / 100

-- Define the combined work rate equation for 10 men and 15 women completing work in 5 days
def combined_work_rate (W : ℝ) : Prop :=
  10 * man_rate + 15 * W = 1 / 5

-- Prove that given the combined work rate equation, one woman alone takes 150 days to complete the work
theorem woman_complete_time (W : ℝ) : combined_work_rate W → W = 1 / 150 :=
by
  intro h
  have h1 : 10 * man_rate + 15 * W = 1 / 5 := h
  rw [man_rate] at h1
  sorry -- Proof steps would go here

end woman_complete_time_l8_8430


namespace abscissa_of_point_P_l8_8568

open Real

noncomputable def hyperbola_abscissa (x y : ℝ) : Prop :=
  (x^2 - y^2 = 4) ∧
  (x > 0) ∧
  ((x + 2 * sqrt 2) * (x - 2 * sqrt 2) = -y^2)

theorem abscissa_of_point_P :
  ∃ (x y : ℝ), hyperbola_abscissa x y ∧ x = sqrt 6 := by
  sorry

end abscissa_of_point_P_l8_8568


namespace test_question_count_l8_8490

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end test_question_count_l8_8490


namespace min_square_distance_l8_8980

theorem min_square_distance (x y z w : ℝ) (h1 : x * y = 4) (h2 : z^2 + 4 * w^2 = 4) : (x - z)^2 + (y - w)^2 ≥ 1.6 :=
sorry

end min_square_distance_l8_8980


namespace carnations_count_l8_8251

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l8_8251


namespace problem1_problem2_l8_8627

open Set Real

-- Definition of sets A, B, and C
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

-- Problem 1: Prove A ∪ B = { x | 1 ≤ x < 10 }
theorem problem1 : A ∪ B = { x : ℝ | 1 ≤ x ∧ x < 10 } :=
sorry

-- Problem 2: Prove the range of a given the conditions
theorem problem2 (a : ℝ) (h1 : (A ∩ C a) ≠ ∅) (h2 : (B ∩ C a) = ∅) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l8_8627


namespace no_n_gt_1_divisibility_l8_8557

theorem no_n_gt_1_divisibility (n : ℕ) (h : n > 1) : ¬ (3 ^ (n - 1) + 5 ^ (n - 1)) ∣ (3 ^ n + 5 ^ n) :=
by
  sorry

end no_n_gt_1_divisibility_l8_8557


namespace tangent_parallel_to_line_l8_8465

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) : 
  (y = 4 * x - 1) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by
  sorry

end tangent_parallel_to_line_l8_8465


namespace slope_of_line_l8_8355

def point1 : (ℤ × ℤ) := (-4, 6)
def point2 : (ℤ × ℤ) := (3, -4)

def slope_formula (p1 p2 : (ℤ × ℤ)) : ℚ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst : ℚ)

theorem slope_of_line : slope_formula point1 point2 = -10 / 7 := by
  sorry

end slope_of_line_l8_8355


namespace simplify_expression_l8_8641

theorem simplify_expression (x y : ℝ) (h_x_ne_0 : x ≠ 0) (h_y_ne_0 : y ≠ 0) :
  (25*x^3*y) * (8*x*y) * (1 / (5*x*y^2)^2) = 8*x^2 / y^2 :=
by
  sorry

end simplify_expression_l8_8641


namespace larger_angle_measure_l8_8340

theorem larger_angle_measure (x : ℝ) (h : 4 * x + 5 * x = 180) : 5 * x = 100 :=
by
  sorry

end larger_angle_measure_l8_8340


namespace Tim_gave_kittens_to_Jessica_l8_8525

def Tim_original_kittens : ℕ := 6
def kittens_given_to_Jessica := 3
def kittens_given_by_Sara : ℕ := 9 
def Tim_final_kittens : ℕ := 12

theorem Tim_gave_kittens_to_Jessica :
  (Tim_original_kittens + kittens_given_by_Sara - kittens_given_to_Jessica = Tim_final_kittens) :=
by sorry

end Tim_gave_kittens_to_Jessica_l8_8525


namespace turtles_remaining_on_log_l8_8089

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l8_8089


namespace tan_alpha_value_l8_8111

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -1 / 2) (h2 : 0 < α ∧ α < π) : tan α = -1 / 3 :=
sorry

end tan_alpha_value_l8_8111


namespace circle_radius_l8_8857

theorem circle_radius (r : ℝ) (π : ℝ) (h1 : π > 0) (h2 : ∀ x, π * x^2 = 100*π → x = 10) : r = 10 :=
by
  have : π * r^2 = 100*π → r = 10 := h2 r
  exact sorry

end circle_radius_l8_8857


namespace bears_in_shipment_l8_8043

theorem bears_in_shipment (initial_bears shipped_bears bears_per_shelf shelves_used : ℕ) 
  (h1 : initial_bears = 4) 
  (h2 : bears_per_shelf = 7) 
  (h3 : shelves_used = 2) 
  (total_bears_on_shelves : ℕ) 
  (h4 : total_bears_on_shelves = shelves_used * bears_per_shelf) 
  (total_bears_after_shipment : ℕ) 
  (h5 : total_bears_after_shipment = total_bears_on_shelves) 
  : shipped_bears = total_bears_on_shelves - initial_bears := 
sorry

end bears_in_shipment_l8_8043


namespace find_n_l8_8767

theorem find_n (a n : ℕ) 
  (h1 : a^2 % n = 8) 
  (h2 : a^3 % n = 25) 
  (h3 : n > 25) : 
  n = 113 := 
sorry

end find_n_l8_8767


namespace common_ratio_of_sequence_l8_8498

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l8_8498


namespace claire_has_gerbils_l8_8784

-- Definitions based on conditions
variables (G H : ℕ)
variables (h1 : G + H = 90) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25)

-- Main statement to prove
theorem claire_has_gerbils : G = 60 :=
sorry

end claire_has_gerbils_l8_8784


namespace representation_of_2015_l8_8216

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l8_8216


namespace division_result_l8_8976

theorem division_result : 210 / (15 + 12 * 3 - 6) = 210 / 45 :=
by
  sorry

end division_result_l8_8976


namespace average_reading_days_l8_8393

def emery_days : ℕ := 20
def serena_days : ℕ := 5 * emery_days
def average_days (e s : ℕ) : ℕ := (e + s) / 2

theorem average_reading_days 
  (e s : ℕ) 
  (h1 : e = emery_days)
  (h2 : s = serena_days) :
  average_days e s = 60 :=
by
  rw [h1, h2, emery_days, serena_days]
  sorry

end average_reading_days_l8_8393


namespace solve_equation_l8_8331

theorem solve_equation :
  ∀ x : ℝ, 
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     16 / (Real.sqrt (x - 10) + 10) = 0)
    ↔ 
    x = 1841 / 121 ∨ x = 190 / 9 :=
by
  sorry

end solve_equation_l8_8331


namespace compute_expression_value_l8_8242

noncomputable def expression := 3 ^ (Real.log 4 / Real.log 3) - 27 ^ (2 / 3) - Real.log 0.01 / Real.log 10 + Real.log (Real.exp 3)

theorem compute_expression_value :
  expression = 0 := 
by
  sorry

end compute_expression_value_l8_8242


namespace largest_reciprocal_l8_8707

-- Definitions for the given numbers
def a := 1/4
def b := 3/7
def c := 2
def d := 10
def e := 2023

-- Statement to prove the problem
theorem largest_reciprocal :
  (1/a) > (1/b) ∧ (1/a) > (1/c) ∧ (1/a) > (1/d) ∧ (1/a) > (1/e) :=
by
  sorry

end largest_reciprocal_l8_8707


namespace intersecting_lines_l8_8110

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l8_8110


namespace student_test_ratio_l8_8850

theorem student_test_ratio :
  ∀ (total_questions correct_responses : ℕ),
  total_questions = 100 →
  correct_responses = 93 →
  (total_questions - correct_responses) / correct_responses = 7 / 93 :=
by
  intros total_questions correct_responses h_total_questions h_correct_responses
  sorry

end student_test_ratio_l8_8850


namespace coprime_n_minus_2_n_squared_minus_n_minus_1_l8_8613

theorem coprime_n_minus_2_n_squared_minus_n_minus_1 (n : ℕ) : n - 2 ∣ n^2 - n - 1 → False :=
by
-- proof omitted as per instructions
sorry

end coprime_n_minus_2_n_squared_minus_n_minus_1_l8_8613


namespace min_value_a_b_inv_a_inv_b_l8_8112

theorem min_value_a_b_inv_a_inv_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 :=
sorry

end min_value_a_b_inv_a_inv_b_l8_8112


namespace correct_result_is_102357_l8_8218

-- Defining the conditions
def number (f : ℕ) : Prop := f * 153 = 102357

-- Stating the proof problem
theorem correct_result_is_102357 (f : ℕ) (h : f * 153 = 102325) (wrong_digits : ℕ) :
  (number f) :=
by
  sorry

end correct_result_is_102357_l8_8218


namespace binary_ternary_product_base_10_l8_8039

theorem binary_ternary_product_base_10 :
  let b2 := 2
  let t3 := 3
  let n1 := 1011 -- binary representation
  let n2 := 122 -- ternary representation
  let a1 := (1 * b2^3) + (0 * b2^2) + (1 * b2^1) + (1 * b2^0)
  let a2 := (1 * t3^2) + (2 * t3^1) + (2 * t3^0)
  a1 * a2 = 187 :=
by
  sorry

end binary_ternary_product_base_10_l8_8039


namespace base7_number_l8_8529

theorem base7_number (A B C : ℕ) (h1 : 1 ≤ A ∧ A ≤ 6) (h2 : 1 ≤ B ∧ B ≤ 6) (h3 : 1 ≤ C ∧ C ≤ 6)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_condition1 : B + C = 7)
  (h_condition2 : A + 1 = C)
  (h_condition3 : A + B = C) :
  A = 5 ∧ B = 1 ∧ C = 6 :=
sorry

end base7_number_l8_8529


namespace soaps_in_one_package_l8_8113

theorem soaps_in_one_package (boxes : ℕ) (packages_per_box : ℕ) (total_packages : ℕ) (total_soaps : ℕ) : 
  boxes = 2 → packages_per_box = 6 → total_packages = boxes * packages_per_box → total_soaps = 2304 → (total_soaps / total_packages) = 192 :=
by
  intros h_boxes h_packages_per_box h_total_packages h_total_soaps
  sorry

end soaps_in_one_package_l8_8113


namespace linear_function_common_quadrants_l8_8573

theorem linear_function_common_quadrants {k b : ℝ} (h : k * b < 0) :
  (exists (q1 q2 : ℕ), q1 = 1 ∧ q2 = 4) := 
sorry

end linear_function_common_quadrants_l8_8573


namespace rotated_intersection_point_l8_8653

theorem rotated_intersection_point (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ P : ℝ × ℝ, P = (-Real.sin θ, Real.cos θ) ∧ 
    ∃ φ : ℝ, φ = θ + π / 2 ∧ 
      P = (Real.cos φ, Real.sin φ) := 
by
  sorry

end rotated_intersection_point_l8_8653


namespace problem_l8_8061

def seq (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem problem (a : ℕ → ℤ) (h₁ : a 1 = 2010) (h₂ : a 2 = 2011) (h₃ : seq a) : a 1000 = 2343 :=
sorry

end problem_l8_8061


namespace triple_hash_72_eq_7_25_l8_8167

def hash (N : ℝ) : ℝ := 0.5 * N - 1

theorem triple_hash_72_eq_7_25 : hash (hash (hash 72)) = 7.25 :=
by
  sorry

end triple_hash_72_eq_7_25_l8_8167


namespace root_of_equation_l8_8671

theorem root_of_equation : 
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2) → (x = -4.5) :=
by sorry

end root_of_equation_l8_8671


namespace samantha_mean_correct_l8_8816

-- Given data: Samantha's assignment scores
def samantha_scores : List ℕ := [84, 89, 92, 88, 95, 91, 93]

-- Definition of the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

-- Prove that the arithmetic mean of Samantha's scores is 90.29
theorem samantha_mean_correct :
  arithmetic_mean samantha_scores = 90.29 := 
by
  -- The proof steps would be filled in here
  sorry

end samantha_mean_correct_l8_8816


namespace num_correct_statements_l8_8045

def doubleAbsDiff (a b c d : ℝ) : ℝ :=
  |a - b| - |c - d|

theorem num_correct_statements : 
  (∀ a b c d : ℝ, (a, b, c, d) = (24, 25, 29, 30) → 
    (doubleAbsDiff a b c d = 0) ∨
    (doubleAbsDiff a c b d = 0) ∨
    (doubleAbsDiff a d b c = -0.5) ∨
    (doubleAbsDiff b c a d = 0.5)) → 
  (∀ x : ℝ, x ≥ 2 → 
    doubleAbsDiff (x^2) (2*x) 1 1 = 7 → 
    (x^4 + 2401 / x^4 = 226)) →
  (∀ x : ℝ, x ≥ -2 → 
    (doubleAbsDiff (2*x-5) (3*x-2) (4*x-1) (5*x+3)) ≠ 0) →
  (0 = 0)
:= by
  sorry

end num_correct_statements_l8_8045


namespace alton_weekly_profit_l8_8183

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l8_8183


namespace x_plus_q_in_terms_of_q_l8_8392

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2 * q + 5 :=
by
  sorry

end x_plus_q_in_terms_of_q_l8_8392


namespace product_of_roots_eq_negative_forty_nine_l8_8904

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l8_8904


namespace union_of_sets_l8_8804

def setA : Set ℝ := { x : ℝ | (x - 2) / (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -2 * x^2 + 7 * x + 4 > 0 }
def unionAB : Set ℝ := { x : ℝ | -1 < x ∧ x < 4 }

theorem union_of_sets :
  ∀ x : ℝ, x ∈ setA ∨ x ∈ setB ↔ x ∈ unionAB :=
by sorry

end union_of_sets_l8_8804


namespace domain_of_fn_l8_8257

noncomputable def domain_fn (x : ℝ) : ℝ := (Real.sqrt (3 * x + 4)) / x

theorem domain_of_fn :
  { x : ℝ | x ≥ -4 / 3 ∧ x ≠ 0 } =
  { x : ℝ | 3 * x + 4 ≥ 0 ∧ x ≠ 0 } :=
by
  ext x
  simp
  exact sorry

end domain_of_fn_l8_8257


namespace geometric_sequence_seventh_term_l8_8954

noncomputable def a_7 (a₁ q : ℝ) : ℝ :=
  a₁ * q^6

theorem geometric_sequence_seventh_term :
  a_7 3 (Real.sqrt 2) = 24 :=
by
  sorry

end geometric_sequence_seventh_term_l8_8954


namespace andrew_age_proof_l8_8588

def andrew_age_problem : Prop :=
  ∃ (a g : ℚ), g = 15 * a ∧ g - a = 60 ∧ a = 30 / 7

theorem andrew_age_proof : andrew_age_problem :=
by
  sorry

end andrew_age_proof_l8_8588


namespace find_y_l8_8280

theorem find_y (y : ℝ) (h : 2 * y / 3 = 30) : y = 45 :=
by
  sorry

end find_y_l8_8280


namespace evaluate_ceiling_neg_cubed_frac_l8_8161

theorem evaluate_ceiling_neg_cubed_frac :
  (Int.ceil ((- (5 : ℚ) / 3) ^ 3 + 1) = -3) :=
sorry

end evaluate_ceiling_neg_cubed_frac_l8_8161


namespace probability_C_l8_8719

variable (pA pB pD pC : ℚ)
variable (hA : pA = 1 / 4)
variable (hB : pB = 1 / 3)
variable (hD : pD = 1 / 6)
variable (total_prob : pA + pB + pD + pC = 1)

theorem probability_C (hA : pA = 1 / 4) (hB : pB = 1 / 3) (hD : pD = 1 / 6) (total_prob : pA + pB + pD + pC = 1) : pC = 1 / 4 :=
sorry

end probability_C_l8_8719


namespace fishes_per_body_of_water_l8_8067

-- Define the number of bodies of water
def n_b : Nat := 6

-- Define the total number of fishes
def n_f : Nat := 1050

-- Prove the number of fishes per body of water
theorem fishes_per_body_of_water : n_f / n_b = 175 := by 
  sorry

end fishes_per_body_of_water_l8_8067


namespace exists_valid_configuration_l8_8424

-- Define the nine circles
def circles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the connections (adjacency list) where each connected pair must sum to 23
def lines : List (ℕ × ℕ) := [(1, 8), (8, 6), (8, 9), (9, 2), (2, 7), (7, 6), (7, 4), (4, 1), (4, 5), (5, 6), (5, 3), (6, 3)]

-- The main theorem that we need to prove: there exists a permutation of circles satisfying the line sum condition
theorem exists_valid_configuration: 
  ∃ (f : ℕ → ℕ), 
    (∀ x ∈ circles, f x ∈ circles) ∧ 
    (∀ (a b : ℕ), (a, b) ∈ lines → f a + f b = 23) :=
sorry

end exists_valid_configuration_l8_8424


namespace range_of_m_l8_8136

-- Define the function g as an even function on the interval [-2, 2] 
-- and monotonically decreasing on [0, 2]

variable {g : ℝ → ℝ}

axiom even_g : ∀ x, g x = g (-x)
axiom mono_dec_g : ∀ {x y}, 0 ≤ x → x ≤ y → g y ≤ g x
axiom domain_g : ∀ x, -2 ≤ x ∧ x ≤ 2

theorem range_of_m (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) (h : g (1 - m) < g m) : -1 ≤ m ∧ m < 1 / 2 :=
sorry

end range_of_m_l8_8136


namespace simplify_trig_expression_trig_identity_l8_8480

-- Defining the necessary functions
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

-- First problem
theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * Real.pi - α) * sin (Real.pi + α) * cos (-Real.pi - α)) / (sin (3 * Real.pi - α) * cos (Real.pi - α)) = sin α :=
sorry

-- Second problem
theorem trig_identity (x : ℝ) (hx : cos x ≠ 0) (hx' : 1 - sin x ≠ 0) :
  (cos x / (1 - sin x)) = ((1 + sin x) / cos x) :=
sorry

end simplify_trig_expression_trig_identity_l8_8480


namespace min_weighings_to_identify_fake_l8_8530

def piles := 1000000
def coins_per_pile := 1996
def weight_real_coin := 10
def weight_fake_coin := 9
def expected_total_weight : Nat :=
  (piles * (piles + 1) / 2) * weight_real_coin

theorem min_weighings_to_identify_fake :
  (∃ k : ℕ, k < piles ∧ 
  ∀ (W : ℕ), W = expected_total_weight - k → k = expected_total_weight - W) →
  true := 
by
  sorry

end min_weighings_to_identify_fake_l8_8530


namespace probability_red_balls_by_4th_draw_l8_8233

theorem probability_red_balls_by_4th_draw :
  let total_balls := 10
  let red_prob := 2 / total_balls
  let white_prob := 1 - red_prob
  (white_prob^3) * red_prob = 0.0434 := sorry

end probability_red_balls_by_4th_draw_l8_8233


namespace ratio_adult_women_to_men_event_l8_8753

theorem ratio_adult_women_to_men_event :
  ∀ (total_members men_ratio women_ratio children : ℕ), 
  total_members = 2000 →
  men_ratio = 30 →
  children = 200 →
  women_ratio = men_ratio →
  women_ratio / men_ratio = 1 / 1 := 
by
  intros total_members men_ratio women_ratio children
  sorry

end ratio_adult_women_to_men_event_l8_8753


namespace solve_inner_circle_radius_l8_8099

noncomputable def isosceles_trapezoid_radius := 
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let radiusA := 4
  let radiusB := 4
  let radiusC := 3
  let radiusD := 3
  let r := (-72 + 60 * Real.sqrt 3) / 26
  r

theorem solve_inner_circle_radius :
  let k := 72
  let m := 60
  let n := 3
  let p := 26
  gcd k p = 1 → -- explicit gcd calculation between k and p 
  (isosceles_trapezoid_radius = (-k + m * Real.sqrt n) / p) ∧ (k + m + n + p = 161) :=
by
  sorry

end solve_inner_circle_radius_l8_8099


namespace negative_integer_solution_l8_8256

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : N^2 + N = 6) : N = -3 := 
by 
  sorry

end negative_integer_solution_l8_8256


namespace concentration_time_within_bounds_l8_8677

-- Define the time bounds for the highest concentration of the drug in the blood
def highest_concentration_time_lower (base : ℝ) (tolerance : ℝ) : ℝ := base - tolerance
def highest_concentration_time_upper (base : ℝ) (tolerance : ℝ) : ℝ := base + tolerance

-- Define the base and tolerance values
def base_time : ℝ := 0.65
def tolerance_time : ℝ := 0.15

-- Define the specific time we want to prove is within the bounds
def specific_time : ℝ := 0.8

-- Theorem statement
theorem concentration_time_within_bounds : 
  highest_concentration_time_lower base_time tolerance_time ≤ specific_time ∧ 
  specific_time ≤ highest_concentration_time_upper base_time tolerance_time :=
by sorry

end concentration_time_within_bounds_l8_8677


namespace least_n_for_factorial_multiple_10080_l8_8446

theorem least_n_for_factorial_multiple_10080 (n : ℕ) 
  (h₁ : 0 < n) 
  (h₂ : ∀ m, m > 0 → (n ≠ m → n! % 10080 ≠ 0)) 
  : n = 8 := 
sorry

end least_n_for_factorial_multiple_10080_l8_8446


namespace non_degenerate_ellipse_l8_8693

theorem non_degenerate_ellipse (k : ℝ) : (∃ a, a = -21) ↔ (k > -21) := by
  sorry

end non_degenerate_ellipse_l8_8693


namespace A_inter_complement_B_is_empty_l8_8907

open Set Real

noncomputable def U : Set Real := univ

noncomputable def A : Set Real := { x : Real | ∃ (y : Real), y = sqrt (Real.log x) }

noncomputable def B : Set Real := { y : Real | ∃ (x : Real), y = sqrt x }

theorem A_inter_complement_B_is_empty :
  A ∩ (U \ B) = ∅ :=
by
    sorry

end A_inter_complement_B_is_empty_l8_8907


namespace Cubs_home_runs_third_inning_l8_8055

variable (X : ℕ)

theorem Cubs_home_runs_third_inning 
  (h : X + 1 + 2 = 2 + 3) : 
  X = 2 :=
by 
  sorry

end Cubs_home_runs_third_inning_l8_8055


namespace Chris_age_l8_8322

variable (a b c : ℕ)

theorem Chris_age : a + b + c = 36 ∧ b = 2*c + 9 ∧ b = a → c = 4 :=
by
  sorry

end Chris_age_l8_8322


namespace time_taken_by_alex_l8_8405

-- Define the conditions
def distance_per_lap : ℝ := 500 -- distance per lap in meters
def distance_first_part : ℝ := 150 -- first part of the distance in meters
def speed_first_part : ℝ := 3 -- speed for the first part in meters per second
def distance_second_part : ℝ := 350 -- remaining part of the distance in meters
def speed_second_part : ℝ := 4 -- speed for the remaining part in meters per second
def num_laps : ℝ := 4 -- number of laps run by Alex

-- Target time, expressed in seconds
def target_time : ℝ := 550 -- 9 minutes and 10 seconds is 550 seconds

-- Prove that given the conditions, the total time Alex takes to run 4 laps is 550 seconds
theorem time_taken_by_alex :
  (distance_first_part / speed_first_part + distance_second_part / speed_second_part) * num_laps = target_time :=
by
  sorry

end time_taken_by_alex_l8_8405


namespace find_common_difference_l8_8159

-- Definitions based on conditions in a)
def common_difference_4_10 (a₁ d : ℝ) : Prop :=
  (a₁ + 3 * d) + (a₁ + 9 * d) = 0

def sum_relation (a₁ d : ℝ) : Prop :=
  2 * (12 * a₁ + 66 * d) = (2 * a₁ + d + 10)

-- Math proof problem statement
theorem find_common_difference (a₁ d : ℝ) 
  (h₁ : common_difference_4_10 a₁ d) 
  (h₂ : sum_relation a₁ d) : 
  d = -10 :=
sorry

end find_common_difference_l8_8159


namespace total_number_of_coins_is_336_l8_8014

theorem total_number_of_coins_is_336 (N20 : ℕ) (N25 : ℕ) (total_value_rupees : ℚ)
    (h1 : N20 = 260) (h2 : total_value_rupees = 71) (h3 : 20 * N20 + 25 * N25 = 7100) :
    N20 + N25 = 336 :=
by
  sorry

end total_number_of_coins_is_336_l8_8014


namespace find_mark_age_l8_8699

-- Define Mark and Aaron's ages
variables (M A : ℕ)

-- The conditions
def condition1 : Prop := M - 3 = 3 * (A - 3) + 1
def condition2 : Prop := M + 4 = 2 * (A + 4) + 2

-- The proof statement
theorem find_mark_age (h1 : condition1 M A) (h2 : condition2 M A) : M = 28 :=
by sorry

end find_mark_age_l8_8699


namespace number_of_men_in_first_group_l8_8738

-- Definitions based on the conditions provided
def work_done (men : ℕ) (days : ℕ) (work_rate : ℝ) : ℝ :=
  men * days * work_rate

-- Given conditions
def condition1 (M : ℕ) : Prop :=
  ∃ work_rate : ℝ, work_done M 12 work_rate = 66

def condition2 : Prop :=
  ∃ work_rate : ℝ, work_done 86 8 work_rate = 189.2

-- Proof goal
theorem number_of_men_in_first_group : 
  ∀ M : ℕ, condition1 M → condition2 → M = 57 := by
  sorry

end number_of_men_in_first_group_l8_8738


namespace park_area_l8_8210

theorem park_area (P : ℝ) (w l : ℝ) (hP : P = 120) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 675 :=
by
  sorry

end park_area_l8_8210


namespace negation_exists_l8_8534

theorem negation_exists {x : ℝ} (h : ∀ x, x > 0 → x^2 - x ≤ 0) : ∃ x, x > 0 ∧ x^2 - x > 0 :=
sorry

end negation_exists_l8_8534


namespace ants_on_track_l8_8894

/-- Given that ants move on a circular track of length 60 cm at a speed of 1 cm/s
and that there are 48 pairwise collisions in a minute, prove that the possible 
total number of ants on the track is 10, 11, 14, or 25. -/
theorem ants_on_track (x y : ℕ) (h : x * y = 24) : x + y = 10 ∨ x + y = 11 ∨ x + y = 14 ∨ x + y = 25 :=
by sorry

end ants_on_track_l8_8894


namespace sum_of_decimals_as_fraction_l8_8601

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l8_8601


namespace evaluate_expression_at_three_l8_8527

-- Define the evaluation of the expression (x^x)^(x^x) at x=3
theorem evaluate_expression_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_three_l8_8527


namespace f_neg_one_f_monotonic_decreasing_solve_inequality_l8_8397

-- Definitions based on conditions in part a)
variables {f : ℝ → ℝ}
axiom f_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂ - 2
axiom f_one : f 1 = 0
axiom f_neg : ∀ x > 1, f x < 0

-- Proof statement for the value of f(-1)
theorem f_neg_one : f (-1) = 4 := by
  sorry

-- Proof statement for the monotonicity of f(x)
theorem f_monotonic_decreasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Proof statement for the inequality solution
theorem solve_inequality (x : ℝ) :
  ∀ t, t = f (x^2 - 2*x) →
  t^2 + 2*t - 8 < 0 → (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) := by
  sorry

end f_neg_one_f_monotonic_decreasing_solve_inequality_l8_8397


namespace subtraction_of_decimals_l8_8302

theorem subtraction_of_decimals :
  888.8888 - 444.4444 = 444.4444 := 
sorry

end subtraction_of_decimals_l8_8302


namespace total_pieces_ten_row_triangle_l8_8051

-- Definitions based on the conditions
def rods (n : ℕ) : ℕ :=
  (n * (2 * 4 + (n - 1) * 5)) / 2

def connectors (n : ℕ) : ℕ :=
  ((n + 1) * (2 * 1 + n * 1)) / 2

def support_sticks (n : ℕ) : ℕ := 
  if n >= 3 then ((n - 2) * (2 * 2 + (n - 3) * 2)) / 2 else 0

-- The theorem stating the total number of pieces is 395 for a ten-row triangle
theorem total_pieces_ten_row_triangle : rods 10 + connectors 10 + support_sticks 10 = 395 :=
by
  sorry

end total_pieces_ten_row_triangle_l8_8051


namespace correct_statements_l8_8286

-- Definitions for statements A, B, C, and D
def statementA (x : ℝ) : Prop := |x| > 1 → x > 1
def statementB (A B C : ℝ) : Prop := (C > 90) ↔ (A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90))
def statementC (a b : ℝ) : Prop := (a * b ≠ 0) ↔ (a ≠ 0 ∧ b ≠ 0)
def statementD (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Proof problem stating which statements are correct
theorem correct_statements :
  (∀ x : ℝ, statementA x = false) ∧ 
  (∀ (A B C : ℝ), statementB A B C = false) ∧ 
  (∀ (a b : ℝ), statementC a b) ∧ 
  (∀ (a b : ℝ), statementD a b = false) :=
by
  sorry

end correct_statements_l8_8286


namespace find_p_plus_q_l8_8815

/--
In \(\triangle{XYZ}\), \(XY = 12\), \(\angle{X} = 45^\circ\), and \(\angle{Y} = 60^\circ\).
Let \(G, E,\) and \(L\) be points on the line \(YZ\) such that \(XG \perp YZ\), 
\(\angle{XYE} = \angle{EYX}\), and \(YL = LY\). Point \(O\) is the midpoint of 
the segment \(GL\), and point \(Q\) is on ray \(XE\) such that \(QO \perp YZ\).
Prove that \(XQ^2 = \dfrac{81}{2}\) and thus \(p + q = 83\), where \(p\) and \(q\) 
are relatively prime positive integers.
-/
theorem find_p_plus_q :
  ∃ (p q : ℕ), gcd p q = 1 ∧ XQ^2 = 81 / 2 ∧ p + q = 83 :=
sorry

end find_p_plus_q_l8_8815


namespace smallest_k_with_properties_l8_8832

noncomputable def exists_coloring_and_function (k : ℕ) : Prop :=
  ∃ (colors : ℤ → Fin k) (f : ℤ → ℤ),
    (∀ m n : ℤ, colors m = colors n → f (m + n) = f m + f n) ∧
    (∃ m n : ℤ, f (m + n) ≠ f m + f n)

theorem smallest_k_with_properties : ∃ (k : ℕ), k > 0 ∧ exists_coloring_and_function k ∧
                                         (∀ k' : ℕ, k' > 0 ∧ k' < k → ¬ exists_coloring_and_function k') :=
by
  sorry

end smallest_k_with_properties_l8_8832


namespace net_profit_expression_and_break_even_point_l8_8489

-- Definitions based on the conditions in a)
def investment : ℝ := 600000
def initial_expense : ℝ := 80000
def expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function as given in the solution
def net_profit (n : ℕ) : ℝ :=
  - (n : ℝ)^2 + 19 * n - 60

-- Statement about the function and where the dealer starts making profit
theorem net_profit_expression_and_break_even_point :
  net_profit n = - (n : ℝ)^2 + 19 * n - 60 ∧ ∃ n ≥ 5, net_profit n > 0 :=
sorry

end net_profit_expression_and_break_even_point_l8_8489


namespace amount_saved_percentage_l8_8576

variable (S : ℝ) 

-- Condition: Last year, Sandy saved 7% of her annual salary
def amount_saved_last_year (S : ℝ) : ℝ := 0.07 * S

-- Condition: This year, she made 15% more money than last year
def salary_this_year (S : ℝ) : ℝ := 1.15 * S

-- Condition: This year, she saved 10% of her salary
def amount_saved_this_year (S : ℝ) : ℝ := 0.10 * salary_this_year S

-- The statement to prove
theorem amount_saved_percentage (S : ℝ) : 
  amount_saved_this_year S = 1.642857 * amount_saved_last_year S :=
by 
  sorry

end amount_saved_percentage_l8_8576


namespace actual_cost_of_article_l8_8585

theorem actual_cost_of_article {x : ℝ} (h : 0.76 * x = 760) : x = 1000 :=
by
  sorry

end actual_cost_of_article_l8_8585


namespace sqrt_two_minus_one_pow_zero_l8_8778

theorem sqrt_two_minus_one_pow_zero : (Real.sqrt 2 - 1)^0 = 1 := by
  sorry

end sqrt_two_minus_one_pow_zero_l8_8778


namespace sum_of_possible_values_of_x_l8_8420

namespace ProofProblem

-- Assume we are working in degrees for angles
def is_scalene_triangle (A B C : ℝ) (a b c : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

def triangle_angle_sum (A B C : ℝ) : Prop :=
  A + B + C = 180

noncomputable def problem_statement (x : ℝ) (A B C : ℝ) (a b c : ℝ) : Prop :=
  is_scalene_triangle A B C a b c ∧
  B = 45 ∧
  (A = x ∨ C = x) ∧
  (a = b ∨ b = c ∨ c = a) ∧
  triangle_angle_sum A B C

theorem sum_of_possible_values_of_x (x : ℝ) (A B C : ℝ) (a b c : ℝ) :
  problem_statement x A B C a b c →
  x = 45 :=
sorry

end ProofProblem

end sum_of_possible_values_of_x_l8_8420


namespace geometric_sequence_304th_term_l8_8263

theorem geometric_sequence_304th_term (a r : ℤ) (n : ℕ) (h_a : a = 8) (h_ar : a * r = -8) (h_n : n = 304) :
  ∃ t : ℤ, t = -8 :=
by
  sorry

end geometric_sequence_304th_term_l8_8263


namespace max_sqrt_expr_l8_8733

variable {x y z : ℝ}

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)

theorem max_sqrt_expr (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  f x y z ≤ 2 * Real.sqrt 3 := by
  sorry

end max_sqrt_expr_l8_8733


namespace max_value_9_l8_8456

noncomputable def max_ab_ac_bc (a b c : ℝ) : ℝ :=
  max (a * b) (max (a * c) (b * c))

theorem max_value_9 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 27) :
  max_ab_ac_bc a b c = 9 :=
sorry

end max_value_9_l8_8456


namespace find_sum_of_squares_l8_8270

theorem find_sum_of_squares (x y : ℝ) (h1: x * y = 16) (h2: x^2 + y^2 = 34) : (x + y) ^ 2 = 66 :=
by sorry

end find_sum_of_squares_l8_8270


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l8_8094

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l8_8094


namespace find_second_number_l8_8192

theorem find_second_number (x : ℝ) (h : (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8) : x = 40 :=
sorry

end find_second_number_l8_8192


namespace no_distinct_positive_integers_2007_l8_8040

theorem no_distinct_positive_integers_2007 (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  ¬ (x^2007 + y! = y^2007 + x!) :=
by
  sorry

end no_distinct_positive_integers_2007_l8_8040


namespace oprq_possible_figures_l8_8982

theorem oprq_possible_figures (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  -- Define the points P, Q, and R
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 - x2, y1 - y2)
  -- Proving the geometric possibilities
  (∃ k : ℝ, x1 = k * x2 ∧ y1 = k * y2) ∨
  -- When the points are collinear
  ((x1 + x2, y1 + y2) = (x1, y1)) :=
sorry

end oprq_possible_figures_l8_8982


namespace max_min_difference_l8_8793

def y (x : ℝ) : ℝ := x * abs (3 - x) - (x - 3) * abs x

theorem max_min_difference : (0 : ℝ) ≤ x → (x < 3 → y x ≤ y (3 / 4)) ∧ (x < 0 → y x = 0) ∧ (x ≥ 3 → y x = 0) → 
  (y (3 / 4) - (min (y 0) (min (y (-1)) (y 3)))) = 1.125 :=
by
  sorry

end max_min_difference_l8_8793


namespace total_feet_l8_8427

theorem total_feet (H C : ℕ) (h1 : H + C = 48) (h2 : H = 28) : 2 * H + 4 * C = 136 := 
by
  sorry

end total_feet_l8_8427


namespace fraction_of_seniors_study_japanese_l8_8048

variable (J S : ℝ)
variable (fraction_seniors fraction_juniors : ℝ)
variable (total_fraction_study_japanese : ℝ)

theorem fraction_of_seniors_study_japanese 
  (h1 : S = 2 * J)
  (h2 : fraction_juniors = 3 / 4)
  (h3 : total_fraction_study_japanese = 1 / 3) :
  fraction_seniors = 1 / 8 :=
by
  -- Here goes the proof.
  sorry

end fraction_of_seniors_study_japanese_l8_8048


namespace intersection_locus_is_vertical_line_l8_8794

/-- 
Given \( 0 < a < b \), lines \( l \) and \( m \) are drawn through the points \( A(a, 0) \) and \( B(b, 0) \), 
respectively, such that these lines intersect the parabola \( y^2 = x \) at four distinct points 
and these four points are concyclic. 

We want to prove that the locus of the intersection point \( P \) of lines \( l \) and \( m \) 
is the vertical line \( x = \frac{a + b}{2} \).
-/
theorem intersection_locus_is_vertical_line (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ P : ℝ × ℝ, P.fst = (a + b) / 2) := 
sorry

end intersection_locus_is_vertical_line_l8_8794


namespace translated_function_symmetry_center_l8_8321

theorem translated_function_symmetry_center :
  let f := fun x : ℝ => Real.sin (6 * x + π / 4)
  let g := fun x : ℝ => f (x / 3)
  let h := fun x : ℝ => g (x - π / 8)
  h π / 2 = 0 :=
by
  sorry

end translated_function_symmetry_center_l8_8321


namespace cheryl_distance_walked_l8_8509

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l8_8509


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l8_8492

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l8_8492


namespace quadratic_completion_l8_8134

theorem quadratic_completion (b c : ℝ) (h : (x : ℝ) → x^2 + 1600 * x + 1607 = (x + b)^2 + c) (hb : b = 800) (hc : c = -638393) : 
  c / b = -797.99125 := by
  sorry

end quadratic_completion_l8_8134


namespace evaluate_expression_l8_8188

theorem evaluate_expression : 
  -((5: ℤ) ^ 2) - (-(3: ℤ) ^ 3) * ((2: ℚ) / 9) - 9 * |((-(2: ℚ)) / 3)| = -25 := by
  sorry

end evaluate_expression_l8_8188


namespace diagonal_crosses_700_cubes_l8_8165

noncomputable def num_cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c

theorem diagonal_crosses_700_cubes :
  num_cubes_crossed 200 300 350 = 700 :=
sorry

end diagonal_crosses_700_cubes_l8_8165


namespace total_wash_time_l8_8799

theorem total_wash_time (clothes_time : ℕ) (towels_time : ℕ) (sheets_time : ℕ) (total_time : ℕ) 
  (h1 : clothes_time = 30) 
  (h2 : towels_time = 2 * clothes_time) 
  (h3 : sheets_time = towels_time - 15) 
  (h4 : total_time = clothes_time + towels_time + sheets_time) : 
  total_time = 135 := 
by 
  sorry

end total_wash_time_l8_8799


namespace green_caps_percentage_l8_8626

variable (total_caps : ℕ) (red_caps : ℕ)

def green_caps (total_caps red_caps: ℕ) : ℕ :=
  total_caps - red_caps

def percentage_of_green_caps (total_caps green_caps: ℕ) : ℕ :=
  (green_caps * 100) / total_caps

theorem green_caps_percentage :
  (total_caps = 125) →
  (red_caps = 50) →
  percentage_of_green_caps total_caps (green_caps total_caps red_caps) = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry  -- The proof is omitted 

end green_caps_percentage_l8_8626


namespace gcd_40_120_80_l8_8104

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end gcd_40_120_80_l8_8104


namespace sum_of_interior_numbers_eighth_row_l8_8868

def sum_of_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem sum_of_interior_numbers_eighth_row : sum_of_interior_numbers 8 = 126 :=
by
  sorry

end sum_of_interior_numbers_eighth_row_l8_8868


namespace probability_red_or_blue_marbles_l8_8965

theorem probability_red_or_blue_marbles (red blue green total : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_green : green = 6) (h_total : total = red + blue + green) :
  (red + blue) / total = 7 / 13 :=
by
  sorry

end probability_red_or_blue_marbles_l8_8965


namespace find_angle_y_l8_8035

theorem find_angle_y (angle_ABC angle_ABD angle_ADB y : ℝ)
  (h1 : angle_ABC = 115)
  (h2 : angle_ABD = 180 - angle_ABC)
  (h3 : angle_ADB = 30)
  (h4 : angle_ABD + angle_ADB + y = 180) :
  y = 85 := 
sorry

end find_angle_y_l8_8035


namespace seashells_broken_l8_8306

theorem seashells_broken (total_seashells : ℕ) (unbroken_seashells : ℕ) (broken_seashells : ℕ) : 
  total_seashells = 6 → unbroken_seashells = 2 → broken_seashells = total_seashells - unbroken_seashells → broken_seashells = 4 :=
by
  intros ht hu hb
  rw [ht, hu] at hb
  exact hb

end seashells_broken_l8_8306


namespace total_packages_l8_8973

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) (h1 : num_trucks = 7) (h2 : packages_per_truck = 70) : num_trucks * packages_per_truck = 490 := by
  sorry

end total_packages_l8_8973


namespace smallest_perimeter_of_consecutive_even_triangle_l8_8637

theorem smallest_perimeter_of_consecutive_even_triangle (n : ℕ) :
  (2 * n + 2 * n + 2 > 2 * n + 4) ∧
  (2 * n + 2 * n + 4 > 2 * n + 2) ∧
  (2 * n + 2 + 2 * n + 4 > 2 * n) →
  2 * n + (2 * n + 2) + (2 * n + 4) = 18 :=
by 
  sorry

end smallest_perimeter_of_consecutive_even_triangle_l8_8637


namespace eraser_cost_l8_8236

noncomputable def price_of_erasers 
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  (bundle_count : ℝ) -- number of bundles sold
  (total_earned : ℝ) -- total amount earned
  (discount : ℝ) -- discount percentage for 20 bundles
  (bundle_contents : ℕ) -- 1 pencil and 2 erasers per bundle
  (price_ratio : ℝ) -- price ratio of eraser to pencil
  : Prop := 
  E = 0.5 * P ∧ -- The price of the erasers is 1/2 the price of the pencils.
  bundle_count = 20 ∧ -- The store sold a total of 20 bundles.
  total_earned = 80 ∧ -- The store earned $80.
  discount = 30 ∧ -- 30% discount for 20 bundles
  bundle_contents = 1 + 2 -- A bundle consists of 1 pencil and 2 erasers

theorem eraser_cost
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  : price_of_erasers P E 20 80 30 (1 + 2) 0.5 → E = 1.43 :=
by
  intro h
  sorry

end eraser_cost_l8_8236


namespace e_count_estimation_l8_8580

-- Define the various parameters used in the conditions
def num_problems : Nat := 76
def avg_words_per_problem : Nat := 40
def avg_letters_per_word : Nat := 5
def frequency_of_e : Float := 0.1
def actual_e_count : Nat := 1661

-- The goal is to prove that the actual number of "e"s is 1661
theorem e_count_estimation : actual_e_count = 1661 := by
  -- Sorry, no proof is required.
  sorry

end e_count_estimation_l8_8580


namespace find_N_l8_8689

theorem find_N (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 :=
by
  intros h
  -- Sorry to skip the proof.
  sorry

end find_N_l8_8689


namespace josh_total_money_left_l8_8308

-- Definitions of the conditions
def profit_per_bracelet : ℝ := 1.5 - 1
def total_bracelets : ℕ := 12
def cost_of_cookies : ℝ := 3

-- The proof problem: 
theorem josh_total_money_left : total_bracelets * profit_per_bracelet - cost_of_cookies = 3 :=
by
  sorry

end josh_total_money_left_l8_8308


namespace interest_calculation_l8_8948

theorem interest_calculation :
  ∃ n : ℝ, 
  (1000 * 0.03 * n + 1400 * 0.05 * n = 350) →
  n = 3.5 := 
by 
  sorry

end interest_calculation_l8_8948


namespace conic_section_is_ellipse_l8_8874

/-- Given two fixed points (0, 2) and (4, -1) and the equation 
    sqrt(x^2 + (y - 2)^2) + sqrt((x - 4)^2 + (y + 1)^2) = 12, 
    prove that the conic section is an ellipse. -/
theorem conic_section_is_ellipse 
  (x y : ℝ)
  (h : Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 4)^2 + (y + 1)^2) = 12) :
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (0, 2) ∧ 
    F2 = (4, -1) ∧ 
    ∀ (P : ℝ × ℝ), P = (x, y) → 
      Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 12 := 
sorry

end conic_section_is_ellipse_l8_8874


namespace f_at_pos_eq_l8_8673

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 0 then x * (x - 1)
  else if h : x > 0 then -x * (x + 1)
  else 0

theorem f_at_pos_eq (x : ℝ) (hx : 0 < x) : f x = -x * (x + 1) :=
by
  -- Assume f is an odd function
  have h_odd : ∀ x : ℝ, f (-x) = -f x := sorry
  
  -- Given for x in (-∞, 0), f(x) = x * (x - 1)
  have h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1) := sorry
  
  -- Prove for x > 0, f(x) = -x * (x + 1)
  sorry

end f_at_pos_eq_l8_8673


namespace exists_neg_monomial_l8_8548

theorem exists_neg_monomial (a : ℤ) (x y : ℤ) (m n : ℕ) (hq : a < 0) (hd : m + n = 5) :
  ∃ a m n, a < 0 ∧ m + n = 5 ∧ a * x^m * y^n = -x^2 * y^3 :=
by
  sorry

end exists_neg_monomial_l8_8548


namespace find_k_l8_8553

noncomputable def arithmetic_sum (n : ℕ) (a1 d : ℚ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_k 
  (a1 d : ℚ) (k : ℕ)
  (h1 : arithmetic_sum (k - 2) a1 d = -4)
  (h2 : arithmetic_sum k a1 d = 0)
  (h3 : arithmetic_sum (k + 2) a1 d = 8) :
  k = 6 :=
by
  sorry

end find_k_l8_8553


namespace number_of_possible_teams_l8_8798

-- Definitions for the conditions
def num_goalkeepers := 3
def num_defenders := 5
def num_midfielders := 5
def num_strikers := 5

-- The number of ways to choose x from y
def choose (y x : ℕ) : ℕ := Nat.factorial y / (Nat.factorial x * Nat.factorial (y - x))

-- Main proof problem statement
theorem number_of_possible_teams :
  (choose num_goalkeepers 1) *
  (choose num_strikers 2) *
  (choose num_midfielders 4) *
  (choose (num_defenders + (num_midfielders - 4)) 4) = 2250 := by
  sorry

end number_of_possible_teams_l8_8798


namespace positive_number_property_l8_8678

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_property : (x^2 / 100) = 9) : x = 30 :=
by
  sorry

end positive_number_property_l8_8678


namespace derivative_of_x_ln_x_l8_8743

noncomputable
def x_ln_x (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_ln_x (x : ℝ) (hx : x > 0) :
  deriv (x_ln_x) x = 1 + Real.log x :=
by
  -- Proof body, with necessary assumptions and justifications
  sorry

end derivative_of_x_ln_x_l8_8743


namespace curtain_length_correct_l8_8169

-- Define the problem conditions in Lean
def room_height_feet : ℝ := 8
def feet_to_inches : ℝ := 12
def additional_material_inches : ℝ := 5

-- Define the target length of the curtains
def curtain_length_inches : ℝ :=
  (room_height_feet * feet_to_inches) + additional_material_inches

-- Statement to prove the length of the curtains is 101 inches.
theorem curtain_length_correct :
  curtain_length_inches = 101 := by
  sorry

end curtain_length_correct_l8_8169


namespace glass_volume_correct_l8_8883

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l8_8883


namespace Megan_seashells_needed_l8_8812

-- Let x be the number of additional seashells needed
def seashells_needed (total_seashells desired_seashells : Nat) : Nat :=
  desired_seashells - total_seashells

-- Given conditions
def current_seashells : Nat := 19
def desired_seashells : Nat := 25

-- The equivalent proof problem
theorem Megan_seashells_needed : seashells_needed current_seashells desired_seashells = 6 := by
  sorry

end Megan_seashells_needed_l8_8812


namespace find_M_value_when_x_3_l8_8887

-- Definitions based on the given conditions
def polynomial (a b c d x : ℝ) : ℝ := a*x^5 + b*x^3 + c*x + d

-- Given conditions
variables (a b c d : ℝ)
axiom h₀ : polynomial a b c d 0 = -5
axiom h₁ : polynomial a b c d (-3) = 7

-- Desired statement: Prove that the value of polynomial at x = 3 is -17
theorem find_M_value_when_x_3 : polynomial a b c d 3 = -17 :=
by sorry

end find_M_value_when_x_3_l8_8887


namespace prob_sin_ge_half_l8_8291

theorem prob_sin_ge_half : 
  let a := -Real.pi / 6
  let b := Real.pi / 2
  let p := (Real.pi / 2 - Real.pi / 6) / (Real.pi / 2 + Real.pi / 6)
  a ≤ b ∧ a = -Real.pi / 6 ∧ b = Real.pi / 2 → p = 1 / 2 :=
by
  sorry

end prob_sin_ge_half_l8_8291


namespace ax5_by5_eq_neg1065_l8_8939

theorem ax5_by5_eq_neg1065 (a b x y : ℝ) 
  (h1 : a*x + b*y = 5) 
  (h2 : a*x^2 + b*y^2 = 9) 
  (h3 : a*x^3 + b*y^3 = 20) 
  (h4 : a*x^4 + b*y^4 = 48) 
  (h5 : x + y = -15) 
  (h6 : x^2 + y^2 = 55) : 
  a * x^5 + b * y^5 = -1065 := 
sorry

end ax5_by5_eq_neg1065_l8_8939


namespace find_number_l8_8549

-- Define the given condition
def number_div_property (num : ℝ) : Prop :=
  num / 0.3 = 7.3500000000000005

-- State the theorem to prove
theorem find_number (num : ℝ) (h : number_div_property num) : num = 2.205 :=
by sorry

end find_number_l8_8549


namespace pitcher_fill_four_glasses_l8_8390

variable (P G : ℚ) -- P: Volume of pitcher, G: Volume of one glass
variable (h : P / 2 = 3 * G)

theorem pitcher_fill_four_glasses : (4 * G = 2 * P / 3) :=
by
  sorry

end pitcher_fill_four_glasses_l8_8390


namespace card_collection_average_l8_8702

theorem card_collection_average (n : ℕ) (h : (2 * n + 1) / 3 = 2017) : n = 3025 :=
by
  sorry

end card_collection_average_l8_8702


namespace projection_coordinates_eq_zero_l8_8214

theorem projection_coordinates_eq_zero (x y z : ℝ) :
  let M := (x, y, z)
  let M₁ := (x, y, 0)
  let M₂ := (0, y, 0)
  let M₃ := (0, 0, 0)
  M₃ = (0, 0, 0) :=
sorry

end projection_coordinates_eq_zero_l8_8214


namespace incorrect_rounding_statement_l8_8834

def rounded_to_nearest (n : ℝ) (accuracy : ℝ) : Prop :=
  ∃ (k : ℤ), abs (n - k * accuracy) < accuracy / 2

theorem incorrect_rounding_statement :
  ¬ rounded_to_nearest 23.9 10 :=
sorry

end incorrect_rounding_statement_l8_8834


namespace factorization_identity_l8_8930

theorem factorization_identity (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) :=
by
  sorry

end factorization_identity_l8_8930


namespace arithmetic_progression_impossible_geometric_progression_possible_l8_8975

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end arithmetic_progression_impossible_geometric_progression_possible_l8_8975


namespace bus_driver_hours_worked_last_week_l8_8499

-- Definitions for given conditions
def regular_rate : ℝ := 12
def passenger_rate : ℝ := 0.50
def overtime_rate_1 : ℝ := 1.5 * regular_rate
def overtime_rate_2 : ℝ := 2 * regular_rate
def total_compensation : ℝ := 1280
def total_passengers : ℝ := 350
def earnings_from_passengers : ℝ := total_passengers * passenger_rate
def earnings_from_hourly_rate : ℝ := total_compensation - earnings_from_passengers
def regular_hours : ℝ := 40
def first_tier_overtime_hours : ℝ := 5

-- Theorem to prove the number of hours worked is 67
theorem bus_driver_hours_worked_last_week :
  ∃ (total_hours : ℝ),
    total_hours = 67 ∧
    earnings_from_passengers = total_passengers * passenger_rate ∧
    earnings_from_hourly_rate = total_compensation - earnings_from_passengers ∧
    (∃ (overtime_hours : ℝ),
      (overtime_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2) ∧
      total_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2 )
  :=
sorry

end bus_driver_hours_worked_last_week_l8_8499


namespace cube_difference_l8_8552

theorem cube_difference {a b : ℝ} (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 :=
sorry

end cube_difference_l8_8552


namespace algae_plants_in_milford_lake_l8_8454

theorem algae_plants_in_milford_lake (original : ℕ) (increase : ℕ) : (original = 809) → (increase = 2454) → (original + increase = 3263) :=
by
  sorry

end algae_plants_in_milford_lake_l8_8454


namespace sum_of_integers_l8_8619

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 :=
by
  sorry

end sum_of_integers_l8_8619


namespace bridge_length_l8_8316

noncomputable def length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  let total_distance := speed_of_train_ms * time_seconds
  total_distance - length_of_train

theorem bridge_length (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) (h1 : length_of_train = 170) (h2 : speed_of_train_kmh = 45) (h3 : time_seconds = 30) :
  length_of_bridge length_of_train speed_of_train_kmh time_seconds = 205 :=
by 
  rw [h1, h2, h3]
  unfold length_of_bridge
  simp
  sorry

end bridge_length_l8_8316


namespace range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l8_8090

theorem range_of_x_if_p_and_q_true (a : ℝ) (p q : ℝ → Prop) (h_a : a = 1) (h_p : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (h_q : ∀ x, q x ↔ (x-3)^2 < 1) (h_pq : ∀ x, p x ∧ q x) :
  ∀ x, 2 < x ∧ x < 3 :=
by
  sorry

theorem range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q (p q : ℝ → Prop) (h_neg : ∀ x, ¬p x → ¬q x) : 
  ∀ a : ℝ, a > 0 → (a ≥ 4/3 ∧ a ≤ 2) :=
by
  sorry

end range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l8_8090


namespace sum_of_eight_terms_l8_8120

theorem sum_of_eight_terms :
  (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) = 3125000 :=
by
  sorry

end sum_of_eight_terms_l8_8120


namespace proportional_segments_l8_8878

-- Define the tetrahedron and points
structure Tetrahedron :=
(A B C D O A1 B1 C1 : ℝ)

-- Define the conditions of the problem
variables {tetra : Tetrahedron}

-- Define the segments and their relationships
axiom segments_parallel (DA : ℝ) (DB : ℝ) (DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1

-- The theorem to prove, which follows directly from the given axiom 
theorem proportional_segments (DA DB DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1 :=
segments_parallel DA DB DC OA1 OB1 OC1

end proportional_segments_l8_8878


namespace kindergarten_classes_l8_8264

theorem kindergarten_classes :
  ∃ (j a m : ℕ), j + a + m = 32 ∧
                  j > 0 ∧ a > 0 ∧ m > 0 ∧
                  j / 2 + a / 4 + m / 8 = 6 ∧
                  (j = 4 ∧ a = 4 ∧ m = 24) :=
by {
  sorry
}

end kindergarten_classes_l8_8264


namespace abs_ineq_range_k_l8_8472

theorem abs_ineq_range_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 :=
by
  sorry

end abs_ineq_range_k_l8_8472


namespace digit_difference_is_one_l8_8501

theorem digit_difference_is_one {p q : ℕ} (h : 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ p ≠ q)
  (digits_distinct : ∀ n ∈ [p, q], ∀ m ∈ [p, q], n ≠ m)
  (interchange_effect : 10 * p + q - (10 * q + p) = 9) : p - q = 1 :=
sorry

end digit_difference_is_one_l8_8501


namespace probability_blue_face_up_l8_8228

def cube_probability_blue : ℚ := 
  let total_faces := 6
  let blue_faces := 4
  blue_faces / total_faces

theorem probability_blue_face_up :
  cube_probability_blue = 2 / 3 :=
by
  sorry

end probability_blue_face_up_l8_8228


namespace water_level_decrease_3m_l8_8926

-- Definitions from conditions
def increase (amount : ℝ) : ℝ := amount
def decrease (amount : ℝ) : ℝ := -amount

-- The claim to be proven
theorem water_level_decrease_3m : decrease 3 = -3 :=
by
  sorry

end water_level_decrease_3m_l8_8926


namespace relative_positions_of_P_on_AB_l8_8003

theorem relative_positions_of_P_on_AB (A B P : ℝ) : 
  A ≤ B → (A ≤ P ∧ P ≤ B ∨ P = A ∨ P = B ∨ P < A ∨ P > B) :=
by
  intro hAB
  sorry

end relative_positions_of_P_on_AB_l8_8003


namespace track_length_l8_8502

theorem track_length (x : ℝ) (tom_dist1 jerry_dist1 : ℝ) (tom_dist2 jerry_dist2 : ℝ) (deg_gap : ℝ) :
  deg_gap = 120 ∧ 
  tom_dist1 = 120 ∧ 
  (tom_dist1 + jerry_dist1 = x * deg_gap / 360) ∧ 
  (jerry_dist1 + jerry_dist2 = x * deg_gap / 360 + 180) →
  x = 630 :=
by
  sorry

end track_length_l8_8502


namespace radar_placement_problem_l8_8177

noncomputable def max_distance (n : ℕ) (coverage_radius : ℝ) (central_angle : ℝ) : ℝ :=
  coverage_radius / Real.sin (central_angle / 2)

noncomputable def ring_area (inner_radius : ℝ) (outer_radius : ℝ) : ℝ :=
  Real.pi * (outer_radius ^ 2 - inner_radius ^ 2)

theorem radar_placement_problem (r : ℝ := 13) (n : ℕ := 5) (width : ℝ := 10) :
  let angle := 2 * Real.pi / n
  let max_dist := max_distance n r angle
  let inner_radius := (r ^ 2 - (r - width) ^ 2) / Real.tan (angle / 2)
  let outer_radius := inner_radius + width
  max_dist = 12 / Real.sin (angle / 2) ∧
  ring_area inner_radius outer_radius = 240 * Real.pi / Real.tan (angle / 2) :=
by
  sorry

end radar_placement_problem_l8_8177


namespace compute_58_sq_pattern_l8_8108

theorem compute_58_sq_pattern : (58 * 58 = 56 * 60 + 4) :=
by
  sorry

end compute_58_sq_pattern_l8_8108


namespace usual_time_to_bus_stop_l8_8432

theorem usual_time_to_bus_stop
  (T : ℕ) (S : ℕ)
  (h : S * T = (4/5 * S) * (T + 9)) :
  T = 36 :=
by
  sorry

end usual_time_to_bus_stop_l8_8432


namespace max_tetrahedron_in_cube_l8_8024

open Real

noncomputable def cube_edge_length : ℝ := 6
noncomputable def max_tetrahedron_edge_length (a : ℝ) : Prop :=
  ∃ x : ℝ, x = 2 * sqrt 6 ∧ 
          (∃ R : ℝ, R = (a * sqrt 3) / 2 ∧ x / sqrt (2 / 3) = 4 * R / 3)

theorem max_tetrahedron_in_cube : max_tetrahedron_edge_length cube_edge_length :=
sorry

end max_tetrahedron_in_cube_l8_8024


namespace integer_solution_x_l8_8323

theorem integer_solution_x (x : ℤ) (h₁ : x + 8 > 10) (h₂ : -3 * x < -9) : x ≥ 4 ↔ x > 3 := by
  sorry

end integer_solution_x_l8_8323


namespace simplify_sqrt_expression_l8_8462

theorem simplify_sqrt_expression :
  (Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)) = 6 :=
by
  sorry

end simplify_sqrt_expression_l8_8462


namespace combined_resistance_l8_8442

theorem combined_resistance (x y : ℝ) (r : ℝ) (hx : x = 4) (hy : y = 6) :
  (1 / r) = (1 / x) + (1 / y) → r = 12 / 5 :=
by
  sorry

end combined_resistance_l8_8442


namespace john_read_bible_in_weeks_l8_8091

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end john_read_bible_in_weeks_l8_8091


namespace fraction_of_blue_cars_l8_8520

-- Definitions of the conditions
def total_cars : ℕ := 516
def red_cars : ℕ := total_cars / 2
def black_cars : ℕ := 86
def blue_cars : ℕ := total_cars - (red_cars + black_cars)

-- Statement to prove that the fraction of blue cars is 1/3
theorem fraction_of_blue_cars :
  (blue_cars : ℚ) / total_cars = 1 / 3 :=
by
  sorry -- Proof to be filled in

end fraction_of_blue_cars_l8_8520


namespace infinite_series_computation_l8_8453

noncomputable def infinite_series_sum (a b : ℝ) : ℝ :=
  ∑' n : ℕ, if n = 0 then (0 : ℝ) else
    (1 : ℝ) / ((2 * (n - 1 : ℕ) * a - (n - 2 : ℕ) * b) * (2 * n * a - (n - 1 : ℕ) * b))

theorem infinite_series_computation (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b) :
  infinite_series_sum a b = 1 / ((2 * a - b) * (2 * b)) :=
by
  sorry

end infinite_series_computation_l8_8453


namespace vec_subtraction_l8_8823

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (0, 1)

theorem vec_subtraction : a - 2 • b = (-1, 0) := by
  sorry

end vec_subtraction_l8_8823


namespace find_three_digit_number_divisible_by_5_l8_8059

theorem find_three_digit_number_divisible_by_5 {n x : ℕ} (hx1 : 100 ≤ x) (hx2 : x < 1000) (hx3 : x % 5 = 0) (hx4 : x = n^3 + n^2) : x = 150 ∨ x = 810 := 
by
  sorry

end find_three_digit_number_divisible_by_5_l8_8059


namespace intersection_M_N_l8_8507

theorem intersection_M_N :
  let M := {x | x^2 < 36}
  let N := {2, 4, 6, 8}
  M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l8_8507


namespace shares_sum_4000_l8_8642

variables (w x y z : ℝ)

def relation_z_w : Prop := z = 1.20 * w
def relation_y_z : Prop := y = 1.25 * z
def relation_x_y : Prop := x = 1.35 * y
def w_after_3_years : ℝ := 8 * w
def z_after_3_years : ℝ := 8 * z
def y_after_3_years : ℝ := 8 * y
def x_after_3_years : ℝ := 8 * x

theorem shares_sum_4000 (w : ℝ) :
  relation_z_w w z →
  relation_y_z z y →
  relation_x_y y x →
  x_after_3_years x + y_after_3_years y + z_after_3_years z + w_after_3_years w = 4000 :=
by
  intros h_z_w h_y_z h_x_y
  rw [relation_z_w, relation_y_z, relation_x_y] at *
  sorry

end shares_sum_4000_l8_8642


namespace correct_calculation_l8_8374

theorem correct_calculation (x : ℝ) :
  (x / 5 + 16 = 58) → (x / 15 + 74 = 88) :=
by
  sorry

end correct_calculation_l8_8374


namespace cape_may_multiple_l8_8782

theorem cape_may_multiple :
  ∃ x : ℕ, 26 = x * 7 + 5 ∧ x = 3 :=
by
  sorry

end cape_may_multiple_l8_8782


namespace morgan_change_l8_8122

theorem morgan_change:
  let hamburger := 5.75
  let onion_rings := 2.50
  let smoothie := 3.25
  let side_salad := 3.75
  let cake := 4.20
  let total_cost := hamburger + onion_rings + smoothie + side_salad + cake
  let payment := 50
  let change := payment - total_cost
  ℝ := by
    exact sorry

end morgan_change_l8_8122


namespace max_ab_value_1_half_l8_8845

theorem max_ab_value_1_half 
  (a b : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : a + 2 * b = 1) :
  a = 1 / 2 → ab = 1 / 8 :=
sorry

end max_ab_value_1_half_l8_8845


namespace common_ratio_geom_series_l8_8632

theorem common_ratio_geom_series :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -16/21
  let a₃ : ℚ := -64/63
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -4/3 := 
by
  sorry

end common_ratio_geom_series_l8_8632


namespace largest_room_length_l8_8431

theorem largest_room_length (L : ℕ) (w_large w_small l_small diff_area : ℕ)
  (h1 : w_large = 45)
  (h2 : w_small = 15)
  (h3 : l_small = 8)
  (h4 : diff_area = 1230)
  (h5 : w_large * L - (w_small * l_small) = diff_area) :
  L = 30 :=
by sorry

end largest_room_length_l8_8431


namespace sin_cos_pi_over_12_l8_8012

theorem sin_cos_pi_over_12 :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
sorry

end sin_cos_pi_over_12_l8_8012


namespace arithmetic_seq_proof_l8_8544

open Nat

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ :=
n * (a 1) + n * (n - 1) / 2 * d

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ)
  (h1 : arithmetic_seq a d)
  (h2 : a 2 = 0)
  (h3 : sum_of_arithmetic_seq a d 3 + sum_of_arithmetic_seq a d 4 = 6) :
  a 5 + a 6 = 21 :=
sorry

end arithmetic_seq_proof_l8_8544


namespace correct_microorganism_dilution_statement_l8_8851

def microorganism_dilution_conditions (A B C D : Prop) : Prop :=
  (A ↔ ∀ (dilutions : ℕ) (n : ℕ), 1000 ≤ dilutions ∧ dilutions ≤ 10000000) ∧
  (B ↔ ∀ (dilutions : ℕ) (actinomycetes : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (C ↔ ∀ (dilutions : ℕ) (fungi : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (D ↔ ∀ (dilutions : ℕ) (bacteria_first_time : ℕ), 10 ≤ dilutions ∧ dilutions ≤ 10000000)

theorem correct_microorganism_dilution_statement (A B C D : Prop)
  (h : microorganism_dilution_conditions A B C D) : D :=
sorry

end correct_microorganism_dilution_statement_l8_8851


namespace triangle_inequality_proof_l8_8473

theorem triangle_inequality_proof 
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_proof_l8_8473


namespace money_last_weeks_l8_8651

theorem money_last_weeks (mowing_earning : ℕ) (weeding_earning : ℕ) (spending_per_week : ℕ) 
  (total_amount : ℕ) (weeks : ℕ) :
  mowing_earning = 9 →
  weeding_earning = 18 →
  spending_per_week = 3 →
  total_amount = mowing_earning + weeding_earning →
  weeks = total_amount / spending_per_week →
  weeks = 9 :=
by
  intros
  sorry

end money_last_weeks_l8_8651


namespace correct_statements_count_l8_8378

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end correct_statements_count_l8_8378


namespace greatest_divisor_condition_l8_8657

-- Define conditions
def leaves_remainder (a b k : ℕ) : Prop := ∃ q : ℕ, a = b * q + k

-- Define the greatest common divisor property
def gcd_of (a b k: ℕ) (g : ℕ) : Prop :=
  leaves_remainder a k g ∧ leaves_remainder b k g ∧ ∀ d : ℕ, (leaves_remainder a k d ∧ leaves_remainder b k d) → d ≤ g

theorem greatest_divisor_condition 
  (N : ℕ) (h1 : leaves_remainder 1657 N 6) (h2 : leaves_remainder 2037 N 5) :
  N = 127 :=
sorry

end greatest_divisor_condition_l8_8657


namespace analytical_expression_range_of_t_l8_8826

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t_l8_8826


namespace cos_double_angle_l8_8791

variable (θ : Real)

theorem cos_double_angle (h : ∑' n, (Real.cos θ)^(2 * n) = 7) : Real.cos (2 * θ) = 5 / 7 := 
  by sorry

end cos_double_angle_l8_8791


namespace pies_sold_in_week_l8_8913

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l8_8913


namespace proof_problem_l8_8688

theorem proof_problem
  (a b : ℝ)
  (h1 : a = -(-3))
  (h2 : b = - (- (1 / 2))⁻¹)
  (m n : ℝ) :
  (|m - a| + |n + b| = 0) → (a = 3 ∧ b = -2 ∧ m = 3 ∧ n = -2) :=
by {
  sorry
}

end proof_problem_l8_8688


namespace matching_pair_probability_correct_l8_8838

-- Define the basic assumptions (conditions)
def black_pairs : Nat := 7
def brown_pairs : Nat := 4
def gray_pairs : Nat := 3
def red_pairs : Nat := 2

def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs + red_pairs
def total_shoes : Nat := 2 * total_pairs

-- The probability calculation will be shown as the final proof requirement
def matching_color_probability : Rat :=  (14 * 7 + 8 * 4 + 6 * 3 + 4 * 2 : Int) / (32 * 31 : Int)

-- The target statement to be proven
theorem matching_pair_probability_correct :
  matching_color_probability = (39 / 248 : Rat) :=
by
  sorry

end matching_pair_probability_correct_l8_8838


namespace total_percentage_of_failed_candidates_l8_8260

-- Define the given conditions
def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_of_boys_passed : ℚ := 0.28
def percentage_of_girls_passed : ℚ := 0.32

-- Define the proof statement
theorem total_percentage_of_failed_candidates : 
  (total_candidates - (percentage_of_boys_passed * number_of_boys + percentage_of_girls_passed * number_of_girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end total_percentage_of_failed_candidates_l8_8260


namespace am_gm_inequality_l8_8679

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z :=
by sorry

end am_gm_inequality_l8_8679


namespace polygon_edges_of_set_S_l8_8916

variable (a : ℝ)

def in_set_S(x y : ℝ) : Prop :=
  (a / 2 ≤ x ∧ x ≤ 2 * a) ∧
  (a / 2 ≤ y ∧ y ≤ 2 * a) ∧
  (x + y ≥ a) ∧
  (x + a ≥ y) ∧
  (y + a ≥ x)

theorem polygon_edges_of_set_S (a : ℝ) (h : 0 < a) :
  (∃ n, ∀ x y, in_set_S a x y → n = 6) :=
sorry

end polygon_edges_of_set_S_l8_8916


namespace total_cost_food_l8_8351

theorem total_cost_food
  (beef_pounds : ℕ)
  (beef_cost_per_pound : ℕ)
  (chicken_pounds : ℕ)
  (chicken_cost_per_pound : ℕ)
  (h_beef : beef_pounds = 1000)
  (h_beef_cost : beef_cost_per_pound = 8)
  (h_chicken : chicken_pounds = 2 * beef_pounds)
  (h_chicken_cost : chicken_cost_per_pound = 3) :
  (beef_pounds * beef_cost_per_pound + chicken_pounds * chicken_cost_per_pound = 14000) :=
by
  sorry

end total_cost_food_l8_8351


namespace february_first_day_of_week_l8_8628

theorem february_first_day_of_week 
  (feb13_is_wednesday : ∃ day, day = 13 ∧ day_of_week = "Wednesday") :
  ∃ day, day = 1 ∧ day_of_week = "Friday" :=
sorry

end february_first_day_of_week_l8_8628


namespace sphere_diameter_l8_8004

theorem sphere_diameter (r : ℝ) (V : ℝ) (threeV : ℝ) (a b : ℕ) :
  (∀ (r : ℝ), r = 5 →
  V = (4 / 3) * π * r^3 →
  threeV = 3 * V →
  D = 2 * (3 * V * 3 / (4 * π))^(1 / 3) →
  D = a * b^(1 / 3) →
  a = 10 ∧ b = 3) →
  a + b = 13 :=
by
  intros
  sorry

end sphere_diameter_l8_8004


namespace pushups_total_l8_8117

theorem pushups_total (x melanie david karen john : ℕ) 
  (hx : x = 51)
  (h_melanie : melanie = 2 * x - 7)
  (h_david : david = x + 22)
  (h_avg : (x + melanie + david) / 3 = (x + (2 * x - 7) + (x + 22)) / 3)
  (h_karen : karen = (x + (2 * x - 7) + (x + 22)) / 3 - 5)
  (h_john : john = (x + 22) - 4) :
  john + melanie + karen = 232 := by
  sorry

end pushups_total_l8_8117


namespace OH_over_ON_eq_2_no_other_common_points_l8_8060

noncomputable def coordinates (t p : ℝ) : ℝ × ℝ :=
  (t^2 / (2 * p), t)

noncomputable def symmetric_point (M P : ℝ × ℝ) : ℝ × ℝ :=
  let (xM, yM) := M;
  let (xP, yP) := P;
  (2 * xP - xM, 2 * yP - yM)

noncomputable def line_ON (p t : ℝ) : ℝ → ℝ :=
  λ x => (p / t) * x

noncomputable def line_MH (t p : ℝ) : ℝ → ℝ :=
  λ x => (p / (2 * t)) * x + t

noncomputable def point_H (t p : ℝ) : ℝ × ℝ :=
  (2 * t^2 / p, 2 * t)

theorem OH_over_ON_eq_2
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  (H.snd) / (N.snd) = 2 := by
  sorry

theorem no_other_common_points
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  ∀ y, (y ≠ H.snd → ¬ ∃ x, line_MH t p x = y ∧ y^2 = 2 * p * x) := by 
  sorry

end OH_over_ON_eq_2_no_other_common_points_l8_8060


namespace eq_of_frac_sub_l8_8037

theorem eq_of_frac_sub (x : ℝ) (hx : x ≠ 1) : 
  (2 / (x^2 - 1) - 1 / (x - 1)) = - (1 / (x + 1)) := 
by sorry

end eq_of_frac_sub_l8_8037


namespace max_value_of_a2b3c4_l8_8821

open Real

theorem max_value_of_a2b3c4
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683 / 472392 :=
sorry

end max_value_of_a2b3c4_l8_8821


namespace compute_sqrt_eq_419_l8_8285

theorem compute_sqrt_eq_419 : Real.sqrt ((22 * 21 * 20 * 19) + 1) = 419 :=
by
  sorry

end compute_sqrt_eq_419_l8_8285


namespace reciprocals_sum_of_roots_l8_8645

theorem reciprocals_sum_of_roots (r s γ δ : ℚ) (h1 : 7 * r^2 + 5 * r + 3 = 0) (h2 : 7 * s^2 + 5 * s + 3 = 0) (h3 : γ = 1/r) (h4 : δ = 1/s) :
  γ + δ = -5/3 := 
  by 
    sorry

end reciprocals_sum_of_roots_l8_8645


namespace no_viable_schedule_l8_8828

theorem no_viable_schedule :
  ∀ (studentsA studentsB : ℕ), 
    studentsA = 29 → 
    studentsB = 32 → 
    ¬ ∃ (a b : ℕ),
      (a = 29 ∧ b = 32 ∧
      (a * b = studentsA * studentsB) ∧
      (∀ (x : ℕ), x < studentsA * studentsB →
        ∃ (iA iB : ℕ), 
          iA < studentsA ∧ 
          iB < studentsB ∧ 
          -- The condition that each pair is unique within this period
          ((iA + iB) % (studentsA * studentsB) = x))) := by
  sorry

end no_viable_schedule_l8_8828


namespace smaller_rectangle_dimensions_l8_8840

theorem smaller_rectangle_dimensions (side_length : ℝ) (L W : ℝ) 
  (h1 : side_length = 10) 
  (h2 : L + 2 * L = side_length) 
  (h3 : W = L) : 
  L = 10 / 3 ∧ W = 10 / 3 :=
by 
  sorry

end smaller_rectangle_dimensions_l8_8840


namespace value_of_a_even_function_monotonicity_on_interval_l8_8481

noncomputable def f (x : ℝ) := (1 / x^2) + 0 * x

theorem value_of_a_even_function 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (-x) = f x) : 
  (∃ a : ℝ, ∀ x, f x = (1 / x^2) + a * x) → a = 0 := by
  -- Placeholder for the proof
  sorry

theorem monotonicity_on_interval 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (1 / x^2) + 0 * x) 
  (h2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2 := by
  -- Placeholder for the proof
  sorry

end value_of_a_even_function_monotonicity_on_interval_l8_8481


namespace fill_time_difference_correct_l8_8410

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l8_8410


namespace pencil_partition_l8_8179

theorem pencil_partition (total_length green_fraction green_length remaining_length white_fraction half_remaining white_length gold_length : ℝ)
  (h1 : green_fraction = 7 / 10)
  (h2 : total_length = 2)
  (h3 : green_length = green_fraction * total_length)
  (h4 : remaining_length = total_length - green_length)
  (h5 : white_fraction = 1 / 2)
  (h6 : white_length = white_fraction * remaining_length)
  (h7 : gold_length = remaining_length - white_length) :
  (gold_length / remaining_length) = 1 / 2 :=
sorry

end pencil_partition_l8_8179


namespace find_c_share_l8_8474

noncomputable def shares (a b c d : ℝ) : Prop :=
  (5 * a = 4 * c) ∧ (7 * b = 4 * c) ∧ (2 * d = 4 * c) ∧ (a + b + c + d = 1200)

theorem find_c_share (A B C D : ℝ) (h : shares A B C D) : C = 275 :=
  by
  sorry

end find_c_share_l8_8474


namespace determine_b_from_quadratic_l8_8512

theorem determine_b_from_quadratic (b n : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, x^2 + b*x + 36 = (x + n)^2 + 20) : b = 8 := 
by 
  sorry

end determine_b_from_quadratic_l8_8512


namespace positive_difference_of_solutions_is_zero_l8_8922

theorem positive_difference_of_solutions_is_zero : ∀ (x : ℂ), (x ^ 2 + 3 * x + 4 = 0) → 
  ∀ (y : ℂ), (y ^ 2 + 3 * y + 4 = 0) → |y.re - x.re| = 0 :=
by
  intro x hx y hy
  sorry

end positive_difference_of_solutions_is_zero_l8_8922


namespace circle_equation_l8_8464

theorem circle_equation (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 4) :
    x^2 + y^2 - 2 * x - 3 = 0 :=
sorry

end circle_equation_l8_8464


namespace simplify_expression_l8_8196

-- Define the expressions and the simplification statement
def expr1 (x : ℝ) := (3 * x - 6) * (x + 8)
def expr2 (x : ℝ) := (x + 6) * (3 * x - 2)
def simplified (x : ℝ) := 2 * x - 36

theorem simplify_expression (x : ℝ) : expr1 x - expr2 x = simplified x := by
  sorry

end simplify_expression_l8_8196


namespace matrix_equation_l8_8749

-- Definitions from conditions
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![ -1, 4], ![ -6, 3]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1  -- Identity matrix

-- Given calculation of N^2
def N_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![ -23, 8], ![ -12, -15]]

-- Goal: prove that N^2 = r*N + s*I for r = 2 and s = -21
theorem matrix_equation (r s : ℤ) (h_r : r = 2) (h_s : s = -21) : N_squared = r • N + s • I := by
  sorry

end matrix_equation_l8_8749


namespace printed_value_l8_8207

theorem printed_value (X S : ℕ) (h1 : X = 5) (h2 : S = 0) : 
  (∃ n, S = (n * (3 * n + 7)) / 2 ∧ S ≥ 15000) → 
  X = 5 + 3 * 122 - 3 :=
by 
  sorry

end printed_value_l8_8207


namespace nicolai_peaches_6_pounds_l8_8303

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l8_8303


namespace fewer_people_correct_l8_8333

def pop_Springfield : ℕ := 482653
def pop_total : ℕ := 845640
def pop_new_city : ℕ := pop_total - pop_Springfield
def fewer_people : ℕ := pop_Springfield - pop_new_city

theorem fewer_people_correct : fewer_people = 119666 :=
by
  unfold fewer_people
  unfold pop_new_city
  unfold pop_total
  unfold pop_Springfield
  sorry

end fewer_people_correct_l8_8333


namespace probability_two_girls_l8_8959

-- Define the conditions
def total_students := 8
def total_girls := 5
def total_boys := 3
def choose_two_from_n (n : ℕ) := n * (n - 1) / 2

-- Define the question as a statement that the probability equals 5/14
theorem probability_two_girls
    (h1 : choose_two_from_n total_students = 28)
    (h2 : choose_two_from_n total_girls = 10) :
    (choose_two_from_n total_girls : ℚ) / choose_two_from_n total_students = 5 / 14 :=
by
  sorry

end probability_two_girls_l8_8959


namespace fruit_problem_l8_8839

theorem fruit_problem
  (A B C : ℕ)
  (hA : A = 4) 
  (hB : B = 6) 
  (hC : C = 12) :
  ∃ x : ℕ, 1 = x / 2 := 
by
  sorry

end fruit_problem_l8_8839


namespace original_price_l8_8701

theorem original_price (saving : ℝ) (percentage : ℝ) (h_saving : saving = 10) (h_percentage : percentage = 0.10) :
  ∃ OP : ℝ, OP = 100 :=
by
  sorry

end original_price_l8_8701


namespace find_C_l8_8691

-- Define the sum of interior angles of a triangle
def sum_of_triangle_angles := 180

-- Define the total angles sum in a closed figure formed by multiple triangles
def total_internal_angles := 1080

-- Define the value to prove
def C := total_internal_angles - sum_of_triangle_angles

theorem find_C:
  C = 900 := by
  sorry

end find_C_l8_8691


namespace donation_to_first_home_l8_8495

theorem donation_to_first_home :
  let total_donation := 700
  let donation_to_second := 225
  let donation_to_third := 230
  total_donation - donation_to_second - donation_to_third = 245 :=
by
  sorry

end donation_to_first_home_l8_8495


namespace problem_statement_l8_8647

theorem problem_statement (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 :=
by
  sorry

end problem_statement_l8_8647


namespace geometric_sequence_third_term_l8_8725

theorem geometric_sequence_third_term (a : ℕ → ℕ) (x : ℕ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : a 3 = x) (h_geom : ∀ n, a (n + 1) = a n * r) :
  x = 9 := 
sorry

end geometric_sequence_third_term_l8_8725


namespace digit_sum_is_14_l8_8243

theorem digit_sum_is_14 (P Q R S T : ℕ) 
  (h1 : P = 1)
  (h2 : Q = 0)
  (h3 : R = 2)
  (h4 : S = 5)
  (h5 : T = 6) :
  P + Q + R + S + T = 14 :=
by 
  sorry

end digit_sum_is_14_l8_8243


namespace max_number_of_kids_on_school_bus_l8_8510

-- Definitions based on the conditions from the problem
def totalRowsLowerDeck : ℕ := 15
def totalRowsUpperDeck : ℕ := 10
def capacityLowerDeckRow : ℕ := 5
def capacityUpperDeckRow : ℕ := 3
def reservedSeatsLowerDeck : ℕ := 10
def staffMembers : ℕ := 4

-- The total capacity of the lower and upper decks
def totalCapacityLowerDeck := totalRowsLowerDeck * capacityLowerDeckRow
def totalCapacityUpperDeck := totalRowsUpperDeck * capacityUpperDeckRow
def totalCapacity := totalCapacityLowerDeck + totalCapacityUpperDeck

-- The maximum number of different kids that can ride the bus
def maxKids := totalCapacity - reservedSeatsLowerDeck - staffMembers

theorem max_number_of_kids_on_school_bus : maxKids = 91 := 
by 
  -- Step-by-step proof not required for this task
  sorry

end max_number_of_kids_on_school_bus_l8_8510


namespace loaned_books_during_month_l8_8697

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def returned_percent : ℚ := 0.65
def end_books : ℕ := 68

-- Proof statement
theorem loaned_books_during_month (x : ℕ) 
  (h1 : returned_percent = 0.65)
  (h2 : initial_books = 75)
  (h3 : end_books = 68) :
  (0.35 * x : ℚ) = (initial_books - end_books) :=
sorry

end loaned_books_during_month_l8_8697


namespace trigonometric_expression_evaluation_l8_8590

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l8_8590


namespace find_x_squared_plus_y_squared_l8_8168

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l8_8168


namespace number_of_even_factors_of_n_l8_8097

noncomputable def n := 2^3 * 3^2 * 7^3

theorem number_of_even_factors_of_n : 
  (∃ (a : ℕ), (1 ≤ a ∧ a ≤ 3)) ∧ 
  (∃ (b : ℕ), (0 ≤ b ∧ b ≤ 2)) ∧ 
  (∃ (c : ℕ), (0 ≤ c ∧ c ≤ 3)) → 
  (even_nat_factors_count : ℕ) = 36 :=
by
  sorry

end number_of_even_factors_of_n_l8_8097


namespace num_ordered_triples_l8_8332

theorem num_ordered_triples : 
  {n : ℕ // ∃ (a b c : ℤ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = (2 * (a * b + b * c + c * a)) / 3 ∧ n = 3} :=
sorry

end num_ordered_triples_l8_8332


namespace judys_school_week_l8_8861

theorem judys_school_week
  (pencils_used : ℕ)
  (packs_cost : ℕ)
  (total_cost : ℕ)
  (days_period : ℕ)
  (pencils_per_pack : ℕ)
  (pencils_in_school_days : ℕ)
  (total_pencil_use : ℕ) :
  (total_cost / packs_cost * pencils_per_pack = total_pencil_use) →
  (total_pencil_use / days_period = pencils_used) →
  (pencils_in_school_days / pencils_used = 5) :=
sorry

end judys_school_week_l8_8861


namespace aquariums_have_13_saltwater_animals_l8_8386

theorem aquariums_have_13_saltwater_animals:
  ∀ x : ℕ, 26 * x = 52 → (∀ n : ℕ, n = 26 → (n * x = 52 ∧ x % 2 = 1 ∧ x > 1)) → x = 13 :=
by
  sorry

end aquariums_have_13_saltwater_animals_l8_8386


namespace original_sum_of_money_l8_8720

theorem original_sum_of_money (P R : ℝ) 
  (h1 : 720 = P + (P * R * 2) / 100) 
  (h2 : 1020 = P + (P * R * 7) / 100) : 
  P = 600 := 
by sorry

end original_sum_of_money_l8_8720


namespace manufacturing_section_degrees_l8_8994

theorem manufacturing_section_degrees (percentage : ℝ) (total_degrees : ℝ) (h1 : total_degrees = 360) (h2 : percentage = 35) : 
  ((percentage / 100) * total_degrees) = 126 :=
by
  sorry

end manufacturing_section_degrees_l8_8994


namespace GoldenRabbitCards_count_l8_8423

theorem GoldenRabbitCards_count :
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  golden_cards = 5904 :=
by
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  sorry

end GoldenRabbitCards_count_l8_8423


namespace mod_21_solution_l8_8032

theorem mod_21_solution (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n < 21) (h₂ : 47635 ≡ n [MOD 21]) : n = 19 :=
by
  sorry

end mod_21_solution_l8_8032


namespace part_I_part_II_l8_8962

-- Part (I)
theorem part_I (x a : ℝ) (h_a : a = 3) (h : abs (x - a) + abs (x + 5) ≥ 2 * abs (x + 5)) : x ≤ -1 := 
sorry

-- Part (II)
theorem part_II (a : ℝ) (h : ∀ x : ℝ, abs (x - a) + abs (x + 5) ≥ 6) : a ≥ 1 ∨ a ≤ -11 := 
sorry

end part_I_part_II_l8_8962


namespace p_sufficient_not_necessary_for_q_l8_8950

def p (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 ≤ 2
def q (x y : ℝ) : Prop := y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ ¬(∀ x y : ℝ, p x y → q x y) := by
  sorry

end p_sufficient_not_necessary_for_q_l8_8950


namespace angles_measure_l8_8222

theorem angles_measure (A B C : ℝ) (h1 : A + B = 180) (h2 : C = 1 / 2 * B) (h3 : A = 6 * B) :
  A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7 :=
by
  sorry

end angles_measure_l8_8222


namespace touching_line_eq_l8_8824

theorem touching_line_eq (f : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^4 - 4 * x^3) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = - (8 / 9) * x - (4 / 27)) ∧ 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (f x₁ = l x₁ ∧ f x₂ = l x₂) :=
by sorry

end touching_line_eq_l8_8824


namespace fruit_vendor_l8_8083

theorem fruit_vendor (x y a b : ℕ) (C1 : 60 * x + 40 * y = 3100) (C2 : x + y = 60) 
                     (C3 : 15 * a + 20 * b = 600) (C4 : 3 * a + 4 * b = 120)
                     (C5 : 3 * a + 4 * b + 3 * (x - a) + 4 * (y - b) = 250) :
  (x = 35 ∧ y = 25) ∧ (820 - 12 * a - 16 * b = 340) ∧ (a + b = 52 ∨ a + b = 53) :=
by
  sorry

end fruit_vendor_l8_8083


namespace difference_of_cubes_not_divisible_by_19_l8_8354

theorem difference_of_cubes_not_divisible_by_19 (a b : ℤ) : 
  ¬ (19 ∣ ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3)) := by
  sorry

end difference_of_cubes_not_divisible_by_19_l8_8354


namespace rhombus_min_rotation_l8_8584

theorem rhombus_min_rotation (α : ℝ) (h1 : α = 60) : ∃ θ, θ = 180 := 
by 
  -- The proof here will show that the minimum rotation angle is 180°
  sorry

end rhombus_min_rotation_l8_8584


namespace remaining_rectangle_area_l8_8997

theorem remaining_rectangle_area (s a b : ℕ) (hs : s = a + b) (total_area_cut : a^2 + b^2 = 40) : s^2 - 40 = 24 :=
by
  sorry

end remaining_rectangle_area_l8_8997


namespace billy_used_54_tickets_l8_8186

-- Definitions
def ferris_wheel_rides := 7
def bumper_car_rides := 3
def ferris_wheel_cost := 6
def bumper_car_cost := 4

-- Theorem Statement
theorem billy_used_54_tickets : 
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := 
by
  sorry

end billy_used_54_tickets_l8_8186


namespace minimum_value_of_a_l8_8419

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp (3 * Real.log x - x)) - x^2 - (a - 4) * x - 4

theorem minimum_value_of_a (h : ∀ x > 0, f x ≤ 0) : a ≥ 4 / Real.exp 2 := by
  sorry

end minimum_value_of_a_l8_8419


namespace scientific_notation_of_125000_l8_8928

theorem scientific_notation_of_125000 :
  125000 = 1.25 * 10^5 := sorry

end scientific_notation_of_125000_l8_8928


namespace percentage_decrease_l8_8968

variable (current_price original_price : ℝ)

theorem percentage_decrease (h1 : current_price = 760) (h2 : original_price = 1000) :
  (original_price - current_price) / original_price * 100 = 24 :=
by
  sorry

end percentage_decrease_l8_8968


namespace ratio_square_correct_l8_8232

noncomputable def ratio_square (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) : ℝ :=
  let k := a / b
  let x := k * k
  x

theorem ratio_square_correct (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) :
  ratio_square a b h = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end ratio_square_correct_l8_8232


namespace pennies_on_friday_l8_8900

-- Define the initial number of pennies and the function for doubling
def initial_pennies : Nat := 3
def double (n : Nat) : Nat := 2 * n

-- Prove the number of pennies on Friday
theorem pennies_on_friday : double (double (double (double initial_pennies))) = 48 := by
  sorry

end pennies_on_friday_l8_8900


namespace Roe_total_savings_l8_8644

-- Define savings amounts per period
def savings_Jan_to_Jul : Int := 7 * 10
def savings_Aug_to_Nov : Int := 4 * 15
def savings_Dec : Int := 20

-- Define total savings for the year
def total_savings : Int := savings_Jan_to_Jul + savings_Aug_to_Nov + savings_Dec

-- Prove that Roe's total savings for the year is $150
theorem Roe_total_savings : total_savings = 150 := by
  -- Proof goes here
  sorry

end Roe_total_savings_l8_8644


namespace more_time_running_than_skipping_l8_8885

def time_running : ℚ := 17 / 20
def time_skipping_rope : ℚ := 83 / 100

theorem more_time_running_than_skipping :
  time_running > time_skipping_rope :=
by
  -- sorry skips the proof
  sorry

end more_time_running_than_skipping_l8_8885


namespace largest_whole_number_value_l8_8802

theorem largest_whole_number_value (n : ℕ) : 
  (1 : ℚ) / 5 + (n : ℚ) / 8 < 9 / 5 → n ≤ 12 := 
sorry

end largest_whole_number_value_l8_8802


namespace tan_ratio_l8_8225

variable (a b : Real)

theorem tan_ratio (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 4) : 
  (Real.tan a) / (Real.tan b) = 7 / 3 := 
by 
  sorry

end tan_ratio_l8_8225


namespace workers_distribution_l8_8773

theorem workers_distribution (x y : ℕ) (h1 : x + y = 32) (h2 : 2 * 5 * x = 6 * y) : 
  (∃ x y : ℕ, x + y = 32 ∧ 2 * 5 * x = 6 * y) :=
sorry

end workers_distribution_l8_8773


namespace subtract_real_numbers_l8_8046

theorem subtract_real_numbers : 3.56 - 1.89 = 1.67 :=
by
  sorry

end subtract_real_numbers_l8_8046


namespace phone_price_is_correct_l8_8903

-- Definition of the conditions
def monthly_cost := 7
def months := 4
def total_cost := 30

-- Definition to be proven
def phone_price := total_cost - (monthly_cost * months)

theorem phone_price_is_correct : phone_price = 2 :=
by
  sorry

end phone_price_is_correct_l8_8903


namespace ratio_of_x_to_y_l8_8016

theorem ratio_of_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l8_8016


namespace quiz_answer_key_count_l8_8881

theorem quiz_answer_key_count :
  let tf_combinations := 6 -- Combinations of true-false questions
  let mc_combinations := 4 ^ 3 -- Combinations of multiple-choice questions
  tf_combinations * mc_combinations = 384 := by
  -- The values and conditions are directly taken from the problem statement.
  let tf_combinations := 6
  let mc_combinations := 4 ^ 3
  sorry

end quiz_answer_key_count_l8_8881


namespace BC_length_l8_8478

theorem BC_length (AD BC MN : ℝ) (h1 : AD = 2) (h2 : MN = 6) (h3 : MN = 0.5 * (AD + BC)) : BC = 10 :=
by
  sorry

end BC_length_l8_8478


namespace symmetric_line_equation_y_axis_l8_8772

theorem symmetric_line_equation_y_axis (x y : ℝ) : 
  (∃ m n : ℝ, (y = 3 * x + 1) ∧ (x + m = 0) ∧ (y = n) ∧ (n = 3 * m + 1)) → 
  y = -3 * x + 1 :=
by
  sorry

end symmetric_line_equation_y_axis_l8_8772


namespace book_pages_l8_8141

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
  pages_per_day = 8 → days = 12 → total_pages = pages_per_day * days → total_pages = 96 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end book_pages_l8_8141


namespace find_digits_l8_8984

theorem find_digits (a b : ℕ) (h1 : (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9)) :
  (∃ (c : ℕ), 10000 * a + 6790 + b = 72 * c) ↔ (a = 3 ∧ b = 2) :=
by
  sorry

end find_digits_l8_8984


namespace solve_for_b_l8_8451

theorem solve_for_b :
  (∀ (x y : ℝ), 4 * y - 3 * x + 2 = 0) →
  (∀ (x y : ℝ), 2 * y + b * x - 1 = 0) →
  (∃ b : ℝ, b = 8 / 3) := 
by
  sorry

end solve_for_b_l8_8451


namespace find_softball_players_l8_8237

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def total_players : ℕ := 59

theorem find_softball_players :
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  S = T - (C + H + F) :=
by
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  show S = T - (C + H + F)
  sorry

end find_softball_players_l8_8237


namespace trigonometric_identity_l8_8394

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 6 / 5 := 
sorry

end trigonometric_identity_l8_8394


namespace induction_divisibility_l8_8569

theorem induction_divisibility (k x y : ℕ) (h : k > 0) :
  (x^(2*k-1) + y^(2*k-1)) ∣ (x + y) → 
  (x^(2*k+1) + y^(2*k+1)) ∣ (x + y) :=
sorry

end induction_divisibility_l8_8569


namespace Harry_Terry_difference_l8_8278

theorem Harry_Terry_difference : 
(12 - (4 * 3)) - (12 - 4 * 3) = -24 := 
by
  sorry

end Harry_Terry_difference_l8_8278


namespace wall_height_to_breadth_ratio_l8_8781

theorem wall_height_to_breadth_ratio :
  ∀ (b : ℝ) (h : ℝ) (l : ℝ),
  b = 0.4 → h = n * b → l = 8 * h → l * b * h = 12.8 →
  n = 5 :=
by
  intros b h l hb hh hl hv
  sorry

end wall_height_to_breadth_ratio_l8_8781


namespace solve_x_l8_8605

theorem solve_x (x : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ) 
  (hA : A = (1, 3)) (hB : B = (2, 4))
  (ha : a = (2 * x - 1, x ^ 2 + 3 * x - 3))
  (hab : a = (B.1 - A.1, B.2 - A.2)) : x = 1 :=
by {
  sorry
}

end solve_x_l8_8605


namespace derivative_of_ln_2x_l8_8860

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

theorem derivative_of_ln_2x (x : ℝ) : deriv f x = 1 / x :=
  sorry

end derivative_of_ln_2x_l8_8860


namespace student_weight_l8_8287

theorem student_weight (S W : ℕ) (h1 : S - 5 = 2 * W) (h2 : S + W = 104) : S = 71 :=
by {
  sorry
}

end student_weight_l8_8287


namespace remainder_when_divided_by_x_minus_2_l8_8522

-- We define the polynomial f(x)
def f (x : ℝ) := x^4 - 6 * x^3 + 11 * x^2 + 20 * x - 8

-- We need to show that the remainder when f(x) is divided by (x - 2) is 44
theorem remainder_when_divided_by_x_minus_2 : f 2 = 44 :=
by {
  -- this is where the proof would go
  sorry
}

end remainder_when_divided_by_x_minus_2_l8_8522


namespace b_came_third_four_times_l8_8156

variable (a b c N : ℕ)

theorem b_came_third_four_times
    (a_pos : a > 0) 
    (b_pos : b > 0) 
    (c_pos : c > 0)
    (a_gt_b : a > b) 
    (b_gt_c : b > c) 
    (a_b_c_sum : a + b + c = 8)
    (score_A : 4 * a + b = 26) 
    (score_B : a + 4 * c = 11) 
    (score_C : 3 * b + 2 * c = 11) 
    (B_won_first_event : a + b + c = 8) : 
    4 * c = 4 := 
sorry

end b_came_third_four_times_l8_8156


namespace find_x_value_l8_8062

theorem find_x_value (x : ℤ)
    (h1 : (5 + 9) / 2 = 7)
    (h2 : (5 + x) / 2 = 10)
    (h3 : (x + 9) / 2 = 12) : 
    x = 15 := 
sorry

end find_x_value_l8_8062


namespace each_friend_should_contribute_equally_l8_8365

-- Define the total expenses and number of friends
def total_expenses : ℝ := 35 + 9 + 9 + 6 + 2
def number_of_friends : ℕ := 5

-- Define the expected contribution per friend
def expected_contribution : ℝ := 12.20

-- Theorem statement
theorem each_friend_should_contribute_equally :
  total_expenses / number_of_friends = expected_contribution :=
by
  sorry

end each_friend_should_contribute_equally_l8_8365


namespace pair_green_shirts_l8_8162

/-- In a regional math gathering, 83 students wore red shirts, and 97 students wore green shirts. 
The 180 students are grouped into 90 pairs. Exactly 35 of these pairs consist of students both 
wearing red shirts. Prove that the number of pairs consisting solely of students wearing green shirts is 42. -/
theorem pair_green_shirts (r g total pairs rr: ℕ) (h_r : r = 83) (h_g : g = 97) (h_total : total = 180) 
    (h_pairs : pairs = 90) (h_rr : rr = 35) : 
    (g - (r - rr * 2)) / 2 = 42 := 
by 
  /- The proof is omitted. -/
  sorry

end pair_green_shirts_l8_8162


namespace smallest_x_for_perfect_cube_l8_8537

theorem smallest_x_for_perfect_cube (x : ℕ) (M : ℤ) (hx : x > 0) (hM : ∃ M, 1680 * x = M^3) : x = 44100 :=
sorry

end smallest_x_for_perfect_cube_l8_8537


namespace area_of_triangle_is_right_angled_l8_8732

noncomputable def vector_a : ℝ × ℝ := (3, 4)
noncomputable def vector_b : ℝ × ℝ := (-4, 3)

theorem area_of_triangle_is_right_angled (h1 : vector_a = (3, 4)) (h2 : vector_b = (-4, 3)) : 
  let det := vector_a.1 * vector_b.2 - vector_a.2 * vector_b.1
  (1 / 2) * abs det = 12.5 :=
by
  sorry

end area_of_triangle_is_right_angled_l8_8732


namespace sqrt_x_minus_2_real_iff_x_ge_2_l8_8396

theorem sqrt_x_minus_2_real_iff_x_ge_2 (x : ℝ) : (∃ r : ℝ, r * r = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_iff_x_ge_2_l8_8396


namespace dog_food_packages_l8_8127

theorem dog_food_packages
  (packages_cat_food : Nat := 9)
  (cans_per_package_cat_food : Nat := 10)
  (cans_per_package_dog_food : Nat := 5)
  (more_cans_cat_food : Nat := 55)
  (total_cans_cat_food : Nat := packages_cat_food * cans_per_package_cat_food)
  (total_cans_dog_food : Nat := d * cans_per_package_dog_food)
  (h : total_cans_cat_food = total_cans_dog_food + more_cans_cat_food) :
  d = 7 :=
by
  sorry

end dog_food_packages_l8_8127


namespace glasses_displayed_is_correct_l8_8745

-- Definitions from the problem conditions
def tall_cupboard_capacity : Nat := 20
def wide_cupboard_capacity : Nat := 2 * tall_cupboard_capacity
def per_shelf_narrow_cupboard : Nat := 15 / 3
def usable_narrow_cupboard_capacity : Nat := 2 * per_shelf_narrow_cupboard

-- Theorem to prove that the total number of glasses displayed is 70
theorem glasses_displayed_is_correct :
  (tall_cupboard_capacity + wide_cupboard_capacity + usable_narrow_cupboard_capacity) = 70 :=
by
  sorry

end glasses_displayed_is_correct_l8_8745


namespace brandon_textbooks_weight_l8_8058

-- Define the weights of Jon's textbooks
def weight_jon_book1 := 2
def weight_jon_book2 := 8
def weight_jon_book3 := 5
def weight_jon_book4 := 9

-- Calculate the total weight of Jon's textbooks
def total_weight_jon := weight_jon_book1 + weight_jon_book2 + weight_jon_book3 + weight_jon_book4

-- Define the condition where Jon's textbooks weigh three times as much as Brandon's textbooks
def jon_to_brandon_ratio := 3

-- Define the weight of Brandon's textbooks
def weight_brandon := total_weight_jon / jon_to_brandon_ratio

-- The goal is to prove that the weight of Brandon's textbooks is 8 pounds.
theorem brandon_textbooks_weight : weight_brandon = 8 := by
  sorry

end brandon_textbooks_weight_l8_8058


namespace arithm_prog_diff_max_l8_8130

noncomputable def find_most_common_difference (a b c : Int) : Prop :=
  let d := a - b
  (b = a - d) ∧ (c = a - 2 * d) ∧
  (2 * a * 2 * a - 4 * 2 * a * c ≥ 0) ∧
  (2 * a * 2 * b - 4 * 2 * a * c ≥ 0) ∧
  (2 * b * 2 * b - 4 * 2 * b * c ≥ 0) ∧
  (2 * b * c - 4 * 2 * b * a ≥ 0) ∧
  (c * c - 4 * c * 2 * b ≥ 0) ∧
  ((2 * a * c - 4 * 2 * c * b) ≥ 0)

theorem arithm_prog_diff_max (a b c Dmax: Int) : 
  find_most_common_difference 4 (-1) (-6) ∧ Dmax = -5 :=
by 
  sorry

end arithm_prog_diff_max_l8_8130


namespace work_completion_time_l8_8209

theorem work_completion_time (days_B days_C days_all : ℝ) (h_B : days_B = 5) (h_C : days_C = 12) (h_all : days_all = 2.2222222222222223) : 
    (1 / ((days_all / 9) * 10) - 1 / days_B - 1 / days_C)⁻¹ = 60 / 37 := by 
  sorry

end work_completion_time_l8_8209


namespace multiple_of_interest_rate_l8_8795

theorem multiple_of_interest_rate (P r : ℝ) (m : ℝ) 
  (h1 : P * r^2 = 40) 
  (h2 : P * m^2 * r^2 = 360) : 
  m = 3 :=
by
  sorry

end multiple_of_interest_rate_l8_8795


namespace ca1_l8_8208

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end ca1_l8_8208


namespace sin_double_theta_l8_8435

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l8_8435


namespace baseball_games_per_month_l8_8166

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l8_8166


namespace integer_solutions_exist_l8_8865

theorem integer_solutions_exist (m n : ℤ) :
  ∃ (w x y z : ℤ), 
  (w + x + 2 * y + 2 * z = m) ∧ 
  (2 * w - 2 * x + y - z = n) := sorry

end integer_solutions_exist_l8_8865


namespace evaluate_polynomial_l8_8554

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x - 5

-- Define the condition: x is the positive root of the quadratic equation
def is_positive_root_of_quadratic (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 9 = 0

-- The main theorem stating the polynomial evaluates to 22 given the condition
theorem evaluate_polynomial {x : ℝ} (h : is_positive_root_of_quadratic x) : polynomial x = 22 := 
by 
  sorry

end evaluate_polynomial_l8_8554


namespace cucumbers_for_20_apples_l8_8506

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l8_8506


namespace original_number_l8_8315

theorem original_number (n : ℕ) (h : (n + 1) % 30 = 0) : n = 29 :=
by
  sorry

end original_number_l8_8315


namespace multiples_of_six_l8_8426

theorem multiples_of_six (a b : ℕ) (h₁ : a = 5) (h₂ : b = 127) :
  ∃ n : ℕ, n = 21 ∧ ∀ x : ℕ, (a < 6 * x ∧ 6 * x < b) ↔ (1 ≤ x ∧ x ≤ 21) :=
by
  sorry

end multiples_of_six_l8_8426


namespace quadratic_to_vertex_form_l8_8925

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 1)^2 + 2

-- State the equivalence we want to prove.
theorem quadratic_to_vertex_form :
  ∀ x : ℝ, quadratic_function x = vertex_form x :=
by
  intro x
  show quadratic_function x = vertex_form x
  sorry

end quadratic_to_vertex_form_l8_8925


namespace find_a_l8_8835

noncomputable def geometric_sequence_solution (a : ℝ) : Prop :=
  (a + 1) ^ 2 = (1 / (a - 1)) * (a ^ 2 - 1)

theorem find_a (a : ℝ) : geometric_sequence_solution a → a = 0 :=
by
  intro h
  sorry

end find_a_l8_8835


namespace max_min_x1_x2_squared_l8_8648

theorem max_min_x1_x2_squared (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - (k-2)*x1 + (k^2 + 3*k + 5) = 0)
  (h2 : x2^2 - (k-2)*x2 + (k^2 + 3*k + 5) = 0)
  (h3 : -4 ≤ k ∧ k ≤ -4/3) : 
  (∃ (k_max k_min : ℝ), 
    k = -4 → x1^2 + x2^2 = 18 ∧ k = -4/3 → x1^2 + x2^2 = 50/9) :=
sorry

end max_min_x1_x2_squared_l8_8648


namespace base_of_first_term_l8_8103

theorem base_of_first_term (e : ℕ) (b : ℝ) (h : e = 35) :
  b^e * (1/4)^18 = 1/(2 * 10^35) → b = 1/5 :=
by
  sorry

end base_of_first_term_l8_8103


namespace sqrt_squared_l8_8034

theorem sqrt_squared (n : ℕ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

example : (Real.sqrt 987654) ^ 2 = 987654 := 
  sqrt_squared 987654 (by norm_num)

end sqrt_squared_l8_8034


namespace find_x_of_series_eq_16_l8_8081

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x ^ n

theorem find_x_of_series_eq_16 (x : ℝ) (h : series_sum x = 16) : x = (4 - Real.sqrt 2) / 4 :=
by
  sorry

end find_x_of_series_eq_16_l8_8081


namespace value_of_m_l8_8338

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m_l8_8338


namespace Ryan_stickers_l8_8150

def Ryan_has_30_stickers (R S T : ℕ) : Prop :=
  S = 3 * R ∧ T = S + 20 ∧ R + S + T = 230 → R = 30

theorem Ryan_stickers : ∃ R S T : ℕ, Ryan_has_30_stickers R S T :=
sorry

end Ryan_stickers_l8_8150


namespace hector_gumballs_remaining_l8_8942

def gumballs_remaining (gumballs : ℕ) (given_todd : ℕ) (given_alisha : ℕ) (given_bobby : ℕ) : ℕ :=
  gumballs - (given_todd + given_alisha + given_bobby)

theorem hector_gumballs_remaining :
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  gumballs_remaining gumballs given_todd given_alisha given_bobby = 6 :=
by 
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  show gumballs_remaining gumballs given_todd given_alisha given_bobby = 6
  sorry

end hector_gumballs_remaining_l8_8942


namespace find_multiple_of_pages_l8_8485

-- Definitions based on conditions
def beatrix_pages : ℕ := 704
def cristobal_extra_pages : ℕ := 1423
def cristobal_pages (x : ℕ) : ℕ := x * beatrix_pages + 15

-- Proposition to prove the multiple x equals 2
theorem find_multiple_of_pages (x : ℕ) (h : cristobal_pages x = beatrix_pages + cristobal_extra_pages) : x = 2 :=
  sorry

end find_multiple_of_pages_l8_8485


namespace squirrel_rise_per_circuit_l8_8740

theorem squirrel_rise_per_circuit
  (h_post_height : ℕ := 12)
  (h_circumference : ℕ := 3)
  (h_travel_distance : ℕ := 9) :
  (h_post_height / (h_travel_distance / h_circumference) = 4) :=
  sorry

end squirrel_rise_per_circuit_l8_8740


namespace sphere_radius_ratio_l8_8735

theorem sphere_radius_ratio (R1 R2 : ℝ) (m n : ℝ) (hm : 1 < m) (hn : 1 < n) 
  (h_ratio1 : (2 * π * R1 * ((2 * R1) / (m + 1))) / (4 * π * R1 * R1) = 1 / (m + 1))
  (h_ratio2 : (2 * π * R2 * ((2 * R2) / (n + 1))) / (4 * π * R2 * R2) = 1 / (n + 1)): 
  R2 / R1 = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := 
by
  sorry

end sphere_radius_ratio_l8_8735


namespace friends_total_l8_8917

-- Define the conditions as constants
def can_go : Nat := 8
def can't_go : Nat := 7

-- Define the total number of friends and the correct answer
def total_friends : Nat := can_go + can't_go
def correct_answer : Nat := 15

-- Prove that the total number of friends is 15
theorem friends_total : total_friends = correct_answer := by
  -- We use the definitions and the conditions directly here
  sorry

end friends_total_l8_8917


namespace line_slope_l8_8545

theorem line_slope (t : ℝ) : 
  (∃ (t : ℝ), x = 1 + 2 * t ∧ y = 2 - 3 * t) → 
  (∃ (m : ℝ), m = -3 / 2) :=
sorry

end line_slope_l8_8545


namespace edward_spent_amount_l8_8261

-- Definitions based on the problem conditions
def initial_amount : ℕ := 18
def remaining_amount : ℕ := 2

-- The statement to prove: Edward spent $16
theorem edward_spent_amount : initial_amount - remaining_amount = 16 := by
  sorry

end edward_spent_amount_l8_8261


namespace annie_building_time_l8_8961

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l8_8961


namespace right_angled_triangle_not_axisymmetric_l8_8591

-- Define a type for geometric figures
inductive Figure
| Angle : Figure
| EquilateralTriangle : Figure
| LineSegment : Figure
| RightAngledTriangle : Figure

open Figure

-- Define a function to determine if a figure is axisymmetric
def is_axisymmetric: Figure -> Prop
| Angle => true
| EquilateralTriangle => true
| LineSegment => true
| RightAngledTriangle => false

-- Statement of the problem
theorem right_angled_triangle_not_axisymmetric : 
  is_axisymmetric RightAngledTriangle = false :=
by
  sorry

end right_angled_triangle_not_axisymmetric_l8_8591


namespace arithmetic_seq_a7_l8_8853

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 + a 5 = 12) : a 7 = 10 :=
by
  sorry

end arithmetic_seq_a7_l8_8853


namespace martian_right_angle_l8_8078

theorem martian_right_angle :
  ∀ (full_circle clerts_per_right_angle : ℕ),
  (full_circle = 600) →
  (clerts_per_right_angle = full_circle / 3) →
  clerts_per_right_angle = 200 :=
by
  intros full_circle clerts_per_right_angle h1 h2
  sorry

end martian_right_angle_l8_8078


namespace total_chocolate_bars_proof_l8_8409

def large_box_contains := 17
def first_10_boxes_contains := 10
def medium_boxes_per_small := 4
def chocolate_bars_per_medium := 26

def remaining_7_boxes := 7
def first_two_boxes := 2
def first_two_bars := 18
def next_three_boxes := 3
def next_three_bars := 22
def last_two_boxes := 2
def last_two_bars := 30

noncomputable def total_chocolate_bars_in_large_box : Nat :=
  let chocolate_in_first_10 := first_10_boxes_contains * medium_boxes_per_small * chocolate_bars_per_medium
  let chocolate_in_remaining_7 :=
    (first_two_boxes * first_two_bars) +
    (next_three_boxes * next_three_bars) +
    (last_two_boxes * last_two_bars)
  chocolate_in_first_10 + chocolate_in_remaining_7

theorem total_chocolate_bars_proof :
  total_chocolate_bars_in_large_box = 1202 :=
by
  -- Detailed calculation is skipped
  sorry

end total_chocolate_bars_proof_l8_8409


namespace largest_four_digit_mod_5_l8_8766

theorem largest_four_digit_mod_5 : ∃ (n : ℤ), n % 5 = 3 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℤ, m % 5 = 3 ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
sorry

end largest_four_digit_mod_5_l8_8766


namespace alice_height_after_growth_l8_8625

/-- Conditions: Bob and Alice were initially the same height. Bob has grown by 25%, Alice 
has grown by one third as many inches as Bob, and Bob is now 75 inches tall. --/
theorem alice_height_after_growth (initial_height : ℕ)
  (bob_growth_rate : ℚ)
  (alice_growth_ratio : ℚ)
  (bob_final_height : ℕ) :
  bob_growth_rate = 0.25 →
  alice_growth_ratio = 1 / 3 →
  bob_final_height = 75 →
  initial_height + (bob_final_height - initial_height) / 3 = 65 :=
by
  sorry

end alice_height_after_growth_l8_8625


namespace probability_of_both_chinese_books_l8_8345

def total_books := 5
def chinese_books := 3
def math_books := 2

theorem probability_of_both_chinese_books (select_books : ℕ) 
  (total_choices : ℕ) (favorable_choices : ℕ) :
  select_books = 2 →
  total_choices = (Nat.choose total_books select_books) →
  favorable_choices = (Nat.choose chinese_books select_books) →
  (favorable_choices : ℚ) / (total_choices : ℚ) = 3 / 10 := by
  intros h1 h2 h3
  sorry

end probability_of_both_chinese_books_l8_8345


namespace shortest_path_Dasha_Vasya_l8_8292

-- Definitions for the given distances
def dist_Asya_Galia : ℕ := 12
def dist_Galia_Borya : ℕ := 10
def dist_Asya_Borya : ℕ := 8
def dist_Dasha_Galia : ℕ := 15
def dist_Vasya_Galia : ℕ := 17

-- Definition for shortest distance by roads from Dasha to Vasya
def shortest_dist_Dasha_Vasya : ℕ := 18

-- Proof statement of the goal that shortest distance from Dasha to Vasya is 18 km
theorem shortest_path_Dasha_Vasya : 
  dist_Dasha_Galia + dist_Vasya_Galia - dist_Asya_Galia - dist_Galia_Borya = shortest_dist_Dasha_Vasya := by
  sorry

end shortest_path_Dasha_Vasya_l8_8292


namespace reciprocals_of_product_one_l8_8146

theorem reciprocals_of_product_one (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x :=
by 
  sorry

end reciprocals_of_product_one_l8_8146


namespace total_pencils_in_drawer_l8_8589

-- Definitions based on conditions from the problem
def initial_pencils : ℕ := 138
def pencils_by_Nancy : ℕ := 256
def pencils_by_Steven : ℕ := 97

-- The theorem proving the total number of pencils in the drawer
theorem total_pencils_in_drawer : initial_pencils + pencils_by_Nancy + pencils_by_Steven = 491 :=
by
  -- This statement is equivalent to the mathematical problem given
  sorry

end total_pencils_in_drawer_l8_8589


namespace casper_initial_candies_l8_8121

theorem casper_initial_candies : 
  ∃ x : ℕ, 
    (∃ y1 : ℕ, y1 = x / 2 - 3) ∧
    (∃ y2 : ℕ, y2 = y1 / 2 - 5) ∧
    (∃ y3 : ℕ, y3 = y2 / 2 - 2) ∧
    (y3 = 10) ∧
    x = 122 := 
sorry

end casper_initial_candies_l8_8121


namespace asymptotes_and_foci_of_hyperbola_l8_8770

def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

theorem asymptotes_and_foci_of_hyperbola :
  (∀ x y : ℝ, hyperbola x y → y = x * (3 / 4) ∨ y = x * -(3 / 4)) ∧
  (∃ x y : ℝ, (x, y) = (15, 0) ∨ (x, y) = (-15, 0)) :=
by {
  -- prove these conditions here
  sorry 
}

end asymptotes_and_foci_of_hyperbola_l8_8770


namespace function_odd_domain_of_f_range_of_f_l8_8752

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem function_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intro x
  trivial

theorem range_of_f : ∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  intro y
  sorry

end function_odd_domain_of_f_range_of_f_l8_8752


namespace extra_time_needed_l8_8255

variable (S : ℝ) (d : ℝ) (T T' : ℝ)

-- Original conditions
def original_speed_at_time_distance (S : ℝ) (T : ℝ) (d : ℝ) : Prop :=
  S * T = d

def decreased_speed (original_S : ℝ) : ℝ :=
  0.80 * original_S

def decreased_speed_time (T' : ℝ ) (decreased_S : ℝ) (d : ℝ) : Prop :=
  decreased_S * T' = d

theorem extra_time_needed
  (h1 : original_speed_at_time_distance S T d)
  (h2 : T = 40)
  (h3 : decreased_speed S = 0.80 * S)
  (h4 : decreased_speed_time T' (decreased_speed S) d) :
  T' - T = 10 :=
by
  sorry

end extra_time_needed_l8_8255


namespace find_angle_C_l8_8617

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C_l8_8617


namespace greater_number_l8_8919

theorem greater_number (x y : ℕ) (h1 : x * y = 2048) (h2 : x + y - (x - y) = 64) : x = 64 :=
by
  sorry

end greater_number_l8_8919


namespace closest_to_fraction_l8_8508

theorem closest_to_fraction (n d : ℝ) (h_n : n = 510) (h_d : d = 0.125) :
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 5000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 6000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 7000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 8000 :=
by
  sorry

end closest_to_fraction_l8_8508


namespace bess_throw_distance_l8_8007

-- Definitions based on the conditions
def bess_throws (x : ℝ) : ℝ := 4 * 2 * x
def holly_throws : ℝ := 5 * 8
def total_throws (x : ℝ) : ℝ := bess_throws x + holly_throws

-- Lean statement for the proof
theorem bess_throw_distance (x : ℝ) (h : total_throws x = 200) : x = 20 :=
by 
  sorry

end bess_throw_distance_l8_8007


namespace probability_of_draw_l8_8539

-- Define the probabilities as constants
def prob_not_lose_xiao_ming : ℚ := 3 / 4
def prob_lose_xiao_dong : ℚ := 1 / 2

-- State the theorem we want to prove
theorem probability_of_draw :
  prob_not_lose_xiao_ming - prob_lose_xiao_dong = 1 / 4 :=
by
  sorry

end probability_of_draw_l8_8539


namespace john_investment_years_l8_8929

theorem john_investment_years (P FVt : ℝ) (r1 r2 : ℝ) (n1 t : ℝ) :
  P = 2000 →
  r1 = 0.08 →
  r2 = 0.12 →
  n1 = 2 →
  FVt = 6620 →
  P * (1 + r1)^n1 * (1 + r2)^(t - n1) = FVt →
  t = 11 :=
by
  sorry

end john_investment_years_l8_8929


namespace g_value_at_5_l8_8044

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_5 (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x ^ 2) : g 5 = 1 := 
by 
  sorry

end g_value_at_5_l8_8044


namespace negation_of_universal_l8_8670

theorem negation_of_universal :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_l8_8670


namespace closest_perfect_square_to_315_l8_8920

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l8_8920


namespace petya_no_win_implies_draw_or_lost_l8_8723

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l8_8723


namespace sandcastle_height_difference_l8_8056

theorem sandcastle_height_difference :
  let Miki_height := 0.8333333333333334
  let Sister_height := 0.5
  Miki_height - Sister_height = 0.3333333333333334 :=
by
  sorry

end sandcastle_height_difference_l8_8056


namespace fiona_pairs_l8_8800

theorem fiona_pairs :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 15 → 45 ≤ (n * (n - 1) / 2) ∧ (n * (n - 1) / 2) ≤ 105 :=
by
  intro n
  intro h
  have h₁ : n ≥ 10 := h.left
  have h₂ : n ≤ 15 := h.right
  sorry

end fiona_pairs_l8_8800


namespace percentage_of_red_shirts_l8_8524

variable (total_students : ℕ) (blue_percent green_percent : ℕ) (other_students : ℕ)
  (H_total : total_students = 800)
  (H_blue : blue_percent = 45)
  (H_green : green_percent = 15)
  (H_other : other_students = 136)
  (H_blue_students : 0.45 * 800 = 360)
  (H_green_students : 0.15 * 800 = 120)
  (H_sum : 360 + 120 + 136 = 616)
  
theorem percentage_of_red_shirts :
  ((total_students - (360 + 120 + other_students)) / total_students) * 100 = 23 := 
by {
  sorry
}

end percentage_of_red_shirts_l8_8524


namespace sandwiches_difference_l8_8226

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l8_8226


namespace hypotenuse_length_l8_8254

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l8_8254


namespace number_of_video_cassettes_in_first_set_l8_8777

/-- Let A be the cost of an audio cassette, and V the cost of a video cassette.
  We are given that V = 300, and we have the following conditions:
  1. 7 * A + n * V = 1110,
  2. 5 * A + 4 * V = 1350.
  Prove that n = 3, the number of video cassettes in the first set -/
theorem number_of_video_cassettes_in_first_set 
    (A V n : ℕ) 
    (hV : V = 300)
    (h1 : 7 * A + n * V = 1110)
    (h2 : 5 * A + 4 * V = 1350) : 
    n = 3 := 
sorry

end number_of_video_cassettes_in_first_set_l8_8777


namespace total_area_of_combined_shape_l8_8603

theorem total_area_of_combined_shape
  (length_rectangle : ℝ) (width_rectangle : ℝ) (side_square : ℝ)
  (h_length : length_rectangle = 0.45)
  (h_width : width_rectangle = 0.25)
  (h_side : side_square = 0.15) :
  (length_rectangle * width_rectangle + side_square * side_square) = 0.135 := 
by 
  sorry

end total_area_of_combined_shape_l8_8603


namespace number_of_10_people_rows_l8_8599

theorem number_of_10_people_rows (x r : ℕ) (h1 : r = 54) (h2 : ∀ i : ℕ, i * 9 + x * 10 = 54) : x = 0 :=
by
  sorry

end number_of_10_people_rows_l8_8599


namespace candy_bar_sales_ratio_l8_8969

theorem candy_bar_sales_ratio
    (candy_bar_cost : ℕ := 2)
    (marvin_candy_sold : ℕ := 35)
    (tina_extra_earnings : ℕ := 140)
    (marvin_earnings := marvin_candy_sold * candy_bar_cost)
    (tina_earnings := marvin_earnings + tina_extra_earnings)
    (tina_candy_sold := tina_earnings / candy_bar_cost):
  tina_candy_sold / marvin_candy_sold = 3 :=
by
  sorry

end candy_bar_sales_ratio_l8_8969


namespace b_sequence_periodic_l8_8587

theorem b_sequence_periodic (b : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + Real.sqrt 3)
  (h_b2021 : b 2021 = 11 + Real.sqrt 3) :
  b 2048 = b 2 :=
sorry

end b_sequence_periodic_l8_8587


namespace average_sales_is_96_l8_8215

-- Definitions for the sales data
def january_sales : ℕ := 110
def february_sales : ℕ := 80
def march_sales : ℕ := 70
def april_sales : ℕ := 130
def may_sales : ℕ := 90

-- Number of months
def num_months : ℕ := 5

-- Total sales calculation
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + may_sales

-- Average sales per month calculation
def average_sales_per_month : ℕ := total_sales / num_months

-- Proposition to prove that the average sales per month is 96
theorem average_sales_is_96 : average_sales_per_month = 96 :=
by
  -- We use 'sorry' here to skip the proof, as the problem requires only the statement
  sorry

end average_sales_is_96_l8_8215


namespace min_value_at_2_l8_8722

noncomputable def f (x : ℝ) : ℝ := (2 / (x^2)) + Real.log x

theorem min_value_at_2 : (∀ x ∈ Set.Ioi (0 : ℝ), f x ≥ f 2) ∧ (∃ x ∈ Set.Ioi (0 : ℝ), f x = f 2) :=
by
  sorry

end min_value_at_2_l8_8722


namespace probability_after_6_passes_l8_8148

noncomputable section

-- We define people
inductive Person
| A | B | C

-- Probability that person A has the ball after n passes
def P : ℕ → Person → ℚ
| 0, Person.A => 1
| 0, _ => 0
| n+1, Person.A => (P n Person.B + P n Person.C) / 2
| n+1, Person.B => (P n Person.A + P n Person.C) / 2
| n+1, Person.C => (P n Person.A + P n Person.B) / 2

theorem probability_after_6_passes :
  P 6 Person.A = 11 / 32 := by
  sorry

end probability_after_6_passes_l8_8148


namespace surface_area_of_solid_block_l8_8944

theorem surface_area_of_solid_block :
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  top_bottom_area + front_back_area + left_right_area = 66 :=
by
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  sorry

end surface_area_of_solid_block_l8_8944


namespace is_quadratic_function_l8_8352

theorem is_quadratic_function (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + 3) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 / x) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = (x - 1)^2 - x^2) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 * x^2 - 1) ∧ (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c))) :=
by
  sorry

end is_quadratic_function_l8_8352


namespace shortest_chord_intercepted_by_line_l8_8542

theorem shortest_chord_intercepted_by_line (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 3 = 0 → y = k*x + 1 → (x - y + 1 = 0)) :=
sorry

end shortest_chord_intercepted_by_line_l8_8542


namespace solve_for_n_l8_8680

theorem solve_for_n (n x y : ℤ) (h : n * (x + y) + 17 = n * (-x + y) - 21) (hx : x = 1) : n = -19 :=
by
  sorry

end solve_for_n_l8_8680


namespace number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l8_8371

noncomputable def a (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem number_of_diagonals_pentagon : a 5 = 5 := sorry

theorem difference_hexagon_pentagon : a 6 - a 5 = 4 := sorry

theorem difference_successive_polygons (n : ℕ) (h : 4 ≤ n) : a (n + 1) - a n = n - 1 := sorry

end number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l8_8371


namespace evaluate_expression_l8_8614

variable (y : ℕ)

theorem evaluate_expression (h : y = 3) : 
    (y^(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) / y^(2 + 4 + 6 + 8 + 10 + 12)) = 3^58 :=
by
  -- Proof will be done here
  sorry

end evaluate_expression_l8_8614


namespace find_AB_l8_8350

theorem find_AB 
  (A B C Q N : Point)
  (h_AQ_QC : AQ / QC = 5 / 2)
  (h_CN_NB : CN / NB = 5 / 2)
  (h_QN : QN = 5 * Real.sqrt 2) : 
  AB = 7 * Real.sqrt 5 :=
sorry

end find_AB_l8_8350


namespace pool_min_cost_l8_8831

noncomputable def CostMinimization (x : ℝ) : ℝ :=
  150 * 1600 + 720 * (x + 1600 / x)

theorem pool_min_cost :
  ∃ (x : ℝ), x = 40 ∧ CostMinimization x = 297600 :=
by
  sorry

end pool_min_cost_l8_8831


namespace sum_ratio_l8_8320

noncomputable def S (n : ℕ) : ℝ := sorry -- placeholder definition

def arithmetic_geometric_sum : Prop :=
  S 3 = 2 ∧ S 6 = 18

theorem sum_ratio :
  arithmetic_geometric_sum → S 10 / S 5 = 33 :=
by
  intros h 
  sorry 

end sum_ratio_l8_8320


namespace packages_of_gum_l8_8882

-- Define the conditions
variables (P : Nat) -- Number of packages Robin has

-- State the theorem
theorem packages_of_gum (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end packages_of_gum_l8_8882


namespace initial_oranges_count_l8_8101

theorem initial_oranges_count 
  (O : ℕ)
  (h1 : 10 = O - 13) : 
  O = 23 := 
sorry

end initial_oranges_count_l8_8101


namespace right_triangles_with_specific_area_and_perimeter_l8_8245

theorem right_triangles_with_specific_area_and_perimeter :
  ∃ (count : ℕ),
    count = 7 ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ (a ≠ b) ∧ (a^2 + b^2 = c^2) ∧ (a * b / 2 = 5 * (a + b + c))) → 
      count = 7 :=
by
  sorry

end right_triangles_with_specific_area_and_perimeter_l8_8245


namespace floor_inequality_sqrt_l8_8412

theorem floor_inequality_sqrt (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (⌊ m * Real.sqrt 2 ⌋) * (⌊ n * Real.sqrt 7 ⌋) < (⌊ m * n * Real.sqrt 14 ⌋) := 
by
  sorry

end floor_inequality_sqrt_l8_8412


namespace six_letter_word_combinations_l8_8639

theorem six_letter_word_combinations : ∃ n : ℕ, n = 26 * 26 * 26 := 
sorry

end six_letter_word_combinations_l8_8639


namespace remainder_of_2_pow_23_mod_5_l8_8212

theorem remainder_of_2_pow_23_mod_5 
    (h1 : (2^2) % 5 = 4)
    (h2 : (2^3) % 5 = 3)
    (h3 : (2^4) % 5 = 1) :
    (2^23) % 5 = 3 :=
by
  sorry

end remainder_of_2_pow_23_mod_5_l8_8212


namespace nina_total_spending_l8_8314

-- Defining the quantities and prices of each category of items
def num_toys : Nat := 3
def price_per_toy : Nat := 10

def num_basketball_cards : Nat := 2
def price_per_card : Nat := 5

def num_shirts : Nat := 5
def price_per_shirt : Nat := 6

-- Calculating the total cost for each category
def cost_toys : Nat := num_toys * price_per_toy
def cost_cards : Nat := num_basketball_cards * price_per_card
def cost_shirts : Nat := num_shirts * price_per_shirt

-- Calculating the total amount spent
def total_cost : Nat := cost_toys + cost_cards + cost_shirts

-- The final theorem statement to verify the answer
theorem nina_total_spending : total_cost = 70 :=
by
  sorry

end nina_total_spending_l8_8314


namespace members_playing_both_sports_l8_8504

theorem members_playing_both_sports 
    (N : ℕ) (B : ℕ) (T : ℕ) (D : ℕ)
    (hN : N = 30) (hB : B = 18) (hT : T = 19) (hD : D = 2) :
    N - D = 28 ∧ B + T = 37 ∧ B + T - (N - D) = 9 :=
by
  sorry

end members_playing_both_sports_l8_8504


namespace frank_whack_a_mole_tickets_l8_8175

variable (W : ℕ)
variable (skee_ball_tickets : ℕ := 9)
variable (candy_cost : ℕ := 6)
variable (candies_bought : ℕ := 7)
variable (total_tickets : ℕ := W + skee_ball_tickets)
variable (required_tickets : ℕ := candy_cost * candies_bought)

theorem frank_whack_a_mole_tickets : W + skee_ball_tickets = required_tickets → W = 33 := by
  sorry

end frank_whack_a_mole_tickets_l8_8175


namespace lisa_punch_l8_8607

theorem lisa_punch (x : ℝ) (H : x = 0.125) :
  (0.3 + x) / (2 + x) = 0.20 :=
by
  sorry

end lisa_punch_l8_8607


namespace range_of_m_l8_8827

noncomputable def equation_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0

theorem range_of_m : ∀ m : ℝ, equation_has_real_roots m ↔ (-3 ≤ m ∧ m < 0) :=
by
  -- Proof omitted
  sorry

end range_of_m_l8_8827


namespace odd_function_a_eq_minus_1_l8_8042

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

theorem odd_function_a_eq_minus_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = -1 :=
by
  intros h
  sorry

end odd_function_a_eq_minus_1_l8_8042


namespace insert_zeros_between_digits_is_cube_l8_8898

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube_l8_8898


namespace fifth_inequality_proof_l8_8661

theorem fifth_inequality_proof : 
  1 + (1 / (2:ℝ)^2) + (1 / (3:ℝ)^2) + (1 / (4:ℝ)^2) + (1 / (5:ℝ)^2) + (1 / (6:ℝ)^2) < (11 / 6) :=
by {
  sorry
}

end fifth_inequality_proof_l8_8661


namespace percentage_difference_l8_8068

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end percentage_difference_l8_8068


namespace find_k_l8_8550

-- Define the conditions
variables (x y k : ℕ)
axiom part_sum : x + y = 36
axiom first_part : x = 19
axiom value_eq : 8 * x + k * y = 203

-- Prove that k is 3
theorem find_k : k = 3 :=
by
  -- Insert your proof here
  sorry

end find_k_l8_8550


namespace tan_add_pi_over_3_l8_8198

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l8_8198


namespace sum_of_consecutive_pages_with_product_15300_l8_8864

theorem sum_of_consecutive_pages_with_product_15300 : 
  ∃ n : ℕ, n * (n + 1) = 15300 ∧ n + (n + 1) = 247 :=
by
  sorry

end sum_of_consecutive_pages_with_product_15300_l8_8864


namespace triangle_perimeter_l8_8297

theorem triangle_perimeter (r A : ℝ) (h_r : r = 2.5) (h_A : A = 50) : 
  ∃ p : ℝ, p = 40 :=
by
  sorry

end triangle_perimeter_l8_8297


namespace red_marbles_eq_14_l8_8846

theorem red_marbles_eq_14 (total_marbles : ℕ) (yellow_marbles : ℕ) (R : ℕ) (B : ℕ)
  (h1 : total_marbles = 85)
  (h2 : yellow_marbles = 29)
  (h3 : B = 3 * R)
  (h4 : (total_marbles - yellow_marbles) = R + B) :
  R = 14 :=
by
  sorry

end red_marbles_eq_14_l8_8846


namespace triangle_perimeter_range_expression_l8_8809

-- Part 1: Prove the perimeter of △ABC
theorem triangle_perimeter (a b c : ℝ) (cosB : ℝ) (area : ℝ)
  (h1 : b^2 = a * c) (h2 : cosB = 3 / 5) (h3 : area = 2) :
  a + b + c = Real.sqrt 5 + Real.sqrt 21 :=
sorry

-- Part 2: Prove the range for the given expression
theorem range_expression (a b c : ℝ) (q : ℝ)
  (h1 : b = a * q) (h2 : c = a * q^2) :
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 :=
sorry

end triangle_perimeter_range_expression_l8_8809


namespace divisor_of_70th_number_l8_8500

-- Define the conditions
def s (d : ℕ) (n : ℕ) : ℕ := n * d + 5

-- Theorem stating the given problem
theorem divisor_of_70th_number (d : ℕ) (h : s d 70 = 557) : d = 8 :=
by
  -- The proof is to be filled in later. 
  -- Now, just create the structure.
  sorry

end divisor_of_70th_number_l8_8500


namespace inequal_f_i_sum_mn_ii_l8_8075

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 3 / 2 then -2 
  else if x > -5 / 2 then -x - 1 / 2 
  else 2

theorem inequal_f_i (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
sorry

theorem sum_mn_ii (m n : ℝ) (h1 : f m + f n = 4) (h2 : m < n) : m + n < -5 :=
sorry

end inequal_f_i_sum_mn_ii_l8_8075


namespace symmetric_circle_equation_l8_8583

theorem symmetric_circle_equation :
  ∀ (a b : ℝ), 
    (∀ (x y : ℝ), (x-2)^2 + (y+1)^2 = 4 → y = x + 1) → 
    (∃ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
  by
    sorry

end symmetric_circle_equation_l8_8583


namespace remaining_water_l8_8927

theorem remaining_water (initial_water : ℚ) (used_water : ℚ) (remaining_water : ℚ) 
  (h1 : initial_water = 3) (h2 : used_water = 5/4) : remaining_water = 7/4 :=
by
  -- The proof would go here, but we are skipping it as per the instructions.
  sorry

end remaining_water_l8_8927


namespace polynomial_factorization_l8_8967

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l8_8967


namespace club_men_count_l8_8833

theorem club_men_count (M W : ℕ) (h1 : M + W = 30) (h2 : M + (W / 3 : ℕ) = 20) : M = 15 := by
  -- proof omitted
  sorry

end club_men_count_l8_8833


namespace power_sum_zero_l8_8622

theorem power_sum_zero (n : ℕ) (h : 0 < n) : (-1:ℤ)^(2*n) + (-1:ℤ)^(2*n+1) = 0 := 
by 
  sorry

end power_sum_zero_l8_8622


namespace number_of_children_coming_to_show_l8_8612

theorem number_of_children_coming_to_show :
  ∀ (cost_adult cost_child : ℕ) (number_adults total_cost : ℕ),
  cost_adult = 12 →
  cost_child = 10 →
  number_adults = 3 →
  total_cost = 66 →
  ∃ (c : ℕ), 3 = c := by
    sorry

end number_of_children_coming_to_show_l8_8612


namespace factorial_expression_equiv_l8_8736

theorem factorial_expression_equiv :
  6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := 
sorry

end factorial_expression_equiv_l8_8736


namespace k_is_odd_l8_8758

theorem k_is_odd (m n k : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_k : 0 < k) (h : 3 * m * k = (m + 3)^n + 1) : Odd k :=
by {
  sorry
}

end k_is_odd_l8_8758


namespace problem1_problem2_l8_8116

theorem problem1 : -1 + (-6) - (-4) + 0 = -3 := by
  sorry

theorem problem2 : 24 * (-1 / 4) / (-3 / 2) = 4 := by
  sorry

end problem1_problem2_l8_8116


namespace real_root_quadratic_complex_eq_l8_8085

open Complex

theorem real_root_quadratic_complex_eq (a : ℝ) :
  ∀ x : ℝ, a * (1 + I) * x^2 + (1 + a^2 * I) * x + (a^2 + I) = 0 →
  a = -1 :=
by
  intros x h
  -- We need to prove this, but we're skipping the proof for now.
  sorry

end real_root_quadratic_complex_eq_l8_8085


namespace simplify_and_evaluate_l8_8479

variable (a : ℝ)
noncomputable def given_expression : ℝ :=
    (3 * a / (a^2 - 4)) * (1 - 2 / a) - (4 / (a + 2))

theorem simplify_and_evaluate (h : a = Real.sqrt 2 - 1) : 
  given_expression a = 1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l8_8479


namespace range_of_x_l8_8993

theorem range_of_x (x : ℝ) (p : x^2 - 2 * x - 3 < 0) (q : 1 / (x - 2) < 0) : -1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l8_8993


namespace not_perfect_square_l8_8505

theorem not_perfect_square (n : ℕ) (h₁ : 100 + 200 = 300) (h₂ : ¬(300 % 9 = 0)) : ¬(∃ m : ℕ, n = m * m) :=
by
  intros
  sorry

end not_perfect_square_l8_8505


namespace geometric_sequence_problem_l8_8411

noncomputable def a : ℕ → ℝ := sorry

theorem geometric_sequence_problem :
  a 4 = 4 →
  a 8 = 8 →
  a 12 = 16 :=
by
  intros h4 h8
  sorry

end geometric_sequence_problem_l8_8411


namespace geometric_sequence_sum_l8_8240

theorem geometric_sequence_sum :
  ∀ {a : ℕ → ℝ} (r : ℝ),
    (∀ n, a (n + 1) = r * a n) →
    a 1 + a 2 = 1 →
    a 3 + a 4 = 4 →
    a 5 + a 6 + a 7 + a 8 = 80 :=
by
  intros a r h_geom h_sum_1 h_sum_2
  sorry

end geometric_sequence_sum_l8_8240


namespace minimum_n_value_l8_8201

def satisfies_terms_condition (n : ℕ) : Prop :=
  (n + 1) * (n + 1) ≥ 2021

theorem minimum_n_value :
  ∃ n : ℕ, n > 0 ∧ satisfies_terms_condition n ∧ ∀ m : ℕ, m > 0 ∧ satisfies_terms_condition m → n ≤ m := by
  sorry

end minimum_n_value_l8_8201


namespace calc_product_l8_8892

def x : ℝ := 150.15
def y : ℝ := 12.01
def z : ℝ := 1500.15
def w : ℝ := 12

theorem calc_product :
  x * y * z * w = 32467532.8227 :=
by
  sorry

end calc_product_l8_8892


namespace ellipse_condition_l8_8274

theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 3) → ((m > 1 ∧ m < 3 ∧ m ≠ 2) ∨ (m = 2)) :=
by
  sorry

end ellipse_condition_l8_8274


namespace length_of_first_train_solution_l8_8125

noncomputable def length_of_first_train (speed1_kmph speed2_kmph : ℝ) (length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  let combined_length_m := relative_speed_mps * time_s
  combined_length_m - length2_m

theorem length_of_first_train_solution 
  (speed1_kmph : ℝ) 
  (speed2_kmph : ℝ) 
  (length2_m : ℝ) 
  (time_s : ℝ) 
  (h₁ : speed1_kmph = 42) 
  (h₂ : speed2_kmph = 30) 
  (h₃ : length2_m = 120) 
  (h₄ : time_s = 10.999120070394369) : 
  length_of_first_train speed1_kmph speed2_kmph length2_m time_s = 99.98 :=
by 
  sorry

end length_of_first_train_solution_l8_8125


namespace shortest_distance_point_on_circle_to_line_l8_8964

theorem shortest_distance_point_on_circle_to_line
  (P : ℝ × ℝ)
  (hP : (P.1 + 1)^2 + (P.2 - 2)^2 = 1) :
  ∃ (d : ℝ), d = 3 :=
sorry

end shortest_distance_point_on_circle_to_line_l8_8964


namespace additional_charge_per_segment_l8_8140

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l8_8140


namespace andrew_kept_correct_l8_8246

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l8_8246


namespace lowest_cost_per_ton_l8_8953

-- Define the conditions given in the problem statement
variable (x : ℝ) (y : ℝ)

-- Define the annual production range
def production_range (x : ℝ) : Prop := x ≥ 150 ∧ x ≤ 250

-- Define the relationship between total annual production cost and annual production
def production_cost_relation (x y : ℝ) : Prop := y = (x^2 / 10) - 30 * x + 4000

-- State the main theorem: the annual production when the cost per ton is the lowest is 200 tons
theorem lowest_cost_per_ton (x : ℝ) (y : ℝ) (h1 : production_range x) (h2 : production_cost_relation x y) : x = 200 :=
sorry

end lowest_cost_per_ton_l8_8953


namespace smallest_value_not_defined_l8_8729

noncomputable def smallest_undefined_x : ℝ :=
  let a := 6
  let b := -37
  let c := 5
  let discriminant := b * b - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 < x2 then x1 else x2

theorem smallest_value_not_defined :
  smallest_undefined_x = 0.1383 :=
by sorry

end smallest_value_not_defined_l8_8729


namespace napkin_coloring_l8_8991

structure Napkin where
  top : ℝ
  bottom : ℝ
  left : ℝ
  right : ℝ

def intersects_vertically (n1 n2 : Napkin) : Prop :=
  n1.left ≤ n2.right ∧ n2.left ≤ n1.right

def intersects_horizontally (n1 n2 : Napkin) : Prop :=
  n1.bottom ≤ n2.top ∧ n2.bottom ≤ n1.top

def can_be_crossed_by_line (n1 n2 : Napkin) : Prop :=
  intersects_vertically n1 n2 ∨ intersects_horizontally n1 n2

theorem napkin_coloring
  (blue_napkins green_napkins : List Napkin)
  (h_cross : ∀ (b : Napkin) (g : Napkin), 
    b ∈ blue_napkins → g ∈ green_napkins → can_be_crossed_by_line b g) :
  ∃ (color : String) (h1 h2 : ℝ) (v : ℝ), 
    (color = "blue" ∧ ∀ b ∈ blue_napkins, (b.bottom ≤ h1 ∧ h1 ≤ b.top) ∨ (b.bottom ≤ h2 ∧ h2 ≤ b.top) ∨ (b.left ≤ v ∧ v ≤ b.right)) ∨
    (color = "green" ∧ ∀ g ∈ green_napkins, (g.bottom ≤ h1 ∧ h1 ≤ g.top) ∨ (g.bottom ≤ h2 ∧ h2 ≤ g.top) ∨ (g.left ≤ v ∧ v ≤ g.right)) :=
sorry

end napkin_coloring_l8_8991


namespace inscribed_sphere_radius_l8_8604

noncomputable def radius_inscribed_sphere (S1 S2 S3 S4 V : ℝ) : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)

theorem inscribed_sphere_radius (S1 S2 S3 S4 V R : ℝ) :
  R = radius_inscribed_sphere S1 S2 S3 S4 V :=
by
  sorry

end inscribed_sphere_radius_l8_8604


namespace traci_flour_brought_l8_8388

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end traci_flour_brought_l8_8388


namespace total_area_of_region_l8_8368

variable (a b c d : ℝ)
variable (ha : a > 0) (hb : b > 0) (hd : d > 0)

theorem total_area_of_region : (a + b) * d + (1 / 2) * Real.pi * c ^ 2 = (a + b) * d + (1 / 2) * Real.pi * c ^ 2 := by
  sorry

end total_area_of_region_l8_8368


namespace mike_initial_games_l8_8581

theorem mike_initial_games (v w: ℕ)
  (h_non_working : v - w = 8)
  (h_earnings : 7 * w = 56)
  : v = 16 :=
by
  sorry

end mike_initial_games_l8_8581


namespace quadrilateral_condition_l8_8382

variable (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a + b + c + d = 2) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ a + b + c > 1 :=
by
  sorry

end quadrilateral_condition_l8_8382


namespace arithmetic_sequence_general_formula_and_geometric_condition_l8_8265

theorem arithmetic_sequence_general_formula_and_geometric_condition :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ} {k : ℕ}, 
    (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →
    a 1 = 9 →
    S 3 = 21 →
    a 5 * S k = a 8 ^ 2 →
    k = 5 :=
by 
  intros a S k hS ha1 hS3 hgeom
  sorry

end arithmetic_sequence_general_formula_and_geometric_condition_l8_8265


namespace train_travel_time_l8_8047

def travel_time (departure arrival : Nat) : Nat :=
  arrival - departure

theorem train_travel_time : travel_time 425 479 = 54 := by
  sorry

end train_travel_time_l8_8047


namespace increase_in_length_and_breadth_is_4_l8_8008

-- Define the variables for the original length and breadth of the room
variables (L B x : ℕ)

-- Define the original perimeter
def P_original : ℕ := 2 * (L + B)

-- Define the new perimeter after the increase
def P_new : ℕ := 2 * ((L + x) + (B + x))

-- Define the condition that the perimeter increases by 16 feet
axiom increase_perimeter : P_new L B x - P_original L B = 16

-- State the theorem that \(x = 4\)
theorem increase_in_length_and_breadth_is_4 : x = 4 :=
by
  -- Proof would be filled in here using the axioms and definitions
  sorry

end increase_in_length_and_breadth_is_4_l8_8008


namespace thickness_of_stack_l8_8756

theorem thickness_of_stack (books : ℕ) (avg_pages_per_book : ℕ) (pages_per_inch : ℕ) (total_pages : ℕ) (thick_in_inches : ℕ)
    (h1 : books = 6)
    (h2 : avg_pages_per_book = 160)
    (h3 : pages_per_inch = 80)
    (h4 : total_pages = books * avg_pages_per_book)
    (h5 : thick_in_inches = total_pages / pages_per_inch) :
    thick_in_inches = 12 :=
by {
    -- statement without proof
    sorry
}

end thickness_of_stack_l8_8756


namespace tangent_perpendicular_point_l8_8400

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

theorem tangent_perpendicular_point :
  ∃ x0, (f x0 = 1) ∧ (x0 = 0) :=
sorry

end tangent_perpendicular_point_l8_8400


namespace no_such_number_exists_l8_8918

-- Definitions for conditions
def base_5_digit_number (x : ℕ) : Prop := 
  ∀ n, 0 ≤ n ∧ n < 2023 → x / 5^n % 5 < 5

def odd_plus_one (n m : ℕ) : Prop :=
  (∀ k < 1012, (n / 5^(2*k) % 25 / 5 = m / 5^(2*k) % 25 / 5 + 1)) ∧
  (∀ k < 1011, (n / 5^(2*k+1) % 25 / 5 = m / 5^(2*k+1) % 25 / 5 - 1))

def has_two_prime_factors_that_differ_by_two (x : ℕ) : Prop :=
  ∃ u v, u * v = x ∧ Prime u ∧ Prime v ∧ v = u + 2

-- Combined conditions for the hypothesized number x
def hypothesized_number (x : ℕ) : Prop := 
  base_5_digit_number x ∧
  odd_plus_one x x ∧
  has_two_prime_factors_that_differ_by_two x

-- The proof statement that the hypothesized number cannot exist
theorem no_such_number_exists : ¬ ∃ x, hypothesized_number x :=
by
  sorry

end no_such_number_exists_l8_8918


namespace largest_fully_communicating_sets_eq_l8_8643

noncomputable def largest_fully_communicating_sets :=
  let total_sets := Nat.choose 99 4
  let non_communicating_sets_per_pod := Nat.choose 48 3
  let total_non_communicating_sets := 99 * non_communicating_sets_per_pod
  total_sets - total_non_communicating_sets

theorem largest_fully_communicating_sets_eq : largest_fully_communicating_sets = 2051652 := by
  sorry

end largest_fully_communicating_sets_eq_l8_8643


namespace range_of_a_l8_8774

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ Set.Icc (-1 : ℝ) 1, |4^x_0 - a * 2^x_0 + 1| ≤ 2^(x_0 + 1)) →
  0 ≤ a ∧ a ≤ (9/2) :=
by
  sorry

end range_of_a_l8_8774


namespace find_m_given_sampling_conditions_l8_8564

-- Definitions for population and sampling conditions
def population_divided_into_groups : Prop :=
  ∀ n : ℕ, n < 100 → ∃ k : ℕ, k < 10 ∧ n / 10 = k

def systematic_sampling_condition (m k : ℕ) : Prop :=
  k < 10 ∧ m < 10 ∧ (m + k - 1) % 10 < 10 ∧ (m + k - 11) % 10 < 10

-- Given conditions
def given_conditions (m k : ℕ) (n : ℕ) : Prop :=
  k = 6 ∧ n = 52 ∧ systematic_sampling_condition m k

-- The statement to prove
theorem find_m_given_sampling_conditions :
  ∃ m : ℕ, given_conditions m 6 52 ∧ m = 7 :=
by
  sorry

end find_m_given_sampling_conditions_l8_8564


namespace shade_half_grid_additional_squares_l8_8705

/-- A 4x5 grid consists of 20 squares, of which 3 are already shaded. 
Prove that the number of additional 1x1 squares needed to shade half the grid is 7. -/
theorem shade_half_grid_additional_squares (total_squares shaded_squares remaining_squares: ℕ) 
  (h1 : total_squares = 4 * 5)
  (h2 : shaded_squares = 3)
  (h3 : remaining_squares = total_squares / 2 - shaded_squares) :
  remaining_squares = 7 :=
by
  -- Proof not required.
  sorry

end shade_half_grid_additional_squares_l8_8705


namespace complex_multiplication_l8_8902

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_l8_8902


namespace javiers_household_legs_l8_8901

-- Definitions given the problem conditions
def humans : ℕ := 6
def human_legs : ℕ := 2

def dogs : ℕ := 2
def dog_legs : ℕ := 4

def cats : ℕ := 1
def cat_legs : ℕ := 4

def parrots : ℕ := 1
def parrot_legs : ℕ := 2

def lizards : ℕ := 1
def lizard_legs : ℕ := 4

def stool_legs : ℕ := 3
def table_legs : ℕ := 4
def cabinet_legs : ℕ := 6

-- Problem statement
theorem javiers_household_legs :
  (humans * human_legs) + (dogs * dog_legs) + (cats * cat_legs) + (parrots * parrot_legs) +
  (lizards * lizard_legs) + stool_legs + table_legs + cabinet_legs = 43 := by
  -- We leave the proof as an exercise for the reader
  sorry

end javiers_household_legs_l8_8901


namespace sin_double_angle_identity_l8_8057

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end sin_double_angle_identity_l8_8057


namespace lesser_number_of_sum_and_difference_l8_8787

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l8_8787


namespace ellipse_sum_l8_8143

theorem ellipse_sum (h k a b : ℤ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 7) (b_val : b = 4) : 
  h + k + a + b = 9 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l8_8143


namespace roads_going_outside_city_l8_8776

theorem roads_going_outside_city (n : ℕ)
  (h : ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 3 ∧
    (n + x1) % 2 = 0 ∧
    (n + x2) % 2 = 0 ∧
    (n + x3) % 2 = 0) :
  ∃ (x1 x2 x3 : ℕ), (x1 = 1) ∧ (x2 = 1) ∧ (x3 = 1) :=
by 
  sorry

end roads_going_outside_city_l8_8776


namespace sum_first_ten_terms_arithmetic_sequence_l8_8272

theorem sum_first_ten_terms_arithmetic_sequence (a₁ d : ℤ) (h₁ : a₁ = -3) (h₂ : d = 4) : 
  let a₁₀ := a₁ + (9 * d)
  let S := ((a₁ + a₁₀) / 2) * 10
  S = 150 :=
by
  subst h₁
  subst h₂
  let a₁₀ := -3 + (9 * 4)
  let S := ((-3 + a₁₀) / 2) * 10
  sorry

end sum_first_ten_terms_arithmetic_sequence_l8_8272


namespace smallest_positive_t_l8_8683

theorem smallest_positive_t (x_1 x_2 x_3 x_4 x_5 t : ℝ) :
  (x_1 + x_3 = 2 * t * x_2) →
  (x_2 + x_4 = 2 * t * x_3) →
  (x_3 + x_5 = 2 * t * x_4) →
  (0 ≤ x_1) →
  (0 ≤ x_2) →
  (0 ≤ x_3) →
  (0 ≤ x_4) →
  (0 ≤ x_5) →
  (x_1 ≠ 0 ∨ x_2 ≠ 0 ∨ x_3 ≠ 0 ∨ x_4 ≠ 0 ∨ x_5 ≠ 0) →
  t = 1 / Real.sqrt 2 → 
  ∃ t, (0 < t) ∧ (x_1 + x_3 = 2 * t * x_2) ∧ (x_2 + x_4 = 2 * t * x_3) ∧ (x_3 + x_5 = 2 * t * x_4)
:=
sorry

end smallest_positive_t_l8_8683


namespace triangle_is_either_isosceles_or_right_angled_l8_8070

theorem triangle_is_either_isosceles_or_right_angled
  (A B : Real)
  (a b c : Real)
  (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  : a = b ∨ a^2 + b^2 = c^2 :=
sorry

end triangle_is_either_isosceles_or_right_angled_l8_8070


namespace geometric_progression_exists_l8_8703

theorem geometric_progression_exists :
  ∃ (b1 b2 b3 b4: ℤ) (q: ℤ), 
    b2 = b1 * q ∧ 
    b3 = b1 * q^2 ∧ 
    b4 = b1 * q^3 ∧  
    b3 - b1 = 9 ∧ 
    b2 - b4 = 18 ∧ 
    b1 = 3 ∧ b2 = -6 ∧ b3 = 12 ∧ b4 = -24 :=
sorry

end geometric_progression_exists_l8_8703


namespace find_edge_lengths_sum_l8_8158

noncomputable def sum_edge_lengths (a d : ℝ) (volume surface_area : ℝ) : ℝ :=
  if (a - d) * a * (a + d) = volume ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = surface_area then
    4 * ((a - d) + a + (a + d))
  else
    0

theorem find_edge_lengths_sum:
  (∃ a d : ℝ, (a - d) * a * (a + d) = 512 ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = 352) →
  sum_edge_lengths (Real.sqrt 59) 1 512 352 = 12 * Real.sqrt 59 :=
by
  sorry

end find_edge_lengths_sum_l8_8158


namespace smaller_angle_parallelogram_l8_8660

theorem smaller_angle_parallelogram (x : ℕ) (h1 : ∀ a b : ℕ, a ≠ b ∧ a + b = 180) (h2 : ∃ y : ℕ, y = x + 70) : x = 55 :=
by
  sorry

end smaller_angle_parallelogram_l8_8660


namespace total_birdseed_amount_l8_8724

-- Define the birdseed amounts in the boxes
def box1_amount : ℕ := 250
def box2_amount : ℕ := 275
def box3_amount : ℕ := 225
def box4_amount : ℕ := 300
def box5_amount : ℕ := 275
def box6_amount : ℕ := 200
def box7_amount : ℕ := 150
def box8_amount : ℕ := 180

-- Define the weekly consumption of each bird
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def canary_consumption : ℕ := 25

-- Define a theorem to calculate the total birdseed that Leah has
theorem total_birdseed_amount : box1_amount + box2_amount + box3_amount + box4_amount + box5_amount + box6_amount + box7_amount + box8_amount = 1855 :=
by
  sorry

end total_birdseed_amount_l8_8724


namespace rectangle_perimeter_change_l8_8528

theorem rectangle_perimeter_change :
  ∀ (a b : ℝ), 
  (2 * (a + b) = 2 * (1.3 * a + 0.8 * b)) →
  ((2 * (0.8 * a + 1.95 * b) - 2 * (a + b)) / (2 * (a + b)) = 0.1) :=
by
  intros a b h
  sorry

end rectangle_perimeter_change_l8_8528


namespace car_b_speed_l8_8998

noncomputable def SpeedOfCarB (Speed_A Time_A Time_B d_ratio: ℝ) : ℝ :=
  let Distance_A := Speed_A * Time_A
  let Distance_B := Distance_A / d_ratio
  Distance_B / Time_B

theorem car_b_speed
  (Speed_A : ℝ) (Time_A : ℝ) (Time_B : ℝ) (d_ratio : ℝ)
  (h1 : Speed_A = 70) (h2 : Time_A = 10) (h3 : Time_B = 10) (h4 : d_ratio = 2) :
  SpeedOfCarB Speed_A Time_A Time_B d_ratio = 35 :=
by
  sorry

end car_b_speed_l8_8998


namespace remainder_b100_mod_81_l8_8996

def b (n : ℕ) := 7^n + 9^n

theorem remainder_b100_mod_81 : (b 100) % 81 = 38 := by
  sorry

end remainder_b100_mod_81_l8_8996


namespace triangle_area_l8_8931

theorem triangle_area (a b c : ℝ) (ha : a = 6) (hb : b = 5) (hc : c = 5) (isosceles : a = 2 * b) :
  let s := (a + b + c) / 2
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt
  area = 12 :=
by
  sorry

end triangle_area_l8_8931


namespace exercise_books_purchasing_methods_l8_8731

theorem exercise_books_purchasing_methods :
  ∃ (ways : ℕ), ways = 5 ∧
  (∃ (x y z : ℕ), 2 * x + 5 * y + 11 * z = 40 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) ∧
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ : ℕ),
    2 * x₁ + 5 * y₁ + 11 * z₂ = 40 ∧ x₁ ≥ 1 ∧ y₁ ≥ 1 ∧ z₁ ≥ 1 →
    2 * x₂ + 5 * y₂ + 11 * z₂ = 40 ∧ x₂ ≥ 1 ∧ y₂ ≥ 1 ∧ z₂ ≥ 1 →
    (x₁, y₁, z₁) = (x₂, y₂, z₂)) := sorry

end exercise_books_purchasing_methods_l8_8731


namespace max_number_of_cubes_l8_8463

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end max_number_of_cubes_l8_8463


namespace bags_sold_in_first_week_l8_8708

def total_bags_sold : ℕ := 100
def bags_sold_week1 (X : ℕ) : ℕ := X
def bags_sold_week2 (X : ℕ) : ℕ := 3 * X
def bags_sold_week3_4 : ℕ := 40

theorem bags_sold_in_first_week (X : ℕ) (h : total_bags_sold = bags_sold_week1 X + bags_sold_week2 X + bags_sold_week3_4) : X = 15 :=
by
  sorry

end bags_sold_in_first_week_l8_8708


namespace right_isosceles_areas_l8_8521

theorem right_isosceles_areas (A B C : ℝ) (hA : A = 1 / 2 * 5 * 5) (hB : B = 1 / 2 * 12 * 12) (hC : C = 1 / 2 * 13 * 13) :
  A + B = C :=
by
  sorry

end right_isosceles_areas_l8_8521


namespace production_equipment_B_l8_8751

theorem production_equipment_B :
  ∃ (X Y : ℕ), X + Y = 4800 ∧ (50 / 80 = 5 / 8) ∧ (X / 4800 = 5 / 8) ∧ Y = 1800 :=
by
  sorry

end production_equipment_B_l8_8751


namespace find_y_l8_8609

theorem find_y (y : ℚ) (h : 6 * y + 3 * y + 4 * y + 2 * y + 1 * y + 5 * y = 360) : y = 120 / 7 := 
sorry

end find_y_l8_8609


namespace kids_on_soccer_field_l8_8000

def original_kids : ℕ := 14
def joined_kids : ℕ := 22
def total_kids : ℕ := 36

theorem kids_on_soccer_field : (original_kids + joined_kids) = total_kids :=
by 
  sorry

end kids_on_soccer_field_l8_8000


namespace find_k_plus_a_l8_8041

theorem find_k_plus_a (k a : ℤ) (h1 : k > a) (h2 : a > 0) 
(h3 : 2 * (Int.natAbs (a - k)) * (Int.natAbs (a + k)) = 32) : k + a = 8 :=
by
  sorry

end find_k_plus_a_l8_8041


namespace arithmetic_mean_midpoint_l8_8341

theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m = a + (b - a) / 2 :=
by
  sorry

end arithmetic_mean_midpoint_l8_8341


namespace intersection_range_l8_8718

theorem intersection_range (k : ℝ) :
  (∃ x y : ℝ, y = k * x + k + 2 ∧ y = -2 * x + 4 ∧ x > 0 ∧ y > 0) ↔ -2/3 < k ∧ k < 2 :=
by
  sorry

end intersection_range_l8_8718


namespace number_of_dimes_l8_8606

theorem number_of_dimes (d q : ℕ) (h₁ : 10 * d + 25 * q = 580) (h₂ : d = q + 10) : d = 23 := 
by 
  sorry

end number_of_dimes_l8_8606


namespace codecracker_number_of_codes_l8_8221

theorem codecracker_number_of_codes : ∃ n : ℕ, n = 6 * 5^4 := by
  sorry

end codecracker_number_of_codes_l8_8221


namespace binary_div_four_remainder_l8_8084

theorem binary_div_four_remainder (n : ℕ) (h : n = 0b111001001101) : n % 4 = 1 := 
sorry

end binary_div_four_remainder_l8_8084


namespace find_second_number_l8_8448

theorem find_second_number (x y z : ℚ) (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 240 / 7 := by
  sorry

end find_second_number_l8_8448


namespace number_of_solutions_eq_one_l8_8629

theorem number_of_solutions_eq_one :
  (∃! y : ℝ, (y ≠ 0) ∧ (y ≠ 3) ∧ ((3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1)) :=
  sorry

end number_of_solutions_eq_one_l8_8629


namespace students_taking_one_language_l8_8511

-- Definitions based on the conditions
def french_class_students : ℕ := 21
def spanish_class_students : ℕ := 21
def both_languages_students : ℕ := 6
def total_students : ℕ := french_class_students + spanish_class_students - both_languages_students

-- The theorem we want to prove
theorem students_taking_one_language :
    total_students = 36 :=
by
  -- Add the proof here
  sorry

end students_taking_one_language_l8_8511


namespace max_friday_more_than_wednesday_l8_8571

-- Definitions and conditions
def played_hours_wednesday : ℕ := 2
def played_hours_thursday : ℕ := 2
def played_average_hours : ℕ := 3
def played_days : ℕ := 3

-- Total hours over three days
def total_hours : ℕ := played_average_hours * played_days

-- Hours played on Friday
def played_hours_wednesday_thursday : ℕ := played_hours_wednesday + played_hours_thursday

def played_hours_friday : ℕ := total_hours - played_hours_wednesday_thursday

-- Proof problem statement
theorem max_friday_more_than_wednesday : 
  played_hours_friday - played_hours_wednesday = 3 := 
sorry

end max_friday_more_than_wednesday_l8_8571


namespace min_buses_l8_8356

theorem min_buses (n : ℕ) : (47 * n >= 625) → (n = 14) :=
by {
  -- Proof is omitted since the problem only asks for the Lean statement, not the solution steps.
  sorry
}

end min_buses_l8_8356


namespace percentage_of_seniors_is_90_l8_8069

-- Definitions of the given conditions
def total_students : ℕ := 120
def students_in_statistics : ℕ := total_students / 2
def seniors_in_statistics : ℕ := 54

-- Statement to prove
theorem percentage_of_seniors_is_90 : 
  ( seniors_in_statistics / students_in_statistics : ℚ ) * 100 = 90 := 
by
  sorry  -- Proof will be provided here.

end percentage_of_seniors_is_90_l8_8069


namespace fly_flies_more_than_10_meters_l8_8417

theorem fly_flies_more_than_10_meters :
  ∃ (fly_path_length : ℝ), 
  (∃ (c : ℝ) (a b : ℝ), c = 5 ∧ a^2 + b^2 = c^2) →
  (fly_path_length > 10) := 
by
  sorry

end fly_flies_more_than_10_meters_l8_8417


namespace smallest_three_digit_times_largest_single_digit_l8_8560

theorem smallest_three_digit_times_largest_single_digit :
  let x := 100
  let y := 9
  ∃ z : ℕ, z = x * y ∧ 100 ≤ z ∧ z < 1000 :=
by
  let x := 100
  let y := 9
  use x * y
  sorry

end smallest_three_digit_times_largest_single_digit_l8_8560


namespace total_surface_area_of_modified_cube_l8_8092

-- Define the side length of the original cube
def side_length_cube := 3

-- Define the side length of the holes
def side_length_hole := 1

-- Define the condition of the surface area calculation
def total_surface_area_including_internal (side_length_cube side_length_hole : ℕ) : ℕ :=
  let original_surface_area := 6 * (side_length_cube * side_length_cube)
  let reduction_area := 6 * (side_length_hole * side_length_hole)
  let remaining_surface_area := original_surface_area - reduction_area
  let interior_surface_area := 6 * (4 * side_length_hole * side_length_cube)
  remaining_surface_area + interior_surface_area

-- Statement for the proof
theorem total_surface_area_of_modified_cube : total_surface_area_including_internal 3 1 = 72 :=
by
  -- This is the statement; the proof is omitted as "sorry"
  sorry

end total_surface_area_of_modified_cube_l8_8092


namespace greatest_possible_value_l8_8470

theorem greatest_possible_value (x : ℝ) : 
  (∃ (k : ℝ), k = (5 * x - 25) / (4 * x - 5) ∧ k^2 + k = 20) → x ≤ 2 := 
sorry

end greatest_possible_value_l8_8470


namespace cement_tesss_street_l8_8095

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street_l8_8095


namespace molecular_weight_CaSO4_2H2O_l8_8676

def Ca := 40.08
def S := 32.07
def O := 16.00
def H := 1.008

def Ca_weight := 1 * Ca
def S_weight := 1 * S
def O_in_sulfate_weight := 4 * O
def O_in_water_weight := 4 * O
def H_in_water_weight := 4 * H

def total_weight := Ca_weight + S_weight + O_in_sulfate_weight + O_in_water_weight + H_in_water_weight

theorem molecular_weight_CaSO4_2H2O : total_weight = 204.182 := 
by {
  sorry
}

end molecular_weight_CaSO4_2H2O_l8_8676


namespace how_many_raisins_did_bryce_receive_l8_8890

def raisins_problem : Prop :=
  ∃ (B C : ℕ), B = C + 8 ∧ C = B / 3 ∧ B + C = 44 ∧ B = 33

theorem how_many_raisins_did_bryce_receive : raisins_problem :=
sorry

end how_many_raisins_did_bryce_receive_l8_8890


namespace trig_log_exp_identity_l8_8063

theorem trig_log_exp_identity : 
  (Real.sin (330 * Real.pi / 180) + 
   (Real.sqrt 2 - 1)^0 + 
   3^(Real.log 2 / Real.log 3)) = 5 / 2 :=
by
  -- Proof omitted
  sorry

end trig_log_exp_identity_l8_8063


namespace Evan_dog_weight_l8_8195

-- Define the weights of the dogs as variables
variables (E I : ℕ)

-- Conditions given in the problem
def Evan_dog_weight_wrt_Ivan (I : ℕ) : ℕ := 7 * I
def dogs_total_weight (E I : ℕ) : Prop := E + I = 72

-- Correct answer we need to prove
theorem Evan_dog_weight (h1 : Evan_dog_weight_wrt_Ivan I = E)
                          (h2 : dogs_total_weight E I)
                          (h3 : I = 9) : E = 63 :=
by
  sorry

end Evan_dog_weight_l8_8195


namespace projection_is_negative_sqrt_10_l8_8880

noncomputable def projection_of_AB_in_direction_of_AC : ℝ :=
  let A := (1, 1)
  let B := (-3, 3)
  let C := (4, 2)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2)
  dot_product / magnitude_AC

theorem projection_is_negative_sqrt_10 :
  projection_of_AB_in_direction_of_AC = -Real.sqrt 10 :=
by
  sorry

end projection_is_negative_sqrt_10_l8_8880


namespace train_length_calculation_l8_8197

noncomputable def length_of_train 
  (time : ℝ) (speed_train : ℝ) (speed_man : ℝ) : ℝ :=
  let speed_relative := speed_train - speed_man
  let speed_relative_mps := speed_relative * (5 / 18)
  speed_relative_mps * time

theorem train_length_calculation :
  length_of_train 29.997600191984642 63 3 = 1666.67 := 
by
  sorry

end train_length_calculation_l8_8197


namespace inequality_proof_l8_8072

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  ¬ (1 / (1 + x + x * y) > 1 / 3 ∧ 
     y / (1 + y + y * z) > 1 / 3 ∧
     (x * z) / (1 + z + x * z) > 1 / 3) :=
by
  sorry

end inequality_proof_l8_8072


namespace container_weight_l8_8867

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l8_8867


namespace sin_alpha_trig_expression_l8_8171

theorem sin_alpha {α : ℝ} (hα : ∃ P : ℝ × ℝ, P = (4/5, -3/5)) :
  Real.sin α = -3/5 :=
sorry

theorem trig_expression {α : ℝ} 
  (hα : Real.sin α = -3/5) : 
  (Real.sin (π / 2 - α) / Real.sin (α + π)) - 
  (Real.tan (α - π) / Real.cos (3 * π - α)) = 19 / 48 :=
sorry

end sin_alpha_trig_expression_l8_8171


namespace find_number_l8_8966

theorem find_number (x : ℝ) :
  0.15 * x = 0.25 * 16 + 2 → x = 40 :=
by
  -- skipping the proof steps
  sorry

end find_number_l8_8966


namespace trig_identity_l8_8856

theorem trig_identity
  (x : ℝ)
  (h : Real.tan (π / 4 + x) = 2014) :
  1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 :=
by
  sorry

end trig_identity_l8_8856


namespace sum_of_reciprocals_of_shifted_roots_l8_8342

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) (h : ∀ x, x^3 - x + 2 = 0 → x = a ∨ x = b ∨ x = c) :
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l8_8342


namespace find_m_if_root_zero_l8_8313

theorem find_m_if_root_zero (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + x + (m^2 - 1) = 0) → m = -1 :=
by
  intro h
  -- Term after this point not necessary, hence a placeholder
  sorry

end find_m_if_root_zero_l8_8313


namespace find_n_l8_8695

-- We need a definition for permutations counting A_n^2 = n(n-1)
def permutations_squared (n : ℕ) : ℕ := n * (n - 1)

theorem find_n (n : ℕ) (h : permutations_squared n = 56) : n = 8 :=
by {
  sorry -- proof omitted as instructed
}

end find_n_l8_8695


namespace inequality_transpose_l8_8999

variable (a b : ℝ)

theorem inequality_transpose (h : a < b) (hab : b < 0) : (1 / a) > (1 / b) := by
  sorry

end inequality_transpose_l8_8999


namespace alpha_tan_beta_gt_beta_tan_alpha_l8_8191

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) 
: α * Real.tan β > β * Real.tan α := 
sorry

end alpha_tan_beta_gt_beta_tan_alpha_l8_8191


namespace operation_three_six_l8_8559

theorem operation_three_six : (3 * 3 * 6) / (3 + 6) = 6 :=
by
  calc (3 * 3 * 6) / (3 + 6) = 6 := sorry

end operation_three_six_l8_8559


namespace trail_length_is_20_km_l8_8700

-- Define the conditions and the question
def length_of_trail (L : ℝ) (hiked_percentage remaining_distance : ℝ) : Prop :=
  hiked_percentage = 0.60 ∧ remaining_distance = 8 ∧ 0.40 * L = remaining_distance

-- The statement: given the conditions, prove that length of trail is 20 km
theorem trail_length_is_20_km : ∃ L : ℝ, length_of_trail L 0.60 8 ∧ L = 20 := by
  -- Proof goes here
  sorry

end trail_length_is_20_km_l8_8700


namespace positive_integer_in_base_proof_l8_8118

noncomputable def base_conversion_problem (A B : ℕ) (n : ℕ) : Prop :=
  n = 9 * A + B ∧ n = 8 * B + A ∧ A < 9 ∧ B < 8 ∧ A ≠ 0 ∧ B ≠ 0

theorem positive_integer_in_base_proof (A B n : ℕ) (h : base_conversion_problem A B n) : n = 0 :=
sorry

end positive_integer_in_base_proof_l8_8118


namespace transformation_is_rotation_l8_8135

-- Define the 90 degree rotation matrix
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- Define the transformation matrix to be proven
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- The theorem that proves they are equivalent
theorem transformation_is_rotation :
  transformation_matrix = rotation_matrix :=
by
  sorry

end transformation_is_rotation_l8_8135


namespace total_weekly_reading_time_l8_8300

def morning_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def morning_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

def evening_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def evening_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

theorem total_weekly_reading_time :
  let morning_minutes := 30
  let evening_minutes := 60
  let weekdays := 5
  let weekend_days := 2
  morning_reading_weekdays morning_minutes weekdays +
  morning_reading_weekends morning_minutes +
  evening_reading_weekdays evening_minutes weekdays +
  evening_reading_weekends evening_minutes = 810 :=
by
  sorry

end total_weekly_reading_time_l8_8300


namespace flour_quantity_l8_8077

-- Define the recipe ratio of eggs to flour
def recipe_ratio : ℚ := 3 / 2

-- Define the number of eggs needed
def eggs_needed := 9

-- Prove that the number of cups of flour needed is 6
theorem flour_quantity (r : ℚ) (n : ℕ) (F : ℕ) 
  (hr : r = 3 / 2) (hn : n = 9) : F = 6 :=
by
  sorry

end flour_quantity_l8_8077


namespace common_ratio_l8_8139

theorem common_ratio
  (a b : ℝ)
  (h_arith : 2 * a = 1 + b)
  (h_geom : (a + 2) ^ 2 = 3 * (b + 5))
  (h_non_zero_a : a + 2 ≠ 0)
  (h_non_zero_b : b + 5 ≠ 0) :
  (a = 4 ∧ b = 7) ∧ (b + 5) / (a + 2) = 2 :=
by {
  sorry
}

end common_ratio_l8_8139


namespace total_number_of_items_l8_8202

-- Define the conditions as equations in Lean
def model_cars_price := 5
def model_trains_price := 8
def total_amount := 31

-- Initialize the variable definitions for number of cars and trains
variables (c t : ℕ)

-- The proof problem: Show that given the equation, the sum of cars and trains is 5
theorem total_number_of_items : (model_cars_price * c + model_trains_price * t = total_amount) → (c + t = 5) := by
  -- Proof steps would go here
  sorry

end total_number_of_items_l8_8202


namespace problem_statement_l8_8957

def f (x : ℤ) : ℤ := 3*x + 4
def g (x : ℤ) : ℤ := 4*x - 3

theorem problem_statement : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 :=
by
  sorry

end problem_statement_l8_8957


namespace min_value_m_plus_2n_exists_min_value_l8_8974

variable (n : ℝ) -- Declare n as a real number.

-- Define m in terms of n
def m (n : ℝ) : ℝ := n^2

-- State and prove that the minimum value of m + 2n is -1
theorem min_value_m_plus_2n : (m n + 2 * n) ≥ -1 :=
by sorry

-- Show there exists an n such that m + 2n = -1
theorem exists_min_value : ∃ n : ℝ, m n + 2 * n = -1 :=
by sorry

end min_value_m_plus_2n_exists_min_value_l8_8974


namespace geometric_sequence_sufficient_condition_l8_8822

theorem geometric_sequence_sufficient_condition 
  (a_1 : ℝ) (q : ℝ) (h_a1 : a_1 < 0) (h_q : 0 < q ∧ q < 1) :
  ∀ n : ℕ, n > 0 -> a_1 * q^(n-1) < a_1 * q^n :=
sorry

end geometric_sequence_sufficient_condition_l8_8822


namespace hyperbola_asymptotes_l8_8983

open Real

noncomputable def hyperbola (x y m : ℝ) : Prop := (x^2 / 9) - (y^2 / m) = 1

noncomputable def on_line (x y : ℝ) : Prop := x + y = 5

theorem hyperbola_asymptotes (m : ℝ) (hm : 9 + m = 25) :
    (∃ x y : ℝ, hyperbola x y m ∧ on_line x y) →
    (∀ x : ℝ, on_line x ((4 / 3) * x) ∧ on_line x (-(4 / 3) * x)) :=
by
  sorry

end hyperbola_asymptotes_l8_8983


namespace area_triangle_l8_8852

theorem area_triangle (A B C: ℝ) (AB AC : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : AB * AC * Real.cos A = 6) :
  (1 / 2) * AB * AC * Real.sin A = 4 :=
by
  sorry

end area_triangle_l8_8852


namespace tan_alpha_value_l8_8946

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l8_8946


namespace larger_of_two_numbers_l8_8792

theorem larger_of_two_numbers (H : Nat := 15) (f1 : Nat := 11) (f2 : Nat := 15) :
  let lcm := H * f1 * f2;
  ∃ (A B : Nat), A = H * f1 ∧ B = H * f2 ∧ A ≤ B := by
  sorry

end larger_of_two_numbers_l8_8792


namespace smallest_d_l8_8088

theorem smallest_d (d : ℝ) : 
  (∃ d, d > 0 ∧ (4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2))) → d = 2 :=
sorry

end smallest_d_l8_8088


namespace hyperbola_asymptotes_l8_8623

theorem hyperbola_asymptotes :
  ∀ {x y : ℝ},
    (x^2 / 9 - y^2 / 16 = 1) →
    (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end hyperbola_asymptotes_l8_8623


namespace trapezoid_cd_length_l8_8742

noncomputable def proof_cd_length (AD BC CD : ℝ) (BD : ℝ) (angle_DBA angle_BDC : ℝ) (ratio_BC_AD : ℝ) : Prop :=
  AD > 0 ∧ BC > 0 ∧
  BD = 1 ∧
  angle_DBA = 23 ∧
  angle_BDC = 46 ∧
  ratio_BC_AD = 9 / 5 ∧
  AD / BC = 5 / 9 ∧
  CD = 4 / 5

theorem trapezoid_cd_length
  (AD BC CD : ℝ)
  (BD : ℝ := 1)
  (angle_DBA : ℝ := 23)
  (angle_BDC : ℝ := 46)
  (ratio_BC_AD : ℝ := 9 / 5)
  (h_conditions : proof_cd_length AD BC CD BD angle_DBA angle_BDC ratio_BC_AD) : CD = 4 / 5 :=
sorry

end trapezoid_cd_length_l8_8742


namespace min_gb_for_plan_y_to_be_cheaper_l8_8326

theorem min_gb_for_plan_y_to_be_cheaper (g : ℕ) : 20 * g > 3000 + 10 * g → g ≥ 301 := by
  sorry

end min_gb_for_plan_y_to_be_cheaper_l8_8326


namespace problem1_problem2_l8_8854

-- Problem 1: Prove that the solution to f(x) <= 0 for a = -2 is [1, +∞)
theorem problem1 (x : ℝ) : (|x + 2| - 2 * x - 1 ≤ 0) ↔ (1 ≤ x) := sorry

-- Problem 2: Prove that the range of m such that there exists x ∈ ℝ satisfying f(x) + |x + 2| ≤ m for a = 1 is m ≥ 0
theorem problem2 (m : ℝ) : (∃ x : ℝ, |x - 1| - 2 * x - 1 + |x + 2| ≤ m) ↔ (0 ≤ m) := sorry

end problem1_problem2_l8_8854


namespace adam_walks_distance_l8_8434

/-- The side length of the smallest squares is 20 cm. --/
def smallest_square_side : ℕ := 20

/-- The side length of the middle-sized square is 2 times the smallest square. --/
def middle_square_side : ℕ := 2 * smallest_square_side

/-- The side length of the largest square is 3 times the smallest square. --/
def largest_square_side : ℕ := 3 * smallest_square_side

/-- The number of smallest squares Adam encounters. --/
def num_smallest_squares : ℕ := 5

/-- The number of middle-sized squares Adam encounters. --/
def num_middle_squares : ℕ := 5

/-- The number of largest squares Adam encounters. --/
def num_largest_squares : ℕ := 2

/-- The total distance Adam walks from P to Q. --/
def total_distance : ℕ :=
  num_smallest_squares * smallest_square_side +
  num_middle_squares * middle_square_side +
  num_largest_squares * largest_square_side

/-- Proof that the total distance Adam walks is 420 cm. --/
theorem adam_walks_distance : total_distance = 420 := by
  sorry

end adam_walks_distance_l8_8434


namespace skateboard_weight_is_18_l8_8114

def weight_of_canoe : Nat := 45
def weight_of_four_canoes := 4 * weight_of_canoe
def weight_of_ten_skateboards := weight_of_four_canoes
def weight_of_one_skateboard := weight_of_ten_skateboards / 10

theorem skateboard_weight_is_18 : weight_of_one_skateboard = 18 := by
  sorry

end skateboard_weight_is_18_l8_8114


namespace ratio_areas_l8_8381

-- Define the perimeter P
variable (P : ℝ) (hP : P > 0)

-- Define the side lengths
noncomputable def side_length_square := P / 4
noncomputable def side_length_triangle := P / 3

-- Define the radius of the circumscribed circle for the square
noncomputable def radius_square := (P * Real.sqrt 2) / 8
-- Define the area of the circumscribed circle for the square
noncomputable def area_circle_square := Real.pi * (radius_square P)^2

-- Define the radius of the circumscribed circle for the equilateral triangle
noncomputable def radius_triangle := (P * Real.sqrt 3) / 9 
-- Define the area of the circumscribed circle for the equilateral triangle
noncomputable def area_circle_triangle := Real.pi * (radius_triangle P)^2

-- Prove the ratio of the areas is 27/32
theorem ratio_areas (P : ℝ) (hP : P > 0) : 
  (area_circle_square P / area_circle_triangle P) = (27 / 32) := by
  sorry

end ratio_areas_l8_8381


namespace factor_expression_l8_8690

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l8_8690


namespace area_after_shortening_other_side_l8_8389

-- Define initial dimensions of the index card
def initial_length := 5
def initial_width := 7
def initial_area := initial_length * initial_width

-- Define the area condition when one side is shortened by 2 inches
def shortened_side_length := initial_length - 2
def new_area_after_shortening_one_side := 21

-- Definition of the problem condition that results in 21 square inches area
def condition := 
  (shortened_side_length * initial_width = new_area_after_shortening_one_side)

-- Final statement
theorem area_after_shortening_other_side :
  condition →
  (initial_length * (initial_width - 2) = 25) :=
by
  intro h
  sorry

end area_after_shortening_other_side_l8_8389


namespace trigonometric_inequality_l8_8663

theorem trigonometric_inequality (x : ℝ) : 0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧ 
                                            5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 :=
by
  sorry

end trigonometric_inequality_l8_8663


namespace regression_estimate_l8_8391

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end regression_estimate_l8_8391


namespace right_triangle_expression_l8_8675

theorem right_triangle_expression (a c b : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : 
  b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_expression_l8_8675


namespace events_related_with_99_confidence_l8_8923

theorem events_related_with_99_confidence (K_squared : ℝ) (h : K_squared > 6.635) : 
  events_A_B_related_with_99_confidence :=
sorry

end events_related_with_99_confidence_l8_8923


namespace repeat_decimals_subtraction_l8_8979

-- Define repeating decimal 0.4 repeating as a fraction
def repr_decimal_4 : ℚ := 4 / 9

-- Define repeating decimal 0.6 repeating as a fraction
def repr_decimal_6 : ℚ := 2 / 3

-- Theorem stating the equivalence of subtraction of these repeating decimals
theorem repeat_decimals_subtraction :
  repr_decimal_4 - repr_decimal_6 = -2 / 9 :=
sorry

end repeat_decimals_subtraction_l8_8979


namespace value_of_ab_l8_8532

theorem value_of_ab (a b : ℝ) (x : ℝ) 
  (h : ∀ x, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : a * b = 0 :=
sorry

end value_of_ab_l8_8532


namespace distance_from_hotel_l8_8019

def total_distance := 600
def speed1 := 50
def time1 := 3
def speed2 := 80
def time2 := 4

theorem distance_from_hotel :
  total_distance - (speed1 * time1 + speed2 * time2) = 130 := 
by
  sorry

end distance_from_hotel_l8_8019


namespace second_order_arithmetic_sequence_20th_term_l8_8801

theorem second_order_arithmetic_sequence_20th_term :
  (∀ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 4 ∧
    a 3 = 9 ∧
    a 4 = 16 ∧
    (∀ n, 2 ≤ n → a n - a (n - 1) = 2 * n - 1) →
    a 20 = 400) :=
by 
  sorry

end second_order_arithmetic_sequence_20th_term_l8_8801


namespace value_of_B_l8_8267

theorem value_of_B (x y : ℕ) (h1 : x > y) (h2 : y > 1) (h3 : x * y = x + y + 22) :
  (x / y) = 12 :=
sorry

end value_of_B_l8_8267


namespace amount_paid_after_discount_l8_8053

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l8_8053


namespace time_spent_watching_movies_l8_8176

def total_flight_time_minutes : ℕ := 11 * 60 + 20
def time_reading_minutes : ℕ := 2 * 60
def time_eating_dinner_minutes : ℕ := 30
def time_listening_radio_minutes : ℕ := 40
def time_playing_games_minutes : ℕ := 1 * 60 + 10
def time_nap_minutes : ℕ := 3 * 60

theorem time_spent_watching_movies :
  total_flight_time_minutes
  - time_reading_minutes
  - time_eating_dinner_minutes
  - time_listening_radio_minutes
  - time_playing_games_minutes
  - time_nap_minutes = 4 * 60 := by
  sorry

end time_spent_watching_movies_l8_8176


namespace solve_inequality_l8_8223

-- Define the function satisfying the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_functional_eq : ∀ (x y : ℝ), f (x / y) = f x - f y
axiom f_not_zero : ∀ x : ℝ, f x ≠ 0
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

-- Define the theorem that proves the inequality given the conditions
theorem solve_inequality (x : ℝ) :
  f x + f (x + 1/2) < 0 ↔ x ∈ (Set.Ioo ( (1 - Real.sqrt 17) / 4 ) 0) ∪ (Set.Ioo 0 ( (1 + Real.sqrt 17) / 4 )) :=
by
  sorry

end solve_inequality_l8_8223


namespace find_m_l8_8805

-- Define the vectors and the real number m
variables {Vec : Type*} [AddCommGroup Vec] [Module ℝ Vec]
variables (e1 e2 : Vec) (m : ℝ)

-- Define the collinearity condition and non-collinearity of the basis vectors.
def non_collinear (v1 v2 : Vec) : Prop := ¬(∃ (a : ℝ), v2 = a • v1)

def collinear (v1 v2 : Vec) : Prop := ∃ (a : ℝ), v2 = a • v1

-- Given conditions
axiom e1_e2_non_collinear : non_collinear e1 e2
axiom AB_eq : ∀ (m : ℝ), Vec
axiom CB_eq : Vec

theorem find_m (h : collinear (e1 + m • e2) (e1 - e2)) : m = -1 :=
sorry

end find_m_l8_8805


namespace steps_in_five_days_l8_8519

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l8_8519


namespace youngsville_population_l8_8546

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end youngsville_population_l8_8546


namespace second_derivative_l8_8684

noncomputable def y (x : ℝ) : ℝ := x^3 + Real.log x / Real.log 2 + Real.exp (-x)

theorem second_derivative (x : ℝ) : (deriv^[2] y x) = 3 * x^2 + (1 / (x * Real.log 2)) - Real.exp (-x) :=
by
  sorry

end second_derivative_l8_8684


namespace no_adjacent_stand_up_probability_l8_8888

noncomputable def coin_flip_prob_adjacent_people_stand_up : ℚ :=
  123 / 1024

theorem no_adjacent_stand_up_probability :
  let num_people := 10
  let total_outcomes := 2^num_people
  (123 : ℚ) / total_outcomes = coin_flip_prob_adjacent_people_stand_up :=
by
  sorry

end no_adjacent_stand_up_probability_l8_8888


namespace dasha_ate_one_bowl_l8_8808

-- Define the quantities for Masha, Dasha, Glasha, and Natasha
variables (M D G N : ℕ)

-- Given conditions
def conditions : Prop :=
  (M + D + G + N = 16) ∧
  (G + N = 9) ∧
  (M > D) ∧
  (M > G) ∧
  (M > N)

-- The problem statement rewritten in Lean: Prove that given the conditions, Dasha ate 1 bowl.
theorem dasha_ate_one_bowl (h : conditions M D G N) : D = 1 :=
sorry

end dasha_ate_one_bowl_l8_8808


namespace sector_area_l8_8128

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 2 * π / 3) (h_r : r = 3) : 
    (theta / (2 * π) * π * r^2) = 3 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end sector_area_l8_8128


namespace first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l8_8830

noncomputable def first_three_digits_of_decimal_part (x : ℝ) : ℕ :=
  -- here we would have the actual definition
  sorry

theorem first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8 :
  first_three_digits_of_decimal_part ((10^1001 + 1)^((9:ℝ) / 8)) = 125 :=
sorry

end first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l8_8830


namespace number_is_4_l8_8820

theorem number_is_4 (x : ℕ) (h : x + 5 = 9) : x = 4 := 
by {
  sorry
}

end number_is_4_l8_8820


namespace pencils_inequalities_l8_8911

theorem pencils_inequalities (x y : ℕ) :
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) :=
sorry

end pencils_inequalities_l8_8911


namespace totalNumberOfPupils_l8_8301

-- Definitions of the conditions
def numberOfGirls : Nat := 232
def numberOfBoys : Nat := 253

-- Statement of the problem
theorem totalNumberOfPupils : numberOfGirls + numberOfBoys = 485 := by
  sorry

end totalNumberOfPupils_l8_8301


namespace biased_die_sum_is_odd_l8_8477

def biased_die_probabilities : Prop :=
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let scenarios := [
    (1/3) * (2/3)^2,
    (1/3)^3
  ]
  let sum := scenarios.sum
  sum = 13 / 27

theorem biased_die_sum_is_odd :
  biased_die_probabilities := by
    sorry

end biased_die_sum_is_odd_l8_8477


namespace initial_money_l8_8395

/-
We had $3500 left after spending 30% of our money on clothing, 
25% on electronics, and saving 15% in a bank account. 
How much money (X) did we start with before shopping and saving?
-/

theorem initial_money (M : ℝ) 
  (h_clothing : 0.30 * M ≠ 0) 
  (h_electronics : 0.25 * M ≠ 0) 
  (h_savings : 0.15 * M ≠ 0) 
  (remaining_money : 0.30 * M = 3500) : 
  M = 11666.67 := 
sorry

end initial_money_l8_8395


namespace surveyed_individuals_not_working_percentage_l8_8686

theorem surveyed_individuals_not_working_percentage :
  (55 / 100 * 0 + 35 / 100 * (1 / 8) + 10 / 100 * (1 / 4)) = 6.875 / 100 :=
by
  sorry

end surveyed_individuals_not_working_percentage_l8_8686


namespace crystal_meals_count_l8_8415

def num_entrees : ℕ := 4
def num_drinks : ℕ := 4
def num_desserts : ℕ := 2

theorem crystal_meals_count : num_entrees * num_drinks * num_desserts = 32 := by
  sorry

end crystal_meals_count_l8_8415


namespace Jamal_crayon_cost_l8_8289

/-- Jamal bought 4 half dozen colored crayons at $2 per crayon. 
    He got a 10% discount on the total cost, and an additional 5% discount on the remaining amount. 
    After paying in US Dollars (USD), we want to know how much he spent in Euros (EUR) and British Pounds (GBP) 
    given that 1 USD is equal to 0.85 EUR and 1 USD is equal to 0.75 GBP. 
    This statement proves that the total cost was 34.884 EUR and 30.78 GBP. -/
theorem Jamal_crayon_cost :
  let number_of_crayons := 4 * 6
  let initial_cost := number_of_crayons * 2
  let first_discount := 0.10 * initial_cost
  let cost_after_first_discount := initial_cost - first_discount
  let second_discount := 0.05 * cost_after_first_discount
  let final_cost_usd := cost_after_first_discount - second_discount
  let final_cost_eur := final_cost_usd * 0.85
  let final_cost_gbp := final_cost_usd * 0.75
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := 
by
  sorry

end Jamal_crayon_cost_l8_8289


namespace money_after_purchase_l8_8577

def initial_money : ℕ := 4
def cost_of_candy_bar : ℕ := 1
def money_left : ℕ := 3

theorem money_after_purchase :
  initial_money - cost_of_candy_bar = money_left := by
  sorry

end money_after_purchase_l8_8577


namespace p_value_for_roots_l8_8992

theorem p_value_for_roots (α β : ℝ) (h1 : 3 * α^2 + 5 * α + 2 = 0) (h2 : 3 * β^2 + 5 * β + 2 = 0)
  (hαβ : α + β = -5/3) (hαβ_prod : α * β = 2/3) : p = -49/9 :=
by
  sorry

end p_value_for_roots_l8_8992


namespace determine_distance_l8_8779

noncomputable def distance_formula (d a b c : ℝ) : Prop :=
  (d / a = (d - 30) / b) ∧
  (d / b = (d - 15) / c) ∧
  (d / a = (d - 40) / c)

theorem determine_distance (d a b c : ℝ) (h : distance_formula d a b c) : d = 90 :=
by {
  sorry
}

end determine_distance_l8_8779


namespace Tobias_change_l8_8250

def cost_of_shoes := 95
def allowance_per_month := 5
def months_saving := 3
def charge_per_lawn := 15
def lawns_mowed := 4
def charge_per_driveway := 7
def driveways_shoveled := 5
def total_amount_saved : ℕ := (allowance_per_month * months_saving)
                          + (charge_per_lawn * lawns_mowed)
                          + (charge_per_driveway * driveways_shoveled)

theorem Tobias_change : total_amount_saved - cost_of_shoes = 15 := by
  sorry

end Tobias_change_l8_8250


namespace max_angle_OAB_l8_8956

/-- Let OA = a, OB = b, and OM = x on the right angle XOY, where a < b. 
    The value of x which maximizes the angle ∠AMB is sqrt(ab). -/
theorem max_angle_OAB (a b x : ℝ) (h : a < b) (h1 : x = Real.sqrt (a * b)) :
  x = Real.sqrt (a * b) :=
sorry

end max_angle_OAB_l8_8956


namespace face_value_amount_of_bill_l8_8788

def true_discount : ℚ := 45
def bankers_discount : ℚ := 54

theorem face_value_amount_of_bill : 
  ∃ (FV : ℚ), bankers_discount = true_discount + (true_discount * bankers_discount / FV) ∧ FV = 270 :=
by
  sorry

end face_value_amount_of_bill_l8_8788


namespace transform_quadratic_to_linear_l8_8455

theorem transform_quadratic_to_linear (x y : ℝ) : 
  x^2 - 4 * x * y + 4 * y^2 = 4 ↔ (x - 2 * y + 2 = 0 ∨ x - 2 * y - 2 = 0) :=
by
  sorry

end transform_quadratic_to_linear_l8_8455


namespace bus_fare_one_way_cost_l8_8193

-- Define the conditions
def zoo_entry (dollars : ℕ) : ℕ := dollars -- Zoo entry cost is $5 per person
def initial_money : ℕ := 40 -- They bring $40 with them
def money_left : ℕ := 24 -- They have $24 left after spending on zoo entry and bus fare

-- Given values
def noah_ava : ℕ := 2 -- Number of persons, Noah and Ava
def zoo_entry_cost : ℕ := 5 -- $5 per person for zoo entry
def total_money_spent := initial_money - money_left -- Money spent on zoo entry and bus fare

-- Function to calculate the total cost based on bus fare x
def total_cost (x : ℕ) : ℕ := noah_ava * zoo_entry_cost + 2 * noah_ava * x

-- Assertion to be proved
theorem bus_fare_one_way_cost : 
  ∃ (x : ℕ), total_cost x = total_money_spent ∧ x = 150 / 100 := sorry

end bus_fare_one_way_cost_l8_8193


namespace age_sum_l8_8227

theorem age_sum (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 10) : a + b + c = 27 := by
  sorry

end age_sum_l8_8227


namespace factor_expression_l8_8006

theorem factor_expression (x y : ℝ) : 5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) :=
by
  sorry

end factor_expression_l8_8006


namespace evaluate_expression_l8_8461

variable (a b c d : ℝ)

theorem evaluate_expression :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c :=
sorry

end evaluate_expression_l8_8461


namespace cost_of_pastrami_l8_8486

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l8_8486


namespace sin_gamma_isosceles_l8_8674

theorem sin_gamma_isosceles (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
  (h1 : a + m_c = s_1) (h2 : c + m_a = s_2) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt ((4 * s_1^2) - s_2^2) :=
sorry

end sin_gamma_isosceles_l8_8674


namespace smallest_integer_to_make_perfect_square_l8_8025

-- Define the number y as specified
def y : ℕ := 2^5 * 3^6 * (2^2)^7 * 5^8 * (2 * 3)^9 * 7^10 * (2^3)^11 * (3^2)^12

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The goal statement
theorem smallest_integer_to_make_perfect_square : 
  ∃ z : ℕ, z > 0 ∧ is_perfect_square (y * z) ∧ ∀ w : ℕ, w > 0 → is_perfect_square (y * w) → z ≤ w := by
  sorry

end smallest_integer_to_make_perfect_square_l8_8025


namespace opposite_and_reciprocal_numbers_l8_8566

theorem opposite_and_reciprocal_numbers (a b c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1) :
  2019 * a + (7 / (c * d)) + 2019 * b = 7 :=
sorry

end opposite_and_reciprocal_numbers_l8_8566


namespace completing_the_square_transformation_l8_8325

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l8_8325


namespace weight_of_each_package_l8_8418

theorem weight_of_each_package (W : ℝ) 
  (h1: 10 * W + 7 * W + 8 * W = 100) : W = 4 :=
by
  sorry

end weight_of_each_package_l8_8418


namespace value_of_a2022_l8_8364

theorem value_of_a2022 (a : ℕ → ℤ) (h : ∀ (n k : ℕ), 1 ≤ n ∧ n ≤ 2022 ∧ 1 ≤ k ∧ k ≤ 2022 → a n - a k ≥ (n^3 : ℤ) - (k^3 : ℤ)) (ha1011 : a 1011 = 0) : 
  a 2022 = 7246031367 := 
by
  sorry

end value_of_a2022_l8_8364


namespace sum_of_two_relatively_prime_integers_l8_8142

theorem sum_of_two_relatively_prime_integers (x y : ℕ) : 0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧
  gcd x y = 1 ∧ x * y + x + y = 119 ∧ x + y = 20 :=
by
  sorry

end sum_of_two_relatively_prime_integers_l8_8142


namespace range_of_m_l8_8307

def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
def g (m x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
by
  sorry

end range_of_m_l8_8307


namespace sample_quantities_and_probability_l8_8401

-- Define the given quantities from each workshop
def q_A := 10
def q_B := 20
def q_C := 30

-- Total sample size
def n := 6

-- Given conditions, the total quantity and sample ratio
def total_quantity := q_A + q_B + q_C
def ratio := n / total_quantity

-- Derived quantities in the samples based on the proportion
def sample_A := q_A * ratio
def sample_B := q_B * ratio
def sample_C := q_C * ratio

-- Combinatorial calculations
def C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_combinations := C 6 2
def workshop_C_combinations := C 3 2
def probability_C_samples := workshop_C_combinations / total_combinations

-- Theorem to prove the quantities and probability
theorem sample_quantities_and_probability :
  sample_A = 1 ∧ sample_B = 2 ∧ sample_C = 3 ∧ probability_C_samples = 1 / 5 :=
by
  sorry

end sample_quantities_and_probability_l8_8401


namespace book_costs_l8_8862

theorem book_costs (C1 C2 : ℝ) (h1 : C1 + C2 = 450) (h2 : 0.85 * C1 = 1.19 * C2) : C1 = 262.5 := 
sorry

end book_costs_l8_8862


namespace perpendicular_lines_solve_b_l8_8429

theorem perpendicular_lines_solve_b (b : ℝ) : (∀ x y : ℝ, y = 3 * x + 7 →
                                                    ∃ y1 : ℝ, y1 = ( - b / 4 ) * x + 3 ∧
                                                               3 * ( - b / 4 ) = -1) → 
                                               b = 4 / 3 :=
by
  sorry

end perpendicular_lines_solve_b_l8_8429


namespace product_of_three_numbers_l8_8029

theorem product_of_three_numbers (p q r m : ℝ) (h1 : p + q + r = 180) (h2 : m = 8 * p)
  (h3 : m = q - 10) (h4 : m = r + 10) : p * q * r = 90000 := by
  sorry

end product_of_three_numbers_l8_8029


namespace ice_cream_volume_l8_8970

-- Definitions based on Conditions
def radius_cone : Real := 3 -- radius at the opening of the cone
def height_cone : Real := 12 -- height of the cone

-- The proof statement
theorem ice_cream_volume :
  (1 / 3 * Real.pi * radius_cone^2 * height_cone) + (4 / 3 * Real.pi * radius_cone^3) = 72 * Real.pi := by
  sorry

end ice_cream_volume_l8_8970


namespace geometric_sequence_term_302_l8_8349

def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geometric_sequence_term_302 :
  let a := 8
  let r := -2
  geometric_sequence a r 302 = -2^304 := by
  sorry

end geometric_sequence_term_302_l8_8349


namespace remainder_when_divide_by_66_l8_8780

-- Define the conditions as predicates
def condition_1 (n : ℕ) : Prop := ∃ l : ℕ, n % 22 = 7
def condition_2 (n : ℕ) : Prop := ∃ m : ℕ, n % 33 = 18

-- Define the main theorem
theorem remainder_when_divide_by_66 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) : n % 66 = 51 :=
  sorry

end remainder_when_divide_by_66_l8_8780


namespace range_of_a_l8_8579

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 → (3 * a * x + (a^2 - 3 * a + 2) * y - 9 = 0 → y > 0)) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l8_8579


namespace chlorine_needed_l8_8963

variable (Methane moles_HCl moles_Cl₂ : ℕ)

-- Given conditions
def reaction_started_with_one_mole_of_methane : Prop :=
  Methane = 1

def reaction_produces_two_moles_of_HCl : Prop :=
  moles_HCl = 2

-- Question to be proved
def number_of_moles_of_Chlorine_combined : Prop :=
  moles_Cl₂ = 2

theorem chlorine_needed
  (h1 : reaction_started_with_one_mole_of_methane Methane)
  (h2 : reaction_produces_two_moles_of_HCl moles_HCl)
  : number_of_moles_of_Chlorine_combined moles_Cl₂ :=
sorry

end chlorine_needed_l8_8963


namespace soccer_team_students_l8_8281

theorem soccer_team_students :
  ∀ (n p b m : ℕ),
    n = 25 →
    p = 10 →
    b = 6 →
    n - (p - b) = m →
    m = 21 :=
by
  intros n p b m h_n h_p h_b h_trivial
  sorry

end soccer_team_students_l8_8281


namespace smallest_d_in_range_l8_8348

theorem smallest_d_in_range (d : ℝ) : (∃ x : ℝ, x^2 + 5 * x + d = 5) ↔ d ≤ 45 / 4 := 
sorry

end smallest_d_in_range_l8_8348


namespace percentage_failed_in_english_l8_8027

theorem percentage_failed_in_english
  (H_perc : ℝ) (B_perc : ℝ) (Passed_in_English_alone : ℝ) (Total_candidates : ℝ)
  (H_perc_eq : H_perc = 36)
  (B_perc_eq : B_perc = 15)
  (Passed_in_English_alone_eq : Passed_in_English_alone = 630)
  (Total_candidates_eq : Total_candidates = 3000) :
  ∃ E_perc : ℝ, E_perc = 85 := by
  sorry

end percentage_failed_in_english_l8_8027


namespace mean_of_all_students_l8_8595

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_l8_8595


namespace harmonic_mean_1999_2001_is_2000_l8_8891

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_1999_2001_is_2000 :
  abs (harmonic_mean 1999 2001 - 2000 : ℚ) < 1 := by
  -- Actual proof omitted
  sorry

end harmonic_mean_1999_2001_is_2000_l8_8891


namespace prob1_prob2_prob3_l8_8672

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

end prob1_prob2_prob3_l8_8672


namespace extremum_at_x1_l8_8002

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_x1 (a b : ℝ) (h1 : (3*1^2 + 2*a*1 + b) = 0) (h2 : 1^3 + a*1^2 + b*1 + a^2 = 10) :
  a = 4 :=
by
  sorry

end extremum_at_x1_l8_8002


namespace gary_egg_collection_l8_8220

-- Conditions
def initial_chickens : ℕ := 4
def multiplier : ℕ := 8
def eggs_per_chicken_per_day : ℕ := 6
def days_in_week : ℕ := 7

-- Definitions derived from conditions
def current_chickens : ℕ := initial_chickens * multiplier
def eggs_per_day : ℕ := current_chickens * eggs_per_chicken_per_day
def eggs_per_week : ℕ := eggs_per_day * days_in_week

-- Proof statement
theorem gary_egg_collection : eggs_per_week = 1344 := by
  unfold eggs_per_week
  unfold eggs_per_day
  unfold current_chickens
  sorry

end gary_egg_collection_l8_8220


namespace complement_intersection_l8_8494

-- Define the universal set U.
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set M.
def M : Set ℕ := {2, 3}

-- Define the set N.
def N : Set ℕ := {1, 3}

-- Define the complement of set M in U.
def complement_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- Define the complement of set N in U.
def complement_U_N : Set ℕ := {x ∈ U | x ∉ N}

-- The statement to be proven.
theorem complement_intersection :
  (complement_U_M ∩ complement_U_N) = {4, 5, 6} :=
sorry

end complement_intersection_l8_8494


namespace Simon_has_72_legos_l8_8796

theorem Simon_has_72_legos 
  (Kent_legos : ℕ)
  (h1 : Kent_legos = 40) 
  (Bruce_legos : ℕ) 
  (h2 : Bruce_legos = Kent_legos + 20) 
  (Simon_legos : ℕ) 
  (h3 : Simon_legos = Bruce_legos + (Bruce_legos/5)) :
  Simon_legos = 72 := 
  by
    -- Begin proof (not required for the problem)
    -- Proof steps would follow here
    sorry

end Simon_has_72_legos_l8_8796


namespace abs_a_lt_abs_b_sub_abs_c_l8_8073

theorem abs_a_lt_abs_b_sub_abs_c (a b c : ℝ) (h : |a + c| < b) : |a| < |b| - |c| :=
sorry

end abs_a_lt_abs_b_sub_abs_c_l8_8073


namespace calculate_expression_l8_8750

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end calculate_expression_l8_8750


namespace solve_eq_integers_l8_8597

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l8_8597


namespace max_distance_between_circle_centers_l8_8635

theorem max_distance_between_circle_centers :
  let rect_width := 20
  let rect_height := 16
  let circle_diameter := 8
  let horiz_distance := rect_width - circle_diameter
  let vert_distance := rect_height - circle_diameter
  let max_distance := Real.sqrt (horiz_distance ^ 2 + vert_distance ^ 2)
  max_distance = 4 * Real.sqrt 13 :=
by
  sorry

end max_distance_between_circle_centers_l8_8635


namespace initial_deposit_l8_8153

theorem initial_deposit :
  ∀ (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ),
    r = 0.05 → n = 1 → t = 2 → P * (1 + r / n) ^ (n * t) = 6615 → P = 6000 :=
by
  intros P r n t h_r h_n h_t h_eq
  rw [h_r, h_n, h_t] at h_eq
  norm_num at h_eq
  sorry

end initial_deposit_l8_8153


namespace geometric_sequence_product_l8_8458

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_cond : a 2 * a 4 = 16) : a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64 :=
by
  sorry

end geometric_sequence_product_l8_8458


namespace sufficient_but_not_necessary_condition_l8_8384

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 2

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) : 
  a ≤ 0 :=
sorry

end sufficient_but_not_necessary_condition_l8_8384


namespace train_length_l8_8873

theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (speed_mps : ℝ) (length_train : ℝ) : 
  speed_kmph = 90 → 
  time_seconds = 6 → 
  speed_mps = (speed_kmph * 1000 / 3600) →
  length_train = (speed_mps * time_seconds) → 
  length_train = 150 :=
by
  intros h_speed h_time h_speed_mps h_length
  sorry

end train_length_l8_8873


namespace percentage_defective_units_shipped_l8_8517

noncomputable def defective_percent : ℝ := 0.07
noncomputable def shipped_percent : ℝ := 0.05

theorem percentage_defective_units_shipped :
  defective_percent * shipped_percent * 100 = 0.35 :=
by
  -- Proof body here
  sorry

end percentage_defective_units_shipped_l8_8517


namespace cost_of_gravelling_path_eq_630_l8_8721

-- Define the dimensions of the grassy plot.
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65

-- Define the width of the gravel path.
def width_gravel_path : ℝ := 2.5

-- Define the cost of gravelling per square meter in INR.
def cost_per_sqm : ℝ := 0.70

-- Compute the dimensions of the plot including the gravel path.
def length_including_path := length_grassy_plot + 2 * width_gravel_path
def width_including_path := width_grassy_plot + 2 * width_gravel_path

-- Compute the area of the plot including the gravel path.
def area_including_path := length_including_path * width_including_path

-- Compute the area of the grassy plot without the gravel path.
def area_grassy_plot := length_grassy_plot * width_grassy_plot

-- Compute the area of the gravel path alone.
def area_gravel_path := area_including_path - area_grassy_plot

-- Compute the total cost of gravelling the path.
def total_cost := area_gravel_path * cost_per_sqm

-- The theorem stating the cost of gravelling the path.
theorem cost_of_gravelling_path_eq_630 : total_cost = 630 := by
  -- Proof goes here
  sorry

end cost_of_gravelling_path_eq_630_l8_8721


namespace parabola_equation_l8_8910

theorem parabola_equation (P : ℝ × ℝ) (hP : P = (-4, -2)) :
  (∃ p : ℝ, P.1^2 = -2 * p * P.2 ∧ p = -4 ∧ x^2 = -8*y) ∨ 
  (∃ p : ℝ, P.2^2 = -2 * p * P.1 ∧ p = -1/2 ∧ y^2 = -x) :=
by
  sorry

end parabola_equation_l8_8910


namespace minimum_paper_toys_is_eight_l8_8806

noncomputable def minimum_paper_toys (s_boats: ℕ) (s_planes: ℕ) : ℕ :=
  s_boats * 8 + s_planes * 6

theorem minimum_paper_toys_is_eight :
  ∀ (s_boats s_planes : ℕ), s_boats >= 1 → minimum_paper_toys s_boats s_planes = 8 → s_planes = 0 :=
by
  intros s_boats s_planes h_boats h_eq
  have h1: s_boats * 8 + s_planes * 6 = 8 := h_eq
  sorry

end minimum_paper_toys_is_eight_l8_8806


namespace train_speed_l8_8555

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_conversion_factor : ℝ) (expected_speed : ℝ) (h_time_conversion : time_conversion_factor = 1 / 60) (h_time : time_minutes / 60 = 0.5) (h_distance : distance = 51) (h_expected_speed : expected_speed = 102) : distance / (time_minutes / 60) = expected_speed :=
by 
  sorry

end train_speed_l8_8555


namespace not_divisible_2310_l8_8013

theorem not_divisible_2310 (n : ℕ) (h : n < 2310) : ¬ (2310 ∣ n * (2310 - n)) :=
sorry

end not_divisible_2310_l8_8013


namespace exponential_inequality_l8_8065

theorem exponential_inequality (k l m : ℕ) : 2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 :=
by
  sorry

end exponential_inequality_l8_8065


namespace triangle_angles_are_30_60_90_l8_8180

theorem triangle_angles_are_30_60_90
  (a b c OH R r : ℝ)
  (h1 : OH = c / 2)
  (h2 : OH = a)
  (h3 : a < b)
  (h4 : b < c)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  ∃ (A B C : ℝ), (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) :=
sorry

end triangle_angles_are_30_60_90_l8_8180


namespace Marissa_has_21_more_marbles_than_Jonny_l8_8592

noncomputable def Mara_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Markus_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Jonny_marbles (total_marbles : ℕ) (bags : ℕ) : ℕ :=
total_marbles

noncomputable def Marissa_marbles (bags1 : ℕ) (marbles1 : ℕ) (bags2 : ℕ) (marbles2 : ℕ) : ℕ :=
(bags1 * marbles1) + (bags2 * marbles2)

noncomputable def Jonny : ℕ := Jonny_marbles 18 3

noncomputable def Marissa : ℕ := Marissa_marbles 3 5 3 8

theorem Marissa_has_21_more_marbles_than_Jonny : (Marissa - Jonny) = 21 :=
by
  sorry

end Marissa_has_21_more_marbles_than_Jonny_l8_8592


namespace face_value_stock_l8_8172

-- Given conditions
variables (F : ℝ) (yield quoted_price dividend_rate : ℝ)
variables (h_yield : yield = 20) (h_quoted_price : quoted_price = 125)
variables (h_dividend_rate : dividend_rate = 0.25)

--Theorem to prove the face value of the stock is 100
theorem face_value_stock : (dividend_rate * F / quoted_price) * 100 = yield ↔ F = 100 :=
by
  sorry

end face_value_stock_l8_8172


namespace multiply_by_11_l8_8445

theorem multiply_by_11 (A B k : ℕ) (h1 : 10 * A + B < 100) (h2 : A + B = 10 + k) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * k + B :=
by 
  sorry

end multiply_by_11_l8_8445


namespace ratio_of_novels_read_l8_8170

theorem ratio_of_novels_read (jordan_read : ℕ) (alexandre_read : ℕ)
  (h_jordan_read : jordan_read = 120) 
  (h_diff : jordan_read = alexandre_read + 108) :
  alexandre_read / jordan_read = 1 / 10 :=
by
  -- Proof skipped
  sorry

end ratio_of_novels_read_l8_8170


namespace urn_problem_l8_8757

noncomputable def count_balls (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) : ℕ :=
initial_white + initial_black + operations

noncomputable def urn_probability (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) (final_white : ℕ) (final_black : ℕ) : ℚ :=
if final_white + final_black = count_balls initial_white initial_black operations &&
   final_white = (initial_white + (operations - (final_black - initial_black))) &&
   (final_white + final_black) = 8 then 3 / 5 else 0

theorem urn_problem :
  let initial_white := 2
  let initial_black := 1
  let operations := 4
  let final_white := 4
  let final_black := 4
  count_balls initial_white initial_black operations = 8 ∧ urn_probability initial_white initial_black operations final_white final_black = 3 / 5 :=
by
  sorry

end urn_problem_l8_8757


namespace sam_initial_balloons_l8_8810

theorem sam_initial_balloons:
  ∀ (S : ℕ), (S - 10 + 16 = 52) → S = 46 :=
by
  sorry

end sam_initial_balloons_l8_8810


namespace voldemort_calorie_intake_limit_l8_8652

theorem voldemort_calorie_intake_limit :
  let breakfast := 560
  let lunch := 780
  let cake := 110
  let chips := 310
  let coke := 215
  let dinner := cake + chips + coke
  let remaining := 525
  breakfast + lunch + dinner + remaining = 2500 :=
by
  -- to clarify, the statement alone is provided, so we add 'sorry' to omit the actual proof steps
  sorry

end voldemort_calorie_intake_limit_l8_8652


namespace ten_percent_of_x_l8_8178

variable (certain_value : ℝ)
variable (x : ℝ)

theorem ten_percent_of_x (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = certain_value) :
  0.1 * x = 0.7 * (1.5 - certain_value) := sorry

end ten_percent_of_x_l8_8178


namespace solution_of_inequality_l8_8694

theorem solution_of_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (x - a) * (x - a⁻¹) < 0 ↔ a < x ∧ x < a⁻¹ :=
by sorry

end solution_of_inequality_l8_8694


namespace y_at_x_equals_2sqrt3_l8_8460

theorem y_at_x_equals_2sqrt3 (k : ℝ) (y : ℝ → ℝ)
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
sorry

end y_at_x_equals_2sqrt3_l8_8460


namespace alpha_beta_sum_l8_8213

theorem alpha_beta_sum (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80 * x + 1551) / (x^2 + 57 * x - 2970)) :
  α + β = 137 :=
by
  sorry

end alpha_beta_sum_l8_8213


namespace smallest_integer_solution_l8_8211

theorem smallest_integer_solution (x : ℤ) (h : 2 * (x : ℝ)^2 + 2 * |(x : ℝ)| + 7 < 25) : x = -2 :=
by
  sorry

end smallest_integer_solution_l8_8211


namespace polynomial_sum_l8_8763

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - 5 * x - 7
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 2 * x + 8

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -5 * x^3 + 11 * x^2 + x - 8 :=
  sorry

end polynomial_sum_l8_8763


namespace arithmetic_sequence_properties_l8_8989

theorem arithmetic_sequence_properties (a b c : ℝ) (h1 : ∃ d : ℝ, [2, a, b, c, 9] = [2, 2 + d, 2 + 2 * d, 2 + 3 * d, 2 + 4 * d]) : 
  c - a = 7 / 2 := 
by
  -- We assume the proof here
  sorry

end arithmetic_sequence_properties_l8_8989


namespace geometric_sequence_a6_l8_8147

theorem geometric_sequence_a6 (a : ℕ → ℝ) (geometric_seq : ∀ n, a (n + 1) = a n * a 1)
  (h1 : (a 4) * (a 8) = 9) (h2 : (a 4) + (a 8) = -11) : a 6 = -3 := by
  sorry

end geometric_sequence_a6_l8_8147


namespace factor_expression_l8_8151

variable (x : ℝ)

def e : ℝ := (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5)

theorem factor_expression : e x = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by
  sorry

end factor_expression_l8_8151


namespace describes_random_event_proof_l8_8747

def describes_random_event (phrase : String) : Prop :=
  match phrase with
  | "Winter turns into spring"  => False
  | "Fishing for the moon in the water" => False
  | "Seeking fish on a tree" => False
  | "Meeting unexpectedly" => True
  | _ => False

theorem describes_random_event_proof : describes_random_event "Meeting unexpectedly" = True :=
by
  sorry

end describes_random_event_proof_l8_8747


namespace geometric_sequence_a5_l8_8185

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r)
  (h_eqn : ∃ x : ℝ, (x^2 + 7*x + 9 = 0) ∧ (a 3 = x) ∧ (a 7 = x)) :
  a 5 = 3 ∨ a 5 = -3 := 
sorry

end geometric_sequence_a5_l8_8185


namespace functional_equation_l8_8133

noncomputable def f : ℝ → ℝ :=
  sorry

theorem functional_equation (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end functional_equation_l8_8133


namespace compare_abc_l8_8380

noncomputable def a : ℝ := (1 / 2) * Real.cos (4 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (2 * 13 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (2 * 23 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l8_8380


namespace groups_partition_count_l8_8972

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end groups_partition_count_l8_8972


namespace largest_divisor_of_m_l8_8789

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (h : 54 ∣ m^2) : 18 ∣ m :=
sorry

end largest_divisor_of_m_l8_8789


namespace diagonals_from_vertex_l8_8268

theorem diagonals_from_vertex (n : ℕ) (h : (n-2) * 180 + 360 = 1800) : (n - 3) = 7 :=
sorry

end diagonals_from_vertex_l8_8268


namespace arithmetic_mean_reciprocal_primes_l8_8977

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l8_8977


namespace regular_polygon_sides_l8_8877

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l8_8877


namespace centroid_y_sum_zero_l8_8028

theorem centroid_y_sum_zero
  (x1 x2 x3 y2 y3 : ℝ)
  (h : y2 + y3 = 0) :
  (x1 + x2 + x3) / 3 = (x1 / 3 + x2 / 3 + x3 / 3) ∧ (y2 + y3) / 3 = 0 :=
by
  sorry

end centroid_y_sum_zero_l8_8028


namespace max_abs_value_l8_8181

theorem max_abs_value (x y : ℝ) (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : |x - 2 * y + 1| ≤ 6 :=
sorry

end max_abs_value_l8_8181


namespace layla_earnings_l8_8036

def rate_donaldsons : ℕ := 15
def bonus_donaldsons : ℕ := 5
def hours_donaldsons : ℕ := 7
def rate_merck : ℕ := 18
def discount_merck : ℝ := 0.10
def hours_merck : ℕ := 6
def rate_hille : ℕ := 20
def bonus_hille : ℕ := 10
def hours_hille : ℕ := 3
def rate_johnson : ℕ := 22
def flat_rate_johnson : ℕ := 80
def hours_johnson : ℕ := 4
def rate_ramos : ℕ := 25
def bonus_ramos : ℕ := 20
def hours_ramos : ℕ := 2

def donaldsons_earnings := rate_donaldsons * hours_donaldsons + bonus_donaldsons
def merck_earnings := rate_merck * hours_merck - (rate_merck * hours_merck * discount_merck : ℝ)
def hille_earnings := rate_hille * hours_hille + bonus_hille
def johnson_earnings := rate_johnson * hours_johnson
def ramos_earnings := rate_ramos * hours_ramos + bonus_ramos

noncomputable def total_earnings : ℝ :=
  donaldsons_earnings + merck_earnings + hille_earnings + johnson_earnings + ramos_earnings

theorem layla_earnings : total_earnings = 435.2 :=
by
  sorry

end layla_earnings_l8_8036


namespace point_B_possible_values_l8_8459

-- Define point A
def A : ℝ := 1

-- Define the condition that B is 3 units away from A
def units_away (a b : ℝ) : ℝ := abs (b - a)

theorem point_B_possible_values :
  ∃ B : ℝ, units_away A B = 3 ∧ (B = 4 ∨ B = -2) := by
  sorry

end point_B_possible_values_l8_8459


namespace smallest_n_circle_l8_8696

theorem smallest_n_circle (n : ℕ) 
    (h1 : ∀ i j : ℕ, i < j → j - i = 3 ∨ j - i = 4 ∨ j - i = 5) :
    n = 7 :=
sorry

end smallest_n_circle_l8_8696


namespace ratio_of_boys_to_total_l8_8761

theorem ratio_of_boys_to_total (p_b p_g : ℝ) (h1 : p_b + p_g = 1) (h2 : p_b = (2 / 3) * p_g) :
  p_b = 2 / 5 :=
by
  sorry

end ratio_of_boys_to_total_l8_8761


namespace A_runs_faster_l8_8706

variable (v_A v_B : ℝ)  -- Speed of A and B
variable (k : ℝ)       -- Factor by which A is faster than B

-- Conditions as definitions in Lean:
def speed_relation (k : ℝ) (v_A v_B : ℝ) : Prop := v_A = k * v_B
def start_difference : ℝ := 60
def race_course_length : ℝ := 80
def reach_finish_same_time (v_A v_B : ℝ) : Prop := (80 / v_A) = ((80 - start_difference) / v_B)

theorem A_runs_faster
  (h1 : speed_relation k v_A v_B)
  (h2 : reach_finish_same_time v_A v_B) : k = 4 :=
by
  sorry

end A_runs_faster_l8_8706


namespace triangular_array_nth_row_4th_number_l8_8295

theorem triangular_array_nth_row_4th_number (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, k = 4 ∧ (2: ℕ)^(n * (n - 1) / 2 + 3) = 2^((n^2 - n + 6) / 2) :=
by
  sorry

end triangular_array_nth_row_4th_number_l8_8295


namespace area_above_the_line_l8_8399

-- Definitions of the circle and the line equations
def circle_eqn (x y : ℝ) := (x - 5)^2 + (y - 3)^2 = 1
def line_eqn (x y : ℝ) := y = x - 5

-- The main statement to prove
theorem area_above_the_line : 
  ∃ (A : ℝ), A = (3 / 4) * Real.pi ∧ 
  ∀ (x y : ℝ), 
    circle_eqn x y ∧ y > x - 5 → 
    A > 0 := 
sorry

end area_above_the_line_l8_8399


namespace solution_set_abs_inequality_l8_8875

theorem solution_set_abs_inequality (x : ℝ) : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end solution_set_abs_inequality_l8_8875


namespace piggy_bank_dimes_l8_8859

theorem piggy_bank_dimes (q d : ℕ) 
  (h1 : q + d = 100) 
  (h2 : 25 * q + 10 * d = 1975) : 
  d = 35 :=
by
  -- skipping the proof
  sorry

end piggy_bank_dimes_l8_8859


namespace negation_of_sine_bound_l8_8570

theorem negation_of_sine_bound (p : ∀ x : ℝ, Real.sin x ≤ 1) : ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x₀ : ℝ, Real.sin x₀ > 1 := 
by 
  sorry

end negation_of_sine_bound_l8_8570


namespace complementary_event_l8_8523

def car_a_selling_well : Prop := sorry
def car_b_selling_poorly : Prop := sorry

def event_A : Prop := car_a_selling_well ∧ car_b_selling_poorly
def event_complement (A : Prop) : Prop := ¬A

theorem complementary_event :
  event_complement event_A = (¬car_a_selling_well ∨ ¬car_b_selling_poorly) :=
by
  sorry

end complementary_event_l8_8523


namespace sam_age_l8_8514

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l8_8514


namespace nathan_tokens_l8_8277

theorem nathan_tokens
  (hockey_games : Nat := 5)
  (hockey_cost : Nat := 4)
  (basketball_games : Nat := 7)
  (basketball_cost : Nat := 5)
  (skee_ball_games : Nat := 3)
  (skee_ball_cost : Nat := 3)
  : hockey_games * hockey_cost + basketball_games * basketball_cost + skee_ball_games * skee_ball_cost = 64 := 
by
  sorry

end nathan_tokens_l8_8277


namespace secret_code_count_l8_8358

noncomputable def number_of_secret_codes (colors slots : ℕ) : ℕ :=
  colors ^ slots

theorem secret_code_count : number_of_secret_codes 9 5 = 59049 := by
  sorry

end secret_code_count_l8_8358


namespace original_denominator_l8_8026

theorem original_denominator (d : ℕ) (h : 11 = 3 * (d + 8)) : d = 25 :=
by
  sorry

end original_denominator_l8_8026


namespace function_inequality_m_l8_8126

theorem function_inequality_m (m : ℝ) : (∀ x : ℝ, (1 / 2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) ↔ m ≥ (3 / 2) := sorry

end function_inequality_m_l8_8126


namespace odd_square_minus_one_div_by_eight_l8_8310

theorem odd_square_minus_one_div_by_eight (n : ℤ) : ∃ k : ℤ, (2 * n + 1) ^ 2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_div_by_eight_l8_8310


namespace area_of_circle_with_endpoints_l8_8895

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (d : ℝ) : ℝ :=
  d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circle_with_endpoints :
  area_of_circle (radius (distance (5, 9) (13, 17))) = 32 * Real.pi :=
by
  sorry

end area_of_circle_with_endpoints_l8_8895


namespace set_operation_equivalence_l8_8234

variable {U : Type} -- U is the universal set
variables {X Y Z : Set U} -- X, Y, and Z are subsets of the universal set U

def star (A B : Set U) : Set U := A ∩ B  -- Define the operation "∗" as intersection

theorem set_operation_equivalence :
  star (star X Y) Z = (X ∩ Y) ∩ Z :=  -- Formulate the problem as a theorem to prove
by
  sorry  -- Proof is omitted

end set_operation_equivalence_l8_8234


namespace remainder_when_A_divided_by_9_l8_8896

theorem remainder_when_A_divided_by_9 (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := 
by {
  sorry
}

end remainder_when_A_divided_by_9_l8_8896


namespace colored_copies_count_l8_8869

theorem colored_copies_count :
  ∃ C W : ℕ, (C + W = 400) ∧ (10 * C + 5 * W = 2250) ∧ (C = 50) :=
by
  sorry

end colored_copies_count_l8_8869


namespace sum_of_coeffs_eq_225_l8_8467

/-- The sum of the coefficients of all terms in the expansion
of (C_x + C_x^2 + C_x^3 + C_x^4)^2 is equal to 225. -/
theorem sum_of_coeffs_eq_225 (C_x : ℝ) : 
  (C_x + C_x^2 + C_x^3 + C_x^4)^2 = 225 :=
sorry

end sum_of_coeffs_eq_225_l8_8467


namespace new_profit_is_122_03_l8_8337

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end new_profit_is_122_03_l8_8337


namespace no_solution_for_12k_plus_7_l8_8872

theorem no_solution_for_12k_plus_7 (k : ℤ) :
  ∀ (a b c : ℕ), 12 * k + 7 ≠ 2^a + 3^b - 5^c := 
by sorry

end no_solution_for_12k_plus_7_l8_8872


namespace part_a_ellipse_and_lines_l8_8206

theorem part_a_ellipse_and_lines (x y : ℝ) : 
  (4 * x^2 + 8 * y^2 + 8 * y * abs y = 1) ↔ 
  ((y ≥ 0 ∧ (x^2 / (1/4) + y^2 / (1/16)) = 1) ∨ 
  (y < 0 ∧ ((x = 1/2) ∨ (x = -1/2)))) := 
sorry

end part_a_ellipse_and_lines_l8_8206


namespace relationship_between_y1_y2_l8_8087

variable (k b y1 y2 : ℝ)

-- Let A = (-3, y1) and B = (4, y2) be points on the line y = kx + b, with k < 0
axiom A_on_line : y1 = k * -3 + b
axiom B_on_line : y2 = k * 4 + b
axiom k_neg : k < 0

theorem relationship_between_y1_y2 : y1 > y2 :=
by sorry

end relationship_between_y1_y2_l8_8087


namespace find_z_l8_8319

noncomputable def solve_for_z (i : ℂ) (z : ℂ) :=
  (2 - i) * z = i ^ 2021

theorem find_z (i z : ℂ) (h1 : solve_for_z i z) : 
  z = -1/5 + 2/5 * i := 
by 
  sorry

end find_z_l8_8319


namespace students_taking_both_languages_l8_8336

theorem students_taking_both_languages (F S B : ℕ) (hF : F = 21) (hS : S = 21) (h30 : 30 = F - B + (S - B)) : B = 6 :=
by
  rw [hF, hS] at h30
  sorry

end students_taking_both_languages_l8_8336


namespace find_general_equation_of_line_l8_8469

variables {x y k b : ℝ}

-- Conditions: slope of the line is -2 and sum of its intercepts is 12.
def slope_of_line (l : ℝ → ℝ → Prop) : Prop := ∃ b, ∀ x y, l x y ↔ y = -2 * x + b
def sum_of_intercepts (l : ℝ → ℝ → Prop) : Prop := ∃ b, b + (b / 2) = 12

-- Question: What is the general equation of the line?
noncomputable def general_equation (l : ℝ → ℝ → Prop) : Prop :=
  slope_of_line l ∧ sum_of_intercepts l → ∀ x y, l x y ↔ 2 * x + y - 8 = 0

-- The theorem we need to prove
theorem find_general_equation_of_line (l : ℝ → ℝ → Prop) : general_equation l :=
sorry

end find_general_equation_of_line_l8_8469


namespace prism_volume_l8_8586

noncomputable def volume_of_prism (x y z : ℝ) : ℝ :=
  x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 40) (h2 : x * z = 50) (h3 : y * z = 100) :
  volume_of_prism x y z = 100 * Real.sqrt 2 :=
by
  sorry

end prism_volume_l8_8586


namespace length_of_second_train_is_319_95_l8_8288

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ) (time_to_cross_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kph * 1000 / 3600
  let speed_second_train_mps := speed_second_train_kph * 1000 / 3600
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross_seconds
  let length_second_train := total_distance_covered - length_first_train
  length_second_train

theorem length_of_second_train_is_319_95 :
  length_of_second_train 180 120 80 9 = 319.95 :=
sorry

end length_of_second_train_is_319_95_l8_8288


namespace second_more_than_third_l8_8748

def firstChapterPages : ℕ := 35
def secondChapterPages : ℕ := 18
def thirdChapterPages : ℕ := 3

theorem second_more_than_third : secondChapterPages - thirdChapterPages = 15 := by
  sorry

end second_more_than_third_l8_8748


namespace find_C_l8_8717

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 := 
by
  sorry

end find_C_l8_8717


namespace how_many_whole_boxes_did_nathan_eat_l8_8031

-- Define the conditions
def gumballs_per_package := 5
def total_gumballs := 20

-- The problem to prove
theorem how_many_whole_boxes_did_nathan_eat : total_gumballs / gumballs_per_package = 4 :=
by sorry

end how_many_whole_boxes_did_nathan_eat_l8_8031


namespace find_b_value_l8_8933

theorem find_b_value {b : ℚ} (h : -8 ^ 2 + b * -8 - 45 = 0) : b = 19 / 8 :=
sorry

end find_b_value_l8_8933


namespace slopes_hyperbola_l8_8949

theorem slopes_hyperbola 
  (x y : ℝ)
  (M : ℝ × ℝ) 
  (t m : ℝ) 
  (h_point_M_on_line: M = (9 / 5, t))
  (h_hyperbola : ∀ t: ℝ, (16 * m^2 - 9) * t^2 + 160 * m * t + 256 = 0)
  (k1 k2 k3 : ℝ)
  (h_k2 : k2 = -5 * t / 16) :
  k1 + k3 = 2 * k2 :=
sorry

end slopes_hyperbola_l8_8949


namespace cos_double_angle_l8_8620

theorem cos_double_angle (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) : 
  Real.cos (20 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end cos_double_angle_l8_8620


namespace twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l8_8343

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l8_8343


namespace trapezoid_perimeter_l8_8441

theorem trapezoid_perimeter (x y : ℝ) (h1 : x ≠ 0)
  (h2 : ∀ (AB CD AD BC : ℝ), AB = 2 * x ∧ CD = 4 * x ∧ AD = 2 * y ∧ BC = y) :
  (∀ (P : ℝ), P = AB + BC + CD + AD → P = 6 * x + 3 * y) :=
by sorry

end trapezoid_perimeter_l8_8441


namespace alloy_cut_weight_l8_8011

variable (a b x : ℝ)
variable (ha : 0 ≤ a ∧ a ≤ 1) -- assuming copper content is a fraction between 0 and 1
variable (hb : 0 ≤ b ∧ b ≤ 1)
variable (h : a ≠ b)
variable (hx : 0 < x ∧ x < 40) -- x is strictly between 0 and 40 (since 0 ≤ x ≤ 40)

theorem alloy_cut_weight (A B : ℝ) (hA : A = 40) (hB : B = 60) (h1 : (a * x + b * (A - x)) / 40 = (b * x + a * (B - x)) / 60) : x = 24 :=
by
  sorry

end alloy_cut_weight_l8_8011


namespace difference_even_odd_sums_l8_8638

def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : sum_first_n_even_numbers 1001 - sum_first_n_odd_numbers 1001 = 1001 := by
  sorry

end difference_even_odd_sums_l8_8638


namespace lowest_possible_number_of_students_l8_8096

theorem lowest_possible_number_of_students :
  ∃ n : ℕ, (n % 12 = 0 ∧ n % 24 = 0) ∧ ∀ m : ℕ, ((m % 12 = 0 ∧ m % 24 = 0) → n ≤ m) :=
sorry

end lowest_possible_number_of_students_l8_8096


namespace sum_of_interior_angles_of_regular_polygon_l8_8952

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : 60 = 360 / n) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l8_8952


namespace selling_price_l8_8370

noncomputable def total_cost_first_mixture : ℝ := 27 * 150
noncomputable def total_cost_second_mixture : ℝ := 36 * 125
noncomputable def total_cost_third_mixture : ℝ := 18 * 175
noncomputable def total_cost_fourth_mixture : ℝ := 24 * 120

noncomputable def total_cost : ℝ := total_cost_first_mixture + total_cost_second_mixture + total_cost_third_mixture + total_cost_fourth_mixture

noncomputable def profit_first_mixture : ℝ := 0.4 * total_cost_first_mixture
noncomputable def profit_second_mixture : ℝ := 0.3 * total_cost_second_mixture
noncomputable def profit_third_mixture : ℝ := 0.2 * total_cost_third_mixture
noncomputable def profit_fourth_mixture : ℝ := 0.25 * total_cost_fourth_mixture

noncomputable def total_profit : ℝ := profit_first_mixture + profit_second_mixture + profit_third_mixture + profit_fourth_mixture

noncomputable def total_weight : ℝ := 27 + 36 + 18 + 24
noncomputable def total_selling_price : ℝ := total_cost + total_profit

noncomputable def selling_price_per_kg : ℝ := total_selling_price / total_weight

theorem selling_price : selling_price_per_kg = 180 := by
  sorry

end selling_price_l8_8370


namespace cost_price_percentage_l8_8986

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end cost_price_percentage_l8_8986


namespace ratio_of_radii_of_truncated_cone_l8_8205

theorem ratio_of_radii_of_truncated_cone 
  (R r s : ℝ) 
  (h1 : s = Real.sqrt (R * r)) 
  (h2 : (π * (R^2 + r^2 + R * r) * (2 * s) / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = 7 := 
sorry

end ratio_of_radii_of_truncated_cone_l8_8205


namespace max_value_of_expression_l8_8978

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ m, m = 15 ∧ x^2 + y^2 + 2 * x ≤ m := 
sorry

end max_value_of_expression_l8_8978


namespace compute_expression_l8_8634

theorem compute_expression (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end compute_expression_l8_8634


namespace abigail_money_left_l8_8866

def initial_amount : ℕ := 11
def spent_in_store : ℕ := 2
def amount_lost : ℕ := 6

theorem abigail_money_left :
  initial_amount - spent_in_store - amount_lost = 3 := 
by {
  sorry
}

end abigail_money_left_l8_8866


namespace max_bio_homework_time_l8_8662

-- Define our variables as non-negative real numbers
variables (B H G : ℝ)

-- Given conditions
axiom h1 : H = 2 * B
axiom h2 : G = 6 * B
axiom h3 : B + H + G = 180

-- We need to prove that B = 20
theorem max_bio_homework_time : B = 20 :=
by
  sorry

end max_bio_homework_time_l8_8662


namespace no_such_fractions_l8_8305

open Nat

theorem no_such_fractions : ¬ ∃ (x y : ℕ), (x.gcd y = 1) ∧ (x > 0) ∧ (y > 0) ∧ ((x + 1) * 5 * y = ((y + 1) * 6 * x)) :=
by
  sorry

end no_such_fractions_l8_8305


namespace feeding_ways_correct_l8_8611

def total_feeding_ways : Nat :=
  (5 * 6 * (5 * 4 * 3 * 2 * 1)^2)

theorem feeding_ways_correct :
  total_feeding_ways = 432000 :=
by
  -- Proof is omitted here
  sorry

end feeding_ways_correct_l8_8611


namespace x_share_of_profit_l8_8398

-- Define the problem conditions
def investment_x : ℕ := 5000
def investment_y : ℕ := 15000
def total_profit : ℕ := 1600

-- Define the ratio simplification
def ratio_x : ℕ := 1
def ratio_y : ℕ := 3
def total_ratio_parts : ℕ := ratio_x + ratio_y

-- Define the profit division per part
def profit_per_part : ℕ := total_profit / total_ratio_parts

-- Lean 4 statement to prove
theorem x_share_of_profit : profit_per_part * ratio_x = 400 := sorry

end x_share_of_profit_l8_8398


namespace constant_term_exists_l8_8496

theorem constant_term_exists:
  ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 10 ∧ 
  (∃ r : ℕ, n = 3 * r) ∧ (∃ k : ℕ, n = 2 * k) ∧ 
  n = 6 :=
sorry

end constant_term_exists_l8_8496


namespace methane_combined_l8_8018

def balancedEquation (CH₄ O₂ CO₂ H₂O : ℕ) : Prop :=
  CH₄ = 1 ∧ O₂ = 2 ∧ CO₂ = 1 ∧ H₂O = 2

theorem methane_combined {moles_CH₄ moles_O₂ moles_H₂O : ℕ}
  (h₁ : moles_O₂ = 2)
  (h₂ : moles_H₂O = 2)
  (h_eq : balancedEquation moles_CH₄ moles_O₂ 1 moles_H₂O) : 
  moles_CH₄ = 1 :=
by
  sorry

end methane_combined_l8_8018


namespace sin_arith_seq_l8_8022

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end sin_arith_seq_l8_8022


namespace jesse_needs_more_carpet_l8_8406

def additional_carpet_needed (carpet : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  let room_area := length * width
  room_area - carpet

theorem jesse_needs_more_carpet
  (carpet : ℕ) (length : ℕ) (width : ℕ)
  (h_carpet : carpet = 18)
  (h_length : length = 4)
  (h_width : width = 20) :
  additional_carpet_needed carpet length width = 62 :=
by {
  -- the proof goes here
  sorry
}

end jesse_needs_more_carpet_l8_8406


namespace min_discount_70_percent_l8_8640

theorem min_discount_70_percent
  (P S : ℝ) (M : ℝ)
  (hP : P = 800)
  (hS : S = 1200)
  (hM : M = 0.05) :
  ∃ D : ℝ, D = 0.7 ∧ S * D - P ≥ P * M :=
by sorry

end min_discount_70_percent_l8_8640


namespace sqrt_arith_progression_impossible_l8_8231

theorem sqrt_arith_progression_impossible (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneab : a ≠ b) (hnebc : b ≠ c) (hneca : c ≠ a) :
  ¬ ∃ d : ℝ, (d = (Real.sqrt b - Real.sqrt a)) ∧ (d = (Real.sqrt c - Real.sqrt b)) :=
sorry

end sqrt_arith_progression_impossible_l8_8231


namespace find_three_digit_number_l8_8837

def digits_to_num (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem find_three_digit_number (a b c : ℕ) (h1 : 8 * a + 5 * b + c = 100) (h2 : a + b + c = 20) :
  digits_to_num a b c = 866 :=
by 
  sorry

end find_three_digit_number_l8_8837


namespace train_speed_in_km_per_hr_l8_8513

-- Conditions
def time_in_seconds : ℕ := 9
def length_in_meters : ℕ := 175

-- Conversion factor from m/s to km/hr
def meters_per_sec_to_km_per_hr (speed_m_per_s : ℚ) : ℚ :=
  speed_m_per_s * 3.6

-- Question as statement
theorem train_speed_in_km_per_hr :
  meters_per_sec_to_km_per_hr ((length_in_meters : ℚ) / (time_in_seconds : ℚ)) = 70 := by
  sorry

end train_speed_in_km_per_hr_l8_8513


namespace closing_price_l8_8144

theorem closing_price
  (opening_price : ℝ)
  (increase_percentage : ℝ)
  (h_opening_price : opening_price = 15)
  (h_increase_percentage : increase_percentage = 6.666666666666665) :
  opening_price * (1 + increase_percentage / 100) = 16 :=
by
  sorry

end closing_price_l8_8144


namespace termite_ridden_fraction_l8_8596

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end termite_ridden_fraction_l8_8596


namespace f_nonneg_f_positive_f_zero_condition_l8_8152

noncomputable def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) +
  B * (a^2 * b + b^2 * c + c^2 * a + a * b^2 + b * c^2 + c * a^2) +
  C * a * b * c

theorem f_nonneg (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 ≥ 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

theorem f_positive (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 > 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c > 0 :=
by sorry

theorem f_zero_condition (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 = 0) 
  (h2 : f A B C 1 1 0 > 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

end f_nonneg_f_positive_f_zero_condition_l8_8152


namespace circumscribed_quadrilateral_arc_sum_l8_8038

theorem circumscribed_quadrilateral_arc_sum 
  (a b c d : ℝ) 
  (h : a + b + c + d = 360) : 
  (1/2 * (b + c + d)) + (1/2 * (a + c + d)) + (1/2 * (a + b + d)) + (1/2 * (a + b + c)) = 540 :=
by
  sorry

end circumscribed_quadrilateral_arc_sum_l8_8038


namespace discounted_price_correct_l8_8841

def discounted_price (P : ℝ) : ℝ :=
  P * 0.80 * 0.90 * 0.95

theorem discounted_price_correct :
  discounted_price 9502.923976608186 = 6498.40 :=
by
  sorry

end discounted_price_correct_l8_8841


namespace Angle_CNB_20_l8_8157

theorem Angle_CNB_20 :
  ∀ (A B C N : Type) 
    (AC BC : Prop) 
    (angle_ACB : ℕ)
    (angle_NAC : ℕ)
    (angle_NCA : ℕ), 
    (AC ↔ BC) →
    angle_ACB = 98 →
    angle_NAC = 15 →
    angle_NCA = 21 →
    ∃ angle_CNB, angle_CNB = 20 :=
by
  sorry

end Angle_CNB_20_l8_8157


namespace six_people_acquaintance_or_strangers_l8_8893

theorem six_people_acquaintance_or_strangers (p : Fin 6 → Prop) :
  ∃ (A B C : Fin 6), (p A ∧ p B ∧ p C) ∨ (¬p A ∧ ¬p B ∧ ¬p C) :=
sorry

end six_people_acquaintance_or_strangers_l8_8893


namespace subset_relation_l8_8932

variables (M N : Set ℕ) 

theorem subset_relation (hM : M = {1, 2, 3, 4}) (hN : N = {2, 3, 4}) : N ⊆ M :=
sorry

end subset_relation_l8_8932


namespace work_completion_days_l8_8630

structure WorkProblem :=
  (total_work : ℝ := 1) -- Assume total work to be 1 unit
  (days_A : ℝ := 30)
  (days_B : ℝ := 15)
  (days_together : ℝ := 5)

noncomputable def total_days_taken (wp : WorkProblem) : ℝ :=
  let work_per_day_A := 1 / wp.days_A
  let work_per_day_B := 1 / wp.days_B
  let work_per_day_together := work_per_day_A + work_per_day_B
  let work_done_together := wp.days_together * work_per_day_together
  let remaining_work := wp.total_work - work_done_together
  let days_for_A := remaining_work / work_per_day_A
  wp.days_together + days_for_A

theorem work_completion_days (wp : WorkProblem) : total_days_taken wp = 20 :=
by
  sorry

end work_completion_days_l8_8630


namespace bus_stops_per_hour_l8_8010

-- Define the constants and conditions given in the problem
noncomputable def speed_without_stoppages : ℝ := 54 -- km/hr
noncomputable def speed_with_stoppages : ℝ := 45 -- km/hr

-- Theorem statement to prove the number of minutes the bus stops per hour
theorem bus_stops_per_hour : (speed_without_stoppages - speed_with_stoppages) / (speed_without_stoppages / 60) = 10 :=
by
  sorry

end bus_stops_per_hour_l8_8010


namespace number_of_arrangements_l8_8491

-- Definitions of the problem's conditions
def student_set : Finset ℕ := {1, 2, 3, 4, 5}

def specific_students : Finset ℕ := {1, 2}

def remaining_students : Finset ℕ := student_set \ specific_students

-- Formalize the problem statement
theorem number_of_arrangements : 
  ∀ (students : Finset ℕ) 
    (specific : Finset ℕ) 
    (remaining : Finset ℕ),
    students = student_set →
    specific = specific_students →
    remaining = remaining_students →
    (specific.card = 2 ∧ students.card = 5 ∧ remaining.card = 3) →
    (∃ (n : ℕ), n = 12) :=
by
  intros
  sorry

end number_of_arrangements_l8_8491


namespace sum_of_integers_l8_8173

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l8_8173


namespace work_together_l8_8079

theorem work_together (A_days B_days : ℕ) (hA : A_days = 8) (hB : B_days = 4)
  (A_work : ℚ := 1 / A_days)
  (B_work : ℚ := 1 / B_days) :
  (A_work + B_work = 3 / 8) :=
by
  rw [hA, hB]
  sorry

end work_together_l8_8079


namespace distinct_exponentiation_values_l8_8452

theorem distinct_exponentiation_values : 
  let a := 3^(3^(3^3))
  let b := 3^((3^3)^3)
  let c := ((3^3)^3)^3
  let d := 3^((3^3)^(3^2))
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → 
  ∃ n, n = 3 := 
sorry

end distinct_exponentiation_values_l8_8452


namespace find_f2_l8_8785

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l8_8785


namespace neq_zero_necessary_not_sufficient_l8_8375

theorem neq_zero_necessary_not_sufficient (x : ℝ) (h : x ≠ 0) : 
  (¬ (x = 0) ↔ x > 0) ∧ ¬ (x > 0 → x ≠ 0) :=
by sorry

end neq_zero_necessary_not_sufficient_l8_8375


namespace marie_initial_erasers_l8_8945

def erasers_problem : Prop :=
  ∃ initial_erasers : ℝ, initial_erasers + 42.0 = 137

theorem marie_initial_erasers : erasers_problem :=
  sorry

end marie_initial_erasers_l8_8945


namespace frequency_of_group_5_l8_8912

/-- Let the total number of data points be 50, number of data points in groups 1, 2, 3, and 4 be
  2, 8, 15, and 5 respectively. Prove that the frequency of group 5 is 0.4. -/
theorem frequency_of_group_5 :
  let total_data_points := 50
  let group1_data_points := 2
  let group2_data_points := 8
  let group3_data_points := 15
  let group4_data_points := 5
  let group5_data_points := total_data_points - group1_data_points - group2_data_points - group3_data_points - group4_data_points
  let frequency_group5 := (group5_data_points : ℝ) / total_data_points
  frequency_group5 = 0.4 := 
by
  sorry

end frequency_of_group_5_l8_8912


namespace candies_per_person_l8_8871

def clowns : ℕ := 4
def children : ℕ := 30
def initial_candies : ℕ := 700
def candies_left : ℕ := 20

def total_people : ℕ := clowns + children
def candies_sold : ℕ := initial_candies - candies_left

theorem candies_per_person : candies_sold / total_people = 20 := by
  sorry

end candies_per_person_l8_8871


namespace parabola_passes_through_points_and_has_solution_4_l8_8540

theorem parabola_passes_through_points_and_has_solution_4 
  (a h k m: ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k → 
    (y = 0 → (x = -1 → x = 5))) → 
  (∃ m, ∀ x, (a * (x - h + m) ^ 2 + k = 0) → x = 4) → 
  m = -5 ∨ m = 1 :=
sorry

end parabola_passes_through_points_and_has_solution_4_l8_8540


namespace sample_size_l8_8054

theorem sample_size 
  (n_A n_B n_C : ℕ)
  (h1 : n_A = 15)
  (h2 : 3 * n_B = 4 * n_A)
  (h3 : 3 * n_C = 7 * n_A) :
  n_A + n_B + n_C = 70 :=
by
sorry

end sample_size_l8_8054


namespace museum_paintings_discarded_l8_8988

def initial_paintings : ℕ := 2500
def percentage_to_discard : ℝ := 0.35
def paintings_discarded : ℝ := initial_paintings * percentage_to_discard

theorem museum_paintings_discarded : paintings_discarded = 875 :=
by
  -- Lean automatically simplifies this using basic arithmetic rules
  sorry

end museum_paintings_discarded_l8_8988


namespace proportion_estimation_chi_squared_test_l8_8174

-- Definitions based on the conditions
def total_elders : ℕ := 500
def not_vaccinated_male : ℕ := 20
def not_vaccinated_female : ℕ := 10
def vaccinated_male : ℕ := 230
def vaccinated_female : ℕ := 240

-- Calculations based on the problem conditions
noncomputable def proportion_vaccinated : ℚ := (vaccinated_male + vaccinated_female) / total_elders

def chi_squared_statistic (a b c d n : ℕ) : ℚ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

noncomputable def K2_value : ℚ :=
  chi_squared_statistic not_vaccinated_male not_vaccinated_female vaccinated_male vaccinated_female total_elders

-- Specify the critical value for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Theorem statements (problems to prove)
theorem proportion_estimation : proportion_vaccinated = 94 / 100 := by
  sorry

theorem chi_squared_test : K2_value < critical_value_99 := by
  sorry

end proportion_estimation_chi_squared_test_l8_8174


namespace arithmetic_mean_value_of_x_l8_8790

theorem arithmetic_mean_value_of_x (x : ℝ) (h : (x + 10 + 20 + 3 * x + 16 + 3 * x + 6) / 5 = 30) : x = 14 := 
by 
  sorry

end arithmetic_mean_value_of_x_l8_8790


namespace LeonaEarnsGivenHourlyRate_l8_8362

theorem LeonaEarnsGivenHourlyRate :
  (∀ (c: ℝ) (t h e: ℝ), 
    (c = 24.75) → 
    (t = 3) → 
    (h = c / t) → 
    (e = h * 5) →
    e = 41.25) :=
by
  intros c t h e h1 h2 h3 h4
  sorry

end LeonaEarnsGivenHourlyRate_l8_8362


namespace max_abc_l8_8947

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l8_8947


namespace arithmetic_sequence_geometric_ratio_l8_8876

theorem arithmetic_sequence_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n : ℕ, a (n+1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (h_geo : (a 2) * (a 9) = (a 3) ^ 2)
  : (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = (8 / 3) :=
by
  sorry

end arithmetic_sequence_geometric_ratio_l8_8876


namespace not_always_possible_triangle_sides_l8_8071

theorem not_always_possible_triangle_sides (α β γ δ : ℝ) 
  (h1 : α + β + γ + δ = 360) 
  (h2 : α < 180) 
  (h3 : β < 180) 
  (h4 : γ < 180) 
  (h5 : δ < 180) : 
  ¬ (∀ (x y z : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) ∧ (y = α ∨ y = β ∨ y = γ ∨ y = δ) ∧ (z = α ∨ z = β ∨ z = γ ∨ z = δ) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) → x + y > z ∧ x + z > y ∧ y + z > x)
:= sorry

end not_always_possible_triangle_sides_l8_8071


namespace Ryan_learning_days_l8_8483

theorem Ryan_learning_days
  (hours_english_per_day : ℕ)
  (hours_chinese_per_day : ℕ)
  (total_hours : ℕ)
  (h1 : hours_english_per_day = 6)
  (h2 : hours_chinese_per_day = 7)
  (h3 : total_hours = 65) :
  total_hours / (hours_english_per_day + hours_chinese_per_day) = 5 := by
  sorry

end Ryan_learning_days_l8_8483


namespace derivative_of_f_l8_8443

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) : (deriv f x) = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by
  -- This statement skips the proof details
  sorry

end derivative_of_f_l8_8443


namespace daisies_multiple_of_4_l8_8199

def num_roses := 8
def num_daisies (D : ℕ) := D
def num_marigolds := 48
def num_arrangements := 4

theorem daisies_multiple_of_4 (D : ℕ) 
  (h_roses_div_4 : num_roses % num_arrangements = 0)
  (h_marigolds_div_4 : num_marigolds % num_arrangements = 0)
  (h_total_div_4 : (num_roses + num_daisies D + num_marigolds) % num_arrangements = 0) :
  D % 4 = 0 :=
sorry

end daisies_multiple_of_4_l8_8199


namespace bacteria_population_at_2_15_l8_8279

noncomputable def bacteria_at_time (initial_pop : ℕ) (start_time end_time : ℕ) (interval : ℕ) : ℕ :=
  initial_pop * 2 ^ ((end_time - start_time) / interval)

theorem bacteria_population_at_2_15 :
  let initial_pop := 50
  let start_time := 0  -- 2:00 p.m.
  let end_time := 15   -- 2:15 p.m.
  let interval := 4
  bacteria_at_time initial_pop start_time end_time interval = 400 := sorry

end bacteria_population_at_2_15_l8_8279


namespace distances_inequality_l8_8353

theorem distances_inequality (x y z : ℝ) (h : x + y + z = 1): x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6 * x * y * z :=
by
  sorry

end distances_inequality_l8_8353


namespace percent_change_is_minus_5_point_5_percent_l8_8849

noncomputable def overall_percent_change (initial_value : ℝ) : ℝ :=
  let day1_value := initial_value * 0.75
  let day2_value := day1_value * 1.4
  let final_value := day2_value * 0.9
  ((final_value / initial_value) - 1) * 100

theorem percent_change_is_minus_5_point_5_percent :
  ∀ (initial_value : ℝ), overall_percent_change initial_value = -5.5 :=
sorry

end percent_change_is_minus_5_point_5_percent_l8_8849


namespace total_apples_eaten_l8_8633

def simone_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day
def lauri_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day

theorem total_apples_eaten :
  simone_consumption 16 (1/2) + lauri_consumption 15 (1/3) = 13 := by
  sorry

end total_apples_eaten_l8_8633


namespace cost_per_liter_of_fuel_l8_8836

-- Definitions and conditions
def fuel_capacity : ℕ := 150
def initial_fuel : ℕ := 38
def change_received : ℕ := 14
def initial_money : ℕ := 350

-- Proof problem
theorem cost_per_liter_of_fuel :
  (initial_money - change_received) / (fuel_capacity - initial_fuel) = 3 :=
by
  sorry

end cost_per_liter_of_fuel_l8_8836


namespace natural_numbers_divisible_by_6_l8_8783

theorem natural_numbers_divisible_by_6 :
  {n : ℕ | 2 ≤ n ∧ n ≤ 88 ∧ 6 ∣ n} = {n | n = 6 * k ∧ 1 ≤ k ∧ k ≤ 14} :=
by
  sorry

end natural_numbers_divisible_by_6_l8_8783


namespace mushroom_mistake_l8_8797

theorem mushroom_mistake (p k v : ℝ) (hk : k = p + v - 10) (hp : p = k + v - 7) : 
  ∃ p k : ℝ, ∀ v : ℝ, (p = k + v - 7) ∧ (k = p + v - 10) → false :=
by
  sorry

end mushroom_mistake_l8_8797


namespace simplify_expression_l8_8366

theorem simplify_expression (x : ℝ) :
  (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 :=
by
  sorry

end simplify_expression_l8_8366


namespace diagonal_of_square_l8_8728

theorem diagonal_of_square (s d : ℝ) (h_perimeter : 4 * s = 40) : d = 10 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_square_l8_8728


namespace cost_equivalence_min_sets_of_A_l8_8387

noncomputable def cost_of_B := 120
noncomputable def cost_of_A := cost_of_B + 30

theorem cost_equivalence (x : ℕ) :
  (1200 / (x + 30) = 960 / x) → x = 120 :=
by
  sorry

theorem min_sets_of_A :
  ∀ m : ℕ, (150 * m + 120 * (20 - m) ≥ 2800) ↔ m ≥ 14 :=
by
  sorry

end cost_equivalence_min_sets_of_A_l8_8387


namespace Yvonne_probability_of_success_l8_8189

theorem Yvonne_probability_of_success
  (P_X : ℝ) (P_Z : ℝ) (P_XY_notZ : ℝ) :
  P_X = 1 / 3 →
  P_Z = 5 / 8 →
  P_XY_notZ = 0.0625 →
  ∃ P_Y : ℝ, P_Y = 0.5 :=
by
  intros hX hZ hXY_notZ
  existsi (0.5 : ℝ)
  sorry

end Yvonne_probability_of_success_l8_8189


namespace team_A_minimum_workers_l8_8359

-- Define the variables and conditions for the problem.
variables (A B c : ℕ)

-- Condition 1: If team A lends 90 workers to team B, Team B will have twice as many workers as Team A.
def condition1 : Prop :=
  2 * (A - 90) = B + 90

-- Condition 2: If team B lends c workers to team A, Team A will have six times as many workers as Team B.
def condition2 : Prop :=
  A + c = 6 * (B - c)

-- Define the proof goal.
theorem team_A_minimum_workers (h1 : condition1 A B) (h2 : condition2 A B c) : 
  153 ≤ A :=
sorry

end team_A_minimum_workers_l8_8359


namespace fewest_number_of_students_l8_8858

theorem fewest_number_of_students :
  ∃ n : ℕ, n ≡ 3 [MOD 6] ∧ n ≡ 5 [MOD 8] ∧ n ≡ 7 [MOD 9] ∧ ∀ m : ℕ, (m ≡ 3 [MOD 6] ∧ m ≡ 5 [MOD 8] ∧ m ≡ 7 [MOD 9]) → m ≥ n := by
  sorry

end fewest_number_of_students_l8_8858


namespace points_on_line_sufficient_but_not_necessary_l8_8414

open Nat

-- Define the sequence a_n
def sequence_a (n : ℕ) : ℕ := n + 1

-- Define a general arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, n < m → a (m) - a (n) = (m - n) * (a 1 - a 0)

-- Define the condition that points (n, a_n), where n is a natural number, lie on the line y = x + 1
def points_on_line (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n) = n + 1

-- Prove that points_on_line is sufficient but not necessary for is_arithmetic_sequence
theorem points_on_line_sufficient_but_not_necessary :
  (∀ a : ℕ → ℕ, points_on_line a → is_arithmetic_sequence a)
  ∧ ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ ¬ points_on_line a := 
by 
  sorry

end points_on_line_sufficient_but_not_necessary_l8_8414


namespace range_of_m_l8_8726

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → x^2 - 2 * x - 3 > 0) → (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l8_8726


namespace find_angle_D_l8_8383

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : ∃ B_adj, B_adj = 60 ∧ A + B_adj + B = 180) : D = 25 :=
sorry

end find_angle_D_l8_8383


namespace lcm_1230_924_l8_8082

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 :=
by
  /- Proof steps skipped -/
  sorry

end lcm_1230_924_l8_8082


namespace find_x_squared_minus_one_l8_8259

theorem find_x_squared_minus_one (x : ℕ) 
  (h : 2^x + 2^x + 2^x + 2^x = 256) : 
  x^2 - 1 = 35 :=
sorry

end find_x_squared_minus_one_l8_8259


namespace order_of_abc_l8_8020

theorem order_of_abc (a b c : ℝ) (h1 : a = 16 ^ (1 / 3))
                                 (h2 : b = 2 ^ (4 / 5))
                                 (h3 : c = 5 ^ (2 / 3)) :
  c > a ∧ a > b :=
by {
  sorry
}

end order_of_abc_l8_8020


namespace acute_triangle_angles_l8_8311

theorem acute_triangle_angles (x y z : ℕ) (angle1 angle2 angle3 : ℕ) 
  (h1 : angle1 = 7 * x) 
  (h2 : angle2 = 9 * y) 
  (h3 : angle3 = 11 * z) 
  (h4 : angle1 + angle2 + angle3 = 180)
  (hx : 1 ≤ x ∧ x ≤ 12)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (hz : 1 ≤ z ∧ z ≤ 8)
  (ha1 : angle1 < 90)
  (ha2 : angle2 < 90)
  (ha3 : angle3 < 90)
  : angle1 = 42 ∧ angle2 = 72 ∧ angle3 = 66 
  ∨ angle1 = 49 ∧ angle2 = 54 ∧ angle3 = 77 
  ∨ angle1 = 56 ∧ angle2 = 36 ∧ angle3 = 88 
  ∨ angle1 = 84 ∧ angle2 = 63 ∧ angle3 = 33 :=
sorry

end acute_triangle_angles_l8_8311


namespace not_in_second_column_l8_8741

theorem not_in_second_column : ¬∃ (n : ℕ), (1 ≤ n ∧ n ≤ 400) ∧ 3 * n + 1 = 131 :=
by sorry

end not_in_second_column_l8_8741


namespace fred_initial_balloons_l8_8844

def green_balloons_initial (given: Nat) (left: Nat) : Nat := 
  given + left

theorem fred_initial_balloons : green_balloons_initial 221 488 = 709 :=
by
  sorry

end fred_initial_balloons_l8_8844


namespace rows_with_exactly_10_people_l8_8843

theorem rows_with_exactly_10_people (x : ℕ) (total_people : ℕ) (row_nine_seat : ℕ) (row_ten_seat : ℕ) 
    (H1 : row_nine_seat = 9) (H2 : row_ten_seat = 10) 
    (H3 : total_people = 55) 
    (H4 : total_people = x * row_ten_seat + (6 - x) * row_nine_seat) 
    : x = 1 :=
by
  sorry

end rows_with_exactly_10_people_l8_8843


namespace jack_pages_l8_8575

theorem jack_pages (pages_per_booklet : ℕ) (num_booklets : ℕ) (h1 : pages_per_booklet = 9) (h2 : num_booklets = 49) : num_booklets * pages_per_booklet = 441 :=
by {
  sorry
}

end jack_pages_l8_8575


namespace range_of_a_l8_8164

-- Definitions of position conditions in the 4th quadrant
def PosInFourthQuad (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- Statement to prove
theorem range_of_a (a : ℝ) (h : PosInFourthQuad (2 * a + 4) (3 * a - 6)) : -2 < a ∧ a < 2 :=
  sorry

end range_of_a_l8_8164


namespace percent_defective_shipped_l8_8190

-- Conditions given in the problem
def percent_defective (percent_total_defective: ℝ) : Prop := percent_total_defective = 0.08
def percent_shipped_defective (percent_defective_shipped: ℝ) : Prop := percent_defective_shipped = 0.04

-- The main theorem we want to prove
theorem percent_defective_shipped (percent_total_defective percent_defective_shipped : ℝ) 
  (h1 : percent_defective percent_total_defective) (h2 : percent_shipped_defective percent_defective_shipped) : 
  (percent_total_defective * percent_defective_shipped * 100) = 0.32 :=
by
  sorry

end percent_defective_shipped_l8_8190


namespace inequality_solution_l8_8646

theorem inequality_solution (x : ℝ) (h : x ≠ 0) : 
  (1 / (x^2 + 1) > 2 * x^2 / x + 13 / 10) ↔ (x ∈ Set.Ioo (-1.6) 0 ∨ x ∈ Set.Ioi 0.8) :=
by sorry

end inequality_solution_l8_8646


namespace male_athletes_drawn_l8_8692

theorem male_athletes_drawn (total_males : ℕ) (total_females : ℕ) (total_sample : ℕ)
  (h_males : total_males = 20) (h_females : total_females = 10) (h_sample : total_sample = 6) :
  (total_sample * total_males) / (total_males + total_females) = 4 := 
  by
  sorry

end male_athletes_drawn_l8_8692


namespace lcm_of_two_numbers_l8_8759

theorem lcm_of_two_numbers (A B : ℕ) (h1 : A * B = 62216) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2828 :=
by
  sorry

end lcm_of_two_numbers_l8_8759


namespace S6_is_48_l8_8106

-- Define the first term and common difference
def a₁ : ℕ := 3
def d : ℕ := 2

-- Define the formula for sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Prove that the sum of the first 6 terms is 48
theorem S6_is_48 : sum_of_arithmetic_sequence 6 = 48 := by
  sorry

end S6_is_48_l8_8106


namespace minimum_shirts_for_saving_money_l8_8324

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money_l8_8324


namespace total_votes_cast_l8_8516

/-- Define the conditions for Elvis's votes and percentage representation -/
def elvis_votes : ℕ := 45
def percentage_representation : ℚ := 1 / 4

/-- The main theorem that proves the total number of votes cast -/
theorem total_votes_cast : (elvis_votes: ℚ) / percentage_representation = 180 := by
  sorry

end total_votes_cast_l8_8516


namespace right_angled_triangles_with_cathetus_2021_l8_8746

theorem right_angled_triangles_with_cathetus_2021 :
  ∃ n : Nat, n = 4 ∧ ∀ (a b c : ℕ), ((a = 2021 ∧ a * a + b * b = c * c) ↔ (a = 2021 ∧ 
    ∃ m n, (m > n ∧ m > 0 ∧ n > 0 ∧ 2021 = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2))) :=
sorry

end right_angled_triangles_with_cathetus_2021_l8_8746


namespace two_mul_seven_pow_n_plus_one_divisible_by_three_l8_8760

-- Definition of natural numbers
variable (n : ℕ)

-- Statement of the problem in Lean
theorem two_mul_seven_pow_n_plus_one_divisible_by_three (n : ℕ) : 3 ∣ (2 * 7^n + 1) := 
sorry

end two_mul_seven_pow_n_plus_one_divisible_by_three_l8_8760


namespace sum_of_g1_l8_8610

noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition : ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y := sorry

theorem sum_of_g1 : g 1 = 1 := 
by
  -- Provide the necessary proof steps to show g(1) = 1
  sorry

end sum_of_g1_l8_8610


namespace butcher_net_loss_l8_8275

noncomputable def dishonest_butcher (advertised_price actual_price : ℝ) (quantity_sold : ℕ) (fine : ℝ) : ℝ :=
  let dishonest_gain_per_kg := actual_price - advertised_price
  let total_dishonest_gain := dishonest_gain_per_kg * quantity_sold
  fine - total_dishonest_gain

theorem butcher_net_loss 
  (advertised_price : ℝ) 
  (actual_price : ℝ) 
  (quantity_sold : ℕ) 
  (fine : ℝ)
  (h_advertised_price : advertised_price = 3.79)
  (h_actual_price : actual_price = 4.00)
  (h_quantity_sold : quantity_sold = 1800)
  (h_fine : fine = 500) :
  dishonest_butcher advertised_price actual_price quantity_sold fine = 122 := 
by
  simp [dishonest_butcher, h_advertised_price, h_actual_price, h_quantity_sold, h_fine]
  sorry

end butcher_net_loss_l8_8275


namespace problem1_problem2_l8_8526

-- Definitions for the number of combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1
theorem problem1 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 4) + (C r 3 * C w 1) + (C r 2 * C w 2) = 115 := 
sorry

-- Problem 2
theorem problem2 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 2 * C w 3) + (C r 3 * C w 2) + (C r 4 * C w 1) = 186 := 
sorry

end problem1_problem2_l8_8526


namespace number_of_cities_from_group_B_l8_8109

theorem number_of_cities_from_group_B
  (total_cities : ℕ)
  (cities_in_A : ℕ)
  (cities_in_B : ℕ)
  (cities_in_C : ℕ)
  (sampled_cities : ℕ)
  (h1 : total_cities = cities_in_A + cities_in_B + cities_in_C)
  (h2 : total_cities = 24)
  (h3 : cities_in_A = 4)
  (h4 : cities_in_B = 12)
  (h5 : cities_in_C = 8)
  (h6 : sampled_cities = 6) :
  cities_in_B * sampled_cities / total_cities = 3 := 
  by 
    sorry

end number_of_cities_from_group_B_l8_8109


namespace x_squared_plus_y_squared_l8_8715

theorem x_squared_plus_y_squared (x y : ℝ) 
   (h1 : (x + y)^2 = 49) 
   (h2 : x * y = 8) 
   : x^2 + y^2 = 33 := 
by
  sorry

end x_squared_plus_y_squared_l8_8715


namespace John_and_Rose_work_together_l8_8253

theorem John_and_Rose_work_together (John_work_days : ℕ) (Rose_work_days : ℕ) (combined_work_days: ℕ) 
  (hJohn : John_work_days = 10) (hRose : Rose_work_days = 40) :
  combined_work_days = 8 :=
by 
  sorry

end John_and_Rose_work_together_l8_8253


namespace find_c_l8_8636

-- Given that the function f(x) = 2^x + c passes through the point (2,5),
-- Prove that c = 1.
theorem find_c (c : ℝ) : (∃ (f : ℝ → ℝ), (∀ x, f x = 2^x + c) ∧ (f 2 = 5)) → c = 1 := by
  sorry

end find_c_l8_8636


namespace tournament_participants_l8_8163

theorem tournament_participants (n : ℕ) (h₁ : 2 * (n * (n - 1) / 2 + 4) - (n - 2) * (n - 3) - 16 = 124) : n = 13 :=
sorry

end tournament_participants_l8_8163


namespace correct_number_of_true_propositions_l8_8593

noncomputable def true_proposition_count : ℕ := 1

theorem correct_number_of_true_propositions (a b c : ℝ) :
    (∀ a b : ℝ, (a > b) ↔ (a^2 > b^2) = false) →
    (∀ a b : ℝ, (a > b) ↔ (a^3 > b^3) = true) →
    (∀ a b : ℝ, (a > b) → (|a| > |b|) = false) →
    (∀ a b c : ℝ, (a > b) → (a*c^2 ≤ b*c^2) = false) →
    (true_proposition_count = 1) :=
by
  sorry

end correct_number_of_true_propositions_l8_8593


namespace solve_medium_apple_cost_l8_8765

def cost_small_apple : ℝ := 1.5
def cost_big_apple : ℝ := 3.0
def num_small_apples : ℕ := 6
def num_medium_apples : ℕ := 6
def num_big_apples : ℕ := 8
def total_cost : ℝ := 45

noncomputable def cost_medium_apple (M : ℝ) : Prop :=
  (6 * cost_small_apple) + (6 * M) + (8 * cost_big_apple) = total_cost

theorem solve_medium_apple_cost : ∃ M : ℝ, cost_medium_apple M ∧ M = 2 := by
  sorry

end solve_medium_apple_cost_l8_8765


namespace find_a_b_transform_line_l8_8714

theorem find_a_b_transform_line (a b : ℝ) (hA : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, a], ![b, 3]]) :
  (∀ x y : ℝ, (2 * (-(x) + a*y) - (b*x + 3*y) - 3 = 0) → (2*x - y - 3 = 0)) →
  a = 1 ∧ b = -4 :=
by {
  sorry
}

end find_a_b_transform_line_l8_8714


namespace persimmons_count_l8_8312

variables {P T : ℕ}

-- Conditions from the problem
axiom total_eq : P + T = 129
axiom diff_eq : P = T - 43

-- Theorem to prove that there are 43 persimmons
theorem persimmons_count : P = 43 :=
by
  -- Putting the proof placeholder
  sorry

end persimmons_count_l8_8312


namespace sale_in_fifth_month_l8_8598

theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 : ℕ)
  (avg : ℕ)
  (h1 : s1 = 5435)
  (h2 : s2 = 5927)
  (h3 : s3 = 5855)
  (h4 : s4 = 6230)
  (h6 : s6 = 3991)
  (hav : avg = 5500) :
  ∃ s5 : ℕ, s1 + s2 + s3 + s4 + s5 + s6 = avg * 6 ∧ s5 = 5562 := 
by
  sorry

end sale_in_fifth_month_l8_8598


namespace tan_sin_difference_l8_8624

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l8_8624


namespace last_four_digits_of_5_pow_2016_l8_8327

theorem last_four_digits_of_5_pow_2016 :
  (5^2016) % 10000 = 625 :=
by
  -- Establish periodicity of last four digits in powers of 5
  sorry

end last_four_digits_of_5_pow_2016_l8_8327


namespace sum_of_money_l8_8282

theorem sum_of_money (x : ℝ)
  (hC : 0.50 * x = 64)
  (hB : ∀ x, B_shares = 0.75 * x)
  (hD : ∀ x, D_shares = 0.25 * x) :
  let total_sum := x + 0.75 * x + 0.50 * x + 0.25 * x
  total_sum = 320 :=
by
  sorry

end sum_of_money_l8_8282


namespace probability_same_plane_l8_8425

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l8_8425


namespace same_exponent_for_all_bases_l8_8298

theorem same_exponent_for_all_bases {a : Type} [LinearOrderedField a] {C : a} (ha : ∀ (a : a), a ≠ 0 → a^0 = C) : C = 1 :=
by
  sorry

end same_exponent_for_all_bases_l8_8298


namespace contrapositive_l8_8296

theorem contrapositive (p q : Prop) (h : p → q) : ¬q → ¬p :=
by
  sorry

end contrapositive_l8_8296


namespace combine_fraction_l8_8385

variable (d : ℤ)

theorem combine_fraction : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end combine_fraction_l8_8385


namespace num_tosses_l8_8958

theorem num_tosses (n : ℕ) (h : (1 - (7 / 8 : ℝ)^n) = 0.111328125) : n = 7 :=
by
  sorry

end num_tosses_l8_8958


namespace crow_distance_l8_8698

theorem crow_distance (trips: ℕ) (hours: ℝ) (speed: ℝ) (distance: ℝ) :
  trips = 15 → hours = 1.5 → speed = 4 → (trips * 2 * distance) = (speed * hours) → distance = 200 / 1000 :=
by
  intros h_trips h_hours h_speed h_eq
  sorry

end crow_distance_l8_8698


namespace greatest_int_satisfying_inequality_l8_8484

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l8_8484


namespace determine_a_l8_8807

noncomputable def f (x a : ℝ) := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem determine_a (a : ℝ) 
  (h₁ : ∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f x a ≤ f 0 a)
  (h₂ : f 0 a = -3) :
  a = 2 + Real.sqrt 6 := 
sorry

end determine_a_l8_8807


namespace original_price_l8_8440

variables (q r : ℝ) (h1 : 0 ≤ q) (h2 : 0 ≤ r)

theorem original_price (h : (2 : ℝ) = (1 + q / 100) * (1 - r / 100) * x) :
  x = 200 / (100 + q - r - (q * r) / 100) :=
by
  sorry

end original_price_l8_8440


namespace total_trees_after_planting_l8_8132

theorem total_trees_after_planting
  (initial_walnut_trees : ℕ) (initial_oak_trees : ℕ) (initial_maple_trees : ℕ)
  (plant_walnut_trees : ℕ) (plant_oak_trees : ℕ) (plant_maple_trees : ℕ) :
  (initial_walnut_trees = 107) →
  (initial_oak_trees = 65) →
  (initial_maple_trees = 32) →
  (plant_walnut_trees = 104) →
  (plant_oak_trees = 79) →
  (plant_maple_trees = 46) →
  initial_walnut_trees + plant_walnut_trees +
  initial_oak_trees + plant_oak_trees +
  initial_maple_trees + plant_maple_trees = 433 :=
by
  intros
  sorry

end total_trees_after_planting_l8_8132


namespace Elle_in_seat_2_given_conditions_l8_8503

theorem Elle_in_seat_2_given_conditions
    (seats : Fin 4 → Type) -- Representation of the seating arrangement.
    (Garry Elle Fiona Hank : Type)
    (seat_of : Type → Fin 4)
    (h1 : seat_of Garry = 0) -- Garry is in seat #1 (index 0)
    (h2 : ¬ (seat_of Elle = seat_of Hank + 1 ∨ seat_of Elle = seat_of Hank - 1)) -- Elle is not next to Hank
    (h3 : ¬ (seat_of Fiona > seat_of Garry ∧ seat_of Fiona < seat_of Hank) ∧ ¬ (seat_of Fiona < seat_of Garry ∧ seat_of Fiona > seat_of Hank)) -- Fiona is not between Garry and Hank
    : seat_of Elle = 1 :=  -- Conclusion: Elle is in seat #2 (index 1)
    sorry

end Elle_in_seat_2_given_conditions_l8_8503


namespace f_increasing_on_Ioo_l8_8377

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_on_Ioo : ∀ x y : ℝ, x < y → f x < f y :=
sorry

end f_increasing_on_Ioo_l8_8377


namespace width_of_vessel_is_5_l8_8538

open Real

noncomputable def width_of_vessel : ℝ :=
  let edge := 5
  let rise := 2.5
  let base_length := 10
  let volume_cube := edge ^ 3
  let volume_displaced := volume_cube
  let width := volume_displaced / (base_length * rise)
  width

theorem width_of_vessel_is_5 :
  width_of_vessel = 5 := by
    sorry

end width_of_vessel_is_5_l8_8538


namespace ellipse_eccentricity_l8_8422

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = (4/3) * c) (h4 : a^2 - b^2 = c^2) : 
  c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l8_8422


namespace area_of_PQRSUV_proof_l8_8317

noncomputable def PQRSW_area (PQ QR RS SW : ℝ) : ℝ :=
  (1 / 2) * PQ * QR + (1 / 2) * (RS + SW) * 5

noncomputable def WUV_area (WU UV : ℝ) : ℝ :=
  WU * UV

theorem area_of_PQRSUV_proof 
  (PQ QR RS SW WU UV : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 5) (hRS : RS = 7) (hSW : SW = 10)
  (hWU : WU = 6) (hUV : UV = 7) :
  PQRSW_area PQ QR RS SW + WUV_area WU UV = 147 :=
by
  simp only [PQRSW_area, WUV_area, hPQ, hQR, hRS, hSW, hWU, hUV]
  norm_num
  sorry

end area_of_PQRSUV_proof_l8_8317


namespace max_type_A_pieces_max_profit_l8_8457

noncomputable def type_A_cost := 80
noncomputable def type_A_sell := 120
noncomputable def type_B_cost := 60
noncomputable def type_B_sell := 90
noncomputable def total_clothes := 100
noncomputable def min_type_A := 65
noncomputable def max_cost := 7500

/-- The maximum number of type A clothing pieces that can be purchased --/
theorem max_type_A_pieces (x : ℕ) : 
  type_A_cost * x + type_B_cost * (total_clothes - x) ≤ max_cost → 
  x ≤ 75 := by 
sorry

variable (a : ℝ) (h_a : 0 < a ∧ a < 10)

/-- The optimal purchase strategy to maximize profit --/
theorem max_profit (x y : ℕ) : 
  (x + y = total_clothes) ∧ 
  (type_A_cost * x + type_B_cost * y ≤ max_cost) ∧
  (min_type_A ≤ x) ∧ 
  (x ≤ 75) → 
  (type_A_sell - type_A_cost - a) * x + (type_B_sell - type_B_cost) * y 
  ≤ (type_A_sell - type_A_cost - a) * 75 + (type_B_sell - type_B_cost) * 25 := by 
sorry

end max_type_A_pieces_max_profit_l8_8457


namespace candy_left_l8_8775

theorem candy_left (d : ℕ) (s : ℕ) (ate : ℕ) (h_d : d = 32) (h_s : s = 42) (h_ate : ate = 35) : d + s - ate = 39 :=
by
  -- d, s, and ate are given as natural numbers
  -- h_d, h_s, and h_ate are the provided conditions
  -- The goal is to prove d + s - ate = 39
  sorry

end candy_left_l8_8775


namespace fraction_sum_zero_implies_square_sum_zero_l8_8330

theorem fraction_sum_zero_implies_square_sum_zero (a b c : ℝ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := 
by
  sorry

end fraction_sum_zero_implies_square_sum_zero_l8_8330


namespace probability_neither_square_nor_cube_l8_8924

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l8_8924


namespace ratio_r_to_pq_l8_8631

theorem ratio_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 5000) (h₂ : r = 2000) :
  r / (p + q) = 2 / 3 := 
by
  sorry

end ratio_r_to_pq_l8_8631


namespace count_4_digit_multiples_of_5_is_9_l8_8184

noncomputable def count_4_digit_multiples_of_5 : Nat :=
  let digits := [2, 7, 4, 5]
  let last_digit := 5
  let remaining_digits := [2, 7, 4]
  let case_1 := 3
  let case_2 := 3 * 2
  case_1 + case_2

theorem count_4_digit_multiples_of_5_is_9 : count_4_digit_multiples_of_5 = 9 :=
by
  sorry

end count_4_digit_multiples_of_5_is_9_l8_8184


namespace chemical_transport_problem_l8_8813

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l8_8813


namespace chord_length_sqrt_10_l8_8100

/-
  Given a line L: 3x - y - 6 = 0 and a circle C: x^2 + y^2 - 2x - 4y = 0,
  prove that the length of the chord AB formed by their intersection is sqrt(10).
-/

noncomputable def line_L : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 3 * x - y - 6 = 0}

noncomputable def circle_C : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 - 2 * x - 4 * y = 0}

noncomputable def chord_length (L C : Set (ℝ × ℝ)) : ℝ :=
  let center := (1, 2)
  let r := Real.sqrt 5
  let d := |3 * 1 - 2 - 6| / Real.sqrt (1 + 3^2)
  2 * Real.sqrt (r^2 - d^2)

theorem chord_length_sqrt_10 : chord_length line_L circle_C = Real.sqrt 10 := sorry

end chord_length_sqrt_10_l8_8100


namespace smaller_number_l8_8328

theorem smaller_number (x y : ℝ) (h1 : x - y = 1650) (h2 : 0.075 * x = 0.125 * y) : y = 2475 := 
sorry

end smaller_number_l8_8328


namespace problem1_problem2_problem3_l8_8033

-- (1) Prove 1 - 2(x - y) + (x - y)^2 = (1 - x + y)^2
theorem problem1 (x y : ℝ) : 1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
sorry

-- (2) Prove 25(a - 1)^2 - 10(a - 1) + 1 = (5a - 6)^2
theorem problem2 (a : ℝ) : 25 * (a - 1)^2 - 10 * (a - 1) + 1 = (5 * a - 6)^2 :=
sorry

-- (3) Prove (y^2 - 4y)(y^2 - 4y + 8) + 16 = (y - 2)^4
theorem problem3 (y : ℝ) : (y^2 - 4 * y) * (y^2 - 4 * y + 8) + 16 = (y - 2)^4 :=
sorry

end problem1_problem2_problem3_l8_8033


namespace express_as_scientific_notation_l8_8102

-- Define the question and condition
def trillion : ℝ := 1000000000000
def num := 6.13 * trillion

-- The main statement to be proven
theorem express_as_scientific_notation : num = 6.13 * 10^12 :=
by
  sorry

end express_as_scientific_notation_l8_8102


namespace neg_q_true_l8_8886

theorem neg_q_true : (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_q_true_l8_8886


namespace favouring_more_than_one_is_39_l8_8080

def percentage_favouring_more_than_one (x : ℝ) : Prop :=
  let sum_two : ℝ := 8 + 6 + 4 + 2 + 7 + 5 + 3 + 5 + 3 + 2
  let sum_three : ℝ := 1 + 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 1.5 + 0.7 + 0.3 + 0.4
  let all_five : ℝ := 0.2
  x = sum_two - sum_three - all_five

theorem favouring_more_than_one_is_39 : percentage_favouring_more_than_one 39 := 
by
  sorry

end favouring_more_than_one_is_39_l8_8080


namespace budget_per_friend_l8_8373

-- Definitions for conditions
def total_budget : ℕ := 100
def parents_gift_cost : ℕ := 14
def number_of_parents : ℕ := 2
def number_of_friends : ℕ := 8

-- Statement to prove
theorem budget_per_friend :
  (total_budget - number_of_parents * parents_gift_cost) / number_of_friends = 9 :=
by
  sorry

end budget_per_friend_l8_8373


namespace find_functional_l8_8582

noncomputable def functional_equation_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = 2 * x + f (f y - x)

theorem find_functional (f : ℝ → ℝ) :
  functional_equation_solution f → ∃ c : ℝ, ∀ x, f x = x + c := 
by
  sorry

end find_functional_l8_8582


namespace function_value_at_6000_l8_8436

theorem function_value_at_6000
  (f : ℝ → ℝ)
  (h0 : f 0 = 1)
  (h1 : ∀ x : ℝ, f (x + 3) = f x + 2 * x + 3) :
  f 6000 = 12000001 :=
by
  sorry

end function_value_at_6000_l8_8436


namespace cost_equation_l8_8621

def cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem cost_equation (W : ℕ) : cost W = 
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10 :=
by
  -- Proof goes here
  sorry

end cost_equation_l8_8621


namespace triangle_area_tangent_log2_l8_8123

open Real

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem triangle_area_tangent_log2 :
  let y := log_base_2
  let f := fun x : ℝ => y x
  let deriv := (deriv f 1)
  let tangent_line := fun x : ℝ => deriv * (x - 1) + f 1
  let x_intercept := 1
  let y_intercept := tangent_line 0
  
  (1 : ℝ) * (abs y_intercept) / 2 = 1 / (2 * log 2) := by
  sorry

end triangle_area_tangent_log2_l8_8123


namespace max_blue_cells_n2_max_blue_cells_n25_l8_8244

noncomputable def max_blue_cells (table_size n : ℕ) : ℕ :=
  if h : (table_size = 50 ∧ n = 2) then 2450
  else if h : (table_size = 50 ∧ n = 25) then 1300
  else 0 -- Default case that should not happen for this problem

theorem max_blue_cells_n2 : max_blue_cells 50 2 = 2450 := 
by
  sorry

theorem max_blue_cells_n25 : max_blue_cells 50 25 = 1300 :=
by
  sorry

end max_blue_cells_n2_max_blue_cells_n25_l8_8244


namespace largest_among_five_numbers_l8_8716

theorem largest_among_five_numbers :
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  sorry

end largest_among_five_numbers_l8_8716


namespace expected_reflection_value_l8_8814

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) *
  (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem expected_reflection_value :
  expected_reflections = (2 / Real.pi) *
    (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end expected_reflection_value_l8_8814


namespace polygon_E_largest_area_l8_8971

def unit_square_area : ℕ := 1
def right_triangle_area : ℚ := 1 / 2
def rectangle_area : ℕ := 2

def polygon_A_area : ℚ := 3 * unit_square_area + 2 * right_triangle_area
def polygon_B_area : ℚ := 2 * unit_square_area + 4 * right_triangle_area
def polygon_C_area : ℚ := 4 * unit_square_area + 1 * rectangle_area
def polygon_D_area : ℚ := 3 * rectangle_area
def polygon_E_area : ℚ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

theorem polygon_E_largest_area :
  polygon_E_area = max polygon_A_area (max polygon_B_area (max polygon_C_area (max polygon_D_area polygon_E_area))) := by
  sorry

end polygon_E_largest_area_l8_8971


namespace line_symmetric_about_y_eq_x_l8_8710

-- Define the line equation types and the condition for symmetry
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions given
variable (a b c : ℝ)
variable (h_ab_pos : a * b > 0)

-- Definition of the problem in Lean
theorem line_symmetric_about_y_eq_x (h_bisector : ∀ x y : ℝ, line_equation a b c x y ↔ line_equation b a c y x) : 
  ∀ x y : ℝ, line_equation b a c x y := by
  sorry

end line_symmetric_about_y_eq_x_l8_8710


namespace tim_used_to_run_days_l8_8258

def hours_per_day := 2
def total_hours_per_week := 10
def added_days := 2

theorem tim_used_to_run_days (runs_per_day : ℕ) (total_weekly_runs : ℕ) (additional_runs : ℕ) : 
  runs_per_day = hours_per_day →
  total_weekly_runs = total_hours_per_week →
  additional_runs = added_days →
  (total_weekly_runs / runs_per_day) - additional_runs = 3 :=
by
  intros h1 h2 h3
  sorry

end tim_used_to_run_days_l8_8258


namespace total_balloons_l8_8049

theorem total_balloons (fred_balloons : ℕ) (sam_balloons : ℕ) (mary_balloons : ℕ) :
  fred_balloons = 5 → sam_balloons = 6 → mary_balloons = 7 → fred_balloons + sam_balloons + mary_balloons = 18 :=
by
  intros
  sorry

end total_balloons_l8_8049


namespace impossible_a_values_l8_8252

theorem impossible_a_values (a : ℝ) :
  ¬((1-a)^2 + (1+a)^2 < 4) → (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end impossible_a_values_l8_8252


namespace distinct_real_nums_condition_l8_8015

theorem distinct_real_nums_condition 
  (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 :=
by
  sorry

end distinct_real_nums_condition_l8_8015


namespace min_value_frac_sum_l8_8023

theorem min_value_frac_sum (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  (∃ m, ∀ x y, m = 1 ∧ (
      (1 / (x + y)^2) + (1 / (x - y)^2) ≥ m)) :=
sorry

end min_value_frac_sum_l8_8023


namespace height_of_new_TV_l8_8203

theorem height_of_new_TV 
  (width1 height1 cost1 : ℝ) 
  (width2 cost2 : ℝ) 
  (cost_diff_per_sq_inch : ℝ) 
  (h1 : width1 = 24) 
  (h2 : height1 = 16) 
  (h3 : cost1 = 672) 
  (h4 : width2 = 48) 
  (h5 : cost2 = 1152) 
  (h6 : cost_diff_per_sq_inch = 1) : 
  ∃ height2 : ℝ, height2 = 32 :=
by
  sorry

end height_of_new_TV_l8_8203


namespace books_on_each_shelf_l8_8344

-- Define the conditions and the problem statement
theorem books_on_each_shelf :
  ∀ (M P : ℕ), 
  -- Conditions
  (5 * M + 4 * P = 72) ∧ (M = P) ∧ (∃ B : ℕ, M = B ∧ P = B) ->
  -- Conclusion
  (∃ B : ℕ, B = 8) :=
by
  sorry

end books_on_each_shelf_l8_8344


namespace calculate_fraction_value_l8_8985

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value_l8_8985


namespace total_play_time_in_hours_l8_8476

def football_time : ℕ := 60
def basketball_time : ℕ := 60

theorem total_play_time_in_hours : (football_time + basketball_time) / 60 = 2 := by
  sorry

end total_play_time_in_hours_l8_8476


namespace smallest_sum_l8_8137

theorem smallest_sum (a b c : ℕ) (h : (13 * a + 11 * b + 7 * c = 1001)) :
    a / 77 + b / 91 + c / 143 = 1 → a + b + c = 79 :=
by
  sorry

end smallest_sum_l8_8137


namespace ratio_of_areas_l8_8145

-- Definitions of conditions
def side_length (s : ℝ) : Prop := s > 0
def original_area (A s : ℝ) : Prop := A = s^2

-- Definition of the new area after folding
def new_area (B A s : ℝ) : Prop := B = (7/8) * s^2

-- The proof statement to show the ratio B/A is 7/8
theorem ratio_of_areas (s A B : ℝ) (h_side : side_length s) (h_area : original_area A s) (h_B : new_area B A s) : 
  B / A = 7 / 8 := 
by 
  sorry

end ratio_of_areas_l8_8145


namespace value_of_expression_l8_8407

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l8_8407


namespace jett_profit_l8_8471

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end jett_profit_l8_8471


namespace right_triangle_condition_l8_8224

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition_l8_8224


namespace cube_faces_edges_vertices_sum_l8_8416

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l8_8416


namespace joan_total_spending_l8_8650

def basketball_game_price : ℝ := 5.20
def basketball_game_discount : ℝ := 0.15 * basketball_game_price
def basketball_game_discounted : ℝ := basketball_game_price - basketball_game_discount

def racing_game_price : ℝ := 4.23
def racing_game_discount : ℝ := 0.10 * racing_game_price
def racing_game_discounted : ℝ := racing_game_price - racing_game_discount

def puzzle_game_price : ℝ := 3.50

def total_before_tax : ℝ := basketball_game_discounted + racing_game_discounted + puzzle_game_price
def sales_tax : ℝ := 0.08 * total_before_tax
def total_with_tax : ℝ := total_before_tax + sales_tax

theorem joan_total_spending : (total_with_tax : ℝ) = 12.67 := by
  sorry

end joan_total_spending_l8_8650


namespace prob_two_packs_tablets_at_10am_dec31_l8_8194
noncomputable def prob_two_packs_tablets (n : ℕ) : ℝ :=
  let numer := (2^n - 1)
  let denom := 2^(n-1) * n
  numer / denom

theorem prob_two_packs_tablets_at_10am_dec31 :
  prob_two_packs_tablets 10 = 1023 / 5120 := by
  sorry

end prob_two_packs_tablets_at_10am_dec31_l8_8194


namespace tan_double_angle_l8_8940

theorem tan_double_angle (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α - π / 2) = 3 / 5) : 
  Real.tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_l8_8940


namespace sum_of_distances_condition_l8_8536

theorem sum_of_distances_condition (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 :=
sorry

end sum_of_distances_condition_l8_8536


namespace locus_of_P_is_single_ray_l8_8842
  
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def N : ℝ × ℝ := (3, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem locus_of_P_is_single_ray (P : ℝ × ℝ) (h : distance P M - distance P N = 2) : 
∃ α : ℝ, P = (3 + α * (P.1 - 3), α * P.2) :=
sorry

end locus_of_P_is_single_ray_l8_8842


namespace Lenny_pens_left_l8_8200

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end Lenny_pens_left_l8_8200


namespace problem_l8_8535

def a (x : ℕ) : ℕ := 2005 * x + 2006
def b (x : ℕ) : ℕ := 2005 * x + 2007
def c (x : ℕ) : ℕ := 2005 * x + 2008

theorem problem (x : ℕ) : (a x)^2 + (b x)^2 + (c x)^2 - (a x) * (b x) - (a x) * (c x) - (b x) * (c x) = 3 :=
by sorry

end problem_l8_8535


namespace find_f_four_thirds_l8_8909

def f (y: ℝ) : ℝ := sorry  -- Placeholder for the function definition

theorem find_f_four_thirds : f (4 / 3) = - (7 / 2) := sorry

end find_f_four_thirds_l8_8909


namespace quadratic_sum_terms_l8_8561

theorem quadratic_sum_terms (a b c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) → a + b + c = -46 :=
by
  sorry

end quadratic_sum_terms_l8_8561


namespace nesting_rectangles_exists_l8_8558

theorem nesting_rectangles_exists :
  ∀ (rectangles : List (ℕ × ℕ)), rectangles.length = 101
    ∧ (∀ r ∈ rectangles, r.fst ≤ 100 ∧ r.snd ≤ 100) 
    → ∃ (A B C : ℕ × ℕ), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles 
    ∧ (A.fst < B.fst ∧ A.snd < B.snd) 
    ∧ (B.fst < C.fst ∧ B.snd < C.snd) := 
by sorry

end nesting_rectangles_exists_l8_8558


namespace solve_for_x_l8_8819

theorem solve_for_x (x : ℝ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l8_8819


namespace tan_double_angle_l8_8518

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin (Real.pi / 2 + theta) + Real.sin (Real.pi + theta) = 0) :
  Real.tan (2 * theta) = -4 / 3 :=
by
  sorry

end tan_double_angle_l8_8518


namespace dishonest_dealer_weight_l8_8960

noncomputable def dealer_weight_equiv (cost_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (1 - profit_percent / 100) * cost_price / selling_price

theorem dishonest_dealer_weight :
  dealer_weight_equiv 1 2 100 = 0.5 :=
by
  sorry

end dishonest_dealer_weight_l8_8960


namespace sum_first_8_geometric_l8_8262

theorem sum_first_8_geometric :
  let a₁ := 1 / 15
  let r := 2
  let S₄ := a₁ * (1 - r^4) / (1 - r)
  let S₈ := a₁ * (1 - r^8) / (1 - r)
  S₄ = 1 → S₈ = 17 := 
by
  intros a₁ r S₄ S₈ h
  sorry

end sum_first_8_geometric_l8_8262


namespace no_solution_system_of_equations_l8_8659

theorem no_solution_system_of_equations :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) :=
by {
  sorry
}

end no_solution_system_of_equations_l8_8659


namespace if_a_eq_b_then_a_squared_eq_b_squared_l8_8744

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l8_8744


namespace difference_between_first_and_third_l8_8155

variable (x : ℕ)

-- Condition 1: The first number is twice the second.
def first_number : ℕ := 2 * x

-- Condition 2: The first number is three times the third.
def third_number : ℕ := first_number x / 3

-- Condition 3: The average of the three numbers is 88.
def average_condition : Prop := (first_number x + x + third_number x) / 3 = 88

-- Prove that the difference between first and third number is 96.
theorem difference_between_first_and_third 
  (h : average_condition x) : first_number x - third_number x = 96 :=
by
  sorry -- Proof omitted

end difference_between_first_and_third_l8_8155


namespace two_pipes_fill_tank_l8_8241

theorem two_pipes_fill_tank (C : ℝ) (hA : ∀ (t : ℝ), t = 10 → t = C / (C / 10)) (hB : ∀ (t : ℝ), t = 15 → t = C / (C / 15)) :
  ∀ (t : ℝ), t = C / (C / 6) → t = 6 :=
by
  sorry

end two_pipes_fill_tank_l8_8241


namespace product_roots_cos_pi_by_9_cos_2pi_by_9_l8_8005

theorem product_roots_cos_pi_by_9_cos_2pi_by_9 :
  ∀ (d e : ℝ), (∀ x, x^2 + d * x + e = (x - Real.cos (π / 9)) * (x - Real.cos (2 * π / 9))) → 
    d * e = -5 / 64 :=
by
  sorry

end product_roots_cos_pi_by_9_cos_2pi_by_9_l8_8005


namespace evaluate_g_at_neg_four_l8_8803

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l8_8803


namespace find_number_l8_8654

theorem find_number (x : ℕ) (h : x * 12 = 540) : x = 45 :=
by sorry

end find_number_l8_8654


namespace negation_of_both_even_l8_8616

-- Definitions
def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Main statement
theorem negation_of_both_even (a b : ℕ) : ¬ (even a ∧ even b) ↔ (¬even a ∨ ¬even b) :=
by sorry

end negation_of_both_even_l8_8616


namespace eval_expression_l8_8655

theorem eval_expression (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m - 3 = -1 :=
by
  sorry

end eval_expression_l8_8655


namespace sqrt_74_between_8_and_9_product_of_consecutive_integers_l8_8771

theorem sqrt_74_between_8_and_9 : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9 := sorry

theorem product_of_consecutive_integers (h : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9) : 8 * 9 = 72 := by
  have h1 : 8 < Real.sqrt 74 := And.left h
  have h2 : Real.sqrt 74 < 9 := And.right h
  calc
    8 * 9 = 72 := by norm_num

end sqrt_74_between_8_and_9_product_of_consecutive_integers_l8_8771


namespace no_intersection_with_x_axis_l8_8050

open Real

theorem no_intersection_with_x_axis (m : ℝ) :
  (∀ x : ℝ, 3 ^ (-(|x - 1|)) + m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) :=
by
  sorry

end no_intersection_with_x_axis_l8_8050


namespace quadratic_trinomial_unique_l8_8829

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4*(a+1)*c = 0)
  (h2 : (b+1)^2 - 4*a*c = 0)
  (h3 : b^2 - 4*a*(c+1) = 0) :
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by
  sorry

end quadratic_trinomial_unique_l8_8829


namespace num_real_numbers_l8_8124

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l8_8124


namespace geometric_sequence_problem_l8_8541

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Geometric sequence definition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
def condition_1 : Prop := a 5 * a 8 = 6
def condition_2 : Prop := a 3 + a 10 = 5

-- Concluded value of q^7
def q_seven (q : ℝ) (a : ℕ → ℝ) : Prop := 
  q^7 = a 20 / a 13

theorem geometric_sequence_problem
  (h1 : is_geometric_sequence a q)
  (h2 : condition_1 a)
  (h3 : condition_2 a) :
  q_seven q a = (q = 3/2) ∨ (q = 2/3) :=
sorry

end geometric_sequence_problem_l8_8541


namespace smallest_number_l8_8309

theorem smallest_number (A B C : ℕ) 
  (h1 : A / 3 = B / 5) 
  (h2 : B / 5 = C / 7) 
  (h3 : C = 56) 
  (h4 : C - A = 32) : 
  A = 24 := 
sorry

end smallest_number_l8_8309


namespace proof_problem_l8_8339

def f (a b c : ℕ) : ℕ :=
  a * 100 + b * 10 + c

def special_op (a b c : ℕ) : ℕ :=
  f (a * b) (b * c / 10) (b * c % 10)

theorem proof_problem :
  special_op 5 7 4 - special_op 7 4 5 = 708 := 
    sorry

end proof_problem_l8_8339


namespace exists_initial_segment_of_power_of_2_l8_8668

theorem exists_initial_segment_of_power_of_2 (m : ℕ) : ∃ n : ℕ, ∃ k : ℕ, k ≥ m ∧ 2^n = 10^k * m ∨ 2^n = 10^k * (m+1) := 
by
  sorry

end exists_initial_segment_of_power_of_2_l8_8668


namespace min_value_f_l8_8361

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

theorem min_value_f (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f a b x ≤ 5) : 
  ∀ x : ℝ, x < 0 → f a b x ≥ -1 :=
by
  -- Since this is a statement-only problem, we leave the proof to be filled in
  sorry

end min_value_f_l8_8361


namespace find_d_l8_8786

theorem find_d (d : ℚ) (h_floor : ∃ x : ℤ, x^2 + 5 * x - 36 = 0 ∧ x = ⌊d⌋)
  (h_frac: ∃ y : ℚ, 3 * y^2 - 11 * y + 2 = 0 ∧ y = d - ⌊d⌋):
  d = 13 / 3 :=
by
  sorry

end find_d_l8_8786


namespace find_expression_max_value_min_value_l8_8937

namespace MathProblem

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

-- Hypotheses based on problem conditions
lemma a_neg (a b : ℝ) : a < 0 := sorry
lemma root_neg2 (a b : ℝ) : f a b (-2) = 0 := sorry
lemma root_6 (a b : ℝ) : f a b 6 = 0 := sorry

-- Proving the explicit expression for f(x)
theorem find_expression (a b : ℝ) (x : ℝ) : 
  a = -4 → 
  b = -8 → 
  f a b x = -4 * x^2 + 16 * x + 48 :=
sorry

-- Maximum value of f(x) on the interval [1, 10]
theorem max_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 2 = 64 :=
sorry

-- Minimum value of f(x) on the interval [1, 10]
theorem min_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 10 = -192 :=
sorry

end MathProblem

end find_expression_max_value_min_value_l8_8937


namespace ratio_lcm_gcf_l8_8438

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 252) (h₂ : b = 675) : 
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  (lcm_ab / gcf_ab) = 2100 :=
by
  sorry

end ratio_lcm_gcf_l8_8438


namespace find_f_neg_a_l8_8271

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l8_8271


namespace product_plus_one_square_l8_8493

theorem product_plus_one_square (n : ℕ):
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 := 
  sorry

end product_plus_one_square_l8_8493


namespace basketball_cost_l8_8402

-- Initial conditions
def initial_amount : Nat := 50
def cost_jerseys (n price_per_jersey : Nat) : Nat := n * price_per_jersey
def cost_shorts : Nat := 8
def remaining_amount : Nat := 14

-- Derived total spent calculation
def total_spent (initial remaining : Nat) : Nat := initial - remaining
def known_cost (jerseys shorts : Nat) : Nat := jerseys + shorts

-- Prove the cost of the basketball
theorem basketball_cost :
  let jerseys := cost_jerseys 5 2
  let shorts := cost_shorts
  let total_spent := total_spent initial_amount remaining_amount
  let known_cost := known_cost jerseys shorts
  total_spent - known_cost = 18 := 
by
  sorry

end basketball_cost_l8_8402


namespace tape_needed_for_large_box_l8_8021

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box_l8_8021
