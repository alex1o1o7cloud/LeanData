import Mathlib

namespace required_number_l397_39784

-- Define the main variables and conditions
variables {i : ℂ} (z : ℂ)
axiom i_squared : i^2 = -1

-- State the theorem that needs to be proved
theorem required_number (h : z + (4 - 8 * i) = 1 + 10 * i) : z = -3 + 18 * i :=
by {
  -- the exact steps for the proof will follow here
  sorry
}

end required_number_l397_39784


namespace eval_expression_l397_39708

theorem eval_expression :
  ((-2 : ℤ) ^ 3 : ℝ) ^ (1/3 : ℝ) - (-1 : ℤ) ^ 0 = -3 := by
  sorry

end eval_expression_l397_39708


namespace camera_pics_l397_39733

-- Definitions of the given conditions
def phone_pictures := 22
def albums := 4
def pics_per_album := 6

-- The statement to prove the number of pictures uploaded from camera
theorem camera_pics : (albums * pics_per_album) - phone_pictures = 2 :=
by
  sorry

end camera_pics_l397_39733


namespace evaluate_difference_of_squares_l397_39765
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l397_39765


namespace krishan_money_l397_39715

-- Define the constants
def Ram : ℕ := 490
def ratio1 : ℕ := 7
def ratio2 : ℕ := 17

-- Defining the relationship
def ratio_RG (Ram Gopal : ℕ) : Prop := Ram / Gopal = ratio1 / ratio2
def ratio_GK (Gopal Krishan : ℕ) : Prop := Gopal / Krishan = ratio1 / ratio2

-- Define the problem
theorem krishan_money (R G K : ℕ) (h1 : R = Ram) (h2 : ratio_RG R G) (h3 : ratio_GK G K) : K = 2890 :=
by
  sorry

end krishan_money_l397_39715


namespace range_of_alpha_div_three_l397_39792

open Real

theorem range_of_alpha_div_three {k : ℤ} {α : ℝ} 
  (h1 : sin α > 0)
  (h2 : cos α < 0)
  (h3 : sin (α / 3) > cos (α / 3)) :
  (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) 
  ∨ (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
sorry

end range_of_alpha_div_three_l397_39792


namespace pascals_triangle_row_20_fifth_element_l397_39742

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- State the theorem about Row 20, fifth element in Pascal's triangle
theorem pascals_triangle_row_20_fifth_element :
  binomial 20 4 = 4845 := 
by
  sorry

end pascals_triangle_row_20_fifth_element_l397_39742


namespace geometric_progression_x_unique_l397_39750

theorem geometric_progression_x_unique (x : ℝ) :
  (70+x)^2 = (30+x)*(150+x) ↔ x = 10 := by
  sorry

end geometric_progression_x_unique_l397_39750


namespace fred_seashells_l397_39755

def seashells_given : ℕ := 25
def seashells_left : ℕ := 22
def seashells_found : ℕ := 47

theorem fred_seashells :
  seashells_found = seashells_given + seashells_left :=
  by sorry

end fred_seashells_l397_39755


namespace energy_savings_l397_39794

theorem energy_savings (x y : ℝ) 
  (h1 : x = y + 27) 
  (h2 : x + 2.1 * y = 405) :
  x = 149 ∧ y = 122 :=
by
  sorry

end energy_savings_l397_39794


namespace find_point_A_l397_39760

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end find_point_A_l397_39760


namespace jim_miles_driven_l397_39725

theorem jim_miles_driven (total_journey : ℕ) (miles_needed : ℕ) (h : total_journey = 1200 ∧ miles_needed = 985) : total_journey - miles_needed = 215 := 
by sorry

end jim_miles_driven_l397_39725


namespace polynomial_evaluation_l397_39739

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l397_39739


namespace inscribed_square_neq_five_l397_39714

theorem inscribed_square_neq_five (a b : ℝ) 
  (h1 : a - b = 1)
  (h2 : a * b = 1)
  (h3 : a + b = Real.sqrt 5) : a^2 + b^2 ≠ 5 :=
by sorry

end inscribed_square_neq_five_l397_39714


namespace line_curve_intersection_symmetric_l397_39778

theorem line_curve_intersection_symmetric (a b : ℝ) 
    (h1 : ∃ p q : ℝ × ℝ, 
          (p.2 = a * p.1 + 1) ∧ 
          (q.2 = a * q.1 + 1) ∧ 
          (p ≠ q) ∧ 
          (p.1^2 + p.2^2 + b * p.1 - p.2 = 1) ∧ 
          (q.1^2 + q.2^2 + b * q.1 - q.2 = 1) ∧ 
          (p.1 + p.2 = -q.1 - q.2)) : 
  a + b = 2 :=
sorry

end line_curve_intersection_symmetric_l397_39778


namespace problem_statement_l397_39774

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem problem_statement (h_odd : is_odd f) (h_decr : is_decreasing f) (a b : ℝ) (h_ab : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 :=
by
  sorry

end problem_statement_l397_39774


namespace numPerfectSquareFactorsOf450_l397_39706

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l397_39706


namespace num_rectangular_arrays_with_36_chairs_l397_39744

theorem num_rectangular_arrays_with_36_chairs :
  ∃ n : ℕ, (∀ r c : ℕ, r * c = 36 ∧ r ≥ 2 ∧ c ≥ 2 ↔ n = 7) :=
sorry

end num_rectangular_arrays_with_36_chairs_l397_39744


namespace total_salary_l397_39761

-- Define the salaries and conditions.
def salaryN : ℝ := 280
def salaryM : ℝ := 1.2 * salaryN

-- State the theorem we want to prove
theorem total_salary : salaryM + salaryN = 616 :=
by
  sorry

end total_salary_l397_39761


namespace degree_of_monomial_x_l397_39746

def is_monomial (e : Expr) : Prop := sorry -- Placeholder definition
def degree (e : Expr) : Nat := sorry -- Placeholder definition

theorem degree_of_monomial_x :
  degree x = 1 :=
by
  sorry

end degree_of_monomial_x_l397_39746


namespace range_of_x_l397_39779

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 → -1 < x ∧ x < 2 := by
  sorry

end range_of_x_l397_39779


namespace find_d_l397_39754

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l397_39754


namespace number_of_segments_before_returning_to_start_l397_39767

-- Definitions based on the conditions
def concentric_circles (r R : ℝ) (h_circle : r < R) : Prop := true

def tangent_chord (circle1 circle2 : Prop) (A B : Point) : Prop := 
  circle1 ∧ circle2

def angle_ABC_eq_60 (A B C : Point) (angle_ABC : ℝ) : Prop :=
  angle_ABC = 60

noncomputable def number_of_segments (n : ℕ) (m : ℕ) : Prop := 
  120 * n = 360 * m

theorem number_of_segments_before_returning_to_start (r R : ℝ)
  (h_circle : r < R)
  (circle1 circle2 : Prop := concentric_circles r R h_circle)
  (A B C : Point)
  (h_tangent : tangent_chord circle1 circle2 A B)
  (angle_ABC : ℝ := 0)
  (h_ABC_eq_60 : angle_ABC_eq_60 A B C angle_ABC) :
  ∃ n : ℕ, number_of_segments n 1 ∧ n = 3 := by
  sorry

end number_of_segments_before_returning_to_start_l397_39767


namespace math_competition_probs_l397_39736

-- Definitions related to the problem conditions
def boys : ℕ := 3
def girls : ℕ := 3
def total_students := boys + girls
def total_combinations := (total_students.choose 2)

-- Definition of the probabilities
noncomputable def prob_exactly_one_boy : ℚ := 0.6
noncomputable def prob_at_least_one_boy : ℚ := 0.8
noncomputable def prob_at_most_one_boy : ℚ := 0.8

-- Lean statement for the proof problem
theorem math_competition_probs :
  prob_exactly_one_boy = 0.6 ∧
  prob_at_least_one_boy = 0.8 ∧
  prob_at_most_one_boy = 0.8 :=
by
  sorry

end math_competition_probs_l397_39736


namespace ab_value_l397_39740

theorem ab_value 
  (a b : ℕ) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h1 : a + b = 30)
  (h2 : 3 * a * b + 4 * a = 5 * b + 318) : 
  (a * b = 56) :=
sorry

end ab_value_l397_39740


namespace find_x_l397_39731

noncomputable def value_of_x (x : ℝ) := (5 * x) ^ 4 = (15 * x) ^ 3

theorem find_x : ∀ (x : ℝ), (value_of_x x) ∧ (x ≠ 0) → x = 27 / 5 :=
by
  intro x
  intro h
  sorry

end find_x_l397_39731


namespace initial_salty_cookies_l397_39734

theorem initial_salty_cookies (sweet_init sweet_eaten sweet_left salty_eaten : ℕ) 
  (h1 : sweet_init = 34)
  (h2 : sweet_eaten = 15)
  (h3 : sweet_left = 19)
  (h4 : salty_eaten = 56) :
  sweet_left + sweet_eaten = sweet_init → 
  sweet_init - sweet_eaten = sweet_left →
  ∃ salty_init, salty_init = salty_eaten :=
by
  sorry

end initial_salty_cookies_l397_39734


namespace pencil_price_is_99c_l397_39720

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c_l397_39720


namespace roses_picked_later_l397_39705

/-- Represents the initial number of roses the florist had. -/
def initial_roses : ℕ := 37

/-- Represents the number of roses the florist sold. -/
def sold_roses : ℕ := 16

/-- Represents the final number of roses the florist ended up with. -/
def final_roses : ℕ := 40

/-- Theorem which states the number of roses picked later is 19 given the conditions. -/
theorem roses_picked_later : (final_roses - (initial_roses - sold_roses)) = 19 :=
by
  -- proof steps are omitted, sorry as a placeholder
  sorry

end roses_picked_later_l397_39705


namespace factorization_a_minus_b_l397_39752

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l397_39752


namespace prob_B_hits_once_prob_hits_with_ABC_l397_39799

section
variable (P_A P_B P_C : ℝ)
variable (hA : P_A = 1 / 2)
variable (hB : P_B = 1 / 3)
variable (hC : P_C = 1 / 4)

-- Part (Ⅰ): Probability of hitting the target exactly once when B shoots twice
theorem prob_B_hits_once : 
  (P_B * (1 - P_B) + (1 - P_B) * P_B) = 4 / 9 := 
by
  rw [hB]
  sorry

-- Part (Ⅱ): Probability of hitting the target when A, B, and C each shoot once
theorem prob_hits_with_ABC :
  (1 - ((1 - P_A) * (1 - P_B) * (1 - P_C))) = 3 / 4 := 
by
  rw [hA, hB, hC]
  sorry

end

end prob_B_hits_once_prob_hits_with_ABC_l397_39799


namespace certain_amount_added_l397_39719

theorem certain_amount_added {x y : ℕ} 
    (h₁ : x = 15) 
    (h₂ : 3 * (2 * x + y) = 105) : y = 5 :=
by
  sorry

end certain_amount_added_l397_39719


namespace evaluate_expression_l397_39782

def f (x : ℕ) : ℕ := 3 * x - 4
def g (x : ℕ) : ℕ := x - 1

theorem evaluate_expression : f (1 + g 3) = 5 := by
  sorry

end evaluate_expression_l397_39782


namespace solve_equation1_solve_equation2_pos_solve_equation2_neg_l397_39712

theorem solve_equation1 (x : ℝ) (h : 2 * x^3 = 16) : x = 2 :=
sorry

theorem solve_equation2_pos (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 :=
sorry

theorem solve_equation2_neg (x : ℝ) (h : (x - 1)^2 = 4) : x = -1 :=
sorry

end solve_equation1_solve_equation2_pos_solve_equation2_neg_l397_39712


namespace num_girls_on_playground_l397_39776

-- Definitions based on conditions
def total_students : ℕ := 20
def classroom_students := total_students / 4
def playground_students := total_students - classroom_students
def boys_playground := playground_students / 3
def girls_playground := playground_students - boys_playground

-- Theorem statement
theorem num_girls_on_playground : girls_playground = 10 :=
by
  -- Begin preparing proofs
  sorry

end num_girls_on_playground_l397_39776


namespace Randy_used_blocks_l397_39787

theorem Randy_used_blocks (initial_blocks blocks_left used_blocks : ℕ) 
  (h1 : initial_blocks = 97) 
  (h2 : blocks_left = 72) 
  (h3 : used_blocks = initial_blocks - blocks_left) : 
  used_blocks = 25 :=
by
  sorry

end Randy_used_blocks_l397_39787


namespace honey_nectar_relationship_l397_39704

-- Definitions representing the conditions
def nectarA_water_content (x : ℝ) := 0.7 * x
def nectarB_water_content (y : ℝ) := 0.5 * y
def final_honey_water_content := 0.3
def evaporation_loss (initial_content : ℝ) := 0.15 * initial_content

-- The system of equations to prove
theorem honey_nectar_relationship (x y : ℝ) :
  (x + y = 1) ∧ (0.595 * x + 0.425 * y = 0.3) :=
sorry

end honey_nectar_relationship_l397_39704


namespace number_of_students_is_20_l397_39724

-- Define the constants and conditions
def average_age_all_students (N : ℕ) : ℕ := 20
def average_age_9_students : ℕ := 11
def average_age_10_students : ℕ := 24
def age_20th_student : ℕ := 61

theorem number_of_students_is_20 (N : ℕ) 
  (h1 : N * average_age_all_students N = 99 + 240 + 61) 
  (h2 : 99 = 9 * average_age_9_students) 
  (h3 : 240 = 10 * average_age_10_students) 
  (h4 : N = 9 + 10 + 1) : N = 20 :=
sorry

end number_of_students_is_20_l397_39724


namespace different_types_of_players_l397_39771

theorem different_types_of_players :
  ∀ (cricket hockey football softball : ℕ) (total_players : ℕ),
    cricket = 12 → hockey = 17 → football = 11 → softball = 10 → total_players = 50 →
    cricket + hockey + football + softball = total_players → 
    4 = 4 :=
by
  intros
  rfl

end different_types_of_players_l397_39771


namespace inequality_solution_l397_39727

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l397_39727


namespace expression_divisible_by_13_l397_39762

theorem expression_divisible_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a ^ 2007 + b ^ 2007 + c ^ 2007 + 2 * 2007 * a * b * c) % 13 = 0 := 
by 
  sorry

end expression_divisible_by_13_l397_39762


namespace perfect_square_as_sum_of_powers_of_2_l397_39798

theorem perfect_square_as_sum_of_powers_of_2 (n a b : ℕ) (h : n^2 = 2^a + 2^b) (hab : a ≥ b) :
  (∃ k : ℕ, n^2 = 4^(k + 1)) ∨ (∃ k : ℕ, n^2 = 9 * 4^k) :=
by
  sorry

end perfect_square_as_sum_of_powers_of_2_l397_39798


namespace units_digit_33_219_89_plus_89_19_l397_39777

theorem units_digit_33_219_89_plus_89_19 :
  let units_digit x := x % 10
  units_digit (33 * 219 ^ 89 + 89 ^ 19) = 8 :=
by
  sorry

end units_digit_33_219_89_plus_89_19_l397_39777


namespace no_integer_y_such_that_abs_g_y_is_prime_l397_39757

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m ≤ n → m ∣ n → m = 1 ∨ m = n

def g (y : ℤ) : ℤ := 8 * y^2 - 55 * y + 21

theorem no_integer_y_such_that_abs_g_y_is_prime : 
  ∀ y : ℤ, ¬ is_prime (|g y|) :=
by sorry

end no_integer_y_such_that_abs_g_y_is_prime_l397_39757


namespace number_of_chips_per_day_l397_39764

def total_chips : ℕ := 100
def chips_first_day : ℕ := 10
def total_days : ℕ := 10
def days_remaining : ℕ := total_days - 1
def chips_remaining : ℕ := total_chips - chips_first_day

theorem number_of_chips_per_day : 
  chips_remaining / days_remaining = 10 :=
by 
  unfold chips_remaining days_remaining total_chips chips_first_day total_days
  sorry

end number_of_chips_per_day_l397_39764


namespace first_term_geometric_progression_l397_39741

theorem first_term_geometric_progression (S a : ℝ) (r : ℝ) 
  (h1 : S = 10) 
  (h2 : a = 10 * (1 - r)) 
  (h3 : a * (1 + r) = 7) : 
  a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10)) := 
by 
  sorry

end first_term_geometric_progression_l397_39741


namespace right_triangle_legs_l397_39775

theorem right_triangle_legs (R r : ℝ) : 
  ∃ a b : ℝ, a = Real.sqrt (2 * (R^2 + r^2)) ∧ b = Real.sqrt (2 * (R^2 - r^2)) :=
by
  sorry

end right_triangle_legs_l397_39775


namespace fraction_addition_simplest_form_l397_39743

theorem fraction_addition_simplest_form :
  (7 / 12) + (3 / 8) = 23 / 24 :=
by
  -- Adding a sorry to skip the proof
  sorry

end fraction_addition_simplest_form_l397_39743


namespace intersection_M_N_is_neq_neg1_0_1_l397_39783

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N_is_neq_neg1_0_1 :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_M_N_is_neq_neg1_0_1_l397_39783


namespace perpendicular_planes_condition_l397_39717

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end perpendicular_planes_condition_l397_39717


namespace percentage_increase_in_cellphone_pay_rate_l397_39751

theorem percentage_increase_in_cellphone_pay_rate
    (regular_rate : ℝ)
    (total_surveys : ℕ)
    (cellphone_surveys : ℕ)
    (total_earnings : ℝ)
    (regular_surveys : ℕ := total_surveys - cellphone_surveys)
    (higher_rate : ℝ := (total_earnings - (regular_surveys * regular_rate)) / cellphone_surveys)
    : regular_rate = 30 ∧ total_surveys = 100 ∧ cellphone_surveys = 50 ∧ total_earnings = 3300
    → ((higher_rate - regular_rate) / regular_rate) * 100 = 20 := by
  sorry

end percentage_increase_in_cellphone_pay_rate_l397_39751


namespace solve_equation_l397_39710

theorem solve_equation (x y : ℝ) : 
    ((16 * x^2 + 1) * (y^2 + 1) = 16 * x * y) ↔ 
        ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) := 
by
  sorry

end solve_equation_l397_39710


namespace quadratic_roots_relation_l397_39789

noncomputable def roots_relation (a b c : ℝ) : Prop :=
  ∃ α β : ℝ, (α * β = c / a) ∧ (α + β = -b / a) ∧ β = 3 * α

theorem quadratic_roots_relation (a b c : ℝ) (h : roots_relation a b c) : 3 * b^2 = 16 * a * c :=
by
  sorry

end quadratic_roots_relation_l397_39789


namespace rate_per_kg_mangoes_is_55_l397_39772

def total_amount : ℕ := 1125
def rate_per_kg_grapes : ℕ := 70
def weight_grapes : ℕ := 9
def weight_mangoes : ℕ := 9

def cost_grapes := rate_per_kg_grapes * weight_grapes
def cost_mangoes := total_amount - cost_grapes

theorem rate_per_kg_mangoes_is_55 (rate_per_kg_mangoes : ℕ) (h : rate_per_kg_mangoes = cost_mangoes / weight_mangoes) : rate_per_kg_mangoes = 55 :=
by
  -- proof construction
  sorry

end rate_per_kg_mangoes_is_55_l397_39772


namespace infinite_non_prime_seq_l397_39722

-- Let's state the theorem in Lean
theorem infinite_non_prime_seq (k : ℕ) : 
  ∃ᶠ n in at_top, ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → ¬ Nat.Prime (n + i) := 
sorry

end infinite_non_prime_seq_l397_39722


namespace total_distance_covered_is_correct_fuel_cost_excess_is_correct_l397_39723

-- Define the ratios and other conditions for Car A
def carA_ratio_gal_per_mile : ℚ := 4 / 7
def carA_gallons_used : ℚ := 44
def carA_cost_per_gallon : ℚ := 3.50

-- Define the ratios and other conditions for Car B
def carB_ratio_gal_per_mile : ℚ := 3 / 5
def carB_gallons_used : ℚ := 27
def carB_cost_per_gallon : ℚ := 3.25

-- Define the budget
def budget : ℚ := 200

-- Combined total distance covered by both cars
theorem total_distance_covered_is_correct :
  (carA_gallons_used * (7 / 4) + carB_gallons_used * (5 / 3)) = 122 :=
by
  sorry

-- Total fuel cost and whether it stays within budget
theorem fuel_cost_excess_is_correct :
  ((carA_gallons_used * carA_cost_per_gallon) + (carB_gallons_used * carB_cost_per_gallon)) - budget = 41.75 :=
by
  sorry

end total_distance_covered_is_correct_fuel_cost_excess_is_correct_l397_39723


namespace mike_training_hours_l397_39791

-- Define the individual conditions
def first_weekday_hours : Nat := 2
def first_weekend_hours : Nat := 1
def first_week_days : Nat := 5
def first_weekend_days : Nat := 2

def second_weekday_hours : Nat := 3
def second_weekend_hours : Nat := 2
def second_week_days : Nat := 4  -- since the first day of second week is a rest day
def second_weekend_days : Nat := 2

def first_week_hours : Nat := (first_weekday_hours * first_week_days) + (first_weekend_hours * first_weekend_days)
def second_week_hours : Nat := (second_weekday_hours * second_week_days) + (second_weekend_hours * second_weekend_days)

def total_training_hours : Nat := first_week_hours + second_week_hours

-- The final proof statement
theorem mike_training_hours : total_training_hours = 28 := by
  exact sorry

end mike_training_hours_l397_39791


namespace no_such_natural_number_exists_l397_39773

theorem no_such_natural_number_exists :
  ¬ ∃ n : ℕ, ∃ m : ℕ, 3^n + 2 * 17^n = m^2 :=
by sorry

end no_such_natural_number_exists_l397_39773


namespace proof_problem_l397_39753

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a n = a1 + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → α}

theorem proof_problem (h_arith_seq : is_arithmetic_sequence a)
    (h_S6_gt_S7 : sum_first_n_terms a 6 > sum_first_n_terms a 7)
    (h_S7_gt_S5 : sum_first_n_terms a 7 > sum_first_n_terms a 5) :
    (∃ d : α, d < 0) ∧ (∃ S11 : α, sum_first_n_terms a 11 > 0) :=
  sorry

end proof_problem_l397_39753


namespace number_of_integer_length_chords_through_point_l397_39709

theorem number_of_integer_length_chords_through_point 
  (r : ℝ) (d : ℝ) (P_is_5_units_from_center : d = 5) (circle_has_radius_13 : r = 13) :
  ∃ n : ℕ, n = 3 := by
  sorry

end number_of_integer_length_chords_through_point_l397_39709


namespace rides_with_remaining_tickets_l397_39763

theorem rides_with_remaining_tickets (T_total : ℕ) (T_spent : ℕ) (C_ride : ℕ)
  (h1 : T_total = 40) (h2 : T_spent = 28) (h3 : C_ride = 4) :
  (T_total - T_spent) / C_ride = 3 := by
  sorry

end rides_with_remaining_tickets_l397_39763


namespace arun_age_in_6_years_l397_39726

theorem arun_age_in_6_years
  (A D n : ℕ)
  (h1 : D = 42)
  (h2 : A = (5 * D) / 7)
  (h3 : A + n = 36) 
  : n = 6 :=
by
  sorry

end arun_age_in_6_years_l397_39726


namespace multiplier_is_3_l397_39793

theorem multiplier_is_3 (x : ℝ) (num : ℝ) (difference : ℝ) (h1 : num = 15.0) (h2 : difference = 40) (h3 : x * num - 5 = difference) : x = 3 := 
by 
  sorry

end multiplier_is_3_l397_39793


namespace congruence_solution_l397_39711

theorem congruence_solution (x : ℤ) (h : 5 * x + 11 ≡ 3 [ZMOD 19]) : 3 * x + 7 ≡ 6 [ZMOD 19] :=
sorry

end congruence_solution_l397_39711


namespace cauliflower_sales_l397_39745

theorem cauliflower_sales :
  let total_earnings := 500
  let b_sales := 57
  let c_sales := 2 * b_sales
  let s_sales := (c_sales / 2) + 16
  let t_sales := b_sales + s_sales
  let ca_sales := total_earnings - (b_sales + c_sales + s_sales + t_sales)
  ca_sales = 126 := by
  sorry

end cauliflower_sales_l397_39745


namespace average_speed_l397_39748

-- Define the conditions
def distance1 := 350 -- miles
def time1 := 6 -- hours
def distance2 := 420 -- miles
def time2 := 7 -- hours

-- Define the total distance and total time (excluding break)
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- Define the statement to prove
theorem average_speed : 
  (total_distance / total_time : ℚ) = 770 / 13 := by
  sorry

end average_speed_l397_39748


namespace right_triangle_ratio_l397_39730

theorem right_triangle_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ (x y : ℝ), 5 * (x * y) = x^2 + y^2 ∧ 5 * (a^2 + b^2) = (x + y)^2 ∧ 
    ((x - y)^2 < x^2 + y^2 ∧ x^2 + y^2 < (x + y)^2)):
  (1/2 < a / b) ∧ (a / b < 2) := by
  sorry

end right_triangle_ratio_l397_39730


namespace f_2014_l397_39795

noncomputable def f : ℕ → ℕ := sorry

axiom f_property : ∀ n, f (f n) + f n = 2 * n + 3
axiom f_zero : f 0 = 1

theorem f_2014 : f 2014 = 2015 := 
by sorry

end f_2014_l397_39795


namespace merchant_discount_l397_39728

-- Definitions used in Lean 4 statement coming directly from conditions
def initial_cost_price : Real := 100
def marked_up_percentage : Real := 0.80
def profit_percentage : Real := 0.35

-- To prove the percentage discount offered
theorem merchant_discount (cp mp sp discount percentage_discount : Real) 
  (H1 : cp = initial_cost_price)
  (H2 : mp = cp + (marked_up_percentage * cp))
  (H3 : sp = cp + (profit_percentage * cp))
  (H4 : discount = mp - sp)
  (H5 : percentage_discount = (discount / mp) * 100) :
  percentage_discount = 25 := 
sorry

end merchant_discount_l397_39728


namespace sam_walking_speed_l397_39701

variable (s : ℝ)
variable (t : ℝ)
variable (fred_speed : ℝ := 2)
variable (sam_distance : ℝ := 25)
variable (total_distance : ℝ := 35)

theorem sam_walking_speed :
  (total_distance - sam_distance) = fred_speed * t ∧
  sam_distance = s * t →
  s = 5 := 
by
  intros
  sorry

end sam_walking_speed_l397_39701


namespace youngest_child_age_l397_39749

theorem youngest_child_age (x y z : ℕ) 
  (h1 : 3 * x + 6 = 48) 
  (h2 : 3 * y + 9 = 60) 
  (h3 : 2 * z + 4 = 30) : 
  z = 13 := 
sorry

end youngest_child_age_l397_39749


namespace slightly_used_crayons_l397_39729

theorem slightly_used_crayons (total_crayons : ℕ) (percent_new : ℚ) (percent_broken : ℚ) 
  (h1 : total_crayons = 250) (h2 : percent_new = 40/100) (h3 : percent_broken = 1/5) : 
  (total_crayons - percent_new * total_crayons - percent_broken * total_crayons) = 100 :=
by
  -- sorry here to indicate the proof is omitted
  sorry

end slightly_used_crayons_l397_39729


namespace scientific_notation_correct_l397_39738

-- Defining the given number in terms of its scientific notation components.
def million : ℝ := 10^6
def num_million : ℝ := 15.276

-- Expressing the number 15.276 million using its definition.
def fifteen_point_two_seven_six_million : ℝ := num_million * million

-- Scientific notation representation to be proved.
def scientific_notation : ℝ := 1.5276 * 10^7

-- The theorem statement.
theorem scientific_notation_correct :
  fifteen_point_two_seven_six_million = scientific_notation :=
by
  sorry

end scientific_notation_correct_l397_39738


namespace find_a_b_l397_39780

noncomputable def z : ℂ := 1 + Complex.I
noncomputable def lhs (a b : ℝ) := (z^2 + a*z + b) / (z^2 - z + 1)
noncomputable def rhs : ℂ := 1 - Complex.I

theorem find_a_b (a b : ℝ) (h : lhs a b = rhs) : a = -1 ∧ b = 2 :=
  sorry

end find_a_b_l397_39780


namespace Elmer_eats_more_than_Penelope_l397_39758

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l397_39758


namespace contractor_work_done_l397_39797

def initial_people : ℕ := 10
def remaining_people : ℕ := 8
def total_days : ℕ := 100
def remaining_days : ℕ := 75
def fraction_done : ℚ := 1/4
def total_work : ℚ := 1

theorem contractor_work_done (x : ℕ) 
  (h1 : initial_people * x = fraction_done * total_work) 
  (h2 : remaining_people * remaining_days = (1 - fraction_done) * total_work) :
  x = 60 :=
by
  sorry

end contractor_work_done_l397_39797


namespace dina_dolls_l397_39769

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l397_39769


namespace find_f_2008_l397_39756

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the problem statement with all given conditions
theorem find_f_2008 (h_odd : is_odd f) (h_f2 : f 2 = 0) (h_rec : ∀ x, f (x + 4) = f x + f 4) : f 2008 = 0 := 
sorry

end find_f_2008_l397_39756


namespace bridge_length_l397_39770

def train_length : ℕ := 170 -- Train length in meters
def train_speed : ℕ := 45 -- Train speed in kilometers per hour
def crossing_time : ℕ := 30 -- Time to cross the bridge in seconds

noncomputable def speed_m_per_s : ℚ := (train_speed * 1000) / 3600

noncomputable def total_distance : ℚ := speed_m_per_s * crossing_time

theorem bridge_length : total_distance - train_length = 205 :=
by
  sorry

end bridge_length_l397_39770


namespace distance_between_tangency_points_l397_39747

theorem distance_between_tangency_points
  (circle_radius : ℝ) (M_distance : ℝ) (A_distance : ℝ) 
  (h1 : circle_radius = 7)
  (h2 : M_distance = 25)
  (h3 : A_distance = 7) :
  ∃ AB : ℝ, AB = 48 :=
by
  -- Definitions and proofs will go here.
  sorry

end distance_between_tangency_points_l397_39747


namespace lcm_36_100_l397_39716

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l397_39716


namespace radius_of_circle_B_l397_39703

-- Definitions of circles and their properties
noncomputable def circle_tangent_externally (r1 r2 : ℝ) := ∃ d : ℝ, d = r1 + r2
noncomputable def circle_tangent_internally (r1 r2 : ℝ) := ∃ d : ℝ, d = r2 - r1

-- Problem statement in Lean 4
theorem radius_of_circle_B
  (rA rB rC rD centerA centerB centerC centerD : ℝ)
  (h_rA : rA = 2)
  (h_congruent_B_C : rB = rC)
  (h_circle_A_tangent_to_B : circle_tangent_externally rA rB)
  (h_circle_A_tangent_to_C : circle_tangent_externally rA rC)
  (h_circle_B_C_tangent_e : circle_tangent_externally rB rC)
  (h_circle_B_D_tangent_i : circle_tangent_internally rB rD)
  (h_center_A_passes_D : centerA = centerD)
  (h_rD : rD = 4) : 
  rB = 1 := sorry

end radius_of_circle_B_l397_39703


namespace johns_age_l397_39702

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 70) : j = 20 := by
  sorry

end johns_age_l397_39702


namespace proof_problem_l397_39707

def x : ℝ := 0.80 * 1750
def y : ℝ := 0.35 * 3000
def z : ℝ := 0.60 * 4500
def w : ℝ := 0.40 * 2800
def a : ℝ := z * w
def b : ℝ := x + y

theorem proof_problem : a - b = 3021550 := by
  sorry

end proof_problem_l397_39707


namespace rational_solutions_k_values_l397_39700

theorem rational_solutions_k_values (k : ℕ) (h₁ : k > 0) 
    (h₂ : ∃ (m : ℤ), 900 - 4 * (k:ℤ)^2 = m^2) : k = 9 ∨ k = 15 := 
by
  sorry

end rational_solutions_k_values_l397_39700


namespace tank_capacity_l397_39781

theorem tank_capacity (C : ℝ) (h1 : 0.40 * C = 0.90 * C - 36) : C = 72 := 
sorry

end tank_capacity_l397_39781


namespace oprah_winfrey_band_weights_l397_39713

theorem oprah_winfrey_band_weights :
  let weight_trombone := 10
  let weight_tuba := 20
  let weight_drum := 15
  let num_trumpets := 6
  let num_clarinets := 9
  let num_trombones := 8
  let num_tubas := 3
  let num_drummers := 2
  let total_weight := 245

  15 * x = total_weight - (num_trombones * weight_trombone + num_tubas * weight_tuba + num_drummers * weight_drum) 
  → x = 5 := by
  sorry

end oprah_winfrey_band_weights_l397_39713


namespace arithmetic_sequence_a_100_l397_39735

theorem arithmetic_sequence_a_100 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 100) → 
  (∀ n : ℕ, a (n + 1) = a n + 2) → 
  a 100 = 298 :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_a_100_l397_39735


namespace compute_value_of_expression_l397_39790

theorem compute_value_of_expression :
  ∃ p q : ℝ, (3 * p^2 - 3 * q^2) / (p - q) = 5 ∧ 3 * p^2 - 5 * p - 14 = 0 ∧ 3 * q^2 - 5 * q - 14 = 0 :=
sorry

end compute_value_of_expression_l397_39790


namespace product_ab_zero_l397_39732

variable {a b : ℝ}

theorem product_ab_zero (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
  sorry

end product_ab_zero_l397_39732


namespace binomial_prime_divisor_l397_39718

theorem binomial_prime_divisor (p k : ℕ) (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end binomial_prime_divisor_l397_39718


namespace C_share_of_profit_l397_39768

-- Given conditions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

-- Objective to prove that C's share of the profit is given by 36000
theorem C_share_of_profit : (total_profit / (investment_A / investment_C + investment_B / investment_C + 1)) = 36000 :=
by
  sorry

end C_share_of_profit_l397_39768


namespace instantaneous_velocity_at_t4_l397_39786

-- Definition of the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The proof problem statement: Proving that the derivative of s at t = 4 is 7
theorem instantaneous_velocity_at_t4 : deriv s 4 = 7 :=
by sorry

end instantaneous_velocity_at_t4_l397_39786


namespace solve_for_x_l397_39759

theorem solve_for_x
  (n m x : ℕ)
  (h1 : 7 / 8 = n / 96)
  (h2 : 7 / 8 = (m + n) / 112)
  (h3 : 7 / 8 = (x - m) / 144) :
  x = 140 :=
by
  sorry

end solve_for_x_l397_39759


namespace sequence_a4_value_l397_39737

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 1) ∧ a 4 = 15 :=
by
  sorry

end sequence_a4_value_l397_39737


namespace trigonometric_identity_l397_39796

theorem trigonometric_identity (α : Real) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10) / Real.sin (α - Real.pi / 5) = 3) :=
sorry

end trigonometric_identity_l397_39796


namespace number_of_baggies_l397_39721

/-- Conditions -/
def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

/-- Question: Prove the total number of baggies Olivia can make is 6 --/
theorem number_of_baggies : (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := sorry

end number_of_baggies_l397_39721


namespace f_2015_l397_39766

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (x + 2) = f (2 - x) + 4 * f 2
axiom symmetric_about_neg1 : ∀ x : ℝ, f (x + 1) = f (-2 - (x + 1))
axiom f_at_1 : f 1 = 3

theorem f_2015 : f 2015 = -3 :=
by
  apply sorry

end f_2015_l397_39766


namespace train_length_l397_39785

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : L = v * 36)
  (h2 : L + 25 = v * 39) :
  L = 300 :=
by
  sorry

end train_length_l397_39785


namespace power_function_at_100_l397_39788

-- Given a power function f(x) = x^α that passes through the point (9, 3),
-- show that f(100) = 10.

theorem power_function_at_100 (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ α)
  (h2 : f 9 = 3) : f 100 = 10 :=
sorry

end power_function_at_100_l397_39788
