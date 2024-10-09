import Mathlib

namespace darla_total_payment_l1426_142632

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l1426_142632


namespace sum_of_roots_of_quadratic_l1426_142657

theorem sum_of_roots_of_quadratic (x1 x2 : ℝ) (h : x1 * x2 + -(x1 + x2) * 6 + 5 = 0) : x1 + x2 = 6 :=
by
-- Vieta's formulas for the sum of the roots of a quadratic equation state that x1 + x2 = -b / a.
sorry

end sum_of_roots_of_quadratic_l1426_142657


namespace range_of_m_l1426_142667

-- Define the polynomial p(x)
def p (x : ℝ) (m : ℝ) := x^2 + 2*x - m

-- Given conditions: p(1) is false and p(2) is true
theorem range_of_m (m : ℝ) : 
  (p 1 m ≤ 0) ∧ (p 2 m > 0) → (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l1426_142667


namespace factorize_1_factorize_2_l1426_142607

-- Define the variables involved
variables (a x y : ℝ)

-- Problem (1): 18a^2 - 32 = 2 * (3a + 4) * (3a - 4)
theorem factorize_1 (a : ℝ) : 
  18 * a^2 - 32 = 2 * (3 * a + 4) * (3 * a - 4) :=
sorry

-- Problem (2): y - 6xy + 9x^2y = y * (1 - 3x) ^ 2
theorem factorize_2 (x y : ℝ) : 
  y - 6 * x * y + 9 * x^2 * y = y * (1 - 3 * x) ^ 2 :=
sorry

end factorize_1_factorize_2_l1426_142607


namespace tim_points_l1426_142636

theorem tim_points (J T K : ℝ) (h1 : T = J + 20) (h2 : T = K / 2) (h3 : J + T + K = 100) : T = 30 := 
by 
  sorry

end tim_points_l1426_142636


namespace product_of_five_consecutive_divisible_by_30_l1426_142693

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l1426_142693


namespace solve_inequality_system_l1426_142687

theorem solve_inequality_system (x : ℝ) :
  (x + 1 < 4 ∧ 1 - 3 * x ≥ -5) ↔ (x ≤ 2) :=
by
  sorry

end solve_inequality_system_l1426_142687


namespace principal_amount_l1426_142698

theorem principal_amount (A2 A3 : ℝ) (interest : ℝ) (principal : ℝ) (h1 : A2 = 3450) 
  (h2 : A3 = 3655) (h_interest : interest = A3 - A2) (h_principal : principal = A2 - interest) : 
  principal = 3245 :=
by
  sorry

end principal_amount_l1426_142698


namespace sum_first_9000_terms_l1426_142605

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l1426_142605


namespace cos_b_eq_one_div_sqrt_two_l1426_142604

variable {a b c : ℝ} -- Side lengths
variable {A B C : ℝ} -- Angles in radians

-- Conditions of the problem
variables (h1 : c = 2 * a) 
          (h2 : b^2 = a * c) 
          (h3 : a^2 + b^2 = c^2 - 2 * a * b * Real.cos C)
          (h4 : A + B + C = Real.pi)

theorem cos_b_eq_one_div_sqrt_two
    (h1 : c = 2 * a)
    (h2 : b = a * Real.sqrt 2)
    (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
    (h4 : A + B + C = Real.pi )
    : Real.cos B = 1 / Real.sqrt 2 := 
sorry

end cos_b_eq_one_div_sqrt_two_l1426_142604


namespace books_remaining_correct_l1426_142615

-- Define the initial number of book donations
def initial_books : ℕ := 300

-- Define the number of people donating and the number of books each donates
def num_people : ℕ := 10
def books_per_person : ℕ := 5

-- Calculate total books donated by all people
def total_donation : ℕ := num_people * books_per_person

-- Define the number of books borrowed by other people
def borrowed_books : ℕ := 140

-- Calculate the total number of books after donations and then subtract the borrowed books
def total_books_remaining : ℕ := initial_books + total_donation - borrowed_books

-- Prove the total number of books remaining is 210
theorem books_remaining_correct : total_books_remaining = 210 := by
  sorry

end books_remaining_correct_l1426_142615


namespace manu_wins_probability_l1426_142635

def prob_manu_wins : ℚ :=
  let a := (1/2) ^ 5
  let r := (1/2) ^ 4
  a / (1 - r)

theorem manu_wins_probability : prob_manu_wins = 1 / 30 :=
  by
  -- here we would have the proof steps
  sorry

end manu_wins_probability_l1426_142635


namespace decimal_representation_of_7_div_12_l1426_142621

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l1426_142621


namespace each_persons_contribution_l1426_142699

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l1426_142699


namespace cuboid_dimensions_l1426_142696

-- Define the problem conditions and the goal
theorem cuboid_dimensions (x y v : ℕ) :
  (v * (x * y - 1) = 602) ∧ (x * (v * y - 1) = 605) →
  v = x + 3 →
  x = 11 ∧ y = 4 ∧ v = 14 :=
by
  sorry

end cuboid_dimensions_l1426_142696


namespace calculate_expression_l1426_142682

theorem calculate_expression : (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := 
by 
  sorry

end calculate_expression_l1426_142682


namespace marbles_leftover_l1426_142643

theorem marbles_leftover (r p j : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) (hj : j % 8 = 2) : (r + p + j) % 8 = 6 := 
sorry

end marbles_leftover_l1426_142643


namespace exists_triangle_with_prime_angles_l1426_142663

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Definition of being an angle of a triangle
def is_valid_angle (α : ℕ) : Prop := α > 0 ∧ α < 180

-- Main statement
theorem exists_triangle_with_prime_angles :
  ∃ (α β γ : ℕ), is_prime α ∧ is_prime β ∧ is_prime γ ∧ is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧ α + β + γ = 180 :=
by
  sorry

end exists_triangle_with_prime_angles_l1426_142663


namespace plastering_cost_correct_l1426_142692

noncomputable def tank_length : ℝ := 25
noncomputable def tank_width : ℝ := 12
noncomputable def tank_depth : ℝ := 6
noncomputable def cost_per_sqm_paise : ℝ := 75
noncomputable def cost_per_sqm_rupees : ℝ := cost_per_sqm_paise / 100

noncomputable def total_cost_plastering : ℝ :=
  let long_wall_area := 2 * (tank_length * tank_depth)
  let short_wall_area := 2 * (tank_width * tank_depth)
  let bottom_area := tank_length * tank_width
  let total_area := long_wall_area + short_wall_area + bottom_area
  total_area * cost_per_sqm_rupees

theorem plastering_cost_correct : total_cost_plastering = 558 := by
  sorry

end plastering_cost_correct_l1426_142692


namespace smallest_possible_N_l1426_142690

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) 
  (hr : r > 0) (hs : s > 0) (ht : t > 0) (h_sum : p + q + r + s + t = 4020) :
  ∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1005 :=
sorry

end smallest_possible_N_l1426_142690


namespace robot_Y_reaches_B_after_B_reaches_A_l1426_142601

-- Definitions for the setup of the problem
def time_J_to_B (t_J_to_B : ℕ) := t_J_to_B = 12
def time_J_catch_up_B (t_J_catch_up_B : ℕ) := t_J_catch_up_B = 9

-- Main theorem to be proved
theorem robot_Y_reaches_B_after_B_reaches_A : 
  ∀ t_J_to_B t_J_catch_up_B, 
    (time_J_to_B t_J_to_B) → 
    (time_J_catch_up_B t_J_catch_up_B) →
    ∃ t : ℕ, t = 56 :=
by 
  sorry

end robot_Y_reaches_B_after_B_reaches_A_l1426_142601


namespace percentage_increase_l1426_142624

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  ((x - 70) / 70) * 100 = 11 := by
  sorry

end percentage_increase_l1426_142624


namespace intersection_complement_eq_find_a_l1426_142611

-- Proof Goal 1: A ∩ ¬B = {x : ℝ | x ∈ (-∞, -3] ∪ [14, ∞)}

def setA : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def setB : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def negB : Set ℝ := {x | x ≤ -2 ∨ x ≥ 14}

theorem intersection_complement_eq :
  setA ∩ negB = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Proof Goal 2: The range of a such that E ⊆ B

def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem find_a (a : ℝ) :
  (∀ x, E a x → setB x) → a ≥ -1 :=
by
  sorry

end intersection_complement_eq_find_a_l1426_142611


namespace households_using_neither_brands_l1426_142645

def total_households : Nat := 240
def only_brand_A_households : Nat := 60
def both_brands_households : Nat := 25
def ratio_B_to_both : Nat := 3
def only_brand_B_households : Nat := ratio_B_to_both * both_brands_households
def either_brand_households : Nat := only_brand_A_households + only_brand_B_households + both_brands_households
def neither_brand_households : Nat := total_households - either_brand_households

theorem households_using_neither_brands :
  neither_brand_households = 80 :=
by
  -- Proof can be filled out here
  sorry

end households_using_neither_brands_l1426_142645


namespace length_of_top_side_l1426_142644

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end length_of_top_side_l1426_142644


namespace max_b_for_integer_solutions_l1426_142625

theorem max_b_for_integer_solutions (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
sorry

end max_b_for_integer_solutions_l1426_142625


namespace tan_alpha_plus_beta_mul_tan_alpha_l1426_142683

theorem tan_alpha_plus_beta_mul_tan_alpha (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := 
by
  sorry

end tan_alpha_plus_beta_mul_tan_alpha_l1426_142683


namespace Amanda_ticket_sales_goal_l1426_142623

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l1426_142623


namespace jail_time_calculation_l1426_142620

-- Define conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def arrests_per_day : ℕ := 10
def pre_trial_days : ℕ := 4
def half_two_week_sentence_days : ℕ := 7 -- 1 week is half of 2 weeks

-- Define the calculation of the total combined weeks of jail time
def total_combined_weeks_jail_time : ℕ :=
  let total_arrests := arrests_per_day * number_of_cities * days_of_protest
  let total_days_jail_per_person := pre_trial_days + half_two_week_sentence_days
  let total_combined_days_jail_time := total_arrests * total_days_jail_per_person
  total_combined_days_jail_time / 7

-- Theorem statement
theorem jail_time_calculation : total_combined_weeks_jail_time = 9900 := by
  sorry

end jail_time_calculation_l1426_142620


namespace find_angle_B_find_a_plus_c_l1426_142602

variable (A B C a b c S : Real)

-- Conditions
axiom h1 : a = (1 / 2) * c + b * Real.cos C
axiom h2 : S = Real.sqrt 3
axiom h3 : b = Real.sqrt 13

-- Questions (Proving the answers from the problem)
theorem find_angle_B (hA : A = Real.pi - (B + C)) : 
  B = Real.pi / 3 := by
  sorry

theorem find_a_plus_c (hac : (1 / 2) * a * c * Real.sin (Real.pi / 3) = Real.sqrt 3) : 
  a + c = 5 := by
  sorry

end find_angle_B_find_a_plus_c_l1426_142602


namespace baseball_team_groups_l1426_142691

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) (h_new : new_players = 48) (h_return : returning_players = 6) (h_per_group : players_per_group = 6) : (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end baseball_team_groups_l1426_142691


namespace number_of_children_l1426_142664

theorem number_of_children (total_crayons children_crayons children : ℕ) 
  (h1 : children_crayons = 3) 
  (h2 : total_crayons = 18) 
  (h3 : total_crayons = children_crayons * children) : 
  children = 6 := 
by 
  sorry

end number_of_children_l1426_142664


namespace ab_value_l1426_142616

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := 
by
  sorry

end ab_value_l1426_142616


namespace find_special_two_digit_integer_l1426_142676

theorem find_special_two_digit_integer (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : (n + 3) % 3 = 0)
  (h3 : (n + 4) % 4 = 0)
  (h4 : (n + 5) % 5 = 0) :
  n = 60 := by
  sorry

end find_special_two_digit_integer_l1426_142676


namespace area_enclosed_curves_l1426_142658

theorem area_enclosed_curves (a : ℝ) (h1 : (1 + 1/a)^5 = 1024) :
  ∫ x in (0 : ℝ)..1, (x^(1/3) - x^2) = 5/12 :=
sorry

end area_enclosed_curves_l1426_142658


namespace arithmetic_statement_not_basic_l1426_142686

-- Define the basic algorithmic statements as a set
def basic_algorithmic_statements : Set String := 
  {"Input statement", "Output statement", "Assignment statement", "Conditional statement", "Loop statement"}

-- Define the arithmetic statement
def arithmetic_statement : String := "Arithmetic statement"

-- Prove that arithmetic statement is not a basic algorithmic statement
theorem arithmetic_statement_not_basic :
  arithmetic_statement ∉ basic_algorithmic_statements :=
sorry

end arithmetic_statement_not_basic_l1426_142686


namespace average_marks_math_chem_l1426_142660

theorem average_marks_math_chem (M P C : ℝ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end average_marks_math_chem_l1426_142660


namespace percent_defective_units_shipped_l1426_142603

variable (P : Real)
variable (h1 : 0.07 * P = d)
variable (h2 : 0.0035 * P = s)

theorem percent_defective_units_shipped (h1 : 0.07 * P = d) (h2 : 0.0035 * P = s) : 
  (s / d) * 100 = 5 := sorry

end percent_defective_units_shipped_l1426_142603


namespace solution_of_system_l1426_142612

theorem solution_of_system :
  ∃ x y z : ℚ,
    x + 2 * y = 12 ∧
    y + 3 * z = 15 ∧
    3 * x - z = 6 ∧
    x = 54 / 17 ∧
    y = 75 / 17 ∧
    z = 60 / 17 :=
by
  exists 54 / 17, 75 / 17, 60 / 17
  repeat { sorry }

end solution_of_system_l1426_142612


namespace casper_entry_exit_ways_correct_l1426_142672

-- Define the total number of windows
def num_windows : Nat := 8

-- Define the number of ways Casper can enter and exit through different windows
def casper_entry_exit_ways (num_windows : Nat) : Nat :=
  num_windows * (num_windows - 1)

-- Create a theorem to state the problem and its solution
theorem casper_entry_exit_ways_correct : casper_entry_exit_ways num_windows = 56 := by
  sorry

end casper_entry_exit_ways_correct_l1426_142672


namespace max_xy_l1426_142640

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l1426_142640


namespace Isabelle_ticket_cost_l1426_142680

theorem Isabelle_ticket_cost :
  (∀ (week_salary : ℕ) (weeks_worked : ℕ) (brother_ticket_cost : ℕ) (brothers_saved : ℕ) (Isabelle_saved : ℕ),
  week_salary = 3 ∧ weeks_worked = 10 ∧ brother_ticket_cost = 10 ∧ brothers_saved = 5 ∧ Isabelle_saved = 5 →
  Isabelle_saved + (week_salary * weeks_worked) - ((brother_ticket_cost * 2) - brothers_saved) = 15) :=
by
  sorry

end Isabelle_ticket_cost_l1426_142680


namespace determine_linear_relation_l1426_142655

-- Define the set of options
inductive PlotType
| Scatter
| StemAndLeaf
| FrequencyHistogram
| FrequencyLineChart

-- Define the question and state the expected correct answer
def correctPlotTypeForLinearRelation : PlotType :=
  PlotType.Scatter

-- Prove that the correct method for determining linear relation in a set of data is a Scatter plot
theorem determine_linear_relation :
  correctPlotTypeForLinearRelation = PlotType.Scatter :=
by
  sorry

end determine_linear_relation_l1426_142655


namespace four_real_solutions_l1426_142641

-- Definitions used in the problem
def P (x : ℝ) : Prop := (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2 / 3

-- Statement of the problem
theorem four_real_solutions : ∃ (x1 x2 x3 x4 : ℝ), P x1 ∧ P x2 ∧ P x3 ∧ P x4 ∧ 
  ∀ x, P x → (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :=
sorry

end four_real_solutions_l1426_142641


namespace first_term_geometric_series_l1426_142638

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l1426_142638


namespace polygon_sides_l1426_142688

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l1426_142688


namespace sum_of_angles_is_90_l1426_142630

variables (α β γ : ℝ)
-- Given angles marked on squared paper, which imply certain geometric properties
axiom angle_properties : α + β + γ = 90

theorem sum_of_angles_is_90 : α + β + γ = 90 := 
by
  apply angle_properties

end sum_of_angles_is_90_l1426_142630


namespace find_initial_amount_l1426_142629

-- defining conditions
def compound_interest (A P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  A - P

-- main theorem to prove the principal amount
theorem find_initial_amount 
  (A P : ℝ) (r : ℝ)
  (n t : ℕ)
  (h_P : A = P * (1 + r / n)^t)
  (compound_interest_eq : A - P = 1785.98)
  (r_eq : r = 0.20)
  (n_eq : n = 1)
  (t_eq : t = 5) :
  P = 1200 :=
by
  sorry

end find_initial_amount_l1426_142629


namespace zoo_animals_left_l1426_142685

noncomputable def totalAnimalsLeft (x : ℕ) : ℕ := 
  let initialFoxes := 2 * x
  let initialRabbits := 3 * x
  let foxesAfterMove := initialFoxes - 10
  let rabbitsAfterMove := initialRabbits / 2
  foxesAfterMove + rabbitsAfterMove

theorem zoo_animals_left (x : ℕ) (h : 20 * x - 100 = 39 * x / 2) : totalAnimalsLeft x = 690 := by
  sorry

end zoo_animals_left_l1426_142685


namespace caterer_cheapest_option_l1426_142654

theorem caterer_cheapest_option :
  ∃ x : ℕ, x ≥ 42 ∧ (∀ y : ℕ, y ≥ x → (20 * y < 120 + 18 * y) ∧ (20 * y < 250 + 14 * y)) := 
by
  sorry

end caterer_cheapest_option_l1426_142654


namespace total_players_on_team_l1426_142648

theorem total_players_on_team (M W : ℕ) (h1 : W = M + 2) (h2 : (M : ℝ) / W = 0.7777777777777778) : M + W = 16 :=
by 
  sorry

end total_players_on_team_l1426_142648


namespace find_A_l1426_142653

-- Define the condition as an axiom
axiom A : ℝ
axiom condition : A + 10 = 15 

-- Prove that given the condition, A must be 5
theorem find_A : A = 5 := 
by {
  sorry
}

end find_A_l1426_142653


namespace product_of_solutions_of_abs_equation_l1426_142679

theorem product_of_solutions_of_abs_equation : 
  (∃ x1 x2 : ℝ, |5 * x1| + 2 = 47 ∧ |5 * x2| + 2 = 47 ∧ x1 ≠ x2 ∧ x1 * x2 = -81) :=
sorry

end product_of_solutions_of_abs_equation_l1426_142679


namespace initial_money_l1426_142633

-- Definitions based on conditions in the problem
def money_left_after_purchase : ℕ := 3
def cost_of_candy_bar : ℕ := 1

-- Theorem statement to prove the initial amount of money
theorem initial_money (initial_amount : ℕ) :
  initial_amount - cost_of_candy_bar = money_left_after_purchase → initial_amount = 4 :=
sorry

end initial_money_l1426_142633


namespace inequality_solution_empty_l1426_142639

theorem inequality_solution_empty {a x: ℝ} : 
  (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0 → 
  (-2 < a) ∧ (a < 6 / 5) :=
sorry

end inequality_solution_empty_l1426_142639


namespace volume_ratio_surface_area_ratio_l1426_142637

theorem volume_ratio_surface_area_ratio (V1 V2 S1 S2 : ℝ) (h : V1 / V2 = 8 / 27) :
  S1 / S2 = 4 / 9 :=
by
  sorry

end volume_ratio_surface_area_ratio_l1426_142637


namespace n_divisibility_and_factors_l1426_142619

open Nat

theorem n_divisibility_and_factors (n : ℕ) (h1 : 1990 ∣ n) (h2 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n):
  n = 4 * 5 * 199 ∨ n = 2 * 25 * 199 ∨ n = 2 * 5 * 39601 := 
sorry

end n_divisibility_and_factors_l1426_142619


namespace possible_values_of_A_l1426_142666

theorem possible_values_of_A :
  ∃ (A : ℕ), (A ≤ 4 ∧ A < 10) ∧ A = 5 :=
sorry

end possible_values_of_A_l1426_142666


namespace quadratic_intersection_l1426_142669

def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_intersection:
  ∃ a b c : ℝ, 
  quadratic a b c (-3) = 16 ∧ 
  quadratic a b c 0 = -5 ∧ 
  quadratic a b c 3 = -8 ∧ 
  quadratic a b c (-1) = 0 :=
sorry

end quadratic_intersection_l1426_142669


namespace degree_equality_l1426_142617

theorem degree_equality (m : ℕ) :
  (∀ x y z : ℕ, 2 + 4 = 1 + (m + 2)) → 3 * m - 2 = 7 :=
by
  intro h
  sorry

end degree_equality_l1426_142617


namespace extreme_point_inequality_l1426_142646

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - a / x - 2 * Real.log x

theorem extreme_point_inequality (x₁ x₂ a : ℝ) (h1 : x₁ < x₂) (h2 : f x₁ a = 0) (h3 : f x₂ a = 0) 
(h_a_range : 0 < a) (h_a_lt_1 : a < 1) :
  f x₂ a < x₂ - 1 :=
sorry

end extreme_point_inequality_l1426_142646


namespace range_of_a_l1426_142608

theorem range_of_a (x a : ℝ) 
  (h₁ : ∀ x, |x + 1| ≤ 2 → x ≤ a) 
  (h₂ : ∃ x, x > a ∧ |x + 1| ≤ 2) 
  : a ≥ 1 :=
sorry

end range_of_a_l1426_142608


namespace first_number_value_l1426_142642

theorem first_number_value (A B LCM HCF : ℕ) (h_lcm : LCM = 2310) (h_hcf : HCF = 30) (h_b : B = 210) (h_mul : A * B = LCM * HCF) : A = 330 := 
by
  -- Use sorry to skip the proof
  sorry

end first_number_value_l1426_142642


namespace terms_before_five_l1426_142681

theorem terms_before_five (a₁ : ℤ) (d : ℤ) (n : ℤ) :
  a₁ = 75 → d = -5 → (a₁ + (n - 1) * d = 5) → n - 1 = 14 :=
by
  intros h1 h2 h3
  sorry

end terms_before_five_l1426_142681


namespace geometric_sequence_problem_l1426_142628

noncomputable def geometric_sum (a q : ℕ) (n : ℕ) : ℕ :=
  a * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem (a : ℕ) (q : ℕ) (n : ℕ) (h_q : q = 2) (h_n : n = 4) :
  (geometric_sum a q 4) / (a * q) = 15 / 2 :=
by
  sorry

end geometric_sequence_problem_l1426_142628


namespace max_weight_of_flock_l1426_142675

def MaxWeight (A E Af: ℕ): ℕ := A * 5 + E * 10 + Af * 15

theorem max_weight_of_flock :
  ∀ (A E Af: ℕ),
    A = 2 * E →
    Af = 3 * A →
    A + E + Af = 120 →
    MaxWeight A E Af = 1415 :=
by
  sorry

end max_weight_of_flock_l1426_142675


namespace find_legs_of_triangle_l1426_142674

theorem find_legs_of_triangle (a b : ℝ) (h : a / b = 3 / 4) (h_sum : a^2 + b^2 = 70^2) : 
  (a = 42) ∧ (b = 56) :=
sorry

end find_legs_of_triangle_l1426_142674


namespace cost_of_apple_is_two_l1426_142661

-- Define the costs and quantities
def cost_of_apple (A : ℝ) : Prop :=
  let total_cost := 12 * A + 4 * 1 + 4 * 3
  let total_pieces := 12 + 4 + 4
  let average_cost := 2
  total_cost = total_pieces * average_cost

theorem cost_of_apple_is_two : cost_of_apple 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cost_of_apple_is_two_l1426_142661


namespace number_of_rods_in_one_mile_l1426_142689

theorem number_of_rods_in_one_mile (miles_to_furlongs : 1 = 10 * 1)
  (furlongs_to_rods : 1 = 50 * 1) : 1 = 500 * 1 :=
by {
  sorry
}

end number_of_rods_in_one_mile_l1426_142689


namespace cherries_per_pound_l1426_142627

-- Definitions from conditions in the problem
def total_pounds_of_cherries : ℕ := 3
def pitting_time_for_20_cherries : ℕ := 10 -- in minutes
def total_pitting_time : ℕ := 2 * 60  -- in minutes (2 hours to minutes)

-- Theorem to prove the question equals the correct answer
theorem cherries_per_pound : (total_pitting_time / pitting_time_for_20_cherries) * 20 / total_pounds_of_cherries = 80 := by
  sorry

end cherries_per_pound_l1426_142627


namespace perpendicular_MP_MQ_l1426_142670

variable (k m : ℝ)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

def line (x y : ℝ) := y = k*x + m

def fixed_point_exists (k m : ℝ) : Prop :=
  let P := (-(4 * k) / m, 3 / m)
  let Q := (4, 4 * k + m)
  ∃ (M : ℝ), (M = 1 ∧ ((P.1 - M) * (Q.1 - M) + P.2 * Q.2 = 0))

theorem perpendicular_MP_MQ : fixed_point_exists k m := sorry

end perpendicular_MP_MQ_l1426_142670


namespace sum_of_vertical_asymptotes_l1426_142656

noncomputable def sum_of_roots (a b c : ℝ) (h_discriminant : b^2 - 4*a*c ≠ 0) : ℝ :=
-(b/a)

theorem sum_of_vertical_asymptotes :
  let f := (6 * (x^2) - 8) / (4 * (x^2) + 7*x + 3)
  ∃ c d, c ≠ d ∧ (4*c^2 + 7*c + 3 = 0) ∧ (4*d^2 + 7*d + 3 = 0)
  ∧ c + d = -7 / 4 :=
by
  sorry

end sum_of_vertical_asymptotes_l1426_142656


namespace prevent_four_digit_number_l1426_142652

theorem prevent_four_digit_number (N : ℕ) (n : ℕ) :
  n = 123 + 102 * N ∧ ∀ x : ℕ, (3 + 2 * x) % 10 < 1000 → x < 1000 := 
sorry

end prevent_four_digit_number_l1426_142652


namespace inverse_tangent_line_l1426_142677

theorem inverse_tangent_line
  (f : ℝ → ℝ)
  (hf₁ : ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x) 
  (hf₂ : ∀ x, deriv f x ≠ 0)
  (h_tangent : ∀ x₀, (2 * x₀ - f x₀ + 3) = 0) :
  ∀ x₀, (x₀ - 2 * f x₀ - 3) = 0 :=
by
  sorry

end inverse_tangent_line_l1426_142677


namespace bacteria_after_10_hours_l1426_142678

def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

theorem bacteria_after_10_hours : bacteria_count 10 = 1024 := by
  sorry

end bacteria_after_10_hours_l1426_142678


namespace roof_ratio_l1426_142684

theorem roof_ratio (L W : ℕ) (h1 : L * W = 768) (h2 : L - W = 32) : L / W = 3 := 
sorry

end roof_ratio_l1426_142684


namespace trail_length_is_48_meters_l1426_142695

noncomputable def length_of_trail (d: ℝ) : Prop :=
  let normal_speed := 8 -- normal speed in m/s
  let mud_speed := normal_speed / 4 -- speed in mud in m/s

  let time_mud := (1 / 3 * d) / mud_speed -- time through the mud in seconds
  let time_normal := (2 / 3 * d) / normal_speed -- time through the normal trail in seconds

  let total_time := 12 -- total time in seconds

  total_time = time_mud + time_normal

theorem trail_length_is_48_meters : ∃ d: ℝ, length_of_trail d ∧ d = 48 :=
sorry

end trail_length_is_48_meters_l1426_142695


namespace servant_leaving_months_l1426_142622

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end servant_leaving_months_l1426_142622


namespace molecular_weight_of_compound_l1426_142606

def n_weight : ℝ := 14.01
def h_weight : ℝ := 1.01
def br_weight : ℝ := 79.90

def molecular_weight : ℝ := (1 * n_weight) + (4 * h_weight) + (1 * br_weight)

theorem molecular_weight_of_compound :
  molecular_weight = 97.95 :=
by
  -- proof steps go here if needed, but currently, we use sorry to complete the theorem
  sorry

end molecular_weight_of_compound_l1426_142606


namespace additional_time_due_to_leak_l1426_142649

theorem additional_time_due_to_leak (fill_time_no_leak: ℝ) (leak_empty_time: ℝ) (fill_rate_no_leak: fill_time_no_leak ≠ 0):
  (fill_time_no_leak = 3) → 
  (leak_empty_time = 12) → 
  (1 / fill_time_no_leak - 1 / leak_empty_time ≠ 0) → 
  ((1 / fill_time_no_leak - 1 / leak_empty_time) / (1 / (1 / fill_time_no_leak - 1 / leak_empty_time)) - fill_time_no_leak = 1) := 
by
  intro h_fill h_leak h_effective_rate
  sorry

end additional_time_due_to_leak_l1426_142649


namespace degree_of_g_l1426_142647

open Polynomial

theorem degree_of_g (f g : Polynomial ℂ) (h1 : f = -3 * X^5 + 4 * X^4 - X^2 + C 2) (h2 : degree (f + g) = 2) : degree g = 5 :=
sorry

end degree_of_g_l1426_142647


namespace football_game_attendance_l1426_142662

theorem football_game_attendance :
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  wednesday - monday = 50 :=
by
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  show wednesday - monday = 50
  sorry

end football_game_attendance_l1426_142662


namespace Im_abcd_eq_zero_l1426_142634

noncomputable def normalized (z : ℂ) : ℂ := z / Complex.abs z

theorem Im_abcd_eq_zero (a b c d : ℂ)
  (h1 : ∃ α : ℝ, ∃ w : ℂ, w = Complex.cos α + Complex.sin α * Complex.I ∧ (normalized b = w * normalized a) ∧ (normalized d = w * normalized c)) :
  Complex.im (a * b * c * d) = 0 :=
by
  sorry

end Im_abcd_eq_zero_l1426_142634


namespace seq_common_max_l1426_142650

theorem seq_common_max : ∃ a, a ≤ 250 ∧ 1 ≤ a ∧ a % 8 = 1 ∧ a % 9 = 4 ∧ ∀ b, b ≤ 250 ∧ 1 ≤ b ∧ b % 8 = 1 ∧ b % 9 = 4 → b ≤ a :=
by 
  sorry

end seq_common_max_l1426_142650


namespace find_abs_product_abc_l1426_142673

theorem find_abs_product_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h : a + 1 / b = b + 1 / c ∧ b + 1 / c = c + 1 / a) : |a * b * c| = 1 :=
sorry

end find_abs_product_abc_l1426_142673


namespace a_plus_b_eq_zero_l1426_142613

-- Define the universal set and the relevant sets
def U : Set ℝ := Set.univ
def M (a : ℝ) : Set ℝ := {x | x^2 + a * x ≤ 0}
def C_U_M (b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Define the proof theorem
theorem a_plus_b_eq_zero (a b : ℝ) (h1 : ∀ x, x ∈ M a ↔ -a < x ∧ x < 0 ∨ 0 < x ∧ x < -a)
                         (h2 : ∀ x, x ∈ C_U_M b ↔ x > b ∨ x < 0) : a + b = 0 := 
sorry

end a_plus_b_eq_zero_l1426_142613


namespace problem1_problem2_l1426_142618

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| - |2 * x + 1|

theorem problem1 (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ -1/3 ≤ x :=
sorry

theorem problem2 (a : ℝ) (b : ℝ) :
  (∀ x, |a + b| - |a - b| ≥ f x) → (a ≥ 5 / 4 ∨ a ≤ -5 / 4) :=
sorry

end problem1_problem2_l1426_142618


namespace dishonest_dealer_profit_l1426_142609

theorem dishonest_dealer_profit (cost_weight actual_weight : ℝ) (kg_in_g : ℝ) 
  (h1 : cost_weight = 1000) (h2 : actual_weight = 920) (h3 : kg_in_g = 1000) :
  ((cost_weight - actual_weight) / actual_weight) * 100 = 8.7 := by
  sorry

end dishonest_dealer_profit_l1426_142609


namespace third_studio_students_l1426_142671

theorem third_studio_students 
  (total_students : ℕ)
  (first_studio : ℕ)
  (second_studio : ℕ) 
  (third_studio : ℕ) 
  (h1 : total_students = 376) 
  (h2 : first_studio = 110) 
  (h3 : second_studio = 135) 
  (h4 : total_students = first_studio + second_studio + third_studio) :
  third_studio = 131 := 
sorry

end third_studio_students_l1426_142671


namespace company_fund_initial_amount_l1426_142668

theorem company_fund_initial_amount
  (n : ℕ) -- number of employees
  (initial_bonus_per_employee : ℕ := 60)
  (shortfall : ℕ := 10)
  (revised_bonus_per_employee : ℕ := 50)
  (fund_remaining : ℕ := 150)
  (initial_fund : ℕ := initial_bonus_per_employee * n - shortfall) -- condition that the fund was $10 short when planning the initial bonus
  (revised_fund : ℕ := revised_bonus_per_employee * n + fund_remaining) -- condition after distributing the $50 bonuses

  (eqn : initial_fund = revised_fund) -- equating initial and revised budget calculations
  
  : initial_fund = 950 := 
sorry

end company_fund_initial_amount_l1426_142668


namespace exists_distinct_abc_sum_l1426_142651

theorem exists_distinct_abc_sum (n : ℕ) (h : n ≥ 1) (X : Finset ℤ)
  (h_card : X.card = n + 2)
  (h_abs : ∀ x ∈ X, abs x ≤ n) :
  ∃ (a b c : ℤ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end exists_distinct_abc_sum_l1426_142651


namespace find_quartic_polynomial_l1426_142614

noncomputable def p (x : ℝ) : ℝ := -(1 / 9) * x^4 + (40 / 9) * x^3 - 8 * x^2 + 10 * x + 2

theorem find_quartic_polynomial :
  p 1 = -3 ∧
  p 2 = -1 ∧
  p 3 = 1 ∧
  p 4 = -7 ∧
  p 0 = 2 :=
by
  sorry

end find_quartic_polynomial_l1426_142614


namespace initial_pieces_l1426_142659

-- Definitions of the conditions
def pieces_eaten : ℕ := 7
def pieces_given : ℕ := 21
def pieces_now : ℕ := 37

-- The proposition to prove
theorem initial_pieces (C : ℕ) (h : C - pieces_eaten + pieces_given = pieces_now) : C = 23 :=
by
  -- Proof would go here
  sorry

end initial_pieces_l1426_142659


namespace find_raspberries_l1426_142631

def total_berries (R : ℕ) : ℕ := 30 + 20 + R

def fresh_berries (R : ℕ) : ℕ := 2 * total_berries R / 3

def fresh_berries_to_keep (R : ℕ) : ℕ := fresh_berries R / 2

def fresh_berries_to_sell (R : ℕ) : ℕ := fresh_berries R - fresh_berries_to_keep R

theorem find_raspberries (R : ℕ) : fresh_berries_to_sell R = 20 → R = 10 := 
by 
sorry

-- To ensure the problem is complete and solvable, we also need assumptions on the domain:
example : ∃ R : ℕ, fresh_berries_to_sell R = 20 := 
by 
  use 10 
  sorry

end find_raspberries_l1426_142631


namespace max_squares_fitting_l1426_142694

theorem max_squares_fitting (L S : ℕ) (hL : L = 8) (hS : S = 2) : (L / S) * (L / S) = 16 := by
  -- Proof goes here
  sorry

end max_squares_fitting_l1426_142694


namespace Brians_trip_distance_l1426_142626

theorem Brians_trip_distance (miles_per_gallon : ℕ) (gallons_used : ℕ) (distance_traveled : ℕ) 
  (h1 : miles_per_gallon = 20) (h2 : gallons_used = 3) : 
  distance_traveled = 60 :=
by
  sorry

end Brians_trip_distance_l1426_142626


namespace bus_full_problem_l1426_142697

theorem bus_full_problem
      (cap : ℕ := 80)
      (first_pickup_ratio : ℚ := 3/5)
      (second_pickup_exit : ℕ := 15)
      (waiting_people : ℕ := 50) :
      waiting_people - (cap - (first_pickup_ratio * cap - second_pickup_exit)) = 3 := by
  sorry

end bus_full_problem_l1426_142697


namespace percentage_difference_l1426_142610

theorem percentage_difference (G P R : ℝ) (h1 : P = 0.9 * G) (h2 : R = 1.125 * G) :
  ((1 - P / R) * 100) = 20 :=
by
  sorry

end percentage_difference_l1426_142610


namespace range_of_a_l1426_142665

noncomputable def condition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
noncomputable def condition_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬(∀ x, condition_p x)) → (¬(∀ x, condition_q x a)) → 
  (∀ x, condition_p x ↔ condition_q x a) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l1426_142665


namespace candies_total_l1426_142600

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l1426_142600
