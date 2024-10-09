import Mathlib

namespace find_b_l1043_104335

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l1043_104335


namespace electricity_average_l1043_104369

-- Define the daily electricity consumptions
def electricity_consumptions : List ℕ := [110, 101, 121, 119, 114]

-- Define the function to calculate the average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Formalize the proof problem
theorem electricity_average :
  average electricity_consumptions = 113 :=
  sorry

end electricity_average_l1043_104369


namespace martin_initial_spending_l1043_104351

theorem martin_initial_spending :
  ∃ (x : ℝ), 
    ∀ (a b : ℝ), 
      a = x - 100 →
      b = a - 0.20 * a →
      x - b = 280 →
      x = 1000 :=
by
  sorry

end martin_initial_spending_l1043_104351


namespace profit_percentage_with_discount_correct_l1043_104388

variable (CP SP_without_discount Discounted_SP : ℝ)
variable (profit_without_discount profit_with_discount : ℝ)
variable (discount_percentage profit_percentage_without_discount profit_percentage_with_discount : ℝ)
variable (h1 : CP = 100)
variable (h2 : SP_without_discount = CP + profit_without_discount)
variable (h3 : profit_without_discount = 1.20 * CP)
variable (h4 : Discounted_SP = SP_without_discount - discount_percentage * SP_without_discount)
variable (h5 : discount_percentage = 0.05)
variable (h6 : profit_with_discount = Discounted_SP - CP)
variable (h7 : profit_percentage_with_discount = (profit_with_discount / CP) * 100)

theorem profit_percentage_with_discount_correct : profit_percentage_with_discount = 109 := by
  sorry

end profit_percentage_with_discount_correct_l1043_104388


namespace fred_balloon_count_l1043_104390

variable (Fred_balloons Sam_balloons Mary_balloons total_balloons : ℕ)

/-- 
  Given:
  - Fred has some yellow balloons
  - Sam has 6 yellow balloons
  - Mary has 7 yellow balloons
  - Total number of yellow balloons (Fred's, Sam's, and Mary's balloons) is 18

  Prove: Fred has 5 yellow balloons.
-/
theorem fred_balloon_count :
  Sam_balloons = 6 →
  Mary_balloons = 7 →
  total_balloons = 18 →
  Fred_balloons = total_balloons - (Sam_balloons + Mary_balloons) →
  Fred_balloons = 5 :=
by
  sorry

end fred_balloon_count_l1043_104390


namespace line_intersects_xaxis_at_l1043_104386

theorem line_intersects_xaxis_at (x y : ℝ) 
  (h : 4 * y - 5 * x = 15) 
  (hy : y = 0) : (x, y) = (-3, 0) :=
by
  sorry

end line_intersects_xaxis_at_l1043_104386


namespace smallest_AAB_value_l1043_104365

theorem smallest_AAB_value {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_distinct : A ≠ B) (h_eq : 10 * A + B = (1 / 9) * (100 * A + 10 * A + B)) :
  100 * A + 10 * A + B = 225 :=
by
  -- Insert proof here
  sorry

end smallest_AAB_value_l1043_104365


namespace greatest_value_of_x_l1043_104327

theorem greatest_value_of_x : ∀ x : ℝ, 4*x^2 + 6*x + 3 = 5 → x ≤ 1/2 :=
by
  intro x
  intro h
  sorry

end greatest_value_of_x_l1043_104327


namespace discount_percentage_l1043_104375

theorem discount_percentage (sale_price original_price : ℝ) (h1 : sale_price = 480) (h2 : original_price = 600) : 
  100 * (original_price - sale_price) / original_price = 20 := by 
  sorry

end discount_percentage_l1043_104375


namespace Panikovsky_share_l1043_104342

theorem Panikovsky_share :
  ∀ (horns hooves weight : ℕ) 
    (k δ : ℝ),
    horns = 17 →
    hooves = 2 →
    weight = 1 →
    (∀ h, h = k + δ) →
    (∀ wt, wt = k + 2 * δ) →
    (20 * k + 19 * δ) / 2 = 10 * k + 9.5 * δ →
    9 * k + 7.5 * δ = (9 * (k + δ) + 2 * k) →
    ∃ (Panikov_hearts Panikov_hooves : ℕ), 
    Panikov_hearts = 9 ∧ Panikov_hooves = 2 := 
by
  intros
  sorry

end Panikovsky_share_l1043_104342


namespace square_diagonal_l1043_104360

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (hA : A = 338) (hs : s^2 = A) (hd : d^2 = 2 * s^2) : d = 26 :=
by
  -- Proof goes here
  sorry

end square_diagonal_l1043_104360


namespace initial_marbles_count_l1043_104396

-- Definitions as per conditions in the problem
variables (x y z : ℕ)

-- Condition 1: Removing one black marble results in one-eighth of the remaining marbles being black
def condition1 : Prop := (x - 1) * 8 = (x + y - 1)

-- Condition 2: Removing three white marbles results in one-sixth of the remaining marbles being black
def condition2 : Prop := x * 6 = (x + y - 3)

-- Proof that initial total number of marbles is 9 given conditions
theorem initial_marbles_count (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 9 :=
by 
  sorry

end initial_marbles_count_l1043_104396


namespace function_property_l1043_104332

def y (x : ℝ) : ℝ := x - 2

theorem function_property : y 1 = -1 :=
by
  -- place for proof
  sorry

end function_property_l1043_104332


namespace darkCubeValidPositions_l1043_104323

-- Conditions:
-- 1. The structure is made up of twelve identical cubes.
-- 2. The dark cube must be relocated to a position where the surface area remains unchanged.
-- 3. The cubes must touch each other with their entire faces.
-- 4. The positions of the light cubes cannot be changed.

-- Let's define the structure and the conditions in Lean.

structure Cube :=
  (id : ℕ) -- unique identifier for each cube

structure Position :=
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

structure Configuration :=
  (cubes : List Cube)
  (positions : Cube → Position)

def initialCondition (config : Configuration) : Prop :=
  config.cubes.length = 12

def surfaceAreaUnchanged (config : Configuration) (darkCube : Cube) (newPos : Position) : Prop :=
  sorry -- This predicate should capture the logic that the surface area remains unchanged

def validPositions (config : Configuration) (darkCube : Cube) : List Position :=
  sorry -- This function should return the list of valid positions for the dark cube

-- Main theorem: The number of valid positions for the dark cube to maintain the surface area.
theorem darkCubeValidPositions (config : Configuration) (darkCube : Cube) :
    initialCondition config →
    (validPositions config darkCube).length = 3 :=
  by
  sorry

end darkCubeValidPositions_l1043_104323


namespace triangle_structure_twelve_rows_l1043_104306

theorem triangle_structure_twelve_rows :
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  rods 12 + connectors 13 = 325 :=
by
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  sorry

end triangle_structure_twelve_rows_l1043_104306


namespace incorrect_statement_C_l1043_104387

-- Lean 4 statement to verify correctness of problem translation
theorem incorrect_statement_C (n : ℕ) (w : ℕ → ℕ) :
  (w 1 = 55) ∧
  (w 2 = 110) ∧
  (w 3 = 160) ∧
  (w 4 = 200) ∧
  (w 5 = 254) ∧
  (w 6 = 300) ∧
  (w 7 = 350) →
  ¬(∀ n, w n = 55 * n) :=
by
  intros h
  sorry

end incorrect_statement_C_l1043_104387


namespace price_reduction_for_target_profit_l1043_104311
-- Import the necessary libraries

-- Define the conditions
def average_sales_per_day := 70
def initial_profit_per_item := 50
def sales_increase_per_dollar_decrease := 2

-- Define the functions for sales volume increase and profit per item
def sales_volume_increase (x : ℝ) : ℝ := 2 * x
def profit_per_item (x : ℝ) : ℝ := initial_profit_per_item - x

-- Define the function for daily profit
def daily_profit (x : ℝ) : ℝ := (profit_per_item x) * (average_sales_per_day + sales_volume_increase x)

-- State the main theorem
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, daily_profit x = 3572 ∧ x = 12 :=
sorry

end price_reduction_for_target_profit_l1043_104311


namespace minimum_workers_in_team_A_l1043_104385

variable (a b c : ℤ)

theorem minimum_workers_in_team_A (h1 : b + 90 = 2 * (a - 90))
                               (h2 : a + c = 6 * (b - c)) :
  ∃ a ≥ 148, a = 153 :=
by
  sorry

end minimum_workers_in_team_A_l1043_104385


namespace initial_apples_l1043_104310

theorem initial_apples (A : ℕ) 
  (H1 : A - 2 + 4 + 5 = 14) : 
  A = 7 := 
by 
  sorry

end initial_apples_l1043_104310


namespace sara_ate_16_apples_l1043_104313

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l1043_104313


namespace solve_eq1_solve_eq2_l1043_104370

theorem solve_eq1 (x : ℝ) : (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : (x + 3)^3 = -27 ↔ x = -6 :=
by sorry

end solve_eq1_solve_eq2_l1043_104370


namespace periodic_odd_function_l1043_104393

theorem periodic_odd_function (f : ℝ → ℝ) (period : ℝ) (h_periodic : ∀ x, f (x + period) = f x) (h_odd : ∀ x, f (-x) = -f x) (h_value : f (-3) = 1) (α : ℝ) (h_tan : Real.tan α = 2) :
  f (20 * Real.sin α * Real.cos α) = -1 := 
sorry

end periodic_odd_function_l1043_104393


namespace red_balloon_count_l1043_104364

theorem red_balloon_count (total_balloons : ℕ) (green_balloons : ℕ) (red_balloons : ℕ) :
  total_balloons = 17 →
  green_balloons = 9 →
  red_balloons = total_balloons - green_balloons →
  red_balloons = 8 := by
  sorry

end red_balloon_count_l1043_104364


namespace line_through_intersection_parallel_to_given_line_l1043_104373

theorem line_through_intersection_parallel_to_given_line :
  ∃ k : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + k = 0 ↔ (x, y) = (2, 1)) ∧
  (∀ m n : ℝ, (2 * m + 3 * n + 5 = 0 → 2 * m + 3 * n + k = 0))) →
  2 * x + 3 * y - 7 = 0 :=
sorry

end line_through_intersection_parallel_to_given_line_l1043_104373


namespace change_received_correct_l1043_104331

-- Define the conditions
def apples := 5
def cost_per_apple_cents := 80
def paid_dollars := 10

-- Convert the cost per apple to dollars
def cost_per_apple_dollars := (cost_per_apple_cents : ℚ) / 100

-- Calculate the total cost for 5 apples
def total_cost_dollars := apples * cost_per_apple_dollars

-- Calculate the change received
def change_received := paid_dollars - total_cost_dollars

-- Prove that the change received by Margie
theorem change_received_correct : change_received = 6 := by
  sorry

end change_received_correct_l1043_104331


namespace math_problem_l1043_104304

theorem math_problem : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end math_problem_l1043_104304


namespace circle_equation_l1043_104337

theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = r ↔ (x = 0 ∧ y = 0) → ((x - 3) ^ 2 + (y - 1) ^ 2 = 10) :=
by
  sorry

end circle_equation_l1043_104337


namespace determine_ab_l1043_104302

theorem determine_ab :
  ∃ a b : ℝ, 
  (3 + 8 * a = 2 - 3 * b) ∧ 
  (-1 - 6 * a = 4 * b) → 
  a = -1 / 14 ∧ b = -1 / 14 := 
by 
sorry

end determine_ab_l1043_104302


namespace simplify_and_evaluate_division_l1043_104383

theorem simplify_and_evaluate_division (m : ℕ) (h : m = 10) : 
  (1 - (m / (m + 2))) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 1 / 4 :=
by sorry

end simplify_and_evaluate_division_l1043_104383


namespace ratio_second_to_first_l1043_104330

noncomputable def ratio_of_second_to_first (x y z : ℕ) (k : ℕ) : ℕ := sorry

theorem ratio_second_to_first
    (x y z : ℕ)
    (h1 : z = 2 * y)
    (h2 : y = k * x)
    (h3 : (x + y + z) / 3 = 78)
    (h4 : x = 18)
    (k_val : k = 4):
  ratio_of_second_to_first x y z k = 4 := sorry

end ratio_second_to_first_l1043_104330


namespace find_number_l1043_104398

theorem find_number (N Q : ℕ) (h1 : N = 5 * Q) (h2 : Q + N + 5 = 65) : N = 50 :=
by
  sorry

end find_number_l1043_104398


namespace right_angle_case_acute_angle_case_obtuse_angle_case_l1043_104391

-- Definitions
def circumcenter (O : Type) (A B C : Type) : Prop := sorry -- Definition of circumcenter.

def orthocenter (H : Type) (A B C : Type) : Prop := sorry -- Definition of orthocenter.

noncomputable def R : ℝ := sorry -- Circumradius of the triangle.

-- Conditions
variables {A B C O H : Type}
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C)

-- The angles α β γ represent the angles of triangle ABC.
variables {α β γ : ℝ}

-- Statements
-- Case 1: ∠C = 90°
theorem right_angle_case (h_angle_C : γ = 90) (h_H_eq_C : H = C) (h_AB_eq_2R : AB = 2 * R) : AH + BH >= AB := by
  sorry

-- Case 2: ∠C < 90°
theorem acute_angle_case (h_angle_C_lt_90 : γ < 90) : O_in_triangle_AHB := by
  sorry

-- Case 3: ∠C > 90°
theorem obtuse_angle_case (h_angle_C_gt_90 : γ > 90) : AH + BH > 2 * R := by
  sorry

end right_angle_case_acute_angle_case_obtuse_angle_case_l1043_104391


namespace exists_n_consecutive_numbers_l1043_104319

theorem exists_n_consecutive_numbers:
  ∃ n : ℕ, n % 5 = 0 ∧ (n + 1) % 4 = 0 ∧ (n + 2) % 3 = 0 := sorry

end exists_n_consecutive_numbers_l1043_104319


namespace factorial_mod_11_l1043_104379

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_11 : (factorial 13) % 11 = 0 := by
  sorry

end factorial_mod_11_l1043_104379


namespace fraction_irreducible_l1043_104349

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l1043_104349


namespace solve_system_eqns_l1043_104303

theorem solve_system_eqns :
  ∀ x y z : ℝ, 
  (x * y + 5 * y * z - 6 * x * z = -2 * z) ∧
  (2 * x * y + 9 * y * z - 9 * x * z = -12 * z) ∧
  (y * z - 2 * x * z = 6 * z) →
  x = -2 ∧ y = 2 ∧ z = 1 / 6 ∨
  y = 0 ∧ z = 0 ∨
  x = 0 ∧ z = 0 :=
by
  sorry

end solve_system_eqns_l1043_104303


namespace circle_center_and_radius_l1043_104399

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Statement of the center and radius of the circle
theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_equation x y) →
  (∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 3 ∧ k = 0 ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l1043_104399


namespace part1_part2_l1043_104394

open Set

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a

theorem part1 (a : ℝ) (hpos : 0 < a) :
  (∀ x, A x → ¬ B x a) ↔ a ≥ 5 :=
sorry

theorem part2 (a : ℝ) (hpos : 0 < a) :
  (∀ x, (¬ A x → B x a) ∧ ∃ x, ¬ A x ∧ ¬ B x a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part1_part2_l1043_104394


namespace solve_inequality_l1043_104350

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∧ x ≤ -1) ∨ 
  (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨ 
  (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨ 
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  a * x ^ 2 + (a - 2) * x - 2 ≥ 0 := 
sorry

end solve_inequality_l1043_104350


namespace red_cards_pick_ordered_count_l1043_104301

theorem red_cards_pick_ordered_count :
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  (red_cards * (red_cards - 1) = 552) :=
by
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  show (red_cards * (red_cards - 1) = 552)
  sorry

end red_cards_pick_ordered_count_l1043_104301


namespace find_f_7_5_l1043_104353

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- The proof goes here
  sorry

end find_f_7_5_l1043_104353


namespace number_of_customers_l1043_104338

-- Definitions based on conditions
def popularity (p : ℕ) (c w : ℕ) (k : ℝ) : Prop :=
  p = k * (w / c)

-- Given values
def given_values : Prop :=
  ∃ k : ℝ, popularity 15 500 1000 k

-- Problem statement
theorem number_of_customers:
  given_values →
  popularity 15 600 1200 7.5 :=
by
  intro h
  -- Proof omitted
  sorry

end number_of_customers_l1043_104338


namespace corey_needs_more_golf_balls_l1043_104314

-- Defining the constants based on the conditions
def goal : ℕ := 48
def found_on_saturday : ℕ := 16
def found_on_sunday : ℕ := 18

-- The number of golf balls Corey has found over the weekend
def total_found : ℕ := found_on_saturday + found_on_sunday

-- The number of golf balls Corey still needs to find to reach his goal
def remaining : ℕ := goal - total_found

-- The desired theorem statement
theorem corey_needs_more_golf_balls : remaining = 14 := 
by 
  sorry

end corey_needs_more_golf_balls_l1043_104314


namespace largest_value_of_n_l1043_104315

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l1043_104315


namespace calculate_weight_5_moles_Al2O3_l1043_104334

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def molecular_weight_Al2O3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_O)
def moles_Al2O3 : ℝ := 5
def weight_5_moles_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem calculate_weight_5_moles_Al2O3 :
  weight_5_moles_Al2O3 = 509.8 :=
by sorry

end calculate_weight_5_moles_Al2O3_l1043_104334


namespace polynomial_roots_condition_l1043_104300

open Real

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem polynomial_roots_condition (a b : ℤ) (h1 : ∀ x ≠ 0, f (x + x⁻¹) a b = f x a b + f x⁻¹ a b) (h2 : ∃ p q : ℤ, f p a b = 0 ∧ f q a b = 0) : a^2 + b^2 = 13 := by
  sorry

end polynomial_roots_condition_l1043_104300


namespace gcd_45045_30030_l1043_104392

/-- The greatest common divisor of 45045 and 30030 is 15015. -/
theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 :=
by 
  sorry

end gcd_45045_30030_l1043_104392


namespace probability_both_selected_l1043_104324

theorem probability_both_selected 
  (p_jamie : ℚ) (p_tom : ℚ) 
  (h1 : p_jamie = 2/3) 
  (h2 : p_tom = 5/7) : 
  (p_jamie * p_tom = 10/21) :=
by
  sorry

end probability_both_selected_l1043_104324


namespace teamA_teamB_repair_eq_l1043_104333

-- conditions
def teamADailyRepair (x : ℕ) := x -- represent Team A repairing x km/day
def teamBDailyRepair (x : ℕ) := x + 3 -- represent Team B repairing x + 3 km/day
def timeTaken (distance rate: ℕ) := distance / rate -- time = distance / rate

-- Proof problem statement
theorem teamA_teamB_repair_eq (x : ℕ) (hx : x > 0) (hx_plus_3 : x + 3 > 0) :
  timeTaken 6 (teamADailyRepair x) = timeTaken 8 (teamBDailyRepair x) → (6 / x = 8 / (x + 3)) :=
by
  intros h
  sorry

end teamA_teamB_repair_eq_l1043_104333


namespace trey_nail_usage_l1043_104381

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end trey_nail_usage_l1043_104381


namespace part1_part2_l1043_104372

open Real

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - 1| - |x - a|

theorem part1 (a : ℝ) (h : a = 0) :
  {x : ℝ | f x a < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a < 1 → |(1 - 2 * a)^2 / 6| > 3 / 2) 
  : a < -1 :=
by
  sorry

end part1_part2_l1043_104372


namespace solve_abs_inequality_l1043_104358

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end solve_abs_inequality_l1043_104358


namespace island_knights_liars_two_people_l1043_104340

def islanders_knights_and_liars (n : ℕ) : Prop :=
  ∃ (knight liar : ℕ),
    knight + liar = n ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ n → 
      ((i % i = 0 → liar > 0 ∧ knight > 0) ∧ (i % i ≠ 0 → liar > 0)))

theorem island_knights_liars_two_people :
  islanders_knights_and_liars 2 :=
sorry

end island_knights_liars_two_people_l1043_104340


namespace frame_percentage_l1043_104367

theorem frame_percentage : 
  let side_length := 80
  let frame_width := 4
  let total_area := side_length * side_length
  let picture_side_length := side_length - 2 * frame_width
  let picture_area := picture_side_length * picture_side_length
  let frame_area := total_area - picture_area
  let frame_percentage := (frame_area * 100) / total_area
  frame_percentage = 19 := 
by
  sorry

end frame_percentage_l1043_104367


namespace problem_statement_l1043_104321

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (-2^x + b) / (2^(x+1) + a)

theorem problem_statement :
  (∀ (x : ℝ), f (x) 2 1 = -f (-x) 2 1) ∧
  (∀ (t : ℝ), f (t^2 - 2*t) 2 1 + f (2*t^2 - k) 2 1 < 0 → k < -1/3) :=
by
  sorry

end problem_statement_l1043_104321


namespace license_plates_count_l1043_104384

noncomputable def num_license_plates : Nat :=
  let num_w := 26 * 26      -- number of combinations for w
  let num_w_orders := 2     -- two possible orders for w
  let num_digits := 10 ^ 5  -- number of combinations for 5 digits
  let num_positions := 6    -- number of valid positions for w
  2 * num_positions * num_digits * num_w

theorem license_plates_count : num_license_plates = 809280000 := by
  sorry

end license_plates_count_l1043_104384


namespace arithmetic_mean_solve_x_l1043_104377

theorem arithmetic_mean_solve_x (x : ℚ) :
  (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30 → x = 99 / 7 :=
by 
sorry

end arithmetic_mean_solve_x_l1043_104377


namespace percentage_female_on_duty_l1043_104346

-- Definition of conditions
def on_duty_officers : ℕ := 152
def female_on_duty : ℕ := on_duty_officers / 2
def total_female_officers : ℕ := 400

-- Proof goal
theorem percentage_female_on_duty : (female_on_duty * 100) / total_female_officers = 19 := by
  -- We would complete the proof here
  sorry

end percentage_female_on_duty_l1043_104346


namespace arithmetic_sequence_conditions_l1043_104325

open Nat

theorem arithmetic_sequence_conditions (S : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  d < 0 ∧ S 11 > 0 := 
sorry

end arithmetic_sequence_conditions_l1043_104325


namespace average_hours_per_day_l1043_104336

theorem average_hours_per_day (h : ℝ) :
  (3 * h * 12 + 2 * h * 9 = 108) → h = 2 :=
by 
  intro h_condition
  sorry

end average_hours_per_day_l1043_104336


namespace chicken_bucket_feeds_l1043_104352

theorem chicken_bucket_feeds :
  ∀ (cost_per_bucket : ℝ) (total_cost : ℝ) (total_people : ℕ),
  cost_per_bucket = 12 →
  total_cost = 72 →
  total_people = 36 →
  (total_people / (total_cost / cost_per_bucket)) = 6 :=
by
  intros cost_per_bucket total_cost total_people h1 h2 h3
  sorry

end chicken_bucket_feeds_l1043_104352


namespace blood_drops_per_liter_l1043_104307

def mosquito_drops : ℕ := 20
def fatal_blood_loss_liters : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

theorem blood_drops_per_liter (D : ℕ) (total_drops : ℕ) : 
  (total_drops = mosquitoes_to_kill * mosquito_drops) → 
  (fatal_blood_loss_liters * D = total_drops) → 
  D = 5000 := 
  by 
    intros h1 h2
    sorry

end blood_drops_per_liter_l1043_104307


namespace first_part_lent_years_l1043_104345

theorem first_part_lent_years (P P1 P2 : ℝ) (rate1 rate2 : ℝ) (years2 : ℝ) (interest1 interest2 : ℝ) (t : ℝ) 
  (h1 : P = 2717)
  (h2 : P2 = 1672)
  (h3 : P1 = P - P2)
  (h4 : rate1 = 0.03)
  (h5 : rate2 = 0.05)
  (h6 : years2 = 3)
  (h7 : interest1 = P1 * rate1 * t)
  (h8 : interest2 = P2 * rate2 * years2)
  (h9 : interest1 = interest2) :
  t = 8 :=
sorry

end first_part_lent_years_l1043_104345


namespace rate_per_sq_meter_l1043_104380

def length : ℝ := 5.5
def width : ℝ := 3.75
def totalCost : ℝ := 14437.5

theorem rate_per_sq_meter : (totalCost / (length * width)) = 700 := 
by sorry

end rate_per_sq_meter_l1043_104380


namespace smallest_n_not_prime_l1043_104363

theorem smallest_n_not_prime : ∃ n, n = 4 ∧ ∀ m : ℕ, m < 4 → Prime (2 * m + 1) ∧ ¬ Prime (2 * 4 + 1) :=
by
  sorry

end smallest_n_not_prime_l1043_104363


namespace count_negative_values_correct_l1043_104357

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l1043_104357


namespace sum_of_cubes_is_nine_l1043_104355

def sum_of_cubes_of_consecutive_integers (n : ℤ) : ℤ :=
  n^3 + (n + 1)^3

theorem sum_of_cubes_is_nine :
  ∃ n : ℤ, sum_of_cubes_of_consecutive_integers n = 9 :=
by
  sorry

end sum_of_cubes_is_nine_l1043_104355


namespace smallest_possible_value_of_b_l1043_104305

theorem smallest_possible_value_of_b (a b x : ℕ) (h_pos_x : 0 < x)
  (h_gcd : Nat.gcd a b = x + 7)
  (h_lcm : Nat.lcm a b = x * (x + 7))
  (h_a : a = 56)
  (h_x : x = 21) :
  b = 294 := by
  sorry

end smallest_possible_value_of_b_l1043_104305


namespace sum_a_16_to_20_l1043_104366

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom S_def : ∀ n, S n = a 0 * (1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0))
axiom S_5_eq_2 : S 5 = 2
axiom S_10_eq_6 : S 10 = 6

-- Theorem to prove
theorem sum_a_16_to_20 : a 16 + a 17 + a 18 + a 19 + a 20 = 16 :=
by
  sorry

end sum_a_16_to_20_l1043_104366


namespace initial_bottles_calculation_l1043_104320

theorem initial_bottles_calculation (maria_bottles : ℝ) (sister_bottles : ℝ) (left_bottles : ℝ) 
  (H₁ : maria_bottles = 14.0) (H₂ : sister_bottles = 8.0) (H₃ : left_bottles = 23.0) :
  maria_bottles + sister_bottles + left_bottles = 45.0 :=
by
  sorry

end initial_bottles_calculation_l1043_104320


namespace telephone_number_problem_l1043_104317

theorem telephone_number_problem
  (digits : Finset ℕ)
  (A B C D E F G H I J : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_distinct : [A, B, C, D, E, F, G, H, I, J].Nodup)
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_DEF_consecutive_odd : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
  (h_GHIJ_consecutive_even : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
  (h_sum_ABC : A + B + C = 15) :
  A = 9 :=
by
  sorry

end telephone_number_problem_l1043_104317


namespace oak_trees_initially_in_park_l1043_104309

def initialOakTrees (new_oak_trees total_oak_trees_after: ℕ) : ℕ :=
  total_oak_trees_after - new_oak_trees

theorem oak_trees_initially_in_park (new_oak_trees total_oak_trees_after initial_oak_trees : ℕ) 
  (h_new_trees : new_oak_trees = 2) 
  (h_total_after : total_oak_trees_after = 11) 
  (h_correct : initial_oak_trees = 9) : 
  initialOakTrees new_oak_trees total_oak_trees_after = initial_oak_trees := 
by 
  rw [h_new_trees, h_total_after, h_correct]
  sorry

end oak_trees_initially_in_park_l1043_104309


namespace find_B_l1043_104359

variables {a b c A B C : ℝ}

-- Conditions
axiom given_condition_1 : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)

-- Law of Sines
axiom law_of_sines_1 : (c - b) / (c - a) = a / (c + b)

-- Law of Cosines
axiom law_of_cosines_1 : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)

-- Target
theorem find_B : B = Real.pi / 3 := 
sorry

end find_B_l1043_104359


namespace initial_distance_planes_l1043_104397

theorem initial_distance_planes (speed_A speed_B : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_A distance_B : ℝ) (total_distance : ℝ) :
  speed_A = 240 ∧ speed_B = 360 ∧ time_seconds = 72000 ∧ time_hours = 20 ∧ 
  time_hours = time_seconds / 3600 ∧
  distance_A = speed_A * time_hours ∧ 
  distance_B = speed_B * time_hours ∧ 
  total_distance = distance_A + distance_B →
  total_distance = 12000 :=
by
  intros
  sorry

end initial_distance_planes_l1043_104397


namespace tourists_count_l1043_104376

theorem tourists_count (n k : ℕ) (h1 : 2 * n % k = 1) (h2 : 3 * n % k = 13) : k = 23 := 
sorry

end tourists_count_l1043_104376


namespace range_of_f_x_lt_1_l1043_104343

theorem range_of_f_x_lt_1 (x : ℝ) (f : ℝ → ℝ) (h : f x = x^3) : f x < 1 ↔ x < 1 := by
  sorry

end range_of_f_x_lt_1_l1043_104343


namespace ratio_of_newspapers_l1043_104329

theorem ratio_of_newspapers (C L : ℕ) (h1 : C = 42) (h2 : L = C + 23) : C / (C + 23) = 42 / 65 := by
  sorry

end ratio_of_newspapers_l1043_104329


namespace train_initial_speed_l1043_104348

theorem train_initial_speed (x : ℝ) (h : 3 * 25 * (x / V + (2 * x / 20)) = 3 * x) : V = 50 :=
  by
  sorry

end train_initial_speed_l1043_104348


namespace price_of_one_liter_l1043_104322

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l1043_104322


namespace find_least_integer_l1043_104316

theorem find_least_integer (x : ℤ) : (3 * |x| - 4 < 20) → (x ≥ -7) :=
by
  sorry

end find_least_integer_l1043_104316


namespace complement_union_example_l1043_104395

open Set

universe u

variable (U : Set ℕ) (A B : Set ℕ)

def U_def : Set ℕ := {0, 1, 2, 3, 4}
def A_def : Set ℕ := {0, 1, 2}
def B_def : Set ℕ := {2, 3}

theorem complement_union_example :
  (U \ A) ∪ B = {2, 3, 4} := 
by
  -- Proving the theorem considering
  -- complement and union operations on sets
  sorry

end complement_union_example_l1043_104395


namespace highest_student_id_in_sample_l1043_104312

variable (n : ℕ) (start : ℕ) (interval : ℕ)

theorem highest_student_id_in_sample :
  start = 5 → n = 54 → interval = 9 → 6 = n / interval → start = 5 →
  5 + (interval * (6 - 1)) = 50 :=
by
  sorry

end highest_student_id_in_sample_l1043_104312


namespace train_passes_jogger_l1043_104371

noncomputable def speed_of_jogger_kmph := 9
noncomputable def speed_of_train_kmph := 45
noncomputable def jogger_lead_m := 270
noncomputable def train_length_m := 120

noncomputable def speed_of_jogger_mps := speed_of_jogger_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def speed_of_train_mps := speed_of_train_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def relative_speed_mps := speed_of_train_mps - speed_of_jogger_mps
noncomputable def total_distance_m := jogger_lead_m + train_length_m
noncomputable def time_to_pass_jogger := total_distance_m / relative_speed_mps

theorem train_passes_jogger : time_to_pass_jogger = 39 :=
  by
    -- Proof steps would be provided here
    sorry

end train_passes_jogger_l1043_104371


namespace probability_of_7_successes_l1043_104341

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l1043_104341


namespace geometric_sequence_sum_l1043_104382

variable {a b : ℝ} -- Parameters for real numbers a and b
variable (a_ne_zero : a ≠ 0) -- condition a ≠ 0

/-- Proof that in the geometric sequence {a_n}, given a_5 + a_6 = a and a_15 + a_16 = b, 
    a_25 + a_26 = b^2 / a --/
theorem geometric_sequence_sum (a5_plus_a6 : ℕ → ℝ) (a15_plus_a16 : ℕ → ℝ) (a25_plus_a26 : ℕ → ℝ)
  (h1 : a5_plus_a6 5 + a5_plus_a6 6 = a)
  (h2 : a15_plus_a16 15 + a15_plus_a16 16 = b) :
  a25_plus_a26 25 + a25_plus_a26 26 = b^2 / a :=
  sorry

end geometric_sequence_sum_l1043_104382


namespace time_required_painting_rooms_l1043_104328

-- Definitions based on the conditions
def alice_rate := 1 / 4
def bob_rate := 1 / 6
def charlie_rate := 1 / 8
def combined_rate := 13 / 24
def required_time : ℚ := 74 / 13

-- Proof problem statement
theorem time_required_painting_rooms (t : ℚ) :
  (combined_rate) * (t - 2) = 2 ↔ t = required_time :=
by
  sorry

end time_required_painting_rooms_l1043_104328


namespace fraction_simplification_l1043_104339

theorem fraction_simplification (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  (x^2 + x) / (x^2 - 1) = x / (x - 1) :=
by
  -- Hint of expected development environment setting
  sorry

end fraction_simplification_l1043_104339


namespace profit_percentage_l1043_104308

theorem profit_percentage (initial_cost_per_pound : ℝ) (ruined_percent : ℝ) (selling_price_per_pound : ℝ) (desired_profit_percent : ℝ) : 
  initial_cost_per_pound = 0.80 ∧ ruined_percent = 0.10 ∧ selling_price_per_pound = 0.96 → desired_profit_percent = 8 := by
  sorry

end profit_percentage_l1043_104308


namespace evaluate_f_at_2_l1043_104389

def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem evaluate_f_at_2 : f 2 = 4 :=
by
  -- Proof goes here
  sorry

end evaluate_f_at_2_l1043_104389


namespace coeff_x3_l1043_104347

noncomputable def M (n : ℕ) : ℝ := (5 * (1:ℝ) - (1:ℝ)^(1/2)) ^ n
noncomputable def N (n : ℕ) : ℝ := 2 ^ n

theorem coeff_x3 (n : ℕ) (h : M n - N n = 240) : 
  (M 3) = 150 := sorry

end coeff_x3_l1043_104347


namespace midpoint_product_l1043_104318

theorem midpoint_product (x y : ℝ) :
  (∃ B : ℝ × ℝ, B = (x, y) ∧ 
  (4, 6) = ( (2 + B.1) / 2, (9 + B.2) / 2 )) → x * y = 18 :=
by
  -- Placeholder for the proof
  sorry

end midpoint_product_l1043_104318


namespace smallest_possible_value_of_c_l1043_104326

theorem smallest_possible_value_of_c
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (H : ∀ x : ℝ, (a * Real.sin (b * x + c)) ≤ (a * Real.sin (b * 0 + c))) :
  c = Real.pi / 2 :=
by
  sorry

end smallest_possible_value_of_c_l1043_104326


namespace complement_A_in_U_l1043_104361

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}

theorem complement_A_in_U : (U \ A) = {x | -1 <= x ∧ x <= 3} :=
by
  sorry

end complement_A_in_U_l1043_104361


namespace abs_sum_zero_l1043_104378

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l1043_104378


namespace circle_radius_l1043_104368

theorem circle_radius (d : ℝ) (h : d = 10) : d / 2 = 5 :=
by
  sorry

end circle_radius_l1043_104368


namespace total_yards_in_marathons_eq_495_l1043_104374

-- Definitions based on problem conditions
def marathon_miles : ℕ := 26
def marathon_yards : ℕ := 385
def yards_in_mile : ℕ := 1760
def marathons_run : ℕ := 15

-- Main proof statement
theorem total_yards_in_marathons_eq_495
  (miles_per_marathon : ℕ := marathon_miles)
  (yards_per_marathon : ℕ := marathon_yards)
  (yards_per_mile : ℕ := yards_in_mile)
  (marathons : ℕ := marathons_run) :
  let total_yards := marathons * yards_per_marathon
  let remaining_yards := total_yards % yards_per_mile
  remaining_yards = 495 :=
by
  sorry

end total_yards_in_marathons_eq_495_l1043_104374


namespace basketball_games_played_l1043_104344

theorem basketball_games_played (G : ℕ) (H1 : 35 ≤ G) (H2 : 25 ≥ 0) (H3 : 64 = 100 * (48 / (G + 25))):
  G = 50 :=
sorry

end basketball_games_played_l1043_104344


namespace sine_theorem_l1043_104362

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β) 
  (h2 : b / Real.sin β = c / Real.sin γ) 
  (h3 : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α :=
by
  sorry

end sine_theorem_l1043_104362


namespace ball_first_less_than_25_cm_l1043_104356

theorem ball_first_less_than_25_cm (n : ℕ) :
  ∀ n, (200 : ℝ) * (3 / 4) ^ n < 25 ↔ n ≥ 6 := by sorry

end ball_first_less_than_25_cm_l1043_104356


namespace expression_is_integer_l1043_104354

theorem expression_is_integer (n : ℕ) : 
  (3 ^ (2 * n) / 112 - 4 ^ (2 * n) / 63 + 5 ^ (2 * n) / 144) = (k : ℤ) :=
sorry

end expression_is_integer_l1043_104354
