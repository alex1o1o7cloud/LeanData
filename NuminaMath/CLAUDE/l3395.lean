import Mathlib

namespace NUMINAMATH_CALUDE_routes_8x5_grid_l3395_339558

/-- The number of routes on a grid from (0,0) to (m,n) where only right and up movements are allowed -/
def numRoutes (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The theorem stating that the number of routes on an 8x5 grid is 12870 -/
theorem routes_8x5_grid : numRoutes 8 5 = 12870 := by sorry

end NUMINAMATH_CALUDE_routes_8x5_grid_l3395_339558


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3395_339590

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3395_339590


namespace NUMINAMATH_CALUDE_sugar_recipe_calculation_l3395_339570

theorem sugar_recipe_calculation (initial_required : ℚ) (available : ℚ) : 
  initial_required = 1/3 → available = 1/6 → 
  (initial_required - available = 1/6) ∧ (2 * (initial_required - available) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_sugar_recipe_calculation_l3395_339570


namespace NUMINAMATH_CALUDE_firm_employs_50_looms_l3395_339580

/-- Represents the number of looms employed by a textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for a month, in rupees. -/
def profit_decrease : ℕ := 7000

/-- Theorem stating that the number of looms employed by the firm is 50. -/
theorem firm_employs_50_looms :
  number_of_looms = 50 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end NUMINAMATH_CALUDE_firm_employs_50_looms_l3395_339580


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3395_339505

theorem quadratic_equation_value (x : ℝ) (h : x^2 - 3*x = 4) : 2*x^2 - 6*x - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3395_339505


namespace NUMINAMATH_CALUDE_sum_of_possible_distances_l3395_339554

/-- Given two points A and B on a number line, where the distance between A and B is 2,
    and the distance between A and the origin O is 3,
    the sum of all possible distances between B and the origin O is 12. -/
theorem sum_of_possible_distances (A B : ℝ) : 
  (|A - B| = 2) → (|A| = 3) → (|B| + |-B| + |B - 2| + |-(B - 2)| = 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_distances_l3395_339554


namespace NUMINAMATH_CALUDE_islander_group_composition_l3395_339597

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents an islander's statement about the group composition -/
inductive Statement
| MoreLiars
| MoreKnights
| Equal

/-- A function that returns the true statement about group composition -/
def trueStatement (knights liars : Nat) : Statement :=
  if knights > liars then Statement.MoreKnights
  else if liars > knights then Statement.MoreLiars
  else Statement.Equal

/-- A function that determines what an islander would say based on their type and the true group composition -/
def islanderStatement (type : IslanderType) (knights liars : Nat) : Statement :=
  match type with
  | IslanderType.Knight => trueStatement knights liars
  | IslanderType.Liar => 
    match trueStatement knights liars with
    | Statement.MoreLiars => Statement.MoreKnights
    | Statement.MoreKnights => Statement.MoreLiars
    | Statement.Equal => Statement.MoreLiars  -- Arbitrarily chosen, could be MoreKnights as well

theorem islander_group_composition 
  (total : Nat) 
  (h_total : total = 10) 
  (knights liars : Nat) 
  (h_sum : knights + liars = total) 
  (h_five_more_liars : ∃ (group : Finset IslanderType), 
    group.card = 5 ∧ 
    ∀ i ∈ group, islanderStatement i knights liars = Statement.MoreLiars) :
  knights = liars ∧ 
  ∃ (other_group : Finset IslanderType), 
    other_group.card = 5 ∧ 
    ∀ i ∈ other_group, islanderStatement i knights liars = Statement.Equal :=
sorry


end NUMINAMATH_CALUDE_islander_group_composition_l3395_339597


namespace NUMINAMATH_CALUDE_kidney_apples_amount_l3395_339547

/-- The amount of golden apples in kg -/
def golden_apples : ℕ := 37

/-- The amount of Canada apples in kg -/
def canada_apples : ℕ := 14

/-- The amount of apples sold in kg -/
def apples_sold : ℕ := 36

/-- The amount of apples left in kg -/
def apples_left : ℕ := 38

/-- The amount of kidney apples in kg -/
def kidney_apples : ℕ := 23

theorem kidney_apples_amount :
  kidney_apples = apples_left + apples_sold - golden_apples - canada_apples :=
by sorry

end NUMINAMATH_CALUDE_kidney_apples_amount_l3395_339547


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l3395_339520

theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l3395_339520


namespace NUMINAMATH_CALUDE_yumis_farm_chickens_l3395_339569

/-- The number of chickens on Yumi's farm -/
def num_chickens : ℕ := 6

/-- The number of pigs on Yumi's farm -/
def num_pigs : ℕ := 9

/-- The number of legs each pig has -/
def pig_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of legs of all animals on Yumi's farm -/
def total_legs : ℕ := 48

theorem yumis_farm_chickens :
  num_chickens * chicken_legs + num_pigs * pig_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_yumis_farm_chickens_l3395_339569


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3395_339524

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4*z) = 7 ∧ z = -23/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3395_339524


namespace NUMINAMATH_CALUDE_money_problem_l3395_339535

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + 2 * b > 110)
  (h2 : 2 * a + 3 * b = 105) :
  a > 15 ∧ b < 25 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3395_339535


namespace NUMINAMATH_CALUDE_mailman_delivery_l3395_339599

theorem mailman_delivery (total_mail junk_mail : ℕ) 
  (h1 : total_mail = 11) 
  (h2 : junk_mail = 6) : 
  total_mail - junk_mail = 5 := by
  sorry

end NUMINAMATH_CALUDE_mailman_delivery_l3395_339599


namespace NUMINAMATH_CALUDE_equation_solution_in_interval_l3395_339595

theorem equation_solution_in_interval :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 3^x + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_in_interval_l3395_339595


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3395_339586

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |5 * y - 6| = 0 ∧ y = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3395_339586


namespace NUMINAMATH_CALUDE_parabola_through_three_points_l3395_339508

/-- A parabola with equation y = x^2 + bx + c passing through (-1, -11), (3, 17), and (2, 5) has b = 13/3 and c = -5 -/
theorem parabola_through_three_points :
  ∀ b c : ℚ,
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 5) →
  (b = 13/3 ∧ c = -5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_three_points_l3395_339508


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3395_339510

theorem condition_neither_sufficient_nor_necessary
  (m n : ℕ+) :
  ¬(∀ a b : ℝ, a > b → (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0) ∧
  ¬(∀ a b : ℝ, (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3395_339510


namespace NUMINAMATH_CALUDE_investment_solution_l3395_339582

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  years : ℕ
  finalAmount : ℝ

/-- Calculates the final amount after compound interest -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the solution to the investment problem -/
theorem investment_solution (inv : Investment) 
  (h1 : inv.total = 1500)
  (h2 : inv.rate1 = 0.04)
  (h3 : inv.rate2 = 0.06)
  (h4 : inv.years = 3)
  (h5 : inv.finalAmount = 1824.89) :
  ∃ (x : ℝ), x = 580 ∧ 
    compoundInterest x inv.rate1 inv.years + 
    compoundInterest (inv.total - x) inv.rate2 inv.years = 
    inv.finalAmount := by
  sorry


end NUMINAMATH_CALUDE_investment_solution_l3395_339582


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3395_339585

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 + 2*x + 3 = (x + 1)^2 + 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3395_339585


namespace NUMINAMATH_CALUDE_logarithmic_equality_l3395_339532

noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log ((7 * x / 2) - (17 / 4)) / Real.log ((x / 2 + 1)^2)

noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log ((3 * x / 2) - 6)^2 / Real.log (((7 * x / 2) - (17 / 4))^(1/2))

noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (x / 2 + 1) / Real.log (((3 * x / 2) - 6)^(1/2))

theorem logarithmic_equality (x : ℝ) :
  (log_expr1 x = log_expr2 x ∧ log_expr1 x = log_expr3 x + 1) ∨
  (log_expr2 x = log_expr3 x ∧ log_expr2 x = log_expr1 x + 1) ∨
  (log_expr3 x = log_expr1 x ∧ log_expr3 x = log_expr2 x + 1) ↔
  x = 7 :=
sorry

end NUMINAMATH_CALUDE_logarithmic_equality_l3395_339532


namespace NUMINAMATH_CALUDE_man_speed_man_speed_result_l3395_339507

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  let train_speed_mps := train_speed * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * 3600 / 1000
  man_speed_kmph

/-- The speed of the man is approximately 6.0024 km/h --/
theorem man_speed_result : 
  ∃ ε > 0, |man_speed 60 110 5.999520038396929 - 6.0024| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_result_l3395_339507


namespace NUMINAMATH_CALUDE_combined_class_average_weight_l3395_339553

/-- Calculates the average weight of a combined class given two sections -/
def averageWeightCombinedClass (studentsA : ℕ) (studentsB : ℕ) (avgWeightA : ℚ) (avgWeightB : ℚ) : ℚ :=
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB)

/-- Theorem stating the average weight of the combined class -/
theorem combined_class_average_weight :
  averageWeightCombinedClass 26 34 50 30 = 2320 / 60 := by
  sorry

#eval averageWeightCombinedClass 26 34 50 30

end NUMINAMATH_CALUDE_combined_class_average_weight_l3395_339553


namespace NUMINAMATH_CALUDE_largest_divisor_power_l3395_339517

-- Define the expression A
def A : ℕ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

-- State the theorem
theorem largest_divisor_power (k : ℕ) : (∀ m : ℕ, m > k → ¬(1991^m ∣ A)) ∧ (1991^k ∣ A) ↔ k = 1991 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l3395_339517


namespace NUMINAMATH_CALUDE_choir_size_proof_l3395_339578

theorem choir_size_proof (n : ℕ) : 
  (∃ (p : ℕ), p > 10 ∧ Prime p ∧ p ∣ n) ∧ 
  9 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n →
  n ≥ 1980 :=
by sorry

end NUMINAMATH_CALUDE_choir_size_proof_l3395_339578


namespace NUMINAMATH_CALUDE_total_money_l3395_339573

/-- Given that A and C together have 400, B and C together have 750, and C has 250,
    prove that the total amount of money A, B, and C have between them is 900. -/
theorem total_money (a b c : ℕ) 
  (h1 : a + c = 400)
  (h2 : b + c = 750)
  (h3 : c = 250) :
  a + b + c = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3395_339573


namespace NUMINAMATH_CALUDE_elective_course_selection_l3395_339501

def category_A : ℕ := 3
def category_B : ℕ := 4
def total_courses : ℕ := 3

theorem elective_course_selection :
  (Nat.choose category_A 1 * Nat.choose category_B 2) +
  (Nat.choose category_A 2 * Nat.choose category_B 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_selection_l3395_339501


namespace NUMINAMATH_CALUDE_tangerines_remain_odd_last_fruit_is_tangerine_l3395_339542

/-- Represents the types of fruits in the vase -/
inductive Fruit
| Tangerine
| Apple

/-- Represents the state of the vase -/
structure VaseState where
  tangerines : Nat
  apples : Nat

/-- Represents the action of taking fruits -/
inductive TakeAction
| TwoTangerines
| TangerineAndApple
| TwoApples

/-- Function to update the vase state based on the take action -/
def updateVase (state : VaseState) (action : TakeAction) : VaseState :=
  match action with
  | TakeAction.TwoTangerines => 
      { tangerines := state.tangerines - 2, apples := state.apples + 1 }
  | TakeAction.TangerineAndApple => state
  | TakeAction.TwoApples => 
      { tangerines := state.tangerines, apples := state.apples - 1 }

/-- Theorem stating that the number of tangerines remains odd throughout the process -/
theorem tangerines_remain_odd (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) :
    let final_state := actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }
    Odd final_state.tangerines ∧ final_state.tangerines > 0 := by
  sorry

/-- Theorem stating that the last fruit in the vase is a tangerine -/
theorem last_fruit_is_tangerine (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) 
    (h_one_left : (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines + 
                  (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).apples = 1) :
    (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_remain_odd_last_fruit_is_tangerine_l3395_339542


namespace NUMINAMATH_CALUDE_unique_angle_satisfying_conditions_l3395_339513

theorem unique_angle_satisfying_conditions :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧ 
    Real.sin x = -(1/2) ∧ Real.cos x = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_satisfying_conditions_l3395_339513


namespace NUMINAMATH_CALUDE_cubic_root_conditions_l3395_339549

theorem cubic_root_conditions (a b c d : ℝ) (ha : a ≠ 0) 
  (h_roots : ∀ z : ℂ, a * z^3 + b * z^2 + c * z + d = 0 → z.re < 0) :
  ab > 0 ∧ bc - ad > 0 ∧ ad > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_conditions_l3395_339549


namespace NUMINAMATH_CALUDE_probability_all_girls_l3395_339588

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5
def chosen_members : ℕ := 3

theorem probability_all_girls :
  (Nat.choose num_girls chosen_members : ℚ) / (Nat.choose total_members chosen_members) = 1 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_girls_l3395_339588


namespace NUMINAMATH_CALUDE_line_through_point_with_given_segment_length_l3395_339598

-- Define the angle BAC
def Angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define a point on the angle bisector
def OnAngleBisector (D A B C : ℝ × ℝ) : Prop := sorry

-- Define a line passing through two points
def Line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the length of a segment
def SegmentLength (P Q : ℝ × ℝ) : ℝ := sorry

-- Define a point being on a line
def OnLine (P : ℝ × ℝ) (L : Set (ℝ × ℝ)) : Prop := sorry

theorem line_through_point_with_given_segment_length 
  (A B C D : ℝ × ℝ) (l : ℝ) 
  (h1 : Angle A B C) 
  (h2 : OnAngleBisector D A B C) 
  (h3 : l > 0) : 
  ∃ (E F : ℝ × ℝ), 
    OnLine E (Line A B) ∧ 
    OnLine F (Line A C) ∧ 
    OnLine D (Line E F) ∧ 
    SegmentLength E F = l := 
sorry

end NUMINAMATH_CALUDE_line_through_point_with_given_segment_length_l3395_339598


namespace NUMINAMATH_CALUDE_integral_equals_two_plus_half_pi_l3395_339561

open Set
open MeasureTheory
open Interval

theorem integral_equals_two_plus_half_pi :
  ∫ x in (Icc (-1) 1), (1 + x + Real.sqrt (1 - x^2)) = 2 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_two_plus_half_pi_l3395_339561


namespace NUMINAMATH_CALUDE_paiges_flowers_l3395_339581

theorem paiges_flowers (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) :
  flowers_per_bouquet = 7 →
  wilted_flowers = 18 →
  remaining_bouquets = 5 →
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 53 :=
by sorry

end NUMINAMATH_CALUDE_paiges_flowers_l3395_339581


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l3395_339538

/-- The number of Nintendo games Kelly needs to give away to have 12 left -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem :
  let initial_nintendo_games : ℕ := 20
  let desired_nintendo_games : ℕ := 12
  games_to_give_away initial_nintendo_games desired_nintendo_games = 8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l3395_339538


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l3395_339577

/-- Given a drawer with initial pencils and some taken out, calculate the remaining pencils -/
def remaining_pencils (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: If there were 9 pencils initially and 4 were taken out, 5 pencils remain -/
theorem pencils_in_drawer : remaining_pencils 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l3395_339577


namespace NUMINAMATH_CALUDE_line_equation_proof_l3395_339564

/-- Given two lines in the form y = mx + b, they are parallel if and only if they have the same slope m -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line y = mx + b if and only if y = mx + b -/
def point_on_line (x y m b : ℝ) : Prop := y = m * x + b

theorem line_equation_proof (x y : ℝ) : 
  parallel_lines (3/2) 3 (3/2) (-11/2) ∧ 
  point_on_line 3 (-1) (3/2) (-11/2) ∧
  3 * x - 2 * y - 11 = 0 ↔ y = (3/2) * x - 11/2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3395_339564


namespace NUMINAMATH_CALUDE_perfect_square_equation_l3395_339552

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l3395_339552


namespace NUMINAMATH_CALUDE_average_of_combined_results_l3395_339548

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l3395_339548


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l3395_339506

theorem fraction_is_positive_integer (p : ℕ+) :
  (∃ k : ℕ+, (5 * p + 15 : ℚ) / (3 * p - 9 : ℚ) = k) ↔ 4 ≤ p ∧ p ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l3395_339506


namespace NUMINAMATH_CALUDE_basketballs_with_holes_l3395_339525

/-- Given the number of soccer balls and basketballs, the number of soccer balls with holes,
    and the total number of balls without holes, calculate the number of basketballs with holes. -/
theorem basketballs_with_holes
  (total_soccer : ℕ)
  (total_basketball : ℕ)
  (soccer_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : total_basketball = 15)
  (h3 : soccer_with_holes = 30)
  (h4 : total_without_holes = 18) :
  total_basketball - (total_without_holes - (total_soccer - soccer_with_holes)) = 7 := by
  sorry


end NUMINAMATH_CALUDE_basketballs_with_holes_l3395_339525


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3395_339521

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := 5 * π / 3
  let φ : ℝ := π / 2
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2, -2 * Real.sqrt 3, 0) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3395_339521


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3395_339575

/-- Given a square with one side on the line y = 7 and endpoints on the parabola y = x^2 + 4x + 3,
    prove that its area is 32. -/
theorem square_area_on_parabola : 
  ∀ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) →
  (x₂^2 + 4*x₂ + 3 = 7) →
  x₁ ≠ x₂ →
  (x₂ - x₁)^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3395_339575


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3395_339567

theorem right_triangle_sides (p Δ : ℝ) (hp : p > 0) (hΔ : Δ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    a * b = 2 * Δ ∧
    c^2 = a^2 + b^2 ∧
    a = (p - (p^2 - 4*Δ)/(2*p) + ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 ∧
    b = (p - (p^2 - 4*Δ)/(2*p) - ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3395_339567


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3395_339519

def cost_price : ℕ := 50
def profit_rate : ℕ := 100

theorem selling_price_calculation (cost_price : ℕ) (profit_rate : ℕ) :
  cost_price = 50 → profit_rate = 100 → cost_price + (profit_rate * cost_price) / 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3395_339519


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l3395_339516

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (books : ℕ) (initial_figures : ℕ) (added_figures : ℕ) : ℕ :=
  (initial_figures + added_figures) - books

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_theorem :
  shelf_difference 3 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l3395_339516


namespace NUMINAMATH_CALUDE_min_attempts_for_two_unknown_digits_l3395_339529

/-- Represents a phone number with known and unknown digits -/
structure PhoneNumber :=
  (known_digits : Nat)
  (unknown_digits : Nat)
  (total_digits : Nat)
  (h_total : total_digits = known_digits + unknown_digits)

/-- The number of possible combinations for the unknown digits -/
def possible_combinations (pn : PhoneNumber) : Nat :=
  10 ^ pn.unknown_digits

/-- The minimum number of attempts required to guarantee dialing the correct number -/
def min_attempts (pn : PhoneNumber) : Nat :=
  possible_combinations pn

theorem min_attempts_for_two_unknown_digits 
  (pn : PhoneNumber) 
  (h_seven_digits : pn.total_digits = 7) 
  (h_five_known : pn.known_digits = 5) 
  (h_two_unknown : pn.unknown_digits = 2) : 
  min_attempts pn = 100 := by
  sorry

#check min_attempts_for_two_unknown_digits

end NUMINAMATH_CALUDE_min_attempts_for_two_unknown_digits_l3395_339529


namespace NUMINAMATH_CALUDE_circle_tangency_count_l3395_339574

theorem circle_tangency_count : ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 120 ∧ 120 % r = 0) ∧ 
  (∀ r < 120, 120 % r = 0 → r ∈ S) ∧ 
  Finset.card S = 15 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_count_l3395_339574


namespace NUMINAMATH_CALUDE_tuna_sales_difference_l3395_339571

/-- Calculates the difference in daily revenue between peak and low seasons for tuna fish sales. -/
theorem tuna_sales_difference (peak_rate : ℕ) (low_rate : ℕ) (price : ℕ) (hours : ℕ) : 
  peak_rate = 6 → low_rate = 4 → price = 60 → hours = 15 →
  (peak_rate * price * hours) - (low_rate * price * hours) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_tuna_sales_difference_l3395_339571


namespace NUMINAMATH_CALUDE_prom_services_cost_l3395_339530

/-- Calculate the total cost of prom services for Keesha --/
theorem prom_services_cost : 
  let hair_cost : ℚ := 50
  let hair_discount : ℚ := 0.1
  let manicure_cost : ℚ := 30
  let pedicure_cost : ℚ := 35
  let pedicure_discount : ℚ := 0.5
  let makeup_cost : ℚ := 40
  let makeup_tax : ℚ := 0.05
  let tip_percentage : ℚ := 0.2

  let hair_total := (hair_cost * (1 - hair_discount)) * (1 + tip_percentage)
  let nails_total := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_percentage)
  let makeup_total := (makeup_cost * (1 + makeup_tax)) * (1 + tip_percentage)

  hair_total + nails_total + makeup_total = 161.4 := by
    sorry

end NUMINAMATH_CALUDE_prom_services_cost_l3395_339530


namespace NUMINAMATH_CALUDE_waiter_customers_l3395_339562

/-- Represents the number of customers a waiter had at lunch -/
def lunch_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : ℕ :=
  non_tipping + (total_tips / tip_amount)

/-- Theorem stating the number of customers the waiter had at lunch -/
theorem waiter_customers :
  lunch_customers 4 9 27 = 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3395_339562


namespace NUMINAMATH_CALUDE_honor_distribution_proof_l3395_339583

/-- The number of ways to distribute honors among people -/
def distribute_honors (num_honors num_people : ℕ) (incompatible_pair : Bool) : ℕ :=
  sorry

/-- The number of ways to distribute honors in the specific problem -/
def problem_distribution : ℕ :=
  distribute_honors 5 3 true

theorem honor_distribution_proof :
  problem_distribution = 114 := by sorry

end NUMINAMATH_CALUDE_honor_distribution_proof_l3395_339583


namespace NUMINAMATH_CALUDE_janet_initial_clips_l3395_339579

/-- The number of paper clips Janet had in the morning -/
def initial_clips : ℕ := sorry

/-- The number of paper clips Janet used during the day -/
def used_clips : ℕ := 59

/-- The number of paper clips Janet had left at the end of the day -/
def remaining_clips : ℕ := 26

/-- Theorem: Janet had 85 paper clips in the morning -/
theorem janet_initial_clips : initial_clips = 85 := by sorry

end NUMINAMATH_CALUDE_janet_initial_clips_l3395_339579


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3395_339559

theorem gcd_lcm_product (a b : ℤ) : Nat.gcd a.natAbs b.natAbs * Nat.lcm a.natAbs b.natAbs = a.natAbs * b.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3395_339559


namespace NUMINAMATH_CALUDE_quadrilateral_impossibility_l3395_339522

theorem quadrilateral_impossibility : ¬ ∃ (a b c d : ℝ),
  (2 * a^2 - 18 * a + 36 = 0 ∨ a^2 - 20 * a + 75 = 0) ∧
  (2 * b^2 - 18 * b + 36 = 0 ∨ b^2 - 20 * b + 75 = 0) ∧
  (2 * c^2 - 18 * c + 36 = 0 ∨ c^2 - 20 * c + 75 = 0) ∧
  (2 * d^2 - 18 * d + 36 = 0 ∨ d^2 - 20 * d + 75 = 0) ∧
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a) ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_impossibility_l3395_339522


namespace NUMINAMATH_CALUDE_sequence_value_l3395_339591

/-- Given a sequence {aₙ} satisfying a₁ = 1 and aₙ - aₙ₋₁ = 2ⁿ⁻¹ for n ≥ 2, prove that a₈ = 255 -/
theorem sequence_value (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n-1) = 2^(n-1)) : 
  a 8 = 255 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l3395_339591


namespace NUMINAMATH_CALUDE_two_digit_decimal_bounds_l3395_339518

-- Define a two-digit decimal number accurate to the tenth place
def TwoDigitDecimal (x : ℝ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ ∃ (n : ℤ), x = n / 10

-- Define the approximation to the tenth place
def ApproximateToTenth (x y : ℝ) : Prop :=
  ∃ (n : ℤ), y = n / 10 ∧ |x - y| < 0.05

-- Theorem statement
theorem two_digit_decimal_bounds :
  ∀ x : ℝ,
  TwoDigitDecimal x →
  ApproximateToTenth x 15.6 →
  x ≤ 15.64 ∧ x ≥ 15.55 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_decimal_bounds_l3395_339518


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3395_339541

theorem square_sum_reciprocal (m : ℝ) (hm : m > 0) (h : m - 1/m = 3) : 
  m^2 + 1/m^2 = 11 := by
sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3395_339541


namespace NUMINAMATH_CALUDE_ray_remaining_nickels_l3395_339566

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def initial_amount : ℕ := 285

/-- Amount given to Peter in cents -/
def peter_amount : ℕ := 55

/-- Amount given to Paula in cents -/
def paula_amount : ℕ := 45

/-- Calculates the number of nickels from a given amount of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / nickel_value

theorem ray_remaining_nickels :
  let initial_nickels := cents_to_nickels initial_amount
  let peter_nickels := cents_to_nickels peter_amount
  let randi_nickels := cents_to_nickels (3 * peter_amount)
  let paula_nickels := cents_to_nickels paula_amount
  initial_nickels - (peter_nickels + randi_nickels + paula_nickels) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ray_remaining_nickels_l3395_339566


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3395_339523

def sequence_a (n : ℕ) : ℕ :=
  2 * n

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_b (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def sum_T (n : ℕ) : ℚ :=
  n / (n + 1)

theorem sequence_sum_theorem (n : ℕ) :
  sequence_a 2 = 4 ∧
  (∀ k : ℕ, sequence_a (k + 1) = sequence_a k + 2) ∧
  (∀ k : ℕ, sum_S k = k * k) ∧
  (∀ k : ℕ, sequence_b k = 1 / sum_S k) →
  sum_T n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3395_339523


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3395_339592

theorem magnitude_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + 2*i) * i
  Complex.abs z = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3395_339592


namespace NUMINAMATH_CALUDE_richard_patrick_diff_l3395_339568

/-- Bowling game results -/
def bowling_game (patrick_round1 richard_round1_diff : ℕ) : ℕ × ℕ :=
  let richard_round1 := patrick_round1 + richard_round1_diff
  let patrick_round2 := 2 * richard_round1
  let richard_round2 := patrick_round2 - 3
  let patrick_total := patrick_round1 + patrick_round2
  let richard_total := richard_round1 + richard_round2
  (patrick_total, richard_total)

/-- Theorem stating the difference in total pins knocked down -/
theorem richard_patrick_diff (patrick_round1 : ℕ) : 
  (bowling_game patrick_round1 15).2 - (bowling_game patrick_round1 15).1 = 12 :=
by sorry

end NUMINAMATH_CALUDE_richard_patrick_diff_l3395_339568


namespace NUMINAMATH_CALUDE_isabel_toy_cost_l3395_339512

theorem isabel_toy_cost (total_money : ℕ) (num_toys : ℕ) (cost_per_toy : ℕ) 
  (h1 : total_money = 14) 
  (h2 : num_toys = 7) 
  (h3 : total_money = num_toys * cost_per_toy) : 
  cost_per_toy = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_toy_cost_l3395_339512


namespace NUMINAMATH_CALUDE_parabola_equation_for_given_focus_and_directrix_l3395_339560

/-- A parabola is defined by a focus point and a directrix line parallel to the x-axis. -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The equation of a parabola given its focus and directrix. -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  λ x y => x^2 = 4 * (p.focus.2 - p.directrix) * (y - (p.focus.2 + p.directrix) / 2)

theorem parabola_equation_for_given_focus_and_directrix :
  let p : Parabola := { focus := (0, 4), directrix := -4 }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 16 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_for_given_focus_and_directrix_l3395_339560


namespace NUMINAMATH_CALUDE_jonah_profit_l3395_339587

def pineapples : ℕ := 60
def base_price : ℚ := 2
def discount_rate : ℚ := 20 / 100
def rings_per_pineapple : ℕ := 12
def single_ring_price : ℚ := 4
def bundle_size : ℕ := 6
def bundle_price : ℚ := 20
def bundles_sold : ℕ := 35
def single_rings_sold : ℕ := 150

def discounted_price : ℚ := base_price * (1 - discount_rate)
def total_cost : ℚ := pineapples * discounted_price
def bundle_revenue : ℚ := bundles_sold * bundle_price
def single_ring_revenue : ℚ := single_rings_sold * single_ring_price
def total_revenue : ℚ := bundle_revenue + single_ring_revenue
def profit : ℚ := total_revenue - total_cost

theorem jonah_profit : profit = 1204 := by
  sorry

end NUMINAMATH_CALUDE_jonah_profit_l3395_339587


namespace NUMINAMATH_CALUDE_survey_income_problem_l3395_339594

/-- Proves that given the conditions from the survey, the average income of the other 40 customers is $42,500 -/
theorem survey_income_problem (total_customers : ℕ) (wealthy_customers : ℕ) 
  (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  let other_customers := total_customers - wealthy_customers
  let total_income := total_avg_income * total_customers
  let wealthy_income := wealthy_avg_income * wealthy_customers
  let other_income := total_income - wealthy_income
  other_income / other_customers = 42500 := by
sorry

end NUMINAMATH_CALUDE_survey_income_problem_l3395_339594


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_twelve_l3395_339572

theorem negative_integer_square_plus_self_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N % 3 = 0 → N = -3 := by sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_twelve_l3395_339572


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3395_339596

theorem fraction_zero_implies_x_one (x : ℝ) : (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3395_339596


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3395_339539

def bag_total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_two_red_balls : 
  (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose bag_total_balls drawn_balls) = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3395_339539


namespace NUMINAMATH_CALUDE_carla_initial_marbles_l3395_339556

/-- The number of marbles Carla bought -/
def marbles_bought : ℝ := 489.0

/-- The total number of marbles Carla has now -/
def total_marbles : ℝ := 2778.0

/-- The number of marbles Carla started with -/
def initial_marbles : ℝ := total_marbles - marbles_bought

theorem carla_initial_marbles : initial_marbles = 2289.0 := by
  sorry

end NUMINAMATH_CALUDE_carla_initial_marbles_l3395_339556


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3395_339540

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 → 
  (4 * x + 5 * y = c) → (8 * y - 10 * x = d) → 
  c / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3395_339540


namespace NUMINAMATH_CALUDE_set_equality_l3395_339593

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

-- Theorem statement
theorem set_equality : N = M ∪ P := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3395_339593


namespace NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l3395_339502

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (∀ m : ℕ, m > 0 → (p^2 * q^3 * r^4) ∣ m → m = m^3 → m ≥ (p^2 * q^2 * r^2)^3) ∧
  (p^2 * q^3 * r^4) ∣ (p^2 * q^2 * r^2)^3 ∧
  ((p^2 * q^2 * r^2)^3)^(1/3) = p^2 * q^2 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l3395_339502


namespace NUMINAMATH_CALUDE_quadratic_equations_roots_l3395_339555

theorem quadratic_equations_roots (a₁ a₂ a₃ : ℝ) 
  (h_positive : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
  (h_geometric : ∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r)
  (h_roots_1 : a₁^2 ≥ 4)
  (h_roots_2 : a₂^2 < 8) :
  a₃^2 < 16 := by
  sorry

#check quadratic_equations_roots

end NUMINAMATH_CALUDE_quadratic_equations_roots_l3395_339555


namespace NUMINAMATH_CALUDE_evas_numbers_l3395_339527

theorem evas_numbers (a b : ℕ) (h1 : a > b) 
  (h2 : 10 ≤ a + b) (h3 : a + b < 100)
  (h4 : 10 ≤ a - b) (h5 : a - b < 100)
  (h6 : (a + b) * (a - b) = 645) : 
  a = 29 ∧ b = 14 := by
sorry

end NUMINAMATH_CALUDE_evas_numbers_l3395_339527


namespace NUMINAMATH_CALUDE_distance_to_place_l3395_339546

/-- Proves that the distance to a place is 72 km given the specified conditions -/
theorem distance_to_place (still_water_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  still_water_speed = 10 →
  current_speed = 2 →
  total_time = 15 →
  ∃ (distance : ℝ), distance = 72 ∧
    distance / (still_water_speed - current_speed) +
    distance / (still_water_speed + current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l3395_339546


namespace NUMINAMATH_CALUDE_multiple_of_nine_three_l3395_339515

theorem multiple_of_nine_three (S : ℤ) : 
  (∀ x : ℤ, 9 ∣ x → 3 ∣ x) →  -- All multiples of 9 are multiples of 3
  (Odd S) →                   -- S is an odd number
  (9 ∣ S) →                   -- S is a multiple of 9
  (3 ∣ S) :=                  -- S is a multiple of 3
by sorry

end NUMINAMATH_CALUDE_multiple_of_nine_three_l3395_339515


namespace NUMINAMATH_CALUDE_binomial_odd_even_difference_squares_l3395_339589

variable (x a : ℝ) (n : ℕ)

def A (x a : ℝ) (n : ℕ) : ℝ := sorry
def B (x a : ℝ) (n : ℕ) : ℝ := sorry

/-- For the binomial expansion (x+a)^n, where A is the sum of odd-position terms
    and B is the sum of even-position terms, A^2 - B^2 = (x^2 - a^2)^n -/
theorem binomial_odd_even_difference_squares :
  (A x a n)^2 - (B x a n)^2 = (x^2 - a^2)^n := by sorry

end NUMINAMATH_CALUDE_binomial_odd_even_difference_squares_l3395_339589


namespace NUMINAMATH_CALUDE_gcd_5factorial_8factorial_div_3factorial_l3395_339537

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5factorial_8factorial_div_3factorial : 
  Nat.gcd (factorial 5) ((factorial 8) / (factorial 3)) = 120 := by sorry

end NUMINAMATH_CALUDE_gcd_5factorial_8factorial_div_3factorial_l3395_339537


namespace NUMINAMATH_CALUDE_box_with_balls_l3395_339544

theorem box_with_balls (total : ℕ) (white blue red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end NUMINAMATH_CALUDE_box_with_balls_l3395_339544


namespace NUMINAMATH_CALUDE_sum_solution_equation_value_l3395_339550

/-- A sum solution equation is an equation of the form a/x = b where the solution for x is 1/(a+b) -/
def IsSumSolutionEquation (a b : ℚ) : Prop :=
  ∀ x, a / x = b ↔ x = 1 / (a + b)

/-- The main theorem: if n/x = 3-n is a sum solution equation, then n = 3/4 -/
theorem sum_solution_equation_value (n : ℚ) :
  IsSumSolutionEquation n (3 - n) → n = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_solution_equation_value_l3395_339550


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3395_339545

/-- A point in the xy-plane is represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of being in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- The point (-1,4) -/
def point : Point := ⟨-1, 4⟩

/-- Theorem: The point (-1,4) is in the second quadrant -/
theorem point_in_second_quadrant : isInSecondQuadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3395_339545


namespace NUMINAMATH_CALUDE_polynomial_product_identity_l3395_339509

theorem polynomial_product_identity (x z : ℝ) :
  (3 * x^4 - 4 * z^3) * (9 * x^8 + 12 * x^4 * z^3 + 16 * z^6) = 27 * x^12 - 64 * z^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_identity_l3395_339509


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l3395_339551

-- Define the number of each vegetable Conor can chop in a day
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8

-- Define the number of days Conor works per week
def work_days_per_week : ℕ := 4

-- Theorem to prove
theorem conor_weekly_vegetables :
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := by
  sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l3395_339551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3395_339584

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => a₁ + n * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- Theorem statement
theorem arithmetic_sequence_sum_6 (a₁ d : ℚ) :
  a₁ = 1/2 →
  sum_arithmetic_sequence a₁ d 4 = 20 →
  sum_arithmetic_sequence a₁ d 6 = 48 := by
  sorry

-- The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3395_339584


namespace NUMINAMATH_CALUDE_wheels_equation_l3395_339557

theorem wheels_equation (x y : ℕ) : 2 * x + 4 * y = 66 → y = (33 - x) / 2 :=
by sorry

end NUMINAMATH_CALUDE_wheels_equation_l3395_339557


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3395_339500

theorem equilateral_triangle_side_length 
  (circumference : ℝ) 
  (h1 : circumference = 4 * 21) 
  (h2 : circumference > 0) : 
  ∃ (side_length : ℝ), side_length = 28 ∧ 3 * side_length = circumference :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3395_339500


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3395_339504

theorem quadratic_root_implies_a (a : ℝ) : (2^2 - 2 + a = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3395_339504


namespace NUMINAMATH_CALUDE_expression_evaluation_l3395_339528

theorem expression_evaluation :
  let a : ℤ := -2
  let b : ℤ := 4
  (-(-3*a)^2 + 6*a*b) - (a^2 + 3*(a - 2*a*b)) = 14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3395_339528


namespace NUMINAMATH_CALUDE_student_A_more_stable_l3395_339503

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : ℝ

/-- Defines the concept of score stability based on variance -/
def moreStable (s1 s2 : Student) : Prop :=
  s1.variance < s2.variance

/-- Theorem stating that student A has more stable scores than student B -/
theorem student_A_more_stable :
  let studentA : Student := ⟨"A", 3.6⟩
  let studentB : Student := ⟨"B", 4.4⟩
  moreStable studentA studentB := by
  sorry

end NUMINAMATH_CALUDE_student_A_more_stable_l3395_339503


namespace NUMINAMATH_CALUDE_sqrt_five_squared_l3395_339526

theorem sqrt_five_squared : (Real.sqrt 5)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_l3395_339526


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3395_339534

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 1) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g(x) = 4^x - 3^x satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ g : ℝ → ℝ, FunctionalEquation g ∧ (∀ x : ℝ, g x = 4^x - 3^x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3395_339534


namespace NUMINAMATH_CALUDE_second_walking_speed_l3395_339536

/-- Proves that the second walking speed is 6 km/h given the problem conditions -/
theorem second_walking_speed (distance : ℝ) (speed1 : ℝ) (miss_time : ℝ) (early_time : ℝ) (v : ℝ) : 
  distance = 13.5 ∧ 
  speed1 = 5 ∧ 
  miss_time = 12 / 60 ∧ 
  early_time = 15 / 60 ∧ 
  distance / speed1 - miss_time = distance / v + early_time → 
  v = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_walking_speed_l3395_339536


namespace NUMINAMATH_CALUDE_leaf_collection_time_l3395_339565

/-- Represents the leaf collection problem --/
structure LeafCollection where
  totalLeaves : ℕ
  collectionRate : ℕ
  scatterRate : ℕ
  cycleTime : ℕ

/-- Calculates the time needed to collect all leaves --/
def collectionTime (lc : LeafCollection) : ℚ :=
  let netIncrease := lc.collectionRate - lc.scatterRate
  let cycles := (lc.totalLeaves - lc.scatterRate) / netIncrease
  let totalSeconds := (cycles + 1) * lc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the collection time for the given problem is 21.5 minutes --/
theorem leaf_collection_time :
  let lc : LeafCollection := {
    totalLeaves := 45,
    collectionRate := 4,
    scatterRate := 3,
    cycleTime := 30
  }
  collectionTime lc = 21.5 := by sorry

end NUMINAMATH_CALUDE_leaf_collection_time_l3395_339565


namespace NUMINAMATH_CALUDE_max_sum_divisible_into_two_parts_l3395_339514

theorem max_sum_divisible_into_two_parts (S : ℕ) : 
  (∃ (nums : List ℕ), 
    (∀ n ∈ nums, 0 < n ∧ n ≤ 10) ∧ 
    (nums.sum = S) ∧ 
    (∀ (partition : List ℕ × List ℕ), 
      partition.1 ∪ partition.2 = nums → 
      partition.1.sum ≤ 70 ∧ partition.2.sum ≤ 70)) →
  S ≤ 133 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_divisible_into_two_parts_l3395_339514


namespace NUMINAMATH_CALUDE_percent_of_y_l3395_339533

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3395_339533


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3395_339576

theorem point_in_first_quadrant (a : ℕ+) :
  (4 > 0 ∧ 2 - a.val > 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3395_339576


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3395_339543

theorem polynomial_remainder (a b : ℤ) : 
  (∀ x : ℤ, ∃ q : ℤ, x^3 - 2*x^2 + a*x + b = (x-1)*(x-2)*q + (2*x + 1)) → 
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3395_339543


namespace NUMINAMATH_CALUDE_A_B_symmetric_x_l3395_339511

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Define symmetry with respect to x-axis
def symmetric_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem A_B_symmetric_x : symmetric_x A B := by
  sorry

end NUMINAMATH_CALUDE_A_B_symmetric_x_l3395_339511


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_third_l3395_339563

theorem sin_alpha_plus_pi_third (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 7 * Real.sin α = 2 * Real.cos (2 * α)) : 
  Real.sin (α + π / 3) = (1 + 3 * Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_third_l3395_339563


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3395_339531

theorem least_positive_integer_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 6 = 5 ∧ m % 7 = 2 → m ≥ n) ∧
  n = 83 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3395_339531
