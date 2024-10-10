import Mathlib

namespace percent_value_in_quarters_l564_56475

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let quarter_value : ℕ := num_quarters * value_quarter
  (quarter_value : ℚ) / (total_value : ℚ) * 100 = 62.5
  := by sorry

end percent_value_in_quarters_l564_56475


namespace derivative_log2_l564_56439

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
by sorry

end derivative_log2_l564_56439


namespace factorization_sum_l564_56449

theorem factorization_sum (A B C D E F G H J K : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (A * x + B * y) * (C * x^2 + D * x * y + E * y^2) * 
                                     (F * x + G * y) * (H * x^2 + J * x * y + K * y^2)) →
  A + B + C + D + E + F + G + H + J + K = 32 := by
  sorry

end factorization_sum_l564_56449


namespace total_committees_is_160_l564_56426

/-- Represents the structure of a committee -/
inductive CommitteeStructure
  | FiveSenators
  | FourSenatorsAndFourAides
  | TwoSenatorsAndTwelveAides

/-- The number of senators -/
def numSenators : ℕ := 100

/-- The number of aides each senator has -/
def aidesPerSenator : ℕ := 4

/-- The number of committees each senator serves on -/
def committeesPerSenator : ℕ := 5

/-- The number of committees each aide serves on -/
def committeesPerAide : ℕ := 3

/-- The total number of committees -/
def totalCommittees : ℕ := 160

/-- Theorem stating that the total number of committees is 160 -/
theorem total_committees_is_160 :
  totalCommittees = 160 :=
by sorry

end total_committees_is_160_l564_56426


namespace solve_for_k_l564_56412

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  (2 * x + y = 4 * k) ∧ (x - y = k)

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  x + 2 * y = 12

-- Theorem statement
theorem solve_for_k :
  ∀ x y k : ℝ, system x y k → additional_eq x y → k = 4 :=
by
  sorry

end solve_for_k_l564_56412


namespace interest_rate_calculation_l564_56479

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P / 5 := by
  sorry

end interest_rate_calculation_l564_56479


namespace root_sum_theorem_l564_56440

-- Define the polynomial
def polynomial (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_sum_theorem (a b c d : ℝ) (h_a : a ≠ 0) :
  polynomial a b c d 4 = 0 ∧
  polynomial a b c d (-1) = 0 ∧
  polynomial a b c d (-3) = 0 →
  (b + c) / a = -1441 / 37 := by
  sorry

end root_sum_theorem_l564_56440


namespace smallest_number_l564_56423

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The number 85 in base 9 --/
def num1 : Nat := to_decimal [5, 8] 9

/-- The number 1000 in base 4 --/
def num2 : Nat := to_decimal [0, 0, 0, 1] 4

/-- The number 111111 in base 2 --/
def num3 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number : num3 < num2 ∧ num3 < num1 := by
  sorry

end smallest_number_l564_56423


namespace cos_sixty_degrees_l564_56476

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l564_56476


namespace baking_powder_difference_l564_56415

theorem baking_powder_difference (yesterday_amount today_amount : ℝ) 
  (h1 : yesterday_amount = 0.4)
  (h2 : today_amount = 0.3) : 
  yesterday_amount - today_amount = 0.1 := by
sorry

end baking_powder_difference_l564_56415


namespace proportion_equality_l564_56437

theorem proportion_equality (m n : ℝ) (h1 : 6 * m = 7 * n) (h2 : n ≠ 0) :
  m / 7 = n / 6 := by
  sorry

end proportion_equality_l564_56437


namespace imaginary_part_of_one_minus_two_i_squared_l564_56404

theorem imaginary_part_of_one_minus_two_i_squared (i : ℂ) : 
  i * i = -1 → Complex.im ((1 - 2*i)^2) = -4 := by
  sorry

end imaginary_part_of_one_minus_two_i_squared_l564_56404


namespace exists_valid_superchess_configuration_l564_56473

/-- Represents a chess piece in the game of superchess -/
structure Piece where
  id : Fin 20

/-- Represents a position on the superchess board -/
structure Position where
  x : Fin 100
  y : Fin 100

/-- Represents the superchess board -/
def Board := Fin 100 → Fin 100 → Option Piece

/-- Predicate to check if a piece attacks a position -/
def attacks (p : Piece) (pos : Position) (board : Board) : Prop :=
  ∃ (attacked : Finset Position), attacked.card ≤ 20 ∧ pos ∈ attacked

/-- Predicate to check if a board configuration is valid (no piece attacks another) -/
def valid_board (board : Board) : Prop :=
  ∀ (p₁ p₂ : Piece) (pos₁ pos₂ : Position),
    board pos₁.x pos₁.y = some p₁ →
    board pos₂.x pos₂.y = some p₂ →
    p₁ ≠ p₂ →
    ¬(attacks p₁ pos₂ board ∨ attacks p₂ pos₁ board)

/-- Theorem stating that there exists a valid board configuration -/
theorem exists_valid_superchess_configuration :
  ∃ (board : Board), (∀ p : Piece, ∃ pos : Position, board pos.x pos.y = some p) ∧ valid_board board :=
sorry

end exists_valid_superchess_configuration_l564_56473


namespace base_conversion_subtraction_l564_56428

-- Define a function to convert a number from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Define a function to convert a number from base 9 to base 10
def base9ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem base_conversion_subtraction :
  base8ToBase10 76432 - base9ToBase10 2541 = 30126 := by
  sorry

end base_conversion_subtraction_l564_56428


namespace president_secretary_selection_l564_56489

/-- The number of ways to select one president and one secretary from five different people. -/
def select_president_and_secretary (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select one president and one secretary from five different people is 20. -/
theorem president_secretary_selection :
  select_president_and_secretary 5 = 20 := by
  sorry

end president_secretary_selection_l564_56489


namespace square_side_length_l564_56414

/-- A square with four identical isosceles triangles on its sides -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  s : ℝ
  /-- Area of one isosceles triangle -/
  triangle_area : ℝ
  /-- The total area of the isosceles triangles equals the area of the remaining region -/
  area_equality : 4 * triangle_area = s^2 - 4 * triangle_area
  /-- The distance between the apexes of two opposite isosceles triangles is 12 -/
  apex_distance : s + 2 * (triangle_area / s) = 12

/-- Theorem: The side length of the square is 24 -/
theorem square_side_length (sq : SquareWithTriangles) : sq.s = 24 :=
sorry

end square_side_length_l564_56414


namespace tammy_orange_picking_l564_56481

/-- Proves that given the conditions of Tammy's orange selling business, 
    she picks 12 oranges from each tree each day. -/
theorem tammy_orange_picking :
  let num_trees : ℕ := 10
  let oranges_per_pack : ℕ := 6
  let price_per_pack : ℕ := 2
  let total_earnings : ℕ := 840
  let num_weeks : ℕ := 3
  let days_per_week : ℕ := 7

  (num_trees > 0) →
  (oranges_per_pack > 0) →
  (price_per_pack > 0) →
  (total_earnings > 0) →
  (num_weeks > 0) →
  (days_per_week > 0) →

  (total_earnings / price_per_pack * oranges_per_pack) / (num_weeks * days_per_week) / num_trees = 12 :=
by
  sorry


end tammy_orange_picking_l564_56481


namespace youngest_boy_age_l564_56429

/-- Given three boys whose ages are in proportion 2 : 6 : 8 and whose average age is 120 years,
    the age of the youngest boy is 45 years. -/
theorem youngest_boy_age (a b c : ℕ) : 
  a + b + c = 360 →  -- Sum of ages is 360 (3 * 120)
  3 * a = b →        -- b is 3 times a
  4 * a = c →        -- c is 4 times a
  a = 45 :=          -- The age of the youngest boy (a) is 45
by sorry

end youngest_boy_age_l564_56429


namespace grade_multiplier_is_five_l564_56443

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]
def total_reward : ℚ := 15

theorem grade_multiplier_is_five :
  let average_grade := (grades.sum : ℚ) / grades.length
  let multiplier := total_reward / average_grade
  multiplier = 5 := by sorry

end grade_multiplier_is_five_l564_56443


namespace sample_size_is_120_l564_56444

/-- Represents the sizes of three population groups -/
structure PopulationGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculates the total sample size in a stratified sampling -/
def calculateSampleSize (groups : PopulationGroups) (samplesFromGroup3 : ℕ) : ℕ :=
  samplesFromGroup3 * (groups.group1 + groups.group2 + groups.group3) / groups.group3

/-- Theorem stating that the sample size is 120 under given conditions -/
theorem sample_size_is_120 (groups : PopulationGroups) (h1 : groups.group1 = 2400) 
    (h2 : groups.group2 = 3600) (h3 : groups.group3 = 6000) (samplesFromGroup3 : ℕ) 
    (h4 : samplesFromGroup3 = 60) : 
  calculateSampleSize groups samplesFromGroup3 = 120 := by
  sorry

#eval calculateSampleSize ⟨2400, 3600, 6000⟩ 60

end sample_size_is_120_l564_56444


namespace problem_1_problem_2_problem_3_problem_4_l564_56461

-- Problem 1
theorem problem_1 : Real.sqrt 32 + 3 * Real.sqrt (1/2) - Real.sqrt 2 = (9 * Real.sqrt 2) / 2 := by sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 - 3)^2 + (Real.sqrt 11 + 3) * (Real.sqrt 11 - 3) = 16 - 6 * Real.sqrt 5 := by sorry

-- Problem 4
theorem problem_4 : (2 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt 3 - 12 * Real.sqrt (1/2) = 6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l564_56461


namespace gcd_of_100_and_250_l564_56433

theorem gcd_of_100_and_250 : Nat.gcd 100 250 = 50 := by
  sorry

end gcd_of_100_and_250_l564_56433


namespace lcm_hcf_problem_l564_56427

/-- Given two natural numbers a and b, their LCM is 2310 and a is 462, prove that their HCF is 1 -/
theorem lcm_hcf_problem (a b : ℕ) (h1 : a = 462) (h2 : Nat.lcm a b = 2310) : Nat.gcd a b = 1 := by
  sorry

end lcm_hcf_problem_l564_56427


namespace zach_needs_six_dollars_l564_56499

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 65 2 = 6 := by
  sorry

end zach_needs_six_dollars_l564_56499


namespace last_period_production_theorem_l564_56455

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : ℕ
  first_period_days : ℕ
  first_period_avg : ℕ
  monthly_avg : ℕ

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : ℚ :=
  let total_production := prod.total_days * prod.monthly_avg
  let first_period_production := prod.first_period_days * prod.first_period_avg
  let last_period_days := prod.total_days - prod.first_period_days
  (total_production - first_period_production) / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_period_production_theorem (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 63)
  (h4 : prod.monthly_avg = 58) :
  last_period_avg prod = 33 := by
  sorry

end last_period_production_theorem_l564_56455


namespace range_of_a_l564_56495

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, 3 * x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3) :=
by sorry

end range_of_a_l564_56495


namespace p_or_q_is_true_l564_56408

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by
  sorry

end p_or_q_is_true_l564_56408


namespace mariela_get_well_cards_l564_56463

theorem mariela_get_well_cards (cards_from_home : ℝ) (cards_from_country : ℕ) 
  (h1 : cards_from_home = 287.0) 
  (h2 : cards_from_country = 116) : 
  ↑cards_from_country + cards_from_home = 403 := by
  sorry

end mariela_get_well_cards_l564_56463


namespace area_ratio_in_square_l564_56460

/-- Given a unit square ABCD with points X on BC and Y on CD such that
    triangles ABX, XCY, and YDA have equal areas, the ratio of the area of
    triangle AXY to the area of triangle XCY is √5. -/
theorem area_ratio_in_square (A B C D X Y : ℝ × ℝ) : 
  let square_side_length : ℝ := 1
  let on_side (P Q R : ℝ × ℝ) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Q + t • R
  let area (P Q R : ℝ × ℝ) : ℝ := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  square_side_length = 1 →
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = 1 ∧ B.2 = 0) →
  (C.1 = 1 ∧ C.2 = 1) →
  (D.1 = 0 ∧ D.2 = 1) →
  on_side X B C →
  on_side Y C D →
  area A B X = area X C Y →
  area X C Y = area Y D A →
  area A X Y / area X C Y = Real.sqrt 5 := by
sorry

end area_ratio_in_square_l564_56460


namespace basketball_probabilities_l564_56445

def probability_A : ℝ := 0.7
def shots : ℕ := 3

theorem basketball_probabilities (a : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : (Nat.choose 3 2 : ℝ) * (1 - probability_A) * probability_A^2 + probability_A^3 - a^3 = 0.659) :
  a = 0.5 ∧ 
  (1 - probability_A)^3 * (1 - a)^3 + 
  (Nat.choose 3 1 : ℝ) * (1 - probability_A)^2 * probability_A * 
  (Nat.choose 3 1 : ℝ) * (1 - a)^2 * a = 0.07425 := by
  sorry

end basketball_probabilities_l564_56445


namespace sequence_equation_l564_56465

theorem sequence_equation (n : ℕ+) : 9 * n + (n - 1) = 10 * n - 1 := by
  sorry

#check sequence_equation

end sequence_equation_l564_56465


namespace siblings_average_age_l564_56407

/-- Given 4 siblings where the youngest is 25.75 years old and the others are 3, 6, and 7 years older,
    the average age of all siblings is 29.75 years. -/
theorem siblings_average_age :
  let youngest_age : ℝ := 25.75
  let sibling_age_differences : List ℝ := [3, 6, 7]
  let all_ages : List ℝ := youngest_age :: (sibling_age_differences.map (λ x => youngest_age + x))
  (all_ages.sum / all_ages.length : ℝ) = 29.75 := by
  sorry

end siblings_average_age_l564_56407


namespace furniture_cost_price_l564_56401

theorem furniture_cost_price (price : ℝ) (discount : ℝ) (profit : ℝ) :
  price = 132 ∧ 
  discount = 0.1 ∧ 
  profit = 0.1 ∧ 
  price * (1 - discount) = (1 + profit) * (price * (1 - discount) / (1 + profit)) →
  price * (1 - discount) / (1 + profit) = 108 :=
by sorry

end furniture_cost_price_l564_56401


namespace quadratic_equation_from_roots_l564_56491

theorem quadratic_equation_from_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 ∧ 
  a = 1 ∧ b = -5 ∧ c = 6 := by
  sorry

end quadratic_equation_from_roots_l564_56491


namespace baxter_spent_105_l564_56466

/-- The cost of peanuts per pound -/
def cost_per_pound : ℕ := 3

/-- The minimum purchase requirement in pounds -/
def minimum_purchase : ℕ := 15

/-- The amount Baxter purchased over the minimum, in pounds -/
def over_minimum : ℕ := 20

/-- Calculates the total amount Baxter spent on peanuts -/
def baxter_spent : ℕ := cost_per_pound * (minimum_purchase + over_minimum)

/-- Proves that Baxter spent $105 on peanuts -/
theorem baxter_spent_105 : baxter_spent = 105 := by
  sorry

end baxter_spent_105_l564_56466


namespace prob_all_red_is_one_third_l564_56462

/-- Represents the number of red chips in the hat -/
def num_red : ℕ := 4

/-- Represents the number of green chips in the hat -/
def num_green : ℕ := 2

/-- Represents the total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- Represents the probability of drawing all red chips before both green chips -/
def prob_all_red : ℚ := 1 / 3

/-- Theorem stating that the probability of drawing all red chips before both green chips is 1/3 -/
theorem prob_all_red_is_one_third :
  prob_all_red = 1 / 3 := by sorry

end prob_all_red_is_one_third_l564_56462


namespace sin_593_degrees_l564_56469

theorem sin_593_degrees (h : Real.sin (37 * π / 180) = 3 / 5) :
  Real.sin (593 * π / 180) = -(3 / 5) := by
  sorry

end sin_593_degrees_l564_56469


namespace eliminate_alpha_l564_56430

theorem eliminate_alpha (x y : ℝ) (α : ℝ) 
  (hx : x = Real.tan α ^ 2) 
  (hy : y = Real.sin α ^ 2) : 
  x - y = x * y := by
  sorry

end eliminate_alpha_l564_56430


namespace consecutive_product_square_append_l564_56490

theorem consecutive_product_square_append (n : ℕ) : ∃ m : ℕ, 100 * (n * (n + 1)) + 25 = m^2 := by
  sorry

end consecutive_product_square_append_l564_56490


namespace inequality_solution_l564_56454

theorem inequality_solution (a b c : ℝ) (h1 : a < b)
  (h2 : ∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -6 ∨ |x - 31| ≤ 1) :
  a + 2 * b + 3 * c = 76 := by
  sorry

end inequality_solution_l564_56454


namespace arithmetic_to_geometric_sequence_ratio_l564_56483

/-- 
Given three distinct real numbers a, b, c forming an arithmetic sequence with a < b < c,
if swapping two of these numbers results in a geometric sequence,
then (a² + c²) / b² = 20.
-/
theorem arithmetic_to_geometric_sequence_ratio (a b c : ℝ) : 
  a < b → b < c → 
  (∃ d : ℝ, c - b = b - a ∧ d = b - a) →
  (∃ (x y z : ℝ) (σ : Equiv.Perm (Fin 3)), 
    ({x, y, z} : Finset ℝ) = {a, b, c} ∧ 
    (y * y = x * z)) →
  (a * a + c * c) / (b * b) = 20 := by
sorry

end arithmetic_to_geometric_sequence_ratio_l564_56483


namespace sine_function_properties_l564_56435

theorem sine_function_properties (ω φ : ℝ) (f : ℝ → ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_f_def : ∀ x, f x = Real.sin (ω * x + φ))
  (h_period : ∀ x, f (x + π) = f x)
  (h_f_zero : f 0 = 1 / 2) :
  (ω = 2) ∧ 
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6),
    x ≤ y → f x ≤ f y) :=
by sorry

end sine_function_properties_l564_56435


namespace perpendicular_planes_l564_56474

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (is_perpendicular : Line → Plane → Prop)
variable (is_subset : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- Define m, n as different lines
variable (m n : Line)
variable (h_diff_lines : m ≠ n)

-- Define α, β as different planes
variable (α β : Plane)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_planes 
  (h1 : is_perpendicular m α) 
  (h2 : is_subset m β) : 
  planes_perpendicular α β := by sorry

end perpendicular_planes_l564_56474


namespace problem_proof_l564_56451

theorem problem_proof : -1^2023 + (Real.pi - 3.14)^0 + |-2| = 2 := by
  sorry

end problem_proof_l564_56451


namespace quadratic_equation_solutions_l564_56425

theorem quadratic_equation_solutions :
  (∀ x, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by sorry

end quadratic_equation_solutions_l564_56425


namespace candy_probability_l564_56459

/-- The probability of picking specific candies from a bag --/
theorem candy_probability : 
  let green : ℕ := 8
  let blue : ℕ := 5
  let red : ℕ := 9
  let yellow : ℕ := 10
  let pink : ℕ := 6
  let total : ℕ := green + blue + red + yellow + pink
  
  -- Probability of picking green first
  let p_green : ℚ := green / total
  
  -- Probability of picking yellow second
  let p_yellow : ℚ := yellow / (total - 1)
  
  -- Probability of picking pink third
  let p_pink : ℚ := pink / (total - 2)
  
  -- Overall probability
  let probability : ℚ := p_green * p_yellow * p_pink
  
  probability = 20 / 2109 := by
  sorry

end candy_probability_l564_56459


namespace michaels_lap_time_l564_56496

/-- Race on a circular track -/
structure RaceTrack where
  length : ℝ
  donovan_lap_time : ℝ
  michael_laps_to_pass : ℕ

/-- Given race conditions, prove Michael's lap time -/
theorem michaels_lap_time (race : RaceTrack)
  (h1 : race.length = 300)
  (h2 : race.donovan_lap_time = 45)
  (h3 : race.michael_laps_to_pass = 9) :
  ∃ t : ℝ, t = 50 ∧ t * race.michael_laps_to_pass = (race.michael_laps_to_pass + 1) * race.donovan_lap_time :=
by sorry

end michaels_lap_time_l564_56496


namespace rope_knot_reduction_l564_56480

theorem rope_knot_reduction 
  (total_length : ℝ) 
  (num_pieces : ℕ) 
  (tied_pieces : ℕ) 
  (final_length : ℝ) 
  (h1 : total_length = 72) 
  (h2 : num_pieces = 12) 
  (h3 : tied_pieces = 3) 
  (h4 : final_length = 15) : 
  (total_length / num_pieces * tied_pieces - final_length) / (tied_pieces - 1) = 1.5 := by
  sorry

end rope_knot_reduction_l564_56480


namespace remainder_17_pow_63_mod_7_l564_56488

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l564_56488


namespace one_more_tile_possible_l564_56431

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)

/-- Represents a T-shaped tile -/
structure TTile :=
  (squares_covered : ℕ)

/-- The number of squares that remain uncovered after placing T-tiles -/
def uncovered_squares (board : Checkerboard) (tiles : ℕ) (tile : TTile) : ℕ :=
  board.size ^ 2 - tiles * tile.squares_covered

/-- Theorem stating that one more T-tile can be placed on the checkerboard -/
theorem one_more_tile_possible (board : Checkerboard) (tiles : ℕ) (tile : TTile) :
  board.size = 100 →
  tiles = 800 →
  tile.squares_covered = 4 →
  uncovered_squares board tiles tile ≥ 4 :=
sorry

end one_more_tile_possible_l564_56431


namespace sum_of_cyclic_equations_l564_56411

theorem sum_of_cyclic_equations (p q r : ℕ+) 
  (eq1 : p * q + r = 47)
  (eq2 : q * r + p = 47)
  (eq3 : r * p + q = 47) :
  p + q + r = 48 := by
  sorry

end sum_of_cyclic_equations_l564_56411


namespace unique_k_for_lcm_l564_56471

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem unique_k_for_lcm : ∃! k : ℕ+, lcm (6^6) (9^9) k = 18^18 := by
  sorry

end unique_k_for_lcm_l564_56471


namespace solve_inequality_1_solve_inequality_2_l564_56482

-- Inequality 1
theorem solve_inequality_1 : 
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

-- Inequality 2
theorem solve_inequality_2 : 
  {x : ℝ | -6*x^2 - x + 2 ≤ 0} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 1/2} := by sorry

end solve_inequality_1_solve_inequality_2_l564_56482


namespace prob_different_colors_specific_l564_56418

/-- The probability of drawing two chips of different colors -/
def prob_different_colors (blue yellow red : ℕ) : ℚ :=
  let total := blue + yellow + red
  let p_blue := blue / total
  let p_yellow := yellow / total
  let p_red := red / total
  p_blue * (p_yellow + p_red) + p_yellow * (p_blue + p_red) + p_red * (p_blue + p_yellow)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific : prob_different_colors 6 4 2 = 11 / 18 := by
  sorry

#eval prob_different_colors 6 4 2

end prob_different_colors_specific_l564_56418


namespace apartment_exchange_in_two_days_l564_56484

universe u

theorem apartment_exchange_in_two_days {α : Type u} [Finite α] :
  ∀ (f : α → α), Function.Bijective f →
  ∃ (g h : α → α), Function.Involutive g ∧ Function.Involutive h ∧ f = g ∘ h :=
by sorry

end apartment_exchange_in_two_days_l564_56484


namespace max_cards_per_box_l564_56442

/-- Given a total of 94 cards and 6 cards in an unfilled box, 
    prove that the maximum number of cards a full box can hold is 22. -/
theorem max_cards_per_box (total_cards : ℕ) (cards_in_unfilled_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_in_unfilled_box = 6) :
  ∃ (max_cards_per_box : ℕ), 
    max_cards_per_box = 22 ∧ 
    max_cards_per_box > cards_in_unfilled_box ∧
    (total_cards - cards_in_unfilled_box) % max_cards_per_box = 0 ∧
    ∀ n : ℕ, n > max_cards_per_box → (total_cards - cards_in_unfilled_box) % n ≠ 0 :=
by sorry

end max_cards_per_box_l564_56442


namespace ellipse_range_theorem_l564_56424

theorem ellipse_range_theorem :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → -Real.sqrt 17 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 17 := by
  sorry

end ellipse_range_theorem_l564_56424


namespace magic_square_g_value_l564_56405

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  i : ℕ+
  row_product : a * b * c = d * e * f ∧ d * e * f = g * h * i
  col_product : a * d * g = b * e * h ∧ b * e * h = c * f * i
  diag_product : a * e * i = c * e * g

/-- The theorem stating that the only possible value for g is 3 --/
theorem magic_square_g_value (ms : MagicSquare) (h1 : ms.a = 90) (h2 : ms.i = 3) :
  ms.g = 3 :=
sorry

end magic_square_g_value_l564_56405


namespace six_different_squares_cannot_form_rectangle_l564_56416

/-- A square with a given side length -/
structure Square where
  sideLength : ℝ
  positive : sideLength > 0

/-- A collection of squares -/
def SquareCollection := List Square

/-- Predicate to check if all squares in a collection have different sizes -/
def allDifferentSizes (squares : SquareCollection) : Prop :=
  ∀ i j, i ≠ j → (squares.get i).sideLength ≠ (squares.get j).sideLength

/-- Predicate to check if squares can form a rectangle -/
def canFormRectangle (squares : SquareCollection) : Prop :=
  ∃ (width height : ℝ), width > 0 ∧ height > 0 ∧
    (squares.map (λ s => s.sideLength ^ 2)).sum = width * height

theorem six_different_squares_cannot_form_rectangle :
  ∀ (squares : SquareCollection),
    squares.length = 6 →
    allDifferentSizes squares →
    ¬ canFormRectangle squares :=
by
  sorry

end six_different_squares_cannot_form_rectangle_l564_56416


namespace repetend_of_four_seventeenths_l564_56498

/-- The decimal representation of 4/17 has a 6-digit repetend of 235294 -/
theorem repetend_of_four_seventeenths : ∃ (a b : ℕ), 
  (4 : ℚ) / 17 = (a : ℚ) / 999999 + (b : ℚ) / (999999 * 1000000) ∧ 
  a = 235294 ∧ 
  b < 999999 := by sorry

end repetend_of_four_seventeenths_l564_56498


namespace quadratic_intersection_l564_56472

/-- A quadratic function of the form y = x^2 + px + q where p + q = 2002 -/
def QuadraticFunction (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q

/-- The theorem stating that all quadratic functions satisfying the condition
    p + q = 2002 intersect at the point (1, 2003) -/
theorem quadratic_intersection (p q : ℝ) (h : p + q = 2002) :
  QuadraticFunction p q 1 = 2003 := by
  sorry

end quadratic_intersection_l564_56472


namespace unique_modular_solution_l564_56417

theorem unique_modular_solution : ∃! n : ℕ, n < 251 ∧ (250 * n) % 251 = 123 % 251 := by
  sorry

end unique_modular_solution_l564_56417


namespace gcd_8251_6105_l564_56420

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l564_56420


namespace quadratic_distinct_roots_implies_c_less_than_one_l564_56409

theorem quadratic_distinct_roots_implies_c_less_than_one :
  ∀ c : ℝ, (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) → c < 1 := by
  sorry

end quadratic_distinct_roots_implies_c_less_than_one_l564_56409


namespace basketball_team_selection_with_twins_l564_56403

def number_of_players : ℕ := 16
def number_of_starters : ℕ := 7

theorem basketball_team_selection_with_twins :
  (Nat.choose (number_of_players - 2) (number_of_starters - 2)) +
  (Nat.choose (number_of_players - 2) number_of_starters) =
  (Nat.choose 14 5) + (Nat.choose 14 7) :=
by sorry

end basketball_team_selection_with_twins_l564_56403


namespace half_radius_circle_y_l564_56446

theorem half_radius_circle_y (x y : Real) :
  (∃ (r : Real), x = π * r^2 ∧ y = π * r^2) →  -- circles x and y have the same area
  (∃ (r : Real), 18 * π = 2 * π * r) →         -- circle x has circumference 18π
  (∃ (r : Real), y = π * r^2 ∧ r / 2 = 4.5) := by
sorry

end half_radius_circle_y_l564_56446


namespace intersection_of_A_and_B_l564_56467

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l564_56467


namespace contest_winner_l564_56400

theorem contest_winner (n : ℕ) : 
  (∀ k : ℕ, k > 0 → n % 100 = 0 ∧ n % 40 = 0) → n ≥ 200 :=
sorry

end contest_winner_l564_56400


namespace sock_pairs_same_color_l564_56413

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem sock_pairs_same_color (red green yellow : ℕ) 
  (h_red : red = 5) (h_green : green = 6) (h_yellow : yellow = 4) :
  choose red 2 + choose green 2 + choose yellow 2 = 31 := by
  sorry

end sock_pairs_same_color_l564_56413


namespace fraction_under_21_l564_56406

theorem fraction_under_21 (total : ℕ) (under_21 : ℕ) (over_65 : ℚ) :
  total > 50 →
  total < 100 →
  over_65 = 5/10 →
  under_21 = 30 →
  (under_21 : ℚ) / total = 1/2 := by
  sorry

end fraction_under_21_l564_56406


namespace payment_calculation_l564_56410

theorem payment_calculation (rate : ℚ) (rooms : ℚ) : 
  rate = 13/3 → rooms = 8/5 → rate * rooms = 104/15 := by
  sorry

end payment_calculation_l564_56410


namespace adjacent_above_350_l564_56422

/-- Represents a position in the triangular grid -/
structure GridPosition where
  row : ℕ
  column : ℕ

/-- Returns the number at a given position in the triangular grid -/
def numberAt (pos : GridPosition) : ℕ := sorry

/-- Returns the position of a given number in the triangular grid -/
def positionOf (n : ℕ) : GridPosition := sorry

/-- Returns the number in the horizontally adjacent triangle in the row above -/
def adjacentAbove (n : ℕ) : ℕ := sorry

theorem adjacent_above_350 : adjacentAbove 350 = 314 := by sorry

end adjacent_above_350_l564_56422


namespace hide_and_seek_l564_56457

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek : 
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end hide_and_seek_l564_56457


namespace temperature_difference_l564_56450

theorem temperature_difference (M L N : ℝ) : 
  M = L + N →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (∃ (M_4 L_4 : ℝ), 
    M_4 = M - 5 ∧  -- Minneapolis temperature falls by 5 degrees at 4:00
    L_4 = L + 3 ∧  -- St. Louis temperature rises by 3 degrees at 4:00
    abs (M_4 - L_4) = 2) →  -- Temperatures differ by 2 degrees at 4:00
  (N = 10 ∨ N = 6) ∧ N * (16 - N) = 60 :=
by sorry

end temperature_difference_l564_56450


namespace shaded_cubes_count_l564_56494

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal_shaded : Bool

/-- Represents the shading pattern on the faces of the large cube -/
structure ShadingPattern where
  diagonal : Bool
  opposite_faces_identical : Bool

/-- Counts the number of smaller cubes with at least one shaded face -/
def count_shaded_cubes (cube : LargeCube) (pattern : ShadingPattern) : Nat :=
  sorry

/-- The main theorem stating that 32 smaller cubes are shaded -/
theorem shaded_cubes_count (cube : LargeCube) (pattern : ShadingPattern) :
  cube.size = 4 ∧ 
  cube.total_cubes = 64 ∧ 
  cube.diagonal_shaded = true ∧
  pattern.diagonal = true ∧
  pattern.opposite_faces_identical = true →
  count_shaded_cubes cube pattern = 32 :=
sorry

end shaded_cubes_count_l564_56494


namespace largest_a_less_than_l564_56402

theorem largest_a_less_than (a b : ℤ) : 
  9 < a → 
  19 < b → 
  b < 31 → 
  (a : ℚ) / (b : ℚ) ≤ 2/3 → 
  a < 21 :=
by sorry

end largest_a_less_than_l564_56402


namespace abc_product_l564_56421

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 171) (h2 : b * (c + a) = 180) (h3 : c * (a + b) = 189) :
  a * b * c = 270 := by
  sorry

end abc_product_l564_56421


namespace scale_model_height_l564_56470

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 20

/-- The actual height of the United States Capitol in feet -/
def actual_height : ℕ := 289

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).toNat

theorem scale_model_height :
  model_height = 14 := by sorry

end scale_model_height_l564_56470


namespace max_m_value_min_objective_value_l564_56436

-- Define the inequality function
def inequality (x m : ℝ) : Prop := |x - 3| + |x - m| ≥ 2 * m

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∀ x : ℝ, inequality x 1) ∧ 
  (∀ m : ℝ, m > 1 → ∃ x : ℝ, ¬(inequality x m)) :=
sorry

-- Define the constraint function
def constraint (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1

-- Define the objective function
def objective (a b c : ℝ) : ℝ := 4 * a^2 + 9 * b^2 + c^2

-- Theorem for the minimum value of the objective function
theorem min_objective_value :
  (∀ a b c : ℝ, constraint a b c → objective a b c ≥ 36/49) ∧
  (∃ a b c : ℝ, constraint a b c ∧ objective a b c = 36/49 ∧ 
    a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end max_m_value_min_objective_value_l564_56436


namespace orthogonal_lines_sweep_l564_56456

-- Define the circle S
def S (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ a^2}

-- Define the point O outside the circle
def O (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- Define a point X inside the circle
def X (c : ℝ) (a : ℝ) : ℝ × ℝ := (c, 0)

-- Define the set of points swept by lines l
def swept_points (a c : ℝ) : Set (ℝ × ℝ) :=
  {p | (c^2 - a^2) * p.1^2 - a^2 * p.2^2 ≤ a^2 * (c^2 - a^2)}

-- State the theorem
theorem orthogonal_lines_sweep (a : ℝ) (x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 ≠ a^2) :
  ∀ c, c^2 < a^2 →
    swept_points a c =
    {p | ∃ (X : ℝ × ℝ), X ∈ S a ∧ (p.1 - X.1) * (O x₀ y₀).1 + (p.2 - X.2) * (O x₀ y₀).2 = 0} :=
by sorry

end orthogonal_lines_sweep_l564_56456


namespace unoccupied_area_formula_l564_56497

/-- The area of a rectangle not occupied by a hole and a square -/
def unoccupied_area (x : ℝ) : ℝ :=
  let large_rect := (2*x + 9) * (x + 6)
  let hole := (x - 1) * (2*x - 5)
  let square := (x + 3)^2
  large_rect - hole - square

/-- Theorem stating the unoccupied area in terms of x -/
theorem unoccupied_area_formula (x : ℝ) :
  unoccupied_area x = -x^2 + 22*x + 40 := by
  sorry

end unoccupied_area_formula_l564_56497


namespace function_equivalence_l564_56453

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x) + 1

theorem function_equivalence : ∀ x : ℝ, f x = g (x + 5 * π / 12) := by sorry

end function_equivalence_l564_56453


namespace intersection_of_M_and_N_l564_56438

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 2*x + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x < 1} := by sorry

end intersection_of_M_and_N_l564_56438


namespace average_problem_l564_56458

theorem average_problem (x : ℝ) : (15 + 25 + x) / 3 = 23 → x = 29 := by
  sorry

end average_problem_l564_56458


namespace zoo_giraffe_difference_l564_56485

theorem zoo_giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 →
  total_giraffes = 3 * other_animals →
  total_giraffes - other_animals = 200 := by
sorry

end zoo_giraffe_difference_l564_56485


namespace jose_investment_is_4500_l564_56486

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment given the shop investment scenario --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  sorry

/-- Theorem stating that Jose's investment is 4500 given the specific scenario --/
theorem jose_investment_is_4500 :
  let s : ShopInvestment := {
    tom_investment := 3000,
    jose_join_delay := 2,
    total_profit := 6300,
    jose_profit_share := 3500
  }
  calculate_jose_investment s = 4500 := by sorry

end jose_investment_is_4500_l564_56486


namespace polynomial_not_divisible_l564_56487

theorem polynomial_not_divisible (k : ℕ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^(2*k) + 1 + (x+1)^(2*k) ≠ 0) ↔ ¬(3 ∣ k) :=
sorry

end polynomial_not_divisible_l564_56487


namespace correct_selling_price_B_l564_56492

/-- Represents the pricing and sales data for laundry detergents --/
structure LaundryDetergentData where
  cost_diff : ℝ               -- Cost difference between brands
  total_cost_A : ℝ            -- Total cost for brand A
  total_cost_B : ℝ            -- Total cost for brand B
  sell_price_A : ℝ            -- Selling price of brand A
  daily_sales_A : ℝ           -- Daily sales of brand A
  base_price_B : ℝ            -- Base selling price of brand B
  base_sales_B : ℝ            -- Base daily sales of brand B
  price_sales_ratio : ℝ       -- Ratio of price increase to sales decrease for B

/-- Calculates the selling price of brand B for a given total daily profit --/
def calculate_selling_price_B (data : LaundryDetergentData) (total_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct selling price for brand B --/
theorem correct_selling_price_B (data : LaundryDetergentData) :
  let d := {
    cost_diff := 10,
    total_cost_A := 3000,
    total_cost_B := 4000,
    sell_price_A := 45,
    daily_sales_A := 100,
    base_price_B := 50,
    base_sales_B := 140,
    price_sales_ratio := 2
  }
  calculate_selling_price_B d 4700 = 80 := by sorry

end correct_selling_price_B_l564_56492


namespace arithmetic_sequence_middle_term_l564_56441

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 3^2 → a₃ = 3^4 → (a₃ - z = z - a₁) → z = 45 := by
  sorry

end arithmetic_sequence_middle_term_l564_56441


namespace mod_thirteen_four_eleven_l564_56452

theorem mod_thirteen_four_eleven (m : ℕ) : 
  13^4 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 5 := by
  sorry

end mod_thirteen_four_eleven_l564_56452


namespace star_properties_l564_56447

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- Theorem statement
theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧ 
  (∀ x : ℝ, star x (-1) = x ∧ star (-1) x = x) ∧
  (∀ x : ℝ, star x x = x^2 + 2*x) := by
  sorry

end star_properties_l564_56447


namespace well_depth_l564_56493

/-- Represents a circular well -/
structure CircularWell where
  diameter : ℝ
  volume : ℝ
  depth : ℝ

/-- Theorem stating the depth of a specific circular well -/
theorem well_depth (w : CircularWell) 
  (h1 : w.diameter = 4)
  (h2 : w.volume = 175.92918860102841) :
  w.depth = 14 := by
  sorry

end well_depth_l564_56493


namespace circle_sum_bounds_l564_56448

/-- The circle defined by the equation x² + y² - 4x + 2 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + 2 = 0}

/-- The sum function x + y for points (x, y) on the circle -/
def sum_func (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem circle_sum_bounds :
  ∃ (min max : ℝ), min = 0 ∧ max = 4 ∧
  ∀ p ∈ Circle, min ≤ sum_func p ∧ sum_func p ≤ max :=
sorry

end circle_sum_bounds_l564_56448


namespace cost_price_calculation_l564_56434

/-- Proves that given a selling price of 400 and a profit percentage of 60%, 
    the cost price of the article is 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 ∧ profit_percentage = 60 →
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 250 := by
sorry

end cost_price_calculation_l564_56434


namespace arithmetic_equality_l564_56419

theorem arithmetic_equality : (469138 * 9999) + (876543 * 12345) = 15512230997 := by
  sorry

end arithmetic_equality_l564_56419


namespace triangle_prime_count_l564_56468

def is_prime (n : ℕ) : Prop := sorry

def count_primes (a b : ℕ) : ℕ := sorry

def triangle_sides_valid (n : ℕ) : Prop :=
  let side1 := Real.log 16 / Real.log 8
  let side2 := Real.log 128 / Real.log 8
  let side3 := Real.log n / Real.log 8
  side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

theorem triangle_prime_count :
  ∀ n : ℕ, 
    n > 0 → 
    is_prime n → 
    triangle_sides_valid n →
    ∃ (count : ℕ), count = count_primes 9 4095 := by
  sorry

end triangle_prime_count_l564_56468


namespace hyperbola_equation_l564_56478

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  asymptote_slope : ℝ
  real_axis_length : ℝ
  foci_on_x_axis : Bool

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 9 = 1

/-- Theorem: Given a hyperbola with asymptote slope 3, real axis length 2, and foci on x-axis,
    its standard equation is x² - y²/9 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 3)
    (h_real_axis : h.real_axis_length = 2)
    (h_foci : h.foci_on_x_axis = true) :
    standard_equation h :=
  sorry

end hyperbola_equation_l564_56478


namespace roots_property_l564_56432

theorem roots_property (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a - 1) * (b - 1) = -2 := by
  sorry

end roots_property_l564_56432


namespace x_plus_y_value_l564_56464

theorem x_plus_y_value (x y : ℝ) (h1 : 1/x = 2) (h2 : 1/x + 3/y = 3) : x + y = 7/2 := by
  sorry

end x_plus_y_value_l564_56464


namespace log_expression_equals_two_l564_56477

-- Define base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end log_expression_equals_two_l564_56477
