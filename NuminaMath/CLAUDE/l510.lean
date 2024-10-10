import Mathlib

namespace mhsc_unanswered_questions_l510_51023

/-- Represents the scoring system for the Math High School Contest -/
structure ScoringSystem where
  initial : ℤ
  correct : ℤ
  wrong : ℤ
  unanswered : ℤ

/-- Calculates the score based on a given scoring system and number of questions -/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem mhsc_unanswered_questions (newSystem oldSystem : ScoringSystem)
    (totalQuestions newScore oldScore : ℕ) :
    newSystem = ScoringSystem.mk 0 6 0 1 →
    oldSystem = ScoringSystem.mk 25 5 (-2) 0 →
    totalQuestions = 30 →
    newScore = 110 →
    oldScore = 95 →
    ∃ (correct wrong unanswered : ℕ),
      correct + wrong + unanswered = totalQuestions ∧
      calculateScore newSystem correct wrong unanswered = newScore ∧
      calculateScore oldSystem correct wrong unanswered = oldScore ∧
      unanswered = 10 :=
  sorry


end mhsc_unanswered_questions_l510_51023


namespace perpendicular_and_parallel_relationships_l510_51045

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_and_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
by sorry

end perpendicular_and_parallel_relationships_l510_51045


namespace inequalities_solution_l510_51007

theorem inequalities_solution :
  (∀ x : ℝ, (x - 2) * (1 - 3 * x) > 2 ↔ 1 < x ∧ x < 4/3) ∧
  (∀ x : ℝ, |((x + 1) / (x - 1))| > 2 ↔ (1/3 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)) :=
by sorry

end inequalities_solution_l510_51007


namespace monthly_expenses_calculation_l510_51034

/-- Calculates monthly expenses given initial investment, monthly revenue, and payback period. -/
def calculate_monthly_expenses (initial_investment : ℕ) (monthly_revenue : ℕ) (payback_months : ℕ) : ℕ :=
  (monthly_revenue * payback_months - initial_investment) / payback_months

theorem monthly_expenses_calculation (initial_investment monthly_revenue payback_months : ℕ) 
  (h1 : initial_investment = 25000)
  (h2 : monthly_revenue = 4000)
  (h3 : payback_months = 10) :
  calculate_monthly_expenses initial_investment monthly_revenue payback_months = 1500 := by
  sorry

#eval calculate_monthly_expenses 25000 4000 10

end monthly_expenses_calculation_l510_51034


namespace digit_sum_divisible_by_11_l510_51038

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any sequence of 39 consecutive natural numbers, 
    there exists at least one number whose digit sum is divisible by 11 -/
theorem digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (N + k) % 11 = 0) := by sorry

end digit_sum_divisible_by_11_l510_51038


namespace negative_fraction_comparison_l510_51008

theorem negative_fraction_comparison : -3/5 < -1/3 := by
  sorry

end negative_fraction_comparison_l510_51008


namespace complex_real_condition_l510_51027

theorem complex_real_condition (a : ℝ) :
  (((a - Complex.I) / (2 + Complex.I)).im = 0) → a = -2 := by sorry

end complex_real_condition_l510_51027


namespace people_remaining_on_bus_l510_51043

/-- The number of people remaining on a bus after a field trip with multiple stops -/
theorem people_remaining_on_bus 
  (left_side : ℕ) 
  (right_side : ℕ) 
  (back_section : ℕ) 
  (standing : ℕ) 
  (teachers : ℕ) 
  (driver : ℕ) 
  (first_stop : ℕ) 
  (second_stop : ℕ) 
  (third_stop : ℕ) 
  (h1 : left_side = 42)
  (h2 : right_side = 38)
  (h3 : back_section = 5)
  (h4 : standing = 15)
  (h5 : teachers = 2)
  (h6 : driver = 1)
  (h7 : first_stop = 15)
  (h8 : second_stop = 19)
  (h9 : third_stop = 5) :
  left_side + right_side + back_section + standing + teachers + driver - 
  (first_stop + second_stop + third_stop) = 64 :=
by sorry

end people_remaining_on_bus_l510_51043


namespace book_arrangement_theorem_l510_51070

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem stating the number of arrangements for 6 books with 3 identical -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

end book_arrangement_theorem_l510_51070


namespace odd_power_divisibility_l510_51068

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∀ n : ℕ, ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) :=
by sorry

end odd_power_divisibility_l510_51068


namespace second_discount_percentage_l510_51015

def initial_price : ℝ := 400
def first_discount : ℝ := 20
def final_price : ℝ := 272

theorem second_discount_percentage :
  ∃ (second_discount : ℝ),
    initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price ∧
    second_discount = 15 := by
  sorry

end second_discount_percentage_l510_51015


namespace negative_one_less_than_negative_two_thirds_l510_51065

theorem negative_one_less_than_negative_two_thirds : -1 < -(2/3) := by
  sorry

end negative_one_less_than_negative_two_thirds_l510_51065


namespace solve_amusement_park_problem_l510_51052

def amusement_park_problem (ticket_price : ℕ) (weekday_visitors : ℕ) (saturday_visitors : ℕ) (total_revenue : ℕ) : Prop :=
  let weekday_total := weekday_visitors * 5
  let sunday_visitors := (total_revenue - ticket_price * (weekday_total + saturday_visitors)) / ticket_price
  sunday_visitors = 300

theorem solve_amusement_park_problem :
  amusement_park_problem 3 100 200 3000 := by
  sorry

end solve_amusement_park_problem_l510_51052


namespace intersection_sum_l510_51025

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- Define the condition for the intersection points
def intersection_condition (k : ℝ) : Prop :=
  |log3 k - log3 (k + 5)| = 0.6

-- Define the form of k
def k_form (a b : ℤ) (k : ℝ) : Prop :=
  k = a + Real.sqrt (b : ℝ)

-- Main theorem
theorem intersection_sum (k : ℝ) (a b : ℤ) :
  intersection_condition k → k_form a b k → a + b = 8 :=
by sorry

end intersection_sum_l510_51025


namespace quadratic_function_properties_l510_51042

/-- The quadratic function y = x^2 - ax + a + 3 -/
def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3

theorem quadratic_function_properties (a : ℝ) :
  (∃ x, f a x = 0) ↔ (a ≤ -2 ∨ a ≥ 6) ∧
  (∀ x, f a x ≥ 4 ↔ 
    (a > 2 ∧ (x ≤ 1 ∨ x ≥ a - 1)) ∨
    (a = 2 ∧ true) ∨
    (a < 2 ∧ (x ≤ a - 1 ∨ x ≥ 1))) ∧
  ((∃ x ∈ Set.Icc 2 4, f a x = 0) → a ∈ Set.Icc 6 7) :=
by sorry


end quadratic_function_properties_l510_51042


namespace symmetric_angle_ratio_l510_51099

/-- 
Given a point P(x,y) on the terminal side of an angle θ (excluding the origin), 
where the terminal side of θ is symmetric to the terminal side of a 480° angle 
with respect to the x-axis, prove that xy/(x^2 + y^2) = √3/4.
-/
theorem symmetric_angle_ratio (x y : ℝ) (h1 : x ≠ 0 ∨ y ≠ 0) 
  (h2 : y = Real.sqrt 3 * x) : 
  (x * y) / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end symmetric_angle_ratio_l510_51099


namespace comparison_proofs_l510_51057

theorem comparison_proofs :
  (-5 < -2) ∧ (-1/3 > -1/2) ∧ (abs (-5) > 0) := by
  sorry

end comparison_proofs_l510_51057


namespace middle_share_in_ratio_l510_51006

/-- Proves that in a 3:5:7 ratio distribution with a 1200 difference between extremes, the middle value is 1500 -/
theorem middle_share_in_ratio (total : ℕ) : 
  let f := 3 * total / 15
  let v := 5 * total / 15
  let r := 7 * total / 15
  r - f = 1200 → v = 1500 := by
  sorry

end middle_share_in_ratio_l510_51006


namespace first_hour_coins_is_20_l510_51014

/-- The number of coins Tina put in the jar during the first hour -/
def first_hour_coins : ℕ := sorry

/-- The number of coins Tina put in the jar during the second hour -/
def second_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the third hour -/
def third_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins Tina took out of the jar during the fifth hour -/
def fifth_hour_coins : ℕ := 20

/-- The total number of coins in the jar after the fifth hour -/
def total_coins : ℕ := 100

/-- Theorem stating that the number of coins Tina put in during the first hour is 20 -/
theorem first_hour_coins_is_20 :
  first_hour_coins = 20 :=
by
  have h : first_hour_coins + second_hour_coins + third_hour_coins + fourth_hour_coins - fifth_hour_coins = total_coins := sorry
  sorry


end first_hour_coins_is_20_l510_51014


namespace smallest_B_for_divisibility_l510_51092

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 80000 + 83961

theorem smallest_B_for_divisibility :
  ∀ B : ℕ, B < 10 →
    (is_divisible_by_4 (seven_digit_number B) → B ≥ 0) ∧
    is_divisible_by_4 (seven_digit_number 0) :=
sorry

end smallest_B_for_divisibility_l510_51092


namespace base5_23104_equals_1654_l510_51051

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (d₄ d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₄ * 5^4 + d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- The base 5 number 23104 is equal to 1654 in base 10 --/
theorem base5_23104_equals_1654 :
  base5ToBase10 2 3 1 0 4 = 1654 := by
  sorry

end base5_23104_equals_1654_l510_51051


namespace sphere_radius_proof_l510_51056

theorem sphere_radius_proof (a b c : ℝ) : 
  (a + b + c = 40) →
  (2 * a * b + 2 * b * c + 2 * c * a = 512) →
  (∃ r : ℝ, r^2 = 130 ∧ r^2 * 4 = a^2 + b^2 + c^2) :=
by
  sorry

end sphere_radius_proof_l510_51056


namespace log_comparison_l510_51097

theorem log_comparison : Real.log 2009 / Real.log 2008 > Real.log 2010 / Real.log 2009 := by sorry

end log_comparison_l510_51097


namespace probability_all_different_digits_l510_51096

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_all_different_digits (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.toFinset.card = 3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_all_different_digits : ℕ := 675

theorem probability_all_different_digits :
  (count_numbers_with_all_different_digits : ℚ) / count_three_digit_numbers = 3 / 4 := by
  sorry

end probability_all_different_digits_l510_51096


namespace log_ratio_squared_l510_51017

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (Real.log a)^2 - 4 * (Real.log a) + 1 = 0) →
  (2 * (Real.log b)^2 - 4 * (Real.log b) + 1 = 0) →
  ((Real.log (a / b))^2 = 2) := by
sorry

end log_ratio_squared_l510_51017


namespace arithmetic_sequence_sum_divisibility_l510_51028

theorem arithmetic_sequence_sum_divisibility :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x c : ℕ), x > 0 → c ≥ 0 → 
    n ∣ (10 * x + 45 * c)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (x c : ℕ), x > 0 ∧ c ≥ 0 ∧ 
      ¬(m ∣ (10 * x + 45 * c))) :=
by sorry

end arithmetic_sequence_sum_divisibility_l510_51028


namespace arithmetic_sequence_sum_l510_51085

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence property
  a 1 = 0 →                        -- first term is 0
  d ≠ 0 →                          -- common difference is non-zero
  a k = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) →  -- sum condition
  k = 22 := by
sorry

end arithmetic_sequence_sum_l510_51085


namespace sin_2theta_value_l510_51073

/-- Given vectors a, b, c, and the condition that (2a - b) is parallel to c, 
    prove that sin(2θ) = -12/13 --/
theorem sin_2theta_value (θ : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (Real.sin θ, 1))
  (hb : b = (-Real.sin θ, 0))
  (hc : c = (Real.cos θ, -1))
  (h_parallel : ∃ (k : ℝ), (2 • a - b) = k • c) :
  Real.sin (2 * θ) = -12/13 := by
  sorry

end sin_2theta_value_l510_51073


namespace problem_1_problem_2_problem_3_problem_4_l510_51064

-- Problem 1
theorem problem_1 : 1/2 + (-2/3) + 4/5 + (-1/2) + (-1/3) = -1/5 := by sorry

-- Problem 2
theorem problem_2 : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1 + 1/6) * (-48) = -46 := by sorry

-- Problem 4
theorem problem_4 : -2^4 - 32 / ((-2)^3 + 4) = -8 := by sorry

end problem_1_problem_2_problem_3_problem_4_l510_51064


namespace determine_set_B_l510_51044

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the theorem
theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {1}) 
  (h2 : A ∩ Bᶜ = {3}) 
  (h3 : A ⊆ U) 
  (h4 : B ⊆ U) : 
  B = {2, 4, 5} := by
  sorry

end determine_set_B_l510_51044


namespace rectangle_formations_with_restrictions_l510_51050

/-- The number of ways to choose 4 lines to form a rectangle -/
def rectangleFormations (h v : ℕ) (hRestricted vRestricted : Fin 2 → ℕ) : ℕ :=
  let hChoices := (Nat.choose h 2) - 1
  let vChoices := (Nat.choose v 2) - 1
  hChoices * vChoices

/-- Theorem stating the number of ways to form a rectangle with given conditions -/
theorem rectangle_formations_with_restrictions :
  rectangleFormations 6 7 ![2, 5] ![3, 6] = 280 := by
  sorry

end rectangle_formations_with_restrictions_l510_51050


namespace will_chocolate_pieces_l510_51000

/-- Calculates the number of chocolate pieces Will has left after giving some boxes away. -/
def chocolate_pieces_left (total_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (total_boxes - boxes_given) * pieces_per_box

/-- Proves that Will has 16 pieces of chocolate left after giving some boxes to his brother. -/
theorem will_chocolate_pieces : chocolate_pieces_left 7 3 4 = 16 := by
  sorry

end will_chocolate_pieces_l510_51000


namespace a_profit_calculation_l510_51013

def total_subscription : ℕ := 50000
def total_profit : ℕ := 36000

def subscription_difference_a_b : ℕ := 4000
def subscription_difference_b_c : ℕ := 5000

def c_subscription (x : ℕ) : ℕ := x
def b_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c
def a_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c + subscription_difference_a_b

theorem a_profit_calculation :
  ∃ x : ℕ,
    c_subscription x + b_subscription x + a_subscription x = total_subscription ∧
    (a_subscription x : ℚ) / (total_subscription : ℚ) * (total_profit : ℚ) = 15120 :=
  sorry

end a_profit_calculation_l510_51013


namespace cube_root_equation_solution_l510_51093

theorem cube_root_equation_solution (x : ℝ) :
  (5 + 2 / x) ^ (1/3 : ℝ) = -3 → x = -(1/16) := by sorry

end cube_root_equation_solution_l510_51093


namespace ball_trajectory_5x5_table_l510_51020

/-- Represents a square pool table --/
structure PoolTable :=
  (size : Nat)

/-- Represents a ball's trajectory on the pool table --/
structure BallTrajectory :=
  (table : PoolTable)
  (start_corner : Nat × Nat)
  (angle : Real)

/-- Represents the final state of the ball --/
structure FinalState :=
  (end_pocket : String)
  (edge_hits : Nat)
  (diagonal_squares : Nat)

/-- Main theorem about the ball's trajectory on a 5x5 pool table --/
theorem ball_trajectory_5x5_table :
  ∀ (t : PoolTable) (b : BallTrajectory),
    t.size = 5 →
    b.table = t →
    b.start_corner = (0, 0) →
    b.angle = 45 →
    ∃ (f : FinalState),
      f.end_pocket = "upper-left" ∧
      f.edge_hits = 5 ∧
      f.diagonal_squares = 23 :=
sorry

end ball_trajectory_5x5_table_l510_51020


namespace equation_one_solution_system_of_equations_solution_l510_51046

-- Equation (1)
theorem equation_one_solution (x : ℝ) : 
  2 * (x - 2) - 3 * (4 * x - 1) = 9 * (1 - x) ↔ x = -10 := by sorry

-- System of Equations (2)
theorem system_of_equations_solution (x y : ℝ) :
  (4 * (x - y - 1) = 3 * (1 - y) - 2 ∧ x / 2 + y / 3 = 2) ↔ (x = 2 ∧ y = 3) := by sorry

end equation_one_solution_system_of_equations_solution_l510_51046


namespace sum_15_l510_51082

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- Sum function

/-- The sum of the first 5 terms is 3 -/
axiom sum_5 (ap : ArithmeticProgression) : ap.S 5 = 3

/-- The sum of the first 10 terms is 12 -/
axiom sum_10 (ap : ArithmeticProgression) : ap.S 10 = 12

/-- Theorem: If S_5 = 3 and S_10 = 12, then S_15 = 39 -/
theorem sum_15 (ap : ArithmeticProgression) : ap.S 15 = 39 := by
  sorry


end sum_15_l510_51082


namespace arithmetic_sequence_inequality_l510_51095

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For any arithmetic sequence, if a₁ ≥ a₂, then a₂² ≥ a₁a₃ -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 1 ≥ a 2 → a 2 ^ 2 ≥ a 1 * a 3 := by
  sorry

end arithmetic_sequence_inequality_l510_51095


namespace shortest_tangent_length_l510_51090

/-- The shortest length of the tangent from a point on the line x - y + 2√2 = 0 to the circle x² + y² = 1 is √3 -/
theorem shortest_tangent_length (x y : ℝ) : 
  (x - y + 2 * Real.sqrt 2 = 0) →
  (x^2 + y^2 = 1) →
  ∃ (px py : ℝ), 
    (px - py + 2 * Real.sqrt 2 = 0) ∧
    Real.sqrt ((px - x)^2 + (py - y)^2) ≥ Real.sqrt 3 :=
by sorry

end shortest_tangent_length_l510_51090


namespace four_girls_wins_l510_51022

theorem four_girls_wins (a b c d : ℕ) : 
  a + b = 8 ∧ 
  a + c = 10 ∧ 
  b + c = 12 ∧ 
  a + d = 12 ∧ 
  b + d = 14 ∧ 
  c + d = 16 → 
  ({a, b, c, d} : Finset ℕ) = {3, 5, 7, 9} := by
sorry

end four_girls_wins_l510_51022


namespace star_value_l510_51061

-- Define the * operation
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

-- State the theorem
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 7) (h4 : a * b = 12) :
  star a b = 7 / 12 := by
  sorry

end star_value_l510_51061


namespace multiple_of_ab_l510_51016

theorem multiple_of_ab (a b : ℕ+) : 
  (∃ k : ℕ, a.val ^ 2017 + b.val = k * a.val * b.val) ↔ 
  ((a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2^2017)) := by
sorry

end multiple_of_ab_l510_51016


namespace bottle_cap_count_l510_51040

/-- Given the total cost of bottle caps and the cost per bottle cap,
    prove that the number of bottle caps is correct. -/
theorem bottle_cap_count 
  (total_cost : ℝ) 
  (cost_per_cap : ℝ) 
  (h1 : total_cost = 25) 
  (h2 : cost_per_cap = 5) : 
  total_cost / cost_per_cap = 5 := by
  sorry

#check bottle_cap_count

end bottle_cap_count_l510_51040


namespace all_statements_imply_target_l510_51072

theorem all_statements_imply_target (p q r : Prop) :
  (p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (p ∧ q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ q ∧ ¬r → ((p → q) → ¬r)) :=
by sorry

end all_statements_imply_target_l510_51072


namespace product_of_extremes_is_cube_l510_51036

theorem product_of_extremes_is_cube (a : Fin 2022 → ℕ)
  (h : ∀ i : Fin 2021, ∃ k : ℕ, a i * a (i.succ) = k^3) :
  ∃ m : ℕ, a 0 * a 2021 = m^3 := by
  sorry

end product_of_extremes_is_cube_l510_51036


namespace varya_used_discount_l510_51098

/-- Represents the quantity of items purchased by each girl -/
structure Purchase where
  pens : ℕ
  pencils : ℕ
  notebooks : ℕ

/-- Given the purchases of three girls and the fact that they all paid equally,
    prove that the second girl (Varya) must have used a discount -/
theorem varya_used_discount (p k l : ℚ) (anya varya sasha : Purchase) 
    (h_positive : p > 0 ∧ k > 0 ∧ l > 0)
    (h_anya : anya = ⟨2, 7, 1⟩)
    (h_varya : varya = ⟨5, 6, 5⟩)
    (h_sasha : sasha = ⟨8, 4, 9⟩)
    (h_equal_payment : ∃ (x : ℚ), 
      x = p * anya.pens + k * anya.pencils + l * anya.notebooks ∧
      x = p * varya.pens + k * varya.pencils + l * varya.notebooks ∧
      x = p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) :
  p * varya.pens + k * varya.pencils + l * varya.notebooks < 
  (p * anya.pens + k * anya.pencils + l * anya.notebooks + 
   p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) / 2 := by
  sorry


end varya_used_discount_l510_51098


namespace russian_dolls_discount_l510_51055

theorem russian_dolls_discount (original_price : ℝ) (original_quantity : ℕ) (discount_rate : ℝ) :
  original_price = 4 →
  original_quantity = 15 →
  discount_rate = 0.2 →
  ⌊(original_price * original_quantity) / (original_price * (1 - discount_rate))⌋ = 18 :=
by
  sorry

end russian_dolls_discount_l510_51055


namespace touching_balls_in_cylinder_l510_51081

theorem touching_balls_in_cylinder (a b d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (h_touch : a + b = d)
  (h_larger_bottom : a ≥ b) : 
  Real.sqrt d = Real.sqrt a + Real.sqrt b := by
sorry

end touching_balls_in_cylinder_l510_51081


namespace largest_mersenne_prime_under_500_l510_51067

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (n : ℕ) : Prop := ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_under_500 : 
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧ 
  is_mersenne_prime 127 ∧ 
  127 < 500 :=
sorry

end largest_mersenne_prime_under_500_l510_51067


namespace waitress_income_fraction_l510_51002

theorem waitress_income_fraction (salary : ℚ) (salary_pos : salary > 0) :
  let first_week_tips := (11 / 4) * salary
  let second_week_tips := (7 / 3) * salary
  let total_salary := 2 * salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 61 / 85 := by
  sorry

end waitress_income_fraction_l510_51002


namespace necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l510_51029

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ m ≠ 4

/-- The condition 2 < m < 6 is necessary for the equation to represent an ellipse -/
theorem necessary_condition (m : ℝ) :
  is_ellipse m → 2 < m ∧ m < 6 := by sorry

/-- The condition 2 < m < 6 is not sufficient for the equation to represent an ellipse -/
theorem not_sufficient_condition :
  ∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m) := by sorry

/-- The main theorem stating that 2 < m < 6 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_ellipse m → 2 < m ∧ m < 6) ∧
  (∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m)) := by sorry

end necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l510_51029


namespace time_equation_l510_51009

/-- Given the equations V = 2gt + V₀ and S = (1/3)gt² + V₀t + Ct³, where C is a constant,
    prove that the time t can be expressed as t = (V - V₀) / (2g). -/
theorem time_equation (g V V₀ S t : ℝ) (C : ℝ) :
  V = 2 * g * t + V₀ ∧ S = (1/3) * g * t^2 + V₀ * t + C * t^3 →
  t = (V - V₀) / (2 * g) := by
  sorry

end time_equation_l510_51009


namespace divisor_exists_l510_51001

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem divisor_exists : ∃ d : ℕ, 
  d > 0 ∧ 
  is_prime (9453 / d) ∧ 
  is_perfect_square (9453 % d) ∧ 
  d = 61 := by
sorry

end divisor_exists_l510_51001


namespace circle_radius_is_sqrt13_l510_51005

/-- A circle with two given points on its circumference and its center on the y-axis -/
structure CircleWithPoints where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  point1_on_circle : (point1.1 - center.1)^2 + (point1.2 - center.2)^2 = (point2.1 - center.1)^2 + (point2.2 - center.2)^2

/-- The radius of the circle is √13 -/
theorem circle_radius_is_sqrt13 (c : CircleWithPoints) 
  (h1 : c.point1 = (2, 5)) 
  (h2 : c.point2 = (3, 6)) : 
  Real.sqrt ((c.point1.1 - c.center.1)^2 + (c.point1.2 - c.center.2)^2) = Real.sqrt 13 := by
  sorry


end circle_radius_is_sqrt13_l510_51005


namespace regression_unit_increase_food_expenditure_increase_l510_51087

/-- Represents a linear regression equation ŷ = ax + b -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Calculates the predicted value for a given x -/
def LinearRegression.predict (reg : LinearRegression) (x : ℝ) : ℝ :=
  reg.a * x + reg.b

/-- The increase in ŷ when x increases by 1 is equal to the coefficient a -/
theorem regression_unit_increase (reg : LinearRegression) :
  reg.predict (x + 1) - reg.predict x = reg.a :=
by sorry

/-- The specific regression equation from the problem -/
def food_expenditure_regression : LinearRegression :=
  { a := 0.254, b := 0.321 }

/-- The increase in food expenditure when income increases by 1 is 0.254 -/
theorem food_expenditure_increase :
  food_expenditure_regression.predict (x + 1) - food_expenditure_regression.predict x = 0.254 :=
by sorry

end regression_unit_increase_food_expenditure_increase_l510_51087


namespace two_points_determine_line_l510_51031

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Two points determine a unique line
theorem two_points_determine_line (P Q : Point) (h : P ≠ Q) :
  ∃! L : Line, (L.a * P.x + L.b * P.y + L.c = 0) ∧ (L.a * Q.x + L.b * Q.y + L.c = 0) :=
sorry

end two_points_determine_line_l510_51031


namespace greatest_four_digit_satisfying_conditions_l510_51084

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_satisfying_conditions :
  is_four_digit 9999 ∧
  ¬(product_of_first_n 9999 % sum_of_first_n 9999 = 0) ∧
  is_perfect_square (9999 + 1) ∧
  ∀ n : ℕ, is_four_digit n →
    n > 9999 ∨
    (product_of_first_n n % sum_of_first_n n = 0) ∨
    ¬(is_perfect_square (n + 1)) :=
  sorry

end greatest_four_digit_satisfying_conditions_l510_51084


namespace smallest_divisor_of_1025_l510_51059

theorem smallest_divisor_of_1025 : 
  ∀ n : ℕ, n > 1 → n ∣ 1025 → n ≥ 5 := by
  sorry

end smallest_divisor_of_1025_l510_51059


namespace gcf_of_54_and_72_l510_51079

theorem gcf_of_54_and_72 : Nat.gcd 54 72 = 18 := by
  sorry

end gcf_of_54_and_72_l510_51079


namespace trick_or_treat_distribution_l510_51018

/-- The number of blocks in the village -/
def num_blocks : ℕ := 9

/-- The total number of children going trick or treating -/
def total_children : ℕ := 54

/-- There are some children on each block -/
axiom children_on_each_block : ∀ b : ℕ, b < num_blocks → ∃ c : ℕ, c > 0

/-- The number of children on each block -/
def children_per_block : ℕ := total_children / num_blocks

theorem trick_or_treat_distribution :
  children_per_block = 6 :=
sorry

end trick_or_treat_distribution_l510_51018


namespace loan_division_l510_51019

theorem loan_division (total : ℝ) (rate1 rate2 years1 years2 : ℝ) : 
  total = 2665 ∧ rate1 = 3/100 ∧ rate2 = 5/100 ∧ years1 = 5 ∧ years2 = 3 →
  ∃ (part1 part2 : ℝ), 
    part1 + part2 = total ∧
    part1 * rate1 * years1 = part2 * rate2 * years2 ∧
    part2 = 1332.5 := by
  sorry

end loan_division_l510_51019


namespace complex_number_theorem_l510_51069

theorem complex_number_theorem (z : ℂ) :
  (z^2).im = 0 ∧ Complex.abs (z - Complex.I) = 1 → z = 0 ∨ z = 2 * Complex.I := by
  sorry

end complex_number_theorem_l510_51069


namespace circle_properties_l510_51054

/-- Given a circle with circumference 36 cm, prove its diameter and area. -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  let d := 2 * r
  let A := Real.pi * r^2
  d = 36 / Real.pi ∧ A = 324 / Real.pi := by
  sorry


end circle_properties_l510_51054


namespace candy_distribution_l510_51076

theorem candy_distribution (marta_candies carmem_candies : ℕ) : 
  (marta_candies + carmem_candies = 200) →
  (marta_candies < 100) →
  (marta_candies > (4 * carmem_candies) / 5) →
  (∃ k : ℕ, marta_candies = 8 * k) →
  (∃ l : ℕ, carmem_candies = 8 * l) →
  (marta_candies = 96 ∧ carmem_candies = 104) := by
sorry

end candy_distribution_l510_51076


namespace new_average_after_adding_l510_51075

theorem new_average_after_adding (n : ℕ) (original_avg : ℚ) (add_value : ℚ) : 
  n > 0 → 
  let original_sum := n * original_avg
  let new_sum := original_sum + n * add_value
  let new_avg := new_sum / n
  n = 15 ∧ original_avg = 40 ∧ add_value = 10 → new_avg = 50 := by
  sorry

end new_average_after_adding_l510_51075


namespace sqrt_sum_equals_seven_l510_51010

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l510_51010


namespace sqrt_equation_solution_l510_51089

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (5 + 2 * z) = 11 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l510_51089


namespace animal_survival_probability_l510_51026

theorem animal_survival_probability (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.7) 
  (h2 : p_25 = 0.56) : 
  p_25 / p_20 = 0.8 := by
  sorry

end animal_survival_probability_l510_51026


namespace line_slope_intercept_sum_l510_51066

/-- Given a line passing through points (2, -3) and (5, 6), prove that m + b = -6 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b ↔ (x = 2 ∧ y = -3) ∨ (x = 5 ∧ y = 6)) →
  m + b = -6 := by
sorry

end line_slope_intercept_sum_l510_51066


namespace total_heads_calculation_l510_51041

theorem total_heads_calculation (num_hens : ℕ) (total_feet : ℕ) : 
  num_hens = 24 → total_feet = 136 → ∃ (num_cows : ℕ), num_hens + num_cows = 46 := by
  sorry

end total_heads_calculation_l510_51041


namespace edward_book_spending_l510_51024

/-- The amount of money Edward spent on books -/
def money_spent (num_books : ℕ) (cost_per_book : ℕ) : ℕ :=
  num_books * cost_per_book

/-- Theorem: Edward spent $6 on books -/
theorem edward_book_spending :
  money_spent 2 3 = 6 := by
  sorry

end edward_book_spending_l510_51024


namespace parabola_through_points_l510_51004

/-- A parabola passing through three specific points -/
def parabola (x y : ℝ) : Prop :=
  y = -x^2 + 2*x + 3

theorem parabola_through_points :
  parabola (-1) 0 ∧ parabola 3 0 ∧ parabola 0 3 := by
  sorry

end parabola_through_points_l510_51004


namespace smallest_sum_for_equation_l510_51078

theorem smallest_sum_for_equation (m n : ℕ+) (h : 3 * m ^ 3 = 5 * n ^ 5) :
  ∀ (x y : ℕ+), 3 * x ^ 3 = 5 * y ^ 5 → m + n ≤ x + y :=
by sorry

end smallest_sum_for_equation_l510_51078


namespace basketball_score_proof_l510_51080

theorem basketball_score_proof (two_points : ℕ) (three_points : ℕ) (free_throws : ℕ) :
  (3 * three_points = 2 * (2 * two_points)) →
  (free_throws = 2 * two_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end basketball_score_proof_l510_51080


namespace intersection_of_M_and_N_l510_51077

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by sorry

end intersection_of_M_and_N_l510_51077


namespace displacement_increment_from_2_to_2_plus_d_l510_51003

/-- Represents the displacement of an object at time t -/
def displacement (t : ℝ) : ℝ := 2 * t^2

/-- Represents the increment in displacement between two time points -/
def displacementIncrement (t₁ t₂ : ℝ) : ℝ := displacement t₂ - displacement t₁

theorem displacement_increment_from_2_to_2_plus_d (d : ℝ) :
  displacementIncrement 2 (2 + d) = 8 * d + 2 * d^2 := by
  sorry

end displacement_increment_from_2_to_2_plus_d_l510_51003


namespace pages_left_to_read_l510_51074

def total_pages : ℕ := 563
def pages_read : ℕ := 147

theorem pages_left_to_read : total_pages - pages_read = 416 := by
  sorry

end pages_left_to_read_l510_51074


namespace sum_of_xyz_l510_51021

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) : 
  x + y + z = 14 * Real.sqrt 5 := by
sorry

end sum_of_xyz_l510_51021


namespace miley_bought_two_cellphones_l510_51011

/-- The number of cellphones Miley bought -/
def num_cellphones : ℕ := 2

/-- The cost of each cellphone in dollars -/
def cost_per_cellphone : ℝ := 800

/-- The discount rate for buying more than one cellphone -/
def discount_rate : ℝ := 0.05

/-- The total amount Miley paid in dollars -/
def total_paid : ℝ := 1520

/-- Theorem stating that the number of cellphones Miley bought is 2 -/
theorem miley_bought_two_cellphones :
  num_cellphones = 2 ∧
  num_cellphones > 1 ∧
  (1 - discount_rate) * (num_cellphones : ℝ) * cost_per_cellphone = total_paid :=
by sorry

end miley_bought_two_cellphones_l510_51011


namespace income_calculation_l510_51012

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 15 →
  income - expenditure = savings →
  savings = 7000 →
  income = 15000 := by
sorry

end income_calculation_l510_51012


namespace system_solution_l510_51037

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -2) ∧ (8 * x + 5 * y = 7) ∧ x = 1/4 ∧ y = 1 := by
  sorry

end system_solution_l510_51037


namespace square_of_1008_l510_51062

theorem square_of_1008 : (1008 : ℕ)^2 = 1016064 := by
  sorry

end square_of_1008_l510_51062


namespace correct_attitude_towards_superstitions_l510_51086

/-- Represents different types of online superstitions -/
inductive OnlineSuperstition
  | AstrologicalFate
  | HoroscopeInterpretation
  | NorthStarBook
  | DreamInterpretation

/-- Represents possible attitudes towards online superstitions -/
inductive Attitude
  | Accept
  | StayAway
  | RespectDiversity
  | ImproveDiscernment

/-- Defines the correct attitude for teenage students -/
def correct_attitude : Attitude := Attitude.ImproveDiscernment

/-- Theorem stating the correct attitude towards online superstitions -/
theorem correct_attitude_towards_superstitions :
  ∀ (s : OnlineSuperstition), correct_attitude = Attitude.ImproveDiscernment :=
by sorry

end correct_attitude_towards_superstitions_l510_51086


namespace tracy_balloons_l510_51047

theorem tracy_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (total_after : ℕ) :
  brooke_initial = 12 →
  brooke_added = 8 →
  tracy_initial = 6 →
  total_after = 35 →
  ∃ (tracy_added : ℕ),
    brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after ∧
    tracy_added = 24 :=
by sorry

end tracy_balloons_l510_51047


namespace trig_problem_l510_51035

theorem trig_problem (θ : Real) 
  (h1 : θ > 0) 
  (h2 : θ < Real.pi / 2) 
  (h3 : Real.cos (θ + Real.pi / 6) = 1 / 3) : 
  Real.sin θ = (2 * Real.sqrt 6 - 1) / 6 ∧ 
  Real.sin (2 * θ + Real.pi / 6) = (4 * Real.sqrt 6 + 7) / 18 := by
sorry

end trig_problem_l510_51035


namespace room_length_l510_51060

/-- Proves that a rectangular room with given volume, height, and width has a specific length -/
theorem room_length (volume : ℝ) (height : ℝ) (width : ℝ) (length : ℝ) 
  (h_volume : volume = 10000)
  (h_height : height = 10)
  (h_width : width = 10)
  (h_room_volume : volume = length * width * height) :
  length = 100 :=
by sorry

end room_length_l510_51060


namespace intersection_P_Q_nonempty_intersection_P_R_nonempty_l510_51091

-- Define the sets P, Q, and R
def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 > 0}
def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 = 0}

-- Theorem for part 1
theorem intersection_P_Q_nonempty (a : ℝ) : 
  (P ∩ Q a).Nonempty → a > -1/2 :=
sorry

-- Theorem for part 2
theorem intersection_P_R_nonempty (a : ℝ) : 
  (P ∩ R a).Nonempty → a ≥ -1/2 ∧ a ≤ 1/2 :=
sorry

end intersection_P_Q_nonempty_intersection_P_R_nonempty_l510_51091


namespace sqrt_equation_solution_l510_51033

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt x + Real.sqrt (x + 8) = 8 ∧ x = 49/4 := by sorry

end sqrt_equation_solution_l510_51033


namespace polynomial_remainder_l510_51048

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 3*x^3 + 2*x^2 + 11*x - 6
  (f x) % (x - 2) = 16 := by
  sorry

end polynomial_remainder_l510_51048


namespace solution_count_l510_51039

-- Define the equation
def equation (x a : ℝ) : Prop :=
  Real.log (2 - x^2) / Real.log (x - a) = 2

-- Theorem statement
theorem solution_count (a : ℝ) :
  (∀ x, ¬ equation x a) ∨
  (∃! x, equation x a) ∨
  (∃ x y, x ≠ y ∧ equation x a ∧ equation y a) :=
by
  -- Case 1: No solution
  have h1 : a ≤ -2 ∨ a = 0 ∨ a ≥ Real.sqrt 2 → ∀ x, ¬ equation x a := by sorry
  -- Case 2: One solution
  have h2 : (-Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < Real.sqrt 2) → ∃! x, equation x a := by sorry
  -- Case 3: Two solutions
  have h3 : -2 < a ∧ a < -Real.sqrt 2 → ∃ x y, x ≠ y ∧ equation x a ∧ equation y a := by sorry
  sorry -- Complete the proof using h1, h2, and h3


end solution_count_l510_51039


namespace C₁_cartesian_polar_equiv_l510_51032

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + 16 = 0

/-- The curve C₁ in polar coordinates -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ^2 - 8*ρ*Real.cos θ - 10*ρ*Real.sin θ + 16 = 0

/-- Theorem stating the equivalence of Cartesian and polar representations of C₁ -/
theorem C₁_cartesian_polar_equiv :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    (C₁ x y ↔ C₁_polar ρ θ) :=
by
  sorry

end C₁_cartesian_polar_equiv_l510_51032


namespace complex_number_problem_l510_51071

theorem complex_number_problem (a b c : ℂ) (h_a_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end complex_number_problem_l510_51071


namespace functional_equation_polynomial_l510_51058

/-- A polynomial that satisfies the functional equation P(X^2 + 1) = P(X)^2 + 1 and P(0) = 0 is equal to the identity function. -/
theorem functional_equation_polynomial (P : Polynomial ℝ) 
  (h1 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1)
  (h2 : P.eval 0 = 0) : 
  P = Polynomial.X :=
sorry

end functional_equation_polynomial_l510_51058


namespace minimum_horses_and_ponies_l510_51063

theorem minimum_horses_and_ponies (ponies horses : ℕ) : 
  (3 * ponies % 10 = 0) →  -- 3/10 of ponies have horseshoes
  (5 * (3 * ponies / 10) % 8 = 0) →  -- 5/8 of ponies with horseshoes are from Iceland
  (horses = ponies + 3) →  -- 3 more horses than ponies
  (∀ p h, p < ponies ∨ h < horses → 
    3 * p % 10 ≠ 0 ∨ 
    5 * (3 * p / 10) % 8 ≠ 0 ∨ 
    h ≠ p + 3) →  -- minimality condition
  ponies + horses = 163 := by
sorry

end minimum_horses_and_ponies_l510_51063


namespace grass_field_path_problem_l510_51083

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem grass_field_path_problem (field_length field_width path_width cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 675 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1350 := by
  sorry

#check grass_field_path_problem

end grass_field_path_problem_l510_51083


namespace f_nonnegative_implies_a_bound_f_inequality_l510_51094

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

theorem f_nonnegative_implies_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) → a ≥ 1 / Real.exp 1 := by sorry

theorem f_inequality (a : ℝ) (x₁ x₂ x : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h : x₁ < x ∧ x < x₂) :
  (f a x - f a x₁) / (x - x₁) < (f a x - f a x₂) / (x - x₂) := by sorry

end f_nonnegative_implies_a_bound_f_inequality_l510_51094


namespace shorter_base_length_for_specific_trapezoid_l510_51053

/-- Represents a trapezoid with a median line divided by a diagonal -/
structure TrapezoidWithDiagonal where
  median_length : ℝ
  segment_difference : ℝ

/-- Calculates the length of the shorter base of the trapezoid -/
def shorter_base_length (t : TrapezoidWithDiagonal) : ℝ :=
  t.median_length - t.segment_difference

/-- Theorem stating the length of the shorter base given specific measurements -/
theorem shorter_base_length_for_specific_trapezoid :
  let t : TrapezoidWithDiagonal := { median_length := 16, segment_difference := 4 }
  shorter_base_length t = 12 := by
  sorry

end shorter_base_length_for_specific_trapezoid_l510_51053


namespace wage_increase_for_unit_productivity_increase_l510_51049

/-- Regression line equation for workers' wages as a function of labor productivity -/
def regression_line (x : ℝ) : ℝ := 80 * x + 50

/-- Theorem: The average increase in wage when labor productivity increases by 1 unit -/
theorem wage_increase_for_unit_productivity_increase :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
sorry

end wage_increase_for_unit_productivity_increase_l510_51049


namespace remainder_proof_l510_51088

theorem remainder_proof : (7 * 10^20 + 1^20) % 9 = 8 := by
  sorry

end remainder_proof_l510_51088


namespace hockey_players_count_l510_51030

/-- The number of hockey players in a games hour -/
def hockey_players (total players cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem: There are 17 hockey players in the ground -/
theorem hockey_players_count : hockey_players 50 12 11 10 = 17 := by
  sorry

end hockey_players_count_l510_51030
