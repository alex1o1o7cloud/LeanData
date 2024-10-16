import Mathlib

namespace NUMINAMATH_CALUDE_three_by_five_rectangle_triangles_l1726_172600

/-- Represents a rectangle divided into a grid with diagonal lines. -/
structure GridRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a GridRectangle. -/
def count_triangles (rect : GridRectangle) : Nat :=
  sorry

/-- Theorem stating that a 3x5 GridRectangle contains 76 triangles. -/
theorem three_by_five_rectangle_triangles :
  count_triangles ⟨3, 5⟩ = 76 := by
  sorry

end NUMINAMATH_CALUDE_three_by_five_rectangle_triangles_l1726_172600


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1726_172602

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1726_172602


namespace NUMINAMATH_CALUDE_opposite_edge_angles_not_all_acute_or_obtuse_l1726_172688

/-- Represents a convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  /-- All dihedral angles are 60° -/
  dihedral_angles_60 : Bool

/-- Represents the angles between opposite edges of a polyhedral angle -/
inductive OppositeEdgeAngles
  | Acute : OppositeEdgeAngles
  | Obtuse : OppositeEdgeAngles
  | Mixed : OppositeEdgeAngles

/-- 
Given a convex polyhedral angle with all dihedral angles equal to 60°, 
it's impossible for the angles between opposite edges to be simultaneously acute or simultaneously obtuse.
-/
theorem opposite_edge_angles_not_all_acute_or_obtuse (angle : ConvexPolyhedralAngle) 
  (h : angle.dihedral_angles_60 = true) : 
  ∃ (opp_angles : OppositeEdgeAngles), opp_angles = OppositeEdgeAngles.Mixed :=
sorry

end NUMINAMATH_CALUDE_opposite_edge_angles_not_all_acute_or_obtuse_l1726_172688


namespace NUMINAMATH_CALUDE_range_of_a_l1726_172612

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 ≤ 4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - a)^2 ≤ 1/4}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) →
  (-3 - Real.sqrt 5 / 2 ≤ a ∧ a ≤ -3 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1726_172612


namespace NUMINAMATH_CALUDE_winnieThePoohServings_l1726_172655

/-- Represents the number of servings eaten by each character -/
structure Servings where
  cheburashka : ℕ
  winnieThePooh : ℕ
  carlson : ℕ

/-- The rate at which characters eat relative to each other -/
def eatingRate (s : Servings) : Prop :=
  5 * s.cheburashka = 2 * s.winnieThePooh ∧
  7 * s.winnieThePooh = 3 * s.carlson

/-- The total number of servings eaten by Cheburashka and Carlson -/
def totalServings (s : Servings) : Prop :=
  s.cheburashka + s.carlson = 82

/-- Theorem stating that Winnie-the-Pooh ate 30 servings -/
theorem winnieThePoohServings (s : Servings) 
  (h1 : eatingRate s) (h2 : totalServings s) : s.winnieThePooh = 30 := by
  sorry

end NUMINAMATH_CALUDE_winnieThePoohServings_l1726_172655


namespace NUMINAMATH_CALUDE_max_salary_is_368000_l1726_172674

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a soccer team -/
def max_player_salary (team : SoccerTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem: The maximum possible salary for a single player in the given conditions is 368000 -/
theorem max_salary_is_368000 :
  let team : SoccerTeam := ⟨25, 18000, 800000⟩
  max_player_salary team = 368000 := by
  sorry

#eval max_player_salary ⟨25, 18000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_is_368000_l1726_172674


namespace NUMINAMATH_CALUDE_problem_solution_l1726_172607

theorem problem_solution (d : ℝ) (a b c : ℤ) (h1 : d ≠ 0) 
  (h2 : (18 * d + 19 + 20 * d^2) + (4 * d + 3 - 2 * d^2) = a * d + b + c * d^2) : 
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1726_172607


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1726_172619

/-- Proves that given the conditions of ticket prices, total revenue, and total tickets sold,
    the number of adult tickets sold is 22. -/
theorem adult_tickets_sold (adult_price child_price total_revenue total_tickets : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end NUMINAMATH_CALUDE_adult_tickets_sold_l1726_172619


namespace NUMINAMATH_CALUDE_coin_identification_l1726_172605

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Counterfeit

/-- Represents the result of weighing two groups of coins -/
inductive WeighResult
| Even
| Odd

/-- Function to determine the coin type based on the weighing result -/
def determineCoinType (result : WeighResult) : CoinType :=
  match result with
  | WeighResult.Even => CoinType.Genuine
  | WeighResult.Odd => CoinType.Counterfeit

theorem coin_identification
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∀ (specified_coin : CoinType) (weigh_result : WeighResult),
    determineCoinType weigh_result = specified_coin :=
  sorry

end NUMINAMATH_CALUDE_coin_identification_l1726_172605


namespace NUMINAMATH_CALUDE_batsman_new_average_l1726_172672

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  runsIn17thInning : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem stating the batsman's new average after the 17th inning -/
theorem batsman_new_average (b : Batsman) 
  (h1 : b.runsIn17thInning = 74)
  (h2 : b.averageIncrease = 3) : 
  newAverage b = 26 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l1726_172672


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1726_172616

-- Define the start time (9:00 AM) in minutes since midnight
def start_time : ℕ := 9 * 60

-- Define the time when one-fourth of the job is completed (12:20 PM) in minutes since midnight
def quarter_time : ℕ := 12 * 60 + 20

-- Define the completion time (10:20 PM) in minutes since midnight
def completion_time : ℕ := 22 * 60 + 20

-- Theorem statement
theorem doughnut_machine_completion_time :
  let quarter_duration : ℕ := quarter_time - start_time
  let total_duration : ℕ := 4 * quarter_duration
  start_time + total_duration = completion_time := by
  sorry


end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1726_172616


namespace NUMINAMATH_CALUDE_no_repeating_subsequence_l1726_172662

/-- Count the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Define the sequence a_n based on the parity of the number of 1's in the binary representation -/
def a (n : ℕ) : ℕ := 
  if countOnes n % 2 = 0 then 0 else 1

/-- The main theorem stating that there are no positive integers k and m satisfying the condition -/
theorem no_repeating_subsequence : 
  ¬ ∃ (k m : ℕ+), ∀ (j : ℕ), j < m → 
    a (k + j) = a (k + m + j) ∧ a (k + j) = a (k + 2*m + j) := by
  sorry

end NUMINAMATH_CALUDE_no_repeating_subsequence_l1726_172662


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l1726_172682

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 8*y + 40 = 0

def point : ℝ × ℝ := (4, -3)

theorem shortest_distance_to_circle :
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 - 1 ∧
  ∀ (p : ℝ × ℝ), circle_equation p.1 p.2 →
  Real.sqrt ((p.1 - point.1)^2 + (p.2 - point.2)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l1726_172682


namespace NUMINAMATH_CALUDE_prob_rain_sunday_and_monday_l1726_172639

-- Define the probabilities
def prob_rain_saturday : ℝ := 0.8
def prob_rain_sunday : ℝ := 0.3
def prob_rain_monday_if_sunday : ℝ := 0.5
def prob_rain_monday_if_not_sunday : ℝ := 0.1

-- Define the independence of Saturday and Sunday
axiom saturday_sunday_independent : True

-- Theorem to prove
theorem prob_rain_sunday_and_monday : 
  prob_rain_sunday * prob_rain_monday_if_sunday = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_sunday_and_monday_l1726_172639


namespace NUMINAMATH_CALUDE_division_problem_l1726_172626

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1726_172626


namespace NUMINAMATH_CALUDE_inequality_solution_l1726_172623

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 1 < y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1726_172623


namespace NUMINAMATH_CALUDE_integers_between_neg_sqrt2_and_sqrt2_l1726_172675

theorem integers_between_neg_sqrt2_and_sqrt2 :
  {x : ℤ | -Real.sqrt 2 < x ∧ x < Real.sqrt 2} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_integers_between_neg_sqrt2_and_sqrt2_l1726_172675


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1726_172673

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 0, 1, 2}
  let B : Set ℤ := {-2, 1, 2}
  A ∩ B = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1726_172673


namespace NUMINAMATH_CALUDE_combination_26_3_minus_10_l1726_172615

theorem combination_26_3_minus_10 : Nat.choose 26 3 - 10 = 2590 := by sorry

end NUMINAMATH_CALUDE_combination_26_3_minus_10_l1726_172615


namespace NUMINAMATH_CALUDE_three_integer_solutions_quadratic_inequality_l1726_172679

theorem three_integer_solutions_quadratic_inequality (b : ℤ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (∃ s : Finset ℤ, s.card = n ∧ 
      (∀ b' ∈ s, (∃! t : Finset ℤ, t.card = 3 ∧ 
        (∀ x ∈ t, x^2 + b' * x + 6 ≤ 0) ∧ 
        (∀ x : ℤ, x^2 + b' * x + 6 ≤ 0 → x ∈ t))))) :=
sorry

end NUMINAMATH_CALUDE_three_integer_solutions_quadratic_inequality_l1726_172679


namespace NUMINAMATH_CALUDE_number_of_divisors_3003_l1726_172633

theorem number_of_divisors_3003 : Nat.card (Nat.divisors 3003) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3003_l1726_172633


namespace NUMINAMATH_CALUDE_matchsticks_left_proof_l1726_172694

/-- The number of matchsticks left in the box after Elvis and Ralph make their squares -/
def matchsticks_left (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  total - (elvis_squares * elvis_per_square + ralph_squares * ralph_per_square)

/-- Theorem stating that 6 matchsticks will be left in the box -/
theorem matchsticks_left_proof :
  matchsticks_left 50 5 3 4 8 = 6 := by
  sorry

#eval matchsticks_left 50 5 3 4 8

end NUMINAMATH_CALUDE_matchsticks_left_proof_l1726_172694


namespace NUMINAMATH_CALUDE_find_a_l1726_172613

theorem find_a : ∃ a : ℚ, (a + 3) / 4 = (2 * a - 3) / 7 + 1 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1726_172613


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1726_172670

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a (n : ℕ) : ℝ := S n - S (n-1)

theorem arithmetic_sequence_proof :
  ∃ (d : ℝ), ∀ (n : ℕ), n ≥ 1 → a n = a 1 + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1726_172670


namespace NUMINAMATH_CALUDE_password_equation_l1726_172642

theorem password_equation : ∃ (A B C P Q R : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ P < 10 ∧ Q < 10 ∧ R < 10) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
   B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
   C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
   P ≠ Q ∧ P ≠ R ∧
   Q ≠ R) ∧
  3 * (100000 * A + 10000 * B + 1000 * C + 100 * P + 10 * Q + R) =
  4 * (100000 * P + 10000 * Q + 1000 * R + 100 * A + 10 * B + C) :=
by sorry

end NUMINAMATH_CALUDE_password_equation_l1726_172642


namespace NUMINAMATH_CALUDE_origin_outside_circle_l1726_172617

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let circle_equation (x y : ℝ) := x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2
  circle_equation 0 0 > 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l1726_172617


namespace NUMINAMATH_CALUDE_trapezoid_properties_l1726_172693

/-- Represents a trapezoid with side lengths and angles -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ

/-- The main theorem about trapezoid properties -/
theorem trapezoid_properties (t : Trapezoid) :
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.β + t.γ)) ∧
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.α + t.δ)) ∧
  (t.a * Real.sin t.α = t.c * Real.sin t.δ + t.b * Real.sin (t.α + t.β)) ∧
  (t.a * Real.sin t.β = t.c * Real.sin t.γ + t.d * Real.sin (t.α + t.β)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l1726_172693


namespace NUMINAMATH_CALUDE_job_completion_time_l1726_172684

/-- The time taken for two workers to complete a job together, given their relative speeds and the time taken by one worker alone. -/
theorem job_completion_time 
  (a_speed : ℝ) -- Speed of worker a
  (b_speed : ℝ) -- Speed of worker b
  (a_alone_time : ℝ) -- Time taken by worker a alone
  (h1 : a_speed = 1.5 * b_speed) -- a is 1.5 times as fast as b
  (h2 : a_alone_time = 30) -- a alone can do the work in 30 days
  : (1 / (1 / a_alone_time + 1 / (a_alone_time * 1.5))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1726_172684


namespace NUMINAMATH_CALUDE_max_product_constraint_l1726_172696

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1726_172696


namespace NUMINAMATH_CALUDE_right_triangle_min_perimeter_l1726_172692

theorem right_triangle_min_perimeter (a b c : ℝ) (h_area : a * b / 2 = 1) (h_right : a^2 + b^2 = c^2) :
  a + b + c ≥ 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_perimeter_l1726_172692


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l1726_172625

theorem spencer_walk_distance (house_to_library : ℝ) (library_to_post_office : ℝ) (post_office_to_home : ℝ)
  (h1 : house_to_library = 0.3)
  (h2 : library_to_post_office = 0.1)
  (h3 : post_office_to_home = 0.4) :
  house_to_library + library_to_post_office + post_office_to_home = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l1726_172625


namespace NUMINAMATH_CALUDE_circle_perimeter_l1726_172637

theorem circle_perimeter (r : ℝ) (h : r = 4 / Real.pi) : 
  2 * Real.pi * r = 8 := by sorry

end NUMINAMATH_CALUDE_circle_perimeter_l1726_172637


namespace NUMINAMATH_CALUDE_power_function_m_value_l1726_172649

/-- A function f(x) is a power function if it can be written in the form f(x) = ax^n, where a and n are constants and a ≠ 0. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- Given that y = (m^2 - 3)x^(2m) is a power function, m equals ±2. -/
theorem power_function_m_value (m : ℝ) :
  IsPowerFunction (fun x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l1726_172649


namespace NUMINAMATH_CALUDE_divide_by_four_twice_l1726_172635

theorem divide_by_four_twice (x : ℝ) : x = 166.08 → (x / 4) / 4 = 10.38 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_four_twice_l1726_172635


namespace NUMINAMATH_CALUDE_ratio_fifth_to_first_l1726_172678

/-- An arithmetic sequence with a non-zero common difference where a₁, a₂, and a₅ form a geometric sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The ratio of the fifth term to the first term in the special arithmetic sequence is 9. -/
theorem ratio_fifth_to_first (seq : ArithmeticSequence) : seq.a 5 / seq.a 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fifth_to_first_l1726_172678


namespace NUMINAMATH_CALUDE_f_extrema_a_range_l1726_172604

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 9

-- Theorem for part 1
theorem f_extrema :
  (∀ x ∈ Set.Icc 0 2, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -4) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ -3) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -3) :=
sorry

-- Theorem for part 2
theorem a_range :
  ∀ a < 0,
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a^2 ≥ 1 + Real.cos x) →
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_a_range_l1726_172604


namespace NUMINAMATH_CALUDE_tax_calculation_correct_l1726_172657

/-- Calculates the personal income tax based on the given salary and tax brackets. -/
def calculate_tax (salary : ℕ) : ℕ :=
  let taxable_income := salary - 5000
  let first_bracket := min taxable_income 3000
  let second_bracket := min (taxable_income - 3000) 9000
  let third_bracket := max (taxable_income - 12000) 0
  (first_bracket * 3 + second_bracket * 10 + third_bracket * 20) / 100

/-- Theorem stating that the calculated tax for a salary of 20000 yuan is 1590 yuan. -/
theorem tax_calculation_correct :
  calculate_tax 20000 = 1590 := by sorry

end NUMINAMATH_CALUDE_tax_calculation_correct_l1726_172657


namespace NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l1726_172614

theorem thirty_divides_p_squared_minus_one (p : ℕ) (h_prime : Nat.Prime p) (h_ge_seven : p ≥ 7) :
  30 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l1726_172614


namespace NUMINAMATH_CALUDE_f_maximum_l1726_172606

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
by sorry

end NUMINAMATH_CALUDE_f_maximum_l1726_172606


namespace NUMINAMATH_CALUDE_correct_calculation_l1726_172648

/-- Represents the loan and investment scenario -/
structure LoanInvestment where
  loan_amount : ℝ
  interest_paid : ℝ
  business_profit_rate : ℝ
  total_profit : ℝ

/-- Calculates the interest rate and investment amount -/
def calculate_rate_and_investment (scenario : LoanInvestment) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem correct_calculation (scenario : LoanInvestment) :
  scenario.loan_amount = 150000 ∧
  scenario.interest_paid = 42000 ∧
  scenario.business_profit_rate = 0.1 ∧
  scenario.total_profit = 25000 →
  let (rate, investment) := calculate_rate_and_investment scenario
  rate = 0.05 ∧ investment = 50000 :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1726_172648


namespace NUMINAMATH_CALUDE_solution_system_equations_l1726_172651

theorem solution_system_equations :
  ∃ (a b : ℝ), 
    (a * 2 + b * 1 = 7 ∧ a * 2 - b * 1 = 1) → 
    (a - b = -1) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l1726_172651


namespace NUMINAMATH_CALUDE_union_of_positive_and_less_than_one_is_reals_l1726_172621

theorem union_of_positive_and_less_than_one_is_reals :
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x < 1}
  A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_positive_and_less_than_one_is_reals_l1726_172621


namespace NUMINAMATH_CALUDE_letter_150_is_z_l1726_172698

def repeating_sequence : ℕ → Char
  | n => if n % 3 = 1 then 'X' else if n % 3 = 2 then 'Y' else 'Z'

theorem letter_150_is_z : repeating_sequence 150 = 'Z' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_z_l1726_172698


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_coefficients_l1726_172603

def f (x : ℝ) : ℝ := 2 * x^2 - x + 5

def g (x : ℝ) : ℝ := f (x - 7) + 3

theorem quadratic_shift_sum_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 86) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_coefficients_l1726_172603


namespace NUMINAMATH_CALUDE_unique_solution_k_l1726_172666

theorem unique_solution_k : ∃! k : ℚ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_k_l1726_172666


namespace NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1726_172695

/-- Represents a structure made of unit cubes -/
structure CubeStructure where
  base : Nat × Nat × Nat  -- dimensions of the base cube
  stacked : Nat  -- number of cubes stacked on top
  total : Nat  -- total number of cubes

/-- Calculates the surface area of a cube structure -/
def surfaceArea (cs : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure in the problem -/
def problemStructure : CubeStructure :=
  { base := (2, 2, 2),
    stacked := 4,
    total := 12 }

theorem surface_area_of_problem_structure :
  surfaceArea problemStructure = 32 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1726_172695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1726_172622

def isArithmeticSequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence b →
  a 1 = 25 →
  b 1 = 125 →
  a 2 + b 2 = 150 →
  (a + b) 2006 = 150 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1726_172622


namespace NUMINAMATH_CALUDE_work_completion_days_l1726_172667

/-- The number of days required for the second group to complete the work -/
def days_for_second_group : ℕ := 4

/-- The daily work output of a boy -/
def boy_work : ℝ := 1

/-- The daily work output of a man -/
def man_work : ℝ := 2 * boy_work

/-- The total amount of work to be done -/
def total_work : ℝ := (12 * man_work + 16 * boy_work) * 5

theorem work_completion_days :
  (13 * man_work + 24 * boy_work) * days_for_second_group = total_work := by sorry

end NUMINAMATH_CALUDE_work_completion_days_l1726_172667


namespace NUMINAMATH_CALUDE_sam_speed_calculation_l1726_172650

def alex_speed : ℚ := 6
def jamie_relative_speed : ℚ := 4/5
def sam_relative_speed : ℚ := 3/4

theorem sam_speed_calculation :
  alex_speed * jamie_relative_speed * sam_relative_speed = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_calculation_l1726_172650


namespace NUMINAMATH_CALUDE_solve_final_grade_problem_l1726_172610

def final_grade_problem (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ) : Prop :=
  let fraction_D := 1 - (fraction_A + fraction_B + fraction_C)
  let num_D := total_students - (total_students * (fraction_A + fraction_B + fraction_C)).floor
  (total_students = 100) ∧
  (fraction_A = 1/5) ∧
  (fraction_B = 1/4) ∧
  (fraction_C = 1/2) ∧
  (num_D = 5)

theorem solve_final_grade_problem :
  ∃ (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ),
    final_grade_problem total_students fraction_A fraction_B fraction_C :=
by
  sorry

end NUMINAMATH_CALUDE_solve_final_grade_problem_l1726_172610


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l1726_172697

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x + 5 < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l1726_172697


namespace NUMINAMATH_CALUDE_hyperbola_foci_intersection_l1726_172676

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then a = 3 and b = 4. -/
theorem hyperbola_foci_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2) →
  (∃ (x y : ℝ), x^2 + y^2 = c^2 ∧ y/x = b/a ∧ x = 3 ∧ y = 4) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_intersection_l1726_172676


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l1726_172646

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion (x y : ℝ) (h : (x, y) = (8, 2 * Real.sqrt 3)) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 19 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l1726_172646


namespace NUMINAMATH_CALUDE_bills_age_l1726_172641

/-- Proves Bill's age given the conditions of the problem -/
theorem bills_age :
  ∀ (bill_age caroline_age : ℕ),
    bill_age = 2 * caroline_age - 1 →
    bill_age + caroline_age = 26 →
    bill_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_bills_age_l1726_172641


namespace NUMINAMATH_CALUDE_soft_drink_pack_size_l1726_172611

/-- The number of cans in a pack of soft drinks -/
def num_cans : ℕ := 11

/-- The cost of a pack of soft drinks in dollars -/
def pack_cost : ℚ := 299/100

/-- The cost of an individual can in dollars -/
def can_cost : ℚ := 1/4

/-- Theorem stating that the number of cans in a pack is 11 -/
theorem soft_drink_pack_size :
  num_cans = ⌊pack_cost / can_cost⌋ := by sorry

end NUMINAMATH_CALUDE_soft_drink_pack_size_l1726_172611


namespace NUMINAMATH_CALUDE_arrangements_count_l1726_172609

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the condition that B and C must be adjacent -/
def bc_adjacent : Prop := True

/-- Represents the condition that A cannot stand at either end -/
def a_not_at_ends : Prop := True

/-- The number of different arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 -/
theorem arrangements_count :
  (num_students = 6) →
  bc_adjacent →
  a_not_at_ends →
  num_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1726_172609


namespace NUMINAMATH_CALUDE_f_form_l1726_172677

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_continuous : Continuous f
axiom f_functional_equation : ∀ x y : ℝ, f (Real.sqrt (x^2 + y^2)) = f x * f y

-- State the theorem to be proved
theorem f_form : ∀ x : ℝ, f x = (f 1) ^ (x^2) := by sorry

end NUMINAMATH_CALUDE_f_form_l1726_172677


namespace NUMINAMATH_CALUDE_inequality_and_equality_l1726_172656

theorem inequality_and_equality (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (b < -1 ∨ b > 0 → (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l1726_172656


namespace NUMINAMATH_CALUDE_factorial_division_l1726_172661

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1726_172661


namespace NUMINAMATH_CALUDE_at_least_one_red_probability_l1726_172689

theorem at_least_one_red_probability
  (prob_red_A prob_red_B : ℚ)
  (h_prob_A : prob_red_A = 1/3)
  (h_prob_B : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_red_probability_l1726_172689


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l1726_172630

def num_chickens : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 6
def total_animals : ℕ := num_chickens + num_dogs + num_cats

def group_arrangements : ℕ := 3

theorem animal_arrangement_count :
  (group_arrangements * num_chickens.factorial * num_dogs.factorial * num_cats.factorial : ℕ) = 1555200 :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l1726_172630


namespace NUMINAMATH_CALUDE_tangent_circles_a_values_l1726_172699

/-- Two circles are tangent if the distance between their centers is equal to
    the sum or difference of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 + r2)^2) ∨
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 - r2)^2)

theorem tangent_circles_a_values :
  ∀ a : ℝ,
  are_tangent (0, 0) (-4, a) 1 5 →
  (a = 0 ∨ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_a_values_l1726_172699


namespace NUMINAMATH_CALUDE_valid_distributions_count_l1726_172624

/-- Represents a triangular array of squares with 11 rows -/
def TriangularArray := Fin 11 → Fin 11 → ℕ

/-- Represents the bottom row of the triangular array -/
def BottomRow := Fin 11 → Fin 2

/-- Calculates the value of a square in the array based on the two squares below it -/
def calculateSquare (array : TriangularArray) (row : Fin 11) (col : Fin 11) : ℕ :=
  if row = 10 then array row col
  else array (row + 1) col + array (row + 1) (col + 1)

/-- Fills the triangular array based on the bottom row -/
def fillArray (bottomRow : BottomRow) : TriangularArray :=
  sorry

/-- Checks if the top square of the array is a multiple of 3 -/
def isTopMultipleOfThree (array : TriangularArray) : Bool :=
  array 0 0 % 3 = 0

/-- Counts the number of valid bottom row distributions -/
def countValidDistributions : ℕ :=
  sorry

theorem valid_distributions_count :
  countValidDistributions = 640 := by sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l1726_172624


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1726_172631

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1726_172631


namespace NUMINAMATH_CALUDE_tree_height_calculation_l1726_172608

/-- Given a tree and a pole with their respective shadows, calculate the height of the tree -/
theorem tree_height_calculation (tree_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  tree_shadow = 30 →
  pole_height = 1.5 →
  pole_shadow = 3 →
  (tree_shadow * pole_height) / pole_shadow = 15 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_calculation_l1726_172608


namespace NUMINAMATH_CALUDE_sequence_equality_l1726_172638

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l1726_172638


namespace NUMINAMATH_CALUDE_weakly_increasing_h_implies_b_eq_one_l1726_172664

/-- A function is weakly increasing in an interval if it's increasing and its ratio to x is decreasing in that interval --/
def WeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∧
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x / x ≥ f y / y)

/-- The function h(x) = x^2 - (b-1)x + b --/
def h (b : ℝ) (x : ℝ) : ℝ := x^2 - (b-1)*x + b

theorem weakly_increasing_h_implies_b_eq_one :
  ∀ b : ℝ, WeaklyIncreasing (h b) 0 1 → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_weakly_increasing_h_implies_b_eq_one_l1726_172664


namespace NUMINAMATH_CALUDE_wendy_bought_four_chairs_l1726_172653

def furniture_problem (chairs : ℕ) : Prop :=
  let tables : ℕ := 4
  let time_per_piece : ℕ := 6
  let total_time : ℕ := 48
  (chairs + tables) * time_per_piece = total_time

theorem wendy_bought_four_chairs :
  ∃ (chairs : ℕ), furniture_problem chairs ∧ chairs = 4 :=
sorry

end NUMINAMATH_CALUDE_wendy_bought_four_chairs_l1726_172653


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l1726_172686

/-- Regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side : ℝ
  height : ℝ

/-- Cutting plane for the pyramid -/
structure CuttingPlane where
  distance_from_apex : ℝ

/-- The area of the cross-section of a regular hexagonal pyramid -/
noncomputable def cross_section_area (p : HexagonalPyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section for the given conditions -/
theorem cross_section_area_theorem (p : HexagonalPyramid) (c : CuttingPlane) :
  p.base_side = 2 →
  c.distance_from_apex = 1 →
  cross_section_area p c = 34 * Real.sqrt 3 / 35 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l1726_172686


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l1726_172634

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l1726_172634


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1726_172636

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (4*x + 1) + Real.sqrt (4*y + 1) + Real.sqrt (4*z + 1) ≤ Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1)) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1726_172636


namespace NUMINAMATH_CALUDE_projection_obtuse_implies_obtuse_projection_acute_inconclusive_l1726_172658

/-- Represents an angle --/
structure Angle where
  measure : ℝ
  is_positive : 0 < measure

/-- Represents the rectangular projection of an angle onto a plane --/
def rectangular_projection (α : Angle) : Angle :=
  sorry

/-- An angle is obtuse if its measure is greater than π/2 --/
def is_obtuse (α : Angle) : Prop :=
  α.measure > Real.pi / 2

/-- An angle is acute if its measure is less than π/2 --/
def is_acute (α : Angle) : Prop :=
  α.measure < Real.pi / 2

theorem projection_obtuse_implies_obtuse (α : Angle) :
  is_obtuse (rectangular_projection α) → is_obtuse α :=
sorry

theorem projection_acute_inconclusive (α : Angle) :
  is_acute (rectangular_projection α) → 
  (is_acute α ∨ is_obtuse α) :=
sorry

end NUMINAMATH_CALUDE_projection_obtuse_implies_obtuse_projection_acute_inconclusive_l1726_172658


namespace NUMINAMATH_CALUDE_dennis_purchase_cost_l1726_172620

/-- The cost of Dennis's purchase after discount --/
def total_cost (pants_price sock_price : ℚ) (pants_quantity sock_quantity : ℕ) (discount : ℚ) : ℚ :=
  let discounted_pants_price := pants_price * (1 - discount)
  let discounted_sock_price := sock_price * (1 - discount)
  (discounted_pants_price * pants_quantity) + (discounted_sock_price * sock_quantity)

/-- Theorem stating the total cost of Dennis's purchase --/
theorem dennis_purchase_cost :
  total_cost 110 60 4 2 (30/100) = 392 := by
  sorry

end NUMINAMATH_CALUDE_dennis_purchase_cost_l1726_172620


namespace NUMINAMATH_CALUDE_five_distinct_dice_probability_l1726_172629

def standard_dice_sides : ℕ := 6

def distinct_rolls (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | k + 1 => (standard_dice_sides - k) * distinct_rolls k

theorem five_distinct_dice_probability : 
  (distinct_rolls 5 : ℚ) / (standard_dice_sides ^ 5) = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_five_distinct_dice_probability_l1726_172629


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_l1726_172644

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c }

/-- Reflects a parabola about the x-axis -/
def reflect (p : Parabola) : Parabola :=
  { a := -p.a
  , b := -p.b
  , c := -p.c }

/-- Adds two parabolas coefficient-wise -/
def add (p q : Parabola) : Parabola :=
  { a := p.a + q.a
  , b := p.b + q.b
  , c := p.c + q.c }

theorem parabola_transformation_sum (p : Parabola) :
  let p1 := translate p 4
  let p2 := translate (reflect p) (-4)
  (add p1 p2).b = -16 * p.a ∧ (add p1 p2).a = 0 ∧ (add p1 p2).c = 0 := by
  sorry

#check parabola_transformation_sum

end NUMINAMATH_CALUDE_parabola_transformation_sum_l1726_172644


namespace NUMINAMATH_CALUDE_ampersand_composition_l1726_172669

-- Define the & operation
def ampersand_right (y : ℤ) : ℤ := 9 - y

-- Define the & operation
def ampersand_left (y : ℤ) : ℤ := y - 9

-- Theorem to prove
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1726_172669


namespace NUMINAMATH_CALUDE_chameleon_color_change_l1726_172654

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 ∧ blue_factor = 5 ∧ red_factor = 3 →
  ∃ (initial_blue initial_red changed : ℕ),
    initial_blue + initial_red = total ∧
    changed = initial_blue - (initial_blue / blue_factor) ∧
    initial_red + changed = (initial_red * red_factor) ∧
    changed = 80 := by
  sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l1726_172654


namespace NUMINAMATH_CALUDE_set_difference_M_N_range_of_a_l1726_172645

-- Define set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - 1)}
def N : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a*x - 1 ∧ a*x - 1 ≤ 5}
def B : Set ℝ := {y | -1/2 < y ∧ y ≤ 2}

-- Theorem 1
theorem set_difference_M_N : set_difference M N = {x | x > 1} := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : set_difference (A a) B = ∅ → a < -12 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_set_difference_M_N_range_of_a_l1726_172645


namespace NUMINAMATH_CALUDE_electronics_store_profit_l1726_172690

theorem electronics_store_profit (n : ℕ) (CA : ℝ) : 
  let CB := 2 * CA
  let SA := (2 / 3) * CA
  let SB := 1.2 * CB
  let total_cost := n * CA + n * CB
  let total_sales := n * SA + n * SB
  (total_sales - total_cost) / total_cost = 0.1
  := by sorry

end NUMINAMATH_CALUDE_electronics_store_profit_l1726_172690


namespace NUMINAMATH_CALUDE_third_circle_radius_is_15_14_l1726_172660

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to two other circles -/
def is_tangent_to_two_circles (c : Circle) (c1 c2 : Circle) : Prop :=
  are_externally_tangent c c1 ∧ are_externally_tangent c c2

/-- Checks if a circle is tangent to the common external tangent of two other circles -/
def is_tangent_to_common_external_tangent (c : Circle) (c1 c2 : Circle) : Prop :=
  sorry  -- The actual implementation would depend on how we represent tangent lines

theorem third_circle_radius_is_15_14 (c1 c2 c3 : Circle) : 
  c1.radius = 2 →
  c2.radius = 3 →
  are_externally_tangent c1 c2 →
  is_tangent_to_two_circles c3 c1 c2 →
  is_tangent_to_common_external_tangent c3 c1 c2 →
  c3.radius = 15/14 :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_is_15_14_l1726_172660


namespace NUMINAMATH_CALUDE_walter_age_2005_l1726_172691

theorem walter_age_2005 (walter_age_2000 : ℕ) (grandmother_age_2000 : ℕ) : 
  walter_age_2000 = grandmother_age_2000 / 3 →
  (2000 - walter_age_2000) + (2000 - grandmother_age_2000) = 3896 →
  walter_age_2000 + 5 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_2005_l1726_172691


namespace NUMINAMATH_CALUDE_volume_of_rectangular_prism_l1726_172647

/-- Represents a rectangular prism with dimensions a, d, and h -/
structure RectangularPrism where
  a : ℝ
  d : ℝ
  h : ℝ
  a_pos : 0 < a
  d_pos : 0 < d
  h_pos : 0 < h

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.d * prism.h

/-- Theorem: The volume of a rectangular prism is equal to a * d * h -/
theorem volume_of_rectangular_prism (prism : RectangularPrism) :
  volume prism = prism.a * prism.d * prism.h :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rectangular_prism_l1726_172647


namespace NUMINAMATH_CALUDE_min_value_product_equality_condition_l1726_172671

theorem min_value_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) 
  (h_eq : x₁ = π/4 ∧ x₂ = π/4 ∧ x₃ = π/4 ∧ x₄ = π/4) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) = 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_equality_condition_l1726_172671


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1726_172628

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x ≥ 1 + a ∨ x ≤ 1 - a}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem 1: If A ∩ B = ∅, then a ≥ 5
theorem intersection_empty_implies_a_ge_5 (a : ℝ) (h : a > 0) :
  A ∩ B a = ∅ → a ≥ 5 := by sorry

-- Theorem 2: If ¬p is a sufficient but not necessary condition for q, then 0 < a ≤ 2
theorem not_p_sufficient_not_necessary_implies_a_le_2 (a : ℝ) (h : a > 0) :
  (∀ x, ¬p x → q a x) ∧ (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1726_172628


namespace NUMINAMATH_CALUDE_bus_problem_solution_l1726_172643

/-- Represents the problem of distributing passengers among buses --/
structure BusProblem where
  m : ℕ  -- Initial number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  initialPassengers : ℕ  -- Initial number of passengers per bus
  maxCapacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus problem --/
def validBusProblem (bp : BusProblem) : Prop :=
  bp.m ≥ 2 ∧
  bp.initialPassengers = 22 ∧
  bp.maxCapacity = 32 ∧
  bp.n ≤ bp.maxCapacity ∧
  bp.initialPassengers * bp.m + 1 = bp.n * (bp.m - 1)

/-- The theorem stating the solution to the bus problem --/
theorem bus_problem_solution (bp : BusProblem) (h : validBusProblem bp) :
  bp.m = 24 ∧ bp.n * (bp.m - 1) = 529 := by
  sorry

#check bus_problem_solution

end NUMINAMATH_CALUDE_bus_problem_solution_l1726_172643


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l1726_172687

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 ≤ 0

-- Theorem for part (1)
theorem range_of_x_when_m_is_one (x : ℝ) :
  (∃ m : ℝ, m = 1 ∧ m > 0 ∧ (p x ∨ q x m)) → x ∈ Set.Icc 1 8 :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_sufficient_condition (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) →
  m ∈ Set.Icc 2 (8/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l1726_172687


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1726_172680

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (2 / (2 + i)) = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1726_172680


namespace NUMINAMATH_CALUDE_exists_M_with_properties_l1726_172668

def is_valid_last_four_digits (n : ℕ) : Prop :=
  n < 10000 ∧ 
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 4*a - 3*b ∧
  (n / 1000 ≠ (n / 100) % 10) ∧
  (n / 1000 ≠ (n / 10) % 10) ∧
  (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ (n / 10) % 10) ∧
  (n / 100 % 10 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem exists_M_with_properties :
  ∃ (M : ℕ), 
    M % 8 = 0 ∧
    M % 16 ≠ 0 ∧
    is_valid_last_four_digits (M % 10000) ∧
    M % 1000 = 624 :=
sorry

end NUMINAMATH_CALUDE_exists_M_with_properties_l1726_172668


namespace NUMINAMATH_CALUDE_problem_statement_l1726_172685

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1726_172685


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l1726_172632

theorem modulus_of_complex_power (z : ℂ) :
  z = 2 - 3 * Real.sqrt 2 * Complex.I →
  Complex.abs (z^4) = 484 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l1726_172632


namespace NUMINAMATH_CALUDE_right_triangle_check_l1726_172652

/-- Check if three numbers form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_check :
  ¬ is_right_triangle (1/3) (1/4) (1/5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 (Real.sqrt 3) 4 :=
by sorry

#check right_triangle_check

end NUMINAMATH_CALUDE_right_triangle_check_l1726_172652


namespace NUMINAMATH_CALUDE_solve_equation_chain_l1726_172683

theorem solve_equation_chain (v y z w x : ℤ) 
  (h1 : x = y + 6)
  (h2 : y = z + 11)
  (h3 : z = w + 21)
  (h4 : w = v + 30)
  (h5 : v = 90) : x = 158 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_chain_l1726_172683


namespace NUMINAMATH_CALUDE_ratio_of_sums_and_differences_l1726_172663

theorem ratio_of_sums_and_differences (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
  (x + 1 / x) / (x - 1 / x) = Real.sqrt 7 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_and_differences_l1726_172663


namespace NUMINAMATH_CALUDE_slope_angle_of_line_through_origin_and_unit_point_l1726_172627

/-- The slope angle of a line passing through (0,0) and (1,1) is π/4 -/
theorem slope_angle_of_line_through_origin_and_unit_point :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (1, 1)
  let slope : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_through_origin_and_unit_point_l1726_172627


namespace NUMINAMATH_CALUDE_triangle_count_corner_l1726_172681

-- Define a structure for a point on a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a structure for a rectangle
structure Rectangle :=
  (A B C D E F : Point)

-- Define a function to count triangles with one vertex at a given point
def countTriangles (r : Rectangle) (p : Point) : ℕ :=
  9

-- Theorem statement
theorem triangle_count_corner (r : Rectangle) :
  (countTriangles r r.A = 9) ∧ (countTriangles r r.F = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_corner_l1726_172681


namespace NUMINAMATH_CALUDE_three_prime_pairs_sum_52_l1726_172665

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def count_prime_pairs (sum : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (sum - p)) (Finset.range (sum / 2 + 1))).card / 2

/-- Theorem stating that there are exactly 3 unordered pairs of prime numbers that sum to 52 -/
theorem three_prime_pairs_sum_52 : count_prime_pairs 52 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_prime_pairs_sum_52_l1726_172665


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1726_172640

theorem smallest_divisible_by_1_to_10 : ∀ n : ℕ, n > 0 → (∀ i : ℕ, 1 ≤ i → i ≤ 10 → i ∣ n) → n ≥ 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1726_172640


namespace NUMINAMATH_CALUDE_divisor_power_difference_l1726_172659

theorem divisor_power_difference (k : ℕ) : 
  (18 ^ k : ℕ) ∣ 624938 → 6 ^ k - k ^ 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l1726_172659


namespace NUMINAMATH_CALUDE_smallest_even_natural_with_properties_l1726_172618

def is_smallest_even_natural_with_properties (a : ℕ) : Prop :=
  Even a ∧
  (∃ k₁, a + 1 = 3 * k₁) ∧
  (∃ k₂, a + 2 = 5 * k₂) ∧
  (∃ k₃, a + 3 = 7 * k₃) ∧
  (∃ k₄, a + 4 = 11 * k₄) ∧
  (∃ k₅, a + 5 = 13 * k₅) ∧
  (∀ b < a, ¬(is_smallest_even_natural_with_properties b))

theorem smallest_even_natural_with_properties : 
  is_smallest_even_natural_with_properties 788 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_natural_with_properties_l1726_172618


namespace NUMINAMATH_CALUDE_product_of_imaginary_parts_l1726_172601

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 + 3*z + (4 - 7*i) = 0

-- Define a function to get the imaginary part of a complex number
def im (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts :
  ∃ (z1 z2 : ℂ), quadratic_eq z1 ∧ quadratic_eq z2 ∧ z1 ≠ z2 ∧ (im z1 * im z2 = -14) :=
sorry

end NUMINAMATH_CALUDE_product_of_imaginary_parts_l1726_172601
