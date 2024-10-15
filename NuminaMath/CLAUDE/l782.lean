import Mathlib

namespace NUMINAMATH_CALUDE_square_side_length_l782_78213

theorem square_side_length (w h r s : ℕ) : 
  w = 4000 →
  h = 2300 →
  2 * r + s = h →
  2 * r + 3 * s = w →
  s = 850 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l782_78213


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l782_78228

/-- Given a line passing through the points (-2, -3) and (3, 4),
    the product of its slope and y-intercept is equal to -7/25. -/
theorem line_slope_intercept_product : 
  let p₁ : ℝ × ℝ := (-2, -3)
  let p₂ : ℝ × ℝ := (3, 4)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)  -- slope
  let b : ℝ := p₁.2 - m * p₁.1  -- y-intercept
  m * b = -7/25 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l782_78228


namespace NUMINAMATH_CALUDE_fraction_addition_theorem_l782_78272

theorem fraction_addition_theorem (a b c d x : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : c ≠ d) 
  (h4 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_theorem_l782_78272


namespace NUMINAMATH_CALUDE_march_greatest_percent_difference_l782_78276

/-- Represents the sales data for a group in a given month -/
structure SalesData where
  drummers : ℕ
  buglePlayers : ℕ

/-- Represents the fixed costs for each group -/
structure FixedCosts where
  drummers : ℕ
  buglePlayers : ℕ

/-- Calculates the net earnings for a group given sales and fixed cost -/
def netEarnings (sales : ℕ) (cost : ℕ) : ℤ :=
  (sales : ℤ) - (sales * cost : ℤ)

/-- Calculates the percent difference between two integer values -/
def percentDifference (a b : ℤ) : ℚ :=
  if b ≠ 0 then (a - b : ℚ) / (b.natAbs : ℚ) * 100
  else if a > 0 then 100
  else if a < 0 then -100
  else 0

/-- Theorem stating that March has the greatest percent difference in net earnings -/
theorem march_greatest_percent_difference 
  (sales : Fin 5 → SalesData) 
  (costs : FixedCosts) 
  (h_jan : sales 0 = ⟨150, 100⟩)
  (h_feb : sales 1 = ⟨200, 150⟩)
  (h_mar : sales 2 = ⟨180, 180⟩)
  (h_apr : sales 3 = ⟨120, 160⟩)
  (h_may : sales 4 = ⟨80, 120⟩)
  (h_costs : costs = ⟨1, 2⟩) :
  ∀ (i : Fin 5), i ≠ 2 → 
    (abs (percentDifference 
      (netEarnings (sales 2).drummers costs.drummers)
      (netEarnings (sales 2).buglePlayers costs.buglePlayers)) ≥
     abs (percentDifference
      (netEarnings (sales i).drummers costs.drummers)
      (netEarnings (sales i).buglePlayers costs.buglePlayers))) :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_percent_difference_l782_78276


namespace NUMINAMATH_CALUDE_domain_of_g_l782_78266

-- Define the function f with domain (-1, 0)
def f : Set ℝ := {x : ℝ | -1 < x ∧ x < 0}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2 * x + 1) ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = {x : ℝ | -1 < x ∧ x < -1/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l782_78266


namespace NUMINAMATH_CALUDE_probability_bound_l782_78291

def is_divisible_by_four (n : ℕ) : Prop := n % 4 = 0

def count_even (n : ℕ) : ℕ := n / 2

def count_divisible_by_four (n : ℕ) : ℕ := n / 4

def probability_three_integers (n : ℕ) : ℚ :=
  let total := n.choose 3
  let favorable := (count_even n).choose 3 + (count_divisible_by_four n) * ((n - count_divisible_by_four n).choose 2)
  favorable / total

theorem probability_bound (n : ℕ) (h : n = 2017) :
  1/8 < probability_three_integers n ∧ probability_three_integers n < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_bound_l782_78291


namespace NUMINAMATH_CALUDE_malfunction_time_proof_l782_78239

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit changed by ±1 -/
def is_malfunction (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem malfunction_time_proof (displayed : Time) 
  (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunction original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_malfunction_time_proof_l782_78239


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_theorem_l782_78248

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ (p : ℝ), ∀ (x : ℝ), f x ≥ p ∧ ∃ (x₀ : ℝ), f x₀ = p :=
  sorry

-- Theorem for the inequality
theorem inequality_theorem (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  |a + 2*b + 3*c| ≤ 6 :=
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_theorem_l782_78248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l782_78241

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + 4

theorem arithmetic_sequence_100th_term (a : ℕ → ℕ) 
  (h : arithmetic_sequence a) : a 100 = 397 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l782_78241


namespace NUMINAMATH_CALUDE_tent_capacity_l782_78205

/-- The number of seating sections in the circus tent -/
def num_sections : ℕ := 4

/-- The number of people each section can accommodate -/
def people_per_section : ℕ := 246

/-- The total number of people the tent can accommodate -/
def total_capacity : ℕ := num_sections * people_per_section

theorem tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_tent_capacity_l782_78205


namespace NUMINAMATH_CALUDE_integer_solutions_l782_78210

def is_solution (x y z : ℤ) : Prop :=
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3

theorem integer_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = 4 ∧ y = 4 ∧ z = -5) ∨
     (x = 4 ∧ y = -5 ∧ z = 4) ∨
     (x = -5 ∧ y = 4 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_l782_78210


namespace NUMINAMATH_CALUDE_length_difference_l782_78253

/-- Represents a rectangular plot. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- The cost of fencing per meter. -/
def fencingCostPerMeter : ℝ := 26.50

/-- The total cost of fencing the plot. -/
def totalFencingCost : ℝ := 5300

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

theorem length_difference (plot : RectangularPlot) :
  plot.length = 57 →
  perimeter plot = totalFencingCost / fencingCostPerMeter →
  plot.length - plot.breadth = 14 := by
  sorry

end NUMINAMATH_CALUDE_length_difference_l782_78253


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l782_78255

theorem complex_arithmetic_result :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let T : ℂ := 2*I
  let U : ℂ := 1 + 5*I
  2*B - Q + 3*T + U = 10 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l782_78255


namespace NUMINAMATH_CALUDE_negation_equivalence_l782_78245

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define the property of being obtuse for an angle
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- Define the statement "at most one angle is obtuse"
def AtMostOneObtuse (t : Triangle) : Prop :=
  ∃ (a : Angle t), IsObtuse t a ∧ ∀ (b : Angle t), IsObtuse t b → b = a

-- Define the statement "at least two angles are obtuse"
def AtLeastTwoObtuse (t : Triangle) : Prop :=
  ∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b

-- The theorem stating the negation equivalence
theorem negation_equivalence (t : Triangle) :
  ¬(AtMostOneObtuse t) ↔ AtLeastTwoObtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l782_78245


namespace NUMINAMATH_CALUDE_min_value_3a_3b_l782_78216

theorem min_value_3a_3b (a b : ℝ) (h : a * b = 2) : 3 * a + 3 * b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_3b_l782_78216


namespace NUMINAMATH_CALUDE_card_area_theorem_l782_78200

/-- Represents the dimensions of a rectangular card -/
structure CardDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card given its dimensions -/
def cardArea (d : CardDimensions) : ℝ := d.length * d.width

/-- Theorem: If shortening one side of a 5x7 card by 2 inches results in an area of 15 square inches,
    then shortening the other side by 2 inches results in an area of 21 square inches -/
theorem card_area_theorem (original : CardDimensions) 
    (h1 : original.length = 5 ∧ original.width = 7)
    (h2 : ∃ (shortened : CardDimensions), 
      (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
      (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
      cardArea shortened = 15) :
  ∃ (other_shortened : CardDimensions),
    ((other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
     (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2)) ∧
    cardArea other_shortened = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l782_78200


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l782_78292

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, n = 980 ∧ 
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l782_78292


namespace NUMINAMATH_CALUDE_marla_errand_time_l782_78283

/-- Calculates the total time Marla spends on her errand to her son's school -/
def total_errand_time (one_way_drive_time parent_teacher_time : ℕ) : ℕ :=
  2 * one_way_drive_time + parent_teacher_time

/-- Proves that Marla spends 110 minutes on her errand -/
theorem marla_errand_time :
  total_errand_time 20 70 = 110 := by
  sorry

end NUMINAMATH_CALUDE_marla_errand_time_l782_78283


namespace NUMINAMATH_CALUDE_undefined_condition_l782_78260

theorem undefined_condition (y : ℝ) : 
  ¬(∃ x : ℝ, x = (3 * y^3 + 5) / (y^2 - 18*y + 81)) ↔ y = 9 := by
  sorry

end NUMINAMATH_CALUDE_undefined_condition_l782_78260


namespace NUMINAMATH_CALUDE_trivia_team_scoring_l782_78220

/-- Trivia team scoring problem -/
theorem trivia_team_scoring
  (total_members : ℕ)
  (absent_members : ℕ)
  (total_points : ℕ)
  (h1 : total_members = 5)
  (h2 : absent_members = 2)
  (h3 : total_points = 18)
  : (total_points / (total_members - absent_members) = 6) :=
by
  sorry

#check trivia_team_scoring

end NUMINAMATH_CALUDE_trivia_team_scoring_l782_78220


namespace NUMINAMATH_CALUDE_complex_subtraction_magnitude_l782_78232

theorem complex_subtraction_magnitude : 
  Complex.abs ((3 - 10 * Complex.I) - (2 + 5 * Complex.I)) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_magnitude_l782_78232


namespace NUMINAMATH_CALUDE_mailing_weight_calculation_l782_78269

/-- The total weight of a mailing with multiple envelopes and additional materials -/
def total_mailing_weight (envelope_weight : ℝ) (num_envelopes : ℕ) (additional_weight : ℝ) : ℝ :=
  (envelope_weight + additional_weight) * num_envelopes

/-- Theorem stating that the total weight of the mailing is 9240 grams -/
theorem mailing_weight_calculation :
  total_mailing_weight 8.5 880 2 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_mailing_weight_calculation_l782_78269


namespace NUMINAMATH_CALUDE_max_value_of_a_l782_78218

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) →
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l782_78218


namespace NUMINAMATH_CALUDE_min_difference_sine_extrema_l782_78211

open Real

theorem min_difference_sine_extrema (f : ℝ → ℝ) (h : ∀ x, f x = 2 * sin x) :
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂ ∧ |x₁ - x₂| = π) ∧
  (∀ x₁ x₂, (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) → |x₁ - x₂| ≥ π) :=
sorry

end NUMINAMATH_CALUDE_min_difference_sine_extrema_l782_78211


namespace NUMINAMATH_CALUDE_fraction_equality_l782_78252

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 8/10) :
  (6 * x^2 + 10 * y) / (60 * x * y) = 11/36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l782_78252


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l782_78289

theorem binomial_expansion_coefficient (x : ℝ) : 
  ∃ (c : ℕ), c = 45 ∧ 
  (∃ (terms : ℕ → ℝ), 
    (∀ r, terms r = (Nat.choose 10 r) * (-1)^r * x^(5 - 3*r/2)) ∧
    (∃ r, 5 - 3*r/2 = 2 ∧ terms r = c * x^2)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l782_78289


namespace NUMINAMATH_CALUDE_lcm_problem_l782_78225

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l782_78225


namespace NUMINAMATH_CALUDE_inequality_system_solution_l782_78242

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (4 + x) / 3 > (x + 2) / 2 ∧ (x + a) / 2 < 0 ↔ x < 2) →
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l782_78242


namespace NUMINAMATH_CALUDE_sequence_equation_l782_78214

theorem sequence_equation (n : ℕ+) : 9 * (n - 1) + n = (n - 1) * 10 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equation_l782_78214


namespace NUMINAMATH_CALUDE_betty_has_winning_strategy_l782_78281

/-- Represents the state of a bowl -/
structure BowlState :=
  (redBalls : Nat)
  (blueBalls : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (blueBowl : BowlState)
  (redBowl : BowlState)

/-- Enum for the possible moves in the game -/
inductive Move
  | TakeRedFromBlue
  | TakeBlueFromRed
  | ThrowAway

/-- Enum for the players -/
inductive Player
  | Albert
  | Betty

/-- Function to check if a game state is winning for the current player -/
def isWinningState (state : GameState) : Bool :=
  state.blueBowl.redBalls = 0 || state.redBowl.blueBalls = 0

/-- Function to get the next player -/
def nextPlayer (player : Player) : Player :=
  match player with
  | Player.Albert => Player.Betty
  | Player.Betty => Player.Albert

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeRedFromBlue => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 2, blueBalls := state.blueBowl.blueBalls },
        redBowl := { redBalls := state.redBowl.redBalls + 2, blueBalls := state.redBowl.blueBalls } }
  | Move.TakeBlueFromRed => 
      { blueBowl := { redBalls := state.blueBowl.redBalls, blueBalls := state.blueBowl.blueBalls + 2 },
        redBowl := { redBalls := state.redBowl.redBalls, blueBalls := state.redBowl.blueBalls - 2 } }
  | Move.ThrowAway => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 1, blueBalls := state.blueBowl.blueBalls - 1 },
        redBowl := state.redBowl }

/-- The initial state of the game -/
def initialState : GameState :=
  { blueBowl := { redBalls := 100, blueBalls := 0 },
    redBowl := { redBalls := 0, blueBalls := 100 } }

/-- Theorem stating that Betty has a winning strategy -/
theorem betty_has_winning_strategy :
  ∃ (strategy : Player → GameState → Move),
    ∀ (game : Nat → GameState),
      game 0 = initialState →
      (∀ n, game (n + 1) = applyMove (game n) (strategy (if n % 2 = 0 then Player.Albert else Player.Betty) (game n))) →
      ∃ n, isWinningState (game n) ∧ n % 2 = 1 :=
sorry


end NUMINAMATH_CALUDE_betty_has_winning_strategy_l782_78281


namespace NUMINAMATH_CALUDE_mean_practice_hours_l782_78265

def practice_hours : List ℕ := [1, 2, 3, 4, 5, 8, 10]
def student_counts : List ℕ := [4, 5, 3, 7, 2, 3, 1]

def total_hours : ℕ := (List.zip practice_hours student_counts).map (fun (h, c) => h * c) |>.sum
def total_students : ℕ := student_counts.sum

theorem mean_practice_hours :
  (total_hours : ℚ) / (total_students : ℚ) = 95 / 25 := by sorry

#eval (95 : ℚ) / 25  -- This should evaluate to 3.8

end NUMINAMATH_CALUDE_mean_practice_hours_l782_78265


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l782_78251

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l782_78251


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l782_78274

theorem quadratic_roots_expression (r s : ℝ) : 
  (3 * r^2 - 5 * r - 8 = 0) → 
  (3 * s^2 - 5 * s - 8 = 0) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l782_78274


namespace NUMINAMATH_CALUDE_circle_points_equidistant_l782_78236

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def IsOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_points_equidistant (c : Circle) (p : ℝ × ℝ) :
  IsOnCircle c p → (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_equidistant_l782_78236


namespace NUMINAMATH_CALUDE_transportation_cost_independent_of_order_l782_78207

/-- Represents a destination with its distance from the city and the weight of goods to be delivered -/
structure Destination where
  distance : ℝ
  weight : ℝ
  weight_eq_distance : weight = distance

/-- Calculates the cost of transportation for a single trip -/
def transportCost (d : Destination) (extraDistance : ℝ) : ℝ :=
  d.weight * (d.distance + extraDistance)

/-- Theorem stating that the total transportation cost is independent of the order of visits -/
theorem transportation_cost_independent_of_order (m n : Destination) :
  transportCost m 0 + transportCost n m.distance =
  transportCost n 0 + transportCost m n.distance := by
  sorry

#check transportation_cost_independent_of_order

end NUMINAMATH_CALUDE_transportation_cost_independent_of_order_l782_78207


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l782_78287

/-- Given that Bryan has 50 candies and Ben has 20 candies, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies :
  let bryan_candies : ℕ := 50
  let ben_candies : ℕ := 20
  bryan_candies - ben_candies = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l782_78287


namespace NUMINAMATH_CALUDE_remaining_payment_l782_78264

/-- Given a product with a 5% deposit of $50, prove that the remaining amount to be paid is $950 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 50 ∧ 
  deposit_percentage = 0.05 ∧ 
  deposit = deposit_percentage * total_price → 
  total_price - deposit = 950 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_l782_78264


namespace NUMINAMATH_CALUDE_quadratic_roots_contradiction_l782_78221

theorem quadratic_roots_contradiction (a : ℝ) : 
  a ≥ 1 → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_contradiction_l782_78221


namespace NUMINAMATH_CALUDE_average_of_subset_l782_78273

theorem average_of_subset (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 63 →
  (list.get! 2 + list.get! 3) / 2 = 60 →
  (list.get! 0 + list.get! 1 + list.get! 4 + list.get! 5 + list.get! 6) / 5 = 64.2 := by
sorry

end NUMINAMATH_CALUDE_average_of_subset_l782_78273


namespace NUMINAMATH_CALUDE_dartboard_angle_l782_78234

theorem dartboard_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l782_78234


namespace NUMINAMATH_CALUDE_complex_product_real_l782_78257

theorem complex_product_real (b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ (r : ℝ), (2 + Complex.I) * (b + Complex.I) = r) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l782_78257


namespace NUMINAMATH_CALUDE_initial_speed_is_4_l782_78223

/-- Represents the scenario of a person walking to a bus stand -/
structure BusScenario where
  distance : ℝ  -- Distance to the bus stand in km
  faster_speed : ℝ  -- Speed at which the person arrives early (km/h)
  early_time : ℝ  -- Time arrived early when walking at faster_speed (minutes)
  late_time : ℝ  -- Time arrived late when walking at initial speed (minutes)

/-- Calculates the initial walking speed given a BusScenario -/
def initial_speed (scenario : BusScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial walking speed is 4 km/h for the given scenario -/
theorem initial_speed_is_4 (scenario : BusScenario) 
  (h1 : scenario.distance = 5)
  (h2 : scenario.faster_speed = 5)
  (h3 : scenario.early_time = 5)
  (h4 : scenario.late_time = 10) :
  initial_speed scenario = 4 :=
sorry

end NUMINAMATH_CALUDE_initial_speed_is_4_l782_78223


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l782_78256

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l782_78256


namespace NUMINAMATH_CALUDE_prime_pair_fraction_integer_l782_78206

theorem prime_pair_fraction_integer :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    (∃ n : ℤ, (((p + q : ℕ)^(p + q) * (p - q : ℕ)^(p - q) - 1) : ℤ) = 
              n * (((p + q : ℕ)^(p - q) * (p - q : ℕ)^(p + q) - 1) : ℤ)) →
    p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_pair_fraction_integer_l782_78206


namespace NUMINAMATH_CALUDE_sin_transformation_equivalence_l782_78235

theorem sin_transformation_equivalence (x : ℝ) :
  let f (x : ℝ) := Real.sin x
  let g (x : ℝ) := Real.sin (2*x - π/5)
  let transform1 (x : ℝ) := Real.sin (2*(x - π/5))
  let transform2 (x : ℝ) := Real.sin (2*(x - π/10))
  (∀ x, g x = transform1 x) ∧ (∀ x, g x = transform2 x) :=
by sorry

end NUMINAMATH_CALUDE_sin_transformation_equivalence_l782_78235


namespace NUMINAMATH_CALUDE_expected_population_after_increase_l782_78263

def current_population : ℝ := 1.75
def percentage_increase : ℝ := 325

theorem expected_population_after_increase :
  let increase_factor := 1 + percentage_increase / 100
  let expected_population := current_population * increase_factor
  expected_population = 7.4375 := by sorry

end NUMINAMATH_CALUDE_expected_population_after_increase_l782_78263


namespace NUMINAMATH_CALUDE_plate_tower_problem_l782_78296

theorem plate_tower_problem (initial_plates : ℕ) (first_addition : ℕ) (common_difference : ℕ) (total_plates : ℕ) :
  initial_plates = 27 →
  first_addition = 12 →
  common_difference = 3 →
  total_plates = 123 →
  ∃ (n : ℕ) (last_addition : ℕ),
    n = 4 ∧
    last_addition = 21 ∧
    total_plates = initial_plates + n * (2 * first_addition + (n - 1) * common_difference) / 2 :=
by sorry

end NUMINAMATH_CALUDE_plate_tower_problem_l782_78296


namespace NUMINAMATH_CALUDE_range_of_x_when_f_positive_l782_78230

/-- A linear function obtained by translating y = x upwards by 2 units -/
def f (x : ℝ) : ℝ := x + 2

/-- The range of x when f(x) > 0 -/
theorem range_of_x_when_f_positive : 
  {x : ℝ | f x > 0} = {x : ℝ | x > -2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_f_positive_l782_78230


namespace NUMINAMATH_CALUDE_seeking_cause_is_necessary_condition_l782_78212

/-- "Seeking the cause from the effect" in analytical proof -/
def seeking_cause_from_effect : Prop := sorry

/-- Necessary condition in a proposition -/
def necessary_condition : Prop := sorry

/-- Theorem stating that "seeking the cause from the effect" refers to seeking the necessary condition -/
theorem seeking_cause_is_necessary_condition : 
  seeking_cause_from_effect ↔ necessary_condition := by sorry

end NUMINAMATH_CALUDE_seeking_cause_is_necessary_condition_l782_78212


namespace NUMINAMATH_CALUDE_permutations_of_repeated_letters_l782_78275

def phrase : String := "mathstest"

def repeated_letters (s : String) : List Char :=
  s.toList.filter (fun c => s.toList.count c > 1)

def unique_permutations (letters : List Char) : ℕ :=
  Nat.factorial letters.length / (Nat.factorial (letters.count 's') * Nat.factorial (letters.count 't'))

theorem permutations_of_repeated_letters :
  unique_permutations (repeated_letters phrase) = 10 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_repeated_letters_l782_78275


namespace NUMINAMATH_CALUDE_sum_of_squares_l782_78277

open BigOperators

/-- Given a sequence {aₙ} where the sum of the first n terms is 3ⁿ - 1,
    prove that the sum of squares of the first n terms is (1/2)(9ⁿ - 1) -/
theorem sum_of_squares (a : ℕ → ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ∑ i in Finset.range k, a i = 3^k - 1) →
  ∑ i in Finset.range n, (a i)^2 = (1/2) * (9^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l782_78277


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l782_78224

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l782_78224


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l782_78222

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l782_78222


namespace NUMINAMATH_CALUDE_q_at_zero_l782_78229

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between p, q, and r
axiom poly_product : r = p * q

-- Define the constant terms of p and r
axiom p_constant : p.coeff 0 = 5
axiom r_constant : r.coeff 0 = -10

-- Theorem to prove
theorem q_at_zero : q.eval 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_q_at_zero_l782_78229


namespace NUMINAMATH_CALUDE_composite_and_three_factors_l782_78295

theorem composite_and_three_factors (n : ℕ) (h : n > 10) :
  let N := n^4 - 90*n^2 - 91*n - 90
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N = x * y * z) :=
by sorry

end NUMINAMATH_CALUDE_composite_and_three_factors_l782_78295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l782_78270

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the solution set of the quadratic inequality
def solution_set (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | x^2 - a 3 * x + a 4 ≤ 0}

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  solution_set a = {x : ℝ | a 1 ≤ x ∧ x ≤ a 2} →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l782_78270


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l782_78246

/-- Linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the value of f at x = -5 -/
def y₁ : ℝ := f (-5)

/-- y₂ is the value of f at x = 3 -/
def y₂ : ℝ := f 3

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l782_78246


namespace NUMINAMATH_CALUDE_square_sum_bound_l782_78298

theorem square_sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_bound_l782_78298


namespace NUMINAMATH_CALUDE_candy_soda_price_before_increase_l782_78249

theorem candy_soda_price_before_increase 
  (candy_price_after : ℝ) 
  (soda_price_after : ℝ) 
  (candy_increase_rate : ℝ) 
  (soda_increase_rate : ℝ) 
  (h1 : candy_price_after = 15) 
  (h2 : soda_price_after = 6) 
  (h3 : candy_increase_rate = 0.25) 
  (h4 : soda_increase_rate = 0.5) : 
  candy_price_after / (1 + candy_increase_rate) + 
  soda_price_after / (1 + soda_increase_rate) = 21 := by
  sorry

#check candy_soda_price_before_increase

end NUMINAMATH_CALUDE_candy_soda_price_before_increase_l782_78249


namespace NUMINAMATH_CALUDE_farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l782_78258

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost 
  (original_bales : ℕ) 
  (multiplier : ℕ) 
  (original_cost_per_bale : ℕ) 
  (premium_cost_per_bale : ℕ) : ℕ :=
  let new_bales := original_bales * multiplier
  let original_total_cost := original_bales * original_cost_per_bale
  let new_total_cost := new_bales * premium_cost_per_bale
  new_total_cost - original_total_cost

/-- The additional cost for Farmer Brown's specific hay requirements is $3500 -/
theorem farmer_brown_specific_additional_cost :
  farmer_brown_additional_cost 20 5 25 40 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l782_78258


namespace NUMINAMATH_CALUDE_expression_evaluation_l782_78297

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the given expression in base 10 -/
def result : ℚ :=
  (toBase10 [4, 5, 2] 8 : ℚ) / (toBase10 [3, 1] 3) +
  (toBase10 [3, 0, 2] 5 : ℚ) / (toBase10 [2, 2] 4)

theorem expression_evaluation :
  result = 33.966666666666665 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l782_78297


namespace NUMINAMATH_CALUDE_nominations_distribution_l782_78237

/-- The number of ways to distribute nominations among schools -/
def distribute_nominations (total_nominations : ℕ) (num_schools : ℕ) : ℕ :=
  Nat.choose (total_nominations - num_schools + num_schools - 1) (num_schools - 1)

/-- Theorem stating that there are 84 ways to distribute 10 nominations among 7 schools -/
theorem nominations_distribution :
  distribute_nominations 10 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_nominations_distribution_l782_78237


namespace NUMINAMATH_CALUDE_base8_to_base10_157_l782_78203

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 5, 1]

theorem base8_to_base10_157 :
  base8ToBase10 base8Number = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_157_l782_78203


namespace NUMINAMATH_CALUDE_jacksons_vacation_months_l782_78202

/-- Proves that Jackson's vacation is 15 months away given his saving plan -/
theorem jacksons_vacation_months (total_savings : ℝ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ)
  (h1 : total_savings = 3000)
  (h2 : paychecks_per_month = 2)
  (h3 : savings_per_paycheck = 100) :
  (total_savings / savings_per_paycheck) / paychecks_per_month = 15 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_vacation_months_l782_78202


namespace NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l782_78279

theorem shortest_side_of_special_triangle : 
  ∀ (a b c : ℕ), 
    a = 18 →
    a + b + c = 42 →
    (∃ A : ℕ, A^2 = (21 * (21 - a) * (21 - b) * (21 - c))) →
    a ≤ b ∧ a ≤ c →
    b + c > a →
    a + c > b →
    a + b > c →
    b ≥ 5 ∧ c ≥ 5 :=
by sorry

#check shortest_side_of_special_triangle

end NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l782_78279


namespace NUMINAMATH_CALUDE_james_tshirts_l782_78243

/-- Calculates the number of t-shirts bought given the discount rate, original price, and total amount paid -/
def tshirts_bought (discount_rate : ℚ) (original_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (original_price * (1 - discount_rate))

/-- Proves that James bought 6 t-shirts -/
theorem james_tshirts :
  let discount_rate : ℚ := 1/2
  let original_price : ℚ := 20
  let total_paid : ℚ := 60
  tshirts_bought discount_rate original_price total_paid = 6 := by
sorry

end NUMINAMATH_CALUDE_james_tshirts_l782_78243


namespace NUMINAMATH_CALUDE_square_difference_inapplicable_l782_78219

/-- The square difference formula cannot be directly applied to (x-y)(-x+y) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ) (c₁ c₂ c₃ c₄ : ℝ), 
    (a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    ((x - y) * (-x + y) = (a + b) * (a - b) ∨ (x - y) * (-x + y) = (a - b) * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_inapplicable_l782_78219


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l782_78209

theorem no_positive_integer_solutions :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
  ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l782_78209


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l782_78244

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 122 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeBalls 6 3 = 122 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l782_78244


namespace NUMINAMATH_CALUDE_vector_linear_combination_l782_78247

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = 2 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l782_78247


namespace NUMINAMATH_CALUDE_pear_sales_l782_78271

/-- Given a salesman who sold pears, prove that if he sold twice as much in the afternoon
    than in the morning, and 480 kilograms in total, then he sold 320 kilograms in the afternoon. -/
theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

#check pear_sales

end NUMINAMATH_CALUDE_pear_sales_l782_78271


namespace NUMINAMATH_CALUDE_G_equals_3F_l782_78293

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (3 * x + x^3) / (1 + 3 * x^2)) / (1 - (3 * x + x^3) / (1 + 3 * x^2)))

theorem G_equals_3F : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → G x = 3 * F x := by sorry

end NUMINAMATH_CALUDE_G_equals_3F_l782_78293


namespace NUMINAMATH_CALUDE_expression_simplification_l782_78204

theorem expression_simplification (b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  1/2 - 1/(1 + b/(1 - 2*b)) = (3*b - 1)/(2*(1 - b)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l782_78204


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l782_78259

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l782_78259


namespace NUMINAMATH_CALUDE_average_not_equal_given_l782_78278

def numbers : List ℝ := [1200, 1300, 1400, 1520, 1530, 1200]

def given_average : ℝ := 1380

theorem average_not_equal_given : (numbers.sum / numbers.length) ≠ given_average := by
  sorry

end NUMINAMATH_CALUDE_average_not_equal_given_l782_78278


namespace NUMINAMATH_CALUDE_whiteboard_numbers_l782_78238

theorem whiteboard_numbers (n k : ℕ) : 
  n > 0 ∧ k > 0 ∧ k ≤ n ∧ 
  Odd n ∧
  (((n * (n + 1)) / 2 - k) : ℚ) / (n - 1) = 22 →
  n = 43 ∧ k = 22 := by
  sorry

end NUMINAMATH_CALUDE_whiteboard_numbers_l782_78238


namespace NUMINAMATH_CALUDE_prob_at_least_four_same_l782_78280

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all five dice showing the same number
def prob_all_same : ℚ := 1 / die_sides^(num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def prob_four_same : ℚ := 
  (num_dice : ℚ) * (1 / die_sides^(num_dice - 2)) * ((die_sides - 1 : ℚ) / die_sides)

-- Theorem statement
theorem prob_at_least_four_same : 
  prob_all_same + prob_four_same = 13 / 648 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_four_same_l782_78280


namespace NUMINAMATH_CALUDE_average_weight_increase_l782_78227

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 4 →
  old_weight = 95 →
  new_weight = 129 →
  (new_weight - old_weight) / initial_count = 8.5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l782_78227


namespace NUMINAMATH_CALUDE_complex_equality_l782_78288

theorem complex_equality (a : ℝ) : 
  (Complex.re ((a - Complex.I) * (1 - Complex.I) * Complex.I) = 
   Complex.im ((a - Complex.I) * (1 - Complex.I) * Complex.I)) → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l782_78288


namespace NUMINAMATH_CALUDE_one_book_selection_ways_l782_78226

/-- The number of ways to take one book from a shelf with Chinese, math, and English books. -/
def ways_to_take_one_book (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books + math_books + english_books

/-- Theorem: There are 37 ways to take one book from a shelf with 12 Chinese books, 14 math books, and 11 English books. -/
theorem one_book_selection_ways :
  ways_to_take_one_book 12 14 11 = 37 := by
  sorry

end NUMINAMATH_CALUDE_one_book_selection_ways_l782_78226


namespace NUMINAMATH_CALUDE_average_physics_math_l782_78208

/-- Given the scores of three subjects, prove the average of two specific subjects -/
theorem average_physics_math (total_average : ℝ) (physics_chem_average : ℝ) (physics_score : ℝ) : 
  total_average = 60 →
  physics_chem_average = 70 →
  physics_score = 140 →
  (physics_score + (3 * total_average - physics_score - 
    (2 * physics_chem_average - physics_score))) / 2 = 90 := by
  sorry


end NUMINAMATH_CALUDE_average_physics_math_l782_78208


namespace NUMINAMATH_CALUDE_book_price_increase_l782_78231

/-- Given a book with an original price and a percentage increase, 
    calculate the new price after the increase. -/
theorem book_price_increase (original_price : ℝ) (percent_increase : ℝ) 
  (h1 : original_price = 300)
  (h2 : percent_increase = 10) : 
  original_price * (1 + percent_increase / 100) = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l782_78231


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l782_78285

-- Define a random variable X following N(0,1) distribution
def X : Real → Real := sorry

-- Define the probability measure for X
def P (s : Set Real) : Real := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h1 : ∀ s, P s = ∫ x in s, X x)
  (h2 : P {x | x ≤ 1} = 0.8413) :
  P {x | -1 < x ∧ x < 0} = 0.3413 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l782_78285


namespace NUMINAMATH_CALUDE_greatest_partition_size_l782_78215

theorem greatest_partition_size (m n p : ℕ) (h_m : m > 0) (h_n : n > 0) (h_p : Nat.Prime p) :
  ∃ (s : ℕ), s > 0 ∧ s ≤ m ∧
  ∀ (t : ℕ), t > s →
    ¬∃ (partition : Fin (t * n * p) → Fin t),
      ∀ (i : Fin t),
        ∃ (r : ℕ),
          ∀ (j k : Fin (t * n * p)),
            partition j = i → partition k = i →
              (j.val + k.val) % p = r :=
by sorry

end NUMINAMATH_CALUDE_greatest_partition_size_l782_78215


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l782_78282

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧ 
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l782_78282


namespace NUMINAMATH_CALUDE_intersection_point_a_l782_78261

-- Define the function f
def f (b : ℤ) (x : ℝ) : ℝ := 2 * x + b

-- Define the theorem
theorem intersection_point_a (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = a ∧ f b (-4) = a) →  -- f and f^(-1) intersect at (-4, a)
  a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_l782_78261


namespace NUMINAMATH_CALUDE_ben_bought_three_cards_l782_78267

/-- The number of cards Ben bought -/
def cards_bought : ℕ := 3

/-- The number of cards Tim had -/
def tim_cards : ℕ := 20

/-- The number of cards Ben initially had -/
def ben_initial_cards : ℕ := 37

theorem ben_bought_three_cards :
  (ben_initial_cards + cards_bought = 2 * tim_cards) ∧
  (cards_bought = 3) := by
  sorry

end NUMINAMATH_CALUDE_ben_bought_three_cards_l782_78267


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_eight_l782_78290

/-- A function that is symmetric about x = 2 and has exactly four distinct zeros -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∃! (z₁ z₂ z₃ z₄ : ℝ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0)

/-- The theorem stating that the sum of zeros for a symmetric function with four distinct zeros is 8 -/
theorem sum_of_zeros_is_eight (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃ z₁ z₂ z₃ z₄ : ℝ, z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0 ∧
    z₁ + z₂ + z₃ + z₄ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_eight_l782_78290


namespace NUMINAMATH_CALUDE_park_expansion_area_ratio_l782_78284

theorem park_expansion_area_ratio :
  ∀ s : ℝ, s > 0 →
  (s^2) / ((3*s)^2) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_park_expansion_area_ratio_l782_78284


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l782_78250

/-- The ratio of the surface area of a cube to the surface area of a regular tetrahedron
    formed by four vertices of the cube, given that the cube has side length 2. -/
theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l782_78250


namespace NUMINAMATH_CALUDE_undefined_values_l782_78299

theorem undefined_values (a : ℝ) : 
  (a + 2) / (a^2 - 9) = 0/0 ↔ a = -3 ∨ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_undefined_values_l782_78299


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_five_l782_78254

theorem quadratic_sum_equals_five (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_five_l782_78254


namespace NUMINAMATH_CALUDE_hoseok_wire_length_l782_78268

/-- The length of wire Hoseok bought, given the conditions of the problem -/
def wire_length (triangle_side_length : ℝ) (remaining_wire : ℝ) : ℝ :=
  3 * triangle_side_length + remaining_wire

/-- Theorem stating that the length of wire Hoseok bought is 72 cm -/
theorem hoseok_wire_length :
  wire_length 19 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_wire_length_l782_78268


namespace NUMINAMATH_CALUDE_no_solution_condition_l782_78201

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (k * x) / (x - 1) - (2 * k - 1) / (1 - x) ≠ 2) ↔ 
  (k = 1/3 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l782_78201


namespace NUMINAMATH_CALUDE_sum_of_roots_l782_78240

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : 8*b^3 - 48*b^2 + 18*b + 162 = 0) : 
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l782_78240


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l782_78262

/-- If 10 packs of DVDs cost 110 dollars, then the cost of one pack is 11 dollars -/
theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 110) (h2 : num_packs = 10) :
  total_cost / num_packs = 11 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l782_78262


namespace NUMINAMATH_CALUDE_parabola_directrix_l782_78294

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y^2 = -12*x

/-- The equation of the directrix -/
def directrix_eq (x : ℝ) : Prop := x = 3

/-- Theorem: The directrix of the parabola y^2 = -12x is x = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → directrix_eq x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l782_78294


namespace NUMINAMATH_CALUDE_area_ratio_l782_78217

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (30, 50, 54)

-- Define points D and E
def point_D (t : Triangle) : ℝ × ℝ := sorry
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the distances AD and AE
def dist_AD (t : Triangle) : ℝ := 21
def dist_AE (t : Triangle) : ℝ := 18

-- Define the areas of triangle ADE and quadrilateral BCED
def area_ADE (t : Triangle) : ℝ := sorry
def area_BCED (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio (t : Triangle) :
  area_ADE t / area_BCED t = 49 / 51 := by sorry

end NUMINAMATH_CALUDE_area_ratio_l782_78217


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l782_78286

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 9, 10} : Set ℤ) →
  (3 * b^3 - 2 * b^2 - b - 2) % 5 ≠ 0 ↔ b = 5 ∨ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l782_78286


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l782_78233

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.43 → p_white = 0.27 → p_red + p_white + p_black = 1 → p_black = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l782_78233
