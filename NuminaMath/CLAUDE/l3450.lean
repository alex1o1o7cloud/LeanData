import Mathlib

namespace NUMINAMATH_CALUDE_outside_door_cost_l3450_345016

/-- Proves that the cost of each outside door is $20 -/
theorem outside_door_cost (bedroom_doors : ℕ) (outside_doors : ℕ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  total_cost = 70 →
  ∃ (outside_door_cost : ℚ),
    outside_door_cost * outside_doors + (outside_door_cost / 2) * bedroom_doors = total_cost ∧
    outside_door_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_outside_door_cost_l3450_345016


namespace NUMINAMATH_CALUDE_fraction_addition_subtraction_l3450_345095

theorem fraction_addition_subtraction :
  (1 / 4 : ℚ) + (3 / 8 : ℚ) - (1 / 8 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_subtraction_l3450_345095


namespace NUMINAMATH_CALUDE_equation_solution_pairs_l3450_345070

theorem equation_solution_pairs : 
  {(p, q) : ℕ × ℕ | (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q} = {(1, 1), (2, 2)} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_pairs_l3450_345070


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3450_345091

theorem lcm_hcf_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 330) : 
  b = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3450_345091


namespace NUMINAMATH_CALUDE_arcsin_sin_equation_solutions_l3450_345074

theorem arcsin_sin_equation_solutions :
  let S := {x : ℝ | -3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2 ∧ Real.arcsin (Real.sin x) = x / 3}
  S = {-3 * Real.pi, -Real.pi, 0, Real.pi, 3 * Real.pi} := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_equation_solutions_l3450_345074


namespace NUMINAMATH_CALUDE_regular_27gon_trapezoid_l3450_345097

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

/-- Main theorem: Among any 7 vertices of a regular 27-gon, 4 can be selected that form a trapezoid -/
theorem regular_27gon_trapezoid (P : RegularPolygon 27) 
  (S : Finset (Fin 27)) (hS : S.card = 7) : 
  ∃ (a b c d : Fin 27), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    IsTrapezoid (P.vertices a) (P.vertices b) (P.vertices c) (P.vertices d) :=
sorry

end NUMINAMATH_CALUDE_regular_27gon_trapezoid_l3450_345097


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3450_345027

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → (slope_of_perpendicular_line = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3450_345027


namespace NUMINAMATH_CALUDE_tangent_circle_exists_l3450_345081

-- Define the types for points and circles
def Point := ℝ × ℝ
def Circle := Point × ℝ  -- Center and radius

-- Define a function to check if a point is on a circle
def is_on_circle (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define a function to check if two circles are tangent
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (center1, radius1) := c1
  let (center2, radius2) := c2
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

-- Theorem statement
theorem tangent_circle_exists (c1 c2 : Circle) (T : Point) 
  (h1 : is_on_circle T c1) : 
  ∃ (c : Circle), are_circles_tangent c c1 ∧ are_circles_tangent c c2 ∧ is_on_circle T c :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_exists_l3450_345081


namespace NUMINAMATH_CALUDE_ornamental_bangles_pairs_l3450_345052

/-- The number of bangles in a dozen -/
def bangles_per_dozen : ℕ := 12

/-- The number of dozens in a box -/
def dozens_per_box : ℕ := 2

/-- The number of boxes needed -/
def num_boxes : ℕ := 20

/-- The number of bangles in a pair -/
def bangles_per_pair : ℕ := 2

theorem ornamental_bangles_pairs :
  (num_boxes * dozens_per_box * bangles_per_dozen) / bangles_per_pair = 240 := by
  sorry

end NUMINAMATH_CALUDE_ornamental_bangles_pairs_l3450_345052


namespace NUMINAMATH_CALUDE_probability_both_win_is_one_third_l3450_345079

/-- Represents the three types of lottery tickets -/
inductive Ticket
  | FirstPrize
  | SecondPrize
  | NonPrize

/-- Represents a draw of two tickets without replacement -/
def Draw := (Ticket × Ticket)

/-- The set of all possible draws -/
def allDraws : Finset Draw := sorry

/-- Predicate to check if a draw results in both people winning a prize -/
def bothWinPrize (draw : Draw) : Prop := 
  draw.1 ≠ Ticket.NonPrize ∧ draw.2 ≠ Ticket.NonPrize

/-- The set of draws where both people win a prize -/
def winningDraws : Finset Draw := sorry

/-- The probability of both people winning a prize -/
def probabilityBothWin : ℚ := (winningDraws.card : ℚ) / (allDraws.card : ℚ)

theorem probability_both_win_is_one_third : 
  probabilityBothWin = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_both_win_is_one_third_l3450_345079


namespace NUMINAMATH_CALUDE_is_circle_center_l3450_345082

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l3450_345082


namespace NUMINAMATH_CALUDE_alyssa_chicken_nuggets_l3450_345069

/-- Given 100 total chicken nuggets and two people eating twice as much as Alyssa,
    prove that Alyssa ate 20 chicken nuggets. -/
theorem alyssa_chicken_nuggets :
  ∀ (total : ℕ) (alyssa : ℕ),
    total = 100 →
    total = alyssa + 2 * alyssa + 2 * alyssa →
    alyssa = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_alyssa_chicken_nuggets_l3450_345069


namespace NUMINAMATH_CALUDE_weaving_increase_l3450_345084

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℚ) : ℚ := n * a + n * (n - 1) / 2 * d

/-- The problem of finding the daily increase in weaving -/
theorem weaving_increase (a₁ : ℚ) (n : ℕ) (S : ℚ) (h1 : a₁ = 5) (h2 : n = 30) (h3 : S = 390) :
  ∃ d : ℚ, arithmetic_sum n a₁ d = S ∧ d = 16/29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_increase_l3450_345084


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3450_345003

theorem chocolate_distribution (num_students : ℕ) (num_choices : ℕ) 
  (h1 : num_students = 211) (h2 : num_choices = 35) : 
  ∃ (group_size : ℕ), group_size ≥ 7 ∧ 
  (∀ (group : ℕ), group ≤ group_size) ∧ 
  (num_students ≤ group_size * num_choices) :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3450_345003


namespace NUMINAMATH_CALUDE_square_root_calculations_l3450_345083

theorem square_root_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  (Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l3450_345083


namespace NUMINAMATH_CALUDE_no_divisor_of_form_24k_plus_20_l3450_345023

theorem no_divisor_of_form_24k_plus_20 (n : ℕ) : ¬ ∃ (k : ℕ), (24 * k + 20) ∣ (3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisor_of_form_24k_plus_20_l3450_345023


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l3450_345090

theorem quadratic_roots_distinct (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l3450_345090


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l3450_345020

/-- Proves that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 12,
    then the principal amount is 1200. -/
theorem interest_difference_implies_principal
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Interest rate (as a decimal)
  (t : ℝ)  -- Time period in years
  (h1 : r = 0.1)  -- Interest rate is 10%
  (h2 : t = 2)    -- Time period is 2 years
  (h3 : P * (1 + r)^t - P - (P * r * t) = 12)  -- Difference between CI and SI is 12
  : P = 1200 :=
by sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l3450_345020


namespace NUMINAMATH_CALUDE_right_triangle_equality_l3450_345031

/-- For a right triangle with sides a and b, and hypotenuse c, 
    the equation √(a^2 + b^2) = a + b is true if and only if 
    the angle θ between sides a and b is 90°. -/
theorem right_triangle_equality (a b c : ℝ) (θ : Real) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) -- Pythagorean theorem
  (h5 : θ = Real.arccos (b / c)) -- Definition of θ
  : Real.sqrt (a^2 + b^2) = a + b ↔ θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l3450_345031


namespace NUMINAMATH_CALUDE_workers_contribution_problem_l3450_345059

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The original total contribution in paise (100 paise = 1 rupee) -/
def original_total : ℕ := 30000000  -- 3 lacs = 300,000 rupees = 30,000,000 paise

/-- The new total contribution if each worker contributed 50 rupees extra, in paise -/
def new_total : ℕ := 36000000  -- 3.60 lacs = 360,000 rupees = 36,000,000 paise

/-- The extra contribution per worker in paise -/
def extra_contribution : ℕ := 5000  -- 50 rupees = 5,000 paise

theorem workers_contribution_problem :
  (original_total / number_of_workers : ℚ) * number_of_workers = original_total ∧
  ((original_total / number_of_workers : ℚ) + extra_contribution) * number_of_workers = new_total :=
sorry

end NUMINAMATH_CALUDE_workers_contribution_problem_l3450_345059


namespace NUMINAMATH_CALUDE_mets_fan_count_l3450_345013

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of fans in the town -/
def total_fans : ℕ := 330

/-- The fan count satisfies the given ratios and total -/
def is_valid_fan_count (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees ∧
  4 * fc.red_sox = 5 * fc.mets ∧
  fc.yankees + fc.mets + fc.red_sox = total_fans

theorem mets_fan_count (fc : FanCount) (h : is_valid_fan_count fc) : fc.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fan_count_l3450_345013


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l3450_345049

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l3450_345049


namespace NUMINAMATH_CALUDE_product_of_exponents_l3450_345078

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^3 = 36 → 2^r + 18 = 50 → 5^s + 7^2 = 1914 → p * r * s = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l3450_345078


namespace NUMINAMATH_CALUDE_escalator_steps_l3450_345033

/-- The number of steps counted by the slower person -/
def walker_count : ℕ := 50

/-- The number of steps counted by the faster person -/
def trotman_count : ℕ := 75

/-- The speed ratio between the faster and slower person -/
def speed_ratio : ℕ := 3

/-- The number of visible steps on the stopped escalator -/
def visible_steps : ℕ := 100

/-- Theorem stating that the number of visible steps on the stopped escalator is 100 -/
theorem escalator_steps :
  ∀ (v : ℚ), v > 0 →
  walker_count + walker_count / v = trotman_count + trotman_count / (speed_ratio * v) →
  visible_steps = walker_count + walker_count / v :=
by sorry

end NUMINAMATH_CALUDE_escalator_steps_l3450_345033


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3450_345063

/-- Function to calculate the sum of squares of decimal digits -/
def sumSquaresDecimal (n : Nat) : Nat :=
  sorry

/-- Function to calculate the sum of squares of binary digits -/
def sumSquaresBinary (n : Nat) : Nat :=
  sorry

/-- Check if a number has equal costs for both options -/
def hasEqualCosts (n : Nat) : Prop :=
  sumSquaresDecimal n = sumSquaresBinary n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 5000 → n > 3999 → ¬(hasEqualCosts n) :=
by sorry

theorem equal_cost_3999 : hasEqualCosts 3999 :=
by sorry

theorem largest_equal_cost_is_3999 :
  ∃ n : Nat, n < 5000 ∧ hasEqualCosts n ∧ ∀ m : Nat, m < 5000 → m > n → ¬(hasEqualCosts m) :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3450_345063


namespace NUMINAMATH_CALUDE_xyz_value_l3450_345046

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x * y + x * z + y * z = 7)
  (sum : x + y + z = 4) :
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3450_345046


namespace NUMINAMATH_CALUDE_black_pens_count_l3450_345011

theorem black_pens_count (total_pens : ℕ) (red_pens black_pens : ℕ) : 
  (3 : ℚ) / 10 * total_pens = red_pens →
  (1 : ℚ) / 5 * total_pens = black_pens →
  red_pens = 12 →
  black_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_black_pens_count_l3450_345011


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l3450_345093

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101(2) -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110(2) -/
def binary2 : List Bool := [false, true, true]

/-- Statement: The sum of the decimal representations of 101(2) and 110(2) is 11 -/
theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l3450_345093


namespace NUMINAMATH_CALUDE_range_of_a_l3450_345017

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) →
  a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3450_345017


namespace NUMINAMATH_CALUDE_sum_of_three_cubes_not_2002_l3450_345035

theorem sum_of_three_cubes_not_2002 : ¬∃ (a b c : ℕ+), a.val^3 + b.val^3 + c.val^3 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_cubes_not_2002_l3450_345035


namespace NUMINAMATH_CALUDE_complex_sum_equality_l3450_345040

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 3*I
  let T : ℂ := 1 + 5*I
  B - Q + R + T = -1 + 10*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l3450_345040


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3450_345077

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3450_345077


namespace NUMINAMATH_CALUDE_participation_plans_count_l3450_345032

/-- The number of students to choose from, excluding the pre-selected student -/
def n : ℕ := 3

/-- The number of students to be selected, excluding the pre-selected student -/
def k : ℕ := 2

/-- The total number of students participating (including the pre-selected student) -/
def total_participants : ℕ := k + 1

/-- The number of subjects -/
def subjects : ℕ := 3

theorem participation_plans_count : 
  (n.choose k) * (Nat.factorial total_participants) = 18 := by
  sorry

end NUMINAMATH_CALUDE_participation_plans_count_l3450_345032


namespace NUMINAMATH_CALUDE_prize_money_calculation_l3450_345029

theorem prize_money_calculation (total : ℚ) (rica_share : ℚ) (rica_spent : ℚ) (rica_left : ℚ) : 
  rica_share = 3 / 8 * total →
  rica_spent = 1 / 5 * rica_share →
  rica_left = rica_share - rica_spent →
  rica_left = 300 →
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_prize_money_calculation_l3450_345029


namespace NUMINAMATH_CALUDE_inequality_proof_l3450_345038

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  |b / a - b / c| + |c / a - c / b| + |b * c + 1| > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3450_345038


namespace NUMINAMATH_CALUDE_mean_temperature_l3450_345008

def temperatures : List ℝ := [78, 76, 80, 83, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 80.4 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3450_345008


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l3450_345080

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x - 5) → 
  a + b + c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l3450_345080


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_parameter_range_l3450_345014

theorem increasing_quadratic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = x^2 - 2*a*x + 2) 
  (h2 : ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f x < f y) : 
  a ∈ Set.Iic 3 := by
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_parameter_range_l3450_345014


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3450_345051

/-- The problem of Jerry's action figures -/
theorem jerrys_action_figures 
  (total : ℕ) -- Total number of action figures after adding
  (added : ℕ) -- Number of added action figures
  (h1 : total = 10) -- Given: The total number of action figures after adding is 10
  (h2 : added = 6) -- Given: The number of added action figures is 6
  : total - added = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3450_345051


namespace NUMINAMATH_CALUDE_ali_baba_walk_possible_l3450_345001

/-- Represents a cell in the cave -/
structure Cell where
  row : Nat
  col : Nat
  isBlack : Bool

/-- Represents the state of the cave -/
structure CaveState where
  m : Nat
  n : Nat
  coins : Cell → Nat

/-- Represents a move in the cave -/
inductive Move
  | up
  | down
  | left
  | right

/-- Predicate to check if a move is valid -/
def isValidMove (state : CaveState) (pos : Cell) (move : Move) : Prop :=
  match move with
  | Move.up => pos.row > 0
  | Move.down => pos.row < state.m - 1
  | Move.left => pos.col > 0
  | Move.right => pos.col < state.n - 1

/-- Function to apply a move and update the cave state -/
def applyMove (state : CaveState) (pos : Cell) (move : Move) : CaveState :=
  sorry

/-- Predicate to check if the final state is correct -/
def isCorrectFinalState (state : CaveState) : Prop :=
  ∀ cell, (cell.isBlack → state.coins cell = 1) ∧ (¬cell.isBlack → state.coins cell = 0)

/-- Theorem stating that Ali Baba's walk is possible -/
theorem ali_baba_walk_possible (m n : Nat) :
  ∃ (initialState : CaveState) (moves : List Move),
    initialState.m = m ∧
    initialState.n = n ∧
    (∀ cell, initialState.coins cell = 0) ∧
    isCorrectFinalState (moves.foldl (λ s m => applyMove s (sorry) m) initialState) :=
  sorry

end NUMINAMATH_CALUDE_ali_baba_walk_possible_l3450_345001


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l3450_345026

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (t - 3, 0)
  let B : ℝ × ℝ := (-1, t + 2)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 + 1) →
  (t = Real.sqrt 2 ∨ t = -Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l3450_345026


namespace NUMINAMATH_CALUDE_grocery_bag_capacity_l3450_345004

theorem grocery_bag_capacity (bag_capacity : ℕ) (green_beans : ℕ) (milk : ℕ) (carrot_multiplier : ℕ) :
  bag_capacity = 20 →
  green_beans = 4 →
  milk = 6 →
  carrot_multiplier = 2 →
  bag_capacity - (green_beans + milk + carrot_multiplier * green_beans) = 2 := by
  sorry

end NUMINAMATH_CALUDE_grocery_bag_capacity_l3450_345004


namespace NUMINAMATH_CALUDE_alexs_score_l3450_345037

theorem alexs_score (total_students : ℕ) (average_without_alex : ℚ) (average_with_alex : ℚ) :
  total_students = 20 →
  average_without_alex = 75 →
  average_with_alex = 76 →
  (total_students - 1) * average_without_alex + 95 = total_students * average_with_alex :=
by sorry

end NUMINAMATH_CALUDE_alexs_score_l3450_345037


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l3450_345018

theorem cuboid_diagonal (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2)
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l3450_345018


namespace NUMINAMATH_CALUDE_x_range_l3450_345087

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y) (h3 : y ≤ 7) :
  1 ≤ x ∧ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3450_345087


namespace NUMINAMATH_CALUDE_solution_set_for_specific_values_minimum_value_for_general_case_l3450_345067

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- Theorem for part (I)
theorem solution_set_for_specific_values (x : ℝ) :
  let a := 1
  let b := 2
  (f x a b ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
sorry

-- Theorem for part (II)
theorem minimum_value_for_general_case (x a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 2*a*b) :
  f x a b ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_specific_values_minimum_value_for_general_case_l3450_345067


namespace NUMINAMATH_CALUDE_cos_theta_range_l3450_345098

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 21 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the center of circle2
def O : ℝ × ℝ := (0, 0)

-- Define a point P on circle1
def P : ℝ × ℝ := sorry

-- Define points A and B where tangents from P touch circle2
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle θ between vectors PA and PB
def θ : ℝ := sorry

-- State the theorem
theorem cos_theta_range :
  circle1 P.1 P.2 →
  circle2 A.1 A.2 →
  circle2 B.1 B.2 →
  (1 : ℝ) / 9 ≤ Real.cos θ ∧ Real.cos θ ≤ 41 / 49 :=
sorry

end NUMINAMATH_CALUDE_cos_theta_range_l3450_345098


namespace NUMINAMATH_CALUDE_marble_probability_l3450_345002

theorem marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) :
  total = 90 →
  p_white = 1/3 →
  p_red_or_blue = 7/15 →
  ∃ (white red blue green : ℕ),
    white + red + blue + green = total ∧
    p_white = white / total ∧
    p_red_or_blue = (red + blue) / total ∧
    green / total = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l3450_345002


namespace NUMINAMATH_CALUDE_mary_fruit_cost_l3450_345099

/-- The total cost of fruits Mary bought -/
def total_cost (berries apples peaches grapes bananas pineapples : ℚ) : ℚ :=
  berries + apples + peaches + grapes + bananas + pineapples

/-- Theorem stating that the total cost of fruits Mary bought is $52.09 -/
theorem mary_fruit_cost :
  total_cost 11.08 14.33 9.31 7.50 5.25 4.62 = 52.09 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_cost_l3450_345099


namespace NUMINAMATH_CALUDE_black_number_equals_sum_of_white_numbers_l3450_345041

theorem black_number_equals_sum_of_white_numbers :
  ∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (Real.sqrt (c + d * Real.sqrt 7) = Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_black_number_equals_sum_of_white_numbers_l3450_345041


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_20112021_base5_l3450_345043

/-- Converts a base 5 number represented as a string to a natural number -/
def base5ToNat (s : String) : ℕ := sorry

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_20112021_base5 :
  largestPrimeDivisor (base5ToNat "20112021") = 419 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_20112021_base5_l3450_345043


namespace NUMINAMATH_CALUDE_five_dogs_not_eating_any_l3450_345009

/-- The number of dogs that do not eat any of the three foods (watermelon, salmon, chicken) -/
def dogs_not_eating_any (total dogs_watermelon dogs_salmon dogs_chicken dogs_watermelon_salmon dogs_chicken_salmon_not_watermelon : ℕ) : ℕ :=
  total - (dogs_watermelon + dogs_salmon + dogs_chicken - dogs_watermelon_salmon - dogs_chicken_salmon_not_watermelon)

/-- Theorem stating that 5 dogs do not eat any of the three foods -/
theorem five_dogs_not_eating_any :
  dogs_not_eating_any 75 15 54 20 12 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_dogs_not_eating_any_l3450_345009


namespace NUMINAMATH_CALUDE_alphametic_puzzle_unique_solution_l3450_345094

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the alphametic puzzle IDA + ME = ORA -/
def AlphameticPuzzle (I D A M E R O : Digit) : Prop :=
  (100 * I.val + 10 * D.val + A.val) + (10 * M.val + E.val) = 
  (100 * O.val + 10 * R.val + A.val)

/-- The main theorem stating that there exists a unique solution to the puzzle -/
theorem alphametic_puzzle_unique_solution : 
  ∃! (I D A M E R O : Digit), 
    AlphameticPuzzle I D A M E R O ∧ 
    I ≠ D ∧ I ≠ A ∧ I ≠ M ∧ I ≠ E ∧ I ≠ R ∧ I ≠ O ∧
    D ≠ A ∧ D ≠ M ∧ D ≠ E ∧ D ≠ R ∧ D ≠ O ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ R ∧ A ≠ O ∧
    M ≠ E ∧ M ≠ R ∧ M ≠ O ∧
    E ≠ R ∧ E ≠ O ∧
    R ≠ O ∧
    R.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_unique_solution_l3450_345094


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l3450_345045

theorem simplified_fraction_ratio (k : ℝ) : 
  let original := (6 * k + 12) / 6
  let simplified := k + 2
  ∃ (a b : ℤ), (simplified = a * k + b) ∧ (a / b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l3450_345045


namespace NUMINAMATH_CALUDE_louisa_travel_speed_l3450_345086

/-- The average speed of Louisa's travel -/
def average_speed : ℝ := 37.5

/-- The distance traveled on the first day -/
def distance_day1 : ℝ := 375

/-- The distance traveled on the second day -/
def distance_day2 : ℝ := 525

/-- The time difference between the two trips -/
def time_difference : ℝ := 4

theorem louisa_travel_speed :
  (distance_day2 / average_speed) = (distance_day1 / average_speed) + time_difference :=
sorry

end NUMINAMATH_CALUDE_louisa_travel_speed_l3450_345086


namespace NUMINAMATH_CALUDE_factorial_difference_l3450_345047

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3450_345047


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_26_l3450_345072

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 3 units
    and area equal to the triangle's area, the perimeter of the rectangle is 26 units. -/
theorem rectangle_perimeter_equals_26 (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width : ℝ) (h1 : triangle_side1 = 5)
  (h2 : triangle_side2 = 12) (h3 : triangle_side3 = 13) (h4 : rectangle_width = 3)
  (h5 : (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) :
  2 * (rectangle_width + (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_26_l3450_345072


namespace NUMINAMATH_CALUDE_functional_inequality_l3450_345012

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  let f : ℝ → ℝ := λ x => (x^3 - x^2 - 1) / (2*x*(x-1))
  f x + f ((x-1)/x) ≥ 1 + x :=
by sorry

end NUMINAMATH_CALUDE_functional_inequality_l3450_345012


namespace NUMINAMATH_CALUDE_position_of_2005_2004_l3450_345056

/-- The sum of numerator and denominator for the fraction 2005/2004 -/
def target_sum : ℕ := 2005 + 2004

/-- The position of a fraction in the sequence -/
def position (n d : ℕ) : ℕ :=
  let s := n + d
  (s - 1) * (s - 2) / 2 + (s - n)

/-- The theorem stating the position of 2005/2004 in the sequence -/
theorem position_of_2005_2004 : position 2005 2004 = 8028032 := by
  sorry


end NUMINAMATH_CALUDE_position_of_2005_2004_l3450_345056


namespace NUMINAMATH_CALUDE_sum_arithmetic_series_base8_l3450_345062

/-- Conversion from base 8 to base 10 -/
def base8ToBase10 (x : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def base10ToBase8 (x : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 8 -/
def sumArithmeticSeriesBase8 (n a l : ℕ) : ℕ :=
  base10ToBase8 ((n * (base8ToBase10 a + base8ToBase10 l)) / 2)

theorem sum_arithmetic_series_base8 :
  sumArithmeticSeriesBase8 36 1 36 = 1056 := by sorry

end NUMINAMATH_CALUDE_sum_arithmetic_series_base8_l3450_345062


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3450_345025

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 36) (h2 : leg = 12) :
  ∃ (perimeter : ℝ), perimeter = 18 + 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3450_345025


namespace NUMINAMATH_CALUDE_total_students_is_thirteen_l3450_345010

/-- The number of students in a presentation order, where Eunjeong's position is known. -/
def total_students (students_before_eunjeong : ℕ) (eunjeong_position_from_last : ℕ) : ℕ :=
  students_before_eunjeong + 1 + (eunjeong_position_from_last - 1)

/-- Theorem stating that the total number of students is 13 given the problem conditions. -/
theorem total_students_is_thirteen :
  total_students 7 6 = 13 := by sorry

end NUMINAMATH_CALUDE_total_students_is_thirteen_l3450_345010


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l3450_345061

/-- Given a quadratic function f(x) = ax² + 2, prove that if its tangent line
    at x = 1 is perpendicular to the line 2x - y + 1 = 0, then a = -1/4. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x
  let tangent_slope : ℝ := f' 1
  let perpendicular_line_slope : ℝ := 2
  tangent_slope * perpendicular_line_slope = -1 → a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l3450_345061


namespace NUMINAMATH_CALUDE_adams_airplane_change_l3450_345064

/-- The change Adam will receive when buying an airplane -/
def adams_change (adams_money : ℚ) (airplane_cost : ℚ) : ℚ :=
  adams_money - airplane_cost

/-- Theorem: Adam's change when buying an airplane -/
theorem adams_airplane_change :
  adams_change 5 4.28 = 0.72 := by sorry

end NUMINAMATH_CALUDE_adams_airplane_change_l3450_345064


namespace NUMINAMATH_CALUDE_students_at_1544_l3450_345007

/-- Calculates the number of students in the computer lab at a given time -/
def studentsInLab (initialTime startTime endTime : Nat) (initialStudents : Nat) 
  (enterInterval enterCount : Nat) (leaveInterval leaveCount : Nat) : Nat :=
  let totalMinutes := endTime - initialTime
  let enterTimes := (totalMinutes - (startTime - initialTime)) / enterInterval
  let leaveTimes := (totalMinutes - (startTime - initialTime)) / leaveInterval
  initialStudents + enterTimes * enterCount - leaveTimes * leaveCount

theorem students_at_1544 :
  studentsInLab 1500 1503 1544 20 3 4 10 8 = 44 := by
  sorry

end NUMINAMATH_CALUDE_students_at_1544_l3450_345007


namespace NUMINAMATH_CALUDE_correct_ab_sample_size_l3450_345030

/-- Represents the number of students to be drawn with blood type AB in a stratified sampling -/
def stratified_sample_ab (total_students : ℕ) (ab_students : ℕ) (sample_size : ℕ) : ℕ :=
  (ab_students * sample_size) / total_students

/-- Theorem stating the correct number of AB blood type students in the sample -/
theorem correct_ab_sample_size :
  stratified_sample_ab 500 50 60 = 6 := by sorry

end NUMINAMATH_CALUDE_correct_ab_sample_size_l3450_345030


namespace NUMINAMATH_CALUDE_costume_ball_same_gender_dance_l3450_345021

/-- Represents a person at the costume ball -/
structure Person :=
  (partners : Nat)

/-- Represents the costume ball -/
structure CostumeBall :=
  (people : Finset Person)
  (total_people : Nat)
  (total_dances : Nat)

/-- The costume ball satisfies the given conditions -/
def valid_costume_ball (ball : CostumeBall) : Prop :=
  ball.total_people = 20 ∧
  (ball.people.filter (λ p => p.partners = 3)).card = 11 ∧
  (ball.people.filter (λ p => p.partners = 5)).card = 1 ∧
  (ball.people.filter (λ p => p.partners = 6)).card = 8 ∧
  ball.total_dances = (11 * 3 + 1 * 5 + 8 * 6) / 2

theorem costume_ball_same_gender_dance (ball : CostumeBall) 
  (h : valid_costume_ball ball) : 
  ¬ (∀ (dance : Nat), dance < ball.total_dances → 
    ∃ (p1 p2 : Person), p1 ∈ ball.people ∧ p2 ∈ ball.people ∧ p1 ≠ p2) :=
by sorry

end NUMINAMATH_CALUDE_costume_ball_same_gender_dance_l3450_345021


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3450_345042

/-- Proves that the given polar equation is equivalent to the given rectangular equation. -/
theorem polar_to_rectangular_equivalence :
  ∀ (r φ x y : ℝ),
  (r = 2 / (4 - Real.sin φ)) ↔ 
  (x^2 / (2/Real.sqrt 15)^2 + (y - 2/15)^2 / (8/15)^2 = 1 ∧
   x = r * Real.cos φ ∧
   y = r * Real.sin φ) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3450_345042


namespace NUMINAMATH_CALUDE_coefficient_a3_equals_80_l3450_345024

theorem coefficient_a3_equals_80 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ),
  (2 * x^2 + 1)^5 = a₀ + a₁ * x^2 + a₂ * x^4 + a₃ * x^6 + a₄ * x^8 + a₅ * x^10 →
  a₃ = 80 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_equals_80_l3450_345024


namespace NUMINAMATH_CALUDE_recurrence_sequence_theorem_l3450_345019

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ∀ n, (a (n + 1))^2 = (a n) * (a (n + 2)) + k

/-- Three terms form an arithmetic sequence -/
def IsArithmeticSequence (x y z : ℝ) : Prop := y - x = z - y

theorem recurrence_sequence_theorem (a : ℕ → ℝ) (k : ℝ) 
  (h : RecurrenceSequence a k) :
  (k = (a 2 - a 1)^2 → IsArithmeticSequence (a 1) (a 2) (a 3)) ∧ 
  (k = 0 → IsArithmeticSequence (a 2) (a 4) (a 5) → 
    (a 2) / (a 1) = 1 ∨ (a 2) / (a 1) = (1 + Real.sqrt 5) / 2) := by
  sorry


end NUMINAMATH_CALUDE_recurrence_sequence_theorem_l3450_345019


namespace NUMINAMATH_CALUDE_characterize_no_solution_set_l3450_345089

/-- The set of real numbers a for which the equation has no solution -/
def NoSolutionSet : Set ℝ :=
  {a | ∀ x, 9 * |x - 4*a| + |x - a^2| + 8*x - 4*a ≠ 0}

/-- The theorem stating the characterization of the set where the equation has no solution -/
theorem characterize_no_solution_set :
  NoSolutionSet = {a | a < -24 ∨ a > 0} :=
by sorry

end NUMINAMATH_CALUDE_characterize_no_solution_set_l3450_345089


namespace NUMINAMATH_CALUDE_tangent_circles_ratio_l3450_345058

/-- Two circles are tangent if the distance between their centers is equal to the sum of their radii -/
def are_tangent (center1 center2 : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (2 * radius)^2

/-- Definition of circle C₁ -/
def circle_C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

/-- Definition of circle C₂ -/
def circle_C2 (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - b)^2 + (p.2 - c)^2 = a^2}

theorem tangent_circles_ratio (a b c : ℝ) (ha : a > 0) :
  are_tangent (0, 0) (b, c) a →
  (b^2 + c^2) / a^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_ratio_l3450_345058


namespace NUMINAMATH_CALUDE_min_investment_optimal_quantities_l3450_345039

/-- Represents the cost and quantity of stationery types A and B -/
structure Stationery where
  cost_A : ℕ
  cost_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Defines the conditions of the stationery purchase problem -/
def stationery_problem (s : Stationery) : Prop :=
  s.cost_A * 2 + s.cost_B = 35 ∧
  s.cost_A + s.cost_B * 3 = 30 ∧
  s.quantity_A + s.quantity_B = 120 ∧
  975 ≤ s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ∧
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≤ 1000

/-- Theorem stating the minimum investment for the stationery purchase -/
theorem min_investment (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≥ 980 :=
by sorry

/-- Theorem stating the optimal purchase quantities -/
theorem optimal_quantities (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B = 980 →
  s.quantity_A = 38 ∧ s.quantity_B = 82 :=
by sorry

end NUMINAMATH_CALUDE_min_investment_optimal_quantities_l3450_345039


namespace NUMINAMATH_CALUDE_hyperbola_iff_mn_negative_l3450_345065

/-- A hyperbola is represented by the equation x²/m + y²/n = 1 where m and n are real numbers. -/
def IsHyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1

/-- The condition mn < 0 is both necessary and sufficient for the equation x²/m + y²/n = 1 
    to represent a hyperbola. -/
theorem hyperbola_iff_mn_negative (m n : ℝ) : IsHyperbola m n ↔ m * n < 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_iff_mn_negative_l3450_345065


namespace NUMINAMATH_CALUDE_bag_to_items_ratio_l3450_345036

/-- The cost of a shirt in dollars -/
def shirt_cost : ℚ := 7

/-- The cost of a pair of shoes in dollars -/
def shoes_cost : ℚ := shirt_cost + 3

/-- The total cost of 2 shirts and a pair of shoes in dollars -/
def total_cost_without_bag : ℚ := 2 * shirt_cost + shoes_cost

/-- The total cost of all items (including the bag) in dollars -/
def total_cost : ℚ := 36

/-- The cost of the bag in dollars -/
def bag_cost : ℚ := total_cost - total_cost_without_bag

/-- Theorem stating that the ratio of the bag cost to the total cost without bag is 1:2 -/
theorem bag_to_items_ratio :
  bag_cost / total_cost_without_bag = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_bag_to_items_ratio_l3450_345036


namespace NUMINAMATH_CALUDE_geometry_relations_l3450_345034

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l : Line) (m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : subset m β) :
  (perpendicular l β → plane_perpendicular α β) ∧
  (parallel α β → line_parallel l β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l3450_345034


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_thirds_l3450_345005

theorem abs_ratio_eq_sqrt_seven_thirds 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : a^2 + b^2 = 5*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_thirds_l3450_345005


namespace NUMINAMATH_CALUDE_total_dog_legs_l3450_345092

/-- Proves that the total number of dog legs on a street is 400, given the conditions. -/
theorem total_dog_legs (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2/3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_legs_l3450_345092


namespace NUMINAMATH_CALUDE_expected_distinct_faces_proof_l3450_345085

/-- A fair six-sided die is rolled six times. -/
def roll_die (n : ℕ) : Type := Fin 6 → Fin n

/-- The probability of a specific face not appearing in a single roll. -/
def prob_not_appear : ℚ := 5 / 6

/-- The expected number of distinct faces appearing in six rolls of a fair die. -/
def expected_distinct_faces : ℚ := (6^6 - 5^6) / 6^5

/-- Theorem stating that the expected number of distinct faces appearing when a fair
    six-sided die is rolled six times is equal to (6^6 - 5^6) / 6^5. -/
theorem expected_distinct_faces_proof :
  expected_distinct_faces = (6^6 - 5^6) / 6^5 :=
by sorry

end NUMINAMATH_CALUDE_expected_distinct_faces_proof_l3450_345085


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_lt_one_l3450_345028

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_square_lt_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x^2 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_lt_one_l3450_345028


namespace NUMINAMATH_CALUDE_birthday_gift_contribution_l3450_345068

theorem birthday_gift_contribution (total_cost boss_contribution num_remaining_employees : ℕ) 
  (h1 : total_cost = 100)
  (h2 : boss_contribution = 15)
  (h3 : num_remaining_employees = 5) :
  let todd_contribution := 2 * boss_contribution
  let remaining_cost := total_cost - todd_contribution - boss_contribution
  remaining_cost / num_remaining_employees = 11 := by
sorry

end NUMINAMATH_CALUDE_birthday_gift_contribution_l3450_345068


namespace NUMINAMATH_CALUDE_vector_parallel_solution_l3450_345015

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

theorem vector_parallel_solution (m : ℝ) : 
  let a : Vec2D := ⟨1, m⟩
  let b : Vec2D := ⟨2, 5⟩
  let c : Vec2D := ⟨m, 3⟩
  parallel (Vec2D.mk (a.x + c.x) (a.y + c.y)) (Vec2D.mk (a.x - b.x) (a.y - b.y)) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_solution_l3450_345015


namespace NUMINAMATH_CALUDE_division_problem_l3450_345088

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2 * 3) = 120 / 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3450_345088


namespace NUMINAMATH_CALUDE_valid_pairs_l3450_345096

theorem valid_pairs : 
  ∀ m n : ℕ, 
    (∃ k : ℕ, m + 1 = n * k) ∧ 
    (∃ l : ℕ, n^2 - n + 1 = m * l) → 
    ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_l3450_345096


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l3450_345075

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) : 0.75 * (1.30 * x) < x := by
  sorry

#check sale_price_lower_than_original

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l3450_345075


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l3450_345050

/-- The function f(x) = xe^x + 1 is decreasing on the interval (-∞, -1) -/
theorem function_decreasing_interval (x : ℝ) : 
  x < -1 → (fun x => x * Real.exp x + 1) '' Set.Ioi x ⊆ Set.Iio ((x * Real.exp x + 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l3450_345050


namespace NUMINAMATH_CALUDE_max_value_w_l3450_345054

theorem max_value_w (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ (w_max : ℝ), w_max = 0 ∧ ∀ (w : ℝ), w = x^2 + y^2 - 8 * x → w ≤ w_max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_w_l3450_345054


namespace NUMINAMATH_CALUDE_animal_sightings_sum_l3450_345073

/-- The number of animal sightings in January -/
def january_sightings : ℕ := 26

/-- The number of animal sightings in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of animal sightings in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The number of animal sightings in April -/
def april_sightings : ℕ := 2 * march_sightings

/-- The total number of animal sightings over the four months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings + april_sightings

theorem animal_sightings_sum : total_sightings = 221 := by
  sorry

end NUMINAMATH_CALUDE_animal_sightings_sum_l3450_345073


namespace NUMINAMATH_CALUDE_select_players_correct_l3450_345000

/-- The number of ways to select k players from m teams, each with n players,
    such that no two selected players are from the same team -/
def select_players (m n k : ℕ) : ℕ :=
  Nat.choose m k * n^k

/-- Theorem stating that select_players gives the correct number of ways
    to form the committee under the given conditions -/
theorem select_players_correct (m n k : ℕ) (h : k ≤ m) :
  select_players m n k = Nat.choose m k * n^k :=
by sorry

end NUMINAMATH_CALUDE_select_players_correct_l3450_345000


namespace NUMINAMATH_CALUDE_inequality_theorem_l3450_345006

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hxy₁ : x₁ * y₁ - z₁^2 > 0) (hxy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3450_345006


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l3450_345022

theorem trig_fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, (Real.sin x)^4 / a + (Real.cos x)^4 / b = 1 / (a + b)) :
  ∀ x : ℝ, (Real.sin x)^8 / a^3 + (Real.cos x)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l3450_345022


namespace NUMINAMATH_CALUDE_yard_area_l3450_345044

/-- The area of a rectangular yard with two cut-out areas -/
theorem yard_area (yard_length yard_width square_side rectangle_length rectangle_width : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : square_side = 3)
  (h4 : rectangle_length = 4)
  (h5 : rectangle_width = 2) :
  yard_length * yard_width - (square_side * square_side + rectangle_length * rectangle_width) = 343 := by
  sorry

#check yard_area

end NUMINAMATH_CALUDE_yard_area_l3450_345044


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l3450_345057

theorem range_of_x2_plus_y2 (x y : ℝ) 
  (h1 : x - 2*y + 1 ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : x ≥ 0) : 
  0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

#check range_of_x2_plus_y2

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l3450_345057


namespace NUMINAMATH_CALUDE_abc_magnitude_order_l3450_345048

/-- Given the definitions of a, b, and c, prove that b > c > a -/
theorem abc_magnitude_order :
  let a := (1/2) * Real.cos (16 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (16 * π / 180)
  let b := (2 * Real.tan (14 * π / 180)) / (1 + Real.tan (14 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_abc_magnitude_order_l3450_345048


namespace NUMINAMATH_CALUDE_inequality_solution_length_l3450_345060

theorem inequality_solution_length (a b : ℝ) : 
  (∀ x, (a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) ↔ ((a - 5) / 3 ≤ x ∧ x ≤ (b - 8) / 3)) →
  ((b - 8) / 3 - (a - 5) / 3 = 18) →
  b - a = 57 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l3450_345060


namespace NUMINAMATH_CALUDE_car_trip_distance_l3450_345053

theorem car_trip_distance (D : ℝ) : D - (1/2) * D - (1/4) * ((1/2) * D) = 135 → D = 360 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l3450_345053


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3450_345071

/-- Given a hyperbola with the following properties:
  - Point P is on the right branch of the hyperbola (x²/a² - y²/b² = 1), where a > 0 and b > 0
  - F₁ and F₂ are the left and right foci of the hyperbola
  - (OP + OF₂) · F₂P = 0, where O is the origin
  - |PF₁| = √3|PF₂|
  Its eccentricity is √3 + 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ O : ℝ × ℝ) 
  (h_a : a > 0) (h_b : b > 0)
  (h_P : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁.1 < 0 ∧ F₂.1 > 0)
  (h_origin : O = (0, 0))
  (h_perpendicular : (P - O + (F₂ - O)) • (P - F₂) = 0)
  (h_distance_ratio : ‖P - F₁‖ = Real.sqrt 3 * ‖P - F₂‖) :
  let c := ‖F₂ - O‖
  c / a = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3450_345071


namespace NUMINAMATH_CALUDE_james_hats_per_yard_l3450_345055

/-- The number of yards of velvet needed to make one cloak -/
def yards_per_cloak : ℕ := 3

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made -/
def num_cloaks : ℕ := 6

/-- The number of hats made -/
def num_hats : ℕ := 12

/-- The number of hats James can make out of one yard of velvet -/
def hats_per_yard : ℕ := 4

theorem james_hats_per_yard :
  (total_yards - num_cloaks * yards_per_cloak) * hats_per_yard = num_hats := by
  sorry

end NUMINAMATH_CALUDE_james_hats_per_yard_l3450_345055


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l3450_345076

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 36 * x - 60 = x * (x + 17)) → x ≥ -12 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l3450_345076


namespace NUMINAMATH_CALUDE_paint_coats_calculation_l3450_345066

/-- Proves the number of coats of paint that can be applied given the wall area,
    paint coverage, paint cost, and individual contributions. -/
theorem paint_coats_calculation (wall_area : ℝ) (paint_coverage : ℝ) (paint_cost : ℝ) (contribution : ℝ)
    (h_wall : wall_area = 1600)
    (h_coverage : paint_coverage = 400)
    (h_cost : paint_cost = 45)
    (h_contribution : contribution = 180) :
    ⌊(2 * contribution) / (paint_cost * (wall_area / paint_coverage))⌋ = 2 := by
  sorry

#check paint_coats_calculation

end NUMINAMATH_CALUDE_paint_coats_calculation_l3450_345066
