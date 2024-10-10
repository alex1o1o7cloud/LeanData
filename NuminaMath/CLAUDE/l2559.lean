import Mathlib

namespace class_size_class_size_problem_l2559_255968

theorem class_size (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) : ℕ :=
  let stickers_to_friends := num_friends * stickers_per_friend
  let remaining_stickers := total_stickers - stickers_to_friends - leftover_stickers
  let other_students := remaining_stickers / stickers_per_other
  other_students + num_friends + 1

theorem class_size_problem : 
  class_size 50 5 4 2 8 = 17 := by
  sorry

end class_size_class_size_problem_l2559_255968


namespace range_of_a_l2559_255987

theorem range_of_a (a : ℝ) : 
  (∀ x > a, x * (x - 1) > 0) ↔ a ≥ 1 := by
sorry

end range_of_a_l2559_255987


namespace not_necessarily_right_triangle_l2559_255984

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  A / 3 = B / 4 →
  B / 4 = C / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end not_necessarily_right_triangle_l2559_255984


namespace reflect_point_across_x_axis_l2559_255981

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflect_point_across_x_axis :
  let P : Point := ⟨2, -3⟩
  let P' : Point := reflectAcrossXAxis P
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end reflect_point_across_x_axis_l2559_255981


namespace triangle_side_length_l2559_255995

/-- Given a triangle ABC with angle A = 30°, angle B = 105°, and side a = 4,
    prove that the length of side c is 4√2. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = 7*π/12 → a = 4 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C → 
  c = 4 * Real.sqrt 2 := by
sorry

end triangle_side_length_l2559_255995


namespace at_most_three_lines_unique_line_through_two_points_l2559_255975

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to create a line from two points
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

-- Theorem 1: At most three lines can be drawn through any two of three distinct points
theorem at_most_three_lines (p1 p2 p3 : Point) 
  (h1 : p1 ≠ p2) (h2 : p2 ≠ p3) (h3 : p1 ≠ p3) : 
  ∃ (l1 l2 l3 : Line), ∀ (l : Line), 
    (isPointOnLine p1 l ∧ isPointOnLine p2 l) ∨
    (isPointOnLine p2 l ∧ isPointOnLine p3 l) ∨
    (isPointOnLine p1 l ∧ isPointOnLine p3 l) →
    l = l1 ∨ l = l2 ∨ l = l3 :=
sorry

-- Theorem 2: Only one line can be drawn through two distinct points
theorem unique_line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! (l : Line), isPointOnLine p1 l ∧ isPointOnLine p2 l :=
sorry

end at_most_three_lines_unique_line_through_two_points_l2559_255975


namespace employee_pay_l2559_255962

theorem employee_pay (x y : ℝ) (h1 : x + y = 616) (h2 : x = 1.2 * y) : y = 280 := by
  sorry

end employee_pay_l2559_255962


namespace mine_locations_determinable_l2559_255907

/-- Represents the state of a cell in the grid -/
inductive CellState
  | Empty
  | Mine

/-- Represents the grid of cells -/
def Grid (n : ℕ) := Fin n → Fin n → CellState

/-- The number displayed in a cell, which is the count of mines in the cell and its surroundings -/
def CellNumber (n : ℕ) (grid : Grid n) (i j : Fin n) : Fin 10 :=
  sorry

/-- Checks if it's possible to uniquely determine mine locations given cell numbers -/
def CanDetermineMineLocations (n : ℕ) (cellNumbers : Fin n → Fin n → Fin 10) : Prop :=
  ∃! (grid : Grid n), ∀ (i j : Fin n), CellNumber n grid i j = cellNumbers i j

/-- Theorem stating that mine locations can be determined for n = 2009 and n = 2007 -/
theorem mine_locations_determinable :
  (∀ (cellNumbers : Fin 2009 → Fin 2009 → Fin 10), CanDetermineMineLocations 2009 cellNumbers) ∧
  (∀ (cellNumbers : Fin 2007 → Fin 2007 → Fin 10), CanDetermineMineLocations 2007 cellNumbers) :=
sorry

end mine_locations_determinable_l2559_255907


namespace worker_b_days_l2559_255930

/-- The number of days it takes for worker b to complete a job alone,
    given that worker a is twice as efficient as worker b and
    together they complete the job in 6 days. -/
theorem worker_b_days (efficiency_a : ℝ) (efficiency_b : ℝ) (days_together : ℝ) : 
  efficiency_a = 2 * efficiency_b →
  days_together = 6 →
  efficiency_a + efficiency_b = 1 / days_together →
  1 / efficiency_b = 18 := by
sorry

end worker_b_days_l2559_255930


namespace M_intersect_N_l2559_255913

def M : Set ℝ := {x | x^2 - 4 < 0}
def N : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end M_intersect_N_l2559_255913


namespace sequence_arrangement_count_l2559_255939

theorem sequence_arrangement_count : ℕ := by
  -- Define the length of the sequence
  let n : ℕ := 6

  -- Define the counts of each number in the sequence
  let count_of_ones : ℕ := 3
  let count_of_twos : ℕ := 2
  let count_of_threes : ℕ := 1

  -- Assert that the sum of counts equals the sequence length
  have h_sum_counts : count_of_ones + count_of_twos + count_of_threes = n := by sorry

  -- Define the number of ways to arrange the sequence
  let arrangement_count : ℕ := n.choose count_of_threes * (n - count_of_threes).choose count_of_twos

  -- Prove that the arrangement count equals 60
  have h_arrangement_count : arrangement_count = 60 := by sorry

  -- Return the final result
  exact 60

end sequence_arrangement_count_l2559_255939


namespace range_of_a_l2559_255936

def star_op (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x, star_op x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) →
  -2 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l2559_255936


namespace laura_age_l2559_255912

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem laura_age :
  ∀ (L A : ℕ),
  is_divisible_by (L - 1) 8 →
  is_divisible_by (A - 1) 8 →
  is_divisible_by (L + 1) 7 →
  is_divisible_by (A + 1) 7 →
  A < 100 →
  L = 41 :=
by sorry

end laura_age_l2559_255912


namespace interest_rate_calculation_interest_rate_value_l2559_255969

/-- The interest rate for Rs 100 over 8 years that produces the same interest as Rs 200 at 10% for 2 years -/
def interest_rate : ℝ := sorry

/-- The initial amount in rupees -/
def initial_amount : ℝ := 100

/-- The time period in years -/
def time_period : ℝ := 8

/-- The comparison amount in rupees -/
def comparison_amount : ℝ := 200

/-- The comparison interest rate -/
def comparison_rate : ℝ := 0.1

/-- The comparison time period in years -/
def comparison_time : ℝ := 2

theorem interest_rate_calculation : 
  initial_amount * interest_rate * time_period = 
  comparison_amount * comparison_rate * comparison_time :=
sorry

theorem interest_rate_value : interest_rate = 0.05 :=
sorry

end interest_rate_calculation_interest_rate_value_l2559_255969


namespace division_problem_l2559_255925

theorem division_problem (A : ℕ) : 
  (A / 6 = 3) ∧ (A % 6 = 2) → A = 20 := by
  sorry

end division_problem_l2559_255925


namespace range_of_f_l2559_255948

/-- The function f(c) defined as (c-a)(c-b) -/
def f (c a b : ℝ) : ℝ := (c - a) * (c - b)

/-- Theorem stating the range of f(c) -/
theorem range_of_f :
  ∀ c a b : ℝ,
  a + b = 1 - c →
  c ≥ 0 →
  a ≥ 0 →
  b ≥ 0 →
  ∃ y : ℝ, f c a b = y ∧ -1/8 ≤ y ∧ y ≤ 1 :=
by sorry

end range_of_f_l2559_255948


namespace xyz_inequality_l2559_255947

theorem xyz_inequality (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z :=
by sorry

end xyz_inequality_l2559_255947


namespace consecutive_integers_problem_l2559_255959

theorem consecutive_integers_problem (n : ℕ) (avg : ℚ) (max : ℕ) 
  (h_consecutive : ∃ (start : ℤ), ∀ i : ℕ, i < n → start + i ∈ (Set.range (fun i => start + i) : Set ℤ))
  (h_average : avg = (↑(n * (2 * max - n + 1)) / (2 * n) : ℚ))
  (h_max : max = 36)
  (h_avg : avg = 33) :
  n = 7 := by
sorry

end consecutive_integers_problem_l2559_255959


namespace x_total_time_is_20_l2559_255906

-- Define the work as a fraction of the total job
def Work := ℚ

-- Define the time y needs to finish the entire work
def y_total_time : ℕ := 16

-- Define the time y worked before leaving
def y_worked_time : ℕ := 12

-- Define the time x needed to finish the remaining work
def x_remaining_time : ℕ := 5

-- Theorem to prove
theorem x_total_time_is_20 : 
  ∃ (x_total_time : ℕ), 
    (y_worked_time : ℚ) / y_total_time + 
    (x_remaining_time : ℚ) / x_total_time = 1 ∧ 
    x_total_time = 20 := by sorry

end x_total_time_is_20_l2559_255906


namespace max_value_constraint_l2559_255911

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * a^2 + 4 * b^2 = 12 → 3 * a + 2 * b ≤ max :=
sorry

end max_value_constraint_l2559_255911


namespace gcd_45123_31207_l2559_255971

theorem gcd_45123_31207 : Nat.gcd 45123 31207 = 1 := by
  sorry

end gcd_45123_31207_l2559_255971


namespace intersection_line_slope_l2559_255992

-- Define the equations of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem stating that the slope of the line passing through the intersection points is 5/2
theorem intersection_line_slope :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧ p1 ≠ p2 ∧
  (p2.2 - p1.2) / (p2.1 - p1.1) = 5/2 :=
sorry

end intersection_line_slope_l2559_255992


namespace complex_fraction_third_quadrant_l2559_255920

/-- Given a complex fraction equal to 2-i, prove the resulting point is in the third quadrant -/
theorem complex_fraction_third_quadrant (a b : ℝ) : 
  (a + Complex.I) / (b - Complex.I) = 2 - Complex.I → 
  a < 0 ∧ b < 0 := by
  sorry

end complex_fraction_third_quadrant_l2559_255920


namespace jason_money_calculation_l2559_255927

/-- Represents the value of coins in cents -/
inductive Coin
  | quarter
  | dime
  | nickel

/-- The value of a coin in cents -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.quarter => 25
  | Coin.dime => 10
  | Coin.nickel => 5

/-- Calculates the total value of coins in dollars -/
def coins_value (quarters dimes nickels : ℕ) : ℚ :=
  (quarters * coin_value Coin.quarter + dimes * coin_value Coin.dime + nickels * coin_value Coin.nickel) / 100

/-- Converts euros to US dollars -/
def euros_to_dollars (euros : ℚ) : ℚ :=
  euros * 1.20

theorem jason_money_calculation (initial_quarters initial_dimes initial_nickels : ℕ)
    (initial_euros : ℚ)
    (additional_quarters additional_dimes additional_nickels : ℕ)
    (additional_euros : ℚ) :
    let initial_coins := coins_value initial_quarters initial_dimes initial_nickels
    let initial_dollars := initial_coins + euros_to_dollars initial_euros
    let additional_coins := coins_value additional_quarters additional_dimes additional_nickels
    let additional_dollars := additional_coins + euros_to_dollars additional_euros
    let total_dollars := initial_dollars + additional_dollars
    initial_quarters = 49 →
    initial_dimes = 32 →
    initial_nickels = 18 →
    initial_euros = 22.50 →
    additional_quarters = 25 →
    additional_dimes = 15 →
    additional_nickels = 10 →
    additional_euros = 12 →
    total_dollars = 66 := by
  sorry

end jason_money_calculation_l2559_255927


namespace lottery_probability_theorem_l2559_255963

def megaBallCount : ℕ := 30
def winnerBallsTotal : ℕ := 50
def winnerBallsPicked : ℕ := 5
def bonusBallCount : ℕ := 15

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (Nat.choose winnerBallsTotal winnerBallsPicked) * bonusBallCount)

theorem lottery_probability_theorem :
  lotteryProbability = 1 / 95673600 := by
  sorry

end lottery_probability_theorem_l2559_255963


namespace zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l2559_255966

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + b * x + (b - 1)

-- Part 1
theorem zeros_of_specific_quadratic :
  let f₁ := f 1 (-2)
  (f₁ 3 = 0) ∧ (f₁ (-1) = 0) ∧ (∀ x, f₁ x = 0 → x = 3 ∨ x = -1) := by sorry

-- Part 2
theorem range_of_a_for_distinct_zeros (a : ℝ) :
  (a ≠ 0) →
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (0 < a ∧ a < 1) := by sorry

end zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l2559_255966


namespace units_digit_problem_l2559_255942

/-- Given a positive even integer with a positive units digit,
    if the units digit of its cube minus the units digit of its square is 0,
    then the number needed to be added to its units digit to get 10 is 4. -/
theorem units_digit_problem (p : ℕ) : 
  p > 0 → 
  Even p → 
  p % 10 > 0 → 
  p % 10 < 10 → 
  (p^3 % 10) - (p^2 % 10) = 0 → 
  10 - (p % 10) = 4 := by
  sorry

end units_digit_problem_l2559_255942


namespace pressure_change_pressure_at_4m3_l2559_255956

/-- Represents the pressure-volume relationship for a gas -/
structure GasPV where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  inverse_square_relation : ∀ v, pressure v = k / (volume v)^2

/-- The theorem stating the pressure when volume changes -/
theorem pressure_change (gas : GasPV) (v₁ v₂ : ℝ) (p₁ : ℝ) 
    (h₁ : gas.pressure v₁ = p₁)
    (h₂ : v₁ > 0)
    (h₃ : v₂ > 0)
    (h₄ : gas.volume v₁ = v₁)
    (h₅ : gas.volume v₂ = v₂) :
  gas.pressure v₂ = p₁ * (v₁ / v₂)^2 :=
by sorry

/-- The specific problem instance -/
theorem pressure_at_4m3 (gas : GasPV) 
    (h₁ : gas.pressure 2 = 25)
    (h₂ : gas.volume 2 = 2)
    (h₃ : gas.volume 4 = 4) :
  gas.pressure 4 = 6.25 :=
by sorry

end pressure_change_pressure_at_4m3_l2559_255956


namespace percentage_problem_l2559_255928

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 126 →
  x = 5600 := by
sorry

end percentage_problem_l2559_255928


namespace brian_shoe_count_l2559_255993

theorem brian_shoe_count :
  ∀ (b e j : ℕ),
    j = e / 2 →
    e = 3 * b →
    b + e + j = 121 →
    b = 22 :=
by
  sorry

end brian_shoe_count_l2559_255993


namespace upstream_speed_calculation_l2559_255994

/-- Calculates the upstream speed of a person given their downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem: Given a downstream speed of 12 km/h and a stream speed of 2 km/h, the upstream speed is 8 km/h. -/
theorem upstream_speed_calculation :
  upstreamSpeed 12 2 = 8 := by
  sorry

#eval upstreamSpeed 12 2

end upstream_speed_calculation_l2559_255994


namespace problem_statement_l2559_255967

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 5) 
  (h2 : 2 * x + 5 * y = 8) : 
  9 * x^2 + 38 * x * y + 41 * y^2 = 153 := by
  sorry

end problem_statement_l2559_255967


namespace RS_length_value_l2559_255934

/-- Triangle ABC with given side lengths and angle bisectors -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Altitude AD
  AD : ℝ
  -- Points R and S on AD
  AR : ℝ
  AS : ℝ
  -- Conditions
  side_lengths : AB = 11 ∧ BC = 13 ∧ CA = 14
  altitude : AD > 0
  R_on_AD : 0 < AR ∧ AR < AD
  S_on_AD : 0 < AS ∧ AS < AD
  BE_bisector : AR / (AD - AR) = CA / BC
  CF_bisector : AS / (AD - AS) = AB / BC

/-- The length of RS in the given triangle -/
def RS_length (t : TriangleABC) : ℝ := t.AR - t.AS

/-- Theorem stating that RS length is equal to 645√95 / 4551 -/
theorem RS_length_value (t : TriangleABC) : RS_length t = 645 * Real.sqrt 95 / 4551 := by
  sorry

end RS_length_value_l2559_255934


namespace regular_tetrahedron_iff_l2559_255938

/-- A tetrahedron -/
structure Tetrahedron where
  /-- The base of the tetrahedron -/
  base : Triangle
  /-- The apex of the tetrahedron -/
  apex : Point

/-- A regular tetrahedron -/
def RegularTetrahedron (t : Tetrahedron) : Prop :=
  sorry

/-- The base of the tetrahedron is an equilateral triangle -/
def HasEquilateralBase (t : Tetrahedron) : Prop :=
  sorry

/-- The dihedral angles between the lateral faces and the base are equal -/
def HasEqualDihedralAngles (t : Tetrahedron) : Prop :=
  sorry

/-- All lateral edges form equal angles with the base -/
def HasEqualLateralEdgeAngles (t : Tetrahedron) : Prop :=
  sorry

/-- Theorem: A tetrahedron is regular if and only if it satisfies certain conditions -/
theorem regular_tetrahedron_iff (t : Tetrahedron) : 
  RegularTetrahedron t ↔ 
  (HasEquilateralBase t ∧ HasEqualDihedralAngles t) ∨
  (HasEqualLateralEdgeAngles t ∧ HasEqualDihedralAngles t) :=
sorry

end regular_tetrahedron_iff_l2559_255938


namespace value_of_N_l2559_255922

theorem value_of_N : ∃ N : ℝ, (0.20 * N = 0.30 * 5000) ∧ (N = 7500) := by
  sorry

end value_of_N_l2559_255922


namespace absolute_value_equation_solution_l2559_255958

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2005 * x - 2005| = 2005 ↔ x = 2 ∨ x = 0 := by sorry

end absolute_value_equation_solution_l2559_255958


namespace circle_radius_from_area_circumference_ratio_l2559_255926

/-- Given a circle with area Q and circumference P, if Q/P = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (Q P : ℝ) (hQ : Q > 0) (hP : P > 0) :
  Q / P = 10 → ∃ (r : ℝ), r > 0 ∧ Q = π * r^2 ∧ P = 2 * π * r ∧ r = 20 := by
  sorry

end circle_radius_from_area_circumference_ratio_l2559_255926


namespace passing_percentage_l2559_255916

theorem passing_percentage (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end passing_percentage_l2559_255916


namespace cake_two_sided_icing_count_l2559_255921

/-- Represents a cube cake with icing on specific faces -/
structure CakeCube where
  size : Nat
  icedFaces : Finset (Fin 3)

/-- Counts the number of 1×1×1 subcubes with icing on exactly two sides -/
def countTwoSidedIcingCubes (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with icing on top, front, and back
    has exactly 12 subcubes with icing on two sides when cut into 1×1×1 cubes -/
theorem cake_two_sided_icing_count :
  let cake : CakeCube := { size := 5, icedFaces := {0, 1, 2} }
  countTwoSidedIcingCubes cake = 12 := by
  sorry

end cake_two_sided_icing_count_l2559_255921


namespace x_value_from_equation_l2559_255904

theorem x_value_from_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) : 
  x = (3 + Real.sqrt 153) / 4 := by
  sorry

end x_value_from_equation_l2559_255904


namespace lower_limit_of_range_l2559_255945

theorem lower_limit_of_range (x : ℕ) : 
  x ≤ 100 ∧ 
  (∃ (S : Finset ℕ), S.card = 13 ∧ 
    (∀ n ∈ S, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0) ∧
    (∀ n, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0 → n ∈ S)) →
  x = 24 :=
by sorry

end lower_limit_of_range_l2559_255945


namespace lcm_24_90_35_l2559_255905

theorem lcm_24_90_35 : Nat.lcm 24 (Nat.lcm 90 35) = 2520 := by
  sorry

end lcm_24_90_35_l2559_255905


namespace number_of_walls_proof_correct_l2559_255961

/-- Proves that the number of walls in a room is 5, given specific conditions about wall size and painting time. -/
theorem number_of_walls (wall_width : ℝ) (wall_height : ℝ) (painting_rate : ℝ) (total_time : ℝ) (spare_time : ℝ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : wall_width = 2 := by sorry
  have h2 : wall_height = 3 := by sorry
  have h3 : painting_rate = 1 / 10 := by sorry  -- 1 square meter per 10 minutes
  have h4 : total_time = 10 := by sorry  -- 10 hours
  have h5 : spare_time = 5 := by sorry  -- 5 hours

  -- Calculate the number of walls
  let wall_area := wall_width * wall_height
  let available_time := total_time - spare_time
  let paintable_area := available_time * 60 * painting_rate  -- Convert hours to minutes
  let number_of_walls := ⌊paintable_area / wall_area⌋  -- Floor division

  -- Prove that the number of walls is 5
  sorry

/-- The number of walls in the room -/
def solution : ℕ := 5

/-- Proves that the calculated number of walls matches the solution -/
theorem proof_correct : number_of_walls 2 3 (1/10) 10 5 = solution := by sorry

end number_of_walls_proof_correct_l2559_255961


namespace grid_drawing_theorem_l2559_255953

/-- Represents a grid configuration -/
structure GridConfiguration (n : ℕ+) :=
  (has_diagonal : Fin n → Fin n → Bool)
  (start_vertex : Fin n × Fin n)
  (is_valid : Bool)

/-- Checks if a grid configuration is valid according to the problem conditions -/
def is_valid_configuration (n : ℕ+) (config : GridConfiguration n) : Prop :=
  -- Adjacent cells have diagonals in different directions
  (∀ i j, config.has_diagonal i j → ¬(config.has_diagonal (i+1) j ∧ config.has_diagonal i (j+1))) ∧
  -- Can be drawn in one stroke starting from bottom-left vertex
  (config.start_vertex = (0, 0)) ∧
  -- Each edge or diagonal is traversed exactly once
  config.is_valid

/-- The main theorem stating that only n = 1, 2, 3 satisfy the conditions -/
theorem grid_drawing_theorem :
  ∀ n : ℕ+, (∃ config : GridConfiguration n, is_valid_configuration n config) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end grid_drawing_theorem_l2559_255953


namespace symmetry_implies_m_sqrt3_l2559_255996

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line mx - y + 1 = 0 -/
def line_of_symmetry (m : ℝ) (p : Point) : Prop :=
  m * p.x - p.y + 1 = 0

/-- The line x + y = 0 -/
def line_xy (p : Point) : Prop :=
  p.x + p.y = 0

/-- Two points are symmetric with respect to a line -/
def symmetric_points (m : ℝ) (p q : Point) : Prop :=
  ∃ (mid : Point), line_of_symmetry m mid ∧
    mid.x = (p.x + q.x) / 2 ∧
    mid.y = (p.y + q.y) / 2

theorem symmetry_implies_m_sqrt3 :
  ∀ (m : ℝ) (N : Point),
    symmetric_points m (Point.mk 1 0) N →
    line_xy N →
    m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by sorry

end symmetry_implies_m_sqrt3_l2559_255996


namespace max_value_x_plus_2y_l2559_255952

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 - 2*x + 4*y = 5) : 
  (∃ (z : ℝ), x + 2*y ≤ z) ∧ (∀ (w : ℝ), x + 2*y ≤ w → 9/2 ≤ w) :=
by sorry

end max_value_x_plus_2y_l2559_255952


namespace intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l2559_255964

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 7 ≤ 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x ≤ 7} := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  (∀ m : ℝ, (B m).Nonempty → (B m ⊆ A ∧ B m ≠ A)) ↔ 2 ≤ m ∧ m ≤ 4 := by sorry

end intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l2559_255964


namespace m_range_theorem_l2559_255937

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (-2 ≤ m ∧ m < 1) ∨ m > 2

/-- Condition p: The solution set of x^2 + mx + 1 < 0 is empty -/
def condition_p (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 ≥ 0

/-- Condition q: The function 4x^2 + 4(m-1)x + 3 has no extreme value -/
def condition_q (m : ℝ) : Prop :=
  ∀ x, 8*x + 4*(m-1) ≠ 0

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ)
  (h1 : condition_p m ∨ condition_q m)
  (h2 : ¬(condition_p m ∧ condition_q m)) :
  m_range m :=
sorry

end m_range_theorem_l2559_255937


namespace tan_seventeen_pi_fourths_l2559_255970

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end tan_seventeen_pi_fourths_l2559_255970


namespace sin_600_plus_tan_240_l2559_255999

theorem sin_600_plus_tan_240 : Real.sin (600 * π / 180) + Real.tan (240 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_600_plus_tan_240_l2559_255999


namespace tangent_line_at_one_max_value_min_value_f_properties_l2559_255941

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one : 
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1)) :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1 :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27 :=
sorry

-- Theorem combining all results
theorem f_properties :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1))) ∧
  (∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1) ∧
  (∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27) :=
sorry

end tangent_line_at_one_max_value_min_value_f_properties_l2559_255941


namespace compound_ratio_l2559_255902

theorem compound_ratio (total_weight : ℝ) (weight_B : ℝ) :
  total_weight = 108 →
  weight_B = 90 →
  let weight_A := total_weight - weight_B
  (weight_A / weight_B) = (1 / 5 : ℝ) := by
sorry

end compound_ratio_l2559_255902


namespace salt_water_fraction_l2559_255957

theorem salt_water_fraction (small_capacity large_capacity : ℝ) 
  (h1 : large_capacity = 5 * small_capacity)
  (h2 : 0.3 * large_capacity = 0.2 * large_capacity + small_capacity * x) : x = 1/2 := by
  sorry

#check salt_water_fraction

end salt_water_fraction_l2559_255957


namespace larger_ball_radius_larger_ball_radius_proof_l2559_255954

theorem larger_ball_radius : ℝ → Prop :=
  fun r : ℝ =>
    -- Volume of a sphere: (4/3) * π * r^3
    let volume_sphere (radius : ℝ) := (4/3) * Real.pi * (radius ^ 3)
    -- Volume of 10 balls with radius 2
    let volume_ten_balls := 10 * volume_sphere 2
    -- Volume of 2 balls with radius 1
    let volume_two_small_balls := 2 * volume_sphere 1
    -- Volume of the larger ball with radius r
    let volume_larger_ball := volume_sphere r
    -- The total volume equality
    volume_ten_balls = volume_larger_ball + volume_two_small_balls →
    -- The radius of the larger ball is ∛78
    r = Real.rpow 78 (1/3)

-- The proof is omitted
theorem larger_ball_radius_proof : larger_ball_radius (Real.rpow 78 (1/3)) := by sorry

end larger_ball_radius_larger_ball_radius_proof_l2559_255954


namespace complex_sum_problem_l2559_255960

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 1 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = -Complex.I → 
  d + f = -2 := by
  sorry

end complex_sum_problem_l2559_255960


namespace square_root_product_plus_one_l2559_255983

theorem square_root_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 33 * 32 * 31 + 1) = 1055 := by sorry

end square_root_product_plus_one_l2559_255983


namespace inequality_must_hold_l2559_255914

theorem inequality_must_hold (a b c : ℝ) (h : a > b ∧ b > c) : a - |c| > b - |c| := by
  sorry

end inequality_must_hold_l2559_255914


namespace probability_of_three_in_six_sevenths_l2559_255982

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem probability_of_three_in_six_sevenths : 
  let rep := decimal_representation 6 7
  ∀ k, k ∈ rep → k ≠ 3 :=
sorry

end probability_of_three_in_six_sevenths_l2559_255982


namespace x_range_l2559_255998

theorem x_range (x : ℝ) (h1 : x^2 - 8*x + 12 < 0) (h2 : x > 3) : 3 < x ∧ x < 6 := by
  sorry

end x_range_l2559_255998


namespace rectangular_prism_volume_l2559_255900

/-- The volume of a rectangular prism with length:width:height ratio of 4:3:1 and height √2 cm is 24√2 cm³ -/
theorem rectangular_prism_volume (height : ℝ) (length width : ℝ) : 
  height = Real.sqrt 2 →
  length = 4 * height →
  width = 3 * height →
  length * width * height = 24 * Real.sqrt 2 := by
sorry

end rectangular_prism_volume_l2559_255900


namespace no_50_cell_crossing_l2559_255979

/-- The maximum number of cells a straight line can cross on an m × n grid -/
def maxCrossedCells (m n : ℕ) : ℕ := m + n - Nat.gcd m n

/-- Theorem: On a 20 × 30 grid, it's impossible to draw a straight line that crosses 50 cells -/
theorem no_50_cell_crossing :
  maxCrossedCells 20 30 < 50 := by
  sorry

end no_50_cell_crossing_l2559_255979


namespace greater_number_proof_l2559_255974

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x + y = 40) (h3 : x - y = 10) : x = 25 := by
  sorry

end greater_number_proof_l2559_255974


namespace monomial_coefficient_l2559_255965

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The monomial -9/4 * x^2 * y -/
def monomial (x y : ℚ) : ℚ := -9/4 * x^2 * y

theorem monomial_coefficient :
  coefficient (-9/4) x y = -9/4 :=
by sorry

end monomial_coefficient_l2559_255965


namespace circle_equation_sum_l2559_255917

/-- Given a circle equation, prove the sum of center coordinates and radius -/
theorem circle_equation_sum (x y : ℝ) :
  (∀ x y, x^2 + 14*y + 65 = -y^2 - 8*x) →
  ∃ a b r : ℝ,
    (∀ x y, (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -11 :=
by sorry

end circle_equation_sum_l2559_255917


namespace inequality_proof_l2559_255977

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b = (a + 1/a^3)/2) 
  (hc : c = (b + 1/b^3)/2) 
  (hb_lt_1 : b < 1) : 
  1 < c ∧ c < a :=
by sorry

end inequality_proof_l2559_255977


namespace frequency_converges_to_probability_l2559_255933

-- Define a random event
def RandomEvent : Type := Unit

-- Define the probability of the event
def probability (e : RandomEvent) : ℝ := sorry

-- Define the observed frequency of the event after n experiments
def observedFrequency (e : RandomEvent) (n : ℕ) : ℝ := sorry

-- Statement: As the number of experiments increases, the frequency of the random event
-- will gradually stabilize at the probability of the random event occurring.
theorem frequency_converges_to_probability (e : RandomEvent) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |observedFrequency e n - probability e| < ε :=
sorry

end frequency_converges_to_probability_l2559_255933


namespace n_squared_plus_inverse_squared_plus_six_l2559_255990

theorem n_squared_plus_inverse_squared_plus_six (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by
  sorry

end n_squared_plus_inverse_squared_plus_six_l2559_255990


namespace quadratic_solution_property_l2559_255972

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
sorry

end quadratic_solution_property_l2559_255972


namespace intersection_points_count_l2559_255985

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if a point (x, y) lies on a line --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem --/
def line1 : Line := { a := -3, b := 4, c := 2 }
def line2 : Line := { a := 2, b := 4, c := 4 }
def line3 : Line := { a := 6, b := -8, c := 3 }

theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ p2 ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (∀ (p : ℝ × ℝ),
      p ≠ p1 → p ≠ p2 →
      ¬((point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line2) ∨
        (point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line3) ∨
        (point_on_line p.1 p.2 line2 ∧ point_on_line p.1 p.2 line3))) :=
by
  sorry

end intersection_points_count_l2559_255985


namespace fraction_evaluation_l2559_255918

theorem fraction_evaluation : 
  let numerator := (12^4 + 288) * (24^4 + 288) * (36^4 + 288) * (48^4 + 288) * (60^4 + 288)
  let denominator := (6^4 + 288) * (18^4 + 288) * (30^4 + 288) * (42^4 + 288) * (54^4 + 288)
  numerator / denominator = -332 := by
sorry

end fraction_evaluation_l2559_255918


namespace census_population_scientific_notation_l2559_255915

/-- 
Given a positive integer n, its scientific notation is a representation of the form a × 10^b, 
where 1 ≤ a < 10 and b is an integer.
-/
def scientific_notation (n : ℕ+) : ℝ × ℤ := sorry

theorem census_population_scientific_notation :
  scientific_notation 932700 = (9.327, 5) := by sorry

end census_population_scientific_notation_l2559_255915


namespace dice_probability_l2559_255903

def num_dice : ℕ := 6
def num_success : ℕ := 3
def prob_success : ℚ := 1/3
def prob_failure : ℚ := 2/3

theorem dice_probability : 
  (Nat.choose num_dice num_success : ℚ) * prob_success^num_success * prob_failure^(num_dice - num_success) = 160/729 := by
  sorry

end dice_probability_l2559_255903


namespace fraction_simplification_l2559_255951

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : (5 * x) / (x - 1) - 5 / (x - 1) = 5 := by
  sorry

end fraction_simplification_l2559_255951


namespace minimum_students_in_class_l2559_255980

theorem minimum_students_in_class (b g : ℕ) : 
  (b ≠ 0 ∧ g ≠ 0) →  -- Ensure non-zero numbers of boys and girls
  (2 * (b / 2) = 3 * (g / 3)) →  -- Half of boys equals two-thirds of girls who passed
  (b / 2 = 2 * (g / 3)) →  -- Boys who failed is twice girls who failed
  7 ≤ b + g  -- The total number of students is at least 7
  ∧ ∃ (b' g' : ℕ), b' ≠ 0 ∧ g' ≠ 0 
     ∧ (2 * (b' / 2) = 3 * (g' / 3))
     ∧ (b' / 2 = 2 * (g' / 3))
     ∧ b' + g' = 7  -- There exists a solution with exactly 7 students
  := by sorry

end minimum_students_in_class_l2559_255980


namespace linear_relationship_scaling_l2559_255973

/-- Given a linear relationship where an increase of 4 units in x results in an increase of 10 units in y,
    prove that an increase of 12 units in x results in an increase of 30 units in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 4) - f x = 10) :
  ∀ x : ℝ, f (x + 12) - f x = 30 := by
sorry

end linear_relationship_scaling_l2559_255973


namespace pole_length_reduction_l2559_255997

theorem pole_length_reduction (original_length : ℝ) (reduction_percentage : ℝ) (new_length : ℝ) :
  original_length = 20 →
  reduction_percentage = 30 →
  new_length = original_length * (1 - reduction_percentage / 100) →
  new_length = 14 :=
by sorry

end pole_length_reduction_l2559_255997


namespace equation_solution_l2559_255955

theorem equation_solution : ∃ x : ℚ, 3 * x - 6 = |(-21 + 8 - 3)| ∧ x = 22 / 3 := by
  sorry

end equation_solution_l2559_255955


namespace perpendicular_necessary_not_sufficient_l2559_255988

-- Define the types for lines and relationships
def Line : Type := ℝ → ℝ → Prop
def Perpendicular (l₁ l₂ : Line) : Prop := sorry
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (a b c : Line) (h : Perpendicular a b) : 
  (∀ (a b c : Line), Parallel b c → Perpendicular a c) ∧ 
  (∃ (a b c : Line), Perpendicular a c ∧ ¬Parallel b c) := by
  sorry

end perpendicular_necessary_not_sufficient_l2559_255988


namespace kangaroo_meeting_count_l2559_255931

def kangaroo_a_period : ℕ := 9
def kangaroo_b_period : ℕ := 6
def total_jumps : ℕ := 2017

def meeting_count (a_period b_period total_jumps : ℕ) : ℕ :=
  let lcm := Nat.lcm a_period b_period
  let meetings_per_cycle := 2  -- They meet twice in each LCM cycle
  let complete_cycles := total_jumps / lcm
  let remainder := total_jumps % lcm
  let meetings_in_complete_cycles := complete_cycles * meetings_per_cycle
  let initial_meeting := 1  -- They start at the same point
  let extra_meeting := if remainder ≥ 1 then 1 else 0
  meetings_in_complete_cycles + initial_meeting + extra_meeting

theorem kangaroo_meeting_count :
  meeting_count kangaroo_a_period kangaroo_b_period total_jumps = 226 := by
  sorry

end kangaroo_meeting_count_l2559_255931


namespace fraction_and_decimal_representation_l2559_255910

theorem fraction_and_decimal_representation :
  (7 : ℚ) / 16 = 7 / 16 ∧ (100.45 : ℝ) = 100 + 4/10 + 5/100 :=
by sorry

end fraction_and_decimal_representation_l2559_255910


namespace shortest_side_length_l2559_255949

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: The radius is positive -/
  r_pos : r > 0
  /-- Condition: The segments are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Condition: The shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: In a triangle with an inscribed circle of radius 5 units, 
    where one side is divided into segments of 9 and 5 units by the point of tangency, 
    the length of the shortest side is 16 units. -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5)
    (h2 : t.a = 9)
    (h3 : t.b = 5) : 
  t.shortest_side = 16 := by
  sorry

end shortest_side_length_l2559_255949


namespace circle_intersection_range_l2559_255932

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the intersection condition
def intersect (a : ℝ) : Prop := ∃ x y : ℝ, circle1 a x y ∧ circle2 x y

-- Define the range of a
def valid_range (a : ℝ) : Prop := (a > -6 ∧ a < -4) ∨ (a > 4 ∧ a < 6)

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect a ↔ valid_range a := by sorry

end circle_intersection_range_l2559_255932


namespace league_games_count_l2559_255919

theorem league_games_count (n : ℕ) (h : n = 14) : 
  (n * (n - 1)) / 2 = 91 := by
  sorry

#check league_games_count

end league_games_count_l2559_255919


namespace smallest_number_with_properties_l2559_255991

/-- Given a prime number p where 2p+1 is a cube of a natural number,
    find the smallest natural number N that is divisible by p,
    ends with p, and has a digit sum equal to p. -/
theorem smallest_number_with_properties (p : ℕ) (h_prime : Nat.Prime p)
  (h_cube : ∃ n : ℕ, 2 * p + 1 = n^3) :
  let N := 11713
  (N % p = 0) ∧
  (N % 100 = p) ∧
  (Nat.digits 10 N).sum = p ∧
  (∀ m : ℕ, m < N →
    (m % p = 0) ∧ (m % 100 = p) ∧ (Nat.digits 10 m).sum = p → False) ∧
  (p = 13) := by
sorry

end smallest_number_with_properties_l2559_255991


namespace min_rb_selling_price_theorem_l2559_255901

/-- Represents the fruit sales problem -/
structure FruitSales where
  total_weight : ℝ
  total_cost : ℝ
  rb_purchase_price : ℝ
  rb_selling_price_last_week : ℝ
  xg_purchase_price : ℝ
  xg_selling_price : ℝ
  rb_damage_rate : ℝ

/-- Calculates the profit from last week's sales -/
def last_week_profit (fs : FruitSales) : ℝ := sorry

/-- Calculates the minimum selling price for Red Beauty this week -/
def min_rb_selling_price_this_week (fs : FruitSales) : ℝ := sorry

/-- Theorem stating the minimum selling price of Red Beauty this week -/
theorem min_rb_selling_price_theorem (fs : FruitSales) 
  (h1 : fs.total_weight = 300)
  (h2 : fs.total_cost = 3000)
  (h3 : fs.rb_purchase_price = 20)
  (h4 : fs.rb_selling_price_last_week = 35)
  (h5 : fs.xg_purchase_price = 5)
  (h6 : fs.xg_selling_price = 10)
  (h7 : fs.rb_damage_rate = 0.1)
  : min_rb_selling_price_this_week fs ≥ 36.7 ∧ 
    last_week_profit fs = 2500 := by sorry

end min_rb_selling_price_theorem_l2559_255901


namespace parabola_point_comparison_l2559_255946

/-- Theorem: For a parabola y = ax^2 - 4ax + 2 where a > 0, 
    and points (-1, y₁) and (1, y₂) on the parabola, y₁ > y₂ -/
theorem parabola_point_comparison 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (y₁ y₂ : ℝ) 
  (h_y₁ : y₁ = a * (-1)^2 - 4 * a * (-1) + 2) 
  (h_y₂ : y₂ = a * 1^2 - 4 * a * 1 + 2) : 
  y₁ > y₂ := by
  sorry

end parabola_point_comparison_l2559_255946


namespace recipe_flour_calculation_l2559_255909

/-- The amount of flour required for a recipe -/
def recipe_flour (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem recipe_flour_calculation (initial : ℕ) (additional : ℕ) :
  recipe_flour initial additional = initial + additional :=
by sorry

end recipe_flour_calculation_l2559_255909


namespace correct_answer_l2559_255943

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

-- Theorem to prove
theorem correct_answer : p ∨ (¬q) := by sorry

end correct_answer_l2559_255943


namespace anderson_shirts_theorem_l2559_255950

theorem anderson_shirts_theorem (total_clothing pieces_of_trousers : ℕ) 
  (h1 : total_clothing = 934)
  (h2 : pieces_of_trousers = 345) :
  total_clothing - pieces_of_trousers = 589 := by
  sorry

end anderson_shirts_theorem_l2559_255950


namespace product_equality_l2559_255976

theorem product_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := by
sorry

end product_equality_l2559_255976


namespace gary_chickens_l2559_255929

theorem gary_chickens (initial_chickens : ℕ) : 
  (∃ (current_chickens : ℕ), 
    current_chickens = 8 * initial_chickens ∧ 
    6 * 7 * current_chickens = 1344) → 
  initial_chickens = 4 := by
sorry

end gary_chickens_l2559_255929


namespace common_point_sum_mod_9_l2559_255989

theorem common_point_sum_mod_9 : ∃ (x : ℤ), 
  (∀ (y : ℤ), (y ≡ 3*x + 5 [ZMOD 9] ↔ y ≡ 7*x + 3 [ZMOD 9])) ∧ 
  (x ≡ 5 [ZMOD 9]) := by
  sorry

end common_point_sum_mod_9_l2559_255989


namespace gardening_club_membership_l2559_255908

theorem gardening_club_membership (initial_total : ℕ) 
  (h1 : initial_total > 0)
  (h2 : (60 : ℚ) / 100 * initial_total = (initial_total * 3) / 5) 
  (h3 : (((initial_total * 3) / 5 - 3 : ℚ) / initial_total) = 1 / 2) : 
  (initial_total * 3) / 5 = 18 := by
sorry

end gardening_club_membership_l2559_255908


namespace a_minus_b_equals_plus_minus_eight_l2559_255986

theorem a_minus_b_equals_plus_minus_eight (a b : ℚ) : 
  (|a| = 5) → (|b| = 3) → (a * b < 0) → (a - b = 8 ∨ a - b = -8) := by
  sorry

end a_minus_b_equals_plus_minus_eight_l2559_255986


namespace least_sum_m_n_l2559_255940

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬∃ k : ℕ, m = k * n) :
  m + n ≥ 377 ∧ ∃ m' n' : ℕ+, m' + n' = 377 ∧ 
    Nat.gcd (m' + n') 330 = 1 ∧ 
    (∃ k : ℕ, (m' : ℕ)^(m' : ℕ) = k * (n' : ℕ)^(n' : ℕ)) ∧ 
    ¬∃ k : ℕ, (m' : ℕ) = k * (n' : ℕ) :=
by sorry

end least_sum_m_n_l2559_255940


namespace football_gear_cost_l2559_255935

theorem football_gear_cost (x : ℝ) 
  (h1 : x + x = 2 * x)  -- Shorts + T-shirt costs twice as much as shorts
  (h2 : x + 4 * x = 5 * x)  -- Shorts + boots costs five times as much as shorts
  (h3 : x + 2 * x = 3 * x)  -- Shorts + shin guards costs three times as much as shorts
  : x + x + 4 * x + 2 * x = 8 * x :=  -- Total cost is 8 times the cost of shorts
by sorry

end football_gear_cost_l2559_255935


namespace polygon_sides_from_diagonals_l2559_255944

theorem polygon_sides_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 3)) / 2 = 15 ∧ 
  (∀ (m : ℕ), m > 2 → (m * (m - 3)) / 2 = 15 → m = n) ∧
  n = 7 := by
sorry

end polygon_sides_from_diagonals_l2559_255944


namespace remaining_money_l2559_255924

def base_8_to_10 (n : Nat) : Nat :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : Nat := 5377
def airline_ticket : Nat := 1200
def travel_pass : Nat := 600

theorem remaining_money :
  base_8_to_10 savings - airline_ticket - travel_pass = 1015 := by
  sorry

end remaining_money_l2559_255924


namespace inequality_and_equality_condition_l2559_255978

theorem inequality_and_equality_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  (a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) ≥ (a + b + c)^2 / 12) ∧
  ((a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) = (a + b + c)^2 / 12) ↔ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end inequality_and_equality_condition_l2559_255978


namespace alia_has_40_markers_l2559_255923

-- Define the number of markers for each person
def steve_markers : ℕ := 60
def austin_markers : ℕ := steve_markers / 3
def alia_markers : ℕ := 2 * austin_markers

-- Theorem statement
theorem alia_has_40_markers : alia_markers = 40 := by
  sorry

end alia_has_40_markers_l2559_255923
