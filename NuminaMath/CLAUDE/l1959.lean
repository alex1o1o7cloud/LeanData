import Mathlib

namespace broken_seashells_l1959_195969

/-- Given the total number of seashells and the number of unbroken seashells,
    calculate the number of broken seashells. -/
theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h : unbroken ≤ total) :
  total - unbroken = total - unbroken :=
by sorry

end broken_seashells_l1959_195969


namespace upstream_journey_distance_l1959_195907

/-- Calculates the effective speed of a boat traveling upstream -/
def effectiveSpeed (boatSpeed currentSpeed : ℝ) : ℝ :=
  boatSpeed - currentSpeed

/-- Calculates the distance traveled in one hour given the effective speed -/
def distanceTraveled (effectiveSpeed : ℝ) : ℝ :=
  effectiveSpeed * 1

theorem upstream_journey_distance 
  (boatSpeed : ℝ) 
  (currentSpeed1 currentSpeed2 currentSpeed3 : ℝ) 
  (h1 : boatSpeed = 50)
  (h2 : currentSpeed1 = 10)
  (h3 : currentSpeed2 = 20)
  (h4 : currentSpeed3 = 15) :
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed1) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed2) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed3) = 105 := by
  sorry

end upstream_journey_distance_l1959_195907


namespace solution_value_l1959_195908

theorem solution_value (a : ℚ) : 
  (∃ x : ℚ, x = -2 ∧ 2 * x + 3 * a = 0) → a = 4/3 := by
  sorry

end solution_value_l1959_195908


namespace intersection_when_m_neg_three_subset_condition_equivalent_l1959_195946

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem for the first part of the problem
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the second part of the problem
theorem subset_condition_equivalent :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end intersection_when_m_neg_three_subset_condition_equivalent_l1959_195946


namespace smallest_odd_four_primes_with_13_l1959_195976

def is_prime (n : ℕ) : Prop := sorry

def prime_factors (n : ℕ) : Finset ℕ := sorry

theorem smallest_odd_four_primes_with_13 :
  ∀ n : ℕ,
  n % 2 = 1 →
  (prime_factors n).card = 4 →
  13 ∈ prime_factors n →
  n ≥ 1365 :=
sorry

end smallest_odd_four_primes_with_13_l1959_195976


namespace min_sum_squares_roots_l1959_195953

/-- The sum of squares of roots of x^2 - (m+1)x + m - 1 = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let sum_squares := (m + 1)^2 - 2*(m - 1)
  ∀ k : ℝ, sum_squares ≤ (k + 1)^2 - 2*(k - 1) → m = 0 :=
by sorry

end min_sum_squares_roots_l1959_195953


namespace imaginary_part_of_complex_fraction_l1959_195932

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + i)^2 / (1 - i)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1959_195932


namespace polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l1959_195992

/-- A real coefficient polynomial f(x) = x^4 + (a-1)x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + (a-1)*x^2 + 1

/-- Theorem: If f(x) is non-negative for all real x, then a ≥ -1 -/
theorem polynomial_nonnegative_implies_a_range (a : ℝ) 
  (h : ∀ x : ℝ, f a x ≥ 0) : a ≥ -1 := by
  sorry

/-- Theorem: If a ≥ -1, then f(x) is non-negative for all real x -/
theorem a_range_implies_polynomial_nonnegative (a : ℝ) 
  (h : a ≥ -1) : ∀ x : ℝ, f a x ≥ 0 := by
  sorry

end polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l1959_195992


namespace helen_to_betsy_win_ratio_l1959_195996

/-- The ratio of Helen's wins to Betsy's wins in a Monopoly game scenario -/
theorem helen_to_betsy_win_ratio :
  ∀ (helen_wins : ℕ),
  let betsy_wins : ℕ := 5
  let susan_wins : ℕ := 3 * betsy_wins
  let total_wins : ℕ := 30
  (betsy_wins + helen_wins + susan_wins = total_wins) →
  (helen_wins : ℚ) / betsy_wins = 2 := by
    sorry

end helen_to_betsy_win_ratio_l1959_195996


namespace contacts_per_dollar_theorem_l1959_195975

/-- Represents a box of contacts with quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box -/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.quantity / box.price

/-- Theorem stating that the number of contacts equal to $1 worth in the box 
    with the lower cost per contact is 3 -/
theorem contacts_per_dollar_theorem (box1 box2 : ContactBox) 
  (h1 : box1.quantity = 50 ∧ box1.price = 25)
  (h2 : box2.quantity = 99 ∧ box2.price = 33) :
  let betterBox := if contactsPerDollar box1 > contactsPerDollar box2 then box1 else box2
  contactsPerDollar betterBox = 3 := by
  sorry

end contacts_per_dollar_theorem_l1959_195975


namespace tina_career_result_l1959_195997

def boxer_career (initial_wins : ℕ) (additional_wins1 : ℕ) (triple_factor : ℕ) (additional_wins2 : ℕ) (double_factor : ℕ) : ℕ × ℕ :=
  let wins1 := initial_wins + additional_wins1
  let wins2 := wins1 * triple_factor
  let wins3 := wins2 + additional_wins2
  let final_wins := wins3 * double_factor
  let losses := 3
  (final_wins, losses)

theorem tina_career_result :
  let (wins, losses) := boxer_career 10 5 3 7 2
  wins - losses = 131 := by sorry

end tina_career_result_l1959_195997


namespace alyssa_cookie_count_l1959_195965

/-- The number of cookies Alyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference between Aiyanna's and Alyssa's cookies -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end alyssa_cookie_count_l1959_195965


namespace points_collinear_opposite_collinear_k_l1959_195979

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-zero vectors a and b
variable (a b : V)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hnc : ¬ ∃ (r : ℝ), a = r • b)

-- Define vectors AB, BC, and CD
def AB : V := a + b
def BC : V := 2 • a + 8 • b
def CD : V := 3 • (a - b)

-- Define collinearity
def collinear (u v : V) : Prop := ∃ (r : ℝ), u = r • v

-- Theorem 1: Points A, B, D are collinear
theorem points_collinear : 
  ∃ (r : ℝ), AB a b = r • (AB a b + BC a b + CD a b) :=
sorry

-- Theorem 2: Value of k for opposite collinearity
theorem opposite_collinear_k : 
  ∃ (k : ℝ), k = -1 ∧ 
  (∃ (r : ℝ), r < 0 ∧ k • a + b = r • (a + k • b)) :=
sorry

end points_collinear_opposite_collinear_k_l1959_195979


namespace inequality_solution_l1959_195909

theorem inequality_solution (x : ℝ) : 
  (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0 ↔ -3 < x ∧ x < 3 := by
sorry

end inequality_solution_l1959_195909


namespace equality_of_four_reals_l1959_195962

theorem equality_of_four_reals (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a → a = b ∧ b = c ∧ c = d := by
  sorry

end equality_of_four_reals_l1959_195962


namespace sector_central_angle_l1959_195930

/-- Given a circular sector with area 4 and arc length 4, its central angle is 2 radians. -/
theorem sector_central_angle (area : ℝ) (arc_length : ℝ) (radius : ℝ) (angle : ℝ) :
  area = 4 →
  arc_length = 4 →
  area = (1 / 2) * radius * arc_length →
  arc_length = radius * angle →
  angle = 2 := by sorry

end sector_central_angle_l1959_195930


namespace deepak_age_l1959_195998

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end deepak_age_l1959_195998


namespace pig_cure_probability_l1959_195917

theorem pig_cure_probability (p : ℝ) (n k : ℕ) (h_p : p = 0.9) (h_n : n = 5) (h_k : k = 3) :
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = (Nat.choose 5 3 : ℝ) * 0.9^3 * 0.1^2 :=
sorry

end pig_cure_probability_l1959_195917


namespace complex_fraction_product_l1959_195934

theorem complex_fraction_product (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 : ℂ) + 7 * Complex.I = (a + b * Complex.I) * (2 - Complex.I) →
  a * b = -3 := by
sorry

end complex_fraction_product_l1959_195934


namespace simplify_fraction_product_l1959_195981

theorem simplify_fraction_product : (2 / (2 + Real.sqrt 3)) * (2 / (2 - Real.sqrt 3)) = 4 := by
  sorry

end simplify_fraction_product_l1959_195981


namespace sum_cube_over_power_of_three_l1959_195956

open Real BigOperators

/-- The sum of the infinite series $\sum_{k=1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{16}$. -/
theorem sum_cube_over_power_of_three :
  ∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ) = 39 / 16 := by sorry

end sum_cube_over_power_of_three_l1959_195956


namespace mark_fruit_theorem_l1959_195947

/-- The number of fruit pieces Mark kept for next week -/
def fruit_kept_for_next_week (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) : ℕ :=
  initial_fruit - pieces_eaten_four_days - pieces_for_friday

theorem mark_fruit_theorem (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) 
  (h1 : initial_fruit = 10)
  (h2 : pieces_eaten_four_days = 5)
  (h3 : pieces_for_friday = 3) :
  fruit_kept_for_next_week initial_fruit pieces_eaten_four_days pieces_for_friday = 2 := by
  sorry

end mark_fruit_theorem_l1959_195947


namespace parametric_elimination_l1959_195952

theorem parametric_elimination (x y t : ℝ) 
  (hx : x = 1 + 2 * t - 2 * t^2) 
  (hy : y = 2 * (1 + t) * Real.sqrt (1 - t^2)) : 
  y^4 + 2 * y^2 * (x^2 - 12 * x + 9) + x^4 + 8 * x^3 + 18 * x^2 - 27 = 0 := by
  sorry

end parametric_elimination_l1959_195952


namespace sandy_mall_change_l1959_195972

/-- The change Sandy received after buying clothes at the mall -/
def sandys_change (pants_cost shirt_cost bill_amount : ℚ) : ℚ :=
  bill_amount - (pants_cost + shirt_cost)

/-- Theorem stating that Sandy's change is $2.51 given the problem conditions -/
theorem sandy_mall_change :
  sandys_change 9.24 8.25 20 = 2.51 := by
  sorry

end sandy_mall_change_l1959_195972


namespace barbell_cost_is_270_l1959_195951

/-- The cost of each barbell given the total amount paid, change received, and number of barbells purchased. -/
def barbell_cost (total_paid : ℕ) (change : ℕ) (num_barbells : ℕ) : ℕ :=
  (total_paid - change) / num_barbells

/-- Theorem stating that the cost of each barbell is $270 under the given conditions. -/
theorem barbell_cost_is_270 :
  barbell_cost 850 40 3 = 270 := by
  sorry

end barbell_cost_is_270_l1959_195951


namespace tank_capacity_correct_l1959_195978

/-- Represents the tank filling problem -/
structure TankProblem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ
  cycle_duration : ℕ
  total_time : ℕ

/-- The specific tank problem instance -/
def tankInstance : TankProblem :=
  { capacity := 850,
    pipeA_rate := 40,
    pipeB_rate := 30,
    pipeC_rate := 20,
    cycle_duration := 3,
    total_time := 51 }

/-- Calculates the net amount filled in one cycle -/
def netFillPerCycle (t : TankProblem) : ℕ :=
  t.pipeA_rate + t.pipeB_rate - t.pipeC_rate

/-- Theorem stating that the given tank instance has the correct capacity -/
theorem tank_capacity_correct (t : TankProblem) : 
  t = tankInstance → 
  t.capacity = (t.total_time / t.cycle_duration) * netFillPerCycle t :=
by
  sorry

end tank_capacity_correct_l1959_195978


namespace division_remainder_proof_l1959_195982

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) 
    (h1 : dividend = 131)
    (h2 : divisor = 14)
    (h3 : quotient = 9)
    (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 5 := by
  sorry

end division_remainder_proof_l1959_195982


namespace intersection_and_perpendicular_line_l1959_195988

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem proves that the line ax + by + c = 0 passes through
    the intersection point of l₁ and l₂, and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line
  (l₁ : Real → Real → Prop) (l₂ : Real → Real → Prop) (l₃ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 10 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x + 4 * y - 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3 * x - 2 * y + 5 = 0)
  : ∃ x y, l₁ x y ∧ l₂ x y ∧ 2 * x + 3 * y - 2 = 0 ∧
    (∀ x₁ y₁ x₂ y₂, l₃ x₁ y₁ → l₃ x₂ y₂ → (y₂ - y₁) * (3 * (x₂ - x₁)) = -2 * (y₂ - y₁)) :=
sorry

end intersection_and_perpendicular_line_l1959_195988


namespace corner_sum_9x9_l1959_195961

def checkerboard_size : ℕ := 9

def corner_sum (n : ℕ) : ℕ :=
  1 + n + (n^2 - n + 1) + n^2

theorem corner_sum_9x9 :
  corner_sum checkerboard_size = 164 :=
by sorry

end corner_sum_9x9_l1959_195961


namespace log_plus_fraction_gt_one_l1959_195960

theorem log_plus_fraction_gt_one (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) :
  Real.log x + a / (x - 1) > 1 := by sorry

end log_plus_fraction_gt_one_l1959_195960


namespace stating_ball_falls_in_hole_iff_rational_ratio_l1959_195990

/-- Represents a rectangular pool table with sides of lengths a and b -/
structure PoolTable where
  a : ℝ
  b : ℝ

/-- Predicate to check if a ratio is rational -/
def isRational (x : ℝ) : Prop := ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- 
  Theorem stating that a ball shot from one corner along the angle bisector 
  will eventually fall into a hole at one of the other three corners 
  if and only if the ratio of side lengths is rational
-/
theorem ball_falls_in_hole_iff_rational_ratio (table : PoolTable) : 
  (∃ (k l : ℤ), k ≠ 0 ∧ l ≠ 0 ∧ table.a * k = table.b * l) ↔ isRational (table.a / table.b) := by
  sorry


end stating_ball_falls_in_hole_iff_rational_ratio_l1959_195990


namespace points_needed_for_average_increase_l1959_195943

/-- Represents a basketball player's scoring history -/
structure PlayerStats where
  gamesPlayed : ℕ
  totalPoints : ℕ

/-- Calculates the average points per game -/
def averagePoints (stats : PlayerStats) : ℚ :=
  stats.totalPoints / stats.gamesPlayed

/-- Updates player stats after a game -/
def updateStats (stats : PlayerStats) (points : ℕ) : PlayerStats :=
  { gamesPlayed := stats.gamesPlayed + 1
  , totalPoints := stats.totalPoints + points }

/-- Theorem: A player who raised their average from 20 to 21 by scoring 36 points
    must score 38 points to raise their average to 22 -/
theorem points_needed_for_average_increase 
  (initialStats : PlayerStats)
  (h1 : averagePoints initialStats = 20)
  (h2 : averagePoints (updateStats initialStats 36) = 21) :
  averagePoints (updateStats (updateStats initialStats 36) 38) = 22 := by
  sorry


end points_needed_for_average_increase_l1959_195943


namespace ones_digit_of_8_to_47_l1959_195912

def ones_digit_cycle : List Nat := [8, 4, 2, 6]

theorem ones_digit_of_8_to_47 (h : ones_digit_cycle = [8, 4, 2, 6]) :
  (8^47 : ℕ) % 10 = 2 := by
  sorry

end ones_digit_of_8_to_47_l1959_195912


namespace sweets_problem_l1959_195987

theorem sweets_problem (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  ∃ total_sweets : ℕ,
    total_sweets = num_children * sweets_per_child / (1 - remaining_fraction) ∧
    total_sweets = 288 := by
  sorry

end sweets_problem_l1959_195987


namespace quadratic_inequality_empty_solution_l1959_195980

theorem quadratic_inequality_empty_solution (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end quadratic_inequality_empty_solution_l1959_195980


namespace hyperbola_and_line_equations_l1959_195958

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 4/3 * x ∨ y = -4/3 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the line passing through the right focus
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 5 * Real.sqrt 3 = 0

-- State the theorem
theorem hyperbola_and_line_equations :
  (∀ x y : ℝ, ellipse x y → asymptotes x y → hyperbola x y) ∧
  (∀ x y : ℝ, ellipse x y → line x y) := by sorry

end hyperbola_and_line_equations_l1959_195958


namespace seashells_given_to_mike_l1959_195931

theorem seashells_given_to_mike (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 79)
  (h2 : remaining_seashells = 16) :
  initial_seashells - remaining_seashells = 63 := by
  sorry

end seashells_given_to_mike_l1959_195931


namespace cosine_equality_l1959_195974

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (331 * π / 180) → n = 29 := by
  sorry

end cosine_equality_l1959_195974


namespace common_material_choices_eq_120_l1959_195923

/-- The number of ways to choose r items from n items --/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items --/
def arrange (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common --/
def commonMaterialChoices : ℕ :=
  choose 6 1 * arrange 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end common_material_choices_eq_120_l1959_195923


namespace liters_to_pints_conversion_l1959_195941

/-- Given that 0.75 liters is approximately 1.575 pints, prove that 3 liters is equal to 6.3 pints. -/
theorem liters_to_pints_conversion (liter_to_pint : ℝ → ℝ) 
  (h : liter_to_pint 0.75 = 1.575) : liter_to_pint 3 = 6.3 := by
  sorry

end liters_to_pints_conversion_l1959_195941


namespace three_fractions_l1959_195935

-- Define the list of expressions
def expressions : List String := [
  "3/a",
  "(a+b)/7",
  "x^2 + (1/2)y^2",
  "5",
  "1/(x-1)",
  "x/(8π)",
  "x^2/x"
]

-- Define what constitutes a fraction
def is_fraction (expr : String) : Prop :=
  ∃ (num denom : String), 
    expr = num ++ "/" ++ denom ∧ 
    denom ≠ "1" ∧
    ¬∃ (simplified : String), simplified ≠ expr ∧ ¬(∃ (n d : String), simplified = n ++ "/" ++ d)

-- Theorem stating that exactly 3 expressions are fractions
theorem three_fractions : 
  ∃ (fracs : List String), 
    fracs.length = 3 ∧ 
    (∀ expr ∈ fracs, expr ∈ expressions ∧ is_fraction expr) ∧
    (∀ expr ∈ expressions, is_fraction expr → expr ∈ fracs) :=
sorry

end three_fractions_l1959_195935


namespace coefficient_d_value_l1959_195989

-- Define the polynomial Q(x)
def Q (x d : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + d*x + 15

-- State the theorem
theorem coefficient_d_value :
  ∃ d : ℝ, (∀ x : ℝ, Q x d = 0 → x = -3) ∧ d = 11 := by
  sorry

end coefficient_d_value_l1959_195989


namespace union_A_B_intersect_complement_A_B_l1959_195929

/-- The set A -/
def A : Set ℝ := {x | x < -5 ∨ x > 1}

/-- The set B -/
def B : Set ℝ := {x | -4 < x ∧ x < 3}

/-- Theorem: The union of A and B -/
theorem union_A_B : A ∪ B = {x : ℝ | x < -5 ∨ x > -4} := by sorry

/-- Theorem: The intersection of the complement of A and B -/
theorem intersect_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | -4 < x ∧ x ≤ 1} := by sorry

end union_A_B_intersect_complement_A_B_l1959_195929


namespace rectangle_triangle_count_l1959_195973

theorem rectangle_triangle_count (n m : ℕ) (hn : n = 6) (hm : m = 7) :
  n.choose 2 * m + m.choose 2 * n = 231 := by
  sorry

end rectangle_triangle_count_l1959_195973


namespace cube_root_and_square_root_l1959_195999

theorem cube_root_and_square_root (x y : ℝ) 
  (h1 : (x - 1) ^ (1/3 : ℝ) = 2) 
  (h2 : (y + 2) ^ (1/2 : ℝ) = 3) : 
  x - 2*y = -5 := by sorry

end cube_root_and_square_root_l1959_195999


namespace probability_of_defective_product_l1959_195920

/-- Given a set of products with some defective ones, calculate the probability of selecting a defective product -/
theorem probability_of_defective_product 
  (total : ℕ) 
  (defective : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : defective ≤ total) : 
  (defective : ℚ) / total = 3 / 10 :=
by sorry

end probability_of_defective_product_l1959_195920


namespace class_composition_l1959_195957

theorem class_composition (total : ℕ) (girls boys : ℕ) : 
  girls = (6 : ℚ) / 10 * total →
  (girls - 1 : ℚ) / (total - 3) = 25 / 40 →
  girls = 21 ∧ boys = 14 := by
  sorry

end class_composition_l1959_195957


namespace goldfish_preference_total_l1959_195995

theorem goldfish_preference_total : 
  let johnson_class := 30
  let johnson_ratio := (1 : ℚ) / 6
  let feldstein_class := 45
  let feldstein_ratio := (2 : ℚ) / 3
  let henderson_class := 36
  let henderson_ratio := (1 : ℚ) / 5
  let dias_class := 50
  let dias_ratio := (3 : ℚ) / 5
  let norris_class := 25
  let norris_ratio := (2 : ℚ) / 5
  ⌊johnson_class * johnson_ratio⌋ +
  ⌊feldstein_class * feldstein_ratio⌋ +
  ⌊henderson_class * henderson_ratio⌋ +
  ⌊dias_class * dias_ratio⌋ +
  ⌊norris_class * norris_ratio⌋ = 82 :=
by sorry


end goldfish_preference_total_l1959_195995


namespace brick_length_calculation_l1959_195902

/-- Calculates the length of a brick given wall and brick specifications --/
theorem brick_length_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_count : ℕ) (brick_width brick_height : ℝ) :
  wall_length = 10 ∧ wall_width = 4 ∧ wall_height = 5 ∧
  mortar_percentage = 0.1 ∧ brick_count = 6000 ∧
  brick_width = 15 ∧ brick_height = 8 →
  ∃ (brick_length : ℝ),
    brick_length = 250 ∧
    (wall_length * wall_width * wall_height * (1 - mortar_percentage) * 1000000) =
    (brick_length * brick_width * brick_height * brick_count) :=
by sorry

end brick_length_calculation_l1959_195902


namespace samantha_buys_four_toys_l1959_195991

/-- Represents the price of a dog toy in cents -/
def toy_price : ℕ := 1200

/-- Represents the total amount spent on dog toys in cents -/
def total_spent : ℕ := 3600

/-- Calculates the cost of a pair of toys under the "buy one get one half off" promotion -/
def pair_cost : ℕ := toy_price + toy_price / 2

/-- Represents the number of toys Samantha buys -/
def num_toys : ℕ := (total_spent / pair_cost) * 2

theorem samantha_buys_four_toys : num_toys = 4 := by
  sorry

end samantha_buys_four_toys_l1959_195991


namespace nested_square_root_value_l1959_195905

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end nested_square_root_value_l1959_195905


namespace sin_1050_degrees_l1959_195916

theorem sin_1050_degrees : Real.sin (1050 * Real.pi / 180) = -1/2 := by
  sorry

end sin_1050_degrees_l1959_195916


namespace best_fit_model_l1959_195971

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |1 - model.r_squared| ≤ |1 - m.r_squared|

theorem best_fit_model :
  let models : List RegressionModel := [
    ⟨"Model 1", 0.25⟩,
    ⟨"Model 2", 0.50⟩,
    ⟨"Model 3", 0.98⟩,
    ⟨"Model 4", 0.80⟩
  ]
  let model3 : RegressionModel := ⟨"Model 3", 0.98⟩
  has_best_fit model3 models := by sorry

end best_fit_model_l1959_195971


namespace dawson_group_size_l1959_195959

/-- The number of people in a group given the total cost and cost per person -/
def group_size (total_cost : ℕ) (cost_per_person : ℕ) : ℕ :=
  total_cost / cost_per_person

/-- Proof that the group size is 15 given the specific costs -/
theorem dawson_group_size :
  group_size 13500 900 = 15 := by
  sorry

end dawson_group_size_l1959_195959


namespace cody_tickets_l1959_195966

theorem cody_tickets (initial : ℝ) (lost_bet : ℝ) (spent_beanie : ℝ) (won_game : ℝ) (dropped : ℝ)
  (h1 : initial = 56.5)
  (h2 : lost_bet = 6.3)
  (h3 : spent_beanie = 25.75)
  (h4 : won_game = 10.25)
  (h5 : dropped = 3.1) :
  initial - lost_bet - spent_beanie + won_game - dropped = 31.6 := by
  sorry

end cody_tickets_l1959_195966


namespace net_change_is_correct_l1959_195954

/-- Calculates the final price after applying two percentage changes -/
def apply_price_changes (original_price : ℚ) (change1 : ℚ) (change2 : ℚ) : ℚ :=
  original_price * (1 + change1) * (1 + change2)

/-- Represents the store inventory with original prices and price changes -/
structure Inventory where
  tv_price : ℚ
  tv_change1 : ℚ
  tv_change2 : ℚ
  fridge_price : ℚ
  fridge_change1 : ℚ
  fridge_change2 : ℚ
  washer_price : ℚ
  washer_change1 : ℚ
  washer_change2 : ℚ

/-- Calculates the net change in total prices -/
def net_change (inv : Inventory) : ℚ :=
  let final_tv_price := apply_price_changes inv.tv_price inv.tv_change1 inv.tv_change2
  let final_fridge_price := apply_price_changes inv.fridge_price inv.fridge_change1 inv.fridge_change2
  let final_washer_price := apply_price_changes inv.washer_price inv.washer_change1 inv.washer_change2
  let total_final_price := final_tv_price + final_fridge_price + final_washer_price
  let total_original_price := inv.tv_price + inv.fridge_price + inv.washer_price
  total_final_price - total_original_price

theorem net_change_is_correct (inv : Inventory) : 
  inv.tv_price = 500 ∧ 
  inv.tv_change1 = -1/5 ∧ 
  inv.tv_change2 = 9/20 ∧
  inv.fridge_price = 1000 ∧ 
  inv.fridge_change1 = 7/20 ∧ 
  inv.fridge_change2 = -3/20 ∧
  inv.washer_price = 750 ∧ 
  inv.washer_change1 = 1/10 ∧ 
  inv.washer_change2 = -1/5 
  → net_change inv = 275/2 := by
  sorry

#eval net_change { 
  tv_price := 500, tv_change1 := -1/5, tv_change2 := 9/20,
  fridge_price := 1000, fridge_change1 := 7/20, fridge_change2 := -3/20,
  washer_price := 750, washer_change1 := 1/10, washer_change2 := -1/5
}

end net_change_is_correct_l1959_195954


namespace gas_used_for_appointments_l1959_195927

def distance_to_dermatologist : ℝ := 30
def distance_to_gynecologist : ℝ := 50
def car_efficiency : ℝ := 20

theorem gas_used_for_appointments : 
  (2 * distance_to_dermatologist + 2 * distance_to_gynecologist) / car_efficiency = 8 := by
  sorry

end gas_used_for_appointments_l1959_195927


namespace sum_of_roots_range_l1959_195913

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 else Real.exp x

-- Define the function F as [f(x)]^2
def F (x : ℝ) : ℝ := (f x)^2

-- Define the property that F(x) = a has exactly two roots
def has_two_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ = a ∧ F x₂ = a ∧ ∀ x, F x = a → x = x₁ ∨ x = x₂

-- Theorem statement
theorem sum_of_roots_range (a : ℝ) (h : has_two_roots a) :
  ∃ x₁ x₂, F x₁ = a ∧ F x₂ = a ∧ x₁ + x₂ > -1 ∧ ∀ M, ∃ b > a, 
  ∃ y₁ y₂, F y₁ = b ∧ F y₂ = b ∧ y₁ + y₂ > M :=
sorry

end

end sum_of_roots_range_l1959_195913


namespace bottle_caps_per_box_l1959_195900

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) :
  total_caps / num_boxes = 4 := by
  sorry

end bottle_caps_per_box_l1959_195900


namespace g_lower_bound_l1959_195949

theorem g_lower_bound (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m < 1) :
  Real.exp (m * x - 1) - (Real.log x + 1) / m > m^(1/m) - m^(-1/m) := by
  sorry

end g_lower_bound_l1959_195949


namespace power_function_above_identity_l1959_195964

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) (hα : α < 1) : x^α > x := by
  sorry

end power_function_above_identity_l1959_195964


namespace bracelet_pairing_impossibility_l1959_195955

theorem bracelet_pairing_impossibility (n : ℕ) (h : n = 100) :
  ¬ ∃ (arrangement : List (Finset (Fin n))),
    (∀ s ∈ arrangement, s.card = 3) ∧
    (∀ i j : Fin n, i ≠ j → 
      (arrangement.filter (λ s => i ∈ s ∧ j ∈ s)).length = 1) :=
by
  sorry

end bracelet_pairing_impossibility_l1959_195955


namespace sum_a4_a5_a6_l1959_195993

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_a2_a3 : a 2 + a 3 = 13
  a1_eq_2 : a 1 = 2

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_a4_a5_a6 (seq : ArithmeticSequence) : seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end sum_a4_a5_a6_l1959_195993


namespace noah_holidays_per_month_l1959_195924

/-- The number of holidays Noah takes in a year -/
def total_holidays : ℕ := 36

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Noah takes each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem noah_holidays_per_month :
  holidays_per_month = 3 := by sorry

end noah_holidays_per_month_l1959_195924


namespace set_equality_l1959_195904

theorem set_equality (A : Set ℕ) : 
  ({1, 3} : Set ℕ) ⊆ A ∧ ({1, 3} : Set ℕ) ∪ A = {1, 3, 5} → A = {1, 3, 5} := by
  sorry

end set_equality_l1959_195904


namespace john_total_spent_l1959_195970

def silver_amount : ℝ := 1.5
def gold_amount : ℝ := 2 * silver_amount
def silver_price_per_ounce : ℝ := 20
def gold_price_per_ounce : ℝ := 50 * silver_price_per_ounce

def total_spent : ℝ := silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

theorem john_total_spent :
  total_spent = 3030 := by sorry

end john_total_spent_l1959_195970


namespace simplest_quadratic_radical_l1959_195903

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def isSimplestQuadraticRadical (n : ℝ) : Prop :=
  n > 0 ∧ ¬(isPerfectSquare n) ∧ ∀ m : ℝ, m > 1 → ¬(isPerfectSquare (n / (m * m)))

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical 6 ∧
  ¬(isSimplestQuadraticRadical 4) ∧
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 12) :=
by sorry

end simplest_quadratic_radical_l1959_195903


namespace dot_product_CA_CB_l1959_195919

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the center of the circle C
def center_C : ℝ × ℝ := (0, 2)

-- Define a point B on the circle C
def point_B : ℝ × ℝ := sorry

-- State that line l is tangent to circle C at point B
axiom tangent_line : (point_B.1 - point_A.1) * (point_B.1 - center_C.1) + 
                     (point_B.2 - point_A.2) * (point_B.2 - center_C.2) = 0

-- The main theorem
theorem dot_product_CA_CB : 
  (point_A.1 - center_C.1) * (point_B.1 - center_C.1) + 
  (point_A.2 - center_C.2) * (point_B.2 - center_C.2) = 5 :=
sorry

end dot_product_CA_CB_l1959_195919


namespace constant_term_expansion_l1959_195918

/-- The constant term in the expansion of x(1 - 2/√x)^6 is 60 -/
theorem constant_term_expansion : ℕ := by
  sorry

end constant_term_expansion_l1959_195918


namespace circle_area_increase_l1959_195915

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end circle_area_increase_l1959_195915


namespace tan_sum_identity_l1959_195939

theorem tan_sum_identity : 
  Real.tan (25 * π / 180) + Real.tan (35 * π / 180) + 
  Real.sqrt 3 * Real.tan (25 * π / 180) * Real.tan (35 * π / 180) = 1 := by
  sorry

end tan_sum_identity_l1959_195939


namespace biathlon_run_distance_l1959_195901

/-- Biathlon problem -/
theorem biathlon_run_distance
  (total_distance : ℝ)
  (bicycle_distance : ℝ)
  (bicycle_velocity : ℝ)
  (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : bicycle_distance = 145)
  (h3 : bicycle_velocity = 29)
  (h4 : total_time = 6)
  : total_distance - bicycle_distance = 10 := by
  sorry

end biathlon_run_distance_l1959_195901


namespace sum_of_fourth_powers_of_roots_l1959_195948

theorem sum_of_fourth_powers_of_roots (P : ℝ → ℝ) (r₁ r₂ : ℝ) : 
  P = (fun x ↦ x^2 + 2*x + 3) →
  P r₁ = 0 →
  P r₂ = 0 →
  r₁^4 + r₂^4 = -14 := by
  sorry

end sum_of_fourth_powers_of_roots_l1959_195948


namespace doris_earnings_l1959_195968

/-- Calculates the number of weeks needed for Doris to earn enough to cover her monthly expenses --/
def weeks_to_earn_expenses (hourly_rate : ℚ) (weekday_hours : ℚ) (saturday_hours : ℚ) (monthly_expense : ℚ) : ℚ :=
  let weekly_hours := 5 * weekday_hours + saturday_hours
  let weekly_earnings := weekly_hours * hourly_rate
  monthly_expense / weekly_earnings

theorem doris_earnings : 
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  weeks_to_earn_expenses hourly_rate weekday_hours saturday_hours monthly_expense = 3 := by
  sorry

end doris_earnings_l1959_195968


namespace equation_solution_l1959_195984

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 6*x*y - 18*y + 3*x - 9 = 0) ↔ x = 3 := by
  sorry

end equation_solution_l1959_195984


namespace linear_equation_solve_l1959_195922

theorem linear_equation_solve (x y : ℝ) : 
  x + 2 * y = 6 → y = (-x + 6) / 2 := by
  sorry

end linear_equation_solve_l1959_195922


namespace cube_root_sum_equation_l1959_195945

theorem cube_root_sum_equation (x : ℝ) :
  x = (11 + Real.sqrt 337) ^ (1/3 : ℝ) + (11 - Real.sqrt 337) ^ (1/3 : ℝ) →
  x^3 + 18*x = 22 := by
sorry

end cube_root_sum_equation_l1959_195945


namespace trig_identity_proof_l1959_195928

theorem trig_identity_proof : 
  4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end trig_identity_proof_l1959_195928


namespace cards_distribution_l1959_195936

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end cards_distribution_l1959_195936


namespace angle_sum_is_pi_over_two_l1959_195914

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) (h_eq : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : α + β = π / 2 := by
  sorry

end angle_sum_is_pi_over_two_l1959_195914


namespace blue_balls_removal_l1959_195942

theorem blue_balls_removal (total : ℕ) (red_percent : ℚ) (target_red_percent : ℚ) 
  (h1 : total = 120) 
  (h2 : red_percent = 2/5) 
  (h3 : target_red_percent = 3/4) : 
  ∃ (removed : ℕ), 
    removed = 56 ∧ 
    (red_percent * total : ℚ) / (total - removed : ℚ) = target_red_percent := by
  sorry

end blue_balls_removal_l1959_195942


namespace divisibility_by_12321_l1959_195921

theorem divisibility_by_12321 (a : ℤ) : 
  (∃ k : ℕ, 12321 ∣ (a^k + 1)) ↔ 
  (∃ n : ℤ, a ≡ 11 [ZMOD 111] ∨ 
            a ≡ 41 [ZMOD 111] ∨ 
            a ≡ 62 [ZMOD 111] ∨ 
            a ≡ 65 [ZMOD 111] ∨ 
            a ≡ 77 [ZMOD 111] ∨ 
            a ≡ 95 [ZMOD 111] ∨ 
            a ≡ 101 [ZMOD 111] ∨ 
            a ≡ 104 [ZMOD 111] ∨ 
            a ≡ 110 [ZMOD 111]) :=
by sorry

end divisibility_by_12321_l1959_195921


namespace right_triangle_consecutive_sides_l1959_195950

theorem right_triangle_consecutive_sides (a c : ℕ) (h1 : c = a + 1) :
  ∃ b : ℕ, b * b = c + a ∧ c * c = a * a + b * b := by
  sorry

end right_triangle_consecutive_sides_l1959_195950


namespace hall_wallpaper_expenditure_l1959_195906

/-- Calculates the total expenditure for covering the walls and ceiling of a rectangular hall with wallpaper. -/
def total_expenditure (length width height cost_per_sqm : ℚ) : ℚ :=
  let wall_area := 2 * (length * height + width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for covering a 30m x 25m x 10m hall with wallpaper costing Rs. 75 per square meter is Rs. 138,750. -/
theorem hall_wallpaper_expenditure :
  total_expenditure 30 25 10 75 = 138750 := by
  sorry

end hall_wallpaper_expenditure_l1959_195906


namespace loss_per_metre_is_five_l1959_195940

/-- Calculates the loss per metre of cloth given the total metres sold, 
    total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_metres

/-- Proves that the loss per metre is 5 given the specified conditions. -/
theorem loss_per_metre_is_five : 
  loss_per_metre 500 18000 41 = 5 := by
  sorry

#eval loss_per_metre 500 18000 41

end loss_per_metre_is_five_l1959_195940


namespace logarithm_identity_l1959_195926

theorem logarithm_identity : Real.log 5 ^ 2 + Real.log 2 * Real.log 50 = 1 := by
  sorry

end logarithm_identity_l1959_195926


namespace line_equation_correct_l1959_195963

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a point (x, y) satisfies the equation of the line -/
def Line.satisfiesEquation (l : Line) (x y : ℝ) : Prop :=
  2 * x - y - 5 = 0

theorem line_equation_correct (l : Line) :
  l.slope = 2 ∧ l.point = (3, 1) →
  ∀ x y : ℝ, l.satisfiesEquation x y ↔ y - 1 = l.slope * (x - 3) :=
sorry

end line_equation_correct_l1959_195963


namespace projection_magnitude_l1959_195994

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem projection_magnitude (k : ℝ) 
  (h : (a.1 + (b k).1, a.2 + (b k).2) • a = 0) : 
  |(a.1 * (b k).1 + a.2 * (b k).2) / Real.sqrt ((b k).1^2 + (b k).2^2)| = 1 :=
sorry

end projection_magnitude_l1959_195994


namespace inverse_proportion_points_l1959_195938

/-- Given that (-1,2) and (2,a) lie on the graph of y = k/x, prove that a = -1 -/
theorem inverse_proportion_points (k a : ℝ) : 
  (2 = k / (-1)) → (a = k / 2) → a = -1 := by sorry

end inverse_proportion_points_l1959_195938


namespace license_plate_difference_l1959_195977

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters at the beginning of a California license plate -/
def ca_prefix_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def ca_digits : ℕ := 3

/-- The number of letters at the end of a California license plate -/
def ca_suffix_letters : ℕ := 2

/-- The number of letters in a Texas license plate -/
def tx_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def tx_digits : ℕ := 4

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : 
  (num_letters ^ (ca_prefix_letters + ca_suffix_letters) * num_digits ^ ca_digits) - 
  (num_letters ^ tx_letters * num_digits ^ tx_digits) = 301093376000 := by
  sorry

end license_plate_difference_l1959_195977


namespace f_neg_two_eq_neg_two_fifths_l1959_195983

noncomputable def g (x : ℝ) : ℝ := 3 - x^2

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else (3 - (g⁻¹ x)^2) / (g⁻¹ x)^2

theorem f_neg_two_eq_neg_two_fifths : f (-2) = -2/5 := by
  sorry

end f_neg_two_eq_neg_two_fifths_l1959_195983


namespace sqrt_sum_equality_l1959_195925

theorem sqrt_sum_equality : Real.sqrt 18 + Real.sqrt 24 / Real.sqrt 3 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equality_l1959_195925


namespace x_value_proof_l1959_195944

theorem x_value_proof (x : ℝ) : (-1 : ℝ) * 2 * x * 4 = 24 → x = -3 := by
  sorry

end x_value_proof_l1959_195944


namespace sales_goals_calculation_l1959_195985

/-- Represents the sales data for a candy store employee over three days. -/
structure SalesData :=
  (jetBarGoal : ℕ)
  (zippyBarGoal : ℕ)
  (candyCloudGoal : ℕ)
  (mondayJetBars : ℕ)
  (mondayZippyBars : ℕ)
  (mondayCandyClouds : ℕ)
  (tuesdayJetBarsDiff : ℤ)
  (tuesdayZippyBarsDiff : ℕ)
  (wednesdayCandyCloudsMultiplier : ℕ)

/-- Calculates the remaining sales needed to reach the weekly goals. -/
def remainingSales (data : SalesData) : ℤ × ℤ × ℤ :=
  let totalJetBars := data.mondayJetBars + (data.mondayJetBars : ℤ) + data.tuesdayJetBarsDiff
  let totalZippyBars := data.mondayZippyBars + data.mondayZippyBars + data.tuesdayZippyBarsDiff
  let totalCandyClouds := data.mondayCandyClouds + data.mondayCandyClouds * data.wednesdayCandyCloudsMultiplier
  ((data.jetBarGoal : ℤ) - totalJetBars,
   (data.zippyBarGoal : ℤ) - (totalZippyBars : ℤ),
   (data.candyCloudGoal : ℤ) - (totalCandyClouds : ℤ))

theorem sales_goals_calculation (data : SalesData)
  (h1 : data.jetBarGoal = 90)
  (h2 : data.zippyBarGoal = 70)
  (h3 : data.candyCloudGoal = 50)
  (h4 : data.mondayJetBars = 45)
  (h5 : data.mondayZippyBars = 34)
  (h6 : data.mondayCandyClouds = 16)
  (h7 : data.tuesdayJetBarsDiff = -16)
  (h8 : data.tuesdayZippyBarsDiff = 8)
  (h9 : data.wednesdayCandyCloudsMultiplier = 2) :
  remainingSales data = (16, -6, 2) :=
by sorry


end sales_goals_calculation_l1959_195985


namespace three_squares_representation_l1959_195967

theorem three_squares_representation (N : ℕ) :
  (∃ a b c : ℤ, N = (3*a)^2 + (3*b)^2 + (3*c)^2) →
  (∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end three_squares_representation_l1959_195967


namespace sequence_general_term_l1959_195986

-- Define the sequence and its partial sum
def S (n : ℕ) : ℤ := n^2 - 4*n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 2*n - 5

-- Theorem statement
theorem sequence_general_term (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n :=
sorry

end sequence_general_term_l1959_195986


namespace carbonic_acid_molecular_weight_l1959_195933

/-- The molecular weight of carbonic acid in grams per mole. -/
def molecular_weight_carbonic_acid : ℝ := 62

/-- The number of moles of carbonic acid in the given sample. -/
def moles_carbonic_acid : ℝ := 8

/-- The total weight of the given sample of carbonic acid in grams. -/
def total_weight_carbonic_acid : ℝ := 496

/-- Theorem stating that the molecular weight of carbonic acid is 62 grams/mole,
    given that 8 moles of carbonic acid weigh 496 grams. -/
theorem carbonic_acid_molecular_weight :
  molecular_weight_carbonic_acid = total_weight_carbonic_acid / moles_carbonic_acid :=
by sorry

end carbonic_acid_molecular_weight_l1959_195933


namespace smallest_sum_of_three_smallest_sum_is_negative_six_l1959_195937

def S : Finset Int := {0, 5, -2, 18, -4, 3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y ∧ y ≠ z ∧ x ≠ z → 
  a + b + c ≤ x + y + z :=
by sorry

theorem smallest_sum_is_negative_six :
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = -6 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y ∧ y ≠ z ∧ x ≠ z → 
   a + b + c ≤ x + y + z) :=
by sorry

end smallest_sum_of_three_smallest_sum_is_negative_six_l1959_195937


namespace F_36_72_equals_48_max_F_happy_pair_equals_58_l1959_195911

/-- Function F calculates the sum of products of digits in two-digit numbers -/
def F (m n : ℕ) : ℕ :=
  (m / 10) * (n % 10) + (m % 10) * (n / 10)

/-- Swaps the digits of a two-digit number -/
def swapDigits (m : ℕ) : ℕ :=
  (m % 10) * 10 + (m / 10)

/-- Checks if two numbers form a "happy pair" -/
def isHappyPair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 5 ∧
  m = 21 * a + b ∧ n = 53 + b ∧
  (swapDigits m + 5 * (n % 10)) % 11 = 0

theorem F_36_72_equals_48 : F 36 72 = 48 := by sorry

theorem max_F_happy_pair_equals_58 :
  (∃ (m n : ℕ), isHappyPair m n ∧ F m n = 58) ∧
  (∀ (m n : ℕ), isHappyPair m n → F m n ≤ 58) := by sorry

end F_36_72_equals_48_max_F_happy_pair_equals_58_l1959_195911


namespace complex_expression_simplification_l1959_195910

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) + 2*i * (3 - 2*i) = 16 := by
  sorry

end complex_expression_simplification_l1959_195910
