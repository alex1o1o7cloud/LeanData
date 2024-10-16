import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2515_251571

theorem expression_evaluation (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hsum : x + 1/y ≠ 0) : 
  (x^2 + 1/y^2) / (x + 1/y) = x - 1/y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2515_251571


namespace NUMINAMATH_CALUDE_complex_modulus_l2515_251567

theorem complex_modulus (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l2515_251567


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2515_251560

theorem arithmetic_expression_evaluation : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2515_251560


namespace NUMINAMATH_CALUDE_ship_speed_ratio_l2515_251574

theorem ship_speed_ratio (downstream_speed upstream_speed average_speed : ℝ) 
  (h1 : downstream_speed / upstream_speed = 5 / 2) 
  (h2 : average_speed = (2 * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)) : 
  average_speed / downstream_speed = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_ratio_l2515_251574


namespace NUMINAMATH_CALUDE_age_sum_problem_l2515_251510

theorem age_sum_problem (a b : ℕ) (h1 : a > b) (h2 : a * b * b * b = 256) : 
  a + b + b + b = 38 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l2515_251510


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2515_251564

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + x - 2 < 0 ↔ -1 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2515_251564


namespace NUMINAMATH_CALUDE_parallel_line_not_through_point_l2515_251545

-- Define a line in 2D space
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

def is_point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

def are_lines_parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

theorem parallel_line_not_through_point 
  (L : Line) (P : Point) (h : ¬ is_point_on_line P L) :
  ∃ (L' : Line), 
    are_lines_parallel L' L ∧ 
    ¬ is_point_on_line P L' ∧
    L'.A = L.A ∧ 
    L'.B = L.B ∧ 
    L'.C = L.C + (L.A * P.x + L.B * P.y + L.C) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_not_through_point_l2515_251545


namespace NUMINAMATH_CALUDE_symmetry_implies_x_equals_one_l2515_251507

/-- A function f: ℝ → ℝ has symmetric graphs for y = f(x-1) and y = f(1-x) with respect to x = 1 -/
def has_symmetric_graphs (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = f (1 - x)

/-- If a function has symmetric graphs for y = f(x-1) and y = f(1-x), 
    then they are symmetric with respect to x = 1 -/
theorem symmetry_implies_x_equals_one (f : ℝ → ℝ) 
    (h : has_symmetric_graphs f) : 
    ∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, f (a + (x - a)) = f (a - (x - a)) :=
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_x_equals_one_l2515_251507


namespace NUMINAMATH_CALUDE_train_average_speed_l2515_251592

/-- Proves that the average speed of a train is 22.5 kmph, given specific travel conditions. -/
theorem train_average_speed 
  (x : ℝ) 
  (h₁ : x > 0)  -- Ensuring x is positive for meaningful distance
  (speed₁ : ℝ) (speed₂ : ℝ)
  (h₂ : speed₁ = 30) -- First speed in kmph
  (h₃ : speed₂ = 20) -- Second speed in kmph
  (distance₁ : ℝ) (distance₂ : ℝ)
  (h₄ : distance₁ = x) -- First distance
  (h₅ : distance₂ = 2 * x) -- Second distance
  (total_distance : ℝ)
  (h₆ : total_distance = distance₁ + distance₂) -- Total distance
  : 
  (total_distance / ((distance₁ / speed₁) + (distance₂ / speed₂))) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l2515_251592


namespace NUMINAMATH_CALUDE_wooden_block_stacks_height_difference_l2515_251582

/-- The height of wooden block stacks problem -/
theorem wooden_block_stacks_height_difference :
  let first_stack : ℕ := 7
  let second_stack : ℕ := first_stack + 3
  let third_stack : ℕ := second_stack - 6
  let fifth_stack : ℕ := 2 * second_stack
  let total_blocks : ℕ := 55
  let other_stacks_total : ℕ := first_stack + second_stack + third_stack + fifth_stack
  let fourth_stack : ℕ := total_blocks - other_stacks_total
  fourth_stack - third_stack = 10 :=
by sorry

end NUMINAMATH_CALUDE_wooden_block_stacks_height_difference_l2515_251582


namespace NUMINAMATH_CALUDE_cats_problem_l2515_251515

/-- Given an initial number of cats and the number of female and male kittens,
    calculate the total number of cats. -/
def total_cats (initial : ℕ) (female_kittens : ℕ) (male_kittens : ℕ) : ℕ :=
  initial + female_kittens + male_kittens

/-- Theorem stating that given 2 initial cats, 3 female kittens, and 2 male kittens,
    the total number of cats is 7. -/
theorem cats_problem : total_cats 2 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cats_problem_l2515_251515


namespace NUMINAMATH_CALUDE_fraction_power_multiplication_compute_fraction_power_l2515_251557

theorem fraction_power_multiplication (a b c : ℚ) (n : ℕ) :
  a * (b / c)^n = (a * b^n) / c^n :=
by sorry

theorem compute_fraction_power : 7 * (1 / 5)^3 = 7 / 125 :=
by sorry

end NUMINAMATH_CALUDE_fraction_power_multiplication_compute_fraction_power_l2515_251557


namespace NUMINAMATH_CALUDE_equation_solution_l2515_251590

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3

-- Theorem statement
theorem equation_solution :
  ∃ (x : ℝ), x > 9 ∧ equation x ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2515_251590


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l2515_251554

/-- The number of distinct sequences possible when flipping a coin n times -/
def coinFlipSequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences possible when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coinFlipSequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l2515_251554


namespace NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l2515_251585

/-- Given that i is the imaginary unit and (x+i)i = y-i where x and y are real numbers,
    prove that the point (x, y) lies in the third quadrant of the complex plane. -/
theorem complex_point_in_third_quadrant (x y : ℝ) (i : ℂ) 
  (h_i : i * i = -1) 
  (h_eq : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l2515_251585


namespace NUMINAMATH_CALUDE_simplify_complex_radical_expression_l2515_251598

theorem simplify_complex_radical_expression :
  (3 * (Real.sqrt 5 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (414 - 98 * Real.sqrt 35) / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_radical_expression_l2515_251598


namespace NUMINAMATH_CALUDE_puzzle_solution_l2515_251537

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) Nat

-- Define the constraint for black dots (ratio of 2)
def blackDotConstraint (a b : Nat) : Prop := a = 2 * b ∨ b = 2 * a

-- Define the constraint for white dots (difference of 1)
def whiteDotConstraint (a b : Nat) : Prop := a = b + 1 ∨ b = a + 1

-- Define the property of having no repeated numbers in a row or column
def noRepeats (g : Grid) : Prop :=
  ∀ i j : Fin 6, i ≠ j → 
    (∀ k : Fin 6, g i k ≠ g j k) ∧ 
    (∀ k : Fin 6, g k i ≠ g k j)

-- Define the property that all numbers are between 1 and 6
def validNumbers (g : Grid) : Prop :=
  ∀ i j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

-- Define the specific constraints for this puzzle
def puzzleConstraints (g : Grid) : Prop :=
  blackDotConstraint (g 0 0) (g 0 1) ∧
  whiteDotConstraint (g 0 4) (g 0 5) ∧
  blackDotConstraint (g 1 2) (g 1 3) ∧
  whiteDotConstraint (g 2 1) (g 2 2) ∧
  blackDotConstraint (g 3 0) (g 3 1) ∧
  whiteDotConstraint (g 3 2) (g 3 3) ∧
  blackDotConstraint (g 4 4) (g 4 5) ∧
  whiteDotConstraint (g 5 3) (g 5 4)

-- Theorem statement
theorem puzzle_solution :
  ∀ g : Grid,
    noRepeats g →
    validNumbers g →
    puzzleConstraints g →
    g 3 0 = 2 ∧ g 3 1 = 1 ∧ g 3 2 = 4 ∧ g 3 3 = 3 ∧ g 3 4 = 6 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2515_251537


namespace NUMINAMATH_CALUDE_mateen_backyard_area_l2515_251584

/-- A rectangular backyard with specific walking distances -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The conditions of Mateen's backyard -/
def mateen_backyard : Backyard where
  length := 40
  width := 10
  total_distance := 1200
  length_walks := 30
  perimeter_walks := 12

/-- Theorem stating the area of Mateen's backyard -/
theorem mateen_backyard_area :
  let b := mateen_backyard
  b.length * b.width = 400 ∧
  b.length_walks * b.length = b.total_distance ∧
  b.perimeter_walks * (2 * b.length + 2 * b.width) = b.total_distance :=
by sorry

end NUMINAMATH_CALUDE_mateen_backyard_area_l2515_251584


namespace NUMINAMATH_CALUDE_pastry_combinations_linda_pastry_purchase_l2515_251544

theorem pastry_combinations : ℕ → ℕ → ℕ
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem linda_pastry_purchase : pastry_combinations 4 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pastry_combinations_linda_pastry_purchase_l2515_251544


namespace NUMINAMATH_CALUDE_ladybug_count_l2515_251501

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l2515_251501


namespace NUMINAMATH_CALUDE_horner_method_v4_l2515_251506

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

theorem horner_method_v4 :
  let x := -4
  let v₄ := horner (f.take 5).reverse x
  v₄ = 220 := by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l2515_251506


namespace NUMINAMATH_CALUDE_vector_properties_l2515_251559

/-- Given points in a 2D Cartesian coordinate system -/
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![1, 2]
def B : Fin 2 → ℝ := ![-3, 4]

/-- Vector AB -/
def vecAB : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

/-- Theorem stating properties of vectors and angles in the given problem -/
theorem vector_properties :
  (vecAB 0 = -4 ∧ vecAB 1 = 2) ∧
  Real.sqrt ((vecAB 0)^2 + (vecAB 1)^2) = 2 * Real.sqrt 5 ∧
  ((A 0 * B 0 + A 1 * B 1) / (Real.sqrt (A 0^2 + A 1^2) * Real.sqrt (B 0^2 + B 1^2))) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l2515_251559


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l2515_251570

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := Nat.factorial (n - 1) * Nat.factorial 2

/-- The number of ways to arrange n people in a row where three specific people are together -/
def arrangementsWithThreeTogether (n : ℕ) : ℕ := Nat.factorial (n - 2) * Nat.factorial 3

/-- The number of ways to arrange 9 people in a row where three specific people cannot sit next to each other -/
theorem arrangements_with_restriction : 
  totalArrangements 9 - (3 * arrangementsWithTwoTogether 9 - arrangementsWithThreeTogether 9) = 181200 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l2515_251570


namespace NUMINAMATH_CALUDE_group_size_l2515_251555

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The original weight of each person in kg -/
def original_weight : ℝ := 50

/-- The weight of the new person in kg -/
def new_person_weight : ℝ := 70

/-- The average weight increase in kg -/
def average_increase : ℝ := 2.5

theorem group_size :
  (n : ℝ) * (original_weight + average_increase) = n * original_weight + (new_person_weight - original_weight) →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_l2515_251555


namespace NUMINAMATH_CALUDE_a_earnings_l2515_251531

-- Define the work rates and total wages
def a_rate : ℚ := 1 / 10
def b_rate : ℚ := 1 / 15
def total_wages : ℚ := 3400

-- Define A's share of the work when working together
def a_share : ℚ := a_rate / (a_rate + b_rate)

-- Theorem stating A's earnings
theorem a_earnings : a_share * total_wages = 2040 := by
  sorry

end NUMINAMATH_CALUDE_a_earnings_l2515_251531


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2515_251504

def A : Set Int := {x | |x| < 3}
def B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2515_251504


namespace NUMINAMATH_CALUDE_item_pricing_and_profit_l2515_251546

/-- Represents the pricing and profit calculation for an item -/
theorem item_pricing_and_profit (a : ℝ) :
  let original_price := a * (1 + 0.2)
  let current_price := original_price * 0.9
  let profit_per_unit := current_price - a
  (current_price = 1.08 * a) ∧
  (1000 * profit_per_unit = 80 * a) := by
  sorry

end NUMINAMATH_CALUDE_item_pricing_and_profit_l2515_251546


namespace NUMINAMATH_CALUDE_power_zero_of_sum_one_l2515_251514

theorem power_zero_of_sum_one (a : ℝ) (h : a ≠ -1) : (a + 1)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_of_sum_one_l2515_251514


namespace NUMINAMATH_CALUDE_units_digit_of_7_19_l2515_251539

theorem units_digit_of_7_19 : (7^19) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_19_l2515_251539


namespace NUMINAMATH_CALUDE_multiply_fractions_equals_thirty_l2515_251553

theorem multiply_fractions_equals_thirty : 15 * (1 / 17) * 34 = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_equals_thirty_l2515_251553


namespace NUMINAMATH_CALUDE_fish_tank_water_l2515_251552

theorem fish_tank_water (current : ℝ) : 
  (current + 7 = 14.75) → current = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_l2515_251552


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2515_251508

/-- Given that a² varies inversely with b², prove that a² = 25/16 when b = 8, given a = 5 when b = 2 -/
theorem inverse_variation_problem (a b : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, x^2 * y^2 = k) 
  (h1 : 5^2 * 2^2 = a^2 * b^2) : 
  8^2 * (25/16) = a^2 * 8^2 := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2515_251508


namespace NUMINAMATH_CALUDE_trivia_game_total_points_l2515_251589

/-- Given the points scored by three teams in a trivia game, prove that the total points scored is 15. -/
theorem trivia_game_total_points (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_points_l2515_251589


namespace NUMINAMATH_CALUDE_solve_equation_l2515_251583

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2515_251583


namespace NUMINAMATH_CALUDE_inequality_proof_l2515_251588

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) : 
  (a^b * b^c * c^d * d^a) / (b^d * c^b * d^c * a^d) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2515_251588


namespace NUMINAMATH_CALUDE_range_of_a_l2515_251548

def p (a : ℝ) : Prop := 2 * a + 1 > 5
def q (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 3

theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 2) ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2515_251548


namespace NUMINAMATH_CALUDE_initial_observations_count_l2515_251535

theorem initial_observations_count (n : ℕ) 
  (h1 : (n : ℝ) > 0)
  (h2 : ∃ S : ℝ, S / n = 11)
  (h3 : ∃ new_obs : ℝ, (S + new_obs) / (n + 1) = 10)
  (h4 : new_obs = 4) :
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_observations_count_l2515_251535


namespace NUMINAMATH_CALUDE_min_value_4x_minus_y_l2515_251549

theorem min_value_4x_minus_y (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + y - 4 ≥ 0) 
  (h3 : x ≤ 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), 
    x' - y' ≥ 0 → x' + y' - 4 ≥ 0 → x' ≤ 4 → 4 * x' - y' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_4x_minus_y_l2515_251549


namespace NUMINAMATH_CALUDE_tea_profit_percentage_l2515_251563

/-- Given a tea mixture and sale price, calculate the profit percentage -/
theorem tea_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20.8) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_tea_profit_percentage_l2515_251563


namespace NUMINAMATH_CALUDE_profit_starts_in_third_year_option1_more_cost_effective_l2515_251586

-- Define the constants
def initial_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def yearly_expense_increase : ℕ := 40000
def annual_income : ℕ := 500000

-- Define a function to calculate expenses for a given year
def expenses (year : ℕ) : ℕ :=
  first_year_expenses + (year - 1) * yearly_expense_increase

-- Define a function to calculate cumulative profit for a given year
def cumulative_profit (year : ℕ) : ℤ :=
  year * annual_income - (initial_cost + (Finset.range year).sum (λ i => expenses (i + 1)))

-- Define a function to calculate average profit for a given year
def average_profit (year : ℕ) : ℚ :=
  (cumulative_profit year : ℚ) / year

-- Theorem 1: The company starts to make a profit in the third year
theorem profit_starts_in_third_year :
  cumulative_profit 3 > 0 ∧ ∀ y : ℕ, y < 3 → cumulative_profit y ≤ 0 := by sorry

-- Define the selling prices for the two options
def option1_price : ℕ := 260000
def option2_price : ℕ := 80000

-- Theorem 2: Option 1 is more cost-effective than Option 2
theorem option1_more_cost_effective :
  ∃ y1 y2 : ℕ,
    (∀ y : ℕ, average_profit y ≤ average_profit y1) ∧
    (∀ y : ℕ, cumulative_profit y ≤ cumulative_profit y2) ∧
    option1_price + cumulative_profit y1 > option2_price + cumulative_profit y2 := by sorry

end NUMINAMATH_CALUDE_profit_starts_in_third_year_option1_more_cost_effective_l2515_251586


namespace NUMINAMATH_CALUDE_total_leaves_in_problem_forest_l2515_251525

/-- Forest structure --/
structure Forest where
  num_trees : ℕ
  num_main_branches : ℕ
  num_sub_branches : ℕ
  num_tertiary_branches : ℕ
  leaves_per_sub_branch : ℕ
  leaves_per_tertiary_branch : ℕ

/-- Calculate the total number of leaves in the forest --/
def total_leaves (f : Forest) : ℕ :=
  f.num_trees * (
    f.num_main_branches * f.num_sub_branches * f.leaves_per_sub_branch +
    f.num_main_branches * f.num_sub_branches * f.num_tertiary_branches * f.leaves_per_tertiary_branch
  )

/-- The specific forest described in the problem --/
def problem_forest : Forest := {
  num_trees := 20,
  num_main_branches := 15,
  num_sub_branches := 25,
  num_tertiary_branches := 30,
  leaves_per_sub_branch := 75,
  leaves_per_tertiary_branch := 45
}

/-- Theorem stating that the total number of leaves in the problem forest is 10,687,500 --/
theorem total_leaves_in_problem_forest :
  total_leaves problem_forest = 10687500 := by
  sorry


end NUMINAMATH_CALUDE_total_leaves_in_problem_forest_l2515_251525


namespace NUMINAMATH_CALUDE_area_of_five_presentable_set_l2515_251542

/-- A complex number is five-presentable if it can be expressed as w - 1/w for some w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of the set T -/
noncomputable def area_T : ℝ := sorry

theorem area_of_five_presentable_set :
  area_T = 624 * Real.pi / 25 := by sorry

end NUMINAMATH_CALUDE_area_of_five_presentable_set_l2515_251542


namespace NUMINAMATH_CALUDE_braiding_time_proof_l2515_251519

/-- Calculates the time in minutes required to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℚ :=
  (num_dancers * braids_per_dancer * seconds_per_braid : ℚ) / 60

/-- Proves that given 15 dancers, 10 braids per dancer, and 45 seconds per braid,
    the total time required to braid all dancers' hair is 112.5 minutes -/
theorem braiding_time_proof :
  braiding_time 15 10 45 = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_proof_l2515_251519


namespace NUMINAMATH_CALUDE_inverse_point_theorem_l2515_251512

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that f(1) + 1 = 2
axiom condition : f 1 + 1 = 2

-- Theorem to prove
theorem inverse_point_theorem : f_inv 1 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_point_theorem_l2515_251512


namespace NUMINAMATH_CALUDE_cutting_process_result_l2515_251572

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with side length -/
structure Square where
  side : ℕ

/-- Cuts the largest possible square from a rectangle and returns the remaining rectangle -/
def cutSquare (r : Rectangle) : Square × Rectangle :=
  if r.width ≤ r.height then
    ({ side := r.width }, { width := r.width, height := r.height - r.width })
  else
    ({ side := r.height }, { width := r.width - r.height, height := r.height })

/-- Applies the cutting process to a rectangle and returns the list of resulting squares -/
def cutProcess (r : Rectangle) : List Square :=
  sorry

/-- Theorem stating the result of applying the cutting process to a 14 × 36 rectangle -/
theorem cutting_process_result :
  let initial_rectangle : Rectangle := { width := 14, height := 36 }
  let result := cutProcess initial_rectangle
  (result.filter (λ s => s.side = 14)).length = 2 ∧
  (result.filter (λ s => s.side = 8)).length = 1 ∧
  (result.filter (λ s => s.side = 6)).length = 1 ∧
  (result.filter (λ s => s.side = 2)).length = 3 :=
sorry

end NUMINAMATH_CALUDE_cutting_process_result_l2515_251572


namespace NUMINAMATH_CALUDE_philips_weekly_mileage_l2515_251528

/-- Calculate Philip's car's mileage for a typical week -/
theorem philips_weekly_mileage (school_distance : ℝ) (market_distance : ℝ)
  (school_trips_per_day : ℕ) (school_days_per_week : ℕ) (market_trips_per_week : ℕ)
  (h1 : school_distance = 2.5)
  (h2 : market_distance = 2)
  (h3 : school_trips_per_day = 2)
  (h4 : school_days_per_week = 4)
  (h5 : market_trips_per_week = 1) :
  school_distance * 2 * ↑school_trips_per_day * ↑school_days_per_week +
  market_distance * 2 * ↑market_trips_per_week = 44 := by
  sorry

#check philips_weekly_mileage

end NUMINAMATH_CALUDE_philips_weekly_mileage_l2515_251528


namespace NUMINAMATH_CALUDE_simultaneous_arrivals_l2515_251534

/-- The distance between points A and B in meters -/
def distance : ℕ := 2010

/-- The speed of the m-th messenger in meters per minute -/
def speed (m : ℕ) : ℕ := m

/-- The time taken by the m-th messenger to reach point B -/
def time (m : ℕ) : ℚ := distance / m

/-- The total number of messengers -/
def total_messengers : ℕ := distance

/-- Predicate for whether two messengers arrive simultaneously -/
def arrive_simultaneously (m n : ℕ) : Prop :=
  1 ≤ m ∧ m < n ∧ n ≤ total_messengers ∧ time m = time n

theorem simultaneous_arrivals :
  ∀ m n : ℕ, arrive_simultaneously m n ↔ m * n = distance ∧ 1 ≤ m ∧ m < n ∧ n ≤ total_messengers :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_arrivals_l2515_251534


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2515_251540

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / 2^n = 220 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2515_251540


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2515_251532

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, k > 0 ∧ (f^[k] n ≤ n + k + 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f → ∀ n : ℕ, f n = n + 2 := by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2515_251532


namespace NUMINAMATH_CALUDE_factorial_multiple_l2515_251543

theorem factorial_multiple (m n : ℕ) : 
  ∃ k : ℕ, (2 * m).factorial * (2 * n).factorial = k * m.factorial * n.factorial * (m + n).factorial := by
sorry

end NUMINAMATH_CALUDE_factorial_multiple_l2515_251543


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalent_to_interval_l2515_251530

theorem quadratic_inequality_equivalent_to_interval (x : ℝ) :
  x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalent_to_interval_l2515_251530


namespace NUMINAMATH_CALUDE_pascal_ratio_row_34_l2515_251524

/-- Pascal's Triangle entry -/
def pascal (n k : ℕ) : ℕ := Nat.choose n k

/-- Checks if three consecutive entries in a row of Pascal's Triangle are in the ratio 2:3:4 -/
def hasRatio234 (n : ℕ) : Prop :=
  ∃ k : ℕ, 
    2 * pascal n (k + 1) = 3 * pascal n k ∧
    3 * pascal n (k + 2) = 4 * pascal n (k + 1)

theorem pascal_ratio_row_34 : hasRatio234 34 := by
  sorry

end NUMINAMATH_CALUDE_pascal_ratio_row_34_l2515_251524


namespace NUMINAMATH_CALUDE_sticker_distribution_l2515_251509

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into 5 or fewer parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2515_251509


namespace NUMINAMATH_CALUDE_vector_norm_sum_l2515_251578

theorem vector_norm_sum (a b : ℝ × ℝ) :
  let m := (2 * a.1 + b.1, 2 * a.2 + b.2) / 2
  m = (-1, 5) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_vector_norm_sum_l2515_251578


namespace NUMINAMATH_CALUDE_sequence_properties_l2515_251599

/-- Sequence a_n with sum S_n = n^2 + pn -/
def S (n : ℕ) (p : ℝ) : ℝ := n^2 + p * n

/-- Sequence b_n with sum T_n = 3n^2 - 2n -/
def T (n : ℕ) : ℝ := 3 * n^2 - 2 * n

/-- a_n is the difference of consecutive S_n terms -/
def a (n : ℕ) (p : ℝ) : ℝ := S n p - S (n-1) p

/-- b_n is the difference of consecutive T_n terms -/
def b (n : ℕ) : ℝ := T n - T (n-1)

/-- c_n is the sequence formed by odd-indexed terms of b_n -/
def c (n : ℕ) : ℝ := b (2*n - 1)

theorem sequence_properties (p : ℝ) :
  (a 10 p = b 10) → p = 36 ∧ ∀ n, c n = 12 * n - 11 := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2515_251599


namespace NUMINAMATH_CALUDE_joan_gave_sam_seashells_l2515_251566

/-- The number of seashells Joan gave to Sam -/
def seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) : ℕ :=
  initial_seashells - remaining_seashells

/-- Theorem: Joan gave Sam 43 seashells -/
theorem joan_gave_sam_seashells (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70)
  (h2 : remaining_seashells = 27) :
  seashells_given_to_sam initial_seashells remaining_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_sam_seashells_l2515_251566


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_squares_sum_l2515_251568

theorem cubic_root_reciprocal_squares_sum (p q : ℂ) (z₁ z₂ z₃ : ℂ) : 
  z₁^3 + p*z₁ + q = 0 → 
  z₂^3 + p*z₂ + q = 0 → 
  z₃^3 + p*z₃ + q = 0 → 
  z₁ ≠ z₂ → z₂ ≠ z₃ → z₃ ≠ z₁ →
  q ≠ 0 →
  1/z₁^2 + 1/z₂^2 + 1/z₃^2 = p^2 / q^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_squares_sum_l2515_251568


namespace NUMINAMATH_CALUDE_point_translation_l2515_251502

/-- Given a point A with coordinates (1, -1), prove that moving it up by 2 units
    and then left by 3 units results in a point B with coordinates (-2, 1). -/
theorem point_translation (A B : ℝ × ℝ) :
  A = (1, -1) →
  B.1 = A.1 - 3 →
  B.2 = A.2 + 2 →
  B = (-2, 1) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l2515_251502


namespace NUMINAMATH_CALUDE_count_12_digit_numbers_with_consecutive_ones_l2515_251577

/-- The sequence of counts of n-digit numbers with digits 1, 2, or 3 without two consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 3
| (n+2) => 2 * F (n+1) + F n

/-- The count of n-digit numbers with digits 1, 2, or 3 -/
def total_count (n : ℕ) : ℕ := 3^n

/-- The count of n-digit numbers with digits 1, 2, or 3 and at least two consecutive 1's -/
def count_with_consecutive_ones (n : ℕ) : ℕ := total_count n - F n

theorem count_12_digit_numbers_with_consecutive_ones : 
  count_with_consecutive_ones 12 = 530456 := by
  sorry

end NUMINAMATH_CALUDE_count_12_digit_numbers_with_consecutive_ones_l2515_251577


namespace NUMINAMATH_CALUDE_probability_heart_then_club_l2515_251579

theorem probability_heart_then_club (total_cards : Nat) (hearts : Nat) (clubs : Nat) :
  total_cards = 52 →
  hearts = 13 →
  clubs = 13 →
  (hearts : ℚ) / total_cards * clubs / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_club_l2515_251579


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l2515_251527

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 5) 
  (h3 : outer_revolutions = 15) : 
  ∃ inner_revolutions : ℕ, 
    (2 * Real.pi * outer_radius * outer_revolutions) = 
    (2 * Real.pi * inner_radius * inner_revolutions) ∧ 
    inner_revolutions = 90 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l2515_251527


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2515_251576

theorem quadratic_rewrite_sum (x : ℝ) :
  ∃ (a b c : ℝ),
    (-4 * x^2 + 16 * x + 128 = a * (x + b)^2 + c) ∧
    (a + b + c = 138) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2515_251576


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2515_251536

/-- Given a geometric sequence {a_n} where a₇ = 1/4 and a₃a₅ = 4(a₄ - 1), prove that a₂ = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (h_a7 : a 7 = 1 / 4)
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2515_251536


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2515_251575

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (a + Complex.I) / (2 - Complex.I) + a ∧ z.re = 0 → Complex.abs z = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2515_251575


namespace NUMINAMATH_CALUDE_triangle_hypotenuse_l2515_251516

theorem triangle_hypotenuse (x y : ℝ) (h : ℝ) : 
  (1/3 : ℝ) * π * y^2 * x = 1200 * π →
  (1/3 : ℝ) * π * x^2 * (2*x) = 3840 * π →
  x^2 + y^2 = h^2 →
  h = 2 * Real.sqrt 131 := by
sorry

end NUMINAMATH_CALUDE_triangle_hypotenuse_l2515_251516


namespace NUMINAMATH_CALUDE_log_sum_simplification_l2515_251513

theorem log_sum_simplification :
  let f (a b : ℝ) := 1 / (Real.log a / Real.log b + 1)
  f 3 12 + f 2 8 + f 7 9 = 1 - Real.log 7 / Real.log 1008 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l2515_251513


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l2515_251597

/-- A shape formed by adding a pyramid to one face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertex : ℕ

/-- The total number of exterior faces in the combined shape -/
def total_faces (pp : PrismPyramid) : ℕ := pp.prism_faces - 1 + pp.pyramid_faces

/-- The total number of edges in the combined shape -/
def total_edges (pp : PrismPyramid) : ℕ := pp.prism_edges + pp.pyramid_edges

/-- The total number of vertices in the combined shape -/
def total_vertices (pp : PrismPyramid) : ℕ := pp.prism_vertices + pp.pyramid_vertex

/-- The sum of exterior faces, edges, and vertices in the combined shape -/
def total_sum (pp : PrismPyramid) : ℕ := total_faces pp + total_edges pp + total_vertices pp

theorem prism_pyramid_sum :
  ∃ (pp : PrismPyramid), total_sum pp = 34 ∧
  ∀ (pp' : PrismPyramid), total_sum pp' ≤ total_sum pp :=
sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l2515_251597


namespace NUMINAMATH_CALUDE_bryan_books_count_l2515_251565

/-- The number of bookshelves Bryan has -/
def num_shelves : ℕ := 2

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 17

/-- The total number of books Bryan has -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem bryan_books_count : total_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_bryan_books_count_l2515_251565


namespace NUMINAMATH_CALUDE_problem_statement_l2515_251594

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2515_251594


namespace NUMINAMATH_CALUDE_fraction_undefined_l2515_251522

theorem fraction_undefined (x : ℝ) : (3 * x - 1) / (x + 3) = 0 / 0 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_l2515_251522


namespace NUMINAMATH_CALUDE_proportional_survey_distribution_l2515_251533

/-- Represents the number of surveys to be drawn from a group -/
def surveyCount (totalSurveys : ℕ) (groupSize : ℕ) (totalPopulation : ℕ) : ℕ :=
  (totalSurveys * groupSize) / totalPopulation

theorem proportional_survey_distribution 
  (totalSurveys : ℕ) 
  (facultyStaffSize juniorHighSize seniorHighSize : ℕ) 
  (h1 : totalSurveys = 120)
  (h2 : facultyStaffSize = 500)
  (h3 : juniorHighSize = 3000)
  (h4 : seniorHighSize = 4000) :
  let totalPopulation := facultyStaffSize + juniorHighSize + seniorHighSize
  (surveyCount totalSurveys facultyStaffSize totalPopulation = 8) ∧ 
  (surveyCount totalSurveys juniorHighSize totalPopulation = 48) ∧
  (surveyCount totalSurveys seniorHighSize totalPopulation = 64) :=
by sorry

#check proportional_survey_distribution

end NUMINAMATH_CALUDE_proportional_survey_distribution_l2515_251533


namespace NUMINAMATH_CALUDE_dog_food_consumption_l2515_251562

/-- Represents the amount of dog food in various units -/
structure DogFood where
  cups : ℚ
  pounds : ℚ

/-- Represents the feeding schedule and food consumption of dogs -/
structure DogFeeding where
  numDogs : ℕ
  feedingsPerDay : ℕ
  daysPerMonth : ℕ
  bagsPerMonth : ℕ
  poundsPerBag : ℚ
  cupWeight : ℚ

/-- Calculates the number of cups of dog food each dog eats at a time -/
def cupsPerFeeding (df : DogFeeding) : ℚ :=
  let totalPoundsPerMonth := df.bagsPerMonth * df.poundsPerBag
  let poundsPerDogPerMonth := totalPoundsPerMonth / df.numDogs
  let feedingsPerMonth := df.feedingsPerDay * df.daysPerMonth
  let poundsPerFeeding := poundsPerDogPerMonth / feedingsPerMonth
  poundsPerFeeding / df.cupWeight

/-- Theorem stating that each dog eats 6 cups of dog food at a time -/
theorem dog_food_consumption (df : DogFeeding) 
  (h1 : df.cupWeight = 1/4)
  (h2 : df.numDogs = 2)
  (h3 : df.feedingsPerDay = 2)
  (h4 : df.bagsPerMonth = 9)
  (h5 : df.poundsPerBag = 20)
  (h6 : df.daysPerMonth = 30) :
  cupsPerFeeding df = 6 := by
  sorry

#eval cupsPerFeeding {
  numDogs := 2,
  feedingsPerDay := 2,
  daysPerMonth := 30,
  bagsPerMonth := 9,
  poundsPerBag := 20,
  cupWeight := 1/4
}

end NUMINAMATH_CALUDE_dog_food_consumption_l2515_251562


namespace NUMINAMATH_CALUDE_remainder_sum_l2515_251526

theorem remainder_sum (p q : ℤ) (hp : p % 80 = 75) (hq : q % 120 = 115) : (p + q) % 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2515_251526


namespace NUMINAMATH_CALUDE_percent_profit_problem_l2515_251521

/-- Given that the cost price of 60 articles equals the selling price of 50 articles,
    prove that the percent profit is 20%. -/
theorem percent_profit_problem (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_problem_l2515_251521


namespace NUMINAMATH_CALUDE_zoo_trip_attendance_l2515_251503

/-- The number of buses available for the zoo trip -/
def num_buses : ℕ := 3

/-- The number of people that would go in each bus if evenly distributed -/
def people_per_bus : ℕ := 73

/-- The total number of people going to the zoo -/
def total_people : ℕ := num_buses * people_per_bus

theorem zoo_trip_attendance : total_people = 219 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_attendance_l2515_251503


namespace NUMINAMATH_CALUDE_product_not_always_minimized_when_closest_l2515_251500

theorem product_not_always_minimized_when_closest (d : ℝ) (h : d > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = d ∧
  ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' - y' = d ∧
  abs (x' - y') < abs (x - y) ∧ x' * y' < x * y :=
by sorry

-- Other statements (A, B, D, E) are correct, but we don't need to prove them for this task

end NUMINAMATH_CALUDE_product_not_always_minimized_when_closest_l2515_251500


namespace NUMINAMATH_CALUDE_difference_of_squares_identity_l2515_251593

theorem difference_of_squares_identity (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_identity_l2515_251593


namespace NUMINAMATH_CALUDE_solve_euro_equation_l2515_251596

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_euro_equation : 
  ∀ x : ℝ, euro x (euro 4 5) = 720 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l2515_251596


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2515_251558

def g (x : ℝ) : ℝ := -3 * x^4 + 5

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2515_251558


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2515_251523

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 3) (h2 : b = -1) :
  (4 * a^2 * b - 5 * b^2) - 3 * (a^2 * b - 2 * b^2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2515_251523


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2515_251511

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways six people can sit in a row of seven chairs with the third chair vacant. -/
def seatingArrangements : ℕ := factorial 6

theorem seating_arrangements_count : seatingArrangements = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2515_251511


namespace NUMINAMATH_CALUDE_lobster_rolls_count_total_plates_sum_l2515_251591

/-- The number of plates of lobster rolls served at a banquet -/
def lobster_rolls : ℕ := 55 - (14 + 16)

/-- The total number of plates served at the banquet -/
def total_plates : ℕ := 55

/-- The number of plates of spicy hot noodles served at the banquet -/
def spicy_hot_noodles : ℕ := 14

/-- The number of plates of seafood noodles served at the banquet -/
def seafood_noodles : ℕ := 16

/-- Theorem stating that the number of lobster roll plates is 25 -/
theorem lobster_rolls_count : lobster_rolls = 25 := by
  sorry

/-- Theorem stating that the total number of plates is the sum of all dishes -/
theorem total_plates_sum : 
  total_plates = lobster_rolls + spicy_hot_noodles + seafood_noodles := by
  sorry

end NUMINAMATH_CALUDE_lobster_rolls_count_total_plates_sum_l2515_251591


namespace NUMINAMATH_CALUDE_river_depth_increase_l2515_251505

/-- River depth problem -/
theorem river_depth_increase (may_depth june_depth july_depth : ℝ) : 
  may_depth = 5 →
  july_depth = 3 * june_depth →
  july_depth = 45 →
  june_depth - may_depth = 10 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_increase_l2515_251505


namespace NUMINAMATH_CALUDE_square_perimeter_l2515_251529

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  let rectangle_perimeter := 2 * (s + s / 5)
  rectangle_perimeter = 48 → 4 * s = 80 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2515_251529


namespace NUMINAMATH_CALUDE_annie_crayons_l2515_251581

theorem annie_crayons (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 36 → final = 40 → initial = 4 := by sorry

end NUMINAMATH_CALUDE_annie_crayons_l2515_251581


namespace NUMINAMATH_CALUDE_find_M_l2515_251573

theorem find_M : ∃ M : ℕ, (9.5 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) ∧ M = 39 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2515_251573


namespace NUMINAMATH_CALUDE_unique_four_digit_square_repeated_digits_l2515_251518

-- Define a four-digit number with repeated digits
def fourDigitRepeated (x y : Nat) : Nat :=
  1100 * x + 11 * y

-- Theorem statement
theorem unique_four_digit_square_repeated_digits :
  ∃! n : Nat, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (∃ x y : Nat, n = fourDigitRepeated x y) ∧  -- repeated digits
    (∃ m : Nat, n = m ^ 2) ∧  -- perfect square
    n = 7744 := by
  sorry


end NUMINAMATH_CALUDE_unique_four_digit_square_repeated_digits_l2515_251518


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2515_251550

theorem square_garden_perimeter (area : Real) (perimeter : Real) :
  area = 90.25 →
  area = 2 * perimeter + 14.25 →
  perimeter = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2515_251550


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2515_251569

theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ+, (∀ y : ℕ+, y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
             (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
             x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2515_251569


namespace NUMINAMATH_CALUDE_max_revenue_at_22_l2515_251580

def cinema_revenue (price : ℕ) : ℤ :=
  if price ≤ 10 then
    1000 * price - 5750
  else
    -30 * price * price + 1300 * price - 5750

def valid_price (price : ℕ) : Prop :=
  (6 ≤ price) ∧ (price ≤ 38)

theorem max_revenue_at_22 :
  (∀ p, valid_price p → cinema_revenue p ≤ cinema_revenue 22) ∧
  cinema_revenue 22 = 8330 :=
sorry

end NUMINAMATH_CALUDE_max_revenue_at_22_l2515_251580


namespace NUMINAMATH_CALUDE_complement_A_relative_to_I_l2515_251556

def I : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {x : Int | x^2 < 3}

theorem complement_A_relative_to_I :
  {x ∈ I | x ∉ A} = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_relative_to_I_l2515_251556


namespace NUMINAMATH_CALUDE_temple_visit_theorem_l2515_251595

/-- The number of people who went to the temple with Nathan -/
def number_of_people : ℕ := 3

/-- The cost per object in dollars -/
def cost_per_object : ℕ := 11

/-- The number of objects per person -/
def objects_per_person : ℕ := 5

/-- The total charge for all objects in dollars -/
def total_charge : ℕ := 165

/-- Theorem stating that the number of people is correct given the conditions -/
theorem temple_visit_theorem : 
  number_of_people * objects_per_person * cost_per_object = total_charge :=
by sorry

end NUMINAMATH_CALUDE_temple_visit_theorem_l2515_251595


namespace NUMINAMATH_CALUDE_matthew_crackers_l2515_251547

theorem matthew_crackers (initial_crackers : ℕ) 
  (friends : ℕ) 
  (crackers_eaten_per_friend : ℕ) 
  (crackers_left : ℕ) : 
  friends = 2 ∧ 
  crackers_eaten_per_friend = 6 ∧ 
  crackers_left = 11 ∧ 
  initial_crackers = friends * (crackers_eaten_per_friend * 2) + crackers_left → 
  initial_crackers = 35 := by
sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2515_251547


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_18_l2515_251551

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_pow_18 : tens_digit (6^18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_18_l2515_251551


namespace NUMINAMATH_CALUDE_distribute_5_3_l2515_251587

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 3^5 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2515_251587


namespace NUMINAMATH_CALUDE_blue_ball_probability_l2515_251541

theorem blue_ball_probability (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue : ℚ) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l2515_251541


namespace NUMINAMATH_CALUDE_compound_inequality_l2515_251538

theorem compound_inequality (x : ℝ) : 
  x > -1/2 → (3 - 1/(3*x + 4) < 5 ∧ 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_compound_inequality_l2515_251538


namespace NUMINAMATH_CALUDE_square_side_lengths_l2515_251561

theorem square_side_lengths (a b : ℕ) : 
  a > b → a^2 = b^2 + 2001 → 
  a ∈ ({49, 55, 335, 1001} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_square_side_lengths_l2515_251561


namespace NUMINAMATH_CALUDE_symmetric_lines_b_value_l2515_251520

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two lines are symmetric with respect to a given point -/
def are_symmetric (l1 l2 : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), l1.a * x + l1.b * y + l1.c = 0 →
    ∃ (x' y' : ℝ), l2.a * x' + l2.b * y' + l2.c = 0 ∧
      p.x = (x + x') / 2 ∧ p.y = (y + y') / 2

/-- The main theorem stating that given the conditions, b must equal 2 -/
theorem symmetric_lines_b_value :
  ∀ (a b : ℝ),
  let l1 : Line := ⟨1, 2, -3⟩
  let l2 : Line := ⟨a, 4, b⟩
  let p : Point := ⟨1, 0⟩
  are_symmetric l1 l2 p → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_b_value_l2515_251520


namespace NUMINAMATH_CALUDE_equal_population_time_l2515_251517

/-- The number of years it takes for two villages' populations to be equal -/
def yearsToEqualPopulation (initialX initialY decreaseRateX increaseRateY : ℕ) : ℕ :=
  (initialX - initialY) / (decreaseRateX + increaseRateY)

theorem equal_population_time :
  yearsToEqualPopulation 70000 42000 1200 800 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_time_l2515_251517
