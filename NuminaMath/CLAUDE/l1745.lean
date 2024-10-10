import Mathlib

namespace average_equality_l1745_174574

theorem average_equality (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg : ℝ := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum : ℝ := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  new_sum / (n + 2) = original_avg := by
  sorry

end average_equality_l1745_174574


namespace plot_length_is_sixty_l1745_174518

/-- Given a rectangular plot with the following properties:
    1. The length is 20 meters more than the breadth.
    2. The cost of fencing the plot at 26.50 per meter is Rs. 5300.
    This theorem proves that the length of the plot is 60 meters. -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 60 := by
sorry

end plot_length_is_sixty_l1745_174518


namespace greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l1745_174525

theorem greatest_integer_solution (x : ℤ) : (7 : ℤ) - 5*x + x^2 > 24 → x ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : (7 : ℤ) - 5*7 + 7^2 > 24 :=
by sorry

theorem no_greater_integer :
  ∀ y : ℤ, y > 7 → ¬((7 : ℤ) - 5*y + y^2 > 24) :=
by sorry

end greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l1745_174525


namespace rectangle_circle_area_ratio_l1745_174586

theorem rectangle_circle_area_ratio (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * Real.pi * r) (h2 : l = 2 * w) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end rectangle_circle_area_ratio_l1745_174586


namespace factorize_quadratic_xy_value_l1745_174531

-- Problem 1
theorem factorize_quadratic (x : ℝ) : 
  x^2 - 120*x + 3456 = (x - 48) * (x - 72) := by sorry

-- Problem 2
theorem xy_value (x y : ℝ) : 
  x^2 + y^2 + 8*x - 12*y + 52 = 0 → x*y = -24 := by sorry

end factorize_quadratic_xy_value_l1745_174531


namespace haley_money_difference_l1745_174561

/-- Calculates the difference between the final and initial amount of money Haley has after various transactions. -/
theorem haley_money_difference :
  let initial_amount : ℚ := 2
  let chores_earnings : ℚ := 5.25
  let birthday_gift : ℚ := 10
  let neighbor_help : ℚ := 7.5
  let found_money : ℚ := 0.5
  let aunt_gift_pounds : ℚ := 3
  let pound_to_dollar : ℚ := 1.3
  let candy_spent : ℚ := 3.75
  let money_lost : ℚ := 1.5
  
  let total_received : ℚ := chores_earnings + birthday_gift + neighbor_help + found_money + aunt_gift_pounds * pound_to_dollar
  let total_spent : ℚ := candy_spent + money_lost
  let final_amount : ℚ := initial_amount + total_received - total_spent
  
  final_amount - initial_amount = 19.9 := by sorry

end haley_money_difference_l1745_174561


namespace prob_odd_divisor_21_factorial_l1745_174569

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primeFactorization (n : ℕ) : List (ℕ × ℕ) := sorry

def numDivisors (n : ℕ) : ℕ := sorry

def numOddDivisors (n : ℕ) : ℕ := sorry

theorem prob_odd_divisor_21_factorial :
  let n := factorial 21
  let totalDivisors := numDivisors n
  let oddDivisors := numOddDivisors n
  (oddDivisors : ℚ) / totalDivisors = 1 / 19 := by sorry

end prob_odd_divisor_21_factorial_l1745_174569


namespace factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l1745_174513

-- Define p-adic valuation
noncomputable def v_p (p : ℕ) (n : ℕ) : ℚ := sorry

-- Define sum of digits in base p
def τ_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define number of carries when adding in base p
def carries_base_p (p : ℕ) (a b : ℕ) : ℕ := sorry

-- Lemma
theorem factorial_p_adic_valuation (p : ℕ) (n : ℕ) : 
  v_p p (n.factorial) = (n - τ_p p n) / (p - 1) := sorry

-- Theorem 1
theorem binomial_p_adic_valuation (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = (τ_p p k + τ_p p (n - k) - τ_p p n) / (p - 1) := sorry

-- Theorem 2
theorem binomial_p_adic_valuation_carries (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = carries_base_p p k (n - k) := sorry

-- Theorem 3
theorem binomial_p_adic_valuation_zero (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = 0 ↔ carries_base_p p k (n - k) = 0 := sorry

end factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l1745_174513


namespace second_polygon_sides_l1745_174526

/-- Given two regular polygons with the same perimeter, where the first has 24 sides
    and its side length is three times that of the second, prove the second has 72 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure side length is positive
  24 * (3 * s) = n * s →  -- Same perimeter condition
  n = 72 := by
sorry

end second_polygon_sides_l1745_174526


namespace geometric_sequence_product_l1745_174580

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a3 : a 3 = 2) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 := by
  sorry

end geometric_sequence_product_l1745_174580


namespace negation_of_all_ge_two_l1745_174579

theorem negation_of_all_ge_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end negation_of_all_ge_two_l1745_174579


namespace hypotenuse_of_6_8_triangle_l1745_174564

/-- The Pythagorean theorem for a right-angled triangle -/
def pythagorean_theorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem: In a right-angled triangle with legs of length 6 and 8, the hypotenuse has a length of 10 -/
theorem hypotenuse_of_6_8_triangle :
  ∃ (c : ℝ), pythagorean_theorem 6 8 c ∧ c = 10 := by
  sorry

end hypotenuse_of_6_8_triangle_l1745_174564


namespace system_solution_conditions_l1745_174563

/-- Given a system of equations, prove the existence of conditions for distinct positive solutions -/
theorem system_solution_conditions (a b : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + z = a) ∧ 
    (x^2 + y^2 + z^2 = b^2) ∧ 
    (x * y = z^2) ∧ 
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
    (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ a = c ∧ b = d) :=
by
  sorry

end system_solution_conditions_l1745_174563


namespace x_squared_minus_5x_is_quadratic_l1745_174582

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: x^2 - 5x = 0 is a quadratic equation -/
theorem x_squared_minus_5x_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_minus_5x_is_quadratic_l1745_174582


namespace youngest_sibling_age_l1745_174545

theorem youngest_sibling_age (y : ℝ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end youngest_sibling_age_l1745_174545


namespace quadratic_equation_roots_l1745_174596

theorem quadratic_equation_roots : ∃! x : ℝ, x^2 - 4*x + 4 = 0 := by
  sorry

end quadratic_equation_roots_l1745_174596


namespace max_cars_in_parking_lot_l1745_174520

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position -/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot -/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid -/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem stating the maximum number of cars that can be parked -/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot), isValidConfig lot ∧ carCount lot = 28 ∧
  ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
sorry

end max_cars_in_parking_lot_l1745_174520


namespace arithmetic_calculation_l1745_174522

theorem arithmetic_calculation : 15 * 20 - 25 * 15 + 10 * 25 = 175 := by
  sorry

end arithmetic_calculation_l1745_174522


namespace a_investment_l1745_174584

/-- A's investment in a partnership business --/
def partners_investment (total_profit partner_a_total_received partner_b_investment : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let partner_a_profit_share := partner_a_total_received - management_fee
  (partner_a_profit_share * partner_b_investment) / (remaining_profit - partner_a_profit_share)

/-- Theorem stating A's investment given the problem conditions --/
theorem a_investment (total_profit partner_a_total_received partner_b_investment : ℚ) 
  (h1 : total_profit = 9600)
  (h2 : partner_a_total_received = 4800)
  (h3 : partner_b_investment = 25000) :
  partners_investment total_profit partner_a_total_received partner_b_investment = 20000 := by
  sorry

end a_investment_l1745_174584


namespace remainder_problem_l1745_174594

theorem remainder_problem (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end remainder_problem_l1745_174594


namespace impossible_to_measure_one_liter_l1745_174552

/-- Represents the state of water in the containers -/
structure WaterState where
  jug : ℕ  -- Amount of water in the 4-liter jug
  pot : ℕ  -- Amount of water in the 6-liter pot

/-- Possible operations on the containers -/
inductive Operation
  | FillJug
  | FillPot
  | EmptyJug
  | EmptyPot
  | PourJugToPot
  | PourPotToJug

/-- Applies an operation to a water state -/
def applyOperation (state : WaterState) (op : Operation) : WaterState :=
  match op with
  | Operation.FillJug => { jug := 4, pot := state.pot }
  | Operation.FillPot => { jug := state.jug, pot := 6 }
  | Operation.EmptyJug => { jug := 0, pot := state.pot }
  | Operation.EmptyPot => { jug := state.jug, pot := 0 }
  | Operation.PourJugToPot =>
      let amount := min state.jug (6 - state.pot)
      { jug := state.jug - amount, pot := state.pot + amount }
  | Operation.PourPotToJug =>
      let amount := min state.pot (4 - state.jug)
      { jug := state.jug + amount, pot := state.pot - amount }

/-- Theorem: It's impossible to measure exactly one liter of water -/
theorem impossible_to_measure_one_liter :
  ∀ (initial : WaterState) (ops : List Operation),
    (initial.jug = 0 ∧ initial.pot = 0) →
    let final := ops.foldl applyOperation initial
    (final.jug ≠ 1 ∧ final.pot ≠ 1) :=
  sorry


end impossible_to_measure_one_liter_l1745_174552


namespace volleyball_team_selection_l1745_174543

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_selection :
  choose total_players starters - choose (total_players - quadruplets) starters = 28392 := by
  sorry

end volleyball_team_selection_l1745_174543


namespace pencil_distribution_l1745_174599

/-- Given a classroom with 4 children and 8 pencils to be distributed,
    prove that each child receives 2 pencils. -/
theorem pencil_distribution (num_children : ℕ) (num_pencils : ℕ) 
  (h1 : num_children = 4) 
  (h2 : num_pencils = 8) : 
  num_pencils / num_children = 2 := by
  sorry

end pencil_distribution_l1745_174599


namespace inverse_existence_l1745_174581

-- Define the three functions
def linear_function (x : ℝ) : ℝ := sorry
def quadratic_function (x : ℝ) : ℝ := sorry
def exponential_function (x : ℝ) : ℝ := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem inverse_existence :
  (has_inverse linear_function) ∧
  (¬ has_inverse quadratic_function) ∧
  (has_inverse exponential_function) := by sorry

end inverse_existence_l1745_174581


namespace special_ellipse_equation_l1745_174556

/-- An ellipse with center at the origin, one focus at (0,2), and a chord formed by
    the intersection with the line y=3x+7 whose midpoint has a y-coordinate of 1 --/
structure SpecialEllipse where
  /-- One focus of the ellipse --/
  focus : ℝ × ℝ
  /-- Slope of the intersecting line --/
  m : ℝ
  /-- y-intercept of the intersecting line --/
  b : ℝ
  /-- y-coordinate of the chord's midpoint --/
  midpoint_y : ℝ
  /-- Conditions for the special ellipse --/
  h1 : focus = (0, 2)
  h2 : m = 3
  h3 : b = 7
  h4 : midpoint_y = 1

/-- The equation of the ellipse --/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 12 = 1

/-- Theorem stating that the given special ellipse has the specified equation --/
theorem special_ellipse_equation (e : SpecialEllipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2} ↔
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 12 = 1} :=
by sorry

end special_ellipse_equation_l1745_174556


namespace car_rental_cost_equality_l1745_174575

/-- The fixed amount Samuel paid for car rental -/
def samuel_fixed_amount : ℝ := 24

/-- The per-kilometer rate for Samuel's rental -/
def samuel_rate : ℝ := 0.16

/-- The fixed amount Carrey paid for car rental -/
def carrey_fixed_amount : ℝ := 20

/-- The per-kilometer rate for Carrey's rental -/
def carrey_rate : ℝ := 0.25

/-- The distance driven by both Samuel and Carrey -/
def distance_driven : ℝ := 44.44444444444444

theorem car_rental_cost_equality :
  samuel_fixed_amount + samuel_rate * distance_driven =
  carrey_fixed_amount + carrey_rate * distance_driven :=
by sorry


end car_rental_cost_equality_l1745_174575


namespace minutes_to_hours_l1745_174555

-- Define the number of minutes Marcia spent
def minutes_spent : ℕ := 300

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem: 300 minutes is equal to 5 hours
theorem minutes_to_hours : 
  (minutes_spent : ℚ) / minutes_per_hour = 5 := by
  sorry

end minutes_to_hours_l1745_174555


namespace correct_conclusions_l1745_174541

theorem correct_conclusions :
  (∀ a b : ℝ, a + b > 0 ∧ a * b > 0 → a > 0 ∧ b > 0) ∧
  (∀ a b : ℝ, b ≠ 0 → a / b = -1 → a + b = 0) ∧
  (∀ a b c : ℝ, a < b ∧ b < c → |a - b| + |b - c| = |a - c|) :=
by sorry

end correct_conclusions_l1745_174541


namespace at_op_zero_at_op_distributive_at_op_max_for_rectangle_l1745_174529

/-- Operation @ for real numbers -/
def at_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem 1: If a @ b = 0, then a = 0 or b = 0 -/
theorem at_op_zero (a b : ℝ) : at_op a b = 0 → a = 0 ∨ b = 0 := by sorry

/-- Theorem 2: a @ (b + c) = a @ b + a @ c -/
theorem at_op_distributive (a b c : ℝ) : at_op a (b + c) = at_op a b + at_op a c := by sorry

/-- Theorem 3: For a rectangle with fixed perimeter, a @ b is maximized when a = b -/
theorem at_op_max_for_rectangle (a b : ℝ) (h : a > 0 ∧ b > 0) (perimeter : ℝ) 
  (h_perimeter : 2 * (a + b) = perimeter) :
  ∀ x y, x > 0 → y > 0 → 2 * (x + y) = perimeter → at_op a b ≥ at_op x y := by sorry

end at_op_zero_at_op_distributive_at_op_max_for_rectangle_l1745_174529


namespace competition_distance_l1745_174591

/-- Represents the distances cycled on each day of the week -/
structure WeekDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculates the total distance cycled in a week -/
def totalDistance (distances : WeekDistances) : ℝ :=
  distances.monday + distances.tuesday + distances.wednesday + 
  distances.thursday + distances.friday + distances.saturday + distances.sunday

/-- Theorem stating the total distance cycled in the competition week -/
theorem competition_distance : ∃ (distances : WeekDistances),
  distances.monday = 40 ∧
  distances.tuesday = 50 ∧
  distances.wednesday = distances.tuesday * 0.5 ∧
  distances.thursday = distances.monday + distances.wednesday ∧
  distances.friday = distances.thursday * 1.2 ∧
  distances.saturday = distances.friday * 0.75 ∧
  distances.sunday = distances.saturday - distances.wednesday ∧
  totalDistance distances = 350 := by
  sorry


end competition_distance_l1745_174591


namespace train_speed_second_part_l1745_174502

/-- Proves that the speed of a train during the second part of a journey is 20 kmph,
    given specific conditions about the journey. -/
theorem train_speed_second_part 
  (x : ℝ) 
  (h_positive : x > 0) 
  (speed_first : ℝ) 
  (h_speed_first : speed_first = 40) 
  (distance_first : ℝ) 
  (h_distance_first : distance_first = x) 
  (distance_second : ℝ) 
  (h_distance_second : distance_second = 2 * x) 
  (distance_total : ℝ) 
  (h_distance_total : distance_total = 6 * x) 
  (speed_average : ℝ) 
  (h_speed_average : speed_average = 48) : 
  ∃ (speed_second : ℝ), speed_second = 20 := by
sorry


end train_speed_second_part_l1745_174502


namespace two_digit_number_solution_l1745_174542

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 99

theorem two_digit_number_solution (cd : TwoDigitNumber) :
  54 * (toRepeatingDecimal cd - toDecimal cd) = (36 : ℚ) / 100 →
  cd.val = 65 := by
  sorry

end two_digit_number_solution_l1745_174542


namespace average_class_size_is_35_l1745_174598

/-- Represents the number of children in each age group --/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school setup --/
def SundaySchool (ages : AgeGroups) : Prop :=
  ages.three_year_olds = 13 ∧
  ages.four_year_olds = 20 ∧
  ages.five_year_olds = 15 ∧
  ages.six_year_olds = 22

/-- Calculates the average class size --/
def averageClassSize (ages : AgeGroups) : ℚ :=
  let class1 := ages.three_year_olds + ages.four_year_olds
  let class2 := ages.five_year_olds + ages.six_year_olds
  (class1 + class2) / 2

/-- Theorem stating that the average class size is 35 --/
theorem average_class_size_is_35 (ages : AgeGroups) 
  (h : SundaySchool ages) : averageClassSize ages = 35 := by
  sorry

end average_class_size_is_35_l1745_174598


namespace blue_section_damage_probability_l1745_174568

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes we're interested in -/
def k : ℕ := 7

/-- The probability of exactly k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem blue_section_damage_probability :
  bernoulli_probability n k p = 128/823543 := by
  sorry

end blue_section_damage_probability_l1745_174568


namespace pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l1745_174511

/-- A pentagonal pyramid is a three-dimensional shape with a pentagonal base and triangular faces connecting the base to an apex. -/
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle

/-- A pentagon is a polygon with 5 sides. -/
structure Pentagon where
  sides : Fin 5 → Segment

/-- Theorem: The number of faces of a pentagonal pyramid is 6. -/
theorem pentagonal_pyramid_faces (p : PentagonalPyramid) : Nat :=
  6

#check pentagonal_pyramid_faces

/-- Proof of the theorem -/
theorem pentagonal_pyramid_faces_proof (p : PentagonalPyramid) : 
  pentagonal_pyramid_faces p = 6 := by
  sorry

end pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l1745_174511


namespace nick_speed_l1745_174595

/-- Given the speeds of Alan, Maria, and Nick in relation to each other,
    prove that Nick's speed is 6 miles per hour. -/
theorem nick_speed (alan_speed : ℝ) (maria_speed : ℝ) (nick_speed : ℝ)
    (h1 : alan_speed = 6)
    (h2 : maria_speed = 3/4 * alan_speed)
    (h3 : nick_speed = 4/3 * maria_speed) :
    nick_speed = 6 := by
  sorry

end nick_speed_l1745_174595


namespace incorrect_vs_correct_calculation_l1745_174557

theorem incorrect_vs_correct_calculation (x : ℝ) (h : x - 3 + 49 = 66) : 
  (3 * x + 49) - 66 = 43 := by
sorry

end incorrect_vs_correct_calculation_l1745_174557


namespace house_resale_price_l1745_174524

theorem house_resale_price (initial_value : ℝ) (loss_percent : ℝ) (interest_rate : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  interest_rate = 0.05 ∧ 
  gain_percent = 0.2 → 
  initial_value * (1 - loss_percent) * (1 + interest_rate) * (1 + gain_percent) = 12852 :=
by sorry

end house_resale_price_l1745_174524


namespace dance_attendance_l1745_174587

/-- The number of boys attending the dance -/
def num_boys : ℕ := 14

/-- The number of girls attending the dance -/
def num_girls : ℕ := num_boys / 2

theorem dance_attendance :
  (num_boys = 2 * num_girls) ∧
  (num_boys = (num_girls - 1) + 8) →
  num_boys = 14 :=
by sorry

end dance_attendance_l1745_174587


namespace no_solutions_absolute_value_equation_l1745_174506

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, x > 0 ∧ |x + 4| = 3 - x := by
  sorry

end no_solutions_absolute_value_equation_l1745_174506


namespace power_sum_equality_l1745_174509

theorem power_sum_equality : (-2)^48 + 3^(4^3 + 5^2 - 7^2) = 2^48 + 3^40 := by
  sorry

end power_sum_equality_l1745_174509


namespace computer_price_problem_l1745_174553

theorem computer_price_problem (P : ℝ) : 
  1.30 * P = 351 ∧ 2 * P = 540 → P = 270 := by
  sorry

end computer_price_problem_l1745_174553


namespace smallest_circle_passing_through_intersection_l1745_174515

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the smallest circle
def smallest_circle (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x - 18*y - 1 = 0

-- Theorem statement
theorem smallest_circle_passing_through_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ line x2 y2 ∧
    original_circle x1 y1 ∧ original_circle x2 y2 ∧
    (∀ (x y : ℝ), smallest_circle x y ↔ 
      ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2 ∧
       ∀ (c : ℝ → ℝ → Prop), (c x1 y1 ∧ c x2 y2) → 
         (∃ (xc yc r : ℝ), ∀ (x y : ℝ), c x y ↔ (x - xc)^2 + (y - yc)^2 = r^2) →
         (∃ (xs ys rs : ℝ), ∀ (x y : ℝ), smallest_circle x y ↔ (x - xs)^2 + (y - ys)^2 = rs^2 ∧ rs ≤ r))) :=
by sorry


end smallest_circle_passing_through_intersection_l1745_174515


namespace chess_tournament_players_l1745_174519

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not among the 12 lowest-scoring players
  total_players : ℕ := n + 12
  total_points : ℕ := n * (n - 1) + 132
  games_played : ℕ := (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players is 24 -/
theorem chess_tournament_players (t : ChessTournament) : t.total_players = 24 := by
  sorry

#check chess_tournament_players

end chess_tournament_players_l1745_174519


namespace best_of_three_match_probability_l1745_174597

/-- The probability of player A winning a single set -/
def p : ℝ := 0.6

/-- The probability of player A winning the match in a best-of-three format -/
def prob_A_wins_match : ℝ := p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_match_probability :
  prob_A_wins_match = 0.648 := by
  sorry

end best_of_three_match_probability_l1745_174597


namespace borrowed_sheets_theorem_l1745_174566

/-- Represents a set of notes with sheets and pages -/
structure Notes where
  total_sheets : ℕ
  pages_per_sheet : ℕ
  total_pages : ℕ
  h_pages : total_pages = total_sheets * pages_per_sheet

/-- Represents the state of notes after some sheets are borrowed -/
structure BorrowedNotes where
  original : Notes
  borrowed_sheets : ℕ
  sheets_before : ℕ
  h_valid : sheets_before + borrowed_sheets < original.total_sheets

/-- Calculates the average page number of remaining sheets -/
def average_page_number (bn : BorrowedNotes) : ℚ :=
  let remaining_pages := bn.original.total_pages - bn.borrowed_sheets * bn.original.pages_per_sheet
  let sum_before := bn.sheets_before * (bn.sheets_before * bn.original.pages_per_sheet + 1)
  let first_after := (bn.sheets_before + bn.borrowed_sheets) * bn.original.pages_per_sheet + 1
  let last_after := bn.original.total_pages
  let sum_after := (first_after + last_after) * (last_after - first_after + 1) / 2
  (sum_before + sum_after) / remaining_pages

/-- Theorem stating that if 17 sheets are borrowed from a 35-sheet set of notes,
    the average page number of remaining sheets is 28 -/
theorem borrowed_sheets_theorem (bn : BorrowedNotes)
  (h_total_sheets : bn.original.total_sheets = 35)
  (h_pages_per_sheet : bn.original.pages_per_sheet = 2)
  (h_total_pages : bn.original.total_pages = 70)
  (h_avg : average_page_number bn = 28) :
  bn.borrowed_sheets = 17 := by
  sorry

end borrowed_sheets_theorem_l1745_174566


namespace delta_composition_l1745_174578

-- Define the Delta operations
def rightDelta (x : ℤ) : ℤ := 9 - x
def leftDelta (x : ℤ) : ℤ := x - 9

-- State the theorem
theorem delta_composition : leftDelta (rightDelta 15) = -15 := by
  sorry

end delta_composition_l1745_174578


namespace greatest_integer_b_for_all_real_domain_l1745_174510

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ),
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 12 ≠ 0) ∧
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + (c : ℝ) * x + 12 = 0) ∧
  b = 6 := by
  sorry

end greatest_integer_b_for_all_real_domain_l1745_174510


namespace system_of_equations_l1745_174535

theorem system_of_equations (a b : ℝ) 
  (eq1 : 2020*a + 2024*b = 2040)
  (eq2 : 2022*a + 2026*b = 2050)
  (eq3 : 2025*a + 2028*b = 2065) :
  a + 2*b = 5 := by
sorry

end system_of_equations_l1745_174535


namespace point_inside_given_circle_l1745_174503

def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 18

def point_inside_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 < 18

theorem point_inside_given_circle :
  point_inside_circle 1 1 := by sorry

end point_inside_given_circle_l1745_174503


namespace division_remainder_problem_l1745_174548

theorem division_remainder_problem : ∃ (x : ℕ), 
  (1782 - x = 1500) ∧ 
  (∃ (r : ℕ), 1782 = 6 * x + r) ∧
  (1782 % x = 90) := by
  sorry

end division_remainder_problem_l1745_174548


namespace market_fruit_count_l1745_174571

/-- The number of apples in the market -/
def num_apples : ℕ := 164

/-- The difference between the number of apples and oranges -/
def apple_orange_diff : ℕ := 27

/-- The number of oranges in the market -/
def num_oranges : ℕ := num_apples - apple_orange_diff

/-- The total number of fruits (apples and oranges) in the market -/
def total_fruits : ℕ := num_apples + num_oranges

theorem market_fruit_count : total_fruits = 301 := by
  sorry

end market_fruit_count_l1745_174571


namespace fill_box_with_cubes_l1745_174527

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.depth

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the side length of the largest cube that can fit evenly into the box -/
def largestCubeSideLength (d : BoxDimensions) : ℕ :=
  gcd3 d.length d.width d.depth

/-- Calculates the number of cubes needed to fill the box completely -/
def numberOfCubes (d : BoxDimensions) : ℕ :=
  boxVolume d / (largestCubeSideLength d)^3

/-- The main theorem stating that 80 cubes are needed to fill the given box -/
theorem fill_box_with_cubes (d : BoxDimensions) 
  (h1 : d.length = 30) (h2 : d.width = 48) (h3 : d.depth = 12) : 
  numberOfCubes d = 80 := by
  sorry

end fill_box_with_cubes_l1745_174527


namespace august_mail_l1745_174551

def mail_sequence (n : ℕ) : ℕ := 5 * 2^n

theorem august_mail :
  mail_sequence 4 = 80 := by
  sorry

end august_mail_l1745_174551


namespace pascal_row_15_sum_l1745_174549

/-- Definition of Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in row 15 of Pascal's Triangle is 32768 -/
theorem pascal_row_15_sum : pascal_sum 15 = 32768 := by
  sorry

end pascal_row_15_sum_l1745_174549


namespace hyperbola_asymptote_l1745_174528

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 3x, then b = 3 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ x y : ℝ, x^2 - y^2/b^2 = 1 ∧ y = 3*x) → b = 3 := by
  sorry

end hyperbola_asymptote_l1745_174528


namespace gcd_problem_l1745_174507

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1177) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 := by sorry

end gcd_problem_l1745_174507


namespace tangent_point_for_equal_volume_l1745_174576

theorem tangent_point_for_equal_volume (ξ η : ℝ) : 
  ξ^2 + η^2 = 1 →  -- Point (ξ, η) is on the unit circle
  0 < ξ →          -- ξ is positive (first quadrant)
  ξ < 1 →          -- ξ is less than 1 (valid tangent)
  (((1 - ξ^2)^2 / (3 * ξ)) - ((1 - ξ)^2 * (2 + ξ) / 3)) * π = 4 * π / 3 →  -- Volume equation
  ξ = 3 - 2 * Real.sqrt 2 :=
by sorry

end tangent_point_for_equal_volume_l1745_174576


namespace lcm_problem_l1745_174567

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 90 := by
  sorry

end lcm_problem_l1745_174567


namespace system_solution_l1745_174504

theorem system_solution (x y z : ℚ) 
  (eq1 : y + z = 15 - 2*x)
  (eq2 : x + z = -10 - 2*y)
  (eq3 : x + y = 4 - 2*z) :
  2*x + 2*y + 2*z = 9/2 := by
  sorry

end system_solution_l1745_174504


namespace seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l1745_174572

theorem seventh_root_of_negative_two_plus_fourth_root_of_negative_three : 
  ((-2 : ℝ) ^ 7) ^ (1/7) + ((-3 : ℝ) ^ 4) ^ (1/4) = 1 := by
  sorry

end seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l1745_174572


namespace expression_simplification_l1745_174560

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hab : a^(1/3) * b^(1/4) ≠ 2) :
  ((a^2 * b * Real.sqrt b - 6 * a^(5/3) * b^(5/4) + 12 * a * b * a^(1/3) - 8 * a * b^(3/4))^(2/3)) /
  (a * b * a^(1/3) - 4 * a * b^(3/4) + 4 * a^(2/3) * Real.sqrt b) = 1 := by
  sorry

end expression_simplification_l1745_174560


namespace school_test_questions_l1745_174570

theorem school_test_questions (sections : ℕ) (correct_answers : ℕ) 
  (h_sections : sections = 5)
  (h_correct : correct_answers = 20)
  (h_percentage : ∀ x : ℕ, x > 0 → (60 : ℚ) / 100 < (correct_answers : ℚ) / x ∧ (correct_answers : ℚ) / x < 70 / 100 → x = 30) :
  ∃! total_questions : ℕ, 
    total_questions > 0 ∧
    total_questions % sections = 0 ∧
    (60 : ℚ) / 100 < (correct_answers : ℚ) / total_questions ∧
    (correct_answers : ℚ) / total_questions < 70 / 100 :=
by
  sorry

end school_test_questions_l1745_174570


namespace hedgehog_strawberry_baskets_l1745_174577

theorem hedgehog_strawberry_baskets :
  ∀ (baskets : ℕ) (strawberries_per_basket : ℕ) (hedgehogs : ℕ) (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_per_basket = 900 →
    hedgehogs = 2 →
    strawberries_eaten_per_hedgehog = 1050 →
    (baskets * strawberries_per_basket : ℚ) * (2 : ℚ) / 9 = 
      baskets * strawberries_per_basket - hedgehogs * strawberries_eaten_per_hedgehog →
    baskets = 3 := by
  sorry

end hedgehog_strawberry_baskets_l1745_174577


namespace total_winter_clothing_l1745_174583

def scarves_boxes : ℕ := 4
def scarves_per_box : ℕ := 8
def mittens_boxes : ℕ := 3
def mittens_per_box : ℕ := 6
def hats_boxes : ℕ := 2
def hats_per_box : ℕ := 5
def jackets_boxes : ℕ := 1
def jackets_per_box : ℕ := 3

theorem total_winter_clothing :
  scarves_boxes * scarves_per_box +
  mittens_boxes * mittens_per_box +
  hats_boxes * hats_per_box +
  jackets_boxes * jackets_per_box = 63 := by
sorry

end total_winter_clothing_l1745_174583


namespace tiles_needed_l1745_174530

-- Define the dimensions
def tile_size : ℕ := 6
def kitchen_width : ℕ := 48
def kitchen_height : ℕ := 72

-- Define the theorem
theorem tiles_needed : 
  (kitchen_width / tile_size) * (kitchen_height / tile_size) = 96 := by
  sorry

end tiles_needed_l1745_174530


namespace distribution_methods_count_l1745_174540

/-- The number of ways to distribute tickets to tourists -/
def distribute_tickets : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * (Nat.factorial 2)

/-- Theorem stating that the number of distribution methods is 180 -/
theorem distribution_methods_count : distribute_tickets = 180 := by
  sorry

end distribution_methods_count_l1745_174540


namespace two_students_choose_A_l1745_174544

/-- The number of ways to choose exactly two students from four to take course A -/
def waysToChooseTwoForA : ℕ := 24

/-- The number of students -/
def numStudents : ℕ := 4

/-- The number of courses -/
def numCourses : ℕ := 3

theorem two_students_choose_A :
  waysToChooseTwoForA = (numStudents.choose 2) * (2^(numStudents - 2)) :=
sorry

end two_students_choose_A_l1745_174544


namespace quadratic_inequality_solution_implies_function_order_l1745_174565

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_order
  (a b c : ℝ)
  (h1 : ∀ x, (x < -2 ∨ x > 4) ↔ a * x^2 + b * x + c < 0) :
  f a b c 5 < f a b c 2 ∧ f a b c 2 < f a b c 1 :=
sorry

end quadratic_inequality_solution_implies_function_order_l1745_174565


namespace closest_integer_to_sqrt13_l1745_174546

theorem closest_integer_to_sqrt13 : 
  ∀ n : ℤ, n ∈ ({2, 3, 4, 5} : Set ℤ) → |n - Real.sqrt 13| ≥ |4 - Real.sqrt 13| :=
by sorry

end closest_integer_to_sqrt13_l1745_174546


namespace repeating_decimal_difference_l1745_174589

/-- Proves that the difference between the repeating decimals 0.353535... and 0.777777... is equal to -14/33 -/
theorem repeating_decimal_difference : 
  (35 : ℚ) / 99 - (7 : ℚ) / 9 = -14 / 33 := by sorry

end repeating_decimal_difference_l1745_174589


namespace sphere_cone_intersection_l1745_174514

/-- Represents the geometry of a sphere and cone with intersecting plane -/
structure GeometrySetup where
  R : ℝ  -- Radius of sphere and base of cone
  m : ℝ  -- Distance from base plane to intersecting plane
  n : ℝ  -- Ratio of truncated cone volume to spherical segment volume

/-- The areas of the circles cut from the sphere and cone are equal -/
def equal_areas (g : GeometrySetup) : Prop :=
  g.m = 2 * g.R / 5 ∨ g.m = 2 * g.R

/-- The volume ratio condition is satisfied -/
def volume_ratio_condition (g : GeometrySetup) : Prop :=
  g.n ≥ 1 / 2

/-- Main theorem combining both conditions -/
theorem sphere_cone_intersection (g : GeometrySetup) :
  (equal_areas g ↔ (2 * g.R * g.m - g.m^2 = g.R^2 * (1 - g.m / (2 * g.R))^2)) ∧
  (volume_ratio_condition g ↔ 
    (π * g.m / 12 * (12 * g.R^2 - 6 * g.R * g.m + g.m^2) = 
     g.n * (π * g.m^2 / 3 * (3 * g.R - g.m)))) := by
  sorry

end sphere_cone_intersection_l1745_174514


namespace quadruple_solution_l1745_174523

theorem quadruple_solution :
  ∀ a b c d : ℕ+,
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d ∧ a * b = c + d →
    (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
    (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
    (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
    (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
    (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
    (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
    (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
    (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
by sorry

end quadruple_solution_l1745_174523


namespace main_theorem_l1745_174517

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem main_theorem (x : ℝ) (h : (lg x)^2 * lg (10 * x) < 0) :
  (1 / lg (10 * x)) * Real.sqrt ((lg x)^2 + (lg (10 * x))^2) = -1 :=
by sorry

end main_theorem_l1745_174517


namespace camera_profit_difference_l1745_174593

/-- Calculates the difference in profit between two camera sellers --/
theorem camera_profit_difference 
  (maddox_cameras : ℕ) (maddox_buy_price : ℚ) (maddox_sell_price : ℚ)
  (maddox_shipping : ℚ) (maddox_listing_fee : ℚ)
  (theo_cameras : ℕ) (theo_buy_price : ℚ) (theo_sell_price : ℚ)
  (theo_shipping : ℚ) (theo_listing_fee : ℚ)
  (h1 : maddox_cameras = 10)
  (h2 : maddox_buy_price = 35)
  (h3 : maddox_sell_price = 50)
  (h4 : maddox_shipping = 2)
  (h5 : maddox_listing_fee = 10)
  (h6 : theo_cameras = 15)
  (h7 : theo_buy_price = 30)
  (h8 : theo_sell_price = 40)
  (h9 : theo_shipping = 3)
  (h10 : theo_listing_fee = 15) :
  (maddox_cameras : ℚ) * maddox_sell_price - 
  (maddox_cameras : ℚ) * maddox_buy_price - 
  (maddox_cameras : ℚ) * maddox_shipping - 
  maddox_listing_fee -
  (theo_cameras : ℚ) * theo_sell_price + 
  (theo_cameras : ℚ) * theo_buy_price + 
  (theo_cameras : ℚ) * theo_shipping + 
  theo_listing_fee = 30 :=
by sorry

end camera_profit_difference_l1745_174593


namespace scientific_notation_of_13000_l1745_174547

theorem scientific_notation_of_13000 :
  ∃ (a : ℝ) (n : ℤ), 13000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.3 ∧ n = 4 :=
sorry

end scientific_notation_of_13000_l1745_174547


namespace area_after_reflection_l1745_174536

/-- Right triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  right_angle : AB > 0 ∧ BC > 0

/-- Points after reflection -/
structure ReflectedPoints where
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

/-- Function to perform reflections -/
def reflect (t : RightTriangle) : ReflectedPoints := sorry

/-- Calculate area of triangle A'B'C' -/
def area_A'B'C' (p : ReflectedPoints) : ℝ := sorry

/-- Main theorem -/
theorem area_after_reflection (t : RightTriangle) 
  (h1 : t.AB = 5)
  (h2 : t.BC = 12) : 
  area_A'B'C' (reflect t) = 17.5 := by sorry

end area_after_reflection_l1745_174536


namespace job_completion_relationship_l1745_174558

/-- Represents the relationship between number of machines and time to finish a job -/
theorem job_completion_relationship (D : ℝ) : 
  D > 0 → -- D is positive (time can't be negative or zero)
  (15 : ℝ) / 20 = (3 / 4 * D) / D := by
  sorry

#check job_completion_relationship

end job_completion_relationship_l1745_174558


namespace f_f_eq_f_solution_l1745_174559

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_f_eq_f_solution :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
  sorry

end f_f_eq_f_solution_l1745_174559


namespace probability_outside_circle_l1745_174532

/-- A die roll outcome is a natural number between 1 and 6 -/
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

/-- A point P is defined by two die roll outcomes -/
structure Point where
  m : DieRoll
  n : DieRoll

/-- A point P(m,n) is outside the circle if m^2 + n^2 > 25 -/
def isOutsideCircle (p : Point) : Prop :=
  (p.m.val ^ 2 + p.n.val ^ 2 : ℚ) > 25

/-- The total number of possible outcomes when rolling a die twice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes resulting in a point outside the circle -/
def favorableOutcomes : ℕ := 11

/-- The main theorem: probability of a point being outside the circle -/
theorem probability_outside_circle :
  (favorableOutcomes : ℚ) / totalOutcomes = 11 / 36 := by sorry

end probability_outside_circle_l1745_174532


namespace parabola_equation_l1745_174533

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through two points -/
structure Line where
  a : Point
  b : Point

theorem parabola_equation (c : Parabola) (l : Line) :
  let f := Point.mk (c.p / 2) 0  -- Focus of the parabola
  let m := Point.mk 3 2  -- Midpoint of AB
  (l.a.y ^ 2 = 2 * c.p * l.a.x) ∧  -- A is on the parabola
  (l.b.y ^ 2 = 2 * c.p * l.b.x) ∧  -- B is on the parabola
  ((l.a.x + l.b.x) / 2 = m.x) ∧  -- M is the midpoint of AB (x-coordinate)
  ((l.a.y + l.b.y) / 2 = m.y) ∧  -- M is the midpoint of AB (y-coordinate)
  (f.x - l.a.x) * (l.b.y - l.a.y) = (f.y - l.a.y) * (l.b.x - l.a.x)  -- L passes through F
  →
  c.p = 2 ∨ c.p = 4 :=
by sorry

end parabola_equation_l1745_174533


namespace nancy_homework_pages_l1745_174538

theorem nancy_homework_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : 
  total_problems = 101 → 
  finished_problems = 47 → 
  problems_per_page = 9 → 
  (total_problems - finished_problems) / problems_per_page = 6 := by
sorry

end nancy_homework_pages_l1745_174538


namespace product_of_primes_l1745_174516

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_three_digit_prime : Nat := 997

theorem product_of_primes :
  (smallest_one_digit_primes.prod * largest_three_digit_prime) = 5982 := by
  sorry

end product_of_primes_l1745_174516


namespace inequality_of_powers_l1745_174512

theorem inequality_of_powers (α : Real) (h : α ∈ Set.Ioo (π/4) (π/2)) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end inequality_of_powers_l1745_174512


namespace solve_for_y_l1745_174573

theorem solve_for_y (x y : ℝ) (h : 3 * x + 5 * y = 10) : y = 2 - (3/5) * x := by
  sorry

end solve_for_y_l1745_174573


namespace right_triangle_sides_l1745_174550

theorem right_triangle_sides : ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 40) →
  (a^2 + b^2 = c^2) →
  ((a + 4)^2 + (b + 1)^2 = (c + 3)^2) →
  (a < b) →
  (a = 8 ∧ b = 15 ∧ c = 17) := by
sorry

end right_triangle_sides_l1745_174550


namespace quadratic_inequality_sum_l1745_174592

theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end quadratic_inequality_sum_l1745_174592


namespace probability_two_red_balls_l1745_174539

/-- The probability of picking two red balls from a bag containing 4 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 4 →
  green_balls = 2 →
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 2 / 15 := by
  sorry

end probability_two_red_balls_l1745_174539


namespace unique_solution_for_cubic_equations_l1745_174585

/-- Represents the roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  r₁ : α
  r₂ : α
  r₃ : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y - x = z - y ∧ y - x ≠ 0

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  ∃ r : α, r ≠ 1 ∧ y = x * r ∧ z = y * r

/-- Represents the coefficients of the first cubic equation -/
structure FirstEquationCoeffs (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the coefficients of the second cubic equation -/
structure SecondEquationCoeffs (α : Type*) [Field α] where
  b : α
  c : α

/-- The main theorem -/
theorem unique_solution_for_cubic_equations 
  (f : FirstEquationCoeffs ℝ) 
  (g : SecondEquationCoeffs ℝ)
  (roots1 : CubicRoots ℝ)
  (roots2 : CubicRoots ℝ)
  (h1 : roots1.r₁^3 - 3*f.a*roots1.r₁^2 + f.b*roots1.r₁ + 18*f.c = 0)
  (h2 : roots1.r₂^3 - 3*f.a*roots1.r₂^2 + f.b*roots1.r₂ + 18*f.c = 0)
  (h3 : roots1.r₃^3 - 3*f.a*roots1.r₃^2 + f.b*roots1.r₃ + 18*f.c = 0)
  (h4 : is_arithmetic_progression roots1.r₁ roots1.r₂ roots1.r₃)
  (h5 : roots2.r₁^3 + g.b*roots2.r₁^2 + roots2.r₁ - g.c^3 = 0)
  (h6 : roots2.r₂^3 + g.b*roots2.r₂^2 + roots2.r₂ - g.c^3 = 0)
  (h7 : roots2.r₃^3 + g.b*roots2.r₃^2 + roots2.r₃ - g.c^3 = 0)
  (h8 : is_geometric_progression roots2.r₁ roots2.r₂ roots2.r₃)
  (h9 : f.b = g.b)
  (h10 : f.c = g.c)
  : f.a = 2 ∧ f.b = 9 := by sorry

end unique_solution_for_cubic_equations_l1745_174585


namespace no_100_equilateral_division_l1745_174500

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- An equilateral triangle -/
structure EquilateralTriangle where
  -- Add necessary fields and conditions for an equilateral triangle
  -- This is a simplified representation
  is_equilateral : Bool

/-- A division of a convex polygon into equilateral triangles -/
structure PolygonDivision (P : ConvexPolygon) where
  triangles : List EquilateralTriangle
  is_valid_division : Bool  -- This would ensure the division is valid

/-- Theorem stating that no convex polygon can be divided into 100 different equilateral triangles -/
theorem no_100_equilateral_division (P : ConvexPolygon) :
  ¬∃ (d : PolygonDivision P), d.is_valid_division ∧ d.triangles.length = 100 := by
  sorry

end no_100_equilateral_division_l1745_174500


namespace tiffany_bags_next_day_l1745_174554

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The additional number of bags Tiffany found on the next day compared to Monday -/
def additional_bags : ℕ := 5

/-- The total number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := monday_bags + additional_bags

theorem tiffany_bags_next_day : next_day_bags = 12 := by
  sorry

end tiffany_bags_next_day_l1745_174554


namespace solution_count_l1745_174501

theorem solution_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, 1 ≤ x ∧ x ≤ 200) ∧
  (∀ x ∈ S, ∃ k ∈ Finset.range 200, x = k + 1) ∧
  (∀ x ∈ S, ∀ k ∈ Finset.range 10, x ≠ (k + 1)^2) ∧
  Finset.card S = 190 := by
sorry

end solution_count_l1745_174501


namespace baker_cupcake_distribution_l1745_174590

/-- The number of cupcakes left over when distributing cupcakes equally -/
def cupcakes_left_over (total : ℕ) (children : ℕ) : ℕ :=
  total % children

/-- Theorem: When distributing 17 cupcakes among 3 children equally, 2 cupcakes are left over -/
theorem baker_cupcake_distribution :
  cupcakes_left_over 17 3 = 2 := by
  sorry

end baker_cupcake_distribution_l1745_174590


namespace participants_meet_on_DA_l1745_174562

/-- Represents a participant in the square walking problem -/
structure Participant where
  speed : ℝ
  startPoint : ℕ

/-- Represents the square and the walking problem -/
structure SquareWalk where
  sideLength : ℝ
  participantA : Participant
  participantB : Participant

/-- The point where the participants meet -/
def meetingPoint (sw : SquareWalk) : ℕ :=
  sorry

theorem participants_meet_on_DA (sw : SquareWalk) 
  (h1 : sw.sideLength = 90)
  (h2 : sw.participantA.speed = 65)
  (h3 : sw.participantB.speed = 72)
  (h4 : sw.participantA.startPoint = 0)
  (h5 : sw.participantB.startPoint = 1) :
  meetingPoint sw = 3 :=
sorry

end participants_meet_on_DA_l1745_174562


namespace combined_average_mark_l1745_174505

/-- Given two classes with specified number of students and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_mark (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / ((n1 : ℚ) + (n2 : ℚ)) =
  ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ)) := by
  sorry

#eval ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ))

end combined_average_mark_l1745_174505


namespace smallest_sum_of_coefficients_l1745_174588

theorem smallest_sum_of_coefficients (a b : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 20)*(x^2 + 17*x + b) = 0 → (∃ k : ℤ, x = ↑k ∧ k < 0)) →
  (∀ c d : ℤ, (∀ y : ℝ, (y^2 + c*y + 20)*(y^2 + 17*y + d) = 0 → (∃ m : ℤ, y = ↑m ∧ m < 0)) → 
    a + b ≤ c + d) →
  a + b = -5 := by
sorry

end smallest_sum_of_coefficients_l1745_174588


namespace complex_magnitude_product_l1745_174537

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_magnitude_product_l1745_174537


namespace system_solution_l1745_174521

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 3980 / 121 := by
sorry

end system_solution_l1745_174521


namespace power_product_equal_thousand_l1745_174534

theorem power_product_equal_thousand : 2^3 * 5^3 = 1000 := by
  sorry

end power_product_equal_thousand_l1745_174534


namespace proposition_truth_l1745_174508

theorem proposition_truth (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  ¬p ∧ q := by sorry

end proposition_truth_l1745_174508
