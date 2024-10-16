import Mathlib

namespace NUMINAMATH_CALUDE_two_by_one_parallelepiped_removals_l1587_158778

/-- Represents a position on the net of a parallelepiped --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the net of a parallelepiped --/
structure ParallelepipedNet :=
  (squares : List Position)
  (width : ℕ)
  (height : ℕ)

/-- Checks if a given position is valid for removal --/
def is_valid_removal (net : ParallelepipedNet) (pos : Position) : Prop := sorry

/-- Counts the number of valid removal positions --/
def count_valid_removals (net : ParallelepipedNet) : ℕ := sorry

/-- Creates a 2x1 parallelepiped net --/
def create_2x1_net : ParallelepipedNet := sorry

theorem two_by_one_parallelepiped_removals :
  count_valid_removals (create_2x1_net) = 5 := by sorry

end NUMINAMATH_CALUDE_two_by_one_parallelepiped_removals_l1587_158778


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1587_158745

/-- The total surface area of a cylinder formed by rolling a rectangle -/
theorem cylinder_surface_area (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 4 * Real.pi)
  (h2 : rectangle_width = 2) : 
  2 * Real.pi * (rectangle_width / 2)^2 + rectangle_length * rectangle_width = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1587_158745


namespace NUMINAMATH_CALUDE_at_least_one_geq_six_l1587_158716

theorem at_least_one_geq_six (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 4 / b ≥ 6) ∨ (b + 9 / c ≥ 6) ∨ (c + 16 / a ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_six_l1587_158716


namespace NUMINAMATH_CALUDE_fifty_black_reachable_l1587_158746

/-- Represents the number of marbles of each color in the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- The initial state of the urn -/
def initial_state : UrnState := ⟨50, 150⟩

/-- Applies one of the four possible operations to the urn -/
def apply_operation (state : UrnState) : UrnState :=
  match state with
  | ⟨w, b⟩ => 
    if b ≥ 3 then ⟨w, b - 1⟩  -- Operation 1
    else if b ≥ 2 ∧ w ≥ 1 then ⟨w, b - 1⟩  -- Operation 2
    else if b ≥ 1 ∧ w ≥ 2 then state  -- Operation 3
    else if w ≥ 3 then ⟨w - 2, b⟩  -- Operation 4
    else state  -- No operation possible

/-- Checks if the given state is reachable from the initial state -/
def is_reachable (state : UrnState) : Prop :=
  ∃ n : ℕ, (n.iterate apply_operation initial_state).black = state.black

/-- The theorem to be proven -/
theorem fifty_black_reachable :
  ∃ w : ℕ, is_reachable ⟨w, 50⟩ :=
sorry

end NUMINAMATH_CALUDE_fifty_black_reachable_l1587_158746


namespace NUMINAMATH_CALUDE_total_trees_count_l1587_158753

/-- Represents the number of Douglas fir trees -/
def D : ℕ := 350

/-- Represents the number of ponderosa pine trees -/
def P : ℕ := 500

/-- The cost of a Douglas fir tree -/
def douglas_cost : ℕ := 300

/-- The cost of a ponderosa pine tree -/
def ponderosa_cost : ℕ := 225

/-- The total cost paid for all trees -/
def total_cost : ℕ := 217500

/-- Theorem stating that given the conditions, the total number of trees is 850 -/
theorem total_trees_count : D + P = 850 ∧ 
  douglas_cost * D + ponderosa_cost * P = total_cost ∧
  (D = 350 ∨ P = 350) := by
  sorry

#check total_trees_count

end NUMINAMATH_CALUDE_total_trees_count_l1587_158753


namespace NUMINAMATH_CALUDE_toms_total_coins_l1587_158700

/-- Represents the number of coins Tom has -/
structure TomCoins where
  quarters : ℕ
  nickels : ℕ

/-- The total number of coins Tom has -/
def total_coins (c : TomCoins) : ℕ :=
  c.quarters + c.nickels

/-- Tom's actual coin count -/
def toms_coins : TomCoins :=
  { quarters := 4, nickels := 8 }

theorem toms_total_coins :
  total_coins toms_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_toms_total_coins_l1587_158700


namespace NUMINAMATH_CALUDE_work_of_two_springs_in_series_l1587_158720

/-- The work required to stretch a system of two springs in series -/
theorem work_of_two_springs_in_series 
  (k₁ k₂ : Real) 
  (x : Real) 
  (h₁ : k₁ = 3000) -- 3 kN/m = 3000 N/m
  (h₂ : k₂ = 6000) -- 6 kN/m = 6000 N/m
  (h₃ : x = 0.05)  -- 5 cm = 0.05 m
  : (1/2) * (1 / (1/k₁ + 1/k₂)) * x^2 = 2.5 := by
  sorry

#check work_of_two_springs_in_series

end NUMINAMATH_CALUDE_work_of_two_springs_in_series_l1587_158720


namespace NUMINAMATH_CALUDE_probability_one_absent_one_present_l1587_158791

/-- The probability of a student being absent on any given day -/
def p_absent : ℚ := 1 / 20

/-- The probability of a student being present on any given day -/
def p_present : ℚ := 1 - p_absent

/-- The probability that among any two randomly selected students, one is absent and the other present on a given day -/
def p_one_absent_one_present : ℚ := 2 * p_absent * p_present

theorem probability_one_absent_one_present :
  p_one_absent_one_present = 19 / 200 :=
sorry

end NUMINAMATH_CALUDE_probability_one_absent_one_present_l1587_158791


namespace NUMINAMATH_CALUDE_football_game_ratio_l1587_158773

theorem football_game_ratio : 
  -- Given conditions
  let total_start : ℕ := 600
  let girls_start : ℕ := 240
  let remaining : ℕ := 480
  let girls_left : ℕ := girls_start / 8

  -- Derived values
  let boys_start : ℕ := total_start - girls_start
  let total_left : ℕ := total_start - remaining
  let boys_left : ℕ := total_left - girls_left

  -- Theorem statement
  boys_left * 4 = boys_start :=
by sorry

end NUMINAMATH_CALUDE_football_game_ratio_l1587_158773


namespace NUMINAMATH_CALUDE_chip_cost_theorem_l1587_158771

theorem chip_cost_theorem (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℕ) (target_calories : ℕ) : 
  calories_per_chip = 10 →
  chips_per_bag = 24 →
  cost_per_bag = 2 →
  target_calories = 480 →
  (target_calories / (calories_per_chip * chips_per_bag)) * cost_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_theorem_l1587_158771


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1587_158761

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 ≥ 10.1 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 = 10.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1587_158761


namespace NUMINAMATH_CALUDE_expression_value_at_five_l1587_158788

theorem expression_value_at_five :
  let x : ℚ := 5
  (x^2 + x - 12) / (x - 4) = 18 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_five_l1587_158788


namespace NUMINAMATH_CALUDE_correct_arrangements_l1587_158781

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculates the number of arrangements given specific conditions -/
noncomputable def arrangements (condition : ℕ) : ℕ :=
  match condition with
  | 1 => 3720  -- A cannot stand at the head, and B cannot stand at the tail
  | 2 => 720   -- A, B, and C must stand next to each other
  | 3 => 1440  -- A, B, and C must not stand next to each other
  | 4 => 1200  -- There is exactly one person between A and B
  | 5 => 840   -- A, B, and C must stand in order from left to right
  | _ => 0     -- Invalid condition

/-- Theorem stating the correct number of arrangements for each condition -/
theorem correct_arrangements :
  (arrangements 1 = 3720) ∧
  (arrangements 2 = 720) ∧
  (arrangements 3 = 1440) ∧
  (arrangements 4 = 1200) ∧
  (arrangements 5 = 840) :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l1587_158781


namespace NUMINAMATH_CALUDE_correct_yellow_balls_drawn_l1587_158799

/-- Calculates the number of yellow balls to be drawn in a stratified sampling -/
def yellowBallsToDraw (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (yellowBalls * sampleSize) / totalBalls

theorem correct_yellow_balls_drawn (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) 
    (h1 : totalBalls = 800) 
    (h2 : yellowBalls = 40) 
    (h3 : sampleSize = 60) : 
  yellowBallsToDraw totalBalls yellowBalls sampleSize = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_yellow_balls_drawn_l1587_158799


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l1587_158707

theorem dress_shop_inventory (total_space : ℕ) (blue_extra : ℕ) (red_dresses : ℕ) 
  (h1 : total_space = 200)
  (h2 : blue_extra = 34)
  (h3 : red_dresses + (red_dresses + blue_extra) = total_space) :
  red_dresses = 83 := by
sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l1587_158707


namespace NUMINAMATH_CALUDE_always_odd_l1587_158756

theorem always_odd (p m : ℤ) (h_p : Odd p) : Odd (p^3 + 3*p*m^2 + 2*m) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1587_158756


namespace NUMINAMATH_CALUDE_set_equality_l1587_158704

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l1587_158704


namespace NUMINAMATH_CALUDE_prism_diagonals_l1587_158750

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (p : RectangularPrism) : ℕ :=
  12 + 4  -- 12 face diagonals + 4 space diagonals

/-- Theorem: A rectangular prism with dimensions 4, 3, and 5 has 16 diagonals -/
theorem prism_diagonals :
  let p : RectangularPrism := ⟨4, 3, 5⟩
  num_diagonals p = 16 := by
  sorry

end NUMINAMATH_CALUDE_prism_diagonals_l1587_158750


namespace NUMINAMATH_CALUDE_hyperbola_and_line_properties_l1587_158714

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the line MN
def line_MN (x y : ℝ) : Prop := 3 * x - 2 * y - 18 = 0

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := 3 * x + 4 * y = 0

-- Theorem statement
theorem hyperbola_and_line_properties :
  ∃ (x y : ℝ), 
    -- The hyperbola passes through (8, 3√3)
    hyperbola 8 (3 * Real.sqrt 3) ∧
    -- The point (8, 3) bisects a chord MN on the hyperbola
    ∃ (x1 y1 x2 y2 : ℝ),
      hyperbola x1 y1 ∧ 
      hyperbola x2 y2 ∧
      line_MN x1 y1 ∧
      line_MN x2 y2 ∧
      (x1 + x2) / 2 = 8 ∧
      (y1 + y2) / 2 = 3 ∧
    -- The asymptotes are given by 3x + 4y = 0
    (∀ (x y : ℝ), asymptotes x y ↔ (x = y ∨ x = -y)) →
    -- The equation of the hyperbola is x²/16 - y²/9 = 1
    (∀ (x y : ℝ), hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) ∧
    -- The equation of the line containing MN is 3x - 2y - 18 = 0
    (∀ (x y : ℝ), line_MN x y ↔ 3 * x - 2 * y - 18 = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_properties_l1587_158714


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1587_158708

/-- Given a quadratic function f(x) = x^2 + ax + b with roots -2 and 3, prove that a + b = -7 -/
theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2 ∨ x = 3) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1587_158708


namespace NUMINAMATH_CALUDE_faster_walking_speed_l1587_158775

/-- Given a person who walked 50 km at 10 km/hr, if they had walked at a faster speed
    that would allow them to cover an additional 20 km in the same time,
    prove that the faster speed would be 14 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l1587_158775


namespace NUMINAMATH_CALUDE_percent_boys_in_class_l1587_158774

theorem percent_boys_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 49)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42.86 := by
  sorry

end NUMINAMATH_CALUDE_percent_boys_in_class_l1587_158774


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1587_158709

theorem sufficient_not_necessary (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0 → (x + 3) * (y - 4) = 0) ∧
  ¬((x + 3) * (y - 4) = 0 → (x + 3)^2 + (y - 4)^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1587_158709


namespace NUMINAMATH_CALUDE_plot_length_is_60_l1587_158757

/-- Proves that the length of a rectangular plot is 60 meters given the specified conditions -/
theorem plot_length_is_60 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_60_l1587_158757


namespace NUMINAMATH_CALUDE_calculate_expression_l1587_158713

theorem calculate_expression : (-1)^2 + (1/3)^0 = 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1587_158713


namespace NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_sixteen_l1587_158735

/-- An isosceles triangle with side lengths 8, 8, and 15 -/
structure IsoscelesTriangle where
  /-- The length of the two equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The side length is 8 -/
  side_eq_eight : side = 8
  /-- The base length is 15 -/
  base_eq_fifteen : base = 15

/-- The sum of the lengths of the two longest altitudes in the isosceles triangle -/
def sum_two_longest_altitudes (t : IsoscelesTriangle) : ℝ := 2 * t.side

/-- Theorem: The sum of the lengths of the two longest altitudes in the given isosceles triangle is 16 -/
theorem sum_two_longest_altitudes_eq_sixteen (t : IsoscelesTriangle) : 
  sum_two_longest_altitudes t = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_sixteen_l1587_158735


namespace NUMINAMATH_CALUDE_sleep_stats_l1587_158715

def sleep_times : List ℕ := [7, 8, 9, 10]
def frequencies : List ℕ := [6, 9, 11, 4]

def mode (times : List ℕ) (freqs : List ℕ) : ℕ := sorry

def median (times : List ℕ) (freqs : List ℕ) : ℚ := sorry

theorem sleep_stats :
  mode sleep_times frequencies = 9 ∧ 
  median sleep_times frequencies = 17/2 := by sorry

end NUMINAMATH_CALUDE_sleep_stats_l1587_158715


namespace NUMINAMATH_CALUDE_min_surface_area_height_l1587_158780

/-- Represents a square-bottomed, lidless rectangular tank -/
structure Tank where
  side : ℝ
  height : ℝ

/-- The volume of the tank -/
def volume (t : Tank) : ℝ := t.side^2 * t.height

/-- The surface area of the tank -/
def surfaceArea (t : Tank) : ℝ := t.side^2 + 4 * t.side * t.height

/-- Theorem: For a tank with volume 4, the height that minimizes surface area is 1 -/
theorem min_surface_area_height :
  ∃ (t : Tank), volume t = 4 ∧ 
    (∀ (t' : Tank), volume t' = 4 → surfaceArea t ≤ surfaceArea t') ∧
    t.height = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_height_l1587_158780


namespace NUMINAMATH_CALUDE_tenth_term_is_21_l1587_158724

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 15
  geometric : (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13)

/-- The 10th term of the arithmetic sequence is 21 -/
theorem tenth_term_is_21 (seq : ArithmeticSequence) : seq.a 10 = 21 := by
  sorry

#check tenth_term_is_21

end NUMINAMATH_CALUDE_tenth_term_is_21_l1587_158724


namespace NUMINAMATH_CALUDE_trailing_zeros_count_product_trailing_zeros_l1587_158718

def product : ℕ := 25^7 * 8^3

theorem trailing_zeros_count (n : ℕ) : ℕ :=
  sorry

theorem product_trailing_zeros : trailing_zeros_count product = 9 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_product_trailing_zeros_l1587_158718


namespace NUMINAMATH_CALUDE_orange_juice_percentage_approx_48_l1587_158710

/-- Represents the juice yield from a specific fruit -/
structure JuiceYield where
  fruit : String
  count : Nat
  ounces : Rat

/-- Calculates the juice blend composition and returns the percentage of orange juice -/
def orangeJuicePercentage (appleYield pearYield orangeYield : JuiceYield) : Rat :=
  let appleJuicePerFruit := appleYield.ounces / appleYield.count
  let pearJuicePerFruit := pearYield.ounces / pearYield.count
  let orangeJuicePerFruit := orangeYield.ounces / orangeYield.count
  let totalJuice := appleJuicePerFruit + pearJuicePerFruit + orangeJuicePerFruit
  (orangeJuicePerFruit / totalJuice) * 100

/-- Theorem stating that the percentage of orange juice in the blend is approximately 48% -/
theorem orange_juice_percentage_approx_48 (appleYield pearYield orangeYield : JuiceYield) 
  (h1 : appleYield.fruit = "apple" ∧ appleYield.count = 5 ∧ appleYield.ounces = 9)
  (h2 : pearYield.fruit = "pear" ∧ pearYield.count = 4 ∧ pearYield.ounces = 10)
  (h3 : orangeYield.fruit = "orange" ∧ orangeYield.count = 3 ∧ orangeYield.ounces = 12) :
  ∃ (ε : Rat), abs (orangeJuicePercentage appleYield pearYield orangeYield - 48) < ε ∧ ε < 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_approx_48_l1587_158710


namespace NUMINAMATH_CALUDE_polynomial_division_l1587_158782

theorem polynomial_division (x y : ℝ) (h : y ≠ 0) : (3 * x * y + y) / y = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1587_158782


namespace NUMINAMATH_CALUDE_min_decimal_digits_l1587_158755

def fraction_numerator : ℕ := 987654321
def fraction_denominator : ℕ := 2^30 * 5^6 * 3

theorem min_decimal_digits : ℕ := by
  -- The minimum number of digits to the right of the decimal point
  -- needed to express fraction_numerator / fraction_denominator as a decimal is 30
  sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l1587_158755


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l1587_158747

/-- Given two points as the endpoints of a circle's diameter, prove the equation of the circle -/
theorem circle_equation_from_diameter (p1 p2 : ℝ × ℝ) :
  p1 = (-1, 3) →
  p2 = (5, -5) →
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (x^2 + y^2 + a*x + b*y + c = 0) ↔
    ((x - ((p1.1 + p2.1) / 2))^2 + (y - ((p1.2 + p2.2) / 2))^2 = ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l1587_158747


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_pi_over_six_l1587_158784

theorem sum_of_solutions_is_pi_over_six :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π ∧
  (1 / Real.sin x) + (1 / Real.cos x) = 2 * Real.sqrt 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π ∧
    (1 / Real.sin y) + (1 / Real.cos y) = 2 * Real.sqrt 3 →
    y = x) ∧
  x = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_pi_over_six_l1587_158784


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l1587_158734

theorem unique_prime_triplet : 
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ∧
    p = 5 ∧ q = 3 ∧ r = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l1587_158734


namespace NUMINAMATH_CALUDE_extreme_value_at_negative_three_l1587_158737

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_negative_three_l1587_158737


namespace NUMINAMATH_CALUDE_simplify_expression_l1587_158752

theorem simplify_expression : (-5)^2 - Real.sqrt 3 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1587_158752


namespace NUMINAMATH_CALUDE_sum_rational_irrational_not_rational_l1587_158703

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sum_rational_irrational_not_rational :
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_rational_irrational_not_rational_l1587_158703


namespace NUMINAMATH_CALUDE_percentage_relationship_l1587_158744

theorem percentage_relationship (x y : ℝ) (h : y = x * 1.6) :
  x = y * (1 - 0.375) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1587_158744


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1587_158798

theorem election_votes_calculation (total_votes : ℕ) : 
  (4 : ℕ) ≤ total_votes ∧ 
  (total_votes : ℚ) * (1/2) - (total_votes : ℚ) * (1/4) = 174 →
  total_votes = 696 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1587_158798


namespace NUMINAMATH_CALUDE_existence_of_divisible_n_l1587_158736

theorem existence_of_divisible_n : ∃ (n : ℕ), n > 0 ∧ (2009 * 2010 * 2011) ∣ ((n^2 - 5) * (n^2 + 6) * (n^2 + 30)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_n_l1587_158736


namespace NUMINAMATH_CALUDE_min_sum_pqr_l1587_158795

/-- Given five positive integers with pairwise GCDs as specified, 
    the minimum sum of p, q, and r is 9 -/
theorem min_sum_pqr (a b c d e : ℕ+) 
  (h : ∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val}) : 
  (∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val} ∧ 
    p.val + q.val + r.val = 9 ∧ 
    ∀ (p' q' r' : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
      Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
      Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
      Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
      Set.toFinset {2, 3, 4, 5, 6, 7, 8, p'.val, q'.val, r'.val} → 
      p'.val + q'.val + r'.val ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_pqr_l1587_158795


namespace NUMINAMATH_CALUDE_bathroom_visits_time_l1587_158777

/-- Given that it takes 20 minutes for 8 bathroom visits, prove that 6 visits take 15 minutes. -/
theorem bathroom_visits_time (total_time : ℝ) (total_visits : ℕ) (target_visits : ℕ)
  (h1 : total_time = 20)
  (h2 : total_visits = 8)
  (h3 : target_visits = 6) :
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visits_time_l1587_158777


namespace NUMINAMATH_CALUDE_plums_picked_total_l1587_158741

/-- The number of plums Melanie picked -/
def melanie_plums : ℕ := 4

/-- The number of plums Dan picked -/
def dan_plums : ℕ := 9

/-- The number of plums Sally picked -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plums_picked_total : total_plums = 16 := by
  sorry

end NUMINAMATH_CALUDE_plums_picked_total_l1587_158741


namespace NUMINAMATH_CALUDE_coupon_one_best_l1587_158748

/-- Represents the discount amount for a given coupon and price -/
def discount (coupon : Nat) (price : ℝ) : ℝ :=
  match coupon with
  | 1 => 0.15 * price
  | 2 => 30
  | 3 => 0.25 * (price - 120)
  | _ => 0

/-- Theorem stating when coupon 1 is the best choice -/
theorem coupon_one_best (price : ℝ) :
  (∀ c : Nat, c ≠ 1 → discount 1 price > discount c price) ↔ 200 < price ∧ price < 300 := by
  sorry


end NUMINAMATH_CALUDE_coupon_one_best_l1587_158748


namespace NUMINAMATH_CALUDE_congruence_problem_l1587_158701

theorem congruence_problem : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 12345 % 7 → n = 11 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1587_158701


namespace NUMINAMATH_CALUDE_square_sum_equals_ten_l1587_158729

theorem square_sum_equals_ten : 2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_ten_l1587_158729


namespace NUMINAMATH_CALUDE_sum_of_roots_l1587_158787

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 8*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 8*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1248 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1587_158787


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1587_158726

theorem quadratic_roots_relation (b c p q r s : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 + p*x + q = 0 ↔ x = r^2 ∨ x = s^2) →
  p = 2*c - b^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1587_158726


namespace NUMINAMATH_CALUDE_f_plus_two_is_odd_l1587_158776

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- State the theorem
theorem f_plus_two_is_odd (h : satisfies_property f) : is_odd (λ x => f x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_two_is_odd_l1587_158776


namespace NUMINAMATH_CALUDE_apple_distribution_l1587_158790

theorem apple_distribution (x : ℕ) : 
  x > 0 →
  x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50 →
  x = 3360 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l1587_158790


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1587_158738

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 34) 
  (product_condition : x * y = 240) : 
  |x - y| = 14 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1587_158738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1587_158749

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 3) ∧ a 11 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1587_158749


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_3_closed_2_l1587_158702

-- Define the sets A and B
def A : Set ℝ := {t : ℝ | ∀ x : ℝ, x^2 + 2*t*x - 4*t - 3 ≠ 0}
def B : Set ℝ := {t : ℝ | ∃ x : ℝ, x^2 + 2*t*x - 2*t = 0}

-- State the theorem
theorem A_intersect_B_eq_open_3_closed_2 : A ∩ B = Set.Ioc (-3) (-2) := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_3_closed_2_l1587_158702


namespace NUMINAMATH_CALUDE_first_class_average_l1587_158768

/-- Proves that given two classes with specified student counts and averages,
    the average of the first class can be determined. -/
theorem first_class_average
  (n1 : ℕ)  -- number of students in the first class
  (n2 : ℕ)  -- number of students in the second class
  (a2 : ℝ)  -- average marks of the second class
  (a_total : ℝ)  -- average marks of all students
  (h1 : n1 = 35)
  (h2 : n2 = 45)
  (h3 : a2 = 60)
  (h4 : a_total = 51.25)
  (h5 : n1 + n2 > 0)  -- to ensure division by zero is avoided
  : ∃ (a1 : ℝ), a1 = 40 ∧ (n1 * a1 + n2 * a2) / (n1 + n2) = a_total :=
sorry

end NUMINAMATH_CALUDE_first_class_average_l1587_158768


namespace NUMINAMATH_CALUDE_triangle_inradius_l1587_158772

/-- Given a triangle with perimeter 20 cm and area 25 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 20) 
  (h_area : A = 25) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1587_158772


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1587_158732

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1587_158732


namespace NUMINAMATH_CALUDE_richards_average_touchdowns_l1587_158758

def archie_record : ℕ := 89
def total_games : ℕ := 16
def richards_games : ℕ := 14
def remaining_games : ℕ := 2
def remaining_avg : ℕ := 3

theorem richards_average_touchdowns :
  let total_touchdowns := archie_record + 1
  let remaining_touchdowns := remaining_games * remaining_avg
  let richards_touchdowns := total_touchdowns - remaining_touchdowns
  (richards_touchdowns : ℚ) / richards_games = 6 := by sorry

end NUMINAMATH_CALUDE_richards_average_touchdowns_l1587_158758


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1587_158797

theorem inscribed_cube_surface_area (outer_cube_surface_area : ℝ) :
  outer_cube_surface_area = 54 →
  ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 ∧
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_side : ℝ),
      outer_cube_side^3 = outer_cube_surface_area / 6 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_side^2 * 3 = sphere_diameter^2 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1587_158797


namespace NUMINAMATH_CALUDE_number_comparisons_l1587_158762

theorem number_comparisons :
  (-7 / 8 : ℚ) > (-8 / 9 : ℚ) ∧ -|(-5)| < -(-4) := by sorry

end NUMINAMATH_CALUDE_number_comparisons_l1587_158762


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1587_158723

theorem quadratic_equations_solutions :
  (∃ (s : Set ℝ), s = {x : ℝ | 2 * x^2 - x = 0} ∧ s = {0, 1/2}) ∧
  (∃ (t : Set ℝ), t = {x : ℝ | (2 * x + 1)^2 - 9 = 0} ∧ t = {1, -2}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1587_158723


namespace NUMINAMATH_CALUDE_frog_border_probability_l1587_158759

/-- Represents a position on the 4x4 grid -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines the grid and movement rules -/
def Grid :=
  { pos : Position // true }

/-- Determines if a position is on the border of the grid -/
def is_border (pos : Position) : Bool :=
  pos.x = 0 ∨ pos.x = 3 ∨ pos.y = 0 ∨ pos.y = 3

/-- Calculates the next position after a hop in a given direction -/
def next_position (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨(pos.x + 1) % 4, pos.y⟩
  | Direction.Down => ⟨(pos.x + 3) % 4, pos.y⟩
  | Direction.Left => ⟨pos.x, (pos.y + 3) % 4⟩
  | Direction.Right => ⟨pos.x, (pos.y + 1) % 4⟩

/-- Calculates the probability of reaching the border within n hops -/
def border_probability (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to be proved -/
theorem frog_border_probability :
  border_probability ⟨1, 1⟩ 3 = 39 / 64 :=
sorry

end NUMINAMATH_CALUDE_frog_border_probability_l1587_158759


namespace NUMINAMATH_CALUDE_tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l1587_158739

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

-- Theorem for part 1
theorem tangent_line_condition (a : ℝ) :
  (∃ k, ∀ x, f a x = 2 * x + k - 2 * Real.exp 1) → a = Real.exp 1 :=
sorry

-- Theorem for part 2
theorem minimum_value_condition (a m : ℝ) (h : m > 0) :
  let min_value := min (f a (2 * m)) (min (f a (1 / Real.exp 1)) (f a m))
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ min_value :=
sorry

-- Additional theorems to specify the exact minimum value based on m
theorem min_value_case1 (a m : ℝ) (h1 : m > 0) (h2 : m ≤ 1 / (2 * Real.exp 1)) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (2 * m) :=
sorry

theorem min_value_case2 (a m : ℝ) (h1 : m > 0) (h2 : 1 / (2 * Real.exp 1) < m) (h3 : m < 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (1 / Real.exp 1) :=
sorry

theorem min_value_case3 (a m : ℝ) (h1 : m > 0) (h2 : m ≥ 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a m :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l1587_158739


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1587_158727

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1587_158727


namespace NUMINAMATH_CALUDE_max_a_value_l1587_158731

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -28) → 
  (a > 0) → 
  a ≤ 29 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1587_158731


namespace NUMINAMATH_CALUDE_field_division_theorem_l1587_158779

/-- Represents a rectangular field of squares -/
structure RectangularField where
  width : ℕ
  height : ℕ
  total_squares : ℕ
  h_total : total_squares = width * height

/-- Represents a line dividing the field -/
structure DividingLine where
  x : ℕ
  y : ℕ

/-- Calculates the area of a triangle formed by a dividing line -/
def triangle_area (field : RectangularField) (line : DividingLine) : ℕ :=
  line.x * line.y / 2

theorem field_division_theorem (field : RectangularField) 
  (h_total : field.total_squares = 18) 
  (line1 : DividingLine) 
  (line2 : DividingLine) 
  (h_line1 : line1 = ⟨4, field.height⟩) 
  (h_line2 : line2 = ⟨field.width, 2⟩) :
  triangle_area field line1 = 6 ∧ 
  triangle_area field line2 = 6 ∧ 
  field.total_squares - triangle_area field line1 - triangle_area field line2 = 6 := by
  sorry

#check field_division_theorem

end NUMINAMATH_CALUDE_field_division_theorem_l1587_158779


namespace NUMINAMATH_CALUDE_city_population_growth_l1587_158764

/-- Represents the birth rate and death rate in a city, and proves the birth rate given conditions --/
theorem city_population_growth (death_rate : ℕ) (net_increase : ℕ) (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 43200)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 4 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
sorry

end NUMINAMATH_CALUDE_city_population_growth_l1587_158764


namespace NUMINAMATH_CALUDE_equation_equivalence_product_l1587_158721

theorem equation_equivalence_product (a b x y : ℝ) (m n p q : ℤ) :
  (a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) →
  ((a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4) →
  m * n * p * q = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_product_l1587_158721


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1587_158763

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ 
  b / Real.sin B = c / Real.sin C ∧
  Real.sin B ^ 2 + Real.sin C ^ 2 = Real.sin A ^ 2 - Real.sqrt 3 * Real.sin B * Real.sin C →
  Real.cos A = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1587_158763


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l1587_158711

theorem average_of_five_numbers (total : ℕ) (avg_all : ℚ) (avg_three : ℚ) :
  total = 8 →
  avg_all = 20 →
  avg_three = 33333333333333336 / 1000000000000000 →
  let sum_all := avg_all * total
  let sum_three := avg_three * 3
  let sum_five := sum_all - sum_three
  sum_five / 5 = 12 := by
sorry

#eval 33333333333333336 / 1000000000000000  -- To verify the fraction equals 33.333333333333336

end NUMINAMATH_CALUDE_average_of_five_numbers_l1587_158711


namespace NUMINAMATH_CALUDE_school_distance_l1587_158754

/-- The distance between a child's home and school, given two walking scenarios. -/
theorem school_distance (v₁ v₂ : ℝ) (t₁ t₂ : ℝ) (D : ℝ) : 
  v₁ = 5 →  -- First walking speed in m/min
  v₂ = 7 →  -- Second walking speed in m/min
  t₁ = 6 →  -- Late time in minutes for first scenario
  t₂ = 30 → -- Early time in minutes for second scenario
  v₁ * (D / v₁ + t₁) = D →  -- Equation for first scenario
  v₂ * (D / v₂ - t₂) = D →  -- Equation for second scenario
  D = 630 := by
sorry

end NUMINAMATH_CALUDE_school_distance_l1587_158754


namespace NUMINAMATH_CALUDE_black_area_after_changes_l1587_158743

/-- Represents the fraction of area that remains black after each change -/
def remaining_black_fraction : ℚ := 8 / 9

/-- The number of times the process is repeated -/
def num_changes : ℕ := 6

/-- The fraction of the original area that remains black after all changes -/
def final_black_fraction : ℚ := remaining_black_fraction ^ num_changes

theorem black_area_after_changes : 
  final_black_fraction = 262144 / 531441 := by sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l1587_158743


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1587_158712

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 1 ↔ (x : ℚ) / 3 + 7 / 4 < 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1587_158712


namespace NUMINAMATH_CALUDE_point_ordering_l1587_158705

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem point_ordering :
  let y₁ := f (-3)
  let y₂ := f 1
  let y₃ := f (-1/2)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_point_ordering_l1587_158705


namespace NUMINAMATH_CALUDE_two_digit_square_sum_l1587_158789

/-- Two-digit integer -/
def TwoDigitInt (x : ℕ) : Prop := 10 ≤ x ∧ x < 100

/-- Reverse digits of a two-digit integer -/
def reverseDigits (x : ℕ) : ℕ := 
  let tens := x / 10
  let ones := x % 10
  10 * ones + tens

theorem two_digit_square_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 + y^2 = n^2 → x + y + n = 264 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_square_sum_l1587_158789


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1587_158722

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_yellow = 0.2 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_green = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1587_158722


namespace NUMINAMATH_CALUDE_ball_arrangements_l1587_158725

def num_balls : ℕ := 5
def num_black_balls : ℕ := 2
def num_colored_balls : ℕ := 3
def balls_in_row : ℕ := 4

theorem ball_arrangements :
  (Nat.factorial balls_in_row) + 
  (num_colored_balls * (Nat.factorial balls_in_row) / 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_l1587_158725


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l1587_158793

theorem simultaneous_inequalities (a b : ℝ) :
  (a < b ∧ 1 / a < 1 / b) ↔ a < 0 ∧ 0 < b := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l1587_158793


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l1587_158785

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (side_relation : a^2 + b^2 = (5/2) * c^2) : 
  ∃ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) ∧ cos_C ≥ 3/5 :=
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l1587_158785


namespace NUMINAMATH_CALUDE_right_triangle_leg_l1587_158760

theorem right_triangle_leg (a c : ℝ) (h1 : a = 12) (h2 : c = 13) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_l1587_158760


namespace NUMINAMATH_CALUDE_internet_cost_proof_l1587_158783

/-- The daily cost of internet service -/
def daily_cost : ℝ := 0.28

/-- The number of days covered by the payment -/
def days_covered : ℕ := 25

/-- The amount paid -/
def payment : ℝ := 7

/-- The maximum allowed debt -/
def max_debt : ℝ := 5

theorem internet_cost_proof :
  daily_cost * days_covered = payment ∧
  daily_cost * (days_covered + 1) > payment + max_debt :=
by sorry

end NUMINAMATH_CALUDE_internet_cost_proof_l1587_158783


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1587_158730

theorem cubic_equation_solution : ∃ x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 ∧ x = 33 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1587_158730


namespace NUMINAMATH_CALUDE_complex_fraction_squared_l1587_158719

theorem complex_fraction_squared (i : ℂ) (hi : i^2 = -1) :
  ((1 - i) / (1 + i))^2 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l1587_158719


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1587_158717

theorem quadratic_roots_difference (P : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - P = 0 ∧ β^2 - 2*β - P = 0 ∧ α - β = 12) → P = 35 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1587_158717


namespace NUMINAMATH_CALUDE_distinct_solutions_count_l1587_158770

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 6*x + 5

-- State the theorem
theorem distinct_solutions_count :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 5) ∧ s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_solutions_count_l1587_158770


namespace NUMINAMATH_CALUDE_f_seven_equals_negative_two_l1587_158786

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_equals_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_one : f 1 = 2) :
  f 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_negative_two_l1587_158786


namespace NUMINAMATH_CALUDE_part_one_part_two_l1587_158706

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 3| + m

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x, g m x ≥ 0 ↔ -5 ≤ x ∧ x ≤ -1) → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f x > g m x) → m < 5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1587_158706


namespace NUMINAMATH_CALUDE_not_integer_negative_nine_tenths_l1587_158742

theorem not_integer_negative_nine_tenths : ¬ (∃ (n : ℤ), (n : ℚ) = -9/10) := by
  sorry

end NUMINAMATH_CALUDE_not_integer_negative_nine_tenths_l1587_158742


namespace NUMINAMATH_CALUDE_cube_edge_length_l1587_158765

theorem cube_edge_length (a : ℕ) (h1 : a > 0) :
  6 * a^2 = 3 * (12 * a) → a + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1587_158765


namespace NUMINAMATH_CALUDE_prime_sum_problem_l1587_158728

theorem prime_sum_problem (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (hab : a + b = 49) (hbc : b + c = 60) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l1587_158728


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_value_l1587_158751

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^3 - a

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * a * x^2

theorem tangent_parallel_implies_a_value (a : ℝ) :
  (f a 1 = a) →                           -- The point (1, a) is on the curve
  (f_derivative a 1 = 2) →                -- The slope of the tangent at (1, a) equals the slope of 2x - y + 1 = 0
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_value_l1587_158751


namespace NUMINAMATH_CALUDE_support_area_l1587_158792

/-- The area of a support satisfying specific mass and area change conditions -/
theorem support_area : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (50 / (x - 5) = 60 / x + 1) ∧ 
  (x = 15) := by
  sorry

end NUMINAMATH_CALUDE_support_area_l1587_158792


namespace NUMINAMATH_CALUDE_pie_point_returns_to_initial_position_l1587_158740

/-- Represents a point on a circular pie --/
structure PiePoint where
  angle : Real
  radius : Real

/-- Represents the operation of cutting, flipping, and rotating the pie --/
def pieOperation (α β : Real) (p : PiePoint) : PiePoint :=
  sorry

/-- The main theorem statement --/
theorem pie_point_returns_to_initial_position
  (α β : Real)
  (h1 : β < α)
  (h2 : α < 180)
  : ∃ N : ℕ, ∀ p : PiePoint,
    (pieOperation α β)^[N] p = p :=
  sorry

end NUMINAMATH_CALUDE_pie_point_returns_to_initial_position_l1587_158740


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1587_158796

theorem unique_solution_condition (k : ℤ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 4) = -40 + k * x) ↔ (k = 8 ∨ k = -22) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1587_158796


namespace NUMINAMATH_CALUDE_certain_number_value_l1587_158794

theorem certain_number_value : ∃ x : ℝ, (35 / 100) * x = (20 / 100) * 700 ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1587_158794


namespace NUMINAMATH_CALUDE_greg_earnings_l1587_158767

/-- Represents the rates and walking details for Greg's dog walking business -/
structure DogWalkingRates where
  small_base : ℝ := 15
  small_per_minute : ℝ := 1
  medium_base : ℝ := 20
  medium_per_minute : ℝ := 1.25
  large_base : ℝ := 25
  large_per_minute : ℝ := 1.5
  small_dogs : ℕ := 3
  small_minutes : ℕ := 12
  medium_dogs : ℕ := 2
  medium_minutes : ℕ := 18
  large_dogs : ℕ := 1
  large_minutes : ℕ := 25

/-- Calculates Greg's total earnings from dog walking -/
def calculateEarnings (rates : DogWalkingRates) : ℝ :=
  (rates.small_base * rates.small_dogs + rates.small_per_minute * rates.small_dogs * rates.small_minutes) +
  (rates.medium_base * rates.medium_dogs + rates.medium_per_minute * rates.medium_dogs * rates.medium_minutes) +
  (rates.large_base * rates.large_dogs + rates.large_per_minute * rates.large_dogs * rates.large_minutes)

/-- Theorem stating that Greg's total earnings are $228.50 -/
theorem greg_earnings (rates : DogWalkingRates) : calculateEarnings rates = 228.5 := by
  sorry

end NUMINAMATH_CALUDE_greg_earnings_l1587_158767


namespace NUMINAMATH_CALUDE_carpet_length_independent_of_steps_carpet_sufficient_l1587_158766

/-- Represents a staircase with its properties --/
structure Staircase :=
  (steps : ℕ)
  (length : ℝ)
  (height : ℝ)

/-- Calculates the length of carpet required for a given staircase --/
def carpet_length (s : Staircase) : ℝ := s.length + s.height

/-- Theorem stating that carpet length depends only on staircase length and height --/
theorem carpet_length_independent_of_steps (s1 s2 : Staircase) :
  s1.length = s2.length → s1.height = s2.height →
  carpet_length s1 = carpet_length s2 := by
  sorry

/-- Specific instance for the problem --/
def staircase1 : Staircase := ⟨9, 2, 2⟩
def staircase2 : Staircase := ⟨10, 2, 2⟩

/-- Theorem stating that the carpet for staircase1 is enough for staircase2 --/
theorem carpet_sufficient : carpet_length staircase1 = carpet_length staircase2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_independent_of_steps_carpet_sufficient_l1587_158766


namespace NUMINAMATH_CALUDE_birdhouse_nails_count_l1587_158769

/-- The number of planks required to build one birdhouse -/
def planks_per_birdhouse : ℕ := 7

/-- The cost of one nail in dollars -/
def nail_cost : ℚ := 0.05

/-- The cost of one plank in dollars -/
def plank_cost : ℕ := 3

/-- The total cost to build 4 birdhouses in dollars -/
def total_cost_4_birdhouses : ℕ := 88

/-- The number of birdhouses built -/
def num_birdhouses : ℕ := 4

/-- The number of nails required to build one birdhouse -/
def nails_per_birdhouse : ℕ := 20

theorem birdhouse_nails_count :
  nails_per_birdhouse * num_birdhouses * nail_cost +
  planks_per_birdhouse * num_birdhouses * plank_cost =
  total_cost_4_birdhouses := by sorry

end NUMINAMATH_CALUDE_birdhouse_nails_count_l1587_158769


namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l1587_158733

/-- Fixed points of a function -/
def fixed_points (f : ℝ → ℝ) : Set ℝ := {x | f x = x}

/-- Stable points of a function -/
def stable_points (f : ℝ → ℝ) : Set ℝ := {x | f (f x) = x}

theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  fixed_points f ⊆ stable_points f := by sorry

theorem quadratic_no_fixed_points_implies_no_stable_points
  (a b c : ℝ) (h : a ≠ 0) (f : ℝ → ℝ) (hf : ∀ x, f x = a * x^2 + b * x + c) :
  fixed_points f = ∅ → stable_points f = ∅ := by sorry

end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l1587_158733
