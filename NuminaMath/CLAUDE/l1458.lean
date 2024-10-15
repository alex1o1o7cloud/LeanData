import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1458_145842

theorem difference_of_squares_special_case : (23 * 2 + 15)^2 - (23 * 2 - 15)^2 = 2760 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1458_145842


namespace NUMINAMATH_CALUDE_tom_bikes_11860_miles_l1458_145813

/-- The number of miles Tom bikes in a year -/
def total_miles : ℕ :=
  let miles_per_day_first_period : ℕ := 30
  let days_first_period : ℕ := 183
  let miles_per_day_second_period : ℕ := 35
  let days_in_year : ℕ := 365
  let days_second_period : ℕ := days_in_year - days_first_period
  miles_per_day_first_period * days_first_period + miles_per_day_second_period * days_second_period

/-- Theorem stating that Tom bikes 11860 miles in a year -/
theorem tom_bikes_11860_miles : total_miles = 11860 := by
  sorry

end NUMINAMATH_CALUDE_tom_bikes_11860_miles_l1458_145813


namespace NUMINAMATH_CALUDE_eight_towns_distances_l1458_145819

/-- The number of unique distances needed to connect n towns -/
def uniqueDistances (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: For 8 towns, the number of unique distances is 28 -/
theorem eight_towns_distances : uniqueDistances 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_towns_distances_l1458_145819


namespace NUMINAMATH_CALUDE_expand_expression_l1458_145804

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1458_145804


namespace NUMINAMATH_CALUDE_gcd_490_910_l1458_145850

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l1458_145850


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1458_145846

/-- 
Given a line passing through points (1, 3) and (3, 7),
prove that the sum of its slope (m) and y-intercept (b) is equal to 3.
-/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1458_145846


namespace NUMINAMATH_CALUDE_cube_root_inequality_l1458_145855

theorem cube_root_inequality (a b : ℝ) (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l1458_145855


namespace NUMINAMATH_CALUDE_expression_evaluation_l1458_145871

theorem expression_evaluation :
  let x : ℝ := 2
  (x^2 * (x - 1) - x * (x^2 + x - 1)) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1458_145871


namespace NUMINAMATH_CALUDE_median_is_eight_l1458_145882

-- Define the daily production values and the number of workers for each value
def daily_production : List ℕ := [5, 6, 7, 8, 9, 10]
def worker_count : List ℕ := [4, 5, 8, 9, 6, 4]

-- Define a function to calculate the median
def median (production : List ℕ) (workers : List ℕ) : ℚ :=
  sorry

-- Theorem statement
theorem median_is_eight :
  median daily_production worker_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_median_is_eight_l1458_145882


namespace NUMINAMATH_CALUDE_alkyne_ch_bond_polarization_l1458_145844

-- Define the hybridization states
inductive Hybridization
| sp
| sp2
| sp3

-- Define a function to represent the s-character percentage
def sCharacter (h : Hybridization) : ℚ :=
  match h with
  | .sp  => 1/2
  | .sp2 => 1/3
  | .sp3 => 1/4

-- Define a function to represent electronegativity
def electronegativity (h : Hybridization) : ℝ := sorry

-- Define a function to represent bond polarization strength
def bondPolarizationStrength (h : Hybridization) : ℝ := sorry

-- Theorem statement
theorem alkyne_ch_bond_polarization :
  (∀ h : Hybridization, h ≠ Hybridization.sp → electronegativity Hybridization.sp > electronegativity h) ∧
  (∀ h : Hybridization, bondPolarizationStrength h = electronegativity h) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp2) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp3) := by
  sorry

end NUMINAMATH_CALUDE_alkyne_ch_bond_polarization_l1458_145844


namespace NUMINAMATH_CALUDE_square_root_problem_l1458_145888

theorem square_root_problem (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x) →
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x ∧ x = 100) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l1458_145888


namespace NUMINAMATH_CALUDE_max_boxes_theorem_l1458_145809

def lifting_capacities : List Nat := [30, 45, 50, 60, 75, 100, 120]
def box_weights : List Nat := [15, 25, 35, 45, 55, 70, 80, 95, 110]

def max_boxes_lifted (capacities : List Nat) (weights : List Nat) : Nat :=
  sorry

theorem max_boxes_theorem :
  max_boxes_lifted lifting_capacities box_weights = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_theorem_l1458_145809


namespace NUMINAMATH_CALUDE_martian_amoeba_nim_exists_l1458_145873

-- Define the set of Martian amoebas
inductive MartianAmoeba
  | A
  | B
  | C

-- Define the function type
def AmoebaNim := MartianAmoeba → Nat

-- Define the bitwise XOR operation
def bxor (a b : Nat) : Nat :=
  Nat.xor a b

-- State the theorem
theorem martian_amoeba_nim_exists : ∃ (f : AmoebaNim),
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.B) = f MartianAmoeba.C) ∧
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.C) = f MartianAmoeba.B) ∧
  (bxor (f MartianAmoeba.B) (f MartianAmoeba.C) = f MartianAmoeba.A) :=
by
  sorry

end NUMINAMATH_CALUDE_martian_amoeba_nim_exists_l1458_145873


namespace NUMINAMATH_CALUDE_melanie_dimes_l1458_145815

theorem melanie_dimes (initial_dimes mother_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : mother_dimes = 4)
  (h3 : final_dimes = 19) :
  final_dimes - (initial_dimes + mother_dimes) = 8 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1458_145815


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1458_145832

/-- The largest prime number with 2009 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2009 digits -/
axiom q_digits : 10^2008 ≤ q ∧ q < 10^2009

/-- q is the largest prime with 2009 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2008 ≤ p ∧ p < 10^2009 → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1458_145832


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1458_145880

theorem largest_solution_of_equation : 
  ∃ (y : ℝ), y = 5 ∧ 
  3 * y^2 + 30 * y - 90 = y * (y + 18) ∧
  ∀ (z : ℝ), 3 * z^2 + 30 * z - 90 = z * (z + 18) → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1458_145880


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1458_145879

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1458_145879


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1458_145866

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1458_145866


namespace NUMINAMATH_CALUDE_least_positive_angle_l1458_145875

open Real

theorem least_positive_angle (θ : ℝ) : 
  (θ > 0 ∧ ∀ φ, φ > 0 ∧ (cos (10 * π / 180) = sin (40 * π / 180) + cos φ) → θ ≤ φ) →
  cos (10 * π / 180) = sin (40 * π / 180) + cos θ →
  θ = 70 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_l1458_145875


namespace NUMINAMATH_CALUDE_largest_a_is_four_l1458_145854

/-- The largest coefficient of x^4 in a polynomial that satisfies the given conditions -/
noncomputable def largest_a : ℝ := 4

/-- A polynomial of degree 4 with real coefficients -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The condition that the polynomial is between 0 and 1 on [-1, 1] -/
def satisfies_condition (a b c d e : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ polynomial a b c d e x ∧ polynomial a b c d e x ≤ 1

/-- The theorem stating that 4 is the largest possible value for a -/
theorem largest_a_is_four :
  (∃ b c d e : ℝ, satisfies_condition largest_a b c d e) ∧
  (∀ a : ℝ, a > largest_a → ¬∃ b c d e : ℝ, satisfies_condition a b c d e) :=
sorry

end NUMINAMATH_CALUDE_largest_a_is_four_l1458_145854


namespace NUMINAMATH_CALUDE_rotation_solutions_l1458_145893

-- Define the basic geometric elements
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop
def Plane : Type := Point → Prop

-- Define the given elements
variable (v : Line) -- Second elevation line
variable (P : Point) -- Original point
variable (P₂'' : Point) -- Inverted point parallel to second elevation plane

-- Define the geometric constructions
def rotationCircle (v : Line) (P : Point) : Set Point := sorry
def firstBisectorPlane : Plane := sorry
def planeS (v : Line) (P : Point) : Plane := sorry
def lineH₁ (v : Line) (P : Point) : Line := sorry

-- Define the number of intersections
def numIntersections (circle : Set Point) (line : Line) : ℕ := sorry

-- Define the number of solutions
def numSolutions (v : Line) (P : Point) : ℕ := sorry

-- State the theorem
theorem rotation_solutions (v : Line) (P : Point) :
  numSolutions v P = numIntersections (rotationCircle v P) (lineH₁ v P) := by sorry

end NUMINAMATH_CALUDE_rotation_solutions_l1458_145893


namespace NUMINAMATH_CALUDE_luncheon_invitees_l1458_145863

theorem luncheon_invitees (no_shows : ℕ) (people_per_table : ℕ) (tables_needed : ℕ) :
  no_shows = 35 →
  people_per_table = 2 →
  tables_needed = 5 →
  no_shows + (people_per_table * tables_needed) = 45 :=
by sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l1458_145863


namespace NUMINAMATH_CALUDE_pizza_delivery_time_l1458_145867

theorem pizza_delivery_time (total_pizzas : ℕ) (double_order_stops : ℕ) (time_per_stop : ℕ) : 
  total_pizzas = 12 →
  double_order_stops = 2 →
  time_per_stop = 4 →
  (total_pizzas - 2 * double_order_stops + double_order_stops) * time_per_stop = 40 :=
by sorry

end NUMINAMATH_CALUDE_pizza_delivery_time_l1458_145867


namespace NUMINAMATH_CALUDE_total_keys_for_tim_l1458_145891

/-- Calculates the total number of keys needed for Tim's rental properties -/
def total_keys (apartment_complex_1 apartment_complex_2 apartment_complex_3 : ℕ)
  (individual_houses : ℕ)
  (keys_per_apartment keys_per_main_entrance keys_per_house : ℕ) : ℕ :=
  (apartment_complex_1 + apartment_complex_2 + apartment_complex_3) * keys_per_apartment +
  3 * keys_per_main_entrance +
  individual_houses * keys_per_house

/-- Theorem stating the total number of keys needed for Tim's rental properties -/
theorem total_keys_for_tim : 
  total_keys 16 20 24 4 4 10 6 = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_keys_for_tim_l1458_145891


namespace NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l1458_145834

/-- Represents a metro line with a specific round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Calculates the least common multiple (LCM) of three natural numbers -/
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

/-- Theorem: All trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (system : MetroSystem) : 
  (2016 % lcm3 system.redLine.roundTripTime system.blueLine.roundTripTime system.greenLine.roundTripTime = 0) → 
  (∀ (line : MetroLine), 2016 % line.roundTripTime = 0) :=
by
  sorry

/-- The actual metro system of city N -/
def cityN : MetroSystem :=
  { redLine := { roundTripTime := 14 }
  , blueLine := { roundTripTime := 16 }
  , greenLine := { roundTripTime := 18 }
  }

/-- Proof that the trains in city N return to their initial positions after 2016 minutes -/
theorem cityN_trains_return_to_initial_positions : 
  (2016 % lcm3 cityN.redLine.roundTripTime cityN.blueLine.roundTripTime cityN.greenLine.roundTripTime = 0) ∧
  (∀ (line : MetroLine), line ∈ [cityN.redLine, cityN.blueLine, cityN.greenLine] → 2016 % line.roundTripTime = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l1458_145834


namespace NUMINAMATH_CALUDE_ben_and_brothers_pizza_order_l1458_145870

/-- The number of small pizzas ordered for Ben and his brothers -/
def small_pizzas_ordered (num_people : ℕ) (slices_per_person : ℕ) (large_pizza_slices : ℕ) (small_pizza_slices : ℕ) (large_pizzas_ordered : ℕ) : ℕ :=
  let total_slices_needed := num_people * slices_per_person
  let slices_from_large := large_pizzas_ordered * large_pizza_slices
  let remaining_slices := total_slices_needed - slices_from_large
  (remaining_slices + small_pizza_slices - 1) / small_pizza_slices

theorem ben_and_brothers_pizza_order :
  small_pizzas_ordered 3 12 14 8 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ben_and_brothers_pizza_order_l1458_145870


namespace NUMINAMATH_CALUDE_rationalize_and_product_l1458_145877

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2:ℝ) + Real.sqrt 5) / ((3:ℝ) - Real.sqrt 5) = A + B * Real.sqrt C) ∧
  (A * B * C = 275) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l1458_145877


namespace NUMINAMATH_CALUDE_room_expansion_theorem_l1458_145848

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a room --/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: If increasing the length and breadth of a rectangular room by y feet
    results in a perimeter increase of 16 feet, then y must equal 4 feet. --/
theorem room_expansion_theorem (r : Room) (y : ℝ) 
    (h : perimeter { length := r.length + y, breadth := r.breadth + y } - perimeter r = 16) : 
  y = 4 := by
  sorry

end NUMINAMATH_CALUDE_room_expansion_theorem_l1458_145848


namespace NUMINAMATH_CALUDE_hearts_to_diamonds_ratio_l1458_145885

/-- Represents the number of cards of each suit in a player's hand -/
structure CardCounts where
  spades : ℕ
  diamonds : ℕ
  hearts : ℕ
  clubs : ℕ

/-- The conditions of the card counting problem -/
def validCardCounts (c : CardCounts) : Prop :=
  c.spades + c.diamonds + c.hearts + c.clubs = 13 ∧
  c.spades + c.clubs = 7 ∧
  c.diamonds + c.hearts = 6 ∧
  c.diamonds = 2 * c.spades ∧
  c.clubs = 6

theorem hearts_to_diamonds_ratio (c : CardCounts) 
  (h : validCardCounts c) : c.hearts = 2 * c.diamonds := by
  sorry

end NUMINAMATH_CALUDE_hearts_to_diamonds_ratio_l1458_145885


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1458_145812

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1458_145812


namespace NUMINAMATH_CALUDE_michaels_subtraction_l1458_145803

theorem michaels_subtraction (a b : ℕ) (h1 : a = 40) (h2 : b = 39) :
  a^2 - b^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_michaels_subtraction_l1458_145803


namespace NUMINAMATH_CALUDE_four_sets_gemstones_l1458_145837

/-- Calculates the number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem stating that 4 sets of earrings require 24 gemstones -/
theorem four_sets_gemstones : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_sets_gemstones_l1458_145837


namespace NUMINAMATH_CALUDE_strawberry_sales_chloe_strawberry_sales_l1458_145869

/-- Calculates the number of dozens of strawberries sold given the cost per dozen,
    selling price per half dozen, and total profit. -/
theorem strawberry_sales 
  (cost_per_dozen : ℚ) 
  (selling_price_per_half_dozen : ℚ) 
  (total_profit : ℚ) : ℚ :=
  let profit_per_half_dozen := selling_price_per_half_dozen - cost_per_dozen / 2
  let half_dozens_sold := total_profit / profit_per_half_dozen
  let dozens_sold := half_dozens_sold / 2
  dozens_sold

/-- Proves that given the specified conditions, Chloe sold 50 dozens of strawberries. -/
theorem chloe_strawberry_sales : 
  strawberry_sales 50 30 500 = 50 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_sales_chloe_strawberry_sales_l1458_145869


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l1458_145845

/-- The number of points on the circle -/
def n : ℕ := 20

/-- The number of ways to choose 2 points from n points -/
def choose_diameter (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to choose 2 points from the remaining n-2 points -/
def choose_remaining (n : ℕ) : ℕ := (n - 2) * (n - 3) / 2

/-- The total number of cyclic quadrilaterals with one right angle -/
def total_quadrilaterals (n : ℕ) : ℕ := choose_diameter n * choose_remaining n

theorem cyclic_quadrilaterals_count :
  total_quadrilaterals n = 29070 :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l1458_145845


namespace NUMINAMATH_CALUDE_problem_solution_l1458_145856

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

def tangent_perpendicular (a : ℝ) : Prop :=
  let f' := deriv (f a) 1
  f' * (-1) = 1

def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → (f 0 x) ≤ m * (x - 1)

theorem problem_solution :
  (∃ a : ℝ, tangent_perpendicular a ∧ a = 0) ∧
  (∃ m : ℝ, inequality_holds m ∧ ∀ m' : ℝ, m' ≥ m → inequality_holds m') :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1458_145856


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l1458_145830

theorem grocery_shop_sales (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) 
  (h1 : sale1 = 6235)
  (h2 : sale2 = 6927)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 5191)
  (h_avg : average_sale = 6500) :
  ∃ sale3 : ℕ, 
    sale3 = 6855 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale :=
sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l1458_145830


namespace NUMINAMATH_CALUDE_chrysler_leeward_floor_difference_l1458_145802

theorem chrysler_leeward_floor_difference :
  ∀ (chrysler_floors leeward_floors : ℕ),
    chrysler_floors > leeward_floors →
    chrysler_floors + leeward_floors = 35 →
    chrysler_floors = 23 →
    chrysler_floors - leeward_floors = 11 := by
  sorry

end NUMINAMATH_CALUDE_chrysler_leeward_floor_difference_l1458_145802


namespace NUMINAMATH_CALUDE_add_7777_seconds_to_11pm_l1458_145820

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a 12-hour time (with PM indicator) to 24-hour format -/
def to24Hour (hours : Nat) (isPM : Bool) : Nat :=
  sorry

theorem add_7777_seconds_to_11pm :
  let startTime := Time.mk (to24Hour 11 true) 0 0
  let endTime := addSeconds startTime 7777
  endTime = Time.mk 1 9 37 :=
sorry

end NUMINAMATH_CALUDE_add_7777_seconds_to_11pm_l1458_145820


namespace NUMINAMATH_CALUDE_count_integers_with_5_or_6_l1458_145862

/-- The number of integers among the first 729 positive integers in base 9 
    that contain either 5 or 6 (or both) as a digit -/
def count_with_5_or_6 : ℕ := 386

/-- The base of the number system we're working with -/
def base : ℕ := 9

/-- The number of smallest positive integers we're considering -/
def total_count : ℕ := 729

theorem count_integers_with_5_or_6 :
  count_with_5_or_6 = total_count - (base - 2)^3 ∧
  total_count = base^3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_5_or_6_l1458_145862


namespace NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l1458_145853

theorem flower_shop_carnation_percentage :
  let carnations : ℝ := 1  -- Arbitrary non-zero value for carnations
  let violets : ℝ := (1/3) * carnations
  let tulips : ℝ := (1/4) * violets
  let roses : ℝ := tulips
  let total : ℝ := carnations + violets + tulips + roses
  (carnations / total) * 100 = 200/3 := by
sorry

end NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l1458_145853


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l1458_145847

/-- Given the following conditions:
    - Nancy has 40 metal beads and 20 more pearl beads than metal beads
    - Rose has 20 crystal beads and twice as many stone beads as crystal beads
    - They can make 20 bracelets
    Prove that the number of beads in each bracelet is 8. -/
theorem beads_per_bracelet :
  let nancy_metal : ℕ := 40
  let nancy_pearl : ℕ := nancy_metal + 20
  let rose_crystal : ℕ := 20
  let rose_stone : ℕ := 2 * rose_crystal
  let total_bracelets : ℕ := 20
  let total_beads : ℕ := nancy_metal + nancy_pearl + rose_crystal + rose_stone
  (total_beads / total_bracelets : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l1458_145847


namespace NUMINAMATH_CALUDE_k_value_proof_l1458_145800

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l1458_145800


namespace NUMINAMATH_CALUDE_triangle_sum_equality_l1458_145816

theorem triangle_sum_equality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = a^2)
  (eq2 : y^2 + y*z + z^2 = b^2)
  (eq3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p * (p - a) * (p - b) * (p - c)) / 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_equality_l1458_145816


namespace NUMINAMATH_CALUDE_unique_prime_with_six_divisors_l1458_145808

/-- A function that counts the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem unique_prime_with_six_divisors :
  ∀ p : ℕ, is_prime p → (count_divisors (p^2 + 11) = 6) → p = 3 := by sorry

end NUMINAMATH_CALUDE_unique_prime_with_six_divisors_l1458_145808


namespace NUMINAMATH_CALUDE_car_speed_increase_car_speed_increase_proof_l1458_145841

/-- Calculates the increased speed of a car given initial conditions and final results -/
theorem car_speed_increase (v : ℝ) (initial_time stop_time delay additional_distance total_distance : ℝ) : ℝ :=
  let original_time := total_distance / v
  let actual_time := original_time + stop_time + delay
  let remaining_time := actual_time - initial_time
  let new_total_distance := total_distance + additional_distance
  let distance_after_stop := new_total_distance - (v * initial_time)
  distance_after_stop / remaining_time

/-- Proves that the increased speed of the car is approximately 34.91 km/hr given the problem conditions -/
theorem car_speed_increase_proof :
  let v := 32
  let initial_time := 3
  let stop_time := 0.25
  let delay := 0.5
  let additional_distance := 28
  let total_distance := 116
  abs (car_speed_increase v initial_time stop_time delay additional_distance total_distance - 34.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_increase_car_speed_increase_proof_l1458_145841


namespace NUMINAMATH_CALUDE_triangle_properties_l1458_145878

theorem triangle_properties (a b c : ℝ) (A B C : Real) (S : ℝ) (D : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  a * Real.sin B = b * Real.sin (A + π / 3) →
  S = 2 * Real.sqrt 3 →
  S = (1 / 2) * b * c * Real.sin A →
  D.1 = (2 / 3) * b →
  D.2 = 0 →
  (∃ (AD : ℝ), AD ≥ (4 * Real.sqrt 3) / 3 ∧
    AD^2 = (1 / 9) * c^2 + (4 / 9) * b^2 + (16 / 9)) →
  A = π / 3 ∧ (∃ (AD_min : ℝ), AD_min = (4 * Real.sqrt 3) / 3 ∧
    ∀ (AD : ℝ), AD ≥ AD_min) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1458_145878


namespace NUMINAMATH_CALUDE_valid_m_values_l1458_145831

-- Define the set A
def A (m : ℝ) : Set ℝ := {1, m + 2, m^2 + 4}

-- State the theorem
theorem valid_m_values :
  ∀ m : ℝ, 5 ∈ A m → (m = 1 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_valid_m_values_l1458_145831


namespace NUMINAMATH_CALUDE_square_sum_existence_l1458_145807

theorem square_sum_existence (k : ℤ) 
  (h1 : 2 * k + 1 > 17) 
  (h2 : ∃ m : ℤ, 6 * k + 1 = m^2) : 
  ∃ b c : ℤ, 
    b > 0 ∧ 
    c > 0 ∧ 
    b ≠ c ∧ 
    (∃ w : ℤ, (2 * k + 1 + b) = w^2) ∧ 
    (∃ x : ℤ, (2 * k + 1 + c) = x^2) ∧ 
    (∃ y : ℤ, (b + c) = y^2) ∧ 
    (∃ z : ℤ, (2 * k + 1 + b + c) = z^2) :=
sorry

end NUMINAMATH_CALUDE_square_sum_existence_l1458_145807


namespace NUMINAMATH_CALUDE_movie_theater_seating_l1458_145840

def seat_arrangements (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n - 4).choose 3 * 2

theorem movie_theater_seating : seat_arrangements 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_seating_l1458_145840


namespace NUMINAMATH_CALUDE_investment_split_l1458_145895

theorem investment_split (alice_share bob_share total : ℕ) : 
  alice_share = 5 →
  bob_share = 3 * (total / bob_share) →
  bob_share = 3 * alice_share + 3 →
  total = bob_share * (total / bob_share) + alice_share →
  total = 113 := by
sorry

end NUMINAMATH_CALUDE_investment_split_l1458_145895


namespace NUMINAMATH_CALUDE_book_arrangement_l1458_145806

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = Nat.choose 8 m :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l1458_145806


namespace NUMINAMATH_CALUDE_solve_equation_l1458_145818

theorem solve_equation (x n : ℚ) (h1 : n * (x - 3) = 15) (h2 : x = 12) : n = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1458_145818


namespace NUMINAMATH_CALUDE_prime_divisor_congruence_l1458_145898

theorem prime_divisor_congruence (p q : ℕ) : 
  Prime p → 
  Prime q → 
  q ∣ ((p^p - 1) / (p - 1)) → 
  q ≡ 1 [ZMOD p] := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_congruence_l1458_145898


namespace NUMINAMATH_CALUDE_line_vector_at_negative_two_l1458_145814

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem line_vector_at_negative_two :
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7))) →
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7)) ∧
    (line_param (-2) = (-1, 17))) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_at_negative_two_l1458_145814


namespace NUMINAMATH_CALUDE_prob_both_genders_selected_l1458_145892

def total_students : ℕ := 8
def male_students : ℕ := 5
def female_students : ℕ := 3
def students_to_select : ℕ := 5

theorem prob_both_genders_selected :
  (Nat.choose total_students students_to_select - Nat.choose male_students students_to_select) /
  Nat.choose total_students students_to_select = 55 / 56 :=
by sorry

end NUMINAMATH_CALUDE_prob_both_genders_selected_l1458_145892


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l1458_145859

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    (∃ q r : ℕ, Nat.Prime q ∧ Nat.Prime r ∧ p = q + r) ∧
    (∃ s t : ℕ, Nat.Prime s ∧ Nat.Prime t ∧ p = s - t) ∧
    p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l1458_145859


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1458_145860

/-- A rhombus with area 192 and diagonal ratio 4:3 has longest diagonal of length 16√2 -/
theorem rhombus_longest_diagonal (d₁ d₂ : ℝ) : 
  d₁ * d₂ / 2 = 192 →  -- Area formula
  d₁ / d₂ = 4 / 3 →    -- Diagonal ratio
  max d₁ d₂ = 16 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1458_145860


namespace NUMINAMATH_CALUDE_complex_modulus_l1458_145896

theorem complex_modulus (z : ℂ) (h : (z - I) / (2 - I) = I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1458_145896


namespace NUMINAMATH_CALUDE_sum_of_ages_l1458_145876

/-- Given the present ages of Henry and Jill, prove that their sum is 48 years. -/
theorem sum_of_ages (henry_age jill_age : ℕ) 
  (henry_present : henry_age = 29)
  (jill_present : jill_age = 19)
  (past_relation : henry_age - 9 = 2 * (jill_age - 9)) :
  henry_age + jill_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1458_145876


namespace NUMINAMATH_CALUDE_hotel_expenditure_l1458_145861

theorem hotel_expenditure (num_persons : ℕ) (regular_spend : ℕ) (extra_spend : ℕ) : 
  num_persons = 9 →
  regular_spend = 12 →
  extra_spend = 8 →
  (num_persons - 1) * regular_spend + 
  (((num_persons - 1) * regular_spend + (regular_spend + extra_spend)) / num_persons + extra_spend) = 117 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l1458_145861


namespace NUMINAMATH_CALUDE_smallest_odd_factors_above_50_l1458_145822

/-- A number has an odd number of positive factors if and only if it is a perfect square. -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The smallest whole number greater than 50 that has an odd number of positive factors is 64. -/
theorem smallest_odd_factors_above_50 : 
  (∀ m : ℕ, m > 50 ∧ m < 64 → ¬(has_odd_factors m)) ∧ 
  (64 > 50 ∧ has_odd_factors 64) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_factors_above_50_l1458_145822


namespace NUMINAMATH_CALUDE_jill_peach_count_jill_peach_count_proof_l1458_145872

/-- Given the peach distribution among Jake, Steven, Jill, and Sam, prove that Jill has 6 peaches. -/
theorem jill_peach_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jake steven jill sam =>
    (jake = steven - 18) →
    (steven = jill + 13) →
    (steven = 19) →
    (sam = 2 * jill) →
    (jill = 6)

/-- Proof of the theorem -/
theorem jill_peach_count_proof : ∃ jake steven jill sam, jill_peach_count jake steven jill sam :=
  sorry

end NUMINAMATH_CALUDE_jill_peach_count_jill_peach_count_proof_l1458_145872


namespace NUMINAMATH_CALUDE_power_equation_exponent_l1458_145883

theorem power_equation_exponent (x : ℝ) (n : ℝ) (h : x ≠ 0) : 
  x^3 / x = x^n → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_equation_exponent_l1458_145883


namespace NUMINAMATH_CALUDE_johnny_savings_l1458_145836

theorem johnny_savings (september : ℕ) (october : ℕ) (spent : ℕ) (left : ℕ) :
  september = 30 →
  october = 49 →
  spent = 58 →
  left = 67 →
  ∃ november : ℕ, november = 46 ∧ september + october + november - spent = left :=
by sorry

end NUMINAMATH_CALUDE_johnny_savings_l1458_145836


namespace NUMINAMATH_CALUDE_pig_count_after_joining_l1458_145826

theorem pig_count_after_joining (initial_pigs joining_pigs : ℕ) :
  initial_pigs = 64 →
  joining_pigs = 22 →
  initial_pigs + joining_pigs = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_pig_count_after_joining_l1458_145826


namespace NUMINAMATH_CALUDE_math_game_result_l1458_145887

theorem math_game_result (a : ℚ) : 
  (1/2 : ℚ) * (-(- a) - 2) = -1/2 * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_math_game_result_l1458_145887


namespace NUMINAMATH_CALUDE_enrollment_calculation_l1458_145897

def final_enrollment (initial : ℕ) (new_interested : ℕ) (new_dropout_rate : ℚ)
  (additional_dropouts : ℕ) (increase_factor : ℕ) (schedule_dropouts : ℕ)
  (final_rally : ℕ) (later_dropout_rate : ℚ) (graduation_rate : ℚ) : ℕ :=
  sorry

theorem enrollment_calculation :
  final_enrollment 8 8 (1/4) 2 5 2 6 (1/2) (1/2) = 19 :=
sorry

end NUMINAMATH_CALUDE_enrollment_calculation_l1458_145897


namespace NUMINAMATH_CALUDE_percentage_of_temporary_workers_l1458_145828

theorem percentage_of_temporary_workers
  (total_workers : ℕ)
  (technician_ratio : ℚ)
  (non_technician_ratio : ℚ)
  (permanent_technician_ratio : ℚ)
  (permanent_non_technician_ratio : ℚ)
  (h1 : technician_ratio = 9/10)
  (h2 : non_technician_ratio = 1/10)
  (h3 : permanent_technician_ratio = 9/10)
  (h4 : permanent_non_technician_ratio = 1/10)
  (h5 : technician_ratio + non_technician_ratio = 1) :
  let permanent_workers := (technician_ratio * permanent_technician_ratio +
                            non_technician_ratio * permanent_non_technician_ratio) * total_workers
  let temporary_workers := total_workers - permanent_workers
  (temporary_workers : ℚ) / (total_workers : ℚ) = 18/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_temporary_workers_l1458_145828


namespace NUMINAMATH_CALUDE_orange_count_l1458_145829

theorem orange_count (initial : ℕ) : 
  initial - 9 + 38 = 60 → initial = 31 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l1458_145829


namespace NUMINAMATH_CALUDE_exists_intersecting_line_no_circle_through_origin_l1458_145811

-- Define the set of circles C_k
def C_k (k : ℕ+) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (k - 1))^2 + (p.2 - 3*k)^2 = 2*k^4}

-- Statement 1: There exists a fixed line that intersects all circles
theorem exists_intersecting_line :
  ∃ (m b : ℝ), ∀ (k : ℕ+), ∃ (x y : ℝ), (y = m*x + b) ∧ (x, y) ∈ C_k k :=
sorry

-- Statement 2: No circle passes through the origin
theorem no_circle_through_origin :
  ∀ (k : ℕ+), (0, 0) ∉ C_k k :=
sorry

end NUMINAMATH_CALUDE_exists_intersecting_line_no_circle_through_origin_l1458_145811


namespace NUMINAMATH_CALUDE_billiard_ball_weight_l1458_145851

/-- Given an empty box weighing 0.5 kg and a box containing 6 identical billiard balls
    weighing 1.82 kg, prove that each billiard ball weighs 0.22 kg. -/
theorem billiard_ball_weight (empty_box_weight : ℝ) (full_box_weight : ℝ) :
  empty_box_weight = 0.5 →
  full_box_weight = 1.82 →
  (full_box_weight - empty_box_weight) / 6 = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_billiard_ball_weight_l1458_145851


namespace NUMINAMATH_CALUDE_fraction_simplification_l1458_145833

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 72) = (5 * Real.sqrt 2) / 34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1458_145833


namespace NUMINAMATH_CALUDE_calculate_hourly_pay_l1458_145821

/-- Calculates the hourly pay per employee given the company's workforce and payment information. -/
theorem calculate_hourly_pay
  (initial_employees : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (additional_employees : ℕ)
  (total_monthly_payment : ℕ)
  (h1 : initial_employees = 500)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : weeks_per_month = 4)
  (h5 : additional_employees = 200)
  (h6 : total_monthly_payment = 1680000) :
  let total_employees := initial_employees + additional_employees
  let hours_per_employee := hours_per_day * days_per_week * weeks_per_month
  let total_hours := total_employees * hours_per_employee
  (total_monthly_payment / total_hours : ℚ) = 12 :=
sorry

end NUMINAMATH_CALUDE_calculate_hourly_pay_l1458_145821


namespace NUMINAMATH_CALUDE_hexagon_triangle_angle_sum_l1458_145835

theorem hexagon_triangle_angle_sum : ∀ (P Q R s t : ℝ),
  P = 40 ∧ Q = 88 ∧ R = 30 →
  (720 : ℝ) = P + Q + R + (120 - t) + (130 - s) + s + t →
  s + t = 312 := by
sorry

end NUMINAMATH_CALUDE_hexagon_triangle_angle_sum_l1458_145835


namespace NUMINAMATH_CALUDE_sum_of_squares_values_l1458_145865

theorem sum_of_squares_values (x y z : ℝ) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (eq1 : x^2 = 2 + y)
  (eq2 : y^2 = 2 + z)
  (eq3 : z^2 = 2 + x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_values_l1458_145865


namespace NUMINAMATH_CALUDE_solution_check_l1458_145890

theorem solution_check (x : ℝ) : x = 2 ↔ -1/3 * x + 2/3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_check_l1458_145890


namespace NUMINAMATH_CALUDE_x_value_is_six_l1458_145824

def star_op (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_is_six (x : ℝ) : star_op 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_six_l1458_145824


namespace NUMINAMATH_CALUDE_concert_cost_l1458_145838

theorem concert_cost (ticket_price : ℚ) (processing_fee_rate : ℚ) 
  (parking_fee : ℚ) (entrance_fee : ℚ) (num_people : ℕ) :
  ticket_price = 50 ∧ 
  processing_fee_rate = 0.15 ∧ 
  parking_fee = 10 ∧ 
  entrance_fee = 5 ∧ 
  num_people = 2 → 
  (ticket_price + ticket_price * processing_fee_rate) * num_people + 
  parking_fee + entrance_fee * num_people = 135 :=
by sorry

end NUMINAMATH_CALUDE_concert_cost_l1458_145838


namespace NUMINAMATH_CALUDE_prob_implies_n_l1458_145857

/-- The probability of selecting a second number greater than a first number -/
def prob : ℚ := 4995 / 10000

/-- The highest number in the range -/
def n : ℕ := 1000

/-- Theorem stating that the given probability results in the correct highest number -/
theorem prob_implies_n : 
  (n : ℚ) - 1 = 2 * n * prob := by sorry

end NUMINAMATH_CALUDE_prob_implies_n_l1458_145857


namespace NUMINAMATH_CALUDE_vacation_cost_l1458_145858

/-- If a total cost C divided among 3 people is $40 more per person than if divided among 4 people, then C equals $480. -/
theorem vacation_cost (C : ℚ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l1458_145858


namespace NUMINAMATH_CALUDE_f_sum_equals_sqrt2_minus_1_l1458_145852

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, 0 ≤ x ∧ x < 1 → f x = 2 * x - 1)

theorem f_sum_equals_sqrt2_minus_1 (f : ℝ → ℝ) (hf : f_properties f) :
  f (1/2) + f 1 + f (3/2) + f (5/2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_sqrt2_minus_1_l1458_145852


namespace NUMINAMATH_CALUDE_art_kits_count_l1458_145899

theorem art_kits_count (total_students : ℕ) (students_per_kit : ℕ) 
  (artworks_group1 : ℕ) (artworks_group2 : ℕ) (total_artworks : ℕ) : ℕ :=
  let num_kits := total_students / students_per_kit
  let half_students := total_students / 2
  let artworks_from_group1 := half_students * artworks_group1
  let artworks_from_group2 := half_students * artworks_group2
  by
    have h1 : total_students = 10 := by sorry
    have h2 : students_per_kit = 2 := by sorry
    have h3 : artworks_group1 = 3 := by sorry
    have h4 : artworks_group2 = 4 := by sorry
    have h5 : total_artworks = 35 := by sorry
    have h6 : artworks_from_group1 + artworks_from_group2 = total_artworks := by sorry
    exact num_kits

end NUMINAMATH_CALUDE_art_kits_count_l1458_145899


namespace NUMINAMATH_CALUDE_cos_graph_shift_l1458_145889

theorem cos_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x - π / 3) = 3 * Real.cos (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_cos_graph_shift_l1458_145889


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l1458_145874

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l1458_145874


namespace NUMINAMATH_CALUDE_inverse_proposition_l1458_145886

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := ¬(x^2 > 0) → ¬(x < 0)

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_proposition :
  (∀ x : ℝ, original_prop x) ↔ (∀ x : ℝ, inverse_prop x) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l1458_145886


namespace NUMINAMATH_CALUDE_trailingZeros_30_factorial_l1458_145849

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ := 
  let factN := factorial n
  (Nat.digits 10 factN).reverse.takeWhile (·= 0) |>.length

theorem trailingZeros_30_factorial : trailingZeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_trailingZeros_30_factorial_l1458_145849


namespace NUMINAMATH_CALUDE_problem_statement_l1458_145817

/-- Given real numbers x and y satisfying x + y/4 = 1, prove:
    1. If |7-y| < 2x+3, then -1 < x < 0
    2. If x > 0 and y > 0, then sqrt(xy) ≥ xy -/
theorem problem_statement (x y : ℝ) (h1 : x + y / 4 = 1) :
  (∀ h2 : |7 - y| < 2*x + 3, -1 < x ∧ x < 0) ∧
  (∀ h3 : x > 0, ∀ h4 : y > 0, Real.sqrt (x * y) ≥ x * y) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1458_145817


namespace NUMINAMATH_CALUDE_linear_function_property_l1458_145839

/-- A linear function f(x) = ax + b satisfying f(1) = 2 and f'(1) = 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem linear_function_property : f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1458_145839


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1458_145894

theorem quadratic_roots_expression (r s : ℝ) : 
  (2 * r^2 - 3 * r = 11) → 
  (2 * s^2 - 3 * s = 11) → 
  r ≠ s →
  (4 * r^3 - 4 * s^3) / (r - s) = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1458_145894


namespace NUMINAMATH_CALUDE_distance_by_car_l1458_145823

/-- Proves that the distance traveled by car is 6 kilometers -/
theorem distance_by_car (total_distance : ℝ) (h1 : total_distance = 24) :
  total_distance - (1/2 * total_distance + 1/4 * total_distance) = 6 := by
  sorry

#check distance_by_car

end NUMINAMATH_CALUDE_distance_by_car_l1458_145823


namespace NUMINAMATH_CALUDE_solution_value_l1458_145843

theorem solution_value (x a : ℝ) : x = 4 ∧ 5 * (x - 1) - 3 * a = -3 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1458_145843


namespace NUMINAMATH_CALUDE_initial_tourists_l1458_145868

theorem initial_tourists (T : ℕ) : 
  (T : ℚ) - 2 - (3/7) * ((T : ℚ) - 2) = 16 → T = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_tourists_l1458_145868


namespace NUMINAMATH_CALUDE_three_solutions_sum_and_m_value_m_range_for_positive_f_l1458_145827

noncomputable section

def f (m : ℝ) (x : ℝ) := 4 - m * Real.sin x - 3 * (Real.cos x)^2

theorem three_solutions_sum_and_m_value 
  (m : ℝ) 
  (h₁ : ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 
                       0 < x₂ ∧ x₂ < π ∧ 
                       0 < x₃ ∧ x₃ < π ∧ 
                       x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
                       f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m x₃ = 0) : 
  m = 4 ∧ ∃ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 3 * π / 2 :=
sorry

theorem m_range_for_positive_f 
  (m : ℝ) 
  (h : ∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π → f m x > 0) : 
  -7/2 < m ∧ m < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_sum_and_m_value_m_range_for_positive_f_l1458_145827


namespace NUMINAMATH_CALUDE_specific_rental_cost_l1458_145805

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that for the given rental conditions, the total cost is $162.5 -/
theorem specific_rental_cost :
  carRentalCost 25 0.25 3 350 = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_rental_cost_l1458_145805


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1458_145881

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  a > b →  -- a is the larger angle
  a = 50 :=  -- The larger angle measures 50°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1458_145881


namespace NUMINAMATH_CALUDE_monotonic_increasing_intervals_inequality_solution_l1458_145825

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the properties of f
def f_properties (a b c d : ℝ) : Prop :=
  -- f is symmetrical about the origin
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  -- f takes minimum value of -2 when x = 1
  (f a b c d 1 = -2) ∧
  (∀ x, f a b c d x ≥ -2)

-- Theorem for monotonically increasing intervals
theorem monotonic_increasing_intervals (a b c d : ℝ) (h : f_properties a b c d) :
  (∀ x y, x < y ∧ x < -1 → f a b c d x < f a b c d y) ∧
  (∀ x y, x < y ∧ y > 1 → f a b c d x < f a b c d y) := by sorry

-- Theorem for inequality solution
theorem inequality_solution (a b c d m : ℝ) (h : f_properties a b c d) :
  (m = 0 → ∀ x, x > 0 → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m > 0 → ∀ x, (x > 4*m ∨ (0 < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m < 0 → ∀ x, (x > 0 ∨ (4*m < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_intervals_inequality_solution_l1458_145825


namespace NUMINAMATH_CALUDE_range_of_f_l1458_145884

def f (x : ℝ) : ℝ := 2 * x - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_f_l1458_145884


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l1458_145864

theorem fourth_degree_polynomial_roots : 
  let p (x : ℂ) := x^4 - 16*x^2 + 51
  ∀ r : ℂ, r^2 = 8 + Real.sqrt 13 → p r = 0 :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l1458_145864


namespace NUMINAMATH_CALUDE_monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l1458_145801

noncomputable section

-- Define the function f(x) = x^2 * e^(ax)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.exp (a * x)

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + a * x^2) * Real.exp (a * x)

-- Theorem for monotonicity when a = 0
theorem monotonicity_a_eq_zero :
  ∀ x : ℝ, x < 0 → (∀ y : ℝ, y < x → f 0 y > f 0 x) ∧
            x > 0 → (∀ y : ℝ, y > x → f 0 y > f 0 x) :=
sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_a_pos :
  ∀ a : ℝ, a > 0 → 
  ∀ x : ℝ, (x < -2/a → (∀ y : ℝ, y < x → f a y < f a x)) ∧
           (x > 0 → (∀ y : ℝ, y > x → f a y > f a x)) ∧
           (-2/a < x ∧ x < 0 → (∀ y : ℝ, -2/a < y ∧ y < x → f a y > f a x)) :=
sorry

-- Theorem for monotonicity when a < 0
theorem monotonicity_a_neg :
  ∀ a : ℝ, a < 0 → 
  ∀ x : ℝ, (x < 0 → (∀ y : ℝ, y < x → f a y > f a x)) ∧
           (x > -2/a → (∀ y : ℝ, y > x → f a y < f a x)) ∧
           (0 < x ∧ x < -2/a → (∀ y : ℝ, x < y ∧ y < -2/a → f a y > f a x)) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l1458_145801


namespace NUMINAMATH_CALUDE_total_spending_over_four_years_l1458_145810

/-- The annual toy spending of three friends over four years. -/
def annual_toy_spending (trevor_spending : ℕ) (reed_diff : ℕ) (quinn_ratio : ℕ) (years : ℕ) : ℕ :=
  let reed_spending := trevor_spending - reed_diff
  let quinn_spending := reed_spending / quinn_ratio
  (trevor_spending + reed_spending + quinn_spending) * years

/-- Theorem stating the total spending of three friends over four years. -/
theorem total_spending_over_four_years :
  annual_toy_spending 80 20 2 4 = 680 := by
  sorry

#eval annual_toy_spending 80 20 2 4

end NUMINAMATH_CALUDE_total_spending_over_four_years_l1458_145810
