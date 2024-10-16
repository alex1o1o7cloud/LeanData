import Mathlib

namespace NUMINAMATH_CALUDE_chemistry_alone_count_l2770_277072

/-- Represents the number of students in a school with chemistry and biology classes -/
structure School where
  total : ℕ
  chemistry : ℕ
  biology : ℕ
  both : ℕ

/-- The conditions of the school -/
def school_conditions (s : School) : Prop :=
  s.total = 100 ∧
  s.chemistry + s.biology - s.both = s.total ∧
  s.chemistry = 4 * s.biology ∧
  s.both = 10

/-- The theorem stating that under the given conditions, 
    the number of students in chemistry class alone is 80 -/
theorem chemistry_alone_count (s : School) 
  (h : school_conditions s) : s.chemistry - s.both = 80 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_alone_count_l2770_277072


namespace NUMINAMATH_CALUDE_fibonacci_matrix_power_fibonacci_determinant_l2770_277094

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fibonacci (n+1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := 
  ![![fibonacci (n+1), fibonacci n],
    ![fibonacci n, fibonacci (n-1)]]

theorem fibonacci_matrix_power (n : ℕ) :
  (Matrix.of ![![1, 1], ![1, 0]] : Matrix (Fin 2) (Fin 2) ℕ) ^ n = fibonacci_matrix n := by
  sorry

theorem fibonacci_determinant (n : ℕ) :
  fibonacci (n+1) * fibonacci (n-1) - fibonacci n ^ 2 = (-1 : ℤ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_matrix_power_fibonacci_determinant_l2770_277094


namespace NUMINAMATH_CALUDE_evaluate_expression_l2770_277051

theorem evaluate_expression : (2 : ℕ) ^ (3 ^ 2) + 3 ^ (2 ^ 3) = 7073 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2770_277051


namespace NUMINAMATH_CALUDE_medium_size_can_be_rational_l2770_277039

-- Define the popcorn sizes
structure PopcornSize where
  name : String
  amount : Nat
  price : Nat

-- Define the customer's preferences
structure CustomerPreferences where
  budget : Nat
  wantsDrink : Bool
  preferBalancedMeal : Bool

-- Define the utility function
def utility (choice : PopcornSize) (prefs : CustomerPreferences) : Nat :=
  sorry

-- Define the theorem
theorem medium_size_can_be_rational (small medium large : PopcornSize) 
  (prefs : CustomerPreferences) : 
  small.name = "small" → 
  small.amount = 50 → 
  small.price = 200 →
  medium.name = "medium" → 
  medium.amount = 70 → 
  medium.price = 400 →
  large.name = "large" → 
  large.amount = 130 → 
  large.price = 500 →
  prefs.budget = 500 →
  prefs.wantsDrink = true →
  prefs.preferBalancedMeal = true →
  ∃ (drink_price : Nat), 
    utility medium prefs + utility (PopcornSize.mk "drink" 0 drink_price) prefs ≥ 
    max (utility small prefs) (utility large prefs) :=
  sorry


end NUMINAMATH_CALUDE_medium_size_can_be_rational_l2770_277039


namespace NUMINAMATH_CALUDE_sin_greater_than_cos_l2770_277083

theorem sin_greater_than_cos (x : Real) (h : -7*Real.pi/4 < x ∧ x < -3*Real.pi/2) :
  Real.sin (x + 9*Real.pi/4) > Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_greater_than_cos_l2770_277083


namespace NUMINAMATH_CALUDE_equation_solutions_l2770_277017

theorem equation_solutions :
  (∀ x : ℝ, 5 * x^2 - 10 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) ∧
  (∀ x : ℝ, 3 * (x - 4)^2 = 375 ↔ x = 4 + 5 * Real.sqrt 5 ∨ x = 4 - 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2770_277017


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2770_277020

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, -1)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2770_277020


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2770_277075

theorem smaller_number_proof (x y : ℕ) 
  (h1 : y - x = 2395)
  (h2 : y = 6 * x + 15) :
  x = 476 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2770_277075


namespace NUMINAMATH_CALUDE_wage_increase_proof_l2770_277028

/-- The original daily wage of a worker -/
def original_wage : ℝ := 20

/-- The percentage increase in the worker's wage -/
def wage_increase_percent : ℝ := 40

/-- The new daily wage after the increase -/
def new_wage : ℝ := 28

/-- Theorem stating that the original wage increased by 40% equals the new wage -/
theorem wage_increase_proof : 
  original_wage * (1 + wage_increase_percent / 100) = new_wage := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_proof_l2770_277028


namespace NUMINAMATH_CALUDE_weight_of_BaO_l2770_277066

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of barium oxide (BaO) in g/mol -/
def molecular_weight_BaO : ℝ := atomic_weight_Ba + atomic_weight_O

/-- The number of moles of barium oxide -/
def moles_BaO : ℝ := 6

/-- Theorem: The weight of 6 moles of barium oxide (BaO) is 919.98 grams -/
theorem weight_of_BaO : moles_BaO * molecular_weight_BaO = 919.98 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaO_l2770_277066


namespace NUMINAMATH_CALUDE_point_coordinates_l2770_277078

/-- Given a point P with coordinates (2m+4, m-1), prove that P has coordinates (-6, -6) 
    under the condition that it lies on the y-axis or its distance from the y-axis is 6, 
    and it lies in the third quadrant and is equidistant from both coordinate axes. -/
theorem point_coordinates (m : ℝ) : 
  (((2*m + 4 = 0) ∨ (|2*m + 4| = 6)) ∧ 
   (2*m + 4 < 0) ∧ (m - 1 < 0) ∧ 
   (|2*m + 4| = |m - 1|)) → 
  (2*m + 4 = -6 ∧ m - 1 = -6) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2770_277078


namespace NUMINAMATH_CALUDE_president_and_committee_count_l2770_277004

/-- The number of ways to choose a president and a committee from a group --/
def choose_president_and_committee (group_size : ℕ) (committee_size : ℕ) : ℕ :=
  group_size * (Nat.choose (group_size - 1) committee_size)

/-- Theorem stating the number of ways to choose a president and a 3-person committee from 10 people --/
theorem president_and_committee_count :
  choose_president_and_committee 10 3 = 840 := by
  sorry

#eval choose_president_and_committee 10 3

end NUMINAMATH_CALUDE_president_and_committee_count_l2770_277004


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2770_277061

/-- Given a rhombus whose diagonal lengths are the roots of x^2 - 14x + 48 = 0, its perimeter is 20 -/
theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 → 
  x₂^2 - 14*x₂ + 48 = 0 → 
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2770_277061


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2770_277099

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = m * 2^1997 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2770_277099


namespace NUMINAMATH_CALUDE_lcm_consecutive_sum_l2770_277016

theorem lcm_consecutive_sum (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (Nat.lcm a (Nat.lcm b c) = 168) → (a + b + c = 21) := by
  sorry

end NUMINAMATH_CALUDE_lcm_consecutive_sum_l2770_277016


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_integers_l2770_277080

theorem largest_divisor_of_consecutive_even_integers (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) = 48 * k :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_integers_l2770_277080


namespace NUMINAMATH_CALUDE_angle_U_measure_l2770_277002

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Measure of angle F
  I : ℝ  -- Measure of angle I
  U : ℝ  -- Measure of angle U
  G : ℝ  -- Measure of angle G
  E : ℝ  -- Measure of angle E
  R : ℝ  -- Measure of angle R

/-- The theorem stating the property of angle U in the given hexagon -/
theorem angle_U_measure (FIGURE : Hexagon) 
  (h1 : FIGURE.F = FIGURE.I ∧ FIGURE.I = FIGURE.U)  -- ∠F ≅ ∠I ≅ ∠U
  (h2 : FIGURE.G + FIGURE.E = 180)  -- ∠G is supplementary to ∠E
  (h3 : FIGURE.R = 2 * FIGURE.U)  -- ∠R = 2∠U
  : FIGURE.U = 108 := by
  sorry

#check angle_U_measure

end NUMINAMATH_CALUDE_angle_U_measure_l2770_277002


namespace NUMINAMATH_CALUDE_coffee_cost_calculation_coffee_cost_calculation_proof_l2770_277041

/-- The daily cost of making coffee given a coffee machine purchase and previous coffee consumption habits. -/
theorem coffee_cost_calculation (machine_cost : ℝ) (discount : ℝ) (previous_coffees_per_day : ℕ) 
  (previous_coffee_price : ℝ) (payback_days : ℕ) (daily_cost : ℝ) : Prop :=
  machine_cost = 200 ∧ 
  discount = 20 ∧
  previous_coffees_per_day = 2 ∧
  previous_coffee_price = 4 ∧
  payback_days = 36 →
  daily_cost = 3

/-- Proof of the coffee cost calculation theorem. -/
theorem coffee_cost_calculation_proof : 
  coffee_cost_calculation 200 20 2 4 36 3 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_calculation_coffee_cost_calculation_proof_l2770_277041


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l2770_277055

/-- The area of the region bounded by x = √(e^y - 1), x = 0, and y = ln 2 -/
theorem area_bounded_by_curves : ∃ (S : ℝ),
  (∀ x y : ℝ, x = Real.sqrt (Real.exp y - 1) → 
    0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ Real.log 2) →
  S = ∫ x in (0)..(1), (Real.log 2 - Real.log (x^2 + 1)) →
  S = 2 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l2770_277055


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2770_277006

/-- Given a line L1 with equation 3x + 2y - 7 = 0, prove that the line L2 passing through
    the point (-1, 2) and perpendicular to L1 has the equation 2x - 3y + 8 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (3 * x + 2 * y - 7 = 0) →  -- equation of L1
  (2 * (-1) - 3 * 2 + 8 = 0) ∧  -- L2 passes through (-1, 2)
  (3 * 2 + 2 * 3 = 0) →  -- perpendicularity condition
  (2 * x - 3 * y + 8 = 0)  -- equation of L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2770_277006


namespace NUMINAMATH_CALUDE_remaining_length_is_24_l2770_277012

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  sides : List ℝ
  perpendicular : Bool

/-- Function to calculate the total length of remaining segments after removal -/
def remainingLength (figure : PerpendicularFigure) (removedSides : ℕ) : ℝ :=
  sorry

/-- Theorem stating the total length of remaining segments is 24 units -/
theorem remaining_length_is_24 (figure : PerpendicularFigure) 
  (h1 : figure.sides = [10, 3, 8, 1, 1, 5]) 
  (h2 : figure.perpendicular = true) 
  (h3 : removedSides = 6) : 
  remainingLength figure removedSides = 24 :=
sorry

end NUMINAMATH_CALUDE_remaining_length_is_24_l2770_277012


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2770_277032

/-- A color type representing red, green, or blue -/
inductive Color
  | Red
  | Green
  | Blue

/-- A type representing a 4 x 82 grid where each point is colored -/
def ColoredGrid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (grid : ColoredGrid) (i j p q : Nat) : Prop :=
  i < j ∧ p < q ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨i, by sorry⟩ ⟨q, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨p, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨q, by sorry⟩

/-- The main theorem stating that any 4 x 82 grid colored with three colors
    contains a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (grid : ColoredGrid) :
  ∃ i j p q, isMonochromaticRectangle grid i j p q :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2770_277032


namespace NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l2770_277071

theorem circle_center_in_second_quadrant (a : ℝ) (h : a > 12) :
  let center := (-(a/2), a)
  (center.1 < 0 ∧ center.2 > 0) ∧
  (∀ x y : ℝ, x^2 + y^2 + a*x - 2*a*y + a^2 + 3*a = 0 ↔ 
    (x - center.1)^2 + (y - center.2)^2 = (a^2/4 - 3*a)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l2770_277071


namespace NUMINAMATH_CALUDE_probability_at_least_one_first_class_l2770_277043

theorem probability_at_least_one_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat) :
  total = 6 →
  first_class = 4 →
  second_class = 2 →
  selected = 2 →
  (1 : ℚ) - (Nat.choose second_class selected : ℚ) / (Nat.choose total selected : ℚ) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_first_class_l2770_277043


namespace NUMINAMATH_CALUDE_expression_simplification_l2770_277073

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (2 * x / (x^2 - 4) - 1 / (x + 2)) / ((x - 1) / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2770_277073


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2770_277088

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2770_277088


namespace NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l2770_277030

theorem units_digit_of_sum_cubes : (52^3 + 29^3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l2770_277030


namespace NUMINAMATH_CALUDE_commercial_break_length_is_47_l2770_277053

/-- Calculates the total length of a commercial break given the following conditions:
    - Three commercials of 5, 6, and 7 minutes
    - Eleven 2-minute commercials
    - Two of the 2-minute commercials overlap with a 3-minute interruption and restart after
-/
def commercial_break_length : ℕ :=
  let long_commercials := 5 + 6 + 7
  let short_commercials := 11 * 2
  let interruption := 3
  let restarted_commercials := 2 * 2
  long_commercials + short_commercials + interruption + restarted_commercials

/-- Theorem stating that the commercial break length is 47 minutes -/
theorem commercial_break_length_is_47 : commercial_break_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_commercial_break_length_is_47_l2770_277053


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l2770_277031

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific conditions -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount ∈ [28, 29, 30, 31]

/-- Function to determine the day of week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l2770_277031


namespace NUMINAMATH_CALUDE_unique_arrangements_zoo_animals_l2770_277027

def num_elephants : ℕ := 4
def num_rabbits : ℕ := 3
def num_parrots : ℕ := 5

def total_animals : ℕ := num_elephants + num_rabbits + num_parrots

theorem unique_arrangements_zoo_animals :
  (Nat.factorial 3) * (Nat.factorial num_elephants) * (Nat.factorial num_rabbits) * (Nat.factorial num_parrots) = 103680 :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangements_zoo_animals_l2770_277027


namespace NUMINAMATH_CALUDE_trolley_theorem_l2770_277085

def trolley_problem (X : ℕ) : Prop :=
  let initial_passengers := 10
  let second_stop_off := 3
  let second_stop_on := 2 * initial_passengers
  let third_stop_off := 18
  let third_stop_on := 2
  let fourth_stop_off := 5
  let fourth_stop_on := X
  let final_passengers := 
    initial_passengers - second_stop_off + second_stop_on - 
    third_stop_off + third_stop_on - fourth_stop_off + fourth_stop_on
  final_passengers = 6 + X

theorem trolley_theorem (X : ℕ) : 
  trolley_problem X :=
sorry

end NUMINAMATH_CALUDE_trolley_theorem_l2770_277085


namespace NUMINAMATH_CALUDE_quadratic_roots_and_reciprocals_l2770_277059

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * (k + 1) * x + k - 1

-- Theorem statement
theorem quadratic_roots_and_reciprocals (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ 
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_reciprocals_l2770_277059


namespace NUMINAMATH_CALUDE_superior_rainbow_max_quantity_l2770_277058

/-- Represents the mixing ratios for Superior Rainbow paint -/
structure MixingRatios where
  red : Rat
  white : Rat
  blue : Rat
  yellow : Rat

/-- Represents the available paint quantities -/
structure AvailablePaint where
  red : Nat
  white : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the maximum quantity of Superior Rainbow paint -/
def maxSuperiorRainbow (ratios : MixingRatios) (available : AvailablePaint) : Nat :=
  sorry

/-- Theorem: The maximum quantity of Superior Rainbow paint is 121 pints -/
theorem superior_rainbow_max_quantity :
  let ratios : MixingRatios := ⟨3/4, 2/3, 1/4, 1/6⟩
  let available : AvailablePaint := ⟨50, 45, 20, 15⟩
  maxSuperiorRainbow ratios available = 121 := by
  sorry

end NUMINAMATH_CALUDE_superior_rainbow_max_quantity_l2770_277058


namespace NUMINAMATH_CALUDE_fifa_world_cup_players_l2770_277060

/-- The number of teams in the 17th FIFA World Cup -/
def num_teams : ℕ := 35

/-- The number of players in each team -/
def players_per_team : ℕ := 23

/-- The total number of players in the 17th FIFA World Cup -/
def total_players : ℕ := num_teams * players_per_team

theorem fifa_world_cup_players :
  total_players = 805 := by sorry

end NUMINAMATH_CALUDE_fifa_world_cup_players_l2770_277060


namespace NUMINAMATH_CALUDE_probability_sum_six_l2770_277095

-- Define a die as having 6 faces
def die : ℕ := 6

-- Define the favorable outcomes (combinations that sum to 6)
def favorable_outcomes : ℕ := 5

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die * die

-- State the theorem
theorem probability_sum_six (d : ℕ) (h : d = die) : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_l2770_277095


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixty_fourth_l2770_277062

theorem sin_product_equals_one_sixty_fourth :
  (Real.sin (70 * π / 180))^2 * (Real.sin (50 * π / 180))^2 * (Real.sin (10 * π / 180))^2 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixty_fourth_l2770_277062


namespace NUMINAMATH_CALUDE_ad_sequence_count_l2770_277045

/-- Represents the number of Olympic ads -/
def num_olympic_ads : ℕ := 3

/-- Represents the number of commercial ads -/
def num_commercial_ads : ℕ := 2

/-- Represents the total number of ads -/
def total_ads : ℕ := num_olympic_ads + num_commercial_ads

/-- Represents the constraint that the last ad must be an Olympic ad -/
def last_ad_is_olympic : Prop := true

/-- Represents the constraint that commercial ads cannot be played consecutively -/
def no_consecutive_commercial_ads : Prop := true

/-- The number of different playback sequences -/
def num_sequences : ℕ := 36

theorem ad_sequence_count :
  num_olympic_ads = 3 →
  num_commercial_ads = 2 →
  total_ads = 5 →
  last_ad_is_olympic →
  no_consecutive_commercial_ads →
  num_sequences = 36 :=
by sorry

end NUMINAMATH_CALUDE_ad_sequence_count_l2770_277045


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l2770_277003

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Conditions for jelly sales -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 6

theorem strawberry_jelly_sales (s : JellySales) :
  valid_jelly_sales s → s.strawberry = 18 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l2770_277003


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_perimeter_6_l2770_277070

/-- The minimum value of the hypotenuse of a right triangle with perimeter 6 -/
theorem min_hypotenuse_right_triangle_perimeter_6 :
  ∃ (c : ℝ), c > 0 ∧ c = 6 * (Real.sqrt 2 - 1) ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = c^2 → a + b + c = 6 →
  c ≤ 6 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_perimeter_6_l2770_277070


namespace NUMINAMATH_CALUDE_small_pump_fills_in_three_hours_l2770_277036

-- Define the filling rates for the pumps
def large_pump_rate : ℝ := 4 -- 1 / (1/4)
def combined_time : ℝ := 0.23076923076923078

-- Define the time it takes for the small pump to fill the tank
def small_pump_time : ℝ := 3

-- Theorem statement
theorem small_pump_fills_in_three_hours :
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate = small_pump_time := by sorry

end NUMINAMATH_CALUDE_small_pump_fills_in_three_hours_l2770_277036


namespace NUMINAMATH_CALUDE_hno3_concentration_after_addition_l2770_277090

/-- Calculates the final concentration of HNO3 after adding pure HNO3 to a solution -/
theorem hno3_concentration_after_addition
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (pure_hno3_added : ℝ)
  (h1 : initial_volume = 60)
  (h2 : initial_concentration = 0.35)
  (h3 : pure_hno3_added = 18) :
  let final_volume := initial_volume + pure_hno3_added
  let initial_hno3 := initial_volume * initial_concentration
  let final_hno3 := initial_hno3 + pure_hno3_added
  let final_concentration := final_hno3 / final_volume
  final_concentration = 0.5 := by sorry

end NUMINAMATH_CALUDE_hno3_concentration_after_addition_l2770_277090


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2770_277015

theorem triangle_angle_proof (a b c : ℝ) (S : ℝ) (C : ℝ) :
  a > 0 → b > 0 → c > 0 → S > 0 →
  0 < C → C < π →
  S = (1/2) * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2770_277015


namespace NUMINAMATH_CALUDE_square_root_sum_equals_five_l2770_277044

theorem square_root_sum_equals_five : 
  Real.sqrt ((5 / 2 - 3 * Real.sqrt 3 / 2) ^ 2) + Real.sqrt ((5 / 2 + 3 * Real.sqrt 3 / 2) ^ 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_five_l2770_277044


namespace NUMINAMATH_CALUDE_car_wash_rate_l2770_277068

def babysitting_families : ℕ := 4
def babysitting_rate : ℕ := 30
def cars_washed : ℕ := 5
def total_raised : ℕ := 180

theorem car_wash_rate :
  (total_raised - babysitting_families * babysitting_rate) / cars_washed = 12 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_rate_l2770_277068


namespace NUMINAMATH_CALUDE_derivative_of_f_l2770_277034

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x

theorem derivative_of_f (x : ℝ) : deriv f x = f' x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2770_277034


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l2770_277050

/-- Given four points on a Cartesian plane, prove that if segment AB is parallel to segment XY, then k = -6 -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ) 
  (hA : A = (-4, 0)) 
  (hB : B = (0, -4)) 
  (hX : X = (0, 8)) 
  (hY : Y = (14, k))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (B.2 - A.2) * (Y.1 - X.1)) : 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l2770_277050


namespace NUMINAMATH_CALUDE_range_of_exponential_function_l2770_277013

theorem range_of_exponential_function :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, 3^x = y := by
  sorry

end NUMINAMATH_CALUDE_range_of_exponential_function_l2770_277013


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2770_277000

def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ 2/x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2770_277000


namespace NUMINAMATH_CALUDE_remainder_problem_l2770_277024

theorem remainder_problem (N : ℕ) : 
  (N / 5 = 5) ∧ (N % 5 = 0) → N % 11 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2770_277024


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2770_277097

theorem sum_product_inequality (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_one : a + b + c + d = 1) :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ 1 / 27 + 176 * a * b * c * d / 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2770_277097


namespace NUMINAMATH_CALUDE_remainder_problem_l2770_277021

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) :
  N % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2770_277021


namespace NUMINAMATH_CALUDE_inequality_of_reciprocal_logs_l2770_277029

theorem inequality_of_reciprocal_logs (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  1 / Real.log a > 1 / Real.log b :=
by sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocal_logs_l2770_277029


namespace NUMINAMATH_CALUDE_extracted_25_30_is_120_l2770_277025

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireCount where
  group_8_12 : ℕ
  group_13_18 : ℕ
  group_19_24 : ℕ
  group_25_30 : ℕ

/-- Represents the sample extracted from the collected questionnaires -/
structure SampleCount where
  total : ℕ
  group_13_18 : ℕ

/-- Calculates the number of questionnaires extracted from the 25-30 age group -/
def extracted_25_30 (collected : QuestionnaireCount) (sample : SampleCount) : ℕ :=
  (collected.group_25_30 * sample.group_13_18) / collected.group_13_18

theorem extracted_25_30_is_120 (collected : QuestionnaireCount) (sample : SampleCount) :
  collected.group_8_12 = 120 →
  collected.group_13_18 = 180 →
  collected.group_19_24 = 240 →
  sample.total = 300 →
  sample.group_13_18 = 60 →
  extracted_25_30 collected sample = 120 := by
  sorry

#check extracted_25_30_is_120

end NUMINAMATH_CALUDE_extracted_25_30_is_120_l2770_277025


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2770_277091

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define point D on the angle bisector of A
variable (D : EuclideanSpace ℝ (Fin 2))

-- Assumption that ABC is a triangle
variable (h_triangle : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))

-- Assumption that D is on BC
variable (h_D_on_BC : D ∈ LineSegment B C)

-- Assumption that AD is the angle bisector of angle BAC
variable (h_angle_bisector : AngleBisector A B C D)

-- Theorem statement
theorem angle_bisector_theorem :
  (dist A B) / (dist A C) = (dist B D) / (dist C D) := by sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2770_277091


namespace NUMINAMATH_CALUDE_int_coord_triangle_area_rational_l2770_277048

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three integer points
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of a triangle with integer coordinates is rational
theorem int_coord_triangle_area_rational (t : IntTriangle) : 
  ∃ q : ℚ, triangleArea t = q :=
sorry

end NUMINAMATH_CALUDE_int_coord_triangle_area_rational_l2770_277048


namespace NUMINAMATH_CALUDE_equation_with_operations_l2770_277052

theorem equation_with_operations : ∃ (op1 op2 op3 : ℕ → ℕ → ℕ), 
  op1 6 (op2 3 (op3 4 2)) = 24 :=
sorry

end NUMINAMATH_CALUDE_equation_with_operations_l2770_277052


namespace NUMINAMATH_CALUDE_fruit_purchase_total_l2770_277009

/-- The total amount paid for a fruit purchase given the quantity and rate per kg for two types of fruits -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that the total amount paid for 8 kg of grapes at 70 per kg and 9 kg of mangoes at 45 per kg is 965 -/
theorem fruit_purchase_total :
  total_amount_paid 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l2770_277009


namespace NUMINAMATH_CALUDE_min_abs_sum_l2770_277010

theorem min_abs_sum (x : ℝ) : 
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l2770_277010


namespace NUMINAMATH_CALUDE_solution_set_implies_a_eq_one_solution_set_varies_with_a_l2770_277056

/-- The quadratic function f(x) = ax^2 + (1-2a)x - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - 2

/-- The solution set of f(x) > 0 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x < -1 ∨ x > 2}

/-- Theorem: When the solution set of f(x) > 0 is {x | x < -1 or x > 2}, a = 1 -/
theorem solution_set_implies_a_eq_one :
  (∀ x, f 1 x > 0 ↔ x ∈ solution_set_a1) → 1 = 1 := by sorry

/-- The solution set of f(x) > 0 for a > 0 -/
def solution_set_a_pos (a : ℝ) : Set ℝ := {x | x < -1/a ∨ x > 2}

/-- The solution set of f(x) > 0 for a = 0 -/
def solution_set_a_zero : Set ℝ := {x | x > 2}

/-- The solution set of f(x) > 0 for -1/2 < a < 0 -/
def solution_set_a_neg_small (a : ℝ) : Set ℝ := {x | 2 < x ∧ x < -1/a}

/-- The solution set of f(x) > 0 for a = -1/2 -/
def solution_set_a_neg_half : Set ℝ := ∅

/-- The solution set of f(x) > 0 for a < -1/2 -/
def solution_set_a_neg_large (a : ℝ) : Set ℝ := {x | -1/a < x ∧ x < 2}

/-- Theorem: The solution set of f(x) > 0 varies for different ranges of a ∈ ℝ -/
theorem solution_set_varies_with_a (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 0 ∧ x ∈ solution_set_a_pos a) ∨
    (a = 0 ∧ x ∈ solution_set_a_zero) ∨
    (-1/2 < a ∧ a < 0 ∧ x ∈ solution_set_a_neg_small a) ∨
    (a = -1/2 ∧ x ∈ solution_set_a_neg_half) ∨
    (a < -1/2 ∧ x ∈ solution_set_a_neg_large a)) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_eq_one_solution_set_varies_with_a_l2770_277056


namespace NUMINAMATH_CALUDE_pythagorean_reciprocal_perimeter_l2770_277042

theorem pythagorean_reciprocal_perimeter 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (pythagorean_reciprocal : (a + b) / c = Real.sqrt 2) 
  (area : a * b / 2 = 4) : 
  a + b + c = 4 * Real.sqrt 2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_reciprocal_perimeter_l2770_277042


namespace NUMINAMATH_CALUDE_smallest_other_integer_l2770_277011

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  Nat.gcd m n = x + 6 →
  Nat.lcm m n = x * (x + 6) →
  m = 60 →
  (∀ k : ℕ, k > 0 ∧ k < n → 
    (Nat.gcd 60 k ≠ x + 6 ∨ Nat.lcm 60 k ≠ x * (x + 6))) →
  n = 93 := by
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l2770_277011


namespace NUMINAMATH_CALUDE_bridge_length_l2770_277069

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2770_277069


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2770_277005

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I * (1 + Complex.I)) / (1 - 2 * Complex.I)
  z.re < 0 ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2770_277005


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_x_l2770_277023

theorem negation_of_universal_positive_square_plus_x (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_x_l2770_277023


namespace NUMINAMATH_CALUDE_subset_complement_relation_l2770_277074

universe u

theorem subset_complement_relation {U : Type u} (M N : Set U) 
  (hM : M.Nonempty) (hN : N.Nonempty) (h : N ⊆ Mᶜ) : M ⊆ Nᶜ := by
  sorry

end NUMINAMATH_CALUDE_subset_complement_relation_l2770_277074


namespace NUMINAMATH_CALUDE_max_red_tiles_100x100_l2770_277040

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents the number of colors used for tiling -/
def num_colors : ℕ := 4

/-- Defines the property that no two tiles of the same color touch each other -/
def no_adjacent_same_color (g : Grid) : Prop := sorry

/-- The maximum number of tiles of a single color in the grid -/
def max_single_color_tiles (g : Grid) : ℕ := (g.size ^ 2) / 4

/-- Theorem stating the maximum number of red tiles in a 100x100 grid -/
theorem max_red_tiles_100x100 (g : Grid) (h1 : g.size = 100) (h2 : no_adjacent_same_color g) : 
  max_single_color_tiles g = 2500 := by sorry

end NUMINAMATH_CALUDE_max_red_tiles_100x100_l2770_277040


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l2770_277026

/-- The transformation matrix M -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

/-- Rotation matrix for 90 degrees counterclockwise -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- Scaling matrix with factor 2 -/
def S : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

theorem transformation_matrix_correct :
  M = S * R :=
sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l2770_277026


namespace NUMINAMATH_CALUDE_function_symmetry_l2770_277001

/-- Given a function f(x) = a*sin(x) - b*cos(x) where f(x) takes an extreme value when x = π/4,
    prove that y = f(3π/4 - x) is an odd function and its graph is symmetric about (π, 0) -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x - b * Real.cos x
  (∃ (extreme : ℝ), f (π/4) = extreme ∧ ∀ x, f x ≤ extreme) →
  let y : ℝ → ℝ := λ x ↦ f (3*π/4 - x)
  (∀ x, y (-x) = -y x) ∧  -- odd function
  (∀ x, y (2*π - x) = -y x)  -- symmetry about (π, 0)
:= by sorry

end NUMINAMATH_CALUDE_function_symmetry_l2770_277001


namespace NUMINAMATH_CALUDE_total_books_l2770_277033

theorem total_books (jason_books mary_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) : 
  jason_books + mary_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2770_277033


namespace NUMINAMATH_CALUDE_total_reams_is_five_l2770_277081

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := 3

/-- The total number of reams of paper bought by Haley's mom -/
def total_reams : ℕ := reams_for_haley + reams_for_sister

theorem total_reams_is_five : total_reams = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_reams_is_five_l2770_277081


namespace NUMINAMATH_CALUDE_weight_difference_l2770_277063

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 50 →
  (w_a + w_b + w_c + w_d) / 4 = 53 →
  (w_b + w_c + w_d + w_e) / 4 = 51 →
  w_a = 73 →
  w_e > w_d →
  w_e - w_d = 3 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l2770_277063


namespace NUMINAMATH_CALUDE_complex_argument_and_reality_l2770_277077

noncomputable def arg (z : ℂ) : ℝ := Real.arctan (z.im / z.re)

theorem complex_argument_and_reality (θ : ℝ) (a : ℝ) :
  0 < θ ∧ θ < 2 * Real.pi →
  let z : ℂ := 1 - Real.cos θ + Complex.I * Real.sin θ
  let u : ℂ := a^2 + Complex.I * a
  (z * u).re = 0 →
  (
    (0 < θ ∧ θ < Real.pi → arg u = θ / 2) ∧
    (Real.pi < θ ∧ θ < 2 * Real.pi → arg u = Real.pi + θ / 2)
  ) ∧
  ∀ ω : ℂ, ω = z^2 + u^2 + 2 * z * u → ω.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_and_reality_l2770_277077


namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_53_l2770_277057

theorem ones_digit_of_3_to_53 : (3^53 : ℕ) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_53_l2770_277057


namespace NUMINAMATH_CALUDE_ratio_problem_l2770_277022

theorem ratio_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y = x * (1 + 14.285714285714285 / 100)) : 
  x / y = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2770_277022


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l2770_277089

def total_students : ℕ := 2700
def freshmen : ℕ := 900
def sophomores : ℕ := 1200
def juniors : ℕ := 600
def sample_size : ℕ := 135

theorem stratified_sample_composition :
  let freshmen_sample := (freshmen * sample_size) / total_students
  let sophomores_sample := (sophomores * sample_size) / total_students
  let juniors_sample := (juniors * sample_size) / total_students
  freshmen_sample = 45 ∧ sophomores_sample = 60 ∧ juniors_sample = 30 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l2770_277089


namespace NUMINAMATH_CALUDE_eggs_taken_away_l2770_277079

/-- Proof that the number of eggs Amy took away is the difference between Virginia's initial and final number of eggs -/
theorem eggs_taken_away (initial_eggs final_eggs : ℕ) (h1 : initial_eggs = 96) (h2 : final_eggs = 93) :
  initial_eggs - final_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_away_l2770_277079


namespace NUMINAMATH_CALUDE_solution_set_when_m_zero_solution_set_all_reals_l2770_277067

/-- The quadratic inequality in question -/
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

/-- The solution set when m = 0 -/
theorem solution_set_when_m_zero :
  {x : ℝ | quadratic_inequality 0 x} = Set.Ioo (-2) 1 := by sorry

/-- The condition for the solution set to be all real numbers -/
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | quadratic_inequality m x} = Set.univ) ↔ (m ∈ Set.Icc 1 9) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_zero_solution_set_all_reals_l2770_277067


namespace NUMINAMATH_CALUDE_unique_pair_power_sum_l2770_277084

theorem unique_pair_power_sum : 
  ∃! (a b : ℕ), ∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_pair_power_sum_l2770_277084


namespace NUMINAMATH_CALUDE_not_multiple_of_121_l2770_277047

theorem not_multiple_of_121 (n : ℤ) : ¬(121 ∣ (n^2 + 2*n + 12)) := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_121_l2770_277047


namespace NUMINAMATH_CALUDE_min_value_theorem_l2770_277035

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y = 4) :
  (∃ (m : ℝ), m = 3/8 + Real.sqrt 2/4 ∧
    ∀ (z : ℝ), z = 1 / (2 * x + 1) + 1 / (3 * y + 2) → z ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2770_277035


namespace NUMINAMATH_CALUDE_ramanujan_hardy_game_l2770_277087

/-- Given two complex numbers whose product is 48 - 12i, and one of the numbers is 6 + 2i,
    prove that the other number is 39/5 - 21/5i. -/
theorem ramanujan_hardy_game (z w : ℂ) : 
  z * w = 48 - 12 * I ∧ w = 6 + 2 * I → z = 39/5 - 21/5 * I := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_game_l2770_277087


namespace NUMINAMATH_CALUDE_distance_for_given_point_l2770_277054

/-- The distance between a point and its symmetric point about the x-axis --/
def distance_to_symmetric_point (x y : ℝ) : ℝ := 2 * |y|

/-- Theorem: The distance between (2, -3) and its symmetric point about the x-axis is 6 --/
theorem distance_for_given_point : distance_to_symmetric_point 2 (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_given_point_l2770_277054


namespace NUMINAMATH_CALUDE_ratio_calculation_l2770_277093

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (5 * A + 3 * B) / (3 * C - 2 * A) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2770_277093


namespace NUMINAMATH_CALUDE_smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l2770_277096

/-- A function that checks if all digits of a natural number are 0 or 1 -/
def all_digits_zero_or_one (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number with digits 0 or 1 divisible by 225 -/
noncomputable def smallest_zero_one_divisible_by_225 : ℕ := sorry

theorem smallest_zero_one_divisible_by_225_is_11111111100 :
  smallest_zero_one_divisible_by_225 = 11111111100 :=
by
  sorry

theorem smallest_zero_one_divisible_by_225_properties :
  let n := smallest_zero_one_divisible_by_225
  all_digits_zero_or_one n ∧ n % 225 = 0 ∧ 
  ∀ m : ℕ, m < n → ¬(all_digits_zero_or_one m ∧ m % 225 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l2770_277096


namespace NUMINAMATH_CALUDE_smallest_high_efficiency_l2770_277037

def efficiency (n : ℕ) : ℚ :=
  (n - (Nat.totient n)) / n

theorem smallest_high_efficiency : 
  ∀ m : ℕ, m < 30030 → efficiency m ≤ 4/5 ∧ efficiency 30030 > 4/5 :=
sorry

end NUMINAMATH_CALUDE_smallest_high_efficiency_l2770_277037


namespace NUMINAMATH_CALUDE_miss_walter_stickers_l2770_277038

theorem miss_walter_stickers (gold : ℕ) (silver : ℕ) (bronze : ℕ) (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : silver = 2 * gold)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver - bronze = 20 := by
  sorry

end NUMINAMATH_CALUDE_miss_walter_stickers_l2770_277038


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_multiple_of_seven_l2770_277064

theorem sum_seven_consecutive_integers_multiple_of_seven (n : ℕ+) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_multiple_of_seven_l2770_277064


namespace NUMINAMATH_CALUDE_sour_candy_percentage_l2770_277065

theorem sour_candy_percentage (total_candies : ℕ) (num_people : ℕ) (good_candies_per_person : ℕ) :
  total_candies = 300 →
  num_people = 3 →
  good_candies_per_person = 60 →
  (total_candies - num_people * good_candies_per_person) / total_candies = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_sour_candy_percentage_l2770_277065


namespace NUMINAMATH_CALUDE_smallest_T_value_l2770_277018

theorem smallest_T_value : ∃ (m : ℕ), 
  (∀ k : ℕ, k < m → 8 * k < 2400) ∧ 
  8 * m ≥ 2400 ∧
  9 * m - 2400 = 300 := by
  sorry

end NUMINAMATH_CALUDE_smallest_T_value_l2770_277018


namespace NUMINAMATH_CALUDE_arun_weight_average_l2770_277019

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < 70)
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l2770_277019


namespace NUMINAMATH_CALUDE_love_all_girls_l2770_277092

-- Define the girls
inductive Girl
| Sue
| Marcia
| Diana

-- Define the love relation
def loves : Girl → Prop := sorry

-- State the theorem
theorem love_all_girls :
  -- Condition 1: I love at least one of the three girls
  (∃ g : Girl, loves g) →
  -- Condition 2: If I love Sue but not Diana, then I also love Marcia
  (loves Girl.Sue ∧ ¬loves Girl.Diana → loves Girl.Marcia) →
  -- Condition 3: I either love both Diana and Marcia, or I love neither of them
  ((loves Girl.Diana ∧ loves Girl.Marcia) ∨ (¬loves Girl.Diana ∧ ¬loves Girl.Marcia)) →
  -- Condition 4: If I love Diana, then I also love Sue
  (loves Girl.Diana → loves Girl.Sue) →
  -- Conclusion: I love all three girls
  (loves Girl.Sue ∧ loves Girl.Marcia ∧ loves Girl.Diana) :=
by sorry

end NUMINAMATH_CALUDE_love_all_girls_l2770_277092


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2770_277046

/-- The ratio of wire lengths for equivalent volume cubes -/
theorem wire_length_ratio (large_cube_edge : ℝ) (small_cube_edge : ℝ) : 
  large_cube_edge = 8 →
  small_cube_edge = 2 →
  (12 * large_cube_edge) / (12 * small_cube_edge * (large_cube_edge / small_cube_edge)^3) = 1/16 := by
  sorry

#check wire_length_ratio

end NUMINAMATH_CALUDE_wire_length_ratio_l2770_277046


namespace NUMINAMATH_CALUDE_unique_prime_base_l2770_277007

theorem unique_prime_base : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_base_l2770_277007


namespace NUMINAMATH_CALUDE_sally_napkins_l2770_277014

def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7
def total_material : ℕ := 5844

theorem sally_napkins :
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let remaining_material := total_material - tablecloth_area
  remaining_material / napkin_area = 8 := by sorry

end NUMINAMATH_CALUDE_sally_napkins_l2770_277014


namespace NUMINAMATH_CALUDE_borrowed_amount_correct_l2770_277076

/-- The amount of money borrowed, in Rupees -/
def borrowed_amount : ℝ := 5000

/-- The interest rate for borrowing, as a decimal -/
def borrow_rate : ℝ := 0.04

/-- The interest rate for lending, as a decimal -/
def lend_rate : ℝ := 0.07

/-- The duration of the loan in years -/
def duration : ℝ := 2

/-- The yearly gain from the transaction, in Rupees -/
def yearly_gain : ℝ := 150

/-- Theorem stating that the borrowed amount is correct given the conditions -/
theorem borrowed_amount_correct :
  borrowed_amount * borrow_rate * duration = 
  borrowed_amount * lend_rate * duration - yearly_gain * duration := by
  sorry

#check borrowed_amount_correct

end NUMINAMATH_CALUDE_borrowed_amount_correct_l2770_277076


namespace NUMINAMATH_CALUDE_difference_of_squares_528_529_l2770_277082

theorem difference_of_squares_528_529 : (528 * 528) - (527 * 529) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_528_529_l2770_277082


namespace NUMINAMATH_CALUDE_inclined_line_and_volume_l2770_277098

/-- A line passing through a point with a given inclination angle cosine -/
structure InclinedLine where
  point : ℝ × ℝ
  cos_angle : ℝ

/-- The general form equation coefficients of a line -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The volume of a geometric body -/
def GeometricBodyVolume : Type := ℝ

/-- Calculate the general form equation of the line -/
def calculateLineEquation (l : InclinedLine) : LineEquation :=
  sorry

/-- Calculate the volume of the geometric body -/
def calculateGeometricBodyVolume (eq : LineEquation) : GeometricBodyVolume :=
  sorry

theorem inclined_line_and_volume 
  (l : InclinedLine) 
  (h1 : l.point = (-1, 2)) 
  (h2 : l.cos_angle = Real.sqrt 2 / 2) : 
  let eq := calculateLineEquation l
  calculateGeometricBodyVolume eq = 9 * Real.pi ∧ 
  eq.a = 1 ∧ eq.b = -1 ∧ eq.c = -3 :=
sorry

end NUMINAMATH_CALUDE_inclined_line_and_volume_l2770_277098


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l2770_277086

theorem prime_power_divisibility (n p : ℕ) : 
  p.Prime → 
  n > 1 → 
  (((p - 1)^n + 1) % n^(p - 1) = 0) → 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l2770_277086


namespace NUMINAMATH_CALUDE_valid_medium_triangle_counts_l2770_277008

/-- Represents the side length of the original equilateral triangle -/
def originalSideLength : ℕ := 10

/-- Represents the side length of the smallest equilateral triangles -/
def smallestSideLength : ℕ := 1

/-- Represents the side length of the medium equilateral triangles -/
def mediumSideLength : ℕ := 2

/-- Represents the total number of shapes (triangles and parallelograms) -/
def totalShapes : ℕ := 25

/-- Predicate to check if a number is a valid count of medium triangles -/
def isValidMediumTriangleCount (m : ℕ) : Prop :=
  m % 2 = 1 ∧ 5 ≤ m ∧ m ≤ 25

/-- The set of all valid counts of medium triangles -/
def validMediumTriangleCounts : Set ℕ :=
  {m | isValidMediumTriangleCount m}

/-- Theorem stating the properties of valid medium triangle counts -/
theorem valid_medium_triangle_counts :
  validMediumTriangleCounts = {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25} :=
sorry

end NUMINAMATH_CALUDE_valid_medium_triangle_counts_l2770_277008


namespace NUMINAMATH_CALUDE_question_paper_combinations_l2770_277049

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem question_paper_combinations : choose 10 8 * choose 10 5 = 11340 := by
  sorry

end NUMINAMATH_CALUDE_question_paper_combinations_l2770_277049
