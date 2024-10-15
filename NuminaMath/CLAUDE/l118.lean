import Mathlib

namespace NUMINAMATH_CALUDE_complex_squared_plus_2i_l118_11858

theorem complex_squared_plus_2i (i : ℂ) : i^2 = -1 → (1 + i)^2 + 2*i = 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_plus_2i_l118_11858


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l118_11808

theorem tangent_point_coordinates (x y : ℝ) : 
  y = x^2 → -- Point (x, y) is on the curve y = x^2
  (2*x = 1) → -- Tangent line has slope 1 (tan(π/4) = 1)
  (x = 1/2 ∧ y = 1/4) := by -- The coordinates are (1/2, 1/4)
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l118_11808


namespace NUMINAMATH_CALUDE_max_value_b_plus_c_l118_11831

theorem max_value_b_plus_c (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + c) * (b^2 + a*c) = 4*a) : 
  b + c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_b_plus_c_l118_11831


namespace NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l118_11871

/-- A hotel charging system with a flat fee for the first night and a separate rate for additional nights. -/
structure HotelCharges where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Rate for each additional night

/-- Calculate the total cost for a given number of nights -/
def totalCost (h : HotelCharges) (nights : ℕ) : ℝ :=
  h.flatFee + h.nightlyRate * (nights - 1)

/-- Theorem stating the flat fee for the first night given the conditions -/
theorem hotel_flat_fee_calculation (h : HotelCharges) :
  totalCost h 2 = 120 ∧ totalCost h 5 = 255 → h.flatFee = 75 := by
  sorry

#check hotel_flat_fee_calculation

end NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l118_11871


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l118_11851

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℚ
  intercept : ℚ

def Line.through_point (l : Line) (x y : ℚ) : Prop :=
  y = l.slope * x + l.intercept

def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_through_point (given_line target_line : Line) 
    (h_parallel : given_line.parallel target_line)
    (h_through_point : target_line.through_point 3 0) :
  target_line = Line.mk (1/2) (-3/2) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l118_11851


namespace NUMINAMATH_CALUDE_work_completion_time_l118_11841

theorem work_completion_time (a_time b_time : ℝ) (work_left : ℝ) : 
  a_time = 15 → b_time = 20 → work_left = 0.7666666666666666 →
  (1 / a_time + 1 / b_time) * 2 = 1 - work_left := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l118_11841


namespace NUMINAMATH_CALUDE_base6_to_base10_conversion_l118_11825

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base 6 representation of the number -/
def base6Number : List Nat := [1, 2, 5, 4, 3]

theorem base6_to_base10_conversion :
  base6ToBase10 base6Number = 4945 := by
  sorry

end NUMINAMATH_CALUDE_base6_to_base10_conversion_l118_11825


namespace NUMINAMATH_CALUDE_jills_bus_journey_ratio_l118_11897

/-- Represents the time in minutes for various parts of Jill's bus journey -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- Calculates the ratio of the second bus ride time to the combined wait and trip time of the first bus -/
def bus_time_ratio (journey : BusJourney) : ℚ :=
  journey.second_bus_ride / (journey.first_bus_wait + journey.first_bus_ride)

/-- Theorem stating that for Jill's specific journey, the bus time ratio is 1/2 -/
theorem jills_bus_journey_ratio :
  let journey : BusJourney := { first_bus_wait := 12, first_bus_ride := 30, second_bus_ride := 21 }
  bus_time_ratio journey = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_jills_bus_journey_ratio_l118_11897


namespace NUMINAMATH_CALUDE_soccer_ball_price_proof_l118_11817

/-- The unit price of B type soccer balls -/
def unit_price_B : ℝ := 60

/-- The unit price of A type soccer balls -/
def unit_price_A : ℝ := 2.5 * unit_price_B

/-- The total cost of A type soccer balls -/
def total_cost_A : ℝ := 7500

/-- The total cost of B type soccer balls -/
def total_cost_B : ℝ := 4800

/-- The quantity difference between B and A type soccer balls -/
def quantity_difference : ℕ := 30

theorem soccer_ball_price_proof :
  (total_cost_A / unit_price_A) + quantity_difference = total_cost_B / unit_price_B :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_price_proof_l118_11817


namespace NUMINAMATH_CALUDE_f_properties_l118_11865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2/x - a*x + 5|

theorem f_properties :
  ∀ a : ℝ,
  (∃ x : ℝ, f a x = 0) ∧
  (a = 3 → ∀ x y : ℝ, x < y → y < -1 → f a x > f a y) ∧
  (a > 0 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ f a x₀ = 8/3 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≤ 8/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l118_11865


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l118_11863

/-- A geometric sequence with a_3 = 2 and a_5 = 6 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 3 = 2 ∧ a 5 = 6

theorem geometric_sequence_a9 (a : ℕ → ℝ) 
  (h : geometric_sequence a) : a 9 = 54 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l118_11863


namespace NUMINAMATH_CALUDE_complex_number_with_prime_modulus_exists_l118_11859

theorem complex_number_with_prime_modulus_exists : ∃ (z : ℂ), 
  z^2 = (3 + Complex.I) * z - 24 + 15 * Complex.I ∧ 
  ∃ (p : ℕ), Nat.Prime p ∧ (z.re^2 + z.im^2 : ℝ) = p :=
sorry

end NUMINAMATH_CALUDE_complex_number_with_prime_modulus_exists_l118_11859


namespace NUMINAMATH_CALUDE_traffic_light_combinations_l118_11892

/-- The number of different signals that can be transmitted by k traffic lights -/
def total_signals (k : ℕ) : ℕ := 3^k

/-- Theorem: Given k traffic lights, each capable of transmitting 3 different signals,
    the total number of unique signal combinations is 3^k -/
theorem traffic_light_combinations (k : ℕ) :
  total_signals k = 3^k := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_combinations_l118_11892


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l118_11860

theorem largest_solution_of_equation : 
  let f : ℝ → ℝ := λ b => (3*b + 7)*(b - 2) - 9*b
  let largest_solution : ℝ := (4 + Real.sqrt 58) / 3
  (f largest_solution = 0) ∧ 
  (∀ b : ℝ, f b = 0 → b ≤ largest_solution) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l118_11860


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_nine_l118_11894

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of 9^n is greater than 9 for all n ≥ 3 -/
theorem sum_of_digits_power_of_nine (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_nine_l118_11894


namespace NUMINAMATH_CALUDE_percentage_of_indian_children_l118_11880

theorem percentage_of_indian_children (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℚ) (percent_indian_women : ℚ) (percent_not_indian : ℚ)
  (h1 : total_men = 700)
  (h2 : total_women = 500)
  (h3 : total_children = 800)
  (h4 : percent_indian_men = 20 / 100)
  (h5 : percent_indian_women = 40 / 100)
  (h6 : percent_not_indian = 79 / 100) :
  (((1 - percent_not_indian) * (total_men + total_women + total_children) -
    percent_indian_men * total_men - percent_indian_women * total_women) /
    total_children : ℚ) = 10 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_indian_children_l118_11880


namespace NUMINAMATH_CALUDE_container_volume_ratio_l118_11869

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/5 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 10/9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l118_11869


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l118_11812

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l118_11812


namespace NUMINAMATH_CALUDE_reinforcement_size_problem_l118_11893

/-- Given a garrison with initial men, initial provision days, days before reinforcement,
    and remaining days after reinforcement, calculate the size of the reinforcement. -/
def reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) 
                       (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := initial_men * (initial_days - days_before_reinforcement)
  let total_men_after := remaining_provisions / remaining_days
  total_men_after - initial_men

/-- The size of the reinforcement for the given problem is 1300. -/
theorem reinforcement_size_problem : 
  reinforcement_size 2000 54 21 20 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_problem_l118_11893


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l118_11823

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l118_11823


namespace NUMINAMATH_CALUDE_extreme_value_implies_fourth_quadrant_l118_11842

/-- A function f(x) = x^3 - ax^2 - bx has an extreme value of 10 at x = 1. -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  f 1 = 10

/-- The point (a, b) lies in the fourth quadrant. -/
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b > 0

/-- Theorem: If f(x) = x^3 - ax^2 - bx has an extreme value of 10 at x = 1,
    then the point (a, b) lies in the fourth quadrant. -/
theorem extreme_value_implies_fourth_quadrant (a b : ℝ) :
  has_extreme_value a b → in_fourth_quadrant a b :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_fourth_quadrant_l118_11842


namespace NUMINAMATH_CALUDE_lenkas_numbers_l118_11878

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def both_digits_even (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 0 ∧ (n / 10) % 2 = 0

def both_digits_odd (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 1 ∧ (n / 10) % 2 = 1

def sum_has_even_odd_digits (n : ℕ) : Prop :=
  is_two_digit n ∧ (n / 10) % 2 = 0 ∧ n % 2 = 1

theorem lenkas_numbers :
  ∀ a b : ℕ,
    both_digits_even a →
    both_digits_odd b →
    sum_has_even_odd_digits (a + b) →
    a % 3 = 0 →
    b % 3 = 0 →
    (a % 10 = 9 ∨ b % 10 = 9 ∨ (a + b) % 10 = 9) →
    ((a = 24 ∧ b = 39) ∨ (a = 42 ∧ b = 39) ∨ (a = 48 ∧ b = 39)) :=
by sorry

end NUMINAMATH_CALUDE_lenkas_numbers_l118_11878


namespace NUMINAMATH_CALUDE_train_passing_platform_l118_11824

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1500) 
  (h2 : tree_crossing_time = 120) 
  (h3 : platform_length = 500) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 160 := by
  sorry

#check train_passing_platform

end NUMINAMATH_CALUDE_train_passing_platform_l118_11824


namespace NUMINAMATH_CALUDE_chocos_remainder_l118_11867

theorem chocos_remainder (n : ℕ) (h : n % 11 = 5) : (4 * n) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chocos_remainder_l118_11867


namespace NUMINAMATH_CALUDE_third_number_proof_l118_11864

theorem third_number_proof (x : ℝ) : 0.3 * 0.8 + x * 0.5 = 0.29 → x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l118_11864


namespace NUMINAMATH_CALUDE_complex_modulus_constraint_l118_11830

theorem complex_modulus_constraint (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a - Real.cos θ) + (a - 1 - Real.sin θ) * Complex.I) ≤ 2) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_constraint_l118_11830


namespace NUMINAMATH_CALUDE_x_completion_time_l118_11877

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  is_positive : days > 0

/-- Represents a worker who can complete a work in a given time -/
structure Worker where
  time_to_complete : WorkTime

/-- The work scenario -/
structure WorkScenario where
  x : Worker
  y : Worker
  x_partial_work : WorkTime
  y_completion_after_x : WorkTime
  y_solo_completion : WorkTime
  work_continuity : x_partial_work.days + y_completion_after_x.days = y_solo_completion.days

/-- The theorem stating that x takes 40 days to complete the work -/
theorem x_completion_time (scenario : WorkScenario) 
  (h1 : scenario.x_partial_work.days = 8)
  (h2 : scenario.y_completion_after_x.days = 16)
  (h3 : scenario.y_solo_completion.days = 20) :
  scenario.x.time_to_complete.days = 40 := by
  sorry


end NUMINAMATH_CALUDE_x_completion_time_l118_11877


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l118_11895

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem seventh_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : sum_of_arithmetic_sequence a 13 = 39) : 
  a 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l118_11895


namespace NUMINAMATH_CALUDE_parallel_transitivity_l118_11898

-- Define a type for lines in a plane
structure Line where
  -- You can add more specific properties here if needed
  mk :: 

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  -- The definition of parallel lines
  sorry

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) : 
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l118_11898


namespace NUMINAMATH_CALUDE_equal_distribution_of_cards_l118_11873

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_cards_l118_11873


namespace NUMINAMATH_CALUDE_orchard_trees_l118_11807

theorem orchard_trees (total : ℕ) (peach : ℕ) (pear : ℕ) 
  (h1 : total = 480) 
  (h2 : pear = 3 * peach) 
  (h3 : total = peach + pear) : 
  peach = 120 ∧ pear = 360 := by
  sorry

end NUMINAMATH_CALUDE_orchard_trees_l118_11807


namespace NUMINAMATH_CALUDE_triangle_formation_l118_11886

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 3 5) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 1 1 3) ∧
  ¬(can_form_triangle 3 4 9) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l118_11886


namespace NUMINAMATH_CALUDE_equation_system_solution_l118_11874

theorem equation_system_solution : 
  ∀ (x y z : ℝ), 
    z ≠ 0 →
    3 * x - 4 * y - 2 * z = 0 →
    x - 2 * y + 5 * z = 0 →
    (2 * x^2 - x * y) / (y^2 + 4 * z^2) = 744 / 305 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l118_11874


namespace NUMINAMATH_CALUDE_poker_hand_probabilities_l118_11876

-- Define the total number of possible 5-card hands
def total_hands : ℕ := 2598960

-- Define the number of ways to get each hand type
def pair_ways : ℕ := 1098240
def two_pair_ways : ℕ := 123552
def three_of_a_kind_ways : ℕ := 54912
def straight_ways : ℕ := 10000
def flush_ways : ℕ := 5108
def full_house_ways : ℕ := 3744
def four_of_a_kind_ways : ℕ := 624
def straight_flush_ways : ℕ := 40

-- Define the probability of each hand type
def prob_pair : ℚ := pair_ways / total_hands
def prob_two_pair : ℚ := two_pair_ways / total_hands
def prob_three_of_a_kind : ℚ := three_of_a_kind_ways / total_hands
def prob_straight : ℚ := straight_ways / total_hands
def prob_flush : ℚ := flush_ways / total_hands
def prob_full_house : ℚ := full_house_ways / total_hands
def prob_four_of_a_kind : ℚ := four_of_a_kind_ways / total_hands
def prob_straight_flush : ℚ := straight_flush_ways / total_hands

-- Theorem stating the probabilities of different poker hands
theorem poker_hand_probabilities :
  (prob_pair = 1098240 / 2598960) ∧
  (prob_two_pair = 123552 / 2598960) ∧
  (prob_three_of_a_kind = 54912 / 2598960) ∧
  (prob_straight = 10000 / 2598960) ∧
  (prob_flush = 5108 / 2598960) ∧
  (prob_full_house = 3744 / 2598960) ∧
  (prob_four_of_a_kind = 624 / 2598960) ∧
  (prob_straight_flush = 40 / 2598960) :=
by sorry

end NUMINAMATH_CALUDE_poker_hand_probabilities_l118_11876


namespace NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l118_11875

/-- Represents the cost of groceries -/
structure GroceryCost where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The total cost of all items is $40 -/
def total_cost (g : GroceryCost) : Prop :=
  g.apples + g.bananas + g.cantaloupe + g.dates = 40

/-- A carton of dates costs three times as much as a sack of apples -/
def dates_cost (g : GroceryCost) : Prop :=
  g.dates = 3 * g.apples

/-- The price of a cantaloupe is equal to half the sum of the price of a sack of apples and a bunch of bananas -/
def cantaloupe_cost (g : GroceryCost) : Prop :=
  g.cantaloupe = (g.apples + g.bananas) / 2

/-- The main theorem: Given the conditions, the cost of a bunch of bananas and a cantaloupe is $8 -/
theorem bananas_cantaloupe_cost (g : GroceryCost) 
  (h1 : total_cost g) 
  (h2 : dates_cost g) 
  (h3 : cantaloupe_cost g) : 
  g.bananas + g.cantaloupe = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l118_11875


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l118_11819

theorem trapezoid_side_length (square_side : ℝ) (trapezoid_area hexagon_area : ℝ) 
  (x : ℝ) : 
  square_side = 1 →
  trapezoid_area = hexagon_area →
  trapezoid_area = 1/4 →
  x = trapezoid_area * 4 / (1 + square_side) →
  x = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l118_11819


namespace NUMINAMATH_CALUDE_recycling_theorem_l118_11849

def recycle (n : ℕ) : ℕ :=
  if n < 5 then 0 else n / 5 + recycle (n / 5)

theorem recycling_theorem :
  recycle 3125 = 781 :=
by
  sorry

end NUMINAMATH_CALUDE_recycling_theorem_l118_11849


namespace NUMINAMATH_CALUDE_gcd_18_30_l118_11818

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l118_11818


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l118_11832

/-- A function f(x) = x³ + ax - 2 that is increasing on (1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x - 2

/-- The derivative of f(x) -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x > 1, ∀ y > x, f a y > f a x) ↔ a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l118_11832


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l118_11889

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 1, 1, 2), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 0, 0)}

theorem quadruple_equation_solutions :
  ∀ x y z n : ℕ, is_solution x y z n ↔ (x, y, z, n) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l118_11889


namespace NUMINAMATH_CALUDE_distance_between_points_l118_11829

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 0)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l118_11829


namespace NUMINAMATH_CALUDE_least_prime_factor_of_7_4_minus_7_3_l118_11815

theorem least_prime_factor_of_7_4_minus_7_3 :
  Nat.minFac (7^4 - 7^3) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_7_4_minus_7_3_l118_11815


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l118_11844

theorem subtracted_value_proof (x : ℕ) (h : x = 124) :
  ∃! y : ℕ, 2 * x - y = 110 :=
sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l118_11844


namespace NUMINAMATH_CALUDE_find_number_l118_11855

theorem find_number (A B : ℕ) (hA : A > 0) (hB : B > 0) : 
  Nat.gcd A B = 15 → Nat.lcm A B = 312 → B = 195 → A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l118_11855


namespace NUMINAMATH_CALUDE_expression_evaluation_l118_11811

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l118_11811


namespace NUMINAMATH_CALUDE_jo_bob_balloon_ride_l118_11806

/-- The problem of Jo-Bob's hot air balloon ride -/
theorem jo_bob_balloon_ride (rise_rate : ℝ) (fall_rate : ℝ) (second_pull_time : ℝ) 
  (fall_time : ℝ) (max_height : ℝ) :
  rise_rate = 50 →
  fall_rate = 10 →
  second_pull_time = 15 →
  fall_time = 10 →
  max_height = 1400 →
  ∃ (first_pull_time : ℝ),
    first_pull_time * rise_rate - fall_time * fall_rate + second_pull_time * rise_rate = max_height ∧
    first_pull_time = 15 := by
  sorry

#check jo_bob_balloon_ride

end NUMINAMATH_CALUDE_jo_bob_balloon_ride_l118_11806


namespace NUMINAMATH_CALUDE_hotel_room_charges_l118_11853

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.4))
  (h2 : P = G * (1 - 0.1)) :
  R = G * 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l118_11853


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l118_11872

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : ∀ n : ℕ+, (a.val^n.val + n.val) ∣ (b.val^n.val + n.val)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l118_11872


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l118_11833

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (3/2 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l118_11833


namespace NUMINAMATH_CALUDE_fountain_position_l118_11847

/-- Two towers with a fountain between them -/
structure TowerSetup where
  tower1_height : ℝ
  tower2_height : ℝ
  distance_between_towers : ℝ
  fountain_distance : ℝ

/-- The setup satisfies the problem conditions -/
def valid_setup (s : TowerSetup) : Prop :=
  s.tower1_height = 30 ∧
  s.tower2_height = 40 ∧
  s.distance_between_towers = 50 ∧
  0 < s.fountain_distance ∧
  s.fountain_distance < s.distance_between_towers

/-- The birds' flight paths are equal -/
def equal_flight_paths (s : TowerSetup) : Prop :=
  s.tower1_height^2 + s.fountain_distance^2 =
  s.tower2_height^2 + (s.distance_between_towers - s.fountain_distance)^2

theorem fountain_position (s : TowerSetup) 
  (h1 : valid_setup s) (h2 : equal_flight_paths s) :
  s.fountain_distance = 32 ∧ 
  s.distance_between_towers - s.fountain_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_fountain_position_l118_11847


namespace NUMINAMATH_CALUDE_ball_probabilities_l118_11887

/-- Represents the box of balls -/
structure BallBox where
  red_balls : ℕ
  white_balls : ℕ

/-- The probability of drawing exactly one red ball and one white ball without replacement -/
def prob_one_red_one_white (box : BallBox) : ℚ :=
  let total := box.red_balls + box.white_balls
  (box.red_balls : ℚ) / total * (box.white_balls : ℚ) / (total - 1) +
  (box.white_balls : ℚ) / total * (box.red_balls : ℚ) / (total - 1)

/-- The probability of getting at least one red ball in three draws with replacement -/
def prob_at_least_one_red (box : BallBox) : ℚ :=
  let p_red := (box.red_balls : ℚ) / (box.red_balls + box.white_balls)
  1 - (1 - p_red) ^ 3

theorem ball_probabilities (box : BallBox) (h1 : box.red_balls = 2) (h2 : box.white_balls = 4) :
  prob_one_red_one_white box = 8/15 ∧ prob_at_least_one_red box = 19/27 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l118_11887


namespace NUMINAMATH_CALUDE_jewels_total_gain_l118_11805

/-- Represents the problem of calculating Jewel's total gain from selling magazines --/
def jewels_magazines_problem (cheap_magazines : ℕ) (expensive_magazines : ℕ) 
  (cheap_buy_price : ℚ) (expensive_buy_price : ℚ)
  (cheap_sell_price : ℚ) (expensive_sell_price : ℚ)
  (cheap_discount_percent : ℚ) (expensive_discount_percent : ℚ)
  (cheap_discount_on : ℕ) (expensive_discount_on : ℕ) : Prop :=
let total_cost := cheap_magazines * cheap_buy_price + expensive_magazines * expensive_buy_price
let total_sell := cheap_magazines * cheap_sell_price + expensive_magazines * expensive_sell_price
let cheap_discount := cheap_sell_price * cheap_discount_percent
let expensive_discount := expensive_sell_price * expensive_discount_percent
let total_discount := cheap_discount + expensive_discount
let total_gain := total_sell - total_discount - total_cost
total_gain = 5.1875

/-- Theorem stating that Jewel's total gain is $5.1875 under the given conditions --/
theorem jewels_total_gain :
  jewels_magazines_problem 5 5 3 4 3.5 4.75 0.1 0.15 2 4 := by
  sorry

end NUMINAMATH_CALUDE_jewels_total_gain_l118_11805


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l118_11809

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 2000000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 2
  exponent := 6
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct : 
  (scientific_repr.coefficient * (10 ^ scientific_repr.exponent : ℝ)) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l118_11809


namespace NUMINAMATH_CALUDE_unique_solution_condition_l118_11883

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℚ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l118_11883


namespace NUMINAMATH_CALUDE_toothpicks_250th_stage_l118_11834

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  arithmeticSequence 4 3 n

theorem toothpicks_250th_stage :
  toothpicks 250 = 751 := by sorry

end NUMINAMATH_CALUDE_toothpicks_250th_stage_l118_11834


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l118_11885

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l118_11885


namespace NUMINAMATH_CALUDE_parallelogram_sticks_l118_11843

/-- A parallelogram formed by four sticks -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  is_parallelogram : side1 = side3 ∧ side2 = side4

/-- The theorem stating that if four sticks of lengths 5, 5, 7, and a can form a parallelogram, then a = 7 -/
theorem parallelogram_sticks (a : ℝ) :
  (∃ p : Parallelogram, p.side1 = 5 ∧ p.side2 = 5 ∧ p.side3 = 7 ∧ p.side4 = a) →
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sticks_l118_11843


namespace NUMINAMATH_CALUDE_floor_expression_equals_twelve_l118_11862

theorem floor_expression_equals_twelve (n : ℕ) (h : n = 1006) : 
  ⌊((n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1)) + 5 : ℝ)⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_twelve_l118_11862


namespace NUMINAMATH_CALUDE_cube_side_length_l118_11861

theorem cube_side_length (surface_area : ℝ) (side_length : ℝ) : 
  surface_area = 600 → 
  6 * side_length^2 = surface_area → 
  side_length = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_side_length_l118_11861


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l118_11881

theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let new_area := (l - 5) * (w + 5)
  new_area = l * w + 75 → l = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l118_11881


namespace NUMINAMATH_CALUDE_square_difference_divided_by_ten_l118_11838

theorem square_difference_divided_by_ten : (305^2 - 295^2) / 10 = 600 := by sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_ten_l118_11838


namespace NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l118_11814

/-- A parabola defined by y² = 4x passing through (1, 2) -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  passes_through : eq 1 2

/-- A line intersecting the parabola at two points -/
structure IntersectingLine (p : Parabola) where
  slope : ℝ
  y_intercept : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    p.eq x₁ y₁ ∧ p.eq x₂ y₂ ∧
    x₁ = slope * y₁ + y_intercept ∧
    x₂ = slope * y₂ + y_intercept ∧
    y₁ * y₂ = -4

/-- The theorem to be proved -/
theorem intersecting_line_passes_through_fixed_point (p : Parabola) (l : IntersectingLine p) :
  ∃ (x y : ℝ), x = l.slope * y + l.y_intercept ∧ x = 1 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l118_11814


namespace NUMINAMATH_CALUDE_parabola_properties_l118_11816

theorem parabola_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : a * (-3)^2 + b * (-3) + c = 0) 
  (h3 : a * 1^2 + b * 1 + c = 0) : 
  (b^2 - 4*a*c > 0) ∧ (3*b + 2*c = 0) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l118_11816


namespace NUMINAMATH_CALUDE_weight_of_new_person_l118_11802

theorem weight_of_new_person (initial_weight : ℝ) (weight_increase : ℝ) :
  initial_weight = 65 →
  weight_increase = 4.5 →
  ∃ (new_weight : ℝ), new_weight = initial_weight + 2 * weight_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l118_11802


namespace NUMINAMATH_CALUDE_missing_number_proof_l118_11826

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + x * 4 / 3) = 13 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l118_11826


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l118_11854

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  d^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l118_11854


namespace NUMINAMATH_CALUDE_samuel_has_five_birds_l118_11899

/-- The number of berries a single bird eats per day -/
def berries_per_bird_per_day : ℕ := 7

/-- The total number of berries eaten by all birds in 4 days -/
def total_berries_in_four_days : ℕ := 140

/-- The number of days over which the total berries are consumed -/
def days : ℕ := 4

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := total_berries_in_four_days / (days * berries_per_bird_per_day)

theorem samuel_has_five_birds : samuels_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_samuel_has_five_birds_l118_11899


namespace NUMINAMATH_CALUDE_this_year_sales_calculation_l118_11866

def last_year_sales : ℝ := 320
def percent_increase : ℝ := 0.25

theorem this_year_sales_calculation :
  last_year_sales * (1 + percent_increase) = 400 := by
  sorry

end NUMINAMATH_CALUDE_this_year_sales_calculation_l118_11866


namespace NUMINAMATH_CALUDE_parking_ticket_multiple_l118_11870

theorem parking_ticket_multiple (total_tickets : ℕ) (alan_tickets : ℕ) (marcy_tickets : ℕ) (m : ℕ) :
  total_tickets = 150 →
  alan_tickets = 26 →
  marcy_tickets = m * alan_tickets - 6 →
  total_tickets = alan_tickets + marcy_tickets →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_parking_ticket_multiple_l118_11870


namespace NUMINAMATH_CALUDE_mass_of_man_is_180_l118_11890

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating that the mass of the man is 180 kg under the given conditions. -/
theorem mass_of_man_is_180 :
  let boat_length : ℝ := 6
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 180 := by
  sorry

#eval mass_of_man 6 3 0.01 1000

end NUMINAMATH_CALUDE_mass_of_man_is_180_l118_11890


namespace NUMINAMATH_CALUDE_exist_six_points_similar_triangles_l118_11810

/-- A point in a plane represented by its coordinates -/
structure Point (α : Type*) where
  x : α
  y : α

/-- A triangle represented by its three vertices -/
structure Triangle (α : Type*) where
  A : Point α
  B : Point α
  C : Point α

/-- Predicate to check if two triangles are similar -/
def similar {α : Type*} (t1 t2 : Triangle α) : Prop :=
  sorry

/-- Theorem stating the existence of six points forming similar triangles -/
theorem exist_six_points_similar_triangles :
  ∃ (X₁ X₂ Y₁ Y₂ Z₁ Z₂ : Point ℝ),
    ∀ (i j k : Fin 2),
      similar
        (Triangle.mk (if i = 0 then X₁ else X₂) (if j = 0 then Y₁ else Y₂) (if k = 0 then Z₁ else Z₂))
        (Triangle.mk X₁ Y₁ Z₁) :=
  sorry

end NUMINAMATH_CALUDE_exist_six_points_similar_triangles_l118_11810


namespace NUMINAMATH_CALUDE_tablespoons_in_half_cup_l118_11840

/-- Proves that there are 8 tablespoons in half a cup of rice -/
theorem tablespoons_in_half_cup (grains_per_cup : ℕ) (teaspoons_per_tablespoon : ℕ) (grains_per_teaspoon : ℕ)
  (h1 : grains_per_cup = 480)
  (h2 : teaspoons_per_tablespoon = 3)
  (h3 : grains_per_teaspoon = 10) :
  (grains_per_cup / 2) / (grains_per_teaspoon * teaspoons_per_tablespoon) = 8 := by
  sorry

#check tablespoons_in_half_cup

end NUMINAMATH_CALUDE_tablespoons_in_half_cup_l118_11840


namespace NUMINAMATH_CALUDE_complex_real_condition_l118_11856

theorem complex_real_condition (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 5*m + 6 : ℝ) + (m - 3 : ℝ)*I ∧ z.im = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l118_11856


namespace NUMINAMATH_CALUDE_difference_of_squares_representation_l118_11835

theorem difference_of_squares_representation (n : ℕ) : 
  n = 2^4035 → 
  (∃ (count : ℕ), count = 2018 ∧ 
    (∃ (S : Finset (ℕ × ℕ)), 
      S.card = count ∧
      ∀ (pair : ℕ × ℕ), pair ∈ S ↔ 
        (∃ (a b : ℕ), pair = (a, b) ∧ n = a^2 - b^2))) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_representation_l118_11835


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l118_11846

-- Define the triangle XYZ
def triangle_XYZ (X Y Z : ℝ × ℝ) : Prop :=
  let d := λ a b : ℝ × ℝ => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  d X Z = 13 ∧ d X Y = 12 ∧ (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem sum_of_square_areas (X Y Z : ℝ × ℝ) (h : triangle_XYZ X Y Z) :
  (13 : ℝ)^2 + (Real.sqrt ((13 : ℝ)^2 - 12^2))^2 = 194 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_square_areas_l118_11846


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l118_11848

def fraction := 800 / (2^5 * 5^11)

def count_nonzero_decimal_digits (x : ℚ) : ℕ := sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l118_11848


namespace NUMINAMATH_CALUDE_sum_exradii_equals_four_circumradius_plus_inradius_l118_11850

/-- Given a triangle with exradii r_a, r_b, r_c, circumradius R, and inradius r,
    prove that the sum of the exradii equals four times the circumradius plus the inradius. -/
theorem sum_exradii_equals_four_circumradius_plus_inradius 
  (r_a r_b r_c R r : ℝ) :
  r_a > 0 → r_b > 0 → r_c > 0 → R > 0 → r > 0 →
  r_a + r_b + r_c = 4 * R + r := by
  sorry

end NUMINAMATH_CALUDE_sum_exradii_equals_four_circumradius_plus_inradius_l118_11850


namespace NUMINAMATH_CALUDE_sum_of_integers_l118_11800

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 128) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l118_11800


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l118_11803

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1 + Complex.I) / 2) :
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l118_11803


namespace NUMINAMATH_CALUDE_rooster_on_roof_no_egg_falls_l118_11822

/-- Represents a bird species -/
inductive BirdSpecies
  | Rooster
  | Hen

/-- Represents the ability to lay eggs -/
def canLayEggs (species : BirdSpecies) : Prop :=
  match species with
  | BirdSpecies.Rooster => False
  | BirdSpecies.Hen => True

/-- Represents a roof with two slopes -/
structure Roof :=
  (slope1 : ℝ)
  (slope2 : ℝ)

/-- Theorem: Given a roof with two slopes and a rooster on the ridge, no egg will fall -/
theorem rooster_on_roof_no_egg_falls (roof : Roof) (bird : BirdSpecies) :
  roof.slope1 = 60 → roof.slope2 = 70 → bird = BirdSpecies.Rooster → ¬(canLayEggs bird) :=
by sorry

end NUMINAMATH_CALUDE_rooster_on_roof_no_egg_falls_l118_11822


namespace NUMINAMATH_CALUDE_job_completion_time_l118_11828

/-- The number of days it takes for two workers to complete a job together,
    given their individual work rates. -/
def days_to_complete (rate_a rate_b : ℚ) : ℚ :=
  1 / (rate_a + rate_b)

theorem job_completion_time 
  (rate_a rate_b : ℚ) 
  (h1 : rate_a = rate_b) 
  (h2 : rate_b = 1 / 12) : 
  days_to_complete rate_a rate_b = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l118_11828


namespace NUMINAMATH_CALUDE_total_holiday_savings_l118_11888

def holiday_savings (sam_savings victory_savings : ℕ) : ℕ :=
  sam_savings + victory_savings

theorem total_holiday_savings : 
  ∀ (sam_savings victory_savings : ℕ),
    sam_savings = 1000 →
    victory_savings = sam_savings - 100 →
    holiday_savings sam_savings victory_savings = 1900 :=
by
  sorry

end NUMINAMATH_CALUDE_total_holiday_savings_l118_11888


namespace NUMINAMATH_CALUDE_patio_rearrangement_l118_11882

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ

/-- Defines the conditions for a valid patio layout --/
def is_valid_layout (layout : PatioLayout) : Prop :=
  layout.total_tiles = layout.rows * layout.columns

/-- Defines the rearrangement of the patio --/
def rearranged_layout (original : PatioLayout) : PatioLayout :=
  { rows := original.total_tiles / (original.columns - 2)
  , columns := original.columns - 2
  , total_tiles := original.total_tiles }

/-- The main theorem to prove --/
theorem patio_rearrangement 
  (original : PatioLayout)
  (h_valid : is_valid_layout original)
  (h_rows : original.rows = 6)
  (h_total : original.total_tiles = 48) :
  (rearranged_layout original).rows - original.rows = 2 :=
sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l118_11882


namespace NUMINAMATH_CALUDE_set_forms_triangle_l118_11801

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (1, 2, 2) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 1 2 2 := by
  sorry

end NUMINAMATH_CALUDE_set_forms_triangle_l118_11801


namespace NUMINAMATH_CALUDE_scientific_notation_of_41600_l118_11804

theorem scientific_notation_of_41600 :
  ∃ (a : ℝ) (n : ℤ), 41600 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.16 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_41600_l118_11804


namespace NUMINAMATH_CALUDE_no_geometric_mean_opposite_signs_l118_11836

/-- The geometric mean of two real numbers does not exist if they have opposite signs -/
theorem no_geometric_mean_opposite_signs (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ¬∃ (x : ℝ), x^2 = a * b :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_mean_opposite_signs_l118_11836


namespace NUMINAMATH_CALUDE_complex_equation_solution_l118_11813

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + (3 - 2*I)*z = 2 + 5*I*z ∧ z = -(9/58) - (21/58)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l118_11813


namespace NUMINAMATH_CALUDE_last_two_digits_of_expression_l118_11857

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_expression : 
  last_two_digits (sum_of_factorials 15 - factorial 5 * factorial 10 * factorial 15) = 13 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_expression_l118_11857


namespace NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_l118_11868

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (2 * x) + 4 / x ≥ 6 ∧
  (2 * Real.sqrt (2 * x) + 4 / x = 6 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_l118_11868


namespace NUMINAMATH_CALUDE_tuesday_poodles_count_l118_11879

/-- Represents the number of hours Charlotte can walk dogs on a weekday -/
def weekday_hours : ℕ := 8

/-- Represents the number of hours Charlotte can walk dogs on a weekend day -/
def weekend_hours : ℕ := 4

/-- Represents the number of weekdays in a week -/
def weekdays : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- Represents the time it takes to walk a poodle -/
def poodle_time : ℕ := 2

/-- Represents the time it takes to walk a Chihuahua -/
def chihuahua_time : ℕ := 1

/-- Represents the time it takes to walk a Labrador -/
def labrador_time : ℕ := 3

/-- Represents the time it takes to walk a Golden Retriever -/
def golden_retriever_time : ℕ := 4

/-- Represents the number of poodles walked on Monday -/
def monday_poodles : ℕ := 4

/-- Represents the number of Chihuahuas walked on Monday and Tuesday -/
def monday_tuesday_chihuahuas : ℕ := 2

/-- Represents the number of Golden Retrievers walked on Monday -/
def monday_golden_retrievers : ℕ := 1

/-- Represents the number of Labradors walked on Wednesday -/
def wednesday_labradors : ℕ := 4

/-- Represents the number of Golden Retrievers walked on Tuesday -/
def tuesday_golden_retrievers : ℕ := 1

theorem tuesday_poodles_count :
  ∃ (tuesday_poodles : ℕ),
    tuesday_poodles = 1 ∧
    weekday_hours * weekdays + weekend_hours * weekend_days ≥
      (monday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       monday_golden_retrievers * golden_retriever_time) +
      (tuesday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       tuesday_golden_retrievers * golden_retriever_time) +
      (wednesday_labradors * labrador_time) :=
by sorry

end NUMINAMATH_CALUDE_tuesday_poodles_count_l118_11879


namespace NUMINAMATH_CALUDE_rogers_remaining_years_l118_11845

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the coworkers' experience -/
def valid_experience (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- Roger's retirement years -/
def retirement_years : ℕ := 50

/-- Theorem stating that Roger needs to work 8 more years before retirement -/
theorem rogers_remaining_years (e : Experience) (h : valid_experience e) :
  retirement_years - e.roger = 8 := by
  sorry


end NUMINAMATH_CALUDE_rogers_remaining_years_l118_11845


namespace NUMINAMATH_CALUDE_centroid_curve_area_centroid_curve_area_for_diameter_30_l118_11884

/-- The area of the region bounded by the curve traced by the centroid of a triangle,
    where two vertices of the triangle are the endpoints of a circle's diameter,
    and the third vertex moves along the circle's circumference. -/
theorem centroid_curve_area (diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  let centroid_radius := radius / 3
  let area := Real.pi * centroid_radius ^ 2
  ⌊area + 0.5⌋

/-- The area of the region bounded by the curve traced by the centroid of triangle ABC,
    where AB is a diameter of a circle with length 30 and C is a point on the circle,
    is approximately 79 (to the nearest positive integer). -/
theorem centroid_curve_area_for_diameter_30 :
  centroid_curve_area 30 = 79 := by
  sorry

end NUMINAMATH_CALUDE_centroid_curve_area_centroid_curve_area_for_diameter_30_l118_11884


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l118_11821

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l118_11821


namespace NUMINAMATH_CALUDE_origin_and_slope_condition_vertical_tangent_condition_l118_11827

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + (1 - a)*x^2 - a*(a + 2)*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*(1 - a)*x - a*(a + 2)

-- Theorem 1: If f(0) = 0 and f'(0) = -3, then (a = -3 or a = 1) and b = 0
theorem origin_and_slope_condition (a b : ℝ) :
  f a b 0 = 0 ∧ f' a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0 := by sorry

-- Theorem 2: The curve y = f(x) has two vertical tangent lines iff a ∈ (-∞, -1/2) ∪ (-1/2, +∞)
theorem vertical_tangent_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) ↔ 
  a < -1/2 ∨ a > -1/2 := by sorry

end NUMINAMATH_CALUDE_origin_and_slope_condition_vertical_tangent_condition_l118_11827


namespace NUMINAMATH_CALUDE_stream_speed_l118_11891

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream. -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 15)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l118_11891


namespace NUMINAMATH_CALUDE_polynomial_expansion_l118_11837

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 2*x - 5)*(x - 2) - (x - 2)*(x^2 - 5*x + 28) + (4*x - 7)*(x - 2)*(x + 4) =
  6*x^3 + 4*x^2 - 93*x + 122 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l118_11837


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_3_6_l118_11839

theorem gcf_lcm_sum_3_6 : Nat.gcd 3 6 + Nat.lcm 3 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_3_6_l118_11839


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l118_11896

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 78
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l118_11896


namespace NUMINAMATH_CALUDE_prime_between_40_and_50_and_largest_below_100_l118_11820

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_between_40_and_50_and_largest_below_100 :
  (∀ p : ℕ, 40 < p ∧ p < 50 ∧ isPrime p ↔ p = 41 ∨ p = 43 ∨ p = 47) ∧
  (∀ q : ℕ, q < 100 ∧ isPrime q → q ≤ 97) ∧
  isPrime 97 :=
sorry

end NUMINAMATH_CALUDE_prime_between_40_and_50_and_largest_below_100_l118_11820


namespace NUMINAMATH_CALUDE_basketball_club_girls_l118_11852

theorem basketball_club_girls (total_members : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_members = 30 →
  attendance = 18 →
  boys + girls = total_members →
  boys + (1/3 : ℚ) * girls = attendance →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_club_girls_l118_11852
