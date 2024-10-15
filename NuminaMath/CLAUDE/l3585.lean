import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3585_358535

theorem rhombus_perimeter (d : ℝ) (h1 : d = 20) : 
  let longer_diagonal := 1.3 * d
  let side := Real.sqrt ((d/2)^2 + (longer_diagonal/2)^2)
  4 * side = 4 * Real.sqrt 269 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3585_358535


namespace NUMINAMATH_CALUDE_chef_lunch_meals_l3585_358578

theorem chef_lunch_meals (meals_sold_lunch : ℕ) (meals_prepared_dinner : ℕ) (total_dinner_meals : ℕ)
  (h1 : meals_sold_lunch = 12)
  (h2 : meals_prepared_dinner = 5)
  (h3 : total_dinner_meals = 10) :
  meals_sold_lunch + (total_dinner_meals - meals_prepared_dinner) = 17 :=
by sorry

end NUMINAMATH_CALUDE_chef_lunch_meals_l3585_358578


namespace NUMINAMATH_CALUDE_sum_of_zero_seven_representable_l3585_358555

/-- A function that checks if a real number can be written using only 0 and 7 in decimal notation -/
def uses_only_zero_and_seven (x : ℝ) : Prop :=
  ∃ (digits : ℕ → ℕ), (∀ n, digits n ∈ ({0, 7} : Set ℕ)) ∧
    x = ∑' n, (digits n : ℝ) / 10^n

/-- Theorem stating that any positive real number can be represented as the sum of nine numbers,
    each of which in decimal notation consists of the digits 0 and 7 -/
theorem sum_of_zero_seven_representable (x : ℝ) (hx : 0 < x) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ),
    x = a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ ∧
    (uses_only_zero_and_seven a₁) ∧
    (uses_only_zero_and_seven a₂) ∧
    (uses_only_zero_and_seven a₃) ∧
    (uses_only_zero_and_seven a₄) ∧
    (uses_only_zero_and_seven a₅) ∧
    (uses_only_zero_and_seven a₆) ∧
    (uses_only_zero_and_seven a₇) ∧
    (uses_only_zero_and_seven a₈) ∧
    (uses_only_zero_and_seven a₉) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_zero_seven_representable_l3585_358555


namespace NUMINAMATH_CALUDE_room_length_from_carpet_cost_room_length_is_208_l3585_358505

/-- The length of a room given carpet and cost information -/
theorem room_length_from_carpet_cost (room_width : ℝ) (carpet_width : ℝ) 
  (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) : ℝ :=
  let total_area := total_cost / carpet_cost_per_sqm
  let carpet_width_m := carpet_width / 100
  total_area / carpet_width_m

/-- Proof that the room length is 208 meters given specific conditions -/
theorem room_length_is_208 :
  room_length_from_carpet_cost 9 75 12 1872 = 208 := by
  sorry

end NUMINAMATH_CALUDE_room_length_from_carpet_cost_room_length_is_208_l3585_358505


namespace NUMINAMATH_CALUDE_prob_three_draws_equals_36_125_l3585_358565

/-- The probability of drawing exactly 3 balls to get two red balls -/
def prob_three_draws (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  2 * (p_red * p_white * p_red)

/-- The box contains 3 red balls and 2 white balls -/
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

theorem prob_three_draws_equals_36_125 :
  prob_three_draws total_balls red_balls white_balls = 36 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_draws_equals_36_125_l3585_358565


namespace NUMINAMATH_CALUDE_percentage_calculation_l3585_358593

theorem percentage_calculation (N P : ℝ) (h1 : N = 75) (h2 : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3585_358593


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3585_358507

/-- Proves that the speed of a boat in still water is 30 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 7)
  (h2 : downstream_distance = 22.2)
  (h3 : downstream_time = 0.6) :
  ∃ (boat_speed : ℝ), 
    boat_speed = 30 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3585_358507


namespace NUMINAMATH_CALUDE_complex_magnitude_l3585_358584

theorem complex_magnitude (z : ℂ) (h : z / (2 - Complex.I) = 2 * Complex.I) : 
  Complex.abs (z + 1) = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3585_358584


namespace NUMINAMATH_CALUDE_sqrt_2023_bound_l3585_358514

theorem sqrt_2023_bound (n : ℤ) 
  (h1 : 43^2 = 1849)
  (h2 : 44^2 = 1936)
  (h3 : 45^2 = 2025)
  (h4 : 46^2 = 2116)
  (h5 : n < Real.sqrt 2023)
  (h6 : Real.sqrt 2023 < n + 1) : 
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2023_bound_l3585_358514


namespace NUMINAMATH_CALUDE_first_group_size_is_20_l3585_358580

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the water fountain built by the first group -/
def first_fountain_length : ℝ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 7

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the water fountain built by the second group -/
def second_fountain_length : ℝ := 42

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- The theorem stating that the first group size is 20 men -/
theorem first_group_size_is_20 :
  first_group_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_20_l3585_358580


namespace NUMINAMATH_CALUDE_h_transformation_l3585_358567

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the transformation h
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x + 3

-- Theorem statement
theorem h_transformation (f : ℝ → ℝ) (x : ℝ) : 
  h f x = 2 * f x + 3 := by
  sorry

end NUMINAMATH_CALUDE_h_transformation_l3585_358567


namespace NUMINAMATH_CALUDE_basketball_team_squads_l3585_358501

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of different team squads that can be formed from a given number of players,
    selecting a captain and a specified number of additional players. -/
def teamSquads (totalPlayers captains additionalPlayers : ℕ) : ℕ :=
  totalPlayers * binomial (totalPlayers - 1) additionalPlayers

theorem basketball_team_squads :
  teamSquads 12 1 5 = 5544 := by sorry

end NUMINAMATH_CALUDE_basketball_team_squads_l3585_358501


namespace NUMINAMATH_CALUDE_tower_surface_area_l3585_358549

/-- Represents a layer in the tower -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the tower of cubes -/
def Tower : List Layer := [
  { cubes := 1, exposed_top := 1, exposed_sides := 5 },
  { cubes := 3, exposed_top := 3, exposed_sides := 8 },
  { cubes := 4, exposed_top := 4, exposed_sides := 6 },
  { cubes := 6, exposed_top := 6, exposed_sides := 0 }
]

/-- The total number of cubes in the tower -/
def total_cubes : ℕ := (Tower.map (·.cubes)).sum

/-- The exposed surface area of the tower -/
def exposed_surface_area : ℕ := 
  (Tower.map (·.exposed_top)).sum + (Tower.map (·.exposed_sides)).sum

theorem tower_surface_area : 
  total_cubes = 14 ∧ exposed_surface_area = 29 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_l3585_358549


namespace NUMINAMATH_CALUDE_count_counterexamples_l3585_358588

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

def counterexample (n : ℕ) : Prop :=
  sum_of_digits n = 5 ∧ has_no_zero_digit n ∧ ¬ Nat.Prime n

theorem count_counterexamples : 
  ∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ counterexample n :=
sorry

end NUMINAMATH_CALUDE_count_counterexamples_l3585_358588


namespace NUMINAMATH_CALUDE_correct_product_l3585_358551

theorem correct_product (a b c : ℚ) (h1 : a = 0.005) (h2 : b = 3.24) (h3 : c = 0.0162) 
  (h4 : (5 : ℚ) * 324 = 1620) : a * b = c := by
  sorry

end NUMINAMATH_CALUDE_correct_product_l3585_358551


namespace NUMINAMATH_CALUDE_pascal_triangle_101_row_third_number_l3585_358558

/-- The number of elements in a row of Pascal's triangle -/
def row_elements (n : ℕ) : ℕ := n + 1

/-- The third number in a row of Pascal's triangle -/
def third_number (n : ℕ) : ℕ := n.choose 2

theorem pascal_triangle_101_row_third_number :
  ∃ (n : ℕ), row_elements n = 101 ∧ third_number n = 4950 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_101_row_third_number_l3585_358558


namespace NUMINAMATH_CALUDE_trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l3585_358582

-- Part I
theorem trig_sum_equals_negative_sqrt3_over_6 :
  Real.sin (5 * Real.pi / 3) + Real.cos (11 * Real.pi / 2) + Real.tan (-11 * Real.pi / 6) = -Real.sqrt 3 / 6 := by
  sorry

-- Part II
theorem trig_fraction_sum_simplification (θ : Real) 
  (h1 : Real.tan θ ≠ 0) (h2 : Real.tan θ ≠ 1) :
  (Real.sin θ / (1 - 1 / Real.tan θ)) + (Real.cos θ / (1 - Real.tan θ)) = Real.sin θ + Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l3585_358582


namespace NUMINAMATH_CALUDE_election_votes_l3585_358591

theorem election_votes (total_votes : ℕ) (winner_votes : ℕ) 
  (diff1 diff2 diff3 : ℕ) : 
  total_votes = 963 →
  winner_votes - diff1 + winner_votes - diff2 + winner_votes - diff3 + winner_votes = total_votes →
  diff1 = 53 →
  diff2 = 79 →
  diff3 = 105 →
  winner_votes = 300 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l3585_358591


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_one_l3585_358583

theorem no_solution_iff_m_equals_one :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 3 → ((3 - 2*x) / (x - 3) - (m*x - 2) / (3 - x) ≠ -1)) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_one_l3585_358583


namespace NUMINAMATH_CALUDE_pot_height_problem_shorter_pot_height_l3585_358515

theorem pot_height_problem (h₁ b₁ b₂ : ℝ) (h₁_pos : 0 < h₁) (b₁_pos : 0 < b₁) (b₂_pos : 0 < b₂) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem shorter_pot_height (tall_pot_height tall_pot_shadow short_pot_shadow : ℝ)
  (tall_pot_height_pos : 0 < tall_pot_height)
  (tall_pot_shadow_pos : 0 < tall_pot_shadow)
  (short_pot_shadow_pos : 0 < short_pot_shadow)
  (h_tall : tall_pot_height = 40)
  (h_tall_shadow : tall_pot_shadow = 20)
  (h_short_shadow : short_pot_shadow = 10) :
  tall_pot_height * short_pot_shadow / tall_pot_shadow = 20 :=
by sorry

end NUMINAMATH_CALUDE_pot_height_problem_shorter_pot_height_l3585_358515


namespace NUMINAMATH_CALUDE_negative_option_l3585_358564

theorem negative_option : ∃ (x : ℝ), x < 0 ∧ 
  x = -(-5)^2 ∧ 
  -(-5) ≥ 0 ∧ 
  |-5| ≥ 0 ∧ 
  (-5) * (-5) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_option_l3585_358564


namespace NUMINAMATH_CALUDE_unique_digit_solution_l3585_358520

theorem unique_digit_solution :
  ∃! (E U L S R T : ℕ),
    (E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (L ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    E ≠ U ∧ E ≠ L ∧ E ≠ S ∧ E ≠ R ∧ E ≠ T ∧
    U ≠ L ∧ U ≠ S ∧ U ≠ R ∧ U ≠ T ∧
    L ≠ S ∧ L ≠ R ∧ L ≠ T ∧
    S ≠ R ∧ S ≠ T ∧
    R ≠ T ∧
    E + U + L = 6 ∧
    S + R + U + T = 18 ∧
    U * T = 15 ∧
    S * L = 8 ∧
    E = 1 ∧ U = 3 ∧ L = 2 ∧ S = 4 ∧ R = 6 ∧ T = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l3585_358520


namespace NUMINAMATH_CALUDE_lot_length_l3585_358532

/-- Given a rectangular lot with width 20 meters, height 2 meters, and volume 1600 cubic meters,
    prove that the length of the lot is 40 meters. -/
theorem lot_length (width : ℝ) (height : ℝ) (volume : ℝ) (length : ℝ) :
  width = 20 →
  height = 2 →
  volume = 1600 →
  volume = length * width * height →
  length = 40 := by
  sorry

end NUMINAMATH_CALUDE_lot_length_l3585_358532


namespace NUMINAMATH_CALUDE_find_y_value_l3585_358533

theorem find_y_value (a b x y : ℤ) : 
  (a + b + 100 + 200300 + x) / 5 = 250 →
  (a + b + 300 + 150100 + x + y) / 6 = 200 →
  a % 5 = 0 →
  b % 5 = 0 →
  y = 49800 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l3585_358533


namespace NUMINAMATH_CALUDE_shirt_price_l3585_358525

/-- The cost of one pair of jeans in dollars -/
def jean_cost : ℝ := sorry

/-- The cost of one shirt in dollars -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jean_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $81 -/
axiom condition2 : 2 * jean_cost + 3 * shirt_cost = 81

/-- Theorem: The cost of one shirt is $21 -/
theorem shirt_price : shirt_cost = 21 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l3585_358525


namespace NUMINAMATH_CALUDE_school_population_l3585_358579

theorem school_population (G B D : ℕ) (h1 : G = 5467) (h2 : D = 1932) (h3 : B = G - D) :
  G + B = 9002 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3585_358579


namespace NUMINAMATH_CALUDE_towel_shrinkage_l3585_358538

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let shrunk_length := 0.8 * L
  let shrunk_breadth := 0.9 * B
  let shrunk_area := shrunk_length * shrunk_breadth
  let cumulative_shrunk_area := 0.95 * shrunk_area
  let folded_area := 0.5 * cumulative_shrunk_area
  let percentage_change := (folded_area - original_area) / original_area * 100
  percentage_change = -65.8 := by
sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l3585_358538


namespace NUMINAMATH_CALUDE_complex_simplification_l3585_358566

theorem complex_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (2 - 2*i) + 4*i * (7 - 3*i) = 26 + 14*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3585_358566


namespace NUMINAMATH_CALUDE_rectangle_area_l3585_358597

def circle_inscribed_rectangle (r : ℝ) (l w : ℝ) : Prop :=
  2 * r = w

def length_width_ratio (l w : ℝ) : Prop :=
  l = 3 * w

theorem rectangle_area (r l w : ℝ) 
  (h1 : circle_inscribed_rectangle r l w) 
  (h2 : length_width_ratio l w) 
  (h3 : r = 7) : l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3585_358597


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3585_358500

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 6x - 15
- The third term is 4x + 3
- The nth term is 4021

Prove that n = 627
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (6 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (4 * x + 3 : ℚ) = (6 * x - 15 : ℚ) + ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) ∧
  (3 * x - 4 : ℚ) + (n - 1 : ℕ) * ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) = 4021 →
  n = 627 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3585_358500


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3585_358594

def f (x : ℝ) : ℝ := x^4 - 2*x^3

theorem tangent_line_at_one (x y : ℝ) :
  (y - f 1 = (4 - 6) * (x - 1)) ↔ (y = -2*x + 1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3585_358594


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l3585_358509

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Define the conditions from the problem
def condition_A (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 5*x ∧ t.b = 12*x ∧ t.c = 13*x

def condition_B (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 2*x ∧ t.b = 3*x ∧ t.c = 5*x

def condition_C (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 9*k ∧ t.b = 40*k ∧ t.c = 41*k

def condition_D (t : Triangle) : Prop :=
  t.a = 3^2 ∧ t.b = 4^2 ∧ t.c = 5^2

-- Theorem statement
theorem right_triangle_conditions :
  (∀ t : Triangle, condition_A t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_B t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) ∧
  (∃ t : Triangle, condition_D t ∧ ¬is_right_triangle t) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l3585_358509


namespace NUMINAMATH_CALUDE_inequality_chain_l3585_358537

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3585_358537


namespace NUMINAMATH_CALUDE_problem_solution_l3585_358562

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) : 
  m = 4 ∧ a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 + c₀^2 = Real.sqrt 3 ∧ 
  4 * a₀^4 + 4 * b₀^4 + 4 * c₀^4 = m := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3585_358562


namespace NUMINAMATH_CALUDE_button_ratio_problem_l3585_358585

/-- Represents the button problem with Mark, Shane, and Sam -/
theorem button_ratio_problem (initial_buttons : ℕ) (shane_multiplier : ℕ) (final_buttons : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  let total_after_shane := initial_buttons + shane_multiplier * initial_buttons
  let sam_took := total_after_shane - final_buttons
  (sam_took : ℚ) / total_after_shane = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_button_ratio_problem_l3585_358585


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3585_358587

/-- Given two vectors a and b in ℝ², where a = (-5, 1) and b = (2, x),
    if a and b are perpendicular, then x = 10. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-5, 1)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3585_358587


namespace NUMINAMATH_CALUDE_six_digit_number_representation_l3585_358526

theorem six_digit_number_representation (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (1000 ≤ b ∧ b < 10000) →  -- b is a four-digit number
  (100000 ≤ 10000 * a + b ∧ 10000 * a + b < 1000000) →  -- result is a six-digit number
  10000 * a + b = 10000 * a + b :=  -- the representation is correct
by sorry

end NUMINAMATH_CALUDE_six_digit_number_representation_l3585_358526


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3585_358541

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (x + 3) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3585_358541


namespace NUMINAMATH_CALUDE_no_difference_of_primes_in_S_l3585_358502

/-- The set of numbers we're considering -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 10 * k + 7}

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if a number can be expressed as the difference of two primes -/
def is_difference_of_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p - q = n

/-- The main theorem: no number in S can be expressed as the difference of two primes -/
theorem no_difference_of_primes_in_S : ∀ n ∈ S, ¬(is_difference_of_primes n) := by
  sorry

end NUMINAMATH_CALUDE_no_difference_of_primes_in_S_l3585_358502


namespace NUMINAMATH_CALUDE_shortest_distance_to_E_l3585_358510

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Calculate the distance between two points on the grid -/
def gridDistance (p1 p2 : GridPoint) : Nat :=
  (p2.x - p1.x) + (p2.y - p1.y)

theorem shortest_distance_to_E :
  let P : GridPoint := ⟨0, 0⟩
  let A : GridPoint := ⟨5, 4⟩
  let B : GridPoint := ⟨6, 2⟩
  let C : GridPoint := ⟨3, 3⟩
  let D : GridPoint := ⟨5, 1⟩
  let E : GridPoint := ⟨1, 4⟩
  (gridDistance P E ≤ gridDistance P A) ∧
  (gridDistance P E ≤ gridDistance P B) ∧
  (gridDistance P E ≤ gridDistance P C) ∧
  (gridDistance P E ≤ gridDistance P D) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_E_l3585_358510


namespace NUMINAMATH_CALUDE_equation_solution_l3585_358570

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 10) / (x - 4) = (x - 3) / (x + 6)) →
  x = -48 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3585_358570


namespace NUMINAMATH_CALUDE_only_negative_four_squared_is_correct_l3585_358528

theorem only_negative_four_squared_is_correct : 
  (2^4 ≠ 8) ∧ 
  (-4^2 = -16) ∧ 
  (-8 - 8 ≠ 0) ∧ 
  ((-3)^2 ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_squared_is_correct_l3585_358528


namespace NUMINAMATH_CALUDE_raduzhny_population_is_900_l3585_358516

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe -/
def znoynoe_population : ℕ := 1000

/-- The difference between Znoynoe's population and the average village population -/
def population_difference : ℕ := 90

/-- The maximum population difference between any village and Znoynoe -/
def max_population_difference : ℕ := 100

/-- The total population of all villages except Znoynoe -/
def other_villages_population : ℕ := (num_villages - 1) * (znoynoe_population - population_difference)

/-- The population of Raduzhny -/
def raduzhny_population : ℕ := other_villages_population / (num_villages - 1)

theorem raduzhny_population_is_900 :
  raduzhny_population = 900 :=
sorry

end NUMINAMATH_CALUDE_raduzhny_population_is_900_l3585_358516


namespace NUMINAMATH_CALUDE_remainder_theorem_l3585_358568

theorem remainder_theorem (r : ℤ) : (r^15 + 1) % (r + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3585_358568


namespace NUMINAMATH_CALUDE_violet_marbles_indeterminate_l3585_358572

/-- Represents the number of marbles Dan has -/
structure DansMarbles where
  initialGreen : ℝ
  takenGreen : ℝ
  finalGreen : ℝ
  violet : ℝ

/-- Theorem stating that the number of violet marbles cannot be determined -/
theorem violet_marbles_indeterminate (d : DansMarbles) 
  (h1 : d.initialGreen = 32)
  (h2 : d.takenGreen = 23)
  (h3 : d.finalGreen = 9)
  (h4 : d.initialGreen - d.takenGreen = d.finalGreen) :
  ∀ v : ℝ, ∃ d' : DansMarbles, d'.initialGreen = d.initialGreen ∧ 
                                d'.takenGreen = d.takenGreen ∧ 
                                d'.finalGreen = d.finalGreen ∧ 
                                d'.violet = v :=
sorry

end NUMINAMATH_CALUDE_violet_marbles_indeterminate_l3585_358572


namespace NUMINAMATH_CALUDE_kids_difference_l3585_358536

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 22) 
  (h2 : tuesday = 14) : 
  monday - tuesday = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l3585_358536


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3585_358512

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x + 3 is at (1, 2) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3585_358512


namespace NUMINAMATH_CALUDE_day_of_week_previous_year_l3585_358508

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeap : Bool

/-- Returns the day of the week given a day number and a year -/
def dayOfWeek (day : ℕ) (year : Year) : DayOfWeek :=
  sorry

/-- Advances the day of the week by a given number of days -/
def advanceDays (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry

theorem day_of_week_previous_year 
  (N : Year)
  (h1 : N.isLeap = true)
  (h2 : dayOfWeek 250 N = DayOfWeek.Wednesday)
  (h3 : dayOfWeek 150 ⟨N.value + 1, false⟩ = DayOfWeek.Wednesday) :
  dayOfWeek 100 ⟨N.value - 1, false⟩ = DayOfWeek.Saturday :=
by sorry

end NUMINAMATH_CALUDE_day_of_week_previous_year_l3585_358508


namespace NUMINAMATH_CALUDE_collapsible_iff_power_of_two_l3585_358531

/-- A token arrangement in the plane -/
structure TokenArrangement :=
  (n : ℕ+)  -- number of tokens
  (positions : Fin n → ℝ × ℝ)  -- positions of tokens in the plane

/-- Predicate for an arrangement being collapsible -/
def Collapsible (arrangement : TokenArrangement) : Prop :=
  ∃ (final_pos : ℝ × ℝ), ∀ i : Fin arrangement.n, 
    ∃ (moves : ℕ), arrangement.positions i = final_pos

/-- Predicate for a number being a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The main theorem -/
theorem collapsible_iff_power_of_two :
  ∀ n : ℕ+, (∀ arrangement : TokenArrangement, arrangement.n = n → Collapsible arrangement) ↔ IsPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_collapsible_iff_power_of_two_l3585_358531


namespace NUMINAMATH_CALUDE_circles_common_chord_and_diameter_l3585_358524

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle with common chord as diameter
def circle_with_common_chord_diameter (x y : ℝ) : Prop := 
  (x + 8/5)^2 + (y - 6/5)^2 = 36/5

-- Theorem statement
theorem circles_common_chord_and_diameter :
  (∃ x y : ℝ, C1 x y ∧ C2 x y ∧ common_chord x y) →
  (∃ a b : ℝ, common_chord a b ∧ 
    (a - (-4))^2 + (b - 0)^2 = 5) ∧
  (∀ x y : ℝ, circle_with_common_chord_diameter x y ↔
    (∃ t : ℝ, x = -4 * (1 - t) + 4/5 * t ∧ 
              y = 0 * (1 - t) + 12/5 * t ∧ 
              0 ≤ t ∧ t ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_circles_common_chord_and_diameter_l3585_358524


namespace NUMINAMATH_CALUDE_reciprocal_multiple_l3585_358563

theorem reciprocal_multiple (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 8) (h3 : x + 8 = k * (1 / x)) : k = 128 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_multiple_l3585_358563


namespace NUMINAMATH_CALUDE_polynomial_value_l3585_358575

theorem polynomial_value (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) :
  (2*a*b^2 + a^2*b) - (3*a*b^2 + a^2*b - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3585_358575


namespace NUMINAMATH_CALUDE_mn_length_is_two_l3585_358511

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the intersection points M and N
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
    x₁ ≠ x₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem mn_length_is_two (k : ℝ) :
  intersection_points k →
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                      y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                      dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                     y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_mn_length_is_two_l3585_358511


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l3585_358552

theorem binomial_coefficient_inequality
  (n k h : ℕ)
  (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l3585_358552


namespace NUMINAMATH_CALUDE_sharon_angela_cutlery_ratio_l3585_358553

/-- Prove that the ratio of Sharon's cutlery to Angela's cutlery is 2:1 -/
theorem sharon_angela_cutlery_ratio :
  let angela_pots : ℕ := 20
  let angela_plates : ℕ := 3 * angela_pots + 6
  let angela_cutlery : ℕ := angela_plates / 2
  let sharon_pots : ℕ := angela_pots / 2
  let sharon_plates : ℕ := 3 * angela_plates - 20
  let sharon_total : ℕ := 254
  let sharon_cutlery : ℕ := sharon_total - (sharon_pots + sharon_plates)
  (sharon_cutlery : ℚ) / (angela_cutlery : ℚ) = 2
  := by sorry

end NUMINAMATH_CALUDE_sharon_angela_cutlery_ratio_l3585_358553


namespace NUMINAMATH_CALUDE_rainy_days_count_l3585_358556

theorem rainy_days_count (n : ℕ) : 
  (∃ (rainy_days non_rainy_days : ℕ),
    rainy_days + non_rainy_days = 7 ∧
    n * rainy_days + 5 * non_rainy_days = 22 ∧
    5 * non_rainy_days - n * rainy_days = 8) →
  (∃ (rainy_days : ℕ), rainy_days = 4) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l3585_358556


namespace NUMINAMATH_CALUDE_book_sale_revenue_l3585_358586

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  (3 * total_books = 108) →  -- Condition: 1/3 of total books is 36
  (price_per_book = 7/2) →   -- Price per book is $3.50
  (2 * total_books / 3 * price_per_book = 252) := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l3585_358586


namespace NUMINAMATH_CALUDE_max_squares_visited_999_board_l3585_358569

/-- A limp rook on a 999 x 999 board can move to adjacent squares and must turn at each move. -/
structure LimpRook where
  board_size : Nat
  move_to_adjacent : Bool
  must_turn : Bool

/-- A route for a limp rook is non-intersecting and cyclic. -/
structure Route where
  non_intersecting : Bool
  cyclic : Bool

/-- The maximum number of squares a limp rook can visit. -/
def max_squares_visited (rook : LimpRook) (route : Route) : Nat :=
  996000

/-- Theorem stating the maximum number of squares a limp rook can visit on a 999 x 999 board. -/
theorem max_squares_visited_999_board (rook : LimpRook) (route : Route) :
  rook.board_size = 999 ∧ rook.move_to_adjacent ∧ rook.must_turn ∧
  route.non_intersecting ∧ route.cyclic →
  max_squares_visited rook route = 996000 := by
  sorry

end NUMINAMATH_CALUDE_max_squares_visited_999_board_l3585_358569


namespace NUMINAMATH_CALUDE_children_off_bus_l3585_358547

theorem children_off_bus (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 22 → got_on = 40 → final = 2 → initial + got_on - final = 60 := by
  sorry

end NUMINAMATH_CALUDE_children_off_bus_l3585_358547


namespace NUMINAMATH_CALUDE_dilan_initial_marbles_l3585_358527

/-- The number of people involved in the marble redistribution --/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution --/
def marbles_after : ℕ := 15

/-- Martha's initial number of marbles --/
def martha_initial : ℕ := 20

/-- Phillip's initial number of marbles --/
def phillip_initial : ℕ := 19

/-- Veronica's initial number of marbles --/
def veronica_initial : ℕ := 7

/-- The theorem stating Dilan's initial number of marbles --/
theorem dilan_initial_marbles :
  (num_people * marbles_after) - (martha_initial + phillip_initial + veronica_initial) = 14 :=
by sorry

end NUMINAMATH_CALUDE_dilan_initial_marbles_l3585_358527


namespace NUMINAMATH_CALUDE_cubic_integer_roots_imply_b_form_l3585_358544

theorem cubic_integer_roots_imply_b_form (a b : ℤ) 
  (h : ∃ (u v w : ℤ), u^3 - a*u^2 - b = 0 ∧ v^3 - a*v^2 - b = 0 ∧ w^3 - a*w^2 - b = 0) :
  ∃ (d k : ℤ), b = d * k^2 ∧ ∃ (m : ℤ), a = d * m :=
by sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_imply_b_form_l3585_358544


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l3585_358546

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The tenth term of a geometric sequence with first term 5 and common ratio 3/4 -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l3585_358546


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3585_358589

theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := -5 * i / (2 + 3 * i)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3585_358589


namespace NUMINAMATH_CALUDE_f_composed_with_g_l3585_358559

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := x + 2

theorem f_composed_with_g : f (2 + g 3) = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l3585_358559


namespace NUMINAMATH_CALUDE_petri_dishes_count_l3585_358598

/-- The number of petri dishes in the biology lab -/
def num_petri_dishes : ℕ :=
  10800

/-- The total number of germs in the lab -/
def total_germs : ℕ :=
  5400000

/-- The number of germs in a single dish -/
def germs_per_dish : ℕ :=
  500

/-- Theorem stating that the number of petri dishes is correct -/
theorem petri_dishes_count :
  num_petri_dishes = total_germs / germs_per_dish :=
by sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l3585_358598


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3585_358503

def i : ℂ := Complex.I

def z : ℂ := 2 * i * (1 + i)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3585_358503


namespace NUMINAMATH_CALUDE_min_max_sum_l3585_358513

theorem min_max_sum (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  max (a + b + c) (max (b + c + d) (max (c + d + e) (max (d + e + f) (e + f + g)))) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l3585_358513


namespace NUMINAMATH_CALUDE_robot_sorting_problem_l3585_358577

/-- Represents the sorting capacity of robots -/
structure RobotSorting where
  typeA : ℕ  -- Number of type A robots
  typeB : ℕ  -- Number of type B robots
  totalPackages : ℕ  -- Total packages sorted per hour

/-- Theorem representing the robot sorting problem -/
theorem robot_sorting_problem 
  (scenario1 : RobotSorting)
  (scenario2 : RobotSorting)
  (h1 : scenario1.typeA = 80 ∧ scenario1.typeB = 100 ∧ scenario1.totalPackages = 8200)
  (h2 : scenario2.typeA = 50 ∧ scenario2.typeB = 50 ∧ scenario2.totalPackages = 4500)
  (totalNewRobots : ℕ)
  (h3 : totalNewRobots = 200)
  (minNewPackages : ℕ)
  (h4 : minNewPackages = 9000) :
  ∃ (maxTypeA : ℕ),
    maxTypeA ≤ totalNewRobots ∧
    ∀ (newTypeA : ℕ),
      newTypeA ≤ totalNewRobots →
      (40 * newTypeA + 50 * (totalNewRobots - newTypeA) ≥ minNewPackages →
       newTypeA ≤ maxTypeA) ∧
    40 * maxTypeA + 50 * (totalNewRobots - maxTypeA) ≥ minNewPackages ∧
    maxTypeA = 100 :=
sorry

end NUMINAMATH_CALUDE_robot_sorting_problem_l3585_358577


namespace NUMINAMATH_CALUDE_fraction_numerator_l3585_358561

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : x / y * y + 3 * y / 10 = 1 / 2 * y) : x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l3585_358561


namespace NUMINAMATH_CALUDE_arrangements_count_is_correct_l3585_358517

/-- The number of ways to divide 2 teachers and 4 students into two groups,
    each containing 1 teacher and 2 students, and then assign these groups to two locations -/
def arrangementsCount : ℕ := 12

/-- The number of ways to choose 2 students from 4 students -/
def waysToChooseStudents : ℕ := Nat.choose 4 2

/-- The number of ways to choose 2 students from 2 students (always 1) -/
def waysToChooseRemainingStudents : ℕ := Nat.choose 2 2

/-- The number of ways to assign 2 groups to 2 locations -/
def waysToAssignGroups : ℕ := 2

theorem arrangements_count_is_correct :
  arrangementsCount = waysToChooseStudents * waysToChooseRemainingStudents * waysToAssignGroups :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_is_correct_l3585_358517


namespace NUMINAMATH_CALUDE_triangle_area_l3585_358522

theorem triangle_area (a b c : ℝ) (A : ℝ) : 
  b = 3 → 
  a - c = 2 → 
  A = 2 * Real.pi / 3 → 
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3585_358522


namespace NUMINAMATH_CALUDE_gcd_lcm_18_24_l3585_358599

theorem gcd_lcm_18_24 :
  (Nat.gcd 18 24 = 6) ∧ (Nat.lcm 18 24 = 72) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_18_24_l3585_358599


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3585_358557

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3585_358557


namespace NUMINAMATH_CALUDE_star_operation_sum_l3585_358554

theorem star_operation_sum (c d : ℕ) : 
  c ≥ 2 → d ≥ 2 → c^d + c*d = 42 → c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_sum_l3585_358554


namespace NUMINAMATH_CALUDE_scientific_notation_of_1268000000_l3585_358571

theorem scientific_notation_of_1268000000 :
  (1268000000 : ℝ) = 1.268 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1268000000_l3585_358571


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3585_358543

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  a + b + c = Real.sqrt 2 + 1 →
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3585_358543


namespace NUMINAMATH_CALUDE_journey_time_increase_l3585_358545

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (overall_speed : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : overall_speed = 40) :
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let total_time := total_distance / overall_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_increase_l3585_358545


namespace NUMINAMATH_CALUDE_average_weight_increase_l3585_358539

theorem average_weight_increase 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 35)
  (h3 : new_weight = 55) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3585_358539


namespace NUMINAMATH_CALUDE_noelle_walking_speed_l3585_358523

/-- Noelle's walking problem -/
theorem noelle_walking_speed (d : ℝ) (h : d > 0) : 
  let v : ℝ := (2 * d) / (d / 15 + d / 5)
  v = 3 := by sorry

end NUMINAMATH_CALUDE_noelle_walking_speed_l3585_358523


namespace NUMINAMATH_CALUDE_triangle_shape_from_complex_product_l3585_358518

open Complex

/-- Given a triangle ABC with sides a, b and angles A, B, C,
    if z₁ = a + bi and z₂ = cos A + i cos B, and their product is purely imaginary,
    then the triangle is either isosceles or right-angled. -/
theorem triangle_shape_from_complex_product (a b : ℝ) (A B C : ℝ) :
  let z₁ : ℂ := ⟨a, b⟩
  let z₂ : ℂ := ⟨Real.cos A, Real.cos B⟩
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →  -- Triangle conditions
  (z₁ * z₂).re = 0 →  -- Product is purely imaginary
  (A = B) ∨ (A + B = π / 2) :=  -- Triangle is isosceles or right-angled
by sorry

end NUMINAMATH_CALUDE_triangle_shape_from_complex_product_l3585_358518


namespace NUMINAMATH_CALUDE_inequality_proof_l3585_358540

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3585_358540


namespace NUMINAMATH_CALUDE_sequence_sum_l3585_358529

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = a n * r

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 6 = 4 →
  a 7 = 1 →
  a 4 + a 5 = 80 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3585_358529


namespace NUMINAMATH_CALUDE_fourth_root_equation_l3585_358504

theorem fourth_root_equation (X : ℝ) : 
  (X^5)^(1/4) = 32 * (32^(1/16)) → X = 16 * (2^(1/4)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l3585_358504


namespace NUMINAMATH_CALUDE_luke_finances_duration_l3585_358550

/-- Represents Luke's financial situation --/
structure LukeFinances where
  total_income : ℕ
  weekly_expenses : ℕ

/-- Calculates how many full weeks Luke's money will last --/
def weeks_money_lasts (finances : LukeFinances) : ℕ :=
  finances.total_income / finances.weekly_expenses

/-- Calculates the remaining money after the last full week --/
def remaining_money (finances : LukeFinances) : ℕ :=
  finances.total_income % finances.weekly_expenses

/-- Theorem stating how long Luke's money will last and how much will remain --/
theorem luke_finances_duration (finances : LukeFinances) 
  (h1 : finances.total_income = 34)
  (h2 : finances.weekly_expenses = 7) : 
  weeks_money_lasts finances = 4 ∧ remaining_money finances = 6 := by
  sorry

#eval weeks_money_lasts ⟨34, 7⟩
#eval remaining_money ⟨34, 7⟩

end NUMINAMATH_CALUDE_luke_finances_duration_l3585_358550


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3585_358574

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 4

-- State the theorem
theorem f_decreasing_interval :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 6 ∧ f x₂ = 2) →  -- Maximum and minimum conditions
  (∀ x, f x ≤ 6) →                           -- 6 is the maximum value
  (∀ x, f x ≥ 2) →                           -- 2 is the minimum value
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3585_358574


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_of_squares_l3585_358506

theorem square_sum_given_product_and_sum_of_squares (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a^2 * b + a * b^2 = 15) : 
  a^2 + b^2 = 19 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_of_squares_l3585_358506


namespace NUMINAMATH_CALUDE_ladder_distance_l3585_358534

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l3585_358534


namespace NUMINAMATH_CALUDE_cubic_root_sum_ninth_power_l3585_358548

theorem cubic_root_sum_ninth_power (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_ninth_power_l3585_358548


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l3585_358592

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x y, -2 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l3585_358592


namespace NUMINAMATH_CALUDE_star_to_square_ratio_is_three_fifths_l3585_358581

/-- Represents a square with side length 5 cm containing a star formed by four identical isosceles triangles, each with height 1 cm -/
structure StarInSquare where
  square_side : ℝ
  triangle_height : ℝ
  square_side_eq : square_side = 5
  triangle_height_eq : triangle_height = 1

/-- Calculates the ratio of the star area to the square area -/
def star_to_square_ratio (s : StarInSquare) : ℚ :=
  3 / 5

/-- Theorem stating that the ratio of the star area to the square area is 3/5 -/
theorem star_to_square_ratio_is_three_fifths (s : StarInSquare) :
  star_to_square_ratio s = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_star_to_square_ratio_is_three_fifths_l3585_358581


namespace NUMINAMATH_CALUDE_nancy_shoe_count_l3585_358595

def shoe_count (boots slippers heels : ℕ) : ℕ :=
  2 * (boots + slippers + heels)

theorem nancy_shoe_count :
  ∀ (boots slippers heels : ℕ),
    boots = 6 →
    slippers = boots + 9 →
    heels = 3 * (boots + slippers) →
    shoe_count boots slippers heels = 168 := by
  sorry

end NUMINAMATH_CALUDE_nancy_shoe_count_l3585_358595


namespace NUMINAMATH_CALUDE_hex_palindrome_probability_l3585_358596

/-- Represents a hexadecimal digit (0-15) -/
def HexDigit := Fin 16

/-- Represents a 6-digit hexadecimal palindrome -/
structure HexPalindrome where
  a : HexDigit
  b : HexDigit
  c : HexDigit
  value : ℕ := 1048592 * a.val + 65792 * b.val + 4096 * c.val

/-- Predicate to check if a natural number is a hexadecimal palindrome -/
def isHexPalindrome (n : ℕ) : Prop := sorry

/-- The total number of 6-digit hexadecimal palindromes -/
def totalPalindromes : ℕ := 3840

/-- The number of 6-digit hexadecimal palindromes that, when divided by 17, 
    result in another hexadecimal palindrome -/
def validPalindromes : ℕ := sorry

theorem hex_palindrome_probability : 
  (validPalindromes : ℚ) / totalPalindromes = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_hex_palindrome_probability_l3585_358596


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l3585_358521

/-- Represents the number of students in grade 10 -/
def grade_10_students : ℕ := sorry

/-- Represents the number of students in grade 11 -/
def grade_11_students : ℕ := grade_10_students + 300

/-- Represents the number of students in grade 12 -/
def grade_12_students : ℕ := 2 * grade_10_students

/-- The total number of students in all three grades -/
def total_students : ℕ := 3500

/-- The sampling ratio -/
def sampling_ratio : ℚ := 1 / 100

/-- Theorem stating the number of grade 10 students to be sampled -/
theorem grade_10_sample_size : 
  grade_10_students + grade_11_students + grade_12_students = total_students →
  (↑grade_10_students * sampling_ratio).floor = 8 := by
  sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l3585_358521


namespace NUMINAMATH_CALUDE_base_4_arithmetic_l3585_358560

/-- Converts a base 4 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def to_base_4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_4_arithmetic :
  to_base_4 (to_base_10 [0, 3, 2] - to_base_10 [1, 0, 1] + to_base_10 [2, 2, 3]) = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_4_arithmetic_l3585_358560


namespace NUMINAMATH_CALUDE_complex_number_problem_l3585_358519

theorem complex_number_problem (a b c : ℂ) 
  (h_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 8) :
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3585_358519


namespace NUMINAMATH_CALUDE_max_two_digit_composite_relatively_prime_l3585_358530

/-- A number is two-digit if it's between 10 and 99 inclusive -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- Two numbers are relatively prime if their greatest common divisor is 1 -/
def areRelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of numbers satisfying our conditions -/
def validSet (S : Finset ℕ) : Prop :=
  ∀ n ∈ S, isTwoDigit n ∧ isComposite n ∧
  ∀ m ∈ S, m ≠ n → areRelativelyPrime m n

theorem max_two_digit_composite_relatively_prime :
  (∃ S : Finset ℕ, validSet S ∧ S.card = 4) ∧
  ∀ T : Finset ℕ, validSet T → T.card ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_composite_relatively_prime_l3585_358530


namespace NUMINAMATH_CALUDE_greatest_n_for_given_conditions_l3585_358576

theorem greatest_n_for_given_conditions (x : ℤ) (N : ℝ) : 
  (N * 10^x < 210000 ∧ x ≤ 4) → 
  ∃ (max_N : ℤ), max_N = 20 ∧ ∀ (m : ℤ), (m : ℝ) * 10^4 < 210000 → m ≤ max_N :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_given_conditions_l3585_358576


namespace NUMINAMATH_CALUDE_vector_dot_product_l3585_358590

/-- Given two vectors a and b in ℝ² satisfying certain conditions, 
    their dot product is equal to -222/25 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + 2 * b.1 = 1 ∧ a.2 + 2 * b.2 = -3)
    (h2 : 2 * a.1 - b.1 = 1 ∧ 2 * a.2 - b.2 = 9) :
    a.1 * b.1 + a.2 * b.2 = -222 / 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3585_358590


namespace NUMINAMATH_CALUDE_power_relation_l3585_358542

theorem power_relation (a : ℝ) (b : ℝ) (h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3585_358542


namespace NUMINAMATH_CALUDE_product_of_roots_eq_one_l3585_358573

theorem product_of_roots_eq_one :
  let f : ℝ → ℝ := λ x => x^(Real.log x / Real.log 5) - 25
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ * r₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_eq_one_l3585_358573
