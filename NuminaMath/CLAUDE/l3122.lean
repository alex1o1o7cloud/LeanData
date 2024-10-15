import Mathlib

namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l3122_312267

/-- Given a right triangle with two sides of 8 units and 15 units, 
    the least possible length of the third side is √161 units. -/
theorem least_possible_third_side_length : ∀ a b c : ℝ,
  a = 8 →
  b = 15 →
  (a = c ∧ b * b = c * c - a * a) ∨ 
  (b = c ∧ a * a = c * c - b * b) ∨
  (c * c = a * a + b * b) →
  c ≥ Real.sqrt 161 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l3122_312267


namespace NUMINAMATH_CALUDE_square_side_length_l3122_312277

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side * side = area → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3122_312277


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3122_312252

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 - (m + 2) * x > 0 ↔ x < 0 ∨ x > 2) →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3122_312252


namespace NUMINAMATH_CALUDE_unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l3122_312230

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (collinear : Point → Point → Point → Prop)
variable (on_plane : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Proposition A
theorem unique_plane_through_three_points 
  (p q r : Point) (h : ¬ collinear p q r) :
  ∃! π : Plane, on_plane p π ∧ on_plane q π ∧ on_plane r π :=
sorry

-- Proposition B
theorem perpendicular_line_implies_parallel_planes 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) :
  parallel_planes α β :=
sorry

-- Proposition C (negation)
theorem parallel_to_plane_not_implies_parallel_lines 
  (m n : Line) (α : Plane) :
  parallel_line_plane m α ∧ parallel_line_plane n α → 
  ¬ (parallel_lines m n → True) :=
sorry

-- Proposition D
theorem perpendicular_to_plane_implies_parallel_lines 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l3122_312230


namespace NUMINAMATH_CALUDE_triangle_properties_l3122_312286

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosC : ℝ

def is_valid_triangle (t : Triangle) : Prop :=
  t.sinA > 0 ∧ t.sinB > 0 ∧ t.sinC > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem triangle_properties (t : Triangle) 
  (h_valid : is_valid_triangle t)
  (h_arith_seq : 2 * t.sinB = t.sinA + t.sinC)
  (h_cosC : t.cosC = 1/3) :
  (t.b / t.a = 10/9) ∧ 
  (t.c = 11 → t.a * t.b * t.sinC / 2 = 30 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3122_312286


namespace NUMINAMATH_CALUDE_representatives_formula_l3122_312219

/-- Represents the number of representatives for a given number of students -/
def num_representatives (x : ℕ) : ℕ :=
  if x % 10 > 6 then
    x / 10 + 1
  else
    x / 10

/-- The greatest integer function -/
def floor (r : ℚ) : ℤ :=
  ⌊r⌋

theorem representatives_formula (x : ℕ) :
  (num_representatives x : ℤ) = floor ((x + 3 : ℚ) / 10) :=
sorry

end NUMINAMATH_CALUDE_representatives_formula_l3122_312219


namespace NUMINAMATH_CALUDE_last_digit_of_largest_known_prime_l3122_312249

theorem last_digit_of_largest_known_prime (n : ℕ) : n = 216091 →
  (2^n - 1) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_largest_known_prime_l3122_312249


namespace NUMINAMATH_CALUDE_fourth_person_height_l3122_312212

/-- Given 4 people with heights in increasing order, prove that the 4th person is 85 inches tall. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  (h₁ < h₂) ∧ (h₂ < h₃) ∧ (h₃ < h₄) ∧  -- Heights in increasing order
  (h₂ - h₁ = 2) ∧                      -- Difference between 1st and 2nd
  (h₃ - h₂ = 2) ∧                      -- Difference between 2nd and 3rd
  (h₄ - h₃ = 6) ∧                      -- Difference between 3rd and 4th
  ((h₁ + h₂ + h₃ + h₄) / 4 = 79)       -- Average height
  → h₄ = 85 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3122_312212


namespace NUMINAMATH_CALUDE_speed_in_still_water_l3122_312207

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 27 →
  downstream_speed = 35 →
  (upstream_speed + downstream_speed) / 2 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l3122_312207


namespace NUMINAMATH_CALUDE_spinner_probability_l3122_312250

theorem spinner_probability (P : Finset (Fin 4) → ℚ) 
  (h_total : P {0, 1, 2, 3} = 1)
  (h_A : P {0} = 1/4)
  (h_B : P {1} = 1/3)
  (h_D : P {3} = 1/6) :
  P {2} = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3122_312250


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l3122_312289

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (x y : ℕ) : 
  x ≠ y →
  is_factor x 48 →
  is_factor y 48 →
  (Even x ∨ Even y) →
  ¬(is_factor (x * y) 48) →
  ∀ (a b : ℕ), a ≠ b ∧ is_factor a 48 ∧ is_factor b 48 ∧ (Even a ∨ Even b) ∧ ¬(is_factor (a * b) 48) →
  x * y ≤ a * b →
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l3122_312289


namespace NUMINAMATH_CALUDE_study_tour_problem_l3122_312227

/-- Represents a bus type with seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the number of buses needed for a given number of participants and bus type -/
def busesNeeded (participants : ℕ) (busType : BusType) : ℕ :=
  (participants + busType.seats - 1) / busType.seats

/-- Calculates the total rental cost for a given number of participants and bus type -/
def rentalCost (participants : ℕ) (busType : BusType) : ℕ :=
  (busesNeeded participants busType) * busType.fee

theorem study_tour_problem (x y : ℕ) (typeA typeB : BusType)
    (h1 : 45 * y + 15 = x)
    (h2 : 60 * (y - 3) = x)
    (h3 : typeA.seats = 45)
    (h4 : typeA.fee = 200)
    (h5 : typeB.seats = 60)
    (h6 : typeB.fee = 300) :
    x = 600 ∧ y = 13 ∧ rentalCost x typeA < rentalCost x typeB := by
  sorry

end NUMINAMATH_CALUDE_study_tour_problem_l3122_312227


namespace NUMINAMATH_CALUDE_problem_solution_l3122_312232

theorem problem_solution (x y : ℝ) 
  (hx : x = Real.sqrt 5 + Real.sqrt 3) 
  (hy : y = Real.sqrt 5 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2) / (x^2 - y^2) = Real.sqrt 15 / 3 ∧ 
  Real.sqrt (x^2 + y^2 - 3) = Real.sqrt 13 ∨ 
  Real.sqrt (x^2 + y^2 - 3) = -Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3122_312232


namespace NUMINAMATH_CALUDE_terminal_side_angle_l3122_312215

/-- Given a point P(-4,3) on the terminal side of angle θ, prove that 2sin θ + cos θ = 2/5 -/
theorem terminal_side_angle (θ : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  (P.1 = -4 ∧ P.2 = 3) →  -- Point P(-4,3)
  (P.1 = Real.cos θ * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin θ * Real.sqrt (P.1^2 + P.2^2)) →  -- P is on the terminal side of θ
  2 * Real.sin θ + Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_angle_l3122_312215


namespace NUMINAMATH_CALUDE_total_unbroken_seashells_is_17_l3122_312290

/-- The number of unbroken seashells Tom found over three days -/
def total_unbroken_seashells : ℕ :=
  let day1_total := 7
  let day1_broken := 4
  let day2_total := 12
  let day2_broken := 5
  let day3_total := 15
  let day3_broken := 8
  (day1_total - day1_broken) + (day2_total - day2_broken) + (day3_total - day3_broken)

/-- Theorem stating that the total number of unbroken seashells is 17 -/
theorem total_unbroken_seashells_is_17 : total_unbroken_seashells = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_unbroken_seashells_is_17_l3122_312290


namespace NUMINAMATH_CALUDE_chess_match_duration_l3122_312285

-- Define the given conditions
def polly_time_per_move : ℕ := 28
def peter_time_per_move : ℕ := 40
def total_moves : ℕ := 30

-- Define the theorem
theorem chess_match_duration :
  (total_moves / 2 * polly_time_per_move + total_moves / 2 * peter_time_per_move) / 60 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_duration_l3122_312285


namespace NUMINAMATH_CALUDE_store_inventory_difference_l3122_312275

theorem store_inventory_difference (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72) 
  (h2 : diet_soda = 32) 
  (h3 : apples = 78) : 
  (regular_soda + diet_soda) - apples = 26 := by
  sorry

end NUMINAMATH_CALUDE_store_inventory_difference_l3122_312275


namespace NUMINAMATH_CALUDE_triangle_problem_l3122_312268

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧
  -- Given equation
  c * Real.cos B + (Real.sqrt 3 / 3) * b * Real.sin C - a = 0 ∧
  -- Given side length
  c = 3 ∧
  -- Given area
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 →
  -- Conclusions
  C = π/3 ∧ a + b = 3 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3122_312268


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l3122_312257

theorem nested_sqrt_value (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l3122_312257


namespace NUMINAMATH_CALUDE_lcm_1260_980_l3122_312237

theorem lcm_1260_980 : Nat.lcm 1260 980 = 8820 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1260_980_l3122_312237


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3122_312246

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = -6)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3122_312246


namespace NUMINAMATH_CALUDE_last_remaining_card_l3122_312209

/-- Represents a playing card --/
inductive Card
  | Joker : Bool → Card  -- True for Big Joker, False for Small Joker
  | Regular : Suit → Rank → Card

/-- Represents the suit of a card --/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card --/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of cards in the specified order --/
def createDeck : Deck := sorry

/-- Combines two decks of cards --/
def combinedDecks : Deck := sorry

/-- Simulates the process of discarding and moving cards --/
def discardAndMove (deck : Deck) : Card := sorry

/-- Theorem stating that the last remaining card is the 6 of Diamonds from the second deck --/
theorem last_remaining_card :
  discardAndMove combinedDecks = Card.Regular Suit.Diamonds Rank.Six := by sorry

end NUMINAMATH_CALUDE_last_remaining_card_l3122_312209


namespace NUMINAMATH_CALUDE_cos_difference_l3122_312247

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_l3122_312247


namespace NUMINAMATH_CALUDE_bert_sandwiches_l3122_312279

def sandwiches_problem (initial_sandwiches : ℕ) : ℕ :=
  let day1_remaining := initial_sandwiches / 2
  let day2_remaining := day1_remaining - (2 * day1_remaining / 3)
  let day3_eaten := (2 * day1_remaining / 3) - 2
  day2_remaining - min day2_remaining day3_eaten

theorem bert_sandwiches :
  sandwiches_problem 36 = 0 := by
  sorry

end NUMINAMATH_CALUDE_bert_sandwiches_l3122_312279


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3122_312288

-- Define the parameters
def purchase_price : ℝ := 4700
def selling_price : ℝ := 5800
def gain_percent : ℝ := 1.7543859649122806

-- Define the theorem
theorem repair_cost_calculation (repair_cost : ℝ) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = gain_percent →
  repair_cost = 1000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l3122_312288


namespace NUMINAMATH_CALUDE_extremum_implies_zero_derivative_l3122_312243

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, |y - x| < 1 → f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem extremum_implies_zero_derivative (f : ℝ → ℝ) (x : ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : is_extremum f x) : 
  deriv f x = 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_zero_derivative_l3122_312243


namespace NUMINAMATH_CALUDE_prob_at_least_one_target_l3122_312297

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 54

/-- The number of cards that are diamonds, aces, or jokers -/
def target_cards : ℕ := 18

/-- The probability of drawing a card that is not a diamond, ace, or joker -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing two cards with replacement, where at least one is a diamond, ace, or joker -/
theorem prob_at_least_one_target : 
  1 - prob_not_target ^ 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_target_l3122_312297


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l3122_312265

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost :
  total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l3122_312265


namespace NUMINAMATH_CALUDE_range_of_a_l3122_312296

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3 ∧ x - a > 0) ↔ (-1 < x ∧ x < 4)) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3122_312296


namespace NUMINAMATH_CALUDE_magnitude_z_l3122_312292

open Complex

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : abs w = Real.sqrt 29) :
  abs z = (25 * Real.sqrt 29) / 29 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_l3122_312292


namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l3122_312254

theorem power_seven_mod_eight : 7^135 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l3122_312254


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l3122_312294

theorem right_triangle_hypotenuse_segment_ratio 
  (A B C D : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (leg_ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 
               (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) 
  (D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)) 
  (D_perpendicular : (B.1 - D.1) * (C.1 - A.1) + (B.2 - D.2) * (C.2 - A.2) = 0) : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 
  4 * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l3122_312294


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l3122_312204

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : Odd n) 
  (h6 : 1/a + 1/b + 1/c = 1/(a+b+c)) : 
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l3122_312204


namespace NUMINAMATH_CALUDE_isosceles_triangle_count_l3122_312206

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

/-- Represents the set of all colored points in the triangle -/
def ColoredPoints (t : EquilateralTriangle) : Set Point :=
  sorry

/-- Checks if a triangle formed by three points is isosceles -/
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  sorry

/-- Counts the number of isosceles triangles formed by the colored points -/
def CountIsoscelesTriangles (t : EquilateralTriangle) : ℕ :=
  sorry

/-- Main theorem: There are exactly 18 isosceles triangles with vertices at the colored points -/
theorem isosceles_triangle_count (t : EquilateralTriangle) :
  CountIsoscelesTriangles t = 18 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_count_l3122_312206


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l3122_312263

/-- Represents a right circular cone filled with liquid -/
structure LiquidCone where
  radius : ℝ
  height : ℝ
  volume : ℝ

/-- Represents the scenario with two cones and a marble -/
structure TwoConesScenario where
  narrow_cone : LiquidCone
  wide_cone : LiquidCone
  marble_radius : ℝ

/-- The rise of liquid level in a cone after dropping the marble -/
def liquid_rise (cone : LiquidCone) (marble_volume : ℝ) : ℝ :=
  sorry

theorem liquid_rise_ratio (scenario : TwoConesScenario) :
  scenario.narrow_cone.radius = 4 ∧
  scenario.wide_cone.radius = 8 ∧
  scenario.narrow_cone.volume = scenario.wide_cone.volume ∧
  scenario.marble_radius = 1.5 →
  let marble_volume := (4/3) * Real.pi * scenario.marble_radius^3
  (liquid_rise scenario.narrow_cone marble_volume) /
  (liquid_rise scenario.wide_cone marble_volume) = 4 := by
  sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l3122_312263


namespace NUMINAMATH_CALUDE_solve_equation_l3122_312253

theorem solve_equation : ∃ y : ℝ, (2 * y) / 3 = 30 ∧ y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3122_312253


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3122_312271

/-- The equation of the trajectory of the midpoint of a line segment -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) : 
  y₁ = 2 * x₁^2 + 1 →  -- P is on the curve y = 2x^2 + 1
  x = (x₁ + 0) / 2 →   -- x-coordinate of midpoint
  y = (y₁ + (-1)) / 2  -- y-coordinate of midpoint
  → y = 4 * x^2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3122_312271


namespace NUMINAMATH_CALUDE_trajectory_of_P_l3122_312248

/-- The trajectory of point P given two fixed points F₁ and F₂ -/
theorem trajectory_of_P (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-1, 0) →
  F₂ = (1, 0) →
  (dist P F₁ + dist P F₂) / 2 = dist F₁ F₂ / 2 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
sorry

#check trajectory_of_P

end NUMINAMATH_CALUDE_trajectory_of_P_l3122_312248


namespace NUMINAMATH_CALUDE_list_length_contradiction_l3122_312225

theorem list_length_contradiction (list_I list_II : List ℕ) : 
  (list_I = [3, 4, 8, 19]) →
  (list_II.length = list_I.length + 1) →
  (list_II.length - list_I.length = 6) →
  False :=
by sorry

end NUMINAMATH_CALUDE_list_length_contradiction_l3122_312225


namespace NUMINAMATH_CALUDE_billy_sleep_problem_l3122_312242

theorem billy_sleep_problem (x : ℝ) : 
  x + (x + 2) + (x + 2) / 2 + 3 * ((x + 2) / 2) = 30 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_billy_sleep_problem_l3122_312242


namespace NUMINAMATH_CALUDE_harvest_season_duration_l3122_312276

/-- Calculates the number of weeks in a harvest season based on weekly earnings, rent, and total savings. -/
def harvest_season_weeks (weekly_earnings : ℕ) (weekly_rent : ℕ) (total_savings : ℕ) : ℕ :=
  total_savings / (weekly_earnings - weekly_rent)

/-- Proves that the number of weeks in the harvest season is 1181 given the specified conditions. -/
theorem harvest_season_duration :
  harvest_season_weeks 491 216 324775 = 1181 := by
  sorry

end NUMINAMATH_CALUDE_harvest_season_duration_l3122_312276


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3122_312299

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 42| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3122_312299


namespace NUMINAMATH_CALUDE_election_votes_calculation_l3122_312221

/-- The total number of votes in the election -/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that candidate A received -/
def candidate_A_percentage : ℚ := 65 / 100

/-- The percentage of invalid votes -/
def invalid_votes_percentage : ℚ := 15 / 100

/-- The number of valid votes for candidate A -/
def candidate_A_valid_votes : ℕ := 309400

theorem election_votes_calculation :
  (1 - invalid_votes_percentage) * candidate_A_percentage * total_votes = candidate_A_valid_votes :=
by sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l3122_312221


namespace NUMINAMATH_CALUDE_dress_price_ratio_l3122_312259

theorem dress_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := 2/3 * selling_price
  cost_price / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dress_price_ratio_l3122_312259


namespace NUMINAMATH_CALUDE_substitution_theorem_l3122_312264

def num_players : ℕ := 15
def starting_players : ℕ := 5
def bench_players : ℕ := 10
def max_substitutions : ℕ := 4

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 5 * (11 - k) * substitution_ways k

def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

theorem substitution_theorem :
  total_substitution_ways = 5073556 ∧
  total_substitution_ways % 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_substitution_theorem_l3122_312264


namespace NUMINAMATH_CALUDE_eating_contest_l3122_312223

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ) 
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 →
  mason_hotdogs = 15 := by
  sorry

end NUMINAMATH_CALUDE_eating_contest_l3122_312223


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3122_312241

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote is y = 2x -/
  h_asymptote1 : ∀ x, asymptote1 x = 2 * x
  /-- Condition that the foci x-coordinate is 4 -/
  h_foci_x : foci_x = 4

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 16

/-- Theorem stating the equation of the other asymptote -/
theorem hyperbola_other_asymptote (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3122_312241


namespace NUMINAMATH_CALUDE_center_trajectory_of_circle_family_l3122_312239

-- Define the family of circles
def circle_family (t x y : ℝ) : Prop :=
  x^2 + y^2 - 4*t*x - 2*t*y + 3*t^2 - 4 = 0

-- Define the trajectory of centers
def center_trajectory (t x y : ℝ) : Prop :=
  x = 2*t ∧ y = t

-- Theorem statement
theorem center_trajectory_of_circle_family :
  ∀ t : ℝ, ∃ x y : ℝ,
    circle_family t x y ↔ center_trajectory t x y :=
sorry

end NUMINAMATH_CALUDE_center_trajectory_of_circle_family_l3122_312239


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3122_312238

theorem cylinder_volume_ratio : 
  let cylinder1_height : ℝ := 10
  let cylinder1_circumference : ℝ := 6
  let cylinder2_height : ℝ := 6
  let cylinder2_circumference : ℝ := 10
  let volume1 := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let volume2 := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  volume2 / volume1 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3122_312238


namespace NUMINAMATH_CALUDE_cookie_bags_l3122_312202

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l3122_312202


namespace NUMINAMATH_CALUDE_simplify_expression_l3122_312256

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3122_312256


namespace NUMINAMATH_CALUDE_partition_equivalence_l3122_312251

/-- Represents a partition of a positive integer -/
def Partition (n : ℕ) := Multiset ℕ

/-- The number of representations of n as a sum of distinct positive integers -/
def distinctSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive odd integers -/
def oddSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is repeated more than k-1 times -/
def limitedRepetitionSum (n k : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is divisible by k -/
def notDivisibleSum (n k : ℕ) : ℕ := sorry

/-- Main theorem stating the equality of representations -/
theorem partition_equivalence (n : ℕ) : 
  (∀ k : ℕ, k > 0 → limitedRepetitionSum n k = notDivisibleSum n k) ∧ 
  distinctSum n = oddSum n := by sorry

end NUMINAMATH_CALUDE_partition_equivalence_l3122_312251


namespace NUMINAMATH_CALUDE_find_k_value_l3122_312272

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l3122_312272


namespace NUMINAMATH_CALUDE_coefficient_a2_equals_56_l3122_312218

/-- Given a polynomial equality, prove that the coefficient a₂ equals 56 -/
theorem coefficient_a2_equals_56 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 
    = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) 
  → a₂ = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_equals_56_l3122_312218


namespace NUMINAMATH_CALUDE_remainder_twice_sum_first_150_l3122_312236

theorem remainder_twice_sum_first_150 : 
  (2 * (List.range 150).sum) % 10000 = 2650 := by
  sorry

end NUMINAMATH_CALUDE_remainder_twice_sum_first_150_l3122_312236


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3122_312213

/-- The number of boys who tried out for the basketball team -/
def boys_tried_out : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def girls_tried_out : ℕ := 9

/-- The number of students who got called back -/
def called_back : ℕ := 2

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 21

theorem basketball_tryouts :
  boys_tried_out = 14 ∧
  girls_tried_out + boys_tried_out = called_back + didnt_make_cut :=
by sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3122_312213


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3122_312280

/-- The range of 'a' for which the ellipse x^2 + 4(y - a)^2 = 4 and the parabola x^2 = 2y intersect -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3122_312280


namespace NUMINAMATH_CALUDE_square_nonnegative_l3122_312244

theorem square_nonnegative (x : ℚ) : x^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l3122_312244


namespace NUMINAMATH_CALUDE_farmer_apples_l3122_312291

theorem farmer_apples (apples_given_away apples_left : ℕ) 
  (h1 : apples_given_away = 88) 
  (h2 : apples_left = 39) : 
  apples_given_away + apples_left = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3122_312291


namespace NUMINAMATH_CALUDE_book_sales_theorem_l3122_312258

def monday_sales : ℕ := 15

def tuesday_sales : ℕ := 2 * monday_sales

def wednesday_sales : ℕ := tuesday_sales + (tuesday_sales / 2)

def thursday_sales : ℕ := wednesday_sales + (wednesday_sales / 2)

def friday_expected_sales : ℕ := thursday_sales + (thursday_sales / 2)

def friday_actual_sales : ℕ := friday_expected_sales + (friday_expected_sales / 4)

def saturday_sales : ℕ := (friday_expected_sales * 7) / 10

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_actual_sales + saturday_sales

theorem book_sales_theorem : total_sales = 357 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_theorem_l3122_312258


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3122_312211

theorem min_value_quadratic (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), m = 12 ∧ ∀ (a b : ℝ), a + b = 4 → 3 * a^2 + b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3122_312211


namespace NUMINAMATH_CALUDE_maria_white_towels_l3122_312233

/-- The number of white towels Maria bought -/
def white_towels (green_towels given_away remaining : ℕ) : ℕ :=
  (remaining + given_away) - green_towels

/-- Proof that Maria bought 21 white towels -/
theorem maria_white_towels : 
  white_towels 35 34 22 = 21 := by
  sorry

end NUMINAMATH_CALUDE_maria_white_towels_l3122_312233


namespace NUMINAMATH_CALUDE_max_c_value_l3122_312240

theorem max_c_value (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2*y^2 ≤ c*x*(y-x)) → 
  c ≤ 2*Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_max_c_value_l3122_312240


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3122_312200

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : a^2 + 1/b = b^2 + 1/a) : a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3122_312200


namespace NUMINAMATH_CALUDE_bird_migration_l3122_312282

theorem bird_migration (total : ℕ) (to_asia : ℕ) (difference : ℕ) : ℕ :=
  let to_africa := to_asia + difference
  to_africa

#check bird_migration 8 31 11 = 42

end NUMINAMATH_CALUDE_bird_migration_l3122_312282


namespace NUMINAMATH_CALUDE_ellipse_focus_y_axis_alpha_range_l3122_312203

/-- Represents an ellipse with equation x^2 * sin(α) - y^2 * cos(α) = 1 --/
structure Ellipse (α : Real) where
  equation : ∀ x y, x^2 * Real.sin α - y^2 * Real.cos α = 1

/-- Predicate to check if the focus of an ellipse is on the y-axis --/
def focus_on_y_axis (e : Ellipse α) : Prop :=
  1 / Real.sin α > 0 ∧ 1 / (-Real.cos α) > 0 ∧ 1 / Real.sin α < 1 / (-Real.cos α)

theorem ellipse_focus_y_axis_alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (e : Ellipse α) (h3 : focus_on_y_axis e) : 
  Real.pi / 2 < α ∧ α < 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_y_axis_alpha_range_l3122_312203


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_tangent_line_through_point_l3122_312208

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 2

theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x : ℝ), (f' 0) * x = m * x ∧ m = 2 :=
sorry

theorem tangent_line_through_point :
  ∃ (m b : ℝ), ∀ (x : ℝ),
    (∃ (x₀ : ℝ), f x₀ = m * x₀ + b ∧ f' x₀ = m) ∧
    (-1 * m + b = -3) ∧
    m = 5 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_tangent_line_through_point_l3122_312208


namespace NUMINAMATH_CALUDE_division_problem_l3122_312281

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 271 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 30 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3122_312281


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3122_312283

theorem coefficient_of_x (x y : ℝ) (some : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (some * x + 5 * y) / (x - 2 * y) = 26) :
  some = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3122_312283


namespace NUMINAMATH_CALUDE_not_divides_power_plus_one_l3122_312229

theorem not_divides_power_plus_one (n : ℕ) (h : n > 1) : ¬(2^n ∣ 3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_plus_one_l3122_312229


namespace NUMINAMATH_CALUDE_grsl_team_count_grsl_solution_l3122_312278

/-- Represents the number of teams in each group of the Greater Regional Soccer League -/
def n : ℕ := sorry

/-- The total number of games played in the league -/
def total_games : ℕ := 56

/-- The number of inter-group games played by each team in Group A -/
def inter_group_games_per_team : ℕ := 2

theorem grsl_team_count :
  n * (n - 1) + 2 * n = total_games :=
sorry

theorem grsl_solution :
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_grsl_team_count_grsl_solution_l3122_312278


namespace NUMINAMATH_CALUDE_second_train_crossing_time_l3122_312273

/-- Represents the time in seconds for two bullet trains to cross each other -/
def crossing_time : ℝ := 16.666666666666668

/-- Represents the length of each bullet train in meters -/
def train_length : ℝ := 120

/-- Represents the time in seconds for the first bullet train to cross a telegraph post -/
def first_train_time : ℝ := 10

theorem second_train_crossing_time :
  let first_train_speed := train_length / first_train_time
  let second_train_time := train_length / ((2 * train_length / crossing_time) - first_train_speed)
  second_train_time = 50 := by sorry

end NUMINAMATH_CALUDE_second_train_crossing_time_l3122_312273


namespace NUMINAMATH_CALUDE_rectangles_count_l3122_312269

/-- The number of rectangles in an n×n square grid -/
def rectangles_in_square (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The number of rectangles in the given arrangement of three n×n square grids -/
def rectangles_in_three_squares (n : ℕ) : ℕ := 
  n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2

theorem rectangles_count (n : ℕ) (h : n > 0) : 
  (rectangles_in_square n = (n * (n + 1) / 2) ^ 2) ∧ 
  (rectangles_in_three_squares n = n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_rectangles_count_l3122_312269


namespace NUMINAMATH_CALUDE_reseat_ten_women_l3122_312222

def reseat_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => reseat_ways (k + 2) + reseat_ways (k + 1)

theorem reseat_ten_women :
  reseat_ways 10 = 89 :=
by sorry

end NUMINAMATH_CALUDE_reseat_ten_women_l3122_312222


namespace NUMINAMATH_CALUDE_iron_chain_links_count_l3122_312266

/-- Represents a piece of an iron chain -/
structure ChainPiece where
  length : ℝ
  links : ℕ

/-- Calculates the length of a chain piece given the number of links and internal diameter -/
def chainLength (links : ℕ) (internalDiameter : ℝ) : ℝ :=
  (links : ℝ) * internalDiameter + 1

theorem iron_chain_links_count :
  let shortPiece : ChainPiece := ⟨22, 9⟩
  let longPiece : ChainPiece := ⟨36, 15⟩
  let internalDiameter : ℝ := 7/3

  (longPiece.links = shortPiece.links + 6) ∧
  (chainLength shortPiece.links internalDiameter = shortPiece.length) ∧
  (chainLength longPiece.links internalDiameter = longPiece.length) :=
by
  sorry

end NUMINAMATH_CALUDE_iron_chain_links_count_l3122_312266


namespace NUMINAMATH_CALUDE_rockets_won_38_games_l3122_312214

/-- Represents the number of wins for each team -/
structure TeamWins where
  sharks : ℕ
  dolphins : ℕ
  rockets : ℕ
  wolves : ℕ
  comets : ℕ

/-- The set of possible win numbers -/
def winNumbers : Finset ℕ := {28, 33, 38, 43}

/-- The conditions of the problem -/
def validTeamWins (tw : TeamWins) : Prop :=
  tw.sharks > tw.dolphins ∧
  tw.rockets > tw.wolves ∧
  tw.comets > tw.rockets ∧
  tw.wolves > 25 ∧
  tw.sharks ∈ winNumbers ∧
  tw.dolphins ∈ winNumbers ∧
  tw.rockets ∈ winNumbers ∧
  tw.wolves ∈ winNumbers ∧
  tw.comets ∈ winNumbers

/-- Theorem: Given the conditions, the Rockets won 38 games -/
theorem rockets_won_38_games (tw : TeamWins) (h : validTeamWins tw) : tw.rockets = 38 := by
  sorry

end NUMINAMATH_CALUDE_rockets_won_38_games_l3122_312214


namespace NUMINAMATH_CALUDE_expand_product_l3122_312260

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3122_312260


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3122_312270

def vector_a (m : ℝ) : ℝ × ℝ := (3, m)
def vector_b : ℝ × ℝ := (2, -4)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (vector_a m) vector_b → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3122_312270


namespace NUMINAMATH_CALUDE_equal_weight_implies_all_genuine_l3122_312293

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- The total number of coins. -/
def total_coins : ℕ := 12

/-- The number of genuine coins. -/
def genuine_coins : ℕ := 9

/-- The number of counterfeit coins. -/
def counterfeit_coins : ℕ := 3

/-- A function that returns the weight of a coin. -/
def weight : Coin → ℝ
| Coin.genuine => 1
| Coin.counterfeit => 2  -- Counterfeit coins are heavier

/-- A type representing a selection of coins. -/
def CoinSelection := Fin 6 → Coin

/-- The property that all coins in a selection are genuine. -/
def all_genuine (selection : CoinSelection) : Prop :=
  ∀ i, selection i = Coin.genuine

/-- The property that the weights of two sets of coins are equal. -/
def weights_equal (selection : CoinSelection) : Prop :=
  (weight (selection 0) + weight (selection 1) + weight (selection 2)) =
  (weight (selection 3) + weight (selection 4) + weight (selection 5))

/-- The main theorem to be proved. -/
theorem equal_weight_implies_all_genuine :
  ∀ (selection : CoinSelection),
  weights_equal selection → all_genuine selection :=
by sorry

end NUMINAMATH_CALUDE_equal_weight_implies_all_genuine_l3122_312293


namespace NUMINAMATH_CALUDE_coordinate_plane_conditions_l3122_312235

-- Define a point on the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions and their corresponding geometric interpretations
theorem coordinate_plane_conditions (p : Point) :
  (p.x = 3 → p ∈ {q : Point | q.x = 3}) ∧
  (p.x < 3 → p ∈ {q : Point | q.x < 3}) ∧
  (p.x > 3 → p ∈ {q : Point | q.x > 3}) ∧
  (p.y = 2 → p ∈ {q : Point | q.y = 2}) ∧
  (p.y > 2 → p ∈ {q : Point | q.y > 2}) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_plane_conditions_l3122_312235


namespace NUMINAMATH_CALUDE_billiard_ball_weight_l3122_312295

/-- Given a box containing 6 equally weighted billiard balls, where the total weight
    of the box with balls is 1.82 kg and the empty box weighs 0.5 kg,
    prove that the weight of one billiard ball is 0.22 kg. -/
theorem billiard_ball_weight
  (num_balls : ℕ)
  (total_weight : ℝ)
  (empty_box_weight : ℝ)
  (h1 : num_balls = 6)
  (h2 : total_weight = 1.82)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_balls = 0.22 := by
  sorry

#eval (1.82 - 0.5) / 6

end NUMINAMATH_CALUDE_billiard_ball_weight_l3122_312295


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l3122_312298

-- Define the curve
def f (x : ℝ) : ℝ := 3*x^2 - 4*x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 * M.1 - 4)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  ∀ (x y : ℝ), line_equation x y → (y - P.2) = m * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l3122_312298


namespace NUMINAMATH_CALUDE_petes_ten_dollar_bills_l3122_312220

theorem petes_ten_dollar_bills (
  total_owed : ℕ)
  (twenty_dollar_bills : ℕ)
  (bottle_refund : ℚ)
  (bottles_to_return : ℕ)
  (h1 : total_owed = 90)
  (h2 : twenty_dollar_bills = 2)
  (h3 : bottle_refund = 1/2)
  (h4 : bottles_to_return = 20)
  : ∃ (ten_dollar_bills : ℕ),
    ten_dollar_bills = 4 ∧
    20 * twenty_dollar_bills + 10 * ten_dollar_bills + (bottle_refund * bottles_to_return) = total_owed :=
by sorry

end NUMINAMATH_CALUDE_petes_ten_dollar_bills_l3122_312220


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_empty_solution_l3122_312262

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 4} = Set.Icc 0 4 := by sorry

-- Part II
theorem range_of_a_for_empty_solution :
  {a : ℝ | ∀ x, f a x ≥ 2} = Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_empty_solution_l3122_312262


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3122_312224

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem perpendicular_vectors_m_value :
  (∀ m : ℝ, vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) →
  (∃ m : ℝ, vector_b m = (6, 3)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3122_312224


namespace NUMINAMATH_CALUDE_triangle_inequality_l3122_312287

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  (3 = a ∧ 7 = b) → 4 < c ∧ c < 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3122_312287


namespace NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l3122_312261

open Real

/-- The function f(x) = x^2 + 2x - a(ln x + x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - a*(log x + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2 - a*(1/x + 1)

theorem f_derivative_positive_at_midpoint (a c : ℝ) (x₁ x₂ : ℝ) 
  (hx₁ : f a x₁ = c) (hx₂ : f a x₂ = c) (hne : x₁ ≠ x₂) :
  f_derivative a ((x₁ + x₂) / 2) > 0 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l3122_312261


namespace NUMINAMATH_CALUDE_rational_absolute_value_equality_l3122_312205

theorem rational_absolute_value_equality (a : ℚ) : 
  |(-3 - a)| = 3 + |a| → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_equality_l3122_312205


namespace NUMINAMATH_CALUDE_translation_correctness_given_translations_correct_l3122_312234

/-- Represents a word in either Russian or Kurdish -/
structure Word :=
  (value : String)

/-- Represents a sentence in either Russian or Kurdish -/
structure Sentence :=
  (words : List Word)

/-- Defines the rules for Kurdish sentence structure -/
def kurdishSentenceStructure (s : Sentence) : Prop :=
  -- The predicate is at the end of the sentence
  -- The subject starts the sentence, followed by the object
  -- Noun-adjective constructions follow the "S (adjective determinant) O (determined word)" structure
  -- The determined word takes the suffix "e"
  sorry

/-- Translates a Russian sentence to Kurdish -/
def translateToKurdish (russianSentence : Sentence) : Sentence :=
  sorry

/-- Verifies that the translated sentence follows Kurdish sentence structure -/
theorem translation_correctness (russianSentence : Sentence) :
  kurdishSentenceStructure (translateToKurdish russianSentence) :=
  sorry

/-- Specific sentences from the problem -/
def sentence1 : Sentence := sorry -- "The lazy lion eats meat"
def sentence2 : Sentence := sorry -- "The healthy poor man carries the burden"
def sentence3 : Sentence := sorry -- "The bull of the poor man does not understand the poor man"

/-- Verifies the correctness of the given translations -/
theorem given_translations_correct :
  kurdishSentenceStructure (translateToKurdish sentence1) ∧
  kurdishSentenceStructure (translateToKurdish sentence2) ∧
  kurdishSentenceStructure (translateToKurdish sentence3) :=
  sorry

end NUMINAMATH_CALUDE_translation_correctness_given_translations_correct_l3122_312234


namespace NUMINAMATH_CALUDE_bike_ride_distance_l3122_312217

/-- Calculates the total distance of a 3-hour bike ride given the conditions -/
theorem bike_ride_distance (second_hour : ℝ) 
  (h1 : second_hour = 12)
  (h2 : second_hour = 1.2 * (second_hour / 1.2))
  (h3 : second_hour * 1.25 = 15) : 
  (second_hour / 1.2) + second_hour + (second_hour * 1.25) = 37 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l3122_312217


namespace NUMINAMATH_CALUDE_florist_initial_roses_l3122_312245

/-- Represents the number of roses picked in the first round -/
def first_pick : ℝ := 16.0

/-- Represents the number of roses picked in the second round -/
def second_pick : ℝ := 19.0

/-- Represents the total number of roses after all picking -/
def total_roses : ℕ := 72

/-- Calculates the initial number of roses the florist had -/
def initial_roses : ℝ := total_roses - (first_pick + second_pick)

/-- Theorem stating that the initial number of roses was 37 -/
theorem florist_initial_roses : initial_roses = 37 := by sorry

end NUMINAMATH_CALUDE_florist_initial_roses_l3122_312245


namespace NUMINAMATH_CALUDE_first_level_spots_l3122_312226

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The parking area satisfies the given conditions -/
def validParkingArea (p : ParkingArea) : Prop :=
  p.second = p.first + 7 ∧
  p.third = p.second + 6 ∧
  p.fourth = 14 ∧
  p.first + p.second + p.third + p.fourth = 46

theorem first_level_spots (p : ParkingArea) (h : validParkingArea p) : p.first = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_level_spots_l3122_312226


namespace NUMINAMATH_CALUDE_shorter_diagonal_of_parallelepiped_l3122_312255

/-- Represents a parallelepiped with a rhombus base -/
structure Parallelepiped where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_angle : ℝ
  section_area : ℝ

/-- Theorem: The shorter diagonal of the base rhombus in the given parallelepiped is 60 -/
theorem shorter_diagonal_of_parallelepiped (p : Parallelepiped) 
  (h1 : p.base_side = 60)
  (h2 : p.lateral_edge = 80)
  (h3 : p.lateral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : p.section_area = 7200) :
  ∃ (shorter_diagonal : ℝ), shorter_diagonal = 60 :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_of_parallelepiped_l3122_312255


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l3122_312210

theorem hyperbola_eccentricity_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (mx^2 - ny^2 = 1 ∧ (m + n) / n = 4) → m / n = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l3122_312210


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3122_312228

theorem complex_magnitude_squared (z₁ z₂ : ℂ) :
  let z₁ : ℂ := 3 * Real.sqrt 2 - 5*I
  let z₂ : ℂ := 2 * Real.sqrt 5 + 4*I
  ‖z₁ * z₂‖^2 = 1548 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3122_312228


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l3122_312216

/-- Proves that a rabbit running at a constant speed of 6 miles per hour will take 20 minutes to travel 2 miles. -/
theorem rabbit_travel_time :
  let rabbit_speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 2 -- miles
  let time_in_hours : ℝ := distance / rabbit_speed
  let time_in_minutes : ℝ := time_in_hours * 60
  time_in_minutes = 20 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l3122_312216


namespace NUMINAMATH_CALUDE_inequality_of_four_terms_l3122_312274

theorem inequality_of_four_terms (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + x)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_of_four_terms_l3122_312274


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3122_312284

/-- The distance between the vertices of a hyperbola given by the equation
    16x^2 + 64x - 4y^2 + 8y + 36 = 0 is 1. -/
theorem hyperbola_vertices_distance :
  let f : ℝ → ℝ → ℝ := fun x y => 16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    (∀ x y, f x y = 0 ↔ 4 * (x + 2)^2 - (y - 1)^2 = 1) ∧
    (x₁, y₁) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₂, y₂) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    ∀ x y, f x y = 0 → (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3122_312284


namespace NUMINAMATH_CALUDE_perpendicular_median_triangle_sides_l3122_312201

/-- A triangle with sides x, y, and z, where two medians are mutually perpendicular -/
structure PerpendicularMedianTriangle where
  x : ℕ
  y : ℕ
  z : ℕ
  perp_medians : x^2 + y^2 = 5 * z^2

/-- The theorem stating that the triangle with perpendicular medians and integer sides has sides 22, 19, and 13 -/
theorem perpendicular_median_triangle_sides :
  ∀ t : PerpendicularMedianTriangle, t.x = 22 ∧ t.y = 19 ∧ t.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_median_triangle_sides_l3122_312201


namespace NUMINAMATH_CALUDE_additional_workers_for_earlier_completion_l3122_312231

/-- Calculates the number of additional workers needed to complete a task earlier -/
def additional_workers (original_days : ℕ) (actual_days : ℕ) (original_workers : ℕ) : ℕ :=
  ⌊(original_workers * original_days / actual_days - original_workers : ℚ)⌋.toNat

/-- Proves that 6 additional workers are needed to complete the task 3 days earlier -/
theorem additional_workers_for_earlier_completion :
  additional_workers 10 7 15 = 6 := by
  sorry

#eval additional_workers 10 7 15

end NUMINAMATH_CALUDE_additional_workers_for_earlier_completion_l3122_312231
