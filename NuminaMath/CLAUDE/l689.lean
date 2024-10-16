import Mathlib

namespace NUMINAMATH_CALUDE_divisors_of_1442_l689_68909

theorem divisors_of_1442 :
  let n : ℕ := 1442
  let divisors : Finset ℕ := {1, 11, 131, 1442}
  (∀ (d : ℕ), d ∣ n ↔ d ∈ divisors) ∧
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ n = p * q) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_1442_l689_68909


namespace NUMINAMATH_CALUDE_white_towels_count_l689_68971

def green_towels : ℕ := 35
def towels_given_away : ℕ := 34
def towels_remaining : ℕ := 22

theorem white_towels_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_white_towels_count_l689_68971


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l689_68919

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l689_68919


namespace NUMINAMATH_CALUDE_increase_by_percentage_l689_68913

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 80 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 120 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l689_68913


namespace NUMINAMATH_CALUDE_bill_toilet_paper_supply_l689_68965

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_days (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) (rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last for 20,000 days -/
theorem bill_toilet_paper_supply : toilet_paper_days 3 5 1000 300 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_bill_toilet_paper_supply_l689_68965


namespace NUMINAMATH_CALUDE_equation_solution_l689_68954

theorem equation_solution : 
  ∃ y : ℝ, (3 / y + (4 / y) / (6 / y) = 1.5) ∧ y = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l689_68954


namespace NUMINAMATH_CALUDE_inequality_proof_l689_68972

theorem inequality_proof (a x : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hx : 0 ≤ x ∧ x ≤ π) :
  (2 * a - 1) * Real.sin x + (1 - a) * Real.sin ((1 - a) * x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l689_68972


namespace NUMINAMATH_CALUDE_starting_elevation_l689_68973

/-- Calculates the starting elevation of a person climbing a hill --/
theorem starting_elevation (final_elevation horizontal_distance : ℝ) : 
  final_elevation = 1450 ∧ 
  horizontal_distance = 2700 → 
  final_elevation - (horizontal_distance / 2) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_starting_elevation_l689_68973


namespace NUMINAMATH_CALUDE_P_investment_theorem_l689_68947

-- Define the investments and profit ratio
def Q_investment : ℕ := 60000
def profit_ratio_P : ℕ := 2
def profit_ratio_Q : ℕ := 3

-- Theorem to prove P's investment
theorem P_investment_theorem :
  ∃ P_investment : ℕ,
    P_investment * profit_ratio_Q = Q_investment * profit_ratio_P ∧
    P_investment = 40000 := by
  sorry

end NUMINAMATH_CALUDE_P_investment_theorem_l689_68947


namespace NUMINAMATH_CALUDE_expression_value_l689_68933

theorem expression_value : 
  let a := 2015
  let b := 2016
  (a^3 - 3*a^2*b + 5*a*b^2 - b^3 + 4) / (a*b) = 4032 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l689_68933


namespace NUMINAMATH_CALUDE_pear_seed_average_l689_68945

theorem pear_seed_average (total_seeds : ℕ) (apple_seeds : ℕ) (grape_seeds : ℕ)
  (num_apples : ℕ) (num_pears : ℕ) (num_grapes : ℕ) (seeds_needed : ℕ) :
  total_seeds = 60 →
  apple_seeds = 6 →
  grape_seeds = 3 →
  num_apples = 4 →
  num_pears = 3 →
  num_grapes = 9 →
  seeds_needed = 3 →
  ∃ (pear_seeds : ℕ), pear_seeds = 2 ∧
    num_apples * apple_seeds + num_pears * pear_seeds + num_grapes * grape_seeds = total_seeds - seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_pear_seed_average_l689_68945


namespace NUMINAMATH_CALUDE_darius_score_l689_68935

/-- Represents the scores of Darius, Matt, and Marius in a table football game. -/
structure TableFootballScores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (scores : TableFootballScores) : Prop :=
  scores.marius = scores.darius + 3 ∧
  scores.matt = scores.darius + 5 ∧
  scores.darius + scores.matt + scores.marius = 38

/-- Theorem stating that under the given conditions, Darius scored 10 points. -/
theorem darius_score (scores : TableFootballScores) 
  (h : game_conditions scores) : scores.darius = 10 := by
  sorry

end NUMINAMATH_CALUDE_darius_score_l689_68935


namespace NUMINAMATH_CALUDE_caz_at_position_p_l689_68996

-- Define the type for positions in the gallery
inductive Position
| P
| Other

-- Define the type for people in the gallery
inductive Person
| Ali
| Bea
| Caz
| Dan

-- Define the visibility relation
def CanSee (a b : Person) : Prop := sorry

-- Define the position of a person
def IsAt (p : Person) (pos : Position) : Prop := sorry

-- State the theorem
theorem caz_at_position_p :
  -- Conditions
  (∀ x, x ≠ Person.Ali → ¬CanSee Person.Ali x) →
  (CanSee Person.Bea Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Bea x) →
  (CanSee Person.Caz Person.Bea) →
  (CanSee Person.Caz Person.Dan) →
  (∀ x, x ≠ Person.Bea ∧ x ≠ Person.Dan → ¬CanSee Person.Caz x) →
  (CanSee Person.Dan Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Dan x) →
  -- Conclusion
  IsAt Person.Caz Position.P :=
by sorry

end NUMINAMATH_CALUDE_caz_at_position_p_l689_68996


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l689_68959

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve 
  (a b c p r : ℝ) : 
  let q := a * p^2 + b * p + c
  let s := a * r^2 + b * r + c
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + (a * (r + p) + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l689_68959


namespace NUMINAMATH_CALUDE_monica_reading_plan_l689_68918

def books_last_year : ℕ := 16
def books_this_year : ℕ := 2 * books_last_year
def books_next_year : ℕ := 69

theorem monica_reading_plan :
  books_next_year - (2 * books_this_year) = 5 := by sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l689_68918


namespace NUMINAMATH_CALUDE_max_area_isosceles_trapezoidal_canal_l689_68967

/-- 
Given an isosceles trapezoidal canal where the legs are equal to the smaller base,
this theorem states that the cross-sectional area is maximized when the angle of 
inclination of the legs is π/3 radians.
-/
theorem max_area_isosceles_trapezoidal_canal :
  ∀ (a : ℝ) (α : ℝ), 
  0 < a → 
  0 < α → 
  α < π / 2 →
  let S := a^2 * (1 + Real.cos α) * Real.sin α
  ∀ (β : ℝ), 0 < β → β < π / 2 → 
  a^2 * (1 + Real.cos β) * Real.sin β ≤ S →
  α = π / 3 :=
by sorry


end NUMINAMATH_CALUDE_max_area_isosceles_trapezoidal_canal_l689_68967


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l689_68941

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r)^2 + h^2) = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l689_68941


namespace NUMINAMATH_CALUDE_not_necessarily_zero_l689_68924

-- Define a vector space V
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (O A B C : V)

-- Define the theorem
theorem not_necessarily_zero : 
  ¬ ∀ (O A B C : V), (A - O) + (C - O) + (O - B) + (O - C) = 0 :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_zero_l689_68924


namespace NUMINAMATH_CALUDE_pyramid_base_area_l689_68968

theorem pyramid_base_area (slant_height height : ℝ) :
  slant_height = 5 →
  height = 7 →
  ∃ (side_length : ℝ), 
    side_length ^ 2 + slant_height ^ 2 = height ^ 2 ∧
    (side_length ^ 2) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_area_l689_68968


namespace NUMINAMATH_CALUDE_taxi_truck_speed_ratio_l689_68998

/-- Given a truck that travels 2.1 km in 1 minute and a taxi that travels 10.5 km in 4 minutes,
    prove that the taxi is 1.25 times faster than the truck. -/
theorem taxi_truck_speed_ratio :
  let truck_speed := 2.1 -- km per minute
  let taxi_speed := 10.5 / 4 -- km per minute
  taxi_speed / truck_speed = 1.25 := by sorry

end NUMINAMATH_CALUDE_taxi_truck_speed_ratio_l689_68998


namespace NUMINAMATH_CALUDE_polynomial_composition_pairs_l689_68902

theorem polynomial_composition_pairs :
  ∀ (a b : ℝ),
    (∃ (P : ℝ → ℝ),
      (∀ x, P (P x) = x^4 - 8*x^3 + a*x^2 + b*x + 40) ∧
      (∃ (c d : ℝ), ∀ x, P x = x^2 + c*x + d)) ↔
    ((a = 28 ∧ b = -48) ∨ (a = 2 ∧ b = 56)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_pairs_l689_68902


namespace NUMINAMATH_CALUDE_triangle_angle_c_is_sixty_degrees_l689_68931

theorem triangle_angle_c_is_sixty_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sin : Real.sin A = 2 * Real.sin B)
  (h_sum : a + b = Real.sqrt 3 * c) :
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_is_sixty_degrees_l689_68931


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l689_68920

theorem lcm_of_ratio_and_hcf (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → a * 4 = b * 3 → Nat.gcd a b = 8 → Nat.lcm a b = 96 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l689_68920


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l689_68921

theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_cross_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_cross_time = 360 →
  let train_speed := train_length / signal_post_time
  let bridge_only_time := bridge_cross_time - signal_post_time
  let bridge_length := train_speed * bridge_only_time
  bridge_length = 4800 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l689_68921


namespace NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l689_68994

theorem difference_from_sum_and_difference_of_squares
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l689_68994


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l689_68948

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day n days after the given day -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

/-- The main theorem -/
theorem tomorrow_is_saturday 
  (h : addDays (addDays DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) : 
  nextDay (addDays DayOfWeek.Wednesday 2) = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l689_68948


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l689_68982

/-- Represents the cost of fencing an irregular pentagonal field --/
def fencing_cost (side_a side_b side_c side_d side_e : ℝ) (cost_per_meter : ℝ) : ℝ :=
  (side_a + side_b + side_c + side_d + side_e) * cost_per_meter

/-- Theorem stating the total cost of fencing the given irregular pentagonal field --/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 35 52 66 40 3 = 705 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l689_68982


namespace NUMINAMATH_CALUDE_player1_wins_533_player1_wins_1000_l689_68958

/-- A game where two players alternately write 1 or 2, and the player who makes the sum reach or exceed the target loses. -/
def Game (target : ℕ) := Unit

/-- A strategy for playing the game. -/
def Strategy (target : ℕ) := Unit

/-- Determines if a strategy is winning for Player 1. -/
def is_winning_strategy (target : ℕ) (s : Strategy target) : Prop := sorry

/-- Player 1 has a winning strategy for the game with target 533. -/
theorem player1_wins_533 : ∃ s : Strategy 533, is_winning_strategy 533 s := sorry

/-- Player 1 has a winning strategy for the game with target 1000. -/
theorem player1_wins_1000 : ∃ s : Strategy 1000, is_winning_strategy 1000 s := sorry

end NUMINAMATH_CALUDE_player1_wins_533_player1_wins_1000_l689_68958


namespace NUMINAMATH_CALUDE_chess_match_max_ab_l689_68987

theorem chess_match_max_ab (a b c : ℝ) : 
  0 ≤ a ∧ a < 1 ∧
  0 ≤ b ∧ b < 1 ∧
  0 ≤ c ∧ c < 1 ∧
  a + b + c = 1 ∧
  3*a + b = 1 →
  a * b ≤ 1/12 := by
sorry

end NUMINAMATH_CALUDE_chess_match_max_ab_l689_68987


namespace NUMINAMATH_CALUDE_cos_24_minus_cos_48_l689_68938

theorem cos_24_minus_cos_48 : Real.cos (24 * Real.pi / 180) - Real.cos (48 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_minus_cos_48_l689_68938


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l689_68929

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l689_68929


namespace NUMINAMATH_CALUDE_triangle_proof_l689_68912

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.b = Real.sqrt 7 ∧
  t.a + t.c = 4

theorem triangle_proof (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 4 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l689_68912


namespace NUMINAMATH_CALUDE_factorization_identities_l689_68903

theorem factorization_identities (a b x m : ℝ) :
  (2 * a^2 - 2*a*b = 2*a*(a-b)) ∧
  (2 * x^2 - 18 = 2*(x+3)*(x-3)) ∧
  (-3*m*a^3 + 6*m*a^2 - 3*m*a = -3*m*a*(a-1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l689_68903


namespace NUMINAMATH_CALUDE_intersection_subset_l689_68926

theorem intersection_subset (A B : Set α) : A ∩ B = B → B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_intersection_subset_l689_68926


namespace NUMINAMATH_CALUDE_triangulation_has_120_triangle_l689_68910

/-- A triangulation of a triangle -/
structure Triangulation :=
  (vertices : Set ℝ × ℝ)
  (edges : Set (ℝ × ℝ × ℝ × ℝ))
  (triangles : Set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ))

/-- The original triangle in a triangulation -/
def originalTriangle (t : Triangulation) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Check if all angles in a triangle are not exceeding 120° -/
def allAnglesWithin120 (triangle : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem statement -/
theorem triangulation_has_120_triangle 
  (t : Triangulation) 
  (h : allAnglesWithin120 (originalTriangle t)) :
  ∃ triangle ∈ t.triangles, allAnglesWithin120 triangle :=
sorry

end NUMINAMATH_CALUDE_triangulation_has_120_triangle_l689_68910


namespace NUMINAMATH_CALUDE_local_max_condition_l689_68993

/-- The function f(x) = x(x-m)² has a local maximum at x = 1 if and only if m = 3 -/
theorem local_max_condition (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), x * (x - m)^2 ≤ 1 * (1 - m)^2) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_local_max_condition_l689_68993


namespace NUMINAMATH_CALUDE_probability_two_pairs_l689_68953

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The number of ways to choose which two dice will form each pair -/
def numWaysToFormPairs : ℕ := Nat.choose numDice 2

/-- The probability of rolling exactly two pairs of matching numbers
    when four standard six-sided dice are tossed simultaneously -/
theorem probability_two_pairs :
  (numWaysToFormPairs : ℚ) * (1 : ℚ) * (1 / numSides : ℚ) * ((numSides - 1) / numSides : ℚ) * (1 / numSides : ℚ) = 5 / 36 := by
  sorry


end NUMINAMATH_CALUDE_probability_two_pairs_l689_68953


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l689_68963

theorem least_sum_of_four_primes (n : ℕ) : 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄) →
  n ≥ 60 :=
by
  sorry

#check least_sum_of_four_primes

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l689_68963


namespace NUMINAMATH_CALUDE_polygon_construction_possible_l689_68990

/-- Represents a line segment with a fixed length -/
structure LineSegment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List LineSegment
  isValid : Bool  -- Indicates if the polygon is valid (closed and non-self-intersecting)

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if it's possible to construct a polygon with given area using given line segments -/
def canConstructPolygon (segments : List LineSegment) (targetArea : ℝ) : Prop :=
  ∃ (p : Polygon), p.segments = segments ∧ p.isValid ∧ calculateArea p = targetArea

theorem polygon_construction_possible :
  let segments := List.replicate 12 { length := 2 }
  canConstructPolygon segments 16 := by
  sorry

end NUMINAMATH_CALUDE_polygon_construction_possible_l689_68990


namespace NUMINAMATH_CALUDE_digit_55_is_2_l689_68960

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : Nat := 16

/-- The 55th digit after the decimal point in the decimal representation of 1/17 -/
def digit_55 : Nat := decimal_rep_1_17[(55 - 1) % repeat_length]

theorem digit_55_is_2 : digit_55 = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_55_is_2_l689_68960


namespace NUMINAMATH_CALUDE_least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l689_68942

theorem least_hour_square_remainder (n : ℕ) : n > 9 ∧ n % 12 = (n^2) % 12 → n ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 15 % 12 = (15^2) % 12 := by
  sorry

theorem fifteen_is_least : ∀ m : ℕ, m > 9 ∧ m % 12 = (m^2) % 12 → m ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l689_68942


namespace NUMINAMATH_CALUDE_factor_expression_l689_68979

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l689_68979


namespace NUMINAMATH_CALUDE_cost_difference_is_1267_50_l689_68957

def initial_order : ℝ := 20000

def scheme1_discount1 : ℝ := 0.25
def scheme1_discount2 : ℝ := 0.15
def scheme1_discount3 : ℝ := 0.05

def scheme2_discount1 : ℝ := 0.20
def scheme2_discount2 : ℝ := 0.10
def scheme2_discount3 : ℝ := 0.05
def scheme2_rebate : ℝ := 300

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme1_discount1) scheme1_discount2) scheme1_discount3

def scheme2_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme2_discount1) scheme2_discount2) scheme2_discount3 - scheme2_rebate

theorem cost_difference_is_1267_50 :
  scheme1_final_cost - scheme2_final_cost = 1267.50 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_is_1267_50_l689_68957


namespace NUMINAMATH_CALUDE_expression_classification_l689_68928

/-- Represents an algebraic expression -/
inductive AlgebraicExpr
  | Constant (n : ℚ)
  | Variable (name : String)
  | Product (coef : ℚ) (terms : List (String × ℕ))

/-- Calculates the degree of an algebraic expression -/
def degree (expr : AlgebraicExpr) : ℕ :=
  match expr with
  | AlgebraicExpr.Constant _ => 0
  | AlgebraicExpr.Variable _ => 1
  | AlgebraicExpr.Product _ terms => terms.foldl (fun acc (_, power) => acc + power) 0

/-- Checks if an expression contains variables -/
def hasVariables (expr : AlgebraicExpr) : Bool :=
  match expr with
  | AlgebraicExpr.Constant _ => false
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product _ terms => terms.length > 0

def expressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (-2) [("a", 1)],
  AlgebraicExpr.Product 3 [("a", 1), ("b", 2)],
  AlgebraicExpr.Constant (2/3),
  AlgebraicExpr.Product 3 [("a", 2), ("b", 1)],
  AlgebraicExpr.Product (-3) [("a", 3)],
  AlgebraicExpr.Constant 25,
  AlgebraicExpr.Product (-(3/4)) [("b", 1)]
]

theorem expression_classification :
  (expressions.filter hasVariables).length = 5 ∧
  (expressions.filter (fun e => ¬hasVariables e)).length = 2 ∧
  (expressions.filter (fun e => degree e = 0)).length = 2 ∧
  (expressions.filter (fun e => degree e = 1)).length = 2 ∧
  (expressions.filter (fun e => degree e = 3)).length = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_classification_l689_68928


namespace NUMINAMATH_CALUDE_miles_left_to_run_l689_68950

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Macy's daily running distance in miles -/
def daily_distance : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Theorem stating the number of miles left for Macy to run after 6 days -/
theorem miles_left_to_run : weekly_goal - (daily_distance * days_run) = 6 := by
  sorry

end NUMINAMATH_CALUDE_miles_left_to_run_l689_68950


namespace NUMINAMATH_CALUDE_division_to_ratio_l689_68991

theorem division_to_ratio (a b : ℝ) (h : a / b = 0.4) : a / b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_division_to_ratio_l689_68991


namespace NUMINAMATH_CALUDE_marble_problem_l689_68916

theorem marble_problem (x : ℝ) : 
  x + 3*x + 6*x + 24*x = 156 → x = 156/34 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l689_68916


namespace NUMINAMATH_CALUDE_bike_ride_distance_l689_68966

/-- Calculates the total distance of a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 18)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 22.5) :
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 55.5 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l689_68966


namespace NUMINAMATH_CALUDE_cube_difference_positive_l689_68904

theorem cube_difference_positive (a b : ℝ) : a > b → a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l689_68904


namespace NUMINAMATH_CALUDE_octal_734_eq_476_l689_68932

-- Define the octal number as a list of digits
def octal_number : List Nat := [7, 3, 4]

-- Define the function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_734_eq_476 :
  octal_to_decimal octal_number = 476 := by
  sorry

end NUMINAMATH_CALUDE_octal_734_eq_476_l689_68932


namespace NUMINAMATH_CALUDE_cookie_pattern_holds_l689_68936

/-- Represents the number of cookies on each plate -/
def cookie_sequence : Fin 6 → ℕ
  | 0 => 5
  | 1 => 7
  | 2 => 10
  | 3 => 14
  | 4 => 19
  | 5 => 25

/-- The difference between consecutive cookie counts increases by 1 each time -/
def increasing_difference (seq : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 4, seq (i + 1) - seq i = seq (i + 2) - seq (i + 1) + 1

theorem cookie_pattern_holds :
  increasing_difference cookie_sequence ∧ cookie_sequence 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookie_pattern_holds_l689_68936


namespace NUMINAMATH_CALUDE_average_price_per_book_l689_68908

theorem average_price_per_book (books1 : ℕ) (price1 : ℝ) (books2 : ℕ) (price2 : ℝ) 
  (h1 : books1 = 65) (h2 : price1 = 1080) (h3 : books2 = 55) (h4 : price2 = 840) :
  (price1 + price2) / (books1 + books2 : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l689_68908


namespace NUMINAMATH_CALUDE_log_equation_solution_l689_68940

theorem log_equation_solution (x : ℝ) (h : Real.log 125 / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = a / b ∧ a ≠ 0 ∧ b > 0 ∧ (∀ n : ℕ, n > 1 → (a : ℝ) / b ≠ n^2 ∧ (a : ℝ) / b ≠ n^3) ∧ ¬∃ (n : ℤ), (a : ℝ) / b = n) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l689_68940


namespace NUMINAMATH_CALUDE_min_value_of_f_l689_68964

noncomputable def f (x : ℝ) := (Real.exp x - 1)^2 + (Real.exp 1 - x - 1)^2

theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l689_68964


namespace NUMINAMATH_CALUDE_company_optimization_l689_68969

/-- Represents the company's employee structure and profit model -/
structure Company where
  total_employees : ℕ
  initial_profit_per_employee : ℝ
  transferred_employees : ℕ
  tertiary_profit_factor : ℝ
  profit_increase_factor : ℝ

/-- Calculates the maximum number of employees that can be transferred while maintaining total profit -/
def max_transferable_employees (c : Company) : ℕ := sorry

/-- Defines the range of tertiary profit factor that satisfies the profit condition -/
def valid_tertiary_profit_range (c : Company) : Set ℝ := sorry

/-- Main theorem stating the maximum transferable employees and valid profit range -/
theorem company_optimization (c : Company) 
  (h1 : c.total_employees = 1000)
  (h2 : c.initial_profit_per_employee = 100000)
  (h3 : c.profit_increase_factor = 0.002)
  (h4 : c.tertiary_profit_factor > 0) :
  (max_transferable_employees c = 500) ∧ 
  (valid_tertiary_profit_range c = {a | 0 < a ∧ a ≤ 5}) := by sorry

end NUMINAMATH_CALUDE_company_optimization_l689_68969


namespace NUMINAMATH_CALUDE_triangle_problem_l689_68911

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
this theorem proves that under certain conditions, the angle C is 60° and 
the sides have specific lengths.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  A > 0 → B > 0 → C > 0 →  -- Positive angles
  a > b →  -- Given condition
  a * (Real.sqrt 3 * Real.tan B - 1) = 
    (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C) →  -- Given equation
  a + b + c = 20 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →  -- Area condition
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l689_68911


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l689_68923

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m ∧
  ∀ (x' y' : ℝ), circle1 x' y' ∧ circle2 x' y' m → (x = x' ∧ y = y')

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → (m = 3 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l689_68923


namespace NUMINAMATH_CALUDE_doll_collection_l689_68937

theorem doll_collection (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_l689_68937


namespace NUMINAMATH_CALUDE_constant_a_value_l689_68975

theorem constant_a_value (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 4 * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_a_value_l689_68975


namespace NUMINAMATH_CALUDE_opposite_five_fourteen_implies_eighteen_l689_68917

/-- A structure representing a circle with n equally spaced natural numbers -/
structure NumberCircle where
  n : ℕ
  numbers : Fin n → ℕ
  ordered : ∀ i : Fin n, numbers i = i.val + 1

/-- Definition of opposite numbers on the circle -/
def are_opposite (circle : NumberCircle) (a b : ℕ) : Prop :=
  ∃ i j : Fin circle.n,
    circle.numbers i = a ∧
    circle.numbers j = b ∧
    (j.val + circle.n / 2) % circle.n = i.val

/-- The main theorem -/
theorem opposite_five_fourteen_implies_eighteen (circle : NumberCircle) :
  are_opposite circle 5 14 → circle.n = 18 :=
by sorry

end NUMINAMATH_CALUDE_opposite_five_fourteen_implies_eighteen_l689_68917


namespace NUMINAMATH_CALUDE_a_range_theorem_l689_68949

/-- Proposition p: The inequality x^2 + 2ax + 4 > 0 holds true for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: The function f(x) = (3-2a)^x is increasing -/
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l689_68949


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l689_68901

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if its asymptotes are given by 2x ± 3y = 0, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 ↔ (2*x + 3*y = 0 ∨ 2*x - 3*y = 0)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l689_68901


namespace NUMINAMATH_CALUDE_stating_grouping_count_is_762_l689_68989

/-- Represents the number of tour guides -/
def num_guides : ℕ := 3

/-- Represents the number of tourists -/
def num_tourists : ℕ := 8

/-- 
Calculates the number of ways to distribute tourists among guides
where one guide has no tourists and the other two have at least one tourist each.
-/
def grouping_count : ℕ := sorry

/-- 
Theorem stating that the number of possible groupings
under the given conditions is 762.
-/
theorem grouping_count_is_762 : grouping_count = 762 := by sorry

end NUMINAMATH_CALUDE_stating_grouping_count_is_762_l689_68989


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l689_68900

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 3)^2 / 49 - (y + 2)^2 / 36 = 1

-- Define the asymptote function
def asymptote (m c x : ℝ) (y : ℝ) : Prop :=
  y = m * x + c

-- Theorem statement
theorem hyperbola_asymptotes :
  ∃ (m₁ m₂ c : ℝ),
    m₁ = 6/7 ∧ m₂ = -6/7 ∧ c = -32/7 ∧
    (∀ (x y : ℝ), hyperbola x y →
      (asymptote m₁ c x y ∨ asymptote m₂ c x y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l689_68900


namespace NUMINAMATH_CALUDE_angle_B_is_45_degrees_l689_68946

theorem angle_B_is_45_degrees 
  (A B C : Real) 
  (a b c : Real) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_angle_correspondence : a = BC ∧ b = AC ∧ c = AB) 
  (equation : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) : 
  B = 45 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_B_is_45_degrees_l689_68946


namespace NUMINAMATH_CALUDE_range_of_fraction_l689_68952

theorem range_of_fraction (x y : ℝ) (h1 : |x + y| ≤ 2) (h2 : |x - y| ≤ 2) :
  ∃ (z : ℝ), z = y / (x - 4) ∧ -1/2 ≤ z ∧ z ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l689_68952


namespace NUMINAMATH_CALUDE_pencil_count_l689_68974

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 5 →
  pencils = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l689_68974


namespace NUMINAMATH_CALUDE_sequence_sum_l689_68984

/-- Given a geometric sequence a, b, c and arithmetic sequences a, x, b and b, y, c, 
    prove that a/x + c/y = 2 -/
theorem sequence_sum (a b c x y : ℝ) 
  (h_geom : b^2 = a*c) 
  (h_arith1 : x = (a + b)/2) 
  (h_arith2 : y = (b + c)/2) : 
  a/x + c/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l689_68984


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l689_68970

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45) 
  (h2 : long_furred = 29) 
  (h3 : brown = 17) 
  (h4 : neither = 8) : 
  long_furred + brown - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l689_68970


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l689_68943

def digits_of (n : ℕ) : List ℕ := sorry

def sum_of_squares (l : List ℕ) : ℕ := sorry

def is_strictly_increasing (l : List ℕ) : Prop := sorry

def product_of_list (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions : 
  let n := 2346
  (sum_of_squares (digits_of n) = 65) ∧ 
  (is_strictly_increasing (digits_of n)) ∧
  (∀ m : ℕ, m > n → 
    (sum_of_squares (digits_of m) ≠ 65) ∨ 
    (¬ is_strictly_increasing (digits_of m))) ∧
  (product_of_list (digits_of n) = 144) := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l689_68943


namespace NUMINAMATH_CALUDE_divisibility_by_24_l689_68944

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l689_68944


namespace NUMINAMATH_CALUDE_water_tank_capacity_l689_68939

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ
  finalWater : ℝ

/-- Proves that the water tank has a capacity of 75 liters --/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 3)
  (h2 : (tank.initialWater + 5) / tank.capacity = 2 / 5) :
  tank.capacity = 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l689_68939


namespace NUMINAMATH_CALUDE_chord_length_limit_l689_68977

theorem chord_length_limit (r : ℝ) (chord_length : ℝ) :
  r = 6 →
  chord_length ≤ 2 * r →
  chord_length ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_limit_l689_68977


namespace NUMINAMATH_CALUDE_kaleb_net_profit_l689_68997

/-- Calculates the net profit for Kaleb's lawn mowing business --/
def net_profit (small_charge medium_charge large_charge : ℕ)
                (spring_small spring_medium spring_large : ℕ)
                (summer_small summer_medium summer_large : ℕ)
                (fuel_expense supply_cost : ℕ) : ℕ :=
  let spring_earnings := small_charge * spring_small + medium_charge * spring_medium + large_charge * spring_large
  let summer_earnings := small_charge * summer_small + medium_charge * summer_medium + large_charge * summer_large
  let total_earnings := spring_earnings + summer_earnings
  let total_lawns := spring_small + spring_medium + spring_large + summer_small + summer_medium + summer_large
  let total_expenses := fuel_expense * total_lawns + supply_cost
  total_earnings - total_expenses

theorem kaleb_net_profit :
  net_profit 10 20 30 2 3 1 10 8 5 2 60 = 402 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_net_profit_l689_68997


namespace NUMINAMATH_CALUDE_income_calculation_l689_68930

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 5 * expenditure / 4 →
  income - expenditure = savings →
  savings = 3600 →
  income = 18000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l689_68930


namespace NUMINAMATH_CALUDE_new_average_score_l689_68907

theorem new_average_score (n : ℕ) (a s : ℚ) (h1 : n = 4) (h2 : a = 78) (h3 : s = 88) :
  (n * a + s) / (n + 1) = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_l689_68907


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l689_68999

theorem ratio_of_linear_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 5 * y - 10 * x = b) (h3 : b ≠ 0) : a / b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l689_68999


namespace NUMINAMATH_CALUDE_cube_side_ratio_l689_68925

theorem cube_side_ratio (s S : ℝ) (h : s > 0) (H : S > 0) :
  (6 * S^2) / (6 * s^2) = 9 → S / s = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l689_68925


namespace NUMINAMATH_CALUDE_allocation_problem_l689_68992

/-- The number of ways to allocate doctors and nurses to schools --/
def allocations (doctors nurses schools : ℕ) : ℕ :=
  (doctors.factorial * nurses.choose (2 * schools)) / (schools.factorial * (2 ^ schools))

/-- Theorem stating the number of allocations for the given problem --/
theorem allocation_problem :
  allocations 3 6 3 = 540 :=
by sorry

end NUMINAMATH_CALUDE_allocation_problem_l689_68992


namespace NUMINAMATH_CALUDE_sam_supplies_cost_school_supplies_cost_proof_l689_68905

/-- Represents the school supplies -/
structure Supplies :=
  (glue_sticks : ℕ)
  (pencils : ℕ)
  (erasers : ℕ)

/-- Calculates the cost of supplies given their quantities and prices -/
def calculate_cost (s : Supplies) (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  s.glue_sticks * glue_price + s.pencils * pencil_price + s.erasers * eraser_price

theorem sam_supplies_cost (total : Supplies) (emily : Supplies) (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  let sam : Supplies := {
    glue_sticks := total.glue_sticks - emily.glue_sticks - sophie.glue_sticks,
    pencils := total.pencils - emily.pencils - sophie.pencils,
    erasers := total.erasers - emily.erasers - sophie.erasers
  }
  calculate_cost sam glue_price pencil_price eraser_price

/-- The main theorem to prove -/
theorem school_supplies_cost_proof 
    (total : Supplies)
    (emily : Supplies)
    (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) :
    total.glue_sticks = 27 ∧ 
    total.pencils = 40 ∧ 
    total.erasers = 15 ∧
    glue_price = 1 ∧
    pencil_price = 1/2 ∧
    eraser_price = 3/4 ∧
    emily.glue_sticks = 9 ∧
    emily.pencils = 18 ∧
    emily.erasers = 5 ∧
    sophie.glue_sticks = 12 ∧
    sophie.pencils = 14 ∧
    sophie.erasers = 4 →
    sam_supplies_cost total emily sophie glue_price pencil_price eraser_price = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_sam_supplies_cost_school_supplies_cost_proof_l689_68905


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l689_68978

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∃ (q₁ : ℕ), 4351 = q₁ * n + 8) ∧
  (∃ (q₂ : ℕ), 5161 = q₂ * n + 10) ∧
  (∃ (q₃ : ℕ), 6272 = q₃ * n + 12) ∧
  (∃ (q₄ : ℕ), 7383 = q₄ * n + 14) ∧
  ∀ (m : ℕ), m > 0 →
    (∃ (r₁ : ℕ), 4351 = r₁ * m + 8) →
    (∃ (r₂ : ℕ), 5161 = r₂ * m + 10) →
    (∃ (r₃ : ℕ), 6272 = r₃ * m + 12) →
    (∃ (r₄ : ℕ), 7383 = r₄ * m + 14) →
    m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l689_68978


namespace NUMINAMATH_CALUDE_cube_difference_square_root_l689_68976

theorem cube_difference_square_root : ∃ (n : ℕ), n > 0 ∧ n^2 = 105^3 - 104^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_difference_square_root_l689_68976


namespace NUMINAMATH_CALUDE_target_hit_probability_l689_68956

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 0.6) 
  (h_prob_B : prob_B = 0.5) : 
  let prob_hit_atleast_once := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B + prob_A * prob_B
  prob_A * (1 - prob_B) / prob_hit_atleast_once + prob_A * prob_B / prob_hit_atleast_once = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l689_68956


namespace NUMINAMATH_CALUDE_x_plus_inv_x_eq_five_l689_68995

theorem x_plus_inv_x_eq_five (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_eq_five_l689_68995


namespace NUMINAMATH_CALUDE_cube_surface_area_equals_volume_l689_68980

theorem cube_surface_area_equals_volume (a : ℝ) (h : a > 0) :
  6 * a^2 = a^3 → a = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equals_volume_l689_68980


namespace NUMINAMATH_CALUDE_red_balls_estimate_l689_68915

/-- Represents a bag of balls -/
structure Bag where
  total : ℕ
  redProb : ℝ

/-- Calculates the expected number of red balls in the bag -/
def expectedRedBalls (b : Bag) : ℝ :=
  b.total * b.redProb

theorem red_balls_estimate (b : Bag) 
  (h1 : b.total = 20)
  (h2 : b.redProb = 0.25) : 
  expectedRedBalls b = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_estimate_l689_68915


namespace NUMINAMATH_CALUDE_square_root_of_nine_l689_68922

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l689_68922


namespace NUMINAMATH_CALUDE_unique_number_division_multiplication_l689_68985

theorem unique_number_division_multiplication : ∃! x : ℚ, (x / 6) * 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_division_multiplication_l689_68985


namespace NUMINAMATH_CALUDE_XAXAXA_divisible_by_seven_l689_68955

/-- Given two digits X and A, XAXAXA is the six-digit number formed by repeating XA three times -/
def XAXAXA (X A : ℕ) : ℕ :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

/-- Theorem: For any two digits X and A, XAXAXA is divisible by 7 -/
theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) :
  ∃ k, XAXAXA X A = 7 * k :=
sorry

end NUMINAMATH_CALUDE_XAXAXA_divisible_by_seven_l689_68955


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l689_68906

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l689_68906


namespace NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l689_68981

theorem product_of_real_parts_complex_equation : 
  let f : ℂ → ℂ := fun x ↦ x^2 - 4*x + 2 - 2*I
  let solutions := {x : ℂ | f x = 0}
  ∃ x₁ x₂ : ℂ, x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  (x₁.re * x₂.re = 3 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l689_68981


namespace NUMINAMATH_CALUDE_perpendicular_slope_l689_68951

/-- The slope of a line perpendicular to a line passing through two given points -/
theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  (- 1 / m) = 4 / 3 →
  x₁ = 3 ∧ y₁ = -7 ∧ x₂ = -5 ∧ y₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l689_68951


namespace NUMINAMATH_CALUDE_amy_connor_score_difference_l689_68961

theorem amy_connor_score_difference
  (connor_score : ℕ)
  (amy_score : ℕ)
  (jason_score : ℕ)
  (connor_scored_two : connor_score = 2)
  (amy_scored_more : amy_score > connor_score)
  (jason_scored_twice_amy : jason_score = 2 * amy_score)
  (team_total_score : connor_score + amy_score + jason_score = 20) :
  amy_score - connor_score = 4 := by
sorry

end NUMINAMATH_CALUDE_amy_connor_score_difference_l689_68961


namespace NUMINAMATH_CALUDE_prob_12th_roll_last_correct_l689_68962

/-- The probability of the 12th roll being the last roll when rolling a standard six-sided die
    until getting the same number on consecutive rolls -/
def prob_12th_roll_last : ℚ := (5^10 : ℚ) / (6^11 : ℚ)

/-- Theorem stating that the probability of the 12th roll being the last roll is correct -/
theorem prob_12th_roll_last_correct :
  prob_12th_roll_last = (5^10 : ℚ) / (6^11 : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_12th_roll_last_correct_l689_68962


namespace NUMINAMATH_CALUDE_expand_expression_l689_68927

theorem expand_expression (x y : ℝ) : 
  (5 * x^2 - 3/2 * y) * (-4 * x^3 * y^2) = -20 * x^5 * y^2 + 6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l689_68927


namespace NUMINAMATH_CALUDE_frog_population_estimate_l689_68988

/-- Estimates the number of frogs in a pond based on capture-recapture data and population changes --/
theorem frog_population_estimate (tagged_april : ℕ) (caught_august : ℕ) (tagged_recaptured : ℕ)
  (left_pond_percent : ℚ) (new_frogs_percent : ℚ)
  (h1 : tagged_april = 100)
  (h2 : caught_august = 90)
  (h3 : tagged_recaptured = 5)
  (h4 : left_pond_percent = 30 / 100)
  (h5 : new_frogs_percent = 35 / 100) :
  let april_frogs_in_august := caught_august * (1 - new_frogs_percent)
  let estimated_april_population := (tagged_april * april_frogs_in_august) / tagged_recaptured
  estimated_april_population = 1180 := by
sorry

end NUMINAMATH_CALUDE_frog_population_estimate_l689_68988


namespace NUMINAMATH_CALUDE_mike_plant_cost_l689_68983

theorem mike_plant_cost (rose_price : ℝ) (rose_quantity : ℕ) 
  (rose_discount : ℝ) (rose_tax : ℝ) (aloe_price : ℝ) 
  (aloe_quantity : ℕ) (aloe_tax : ℝ) (friend_roses : ℕ) :
  rose_price = 75 ∧ 
  rose_quantity = 6 ∧ 
  rose_discount = 0.1 ∧ 
  rose_tax = 0.05 ∧ 
  aloe_price = 100 ∧ 
  aloe_quantity = 2 ∧ 
  aloe_tax = 0.07 ∧ 
  friend_roses = 2 →
  let total_rose_cost := rose_price * rose_quantity * (1 - rose_discount) * (1 + rose_tax)
  let friend_rose_cost := rose_price * friend_roses * (1 - rose_discount) * (1 + rose_tax)
  let aloe_cost := aloe_price * aloe_quantity * (1 + aloe_tax)
  total_rose_cost - friend_rose_cost + aloe_cost = 497.50 := by
sorry


end NUMINAMATH_CALUDE_mike_plant_cost_l689_68983


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l689_68914

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℚ := 0.16666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℚ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℚ := 0.08333333333333333

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℚ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall equals 0.6666666666666667 cm -/
theorem total_rainfall_sum :
  total_rainfall = 0.6666666666666667 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l689_68914


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l689_68986

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (Set.univ \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l689_68986


namespace NUMINAMATH_CALUDE_polynomial_simplification_l689_68934

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l689_68934
