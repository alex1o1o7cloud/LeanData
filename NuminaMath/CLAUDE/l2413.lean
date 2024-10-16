import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2413_241345

theorem perfect_square_divisibility (a b : ℕ) 
  (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2413_241345


namespace NUMINAMATH_CALUDE_divisibility_and_primality_l2413_241333

def ten_eights_base_nine : ℕ := 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def twelve_eights_base_nine : ℕ := 8 * 9^11 + 8 * 9^10 + 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def divisor1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def divisor2 : ℕ := 9^4 - 9^2 + 1

theorem divisibility_and_primality :
  (ten_eights_base_nine % divisor1 = 0) ∧
  (twelve_eights_base_nine % divisor2 = 0) ∧
  (¬ Nat.Prime divisor1) ∧
  (Nat.Prime divisor2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_primality_l2413_241333


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2413_241301

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2413_241301


namespace NUMINAMATH_CALUDE_sequence_convergence_condition_l2413_241329

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The limit of a sequence -/
def HasLimit (s : Sequence) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - l| < ε

/-- The condition on the sequence -/
def SequenceCondition (a b : ℝ) (x : Sequence) : Prop :=
  HasLimit (fun n => a * x (n + 1) - b * x n) 0

/-- The main theorem -/
theorem sequence_convergence_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : Sequence, SequenceCondition a b x → HasLimit x 0) ↔
  (a = 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ |b / a| < 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_condition_l2413_241329


namespace NUMINAMATH_CALUDE_spacy_subsets_count_l2413_241354

/-- A set of integers is spacy if it contains no more than one out of any four consecutive integers. -/
def IsSpacy (s : Set ℕ) : Prop :=
  ∀ n : ℕ, (n ∈ s → (n + 1 ∉ s ∧ n + 2 ∉ s ∧ n + 3 ∉ s))

/-- The number of spacy subsets of {1, 2, ..., n} -/
def NumSpacySubsets (n : ℕ) : ℕ :=
  if n ≤ 4 then
    n + 1
  else
    NumSpacySubsets (n - 1) + NumSpacySubsets (n - 4)

theorem spacy_subsets_count :
  NumSpacySubsets 15 = 181 :=
by sorry

end NUMINAMATH_CALUDE_spacy_subsets_count_l2413_241354


namespace NUMINAMATH_CALUDE_quadratic_derivative_bound_l2413_241320

-- Define a quadratic function
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem quadratic_derivative_bound :
  ∃ A : ℝ,
    (∀ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
      |b| ≤ A) ∧
    (∃ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) ∧
      |b| = A) ∧
    (∀ A' : ℝ,
      (∀ a b c : ℝ,
        (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
        |b| ≤ A') →
      A ≤ A') :=
by sorry

end NUMINAMATH_CALUDE_quadratic_derivative_bound_l2413_241320


namespace NUMINAMATH_CALUDE_annas_weight_anna_weighs_80_l2413_241319

/-- The weight of Anna given Jack's weight and the balancing condition on a see-saw -/
theorem annas_weight (jack_weight : ℕ) (rock_weight : ℕ) (rock_count : ℕ) : ℕ :=
  jack_weight + rock_weight * rock_count

/-- Proof that Anna weighs 80 pounds given the conditions -/
theorem anna_weighs_80 :
  annas_weight 60 4 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_annas_weight_anna_weighs_80_l2413_241319


namespace NUMINAMATH_CALUDE_jacqueline_erasers_l2413_241380

-- Define the given quantities
def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

-- Define the total number of erasers
def total_erasers : ℕ := cases * boxes_per_case * erasers_per_box

-- Theorem to prove
theorem jacqueline_erasers : total_erasers = 2100 := by
  sorry

end NUMINAMATH_CALUDE_jacqueline_erasers_l2413_241380


namespace NUMINAMATH_CALUDE_factoring_expression_l2413_241332

theorem factoring_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2413_241332


namespace NUMINAMATH_CALUDE_football_team_practice_hours_l2413_241307

/-- Given a football team's practice schedule, calculate the total practice hours in a week with one missed day. -/
theorem football_team_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end NUMINAMATH_CALUDE_football_team_practice_hours_l2413_241307


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2413_241384

/-- Represents a line in a 2D Cartesian coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a point (x, y) is in the fourth quadrant -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Determines if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ is_in_fourth_quadrant x y

/-- The main theorem: the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2413_241384


namespace NUMINAMATH_CALUDE_prob_two_spades_is_one_seventeenth_l2413_241310

/-- A standard deck of cards --/
structure Deck :=
  (total_cards : Nat)
  (spade_cards : Nat)
  (h_total : total_cards = 52)
  (h_spades : spade_cards = 13)

/-- The probability of drawing two spades as the first two cards --/
def prob_two_spades (d : Deck) : ℚ :=
  (d.spade_cards : ℚ) / d.total_cards * (d.spade_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two spades as the first two cards is 1/17 --/
theorem prob_two_spades_is_one_seventeenth (d : Deck) : prob_two_spades d = 1 / 17 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_spades_is_one_seventeenth_l2413_241310


namespace NUMINAMATH_CALUDE_area_of_inscribed_square_on_hypotenuse_l2413_241318

/-- An isosceles right triangle with inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the square inscribed with one side on a leg -/
  s : ℝ
  /-- Side length of the square inscribed with one side on the hypotenuse -/
  S : ℝ
  /-- The area of the square inscribed with one side on a leg is 484 -/
  h_area_s : s^2 = 484
  /-- Relationship between s and S in an isosceles right triangle -/
  h_relation : 3 * S = s * Real.sqrt 2

/-- 
Theorem: In an isosceles right triangle, if a square inscribed with one side on a leg 
has an area of 484 cm², then a square inscribed with one side on the hypotenuse 
has an area of 968/9 cm².
-/
theorem area_of_inscribed_square_on_hypotenuse 
  (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 968 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_square_on_hypotenuse_l2413_241318


namespace NUMINAMATH_CALUDE_stone_piles_theorem_l2413_241321

/-- Represents the state of stone piles after operations -/
structure StonePiles :=
  (num_piles : Nat)
  (initial_stones : Nat)
  (operations : Nat)
  (pile_a_stones : Nat)
  (pile_b_stones : Nat)

/-- The theorem to prove -/
theorem stone_piles_theorem (sp : StonePiles) : 
  sp.num_piles = 20 →
  sp.initial_stones = 2006 →
  sp.operations < 20 →
  sp.pile_a_stones = 1990 →
  2080 ≤ sp.pile_b_stones →
  sp.pile_b_stones ≤ 2100 →
  sp.pile_b_stones = 2090 := by
sorry

end NUMINAMATH_CALUDE_stone_piles_theorem_l2413_241321


namespace NUMINAMATH_CALUDE_intersecting_circles_m_plus_c_l2413_241343

/-- Two circles intersect at points A and B, with the centers of the circles lying on a line. -/
structure IntersectingCircles where
  m : ℝ
  c : ℝ
  A : ℝ × ℝ := (1, 3)
  B : ℝ × ℝ := (m, -1)
  centers_line : ℝ → ℝ := fun x ↦ x + c

/-- The value of m+c for the given intersecting circles configuration is 3. -/
theorem intersecting_circles_m_plus_c (circles : IntersectingCircles) : 
  circles.m + circles.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_circles_m_plus_c_l2413_241343


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2413_241391

theorem boy_scouts_permission_slips 
  (total_with_slips : ℝ) 
  (boy_scouts_percentage : ℝ) 
  (girl_scouts_with_slips : ℝ) :
  total_with_slips = 0.60 →
  boy_scouts_percentage = 0.45 →
  girl_scouts_with_slips = 0.6818 →
  (boy_scouts_percentage * (total_with_slips - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) / 
  (boy_scouts_percentage * (1 - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) = 0.50 :=
by sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2413_241391


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2413_241386

-- Define the vectors a and b as functions of x
def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 1, 3]

-- Define what it means for two vectors to be parallel
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

-- State the theorem
theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, parallel (a x) (b x) → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2413_241386


namespace NUMINAMATH_CALUDE_abduls_numbers_l2413_241397

theorem abduls_numbers (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (eq1 : a + (b + c + d) / 3 = 17)
  (eq2 : b + (a + c + d) / 3 = 21)
  (eq3 : c + (a + b + d) / 3 = 23)
  (eq4 : d + (a + b + c) / 3 = 29) :
  max a (max b (max c d)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_abduls_numbers_l2413_241397


namespace NUMINAMATH_CALUDE_project_completion_days_l2413_241388

/-- Represents the work rates and schedule for a project completed by three persons. -/
structure ProjectSchedule where
  rate_A : ℚ  -- Work rate of person A (fraction of work completed per day)
  rate_B : ℚ  -- Work rate of person B
  rate_C : ℚ  -- Work rate of person C
  days_A : ℕ  -- Number of days A works alone
  days_BC : ℕ  -- Number of days B and C work together

/-- Calculates the total number of days needed to complete the project. -/
def totalDays (p : ProjectSchedule) : ℚ :=
  let work_A := p.rate_A * p.days_A
  let rate_BC := p.rate_B + p.rate_C
  let work_BC := rate_BC * p.days_BC
  let remaining_work := 1 - (work_A + work_BC)
  p.days_A + p.days_BC + remaining_work / p.rate_C

/-- Theorem stating that for the given project schedule, the total days needed is 9. -/
theorem project_completion_days :
  let p := ProjectSchedule.mk (1/10) (1/12) (1/15) 2 4
  totalDays p = 9 := by sorry

end NUMINAMATH_CALUDE_project_completion_days_l2413_241388


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2413_241309

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l: x = my + 4 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

/-- Point on parabola C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : parabola_C x y

/-- Points M and N are perpendicular from origin -/
def perpendicular_from_origin (M N : PointOnC) : Prop :=
  M.x * N.x + M.y * N.y = 0

theorem line_passes_through_fixed_point (m : ℝ) 
  (M N : PointOnC) (h_distinct : M ≠ N) 
  (h_on_l : line_l m M.x M.y ∧ line_l m N.x N.y)
  (h_perp : perpendicular_from_origin M N) :
  line_l m 4 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2413_241309


namespace NUMINAMATH_CALUDE_sequence_general_term_l2413_241340

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℝ := 4 * n^2 + 2 * n

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 8 * n - 2

/-- Theorem stating that the given general term formula is correct -/
theorem sequence_general_term (n : ℕ) : 
  S n - S (n - 1) = a n := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2413_241340


namespace NUMINAMATH_CALUDE_temperature_is_dependent_l2413_241375

/-- Represents the variables in the solar water heater scenario -/
inductive SolarHeaterVariable
  | IntensityOfSunlight
  | TemperatureOfWater
  | DurationOfExposure
  | CapacityOfHeater

/-- Represents the relationship between two variables -/
structure Relationship where
  independent : SolarHeaterVariable
  dependent : SolarHeaterVariable

/-- Defines the relationship in the solar water heater scenario -/
def solarHeaterRelationship : Relationship :=
  { independent := SolarHeaterVariable.DurationOfExposure,
    dependent := SolarHeaterVariable.TemperatureOfWater }

/-- Theorem: The temperature of water is the dependent variable in the solar water heater scenario -/
theorem temperature_is_dependent :
  solarHeaterRelationship.dependent = SolarHeaterVariable.TemperatureOfWater :=
by sorry

end NUMINAMATH_CALUDE_temperature_is_dependent_l2413_241375


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2413_241387

theorem exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 ≤ (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2413_241387


namespace NUMINAMATH_CALUDE_zebra_speed_l2413_241362

/-- Proves that given a tiger with an average speed of 30 kmph and a 5-hour head start,
    if a zebra catches up to the tiger in 6 hours, the zebra's average speed is 55 kmph. -/
theorem zebra_speed (tiger_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  tiger_speed = 30 →
  head_start = 5 →
  catch_up_time = 6 →
  (head_start + catch_up_time) * tiger_speed = catch_up_time * 55 :=
by sorry

end NUMINAMATH_CALUDE_zebra_speed_l2413_241362


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2413_241363

/-- A geometric sequence with a_1 = 1/9 and a_5 = 9 has a_3 = 1 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 9 →
  a 5 = 9 →
  a 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2413_241363


namespace NUMINAMATH_CALUDE_inverse_of_inverse_sixteen_l2413_241330

def f (x : ℝ) : ℝ := 5 * x + 6

theorem inverse_of_inverse_sixteen (hf : ∀ x, f x = 5 * x + 6) :
  (f ∘ f) (-4/5) = 16 :=
sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_sixteen_l2413_241330


namespace NUMINAMATH_CALUDE_missing_digit_is_one_l2413_241351

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_sum (d : ℕ) : ℕ :=
  3 + 5 + 7 + 2 + d + 9

theorem missing_digit_is_one :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_3 (357200 + d * 10 + 9) ↔ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_one_l2413_241351


namespace NUMINAMATH_CALUDE_prism_dimension_is_five_l2413_241359

/-- Represents a rectangular prism with dimensions n × n × 2n -/
structure RectangularPrism (n : ℕ) where
  length : ℕ := n
  width : ℕ := n
  height : ℕ := 2 * n

/-- The number of unit cubes obtained by cutting the prism -/
def num_unit_cubes (n : ℕ) : ℕ := 2 * n^3

/-- The total number of faces of all unit cubes -/
def total_faces (n : ℕ) : ℕ := 6 * num_unit_cubes n

/-- The number of blue faces (painted faces of the original prism) -/
def blue_faces (n : ℕ) : ℕ := 2 * n^2 + 4 * (2 * n^2)

/-- Theorem stating that if one-sixth of the total faces are blue, then n = 5 -/
theorem prism_dimension_is_five (n : ℕ) :
  (blue_faces n : ℚ) / (total_faces n : ℚ) = 1 / 6 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_dimension_is_five_l2413_241359


namespace NUMINAMATH_CALUDE_point_on_extension_line_l2413_241334

/-- Given two points in a 2D plane and a third point on their extension line,
    prove that the third point has specific coordinates. -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t > 1 ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by
  sorry


end NUMINAMATH_CALUDE_point_on_extension_line_l2413_241334


namespace NUMINAMATH_CALUDE_cubic_factorization_l2413_241367

theorem cubic_factorization (y : ℝ) : y^3 - 16*y = y*(y+4)*(y-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2413_241367


namespace NUMINAMATH_CALUDE_polygon_35_sides_5_restricted_l2413_241313

/-- The number of diagonals in a convex polygon with restricted vertices -/
def diagonals_with_restrictions (n : ℕ) (r : ℕ) : ℕ :=
  let effective_vertices := n - r
  (effective_vertices * (effective_vertices - 3)) / 2

/-- Theorem: A convex polygon with 35 sides and 5 restricted vertices has 405 diagonals -/
theorem polygon_35_sides_5_restricted : diagonals_with_restrictions 35 5 = 405 := by
  sorry

end NUMINAMATH_CALUDE_polygon_35_sides_5_restricted_l2413_241313


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2413_241349

theorem solution_set_inequality (x : ℝ) : 
  x * (x + 3) ≥ 0 ↔ x ≥ 0 ∨ x ≤ -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2413_241349


namespace NUMINAMATH_CALUDE_electric_bus_pricing_and_optimal_plan_l2413_241393

/-- Represents the unit price of a type A electric bus in million yuan -/
def type_a_price : ℝ := 36

/-- Represents the unit price of a type B electric bus in million yuan -/
def type_b_price : ℝ := 40

/-- Represents the number of type A buses in the optimal plan -/
def optimal_type_a : ℕ := 20

/-- Represents the number of type B buses in the optimal plan -/
def optimal_type_b : ℕ := 10

/-- Represents the total cost of the optimal plan in million yuan -/
def optimal_total_cost : ℝ := 1120

theorem electric_bus_pricing_and_optimal_plan :
  (type_b_price = type_a_price + 4) ∧
  (720 / type_a_price = 800 / type_b_price) ∧
  (optimal_type_a + optimal_type_b = 30) ∧
  (optimal_type_a ≥ 10) ∧
  (optimal_type_a ≤ 2 * optimal_type_b) ∧
  (∀ m n : ℕ, m + n = 30 → m ≥ 10 → m ≤ 2 * n →
    type_a_price * m + type_b_price * n ≥ optimal_total_cost) ∧
  (optimal_total_cost = type_a_price * optimal_type_a + type_b_price * optimal_type_b) :=
by sorry


end NUMINAMATH_CALUDE_electric_bus_pricing_and_optimal_plan_l2413_241393


namespace NUMINAMATH_CALUDE_water_dispenser_capacity_l2413_241368

/-- A cylindrical water dispenser with capacity x liters -/
structure WaterDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- The water dispenser contains 60 liters when it is 25% full -/
def quarter_full (d : WaterDispenser) : Prop :=
  0.25 * d.capacity = 60

/-- Theorem: A cylindrical water dispenser that contains 60 liters when 25% full has a total capacity of 240 liters -/
theorem water_dispenser_capacity (d : WaterDispenser) 
  (h1 : d.cylindrical = true) 
  (h2 : quarter_full d) : 
  d.capacity = 240 := by
  sorry

end NUMINAMATH_CALUDE_water_dispenser_capacity_l2413_241368


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l2413_241315

theorem sum_of_squares_divisible_by_seven (a b : ℤ) : 
  (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l2413_241315


namespace NUMINAMATH_CALUDE_max_t_value_l2413_241347

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| - x

-- State the theorem
theorem max_t_value (a : ℝ) (h : a ≤ 1) :
  (∃ t : ℝ, t = 1 + Real.sqrt 7 ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a x ∧ f a x ≤ 6) ∧
   (∀ t' : ℝ, t' > t →
     ∃ x : ℝ, x ∈ Set.Icc 0 t' ∧ (f a x < -1 ∨ f a x > 6))) ∧
  (∀ a' : ℝ, a' ≤ 1 →
    ∀ t : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a' x ∧ f a' x ≤ 6) →
      t ≤ 1 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l2413_241347


namespace NUMINAMATH_CALUDE_prob_even_sum_specific_wheels_l2413_241376

/-- Represents a wheel with even and odd numbered sections -/
structure Wheel :=
  (total : ℕ)
  (even : ℕ)
  (odd : ℕ)
  (sum_sections : even + odd = total)

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def prob_even_sum (w1 w2 : Wheel) : ℚ :=
  let p1_even := w1.even / w1.total
  let p1_odd := w1.odd / w1.total
  let p2_even := w2.even / w2.total
  let p2_odd := w2.odd / w2.total
  (p1_even * p2_even) + (p1_odd * p2_odd)

theorem prob_even_sum_specific_wheels :
  let wheel1 : Wheel := ⟨6, 2, 4, rfl⟩
  let wheel2 : Wheel := ⟨8, 3, 5, rfl⟩
  prob_even_sum wheel1 wheel2 = 13/24 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_specific_wheels_l2413_241376


namespace NUMINAMATH_CALUDE_hexagon_circumradius_theorem_l2413_241395

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon : Type := Fin 6 → ℝ × ℝ

-- Define the property of being convex for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property that all sides of the hexagon have length 1
def all_sides_unit_length (h : Hexagon) : Prop := sorry

-- Define the circumradius of a triangle given by three points
def circumradius (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hexagon_circumradius_theorem (h : Hexagon) 
  (convex : is_convex h) 
  (unit_sides : all_sides_unit_length h) : 
  max (circumradius (h 0) (h 2) (h 4)) (circumradius (h 1) (h 3) (h 5)) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hexagon_circumradius_theorem_l2413_241395


namespace NUMINAMATH_CALUDE_largest_three_digit_number_l2413_241372

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_equation (a b c d e f : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ f ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + 10 * b + c = 100 * d + 10 * e + f

theorem largest_three_digit_number :
  ∀ a b c d e f : Nat,
    valid_equation a b c d e f →
    100 * d + 10 * e + f ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_l2413_241372


namespace NUMINAMATH_CALUDE_one_row_with_ten_seats_l2413_241323

/-- Represents the seating arrangement in a theater --/
structure TheaterSeating where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if the seating arrangement is valid --/
def is_valid_seating (s : TheaterSeating) : Prop :=
  s.total_people = 55 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

/-- Theorem stating that there is exactly one row seating 10 people --/
theorem one_row_with_ten_seats :
  ∃! s : TheaterSeating, is_valid_seating s ∧ s.rows_with_ten = 1 :=
sorry

end NUMINAMATH_CALUDE_one_row_with_ten_seats_l2413_241323


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2413_241338

/-- The inclination angle of a line is the angle it makes with the positive x-axis. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := -1, c := 1 }
  inclination_angle l.a l.b l.c = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2413_241338


namespace NUMINAMATH_CALUDE_seventeen_in_binary_l2413_241352

theorem seventeen_in_binary : 17 = 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_binary_l2413_241352


namespace NUMINAMATH_CALUDE_rectangular_prism_max_volume_l2413_241337

/-- Given a rectangular prism with space diagonal 10 and orthogonal projection 8,
    its maximum volume is 192 -/
theorem rectangular_prism_max_volume :
  ∀ (a b h : ℝ),
  (a > 0) → (b > 0) → (h > 0) →
  (a^2 + b^2 + h^2 = 10^2) →
  (a^2 + b^2 = 8^2) →
  ∀ (V : ℝ), V = a * b * h →
  V ≤ 192 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_max_volume_l2413_241337


namespace NUMINAMATH_CALUDE_state_tax_rate_is_4_percent_l2413_241378

/-- Calculates the state tax rate given the following conditions:
  * The taxpayer was a resident for 9 months out of the year
  * The taxpayer's taxable income for the year
  * The prorated tax amount paid for the time of residency
-/
def calculate_state_tax_rate (months_resident : ℕ) (taxable_income : ℚ) (tax_paid : ℚ) : ℚ :=
  let full_year_months : ℕ := 12
  let residence_ratio : ℚ := months_resident / full_year_months
  let full_year_tax : ℚ := tax_paid / residence_ratio
  (full_year_tax / taxable_income) * 100

theorem state_tax_rate_is_4_percent :
  let months_resident : ℕ := 9
  let taxable_income : ℚ := 42500
  let tax_paid : ℚ := 1275
  calculate_state_tax_rate months_resident taxable_income tax_paid = 4 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_rate_is_4_percent_l2413_241378


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l2413_241382

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_range (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l2413_241382


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_one_l2413_241304

theorem sqrt_real_implies_x_leq_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) → x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_one_l2413_241304


namespace NUMINAMATH_CALUDE_negation_of_all_even_divisible_by_two_l2413_241325

theorem negation_of_all_even_divisible_by_two :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Even n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_even_divisible_by_two_l2413_241325


namespace NUMINAMATH_CALUDE_good_number_ending_8_has_9_before_l2413_241361

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

def ends_with_8 (n : ℕ) : Prop :=
  n % 10 = 8

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem good_number_ending_8_has_9_before :
  ∀ n : ℕ, is_good n → ends_with_8 n → second_to_last_digit n = 9 := by
  sorry

end NUMINAMATH_CALUDE_good_number_ending_8_has_9_before_l2413_241361


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l2413_241317

theorem angle_measure_in_special_triangle (A B C : ℝ) :
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  B = 2 * A →        -- ∠B is twice ∠A
  C = 4 * A →        -- ∠C is four times ∠A
  B = 360 / 7 :=     -- Measure of ∠B is 360/7°
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l2413_241317


namespace NUMINAMATH_CALUDE_painting_progress_l2413_241355

/-- Represents the fraction of a wall painted in a given time -/
def fraction_painted (total_time minutes : ℕ) : ℚ :=
  minutes / total_time

theorem painting_progress (heidi_time karl_time minutes : ℕ) 
  (h1 : heidi_time = 60)
  (h2 : karl_time = heidi_time / 2)
  (h3 : minutes = 20) :
  (fraction_painted heidi_time minutes = 1/3) ∧ 
  (fraction_painted karl_time minutes = 2/3) := by
  sorry

#check painting_progress

end NUMINAMATH_CALUDE_painting_progress_l2413_241355


namespace NUMINAMATH_CALUDE_sum_lower_bound_l2413_241369

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l2413_241369


namespace NUMINAMATH_CALUDE_books_read_by_three_l2413_241335

/-- The number of different books read by three people given their individual book counts and overlap -/
def total_different_books (tony_books dean_books breanna_books tony_dean_overlap all_overlap : ℕ) : ℕ :=
  tony_books + dean_books + breanna_books - tony_dean_overlap - 2 * all_overlap

/-- Theorem stating the total number of different books read by Tony, Dean, and Breanna -/
theorem books_read_by_three :
  total_different_books 23 12 17 3 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_books_read_by_three_l2413_241335


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2413_241346

theorem last_two_digits_sum (n : ℕ) : (7^30 + 13^30) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2413_241346


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l2413_241302

/-- Represents a coin showing either heads or tails -/
inductive CoinFace
| Heads
| Tails

/-- Checks if two coin faces are the same -/
def same_face (a b : CoinFace) : Prop :=
  (a = CoinFace.Heads ∧ b = CoinFace.Heads) ∨ (a = CoinFace.Tails ∧ b = CoinFace.Tails)

/-- A circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → CoinFace

theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ i : Fin 11, same_face (arrangement i) (arrangement ((i + 1) % 11)) :=
sorry


end NUMINAMATH_CALUDE_adjacent_same_face_exists_l2413_241302


namespace NUMINAMATH_CALUDE_pattern_equality_l2413_241350

theorem pattern_equality (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l2413_241350


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2413_241377

theorem gcd_of_three_numbers : Nat.gcd 17420 (Nat.gcd 23826 36654) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2413_241377


namespace NUMINAMATH_CALUDE_f_continuous_at_2_l2413_241353

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 * x^2 - 7 else b * (x - 2)^2 + 5

-- State the theorem
theorem f_continuous_at_2 (b : ℝ) : 
  ContinuousAt (f b) 2 := by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_l2413_241353


namespace NUMINAMATH_CALUDE_intersection_point_solution_l2413_241392

/-- Given two lines y = x + 1 and y = mx + n that intersect at point (1,b),
    prove that the solution to the system of equations { x + 1 = y, y - mx = n }
    is x = 1 and y = 2. -/
theorem intersection_point_solution (m n b : ℝ) :
  (∃ x y : ℝ, x + 1 = y ∧ y - m*x = n) →
  (1 + 1 = b) →
  (∀ x y : ℝ, x + 1 = y ∧ y - m*x = n → x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l2413_241392


namespace NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l2413_241360

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in 2D space of the form y^2 = kx -/
structure Parabola where
  k : ℝ

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def on_parabola (p : Point) (para : Parabola) : Prop :=
  p.y^2 = para.k * p.x

def right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (c.x - a.x) + (b.y - a.y) * (c.y - a.y) = 0

theorem parabola_line_intersection_right_angle :
  ∀ (l : Line) (para : Parabola) (a b c : Point),
    l.a = 1 ∧ l.b = -2 ∧ l.c = -1 →
    para.k = 4 →
    on_line a l ∧ on_parabola a para →
    on_line b l ∧ on_parabola b para →
    on_parabola c para →
    right_angle a c b →
    (c.x = 1 ∧ c.y = -2) ∨ (c.x = 9 ∧ c.y = -6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l2413_241360


namespace NUMINAMATH_CALUDE_platform_length_l2413_241326

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_cross_time = 30 →
  man_cross_time = 17 →
  (train_speed * platform_cross_time) - (train_speed * man_cross_time) = 260 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2413_241326


namespace NUMINAMATH_CALUDE_zip_code_relationship_l2413_241336

/-- 
Theorem: Given a sequence of five numbers A, B, C, D, and E satisfying certain conditions,
prove that the sum of the first two numbers (A + B) equals 2.
-/
theorem zip_code_relationship (A B C D E : ℕ) 
  (sum_condition : A + B + C + D + E = 10)
  (third_zero : C = 0)
  (fourth_double_first : D = 2 * A)
  (fourth_fifth_sum : D + E = 8) :
  A + B = 2 := by
  sorry

end NUMINAMATH_CALUDE_zip_code_relationship_l2413_241336


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l2413_241306

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x + 20 = x + 51

-- Define the solutions of the quadratic equation
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

-- State the theorem
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 4 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l2413_241306


namespace NUMINAMATH_CALUDE_division_and_power_equality_l2413_241365

theorem division_and_power_equality : ((-125) / (-25)) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_division_and_power_equality_l2413_241365


namespace NUMINAMATH_CALUDE_power_product_simplification_l2413_241344

theorem power_product_simplification :
  (5 / 3 : ℚ) ^ 2023 * (6 / 10 : ℚ) ^ 2022 = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l2413_241344


namespace NUMINAMATH_CALUDE_pet_ownership_percentages_l2413_241311

def total_students : ℕ := 500
def dog_owners : ℕ := 125
def cat_owners : ℕ := 100
def rabbit_owners : ℕ := 50

def percent_dog_owners : ℚ := dog_owners / total_students * 100
def percent_cat_owners : ℚ := cat_owners / total_students * 100
def percent_rabbit_owners : ℚ := rabbit_owners / total_students * 100

theorem pet_ownership_percentages :
  percent_dog_owners = 25 ∧
  percent_cat_owners = 20 ∧
  percent_rabbit_owners = 10 :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_percentages_l2413_241311


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2413_241308

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 = 2 → a 14 = 18 → a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2413_241308


namespace NUMINAMATH_CALUDE_unbroken_seashells_l2413_241366

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) : 
  total_seashells - broken_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l2413_241366


namespace NUMINAMATH_CALUDE_total_lives_calculation_l2413_241356

theorem total_lives_calculation (initial_players : ℕ) (new_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 16 → new_players = 4 → lives_per_player = 10 →
  (initial_players + new_players) * lives_per_player = 200 := by
sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l2413_241356


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2413_241383

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio_two (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  (2 * a 2 + a 3) / (2 * a 4 + a 5) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2413_241383


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2413_241328

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_sq_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2413_241328


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_root_l2413_241357

theorem quadratic_equation_positive_root (m : ℝ) :
  ∃ x : ℝ, x > 0 ∧ (x - 1) * (x - 2) - m^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_root_l2413_241357


namespace NUMINAMATH_CALUDE_V_min_at_2_minus_sqrt2_l2413_241394

open Real

/-- The volume function V(a) -/
noncomputable def V (a : ℝ) : ℝ := 
  π * ((3-a) * (log (3-a))^2 + 2*a * log (3-a) - (1-a) * (log (1-a))^2 - 2*a * log (1-a))

/-- The theorem stating that V(a) has a minimum at a = 2 - √2 -/
theorem V_min_at_2_minus_sqrt2 :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ 
  (∀ (x : ℝ), 0 < x → x < 1 → V x ≥ V a) ∧ 
  a = 2 - sqrt 2 := by
  sorry

/-- Verify that 2 - √2 is indeed between 0 and 1 -/
lemma two_minus_sqrt_two_in_range : 
  0 < 2 - sqrt 2 ∧ 2 - sqrt 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_V_min_at_2_minus_sqrt2_l2413_241394


namespace NUMINAMATH_CALUDE_sliced_meat_cost_l2413_241303

/-- Given a 4 pack of sliced meat costing $40.00 with an additional 30% for rush delivery,
    the cost per type of meat is $13.00. -/
theorem sliced_meat_cost (pack_size : ℕ) (base_cost rush_percentage : ℚ) :
  pack_size = 4 →
  base_cost = 40 →
  rush_percentage = 0.3 →
  (base_cost + base_cost * rush_percentage) / pack_size = 13 :=
by sorry

end NUMINAMATH_CALUDE_sliced_meat_cost_l2413_241303


namespace NUMINAMATH_CALUDE_laundry_dishes_time_difference_l2413_241396

theorem laundry_dishes_time_difference 
  (dawn_dish_time andy_laundry_time : ℕ) 
  (h1 : dawn_dish_time = 20) 
  (h2 : andy_laundry_time = 46) : 
  andy_laundry_time - 2 * dawn_dish_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_laundry_dishes_time_difference_l2413_241396


namespace NUMINAMATH_CALUDE_find_d_l2413_241399

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = 2*d + Real.sqrt (a + b + c + d)) : 
  d = (-7 + Real.sqrt 33) / 8 := by
sorry

end NUMINAMATH_CALUDE_find_d_l2413_241399


namespace NUMINAMATH_CALUDE_three_student_committees_l2413_241312

theorem three_student_committees (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_student_committees_l2413_241312


namespace NUMINAMATH_CALUDE_unique_a_value_l2413_241371

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a+2}

theorem unique_a_value : ∀ a : ℝ, A a ∩ B a = B a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2413_241371


namespace NUMINAMATH_CALUDE_min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l2413_241390

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point
  total_area : ℝ
  folded_area : ℝ

/-- The ratio of folded visible area to total area is at least 2/3 -/
theorem min_ratio_folded_to_total_area (t : Tablecloth) : 
  t.folded_area / t.total_area ≥ 2/3 := by
  sorry

/-- The ratio of 2/3 is achievable -/
theorem ratio_two_thirds_achievable : 
  ∃ t : Tablecloth, t.folded_area / t.total_area = 2/3 := by
  sorry

/-- The minimum ratio of folded visible area to total area is exactly 2/3 -/
theorem min_ratio_is_two_thirds : 
  (∀ t : Tablecloth, t.folded_area / t.total_area ≥ 2/3) ∧
  (∃ t : Tablecloth, t.folded_area / t.total_area = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l2413_241390


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2413_241370

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | x < -2 or x > 3}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧
  (a + b + c < 0) ∧
  (∀ x, c*x^2 - b*x + a < 0 ↔ x < -1/3 ∨ x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2413_241370


namespace NUMINAMATH_CALUDE_problem_statements_l2413_241327

theorem problem_statements :
  (∀ (x : ℝ), x ≥ 3 → 2*x - 10 ≥ 0) ↔ ¬(∃ (x : ℝ), x ≥ 3 ∧ 2*x - 10 < 0) ∧
  (∀ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ (a b m : ℝ), a > b ∧ b > 0 ∧ m > 0 → a / b > (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2413_241327


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2413_241316

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ d : ℝ) 
  (h₁ : arithmetic_sequence a₁ d 3 = 10) 
  (h₂ : arithmetic_sequence a₁ d 8 = 30) : 
  arithmetic_sequence a₁ d 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2413_241316


namespace NUMINAMATH_CALUDE_tv_screen_area_l2413_241305

theorem tv_screen_area (width height area : ℝ) : 
  width = 3 ∧ height = 7 ∧ area = width * height → area = 21 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_area_l2413_241305


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2413_241300

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2413_241300


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_33_l2413_241389

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_33 : count_D_eq_3 = 33 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_33_l2413_241389


namespace NUMINAMATH_CALUDE_binomial_sum_l2413_241379

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l2413_241379


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2413_241373

theorem linear_equation_condition (a : ℝ) : 
  (|a| - 1 = 1 ∧ a - 2 ≠ 0) ↔ a = -2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2413_241373


namespace NUMINAMATH_CALUDE_max_a1_is_26_l2413_241314

/-- A sequence of positive integers satisfying the given conditions --/
def GoodSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ n, a (n + 1) ≤ a n + 5) ∧
  (∀ n, n ∣ a n)

/-- The maximum possible value of a₁ in a good sequence --/
def MaxA1 : ℕ := 26

/-- The theorem stating that the maximum possible value of a₁ in a good sequence is 26 --/
theorem max_a1_is_26 :
  (∃ a, GoodSequence a ∧ a 1 = MaxA1) ∧
  (∀ a, GoodSequence a → a 1 ≤ MaxA1) :=
sorry

end NUMINAMATH_CALUDE_max_a1_is_26_l2413_241314


namespace NUMINAMATH_CALUDE_james_class_size_l2413_241358

theorem james_class_size (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 1) ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ k : ℕ, n = 6 * k - 3) →
  n = 123 ∨ n = 183 := by
sorry

end NUMINAMATH_CALUDE_james_class_size_l2413_241358


namespace NUMINAMATH_CALUDE_aaron_used_three_boxes_l2413_241341

/-- Given the initial number of can lids, final number of can lids, and lids per box,
    calculate the number of boxes used. -/
def boxes_used (initial_lids : ℕ) (final_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  (final_lids - initial_lids) / lids_per_box

/-- Theorem stating that Aaron used 3 boxes of canned tomatoes. -/
theorem aaron_used_three_boxes :
  boxes_used 14 53 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_aaron_used_three_boxes_l2413_241341


namespace NUMINAMATH_CALUDE_water_bottle_cost_l2413_241322

/-- Given Barbara's shopping information, prove the cost of each water bottle -/
theorem water_bottle_cost
  (tuna_packs : ℕ)
  (tuna_cost_per_pack : ℚ)
  (water_bottles : ℕ)
  (total_spent : ℚ)
  (different_goods_cost : ℚ)
  (h1 : tuna_packs = 5)
  (h2 : tuna_cost_per_pack = 2)
  (h3 : water_bottles = 4)
  (h4 : total_spent = 56)
  (h5 : different_goods_cost = 40) :
  (total_spent - different_goods_cost - tuna_packs * tuna_cost_per_pack) / water_bottles = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l2413_241322


namespace NUMINAMATH_CALUDE_apples_bought_l2413_241374

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : 
  initial = 17 → used = 2 → final = 38 → final - (initial - used) = 23 := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_l2413_241374


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l2413_241385

/-- Given a regular polygon with sum of interior angles 1260°, 
    prove that each exterior angle measures 40° -/
theorem exterior_angle_measure (n : ℕ) : 
  (n - 2) * 180 = 1260 → 360 / n = 40 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l2413_241385


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l2413_241324

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x / x

-- State the theorem
theorem tangent_line_and_inequality (x : ℝ) (h : x > 0) :
  -- Part 1: Equation of the tangent line
  (let slope := (Real.exp 2 * (2 - 1)) / (2^2);
   let point := (2, Real.exp 2 / 2);
   let tangent_line := fun x => slope * (x - point.1) + point.2;
   ∀ x, tangent_line x = Real.exp 2 * x / 4) ∧
  -- Part 2: Inequality
  f x > 2 * (x - Real.log x) :=
by sorry

end

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l2413_241324


namespace NUMINAMATH_CALUDE_binomial_properties_l2413_241364

variable (X : Nat → ℝ)

def binomial_distribution (n : Nat) (p : ℝ) (X : Nat → ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

def expectation (X : Nat → ℝ) : ℝ := sorry
def variance (X : Nat → ℝ) : ℝ := sorry

theorem binomial_properties :
  binomial_distribution 8 (1/2) X →
  expectation X = 4 ∧
  variance X = 2 ∧
  X 3 = X 5 := by sorry

end NUMINAMATH_CALUDE_binomial_properties_l2413_241364


namespace NUMINAMATH_CALUDE_weight_measurement_l2413_241381

def weights : List ℕ := [1, 3, 9, 27]

theorem weight_measurement (w : List ℕ := weights) :
  (∃ (S : List ℕ), S.sum = (List.sum w)) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ (List.sum w) → 
    ∃ (S : List ℕ), (∀ x ∈ S, x ∈ w) ∧ S.sum = n) :=
by sorry

end NUMINAMATH_CALUDE_weight_measurement_l2413_241381


namespace NUMINAMATH_CALUDE_trains_combined_length_l2413_241398

/-- The combined length of two trains crossing a platform -/
theorem trains_combined_length (speed_A speed_B : ℝ) (platform_length time : ℝ) : 
  speed_A = 72 * (5/18) → 
  speed_B = 54 * (5/18) → 
  platform_length = 210 → 
  time = 26 → 
  (speed_A + speed_B) * time - platform_length = 700 := by
  sorry


end NUMINAMATH_CALUDE_trains_combined_length_l2413_241398


namespace NUMINAMATH_CALUDE_andrew_kept_130_stickers_l2413_241342

def total_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def andrew_kept_stickers : ℕ := total_stickers - (daniel_stickers + (daniel_stickers + fred_extra_stickers))

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_130_stickers_l2413_241342


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2413_241339

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 72 → x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2413_241339


namespace NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l2413_241331

/-- A quadrilateral in the complex plane -/
structure ComplexQuadrilateral where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ
  z₄ : ℂ

/-- Predicate to check if a complex number has unit modulus -/
def hasUnitModulus (z : ℂ) : Prop := Complex.abs z = 1

/-- Predicate to check if a ComplexQuadrilateral is a rectangle -/
def isRectangle (q : ComplexQuadrilateral) : Prop :=
  -- Define what it means for a quadrilateral to be a rectangle
  -- This is a placeholder and should be properly defined
  True

/-- Main theorem: Under given conditions, the quadrilateral is a rectangle -/
theorem quadrilateral_is_rectangle (q : ComplexQuadrilateral) 
  (h₁ : hasUnitModulus q.z₁)
  (h₂ : hasUnitModulus q.z₂)
  (h₃ : hasUnitModulus q.z₃)
  (h₄ : hasUnitModulus q.z₄)
  (h_sum : q.z₁ + q.z₂ + q.z₃ + q.z₄ = 0) :
  isRectangle q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l2413_241331


namespace NUMINAMATH_CALUDE_min_value_expression_l2413_241348

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 3*b + 4*c = 1) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2*x + 3*y + 4*z = 1 → 
    1/a + 2/b + 3/c ≤ 1/x + 2/y + 3/z) ∧ 
  1/a + 2/b + 3/c = 20 + 4*Real.sqrt 3 + 20*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2413_241348
