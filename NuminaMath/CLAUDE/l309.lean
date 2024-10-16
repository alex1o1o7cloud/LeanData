import Mathlib

namespace NUMINAMATH_CALUDE_matrix_power_2020_l309_30983

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2020 :
  A ^ 2020 = !![1, 0; 6060, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2020_l309_30983


namespace NUMINAMATH_CALUDE_finite_valid_hexagon_angles_l309_30955

/-- Represents a sequence of interior angles of a hexagon -/
structure HexagonAngles where
  x : ℕ
  d : ℕ

/-- Checks if a given HexagonAngles satisfies the required conditions -/
def isValidHexagonAngles (angles : HexagonAngles) : Prop :=
  angles.x > 30 ∧
  angles.x + 5 * angles.d < 150 ∧
  2 * angles.x + 5 * angles.d = 240

/-- The set of all valid HexagonAngles -/
def validHexagonAnglesSet : Set HexagonAngles :=
  {angles | isValidHexagonAngles angles}

theorem finite_valid_hexagon_angles : Set.Finite validHexagonAnglesSet := by
  sorry

end NUMINAMATH_CALUDE_finite_valid_hexagon_angles_l309_30955


namespace NUMINAMATH_CALUDE_gcd_888_1147_l309_30935

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_888_1147_l309_30935


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l309_30997

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (10 * x₁^2 + 15 * x₁ - 17 = 0) →
  (10 * x₂^2 + 15 * x₂ - 17 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 113/20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l309_30997


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l309_30966

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l309_30966


namespace NUMINAMATH_CALUDE_derivative_at_one_l309_30959

open Real

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 1) + log x) :
  deriv f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l309_30959


namespace NUMINAMATH_CALUDE_sine_rule_application_l309_30916

theorem sine_rule_application (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 →
  a = 3 * b * Real.sin A →
  Real.sin B = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sine_rule_application_l309_30916


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l309_30962

/-- Proves that in a rhombus with an area of 127.5 cm² and one diagonal of 15 cm, 
    the length of the other diagonal is 17 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 127.5 → d1 = 15 → area = (d1 * d2) / 2 → d2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l309_30962


namespace NUMINAMATH_CALUDE_subset_from_intersection_l309_30954

theorem subset_from_intersection (M N : Set α) : M ∩ N = M → M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_subset_from_intersection_l309_30954


namespace NUMINAMATH_CALUDE_jake_present_weight_l309_30900

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℕ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := 224

theorem jake_present_weight : 
  (jake_weight - 20 = 2 * sister_weight) ∧ 
  (jake_weight + sister_weight = combined_weight) → 
  jake_weight = 156 := by
sorry

end NUMINAMATH_CALUDE_jake_present_weight_l309_30900


namespace NUMINAMATH_CALUDE_total_marbles_relation_l309_30979

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  red : ℝ
  blue : ℝ
  green : ℝ

/-- Conditions for the marble collection -/
def validCollection (c : MarbleCollection) : Prop :=
  c.red = 1.4 * c.blue ∧ c.green = 1.5 * c.red

/-- Total number of marbles in the collection -/
def totalMarbles (c : MarbleCollection) : ℝ :=
  c.red + c.blue + c.green

/-- Theorem stating the relationship between total marbles and red marbles -/
theorem total_marbles_relation (c : MarbleCollection) (h : validCollection c) :
    totalMarbles c = 3.21 * c.red := by
  sorry

#check total_marbles_relation

end NUMINAMATH_CALUDE_total_marbles_relation_l309_30979


namespace NUMINAMATH_CALUDE_competitive_examination_selection_l309_30971

theorem competitive_examination_selection (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_competitive_examination_selection_l309_30971


namespace NUMINAMATH_CALUDE_power_sum_value_l309_30931

theorem power_sum_value (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 1) :
  a^(m + 2*n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l309_30931


namespace NUMINAMATH_CALUDE_polynomial_simplification_l309_30942

theorem polynomial_simplification (q : ℝ) :
  (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q) =
  5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l309_30942


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l309_30951

/-- The quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

/-- Predicate indicating if f has a root for a given m -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

theorem sufficient_not_necessary :
  (∀ m, m > 7 → has_root m) ∧ 
  (∃ m, has_root m ∧ m ≤ 7) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l309_30951


namespace NUMINAMATH_CALUDE_second_half_wants_fifteen_l309_30934

/-- Represents the BBQ scenario with given conditions -/
structure BBQScenario where
  cooking_time_per_side : ℕ  -- Time to cook one side of a burger
  grill_capacity : ℕ         -- Number of burgers that can fit on the grill
  total_guests : ℕ           -- Total number of guests
  first_half_burgers : ℕ     -- Number of burgers each guest in the first half wants
  total_cooking_time : ℕ     -- Total time taken to cook all burgers

/-- Calculates the number of burgers wanted by the second half of guests -/
def second_half_burgers (scenario : BBQScenario) : ℕ :=
  let total_burgers := scenario.total_cooking_time / (2 * scenario.cooking_time_per_side) * scenario.grill_capacity
  let first_half_total := scenario.total_guests / 2 * scenario.first_half_burgers
  total_burgers - first_half_total

/-- Theorem stating that the second half of guests want 15 burgers -/
theorem second_half_wants_fifteen (scenario : BBQScenario) 
  (h1 : scenario.cooking_time_per_side = 4)
  (h2 : scenario.grill_capacity = 5)
  (h3 : scenario.total_guests = 30)
  (h4 : scenario.first_half_burgers = 2)
  (h5 : scenario.total_cooking_time = 72) : 
  second_half_burgers scenario = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_half_wants_fifteen_l309_30934


namespace NUMINAMATH_CALUDE_curve_in_second_quadrant_l309_30969

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  ∀ a : ℝ, (∀ x y : ℝ, C a x y → second_quadrant x y) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_curve_in_second_quadrant_l309_30969


namespace NUMINAMATH_CALUDE_highway_speed_l309_30923

/-- Prove that given the conditions, the average speed on the highway is 87 km/h -/
theorem highway_speed (total_distance : ℝ) (total_time : ℝ) (highway_time : ℝ) (city_time : ℝ) (city_speed : ℝ) :
  total_distance = 59 →
  total_time = 1 →
  highway_time = 1/3 →
  city_time = 2/3 →
  city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 := by
sorry

end NUMINAMATH_CALUDE_highway_speed_l309_30923


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l309_30926

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l309_30926


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_not_neg_6_4_l309_30999

/-- Two lines in 3D space --/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew --/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point.2.2 = b ∧ 
  ¬∃ (t u : ℝ), 
    (l1.point.1 + t * l1.direction.1 = l2.point.1 + u * l2.direction.1) ∧
    (l1.point.2.1 + t * l1.direction.2.1 = l2.point.2.1 + u * l2.direction.2.1) ∧
    (b + t * l1.direction.2.2 = l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_b_not_neg_6_4 :
  ∀ (b : ℝ), are_skew 
    (Line3D.mk (2, 3, b) (3, 4, 5)) 
    (Line3D.mk (5, 2, 1) (6, 3, 2))
  ↔ b ≠ -6.4 := by sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_not_neg_6_4_l309_30999


namespace NUMINAMATH_CALUDE_number_above_265_l309_30975

/-- Represents the pyramid-like array of numbers -/
def pyramid_array (n : ℕ) : List ℕ :=
  List.range (n * n + 1) -- This generates a list of numbers from 0 to n^2

/-- The number of elements in the nth row of the pyramid -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- The starting number of the nth row -/
def row_start (n : ℕ) : ℕ := (n - 1) ^ 2 + 1

/-- The position of a number in its row -/
def position_in_row (x : ℕ) : ℕ :=
  x - row_start (Nat.sqrt x) + 1

/-- The number directly above a given number in the pyramid -/
def number_above (x : ℕ) : ℕ :=
  row_start (Nat.sqrt x - 1) + position_in_row x - 1

theorem number_above_265 :
  number_above 265 = 234 := by sorry

end NUMINAMATH_CALUDE_number_above_265_l309_30975


namespace NUMINAMATH_CALUDE_base_number_proof_l309_30988

theorem base_number_proof (e : ℕ) (x : ℕ) : 
  e = x^19 ∧ e % 10 = 7 → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_base_number_proof_l309_30988


namespace NUMINAMATH_CALUDE_bike_distance_l309_30903

/-- The distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A bike traveling at 50 m/s for 7 seconds covers a distance of 350 meters -/
theorem bike_distance : distance_traveled 50 7 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_l309_30903


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l309_30957

open Set

theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  let M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
  let N : Set ℝ := {y | y < a}
  (M ∩ N).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l309_30957


namespace NUMINAMATH_CALUDE_AF_AT_ratio_l309_30978

-- Define the triangle ABC and points D, E, F, T
variable (A B C D E F T : ℝ × ℝ)

-- Define the conditions
axiom on_AB : ∃ t : ℝ, D = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1
axiom on_AC : ∃ s : ℝ, E = (1 - s) • A + s • C ∧ 0 ≤ s ∧ s ≤ 1
axiom on_DE : ∃ r : ℝ, F = (1 - r) • D + r • E ∧ 0 ≤ r ∧ r ≤ 1
axiom on_AT : ∃ q : ℝ, F = (1 - q) • A + q • T ∧ 0 ≤ q ∧ q ≤ 1

axiom AD_length : dist A D = 1
axiom DB_length : dist D B = 4
axiom AE_length : dist A E = 3
axiom EC_length : dist E C = 3

axiom angle_bisector : 
  dist B T / dist T C = dist A B / dist A C

-- Define the theorem to be proved
theorem AF_AT_ratio : 
  dist A F / dist A T = 11 / 40 :=
sorry

end NUMINAMATH_CALUDE_AF_AT_ratio_l309_30978


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l309_30992

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ digit_sum year = 15

theorem first_year_after_2010_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l309_30992


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l309_30902

theorem quadratic_inequality_solution_set (m : ℝ) : 
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l309_30902


namespace NUMINAMATH_CALUDE_students_on_south_side_l309_30982

theorem students_on_south_side (total : ℕ) (difference : ℕ) (south : ℕ) : 
  total = 41 → difference = 3 → south = total / 2 + difference / 2 → south = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_on_south_side_l309_30982


namespace NUMINAMATH_CALUDE_nell_card_difference_l309_30960

/-- Represents the number of cards Nell has -/
structure CardCounts where
  initial_baseball : Nat
  initial_ace : Nat
  final_baseball : Nat
  final_ace : Nat

/-- Calculates the difference between Ace cards and baseball cards -/
def ace_baseball_difference (counts : CardCounts) : Int :=
  counts.final_ace - counts.final_baseball

/-- Theorem stating the difference between Ace cards and baseball cards -/
theorem nell_card_difference (counts : CardCounts) 
  (h1 : counts.initial_baseball = 239)
  (h2 : counts.initial_ace = 38)
  (h3 : counts.final_baseball = 111)
  (h4 : counts.final_ace = 376) :
  ace_baseball_difference counts = 265 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l309_30960


namespace NUMINAMATH_CALUDE_valid_sequences_characterization_l309_30949

/-- Represents the possible weather observations: Plus for no rain, Minus for rain -/
inductive WeatherObservation
| Plus : WeatherObservation
| Minus : WeatherObservation

/-- Represents a sequence of three weather observations -/
structure ObservationSequence :=
  (first : WeatherObservation)
  (second : WeatherObservation)
  (third : WeatherObservation)

/-- Determines if a sequence is valid based on the third student's rule -/
def isValidSequence (seq : ObservationSequence) : Prop :=
  match seq.third with
  | WeatherObservation.Minus => 
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Minus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Minus)
  | WeatherObservation.Plus =>
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus)

/-- The set of all valid observation sequences -/
def validSequences : Set ObservationSequence :=
  { seq | isValidSequence seq }

theorem valid_sequences_characterization :
  validSequences = {
    ⟨WeatherObservation.Plus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Minus⟩
  } := by
  sorry

#check valid_sequences_characterization

end NUMINAMATH_CALUDE_valid_sequences_characterization_l309_30949


namespace NUMINAMATH_CALUDE_talent_show_gender_difference_l309_30985

theorem talent_show_gender_difference (total : ℕ) (girls : ℕ) :
  total = 34 →
  girls = 28 →
  girls > total - girls →
  girls - (total - girls) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_talent_show_gender_difference_l309_30985


namespace NUMINAMATH_CALUDE_f_is_odd_ellipse_y_axis_iff_l309_30901

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2))

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Define the ellipse equation
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1

-- Theorem 2: Necessary and sufficient condition for ellipse with foci on y-axis
theorem ellipse_y_axis_iff (m n : ℝ) : 
  is_ellipse_y_axis m n ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_ellipse_y_axis_iff_l309_30901


namespace NUMINAMATH_CALUDE_hexagon_triangle_quadrilateral_area_ratio_l309_30970

/-- A regular hexagon with vertices labeled A to F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A quadrilateral -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a polygon -/
noncomputable def area {n : ℕ} (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem hexagon_triangle_quadrilateral_area_ratio
  (h : RegularHexagon)
  (triangles : Fin 6 → EquilateralTriangle)
  (quad : Quadrilateral) :
  (∀ i, area (triangles i).vertices = area (triangles 0).vertices) →
  (quad.vertices 0 = h.vertices 0) →
  (quad.vertices 1 = h.vertices 2) →
  (quad.vertices 2 = h.vertices 4) →
  (quad.vertices 3 = h.vertices 1) →
  area (triangles 0).vertices / area quad.vertices = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_quadrilateral_area_ratio_l309_30970


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l309_30933

/-- A cube with volume 8x and surface area 2x has x = 1728 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l309_30933


namespace NUMINAMATH_CALUDE_flight_cost_calculation_l309_30922

def trip_expenses (initial_savings hotel_cost food_cost remaining_money : ℕ) : Prop :=
  ∃ flight_cost : ℕ, 
    initial_savings = hotel_cost + food_cost + flight_cost + remaining_money

theorem flight_cost_calculation (initial_savings hotel_cost food_cost remaining_money : ℕ) 
  (h : trip_expenses initial_savings hotel_cost food_cost remaining_money) :
  ∃ flight_cost : ℕ, flight_cost = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_flight_cost_calculation_l309_30922


namespace NUMINAMATH_CALUDE_chess_group_players_l309_30918

theorem chess_group_players (n : ℕ) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < n * (n - 1) / 2) →
  (∀ (game : ℕ), game < n * (n - 1) / 2 → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) →
  n * (n - 1) / 2 = 105 →
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_group_players_l309_30918


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l309_30936

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l309_30936


namespace NUMINAMATH_CALUDE_complex_multiplication_l309_30937

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l309_30937


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l309_30913

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| + |a*x - a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ↔ x ≥ 2.5 ∨ x ≤ 0.5 := by sorry

-- Theorem 2: Range of a when solution set is ℝ
theorem range_of_a_when_solution_set_is_real :
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≥ 2)) → (∀ a : ℝ, a > 0 → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l309_30913


namespace NUMINAMATH_CALUDE_bamboo_problem_l309_30911

/-- 
Given a geometric sequence of 9 terms where the sum of the first 3 terms is 2 
and the sum of the last 3 terms is 128, the 5th term is equal to 32/7.
-/
theorem bamboo_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 + a 2 + a 3 = 2 →         -- sum of first 3 terms
  a 7 + a 8 + a 9 = 128 →       -- sum of last 3 terms
  a 5 = 32 / 7 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l309_30911


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_neg_one_l309_30968

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x+m)(x+1) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (x + m) * (x + 1)

/-- If f(x) = (x+m)(x+1) is an even function, then m = -1 -/
theorem even_function_implies_m_equals_neg_one :
  ∀ m : ℝ, IsEven (f m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_neg_one_l309_30968


namespace NUMINAMATH_CALUDE_three_equal_differences_exist_l309_30996

theorem three_equal_differences_exist (a : Fin 19 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i < 91) :
  ∃ i j k l m n, i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    i ≠ k ∧ i ≠ m ∧ k ≠ m ∧
    a j - a i = a l - a k ∧ a n - a m = a j - a i :=
sorry

end NUMINAMATH_CALUDE_three_equal_differences_exist_l309_30996


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l309_30984

def total_amount : ℚ := 180
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def leftover_amount : ℚ := 24

theorem museum_ticket_fraction :
  let spent_amount := total_amount - leftover_amount
  let sandwich_cost := sandwich_fraction * total_amount
  let book_cost := book_fraction * total_amount
  let museum_ticket_cost := spent_amount - sandwich_cost - book_cost
  museum_ticket_cost / total_amount = 1/6 := by sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l309_30984


namespace NUMINAMATH_CALUDE_hiking_team_participants_l309_30961

/-- The number of gloves needed for the hiking team -/
def total_gloves : ℕ := 126

/-- The number of gloves each participant needs -/
def gloves_per_participant : ℕ := 2

/-- The number of participants in the hiking team -/
def num_participants : ℕ := total_gloves / gloves_per_participant

theorem hiking_team_participants : num_participants = 63 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l309_30961


namespace NUMINAMATH_CALUDE_sphere_volume_in_cone_l309_30973

/-- A right circular cone with a sphere inscribed inside it -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the given cone -/
theorem sphere_volume_in_cone (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90) : 
  sphere_volume cone = 2304 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cone_l309_30973


namespace NUMINAMATH_CALUDE_train_capacity_ratio_l309_30914

def train_problem (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) : Prop :=
  red_boxcars = 3 ∧
  blue_boxcars = 4 ∧
  black_boxcars = 7 ∧
  black_capacity = 4000 ∧
  red_multiplier = 3 ∧
  total_capacity = 132000 ∧
  ∃ (blue_capacity : ℕ),
    red_boxcars * (red_multiplier * blue_capacity) +
    blue_boxcars * blue_capacity +
    black_boxcars * black_capacity = total_capacity ∧
    2 * black_capacity = blue_capacity

theorem train_capacity_ratio 
  (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) :
  train_problem red_boxcars blue_boxcars black_boxcars black_capacity red_multiplier total_capacity →
  ∃ (blue_capacity : ℕ), 2 * black_capacity = blue_capacity :=
by sorry

end NUMINAMATH_CALUDE_train_capacity_ratio_l309_30914


namespace NUMINAMATH_CALUDE_conic_sections_from_equation_l309_30967

/-- The equation y^4 - 8x^4 = 4y^2 - 4 represents two conic sections -/
theorem conic_sections_from_equation :
  ∃ (f g : ℝ → ℝ → Prop),
    (∀ x y, y^4 - 8*x^4 = 4*y^2 - 4 ↔ f x y ∨ g x y) ∧
    (∃ a b c d e : ℝ, ∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    (∃ a b c d e : ℝ, ∀ x y, g x y ↔ (x^2 / c^2) + (y^2 / d^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_from_equation_l309_30967


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l309_30986

/-- Represents the number of students in each grade -/
structure GradePopulation where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.first + pop.second + pop.third
  { first := (totalSample * pop.first + totalPop - 1) / totalPop,
    second := (totalSample * pop.second + totalPop - 1) / totalPop,
    third := (totalSample * pop.third + totalPop - 1) / totalPop }

theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 680 720
  let sample := stratifiedSample pop 50
  sample.first = 15 ∧ sample.second = 17 ∧ sample.third = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l309_30986


namespace NUMINAMATH_CALUDE_trigonometric_identity_l309_30939

theorem trigonometric_identity (θ : Real) 
  (h : Real.sin (π / 3 - θ) = 1 / 2) : 
  Real.cos (π / 6 + θ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l309_30939


namespace NUMINAMATH_CALUDE_race_outcomes_l309_30952

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the number of permutations of n items -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different 1st-2nd-3rd place outcomes in a race with 6 participants,
    where one specific participant is guaranteed to be in the top 3 and there are no ties -/
theorem race_outcomes : 
  top_positions * choose (num_participants - 1) (top_positions - 1) * permutations (top_positions - 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l309_30952


namespace NUMINAMATH_CALUDE_bucket_weight_l309_30965

theorem bucket_weight (c d : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = c ∧ x + 1/3 * y = d) → 
  (∃ (full_weight : ℝ), full_weight = (8*c - 3*d) / 5) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l309_30965


namespace NUMINAMATH_CALUDE_expression_evaluation_l309_30915

theorem expression_evaluation : 23 - 17 - (-7) + (-16) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l309_30915


namespace NUMINAMATH_CALUDE_fill_time_with_leak_l309_30944

/-- Time taken to fill a tank with two pipes and a leak -/
theorem fill_time_with_leak (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 :=
by sorry

end NUMINAMATH_CALUDE_fill_time_with_leak_l309_30944


namespace NUMINAMATH_CALUDE_correct_proposition_l309_30941

def p : Prop := ∀ x : ℝ, x^2 - x + 2 < 0

def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

theorem correct_proposition : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l309_30941


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l309_30919

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 5)
def a (x : ℝ) : ℝ × ℝ := (x, 6)

def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem parallel_vectors_x_value (x : ℝ) :
  (∃ k : ℝ, a x = k • vecAB) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l309_30919


namespace NUMINAMATH_CALUDE_age_difference_l309_30950

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l309_30950


namespace NUMINAMATH_CALUDE_road_division_l309_30917

theorem road_division (a b c : ℝ) : 
  a + b + c = 28 →
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c / 2) - a / 2 = 16 →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_road_division_l309_30917


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l309_30932

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, -2; 0, -1]

theorem inverse_of_B_cubed :
  let B := B_inv⁻¹
  (B^3)⁻¹ = !![27, -24; 0, -1] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l309_30932


namespace NUMINAMATH_CALUDE_paco_cookies_l309_30976

def cookie_problem (initial_cookies : ℕ) (given_to_friend1 : ℕ) (given_to_friend2 : ℕ) (eaten : ℕ) : Prop :=
  let total_given := given_to_friend1 + given_to_friend2
  eaten - total_given = 0

theorem paco_cookies : cookie_problem 100 15 25 40 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l309_30976


namespace NUMINAMATH_CALUDE_nathan_gave_six_apples_l309_30958

/-- The number of apples Nathan gave to Annie -/
def apples_from_nathan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

theorem nathan_gave_six_apples :
  apples_from_nathan 6 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nathan_gave_six_apples_l309_30958


namespace NUMINAMATH_CALUDE_trajectory_of_P_l309_30930

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 11 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 0)

-- Define the moving point B on circle F
def point_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | circle_F p.1 p.2}

-- Define the perpendicular bisector of AB
def perp_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {P : ℝ × ℝ | (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2}

-- Define point P
def point_P (B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | P ∈ perp_bisector point_A B ∧ 
               ∃ t : ℝ, P = (t * B.1 + (1-t) * 1, t * B.2)}

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ B ∈ point_B, P ∈ point_P B) → 
  P.1^2 / 3 + P.2^2 / 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l309_30930


namespace NUMINAMATH_CALUDE_inequality_holds_l309_30908

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) :
  a / c^2 > b / c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l309_30908


namespace NUMINAMATH_CALUDE_sum_in_B_l309_30995

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define set C (although not used in the theorem, it's part of the original problem)
def C : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end NUMINAMATH_CALUDE_sum_in_B_l309_30995


namespace NUMINAMATH_CALUDE_sum_range_l309_30906

theorem sum_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 9/b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l309_30906


namespace NUMINAMATH_CALUDE_red_card_events_l309_30929

-- Define the set of colors
inductive Color : Type
| Red | Black | White | Blue

-- Define the set of individuals
inductive Person : Type
| A | B | C | D

-- Define a distribution as a function from Person to Color
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Color.Red

-- Theorem: A_gets_red and B_gets_red are mutually exclusive but not complementary
theorem red_card_events (d : Distribution) :
  (¬ (A_gets_red d ∧ B_gets_red d)) ∧
  (∃ (d : Distribution), ¬ A_gets_red d ∧ ¬ B_gets_red d) :=
by sorry

end NUMINAMATH_CALUDE_red_card_events_l309_30929


namespace NUMINAMATH_CALUDE_third_player_win_probability_is_one_fifteenth_l309_30993

/-- Represents the probability of the third player winning in a four-player coin-flipping game -/
def third_player_win_probability : ℚ := 1 / 15

/-- The game has four players taking turns -/
def number_of_players : ℕ := 4

/-- The position of the player we're calculating the probability for -/
def target_player_position : ℕ := 3

/-- Theorem stating that the probability of the third player winning is 1/15 -/
theorem third_player_win_probability_is_one_fifteenth :
  third_player_win_probability = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_third_player_win_probability_is_one_fifteenth_l309_30993


namespace NUMINAMATH_CALUDE_round_robin_tournament_probability_l309_30921

def num_teams : ℕ := 5

-- Define the type for tournament outcomes
def TournamentOutcome := Fin num_teams → Fin num_teams

-- Function to check if an outcome has unique win counts
def has_unique_win_counts (outcome : TournamentOutcome) : Prop :=
  ∀ i j, i ≠ j → outcome i ≠ outcome j

-- Total number of possible outcomes
def total_outcomes : ℕ := 2^(num_teams * (num_teams - 1) / 2)

-- Number of favorable outcomes (where no two teams have the same number of wins)
def favorable_outcomes : ℕ := Nat.factorial num_teams

-- The probability we want to prove
def target_probability : ℚ := favorable_outcomes / total_outcomes

theorem round_robin_tournament_probability :
  target_probability = 15 / 128 := by sorry

end NUMINAMATH_CALUDE_round_robin_tournament_probability_l309_30921


namespace NUMINAMATH_CALUDE_equation_solution_l309_30980

theorem equation_solution : ∃ x : ℚ, (2 / 5 : ℚ) - (1 / 7 : ℚ) = 1 / x ∧ x = 35 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l309_30980


namespace NUMINAMATH_CALUDE_symmetric_curve_equation_l309_30956

/-- Given a curve C defined by F(x, y) = 0 and a point of symmetry (a, b),
    the equation of the curve symmetric to C about (a, b) is F(2a-x, 2b-y) = 0 -/
theorem symmetric_curve_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  (∀ x y, F x y = 0 ↔ F (2*a - x) (2*b - y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_equation_l309_30956


namespace NUMINAMATH_CALUDE_polynomial_inequality_l309_30945

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Predicate to check if a function is a polynomial with integer coefficients -/
def is_int_polynomial (p : ℤ → ℤ) : Prop := sorry

theorem polynomial_inequality (p : IntPolynomial) (n : ℤ) 
  (h_poly : is_int_polynomial p)
  (h_ineq : p (-n) < p n ∧ p n < n) : 
  p (-n) < -n := by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l309_30945


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_l309_30964

theorem power_of_two_plus_one_square (k z : ℕ) :
  2^k + 1 = z^2 → k = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_l309_30964


namespace NUMINAMATH_CALUDE_ball_probability_l309_30981

theorem ball_probability (red green yellow total : ℕ) (p_green : ℚ) :
  red = 8 →
  green = 10 →
  total = red + green + yellow →
  p_green = 1 / 4 →
  p_green = green / total →
  yellow / total = 11 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l309_30981


namespace NUMINAMATH_CALUDE_sum_70_is_negative_350_l309_30947

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  /-- First term of the progression -/
  a : ℚ
  /-- Common difference of the progression -/
  d : ℚ
  /-- Sum of first 20 terms is 200 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (20 - 1) * d) = 200
  /-- Sum of first 50 terms is 50 -/
  sum_50 : (50 : ℚ) / 2 * (2 * a + (50 - 1) * d) = 50

/-- The sum of the first 70 terms of the arithmetic progression is -350 -/
theorem sum_70_is_negative_350 (ap : ArithmeticProgression) :
  (70 : ℚ) / 2 * (2 * ap.a + (70 - 1) * ap.d) = -350 := by
  sorry

end NUMINAMATH_CALUDE_sum_70_is_negative_350_l309_30947


namespace NUMINAMATH_CALUDE_pta_fundraiser_remaining_money_l309_30925

theorem pta_fundraiser_remaining_money (initial_amount : ℝ) : 
  initial_amount = 400 → 
  (initial_amount - initial_amount / 4) / 2 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pta_fundraiser_remaining_money_l309_30925


namespace NUMINAMATH_CALUDE_polynomial_simplification_l309_30907

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 8) + (2*x - 3)*(x - 2)*(x + 4) = 3*x^3 - 8*x^2 + 5*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l309_30907


namespace NUMINAMATH_CALUDE_min_angle_B_in_special_triangle_l309_30998

open Real

theorem min_angle_B_in_special_triangle (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  ∃ k : ℝ, tan A + k = (1 + sqrt 2) * tan B ∧ (1 + sqrt 2) * tan B + k = tan C →
  π / 4 ≤ B :=
by sorry

end NUMINAMATH_CALUDE_min_angle_B_in_special_triangle_l309_30998


namespace NUMINAMATH_CALUDE_initial_players_l309_30991

theorem initial_players (quit_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) :
  quit_players = 5 →
  lives_per_player = 5 →
  total_lives = 15 →
  ∃ initial_players : ℕ, initial_players = 8 ∧ (initial_players - quit_players) * lives_per_player = total_lives :=
by sorry

end NUMINAMATH_CALUDE_initial_players_l309_30991


namespace NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l309_30989

/-- Represents the scenario where a person is between two points -/
structure WalkRideScenario where
  /-- Distance from the person to the apartment -/
  dist_to_apartment : ℝ
  /-- Distance from the person to the library -/
  dist_to_library : ℝ
  /-- Walking speed -/
  walking_speed : ℝ
  /-- Assumption that distances and speed are positive -/
  dist_apartment_pos : 0 < dist_to_apartment
  dist_library_pos : 0 < dist_to_library
  speed_pos : 0 < walking_speed
  /-- Assumption that the person is between the apartment and library -/
  between_points : dist_to_apartment + dist_to_library > 0

/-- The theorem stating that under the given conditions, the ratio of distances is 2/3 -/
theorem distance_ratio_is_two_thirds (scenario : WalkRideScenario) :
  scenario.dist_to_library / scenario.walking_speed =
  scenario.dist_to_apartment / scenario.walking_speed +
  (scenario.dist_to_apartment + scenario.dist_to_library) / (5 * scenario.walking_speed) →
  scenario.dist_to_apartment / scenario.dist_to_library = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l309_30989


namespace NUMINAMATH_CALUDE_triangle_right_angled_l309_30920

theorem triangle_right_angled (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π) 
  (h5 : α + β < π) (h6 : Real.sin α = Real.cos β) : 
  ∃ γ : Real, γ = π / 2 ∧ α + β + γ = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l309_30920


namespace NUMINAMATH_CALUDE_total_study_time_l309_30946

def study_time (wednesday thursday friday weekend : ℕ) : Prop :=
  (wednesday = 2) ∧
  (thursday = 3 * wednesday) ∧
  (friday = thursday / 2) ∧
  (weekend = wednesday + thursday + friday) ∧
  (wednesday + thursday + friday + weekend = 22)

theorem total_study_time :
  ∃ (wednesday thursday friday weekend : ℕ),
    study_time wednesday thursday friday weekend :=
by sorry

end NUMINAMATH_CALUDE_total_study_time_l309_30946


namespace NUMINAMATH_CALUDE_middle_guard_hours_l309_30943

theorem middle_guard_hours (total_hours : ℕ) (num_guards : ℕ) (first_guard_hours : ℕ) (last_guard_hours : ℕ) :
  total_hours = 9 ∧ num_guards = 4 ∧ first_guard_hours = 3 ∧ last_guard_hours = 2 →
  (total_hours - first_guard_hours - last_guard_hours) / (num_guards - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_middle_guard_hours_l309_30943


namespace NUMINAMATH_CALUDE_mod_thirteen_equiv_l309_30948

theorem mod_thirteen_equiv (n : ℤ) : 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2345 [ZMOD 13] → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_equiv_l309_30948


namespace NUMINAMATH_CALUDE_prob_same_suit_60_card_deck_l309_30910

/-- A deck of cards with a specified number of ranks and suits. -/
structure Deck :=
  (num_ranks : ℕ)
  (num_suits : ℕ)

/-- The probability of drawing two cards of the same suit from a deck. -/
def prob_same_suit (d : Deck) : ℚ :=
  if d.num_ranks * d.num_suits = 0 then 0
  else (d.num_ranks - 1) / (d.num_ranks * d.num_suits - 1)

/-- Theorem stating the probability of drawing two cards of the same suit
    from a 60-card deck with 15 ranks and 4 suits. -/
theorem prob_same_suit_60_card_deck :
  prob_same_suit ⟨15, 4⟩ = 14 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_suit_60_card_deck_l309_30910


namespace NUMINAMATH_CALUDE_darla_electricity_bill_l309_30963

-- Define the cost per watt in cents
def cost_per_watt : ℕ := 400

-- Define the amount of electricity used in watts
def electricity_used : ℕ := 300

-- Define the late fee in cents
def late_fee : ℕ := 15000

-- Define the total cost in cents
def total_cost : ℕ := cost_per_watt * electricity_used + late_fee

-- Theorem statement
theorem darla_electricity_bill : total_cost = 135000 := by
  sorry

end NUMINAMATH_CALUDE_darla_electricity_bill_l309_30963


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l309_30977

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l309_30977


namespace NUMINAMATH_CALUDE_existence_of_x_with_abs_f_ge_2_l309_30924

theorem existence_of_x_with_abs_f_ge_2 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_with_abs_f_ge_2_l309_30924


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l309_30904

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l309_30904


namespace NUMINAMATH_CALUDE_smallest_common_pet_count_l309_30940

theorem smallest_common_pet_count : ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 15 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 15 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_pet_count_l309_30940


namespace NUMINAMATH_CALUDE_loan_interest_time_l309_30912

/-- Given two loans and their interest rates, calculate the time needed to reach a specific total interest. -/
theorem loan_interest_time (loan1 loan2 rate1 rate2 total_interest : ℚ) : 
  loan1 = 1000 →
  loan2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  total_interest = 350 →
  ∃ (time : ℚ), time * (loan1 * rate1 + loan2 * rate2) = total_interest ∧ time = 7 / 2 := by
  sorry

#check loan_interest_time

end NUMINAMATH_CALUDE_loan_interest_time_l309_30912


namespace NUMINAMATH_CALUDE_square_trinomial_equality_l309_30974

theorem square_trinomial_equality : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_trinomial_equality_l309_30974


namespace NUMINAMATH_CALUDE_circle_center_l309_30927

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenCircleEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center :
  ∃ r, ∀ x y, GivenCircleEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end NUMINAMATH_CALUDE_circle_center_l309_30927


namespace NUMINAMATH_CALUDE_bayswater_volleyball_club_members_l309_30938

theorem bayswater_volleyball_club_members : 
  let knee_pad_cost : ℕ := 6
  let jersey_cost : ℕ := knee_pad_cost + 7
  let member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost)
  let total_expenditure : ℕ := 3120
  total_expenditure / member_cost = 82 :=
by sorry

end NUMINAMATH_CALUDE_bayswater_volleyball_club_members_l309_30938


namespace NUMINAMATH_CALUDE_courier_journey_l309_30987

/-- The specified time for the courier's journey in minutes -/
def specified_time : ℝ := 40

/-- The total distance the courier traveled in kilometers -/
def total_distance : ℝ := 36

/-- The speed at which the courier arrives early in km/min -/
def early_speed : ℝ := 1.2

/-- The speed at which the courier arrives late in km/min -/
def late_speed : ℝ := 0.8

/-- The time by which the courier arrives early in minutes -/
def early_time : ℝ := 10

/-- The time by which the courier arrives late in minutes -/
def late_time : ℝ := 5

theorem courier_journey :
  early_speed * (specified_time - early_time) = late_speed * (specified_time + late_time) ∧
  total_distance = early_speed * (specified_time - early_time) :=
by sorry

end NUMINAMATH_CALUDE_courier_journey_l309_30987


namespace NUMINAMATH_CALUDE_probability_three_primes_six_dice_l309_30953

-- Define a 12-sided die
def die := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on one die
def prob_prime : ℚ := (primes.card : ℚ) / (die.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def prob_not_prime : ℚ := 1 - prob_prime

-- Define the number of ways to choose 3 dice from 6
def choose_3_from_6 : ℕ := Nat.choose 6 3

-- Statement of the theorem
theorem probability_three_primes_six_dice :
  (choose_3_from_6 : ℚ) * prob_prime^3 * prob_not_prime^3 = 857500 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_six_dice_l309_30953


namespace NUMINAMATH_CALUDE_pq_equals_10_l309_30972

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop := sorry
def anglePRQ (t : Triangle) : ℝ := sorry
def lengthPR (t : Triangle) : ℝ := sorry
def lengthPQ (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem pq_equals_10 (t : Triangle) 
  (h1 : isRightAngled t) 
  (h2 : anglePRQ t = 45) 
  (h3 : lengthPR t = 10) : 
  lengthPQ t = 10 := by sorry

end NUMINAMATH_CALUDE_pq_equals_10_l309_30972


namespace NUMINAMATH_CALUDE_interchangeable_statements_l309_30905

-- Define the concept of geometric objects
inductive GeometricObject
| Line
| Plane

-- Define the relationships between geometric objects
inductive Relationship
| Perpendicular
| Parallel

-- Define a geometric statement
structure GeometricStatement where
  obj1 : GeometricObject
  obj2 : GeometricObject
  rel1 : Relationship
  obj3 : GeometricObject
  rel2 : Relationship

-- Define the concept of an interchangeable statement
def isInterchangeable (s : GeometricStatement) : Prop :=
  (s.obj1 = GeometricObject.Line ∧ s.obj2 = GeometricObject.Plane) ∨
  (s.obj1 = GeometricObject.Plane ∧ s.obj2 = GeometricObject.Line)

-- Define the four statements
def statement1 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement2 : GeometricStatement :=
  { obj1 := GeometricObject.Plane
  , obj2 := GeometricObject.Plane
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement3 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Line
  , rel2 := Relationship.Parallel }

def statement4 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

-- Theorem to prove
theorem interchangeable_statements :
  isInterchangeable statement1 ∧ isInterchangeable statement3 ∧
  ¬isInterchangeable statement2 ∧ ¬isInterchangeable statement4 :=
sorry

end NUMINAMATH_CALUDE_interchangeable_statements_l309_30905


namespace NUMINAMATH_CALUDE_parallelogram_to_triangle_impossibility_l309_30909

theorem parallelogram_to_triangle_impossibility (a : ℝ) (h : a > 0) :
  ¬ (a + a > 2*a ∧ a + 2*a > a ∧ 2*a + a > a) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_to_triangle_impossibility_l309_30909


namespace NUMINAMATH_CALUDE_horse_distance_in_day_l309_30994

/-- The distance a horse can run in one day -/
def horse_distance (speed : ℝ) (hours_per_day : ℝ) : ℝ :=
  speed * hours_per_day

/-- Theorem: A horse running at 10 miles/hour for 24 hours covers 240 miles -/
theorem horse_distance_in_day :
  horse_distance 10 24 = 240 := by
  sorry

end NUMINAMATH_CALUDE_horse_distance_in_day_l309_30994


namespace NUMINAMATH_CALUDE_remainder_of_binary_division_l309_30928

-- Define the binary number
def binary_number : ℕ := 101100110011

-- Define the divisor
def divisor : ℕ := 8

-- Theorem statement
theorem remainder_of_binary_division :
  binary_number % divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_division_l309_30928


namespace NUMINAMATH_CALUDE_log_function_passes_through_point_l309_30990

theorem log_function_passes_through_point 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a - 2
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_passes_through_point_l309_30990
