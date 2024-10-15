import Mathlib

namespace NUMINAMATH_CALUDE_remaining_soup_feeds_six_adults_l417_41769

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Proves that given 5 cans of soup, where each can feeds 3 adults or 5 children,
    if 15 children are fed, the remaining soup will feed 6 adults -/
theorem remaining_soup_feeds_six_adults 
  (can : SoupCan) 
  (h1 : can.adults = 3) 
  (h2 : can.children = 5) 
  (total_cans : ℕ) 
  (h3 : total_cans = 5) 
  (children_fed : ℕ) 
  (h4 : children_fed = 15) : 
  (total_cans - (children_fed / can.children)) * can.adults = 6 := by
sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_six_adults_l417_41769


namespace NUMINAMATH_CALUDE_ricks_sisters_cards_l417_41701

/-- The number of cards Rick's sisters receive -/
def cards_per_sister (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (num_friends * cards_per_friend)
  remaining_cards / num_sisters

/-- Proof that each of Rick's sisters received 3 cards -/
theorem ricks_sisters_cards : 
  cards_per_sister 130 15 13 8 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ricks_sisters_cards_l417_41701


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_condition_l417_41705

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_intersection_condition (a : ℝ) :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (A ∩ C a ≠ ∅ → a < 8) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_condition_l417_41705


namespace NUMINAMATH_CALUDE_quadrilateral_Q₁PNF_is_cyclic_l417_41787

/-- Two circles with points on them and their intersections -/
structure TwoCirclesConfig where
  /-- The first circle -/
  circle1 : Set (ℝ × ℝ)
  /-- The second circle -/
  circle2 : Set (ℝ × ℝ)
  /-- Point Q₁, an intersection of the two circles -/
  Q₁ : ℝ × ℝ
  /-- Point Q₂, another intersection of the two circles -/
  Q₂ : ℝ × ℝ
  /-- Point A on the first circle -/
  A : ℝ × ℝ
  /-- Point B on the first circle -/
  B : ℝ × ℝ
  /-- Point C, where AQ₂ intersects circle2 again -/
  C : ℝ × ℝ
  /-- Point F on arc Q₁Q₂ of circle1, inside circle2 -/
  F : ℝ × ℝ
  /-- Point P, intersection of AF and BQ₁ -/
  P : ℝ × ℝ
  /-- Point N, where PC intersects circle2 again -/
  N : ℝ × ℝ

  /-- Q₁ and Q₂ are on both circles -/
  h1 : Q₁ ∈ circle1 ∧ Q₁ ∈ circle2
  h2 : Q₂ ∈ circle1 ∧ Q₂ ∈ circle2
  /-- A and B are on circle1 -/
  h3 : A ∈ circle1
  h4 : B ∈ circle1
  /-- C is on circle2 -/
  h5 : C ∈ circle2
  /-- F is on arc Q₁Q₂ of circle1, inside circle2 -/
  h6 : F ∈ circle1
  /-- N is on circle2 -/
  h7 : N ∈ circle2

/-- The main theorem: Quadrilateral Q₁PNF is cyclic -/
theorem quadrilateral_Q₁PNF_is_cyclic (config : TwoCirclesConfig) :
  ∃ (circle : Set (ℝ × ℝ)), config.Q₁ ∈ circle ∧ config.P ∈ circle ∧ config.N ∈ circle ∧ config.F ∈ circle :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_Q₁PNF_is_cyclic_l417_41787


namespace NUMINAMATH_CALUDE_empty_tank_weight_is_80_l417_41782

/-- The weight of an empty water tank --/
def empty_tank_weight (tank_capacity : ℝ) (fill_percentage : ℝ) (water_weight : ℝ) (filled_weight : ℝ) : ℝ :=
  filled_weight - (tank_capacity * fill_percentage * water_weight)

/-- Theorem stating the weight of the empty tank --/
theorem empty_tank_weight_is_80 :
  empty_tank_weight 200 0.80 8 1360 = 80 := by
  sorry

end NUMINAMATH_CALUDE_empty_tank_weight_is_80_l417_41782


namespace NUMINAMATH_CALUDE_final_shell_count_l417_41796

def calculate_final_shells (initial : ℕ) 
  (vacation1_day1to3 : ℕ) (vacation1_day4 : ℕ) (vacation1_lost : ℕ)
  (vacation2_day1to2 : ℕ) (vacation2_day3 : ℕ) (vacation2_given : ℕ)
  (vacation3_day1 : ℕ) (vacation3_day2 : ℕ) (vacation3_day3to4 : ℕ) (vacation3_misplaced : ℕ) : ℕ :=
  initial + 
  (vacation1_day1to3 * 3 + vacation1_day4 - vacation1_lost) +
  (vacation2_day1to2 * 2 + vacation2_day3 - vacation2_given) +
  (vacation3_day1 + vacation3_day2 + vacation3_day3to4 * 2 - vacation3_misplaced)

theorem final_shell_count :
  calculate_final_shells 20 5 6 4 4 7 3 8 4 3 5 = 62 := by
  sorry

end NUMINAMATH_CALUDE_final_shell_count_l417_41796


namespace NUMINAMATH_CALUDE_study_session_duration_in_minutes_l417_41748

-- Define the duration of the study session
def study_session_hours : ℕ := 8
def study_session_minutes : ℕ := 45

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem study_session_duration_in_minutes :
  study_session_hours * minutes_per_hour + study_session_minutes = 525 :=
by sorry

end NUMINAMATH_CALUDE_study_session_duration_in_minutes_l417_41748


namespace NUMINAMATH_CALUDE_tree_height_difference_l417_41785

-- Define the heights of the trees
def pine_height : ℚ := 49/4
def maple_height : ℚ := 75/4

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l417_41785


namespace NUMINAMATH_CALUDE_factor_expression_l417_41759

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l417_41759


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l417_41727

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l417_41727


namespace NUMINAMATH_CALUDE_overlapping_area_is_75_over_8_l417_41708

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- The area of the overlapping region formed by two 30-60-90 triangles -/
def overlapping_area (t1 t2 : Triangle30_60_90) : ℝ :=
  sorry

/-- The theorem stating the area of the overlapping region -/
theorem overlapping_area_is_75_over_8 (t1 t2 : Triangle30_60_90) 
  (h1 : t1.hypotenuse = 10)
  (h2 : t2.hypotenuse = 10)
  (h3 : overlapping_area t1 t2 ≠ 0) : 
  overlapping_area t1 t2 = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_75_over_8_l417_41708


namespace NUMINAMATH_CALUDE_specific_grid_toothpicks_l417_41791

/-- Represents a rectangular grid of toothpicks with reinforcements -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  horizontalReinforcementInterval : ℕ
  verticalReinforcementInterval : ℕ

/-- Calculates the total number of toothpicks in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalLines := grid.height + 1
  let verticalLines := grid.width + 1
  let baseHorizontal := horizontalLines * grid.width
  let baseVertical := verticalLines * grid.height
  let reinforcedHorizontal := (horizontalLines / grid.horizontalReinforcementInterval) * grid.width
  let reinforcedVertical := (verticalLines / grid.verticalReinforcementInterval) * grid.height
  baseHorizontal + baseVertical + reinforcedHorizontal + reinforcedVertical

/-- Theorem stating that the specific grid configuration results in 990 toothpicks -/
theorem specific_grid_toothpicks :
  totalToothpicks { height := 25, width := 15, horizontalReinforcementInterval := 5, verticalReinforcementInterval := 3 } = 990 := by
  sorry

end NUMINAMATH_CALUDE_specific_grid_toothpicks_l417_41791


namespace NUMINAMATH_CALUDE_translation_of_quadratic_l417_41749

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The translated quadratic function -/
def g (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- Theorem stating that g is the result of translating f one unit left and two units up -/
theorem translation_of_quadratic :
  ∀ x : ℝ, g x = f (x - 1) + 2 := by sorry

end NUMINAMATH_CALUDE_translation_of_quadratic_l417_41749


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l417_41728

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, f a y ≥ f a x) → a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l417_41728


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l417_41797

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for ellipses
structure Ellipse where
  a : ℝ
  b : ℝ

def is_on_ellipse (e : Ellipse) (p : Point2D) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

def has_common_focus (e1 e2 : Ellipse) : Prop :=
  ∃ (f : Point2D), (f.x^2 = e1.a^2 - e1.b^2) ∧ (f.x^2 = e2.a^2 - e2.b^2)

def has_foci_on_axes (e : Ellipse) : Prop :=
  ∃ (f : ℝ), (f^2 = e.a^2 - e.b^2) ∧ (f ≠ 0)

theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    has_common_focus e (Ellipse.mk 3 2) ∧
    is_on_ellipse e (Point2D.mk 3 (-2)) ∧
    has_foci_on_axes e ∧
    e.a^2 = 15 ∧ e.b^2 = 10 := by sorry

theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    has_foci_on_axes e ∧
    is_on_ellipse e (Point2D.mk (Real.sqrt 3) (-2)) ∧
    is_on_ellipse e (Point2D.mk (-2 * Real.sqrt 3) 1) ∧
    e.a^2 = 15 ∧ e.b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l417_41797


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l417_41742

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, y > 0 ∧ 
  ((-3 - 0)^2 + (0 - y)^2 = (-2 - 0)^2 + (5 - y)^2) ∧ 
  y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l417_41742


namespace NUMINAMATH_CALUDE_negation_of_existence_geq_l417_41716

theorem negation_of_existence_geq (p : Prop) :
  (¬ (∃ x : ℝ, x^2 ≥ x)) ↔ (∀ x : ℝ, x^2 < x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_geq_l417_41716


namespace NUMINAMATH_CALUDE_base_conversion_2023_l417_41713

/-- Converts a number from base 10 to base 8 --/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2023 :
  toBase8 2023 = 3747 := by sorry

end NUMINAMATH_CALUDE_base_conversion_2023_l417_41713


namespace NUMINAMATH_CALUDE_unique_cookie_distribution_l417_41758

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  ben : ℕ
  mia : ℕ
  leo : ℕ

/-- Checks if a cookie distribution satisfies the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.ben + d.mia + d.leo = 30 ∧
  d.mia = 2 * d.ben ∧
  d.leo = d.ben + d.mia

/-- The correct cookie distribution -/
def correctDistribution : CookieDistribution :=
  { ben := 5, mia := 10, leo := 15 }

/-- Theorem stating that the correct distribution is the only valid one -/
theorem unique_cookie_distribution :
  isValidDistribution correctDistribution ∧
  ∀ d : CookieDistribution, isValidDistribution d → d = correctDistribution :=
sorry

end NUMINAMATH_CALUDE_unique_cookie_distribution_l417_41758


namespace NUMINAMATH_CALUDE_pole_height_is_seven_meters_l417_41753

/-- Represents the geometry of a leaning telephone pole supported by a cable --/
structure LeaningPole where
  /-- Angle between the pole and the horizontal ground in degrees --/
  angle : ℝ
  /-- Distance from the pole base to the cable attachment point on the ground in meters --/
  cable_ground_distance : ℝ
  /-- Height of the person touching the cable in meters --/
  person_height : ℝ
  /-- Distance the person walks from the pole base towards the cable attachment point in meters --/
  person_distance : ℝ

/-- Calculates the height of the leaning pole given the geometry --/
def calculate_pole_height (pole : LeaningPole) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the pole height is 7 meters --/
theorem pole_height_is_seven_meters (pole : LeaningPole) 
  (h_angle : pole.angle = 85)
  (h_cable : pole.cable_ground_distance = 4)
  (h_person_height : pole.person_height = 1.75)
  (h_person_distance : pole.person_distance = 3)
  : calculate_pole_height pole = 7 := by
  sorry

end NUMINAMATH_CALUDE_pole_height_is_seven_meters_l417_41753


namespace NUMINAMATH_CALUDE_fraction_filled_equals_half_l417_41754

/-- Represents the fraction of a cistern that can be filled in 15 minutes -/
def fraction_filled_in_15_min : ℚ := 1 / 2

/-- The time it takes to fill half of the cistern -/
def time_to_fill_half : ℕ := 15

/-- Theorem stating that the fraction of the cistern filled in 15 minutes is 1/2 -/
theorem fraction_filled_equals_half : 
  fraction_filled_in_15_min = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_filled_equals_half_l417_41754


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l417_41761

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 36 ≤ 0) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l417_41761


namespace NUMINAMATH_CALUDE_election_result_l417_41740

theorem election_result (total_votes : ℕ) (winner_votes first_opponent_votes second_opponent_votes third_opponent_votes : ℕ)
  (h1 : total_votes = 963)
  (h2 : winner_votes = 195)
  (h3 : first_opponent_votes = 142)
  (h4 : second_opponent_votes = 116)
  (h5 : third_opponent_votes = 90)
  (h6 : total_votes = winner_votes + first_opponent_votes + second_opponent_votes + third_opponent_votes) :
  winner_votes - first_opponent_votes = 53 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l417_41740


namespace NUMINAMATH_CALUDE_negative_x_implies_a_greater_than_five_thirds_l417_41712

theorem negative_x_implies_a_greater_than_five_thirds
  (x a : ℝ) -- x and a are real numbers
  (h1 : x - 5 = -3 * a) -- given equation
  (h2 : x < 0) -- x is negative
  : a > 5/3 := by
sorry

end NUMINAMATH_CALUDE_negative_x_implies_a_greater_than_five_thirds_l417_41712


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_condition_l417_41711

/-- A geometric sequence with first term a and common ratio q -/
def GeometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Predicate for an increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f n ≤ f (n + 1)

theorem neither_necessary_nor_sufficient_condition
  (a q : ℝ) :
  ¬(((a * q > 0) ↔ IsIncreasing (GeometricSequence a q))) :=
sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_condition_l417_41711


namespace NUMINAMATH_CALUDE_distances_to_other_vertices_l417_41707

/-- A circle with radius 5 and an inscribed square -/
structure CircleSquare where
  center : ℝ × ℝ
  radius : ℝ
  square_vertices : Fin 4 → ℝ × ℝ

/-- A point on the circle -/
def PointOnCircle (cs : CircleSquare) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances to other vertices -/
theorem distances_to_other_vertices (cs : CircleSquare) 
  (h_radius : cs.radius = 5)
  (h_inscribed : ∀ v, distance cs.center (cs.square_vertices v) = cs.radius)
  (h_on_circle : distance cs.center (PointOnCircle cs) = cs.radius)
  (h_distance_to_one : ∃ v, distance (PointOnCircle cs) (cs.square_vertices v) = 6) :
  ∃ (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    distance (PointOnCircle cs) (cs.square_vertices v1) = Real.sqrt 2 ∧
    distance (PointOnCircle cs) (cs.square_vertices v2) = 8 ∧
    distance (PointOnCircle cs) (cs.square_vertices v3) = 7 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_distances_to_other_vertices_l417_41707


namespace NUMINAMATH_CALUDE_tethered_dog_area_l417_41793

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  let outside_area := (rope_length^2 * (5/6) + 2 * (rope_length - side_length)^2 * (1/6)) * π
  outside_area = (49/6) * π :=
by sorry

end NUMINAMATH_CALUDE_tethered_dog_area_l417_41793


namespace NUMINAMATH_CALUDE_committee_selection_l417_41762

theorem committee_selection (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l417_41762


namespace NUMINAMATH_CALUDE_total_combinations_l417_41773

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- Theorem: The total number of combinations of color and painting method is 20 -/
theorem total_combinations : num_colors * num_methods = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l417_41773


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l417_41783

theorem systematic_sampling_removal (total : Nat) (sample_size : Nat) (h : total = 162 ∧ sample_size = 16) :
  total % sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l417_41783


namespace NUMINAMATH_CALUDE_max_value_M_l417_41757

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (max : ℝ), max = (3 : ℝ) / 2 ∧ 
  ∀ (a b c d : ℝ), a + b + c + d = 1 → 
  a * d + 2 * b * d + 3 * a * b + 3 * c * d + 4 * a * c + 5 * b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_M_l417_41757


namespace NUMINAMATH_CALUDE_value_of_a_l417_41747

theorem value_of_a (a b : ℚ) (h1 : b / a = 3) (h2 : b = 12 - 5 * a) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l417_41747


namespace NUMINAMATH_CALUDE_quadratic_polynomial_equality_l417_41781

theorem quadratic_polynomial_equality 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) 
  (h_equality : ∀ x, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) :
  ∃ (a b c : ℝ), (a = 1 ∧ b = 5 ∧ c = 1) ∧ (∀ x, f x = x^2 + 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_equality_l417_41781


namespace NUMINAMATH_CALUDE_cloud_ratio_l417_41731

theorem cloud_ratio : 
  let carson_clouds : ℕ := 6
  let total_clouds : ℕ := 24
  let brother_clouds : ℕ := total_clouds - carson_clouds
  (brother_clouds : ℚ) / carson_clouds = 3 := by
  sorry

end NUMINAMATH_CALUDE_cloud_ratio_l417_41731


namespace NUMINAMATH_CALUDE_johns_mean_score_l417_41704

def johns_scores : List ℝ := [89, 92, 95, 88, 90]

theorem johns_mean_score :
  (johns_scores.sum / johns_scores.length : ℝ) = 90.8 := by
  sorry

end NUMINAMATH_CALUDE_johns_mean_score_l417_41704


namespace NUMINAMATH_CALUDE_onion_basket_change_l417_41789

theorem onion_basket_change (x : ℤ) : x + 4 - 5 + 9 = x + 8 := by
  sorry

end NUMINAMATH_CALUDE_onion_basket_change_l417_41789


namespace NUMINAMATH_CALUDE_line_vertical_shift_specific_line_shift_l417_41774

/-- Given a line y = mx + b, moving it down by k units results in y = mx + (b - k) -/
theorem line_vertical_shift (m b k : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let shifted_line := fun (x : ℝ) => m * x + (b - k)
  (∀ x, shifted_line x = original_line x - k) :=
by sorry

/-- Moving the line y = 3x down 2 units results in y = 3x - 2 -/
theorem specific_line_shift :
  let original_line := fun (x : ℝ) => 3 * x
  let shifted_line := fun (x : ℝ) => 3 * x - 2
  (∀ x, shifted_line x = original_line x - 2) :=
by sorry

end NUMINAMATH_CALUDE_line_vertical_shift_specific_line_shift_l417_41774


namespace NUMINAMATH_CALUDE_jakes_weight_l417_41794

theorem jakes_weight (jake sister brother : ℝ) : 
  (0.8 * jake = 2 * sister) →
  (jake + sister = 168) →
  (brother = 1.25 * (jake + sister)) →
  (jake + sister + brother = 221) →
  jake = 120 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l417_41794


namespace NUMINAMATH_CALUDE_B_equals_A_l417_41729

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x ∈ A}

theorem B_equals_A : B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_A_l417_41729


namespace NUMINAMATH_CALUDE_garden_separation_possible_l417_41745

/-- Represents the content of a garden plot -/
inductive PlotContent
  | Empty
  | Cabbage
  | Goat

/-- Represents a position in the garden -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a fence in the garden -/
inductive Fence
  | Vertical (x : Fin 3) -- A vertical fence after column x
  | Horizontal (y : Fin 3) -- A horizontal fence after row y

/-- Represents the garden layout -/
def Garden := Position → PlotContent

/-- Checks if a fence separates two positions -/
def separates (f : Fence) (p1 p2 : Position) : Prop :=
  match f with
  | Fence.Vertical x => p1.x ≤ x ∧ x < p2.x
  | Fence.Horizontal y => p1.y ≤ y ∧ y < p2.y

/-- The theorem to be proved -/
theorem garden_separation_possible (g : Garden) :
  ∃ (f1 f2 f3 : Fence),
    (∀ p1 p2 : Position,
      g p1 = PlotContent.Goat →
      g p2 = PlotContent.Cabbage →
      separates f1 p1 p2 ∨ separates f2 p1 p2 ∨ separates f3 p1 p2) ∧
    (∀ f : Fence, f ∈ [f1, f2, f3] →
      ∀ p : Position,
        g p ≠ PlotContent.Empty →
        ¬(∃ p' : Position, g p' ≠ PlotContent.Empty ∧ separates f p p')) :=
by sorry

end NUMINAMATH_CALUDE_garden_separation_possible_l417_41745


namespace NUMINAMATH_CALUDE_complex_square_equality_l417_41755

theorem complex_square_equality (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (a : ℂ) + (b : ℂ) * Complex.I = 4 + 3 * Complex.I ↔ 
  ((a : ℂ) + (b : ℂ) * Complex.I) ^ 2 = 7 + 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l417_41755


namespace NUMINAMATH_CALUDE_smallest_valid_number_l417_41726

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧
  (n / 10) % 20 = 0 ∧
  (n % 1000) % 21 = 0 ∧
  (n / 100 % 10) ≠ 0

theorem smallest_valid_number :
  is_valid 1609 ∧ ∀ m < 1609, ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l417_41726


namespace NUMINAMATH_CALUDE_third_number_proof_l417_41723

/-- The largest five-digit number with all even digits -/
def largest_even_five_digit : ℕ := 88888

/-- The smallest four-digit number with all odd digits -/
def smallest_odd_four_digit : ℕ := 1111

/-- The sum of the three numbers -/
def total_sum : ℕ := 121526

/-- The third number -/
def third_number : ℕ := total_sum - largest_even_five_digit - smallest_odd_four_digit

theorem third_number_proof :
  third_number = 31527 :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l417_41723


namespace NUMINAMATH_CALUDE_marbles_given_to_juan_l417_41730

theorem marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 73)
  (h2 : remaining_marbles = 3) :
  initial_marbles - remaining_marbles = 70 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_juan_l417_41730


namespace NUMINAMATH_CALUDE_mean_median_difference_l417_41732

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_60_percent : ℚ
  score_75_percent : ℚ
  score_85_percent : ℚ
  score_90_percent : ℚ
  score_100_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (dist : ScoreDistribution) : ℚ :=
  (60 * dist.score_60_percent + 75 * dist.score_75_percent + 
   85 * dist.score_85_percent + 90 * dist.score_90_percent + 
   100 * dist.score_100_percent) / 1

/-- Calculates the median score given a score distribution -/
def median_score (dist : ScoreDistribution) : ℚ := 85

/-- Theorem stating the difference between mean and median scores -/
theorem mean_median_difference (dist : ScoreDistribution) : 
  dist.total_students = 25 ∧
  dist.score_60_percent = 15/100 ∧
  dist.score_75_percent = 20/100 ∧
  dist.score_85_percent = 30/100 ∧
  dist.score_90_percent = 20/100 ∧
  dist.score_100_percent = 15/100 →
  mean_score dist - median_score dist = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l417_41732


namespace NUMINAMATH_CALUDE_doll_collection_increase_l417_41779

theorem doll_collection_increase (initial_count : ℕ) : 
  (initial_count : ℚ) * (1 + 1/4) = initial_count + 2 → 
  initial_count + 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l417_41779


namespace NUMINAMATH_CALUDE_congruence_iff_divisible_l417_41772

theorem congruence_iff_divisible (a b m : ℤ) : a ≡ b [ZMOD m] ↔ m ∣ (a - b) := by sorry

end NUMINAMATH_CALUDE_congruence_iff_divisible_l417_41772


namespace NUMINAMATH_CALUDE_cookie_problem_l417_41767

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l417_41767


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l417_41790

theorem tangent_point_x_coordinate
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 1)
  (h2 : ∃ x, HasDerivAt f 4 x) :
  ∃ x, HasDerivAt f 4 x ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l417_41790


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_55_l417_41722

/-- Converts Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℚ) : ℚ := (c * 9 / 5) + 32

/-- Water boiling point in Fahrenheit -/
def water_boiling_f : ℚ := 212

/-- Water boiling point in Celsius -/
def water_boiling_c : ℚ := 100

/-- Ice melting point in Fahrenheit -/
def ice_melting_f : ℚ := 32

/-- Ice melting point in Celsius -/
def ice_melting_c : ℚ := 0

/-- The temperature of the pot of water in Celsius -/
def pot_temp_c : ℚ := 55

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temp_f : ℚ := 131

theorem celsius_to_fahrenheit_55 :
  celsius_to_fahrenheit pot_temp_c = pot_temp_f := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_55_l417_41722


namespace NUMINAMATH_CALUDE_sin_alpha_plus_seven_pi_sixth_l417_41770

theorem sin_alpha_plus_seven_pi_sixth (α : ℝ) 
  (h : Real.sin α + Real.cos (α - π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 6) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_seven_pi_sixth_l417_41770


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l417_41737

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Define the property of monotonic interval starting at 1
def monotonic_from_one (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f a x < f a y

-- Theorem statement
theorem quadratic_monotonic_condition (a : ℝ) :
  monotonic_from_one a → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l417_41737


namespace NUMINAMATH_CALUDE_parallelogram_height_l417_41703

/-- Proves that the height of a parallelogram is 18 cm given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h1 : area = 648) (h2 : base = 36) :
  area / base = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l417_41703


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l417_41714

/-- If the terminal side of angle α passes through point (-1, 2), then sin α = 2√5/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l417_41714


namespace NUMINAMATH_CALUDE_line_parameterization_l417_41744

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = 3 * x₀ + 5 ∧ ∃ (k : ℝ), dx = k * 1 ∧ dy = k * 3

theorem line_parameterization 
  (x₀ y₀ dx dy t : ℝ) :
  is_valid_parameterization x₀ y₀ dx dy ↔ 
  ∀ t, (3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l417_41744


namespace NUMINAMATH_CALUDE_abdul_binh_age_difference_l417_41710

/- Define Susie's age -/
variable (S : ℤ)

/- Define Abdul's age in terms of Susie's -/
def A : ℤ := S + 9

/- Define Binh's age in terms of Susie's -/
def B : ℤ := S + 2

/- Theorem statement -/
theorem abdul_binh_age_difference : A - B = 7 := by
  sorry

end NUMINAMATH_CALUDE_abdul_binh_age_difference_l417_41710


namespace NUMINAMATH_CALUDE_probability_Sa_before_Sb_l417_41706

/-- Represents a three-letter string -/
structure ThreeLetterString :=
  (letters : Fin 3 → Char)

/-- The probability of a letter being received correctly -/
def correct_probability : ℚ := 2/3

/-- The probability of a letter being received incorrectly -/
def incorrect_probability : ℚ := 1/3

/-- The transmitted string aaa -/
def aaa : ThreeLetterString :=
  { letters := λ _ => 'a' }

/-- The transmitted string bbb -/
def bbb : ThreeLetterString :=
  { letters := λ _ => 'b' }

/-- The received string when aaa is transmitted -/
def Sa : ThreeLetterString :=
  sorry

/-- The received string when bbb is transmitted -/
def Sb : ThreeLetterString :=
  sorry

/-- The probability that Sa comes before Sb in alphabetical order -/
def p : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_Sa_before_Sb : p = 532/729 :=
  sorry

end NUMINAMATH_CALUDE_probability_Sa_before_Sb_l417_41706


namespace NUMINAMATH_CALUDE_limit_sequence_is_zero_l417_41786

/-- The limit of the sequence (n - (n^5 - 5)^(1/3)) * n * sqrt(n) as n approaches infinity is 0. -/
theorem limit_sequence_is_zero :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n : ℝ) - ((n : ℝ)^5 - 5)^(1/3)) * n * (n : ℝ).sqrt| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sequence_is_zero_l417_41786


namespace NUMINAMATH_CALUDE_four_people_five_chairs_middle_empty_l417_41743

/-- The number of ways to arrange people in chairs. -/
def seating_arrangements (total_chairs : ℕ) (people : ℕ) (empty_chair : ℕ) : ℕ :=
  (total_chairs - 1).factorial / ((total_chairs - 1 - people).factorial)

/-- Theorem: There are 24 ways to arrange 4 people in 5 chairs with the middle chair empty. -/
theorem four_people_five_chairs_middle_empty :
  seating_arrangements 5 4 3 = 24 := by sorry

end NUMINAMATH_CALUDE_four_people_five_chairs_middle_empty_l417_41743


namespace NUMINAMATH_CALUDE_eve_can_discover_secret_number_l417_41734

theorem eve_can_discover_secret_number :
  ∀ x : ℕ, ∃ (k : ℕ) (n : Fin k → ℕ),
    ∀ y : ℕ, (∀ i : Fin k, Prime (x + n i) ↔ Prime (y + n i)) → x = y :=
sorry

end NUMINAMATH_CALUDE_eve_can_discover_secret_number_l417_41734


namespace NUMINAMATH_CALUDE_standing_arrangements_eq_210_l417_41724

/-- The number of ways to arrange n distinct objects in k positions --/
def arrangement (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k objects from n distinct objects --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways 3 people can stand on 6 steps with given conditions --/
def standing_arrangements : ℕ :=
  arrangement 6 3 + choose 3 1 * arrangement 6 2

theorem standing_arrangements_eq_210 : standing_arrangements = 210 := by sorry

end NUMINAMATH_CALUDE_standing_arrangements_eq_210_l417_41724


namespace NUMINAMATH_CALUDE_series_sum_l417_41751

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) from k = 0 to infinity is 1/2 -/
theorem series_sum : 
  ∑' k, (3 ^ (2 ^ k) : ℝ) / ((9 : ℝ) ^ (2 ^ k) - 1) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l417_41751


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l417_41788

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 7 * y = 5) ∧ (x = 62 / 3) ∧ (y = 17) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l417_41788


namespace NUMINAMATH_CALUDE_perfect_square_condition_l417_41715

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (c : ℝ), 4 * a^2 + k * a * b + 9 * b^2 = c^2) →
  k = 12 ∨ k = -12 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l417_41715


namespace NUMINAMATH_CALUDE_tangent_line_determines_n_l417_41738

/-- A curve defined by a cubic function -/
structure CubicCurve where
  m : ℝ
  n : ℝ

/-- A line defined by a linear function -/
structure Line where
  k : ℝ

/-- Checks if a line is tangent to a cubic curve at a given point -/
def is_tangent_at (c : CubicCurve) (l : Line) (x₀ y₀ : ℝ) : Prop :=
  y₀ = x₀^3 + c.m * x₀ + c.n ∧
  y₀ = l.k * x₀ + 2 ∧
  3 * x₀^2 + c.m = l.k

theorem tangent_line_determines_n (c : CubicCurve) (l : Line) :
  is_tangent_at c l 1 4 → c.n = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_n_l417_41738


namespace NUMINAMATH_CALUDE_job_interviews_comprehensive_l417_41778

/-- Represents a scenario that may or may not require comprehensive investigation. -/
inductive Scenario
| AirQuality
| VisionStatus
| JobInterviews
| FishCount

/-- Determines if a scenario requires comprehensive investigation. -/
def requiresComprehensiveInvestigation (s : Scenario) : Prop :=
  match s with
  | Scenario.JobInterviews => True
  | _ => False

/-- Theorem stating that job interviews is the only scenario requiring comprehensive investigation. -/
theorem job_interviews_comprehensive :
  ∀ s : Scenario, requiresComprehensiveInvestigation s ↔ s = Scenario.JobInterviews :=
by sorry

end NUMINAMATH_CALUDE_job_interviews_comprehensive_l417_41778


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l417_41765

theorem cubic_root_sum_cubes (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (2 - 3*ω + 4*ω^2)^3 + (3 + 2*ω - ω^2)^3 = 1191 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l417_41765


namespace NUMINAMATH_CALUDE_distinct_elements_condition_l417_41799

theorem distinct_elements_condition (x : ℝ) : 
  ({1, x, x^2 - x} : Set ℝ).ncard = 3 ↔ 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_elements_condition_l417_41799


namespace NUMINAMATH_CALUDE_loop_iterations_count_l417_41764

theorem loop_iterations_count (i : ℕ) : 
  i = 20 → (∀ n : ℕ, n < 20 → i - n > 0) ∧ (i - 20 = 0) := by sorry

end NUMINAMATH_CALUDE_loop_iterations_count_l417_41764


namespace NUMINAMATH_CALUDE_alice_painted_six_cuboids_l417_41718

/-- The number of outer faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end NUMINAMATH_CALUDE_alice_painted_six_cuboids_l417_41718


namespace NUMINAMATH_CALUDE_bowling_team_score_l417_41784

theorem bowling_team_score (total_score : ℕ) (bowler1 bowler2 bowler3 : ℕ) : 
  total_score = 810 →
  bowler1 = bowler2 / 3 →
  bowler2 = 3 * bowler3 →
  bowler1 + bowler2 + bowler3 = total_score →
  bowler3 = 162 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_score_l417_41784


namespace NUMINAMATH_CALUDE_triangle_inequality_with_heights_l417_41795

theorem triangle_inequality_with_heights 
  (a b c h_a h_b h_c t : ℝ) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) 
  (heights_def : h_a * a = h_b * b ∧ h_b * b = h_c * c) 
  (t_bound : t ≥ (1 : ℝ) / 2) : 
  (t * a + h_a) + (t * b + h_b) > t * c + h_c ∧ 
  (t * b + h_b) + (t * c + h_c) > t * a + h_a ∧ 
  (t * c + h_c) + (t * a + h_a) > t * b + h_b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_heights_l417_41795


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l417_41756

theorem negation_of_conjunction (x y : ℝ) : 
  ¬(x = 2 ∧ y = 3) ↔ (x ≠ 2 ∨ y ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l417_41756


namespace NUMINAMATH_CALUDE_max_coins_distribution_l417_41720

theorem max_coins_distribution (k : ℕ) : 
  (∀ n : ℕ, n < 100 ∧ ∃ k : ℕ, n = 13 * k + 3) → 
  (∀ m : ℕ, m < 100 ∧ ∃ k : ℕ, m = 13 * k + 3 → m ≤ 91) ∧
  (∃ k : ℕ, 91 = 13 * k + 3) ∧ 
  91 < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l417_41720


namespace NUMINAMATH_CALUDE_average_age_of_new_joiners_l417_41780

/-- Given a group of people going for a picnic, prove the average age of new joiners -/
theorem average_age_of_new_joiners
  (initial_count : ℕ)
  (initial_avg_age : ℝ)
  (new_count : ℕ)
  (new_total_avg_age : ℝ)
  (h1 : initial_count = 12)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 12)
  (h4 : new_total_avg_age = 15.5) :
  let total_count := initial_count + new_count
  let new_joiners_avg_age := (total_count * new_total_avg_age - initial_count * initial_avg_age) / new_count
  new_joiners_avg_age = 15 := by
sorry

end NUMINAMATH_CALUDE_average_age_of_new_joiners_l417_41780


namespace NUMINAMATH_CALUDE_solution_set_for_decreasing_function_l417_41725

/-- A function f is decreasing on its domain -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of x satisfying f(1/x) > f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (1/x) > f 1}

theorem solution_set_for_decreasing_function (f : ℝ → ℝ) (h : IsDecreasing f) :
    SolutionSet f = {x | x < 0 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_for_decreasing_function_l417_41725


namespace NUMINAMATH_CALUDE_jerry_one_way_time_15_minutes_l417_41741

-- Define the distance to school in miles
def distance_to_school : ℝ := 4

-- Define Carson's speed in miles per hour
def carson_speed : ℝ := 8

-- Define the relationship between Jerry's round trip and Carson's one-way trip
axiom jerry_carson_time_relation : 
  ∀ (jerry_round_trip_time carson_one_way_time : ℝ), 
    jerry_round_trip_time = carson_one_way_time

-- Theorem: Jerry's one-way trip time to school is 15 minutes
theorem jerry_one_way_time_15_minutes : 
  ∃ (jerry_one_way_time : ℝ), 
    jerry_one_way_time = 15 := by sorry

end NUMINAMATH_CALUDE_jerry_one_way_time_15_minutes_l417_41741


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_divisible_l417_41792

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

/-- The product of all elements in a list -/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem arithmetic_progression_product_divisible (a : ℕ) :
  (listProduct (arithmeticProgression a 11 10)) % (Nat.factorial 10) = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_divisible_l417_41792


namespace NUMINAMATH_CALUDE_cubic_inequality_l417_41798

theorem cubic_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : a^3 < a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l417_41798


namespace NUMINAMATH_CALUDE_bob_gardening_project_cost_l417_41771

/-- The total cost of Bob's gardening project --/
def gardening_project_cost 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) 
  (gardener_hours_per_day : ℕ) 
  (gardener_work_days : ℕ) 
  (soil_volume : ℕ) 
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush + 
  gardener_hourly_rate * gardener_hours_per_day * gardener_work_days + 
  soil_volume * soil_cost_per_unit

/-- Theorem stating that the total cost of Bob's gardening project is $4100 --/
theorem bob_gardening_project_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_bob_gardening_project_cost_l417_41771


namespace NUMINAMATH_CALUDE_complex_subtraction_l417_41776

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l417_41776


namespace NUMINAMATH_CALUDE_fraction_equality_l417_41750

theorem fraction_equality (a b : ℝ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l417_41750


namespace NUMINAMATH_CALUDE_order_of_t_squared_t_neg_t_l417_41702

theorem order_of_t_squared_t_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t := by
  sorry

end NUMINAMATH_CALUDE_order_of_t_squared_t_neg_t_l417_41702


namespace NUMINAMATH_CALUDE_geometric_progression_perfect_square_sum_l417_41752

/-- A geometric progression starting with 1 -/
def GeometricProgression (r : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => r^i)

/-- The sum of a list of natural numbers -/
def ListSum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem geometric_progression_perfect_square_sum :
  ∃ r₁ r₂ n₁ n₂ : ℕ,
    r₁ ≠ r₂ ∧
    n₁ ≥ 3 ∧
    n₂ ≥ 3 ∧
    IsPerfectSquare (ListSum (GeometricProgression r₁ n₁)) ∧
    IsPerfectSquare (ListSum (GeometricProgression r₂ n₂)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_perfect_square_sum_l417_41752


namespace NUMINAMATH_CALUDE_min_value_line_circle_l417_41733

/-- Given a line ax + by + c - 1 = 0 that passes through the center of the circle x^2 + y^2 - 2y - 5 = 0,
    prove that the minimum value of 4/b + 1/c is 9, where b > 0 and c > 0. -/
theorem min_value_line_circle (a b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, a * x + b * y + c - 1 = 0 → x^2 + y^2 - 2*y - 5 = 0) →
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 4 / b' + 1 / c' ≥ 9) ∧
  (∃ b' c' : ℝ, b' > 0 ∧ c' > 0 ∧ 4 / b' + 1 / c' = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_line_circle_l417_41733


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l417_41719

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n+1 vertices -/
theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices.choose 2 - vertices
  let intersecting_pairs := vertices.choose 4
  let probability := intersecting_pairs / total_diagonals.choose 2
  probability = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l417_41719


namespace NUMINAMATH_CALUDE_vasya_upward_run_time_l417_41775

/-- Represents the speed and time properties of Vasya's escalator run -/
structure EscalatorRun where
  -- Vasya's speed going down (in units per minute)
  speed_down : ℝ
  -- Vasya's speed going up (in units per minute)
  speed_up : ℝ
  -- Escalator's speed (in units per minute)
  escalator_speed : ℝ
  -- Time for stationary run (in minutes)
  time_stationary : ℝ
  -- Time for downward moving escalator run (in minutes)
  time_down : ℝ
  -- Constraint: Vasya runs down twice as fast as he runs up
  speed_constraint : speed_down = 2 * speed_up
  -- Constraint: Stationary run takes 6 minutes
  stationary_constraint : time_stationary = 6
  -- Constraint: Downward moving escalator run takes 13.5 minutes
  down_constraint : time_down = 13.5

/-- Theorem stating the time for Vasya's upward moving escalator run -/
theorem vasya_upward_run_time (run : EscalatorRun) :
  let time_up := (1 / (run.speed_down - run.escalator_speed) + 1 / (run.speed_up + run.escalator_speed)) * 60
  time_up = 324 := by
  sorry

end NUMINAMATH_CALUDE_vasya_upward_run_time_l417_41775


namespace NUMINAMATH_CALUDE_expression_evaluation_l417_41717

theorem expression_evaluation : 
  let x := Real.sqrt (6000 - (3^3 : ℝ))
  let y := (105 / 21 : ℝ)^2
  abs (x * y - 1932.25) < 0.01 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l417_41717


namespace NUMINAMATH_CALUDE_washington_dc_trip_cost_l417_41760

/-- Calculates the total cost per person for a group trip to Washington D.C. -/
theorem washington_dc_trip_cost 
  (num_friends : ℕ)
  (airfare_hotel_cost : ℚ)
  (food_expenses : ℚ)
  (transportation_expenses : ℚ)
  (smithsonian_tour_cost : ℚ)
  (zoo_entry_fee : ℚ)
  (zoo_spending_allowance : ℚ)
  (river_cruise_cost : ℚ)
  (h1 : num_friends = 15)
  (h2 : airfare_hotel_cost = 13500)
  (h3 : food_expenses = 4500)
  (h4 : transportation_expenses = 3000)
  (h5 : smithsonian_tour_cost = 50)
  (h6 : zoo_entry_fee = 75)
  (h7 : zoo_spending_allowance = 15)
  (h8 : river_cruise_cost = 100) :
  (airfare_hotel_cost + food_expenses + transportation_expenses + 
   num_friends * (smithsonian_tour_cost + zoo_entry_fee + zoo_spending_allowance + river_cruise_cost)) / num_friends = 1640 := by
sorry

end NUMINAMATH_CALUDE_washington_dc_trip_cost_l417_41760


namespace NUMINAMATH_CALUDE_salt_water_ratio_l417_41777

theorem salt_water_ratio (salt : ℕ) (water : ℕ) :
  salt = 1 ∧ water = 10 →
  (salt : ℚ) / (salt + water : ℚ) = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_salt_water_ratio_l417_41777


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l417_41763

theorem polynomial_factor_theorem (c q k : ℝ) : 
  (∀ x, 3 * x^3 + c * x + 8 = (x^2 + q * x + 2) * (3 * x + k)) →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l417_41763


namespace NUMINAMATH_CALUDE_particular_solution_l417_41766

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 + 2 * Real.log ((1 + Real.exp x) / 2))

theorem particular_solution (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x ∧ y 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_particular_solution_l417_41766


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l417_41721

theorem correct_parentheses_removal (x : ℝ) : -0.5 * (1 - 2 * x) = -0.5 + x := by
  sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l417_41721


namespace NUMINAMATH_CALUDE_elder_person_age_l417_41709

theorem elder_person_age (y e : ℕ) : 
  e = y + 16 →                     -- The ages differ by 16 years
  e - 6 = 3 * (y - 6) →            -- 6 years ago, elder was 3 times younger's age
  e = 30                           -- Elder's present age is 30
  := by sorry

end NUMINAMATH_CALUDE_elder_person_age_l417_41709


namespace NUMINAMATH_CALUDE_store_loss_percentage_l417_41736

/-- Calculate the store's loss percentage on a radio sale -/
theorem store_loss_percentage
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (actual_selling_price : ℝ)
  (h1 : cost_price = 25000)
  (h2 : discount_rate = 0.15)
  (h3 : tax_rate = 0.05)
  (h4 : actual_selling_price = 22000) :
  let discounted_price := cost_price * (1 - discount_rate)
  let final_selling_price := discounted_price * (1 + tax_rate)
  let loss := final_selling_price - actual_selling_price
  let loss_percentage := (loss / cost_price) * 100
  loss_percentage = 1.25 := by
sorry


end NUMINAMATH_CALUDE_store_loss_percentage_l417_41736


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficients_sum_l417_41768

/-- Given a quadratic inequality x² - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that a + b = 5 -/
theorem quadratic_inequality_coefficients_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficients_sum_l417_41768


namespace NUMINAMATH_CALUDE_box_volume_increase_l417_41746

/-- Given a rectangular box with length l, width w, and height h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a volume of 7208 cubic inches. -/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * (l * w + w * h + h * l) = 1560)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7208 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l417_41746


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l417_41739

/-- Proves that the actual distance traveled is 60 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_difference : ℝ) 
  (h1 : speed_slow = 15)
  (h2 : speed_fast = 30)
  (h3 : distance_difference = 60)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ), 
    actual_distance / speed_slow = (actual_distance + distance_difference) / speed_fast ∧ 
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l417_41739


namespace NUMINAMATH_CALUDE_product_equals_sum_solution_l417_41735

theorem product_equals_sum_solution :
  ∀ (a b c d e f : ℕ),
    a * b * c * d * e * f = a + b + c + d + e + f →
    ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_product_equals_sum_solution_l417_41735


namespace NUMINAMATH_CALUDE_marble_probability_l417_41700

theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ) :
  total_marbles = 90 →
  p_white = 1/3 →
  p_green = 1/5 →
  (1 : ℚ) - (p_white + p_green) = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l417_41700
