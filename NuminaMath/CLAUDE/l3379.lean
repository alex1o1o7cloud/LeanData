import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_roots_l3379_337967

theorem right_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) →
  (z₂^2 + a*z₂ + b = 0) →
  (z₂ = Complex.I * z₁) →
  a^2 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l3379_337967


namespace NUMINAMATH_CALUDE_smallest_ending_number_l3379_337955

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a number has an even number of factors -/
def has_even_factors (n : ℕ+) : Prop :=
  Even (num_factors n)

/-- A function that counts the number of even integers with an even number of factors
    in the range from 1 to n (inclusive) -/
def count_even_with_even_factors (n : ℕ) : ℕ := sorry

theorem smallest_ending_number :
  ∀ k : ℕ, k < 14 → count_even_with_even_factors k < 5 ∧
  count_even_with_even_factors 14 ≥ 5 := by sorry

end NUMINAMATH_CALUDE_smallest_ending_number_l3379_337955


namespace NUMINAMATH_CALUDE_carrot_cost_correct_l3379_337951

/-- Represents the cost of carrots for all students -/
def carrot_cost : ℚ := 185

/-- Represents the number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_students_per_class : ℕ := 30

/-- Represents the number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_students_per_class : ℕ := 28

/-- Represents the number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_students_per_class : ℕ := 27

/-- Represents the cost of a hamburger -/
def hamburger_cost : ℚ := 21/10

/-- Represents the cost of a cookie -/
def cookie_cost : ℚ := 1/5

/-- Represents the total cost of lunch for all students -/
def total_lunch_cost : ℚ := 1036

/-- Theorem stating that the cost of carrots is correct given the conditions -/
theorem carrot_cost_correct : 
  let total_students := third_grade_classes * third_grade_students_per_class + 
                        fourth_grade_classes * fourth_grade_students_per_class + 
                        fifth_grade_classes * fifth_grade_students_per_class
  total_lunch_cost = total_students * (hamburger_cost + cookie_cost) + carrot_cost :=
by sorry

end NUMINAMATH_CALUDE_carrot_cost_correct_l3379_337951


namespace NUMINAMATH_CALUDE_triangle_AXY_is_obtuse_l3379_337929

-- Define the triangular pyramid ABCD
structure TriangularPyramid where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the inscribed and exscribed spheres
structure InscribedSphere where
  center : Point
  radius : ℝ

structure ExscribedSphere where
  center : Point
  radius : ℝ

-- Define the points where the spheres touch face BCD
def X (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) : Point := sorry
def Y (pyramid : TriangularPyramid) (exscribedSphere : ExscribedSphere) : Point := sorry

-- Define the angle AXY
def angle_AXY (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) : ℝ := sorry

-- Theorem statement
theorem triangle_AXY_is_obtuse (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) :
  angle_AXY pyramid inscribedSphere exscribedSphere > π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_AXY_is_obtuse_l3379_337929


namespace NUMINAMATH_CALUDE_bridget_apples_proof_l3379_337985

/-- The number of apples Bridget bought -/
def total_apples : ℕ := 21

/-- The number of apples Bridget gave to Cassie and Dan -/
def apples_to_cassie_and_dan : ℕ := 7

/-- The number of apples Bridget kept for herself -/
def apples_kept : ℕ := 7

theorem bridget_apples_proof :
  total_apples = 21 ∧
  (2 * total_apples) / 3 = apples_to_cassie_and_dan + apples_kept :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_proof_l3379_337985


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l3379_337960

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 780 →
  total_participants = male_participants + female_participants →
  male_democrats = male_participants / 4 →
  female_democrats = 130 →
  male_democrats + female_democrats = total_participants / 3 →
  (female_democrats : ℚ) / female_participants = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l3379_337960


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3379_337944

theorem quadratic_function_property 
  (a c y₁ y₂ y₃ y₄ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-2)^2 - 4 * a * (-2) + c)
  (h_y₂ : y₂ = c)
  (h_y₃ : y₃ = a * 3^2 - 4 * a * 3 + c)
  (h_y₄ : y₄ = a * 5^2 - 4 * a * 5 + c)
  (h_y₂y₄ : y₂ * y₄ < 0) :
  y₁ * y₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3379_337944


namespace NUMINAMATH_CALUDE_sum_of_angles_three_triangles_l3379_337908

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that the sum of angles in a triangle is 180°
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define three non-overlapping triangles
variable (A B C : Triangle)

-- Assume each triangle is valid
variable (hA : is_valid_triangle A)
variable (hB : is_valid_triangle B)
variable (hC : is_valid_triangle C)

-- Theorem: The sum of all angles in the three triangles is 540°
theorem sum_of_angles_three_triangles :
  A.angle1 + A.angle2 + A.angle3 +
  B.angle1 + B.angle2 + B.angle3 +
  C.angle1 + C.angle2 + C.angle3 = 540 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_three_triangles_l3379_337908


namespace NUMINAMATH_CALUDE_words_with_e_count_l3379_337978

/-- The number of letters in the alphabet we're using -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 3

/-- The number of letters in the alphabet excluding E -/
def m : ℕ := 4

/-- The number of 3-letter words that can be made from the letters A, B, C, D, and E, 
    allowing repetition and using the letter E at least once -/
def num_words_with_e : ℕ := n^k - m^k

theorem words_with_e_count : num_words_with_e = 61 := by
  sorry

end NUMINAMATH_CALUDE_words_with_e_count_l3379_337978


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l3379_337986

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-2))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l3379_337986


namespace NUMINAMATH_CALUDE_statements_b_and_c_are_correct_l3379_337949

theorem statements_b_and_c_are_correct (a b c d : ℝ) :
  (ab > 0 ∧ b*c - a*d > 0 → c/a - d/b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) := by
sorry

end NUMINAMATH_CALUDE_statements_b_and_c_are_correct_l3379_337949


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3379_337930

def A : Set ℤ := {-1, 1}
def B : Set ℤ := {-3, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3379_337930


namespace NUMINAMATH_CALUDE_input_is_input_statement_l3379_337941

-- Define an enumeration for different types of statements
inductive StatementType
  | Print
  | Input
  | If
  | End

-- Define a function to classify statements
def classifyStatement (s : StatementType) : String :=
  match s with
  | StatementType.Print => "output"
  | StatementType.Input => "input"
  | StatementType.If => "conditional"
  | StatementType.End => "end"

-- Theorem to prove
theorem input_is_input_statement :
  classifyStatement StatementType.Input = "input" := by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l3379_337941


namespace NUMINAMATH_CALUDE_fraction_equivalent_with_difference_l3379_337971

theorem fraction_equivalent_with_difference : ∃ (a b : ℕ), 
  a > 0 ∧ b > 0 ∧ (a : ℚ) / b = 7 / 13 ∧ b - a = 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalent_with_difference_l3379_337971


namespace NUMINAMATH_CALUDE_area_fraction_on_7x7_grid_l3379_337997

/-- Represents a square grid of points -/
structure PointGrid :=
  (size : ℕ)

/-- Represents a square on the grid -/
structure GridSquare :=
  (sideLength : ℕ)

/-- The larger square formed by the outer points of the grid -/
def outerSquare (grid : PointGrid) : GridSquare :=
  { sideLength := grid.size - 1 }

/-- The shaded square inside the grid -/
def innerSquare : GridSquare :=
  { sideLength := 2 }

/-- Calculate the area of a square -/
def area (square : GridSquare) : ℕ :=
  square.sideLength * square.sideLength

/-- The fraction of the outer square's area occupied by the inner square -/
def areaFraction (grid : PointGrid) : ℚ :=
  (area innerSquare : ℚ) / (area (outerSquare grid))

theorem area_fraction_on_7x7_grid :
  areaFraction { size := 7 } = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_fraction_on_7x7_grid_l3379_337997


namespace NUMINAMATH_CALUDE_ellipse_on_y_axis_l3379_337950

/-- Given real numbers m and n where m > n > 0, the equation mx² + ny² = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∃ (c : ℝ), c > 0 ∧ c^2 = a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_on_y_axis_l3379_337950


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3379_337901

/-- Given a book with a specific number of chapters and pages per chapter, 
    calculate the total number of pages in the book. -/
theorem book_pages_calculation (num_chapters : ℕ) (pages_per_chapter : ℕ) 
    (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) : 
    num_chapters * pages_per_chapter = 1891 := by
  sorry

#check book_pages_calculation

end NUMINAMATH_CALUDE_book_pages_calculation_l3379_337901


namespace NUMINAMATH_CALUDE_earnings_per_lawn_l3379_337922

theorem earnings_per_lawn (total_lawns forgotten_lawns : ℕ) (total_earnings : ℚ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earnings = 36 →
  total_earnings / (total_lawns - forgotten_lawns) = 9 := by
sorry

end NUMINAMATH_CALUDE_earnings_per_lawn_l3379_337922


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3379_337911

theorem greatest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (5^8 + 10^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (5^8 + 10^7) → q ≤ p :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3379_337911


namespace NUMINAMATH_CALUDE_dribbles_proof_l3379_337972

/-- Calculates the total number of dribbles in a given time period with decreasing dribble rate --/
def totalDribbles (initialDribbles : ℕ) (initialTime : ℕ) (secondIntervalDribbles : ℕ) (secondIntervalTime : ℕ) (totalTime : ℕ) (decreaseRate : ℕ) : ℕ :=
  let remainingTime := totalTime - initialTime - secondIntervalTime
  let fullIntervals := remainingTime / secondIntervalTime
  let lastIntervalTime := remainingTime % secondIntervalTime
  
  let initialPeriodDribbles := initialDribbles
  let secondPeriodDribbles := secondIntervalDribbles
  
  let remainingFullIntervalsDribbles := 
    (List.range fullIntervals).foldl (fun acc i => 
      acc + (secondIntervalDribbles - i * decreaseRate)) 0
  
  let lastIntervalDribbles := 
    (secondIntervalDribbles - fullIntervals * decreaseRate) * lastIntervalTime / secondIntervalTime
  
  initialPeriodDribbles + secondPeriodDribbles + remainingFullIntervalsDribbles + lastIntervalDribbles

/-- The total number of dribbles in 27 seconds is 83 --/
theorem dribbles_proof : 
  totalDribbles 13 3 18 5 27 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_dribbles_proof_l3379_337972


namespace NUMINAMATH_CALUDE_different_suit_card_selection_l3379_337928

theorem different_suit_card_selection :
  let total_cards : ℕ := 52
  let num_suits : ℕ := 4
  let cards_per_suit : ℕ := 13
  let selection_size : ℕ := 4

  (num_suits ^ selection_size) = 28561 :=
by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_selection_l3379_337928


namespace NUMINAMATH_CALUDE_log_equality_l3379_337988

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l3379_337988


namespace NUMINAMATH_CALUDE_fourth_side_length_l3379_337993

/-- A quadrilateral inscribed in a circle with three known side lengths -/
structure InscribedQuadrilateral where
  -- Three known side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- The fourth side length
  d : ℝ
  -- Condition that the quadrilateral is inscribed in a circle
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  -- Condition that the areas of triangles ABC and ACD are equal
  equal_areas : a * b = c * d

/-- Theorem stating the possible values of the fourth side length -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.a = 5 ∨ q.b = 5 ∨ q.c = 5)
  (h2 : q.a = 8 ∨ q.b = 8 ∨ q.c = 8)
  (h3 : q.a = 10 ∨ q.b = 10 ∨ q.c = 10) :
  q.d = 4 ∨ q.d = 6.25 ∨ q.d = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l3379_337993


namespace NUMINAMATH_CALUDE_team_selection_with_twins_l3379_337987

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players to be chosen
def chosen_players : ℕ := 7

-- Define the number of twins
def num_twins : ℕ := 2

-- Theorem statement
theorem team_selection_with_twins :
  (Nat.choose total_players chosen_players) - 
  (Nat.choose (total_players - num_twins) chosen_players) = 20384 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_with_twins_l3379_337987


namespace NUMINAMATH_CALUDE_adams_shelves_l3379_337977

theorem adams_shelves (figures_per_shelf : ℕ) (total_figures : ℕ) (h1 : figures_per_shelf = 10) (h2 : total_figures = 80) :
  total_figures / figures_per_shelf = 8 := by
sorry

end NUMINAMATH_CALUDE_adams_shelves_l3379_337977


namespace NUMINAMATH_CALUDE_lewis_earnings_l3379_337936

/-- Calculates the weekly earnings of a person given the number of weeks worked,
    weekly rent, and final savings. -/
def weekly_earnings (weeks : ℕ) (rent : ℚ) (final_savings : ℚ) : ℚ :=
  (final_savings + weeks * rent) / weeks

theorem lewis_earnings :
  let weeks : ℕ := 1181
  let rent : ℚ := 216
  let final_savings : ℚ := 324775
  weekly_earnings weeks rent final_savings = 490.75 := by sorry

end NUMINAMATH_CALUDE_lewis_earnings_l3379_337936


namespace NUMINAMATH_CALUDE_inequality_proof_l3379_337940

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3379_337940


namespace NUMINAMATH_CALUDE_rain_probability_l3379_337959

theorem rain_probability (weihai_rain : Real) (zibo_rain : Real) (both_rain : Real) :
  weihai_rain = 0.2 →
  zibo_rain = 0.15 →
  both_rain = 0.06 →
  both_rain / weihai_rain = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l3379_337959


namespace NUMINAMATH_CALUDE_fraction_comparison_l3379_337903

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3379_337903


namespace NUMINAMATH_CALUDE_equation_positive_root_l3379_337921

theorem equation_positive_root (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 / (x - 1) - k / (1 - x) = 1)) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l3379_337921


namespace NUMINAMATH_CALUDE_f_properties_l3379_337990

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := sin x + m * x

theorem f_properties (m : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (deriv (f m)) x₁ = (deriv (f m)) x₂) ∧
  (∃ s : ℕ → ℝ, ∀ i j, i ≠ j → (deriv (f m)) (s i) = (deriv (f m)) (s j)) ∧
  (∃ t : ℕ → ℝ, ∀ i j, (deriv (f m)) (t i) = (deriv (f m)) (t j)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3379_337990


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3379_337983

theorem arithmetic_calculations : 
  (1 - (-4) + (-1) - 5 = -1) ∧ 
  (-1^4 + |5-8| + 27 / (-3) * (1/3) = -1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3379_337983


namespace NUMINAMATH_CALUDE_angle_mor_measure_l3379_337996

/-- A regular octagon with vertices LMNOPQR -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices j) (vertices (j + 1))

/-- The measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the measure of angle MOR in a regular octagon is π/8 radians (22.5°) -/
theorem angle_mor_measure (octagon : RegularOctagon) :
  let vertices := octagon.vertices
  angle_measure (vertices 0) (vertices 3) (vertices 5) = π / 8 := by sorry

end NUMINAMATH_CALUDE_angle_mor_measure_l3379_337996


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l3379_337900

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a^2 + 1 }

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := { x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

/-- Theorem stating the range of values for a where A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A a ⊆ B a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l3379_337900


namespace NUMINAMATH_CALUDE_rancher_animals_count_l3379_337942

/-- Proves that a rancher with 5 times as many cows as horses and 140 cows has 168 animals in total -/
theorem rancher_animals_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses → cows = 140 → horses + cows = 168 := by
  sorry

end NUMINAMATH_CALUDE_rancher_animals_count_l3379_337942


namespace NUMINAMATH_CALUDE_fort_food_duration_l3379_337923

/-- Calculates the initial number of days the food provision was meant to last given:
  * The initial number of men
  * The number of days after which some men left
  * The number of men who left
  * The number of days the food lasted after some men left
-/
def initialFoodDuration (initialMen : ℕ) (daysBeforeLeaving : ℕ) (menWhoLeft : ℕ) (remainingDays : ℕ) : ℕ :=
  (initialMen * daysBeforeLeaving + (initialMen - menWhoLeft) * remainingDays) / initialMen

theorem fort_food_duration :
  initialFoodDuration 150 10 25 42 = 45 := by
  sorry

#eval initialFoodDuration 150 10 25 42

end NUMINAMATH_CALUDE_fort_food_duration_l3379_337923


namespace NUMINAMATH_CALUDE_chess_positions_l3379_337956

/-- The number of different positions on a chessboard after both players make one move each -/
def num_positions : ℕ :=
  let pawns_per_player := 8
  let knights_per_player := 2
  let pawn_moves := 2
  let knight_moves := 2
  let moves_per_player := pawns_per_player * pawn_moves + knights_per_player * knight_moves
  moves_per_player * moves_per_player

theorem chess_positions : num_positions = 400 := by
  sorry

end NUMINAMATH_CALUDE_chess_positions_l3379_337956


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l3379_337975

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * m / 2

/-- Theorem: In a hockey league with 15 teams, where each team plays every other team 10 times,
    the total number of games played in the season is 1050. -/
theorem hockey_league_season_games :
  hockey_league_games 15 10 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l3379_337975


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l3379_337938

def current_hair_length : ℕ := 18
def hair_growth : ℕ := 4

theorem isabella_hair_growth :
  current_hair_length + hair_growth = 22 :=
by sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l3379_337938


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l3379_337946

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def is_parallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α = q.γ ∧ q.β = q.δ

/-- Definition of a trapezoid -/
def is_trapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral)
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  is_parallelogram q ∨ is_trapezoid q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l3379_337946


namespace NUMINAMATH_CALUDE_dog_travel_time_l3379_337957

theorem dog_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 5) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_travel_time_l3379_337957


namespace NUMINAMATH_CALUDE_jerry_initial_money_l3379_337963

-- Define Jerry's financial situation
def jerry_spent : ℕ := 6
def jerry_left : ℕ := 12

-- Theorem to prove
theorem jerry_initial_money : 
  jerry_spent + jerry_left = 18 := by sorry

end NUMINAMATH_CALUDE_jerry_initial_money_l3379_337963


namespace NUMINAMATH_CALUDE_inequality_proof_l3379_337918

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3379_337918


namespace NUMINAMATH_CALUDE_cos_210_degrees_l3379_337980

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l3379_337980


namespace NUMINAMATH_CALUDE_percentage_calculation_l3379_337958

theorem percentage_calculation (N I P : ℝ) : 
  N = 93.75 →
  I = 0.4 * N →
  (P / 100) * I = 6 →
  P = 16 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3379_337958


namespace NUMINAMATH_CALUDE_factorization_equality_l3379_337937

-- Define the equality we want to prove
theorem factorization_equality (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3379_337937


namespace NUMINAMATH_CALUDE_system_solution_l3379_337902

theorem system_solution (a b c d e : ℝ) : 
  (3 * a = (b + c + d)^3 ∧
   3 * b = (c + d + e)^3 ∧
   3 * c = (d + e + a)^3 ∧
   3 * d = (e + a + b)^3 ∧
   3 * e = (a + b + c)^3) →
  ((a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨
   (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨
   (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3379_337902


namespace NUMINAMATH_CALUDE_hexagon_minus_sectors_area_l3379_337945

/-- The area of the region inside a regular hexagon but outside circular sectors --/
theorem hexagon_minus_sectors_area (s : ℝ) (r : ℝ) (θ : ℝ) : 
  s = 10 → r = 5 → θ = 120 → 
  (6 * (s^2 * Real.sqrt 3 / 4)) - (6 * (θ / 360) * Real.pi * r^2) = 150 * Real.sqrt 3 - 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hexagon_minus_sectors_area_l3379_337945


namespace NUMINAMATH_CALUDE_yacht_capacity_problem_l3379_337973

theorem yacht_capacity_problem (large_capacity small_capacity : ℕ) : 
  (2 * large_capacity + 3 * small_capacity = 57) → 
  (3 * large_capacity + 2 * small_capacity = 68) → 
  (3 * large_capacity + 6 * small_capacity = 96) :=
by sorry

end NUMINAMATH_CALUDE_yacht_capacity_problem_l3379_337973


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l3379_337947

theorem complex_roots_on_circle : 
  ∃ (r : ℝ), r = 2/3 ∧ 
  ∀ (z : ℂ), (z + 1)^5 = 32 * z^5 → Complex.abs z = r :=
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l3379_337947


namespace NUMINAMATH_CALUDE_first_week_rate_correct_l3379_337952

/-- The daily rate for the first week in a student youth hostel. -/
def first_week_rate : ℝ := 18

/-- The daily rate for days after the first week. -/
def additional_week_rate : ℝ := 12

/-- The total number of days stayed. -/
def total_days : ℕ := 23

/-- The total cost for the stay. -/
def total_cost : ℝ := 318

/-- Theorem stating that the first week rate is correct given the conditions. -/
theorem first_week_rate_correct :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_week_rate_correct_l3379_337952


namespace NUMINAMATH_CALUDE_eighth_term_binomial_expansion_l3379_337984

theorem eighth_term_binomial_expansion :
  let n : ℕ := 10
  let r : ℕ := 8
  let coeff : ℕ := n.choose (r - 1)
  let term := coeff * (2 ^ (r - 1)) * x ^ (r - 1)
  term = 960 * x ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_eighth_term_binomial_expansion_l3379_337984


namespace NUMINAMATH_CALUDE_cos_a_plus_beta_half_l3379_337954

theorem cos_a_plus_beta_half (a β : ℝ) 
  (h1 : 0 < a ∧ a < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (a + π / 4) = 1 / 3)
  (h4 : Real.sin (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (a + β / 2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_plus_beta_half_l3379_337954


namespace NUMINAMATH_CALUDE_complex_abs_from_square_l3379_337924

theorem complex_abs_from_square (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_from_square_l3379_337924


namespace NUMINAMATH_CALUDE_rectangle_property_l3379_337989

/-- Represents a complex number --/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Represents a rectangle in the complex plane --/
structure ComplexRectangle where
  A : ComplexNumber
  B : ComplexNumber
  C : ComplexNumber
  D : ComplexNumber

/-- The theorem stating the properties of the given rectangle --/
theorem rectangle_property (rect : ComplexRectangle) :
  rect.A = ComplexNumber.mk 2 3 →
  rect.B = ComplexNumber.mk 3 2 →
  rect.C = ComplexNumber.mk (-2) (-3) →
  rect.D = ComplexNumber.mk (-3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_property_l3379_337989


namespace NUMINAMATH_CALUDE_stomachion_gray_area_l3379_337994

/-- A square with side length 12 cm divided into 14 polygons -/
structure StomachionPuzzle where
  side_length : ℝ
  num_polygons : ℕ
  h_side : side_length = 12
  h_polygons : num_polygons = 14

/-- A quadrilateral in the Stomachion puzzle -/
structure Quadrilateral (puzzle : StomachionPuzzle) where
  area : ℝ

/-- There exists a quadrilateral in the Stomachion puzzle with an area of 12 cm² -/
theorem stomachion_gray_area (puzzle : StomachionPuzzle) :
  ∃ (q : Quadrilateral puzzle), q.area = 12 := by
  sorry

end NUMINAMATH_CALUDE_stomachion_gray_area_l3379_337994


namespace NUMINAMATH_CALUDE_frank_bags_theorem_l3379_337905

/-- Given that Frank has a total number of candy pieces and puts an equal number of pieces in each bag, 
    calculate the number of bags used. -/
def bags_used (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem stating that Frank used 2 bags given the problem conditions -/
theorem frank_bags_theorem (total_candy : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : 
  bags_used total_candy candy_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_bags_theorem_l3379_337905


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3379_337910

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (10 * π / 180) = 2 * (3 * Real.sqrt 3 + 4) / 9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3379_337910


namespace NUMINAMATH_CALUDE_bottles_per_case_is_ten_l3379_337927

/-- The number of bottles produced per day -/
def bottles_per_day : ℕ := 72000

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 7200

/-- The number of bottles that a case can hold -/
def bottles_per_case : ℕ := bottles_per_day / cases_per_day

theorem bottles_per_case_is_ten : bottles_per_case = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_is_ten_l3379_337927


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3379_337995

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  Complex.im (5 * i / (3 + 4 * i)) = 3 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3379_337995


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l3379_337904

theorem product_purely_imaginary (x : ℝ) : 
  (∃ y : ℝ, (x + 2*I) * ((x + 1) + 3*I) * ((x + 2) + 4*I) = y*I) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l3379_337904


namespace NUMINAMATH_CALUDE_probability_no_distinct_roots_l3379_337981

-- Define the range of b and c
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Define the condition for not having distinct real roots
def no_distinct_roots (b c : Int) : Prop := b^2 - 4*c ≤ 0

-- Define the total number of possible pairs
def total_pairs : Nat := (15 * 15 : Nat)

-- Define the number of pairs that don't have distinct roots
def pairs_without_distinct_roots : Nat := 180

-- Theorem statement
theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : Rat) / total_pairs = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_distinct_roots_l3379_337981


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3379_337961

/-- Represents the number of students in each grade --/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the number of students to be sampled from each grade --/
structure SampleSize where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the stratified sample size for each grade --/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.grade10 + pop.grade11 + pop.grade12
  { grade10 := totalSample * pop.grade10 / totalPop,
    grade11 := totalSample * pop.grade11 / totalPop,
    grade12 := totalSample * pop.grade12 / totalPop }

/-- Theorem: The stratified sample for the given population and sample size is correct --/
theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 800 400
  let sample := stratifiedSample pop 18
  sample = SampleSize.mk 6 8 4 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3379_337961


namespace NUMINAMATH_CALUDE_grant_coverage_percentage_l3379_337968

def total_cost : ℝ := 30000
def savings : ℝ := 10000
def loan_amount : ℝ := 12000

theorem grant_coverage_percentage : 
  let remainder := total_cost - savings
  let grant_amount := remainder - loan_amount
  (grant_amount / remainder) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_grant_coverage_percentage_l3379_337968


namespace NUMINAMATH_CALUDE_james_purchase_cost_l3379_337906

/-- The total cost of James' purchase of shirts and pants -/
def total_cost (num_shirts : ℕ) (shirt_price : ℕ) (pant_price : ℕ) : ℕ :=
  let num_pants := num_shirts / 2
  num_shirts * shirt_price + num_pants * pant_price

/-- Theorem stating that James' purchase costs $100 -/
theorem james_purchase_cost : total_cost 10 6 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l3379_337906


namespace NUMINAMATH_CALUDE_product_sum_inequality_l3379_337915

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l3379_337915


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_theorem_l3379_337939

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The condition that |g(x)| = 6 for x = 0, 1, 3, 4, 5 -/
def SatisfiesCondition (g : FourthDegreePolynomial) : Prop :=
  |g 0| = 6 ∧ |g 1| = 6 ∧ |g 3| = 6 ∧ |g 4| = 6 ∧ |g 5| = 6

theorem fourth_degree_polynomial_theorem (g : FourthDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 7| = 106.8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_theorem_l3379_337939


namespace NUMINAMATH_CALUDE_optimal_schedule_l3379_337992

/-- Represents the construction teams -/
inductive Team
| A
| B

/-- Represents the construction schedule -/
structure Schedule where
  teamA_months : ℕ
  teamB_months : ℕ

/-- Calculates the total work done by a team given the months worked and their efficiency -/
def work_done (months : ℕ) (efficiency : ℚ) : ℚ :=
  months * efficiency

/-- Calculates the cost of a schedule given the monthly rates -/
def schedule_cost (s : Schedule) (rateA rateB : ℕ) : ℕ :=
  s.teamA_months * rateA + s.teamB_months * rateB

/-- Checks if a schedule is valid according to the given constraints -/
def is_valid_schedule (s : Schedule) : Prop :=
  s.teamA_months > 0 ∧ s.teamA_months ≤ 6 ∧
  s.teamB_months > 0 ∧ s.teamB_months ≤ 24 ∧
  s.teamA_months + s.teamB_months ≤ 24

/-- The main theorem to be proved -/
theorem optimal_schedule :
  ∃ (s : Schedule),
    is_valid_schedule s ∧
    work_done s.teamA_months (1 / 18) + work_done s.teamB_months (1 / 27) = 1 ∧
    ∀ (s' : Schedule),
      is_valid_schedule s' ∧
      work_done s'.teamA_months (1 / 18) + work_done s'.teamB_months (1 / 27) = 1 →
      schedule_cost s 80000 50000 ≤ schedule_cost s' 80000 50000 ∧
    s.teamA_months = 2 ∧ s.teamB_months = 24 :=
  sorry

end NUMINAMATH_CALUDE_optimal_schedule_l3379_337992


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3379_337974

theorem no_such_function_exists : ¬∃ (f : ℝ → ℝ), 
  (∀ x, f x ≠ 0) ∧ 
  (∀ x, 2 * f (f x) = f x) ∧ 
  (∀ x, f x ≥ 0) ∧
  Differentiable ℝ f :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3379_337974


namespace NUMINAMATH_CALUDE_special_function_properties_l3379_337982

/-- A function satisfying the given properties -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g x > 0) ∧ (∀ a b, g a * g b = g (a * b))

theorem special_function_properties (g : ℝ → ℝ) (h : SpecialFunction g) :
  (g 1 = 1) ∧ (∀ a, g (1 / a) = 1 / g a) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l3379_337982


namespace NUMINAMATH_CALUDE_inscribe_two_equal_circles_l3379_337964

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to a line segment --/
def isTangentToSide (c : Circle) (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two circles are tangent to each other --/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is inside a triangle --/
def isInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Theorem stating that two equal circles can be inscribed in any triangle,
    each touching two sides of the triangle and the other circle --/
theorem inscribe_two_equal_circles (t : Triangle) : 
  ∃ (c1 c2 : Circle), 
    c1.radius = c2.radius ∧ 
    isInside c1 t ∧ 
    isInside c2 t ∧ 
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    areTangent c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_inscribe_two_equal_circles_l3379_337964


namespace NUMINAMATH_CALUDE_complex_number_problem_l3379_337933

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * (1 + Complex.I) = 1 - Complex.I) →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I ∧ Complex.abs z₂ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3379_337933


namespace NUMINAMATH_CALUDE_maggie_goldfish_fraction_l3379_337976

def total_goldfish : ℕ := 100
def allowed_fraction : ℚ := 1 / 2
def remaining_to_catch : ℕ := 20

theorem maggie_goldfish_fraction :
  let allowed_total := total_goldfish * allowed_fraction
  let caught := allowed_total - remaining_to_catch
  caught / allowed_total = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_maggie_goldfish_fraction_l3379_337976


namespace NUMINAMATH_CALUDE_fish_pond_population_l3379_337943

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 50 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#eval (50 * 50) / 2  -- Should evaluate to 1250

end NUMINAMATH_CALUDE_fish_pond_population_l3379_337943


namespace NUMINAMATH_CALUDE_fraction_of_lunch_eaten_l3379_337999

def total_calories : ℕ := 40
def recommended_calories : ℕ := 25
def extra_calories : ℕ := 5

def actual_calories : ℕ := recommended_calories + extra_calories

theorem fraction_of_lunch_eaten :
  (actual_calories : ℚ) / total_calories = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_lunch_eaten_l3379_337999


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l3379_337969

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l3379_337969


namespace NUMINAMATH_CALUDE_area_outside_rectangle_within_square_l3379_337914

/-- The area of the region outside a centered rectangle but within a square. -/
theorem area_outside_rectangle_within_square : 
  ∀ (square_side rectangle_length rectangle_width : ℝ),
    square_side = 10 →
    rectangle_length = 5 →
    rectangle_width = 2 →
    square_side > rectangle_length ∧ square_side > rectangle_width →
    square_side^2 - rectangle_length * rectangle_width = 90 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_rectangle_within_square_l3379_337914


namespace NUMINAMATH_CALUDE_max_work_hours_l3379_337932

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  regularHours : ℕ := 20
  regularRate : ℚ := 8
  overtimeRate : ℚ := 10
  maxEarnings : ℚ := 760

/-- Calculates the total hours worked given regular and overtime hours --/
def totalHours (regular : ℕ) (overtime : ℕ) : ℕ :=
  regular + overtime

/-- Calculates the total earnings given regular and overtime hours --/
def totalEarnings (schedule : WorkSchedule) (overtime : ℕ) : ℚ :=
  (schedule.regularHours : ℚ) * schedule.regularRate + (overtime : ℚ) * schedule.overtimeRate

/-- Theorem: The maximum number of hours Mary can work in a week is 80 --/
theorem max_work_hours (schedule : WorkSchedule) : 
  ∃ (overtime : ℕ), 
    totalHours schedule.regularHours overtime = 80 ∧ 
    totalEarnings schedule overtime ≤ schedule.maxEarnings ∧
    ∀ (h : ℕ), totalEarnings schedule h ≤ schedule.maxEarnings → 
      totalHours schedule.regularHours h ≤ 80 :=
by
  sorry

end NUMINAMATH_CALUDE_max_work_hours_l3379_337932


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_280_476_l3379_337965

theorem lcm_gcf_ratio_280_476 : 
  Nat.lcm 280 476 / Nat.gcd 280 476 = 170 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_280_476_l3379_337965


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l3379_337917

theorem zigzag_angle_theorem (ACB FEG DCE DEC : Real) : 
  ACB = 10 →
  FEG = 26 →
  DCE + 14 + 80 = 180 →
  DEC + 33 + 64 = 180 →
  ∃ θ : Real, θ = 11 ∧ θ + DCE + DEC = 180 :=
by sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l3379_337917


namespace NUMINAMATH_CALUDE_train_length_proof_l3379_337925

/-- Proves that the length of each train is 150 meters given the specified conditions -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 108 →
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let train_length := relative_speed * passing_time / 2
  train_length = 150 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l3379_337925


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3379_337913

open Real

theorem triangle_abc_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  -- Given conditions
  (0 < A) → (A < 2 * π / 3) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (m = (a / 2, c / 2)) →
  (n = (cos C, cos A)) →
  (n.1 * m.1 + n.2 * m.2 = b * cos B) →
  (cos ((A - C) / 2) = sqrt 3 * sin A) →
  (m.1 * m.1 + m.2 * m.2 = 5) →
  -- Conclusions
  (B = π / 3) ∧
  (1 / 2 * a * b * sin C = 2 * sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3379_337913


namespace NUMINAMATH_CALUDE_pizza_combinations_l3379_337912

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 3

/-- The maximum number of unique pizzas that can be made -/
def max_pizzas : ℕ := Nat.choose n k

theorem pizza_combinations :
  max_pizzas = 56 := by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3379_337912


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3379_337998

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x = x^2 - 2) →
  (∀ x, a * x^2 + b * x + c = 0) →
  (a = 1 ∧ b = -3 ∧ c = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3379_337998


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_minus_2_bounds_l3379_337909

theorem sqrt_3_times_sqrt_6_minus_2_bounds : 2 < Real.sqrt 3 * Real.sqrt 6 - 2 ∧ Real.sqrt 3 * Real.sqrt 6 - 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_minus_2_bounds_l3379_337909


namespace NUMINAMATH_CALUDE_factorization_problem_1_l3379_337920

theorem factorization_problem_1 (a : ℝ) :
  3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l3379_337920


namespace NUMINAMATH_CALUDE_sin_difference_bound_l3379_337916

theorem sin_difference_bound (N : ℕ) :
  ∃ (n k : ℕ), n ≠ k ∧ n ≤ N + 1 ∧ k ≤ N + 1 ∧ |Real.sin n - Real.sin k| < 2 / N :=
sorry

end NUMINAMATH_CALUDE_sin_difference_bound_l3379_337916


namespace NUMINAMATH_CALUDE_average_weight_proof_l3379_337991

theorem average_weight_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l3379_337991


namespace NUMINAMATH_CALUDE_remove_parentheses_first_step_l3379_337931

/-- Represents the steps in solving a linear equation -/
inductive SolvingStep
  | RemoveParentheses
  | EliminateDenominator
  | MoveTerms
  | CombineTerms

/-- Represents a linear equation -/
structure LinearEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- The given equation: 2x + 3(2x - 1) = 16 - (x + 1) -/
def givenEquation : LinearEquation :=
  { lhs := λ x ↦ 2*x + 3*(2*x - 1)
    rhs := λ x ↦ 16 - (x + 1) }

/-- The first step in solving the given linear equation -/
def firstSolvingStep (eq : LinearEquation) : SolvingStep := sorry

/-- Theorem stating that removing parentheses is the first step for the given equation -/
theorem remove_parentheses_first_step :
  firstSolvingStep givenEquation = SolvingStep.RemoveParentheses := sorry

end NUMINAMATH_CALUDE_remove_parentheses_first_step_l3379_337931


namespace NUMINAMATH_CALUDE_largest_increase_2018_2019_l3379_337962

def students : Fin 6 → ℕ
  | 0 => 110  -- 2015
  | 1 => 125  -- 2016
  | 2 => 130  -- 2017
  | 3 => 140  -- 2018
  | 4 => 160  -- 2019
  | 5 => 165  -- 2020

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 5 := sorry

theorem largest_increase_2018_2019 :
  largestIncreaseYears = 3 ∧
  ∀ i : Fin 5, percentageIncrease (students i) (students (i + 1)) ≤
    percentageIncrease (students 3) (students 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2018_2019_l3379_337962


namespace NUMINAMATH_CALUDE_equation_solver_l3379_337919

theorem equation_solver (m : ℕ) (p : ℝ) 
  (h1 : ((1^m) / (5^m)) * ((1^16) / (4^16)) = 1 / (2*(p^31)))
  (h2 : m = 31) : 
  p = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solver_l3379_337919


namespace NUMINAMATH_CALUDE_root_cube_sum_condition_l3379_337907

theorem root_cube_sum_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) ∧ 
    (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) ∧ 
    (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) ∧ 
    ((x₁-3)^3 + (x₂-3)^3 + (x₃-3)^3 = 0)) ↔ 
  (a = -9) :=
sorry

end NUMINAMATH_CALUDE_root_cube_sum_condition_l3379_337907


namespace NUMINAMATH_CALUDE_expression_simplification_l3379_337979

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a / (a + 1) - 1 / (a + 1)) / ((a - 1) / (a^2 + 2*a + 1)) = 2024 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3379_337979


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3379_337966

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_positive : d > 0

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h_sum : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h_product : seq.a 1 * seq.a 2 * seq.a 3 = 45) :
  seq.a 2009 + seq.a 2010 + seq.a 2011 = 24111 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3379_337966


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l3379_337970

-- Problem 1
theorem calculation_proof :
  18 + |-(Real.sqrt 2)| - (2012 - Real.pi)^0 - 4 * Real.sin (45 * π / 180) = 17 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem equation_solution_proof :
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x) / (x^2 - 4) - 2 / (x - 2) = 1 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l3379_337970


namespace NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l3379_337935

theorem container_capacity : ℝ → Prop :=
  fun C =>
    (C > 0) ∧                   -- Capacity is positive
    (1/2 * C + 20 = 3/4 * C) →  -- Adding 20 liters to half-full makes it 3/4 full
    C = 80                      -- The capacity is 80 liters

-- Proof
theorem container_capacity_proof : ∃ C, container_capacity C :=
  sorry

end NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l3379_337935


namespace NUMINAMATH_CALUDE_mother_age_is_36_l3379_337934

/-- Petra's age -/
def petra_age : ℕ := 11

/-- The sum of Petra's and her mother's ages -/
def age_sum : ℕ := 47

/-- Petra's mother's age -/
def mother_age : ℕ := age_sum - petra_age

/-- Theorem: Petra's mother is 36 years old -/
theorem mother_age_is_36 : mother_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_is_36_l3379_337934


namespace NUMINAMATH_CALUDE_marker_cost_l3379_337948

/-- The cost of notebooks and markers -/
theorem marker_cost (n m : ℝ) 
  (eq1 : 3 * n + 4 * m = 5.70)
  (eq2 : 5 * n + 2 * m = 4.90) : 
  m = 0.9857 := by
sorry

end NUMINAMATH_CALUDE_marker_cost_l3379_337948


namespace NUMINAMATH_CALUDE_modular_arithmetic_proof_l3379_337953

theorem modular_arithmetic_proof : (305 * 20 - 20 * 9 + 5) % 19 = 16 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_proof_l3379_337953


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3379_337926

theorem complex_modulus_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3379_337926
