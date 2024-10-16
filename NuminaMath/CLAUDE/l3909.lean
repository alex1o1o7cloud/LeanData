import Mathlib

namespace NUMINAMATH_CALUDE_sin_double_angle_problem_l3909_390912

theorem sin_double_angle_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (π / 2 - α) = 3 / 5) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_problem_l3909_390912


namespace NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_6_l3909_390905

theorem sqrt_combinable_with_sqrt_6 :
  ∀ x : ℝ, x > 0 →
  (x = 12 ∨ x = 15 ∨ x = 18 ∨ x = 24) →
  (∃ q : ℚ, Real.sqrt x = q * Real.sqrt 6) ↔ x = 24 := by
sorry

end NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_6_l3909_390905


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l3909_390911

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 
  n = 1008 ∧ 
  n % 18 = 0 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m : ℕ, m % 18 = 0 → 1000 ≤ m → m < 10000 → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l3909_390911


namespace NUMINAMATH_CALUDE_parallel_segments_determine_y_coordinate_l3909_390917

/-- Given four points on a Cartesian plane, if two line segments formed by these points
    are parallel, then the y-coordinate of the fourth point is determined. -/
theorem parallel_segments_determine_y_coordinate 
  (A B X Y : ℝ × ℝ) 
  (hA : A = (-4, 0)) 
  (hB : B = (0, -4)) 
  (hX : X = (5, 8)) 
  (hY : ∃ k, Y = (19, k)) 
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (B.2 - A.2) * (Y.1 - X.1)) :
  Y.2 = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_determine_y_coordinate_l3909_390917


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l3909_390983

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 7 ∧ 
  (n / 100) % 10 = 8 ∧ 
  n % 11 = 0 ∧
  n = 847 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l3909_390983


namespace NUMINAMATH_CALUDE_triangle_area_l3909_390982

/-- The area of a triangle with base 8 and height 4 is 16 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 8 → 
  height = 4 → 
  area = (base * height) / 2 → 
  area = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3909_390982


namespace NUMINAMATH_CALUDE_gcd_minus_twelve_equals_thirtysix_l3909_390960

theorem gcd_minus_twelve_equals_thirtysix :
  Nat.gcd 7344 48 - 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_minus_twelve_equals_thirtysix_l3909_390960


namespace NUMINAMATH_CALUDE_candy_jar_problem_l3909_390922

theorem candy_jar_problem (banana_jar grape_jar peanut_butter_jar : ℕ) : 
  banana_jar = 43 →
  grape_jar = banana_jar + 5 →
  peanut_butter_jar = 4 * grape_jar →
  peanut_butter_jar = 192 := by
sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l3909_390922


namespace NUMINAMATH_CALUDE_coronavirus_survey_census_l3909_390913

/-- A survey type -/
inductive SurveyType
| HeightOfStudents
| LightBulbLifespan
| GlobalGenderRatio
| CoronavirusExposure

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  smallGroup : Bool
  specificGroup : Bool
  completeDataNecessary : Bool

/-- Define what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.smallGroup ∧ c.specificGroup ∧ c.completeDataNecessary

/-- Assign characteristics to each survey type -/
def surveyCharacteristics : SurveyType → SurveyCharacteristics
| SurveyType.HeightOfStudents => ⟨false, true, false⟩
| SurveyType.LightBulbLifespan => ⟨true, true, false⟩
| SurveyType.GlobalGenderRatio => ⟨false, false, false⟩
| SurveyType.CoronavirusExposure => ⟨true, true, true⟩

/-- Theorem: The coronavirus exposure survey is the only one suitable for a census -/
theorem coronavirus_survey_census :
  ∀ (s : SurveyType), suitableForCensus (surveyCharacteristics s) ↔ s = SurveyType.CoronavirusExposure :=
sorry

end NUMINAMATH_CALUDE_coronavirus_survey_census_l3909_390913


namespace NUMINAMATH_CALUDE_jason_seashells_l3909_390939

/-- The number of seashells Jason has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Jason has 36 seashells after starting with 49 and giving away 13 -/
theorem jason_seashells : remaining_seashells 49 13 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l3909_390939


namespace NUMINAMATH_CALUDE_problem_statement_l3909_390968

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3909_390968


namespace NUMINAMATH_CALUDE_function_inequality_l3909_390953

theorem function_inequality (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y ≥ Real.sin x + Real.cos y) :
  ∃ p q : ℝ → ℝ, 
    (∀ x : ℝ, f x = (Real.sin x + Real.cos x + p x - q x) / 2) ∧
    (∀ x : ℝ, g x = (Real.sin x - Real.cos x + p x + q x) / 2) ∧
    (∀ x y : ℝ, p x ≥ q y) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3909_390953


namespace NUMINAMATH_CALUDE_total_profit_calculation_l3909_390971

/-- Calculates the total profit given investments and one partner's share --/
def calculate_total_profit (anand_investment deepak_investment deepak_share : ℚ) : ℚ :=
  let total_parts := anand_investment + deepak_investment
  let deepak_parts := deepak_investment
  deepak_share * total_parts / deepak_parts

/-- The total profit is 1380.48 given the investments and Deepak's share --/
theorem total_profit_calculation (anand_investment deepak_investment deepak_share : ℚ) 
  (h1 : anand_investment = 2250)
  (h2 : deepak_investment = 3200)
  (h3 : deepak_share = 810.28) :
  calculate_total_profit anand_investment deepak_investment deepak_share = 1380.48 := by
  sorry

#eval calculate_total_profit 2250 3200 810.28

end NUMINAMATH_CALUDE_total_profit_calculation_l3909_390971


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l3909_390949

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4  -- radius of smaller sphere
  let r₂ : ℝ := 7  -- radius of larger sphere
  let V : ℝ := (4 / 3) * Real.pi * (r₂^3 - r₁^3)  -- volume between spheres
  V = 372 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l3909_390949


namespace NUMINAMATH_CALUDE_no_positive_solution_l3909_390970

theorem no_positive_solution :
  ¬ ∃ (a b c d : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
    a * d + b = c ∧
    Real.sqrt a * Real.sqrt d + Real.sqrt b = Real.sqrt c :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l3909_390970


namespace NUMINAMATH_CALUDE_solution_306_is_valid_l3909_390956

def is_valid_solution (a b c : Nat) : Prop :=
  a ≠ 0 ∧ 
  b = 0 ∧ 
  c ≠ 0 ∧ 
  a ≠ c ∧
  1995 * (a * 100 + c) = 1995 * a * 100 + 1995 * c

theorem solution_306_is_valid : is_valid_solution 3 0 6 := by
  sorry

#check solution_306_is_valid

end NUMINAMATH_CALUDE_solution_306_is_valid_l3909_390956


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3909_390975

theorem simplify_sqrt_expression :
  2 * Real.sqrt 5 - 3 * Real.sqrt 25 + 4 * Real.sqrt 80 = 18 * Real.sqrt 5 - 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3909_390975


namespace NUMINAMATH_CALUDE_fraction_equality_l3909_390985

theorem fraction_equality (x y : ℝ) (h : (x - y) / (x + y) = 5) :
  (2 * x + 3 * y) / (3 * x - 2 * y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3909_390985


namespace NUMINAMATH_CALUDE_consecutive_blue_red_probability_l3909_390914

def num_green : ℕ := 4
def num_blue : ℕ := 3
def num_red : ℕ := 5
def total_chips : ℕ := num_green + num_blue + num_red

def probability_consecutive_blue_red : ℚ :=
  (num_blue.factorial * num_red.factorial * (Nat.choose (num_green + 2) 2)) /
  total_chips.factorial

theorem consecutive_blue_red_probability :
  probability_consecutive_blue_red = 1 / 44352 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_blue_red_probability_l3909_390914


namespace NUMINAMATH_CALUDE_average_weight_increase_l3909_390987

/-- Proves that replacing a person weighing 45 kg with a person weighing 65 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_group_size : ℕ) 
                                 (old_weight new_weight : ℝ) : 
  initial_group_size = 8 →
  old_weight = 45 →
  new_weight = 65 →
  (new_weight - old_weight) / initial_group_size = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3909_390987


namespace NUMINAMATH_CALUDE_three_fourths_of_forty_l3909_390936

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_forty_l3909_390936


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l3909_390998

/-- Represents a sequence of 5 integers -/
def Sequence := Fin 5 → ℕ

/-- Checks if a sequence is valid for systematic sampling -/
def isValidSample (s : Sequence) (totalBags : ℕ) (sampleSize : ℕ) : Prop :=
  ∃ (start : ℕ) (interval : ℕ),
    (∀ i : Fin 5, s i = start + i.val * interval) ∧
    (∀ i : Fin 5, 1 ≤ s i ∧ s i ≤ totalBags) ∧
    interval = totalBags / sampleSize

theorem systematic_sampling_proof :
  let s : Sequence := fun i => [7, 17, 27, 37, 47][i]
  let totalBags := 50
  let sampleSize := 5
  isValidSample s totalBags sampleSize :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l3909_390998


namespace NUMINAMATH_CALUDE_sqrt_sum_inverse_squares_l3909_390951

theorem sqrt_sum_inverse_squares : 
  Real.sqrt (1 / 25 + 1 / 36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inverse_squares_l3909_390951


namespace NUMINAMATH_CALUDE_fibonacci_factorial_sum_last_two_digits_l3909_390925

def fibonacci_factorial_series := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem fibonacci_factorial_sum_last_two_digits : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 50 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_sum_last_two_digits_l3909_390925


namespace NUMINAMATH_CALUDE_teaching_years_difference_l3909_390973

/-- The combined total of teaching years for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 102

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 43

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := 34

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := 25

theorem teaching_years_difference :
  total_years = virginia_years + adrienne_years + dennis_years ∧
  virginia_years = adrienne_years + 9 ∧
  virginia_years < dennis_years →
  dennis_years - virginia_years = 9 := by
sorry

end NUMINAMATH_CALUDE_teaching_years_difference_l3909_390973


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l3909_390977

-- Define the total points scored
def total_points : ℕ := 1200

-- Define the number of games played
def num_games : ℕ := 10

-- Define the points per game
def points_per_game : ℕ := total_points / num_games

-- Theorem statement
theorem melissa_points_per_game : points_per_game = 120 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l3909_390977


namespace NUMINAMATH_CALUDE_accommodation_arrangements_count_l3909_390965

/-- Represents the types of rooms in the hotel -/
inductive RoomType
  | Triple
  | Double
  | Single

/-- Represents a person staying in the hotel -/
inductive Person
  | Adult
  | Child

/-- Calculates the number of ways to arrange accommodation for 3 adults and 2 children
    in a hotel with one triple room, one double room, and one single room,
    where children must be accompanied by an adult -/
def accommodationArrangements (rooms : List RoomType) (people : List Person) : Nat :=
  sorry

/-- The main theorem stating that there are 27 different ways to arrange the accommodation -/
theorem accommodation_arrangements_count :
  accommodationArrangements
    [RoomType.Triple, RoomType.Double, RoomType.Single]
    [Person.Adult, Person.Adult, Person.Adult, Person.Child, Person.Child] = 27 :=
by sorry

end NUMINAMATH_CALUDE_accommodation_arrangements_count_l3909_390965


namespace NUMINAMATH_CALUDE_banana_arrangements_l3909_390964

-- Define the word and its properties
def banana_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

-- Theorem statement
theorem banana_arrangements : 
  (banana_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3909_390964


namespace NUMINAMATH_CALUDE_convex_polygon_coverage_l3909_390944

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a function to check if a polygon can cover a triangle of given area
def can_cover (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Define a function to check if a polygon can be covered by a triangle of given area
def can_be_covered (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Theorem statement
theorem convex_polygon_coverage (M : ConvexPolygon) :
  (¬ can_cover M 1) → can_be_covered M 4 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_coverage_l3909_390944


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3909_390935

open Real

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (sin x + 3 * (1 / sin x))^2 + (cos x + 3 * (1 / cos x))^2 ≥ 52 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (sin y + 3 * (1 / sin y))^2 + (cos y + 3 * (1 / cos y))^2 = 52 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3909_390935


namespace NUMINAMATH_CALUDE_find_m_l3909_390999

def U : Set Nat := {0, 1, 2, 3}

def A (m : ℝ) : Set Nat := {x ∈ U | x^2 + m * x = 0}

def complement_A : Set Nat := {1, 2}

theorem find_m :
  ∃ m : ℝ, (A m = U \ complement_A) ∧ (m = -3) :=
sorry

end NUMINAMATH_CALUDE_find_m_l3909_390999


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3909_390980

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-3, -4)

/-- The expected reflected point -/
def expected_reflection : ℝ × ℝ := (-3, 4)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3909_390980


namespace NUMINAMATH_CALUDE_unique_plane_for_skew_lines_l3909_390928

/-- Two lines in 3D space -/
structure Line3D where
  -- Define properties of a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a 3D plane

/-- Two lines are skew if they are not coplanar and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

theorem unique_plane_for_skew_lines (a b : Line3D) 
  (h1 : skew a b) (h2 : ¬perpendicular a b) : 
  ∃! α : Plane3D, contained_in a α ∧ parallel_to b α :=
sorry

end NUMINAMATH_CALUDE_unique_plane_for_skew_lines_l3909_390928


namespace NUMINAMATH_CALUDE_correct_field_equation_l3909_390974

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  lengthWidthDiff : ℕ

/-- The equation representing the relationship between length and area for the given field -/
def fieldEquation (field : RectangularField) (x : ℕ) : Prop :=
  x * (x - field.lengthWidthDiff) = field.area

/-- Theorem stating that the equation correctly represents the given field properties -/
theorem correct_field_equation (field : RectangularField) 
    (h1 : field.area = 864) (h2 : field.lengthWidthDiff = 12) :
    ∃ x : ℕ, fieldEquation field x :=
  sorry

end NUMINAMATH_CALUDE_correct_field_equation_l3909_390974


namespace NUMINAMATH_CALUDE_max_value_theorem_l3909_390984

theorem max_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3909_390984


namespace NUMINAMATH_CALUDE_pythagorean_triple_product_divisible_by_six_l3909_390957

theorem pythagorean_triple_product_divisible_by_six (A B C : ℤ) : 
  A^2 + B^2 = C^2 → (6 ∣ A * B) := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_product_divisible_by_six_l3909_390957


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3909_390986

theorem alcohol_mixture_percentage :
  let initial_volume : ℝ := 24
  let initial_alcohol_percentage : ℝ := 90
  let added_water : ℝ := 16
  let initial_alcohol_volume : ℝ := initial_volume * (initial_alcohol_percentage / 100)
  let final_volume : ℝ := initial_volume + added_water
  let final_alcohol_percentage : ℝ := (initial_alcohol_volume / final_volume) * 100
  final_alcohol_percentage = 54 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3909_390986


namespace NUMINAMATH_CALUDE_race_distance_is_400_l3909_390908

/-- Represents the speed of a runner relative to others -/
structure RelativeSpeed where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculate the relative speeds based on race results -/
def calculate_relative_speeds : RelativeSpeed :=
  let ab_ratio := 500 / 450
  let bc_ratio := 500 / 475
  { a := ab_ratio * bc_ratio
  , b := bc_ratio
  , c := 1 }

/-- The race distance where A beats C by 58 meters -/
def race_distance (speeds : RelativeSpeed) : ℚ :=
  58 * speeds.a / (speeds.a - speeds.c)

/-- Theorem stating that the race distance is 400 meters -/
theorem race_distance_is_400 :
  race_distance calculate_relative_speeds = 400 := by sorry

end NUMINAMATH_CALUDE_race_distance_is_400_l3909_390908


namespace NUMINAMATH_CALUDE_right_pyramid_height_l3909_390903

/-- The height of a right pyramid with a square base -/
theorem right_pyramid_height (perimeter base_side diagonal_half : ℝ) 
  (apex_to_vertex : ℝ) (h_perimeter : perimeter = 40) 
  (h_base_side : base_side = perimeter / 4)
  (h_diagonal_half : diagonal_half = base_side * Real.sqrt 2 / 2)
  (h_apex_to_vertex : apex_to_vertex = 15) : 
  Real.sqrt (apex_to_vertex ^ 2 - diagonal_half ^ 2) = 5 * Real.sqrt 7 := by
  sorry

#check right_pyramid_height

end NUMINAMATH_CALUDE_right_pyramid_height_l3909_390903


namespace NUMINAMATH_CALUDE_nonreal_cube_root_sum_l3909_390907

/-- Given that ω is a nonreal root of x^3 = 1, prove that 
    (2 - 2ω + 2ω^2)^3 + (2 + 2ω - 2ω^2)^3 = 0 -/
theorem nonreal_cube_root_sum (ω : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : ω ≠ 1) : 
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_sum_l3909_390907


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3909_390950

theorem min_value_and_inequality (x a b : ℝ) : x > 0 ∧ a > 0 ∧ b > 0 → 
  (∀ y : ℝ, y > 0 → x + 1/x ≤ y + 1/y) ∧ 
  (a * b ≤ ((a + b) / 2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3909_390950


namespace NUMINAMATH_CALUDE_xyz_value_l3909_390955

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3909_390955


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l3909_390989

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l3909_390989


namespace NUMINAMATH_CALUDE_hanna_erasers_l3909_390900

/-- Given information about erasers owned by Tanya, Rachel, and Hanna -/
theorem hanna_erasers (tanya_erasers : ℕ) (tanya_red_erasers : ℕ) (rachel_erasers : ℕ) (hanna_erasers : ℕ) : 
  tanya_erasers = 20 →
  tanya_red_erasers = tanya_erasers / 2 →
  rachel_erasers = tanya_red_erasers / 2 - 3 →
  hanna_erasers = 2 * rachel_erasers →
  hanna_erasers = 4 := by
sorry

end NUMINAMATH_CALUDE_hanna_erasers_l3909_390900


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3909_390924

/-- Represents a normally distributed random variable -/
structure NormalRV (μ : ℝ) (σ : ℝ) where
  (μ_pos : μ > 0)
  (σ_pos : σ > 0)

/-- The probability of a random variable being in an interval -/
noncomputable def prob (X : Type) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (a σ : ℝ) (ξ : NormalRV a σ) 
  (h : prob (NormalRV a σ) 0 a = 0.3) : 
  prob (NormalRV a σ) 0 (2 * a) = 0.6 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3909_390924


namespace NUMINAMATH_CALUDE_flour_per_pizza_l3909_390966

def carnival_time : ℕ := 7 * 60 -- 7 hours in minutes
def flour_amount : ℚ := 22 -- 22 kg of flour
def pizza_time : ℕ := 10 -- 10 minutes per pizza
def extra_pizzas : ℕ := 2 -- 2 additional pizzas from leftover flour

theorem flour_per_pizza :
  let total_pizzas := carnival_time / pizza_time + extra_pizzas
  flour_amount / total_pizzas = 1/2 := by sorry

end NUMINAMATH_CALUDE_flour_per_pizza_l3909_390966


namespace NUMINAMATH_CALUDE_problem_sampling_is_systematic_l3909_390958

/-- Represents a sampling method -/
inductive SamplingMethod
| DrawingLots
| RandomNumberTable
| SystematicSampling
| Other

/-- Represents a high school with classes and student numbering -/
structure HighSchool where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Defines the conditions of the problem -/
def problem_conditions : HighSchool :=
  { num_classes := 12
  , students_per_class := 50
  , selected_number := 20 }

/-- Defines systematic sampling -/
def is_systematic_sampling (school : HighSchool) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SystematicSampling ∧
  school.num_classes > 0 ∧
  school.students_per_class > 0 ∧
  school.selected_number > 0 ∧
  school.selected_number ≤ school.students_per_class

/-- Theorem stating that the sampling method in the problem is systematic sampling -/
theorem problem_sampling_is_systematic :
  is_systematic_sampling problem_conditions SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_problem_sampling_is_systematic_l3909_390958


namespace NUMINAMATH_CALUDE_highest_probability_event_l3909_390943

-- Define the events and their probabilities
def event_A : ℝ := 0.5
def event_B : ℝ := 0.1
def event_C : ℝ := 0.9

-- Define a function to determine if an event has a high possibility
def high_possibility (p : ℝ) : Prop := p > 0.5

-- Theorem statement
theorem highest_probability_event :
  high_possibility event_C ∧
  ¬high_possibility event_A ∧
  ¬high_possibility event_B ∧
  event_C > event_A ∧
  event_C > event_B :=
sorry

end NUMINAMATH_CALUDE_highest_probability_event_l3909_390943


namespace NUMINAMATH_CALUDE_simplify_expression_l3909_390972

theorem simplify_expression (x : ℝ) : (3*x)^5 - (4*x)*(x^4) = 239*(x^5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3909_390972


namespace NUMINAMATH_CALUDE_joes_test_count_l3909_390993

theorem joes_test_count (n : ℕ) (initial_average final_average lowest_score : ℚ) 
  (h1 : initial_average = 50)
  (h2 : final_average = 55)
  (h3 : lowest_score = 35)
  (h4 : n * initial_average = (n - 1) * final_average + lowest_score)
  (h5 : n > 1) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_joes_test_count_l3909_390993


namespace NUMINAMATH_CALUDE_expression_equals_four_l3909_390981

theorem expression_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y^2) :
  (x + 1/x) * (y - 1/y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l3909_390981


namespace NUMINAMATH_CALUDE_range_of_a_l3909_390961

theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, x > 0 → (Real.exp x / x + a * Real.log x - a * x + Real.exp 2) ≥ 0) →
  (0 < a ∧ a ≤ Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3909_390961


namespace NUMINAMATH_CALUDE_smallest_possible_area_l3909_390904

theorem smallest_possible_area (S : ℕ) (A : ℕ) : 
  S * S = 2019 + A  -- Total area equation
  → A ≠ 1  -- Area of 2020th square is not 1
  → A ≥ 9  -- Smallest possible area is at least 9
  → ∃ (S' : ℕ), S' * S' = 2019 + 9  -- There exists a solution with area 9
  → A = 9  -- The smallest area is indeed 9
  := by sorry

end NUMINAMATH_CALUDE_smallest_possible_area_l3909_390904


namespace NUMINAMATH_CALUDE_proposition_and_variants_true_l3909_390976

theorem proposition_and_variants_true :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ∧
  (∀ x y : ℝ, x = 0 ∧ y = 0 → x^2 + y^2 = 0) ∧
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_true_l3909_390976


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3909_390931

/-- Represents the online store's sales and profit model -/
structure OnlineStore where
  initialPrice : ℕ
  initialSales : ℕ
  cost : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the monthly sales volume based on the price -/
def monthlySales (store : OnlineStore) (price : ℕ) : ℤ :=
  store.initialSales + store.salesIncrease * (store.initialPrice - price)

/-- Calculates the monthly profit based on the price -/
def monthlyProfit (store : OnlineStore) (price : ℕ) : ℤ :=
  (price - store.cost) * (monthlySales store price)

/-- Theorem stating the maximum profit and optimal price reduction -/
theorem max_profit_theorem (store : OnlineStore) :
  store.initialPrice = 80 ∧
  store.initialSales = 100 ∧
  store.cost = 40 ∧
  store.salesIncrease = 5 →
  ∃ (optimalReduction : ℕ),
    optimalReduction = 10 ∧
    monthlyProfit store (store.initialPrice - optimalReduction) = 4500 ∧
    ∀ (price : ℕ), monthlyProfit store price ≤ 4500 :=
by sorry

#check max_profit_theorem

end NUMINAMATH_CALUDE_max_profit_theorem_l3909_390931


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangent_radius_l3909_390948

/-- The radius of a circle that is tangent to the asymptotes of a specific hyperbola -/
theorem hyperbola_circle_tangent_radius : ∀ (r : ℝ), r > 0 →
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 4 = 1 →
    (∃ (t : ℝ), (x - 3)^2 + y^2 = r^2 ∧
      (y = (2/3) * x ∨ y = -(2/3) * x) ∧
      (∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 < r^2 →
        y' ≠ (2/3) * x' ∧ y' ≠ -(2/3) * x'))) →
  r = 6 * Real.sqrt 13 / 13 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangent_radius_l3909_390948


namespace NUMINAMATH_CALUDE_wills_remaining_money_l3909_390927

/-- Calculates the remaining money after a shopping trip with a refund -/
def remaining_money (initial_amount sweater_price tshirt_price shoes_price refund_percentage : ℚ) : ℚ :=
  initial_amount - sweater_price - tshirt_price + (shoes_price * refund_percentage)

/-- Theorem stating that Will's remaining money after the shopping trip is $81 -/
theorem wills_remaining_money :
  remaining_money 74 9 11 30 0.9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_wills_remaining_money_l3909_390927


namespace NUMINAMATH_CALUDE_decimal_division_l3909_390934

theorem decimal_division (x y : ℚ) (hx : x = 0.54) (hy : y = 0.006) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l3909_390934


namespace NUMINAMATH_CALUDE_prime_square_plus_200_is_square_l3909_390915

theorem prime_square_plus_200_is_square (p : ℕ) : 
  Prime p ∧ ∃ (n : ℕ), p^2 + 200 = n^2 ↔ p = 5 ∨ p = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_200_is_square_l3909_390915


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3909_390916

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3909_390916


namespace NUMINAMATH_CALUDE_box_comparison_l3909_390952

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def box_lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.x ∧ a.y ≤ b.z ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.x ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.z ∧ a.z ≤ b.x) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.x ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.y ∧ a.z ≤ b.x)

def box_gt (a b : Box) : Prop := box_lt b a

theorem box_comparison :
  let A : Box := ⟨5, 6, 3⟩
  let B : Box := ⟨1, 5, 4⟩
  let C : Box := ⟨2, 2, 3⟩
  (box_gt A B) ∧ (box_lt C A) := by sorry

end NUMINAMATH_CALUDE_box_comparison_l3909_390952


namespace NUMINAMATH_CALUDE_dividend_calculation_l3909_390901

theorem dividend_calculation (quotient divisor : ℝ) (h1 : quotient = 0.0012000000000000001) (h2 : divisor = 17) :
  quotient * divisor = 0.0204000000000000027 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3909_390901


namespace NUMINAMATH_CALUDE_solve_eggs_problem_l3909_390990

def eggs_problem (total_cost : ℝ) (price_per_egg : ℝ) (remaining_eggs : ℕ) : Prop :=
  let eggs_sold := total_cost / price_per_egg
  let initial_eggs := eggs_sold + remaining_eggs
  initial_eggs = 30

theorem solve_eggs_problem :
  eggs_problem 5 0.20 5 :=
sorry

end NUMINAMATH_CALUDE_solve_eggs_problem_l3909_390990


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3909_390906

theorem reciprocal_sum_of_quadratic_roots (α β : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 - 8 * r + 6 = 0 ∧ 
               7 * s^2 - 8 * s + 6 = 0 ∧ 
               α = 1 / r ∧ 
               β = 1 / s) → 
  α + β = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3909_390906


namespace NUMINAMATH_CALUDE_sam_non_black_cows_l3909_390996

/-- Given a herd of cows, calculate the number of non-black cows. -/
def non_black_cows (total : ℕ) (black : ℕ) : ℕ :=
  total - black

theorem sam_non_black_cows :
  let total := 18
  let black := (total / 2) + 5
  non_black_cows total black = 4 := by
sorry

end NUMINAMATH_CALUDE_sam_non_black_cows_l3909_390996


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3909_390995

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x - 1 = 0) ↔ k ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3909_390995


namespace NUMINAMATH_CALUDE_division_remainder_l3909_390923

theorem division_remainder : ∃ q : ℕ, 1234567 = 137 * q + 102 ∧ 102 < 137 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3909_390923


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l3909_390902

/-- Given an object with acceleration a, initial velocity V₀, 
    final velocity V, and displacement S, prove the time t 
    taken to reach V from V₀ -/
theorem time_from_velocity_and_displacement 
  (a V₀ V S t : ℝ) 
  (hv : V = a * t + V₀) 
  (hs : S = (1/3) * a * t^3 + V₀ * t) :
  t = (V - V₀) / a :=
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l3909_390902


namespace NUMINAMATH_CALUDE_problem_statement_l3909_390930

theorem problem_statement (a b k : ℝ) 
  (h1 : 2^a = k) 
  (h2 : 3^b = k) 
  (h3 : k ≠ 1) 
  (h4 : 1/a + 2/b = 1) : 
  k = 18 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3909_390930


namespace NUMINAMATH_CALUDE_no_winning_strategy_l3909_390988

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)
  (total : ℕ)
  (h_total : total = red + black)
  (h_standard : total = 52 ∧ red = 26 ∧ black = 26)

/-- Represents a strategy for playing the card game -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state and a strategy -/
noncomputable def win_probability (d : Deck) (s : Strategy) : ℝ :=
  d.red / d.total

/-- Theorem stating that no strategy can have a winning probability greater than 0.5 -/
theorem no_winning_strategy (s : Strategy) :
  ∀ d : Deck, win_probability d s ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l3909_390988


namespace NUMINAMATH_CALUDE_class_age_problem_l3909_390947

/-- Proves that if the average age of 6 people remains 19 years after adding a 1-year-old person,
    then the original average was calculated 1 year ago. -/
theorem class_age_problem (initial_total_age : ℕ) (years_passed : ℕ) : 
  initial_total_age / 6 = 19 →
  (initial_total_age + 6 * years_passed + 1) / 7 = 19 →
  years_passed = 1 := by
sorry

end NUMINAMATH_CALUDE_class_age_problem_l3909_390947


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l3909_390937

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : Nat
  height : Nat

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : Nat
  possibleLengths : List Nat

/-- Represents the constraints for building the wall --/
structure WallConstraints where
  noCutting : Bool
  staggeredJoints : Bool
  evenEnds : Bool

/-- Calculates the smallest number of blocks needed to build the wall --/
def minBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) (constraints : WallConstraints) : Nat :=
  sorry

/-- Theorem stating the smallest number of blocks needed for the given wall --/
theorem smallest_number_of_blocks
  (wall : WallDimensions)
  (block : BlockDimensions)
  (constraints : WallConstraints)
  (h_wall_length : wall.length = 120)
  (h_wall_height : wall.height = 7)
  (h_block_height : block.height = 1)
  (h_block_lengths : block.possibleLengths = [2, 3])
  (h_no_cutting : constraints.noCutting = true)
  (h_staggered : constraints.staggeredJoints = true)
  (h_even_ends : constraints.evenEnds = true) :
  minBlocksNeeded wall block constraints = 357 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l3909_390937


namespace NUMINAMATH_CALUDE_min_abs_beta_plus_delta_l3909_390978

open Complex

theorem min_abs_beta_plus_delta :
  ∀ β δ : ℂ,
  let g : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β*z + δ
  (g 1).im = 0 →
  (g (-I)).im = 0 →
  ∃ (β' δ' : ℂ),
    let g' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β'*z + δ'
    (g' 1).im = 0 ∧
    (g' (-I)).im = 0 ∧
    Complex.abs β' + Complex.abs δ' = 4 ∧
    ∀ (β'' δ'' : ℂ),
      let g'' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β''*z + δ''
      (g'' 1).im = 0 →
      (g'' (-I)).im = 0 →
      Complex.abs β'' + Complex.abs δ'' ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_beta_plus_delta_l3909_390978


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3909_390991

/-- 
Given a geometric sequence where:
- a₂ is the second term
- S₃ is the sum of the first three terms
- q is the common ratio

This theorem states that if a₂ = 9 and S₃ = 39, then q satisfies the equation q² - (10/3)q + 1 = 0.
-/
theorem geometric_sequence_common_ratio 
  (a₂ : ℝ) 
  (S₃ : ℝ) 
  (q : ℝ) 
  (h1 : a₂ = 9) 
  (h2 : S₃ = 39) : 
  q^2 - (10/3)*q + 1 = 0 := by
  sorry

#check geometric_sequence_common_ratio

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3909_390991


namespace NUMINAMATH_CALUDE_fraction_decomposition_sum_l3909_390918

theorem fraction_decomposition_sum : ∃ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 2 → x ≠ 4 → (D * x - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4)) ∧
  C + D = 8.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_sum_l3909_390918


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l3909_390929

theorem quadratic_roots_ratio (m : ℚ) : 
  (∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 + 9*r + m = 0 ∧ s^2 + 9*s + m = 0) → 
  m = 243/16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l3909_390929


namespace NUMINAMATH_CALUDE_roots_quadratic_sum_l3909_390932

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 - 3*a + 1 = 0) → 
  (b^2 - 3*b + 1 = 0) → 
  (1 / (a^2 + 1) + 1 / (b^2 + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_sum_l3909_390932


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3909_390954

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

-- Define a square inscribed in the ellipse
structure InscribedSquare where
  side : ℝ
  vertex_on_ellipse : ellipse (side/2) (side/2)

-- Theorem statement
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3909_390954


namespace NUMINAMATH_CALUDE_complex_sum_real_l3909_390909

theorem complex_sum_real (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = (3 / (a + 5) : ℂ) + (10 - a^2 : ℂ) * Complex.I ∧
  z₂ = (2 / (1 - a) : ℂ) + (2*a - 5 : ℂ) * Complex.I ∧
  (z₁ + z₂).im = 0 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_l3909_390909


namespace NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_l3909_390941

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_l3909_390941


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l3909_390933

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 3 = 2024)
                                  (h2 : Nat.choose 24 4 = 10626)
                                  (h3 : Nat.choose 24 5 = 42504) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l3909_390933


namespace NUMINAMATH_CALUDE_smallest_value_quadratic_l3909_390979

theorem smallest_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_quadratic_l3909_390979


namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l3909_390969

theorem fraction_subtraction_theorem : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l3909_390969


namespace NUMINAMATH_CALUDE_watch_cost_price_l3909_390919

/-- Proves that the cost price of a watch is 3000, given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 8 / 100 →
  price_difference = 540 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 3000 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3909_390919


namespace NUMINAMATH_CALUDE_tv_price_change_l3909_390942

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.4) = P * 1.12 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l3909_390942


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3909_390962

theorem min_value_of_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z →
    3*x^2 + 2*y^2 + 4*z^2 - x*y - 3*y*z - 5*z*x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3909_390962


namespace NUMINAMATH_CALUDE_calculate_expression_l3909_390926

theorem calculate_expression : 3 * 3^3 + 4^7 / 4^5 = 97 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3909_390926


namespace NUMINAMATH_CALUDE_forever_alive_characterization_l3909_390921

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i j : Fin m) : ℕ := sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i j : Fin m) : CellState := sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n := sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop := sorry

/-- Represents the existence of an initial configuration that stays alive forever -/
def existsForeverAliveConfig (m n : ℕ) : Prop :=
  ∃ (initial : Grid m n), ∀ (t : ℕ), hasAliveCell (Nat.iterate updateGrid t initial)

/-- The main theorem: characterizes the pairs (m, n) for which an eternally alive configuration exists -/
theorem forever_alive_characterization (m n : ℕ) :
  existsForeverAliveConfig m n ↔ (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1) :=
sorry

end NUMINAMATH_CALUDE_forever_alive_characterization_l3909_390921


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_parts_l3909_390940

theorem quadratic_roots_integer_parts (n : ℕ) (h : n ≥ 1) :
  let original_eq := fun x : ℝ => x^2 + (2*n + 1)*x + 6*n - 5
  let result_eq := fun x : ℝ => x^2 + 2*(n + 1)*x + 8*(n - 1)
  ∃ (r₁ r₂ : ℝ), original_eq r₁ = 0 ∧ original_eq r₂ = 0 ∧
    result_eq (⌊r₁⌋) = 0 ∧ result_eq (⌊r₂⌋) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_parts_l3909_390940


namespace NUMINAMATH_CALUDE_binomial_600_600_eq_1_l3909_390963

theorem binomial_600_600_eq_1 : Nat.choose 600 600 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_600_600_eq_1_l3909_390963


namespace NUMINAMATH_CALUDE_system_solution_correctness_l3909_390992

theorem system_solution_correctness :
  let x₁ : ℚ := -4
  let x₂ : ℚ := 3
  let x₃ : ℚ := -1
  (2 * x₁ + 7 * x₂ + 13 * x₃ = 0) ∧
  (3 * x₁ + 14 * x₂ + 12 * x₃ = 18) ∧
  (5 * x₁ + 25 * x₂ + 16 * x₃ = 39) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_correctness_l3909_390992


namespace NUMINAMATH_CALUDE_officers_selection_count_l3909_390959

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  total_members * girls * (boys - 1)

/-- Theorem: The number of ways to choose officers under given conditions is 6300 -/
theorem officers_selection_count :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 6300 := by
  sorry

#eval choose_officers 30 15 15

end NUMINAMATH_CALUDE_officers_selection_count_l3909_390959


namespace NUMINAMATH_CALUDE_abs_neg_five_l3909_390994

theorem abs_neg_five : abs (-5 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_l3909_390994


namespace NUMINAMATH_CALUDE_problem_solution_l3909_390938

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log (abs x)
def g (a : ℝ) (x : ℝ) : ℝ := 1 / (deriv f x) + a * (deriv f x)

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g a x = x + a / x) ∧
  (a > 0 ∧ (∀ x : ℝ, x > 0 → g a x ≥ 2) ∧ ∃ x : ℝ, x > 0 ∧ g a x = 2) →
  (a = 1 ∧
   (∫ x in (3/2)..(2), (2/3 * x + 7/6) - (x + 1/x)) = 7/24 + Real.log 3 - 2 * Real.log 2) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3909_390938


namespace NUMINAMATH_CALUDE_car_max_acceleration_l3909_390945

theorem car_max_acceleration
  (g : ℝ) -- acceleration due to gravity
  (θ : ℝ) -- angle of the hill
  (μ : ℝ) -- coefficient of static friction
  (h1 : 0 < g)
  (h2 : 0 ≤ θ)
  (h3 : θ < π / 2)
  (h4 : μ > Real.tan θ) :
  ∃ a : ℝ,
    a = g * (μ * Real.cos θ - Real.sin θ) ∧
    ∀ a' : ℝ,
      (∃ m : ℝ, 0 < m ∧
        m * a' ≤ μ * (m * g * Real.cos θ) - m * g * Real.sin θ) →
      a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_car_max_acceleration_l3909_390945


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3909_390910

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
  to_bits n

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, true, true]               -- 111₂
  let result := [true, false, false, false, false, true, false, true]  -- 10000101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

#eval binary_to_nat [true, true, false, true, true]  -- Should output 27
#eval binary_to_nat [true, true, true]               -- Should output 7
#eval binary_to_nat [true, false, false, false, false, true, false, true]  -- Should output 133
#eval 27 * 7  -- Should output 189

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3909_390910


namespace NUMINAMATH_CALUDE_original_triangle_area_l3909_390946

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet,
    prove that the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 256 → -- area of the new triangle
  new = original * 16 → -- relationship between new and original areas
  original = 16 := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3909_390946


namespace NUMINAMATH_CALUDE_total_spider_legs_l3909_390997

/-- The number of spiders in Ivy's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Ivy's room is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l3909_390997


namespace NUMINAMATH_CALUDE_inequality_proof_l3909_390967

theorem inequality_proof (x y z : ℝ) 
  (h1 : x + 2*y + 4*z ≥ 3) 
  (h2 : y - 3*x + 2*z ≥ 5) : 
  y - x + 2*z ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3909_390967


namespace NUMINAMATH_CALUDE_value_of_expression_l3909_390920

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 2 * x + y - z = 7)
  (eq2 : x + 2 * y + z = 5)
  (eq3 : x - y + 2 * z = 3) :
  2 * x * y / 3 = 1.625 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3909_390920
