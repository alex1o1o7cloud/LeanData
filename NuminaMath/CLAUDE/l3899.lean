import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_value_l3899_389987

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/5) : Real.tan α = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3899_389987


namespace NUMINAMATH_CALUDE_dennis_rocks_l3899_389999

theorem dennis_rocks (initial_rocks : ℕ) : 
  (initial_rocks / 2 + 2 = 7) → initial_rocks = 10 := by
sorry

end NUMINAMATH_CALUDE_dennis_rocks_l3899_389999


namespace NUMINAMATH_CALUDE_equidistant_line_slope_l3899_389954

-- Define the points P and Q
def P : ℝ × ℝ := (4, 6)
def Q : ℝ × ℝ := (6, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem equidistant_line_slope :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = m * x → 
      (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2) ∧
    m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_slope_l3899_389954


namespace NUMINAMATH_CALUDE_gwen_recycled_amount_l3899_389970

-- Define the recycling rate
def recycling_rate : ℕ := 3

-- Define the points earned
def points_earned : ℕ := 6

-- Define the amount recycled by friends
def friends_recycled : ℕ := 13

-- Theorem to prove
theorem gwen_recycled_amount : 
  ∃ (gwen_amount : ℕ), 
    (gwen_amount + friends_recycled) / recycling_rate = points_earned ∧
    gwen_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycled_amount_l3899_389970


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3899_389935

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3899_389935


namespace NUMINAMATH_CALUDE_intersection_of_modified_functions_l3899_389983

/-- Two functions that intersect at specific points -/
def IntersectingFunctions (p q : ℝ → ℝ) : Prop :=
  p 1 = q 1 ∧ p 1 = 1 ∧
  p 3 = q 3 ∧ p 3 = 3 ∧
  p 5 = q 5 ∧ p 5 = 5 ∧
  p 7 = q 7 ∧ p 7 = 7

/-- Theorem stating that given two functions p and q that intersect at specific points,
    the functions p(2x) and 2q(x) must intersect at (3.5, 7) -/
theorem intersection_of_modified_functions (p q : ℝ → ℝ) 
    (h : IntersectingFunctions p q) : 
    p (2 * 3.5) = 2 * q 3.5 ∧ p (2 * 3.5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_modified_functions_l3899_389983


namespace NUMINAMATH_CALUDE_solve_for_a_l3899_389980

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : x = 0.3) 
  (h2 : (a * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : a = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3899_389980


namespace NUMINAMATH_CALUDE_rectangular_prism_to_cube_l3899_389916

theorem rectangular_prism_to_cube (a b c : ℝ) (h1 : a = 8) (h2 : b = 8) (h3 : c = 27) :
  ∃ s : ℝ, s^3 = a * b * c ∧ s = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_to_cube_l3899_389916


namespace NUMINAMATH_CALUDE_july_capsule_intake_l3899_389944

/-- Calculates the number of capsules taken in a month given the total days and missed days -/
def capsules_taken (total_days : ℕ) (missed_days : ℕ) : ℕ :=
  total_days - missed_days

/-- Theorem stating that for a 31-day month with 3 missed days, 28 capsules are taken -/
theorem july_capsule_intake : capsules_taken 31 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_july_capsule_intake_l3899_389944


namespace NUMINAMATH_CALUDE_businessmen_beverage_problem_l3899_389905

theorem businessmen_beverage_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea coffee_soda tea_soda : ℕ) (all_three : ℕ) 
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_soda : soda = 8)
  (h_coffee_tea : coffee_tea = 6)
  (h_coffee_soda : coffee_soda = 2)
  (h_tea_soda : tea_soda = 3)
  (h_all_three : all_three = 1) : 
  total - (coffee + tea + soda - coffee_tea - coffee_soda - tea_soda + all_three) = 4 := by
sorry

end NUMINAMATH_CALUDE_businessmen_beverage_problem_l3899_389905


namespace NUMINAMATH_CALUDE_dog_weight_gain_l3899_389929

/-- Given a golden retriever that:
    - Is 8 years old
    - Currently weighs 88 pounds
    - Weighed 40 pounds at 1 year old
    Prove that the average yearly weight gain is 6 pounds. -/
theorem dog_weight_gain (current_weight : ℕ) (age : ℕ) (initial_weight : ℕ) 
  (h1 : current_weight = 88)
  (h2 : age = 8)
  (h3 : initial_weight = 40) :
  (current_weight - initial_weight) / (age - 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_dog_weight_gain_l3899_389929


namespace NUMINAMATH_CALUDE_ratio_x_to_2y_l3899_389926

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 5 * y) / (x - 2 * y) = 26) : 
  x / (2 * y) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_2y_l3899_389926


namespace NUMINAMATH_CALUDE_solution_mixture_percentage_l3899_389934

/-- Proves that in a mixture of solutions X and Y, where X is 40% chemical A and Y is 50% chemical A,
    if the final mixture is 47% chemical A, then the percentage of solution X in the mixture is 30%. -/
theorem solution_mixture_percentage (x y : ℝ) :
  x + y = 100 →
  0.40 * x + 0.50 * y = 47 →
  x = 30 := by sorry

end NUMINAMATH_CALUDE_solution_mixture_percentage_l3899_389934


namespace NUMINAMATH_CALUDE_percentage_70_79_is_800_27_l3899_389985

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution where
  score_90_100 : Nat
  score_80_89 : Nat
  score_70_79 : Nat
  score_60_69 : Nat
  score_below_60 : Nat

/-- Calculates the percentage of students in the 70%-79% range -/
def percentage_70_79 (dist : ScoreDistribution) : Rat :=
  let total := dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + dist.score_60_69 + dist.score_below_60
  (dist.score_70_79 : Rat) / total * 100

/-- The given frequency distribution -/
def history_class_distribution : ScoreDistribution :=
  { score_90_100 := 5
    score_80_89 := 7
    score_70_79 := 8
    score_60_69 := 4
    score_below_60 := 3 }

theorem percentage_70_79_is_800_27 :
  percentage_70_79 history_class_distribution = 800 / 27 := by
  sorry

end NUMINAMATH_CALUDE_percentage_70_79_is_800_27_l3899_389985


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l3899_389971

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element in a row of Pascal's triangle -/
def fifthElement (row : ℕ) : ℕ := binomial row 4

theorem fifth_element_row_20 : fifthElement 20 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l3899_389971


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3899_389956

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    pairs.card = count ∧
    count = 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3899_389956


namespace NUMINAMATH_CALUDE_power_equality_l3899_389955

theorem power_equality : 5^29 * 4^15 = 2 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3899_389955


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3899_389984

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / 15 → m ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3899_389984


namespace NUMINAMATH_CALUDE_students_liking_both_channels_l3899_389928

theorem students_liking_both_channels
  (total : ℕ)
  (sports : ℕ)
  (arts : ℕ)
  (neither : ℕ)
  (h1 : total = 100)
  (h2 : sports = 68)
  (h3 : arts = 55)
  (h4 : neither = 3)
  : (sports + arts) - (total - neither) = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_channels_l3899_389928


namespace NUMINAMATH_CALUDE_snow_probability_l3899_389931

theorem snow_probability (p1 p2 : ℝ) (h1 : p1 = 1/4) (h2 : p2 = 1/3) :
  let prob_no_snow := (1 - p1)^4 * (1 - p2)^3
  1 - prob_no_snow = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l3899_389931


namespace NUMINAMATH_CALUDE_rotated_square_base_vertex_on_line_l3899_389918

/-- Represents a square with side length 2 inches -/
structure Square :=
  (side : ℝ)
  (is_two_inch : side = 2)

/-- Represents the configuration of three squares -/
structure SquareConfiguration :=
  (left : Square)
  (center : Square)
  (right : Square)
  (rotation_angle : ℝ)
  (is_thirty_degrees : rotation_angle = π / 6)

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The base vertex of the rotated square after lowering -/
def base_vertex (config : SquareConfiguration) : Point :=
  sorry

theorem rotated_square_base_vertex_on_line (config : SquareConfiguration) :
  (base_vertex config).y = 0 := by
  sorry

end NUMINAMATH_CALUDE_rotated_square_base_vertex_on_line_l3899_389918


namespace NUMINAMATH_CALUDE_total_questions_is_60_l3899_389917

/-- Represents the citizenship test study problem --/
def CitizenshipTestStudy : Prop :=
  let multipleChoice : ℕ := 30
  let fillInBlank : ℕ := 30
  let multipleChoiceTime : ℕ := 15
  let fillInBlankTime : ℕ := 25
  let totalStudyTime : ℕ := 20 * 60

  (multipleChoice * multipleChoiceTime + fillInBlank * fillInBlankTime = totalStudyTime) →
  (multipleChoice + fillInBlank = 60)

/-- Theorem stating that the total number of questions on the test is 60 --/
theorem total_questions_is_60 : CitizenshipTestStudy := by
  sorry

end NUMINAMATH_CALUDE_total_questions_is_60_l3899_389917


namespace NUMINAMATH_CALUDE_sum_of_eight_five_to_eight_l3899_389995

theorem sum_of_eight_five_to_eight (n : ℕ) :
  (Finset.range 8).sum (λ _ => 5^8) = 3125000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_five_to_eight_l3899_389995


namespace NUMINAMATH_CALUDE_min_vector_difference_l3899_389978

/-- Given planar vectors a, b, c satisfying the conditions, 
    the minimum value of |a - b| is 6 -/
theorem min_vector_difference (a b c : ℝ × ℝ) 
    (h1 : a • b = 0)
    (h2 : ‖c‖ = 1)
    (h3 : ‖a - c‖ = 5)
    (h4 : ‖b - c‖ = 5) :
    6 ≤ ‖a - b‖ ∧ ∃ (a' b' c' : ℝ × ℝ), 
      a' • b' = 0 ∧ 
      ‖c'‖ = 1 ∧ 
      ‖a' - c'‖ = 5 ∧ 
      ‖b' - c'‖ = 5 ∧
      ‖a' - b'‖ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_vector_difference_l3899_389978


namespace NUMINAMATH_CALUDE_max_area_enclosure_l3899_389998

/-- Represents a rectangular enclosure. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is exactly 420 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 420

/-- The length of the enclosure is at least 100 feet. -/
def lengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def widthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- The theorem stating that the maximum area is achieved when length = width = 105 feet. -/
theorem max_area_enclosure :
  ∀ e : Enclosure,
    perimeterConstraint e → lengthConstraint e → widthConstraint e →
    area e ≤ 11025 ∧
    (area e = 11025 ↔ e.length = 105 ∧ e.width = 105) :=
by sorry

end NUMINAMATH_CALUDE_max_area_enclosure_l3899_389998


namespace NUMINAMATH_CALUDE_range_of_m_l3899_389911

def A : Set ℝ := {y | ∃ x ∈ Set.Icc (3/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

theorem range_of_m :
  {m : ℝ | A ⊆ B m} = Set.Iic (-3/4) ∪ Set.Ici (3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3899_389911


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l3899_389996

theorem factorization_cubic_minus_linear_times_square (a b : ℝ) : 
  a^3 - a*b^2 = a*(a+b)*(a-b) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l3899_389996


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3899_389904

theorem geometric_sequence_sum (n : ℕ) :
  let a : ℝ := 1
  let r : ℝ := 1/2
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 31/16 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3899_389904


namespace NUMINAMATH_CALUDE_total_remaining_candle_life_l3899_389933

/-- Calculates the total remaining candle life in a house given the number of candles and their remaining life percentages in different rooms. -/
theorem total_remaining_candle_life
  (bedroom_candles : ℕ)
  (bedroom_life : ℚ)
  (living_room_candles : ℕ)
  (living_room_life : ℚ)
  (hallway_candles : ℕ)
  (hallway_life : ℚ)
  (study_room_life : ℚ)
  (h1 : bedroom_candles = 20)
  (h2 : living_room_candles = bedroom_candles / 2)
  (h3 : hallway_candles = 20)
  (h4 : bedroom_life = 60 / 100)
  (h5 : living_room_life = 80 / 100)
  (h6 : hallway_life = 50 / 100)
  (h7 : study_room_life = 70 / 100) :
  let study_room_candles := bedroom_candles + living_room_candles + 5
  (bedroom_candles : ℚ) * bedroom_life +
  (living_room_candles : ℚ) * living_room_life +
  (hallway_candles : ℚ) * hallway_life +
  (study_room_candles : ℚ) * study_room_life = 54.5 :=
sorry

end NUMINAMATH_CALUDE_total_remaining_candle_life_l3899_389933


namespace NUMINAMATH_CALUDE_function_always_one_l3899_389921

theorem function_always_one (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, n > 0 → f (n + f n) = f n)
  (h2 : ∃ n₀ : ℕ, f n₀ = 1) : 
  ∀ n : ℕ, f n = 1 := by
sorry

end NUMINAMATH_CALUDE_function_always_one_l3899_389921


namespace NUMINAMATH_CALUDE_OPRQ_shape_l3899_389963

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.P.x - quad.O.x, quad.P.y - quad.O.y) = (quad.R.x - quad.Q.x, quad.R.y - quad.Q.y) ∧
  (quad.Q.x - quad.O.x, quad.Q.y - quad.O.y) = (quad.R.x - quad.P.x, quad.R.y - quad.P.y)

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Checks if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  areCollinear quad.O quad.P quad.Q ∧ areCollinear quad.O quad.Q quad.R

/-- Checks if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  ((quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) = (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y) ∧
   (quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) ≠ (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y)) ∨
  ((quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) = (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y) ∧
   (quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) ≠ (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y))

theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : y₁ ≠ y₂) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk (x₁ + 2*x₂) (y₁ + 2*y₂))
    (Point.mk x₂ y₂)
  ¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ (isTrapezoid quad ∨ (¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ ¬(isTrapezoid quad))) := by
  sorry

end NUMINAMATH_CALUDE_OPRQ_shape_l3899_389963


namespace NUMINAMATH_CALUDE_liangliang_speed_l3899_389967

/-- The walking speeds of Mingming and Liangliang -/
structure WalkingSpeeds where
  mingming : ℝ
  liangliang : ℝ

/-- The initial and final distances between Mingming and Liangliang -/
structure Distances where
  initial : ℝ
  final : ℝ

/-- The time elapsed between the initial and final measurements -/
def elapsedTime : ℝ := 20

/-- The theorem stating the possible walking speeds of Liangliang -/
theorem liangliang_speed 
  (speeds : WalkingSpeeds) 
  (distances : Distances) 
  (h1 : speeds.mingming = 80) 
  (h2 : distances.initial = 3000) 
  (h3 : distances.final = 2900) :
  speeds.liangliang = 85 ∨ speeds.liangliang = 75 :=
sorry

end NUMINAMATH_CALUDE_liangliang_speed_l3899_389967


namespace NUMINAMATH_CALUDE_exists_non_complementary_acute_angles_l3899_389981

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- Define what complementary angles are
def are_complementary (angle1 angle2 : ℝ) : Prop := angle1 + angle2 = 90

-- Theorem statement
theorem exists_non_complementary_acute_angles :
  ∃ (angle1 angle2 : ℝ), is_acute_angle angle1 ∧ is_acute_angle angle2 ∧ ¬(are_complementary angle1 angle2) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_complementary_acute_angles_l3899_389981


namespace NUMINAMATH_CALUDE_julian_frederick_age_difference_l3899_389932

/-- Given the ages of Kyle, Julian, Frederick, and Tyson, prove that Julian is 20 years younger than Frederick. -/
theorem julian_frederick_age_difference :
  ∀ (kyle_age julian_age frederick_age tyson_age : ℕ),
  kyle_age = julian_age + 5 →
  frederick_age > julian_age →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 →
  kyle_age = 25 →
  frederick_age - julian_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_julian_frederick_age_difference_l3899_389932


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3899_389919

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def possible_roots : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (y : ℤ), polynomial b₂ b₁ x = 0} = possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3899_389919


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3899_389977

theorem polynomial_factorization (u : ℝ) : 
  u^4 - 81*u^2 + 144 = (u^2 - 72)*(u - 3)*(u + 3) := by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l3899_389977


namespace NUMINAMATH_CALUDE_line_intercepts_opposite_l3899_389948

/-- A line with equation (a-2)x + y - a = 0 has intercepts on the coordinate axes that are opposite numbers if and only if a = 0 or a = 1 -/
theorem line_intercepts_opposite (a : ℝ) : 
  (∃ x y : ℝ, (a - 2) * x + y - a = 0 ∧ 
   ((x = 0 ∧ y ≠ 0) ∨ (x ≠ 0 ∧ y = 0)) ∧
   (x = 0 → y = a) ∧
   (y = 0 → x = a / (a - 2)) ∧
   x = -y) ↔ 
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_opposite_l3899_389948


namespace NUMINAMATH_CALUDE_negation_of_forall_abs_sum_nonnegative_l3899_389989

theorem negation_of_forall_abs_sum_nonnegative :
  (¬ (∀ x : ℝ, x + |x| ≥ 0)) ↔ (∃ x : ℝ, x + |x| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_abs_sum_nonnegative_l3899_389989


namespace NUMINAMATH_CALUDE_square_sum_digits_l3899_389937

theorem square_sum_digits (n : ℕ) : 
  let A := 4 * (10^(2*n) - 1) / 9
  let B := 8 * (10^n - 1) / 9
  A + 2*B + 4 = (2*(10^n + 2)/3)^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_digits_l3899_389937


namespace NUMINAMATH_CALUDE_total_ladybugs_count_l3899_389979

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_count_l3899_389979


namespace NUMINAMATH_CALUDE_rectangle_area_l3899_389908

/-- 
Given a rectangle with length l and width w, 
if the length is four times the width and the perimeter is 200,
then the area of the rectangle is 1600.
-/
theorem rectangle_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3899_389908


namespace NUMINAMATH_CALUDE_sin_negative_1665_degrees_l3899_389909

theorem sin_negative_1665_degrees :
  Real.sin ((-1665 : ℝ) * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1665_degrees_l3899_389909


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l3899_389960

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 36) :
  x^3 * y^4 ≤ 18^3 * 6^4 ∧ (x^3 * y^4 = 18^3 * 6^4 ↔ x = 18 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l3899_389960


namespace NUMINAMATH_CALUDE_fraction_inequality_l3899_389901

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3899_389901


namespace NUMINAMATH_CALUDE_gcd_20244_46656_l3899_389910

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20244_46656_l3899_389910


namespace NUMINAMATH_CALUDE_gabby_shopping_funds_l3899_389939

theorem gabby_shopping_funds (total_cost available_funds : ℕ) : 
  total_cost = 165 → available_funds = 110 → total_cost - available_funds = 55 := by
  sorry

end NUMINAMATH_CALUDE_gabby_shopping_funds_l3899_389939


namespace NUMINAMATH_CALUDE_smallest_ten_digit_divisible_by_first_five_primes_l3899_389982

/-- The product of the first five prime numbers -/
def first_five_primes_product : ℕ := 2 * 3 * 5 * 7 * 11

/-- A number is a 10-digit number if it's between 1000000000 and 9999999999 -/
def is_ten_digit (n : ℕ) : Prop := 1000000000 ≤ n ∧ n ≤ 9999999999

theorem smallest_ten_digit_divisible_by_first_five_primes :
  ∀ n : ℕ, is_ten_digit n ∧ n % first_five_primes_product = 0 → n ≥ 1000000310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ten_digit_divisible_by_first_five_primes_l3899_389982


namespace NUMINAMATH_CALUDE_triangle_similarity_BL_calculation_l3899_389940

theorem triangle_similarity_BL_calculation (AD BC AL BL LD LC : ℝ) 
  (h_similar : AD / BC = AL / BL ∧ AL / BL = LD / LC) :
  (∀ AB BD : ℝ, 
    (AB = 6 * Real.sqrt 13 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 3) → 
    BL = 16 * Real.sqrt 3 - 12) ∧
  (∀ AB BD : ℝ, 
    (AB = 30 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 6) → 
    BL = (16 * Real.sqrt 6 - 6) / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_BL_calculation_l3899_389940


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_triple_work_time_l3899_389915

/-- If a person can complete a piece of work in a given number of days,
    then the time required to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional
  (days_for_single_work : ℕ) (work_multiple : ℕ) :
  let days_for_multiple_work := days_for_single_work * work_multiple
  days_for_multiple_work = days_for_single_work * work_multiple :=
by sorry

/-- Aarti's work completion time for triple work -/
theorem aarti_triple_work_time :
  let days_for_single_work := 6
  let work_multiple := 3
  let days_for_triple_work := days_for_single_work * work_multiple
  days_for_triple_work = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_triple_work_time_l3899_389915


namespace NUMINAMATH_CALUDE_sum_product_range_l3899_389992

theorem sum_product_range (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x : ℝ, x ≤ 0 → ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_range_l3899_389992


namespace NUMINAMATH_CALUDE_cordelia_hair_dyeing_l3899_389925

/-- Cordelia's hair dyeing problem -/
theorem cordelia_hair_dyeing (total_time bleach_time dye_time : ℝ) : 
  total_time = 9 ∧ dye_time = 2 * bleach_time ∧ total_time = bleach_time + dye_time → 
  bleach_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_hair_dyeing_l3899_389925


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l3899_389950

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l3899_389950


namespace NUMINAMATH_CALUDE_jazel_sticks_total_length_l3899_389903

/-- Given Jazel's three sticks with specified lengths, prove that their total length is 14 centimeters. -/
theorem jazel_sticks_total_length :
  let first_stick : ℕ := 3
  let second_stick : ℕ := 2 * first_stick
  let third_stick : ℕ := second_stick - 1
  first_stick + second_stick + third_stick = 14 := by
  sorry

end NUMINAMATH_CALUDE_jazel_sticks_total_length_l3899_389903


namespace NUMINAMATH_CALUDE_john_tax_rate_l3899_389964

/-- Given the number of shirts, price per shirt, and total payment including tax,
    calculate the tax rate as a percentage. -/
def calculate_tax_rate (num_shirts : ℕ) (price_per_shirt : ℚ) (total_payment : ℚ) : ℚ :=
  let cost_before_tax := num_shirts * price_per_shirt
  let tax_amount := total_payment - cost_before_tax
  (tax_amount / cost_before_tax) * 100

/-- Theorem stating that for 3 shirts at $20 each and a total payment of $66,
    the tax rate is 10%. -/
theorem john_tax_rate :
  calculate_tax_rate 3 20 66 = 10 := by
  sorry

#eval calculate_tax_rate 3 20 66

end NUMINAMATH_CALUDE_john_tax_rate_l3899_389964


namespace NUMINAMATH_CALUDE_problem_solution_l3899_389976

theorem problem_solution (m n x y : ℝ) 
  (h1 : m - n = 8) 
  (h2 : x + y = 1) : 
  (n + x) - (m - y) = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3899_389976


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3899_389986

theorem fraction_sum_equality (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3899_389986


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389936

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389936


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l3899_389900

/-- Proves that for points on an inverse proportion function, 
    if x₁ < 0 < x₂, then y₁ < y₂ -/
theorem inverse_proportion_ordering (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l3899_389900


namespace NUMINAMATH_CALUDE_probability_greater_than_three_l3899_389951

-- Define a standard die
def StandardDie : ℕ := 6

-- Define the favorable outcomes (numbers greater than 3)
def FavorableOutcomes : ℕ := 3

-- Theorem statement
theorem probability_greater_than_three (d : ℕ) (h : d = StandardDie) : 
  (FavorableOutcomes : ℚ) / d = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_greater_than_three_l3899_389951


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3899_389953

theorem solution_set_inequality (x : ℝ) : 
  (3 - 2*x) * (x + 1) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3899_389953


namespace NUMINAMATH_CALUDE_compute_expression_l3899_389968

theorem compute_expression : 20 * (144 / 3 + 36 / 6 + 16 / 32 + 2) = 1130 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3899_389968


namespace NUMINAMATH_CALUDE_range_of_sum_reciprocals_l3899_389930

theorem range_of_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_reciprocals_l3899_389930


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3899_389991

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 48 → Nat.gcd n 12 = 8 → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3899_389991


namespace NUMINAMATH_CALUDE_linear_function_k_value_l3899_389961

/-- Given a linear function y = kx + 2 passing through the point (-2, -1), prove that k = 3/2 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2) →  -- Linear function condition
  (-1 : ℝ) = k * (-2 : ℝ) + 2 →  -- Point (-2, -1) condition
  k = 3/2 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l3899_389961


namespace NUMINAMATH_CALUDE_chord_equation_l3899_389927

/-- Given a parabola and a chord, prove the equation of the line containing the chord -/
theorem chord_equation (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 = -2*y₁) →  -- Point A on parabola
  (x₂^2 = -2*y₂) →  -- Point B on parabola
  (x₁ + x₂ = -2) →  -- Sum of x-coordinates
  ((x₁ + x₂)/2 = -1) →  -- x-coordinate of midpoint
  ((y₁ + y₂)/2 = -5) →  -- y-coordinate of midpoint
  ∃ (m b : ℝ), ∀ x y, (y = m*x + b) ↔ (y - y₁)*(x₂ - x₁) = (x - x₁)*(y₂ - y₁) ∧ m = 1 ∧ b = -4 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l3899_389927


namespace NUMINAMATH_CALUDE_infinite_triplets_existence_l3899_389922

theorem infinite_triplets_existence : ∀ n : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
  0 < p ∧ p ≤ 2 * n^2 ∧ 
  |p * Real.sqrt 2 - q₁| * |p * Real.sqrt 3 - q₂| ≤ 1 / (4 * ↑n^2) :=
sorry

end NUMINAMATH_CALUDE_infinite_triplets_existence_l3899_389922


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_2005_2006_l3899_389906

theorem consecutive_odd_sum_2005_2006 :
  (∃ (n k : ℕ), n ≥ 2 ∧ 2005 = n * (2 * k + n)) ∧
  (¬ ∃ (n k : ℕ), n ≥ 2 ∧ 2006 = n * (2 * k + n)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_2005_2006_l3899_389906


namespace NUMINAMATH_CALUDE_expression_value_l3899_389949

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4*x*z + y*w = 4) 
  (eq2 : x*w + y*z = 8) : 
  (2*x + y) * (2*z + w) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3899_389949


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3899_389920

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) :
  ∃ ζ : ℂ, ζ^3 = 1 ∧ ζ ≠ 1 ∧ (a^9 + b^9) / (a - b)^9 = 2 / (81 * (ζ - 1)) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3899_389920


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_neg_nineteen_fifths_l3899_389942

theorem greatest_integer_less_than_neg_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_neg_nineteen_fifths_l3899_389942


namespace NUMINAMATH_CALUDE_park_area_l3899_389962

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 15
  perimeter_eq : 2 * (width + length) = 800

/-- The area of a rectangular park with the given properties is 29234.375 square feet -/
theorem park_area (park : RectangularPark) : park.width * park.length = 29234.375 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l3899_389962


namespace NUMINAMATH_CALUDE_compute_expression_l3899_389997

theorem compute_expression : 25 * (216 / 3 + 49 / 7 + 16 / 25 + 2) = 2041 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3899_389997


namespace NUMINAMATH_CALUDE_water_tank_emptying_time_l3899_389943

/-- Represents a water tank with constant inflow and three identical taps. -/
structure WaterTank where
  /-- Time to empty the tank with one tap open (in hours) -/
  one_tap_time : ℝ
  /-- Time to empty the tank with two taps open (in hours) -/
  two_tap_time : ℝ
  /-- Assumption that one_tap_time is positive -/
  one_tap_positive : one_tap_time > 0
  /-- Assumption that two_tap_time is positive -/
  two_tap_positive : two_tap_time > 0
  /-- Assumption that one_tap_time is greater than two_tap_time -/
  time_order : one_tap_time > two_tap_time

/-- 
Given a water tank with constant inflow and three identical taps, 
if it takes 9 hours to empty with one tap open and 3 hours with two taps open, 
then it will take 9/5 hours to empty with all three taps open.
-/
theorem water_tank_emptying_time (tank : WaterTank) 
  (h1 : tank.one_tap_time = 9) 
  (h2 : tank.two_tap_time = 3) : 
  ∃ (three_tap_time : ℝ), three_tap_time = 9/5 ∧ 
  (1 + three_tap_time / tank.one_tap_time = 2 / tank.one_tap_time * 3 * three_tap_time) :=
sorry

end NUMINAMATH_CALUDE_water_tank_emptying_time_l3899_389943


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3899_389959

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 2) * (x - 2) + 3 * (1 - x) = 1 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3899_389959


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389975

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389975


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3899_389993

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 (a > 0) and eccentricity √3,
    prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) :
  let hyperbola := λ (x y : ℝ) => x^2 / a^2 - y^2 / 2 = 1
  let eccentricity := Real.sqrt 3
  let asymptotes := λ (x y : ℝ) => y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x
  ∀ (x y : ℝ), hyperbola x y ∧ eccentricity = Real.sqrt 3 → asymptotes x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3899_389993


namespace NUMINAMATH_CALUDE_llama_cost_increase_l3899_389988

/-- Proves that the percentage increase in the cost of each llama compared to each goat is 50% -/
theorem llama_cost_increase (goat_cost : ℝ) (total_cost : ℝ) : 
  goat_cost = 400 →
  total_cost = 4800 →
  let num_goats : ℕ := 3
  let num_llamas : ℕ := 2 * num_goats
  let total_goat_cost : ℝ := goat_cost * num_goats
  let total_llama_cost : ℝ := total_cost - total_goat_cost
  let llama_cost : ℝ := total_llama_cost / num_llamas
  (llama_cost - goat_cost) / goat_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_llama_cost_increase_l3899_389988


namespace NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l3899_389923

-- Define sets A and B
variable (A B : Type)

-- Define a mapping from A to B
variable (f : A → B)

-- Theorem stating that it's possible for two different elements in A to have the same image in B
theorem mapping_not_necessarily_injective :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

end NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l3899_389923


namespace NUMINAMATH_CALUDE_b_range_l3899_389945

theorem b_range (a b : ℝ) (h1 : a * b^2 > a) (h2 : a > a * b) (h3 : a ≠ 0) : b < -1 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l3899_389945


namespace NUMINAMATH_CALUDE_line_parameterization_l3899_389965

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 7

/-- The parametric equation of the line -/
def parametric_equation (s n t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = -3 + n * t

/-- The theorem stating the values of s and n -/
theorem line_parameterization :
  ∃ (s n : ℝ), (∀ (t x y : ℝ), parametric_equation s n t x y → line_equation x y) ∧ s = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3899_389965


namespace NUMINAMATH_CALUDE_system_solution_l3899_389941

theorem system_solution : 
  ∀ (x y z t : ℕ), 
    x + y = z * t ∧ z + t = x * y → 
      ((x = 1 ∧ y = 5 ∧ z = 2 ∧ t = 3) ∨ 
       (x = 5 ∧ y = 1 ∧ z = 3 ∧ t = 2) ∨ 
       (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2)) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3899_389941


namespace NUMINAMATH_CALUDE_simplify_and_multiply_l3899_389907

theorem simplify_and_multiply (b : ℝ) : (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_multiply_l3899_389907


namespace NUMINAMATH_CALUDE_unique_four_digit_consecutive_square_swap_l3899_389974

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ x : ℕ, x ≤ 6 ∧ 
    n = 1000 * x + 100 * (x + 1) + 10 * (x + 2) + (x + 3)

def swap_thousands_hundreds (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let ones := n % 10
  1000 * hundreds + 100 * thousands + 10 * tens + ones

theorem unique_four_digit_consecutive_square_swap :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_consecutive_digits n ∧
    ∃ m : ℕ, swap_thousands_hundreds n = m * m :=
by
  use 3456
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_consecutive_square_swap_l3899_389974


namespace NUMINAMATH_CALUDE_test_maximum_marks_l3899_389990

theorem test_maximum_marks :
  let passing_percentage : ℚ := 60 / 100
  let student_score : ℕ := 80
  let marks_needed_to_pass : ℕ := 100
  let maximum_marks : ℕ := 300
  passing_percentage * maximum_marks = student_score + marks_needed_to_pass →
  maximum_marks = 300 := by
sorry

end NUMINAMATH_CALUDE_test_maximum_marks_l3899_389990


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l3899_389938

/-- Represents the problem of finding the minimum number of workers needed for profit --/
theorem min_workers_for_profit :
  let maintenance_fee : ℝ := 470
  let hourly_wage : ℝ := 10
  let widgets_per_hour : ℝ := 6
  let widget_price : ℝ := 3.5
  let work_hours : ℝ := 8
  let min_workers : ℕ := 6

  ∀ n : ℕ, n ≥ min_workers →
    work_hours * widgets_per_hour * widget_price * n > maintenance_fee + work_hours * hourly_wage * n ∧
    ∀ m : ℕ, m < min_workers →
      work_hours * widgets_per_hour * widget_price * m ≤ maintenance_fee + work_hours * hourly_wage * m :=
by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l3899_389938


namespace NUMINAMATH_CALUDE_factor_expression_l3899_389969

theorem factor_expression (x : ℝ) : 63 * x + 45 = 9 * (7 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3899_389969


namespace NUMINAMATH_CALUDE_prime_divisor_of_3n_minus_1_and_n_minus_10_l3899_389972

theorem prime_divisor_of_3n_minus_1_and_n_minus_10 (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_div_3n_minus_1 : p ∣ (3 * n - 1)) (h_div_n_minus_10 : p ∣ (n - 10)) : p = 29 :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_of_3n_minus_1_and_n_minus_10_l3899_389972


namespace NUMINAMATH_CALUDE_clothes_to_earnings_ratio_l3899_389914

/-- Proves that the ratio of clothes spending to initial earnings is 1:2 given the conditions --/
theorem clothes_to_earnings_ratio 
  (initial_earnings : ℚ)
  (clothes_spending : ℚ)
  (book_spending : ℚ)
  (remaining : ℚ)
  (h1 : initial_earnings = 600)
  (h2 : book_spending = (initial_earnings - clothes_spending) / 2)
  (h3 : remaining = initial_earnings - clothes_spending - book_spending)
  (h4 : remaining = 150) :
  clothes_spending / initial_earnings = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_clothes_to_earnings_ratio_l3899_389914


namespace NUMINAMATH_CALUDE_slow_car_speed_is_correct_l3899_389966

/-- The speed of the slow car in km/h -/
def slow_car_speed : ℝ := 40

/-- The speed of the fast car in km/h -/
def fast_car_speed : ℝ := 1.5 * slow_car_speed

/-- The distance to the memorial hall in km -/
def distance : ℝ := 60

/-- The time difference between departures in hours -/
def time_difference : ℝ := 0.5

theorem slow_car_speed_is_correct :
  (distance / slow_car_speed) - (distance / fast_car_speed) = time_difference :=
sorry

end NUMINAMATH_CALUDE_slow_car_speed_is_correct_l3899_389966


namespace NUMINAMATH_CALUDE_vacuum_time_per_room_l3899_389957

-- Define the total vacuuming time in hours
def total_time : ℝ := 2

-- Define the number of rooms
def num_rooms : ℕ := 6

-- Define the function to convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem statement
theorem vacuum_time_per_room : 
  (hours_to_minutes total_time) / num_rooms = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_per_room_l3899_389957


namespace NUMINAMATH_CALUDE_a_less_than_b_l3899_389924

theorem a_less_than_b (x a b : ℝ) (h1 : x > 0) (h2 : a * b ≠ 0) (h3 : a * x < b * x + 1) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l3899_389924


namespace NUMINAMATH_CALUDE_barbara_wins_iff_odd_sum_l3899_389994

/-- Newspaper cutting game -/
def newspaper_game_winner (a b d : ℝ) : Prop :=
  let x := ⌊a / d⌋
  let y := ⌊b / d⌋
  Odd (x + y)

/-- Barbara wins the newspaper cutting game if and only if the sum of the floor divisions is odd -/
theorem barbara_wins_iff_odd_sum (a b d : ℝ) (h : d > 0) :
  newspaper_game_winner a b d ↔ Barbara_wins :=
sorry

end NUMINAMATH_CALUDE_barbara_wins_iff_odd_sum_l3899_389994


namespace NUMINAMATH_CALUDE_fib_sum_39_40_l3899_389947

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem fib_sum_39_40 : fib 39 + fib 40 = fib 41 := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_39_40_l3899_389947


namespace NUMINAMATH_CALUDE_total_pitchers_is_one_and_half_l3899_389973

/-- The total number of pitchers of lemonade served during a school play -/
def total_pitchers (first second third fourth : ℚ) : ℚ :=
  first + second + third + fourth

/-- Theorem stating that the total number of pitchers served is 1.5 -/
theorem total_pitchers_is_one_and_half :
  total_pitchers 0.25 0.4166666666666667 0.25 0.5833333333333334 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_total_pitchers_is_one_and_half_l3899_389973


namespace NUMINAMATH_CALUDE_three_heads_before_four_tails_l3899_389902

/-- The probability of encountering 3 heads before 4 tails in repeated fair coin flips -/
def probability_three_heads_before_four_tails : ℚ := 4/7

/-- A fair coin has equal probability of heads and tails -/
axiom fair_coin : ℚ

/-- The probability of heads for a fair coin is 1/2 -/
axiom fair_coin_probability : fair_coin = 1/2

theorem three_heads_before_four_tails :
  probability_three_heads_before_four_tails = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_three_heads_before_four_tails_l3899_389902


namespace NUMINAMATH_CALUDE_regular_9gon_coloring_l3899_389958

-- Define a regular 9-gon
structure RegularNineGon where
  vertices : Fin 9 → Point

-- Define a coloring of the vertices
inductive Color
| Black
| White

def Coloring := Fin 9 → Color

-- Define adjacency in the 9-gon
def adjacent (i j : Fin 9) : Prop :=
  (i.val + 1) % 9 = j.val ∨ (j.val + 1) % 9 = i.val

-- Define an isosceles triangle in the 9-gon
def isoscelesTriangle (i j k : Fin 9) (polygon : RegularNineGon) : Prop :=
  let d := (i.val - j.val + 9) % 9
  (i.val - k.val + 9) % 9 = d ∨ (j.val - k.val + 9) % 9 = d

theorem regular_9gon_coloring 
  (polygon : RegularNineGon) 
  (coloring : Coloring) : 
  (∃ i j : Fin 9, adjacent i j ∧ coloring i = coloring j) ∧ 
  (∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    coloring i = coloring j ∧ coloring j = coloring k ∧ 
    isoscelesTriangle i j k polygon) :=
by sorry

end NUMINAMATH_CALUDE_regular_9gon_coloring_l3899_389958


namespace NUMINAMATH_CALUDE_maintenance_interval_increase_l3899_389913

/-- The combined percentage increase in maintenance interval when using three additives -/
theorem maintenance_interval_increase (increase_a increase_b increase_c : ℝ) :
  increase_a = 0.2 →
  increase_b = 0.3 →
  increase_c = 0.4 →
  ((1 + increase_a) * (1 + increase_b) * (1 + increase_c) - 1) * 100 = 118.4 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_increase_l3899_389913


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l3899_389952

theorem add_preserves_inequality (a b : ℝ) (h : a < b) : 3 + a < 3 + b := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l3899_389952


namespace NUMINAMATH_CALUDE_race_heartbeats_l3899_389946

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (initial_rate : ℕ) (rate_increase : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  let final_rate := initial_rate + (distance - 1) * rate_increase
  let avg_rate := (initial_rate + final_rate) / 2
  avg_rate * distance * pace

/-- Theorem stating that the total heartbeats for the given conditions is 9750 -/
theorem race_heartbeats :
  total_heartbeats 140 5 10 6 = 9750 := by
  sorry

#eval total_heartbeats 140 5 10 6

end NUMINAMATH_CALUDE_race_heartbeats_l3899_389946


namespace NUMINAMATH_CALUDE_integral_of_derivative_scaled_l3899_389912

theorem integral_of_derivative_scaled (f : ℝ → ℝ) (a b : ℝ) (hf : Differentiable ℝ f) (hab : a < b) :
  ∫ x in a..b, (deriv f (3 * x)) = (1 / 3) * (f (3 * b) - f (3 * a)) := by
  sorry

end NUMINAMATH_CALUDE_integral_of_derivative_scaled_l3899_389912
