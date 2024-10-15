import Mathlib

namespace NUMINAMATH_CALUDE_mrs_awesome_class_size_l3663_366323

theorem mrs_awesome_class_size :
  ∀ (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) (boy_girl_difference : ℕ),
    total_jelly_beans = 480 →
    leftover_jelly_beans = 5 →
    boy_girl_difference = 3 →
    ∃ (girls : ℕ) (boys : ℕ),
      girls + boys = 31 ∧
      boys = girls + boy_girl_difference ∧
      girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_mrs_awesome_class_size_l3663_366323


namespace NUMINAMATH_CALUDE_cake_volume_and_icing_sum_l3663_366328

/-- Represents a point in 3D space -/
structure Point3D where
  x : Real
  y : Real
  z : Real

/-- Represents a triangular piece of cake -/
structure CakePiece where
  corner : Point3D
  midpoint1 : Point3D
  midpoint2 : Point3D

/-- Calculates the volume of the triangular cake piece -/
def volume (piece : CakePiece) : Real :=
  sorry

/-- Calculates the area of icing on the triangular cake piece -/
def icingArea (piece : CakePiece) : Real :=
  sorry

/-- The main theorem to prove -/
theorem cake_volume_and_icing_sum (cubeEdgeLength : Real) (piece : CakePiece) : 
  cubeEdgeLength = 3 →
  piece.corner = ⟨0, 0, 0⟩ →
  piece.midpoint1 = ⟨3, 3, 1.5⟩ →
  piece.midpoint2 = ⟨1.5, 3, 3⟩ →
  volume piece + icingArea piece = 24 :=
sorry

end NUMINAMATH_CALUDE_cake_volume_and_icing_sum_l3663_366328


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_equals_half_l3663_366338

theorem x_plus_2y_squared_equals_half (x y : ℝ) 
  (h : 8*y^4 + 4*x^2*y^2 + 4*x*y^2 + 2*x^3 + 2*y^2 + 2*x = x^2 + 1) : 
  x + 2*y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_equals_half_l3663_366338


namespace NUMINAMATH_CALUDE_coeff_x2y2_in_expansion_l3663_366372

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^a * y^b in (1+x)^m * (1+y)^n
def coeff (m n a b : ℕ) : ℕ := binomial m a * binomial n b

-- Theorem statement
theorem coeff_x2y2_in_expansion : coeff 3 4 2 2 = 18 := by sorry

end NUMINAMATH_CALUDE_coeff_x2y2_in_expansion_l3663_366372


namespace NUMINAMATH_CALUDE_three_balls_four_boxes_l3663_366368

theorem three_balls_four_boxes :
  (∀ n : ℕ, n ≤ 3 → n > 0 → 4 ^ n = (Fintype.card (Fin 4)) ^ n) →
  4 ^ 3 = 64 :=
by sorry

end NUMINAMATH_CALUDE_three_balls_four_boxes_l3663_366368


namespace NUMINAMATH_CALUDE_fraction_factorization_l3663_366311

theorem fraction_factorization (a b c : ℝ) : 
  ((a^3 - b^3)^4 + (b^3 - c^3)^4 + (c^3 - a^3)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_factorization_l3663_366311


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3663_366346

theorem rationalize_denominator :
  7 / (2 * Real.sqrt 50) = (7 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3663_366346


namespace NUMINAMATH_CALUDE_obtuse_triangle_k_range_l3663_366388

/-- An obtuse triangle ABC with sides a = k, b = k + 2, and c = k + 4 -/
structure ObtuseTriangle (k : ℝ) where
  a : ℝ := k
  b : ℝ := k + 2
  c : ℝ := k + 4
  is_obtuse : c^2 > a^2 + b^2

/-- The range of possible values for k in an obtuse triangle with sides k, k+2, k+4 -/
theorem obtuse_triangle_k_range (k : ℝ) :
  (∃ t : ObtuseTriangle k, True) ↔ 2 < k ∧ k < 6 := by
  sorry

#check obtuse_triangle_k_range

end NUMINAMATH_CALUDE_obtuse_triangle_k_range_l3663_366388


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_achievable_l3663_366393

def S : Finset Int := {8, -7, 2, -4, 20}

theorem smallest_sum_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a + b + c ≥ -9 :=
by sorry

theorem smallest_sum_is_achievable :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_achievable_l3663_366393


namespace NUMINAMATH_CALUDE_square_of_1005_l3663_366374

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := by sorry

end NUMINAMATH_CALUDE_square_of_1005_l3663_366374


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3663_366309

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 5) (h2 : b^2 = 64) (h3 : a * b > 0) :
  a - b = 3 ∨ a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3663_366309


namespace NUMINAMATH_CALUDE_exam_marks_theorem_l3663_366312

theorem exam_marks_theorem (T : ℝ) 
  (h1 : 0.40 * T + 40 = 160) 
  (h2 : 0.60 * T - 160 = 20) : True :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_theorem_l3663_366312


namespace NUMINAMATH_CALUDE_find_number_l3663_366339

theorem find_number (x : ℝ) : ((x * 14) / 100) = 0.045374000000000005 → x = 0.3241 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3663_366339


namespace NUMINAMATH_CALUDE_body_temperature_survey_most_suitable_for_census_l3663_366349

/-- Represents a survey option -/
inductive SurveyOption
| HeightSurvey
| TrafficRegulationsSurvey
| BodyTemperatureSurvey
| MovieViewershipSurvey

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresCompleteData : Bool
  impactsSafety : Bool
  populationSize : Nat

/-- Defines what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.requiresCompleteData ∧ c.impactsSafety ∧ c.populationSize > 0

/-- Assigns characteristics to each survey option -/
def getSurveyCharacteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.HeightSurvey => ⟨false, false, 1000⟩
| SurveyOption.TrafficRegulationsSurvey => ⟨false, false, 10000⟩
| SurveyOption.BodyTemperatureSurvey => ⟨true, true, 500⟩
| SurveyOption.MovieViewershipSurvey => ⟨false, false, 2000⟩

theorem body_temperature_survey_most_suitable_for_census :
  suitableForCensus (getSurveyCharacteristics SurveyOption.BodyTemperatureSurvey) ∧
  ∀ (s : SurveyOption), s ≠ SurveyOption.BodyTemperatureSurvey →
    ¬(suitableForCensus (getSurveyCharacteristics s)) :=
  sorry

end NUMINAMATH_CALUDE_body_temperature_survey_most_suitable_for_census_l3663_366349


namespace NUMINAMATH_CALUDE_strawberry_ratio_l3663_366350

def strawberry_problem (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
  (strawberries_per_jar jar_price total_revenue : ℕ) : Prop :=
  betty_strawberries = 16 ∧
  matthew_strawberries = betty_strawberries + 20 ∧
  matthew_strawberries = natalie_strawberries ∧
  strawberries_per_jar = 7 ∧
  jar_price = 4 ∧
  total_revenue = 40 ∧
  (matthew_strawberries : ℚ) / natalie_strawberries = 1

theorem strawberry_ratio :
  ∀ (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
    (strawberries_per_jar jar_price total_revenue : ℕ),
  strawberry_problem betty_strawberries matthew_strawberries natalie_strawberries
    strawberries_per_jar jar_price total_revenue →
  (matthew_strawberries : ℚ) / natalie_strawberries = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_ratio_l3663_366350


namespace NUMINAMATH_CALUDE_problem_solution_l3663_366379

theorem problem_solution (a b : ℤ) 
  (eq1 : 1010 * a + 1014 * b = 1018)
  (eq2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3663_366379


namespace NUMINAMATH_CALUDE_roof_dimensions_difference_l3663_366317

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof side -/
def area (side : RoofSide) : ℝ := side.width * side.length

theorem roof_dimensions_difference (roof : RoofSide) 
  (h1 : roof.length = 4 * roof.width)  -- Length is 3 times longer than width
  (h2 : 2 * area roof = 588)  -- Combined area of two sides is 588
  : roof.length - roof.width = 3 * Real.sqrt (588 / 8) := by
  sorry

end NUMINAMATH_CALUDE_roof_dimensions_difference_l3663_366317


namespace NUMINAMATH_CALUDE_volume_of_region_l3663_366336

-- Define the function f
def f (x y z : ℝ) : ℝ := |x - y + z| + |x - y - z| + |x + y - z| + |-x + y - z|

-- Define the region R
def R : Set (ℝ × ℝ × ℝ) := {(x, y, z) | f x y z ≤ 6}

-- Theorem statement
theorem volume_of_region : MeasureTheory.volume R = 36 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l3663_366336


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3663_366302

theorem hyperbola_condition (m : ℝ) :
  (m > 0 → m * (m + 2) > 0) ∧ ¬(m * (m + 2) > 0 → m > 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3663_366302


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l3663_366398

open Complex

theorem max_distance_on_circle (z : ℂ) :
  abs (z - (1 + I)) = 1 →
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) ≥ abs (z - (4 + 5*I))) ∧
  (∀ (w : ℂ), abs (w - (1 + I)) = 1 → abs (w - (4 + 5*I)) ≤ 6) ∧
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l3663_366398


namespace NUMINAMATH_CALUDE_marble_selection_ways_l3663_366300

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def red_marbles : ℕ := 1

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_ways : 
  choose_marbles (total_marbles - red_marbles) (marbles_to_choose - red_marbles) = 56 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l3663_366300


namespace NUMINAMATH_CALUDE_P_superset_Q_l3663_366396

def P : Set ℝ := {x | x < 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

theorem P_superset_Q : P ⊃ Q := by
  sorry

end NUMINAMATH_CALUDE_P_superset_Q_l3663_366396


namespace NUMINAMATH_CALUDE_john_payment_l3663_366305

def lawyer_fee (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) : ℕ :=
  upfront_fee + hourly_rate * (court_hours + prep_time_multiplier * court_hours)

theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) :
  upfront_fee = 1000 →
  hourly_rate = 100 →
  court_hours = 50 →
  prep_time_multiplier = 2 →
  lawyer_fee upfront_fee hourly_rate court_hours prep_time_multiplier / 2 = 8000 :=
by
  sorry

#check john_payment

end NUMINAMATH_CALUDE_john_payment_l3663_366305


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3663_366371

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:8,
    given specific dimensions of the field and pond. -/
theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 36 →
    field_length = 2 * field_width →
    pond_side = 9 →
    (pond_side^2) / (field_length * field_width) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3663_366371


namespace NUMINAMATH_CALUDE_initial_percent_problem_l3663_366356

theorem initial_percent_problem (x : ℝ) : 
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_percent_problem_l3663_366356


namespace NUMINAMATH_CALUDE_min_points_theorem_min_points_is_minimal_l3663_366376

/-- Represents a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle APB given three points A, P, and B -/
def angle (A P B : Point) : ℝ := sorry

/-- The minimum number of points satisfying the given condition -/
def min_points : ℕ := 1993

theorem min_points_theorem (A B : Point) :
  ∀ (points : Finset Point),
    points.card ≥ min_points →
    ∃ (Pi Pj : Point), Pi ∈ points ∧ Pj ∈ points ∧ Pi ≠ Pj ∧
      |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| ≤ 1 / 1992 :=
by sorry

theorem min_points_is_minimal :
  ∀ k : ℕ, k < min_points →
    ∃ (A B : Point) (points : Finset Point),
      points.card = k ∧
      ∀ (Pi Pj : Point), Pi ∈ points → Pj ∈ points → Pi ≠ Pj →
        |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| > 1 / 1992 :=
by sorry

end NUMINAMATH_CALUDE_min_points_theorem_min_points_is_minimal_l3663_366376


namespace NUMINAMATH_CALUDE_clinton_shoes_count_l3663_366313

theorem clinton_shoes_count (hats belts shoes : ℕ) : 
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by
sorry

end NUMINAMATH_CALUDE_clinton_shoes_count_l3663_366313


namespace NUMINAMATH_CALUDE_distinct_positive_solutions_l3663_366353

theorem distinct_positive_solutions (a b : ℝ) :
  (∃ (x y z : ℝ), x + y + z = a ∧ x^2 + y^2 + z^2 = b^2 ∧ x*y = z^2 ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (abs b < a ∧ a < Real.sqrt 3 * abs b) :=
sorry

end NUMINAMATH_CALUDE_distinct_positive_solutions_l3663_366353


namespace NUMINAMATH_CALUDE_students_on_bleachers_l3663_366378

/-- Given a total of 26 students and a ratio of 11:13 for students on the floor to total students,
    prove that the number of students on the bleachers is 4. -/
theorem students_on_bleachers :
  ∀ (floor bleachers : ℕ),
    floor + bleachers = 26 →
    floor / (floor + bleachers : ℚ) = 11 / 13 →
    bleachers = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_on_bleachers_l3663_366378


namespace NUMINAMATH_CALUDE_deduced_card_final_card_l3663_366395

-- Define the suits and ranks
inductive Suit
| Hearts | Spades | Clubs | Diamonds

inductive Rank
| A | K | Q | J | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two

-- Define a card as a pair of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Define the set of cards in the drawer
def drawer : List Card := [
  ⟨Suit.Hearts, Rank.A⟩, ⟨Suit.Hearts, Rank.Q⟩, ⟨Suit.Hearts, Rank.Four⟩,
  ⟨Suit.Spades, Rank.J⟩, ⟨Suit.Spades, Rank.Eight⟩, ⟨Suit.Spades, Rank.Four⟩,
  ⟨Suit.Spades, Rank.Two⟩, ⟨Suit.Spades, Rank.Seven⟩, ⟨Suit.Spades, Rank.Three⟩,
  ⟨Suit.Clubs, Rank.K⟩, ⟨Suit.Clubs, Rank.Q⟩, ⟨Suit.Clubs, Rank.Five⟩,
  ⟨Suit.Clubs, Rank.Four⟩, ⟨Suit.Clubs, Rank.Six⟩,
  ⟨Suit.Diamonds, Rank.A⟩, ⟨Suit.Diamonds, Rank.Five⟩
]

-- Define the conditions based on the conversation
def qian_first_statement (c : Card) : Prop :=
  c.rank = Rank.A ∨ c.rank = Rank.Q ∨ c.rank = Rank.Five ∨ c.rank = Rank.Four

def sun_first_statement (c : Card) : Prop :=
  c.suit = Suit.Hearts ∨ c.suit = Suit.Diamonds

def qian_second_statement (c : Card) : Prop :=
  c.rank ≠ Rank.A

-- The main theorem
theorem deduced_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c :=
  sorry

-- The final conclusion
theorem final_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c ∧
    c = ⟨Suit.Diamonds, Rank.Five⟩ :=
  sorry

end NUMINAMATH_CALUDE_deduced_card_final_card_l3663_366395


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l3663_366331

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℝ) (num_days : ℕ) (h1 : total_earnings = 120) (h2 : num_days = 20) :
  total_earnings / num_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l3663_366331


namespace NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l3663_366366

/-- A 9-pointed star is formed by connecting every fourth point of 9 evenly spaced points on a circle. -/
structure NinePointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The number of points to skip when forming the star -/
  skip_points : ℕ
  /-- The number of points is 9 -/
  points_eq_nine : num_points = 9
  /-- We skip every 3 points (connect every 4th) -/
  skip_three : skip_points = 3

/-- The sum of the angles at the tips of a 9-pointed star is 540 degrees -/
theorem nine_pointed_star_angle_sum (star : NinePointedStar) : 
  (star.num_points : ℝ) * (360 / (2 * star.num_points : ℝ) * star.skip_points) = 540 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l3663_366366


namespace NUMINAMATH_CALUDE_egg_ratio_is_two_to_one_l3663_366377

def egg_laying_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 50 ∧
  day2 = 2 * day1 ∧
  day3 = day2 + 20 ∧
  total = 810 ∧
  day4 = total - (day1 + day2 + day3)

theorem egg_ratio_is_two_to_one :
  ∀ day1 day2 day3 day4 total : ℕ,
    egg_laying_problem day1 day2 day3 day4 total →
    day4 * (day1 + day2 + day3) = 2 * (day1 + day2 + day3) * (day1 + day2 + day3) :=
by
  sorry

end NUMINAMATH_CALUDE_egg_ratio_is_two_to_one_l3663_366377


namespace NUMINAMATH_CALUDE_badge_making_contest_tables_l3663_366385

theorem badge_making_contest_tables (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) : 
  stools_per_table = 7 → 
  stool_legs = 4 → 
  table_legs = 5 → 
  total_legs = 658 → 
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    total_legs = stool_legs * stools_per_table * num_tables + table_legs * num_tables :=
by sorry

end NUMINAMATH_CALUDE_badge_making_contest_tables_l3663_366385


namespace NUMINAMATH_CALUDE_radio_price_rank_l3663_366397

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 16 →
  prices.card = n + 1 →
  (∀ (p q : ℕ), p ∈ prices → q ∈ prices → p ≠ q) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 3 →
  (prices.filter (λ p => p < radio_price)).card = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_radio_price_rank_l3663_366397


namespace NUMINAMATH_CALUDE_centers_regular_iff_original_affinely_regular_l3663_366373

open Complex

/-- Definition of an n-gon as a list of complex numbers -/
def NGon (n : ℕ) := List ℂ

/-- A convex n-gon -/
def ConvexNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Centers of regular n-gons constructed on sides of an n-gon -/
def CentersOfExternalNGons (n : ℕ) (A : NGon n) : NGon n := sorry

/-- Check if an n-gon is regular -/
def IsRegularNGon (n : ℕ) (B : NGon n) : Prop := sorry

/-- Check if an n-gon is affinely regular -/
def IsAffinelyRegularNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Main theorem: The centers form a regular n-gon iff the original n-gon is affinely regular -/
theorem centers_regular_iff_original_affinely_regular 
  (n : ℕ) (A : NGon n) (h : ConvexNGon n A) :
  IsRegularNGon n (CentersOfExternalNGons n A) ↔ IsAffinelyRegularNGon n A :=
sorry

end NUMINAMATH_CALUDE_centers_regular_iff_original_affinely_regular_l3663_366373


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l3663_366347

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width length_price width_price : ℕ) : ℕ :=
  2 * (length * length_price + width * width_price)

/-- Theorem stating the total cost of building the fence -/
theorem fence_cost_calculation :
  fence_cost 35 25 60 50 = 6700 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l3663_366347


namespace NUMINAMATH_CALUDE_subtraction_result_l3663_366360

theorem subtraction_result : 888888888888 - 111111111111 = 777777777777 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3663_366360


namespace NUMINAMATH_CALUDE_bat_ball_cost_difference_l3663_366341

theorem bat_ball_cost_difference (bat_cost ball_cost : ℕ) : 
  (2 * bat_cost + 3 * ball_cost = 1300) →
  (3 * bat_cost + 2 * ball_cost = 1200) →
  (ball_cost - bat_cost = 100) := by
sorry

end NUMINAMATH_CALUDE_bat_ball_cost_difference_l3663_366341


namespace NUMINAMATH_CALUDE_error_clock_correct_fraction_l3663_366386

/-- Represents a 12-hour digital clock with display errors -/
structure ErrorClock where
  /-- The clock displays '1' as '9' -/
  one_as_nine : Bool
  /-- The clock displays '2' as '5' -/
  two_as_five : Bool

/-- Calculates the fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  sorry

/-- Theorem stating that for a clock with both display errors, 
    the fraction of correct time is 49/144 -/
theorem error_clock_correct_fraction :
  ∀ (clock : ErrorClock), 
  clock.one_as_nine ∧ clock.two_as_five → 
  correct_time_fraction clock = 49 / 144 :=
sorry

end NUMINAMATH_CALUDE_error_clock_correct_fraction_l3663_366386


namespace NUMINAMATH_CALUDE_system_of_equations_l3663_366364

theorem system_of_equations (x y a : ℝ) : 
  (3 * x + y = a + 1) → 
  (x + 3 * y = 3) → 
  (x + y > 5) → 
  (a > 16) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l3663_366364


namespace NUMINAMATH_CALUDE_power_division_rule_l3663_366326

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^4 / a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3663_366326


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3663_366340

def n : ℕ := 2^100 * 3^4 * 5 * 7

theorem smallest_number_with_2020_divisors :
  (∀ m : ℕ, m < n → (Nat.divisors m).card ≠ 2020) ∧
  (Nat.divisors n).card = 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3663_366340


namespace NUMINAMATH_CALUDE_initial_money_amount_initial_money_amount_proof_l3663_366334

/-- Proves that given the conditions in the problem, the initial amount of money is 160 dollars --/
theorem initial_money_amount : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun your_weekly_savings friend_initial_money friend_weekly_savings weeks initial_money =>
    your_weekly_savings = 7 →
    friend_initial_money = 210 →
    friend_weekly_savings = 5 →
    weeks = 25 →
    initial_money + (your_weekly_savings * weeks) = friend_initial_money + (friend_weekly_savings * weeks) →
    initial_money = 160

/-- The proof of the theorem --/
theorem initial_money_amount_proof :
  initial_money_amount 7 210 5 25 160 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_amount_initial_money_amount_proof_l3663_366334


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3663_366306

theorem imaginary_part_of_z (z : ℂ) : z = (3 - I) / (1 + I) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3663_366306


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3663_366355

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ θ => 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ)
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    solutions.card = 4 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3663_366355


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3663_366324

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3663_366324


namespace NUMINAMATH_CALUDE_square_root_3adic_l3663_366314

/-- Checks if 201 is the square root of 112101 in 3-adic arithmetic up to 3 digits of precision -/
theorem square_root_3adic (n : Nat) : n = 201 → n * n ≡ 112101 [ZMOD 27] := by
  sorry

end NUMINAMATH_CALUDE_square_root_3adic_l3663_366314


namespace NUMINAMATH_CALUDE_pepperoni_coverage_l3663_366389

theorem pepperoni_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) (pepperoni_across : ℕ) :
  pizza_diameter = 18 →
  pepperoni_count = 36 →
  pepperoni_across = 9 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2 * Real.pi) / (pizza_diameter / 2)^2 / Real.pi = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_l3663_366389


namespace NUMINAMATH_CALUDE_charity_event_volunteers_l3663_366357

theorem charity_event_volunteers (n : ℕ) : 
  (n : ℚ) / 2 = (((n : ℚ) / 2 - 3) / n) * n → n / 2 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_volunteers_l3663_366357


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3663_366392

def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 7 * x + 1
def parabola2 (x : ℝ) : ℝ := 8 * x^2 + 5 * x + 1

theorem parabolas_intersection :
  ∃! (s : Set (ℝ × ℝ)), s = {(-2, 23), (0, 1)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ parabola1 x = y ∧ parabola2 x = y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3663_366392


namespace NUMINAMATH_CALUDE_september_electricity_usage_l3663_366382

theorem september_electricity_usage
  (october_usage : ℕ)
  (savings_percentage : ℚ)
  (h1 : october_usage = 1400)
  (h2 : savings_percentage = 30 / 100)
  (h3 : october_usage = (1 - savings_percentage) * september_usage) :
  september_usage = 2000 :=
sorry

end NUMINAMATH_CALUDE_september_electricity_usage_l3663_366382


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l3663_366332

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l3663_366332


namespace NUMINAMATH_CALUDE_ratio_problem_l3663_366367

theorem ratio_problem (second_term : ℝ) (ratio_percent : ℝ) (first_term : ℝ) :
  second_term = 25 →
  ratio_percent = 60 →
  first_term / second_term = ratio_percent / 100 →
  first_term = 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3663_366367


namespace NUMINAMATH_CALUDE_prob_product_odd_eight_rolls_l3663_366320

-- Define a standard die
def StandardDie : Type := Fin 6

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- Define the probability of rolling an odd number on a standard die
def probOddRoll : ℚ := 1 / 2

-- Define the number of rolls
def numRolls : Nat := 8

-- Theorem statement
theorem prob_product_odd_eight_rolls :
  (probOddRoll ^ numRolls : ℚ) = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_odd_eight_rolls_l3663_366320


namespace NUMINAMATH_CALUDE_roots_are_irrational_l3663_366342

theorem roots_are_irrational (j : ℝ) : 
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ x * y = 11) →
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ ¬(∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l3663_366342


namespace NUMINAMATH_CALUDE_max_product_sum_l3663_366304

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({5, 7, 9, 11} : Set ℕ) →
  g ∈ ({5, 7, 9, 11} : Set ℕ) →
  h ∈ ({5, 7, 9, 11} : Set ℕ) →
  j ∈ ({5, 7, 9, 11} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j : ℕ) ≤ 240 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l3663_366304


namespace NUMINAMATH_CALUDE_todd_spending_l3663_366375

/-- The amount Todd spent on the candy bar in cents -/
def candy_cost : ℕ := 14

/-- The amount Todd spent on the box of cookies in cents -/
def cookies_cost : ℕ := 39

/-- The total amount Todd spent in cents -/
def total_spent : ℕ := candy_cost + cookies_cost

theorem todd_spending :
  total_spent = 53 := by sorry

end NUMINAMATH_CALUDE_todd_spending_l3663_366375


namespace NUMINAMATH_CALUDE_right_triangle_angle_measure_l3663_366394

theorem right_triangle_angle_measure (A B C : ℝ) : 
  A = 90 →  -- A is the right angle (90 degrees)
  C = 3 * B →  -- C is three times B
  A + B + C = 180 →  -- Sum of angles in a triangle
  B = 22.5 :=  -- B is 22.5 degrees
by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_measure_l3663_366394


namespace NUMINAMATH_CALUDE_zoe_water_bottles_l3663_366303

/-- The initial number of water bottles Zoe had in her fridge -/
def initial_bottles : ℕ := 42

/-- The number of bottles Zoe drank -/
def bottles_drank : ℕ := 25

/-- The number of bottles Zoe bought -/
def bottles_bought : ℕ := 30

/-- The final number of bottles Zoe has -/
def final_bottles : ℕ := 47

theorem zoe_water_bottles :
  initial_bottles - bottles_drank + bottles_bought = final_bottles :=
sorry

end NUMINAMATH_CALUDE_zoe_water_bottles_l3663_366303


namespace NUMINAMATH_CALUDE_min_ab_in_triangle_l3663_366333

theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1/2) * a * b * Real.sin C = (Real.sqrt 3 / 2) * c →
  a * b ≥ 12 := by
sorry

end NUMINAMATH_CALUDE_min_ab_in_triangle_l3663_366333


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3663_366316

/-- Arithmetic sequence sum and remainder theorem -/
theorem arithmetic_sequence_sum_remainder
  (a : ℕ) -- First term
  (d : ℕ) -- Common difference
  (l : ℕ) -- Last term
  (h1 : a = 2)
  (h2 : d = 5)
  (h3 : l = 142)
  : (((l - a) / d + 1) * (a + l) / 2) % 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3663_366316


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3663_366335

def total_balls (a : ℕ) : ℕ := 2 + 3 + a

def probability_red (a : ℕ) : ℚ := 2 / total_balls a

theorem yellow_balls_count : ∃ a : ℕ, probability_red a = 1/3 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3663_366335


namespace NUMINAMATH_CALUDE_power_greater_than_square_l3663_366383

theorem power_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_l3663_366383


namespace NUMINAMATH_CALUDE_golf_ball_difference_l3663_366315

theorem golf_ball_difference (bin_f bin_g : ℕ) : 
  bin_f = (2 * bin_g) / 3 →
  bin_f + bin_g = 150 →
  bin_g - bin_f = 30 := by
sorry

end NUMINAMATH_CALUDE_golf_ball_difference_l3663_366315


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3663_366325

theorem ball_hitting_ground_time :
  ∃ t : ℝ, t > 0 ∧ -10 * t^2 - 20 * t + 180 = 0 ∧ t = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3663_366325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3663_366352

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  isArithmeticSequence a → a 6 + a 9 + a 12 = 48 → a 8 + a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3663_366352


namespace NUMINAMATH_CALUDE_integer_congruence_problem_l3663_366387

theorem integer_congruence_problem :
  ∀ n : ℤ, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_integer_congruence_problem_l3663_366387


namespace NUMINAMATH_CALUDE_tangent_line_proof_l3663_366348

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The proposed tangent line function -/
def g (x : ℝ) : ℝ := x + 1

theorem tangent_line_proof :
  (∃ x₀ : ℝ, f x₀ = g x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → f x < g x ∨ f x > g x)) ∧
  g (-1) = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l3663_366348


namespace NUMINAMATH_CALUDE_potato_fries_price_l3663_366307

/-- The price of a pack of potato fries given Einstein's fundraising scenario -/
theorem potato_fries_price (total_goal : ℚ) (pizza_price : ℚ) (soda_price : ℚ)
  (pizzas_sold : ℕ) (fries_sold : ℕ) (sodas_sold : ℕ) (remaining : ℚ)
  (h1 : total_goal = 500)
  (h2 : pizza_price = 12)
  (h3 : soda_price = 2)
  (h4 : pizzas_sold = 15)
  (h5 : fries_sold = 40)
  (h6 : sodas_sold = 25)
  (h7 : remaining = 258)
  : (total_goal - remaining - (pizza_price * pizzas_sold + soda_price * sodas_sold)) / fries_sold = (3 / 10) :=
sorry

end NUMINAMATH_CALUDE_potato_fries_price_l3663_366307


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3663_366351

theorem largest_divisor_of_n (n : ℕ+) (h : 50 ∣ n^2) : 5 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3663_366351


namespace NUMINAMATH_CALUDE_b_investment_is_60000_l3663_366363

/-- Represents the investment and profit sharing structure of a business partnership --/
structure BusinessPartnership where
  total_profit : ℝ
  a_investment : ℝ
  b_investment : ℝ
  a_management_share : ℝ
  a_total_share : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    B's investment is 60,000 --/
theorem b_investment_is_60000 (bp : BusinessPartnership)
  (h1 : bp.total_profit = 8800)
  (h2 : bp.a_investment = 50000)
  (h3 : bp.a_management_share = 0.125 * bp.total_profit)
  (h4 : bp.a_total_share = 4600)
  (h5 : bp.a_total_share = bp.a_management_share +
        (bp.total_profit - bp.a_management_share) * (bp.a_investment / (bp.a_investment + bp.b_investment)))
  : bp.b_investment = 60000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_60000_l3663_366363


namespace NUMINAMATH_CALUDE_simplify_expression_l3663_366319

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3663_366319


namespace NUMINAMATH_CALUDE_team_a_games_won_lost_team_b_minimum_wins_l3663_366345

/-- Represents the number of games a team plays in the tournament -/
def total_games : ℕ := 10

/-- Represents the points earned for a win -/
def win_points : ℕ := 2

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 1

/-- Represents the minimum points needed to qualify for the next round -/
def qualification_points : ℕ := 15

theorem team_a_games_won_lost (points : ℕ) (h : points = 18) :
  ∃ (wins losses : ℕ), wins + losses = total_games ∧
                        wins * win_points + losses * loss_points = points ∧
                        wins = 8 ∧ losses = 2 := by sorry

theorem team_b_minimum_wins :
  ∃ (min_wins : ℕ), ∀ (wins : ℕ),
    wins * win_points + (total_games - wins) * loss_points > qualification_points →
    wins ≥ min_wins ∧
    min_wins = 6 := by sorry

end NUMINAMATH_CALUDE_team_a_games_won_lost_team_b_minimum_wins_l3663_366345


namespace NUMINAMATH_CALUDE_max_discount_rate_l3663_366384

theorem max_discount_rate (cost_price : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 →
  original_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 60 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

#check max_discount_rate

end NUMINAMATH_CALUDE_max_discount_rate_l3663_366384


namespace NUMINAMATH_CALUDE_nursery_school_count_nursery_school_count_proof_l3663_366301

theorem nursery_school_count : ℕ → Prop :=
  fun total_students =>
    let students_4_and_older := total_students / 10
    let students_under_3 := 20
    let students_not_between_3_and_4 := 50
    students_4_and_older = students_not_between_3_and_4 - students_under_3 ∧
    total_students = 300

-- The proof of the theorem
theorem nursery_school_count_proof : ∃ n : ℕ, nursery_school_count n :=
  sorry

end NUMINAMATH_CALUDE_nursery_school_count_nursery_school_count_proof_l3663_366301


namespace NUMINAMATH_CALUDE_max_areas_theorem_l3663_366359

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ) : ℕ := 3 * n + 1

/-- 
Theorem: Given a circular disk divided by 2n equally spaced radii (n > 0) and one secant line, 
the maximum number of non-overlapping areas is 3n + 1.
-/
theorem max_areas_theorem (n : ℕ) (h : n > 0) : 
  max_areas n = 3 * n + 1 := by
  sorry

#check max_areas_theorem

end NUMINAMATH_CALUDE_max_areas_theorem_l3663_366359


namespace NUMINAMATH_CALUDE_product_remainder_l3663_366330

theorem product_remainder (k : ℕ) : ∃ n : ℕ, 
  n = 5 * k + 1 ∧ (14452 * 15652 * n) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3663_366330


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3663_366329

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism. -/
theorem cone_prism_volume_ratio
  (a b h_c h_p : ℝ)
  (h_ab : b < a)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_pos_h_c : h_c > 0)
  (h_pos_h_p : h_p > 0) :
  (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) = π * b * h_c / (12 * a * h_p) :=
by sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3663_366329


namespace NUMINAMATH_CALUDE_refund_calculation_l3663_366365

/-- Calculates the refund amount for returned cans given specific conditions -/
theorem refund_calculation (total_cans brand_a_price brand_b_price average_price discount restocking_fee tax : ℚ)
  (h1 : total_cans = 6)
  (h2 : brand_a_price = 33 / 100)
  (h3 : brand_b_price = 40 / 100)
  (h4 : average_price = 365 / 1000)
  (h5 : discount = 20 / 100)
  (h6 : restocking_fee = 5 / 100)
  (h7 : tax = 8 / 100)
  (h8 : ∃ (brand_a_count brand_b_count : ℚ), 
    brand_a_count + brand_b_count = total_cans ∧ 
    brand_a_count * brand_a_price + brand_b_count * brand_b_price = total_cans * average_price ∧
    brand_a_count > brand_b_count) :
  ∃ (refund : ℚ), refund = 55 / 100 := by
  sorry


end NUMINAMATH_CALUDE_refund_calculation_l3663_366365


namespace NUMINAMATH_CALUDE_keith_purchases_cost_l3663_366390

/-- The total cost of Keith's purchases -/
def total_cost (rabbit_toy pet_food cage water_bottle bedding found_money : ℝ)
  (rabbit_discount cage_tax : ℝ) : ℝ :=
  let rabbit_toy_original := rabbit_toy / (1 - rabbit_discount)
  let cage_with_tax := cage * (1 + cage_tax)
  rabbit_toy + pet_food + cage_with_tax + water_bottle + bedding - found_money

/-- Theorem stating the total cost of Keith's purchases -/
theorem keith_purchases_cost :
  total_cost 6.51 5.79 12.51 4.99 7.65 1 0.1 0.08 = 37.454 := by
  sorry

end NUMINAMATH_CALUDE_keith_purchases_cost_l3663_366390


namespace NUMINAMATH_CALUDE_first_rope_longer_l3663_366358

-- Define the initial length of the ropes
variable (initial_length : ℝ)

-- Define the lengths cut from each rope
def cut_length_1 : ℝ := 0.3
def cut_length_2 : ℝ := 3

-- Define the remaining lengths of each rope
def remaining_length_1 : ℝ := initial_length - cut_length_1
def remaining_length_2 : ℝ := initial_length - cut_length_2

-- Theorem statement
theorem first_rope_longer :
  remaining_length_1 initial_length > remaining_length_2 initial_length :=
by sorry

end NUMINAMATH_CALUDE_first_rope_longer_l3663_366358


namespace NUMINAMATH_CALUDE_car_cost_calculation_l3663_366370

/-- The cost of a car shared between two people, where one pays $900 for 3/7 of the usage -/
theorem car_cost_calculation (sue_payment : ℝ) (sue_usage : ℚ) (total_cost : ℝ) : 
  sue_payment = 900 → 
  sue_usage = 3/7 → 
  sue_payment / total_cost = sue_usage →
  total_cost = 2100 := by
  sorry

#check car_cost_calculation

end NUMINAMATH_CALUDE_car_cost_calculation_l3663_366370


namespace NUMINAMATH_CALUDE_sine_graph_shift_l3663_366321

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x - 2 * π / 3)) = 2 * Real.sin (2 * ((x + 2 * π / 3) - 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l3663_366321


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3663_366308

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_properties
  (a b : ℝ) (h_a : a ≠ 0)
  (h_min : ∀ x, f a b x ≥ f a b 1)
  (h_zero : f a b 1 = 0) :
  -- 1. f(x) = x² - 2x + 1
  (∀ x, f a b x = x^2 - 2*x + 1) ∧
  -- 2. f(x) is decreasing on (-∞, 1] and increasing on [1, +∞)
  (∀ x y, x ≤ 1 → y ≤ 1 → x ≤ y → f a b x ≥ f a b y) ∧
  (∀ x y, 1 ≤ x → 1 ≤ y → x ≤ y → f a b x ≤ f a b y) ∧
  -- 3. If f(x) > x + k for all x ∈ [1, 3], then k < -5/4
  (∀ k, (∀ x, 1 ≤ x → x ≤ 3 → f a b x > x + k) → k < -5/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3663_366308


namespace NUMINAMATH_CALUDE_student_selection_methods_l3663_366327

/-- Represents the number of ways to select students by gender from a group -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select 4 students by gender from a group of 8 students (6 boys and 2 girls) is 40 -/
theorem student_selection_methods :
  select_students 8 6 2 4 = 40 :=
sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3663_366327


namespace NUMINAMATH_CALUDE_base3_10201_equals_100_l3663_366310

def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 3^i) 0

theorem base3_10201_equals_100 :
  base3ToDecimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_base3_10201_equals_100_l3663_366310


namespace NUMINAMATH_CALUDE_circle_equation_l3663_366362

/-- A circle C with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line x - 2y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- A point (x, y) lies on the circle C -/
def on_circle (C : Circle) (x y : ℝ) : Prop :=
  (x - C.h)^2 + (y - C.k)^2 = C.r^2

theorem circle_equation : ∃ C : Circle,
  (line C.h C.k) ∧
  (on_circle C 0 0) ∧
  (on_circle C 1 2) ∧
  (C.h = 7/4 ∧ C.k = 3/8 ∧ C.r^2 = 205/64) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3663_366362


namespace NUMINAMATH_CALUDE_parabola_directrix_l3663_366381

/-- Given a fixed point A(2,1) and a parabola y^2 = 2px (p > 0) whose focus lies on the perpendicular 
    bisector of OA, prove that the directrix of the parabola has the equation x = -5/4 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (2, 1)
  let O : ℝ × ℝ := (0, 0)
  let focus : ℝ × ℝ := (p/2, 0)
  let perp_bisector (x y : ℝ) := 4*x + 2*y - 5 = 0
  let parabola (x y : ℝ) := y^2 = 2*p*x
  let directrix (x : ℝ) := x = -5/4
  (perp_bisector (focus.1) (focus.2)) →
  (∀ x y, parabola x y → (x = -p/2 ↔ directrix x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3663_366381


namespace NUMINAMATH_CALUDE_computer_price_increase_l3663_366391

theorem computer_price_increase (d : ℝ) (h1 : d * 1.3 = 338) (h2 : ∃ x : ℝ, x * d = 520) : 
  ∃ x : ℝ, x * d = 520 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3663_366391


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l3663_366369

theorem cylinder_volume_equality (x : ℚ) : x > 0 →
  (5 + x)^2 * 4 = 25 * (4 + x) → x = 35/4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l3663_366369


namespace NUMINAMATH_CALUDE_cube_of_complex_number_l3663_366343

theorem cube_of_complex_number :
  let z : ℂ := 2 + 5*I
  z^3 = -142 - 65*I := by sorry

end NUMINAMATH_CALUDE_cube_of_complex_number_l3663_366343


namespace NUMINAMATH_CALUDE_range_of_a_l3663_366337

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) → (a > 3 ∨ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3663_366337


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3663_366380

theorem parallel_vectors_sum (x y : ℝ) 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (ha : a = (2, 1, x)) (hb : b = (4, y, -1)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  2 * x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3663_366380


namespace NUMINAMATH_CALUDE_line_intercepts_sum_and_product_l3663_366354

/-- Given a line with equation y - 2 = -3(x + 5), prove that the sum of its
    x-intercept and y-intercept is -52/3, and their product is 169/3. -/
theorem line_intercepts_sum_and_product :
  let f : ℝ → ℝ := λ x => -3 * (x + 5) + 2
  let x_intercept := -13 / 3
  let y_intercept := f 0
  (x_intercept + y_intercept = -52 / 3) ∧ (x_intercept * y_intercept = 169 / 3) := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_and_product_l3663_366354


namespace NUMINAMATH_CALUDE_prob_second_draw_l3663_366361

structure Bag where
  red : ℕ
  blue : ℕ

def initial_bag : Bag := ⟨5, 4⟩

def P_A2 (b : Bag) : ℚ :=
  (b.red : ℚ) / (b.red + b.blue)

def P_B2 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue)

def P_A2_given_A1 (b : Bag) : ℚ :=
  ((b.red - 1) : ℚ) / (b.red + b.blue - 1)

def P_B2_given_A1 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue - 1)

theorem prob_second_draw (b : Bag) :
  P_A2 b = 5/9 ∧
  P_A2 b + P_B2 b = 1 ∧
  P_A2_given_A1 b + P_B2_given_A1 b = 1 :=
by sorry

end NUMINAMATH_CALUDE_prob_second_draw_l3663_366361


namespace NUMINAMATH_CALUDE_product_closure_l3663_366399

def A : Set ℤ := {z | ∃ a b : ℤ, z = a^2 + 4*a*b + b^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l3663_366399


namespace NUMINAMATH_CALUDE_function_value_at_two_l3663_366344

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) : f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3663_366344


namespace NUMINAMATH_CALUDE_marys_blueberries_l3663_366318

theorem marys_blueberries (apples oranges total_left : ℕ) (h1 : apples = 14) (h2 : oranges = 9) (h3 : total_left = 26) :
  ∃ blueberries : ℕ, blueberries = 5 ∧ total_left = (apples - 1) + (oranges - 1) + (blueberries - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_marys_blueberries_l3663_366318


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l3663_366322

theorem jason_pokemon_cards (initial_cards : ℕ) (given_away : ℕ) : 
  initial_cards = 9 → given_away = 4 → initial_cards - given_away = 5 := by
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l3663_366322
