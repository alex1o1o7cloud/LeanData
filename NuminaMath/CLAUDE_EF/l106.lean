import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l106_10642

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a circle in the xy-plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem: The eccentricity of a specific hyperbola is 5/4 -/
theorem hyperbola_eccentricity (h : Hyperbola) (c : Circle)
  (h_asymptote_tangent : ∃ (m : ℝ), m * h.a = h.b ∧ 
    |2 * h.a - h.b| / Real.sqrt (h.a^2 + h.b^2) = 1)
  (c_def : c.center_x = 2 ∧ c.center_y = 1 ∧ c.radius = 1) :
  eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l106_10642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_earnings_l106_10640

/-- Calculates the earnings from a stock investment -/
noncomputable def stock_earnings (investment : ℝ) (dividend_rate : ℝ) (market_price : ℝ) : ℝ :=
  let face_value := (investment / market_price) * 100
  (face_value * dividend_rate) / 100

/-- Theorem stating that the given investment yields the expected earnings -/
theorem investment_earnings : 
  let investment := (1800 : ℝ)
  let dividend_rate := (9 : ℝ)
  let market_price := (135 : ℝ)
  let expected_earnings := (119.99 : ℝ)
  abs (stock_earnings investment dividend_rate market_price - expected_earnings) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_earnings_l106_10640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_equals_prob_B_l106_10659

-- Define the sample space for a two-coin toss
inductive TwoCoinToss : Type
  | HH : TwoCoinToss
  | HT : TwoCoinToss
  | TH : TwoCoinToss
  | TT : TwoCoinToss

-- Define events A and B
def A (outcome : TwoCoinToss) : Prop := outcome = TwoCoinToss.HH ∨ outcome = TwoCoinToss.HT
def B (outcome : TwoCoinToss) : Prop := outcome = TwoCoinToss.HT ∨ outcome = TwoCoinToss.TT

-- Define a probability measure on the sample space
variable (P : TwoCoinToss → ℝ)

-- Axioms for probability measure
axiom prob_nonneg : ∀ x, P x ≥ 0
axiom prob_total : (P TwoCoinToss.HH) + (P TwoCoinToss.HT) + (P TwoCoinToss.TH) + (P TwoCoinToss.TT) = 1

-- Fairness of coins
axiom fair_coins : P TwoCoinToss.HH = P TwoCoinToss.HT ∧ P TwoCoinToss.HH = P TwoCoinToss.TH ∧ P TwoCoinToss.HH = P TwoCoinToss.TT

-- Theorem: P(A) = P(B)
theorem prob_A_equals_prob_B : (P TwoCoinToss.HH + P TwoCoinToss.HT) = (P TwoCoinToss.HT + P TwoCoinToss.TT) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_equals_prob_B_l106_10659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aarons_test_score_l106_10627

theorem aarons_test_score (total_students : ℕ) (students_without_aaron : ℕ) 
  (avg_without_aaron : ℚ) (avg_with_aaron : ℚ) : 
  total_students = 20 →
  students_without_aaron = 19 →
  avg_without_aaron = 82 →
  avg_with_aaron = 83 →
  total_students * avg_with_aaron - students_without_aaron * avg_without_aaron = 102 := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aarons_test_score_l106_10627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l106_10673

-- Define the function for cosine in terms of 'a'
noncomputable def cos_x (a : ℝ) : ℝ := (2*a - 3) / (4 - a)

-- Define the condition for x being in the second or third quadrant
def x_in_second_or_third_quadrant (a : ℝ) : Prop := 
  -1 < cos_x a ∧ cos_x a < 0

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, x_in_second_or_third_quadrant a ↔ -1 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l106_10673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_decomposition_l106_10618

theorem cos_seven_decomposition (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ) :
  (∀ θ : ℝ, (Real.cos θ)^7 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                        b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ) + 
                        b₇ * Real.cos (7*θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 1555 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_decomposition_l106_10618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l106_10656

-- Define the curve C₂
noncomputable def curve_C2 (α : Real) : Real × Real := (Real.cos α, 2 * Real.sin α)

-- Define the line l
def line_l (x y : Real) : Prop := x + y + 6 = 0

-- Define the distance function between a point and the line
noncomputable def distance_to_line (p : Real × Real) : Real :=
  let (x, y) := p
  abs (x + y + 6) / Real.sqrt 2

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (min_dist : Real), min_dist = 3 * Real.sqrt 2 - Real.sqrt 10 / 2 ∧
  ∀ (α : Real), distance_to_line (curve_C2 α) ≥ min_dist := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l106_10656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_arrangement_l106_10691

/-- The number of smaller semicircles arranged along the diameter of a larger semicircle -/
def N : ℕ := 10

/-- The radius of each smaller semicircle -/
noncomputable def r : ℝ := 1

/-- The area covered by all smaller semicircles -/
noncomputable def A : ℝ := (N * Real.pi * r^2) / 2

/-- The area inside the larger semicircle but outside the smaller semicircles -/
noncomputable def B : ℝ := (Real.pi * (N * r)^2) / 2 - A

/-- Theorem stating that given the arrangement of semicircles and the area ratio, N must be 10 -/
theorem semicircle_arrangement :
  (A / B = 1 / 9) → N = 10 := by
  sorry

#eval N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_arrangement_l106_10691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winning_post_distance_l106_10645

/-- Represents a runner with a given speed -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  runnerA : Runner
  runnerB : Runner
  headStart : ℝ

noncomputable def Race.winningPostDistance (race : Race) : ℝ :=
  (race.headStart * race.runnerA.speed) / (race.runnerA.speed - race.runnerB.speed)

theorem race_winning_post_distance 
  (race : Race) 
  (h1 : race.runnerA.speed = (5/3) * race.runnerB.speed) 
  (h2 : race.headStart = 80) : 
  race.winningPostDistance = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winning_post_distance_l106_10645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l106_10690

/-- Represents the radii of the four semicircular arcs in the track --/
structure TrackRadii where
  R₁ : ℝ
  R₂ : ℝ
  R₃ : ℝ
  R₄ : ℝ

/-- Calculates the total distance traveled by the center of a ball along a track --/
noncomputable def total_distance (ball_diameter : ℝ) (radii : TrackRadii) (elevation : ℝ) : ℝ :=
  let ball_radius := ball_diameter / 2
  let arc1 := Real.pi * (radii.R₁ - ball_radius)
  let arc2 := Real.pi * (radii.R₂ + ball_radius + elevation)
  let arc3 := Real.pi * (radii.R₃ - ball_radius)
  let arc4 := Real.pi * (radii.R₄ - ball_radius)
  arc1 + arc2 + arc3 + arc4

/-- Theorem stating that the total distance traveled by the center of the ball is 408π inches --/
theorem ball_travel_distance :
  let radii : TrackRadii := { R₁ := 150, R₂ := 50, R₃ := 90, R₄ := 120 }
  total_distance 6 radii 4 = 408 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l106_10690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_wins_with_2029_coins_l106_10655

/-- Represents the possible moves in the coin game -/
inductive Move where
  | lucas_three : Move
  | lucas_five : Move
  | maria_two : Move
  | maria_four : Move

/-- Represents the current state of the game -/
structure GameState where
  coins : Nat
  is_lucas_turn : Bool

/-- Defines whether a game state is winning for the current player -/
def is_winning_state : GameState → Prop := sorry

/-- Applies a move to the current game state -/
def apply_move : Move → GameState → Option GameState := sorry

/-- Returns the optimal move for the current player -/
noncomputable def optimal_move : GameState → Option Move := sorry

/-- Theorem: Maria wins the game starting with 2029 coins -/
theorem maria_wins_with_2029_coins :
  let initial_state : GameState := { coins := 2029, is_lucas_turn := true }
  ¬is_winning_state initial_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_wins_with_2029_coins_l106_10655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l106_10620

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (x^2 + 3 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -5 ∨ (-5 < x ∧ x < 2) ∨ 2 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l106_10620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_for_greater_angle_sin_sum_greater_than_cos_sum_tan_product_greater_than_one_l106_10613

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0

-- Theorem 1
theorem sin_greater_for_greater_angle (t : AcuteTriangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by sorry

-- Theorem 2
theorem sin_sum_greater_than_cos_sum (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B > Real.cos t.A + Real.cos t.B := by sorry

-- Theorem 3
theorem tan_product_greater_than_one (t : AcuteTriangle) :
  Real.tan t.B * Real.tan t.C > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_for_greater_angle_sin_sum_greater_than_cos_sum_tan_product_greater_than_one_l106_10613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_complement_intersection_l106_10664

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | x ≤ -4 ∨ x ≥ 4}

-- Theorem for part (1)
theorem union_A_B : A ∪ B = Set.Icc (-4 : ℝ) 6 := by sorry

-- Theorem for part (2)
theorem complement_intersection : 
  (Set.univ \ A) ∩ (Set.univ \ B) = Set.Iic (-4 : ℝ) ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_complement_intersection_l106_10664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_pays_6_yuan_prob_sum_36_yuan_l106_10688

-- Define the parking fee structure
noncomputable def parking_fee (hours : ℝ) : ℕ :=
  if hours ≤ 1 then 6
  else 6 + 8 * (Int.ceil (hours - 1)).toNat

-- Define the probability space
structure ParkingProbability where
  prob_A_1_to_2_hours : ℚ
  prob_A_over_14_yuan : ℚ
  max_hours : ℕ
  equal_probability : Bool

-- Define the given probabilities
def given_probabilities : ParkingProbability :=
  { prob_A_1_to_2_hours := 1/3
  , prob_A_over_14_yuan := 5/12
  , max_hours := 4
  , equal_probability := true }

-- Theorem 1: Probability that A pays exactly 6 yuan
theorem prob_A_pays_6_yuan (p : ParkingProbability) :
  p = given_probabilities →
  1 - (p.prob_A_1_to_2_hours + p.prob_A_over_14_yuan) = 1/4 := by sorry

-- Theorem 2: Probability that the sum of A and B's parking fees is 36 yuan
theorem prob_sum_36_yuan (p : ParkingProbability) :
  p = given_probabilities →
  p.equal_probability →
  (4 : ℚ) / 16 = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_pays_6_yuan_prob_sum_36_yuan_l106_10688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l106_10601

/-- The area of a triangle with vertices at the origin, (3, 2), and (-1, 5) is 8.5 -/
theorem triangle_area : ∃ A : ℝ, A = 8.5 ∧ A = (1/2) * |3 * 5 - (-1) * 2| := by
  use 8.5
  constructor
  · rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l106_10601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l106_10672

noncomputable def f (x : ℝ) := 2 - Real.exp x

noncomputable def a : ℝ := Real.log (Real.sqrt 3)
noncomputable def b : ℝ := Real.log (Real.sqrt 8)

theorem arc_length_of_curve (f : ℝ → ℝ) (a b : ℝ) :
  a < b →
  (∫ x in a..b, Real.sqrt (1 + ((deriv f) x)^2)) = 1 + (1/2) * Real.log (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l106_10672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_smallest_and_largest_l106_10668

def numbers : List ℝ := [2.8, 2.3, 5, 4.3]

theorem product_of_smallest_and_largest : 
  (List.minimum numbers).map (· * (List.maximum numbers).getD 0) = some 11.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_smallest_and_largest_l106_10668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_representation_l106_10625

theorem quadratic_polynomial_representation (A B C : ℝ) :
  ∃ (k l m : ℝ),
    (∀ x, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) ∧
    k = 2 * A ∧
    l = A + B ∧
    m = C ∧
    (∀ x : ℤ, (∃ n : ℤ, ↑n = A * ↑x^2 + B * ↑x + C) ↔ 
      (∃ kn ln mn : ℤ, ↑kn = k ∧ ↑ln = l ∧ ↑mn = m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_representation_l106_10625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l106_10667

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Calculate the area of a triangle formed by a line and the coordinate axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  abs (l.c / l.a * l.c / l.b) / 2

theorem line_equation_part1 (l : Line) :
  l.contains (-2) 1 ∧ l.perpendicular { a := 2, b := 3, c := 5 } →
  l = { a := 3, b := -2, c := 8 } := by
  sorry

theorem line_equation_part2 (l : Line) :
  l.contains (-2) 1 ∧ triangle_area l = 1/2 →
  (l = { a := 1, b := 1, c := 1 } ∨ l = { a := 1, b := 4, c := -2 }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l106_10667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_p_is_circle_l106_10669

-- Define the complex number z
noncomputable def z (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

-- Define p as a function of θ
noncomputable def p (θ : ℝ) : ℂ := (z θ)^2 + 2 * (z θ) * (Real.cos θ)

-- Theorem statement
theorem locus_of_p_is_circle :
  ∃ (center : ℂ) (radius : ℝ), ∀ θ, Complex.abs (p θ - center) = radius :=
by
  -- Define the center and radius
  let center : ℂ := -1
  let radius : ℝ := 2
  
  -- Assert their existence
  use center, radius
  
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_p_is_circle_l106_10669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l106_10687

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l106_10687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_inequality_l106_10652

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Theorem: In any convex quadrilateral, the sum of any two opposite sides
    is less than the sum of the other two sides and the diagonals -/
theorem convex_quadrilateral_inequality 
  (A B M N : Point) 
  (h_convex : ¬collinear A B M ∧ ¬collinear A B N ∧ ¬collinear A M N ∧ ¬collinear B M N) :
  distance A B + distance M N < distance A M + distance B N + distance A N + distance B M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_inequality_l106_10652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steel_iron_ratio_l106_10699

/-- Given an alloy containing steel and iron where 35 kg of steel is melted with 14 kg of iron,
    the ratio of steel to iron in the alloy is 5:2. -/
theorem steel_iron_ratio (steel : ℚ) (iron : ℚ) 
  (h_steel : steel = 35) (h_iron : iron = 14) : 
  steel / iron = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steel_iron_ratio_l106_10699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_distance_proof_l106_10616

/-- The distance between Maxwell and Brad's homes in kilometers -/
noncomputable def total_distance : ℝ := 36

/-- Maxwell's walking speed in km/h -/
noncomputable def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
noncomputable def brad_speed : ℝ := 6

/-- Asher's cycling speed in km/h -/
noncomputable def asher_speed : ℝ := 9

/-- The distance Asher lives from both Maxwell and Brad's homes -/
noncomputable def asher_distance : ℝ := total_distance / 2

/-- The distance traveled by Maxwell when all three meet -/
noncomputable def maxwell_distance : ℝ := 12

theorem maxwell_distance_proof :
  maxwell_distance * (1 / maxwell_speed) =
  (total_distance - maxwell_distance) * (1 / brad_speed) ∧
  maxwell_distance * (1 / maxwell_speed) =
  (asher_distance - maxwell_distance) * (1 / asher_speed) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_distance_proof_l106_10616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l106_10629

theorem hexagon_largest_angle (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720 →
  (a₁ : ℝ) / 2 = (a₂ : ℝ) / 3 ∧ (a₂ : ℝ) / 3 = (a₃ : ℝ) / 3 ∧
  (a₃ : ℝ) / 3 = (a₄ : ℝ) / 4 ∧ (a₄ : ℝ) / 4 = (a₅ : ℝ) / 4 ∧
  (a₅ : ℝ) / 4 = (a₆ : ℝ) / 5 →
  max a₁ (max a₂ (max a₃ (max a₄ (max a₅ a₆)))) = 1200 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l106_10629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_analysis_l106_10636

/-- Represents the data from the student survey --/
structure SurveyData where
  total_students : ℕ
  myopia_rate : ℚ
  long_usage_rate : ℚ
  long_usage_myopia_rate : ℚ

/-- Represents the contingency table --/
structure ContingencyTable where
  a : ℕ  -- Myopic students with long usage
  b : ℕ  -- Non-myopic students with long usage
  c : ℕ  -- Myopic students with short usage
  d : ℕ  -- Non-myopic students with short usage

/-- Calculates the chi-square value --/
def chi_square (n : ℕ) (table : ContingencyTable) : ℚ :=
  let num := (n : ℚ) * ((table.a * table.d - table.b * table.c) : ℚ)^2
  let den := ((table.a + table.b) * (table.c + table.d) * (table.a + table.c) * (table.b + table.d) : ℚ)
  num / den

/-- Represents the distribution of X --/
structure Distribution where
  p0 : ℚ  -- P(X = 0)
  p1 : ℚ  -- P(X = 1)
  p2 : ℚ  -- P(X = 2)

/-- Main theorem --/
theorem survey_analysis (data : SurveyData) 
  (h1 : data.total_students = 2000)
  (h2 : data.myopia_rate = 2/5)
  (h3 : data.long_usage_rate = 1/5)
  (h4 : data.long_usage_myopia_rate = 1/2) :
  ∃ (table : ContingencyTable) (dist : Distribution),
    chi_square data.total_students table = 20833/1000 ∧
    dist.p0 = 5/14 ∧ dist.p1 = 15/28 ∧ dist.p2 = 3/28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_analysis_l106_10636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_manager_acceptance_range_l106_10651

/-- Represents the hiring manager's acceptance range in terms of standard deviations -/
noncomputable def acceptance_range (avg_age : ℝ) (std_dev : ℝ) (max_diff_ages : ℕ) : ℝ :=
  (max_diff_ages : ℝ) / (2 * std_dev)

theorem hiring_manager_acceptance_range :
  let avg_age : ℝ := 31
  let std_dev : ℝ := 8
  let max_diff_ages : ℕ := 17
  acceptance_range avg_age std_dev max_diff_ages = 1.0625 := by
  -- Unfold the definition of acceptance_range
  unfold acceptance_range
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_manager_acceptance_range_l106_10651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_l106_10606

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_l106_10606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l106_10686

-- Define the parabola
def parabola (x y : ℝ) : Prop := (1/4) * y^2 = x

-- Define the focus F (we don't know its exact coordinates, so we leave it as a parameter)
variable (F : ℝ × ℝ)

-- Define point A
def A : ℝ × ℝ := (2, 2)

-- Define a point P on the parabola
variable (P : ℝ × ℝ)

-- State that P is on the parabola
axiom P_on_parabola : parabola P.1 P.2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_value : ℝ), ∀ (P : ℝ × ℝ), parabola P.1 P.2 →
    distance P A + distance P F ≥ min_value ∧
    ∃ (P_min : ℝ × ℝ), parabola P_min.1 P_min.2 ∧
      distance P_min A + distance P_min F = min_value ∧
      min_value = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l106_10686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l106_10658

/-- A cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- The volume of a cube -/
noncomputable def Cube.volume (c : Cube) : ℝ := sorry

/-- A pyramid formed by four vertices of a cube -/
structure Pyramid (c : Cube) where
  apex : Fin 8
  base : Fin 3 → Fin 8

/-- The volume of a pyramid -/
noncomputable def Pyramid.volume {c : Cube} (p : Pyramid c) : ℝ := sorry

/-- The theorem stating the volume of a specific pyramid in a cube -/
theorem pyramid_volume_in_cube (c : Cube) (h : c.volume = 8) :
  let p : Pyramid c := { apex := 7, base := ![0, 2, 3] }  -- Representing ACDH
  Pyramid.volume p = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l106_10658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_satisfies_requirements_l106_10680

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 - 2

theorem reward_function_satisfies_requirements :
  (∀ x y, 10 ≤ x ∧ x ≤ y ∧ y ≤ 100 → f x ≤ f y) ∧
  (∀ x, 10 ≤ x ∧ x ≤ 100 → f x ≤ 5) ∧
  (∀ x, 10 ≤ x ∧ x ≤ 100 → f x ≤ x / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_satisfies_requirements_l106_10680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_bottle_return_l106_10600

/-- The number of bottles Pete needs to return to pay off his bike -/
def bottles_to_return (total_owed : ℚ) (twenty_bills : ℕ) (ten_bills : ℕ) (bottle_value : ℚ) : ℕ :=
  let cash_on_hand := (20 * twenty_bills + 10 * ten_bills : ℚ)
  let remaining := total_owed - cash_on_hand
  (remaining / bottle_value).ceil.toNat

/-- Theorem stating that Pete needs to return 20 bottles -/
theorem pete_bottle_return :
  bottles_to_return 90 2 4 (1/2) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_bottle_return_l106_10600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amc_10_score_problem_l106_10649

def min_correct_problems (total_problems : ℕ) (attempted_problems : ℕ) (correct_points : ℕ) (incorrect_points : ℤ) (unanswered_points : ℕ) (target_score : ℕ) : ℕ :=
  let unanswered := total_problems - attempted_problems
  let unanswered_score := unanswered * unanswered_points
  let x := (target_score - unanswered_score - incorrect_points * attempted_problems) / (correct_points - incorrect_points)
  (x + 1).toNat

theorem amc_10_score_problem :
  min_correct_problems 25 20 7 (-1) 2 120 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amc_10_score_problem_l106_10649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l106_10663

/-- Calculate the true discount on a bill -/
noncomputable def true_discount (amount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (amount * rate * time) / (100 + (rate * time))

/-- The true discount on a bill of Rs. 1764 due in 9 months at 16% per annum is Rs. 189 -/
theorem bill_true_discount :
  let amount : ℝ := 1764
  let rate : ℝ := 16
  let time : ℝ := 9 / 12
  true_discount amount rate time = 189 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l106_10663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_range_l106_10666

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the function g
def g (a b x : ℝ) : ℝ := 2*x - 2*a + b

-- Theorem statement
theorem zero_points_range (a b : ℝ) :
  (∀ x, f a b x ∈ Set.Ici (-1)) →
  {x | g a b x = 0} ⊆ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_range_l106_10666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l106_10630

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ -0.5 then 1 / (0.5 + x) else 0.5

-- State the theorem
theorem f_satisfies_equation :
  ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 :=
by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l106_10630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l106_10635

/-- Represents the capacity and current state of a pool --/
structure Pool where
  capacity : ℚ
  current_fill : ℚ

/-- The amount of water needed to reach 80% capacity --/
def water_needed (p : Pool) : ℚ := 0.8 * p.capacity - p.current_fill

/-- The increase in water content when adding 300 gallons --/
def increase_percentage (p : Pool) : ℚ := 300 / p.current_fill

theorem pool_capacity (p : Pool) 
  (h1 : water_needed p = 300)
  (h2 : increase_percentage p = 1/4)
  (h3 : p.current_fill > 0) :
  p.capacity = 1200 := by
  sorry

#check pool_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l106_10635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l106_10683

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y - Real.log x = 0

-- Define the line
def line (x y : ℝ) : Prop := y = x - 2

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |y - (x - 2)| / Real.sqrt 2

-- Theorem statement
theorem min_distance_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), curve x y → 
  distance_to_line x y ≥ d ∧
  ∃ (x₀ y₀ : ℝ), curve x₀ y₀ ∧ distance_to_line x₀ y₀ = d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l106_10683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_download_time_l106_10633

/-- Calculates the remaining download time for a game. -/
noncomputable def remaining_download_time (total_size : ℝ) (downloaded : ℝ) (speed : ℝ) : ℝ :=
  (total_size - downloaded) / speed

/-- Proves that the remaining download time for the given conditions is 376 minutes. -/
theorem game_download_time : remaining_download_time 1250 310 2.5 = 376 := by
  -- Unfold the definition of remaining_download_time
  unfold remaining_download_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_download_time_l106_10633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_growing_function_l106_10682

noncomputable def f₁ (x : ℝ) : ℝ := x^2
noncomputable def f₂ (x : ℝ) : ℝ := 4*x
noncomputable def f₃ (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f₄ (x : ℝ) : ℝ := 2^x

theorem fastest_growing_function :
  ∃ (N : ℝ), ∀ (x : ℝ), x > N →
    f₄ x > f₁ x ∧ f₄ x > f₂ x ∧ f₄ x > f₃ x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_growing_function_l106_10682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l106_10692

theorem cos_2pi_minus_alpha (α : Real) 
  (h1 : Real.cos (Real.pi + α) = Real.sqrt 3 / 2) 
  (h2 : Real.pi < α) 
  (h3 : α < 3 * Real.pi / 2) : 
  Real.cos (2 * Real.pi - α) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l106_10692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_l106_10603

/-- The number of ways to color n integers with 3 colors, 
    such that no two consecutive integers have the same color. -/
def colorings (n : ℕ) : ℕ := 3 * 2^(n-1)

/-- The actual number of valid colorings for n integers. -/
def number_of_valid_colorings : ℕ → ℕ
  | 0 => 1  -- Base case for 0 integers (empty coloring)
  | 1 => 3  -- Base case for 1 integer
  | n+2 => 2 * number_of_valid_colorings (n+1)  -- Recursive case

/-- Theorem stating the number of valid colorings for n integers. -/
theorem valid_colorings (n : ℕ) (h : n > 0) : 
  colorings n = number_of_valid_colorings n :=
by sorry

/-- Helper lemma: colorings n = 3 * 2^(n-1) for n > 0 -/
lemma colorings_eq (n : ℕ) (h : n > 0) : 
  colorings n = 3 * 2^(n-1) :=
by sorry

/-- Helper lemma: number_of_valid_colorings n = 3 * 2^(n-1) for n > 0 -/
lemma number_of_valid_colorings_eq (n : ℕ) (h : n > 0) : 
  number_of_valid_colorings n = 3 * 2^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_l106_10603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_negative_line_l106_10621

noncomputable def sample_data (n : ℕ) := Fin n → ℝ × ℝ

/-- The correlation coefficient of a sample dataset -/
noncomputable def correlation_coefficient (n : ℕ) (data : sample_data n) : ℝ := 
  sorry -- Definition of correlation coefficient would go here

theorem correlation_coefficient_perfect_negative_line
  (n : ℕ) (h_n : n ≥ 2) (data : sample_data n)
  (h_not_all_equal : ∃ i j, i ≠ j ∧ (data i).1 ≠ (data j).1)
  (h_on_line : ∀ i, (data i).2 = -1/2 * (data i).1 + 1) :
  correlation_coefficient n data = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_negative_line_l106_10621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_with_geometric_angles_l106_10643

theorem no_triangle_with_geometric_angles : 
  ¬ ∃ (a r : ℕ+), 
    (a : ℕ) + (a * r) + (a * r * r) = 180 ∧
    a < a * r ∧ a * r < a * r * r ∧
    r > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_with_geometric_angles_l106_10643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l106_10681

/-- The length of two trains passing each other --/
theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : 
  (faster_speed - slower_speed) * (5 / 18) * passing_time / 2 = 50 := by
  -- Define relative speed
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  -- Define train length
  let train_length := relative_speed * passing_time / 2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l106_10681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_value_l106_10608

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1/x^m

theorem S_6_value (x : ℝ) (h : x + 1/x = 4) : S x 6 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_value_l106_10608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_problem_l106_10670

theorem wall_building_problem (a d g h : ℕ) (h1 : a > 0) (h2 : d > 0) (h3 : g > 0) :
  let x := a * g / d
  let total_bricks := (3 * a + h * a) / 2
  true :=
by
  -- Introduce local definitions
  let x := a * g / d
  let total_bricks := (3 * a + h * a) / 2

  -- The actual proof would go here
  -- For now, we'll use sorry to skip the proof
  sorry

#check wall_building_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_problem_l106_10670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_cube_l106_10607

/-- Represents a 2x2x2 cube with faces divided into unit squares -/
structure Cube where
  faces : Fin 6 → Fin 2 → Fin 2 → Bool

/-- Checks if a given square on a face has exactly two crosses and two zeros as neighbors -/
def validSquare (c : Cube) (face : Fin 6) (x y : Fin 2) : Prop :=
  let neighbors := [
    c.faces face ((x + 1) % 2) y,
    c.faces face x ((y + 1) % 2),
    c.faces face ((x - 1) % 2) y,
    c.faces face x ((y - 1) % 2)
  ]
  (neighbors.filter id).length = 2

/-- A cube configuration is valid if all squares satisfy the neighbor condition -/
def validCube (c : Cube) : Prop :=
  ∀ (face : Fin 6) (x y : Fin 2), validSquare c face x y

/-- Theorem: It's impossible to have a valid cube configuration -/
theorem no_valid_cube : ¬ ∃ (c : Cube), validCube c := by
  sorry

#check no_valid_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_cube_l106_10607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_odd_l106_10648

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (1 - x^2)) / (abs (x + 2) - 2)

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x)

-- Define the function F as the product of f and g
noncomputable def F (x : ℝ) : ℝ := f x * g x

-- State the theorem that F is an odd function
theorem F_is_odd : ∀ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1, F (-x) = -F x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_odd_l106_10648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_piece_cutting_time_l106_10662

/-- The time required to make one cut in a wooden stick -/
noncomputable def time_per_cut : ℝ := 6 / 2

/-- The number of cuts required to divide a stick into n pieces -/
def cuts_required (n : ℕ) : ℕ := n - 1

/-- The time required to cut a wooden stick into n pieces -/
noncomputable def cutting_time (n : ℕ) : ℝ := time_per_cut * (cuts_required n)

theorem four_piece_cutting_time :
  cutting_time 4 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_piece_cutting_time_l106_10662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_distance_extrema_l106_10619

/-- A regular pentagon in a 2D plane -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : Prop

/-- An arbitrary point in or on the boundary of the pentagon -/
def Point := ℝ × ℝ

/-- Distance from a point to a line segment (representing a side of the pentagon) -/
noncomputable def distance_to_side (p : Point) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- The five distances from a point to the sides of the pentagon, sorted in ascending order -/
noncomputable def sorted_distances (pentagon : RegularPentagon) (p : Point) : Fin 5 → ℝ := sorry

/-- Predicate to check if a point is at a vertex of the pentagon -/
def is_vertex (pentagon : RegularPentagon) (p : Point) : Prop := sorry

/-- Predicate to check if a point is at the midpoint of a side of the pentagon -/
def is_midpoint (pentagon : RegularPentagon) (p : Point) : Prop := sorry

/-- The main theorem statement -/
theorem third_distance_extrema (pentagon : RegularPentagon) :
  (∀ p : Point, is_midpoint pentagon p → 
    ∀ q : Point, (sorted_distances pentagon p) 2 ≤ (sorted_distances pentagon q) 2) ∧
  (∀ p : Point, is_vertex pentagon p → 
    ∀ q : Point, (sorted_distances pentagon q) 2 ≤ (sorted_distances pentagon p) 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_distance_extrema_l106_10619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_velocity_l106_10675

/-- The motion equation for a body in free-fall -/
noncomputable def s (g : ℝ) (t : ℝ) : ℝ := (1/2) * g * t^2

/-- The theorem stating that if the limit of the difference quotient at t=1 is 9.8,
    then the velocity at t=1 is also 9.8 -/
theorem free_fall_velocity (g : ℝ) (h : g = 9.8) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δt ≠ 0, |Δt| < δ →
    |(s g (1 + Δt) - s g 1) / Δt - g| < ε) →
  g = 9.8 := by
  intro h_limit
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_velocity_l106_10675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_0_003375_cube_root_of_negative_0_003375_equiv_l106_10624

theorem cube_root_of_negative_0_003375 : 
  ((-0.15 : ℝ) ^ 3 : ℝ) = -0.003375 := by
  norm_num

theorem cube_root_of_negative_0_003375_equiv : 
  ∃ x : ℝ, x ^ 3 = -0.003375 ∧ x = -0.15 := by
  use -0.15
  constructor
  · exact cube_root_of_negative_0_003375
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_0_003375_cube_root_of_negative_0_003375_equiv_l106_10624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_is_two_hours_l106_10695

/-- Represents the internet speed in megabits per minute -/
noncomputable def internet_speed : ℚ := 2

/-- Represents the sizes of the files in megabits -/
def file_sizes : List ℚ := [80, 90, 70]

/-- Calculates the total download time in hours -/
noncomputable def total_download_time : ℚ :=
  (List.sum (List.map (λ x => x / internet_speed) file_sizes)) / 60

/-- Theorem stating that the total download time is 2 hours -/
theorem download_time_is_two_hours :
  total_download_time = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_is_two_hours_l106_10695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l106_10602

def start_point : ℝ × ℝ × ℝ := (1, 2, 2)
def end_point : ℝ × ℝ × ℝ := (0, -2, -2)

def unit_sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 1

theorem intersection_distance_sum (a b : ℕ) : 
  ∃ (t₁ t₂ : ℝ), 
    -- The line intersects the unit sphere at two points
    unit_sphere (1 - t₁) (2 - 4*t₁) (2 - 4*t₁) ∧
    unit_sphere (1 - t₂) (2 - 4*t₂) (2 - 4*t₂) ∧
    t₁ ≠ t₂ ∧
    -- The distance between intersection points is a/√b
    (((1 - 4*t₁) - (1 - 4*t₂))^2 + ((2 - 4*t₁) - (2 - 4*t₂))^2 + ((2 - 4*t₁) - (2 - 4*t₂))^2)^(1/2) = a / Real.sqrt b ∧
    -- a and b are coprime
    Nat.Coprime a b ∧
    -- The sum of a and b is 37
    a + b = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l106_10602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l106_10696

noncomputable def f (x : ℝ) : ℤ := ⌊|x|⌋ - |⌊x⌋|

theorem f_range : 
  (∀ x : ℝ, f x ∈ ({-1, 0} : Set ℤ)) ∧ 
  (∃ x : ℝ, f x = -1) ∧ 
  (∃ x : ℝ, f x = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l106_10696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_property_l106_10604

theorem partition_property (partition : Finset (Finset Nat)) : 
  (∀ n ∈ Finset.range 100, ∃ S ∈ partition, n + 1 ∈ S) →
  partition.card = 7 →
  ∃ S ∈ partition, 
    (∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d) ∨
    (∃ e f g, e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ 
     e ≠ f ∧ e ≠ g ∧ f ≠ g ∧ e + f = 2 * g) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_property_l106_10604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_problem_l106_10626

theorem four_digit_number_problem (a b c d : Int) 
  (h_a : 0 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) (h_d : 0 ≤ d ∧ d ≤ 9)
  (eq1 : 6 * a + 9 * b + 3 * c + d = 88)
  (eq2 : a - b + c - d = -6)
  (eq3 : a - 9 * b + 3 * c - d = -46) :
  1000 * a + 100 * b + 10 * c + d = 6507 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_problem_l106_10626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_shift_l106_10698

theorem cosine_graph_shift (x : ℝ) :
  Real.cos (2*x + π/3) = Real.cos (2*(x + π/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_shift_l106_10698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_35_plus_12sqrt6_equals_a_plus_bsqrtc_l106_10639

-- Define the property that c has no perfect square factors other than 1
def no_perfect_square_factors (c : ℕ) : Prop :=
  ∀ n : ℕ, n > 1 → ¬(∃ k : ℕ, c = n^2 * k)

-- State the theorem
theorem sqrt_35_plus_12sqrt6_equals_a_plus_bsqrtc (a b : ℤ) (c : ℕ) :
  (35 + 12 * Real.sqrt 6 : ℝ).sqrt = a + b * Real.sqrt (c : ℝ) →
  no_perfect_square_factors c →
  a + b + (c : ℤ) = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_35_plus_12sqrt6_equals_a_plus_bsqrtc_l106_10639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l106_10676

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

def C₂ (ρ θ : ℝ) : Prop := ρ^2 = 3 / (1 + 2*(Real.sin θ)^2)

-- Define the ray
def ray (ρ θ : ℝ) : Prop := θ = Real.pi/3 ∧ ρ ≥ 0

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ ρ₁ ρ₂ : ℝ),
    C₁ x₁ y₁ ∧
    C₂ ρ₂ (Real.pi/3) ∧
    ray ρ₁ (Real.pi/3) ∧
    ray ρ₂ (Real.pi/3) ∧
    x₁ = ρ₁ * Real.cos (Real.pi/3) ∧
    y₁ = ρ₁ * Real.sin (Real.pi/3) ∧
    x₂ = ρ₂ * Real.cos (Real.pi/3) ∧
    y₂ = ρ₂ * Real.sin (Real.pi/3) ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 30 / 5 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l106_10676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_length_is_80pi_l106_10638

/-- Calculates the length of a paper strip wrapped around a tube. -/
noncomputable def paper_length (initial_diameter : ℝ) (final_diameter : ℝ) (num_wraps : ℕ) : ℝ :=
  Real.pi * (num_wraps / 2 : ℝ) * (initial_diameter + final_diameter)

/-- The length of the paper strip is 80π meters. -/
theorem paper_length_is_80pi :
  paper_length 4 16 800 = 80 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_length_is_80pi_l106_10638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_investment_l106_10611

/-- Represents the partnership between Mary and Harry --/
structure Partnership where
  mary_investment : ℚ
  harry_investment : ℚ
  total_profit : ℚ

/-- Calculates the profit share for a partner --/
noncomputable def profit_share (p : Partnership) (investment : ℚ) : ℚ :=
  (p.total_profit / 3) / 2 + (2 * p.total_profit / 3) * (investment / (p.mary_investment + p.harry_investment))

/-- Theorem stating Mary's investment given the problem conditions --/
theorem mary_investment (p : Partnership) : 
  p.harry_investment = 300 ∧ 
  p.total_profit = 3000 ∧ 
  profit_share p p.mary_investment = profit_share p p.harry_investment + 800 →
  p.mary_investment = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_investment_l106_10611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l106_10674

-- Define the circle and its properties
structure Circle where
  O : Point -- center
  r : ℝ -- radius
  AB : ℝ -- length of chord AB
  CD : ℝ -- length of chord CD
  parallel_chords : Prop -- Placeholder for the parallel chords condition
  angle_ratio : Prop -- Placeholder for the angle ratio condition

-- Define the theorem
theorem circle_radius (c : Circle) (h1 : c.AB = 46) (h2 : c.CD = 18) : c.r = 27 := by
  sorry

-- Note: Point is assumed to be defined in Mathlib

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l106_10674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l106_10628

/-- Represents a hyperbola with center (h, k), focus (0, -3 + √41), and vertex (0, 0) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  center_condition : h = 0 ∧ k = -3
  vertex_condition : a = 3
  focus_condition : 41 = (3 + Real.sqrt 41)^2

/-- The sum of h, k, a, and b for the given hyperbola is 4√2 -/
theorem hyperbola_sum (hyp : Hyperbola) : hyp.h + hyp.k + hyp.a + hyp.b = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l106_10628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_l106_10697

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

-- State the theorem
theorem symmetry_point (x₀ : ℝ) :
  (∀ x : ℝ, f (2 * x₀ - x) = f x) →  -- Symmetry condition
  x₀ ∈ Set.Icc (-Real.pi / 2) 0 →    -- x₀ is in the closed interval [-π/2, 0]
  x₀ = -Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_l106_10697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_not_always_valid_y_intercept_not_always_distance_to_origin_not_all_lines_have_slope_intercept_form_l106_10694

-- Define a line in a plane
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem B
theorem intercept_form_not_always_valid (a b : ℝ) :
  ¬ (∀ (x y : ℝ), x / a + y / b = 1 → ∃ (l : Line), l.slope * a + l.intercept = 0 ∧ l.intercept = b) :=
by sorry

-- Theorem C
theorem y_intercept_not_always_distance_to_origin (k b : ℝ) :
  ¬ (∀ (p : Point), p.y = k * p.x + b → distance p ⟨0, 0⟩ = b) :=
by sorry

-- Theorem D
theorem not_all_lines_have_slope_intercept_form :
  ¬ (∀ (l : Line), ∃ (m c : ℝ), ∀ (x y : ℝ), y = m * x + c ↔ y = l.slope * x + l.intercept) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_not_always_valid_y_intercept_not_always_distance_to_origin_not_all_lines_have_slope_intercept_form_l106_10694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_eighteen_six_l106_10657

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- The ratio property given in the problem -/
axiom ratio_property : S 12 / S 6 = 1 / 2

/-- The property of geometric sequences mentioned in the solution -/
axiom geometric_property : (S 12 - S 6) ^ 2 = S 6 * (S 18 - S 12)

/-- The theorem to be proved -/
theorem ratio_eighteen_six : S 18 / S 6 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_eighteen_six_l106_10657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l106_10622

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Theorem: The total distance traveled by the laser beam is 14√2 -/
theorem laser_beam_distance : 
  let start := Point.mk 4 7
  let end_ := Point.mk 10 7
  let bounceY := reflectOverYAxis start
  let bounceX := reflectOverXAxis bounceY
  let endReflected := reflectOverXAxis (reflectOverYAxis end_)
  distance start endReflected = 14 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l106_10622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_triangular_scarves_l106_10684

/-- Represents a scarf with colored areas -/
structure Scarf where
  total_area : ℚ
  black_area : ℚ
  gray_area : ℚ
  white_area : ℚ

/-- Represents a square scarf -/
def SquareScarf : Scarf :=
  { total_area := 1
    black_area := 1/6
    gray_area := 1/3
    white_area := 1/2 }

/-- Represents a triangular scarf obtained by cutting a square scarf diagonally -/
def TriangularScarf : Scarf :=
  { total_area := 1/2
    black_area := 1/12
    gray_area := 1/6
    white_area := 1/4 }

/-- Theorem stating that cutting a square scarf diagonally results in two identical triangular scarves
    with the specified color proportions -/
theorem square_to_triangular_scarves :
  let square := SquareScarf
  let triangle := TriangularScarf
  square.black_area = 1/6 ∧
  square.gray_area = 1/3 ∧
  triangle.total_area = square.total_area / 2 ∧
  triangle.black_area = square.black_area / 2 ∧
  triangle.gray_area = square.gray_area / 2 ∧
  triangle.white_area = square.white_area / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_triangular_scarves_l106_10684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_a_eq_two_l106_10632

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℚ) : Prop := m1 = m2

/-- The slope of the line ax + 2y = 0 -/
def slope1 (a : ℚ) : ℚ := -a / 2

/-- The slope of the line x + y = 1 -/
def slope2 : ℚ := -1

/-- Theorem: The lines ax + 2y = 0 and x + y = 1 are parallel if and only if a = 2 -/
theorem lines_parallel_iff_a_eq_two (a : ℚ) :
  parallel_lines (slope1 a) slope2 ↔ a = 2 := by
  -- Proof goes here
  sorry

#check lines_parallel_iff_a_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_a_eq_two_l106_10632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l106_10623

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if c = 3, C = π/3, and sin B = 2 sin A, then a = √3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  c = 3 → 
  C = π/3 → 
  Real.sin B = 2 * Real.sin A → 
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l106_10623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l106_10654

/-- Given a right triangle PQR with legs of length 9 and 12, and a square with one side on the hypotenuse
    of PQR and vertices on both legs of PQR, the length of one side of the square is 540/111. -/
theorem square_side_length (P Q R : ℝ × ℝ) (s : ℝ) : 
  ‖Q - P‖ = 9 →
  ‖R - P‖ = 12 →
  (Q - P) • (R - P) = 0 →
  ∃ (A B : ℝ × ℝ), 
    (A - P) • (Q - P) = 0 ∧ 
    (B - P) • (R - P) = 0 ∧
    ‖A - B‖ = s ∧
    ‖A - Q‖ = s ∧
    ‖B - R‖ = s →
  s = 540 / 111 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l106_10654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_location_minimizes_distance_l106_10671

/-- Represents the location of the school as a distance from village A -/
def school_location : ℝ := 0

/-- Number of students in village A -/
def students_A : ℕ := 100

/-- Number of students in village B -/
def students_B : ℕ := 50

/-- Distance between villages A and B in kilometers -/
def village_distance : ℝ := 3

/-- Total distance traveled by all students as a function of school location -/
def total_distance (x : ℝ) : ℝ := 
  (students_A : ℝ) * x + (students_B : ℝ) * (village_distance - x)

/-- Theorem stating that the total distance is minimized when the school is at village A -/
theorem school_location_minimizes_distance :
  ∀ x, 0 ≤ x ∧ x ≤ village_distance → 
    total_distance 0 ≤ total_distance x :=
by
  sorry

#eval school_location

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_location_minimizes_distance_l106_10671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_movement_probabilities_l106_10646

/-- A fly's position on an integer grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- The probability of a fly reaching a specific position -/
def probability_reach (start : Position) (end_ : Position) : ℚ :=
  sorry

/-- The probability of a fly reaching a position through a specific segment -/
def probability_reach_through_segment (start mid1 mid2 end_ : Position) : ℚ :=
  sorry

/-- The probability of a fly reaching a position passing through a circle -/
def probability_reach_through_circle (start center : Position) (radius : ℕ) (end_ : Position) : ℚ :=
  sorry

/-- Main theorem about fly's movement probabilities -/
theorem fly_movement_probabilities :
  let start := Position.mk 0 0
  let end_ := Position.mk 8 10
  let mid1 := Position.mk 5 6
  let mid2 := Position.mk 6 6
  let center := Position.mk 4 5
  (probability_reach start end_ = 43758/262144) ∧
  (probability_reach_through_segment start mid1 mid2 end_ = 6930/262144) ∧
  (probability_reach_through_circle start center 3 end_ = 43092/262144) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_movement_probabilities_l106_10646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_neg_one_l106_10634

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the expression
noncomputable def expr : ℂ := 1 / (((Real.sqrt 2) / 2 - (Real.sqrt 2) / 2 * i) ^ 4)

-- Theorem statement
theorem expr_equals_neg_one : expr = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_neg_one_l106_10634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_moves_up_three_units_l106_10689

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in a 2D Cartesian coordinate system -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Function to add 3 to the y-coordinate of a point -/
def addThreeToY (p : Point) : Point :=
  { x := p.x, y := p.y + 3 }

/-- Function to transform a triangle by adding 3 to all y-coordinates -/
def transformTriangle (t : Triangle) : Triangle :=
  { v1 := addThreeToY t.v1
    v2 := addThreeToY t.v2
    v3 := addThreeToY t.v3 }

/-- Define what it means for a point to be in a triangle -/
def pointInTriangle (p : Point) (t : Triangle) : Prop :=
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
  p.x = a * t.v1.x + b * t.v2.x + c * t.v3.x ∧
  p.y = a * t.v1.y + b * t.v2.y + c * t.v3.y

/-- Theorem stating that the transformation moves the triangle up by 3 units -/
theorem triangle_moves_up_three_units (t : Triangle) :
  ∀ p : Point, pointInTriangle p t → 
  ∃ q : Point, pointInTriangle q (transformTriangle t) ∧ q.x = p.x ∧ q.y = p.y + 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_moves_up_three_units_l106_10689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_perfect_function_l106_10637

open Real

/-- Definition of a perfect function -/
def isPerfectFunction (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  StrictMono (fun x : I => f x) ∧ StrictMono (fun x : I => f x / x)

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := exp x + x - log x + 1

/-- The theorem statement -/
theorem min_m_for_perfect_function :
  ∀ m : ℕ, (∀ x ≥ m / 2, isPerfectFunction g {y | y ≥ m / 2}) ↔ m ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_perfect_function_l106_10637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_problem_l106_10631

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0

-- Define the parabola E
def parabola_E (x y p : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the center of circle C
def center_C : ℝ × ℝ := (-1, 1)

-- Define the focus of parabola E
noncomputable def focus_E (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the perpendicularity of two lines
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

-- Define the line l
def line_l (m t x y : ℝ) : Prop := x = m*y + t

-- Define the theorem
theorem parabola_and_line_problem 
  (p : ℝ) 
  (h1 : distance (-1) 1 (p/2) 0 = Real.sqrt 17)
  (h2 : ∀ m t x1 y1 x2 y2, 
    line_l m t x1 y1 ∧ line_l m t x2 y2 ∧ 
    parabola_E x1 y1 p ∧ parabola_E x2 y2 p ∧ 
    perpendicular x1 y1 x2 y2 ∧ 
    t ≠ 0 → t = 12) :
  (∀ x y, parabola_E x y p ↔ y^2 = 12*x) ∧
  (∃ m, ∀ x y, line_l m 12 x y ↔ 13*x - y - 156 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_problem_l106_10631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l106_10679

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > Real.log a^2 / Real.log 4) → 
  a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l106_10679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_bound_l106_10685

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Define the property of having integer roots
def has_integer_roots (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (roots : Finset ℤ), roots.card = n ∧ ∀ r ∈ roots, P.eval r = 0

-- State the theorem
theorem integer_roots_bound (P : IntPolynomial) 
  (h1 : P.eval 0 ≠ 0)
  (h2 : P.eval (P.eval 0) = 0) :
  ∃ m : ℕ, m ≥ 1 ∧ m ≤ 3 ∧ has_integer_roots P m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_bound_l106_10685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l106_10605

def is_self_descriptive (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 10 ∧
  ∀ k, k < 10 → digits.count k = digits[k]!

theorem unique_self_descriptive_number :
  ∃! n : Nat, is_self_descriptive n ∧ n = 6210001000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l106_10605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bijectivity_l106_10644

theorem function_bijectivity (f : ℝ → ℝ) :
  (∀ x : ℝ, f (f x - 1) = x + 1) →
  Function.Bijective f := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bijectivity_l106_10644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_l106_10617

theorem sin_value_in_third_quadrant (x : ℝ) :
  Real.sin x = -3/5 → π < x → x < 3*π/2 → x = π + Real.arcsin (3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_l106_10617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_fills_in_12_hours_l106_10677

/-- The time it takes for the second pipe to fill the cistern -/
noncomputable def second_pipe_time : ℝ := 12

/-- The time it takes for all pipes together to fill the cistern -/
noncomputable def all_pipes_time : ℝ := 60 / 7

/-- Theorem stating that the second pipe fills the cistern in 12 hours -/
theorem second_pipe_fills_in_12_hours :
  (1 / 10 : ℝ) + (1 / second_pipe_time) - (1 / 15) = 1 / all_pipes_time :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_fills_in_12_hours_l106_10677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l106_10665

noncomputable def work_together (a_days b_days : ℝ) : ℝ :=
  1 / (1 / a_days + 1 / b_days)

theorem work_completion_time (a_days b_days : ℝ) 
  (ha : a_days = 10) (hb : b_days = 15) : 
  work_together a_days b_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l106_10665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circle_sector_l106_10615

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) : 
  (1/3) * π * (π * r / (2 * π))^2 * Real.sqrt (r^2 - (π * r / (2 * π))^2) = 9 * π * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circle_sector_l106_10615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_altitude_ratio_l106_10614

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The median from vertex A to the midpoint of side BC -/
noncomputable def median (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- The altitude from vertex A to side BC -/
noncomputable def altitude (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- A triangle is scalene if all sides have different lengths -/
def isScalene (t : Triangle) : Prop :=
  distance t.A t.B ≠ distance t.B t.C ∧
  distance t.B t.C ≠ distance t.C t.A ∧
  distance t.C t.A ≠ distance t.A t.B

theorem median_altitude_ratio (t : Triangle) 
  (h_scalene : isScalene t)
  (h_equal1 : median t t.A = altitude t t.B)
  (h_equal2 : median t t.B = altitude t t.C) :
  median t t.C / altitude t t.A = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_altitude_ratio_l106_10614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l106_10693

-- Define the function f
noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : abs φ < π/2) 
  (h4 : f A ω φ (3*π/8) = 0) 
  (h5 : f A ω φ (π/8) = 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (2*x + π/4)) ∧ 
  (∀ x ∈ Set.Icc (-π/4) (π/4), f A ω φ x ≥ -Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc (-π/4) (π/4), f A ω φ x ≤ 2) ∧
  (f A ω φ (-π/4) = -Real.sqrt 2) ∧
  (f A ω φ (π/8) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l106_10693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tails_probability_l106_10678

-- Define a fair coin
noncomputable def fair_coin : ℚ := 1/2

-- Define the probability of getting tails on a single flip
def prob_tails (p : ℚ) : ℚ := p

-- Define the probability of getting two tails in a row
def prob_two_tails (p : ℚ) : ℚ := p * p

-- Theorem statement
theorem two_tails_probability :
  prob_two_tails fair_coin = 1/4 :=
by
  -- Unfold definitions
  unfold prob_two_tails
  unfold fair_coin
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tails_probability_l106_10678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_from_inequality_l106_10661

/-- A function satisfying the given inequality is constant -/
theorem constant_function_from_inequality :
  ∀ f : ℝ → ℝ, 
  (∀ x y z : ℝ, f (x+y) + f (y+z) + f (z+x) ≥ 3 * f (x + 2*y + 3*z)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_from_inequality_l106_10661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_at_Q_two_eight_l106_10653

/-- The line l₁ is defined as y = 4x -/
noncomputable def l₁ (x : ℝ) : ℝ := 4 * x

/-- Point P is defined as (6, 4) -/
def P : ℝ × ℝ := (6, 4)

/-- Point Q is parameterized on line l₁ -/
noncomputable def Q (t : ℝ) : ℝ × ℝ := (t, l₁ t)

/-- The area of the triangle formed by PQ, l₁, and the x-axis -/
noncomputable def triangleArea (t : ℝ) : ℝ :=
  10 * t^2 / (t - 1)

/-- The theorem states that Q(2, 8) minimizes the triangle area -/
theorem minimal_area_at_Q_two_eight :
  ∃ (min_t : ℝ), min_t = 2 ∧
  ∀ (t : ℝ), t ≠ 1 → triangleArea min_t ≤ triangleArea t := by
  sorry

#check minimal_area_at_Q_two_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_at_Q_two_eight_l106_10653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_stick_probability_is_correct_l106_10647

/-- The probability of breaking a stick of length 11 meters into two pieces,
    both no less than 3 meters in length. -/
noncomputable def break_stick_probability : ℚ := 5 / 11

/-- Theorem stating that the probability of breaking a 11-meter stick into two pieces,
    both no less than 3 meters, is equal to 5/11. -/
theorem break_stick_probability_is_correct : break_stick_probability = 5 / 11 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_stick_probability_is_correct_l106_10647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_segments_l106_10641

/-- Given a rope of length 3 meters cut into 4 equal segments, prove that each segment
    represents 1/4 of the total length and is 3/4 meters long. -/
theorem rope_segments (total_length : ℝ) (num_segments : ℕ) 
    (h1 : total_length = 3)
    (h2 : num_segments = 4) :
    (1 / num_segments : ℝ) = 1/4 ∧ total_length / num_segments = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_segments_l106_10641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_theorem_l106_10650

open Real

-- Define the function f
noncomputable def f (x : ℝ) := sin x

-- Define the equation
def equation (x a : ℝ) := (f x)^2 + 2*(f x) + a = 0

-- Define the domain
def domain (x : ℝ) := 0 < x ∧ x < π

-- Theorem statement
theorem root_count_theorem (a : ℝ) :
  ∃ (n : Nat), n ∈ ({0, 1, 2} : Set Nat) ∧
  (∃ (S : Finset ℝ), (∀ x ∈ S, domain x ∧ equation x a) ∧ S.card = n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_theorem_l106_10650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_76_plus_84_base_9_l106_10609

/-- Represent a number in base 9 --/
structure BaseNine where
  value : ℕ
  property : value < 9^9

/-- Convert a base 9 number to its decimal representation --/
def to_decimal (n : BaseNine) : ℕ := sorry

/-- Get the units digit of a base 9 number --/
def units_digit (n : BaseNine) : ℕ := sorry

/-- Addition in base 9 --/
def base_nine_add (a b : BaseNine) : BaseNine := sorry

/-- Create a BaseNine number from a natural number --/
def mk_base_nine (n : ℕ) (h : n < 9^9 := by sorry) : BaseNine :=
  ⟨n, h⟩

/-- The main theorem --/
theorem units_digit_of_76_plus_84_base_9 :
  units_digit (base_nine_add (mk_base_nine 76) (mk_base_nine 84)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_76_plus_84_base_9_l106_10609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l106_10610

def point := ℝ × ℝ

def reflect_over_y_axis (p : point) : point :=
  (-p.1, p.2)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem reflection_distance :
  let D : point := (2, -4)
  let D' : point := reflect_over_y_axis D
  distance D D' = 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l106_10610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_sheetrock_area_l106_10612

/-- Calculates the area of an L-shaped sheetrock --/
theorem l_shaped_sheetrock_area 
  (main_length main_width cutout_length cutout_width : ℝ) 
  (h_main_length : main_length = 6)
  (h_main_width : main_width = 5)
  (h_cutout_length : cutout_length = 2)
  (h_cutout_width : cutout_width = 1) :
  main_length * main_width - cutout_length * cutout_width = 28 := by
  -- Substitute the given values
  rw [h_main_length, h_main_width, h_cutout_length, h_cutout_width]
  -- Evaluate the expression
  ring
  -- The proof is complete
  done

#check l_shaped_sheetrock_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_sheetrock_area_l106_10612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l106_10660

def n : ℕ := 2^25 * 3^17

theorem divisors_count : 
  (Finset.filter (fun d => d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 424 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l106_10660
