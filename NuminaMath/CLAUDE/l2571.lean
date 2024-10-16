import Mathlib

namespace NUMINAMATH_CALUDE_power_of_power_l2571_257133

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2571_257133


namespace NUMINAMATH_CALUDE_segment_length_segment_length_is_ten_l2571_257108

theorem segment_length : ℝ → Prop :=
  fun length => ∃ x₁ x₂ : ℝ,
    (|x₁ - Real.sqrt 25| = 5) ∧
    (|x₂ - Real.sqrt 25| = 5) ∧
    (x₁ ≠ x₂) ∧
    (length = |x₁ - x₂|) ∧
    (length = 10)

-- The proof goes here
theorem segment_length_is_ten : segment_length 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_segment_length_is_ten_l2571_257108


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l2571_257107

/-- Percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 12

/-- Mass of the first alloy in kg -/
def mass_1 : ℝ := 15

/-- Mass of the second alloy in kg -/
def mass_2 : ℝ := 35

/-- Percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 10.6

/-- Percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 10

theorem chromium_percentage_calculation :
  (chromium_percentage_1 / 100 * mass_1 + chromium_percentage_2 / 100 * mass_2) / (mass_1 + mass_2) * 100 = chromium_percentage_new :=
sorry

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l2571_257107


namespace NUMINAMATH_CALUDE_projection_shape_theorem_l2571_257184

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a point in 3D space -/
structure Point

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the projection of a point onto a plane -/
def project (p : Point) (plane : Plane) : Point :=
  sorry

/-- Determines if a point is outside a plane -/
def isOutside (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a point is on a plane -/
def isOn (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  sorry

/-- Represents the shape formed by projections -/
inductive ProjectionShape
  | LineSegment
  | ObtuseTriangle

theorem projection_shape_theorem (ABC : Triangle) (a : Plane) :
  isRightTriangle ABC →
  isOn ABC.B a →
  isOn ABC.C a →
  isOutside ABC.A a →
  (project ABC.A a ≠ ABC.B ∧ project ABC.A a ≠ ABC.C) →
  (∃ shape : ProjectionShape, 
    (shape = ProjectionShape.LineSegment ∨ shape = ProjectionShape.ObtuseTriangle)) :=
  sorry

end NUMINAMATH_CALUDE_projection_shape_theorem_l2571_257184


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2571_257167

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (x y : ℝ) :
  symmetric_wrt_origin (x, -2) (3, y) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2571_257167


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2571_257181

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 3 = 1

-- Define a focus of the hyperbola
def focus (F : ℝ × ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola x y ∧ F = (Real.sqrt 6, 0)

-- Define an asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem focus_to_asymptote_distance (F : ℝ × ℝ) :
  focus F → (∃ (x y : ℝ), asymptote x y ∧ 
    Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) / Real.sqrt (1 + 1^2) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2571_257181


namespace NUMINAMATH_CALUDE_vector_sum_length_l2571_257187

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_length (a b : ℝ × ℝ) : 
  angle_between a b = π / 3 →
  a = (3, -4) →
  Real.sqrt ((a.1)^2 + (a.2)^2) = 2 →
  Real.sqrt (((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2)) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_length_l2571_257187


namespace NUMINAMATH_CALUDE_quartic_at_one_equals_three_l2571_257153

/-- Horner's method for evaluating a quartic polynomial at x = 1 -/
def horner_quartic (a₄ a₃ a₂ a₁ a₀ : ℤ) : ℤ :=
  ((((1 * a₄ + a₃) * 1 + a₂) * 1 + a₁) * 1 + a₀)

/-- The given quartic polynomial evaluated at x = 1 equals 3 -/
theorem quartic_at_one_equals_three :
  horner_quartic 1 (-7) (-9) 11 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quartic_at_one_equals_three_l2571_257153


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2571_257116

/-- Given a parallelogram with opposite vertices (1, -5) and (11, 7),
    the point of intersection of the diagonals is (6, 1). -/
theorem parallelogram_diagonal_intersection :
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (11, 7)
  let midpoint : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  midpoint = (6, 1) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2571_257116


namespace NUMINAMATH_CALUDE_function_identically_zero_l2571_257157

/-- A function satisfying f(a · b) = a f(b) + b f(a) and |f(x)| ≤ 1 is identically zero -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) 
  (h2 : ∀ x : ℝ, |f x| ≤ 1) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_identically_zero_l2571_257157


namespace NUMINAMATH_CALUDE_inequality_proof_l2571_257144

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_abc : a * b + b * c + c * a = a * b * c) :
  (a^a * (b^2 + c^2)) / ((a^a - 1)^2) +
  (b^b * (c^2 + a^2)) / ((b^b - 1)^2) +
  (c^c * (a^2 + b^2)) / ((c^c - 1)^2) ≥
  18 * ((a + b + c) / (a * b * c - 1))^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2571_257144


namespace NUMINAMATH_CALUDE_expansive_sequence_existence_l2571_257164

def expansive (a : ℕ → ℝ) : Prop :=
  ∀ i j : ℕ, i < j → |a i - a j| ≥ 1 / j

theorem expansive_sequence_existence (C : ℝ) :
  (C > 0 ∧ ∃ a : ℕ → ℝ, expansive a ∧ ∀ n, 0 ≤ a n ∧ a n ≤ C) ↔ C ≥ 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_expansive_sequence_existence_l2571_257164


namespace NUMINAMATH_CALUDE_round_trip_speed_l2571_257101

/-- Proves that given specific conditions for a round trip, the outward speed is 3 km/hr -/
theorem round_trip_speed (return_speed : ℝ) (total_time : ℝ) (one_way_distance : ℝ)
  (h1 : return_speed = 2)
  (h2 : total_time = 5)
  (h3 : one_way_distance = 6) :
  (one_way_distance / (total_time - one_way_distance / return_speed) = 3) :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l2571_257101


namespace NUMINAMATH_CALUDE_octal_to_binary_conversion_l2571_257188

/-- Converts an octal number to decimal -/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary -/
def decimal_to_binary (decimal : ℕ) : ℕ := sorry

/-- The octal representation of the number -/
def octal_num : ℕ := 135

/-- The binary representation of the number -/
def binary_num : ℕ := 1011101

theorem octal_to_binary_conversion :
  decimal_to_binary (octal_to_decimal octal_num) = binary_num := by sorry

end NUMINAMATH_CALUDE_octal_to_binary_conversion_l2571_257188


namespace NUMINAMATH_CALUDE_vector_dot_product_and_trigonometry_l2571_257100

/-- Given vectors a and b, and a function f, prove the following statements. -/
theorem vector_dot_product_and_trigonometry 
  (a : ℝ × ℝ) 
  (b : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (h_a : a = (Real.sqrt 3, 1))
  (h_b : ∀ x, b x = (Real.cos x, Real.sin x))
  (h_f : ∀ x, f x = a.1 * (b x).1 + a.2 * (b x).2)
  (h_x : ∀ x, 0 < x ∧ x < Real.pi)
  (α : ℝ)
  (h_α : f α = 2 * Real.sqrt 2 / 3) :
  (∃ x, a.1 * (b x).1 + a.2 * (b x).2 = 0 → x = 2 * Real.pi / 3) ∧ 
  Real.sin (2 * α + Real.pi / 6) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_and_trigonometry_l2571_257100


namespace NUMINAMATH_CALUDE_regions_count_l2571_257127

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (-2, -2)
def D : ℝ × ℝ := (2, -2)
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (0, 1)
def G : ℝ × ℝ := (-1, 0)
def H : ℝ × ℝ := (0, -1)

-- Define the set of all points
def points : Set (ℝ × ℝ) := {A, B, C, D, E, F, G, H}

-- Define the square ABCD
def squareABCD : Set (ℝ × ℝ) := {(x, y) | -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2}

-- Define a function to count regions formed by line segments
def countRegions (pts : Set (ℝ × ℝ)) (square : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem regions_count : countRegions points squareABCD = 60 := by sorry

end NUMINAMATH_CALUDE_regions_count_l2571_257127


namespace NUMINAMATH_CALUDE_exam_score_proof_l2571_257197

theorem exam_score_proof (total_questions : ℕ) 
                         (correct_score wrong_score : ℤ) 
                         (total_score : ℤ) 
                         (h1 : total_questions = 100)
                         (h2 : correct_score = 5)
                         (h3 : wrong_score = -2)
                         (h4 : total_score = 150) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + 
    wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l2571_257197


namespace NUMINAMATH_CALUDE_stating_retirement_benefit_formula_l2571_257175

/-- Represents the retirement benefit calculation for a teacher. -/
structure TeacherBenefit where
  /-- The number of years the teacher has taught. -/
  y : ℝ
  /-- The proportionality constant for the benefit calculation. -/
  k : ℝ
  /-- The additional years in the first scenario. -/
  c : ℝ
  /-- The additional years in the second scenario. -/
  d : ℝ
  /-- The benefit increase in the first scenario. -/
  r : ℝ
  /-- The benefit increase in the second scenario. -/
  s : ℝ
  /-- Ensures that c and d are different. -/
  h_c_neq_d : c ≠ d
  /-- The benefit is proportional to the square root of years taught. -/
  h_benefit : k * Real.sqrt y > 0
  /-- The equation for the first scenario. -/
  h_eq1 : k * Real.sqrt (y + c) = k * Real.sqrt y + r
  /-- The equation for the second scenario. -/
  h_eq2 : k * Real.sqrt (y + d) = k * Real.sqrt y + s

/-- 
Theorem stating that the original annual retirement benefit 
is equal to (s² - r²) / (2(s - r)) given the conditions.
-/
theorem retirement_benefit_formula (tb : TeacherBenefit) : 
  tb.k * Real.sqrt tb.y = (tb.s^2 - tb.r^2) / (2 * (tb.s - tb.r)) := by
  sorry


end NUMINAMATH_CALUDE_stating_retirement_benefit_formula_l2571_257175


namespace NUMINAMATH_CALUDE_circular_track_length_l2571_257125

-- Define the track length
def track_length : ℝ := 350

-- Define the constants given in the problem
def first_meeting_distance : ℝ := 80
def second_meeting_distance : ℝ := 140

-- Theorem statement
theorem circular_track_length :
  ∀ (brenda_speed sally_speed : ℝ),
  brenda_speed > 0 ∧ sally_speed > 0 →
  ∃ (t₁ t₂ : ℝ),
  t₁ > 0 ∧ t₂ > 0 ∧
  brenda_speed * t₁ = first_meeting_distance ∧
  sally_speed * t₁ = track_length / 2 - first_meeting_distance ∧
  brenda_speed * (t₁ + t₂) = track_length / 2 + first_meeting_distance ∧
  sally_speed * (t₁ + t₂) = track_length / 2 + second_meeting_distance →
  track_length = 350 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_circular_track_length_l2571_257125


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2571_257142

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the line L
def line_L (k x y : ℝ) : Prop := y = k*x - 3*k + 1

-- Theorem statement
theorem line_intersects_circle :
  ∀ (k : ℝ), ∃ (x y : ℝ), circle_C x y ∧ line_L k x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2571_257142


namespace NUMINAMATH_CALUDE_jake_first_test_score_l2571_257163

/-- Represents the marks Jake scored in his tests -/
structure JakeScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that Jake's first test score was 80 given the conditions -/
theorem jake_first_test_score (scores : JakeScores) :
  (scores.first + scores.second + scores.third + scores.fourth) / 4 = 75 →
  scores.second = scores.first + 10 →
  scores.third = scores.fourth →
  scores.third = 65 →
  scores.first = 80 := by
  sorry

#check jake_first_test_score

end NUMINAMATH_CALUDE_jake_first_test_score_l2571_257163


namespace NUMINAMATH_CALUDE_abc_fraction_simplification_l2571_257186

theorem abc_fraction_simplification 
  (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_condition : a + b + c = 1) :
  let s := a * b + b * c + c * a
  (a^2 + b^2 + c^2) ≠ 0 ∧ 
  (ab+bc+ca) / (a^2+b^2+c^2) = s / (1 - 2*s) := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_simplification_l2571_257186


namespace NUMINAMATH_CALUDE_rocket_heights_sum_l2571_257120

/-- The height of the first rocket in feet -/
def first_rocket_height : ℝ := 500

/-- The height of the second rocket in feet -/
def second_rocket_height : ℝ := 2 * first_rocket_height

/-- The combined height of both rockets in feet -/
def combined_height : ℝ := first_rocket_height + second_rocket_height

theorem rocket_heights_sum :
  combined_height = 1500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_heights_sum_l2571_257120


namespace NUMINAMATH_CALUDE_second_place_score_l2571_257104

/-- Represents a player in the chess tournament -/
structure Player where
  score : ℕ

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Finset Player
  secondPlace : Player
  lastFour : Finset Player

/-- The rules and conditions of the tournament -/
def TournamentRules (t : ChessTournament) : Prop :=
  -- 8 players in total
  t.players.card = 8 ∧
  -- Second place player is in the set of all players
  t.secondPlace ∈ t.players ∧
  -- Last four players are in the set of all players
  t.lastFour ⊆ t.players ∧
  -- Last four players are distinct and have 4 members
  t.lastFour.card = 4 ∧
  -- All scores are different
  ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → p1 ≠ p2 → p1.score ≠ p2.score ∧
  -- Second place score equals sum of last four scores
  t.secondPlace.score = (t.lastFour.toList.map Player.score).sum ∧
  -- Maximum possible score is 14
  ∀ p : Player, p ∈ t.players → p.score ≤ 14

/-- The main theorem -/
theorem second_place_score (t : ChessTournament) :
  TournamentRules t → t.secondPlace.score = 12 := by sorry

end NUMINAMATH_CALUDE_second_place_score_l2571_257104


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2571_257105

/-- The coefficient of x^2 in the expansion of (1+x+x^2)^6 -/
def a₂ : ℕ := (6 * (6 + 1)) / 2

/-- The expansion of (1+x+x^2)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x + x^2)^6

theorem coefficient_of_x_squared :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    expansion = λ x => a₂ * x^2 + f x * x^3 + g x := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2571_257105


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l2571_257155

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l2571_257155


namespace NUMINAMATH_CALUDE_equation_solutions_l2571_257171

theorem equation_solutions :
  (∃ x : ℝ, x = 1/3 ∧ 3/(1-6*x) = 2/(6*x+1) - (8+9*x)/(36*x^2-1)) ∧
  (∃ z : ℝ, z = -3/7 ∧ 3/(1-z^2) = 2/((1+z)^2) - 5/((1-z)^2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2571_257171


namespace NUMINAMATH_CALUDE_adam_tattoo_count_l2571_257195

/-- The number of tattoos on each of Jason's arms -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos on each of Jason's legs -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- Adam has three more than twice as many tattoos as Jason -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by sorry

end NUMINAMATH_CALUDE_adam_tattoo_count_l2571_257195


namespace NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l2571_257146

/-- An animal is a connected figure consisting of equal-sized square cells. -/
structure Animal where
  cells : ℕ
  is_connected : Bool

/-- A dinosaur is an animal with at least 2007 cells. -/
def Dinosaur (a : Animal) : Prop :=
  a.cells ≥ 2007

/-- A primitive dinosaur cannot be partitioned into two or more dinosaurs. -/
def PrimitiveDinosaur (a : Animal) : Prop :=
  Dinosaur a ∧ ¬∃ (b c : Animal), Dinosaur b ∧ Dinosaur c ∧ b.cells + c.cells ≤ a.cells

/-- The maximum number of cells in a primitive dinosaur is 8025. -/
theorem max_primitive_dinosaur_cells :
  ∃ (a : Animal), PrimitiveDinosaur a ∧ a.cells = 8025 ∧
  ∀ (b : Animal), PrimitiveDinosaur b → b.cells ≤ 8025 := by
  sorry


end NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l2571_257146


namespace NUMINAMATH_CALUDE_income_left_percentage_l2571_257156

/-- Given a man's spending habits, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ)
  (h1 : food_percent = 50)
  (h2 : education_percent = 15)
  (h3 : rent_percent = 50)
  (h4 : total_income > 0) :
  let remaining_after_food := total_income * (1 - food_percent / 100)
  let remaining_after_education := remaining_after_food - (total_income * education_percent / 100)
  let remaining_after_rent := remaining_after_education * (1 - rent_percent / 100)
  remaining_after_rent / total_income * 100 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_income_left_percentage_l2571_257156


namespace NUMINAMATH_CALUDE_prince_gvidon_descendants_l2571_257110

/-- The total number of descendants of Prince Gvidon -/
def total_descendants : ℕ := 189

/-- The number of sons Prince Gvidon had -/
def initial_sons : ℕ := 3

/-- The number of descendants who had two sons each -/
def descendants_with_sons : ℕ := 93

/-- The number of sons each descendant with sons had -/
def sons_per_descendant : ℕ := 2

theorem prince_gvidon_descendants :
  total_descendants = initial_sons + descendants_with_sons * sons_per_descendant :=
by sorry

end NUMINAMATH_CALUDE_prince_gvidon_descendants_l2571_257110


namespace NUMINAMATH_CALUDE_cheryl_strawberries_l2571_257183

theorem cheryl_strawberries (num_buckets : ℕ) (removed_per_bucket : ℕ) (remaining_per_bucket : ℕ) : 
  num_buckets = 5 →
  removed_per_bucket = 20 →
  remaining_per_bucket = 40 →
  num_buckets * (removed_per_bucket + remaining_per_bucket) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_strawberries_l2571_257183


namespace NUMINAMATH_CALUDE_time_conversion_l2571_257150

-- Define the conversion rates
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the given time
def hours : ℕ := 3
def minutes : ℕ := 25

-- Theorem to prove
theorem time_conversion :
  (hours * minutes_per_hour + minutes) * seconds_per_minute = 12300 := by
  sorry

end NUMINAMATH_CALUDE_time_conversion_l2571_257150


namespace NUMINAMATH_CALUDE_weeks_to_save_dress_l2571_257190

def original_price : ℚ := 150
def discount_rate : ℚ := 15 / 100
def initial_savings : ℚ := 35
def odd_week_allowance : ℚ := 30
def even_week_allowance : ℚ := 35
def weekly_arcade_expense : ℚ := 20
def weekly_snack_expense : ℚ := 10

def discounted_price : ℚ := original_price * (1 - discount_rate)
def amount_to_save : ℚ := discounted_price - initial_savings
def biweekly_allowance : ℚ := odd_week_allowance + even_week_allowance
def weekly_expenses : ℚ := weekly_arcade_expense + weekly_snack_expense
def biweekly_savings : ℚ := biweekly_allowance - 2 * weekly_expenses
def average_weekly_savings : ℚ := biweekly_savings / 2

theorem weeks_to_save_dress : 
  ⌈amount_to_save / average_weekly_savings⌉ = 37 := by sorry

end NUMINAMATH_CALUDE_weeks_to_save_dress_l2571_257190


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l2571_257173

theorem prime_sum_of_squares (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ (x y m : ℕ), x^2 + y^2 = m * p) ∧
  (∀ (x y m : ℕ), x^2 + y^2 = m * p → m > 1 → 
    ∃ (X Y m' : ℕ), X^2 + Y^2 = m' * p ∧ 0 < m' ∧ m' < m) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l2571_257173


namespace NUMINAMATH_CALUDE_cube_root_two_solves_equation_l2571_257159

theorem cube_root_two_solves_equation :
  let x : ℝ := Real.rpow 2 (1/3)
  (x + 1)^3 = 1 / (x - 1) ∧ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_two_solves_equation_l2571_257159


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l2571_257185

open Real

/-- The minimum distance between a point on the line y = (12/5)x - 3 and a point on the parabola y = x^2 is 3/5. -/
theorem min_distance_line_parabola :
  let line := fun x => (12/5) * x - 3
  let parabola := fun x => x^2
  ∃ (a b : ℝ),
    (∀ x y : ℝ, 
      (y = line x ∨ y = parabola x) → 
      (a - x)^2 + (line a - y)^2 ≥ (3/5)^2) ∧
    line a = parabola b ∧
    (a - b)^2 + (line a - parabola b)^2 = (3/5)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l2571_257185


namespace NUMINAMATH_CALUDE_same_heads_probability_l2571_257191

/-- The number of coins Keiko tosses -/
def keiko_coins : ℕ := 2

/-- The number of coins Ephraim tosses -/
def ephraim_coins : ℕ := 3

/-- The probability of getting the same number of heads -/
def same_heads_prob : ℚ := 3/16

/-- 
Theorem: Given that Keiko tosses 2 coins and Ephraim tosses 3 coins, 
the probability that Ephraim gets the same number of heads as Keiko is 3/16.
-/
theorem same_heads_probability : 
  let outcomes := 2^(keiko_coins + ephraim_coins)
  let favorable_outcomes := (keiko_coins + 1) * (ephraim_coins + 1) / 2
  (favorable_outcomes : ℚ) / outcomes = same_heads_prob := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l2571_257191


namespace NUMINAMATH_CALUDE_line_intercept_l2571_257141

/-- Given a line y = ax + b passing through the points (3, -2) and (7, 14), prove that b = -14 -/
theorem line_intercept (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) →   -- Definition of the line
  (-2 : ℝ) = a * 3 + b →         -- Line passes through (3, -2)
  (14 : ℝ) = a * 7 + b →         -- Line passes through (7, 14)
  b = -14 := by sorry

end NUMINAMATH_CALUDE_line_intercept_l2571_257141


namespace NUMINAMATH_CALUDE_weed_difference_l2571_257176

/-- The number of weeds Sarah pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Sarah's weed-pulling over four days --/
def SarahsWeedPulling (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday < w.thursday ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- Theorem stating the difference in weeds pulled between Thursday and Friday --/
theorem weed_difference (w : WeedCount) (h : SarahsWeedPulling w) :
  w.thursday - w.friday = 10 := by
  sorry

end NUMINAMATH_CALUDE_weed_difference_l2571_257176


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2571_257193

theorem absolute_value_inequality (x : ℝ) : 
  |x - x^2 - 2| > x^2 - 3*x - 4 ↔ x > -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2571_257193


namespace NUMINAMATH_CALUDE_correct_operation_l2571_257134

theorem correct_operation (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2571_257134


namespace NUMINAMATH_CALUDE_dice_sum_symmetry_l2571_257102

/-- The number of dice being rolled -/
def num_dice : ℕ := 9

/-- The minimum value on each die -/
def min_value : ℕ := 1

/-- The maximum value on each die -/
def max_value : ℕ := 6

/-- The sum we're comparing to -/
def comparison_sum : ℕ := 15

/-- The function to calculate the symmetric sum -/
def symmetric_sum (s : ℕ) : ℕ :=
  2 * ((num_dice * min_value + num_dice * max_value) / 2) - s

theorem dice_sum_symmetry :
  symmetric_sum comparison_sum = 48 :=
sorry

end NUMINAMATH_CALUDE_dice_sum_symmetry_l2571_257102


namespace NUMINAMATH_CALUDE_sum_of_digits_is_400_l2571_257151

/-- A number system with base r -/
structure BaseR where
  r : ℕ
  h_r : r ≤ 400

/-- A number x in base r of the form ppqq -/
structure NumberX (b : BaseR) where
  p : ℕ
  q : ℕ
  h_pq : 7 * q = 17 * p
  x : ℕ
  h_x : x = p * b.r^3 + p * b.r^2 + q * b.r + q

/-- The square of x is a seven-digit palindrome with middle digit zero -/
def is_palindrome_square (b : BaseR) (x : NumberX b) : Prop :=
  ∃ (a c : ℕ),
    x.x^2 = a * b.r^6 + c * b.r^5 + c * b.r^4 + 0 * b.r^3 + c * b.r^2 + c * b.r + a

/-- The sum of digits of x^2 in base r -/
def sum_of_digits (b : BaseR) (x : NumberX b) : ℕ :=
  sorry  -- Definition of sum of digits

/-- Main theorem -/
theorem sum_of_digits_is_400 (b : BaseR) (x : NumberX b) 
    (h_palindrome : is_palindrome_square b x) : 
    sum_of_digits b x = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_400_l2571_257151


namespace NUMINAMATH_CALUDE_floor_sqrt_225_l2571_257154

theorem floor_sqrt_225 : ⌊Real.sqrt 225⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_225_l2571_257154


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2571_257166

/-- Definition of the sequence a_n -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- Main theorem: 2^k divides a_n if and only if 2^k divides n -/
theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ a n ↔ 2^k ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2571_257166


namespace NUMINAMATH_CALUDE_sum_and_round_to_nearest_ten_l2571_257180

-- Define a function to round to the nearest ten
def roundToNearestTen (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

-- Theorem statement
theorem sum_and_round_to_nearest_ten :
  roundToNearestTen (54 + 29) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_round_to_nearest_ten_l2571_257180


namespace NUMINAMATH_CALUDE_polynomial_expansion_alternating_sum_l2571_257178

theorem polynomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_alternating_sum_l2571_257178


namespace NUMINAMATH_CALUDE_distinct_remainders_of_sums_l2571_257179

theorem distinct_remainders_of_sums (n : ℕ) (h : n > 1) :
  let S := Finset.range n
  ∀ (i j k l : ℕ) (hi : i ∈ S) (hj : j ∈ S) (hk : k ∈ S) (hl : l ∈ S)
    (hij : i ≤ j) (hkl : k ≤ l),
  (i + j) % (n * (n + 1) / 2) = (k + l) % (n * (n + 1) / 2) →
  i = k ∧ j = l :=
by sorry

end NUMINAMATH_CALUDE_distinct_remainders_of_sums_l2571_257179


namespace NUMINAMATH_CALUDE_carla_water_calculation_l2571_257122

/-- The amount of water Carla needs to bring for her animals -/
def water_needed (pig_count : ℕ) (horse_count : ℕ) (pig_water : ℕ) (chicken_tank : ℕ) : ℕ :=
  let pig_total := pig_count * pig_water
  let horse_total := horse_count * (2 * pig_water)
  pig_total + horse_total + chicken_tank

/-- Theorem stating the total amount of water Carla needs -/
theorem carla_water_calculation :
  water_needed 8 10 3 30 = 114 := by
  sorry

end NUMINAMATH_CALUDE_carla_water_calculation_l2571_257122


namespace NUMINAMATH_CALUDE_initial_fraction_is_half_l2571_257196

/-- Represents a journey with two parts at different speeds -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  remainingSpeed : ℝ
  initialFraction : ℝ

/-- The conditions of the journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed = 40 ∧
  j.remainingSpeed = 20 ∧
  j.initialFraction * j.totalDistance = j.initialSpeed * (j.totalTime / 3) ∧
  (1 - j.initialFraction) * j.totalDistance = j.remainingSpeed * (2 * j.totalTime / 3) ∧
  j.totalDistance > 0 ∧
  j.totalTime > 0

/-- The theorem stating that under the given conditions, the initial fraction is 1/2 -/
theorem initial_fraction_is_half (j : Journey) :
  journeyConditions j → j.initialFraction = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_initial_fraction_is_half_l2571_257196


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2571_257103

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Theorem stating that the total selling price of 85 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 100 per meter is Rs. 8925. -/
theorem cloth_selling_price :
  total_selling_price 85 5 100 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l2571_257103


namespace NUMINAMATH_CALUDE_integral_absolute_value_l2571_257131

theorem integral_absolute_value : 
  ∫ x in (0 : ℝ)..2, (2 - |1 - x|) = 3 := by sorry

end NUMINAMATH_CALUDE_integral_absolute_value_l2571_257131


namespace NUMINAMATH_CALUDE_inverse_f_f_condition_l2571_257139

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem inverse_f (x : ℝ) (h : x ≥ -1) : 
  f⁻¹ (x + 1) = 2 - Real.sqrt (x + 1) := by sorry

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x ≤ 2}

-- State the condition given in the problem
theorem f_condition (x : ℝ) (h : x ≤ 1) : 
  f (x + 1) = (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_f_condition_l2571_257139


namespace NUMINAMATH_CALUDE_system_solution_l2571_257121

theorem system_solution (x y k : ℝ) : 
  x + 3*y = 2*k + 1 → 
  x - y = 1 → 
  x = -y → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2571_257121


namespace NUMINAMATH_CALUDE_roof_collapse_leaves_l2571_257135

theorem roof_collapse_leaves (roof_capacity : ℕ) (leaves_per_pound : ℕ) (days_to_collapse : ℕ) :
  roof_capacity = 500 →
  leaves_per_pound = 1000 →
  days_to_collapse = 5000 →
  (roof_capacity * leaves_per_pound) / days_to_collapse = 100 :=
by sorry

end NUMINAMATH_CALUDE_roof_collapse_leaves_l2571_257135


namespace NUMINAMATH_CALUDE_sample_is_weights_l2571_257149

/-- Represents a student in the survey -/
structure Student where
  weight : ℝ

/-- Represents the survey conducted by the city -/
structure Survey where
  students : Finset Student
  grade : Nat

/-- Definition of a sample in this context -/
def Sample (survey : Survey) : Set ℝ :=
  {w | ∃ s ∈ survey.students, w = s.weight}

/-- The theorem stating that the sample is the weight of 100 students -/
theorem sample_is_weights (survey : Survey) 
    (h1 : survey.grade = 9) 
    (h2 : survey.students.card = 100) : 
  Sample survey = {w | ∃ s ∈ survey.students, w = s.weight} := by
  sorry

end NUMINAMATH_CALUDE_sample_is_weights_l2571_257149


namespace NUMINAMATH_CALUDE_quadratic_equation_with_ratio_roots_l2571_257198

theorem quadratic_equation_with_ratio_roots (k : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 8*x + k = 0 ↔ (x = 3*r ∨ x = r)) ∧
    3*r ≠ r) → 
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_ratio_roots_l2571_257198


namespace NUMINAMATH_CALUDE_monica_reading_plan_l2571_257136

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 25

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 3 * books_this_year + 7

theorem monica_reading_plan : books_next_year = 232 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l2571_257136


namespace NUMINAMATH_CALUDE_clarinet_fraction_in_band_l2571_257189

theorem clarinet_fraction_in_band (total_band : ℕ) (flutes_in : ℕ) (trumpets_in : ℕ) (pianists_in : ℕ) (clarinets_total : ℕ) :
  total_band = 53 →
  flutes_in = 16 →
  trumpets_in = 20 →
  pianists_in = 2 →
  clarinets_total = 30 →
  (total_band - (flutes_in + trumpets_in + pianists_in)) / clarinets_total = 1 / 2 :=
by
  sorry

#check clarinet_fraction_in_band

end NUMINAMATH_CALUDE_clarinet_fraction_in_band_l2571_257189


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2571_257192

/-- Given a cube with surface area 864 square units, its volume is 1728 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  (6 * s^2 = 864) →
  s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2571_257192


namespace NUMINAMATH_CALUDE_total_rope_length_l2571_257111

-- Define the lengths of rope used for each post
def post1_length : ℕ := 24
def post2_length : ℕ := 20
def post3_length : ℕ := 14
def post4_length : ℕ := 12

-- Theorem stating that the total rope length is 70 inches
theorem total_rope_length :
  post1_length + post2_length + post3_length + post4_length = 70 :=
by sorry

end NUMINAMATH_CALUDE_total_rope_length_l2571_257111


namespace NUMINAMATH_CALUDE_brookes_added_balloons_l2571_257147

/-- Prove that Brooke added 8 balloons to his collection -/
theorem brookes_added_balloons :
  ∀ (brooke_initial tracy_initial tracy_added total_after : ℕ) 
    (brooke_added : ℕ),
  brooke_initial = 12 →
  tracy_initial = 6 →
  tracy_added = 24 →
  total_after = 35 →
  brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after →
  brooke_added = 8 := by
sorry

end NUMINAMATH_CALUDE_brookes_added_balloons_l2571_257147


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2571_257119

/-- The x-coordinate of a point on a parabola with a given distance from its focus -/
theorem parabola_point_x_coordinate (x y : ℝ) :
  y^2 = 4*x →  -- Point (x, y) is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 4^2 →  -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2571_257119


namespace NUMINAMATH_CALUDE_equal_interest_l2571_257114

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: Rs 600 at 10% for 4 years produces the same interest as Rs 100 at 5% for 48 years -/
theorem equal_interest : simple_interest 100 0.05 48 = simple_interest 600 0.10 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_interest_l2571_257114


namespace NUMINAMATH_CALUDE_five_eighteenths_decimal_l2571_257177

theorem five_eighteenths_decimal : 
  (5 : ℚ) / 18 = 0.2777777777777777 :=
by sorry

end NUMINAMATH_CALUDE_five_eighteenths_decimal_l2571_257177


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l2571_257106

theorem quadratic_equation_proof (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has roots x₁ and x₂
  x₁ ≠ x₂ →  -- roots are distinct
  m > -1/3 →  -- condition from part 1
  m ≠ 0 →  -- condition from part 1
  x₁^2 + x₂^2 = 8 →  -- given condition
  m = 2 :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l2571_257106


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2571_257170

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x + 2 = 0 ∧ k * y^2 - 2 * y + 2 = 0) ↔ 
  (k < 1/2 ∧ k ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2571_257170


namespace NUMINAMATH_CALUDE_sin_theta_equals_sqrt2_over_2_l2571_257169

theorem sin_theta_equals_sqrt2_over_2 (θ : Real) (a : Real) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : Real), x = a ∧ y = a ∧ Real.cos θ * Real.cos θ + Real.sin θ * Real.sin θ = 1 ∧ 
    Real.cos θ * x - Real.sin θ * y = 0 ∧ Real.sin θ * x + Real.cos θ * y = 0) : 
  |Real.sin θ| = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_equals_sqrt2_over_2_l2571_257169


namespace NUMINAMATH_CALUDE_find_a_l2571_257168

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧
  (∀ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_l2571_257168


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2571_257138

/-- A regular polygon with interior angles measuring 150° has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) →
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2571_257138


namespace NUMINAMATH_CALUDE_lawrence_county_kids_l2571_257130

/-- The number of kids from Lawrence county going to camp -/
def kids_camp : ℕ := 610769

/-- The number of kids from Lawrence county staying home -/
def kids_home : ℕ := 590796

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_camp + kids_home

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids going to camp and kids staying home -/
theorem lawrence_county_kids : total_kids = 1201565 := by sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_l2571_257130


namespace NUMINAMATH_CALUDE_quadrilateral_vector_relation_l2571_257152

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define vectors a, b, c
variable (a b c : ℝ × ℝ)

-- State the theorem
theorem quadrilateral_vector_relation 
  (h1 : B - A = a) 
  (h2 : D - A = b) 
  (h3 : C - B = c) : 
  D - C = b - a - c := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_relation_l2571_257152


namespace NUMINAMATH_CALUDE_p_plus_q_equals_42_l2571_257113

theorem p_plus_q_equals_42 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 4 → P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4)) →
  P + Q = 42 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_42_l2571_257113


namespace NUMINAMATH_CALUDE_calculate_brokerage_percentage_l2571_257143

/-- Calculate the brokerage percentage for a stock investment --/
theorem calculate_brokerage_percentage
  (stock_rate : ℝ)
  (income : ℝ)
  (investment : ℝ)
  (market_value : ℝ)
  (h1 : stock_rate = 10.5)
  (h2 : income = 756)
  (h3 : investment = 8000)
  (h4 : market_value = 110.86111111111111)
  : ∃ (brokerage_percentage : ℝ),
    brokerage_percentage = 0.225 ∧
    brokerage_percentage = (investment - (income * 100 / stock_rate) * market_value / 100) / investment * 100 :=
by sorry

end NUMINAMATH_CALUDE_calculate_brokerage_percentage_l2571_257143


namespace NUMINAMATH_CALUDE_investment_interest_proof_l2571_257123

/-- Calculates the total interest earned on an investment with compound interest. -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on a $2000 investment at 8% annual interest
    compounded annually for 5 years is approximately $938.656. -/
theorem investment_interest_proof :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.656) < 0.001 := by
  sorry

#eval totalInterestEarned 2000 0.08 5

end NUMINAMATH_CALUDE_investment_interest_proof_l2571_257123


namespace NUMINAMATH_CALUDE_optimal_game_outcome_l2571_257112

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a strategy for a player -/
def Strategy := List ℤ → ℤ

/-- The game state, including the current sum and remaining numbers -/
structure GameState :=
  (sum : ℤ)
  (remaining : List ℤ)

/-- The result of playing the game with given strategies -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : ℤ :=
  sorry

/-- An optimal strategy for the first player -/
def optimalFirstStrategy : Strategy :=
  sorry

/-- An optimal strategy for the second player -/
def optimalSecondStrategy : Strategy :=
  sorry

/-- The theorem stating the optimal outcome of the game -/
theorem optimal_game_outcome :
  playGame optimalFirstStrategy optimalSecondStrategy = 30 :=
sorry

end NUMINAMATH_CALUDE_optimal_game_outcome_l2571_257112


namespace NUMINAMATH_CALUDE_order_of_abc_l2571_257160

theorem order_of_abc (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1)
  (hb : b = 1/9)
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_order_of_abc_l2571_257160


namespace NUMINAMATH_CALUDE_element_in_complement_l2571_257174

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,5}
def P : Set Nat := {2,4}

theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by
  sorry

end NUMINAMATH_CALUDE_element_in_complement_l2571_257174


namespace NUMINAMATH_CALUDE_max_area_at_150_l2571_257109

/-- Represents a rectangular pasture with a fence on three sides and a barn on the fourth side. -/
structure Pasture where
  fenceLength : ℝ  -- Total length of fence available
  barnLength : ℝ   -- Length of the barn side

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn. -/
def Pasture.area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fenceLength - 2 * x)

/-- Theorem stating that the maximum area of the pasture occurs when the side parallel to the barn is 150 feet. -/
theorem max_area_at_150 (p : Pasture) (h1 : p.fenceLength = 300) (h2 : p.barnLength = 350) :
  ∃ (x : ℝ), x > 0 ∧ x < p.barnLength ∧
  (∀ (y : ℝ), y > 0 → y < p.barnLength → p.area x ≥ p.area y) ∧
  p.fenceLength - 2 * x = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_area_at_150_l2571_257109


namespace NUMINAMATH_CALUDE_monday_calls_l2571_257165

/-- Represents the number of calls answered on each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day -/
def averageCalls : ℕ := 40

/-- The number of working days in a week -/
def workDays : ℕ := 5

/-- Jean's call data for the week -/
def jeanCalls : WeekCalls := {
  monday := 0,  -- We don't know this value yet
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31
}

theorem monday_calls : jeanCalls.monday = 35 := by sorry

end NUMINAMATH_CALUDE_monday_calls_l2571_257165


namespace NUMINAMATH_CALUDE_P_symmetric_l2571_257132

variable (x y z : ℝ)

noncomputable def P : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, _, _, _ => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem P_symmetric (m : ℕ) : 
  P m x y z = P m y x z ∧ 
  P m x y z = P m x z y ∧ 
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_P_symmetric_l2571_257132


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2571_257158

theorem six_digit_divisibility (a b c d e f : ℕ) 
  (h_six_digit : 100000 ≤ a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧ 
                 a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f < 1000000)
  (h_sum_equal : a + d = b + e ∧ b + e = c + f) : 
  ∃ k : ℕ, a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f = 37 * k :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2571_257158


namespace NUMINAMATH_CALUDE_function_value_at_specific_point_l2571_257117

/-- The base-3 logarithm -/
noncomputable def log3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := (Real.log x) / (Real.log 10)

/-- The given function f -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin x - b * log3 (Real.sqrt (x^2 + 1) - x) + 1

theorem function_value_at_specific_point
  (a b : ℝ) (h : f a b (lg (log3 10)) = 5) :
  f a b (lg (lg 3)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_specific_point_l2571_257117


namespace NUMINAMATH_CALUDE_playground_width_l2571_257182

/-- The number of playgrounds -/
def num_playgrounds : ℕ := 8

/-- The length of each playground in meters -/
def playground_length : ℝ := 300

/-- The total area of all playgrounds in square kilometers -/
def total_area_km2 : ℝ := 0.6

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

theorem playground_width :
  ∀ (width : ℝ),
  (width * playground_length * num_playgrounds = total_area_km2 * km2_to_m2) →
  width = 250 := by
sorry

end NUMINAMATH_CALUDE_playground_width_l2571_257182


namespace NUMINAMATH_CALUDE_prob_ten_then_spade_value_l2571_257161

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a 10 as the first card -/
def first_card_is_ten (d : Deck) : Prop :=
  ∃ c ∈ d.cards, c < 4

/-- Represents the event of drawing a spade as the second card -/
def second_card_is_spade (d : Deck) : Prop :=
  ∃ c ∈ d.cards, 39 ≤ c ∧ c < 52

/-- The probability of drawing a 10 as the first card -/
def prob_first_ten (d : Deck) : ℚ :=
  4 / 52

/-- The probability of drawing a spade as the second card -/
def prob_second_spade (d : Deck) : ℚ :=
  13 / 51

/-- The probability of drawing a 10 as the first card and a spade as the second card -/
def prob_ten_then_spade (d : Deck) : ℚ :=
  prob_first_ten d * prob_second_spade d

theorem prob_ten_then_spade_value (d : Deck) :
  prob_ten_then_spade d = 12 / 663 :=
sorry

end NUMINAMATH_CALUDE_prob_ten_then_spade_value_l2571_257161


namespace NUMINAMATH_CALUDE_max_value_f_l2571_257162

/-- The function f defined on ℝ -/
def f (a b x : ℝ) : ℝ := 2 * a * x + b

/-- The theorem stating the conditions and the result -/
theorem max_value_f (a b : ℝ) : 
  b > 0 ∧ 
  (∀ x ∈ Set.Icc (-1/2) (1/2), |f a b x| ≤ 2) ∧
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x ∈ Set.Icc (-1/2) (1/2), |f a' b' x| ≤ 2) → a * b ≥ a' * b') →
  f a b 2017 = 4035 := by
sorry

end NUMINAMATH_CALUDE_max_value_f_l2571_257162


namespace NUMINAMATH_CALUDE_least_candies_eleven_candies_maria_candies_l2571_257126

theorem least_candies (c : ℕ) : c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 → c ≥ 11 :=
by sorry

theorem eleven_candies : 11 % 3 = 2 ∧ 11 % 4 = 3 ∧ 11 % 6 = 5 :=
by sorry

theorem maria_candies : ∃ (c : ℕ), c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 ∧ c = 11 :=
by sorry

end NUMINAMATH_CALUDE_least_candies_eleven_candies_maria_candies_l2571_257126


namespace NUMINAMATH_CALUDE_simplify_expression_l2571_257140

theorem simplify_expression (w : ℝ) : w - 2*w + 4*w - 5*w + 3 - 5 + 7 - 9 = -2*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2571_257140


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2571_257137

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_area_proof (A B C : ℝ) (hA : f A = 1) (ha : Real.sqrt 3 = A) (hbc : B + C = 3) :
  (1 / 2 : ℝ) * B * C * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2571_257137


namespace NUMINAMATH_CALUDE_total_sleep_l2571_257118

def sleep_pattern (first_night : ℕ) : Fin 4 → ℕ
| 0 => first_night
| 1 => 2 * first_night
| 2 => 2 * first_night - 3
| 3 => 3 * (2 * first_night - 3)

theorem total_sleep (first_night : ℕ) (h : first_night = 6) : 
  (Finset.sum Finset.univ (sleep_pattern first_night)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_l2571_257118


namespace NUMINAMATH_CALUDE_ellipse_sum_l2571_257128

/-- Theorem: For an ellipse with center (-3, 1), horizontal semi-major axis length 4,
    and vertical semi-minor axis length 2, the sum of h, k, a, and c is equal to 4. -/
theorem ellipse_sum (h k a c : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 4 ∧ c = 2 → h + k + a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2571_257128


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2571_257148

-- Define the solution set
def solution_set : Set ℝ := {x | x > 3 ∨ x < -1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x - 1| > 2} = solution_set := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2571_257148


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l2571_257199

/-- Given a rectangle with dimensions and a shaded area, calculate the perimeter of the non-shaded region --/
theorem non_shaded_perimeter (large_width large_height ext_width ext_height shaded_area : ℝ) :
  large_width = 12 →
  large_height = 8 →
  ext_width = 5 →
  ext_height = 2 →
  shaded_area = 104 →
  let total_area := large_width * large_height + ext_width * ext_height
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := ext_height
  let non_shaded_height := non_shaded_area / non_shaded_width
  2 * (non_shaded_width + non_shaded_height) = 6 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l2571_257199


namespace NUMINAMATH_CALUDE_cows_that_ran_away_l2571_257172

/-- Represents the problem of determining how many cows ran away --/
theorem cows_that_ran_away 
  (initial_cows : ℕ) 
  (feeding_period : ℕ) 
  (days_passed : ℕ) 
  (h1 : initial_cows = 1000)
  (h2 : feeding_period = 50)
  (h3 : days_passed = 10)
  : ∃ (cows_ran_away : ℕ),
    cows_ran_away = 200 ∧ 
    (initial_cows * feeding_period - initial_cows * days_passed) 
    = (initial_cows - cows_ran_away) * feeding_period :=
by sorry


end NUMINAMATH_CALUDE_cows_that_ran_away_l2571_257172


namespace NUMINAMATH_CALUDE_history_book_cost_l2571_257145

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books_bought : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : total_price = 390)
  (h4 : math_books_bought = 10) :
  (total_price - math_books_bought * math_book_cost) / (total_books - math_books_bought) = 5 := by
sorry

end NUMINAMATH_CALUDE_history_book_cost_l2571_257145


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2571_257194

/-- Given a quadratic function f(x) = 3x^2 - 2x + 8, when shifted 6 units to the left,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 141 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 8) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 141 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2571_257194


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2571_257124

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2571_257124


namespace NUMINAMATH_CALUDE_conditional_probability_l2571_257115

/-- The total number of balls in the box -/
def total_balls : ℕ := 12

/-- The number of yellow balls in the box -/
def yellow_balls : ℕ := 5

/-- The number of blue balls in the box -/
def blue_balls : ℕ := 4

/-- The number of green balls in the box -/
def green_balls : ℕ := 3

/-- Event A: The two balls drawn have different colors -/
def event_A : ℚ := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls.choose 2)

/-- Event B: One yellow ball and one blue ball are drawn -/
def event_B : ℚ := (yellow_balls * blue_balls) / (total_balls.choose 2)

/-- The conditional probability of B given A -/
theorem conditional_probability : event_B / event_A = 20 / 47 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_l2571_257115


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_l2571_257129

def is_interesting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

theorem max_consecutive_interesting :
  (∃ a : ℕ, ∀ k : ℕ, k < 3 → is_interesting (a + k)) ∧
  (∀ a : ℕ, ∃ k : ℕ, k < 4 → ¬is_interesting (a + k)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_interesting_l2571_257129
