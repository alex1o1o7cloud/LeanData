import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_loss_pair_l540_54017

/-- Represents the cost of a pair of books -/
structure BookPairCost where
  cost : ℝ

/-- Represents the total cost of four books -/
def total_cost : ℝ := 720

/-- Represents the loss percentage -/
def loss_percentage : ℝ := 0.12

/-- Represents the gain percentage -/
def gain_percentage : ℝ := 0.24

/-- Theorem stating the cost of the pair of books sold at a loss -/
theorem cost_of_loss_pair (pair_loss : BookPairCost) (pair_gain : BookPairCost) : 
  pair_loss.cost + pair_gain.cost = total_cost ∧ 
  (1 - loss_percentage) * pair_loss.cost = (1 + gain_percentage) * pair_gain.cost →
  abs (pair_loss.cost - 421.13) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_loss_pair_l540_54017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l540_54054

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of a side
noncomputable def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the angle between two sides
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  Real.arccos ((p1.1 - p2.1) * (p3.1 - p2.1) + (p1.2 - p2.2) * (p3.2 - p2.2)) /
    (sideLength p1 p2 * sideLength p2 p3)

-- Define the angle bisector
def angleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  angle t.A t.B D = angle t.A t.C D

-- Main theorem
theorem angle_bisector_length (t : Triangle) (D : ℝ × ℝ) :
  sideLength t.A t.B = 2 →
  angle t.B t.A t.C = π / 3 →
  sideLength t.B t.C = Real.sqrt 6 →
  angleBisector t D →
  sideLength t.A D = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l540_54054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_directrix_distance_l540_54047

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the left branch of a hyperbola -/
def isOnLeftBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x < 0 ∧ p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point on the left branch of the given hyperbola, 
    if its distance to the left focus is 16, 
    then its distance to the right directrix is 10 -/
theorem hyperbola_focus_directrix_distance 
  (h : Hyperbola) 
  (p : Point) 
  (leftFocus rightDirectrix : Point) 
  (h_eq : h.a = 5 ∧ h.b = 12) 
  (h_on_left : isOnLeftBranch h p) 
  (h_left_focus_dist : distance p leftFocus = 16) : 
  distance p rightDirectrix = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_directrix_distance_l540_54047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l540_54049

/-- Two non-intersecting circles with given properties -/
structure TwoCircles where
  ω₁ : Set (ℝ × ℝ)
  ω₂ : Set (ℝ × ℝ)
  center₁ : ℝ × ℝ
  center₂ : ℝ × ℝ
  radius₁ : ℝ
  radius₂ : ℝ
  non_intersecting : ω₁ ∩ ω₂ = ∅
  internal_tangent : (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 - (radius₁ + radius₂)^2 = 19^2
  external_tangent : (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 - (radius₁ - radius₂)^2 = 37^2
  expected_distance : (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 + radius₁^2 + radius₂^2 = 2023

/-- The distance between the centers of two circles with given properties is 38 -/
theorem distance_between_centers (c : TwoCircles) :
  Real.sqrt ((c.center₁.1 - c.center₂.1)^2 + (c.center₁.2 - c.center₂.2)^2) = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l540_54049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_l540_54050

/-- Calculates the face value of a bill given its true discount, time to maturity, and interest rate. -/
noncomputable def face_value (true_discount : ℝ) (time_to_maturity : ℝ) (interest_rate : ℝ) : ℝ :=
  (true_discount * (1 + interest_rate * time_to_maturity)) / (interest_rate * time_to_maturity)

/-- Theorem stating that under the given conditions, the face value of the bill is 1960. -/
theorem bill_face_value : 
  let true_discount : ℝ := 210
  let time_to_maturity : ℝ := 9 / 12  -- 9 months in years
  let interest_rate : ℝ := 16 / 100   -- 16% as a decimal
  face_value true_discount time_to_maturity interest_rate = 1960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_l540_54050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_possible_l540_54058

theorem triangle_construction_possible (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (A B C : EuclideanSpace ℝ (Fin 2)), 
    norm (A - B) = a ∧
    norm (B - C) = b ∧
    norm (C - A) = c) ↔
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_possible_l540_54058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l540_54078

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.tan (ω * x + φ)

-- State the theorem
theorem function_values (ω φ : ℝ) : 
  ω > 0 ∧ 
  |φ| < Real.pi / 2 ∧ 
  (∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) ∧ 
  (∀ y, y > 0 → y < Real.pi / (2 * ω) → ∃ x, f ω φ (x + y) ≠ f ω φ x) ∧ 
  f ω φ (Real.pi / 2) = -2 → 
  ω = 2 ∧ φ = -Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l540_54078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l540_54000

-- Define constants and variables
variable (C : ℝ) -- Coefficient of aerodynamic force
variable (S : ℝ) -- Area of the sail
variable (ρ : ℝ) -- Air density
variable (v₀ : ℝ) -- Wind speed
variable (v : ℝ) -- Speed of the sailboat

-- Define the force equation
noncomputable def F (C S ρ v₀ v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

-- Define the power equation
noncomputable def N (C S ρ v₀ v : ℝ) : ℝ := F C S ρ v₀ v * v

-- Theorem statement
theorem sailboat_max_power_speed (C S ρ v₀ : ℝ) (h1 : v₀ > 0) (h2 : C > 0) (h3 : S > 0) (h4 : ρ > 0) :
  ∃ v : ℝ, v > 0 ∧ v = v₀ / 3 ∧ ∀ u : ℝ, u ≠ v → N C S ρ v₀ v ≥ N C S ρ v₀ u := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l540_54000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_best_consistency_measure_l540_54012

/-- A type representing a group of students -/
structure StudentGroup where
  results : List Float
  size : Nat
  size_eq : results.length = size

/-- Definition of variance for a list of numbers -/
def variance (l : List Float) : Float :=
  let mean := l.sum / l.length.toFloat
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length.toFloat

/-- Consistency measure for a group of students -/
def consistencyMeasure (g : StudentGroup) : Float :=
  variance g.results

/-- Predicate to compare consistency between two lists of results -/
def moreConsistent (l1 l2 : List Float) : Prop :=
  variance l1 < variance l2

/-- Theorem stating that variance is the most appropriate measure for consistency -/
theorem variance_best_consistency_measure 
  (group1 group2 : StudentGroup) 
  (h1 : group1.size = 10) 
  (h2 : group2.size = 10) : 
  consistencyMeasure group1 < consistencyMeasure group2 → 
  moreConsistent group1.results group2.results :=
by
  intro h
  unfold moreConsistent
  unfold consistencyMeasure at h
  exact h

#check variance_best_consistency_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_best_consistency_measure_l540_54012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_rainy_days_approx_l540_54051

/-- The probability of at most 3 rainy days in July -/
noncomputable def prob_at_most_3_rainy_days : ℝ :=
  let n : ℕ := 31  -- number of days in July
  let p : ℝ := 3/20  -- probability of rain on any given day
  let q : ℝ := 1 - p
  (n.choose 0) * p^0 * q^31 +
  (n.choose 1) * p^1 * q^30 +
  (n.choose 2) * p^2 * q^29 +
  (n.choose 3) * p^3 * q^28

/-- The probability of at most 3 rainy days in July is approximately 0.625 -/
theorem prob_at_most_3_rainy_days_approx :
  abs (prob_at_most_3_rainy_days - 0.625) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_rainy_days_approx_l540_54051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_left_l540_54023

/-- Shifting the graph of y = sin(2x) by π/8 units to the left results in y = sin(2x + π/4) -/
theorem sin_shift_left (x : ℝ) : 
  Real.sin (2 * (x + Real.pi/8)) = Real.sin (2*x + Real.pi/4) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_left_l540_54023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l540_54086

/-- An ellipse with special properties -/
structure SpecialEllipse where
  /-- The ellipse has a circle passing through its two foci -/
  has_circle_through_foci : Bool
  /-- The circle's diameter endpoints are the foci of the ellipse -/
  circle_diameter_is_foci : Bool
  /-- The circle and ellipse intersect at four distinct points -/
  has_four_intersections : Bool
  /-- When these four points and the two foci are connected in sequence, they form a regular hexagon -/
  forms_regular_hexagon : Bool

/-- The eccentricity of an ellipse with the special properties -/
noncomputable def eccentricity (e : SpecialEllipse) : ℝ :=
  Real.sqrt 3 - 1

/-- Theorem stating the eccentricity of the special ellipse -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) 
  (h1 : e.has_circle_through_foci)
  (h2 : e.circle_diameter_is_foci)
  (h3 : e.has_four_intersections)
  (h4 : e.forms_regular_hexagon) :
  eccentricity e = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l540_54086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_specific_figure_l540_54087

/-- A geometric figure composed of six identical squares arranged in two columns of three squares each -/
structure GeometricFigure where
  area : ℝ
  num_squares : ℕ
  num_columns : ℕ
  num_rows : ℕ

/-- The perimeter of the geometric figure -/
noncomputable def perimeter (figure : GeometricFigure) : ℝ :=
  let square_side := Real.sqrt (figure.area / figure.num_squares)
  (2 * figure.num_columns + 2 * figure.num_rows) * square_side

/-- Theorem stating that the perimeter of the given geometric figure is approximately 61.23 cm -/
theorem perimeter_of_specific_figure :
  let figure : GeometricFigure := {
    area := 225,
    num_squares := 6,
    num_columns := 2,
    num_rows := 3
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |perimeter figure - 61.23| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_specific_figure_l540_54087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_twentieth_power_l540_54077

theorem last_two_digits_of_twentieth_power (x : ℤ) : 
  ∃ (y : ℕ), y ∈ ({0, 1, 25, 76} : Set ℕ) ∧ x^20 ≡ y [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_twentieth_power_l540_54077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l540_54048

noncomputable def h (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 2)

theorem h_range :
  Set.range h = Set.Icc ((1:ℝ)/2 - (3*Real.sqrt 2)/16) ((1:ℝ)/2 + (3*Real.sqrt 2)/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l540_54048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l540_54045

/-- Binary operation ◆ defined on nonzero real numbers -/
noncomputable def diamond : ℝ → ℝ → ℝ := sorry

/-- First property of ◆: a ◆ (b ◆ c) = (a ◆ b) * c -/
axiom diamond_assoc (a b c : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- Second property of ◆: a ◆ a = 1 for all nonzero real numbers a -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The solution to the equation 1024 ◆ (8 ◆ x) = 50 is x = 25/64 -/
theorem diamond_equation_solution : 
  ∃ x : ℝ, x ≠ 0 ∧ diamond 1024 (diamond 8 x) = 50 ∧ x = 25/64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l540_54045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l540_54002

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the time taken for two trains to cross each other -/
noncomputable def time_to_cross (train_a_length train_b_length train_a_speed train_b_speed : ℝ) : ℝ :=
  let relative_speed := kmph_to_mps train_a_speed + kmph_to_mps train_b_speed
  let total_length := train_a_length + train_b_length
  total_length / relative_speed

/-- Theorem stating that the time for the trains to cross is approximately 9 seconds -/
theorem trains_crossing_time :
  let train_a_length : ℝ := 300
  let train_b_length : ℝ := 200.04
  let train_a_speed : ℝ := 120
  let train_b_speed : ℝ := 80
  ∃ ε > 0, |time_to_cross train_a_length train_b_length train_a_speed train_b_speed - 9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l540_54002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l540_54097

theorem equidistant_point (A B C D P : ℝ × ℝ × ℝ) : 
  A = (10, 0, 0) →
  B = (0, -6, 0) →
  C = (0, 0, 8) →
  D = (0, 0, 0) →
  P = (5, -3, 4) →
  ‖A - P‖ = ‖B - P‖ ∧ ‖A - P‖ = ‖C - P‖ ∧ ‖A - P‖ = ‖D - P‖ := by
  sorry

#check equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l540_54097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_16_l540_54065

/-- Represents the price and tax information for an oil type -/
structure Oil where
  price : ℝ
  taxRate : ℝ

/-- Calculates the amount of oil that can be purchased for a given budget -/
noncomputable def amountPurchased (o : Oil) (budget : ℝ) : ℝ :=
  budget / (o.price * (1 + o.taxRate))

/-- Theorem: If a 10% price reduction allows 5 kg more to be purchased for Rs. 800,
    then the reduced price per kg including tax is Rs. 16 -/
theorem reduced_price_is_16 (o : Oil) :
  amountPurchased o 800 + 5 = amountPurchased ⟨o.price * 0.9, o.taxRate⟩ 800 →
  o.price * 0.9 * (1 + o.taxRate) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_16_l540_54065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recruitment_test_results_l540_54063

/-- Represents a candidate's test scores -/
structure CandidateScores where
  reading : ℚ
  thinking : ℚ
  expression : ℚ

/-- Calculates the average score of a candidate -/
def averageScore (scores : CandidateScores) : ℚ :=
  (scores.reading + scores.thinking + scores.expression) / 3

/-- Calculates the total score of a candidate based on the given ratio -/
def totalScore (scores : CandidateScores) : ℚ :=
  3/10 * scores.reading + 1/2 * scores.thinking + 1/5 * scores.expression

/-- The main theorem encompassing all parts of the problem -/
theorem recruitment_test_results :
  let candidateA : CandidateScores := ⟨93, 86, 73⟩
  let candidateB (x : ℚ) : CandidateScores := ⟨95, 81, x⟩
  (averageScore candidateA = 84) ∧
  (totalScore candidateA = 85.5) ∧
  (∀ x : ℚ, totalScore (candidateB x) > totalScore candidateA ↔ x > 82.5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recruitment_test_results_l540_54063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_krystiana_earnings_l540_54082

/-- Represents the monthly earnings from an apartment building --/
def apartment_earnings (
  floors : Nat
) (rooms_per_floor : Nat)
  (first_floor_price : Nat)
  (second_floor_price : Nat)
  (third_floor_price : Nat)
  (third_floor_occupied : Nat) : Nat :=
  (first_floor_price * rooms_per_floor) +
  (second_floor_price * rooms_per_floor) +
  (third_floor_price * third_floor_occupied)

/-- Krystiana's apartment building earnings theorem --/
theorem krystiana_earnings :
  apartment_earnings 3 3 15 20 30 2 = 165 := by
  rfl

#eval apartment_earnings 3 3 15 20 30 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_krystiana_earnings_l540_54082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l540_54091

/-- Given an isosceles right triangle ABC with angle A = 90°, 
    where vector AB = a + b, vector AC = a - b, and a = (cos θ, sin θ) for some real θ,
    prove that the area of the triangle is 1. -/
theorem isosceles_right_triangle_area (θ : ℝ) 
  (a b : Fin 2 → ℝ)
  (h_a : a = λ i => if i = 0 then Real.cos θ else Real.sin θ)
  (h_ab : a + b = λ _ => Real.sqrt 2)
  (h_ac : a - b = λ _ => Real.sqrt 2) :
  (1 / 2 : ℝ) * ‖a + b‖ * ‖a - b‖ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l540_54091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l540_54011

/-- Calculate the loss percentage given the cost price and selling price -/
noncomputable def loss_percentage (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1260 is 16% -/
theorem radio_loss_percentage :
  let cost_price : ℝ := 1500
  let selling_price : ℝ := 1260
  loss_percentage cost_price selling_price = 16 := by
  -- Unfold the definition of loss_percentage
  unfold loss_percentage
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l540_54011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_equality_l540_54043

theorem abc_equality (n k a b c : ℕ) 
  (hn : n > 0) 
  (hk : k > 0 ∧ k % 2 = 1) 
  (heq : (a : ℤ)^n + k*b = (b : ℤ)^n + k*c ∧ (b : ℤ)^n + k*c = (c : ℤ)^n + k*a) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_equality_l540_54043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l540_54038

/-- The number of boxes in the row -/
def num_boxes : ℕ := 2023

/-- The probability of drawing a green marble from the kth box -/
def prob_green (k : ℕ) : ℚ := 1 / (k^2 + k + 1)

/-- The probability of not drawing a green marble from the kth box -/
def prob_not_green (k : ℕ) : ℚ := 1 - prob_green k

/-- The probability of stopping after drawing exactly n marbles -/
def Q (n : ℕ) : ℚ :=
  (Finset.range (n - 1)).prod (fun k => prob_not_green (k + 1)) * prob_green n

/-- The smallest n for which Q(n) < 1/2023 -/
theorem smallest_n_for_Q_less_than_threshold : 
  (∀ m < 45, Q m ≥ 1 / num_boxes) ∧ Q 45 < 1 / num_boxes := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l540_54038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l540_54070

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) (A F1 F2 : Point) :
  on_ellipse e A →
  A.x = 1 ∧ A.y = 3/2 →
  distance A F1 + distance A F2 = 4 →
  ∃ (K : Point → Point),
    (∀ p, on_ellipse e p → on_ellipse e (K p)) ∧
    (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
    (F1.x = -1 ∧ F1.y = 0 ∧ F2.x = 1 ∧ F2.y = 0) ∧
    (∀ p, on_ellipse e p →
      let midpoint : Point := ⟨(F1.x + p.x) / 2, (F1.y + p.y) / 2⟩
      (midpoint.x + 1/2)^2 + 4 * midpoint.y^2 / 3 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l540_54070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_formula_l540_54093

/-- A cubic polynomial with specific properties -/
def q : ℝ → ℝ := sorry

/-- The graph of 1/q(x) has vertical asymptotes at x = -1, x = 1, and x = 3 -/
axiom asymptotes : ∀ (x : ℝ), (x = -1 ∨ x = 1 ∨ x = 3) → q x = 0

/-- q(x) is a cubic polynomial -/
axiom cubic : ∃ (a b c d : ℝ), ∀ (x : ℝ), q x = a * x^3 + b * x^2 + c * x + d

/-- q(2) = 12 -/
axiom q_at_2 : q 2 = 12

/-- The main theorem: q(x) = -4x³ + 12x² + 4x - 12 -/
theorem q_formula : ∀ (x : ℝ), q x = -4 * x^3 + 12 * x^2 + 4 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_formula_l540_54093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l540_54062

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the middle digit of a 3-digit number -/
def middle_digit (n : ℕ) : ℕ := sorry

/-- The set of 3-digit whole numbers whose digit sum is 25, are even, and have a middle digit greater than 5 -/
def special_numbers : Finset ℕ :=
  Finset.filter (fun n => 
    100 ≤ n ∧ n < 1000 ∧
    digit_sum n = 25 ∧ 
    Even n ∧ 
    middle_digit n > 5)
  (Finset.range 1000)

theorem count_special_numbers : special_numbers.card = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l540_54062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l540_54085

-- Define a perfect square trinomial
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (k l : ℝ), ∀ x : ℝ, a * x^2 + b * x + c = (k * x + l)^2 

-- State the theorem
theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, (is_perfect_square_trinomial 1 (-6) m) → m = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l540_54085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sequence_coefficient_l540_54074

/-- A sine sequence is a sequence where even-indexed terms are greater than their adjacent odd-indexed terms -/
def IsSineSequence (s : List ℕ) : Prop :=
  ∀ i, i % 2 = 1 → i + 2 < s.length → s[i]! < s[i + 1]! ∧ s[i + 1]! > s[i + 2]!

/-- The number of sine sequences that can be formed from the numbers 1, 2, 3, 4, 5 -/
def a : ℕ := 16

/-- The coefficient of x^2 in the expansion of (√x - a/√x)^6 -/
def coefficientOfXSquared : ℤ := -96

theorem sine_sequence_coefficient :
  coefficientOfXSquared = (-(6 : ℤ)) * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sequence_coefficient_l540_54074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_from_polynomial_roots_l540_54088

theorem min_square_area_from_polynomial_roots (p q r s : ℤ) :
  let f : ℂ → ℂ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  let roots := {z : ℂ | f z = 0}
  (∃ α β : ℝ, β ≠ 0 ∧ roots = {Complex.mk α β, Complex.mk α (-β), Complex.mk (-α) β, Complex.mk (-α) (-β)}) →
  (∀ A : ℝ, (∃ α' β' : ℝ, β' ≠ 0 ∧ 
    roots = {Complex.mk α' β', Complex.mk α' (-β'), Complex.mk (-α') β', Complex.mk (-α') (-β')} ∧ 
    A = 4 * β'^2) → A ≥ 4) :=
by sorry

#check min_square_area_from_polynomial_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_from_polynomial_roots_l540_54088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l540_54033

/-- The time taken for a train to pass a platform -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 360 m long, traveling at 45 km/hr, takes 43.2 seconds to pass a 180 m long platform -/
theorem train_platform_passing_time :
  train_passing_time 360 45 180 = 43.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l540_54033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l540_54018

/-- Given two lines in the 2D plane, this theorem proves that one line
    passes through a specific point and is parallel to the other line. -/
theorem parallel_line_through_point :
  -- Define the given line
  let given_line := (fun (x y : ℝ) => 2 * x - y - 1 = 0)
  -- Define the point that the parallel line should pass through
  let point := (-1, 1)
  -- Define the equation of the parallel line
  let parallel_line := (fun (x y : ℝ) => 2 * x - y + 3 = 0)
  -- The parallel line passes through the given point
  (parallel_line point.1 point.2) ∧
  -- The two lines are parallel (have the same slope)
  (∀ x y : ℝ, given_line x y ↔ ∃ c : ℝ, parallel_line x (y + c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l540_54018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_octagon_area_swept_area_swept_relationship_l540_54053

/-- Represents the area swept by a triangular plate inside an octagonal frame -/
structure TriangleInOctagon where
  c : ℝ  -- Side length of both the octagon and the triangle
  t₁ : ℝ  -- Area swept until reaching the opposite side
  t₂ : ℝ  -- Total area swept for a complete rotation

/-- Theorem stating the correct areas swept by the triangular plate -/
theorem triangle_in_octagon_area_swept (tio : TriangleInOctagon) :
  tio.t₁ = (tio.c^2 / 4) * (2 * Real.pi + 3 * Real.sqrt 3) ∧
  tio.t₂ = (tio.c^2 / 3) * (Real.pi + 6 * Real.sqrt 3) :=
by sorry

/-- Auxiliary function to calculate the area swept until reaching the opposite side -/
noncomputable def area_swept_half (c : ℝ) : ℝ :=
  (c^2 / 4) * (2 * Real.pi + 3 * Real.sqrt 3)

/-- Auxiliary function to calculate the total area swept for a complete rotation -/
noncomputable def area_swept_full (c : ℝ) : ℝ :=
  (c^2 / 3) * (Real.pi + 6 * Real.sqrt 3)

/-- Theorem stating the relationship between the two swept areas -/
theorem area_swept_relationship (c : ℝ) :
  area_swept_full c > area_swept_half c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_octagon_area_swept_area_swept_relationship_l540_54053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_distance_l540_54061

theorem complex_distance (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs (z₁ + z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_distance_l540_54061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_is_translation_l540_54019

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a 3D vector
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Vector3D

-- Define the reflection of a point across a line
noncomputable def reflect (p : Point3D) (l : Line3D) : Point3D :=
  sorry

-- Define three pairwise perpendicular lines
noncomputable def e₁ : Line3D := sorry
noncomputable def e₂ : Line3D := sorry
noncomputable def e₃ : Line3D := sorry

axiom perpendicular : (e₁.direction.x * e₂.direction.x + e₁.direction.y * e₂.direction.y + e₁.direction.z * e₂.direction.z = 0) ∧
                      (e₂.direction.x * e₃.direction.x + e₂.direction.y * e₃.direction.y + e₂.direction.z * e₃.direction.z = 0) ∧
                      (e₃.direction.x * e₁.direction.x + e₃.direction.y * e₁.direction.y + e₃.direction.z * e₁.direction.z = 0)

-- Define the translation vector
noncomputable def translationVector : Vector3D :=
  { x := 2 * e₁.point.x,
    y := -2 * e₂.point.y,
    z := 2 * e₃.point.z }

-- Theorem: The composition of three reflections is equivalent to a translation
theorem reflection_composition_is_translation (p : Point3D) :
  let q := reflect p e₁
  let r := reflect q e₂
  let s := reflect r e₃
  s = { x := p.x + translationVector.x,
        y := p.y + translationVector.y,
        z := p.z + translationVector.z } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_is_translation_l540_54019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_k_for_lcm_l540_54036

/-- The number of positive integers k such that 60^10 is the LCM of 10^10, 12^12, and k -/
theorem count_k_for_lcm : ∃! n : ℕ, n = (Finset.filter (fun k : ℕ => 
  Nat.lcm (10^10) (Nat.lcm (12^12) k) = 60^10) (Finset.range (60^10 + 1))).card ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_k_for_lcm_l540_54036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_inequality_l540_54013

open Real

theorem golden_ratio_inequality (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : 
  let f := λ x => log x + x^2 + x
  (f x₁ + f x₂ + x₁ * x₂ = 0) → x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_inequality_l540_54013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_problem_circles_intersecting_l540_54028

/-- Two circles intersect if the distance between their centers is greater than the absolute
    difference of their radii and less than the sum of their radii. -/
def circles_intersect (r₁ r₂ d : ℝ) : Prop :=
  d > |r₁ - r₂| ∧ d < r₁ + r₂

/-- Theorem stating the condition for two circles to intersect. -/
theorem circles_intersecting (r₁ r₂ d : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) (hd : d > 0) :
  d > |r₁ - r₂| ∧ d < r₁ + r₂ → circles_intersect r₁ r₂ d :=
by
  intro h
  exact h

/-- Given two circles with radii 3 cm and 2 cm, whose centers are 4 cm apart, prove they are intersecting. -/
theorem problem_circles_intersecting :
  circles_intersect 3 2 4 :=
by
  apply circles_intersecting
  · show 3 > 0
    norm_num
  · show 2 > 0
    norm_num
  · show 4 > 0
    norm_num
  · constructor
    · show 4 > |3 - 2|
      norm_num
    · show 4 < 3 + 2
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_problem_circles_intersecting_l540_54028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_amateurs_l540_54080

/-- The number of chess amateurs in a tournament where each amateur plays with
    exactly 15 others and the total number of games is 45. -/
def num_chess_amateurs : ℕ := 10

theorem chess_tournament_amateurs :
  num_chess_amateurs = 10 ∧
  (num_chess_amateurs * (num_chess_amateurs - 1)) / 2 = 45 ∧
  ∀ amateur : Fin num_chess_amateurs, 
    Finset.card (Finset.univ.erase amateur) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_amateurs_l540_54080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_diagonals_equal_in_rectangle_l540_54009

-- Define a structure for Quadrilateral
structure Quadrilateral where
  -- You might want to add more properties here, but for now we'll keep it empty
  mk :: 

-- Define propositions as functions
def IsRectangle (q : Quadrilateral) : Prop := sorry
def DiagonalsEqual (q : Quadrilateral) : Prop := sorry

-- Define a proposition for "The diagonals of a rectangle are equal"
def diagonals_equal_in_rectangle (q : Quadrilateral) : Prop :=
  IsRectangle q → DiagonalsEqual q

-- Define the inverse proposition
def inverse_proposition (q : Quadrilateral) : Prop :=
  DiagonalsEqual q → IsRectangle q

-- Theorem stating that the inverse proposition is correct
theorem inverse_of_diagonals_equal_in_rectangle :
  (∀ q : Quadrilateral, diagonals_equal_in_rectangle q) ↔
  (∀ q : Quadrilateral, inverse_proposition q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_diagonals_equal_in_rectangle_l540_54009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l540_54066

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 3 - 2 * x)

theorem monotonic_increasing_interval_of_f :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc ((k : ℝ) * Real.pi - Real.pi / 3) ((k : ℝ) * Real.pi + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l540_54066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_12_l540_54079

def dice_outcomes : ℕ := 6 * 6
def coin_outcomes : ℕ := 2 * 2
def total_outcomes : ℕ := dice_outcomes * coin_outcomes

def nickel_value : ℚ := 5 / 100
def quarter_value : ℚ := 25 / 100

def is_favorable_outcome (dice_sum : ℕ) (nickel_heads : Bool) (quarter_heads : Bool) : Bool :=
  let coin_value := (if nickel_heads then nickel_value else 0) + (if quarter_heads then quarter_value else 0)
  dice_sum + (coin_value * 100).floor ≥ 12

def count_favorable_outcomes : ℕ := sorry

theorem probability_at_least_12 : 
  (count_favorable_outcomes : ℚ) / total_outcomes = 47 / 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_12_l540_54079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l540_54083

open Set
open Real

-- Define the domain M
def M : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 4^x

-- Theorem statement
theorem range_of_f :
  range (f ∘ (coe : M → ℝ)) = Ioo (-32 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l540_54083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l540_54015

/-- Given an infinite geometric sequence with first term a and common ratio q,
    Sₙ represents the sum of its first n terms. -/
noncomputable def S (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

/-- Theorem: For an infinite geometric sequence with common ratio q,
    if the ratio of S₆ to S₃ is 9/8, then q = 1/2. -/
theorem geometric_sequence_ratio (a q : ℝ) (h₁ : q ≠ 1) :
  (S a q 6) / (S a q 3) = 9/8 → q = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l540_54015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_viewing_directions_l540_54006

/-- Represents the viewing direction for orthographic projection -/
inductive ViewingDirection
  | Front
  | Top
  | Side

/-- Represents the observation direction for orthographic projection -/
inductive ObservationDirection
  | DirectlyInFront
  | DirectlyAbove
  | DirectlyToTheLeft

/-- Theorem stating the correct observation directions for each view in orthographic projection -/
theorem orthographic_projection_viewing_directions :
  ∀ (vd : ViewingDirection),
    (vd = ViewingDirection.Front → ObservationDirection.DirectlyInFront = ObservationDirection.DirectlyInFront) ∧
    (vd = ViewingDirection.Top → ObservationDirection.DirectlyAbove = ObservationDirection.DirectlyAbove) ∧
    (vd = ViewingDirection.Side → ObservationDirection.DirectlyToTheLeft = ObservationDirection.DirectlyToTheLeft) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_viewing_directions_l540_54006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l540_54069

theorem power_comparison : (2 : ℝ)^(0.6 : ℝ) > (0.6 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l540_54069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l540_54046

noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (abs (a^2 - b^2)) / max a b

theorem conic_section_eccentricity :
  let m : ℝ := geometric_mean 2 8
  let e₁ : ℝ := eccentricity 2 1  -- For the ellipse case
  let e₂ : ℝ := eccentricity 1 2  -- For the hyperbola case
  (e₁ = Real.sqrt 3 / 2 ∧ e₂ = Real.sqrt 5) ∨
  (e₁ = Real.sqrt 5 ∧ e₂ = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l540_54046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l540_54094

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

-- Define the shifted function
noncomputable def f_shifted (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

-- Define symmetry about y-axis
def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem min_shift_for_symmetry :
  ∃ φ : ℝ, φ > 0 ∧
    symmetric_about_y_axis (f_shifted φ) ∧
    (∀ ψ, ψ > 0 → symmetric_about_y_axis (f_shifted ψ) → φ ≤ ψ) ∧
    φ = Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l540_54094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l540_54022

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

-- State the theorem
theorem function_properties (m : ℝ) (h : f m (Real.pi / 2) = 1) :
  -- 1. Explicit formula
  (∀ x, f m x = Real.sqrt 2 * Real.sin (x + Real.pi / 4)) ∧
  -- 2. Smallest positive period
  (∃ T > 0, ∀ x, f m (x + T) = f m x ∧ 
    ∀ S, (0 < S ∧ S < T) → ∃ y, f m (y + S) ≠ f m y) ∧
  -- 3. Maximum value
  (∀ x, f m x ≤ Real.sqrt 2 ∧ ∃ y, f m y = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l540_54022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l540_54081

/-- The perimeter of a quadrilateral with vertices at (1,2), (4,5), (5,4), and (4,1) -/
noncomputable def quadrilateral_perimeter : ℝ :=
  let d1 := Real.sqrt ((4 - 1)^2 + (5 - 2)^2)
  let d2 := Real.sqrt ((5 - 4)^2 + (4 - 5)^2)
  let d3 := Real.sqrt ((4 - 5)^2 + (1 - 4)^2)
  let d4 := Real.sqrt ((1 - 4)^2 + (2 - 1)^2)
  d1 + d2 + d3 + d4

theorem quadrilateral_perimeter_sum (c d : ℤ) :
  quadrilateral_perimeter = c * Real.sqrt 2 + d * Real.sqrt 10 →
  c + d = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l540_54081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_final_amounts_l540_54027

/-- Represents the amounts of different currencies --/
structure CurrencyAmounts where
  usd : ℝ
  quarters : ℕ
  eur : ℝ
  dimes : ℕ
  jpy : ℝ
  gbp : ℝ
  chf : ℝ
  cad : ℝ
  aud : ℝ

/-- Represents the exchange rates --/
structure ExchangeRates where
  usd_to_gbp : ℝ
  eur_to_gbp : ℝ
  usd_to_chf : ℝ
  eur_to_chf : ℝ
  jpy_to_cad : ℝ
  eur_to_aud : ℝ

/-- Represents the transactions --/
structure Transactions where
  usd_to_exchange : ℝ
  eur_to_exchange : ℝ
  jpy_to_cad : ℝ
  eur_to_aud : ℝ
  given_to_sister : CurrencyAmounts

def initial_amounts : CurrencyAmounts := {
  usd := 40,
  quarters := 200,
  eur := 15,
  dimes := 100,
  jpy := 3000,
  gbp := 0,
  chf := 0,
  cad := 0,
  aud := 0
}

def exchange_rates : ExchangeRates := {
  usd_to_gbp := 0.75,
  eur_to_gbp := 0.85,
  usd_to_chf := 0.90,
  eur_to_chf := 1.05,
  jpy_to_cad := 0.012,
  eur_to_aud := 1.50
}

def transactions : Transactions := {
  usd_to_exchange := 10,
  eur_to_exchange := 5,
  jpy_to_cad := 2000,
  eur_to_aud := 8,
  given_to_sister := {
    usd := 5,
    quarters := 120,
    eur := 0,
    dimes := 50,
    jpy := 500,
    gbp := 3.5,
    chf := 2,
    cad := 0,
    aud := 7
  }
}

def final_amounts : CurrencyAmounts := {
  usd := 20,
  quarters := 0,
  eur := 2,
  dimes := 0,
  jpy := 0,
  gbp := 8.25,
  chf := 12.25,
  cad := 24,
  aud := 5
}

def calculate_final_amounts (initial : CurrencyAmounts) (rates : ExchangeRates) (trans : Transactions) : CurrencyAmounts :=
sorry

theorem oliver_final_amounts (initial : CurrencyAmounts) (rates : ExchangeRates) (trans : Transactions) :
  initial = initial_amounts → rates = exchange_rates → trans = transactions →
  (calculate_final_amounts initial rates trans) = final_amounts :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_final_amounts_l540_54027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_l540_54039

open Real

variable (α β γ : ℝ)

noncomputable def f (a b c : ℝ) : ℝ := (1/3) * (a + b + c)

noncomputable def x : ℝ := f α β γ
noncomputable def y : ℝ := f (arcsin α) (arcsin β) (arcsin γ)
noncomputable def z : ℝ := arcsin (f α β γ)
noncomputable def w : ℝ := sin (f α β γ)
noncomputable def t : ℝ := f (sin α) (sin β) (sin γ)

theorem inequality_chain (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) (hγ : 0 < γ ∧ γ < 1) :
  y α β γ > z α β γ ∧ z α β γ > x α β γ ∧ x α β γ > w α β γ ∧ w α β γ > t α β γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_l540_54039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_of_25_sided_polygon_l540_54059

structure ConvexPolygon where
  sides : ℕ
  diagonals : ℕ

theorem diagonals_of_25_sided_polygon : 
  ∀ (P : ConvexPolygon), 
    P.sides = 25 → P.diagonals = 275 := by
  intro P h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_of_25_sided_polygon_l540_54059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l540_54067

theorem sin_plus_cos_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo (3*Real.pi/4) Real.pi) 
  (h2 : Real.cos (θ + Real.pi/4) * Real.cos (θ - Real.pi/4) = Real.sqrt 3/4) : 
  Real.sin θ + Real.cos θ = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l540_54067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_rounding_to_hundredths_l540_54060

/-- Rounds a number to the nearest hundredth -/
noncomputable def round_to_hundredths (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

theorem correct_rounding_to_hundredths :
  let x : ℝ := 5.13586
  round_to_hundredths x = 5.14 ∧ round_to_hundredths x ≠ 5.136 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_rounding_to_hundredths_l540_54060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l540_54095

theorem problem_solution : 
  ((∀ (x y z : ℝ), x = Real.sqrt 8 ∧ y = Real.sqrt 18 ∧ z = Real.sqrt 2 → (x + y) / z = 5) ∧
   (∀ (a : ℝ), (1 - Real.sqrt 3)^2 + 3 * Real.sqrt (1/3) = 4 - Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l540_54095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l540_54064

/-- Represents a salt solution with a given volume and concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents the mixture of salt solutions -/
noncomputable def mix (s1 s2 s3 : SaltSolution) : SaltSolution :=
  { volume := s1.volume + s2.volume + s3.volume,
    concentration := (s1.volume * s1.concentration + s2.volume * s2.concentration + s3.volume * s3.concentration) / (s1.volume + s2.volume + s3.volume) }

theorem salt_solution_mixture :
  let initial := SaltSolution.mk 70 0.20
  let solution1 := SaltSolution.mk 122 0.60
  let solution2 := SaltSolution.mk 8 0.35
  let final := mix initial solution1 solution2
  final.volume = 200 ∧ final.concentration = 0.45 := by
    sorry

#eval "Salt solution mixture theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l540_54064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_product_l540_54034

theorem determinant_product (a b c d e f g h : ℝ) :
  Matrix.det !![a*e + b*g, c*e + d*g; a*f + b*h, c*f + d*h] =
  Matrix.det !![a, b; c, d] * Matrix.det !![e, f; g, h] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_product_l540_54034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l540_54089

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : a > b
  h' : b > 0
  h'' : e = 1/2
  h''' : a - (a * e) = 1

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / E.a^2 + y^2 / E.b^2 = 1
  h' : (x, y) ≠ (-2, 0) ∧ (x, y) ≠ (2, 0)

/-- Theorem about the ellipse equation and slope product -/
theorem ellipse_properties (E : Ellipse) :
  (∀ x y, x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / E.a^2 + y^2 / E.b^2 = 1) ∧
  (∀ P : PointOnEllipse E, 
    (P.y / (P.x + 2)) * (P.y / (P.x - 2)) = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l540_54089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l540_54010

/-- Sum of a geometric series with n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The sum of the geometric series with a = -1, r = 3, and n = 7 is -1093 -/
theorem specific_geometric_sum :
  geometric_sum (-1 : ℝ) 3 7 = -1093 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l540_54010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l540_54044

theorem angle_quadrant (α : Real) (m : Real) :
  m ≠ 0 →
  (∃ P : Real × Real, P = (-Real.sqrt 3, m) ∧ P ∈ Set.range (λ t : Real => (t * Real.cos α, t * Real.sin α))) →
  Real.sin α = (Real.sqrt 3 / 4) * m →
  (α > Real.pi / 2 ∧ α < Real.pi) ∨ (α > Real.pi ∧ α < 3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l540_54044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l540_54024

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^2 / ((x - 3) * (x + 2))

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, x ≠ 3 ∧ x ≠ -2 →
  (f x ≥ 0 ↔ x ∈ Set.Ici 3 ∪ Set.Iic (-2)) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l540_54024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_necessary_not_sufficient_l540_54055

open Real

-- Define the property P
def P (α β : ℝ) : Prop := sin α + cos β = 0

-- Define the condition Q
def Q (α β : ℝ) : Prop := sin α ^ 2 + sin β ^ 2 = 1

-- Theorem statement
theorem sin_squared_necessary_not_sufficient :
  (∀ α β : ℝ, P α β → Q α β) ∧ 
  ¬(∀ α β : ℝ, Q α β → P α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_necessary_not_sufficient_l540_54055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_plus_cos_l540_54016

theorem min_sin_plus_cos (A : Real) : 
  (∀ θ : Real, Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sin (θ / 2) + Real.cos (θ / 2)) → 
  A = 450 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_plus_cos_l540_54016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l540_54072

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a * abc.c * Real.cos abc.B - abc.b * abc.c * Real.cos abc.A = 3 * abc.b^2)
  (h2 : 0 < abc.C ∧ abc.C < Real.pi/2)  -- C is acute
  (h3 : abc.c = Real.sqrt 11)
  (h4 : Real.sin abc.C = 2 * Real.sqrt 2 / 3) :
  abc.a / abc.b = 2 ∧ 
  (1/2) * abc.a * abc.b * Real.sin abc.C = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l540_54072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_distance_l540_54099

/-- Proves that B catches up with A 140 km from the start, given the specified conditions. -/
theorem catch_up_distance (speed_A speed_B : ℝ) (delay : ℝ) (catch_up_distance : ℝ) : 
  speed_A = 10 →
  speed_B = 20 →
  delay = 7 →
  catch_up_distance = speed_B * (delay + (catch_up_distance - delay * speed_A) / (speed_B - speed_A)) →
  catch_up_distance = 140 := by
  intros h_speed_A h_speed_B h_delay h_catch_up
  -- The proof steps would go here
  sorry

#check catch_up_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_distance_l540_54099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drill_bits_tax_rate_l540_54003

theorem drill_bits_tax_rate 
  (num_sets : ℕ) 
  (cost_per_set : ℚ) 
  (total_paid : ℚ) 
  (h1 : num_sets = 5)
  (h2 : cost_per_set = 6)
  (h3 : total_paid = 33) : ℚ := by
  let tax_rate := (total_paid - num_sets * cost_per_set) / (num_sets * cost_per_set)
  have : tax_rate = 10 / 100 := by
    sorry
  exact 10 / 100

#check drill_bits_tax_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drill_bits_tax_rate_l540_54003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factorization_l540_54032

theorem cubic_factorization (a b c : ℕ) (h1 : a > b) (h2 : b > c) :
  (fun x : ℝ ↦ x^3 - 16*x^2 + 65*x - 80) = 
  (fun x : ℝ ↦ (x - a) * (x - b) * (x - c)) →
  3*b - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factorization_l540_54032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_correct_l540_54025

/-- The speed of a particle with position (t^2 + 2t + 7, 3t - 13) at time t -/
noncomputable def particleSpeed (t : ℝ) : ℝ :=
  Real.sqrt (4 * t^2 + 12 * t + 18)

/-- The position of the particle at time t -/
def particlePosition (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2*t + 7, 3*t - 13)

theorem particle_speed_correct (t : ℝ) :
  let (x, y) := particlePosition t
  let (x', y') := particlePosition (t + 1)
  Real.sqrt ((x' - x)^2 + (y' - y)^2) = particleSpeed t := by
  sorry

#check particle_speed_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_correct_l540_54025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_interpretations_count_l540_54084

/-- The number of ways to fully parenthesize n factors -/
def my_catalan (n : ℕ) : ℕ := (Nat.choose (2 * n) n) / (n + 1)

/-- The number of ways to fully parenthesize 5 factors -/
def ways_to_parenthesize_5_factors : ℕ := my_catalan 4

theorem product_interpretations_count :
  ways_to_parenthesize_5_factors = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_interpretations_count_l540_54084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_solvability_l540_54035

/-- Given a triangle ABC with circumcenter O and point F, the problem has a solution
    if and only if there exists a point D satisfying specific geometric conditions. -/
theorem triangle_construction_solvability
  (A O F : EuclideanSpace ℝ (Fin 2))
  (h_distinct : A ≠ O ∧ A ≠ F ∧ O ≠ F) :
  (∃ (B C D : EuclideanSpace ℝ (Fin 2)),
    -- D is on the Thales circle with diameter OA
    ‖D - O‖ = ‖D - A‖ ∧
    -- D is on the reflection of the circumcircle about F
    ‖D - F‖ = ‖O - F‖ ∧
    -- D is the midpoint of AC
    2 • D = A + C ∧
    -- B is the reflection of D about F
    B = 2 • F - D ∧
    -- O is the circumcenter of triangle ABC
    ‖O - A‖ = ‖O - B‖ ∧ ‖O - B‖ = ‖O - C‖) ↔
  -- Conditions for the existence of a solution
  ∃ (D : EuclideanSpace ℝ (Fin 2)),
    ‖D - O‖ = ‖D - A‖ ∧
    ‖D - F‖ = ‖O - F‖ ∧
    ∃ (C : EuclideanSpace ℝ (Fin 2)), 2 • D = A + C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_solvability_l540_54035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l540_54056

theorem ordering_of_abc : ∀ (a b c : ℝ), 
  a = 3^(1/4 : ℝ) - 3^(-(1/4) : ℝ) → 
  b = (1/2) * Real.log 3 → 
  c = 4 - 2 * Real.sqrt 3 → 
  a > b ∧ b > c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l540_54056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_four_l540_54075

/-- The set S containing integers from 1 to 13 -/
def S : Finset Nat := Finset.range 13

/-- A family of subsets of S -/
def A : Nat → Finset Nat := sorry

/-- The number of subsets in the family A -/
def k : Nat := sorry

/-- Each subset in A has 6 elements -/
axiom subset_size : ∀ i, (A i).card = 6

/-- Each subset in A is a subset of S -/
axiom subset_of_S : ∀ i, A i ⊆ S

/-- The intersection of any two distinct subsets in A has at most 2 elements -/
axiom intersection_size : ∀ i j, i < j → (A i ∩ A j).card ≤ 2

/-- The maximum value of k is 4 -/
theorem max_k_is_four : k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_four_l540_54075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l540_54030

theorem coefficient_x_cubed_in_expansion : 
  (λ r : ℕ => (-1)^r * 2^(5-r) * (Nat.choose 5 r : ℤ) * X^(5-2*r)) 1 / X^3 = -80 := by
  -- Define the binomial expansion of (2x-1/x)^5
  let expansion := (λ r : ℕ => (-1)^r * 2^(5-r) * (Nat.choose 5 r : ℤ) * X^(5-2*r))
  
  -- The coefficient of x^3 is the term where 5-2r = 3, which gives r = 1
  let r := 1
  
  -- The coefficient is the value of the expansion term without X^3
  let coefficient := expansion r / X^3
  
  -- Assert that this coefficient is equal to -80
  sorry

#check coefficient_x_cubed_in_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l540_54030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_fourth_quadrant_z_bottom_right_line_l540_54073

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 + 3*m - 28)

theorem z_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ m > -7 ∧ m < 3 := by sorry

theorem z_bottom_right_line (m : ℝ) :
  (z m).im < 2 * (z m).re - 40 ↔ m < 1 ∨ m > 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_fourth_quadrant_z_bottom_right_line_l540_54073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l540_54021

/-- The slope of the angle bisector of two intersecting lines -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ - Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

/-- Theorem: The slope of the angle bisector of y = (3/2)x and y = 2x - 1 -/
theorem angle_bisector_slope_specific : 
  angle_bisector_slope (3/2) 2 = (7 - Real.sqrt 29) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l540_54021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_MNO_l540_54076

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Represents the triangle MNO in the prism -/
structure TriangleMNO (prism : RightPrism) where
  M : Point3D
  N : Point3D
  O : Point3D

/-- Calculates the perimeter of triangle MNO -/
noncomputable def perimeterMNO (prism : RightPrism) (triangle : TriangleMNO prism) : ℝ :=
  distance triangle.M triangle.N + distance triangle.N triangle.O + distance triangle.O triangle.M

/-- Theorem stating the perimeter of triangle MNO in the given prism -/
theorem perimeter_of_triangle_MNO (prism : RightPrism)
  (h1 : prism.height = 20)
  (h2 : prism.baseSideLength = 10)
  (triangle : TriangleMNO prism)
  (h3 : triangle.M.z = 0)
  (h4 : triangle.N.z = 0)
  (h5 : triangle.O.z = prism.height / 2)
  (h6 : distance triangle.M triangle.N = prism.baseSideLength / 2) :
  perimeterMNO prism triangle = 5 + 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_MNO_l540_54076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_division_l540_54071

/-- Represents a partnership between two people --/
structure Partnership where
  investment1 : ℚ
  investment2 : ℚ
  total_profit : ℚ
  equal_portion : ℚ

/-- Calculates the profit share for a partner --/
noncomputable def profit_share (p : Partnership) (investment : ℚ) : ℚ :=
  p.equal_portion / 2 + (investment / (p.investment1 + p.investment2)) * (p.total_profit - p.equal_portion)

/-- Theorem stating the conditions and the result to be proved --/
theorem partnership_profit_division 
  (p : Partnership) 
  (h1 : p.investment1 = 550)
  (h2 : p.investment2 = 450)
  (h3 : p.total_profit = 15000)
  (h4 : profit_share p p.investment1 = profit_share p p.investment2 + 1000) :
  p.equal_portion = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_division_l540_54071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_admission_price_correct_l540_54001

/-- Calculates the admission price for children given the total revenue, attendance, and number of children. -/
def children_admission_price (adult_price : ℕ) (total_attendance : ℕ) 
  (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := total_attendance - num_children
  (total_revenue * 100 - num_adults * adult_price) / num_children

/-- Theorem stating that the calculated children's admission price is correct. -/
theorem children_admission_price_correct (adult_price : ℕ) (total_attendance : ℕ) 
  (total_revenue : ℕ) (num_children : ℕ) :
  let child_price := children_admission_price adult_price total_attendance total_revenue num_children
  let num_adults := total_attendance - num_children
  num_adults * adult_price + num_children * child_price = total_revenue * 100 :=
by
  -- Unfold the definition of children_admission_price
  simp [children_admission_price]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval children_admission_price 60 280 140 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_admission_price_correct_l540_54001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l540_54090

noncomputable def m (x : ℝ) : ℝ × ℝ := (1/2 * Real.sin x, Real.sqrt 3 / 2)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x ^ 2 - 1/2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_side_b (A B : ℝ) (a b : ℝ) :
  f A = 0 →
  Real.sin B = 4/5 →
  a = Real.sqrt 3 →
  b = 8/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l540_54090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l540_54098

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - 3 * Real.cos x + 2

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 5 := by
  -- We'll use M = 5 as the maximum value
  let M := 5
  
  -- Prove that f x ≤ M for all x
  have h1 : ∀ (x : ℝ), f x ≤ M := by
    intro x
    sorry -- The actual proof would go here
  
  -- Prove that there exists an x such that f x = M
  have h2 : ∃ (x : ℝ), f x = M := by
    use Real.pi -- x = π maximizes the function
    sorry -- The actual proof would go here
  
  -- Combine the results
  exact ⟨M, h1, h2, rfl⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l540_54098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l540_54041

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ((5*x + x^5) / (1 + 5*x^4)) = 5 * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l540_54041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_division_l540_54096

theorem right_angle_division : ∃ (α β : Real),
  0 < α ∧ α < π/2 ∧
  0 < β ∧ β < π/2 ∧
  α + β = π/2 ∧
  Real.sin α - Real.sin β = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_division_l540_54096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_pairs_l540_54007

theorem integer_fraction_pairs (a b : ℕ) :
  (∃ k : ℤ, k * (b^2 - a) = a^2 + b) ∧
  (∃ m : ℤ, m * (a^2 - b) = b^2 + a) →
  ((a = 1 ∧ b = 2) ∨
   (a = 2 ∧ b = 1) ∨
   (a = 2 ∧ b = 2) ∨
   (a = 2 ∧ b = 3) ∨
   (a = 3 ∧ b = 2) ∨
   (a = 3 ∧ b = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_pairs_l540_54007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l540_54037

/-- The inclination angle of a line passing through two given points -/
noncomputable def inclinationAngle (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.arctan ((y₂ - y₁) / (x₂ - x₁))

/-- Theorem: The inclination angle of a line passing through (1, 2) and (4, 2 + √3) is π/6 -/
theorem line_inclination_angle : inclinationAngle 1 2 4 (2 + Real.sqrt 3) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l540_54037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_line_equation_altitude_line_equation_l540_54008

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (2, -2)
noncomputable def B : ℝ × ℝ := (6, 6)
noncomputable def C : ℝ × ℝ := (0, 6)

-- Define the midpoint M of AB
noncomputable def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Function to calculate the perpendicular vector
def perpendicular (v : ℝ × ℝ) : ℝ × ℝ := (-v.2, v.1)

-- Theorem for the median line CM
theorem median_line_equation :
  ∀ (x y : ℝ), (x + y - 6 = 0) ↔ (∃ t : ℝ, (x, y) = (1 - t) • M + t • C) :=
sorry

-- Theorem for the altitude line from C to AB
theorem altitude_line_equation :
  ∀ (x y : ℝ), (x + 2*y - 12 = 0) ↔ (∃ t : ℝ, (x, y) = C + t • perpendicular (B.1 - A.1, B.2 - A.2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_line_equation_altitude_line_equation_l540_54008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l540_54029

/-- An ellipse with the given properties has a major axis of length 6 -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
    (∃ (x y : ℝ), (x, 0) ∈ E ∧ (0, y) ∈ E) →  -- tangent to x and y axes
    ((2, -3 + Real.sqrt 5) ∈ E ∧ (2, -3 - Real.sqrt 5) ∈ E) →  -- foci locations
    (∃ (a b : ℝ × ℝ), a ∈ E ∧ b ∈ E ∧ dist a b = 6) :=  -- major axis length
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l540_54029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_goats_count_l540_54031

/-- Represents the number of animals on a farm -/
structure Farm where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The initial state of the farm -/
def initial_farm (g : ℕ) : Farm :=
  { cows := 2, pigs := 3, goats := g }

/-- The added animals -/
def added_animals : Farm :=
  { cows := 3, pigs := 5, goats := 2 }

/-- The total number of animals after adding -/
def total_animals (f : Farm) : ℕ :=
  f.cows + f.pigs + f.goats

/-- Addition operation for Farm -/
instance : Add Farm where
  add f1 f2 := { 
    cows := f1.cows + f2.cows, 
    pigs := f1.pigs + f2.pigs, 
    goats := f1.goats + f2.goats 
  }

/-- Theorem stating the initial number of goats -/
theorem initial_goats_count :
  ∃ g : ℕ, 
    total_animals (initial_farm g + added_animals) = 21 ∧ 
    g = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_goats_count_l540_54031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l540_54057

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ M : Point) :
  (∃ (k : ℝ), M.y = k * M.x ∧ k^2 * h.a^2 = h.b^2) →  -- M is on the asymptote
  (M.x - F₁.x) * (M.x - F₂.x) + (M.y - F₁.y) * (M.y - F₂.y) = 0 →  -- MF₁ ⟂ MF₂
  Real.sin (Real.arctan ((M.y - F₁.y) / (M.x - F₁.x))) = 1 / 3 →  -- sin ∠MF₁F₂ = 1/3
  eccentricity h = 9 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l540_54057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_count_l540_54004

theorem team_games_count (G : ℚ) : 
  (0.55 * 35 + 0.9 * (G - 35) = 0.8 * G) → G = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_count_l540_54004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l540_54014

/-- Given a parabola y^2 = 2px (p > 0) and a line passing through F(p/2, 0)
    intersecting the parabola at A and B, if |AF|:|BF| = 3:1,
    then the slope of the line is ± √3 -/
theorem parabola_intersection_slope (p : ℝ) (hp : p > 0) 
    (A B F : ℝ × ℝ) (hF : F = (p/2, 0)) 
    (hA : A.2^2 = 2*p*A.1) (hB : B.2^2 = 2*p*B.1)
    (hAF_BF : (dist A F) / (dist B F) = 3) :
  let slope := (A.2 - F.2) / (A.1 - F.1)
  slope = Real.sqrt 3 ∨ slope = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l540_54014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_expansion_l540_54052

/-- The coefficient of x^3y^3 in the expansion of (x+y)(2x-y)^5 is 40 -/
theorem coefficient_x3y3_expansion : ∃ (c : ℤ), c = 40 ∧ 
  (∀ (x y : ℝ), (x + y) * (2*x - y)^5 = c * x^3 * y^3 + 
    ((fun (a b : ℝ) => (a + b) * (2*a - b)^5 - c * a^3 * b^3) x y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_expansion_l540_54052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_function_range_l540_54020

def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def g (c : ℝ) (x : ℝ) : ℝ := -x^3 + 3*c*x

theorem tangent_point_and_function_range (a b c : ℝ) :
  (∀ x > 0, f a b x = x^3 + a*x^2 + b*x) →
  f a b 3 = 0 →
  (deriv (f a b)) 3 = 0 →
  (∀ x, g c x + f a b x = -6*x^2 + (3*c + 9)*x) →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |g c x₁ - g c x₂| ≤ 1) →
  ((∀ x, f a b x = x^3 - 6*x^2 + 9*x) ∧ (1/6 ≤ c ∧ c ≤ (4^(1/3))/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_function_range_l540_54020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l540_54092

noncomputable def proj (a : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * a.1 + v.2 * a.2) / (a.1^2 + a.2^2)
  (scalar * a.1, scalar * a.2)

theorem line_equation_from_projection (x y : ℝ) :
  proj (3, 4) (x, y) = (9/2, 6) →
  y = -3/4 * x + 75/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l540_54092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l540_54068

theorem cos_neg_x_minus_pi_half (x : ℝ) :
  x ∈ Set.Ioo (π / 2) π →
  Real.tan x = -4 / 3 →
  Real.cos (-x - π / 2) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l540_54068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_X_bounds_l540_54042

/-- Triangle ABC with given properties -/
structure TriangleABC where
  BC : ℝ
  AD : ℝ  -- median from A to BC
  BE : ℝ  -- median from B to AC
  bc_eq : BC = 10
  ad_eq : AD = 6
  be_eq : BE = 7.5

/-- Sum of squares of sides of triangle ABC -/
noncomputable def X (t : TriangleABC) : ℝ := t.BC^2 + (t.AD^2 - (t.BC/2)^2) + (t.BE^2 - (t.AD/2)^2)

/-- Theorem about the maximum and minimum values of X -/
theorem triangle_X_bounds (t : TriangleABC) :
  (X t ≤ 167.25) ∧ (X t ≥ 111) ∧ (167.25 - 111 = 56.25) := by
  sorry

#eval 167.25 - 111  -- This line is added to check the arithmetic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_X_bounds_l540_54042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l540_54026

noncomputable def A : ℝ × ℝ := (-4, -1)
noncomputable def B : ℝ × ℝ := (-3, 2)
noncomputable def C : ℝ × ℝ := (3, 2)
noncomputable def D : ℝ × ℝ := (4, -1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem point_P_y_coordinate :
  ∃ (P : ℝ × ℝ), 
    distance P A + distance P D = 10 ∧
    distance P B + distance P C = 10 ∧
    P.1 > 0 →
    P.2 = 2/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l540_54026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l540_54005

theorem unique_solution : ∃! (x y z n : ℕ), 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧
  (x : ℤ)^(2*n+1) - (y : ℤ)^(2*n+1) = (x * y * z : ℤ) + 2^(2*n+1) ∧
  n ≥ 2 ∧
  z ≤ 5 * 2^(2*n) ∧
  x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l540_54005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l540_54040

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sqrt x + Real.sqrt (x * (x - 1))

-- Define the domain set
def domain : Set ℝ := {x : ℝ | x = 0 ∨ x ≥ 1}

-- Theorem stating that the domain of f is equal to the defined domain set
theorem f_domain : {x : ℝ | ∃ y, f x = y} = domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l540_54040
