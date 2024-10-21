import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_inscribed_circle_value_l807_80719

/-- An isosceles right triangle with an inscribed circle of radius 2 -/
structure IsoscelesRightTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The radius of the inscribed circle is 2 -/
  inscribed_radius : leg = 4 + 2 * Real.sqrt 2

/-- The distance from the vertex of the acute angle to the point where the inscribed circle
    touches the leg opposite to this angle -/
noncomputable def distance_to_inscribed_circle (t : IsoscelesRightTriangle) : ℝ :=
  2 * Real.sqrt (7 + 4 * Real.sqrt 2)

/-- Theorem stating the distance from the vertex of the acute angle to the point where
    the inscribed circle touches the leg opposite to this angle -/
theorem distance_to_inscribed_circle_value (t : IsoscelesRightTriangle) :
  distance_to_inscribed_circle t = 2 * Real.sqrt (7 + 4 * Real.sqrt 2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_inscribed_circle_value_l807_80719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_7_sqrt_3_l807_80763

/-- Triangle ABC with right angle at B, D is foot of altitude from B to AC -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_right_angle_B : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  BD_perp_AC : (D.1 - B.1) * (C.1 - A.1) + (D.2 - B.2) * (C.2 - A.2) = 0

/-- The lengths of AD and DC segments -/
def segment_lengths (t : RightTriangle) : ℝ × ℝ :=
  (3, 4)

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- The main theorem: area of triangle ABC is 7√3 -/
theorem area_is_7_sqrt_3 (t : RightTriangle) :
  triangle_area t.A t.B t.C = 7 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_7_sqrt_3_l807_80763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l807_80725

/-- The plane equation passing through point A and perpendicular to vector BC -/
theorem plane_equation_through_point_perpendicular_to_vector 
  (A B C : ℝ × ℝ × ℝ) 
  (hA : A = (-8, 0, 7)) 
  (hB : B = (-3, 2, 4)) 
  (hC : C = (-1, 4, 5)) : 
  ∃ (a b c d : ℝ), 
    (∀ (x y z : ℝ), (x, y, z) ∈ {p : ℝ × ℝ × ℝ | a * p.fst + b * p.snd.fst + c * p.snd.snd + d = 0} ↔ 
      (2 * x + 2 * y + z + 9 = 0)) ∧
    (a * A.fst + b * A.snd.fst + c * A.snd.snd + d = 0) ∧
    ∃ (k : ℝ), k ≠ 0 ∧ (a, b, c) = k • (C.fst - B.fst, C.snd.fst - B.snd.fst, C.snd.snd - B.snd.snd) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l807_80725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_for_ten_not_fourteen_l807_80742

/-- Represents a sequence of n real numbers in [0,1] -/
def ValidSequence (n : ℕ) := { s : Fin n → ℝ // ∀ i, 0 ≤ s i ∧ s i ≤ 1 }

/-- Checks if a sequence satisfies the distribution condition for k elements -/
def SatisfiesCondition (n : ℕ) (s : ValidSequence n) (k : ℕ) : Prop :=
  ∀ j : Fin k, ∃ i : Fin n, (j : ℝ) / k ≤ s.val i ∧ s.val i < ((j : ℝ) + 1) / k

/-- A sequence is a solution if it satisfies all conditions up to n -/
def IsSolution (n : ℕ) (s : ValidSequence n) : Prop :=
  ∀ k : Fin n, SatisfiesCondition n s (k + 1)

theorem solution_exists_for_ten_not_fourteen :
  (∃ s : ValidSequence 10, IsSolution 10 s) ∧ 
  (¬ ∃ s : ValidSequence 14, IsSolution 14 s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_for_ten_not_fourteen_l807_80742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_alex_meets_train_l807_80723

open Set MeasureTheory

/-- Train arrival time in minutes after 12:00 PM -/
def train_arrival : Set ℝ := Icc 0 120

/-- Alex arrival time in minutes after 12:00 PM -/
def alex_arrival : Set ℝ := Icc 60 180

/-- Train waiting time in minutes -/
def train_wait : ℝ := 15

/-- Event where Alex arrives while the train is still at the station -/
def alex_meets_train : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ alex_arrival ∧ p.2 ∈ train_arrival ∧ p.1 ≤ p.2 + train_wait}

/-- Probability of Alex meeting the train -/
theorem prob_alex_meets_train : 
  (volume alex_meets_train) / (volume (alex_arrival.prod train_arrival)) = 7 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_alex_meets_train_l807_80723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_triangle_l807_80720

noncomputable def perimeter (X Y Z : ℝ) : ℝ :=
  sorry

theorem least_perimeter_triangle (X Y Z : ℝ) (x y z : ℕ) : 
  Real.cos X = 3/5 → Real.cos Y = 1/2 → Real.cos Z = -1/3 → 
  x + y + z = perimeter X Y Z →
  perimeter X Y Z ≥ 78 :=
by sorry

notation "cos" => Real.cos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_triangle_l807_80720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_positive_l807_80750

theorem negation_of_sin_positive :
  (¬ ∀ x : ℝ, x > 0 → Real.sin x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sin x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_positive_l807_80750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_through_1_2_with_equal_intercepts_l807_80759

/-- A line in two-dimensional space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Checks if a line has equal intercepts on the coordinate axes -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.slope ≠ 0 ∧ abs (l.intercept / l.slope) = abs l.intercept

/-- The set of lines passing through (1, 2) with equal intercepts -/
def linesThrough1_2WithEqualIntercepts : Set Line :=
  {l : Line | l.passesThrough 1 2 ∧ l.hasEqualIntercepts}

theorem two_lines_through_1_2_with_equal_intercepts :
  ∃ (s : Finset Line), s.card = 2 ∧ ∀ l : Line, l ∈ s ↔ l ∈ linesThrough1_2WithEqualIntercepts := by
  sorry

#check two_lines_through_1_2_with_equal_intercepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_through_1_2_with_equal_intercepts_l807_80759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l807_80701

def sequenceProperty (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → 2 * n * a (n + 1) = (n + 1) * a n

theorem sequence_general_term (a : ℕ → ℚ) (h : sequenceProperty a) :
  ∀ n : ℕ, n ≥ 1 → a n = n / (2^(n - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l807_80701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_players_l807_80704

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players not among the lowest 12
  total_players : n + 12 > 0
  points_distribution : 
    2 * (Nat.choose n 2) + 132 = ((n + 12) * (n + 11)) / 2
  lowest_twelve_condition : 
    66 = Nat.choose 12 2

/-- The total number of players in a chess tournament with the given conditions is 35 -/
theorem chess_tournament_players (t : ChessTournament) : 
  t.n + 12 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_players_l807_80704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_fuel_consumption_l807_80736

/-- Represents the fuel efficiency of a car in different environments --/
structure FuelEfficiency where
  highway : ℚ
  city : ℚ

/-- Represents a segment of the road trip --/
structure TripSegment where
  distance : ℚ
  isHighway : Bool

/-- Calculates the total gallons of gasoline needed for a road trip --/
def totalGallonsNeeded (efficiency : FuelEfficiency) (segments : List TripSegment) : ℚ :=
  segments.foldl (fun acc segment =>
    acc + segment.distance / (if segment.isHighway then efficiency.highway else efficiency.city)
  ) 0

/-- Theorem stating that the car needs 5.75 gallons for the given trip --/
theorem road_trip_fuel_consumption
  (efficiency : FuelEfficiency)
  (segments : List TripSegment)
  (h1 : efficiency.highway = 40)
  (h2 : efficiency.city = 30)
  (h3 : segments = [
    { distance := 65, isHighway := true },
    { distance := 90, isHighway := false },
    { distance := 45, isHighway := true }
  ]) :
  totalGallonsNeeded efficiency segments = 23/4 := by
  sorry

#eval totalGallonsNeeded
  { highway := 40, city := 30 }
  [
    { distance := 65, isHighway := true },
    { distance := 90, isHighway := false },
    { distance := 45, isHighway := true }
  ]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_fuel_consumption_l807_80736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l807_80798

/-- Calculates the time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem: A train with length 180 meters traveling at 72 km/h takes 9 seconds to cross a pole -/
theorem train_crossing_pole_time :
  train_crossing_time 180 72 = 9 := by
  sorry

-- Use #eval only for nat, Int, or String types
#check train_crossing_time 180 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l807_80798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_300_and_700_l807_80776

theorem count_even_numbers_between_300_and_700 : 
  (Finset.filter (fun n : ℕ => n % 2 = 0 ∧ 300 < n ∧ n < 700) (Finset.range 700)).card = 199 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_300_and_700_l807_80776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyper_volume_derivative_l807_80791

-- Define the measures for different dimensions
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3
noncomputable def hyper_surface (r : ℝ) : ℝ := 8 * Real.pi * r^3
noncomputable def hyper_volume (r : ℝ) : ℝ := 2 * Real.pi * r^4

-- State the theorem
theorem hyper_volume_derivative (r : ℝ) : 
  (deriv hyper_volume r) = hyper_surface r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyper_volume_derivative_l807_80791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_estate_area_l807_80777

/-- Represents the scale of the map in miles per inch -/
noncomputable def scale : ℝ := 300

/-- Represents the length of one side of the hexagon on the map in inches -/
noncomputable def mapSideLength : ℝ := 6

/-- Calculates the area of a regular hexagon given its side length -/
noncomputable def hexagonArea (sideLength : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * sideLength ^ 2

/-- Converts the map length to actual length in miles -/
noncomputable def actualSideLength : ℝ := mapSideLength * scale

theorem hexagonal_estate_area : 
  hexagonArea actualSideLength = 4860000 * Real.sqrt 3 := by
  -- Expand the definition of hexagonArea
  unfold hexagonArea
  -- Expand the definition of actualSideLength
  unfold actualSideLength
  -- Perform algebraic simplifications
  simp [scale, mapSideLength]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_estate_area_l807_80777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_cover_area_ratio_small_semicircles_count_l807_80700

/-- The number of small semicircles along the diameter of a large semicircle -/
def N : ℕ := 9

/-- The radius of each small semicircle -/
noncomputable def r : ℝ := 1

/-- The radius of the large semicircle -/
noncomputable def R : ℝ := (N + 1) * r

/-- The combined area of all small semicircles -/
noncomputable def A : ℝ := (N + 1) * Real.pi * r^2 / 2

/-- The area of the region inside the large semicircle but outside all the small semicircles -/
noncomputable def B : ℝ := Real.pi * R^2 / 2 - A

/-- The statement that the diameters of the small semicircles cover the diameter of the large semicircle -/
theorem diameter_cover : R = (N + 1) * r := by rfl

/-- The ratio of areas A to B is 1:9 -/
theorem area_ratio : A / B = 1 / 9 := by sorry

theorem small_semicircles_count : N = 9 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_cover_area_ratio_small_semicircles_count_l807_80700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_professors_asleep_l807_80708

/-- Represents a moment during the lecture -/
def Moment : Type := ℕ

/-- Represents a professor -/
def Professor : Type := Fin 5

/-- The total number of professors -/
def num_professors : ℕ := 5

/-- A function that returns the moments when a professor falls asleep -/
def sleep_moments (p : Professor) : Finset Moment := sorry

/-- A predicate that checks if a professor is asleep at a given moment -/
def is_asleep (p : Professor) (m : Moment) : Prop :=
  m ∈ sleep_moments p

theorem three_professors_asleep :
  (∀ p : Professor, Finset.card (sleep_moments p) = 2) →
  (∀ p1 p2 : Professor, p1 ≠ p2 →
    ∃ m : Moment, is_asleep p1 m ∧ is_asleep p2 m) →
  ∃ m : Moment, ∃ p1 p2 p3 : Professor,
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    is_asleep p1 m ∧ is_asleep p2 m ∧ is_asleep p3 m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_professors_asleep_l807_80708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_profit_percentage_l807_80717

/-- Calculate the profit percentage for a refrigerator sale given the following conditions:
    - Purchase price after discount: 16500
    - Discount percentage: 20%
    - Transport cost: 125
    - Installation cost: 250
    - Selling price: 23100
-/
theorem refrigerator_profit_percentage 
  (purchase_price : ℝ)
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 16500)
  (h2 : discount_percentage = 20)
  (h3 : transport_cost = 125)
  (h4 : installation_cost = 250)
  (h5 : selling_price = 23100)
  : ∃ (profit_percentage : ℝ), 
    (abs (profit_percentage - 36.89) < 0.01) ∧
    (profit_percentage = 
      ((selling_price - (purchase_price + transport_cost + installation_cost)) / 
       (purchase_price + transport_cost + installation_cost)) * 100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_profit_percentage_l807_80717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_l807_80739

/-- The ellipse C2 -/
def C2 (x y : ℝ) : Prop := x^2/6 + y^2/5 = 1

/-- The parabola C1 -/
def C1 (x y : ℝ) : Prop := y^2 = 4*x

/-- The line l passing through M(4,0) -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 4)

/-- The area of triangle ABO -/
noncomputable def area_ABO (k : ℝ) : ℝ := 2 * Real.sqrt ((16/k^2) + 64)

theorem min_area_ABO :
  ∀ k : ℝ, k ≠ 0 → area_ABO k ≥ 16 ∧
  ∃ k₀ : ℝ, area_ABO k₀ = 16 := by
  sorry

#check min_area_ABO

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_l807_80739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_value_satisfies_conditions_l807_80721

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def y (x u : ℝ) : ℝ := 4 * ((x + floor u) / 4 - floor ((x + floor u) / 4))

theorem u_value_satisfies_conditions :
  ∃ u : ℝ → ℝ,
    (∀ x : ℝ, x ∈ ({1, 8, 11, 14} : Set ℝ) → y x (u x) = 1) ∧
    (∀ x : ℝ, x ∈ ({2, 5, 12, 15} : Set ℝ) → y x (u x) = 2) ∧
    (∀ x : ℝ, x ∈ ({3, 6, 9, 16} : Set ℝ) → y x (u x) = 3) ∧
    (∀ x : ℝ, x ∈ ({4, 7, 10, 13} : Set ℝ) → y x (u x) = 0) ∧
    (∀ x : ℝ, u x = (x - 1) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_value_satisfies_conditions_l807_80721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_negative_l807_80718

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) else 1

-- State the theorem
theorem f_inequality_iff_negative (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_negative_l807_80718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_five_presentable_set_l807_80782

/-- A complex number is five-presentable if it can be expressed as w^2 - 1/w^2 for some complex number w with absolute value 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w^2 - 1 / w^2

/-- The set of all five-presentable complex numbers -/
def U : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def EllipseArea (a b : ℝ) : ℝ :=
  Real.pi * a * b

theorem area_of_five_presentable_set :
  EllipseArea (5 * 24 / 25) (5 * 2) = 48 * Real.pi := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_five_presentable_set_l807_80782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l807_80799

theorem trigonometric_identity (θ : ℝ) (h : θ ≠ 0) (h' : θ ≠ π/2) :
  (Real.sin θ + 1 / Real.sin θ)^2 + (Real.cos θ + 1 / Real.cos θ)^2 = 5 + 2 * (Real.tan θ^2 + (1 / Real.tan θ)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l807_80799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_is_two_years_l807_80761

/-- Calculates the number of years for an investment given the initial amount,
    annual interest rate, compounding frequency, and total compound interest. -/
noncomputable def investment_years (principal : ℝ) (rate : ℝ) (frequency : ℝ) (compound_interest : ℝ) : ℝ :=
  let future_value := principal + compound_interest
  let periodic_rate := rate / frequency
  (Real.log (future_value / principal)) / (frequency * Real.log (1 + periodic_rate))

/-- Theorem stating that for the given investment parameters, the investment period is 2 years. -/
theorem investment_period_is_two_years :
  let principal := 20000
  let rate := 0.04
  let frequency := 2
  let compound_interest := 1648.64
  ∃ ε > 0, |investment_years principal rate frequency compound_interest - 2| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_is_two_years_l807_80761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x4_coefficient_l807_80766

/-- The coefficient of x^4 in the given polynomial expression is -3 -/
theorem x4_coefficient : 
  let p : Polynomial ℝ := 4 * (X^2 - 2*X^4 + X^3) + 2 * (X^4 + 3*X^3 - 2*X^2 + X^5) - 3 * (2 + X^2 - X^4 + 2*X^3)
  p.coeff 4 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x4_coefficient_l807_80766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l807_80758

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the right branch of the hyperbola -/
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.x > 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem hyperbola_triangle_perimeter
  (h : Hyperbola)
  (F P Q A : Point)
  (h_equation : h.a^2 = 9 ∧ h.b^2 = 16)
  (h_F : F = ⟨-5, 0⟩)
  (h_P : isOnRightBranch h P)
  (h_Q : isOnRightBranch h Q)
  (h_PQ : distance P Q = 2 * h.b)
  (h_A : A = ⟨5, 0⟩)
  (h_A_on_PQ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = ⟨t * P.x + (1 - t) * Q.x, t * P.y + (1 - t) * Q.y⟩)
  : distance F P + distance F Q + distance P Q = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l807_80758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oleum_mixing_theorem_l807_80737

/-- Represents the composition of an oleum mixture -/
structure OleumComposition where
  h2so4 : ℚ  -- Percentage of H₂SO₄
  so3 : ℚ    -- Percentage of SO₃
  h2s2o7 : ℚ -- Percentage of H₂S₂O₇

/-- Represents a mixture of oleum with its mass -/
structure OleumMixture where
  composition : OleumComposition
  mass : ℚ

/-- Calculates the final composition of oleum when mixing multiple mixtures -/
noncomputable def mixOleum (mixtures : List OleumMixture) : OleumComposition :=
  let totalMass := mixtures.foldl (fun acc m => acc + m.mass) 0
  let totalH2SO4 := mixtures.foldl (fun acc m => acc + m.mass * m.composition.h2so4 / 100) 0
  let totalSO3 := mixtures.foldl (fun acc m => acc + m.mass * m.composition.so3 / 100) 0
  let totalH2S2O7 := mixtures.foldl (fun acc m => acc + m.mass * m.composition.h2s2o7 / 100) 0
  { h2so4 := totalH2SO4 / totalMass * 100,
    so3 := totalSO3 / totalMass * 100,
    h2s2o7 := totalH2S2O7 / totalMass * 100 }

theorem oleum_mixing_theorem (mixtures : List OleumMixture) :
  let result := mixOleum mixtures
  (result.h2so4 + result.so3 + result.h2s2o7 = 100) ∧
  (55 - 1 < result.h2so4 ∧ result.h2so4 < 55 + 1) ∧
  (result.so3 = 10) ∧
  (35 - 1 < result.h2s2o7 ∧ result.h2s2o7 < 35 + 1) :=
by sorry

#check oleum_mixing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oleum_mixing_theorem_l807_80737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_smallest_period_f_origin_symmetric_l807_80740

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x, f (-x) = -f x) :=
by
  constructor
  · intro x
    simp [f]
    -- Proof for periodicity
    sorry
  · intro x
    simp [f]
    -- Proof for odd function property
    sorry

-- Additional theorem to show that f has the smallest positive period of π
theorem f_smallest_period :
  ∀ y, y > 0 ∧ (∀ x, f (x + y) = f x) → y ≥ Real.pi :=
by
  -- Proof that π is the smallest positive period
  sorry

-- Theorem to show that the graph of f is symmetric with respect to the origin
theorem f_origin_symmetric :
  ∀ x, f (-x) = -f x :=
by
  -- This is already proved in f_properties, but we can state it separately
  exact (f_properties.2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_smallest_period_f_origin_symmetric_l807_80740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l807_80707

def is_valid_pair (a b : ℂ) : Prop :=
  a^4 * b^6 = 1 ∧ a^8 * b^3 = 1

-- Define the set of valid pairs
def valid_pairs : Set (ℂ × ℂ) :=
  {p | is_valid_pair p.1 p.2}

-- Prove that the set of valid pairs is finite
noncomputable instance : Fintype valid_pairs := sorry

theorem count_valid_pairs :
  Fintype.card valid_pairs = 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l807_80707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l807_80783

-- Define complex number equality
def complex_eq (a b : ℂ) : Prop := a.re = b.re ∧ a.im = b.im

theorem complex_equation_solution (z : ℂ) :
  complex_eq (z * (Complex.I + 1)) (3 - 4 * Complex.I) →
  complex_eq z (-((7 * Complex.I + 1) / 2)) :=
by
  intro h
  sorry  -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l807_80783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_salary_l807_80779

/-- Jill's net monthly salary in dollars -/
noncomputable def net_salary : ℚ := 3400

/-- Jill's discretionary income as a fraction of her net salary -/
def discretionary_fraction : ℚ := 1/5

/-- Percentage of discretionary income allocated to vacation fund -/
def vacation_fund_percent : ℚ := 30

/-- Percentage of discretionary income allocated to savings -/
def savings_percent : ℚ := 20

/-- Percentage of discretionary income allocated to eating out and socializing -/
def social_percent : ℚ := 35

/-- Amount left for gifts and charitable causes in dollars -/
def charitable_amount : ℚ := 102

theorem jills_salary :
  net_salary * discretionary_fraction * 
  (100 - (vacation_fund_percent + savings_percent + social_percent)) / 100 = 
  charitable_amount := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_salary_l807_80779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_got_one_five_l807_80796

noncomputable def cash_reward : ℚ := 15
noncomputable def reward_multiplier : ℚ := 5
def num_twos : ℕ := 3
def num_threes : ℕ := 4
def num_fours : ℕ := 1

noncomputable def average_grade (num_fives : ℕ) : ℚ :=
  (2 * num_twos + 3 * num_threes + 4 * num_fours + 5 * num_fives : ℚ) / 
  (num_twos + num_threes + num_fours + num_fives : ℚ)

theorem milo_got_one_five : 
  ∃ (num_fives : ℕ), 
    reward_multiplier * average_grade num_fives = cash_reward ∧ 
    num_fives = 1 := by
  use 1
  constructor
  · norm_num
    rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_got_one_five_l807_80796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_amount_reservoir_amount_exact_l807_80748

-- Define the reservoir capacity in million gallons
variable (C : ℝ)

-- Define the normal level as C - 5 million gallons
def normal_level (C : ℝ) : ℝ := C - 5

-- Define the current amount as twice the normal level
def current_amount (C : ℝ) : ℝ := 2 * normal_level C

-- State the theorem
theorem reservoir_amount (h1 : current_amount C = 0.6 * C) :
  ∃ ε > 0, |current_amount C - 4.284| < ε := by
  sorry

-- Additional helper theorem to show the exact value
theorem reservoir_amount_exact (h1 : current_amount C = 0.6 * C) :
  C = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_amount_reservoir_amount_exact_l807_80748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_ln_curve_l807_80793

/-- The length of the arc of y = ln x from x = √3 to x = √15 -/
noncomputable def arc_length : ℝ :=
  (1/2) * Real.log (9/5) + 2

/-- The curve function y = ln x -/
noncomputable def curve (x : ℝ) : ℝ := Real.log x

theorem arc_length_of_ln_curve :
  ∫ x in Set.Icc (Real.sqrt 3) (Real.sqrt 15), 
    Real.sqrt (1 + (deriv curve x) ^ 2) = arc_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_ln_curve_l807_80793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_zero_one_f_has_extreme_values_iff_a_in_zero_half_l807_80764

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / (x - 1) + a) * Real.log x

-- Theorem 1: f(x) > 0 for all x > 0 iff a ∈ [0, 1]
theorem f_positive_iff_a_in_zero_one (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 2: f(x) has extreme values on (1, +∞) iff a ∈ (0, 1/2)
theorem f_has_extreme_values_iff_a_in_zero_half (a : ℝ) :
  (∃ x > 1, ∀ y > 1, f a x ≥ f a y ∨ f a x ≤ f a y) ↔ 0 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_zero_one_f_has_extreme_values_iff_a_in_zero_half_l807_80764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x0_value_l807_80795

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem min_x0_value (ω : ℝ) (x₀ : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + Real.pi / (2 * ω)) = f ω x) →  -- Distance between adjacent axes of symmetry is π/2
  (∀ x : ℝ, f ω (2 * x₀ - x) = -f ω x) →  -- Graph is centrally symmetric about (x₀, 0)
  x₀ > 0 →
  (∀ x₀' : ℝ, x₀' > 0 → (∀ x : ℝ, f ω (2 * x₀' - x) = -f ω x) → x₀ ≤ x₀') →
  x₀ = 5 * Real.pi / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x0_value_l807_80795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_one_inequality_implies_a_gt_13_div_4_l807_80746

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * 2^x - 1) / (2^x + 1)

-- Theorem 1: If f is an odd function, then m = 1
theorem odd_function_implies_m_eq_one (m : ℝ) :
  (∀ x : ℝ, f m (-x) = -(f m x)) → m = 1 :=
by sorry

-- Theorem 2: If f(x-3) + f(a+x^2) > 0 for all real x, then a > 13/4
theorem inequality_implies_a_gt_13_div_4 (a : ℝ) :
  (∀ x : ℝ, f 1 (x - 3) + f 1 (a + x^2) > 0) → a > 13/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_one_inequality_implies_a_gt_13_div_4_l807_80746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l807_80768

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_p_pos : 0 < p
  h_p_le_8 : p ≤ 8
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h_focus : focus = (p/2, 0)

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

/-- Auxiliary definitions -/
def TangentLine (l : ℝ → ℝ → Prop) (C : Circle) (P : ℝ × ℝ) : Prop := sorry
def PassesThrough (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop := sorry
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ := sorry
def OnLine (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop := sorry

/-- Theorem statement -/
theorem parabola_circle_tangent
  (Ω : Parabola) (C : Circle)
  (h_C : C.center = (3, 0) ∧ C.radius = 1)
  (h_tangent : ∃ T : ℝ × ℝ, (∃ l : ℝ → ℝ → Prop, TangentLine l C T ∧ PassesThrough l Ω.focus) ∧ 
               Distance Ω.focus T = Real.sqrt 3) :
  (Ω.eq = λ x y => y^2 = 4*x) ∧
  (∃ min_val : ℝ, min_val = 9 ∧
    ∀ A B : ℝ × ℝ, ∀ l : ℝ → ℝ → Prop, 
      TangentLine l C A → Ω.eq A.1 A.2 → Ω.eq B.1 B.2 → 
      OnLine l A → OnLine l B → 
      Distance Ω.focus A * Distance Ω.focus B ≥ min_val) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l807_80768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_after_two_years_l807_80780

def village_population (initial_population : ℕ) 
  (decrease_rate_1 decrease_rate_2 : ℚ)
  (migration_out_1 migration_in_1 migration_out_2 migration_in_2 : ℕ)
  (birth_rate death_rate : ℚ) : ℕ :=
  let population_1 := (initial_population:ℚ) * (1 - decrease_rate_1) - 
    (migration_out_1 - migration_in_1:ℚ) + 
    (initial_population:ℚ) * (birth_rate - death_rate)
  let population_2 := population_1 * (1 - decrease_rate_2) - 
    (migration_out_2 - migration_in_2:ℚ) + 
    population_1 * (birth_rate - death_rate)
  population_2.floor.toNat

theorem village_population_after_two_years :
  village_population 6000 (7/100) (12/100) 200 100 150 80 (5/200) (3/200) = 4914 := by
  sorry

#eval village_population 6000 (7/100) (12/100) 200 100 150 80 (5/200) (3/200)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_after_two_years_l807_80780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_thirteenths_negative_l807_80710

/-- A function satisfying the given properties -/
def special_function (f : ℚ → ℚ) : Prop :=
  (∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b) ∧
  (∀ n : ℤ, f n = n)

/-- Theorem stating that f(8/13) < 0 for any function satisfying the given properties -/
theorem f_eight_thirteenths_negative (f : ℚ → ℚ) (h : special_function f) :
  f (8 / 13) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_thirteenths_negative_l807_80710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expelled_natives_l807_80751

theorem max_expelled_natives (initial_people : Nat) (total_coins : Nat) 
  (h1 : initial_people = 30) (h2 : total_coins = 270) : 
  (∃ (expelled : Nat), 
    expelled ≤ initial_people ∧ 
    (∀ (remaining : Nat), remaining = initial_people - expelled → 
      (Finset.sum (Finset.range remaining) (fun i => i + 1)) ≤ total_coins) ∧
    (∀ (more_expelled : Nat), more_expelled > expelled → 
      (Finset.sum (Finset.range (initial_people - more_expelled)) (fun i => i + 1)) > total_coins)) ∧
  (∀ (x : Nat), x > 6 → 
    (Finset.sum (Finset.range (initial_people - x)) (fun i => i + 1)) > total_coins) := by
  sorry

#check max_expelled_natives

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expelled_natives_l807_80751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_identification_l807_80755

/-- A quadratic function f(x) = ax^2 + bx + c where a, b, c are integers -/
def quadratic_function (a b c : ℤ) : ℤ → ℤ := λ x ↦ a * x^2 + b * x + c

theorem incorrect_value_identification 
  (a b c : ℤ) 
  (h1 : quadratic_function a b c 7 = -1)
  (h2 : quadratic_function a b c 4 = -4)
  (h3 : quadratic_function a b c 2 = 4) :
  quadratic_function a b c 1 ≠ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_identification_l807_80755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_sum_positive_l807_80716

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

noncomputable def F (a b : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then f a b x else -f a b x

theorem F_sum_positive 
  (a b m n : ℝ) 
  (h1 : f a b (-1) = 0)
  (h2 : ∀ x, f a b x ≥ 0)
  (h3 : m * n < 0)
  (h4 : m + n > 0)
  (h5 : a > 0)
  (h6 : ∀ x, f a b x = f a b (-x)) :
  F a b m + F a b n > 0 := by
  sorry

#check F_sum_positive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_sum_positive_l807_80716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_intercepts_chord_l807_80705

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 11 = 0

-- Define the point through which the line passes
def point : ℝ × ℝ := (-1, 1)

-- Define the length of the intercepted chord
noncomputable def chord_length : ℝ := 4 * Real.sqrt 3

-- Define the vertical line
def vertical_line (x : ℝ) : Prop := x = -1

-- Define the non-vertical line
def non_vertical_line (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

theorem line_through_point_intercepts_chord :
  ∃ (x y : ℝ), (vertical_line x ∧ y = point.2) ∨ 
               (non_vertical_line x y ∧ y - point.2 = (x - point.1) * (4/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_intercepts_chord_l807_80705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l807_80792

/-- The amount of water initially in the tank -/
def initial_water : ℝ := 6000

/-- The amount of water that evaporated -/
def evaporated_water : ℝ := 2000

/-- The amount of water drained by Bob -/
def drained_water : ℝ := 3500

/-- The duration of rainfall in minutes -/
def rainfall_duration : ℝ := 30

/-- The amount of water added by rain every 10 minutes -/
def rain_rate : ℝ := 350

/-- The final amount of water in the tank -/
def final_water : ℝ := 1550

theorem initial_water_amount :
  initial_water = 
    final_water + evaporated_water + drained_water - 
    (rainfall_duration / 10 * rain_rate) := by
  -- The proof is omitted for now
  sorry

#eval initial_water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l807_80792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_wire_volume_conservation_l807_80754

/-- The radius of a sphere that, when melted and drawn into a wire,
    produces a wire with given dimensions -/
noncomputable def sphere_radius (wire_radius : ℝ) (wire_length : ℝ) : ℝ :=
  (3 * wire_radius^2 * wire_length / 4)^(1/3)

theorem sphere_wire_volume_conservation
  (wire_radius : ℝ) (wire_length : ℝ) (sphere_rad : ℝ)
  (h_wire_radius : wire_radius = 8)
  (h_wire_length : wire_length = 36)
  (h_sphere_rad : sphere_rad = sphere_radius wire_radius wire_length) :
  sphere_rad = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_wire_volume_conservation_l807_80754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l807_80734

theorem triangle_side_values (A B C : Real) (a b c : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (a > 0) → (b > 0) → (c > 0) →
  -- Given conditions
  (Real.cos A = 1/4) →
  (a = 4) →
  (b + c = 6) →
  (b < c) →
  -- Law of Cosines (included as a condition since it's a fundamental property)
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →
  -- Conclusion
  (b = 2 ∧ c = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l807_80734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_flips_formula_l807_80774

/-- A deck of cards with three special cards (A's) -/
structure Deck where
  N : ℕ  -- Total number of cards
  h : N ≥ 3  -- Ensure there are at least 3 cards for the three A's

/-- The position of the second A in a shuffled deck -/
def second_A_position (d : Deck) : ℕ → ℕ := sorry

/-- The probability distribution of shuffling the deck -/
def shuffle_distribution (d : Deck) : Finset ℕ → ℝ := sorry

/-- The expected number of cards flipped until the second A appears -/
noncomputable def expected_flips (d : Deck) : ℝ :=
  Finset.sum (Finset.range d.N) (λ i => i * shuffle_distribution d {i})

/-- The main theorem: expected number of flips is (N+1)/2 -/
theorem expected_flips_formula (d : Deck) :
  expected_flips d = (d.N + 1) / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_flips_formula_l807_80774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_range_condition_l807_80729

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * log x

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement for the minimum value of f
theorem f_min_value : ∃ (c : ℝ), c = -1/exp 1 ∧ ∀ x > 0, f x ≥ c := by
  sorry

-- Statement for the range of a
theorem g_range_condition (a : ℝ) : 
  (∀ x > 0, 2 * f x ≥ g a x) ↔ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_range_condition_l807_80729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_relation_l807_80749

theorem trigonometric_relation (α β : Real) : 
  α ∈ Set.Ioo 0 (π/2) → 
  β ∈ Set.Ioo 0 (π/2) → 
  Real.tan α = (1 + Real.sin β) / Real.cos β → 
  2 * α - β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_relation_l807_80749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_zero_in_interval_l807_80762

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x - (1 / x)

-- Theorem statement
theorem g_has_unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo 1 2 ∧ g x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_zero_in_interval_l807_80762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_specific_line_l807_80769

/-- The angle of inclination of a line in a Cartesian coordinate system -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- The line equation in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem angle_of_inclination_specific_line :
  angle_of_inclination 3 (Real.sqrt 3) (-3) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_specific_line_l807_80769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nets_win_in_seven_l807_80787

/-- The probability of a team winning a single game -/
def p : ℚ := 1/4

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The total number of games in the series -/
def total_games : ℕ := 7

/-- The probability of the Nets winning the NBA finals in exactly 7 games -/
theorem nets_win_in_seven (p : ℚ) (games_to_win : ℕ) (total_games : ℕ) :
  p = 1/4 ∧ games_to_win = 4 ∧ total_games = 7 →
  (Nat.choose (total_games - 1) (games_to_win - 1) : ℚ) *
  (p ^ (games_to_win - 1)) * ((1 - p) ^ (games_to_win - 1)) * (1 - p) = 405/4096 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nets_win_in_seven_l807_80787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_fractions_l807_80773

theorem min_product_of_fractions (x y : ℕ+) 
  (h : (1 : ℚ) / x.val + (1 : ℚ) / (3 * y.val) = (1 : ℚ) / 6) :
  x.val * y.val ≥ 192 ∧ ∃ (a b : ℕ+), a.val * b.val = 192 ∧ 
    (1 : ℚ) / a.val + (1 : ℚ) / (3 * b.val) = (1 : ℚ) / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_fractions_l807_80773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l807_80744

/-- The area of an octagon inscribed in a circle with radius 3 units -/
noncomputable def inscribed_octagon_area : ℝ := 18 * Real.sqrt 2

/-- The radius of the circle in which the octagon is inscribed -/
def circle_radius : ℝ := 3

theorem octagon_area_in_circle :
  inscribed_octagon_area = 8 * (1/2 * circle_radius * circle_radius * Real.sin (π/4)) :=
by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l807_80744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l807_80706

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (2 * x)

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-2 : ℝ) (9/8 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l807_80706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_weight_calculation_l807_80730

/-- Proves that the total weight of 800 envelopes, each weighing 8.5 grams, is equal to 6.8 kilograms. -/
theorem envelope_weight_calculation :
  let envelope_weight : ℝ := 8.5  -- weight of one envelope in grams
  let num_envelopes : ℕ := 800    -- number of envelopes
  let grams_per_kg : ℝ := 1000    -- conversion factor from grams to kilograms
  (envelope_weight * (num_envelopes : ℝ)) / grams_per_kg = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_weight_calculation_l807_80730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_pi_half_g_max_value_l807_80743

noncomputable section

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_property (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * Real.cos y
axiom f_zero : f 0 = 1
axiom f_pi_half : f (Real.pi / 2) = 2

-- Define the function g
def g (x : ℝ) : ℝ := (4 * f x - 2 * (3 - Real.sqrt 3) * Real.sin x) / (Real.sin x + Real.sqrt (1 - Real.sin x))

-- Define the domain of x for g
def g_domain (x : ℝ) : Prop := x ∈ Set.Icc 0 (Real.pi / 3) ∪ Set.Icc (5 * Real.pi / 6) Real.pi

-- Theorem 1: Value of f(-π/2)
theorem f_neg_pi_half : f (-Real.pi / 2) = -2 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem g_max_value : 
  ∃ (x : ℝ), g_domain x ∧ ∀ (y : ℝ), g_domain y → g y ≤ g x ∧ g x = 2 * (Real.sqrt 3 + 1) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_pi_half_g_max_value_l807_80743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l807_80745

-- Define the fraction as noncomputable
noncomputable def f (x : ℝ) : ℝ := 5 / (x - 3)

-- Theorem stating the condition for the fraction to be meaningful
theorem fraction_meaningful (x : ℝ) : 
  (∃ y, f x = y) ↔ x ≠ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l807_80745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l807_80790

/-- 
Given an angle α in the fourth quadrant and a point P(√5, x) on its terminal side,
prove that if sin α = (√2/4)x, then cos α = √10/4.
-/
theorem cos_alpha_value (α : ℝ) (x : ℝ) 
  (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- α is in the fourth quadrant
  (h2 : x > 0) -- x is positive in the fourth quadrant
  (h3 : x^2 + 5 = (x / Real.sin α)^2) -- Point P(√5, x) lies on the terminal side
  (h4 : Real.sin α = (Real.sqrt 2 / 4) * x) -- Given condition
  : Real.cos α = Real.sqrt 10 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l807_80790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_l807_80735

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions here
  A = 60 * Real.pi / 180 ∧  -- Convert 60° to radians
  b = 16 ∧
  1/2 * b * c * Real.sin A = 220 * Real.sqrt 3

-- State the theorem
theorem side_c_value :
  ∀ A B C a b c : ℝ,
  triangle_ABC A B C a b c →
  c = 55 :=
by
  -- Introduce variables and hypothesis
  intros A B C a b c h
  -- Unfold the definition of triangle_ABC
  unfold triangle_ABC at h
  -- Extract the conditions from the hypothesis
  have h1 : A = 60 * Real.pi / 180 := h.left
  have h2 : b = 16 := h.right.left
  have h3 : 1/2 * b * c * Real.sin A = 220 * Real.sqrt 3 := h.right.right
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_l807_80735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iterate_s_six_times_30_l807_80757

-- Define the function s as noncomputable
noncomputable def s (θ : ℝ) : ℝ := 1 / (2 - θ)

-- State the theorem
theorem iterate_s_six_times_30 : s (s (s (s (s (s 30))))) = 146 / 175 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iterate_s_six_times_30_l807_80757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l807_80727

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (-2 + Real.cos θ, Real.sin θ)

-- Define the range of θ
def θ_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 2 * Real.pi

theorem curve_properties :
  ∀ θ, θ_range θ →
    -- 1. Standard equation
    let (x, y) := C θ
    (x + 2)^2 + y^2 = 1 ∧
    -- 2. Range of y/x
    -(Real.sqrt 3 / 3) ≤ y / x ∧ y / x ≤ Real.sqrt 3 / 3 ∧
    -- 3. Range of 2x + y
    -4 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ -4 + Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l807_80727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noncrossing_chords_eq_catalan_catalan_is_nat_l807_80732

/-- The number of ways to pair 2n points on a circle with n non-intersecting chords -/
def noncrossingChords (n : ℕ) : ℚ :=
  (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- The n-th Catalan number -/
def catalanNumber (n : ℕ) : ℚ :=
  (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- Theorem: The number of ways to pair 2n points on a circle with n non-intersecting chords
    is equal to the n-th Catalan number -/
theorem noncrossing_chords_eq_catalan (n : ℕ) :
  noncrossingChords n = catalanNumber n := by
  -- Proof
  rfl

/-- The n-th Catalan number is always a natural number -/
theorem catalan_is_nat (n : ℕ) :
  ∃ m : ℕ, catalanNumber n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_noncrossing_chords_eq_catalan_catalan_is_nat_l807_80732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_l807_80781

noncomputable def f (x : ℝ) : ℝ := (x^3 + 10*x^2 + 31*x + 30) / (x + 3)

theorem function_simplification :
  ∃ (A B C D : ℝ),
    (∀ x, x ≠ D → f x = A*x^2 + B*x + C) ∧
    (∀ x, x = D ↔ ¬ ∃ y, f x = y) ∧
    A + B + C + D = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_l807_80781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_48_l807_80726

theorem product_not_equal_48 : ∃! (pair : ℚ × ℚ), 
  (pair ∈ [((-6 : ℚ), -8), (-4, -12), (3/4, -64), (3, 16), (4/3, 36)]) ∧ 
  (pair.1 * pair.2 ≠ 48) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_48_l807_80726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_proof_l807_80767

theorem triangle_sides_proof (x y z : ℝ) (h1 : x + y = 12) 
  (h2 : (1/2) * x * y * Real.sin (2 * π / 3) = 35/2 * Real.sin (2 * π / 3))
  (h3 : x^2 = y^2 + z^2 - 2*y*z*Real.cos (2 * π / 3))
  (h4 : x ≥ y ∧ y ≥ z ∧ z > 0) : 
  x = 7 ∧ y = 5 ∧ z = 3 := by
  sorry

#check triangle_sides_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_proof_l807_80767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l807_80712

/-- A line passing through point M(2, 1) intersects the X-axis and Y-axis at points P and Q respectively, where |MP| = |MQ|. -/
def line_through_M (P Q : ℝ × ℝ) : Prop :=
  P.2 = 0 ∧ Q.1 = 0 ∧ (P.1 - 2)^2 + 1^2 = 2^2 + (Q.2 - 1)^2

/-- The equation of the line is x + 2y - 4 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

/-- Theorem stating that the line passing through M(2, 1) with |MP| = |MQ| has the equation x + 2y - 4 = 0 -/
theorem line_equation_proof :
  ∃ P Q : ℝ × ℝ, line_through_M P Q → ∀ x y : ℝ, ((x, y) ∈ Set.Icc P Q) → line_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l807_80712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_with_r_l807_80711

/-- The total amount of money among four individuals -/
noncomputable def total_amount : ℝ := 12000

/-- The fraction of money that r has compared to the total of p, q, and s -/
noncomputable def r_fraction : ℝ := 3/4

/-- Theorem stating the amount of money r has -/
theorem amount_with_r (p q r s : ℝ) 
  (h1 : p + q + r + s = total_amount)
  (h2 : r = r_fraction * (p + q + s)) :
  r = total_amount / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_with_r_l807_80711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_q_value_l807_80731

/-- A geometric progression with positive terms -/
structure GeometricProgression where
  a : ℕ → ℝ
  q : ℝ
  first_term_positive : 0 < a 1
  common_ratio_positive : 0 < q
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first n terms of a geometric progression -/
noncomputable def sum_n (gp : GeometricProgression) (n : ℕ) : ℝ :=
  (gp.a 1) * (1 - gp.q^n) / (1 - gp.q)

theorem geometric_progression_q_value
  (gp : GeometricProgression)
  (first_term : gp.a 1 = 3)
  (sum_three : sum_n gp 3 = 21) :
  gp.q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_q_value_l807_80731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l807_80703

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 14 cm between them, is equal to 266 cm². -/
theorem trapezium_area_example : trapeziumArea 20 18 14 = 266 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_right_comm]
  -- Check that the result is equal to 266
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l807_80703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l807_80765

theorem regular_octagon_interior_angle : ∃ (angle : ℝ), angle = 135 := by
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_angles : ℝ := (n - 2) * 180
  let each_angle : ℝ := sum_of_angles / n
  
  have h : each_angle = 135 := by
    -- Proof steps would go here
    sorry

  exact ⟨each_angle, h⟩

#eval (8 - 2) * 180 / 8  -- This should evaluate to 135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l807_80765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_11_and_31_divisible_by_5_l807_80771

def is_divisible_by_5 (n : ℕ) : Bool := n % 5 = 0

def numbers_between_11_and_31_divisible_by_5 : List ℕ :=
  (List.range 21).map (· + 11) |>.filter is_divisible_by_5

theorem average_of_numbers_between_11_and_31_divisible_by_5 :
  let numbers := numbers_between_11_and_31_divisible_by_5
  (numbers.sum : ℚ) / numbers.length = 45/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_11_and_31_divisible_by_5_l807_80771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l807_80794

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem stating the relationship between cylinder heights -/
theorem cylinder_height_relation (c1 c2 : Cylinder) 
  (h_volume : volume c1 = volume c2)
  (h_radius : c2.radius = 1.1 * c1.radius) :
  c1.height = 1.21 * c2.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l807_80794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoints_form_regular_12gon_l807_80738

/-- Predicate stating that four points form a square in the complex plane. -/
def is_square (A B C D : ℂ) : Prop := sorry

/-- Predicate stating that three points form an equilateral triangle in the complex plane. -/
def is_equilateral_triangle (A B C : ℂ) : Prop := sorry

/-- Predicate stating that a list of 12 complex numbers form the vertices of a regular 12-gon. -/
def is_regular_12gon (vertices : List ℂ) : Prop := sorry

/-- Given a square ABCD and equilateral triangles constructed inward on each side,
    the midpoints of certain segments form a regular 12-gon. -/
theorem midpoints_form_regular_12gon 
  (A B C D K L M N : ℂ) -- Points in the complex plane
  (h_square : is_square A B C D) -- ABCD is a square
  (h_equilateral_ABK : is_equilateral_triangle A B K) -- ABK is equilateral
  (h_equilateral_BCL : is_equilateral_triangle B C L) -- BCL is equilateral
  (h_equilateral_CDM : is_equilateral_triangle C D M) -- CDM is equilateral
  (h_equilateral_DAN : is_equilateral_triangle D A N) -- DAN is equilateral
  : is_regular_12gon 
      [(A + K) / 2, (B + K) / 2, (B + L) / 2, (C + L) / 2, 
       (C + M) / 2, (D + M) / 2, (D + N) / 2, (A + N) / 2, 
       (K + L) / 2, (L + M) / 2, (M + N) / 2, (N + K) / 2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoints_form_regular_12gon_l807_80738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_right_angled_triangles_l807_80724

theorem pyramid_right_angled_triangles :
  ∃ (n : ℕ), n ≤ 4 ∧ n = max_right_angled_triangles_in_pyramid :=
by
  -- We assert that there exists a number n that is at most 4
  -- and is equal to the maximum number of right-angled triangles in a pyramid
  use 4
  constructor
  · -- Prove n ≤ 4
    simp
  · -- Define and prove n = max_right_angled_triangles_in_pyramid
    sorry  -- The actual proof would require more detailed geometry

-- Define max_right_angled_triangles_in_pyramid
def max_right_angled_triangles_in_pyramid : ℕ := 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_right_angled_triangles_l807_80724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_at_pi_over_2_l807_80709

noncomputable section

variable (a : ℝ)

def x (t : ℝ) : ℝ := a * t * Real.cos t
def y (t : ℝ) : ℝ := a * t * Real.sin t

def tangent_line (a : ℝ) (x : ℝ) : ℝ := -2 * x / Real.pi + a * Real.pi / 2
def normal_line (a : ℝ) (x : ℝ) : ℝ := Real.pi * x / 2 + a * Real.pi / 2

theorem curve_tangent_and_normal_at_pi_over_2 (a : ℝ) :
  let t₀ : ℝ := Real.pi / 2
  let x₀ : ℝ := x a t₀
  let y₀ : ℝ := y a t₀
  (∀ x, tangent_line a x = -2 * x / Real.pi + a * Real.pi / 2) ∧
  (∀ x, normal_line a x = Real.pi * x / 2 + a * Real.pi / 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_at_pi_over_2_l807_80709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_equals_one_l807_80756

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (1 / x - 1) + Real.log x

/-- Theorem stating that if f(x) ≥ 0 for all x > 0, then a = 1 -/
theorem f_nonnegative_implies_a_equals_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) → a = 1 := by
  sorry

#check f_nonnegative_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_equals_one_l807_80756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l807_80722

noncomputable section

/-- A function f(x) with parameters A and ω -/
def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x - Real.pi / 6) + 1

/-- The theorem stating the properties of the function and the values of A, ω, and α -/
theorem function_properties (A ω : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x, f A ω x ≤ 3) ∧ 
  (∃ x, f A ω x = 3) ∧
  (∀ x, f A ω (x + Real.pi / (2 * ω)) = f A ω x) →
  A = 2 ∧ ω = 2 ∧
  ∀ α, 0 < α ∧ α < Real.pi / 2 → f A ω (α / 2) = 2 → α = Real.pi / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l807_80722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_is_400_miles_l807_80728

/-- Represents the distance between two cities on a map and in reality -/
structure CityDistance where
  map_distance : ℚ
  first_scale_distance : ℚ
  first_scale_miles : ℚ
  second_scale_distance : ℚ
  second_scale_miles : ℚ

/-- Calculates the actual distance between two cities based on map distance and scales -/
def actual_distance (cd : CityDistance) : ℚ :=
  (cd.first_scale_miles / cd.first_scale_distance) * (cd.map_distance / 2) +
  (cd.second_scale_miles / cd.second_scale_distance) * (cd.map_distance / 2)

/-- Theorem stating that the actual distance between the cities is 400 miles -/
theorem city_distance_is_400_miles (cd : CityDistance) 
  (h1 : cd.map_distance = 20)
  (h2 : cd.first_scale_distance = 1/2)
  (h3 : cd.first_scale_miles = 10)
  (h4 : cd.second_scale_distance = 1/4)
  (h5 : cd.second_scale_miles = 5) :
  actual_distance cd = 400 := by
  sorry

#eval actual_distance {
  map_distance := 20,
  first_scale_distance := 1/2,
  first_scale_miles := 10,
  second_scale_distance := 1/4,
  second_scale_miles := 5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_is_400_miles_l807_80728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_intersection_l807_80775

universe u

def U : Finset (Fin 5) := {0, 1, 2, 3, 4}
def M : Finset (Fin 5) := {0, 1, 2}

theorem subset_count_of_intersection (N : Finset (Fin 5)) 
  (h : M ∩ (U \ N) = {1}) : 
  Finset.card (Finset.powerset (M ∩ N)) = 4 := by
  sorry

#check subset_count_of_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_intersection_l807_80775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l807_80788

theorem choir_average_age (female_count : ℕ) (male_count : ℕ) (child_count : ℕ)
  (female_avg_age : ℝ) (male_avg_age : ℝ) (child_avg_age : ℝ)
  (h_female_count : female_count = 12)
  (h_male_count : male_count = 18)
  (h_child_count : child_count = 5)
  (h_female_avg : female_avg_age = 32)
  (h_male_avg : male_avg_age = 38)
  (h_child_avg : child_avg_age = 10) :
  let total_count := female_count + male_count + child_count
  let total_age := female_count * female_avg_age + male_count * male_avg_age + child_count * child_avg_age
  (31.9 < total_age / total_count) ∧ (total_age / total_count < 32.1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l807_80788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_difference_l807_80747

open BigOperators

def S (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, (-1 : ℤ)^(k+1) * (4*(k+1) - 3)

theorem sequence_sum_difference : S 15 - S 22 + S 31 = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_difference_l807_80747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_correct_l807_80733

/-- The width of an alley where a ladder rests against opposite walls. -/
noncomputable def alley_width (l : ℝ) : ℝ :=
  l * (1 + Real.sqrt 3) / 2

/-- Theorem stating that the width of the alley is correct given the ladder's position. -/
theorem alley_width_correct (l : ℝ) (h : l > 0) :
  let w := alley_width l
  let angle1 := 60 * Real.pi / 180  -- 60° in radians
  let angle2 := 30 * Real.pi / 180  -- 30° in radians
  w = l * Real.cos angle1 + l * Real.cos angle2 :=
by
  -- Unfold the definition of alley_width
  unfold alley_width
  -- Simplify the expression
  simp [Real.cos_pi_div_three, Real.cos_pi_div_six]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_correct_l807_80733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_damage_conversion_l807_80714

/-- Converts Canadian dollars to American dollars given an exchange rate -/
noncomputable def cad_to_usd (cad : ℝ) (exchange_rate : ℝ) : ℝ :=
  cad / exchange_rate

/-- Theorem stating the conversion of storm damage from CAD to USD -/
theorem storm_damage_conversion (damage_cad : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_cad = 45000000)
  (h2 : exchange_rate = 1.25) :
  cad_to_usd damage_cad exchange_rate = 36000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_damage_conversion_l807_80714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_pillow_price_l807_80789

/-- Given 4 pillows with an average cost of $5 and a fifth pillow that
    makes the new average of 5 pillows $6, prove that the price of the
    fifth pillow is $10. -/
theorem fifth_pillow_price (num_pillows : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (fifth_price : ℝ) :
  num_pillows = 4 →
  initial_avg = 5 →
  new_avg = 6 →
  (num_pillows * initial_avg + fifth_price) / (num_pillows + 1) = new_avg →
  fifth_price = 10 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check fifth_pillow_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_pillow_price_l807_80789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_monotonically_decreasing_l807_80713

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

-- State the theorem
theorem sufficient_condition_for_monotonically_decreasing :
  ∃ (a b : ℝ), a < 2 ∧ 3 < b ∧
  (∀ x, 2 ≤ x ∧ x ≤ 3 → monotonically_decreasing (λ y ↦ f' (y + 1)) 2 3) ∧
  (monotonically_decreasing (λ y ↦ f' (y + 1)) a b) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_monotonically_decreasing_l807_80713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_solution_is_three_l807_80753

/-- The height function of the projectile -/
def height_func (t : ℝ) : ℝ := -16 * t^2 + 64 * t

/-- The equation representing the projectile at 49 feet -/
def height_equation (t : ℝ) : Prop := height_func t = 49

/-- The second positive solution to the height equation is 3 -/
theorem second_solution_is_three : 
  ∃ t₁ t₂ : ℝ, 0 < t₁ ∧ t₁ < t₂ ∧ t₂ = 3 ∧ 
  height_equation t₁ ∧ height_equation t₂ ∧
  ∀ t, 0 < t ∧ t < t₂ ∧ height_equation t → t = t₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_solution_is_three_l807_80753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_birthday_next_monday_l807_80741

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Returns the weekday of the next year's same date -/
def nextYearWeekday (currentWeekday : Weekday) (currentYear : ℕ) : Weekday :=
  match currentWeekday with
  | Weekday.Monday => if isLeapYear currentYear then Weekday.Wednesday else Weekday.Tuesday
  | Weekday.Tuesday => if isLeapYear currentYear then Weekday.Thursday else Weekday.Wednesday
  | Weekday.Wednesday => if isLeapYear currentYear then Weekday.Friday else Weekday.Thursday
  | Weekday.Thursday => if isLeapYear currentYear then Weekday.Saturday else Weekday.Friday
  | Weekday.Friday => if isLeapYear currentYear then Weekday.Sunday else Weekday.Saturday
  | Weekday.Saturday => if isLeapYear currentYear then Weekday.Monday else Weekday.Sunday
  | Weekday.Sunday => if isLeapYear currentYear then Weekday.Tuesday else Weekday.Monday

/-- Theorem: Liam's birthday (June 4) will next fall on a Monday in 2018 -/
theorem liam_birthday_next_monday :
  ∃ (startDate : Date) (startWeekday : Weekday),
    startDate.year = 2015 ∧
    startDate.month = 6 ∧
    startDate.day = 4 ∧
    startWeekday = Weekday.Thursday →
    ∃ (nextMondayYear : ℕ),
      nextMondayYear > 2015 ∧
      nextMondayYear = 2018 ∧
      (nextYearWeekday (nextYearWeekday (nextYearWeekday startWeekday 2015) 2016) 2017) = Weekday.Monday :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_birthday_next_monday_l807_80741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l807_80760

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x - 3)

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l807_80760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l807_80715

theorem vector_difference_magnitude 
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 2) 
  (hb : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 5) 
  (hab : a.1 * b.1 + a.2 * b.2 = -3) : 
  Real.sqrt (((a.1 - b.1) ^ 2) + ((a.2 - b.2) ^ 2)) = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l807_80715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_three_polygons_l807_80772

/-- Regular polygon with n sides and side length 2 -/
structure RegularPolygon where
  n : ℕ
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The interior angle of a regular polygon -/
noncomputable def interior_angle (p : RegularPolygon) : ℝ :=
  180 * (p.n - 2 : ℝ) / p.n

/-- The perimeter of the shape formed by three polygons meeting at a point -/
def perimeter (p1 p2 p3 : RegularPolygon) : ℝ :=
  2 * (p1.n + p2.n + p3.n : ℝ) - 6

theorem max_perimeter_of_three_polygons :
  ∀ p1 p2 p3 : RegularPolygon,
    p1.n ≥ 3 → p2.n ≥ 3 → p3.n ≥ 3 →
    interior_angle p1 + interior_angle p2 + interior_angle p3 = 360 →
    (p1.n = 4 ∨ p2.n = 4 ∨ p3.n = 4) →
    perimeter p1 p2 p3 ≤ 34 :=
by sorry

#check max_perimeter_of_three_polygons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_three_polygons_l807_80772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_parallel_l807_80702

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := x/2 - y/4 = 1

-- Define what it means for a line to bisect a circle
def bisects (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x₀ y₀ : ℝ), circle x₀ y₀ ∧ line x₀ y₀ ∧
  ∀ (x y : ℝ), circle x y → line x y → (x = x₀ ∧ y = y₀)

-- Define what it means for two lines to be parallel
def parallel (line1 line2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
  ∀ (x y : ℝ), line1 x y ↔ line2 (k*x) (k*y)

-- Theorem statement
theorem line_bisects_and_parallel :
  bisects line_l my_circle ∧ parallel line_l parallel_line :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_parallel_l807_80702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_theorem_l807_80770

/-- The distance Bob walked when he met Yolanda --/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (head_start : ℝ) : ℝ :=
  (bob_speed * (total_distance - yolanda_speed * head_start)) / (yolanda_speed + bob_speed)

/-- Theorem stating the distance Bob walked --/
theorem bob_distance_theorem (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (head_start : ℝ)
    (h1 : total_distance = 80)
    (h2 : yolanda_speed = 8)
    (h3 : bob_speed = 9)
    (h4 : head_start = 1) :
  distance_bob_walked total_distance yolanda_speed bob_speed head_start = 648 / 17 := by
  sorry

#eval (648 : ℚ) / 17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_theorem_l807_80770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l807_80784

noncomputable def simple_interest (P : ℝ) (R : ℝ) : ℝ := P * R * 2 / 100

noncomputable def compound_interest (P : ℝ) (R : ℝ) : ℝ := P * ((1 + R / 100) ^ 2 - 1)

theorem interest_rate_calculation (P : ℝ) (R : ℝ) 
  (h1 : simple_interest P R = 58)
  (h2 : compound_interest P R = 59.45) :
  ∃ ε > 0, |R - 22.36| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l807_80784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_existence_and_uniqueness_l807_80786

theorem quadratic_polynomial_existence_and_uniqueness :
  ∃! q : ℚ → ℚ, 
    (q (-8) = 0) ∧ (q 3 = 0) ∧ (q 4 = 40) ∧
    (∃ a b c : ℚ, ∀ x, q x = a * x^2 + b * x + c ∧
                       a = 10/3 ∧ b = 50/3 ∧ c = -80) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_existence_and_uniqueness_l807_80786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_speed_for_fuel_efficiency_l807_80797

/-- Fuel consumption function (L/h) for a given speed x (km/h) -/
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1/5) * (x - 100 + 4500/x)

/-- The problem statement -/
theorem max_speed_for_fuel_efficiency :
  ∀ x : ℝ, 60 ≤ x ∧ x ≤ 120 →
    (fuel_consumption x ≤ 9 ↔ x ≤ 100) :=
by
  intro x h
  sorry

#check max_speed_for_fuel_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_speed_for_fuel_efficiency_l807_80797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_linear_function_plus_inverse_equals_neg_two_l807_80778

theorem no_fractional_linear_function_plus_inverse_equals_neg_two :
  ∀ (a b c d : ℝ), c ≠ 0 →
    ¬∃ (f : ℝ → ℝ),
      (∀ x, f x = (a * x + b) / (c * x + d)) ∧
      (∀ x, x ≠ -d/c → x ≠ a/c → f x + (f⁻¹) x = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_linear_function_plus_inverse_equals_neg_two_l807_80778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cities_distance_l807_80752

/-- Calculates the actual distance between two cities given the map distance and scale. -/
noncomputable def actual_distance (map_distance : ℝ) (scale_map : ℝ) (scale_actual : ℝ) : ℝ :=
  (map_distance * scale_actual) / scale_map

/-- Theorem stating that the actual distance between two cities is 180 miles. -/
theorem cities_distance : 
  let map_distance : ℝ := 15
  let scale_map : ℝ := 0.25
  let scale_actual : ℝ := 3
  actual_distance map_distance scale_map scale_actual = 180 := by
  -- Unfold the definition of actual_distance
  unfold actual_distance
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cities_distance_l807_80752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_scaling_l807_80785

noncomputable def variance (data : List ℝ) : ℝ := sorry
noncomputable def stdDev (data : List ℝ) : ℝ := sorry

theorem data_scaling (x₁ x₂ x₃ x₄ x₅ : ℝ) (a : ℝ) :
  let X := [x₁, x₂, x₃, x₄, x₅]
  let aX := [a * x₁, a * x₂, a * x₃, a * x₄, a * x₅]
  variance X = 2 ∧ a > 0 ∧ stdDev aX = 2 * Real.sqrt 2 → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_scaling_l807_80785
