import Mathlib

namespace NUMINAMATH_CALUDE_horner_evaluation_f_5_l1738_173867

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

theorem horner_evaluation_f_5 : f 5 = 56 := by sorry

end NUMINAMATH_CALUDE_horner_evaluation_f_5_l1738_173867


namespace NUMINAMATH_CALUDE_equation_solution_l1738_173842

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-5 * 4 + 2)| ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1738_173842


namespace NUMINAMATH_CALUDE_ryan_marbles_ryan_has_28_marbles_l1738_173875

theorem ryan_marbles (chris_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let total_marbles := chris_marbles + remaining_marbles * 2
  total_marbles - chris_marbles

theorem ryan_has_28_marbles :
  ryan_marbles 12 20 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ryan_marbles_ryan_has_28_marbles_l1738_173875


namespace NUMINAMATH_CALUDE_unique_solution_fractional_equation_l1738_173888

theorem unique_solution_fractional_equation :
  ∃! x : ℚ, (1 : ℚ) / (x - 3) = (3 : ℚ) / (x - 6) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fractional_equation_l1738_173888


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l1738_173872

/-- Given two lines in the plane, if they are perpendicular, then the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 1 = 0 → x + a*y + 3 = 0 → (2 : ℝ) * (1/a) = -1) → 
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l1738_173872


namespace NUMINAMATH_CALUDE_variance_range_best_for_stability_l1738_173878

/-- Represents a set of exam scores -/
def ExamScores := List ℝ

/-- Calculates the variance of a list of numbers -/
def variance (scores : ExamScores) : ℝ := sorry

/-- Calculates the range of a list of numbers -/
def range (scores : ExamScores) : ℝ := sorry

/-- Calculates the mean of a list of numbers -/
def mean (scores : ExamScores) : ℝ := sorry

/-- Calculates the median of a list of numbers -/
def median (scores : ExamScores) : ℝ := sorry

/-- Calculates the mode of a list of numbers -/
def mode (scores : ExamScores) : ℝ := sorry

/-- Measures how well a statistic represents the stability of scores -/
def stabilityMeasure (f : ExamScores → ℝ) : ℝ := sorry

theorem variance_range_best_for_stability (scores : ExamScores) 
  (h : scores.length = 5) :
  (stabilityMeasure variance > stabilityMeasure mean) ∧
  (stabilityMeasure variance > stabilityMeasure median) ∧
  (stabilityMeasure variance > stabilityMeasure mode) ∧
  (stabilityMeasure range > stabilityMeasure mean) ∧
  (stabilityMeasure range > stabilityMeasure median) ∧
  (stabilityMeasure range > stabilityMeasure mode) := by
  sorry

end NUMINAMATH_CALUDE_variance_range_best_for_stability_l1738_173878


namespace NUMINAMATH_CALUDE_sequence_problem_l1738_173863

/-- Given a sequence a_n and a geometric sequence b_n where
    a_1 = 2, b_n = a_{n+1} / a_n for all n, and b_10 * b_11 = 2,
    prove that a_21 = 2^11 -/
theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  a 1 = 2 →
  (∀ n, b n = a (n + 1) / a n) →
  (∃ r, ∀ n, b (n + 1) = r * b n) →
  b 10 * b 11 = 2 →
  a 21 = 2^11 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1738_173863


namespace NUMINAMATH_CALUDE_probability_one_more_red_eq_three_eighths_l1738_173871

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | White

/-- Represents the outcome of three draws -/
def ThreeDraw := (BallColor × BallColor × BallColor)

/-- The set of all possible outcomes when drawing a ball three times with replacement -/
def allOutcomes : Finset ThreeDraw := sorry

/-- Predicate to check if an outcome has one more red ball than white balls -/
def hasOneMoreRed (draw : ThreeDraw) : Prop := sorry

/-- The set of favorable outcomes (one more red than white) -/
def favorableOutcomes : Finset ThreeDraw := sorry

/-- The probability of drawing the red ball one more time than the white ball -/
def probabilityOneMoreRed : ℚ := (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem: The probability of drawing the red ball one more time than the white ball is 3/8 -/
theorem probability_one_more_red_eq_three_eighths : 
  probabilityOneMoreRed = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_one_more_red_eq_three_eighths_l1738_173871


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1738_173818

theorem average_of_remaining_numbers
  (total : ℝ)
  (group1 : ℝ)
  (group2 : ℝ)
  (h1 : total = 6 * 6.40)
  (h2 : group1 = 2 * 6.2)
  (h3 : group2 = 2 * 6.1) :
  (total - group1 - group2) / 2 = 6.9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1738_173818


namespace NUMINAMATH_CALUDE_max_visible_faces_sum_l1738_173840

/-- Represents a single die -/
structure Die :=
  (top : ℕ)
  (bottom : ℕ)
  (left : ℕ)
  (right : ℕ)
  (front : ℕ)
  (back : ℕ)

/-- The grid of dice -/
def DiceGrid := Matrix (Fin 10) (Fin 10) Die

/-- Condition: sum of dots on opposite faces is 7 -/
def oppositeFacesSum7 (d : Die) : Prop :=
  d.top + d.bottom = 7 ∧ d.left + d.right = 7 ∧ d.front + d.back = 7

/-- All dice in the grid satisfy the opposite faces sum condition -/
def allDiceSatisfyCondition (grid : DiceGrid) : Prop :=
  ∀ i j, oppositeFacesSum7 (grid i j)

/-- Count of visible faces -/
def visibleFacesCount : ℕ := 240

/-- Sum of dots on visible faces -/
def visibleFacesSum (grid : DiceGrid) : ℕ :=
  sorry  -- Definition would involve summing specific faces based on visibility

/-- Main theorem -/
theorem max_visible_faces_sum (grid : DiceGrid) 
  (h1 : allDiceSatisfyCondition grid) : 
  visibleFacesSum grid ≤ 920 :=
sorry

end NUMINAMATH_CALUDE_max_visible_faces_sum_l1738_173840


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l1738_173835

/-- A right triangle with a circle inscribed on one side --/
structure RightTriangleWithCircle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Point where the circle meets AC
  D : ℝ × ℝ
  -- B is a right angle
  right_angle_at_B : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- BC is the diameter of the circle
  BC_is_diameter : ∃ (center : ℝ × ℝ), 
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = (center.1 - C.1)^2 + (center.2 - C.2)^2
  -- D lies on AC
  D_on_AC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t*(C.1 - A.1), A.2 + t*(C.2 - A.2))
  -- D lies on the circle
  D_on_circle : ∃ (center : ℝ × ℝ), 
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = (center.1 - B.1)^2 + (center.2 - B.2)^2

/-- The theorem to be proved --/
theorem right_triangle_circle_theorem (t : RightTriangleWithCircle) 
  (h1 : Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = 3)
  (h2 : Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 6) :
  Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2) = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l1738_173835


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1738_173810

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1738_173810


namespace NUMINAMATH_CALUDE_determine_opposite_resident_l1738_173891

/-- Represents a resident on the hexagonal street -/
inductive Resident
| Knight
| Liar

/-- Represents a vertex of the hexagonal street -/
def Vertex := Fin 6

/-- Represents the street layout -/
structure HexagonalStreet where
  residents : Vertex → Resident

/-- Represents a letter asking about neighbor relationships -/
structure Letter where
  sender : Vertex
  recipient : Vertex
  askedAbout : Vertex

/-- Determines if two vertices are neighbors in a regular hexagon -/
def areNeighbors (v1 v2 : Vertex) : Bool :=
  (v1.val + 1) % 6 = v2.val ∨ (v1.val + 5) % 6 = v2.val

/-- The main theorem stating that it's possible to determine the opposite resident with at most 4 letters -/
theorem determine_opposite_resident (street : HexagonalStreet) (start : Vertex) :
  ∃ (letters : List Letter), letters.length ≤ 4 ∧
    ∃ (opposite : Vertex), (start.val + 3) % 6 = opposite.val ∧
      (∀ (response : Letter → Bool), 
        ∃ (deduced_resident : Resident), street.residents opposite = deduced_resident) :=
  sorry

end NUMINAMATH_CALUDE_determine_opposite_resident_l1738_173891


namespace NUMINAMATH_CALUDE_ellipse_problem_l1738_173812

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The problem statement -/
theorem ellipse_problem (E : Ellipse) 
  (h_major_axis : E.a = 2 * Real.sqrt 2)
  (A B C : PointOnEllipse E)
  (h_A_vertex : A.x = E.a ∧ A.y = 0)
  (h_BC_origin : ∃ t : ℝ, B.x * t = C.x ∧ B.y * t = C.y)
  (h_B_first_quad : B.x > 0 ∧ B.y > 0)
  (h_BC_AB : Real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2) = 2 * Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2))
  (h_cos_ABC : (A.x - B.x) / Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 1/5) :
  (E.a^2 = 8 ∧ E.b^2 = 4) ∧
  ∃ (lower upper : ℝ), lower = Real.sqrt 14 / 2 ∧ upper = Real.sqrt 6 ∧
    ∀ (M N : PointOnEllipse E) (l : ℝ → ℝ),
      (∀ x y : ℝ, x^2 + y^2 = 1 → (y - l x) * (1 + l x * l x) = 0) →
      M ≠ N →
      (∃ t : ℝ, M.y = l M.x + t ∧ N.y = l N.x + t) →
      lower < (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ∧
      (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ≤ upper := by
  sorry

end NUMINAMATH_CALUDE_ellipse_problem_l1738_173812


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1738_173895

/-- Given a wire of length 52 cm, the maximum area of a rectangle that can be formed is 169 cm². -/
theorem max_rectangle_area (wire_length : ℝ) (h : wire_length = 52) : 
  (∀ l w : ℝ, l > 0 → w > 0 → 2 * (l + w) ≤ wire_length → l * w ≤ 169) ∧ 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = wire_length ∧ l * w = 169) :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1738_173895


namespace NUMINAMATH_CALUDE_max_diff_color_triangles_17gon_l1738_173892

/-- Regular 17-gon with colored edges -/
structure ColoredPolygon where
  n : Nat
  colors : Nat
  no_monochromatic : Bool

/-- The number of edges in a regular 17-gon -/
def num_edges (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1)) / 2

/-- The total number of triangles in a regular 17-gon -/
def total_triangles (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1) * (p.n - 2)) / 6

/-- The minimum number of isosceles triangles (triangles with at least two sides of the same color) -/
def min_isosceles_triangles (p : ColoredPolygon) : Nat :=
  p.n * p.colors

/-- The maximum number of triangles with all edges of different colors -/
def max_diff_color_triangles (p : ColoredPolygon) : Nat :=
  total_triangles p - min_isosceles_triangles p

/-- Theorem: The maximum number of triangles with all edges of different colors in a regular 17-gon
    with 8 colors and no monochromatic triangles is 544 -/
theorem max_diff_color_triangles_17gon :
  ∀ p : ColoredPolygon,
    p.n = 17 →
    p.colors = 8 →
    p.no_monochromatic = true →
    num_edges p = 136 →
    max_diff_color_triangles p = 544 := by
  sorry

end NUMINAMATH_CALUDE_max_diff_color_triangles_17gon_l1738_173892


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1738_173852

theorem consecutive_integers_sum_of_squares : 
  ∀ a : ℤ, (a - 2) * a * (a + 2) = 36 * a → 
  (a - 2)^2 + a^2 + (a + 2)^2 = 200 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1738_173852


namespace NUMINAMATH_CALUDE_james_record_beat_l1738_173841

/-- James' football scoring record --/
theorem james_record_beat (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : 
  touchdowns_per_game = 4 →
  points_per_touchdown = 6 →
  games_in_season = 15 →
  two_point_conversions = 6 →
  old_record = 300 →
  (touchdowns_per_game * points_per_touchdown * games_in_season + 
   two_point_conversions * 2) - old_record = 72 := by
  sorry

#check james_record_beat

end NUMINAMATH_CALUDE_james_record_beat_l1738_173841


namespace NUMINAMATH_CALUDE_lateralEdgeAngle_specific_pyramid_l1738_173882

/-- A regular truncated quadrangular pyramid -/
structure TruncatedPyramid where
  upperBaseSide : ℝ
  lowerBaseSide : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ

/-- The angle between the lateral edge and the base plane of a truncated pyramid -/
def lateralEdgeAngle (p : TruncatedPyramid) : ℝ := sorry

/-- Theorem: The angle between the lateral edge and the base plane of a specific truncated pyramid -/
theorem lateralEdgeAngle_specific_pyramid :
  ∀ (p : TruncatedPyramid),
    p.lowerBaseSide = 5 * p.upperBaseSide →
    p.lateralSurfaceArea = p.height ^ 2 →
    lateralEdgeAngle p = Real.arctan (Real.sqrt (9 + 3 * Real.sqrt 10)) := by
  sorry

end NUMINAMATH_CALUDE_lateralEdgeAngle_specific_pyramid_l1738_173882


namespace NUMINAMATH_CALUDE_g_of_negative_two_l1738_173811

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- Theorem statement
theorem g_of_negative_two : g (-2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_two_l1738_173811


namespace NUMINAMATH_CALUDE_square_roots_sum_product_l1738_173813

theorem square_roots_sum_product (m n : ℂ) : 
  m ^ 2 = 2023 → n ^ 2 = 2023 → m + 2 * m * n + n = -4046 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_product_l1738_173813


namespace NUMINAMATH_CALUDE_club_enrollment_l1738_173849

theorem club_enrollment (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 150 →
  math = 90 →
  chem = 70 →
  both = 20 →
  total - (math + chem - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_club_enrollment_l1738_173849


namespace NUMINAMATH_CALUDE_train_cars_count_l1738_173846

theorem train_cars_count (cars_per_15s : ℕ) (total_time : ℕ) : 
  cars_per_15s = 10 → total_time = 210 → (total_time * cars_per_15s) / 15 = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l1738_173846


namespace NUMINAMATH_CALUDE_min_value_theorem_l1738_173843

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1738_173843


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1738_173884

/-- Two people A and B walk towards each other and meet. This theorem proves B's speed. -/
theorem walking_speed_problem (speed_A speed_B : ℝ) (initial_distance total_time : ℝ) : 
  speed_A = 5 →
  initial_distance = 24 →
  total_time = 2 →
  speed_A * total_time + speed_B * total_time = initial_distance →
  speed_B = 7 := by
  sorry

#check walking_speed_problem

end NUMINAMATH_CALUDE_walking_speed_problem_l1738_173884


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1738_173837

theorem inequality_system_solution (x : ℝ) :
  (x - 1 < 2*x + 1) ∧ ((2*x - 5) / 3 ≤ 1) → -2 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1738_173837


namespace NUMINAMATH_CALUDE_NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l1738_173807

/-- Represents the solubility of a compound in water -/
inductive Solubility
  | Soluble
  | SlightlySoluble
  | Insoluble

/-- Represents a metal ion -/
inductive MetalIon
  | Ag
  | Mg
  | Sr

/-- Represents a reagent -/
inductive Reagent
  | NaCl
  | NaOH
  | Na2SO4
  | Na3PO4

/-- Returns the solubility of the compound formed by a metal ion and a reagent -/
def solubility (ion : MetalIon) (reagent : Reagent) : Solubility :=
  match ion, reagent with
  | MetalIon.Ag, Reagent.NaCl => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Sr, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Ag, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Sr, Reagent.NaOH => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Mg, Reagent.Na2SO4 => Solubility.Soluble
  | MetalIon.Sr, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Mg, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Sr, Reagent.Na3PO4 => Solubility.Insoluble

/-- Checks if a reagent selectively precipitates Ag+ -/
def selectivelyPrecipitatesAg (reagent : Reagent) : Prop :=
  solubility MetalIon.Ag reagent = Solubility.Insoluble ∧
  solubility MetalIon.Mg reagent = Solubility.Soluble ∧
  solubility MetalIon.Sr reagent = Solubility.Soluble

theorem NaCl_selectively_precipitates_Ag :
  selectivelyPrecipitatesAg Reagent.NaCl :=
by sorry

theorem other_reagents_do_not_selectively_precipitate_Ag :
  ∀ r : Reagent, r ≠ Reagent.NaCl → ¬selectivelyPrecipitatesAg r :=
by sorry

end NUMINAMATH_CALUDE_NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l1738_173807


namespace NUMINAMATH_CALUDE_smallest_b_value_l1738_173850

theorem smallest_b_value (b : ℤ) (Q : ℤ → ℤ) : 
  b > 0 →
  (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), Q x = a₀ * x^2 + a₁ * x + a₂) →
  Q 1 = b ∧ Q 4 = b ∧ Q 7 = b ∧ Q 10 = b →
  Q 2 = -b ∧ Q 5 = -b ∧ Q 8 = -b ∧ Q 11 = -b →
  (∀ c : ℤ, c > 0 ∧ 
    (∃ (P : ℤ → ℤ), (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), P x = a₀ * x^2 + a₁ * x + a₂) ∧
      P 1 = c ∧ P 4 = c ∧ P 7 = c ∧ P 10 = c ∧
      P 2 = -c ∧ P 5 = -c ∧ P 8 = -c ∧ P 11 = -c) →
    c ≥ b) →
  b = 1260 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1738_173850


namespace NUMINAMATH_CALUDE_no_integer_solution_l1738_173834

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1738_173834


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1738_173823

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1738_173823


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1738_173881

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ 1/2 < x ∧ x < 2

-- Theorem statement
theorem quadratic_inequality (a : ℝ) (h : solution_set a) :
  a = -2 ∧
  ∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1738_173881


namespace NUMINAMATH_CALUDE_anns_sledding_speed_l1738_173898

/-- Given the conditions of Mary and Ann's sledding trip, prove Ann's speed -/
theorem anns_sledding_speed 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : time_difference = 13)
  : ∃ (ann_speed : ℝ), ann_speed = 40 ∧ 
    ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_anns_sledding_speed_l1738_173898


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1738_173848

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1738_173848


namespace NUMINAMATH_CALUDE_quadratic_function_negative_at_four_l1738_173861

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_negative_at_four
  (a b c : ℝ)
  (h1 : f a b c (-1) = -3)
  (h2 : f a b c 0 = 1)
  (h3 : f a b c 1 = 3)
  (h4 : f a b c 3 = 1) :
  f a b c 4 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_negative_at_four_l1738_173861


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1738_173809

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1738_173809


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1738_173887

theorem opposite_of_2023 : 
  ∃ y : ℤ, y + 2023 = 0 ∧ y = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1738_173887


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1738_173827

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  first : ℤ
  /-- The common difference between consecutive terms -/
  diff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_second_term
  (seq : ArithmeticSequence)
  (h16 : seq.nthTerm 16 = 8)
  (h17 : seq.nthTerm 17 = 10) :
  seq.nthTerm 2 = -20 := by
  sorry

#check arithmetic_sequence_second_term

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1738_173827


namespace NUMINAMATH_CALUDE_senior_class_boys_count_l1738_173899

theorem senior_class_boys_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (sample_boys_girls_diff : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 200)
  (h3 : sample_boys_girls_diff = 10) :
  ∃ (total_boys : ℕ), 
    total_boys = 660 ∧ 
    (total_boys : ℚ) / total_students = 
      ((sample_size + sample_boys_girls_diff) / 2 : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_senior_class_boys_count_l1738_173899


namespace NUMINAMATH_CALUDE_prob_sum_five_twice_l1738_173889

/-- A die with 4 sides numbered 1 to 4. -/
def FourSidedDie : Finset ℕ := {1, 2, 3, 4}

/-- The set of all possible outcomes when rolling two 4-sided dice. -/
def TwoDiceOutcomes : Finset (ℕ × ℕ) :=
  FourSidedDie.product FourSidedDie

/-- The sum of two dice rolls. -/
def diceSum (roll : ℕ × ℕ) : ℕ := roll.1 + roll.2

/-- The set of all rolls that sum to 5. -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  TwoDiceOutcomes.filter (λ roll => diceSum roll = 5)

/-- The probability of rolling a sum of 5 with two 4-sided dice. -/
def probSumFive : ℚ :=
  (sumFiveOutcomes.card : ℚ) / (TwoDiceOutcomes.card : ℚ)

theorem prob_sum_five_twice :
  probSumFive * probSumFive = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_five_twice_l1738_173889


namespace NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l1738_173825

/-- The number of apps Dave deleted -/
def apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : ℕ :=
  initial_apps - remaining_apps

/-- Theorem stating that Dave deleted 11 apps -/
theorem dave_deleted_eleven_apps : apps_deleted 16 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l1738_173825


namespace NUMINAMATH_CALUDE_four_six_eight_triangle_l1738_173886

/-- A predicate that determines if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 4, 6, and 8 can form a triangle -/
theorem four_six_eight_triangle :
  canFormTriangle 4 6 8 := by sorry

end NUMINAMATH_CALUDE_four_six_eight_triangle_l1738_173886


namespace NUMINAMATH_CALUDE_typing_area_percentage_l1738_173836

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_length sheet_width side_margin top_bottom_margin : ℝ) 
  (sheet_length_pos : 0 < sheet_length)
  (sheet_width_pos : 0 < sheet_width)
  (side_margin_pos : 0 < side_margin)
  (top_bottom_margin_pos : 0 < top_bottom_margin)
  (side_margin_fit : 2 * side_margin < sheet_length)
  (top_bottom_margin_fit : 2 * top_bottom_margin < sheet_width) :
  let total_area := sheet_length * sheet_width
  let typing_length := sheet_length - 2 * side_margin
  let typing_width := sheet_width - 2 * top_bottom_margin
  let typing_area := typing_length * typing_width
  (typing_area / total_area) * 100 = 64 :=
sorry

end NUMINAMATH_CALUDE_typing_area_percentage_l1738_173836


namespace NUMINAMATH_CALUDE_volume_problem_l1738_173877

/-- Given a volume that is the product of three numbers, where two of the numbers are 18 and 6,
    and 48 cubes of edge 3 can be inserted into this volume, prove that the first number in the product is 12. -/
theorem volume_problem (volume : ℝ) (first_number : ℝ) : 
  volume = first_number * 18 * 6 →
  volume = 48 * (3 : ℝ)^3 →
  first_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_volume_problem_l1738_173877


namespace NUMINAMATH_CALUDE_geometric_progression_cubic_roots_l1738_173859

theorem geometric_progression_cubic_roots (x y z r p q : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  y^2 = r * x^2 →
  z^2 = r * y^2 →
  x^3 - p*x^2 + q*x - r = 0 →
  y^3 - p*y^2 + q*y - r = 0 →
  z^3 - p*z^2 + q*z - r = 0 →
  r^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_cubic_roots_l1738_173859


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_l1738_173876

theorem tangent_slope_three_points (x : ℝ) :
  (3 * x^2 = 3) → (x = 1 ∨ x = -1) := by sorry

#check tangent_slope_three_points

end NUMINAMATH_CALUDE_tangent_slope_three_points_l1738_173876


namespace NUMINAMATH_CALUDE_quotient_relation_l1738_173869

theorem quotient_relation : ∃ (k l : ℝ), k ≠ l ∧ (64 / k = 4 * (64 / l)) := by
  sorry

end NUMINAMATH_CALUDE_quotient_relation_l1738_173869


namespace NUMINAMATH_CALUDE_karting_track_routes_l1738_173820

/-- Represents the number of distinct routes ending at point A after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 0
| (n+2) => M (n+1) + M n

/-- The karting track problem -/
theorem karting_track_routes : M 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_karting_track_routes_l1738_173820


namespace NUMINAMATH_CALUDE_correct_calculation_l1738_173860

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1738_173860


namespace NUMINAMATH_CALUDE_equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l1738_173894

/-- Represents a triangle ABC with inscribed rectangles --/
structure TriangleWithRectangles where
  -- The length of the base AB
  base : ℝ
  -- The height of the triangle corresponding to base AB
  height : ℝ
  -- A function that given a real number between 0 and 1, returns the dimensions of the inscribed rectangle
  rectangleDimensions : ℝ → (ℝ × ℝ)

/-- The perimeter of a rectangle given its dimensions --/
def rectanglePerimeter (dimensions : ℝ × ℝ) : ℝ :=
  2 * (dimensions.1 + dimensions.2)

/-- Theorem: If base equals height, then all inscribed rectangles have equal perimeters --/
theorem equal_perimeters_if_base_equals_height (triangle : TriangleWithRectangles) :
  triangle.base = triangle.height →
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y) :=
sorry

/-- Theorem: If all inscribed rectangles have equal perimeters, then base equals height --/
theorem base_equals_height_if_equal_perimeters (triangle : TriangleWithRectangles) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
   rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y)) →
  triangle.base = triangle.height :=
sorry

end NUMINAMATH_CALUDE_equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l1738_173894


namespace NUMINAMATH_CALUDE_parabola_translation_l1738_173883

/-- A parabola in the Cartesian coordinate system. -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x-h)^2 + k. -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- The translation of a parabola. -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation (p1 p2 : Parabola) :
  p1.a = 1/2 ∧ p1.h = 0 ∧ p1.k = -1 ∧
  p2.a = 1/2 ∧ p2.h = 4 ∧ p2.k = 2 →
  p2 = translate p1 4 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l1738_173883


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1738_173866

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ n % 9 = 0 → n ≥ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1738_173866


namespace NUMINAMATH_CALUDE_dawn_time_verify_solution_l1738_173815

/-- Represents the time in hours from dawn to noon -/
def time_dawn_to_noon : ℝ := sorry

/-- Represents the time in hours from noon to 4 PM -/
def time_noon_to_4pm : ℝ := 4

/-- Represents the time in hours from noon to 9 PM -/
def time_noon_to_9pm : ℝ := 9

/-- The theorem stating that the time from dawn to noon is 6 hours -/
theorem dawn_time : time_dawn_to_noon = 6 := by
  sorry

/-- Verification of the solution using speed ratios -/
theorem verify_solution :
  time_dawn_to_noon / time_noon_to_4pm = time_noon_to_9pm / time_dawn_to_noon := by
  sorry

end NUMINAMATH_CALUDE_dawn_time_verify_solution_l1738_173815


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l1738_173880

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α β : Plane) :
  contained_in a β → parallel α β → line_parallel_to_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l1738_173880


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l1738_173855

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  (∀ (x₁ x₂ : ℝ), 
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
  d = Real.sqrt 2 * (1 - Real.log 2) := by
  sorry

#check min_distance_between_curves

end NUMINAMATH_CALUDE_min_distance_between_curves_l1738_173855


namespace NUMINAMATH_CALUDE_quartic_to_quadratic_reduction_l1738_173844

/-- Given a quartic equation and a substitution, prove it can be reduced to two quadratic equations -/
theorem quartic_to_quadratic_reduction (a b c : ℝ) (x y : ℝ) :
  (a * x^4 + b * x^3 + c * x^2 + b * x + a = 0) →
  (y = x + 1/x) →
  ∃ (y₁ y₂ : ℝ),
    (a * y^2 + b * y + (c - 2*a) = 0) ∧
    (x^2 - y₁ * x + 1 = 0 ∨ x^2 - y₂ * x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quartic_to_quadratic_reduction_l1738_173844


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l1738_173838

/-- The number of ways to distribute n distinct objects into k distinct bins,
    where each bin must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (Nat.choose n k + (Nat.choose n 2 * Nat.choose 3 2) / 2) * Nat.factorial k

/-- The theorem stating that distributing 5 distinct objects into 3 distinct bins,
    where each bin must contain at least one object, results in 150 ways. -/
theorem distribute_five_into_three :
  distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l1738_173838


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1738_173864

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x^2 + 1 < 0) ↔ (∃ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1738_173864


namespace NUMINAMATH_CALUDE_permutation_sum_l1738_173805

theorem permutation_sum (n : ℕ+) (h1 : n + 3 ≤ 2*n) (h2 : n + 1 ≤ 4) : 
  (Nat.descFactorial (2*n) (n+3)) + (Nat.descFactorial 4 (n+1)) = 744 :=
by sorry

end NUMINAMATH_CALUDE_permutation_sum_l1738_173805


namespace NUMINAMATH_CALUDE_least_meeting_time_for_four_horses_l1738_173885

def horse_lap_time (k : Nat) : Nat := 2 * k

def is_meeting_time (t : Nat) (horses : List Nat) : Prop :=
  ∀ h ∈ horses, t % (horse_lap_time h) = 0

theorem least_meeting_time_for_four_horses :
  ∃ T : Nat,
    T > 0 ∧
    (∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time T horses) ∧
    (∀ t : Nat, 0 < t ∧ t < T →
      ¬∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time t horses) ∧
    T = 24 := by sorry

end NUMINAMATH_CALUDE_least_meeting_time_for_four_horses_l1738_173885


namespace NUMINAMATH_CALUDE_well_problem_solution_l1738_173839

/-- The depth of the well and the rope lengths of five families -/
def well_problem (e : ℚ) : Prop :=
  ∃ (x a b c d : ℚ),
    -- Depth equations
    x = 2*a + b ∧
    x = 3*b + c ∧
    x = 4*c + d ∧
    x = 5*d + e ∧
    x = 6*e + a ∧
    -- Solutions
    x = (721/76)*e ∧
    a = (265/76)*e ∧
    b = (191/76)*e ∧
    c = (37/19)*e ∧
    d = (129/76)*e

/-- The well depth and rope lengths satisfy the given conditions -/
theorem well_problem_solution :
  ∀ e : ℚ, well_problem e :=
by sorry

end NUMINAMATH_CALUDE_well_problem_solution_l1738_173839


namespace NUMINAMATH_CALUDE_negation_of_forall_implication_l1738_173803

theorem negation_of_forall_implication (A B : Set α) :
  (¬ (∀ x, x ∈ A → x ∈ B)) ↔ (∃ x, x ∈ A ∧ x ∉ B) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_implication_l1738_173803


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1738_173800

/-- Given a hyperbola with standard equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if one of its asymptotes has equation y = 3x, then its eccentricity is √10. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1738_173800


namespace NUMINAMATH_CALUDE_min_obtuse_triangles_2003gon_l1738_173858

/-- A polygon inscribed in a circle -/
structure InscribedPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : InscribedPolygon) where
  triangle_count : ℕ
  triangle_count_eq : triangle_count = P.n - 2
  obtuse_count : ℕ
  acute_count : ℕ
  right_count : ℕ
  total_count : obtuse_count + acute_count + right_count = triangle_count
  max_non_obtuse : acute_count + right_count ≤ 2

/-- The theorem statement -/
theorem min_obtuse_triangles_2003gon :
  let P : InscribedPolygon := ⟨2003, by norm_num⟩
  ∀ T : Triangulation P, T.obtuse_count ≥ 1999 :=
by sorry

end NUMINAMATH_CALUDE_min_obtuse_triangles_2003gon_l1738_173858


namespace NUMINAMATH_CALUDE_quadratic_inequality_1_l1738_173893

theorem quadratic_inequality_1 (x : ℝ) :
  x^2 - 7*x + 12 > 0 ↔ x < 3 ∨ x > 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_1_l1738_173893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1738_173819

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_a3 : a 3 = 7) :
  a 8 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1738_173819


namespace NUMINAMATH_CALUDE_cantaloupe_price_l1738_173856

/-- Represents the problem of finding the price of cantaloupes --/
def CantalouperPriceProblem (C : ℚ) : Prop :=
  let initial_cantaloupes : ℕ := 30
  let initial_honeydews : ℕ := 27
  let dropped_cantaloupes : ℕ := 2
  let rotten_honeydews : ℕ := 3
  let final_cantaloupes : ℕ := 8
  let final_honeydews : ℕ := 9
  let honeydew_price : ℚ := 3
  let total_revenue : ℚ := 85
  let sold_cantaloupes : ℕ := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews : ℕ := initial_honeydews - final_honeydews - rotten_honeydews
  C * sold_cantaloupes + honeydew_price * sold_honeydews = total_revenue

/-- Theorem stating that the price of each cantaloupe is $2 --/
theorem cantaloupe_price : ∃ C : ℚ, CantalouperPriceProblem C ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupe_price_l1738_173856


namespace NUMINAMATH_CALUDE_venerts_in_45_degrees_l1738_173832

/-- Converts degrees to venerts given the number of venerts in a full circle -/
def degrees_to_venerts (full_circle_venerts : ℚ) (degrees : ℚ) : ℚ :=
  (degrees * full_circle_venerts) / 360

/-- Theorem: Given 600 venerts in a full circle, 45° is equivalent to 75 venerts -/
theorem venerts_in_45_degrees :
  degrees_to_venerts 600 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_venerts_in_45_degrees_l1738_173832


namespace NUMINAMATH_CALUDE_boris_neighbors_l1738_173890

-- Define the type for people
inductive Person : Type
  | Arkady | Boris | Vera | Galya | Danya | Egor

-- Define the circle as a function from positions to people
def Circle := Fin 6 → Person

-- Define the conditions
def satisfies_conditions (c : Circle) : Prop :=
  -- Danya stands next to Vera, on her right side
  ∃ i, c i = Person.Vera ∧ c (i + 1) = Person.Danya
  -- Galya stands opposite Egor
  ∧ ∃ j, c j = Person.Egor ∧ c (j + 3) = Person.Galya
  -- Egor stands next to Danya
  ∧ ∃ k, c k = Person.Danya ∧ (c (k + 1) = Person.Egor ∨ c (k - 1) = Person.Egor)
  -- Arkady and Galya do not stand next to each other
  ∧ ∀ l, c l = Person.Arkady → c (l + 1) ≠ Person.Galya ∧ c (l - 1) ≠ Person.Galya

-- Theorem statement
theorem boris_neighbors (c : Circle) (h : satisfies_conditions c) :
  ∃ i, c i = Person.Boris ∧ 
    ((c (i - 1) = Person.Arkady ∧ c (i + 1) = Person.Galya) ∨
     (c (i - 1) = Person.Galya ∧ c (i + 1) = Person.Arkady)) :=
by
  sorry

end NUMINAMATH_CALUDE_boris_neighbors_l1738_173890


namespace NUMINAMATH_CALUDE_sine_shift_overlap_l1738_173816

/-- The smallest positive value of ω that makes the sine function overlap with its shifted version -/
theorem sine_shift_overlap : ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x : ℝ, Real.sin (ω * x + π / 3) = Real.sin (ω * (x - π / 3) + π / 3)) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x + π / 3) = Real.sin (ω' * (x - π / 3) + π / 3)) → 
    ω ≤ ω') ∧
  ω = 2 * π := by
sorry

end NUMINAMATH_CALUDE_sine_shift_overlap_l1738_173816


namespace NUMINAMATH_CALUDE_remainder_17_power_1999_mod_29_l1738_173865

theorem remainder_17_power_1999_mod_29 : 17^1999 % 29 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_1999_mod_29_l1738_173865


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l1738_173853

theorem complex_magnitude_proof (z : ℂ) :
  (Complex.arg z = Real.pi / 3) →
  (Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2)) →
  (Complex.abs z = Real.sqrt 2 + 1 ∨ Complex.abs z = Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l1738_173853


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l1738_173826

/-- Represents a geometric figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l1738_173826


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1738_173831

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 players, where each player plays every other player
    exactly once, and each game involves two players, the total number of games played is 190. --/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1738_173831


namespace NUMINAMATH_CALUDE_original_ratio_l1738_173854

theorem original_ratio (x y : ℕ) (h1 : y = 72) (h2 : (x + 6) / y = 1 / 3) : y / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l1738_173854


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1738_173804

def is_nonprime (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

def has_no_small_prime_factor (n : ℕ) : Prop := ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_nonprime_with_large_factors : 
  ∃ n : ℕ, is_nonprime n ∧ has_no_small_prime_factor n ∧ 
  (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_small_prime_factor m)) ∧
  n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1738_173804


namespace NUMINAMATH_CALUDE_poster_cost_l1738_173830

theorem poster_cost (initial_money : ℕ) (book1_cost : ℕ) (book2_cost : ℕ) (num_posters : ℕ) :
  initial_money = 20 →
  book1_cost = 8 →
  book2_cost = 4 →
  num_posters = 2 →
  initial_money - (book1_cost + book2_cost) = num_posters * (initial_money - (book1_cost + book2_cost)) / num_posters :=
by
  sorry

end NUMINAMATH_CALUDE_poster_cost_l1738_173830


namespace NUMINAMATH_CALUDE_no_real_solutions_for_square_rectangle_area_relation_l1738_173851

theorem no_real_solutions_for_square_rectangle_area_relation :
  ¬ ∃ x : ℝ, (x + 2) * (x - 5) = 2 * (x - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_square_rectangle_area_relation_l1738_173851


namespace NUMINAMATH_CALUDE_f_negative_two_eq_one_l1738_173801

/-- The function f(x) defined as x^5 + ax^3 + x^2 + bx + 2 -/
noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 1 -/
theorem f_negative_two_eq_one (a b : ℝ) (h : f a b 2 = 3) : f a b (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_two_eq_one_l1738_173801


namespace NUMINAMATH_CALUDE_min_photos_theorem_l1738_173829

/-- Represents the number of photographs for each grade --/
structure PhotoDistribution where
  total : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ
  seventh : ℕ
  first_to_third : ℕ

/-- The minimum number of photographs needed to ensure at least 15 from one grade (4th to 7th) --/
def min_photos_for_fifteen (d : PhotoDistribution) : ℕ := 
  d.first_to_third + 4 * 14 + 1

/-- The theorem stating the minimum number of photographs needed --/
theorem min_photos_theorem (d : PhotoDistribution) 
  (h_total : d.total = 130)
  (h_fourth : d.fourth = 35)
  (h_fifth : d.fifth = 30)
  (h_sixth : d.sixth = 25)
  (h_seventh : d.seventh = 20)
  (h_first_to_third : d.first_to_third = d.total - (d.fourth + d.fifth + d.sixth + d.seventh)) :
  min_photos_for_fifteen d = 77 := by
  sorry

#eval min_photos_for_fifteen ⟨130, 35, 30, 25, 20, 20⟩

end NUMINAMATH_CALUDE_min_photos_theorem_l1738_173829


namespace NUMINAMATH_CALUDE_missing_score_is_90_l1738_173822

def known_scores : List ℕ := [85, 90, 87, 93]

theorem missing_score_is_90 (x : ℕ) :
  (x :: known_scores).sum / (x :: known_scores).length = 89 →
  x = 90 := by
  sorry

end NUMINAMATH_CALUDE_missing_score_is_90_l1738_173822


namespace NUMINAMATH_CALUDE_zero_existence_l1738_173802

theorem zero_existence (f : ℝ → ℝ) (hf : Continuous f) 
  (h0 : f 0 = -3) (h1 : f 1 = 6) (h3 : f 3 = -5) :
  ∃ x₁ ∈ Set.Ioo 0 1, f x₁ = 0 ∧ ∃ x₂ ∈ Set.Ioo 1 3, f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_existence_l1738_173802


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1738_173873

/-- Proves that the ratio of Jake's weight after losing 33 pounds to his sister's weight is 2:1 -/
theorem jake_sister_weight_ratio :
  let jakes_current_weight : ℕ := 113
  let combined_weight : ℕ := 153
  let weight_loss : ℕ := 33
  let jakes_new_weight : ℕ := jakes_current_weight - weight_loss
  let sisters_weight : ℕ := combined_weight - jakes_current_weight
  (jakes_new_weight : ℚ) / (sisters_weight : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1738_173873


namespace NUMINAMATH_CALUDE_arrangements_eq_18_l1738_173845

/-- Represents the number of people in the lineup --/
def n : ℕ := 5

/-- Represents the possible positions for Person A --/
def A_positions : Set ℕ := {1, 2}

/-- Represents the possible positions for Person B --/
def B_positions : Set ℕ := {2, 3}

/-- The number of ways to arrange n people with the given constraints --/
def num_arrangements (n : ℕ) (A_pos : Set ℕ) (B_pos : Set ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 18 --/
theorem arrangements_eq_18 :
  num_arrangements n A_positions B_positions = 18 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_18_l1738_173845


namespace NUMINAMATH_CALUDE_frog_corner_probability_l1738_173824

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure State where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (s : State) : List State := sorry

/-- Probability of reaching a corner after exactly 4 hops -/
def probability_corner_4_hops : ℚ := sorry

/-- Main theorem stating the probability of reaching a corner after exactly 4 hops -/
theorem frog_corner_probability :
  probability_corner_4_hops = 217 / 256 := by sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l1738_173824


namespace NUMINAMATH_CALUDE_distance_between_points_l1738_173828

theorem distance_between_points : 
  let p1 : ℚ × ℚ := (-3/2, -1/2)
  let p2 : ℚ × ℚ := (9/2, 7/2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1738_173828


namespace NUMINAMATH_CALUDE_janice_earnings_this_week_l1738_173806

/-- Calculates Janice's earnings after deductions for a week -/
def janice_earnings (regular_daily_rate : ℚ) (days_worked : ℕ) 
  (weekday_overtime_rate : ℚ) (weekend_overtime_rate : ℚ)
  (weekday_overtime_shifts : ℕ) (weekend_overtime_shifts : ℕ)
  (tips : ℚ) (tax_rate : ℚ) : ℚ :=
  let regular_earnings := regular_daily_rate * days_worked
  let weekday_overtime := weekday_overtime_rate * weekday_overtime_shifts
  let weekend_overtime := weekend_overtime_rate * weekend_overtime_shifts
  let total_before_tax := regular_earnings + weekday_overtime + weekend_overtime + tips
  let tax := tax_rate * total_before_tax
  total_before_tax - tax

/-- Theorem stating Janice's earnings after deductions -/
theorem janice_earnings_this_week : 
  janice_earnings 30 6 15 20 2 1 10 (1/10) = 216 := by
  sorry


end NUMINAMATH_CALUDE_janice_earnings_this_week_l1738_173806


namespace NUMINAMATH_CALUDE_trisection_intersection_x_coordinate_l1738_173808

theorem trisection_intersection_x_coordinate : 
  let f : ℝ → ℝ := λ x => Real.log x
  let x₁ : ℝ := 2
  let x₂ : ℝ := 500
  let y₁ : ℝ := f x₁
  let y₂ : ℝ := f x₂
  let yC : ℝ := (2/3) * y₁ + (1/3) * y₂
  ∃ x₃ : ℝ, f x₃ = yC ∧ x₃ = 10 * (2^(2/3)) * (5^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_trisection_intersection_x_coordinate_l1738_173808


namespace NUMINAMATH_CALUDE_tripled_rectangle_area_l1738_173862

/-- Theorem: New area of a tripled rectangle --/
theorem tripled_rectangle_area (k m : ℝ) (hk : k > 0) (hm : m > 0) : 
  let original_area := (6 * k) * (4 * m)
  let new_area := 3 * original_area
  new_area = 72 * k * m := by
  sorry


end NUMINAMATH_CALUDE_tripled_rectangle_area_l1738_173862


namespace NUMINAMATH_CALUDE_bottle_production_l1738_173821

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 8 such machines will produce 1440 bottles in 4 minutes. -/
theorem bottle_production (machines_base : ℕ) (bottles_per_minute : ℕ) (machines_new : ℕ) (minutes : ℕ)
    (h1 : machines_base = 6)
    (h2 : bottles_per_minute = 270)
    (h3 : machines_new = 8)
    (h4 : minutes = 4) :
    (machines_new * (bottles_per_minute / machines_base) * minutes) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l1738_173821


namespace NUMINAMATH_CALUDE_max_product_sum_l1738_173857

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 4, 5} : Finset ℕ) →
  b ∈ ({1, 3, 4, 5} : Finset ℕ) →
  c ∈ ({1, 3, 4, 5} : Finset ℕ) →
  d ∈ ({1, 3, 4, 5} : Finset ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a * b + b * c + c * d + d * a) ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l1738_173857


namespace NUMINAMATH_CALUDE_adlai_animal_legs_l1738_173868

/-- The number of legs for each animal type --/
def dogsLegs : ℕ := 4
def chickenLegs : ℕ := 2
def catsLegs : ℕ := 4
def spidersLegs : ℕ := 8
def octopusLegs : ℕ := 0

/-- Adlai's animal collection --/
def numDogs : ℕ := 2
def numChickens : ℕ := 1
def numCats : ℕ := 3
def numSpiders : ℕ := 4
def numOctopuses : ℕ := 5

/-- The total number of animal legs in Adlai's collection --/
def totalLegs : ℕ := 
  numDogs * dogsLegs + 
  numChickens * chickenLegs + 
  numCats * catsLegs + 
  numSpiders * spidersLegs + 
  numOctopuses * octopusLegs

theorem adlai_animal_legs : totalLegs = 54 := by
  sorry

end NUMINAMATH_CALUDE_adlai_animal_legs_l1738_173868


namespace NUMINAMATH_CALUDE_original_number_proof_l1738_173817

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 34 → 
  x + y = 37.2 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1738_173817


namespace NUMINAMATH_CALUDE_nested_radicals_solution_l1738_173814

-- Define the left-hand side of the equation
noncomputable def leftSide (x : ℝ) : ℝ := 
  Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- Define the right-hand side of the equation
noncomputable def rightSide (x : ℝ) : ℝ := 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

-- State the theorem
theorem nested_radicals_solution :
  ∃! (x : ℝ), x > 0 ∧ leftSide x = rightSide x :=
by
  -- The unique solution is 2
  use 2
  sorry

end NUMINAMATH_CALUDE_nested_radicals_solution_l1738_173814


namespace NUMINAMATH_CALUDE_chair_sequence_l1738_173897

theorem chair_sequence (seq : ℕ → ℕ) 
  (h1 : seq 1 = 14)
  (h2 : seq 2 = 23)
  (h3 : seq 3 = 32)
  (h4 : seq 4 = 41)
  (h6 : seq 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → seq (n + 1) - seq n = seq 2 - seq 1) :
  seq 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_chair_sequence_l1738_173897


namespace NUMINAMATH_CALUDE_stationery_shop_restocking_l1738_173847

/-- Calculates the total number of pencils and rulers after restocking in a stationery shop. -/
theorem stationery_shop_restocking
  (initial_pencils : ℕ)
  (initial_pens : ℕ)
  (initial_rulers : ℕ)
  (sold_pencils : ℕ)
  (sold_pens : ℕ)
  (given_rulers : ℕ)
  (pencil_restock_factor : ℕ)
  (ruler_restock_factor : ℕ)
  (h1 : initial_pencils = 112)
  (h2 : initial_pens = 78)
  (h3 : initial_rulers = 46)
  (h4 : sold_pencils = 32)
  (h5 : sold_pens = 56)
  (h6 : given_rulers = 12)
  (h7 : pencil_restock_factor = 5)
  (h8 : ruler_restock_factor = 3)
  : (initial_pencils - sold_pencils + pencil_restock_factor * (initial_pencils - sold_pencils)) +
    (initial_rulers - given_rulers + ruler_restock_factor * (initial_rulers - given_rulers)) = 616 := by
  sorry

#check stationery_shop_restocking

end NUMINAMATH_CALUDE_stationery_shop_restocking_l1738_173847


namespace NUMINAMATH_CALUDE_original_number_is_two_l1738_173833

theorem original_number_is_two :
  ∃ (x : ℕ), 
    (∃ (y : ℕ), 
      (∀ (z : ℕ), z < y → ¬∃ (w : ℕ), x * z = w^3) ∧ 
      (∃ (w : ℕ), x * y = w^3) ∧
      x * y = 4 * x) →
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_two_l1738_173833


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1738_173874

theorem exponent_multiplication (a : ℝ) : a * a^3 = a^4 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1738_173874


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1738_173879

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 295 →
  train_speed_kmh = 75 →
  crossing_time = 45 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 642.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1738_173879


namespace NUMINAMATH_CALUDE_polynomial_with_geometric_zeros_l1738_173870

/-- A polynomial of the form x^4 + jx^2 + kx + 256 with four distinct real zeros in geometric progression has j = -32 -/
theorem polynomial_with_geometric_zeros (j k : ℝ) : 
  (∃ (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0), 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 256 = (x - a*r^3) * (x - a*r^2) * (x - a*r) * (x - a)) ∧ 
    (a*r^3 ≠ a*r^2) ∧ (a*r^2 ≠ a*r) ∧ (a*r ≠ a)) → 
  j = -32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_geometric_zeros_l1738_173870


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1738_173896

/-- The area of a square given its diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d ^ 2 / 2 : ℝ) = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1738_173896
