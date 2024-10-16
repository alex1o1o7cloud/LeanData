import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_dc_length_l3237_323725

theorem quadrilateral_dc_length
  (AB : ℝ) (sinA sinC : ℝ)
  (h1 : AB = 30)
  (h2 : sinA = 1/2)
  (h3 : sinC = 2/5)
  : ∃ (DC : ℝ), DC = 5 * Real.sqrt 47.25 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_dc_length_l3237_323725


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3237_323716

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2.5) : 
  x^2 + 1/x^2 = 4.25 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3237_323716


namespace NUMINAMATH_CALUDE_common_chord_circle_center_on_line_smallest_circle_l3237_323779

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define points A and B as the intersection of C₁ and C₂
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Theorem for the common chord
theorem common_chord : 
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
by sorry

-- Theorem for the circle with center on y = -x
theorem circle_center_on_line : 
  ∃ h k : ℝ, h = -k ∧ 
  (A.1 - h)^2 + (A.2 - k)^2 = (B.1 - h)^2 + (B.2 - k)^2 ∧
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

-- Theorem for the smallest circle
theorem smallest_circle : 
  ∀ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_circle_center_on_line_smallest_circle_l3237_323779


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3237_323766

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) :
  ∃ (min_dist : ℝ), min_dist = 120 / 17 ∧ 
    ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3237_323766


namespace NUMINAMATH_CALUDE_final_positions_l3237_323757

/-- Represents the positions of the cat -/
inductive CatPosition
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
| TopLeft
| TopMiddleRight
| MiddleRight
| BottomRight
| BottomMiddleLeft
| MiddleLeft

/-- The number of squares in the cat's cycle -/
def catCycleLength : Nat := 4

/-- The number of segments in the mouse's cycle -/
def mouseCycleLength : Nat := 6

/-- The total number of moves -/
def totalMoves : Nat := 320

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : Nat) : CatPosition := 
  match moves % catCycleLength with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.TopRight
  | 2 => CatPosition.BottomRight
  | _ => CatPosition.BottomLeft

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : Nat) : MousePosition := 
  match moves % mouseCycleLength with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddleRight
  | 2 => MousePosition.MiddleRight
  | 3 => MousePosition.BottomRight
  | 4 => MousePosition.BottomMiddleLeft
  | _ => MousePosition.MiddleLeft

theorem final_positions : 
  catPosition totalMoves = CatPosition.TopLeft ∧ 
  mousePosition totalMoves = MousePosition.MiddleRight := by
  sorry

end NUMINAMATH_CALUDE_final_positions_l3237_323757


namespace NUMINAMATH_CALUDE_hockey_league_games_l3237_323791

/-- Represents a hockey league with two divisions -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games played in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let intra_games := league.divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_games := league.divisions * league.teams_per_division * league.teams_per_division * league.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the total number of games in the described hockey league is 192 -/
theorem hockey_league_games :
  let league : HockeyLeague := {
    divisions := 2,
    teams_per_division := 6,
    intra_division_games := 4,
    inter_division_games := 2
  }
  total_games league = 192 := by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3237_323791


namespace NUMINAMATH_CALUDE_stating_fifteenth_term_is_43_l3237_323788

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- 
Theorem stating that the 15th term of the arithmetic sequence
with first term 1 and common difference 3 is 43.
-/
theorem fifteenth_term_is_43 : 
  arithmeticSequenceTerm 1 3 15 = 43 := by
sorry

end NUMINAMATH_CALUDE_stating_fifteenth_term_is_43_l3237_323788


namespace NUMINAMATH_CALUDE_twenty_five_percent_of_2004_l3237_323736

theorem twenty_five_percent_of_2004 : (25 : ℚ) / 100 * 2004 = 501 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_of_2004_l3237_323736


namespace NUMINAMATH_CALUDE_test_scores_l3237_323721

theorem test_scores (keith_score : Real) (larry_multiplier : Real) (danny_difference : Real)
  (h1 : keith_score = 3.5)
  (h2 : larry_multiplier = 3.2)
  (h3 : danny_difference = 5.7) :
  let larry_score := keith_score * larry_multiplier
  let danny_score := larry_score + danny_difference
  keith_score + larry_score + danny_score = 31.6 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l3237_323721


namespace NUMINAMATH_CALUDE_dimes_borrowed_proof_l3237_323703

/-- Represents the number of dimes Fred had initially -/
def initial_dimes : ℕ := 7

/-- Represents the number of dimes Fred has left -/
def remaining_dimes : ℕ := 4

/-- Represents the number of dimes Fred's sister borrowed -/
def borrowed_dimes : ℕ := initial_dimes - remaining_dimes

/-- Proves that the number of borrowed dimes is equal to the difference between
    the initial number of dimes and the remaining number of dimes -/
theorem dimes_borrowed_proof :
  borrowed_dimes = initial_dimes - remaining_dimes := by
  sorry

end NUMINAMATH_CALUDE_dimes_borrowed_proof_l3237_323703


namespace NUMINAMATH_CALUDE_greatest_y_value_l3237_323746

theorem greatest_y_value (y : ℝ) : 
  (3 * y^2 + 5 * y + 2 = 6) → 
  y ≤ (-5 + Real.sqrt 73) / 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_y_value_l3237_323746


namespace NUMINAMATH_CALUDE_snyder_cookies_l3237_323737

/-- Mrs. Snyder's cookie problem -/
theorem snyder_cookies (red_cookies pink_cookies : ℕ) 
  (h1 : red_cookies = 36)
  (h2 : pink_cookies = 50) :
  red_cookies + pink_cookies = 86 := by
  sorry

end NUMINAMATH_CALUDE_snyder_cookies_l3237_323737


namespace NUMINAMATH_CALUDE_next_shared_meeting_l3237_323749

/-- The number of days between meetings for the drama club -/
def drama_interval : ℕ := 3

/-- The number of days between meetings for the choir -/
def choir_interval : ℕ := 5

/-- The number of days between meetings for the debate team -/
def debate_interval : ℕ := 7

/-- The theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  Nat.lcm (Nat.lcm drama_interval choir_interval) debate_interval = 105 := by
  sorry

end NUMINAMATH_CALUDE_next_shared_meeting_l3237_323749


namespace NUMINAMATH_CALUDE_equilateral_center_triangles_properties_l3237_323774

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an equilateral triangle constructed on a side of another triangle -/
structure EquilateralTriangle where
  base : ℝ × ℝ
  apex : ℝ × ℝ

/-- The triangle formed by centers of equilateral triangles -/
def CenterTriangle (T : Triangle) (outward : Bool) : Triangle := sorry

/-- The centroid of a triangle -/
def centroid (T : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Main theorem about properties of triangles formed by centers of equilateral triangles -/
theorem equilateral_center_triangles_properties (T : Triangle) :
  let Δ := CenterTriangle T true
  let δ := CenterTriangle T false
  -- 1) Δ and δ are equilateral
  (∀ (X Y Z : ℝ × ℝ), (X = Δ.A ∧ Y = Δ.B ∧ Z = Δ.C) ∨ (X = δ.A ∧ Y = δ.B ∧ Z = δ.C) →
    dist X Y = dist Y Z ∧ dist Y Z = dist Z X) ∧
  -- 2) Centers of Δ and δ coincide with the centroid of T
  (centroid Δ = centroid T ∧ centroid δ = centroid T) ∧
  -- 3) Area(Δ) - Area(δ) = Area(T)
  (area Δ - area δ = area T) := by
  sorry


end NUMINAMATH_CALUDE_equilateral_center_triangles_properties_l3237_323774


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l3237_323763

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (xyz : ℕ × ℕ × ℕ) => 
    xyz.1 + xyz.2.1 + xyz.2.2 = 12 ∧ 
    xyz.1 > 0 ∧ xyz.2.1 > 0 ∧ xyz.2.2 > 0) 
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card = 55 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l3237_323763


namespace NUMINAMATH_CALUDE_soap_brand_usage_l3237_323709

/-- Given a survey of households and their soap brand usage, prove the number of households using both brands. -/
theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_a : ℕ) (both_to_only_b_ratio : ℕ) 
  (h_total : total = 240)
  (h_neither : neither = 80)
  (h_only_a : only_a = 60)
  (h_ratio : both_to_only_b_ratio = 3) :
  ∃ (both : ℕ), both = 25 ∧ total = neither + only_a + both_to_only_b_ratio * both + both :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l3237_323709


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l3237_323780

theorem wall_painting_fraction (total_time minutes : ℕ) (fraction : ℚ) : 
  total_time = 60 → 
  minutes = 12 → 
  fraction = minutes / total_time → 
  fraction = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l3237_323780


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l3237_323754

theorem least_n_for_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 15) := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l3237_323754


namespace NUMINAMATH_CALUDE_brownie_pieces_l3237_323772

/-- Proves that a 24-inch by 15-inch pan can be divided into exactly 40 pieces of 3-inch by 3-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_size : ℕ) : 
  pan_length = 24 → pan_width = 15 → piece_size = 3 → 
  (pan_length * pan_width) / (piece_size * piece_size) = 40 := by
  sorry

#check brownie_pieces

end NUMINAMATH_CALUDE_brownie_pieces_l3237_323772


namespace NUMINAMATH_CALUDE_sin_x_equals_x_unique_root_l3237_323731

theorem sin_x_equals_x_unique_root :
  ∃! x : ℝ, x ∈ Set.Icc (-π) π ∧ x = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_x_equals_x_unique_root_l3237_323731


namespace NUMINAMATH_CALUDE_exist_unit_tetrahedron_with_interior_point_l3237_323707

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the volume of a tetrahedron given four points -/
def tetrahedronVolume (p1 p2 p3 p4 : Point3D) : ℝ := sorry

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if a point is inside a tetrahedron -/
def isInsideTetrahedron (p : Point3D) (t1 t2 t3 t4 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem exist_unit_tetrahedron_with_interior_point 
  (n : ℕ) 
  (points : Fin n → Point3D) 
  (h_not_coplanar : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → ¬areCoplanar (points i) (points j) (points k) (points l))
  (h_max_volume : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → tetrahedronVolume (points i) (points j) (points k) (points l) ≤ 0.037)
  : ∃ (t1 t2 t3 t4 : Point3D), 
    tetrahedronVolume t1 t2 t3 t4 = 1 ∧ 
    ∃ (i : Fin n), isInsideTetrahedron (points i) t1 t2 t3 t4 := by
  sorry

end NUMINAMATH_CALUDE_exist_unit_tetrahedron_with_interior_point_l3237_323707


namespace NUMINAMATH_CALUDE_triangle_problem_l3237_323735

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 6 →
  Real.cos A = -1/3 →
  (c = 2 ∧ Real.cos (2 * B - π/4) = (4 - Real.sqrt 2) / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3237_323735


namespace NUMINAMATH_CALUDE_flowerbed_fraction_is_one_eighth_l3237_323719

/-- Represents a rectangular park with flower beds -/
structure Park where
  /-- Length of the shorter parallel side of the trapezoidal area -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal area -/
  long_side : ℝ
  /-- Number of congruent isosceles right triangle flower beds -/
  num_flowerbeds : ℕ

/-- The fraction of the park occupied by flower beds -/
def flowerbed_fraction (p : Park) : ℝ :=
  -- Define the fraction calculation here
  sorry

/-- Theorem stating that for a park with specific dimensions, 
    the fraction of area occupied by flower beds is 1/8 -/
theorem flowerbed_fraction_is_one_eighth :
  ∀ (p : Park), 
  p.short_side = 30 ∧ 
  p.long_side = 50 ∧ 
  p.num_flowerbeds = 3 →
  flowerbed_fraction p = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_flowerbed_fraction_is_one_eighth_l3237_323719


namespace NUMINAMATH_CALUDE_complex_number_properties_l3237_323730

/-- For a real number m and a complex number z = (m^2 - 5m + 6) + (m^2 - 3m)i, we define the following properties --/

def is_real (m : ℝ) : Prop := m^2 - 3*m = 0

def is_complex (m : ℝ) : Prop := m^2 - 3*m ≠ 0

def is_purely_imaginary (m : ℝ) : Prop := m^2 - 5*m + 6 = 0 ∧ m^2 - 3*m ≠ 0

def is_in_third_quadrant (m : ℝ) : Prop := m^2 - 5*m + 6 < 0 ∧ m^2 - 3*m < 0

/-- Main theorem stating the conditions for each case --/
theorem complex_number_properties (m : ℝ) :
  (is_real m ↔ (m = 0 ∨ m = 3)) ∧
  (is_complex m ↔ (m ≠ 0 ∧ m ≠ 3)) ∧
  (is_purely_imaginary m ↔ m = 2) ∧
  (is_in_third_quadrant m ↔ (2 < m ∧ m < 3)) :=
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3237_323730


namespace NUMINAMATH_CALUDE_mary_flour_amount_l3237_323761

/-- The amount of flour Mary uses in her cake recipe -/
def flour_recipe : ℝ := 7.0

/-- The extra amount of flour Mary adds -/
def flour_extra : ℝ := 2.0

/-- The total amount of flour Mary uses -/
def flour_total : ℝ := flour_recipe + flour_extra

theorem mary_flour_amount : flour_total = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l3237_323761


namespace NUMINAMATH_CALUDE_number_line_order_l3237_323755

theorem number_line_order (x y : ℝ) : x > y ↔ (∃ (d : ℝ), d > 0 ∧ x = y + d) :=
sorry

end NUMINAMATH_CALUDE_number_line_order_l3237_323755


namespace NUMINAMATH_CALUDE_remainder_theorem_l3237_323714

def P (x : ℝ) : ℝ := x^100 - x^99 + x^98 - x^97 + x^96 - x^95 + x^94 - x^93 + x^92 - x^91 + x^90 - x^89 + x^88 - x^87 + x^86 - x^85 + x^84 - x^83 + x^82 - x^81 + x^80 - x^79 + x^78 - x^77 + x^76 - x^75 + x^74 - x^73 + x^72 - x^71 + x^70 - x^69 + x^68 - x^67 + x^66 - x^65 + x^64 - x^63 + x^62 - x^61 + x^60 - x^59 + x^58 - x^57 + x^56 - x^55 + x^54 - x^53 + x^52 - x^51 + x^50 - x^49 + x^48 - x^47 + x^46 - x^45 + x^44 - x^43 + x^42 - x^41 + x^40 - x^39 + x^38 - x^37 + x^36 - x^35 + x^34 - x^33 + x^32 - x^31 + x^30 - x^29 + x^28 - x^27 + x^26 - x^25 + x^24 - x^23 + x^22 - x^21 + x^20 - x^19 + x^18 - x^17 + x^16 - x^15 + x^14 - x^13 + x^12 - x^11 + x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_theorem (a b : ℝ) : 
  (∃ Q : ℝ → ℝ, ∀ x, P x = Q x * (x^2 - 1) + a * x + b) → 
  2 * a + b = -49 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3237_323714


namespace NUMINAMATH_CALUDE_period_of_sin_plus_cos_l3237_323743

/-- The period of the function y = 3sin(x) + 3cos(x) is 2π -/
theorem period_of_sin_plus_cos : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 3 * Real.cos x
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_sin_plus_cos_l3237_323743


namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l3237_323739

/-- Given a triangle ABC with median CM and angle bisector BL, prove that the coordinates of C are (14, 2) -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) : 
  A = (2, 8) → M = (4, 11) → L = (6, 6) → 
  ∃ (B C : ℝ × ℝ), 
    (M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2) ∧  -- M is midpoint of AC
    (∃ (t : ℝ), L = B + t • (A - C)) ∧                 -- L is on angle bisector BL
    C = (14, 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_point_coordinates_l3237_323739


namespace NUMINAMATH_CALUDE_rectangle_inside_circle_l3237_323717

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the unit circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  unit_circle p.1 p.2

-- Define a point inside the unit circle
def point_inside_circle (q : ℝ × ℝ) : Prop :=
  q.1^2 + q.2^2 < 1

-- Define the rectangle with diagonal pq and sides parallel to axes
def rectangle_with_diagonal (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | p.1 ≤ r.1 ∧ r.1 ≤ q.1 ∧ q.2 ≤ r.2 ∧ r.2 ≤ p.2 ∨
       q.1 ≤ r.1 ∧ r.1 ≤ p.1 ∧ p.2 ≤ r.2 ∧ r.2 ≤ q.2}

-- Theorem statement
theorem rectangle_inside_circle (p q : ℝ × ℝ) :
  point_on_circle p → point_inside_circle q →
  ∀ r ∈ rectangle_with_diagonal p q, r.1^2 + r.2^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_inside_circle_l3237_323717


namespace NUMINAMATH_CALUDE_second_train_length_calculation_l3237_323799

/-- Calculates the length of the second train given the speeds of two trains,
    the length of the first train, and the time they take to clear each other. -/
def second_train_length (speed1 speed2 : ℝ) (length1 time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let distance := relative_speed * time
  distance - length1

theorem second_train_length_calculation :
  let speed1 := 42 * (1000 / 3600)  -- Convert 42 kmph to m/s
  let speed2 := 30 * (1000 / 3600)  -- Convert 30 kmph to m/s
  let length1 := 200
  let time := 23.998
  abs (second_train_length speed1 speed2 length1 time - 279.96) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_second_train_length_calculation_l3237_323799


namespace NUMINAMATH_CALUDE_boys_speed_l3237_323793

/-- The speed of a boy traveling from home to school on the first day, given certain conditions. -/
theorem boys_speed (distance : ℝ) (late_time : ℝ) (early_time : ℝ) (second_day_speed : ℝ) : 
  distance = 2.5 ∧ 
  late_time = 7 / 60 ∧ 
  early_time = 8 / 60 ∧ 
  second_day_speed = 10 → 
  ∃ (first_day_speed : ℝ), first_day_speed = 9.375 := by
  sorry

#eval (9.375 : Float)

end NUMINAMATH_CALUDE_boys_speed_l3237_323793


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_seven_l3237_323773

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + 2

-- State the theorem
theorem monotone_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a y < f a x) →
  a ≤ -7 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_seven_l3237_323773


namespace NUMINAMATH_CALUDE_order_of_magnitude_l3237_323759

theorem order_of_magnitude (a b c x y z : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (hx : x = Real.sqrt (a^2 + (b+c)^2))
  (hy : y = Real.sqrt (b^2 + (c+a)^2))
  (hz : z = Real.sqrt (c^2 + (a+b)^2)) :
  z > y ∧ y > x := by
  sorry

#check order_of_magnitude

end NUMINAMATH_CALUDE_order_of_magnitude_l3237_323759


namespace NUMINAMATH_CALUDE_carrot_distribution_l3237_323752

theorem carrot_distribution (total_carrots : ℕ) (num_goats : ℕ) 
  (h1 : total_carrots = 47) (h2 : num_goats = 4) : 
  total_carrots % num_goats = 3 := by
  sorry

end NUMINAMATH_CALUDE_carrot_distribution_l3237_323752


namespace NUMINAMATH_CALUDE_range_of_a_l3237_323777

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x ≤ a}
  let B := Set.Iio 2
  A ⊆ B → a < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3237_323777


namespace NUMINAMATH_CALUDE_puppy_weight_l3237_323784

/-- Given the weights of a puppy and two cats satisfying certain conditions,
    prove that the puppy weighs 12 pounds. -/
theorem puppy_weight (a b c : ℝ) 
    (h1 : a + b + c = 36)
    (h2 : a + c = 3 * b)
    (h3 : a + b = c + 6) :
    a = 12 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l3237_323784


namespace NUMINAMATH_CALUDE_problem_statement_l3237_323783

theorem problem_statement (x y : ℝ) 
  (hx : x > 4) 
  (hy : y > 9) 
  (h : (Real.log x / Real.log 4)^4 + (Real.log y / Real.log 9)^4 + 18 = 18 * (Real.log x / Real.log 4) * (Real.log y / Real.log 9)) : 
  x^2 + y^2 = 4^(2 * Real.sqrt 3) + 9^(2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3237_323783


namespace NUMINAMATH_CALUDE_second_plant_production_l3237_323744

/-- Represents the production of tomatoes from three plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def TomatoProblem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.third = p.second + 2 ∧
  p.first + p.second + p.third = 60

theorem second_plant_production (p : TomatoProduction) 
  (h : TomatoProblem p) : p.first - p.second = 7 :=
by
  sorry

#check second_plant_production

end NUMINAMATH_CALUDE_second_plant_production_l3237_323744


namespace NUMINAMATH_CALUDE_inconsistent_extension_system_l3237_323771

/-- Represents a 4-digit extension number -/
structure Extension :=
  (digits : Fin 4 → Nat)
  (valid : ∀ i, digits i < 10)
  (even : digits 3 % 2 = 0)

/-- The set of 4 specific digits used for extensions -/
def SpecificDigits : Finset Nat := sorry

/-- The set of all valid extensions -/
def AllExtensions : Finset Extension :=
  sorry

theorem inconsistent_extension_system :
  (∀ e ∈ AllExtensions, (∀ i, e.digits i ∈ SpecificDigits)) →
  (Finset.card AllExtensions = 12) →
  False :=
sorry

end NUMINAMATH_CALUDE_inconsistent_extension_system_l3237_323771


namespace NUMINAMATH_CALUDE_box_content_theorem_l3237_323747

theorem box_content_theorem (total : ℕ) (pencil : ℕ) (pen : ℕ) (both : ℕ) :
  total = 12 →
  pencil = 7 →
  pen = 4 →
  both = 3 →
  total - (pencil + pen - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_box_content_theorem_l3237_323747


namespace NUMINAMATH_CALUDE_monkey_peach_division_l3237_323733

theorem monkey_peach_division (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, 100 = n * k + 10) →
  (∃ m : ℕ, 1000 = n * m + 10) :=
by sorry

end NUMINAMATH_CALUDE_monkey_peach_division_l3237_323733


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_cylinder_l3237_323710

/-- The volume of a cone with an inscribed cylinder -/
def cone_volume (cylinder_volume : ℝ) (truncated_cone_volume : ℝ) : ℝ :=
  94.5

/-- Theorem stating the volume of the cone given the conditions -/
theorem cone_volume_with_inscribed_cylinder
  (cylinder_volume : ℝ)
  (truncated_cone_volume : ℝ)
  (h1 : cylinder_volume = 21)
  (h2 : truncated_cone_volume = 91) :
  cone_volume cylinder_volume truncated_cone_volume = 94.5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_cylinder_l3237_323710


namespace NUMINAMATH_CALUDE_circle_radius_l3237_323769

/-- Given a circle centered at (0,k) with k > 4, which is tangent to the lines y=x, y=-x, and y=4,
    the radius of the circle is 4(1+√2). -/
theorem circle_radius (k : ℝ) (h1 : k > 4) : ∃ r : ℝ,
  (∀ x y : ℝ, (x = y ∨ x = -y ∨ y = 4) → (x^2 + (y - k)^2 = r^2)) ∧
  r = 4*(1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3237_323769


namespace NUMINAMATH_CALUDE_odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3237_323740

theorem odd_divisors_of_power_minus_one (n : ℕ) :
  Odd n → n ∣ 2023^n - 1 → n ≥ 3 :=
sorry

theorem smallest_odd_divisors :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ n < 3) → False :=
sorry

theorem second_smallest_odd_divisor :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ 3 < n ∧ n < 9) → False :=
sorry

theorem three_divides : 3 ∣ 2023^3 - 1 :=
sorry

theorem nine_divides : 9 ∣ 2023^9 - 1 :=
sorry

theorem infinitely_many_divisors (k : ℕ) :
  k ≥ 1 → 3^k ∣ 2023^(3^k) - 1 :=
sorry

end NUMINAMATH_CALUDE_odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3237_323740


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3237_323702

theorem complex_product_magnitude (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 1 - I) : 
  Complex.abs (z₁ * z₂) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3237_323702


namespace NUMINAMATH_CALUDE_group_average_l3237_323762

theorem group_average (x : ℝ) : 
  (5 + 5 + x + 6 + 8) / 5 = 6 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_group_average_l3237_323762


namespace NUMINAMATH_CALUDE_power_function_properties_l3237_323729

def f (m n : ℕ+) (x : ℝ) : ℝ := x ^ (m.val / n.val)

theorem power_function_properties (m n : ℕ+) (h_coprime : Nat.Coprime m.val n.val) :
  (∀ x, m.val % 2 = 1 ∧ n.val % 2 = 1 → f m n (-x) = -f m n x) ∧
  (∀ x, m.val % 2 = 0 ∧ n.val % 2 = 1 → f m n (-x) = f m n x) :=
sorry

end NUMINAMATH_CALUDE_power_function_properties_l3237_323729


namespace NUMINAMATH_CALUDE_inequality_solution_l3237_323723

theorem inequality_solution (a : ℝ) (ha : a > 0) (ha_neq_1 : a ≠ 1) :
  (∀ x : ℝ, a^(x + 5) < a^(4*x - 1) ↔ (0 < a ∧ a < 1 ∧ x < 2) ∨ (a > 1 ∧ x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3237_323723


namespace NUMINAMATH_CALUDE_factory_sampling_probability_l3237_323787

/-- Represents a district with a number of factories -/
structure District where
  name : String
  factories : ℕ

/-- Represents the sampling result -/
structure SamplingResult where
  districtA : ℕ
  districtB : ℕ
  districtC : ℕ

/-- The stratified sampling function -/
def stratifiedSampling (districts : List District) (totalSample : ℕ) : SamplingResult :=
  sorry

/-- The probability calculation function -/
def probabilityAtLeastOneFromC (sample : SamplingResult) : ℚ :=
  sorry

theorem factory_sampling_probability :
  let districts := [
    { name := "A", factories := 9 },
    { name := "B", factories := 18 },
    { name := "C", factories := 18 }
  ]
  let sample := stratifiedSampling districts 5
  sample.districtA = 1 ∧
  sample.districtB = 2 ∧
  sample.districtC = 2 ∧
  probabilityAtLeastOneFromC sample = 7/10 :=
by sorry

end NUMINAMATH_CALUDE_factory_sampling_probability_l3237_323787


namespace NUMINAMATH_CALUDE_gambler_final_amount_l3237_323742

def bet_sequence := [true, false, true, false, false, true, false, true]

def apply_bet (current_amount : ℚ) (is_win : Bool) : ℚ :=
  if is_win then
    current_amount + (current_amount / 2)
  else
    current_amount / 2

def final_amount (initial_amount : ℚ) (bets : List Bool) : ℚ :=
  bets.foldl apply_bet initial_amount

theorem gambler_final_amount :
  final_amount 128 bet_sequence = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_gambler_final_amount_l3237_323742


namespace NUMINAMATH_CALUDE_total_pools_count_l3237_323748

/-- The number of stores operated by Pat's Pool Supply -/
def pool_supply_stores : ℕ := 4

/-- The number of stores operated by Pat's Ark & Athletic Wear -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of swimming pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 3

/-- The number of pools in one Pat's Ark & Athletic Wear store -/
def pools_per_ark_athletic : ℕ := 200

/-- The total number of swimming pools across all Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def total_pools : ℕ := pool_supply_stores * pool_ratio * pools_per_ark_athletic + ark_athletic_stores * pools_per_ark_athletic

theorem total_pools_count : total_pools = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_pools_count_l3237_323748


namespace NUMINAMATH_CALUDE_max_value_x_minus_x_squared_l3237_323728

theorem max_value_x_minus_x_squared (f : ℝ → ℝ) (h : ∀ x, 0 < x → x < 1 → f x = x * (1 - x)) :
  ∃ m : ℝ, m = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_x_squared_l3237_323728


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3237_323738

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3237_323738


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3237_323768

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3237_323768


namespace NUMINAMATH_CALUDE_one_fourth_of_six_point_eight_l3237_323751

theorem one_fourth_of_six_point_eight : (1 / 4 : ℚ) * (68 / 10 : ℚ) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_point_eight_l3237_323751


namespace NUMINAMATH_CALUDE_tangent_secant_theorem_l3237_323726

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is tangent to a circle -/
def is_tangent (p q : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is a secant of a circle -/
def is_secant (p q r : Point) (c : Circle) : Prop := sorry

theorem tangent_secant_theorem (C : Circle) (Q U M N : Point) :
  is_outside Q C →
  is_tangent Q U C →
  is_secant Q M N C →
  distance Q M < distance Q N →
  distance Q M = 4 →
  distance Q U = distance M N - distance Q M →
  distance Q N = 16 := by sorry

end NUMINAMATH_CALUDE_tangent_secant_theorem_l3237_323726


namespace NUMINAMATH_CALUDE_circle_equation_l3237_323782

/-- The standard equation of a circle with center (-1, 2) passing through (2, -2) -/
theorem circle_equation : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 25 ↔ 
  ((x + 1)^2 + (y - 2)^2 = ((2 + 1)^2 + (-2 - 2)^2) ∧ 
   (x, y) ≠ (-1, 2)) := by sorry

end NUMINAMATH_CALUDE_circle_equation_l3237_323782


namespace NUMINAMATH_CALUDE_twins_age_problem_l3237_323722

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 17 → age = 8 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l3237_323722


namespace NUMINAMATH_CALUDE_power_multiplication_l3237_323790

theorem power_multiplication (t : ℝ) : t^5 * t^2 = t^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3237_323790


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3237_323705

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond 5 x = 12 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3237_323705


namespace NUMINAMATH_CALUDE_percentage_watching_two_shows_l3237_323798

def total_residents : ℕ := 600
def watch_island_survival : ℕ := (35 * total_residents) / 100
def watch_lovelost_lawyers : ℕ := (40 * total_residents) / 100
def watch_medical_emergency : ℕ := (50 * total_residents) / 100
def watch_all_three : ℕ := 21

theorem percentage_watching_two_shows :
  let watch_two_shows := watch_island_survival + watch_lovelost_lawyers + watch_medical_emergency - total_residents + watch_all_three
  (watch_two_shows : ℚ) / total_residents * 100 = 285 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_watching_two_shows_l3237_323798


namespace NUMINAMATH_CALUDE_inequality_theorem_l3237_323794

theorem inequality_theorem (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ 2*(x*y - x + y) ∧ 
  (x^2 + y^2 + 1 = 2*(x*y - x + y) ↔ x = y - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3237_323794


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3237_323760

theorem quadratic_one_solution (c : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + c = 0) ↔ c = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3237_323760


namespace NUMINAMATH_CALUDE_units_digit_product_zero_exists_l3237_323701

theorem units_digit_product_zero_exists : ∃ (a b : ℕ), 
  (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) ∧ ((a * b) % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_zero_exists_l3237_323701


namespace NUMINAMATH_CALUDE_tony_midpoint_age_l3237_323750

/-- Represents Tony's age and earnings over a 60-day period --/
structure TonyEarnings where
  daysWorked : Nat
  hoursPerDay : Nat
  hourlyRateMultiplier : Rat
  startAge : Nat
  midAge : Nat
  endAge : Nat
  totalEarnings : Rat

/-- Calculates Tony's earnings based on his age and work details --/
def calculateEarnings (t : TonyEarnings) : Rat :=
  let firstHalfDays := t.daysWorked / 2
  let secondHalfDays := t.daysWorked - firstHalfDays
  (t.hoursPerDay * t.hourlyRateMultiplier * t.startAge * firstHalfDays : Rat) +
  (t.hoursPerDay * t.hourlyRateMultiplier * t.endAge * secondHalfDays : Rat)

/-- Theorem stating that Tony's age at the midpoint was 11 --/
theorem tony_midpoint_age (t : TonyEarnings) 
  (h1 : t.daysWorked = 60)
  (h2 : t.hoursPerDay = 3)
  (h3 : t.hourlyRateMultiplier = 3/4)
  (h4 : t.startAge = 10)
  (h5 : t.endAge = 12)
  (h6 : t.totalEarnings = 1125)
  (h7 : calculateEarnings t = t.totalEarnings) :
  t.midAge = 11 := by
  sorry

end NUMINAMATH_CALUDE_tony_midpoint_age_l3237_323750


namespace NUMINAMATH_CALUDE_first_half_speed_calculation_l3237_323756

/-- Represents a trip with two halves -/
structure Trip where
  total_distance : ℝ
  first_half_speed : ℝ
  second_half_time_multiplier : ℝ
  average_speed : ℝ

/-- Calculates the speed of the first half of the trip -/
def first_half_speed (t : Trip) : ℝ :=
  2 * t.average_speed

/-- Theorem stating the conditions and the result to be proved -/
theorem first_half_speed_calculation (t : Trip)
  (h1 : t.total_distance = 640)
  (h2 : t.second_half_time_multiplier = 3)
  (h3 : t.average_speed = 40) :
  first_half_speed t = 80 := by
  sorry

#eval first_half_speed { total_distance := 640, first_half_speed := 0, second_half_time_multiplier := 3, average_speed := 40 }

end NUMINAMATH_CALUDE_first_half_speed_calculation_l3237_323756


namespace NUMINAMATH_CALUDE_certain_value_proof_l3237_323795

theorem certain_value_proof (n : ℤ) (x : ℤ) 
  (h1 : 101 * n^2 ≤ x)
  (h2 : ∀ m : ℤ, 101 * m^2 ≤ x → m ≤ 7) :
  x = 4979 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l3237_323795


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3237_323758

theorem fraction_equals_zero (x : ℝ) : x = 5 → (x - 5) / (6 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3237_323758


namespace NUMINAMATH_CALUDE_parallel_lines_circle_solution_l3237_323770

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords formed by the intersection -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are formed by equally spaced parallel lines -/
  parallel_lines : chord1 = chord3
  /-- The given chord lengths -/
  chord_lengths : chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 40

/-- The theorem stating the distance between lines and radius of the circle -/
theorem parallel_lines_circle_solution (c : ParallelLinesCircle) :
  c.d = Real.sqrt 1188 ∧ c.r = Real.sqrt 357 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_solution_l3237_323770


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3237_323767

/-- Given a geometric sequence {a_n} where all terms are positive and 
    (a₁, ½a₃, 2a₂) forms an arithmetic sequence, 
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
  (a 1 + 2 * a 2 = a 3) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3237_323767


namespace NUMINAMATH_CALUDE_intersectionRangeOfB_l3237_323796

/-- Two lines y = 2x + 1 and y = 3x + b intersect in the third quadrant -/
def linesIntersectInThirdQuadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0

/-- The range of b for which the lines intersect in the third quadrant -/
theorem intersectionRangeOfB :
  ∀ b : ℝ, linesIntersectInThirdQuadrant b ↔ b > 3/2 :=
sorry

end NUMINAMATH_CALUDE_intersectionRangeOfB_l3237_323796


namespace NUMINAMATH_CALUDE_total_children_l3237_323712

/-- Given a group of children where:
    k children are initially selected and given an apple,
    m children are selected later,
    n of the m children had previously received an apple,
    prove that the total number of children is k * (m/n) -/
theorem total_children (k m n : ℕ) (h : n ≤ m) (h' : n > 0) :
  ∃ (total : ℚ), total = k * (m / n) := by
  sorry

end NUMINAMATH_CALUDE_total_children_l3237_323712


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l3237_323704

theorem quadratic_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 25 = 0 ↔ (x + a)^2 = b) → a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l3237_323704


namespace NUMINAMATH_CALUDE_sticks_difference_l3237_323792

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14) 
  (h2 : left = 4) : 
  picked_up - left = 10 := by
sorry

end NUMINAMATH_CALUDE_sticks_difference_l3237_323792


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3237_323797

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (x % 2 = 1) →                   -- x is odd
  ((x + 2) % 2 = 1) →             -- x + 2 is the next consecutive odd integer
  (x + 2 = 5 * x - 2) →           -- The larger is five times the smaller minus two
  (x + (x + 2) = 4) :=            -- Their sum is 4
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3237_323797


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3237_323713

theorem quadratic_equation_solutions : ∀ x : ℝ,
  (2 * x^2 + 7 * x - 1 = 4 * x + 1 ↔ x = -2 ∨ x = 1/2) ∧
  (2 * x^2 + 7 * x - 1 = -(x^2 - 19) ↔ x = -4 ∨ x = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3237_323713


namespace NUMINAMATH_CALUDE_na_mass_percentage_in_bleach_l3237_323711

/-- The mass percentage of Na in bleach with given composition -/
theorem na_mass_percentage_in_bleach (na_mass : ℝ) (cl_mass : ℝ) (o_mass : ℝ) 
  (naclo_concentration : ℝ) :
  na_mass = 22.99 →
  cl_mass = 35.45 →
  o_mass = 16.00 →
  naclo_concentration = 0.05 →
  let naclo_mass := na_mass + cl_mass + o_mass
  let na_percentage_in_naclo := na_mass / naclo_mass
  let na_percentage_in_bleach := na_percentage_in_naclo * naclo_concentration
  abs (na_percentage_in_bleach - 0.015445) < 0.000001 := by
  sorry

#check na_mass_percentage_in_bleach

end NUMINAMATH_CALUDE_na_mass_percentage_in_bleach_l3237_323711


namespace NUMINAMATH_CALUDE_hike_weight_after_six_hours_l3237_323765

/-- Calculates the remaining weight after a hike given initial weights and consumption rates -/
def remaining_weight (initial_water : ℝ) (initial_food : ℝ) (initial_gear : ℝ) 
                     (water_rate : ℝ) (food_rate : ℝ) (hours : ℝ) : ℝ :=
  let remaining_water := initial_water - water_rate * hours
  let remaining_food := initial_food - food_rate * hours
  remaining_water + remaining_food + initial_gear

/-- Theorem: The remaining weight after 6 hours of hiking is 34 pounds -/
theorem hike_weight_after_six_hours :
  remaining_weight 20 10 20 2 (2/3) 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_hike_weight_after_six_hours_l3237_323765


namespace NUMINAMATH_CALUDE_greatest_area_difference_l3237_323745

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_condition : length * 2 + width * 2 = 160

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the greatest possible difference between areas of two such rectangles -/
theorem greatest_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    (area r1 - area r2 : ℤ) ≥ (area s1 - area s2 : ℤ) ∧
    (area r1 - area r2 : ℕ) = 1521 := by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l3237_323745


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3237_323708

/-- The function f(x) = x^3 - ax^2 - bx, where f(1) = 0 -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x

/-- The condition that f(1) = 0 -/
def f_intersects_x_axis (a b : ℝ) : Prop := f a b 1 = 0

theorem extreme_values_of_f (a b : ℝ) (h : f_intersects_x_axis a b) :
  (∃ x, f a b x = 4/27) ∧ (∃ x, f a b x = 0) ∧
  (∀ x, f a b x ≤ 4/27) ∧ (∀ x, f a b x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3237_323708


namespace NUMINAMATH_CALUDE_good_number_properties_l3237_323753

def is_good (n : ℕ) : Prop := n % 6 = 3

theorem good_number_properties :
  (∀ n : ℕ, is_good n ↔ n % 6 = 3) ∧
  (is_good 2001 ∧ ¬is_good 3001) ∧
  (∀ a b : ℕ, is_good a → is_good b → is_good (a * b)) ∧
  (∀ a b : ℕ, is_good (a * b) → is_good a ∨ is_good b) :=
by sorry

end NUMINAMATH_CALUDE_good_number_properties_l3237_323753


namespace NUMINAMATH_CALUDE_triangle_inequality_l3237_323789

theorem triangle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) : 
  (Real.sqrt (Real.sin A * Real.sin B) / Real.sin (C / 2)) + 
  (Real.sqrt (Real.sin B * Real.sin C) / Real.sin (A / 2)) + 
  (Real.sqrt (Real.sin C * Real.sin A) / Real.sin (B / 2)) ≥ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3237_323789


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3237_323778

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 5 = 11 ∧ a 12 = 31 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : 
  a 1 = -2 ∧ 
  (∀ n : ℕ, a (n + 1) - a n = 3) ∧
  a 20 = 55 ∧
  (∀ n : ℕ, a n = 3 * n - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3237_323778


namespace NUMINAMATH_CALUDE_biology_score_is_85_l3237_323775

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def average_score : ℕ := 71
def total_subjects : ℕ := 5

def biology_score : ℕ := 
  average_score * total_subjects - (mathematics_score + science_score + social_studies_score + english_score)

theorem biology_score_is_85 : biology_score = 85 := by sorry

end NUMINAMATH_CALUDE_biology_score_is_85_l3237_323775


namespace NUMINAMATH_CALUDE_derek_journey_l3237_323718

/-- Proves that given a journey where half the distance is traveled at 20 km/h 
    and the other half at 4 km/h, with a total travel time of 54 minutes, 
    the distance walked is 3.0 km. -/
theorem derek_journey (total_distance : ℝ) (total_time : ℝ) : 
  (total_distance / 2) / 20 + (total_distance / 2) / 4 = total_time ∧
  total_time = 54 / 60 →
  total_distance / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_derek_journey_l3237_323718


namespace NUMINAMATH_CALUDE_polygon_sides_l3237_323724

theorem polygon_sides (n : ℕ) (missing_angle : ℝ) : 
  (n ≥ 3) →
  (missing_angle < 170) →
  ((n - 2) * 180 - missing_angle = 2970) →
  n = 19 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3237_323724


namespace NUMINAMATH_CALUDE_triangle_inequality_bound_l3237_323781

theorem triangle_inequality_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / (b + c)^2 ≤ (1 : ℝ) / 2 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + c'^2) / (b' + c')^2 > (1 : ℝ) / 2 - ε :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_l3237_323781


namespace NUMINAMATH_CALUDE_unique_solution_l3237_323786

def complex_number (a : ℝ) : ℂ := Complex.mk (a^2 - 2) (3*a - 4)

theorem unique_solution :
  ∃! a : ℝ,
    (complex_number a).re = (complex_number a).im ∧
    (complex_number a).re < 0 ∧
    (complex_number a).im < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3237_323786


namespace NUMINAMATH_CALUDE_intersection_A_B_l3237_323741

def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3237_323741


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_in_open_unit_interval_l3237_323734

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 3) - x + 2 * a

theorem two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (a > 0) →
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_in_open_unit_interval_l3237_323734


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3237_323727

/-- Given two quadratic functions f and g, if the sum of roots of f equals the product of roots of g,
    and the product of roots of f equals the sum of roots of g, then f attains its minimum at x = 3 -/
theorem quadratic_minimum (r s : ℝ) : 
  let f (x : ℝ) := x^2 + r*x + s
  let g (x : ℝ) := x^2 - 9*x + 6
  let sum_roots_f := -r
  let prod_roots_f := s
  let sum_roots_g := 9
  let prod_roots_g := 6
  (sum_roots_f = prod_roots_g) → (prod_roots_f = sum_roots_g) →
  ∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), f x ≥ f a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3237_323727


namespace NUMINAMATH_CALUDE_prob_missing_one_equals_two_prob_decreasing_sequence_l3237_323706

-- Define the number of items in the collection
def n : ℕ := 10

-- Define the probability of finding each item
def p : ℝ := 0.1

-- Define the probability of missing exactly k items in the second set
-- when the first set is completed
noncomputable def p_k (k : ℕ) : ℝ := sorry

-- Theorem 1: p_1 = p_2
theorem prob_missing_one_equals_two : p_k 1 = p_k 2 := sorry

-- Theorem 2: p_2 > p_3 > p_4 > ... > p_10
theorem prob_decreasing_sequence : 
  ∀ k₁ k₂ : ℕ, 2 ≤ k₁ → k₁ < k₂ → k₂ ≤ n → p_k k₁ > p_k k₂ := sorry

end NUMINAMATH_CALUDE_prob_missing_one_equals_two_prob_decreasing_sequence_l3237_323706


namespace NUMINAMATH_CALUDE_black_chess_pieces_count_l3237_323732

theorem black_chess_pieces_count 
  (white_pieces : ℕ) 
  (white_probability : ℚ) 
  (h1 : white_pieces = 9)
  (h2 : white_probability = 3/10) : 
  ∃ (black_pieces : ℕ), 
    (white_pieces : ℚ) / (white_pieces + black_pieces) = white_probability ∧ 
    black_pieces = 21 := by
  sorry

end NUMINAMATH_CALUDE_black_chess_pieces_count_l3237_323732


namespace NUMINAMATH_CALUDE_probability_white_ball_l3237_323700

def num_white_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_red_balls : ℕ := 5

def total_balls : ℕ := num_white_balls + num_black_balls + num_red_balls

theorem probability_white_ball :
  (num_white_balls : ℚ) / (total_balls : ℚ) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_white_ball_l3237_323700


namespace NUMINAMATH_CALUDE_largest_circle_radius_l3237_323764

/-- Represents a chessboard square --/
structure Square where
  x : Nat
  y :Nat
  isWhite : Bool

/-- Represents a chessboard --/
def Chessboard := List Square

/-- Creates an 8x8 chessboard with alternating white and black squares --/
def createChessboard : Chessboard :=
  sorry

/-- Checks if a given point (x, y) is on a white square or corner --/
def isOnWhiteSquareOrCorner (board : Chessboard) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Represents a circle on the chessboard --/
structure Circle where
  centerX : ℝ
  centerY : ℝ
  radius : ℝ

/-- Checks if a circle's circumference is entirely on white squares or corners --/
def isValidCircle (board : Chessboard) (circle : Circle) : Prop :=
  sorry

/-- The theorem to be proved --/
theorem largest_circle_radius (board : Chessboard := createChessboard) :
  ∃ (c : Circle), isValidCircle board c ∧
    ∀ (c' : Circle), isValidCircle board c' → c'.radius ≤ c.radius ∧
    c.radius = Real.sqrt 10 / 2 :=
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_l3237_323764


namespace NUMINAMATH_CALUDE_class_size_l3237_323776

theorem class_size (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : num_groups = 5) 
  (h2 : students_per_group = 6) : 
  num_groups * students_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3237_323776


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l3237_323785

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ n : ℕ,
    n ≥ 1000 ∧
    n < 10000 ∧
    2 ∣ n ∧
    3 ∣ n ∧
    5 ∣ n ∧
    7 ∣ n ∧
    11 ∣ n ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
    n = 2310 :=
by
  sorry

#eval 2310

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l3237_323785


namespace NUMINAMATH_CALUDE_odd_function_a_value_l3237_323720

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then 2^x - a*x else -(2^(-x)) - a*x

-- State the theorem
theorem odd_function_a_value :
  -- f is an odd function
  (∀ x, f a x = -(f a (-x))) →
  -- f(x) = 2^x - ax when x < 0
  (∀ x, x < 0 → f a x = 2^x - a*x) →
  -- f(2) = 2
  f a 2 = 2 →
  -- Then a = -9/8
  a = -9/8 :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l3237_323720


namespace NUMINAMATH_CALUDE_divideAthletes_eq_56_l3237_323715

/-- The number of ways to divide 10 athletes into two teams of 5 people each,
    given that two specific athletes must be on the same team -/
def divideAthletes : ℕ :=
  Nat.choose 8 3

theorem divideAthletes_eq_56 : divideAthletes = 56 := by
  sorry

end NUMINAMATH_CALUDE_divideAthletes_eq_56_l3237_323715
