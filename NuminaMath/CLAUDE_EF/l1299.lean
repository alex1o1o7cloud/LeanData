import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_B_fastest_l1299_129923

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℚ := 621371 / 1000000

/-- Distance traveled by Car A in miles -/
def distance_A : ℚ := 360

/-- Time taken by Car A in hours -/
def time_A : ℚ := 9 / 2

/-- Distance traveled by Car B in kilometers -/
def distance_B_km : ℚ := 870

/-- Time taken by Car B in hours -/
def time_B : ℚ := 27 / 4

/-- Distance traveled by Car C in kilometers -/
def distance_C_km : ℚ := 1150

/-- Time taken by Car C in hours -/
def time_C : ℚ := 10

/-- Speed of Car A in miles per hour -/
def speed_A : ℚ := distance_A / time_A

/-- Speed of Car B in miles per hour -/
def speed_B : ℚ := (distance_B_km * km_to_miles) / time_B

/-- Speed of Car C in miles per hour -/
def speed_C : ℚ := (distance_C_km * km_to_miles) / time_C

theorem car_B_fastest : speed_B > speed_A ∧ speed_B > speed_C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_B_fastest_l1299_129923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_required_l1299_129982

-- Define the room dimensions
noncomputable def room_length : ℝ := 15
noncomputable def room_width : ℝ := 18

-- Define the tile dimensions in feet
noncomputable def tile_length : ℝ := 1 / 2  -- 6 inches = 1/2 foot
noncomputable def tile_width : ℝ := 2 / 3   -- 8 inches = 2/3 foot

-- Theorem statement
theorem tiles_required :
  (room_length * room_width) / (tile_length * tile_width) = 810 := by
  -- Calculate the area of the room
  have room_area : ℝ := room_length * room_width
  -- Calculate the area of a single tile
  have tile_area : ℝ := tile_length * tile_width
  -- Calculate the number of tiles needed
  have num_tiles : ℝ := room_area / tile_area
  -- Prove that the number of tiles is equal to 810
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_required_l1299_129982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l1299_129913

theorem inscribed_squares_area_ratio (s : ℝ) (h : s > 0) : 
  (s * Real.sqrt 3 / 6) ^ 2 / s ^ 2 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l1299_129913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l1299_129954

open Real

/-- A function satisfying the given differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  twice_diff : DifferentiableOn ℝ (λ x => (deriv (deriv f)) x) domain
  ineq : ∀ x ∈ domain, (deriv f x) * (log x) < ((1 - log x) / x) * f x

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  4 * sf.f (exp 1) > 2 * (exp 1) * sf.f (exp 2) ∧
  2 * (exp 1) * sf.f (exp 2) > (exp 3) * sf.f (exp 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l1299_129954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_proof_l1299_129979

noncomputable def length (b : ℝ) : ℝ := b + 14

theorem plot_length_proof (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 26.5) →
  (total_cost = 5300) →
  (cost_per_meter * (2 * (length breadth + breadth)) = total_cost) →
  (length breadth = 57) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_proof_l1299_129979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1299_129951

/-- Definition of the function f -/
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- Main theorem -/
theorem main_theorem (ω : ℝ) (φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π / 2)
  (h3 : ∀ x, f ω φ (x + π / 2) = -(f ω φ x))
  (h4 : ∃ g : ℝ → ℝ, (∀ x, g x = f ω φ (x + π / 6)) ∧ Odd g) :
  (∀ x, f ω φ x = Real.sin (2 * x - π / 3)) ∧
  (∀ A B C a b c : ℝ,
    0 < A ∧ A < π / 2 ∧
    0 < B ∧ B < π / 2 ∧
    0 < C ∧ C < π / 2 ∧
    A + B + C = π ∧
    (2 * c - a) * Real.cos B = b * Real.cos A →
    ∃ y ∈ Set.Ioo 0 1, f ω φ A = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1299_129951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_average_speed_l1299_129941

/-- Calculates the average speed given initial and final odometer readings and travel time -/
noncomputable def average_speed (initial_reading final_reading : ℕ) (travel_time : ℝ) : ℝ :=
  (final_reading - initial_reading : ℝ) / travel_time

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem jessica_average_speed :
  let initial_reading : ℕ := 13431
  let final_reading : ℕ := 13531
  let travel_time : ℝ := 3
  round_to_nearest (average_speed initial_reading final_reading travel_time) = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_average_speed_l1299_129941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_girls_fraction_l1299_129983

theorem field_trip_girls_fraction (total_students : ℕ) (h_positive : total_students > 0) :
  let boys := total_students / 2
  let girls := total_students / 2
  let girls_on_trip := girls / 2
  let boys_on_trip := (boys * 3) / 4
  let total_on_trip := girls_on_trip + boys_on_trip
  (girls_on_trip : ℚ) / total_on_trip = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_girls_fraction_l1299_129983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_RQS_l1299_129960

/-- Triangle PQR -/
structure Triangle (P Q R : ℝ × ℝ) where
  isosceles : dist P R = dist P Q
  area : ℝ
  smallTriangles : ℕ
  smallestArea : ℝ

/-- Triangle PQS -/
structure InnerTriangle (P Q S : ℝ × ℝ) where
  numSmallTriangles : ℕ
  similarToPQR : Bool

/-- Represents the problem setup -/
structure TriangleProblem (P Q R S : ℝ × ℝ) where
  pqr : Triangle P Q R
  pqs : InnerTriangle P Q S
  pqrArea : pqr.area = 100
  pqrSmallTriangles : pqr.smallTriangles = 20
  pqrSmallestArea : pqr.smallestArea = 1
  pqsSmallTriangles : pqs.numSmallTriangles = 6
  pqsSimilar : pqs.similarToPQR = true

/-- The main theorem to prove -/
theorem area_of_trapezoid_RQS 
  (P Q R S : ℝ × ℝ) 
  (h : TriangleProblem P Q R S) : 
  (h.pqr.area - h.pqs.numSmallTriangles * h.pqr.smallestArea) = 94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_RQS_l1299_129960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l1299_129966

-- Define the properties of cones C and D
def radius_C : ℝ := 20
def height_C : ℝ := 40
def radius_D : ℝ := 40
def height_D : ℝ := 20

-- Define the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem volume_ratio_of_cones :
  (cone_volume radius_C height_C) / (cone_volume radius_D height_D) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l1299_129966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1299_129926

/-- The diameter of a bicycle wheel that makes 393.1744908390343 revolutions in 1 km -/
noncomputable def wheel_diameter : ℝ :=
  let revolutions : ℝ := 393.1744908390343
  let distance : ℝ := 1000  -- 1 km in meters
  let circumference : ℝ := distance / revolutions
  circumference / Real.pi

/-- Theorem stating that the wheel diameter is approximately 0.809 meters -/
theorem wheel_diameter_approx :
  ∃ ε > 0, abs (wheel_diameter - 0.809) < ε ∧ ε < 0.001 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1299_129926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_mixed_number_l1299_129963

theorem reciprocal_of_mixed_number :
  (-(1 + 4/5 : ℚ))⁻¹ = -5/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_mixed_number_l1299_129963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_league_games_l1299_129907

-- Define the total number of games
def total_games : ℕ := sorry

-- Define the number of games won in the first 100 games
def first_100_wins : ℕ := 63

-- Define the number of remaining games (excluding first 100 and ties)
def remaining_games : ℕ := sorry

-- Define the number of games won in the remaining games
def remaining_wins : ℕ := (remaining_games * 48) / 100

-- Define the number of tie games
def tie_games : ℕ := 5

-- Define the total number of games won
def total_wins : ℕ := first_100_wins + remaining_wins

-- State the theorem
theorem football_league_games :
  (total_games = 100 + remaining_games + tie_games) ∧
  (total_wins = (total_games - tie_games) * 58 / 100) →
  total_games = 155 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_league_games_l1299_129907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_paths_l1299_129906

/-- The number of paths in a grid from (0,0) to (i,j) moving right, down, or diagonally down-right -/
def num_paths : Nat → Nat → Nat
  | 0, 0 => 1
  | 0, j+1 => num_paths 0 j
  | i+1, 0 => num_paths i 0
  | i+1, j+1 => num_paths i (j+1) + num_paths (i+1) j + num_paths i j

/-- The number of paths from January 1 to December 31 in a calendar grid -/
theorem calendar_paths : num_paths 11 30 = 372 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_paths_l1299_129906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_necessary_not_sufficient_l1299_129971

-- Define a complex number
def complex (x y : ℝ) : ℂ := ⟨x, y⟩

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem x_zero_necessary_not_sufficient :
  ∀ (x y : ℝ),
    (is_purely_imaginary (complex x y) → x = 0) ∧
    ¬(x = 0 → is_purely_imaginary (complex x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_necessary_not_sufficient_l1299_129971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alphabet_sum_l1299_129964

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 1 => 1
  | 2 => 2
  | 3 => 0
  | 4 => -1
  | 5 => -2
  | 6 => -1
  | 7 => 0
  | 8 => 1
  | 9 => 2
  | 0 => 1
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

def alphabet_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0  -- This case should never occur for valid lowercase letters

def word_value (word : String) : ℤ :=
  word.toList.map (λ c => letter_value (alphabet_position c)) |>.sum

theorem alphabet_sum : word_value "alphabet" = 4 := by
  sorry

#eval word_value "alphabet"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alphabet_sum_l1299_129964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_max_min_part2_exists_q_l1299_129961

-- Define the quadratic function
def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

-- Part 1
theorem part1_max_min :
  let q := 1
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 →
    f q x ≤ 21 ∧
    f q x ≥ -11 ∧
    ∃ x₁ x₂, x₁ ∈ Set.Icc (-1 : ℝ) 1 ∧ x₂ ∈ Set.Icc (-1 : ℝ) 1 ∧ f q x₁ = 21 ∧ f q x₂ = -11 :=
by sorry

-- Part 2
theorem part2_exists_q :
  ∃ q : ℝ, 0 < q ∧ q < 10 ∧
    (∀ x, x ∈ Set.Icc q 10 → f q x ≥ -51) ∧
    (∃ x, x ∈ Set.Icc q 10 ∧ f q x = -51) ∧
    q = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_max_min_part2_exists_q_l1299_129961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_sector_l1299_129930

/-- Given a circle sector with area 118.8 square meters and central angle 42 degrees,
    the radius of the circle is approximately 17.99 meters. -/
theorem circle_radius_from_sector (area : ℝ) (angle : ℝ) (r : ℝ) : 
  area = 118.8 →
  angle = 42 →
  area = (angle / 360) * Real.pi * r^2 →
  ∃ ε > 0, |r - 17.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_sector_l1299_129930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1299_129927

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 / 16 + y^2 / 25 = 1

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := y^2 / 5 - x^2 / 4 = 1

/-- The point P through which the hyperbola passes -/
noncomputable def point_P : ℝ × ℝ := (-2, Real.sqrt 10)

/-- The focus of the ellipse -/
def ellipse_focus : ℝ × ℝ := (0, 3)

theorem hyperbola_properties :
  (∃ (c : ℝ), c^2 = 5 + 4 ∧ 
              (ellipse_focus.1, c) = ellipse_focus) ∧
  hyperbola_eq point_P.1 point_P.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1299_129927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l1299_129933

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a - 3*x| - |2 + x|

-- Theorem for part (1)
theorem solution_part1 :
  let S := {x : ℝ | f 2 x ≤ 3}
  S = Set.Icc (-3/4) (7/2) := by sorry

-- Theorem for part (2)
theorem solution_part2 :
  {a : ℝ | ∀ x, f a x ≥ 1 - a + 2*|2 + x|} = Set.Ici (-5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l1299_129933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercepts_l1299_129928

noncomputable def ellipse_foci : ℝ × ℝ × ℝ × ℝ := (0, 3, 4, 0)

def point_on_ellipse : ℝ × ℝ := (0, 0)

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem ellipse_x_intercepts :
  let (f₁x, f₁y, f₂x, f₂y) := ellipse_foci
  let (px, py) := point_on_ellipse
  let constant_sum := distance f₁x f₁y px py + distance f₂x f₂y px py
  ∀ x : ℝ, (x = 0 ∨ x = 20/7) ↔
    distance f₁x f₁y x 0 + distance f₂x f₂y x 0 = constant_sum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercepts_l1299_129928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1299_129977

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| - |x - 2|

-- Part I
theorem part_one : 
  {x : ℝ | f (-2) x ≤ 4} = Set.Icc (-4) 4 := by sorry

-- Part II
theorem part_two : 
  (∀ x, f a x ≥ 3*a^2 - 3*|2 - x|) → a ∈ Set.Icc (-1) (4/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1299_129977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_l1299_129992

/-- The original function f(x) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (3 * x + φ)

/-- The shifted function g(x) -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (3 * (x - Real.pi/12) + φ)

/-- Theorem stating the smallest possible value of |φ| -/
theorem smallest_phi : ∃ (φ : ℝ), 
  (∀ (x : ℝ), g φ x = g φ (-x)) ∧ 
  (∀ (ψ : ℝ), (∀ (x : ℝ), g ψ x = g ψ (-x)) → |φ| ≤ |ψ|) ∧
  |φ| = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_l1299_129992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_l1299_129947

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^5
noncomputable def g (x : ℝ) : ℝ := x + 1/x

-- Theorem statement
theorem f_and_g_are_odd : IsOdd f ∧ IsOdd g := by
  constructor
  · intro x
    simp [f]
    ring
  · intro x
    simp [g]
    field_simp
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_l1299_129947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1299_129998

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (right_angle_B : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (AC_perp_CD : (C.1 - A.1) * (D.1 - C.1) + (C.2 - A.2) * (D.2 - C.2) = 0)
  (AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15^2)
  (BC_length : ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 20^2)
  (CD_length : ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 9^2)

/-- The perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2) +
  Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2) +
  Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2) +
  Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2)

/-- Theorem: The perimeter of the given quadrilateral is 44 + √706 -/
theorem quadrilateral_perimeter (q : Quadrilateral) : perimeter q = 44 + Real.sqrt 706 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1299_129998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1299_129911

noncomputable def v1 : ℝ × ℝ := (5, 2)
noncomputable def v2 : ℝ × ℝ := (-2, 5)
noncomputable def p : ℝ × ℝ := (3/2, 7/2)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := v.1 * v.1 + v.2 * v.2
  ((dot_product / magnitude_squared) * v.1, (dot_product / magnitude_squared) * v.2)

theorem projection_equality :
  proj v1 p = proj v2 p ∧
  ∀ q : ℝ × ℝ, q ≠ p → proj v1 q ≠ proj v2 q := by
  sorry

#check projection_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1299_129911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_FIGOYZ_l1299_129975

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the geometric objects
variable (A B C D E F G I O T U Y Z : Point)
variable (Γ : Circle)

-- Define the geometric relationships
variable (triangle_ABC : Set Point)
variable (circumcircle : Set Point → Circle)
variable (intersect : Point → Point → Set Point)
variable (on_circle : Point → Circle → Prop)
variable (tangent_point : Circle → Point → Point)
variable (reflect : Point → Point → Point → Point)
variable (collinear : Point → Point → Point → Prop)
variable (on_line : Point → Set Point → Prop)
variable (line : Point → Point → Set Point)
variable (segment : Point → Point → Set Point)

-- State the theorem
theorem concyclic_FIGOYZ 
  (h1 : Γ = circumcircle triangle_ABC)
  (h2 : D ∈ intersect C O ∩ segment A B)
  (h3 : E ∈ intersect B O ∩ segment C A)
  (h4 : on_circle F Γ ∧ F ≠ A ∧ collinear A O F)
  (h5 : on_circle I Γ ∧ on_circle I (circumcircle {A, D, E}))
  (h6 : on_line Y (line B E) ∧ on_circle Y (circumcircle {C, E, I}))
  (h7 : on_line Z (line C D) ∧ on_circle Z (circumcircle {B, D, I}))
  (h8 : T ∈ intersect (tangent_point Γ B) (tangent_point Γ C))
  (h9 : on_circle U Γ ∧ collinear T F U ∧ U ≠ F)
  (h10 : G = reflect U B C) :
  ∃ (circ : Circle), on_circle F circ ∧ on_circle I circ ∧ on_circle G circ ∧ 
                     on_circle O circ ∧ on_circle Y circ ∧ on_circle Z circ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_FIGOYZ_l1299_129975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_calculation_l1299_129919

/-- Calculates the present value given future value, interest rate, and time period. -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- Theorem: The present worth of $3600 due in 2 years at 20% per annum compound interest is $2500. -/
theorem present_worth_calculation :
  let futureValue : ℝ := 3600
  let interestRate : ℝ := 0.20
  let years : ℝ := 2
  presentValue futureValue interestRate years = 2500 := by
  -- Unfold the definition of presentValue
  unfold presentValue
  -- Simplify the expression
  simp
  -- The proof is complete, but we use sorry to skip the detailed steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_calculation_l1299_129919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_value_l1299_129995

noncomputable section

-- Define the points and line
def F : ℝ × ℝ := (0, 1)
def D : ℝ × ℝ := (0, 2)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -1}

-- Define the moving point P and its foot Q
variable (P : ℝ × ℝ)
def Q (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -1)

-- Define the dot product condition
def dot_product_condition (P : ℝ × ℝ) : Prop :=
  (P.1 - (Q P).1) * (F.1 - (Q P).1) + (P.2 - (Q P).2) * (F.2 - (Q P).2) =
  (P.1 - F.1) * ((Q P).1 - F.1) + (P.2 - F.2) * ((Q P).2 - F.2)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define circle M
def M (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - D.1)^2 + (center.2 - D.2)^2}

-- Define A and B as intersections of M with x-axis
def A (center : ℝ × ℝ) : ℝ × ℝ := (center.1 - 2, 0)
def B (center : ℝ × ℝ) : ℝ × ℝ := (center.1 + 2, 0)

-- Define l₁ and l₂
noncomputable def l₁ (center : ℝ × ℝ) : ℝ := Real.sqrt ((A center).1^2 + (A center).2^2)
noncomputable def l₂ (center : ℝ × ℝ) : ℝ := Real.sqrt ((B center).1^2 + (B center).2^2)

-- State the theorem
theorem trajectory_and_max_value :
  (∀ P, dot_product_condition P → P ∈ C) ∧
  (∃ max_value : ℝ, max_value = 2 * Real.sqrt 2 ∧
    ∀ center ∈ C, l₁ center / l₂ center + l₂ center / l₁ center ≤ max_value) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_value_l1299_129995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_equal_volume_prism_l1299_129914

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume_prism (prism_length prism_width prism_height : ℝ)
  (h_length : prism_length = 5)
  (h_width : prism_width = 3)
  (h_height : prism_height = 30) :
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := Real.rpow prism_volume (1/3)
  let cube_surface_area := 6 * cube_edge^2
  ∃ ε > 0, |cube_surface_area - 353| < ε := by
  sorry

#check cube_surface_area_equal_volume_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_equal_volume_prism_l1299_129914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_in_cube_l1299_129937

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the dihedral angle between two planes -/
noncomputable def dihedralAngle (p1 p2 : Plane3D) : ℝ := sorry

/-- Constructs a unit cube with vertices ABCD-A'B'C'D' -/
def unitCube : List Point3D := sorry

/-- Constructs the plane A'BD from the unit cube -/
def planeA'BD (cube : List Point3D) : Plane3D := sorry

/-- Constructs the plane BDC' from the unit cube -/
def planeBDC' (cube : List Point3D) : Plane3D := sorry

theorem dihedral_angle_in_cube :
  let cube := unitCube
  let p1 := planeA'BD cube
  let p2 := planeBDC' cube
  dihedralAngle p1 p2 = Real.arccos (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_in_cube_l1299_129937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1299_129900

/-- The circle defined by the equation x^2 + y^2 + 4x - 2y + 4 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.1 - 2*p.2 + 4 = 0}

/-- The line defined by the equation y = x - 1 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- The shortest distance from a point on the circle to the line -/
noncomputable def shortestDistance : ℝ := 2 * Real.sqrt 2 - 1

theorem shortest_distance_circle_to_line :
  ∀ p ∈ Circle, ∃ q ∈ Line, ∀ r ∈ Line,
    dist p q ≤ dist p r ∧
    dist p q = shortestDistance := by
  sorry

#check shortest_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1299_129900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_composition_correct_equation_l1299_129920

/-- Represents the number of students in a class -/
def total_students : ℕ := 49

/-- Represents the number of boys in the class -/
def boys (x : ℕ) : Prop := True

/-- Represents the number of girls in the class -/
def girls (x : ℕ) : Prop := True

/-- The condition that when there is one less boy, the number of boys is exactly half the number of girls -/
axiom half_condition (x : ℕ) : boys x → girls (2 * (x - 1))

/-- The theorem stating that the equation 2(x-1) + x = 49 correctly represents the class composition -/
theorem class_composition (x : ℕ) : 
  boys x → girls (2 * (x - 1)) → 2 * (x - 1) + x = total_students :=
by
  sorry

/-- The main theorem proving that the equation 2(x-1) + x = 49 is correct -/
theorem correct_equation : 
  ∃ x : ℕ, boys x ∧ girls (2 * (x - 1)) ∧ 2 * (x - 1) + x = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_composition_correct_equation_l1299_129920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_roots_l1299_129958

-- Define the functions f(a) and g(a)
noncomputable def f (a : ℝ) : ℝ := a^2 - Real.sqrt 21 * a + 26
noncomputable def g (a : ℝ) : ℝ := (3/2) * a^2 - Real.sqrt 21 * a + 27

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (f a * x^2 + 1) / (x^2 + g a) = Real.sqrt ((x * g a - 1) / (f a - x))

-- State the theorem
theorem minimize_sum_of_roots :
  ∃ (a : ℝ), a = Real.sqrt 21 / 2 ∧
  ∀ (b : ℝ), f a ≤ f b ∧
  (∀ (x : ℝ), equation a x → (∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = f a)) := by
  sorry

#check minimize_sum_of_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_roots_l1299_129958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_circles_l1299_129902

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The fixed unit circle S -/
def S : Circle :=
  { center := (0, 0), radius := 1 }

/-- Predicate to check if two circles intersect -/
def intersects (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (c1.radius + c2.radius)^2

/-- Predicate to check if one circle contains the center of another -/
def contains_center (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 < c1.radius^2

/-- The main theorem -/
theorem max_intersecting_circles :
  ∀ (circles : List Circle),
    (∀ c, c ∈ circles → c.radius = 1) →
    (∀ c, c ∈ circles → intersects c S) →
    (∀ c, c ∈ circles → ¬contains_center c S) →
    (∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → ¬contains_center c1 c2) →
    circles.length ≤ 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_circles_l1299_129902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_l1299_129986

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 5

theorem inverse_g : g⁻¹ (-1/40) = (7/8)^(1/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_l1299_129986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_correct_fare_20km_l1299_129935

/-- Fare function for a taxi ride based on distance traveled -/
noncomputable def fare (x : ℝ) : ℝ :=
  if x ≤ 4 then 10
  else if x ≤ 18 then 1.2 * x + 5.2
  else 1.8 * x - 5.6

theorem fare_correct (x : ℝ) (h : x > 0) : 
  fare x = if x ≤ 4 then 10
           else if x ≤ 18 then 1.2 * x + 5.2
           else 1.8 * x - 5.6 := by sorry

theorem fare_20km : fare 20 = 30.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_correct_fare_20km_l1299_129935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l1299_129944

/-- Represents the number of family members -/
def family_size : ℕ := 5

/-- Represents the number of adults who can drive -/
def driver_options : ℕ := 2

/-- Represents the number of front seats -/
def front_seats : ℕ := 2

/-- Represents the number of back seats -/
def back_seats : ℕ := 3

/-- Calculates the number of seating arrangements -/
def seating_arrangements : ℕ :=
  driver_options * (family_size - 1) * Nat.factorial (family_size - 2)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_count :
  seating_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l1299_129944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_improved_score_theorem_l1299_129999

/-- Represents a student's test performance --/
structure TestPerformance where
  score : ℚ
  studyTime : ℚ

/-- Calculates the score given study time and efficiency --/
def calculateScore (initialPerformance : TestPerformance) (newStudyTime : ℚ) (efficiencyImprovement : ℚ) : ℚ :=
  (initialPerformance.score / initialPerformance.studyTime) * (newStudyTime * (1 + efficiencyImprovement))

theorem improved_score_theorem (initialPerformance : TestPerformance) 
    (h1 : initialPerformance.score = 80)
    (h2 : initialPerformance.studyTime = 4)
    (h3 : calculateScore initialPerformance 5 (1/10) = 110) : 
  calculateScore initialPerformance 5 (1/10) = 110 := by
  sorry

#eval calculateScore { score := 80, studyTime := 4 } 5 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_improved_score_theorem_l1299_129999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_is_sixty_percent_l1299_129943

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  grape_amount : ℝ

/-- Calculate the percentage of watermelon juice in the fruit drink -/
noncomputable def watermelon_percentage (drink : FruitDrink) : ℝ :=
  100 * (drink.total - (drink.orange_percent / 100 * drink.total) - drink.grape_amount) / drink.total

/-- Theorem stating that for a specific fruit drink composition, 
    the percentage of watermelon juice is 60% -/
theorem watermelon_is_sixty_percent 
  (drink : FruitDrink) 
  (h1 : drink.total = 140)
  (h2 : drink.orange_percent = 15)
  (h3 : drink.grape_amount = 35) : 
  watermelon_percentage drink = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_is_sixty_percent_l1299_129943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_product_l1299_129948

theorem cos_difference_product (a b : ℝ) :
  Real.cos (a + b) - Real.cos (a - b) = -2 * Real.sin (2 * a) * Real.sin (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_product_l1299_129948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1299_129942

theorem stones_partition (n : ℕ) (h : n = 660) :
  ∃ (partition : Finset (Finset ℕ)),
    Finset.card partition = 30 ∧
    (∀ s, s ∈ partition → Finset.card s > 0) ∧
    (Finset.sum partition (λ s => Finset.sum s id) = n) ∧
    (∀ s t, s ∈ partition → t ∈ partition → ∀ x y, x ∈ s → y ∈ t → 2 * x > y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1299_129942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_matching_probability_l1299_129962

theorem sock_matching_probability (black blue : ℕ) (h1 : black = 12) (h2 : blue = 10) :
  let total := black + blue
  let matching_pairs := (black * (black - 1) + blue * (blue - 1)) / 2
  let total_pairs := total * (total - 1) / 2
  (matching_pairs : ℚ) / total_pairs = 111 / 231 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_matching_probability_l1299_129962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_thirds_equals_three_fourths_l1299_129952

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

-- Define the function f composed with g
noncomputable def f_comp_g (x : ℝ) : ℝ := 
  if x ≠ 0 then (2 - 3 * x^2) / (2 * x^2) else 0

-- Theorem statement
theorem f_two_thirds_equals_three_fourths : 
  ∃ (f : ℝ → ℝ), f (2/3) = 3/4 ∧ ∀ (x : ℝ), x ≠ 0 → f (g x) = f_comp_g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_thirds_equals_three_fourths_l1299_129952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_sum_is_zero_l1299_129993

/-- Represents a line in the coordinate plane -/
structure Line where
  slope : ℚ
  intercept : ℚ
  is_x_intercept : Bool

/-- The intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- Theorem: The sum of coordinates of the intersection point is 0 -/
theorem intersection_coordinate_sum_is_zero :
  let line_a := Line.mk (-1) 2 true
  let line_b := Line.mk 5 (-10) false
  let (a, b) := intersection line_a line_b
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_sum_is_zero_l1299_129993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1299_129990

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h1 : cos (α - β/2) = -2*sqrt 7/7)
  (h2 : sin (α/2 - β) = 1/2)
  (h3 : α ∈ Set.Ioo (π/2) π)
  (h4 : β ∈ Set.Ioo 0 (π/2)) :
  cos ((α + β)/2) = -sqrt 21/14 ∧ 
  tan (α + β) = 5*sqrt 3/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1299_129990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_distance_bounds_l1299_129912

/-- The curve C: x²/4 + y²/9 = 1 -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

/-- The line l: x = 2 + t, y = 2 - 2t -/
def L (x y t : ℝ) : Prop := x = 2 + t ∧ y = 2 - 2*t

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Angle between two lines -/
noncomputable def angle (m1 m2 : ℝ) : ℝ :=
  Real.arctan ((m2 - m1) / (1 + m1*m2))

theorem curve_line_intersection_distance_bounds :
  ∀ (x y : ℝ), C x y →
  ∃ (t x_int y_int : ℝ),
    L x_int y_int t ∧
    angle ((y - y_int) / (x - x_int)) ((2 - y_int) / (2 + t - x_int)) = π/6 →
    let d := distance x y x_int y_int
    2*Real.sqrt 5/5 ≤ d ∧ d ≤ 22*Real.sqrt 5/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_distance_bounds_l1299_129912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1299_129997

/-- The time taken for a train to cross a platform -/
noncomputable def cross_time (train_length platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

/-- Theorem: The time taken for a train to cross a 250 m platform is 20 seconds -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (first_platform_length : ℝ) 
  (second_platform_length : ℝ) 
  (first_crossing_time : ℝ) :
  train_length = 150 →
  first_platform_length = 150 →
  second_platform_length = 250 →
  first_crossing_time = 15 →
  cross_time train_length second_platform_length 
    ((train_length + first_platform_length) / first_crossing_time) = 20 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1299_129997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_equal_distribution_l1299_129909

/-- Represents the number of students in the circle. -/
def n : ℕ := 2018

/-- Represents the initial number of cards one student has. -/
def initialCards : ℕ := 2018

/-- Represents the sum of cards held by students at even positions. -/
def S₀ : ℕ → ℕ := sorry

/-- Represents the sum of cards held by students at odd positions. -/
def S₁ : ℕ → ℕ := sorry

/-- Represents a single turn where each student gives one card to each neighbor. -/
def turn : (ℕ → ℕ) → (ℕ → ℕ) := sorry

/-- States that initially, S₀ = 0 and S₁ = initialCards. -/
axiom initial_state : S₀ 0 = 0 ∧ S₁ 0 = initialCards

/-- States that after each turn, the parity of S₀ and S₁ is preserved. -/
axiom parity_preserved (t : ℕ) : 
  S₀ (t + 1) % 2 = S₀ t % 2 ∧ S₁ (t + 1) % 2 = S₁ t % 2

/-- The main theorem stating it's impossible for each student to end up with one card. -/
theorem impossible_equal_distribution : 
  ¬ ∃ t : ℕ, S₀ t = n / 2 ∧ S₁ t = n / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_equal_distribution_l1299_129909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1299_129946

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := (x ∈ Set.Icc (-1) 2 ∧ x ≠ 2) ∨ (x ∈ Set.Ioc 2 5)

-- State the symmetry property
axiom symmetry_property : ∀ x, domain x → domain (4 - x) → f x + f (4 - x) = 2

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-1) (1/2) ∪ Set.Ioc 2 5

-- State the theorem
theorem inequality_solution :
  ∀ x, domain x → (f x - f (4 - x) > -2 ↔ x ∈ solution_set) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1299_129946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_alpha_l1299_129989

theorem tan_double_alpha (α : Real) 
  (h1 : Real.sin α + Real.cos α = 1/5)
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (2*α) = 24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_alpha_l1299_129989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_win_chance_l1299_129916

/-- Represents a lottery ticket -/
structure LotteryTicket where
  id : ℕ

/-- Represents a lottery -/
structure Lottery where
  winProbability : ℝ
  tickets : List LotteryTicket

/-- The probability of winning for a single ticket is equal to the lottery's win probability -/
def ticketWinProbability (lottery : Lottery) (ticket : LotteryTicket) : ℝ :=
  lottery.winProbability

theorem equal_win_chance (lottery : Lottery) 
    (h1 : lottery.winProbability = 0.001) 
    (h2 : lottery.tickets.length = 1000) :
    ∀ t1 t2, t1 ∈ lottery.tickets → t2 ∈ lottery.tickets → 
    ticketWinProbability lottery t1 = ticketWinProbability lottery t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_win_chance_l1299_129916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1299_129980

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ m : ℝ, f (1 - m) + f (1 - m^2) < 0 ↔ m < -2 ∨ m > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1299_129980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hamiltonian_path_iff_mn_even_l1299_129994

/-- A rectangular grid of streets -/
structure RectangularGrid where
  m : ℕ
  n : ℕ
  m_gt_one : m > 1
  n_gt_one : n > 1

/-- A path through all intersections in a rectangular grid -/
def HamiltonianPath (grid : RectangularGrid) :=
  ∃ (path : List (ℕ × ℕ)),
    path.length = grid.m * grid.n ∧
    path.head? = path.getLast? ∧
    ∀ i j, i < j → i < path.length → j < path.length → path[i]! ≠ path[j]!

/-- Theorem: A Hamiltonian path exists if and only if mn is even -/
theorem hamiltonian_path_iff_mn_even (grid : RectangularGrid) :
  HamiltonianPath grid ↔ Even (grid.m * grid.n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hamiltonian_path_iff_mn_even_l1299_129994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1299_129965

/-- The minimum of two real numbers -/
noncomputable def m (x y : ℝ) : ℝ := min x y

/-- The maximum of two real numbers -/
noncomputable def M (x y : ℝ) : ℝ := max x y

/-- Theorem: For distinct real numbers a, b, c, d, e such that a < c < b < e < d,
    M(m(b,m(c,d)), M(a,m(c,e))) = c -/
theorem problem_statement 
  (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_order : a < c ∧ c < b ∧ b < e ∧ e < d) : 
  M (m b (m c d)) (M a (m c e)) = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1299_129965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_most_efficient_l1299_129915

/-- Represents a participant in the long jump event -/
structure Participant where
  name : String
  weight : ℝ
  jump_distance : ℝ

/-- Calculates the weight-distance ratio for a participant -/
noncomputable def weight_distance_ratio (p : Participant) : ℝ :=
  p.weight / p.jump_distance

/-- Theorem: Isabella is the most efficient jumper -/
theorem isabella_most_efficient (ricciana margarita isabella : Participant)
  (h_ricciana : ricciana.name = "Ricciana" ∧ ricciana.weight = 120 ∧ ricciana.jump_distance = 4)
  (h_margarita : margarita.name = "Margarita" ∧ margarita.weight = 110 ∧ margarita.jump_distance = 7)
  (h_isabella : isabella.name = "Isabella" ∧ isabella.weight = 100 ∧ isabella.jump_distance = 7) :
  weight_distance_ratio isabella < weight_distance_ratio ricciana ∧
  weight_distance_ratio isabella < weight_distance_ratio margarita :=
by
  sorry

#eval "Isabella is the most efficient jumper"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_most_efficient_l1299_129915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1299_129904

-- Define the parabola C₁
noncomputable def C₁ (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola C₂
noncomputable def C₂ (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a c : ℝ) : ℝ := c/a

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x₀ y₀ : ℝ), C₁ x₀ y₀ ∧ 
   (∃ (m n : ℝ), m*x₀ + n*y₀ = Real.sqrt 3/3 ∧ 
    ∀ (x y : ℝ), C₂ x y a b → m*x + n*y = 0)) →
  eccentricity a (Real.sqrt (a^2 + b^2)) = Real.sqrt 6/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1299_129904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_sum_cube_not_prime_l1299_129974

theorem abc_sum_cube_not_prime (k : ℕ) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = k * (a + b + c)) ∧
  (∀ (a b c : ℕ), a > 0 → b > 0 → c > 0 → a * b * c = k * (a + b + c) → ¬ Nat.Prime (a^3 + b^3 + c^3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_sum_cube_not_prime_l1299_129974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_of_inclination_l1299_129931

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 3 * x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) - 3

-- Define the angle of inclination
noncomputable def α (x : ℝ) : ℝ := Real.arctan (f' x)

-- State the theorem
theorem min_angle_of_inclination :
  ∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → α x ≥ 3 * Real.pi / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_of_inclination_l1299_129931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probabilities_l1299_129950

noncomputable def successful_roll_prob : ℝ := 2/3

noncomputable def at_least_one_of_two : ℝ := 1 - (1 - successful_roll_prob)^2

noncomputable def at_least_two_of_four : ℝ := 1 - (1 - successful_roll_prob)^4 - 4 * successful_roll_prob * (1 - successful_roll_prob)^3

theorem equal_probabilities : at_least_one_of_two = at_least_two_of_four := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probabilities_l1299_129950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l1299_129945

theorem cos_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α > 0) (h2 : α < π / 2) (h3 : Real.tan α = 2) : 
  Real.cos (α + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l1299_129945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_scaling_l1299_129988

theorem equilateral_triangle_scaling (ω : ℂ) (l : ℝ) : 
  Complex.abs ω = 3 →
  l > 1 →
  (Complex.abs (ω - ω^2) = Complex.abs (ω^2 - l*ω) ∧ 
   Complex.abs (ω^2 - l*ω) = Complex.abs (l*ω - ω)) →
  l = (1 + Real.sqrt 33) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_scaling_l1299_129988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_four_f_strictly_increasing_l1299_129968

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- Theorem 1: Prove that a = 4 given f(1) = 5
theorem a_equals_four : f 4 1 = 5 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Check that 1 + 4/1 = 5
  norm_num

-- Theorem 2: Prove that f(x) is strictly increasing on (2, +∞)
theorem f_strictly_increasing :
  ∀ x y : ℝ, 2 < x → x < y → f 4 x < f 4 y := by
  -- Introduce variables and hypotheses
  intros x y hx hy
  -- Unfold the definition of f
  unfold f
  -- Use calculus to prove the inequality
  sorry  -- The full proof would require more advanced Lean tactics


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_four_f_strictly_increasing_l1299_129968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1299_129903

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- Definition of the left focus F₁ -/
def left_focus : ℝ × ℝ := (-3, 0)

/-- Definition of the right focus F₂ -/
def right_focus : ℝ × ℝ := (3, 0)

/-- Definition of a point being on the ellipse -/
def on_ellipse (p : ℝ × ℝ) : Prop := is_ellipse p.1 p.2

/-- Definition of a chord passing through a point -/
def chord_through (a b p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = (t * a.1 + (1 - t) * b.1, t * a.2 + (1 - t) * b.2)

/-- The main theorem -/
theorem ellipse_chord_theorem (a b : ℝ × ℝ) :
  on_ellipse a ∧ on_ellipse b ∧
  chord_through a b left_focus ∧
  (let s := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
   let p := (s + Real.sqrt ((a.1 - right_focus.1)^2 + (a.2 - right_focus.2)^2) +
             Real.sqrt ((b.1 - right_focus.1)^2 + (b.2 - right_focus.2)^2)) / 2
   p = 1) →
  |a.2 - b.2| = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1299_129903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_final_count_l1299_129905

-- Define the initial number of balls for Robert and Tim
def robert_initial : ℕ := 25
def tim_total : ℕ := 40

-- Define the fraction of Tim's balls given to Robert
def fraction_given : ℚ := 1 / 2

-- Theorem to prove Robert's final ball count
theorem robert_final_count :
  robert_initial + (fraction_given * tim_total).floor = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_final_count_l1299_129905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_dry_person_l1299_129959

/-- A group of people where each person sprays their closest neighbor -/
structure SprayGroup (n : ℕ) where
  /-- The set of people in the group -/
  people : Finset (Fin (2*n + 1))
  /-- The spraying relation: who sprays whom -/
  sprays : Fin (2*n + 1) → Fin (2*n + 1)
  /-- Each person sprays their closest neighbor -/
  spray_closest : ∀ i, sprays i ≠ i ∧ ∀ j ≠ i, sprays i ≠ j → 
    dist (i : ℝ) (sprays i : ℝ) ≤ dist (i : ℝ) (j : ℝ)

/-- There is always one person who doesn't get sprayed in a SprayGroup -/
theorem one_dry_person (n : ℕ) (g : SprayGroup n) : 
  ∃ i, ∀ j, g.sprays j ≠ i := by
  sorry

#check one_dry_person

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_dry_person_l1299_129959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_ear_weight_l1299_129925

/-- The weight of a bushel of corn in pounds -/
noncomputable def bushel_weight : ℚ := 56

/-- The number of bushels Clyde picked -/
noncomputable def bushels_picked : ℚ := 2

/-- The number of individual corn cobs Clyde picked -/
noncomputable def cobs_picked : ℚ := 224

/-- The weight of an individual ear of corn in pounds -/
noncomputable def ear_weight : ℚ := (bushel_weight * bushels_picked) / cobs_picked

theorem individual_ear_weight :
  ear_weight = 1/2 := by
  -- Unfold definitions
  unfold ear_weight bushel_weight bushels_picked cobs_picked
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_ear_weight_l1299_129925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_equal_points_l1299_129972

/-- Represents a participant in the tournament -/
structure Participant where
  id : Nat
  points : Nat
  defeated : List Nat

/-- Represents the tournament -/
structure Tournament where
  participants : List Participant
  num_participants : Nat
  no_ties : Bool
  win_points : Nat
  loss_points : Nat

/-- Calculate the coefficient of a participant -/
def coefficient (t : Tournament) (p : Participant) : Nat :=
  (p.defeated.map (fun id => 
    match t.participants.find? (fun x => x.id = id) with
    | some x => x.points
    | none => 0
  )).sum

/-- All participants have equal coefficients -/
def equal_coefficients (t : Tournament) : Prop :=
  ∀ p1 p2 : Participant, p1 ∈ t.participants → p2 ∈ t.participants → coefficient t p1 = coefficient t p2

theorem all_equal_points (t : Tournament) 
  (h1 : t.num_participants > 2)
  (h2 : t.no_ties = true)
  (h3 : t.win_points = 1)
  (h4 : t.loss_points = 0)
  (h5 : equal_coefficients t) :
  ∀ p1 p2 : Participant, p1 ∈ t.participants → p2 ∈ t.participants → p1.points = p2.points :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_equal_points_l1299_129972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l1299_129936

/-- A graph representing circles in a plane -/
structure CircleGraph where
  G : SimpleGraph (Fin 2015)
  centers : Fin 2015 → ℝ × ℝ
  valid_edges : ∀ i j, G.Adj i j ↔ ‖centers i - centers j‖ ≤ 2

/-- The size of the largest clique in a graph -/
noncomputable def clique_number {α : Type*} (G : SimpleGraph α) : ℕ := sorry

/-- The size of the largest independent set in a graph -/
noncomputable def independence_number {α : Type*} (G : SimpleGraph α) : ℕ := sorry

/-- Main theorem: There exists either a clique or an independent set of size at least 27 -/
theorem circle_intersection_theorem (CG : CircleGraph) :
  max (clique_number CG.G) (independence_number CG.G) ≥ 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l1299_129936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1299_129985

/-- Represents a number in base 8 (octal) --/
structure Octal where
  value : ℕ

/-- Convert an Octal number to its decimal (base 10) representation --/
def octal_to_decimal (n : Octal) : ℕ := n.value

/-- Convert a decimal (base 10) number to its Octal representation --/
def decimal_to_octal (n : ℕ) : Octal := ⟨n⟩

/-- Subtraction operation for Octal numbers --/
def octal_sub (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

instance : OfNat Octal n where
  ofNat := ⟨n⟩

theorem octal_subtraction_theorem :
  octal_sub 52 35 = 13 := by
  -- The proof would go here, but for now we'll use sorry
  sorry

#eval octal_to_decimal (octal_sub 52 35)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1299_129985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1299_129969

theorem sufficient_not_necessary : 
  (∀ a : ℝ, a > 0 → |2*a + 1| > 1) ∧ 
  (∃ a : ℝ, a ≤ 0 ∧ |2*a + 1| > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1299_129969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_soaring_speed_l1299_129901

/-- Given a rocket's flight parameters, prove its soaring speed --/
theorem rocket_soaring_speed 
  (soaring_time : ℝ) 
  (plummeting_time : ℝ)
  (plummeting_distance : ℝ)
  (average_speed : ℝ)
  (h1 : soaring_time = 12)
  (h2 : plummeting_time = 3)
  (h3 : plummeting_distance = 600)
  (h4 : average_speed = 160)
  : 
  (average_speed * (soaring_time + plummeting_time) - plummeting_distance) / soaring_time = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_soaring_speed_l1299_129901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1299_129949

-- Define the function f
noncomputable def f (n m : ℝ) (x : ℝ) : ℝ := (n - 2^x) / (2^(x+1) + m)

-- State the theorem
theorem odd_function_properties :
  ∃ (n m : ℝ), 
    (∀ x, f n m x = -f n m (-x)) ∧ 
    (∀ x ∈ Set.Icc (1/2) 3, ∀ k < -1, f n m (k*x^2) + f n m (2*x-1) > 0) ∧
    n = 1 ∧ m = 2 := by
  -- Proof goes here
  sorry

#check odd_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1299_129949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_has_no_real_roots_l1299_129929

/-- The polynomial P(x) = x^8 - x^7 + 2x^6 - 2x^5 + 3x^4 - 3x^3 + 4x^2 - 4x + 5/2 -/
noncomputable def P (x : ℝ) : ℝ := x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2

/-- Theorem stating that the polynomial P(x) has no real roots -/
theorem P_has_no_real_roots : ∀ x : ℝ, P x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_has_no_real_roots_l1299_129929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_m_value_l1299_129939

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_m_value
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a 2 = 4)
  (h4 : ∃ m : ℝ, f a m = 8) :
  ∃ m : ℝ, f a m = 8 ∧ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_m_value_l1299_129939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l1299_129940

noncomputable section

open Real

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (6, 4)
def D : ℝ × ℝ := (6, 0)

-- Define the angles of the lines
def angle1 : ℝ := 45 * π / 180
def angle2 : ℝ := 75 * π / 180

-- Define the lines from A and B
def lineA1 (x : ℝ) : ℝ := x * tan angle1
def lineA2 (x : ℝ) : ℝ := x * tan angle2
def lineB1 (x : ℝ) : ℝ := 4 - (x * tan angle1)
def lineB2 (x : ℝ) : ℝ := 4 - (x * tan angle2)

-- Define the intersection points
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry
def S : ℝ × ℝ := sorry

-- Define what it means for four points to form a rectangle
def IsRectangle (P Q R S : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem intersection_points_form_rectangle : 
  IsRectangle P Q R S := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l1299_129940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_theorem_l1299_129932

/-- The number of ways to color n connected regions using 6 colors -/
def color_count (n : ℕ) : ℤ :=
  5 * ((-1) ^ n) + 5 ^ n

/-- The properties of the coloring problem -/
structure ColoringProblem where
  n : ℕ
  n_ge_2 : n ≥ 2
  regions_connected : True  -- Placeholder for the connectivity property
  num_colors : Fin 6 → Color  -- 6 different colors

/-- Theorem stating the existence and uniqueness of the coloring count -/
theorem coloring_theorem (p : ColoringProblem) :
  ∃ (f : Fin p.n → Color),
  (∀ i j : Fin p.n, i ≠ j → f i ≠ f j) →  -- Adjacent regions have different colors
  (∃! count : ℤ, count = color_count p.n) :=
by
  sorry

#check coloring_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_theorem_l1299_129932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_girls_count_specific_l1299_129908

noncomputable def school_girls_count (initial_girl_percentage : ℝ) 
                                     (yearly_decrease : ℝ) 
                                     (num_years : ℕ) 
                                     (num_boys : ℕ) : ℕ :=
  let total_students := (num_boys : ℝ) / (1 - initial_girl_percentage)
  let final_girl_percentage := initial_girl_percentage - yearly_decrease * (num_years : ℝ)
  let final_num_girls := final_girl_percentage * total_students
  Int.floor final_num_girls |>.toNat

theorem school_girls_count_specific : 
  school_girls_count 0.6 0.05 2 300 = 375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_girls_count_specific_l1299_129908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_8_l1299_129984

/-- A quadratic polynomial P(x) such that P(P(x)) = x^4 - 2x^3 + 4x^2 - 3x + 4 -/
noncomputable def P : ℝ → ℝ := sorry

/-- The property that P(P(x)) = x^4 - 2x^3 + 4x^2 - 3x + 4 for all x -/
axiom P_composition (x : ℝ) : P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4

/-- P is a quadratic polynomial -/
axiom P_quadratic : ∃ (a b c : ℝ), ∀ x, P x = a*x^2 + b*x + c

/-- Theorem: P(8) = 58 -/
theorem P_at_8 : P 8 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_8_l1299_129984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_below_100_l1299_129934

def f : ℤ → ℤ 
| n => if n > 100 then n - 10 else 91

theorem f_constant_below_100 : ∀ n : ℤ, n ≤ 100 → f n = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_below_100_l1299_129934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_four_equal_parts_l1299_129981

-- Define a convex polygon
structure ConvexPolygon where
  -- We'll leave the internal structure abstract for now
  dummy : Unit

-- Define the area of a polygon
noncomputable def area (p : ConvexPolygon) : ℝ := sorry

-- Define a line in 2D space
structure Line where
  -- We'll leave the internal structure abstract for now
  dummy : Unit

-- Define perpendicularity of lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the division of a polygon by a line
def divide (p : ConvexPolygon) (l : Line) : Prop := sorry

-- Define the regions created by two lines intersecting a polygon
def regions (p : ConvexPolygon) (l1 l2 : Line) : Fin 4 → ConvexPolygon := sorry

-- The main theorem
theorem convex_polygon_four_equal_parts (p : ConvexPolygon) :
  ∃ (l1 l2 : Line), perpendicular l1 l2 ∧
  divide p l1 ∧ divide p l2 ∧
  ∀ (i j : Fin 4), area (regions p l1 l2 i) = area (regions p l1 l2 j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_four_equal_parts_l1299_129981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sufficient_not_necessary_l1299_129970

-- Define the property of being in the second quadrant
def is_second_quadrant (α : ℝ) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

-- Define the condition sin α > cos α
def sin_greater_cos (α : ℝ) : Prop :=
  Real.sin α > Real.cos α

-- Theorem statement
theorem second_quadrant_sufficient_not_necessary :
  (∀ α : ℝ, is_second_quadrant α → sin_greater_cos α) ∧
  (∃ α : ℝ, ¬is_second_quadrant α ∧ sin_greater_cos α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sufficient_not_necessary_l1299_129970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_even_digits_l1299_129918

/-- A function that checks if a natural number has all even digits -/
def allEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → Even d

/-- A function that checks if a natural number has at least one odd digit -/
def hasOddDigit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ Odd d

/-- Theorem stating the largest possible difference between two 6-digit numbers
    with all even digits, where all numbers between them have at least one odd digit -/
theorem largest_difference_even_digits :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    allEvenDigits a ∧
    allEvenDigits b ∧
    (∀ k, a < k ∧ k < b → hasOddDigit k) ∧
    b - a = 111112 ∧
    (∀ a' b', (100000 ≤ a' ∧ a' < 1000000) →
              (100000 ≤ b' ∧ b' < 1000000) →
              allEvenDigits a' →
              allEvenDigits b' →
              (∀ k, a' < k ∧ k < b' → hasOddDigit k) →
              b' - a' ≤ 111112) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_even_digits_l1299_129918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_workers_with_conditions_l1299_129922

/-- The number of ways to distribute 5 workers to 3 intersections -/
def distribute_workers (arrangements : Set (Fin 5 → Fin 3)) : ℕ := sorry

/-- The condition that A and B are at the same intersection -/
def A_and_B_together (arrangement : Fin 5 → Fin 3) : Prop :=
  arrangement 0 = arrangement 1

/-- The condition that each intersection has at least one person -/
def at_least_one_per_intersection (arrangement : Fin 5 → Fin 3) : Prop :=
  ∀ i : Fin 3, ∃ w : Fin 5, arrangement w = i

theorem distribute_workers_with_conditions :
  distribute_workers {arrangement : Fin 5 → Fin 3 |
    A_and_B_together arrangement ∧ at_least_one_per_intersection arrangement} = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_workers_with_conditions_l1299_129922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1299_129996

variable (a m : ℝ)

noncomputable def z₁ : ℂ := (2 * a) / (a - 1) + (a^2 - 1) * Complex.I
noncomputable def z₂ : ℂ := m + (m - 1) * Complex.I

theorem complex_number_problem :
  (z₁ a ∈ Set.range (Complex.ofReal)) → a = -1 ∧
  (a = -1 → ‖z₁ a‖ < ‖z₂ m‖ → m < 0 ∨ m > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1299_129996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_n_equals_581_l1299_129978

def b_n (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 17 = 0 then 15
  else if n % 17 = 0 ∧ n % 13 = 0 then 17
  else if n % 13 = 0 ∧ n % 15 = 0 then 13
  else 0

theorem sum_b_n_equals_581 :
  (Finset.range 3000).sum (fun n => b_n (n + 1)) = 581 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_n_equals_581_l1299_129978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_second_year_books_l1299_129921

/-- The number of books Matt read in the first year -/
def matt_year1 : ℕ := 50

/-- The number of books Pete read in the first year -/
def pete_year1 : ℕ := 2 * matt_year1

/-- The number of books Matt read in the second year -/
def matt_year2 : ℕ := matt_year1 + (matt_year1 / 2)

/-- The number of books Pete read in the second year -/
def pete_year2 : ℕ := 2 * pete_year1

/-- The total number of books Pete read across both years -/
def pete_total : ℕ := 300

theorem matt_second_year_books :
  pete_year1 + pete_year2 = pete_total →
  matt_year2 = 75 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_second_year_books_l1299_129921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_of_30_60_90_triangles_l1299_129924

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- The area of the overlapping region of two congruent 30-60-90 triangles -/
noncomputable def overlapping_area (t : Triangle30_60_90) (overlap : ℝ) : ℝ :=
  (25 * Real.sqrt 3) / 4

/-- Theorem stating the area of the overlapping region of two congruent 30-60-90 triangles -/
theorem overlapping_area_of_30_60_90_triangles
  {t1 t2 : Triangle30_60_90}
  {overlap : ℝ}
  (h_congruent : t1 = t2)
  (h_hypotenuse : t1.hypotenuse = 10)
  (h_overlap : overlap = 5) :
  overlapping_area t1 overlap = (25 * Real.sqrt 3) / 4 := by
  sorry

#check overlapping_area_of_30_60_90_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_of_30_60_90_triangles_l1299_129924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1299_129953

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1/2) + 3/(x+1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1299_129953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_savings_calculation_l1299_129973

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem initial_savings_calculation
  (simple_principal : ℝ)
  (compound_principal : ℝ)
  (simple_rate : ℝ)
  (compound_rate : ℝ)
  (time : ℝ)
  (simple_interest_earned : ℝ)
  (compound_interest_earned : ℝ)
  (h1 : simple_principal = compound_principal)
  (h2 : simple_rate = 0.04)
  (h3 : compound_rate = 0.05)
  (h4 : time = 5)
  (h5 : simple_interest_earned = 575)
  (h6 : compound_interest_earned = 635)
  (h7 : simple_interest_earned = simple_interest simple_principal simple_rate time)
  (h8 : compound_interest_earned = compound_interest compound_principal compound_rate time) :
  simple_principal + compound_principal = 5750 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_savings_calculation_l1299_129973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l1299_129938

/-- Represents the number of sides in a convex polygon --/
def n : ℕ := 12

/-- Represents the common difference between consecutive angles in degrees --/
def d : ℚ := 10

/-- Represents the largest angle in the polygon in degrees --/
def largest_angle : ℚ := 150

/-- Represents the smallest angle in the polygon in degrees --/
def smallest_angle : ℚ := largest_angle - d * (n - 1)

/-- The sum of interior angles of a polygon with n sides --/
def sum_interior_angles : ℚ := 180 * (n - 2)

/-- The sum of angles in arithmetic progression --/
def sum_arithmetic_progression : ℚ := n * (smallest_angle + largest_angle) / 2

theorem polygon_sides : 
  sum_interior_angles = sum_arithmetic_progression ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l1299_129938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_sales_after_returns_l1299_129957

/-- Calculates the total sales after returns for a bookstore --/
theorem bookstore_sales_after_returns 
  (total_customers : ℕ) 
  (return_rate : ℚ) 
  (book_price : ℚ) 
  (h1 : total_customers = 1000)
  (h2 : return_rate = 37 / 100)
  (h3 : book_price = 15)
  : 
  (total_customers : ℚ) * (1 - return_rate) * book_price = 9450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_sales_after_returns_l1299_129957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l1299_129967

noncomputable section

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 9 * x^2 + 4 * y^2 = 36

-- Define the new ellipse
def new_ellipse (x y : ℝ) : Prop := y^2 / 16 + x^2 / 11 = 1

-- Function to calculate the distance between foci
noncomputable def foci_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

theorem ellipse_proof :
  -- The new ellipse passes through (0, 4)
  new_ellipse 0 4 ∧
  -- The new ellipse has the same foci as the given ellipse
  foci_distance 3 2 = foci_distance 4 (Real.sqrt 11) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l1299_129967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l1299_129991

def spinner_numbers : List Nat := [2, 3, 4, 5, 7, 9, 10, 11]

def is_prime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def count_primes (l : List Nat) : Nat :=
  (l.filter is_prime).length

theorem spinner_prime_probability :
  let total_sections := spinner_numbers.length
  let prime_sections := count_primes spinner_numbers
  (prime_sections : Rat) / total_sections = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l1299_129991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_two_digit_is_five_ninths_l1299_129955

/-- The number of cards in the set -/
def num_cards : ℕ := 4

/-- The set of cards -/
def card_set : Finset ℕ := {0, 1, 2, 3}

/-- Function to check if a two-digit number is even -/
def is_even_two_digit (a b : ℕ) : Bool := (10 * a + b) % 2 = 0

/-- The probability of drawing two cards to form an even two-digit number -/
noncomputable def prob_even_two_digit : ℚ :=
  (Finset.filter (fun (pair : ℕ × ℕ) => is_even_two_digit pair.1 pair.2)
    (card_set.product card_set)).card /
  (card_set.product card_set).card

theorem prob_even_two_digit_is_five_ninths :
  prob_even_two_digit = 5/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_two_digit_is_five_ninths_l1299_129955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_gender_relation_and_probability_l1299_129910

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ := 
  Matrix.of ![
    ![300, 140],
    ![280, 180]
  ]

-- Define the total number of students
def total_students : ℕ := 900

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Define the K^2 formula
noncomputable def K_squared (table : Matrix (Fin 2) (Fin 2) ℕ) : ℚ :=
  let a := table 0 0
  let b := table 0 1
  let c := table 1 0
  let d := table 1 1
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the probability calculation function
def probability_same_combination (group1_size : ℕ) (group2_size : ℕ) : ℚ :=
  let total_size := group1_size + group2_size
  let favorable_outcomes := group1_size.choose 2 + group2_size.choose 2
  let total_outcomes := total_size.choose 2
  (favorable_outcomes : ℚ) / total_outcomes

-- State the theorem
theorem physics_gender_relation_and_probability :
  K_squared contingency_table < critical_value ∧
  probability_same_combination 4 2 = 7 / 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_gender_relation_and_probability_l1299_129910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l1299_129956

def total_members : ℕ := 18
def boys : ℕ := 8
def girls : ℕ := 10
def committee_size : ℕ := 3

def probability_mixed_committee : ℚ := 40 / 51

theorem mixed_committee_probability :
  (Nat.choose total_members committee_size - Nat.choose boys committee_size - Nat.choose girls committee_size : ℚ) /
  Nat.choose total_members committee_size = probability_mixed_committee := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l1299_129956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_101st_term_l1299_129987

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 1

theorem sequence_101st_term (a : ℕ → ℝ) (h : arithmeticSequence a) : a 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_101st_term_l1299_129987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l1299_129917

noncomputable def data : List ℝ := [198, 199, 200, 201, 202]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l1299_129917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_min_positive_period_of_f_range_of_m_l1299_129976

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * cos (x + π/3) * (sin (x + π/3) - sqrt 3 * cos (x + π/3))

-- Theorem for the range of f(x)
theorem range_of_f : 
  Set.range f = Set.Icc (-2 - sqrt 3) (2 - sqrt 3) :=
sorry

-- Theorem for the minimum positive period of f(x)
theorem min_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

-- Theorem for the range of m
theorem range_of_m : 
  {m : ℝ | ∃ (x : ℝ), x ∈ Set.Icc 0 (π/6) ∧ m * (f x + sqrt 3) + 2 = 0} = 
  Set.Icc (-2 * sqrt 3 / 3) (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_min_positive_period_of_f_range_of_m_l1299_129976
