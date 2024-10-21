import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l166_16678

def f (n : ℕ+) : ℚ := (Finset.range n).sum (fun i => 1 / ((i + 1 : ℕ) : ℚ) ^ 3)

def g (n : ℕ+) : ℚ := (1 / 2) * (3 - 1 / ((n : ℚ) ^ 2))

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l166_16678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_three_less_than_three_over_e_l166_16668

-- Define e as the base of natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem to be proved
theorem ln_three_less_than_three_over_e : Real.log 3 < 3 / e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_three_less_than_three_over_e_l166_16668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_cycling_time_l166_16661

/-- Represents the time in minutes to cycle a given distance at a constant speed -/
noncomputable def cyclingTime (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

theorem library_cycling_time 
  (speed : ℝ) 
  (park_distance : ℝ) 
  (library_distance : ℝ) 
  (park_time : ℝ) 
  (h1 : speed > 0)
  (h2 : park_distance = 5)
  (h3 : library_distance = 3)
  (h4 : park_time = 30)
  (h5 : cyclingTime speed park_distance = park_time) :
  cyclingTime speed library_distance = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_cycling_time_l166_16661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l166_16686

-- Define the function f
noncomputable def f (k a x : ℝ) : ℝ := k * (a^x) - a^(-x)

-- State the theorem
theorem function_properties
  (a k : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x, f k a x = -f k a (-x)) -- f is odd
  : 
  -- Part 1: k = 1
  (k = 1) ∧
  -- Part 2: If f(1) > 0, then solution set of f(x^2 + 2x) + f(x - 4) > 0 is {x | x > 1 ∨ x < -4}
  (f k a 1 > 0 → ∀ x, f k a (x^2 + 2*x) + f k a (x - 4) > 0 ↔ (x > 1 ∨ x < -4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l166_16686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l166_16619

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3/4

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  -- Equation of the axis of symmetry
  (∀ x, ∃ k : ℤ, f (5*Real.pi/12 + k*Real.pi/2 + x) = f (5*Real.pi/12 + k*Real.pi/2 - x)) ∧
  -- Range of x for which f(x) ≥ 1/4
  (∀ x, f x ≥ 1/4 ↔ ∃ k : ℤ, Real.pi/4 + k*Real.pi ≤ x ∧ x ≤ 7*Real.pi/12 + k*Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l166_16619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_one_l166_16648

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 / 2) * mySequence n + 1 / 2

theorem fourth_term_is_one : mySequence 3 = 1 := by
  -- Unfold the definition of mySequence
  unfold mySequence
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval mySequence 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_one_l166_16648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_21_over_20_l166_16630

/-- Profit function for two products given investment x -/
noncomputable def profit (x : ℝ) : ℝ :=
  (1/5) * (3 - x) + (3/5) * Real.sqrt x

/-- Theorem stating the maximum profit -/
theorem max_profit_is_21_over_20 :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 3 → profit y ≤ profit x) ∧
  profit x = 21/20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_21_over_20_l166_16630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_light_properties_main_result_l166_16616

/-- Represents the probability of encountering a red light at each intersection -/
structure IntersectionProbabilities where
  first : ℝ
  second : ℝ
  third : ℝ

/-- The probabilities of encountering red lights at the intersections -/
noncomputable def red_light_probs : IntersectionProbabilities :=
  { first := 1/2,
    second := 2/3,
    third := 3/4 }

/-- Theorem stating the properties of the red light probabilities -/
theorem red_light_properties (p : IntersectionProbabilities) :
  p.first = 1/2 ∧
  p.second > p.first ∧
  p.third > p.second ∧
  (1 - p.first) * (1 - p.second) * (1 - p.third) = 1/24 ∧
  p.first * p.second * p.third = 1/4 →
  (1 - p.first) * (1 - p.second) * p.third = 1/8 ∧
  (0 * 1/24 + 1 * 1/4 + 2 * 11/24 + 3 * 1/4 : ℝ) = 23/12 := by
  sorry

/-- The main theorem combining all results -/
theorem main_result :
  (1 - red_light_probs.first) * (1 - red_light_probs.second) * red_light_probs.third = 1/8 ∧
  (0 * 1/24 + 1 * 1/4 + 2 * 11/24 + 3 * 1/4 : ℝ) = 23/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_light_properties_main_result_l166_16616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inf_complex_polynomial_l166_16675

theorem inf_complex_polynomial (a : ℝ) (h : a ∈ Set.Icc (2 + Real.sqrt 2) 4) :
  ∃ (z : ℂ), ∀ (w : ℂ), Complex.abs w ≤ 1 →
    Complex.abs (w ^ 2 - a * w + a) ≥ Complex.abs (z ^ 2 - a * z + a) ∧
    Complex.abs (z ^ 2 - a * z + a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inf_complex_polynomial_l166_16675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grazing_area_at_B_l166_16659

/-- Represents the side length of the square pond -/
noncomputable def pondSideLength : ℝ := 12

/-- Represents the distance between adjacent stakes -/
noncomputable def stakeSeparation : ℝ := 3

/-- Represents the length of the rope -/
noncomputable def ropeLength : ℝ := 4

/-- Calculates the grazing area when the goat is tethered to stake A or C -/
noncomputable def grazingAreaAC : ℝ := (1/2 * Real.pi * ropeLength^2) + (1/4 * Real.pi * (ropeLength - stakeSeparation)^2)

/-- Calculates the grazing area when the goat is tethered to stake B -/
noncomputable def grazingAreaB : ℝ := 3/4 * Real.pi * ropeLength^2

/-- Calculates the grazing area when the goat is tethered to stake D -/
noncomputable def grazingAreaD : ℝ := 1/2 * Real.pi * ropeLength^2

/-- Theorem stating that the grazing area is maximized when the goat is tethered to stake B -/
theorem max_grazing_area_at_B :
  grazingAreaB > grazingAreaAC ∧ grazingAreaB > grazingAreaD :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grazing_area_at_B_l166_16659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_is_correct_l166_16639

/-- The volume of the solid obtained by rotating the region bounded by y=x^3 and y=x about the y-axis -/
noncomputable def rotationVolume : ℝ := (4 / 15) * Real.pi

/-- The lower bound of the region -/
def lowerBound (y : ℝ) : ℝ := y

/-- The upper bound of the region -/
def upperBound (y : ℝ) : ℝ := y^(1/3)

theorem rotation_volume_is_correct : 
  rotationVolume = Real.pi * ∫ y in (0)..(1), (upperBound y)^2 - (lowerBound y)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_is_correct_l166_16639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_good_numbers_l166_16679

/-- A positive integer is a good number if it's divisible by the squares of all its prime factors -/
def is_good_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p^2 ∣ n

/-- There exist infinitely many pairs of consecutive good numbers -/
theorem infinitely_many_consecutive_good_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_good_number n ∧ is_good_number (n + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_good_numbers_l166_16679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l166_16664

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : ℝ  -- y-coordinate of the directrix

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Distance between intersection points of parabola and ellipse -/
theorem intersection_distance
  (e : Ellipse)
  (p : Parabola)
  (h1 : e.center = Point.mk 0 0)
  (h2 : e.a = 6 ∧ e.b = 4)
  (h3 : p.directrix = 0)
  (h4 : p.focus.x = 2 * Real.sqrt 5 ∧ p.focus.y = 0)
  (i1 i2 : Point)
  (h5 : i1.x^2 / 16 + i1.y^2 / 36 = 1)
  (h6 : i2.x^2 / 16 + i2.y^2 / 36 = 1)
  (h7 : i1.x = 4 * i1.y^2 + 2 * Real.sqrt 5)
  (h8 : i2.x = 4 * i2.y^2 + 2 * Real.sqrt 5)
  : distance i1 i2 = 2 * |i1.y - i2.y| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l166_16664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l166_16618

theorem number_of_subsets (M : Finset ℕ) : M = {1, 2, 3} → Finset.card (Finset.powerset M) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l166_16618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l166_16622

/-- Given an angle α with vertex at the origin, initial side on the non-negative x-axis,
    and terminal side passing through (1, 2), prove that cos(π - α) = -√5/5 -/
theorem cos_pi_minus_alpha (α : ℝ) : 
  (∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.cos (Real.pi - α) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l166_16622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l166_16672

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := 1 / x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x
noncomputable def g' (x : ℝ) : ℝ := -1 / (x^2)

theorem tangent_perpendicular :
  ∃! P : ℝ × ℝ, 
    P.1 > 0 ∧ 
    P.2 = g P.1 ∧ 
    f' 0 * g' P.1 = -1 ∧ 
    P = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l166_16672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_ellipse_l166_16687

-- Define the line 4x + 3y = 24
def line (x y : ℝ) : Prop := 4 * x + 3 * y = 24

-- Define the ellipse (x²/8) + (y²/4) = 1
def ellipse (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 4) = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_line_ellipse :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ ellipse x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ),
      line x₃ y₃ ∧ ellipse x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄ ∧
      distance x₁ y₁ x₂ y₂ = (24 - 2 * Real.sqrt 41) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_ellipse_l166_16687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deeper_side_depth_is_four_l166_16613

/-- Represents a swimming pool with trapezoidal cross-section --/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  shallowDepth : ℝ
  volume : ℝ

/-- Calculates the depth of the deeper side of the swimming pool --/
noncomputable def deeperSideDepth (pool : SwimmingPool) : ℝ :=
  2 * pool.volume / (pool.width * pool.length) - pool.shallowDepth

/-- Theorem stating that for the given swimming pool dimensions, the deeper side depth is 4 meters --/
theorem deeper_side_depth_is_four :
  let pool : SwimmingPool := {
    width := 9,
    length := 12,
    shallowDepth := 1,
    volume := 270
  }
  deeperSideDepth pool = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deeper_side_depth_is_four_l166_16613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_l166_16658

theorem tan_product (x y : ℝ) 
  (h1 : Real.sin x * Real.sin y = 24 / 65)
  (h2 : Real.cos x * Real.cos y = 48 / 65) : 
  Real.tan x * Real.tan y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_l166_16658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l166_16638

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 2 * x^2 + x^2 * Real.cos (1 / (9 * x))
  else 0

-- State the theorem
theorem derivative_f_at_zero :
  HasDerivAt f 0 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l166_16638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_ratio_l166_16680

theorem cube_side_ratio (v1 v2 : ℝ) (h : v1 / v2 = 512 / 216) :
  ∃ (m n p : ℕ), 
    (m : ℝ) * Real.sqrt n / p = (v1 / v2) ^ (1/3 : ℝ) ∧
    m = 4 ∧ n = 1 ∧ p = 3 ∧
    m + n + p = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_ratio_l166_16680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_meeting_problem_l166_16657

/-- The duration of stay at the park -/
noncomputable def n : ℝ := 60 - 50 * Real.sqrt 1008

/-- The probability of the two friends meeting -/
def meeting_probability : ℝ := 0.3

/-- The total time range for arrivals in minutes -/
def total_time : ℕ := 60

theorem park_meeting_problem :
  let p : ℕ := 60
  let q : ℕ := 50
  let r : ℕ := 1008
  (1 - (total_time - n)^2 / (total_time^2 : ℝ) = meeting_probability) ∧
  (n = p - q * Real.sqrt r) ∧
  (p + q + r = 1118) := by
  sorry

#eval meeting_probability
#eval total_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_meeting_problem_l166_16657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l166_16670

/-- Predicate to determine if a given equation represents an ellipse -/
def IsEllipse (x y : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1

/-- The range of k for which x^2 + ky^2 = 2 represents an ellipse -/
theorem ellipse_k_range : 
  ∀ k : ℝ, (∀ x y : ℝ, x^2 + k*y^2 = 2 → IsEllipse x y) ↔ 
  (k ∈ Set.Ioo 0 1 ∪ Set.Ioi 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l166_16670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_proof_l166_16626

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := my_circle P.1 P.2

-- Define symmetry of two points about a line
def symmetric_points (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  symmetry_line m ((P.1 + Q.1)/2) ((P.2 + Q.2)/2)

-- Define dot product of two points
def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

theorem circle_symmetry_line_proof (P Q : ℝ × ℝ) (m : ℝ) :
  point_on_circle P ∧ point_on_circle Q ∧
  symmetric_points P Q m ∧
  dot_product P Q = 0 →
  m = -1 ∧ Q.2 = -Q.1 + 1 ∧ P.2 = -P.1 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_proof_l166_16626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_600_l166_16621

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Calculates the distance between two runners at a given time -/
noncomputable def distance_between (track : CircularTrack) (time : ℝ) : ℝ :=
  (track.runner1_speed + track.runner2_speed) * time % track.length

/-- Theorem: The track length is 600 meters given the specified conditions -/
theorem track_length_is_600 (track : CircularTrack) :
  (∃ t1 t2 : ℝ, 
    t1 > 0 ∧ t2 > t1 ∧
    track.runner1_speed * t1 = 120 ∧
    distance_between track t1 = 0 ∧
    track.runner2_speed * (t2 - t1) = 180 ∧
    distance_between track t2 = 0) →
  track.length = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_600_l166_16621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l166_16681

theorem round_to_nearest_integer (x : ℝ) (h : x = 5278132.764501) :
  ⌊x + 0.5⌋ = 5278133 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l166_16681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l166_16688

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d

theorem sequence_properties (q : ℝ) (h : q > 0) :
  let a : ℕ → ℝ := fun n ↦ q ^ n
  (is_geometric_sequence (fun n ↦ a (2 * n))) ∧
  (is_geometric_sequence (fun n ↦ 1 / a n)) ∧
  (is_arithmetic_sequence (fun n ↦ Real.log (a n))) ∧
  (is_arithmetic_sequence (fun n ↦ Real.log ((a n) ^ 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l166_16688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l166_16601

/-- Line l parametric equation -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, (Real.sqrt 3 * t) / 2)

/-- Curve C parametric equation -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos θ, 2 * Real.sqrt 3 + 2 * Real.sin θ)

/-- Point P in polar coordinates -/
noncomputable def point_P : ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.pi / 3)

/-- Calculate the area of a triangle given three points -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle PAB is 3√13/2 -/
theorem area_of_triangle_PAB :
  ∃ A B : ℝ × ℝ,
  (area_triangle point_P A B) = (3 * Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l166_16601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_five_zeros_l166_16645

noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

def f (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def F (x : ℝ) : ℝ := sgn (f x) - f x

theorem F_has_five_zeros :
  ∃ (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, F x = 0 ∧
  ∀ y : ℝ, F y = 0 → y ∈ s := by
  sorry

#check F_has_five_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_five_zeros_l166_16645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_total_distance_fuel_consumption_l166_16629

-- Define the travel records
def travel_records : List Int := [-9, 7, -3, -6, -8, 5]

-- Define the fuel consumption rate
def fuel_rate : Float := 0.1

-- Theorem for final position
theorem final_position : 
  (travel_records.sum : Int) = -14 := by sorry

-- Theorem for total distance
theorem total_distance : 
  (travel_records.map Int.natAbs).sum = 38 := by sorry

-- Theorem for fuel consumption
theorem fuel_consumption : 
  ((travel_records.map Int.natAbs).sum.toFloat * fuel_rate : Float) = 3.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_total_distance_fuel_consumption_l166_16629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l166_16677

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := 
  (7 * x^3 - 4 * x^2 - 32 * x - 37) / ((x + 2) * (2 * x - 1) * (x^2 + 2 * x + 3))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 
  3 * log (abs (x + 2)) - (5/2) * log (abs (2 * x - 1)) + 
  (3/2) * log (abs (x^2 + 2 * x + 3)) - 
  (4 / Real.sqrt 2) * arctan ((x + 1) / Real.sqrt 2)

-- State the theorem
theorem integral_equality (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 1/2) (h3 : x^2 + 2*x + 3 ≠ 0) : 
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l166_16677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l166_16605

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x - 1 -/
  asymptote1 : ℝ → ℝ := fun x ↦ 2 * x - 1
  /-- Second asymptote: y = -2x + 7 -/
  asymptote2 : ℝ → ℝ := fun x ↦ -2 * x + 7
  /-- The hyperbola passes through the point (5,5) -/
  passes_through : (5, 5) ∈ {p : ℝ × ℝ | ∃ a b : ℝ, (p.1 - 2)^2 / a^2 - (p.2 - 3)^2 / b^2 = 1}

/-- The distance between the foci of the hyperbola is √34 -/
theorem hyperbola_foci_distance (h : Hyperbola) : 
  ∃ f1 f2 : ℝ × ℝ, Real.sqrt 34 = Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l166_16605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l166_16652

/-- The parametric equations defining the curve -/
noncomputable def x (t : ℝ) : ℝ := 3 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 3 * (1 - Real.cos t)

/-- The bounds for x and y -/
def x_bounds (t : ℝ) : Prop := 0 < x t ∧ x t < 6 * Real.pi
def y_bounds (t : ℝ) : Prop := y t ≥ 3

/-- The theorem stating the area of the bounded figure -/
theorem area_of_bounded_figure :
  ∃ (a b : ℝ), a < b ∧
  (∀ t ∈ Set.Icc a b, x_bounds t ∧ y_bounds t) ∧
  (∫ (t : ℝ) in a..b, y t * (deriv x t)) = 9 * Real.pi + 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l166_16652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_hundredth_digit_of_fraction_l166_16683

-- Define the fraction
def fraction : ℚ := 5 / 13

-- Define the length of the repeating decimal
def repeat_length : ℕ := 6

-- Define the repeating sequence of digits
def repeat_sequence : List ℕ := [3, 8, 4, 6, 1, 5]

-- The theorem to prove
theorem three_hundredth_digit_of_fraction (n : ℕ) (h : n = 300) : 
  (repeat_sequence.get! ((n - 1) % repeat_length)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_hundredth_digit_of_fraction_l166_16683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_integer_between_square_roots_l166_16689

theorem cube_root_of_integer_between_square_roots (a : ℤ) 
  (h1 : (59 : ℝ).sqrt < a) 
  (h2 : a < (65 : ℝ).sqrt) : 
  (a : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_integer_between_square_roots_l166_16689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l166_16637

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Define the length of a vector
noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem quadrilateral_area (q : Quadrilateral) :
  (vector_length (vector_sub q.B q.D) = 2) →
  (dot_product (vector_sub q.A q.C) (vector_sub q.B q.D) = 0) →
  (dot_product (vector_add (vector_sub q.A q.B) (vector_sub q.D q.C))
               (vector_add (vector_sub q.B q.C) (vector_sub q.A q.D)) = 5) →
  area q = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l166_16637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bankers_discount_problem_l166_16671

/-- Banker's discount formula -/
noncomputable def bankers_discount (amount : ℝ) (true_discount : ℝ) : ℝ :=
  (amount * true_discount) / (amount - true_discount)

/-- The problem statement -/
theorem total_bankers_discount_problem :
  let a₁ : ℝ := 2260
  let a₂ : ℝ := 3280
  let a₃ : ℝ := 4510
  let a₄ : ℝ := 6240
  let td₁ : ℝ := 360
  let td₂ : ℝ := 520
  let td₃ : ℝ := 710
  let td₄ : ℝ := 980
  abs ((bankers_discount a₁ td₁ + bankers_discount a₂ td₂ + 
        bankers_discount a₃ td₃ + bankers_discount a₄ td₄) - 3050.96) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bankers_discount_problem_l166_16671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l166_16665

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 3 - 2*t)

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.cos α, 2 + Real.cos (2*α))

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem unique_intersection :
  (∃! p : ℝ × ℝ, (∃ t : ℝ, line_l t = p) ∧ (∃ α : ℝ, α ∈ Set.Icc 0 (2*Real.pi) ∧ curve_C α = p)) ∧
  (∃ t : ℝ, line_l t = intersection_point) ∧
  (∃ α : ℝ, α ∈ Set.Icc 0 (2*Real.pi) ∧ curve_C α = intersection_point) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l166_16665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_b_l166_16641

/-- The minimum value of |a| + 2|b| given the conditions -/
theorem min_value_a_b : 
  ∃ (a b : ℝ), (∃ (x₁ x₂ x₃ : ℝ), 
    x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1 ∧
    (fun x ↦ x^3 + a*x^2 + b*x) x₁ = (fun x ↦ x^3 + a*x^2 + b*x) x₂ ∧
    (fun x ↦ x^3 + a*x^2 + b*x) x₂ = (fun x ↦ x^3 + a*x^2 + b*x) x₃) ∧
  (∀ a' b' : ℝ, |a'| + 2*|b'| ≥ Real.sqrt 3) ∧
  |a| + 2*|b| = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_b_l166_16641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l166_16690

theorem min_value_A_squared_minus_B_squared (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 4) + Real.sqrt (z + 7)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l166_16690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l166_16610

-- Define the function f(x) = √(x-2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ 2}

-- Theorem stating that the domain of f is {x ∈ ℝ | x ≥ 2}
theorem domain_of_sqrt_x_minus_two :
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l166_16610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_A_sequence_l166_16624

def A (n : ℕ) : ℕ := 2^(3*n) + 3^(6*n + 2) + 5^(6*n + 2)

theorem gcd_of_A_sequence : 
  Nat.gcd (A 0) (Nat.gcd (A 1) (Nat.gcd (A 2) (Nat.gcd (A 3) (A 1999)))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_A_sequence_l166_16624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l166_16603

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_theorem (u : ℝ × ℝ) : 
  vector_projection (1, 2) u = (3/5, 6/5) →
  vector_projection (3, -4) u = (-1, -2) ∧
  vector_projection (-1, -2) (3, 1) = (-3/2, -1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l166_16603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_l166_16609

/-- A cubic function with integer coefficients -/
def cubic_function (a b c : ℤ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + c

theorem cubic_sum (a b c : ℤ) :
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  cubic_function a b c a = a^3 ∧
  cubic_function a b c b = b^3 →
  a + b + c = 16 := by
  sorry

#check cubic_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_l166_16609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_OZ_l166_16632

noncomputable def z : ℂ := 1 - Complex.I

noncomputable def OZ : ℂ := 2 / z + z^2

theorem magnitude_of_OZ : Complex.abs OZ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_OZ_l166_16632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_and_g_max_t_l166_16692

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2 * x + Real.log (x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^3 + (1/2) * m * x^2 - 2 * x

theorem f_local_minimum_and_g_max_t (m : ℝ) :
  (∀ x > -1, ∃ y, f m x = y) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f (3/2) x ≥ f (3/2) 1) ∧
  (m ∈ Set.Icc (-4) (-1) →
    (∀ t > 1, (∀ x ∈ Set.Icc 1 t, g m x ≤ g m 1) →
      t ≤ (1 + Real.sqrt 13) / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_and_g_max_t_l166_16692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_sum_l166_16651

/-- A configuration of numbers on an 8x8 chess board -/
def BoardConfig := Fin 8 → Fin 8 → Fin 64

/-- Predicate to check if two positions share an edge -/
def adjacent (p q : Fin 8 × Fin 8) : Prop :=
  (p.1 = q.1 ∧ p.2.val + 1 = q.2.val) ∨
  (p.1 = q.1 ∧ p.2.val = q.2.val + 1) ∨
  (p.1.val + 1 = q.1.val ∧ p.2 = q.2) ∨
  (p.1.val = q.1.val + 1 ∧ p.2 = q.2)

/-- Predicate to check if a configuration is valid -/
def valid_config (c : BoardConfig) : Prop :=
  ∀ i j : Fin 8, ∃ p q : Fin 8 × Fin 8,
    c p.1 p.2 = i ∧ c q.1 q.2 = j ∧
    (i.val + 1 = j.val → adjacent p q)

/-- The sum of numbers along a diagonal -/
def diagonal_sum (c : BoardConfig) : ℕ :=
  (Finset.sum (Finset.range 8) (fun i => (c i i).val + 1)) +
  (Finset.sum (Finset.range 8) (fun i => (c i (7 - i)).val + 1))

/-- The main theorem -/
theorem max_diagonal_sum (c : BoardConfig) (h : valid_config c) :
  diagonal_sum c ≤ 432 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_sum_l166_16651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_a_lt_neg_two_l166_16640

/-- The function f(x) = (ax + 2) / (x - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 2) / (x - 1)

/-- The theorem stating that f is strictly increasing on [2, +∞) iff a < -2 -/
theorem f_strictly_increasing_iff_a_lt_neg_two (a : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x < y → f a x < f a y) ↔ a < -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_a_lt_neg_two_l166_16640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_expression_l166_16669

def prod_terms (n : ℕ) : ℕ := 2^n + 1

def expression : ℕ := 3 * (List.prod (List.map (λ n => prod_terms (2*n)) (List.range 16))) + 1

theorem unit_digit_of_expression :
  expression % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_expression_l166_16669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_l166_16667

open Real

-- Define the set of points in polar coordinates satisfying θ = π/4
def polar_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = π/4 ∨ p.2 = 5*π/4}

-- State the theorem
theorem polar_set_is_line :
  ∃ (a b : ℝ), a ≠ 0 ∧ 
  (∀ (x y : ℝ), (x, y) ∈ polar_set ↔ a * x + b * y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_l166_16667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_slope_l166_16694

/-- The area of a circle with radius r -/
noncomputable def area_circle (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a circle segment defined by a line y = mx + b -/
noncomputable def area_circle_segment (r : ℝ) (c : ℝ × ℝ) (x y : ℝ) : ℝ := 
  sorry -- Definition of circle segment area

/-- Two circles with radius 4 and centers at (20, 100) and (25, 90) are bisected by a line passing through (20, 90). The absolute value of the slope of this line is 2. -/
theorem bisecting_line_slope (r : ℝ) (c1 c2 p : ℝ × ℝ) :
  r = 4 ∧ 
  c1 = (20, 100) ∧ 
  c2 = (25, 90) ∧ 
  p = (20, 90) ∧ 
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - p.2 = m * (x - p.1) → 
      (area_circle_segment r c1 x y = area_circle r - area_circle_segment r c1 x y) ∧
      (area_circle_segment r c2 x y = area_circle r - area_circle_segment r c2 x y)) →
    |m| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_slope_l166_16694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l166_16646

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

-- Define the eccentricity
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

-- Define the vertices
def left_vertex (e : Ellipse) : ℝ × ℝ := (-e.a, 0)
def right_vertex (e : Ellipse) : ℝ × ℝ := (e.a, 0)
def top_vertex (e : Ellipse) : ℝ × ℝ := (0, e.b)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem ellipse_equation (e : Ellipse) 
  (h_ecc : eccentricity e = 1/3)
  (h_dot : dot_product 
    (left_vertex e - top_vertex e) 
    (right_vertex e - top_vertex e) = -1) : 
  e.a^2 = 9 ∧ e.b^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l166_16646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_proof_f_max_at_smallest_max_x_smallest_max_x_is_smallest_l166_16684

/-- The function f(x) defined as sin(x/4) + sin(x/12) --/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

/-- The smallest positive value of x in degrees for which f(x) achieves its maximum value --/
def smallest_max_x : ℝ := 13230

theorem smallest_max_x_proof :
  ∀ y : ℝ, y > 0 → y < smallest_max_x →
    ∃ z : ℝ, z > 0 ∧ f z > f y :=
by sorry

theorem f_max_at_smallest_max_x :
  ∀ y : ℝ, y > 0 → f y ≤ f smallest_max_x :=
by sorry

theorem smallest_max_x_is_smallest :
  ∀ y : ℝ, y > 0 → y < smallest_max_x →
    ¬(∀ z : ℝ, z > 0 → f z ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_proof_f_max_at_smallest_max_x_smallest_max_x_is_smallest_l166_16684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentachoron_intersection_forms_desargues_configuration_l166_16614

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a pentachoron (5-vertex figure)
structure Pentachoron where
  vertices : Fin 5 → Point3D
  no_four_coplanar : ∀ (a b c d : Fin 5), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ∃ (n : Point3D) (k : ℝ), n.x * (vertices a).x + n.y * (vertices a).y + n.z * (vertices a).z + k ≠ 0 ∨
                              n.x * (vertices b).x + n.y * (vertices b).y + n.z * (vertices b).z + k ≠ 0 ∨
                              n.x * (vertices c).x + n.y * (vertices c).y + n.z * (vertices c).z + k ≠ 0 ∨
                              n.x * (vertices d).x + n.y * (vertices d).y + n.z * (vertices d).z + k ≠ 0

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the Desargues configuration
structure DesarguesConfiguration where
  points : Fin 10 → Point3D
  lines : Fin 10 → Set Point3D
  -- Add more properties to fully define the Desargues configuration

-- Define the intersection of a pentachoron with a plane
noncomputable def intersectionWithPlane (p : Pentachoron) (plane : Plane) : Set Point3D :=
  sorry

-- Helper function to create a point on a line
noncomputable def pointOnLine (a b : Point3D) (t : ℝ) : Point3D :=
  { x := a.x + t * (b.x - a.x)
  , y := a.y + t * (b.y - a.y)
  , z := a.z + t * (b.z - a.z) }

-- The main theorem
theorem pentachoron_intersection_forms_desargues_configuration
  (p : Pentachoron) (plane : Plane)
  (intersects_all_lines : ∀ (a b : Fin 5), a ≠ b →
    ∃ (point : Point3D) (t : ℝ), point ∈ intersectionWithPlane p plane ∧
      point = pointOnLine (p.vertices a) (p.vertices b) t) :
  ∃ (dc : DesarguesConfiguration), intersectionWithPlane p plane = Set.range dc.points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentachoron_intersection_forms_desargues_configuration_l166_16614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_R_l166_16608

open Real Set

noncomputable section

variable (O : ℝ × ℝ)

def UnitCircle (c : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | dist p c = 1}

def PerpendicularBisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | dist p A = dist p B}

theorem locus_of_R (k : Set (ℝ × ℝ)) (P : ℝ × ℝ) 
  (hk : k = UnitCircle O) (hP : P ∈ k) :
  ∀ R ∈ PerpendicularBisector P O,
  ∃ Q ∈ k, dist P Q * dist P R = 1 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_R_l166_16608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l166_16676

def A : Finset ℕ := {1, 2}

theorem proper_subsets_count :
  (Finset.filter (λ s : Finset ℕ => s ⊂ A) (Finset.powerset A)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l166_16676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_ratio_l166_16635

-- Define the triangles and their properties
structure EquilateralTriangle where
  side : ℝ
  area : ℝ
  area_eq : area = (Real.sqrt 3 / 4) * side ^ 2

-- Define the problem setup
def problem_setup (ABC DBE IEF HIG : EquilateralTriangle) : Prop :=
  DBE.area / IEF.area = 9 / 16 ∧ IEF.area / HIG.area = 16 / 4

-- Define the theorem to be proved
theorem equilateral_triangles_ratio 
  (ABC DBE IEF HIG : EquilateralTriangle) 
  (h : problem_setup ABC DBE IEF HIG) :
  (HIG.side / IEF.side = 1 / 2) ∧ 
  (ABC.area / (ABC.area - HIG.area - IEF.area) = 9 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_ratio_l166_16635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l166_16600

theorem angle_terminal_side_point (α : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ k * Real.sin α = Real.sin (7*π/6) ∧ k * Real.cos α = Real.cos (11*π/6)) →
  1 / (3 * Real.sin α^2 - Real.cos α^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l166_16600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l166_16627

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.cos (ω * x)

-- State the theorem
theorem function_properties (ω : ℝ) (h₁ : ω > 0) (h₂ : ∀ x, f ω (x + π) = f ω x) :
  (∀ k : ℤ, ∃ x₀, ∀ x, f ω x = f ω (2 * x₀ - x) → x = k * π / 2 + 3 * π / 8) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (3 * π / 8), ∀ y ∈ Set.Icc (0 : ℝ) (3 * π / 8), x ≤ y → f ω x ≤ f ω y) ∧
  (∀ x ∈ Set.Icc (3 * π / 8 : ℝ) (π / 2), ∀ y ∈ Set.Icc (3 * π / 8 : ℝ) (π / 2), x ≤ y → f ω x ≥ f ω y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l166_16627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_picked_43_apples_l166_16666

/-- The number of apples Joan picked from the orchard -/
def apples_picked : ℕ := 43

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has left -/
def apples_left : ℕ := 16

/-- Theorem stating that Joan picked 43 apples from the orchard -/
theorem joan_picked_43_apples : apples_picked = apples_given + apples_left := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_picked_43_apples_l166_16666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sequence_l166_16636

/-- Sum of digits of a positive integer -/
def S (m : ℕ) : ℕ :=
  sorry

/-- Product of digits of a positive integer -/
def P (m : ℕ) : ℕ :=
  sorry

/-- For any positive integer n, there exists a sequence of positive integers
    satisfying the given conditions -/
theorem existence_of_sequence (n : ℕ) (hn : n > 0) :
  ∃ (a : ℕ → ℕ), 
    (∀ i : ℕ, i < n → S (a i) < S (a (i + 1))) ∧ 
    (∀ i : ℕ, i < n → S (a i) = P (a ((i + 1) % n))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sequence_l166_16636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_implies_bounded_l166_16698

theorem sequence_limit_implies_bounded 
  (x : ℕ → ℝ) (x₀ : ℝ) (h : Filter.Tendsto x atTop (nhds x₀)) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_implies_bounded_l166_16698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2006_value_l166_16693

def b : ℕ → ℚ
  | 0 => 3  -- Adding this case to handle Nat.zero
  | 1 => 3
  | 2 => 4
  | n+3 => (b (n+2))^2 / (b (n+1))

theorem b_2006_value : b 2006 = 64/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2006_value_l166_16693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_digits_l166_16617

theorem count_divisible_digits : 
  (Finset.filter (fun n : ℕ => n ≥ 1 ∧ n ≤ 9 ∧ (240 + n) % n = 0) (Finset.range 10)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_digits_l166_16617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_cubes_l166_16642

theorem divisibility_of_sum_of_cubes (x y : ℚ) : 
  ∃ P : ℚ, 
    (x^2 - x*y + y^2)^3 + (x^2 + x*y + y^2)^3 = (2*x^2 + 2*y^2) * P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_cubes_l166_16642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l166_16663

-- Define the necessary types and variables
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (A B : V) (m n k : ℝ)

-- Define the line type
def Line (V : Type*) [NormedAddCommGroup V] := V → Prop

-- Define the distance function
noncomputable def dist_to_line (p : V) (L : Line V) : ℝ := sorry

-- Define the condition for points being on the same side of a line
def same_side (A B : V) (L : Line V) : Prop := sorry

-- Define the tangency condition
def is_tangent (L : Line V) (center : V) (radius : ℝ) : Prop := sorry

-- State the theorem
theorem line_tangent_to_circle
  (hm : m > 0) (hn : n > 0)
  (h_constant : ∀ L : Line V, same_side A B L → m * dist_to_line A L + n * dist_to_line B L = k) :
  ∃ (center : V) (radius : ℝ), 
    radius = k / (m + n) ∧
    ∀ L : Line V, same_side A B L → 
      (m * dist_to_line A L + n * dist_to_line B L = k) → 
      is_tangent L center radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l166_16663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_questions_l166_16685

/-- Represents a participant's answers to n questions -/
def Answers (n : ℕ) := Fin n → Bool

/-- The type of all participants' answers -/
def AllAnswers (n : ℕ) := Fin 8 → Answers n

/-- Checks if the distribution of answers for a pair of questions is valid -/
def validDistribution (answers : AllAnswers n) (i j : Fin n) : Prop :=
  (∃ (a b c d : Fin 8), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    answers a i = true  ∧ answers a j = true  ∧
    answers b i = true  ∧ answers b j = false ∧
    answers c i = false ∧ answers c j = true  ∧
    answers d i = false ∧ answers d j = false) ∧
  (∀ (x : Fin 8), (x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d) →
    (answers x i = true  ∧ answers x j = true)  ∨
    (answers x i = true  ∧ answers x j = false) ∨
    (answers x i = false ∧ answers x j = true)  ∨
    (answers x i = false ∧ answers x j = false))
  where
    a := ⟨0, by norm_num⟩
    b := ⟨1, by norm_num⟩
    c := ⟨2, by norm_num⟩
    d := ⟨3, by norm_num⟩

/-- The main theorem: the maximum number of questions is 7 -/
theorem max_questions :
  ∀ n : ℕ, (∃ (answers : AllAnswers n),
    ∀ (i j : Fin n), i ≠ j → validDistribution answers i j) →
  n ≤ 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_questions_l166_16685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_succ_l166_16660

open BigOperators

def f (n : ℕ) : ℚ := ∑ i in Finset.range (2*n+1), 1 / (n + i + 1)

theorem f_succ (k : ℕ) : f (k + 1) = f k + 1 / (3*k + 2) + 1 / (3*k + 3) + 1 / (3*k + 4) - 1 / (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_succ_l166_16660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_boy_certain_l166_16655

/-- The event that at least one boy is selected -/
def EventAtLeastOneBoy (num_boys : ℕ) (num_girls : ℕ) (selection_size : ℕ) : Set (Fin (num_boys + num_girls)) :=
sorry

/-- The probability of an event -/
noncomputable def Prob {α : Type*} [MeasurableSpace α] (event : Set α) : ℝ :=
sorry

/-- Given a group of 5 boys and 2 girls, when 3 people are randomly selected,
    the event "At least 1 boy is selected" is certain to occur. -/
theorem at_least_one_boy_certain (num_boys : ℕ) (num_girls : ℕ) (selection_size : ℕ) :
  num_boys = 5 → num_girls = 2 → selection_size = 3 →
  Prob (EventAtLeastOneBoy num_boys num_girls selection_size) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_boy_certain_l166_16655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l166_16696

/-- Given two overlapping circular sectors with radius 10, where one sector has a central angle of 90° 
    and the other has a central angle of 60°, the area of the non-overlapping regions combined is 25π/3 -/
theorem overlapping_sectors_area (r angle1 angle2 : ℝ) : 
  r = 10 → angle1 = 90 → angle2 = 60 → 
  (angle1 / 360) * π * r^2 - min ((angle1 / 360) * π * r^2) ((angle2 / 360) * π * r^2) = 25 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l166_16696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_theorem_l166_16691

theorem complex_magnitude_theorem : 
  Complex.abs (2 + Complex.I^2 + 2*Complex.I^3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_theorem_l166_16691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l166_16607

theorem cosine_sum_zero (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos (y + π/3) + Real.cos (z - π/3) = 0)
  (h2 : Real.sin x + Real.sin (y + π/3) + Real.sin (z - π/3) = 0) :
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l166_16607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l166_16674

noncomputable section

-- Define the principal amount
def P : ℝ := 3600

-- Define the annual interest rate
def r : ℝ := 0.25

-- Define the time in years
def t : ℝ := 2

-- Define simple interest
def simple_interest : ℝ := P * r * t

-- Define compound interest
noncomputable def compound_interest : ℝ := P * (1 + r) ^ t - P

-- Theorem statement
theorem interest_difference : compound_interest - simple_interest = 225 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l166_16674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_of_y_in_base_nine_l166_16615

def base_three_representation : List ℕ := [2, 0, 2, 2, 2, 1, 2, 0, 2, 1, 1, 1, 2, 2, 2, 1, 2, 0, 2, 1]

def y : ℕ := base_three_representation.enum.foldl (λ acc (i, digit) => acc + digit * (3 ^ i)) 0

def first_digit_base_nine (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let log9 := Nat.log n 9
    n / (9 ^ log9)

theorem first_digit_of_y_in_base_nine :
  first_digit_base_nine y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_of_y_in_base_nine_l166_16615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_exponents_l166_16654

theorem gcd_of_exponents (a p q : ℕ) (ha : a > 1) :
  Nat.gcd (a^p - 1) (a^q - 1) = a^(Nat.gcd p q) - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_exponents_l166_16654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_a_geq_g_4_b_l166_16644

/-- The function g as defined in the problem -/
noncomputable def g (a b : ℝ) : ℝ := a * Real.sqrt b - (1/4) * b

/-- The main theorem to be proved -/
theorem infinitely_many_a_geq_g_4_b :
  ∃ S : Set ℝ, (Set.Infinite S) ∧ 
  (∀ a ∈ S, ∀ b > 0, g a 4 ≥ g a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_a_geq_g_4_b_l166_16644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l166_16656

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - (Real.cos (x + Real.pi / 4))^2

theorem triangle_max_area (A B C : ℝ) (hA : 0 < A ∧ A < Real.pi / 2) 
  (hB : 0 < B ∧ B < Real.pi / 2) (hC : 0 < C ∧ C < Real.pi / 2) 
  (hABC : A + B + C = Real.pi) (hf : f (A / 2) = 0) 
  (ha : Real.cos B * Real.sin C = 1) :
  ∃ (S : ℝ), S ≤ (2 + Real.sqrt 3) / 4 ∧
  (∀ (S' : ℝ), S' = 1 / 2 * Real.sin A * Real.sin B / Real.sin C → S' ≤ S) :=
by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l166_16656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nontransitive_dice_exist_l166_16606

/-- Represents a die with 6 faces --/
def Die := Fin 6 → ℕ

/-- Checks if one die is more advantageous than another --/
def more_advantageous (d1 d2 : Die) : Prop :=
  (Finset.sum Finset.univ (λ i => (Finset.filter (λ j => d1 i > d2 j) Finset.univ).card)) > 18

/-- The set of the first 18 natural numbers --/
def first_18 : Finset ℕ := Finset.range 18

theorem nontransitive_dice_exist : 
  ∃ (A B C : Die), 
    (∀ i : Fin 6, A i ∈ first_18) ∧ 
    (∀ i : Fin 6, B i ∈ first_18) ∧ 
    (∀ i : Fin 6, C i ∈ first_18) ∧
    (Finset.card (Finset.image A Finset.univ ∪ 
                  Finset.image B Finset.univ ∪ 
                  Finset.image C Finset.univ) = 18) ∧
    more_advantageous A B ∧ 
    more_advantageous B C ∧ 
    more_advantageous C A :=
  sorry

#check nontransitive_dice_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nontransitive_dice_exist_l166_16606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_is_6pi_l166_16612

/-- Represents a tetrahedron with vertex P and base ABC. -/
structure Tetrahedron where
  PA : ℝ
  PB : ℝ
  PC : ℝ

/-- The circumscribed sphere of a tetrahedron. -/
noncomputable def circumscribed_sphere_area (t : Tetrahedron) : ℝ :=
  4 * Real.pi * ((t.PA ^ 2 + t.PB ^ 2 + t.PC ^ 2) / 4)

/-- Theorem: The surface area of the circumscribed sphere of a tetrahedron P-ABC is 6π,
    given that edges PA, PB, and PC are mutually perpendicular, PA = 2, and PB = PC = 1. -/
theorem circumscribed_sphere_area_is_6pi (t : Tetrahedron)
    (h1 : t.PA = 2)
    (h2 : t.PB = 1)
    (h3 : t.PC = 1) :
    circumscribed_sphere_area t = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_is_6pi_l166_16612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_fill_time_correct_l166_16695

/-- Represents the volume of water that can be filled by a sluice gate in one hour -/
structure SluiceGate where
  flow_rate : ℚ

/-- Represents an artificial lake with two sluice gates -/
structure ArtificialLake where
  volume : ℚ
  sluice1 : SluiceGate
  sluice2 : SluiceGate

/-- The time required to fill the lake when both sluice gates are open equally -/
def equal_fill_time (lake : ArtificialLake) : ℚ :=
  lake.volume / (lake.sluice1.flow_rate + lake.sluice2.flow_rate)

theorem equal_fill_time_correct (lake : ArtificialLake) :
  lake.volume = 9900 ∧
  10 * lake.sluice1.flow_rate + 14 * lake.sluice2.flow_rate = lake.volume ∧
  18 * lake.sluice1.flow_rate + 12 * lake.sluice2.flow_rate = lake.volume →
  equal_fill_time lake = 13 + 1/5 := by
  sorry

#eval (13 : ℚ) + 1/5  -- Should output 66/5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_fill_time_correct_l166_16695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_squared_A_over_sin_B_minus_A_l166_16643

-- Define the triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the theorem
theorem range_of_sin_squared_A_over_sin_B_minus_A 
  (triangle : AcuteTriangle) 
  (h : triangle.c - triangle.a = 2 * triangle.a * Real.cos triangle.B) : 
  ∃ (x : ℝ), 1/2 < x ∧ x < Real.sqrt 2 / 2 ∧ 
  (Real.sin triangle.A)^2 / Real.sin (triangle.B - triangle.A) = x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_squared_A_over_sin_B_minus_A_l166_16643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_exists_l166_16697

/-- A collection of sets satisfying the problem conditions -/
structure SetCollection where
  sets : Finset (Finset ℕ)
  card_sets : sets.card = 1978
  card_each_set : ∀ s, s ∈ sets → s.card = 40
  common_element : ∀ s t, s ∈ sets → t ∈ sets → s ≠ t → (s ∩ t).card = 1

/-- The main theorem statement -/
theorem common_element_exists (c : SetCollection) :
  ∃ x : ℕ, ∀ s, s ∈ c.sets → x ∈ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_exists_l166_16697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_gcd_equivalence_l166_16625

theorem divisibility_and_gcd_equivalence (a b m : ℕ) (ha : a > 0) (hb : b > 0) (hm : m > 0) :
  (∃ n : ℕ, n > 0 ∧ m ∣ (a^n - 1) * b) ↔ Nat.gcd (a * b) m = Nat.gcd b m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_gcd_equivalence_l166_16625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l166_16682

-- Define the problem parameters
def track_length : ℝ := 16.8
def station_interval : ℝ := 2.4
def robot_speed : ℝ := 0.8
def robot_a_rest_time : ℝ := 1
def total_time : ℝ := 120

-- Define the number of stations
noncomputable def num_stations : ℕ := ⌊track_length / station_interval⌋₊ + 1

-- Define the time for each robot to move one station
noncomputable def time_per_station_a : ℝ := station_interval / robot_speed + robot_a_rest_time
noncomputable def time_per_station_b : ℝ := station_interval / robot_speed

-- Define the theorem
def robots_meet_count : ℕ := 6

-- State the main theorem
theorem main_theorem : robots_meet_count = 6 := by
  -- The proof goes here
  sorry

#eval robots_meet_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l166_16682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_seventeen_l166_16633

theorem floor_expression_equals_seventeen : 
  ⌊(4 : ℝ) * (5 - 3 / 4)⌋ = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_seventeen_l166_16633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l166_16647

/-- The sum of the infinite series 1 + 3x + 5x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := 
  1 + 3*x + 5*x^2 + (2 / (1 - x)^2 - 2 / (1 - x))

theorem infinite_series_solution (x : ℝ) (hx : |x| < 1) :
  S x = 4 → x = (5 - Real.sqrt 13) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l166_16647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_problem_l166_16649

/-- Given a bowl of water with a constant daily evaporation rate over a fixed period,
    prove that the initial amount of water can be determined if we know the percentage
    of water that evaporated. -/
theorem water_evaporation_problem (daily_rate : ℝ) (days : ℕ) (evaporation_percentage : ℝ) 
    (h1 : daily_rate = 0.008)
    (h2 : days = 50)
    (h3 : evaporation_percentage = 0.04)
    : ∃ (initial_amount : ℝ), 
      initial_amount * evaporation_percentage = daily_rate * (days : ℝ) ∧ 
      initial_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_problem_l166_16649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resistor_x_value_l166_16604

/-- Represents the resistance of a parallel circuit with two resistors -/
noncomputable def parallel_resistance (x y : ℝ) : ℝ :=
  1 / ((1 / x) + (1 / y))

/-- Proves that given the conditions, the resistance of resistor x is 5 ohms -/
theorem resistor_x_value (r y : ℝ) (h1 : r = 2.9166666666666665) (h2 : y = 7) :
  ∃ x : ℝ, parallel_resistance x y = r ∧ x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resistor_x_value_l166_16604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_permutations_no_cube_permutations_l166_16623

/-- A permutation of 1..n is a square permutation if aᵢaᵢ₊₁ + 1 is a perfect square for all 1 ≤ i < n -/
def IsSquarePermutation (n : ℕ+) (a : Fin n → ℕ+) : Prop :=
  Function.Bijective a ∧ ∀ i : Fin n.1.pred, ∃ k : ℕ, (a i * a i.succ + 1 : ℕ) = k ^ 2

/-- A permutation of 1..n is a cube permutation if aᵢaᵢ₊₁ + 1 is a perfect cube for all 1 ≤ i < n -/
def IsCubePermutation (n : ℕ+) (a : Fin n → ℕ+) : Prop :=
  Function.Bijective a ∧ ∀ i : Fin n.1.pred, ∃ k : ℕ, (a i * a i.succ + 1 : ℕ) = k ^ 3

/-- The set of positive integers n for which a square permutation exists is infinite -/
theorem infinite_square_permutations :
    Set.Infinite {n : ℕ+ | ∃ a : Fin n → ℕ+, IsSquarePermutation n a} := by
  sorry

/-- There does not exist a positive integer n for which a cube permutation exists -/
theorem no_cube_permutations :
    ∀ n : ℕ+, ¬∃ a : Fin n → ℕ+, IsCubePermutation n a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_permutations_no_cube_permutations_l166_16623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_one_l166_16628

/-- Geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem common_ratio_is_one 
  (a : ℝ) (q : ℝ) 
  (h : geometric_sum a q 8 / geometric_sum a q 4 = 2) :
  q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_one_l166_16628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l166_16611

/-- Approximate values of A, B, C, D, and E --/
noncomputable def A : ℝ := 2.5
noncomputable def B : ℝ := -0.8
noncomputable def C : ℝ := -2.2
noncomputable def D : ℝ := 1.1
noncomputable def E : ℝ := -3.1

/-- The expressions to be evaluated --/
noncomputable def expr1 : ℝ := A * B
noncomputable def expr2 : ℝ := (B + C) / E
noncomputable def expr3 : ℝ := B * D - A * C
noncomputable def expr4 : ℝ := C / (A * B)
noncomputable def expr5 : ℝ := E - A

theorem expressions_evaluation :
  expr2 > 0 ∧ expr3 > 0 ∧ expr4 > 0 ∧ expr1 ≤ 0 ∧ expr5 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l166_16611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l166_16650

/-- The circle C defined by the equation x^2 + y^2 - 2x + 2y - 2 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 2 = 0

/-- Point P at (0,0) -/
def point_P : ℝ × ℝ := (0, 0)

/-- The shortest chord length cut by a line passing through point P on circle C -/
noncomputable def shortest_chord_length : ℝ := 2 * Real.sqrt 2

/-- Theorem stating that the shortest chord length is 2√2 -/
theorem shortest_chord_theorem :
  shortest_chord_length = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l166_16650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l166_16653

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def validPairs : Finset (ℕ × ℕ) := 
  {(1, 4), (4, 1), (2, 3), (3, 2)}

def totalPairs : Finset (ℕ × ℕ) := 
  numbers.product numbers

theorem probability_sum_is_five : 
  (validPairs.filter (fun p => p.1 ≠ p.2)).card / (totalPairs.filter (fun p => p.1 ≠ p.2)).card = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l166_16653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_l_max_distance_C_to_l_l166_16602

/-- Curve C parametric equations -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

/-- Line l in Cartesian form -/
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 12 = 0

/-- Point P -/
noncomputable def point_P : ℝ × ℝ := (1, -Real.sqrt 3)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- Maximum distance from curve C to line l -/
noncomputable def max_distance_curve_to_line : ℝ :=
  sorry

theorem distance_P_to_l :
  distance_point_to_line point_P line_l = 4 := by
  sorry

theorem max_distance_C_to_l :
  max_distance_curve_to_line = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_l_max_distance_C_to_l_l166_16602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l166_16631

/-- The average speed of a car traveling in three equal parts of a journey -/
noncomputable def average_speed (total_distance : ℝ) : ℝ :=
  let speed1 := 80 -- km/h
  let speed2 := 24 -- km/h
  let speed3 := 30 -- km/h
  let time1 := total_distance / (3 * speed1)
  let time2 := total_distance / (3 * speed2)
  let time3 := total_distance / (3 * speed3)
  let total_time := time1 + time2 + time3
  total_distance / total_time

/-- Theorem stating that the average speed is approximately 34.2857 km/h -/
theorem average_speed_theorem (total_distance : ℝ) (h : total_distance > 0) :
  abs (average_speed total_distance - 34.2857) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l166_16631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_faces_on_two_dice_l166_16634

-- Define die_face function
def die_face (n x y : ℕ) : Prop := 
  x ≤ n ∧ y ≤ n ∧ x = y

-- Define ways_to_sum function
def ways_to_sum (sum num_faces1 num_faces2 : ℕ) : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = sum) 
    (Finset.product (Finset.range num_faces1) (Finset.range num_faces2))).card

-- Define prob_sum function
noncomputable def prob_sum (sum num_faces1 num_faces2 : ℕ) : ℚ :=
  (ways_to_sum sum num_faces1 num_faces2 : ℚ) / ((num_faces1 * num_faces2) : ℚ)

theorem least_faces_on_two_dice (a b : ℕ) : 
  a ≥ 8 → b ≥ 8 → 
  (∀ x : ℕ, x ≤ a → ∃! y : ℕ, y ≤ a ∧ die_face a x y) →
  (∀ x : ℕ, x ≤ b → ∃! y : ℕ, y ≤ b ∧ die_face b x y) →
  (prob_sum 9 a b : ℚ) = 5 / 6 * (prob_sum 13 a b : ℚ) →
  (prob_sum 16 a b : ℚ) = 1 / 10 →
  a + b ≥ 20 ∧ ∀ c d : ℕ, c ≥ 8 → d ≥ 8 → 
    (∀ x : ℕ, x ≤ c → ∃! y : ℕ, y ≤ c ∧ die_face c x y) →
    (∀ x : ℕ, x ≤ d → ∃! y : ℕ, y ≤ d ∧ die_face d x y) →
    (prob_sum 9 c d : ℚ) = 5 / 6 * (prob_sum 13 c d : ℚ) →
    (prob_sum 16 c d : ℚ) = 1 / 10 →
    c + d ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_faces_on_two_dice_l166_16634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_1000_l166_16699

def paperclips : ℕ → ℕ
  | 0 => 5  -- Sunday
  | 1 => 15 -- Monday (5 + 10)
  | n + 2 => 2 * paperclips (n + 1)

def days_of_week : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def day_name (n : ℕ) : String :=
  days_of_week[n % 7]'(by
    simp [days_of_week]
    exact Nat.mod_lt n (by norm_num))

theorem first_day_exceeding_1000 :
  (∃ n : ℕ, paperclips n > 1000 ∧ ∀ m : ℕ, m < n → paperclips m ≤ 1000) →
  (∃ n : ℕ, paperclips n > 1000 ∧ ∀ m : ℕ, m < n → paperclips m ≤ 1000 ∧ day_name n = "Monday" ∧ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_1000_l166_16699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_sharing_l166_16662

/-- Represents the length of a burger in inches -/
noncomputable def burger_length : ℚ := 12

/-- Represents the fraction of the burger shared -/
noncomputable def shared_fraction : ℚ := 2 / 5

/-- Calculates the remaining portion of the burger -/
noncomputable def remaining_portion (total : ℚ) (shared : ℚ) : ℚ :=
  total - (shared * total)

theorem burger_sharing :
  remaining_portion burger_length shared_fraction = 72 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_sharing_l166_16662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l166_16673

/-- The function f(x) = 2^x + x/3 -/
noncomputable def f (x : ℝ) : ℝ := 2^x + x/3

/-- Theorem stating that there exists a unique zero point of f in (-2, -1) -/
theorem zero_point_existence :
  ∃! x₀ : ℝ, -2 < x₀ ∧ x₀ < -1 ∧ f x₀ = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l166_16673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_52789_between_integers_l166_16620

theorem log_52789_between_integers (h1 : 10000 < 52789) (h2 : 52789 < 100000)
  (h3 : Real.log 10000 = 4) (h4 : Real.log 100000 = 5) :
  ∃ c d : ℤ, c + 1 = d ∧ 
    ↑c < Real.log 52789 ∧ 
    Real.log 52789 < ↑d ∧ 
    c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_52789_between_integers_l166_16620
