import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l1103_110369

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

-- State the theorem
theorem sine_value_from_cosine (α : ℝ) (h : f α = 1/3) : Real.sin α = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l1103_110369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speeds_l1103_110364

/-- Represents the speeds of vehicles in km/h -/
structure VehicleSpeeds where
  scooter : ℝ
  car_sharing : ℝ
  professor : ℝ

/-- Represents the problem conditions -/
structure ProblemConditions where
  total_distance : ℝ
  meeting_time : ℝ
  scooter_meeting_distance : ℝ
  scooter_distance : ℝ
  scooter_time_difference : ℝ
  scooter_speed_ratio : ℝ

/-- Theorem stating the correct speeds given the problem conditions -/
theorem correct_speeds (conditions : ProblemConditions) :
  ∃ (speeds : VehicleSpeeds),
    speeds.scooter = 15 ∧
    speeds.car_sharing = 40 ∧
    speeds.professor = 60 ∧
    -- Cars meet after 3 minutes (0.05 hours)
    speeds.car_sharing * conditions.meeting_time + speeds.professor * conditions.meeting_time = conditions.total_distance ∧
    -- Scooter covers 30 km in 1.25 hours longer than the car
    conditions.scooter_distance / speeds.scooter = conditions.scooter_distance / speeds.car_sharing + conditions.scooter_time_difference ∧
    -- Scooter speed is 4 times less than professor's car speed
    speeds.scooter * conditions.scooter_speed_ratio = speeds.professor :=
by
  sorry

-- Remove the #eval statement as it was causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speeds_l1103_110364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_perpendicular_transitivity_l1103_110333

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (m n : Line) (a : Plane)
  (h1 : perpendicular m a)
  (h2 : parallel_line_plane n a) :
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_perpendicular_transitivity
  (m : Line) (a b γ : Plane)
  (h1 : parallel_planes a b)
  (h2 : parallel_planes b γ)
  (h3 : perpendicular m a) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_perpendicular_transitivity_l1103_110333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_is_negative_seven_l1103_110301

noncomputable def binomial_expansion (x : ℝ) : ℝ := (3*x - 1/(2*3*x))^8

noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ := 
  (Nat.choose 8 r) * (-1/2)^r * x^((8 - 2*r)/3)

noncomputable def fourth_term_coefficient : ℝ := (Nat.choose 8 3) * (-1/8)

theorem fourth_term_coefficient_is_negative_seven : 
  fourth_term_coefficient = -7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_is_negative_seven_l1103_110301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l1103_110388

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 3a₂ + 7a₃ is -9/196 -/
theorem min_value_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : 
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence condition
  a 1 = 1 →                     -- First term is 1
  (∃ m : ℝ, ∀ x : ℝ, 3 * (a 2) + 7 * (a 3) ≥ m) ∧ 
  (∃ r : ℝ, 3 * (a 2) + 7 * (a 3) = -9/196) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l1103_110388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_weakly_increasing_g_not_weakly_increasing_h_weakly_increasing_conditions_l1103_110362

-- Define a "weakly increasing function" on an interval
def weakly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → (f x / x) > (f y / y))

-- Theorem for f(x) = x + 4 on (1, 2)
theorem f_weakly_increasing :
  weakly_increasing (fun x => x + 4) (Set.Ioo 1 2) := by sorry

-- Theorem for g(x) = x^2 + 4x + 2 on (1, 2)
theorem g_not_weakly_increasing :
  ¬weakly_increasing (fun x => x^2 + 4*x + 2) (Set.Ioo 1 2) := by sorry

-- Theorem for h(x) = x^2 + (m - 1/2)x + b on (0, 1]
theorem h_weakly_increasing_conditions (m b : ℝ) :
  (m ≥ 1/2 ∧ b ≥ 1) ↔
  weakly_increasing (fun x => x^2 + (m - 1/2)*x + b) (Set.Ioc 0 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_weakly_increasing_g_not_weakly_increasing_h_weakly_increasing_conditions_l1103_110362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_two_l1103_110396

/-- Represents the speed of a boat in still water -/
noncomputable def boat_speed : ℝ → ℝ := sorry

/-- Represents the speed of the water current -/
noncomputable def current_speed : ℝ := sorry

/-- The total distance traveled by the boat -/
def distance : ℝ := 15

/-- Time taken for downstream travel at normal speed -/
noncomputable def downstream_time (s : ℝ) : ℝ := distance / (boat_speed s + current_speed)

/-- Time taken for upstream travel at normal speed -/
noncomputable def upstream_time (s : ℝ) : ℝ := distance / (boat_speed s - current_speed)

/-- Condition for normal speed travel -/
axiom normal_speed_condition : 
  ∃ s : ℝ, upstream_time s = downstream_time s + 5

/-- Condition for double speed travel -/
axiom double_speed_condition : 
  ∃ s : ℝ, upstream_time (2 * s) = downstream_time (2 * s) + 1

/-- The theorem to be proved -/
theorem current_speed_is_two : current_speed = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_two_l1103_110396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_theorem_l1103_110392

/-- Given a tetrahedron with volume V and four smaller tetrahedrons with volumes V₁, V₂, V₃, V₄
    formed by planes parallel to its faces passing through an interior point,
    prove that ∛V = ∛V₁ + ∛V₂ + ∛V₃ + ∛V₄ -/
theorem tetrahedron_volume_theorem (V V₁ V₂ V₃ V₄ : ℝ) 
    (h_pos : V > 0 ∧ V₁ > 0 ∧ V₂ > 0 ∧ V₃ > 0 ∧ V₄ > 0) :
    V^(1/3) = V₁^(1/3) + V₂^(1/3) + V₃^(1/3) + V₄^(1/3) := by
  sorry

#check tetrahedron_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_theorem_l1103_110392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1103_110352

def is_square_of_prime (n : ℤ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = (p : ℤ)^2

def satisfies_condition (x : ℤ) : Prop :=
  is_square_of_prime (x^2 + 28*x + 889)

theorem solution_set : 
  ∀ x : ℤ, satisfies_condition x ↔ x ∈ ({-360, -60, -48, -40, 8, 20, 32, 332} : Finset ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1103_110352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l1103_110305

theorem biased_coin_probability : ∃! p : ℝ, 0 < p ∧ p < 1/2 ∧ 3 * p * (1 - p)^2 = 1/2 ∧ |p - 0.3177| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l1103_110305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_l1103_110357

-- Define the theorem
theorem complex_square : (1 + 2*Complex.I)^2 = -3 + 4*Complex.I := by
  -- Expand the square
  calc (1 + 2*Complex.I)^2
    = 1^2 + 2*(1)*(2*Complex.I) + (2*Complex.I)^2 := by ring
  -- Simplify
  _ = 1 + 4*Complex.I + 4*Complex.I^2 := by ring
  -- Use the property I^2 = -1
  _ = 1 + 4*Complex.I + 4*(-1) := by simp [Complex.I_sq]
  -- Simplify to the final result
  _ = -3 + 4*Complex.I := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_l1103_110357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1103_110380

-- Define T(s) as the sum of the geometric series
noncomputable def T (s : ℝ) : ℝ := 9 / (1 - s)

theorem geometric_series_sum (b : ℝ) :
  -1 < b → b < 1 → T b * T (-b) = 1458 → T b + T (-b) = 324 := by
  -- Introduce the assumptions
  intro h1 h2 h3
  -- Unfold the definition of T
  unfold T at *
  -- Simplify the expression
  simp [h1, h2] at *
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1103_110380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l1103_110371

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Check if a line segment is tangent to a circle -/
def is_tangent_to_circle (p1 p2 : Point) (c : Circle) : Prop :=
  let (cx, cy) := c.center
  let dx := p2.x - p1.x
  let dy := p2.y - p1.y
  let fx := p1.x - cx
  let fy := p1.y - cy
  (fx * dx + fy * dy)^2 = (dx^2 + dy^2) * (fx^2 + fy^2 - c.radius^2)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (cx, cy) := c.center
  (p.x - cx)^2 + (p.y - cy)^2 = c.radius^2

theorem circle_triangle_area 
  (ω₁ ω₂ ω₃ : Circle)
  (Q₁ Q₂ Q₃ : Point) :
  ω₁.radius = 5 →
  ω₂.radius = 5 →
  ω₃.radius = 5 →
  are_externally_tangent ω₁ ω₂ →
  are_externally_tangent ω₂ ω₃ →
  are_externally_tangent ω₃ ω₁ →
  point_on_circle Q₁ ω₁ →
  point_on_circle Q₂ ω₂ →
  point_on_circle Q₃ ω₃ →
  distance Q₁ Q₂ = distance Q₂ Q₃ →
  distance Q₂ Q₃ = distance Q₃ Q₁ →
  is_tangent_to_circle Q₁ Q₂ ω₁ →
  is_tangent_to_circle Q₂ Q₃ ω₂ →
  is_tangent_to_circle Q₃ Q₁ ω₃ →
  (Q₁.x - Q₂.x) * (Q₃.x - Q₂.x) + (Q₁.y - Q₂.y) * (Q₃.y - Q₂.y) = 0 →  -- right angle condition
  triangle_area Q₁ Q₂ Q₃ = Real.sqrt 1406.25 + Real.sqrt 1250 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l1103_110371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_n_zero_implies_n_zero_l1103_110376

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the expression (1 + 4 ∛2 - 4 ∛4)
noncomputable def expression : ℝ := 1 + 4 * cubeRoot 2 - 4 * cubeRoot 4

-- Define the theorem
theorem c_n_zero_implies_n_zero (n : ℕ) 
  (h : ∃! (a b c : ℤ), expression ^ n = a + b * cubeRoot 2 + c * cubeRoot 4) :
  (∃! (a b c : ℤ), expression ^ n = a + b * cubeRoot 2 + c * cubeRoot 4 ∧ c = 0) → n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_n_zero_implies_n_zero_l1103_110376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_9pi_l1103_110308

theorem tan_theta_minus_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -(1/2)) :
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_9pi_l1103_110308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abao_stilts_height_l1103_110377

/-- The initial total height of A Bao and the stilts in dm -/
noncomputable def initial_height : ℝ := 160

/-- A Bao's height in dm -/
noncomputable def abao_height : ℝ := initial_height / 4

/-- The length broken off each stilt in dm -/
noncomputable def broken_length : ℝ := 20

/-- The new total height after breaking the stilts -/
noncomputable def new_height : ℝ := initial_height - 2 * broken_length

theorem abao_stilts_height : 
  (abao_height = initial_height / 4) ∧ 
  (abao_height = new_height / 3) ∧ 
  (new_height = initial_height - 2 * broken_length) ∧ 
  (initial_height = 160) := by
  sorry

#check abao_stilts_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abao_stilts_height_l1103_110377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1103_110311

/-- Line in 2D space represented by ax - 3y + c = 0 --/
structure Line where
  a : ℝ
  c : ℝ

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line --/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x - 3 * p.y + l.c = 0

/-- Check if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a / 4 = -3 / -6

/-- Calculate distance between a point and a line --/
noncomputable def distance (p : Point) (l : Line) : ℝ :=
  |l.a * p.x - 3 * p.y + l.c| / Real.sqrt (l.a^2 + (-3)^2)

theorem line_properties (m n : Line) (M : Point) :
  on_line ⟨-2, 0⟩ m →
  on_line M n →
  parallel m n →
  M.x = 3 →
  M.y = 1 →
  n.a = 4 →
  n.c = -6 →
  m.c = 2 →
  (m.a = 2 ∧ distance M m = (5 * Real.sqrt 13) / 13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1103_110311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_n_l1103_110335

def geometric_sum (n : ℕ) : ℕ := (n^2008 - 1) / (n - 1)

def expression (n : ℕ) : ℚ := (geometric_sum n : ℚ) / (n + 2007 : ℚ)

def largest_n : ℤ := (2007^2008 - 1) / 2008 - 2007

theorem largest_integer_n :
  ∀ k : ℕ, k > largest_n.toNat → ¬(expression k).isInt ∧
  (expression largest_n.toNat).isInt ∧
  ∀ m : ℕ, m < largest_n.toNat → (expression m).isInt → m < largest_n.toNat :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_n_l1103_110335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_invariant_under_permutation_l1103_110322

/-- The universe set U --/
def U : Finset ℕ := Finset.range 2014

/-- The function f as described in the problem --/
def f (a b c : ℕ) : ℕ :=
  Finset.card (Finset.filter (fun t : Finset ℕ × Finset ℕ × Finset ℕ × Finset ℕ × Finset ℕ × Finset ℕ =>
    let (X₁, X₂, X₃, Y₁, Y₂, Y₃) := t
    Y₁ ⊆ X₁ ∧ X₁ ⊆ U ∧ X₁.card = a ∧
    Y₂ ⊆ X₂ ∧ X₂ ⊆ U \ Y₁ ∧ X₂.card = b ∧
    Y₃ ⊆ X₃ ∧ X₃ ⊆ U \ (Y₁ ∪ Y₂) ∧ X₃.card = c)
    (Finset.powerset U ×ˢ Finset.powerset U ×ˢ Finset.powerset U ×ˢ
     Finset.powerset U ×ˢ Finset.powerset U ×ˢ Finset.powerset U))

/-- The main theorem stating that f is invariant under permutations of its arguments --/
theorem f_invariant_under_permutation (a b c : ℕ) :
  f a b c = f b c a ∧ f a b c = f a c b ∧ f a b c = f c a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_invariant_under_permutation_l1103_110322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_set_characterization_l1103_110326

-- Define the function f and its derivative g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_is_derivative : ∀ x, HasDerivAt f (g x) x
axiom f_2_eq_0 : f 2 = 0
axiom xg_minus_f_neg : ∀ x > 0, x * g x - f x < 0

-- Define the set of x values for which f(x) < 0
def f_neg_set : Set ℝ := {x | f x < 0}

-- State the theorem
theorem f_neg_set_characterization :
  f_neg_set = Set.Ioc (-2) 0 ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_set_characterization_l1103_110326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1103_110345

/-- The central angle of the unfolded side surface of a cone -/
noncomputable def central_angle (base_radius : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * base_radius * 180) / slant_height

theorem cone_central_angle :
  let base_radius := (1 : ℝ)
  let slant_height := (3 : ℝ)
  central_angle base_radius slant_height = 120 := by
  -- Unfold the definition of central_angle
  unfold central_angle
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1103_110345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transformation_matrix_l1103_110390

open Matrix

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![0, -1; 1, 0]
def projection_vector : Fin 2 → ℚ := ![3, -4]

def combined_transformation (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let rotated := rotation_matrix.mulVec v
  let projected := (dotProduct rotated projection_vector / dotProduct projection_vector projection_vector) • projection_vector
  projected

theorem combined_transformation_matrix : 
  ∀ v : Fin 2 → ℚ, combined_transformation v = 
    mulVec (!![(-12:ℚ)/25, -9/25; 16/25, -12/25]) v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transformation_matrix_l1103_110390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_cube_roots_l1103_110324

theorem whole_numbers_between_cube_roots :
  ∃ (a b : ℝ),
    (a < 4) ∧ (7 < b) ∧
    (∃ (x y : ℕ), x^3 = 50 ∧ y^3 = 500 ∧ a = x ∧ b = y) ∧
    (Finset.filter (fun n : ℕ ↦ a < n ∧ n < b) (Finset.range 1000)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_cube_roots_l1103_110324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_perimeter_l1103_110366

-- Define the equilateral triangle
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

-- Define a line
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ t : ℝ, x = (1 - t) • p + t • q}

-- Define a parabola
def Parabola (focus : ℝ × ℝ) (directrix : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    ∃ d ∈ directrix, dist p focus = dist p d}

-- Define the intersection points
def IntersectionPoints (A B C A₁ A₂ B₁ B₂ C₁ C₂ : ℝ × ℝ) : Prop :=
  A₁ ∈ Parabola A (Line B C) ∧ A₂ ∈ Parabola A (Line B C) ∧
  B₁ ∈ Parabola B (Line C A) ∧ B₂ ∈ Parabola B (Line C A) ∧
  C₁ ∈ Parabola C (Line A B) ∧ C₂ ∈ Parabola C (Line A B)

-- The main theorem
theorem parabola_triangle_perimeter 
  (A B C A₁ A₂ B₁ B₂ C₁ C₂ : ℝ × ℝ) 
  (h₁ : Triangle A B C)
  (h₂ : IntersectionPoints A B C A₁ A₂ B₁ B₂ C₁ C₂) :
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A₁ A₂ + dist B₁ B₂ + dist C₁ C₂ = 66 - 36 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_perimeter_l1103_110366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1103_110314

-- Define the sequence a_n
noncomputable def a (n : ℕ+) : ℝ := (5 : ℝ)^(n : ℕ) - (2 : ℝ)^(n : ℕ)

-- Define b_n
noncomputable def b (n : ℕ+) : ℝ := (n.val^2 - n.val : ℝ) / 2

-- Define c
noncomputable def c (t : ℝ) : ℝ := (3/4) * t - 2

-- State the theorem
theorem min_value_theorem :
  ∃ (min_val : ℝ), min_val = 4/25 ∧
  ∀ (n : ℕ+) (t : ℝ), (n.val - t)^2 + (b n + c t)^2 ≥ min_val :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1103_110314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l1103_110387

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 - b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- Theorem statement
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution : solution_set a b c = {x : ℝ | x < -2 ∨ x > 3}) :
  (a + 5*b + c = 0) ∧
  ({x : ℝ | b * x^2 - a * x + c > 0} = Set.Ioo (-2) 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l1103_110387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_is_simplest_l1103_110385

-- Define a function to represent the simplicity of a square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y^2 = x → y = Real.sqrt x

-- Define the set of given square roots
def sqrt_options : Set ℝ := {Real.sqrt 4, Real.sqrt 5, Real.sqrt 8, Real.sqrt (1/2)}

-- State the theorem
theorem sqrt_5_is_simplest :
  Real.sqrt 5 ∈ sqrt_options ∧
  is_simplest_sqrt 5 ∧
  ∀ x ∈ sqrt_options, x ≠ Real.sqrt 5 → ¬(is_simplest_sqrt (x^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_is_simplest_l1103_110385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1103_110374

/-- The circle with center (1,1) and radius 1 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

/-- A point is on the tangent line if it satisfies either x = 2 or y = 0 -/
def on_tangent_line (x y : ℝ) : Prop := x = 2 ∨ y = 0

/-- The point (2,0) -/
def point : ℝ × ℝ := (2, 0)

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def is_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃! p : ℝ × ℝ, my_circle p.1 p.2 ∧ p.2 = f p.1

theorem tangent_lines_to_circle :
  ∀ f : ℝ → ℝ, is_tangent_line f ∧ f point.1 = point.2 →
    ∀ x y : ℝ, y = f x → on_tangent_line x y := by
  sorry

#check tangent_lines_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1103_110374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1103_110303

/-- The line l: x + y - 1 = 0 -/
def line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The parabola y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- Point M with coordinates (-1, 2) -/
def M : ℝ × ℝ := (-1, 2)

/-- A and B are the intersection points of the line and parabola -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

/-- The length of a segment between two points -/
noncomputable def segment_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := segment_length P Q

theorem intersection_properties (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  segment_length A B = Real.sqrt 10 ∧ 
  distance M A * distance M B = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1103_110303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_at_hyperbola_focus_l1103_110315

/-- The equation of a circle centered at the right focus of a hyperbola x^2 - y^2 = 2,
    which tangentially touches the asymptotes of the hyperbola -/
theorem circle_equation_at_hyperbola_focus (x y : ℝ) : 
  (∃ (a b c : ℝ), 
    -- Given hyperbola equation
    (x^2 - y^2 = 2) ∧
    -- Standard form of hyperbola
    (x^2 / a^2 - y^2 / b^2 = 1) ∧
    -- Relation between a, b, and c
    (c^2 = a^2 + b^2) ∧
    -- Right focus coordinates
    (c = 2) ∧
    -- Asymptote equations
    (y = x ∨ y = -x) ∧
    -- Circle touches asymptotes
    (∃ (r : ℝ), r > 0 ∧ 
      ((x - c)^2 + y^2 = r^2) ∧
      (r = |y - x| / Real.sqrt 2 ∨ r = |y + x| / Real.sqrt 2))) →
  (x - 2)^2 + y^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_at_hyperbola_focus_l1103_110315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_ratio_squared_l1103_110323

/-- An isosceles right triangle with leg length a and hypotenuse length d -/
structure IsoscelesRightTriangle where
  a : ℝ
  d : ℝ
  h_positive : 0 < a
  h_isosceles : d = a * Real.sqrt 2

/-- The ratio of the leg to the hypotenuse in an isosceles right triangle -/
noncomputable def leg_hypotenuse_ratio (t : IsoscelesRightTriangle) : ℝ :=
  t.a / t.d

theorem isosceles_right_triangle_ratio_squared (t : IsoscelesRightTriangle) :
  (leg_hypotenuse_ratio t)^2 = 1/2 := by
  sorry

#check isosceles_right_triangle_ratio_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_ratio_squared_l1103_110323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_line_l1103_110379

-- Define the line
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the circle
def circle_eq (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the angle AOB
def angle_AOB (A B : ℝ × ℝ) : ℝ := 120

-- Theorem statement
theorem intersection_circle_line (r : ℝ) (A B : ℝ × ℝ) (h1 : r > 0) 
  (h2 : line A.1 A.2) (h3 : line B.1 B.2)
  (h4 : circle_eq A.1 A.2 r) (h5 : circle_eq B.1 B.2 r)
  (h6 : angle_AOB A B = 120) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_line_l1103_110379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1103_110334

-- Define the hyperbola equation
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem hyperbola_eccentricity :
  ∃ (a c : ℝ), a > 0 ∧ c > 0 ∧
  (∀ x y : ℝ, hyperbola_equation x y ↔ x^2 / (a^2) - y^2 / (c^2 - a^2) = 1) ∧
  eccentricity a c = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1103_110334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1103_110347

noncomputable def series_term (n : ℕ) : ℝ := (2 * n + 1) * (1 / 2021) ^ n

noncomputable def series_sum : ℝ := ∑' n, series_term n

theorem series_sum_value : series_sum = 404602 / 404000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1103_110347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1103_110391

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

-- Define the theorem
theorem function_properties 
  (α β : ℝ) 
  (h1 : Real.cos (β - α) = 4/5)
  (h2 : Real.cos (β + α) = -4/5)
  (h3 : 0 < α)
  (h4 : α < β)
  (h5 : β ≤ Real.pi / 2) :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m) ∧
  (f β)^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1103_110391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_solution_l1103_110302

/-- Given function g -/
def g (x : ℝ) : ℝ := -x^2 - 3

/-- The quadratic function f we're looking for -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_function_solution :
  ∃ a b c : ℝ, 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, f a b c x ≥ 1) ∧ 
    (∃ x ∈ Set.Icc (-1 : ℝ) 2, f a b c x = 1) ∧
    (∀ x : ℝ, f a b c x + g x = -(f a b c (-x) + g (-x))) ∧
    ((b = -2*Real.sqrt 2 ∧ c = 3 ∧ a = 1) ∨ (b = 3 ∧ c = 3 ∧ a = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_solution_l1103_110302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_m_range_l1103_110367

noncomputable def f (a m x : ℝ) : ℝ := Real.log x / Real.log a + x - m

theorem zero_point_implies_m_range (a m : ℝ) (ha : a > 1) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ f a m x = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_m_range_l1103_110367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1103_110394

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℝ) : ℝ := (1/6) * n * (n + 1) * (n + 3)

/-- Theorem stating the difference between g(r) and g(r-1) -/
theorem g_difference (r : ℝ) : g r - g (r - 1) = (3/2) * r^2 + (5/2) * r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1103_110394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1103_110359

noncomputable section

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 8 * (x + 2)

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 0)

/-- The line passing through the focus with inclination angle 60° -/
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x

/-- Intersection points of the line and the parabola -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}

/-- The perpendicular bisector of chord AB -/
def perpendicular_bisector (x y : ℝ) : Prop :=
  y - (4 * Real.sqrt 3 / 3) = -(1 / Real.sqrt 3) * (x - 4 / 3)

/-- Point P where the perpendicular bisector intersects the x-axis -/
def point_p : ℝ × ℝ := (16 / 3, 0)

/-- The theorem to be proved -/
theorem parabola_chord_theorem :
  ∀ (A B : ℝ × ℝ),
    A ∈ intersection_points →
    B ∈ intersection_points →
    A ≠ B →
    perpendicular_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) →
    (point_p.1 - focus.1)^2 + (point_p.2 - focus.2)^2 = (16 / 3)^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1103_110359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_length_and_cos_theta_l1103_110349

-- Define the domain D
def D (x y : ℝ) : Prop := x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ Real.sqrt 2 / 3

-- Define a line passing through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the length of the line segment
noncomputable def segment_length (m : ℝ) : ℝ :=
  let x₁ := Real.sqrt 2 / 3
  let y₁ := m * x₁
  let x₂ := (2 * m) / (1 + m^2)
  let y₂ := (2 * m^2) / (1 + m^2)
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem max_segment_length_and_cos_theta :
  ∃ (m : ℝ), 
    (∀ m' : ℝ, segment_length m ≥ segment_length m') ∧
    segment_length m = Real.sqrt (2/3) ∧
    1 / Real.sqrt (1 + m^2) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_length_and_cos_theta_l1103_110349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1103_110330

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- State the theorem
theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 3 * Real.pi / 4 ∧ b = 2 * Real.pi ∧
  (∀ x y, x ∈ interval → y ∈ interval → a ≤ x → x < y → y ≤ b → f y < f x) ∧
  (∀ x y, x ∈ interval → y ∈ interval → x < y → (x < a ∨ b < y) → f y ≥ f x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1103_110330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_final_lead_l1103_110365

/-- Calculates Brenda's final lead in a Scrabble game given initial conditions --/
theorem brendas_final_lead (initial_lead brendas_play davids_play : ℤ) :
  initial_lead = 22 →
  brendas_play = 15 →
  davids_play = 32 →
  initial_lead + brendas_play - davids_play = 5 := by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num

#check brendas_final_lead

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_final_lead_l1103_110365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_divisible_by_10_correct_l1103_110320

/-- The probability that the product of n randomly generated non-zero digits is divisible by 10 -/
noncomputable def prob_divisible_by_10 (n : ℕ) : ℝ :=
  1 - ((8/9)^n + (5/9)^n - (4/9)^n)

/-- Theorem stating that the probability of n randomly generated non-zero digits
    having a product divisible by 10 is equal to 1 - ((8/9)^n + (5/9)^n - (4/9)^n) -/
theorem prob_divisible_by_10_correct (n : ℕ) :
  prob_divisible_by_10 n = 1 - ((8/9)^n + (5/9)^n - (4/9)^n) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_divisible_by_10_correct_l1103_110320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_l1103_110361

-- Part 1
theorem trigonometric_simplification (α : Real) :
  (Real.cos (α - π / 2)) / (Real.sin (5 * π / 2 + α)) * Real.sin (α - 2 * π) * Real.cos (2 * π - α) = Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem trigonometric_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_l1103_110361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1103_110300

-- Define the function f(x) = x ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the theorem
theorem min_value_f (t : ℝ) (h : t > 0) :
  ∃ (m : ℝ), m = 0 ∧ ∀ x ∈ Set.Icc 1 (t + 1), f x ≥ m := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1103_110300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_time_l1103_110338

/-- The time it takes for A and B to finish the work together -/
noncomputable def total_time_together : ℝ := 40

/-- The time A and B worked together before B left -/
noncomputable def time_worked_together : ℝ := 10

/-- The time it took A to finish the remaining work after B left -/
noncomputable def time_a_finished : ℝ := 12

/-- The rate at which A works (portion of work completed per day) -/
noncomputable def rate_a : ℝ := 1 / 16

theorem a_alone_time (h1 : rate_a + (1 / total_time_together - rate_a) = 1 / total_time_together)
                     (h2 : time_worked_together * (1 / total_time_together) + 
                           time_a_finished * rate_a = 1) : 
  1 / rate_a = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_time_l1103_110338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1103_110350

/-- Calculates the length of a train given the conditions of two trains crossing each other --/
noncomputable def calculate_train_length (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := speed_B - speed_A
  let relative_speed_ms := relative_speed * 1000 / 3600
  relative_speed_ms * crossing_time - length_A

theorem train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (crossing_time : ℝ)
  (h1 : length_A = 200)
  (h2 : speed_A = 40)
  (h3 : speed_B = 45)
  (h4 : crossing_time = 273.6) :
  ∃ ε > 0, |calculate_train_length length_A speed_A speed_B crossing_time - 180.02| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1103_110350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1103_110375

/-- Calculate present value given future value, interest rate, and time period -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- The problem statement -/
theorem investment_present_value :
  let futureValue : ℝ := 750000
  let interestRate : ℝ := 0.07
  let years : ℕ := 15
  let calculatedPV : ℝ := presentValue futureValue interestRate years
  let roundedPV : ℝ := (⌊calculatedPV * 100 + 0.5⌋ : ℝ) / 100
  roundedPV = 271971.95 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1103_110375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1103_110351

theorem max_value_theorem (x : ℝ) (hx : x > 0) :
  (x^3 + 1 - Real.sqrt (x^6 + 8)) / x ≤ -1.25 + 1.25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1103_110351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_theorem_l1103_110319

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_theorem :
  ∀ (t : Triangle),
    t.a = 3 ∧ t.b = 5 ∧ (t.c^2 - 7*t.c + 12 = 0) →
    perimeter t = 11 ∨ perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_theorem_l1103_110319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrypt_encrypt_inverse_ciphertext_1_7_decrypts_to_3_1_l1103_110368

/-- Encryption function that maps plaintext (a, b) to ciphertext (a-2b, 2a+b) -/
noncomputable def encrypt (a b : ℝ) : ℝ × ℝ := (a - 2*b, 2*a + b)

/-- Decryption function that maps ciphertext (x, y) to plaintext (a, b) -/
noncomputable def decrypt (x y : ℝ) : ℝ × ℝ := 
  let a := (2*x + y) / 5
  let b := (-x + 2*y) / 5
  (a, b)

theorem decrypt_encrypt_inverse (a b : ℝ) : 
  decrypt (encrypt a b).1 (encrypt a b).2 = (a, b) := by sorry

theorem ciphertext_1_7_decrypts_to_3_1 : 
  decrypt 1 7 = (3, 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrypt_encrypt_inverse_ciphertext_1_7_decrypts_to_3_1_l1103_110368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_divisible_by_two_l1103_110344

theorem sum_of_two_divisible_by_two (a b c : ℕ) : 
  ∃ x y, x ∈ ({a, b, c} : Set ℕ) ∧ y ∈ ({a, b, c} : Set ℕ) ∧ x ≠ y ∧ Even (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_divisible_by_two_l1103_110344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_sum_l1103_110389

/-- Given two triangles ABC and DEF in the coordinate plane,
    prove that the sum of rotation angle, rotation center coordinates,
    and scaling factor equals 111. -/
theorem triangle_transformation_sum : ∃ (n p q k : ℝ), n + p + q + k = 111 := by
  -- Define the vertices of triangles ABC and DEF
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 10)
  let C : ℝ × ℝ := (18, 0)
  let D : ℝ × ℝ := (45, 30)
  let E : ℝ × ℝ := (63, 30)
  let F : ℝ × ℝ := (45, 10)

  -- Assume the rotation angle is between 0 and 180 degrees
  have h1 : ∃ n : ℝ, 0 < n ∧ n < 180 := by
    use 90
    norm_num

  -- Assume the transformation maps ABC to DEF
  have h2 : ∃ (p q k : ℝ) (rotation : ℝ × ℝ → ℝ × ℝ) (scaling : ℝ × ℝ → ℝ × ℝ),
    (scaling ∘ rotation) A = D ∧
    (scaling ∘ rotation) B = E ∧
    (scaling ∘ rotation) C = F := by
    sorry -- This assumption is true based on the problem statement

  -- Prove that there exist n, p, q, k such that n + p + q + k = 111
  use 90, 3, 15, 3
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_sum_l1103_110389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julians_girl_friends_percentage_l1103_110325

theorem julians_girl_friends_percentage (julian_total_friends : ℕ) 
  (julian_boys_percentage : ℚ) (boyd_total_friends : ℕ) 
  (boyd_boys_percentage : ℚ) :
  julian_total_friends = 80 →
  julian_boys_percentage = 60 / 100 →
  boyd_total_friends = 100 →
  boyd_boys_percentage = 36 / 100 →
  (boyd_total_friends - (boyd_boys_percentage * boyd_total_friends).floor) = 
    2 * (julian_total_friends - (julian_boys_percentage * julian_total_friends).floor) →
  (1 - julian_boys_percentage) * 100 = 40 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julians_girl_friends_percentage_l1103_110325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_average_weight_l1103_110332

/-- Represents a group of players in a basketball team -/
structure PlayerGroup where
  count : Nat
  avgWeight : ℝ
  avgHeight : ℝ

/-- Represents a basketball team -/
structure BasketballTeam where
  tallestPlayers : PlayerGroup
  shortestPlayers : PlayerGroup
  remainingPlayers : PlayerGroup

/-- Calculates the overall average weight of a basketball team -/
noncomputable def overallAverageWeight (team : BasketballTeam) : ℝ :=
  let totalWeight := team.tallestPlayers.count * team.tallestPlayers.avgWeight +
                     team.shortestPlayers.count * team.shortestPlayers.avgWeight +
                     team.remainingPlayers.count * team.remainingPlayers.avgWeight
  let totalPlayers := team.tallestPlayers.count + team.shortestPlayers.count + team.remainingPlayers.count
  totalWeight / totalPlayers

/-- The main theorem stating that the overall average weight of the given team is 82.5 kg -/
theorem team_average_weight (team : BasketballTeam)
  (h1 : team.tallestPlayers = { count := 5, avgWeight := 90, avgHeight := 6.3 })
  (h2 : team.shortestPlayers = { count := 4, avgWeight := 75, avgHeight := 5.7 })
  (h3 : team.remainingPlayers = { count := 3, avgWeight := 80, avgHeight := 6 }) :
  overallAverageWeight team = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_average_weight_l1103_110332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1103_110393

def proposition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*m - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m - 3)*x₂ + 1 = 0

def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*m > 3 ∧ ∀ x y : ℝ, x^2/(2*m) + y^2/3 = 1 → (x/a)^2 + (y/b)^2 = 1

theorem range_of_m :
  (∀ m : ℝ, ¬(proposition_p m ∧ proposition_q m)) ∧
  (∀ m : ℝ, proposition_p m ∨ proposition_q m) →
  ∀ m : ℝ, (m < 1/2 ∨ (3/2 < m ∧ m ≤ 5/2)) ↔ (proposition_p m ∨ proposition_q m) :=
by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1103_110393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l1103_110331

/-- The shortest distance from a point on the parabola x^2 = y to the line y = 2x + m is √5 if and only if m = -6 -/
theorem shortest_distance_parabola_to_line (m : ℝ) : 
  (∃ (x y : ℝ), y = x^2 ∧ (∀ (a b : ℝ), b = a^2 → 
    Real.sqrt ((x - a)^2 + (y - b)^2) ≥ |2*x - y + m| / Real.sqrt 5) ∧ 
  (∃ (x y : ℝ), y = x^2 ∧ |2*x - y + m| / Real.sqrt 5 = Real.sqrt 5)) ↔ 
  m = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l1103_110331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_uniqueness_l1103_110356

-- Define the functions
def f1 (x : ℝ) := x^2 + 1
noncomputable def f2 (x : ℝ) := Real.log x
def f3 (x : ℝ) := abs x
noncomputable def f4 (x : ℝ) := x * Real.cos x

-- Define evenness
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

-- Define the range property
def has_range_zero_to_inf (f : ℝ → ℝ) := 
  (∀ y, y ∈ Set.range f → y ≥ 0) ∧ (∀ y ≥ 0, ∃ x, f x = y)

theorem absolute_value_uniqueness :
  is_even f3 ∧ 
  has_range_zero_to_inf f3 ∧
  (¬(is_even f1 ∧ has_range_zero_to_inf f1)) ∧
  (¬(is_even f2 ∧ has_range_zero_to_inf f2)) ∧
  (¬(is_even f4 ∧ has_range_zero_to_inf f4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_uniqueness_l1103_110356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_time_l1103_110384

/-- Represents the mowing scenario for Jerry --/
structure MowingScenario where
  total_acres : ℚ
  riding_mower_fraction : ℚ
  riding_mower_speed : ℚ
  push_mower_speed : ℚ

/-- Calculates the total mowing time for Jerry --/
def total_mowing_time (scenario : MowingScenario) : ℚ :=
  let riding_acres := scenario.total_acres * scenario.riding_mower_fraction
  let push_acres := scenario.total_acres - riding_acres
  let riding_time := riding_acres / scenario.riding_mower_speed
  let push_time := push_acres / scenario.push_mower_speed
  riding_time + push_time

/-- Theorem stating that Jerry mows for 5 hours each week --/
theorem jerry_mowing_time :
  let scenario : MowingScenario := {
    total_acres := 8,
    riding_mower_fraction := 3/4,
    riding_mower_speed := 2,
    push_mower_speed := 1
  }
  total_mowing_time scenario = 5 := by
  -- Proof goes here
  sorry

#eval total_mowing_time {
  total_acres := 8,
  riding_mower_fraction := 3/4,
  riding_mower_speed := 2,
  push_mower_speed := 1
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_time_l1103_110384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_determine_coefficients_l1103_110373

/-- A rational function with specific asymptotic behavior -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (x + 3) / (x^2 + a*x + b)

/-- Vertical asymptotes of f occur at x = 2 and x = -3 -/
def has_vertical_asymptotes (a b : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = 2 ∨ x = -3

/-- Horizontal asymptote of f occurs at y = 0 -/
def has_horizontal_asymptote_zero (a b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M : ℝ, ∀ x : ℝ, |x| > M → |f a b x| < ε

theorem asymptotes_determine_coefficients :
  ∀ a b : ℝ, has_vertical_asymptotes a b → has_horizontal_asymptote_zero a b →
  a = 1 ∧ b = -6 ∧ a + b = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_determine_coefficients_l1103_110373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1103_110355

/-- The smaller angle between clock hands at 7:30 -/
theorem clock_angle_at_7_30 : ∃ (angle : ℝ), angle = 45 := by
  -- Define constants
  let total_degrees : ℝ := 360
  let hours : ℝ := 12
  let degrees_per_hour : ℝ := total_degrees / hours

  -- Calculate hand positions
  let hour_hand_position : ℝ := 7 * degrees_per_hour + (degrees_per_hour / 2)
  let minute_hand_position : ℝ := 6 * degrees_per_hour

  -- Calculate angle between hands
  let angle : ℝ := hour_hand_position - minute_hand_position

  -- Define smaller angle
  let smaller_angle : ℝ := min angle (total_degrees - angle)

  -- Prove that smaller_angle = 45
  have h : smaller_angle = 45 := by
    -- Proof steps would go here
    sorry

  -- Conclude the theorem
  exact ⟨smaller_angle, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1103_110355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1103_110383

variable (a b : ℝ × ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

theorem vector_sum_magnitude 
  (h1 : dot_product a (a.1 + b.1, a.2 + b.2) = 3)
  (h2 : magnitude a = 1)
  (h3 : magnitude b = 2) :
  magnitude (a.1 + b.1, a.2 + b.2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1103_110383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloak_change_theorem_l1103_110328

/-- Represents the price of an invisibility cloak and the change received in different scenarios -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Represents the exchange rate between silver and gold coins -/
def exchange_rate : ℚ := 5 / 3

/-- The price of an invisibility cloak in gold coins -/
def cloak_price : ℕ := 8

/-- Calculates the change in silver coins when buying a cloak with gold coins -/
def calculate_change (gold_paid : ℕ) : ℕ :=
  (((gold_paid - cloak_price : ℚ) * exchange_rate).floor : ℤ).toNat

theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1) :
  calculate_change 14 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloak_change_theorem_l1103_110328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_with_collinear_intersections_exists_l1103_110395

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a pentagon
structure Pentagon where
  A : Fin 5 → Point2D

-- Define the intersection of two lines
noncomputable def intersect (p1 p2 p3 p4 : Point2D) : Point2D :=
  sorry

-- Define collinearity of points
def collinear (points : List Point2D) : Prop :=
  sorry

-- Define convexity of a pentagon
def convex (p : Pentagon) : Prop :=
  sorry

theorem convex_pentagon_with_collinear_intersections_exists : ∃ (p : Pentagon), 
  convex p ∧ 
  (∀ i : Fin 5, ∃ B : Point2D, 
    B = intersect (p.A i) (p.A ((i + 3) % 5)) (p.A ((i + 1) % 5)) (p.A ((i + 2) % 5))) ∧
  (collinear (List.map 
    (fun i => intersect (p.A i) (p.A ((i + 3) % 5)) (p.A ((i + 1) % 5)) (p.A ((i + 2) % 5))) 
    [0, 1, 2, 3, 4])) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_with_collinear_intersections_exists_l1103_110395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_power_four_l1103_110360

open Real MeasureTheory

/-- The definite integral of 2^4 * sin^4(x/2) * cos^4(x/2) from 0 to π equals 3π/8 -/
theorem integral_sin_cos_power_four :
  ∫ x in Set.Icc 0 π, (2^4 * Real.sin (x/2)^4 * Real.cos (x/2)^4) = 3*π/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_power_four_l1103_110360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_hexagon_area_ratio_l1103_110342

/-- The ratio of the area of a regular hexagon inscribed in a circle
    to the area of a regular hexagon circumscribed about the same circle -/
theorem inscribed_circumscribed_hexagon_area_ratio :
  ∀ r : ℝ, r > 0 →
  (3 * Real.sqrt 3 / 2 * r^2) / (2 * Real.sqrt 3 * r^2) = 3 / 4 := by
  intro r hr
  -- Simplify the expression
  have h1 : (3 * Real.sqrt 3 / 2 * r^2) / (2 * Real.sqrt 3 * r^2) = 
            (3 * Real.sqrt 3 / 2) / (2 * Real.sqrt 3) := by
    -- Proof of this step goes here
    sorry
  -- Simplify further
  have h2 : (3 * Real.sqrt 3 / 2) / (2 * Real.sqrt 3) = 3 / 4 := by
    -- Proof of this step goes here
    sorry
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_hexagon_area_ratio_l1103_110342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l1103_110316

/-- Definition of the ellipse -/
def ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 2 + p.2^2 = 1

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The fixed point Q -/
def Q : ℝ × ℝ := (-2, 0)

/-- A line passing through F and a point p -/
def line_through_F (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(t, s) | ∃ k, s - F.2 = k * (t - F.1) ∧ s - p.2 = k * (t - p.1)}

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

theorem ellipse_fixed_point :
  ∀ M N : ℝ × ℝ,
  ellipse M →
  ellipse N →
  M ≠ N →
  (∃ l : Set (ℝ × ℝ), l = line_through_F M ∧ N ∈ l) →
  M.2 ≠ 0 ∨ N.2 ≠ 0 →
  distance_to_line F (line_through_F Q) =
  distance_to_line F (line_through_F N) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l1103_110316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1103_110310

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_range (ω φ : ℝ) (h1 : ω > 0) (h2 : abs φ < π / 2) 
  (h3 : (π / ω) / 2 = π / 2)
  (h4 : ∀ x, f ω φ (x - π / 8) = -f ω φ (-x - π / 8)) :
  Set.range (fun x ↦ f ω φ x) = Set.Icc (-Real.sqrt 2 / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1103_110310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_arrival_time_l1103_110399

/-- Represents the time taken for both cyclists to reach the destination -/
noncomputable def total_time : ℝ := 620

/-- Rudolph's biking rate in miles per minute -/
noncomputable def rudolph_rate : ℝ := 50 / (total_time - 245)

/-- Jennifer's biking rate in miles per minute -/
noncomputable def jennifer_rate : ℝ := (3/4) * rudolph_rate

/-- The distance traveled by both cyclists -/
def distance : ℝ := 50

/-- Rudolph's break time in minutes -/
def rudolph_break_time : ℝ := 245

/-- Jennifer's break time in minutes -/
def jennifer_break_time : ℝ := 120

/-- Theorem stating that both cyclists arrive at the same time -/
theorem cyclists_arrival_time :
  (rudolph_rate * (total_time - rudolph_break_time) = distance) ∧
  (jennifer_rate * (total_time - jennifer_break_time) = distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_arrival_time_l1103_110399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l1103_110337

-- Define the cone properties
noncomputable def slant_height : ℝ := 4
noncomputable def slant_angle : ℝ := 30 * Real.pi / 180  -- Convert to radians

-- Define the lateral surface area of the cone
noncomputable def lateral_surface_area (s : ℝ) (θ : ℝ) : ℝ :=
  Real.pi * s^2 * Real.sin θ

-- Theorem statement
theorem cone_lateral_surface_area :
  lateral_surface_area slant_height slant_angle = 8 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l1103_110337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l1103_110358

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_equation_and_fixed_point 
  (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 3 / 2) 
  (A B : PointOnEllipse e)
  (h_vertices : A.x = -e.a ∧ B.x = e.a)
  (P : PointOnEllipse e)
  (h_distinct : P ≠ A ∧ P ≠ B)
  (h_isosceles : ‖(A.x - P.x, A.y - P.y)‖ = ‖(B.x - P.x, B.y - P.y)‖)
  (h_area : abs ((A.x - P.x) * (B.y - P.y) - (B.x - P.x) * (A.y - P.y)) / 2 = 2) :
  (∃ (k m : ℝ), e.a = 2 ∧ e.b = 1) ∧ 
  (∀ (M : ℝ × ℝ) (Q : PointOnEllipse e),
    (∃ t : ℝ, M = (1 - t) • ⟨A.x, A.y⟩ + t • ⟨P.x, P.y⟩) →
    M.1 = 4 →
    (∃ s : ℝ, Q.x = (1 - s) * M.1 + s * B.x ∧ Q.y = (1 - s) * M.2 + s * B.y) →
    ∃ (k : ℝ), P.y - Q.y = k * (P.x - Q.x) ∧ P.y - 0 = k * (P.x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l1103_110358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_mod_100_l1103_110398

/-- Recursively defined sequence a_n -/
def a : ℕ → ℕ
  | 0 => 7  -- Define for 0 to cover all natural numbers
  | 1 => 7
  | n + 2 => 7^(a (n + 1))

/-- Theorem: The 2007th term of sequence a_n is congruent to 43 modulo 100 -/
theorem a_2007_mod_100 : a 2007 ≡ 43 [ZMOD 100] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_mod_100_l1103_110398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_gain_percentage_is_five_percent_l1103_110313

/-- Calculates the gain percentage given the cost price and selling price -/
noncomputable def gainPercentage (costPrice sellingPrice : ℝ) : ℝ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Proves that the original gain percentage is 5% given the conditions -/
theorem original_gain_percentage_is_five_percent 
  (costPrice : ℝ)
  (sellingPrice : ℝ)
  (h1 : costPrice = 200)
  (h2 : gainPercentage (costPrice * 0.95) (sellingPrice - 1) = 10) :
  gainPercentage costPrice sellingPrice = 5 := by
  sorry

#check original_gain_percentage_is_five_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_gain_percentage_is_five_percent_l1103_110313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_T_l1103_110340

def geometric_sequence (a : ℕ → ℚ) := ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0

def arithmetic_sequence (a : ℕ → ℚ) := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

noncomputable def S (n : ℕ) : ℚ := sorry

noncomputable def T (n : ℕ) : ℚ := S n + 1 / S n

theorem geometric_sequence_max_T :
  ∀ a : ℕ → ℚ,
  geometric_sequence a →
  a 0 = 3/2 →
  arithmetic_sequence (λ n ↦ [-2 * S 1, S 2, 4 * S 3].get! n) →
  (∀ n : ℕ, T n ≤ 13/6) ∧ (∃ n : ℕ, T n = 13/6) :=
by
  sorry

#check geometric_sequence_max_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_T_l1103_110340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_and_dot_product_range_l1103_110327

noncomputable def ellipse (x y : ℝ) : Prop := y^2/3 + x^2 = 1

noncomputable def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 1

noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

noncomputable def lower_vertex : ℝ × ℝ := (0, -Real.sqrt 3)

noncomputable def fixed_point : ℝ × ℝ := (0, -2 * Real.sqrt 3)

noncomputable def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

noncomputable def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_equation_and_fixed_point_and_dot_product_range :
  ∀ (x1 y1 x2 y2 : ℝ),
  hyperbola 0 (Real.sqrt 2) →  -- Assuming the foci of the hyperbola are at (0, ±√2)
  ellipse x1 y1 →
  ellipse x2 y2 →
  (x1, y1) ≠ lower_vertex →
  (x2, y2) ≠ lower_vertex →
  (x1, y1) ≠ (x2, y2) →
  line_slope 0 (-Real.sqrt 3) x1 y1 * line_slope 0 (-Real.sqrt 3) x2 y2 = 1 →
  (∃ (k t : ℝ), y1 = k * x1 + t ∧ y2 = k * x2 + t ∧ t = -2 * Real.sqrt 3) ∧
  -3 < dot_product x1 y1 x2 y2 ∧ dot_product x1 y1 x2 y2 < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_and_dot_product_range_l1103_110327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1103_110307

/-- The volume of a cylindrical tank -/
noncomputable def cylindrical_tank_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical water tank with diameter 20 feet and depth 10 feet is 1000π cubic feet -/
theorem water_tank_volume :
  cylindrical_tank_volume 20 10 = 1000 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1103_110307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_creative_value_l1103_110304

def letter_value (n : ℕ) : ℤ :=
  match n % 12 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 2
  | 5 => 1
  | 6 => 0
  | 7 => -1
  | 8 => -2
  | 9 => -3
  | 10 => -2
  | 11 => -1
  | _ => 0

def letter_position (c : Char) : ℕ :=
  (c.toNat - 'a'.toNat + 1)

def word_value (word : String) : ℤ :=
  word.toList.map (fun c => letter_value (letter_position c)) |>.sum

theorem creative_value :
  word_value "creative" = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_creative_value_l1103_110304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1103_110343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a

noncomputable def g (x : ℝ) : ℝ := x + 4 / x

theorem min_a_value (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 1 3, ∃ x₂ ∈ Set.Icc 1 4, f a x₁ ≥ g x₂) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1103_110343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_iff_l1103_110370

noncomputable def my_sequence (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => 2^n - 3 * my_sequence a₀ n

def is_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s (n + 1) > s n

theorem sequence_increasing_iff (a₀ : ℝ) :
  is_increasing (my_sequence a₀) ↔ a₀ = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_iff_l1103_110370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1103_110309

noncomputable section

-- Define the hyperbola C
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ x^2 / a - y^2 / (a^2 - a + 4) = 1

-- Define eccentricity
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (a + 4/a)

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_properties :
  ∀ a : ℝ,
  (∀ x y : ℝ, hyperbola a x y →
    (∃ e_min : ℝ, e_min = 2 ∧ ∀ e : ℝ, e = eccentricity a → e ≥ e_min) ∧
    (eccentricity a = 2 → ∀ x y : ℝ, asymptotes x y) ∧
    (∀ l : Set (ℝ × ℝ), ∀ A B C D : ℝ × ℝ,
      (A ∈ l ∧ B ∈ l ∧ hyperbola a A.1 A.2 ∧ hyperbola a B.1 B.2) →
      (C ∈ l ∧ D ∈ l ∧ asymptotes C.1 C.2 ∧ asymptotes D.1 D.2) →
      (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
    (a = 1 →
      ∀ A E F : ℝ × ℝ,
      hyperbola 1 A.1 A.2 →
      asymptotes E.1 E.2 ∧ asymptotes F.1 F.2 →
      ∃ k : ℝ, k > 0 ∧
        abs ((E.1 - F.1) * (A.2 - F.2) - (A.1 - F.1) * (E.2 - F.2)) = k)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1103_110309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l1103_110312

-- Define the points
variable (A B C D F : ℝ × ℝ)

-- Define the conditions
def right_angle (P Q R : ℝ × ℝ) : Prop := sorry
def length (P Q : ℝ × ℝ) : ℝ := sorry
def intersect (P Q R S : ℝ × ℝ) (T : ℝ × ℝ) : Prop := sorry
def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_difference (h1 : right_angle F A B)
                        (h2 : right_angle A B C)
                        (h3 : length A B = 5)
                        (h4 : length B C = 8)
                        (h5 : length A F = 10)
                        (h6 : intersect A C B F D) :
  area A D F - area B D C = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l1103_110312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l1103_110354

theorem m_greater_than_n (a : ℝ) : 
  (5 * a^2 - a + 1) > (4 * a^2 + a - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l1103_110354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1103_110372

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 - 2 * (Real.cos x)^2 - m

-- State the theorem
theorem range_of_m : 
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x m = 0) ↔ m ∈ Set.Icc (-1) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1103_110372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1103_110381

/-- Given vectors a and b in a real inner product space, 
    if |a| = 1, a · b = 3/2, and |a + b| = 2√2, then |b| = √5 -/
theorem magnitude_of_b {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V) 
    (h1 : ‖a‖ = 1)
    (h2 : inner a b = (3 : ℝ) / 2)
    (h3 : ‖a + b‖ = 2 * Real.sqrt 2) :
    ‖b‖ = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1103_110381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_three_l1103_110382

/-- Represents the expected number of draws to reach a multiple of 30 Ft -/
noncomputable def expected_draws (n : ℕ) : ℝ :=
  (2 * (1 / (n + 1 : ℝ))) + (2 * (1 / (n + 1 : ℝ))) + (5 * (1 / (n + 1 : ℝ))) + (3 * ((n - 2 : ℝ) / (n + 1 : ℝ)))

/-- Theorem stating that the expected number of draws is always 3 -/
theorem expected_draws_is_three (n : ℕ) (h : n ≥ 4) :
  expected_draws n = 3 := by
  sorry

#check expected_draws_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_three_l1103_110382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l1103_110336

noncomputable def reward_function (a : ℝ) (x : ℝ) : ℝ := (10 * x - 3 * a) / (x + 2)

theorem minimum_a_value :
  ∃ (a : ℕ), (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 →
    (reward_function (a : ℝ) x ≤ 9 ∧
     reward_function (a : ℝ) x ≤ x / 5 ∧
     (∀ y : ℝ, 10 ≤ y ∧ y ≤ x → reward_function (a : ℝ) y ≤ reward_function (a : ℝ) x))) ∧
  (∀ b : ℕ, b < a →
    ¬(∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 →
      (reward_function (b : ℝ) x ≤ 9 ∧
       reward_function (b : ℝ) x ≤ x / 5 ∧
       (∀ y : ℝ, 10 ≤ y ∧ y ≤ x → reward_function (b : ℝ) y ≤ reward_function (b : ℝ) x)))) ∧
  a = 328 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l1103_110336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_at_origin_l1103_110321

theorem tangent_line_sin_at_origin :
  ∃ (x₀ : ℝ), 
    (Real.sin x₀ = 0) ∧ 
    (Real.cos x₀ = 1) ∧ 
    (∀ x : ℝ, Real.sin x ≤ x) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |Real.sin x - (x - x₀)| < ε * |x - x₀|) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_at_origin_l1103_110321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_polynomial_values_l1103_110329

theorem infinite_primes_dividing_polynomial_values (f : Polynomial ℤ) 
  (h_deg : 1 ≤ f.natDegree) : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℤ, f.eval n ≠ 0 ∧ (p : ℤ) ∣ f.eval n} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_polynomial_values_l1103_110329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_exists_l1103_110348

/-- A word is a list of booleans, where true represents a dot and false represents a dash. -/
def Word := List Bool

/-- The set of all possible ten-digit words. -/
def AllWords : Set Word :=
  {w : Word | w.length = 10}

/-- The Hamming distance between two words. -/
def hammingDistance (w1 w2 : Word) : Nat :=
  (w1.zip w2).filter (fun (b1, b2) => b1 ≠ b2) |>.length

/-- A valid partition of words satisfies the condition that any two words in the same group
    differ in at least three positions. -/
def isValidPartition (p1 p2 : Set Word) : Prop :=
  (∀ w1 w2, w1 ∈ p1 → w2 ∈ p1 → w1 ≠ w2 → hammingDistance w1 w2 ≥ 3) ∧
  (∀ w1 w2, w1 ∈ p2 → w2 ∈ p2 → w1 ≠ w2 → hammingDistance w1 w2 ≥ 3) ∧
  (p1 ∪ p2 = AllWords) ∧
  (p1 ∩ p2 = ∅)

/-- Theorem stating that a valid partition of all ten-digit words does not exist. -/
theorem no_valid_partition_exists : ¬∃ (p1 p2 : Set Word), isValidPartition p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_exists_l1103_110348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_charges_theorem_percentage_decrease_theorem_l1103_110339

/-- Represents the charge for a single room at a hotel -/
structure HotelCharge where
  amount : ℝ
  amount_pos : amount > 0

/-- The hotel charges satisfy the given conditions -/
structure HotelCharges where
  G : HotelCharge
  R : HotelCharge
  P : HotelCharge
  R_greater_G : R.amount = 1.8 * G.amount
  P_less_R : P.amount = 0.5 * R.amount

theorem hotel_charges_theorem (charges : HotelCharges) :
  charges.P.amount = 0.9 * charges.G.amount := by
  sorry

theorem percentage_decrease_theorem (charges : HotelCharges) :
  (charges.G.amount - charges.P.amount) / charges.G.amount = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_charges_theorem_percentage_decrease_theorem_l1103_110339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1103_110306

theorem angle_properties (α : Real) (h1 : α ∈ Set.Icc π (3*π/2)) (h2 : Real.sin α = -3/5) :
  Real.tan α = 3/4 ∧ Real.tan (α - π/4) = -1/7 ∧ Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1103_110306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_from_lcm_and_gcd_l1103_110346

theorem product_from_lcm_and_gcd (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  lcm a b = 560 → gcd a b = 75 → a * b = 42000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_from_lcm_and_gcd_l1103_110346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_truncator_more_wins_probability_l1103_110341

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The probability of having more wins than losses -/
def more_wins_probability : ℚ := 2741/6561

/-- Theorem stating the probability of Club Truncator having more wins than losses -/
theorem club_truncator_more_wins_probability :
  let outcomes := (3 : ℕ) ^ num_matches
  let balanced_outcomes := Nat.choose num_matches 0 * Nat.choose (num_matches - 0) 4 +
                           Nat.choose num_matches 2 * Nat.choose (num_matches - 2) 3 +
                           Nat.choose num_matches 4 * Nat.choose (num_matches - 4) 2 +
                           Nat.choose num_matches 6 * Nat.choose (num_matches - 6) 1 +
                           Nat.choose num_matches 8
  (1 - (balanced_outcomes : ℚ) / outcomes) / 2 = more_wins_probability := by
  sorry

#eval num_matches
#eval match_probability
#eval more_wins_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_truncator_more_wins_probability_l1103_110341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_b_power_n_l1103_110318

def num_factors (m : ℕ) : ℕ :=
  (Finset.filter (· ∣ m) (Finset.range m)).card

theorem max_factors_b_power_n (b n : ℕ) (hb : b ≤ 15) (hn : n ≤ 15) (hn_eq : n = 10) :
  (∀ b' : ℕ, b' ≤ 15 → num_factors (b' ^ n) ≤ num_factors (b ^ n)) →
  num_factors (b ^ n) = 121 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_b_power_n_l1103_110318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_proof_l1103_110397

/-- Calculates the annual interest rate given loan details -/
noncomputable def calculate_annual_interest_rate (principal : ℝ) (total_repayment : ℝ) (months : ℝ) : ℝ :=
  let interest := total_repayment - principal
  let years := months / 12
  (interest / (principal * years)) * 100

/-- The loan details and the expected interest rate -/
theorem loan_interest_rate_proof :
  let principal := (150 : ℝ)
  let total_repayment := (162 : ℝ)
  let months := (18 : ℝ)
  let calculated_rate := calculate_annual_interest_rate principal total_repayment months
  abs (calculated_rate - 5) < 0.5 := by
  -- Proof steps would go here
  sorry

-- Use #eval only for functions that can be computed
def approximate_rate : ℚ := 
  let principal := 150
  let total_repayment := 162
  let months := 18
  let interest := total_repayment - principal
  let years := months / 12
  (interest / (principal * years)) * 100

#eval approximate_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_proof_l1103_110397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_divides_space_into_two_parts_l1103_110363

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define 3D space
def Space := ℝ × ℝ × ℝ

-- Define the division of space by a plane
def DividedSpace (p : Plane) (s : Set Space) : Prop :=
  -- This is a placeholder definition. In a full implementation, this would
  -- precisely define how a plane divides space.
  True

-- Theorem statement
theorem plane_divides_space_into_two_parts (p : Plane) :
  ∃ (part1 part2 : Set Space),
    (DividedSpace p part1) ∧ 
    (DividedSpace p part2) ∧ 
    (part1 ∪ part2 = Set.univ) ∧ 
    (part1 ∩ part2 = ∅) ∧
    (part1 ≠ ∅) ∧ 
    (part2 ≠ ∅) :=
by
  sorry

#check plane_divides_space_into_two_parts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_divides_space_into_two_parts_l1103_110363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l1103_110386

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define monotonically increasing on (-∞, 0)
def monotonic_increasing_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y

-- Define even symmetry
def even_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define odd symmetry
def odd_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define having a maximum value
def has_maximum (f : ℝ → ℝ) : Prop :=
  ∃ M, ∀ x, f x ≤ M

theorem function_existence : 
  ∃ f : ℝ → ℝ, 
    monotonic_increasing_neg f ∧ 
    (even_symmetry f ∨ odd_symmetry f) ∧ 
    has_maximum f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l1103_110386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_ratio_l1103_110353

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := sorry

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := sorry

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := sorry

/-- Donna bought 3 times as many pencils as Marcia -/
axiom donna_triple : donna_pencils = 3 * marcia_pencils

/-- Donna and Marcia bought 480 pencils in total -/
axiom total_pencils : donna_pencils + marcia_pencils = 480

/-- Cindi spent $30 on pencils costing $0.50 each -/
axiom cindi_spent : cindi_pencils = 60

/-- The ratio of pencils Marcia bought to pencils Cindi bought is 2:1 -/
theorem pencil_ratio : marcia_pencils / cindi_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_ratio_l1103_110353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1103_110317

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem vector_sum_magnitude : 
  Real.sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1103_110317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_engine_efficiency_efficiency_approx_8_8_percent_l1103_110378

/-- Represents a heat engine cycle with specific properties -/
structure HeatEngineCycle where
  /-- The working substance is an ideal monoatomic gas -/
  is_ideal_monoatomic : Bool
  /-- The maximum temperature is twice the minimum temperature -/
  temp_ratio : ℝ
  /-- The cycle includes an isochoric process -/
  has_isochoric_process : Bool
  /-- The cycle includes an isothermal process -/
  has_isothermal_process : Bool
  /-- The cycle includes a process where pressure is directly proportional to volume -/
  has_pv_proportional_process : Bool

/-- Calculates the efficiency of the heat engine -/
noncomputable def calculate_efficiency (cycle : HeatEngineCycle) : ℝ :=
  (1.5 + Real.log 2 - 2) / (1.5 + Real.log 2)

/-- Theorem stating the efficiency of the specific heat engine cycle -/
theorem heat_engine_efficiency (cycle : HeatEngineCycle) 
  (h1 : cycle.is_ideal_monoatomic = true)
  (h2 : cycle.temp_ratio = 2)
  (h3 : cycle.has_isochoric_process = true)
  (h4 : cycle.has_isothermal_process = true)
  (h5 : cycle.has_pv_proportional_process = true) :
  calculate_efficiency cycle = (1.5 + Real.log 2 - 2) / (1.5 + Real.log 2) :=
by
  -- The proof is omitted for now
  sorry

/-- The efficiency of the heat engine is approximately 8.8% -/
theorem efficiency_approx_8_8_percent (cycle : HeatEngineCycle) 
  (h1 : cycle.is_ideal_monoatomic = true)
  (h2 : cycle.temp_ratio = 2)
  (h3 : cycle.has_isochoric_process = true)
  (h4 : cycle.has_isothermal_process = true)
  (h5 : cycle.has_pv_proportional_process = true) :
  ∃ ε > 0, |calculate_efficiency cycle - 0.088| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_engine_efficiency_efficiency_approx_8_8_percent_l1103_110378
