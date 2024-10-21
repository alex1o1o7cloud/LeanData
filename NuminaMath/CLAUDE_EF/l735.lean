import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_property_l735_73518

/-- Represents the common ratio of the geometric progression -/
noncomputable def r : ℝ := (1 + Real.sqrt 5) / 2

/-- Represents the nth term of the geometric progression -/
noncomputable def a (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * r ^ (n - 1)

/-- Theorem stating that for n ≥ 2, each term is the difference of the two preceding terms -/
theorem geometric_progression_property (n : ℕ) (a₁ : ℝ) (h : n ≥ 2) :
  a n a₁ = a (n - 1) a₁ - a (n - 2) a₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_property_l735_73518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_theorem_l735_73538

/-- The value of 'a' for which the tangent line of y = x + ln x at (1, 1) intersects y = ax^2 + (a+2)x + 1 -/
def tangent_intersection_a : ℝ := 8

/-- The first curve -/
noncomputable def curve1 (x : ℝ) : ℝ := x + Real.log x

/-- The second curve parameterized by 'a' -/
def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

/-- The tangent line of curve1 at x = 1 -/
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

theorem tangent_intersection_theorem :
  ∃! (a : ℝ), a = tangent_intersection_a ∧
  ∃! (x : ℝ), curve2 a x = tangent_line x ∧
  (∀ y : ℝ, y ≠ x → curve2 a y ≠ tangent_line y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_theorem_l735_73538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_proof_l735_73555

/-- Represents a grid-like town structure -/
structure Town where
  intersections : Nat
  is_square : intersections = 36

/-- Represents a path through the town -/
structure TownPath (t : Town) where
  length : Nat
  start_end_same_color : Bool
  no_revisit : Bool

/-- The longest possible path in the town -/
def longest_path (t : Town) : Nat :=
  34

/-- Theorem stating that the longest path is at most 34 -/
theorem longest_path_proof (t : Town) (p : TownPath t) :
  p.start_end_same_color →
  p.no_revisit →
  p.length ≤ longest_path t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_proof_l735_73555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l735_73501

theorem quadratic_equation_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let equations := [
    (a, b, c), (a, c, b), (b, a, c),
    (b, c, a), (c, a, b), (c, b, a)
  ]
  let has_root (eq : ℝ × ℝ × ℝ) := 
    let (x, y, z) := eq
    (y^2 - 4*x*z) ≥ 0
  (has_root (equations.get! 0) ∧ 
   has_root (equations.get! 1) ∧ 
   has_root (equations.get! 2) ∧ 
   ¬has_root (equations.get! 3) ∧ 
   ¬has_root (equations.get! 4)) →
  has_root (equations.get! 5) :=
by
  intros equations has_root h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l735_73501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_pedal_congruent_to_original_l735_73552

-- Define a triangle type
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_of_angles : α + β + γ = Real.pi

-- Define the pedal triangle function
noncomputable def pedal_triangle (t : Triangle) : Triangle where
  α := 2 * t.α
  β := if t.β > Real.pi/2 then 2 * t.β - Real.pi else 2 * t.β
  γ := if t.γ > Real.pi/2 then 2 * t.γ - Real.pi else 2 * t.γ
  sum_of_angles := by sorry

-- Define the original triangle with angles in ratio 1:4:10
noncomputable def original_triangle : Triangle where
  α := Real.pi / 15
  β := 4 * Real.pi / 15
  γ := 2 * Real.pi / 3
  sum_of_angles := by sorry

-- Define the fourth pedal triangle
noncomputable def fourth_pedal_triangle : Triangle :=
  pedal_triangle (pedal_triangle (pedal_triangle (pedal_triangle original_triangle)))

-- Theorem statement
theorem fourth_pedal_congruent_to_original :
  fourth_pedal_triangle = original_triangle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_pedal_congruent_to_original_l735_73552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_part1_range_a_part2_l735_73553

/-- Function f(x) = x + a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

/-- Function g(x) = x^2 - 2ax + 2 -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- Function F(x) defined piecewise -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then f a x else g a x

/-- Theorem for the first part of the problem -/
theorem range_a_part1 (a : ℝ) :
  (∀ x ≥ 2, f a x ≥ 2) ∧ 
  (∀ x ≥ 2, ∀ y ≥ 2, x ≤ y → g a x ≤ g a y) →
  0 ≤ a ∧ a ≤ 2 := by
  sorry

/-- Theorem for the second part of the problem -/
theorem range_a_part2 (a : ℝ) :
  (∀ x₁ ≥ 2, ∃ x₂ < 2, F a x₁ = F a x₂) →
  a ≤ -1/2 ∨ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_part1_range_a_part2_l735_73553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainville_2000_rainfall_l735_73564

/-- Represents the rainfall data for Rainville in 2000 -/
structure RainfallData where
  /-- Average monthly rainfall from January to June in mm -/
  first_half_avg : ℚ
  /-- Increase in average monthly rainfall starting from July in mm -/
  second_half_increase : ℚ
  /-- Number of months in a year -/
  months_in_year : ℕ

/-- Calculates the total rainfall in Rainville for 2000 given the rainfall data -/
def total_rainfall (data : RainfallData) : ℚ :=
  (data.first_half_avg * (data.months_in_year / 2)) +
  ((data.first_half_avg + data.second_half_increase) * (data.months_in_year / 2))

/-- Theorem stating that the total rainfall in Rainville for 2000 was 390 mm -/
theorem rainville_2000_rainfall :
  let data : RainfallData := {
    first_half_avg := 30,
    second_half_increase := 5,
    months_in_year := 12
  }
  total_rainfall data = 390 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainville_2000_rainfall_l735_73564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_is_zero_l735_73508

/-- Represents the state of the program -/
structure ProgramState where
  n : Int
  s : Int

/-- Simulates one iteration of the loop -/
def iterate (state : ProgramState) : ProgramState :=
  { n := state.n - 1, s := state.s + state.n }

/-- Simulates the entire program execution -/
def execute (initial : ProgramState) : Int :=
  let rec loop (state : ProgramState) (fuel : Nat) : Int :=
    if fuel = 0 then
      state.n  -- Emergency exit to ensure termination
    else if state.s < 15 then
      loop (iterate state) (fuel - 1)
    else
      state.n
  loop initial 100  -- Arbitrary large number for maximum iterations

/-- The main theorem to prove -/
theorem final_n_is_zero :
  execute { n := 5, s := 0 } = 0 := by
  sorry

#eval execute { n := 5, s := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_is_zero_l735_73508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l735_73537

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y => x - a * y + 1 = 0
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y => (a - 1) * x - 12 * y - 4 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ g (k * x) (k * y)

-- Theorem statement
theorem parallel_condition :
  ∀ a : ℝ, (a = 4 ↔ parallel (l₁ a) (l₂ a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l735_73537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_is_16_l735_73584

/-- Represents a row in the random number table -/
def RandomNumberRow := List Nat

/-- The random number table -/
def randomNumberTable : List RandomNumberRow := [
  [2748, 6198, 7164, 4148, 7086, 2888, 8519, 1620],
  [7477, 0111, 1630, 2404, 2979, 7991, 9683, 5125]
]

/-- The starting position for selection (9th and 10th columns of 6th row) -/
def startPosition : Nat × Nat := (5, 8)

/-- The number of individuals to select -/
def selectCount : Nat := 6

/-- The position we're interested in (5th selected individual) -/
def targetPosition : Nat := 5

/-- Function to select numbers from the table -/
def selectNumbers (table : List RandomNumberRow) (start : Nat × Nat) (count : Nat) : List Nat :=
  sorry

theorem fifth_selected_is_16 :
  let selected := selectNumbers randomNumberTable startPosition selectCount
  List.get! selected (targetPosition - 1) = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_is_16_l735_73584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l735_73503

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (3 + Real.sin θ, Real.cos θ)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∃ (max_dist : ℝ), max_dist = 4 ∧
  (∀ θ : ℝ, distance O (C θ) ≤ max_dist) ∧
  (∃ θ : ℝ, distance O (C θ) = max_dist) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l735_73503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_slope_l735_73551

-- Define the curve y = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the tangent line
noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := (1 / a) * (x - a) + f a

-- Theorem statement
theorem tangent_through_origin_slope (a : ℝ) (h1 : a > 0) :
  tangent_line a 0 = 0 → (1 / a) = 1 / Real.exp 1 := by
  sorry

#check tangent_through_origin_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_slope_l735_73551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l735_73548

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2*a}

theorem subset_count : 
  ∃! (S : Finset ℝ), 
    (∀ a ∈ S, B a ⊆ A) ∧ 
    (∀ a ∉ S, ¬(B a ⊆ A)) ∧
    Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l735_73548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_shape_l735_73543

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- A cylinder in 3D space -/
def Cylinder (c : ℝ) : Set CylindricalCoord :=
  {p : CylindricalCoord | p.r = c}

/-- Predicate to check if a set of points forms a cylinder -/
def IsACylinder (S : Set CylindricalCoord) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ 
    ∀ p, p ∈ S ↔ p.r = c ∧ p.θ ∈ Set.Icc 0 (2 * Real.pi) ∧ p.z ∈ Set.univ

/-- The main theorem -/
theorem cylinder_shape (c : ℝ) (h : c > 0) :
  ∃ (S : Set CylindricalCoord), S = Cylinder c ∧ IsACylinder S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_shape_l735_73543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laborers_attendance_percentage_l735_73507

def total_laborers : ℕ := 26
def present_laborers : ℕ := 10

def percentage_present : ℚ :=
  (present_laborers : ℚ) / (total_laborers : ℚ) * 100

noncomputable def rounded_percentage : ℚ :=
  ⌊percentage_present * 10 + 0.5⌋ / 10

theorem laborers_attendance_percentage :
  rounded_percentage = 385 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laborers_attendance_percentage_l735_73507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_in_terms_of_a_l735_73590

theorem log_expression_in_terms_of_a (a : ℝ) (h : (3 : ℝ)^a = 2) :
  Real.log 8 / Real.log 3 - 2 * (Real.log 6 / Real.log 3) = a - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_in_terms_of_a_l735_73590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_engineering_department_men_count_l735_73514

theorem engineering_department_men_count 
  (total : ℕ) 
  (men_percentage : ℚ) 
  (women_count : ℕ) 
  (h1 : men_percentage = 7/10) 
  (h2 : women_count = 180) 
  (h3 : ↑women_count = (1 - men_percentage) * ↑total) : 
  (men_percentage * ↑total).floor = 420 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_engineering_department_men_count_l735_73514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sevenths_rounded_l735_73505

/-- Rounds a real number to a specified number of decimal places -/
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋) / 10^n

/-- The fraction 8/7 rounded to 3 decimal places is equal to 1.143 -/
theorem eight_sevenths_rounded : round_to_decimal_places (8/7) 3 = 1.143 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sevenths_rounded_l735_73505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_mod_n_l735_73527

theorem xy_mod_n (n : ℕ+) (x y : ZMod n) (hx : IsUnit x) (hy : IsUnit y)
  (hxy : x = 2 * y) (hyx : y = 3 * x⁻¹) :
  x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_mod_n_l735_73527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_C₁_C₂_l735_73580

/-- Polar equation of curve C₁ -/
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 4

/-- Polar equation of curve C₂ -/
noncomputable def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The ratio of distances |OB|/|OA| for a given angle α -/
noncomputable def ratio (α : ℝ) : ℝ :=
  (2 * Real.cos α) / (4 / (Real.cos α + Real.sin α))

theorem max_ratio_C₁_C₂ :
  ∃ (α : ℝ), ∀ (β : ℝ), ratio α ≥ ratio β ∧ ratio α = (Real.sqrt 2 + 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_C₁_C₂_l735_73580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_sides_sum_ge_one_l735_73576

/-- Represents a rectangle with a marked side -/
structure MarkedRectangle where
  x : ℝ  -- Length of the marked side
  y : ℝ  -- Length of the other side
  h1 : 0 < x ∧ x ≤ 1
  h2 : 0 < y ∧ y ≤ 1

/-- A partition of a unit square into rectangles with marked sides -/
def SquarePartition := List MarkedRectangle

/-- The sum of areas of all rectangles in a partition equals 1 -/
def valid_partition (p : SquarePartition) : Prop :=
  (p.map (λ r => r.x * r.y)).sum = 1

/-- The theorem statement -/
theorem marked_sides_sum_ge_one (p : SquarePartition) (h : valid_partition p) :
  (p.map (λ r => r.x)).sum ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_sides_sum_ge_one_l735_73576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l735_73567

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  f a (Real.sqrt 2) = 2 → f a 4 = 16 := by
  intro h
  -- The proof steps would go here
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l735_73567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_B_special_triangle_l735_73512

/-- In a triangle ABC, if b is the geometric mean of a and c, and sin A is the arithmetic mean of sin(B-A) and sin C, then cos B = (√5 - 1) / 2 -/
theorem cosine_B_special_triangle (a b c A B C : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_geometric_mean : b^2 = a * c)
  (h_arithmetic_mean : Real.sin A = (Real.sin (B - A) + Real.sin C) / 2) :
  Real.cos B = (Real.sqrt 5 - 1) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_B_special_triangle_l735_73512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l735_73598

/-- Represents the marks of a student in different subjects -/
structure Marks where
  english : ℚ
  mathematics : ℚ
  physics : ℚ
  chemistry : ℚ
  biology : ℚ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem davids_english_marks :
  ∃ m : Marks,
    m.mathematics = 95 ∧
    m.physics = 82 ∧
    m.chemistry = 87 ∧
    m.biology = 92 ∧
    average m = 904 / 10 ∧
    m.english = 96 := by
  sorry

#eval (96 : ℚ) + 95 + 82 + 87 + 92
#eval ((96 : ℚ) + 95 + 82 + 87 + 92) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l735_73598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_is_ellipse_l735_73521

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define membership for Point in Ellipse -/
instance : Membership Point Ellipse where
  mem p e := (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse) where
  p1 : Point
  p2 : Point
  p3 : Point
  (on_ellipse : p1 ∈ e ∧ p2 ∈ e ∧ p3 ∈ e)

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : InscribedTriangle e) : Point :=
  sorry

/-- The geometric locus of orthocenters -/
def orthocenterLocus (e : Ellipse) : Set Point :=
  {p | ∃ t : InscribedTriangle e, orthocenter t = p}

/-- Theorem: The geometric locus of orthocenters is an ellipse -/
theorem orthocenter_locus_is_ellipse (e : Ellipse) :
  ∃ e' : Ellipse, orthocenterLocus e = {p | p ∈ e'} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_is_ellipse_l735_73521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l735_73591

def pillar_count : ℕ := 20
def pillar_height : ℝ := 15
def pillar_diameter : ℝ := 12
def paint_coverage : ℝ := 320

noncomputable def lateral_surface_area (count : ℕ) (height : ℝ) (diameter : ℝ) : ℝ :=
  Real.pi * diameter * height * (count : ℝ)

theorem paint_gallons_needed :
  ⌈lateral_surface_area pillar_count pillar_height pillar_diameter / paint_coverage⌉ = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l735_73591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l735_73550

theorem cos_alpha_minus_pi_third (α : ℝ) 
  (h : Real.cos α + Real.sqrt 3 * Real.sin α = 8 / 5) : 
  Real.cos (α - π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l735_73550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_undefined_inverse_l735_73539

theorem smallest_integer_undefined_inverse (a : ℕ) :
  a > 0 ∧ 
  (∀ x : ℕ, x < a → (∃ y : ℕ, y * x % 99 = 1 ∨ y * x % 45 = 1)) ∧
  (∀ y : ℕ, y * a % 99 ≠ 1 ∧ y * a % 45 ≠ 1) →
  a = 3 := by
  sorry

#check smallest_integer_undefined_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_undefined_inverse_l735_73539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_range_l735_73585

-- Define the exponential function f(x) = (a-1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- State the theorem
theorem decreasing_exp_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_range_l735_73585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l735_73574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2-x) - 3/4

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem fixed_point_power_function 
  (a : ℝ) (α : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 2 = 1/4) 
  (h4 : power_function α 2 = 1/4) :
  power_function α (1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l735_73574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_complement_intersection_range_of_a_l735_73578

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2 * x^2 + 1)) / (Real.sqrt (5 - x)) + Real.sqrt (x - 2)

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 1 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a - 1 ∨ x > a}

-- Theorem statements
theorem domain_of_f : Set.range f = A := by sorry

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2 ∨ 5 ≤ x ∧ x < 8} := by sorry

theorem range_of_a (h : A ∪ C a = Set.univ) :
  {a | A ∪ C a = Set.univ} = Set.Iic 2 ∪ Set.Ioi 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_complement_intersection_range_of_a_l735_73578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_distance_l735_73572

/-- The distance between cities A and B in kilometers -/
noncomputable def distance_AB : ℝ := 245

/-- The speed of the car from city A in km/h -/
noncomputable def speed_A : ℝ := 70

/-- The speed of the car from city B in km/h -/
noncomputable def speed_B : ℝ := 90

/-- The midpoint C between cities A and B -/
noncomputable def midpoint_C : ℝ := distance_AB / 2

/-- The meeting point of the two cars -/
noncomputable def meeting_point : ℝ :=
  midpoint_C - speed_A * (distance_AB / (speed_A + speed_B))

/-- Theorem stating that the meeting point is approximately 15.31 km from the midpoint -/
theorem cars_meet_distance :
  ∀ ε > 0, |meeting_point - (midpoint_C - 15.31)| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_distance_l735_73572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_efficient_lamp_cheaper_l735_73561

/-- Represents the cost calculation for lamp usage --/
structure LampCost where
  wattage : ℝ
  usageHours : ℝ
  tariff : ℝ
  initialCost : ℝ

noncomputable def monthlyCost (l : LampCost) : ℝ :=
  l.wattage * l.usageHours / 1000 * l.tariff

noncomputable def totalCost (l : LampCost) (months : ℕ) : ℝ :=
  l.initialCost + monthlyCost l * (months : ℝ)

noncomputable def companySavings (old : LampCost) (new : LampCost) (months : ℕ) : ℝ :=
  0.75 * (monthlyCost old - monthlyCost new) * (months : ℝ)

theorem energy_efficient_lamp_cheaper (old : LampCost) (new : LampCost) :
  (old.wattage = 60 ∧ old.usageHours = 100 ∧ old.tariff = 5 ∧ old.initialCost = 0) →
  (new.wattage = 12 ∧ new.usageHours = 100 ∧ new.tariff = 5 ∧ new.initialCost = 120) →
  (totalCost new 10 < totalCost old 10 - companySavings old new 10) ∧
  (totalCost new 36 < totalCost old 36 - companySavings old new 10 - monthlyCost new * 26) :=
by sorry

#check energy_efficient_lamp_cheaper

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_efficient_lamp_cheaper_l735_73561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l735_73531

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x / a) + (a / Real.exp x)

theorem f_properties (a : ℝ) (h_a : a > 0) 
  (h_even : ∀ x, f a x = f a (-x)) : 
  (a = 1) ∧ 
  (∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l735_73531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l735_73573

/-- The time it takes for two workers to complete a job together, given their individual completion times -/
theorem job_completion_time (a_time b_time : ℝ) (ha : a_time > 0) (hb : b_time > 0) :
  (a_time = 10 ∧ b_time = 9) →
  (1 / (1 / a_time + 1 / b_time)) = 90 / 19 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l735_73573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_cube_edge_max_cube_edge_correct_l735_73530

/-- The edge length of the regular tetrahedron --/
def tetrahedron_edge : ℝ := 6

/-- The maximum edge length of the inscribed cube --/
noncomputable def max_cube_edge : ℝ := Real.sqrt 2

/-- Function to calculate the maximum edge length of a cube inscribed in a regular tetrahedron --/
noncomputable def max_cube_edge_in_tetrahedron (tetrahedron_edge : ℝ) : ℝ :=
  Real.sqrt 2

/-- Theorem: The maximum edge length of a cube that can be inscribed in a regular tetrahedron
    with edge length 6 is √2 --/
theorem max_inscribed_cube_edge :
  ∃ (cube_edge : ℝ), cube_edge = max_cube_edge ∧
  cube_edge = max_cube_edge_in_tetrahedron tetrahedron_edge :=
by
  use max_cube_edge
  constructor
  · rfl
  · rfl

/-- Proof that the calculated maximum cube edge is correct --/
theorem max_cube_edge_correct :
  max_cube_edge = max_cube_edge_in_tetrahedron tetrahedron_edge :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_cube_edge_max_cube_edge_correct_l735_73530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_equation_l735_73577

/-- Triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Predicate to check if a circle is inscribed in a triangle -/
def is_inscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- The given triangle -/
def given_triangle : Triangle :=
  { A := (-2, 1)
  , B := (2, 5)
  , C := (5, 2) }

/-- The circle to be proved -/
noncomputable def circle_to_prove : Circle :=
  { h := 2
  , k := 3
  , r := Real.sqrt 2 }

theorem inscribed_circle_equation :
  is_inscribed circle_to_prove given_triangle := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_equation_l735_73577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_child_score_l735_73536

/-- Given four children's test scores with an average of 94, where three scores are known (97, 85, and 94), prove that the fourth score must be 100. -/
theorem fourth_child_score (score1 score2 score3 score4 : ℕ) : 
  score1 = 97 →
  score2 = 85 →
  score3 = 94 →
  (score1 + score2 + score3 + score4) / 4 = 94 →
  score4 = 100 := by
  sorry

#check fourth_child_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_child_score_l735_73536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_l735_73575

-- Define what we mean by "well-defined" for a real number
def IsWellDefined (x : ℝ) : Prop := x ≠ 0

theorem fraction_defined (x : ℝ) : IsWellDefined (1 / (x + 1)) ↔ x ≠ -1 := by
  -- Unfold the definition of IsWellDefined
  simp [IsWellDefined]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_l735_73575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l735_73599

/-- The number of days it takes for worker c to complete the work alone -/
noncomputable def c_days : ℝ := 12

/-- The time it takes for all workers together to complete the work -/
noncomputable def total_time : ℝ := 24 / 7

theorem work_completion_time :
  (1 / 24 : ℝ) + (1 / 6 : ℝ) + (1 / c_days) = 1 / total_time := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l735_73599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_4x4_sum_2007_l735_73532

/-- Represents a chessboard with integer values -/
def Chessboard := Matrix (Fin 8) (Fin 8) ℕ

/-- The operation of incrementing a 3x3 square on the chessboard -/
def increment_3x3 (board : Chessboard) (i j : Fin 6) : Chessboard := sorry

/-- Perform the increment operation n times -/
def perform_operations (n : ℕ) (board : Chessboard) : Chessboard := sorry

/-- Initial board with all 1s -/
def initial_board : Chessboard := λ i j ↦ 1

/-- Sum of corners of a 4x4 square -/
def corner_sum (board : Chessboard) (i j : Fin 5) : ℕ := sorry

/-- Main theorem -/
theorem exists_4x4_sum_2007 :
  ∃ i j : Fin 5, corner_sum (perform_operations 2003 initial_board) i j = 2007 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_4x4_sum_2007_l735_73532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_power_plus_one_l735_73568

theorem divisors_of_power_plus_one (n : ℕ) (p : Fin n → ℕ) 
  (h_n : n ≥ 1) 
  (h_p : ∀ i, Prime (p i) ∧ p i ≥ 5) : 
  let N := 2^(Finset.prod Finset.univ (fun i => p i)) + 1
  (Nat.divisors N).card ≥ 2^(2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_power_plus_one_l735_73568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l735_73556

theorem triangle_side_count : 
  let a := 8
  let b := 5
  let possible_sides := Finset.filter (fun x => 
    x + b > a ∧ 
    x + a > b ∧ 
    a + b > x) (Finset.range 13)
  Finset.card possible_sides = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l735_73556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_U_l735_73597

theorem range_of_U (x y : ℝ) (h : (2:ℝ)^x + (3:ℝ)^y = (4:ℝ)^x + (9:ℝ)^y) :
  ∃ (U : ℝ), U = (8:ℝ)^x + (27:ℝ)^y ∧ U ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_U_l735_73597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l735_73534

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * (Real.cos (x / 2))^2

/-- The main theorem -/
theorem triangle_problem (t : Triangle) (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) 
    (h2 : t.a = 2) (h3 : f t.B = Real.sqrt 2 + 1) : 
  t.A = π / 3 ∧ t.b = 2 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l735_73534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_properties_l735_73581

/-- Definition of the ellipse C₁ -/
def C₁ (b : ℝ) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / b^2 = 1

/-- Definition of the curve C₂, symmetric to C₁ about y = -x -/
def C₂ (b : ℝ) (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 / b^2 = 1

/-- Intersection points of C₁ and C₂ -/
def intersection_points (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C₁ b p.1 p.2 ∧ C₂ b p.1 p.2}

/-- Area of the quadrilateral formed by intersection points -/
noncomputable def quadrilateral_area (b : ℝ) : ℝ :=
  4 -- Given in the problem

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_intersection_properties :
  (∃ (x y : ℝ), (x, y) ∈ intersection_points (Real.sqrt (4/3)) ∧ x = 1 ∧ y = 1) ∧
  eccentricity 2 (Real.sqrt (4/3)) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_properties_l735_73581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_k_bound_l735_73570

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 + k * Real.log x

-- State the theorem
theorem function_inequality_implies_k_bound
  (k : ℝ)
  (h : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 →
       (f k x2 - f k x1) / (x2 - x1) > 2022) :
  k ≥ (1011^2) / 2 :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_k_bound_l735_73570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_B_around_A_clockwise_l735_73506

noncomputable section

/-- Rotate a vector (x, y) counterclockwise by angle θ -/
def rotate_vector (x y θ : ℝ) : ℝ × ℝ :=
  (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)

/-- The coordinates of point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- The coordinates of point B -/
def point_B : ℝ × ℝ := (1 + Real.sqrt 3, 4)

/-- The angle of rotation (clockwise) -/
def rotation_angle : ℝ := Real.pi / 3

/-- The expected coordinates of point P after rotation -/
def expected_point_P : ℝ × ℝ := (3 * Real.sqrt 3 / 2 + 1, 3 / 2)

theorem rotate_B_around_A_clockwise :
  let vector_AB := (point_B.1 - point_A.1, point_B.2 - point_A.2)
  let rotated_vector := rotate_vector vector_AB.1 vector_AB.2 (-rotation_angle)
  let point_P := (point_A.1 + rotated_vector.1, point_A.2 + rotated_vector.2)
  point_P = expected_point_P := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_B_around_A_clockwise_l735_73506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_isosceles_triangle_l735_73513

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the radii of the incircle and excircles of a triangle -/
structure CircleRadii where
  inradius : ℝ
  exradius_a : ℝ
  exradius_b : ℝ
  exradius_c : ℝ

/-- Checks if a triangle is isosceles with two sides twice the length of the third -/
def isIsoscelesDoubled (t : Triangle) : Prop :=
  t.b = t.c ∧ t.b = 2 * t.a

/-- Calculates the semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Calculates the radii of the incircle and excircles -/
noncomputable def calculateRadii (t : Triangle) : CircleRadii :=
  let s := semiperimeter t
  let a := area t
  { inradius := a / s
  , exradius_a := a / (s - t.a)
  , exradius_b := a / (s - t.b)
  , exradius_c := a / (s - t.c) }

/-- Checks if the radii satisfy the tangency conditions -/
def satisfiesTangencyConditions (r : CircleRadii) : Prop :=
  r.inradius + r.exradius_a = r.exradius_b - r.inradius ∧
  r.inradius + r.exradius_a = r.exradius_c - r.inradius

/-- The main theorem stating the minimum perimeter of the triangle -/
theorem min_perimeter_isosceles_triangle :
  ∀ t : Triangle,
    isIsoscelesDoubled t →
    satisfiesTangencyConditions (calculateRadii t) →
    t.a + t.b + t.c ≥ 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_isosceles_triangle_l735_73513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_time_is_ten_l735_73583

/-- The number of possible government vehicle license plates starting with 79 -/
def num_plates : ℕ := 900

/-- The probability of finding the transmitter within 3 hours -/
def prob_find : ℚ := 1/50

/-- The total inspection time in minutes -/
def total_time : ℕ := 180

/-- The time to inspect each vehicle in minutes -/
noncomputable def inspection_time : ℚ := total_time / (prob_find * num_plates)

theorem inspection_time_is_ten : inspection_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_time_is_ten_l735_73583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_fourth_derivative_correct_l735_73510

/-- The function y(x) = e^(x/2) * sin(2x) -/
noncomputable def y (x : ℝ) : ℝ := Real.exp (x/2) * Real.sin (2*x)

/-- The fourth derivative of y(x) -/
noncomputable def y_fourth_derivative (x : ℝ) : ℝ := (161/16) * Real.exp (x/2) * Real.sin (2*x) - 15 * Real.exp (x/2) * Real.cos (2*x)

/-- Theorem stating that the fourth derivative of y(x) is equal to y_fourth_derivative(x) -/
theorem y_fourth_derivative_correct : 
  ∀ x, (deriv^[4] y) x = y_fourth_derivative x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_fourth_derivative_correct_l735_73510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tangent_circles_bound_l735_73511

/-- Helper function to calculate the maximum number of smaller circles
    that are tangent to both given circles and mutually non-intersecting. -/
def max_tangent_circles (R r : ℝ) : ℕ := sorry

/-- Given two concentric circles with radii R and r (R > r), this theorem proves
    that the maximum number of smaller circles n that are tangent to both circles
    and mutually non-intersecting satisfies the given inequality. -/
theorem max_tangent_circles_bound (R r : ℝ) (h : R > r) :
  let n := max_tangent_circles R r
  (3/2 : ℝ) * (Real.sqrt R + Real.sqrt r) / (Real.sqrt R - Real.sqrt r) - 1 ≤ n ∧
  (n : ℝ) ≤ (63/20 : ℝ) * (R + r) / (R - r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tangent_circles_bound_l735_73511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_distance_l735_73517

/-- The equation of a parabola is y = x^2 - 10x + c, where c is a real number. -/
def parabola_equation (x y c : ℝ) : Prop :=
  y = x^2 - 10*x + c

/-- The distance between two points (x₁, y₁) and (x₂, y₂) in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The x-coordinate of the vertex of the parabola. -/
def vertex_x : ℝ := 5

/-- The y-coordinate of the vertex of the parabola. -/
def vertex_y (c : ℝ) : ℝ := c - 25

/-- The theorem stating that if the vertex of the parabola y = x^2 - 10x + c
    is exactly 10 units from the origin, then c = 25 + 5√3 or c = 25 - 5√3. -/
theorem parabola_vertex_distance (c : ℝ) :
  distance 0 0 vertex_x (vertex_y c) = 10 →
  c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3 := by
  sorry

#check parabola_vertex_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_distance_l735_73517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l735_73571

/-- The function f(x) = (6x^2 - 11) / (4x^2 + 6x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3)

/-- The horizontal asymptote of f(x) is 3/2 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 3/2| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l735_73571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l735_73529

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the point M on AB
variable (M : EuclideanSpace ℝ (Fin 2))

-- Define points A₁ and B₁
variable (A₁ B₁ : EuclideanSpace ℝ (Fin 2))

-- ABC is an isosceles triangle
axiom isosceles : ‖A - C‖ = ‖B - C‖

-- M is on the line AB
axiom M_on_AB : ∃ t : ℝ, M = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1

-- A₁ is on AC
axiom A₁_on_AC : ∃ s : ℝ, A₁ = (1 - s) • A + s • C ∧ 0 ≤ s ∧ s ≤ 1

-- B₁ is on BC
axiom B₁_on_BC : ∃ r : ℝ, B₁ = (1 - r) • B + r • C ∧ 0 ≤ r ∧ r ≤ 1

-- M, A₁, and B₁ are collinear
axiom collinear : ∃ k : ℝ, A₁ - M = k • (B₁ - M)

-- Theorem to prove
theorem ratio_equality : ‖A - A₁‖ / ‖A₁ - M‖ = ‖B - B₁‖ / ‖B₁ - M‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l735_73529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l735_73540

/-- Calculates the required run rate for the remaining overs in a cricket match -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_in_first_overs := (first_run_rate * first_overs : ℚ)
  let remaining_runs := target - runs_in_first_overs.floor
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_run_rate_theorem :
  required_run_rate 50 10 (32/10) 272 = 6 := by
  sorry

#eval required_run_rate 50 10 (32/10) 272

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l735_73540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_liter_in_pints_l735_73515

-- Define the conversion rate from liters to pints
noncomputable def liters_to_pints (liters : ℝ) : ℝ := 1.575 * (liters / 0.75)

-- Theorem statement
theorem one_liter_in_pints :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |liters_to_pints 1 - 2.1| < ε := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_liter_in_pints_l735_73515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l735_73582

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 2 * (Real.sin x) ^ 2

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : f (A / 2) = 3 / 2) 
  (h2 : B + C = 7) 
  (h3 : (1 / 2) * B * C * Real.sin A = 2 * Real.sqrt 3) :
  ∃ (a : ℝ), a ^ 2 = B ^ 2 + C ^ 2 - 2 * B * C * Real.cos A ∧ a = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l735_73582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l735_73523

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 1 →
    f (x + 1 / y) + f (y + 1 / z) + f (z + 1 / x) = 1

/-- The set of functions that satisfy the functional equation. -/
def FunctionalSolutions : Set (ℝ → ℝ) :=
  {f | SatisfiesFunctionalEquation f}

/-- The set of functions of the form k/(x+1) + (1-k)/3 for k in [-1/2, 1]. -/
def ProposedSolutions : Set (ℝ → ℝ) :=
  {f | ∃ k : ℝ, -1/2 ≤ k ∧ k ≤ 1 ∧
    ∀ x : ℝ, x > 0 → f x = k / (x + 1) + (1 - k) / 3}

/-- The main theorem stating that the functional solutions are exactly the proposed solutions. -/
theorem functional_equation_solutions :
    FunctionalSolutions = ProposedSolutions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l735_73523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_speed_approx_l735_73526

/-- The speed of the moon in kilometers per hour -/
noncomputable def moon_speed_km_per_hour : ℝ := 3744

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℝ := 3600

/-- The speed of the moon in kilometers per second -/
noncomputable def moon_speed_km_per_second : ℝ := moon_speed_km_per_hour / seconds_per_hour

/-- Theorem stating that the moon's speed in km/s is approximately 1.04 -/
theorem moon_speed_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |moon_speed_km_per_second - 1.04| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_speed_approx_l735_73526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_45_plus_37_in_octal_l735_73554

/-- Represents a number in base 8 (octal) --/
structure Octal where
  value : Nat

/-- Adds two octal numbers and returns the result in octal --/
def octal_add (a b : Octal) : Octal :=
  Octal.mk ((a.value + b.value) % 8)

/-- Gets the units digit of an octal number --/
def units_digit (a : Octal) : Octal :=
  Octal.mk (a.value % 8)

/-- Converts a natural number to its octal representation --/
def to_octal (n : Nat) : Octal :=
  Octal.mk n

theorem units_digit_of_45_plus_37_in_octal :
  units_digit (octal_add (to_octal 45) (to_octal 37)) = to_octal 4 := by
  sorry

#eval (octal_add (to_octal 45) (to_octal 37)).value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_45_plus_37_in_octal_l735_73554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_imply_a_range_l735_73587

noncomputable section

/-- The function f(x) = (1/3)x³ + x² - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - a*x

/-- f(x) is monotonically increasing in (1, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f a x < f a y

/-- f(x) has a zero point in (1, 2) -/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 2 ∧ f a x = 0

theorem f_properties_imply_a_range :
  ∀ a : ℝ, is_monotone_increasing a ∧ has_zero_in_interval a →
    4/3 < a ∧ a ≤ 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_imply_a_range_l735_73587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l735_73500

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x - 3 = (y + 2)^2

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 1/2)^2 = 9/2

-- Theorem statement
theorem intersection_points_on_circle :
  ∃ (A B C D : ℝ × ℝ),
    (parabola1 A.1 A.2 ∧ parabola2 A.1 A.2) ∧
    (parabola1 B.1 B.2 ∧ parabola2 B.1 B.2) ∧
    (parabola1 C.1 C.2 ∧ parabola2 C.1 C.2) ∧
    (parabola1 D.1 D.2 ∧ parabola2 D.1 D.2) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) ∧
    (my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧ my_circle C.1 C.2 ∧ my_circle D.1 D.2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l735_73500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l735_73579

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the specific triangle from the problem
noncomputable def specialTriangle : Triangle where
  A := Real.arccos (Real.sqrt 6 / 3)
  B := 2 * Real.arccos (Real.sqrt 6 / 3)
  C := Real.pi - 3 * Real.arccos (Real.sqrt 6 / 3)
  a := 3
  b := 2 * Real.sqrt 6
  c := 5

-- Theorem statement
theorem special_triangle_properties (t : Triangle) 
  (h1 : t.a = 3) 
  (h2 : t.b = 2 * Real.sqrt 6) 
  (h3 : t.B = 2 * t.A) : 
  Real.cos t.A = Real.sqrt 6 / 3 ∧ t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l735_73579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l735_73544

/-- Proves that increasing 240 by 20% results in 288 -/
theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) (final : ℕ) : 
  initial = 240 → percentage = 20 / 100 → final = initial + (percentage * ↑initial).floor → final = 288 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l735_73544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_l735_73594

/-- A function that represents the number of coins in each chest -/
def coin_distribution : Fin 6 → ℕ := sorry

/-- The property that any subset of chests can have their coins evenly distributed -/
def can_distribute (d : Fin 6 → ℕ) : Prop :=
  ∀ (s : Finset (Fin 6)), 2 ≤ s.card → s.card ≤ 5 → 
    ∃ (n : ℕ), s.sum d = n * s.card

/-- The theorem stating that if the distribution satisfies can_distribute, 
    then all chests can have an equal number of coins -/
theorem equal_distribution (d : Fin 6 → ℕ) (h : can_distribute d) :
  ∃ (n : ℕ), (Finset.univ : Finset (Fin 6)).sum d = 6 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_l735_73594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l735_73542

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp (2 * x) + (a - 2) * exp x - x

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → a ≤ 0 → f a x > f a y) ∧
  (a > 0 → ∀ x y : ℝ, x < y → x < log (1/a) → f a x > f a y) ∧
  (a > 0 → ∀ x y : ℝ, x < y → x > log (1/a) → f a x < f a y) ∧
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ↔ 0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l735_73542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_unit_circle_l735_73504

theorem four_points_unit_circle (α : ℝ) : 
  Real.sin α + Real.sin (α + π/2) + Real.sin (α + π) + Real.sin (α + 3*π/2) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_unit_circle_l735_73504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l735_73546

theorem trigonometric_simplification (α : ℝ) :
  (Real.cos (α + π) * Real.sin (-α)) / (Real.cos (-3*π - α) * Real.sin (-α - 4*π)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l735_73546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l735_73549

-- Define the functions
def f (x : ℝ) : ℝ := -x^3 + x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

-- State the theorem
theorem math_problem (a : ℝ) (h_a : a ≠ 0) :
  -- 1. Extreme values of f
  (∀ x : ℝ, f x ≥ 0) ∧ (∃ x : ℝ, f x = 0) ∧
  (∀ x : ℝ, f x ≤ 4/27) ∧ (∃ x : ℝ, f x = 4/27) ∧
  
  -- 2. Condition for a
  ((∀ x : ℝ, x ≥ 1 → f x + g a x ≥ -x^3 + (a+2)*x) → a ≤ -1) ∧
  
  -- 3. Inequality for natural numbers
  (∀ n : ℕ, n > 0 →
    (Finset.sum (Finset.range 2015) (λ i ↦ 1 / Real.log (n + i + 1))) >
    2015 / (n * (n + 2015))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l735_73549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l735_73502

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| < 3}
def B : Set ℝ := {x : ℝ | x - 1 ≤ 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l735_73502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l735_73596

/-- The speed of a car that takes 2 seconds longer to travel 1 kilometer than it would at 90 km/h -/
noncomputable def car_speed : ℝ := 600 / 7

theorem car_speed_proof (v : ℝ) :
  (1 / v - 1 / 90) * 3600 = 2 ↔ v = car_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l735_73596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_proof_l735_73565

/-- Proves that the total number of students in a class is 20, given specific average scores for different groups and the overall class average. -/
theorem class_size_proof (n : ℕ) : n = 20 := by
  let group1_size : ℕ := 10
  let group1_avg : ℚ := 80
  let group2_avg : ℚ := 60
  let class_avg : ℚ := 70
  have h1 : group1_size * group1_avg + (n - group1_size) * group2_avg = n * class_avg := by
    sorry
  sorry

/-- The total number of students in the class -/
def total_students : ℕ := 20

#check class_size_proof total_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_proof_l735_73565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l735_73562

/-- An arithmetic sequence with first term 3 and last term 33 -/
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ (n : ℕ) (d : ℝ), n ≥ 2 ∧
  (∀ i : ℕ, i ≤ n → (3 + (i - 1) * d) ∈ ({3, 8, 13, a, b, 33} : Set ℝ)) ∧
  (∀ x ∈ ({3, 8, 13, a, b, 33} : Set ℝ), ∃ i : ℕ, i ≤ n ∧ x = 3 + (i - 1) * d)

theorem arithmetic_sequence_sum (a b : ℝ) :
  arithmetic_sequence a b → a + b = 51 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l735_73562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l735_73522

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l735_73522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l735_73545

-- Define the points
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 1)
def C : ℝ × ℝ := (1, 5)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to calculate the area of a triangle using Heron's formula
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem area_of_triangle_ABC :
  triangleArea (distance A B) (distance B C) (distance C A) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l735_73545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l735_73563

noncomputable section

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The foci of an ellipse -/
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((-c, 0), (c, 0))

/-- A point is on the ellipse -/
def onEllipse (p : ℝ × ℝ) (a b : ℝ) : Prop :=
  (p.1^2 / a^2) + (p.2^2 / b^2) = 1

/-- Three numbers form an arithmetic sequence -/
def isArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem ellipse_theorem (a b : ℝ) (h_ab : a > b ∧ b > 0) 
  (A B : ℝ × ℝ) (h_A : onEllipse A a b) (h_B : onEllipse B a b)
  (h_chord : B.1 = (foci a b).2.1)
  (h_perimeter : distance A (foci a b).1 + distance A (foci a b).2 + distance A B = 16)
  (h_arithmetic : isArithmeticSequence (distance A (foci a b).1) (distance (foci a b).1 (foci a b).2) (distance A (foci a b).2)) :
  a = 4 ∧ b = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l735_73563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_flip_probability_l735_73595

noncomputable def fair_coin : ℚ := 1 / 2

def prob_tail (p : ℚ) : ℚ := p

def prob_head (p : ℚ) : ℚ := 1 - p

theorem fifth_flip_probability (p : ℚ) (h1 : p = fair_coin) 
  (h2 : prob_tail p = p) (h3 : prob_head p = 1 - p) :
  prob_head p = 1 / 2 := by
  rw [h1, fair_coin]
  simp [prob_head]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_flip_probability_l735_73595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_y_axis_through_point_one_zero_l735_73566

/-- A line parallel to the y-axis passing through (1,0) has the equation x = 1 -/
theorem line_parallel_y_axis_through_point_one_zero :
  ∀ (x y : ℝ), (x = 1 ∧ y ∈ Set.univ) ↔ (∃ (k : ℝ), y = k ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_y_axis_through_point_one_zero_l735_73566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_course_combinations_correct_l735_73569

def elective_course_combinations (type_a : ℕ) (type_b : ℕ) : ℕ :=
  if type_a = 3 && type_b = 4 then
    30
  else
    0  -- Default case for other inputs

theorem elective_course_combinations_correct :
  elective_course_combinations 3 4 = 30 :=
by
  rfl  -- Reflexivity proves this, as it's true by definition

#eval elective_course_combinations 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_course_combinations_correct_l735_73569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_fraction_l735_73541

theorem min_decimal_digits_fraction (n d : ℚ) (h : d = (2^30 * 5^6 : ℚ)) :
  let f := n / d
  (∃ (k : ℕ), k ≥ 30 ∧ (f * 10^k).isInt) ∧
  (∀ (m : ℕ), m < 30 → ¬(f * 10^m).isInt) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_fraction_l735_73541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l735_73528

/-- The function f(x) = log₂(x) + x - 4 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 4

theorem zero_point_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  have h1 : Continuous f := by sorry
  have h2 : StrictMono f := by sorry
  have h3 : f 2 < 0 := by sorry
  have h4 : f 3 > 0 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l735_73528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l735_73524

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List (Char × ℕ)) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map (fun (_, count) => Nat.factorial count)).prod

/-- The word "BALLOON" has 7 letters with 'L' and 'O' each repeating twice -/
def balloonWord : (ℕ × List (Char × ℕ)) :=
  (7, [('L', 2), ('O', 2)])

theorem balloon_arrangements :
  distinctArrangements balloonWord.1 balloonWord.2 = 1260 := by
  sorry

#eval distinctArrangements balloonWord.1 balloonWord.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l735_73524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_repeat_divisible_by_101_l735_73557

/-- Represents a two-digit number -/
def TwoDigitNumber : Type := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Constructs a six-digit number by repeating a two-digit number three times -/
def repeatTwoDigits (n : TwoDigitNumber) : ℕ :=
  100000 * n.val + 1000 * n.val + 10 * n.val

theorem six_digit_repeat_divisible_by_101 (n : TwoDigitNumber) :
  (repeatTwoDigits n) % 101 = 0 := by
  sorry

#eval repeatTwoDigits ⟨42, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_repeat_divisible_by_101_l735_73557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_minus_n_equals_interval_l735_73516

-- Define the set M
def M : Set ℝ := {x | |x + 1| ≤ 2}

-- Define the set N
def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α|}

-- Define set difference
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem m_minus_n_equals_interval :
  setDifference M N = Set.Ioc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_minus_n_equals_interval_l735_73516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_before_jack_l735_73547

/-- The time difference in minutes between Jill's and Jack's arrivals at the lake --/
noncomputable def time_difference (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (jill_stop_time : ℝ) : ℝ :=
  (distance / jack_speed * 60) - (distance / jill_speed * 60 + jill_stop_time)

/-- Theorem stating that Jill arrives 43 minutes before Jack --/
theorem jill_arrives_before_jack : 
  time_difference 3 12 3 2 = 43 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_before_jack_l735_73547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_root_implies_m_range_l735_73520

-- Define the function f(x) = 2mx + 4
def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x + 4

-- Define the interval [-2, 1]
def interval : Set ℝ := Set.Icc (-2) 1

-- Theorem statement
theorem one_root_implies_m_range (m : ℝ) :
  (∃! x, x ∈ interval ∧ f m x = 0) →
  m ∈ Set.Iic (-2) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_root_implies_m_range_l735_73520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l735_73509

-- Define the statements as axioms instead of strings
axiom statement_1 : Prop
axiom statement_2 : Prop
axiom statement_3 : Prop
axiom statement_4 : Prop
axiom statement_5 : Prop

-- Define the correctness of each statement
def is_correct (s : Prop) : Prop := s

-- Theorem stating which statements are correct
theorem correct_statements :
  is_correct statement_1 ∧
  ¬is_correct statement_2 ∧
  is_correct statement_3 ∧
  is_correct statement_4 ∧
  is_correct statement_5 := by
  sorry

-- Optionally, you can add descriptions for each statement
/- 
statement_1 : The smaller the standard deviation, the smaller the fluctuation in sample data
statement_2 : Regression analysis studies the independence between two related events
statement_3 : In regression analysis, the predicted variable is determined by both the explanatory variable and random error
statement_4 : The coefficient of determination, R², is used to characterize the regression effect; the larger the R², the better the fit of the regression model
statement_5 : For the observed value k of the random variable K² for categorical variables X and Y, the smaller the k, the less confident one can be about the relationship between X and Y
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l735_73509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_point_with_greater_distance_to_points_than_lines_l735_73519

-- Define a structure for a point in 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a line in 2D plane
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a function to calculate distance from a point to a line
noncomputable def distanceToLine (p : Point2D) (l : Line2D) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

-- The main theorem
theorem existence_of_point_with_greater_distance_to_points_than_lines 
  (points : Set Point2D) (lines : Set Line2D) 
  (h1 : points.Finite) (h2 : lines.Finite) :
  ∃ A : Point2D, A ∉ points ∧ 
  ∀ (p : Point2D) (l : Line2D), p ∈ points → l ∈ lines → 
  distance A p > distanceToLine A l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_point_with_greater_distance_to_points_than_lines_l735_73519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_symmetry_l735_73592

-- Define a polynomial with real coefficients
def RealPolynomial := Polynomial ℝ

-- Define the property of having infinitely many integer solutions
def HasInfinitelyManyIntegerSolutions (P : RealPolynomial) : Prop :=
  ∀ n : ℕ, ∃ m k : ℤ, m ≠ k ∧ P.eval (↑m) + P.eval (↑k) = 0

-- Define the property of having a center of symmetry
def HasCenterOfSymmetry (P : RealPolynomial) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, P.eval (c - x) = -P.eval x

-- State the theorem
theorem polynomial_symmetry (P : RealPolynomial) 
  (h : HasInfinitelyManyIntegerSolutions P) : 
  HasCenterOfSymmetry P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_symmetry_l735_73592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l735_73586

/-- The area of a right triangle with legs of 36 and 45 units is 810 square units. -/
theorem right_triangle_area : 
  ∀ (t : Set (ℝ × ℝ)) (a b : ℝ),
  (∃ (c : ℝ), t = {(0, 0), (a, 0), (0, b)}) →  -- t is a right triangle in the plane
  a = 36 →                                    -- one leg is 36 units
  b = 45 →                                    -- other leg is 45 units
  (1/2 : ℝ) * a * b = 810 :=                  -- area is 810 square units
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l735_73586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_area_theorem_l735_73588

noncomputable def tourist_function (n : ℕ) (A w θ k : ℝ) : ℝ :=
  A * Real.cos (w * (n : ℝ) + θ) + k

theorem tourist_area_theorem 
  (f : ℕ → ℝ) 
  (h_f : ∀ n, f n = tourist_function n 200 (Real.pi / 6) ((2 * Real.pi) / 3) 300) 
  (h_periodic : ∀ n, f (n + 12) = f n)
  (h_range : ∀ n, 1 ≤ n ∧ n ≤ 12 → 0 < f n)
  (h_diff : f 8 - f 2 = 400)
  (h_feb : f 2 = 100)
  (h_increase : ∀ n, 2 ≤ n ∧ n < 8 → f n < f (n + 1)) :
  (∀ n, f n = 200 * Real.cos (Real.pi * (n : ℝ) / 6 + 2 * Real.pi / 3) + 300) ∧ 
  ({n : ℕ | 1 ≤ n ∧ n ≤ 12 ∧ f n ≥ 400} = {6, 7, 8, 9, 10}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_area_theorem_l735_73588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_is_worst_player_l735_73535

structure Player where
  name : String
  generation : Nat
  sex : Bool

structure ChessGame where
  players : List Player
  worst_player : Player
  best_player : Player

def has_twin (p : Player) (players : List Player) : Prop :=
  ∃ q, q ∈ players ∧ q.generation = p.generation ∧ q.sex = p.sex ∧ q ≠ p

theorem daughter_is_worst_player (game : ChessGame) 
  (grandfather mother son daughter : Player)
  (h1 : grandfather ∈ game.players)
  (h2 : mother ∈ game.players)
  (h3 : son ∈ game.players)
  (h4 : daughter ∈ game.players)
  (h5 : grandfather.generation < mother.generation)
  (h6 : mother.generation < son.generation)
  (h7 : mother.generation < daughter.generation)
  (h8 : son.generation = daughter.generation)
  (h9 : son.sex ≠ daughter.sex)
  (h10 : has_twin game.worst_player game.players)
  (h11 : ∃ twin, twin ∈ game.players ∧ has_twin twin game.players ∧ twin.sex = game.best_player.sex)
  (h12 : game.worst_player.generation = game.best_player.generation) :
  game.worst_player = daughter :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_is_worst_player_l735_73535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ants_count_l735_73589

/-- The number of ants in the jar after n hours -/
def ants (n : ℕ) : ℕ := sorry

/-- The number of ants doubles each hour -/
axiom double_each_hour (n : ℕ) : ants (n + 1) = 2 * ants n

/-- After 5 hours, there are 1600 ants -/
axiom ants_after_5_hours : ants 5 = 1600

/-- The theorem to prove: the initial number of ants is 50 -/
theorem initial_ants_count : ants 0 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ants_count_l735_73589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l735_73593

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem cos_2theta_value (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 6) (h3 : f θ = 4 / 3) :
  Real.cos (2 * θ) = (Real.sqrt 15 + 2) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l735_73593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l735_73525

def A : Set ℤ := {x : ℤ | x^2 - 1 ≤ 0}

def B : Set ℤ := {x : ℤ | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l735_73525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l735_73558

noncomputable def expression (x y z : ℝ) : ℝ :=
  x / abs x + y / abs y + z / abs z + abs (x * y * z) / (x * y * z)

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ∃ M : Set ℝ, M = {4, -4, 0} ∧ expression x y z ∈ M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l735_73558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribedSphereSurfaceArea_is_100pi_l735_73560

/-- Represents a tetrahedron S-ABC with specific properties -/
structure Tetrahedron where
  /-- Length of the hypotenuse AB of the right triangle ABC -/
  ab : ℝ
  /-- Length of the edge SC perpendicular to plane ABC -/
  sc : ℝ
  /-- Assumption that ABC is a right triangle -/
  abc_right_triangle : True
  /-- Assumption that SC is perpendicular to plane ABC -/
  sc_perpendicular : True

/-- Calculate the surface area of the circumscribed sphere of the tetrahedron -/
noncomputable def circumscribedSphereSurfaceArea (t : Tetrahedron) : ℝ :=
  4 * Real.pi * (t.ab ^ 2 / 4 + t.sc ^ 2 / 4)

/-- Theorem stating that the surface area of the circumscribed sphere is 100π -/
theorem circumscribedSphereSurfaceArea_is_100pi (t : Tetrahedron)
    (h1 : t.ab = 8)
    (h2 : t.sc = 6) :
    circumscribedSphereSurfaceArea t = 100 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribedSphereSurfaceArea_is_100pi_l735_73560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l735_73559

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 3 - x^2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- State the theorem
theorem ellipse_line_intersection 
  (h : ∀ x y, hyperbola x y → (x = 0 ∧ y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) 
  (e : ∀ x y, ellipse_C x y) :
  ∀ m : ℝ, (∃ x y, ellipse_C x y ∧ line x y m) ↔ -Real.sqrt 5 ≤ m ∧ m ≤ Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l735_73559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_both_perfect_squares_l735_73533

theorem no_positive_integer_both_perfect_squares :
  ∀ n : ℕ, n > 0 → ¬(∃ a b : ℕ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_both_perfect_squares_l735_73533
