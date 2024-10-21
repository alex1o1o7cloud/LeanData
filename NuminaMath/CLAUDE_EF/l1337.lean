import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_construction_l1337_133731

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given elements
variable (ABC : Triangle) (circumCircle : Circle) (I : Point)

-- Define the properties of the given elements
axiom ABC_scalene : ABC.A ≠ ABC.B ∧ ABC.B ≠ ABC.C ∧ ABC.C ≠ ABC.A

-- Define helper functions (these would need to be implemented properly)
def is_incenter (p : Point) (t : Triangle) : Prop := sorry
def on_circle (p : Point) (c : Circle) : Prop := sorry
def is_diameter (c : Circle) (l : Line) : Prop := sorry
def intersection_of_external_angle_bisector_and_circle (p : Point) (c : Circle) : Point := sorry
def midpoint_of_arc (p1 p2 : Point) (c : Circle) : Point := sorry

-- Define the constructed points
def K : Point := sorry
def W : Point := sorry

-- State the theorem
theorem diameter_construction :
  ∃ (K W : Point),
    on_circle K circumCircle ∧
    on_circle W circumCircle ∧
    is_diameter circumCircle (Line.mk 0 0 0) ∧
    K = intersection_of_external_angle_bisector_and_circle ABC.B circumCircle ∧
    W = midpoint_of_arc ABC.A ABC.C circumCircle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_construction_l1337_133731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l1337_133791

theorem book_distribution (total_books : ℕ) (senior_percentage : ℚ) 
  (junior_ratio middle_ratio : ℕ) : 
  total_books = 180 → 
  senior_percentage = 2/5 → 
  junior_ratio = 4 → 
  middle_ratio = 5 → 
  let remaining_books := total_books - (senior_percentage * total_books).floor
  let junior_books := (remaining_books * junior_ratio / (junior_ratio + middle_ratio))
  let middle_books := (remaining_books * middle_ratio / (junior_ratio + middle_ratio))
  junior_books = 48 ∧ middle_books = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l1337_133791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agricultural_experiment_l1337_133793

noncomputable def seeds_in_second_plot (seeds_first_plot : ℕ) (germination_rate_first : ℚ) 
  (germination_rate_second : ℚ) (total_germination_rate : ℚ) : ℚ :=
  (total_germination_rate * (seeds_first_plot : ℚ) - germination_rate_first * seeds_first_plot) / 
  (germination_rate_second - total_germination_rate)

theorem agricultural_experiment : 
  ⌊seeds_in_second_plot 300 (25/100) (30/100) (27/100)⌋ = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_agricultural_experiment_l1337_133793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_in_zero_one_l1337_133770

noncomputable def f (x : ℝ) := (x^2 - 2*x + 2) * Real.exp x - (1/3) * x^3 - (1/2) * x^2

theorem extreme_point_in_zero_one (x₀ : ℝ) :
  (∀ x, f x ≤ f x₀) → 
  (∃ n : ℤ, x₀ > n ∧ x₀ < n + 1) →
  ∃ n : ℤ, n = 0 ∧ x₀ > n ∧ x₀ < n + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_in_zero_one_l1337_133770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l1337_133764

/-- The line equation -/
def line_eq (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (1 - m)*y - m - 2 = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

/-- The theorem stating the properties of the line and circle -/
theorem line_circle_properties :
  (∀ m : ℝ, line_eq m 1 1) ∧
  (∀ m : ℝ, ∃ x y : ℝ, line_eq m x y ∧ circle_eq x y) ∧
  (∃ m : ℝ, ∃ x1 y1 x2 y2 : ℝ,
    line_eq m x1 y1 ∧ circle_eq x1 y1 ∧
    line_eq m x2 y2 ∧ circle_eq x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l1337_133764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l1337_133729

/-- Represents the gain percentage for a cloth sale -/
noncomputable def gain_percentage (cost_price selling_price : ℝ) : ℝ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem: If the selling price of 30 meters equals the cost price of 40 meters,
    then the gain percentage is 4/9 * 100% -/
theorem cloth_sale_gain_percentage 
  (cost_per_meter selling_per_meter : ℝ) 
  (h : 30 * selling_per_meter = 40 * cost_per_meter) :
  gain_percentage (30 * cost_per_meter) (30 * selling_per_meter) = 4/9 * 100 := by
  sorry

#eval (4/9 : ℚ) * 100  -- To show the approximate result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l1337_133729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conflicting_pairs_identifiable_l1337_133761

/-- Represents a reactant -/
structure Reactant where
  id : ℕ

/-- Represents a test tube -/
structure TestTube where
  id : ℕ
  contents : List Reactant

/-- Represents the state of the experiment -/
structure ExperimentState where
  reactants : List Reactant
  testTubes : List TestTube
  conflictingPairs : List (Reactant × Reactant)

/-- Represents an action of adding a reactant to a test tube -/
inductive ExperimentAction where
  | Add : Reactant → TestTube → ExperimentAction

/-- Represents the result of an action -/
inductive ActionResult where
  | NoReaction : ActionResult
  | Burst : Reactant → Reactant → ActionResult

/-- A strategy is a function that decides the next action based on the current state -/
def Strategy := ExperimentState → Option ExperimentAction

/-- Executes a strategy until all conflicting pairs are found or no more actions are possible -/
def executeStrategy (s : Strategy) (initialState : ExperimentState) : List (Reactant × Reactant) :=
  sorry

/-- The main theorem: there exists a strategy that can identify all conflicting pairs -/
theorem conflicting_pairs_identifiable 
  (n : ℕ) 
  (initialReactants : List Reactant) 
  (initialTestTubes : List TestTube) 
  (h1 : initialReactants.length = 2 * n) 
  (h2 : initialTestTubes.length = n) 
  (h3 : ∃ (pairs : List (Reactant × Reactant)), 
    pairs.length = n ∧ 
    (∀ r1 r2, (r1, r2) ∈ pairs → r1 ∈ initialReactants ∧ r2 ∈ initialReactants ∧ r1 ≠ r2)) :
  ∃ (s : Strategy), 
    let finalPairs := executeStrategy s { 
      reactants := initialReactants, 
      testTubes := initialTestTubes, 
      conflictingPairs := [] 
    }
    finalPairs.length = n ∧
    (∀ p, p ∈ finalPairs → p.1 ∈ initialReactants ∧ p.2 ∈ initialReactants ∧ p.1 ≠ p.2) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conflicting_pairs_identifiable_l1337_133761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationships_l1337_133745

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- Given conditions
variable (m : Line) (n : Line) (α : Plane) (β : Plane)
variable (h1 : perpendicular_line_plane m α)
variable (h2 : in_plane n β)

-- Theorem to prove
theorem line_plane_relationships :
  (parallel_plane α β → perpendicular_line_line m n) ∧
  (parallel_line m n → perpendicular_plane α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationships_l1337_133745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1337_133707

/-- Given two circles in a 2D plane, one with radius 3 and center at (3,0),
    and another with radius 2 and center at (7,0), this theorem states that
    the y-intercept of a line tangent to both circles at points in the first
    quadrant is equal to 35√776 / 776. -/
theorem tangent_line_y_intercept :
  let circle1 : { center : ℝ × ℝ // True } := ⟨(3, 0), trivial⟩
  let circle2 : { center : ℝ × ℝ // True } := ⟨(7, 0), trivial⟩
  ∃ (line : Set (ℝ × ℝ)) (p1 p2 : ℝ × ℝ),
    p1.1 > 3 ∧ p1.2 > 0 ∧ p2.1 > 7 ∧ p2.2 > 0 ∧
    (p1 ∈ line) ∧ (p2 ∈ line) ∧
    (∀ (x y : ℝ), (x, y) ∈ line ↔ ∃ (t : ℝ), (x, y) = (1 - t) • p1 + t • p2) ∧
    ((p1.1 - circle1.val.1)^2 + (p1.2 - circle1.val.2)^2 = 3^2) ∧
    ((p2.1 - circle2.val.1)^2 + (p2.2 - circle2.val.2)^2 = 2^2) ∧
    (∀ (q : ℝ × ℝ), q ∈ line → (q.1 - circle1.val.1)^2 + (q.2 - circle1.val.2)^2 ≥ 3^2) ∧
    (∀ (q : ℝ × ℝ), q ∈ line → (q.1 - circle2.val.1)^2 + (q.2 - circle2.val.2)^2 ≥ 2^2) →
    ∃ (y : ℝ), y = (35 * Real.sqrt 776) / 776 ∧ (0, y) ∈ line :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1337_133707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_ellipse_equation_second_ellipse_equation_l1337_133733

-- Define the ellipse type
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  center : ℝ × ℝ

-- Define the standard form of an ellipse equation
def standardForm (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.center.1)^2 / e.a^2 + (y - e.center.2)^2 / e.b^2 = 1

-- First ellipse theorem
theorem first_ellipse_equation :
  ∀ e : Ellipse,
  e.a = 6 →
  e.c = 3 * Real.sqrt 3 →
  e.center = (0, 0) →
  ∀ x y : ℝ,
  standardForm e x y ↔ x^2 / 36 + y^2 / 9 = 1 := by sorry

-- Second ellipse theorem
theorem second_ellipse_equation :
  ∀ e : Ellipse,
  e.center = (0, 0) →
  (∀ x y : ℝ, x = 0 ∧ y = 2 → (x, y) ∈ Set.range (λ t : ℝ ↦ (e.c * Real.cos t, e.a * Real.sin t))) →
  (∀ x y : ℝ, x = 0 ∧ y = -2 → (x, y) ∈ Set.range (λ t : ℝ ↦ (e.c * Real.cos t, e.a * Real.sin t))) →
  (3, 2) ∈ Set.range (λ t : ℝ ↦ (e.a * Real.cos t, e.b * Real.sin t)) →
  ∀ x y : ℝ,
  standardForm e x y ↔ y^2 / 16 + x^2 / 12 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_ellipse_equation_second_ellipse_equation_l1337_133733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_c_l1337_133758

-- Define the arithmetic sequence
def arithmetic_sequence (a₀ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => arithmetic_sequence a₀ d n + d

-- Define the constant c
noncomputable def c (d : ℝ) : ℝ := (1 + Real.sqrt (1 + 4 / d)) / 2

-- Define the recursive sequence S_n and b_n
def S (a₀ d : ℝ) : ℕ → List ℝ
  | 0 => [a₀]
  | n + 1 =>
    let prev := S a₀ d n
    let b := prev.head!
    (prev.tail.map (· + 1)).cons b

def b (a₀ d : ℝ) : ℕ → ℝ
  | n => (S a₀ d n).head!

-- State the theorem
theorem existence_of_c (a₀ d : ℝ) (h1 : 1 ≤ a₀) (h2 : a₀ ≤ d) :
  ∃ c : ℝ, ∀ n : ℕ, b a₀ d n = ⌊c * arithmetic_sequence a₀ d n⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_c_l1337_133758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1337_133783

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (2,1) and b = (-4,λ), then λ = -2 -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (l : ℝ) : 
  a = (2, 1) → b = (-4, l) → (∃ (k : ℝ), a = k • b) → l = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1337_133783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_cosine_sine_inequality_l1337_133701

theorem acute_angles_cosine_sine_inequality (α β : ℝ) : 
  0 < α ∧ α < π / 2 →
  0 < β ∧ β < π / 2 →
  Real.cos (α + β) < 0 →
  Real.sin α > Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_cosine_sine_inequality_l1337_133701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_r_plus_s_zero_l1337_133716

/-- The curve equation -/
noncomputable def curve (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x - s)

/-- Theorem: If y = -x is an axis of symmetry for the curve y = (px + q) / (rx - s),
    where p, q, r, s are all nonzero, then r + s = 0 -/
theorem symmetry_implies_r_plus_s_zero
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = curve p q r s x → -x = curve p q r s (-y)) :
  r + s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_r_plus_s_zero_l1337_133716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delegate_seating_probability_l1337_133768

/-- Represents the number of delegates --/
def total_delegates : ℕ := 12

/-- Represents the number of countries --/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- Calculates the total number of possible seating arrangements --/
def total_arrangements : ℕ := (total_delegates - 1).factorial / (delegates_per_country.factorial ^ num_countries)

/-- Calculates the number of unwanted arrangements where at least one country's delegates sit together --/
def unwanted_arrangements : ℕ := 
  num_countries * total_delegates * ((total_delegates - delegates_per_country).factorial / (delegates_per_country.factorial ^ (num_countries - 1)))
  - (num_countries.choose 2) * total_delegates
  + total_delegates

/-- The probability that each delegate sits next to at least one delegate from another country --/
noncomputable def probability : ℚ := (total_arrangements - unwanted_arrangements : ℚ) / total_arrangements

theorem delegate_seating_probability : probability = 18507 / 19250 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delegate_seating_probability_l1337_133768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_area_unit_circle_area_deductive_reasoning_l1337_133771

-- Define the area of a circle as a function of its radius
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define a unit circle
def unit_circle_radius : ℝ := 1

-- Theorem: The area of a unit circle is π
theorem unit_circle_area : circle_area unit_circle_radius = Real.pi := by
  -- Expand the definition of circle_area
  unfold circle_area
  -- Simplify the expression
  simp [unit_circle_radius]

-- Definition of deductive reasoning
def is_deductive_reasoning (premise : Prop) (conclusion : Prop) : Prop :=
  premise → conclusion

-- Theorem: The reasoning used to determine the area of a unit circle
-- from the general formula for the area of a circle is deductive reasoning
theorem unit_circle_area_deductive_reasoning :
  is_deductive_reasoning
    (∀ r : ℝ, circle_area r = Real.pi * r^2)
    (circle_area unit_circle_radius = Real.pi) := by
  -- Introduce the premise
  intro h
  -- Apply the premise to the unit circle radius
  have h1 := h unit_circle_radius
  -- Rewrite using the definition of unit_circle_radius
  rw [unit_circle_radius] at h1
  -- Simplify the expression
  simp at h1
  -- The conclusion follows directly
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_area_unit_circle_area_deductive_reasoning_l1337_133771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l1337_133717

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume between two concentric spheres -/
noncomputable def volume_between_spheres (r₁ r₂ : ℝ) : ℝ :=
  sphere_volume r₂ - sphere_volume r₁

theorem volume_between_specific_spheres :
  volume_between_spheres 5 10 = (3500 / 3) * Real.pi := by
  -- Expand the definitions
  unfold volume_between_spheres
  unfold sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l1337_133717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_l1337_133705

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

noncomputable def scale (x y : ℝ) : ℝ × ℝ := (x / 2, 3 * y)

theorem scaling_transformation (x : ℝ) : 
  let (x', y') := scale x (f x)
  g x' = y' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_l1337_133705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1337_133773

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perpendicularity of two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- A line is an asymptote of a hyperbola -/
def is_asymptote (l : Line) (h : Hyperbola) : Prop :=
  l.b / l.a = h.b / h.a ∨ l.b / l.a = - h.b / h.a

/-- The theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) 
  (h_perp : ∃ (asymptote : Line), perpendicular asymptote l ∧ is_asymptote asymptote h) 
  (h_line : l.a = 1 ∧ l.b = -3 ∧ l.c = 1) : 
  eccentricity h = Real.sqrt 10 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1337_133773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_average_speed_l1337_133759

/-- Represents a lap in Bob's run -/
structure Lap where
  length : ℝ
  elevation_change : ℝ
  time : ℝ

/-- Calculates the average speed given a list of laps -/
noncomputable def average_speed (laps : List Lap) : ℝ :=
  let total_distance := laps.foldl (fun acc lap => acc + lap.length) 0
  let total_time := laps.foldl (fun acc lap => acc + lap.time) 0
  total_distance / total_time

theorem bobs_average_speed :
  let lap1 : Lap := { length := 380, elevation_change := 50, time := 70 }
  let lap2 : Lap := { length := 420, elevation_change := -30, time := 85 }
  let lap3 : Lap := { length := 400, elevation_change := 0, time := 80 }
  let bobs_run := [lap1, lap2, lap3]
  abs (average_speed bobs_run - 5.106) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_average_speed_l1337_133759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equivalence_l1337_133798

theorem complex_number_equivalence : 
  (2 - Complex.I) / Complex.I = -1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equivalence_l1337_133798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_four_l1337_133726

/-- Represents the distribution of name lengths --/
def name_lengths : List (Nat × Nat) := [
  (3, 8),  -- 8 names of length 3
  (4, 5),  -- 5 names of length 4
  (5, 3),  -- 3 names of length 5
  (6, 6),  -- 6 names of length 6
  (7, 3)   -- 3 names of length 7
]

/-- The total number of names --/
def total_names : Nat := name_lengths.foldr (fun p acc => p.2 + acc) 0

/-- The position of the median in a sorted list --/
def median_position : Nat := (total_names + 1) / 2

/-- Theorem: The median length of the names is 4 --/
theorem median_length_is_four :
  total_names = 25 ∧ 
  median_position = 13 ∧
  (name_lengths.take 2).foldr (fun p acc => p.2 + acc) 0 ≥ 13 ∧
  (name_lengths.take 1).foldr (fun p acc => p.2 + acc) 0 < 13 →
  (name_lengths.find? (fun p => p.1 = 4)).map Prod.fst = some 4 := by
  sorry

#eval total_names        -- Should output 25
#eval median_position    -- Should output 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_four_l1337_133726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l1337_133730

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x / (1 + 2^x) - 1/2

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem range_of_floor_f :
  ∀ y : ℤ, (∃ x : ℝ, floor (f x) = y) ↔ y = -1 ∨ y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l1337_133730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l1337_133762

theorem number_of_observations :
  ∃ (n : ℕ), n = 27 ∧ 
  (n : ℝ) * 99.075 = n * 100 - (75 - 50) := by
  sorry

#check number_of_observations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l1337_133762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_minus_e_l1337_133742

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define e as the base of natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the conditions
axiom symmetry : ∀ x y : ℝ, f (x - 1) = y ↔ f (3 - x) = -y
axiom periodicity : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom definition_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.exp x - 1

-- State the theorem to be proved
theorem f_sum_equals_one_minus_e : f 2016 + f (-2017) = 1 - e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_minus_e_l1337_133742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_two_l1337_133724

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its bottom-left and top-right corners -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of overlap between two rectangles -/
noncomputable def overlapArea (r1 r2 : Rectangle) : ℝ :=
  let xOverlap := min r1.topRight.x r2.topRight.x - max r1.bottomLeft.x r2.bottomLeft.x
  let yOverlap := min r1.topRight.y r2.topRight.y - max r1.bottomLeft.y r2.bottomLeft.y
  max 0 xOverlap * max 0 yOverlap

/-- The main theorem stating that the overlap area is 2 square units -/
theorem overlap_area_is_two :
  let r1 : Rectangle := { bottomLeft := { x := 0, y := 0 }, topRight := { x := 3, y := 2 } }
  let r2 : Rectangle := { bottomLeft := { x := 1, y := 1 }, topRight := { x := 4, y := 3 } }
  overlapArea r1 r2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_two_l1337_133724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_is_single_function_l1337_133751

/-- A piecewise function defined on a real domain -/
def PiecewiseFunction (α : Type*) := ℝ → α

/-- A predicate that checks if a function is piecewise -/
def IsPiecewise (f : PiecewiseFunction α) : Prop :=
  ∃ (a : ℝ) (f₁ f₂ : ℝ → α), ∀ x, f x = if x < a then f₁ x else f₂ x

theorem piecewise_is_single_function 
  (f : PiecewiseFunction α) (h : IsPiecewise f) : 
  ∀ x y, f x = f y → x = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_is_single_function_l1337_133751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1337_133719

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + 1

-- Define the proposed inverse function
def g (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Theorem statement
theorem inverse_function_proof (x : ℝ) (h : x ≥ 1) :
  (∀ y, y ≥ 1 → f (g y) = y) ∧ (g (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1337_133719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1337_133703

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 2 else 3*x - 7

-- Define the equation g(g(x)) = 3
def equation (x : ℝ) : Prop :=
  g (g x) = 3

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, equation x) ∧ (s.card = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1337_133703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l1337_133780

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_difference (a : ℝ) :
  a > 0 →
  a ≠ 1 →
  (∀ x ∈ Set.Icc 1 2, f a x ≤ f a 2) →
  (∀ x ∈ Set.Icc 1 2, f a x ≥ f a 1) →
  f a 2 - f a 1 = 6 →
  a = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l1337_133780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_arithmetic_progression_l1337_133723

def X : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def contains_arithmetic_progression (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a < b ∧ b < c ∧ is_arithmetic_progression a b c

theorem partition_contains_arithmetic_progression :
  ∀ (P Q : Set ℕ), P ∪ Q = X → P ∩ Q = ∅ →
  contains_arithmetic_progression P ∨ contains_arithmetic_progression Q := by
  sorry

#check partition_contains_arithmetic_progression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_arithmetic_progression_l1337_133723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1337_133797

-- Define the necessary structures and properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def is_foot_of_altitude (T : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry

def is_on_angle_bisector (T : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry

def is_centroid (T : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry

def are_not_collinear (P Q R : ℝ × ℝ) : Prop :=
  sorry

-- State the theorem
theorem triangle_construction_theorem 
  (A₁ A' S : ℝ × ℝ) 
  (h : are_not_collinear A₁ A' S) : 
  ∃ T : Triangle, 
    is_foot_of_altitude T A₁ ∧ 
    is_on_angle_bisector T A' ∧ 
    is_centroid T S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1337_133797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_max_value_l1337_133704

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + a * Real.cos (2 * x)

-- Theorem 1
theorem find_a (a : ℝ) : f a (π / 6) = 2 → a = 1 := by sorry

-- Theorem 2
theorem max_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (π / 12) (7 * π / 12), StrictMonoOn (fun x => - f a x) (Set.Icc (π / 12) (7 * π / 12))) → 
  (∃ x ∈ Set.Icc (π / 12) (7 * π / 12), f a x = 2 * Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc (π / 12) (7 * π / 12), f a x ≤ 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_max_value_l1337_133704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1337_133754

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  4 * x - 3 * y + 6 * z = 42

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (2, 1, 4)

/-- The proposed closest point -/
noncomputable def closest_point : ℝ × ℝ × ℝ := (174/61, 22/61, 322/61)

/-- Theorem stating that the closest_point is indeed the closest point on the plane to the given_point -/
theorem closest_point_on_plane :
  plane_equation closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  ∀ p : ℝ × ℝ × ℝ, plane_equation p.1 p.2.1 p.2.2 →
    Real.sqrt ((p.1 - given_point.1)^2 + (p.2.1 - given_point.2.1)^2 + (p.2.2 - given_point.2.2)^2) ≥
    Real.sqrt ((closest_point.1 - given_point.1)^2 + (closest_point.2.1 - given_point.2.1)^2 + (closest_point.2.2 - given_point.2.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1337_133754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l1337_133749

theorem product_digits : 
  let a : Nat := 4567823456789012
  let b : Nat := 567345234567891
  let Q : Nat := a * b
  (Nat.log Q 10 + 1 : Nat) = 37 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l1337_133749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_implies_a_le_e_plus_one_l1337_133767

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a*x

noncomputable def g (x : ℝ) : ℝ := Real.exp x + (3/2) * x^2

theorem f_le_g_implies_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≤ g x) → a ≤ Real.exp 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_implies_a_le_e_plus_one_l1337_133767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1337_133732

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sin α)

-- Define the line l in polar form
def line_l (θ ρ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi/4) = Real.sqrt 2

-- Point P
def point_P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem curve_and_line_properties :
  -- 1. Cartesian equation of C
  (∀ x y : ℝ, (∃ α : ℝ, curve_C α = (x, y)) ↔ x^2/9 + y^2 = 1) ∧
  -- 2. Angle of inclination of l
  (∀ x y : ℝ, (∃ θ ρ : ℝ, line_l θ ρ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) 
    → y = x + 2) ∧
  -- 3. Sum of distances |PA| + |PB|
  (∃ A B : ℝ × ℝ, 
    (∃ α : ℝ, curve_C α = A) ∧
    (∃ α : ℝ, curve_C α = B) ∧
    (∃ θ ρ : ℝ, line_l θ ρ ∧ A.1 = ρ * Real.cos θ ∧ A.2 = ρ * Real.sin θ) ∧
    (∃ θ ρ : ℝ, line_l θ ρ ∧ B.1 = ρ * Real.cos θ ∧ B.2 = ρ * Real.sin θ) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 18 * Real.sqrt 2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1337_133732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1337_133766

/-- A parabola passing through points A(-1,0), B(3,0), and C(0,3) -/
structure Parabola where
  f : ℝ → ℝ
  h1 : f (-1) = 0
  h2 : f 3 = 0
  h3 : f 0 = 3

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def area_triangle (a b c : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the properties of the specific parabola -/
theorem parabola_properties (p : Parabola) :
  (∀ x, p.f x = -x^2 + 2*x + 3) ∧
  let v := vertex p
  area_triangle (-1, 0) (3, 0) v = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1337_133766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_analysis_l1337_133706

structure TreeSample where
  x : ℝ  -- root cross-sectional area in m²
  y : ℝ  -- volume in m³

def sample : List TreeSample := [
  ⟨0.04, 0.25⟩, ⟨0.06, 0.40⟩, ⟨0.04, 0.22⟩, ⟨0.08, 0.54⟩, ⟨0.08, 0.51⟩,
  ⟨0.05, 0.34⟩, ⟨0.05, 0.36⟩, ⟨0.07, 0.46⟩, ⟨0.07, 0.42⟩, ⟨0.06, 0.40⟩
]

def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_forest_area : ℝ := 186

def sample_size : ℕ := 10

theorem tree_analysis :
  let avg_x := sum_x / sample_size
  let avg_y := sum_y / sample_size
  let correlation := (sum_xy - sample_size * avg_x * avg_y) /
    (Real.sqrt ((sum_x_squared - sample_size * avg_x^2) * (sum_y_squared - sample_size * avg_y^2)))
  let estimated_total_volume := (avg_y / avg_x) * total_forest_area
  (avg_x = 0.06 ∧ avg_y = 0.39) ∧
  (abs (correlation - 0.97) < 0.01) ∧
  (abs (estimated_total_volume - 1209) < 1) := by
  sorry

#check tree_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_analysis_l1337_133706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1337_133744

/-- The parabola y^2 = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus of the parabola -/
def LineThroughFocus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = Focus.1 + t ∧ y = Focus.2 + t

/-- Theorem: Length of AB given the conditions -/
theorem length_AB (x₁ y₁ x₂ y₂ : ℝ) :
  Parabola x₁ y₁ →
  Parabola x₂ y₂ →
  LineThroughFocus x₁ y₁ →
  LineThroughFocus x₂ y₂ →
  x₁ + x₂ = 6 →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1337_133744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_sum_three_coins_l1337_133787

-- Define a fair coin
def fair_coin : Finset Bool := {true, false}

-- Define a fair die
def fair_die : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define the probability of getting heads on a fair coin
def prob_heads : ℚ := 1/2

-- Define the probability of getting an odd number on a fair die
def prob_odd_die : ℚ := 1/2

-- Define the function to calculate the probability of getting k heads out of n tosses
def prob_k_heads (n k : Nat) : ℚ :=
  (Nat.choose n k : ℚ) * prob_heads^k * (1 - prob_heads)^(n - k)

-- Define the function to calculate the probability of getting an odd sum when rolling k dice
def prob_odd_sum (k : Nat) : ℚ :=
  if k = 0 then 0
  else if k = 1 then 1/2
  else if k = 2 then 1/2
  else 1/2

-- Theorem statement
theorem prob_odd_sum_three_coins :
  (prob_k_heads 3 0 * prob_odd_sum 0) +
  (prob_k_heads 3 1 * prob_odd_sum 1) +
  (prob_k_heads 3 2 * prob_odd_sum 2) +
  (prob_k_heads 3 3 * prob_odd_sum 3) = 7/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_sum_three_coins_l1337_133787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_correct_l1337_133752

/-- The smallest integer side length of a square room that allows a 9' × 12' 
    rectangular table to be rotated from one corner to another without tilting 
    or disassembling. -/
def min_room_side : ℕ := 15

/-- The length of the table -/
def table_length : ℝ := 12

/-- The width of the table -/
def table_width : ℝ := 9

/-- The diagonal length of the table -/
noncomputable def table_diagonal : ℝ := Real.sqrt (table_length ^ 2 + table_width ^ 2)

theorem min_room_side_correct :
  (table_diagonal ≤ min_room_side) ∧ 
  (∀ n : ℕ, n < min_room_side → table_diagonal > ↑n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_correct_l1337_133752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1337_133763

def a (n : ℕ) : ℚ := 2^(5 - n)

def b (n : ℕ) (k : ℚ) : ℚ := n + k

noncomputable def c (n : ℕ) (k : ℚ) : ℚ := 
  if a n ≤ b n k then b n k else a n

theorem range_of_k : 
  ∃ (l u : ℚ), l = -5 ∧ u = -3 ∧ 
  (∀ k, (∀ n : ℕ, n ≥ 1 → c 5 k ≤ c n k) ↔ l ≤ k ∧ k ≤ u) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1337_133763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1337_133741

/-- An equilateral triangle with side length 10 meters -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 10

/-- The perimeter of an equilateral triangle -/
def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- The area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length ^ 2

theorem equilateral_triangle_properties (t : EquilateralTriangle) :
  perimeter t = 30 ∧ area t = 25 * Real.sqrt 3 := by
  sorry

#check equilateral_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1337_133741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pria_car_efficiency_l1337_133786

/-- Represents the fuel efficiency of a car under different conditions -/
structure CarEfficiency where
  advertised_mpg : ℚ
  tank_capacity : ℚ
  regular_mpg : ℚ
  premium_mpg : ℚ
  diesel_mpg : ℚ

/-- Calculates the weighted average MPG and its difference from the advertised MPG -/
def calculate_efficiency (car : CarEfficiency) : ℚ × ℚ :=
  let weighted_avg := (car.regular_mpg + car.premium_mpg + car.diesel_mpg) / 3
  let difference := car.advertised_mpg - weighted_avg
  (weighted_avg, difference)

/-- Theorem stating the efficiency calculation for Pria's car -/
theorem pria_car_efficiency :
  let car := CarEfficiency.mk 35 12 30 40 32
  let (weighted_avg, difference) := calculate_efficiency car
  weighted_avg = 34 ∧ difference = 1 := by
  -- Unfold definitions
  unfold calculate_efficiency
  -- Simplify arithmetic
  simp [CarEfficiency.mk]
  -- Split the conjunction
  apply And.intro
  -- Prove weighted_avg = 34
  · norm_num
  -- Prove difference = 1
  · norm_num

#check pria_car_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pria_car_efficiency_l1337_133786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l1337_133785

open Real

theorem vector_inequality {n : Type*} [NormedAddCommGroup ℝ] [InnerProductSpace ℝ ℝ] (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  min (‖a + b‖^2) (‖a - b‖^2) ≤ ‖a‖^2 + ‖b‖^2 := by
  sorry

#check vector_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l1337_133785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_b_yield_l1337_133784

/-- Proves that the percentage yield of scheme B is 50% given the investment conditions --/
theorem scheme_b_yield (investment_a investment_b : ℝ) 
  (yield_a : ℝ) (yield_difference : ℝ) : ℝ :=
  let scheme_b_yield := 50
  have h1 : investment_a = 300 := by sorry
  have h2 : investment_b = 200 := by sorry
  have h3 : yield_a = 30 := by sorry
  have h4 : yield_difference = 90 := by sorry
  have h5 : (investment_a * (1 + yield_a / 100)) = 
    (investment_b * (1 + scheme_b_yield / 100) + yield_difference) := by sorry
  scheme_b_yield

#check scheme_b_yield

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_b_yield_l1337_133784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1337_133757

-- Define the function s(x)
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^2

-- State the theorem about the range of s(x)
theorem range_of_s :
  {y : ℝ | ∃ x : ℝ, x ≠ 2 ∧ s x = y} = Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1337_133757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_simplest_fraction_parts_l1337_133711

-- Define the repeating decimal
def repeating_decimal : ℚ := 27 / 99

-- Theorem statement
theorem sum_of_simplest_fraction_parts : 
  ∃ (n d : ℕ), (n : ℚ) / d = repeating_decimal ∧ 
  Nat.gcd n d = 1 ∧ n + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_simplest_fraction_parts_l1337_133711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_submerged_height_l1337_133740

/-- The height of the submerged part of a cone immersed in a liquid -/
noncomputable def submergedHeight (s s' m r : ℝ) : ℝ :=
  m * (1 - ((s' - s) / s') ^ (1/3))

/-- Theorem: The height of the submerged part of a cone with specific gravity s, 
    height m, and base radius r, when immersed base downward into a liquid 
    with specific gravity s', is equal to m * (1 - (((s' - s) / s')^(1/3))). -/
theorem cone_submerged_height 
  (s s' m r : ℝ) 
  (h_s_pos : 0 < s) 
  (h_s'_pos : 0 < s') 
  (h_m_pos : 0 < m) 
  (h_r_pos : 0 < r) 
  (h_s'_gt_s : s' > s) : 
  ∃ h, h = submergedHeight s s' m r ∧ 
          0 < h ∧ 
          h < m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_submerged_height_l1337_133740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_matching_calendar_l1337_133713

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def calendar_shift (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).foldl
    (fun shift year ↦ (shift + days_in_year (start_year + year)) % 7) 0

theorem next_matching_calendar (start_year : ℕ) (h : start_year = 1990) :
  ∃ (next_year : ℕ), next_year > 1991 ∧ 
    calendar_shift start_year next_year = 0 ∧
    (∀ (year : ℕ), 1991 < year ∧ year < next_year → 
      calendar_shift start_year year ≠ 0) ∧
    next_year = 2004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_matching_calendar_l1337_133713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1337_133721

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and one of its asymptotes
    having the equation y = (3/4)x, prove that its eccentricity is 5/4. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_asymptote : b / a = 3 / 4) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1337_133721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_neg_one_f_geq_two_iff_a_geq_one_l1337_133736

/-- The function f(x) = e^x + ae^(-x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

/-- f is an odd function iff a = -1 -/
theorem f_is_odd_iff_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) ↔ a = -1 := by sorry

/-- f(x) ≥ 2 for all real x iff a ≥ 1 -/
theorem f_geq_two_iff_a_geq_one (a : ℝ) :
  (∀ x, f a x ≥ 2) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_neg_one_f_geq_two_iff_a_geq_one_l1337_133736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1337_133794

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 / 3 = 1

-- Define the focus F on the positive y-axis
def F : ℝ × ℝ := (0, 1)

-- Define the line l passing through F with slope angle 3π/4
def l (x y : ℝ) : Prop := y = -x + 1

-- Define points M and N as intersections of C and l
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- Define point P to complete parallelogram OMPN
noncomputable def P : ℝ × ℝ := (4/5, 6/5)

-- Define point O (origin)
def O : ℝ × ℝ := (0, 0)

theorem ellipse_problem :
  (C P.1 P.2 → False) ∧  -- P is inside the ellipse
  (∀ ε > 0, abs ((Real.sqrt 6 * 4 / 5) - (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) * (Real.sqrt 2 / 2))) < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1337_133794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sheep_is_ten_l1337_133738

/-- The number of chickens in the pen -/
def C : ℕ := sorry

/-- The number of sheep in the pen -/
def S : ℕ := sorry

/-- The condition when half of the chickens are driven out -/
axiom half_chickens_out : C + 4 * S = 2 * C

/-- The condition when 4 sheep are driven out -/
axiom four_sheep_out : 2 * C + 4 * S - 16 = 16 * (S - 4)

/-- The theorem stating that the total number of sheep is 10 -/
theorem total_sheep_is_ten : S = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sheep_is_ten_l1337_133738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cheap_is_necessary_for_good_quality_l1337_133702

-- Define a universe variable
universe u

-- Define the properties
variable {α : Type u}
def good_quality (x : α) : Prop := sorry
def not_cheap (x : α) : Prop := sorry

-- Define the given condition
axiom condition : ∀ (x : α), good_quality x → not_cheap x

-- Theorem to prove
theorem not_cheap_is_necessary_for_good_quality :
  ∀ (x : α), good_quality x → not_cheap x :=
by
  intro x
  exact condition x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cheap_is_necessary_for_good_quality_l1337_133702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_surface_area_l1337_133756

/-- A regular tetrahedron is a tetrahedron where all edges have the same length. -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A sphere inscribed in a tetrahedron is tangent to all faces of the tetrahedron. -/
structure InscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  radius_pos : radius > 0

/-- The surface area of a sphere is 4πr², where r is the radius of the sphere. -/
noncomputable def surface_area (s : InscribedSphere t) : ℝ := 4 * Real.pi * s.radius^2

/-- The theorem stating that the surface area of a sphere inscribed in a regular tetrahedron
    with edge length a is equal to πa²/6. -/
theorem inscribed_sphere_surface_area (t : RegularTetrahedron) 
  (s : InscribedSphere t) : surface_area s = Real.pi * t.edge_length^2 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_surface_area_l1337_133756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1337_133710

/-- The circle with equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The line with equation 3x + 4y + 8 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 + 4*p.2 + 8 = 0}

/-- Distance function from a point to the line -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |3*p.1 + 4*p.2 + 8| / Real.sqrt 25

/-- Theorem: The minimum distance from any point on the circle to the line is 2 -/
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 ∧ ∀ (p : ℝ × ℝ), p ∈ Circle → distToLine p ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1337_133710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_two_l1337_133788

-- Define the vectors
noncomputable def F₁ : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F₂ : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Define the work function
def work (F₁ F₂ S : ℝ × ℝ) : ℝ :=
  let F := (F₁.1 + F₂.1, F₁.2 + F₂.2)
  F.1 * S.1 + F.2 * S.2

-- Theorem statement
theorem work_equals_two : work F₁ F₂ S = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_two_l1337_133788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_sum_l1337_133715

noncomputable section

-- Define the function f(x) = a^x
def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the symmetric function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem symmetric_function_sum (a : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ f a 2 = 9 → g a (1/9) + f a 3 = 25 := by
  intro h
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_sum_l1337_133715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l1337_133750

noncomputable def series_term (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    (2 * n + 2) / (2^(2 * n + 2))
  else
    (2 * n + 4) / (3^(2 * n + 1))

noncomputable def series_sum : ℚ := ∑' n, series_term n

theorem sum_of_coprime_integers (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Nat.Coprime a b) 
  (h4 : (a : ℚ) / (b : ℚ) = series_sum) : a + b = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l1337_133750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_probability_l1337_133748

/-- The probability of cutting a 3-meter rope such that both resulting pieces
    are not less than 1 meter is 1/3. -/
theorem rope_cutting_probability : ∃ (probability : ℝ), probability = 1/3 := by
  let rope_length : ℝ := 3
  let min_piece_length : ℝ := 1
  let favorable_cut_length : ℝ := rope_length - 2 * min_piece_length
  let probability : ℝ := favorable_cut_length / rope_length
  
  have h : probability = 1/3 := by
    -- Proof steps would go here
    sorry
  
  exact ⟨probability, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_probability_l1337_133748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1337_133708

/-- Represents the nth term of a geometric sequence -/
noncomputable def geometric_term (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (q : ℝ) (h_q : q ≠ -1) (h_q_nonzero : q ≠ 0) (h_a₁_nonzero : a₁ ≠ 0) :
  (4 * (geometric_sum a₁ q 3) = geometric_term a₁ q 4 - 2) →
  (4 * (geometric_sum a₁ q 2) = 5 * (geometric_term a₁ q 2) - 2) →
  q = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1337_133708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_correct_l1337_133760

-- Define the functions
noncomputable def f (x : ℝ) := Real.sin (2 * x - 1)
noncomputable def g (x : ℝ) := x * Real.log x

-- State the theorem
theorem derivatives_correct :
  (∀ x, deriv f x = 2 * Real.cos (2 * x - 1)) ∧
  (∀ x, x > 0 → deriv g x = Real.log x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_correct_l1337_133760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1337_133753

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides 20 cm and 16 cm, 
    and height 15 cm, is 270 square centimeters -/
theorem trapezium_area_example : trapezium_area 20 16 15 = 270 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [add_mul, mul_div_right_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1337_133753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_value_l1337_133772

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Point G relative to triangle ABC -/
structure PointG (t : Triangle) where
  BG : ℝ
  BA_plus_BC : ℝ

/-- Main theorem -/
theorem triangle_sin_A_value (t : Triangle) (g : PointG t) :
  t.a^2 + t.c^2 - t.b^2 = t.a * t.c →
  t.c = 2 →
  g.BG = Real.sqrt 19 / 3 →
  g.BG = (1 / 3) * g.BA_plus_BC →
  Real.sin t.A = (3 * Real.sqrt 21) / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_value_l1337_133772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1337_133722

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallelLine : Line → Line → Prop)
variable (contained : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)

-- Non-coincidence properties
variable (nonCoincidentLines : Line → Line → Prop)
variable (nonCoincidentPlanes : Plane → Plane → Prop)

theorem geometry_theorem 
  (a b : Line) (α β : Plane)
  (h_noncoincident_lines : nonCoincidentLines a b)
  (h_noncoincident_planes : nonCoincidentPlanes α β) :
  (∀ a α β, perpendicular a α → perpendicular a β → parallel α β) ∧
  (∀ a b β, perpendicular a β → perpendicular b β → parallelLine a b) ∧
  (∀ a b β, perpendicular b β → contained a β → perpendicularLines a b) ∧
  ¬(∀ a b β, parallelToPlane a β → contained b β → parallelLine a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1337_133722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_i_l1337_133799

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression
noncomputable def expression : ℂ := ((1 + i) / (1 - i)) ^ 2009

-- Theorem statement
theorem expression_equals_i : expression = i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_i_l1337_133799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_satisfying_ratio_condition_l1337_133777

theorem smallest_k_satisfying_ratio_condition : ∃ k : ℕ+, 
  (k = 7) ∧ 
  (∀ A : Finset ℕ, A ⊆ Finset.range 26 → A.card = k → 
    ∃ x y, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 : ℚ)/3 ≤ (x : ℚ)/y ∧ (x : ℚ)/y ≤ 3/2) ∧
  (∀ k' < k, ∃ A : Finset ℕ, A ⊆ Finset.range 26 ∧ A.card = k' ∧
    ∀ x y, x ∈ A → y ∈ A → x ≠ y → (x : ℚ)/y < 2/3 ∨ 3/2 < (x : ℚ)/y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_satisfying_ratio_condition_l1337_133777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1337_133700

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points D, E, and F
variable (D E F : ℝ × ℝ)

-- D is the midpoint of BC
def is_midpoint (D B C : ℝ × ℝ) : Prop :=
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- E is on AC such that AE:EC = 2:3
def on_line_with_ratio (E A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = 2/5 ∧ E = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))

-- F is on AD such that AF:FD = 2:1
def on_line_with_ratio2 (F A D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = 2/3 ∧ F = (A.1 + t * (D.1 - A.1), A.2 + t * (D.2 - A.2))

-- Function to calculate area of a triangle
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

-- Theorem statement
theorem area_of_triangle_ABC (A B C D E F : ℝ × ℝ)
  (h1 : is_midpoint D B C)
  (h2 : on_line_with_ratio E A C)
  (h3 : on_line_with_ratio2 F A D)
  (h4 : triangle_area D E F = 18) :
  triangle_area A B C = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1337_133700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1337_133796

noncomputable def line_equation (x y : ℝ) : Prop := y = -x + 1

noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m * (180 / Real.pi)

theorem slope_angle_of_line : 
  ∀ x y : ℝ, line_equation x y → slope_angle (-1) = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1337_133796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1337_133755

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | (p.1^2 / 16) + (p.2^2 / 9) = 1}
def N : Set (ℝ × ℝ) := {p | (p.1 / 4) + (p.2 / 3) = 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {p : ℝ × ℝ | p.1 ∈ Set.Icc (-4 : ℝ) 4 ∧ p.2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1337_133755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_collection_l1337_133769

theorem coin_collection (total_coins : ℕ) 
  (quarter_ratio dime_ratio : ℚ)
  (state_quarter_ratio : ℚ)
  (pennsylvania_quarter_ratio : ℚ)
  (roosevelt_dime_ratio : ℚ)
  (h_total : total_coins = 50)
  (h_quarter : quarter_ratio = 3/10)
  (h_dime : dime_ratio = 2/5)
  (h_state_quarter : state_quarter_ratio = 2/5)
  (h_pennsylvania : pennsylvania_quarter_ratio = 3/8)
  (h_roosevelt : roosevelt_dime_ratio = 3/4) :
  ∃ (quarters dimes nickels state_quarters pennsylvania_quarters roosevelt_dimes : ℕ),
    quarters = (total_coins : ℚ) * quarter_ratio ∧
    dimes = (total_coins : ℚ) * dime_ratio ∧
    nickels = total_coins - (quarters + dimes) ∧
    state_quarters = (quarters : ℚ) * state_quarter_ratio ∧
    pennsylvania_quarters = ⌊(state_quarters : ℚ) * pennsylvania_quarter_ratio⌋ ∧
    roosevelt_dimes = ⌊(dimes : ℚ) * roosevelt_dime_ratio⌋ ∧
    pennsylvania_quarters = 2 ∧
    roosevelt_dimes = 15 ∧
    nickels = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_collection_l1337_133769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_expression_range_l1337_133776

/-- Given a cubic equation t³ + 4t² + 4t + a = 0 with three distinct real roots,
    the expression A = x³ - 4y² - 4z² - 4y - 4z + 32 takes values in (14 22/27, 16) -/
theorem cubic_roots_expression_range (a : ℝ) (x y z : ℝ) :
  (∀ t : ℝ, t^3 + 4*t^2 + 4*t + a = 0 ↔ t = x ∨ t = y ∨ t = z) →
  x < y →
  y < z →
  let A := x^3 - 4*y^2 - 4*z^2 - 4*y - 4*z + 32
  (400/27 : ℝ) < A ∧ A < 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_expression_range_l1337_133776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1337_133712

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 2 * Real.sin x ^ 2 - 1/2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ y : ℝ, y ∈ Set.Icc 0 (3/2) ↔ ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1337_133712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_min_slopes_product_l1337_133709

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the point P on the parabola
def point_on_parabola (p : ℝ) (t : ℝ) : Prop := parabola p 3 t

-- Define the distance from P to focus F
def distance_to_focus (p : ℝ) (t : ℝ) : ℝ := 4

-- Theorem for the equation of the parabola
theorem parabola_equation (p : ℝ) (t : ℝ) 
  (h1 : p > 0) 
  (h2 : point_on_parabola p t) 
  (h3 : distance_to_focus p t = 4) : 
  ∀ x y : ℝ, parabola 2 x y ↔ y^2 = 4*x :=
by sorry

-- Define the line l passing through (4,0)
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

-- Define the point M
def point_M : ℝ × ℝ := (-4, 0)

-- Define the slopes product
noncomputable def slopes_product (m : ℝ) : ℝ := -1 / (m^2 + 4)

-- Theorem for the minimum value of k₁ · k₂
theorem min_slopes_product :
  ∃ min : ℝ, min = -1/4 ∧ ∀ m : ℝ, slopes_product m ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_min_slopes_product_l1337_133709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_monkeys_eating_time_l1337_133746

/-- The time taken for monkeys to eat peaches -/
def eating_time (num_monkeys : ℕ) (num_peaches : ℕ) : ℝ := sorry

/-- Assumption: 5 little monkeys eat 5 peaches in 2 minutes -/
axiom base_case : eating_time 5 5 = 2

/-- Theorem: The time taken for 15 little monkeys to eat 15 peaches
    is equal to the time taken for 5 little monkeys to eat 5 peaches -/
theorem fifteen_monkeys_eating_time :
  eating_time 15 15 = eating_time 5 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_monkeys_eating_time_l1337_133746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_subset_implies_integer_ratio_l1337_133734

-- Define the set A(x)
def A (x : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * x⌋}

-- State the theorem
theorem irrational_subset_implies_integer_ratio 
  (α : ℝ) (hα_irrational : Irrational α) (hα_gt_two : α > 2) :
  ∀ β : ℝ, β > 0 → A β ⊆ A α → ∃ k : ℤ, β = k * α :=
by
  sorry

-- Additional lemma that might be useful
lemma floor_eq_implies_eq (x y : ℝ) :
  (∀ n : ℕ, ⌊n * x⌋ = ⌊n * y⌋) → x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_subset_implies_integer_ratio_l1337_133734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_time_is_nine_hours_l1337_133781

/-- Represents the parking cost structure and average cost scenario -/
structure ParkingScenario where
  base_cost : ℚ  -- Cost for the first 2 hours
  hourly_rate : ℚ  -- Cost per hour after the first 2 hours
  average_cost : ℚ  -- Given average cost per hour

/-- Calculates the total cost for a given number of hours -/
def total_cost (p : ParkingScenario) (hours : ℚ) : ℚ :=
  if hours ≤ 2 then p.base_cost
  else p.base_cost + p.hourly_rate * (hours - 2)

/-- Theorem stating that for the given scenario, the total parking time is 9 hours -/
theorem parking_time_is_nine_hours (p : ParkingScenario)
    (h1 : p.base_cost = 12)
    (h2 : p.hourly_rate = 1.75)
    (h3 : p.average_cost = 2.6944444444444446) :
    ∃ (hours : ℚ), hours = 9 ∧ p.average_cost = (total_cost p hours) / hours := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_time_is_nine_hours_l1337_133781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_imply_a_range_l1337_133720

noncomputable def f (a x : ℝ) : ℝ := (2 - x) * Real.exp x - a * x - a

theorem f_solutions_imply_a_range :
  ∀ a : ℝ,
    (∃! (n : ℕ), n = 2 ∧ ∀ k : ℕ, k > 0 → (f a k > 0 ↔ k ≤ n)) →
    a ∈ Set.Icc (-(Real.exp 3) / 4) 0 ∧ a ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_imply_a_range_l1337_133720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1337_133792

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1337_133792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_coordinates_l1337_133743

-- Define the points A and B
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (-3, 4)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the vector OC
def OC : ℝ × ℝ → Prop
  | (x, y) => x^2 + y^2 = 4 -- |OC| = 2

-- Define the angle bisector condition
def on_angle_bisector (C : ℝ × ℝ) : Prop :=
  let (x, y) := C
  x^2 + y^2 = (x + 3)^2 + (y - 4)^2 -- |OC| = |BC|

-- The main theorem
theorem OC_coordinates :
  ∃ (C : ℝ × ℝ), OC C ∧ on_angle_bisector C ∧ C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_coordinates_l1337_133743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_intersecting_circles_l1337_133747

/-- The area of the shaded region formed by four intersecting circles of radius 5 units at the origin -/
noncomputable def shaded_area : ℝ := 50 * Real.pi - 100

/-- The radius of each circle -/
def circle_radius : ℝ := 5

/-- Theorem stating that the shaded area equals 8 times the difference between
    a quarter circle's area and an isosceles right triangle's area -/
theorem shaded_area_of_intersecting_circles :
  shaded_area = 8 * (π * circle_radius^2 / 4 - circle_radius^2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_intersecting_circles_l1337_133747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bars_is_180_l1337_133795

/- Define the total number of bars -/
def total_bars : ℕ := 180

/- Define the number of each type of bar -/
def snickers : ℕ := (60 * total_bars) / 100
def mars : ℕ := (30 * total_bars) / 100
def bounty : ℕ := (10 * total_bars) / 100

/- Define the number of eaten bars for each type -/
def eaten_bounty : ℕ := bounty / 3
def eaten_mars : ℕ := (5 * eaten_bounty) / 6
def eaten_snickers : ℕ := (10 * eaten_bounty) / 3

/- Define the remaining Snickers bars -/
def remaining_snickers : ℕ := snickers - eaten_snickers

/- State the theorem -/
theorem total_bars_is_180 : 
  snickers + mars + bounty = total_bars ∧
  eaten_bounty = (120 * eaten_mars) / 100 ∧
  eaten_bounty = (30 * eaten_snickers) / 100 ∧
  remaining_snickers ≤ 150 ∧
  total_bars = 180 := by
  sorry

#eval total_bars

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bars_is_180_l1337_133795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_zeros_l1337_133727

/-- An odd function with specific zero properties -/
def OddFunctionWithZeros (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∃ a b, 0 < a ∧ a < b ∧ f 0 = 0 ∧ f a = 0 ∧ f b = 0) ∧
  (∀ x, 0 < x → f x = 0 → (∃ a b, 0 < a ∧ a < b ∧ x = a ∨ x = b))

/-- The number of zeros of f on ℝ is 3 -/
theorem odd_function_zeros (f : ℝ → ℝ) (h : OddFunctionWithZeros f) : 
  ∃ x y z, x < y ∧ y < z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0 ∧ 
  (∀ w, f w = 0 → w = x ∨ w = y ∨ w = z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_zeros_l1337_133727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l1337_133775

open Nat Finset

def T : Finset ℕ := range 100

def not_multiple_of_4_or_5 (n : ℕ) : Prop := n % 4 ≠ 0 ∧ n % 5 ≠ 0

theorem remaining_integers_count : (T.filter (λ n => n % 4 ≠ 0 ∧ n % 5 ≠ 0)).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l1337_133775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_count_l1337_133718

theorem divisibility_count : ∃ (n : ℕ), n = (Finset.filter (fun x => x % 3 = 0 ∨ x % 5 = 0 ∨ x % 7 = 0) (Finset.range 60)).card ∧ n = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_count_l1337_133718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_drivers_equals_808_l1337_133774

/-- The total number of drivers in community A -/
def drivers_in_A : ℕ := 96

/-- The number of drivers sampled from community A -/
def sampled_from_A : ℕ := 12

/-- The total number of drivers sampled from all communities -/
def total_sampled : ℕ := 12 + 21 + 25 + 43

/-- The sampling fraction -/
def sampling_fraction : ℚ := sampled_from_A / drivers_in_A

/-- The total number of drivers in all communities -/
def total_drivers : ℕ := (total_sampled * (1 / sampling_fraction).num).toNat

theorem total_drivers_equals_808 : total_drivers = 808 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_drivers_equals_808_l1337_133774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1337_133789

open Real

/-- Given a triangle ABC with angles A, B, C and sides a, b, c opposite to these angles respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define vectors m and n -/
noncomputable def m (t : Triangle) : ℝ × ℝ := (1 - cos (t.A + t.B), cos ((t.A - t.B) / 2))
noncomputable def n (t : Triangle) : ℝ × ℝ := (5/8, cos ((t.A - t.B) / 2))

/-- The dot product of m and n -/
noncomputable def dot_product (t : Triangle) : ℝ := (m t).1 * (n t).1 + (m t).2 * (n t).2

theorem triangle_theorem (t : Triangle) 
  (h : dot_product t = 9/8) : 
  (tan t.A * tan t.B = 1/9) ∧ 
  (∀ (s : Triangle), (s.a * s.b * sin s.C) / (s.a^2 + s.b^2 - s.c^2) ≤ -3/8) ∧
  (∃ (s : Triangle), (s.a * s.b * sin s.C) / (s.a^2 + s.b^2 - s.c^2) = -3/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1337_133789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1337_133737

noncomputable def f (n : ℕ+) : ℝ := Real.log n^2 / Real.log 1806

theorem f_sum_equals_two :
  f 17 + f 19 + f 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1337_133737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_workers_l1337_133725

/-- Represents the number of workers on the small field -/
def n : ℕ := sorry

/-- Represents the area of the small field -/
def S : ℝ := sorry

/-- Represents the productivity of each worker -/
def a : ℝ := sorry

/-- The area of the large field is twice the area of the small field -/
axiom large_field_area : 2 * S = S + S

/-- There are 6 more workers on the large field than on the small field -/
axiom worker_difference : (n : ℝ) + 6 = (n + 6 : ℕ)

/-- The time taken to complete the small field is greater than the time taken for the large field -/
axiom work_time_inequality : S / (a * n) > (2 * S) / (a * ((n : ℝ) + 6))

/-- The maximum number of workers in the team is 16 -/
theorem max_workers : n + (n + 6) ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_workers_l1337_133725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniqueTriangleConstruction_l1337_133782

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : Point :=
  sorry

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Checks if a point is symmetric to another point with respect to a line -/
def isSymmetric (p1 p2 : Point) (line : Point → Point → Prop) : Prop :=
  sorry

/-- Main theorem -/
theorem uniqueTriangleConstruction
  (A' B' C' : Point) :
  (∃ (t : Triangle), isAcuteAngled t ∧
    isSymmetric A' (orthocenter t) (λ p q ↦ p = t.B ∨ p = t.C) ∧
    isSymmetric B' (orthocenter t) (λ p q ↦ p = t.A ∨ p = t.C) ∧
    isSymmetric C' (orthocenter t) (λ p q ↦ p = t.A ∨ p = t.B)) →
  ∃! (t : Triangle), isAcuteAngled t ∧
    isSymmetric A' (orthocenter t) (λ p q ↦ p = t.B ∨ p = t.C) ∧
    isSymmetric B' (orthocenter t) (λ p q ↦ p = t.A ∨ p = t.C) ∧
    isSymmetric C' (orthocenter t) (λ p q ↦ p = t.A ∨ p = t.B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniqueTriangleConstruction_l1337_133782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_time_is_one_hour_l1337_133714

/-- Represents a journey between two towns -/
structure Journey where
  totalDistance : ℝ
  firstPartFraction : ℝ
  lunchBreakTime : ℝ
  totalTime : ℝ

/-- Calculates the time taken for the first part of the journey -/
noncomputable def firstPartTime (j : Journey) : ℝ :=
  let remainingDistance := j.totalDistance * (1 - j.firstPartFraction)
  let drivingTime := j.totalTime - j.lunchBreakTime
  (j.totalDistance * j.firstPartFraction) * drivingTime / j.totalDistance

/-- Theorem stating the time taken for the first part of the journey is 1 hour -/
theorem first_part_time_is_one_hour (j : Journey) 
    (h1 : j.totalDistance = 200)
    (h2 : j.firstPartFraction = 1/4)
    (h3 : j.lunchBreakTime = 1)
    (h4 : j.totalTime = 5) :
  firstPartTime j = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_time_is_one_hour_l1337_133714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1337_133739

theorem problem_statement :
  (∀ (x θ : ℝ), x^2 - 2*x*Real.sin θ + 1 ≥ 0) ∧
  (∃ (α β : ℝ), Real.sin (α + β) > Real.sin α + Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1337_133739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_sequence_properties_l1337_133765

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem triangle_and_sequence_properties 
  (t : Triangle) 
  (h1 : t.a^2 - (t.b - t.c)^2 = (2 - Real.sqrt 3) * t.b * t.c)
  (h2 : Real.sin t.A * Real.sin t.B = (Real.cos (t.C / 2))^2)
  (a₁ : ℝ)
  (d : ℝ)
  (h3 : d ≠ 0)
  (h4 : a₁ * Real.cos (2 * t.B) = 1)
  (h5 : (arithmetic_sequence a₁ d 2)^2 = 
        (arithmetic_sequence a₁ d 1) * (arithmetic_sequence a₁ d 8)) :
  t.B = π / 6 ∧ 
  (fun (n : ℕ) => (n : ℝ) / (n + 1)) = 
  (fun (n : ℕ) => (Finset.range n).sum (fun k => 4 / ((arithmetic_sequence a₁ d k) * (arithmetic_sequence a₁ d (k + 1))))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_sequence_properties_l1337_133765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l1337_133735

theorem no_integer_solution (k : ℝ) (h : (4 : ℝ)^k = 5) :
  ¬ ∃ m : ℤ, (4 : ℝ)^(↑m * k + 2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l1337_133735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1337_133728

noncomputable def f (x θ : ℝ) : ℝ := Real.cos (2 * x + θ)

theorem f_range (θ : ℝ) (h_θ : |θ| ≤ π/2) (m : ℝ)
  (h_mono : ∀ x y, -3*π/8 ≤ x ∧ x < y ∧ y ≤ -π/6 → f x θ < f y θ)
  (h_bound : ∀ θ', |θ'| ≤ π/2 → f (π/8) θ' ≤ m) :
  m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1337_133728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_consecutive_integers_divisible_by_ten_l1337_133778

theorem product_of_five_consecutive_integers_divisible_by_ten (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 10 * k := by
  sorry

#check product_of_five_consecutive_integers_divisible_by_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_consecutive_integers_divisible_by_ten_l1337_133778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1337_133790

theorem triangle_side_length (D E F : ℝ × ℝ) :
  let angle_D := Real.arccos (((E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2)) / 
    (Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) * Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)))
  let DE := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let DF := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  angle_D = π / 4 ∧ DE = 100 ∧ DF = 50 * Real.sqrt 2 →
  Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 50 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1337_133790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_interval_l1337_133779

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the union of A and B
def AUnionB : Set ℝ := A ∪ B

-- Define the interval (-1, 3]
def IntervalMinusOneThree : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Theorem statement
theorem union_equals_interval : AUnionB = IntervalMinusOneThree := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_interval_l1337_133779
